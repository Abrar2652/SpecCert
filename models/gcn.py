"""
GCN variant with the same forward signature as models/graphcnn.py's GIN.
Accepts a list of S2VGraph objects and returns per-graph logits.

Used for Figure 8 (architecture ablation).  The key design constraint is
that the forward signature is identical to GraphCNN — drop-in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """Symmetric-normalised GCN (Kipf & Welling 2017) with sum graph pooling.

    Mirrors GraphCNN's constructor signature so it can be swapped in for
    architecture-ablation experiments (Figure 8) with zero changes to the
    training loop.
    """

    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, device):
        super().__init__()
        del num_mlp_layers, learn_eps, neighbor_pooling_type  # unused (match GIN sig)
        self.device = device
        self.num_layers = num_layers
        self.final_dropout = final_dropout
        self.graph_pooling_type = graph_pooling_type

        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers - 1):
            in_d = input_dim if layer == 0 else hidden_dim
            self.linears.append(nn.Linear(in_d, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            in_d = input_dim if layer == 0 else hidden_dim
            self.linears_prediction.append(nn.Linear(in_d, output_dim))

    def _preprocess_adj(self, batch_graph):
        """Build the symmetric-normalised adjacency  D^-1/2 (A+I) D^-1/2."""
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        if edge_mat_list:
            Adj_idx = torch.cat(edge_mat_list, 1)
        else:
            Adj_idx = torch.zeros((2, 0), dtype=torch.long)
        n = start_idx[-1]

        # Self-loops
        self_loop = torch.arange(n, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        Adj_idx = torch.cat([Adj_idx, self_loop], 1)
        Adj_val = torch.ones(Adj_idx.shape[1])

        # Compute degree
        deg = torch.zeros(n)
        deg.index_add_(0, Adj_idx[0], Adj_val)
        d_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        # Normalise: A_hat[u,v] = d_inv_sqrt[u] * d_inv_sqrt[v]
        norm_val = d_inv_sqrt[Adj_idx[0]] * d_inv_sqrt[Adj_idx[1]]

        return (torch.sparse_coo_tensor(Adj_idx, norm_val, (n, n)).to(self.device),
                start_idx)

    def _preprocess_graphpool(self, batch_graph, start_idx):
        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                elem.extend([1. / max(len(graph.g), 1)] * len(graph.g))
            else:
                elem.extend([1] * len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1])])
        if not idx:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0), (len(batch_graph), start_idx[-1])).to(self.device)
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        return torch.sparse_coo_tensor(
            idx, elem, (len(batch_graph), start_idx[-1])).to(self.device)

    def forward(self, batch_graph):
        X = torch.cat([g.node_features for g in batch_graph], 0).float().to(self.device)
        Adj_norm, start_idx = self._preprocess_adj(batch_graph)
        graph_pool = self._preprocess_graphpool(batch_graph, start_idx)

        hidden_rep = [X]
        h = X
        for layer in range(self.num_layers - 1):
            h = torch.spmm(Adj_norm, h)
            h = self.linears[layer](h)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score = 0
        for layer, h_l in enumerate(hidden_rep):
            pooled = torch.spmm(graph_pool, h_l)
            score = score + F.dropout(
                self.linears_prediction[layer](pooled),
                self.final_dropout, training=self.training)
        return score
