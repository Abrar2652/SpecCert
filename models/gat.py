"""
GAT variant with the same forward signature as models/graphcnn.py's GIN.
Accepts a list of S2VGraph objects and returns per-graph logits.

Used for Figure 8 (architecture ablation).  Single-head attention (sufficient
for the ablation) implemented directly with sparse ops so we do not take a
hard dependency on torch_scatter / torch_sparse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """One single-head GAT layer (Velickovic et al. 2018)."""

    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.view(1, -1))
        nn.init.xavier_uniform_(self.a_dst.view(1, -1))

    def forward(self, X, edge_index, num_nodes):
        """X : [N, in_dim],  edge_index : [2, E]  (already self-looped)."""
        H = self.W(X)                         # [N, out]
        src, dst = edge_index[0], edge_index[1]
        # Unnormalised attention logits
        e = (H[src] * self.a_src).sum(-1) + (H[dst] * self.a_dst).sum(-1)
        e = F.leaky_relu(e, self.negative_slope)

        # softmax over incoming edges per node (dst)
        e = e - e.max()                       # numerical stability
        exp_e = torch.exp(e)
        denom = torch.zeros(num_nodes, device=X.device)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-12)

        # Weighted aggregation
        weighted = H[src] * alpha.unsqueeze(-1)
        out = torch.zeros(num_nodes, H.shape[1], device=X.device)
        out.index_add_(0, dst, weighted)
        return out


class GAT(nn.Module):
    """GAT with identical constructor signature to GraphCNN (drop-in)."""

    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, device):
        super().__init__()
        del num_mlp_layers, learn_eps, neighbor_pooling_type
        self.device = device
        self.num_layers = num_layers
        self.final_dropout = final_dropout
        self.graph_pooling_type = graph_pooling_type

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers - 1):
            in_d = input_dim if layer == 0 else hidden_dim
            self.gat_layers.append(GATLayer(in_d, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            in_d = input_dim if layer == 0 else hidden_dim
            self.linears_prediction.append(nn.Linear(in_d, output_dim))

    def _build_edges(self, batch_graph):
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        if edge_mat_list:
            edge_idx = torch.cat(edge_mat_list, 1)
        else:
            edge_idx = torch.zeros((2, 0), dtype=torch.long)
        n = start_idx[-1]
        self_loop = torch.arange(n, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        edge_idx = torch.cat([edge_idx, self_loop], 1).to(self.device)
        return edge_idx, n, start_idx

    def _graph_pool(self, batch_graph, start_idx):
        idx, elem = [], []
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
        edge_idx, n, start_idx = self._build_edges(batch_graph)
        graph_pool = self._graph_pool(batch_graph, start_idx)

        hidden_rep = [X]
        h = X
        for layer in range(self.num_layers - 1):
            h = self.gat_layers[layer](h, edge_idx, n)
            h = self.batch_norms[layer](h)
            h = F.elu(h)
            hidden_rep.append(h)

        score = 0
        for layer, h_l in enumerate(hidden_rep):
            pooled = torch.spmm(graph_pool, h_l)
            score = score + F.dropout(
                self.linears_prediction[layer](pooled),
                self.final_dropout, training=self.training)
        return score
