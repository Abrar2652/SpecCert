"""
SpecCert Utility Module
=======================
Data loading and preprocessing identical to GNNCert for fair benchmarking.
"""

import networkx as nx
import numpy as np
import random
import torch
import os
from sklearn.model_selection import StratifiedKFold
import copy
import time


def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0


def load_dataset_from_TUDataset(dataset, degree_as_tag):
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx

    tu_name = dataset

    if dataset in ['Synthie', 'ENZYMES', 'Fingerprint', 'DD']:
        ds = TUDataset(root='./dataset', name=tu_name, use_node_attr=True)
    else:
        ds = TUDataset(root='./dataset', name=tu_name)

    graphs = []
    for graph in ds:
        g_nx = to_networkx(graph, to_undirected=True)
        g = S2VGraph(g_nx, graph.y.item())

        if graph.x is not None:
            g.node_features = graph.x
        else:
            # Fallback: use node degree as a 1-D feature
            import torch
            degrees = torch.tensor(
                [[g_nx.degree(n)] for n in range(len(g_nx))], dtype=torch.float)
            g.node_features = degrees

        if graph.edge_index is not None:
            g.edge_mat = graph.edge_index
        else:
            import torch
            g.edge_mat = torch.zeros((2, 0), dtype=torch.long)

        g.neighbors = [list(g.g.neighbors(node)) for node in range(len(g.g))]
        if degree_as_tag:
            g.node_tags = [g.g.degree(node) for node in range(len(g.g))]
        g.max_neighbor = max([len(n) for n in g.neighbors]) if g.neighbors else 0
        graphs.append(g)

    return graphs, ds.num_classes, None


def _resolve_text_dataset_file(dataset):
    dataset_dir = os.path.join("dataset", dataset)
    if not os.path.isdir(dataset_dir):
        return None

    # Standard TU text-file name used in this codebase.
    default_txt = os.path.join(dataset_dir, f"{dataset}.txt")
    if os.path.exists(default_txt):
        return default_txt

    # Some local copies may keep the same basename without extension.
    no_ext = os.path.join(dataset_dir, dataset)
    if os.path.exists(no_ext) and os.path.isfile(no_ext):
        return no_ext

    # Final fallback: case-insensitive match for "<dataset>.txt".
    target = f"{dataset}.txt".lower()
    for name in os.listdir(dataset_dir):
        if name.lower() == target:
            candidate = os.path.join(dataset_dir, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def load_data(dataset, degree_as_tag, data_file=None):
    g_list = []
    label_dict = {}
    feat_dict = {}

    if data_file is None:
        data_file = _resolve_text_dataset_file(dataset)
        if data_file is None:
            raise FileNotFoundError(
                f"Could not find local text dataset file for '{dataset}' "
                f"under dataset/{dataset}/."
            )

    with open(data_file, 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])
                if tmp > len(row):
                    node_features.append(attr)
                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            if n < 400:
                g_list.append(S2VGraph(g, l, node_tags))

    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list) if degree_list else 0
        g.label = label_dict[g.label]
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1) if edges else torch.LongTensor([[], []])

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    max_degree = max(tagset)
    tag2index = {i: i for i in range(max_degree + 1)}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tag2index))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    return g_list, len(label_dict), tag2index


def load_dblp_v1_from_raw(degree_as_tag):
    """Memory-safe loader for DBLP_v1 using TU raw files directly.

    DBLP_v1 contains a very large node-label space; the default PyG TUDataset
    processing can one-hot encode node labels into a huge feature matrix.
    This loader avoids that by constructing compact 1-D features per node.
    """
    raw_dir = os.path.join("dataset", "DBLP_v1", "raw")
    graph_indicator_path = os.path.join(raw_dir, "DBLP_v1_graph_indicator.txt")
    graph_labels_path = os.path.join(raw_dir, "DBLP_v1_graph_labels.txt")
    node_labels_path = os.path.join(raw_dir, "DBLP_v1_node_labels.txt")
    edges_path = os.path.join(raw_dir, "DBLP_v1_A.txt")

    required = [graph_indicator_path, graph_labels_path, node_labels_path, edges_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "DBLP_v1 raw files missing. Expected: " + ", ".join(missing)
        )

    with open(graph_indicator_path, "r") as f:
        graph_indicator = [int(line.strip()) for line in f if line.strip()]
    with open(graph_labels_path, "r") as f:
        graph_labels_raw = [int(line.strip()) for line in f if line.strip()]
    with open(node_labels_path, "r") as f:
        node_labels_raw = [int(line.strip()) for line in f if line.strip()]

    num_nodes = len(graph_indicator)
    num_graphs = max(graph_indicator) if graph_indicator else 0

    if len(graph_labels_raw) != num_graphs:
        raise ValueError(
            f"DBLP_v1 graph-label mismatch: got {len(graph_labels_raw)} labels "
            f"for {num_graphs} graphs."
        )
    if len(node_labels_raw) != num_nodes:
        raise ValueError(
            f"DBLP_v1 node-label mismatch: got {len(node_labels_raw)} labels "
            f"for {num_nodes} nodes."
        )

    # Map dataset labels to contiguous 0..C-1.
    label_dict = {}
    mapped_graph_labels = []
    for lab in graph_labels_raw:
        if lab not in label_dict:
            label_dict[lab] = len(label_dict)
        mapped_graph_labels.append(label_dict[lab])

    # Global-node -> local-node mapping inside each graph.
    graph_node_counts = [0] * num_graphs
    local_node_id = [0] * (num_nodes + 1)  # 1-indexed with TU files.
    for nid, gid in enumerate(graph_indicator, start=1):
        gidx = gid - 1
        local_node_id[nid] = graph_node_counts[gidx]
        graph_node_counts[gidx] += 1

    graphs_nx = [nx.Graph() for _ in range(num_graphs)]
    per_graph_node_labels = [[0] * cnt for cnt in graph_node_counts]

    for gidx, cnt in enumerate(graph_node_counts):
        if cnt:
            graphs_nx[gidx].add_nodes_from(range(cnt))

    for nid in range(1, num_nodes + 1):
        gidx = graph_indicator[nid - 1] - 1
        per_graph_node_labels[gidx][local_node_id[nid]] = node_labels_raw[nid - 1]

    with open(edges_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.replace(",", " ").split()
            u = int(toks[0])
            v = int(toks[1])

            gu = graph_indicator[u - 1] - 1
            gv = graph_indicator[v - 1] - 1
            if gu != gv:
                continue

            lu = local_node_id[u]
            lv = local_node_id[v]
            if lu != lv:
                graphs_nx[gu].add_edge(lu, lv)

    max_node_label = max(node_labels_raw) if node_labels_raw else 1

    # Try to load 128-dim pretrained node embeddings from processed/dblp.npy.
    # File is (K, V, 128) where K slices are training epochs and V=28085 is
    # the vocabulary size.  Use the last complete slice as node-features.
    # Each graph-node's raw_label is mapped to a vocab-id via modulo.
    processed_npy = os.path.join("dataset", "DBLP_v1", "processed", "dblp.npy")
    node_embeddings = None
    if os.path.exists(processed_npy):
        try:
            import numpy as _np
            arr = _np.fromfile(processed_npy, dtype=_np.float32, offset=128)
            per_slice = 28085 * 128
            n_slices = len(arr) // per_slice
            if n_slices > 0:
                arr = arr[:n_slices * per_slice].reshape(n_slices, 28085, 128)
                node_embeddings = arr[-1]  # last complete slice (best-trained)
                print(f"[DBLP_v1] Using 128-dim pretrained embeddings "
                      f"from {processed_npy} (slice {n_slices-1}/{n_slices-1}, "
                      f"vocab=28085)")
        except Exception as exc:
            print(f"[DBLP_v1] WARNING: could not load {processed_npy}: {exc}")

    g_list = []
    for gidx, g_nx in enumerate(graphs_nx):
        g = S2VGraph(g_nx, mapped_graph_labels[gidx], per_graph_node_labels[gidx])
        g.neighbors = [list(g_nx.neighbors(node)) for node in range(len(g_nx))]
        g.max_neighbor = max((len(nbs) for nbs in g.neighbors), default=0)

        if degree_as_tag:
            node_vals = [len(nbs) for nbs in g.neighbors]
            g.node_tags = node_vals
            g.node_features = torch.tensor(node_vals, dtype=torch.float32).unsqueeze(1)
        elif node_embeddings is not None:
            node_vals = per_graph_node_labels[gidx]
            g.node_tags = node_vals
            # map raw labels (range: 0..~41324) into the 28085-vocab via modulo
            vocab_ids = [int(v) % 28085 for v in node_vals]
            import numpy as _np
            feats = _np.stack([node_embeddings[i] for i in vocab_ids], axis=0)
            g.node_features = torch.from_numpy(feats).float()
        else:
            node_vals = per_graph_node_labels[gidx]
            g.node_tags = node_vals
            denom = float(max_node_label) if max_node_label > 0 else 1.0
            g.node_features = (
                torch.tensor(node_vals, dtype=torch.float32).unsqueeze(1) / denom
            )

        edges = [list(pair) for pair in g_nx.edges()]
        edges.extend([[v, u] for u, v in edges])
        g.edge_mat = (
            torch.LongTensor(edges).transpose(0, 1)
            if edges else torch.LongTensor([[], []])
        )
        g_list.append(g)

    return g_list, len(label_dict), None


def load_dataset(dataset, degree_as_tag):
    # Datasets with local text files in dataset/<NAME>/<NAME>.txt
    TEXT_DATASETS = {
        "MUTAG", "PROTEINS", "NCI1", "REDDITBINARY",
        # kept for backwards-compat with any existing local files
        "COLLAB", "IMDBBINARY", "IMDBMULTI", "REDDITMULTI5K", "PTC",
    }
    if dataset == "DBLP_v1":
        graphs, num_classes, _ = load_dblp_v1_from_raw(degree_as_tag)
    elif dataset in TEXT_DATASETS:
        text_file = _resolve_text_dataset_file(dataset)
        if text_file is not None:
            graphs, num_classes, _ = load_data(dataset, degree_as_tag, text_file)
        else:
            graphs, num_classes, _ = load_dataset_from_TUDataset(dataset, degree_as_tag)
    else:
        # DBLP, DD, ENZYMES (and any other TUDataset) downloaded automatically
        graphs, num_classes, _ = load_dataset_from_TUDataset(dataset, degree_as_tag)
    return graphs, num_classes, _


def separate_data(graph_list, seed, fold_idx, n=3):
    assert 0 <= fold_idx and fold_idx < n, f"fold_idx must be from 0 to {n}."
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    return train_graph_list, test_graph_list, test_idx
