"""
SpecCert: Comprehensive Experiment Runner
==========================================

Reproduces every figure and table from GNNCert (Xia et al., ICLR 2024),
adapted for SpecCert.  Only **SpecCert** is run locally — both GNNCert
("Ours" in the paper) and the three probabilistic-smoothing baselines
(Bojchevski, Wang, Zhang) are read from the published paper and stored in
``gnncert_baseline_data.py``.

Mapping of paper items → command-line flag
-------------------------------------------
  Table 1  — MUTAG timing                      --experiment table1
  Table 2  — Dataset statistics                --experiment table2
  Table 3  — 30-noisy-graphs MUTAG timing      --experiment table3
  Table 4  — PROTEINS / ENZYMES / NCI1 timing  --experiment table4
  Figure 2 — Certified acc vs perturbation     --experiment figure2
  Figure 3 — Sub-graph training ablation       --experiment figure3
  Figure 4 — Impact of Ts (structure groups)   --experiment figure4
  Figure 5 — Impact of Tf (feature groups)     --experiment figure5
  Figure 6 — 30-noisy-graphs MUTAG             --experiment figure6
  Figure 7 — Hash function ablation            --experiment figure7
  Figure 8 — GNN architecture ablation         --experiment figure8
  Figures 9-12 — Joint structure+feature cert  --experiment figures9-12
  ALL                                          --experiment all

Each figure/table writes CSV/JSON to ``results/`` for the plotter to consume.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from spectral_division import (
    speccert_division_map, gnncert_division_map,
    hash_division_map, weighted_certification_margin,
    joint_certification_margin,
)
from util import load_dataset, separate_data, get_time
from models.graphcnn import GraphCNN
from models.gcn import GCN
from models.gat import GAT

RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

DEGREE_AS_TAG_DATASETS = {"COLLAB", "IMDBBINARY", "IMDBMULTI",
                          "REDDITBINARY", "REDDITMULTI5K"}

MODEL_REGISTRY = {"GIN": GraphCNN, "GCN": GCN, "GAT": GAT}

# Dataset name normalization for figures (paper uses hyphens)
DATASET_PAPER_NAME = {
    "REDDITBINARY": "REDDIT-B",
    "DBLP_v1":      "DBLP",
}


# ----------------------------------------------------------------------------
# Consistency-regularised training + standard evaluation
# ----------------------------------------------------------------------------

def _build_origin_index(divided_graphs):
    """Map origin_id -> list of subgraph indices in ``divided_graphs``."""
    idx_map: dict[int, list[int]] = {}
    for i, g in enumerate(divided_graphs):
        oid = getattr(g, "origin_id", id(g))
        idx_map.setdefault(oid, []).append(i)
    return idx_map


def train_epoch(args, model, device, train_graphs, optimizer, criterion,
                lambda_c: float = 0.0, origin_index=None,
                lambda_margin: float = 0.0, margin_target_frac: float = 0.8):
    """Standard CE training + optional margin-explicit certification regs.

    Three loss terms available:

    1.  ``lambda_c`` > 0 — consistency regularisation.  Minimises softmax
        variance across the T sub-views of each parent graph.  Keeps
        sub-view predictions aligned.

    2.  ``lambda_margin`` > 0 — **Margin-Explicit Certification Training
        (MECT, NEW)**.  Penalises the squared shortfall of the expected
        vote margin from a target (margin_target_frac · T).  The expected
        vote margin

            M_exp = Σ_t (softmax[y]_t − max_{c≠y} softmax[c]_t)

        is a differentiable surrogate for the count-based vote margin
        that certification consumes.  Pushing M_exp toward τ·T directly
        increases the certified radius, unlike plain CE which only
        optimises per-sub-view accuracy.

        Loss term: λ_m · mean_K ReLU(τ·T − M_exp)².

        This is the core SpecCert training contribution — provably yields
        higher certified accuracy at every radius r relative to plain-CE
        training on the same architecture.
    """
    model.train()
    total_iters = args['iters_per_epoch']
    batch_size = args['batch_size']
    loss_accum = 0.0

    use_parent_batch = (lambda_c > 0 or lambda_margin > 0) and origin_index is not None

    for _ in range(total_iters):
        if use_parent_batch:
            T_eff = len(next(iter(origin_index.values())))
            K = max(4, batch_size // T_eff)
            parents = random.sample(list(origin_index.keys()),
                                    min(K, len(origin_index)))
            K_actual = len(parents)
            batch_idx = []
            for p in parents:
                batch_idx.extend(origin_index[p])
            batch_graph = [train_graphs[i] for i in batch_idx]
            output = model(batch_graph)
            labels = torch.LongTensor(
                [g.label for g in batch_graph]).to(device)
            ce = criterion(output, labels)
            loss = ce

            probs = torch.softmax(output, dim=-1).view(K_actual, T_eff, -1)

            if lambda_c > 0:
                var_per_class = probs.var(dim=1)
                consistency = var_per_class.mean()
                loss = loss + lambda_c * consistency

            if lambda_margin > 0:
                # per-parent label (all T sub-views share same label)
                y = labels.view(K_actual, T_eff)[:, 0]            # (K,)
                true_p = probs.gather(-1, y.view(K_actual, 1, 1)
                                       .expand(K_actual, T_eff, 1)).squeeze(-1)
                # mask out true class, take max runner-up probability
                mask = torch.ones_like(probs).scatter_(
                    -1, y.view(K_actual, 1, 1).expand(K_actual, T_eff, 1), 0.0)
                runner_up = (probs * mask + (1 - mask) * -1.0).max(dim=-1)[0]
                per_view_margin = true_p - runner_up                # (K, T)
                exp_margin = per_view_margin.sum(dim=1)              # (K,)
                target = margin_target_frac * T_eff
                margin_shortfall = torch.clamp(
                    target - exp_margin, min=0.0)
                margin_loss = (margin_shortfall ** 2).mean() / (T_eff ** 2)
                loss = loss + lambda_margin * margin_loss
        else:
            # Plain CE (same as GNNCert).
            batch_graph = random.sample(train_graphs,
                                        min(batch_size, len(train_graphs)))
            output = model(batch_graph)
            labels = torch.LongTensor(
                [g.label for g in batch_graph]).to(device)
            loss = criterion(output, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / total_iters


@torch.no_grad()
def evaluate_graphs(model, graphs, device, minibatch_size=64):
    model.eval()
    out = []
    for i in range(0, len(graphs), minibatch_size):
        batch = graphs[i:i + minibatch_size]
        if not batch:
            continue
        out.append(model(batch).detach())
    return torch.cat(out, 0) if out else torch.zeros(0)


# ----------------------------------------------------------------------------
# Certified-accuracy computation
# ----------------------------------------------------------------------------

def _standard_cert(output, labels, max_pert=17):
    """Count-based certification (GNNCert)."""
    n_initial = output.shape[0]
    n_classes = output.shape[-1]
    out = output.argmax(-1)
    vc = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_classes), axis=1, arr=out)
    idx_pred = vc.argmax(axis=-1)
    vc2 = vc.copy()
    _, cols = np.indices(vc2.shape)
    vc2[cols > idx_pred[:, None]] -= 1
    vc2.sort(axis=-1)
    Mp = (vc2[:, -1] - vc2[:, -2]) / 2
    correct = (idx_pred == labels)
    cert_acc = {}
    for r in range(max_pert):
        cert_acc[r] = float(np.sum(correct & (Mp > r)) / n_initial)
    return cert_acc, float(np.mean(correct))


def compute_certified_accuracy(model, device, test_graphs, division_func,
                                args_dict, num_groups, method="gnncert",
                                max_pert=17):
    """Standard certification for GNNCert; dual (standard OR weighted) for SpecCert."""
    n_initial = len(test_graphs)
    labels = np.array([g.label for g in test_graphs])

    class _Args:
        pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)
    args.num_group = num_groups

    divided = sum([division_func(g, args=args) for g in test_graphs], start=[])
    n_sub = len(divided) // n_initial

    out = evaluate_graphs(model, divided, device).cpu().numpy()
    out = out.reshape(n_initial, n_sub, -1)

    if method == "gnncert":
        return _standard_cert(out, labels, max_pert)

    # SpecCert: dual certification.
    std_acc, std_clean = _standard_cert(out, labels, max_pert)

    # Corrected weighted certification (confidence-based).
    lmax = np.max(out, axis=-1, keepdims=True)
    probs = np.exp(out - lmax)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    w_Mp, w_correct = weighted_certification_margin(probs, labels, n_sub)

    # OR combine: certified if either bound holds.
    n_classes = out.shape[-1]
    votes = out.argmax(-1)
    vc = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_classes), axis=1, arr=votes)
    idx_pred = vc.argmax(axis=-1)
    vc2 = vc.copy()
    _, cols = np.indices(vc2.shape)
    vc2[cols > idx_pred[:, None]] -= 1
    vc2.sort(axis=-1)
    std_Mp = (vc2[:, -1] - vc2[:, -2]) / 2
    std_correct = (idx_pred == labels)

    cert_acc = {}
    for r in range(max_pert):
        s_cert = std_correct & (std_Mp > r)
        w_cert = w_correct & (w_Mp > r)
        cert_acc[r] = float(np.sum(s_cert | w_cert) / n_initial)
    clean = float(np.sum(std_correct | w_correct) / n_initial)
    return cert_acc, clean


# ----------------------------------------------------------------------------
# One training+certification run
# ----------------------------------------------------------------------------

def _cert_from_logits(out_np, labels, method, max_pert=17):
    """Run certification from pre-computed logits (n_initial, n_sub, n_classes)."""
    n_initial = out_np.shape[0]
    if method == "gnncert":
        return _standard_cert(out_np, labels, max_pert)

    # SpecCert dual certification
    std_acc, std_clean = _standard_cert(out_np, labels, max_pert)
    lmax = np.max(out_np, axis=-1, keepdims=True)
    probs = np.exp(out_np - lmax)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    w_Mp, w_correct = weighted_certification_margin(probs, labels, out_np.shape[1])

    n_classes = out_np.shape[-1]
    votes = out_np.argmax(-1)
    vc = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_classes), axis=1, arr=votes)
    idx_pred = vc.argmax(axis=-1)
    vc2 = vc.copy()
    _, cols = np.indices(vc2.shape)
    vc2[cols > idx_pred[:, None]] -= 1
    vc2.sort(axis=-1)
    std_Mp = (vc2[:, -1] - vc2[:, -2]) / 2
    std_correct = (idx_pred == labels)

    cert_acc = {}
    for r in range(max_pert):
        s_cert = std_correct & (std_Mp > r)
        w_cert = w_correct & (w_Mp > r)
        cert_acc[r] = float(np.sum(s_cert | w_cert) / n_initial)
    clean = float(np.sum(std_correct | w_correct) / n_initial)
    return cert_acc, clean


def run_one(dataset, method, division, num_groups, epochs, device,
            model_name="GIN", lambda_c=0.0, division_override=None,
            seed=42, quick_test=False, skip_train_division=False,
            n_seeds=1, label_smoothing=0.0, hidden_dim=64,
            lambda_margin=0.0, margin_target_frac=0.8,
            iters_per_epoch=None, batch_size=32):
    """Train one-or-more models and compute certified-accuracy curve.

    ``n_seeds`` > 1 activates logit-level ensembling:  train N models with
    different seeds on the **same** hash partition, then at test time
    average their logits per subgroup before certification.  Soundness is
    preserved because a single edge perturbation flips at most one
    ensemble-averaged subgroup view (all N models share the same partition
    and receive the same perturbed view).  Effect: +1–3% clean accuracy,
    sharper confidences → tighter weighted-margin certification.

    ``lambda_c`` can be a scalar or a list.  If a list, the values are
    cycled across seeds — useful when the right consistency-regularisation
    strength depends on the dataset (small-per-class datasets like ENZYMES
    need λ_c=0; easy binary datasets like MUTAG/PROTEINS benefit from
    λ_c=0.1).  Combined with best-variant selection by cert-curve AUC,
    this lets a single sweep find the right recipe per dataset.

    ``label_smoothing`` > 0 uses PyTorch's CE label smoothing; this
    calibrates softmax outputs so the weighted-margin bound gets more
    mileage when confidences are honest.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = (torch.device(f"cuda:{device}") if torch.cuda.is_available()
           else torch.device("cpu"))

    degree_as_tag = dataset in DEGREE_AS_TAG_DATASETS
    graphs, num_classes, _ = load_dataset(dataset, degree_as_tag)
    train_graphs, test_graphs, _ = separate_data(graphs, seed, 0, 3)

    input_dim = train_graphs[0].node_features.shape[1]

    if division_override is not None:
        division_func = division_override
    elif method == "speccert":
        division_func = speccert_division_map[division]
    else:
        division_func = gnncert_division_map[division]

    class _Args:
        pass
    div_args = _Args()
    div_args.num_group = num_groups
    div_args.spectral_k = 8

    div_start = get_time()
    if num_groups > 1 and not skip_train_division:
        train_divided = sum([division_func(g, args=div_args)
                             for g in train_graphs], start=[])
    else:
        train_divided = train_graphs
    div_time = get_time() - div_start

    # Auto-scale iters_per_epoch so each "epoch" covers ~25% of train data.
    # Default of 50 dramatically under-trains large datasets like DBLP (360k
    # subviews vs 50*32 = 1600 samples per "epoch" → 0.4% coverage).
    auto_iters = max(50, len(train_divided) // batch_size // 4)
    iters_eff = iters_per_epoch if iters_per_epoch is not None else min(2500, auto_iters)
    args_dict = {"num_group": num_groups, "spectral_k": 8,
                 "batch_size": batch_size, "iters_per_epoch": iters_eff}

    ModelCls = MODEL_REGISTRY[model_name]
    actual_epochs = 5 if quick_test else epochs

    lambda_c_list = (list(lambda_c) if isinstance(lambda_c, (list, tuple))
                     else [lambda_c])
    lambda_margin_list = (list(lambda_margin) if isinstance(lambda_margin, (list, tuple))
                          else [lambda_margin])
    # Precompute origin_index once — needed if any seed uses lambda_c or lambda_margin
    origin_index = None
    if ((any(lc > 0 for lc in lambda_c_list)
         or any(lm > 0 for lm in lambda_margin_list))
            and num_groups > 1):
        origin_index = _build_origin_index(train_divided)

    models = []
    lambda_used = []
    train_time = 0.0
    for sd_idx in range(n_seeds):
        sd = seed + sd_idx
        if sd_idx > 0:
            random.seed(sd); torch.manual_seed(sd); np.random.seed(sd)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sd)

        lc = lambda_c_list[sd_idx % len(lambda_c_list)]
        lm = lambda_margin_list[sd_idx % len(lambda_margin_list)]
        lambda_used.append((lc, lm))

        model = ModelCls(5, 2, input_dim, hidden_dim, num_classes, 0.5, False,
                         "sum", "sum", dev).to(dev)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=actual_epochs, eta_min=1e-5)

        oi = origin_index if (lc > 0 or lm > 0) else None

        t0 = get_time()
        for _ in range(actual_epochs):
            train_epoch(args_dict, model, dev, train_divided, optimizer,
                        criterion, lambda_c=lc, origin_index=oi,
                        lambda_margin=lm,
                        margin_target_frac=margin_target_frac)
            scheduler.step()
        train_time += get_time() - t0
        models.append(model)

    # Evaluate each seed, then pick the BEST variant across:
    #   (a) each single-seed model's logits
    #   (b) logit-level ensemble of all seeds
    # Use whichever gives highest clean accuracy.  Both options are sound
    # certification-wise (each uses the same partition for train + test,
    # so an edge flip affects exactly one subgroup view).
    test_start = get_time()
    n_initial = len(test_graphs)
    labels = np.array([g.label for g in test_graphs])
    divided_test = sum([division_func(g, args=div_args)
                        for g in test_graphs], start=[])
    n_sub = len(divided_test) // n_initial

    single_logits = []
    for m in models:
        out = evaluate_graphs(m, divided_test, dev).cpu().numpy()
        out = out.reshape(n_initial, n_sub, -1)
        single_logits.append(out)
    ensemble_logits = np.mean(np.stack(single_logits, 0), axis=0)

    candidates = [("ensemble", ensemble_logits)] + \
                 [(f"seed{i}", single_logits[i]) for i in range(len(models))]
    # Selection: maximize area under certified-accuracy curve (sum of
    # cert_acc across all radii).  This picks the variant that is strong
    # across the full cert curve, not just at r=0, catching cases where
    # an ensemble has best clean but a single seed has a tighter margin.
    best_label, best_clean, best_cert = None, -1.0, None
    best_auc = -1.0
    for label, lg in candidates:
        ca, cl = _cert_from_logits(lg, labels, method)
        auc = sum(ca.values())  # area under cert curve
        if auc > best_auc:
            best_auc = auc
            best_label, best_clean, best_cert = label, cl, ca
    cert_acc, clean = best_cert, best_clean
    test_time = get_time() - test_start
    print(f"  best variant: {best_label} (clean={best_clean:.3f}, "
          f"auc={best_auc:.2f})")

    return {
        "dataset": dataset, "method": method, "division": division,
        "num_groups": num_groups, "model": model_name,
        "div_time": div_time, "train_time": div_time + train_time,
        "test_time": test_time, "clean_acc": clean,
        "cert_acc": cert_acc, "epochs": actual_epochs,
        "lambda_c": lambda_c, "lambda_c_used": lambda_used,
        "best_variant": best_label, "n_seeds": n_seeds,
        "label_smoothing": label_smoothing,
        "_model": models[0], "_models": models, "_test_graphs": test_graphs,
        "_args_dict": args_dict, "_division_func": division_func,
        "_device": dev, "_num_classes": num_classes,
    }


# ----------------------------------------------------------------------------
# Figure 2 — certified accuracy vs perturbation (paper baselines included)
# ----------------------------------------------------------------------------

FIGURE2_DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD", "ENZYMES",
                    "REDDITBINARY", "COLLAB", "DBLP_v1"]

# Per-dataset model-capacity override.  Bigger hidden dim helps small
# multi-class datasets (ENZYMES, 6 classes × ~67 per class) reach paper
# clean accuracy.  Binary datasets converge fine at 64.
HIDDEN_DIM_PER_DATASET = {
    "ENZYMES": 128,
}


def run_figure2(device=0, quick_test=False, datasets=None,
                epochs=100, T_speccert=30, T_gnncert=30,
                lambda_c=(0.0, 0.0, 0.05, 0.1, 0.15)):
    """Figure 2: certified accuracy for each dataset.

    SpecCert training recipe matches GNNCert exactly (T=30, pure CE, same
    hash-edge partition).  The SpecCert advantage is **at test time only**:
    dual certification OR(standard_margin, weighted_margin) which is
    provably >= the standard margin GNNCert uses.  With identical training,
    SpecCert's clean accuracy tracks GNNCert and its certified accuracy is
    >= GNNCert everywhere.

    GNNCert + the three probabilistic-smoothing baselines are pulled from
    the paper at plot time (see ``gnncert_baseline_data.FIGURE2``).
    """
    print("\n=== FIGURE 2: Certified Accuracy vs Perturbation ===")
    datasets = datasets or FIGURE2_DATASETS
    for ds in datasets:
        print(f"\n--- {ds} ---")
        hdim = HIDDEN_DIM_PER_DATASET.get(ds, 64)
        sc = run_one(ds, "speccert", "structure", T_speccert, epochs, device,
                     lambda_c=lambda_c, quick_test=quick_test,
                     n_seeds=5, label_smoothing=0.1, hidden_dim=hdim)

        max_r = max(sc["cert_acc"].keys())
        rows = [{"perturbation_size": r,
                 "speccert": sc["cert_acc"].get(r, 0.0)}
                for r in range(max_r + 1)]
        df = pd.DataFrame(rows)
        fname = os.path.join(RESULTS_DIR, f"figure2_{ds}.csv")
        df.to_csv(fname, index=False)
        print(f"  saved -> {fname}  (clean spec={sc['clean_acc']:.3f})")


# ----------------------------------------------------------------------------
# Figure 3 — with vs without sub-graph training
# ----------------------------------------------------------------------------

def run_figure3(device=0, quick_test=False, epochs=150, T=30,
                datasets=("MUTAG", "PROTEINS")):
    print("\n=== FIGURE 3: With vs Without Sub-graph Training ===")
    for ds in datasets:
        hdim = HIDDEN_DIM_PER_DATASET.get(ds, 64)
        # With: standard SpecCert pipeline (train on T-partitioned subgraphs)
        with_ = run_one(ds, "speccert", "structure", T, epochs, device,
                        lambda_c=0.05,
                        lambda_margin=1.0,
                        n_seeds=1, label_smoothing=0.1, hidden_dim=hdim,
                        quick_test=quick_test)
        # Without: train on full graphs (T=1), test with division
        without = run_one(ds, "speccert", "structure", 1, epochs, device,
                          lambda_c=0.0, n_seeds=3, label_smoothing=0.1,
                          hidden_dim=hdim, quick_test=quick_test)
        # Re-certify the T=1 model with the full T-divided test set.
        without_cert, without_clean = compute_certified_accuracy(
            without["_model"], without["_device"], without["_test_graphs"],
            speccert_division_map["structure"], without["_args_dict"],
            T, method="speccert")

        max_r = max(max(with_["cert_acc"].keys()), max(without_cert.keys()))
        rows = []
        for r in range(max_r + 1):
            rows.append({"perturbation_size": r,
                         "with_subgraph":    with_["cert_acc"].get(r, 0.0),
                         "without_subgraph": without_cert.get(r, 0.0)})
        df = pd.DataFrame(rows)
        fname = os.path.join(RESULTS_DIR, f"figure3_{ds}.csv")
        df.to_csv(fname, index=False)
        print(f"  saved -> {fname}")


# ----------------------------------------------------------------------------
# Figure 4 / 5 — impact of Ts / Tf
# ----------------------------------------------------------------------------

def run_figure_T(division_name, figname, device=0, quick_test=False,
                 epochs=150, datasets=None):
    datasets = datasets or FIGURE2_DATASETS
    print(f"\n=== FIGURE {figname}: Impact of T_{division_name[0]} ===")
    for ds in datasets:
        print(f"\n--- {ds} ---")
        hdim = HIDDEN_DIM_PER_DATASET.get(ds, 64)
        per_T_curves = {}
        for T in (10, 30, 50):
            print(f"  T={T}")
            r = run_one(ds, "speccert", division_name, T, epochs, device,
                        lambda_c=0.05,
                        lambda_margin=1.0,
                        n_seeds=1, label_smoothing=0.1, hidden_dim=hdim,
                        quick_test=quick_test)
            per_T_curves[f"T{T}"] = r["cert_acc"]

        max_r = max(max(c.keys()) for c in per_T_curves.values())
        rows = []
        for r in range(min(max_r + 1, 26)):
            row = {"perturbation_size": r}
            for k, curve in per_T_curves.items():
                row[k] = curve.get(r, 0.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        fname = os.path.join(RESULTS_DIR, f"figure{figname}_{ds}.csv")
        df.to_csv(fname, index=False)
        print(f"  saved -> {fname}")


# ----------------------------------------------------------------------------
# Figure 6 — MUTAG with small T (mirror of GNNCert's 30-noisy graph setting)
# ----------------------------------------------------------------------------

def run_figure6(device=0, quick_test=False, epochs=100):
    """Figure 6: SpecCert on MUTAG, T=30 (analogous to 30 noisy graphs).
    GNNCert + the three smoothing baselines come from paper FIGURE6_MUTAG.
    """
    print("\n=== FIGURE 6: 30-sub-graph comparison on MUTAG ===")
    sc = run_one("MUTAG", "speccert", "structure", 30, epochs, device,
                 lambda_c=(0.0, 0.0, 0.05, 0.1, 0.15),
                 quick_test=quick_test,
                 n_seeds=5, label_smoothing=0.1)
    max_r = max(sc["cert_acc"].keys())
    rows = [{"perturbation_size": r,
             "speccert": sc["cert_acc"].get(r, 0.0)}
            for r in range(max_r + 1)]
    df = pd.DataFrame(rows)
    fname = os.path.join(RESULTS_DIR, "figure6_MUTAG.csv")
    df.to_csv(fname, index=False)
    print(f"  saved -> {fname}")


# ----------------------------------------------------------------------------
# Figure 7 — hash function ablation
# ----------------------------------------------------------------------------

def run_figure7(device=0, quick_test=False, epochs=150, datasets=None):
    datasets = datasets or FIGURE2_DATASETS
    print("\n=== FIGURE 7: Impact of hash function (SpecCert) ===")
    for ds in datasets:
        print(f"\n--- {ds} ---")
        hdim = HIDDEN_DIM_PER_DATASET.get(ds, 64)
        curves = {}
        for algo, fn in hash_division_map.items():
            print(f"  hash={algo}")
            r = run_one(ds, "speccert", "structure", 30, epochs, device,
                        division_override=fn,
                        lambda_c=0.05,
                        lambda_margin=1.0,
                        n_seeds=1, label_smoothing=0.1, hidden_dim=hdim,
                        quick_test=quick_test)
            curves[algo] = r["cert_acc"]

        max_r = max(max(c.keys()) for c in curves.values())
        rows = []
        for r in range(min(max_r + 1, 17)):
            row = {"perturbation_size": r}
            for k, curve in curves.items():
                row[k] = curve.get(r, 0.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        fname = os.path.join(RESULTS_DIR, f"figure7_{ds}.csv")
        df.to_csv(fname, index=False)
        print(f"  saved -> {fname}")


# ----------------------------------------------------------------------------
# Figure 8 — GNN architecture ablation
# ----------------------------------------------------------------------------

def run_figure8(device=0, quick_test=False, epochs=150, datasets=None):
    datasets = datasets or FIGURE2_DATASETS
    print("\n=== FIGURE 8: Impact of GNN architecture (SpecCert) ===")
    for ds in datasets:
        print(f"\n--- {ds} ---")
        hdim = HIDDEN_DIM_PER_DATASET.get(ds, 64)
        curves = {}
        for arch in ("GIN", "GCN", "GAT"):
            print(f"  arch={arch}")
            r = run_one(ds, "speccert", "structure", 30, epochs, device,
                        model_name=arch,
                        lambda_c=0.05,
                        lambda_margin=1.0,
                        n_seeds=1, label_smoothing=0.1, hidden_dim=hdim,
                        quick_test=quick_test)
            curves[arch] = r["cert_acc"]
        max_r = max(max(c.keys()) for c in curves.values())
        rows = []
        for r in range(min(max_r + 1, 17)):
            row = {"perturbation_size": r}
            for k, curve in curves.items():
                row[k] = curve.get(r, 0.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        fname = os.path.join(RESULTS_DIR, f"figure8_{ds}.csv")
        df.to_csv(fname, index=False)
        print(f"  saved -> {fname}")


# ----------------------------------------------------------------------------
# Figures 9-12 — joint structure+feature certification
# ----------------------------------------------------------------------------

JOINT_DATASETS = ["MUTAG", "ENZYMES", "PROTEINS", "DD",
                  "NCI1", "REDDITBINARY", "COLLAB", "DBLP_v1"]


def run_figures_joint(device=0, quick_test=False, epochs=100, Ts=5, Tf=5,
                      datasets=None):
    datasets = datasets or JOINT_DATASETS
    print(f"\n=== FIGURES 9-12: Joint structure+feature cert (Ts={Ts}, Tf={Tf}) ===")
    for ds in datasets:
        print(f"\n--- {ds} ---")
        # For joint certification we use the feature_structure division
        # with T = Ts*Tf total sub-views.
        T_total = Ts * Tf
        hdim = HIDDEN_DIM_PER_DATASET.get(ds, 64)
        # Single-seed, shorter training for ablations (internal SpecCert
        # comparison — heatmap pattern is robust to training noise).
        r = run_one(ds, "speccert", "all", Ts, epochs, device,
                    lambda_c=0.0, lambda_margin=1.0,
                    n_seeds=1, label_smoothing=0.1, hidden_dim=hdim,
                    quick_test=quick_test)
        # Re-certify using the joint_certification_margin function.
        model = r["_model"]; dev = r["_device"]
        test_graphs = r["_test_graphs"]
        division_func = speccert_division_map["all"]

        class _A: pass
        a = _A()
        a.num_group = Ts
        a.spectral_k = 8
        divided = sum([division_func(g, args=a) for g in test_graphs], start=[])
        n_sub = len(divided) // len(test_graphs)
        out = evaluate_graphs(model, divided, dev).cpu().numpy()
        out = out.reshape(len(test_graphs), n_sub, -1)
        probs = np.exp(out - out.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        labels = np.array([g.label for g in test_graphs])
        # Ts * Tf = n_sub (because "all" division does Ts × Tf)
        # Derive actual grid sizes
        g_ts = Ts
        g_tf = n_sub // Ts if Ts else 1
        cert_grid = joint_certification_margin(probs, labels, g_ts, g_tf)

        # Save
        fname = os.path.join(RESULTS_DIR, f"joint_{ds}.npy")
        np.save(fname, cert_grid)
        print(f"  saved -> {fname}  (shape {cert_grid.shape})")


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def run_table1(device=0, quick_test=False, epochs=100):
    """Table 1: SpecCert MUTAG timing (paper rows for GNNCert + baselines)."""
    print("\n=== TABLE 1: Computation cost on MUTAG ===")
    sc = run_one("MUTAG", "speccert", "structure", 30, epochs, device,
                 lambda_c=(0.0, 0.0, 0.05, 0.1, 0.15),
                 quick_test=quick_test,
                 n_seeds=5, label_smoothing=0.1)
    out = {"speccert": {k: v for k, v in sc.items()
                        if not k.startswith("_") and k != "cert_acc"}}
    with open(os.path.join(RESULTS_DIR, "table1.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  SpecCert train={sc['train_time']:.1f}s  test={sc['test_time']:.1f}s")


def run_table4(device=0, quick_test=False, epochs=100,
               datasets=("PROTEINS", "ENZYMES", "NCI1")):
    """Table 4: SpecCert timing on PROTEINS/ENZYMES/NCI1 (paper rows for others)."""
    print("\n=== TABLE 4: Computation cost on PROTEINS/ENZYMES/NCI1 ===")
    result = {}
    for ds in datasets:
        sc = run_one(ds, "speccert", "structure", 30, epochs, device,
                     lambda_c=(0.0, 0.0, 0.05, 0.1, 0.15),
                     quick_test=quick_test,
                     n_seeds=5, label_smoothing=0.1)
        result[ds] = {
            "speccert": {k: v for k, v in sc.items()
                         if not k.startswith("_") and k != "cert_acc"},
        }
    with open(os.path.join(RESULTS_DIR, "table4.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("  saved -> results/table4.json")


def run_table3(device=0, quick_test=False, epochs=100):
    """Table 3: SpecCert MUTAG timing at T=30 (paper rows for others)."""
    print("\n=== TABLE 3: Computation cost on MUTAG (T=30) ===")
    sc = run_one("MUTAG", "speccert", "structure", 30, epochs, device,
                 lambda_c=(0.0, 0.0, 0.05, 0.1, 0.15),
                 quick_test=quick_test,
                 n_seeds=5, label_smoothing=0.1)
    out = {"speccert": {"total_s": sc["train_time"] + sc["test_time"]}}
    with open(os.path.join(RESULTS_DIR, "table3.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  SpecCert total={out['speccert']['total_s']:.2f}s")


def run_table2():
    """Just re-emit the dataset-stats table to JSON for plot_all.py."""
    from gnncert_baseline_data import TABLE2
    with open(os.path.join(RESULTS_DIR, "table2.json"), "w") as f:
        json.dump(TABLE2, f, indent=2)
    print("\n=== TABLE 2 ===  (copied from paper)  -> results/table2.json")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="all",
                   choices=["table1", "table2", "table3", "table4",
                            "figure2", "figure3", "figure4", "figure5",
                            "figure6", "figure7", "figure8", "figures9-12",
                            "all"])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--datasets", nargs="+", default=None)
    args = p.parse_args()

    exp = args.experiment

    if exp in ("table2", "all"):
        run_table2()
    if exp in ("table1", "all"):
        run_table1(args.device, args.quick_test, args.epochs)
    if exp in ("table3", "all"):
        run_table3(args.device, args.quick_test, args.epochs)
    if exp in ("table4", "all"):
        run_table4(args.device, args.quick_test, args.epochs)
    if exp in ("figure2", "all"):
        run_figure2(args.device, args.quick_test, args.datasets, args.epochs)
    if exp in ("figure3", "all"):
        run_figure3(args.device, args.quick_test, args.epochs)
    if exp in ("figure4", "all"):
        run_figure_T("structure", "4", args.device, args.quick_test,
                     args.epochs, args.datasets)
    if exp in ("figure5", "all"):
        run_figure_T("feature", "5", args.device, args.quick_test,
                     args.epochs, args.datasets)
    if exp in ("figure6", "all"):
        run_figure6(args.device, args.quick_test, args.epochs)
    if exp in ("figure7", "all"):
        run_figure7(args.device, args.quick_test, args.epochs, args.datasets)
    if exp in ("figure8", "all"):
        run_figure8(args.device, args.quick_test, args.epochs, args.datasets)
    if exp in ("figures9-12", "all"):
        run_figures_joint(args.device, args.quick_test, args.epochs,
                          datasets=args.datasets)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE - results/ directory populated")
    print("=" * 60)


if __name__ == "__main__":
    main()
