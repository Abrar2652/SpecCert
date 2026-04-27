"""
Dual-certification ablation.

For each dataset, trains a single SpecCert model (1 seed, no MECT) and then
decomposes the certified accuracy at test time into three curves:
  - "standard"  : GNNCert-style majority-vote margin
  - "weighted"  : confidence-weighted margin alone
  - "or"        : OR(standard, weighted)  — what SpecCert reports

Saves results/dualcert_ablation_{dataset}.csv with 4 columns.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from run_all_experiments import run_one, RESULTS_DIR, evaluate_graphs, _standard_cert
from spectral_division import weighted_certification_margin, speccert_division_map

DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD", "ENZYMES",
            "REDDITBINARY", "COLLAB"]


def compute_three(model, dev, test_graphs, division_func, args_dict,
                  num_groups, max_pert=17):
    """Return (std_curve, weighted_curve, or_curve) as dicts {r: cert_acc}."""
    n_initial = len(test_graphs)
    labels = np.array([g.label for g in test_graphs])

    class _A:
        pass
    a = _A(); a.num_group = num_groups; a.spectral_k = 8
    divided = sum([division_func(g, args=a) for g in test_graphs], start=[])
    n_sub = len(divided) // n_initial

    out = evaluate_graphs(model, divided, dev).cpu().numpy()
    out = out.reshape(n_initial, n_sub, -1)

    # Standard margin curve
    std_curve, _ = _standard_cert(out, labels, max_pert)

    # Weighted margin curve
    lmax = np.max(out, axis=-1, keepdims=True)
    probs = np.exp(out - lmax)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    w_Mp, w_correct = weighted_certification_margin(probs, labels, n_sub)

    weighted_curve = {}
    for r in range(max_pert):
        weighted_curve[r] = float(np.sum(w_correct & (w_Mp > r)) / n_initial)

    # OR-combined (what SpecCert reports)
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

    or_curve = {}
    for r in range(max_pert):
        s = std_correct & (std_Mp > r)
        w = w_correct & (w_Mp > r)
        or_curve[r] = float(np.sum(s | w) / n_initial)

    return std_curve, weighted_curve, or_curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--datasets", nargs="+", default=None)
    args = p.parse_args()

    datasets = args.datasets or DATASETS
    for ds in datasets:
        print(f"\n=== Dual-cert ablation: {ds} ===", flush=True)
        r = run_one(ds, "speccert", "structure", 30, args.epochs, args.device,
                    lambda_c=0.0, lambda_margin=0.0,
                    n_seeds=1, label_smoothing=0.1, hidden_dim=64)
        model = r["_model"]; dev = r["_device"]
        test_graphs = r["_test_graphs"]
        args_dict = r["_args_dict"]
        division_func = r["_division_func"]

        std_c, w_c, or_c = compute_three(model, dev, test_graphs,
                                          division_func, args_dict, 30)

        max_r = 17
        rows = []
        for rr in range(max_r):
            rows.append({"perturbation_size": rr,
                         "standard": std_c.get(rr, 0.0),
                         "weighted": w_c.get(rr, 0.0),
                         "or_dual":  or_c.get(rr, 0.0)})
        out = os.path.join(RESULTS_DIR, f"dualcert_ablation_{ds}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  saved -> {out}", flush=True)
        print(f"  r=14 std={std_c.get(14,0):.3f}  "
              f"weighted={w_c.get(14,0):.3f}  or={or_c.get(14,0):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
