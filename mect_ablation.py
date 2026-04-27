"""
MECT (Margin-Explicit Certification Training) ablation.

For each of 4 datasets, trains SpecCert with lambda_margin ∈ {0, 0.5, 1.0, 2.0}
(keeping all other hyperparameters fixed) and saves certified-accuracy curves
to results/mect_ablation_{dataset}.csv.  plot_all.py -> plot_figure_mect()
consumes these CSVs.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from run_all_experiments import run_one, RESULTS_DIR

DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
LAMBDA_M_VALUES = [0.0, 0.5, 1.0, 2.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--datasets", nargs="+", default=None)
    args = p.parse_args()

    datasets = args.datasets or DATASETS
    for ds in datasets:
        print(f"\n=== MECT ablation: {ds} ===", flush=True)
        curves = {}
        for lm in LAMBDA_M_VALUES:
            print(f"  λ_m = {lm}", flush=True)
            r = run_one(ds, "speccert", "structure", 30, args.epochs, args.device,
                        lambda_c=0.0, lambda_margin=lm,
                        n_seeds=1, label_smoothing=0.1, hidden_dim=64)
            curves[f"lm_{lm}"] = r["cert_acc"]
            print(f"    clean={r['clean_acc']:.3f}  "
                  f"cert@7={r['cert_acc'].get(7,0):.3f}  "
                  f"cert@14={r['cert_acc'].get(14,0):.3f}", flush=True)

        max_r = max(max(c.keys()) for c in curves.values())
        rows = []
        for rr in range(min(max_r + 1, 17)):
            row = {"perturbation_size": rr}
            for k, c in curves.items():
                row[k] = c.get(rr, 0.0)
            rows.append(row)
        out = os.path.join(RESULTS_DIR, f"mect_ablation_{ds}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
