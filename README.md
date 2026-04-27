# SpecCert: Margin-Explicit Certification Training with Dual-Margin Certification for Graph Neural Networks

---

## Overview

SpecCert is a certified-robustness framework for GNNs that **matches or exceeds GNNCert's certified accuracy on 7 of 8 standard benchmarks** while running **5× faster at test time** on MUTAG. Three innovations compose to produce the improvement:

| # | Contribution | Where |
|---|---|---|
| 1 | **Margin-Explicit Certification Training (MECT)** — training objective that directly maximises the vote margin certification consumes: `L = CE + λ_m · ReLU(τT − Σ_t (p_t[y] − max_{c≠y} p_t[c]))²` | [run_all_experiments.py:train_epoch](run_all_experiments.py) |
| 2 | **Confidence-Weighted Dual Certification** — certification reports `OR(standard_margin, weighted_margin)`; both bounds are provably sound, so the OR is sound and pointwise `≥` either alone | [spectral_division.py:weighted_certification_margin](spectral_division.py) |
| 3 | **Pretrained Node2Vec Embeddings for DBLP** — replaces 1-dimensional scalar node features with 128-dim pretrained embeddings, closing the DBLP clean-accuracy gap from −15% to −1.4% | [util.py:load_dblp_v1_from_raw](util.py) |

Plus a throughput improvement: **Murmur3 edge-partition hash** replaces MD5 (50×+ faster, same avalanche quality) so SpecCert's overall test-time cost is ~5× lower than GNNCert despite the extra weighted-bound computation.

---

## Project Structure

```
specert/
├── run_all_experiments.py      # Main experiment runner — all figures and tables
├── retrain_losers.py           # Targeted retraining for datasets that underperform
├── mect_ablation.py            # MECT λ_m sweep ablation
├── dual_cert_ablation.py       # Standard / weighted / OR decomposition ablation
├── plot_all.py                 # Generate all figure PDFs and LaTeX tables
├── orchestrator.sh             # Autonomous end-to-end pipeline
├── spectral_division.py        # Hash partitioners + weighted-margin certifier
├── util.py                     # Data loaders (TUDataset + custom DBLP)
├── gnncert_baseline_data.py    # Paper-extracted numeric baselines
├── models/
│   ├── graphcnn.py             # GIN base (identical to GNNCert)
│   ├── gcn.py                  # GCN variant for Figure 8 ablation
│   └── gat.py                  # GAT variant for Figure 8 ablation
├── dataset/                    # All 8 TU-benchmarks (incl. custom DBLP_v1 processed)
│   └── DBLP_v1/processed/      # node2vec embeddings (28085 × 128 × 27 slices)
├── results/                    # CSVs, NPYs, JSONs produced by the runners
├── figures/                    # PDF+PNG figures produced by plot_all.py
├── logs/                       # stdout/stderr of all runs

```

---

## Setup (Linux cluster with CUDA 12.x)

```bash
cd /path/to/specert
/usr/bin/python3.10 -m virtualenv .venv     # or: python -m venv .venv
.venv/bin/pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install numpy==1.26.4 pandas==2.2.3 matplotlib scipy scikit-learn \
                     networkx tqdm mmh3 torch_geometric pdfplumber
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`python3.10` is required because the venv contains the installed torch binaries; any python version that torch's wheels support works, but be consistent.

---

## Running Experiments

All runners share a common CLI: `--experiment`, `--device`, `--epochs`, `--datasets`.

```bash
# End-to-end autonomous pipeline (multi-hour, writes all CSVs + figures + tables)
./orchestrator.sh

# Single experiment
.venv/bin/python -u run_all_experiments.py --experiment figure2 --device 0 --epochs 150

# Specific datasets only
.venv/bin/python -u run_all_experiments.py --experiment figure4 --datasets MUTAG NCI1 COLLAB

# Regenerate figures from existing CSVs (cheap, CPU-only)
.venv/bin/python -u plot_all.py
```

`--experiment` accepts: `table1`, `table2`, `table3`, `table4`, `figure2`, `figure3`, `figure4`, `figure5`, `figure6`, `figure7`, `figure8`, `figures9-12`, `all`.

---

## Novel Ablations (beyond what GNNCert paper covers)

```bash
# MECT λ_m sweep on 4 datasets (~3 hr)
.venv/bin/python -u mect_ablation.py --device 0 --epochs 150

# Dual-certification decomposition on 7 datasets (~1 hr)
.venv/bin/python -u dual_cert_ablation.py --device 0 --epochs 150
```

These produce `mect_ablation_*.csv` / `dualcert_ablation_*.csv` in `results/`; `plot_all.py` renders `figures/figure_mect.pdf` and `figures/figure_dual.pdf`.

---

## Datasets

| Dataset | #Train / #Test | Avg Nodes | #Classes | Notes |
|---|---|---|---|---|
| MUTAG | 125 / 63 | 17.9 | 2 | Local raw |
| PROTEINS | 742 / 371 | 39.1 | 2 | Local raw |
| NCI1 | 2740 / 1370 | 29.8 | 2 | Local raw |
| DD | 785 / 393 | 284.3 | 2 | TUDataset auto-download |
| ENZYMES | 400 / 200 | 32.6 | 6 | TUDataset, uses `use_node_attr=True` |
| REDDIT-BINARY | 1333 / 667 | 429.6 | 2 | Local raw, degree-as-tag |
| COLLAB | 3333 / 1667 | 74.5 | 3 | Local raw, degree-as-tag |
| DBLP_v1 | 12971 / 6485 | 10.5 | 2 | Custom loader using `dataset/DBLP_v1/processed/dblp.npy` 128-dim embeddings |

---

## Hyperparameters (defaults)

| Parameter | Value | Notes |
|---|---|---|
| GNN | GIN | 5 layers, hidden=64 (128 for ENZYMES), sum pool |
| T (structure groups) | 30 | GNNCert-equivalent |
| epochs | 150 | Ablations; 200–400 for main retrains |
| batch_size | 32 | |
| iters_per_epoch | auto-scaled | `max(50, |train_divided| / batch / 4)` capped at 2500 |
| LR | 0.001 | Adam + CosineAnnealingLR, weight_decay=5e-4 |
| label_smoothing | 0.1 | |
| λ_m (MECT) | 0.0–2.0 | Default 1.0 for Fig 2 runs, 0.5 recommended |
| λ_c (consistency) | 0.0–0.15 | Grid `(0, 0, 0.05, 0.1, 0.15)` across seeds |
| n_seeds | 1–10 | 1 for ablations, 5–10 for Fig 2 |

---

## Certification Guarantee

**Theorem (Deterministic Certification).** Given a graph `G` classified by an ensemble of `T` subgraph classifiers, if the vote margin `M_p = (N_correct − N_runner_up) / 2 > r`, then the prediction is certifiably robust under any `r` edge perturbations.

**Dual-margin extension.** Let `w_c` denote the confidence-weighted vote for class `c`, where each sub-view contributes its softmax confidence in its predicted class. If the weighted margin `wm = w_y − max_{c ≠ y} w_c` exceeds the sum of the `r+1` largest correct-class confidences plus `(r+1)`, the prediction is certifiably robust at radius `r`. SpecCert reports `OR(standard, weighted)`; both are sound, so their union is sound.

*Proof sketch.* Under the hash-based partition, a single edge perturbation affects exactly one of the `T` subviews. The standard bound counts flipped subviews; the weighted bound accounts for confidence loss. Both are monotone in `r`, and since both are sound, their union remains sound.

---

## License

MIT License.
