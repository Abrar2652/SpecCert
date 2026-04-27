"""
plot_all.py — Generate every figure and table in NeurIPS style.

Styling rules (per user spec):
    * No chart titles.
    * Eye-catching but print-friendly colors.
    * Every line marker has edgecolor='black'.
    * No overlapping legends / axis labels.

Reads:
    results/figure{N}_{DATASET}.csv  — local SpecCert runs only
    gnncert_baseline_data.py         — paper-extracted curves
                                        (Bojchevski, Wang, Zhang, GNNCert)

Writes:
    figures/figure{N}.pdf  (and optional PNG mirrors)
    tables/table{N}.tex
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gnncert_baseline_data import (
    FIGURE2, FIGURE6_MUTAG, TABLE1, TABLE2, TABLE3, TABLE4, METHOD_STYLE,
)

RESULTS = os.path.join(_HERE, "results")
FIGDIR = os.path.join(_HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
TABDIR = os.path.join(_HERE, "tables");  os.makedirs(TABDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# NeurIPS matplotlib rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Nimbus Roman No9 L",
                           "Liberation Serif", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "black",
    "grid.alpha":         0.3,
    "grid.linestyle":     ":",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,        # TrueType for camera-ready
    "ps.fonttype":        42,
})


def _plot_line(ax, x, y, style_key):
    """Plot a single method's line with the canonical style."""
    label, color, marker, ls, ms, lw = METHOD_STYLE[style_key]
    ax.plot(x, y, color=color, marker=marker, linestyle=ls,
            markersize=ms, linewidth=lw, label=label,
            markeredgecolor="black", markeredgewidth=0.6, clip_on=True)


def _axis_style(ax, xlabel, ylabel, xlim=(0, 16), ylim=(0, 1.02),
                xticks=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, which="both")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Figure 2 — certified accuracy, 8 datasets, 4 methods + SpecCert
# ---------------------------------------------------------------------------

def _read_local(filename, cols):
    """Read a result CSV if it exists, return {col: list[float]}."""
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = df[c].tolist()
    return out


FIG2_DATASETS = [
    ("DBLP",     "DBLP_v1",       "(a) DBLP"),
    ("DD",       "DD",            "(b) DD"),
    ("ENZYMES",  "ENZYMES",       "(c) ENZYMES"),
    ("MUTAG",    "MUTAG",         "(d) MUTAG"),
    ("NCI1",     "NCI1",          "(e) NCI1"),
    ("PROTEINS", "PROTEINS",      "(f) PROTEINS"),
    ("REDDIT-B", "REDDITBINARY",  "(g) REDDIT-B"),
    ("COLLAB",   "COLLAB",        "(h) COLLAB"),
]


def plot_figure2():
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    axes = axes.ravel()
    x = list(range(17))

    for idx, (paper_name, local_name, caption) in enumerate(FIG2_DATASETS):
        ax = axes[idx]

        # Paper curves: Bojchevski / Wang / Zhang / GNNCert
        base = FIGURE2.get(paper_name, {})
        for k in ("bojchevski", "wang", "zhang", "gnncert"):
            if k in base:
                _plot_line(ax, x, base[k], k)

        # Local SpecCert run
        local = _read_local(f"figure2_{local_name}.csv",
                            ["perturbation_size", "speccert"])
        if local is not None and "speccert" in local:
            lx = local.get("perturbation_size", x)
            _plot_line(ax, lx, local["speccert"], "speccert")

        _axis_style(ax, "Perturbation Size", "Certified Accuracy",
                    xticks=list(range(0, 17, 2)))
        ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=9)

        # Legend only on first subplot
        if idx == 0:
            ax.legend(loc="lower left", ncol=1)

    plt.tight_layout(h_pad=2.4, w_pad=1.0)
    out = os.path.join(FIGDIR, "figure2.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 3 — with vs without sub-graph training (4 datasets from paper)
# ---------------------------------------------------------------------------

FIG3_DATASETS = [("DBLP_v1", "(a) DBLP"), ("DD", "(b) DD"),
                 ("ENZYMES", "(c) ENZYMES"), ("MUTAG", "(d) MUTAG"),
                 ("NCI1", "(e) NCI1"), ("PROTEINS", "(f) PROTEINS"),
                 ("REDDITBINARY", "(g) REDDIT-B"), ("COLLAB", "(h) COLLAB")]


def plot_figure3():
    available = [(d, c) for d, c in FIG3_DATASETS
                 if os.path.exists(os.path.join(RESULTS, f"figure3_{d}.csv"))]
    if not available:
        print("  figure3: no CSVs found — skipping")
        return
    n = len(available)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    axes = axes.ravel()
    for i, (ds, caption) in enumerate(available):
        ax = axes[i]
        df = pd.read_csv(os.path.join(RESULTS, f"figure3_{ds}.csv"))
        x = df["perturbation_size"].tolist()
        ax.plot(x, df["with_subgraph"],
                color="#1f77b4", linestyle="-", marker="o",
                markersize=5, linewidth=1.8, markeredgecolor="black",
                markeredgewidth=0.6, label="Training with sub-graph")
        ax.plot(x, df["without_subgraph"],
                color="#d62728", linestyle="--", marker="s",
                markersize=5, linewidth=1.8, markeredgecolor="black",
                markeredgewidth=0.6, label="Training without sub-graph")
        _axis_style(ax, "Perturbation Size", "Certified Accuracy",
                    xticks=list(range(0, 17, 2)))
        ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        if i == 0:
            ax.legend(loc="lower left")
    for j in range(len(available), len(axes)):
        axes[j].axis("off")
    plt.tight_layout(h_pad=2.4, w_pad=1.0)
    out = os.path.join(FIGDIR, "figure3.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 4 / 5 — T ablation
# ---------------------------------------------------------------------------

def _plot_T_ablation(prefix, fig_num):
    datasets = [(d, lbl) for d, lbl in FIG3_DATASETS
                if os.path.exists(os.path.join(RESULTS,
                                                f"figure{fig_num}_{d}.csv"))]
    if not datasets:
        print(f"  figure{fig_num}: no CSVs found — skipping")
        return
    cols = 4; rows = (len(datasets) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    axes = axes.ravel()
    palette = {"T10": "#2ca02c", "T30": "#ff7f0e", "T50": "#1f77b4"}
    markers = {"T10": "^", "T30": "s", "T50": "o"}
    styles = {"T10": "-.", "T30": "--", "T50": "-"}
    labels = {"T10": f"{prefix} = 10", "T30": f"{prefix} = 30",
              "T50": f"{prefix} = 50"}
    for i, (ds, caption) in enumerate(datasets):
        ax = axes[i]
        df = pd.read_csv(os.path.join(RESULTS, f"figure{fig_num}_{ds}.csv"))
        x = df["perturbation_size"].tolist()
        for k in ("T10", "T30", "T50"):
            if k in df.columns:
                ax.plot(x, df[k], color=palette[k], marker=markers[k],
                        linestyle=styles[k], markersize=5, linewidth=1.6,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=labels[k])
        _axis_style(ax, "Perturbation Size", "Certified Accuracy",
                    xlim=(0, 25), xticks=list(range(0, 26, 5)))
        ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        if i == 0:
            ax.legend(loc="lower left")
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    plt.tight_layout(h_pad=2.4, w_pad=1.0)
    out = os.path.join(FIGDIR, f"figure{fig_num}.pdf")
    plt.savefig(out); plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")


def plot_figure4():
    _plot_T_ablation("T_s", 4)


def plot_figure5():
    _plot_T_ablation("T_f", 5)


# ---------------------------------------------------------------------------
# Figure 6 — MUTAG, T=30 vs baselines with 30 noisy graphs
# ---------------------------------------------------------------------------

def plot_figure6():
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = list(range(17))
    for k, vals in FIGURE6_MUTAG.items():
        _plot_line(ax, x, vals, k)
    local = _read_local("figure6_MUTAG.csv",
                        ["perturbation_size", "speccert"])
    if local is not None and "speccert" in local:
        lx = local.get("perturbation_size", x)
        _plot_line(ax, lx, local["speccert"], "speccert")
    _axis_style(ax, "Perturbation Size", "Certified Accuracy",
                xticks=list(range(0, 17, 2)))
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "figure6.pdf")
    plt.savefig(out); plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 7 — hash-function ablation
# ---------------------------------------------------------------------------

def plot_figure7():
    datasets = [(d, lbl) for d, lbl in FIG3_DATASETS
                if os.path.exists(os.path.join(RESULTS, f"figure7_{d}.csv"))]
    if not datasets:
        print("  figure7: no CSVs found — skipping")
        return
    cols = 4; rows = (len(datasets) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    axes = axes.ravel()
    palette = {"murmur3": "#9467bd", "md5": "#1f77b4",
               "sha1": "#2ca02c",   "sha256": "#d62728"}
    markers = {"murmur3": "P", "md5": "o", "sha1": "^", "sha256": "s"}
    styles = {"murmur3": "-", "md5": "--", "sha1": "-.", "sha256": ":"}
    label_map = {"murmur3": "Murmur3 (Ours)", "md5": "MD5",
                 "sha1": "SHA1", "sha256": "SHA256"}
    for i, (ds, caption) in enumerate(datasets):
        ax = axes[i]
        df = pd.read_csv(os.path.join(RESULTS, f"figure7_{ds}.csv"))
        x = df["perturbation_size"].tolist()
        for k in palette:
            if k in df.columns:
                ax.plot(x, df[k], color=palette[k], marker=markers[k],
                        linestyle=styles[k], markersize=5, linewidth=1.6,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=label_map[k])
        _axis_style(ax, "Perturbation Size", "Certified Accuracy",
                    xticks=list(range(0, 17, 2)))
        ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        if i == 0:
            ax.legend(loc="lower left")
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    plt.tight_layout(h_pad=2.4, w_pad=1.0)
    out = os.path.join(FIGDIR, "figure7.pdf")
    plt.savefig(out); plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 8 — architecture ablation
# ---------------------------------------------------------------------------

def plot_figure8():
    datasets = [(d, lbl) for d, lbl in FIG3_DATASETS
                if os.path.exists(os.path.join(RESULTS, f"figure8_{d}.csv"))]
    if not datasets:
        print("  figure8: no CSVs found — skipping")
        return
    cols = 4; rows = (len(datasets) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    axes = axes.ravel()
    palette = {"GIN": "#1f77b4", "GCN": "#2ca02c", "GAT": "#d62728"}
    markers = {"GIN": "o",        "GCN": "^",      "GAT": "s"}
    styles = {"GIN": "-",         "GCN": "--",      "GAT": "-."}
    for i, (ds, caption) in enumerate(datasets):
        ax = axes[i]
        df = pd.read_csv(os.path.join(RESULTS, f"figure8_{ds}.csv"))
        x = df["perturbation_size"].tolist()
        for k in palette:
            if k in df.columns:
                ax.plot(x, df[k], color=palette[k], marker=markers[k],
                        linestyle=styles[k], markersize=5, linewidth=1.6,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=k)
        _axis_style(ax, "Perturbation Size", "Certified Accuracy",
                    xticks=list(range(0, 17, 2)))
        ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        if i == 0:
            ax.legend(loc="lower left")
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    plt.tight_layout(h_pad=2.4, w_pad=1.0)
    out = os.path.join(FIGDIR, "figure8.pdf")
    plt.savefig(out); plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")


# ---------------------------------------------------------------------------
# Figures 9-12 — joint heatmaps (2 datasets per figure, 4 figures total)
# ---------------------------------------------------------------------------

def plot_joint_heatmaps():
    """Single Figure 9 showing joint structure+feature cert for 7 datasets.

    DBLP_v1 is omitted (capacity-limited in our pipeline; Figure 2 already
    reports its cert curve).  The 7 datasets are laid out as 2 rows x 4
    columns, with one column in the first row left empty so the grid still
    reads cleanly (or we can squeeze to 2x4 with one blank cell).
    """
    datasets = ["MUTAG", "ENZYMES", "PROTEINS", "DD",
                "NCI1", "REDDITBINARY", "COLLAB"]
    labels = {"REDDITBINARY": "REDDIT-B"}

    n = len(datasets)
    ncols = 4
    nrows = (n + ncols - 1) // ncols   # = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        path = os.path.join(RESULTS, f"joint_{ds}.npy")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"missing\n{ds}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        grid = np.load(path)
        im = ax.imshow(grid, origin="lower", cmap="viridis",
                       aspect="auto", vmin=0, vmax=max(grid.max(), 1e-6))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Feature Perturbation Size", fontsize=9)
        ax.set_ylabel("Structure Perturbation Size", fontsize=9)
        nice = labels.get(ds, ds)
        ax.text(0.5, -0.32, f"({'abcdefgh'[idx]}) {nice}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)

    # Hide unused cells (e.g. 8th slot when we have 7 datasets).
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(h_pad=2.4, w_pad=1.5)
    out = os.path.join(FIGDIR, "figure9.pdf")
    plt.savefig(out); plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")

    # Remove any stale figure10/11/12 files from a previous run so the user
    # isn't confused by old PDFs.
    for stale in ("figure10", "figure11", "figure12"):
        for ext in (".pdf", ".png"):
            p = os.path.join(FIGDIR, stale + ext)
            if os.path.exists(p):
                os.remove(p)
                print(f"  removed stale {p}")


# ---------------------------------------------------------------------------
# MECT ablation — lambda_margin sweep
# ---------------------------------------------------------------------------

def plot_figure_mect():
    """MECT ablation: lambda_margin sweep on 4 datasets."""
    ds_list = [("MUTAG", "(a) MUTAG"), ("PROTEINS", "(b) PROTEINS"),
               ("NCI1", "(c) NCI1"), ("DD", "(d) DD")]
    available = [(d, c) for d, c in ds_list
                 if os.path.exists(os.path.join(RESULTS,
                                                 f"mect_ablation_{d}.csv"))]
    if not available:
        print("  figure_mect: no CSVs found — skipping")
        return
    cols = len(available); rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2),
                             squeeze=False)
    axes = axes.ravel()
    palette = {"lm_0.0": "#808080", "lm_0.5": "#2ca02c",
               "lm_1.0": "#1f77b4", "lm_2.0": "#9467bd"}
    markers = {"lm_0.0": "^", "lm_0.5": "s",
               "lm_1.0": "o", "lm_2.0": "P"}
    styles = {"lm_0.0": "--", "lm_0.5": "-.",
              "lm_1.0": "-", "lm_2.0": "-"}
    label_map = {"lm_0.0": r"$\lambda_m = 0$ (plain CE)",
                 "lm_0.5": r"$\lambda_m = 0.5$",
                 "lm_1.0": r"$\lambda_m = 1.0$",
                 "lm_2.0": r"$\lambda_m = 2.0$"}

    for i, (ds, caption) in enumerate(available):
        ax = axes[i]
        df = pd.read_csv(os.path.join(RESULTS, f"mect_ablation_{ds}.csv"))
        x = df["perturbation_size"].tolist()
        for k in palette:
            if k in df.columns:
                ax.plot(x, df[k], color=palette[k], marker=markers[k],
                        linestyle=styles[k], markersize=5, linewidth=1.6,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=label_map[k])
        _axis_style(ax, "Perturbation Size", "Certified Accuracy",
                    xticks=list(range(0, 17, 2)))
        ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        if i == 0:
            ax.legend(loc="lower left", fontsize=7)
    plt.tight_layout(h_pad=2.4, w_pad=1.0)
    out = os.path.join(FIGDIR, "figure_mect.pdf")
    plt.savefig(out); plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")


# ---------------------------------------------------------------------------
# Dual-certification ablation — std vs weighted vs OR
# ---------------------------------------------------------------------------

def plot_figure_dualcert():
    ds_list = [("MUTAG", "(a) MUTAG"), ("PROTEINS", "(b) PROTEINS"),
               ("NCI1", "(c) NCI1"), ("DD", "(d) DD"),
               ("ENZYMES", "(e) ENZYMES"), ("REDDITBINARY", "(f) REDDIT-B"),
               ("COLLAB", "(g) COLLAB")]
    available = [(d, c) for d, c in ds_list
                 if os.path.exists(os.path.join(RESULTS,
                                                 f"dualcert_ablation_{d}.csv"))]
    if not available:
        print("  figure_dual: no CSVs found — skipping")
        return
    cols = min(4, len(available))
    rows = (len(available) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    axes = axes.ravel()
    palette = {"standard": "#1f77b4", "weighted": "#d62728",
               "or_dual":  "#9467bd"}
    markers = {"standard": "o", "weighted": "s", "or_dual": "P"}
    styles = {"standard": "--", "weighted": ":", "or_dual": "-"}
    label_map = {"standard": "standard margin (GNNCert)",
                 "weighted": "weighted margin (new)",
                 "or_dual":  "OR (SpecCert)"}

    for i, (ds, caption) in enumerate(available):
        ax = axes[i]
        df = pd.read_csv(os.path.join(RESULTS, f"dualcert_ablation_{ds}.csv"))
        x = df["perturbation_size"].tolist()
        for k in palette:
            if k in df.columns:
                ax.plot(x, df[k], color=palette[k], marker=markers[k],
                        linestyle=styles[k], markersize=5, linewidth=1.6,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=label_map[k])
        _axis_style(ax, "Perturbation Size", "Certified Accuracy",
                    xticks=list(range(0, 17, 2)))
        ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        if i == 0:
            ax.legend(loc="lower left", fontsize=7)
    for j in range(len(available), len(axes)):
        axes[j].axis("off")
    plt.tight_layout(h_pad=2.4, w_pad=1.0)
    out = os.path.join(FIGDIR, "figure_dual.pdf")
    plt.savefig(out); plt.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved -> {out}")


# ---------------------------------------------------------------------------
# Tables — LaTeX output
# ---------------------------------------------------------------------------

def _load_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def emit_tables():
    # Table 1 - computation cost on MUTAG
    t1 = _load_json("table1.json")
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Method & Training (s) & Testing (s) \\", r"\midrule"]
    for method, (tr, te) in TABLE1.items():
        lines.append(f"{method} & {tr} & {te} \\\\")
    if t1 is not None:
        lines.append(f"SpecCert (Ours) & {t1['speccert']['train_time']:.0f} "
                     f"& {t1['speccert']['test_time']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABDIR, "table1.tex"), "w") as f:
        f.write("\n".join(lines))

    # Table 2 - dataset stats
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Dataset & \#Train & \#Test & \#Classes \\", r"\midrule"]
    for ds, (tr, te, nc) in TABLE2.items():
        lines.append(f"{ds} & {tr} & {te} & {nc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABDIR, "table2.tex"), "w") as f:
        f.write("\n".join(lines))

    # Table 3 - 30 noisy graphs timing
    t3 = _load_json("table3.json")
    lines = [r"\begin{tabular}{lr}", r"\toprule",
             r"Method & Total (s) \\", r"\midrule"]
    for m, s in TABLE3.items():
        lines.append(f"{m} & {s:.2f} \\\\")
    if t3 is not None:
        lines.append(f"SpecCert (Ours) & {t3['speccert']['total_s']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABDIR, "table3.tex"), "w") as f:
        f.write("\n".join(lines))

    # Table 4 - PROTEINS / ENZYMES / NCI1 timing
    t4 = _load_json("table4.json")
    ds_list = ["PROTEINS", "ENZYMES", "NCI1"]
    lines = [r"\begin{tabular}{l" + "rr" * len(ds_list) + "}",
             r"\toprule",
             r"Method" + "".join(f" & \\multicolumn{{2}}{{c}}{{{d}}}"
                                  for d in ds_list) + r" \\",
             r"".join(f" & Train & Test" for _ in ds_list) + r" \\",
             r"\midrule"]
    # Paper baselines
    for method in TABLE4["PROTEINS"]:
        row = [method]
        for ds in ds_list:
            tr, te = TABLE4[ds][method]
            row += [str(tr), str(te)]
        lines.append(" & ".join(row) + r" \\")
    # Local SpecCert
    if t4 is not None:
        row = ["SpecCert (Ours)"]
        for ds in ds_list:
            v = t4.get(ds, {}).get("speccert")
            if v:
                row += [f"{v['train_time']:.0f}", f"{v['test_time']:.1f}"]
            else:
                row += ["--", "--"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABDIR, "table4.tex"), "w") as f:
        f.write("\n".join(lines))

    print("  saved -> tables/table{1,2,3,4}.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== PLOT FIGURE 2 ===")
    plot_figure2()
    print("\n=== PLOT FIGURE 3 ===")
    plot_figure3()
    print("\n=== PLOT FIGURE 4 ===")
    plot_figure4()
    print("\n=== PLOT FIGURE 5 ===")
    plot_figure5()
    print("\n=== PLOT FIGURE 6 ===")
    plot_figure6()
    print("\n=== PLOT FIGURE 7 ===")
    plot_figure7()
    print("\n=== PLOT FIGURE 8 ===")
    plot_figure8()
    print("\n=== PLOT FIGURES 9-12 ===")
    plot_joint_heatmaps()
    print("\n=== PLOT FIGURE MECT ===")
    plot_figure_mect()
    print("\n=== PLOT FIGURE DUAL-CERT ===")
    plot_figure_dualcert()
    print("\n=== EMIT TABLES ===")
    emit_tables()


if __name__ == "__main__":
    main()
