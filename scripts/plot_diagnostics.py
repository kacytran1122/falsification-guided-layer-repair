#!/usr/bin/env python3
"""Plot only values recovered from the checksummed A6000 pilot manifest."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "depth16_diagnostics_manifest_subset.json"
SUPP = ROOT / "results" / "figures"

ORDER = ["plain", "targeted", "uniform", "wrong_location", "mismatched"]
LABELS = {
    "plain": "Plain",
    "targeted": "Targeted PairNorm",
    "uniform": "Uniform PairNorm",
    "wrong_location": "Wrong-location PairNorm",
    "mismatched": "Mismatched residual",
}
COLORS = {
    "plain": "#4D4D4D",
    "targeted": "#0072B2",
    "uniform": "#009E73",
    "wrong_location": "#D55E00",
    "mismatched": "#CC79A7",
}
LINESTYLES = {
    "plain": (0, (1, 1)),
    "targeted": "-",
    "uniform": "--",
    "wrong_location": "-.",
    "mismatched": (0, (4, 1, 1, 1)),
}
MARKERS = {
    "plain": "o",
    "targeted": "s",
    "uniform": "^",
    "wrong_location": "D",
    "mismatched": "v",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.5,
            "legend.fontsize": 6.1,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load() -> dict:
    data = json.loads(DATA.read_text())
    assert data["environment"]["gpu"]["name"] == "NVIDIA RTX A6000"
    assert data["environment"]["device"] == "cuda:0"
    assert len(data["records"]) == 15
    for mode in ORDER:
        rows = [row for row in data["records"] if row["mode"] == mode]
        assert sorted(row["seed"] for row in rows) == [0, 1, 2]
    return data


def grouped(data: dict, mode: str, key: str) -> np.ndarray:
    rows = sorted(
        (row for row in data["records"] if row["mode"] == mode),
        key=lambda row: row["seed"],
    )
    return np.asarray([row[key] for row in rows], dtype=float)


def save(fig: plt.Figure, stem: str) -> None:
    SUPP.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUPP / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(SUPP / f"{stem}.png", dpi=360, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def diagnostic_trajectories(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.28), constrained_layout=True)
    specs = [
        ("energy", "Recorded energy", np.arange(1, 17)),
        ("effective_rank", "Effective rank", np.arange(1, 17)),
        ("local_gradient_transmission", "Local gradient transmission", np.arange(2, 17)),
    ]
    for ax, (key, ylabel, layers) in zip(axes, specs):
        for mode in ORDER:
            values = grouped(data, mode, key)
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            color = COLORS[mode]
            lower = mean - sd
            if key == "local_gradient_transmission":
                # The statistic is nonnegative; clip only the uncertainty band's
                # display boundary so a logarithmic axis can show small values.
                lower = np.maximum(lower, 1e-4)
            ax.fill_between(layers, lower, mean + sd, color=color, alpha=0.10, linewidth=0)
            ax.plot(
                layers,
                mean,
                color=color,
                linestyle=LINESTYLES[mode],
                marker=MARKERS[mode],
                markersize=2.2,
                markevery=2,
                label=LABELS[mode],
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 4, 8, 12, 16])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("(a) Representation energy")
    axes[1].set_title("(b) Rank trajectory")
    axes[2].set_title("(c) Backward transmission")
    axes[2].set_yscale("log")
    axes[2].axhline(0.5, color="#777777", linewidth=0.6, linestyle=":", zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, handlelength=2.4)
    fig.suptitle("Depth-16 A6000 diagnostic trajectories (mean and sample SD, three seeds)", y=1.03, fontsize=8.0)
    save(fig, "TNNLS_Fig4_Depth16DiagnosticTrajectories")


def execution_profile(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.15), constrained_layout=True)
    x = np.arange(len(ORDER), dtype=float)
    labels = ["Plain", "Targeted", "Uniform", "Wrong loc.", "Mismatch"]
    for ax, key, scale, ylabel, title in [
        (axes[0], "peak_allocated_bytes", 2**20, "Peak allocated CUDA memory (MiB)", "(a) PyTorch peak allocation"),
        (axes[1], "elapsed_seconds", 1.0, "Recorded end-to-end elapsed time (s)", "(b) Whole-run elapsed time"),
    ]:
        means = []
        sds = []
        for mode in ORDER:
            values = grouped(data, mode, key).reshape(-1) / scale
            means.append(values.mean())
            sds.append(values.std(ddof=1))
        for i, mode in enumerate(ORDER):
            ax.errorbar(
                x[i],
                means[i],
                yerr=sds[i],
                color=COLORS[mode],
                marker=MARKERS[mode],
                markersize=5.0,
                markeredgecolor="black",
                markeredgewidth=0.35,
                capsize=2.5,
                elinewidth=0.9,
            )
            ax.text(x[i], means[i], f" {means[i]:.1f}", fontsize=5.8, va="bottom", ha="left")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    fig.suptitle("Depth-16 A6000 execution profile (mean and sample SD, three seeds)", y=1.03, fontsize=8.0)
    save(fig, "TNNLS_Fig5_A6000ExecutionProfile")


def main() -> None:
    configure()
    data = load()
    diagnostic_trajectories(data)
    execution_profile(data)


if __name__ == "__main__":
    main()
