#!/usr/bin/env python3
"""Generate only figures supported by the supplied compact pilot log.

The CSV contains final validation/test accuracy, end-to-end elapsed time, and
peak CUDA allocated bytes. It does not contain per-epoch loss, accuracy,
memory, or timing, so this script intentionally cannot draw learning curves.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "pilot_compact_evidence.csv"
OUT = ROOT / "results" / "figures"

# Okabe-Ito palette. Line styles, markers, and hatching preserve meaning in B/W.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
YELLOW = "#E69F00"
GRAY = "#666666"
BLACK = "#000000"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "seed": int(raw["seed"]),
                    "depth": int(raw["depth"]),
                    "test_accuracy": float(raw["test_accuracy"]),
                    "best_val_accuracy": float(raw["best_val_accuracy"]),
                    "elapsed_seconds": float(raw["elapsed_seconds"]),
                    "peak_allocated_bytes": int(raw["peak_allocated_bytes"]),
                }
            )
    if len(rows) != 27:
        raise ValueError(f"expected 27 supplied records, found {len(rows)}")
    return rows


def sample_mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    plt.close(fig)


def accuracy_depth_band(rows: list[dict[str, object]]) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["mode"] == "plain":
            grouped[int(row["depth"])].append(float(row["test_accuracy"]))
    depths = sorted(grouped)
    means, sds = zip(*(sample_mean_sd(grouped[depth]) for depth in depths))
    x = list(range(len(depths)))

    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    lower = [mean - sd for mean, sd in zip(means, sds)]
    upper = [mean + sd for mean, sd in zip(means, sds)]
    ax.fill_between(x, lower, upper, color=BLUE, alpha=0.20, linewidth=0)
    ax.plot(x, means, color=BLUE, marker="o", linestyle="-", label="Mean test accuracy")
    for xi, mean in zip(x, means):
        ax.annotate(f"{mean:.3f}", (xi, mean), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.2)
    ax.set_xticks(x, [str(depth) for depth in depths])
    ax.set_xlabel("Message-passing depth")
    ax.set_ylabel("Final test accuracy")
    ax.set_ylim(0.0, 0.76)
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.4, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handlelength=2.4)
    save(fig, "TNNLS_Fig_FinalAccuracyDepthBand")


def depth16_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    labels = {
        "plain": "Plain",
        "targeted": "Targeted",
        "uniform": "Uniform",
        "wrong_location": "Wrong loc.",
        "mismatched": "Mismatch",
    }
    grouped: dict[str, list[dict[str, object]]] = {label: [] for label in labels.values()}
    for row in rows:
        if int(row["depth"]) == 16 and str(row["mode"]) in labels:
            grouped[labels[str(row["mode"])]].append(row)
    return grouped


def peak_memory(rows: list[dict[str, object]]) -> None:
    grouped = depth16_rows(rows)
    labels = list(grouped)
    values = [sample_mean_sd([float(row["peak_allocated_bytes"]) / 1e6 for row in grouped[label]])[0] for label in labels]
    colors = [GRAY, BLUE, GREEN, ORANGE, PURPLE]
    hatches = ["//", "", "..", "xx", "\\\\"]

    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor=BLACK, linewidth=0.55)
    for bar, hatch, value in zip(bars, hatches, values):
        bar.set_hatch(hatch)
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=6.1)
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.set_ylabel("Peak CUDA allocated memory (MB)")
    ax.set_ylim(0, max(values) * 1.20)
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.4, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "TNNLS_Fig_PeakAllocatedMemory")


def elapsed_time(rows: list[dict[str, object]]) -> None:
    grouped = depth16_rows(rows)
    labels = list(grouped)
    markers = ["o", "s", "^"]
    linestyles = ["-", "--", ":"]
    seed_colors = [BLUE, ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    for seed, marker, linestyle, color in zip([0, 1, 2], markers, linestyles, seed_colors):
        values = []
        for label in labels:
            match = [float(row["elapsed_seconds"]) for row in grouped[label] if int(row["seed"]) == seed]
            if len(match) != 1:
                raise ValueError(f"missing elapsed record for seed={seed}, condition={label}")
            values.append(match[0])
        ax.plot(range(len(labels)), values, color=color, marker=marker, linestyle=linestyle, label=f"Seed {seed}")
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.set_ylabel("Recorded run elapsed time (s)")
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.4, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", handlelength=2.3)
    save(fig, "TNNLS_Fig_RecordedRunElapsed")


def add_round_box(ax, x, y, w, h, title, lines) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=1.05,
        edgecolor=BLACK,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(box)
    # Reserve separate vertical zones for the heading and the two detail lines.
    # This prevents glyph overlap after the figure is reduced to IEEE width.
    ax.text(
        x + w / 2,
        y + h * 0.74,
        title,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=6.15,
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=5.15,
        linespacing=1.25,
        zorder=3,
    )


def add_arrow(ax, x1, x2, y) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.05,
            color=BLACK,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 4.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    headings = [(0.84, "1", "Layer diagnosis and repair construction"), (0.40, "2", "Matched falsification and evidence gate")]
    for y, number, title in headings:
        ax.add_patch(Circle((0.055, y), 0.021, facecolor="white", edgecolor=BLACK, linewidth=1.2))
        ax.text(0.055, y, number, ha="center", va="center", fontweight="bold", fontsize=8)
        ax.text(0.085, y, title, ha="left", va="center", fontweight="bold", fontsize=10)

    lane_specs = [(0.54, 0.75), (0.10, 0.31)]
    for y0, y1 in lane_specs:
        ax.add_patch(
            FancyBboxPatch(
                (0.035, y0),
                0.93,
                y1 - y0,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                linewidth=1.2,
                edgecolor=BLACK,
                facecolor="none",
                linestyle=(0, (3, 2)),
                zorder=1,
            )
        )

    x = [0.055, 0.215, 0.375, 0.535, 0.695, 0.855]
    w = [0.115, 0.115, 0.125, 0.120, 0.125, 0.090]
    h = 0.130
    top_y = 0.570
    top = [
        ("Graph + split", ["public data", "fixed masks"]),
        ("Deep GNN", ["L layers", "fixed budget"]),
        ("Layer metrics", ["energy, rank", "gradient ratio"]),
        ("Fixed selector", ["epsilon = 0.5", "declared tie rule"]),
        ("Repair mask", ["diagnosed suffix", "same repair"]),
        ("Variants", ["targeted", "uniform"]),
    ]
    for xi, wi, (title, lines) in zip(x, w, top):
        add_round_box(ax, xi, top_y, wi, h, title, lines)
    for i in range(len(x) - 1):
        add_arrow(ax, x[i] + w[i], x[i + 1], top_y + h / 2)

    bottom_y = 0.135
    bottom = [
        ("Paired seeds", ["same split", "same budget"]),
        ("Untreated", ["plain depth-16", "reference"]),
        ("Wrong location", ["same cardinality", "same repair"]),
        ("Mismatch", ["other repair", "same mask"]),
        ("Paired analysis", ["CI + effect size", "corrected tests"]),
        ("Decision gate", ["support", "or retire claim"]),
    ]
    for xi, wi, (title, lines) in zip(x, w, bottom):
        add_round_box(ax, xi, bottom_y, wi, h, title, lines)
    for i in range(len(x) - 1):
        add_arrow(ax, x[i] + w[i], x[i + 1], bottom_y + h / 2)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "TNNLS_Fig3_Pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "TNNLS_Fig3_Pipeline.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "TNNLS_Fig3_Pipeline.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_captions() -> None:
    captions = """TNNLS_Fig_FinalAccuracyDepthBand: Mean final Cora test accuracy of the untreated model across three supplied seeds, with the shaded region showing plus or minus one sample standard deviation (not a confidence interval).

TNNLS_Fig_PeakAllocatedMemory: Peak memory allocated by PyTorch CUDA for each depth-16 condition; identical seed values arise from identical tensor shapes and do not represent total board memory or reserved memory.

TNNLS_Fig_RecordedRunElapsed: Recorded end-to-end elapsed time for each depth-16 run and seed, covering training, validation, test evaluation, and the final diagnostic rather than isolated epoch wall-clock time.

TNNLS_Fig3_Pipeline: The proposed workflow converts fixed layerwise diagnostics into repair masks and evaluates them against matched falsification controls before any effectiveness claim is admitted.

TNNLS_Fig4_Depth16DiagnosticTrajectories: Recorded depth-16 layer trajectories from the real RTX A6000 manifest; lines are seed means and shaded bands are sample standard deviations (n = 3), with no smoothing or interpolation, and the dotted line in panel (c) marks epsilon = 0.5.

TNNLS_Fig5_A6000ExecutionProfile: Recorded depth-16 PyTorch peak allocated CUDA memory and whole-run elapsed time from the same three RTX A6000 seeds; elapsed time is neither per-epoch time nor synchronized throughput.
"""
    (OUT / "figure_captions.txt").write_text(captions)


def main() -> None:
    configure()
    rows = load_rows()
    accuracy_depth_band(rows)
    peak_memory(rows)
    elapsed_time(rows)
    pipeline()
    write_captions()


if __name__ == "__main__":
    main()
