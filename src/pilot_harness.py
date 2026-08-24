#!/usr/bin/env python3
"""Real-data A6000 pilot for localized GNN-depth diagnostics.

This script intentionally makes no journal-scale claim.  It trains only on the
public Planetoid Cora split and records enough provenance to decide whether the
localized-repair hypothesis deserves the full multi-dataset, ten-seed study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


PLANETOID_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
PLANETOID_OBJECTS = ("x", "y", "tx", "ty", "allx", "ally", "graph")


@dataclass
class RunRecord:
    seed: int
    depth: int
    mode: str
    repair_kind: str
    repair_layers: list[int]
    best_epoch: int
    best_val_accuracy: float
    test_accuracy: float
    elapsed_seconds: float
    peak_allocated_bytes: int
    energy: list[float]
    effective_rank: list[float]
    gradient_norm: list[float]
    local_gradient_transmission: list[float]
    energy_collapse_layer: int | None
    rank_collapse_layer: int | None
    gradient_contraction_layer: int | None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fetch_cora(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in PLANETOID_OBJECTS:
        path = root / f"ind.cora.{name}"
        if not path.exists():
            urllib.request.urlretrieve(f"{PLANETOID_BASE}/{path.name}", path)
        paths[name] = path
    index_path = root / "ind.cora.test.index"
    if not index_path.exists():
        urllib.request.urlretrieve(f"{PLANETOID_BASE}/{index_path.name}", index_path)

    objects: dict[str, object] = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            objects[name] = pickle.load(handle, encoding="latin1")
    test_index = np.array([int(line) for line in index_path.read_text().splitlines()])
    test_index_sorted = np.sort(test_index)

    features = sp.vstack((objects["allx"], objects["tx"])).tolil()
    features[test_index, :] = features[test_index_sorted, :]
    labels = np.vstack((objects["ally"], objects["ty"]))
    labels[test_index, :] = labels[test_index_sorted, :]
    y = labels.argmax(axis=1).astype(np.int64)

    graph = objects["graph"]
    undirected_edges: set[tuple[int, int]] = set()
    for src, dsts in graph.items():
        for dst in dsts:
            if int(src) != int(dst):
                a, b = sorted((int(src), int(dst)))
                undirected_edges.add((a, b))
    pairs = np.asarray(sorted(undirected_edges), dtype=np.int64)
    directed = np.concatenate((pairs, pairs[:, ::-1]), axis=0)

    x = np.asarray(features.todense(), dtype=np.float32)
    row_sum = x.sum(axis=1, keepdims=True)
    x = x / np.maximum(row_sum, 1.0)
    n = x.shape[0]
    self_loops = np.arange(n, dtype=np.int64)
    row = np.concatenate((directed[:, 0], self_loops))
    col = np.concatenate((directed[:, 1], self_loops))
    deg = np.bincount(row, minlength=n).astype(np.float32)
    values = 1.0 / np.sqrt(deg[row] * deg[col])
    indices = torch.from_numpy(np.stack((row, col), axis=0))
    adjacency = torch.sparse_coo_tensor(
        indices, torch.from_numpy(values), (n, n)
    ).coalesce()

    train_index = np.arange(objects["y"].shape[0], dtype=np.int64)
    val_index = np.arange(objects["y"].shape[0], objects["y"].shape[0] + 500)
    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "adjacency": adjacency,
        "edge_index": torch.from_numpy(pairs.T.copy()),
        "degree": torch.from_numpy(np.bincount(directed[:, 0], minlength=n).astype(np.float32)),
        "train_index": torch.from_numpy(train_index),
        "val_index": torch.from_numpy(val_index),
        "test_index": torch.from_numpy(test_index),
        "checksums": {path.name: sha256(path) for path in [*paths.values(), index_path]},
    }


class PairNorm(nn.Module):
    def __init__(self, scale: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=0, keepdim=True)
        root_mean_square = centered.pow(2).sum(dim=1).mean().sqrt()
        return self.scale * centered / root_mean_square.clamp_min(self.eps)


class DeepGCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int,
        classes: int,
        depth: int,
        dropout: float,
        repair_kind: str = "none",
        repair_layers: set[int] | None = None,
        residual_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_features, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth))
        self.readout = nn.Linear(hidden, classes)
        self.pairnorm = PairNorm()
        self.dropout = dropout
        self.repair_kind = repair_kind
        self.repair_layers = repair_layers or set()
        self.residual_alpha = residual_alpha

    def forward(
        self, adjacency: torch.Tensor, x: torch.Tensor, retain: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h0 = F.relu(self.input_projection(x))
        h = h0
        activations: list[torch.Tensor] = []
        for layer_number, layer in enumerate(self.layers, start=1):
            h = torch.sparse.mm(adjacency, h)
            h = F.relu(layer(h))
            if layer_number in self.repair_layers:
                if self.repair_kind == "pairnorm":
                    h = self.pairnorm(h)
                elif self.repair_kind == "residual":
                    h = (1.0 - self.residual_alpha) * h + self.residual_alpha * h0
            if retain:
                h.retain_grad()
            activations.append(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.readout(h), activations


def normalized_dirichlet_energy(
    h: torch.Tensor, edge_index: torch.Tensor, degree: torch.Tensor
) -> float:
    src, dst = edge_index
    scaled = h / torch.sqrt(1.0 + degree).unsqueeze(1)
    numerator = (scaled[src] - scaled[dst]).pow(2).sum()
    denominator = h.pow(2).sum().clamp_min(1e-12)
    return float((numerator / denominator).item())


def effective_rank(h: torch.Tensor) -> float:
    centered = h - h.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.float())
    probabilities = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def threshold_layer(values: list[float], epsilon: float) -> int | None:
    if not values or values[0] <= 0:
        return None
    threshold = epsilon * values[0]
    for layer, value in enumerate(values, start=1):
        if value < threshold:
            return layer
    return None


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor, index: torch.Tensor) -> float:
    return float((logits[index].argmax(dim=1) == labels[index]).float().mean().item())


def diagnose(
    model: DeepGCN,
    data: dict[str, object],
    epsilon: float,
) -> dict[str, object]:
    model.eval()
    model.zero_grad(set_to_none=True)
    logits, activations = model(data["adjacency"], data["x"], retain=True)
    loss = F.cross_entropy(logits[data["train_index"]], data["y"][data["train_index"]])
    loss.backward()
    energy = [
        normalized_dirichlet_energy(h.detach(), data["edge_index"], data["degree"])
        for h in activations
    ]
    rank = [effective_rank(h.detach()) for h in activations]
    gradient = [float(h.grad.norm().item()) for h in activations]
    local = [
        gradient[layer - 1] / max(gradient[layer], 1e-30)
        for layer in range(1, len(gradient))
    ]
    contraction = next(
        (layer + 1 for layer, value in enumerate(local, start=1) if value < epsilon),
        None,
    )
    return {
        "energy": energy,
        "effective_rank": rank,
        "gradient_norm": gradient,
        "local_gradient_transmission": local,
        "energy_collapse_layer": threshold_layer(energy, epsilon),
        "rank_collapse_layer": threshold_layer(rank, epsilon),
        "gradient_contraction_layer": contraction,
    }


def train_one(
    data: dict[str, object],
    seed: int,
    depth: int,
    mode: str,
    repair_kind: str,
    repair_layers: set[int],
    args: argparse.Namespace,
) -> RunRecord:
    seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = DeepGCN(
        in_features=data["x"].shape[1],
        hidden=args.hidden,
        classes=int(data["y"].max().item()) + 1,
        depth=depth,
        dropout=args.dropout,
        repair_kind=repair_kind,
        repair_layers=repair_layers,
        residual_alpha=args.residual_alpha,
    ).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_val = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(data["adjacency"], data["x"])
        loss = F.cross_entropy(logits[data["train_index"]], data["y"][data["train_index"]])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(data["adjacency"], data["x"])
            val = accuracy(val_logits, data["y"], data["val_index"])
        if val > best_val + 1e-12:
            best_val = val
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(data["adjacency"], data["x"])
        test = accuracy(test_logits, data["y"], data["test_index"])
    diagnostic = diagnose(model, data, args.epsilon)
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return RunRecord(
        seed=seed,
        depth=depth,
        mode=mode,
        repair_kind=repair_kind,
        repair_layers=sorted(repair_layers),
        best_epoch=best_epoch,
        best_val_accuracy=best_val,
        test_accuracy=test,
        elapsed_seconds=elapsed,
        peak_allocated_bytes=peak,
        **diagnostic,
    )


def choose_diagnosis(record: RunRecord) -> tuple[str, int]:
    candidates: list[tuple[int, str]] = []
    if record.energy_collapse_layer is not None:
        candidates.append((record.energy_collapse_layer, "pairnorm"))
    if record.rank_collapse_layer is not None:
        candidates.append((record.rank_collapse_layer, "pairnorm"))
    if record.gradient_contraction_layer is not None:
        candidates.append((record.gradient_contraction_layer, "residual"))
    if not candidates:
        return "residual", max(2, record.depth // 2)
    layer, kind = min(candidates, key=lambda item: (item[0], item[1]))
    return kind, layer


def wrong_location_mask(depth: int, target: set[int], seed: int) -> set[int]:
    count = len(target)
    if count == 0 or count == depth:
        return set(range(1, count + 1))
    rng = random.Random(10_000 + seed)
    for _ in range(1000):
        candidate = set(rng.sample(range(1, depth + 1), count))
        if candidate != target:
            return candidate
    raise RuntimeError("could not construct wrong-location control")


def environment_manifest(device: torch.device) -> dict[str, object]:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - only for provenance fallback
        nvidia_smi = f"unavailable: {exc}"
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
        "nvidia_smi": nvidia_smi,
    }


def summarize(records: list[RunRecord]) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(f"depth={record.depth}|mode={record.mode}", []).append(record.test_accuracy)
    return {
        key: {
            "n": len(values),
            "mean_test_accuracy": float(np.mean(values)),
            "sample_std_test_accuracy": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
        for key, values in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--data", type=Path, default=Path("data/planetoid"))
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--repair-depth", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--residual-alpha", type=float, default=0.1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: refusing to report a CPU run as A6000 evidence")
    args.device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)

    raw = fetch_cora(args.data)
    checksums = raw.pop("checksums")
    data = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }
    records: list[RunRecord] = []
    jsonl_path = args.output / "records.jsonl"
    for depth in args.depths:
        for seed in args.seeds:
            record = train_one(data, seed, depth, "plain", "none", set(), args)
            records.append(record)
            with jsonl_path.open("a") as handle:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            print(json.dumps({"finished": f"plain/d{depth}/s{seed}", "test": record.test_accuracy}), flush=True)

    baselines = {
        record.seed: record
        for record in records
        if record.depth == args.repair_depth and record.mode == "plain"
    }
    for seed in args.seeds:
        diagnosed_kind, start_layer = choose_diagnosis(baselines[seed])
        target = set(range(start_layer, args.repair_depth + 1))
        wrong = wrong_location_mask(args.repair_depth, target, seed)
        other_kind = "residual" if diagnosed_kind == "pairnorm" else "pairnorm"
        conditions = [
            ("targeted", diagnosed_kind, target),
            ("uniform", diagnosed_kind, set(range(1, args.repair_depth + 1))),
            ("wrong_location", diagnosed_kind, wrong),
            ("mismatched", other_kind, target),
        ]
        for mode, kind, layers in conditions:
            record = train_one(data, seed, args.repair_depth, mode, kind, layers, args)
            records.append(record)
            with jsonl_path.open("a") as handle:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            print(json.dumps({"finished": f"{mode}/d{args.repair_depth}/s{seed}", "test": record.test_accuracy}), flush=True)

    manifest = {
        "status": "real-data pilot; not submission evidence",
        "dataset": "Planetoid Cora public split",
        "dataset_sha256": checksums,
        "environment": environment_manifest(args.device),
        "arguments": {
            key: str(value) if isinstance(value, (Path, torch.device)) else value
            for key, value in vars(args).items()
        },
        "summary": summarize(records),
        "records": [asdict(record) for record in records],
    }
    (args.output / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (args.output / "pilot_manifest.sha256").write_text(
        f"{sha256(args.output / 'pilot_manifest.json')}  pilot_manifest.json\n"
    )
    os.chmod(jsonl_path, 0o600)
    os.chmod(args.output / "pilot_manifest.json", 0o600)
    os.chmod(args.output / "pilot_manifest.sha256", 0o600)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
