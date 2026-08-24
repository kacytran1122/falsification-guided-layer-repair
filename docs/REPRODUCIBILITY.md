# Reproducibility Guide

## Released evidence

The repository includes a compact table for 27 completed conditions, an exact 15-record depth-16 diagnostic subset, grouped summary values, the training harness, integrity checks, and plotting programs. The recorded environment identifies one NVIDIA RTX A6000 at `cuda:0`, PyTorch `2.6.0+cu124`, CUDA runtime `12.4`, NVIDIA driver `595.84`, and Python `3.12.3`.

The release does not contain raw per-epoch histories, the full remote manifest, checkpoints, standard output or error logs, or a captured environment lockfile. The provided dependency files describe compatible packages; they do not claim bitwise reconstruction of the original environment.

## Local verification

Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the artifact checks:

```bash
python -m unittest discover -s tests -v
```

Regenerate all figures supported by the released evidence:

```bash
python scripts/plot_summary.py
python scripts/plot_diagnostics.py
```

The plotting programs consume only committed CSV and JSON values. They do not smooth, extrapolate, impute, or synthesize missing measurements.

## RTX A6000 replication

Use a CUDA host with a PyTorch build compatible with the installed driver. Install the declared GPU environment:

```bash
python -m pip install -r requirements-a6000.txt
```

Launch the complete pilot:

```bash
python src/pilot_harness.py --output runs/pilot --data data/planetoid
```

The harness downloads the public Planetoid Cora files when they are absent, verifies that CUDA is available, runs the five untreated depth conditions and four depth-16 repair conditions over seeds 0, 1, and 2, and writes a new result directory. Preserve that directory immutably before analysis. Newly produced measurements are independent replication evidence, not the original execution.

## Required logging for a confirmatory run

Record one append-only row per epoch with `run_id`, dataset, backbone, split revision, seed, condition, depth, epoch, training loss, validation loss, training accuracy, validation accuracy, learning rate, allocated CUDA bytes, reserved CUDA bytes, synchronized epoch duration, cumulative duration, and stopping reason. At completion, retain the selected epoch, one-time test prediction, parameter count, a validated FLOP or operator count, throughput, command line, configuration, environment lockfile, dataset hashes, checkpoint hashes, predictions, logits, and standard output and error logs.

The full confirmatory design is listed in [EXPERIMENTS_REQUIRED.md](EXPERIMENTS_REQUIRED.md).
