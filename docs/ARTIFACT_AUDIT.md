# Artifact Audit

## Evidence levels

| Artifact | Status | Supported use |
|---|---|---|
| `data/pilot_compact_evidence.csv` | Present, 27 records | Final validation accuracy, final test accuracy, whole-run elapsed time, and peak allocated CUDA memory |
| `data/depth16_diagnostics_manifest_subset.json` | Present, 15 records, checksummed | Depth-16 layerwise energy, effective rank, gradient values, repair masks, final metrics, and recorded GPU metadata |
| `data/pilot_summary.json` | Present | Grouped descriptive statistics and recorded execution environment |
| `src/pilot_harness.py` | Present | Inspection and independent rerunning of the released pilot procedure |
| Figure-generation programs | Present | Regeneration of figures from committed values |
| Per-epoch records | Absent | No loss, accuracy, learning-rate, epoch-time, or epoch-memory curves |
| Checkpoints and predictions | Absent | No exact prediction audit, calibration analysis, or checkpoint replay |
| Full remote manifest and logs | Absent | No complete reconstruction of all run-time events |
| Captured environment lockfile | Absent | No bitwise environment reconstruction |

## Integrity checks

The committed unit tests require 27 compact records, seeds 0 through 2, all five depth-16 conditions, the recorded `NVIDIA RTX A6000` device at `cuda:0`, and the exact diagnostic-subset digest:

```text
c1905c5feb70d6865e2e1b3f1bce31080c85ccebc1c2ef57a45c4668bc658c61
```

The figure programs stop when required rows or condition coverage are missing. They use sample standard deviations across the three available seeds and do not transform missing evidence into numeric results.

## Claim audit

| Claim | Released evidence | Status |
|---|---|---|
| Deep plain-GCN degradation on the evaluated Cora configuration | Five depths, three seeds | Descriptive pilot result |
| Repair viability at depth 16 | Every repaired run exceeds its paired plain run | Pilot support only |
| Diagnostic layer localization | Targeted repair does not beat the wrong-location control; mask overlap is 0.875 | Unsupported |
| Repair-type prescription | Alternative repairs did not receive equal tuning budgets | Unsupported |
| Convergence or optimization speed | Per-epoch histories are absent | Not testable |
| Efficiency | Whole-run elapsed time and peak allocated memory are available, but synchronized epoch counts and throughput are absent | Not established |

## Publication boundary

The artifact supports transparent reporting of the negative pilot and the proposed falsification protocol. It does not support a journal-scale effectiveness claim. The smallest experiments required to test that claim are specified in [EXPERIMENTS_REQUIRED.md](EXPERIMENTS_REQUIRED.md).
