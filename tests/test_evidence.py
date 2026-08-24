import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_compact_table_has_all_conditions(self) -> None:
        with (ROOT / "data" / "pilot_compact_evidence.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 27)
        self.assertEqual(sorted({int(row["seed"]) for row in rows}), [0, 1, 2])

    def test_depth16_subset_is_complete(self) -> None:
        path = ROOT / "data" / "depth16_diagnostics_manifest_subset.json"
        payload = json.loads(path.read_text())
        self.assertEqual(len(payload["records"]), 15)
        self.assertEqual(payload["environment"]["gpu"]["name"], "NVIDIA RTX A6000")
        self.assertEqual(payload["environment"]["device"], "cuda:0")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "c1905c5feb70d6865e2e1b3f1bce31080c85ccebc1c2ef57a45c4668bc658c61",
        )

    def test_depth16_condition_counts(self) -> None:
        payload = json.loads(
            (ROOT / "data" / "depth16_diagnostics_manifest_subset.json").read_text()
        )
        modes = ["plain", "targeted", "uniform", "wrong_location", "mismatched"]
        for mode in modes:
            rows = [record for record in payload["records"] if record["mode"] == mode]
            self.assertEqual(sorted(record["seed"] for record in rows), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
