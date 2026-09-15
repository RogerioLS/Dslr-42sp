"""Integration Test Suite for logreg_train.py CLI Executable (DSLR-10).

Validates:
1. CLI help flags (-h, --help) and usage instructions.
2. Error handling when input files are missing or invalid.
3. End-to-end model training execution and weights.json generation via subprocess.
4. Feature selection strategies ('all' vs 'selected').
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLI_PATH = BASE_DIR / "logreg_train.py"
TRAIN_DATASET = BASE_DIR / "datasets" / "dataset_train.csv"


class TestCLITrainIntegration(unittest.TestCase):
    """End-to-end subprocess integration tests for logreg_train.py."""

    def test_cli_help(self) -> None:
        """Tests that invoking with --help displays usage and exits with code 0."""
        res = subprocess.run(
            [sys.executable, str(CLI_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("usage: logreg_train.py", res.stdout.lower())
        self.assertIn("--output", res.stdout)
        self.assertIn("--epochs", res.stdout)
        self.assertIn("--learning-rate", res.stdout)

    def test_cli_missing_argument(self) -> None:
        """Tests that invoking without arguments exits with non-zero code."""
        res = subprocess.run(
            [sys.executable, str(CLI_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("error", res.stderr.lower())

    def test_cli_nonexistent_dataset(self) -> None:
        """Tests that providing a non-existent file path exits with code 1."""
        res = subprocess.run(
            [sys.executable, str(CLI_PATH), "non_existent_file.csv"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("error", res.stderr.lower())

    def test_cli_train_execution_end_to_end(self) -> None:
        """Tests complete end-to-end training and weights generation on dataset_train.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_weights = Path(tmpdir) / "output_weights.json"

            res = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    str(TRAIN_DATASET),
                    "--output",
                    str(output_weights),
                    "--epochs",
                    "50",
                    "--learning-rate",
                    "0.5",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(res.returncode, 0)
            self.assertTrue(output_weights.exists())

            # Validate generated weights file content
            with open(output_weights, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertIn("classes", payload)
            self.assertEqual(
                payload["classes"],
                ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
            )
            self.assertEqual(len(payload["features"]), 13)
            self.assertIn("scaler", payload)
            self.assertIn("weights", payload)

            for house in payload["classes"]:
                self.assertIn(house, payload["weights"])
                self.assertEqual(len(payload["weights"][house]), 14)  # 1 bias + 13 features

    def test_cli_train_selected_features(self) -> None:
        """Tests training with the 9 selected features strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_weights = Path(tmpdir) / "selected_weights.json"

            res = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    str(TRAIN_DATASET),
                    "--output",
                    str(output_weights),
                    "--features",
                    "selected",
                    "--epochs",
                    "30",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(res.returncode, 0)
            self.assertTrue(output_weights.exists())

            with open(output_weights, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(len(payload["features"]), 9)
            for house in payload["classes"]:
                self.assertEqual(len(payload["weights"][house]), 10)  # 1 bias + 9 features

    def test_cli_train_optimizers_bonus(self) -> None:
        """Tests training CLI with mini-batch and SGD optimizer flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mb_weights = Path(tmpdir) / "mb_weights.json"
            sgd_weights = Path(tmpdir) / "sgd_weights.json"

            # Mini-batch run
            res_mb = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    str(TRAIN_DATASET),
                    "--output",
                    str(mb_weights),
                    "--method",
                    "minibatch",
                    "--batch-size",
                    "64",
                    "--epochs",
                    "20",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res_mb.returncode, 0)
            self.assertTrue(mb_weights.exists())

            # SGD run
            res_sgd = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    str(TRAIN_DATASET),
                    "--output",
                    str(sgd_weights),
                    "--method",
                    "sgd",
                    "--epochs",
                    "10",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res_sgd.returncode, 0)
            self.assertTrue(sgd_weights.exists())


if __name__ == "__main__":
    unittest.main()
