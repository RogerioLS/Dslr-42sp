"""Integration Test Suite for logreg_predict.py CLI Executable (DSLR-11).

Validates:
1. CLI help flags (-h, --help) and usage instructions.
2. Error handling when input dataset or weights files are missing or corrupt.
3. End-to-end inference execution generating a valid houses.csv via subprocess.
4. Conformance of generated CSV output to 42 subject specifications.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLI_PATH = BASE_DIR / "logreg_predict.py"
TEST_DATASET = BASE_DIR / "datasets" / "dataset_test.csv"
WEIGHTS_FILE = BASE_DIR / "weights.json"
EXPECTED_HOUSES = {"Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"}


class TestCLIPredictIntegration(unittest.TestCase):
    """End-to-end subprocess integration tests for logreg_predict.py."""

    def test_cli_help(self) -> None:
        """Tests that invoking with --help displays usage and exits with code 0."""
        res = subprocess.run(
            [sys.executable, str(CLI_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("usage: logreg_predict.py", res.stdout.lower())
        self.assertIn("--output", res.stdout)
        self.assertIn("--quiet", res.stdout)

    def test_cli_missing_arguments(self) -> None:
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
        """Tests that providing a non-existent dataset exits with code 1."""
        res = subprocess.run(
            [sys.executable, str(CLI_PATH), "non_existent_data.csv", str(WEIGHTS_FILE)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("error", res.stderr.lower())

    def test_cli_nonexistent_weights(self) -> None:
        """Tests that providing a non-existent weights file exits with code 1."""
        res = subprocess.run(
            [sys.executable, str(CLI_PATH), str(TEST_DATASET), "non_existent_weights.json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("error", res.stderr.lower())

    def test_cli_end_to_end_inference(self) -> None:
        """Tests complete inference run on test dataset creating valid houses.csv."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            tmp_output = Path(tmp_file.name)

        try:
            res = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    str(TEST_DATASET),
                    str(WEIGHTS_FILE),
                    "--output",
                    str(tmp_output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(res.returncode, 0, msg=f"CLI failed with stderr: {res.stderr}")
            self.assertTrue(tmp_output.exists())

            # Verify CSV content
            df = pd.read_csv(tmp_output)
            self.assertListEqual(list(df.columns), ["Index", "Hogwarts House"])
            self.assertEqual(len(df), 400)

            # Check that houses are valid Hogwarts houses
            predicted_houses = set(df["Hogwarts House"].unique())
            self.assertTrue(predicted_houses.issubset(EXPECTED_HOUSES))
        finally:
            if tmp_output.exists():
                tmp_output.unlink()

    def test_cli_quiet_flag(self) -> None:
        """Tests that --quiet suppresses banner and breakdown output."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            tmp_output = Path(tmp_file.name)

        try:
            res = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    str(TEST_DATASET),
                    str(WEIGHTS_FILE),
                    "--output",
                    str(tmp_output),
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0)
            self.assertEqual(res.stdout.strip(), "")
        finally:
            if tmp_output.exists():
                tmp_output.unlink()


if __name__ == "__main__":
    unittest.main()
