"""Integration tests for histogram.py command-line interface.

Verifies end-to-end execution, CLI argument parsing, missing file handling,
and course distribution homogeneity analysis matching the 42 DSLR specification.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestCLIHistogram(unittest.TestCase):
    """Integration test suite for histogram.py CLI."""

    def setUp(self) -> None:
        """Sets up repository paths and dataset references."""
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.histogram_script = self.root_dir / "histogram.py"
        self.train_dataset = self.root_dir / "datasets" / "dataset_train.csv"

        # Headless matplotlib environment to prevent GUI popup in CI/CD
        self.env = os.environ.copy()
        self.env["MPLBACKEND"] = "Agg"

    def test_histogram_no_args_fails(self) -> None:
        """Verifies running histogram.py without args exits with code 1 and usage error."""
        result = subprocess.run(
            [sys.executable, str(self.histogram_script)],
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage: python3 histogram.py", result.stderr)

    def test_histogram_missing_file_fails(self) -> None:
        """Verifies running histogram.py with a non-existent file exits with code 1."""
        result = subprocess.run(
            [sys.executable, str(self.histogram_script), "non_existent.csv"],
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stderr)

    def test_histogram_train_dataset_success(self) -> None:
        """Verifies running histogram.py on dataset_train.csv executes with code 0."""
        if not self.train_dataset.exists():
            self.skipTest("dataset_train.csv not found.")

        result = subprocess.run(
            [sys.executable, str(self.histogram_script), str(self.train_dataset)],
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("HISTOGRAM DISTRIBUTION ANALYSIS", result.stdout)
        self.assertIn("Most Homogeneous Course", result.stdout)


if __name__ == "__main__":
    unittest.main()
