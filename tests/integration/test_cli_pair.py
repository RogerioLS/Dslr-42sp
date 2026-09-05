"""Integration tests for pair_plot.py command-line interface.

Verifies end-to-end execution, CLI argument parsing, missing file handling,
and multivariate pair plot matrix generation matching the 42 DSLR specification.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestCLIPairPlot(unittest.TestCase):
    """Integration test suite for pair_plot.py CLI."""

    def setUp(self) -> None:
        """Sets up repository paths and test environments."""
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.pair_script = self.root_dir / "pair_plot.py"

        self.env = os.environ.copy()
        self.env["MPLBACKEND"] = "Agg"

    def test_pair_plot_no_args_fails(self) -> None:
        """Verifies running pair_plot.py without args exits with code 1 and usage error."""
        result = subprocess.run(
            [sys.executable, str(self.pair_script)],
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage: python3 pair_plot.py", result.stderr)

    def test_pair_plot_missing_file_fails(self) -> None:
        """Verifies running pair_plot.py with a non-existent file exits with code 1."""
        result = subprocess.run(
            [sys.executable, str(self.pair_script), "non_existent_dataset.csv"],
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stderr)

    def test_pair_plot_execution_sample(self) -> None:
        """Verifies fast end-to-end execution on a sample CSV."""
        sample_df = pd.DataFrame(
            {
                "Hogwarts House": ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"] * 5,
                "Astronomy": [10.0, 20.0, 30.0, 40.0] * 5,
                "Herbology": [-5.0, -10.0, 15.0, 20.0] * 5,
                "Ancient Runes": [100.0, 200.0, 300.0, 400.0] * 5,
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
            sample_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, str(self.pair_script), tmp_path],
                capture_output=True,
                text=True,
                env=self.env,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("PAIR PLOT MULTIVARIATE FEATURE SELECTION", result.stdout)
            self.assertIn("Selected Features for Logistic Regression", result.stdout)
            self.assertIn("Excluded Features", result.stdout)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Remove generated pair_plot.png if created in root
            plot_png = self.root_dir / "pair_plot.png"
            if plot_png.exists():
                plot_png.unlink()


if __name__ == "__main__":
    unittest.main()
