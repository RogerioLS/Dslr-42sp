"""Integration tests for describe.py command-line interface.

Verifies end-to-end execution, CLI argument handling, error reporting, and
output format alignment matching the 42 DSLR specification.
"""

import subprocess
import sys
import unittest
from pathlib import Path

from describe import format_describe_table, run_describe


class TestCLIDescribe(unittest.TestCase):
    """Integration test suite for describe.py CLI."""

    def setUp(self) -> None:
        """Sets up repository paths and dataset references."""
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.describe_script = self.root_dir / "describe.py"
        self.train_dataset = self.root_dir / "datasets" / "dataset_train.csv"
        self.test_dataset = self.root_dir / "datasets" / "dataset_test.csv"

    def test_describe_train_dataset_subprocess(self) -> None:
        """Verifies running describe.py on train dataset exits with code 0 and correct structure."""
        result = subprocess.run(
            [sys.executable, str(self.describe_script), str(self.train_dataset)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout

        expected_metrics = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        for metric in expected_metrics:
            self.assertIn(metric, output)

        expected_courses = [
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Care of Magical Creatures",
            "Charms",
            "Flying",
        ]
        for course in expected_courses:
            self.assertIn(course, output)

    def test_describe_test_dataset_subprocess(self) -> None:
        """Verifies running describe.py on test dataset exits with code 0."""
        result = subprocess.run(
            [sys.executable, str(self.describe_script), str(self.test_dataset)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Count", result.stdout)

    def test_describe_missing_arguments(self) -> None:
        """Verifies describe.py exits with code 1 when invoked without arguments."""
        result = subprocess.run(
            [sys.executable, str(self.describe_script)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_describe_nonexistent_file(self) -> None:
        """Verifies describe.py exits with code 1 when given a non-existent file path."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.describe_script),
                "nonexistent_dataset_path.csv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Error:", result.stderr)

    def test_run_describe_function_directly(self) -> None:
        """Verifies direct programmatic execution via run_describe helper."""
        self.assertEqual(run_describe(str(self.train_dataset)), 0)
        self.assertEqual(run_describe("invalid_file.csv"), 1)

    def test_format_describe_table_empty(self) -> None:
        """Verifies format_describe_table gracefully handles empty feature lists."""
        self.assertEqual(format_describe_table([], {}), "")


if __name__ == "__main__":
    unittest.main()
