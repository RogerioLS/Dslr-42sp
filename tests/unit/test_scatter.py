"""Unit tests for src.visualization.scatter module."""

import unittest
from pathlib import Path

import matplotlib
import pandas as pd

from src.analytics.loader import load_csv
from src.visualization.scatter import (
    compute_correlation_matrix,
    find_most_correlated_pair,
    plot_scatter,
)

matplotlib.use("Agg")  # Non-GUI headless backend for testing


class TestScatterModule(unittest.TestCase):
    """Test suite for scatter plot generation and bivariate correlation analysis."""

    def setUp(self) -> None:
        """Sets up test dataset paths and mock DataFrames."""
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.train_dataset_path = self.root_dir / "datasets" / "dataset_train.csv"

        self.mock_df = pd.DataFrame(
            {
                "Hogwarts House": [
                    "Gryffindor",
                    "Gryffindor",
                    "Hufflepuff",
                    "Hufflepuff",
                    "Ravenclaw",
                    "Ravenclaw",
                    "Slytherin",
                    "Slytherin",
                ],
                "CourseA": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "CourseB": [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0],
                "CourseC": [5.0, 1.0, 8.0, 2.0, 7.0, 3.0, 9.0, 4.0],
            }
        )

    def test_find_most_correlated_pair_mock(self) -> None:
        """Verifies identification of most correlated pair with mock linear relation."""
        c1, c2, r = find_most_correlated_pair(self.mock_df, ["CourseA", "CourseB", "CourseC"])
        pair = {c1, c2}
        self.assertEqual(pair, {"CourseA", "CourseB"})
        self.assertAlmostEqual(r, -1.0, places=5)

    def test_find_most_correlated_pair_real_dataset(self) -> None:
        """Verifies that Astronomy and Defense Against the Dark Arts are identified."""
        if not self.train_dataset_path.exists():
            self.skipTest("dataset_train.csv not found.")

        df = load_csv(self.train_dataset_path)
        c1, c2, r = find_most_correlated_pair(df)
        pair = {c1, c2}
        expected_pair = {"Astronomy", "Defense Against the Dark Arts"}
        self.assertEqual(pair, expected_pair)
        self.assertAlmostEqual(r, -1.0, places=4)

    def test_compute_correlation_matrix_mock(self) -> None:
        """Verifies computation of full pairwise correlation matrix."""
        matrix = compute_correlation_matrix(self.mock_df, ["CourseA", "CourseB"])
        self.assertAlmostEqual(matrix["CourseA"]["CourseA"], 1.0)
        self.assertAlmostEqual(matrix["CourseA"]["CourseB"], -1.0)
        self.assertAlmostEqual(matrix["CourseB"]["CourseA"], -1.0)
        self.assertAlmostEqual(matrix["CourseB"]["CourseB"], 1.0)

    def test_find_most_correlated_pair_few_courses(self) -> None:
        """Verifies ValueError when fewer than 2 courses are provided."""
        with self.assertRaises(ValueError):
            find_most_correlated_pair(self.mock_df, ["CourseA"])

    def test_plot_scatter_mock(self) -> None:
        """Verifies scatter plot generation in headless mode without errors."""
        fig = plot_scatter(
            self.mock_df,
            "CourseA",
            "CourseB",
            output_path=None,
            show=False,
        )
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    unittest.main()
