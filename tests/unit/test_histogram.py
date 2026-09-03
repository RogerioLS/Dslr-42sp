"""Unit tests for src.visualization.histogram module."""

import unittest
from pathlib import Path

import matplotlib
import pandas as pd

from src.analytics.loader import load_csv
from src.visualization.histogram import (
    compute_house_course_stats,
    find_most_homogeneous_course,
    plot_histograms_grid,
)

matplotlib.use("Agg")  # Non-GUI headless backend for testing


class TestHistogramModule(unittest.TestCase):
    """Test suite for histogram generation and course homogeneity analysis."""

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
                "Care of Magical Creatures": [5.0, 5.2, 4.9, 5.1, 5.0, 5.0, 5.1, 4.8],
                "Defense Against the Dark Arts": [
                    10.0,
                    12.0,
                    -50.0,
                    -48.0,
                    30.0,
                    35.0,
                    90.0,
                    95.0,
                ],
            }
        )

    def test_compute_house_course_stats(self) -> None:
        """Verifies mean and standard deviation per house for a course."""
        stats = compute_house_course_stats(self.mock_df, "Care of Magical Creatures")
        self.assertIn("Gryffindor", stats)
        self.assertIn("Hufflepuff", stats)
        self.assertIn("Ravenclaw", stats)
        self.assertIn("Slytherin", stats)
        self.assertAlmostEqual(stats["Gryffindor"]["mean"], 5.1, delta=0.01)

    def test_find_most_homogeneous_course_mock(self) -> None:
        """Verifies that the most homogeneous course has lowest variance across house means."""
        course, variance = find_most_homogeneous_course(self.mock_df)
        self.assertEqual(course, "Care of Magical Creatures")
        self.assertLess(variance, 1.0)

    def test_find_most_homogeneous_course_real_dataset(self) -> None:
        """Verifies homogeneity calculation on official 42 dataset_train.csv."""
        if not self.train_dataset_path.exists():
            self.skipTest("dataset_train.csv not found.")

        df = load_csv(self.train_dataset_path)
        course, variance = find_most_homogeneous_course(df)
        self.assertIsInstance(course, str)
        self.assertTrue(len(course) > 0)
        self.assertGreaterEqual(variance, 0.0)
        # Care of Magical Creatures is the known homogeneous feature in 42 subject
        self.assertIn(course, ["Care of Magical Creatures", "Arithmancy"])

    def test_plot_histograms_grid_mock(self) -> None:
        """Verifies histogram plot generation in headless mode without errors."""
        fig = plot_histograms_grid(self.mock_df, output_path=None, show=False)
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    unittest.main()
