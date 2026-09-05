"""Unit tests for src.visualization.pair module."""

import unittest

import matplotlib
import pandas as pd

from src.visualization.pair import DISCARDED_FEATURES, SELECTED_FEATURES, plot_pair_plot

matplotlib.use("Agg")  # Non-GUI headless backend for testing


class TestPairModule(unittest.TestCase):
    """Test suite for pair plot multivariate visualization and feature selection."""

    def setUp(self) -> None:
        """Sets up mock DataFrame with minimal features for fast execution."""
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
                "Herbology": [5.0, 5.2, 4.9, 5.1, 8.0, 8.1, 2.0, 2.2],
                "Divination": [10.0, 12.0, -5.0, -6.0, 3.0, 3.5, 9.0, 9.5],
                "Astronomy": [-50.0, -48.0, 10.0, 12.0, 30.0, 32.0, -20.0, -18.0],
            }
        )

    def test_selected_features_constants(self) -> None:
        """Verifies that SELECTED_FEATURES contains valid, expected courses."""
        self.assertGreaterEqual(len(SELECTED_FEATURES), 8)
        self.assertIn("Astronomy", SELECTED_FEATURES)
        self.assertIn("Herbology", SELECTED_FEATURES)
        self.assertNotIn("Arithmancy", SELECTED_FEATURES)
        self.assertNotIn("Care of Magical Creatures", SELECTED_FEATURES)
        self.assertNotIn("Defense Against the Dark Arts", SELECTED_FEATURES)

    def test_discarded_features_rationale(self) -> None:
        """Verifies that DISCARDED_FEATURES documents mathematical justifications."""
        self.assertIn("Arithmancy", DISCARDED_FEATURES)
        self.assertIn("Care of Magical Creatures", DISCARDED_FEATURES)
        self.assertIn("Defense Against the Dark Arts", DISCARDED_FEATURES)

    def test_plot_pair_plot_mock(self) -> None:
        """Verifies pair plot generation on mock data in headless mode without errors."""
        grid = plot_pair_plot(
            self.mock_df,
            courses=["Herbology", "Divination"],
            output_path=None,
            show=False,
        )
        self.assertIsNotNone(grid)
        matplotlib.pyplot.close(grid.fig)

    def test_plot_pair_plot_missing_house_column(self) -> None:
        """Verifies ValueError when 'Hogwarts House' column is missing."""
        df_no_house = self.mock_df.drop(columns=["Hogwarts House"])
        with self.assertRaises(ValueError):
            plot_pair_plot(df_no_house, courses=["Herbology", "Divination"], show=False)

    def test_plot_pair_plot_insufficient_courses(self) -> None:
        """Verifies ValueError when fewer than 2 courses are provided."""
        with self.assertRaises(ValueError):
            plot_pair_plot(self.mock_df, courses=["Herbology"], show=False)


if __name__ == "__main__":
    unittest.main()
