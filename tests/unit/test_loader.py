"""Unit tests for src.analytics.loader module."""

import unittest
from pathlib import Path
import pandas as pd

from src.analytics.loader import (
    HOGWARTS_COURSES,
    extract_valid_feature_values,
    get_numerical_features,
    load_csv,
)


class TestDataLoader(unittest.TestCase):
    """Test suite for CSV dataset loader and feature extraction functions."""

    def setUp(self) -> None:
        """Sets up test dataset paths and sample DataFrame."""
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.train_dataset_path = self.root_dir / "datasets" / "dataset_train.csv"

        # Create temporary sample DataFrame with NaNs
        self.sample_df = pd.DataFrame(
            {
                "Index": [0, 1, 2, 3],
                "Hogwarts House": ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"],
                "First Name": ["Harry", "Luna", "Draco", "Cedric"],
                "Arithmancy": [10.0, None, 30.0, 40.0],
                "Astronomy": [-50.0, -20.0, None, 10.0],
            }
        )

    def test_load_csv_success(self) -> None:
        """Verifies that load_csv successfully loads existing CSV files."""
        df = load_csv(self.train_dataset_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("Hogwarts House", df.columns)

    def test_load_csv_file_not_found(self) -> None:
        """Verifies that load_csv raises FileNotFoundError for missing paths."""
        missing_path = self.root_dir / "datasets" / "non_existent.csv"
        with self.assertRaises(FileNotFoundError):
            load_csv(missing_path)

    def test_get_numerical_features(self) -> None:
        """Verifies that get_numerical_features excludes metadata columns."""
        features = get_numerical_features(self.sample_df)
        self.assertEqual(features, ["Arithmancy", "Astronomy"])
        self.assertNotIn("Hogwarts House", features)
        self.assertNotIn("Index", features)

    def test_extract_valid_feature_values(self) -> None:
        """Verifies that extract_valid_feature_values filters out NaN values."""
        arithmancy_vals = extract_valid_feature_values(self.sample_df, "Arithmancy")
        self.assertEqual(arithmancy_vals, [10.0, 30.0, 40.0])

        astronomy_vals = extract_valid_feature_values(self.sample_df, "Astronomy")
        self.assertEqual(astronomy_vals, [-50.0, -20.0, 10.0])

    def test_extract_valid_feature_values_key_error(self) -> None:
        """Verifies that extract_valid_feature_values raises KeyError for invalid column."""
        with self.assertRaises(KeyError):
            extract_valid_feature_values(self.sample_df, "InvalidCourse")


if __name__ == "__main__":
    unittest.main()
