"""Unit tests for src.analytics.statistics module (DSLR-02 and DSLR-03 Core Math).

Verifies mathematical accuracy of handcrafted statistical functions against
official ground truth and Pandas/NumPy reference outputs with 1e-6 precision tolerance.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.loader import extract_valid_feature_values, get_numerical_features, load_csv
from src.analytics.statistics import (
    compute_count,
    compute_max,
    compute_mean,
    compute_min,
    compute_percentile,
    compute_stats_summary,
    compute_std,
)


class TestStatisticsMath(unittest.TestCase):
    """Test suite for mathematical statistics computed from first principles."""

    def setUp(self) -> None:
        """Sets up test synthetic collections and dataset paths."""
        self.sample_data = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.even_data = [1.0, 2.0, 3.0, 4.0]
        self.mixed_data = [-15.5, 0.0, 42.123, -100.8, 88.9, 3.1415]
        self.single_data = [42.0]

        root_dir = Path(__file__).resolve().parent.parent.parent
        self.train_dataset_path = root_dir / "datasets" / "dataset_train.csv"

    def test_compute_count(self) -> None:
        """Verifies compute_count returns exact number of elements."""
        self.assertEqual(compute_count(self.sample_data), 5)
        self.assertEqual(compute_count(self.mixed_data), 6)
        self.assertEqual(compute_count(self.single_data), 1)
        self.assertEqual(compute_count([]), 0)

    def test_compute_mean(self) -> None:
        """Verifies compute_mean calculates exact arithmetic mean."""
        self.assertAlmostEqual(compute_mean(self.sample_data), 30.0, places=7)
        expected_mixed = float(np.mean(self.mixed_data))
        self.assertAlmostEqual(compute_mean(self.mixed_data), expected_mixed, places=7)

        with self.assertRaises(ValueError):
            compute_mean([])

    def test_compute_std_sample_bessel(self) -> None:
        """Verifies compute_std uses Bessel's correction (N - 1) matching Pandas."""
        expected_std = float(pd.Series(self.sample_data).std())
        self.assertAlmostEqual(compute_std(self.sample_data), expected_std, places=7)

        expected_mixed_std = float(pd.Series(self.mixed_data).std())
        self.assertAlmostEqual(compute_std(self.mixed_data), expected_mixed_std, places=7)

        with self.assertRaises(ValueError):
            compute_std(self.single_data)

        with self.assertRaises(ValueError):
            compute_std([])

    def test_compute_min_max(self) -> None:
        """Verifies compute_min and compute_max find absolute bounds."""
        self.assertEqual(compute_min(self.sample_data), 10.0)
        self.assertEqual(compute_max(self.sample_data), 50.0)

        self.assertEqual(compute_min(self.mixed_data), -100.8)
        self.assertEqual(compute_max(self.mixed_data), 88.9)

        with self.assertRaises(ValueError):
            compute_min([])
        with self.assertRaises(ValueError):
            compute_max([])

    def test_compute_percentile_basics(self) -> None:
        """Verifies percentiles for boundary conditions, single elements, and medians."""
        self.assertEqual(compute_percentile(self.single_data, 0.5), 42.0)
        self.assertEqual(compute_percentile(self.single_data, 0.0), 42.0)
        self.assertEqual(compute_percentile(self.single_data, 1.0), 42.0)

        self.assertEqual(compute_percentile(self.sample_data, 0.0), 10.0)
        self.assertEqual(compute_percentile(self.sample_data, 0.5), 30.0)
        self.assertEqual(compute_percentile(self.sample_data, 1.0), 50.0)

        # Even dataset [1.0, 2.0, 3.0, 4.0]: median is 2.5
        self.assertEqual(compute_percentile(self.even_data, 0.5), 2.5)

    def test_compute_percentile_against_pandas(self) -> None:
        """Verifies Method 7 interpolation against Pandas series.quantile()."""
        series = pd.Series(self.mixed_data)
        quantiles = [0.0, 0.1, 0.25, 0.33, 0.5, 0.75, 0.9, 1.0]

        for q in quantiles:
            expected = float(series.quantile(q))
            result = compute_percentile(self.mixed_data, q)
            self.assertAlmostEqual(result, expected, places=7)

    def test_compute_percentile_exceptions(self) -> None:
        """Verifies compute_percentile raises ValueError on invalid inputs."""
        with self.assertRaises(ValueError):
            compute_percentile([], 0.5)
        with self.assertRaises(ValueError):
            compute_percentile(self.sample_data, -0.1)
        with self.assertRaises(ValueError):
            compute_percentile(self.sample_data, 1.1)

    def test_compute_stats_summary(self) -> None:
        """Verifies compute_stats_summary returns all 8 required metrics."""
        summary = compute_stats_summary(self.sample_data)
        expected_keys = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        self.assertEqual(list(summary.keys()), expected_keys)
        self.assertEqual(summary["Count"], 5.0)
        self.assertEqual(summary["Mean"], 30.0)
        self.assertEqual(summary["Min"], 10.0)
        self.assertEqual(summary["25%"], 20.0)
        self.assertEqual(summary["50%"], 30.0)
        self.assertEqual(summary["75%"], 40.0)
        self.assertEqual(summary["Max"], 50.0)

    def test_dataset_train_all_features_ground_truth(self) -> None:
        """Validates handcrafted statistics across all 13 Hogwarts courses in real dataset."""
        df = load_csv(self.train_dataset_path)
        features = get_numerical_features(df)

        for feature in features:
            values = extract_valid_feature_values(df, feature)
            series = df[feature].dropna()

            self.assertEqual(compute_count(values), int(series.count()))
            self.assertAlmostEqual(compute_mean(values), float(series.mean()), places=5)
            self.assertAlmostEqual(compute_std(values), float(series.std()), places=5)
            self.assertAlmostEqual(compute_min(values), float(series.min()), places=5)
            self.assertAlmostEqual(
                compute_percentile(values, 0.25), float(series.quantile(0.25)), places=5
            )
            self.assertAlmostEqual(
                compute_percentile(values, 0.50), float(series.quantile(0.50)), places=5
            )
            self.assertAlmostEqual(
                compute_percentile(values, 0.75), float(series.quantile(0.75)), places=5
            )
            self.assertAlmostEqual(compute_max(values), float(series.max()), places=5)


if __name__ == "__main__":
    unittest.main()
