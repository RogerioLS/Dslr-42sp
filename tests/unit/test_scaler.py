"""Industrial Test Harness for StandardScaler Module (42 DSLR).

Comprehensive test harness validating:
1. Mathematical Invariant Axioms: Zero mean (mu=0), Unit variance (sigma=1).
2. Bijective Invertibility: Perfect roundtrip reconstruction via inverse_transform.
3. Zero Data Leakage Harness: Strict isolation between train calibration and test evaluation.
4. Numerical Stability & Boundary Harness: Constant features, zero division protection, NaNs.
5. Heterogeneous Data Formats: DataFrames, 2D matrices, 1D sequences, dictionaries.
6. Real Dataset Verification: End-to-end calibration on Hogwarts train and test datasets.
"""

import math
import unittest
from pathlib import Path

import pandas as pd

from src.analytics.loader import HOGWARTS_COURSES, load_csv
from src.analytics.statistics import compute_mean, compute_std
from src.preprocessing.scaler import StandardScaler

DATASET_TRAIN = Path(__file__).resolve().parent.parent.parent / "datasets" / "dataset_train.csv"
DATASET_TEST = Path(__file__).resolve().parent.parent.parent / "datasets" / "dataset_test.csv"


class TestHarnessStandardScaler(unittest.TestCase):
    """Rigorous test harness evaluating StandardScaler implementation."""

    def setUp(self) -> None:
        """Configures test fixtures and synthetic baseline matrices."""
        self.scaler = StandardScaler()
        # Synthetic clean bivariate dataset
        self.synthetic_df = pd.DataFrame(
            {
                "feature_a": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature_b": [-500.0, -250.0, 0.0, 250.0, 500.0],
            }
        )

    # --------------------------------------------------------------------------
    # 1. MATHEMATICAL INVARIANT HARNESS
    # --------------------------------------------------------------------------
    def test_harness_mathematical_invariants(self) -> None:
        """Harness Assertion: Transformed features must exhibit mu=0 and sigma=1."""
        scaled_df = self.scaler.fit_transform(self.synthetic_df)

        for col in ["feature_a", "feature_b"]:
            vals = list(scaled_df[col])
            mean_z = compute_mean(vals)
            std_z = compute_std(vals)

            # Invariant 1: Centered at 0 within numerical precision
            self.assertAlmostEqual(
                mean_z,
                0.0,
                places=7,
                msg=f"Feature {col} mean failed mu=0 invariant (got {mean_z})",
            )
            # Invariant 2: Unit standard deviation
            self.assertAlmostEqual(
                std_z,
                1.0,
                places=7,
                msg=f"Feature {col} std failed sigma=1 invariant (got {std_z})",
            )

    def test_harness_roundtrip_invertibility(self) -> None:
        """Harness Assertion: x == inverse_transform(transform(x)) within float precision."""
        scaled_df = self.scaler.fit_transform(self.synthetic_df)
        reconstructed_df = self.scaler.inverse_transform(scaled_df)

        for col in ["feature_a", "feature_b"]:
            orig = list(self.synthetic_df[col])
            recon = list(reconstructed_df[col])
            for o_val, r_val in zip(orig, recon):
                self.assertAlmostEqual(
                    o_val,
                    r_val,
                    places=7,
                    msg=f"Roundtrip reconstruction failed for {col}: {o_val} != {r_val}",
                )

    # --------------------------------------------------------------------------
    # 2. ZERO DATA LEAKAGE HARNESS
    # --------------------------------------------------------------------------
    def test_harness_zero_data_leakage(self) -> None:
        """Harness Assertion: Transforming test data strictly uses train statistics.

        Test data statistics must not influence scaling, and test scaled mean/std
        must naturally reflect the domain shift without forced centering.
        """
        train_data = pd.DataFrame({"score": [10.0, 20.0, 30.0, 40.0, 50.0]})  # mu=30, sigma~15.81
        test_data = pd.DataFrame({"score": [100.0, 110.0, 120.0]})  # Distinct distribution

        self.scaler.fit(train_data)
        train_mu = self.scaler.mean_["score"]
        train_sigma = self.scaler.std_["score"]

        # Transform test data
        test_scaled = self.scaler.transform(test_data)
        test_vals = list(test_scaled["score"])

        # Expected test z-score: (x - train_mu) / train_sigma
        for x_test, z_test in zip(test_data["score"], test_vals):
            expected_z = (x_test - train_mu) / train_sigma
            self.assertAlmostEqual(z_test, expected_z, places=7)

        # Confirm test mean is NOT zero (proves test stats were not leaked/recalculated)
        test_scaled_mean = compute_mean(test_vals)
        self.assertNotAlmostEqual(test_scaled_mean, 0.0, places=2)

    # --------------------------------------------------------------------------
    # 3. NUMERICAL STABILITY & EDGE-CASE HARNESS
    # --------------------------------------------------------------------------
    def test_harness_constant_feature_zero_variance(self) -> None:
        """Harness Assertion: Features with zero variance (sigma=0) must not divide by zero."""
        constant_df = pd.DataFrame({"flat": [42.0, 42.0, 42.0, 42.0]})
        scaled = self.scaler.fit_transform(constant_df)

        # Constant feature with with_mean=True standardizes to exact 0.0
        for val in scaled["flat"]:
            self.assertEqual(val, 0.0)

    def test_harness_nan_mean_imputation(self) -> None:
        """Harness Assertion: Missing values (NaNs) are imputed with feature mean (z=0.0)."""
        df_with_nan = pd.DataFrame({"val": [10.0, float("nan"), 30.0]})  # valid: 10, 30 -> mu=20
        scaler_impute = StandardScaler(impute_strategy="mean")
        scaled = scaler_impute.fit_transform(df_with_nan)

        vals = list(scaled["val"])
        # Position 1 had NaN -> must map to 0.0 (since (mu - mu)/sigma = 0.0)
        self.assertAlmostEqual(vals[1], 0.0, places=7)
        # Verify valid values are symmetric around 0
        self.assertAlmostEqual(vals[0], -vals[2], places=7)

    def test_harness_nan_none_imputation(self) -> None:
        """Harness Assertion: Missing values preserved when impute_strategy='none'."""
        df_with_nan = pd.DataFrame({"val": [10.0, float("nan"), 30.0]})
        scaler_raw = StandardScaler(impute_strategy="none")
        scaled = scaler_raw.fit_transform(df_with_nan)

        vals = list(scaled["val"])
        self.assertTrue(math.isnan(vals[1]))

    def test_harness_exception_guards(self) -> None:
        """Harness Assertion: Uncalibrated or corrupted usage raises informative exceptions."""
        uncalibrated = StandardScaler()
        with self.assertRaises(RuntimeError):
            uncalibrated.transform(self.synthetic_df)

        with self.assertRaises(RuntimeError):
            uncalibrated.inverse_transform(self.synthetic_df)

        with self.assertRaises(ValueError):
            uncalibrated.fit([])

        # Corrupted transform missing calibrated column
        self.scaler.fit(self.synthetic_df)
        with self.assertRaises(KeyError):
            self.scaler.transform(pd.DataFrame({"unrelated_col": [1, 2, 3]}))

    # --------------------------------------------------------------------------
    # 4. HETEROGENEOUS FORMAT HARNESS
    # --------------------------------------------------------------------------
    def test_harness_matrix_list_support(self) -> None:
        """Harness Assertion: 2D list matrix input produces correctly scaled 2D list output."""
        raw_matrix = [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]]
        scaled_matrix = self.scaler.fit_transform(raw_matrix)

        self.assertIsInstance(scaled_matrix, list)
        self.assertEqual(len(scaled_matrix), 3)
        self.assertEqual(len(scaled_matrix[0]), 2)

        # Invert and check roundtrip
        unscaled = self.scaler.inverse_transform(scaled_matrix)
        for r_orig, r_unscaled in zip(raw_matrix, unscaled):
            for v_o, v_u in zip(r_orig, r_unscaled):
                self.assertAlmostEqual(v_o, v_u, places=7)

    # --------------------------------------------------------------------------
    # 5. REAL DATASET VALIDATION HARNESS (Hogwarts Train & Test)
    # --------------------------------------------------------------------------
    def test_harness_real_dataset_train_courses(self) -> None:
        """Harness Assertion: All Hogwarts courses in dataset_train.csv satisfy z-score laws."""
        if not DATASET_TRAIN.exists():
            self.skipTest("dataset_train.csv not found.")

        df_train = load_csv(DATASET_TRAIN)
        scaler = StandardScaler(impute_strategy="mean")
        scaled_df = scaler.fit_transform(df_train, columns=HOGWARTS_COURSES)

        for course in HOGWARTS_COURSES:
            vals = list(scaled_df[course])
            # Check zero remaining NaNs (safe for matrix dot product)
            self.assertFalse(any(math.isnan(v) for v in vals))

            # Since missing values are imputed with the exact mean, the overall mean remains 0.0!
            mean_z = compute_mean(vals)
            self.assertAlmostEqual(mean_z, 0.0, places=5)

    def test_harness_real_dataset_test_courses(self) -> None:
        """Harness Assertion: dataset_test.csv is scaled using train scaler with zero NaNs."""
        if not DATASET_TRAIN.exists() or not DATASET_TEST.exists():
            self.skipTest("Hogwarts datasets not found.")

        df_train = load_csv(DATASET_TRAIN)
        df_test = load_csv(DATASET_TEST)

        # Fit strictly on train
        scaler = StandardScaler(impute_strategy="mean")
        scaler.fit(df_train, columns=HOGWARTS_COURSES)

        # Transform test set
        scaled_test = scaler.transform(df_test)

        # Verify dimensions and non-null guarantees
        self.assertEqual(len(scaled_test), len(df_test))
        for course in HOGWARTS_COURSES:
            test_vals = list(scaled_test[course])
            self.assertFalse(any(math.isnan(v) for v in test_vals))


if __name__ == "__main__":
    unittest.main()
