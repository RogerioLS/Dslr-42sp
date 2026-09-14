"""Unit Test Suite for Inference Engine and Prediction Utilities (DSLR-11).

Validates:
1. CSV prediction serialization format (Index, Hogwarts House).
2. End-to-end inference execution via run_prediction function.
3. Error handling on non-existent dataset, weights, or corrupt feature schemas.
4. Correctness of index extraction and student house classification.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from logreg_predict import run_prediction, save_predictions_csv
from src.model.multiclass import OneVsRestLogisticRegression


class TestPredictUnit(unittest.TestCase):
    """Unit tests for inference routines and CSV persistence in logreg_predict."""

    def setUp(self) -> None:
        """Creates temporary test dataset and minimal serialized weights."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Synthetic test data
        self.test_csv_path = self.dir_path / "test_data.csv"
        df = pd.DataFrame(
            {
                "Index": [0, 1, 2],
                "First Name": ["Harry", "Ron", "Hermione"],
                "feat_a": [1.0, 2.0, 3.0],
                "feat_b": [10.0, 20.0, 30.0],
            }
        )
        df.to_csv(self.test_csv_path, index=False)

        # Train a toy model to generate a valid weights.json
        train_df = pd.DataFrame(
            {
                "Index": [0, 1, 2, 3],
                "feat_a": [1.0, 1.1, 10.0, 10.2],
                "feat_b": [2.0, 2.1, 20.0, 20.2],
                "Hogwarts House": ["Gryffindor", "Gryffindor", "Slytherin", "Slytherin"],
            }
        )
        clf = OneVsRestLogisticRegression(
            learning_rate=0.1,
            epochs=20,
            features=["feat_a", "feat_b"],
            classes=["Gryffindor", "Slytherin"],
        )
        clf.fit(train_df, target_column="Hogwarts House")
        self.weights_path = self.dir_path / "weights.json"
        clf.save_weights(self.weights_path)

    def tearDown(self) -> None:
        """Cleans up temporary directory after tests."""
        self.temp_dir.cleanup()

    def test_save_predictions_csv_format(self) -> None:
        """Tests that save_predictions_csv creates the exact required CSV format."""
        out_path = self.dir_path / "houses.csv"
        indices = pd.Series([0, 1, 2])
        preds = ["Gryffindor", "Hufflepuff", "Ravenclaw"]

        save_predictions_csv(indices, preds, str(out_path))

        self.assertTrue(out_path.exists())
        df = pd.read_csv(out_path)
        self.assertListEqual(list(df.columns), ["Index", "Hogwarts House"])
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["Index"]), [0, 1, 2])
        self.assertEqual(list(df["Hogwarts House"]), preds)

    def test_run_prediction_success(self) -> None:
        """Tests that run_prediction executes successfully and generates houses.csv."""
        out_path = self.dir_path / "houses_out.csv"
        status = run_prediction(
            dataset_path=str(self.test_csv_path),
            weights_path=str(self.weights_path),
            output_path=str(out_path),
            quiet=True,
        )

        self.assertEqual(status, 0)
        self.assertTrue(out_path.exists())
        df = pd.read_csv(out_path)
        self.assertEqual(len(df), 3)
        self.assertIn("Hogwarts House", df.columns)

    def test_run_prediction_nonexistent_dataset(self) -> None:
        """Tests that run_prediction exits with code 1 if dataset file is missing."""
        status = run_prediction(
            dataset_path=str(self.dir_path / "non_existent.csv"),
            weights_path=str(self.weights_path),
            quiet=True,
        )
        self.assertEqual(status, 1)

    def test_run_prediction_nonexistent_weights(self) -> None:
        """Tests that run_prediction exits with code 1 if weights file is missing."""
        status = run_prediction(
            dataset_path=str(self.test_csv_path),
            weights_path=str(self.dir_path / "missing_weights.json"),
            quiet=True,
        )
        self.assertEqual(status, 1)

    def test_run_prediction_empty_dataset(self) -> None:
        """Tests that run_prediction exits with code 1 when given an empty CSV."""
        empty_csv = self.dir_path / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)

        status = run_prediction(
            dataset_path=str(empty_csv),
            weights_path=str(self.weights_path),
            quiet=True,
        )
        self.assertEqual(status, 1)

    def test_run_prediction_missing_features(self) -> None:
        """Tests that run_prediction exits with code 1 if required feature columns are missing."""
        bad_df = pd.DataFrame({"Index": [0], "other_col": [1.0]})
        bad_csv = self.dir_path / "bad.csv"
        bad_df.to_csv(bad_csv, index=False)

        status = run_prediction(
            dataset_path=str(bad_csv),
            weights_path=str(self.weights_path),
            quiet=True,
        )
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
