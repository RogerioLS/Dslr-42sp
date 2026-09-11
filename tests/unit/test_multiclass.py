"""Unit Test Suite for Multiclass One-vs-Rest Logistic Regression (DSLR-10).

Validates:
1. OneVsRest model initialization, parameter validation, and class discovery.
2. Probability estimation and class prediction accuracy on synthetic multiclass data.
3. Serialization and deserialization of weights and scaler parameters via JSON.
4. Error handling on unfitted models and invalid inputs.
5. Training callback hook execution.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.multiclass import OneVsRestLogisticRegression


class TestOneVsRestLogisticRegression(unittest.TestCase):
    """Unit tests for the OneVsRestLogisticRegression classifier class."""

    def setUp(self) -> None:
        """Sets up a synthetic 4-class Gaussian cluster dataset."""
        np.random.seed(42)
        samples_per_class = 30

        # Create 4 distinct clusters in 2D space
        c1 = np.random.randn(samples_per_class, 2) + np.array([-4.0, -4.0])
        c2 = np.random.randn(samples_per_class, 2) + np.array([4.0, -4.0])
        c3 = np.random.randn(samples_per_class, 2) + np.array([-4.0, 4.0])
        c4 = np.random.randn(samples_per_class, 2) + np.array([4.0, 4.0])

        X = np.vstack((c1, c2, c3, c4))
        houses = (
            ["Gryffindor"] * samples_per_class
            + ["Hufflepuff"] * samples_per_class
            + ["Ravenclaw"] * samples_per_class
            + ["Slytherin"] * samples_per_class
        )

        self.df = pd.DataFrame(X, columns=["feat_1", "feat_2"])
        self.df["Hogwarts House"] = houses

    def test_init_validation(self) -> None:
        """Tests parameter validation during model initialization."""
        with self.assertRaises(ValueError):
            OneVsRestLogisticRegression(learning_rate=-0.1)

        with self.assertRaises(ValueError):
            OneVsRestLogisticRegression(epochs=0)

    def test_fit_and_multiclass_prediction(self) -> None:
        """Tests that OneVsRest fits all classes and achieves high accuracy on separable data."""
        clf = OneVsRestLogisticRegression(
            learning_rate=0.5,
            epochs=200,
            features=["feat_1", "feat_2"],
        )
        clf.fit(self.df, target_column="Hogwarts House")

        self.assertTrue(clf.is_fitted_)
        self.assertEqual(len(clf.models_), 4)
        for house in ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]:
            self.assertIn(house, clf.models_)

        probas = clf.predict_proba(self.df)
        self.assertEqual(probas.shape, (120, 4))
        self.assertTrue(np.all(probas >= 0.0))
        self.assertTrue(np.all(probas <= 1.0))

        preds = clf.predict(self.df)
        self.assertEqual(len(preds), 120)
        accuracy = (np.array(preds) == self.df["Hogwarts House"].values).mean()
        self.assertGreaterEqual(accuracy, 0.95)

    def test_training_callback_hook(self) -> None:
        """Tests that the training callback hook is invoked with expected arguments."""
        recorded_calls = []

        def hook(class_name: str, epoch: int, loss: float) -> None:
            recorded_calls.append((class_name, epoch, loss))

        clf = OneVsRestLogisticRegression(
            learning_rate=0.5,
            epochs=50,
            features=["feat_1", "feat_2"],
        )
        clf.fit(self.df, target_column="Hogwarts House", callback=hook)

        self.assertTrue(len(recorded_calls) > 0)
        class_names_reported = {c[0] for c in recorded_calls}
        self.assertEqual(
            class_names_reported,
            {"Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"},
        )

    def test_save_and_load_weights_roundtrip(self) -> None:
        """Tests saving weights to JSON and reconstructing an identical model."""
        clf = OneVsRestLogisticRegression(
            learning_rate=0.3,
            epochs=50,
            features=["feat_1", "feat_2"],
        )
        clf.fit(self.df, target_column="Hogwarts House")

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "test_weights.json"
            clf.save_weights(weights_path)

            self.assertTrue(weights_path.exists())

            with open(weights_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("classes", payload)
            self.assertIn("features", payload)
            self.assertIn("weights", payload)
            self.assertIn("scaler", payload)

            restored = OneVsRestLogisticRegression.load_weights(weights_path)
            self.assertTrue(restored.is_fitted_)
            self.assertEqual(restored.classes_, clf.classes_)
            self.assertEqual(restored.features_, clf.features_)

            orig_preds = clf.predict(self.df)
            rest_preds = restored.predict(self.df)
            self.assertEqual(orig_preds, rest_preds)

    def test_unfitted_model_errors(self) -> None:
        """Tests that invoking prediction or save on an unfitted model raises RuntimeError."""
        clf = OneVsRestLogisticRegression()
        with self.assertRaises(RuntimeError):
            clf.predict_proba(self.df)

        with self.assertRaises(RuntimeError):
            clf.predict(self.df)

        with self.assertRaises(RuntimeError):
            clf.save_weights("unfitted_weights.json")

    def test_missing_target_column_raises(self) -> None:
        """Tests that fitting on a DataFrame without the target column raises ValueError."""
        clf = OneVsRestLogisticRegression()
        with self.assertRaises(ValueError):
            clf.fit(self.df, target_column="NonExistentColumn")


if __name__ == "__main__":
    unittest.main()
