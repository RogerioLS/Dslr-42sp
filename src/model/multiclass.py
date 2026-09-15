"""Multiclass One-vs-Rest (OvR) Logistic Regression Classifier (42 DSLR).

Coordinates multiple binary logistic regression classifiers (one per Hogwarts house)
and unifies feature standardization, training convergence, and weight persistence.
Adheres strictly to École 42 standards: 100% handcrafted implementation
without external ML libraries.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.model.logistic_regression import BinaryLogisticRegression
from src.preprocessing.scaler import StandardScaler

DEFAULT_HOGWARTS_HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


class OneVsRestLogisticRegression:
    """Multiclass classifier implementing One-vs-Rest (OvR) Logistic Regression.

    Trains K independent binary logistic regression models for K mutually exclusive classes.
    For each class k, samples of class k are labeled 1 and all other samples are labeled 0.
    Predictions choose the class maximizing positive posterior probability:
        y_pred = argmax_k P(Y = k | X)
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int = 2000,
        tolerance: float = 1e-7,
        method: str = "batch",
        batch_size: int = 32,
        random_state: Optional[int] = 42,
        features: Optional[Sequence[str]] = None,
        classes: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the One-vs-Rest logistic regression engine.

        Args:
            learning_rate (float): Learning rate alpha for gradient descent. Defaults to 0.5.
            epochs (int): Number of training epochs. Defaults to 2000.
            tolerance (float): Convergence threshold on loss change. Defaults to 1e-7.
            method (str): Optimization method ('batch', 'sgd', 'minibatch'). Defaults to 'batch'.
            batch_size (int): Size of batches when method is 'minibatch'. Defaults to 32.
            random_state (Optional[int]): Random seed for reproducibility in SGD/Mini-Batch.
            features (Optional[Sequence[str]]): Specific feature column names to train on.
            classes (Optional[Sequence[str]]): Expected target classes. Defaults to Hogwarts houses.
        """
        if learning_rate <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}.")
        if epochs <= 0:
            raise ValueError(f"Epochs must be a positive integer, got {epochs}.")

        valid_methods = {"batch", "sgd", "minibatch"}
        if method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid optimization method '{method}'. Choose from {sorted(valid_methods)}."
            )

        if method.lower() == "minibatch" and batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer, got {batch_size}.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.method = method.lower()
        self.batch_size = 1 if self.method == "sgd" else batch_size
        self.random_state = random_state
        self.features_: List[str] = list(features) if features is not None else []
        self.classes_: List[str] = list(classes) if classes is not None else DEFAULT_HOGWARTS_HOUSES

        self.scaler_: StandardScaler = StandardScaler(with_mean=True, with_std=True)
        self.models_: Dict[str, BinaryLogisticRegression] = {}
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str = "Hogwarts House",
        callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> "OneVsRestLogisticRegression":
        """Fits the One-vs-Rest classifier on a tabular training dataset.

        Args:
            df (pd.DataFrame): Training DataFrame containing features and target column.
            target_column (str): Name of the target column containing class labels.
            callback (Optional[Callable[[str, int, float], None]]): Optional progress callback
                called with (class_name, epoch, loss).

        Returns:
            OneVsRestLogisticRegression: The fitted estimator instance.

        Raises:
            ValueError: If target column is missing or valid features cannot be identified.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in training dataset.")

        # Clean target: remove rows with null target
        valid_mask = df[target_column].notna()
        df_clean = df[valid_mask].reset_index(drop=True)

        if len(df_clean) == 0:
            raise ValueError("Training dataset has no samples with valid target labels.")

        # Determine features if not already set
        if not self.features_:
            metadata_cols = {
                "Index",
                "Hogwarts House",
                "First Name",
                "Last Name",
                "Birthday",
                "Best Hand",
            }
            self.features_ = [
                col
                for col in df_clean.columns
                if col not in metadata_cols and pd.api.types.is_numeric_dtype(df_clean[col])
            ]

        if not self.features_:
            raise ValueError("No numeric feature columns found for model training.")

        # Fit feature standardization on training data
        scaled_df = self.scaler_.fit_transform(df_clean, columns=self.features_)
        X_mat = np.asarray(scaled_df[self.features_].values, dtype=float)

        # Determine target classes
        actual_classes = sorted(df_clean[target_column].unique().tolist())
        if actual_classes:
            self.classes_ = actual_classes

        # Train one binary classifier per class
        self.models_ = {}
        y_raw = df_clean[target_column].values

        for idx_c, class_name in enumerate(self.classes_):
            y_binary = (y_raw == class_name).astype(float)
            # Create deterministic distinct seed per house class if random_state given
            seed = (self.random_state + idx_c) if self.random_state is not None else None

            clf = BinaryLogisticRegression(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                fit_intercept=True,
                tolerance=self.tolerance,
                method=self.method,
                batch_size=self.batch_size,
                random_state=seed,
            )

            # Manual training loop if callback is provided, else use clf.fit()
            if callback:
                self._train_with_callback(clf, X_mat, y_binary, class_name, callback)
            else:
                clf.fit(X_mat, y_binary)

            self.models_[class_name] = clf

        self.is_fitted_ = True
        return self

    def _train_with_callback(
        self,
        clf: BinaryLogisticRegression,
        X_mat: np.ndarray,
        y_binary: np.ndarray,
        class_name: str,
        callback: Callable[[str, int, float], None],
    ) -> None:
        """Trains a single binary classifier while firing progress callbacks.

        Args:
            clf (BinaryLogisticRegression): The binary classifier instance.
            X_mat (np.ndarray): Scaled design matrix.
            y_binary (np.ndarray): Binary targets vector.
            class_name (str): The class name being fitted.
            callback (Callable[[str, int, float], None]): Progress hook.
        """
        from src.model.logistic_regression import (
            add_bias,
            compute_gradient,
            compute_hypothesis,
            compute_loss,
            gradient_descent_step,
        )

        X_aug = add_bias(X_mat)
        m_samples = X_aug.shape[0]
        clf.weights_ = np.zeros(X_aug.shape[1], dtype=float)
        clf.loss_history_ = []

        report_interval = max(1, self.epochs // 10)
        rng = np.random.RandomState(clf.random_state)
        indices = np.arange(m_samples)
        eff_batch_size = (
            1
            if clf.method == "sgd"
            else (clf.batch_size if clf.method == "minibatch" else m_samples)
        )

        for epoch in range(self.epochs):
            y_pred_full = compute_hypothesis(X_aug, clf.weights_)
            loss = compute_loss(y_binary, y_pred_full)
            clf.loss_history_.append(loss)

            if epoch % report_interval == 0 or epoch == self.epochs - 1:
                callback(class_name, epoch + 1, loss)

            if epoch > 0 and abs(clf.loss_history_[-2] - loss) < clf.tolerance:
                break

            if clf.method == "batch":
                grad = compute_gradient(X_aug, y_binary, y_pred_full)
                clf.weights_ = gradient_descent_step(clf.weights_, grad, clf.learning_rate)
            else:
                rng.shuffle(indices)
                for start_idx in range(0, m_samples, eff_batch_size):
                    end_idx = min(start_idx + eff_batch_size, m_samples)
                    b_idx = indices[start_idx:end_idx]
                    X_b = X_aug[b_idx]
                    y_b = y_binary[b_idx]
                    y_pred_b = compute_hypothesis(X_b, clf.weights_)
                    grad = compute_gradient(X_b, y_b, y_pred_b)
                    clf.weights_ = gradient_descent_step(clf.weights_, grad, clf.learning_rate)

        clf.is_fitted_ = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Estimates class posterior probabilities P(Y = k | X) for all classes.

        Args:
            df (pd.DataFrame): Input dataset containing the feature columns.

        Returns:
            np.ndarray: Probability matrix of shape (m_samples, n_classes).

        Raises:
            RuntimeError: If model is not yet fitted.
        """
        if not self.is_fitted_ or not self.models_:
            raise RuntimeError("OneVsRestLogisticRegression must be fitted before predicting.")

        scaled_df = self.scaler_.transform(df)
        X_mat = np.asarray(scaled_df[self.features_].values, dtype=float)

        probas = np.column_stack([self.models_[cls].predict_proba(X_mat) for cls in self.classes_])
        return probas

    def predict(self, df: pd.DataFrame) -> List[str]:
        """Classifies samples by assigning the class with the highest probability.

        Args:
            df (pd.DataFrame): Input dataset containing the feature columns.

        Returns:
            List[str]: Predicted class name for each input sample.
        """
        probas = self.predict_proba(df)
        best_indices = np.argmax(probas, axis=1)
        return [self.classes_[idx] for idx in best_indices]

    def save_weights(self, filepath: Union[str, Path]) -> None:
        """Serializes learned weights and scaler parameters to a JSON file.

        Args:
            filepath (Union[str, Path]): Target output file path.

        Raises:
            RuntimeError: If called before model has been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save weights of an unfitted OneVsRest model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "version": "1.0.0",
            "model_type": "OneVsRestLogisticRegression",
            "classes": self.classes_,
            "features": self.features_,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "tolerance": self.tolerance,
            },
            "scaler": self.scaler_.to_dict(),
            "weights": {
                class_name: model.weights_.tolist()
                for class_name, model in self.models_.items()
                if model.weights_ is not None
            },
            "final_losses": {
                class_name: model.loss_history_[-1]
                for class_name, model in self.models_.items()
                if model.loss_history_
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_weights(cls, filepath: Union[str, Path]) -> "OneVsRestLogisticRegression":
        """Loads and reconstructs a OneVsRest model from a weights JSON file.

        Args:
            filepath (Union[str, Path]): Path to the weights JSON file.

        Returns:
            OneVsRestLogisticRegression: Reconstructed and calibrated model instance.

        Raises:
            FileNotFoundError: If the weights file does not exist.
            ValueError: If the file is invalid or missing required keys.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found at: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        required_keys = {"classes", "features", "weights", "scaler"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(f"Corrupt weights file, missing required keys: {missing}")

        hparams = payload.get("hyperparameters", {})
        instance = cls(
            learning_rate=hparams.get("learning_rate", 0.5),
            epochs=hparams.get("epochs", 2000),
            tolerance=hparams.get("tolerance", 1e-7),
            features=payload["features"],
            classes=payload["classes"],
        )

        instance.scaler_ = StandardScaler.from_dict(payload["scaler"])

        instance.models_ = {}
        for class_name in instance.classes_:
            w_list = payload["weights"].get(class_name)
            if w_list is None:
                raise ValueError(f"Missing weights entry for class '{class_name}'.")

            model = BinaryLogisticRegression(
                learning_rate=instance.learning_rate,
                epochs=instance.epochs,
                fit_intercept=True,
                tolerance=instance.tolerance,
            )
            model.weights_ = np.asarray(w_list, dtype=float)
            model.is_fitted_ = True
            if "final_losses" in payload and class_name in payload["final_losses"]:
                model.loss_history_ = [payload["final_losses"][class_name]]

            instance.models_[class_name] = model

        instance.is_fitted_ = True
        return instance
