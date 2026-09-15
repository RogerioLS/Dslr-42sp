"""Core Mathematical Foundations for Logistic Regression (42 DSLR).

Implements vectorized, numerically stable primitives for:
1. Bias addition (intercept column).
2. Sigmoid activation function with overflow clipping.
3. Linear hypothesis evaluation: h_theta(X) = g(X * theta).
4. Binary Cross-Entropy Loss (Log-Loss) with epsilon smoothing.
5. Analytical Cost Gradient vector: nabla J(theta) = (1/m) * X^T * (h - y).
6. Single-step Batch Gradient Descent update.
7. BinaryLogisticRegression estimator for One-vs-Rest classification engines.

Adheres strictly to École 42 standards: 100% handcrafted math without external ML libraries.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepends an intercept column of ones (X_0 = 1) to the feature matrix.

    Args:
        X (np.ndarray): 2D feature matrix of shape (m_samples, n_features).

    Returns:
        np.ndarray: Augmented matrix of shape (m_samples, n_features + 1).

    Raises:
        ValueError: If X is not a 2D array or is empty.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Input feature matrix must be 2-dimensional, got {arr.ndim}D.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Cannot add bias to an empty feature matrix.")

    ones = np.ones((arr.shape[0], 1), dtype=float)
    return np.hstack((ones, arr))


def sigmoid(z: Union[float, np.ndarray]) -> np.ndarray:
    """Computes the numerically stable sigmoid activation function g(z).

    Clips values to the range [-250.0, 250.0] to prevent 64-bit floating-point
    exponential overflow in exp(-z).

    Formula:
        g(z) = 1 / (1 + e^(-z))

    Args:
        z (Union[float, np.ndarray]): Input scalars, vectors, or matrices.

    Returns:
        np.ndarray: Transformed values in the open interval (0, 1).
    """
    arr = np.asarray(z, dtype=float)
    # Clip to avoid overflow in exp(-z)
    z_clipped = np.clip(arr, -250.0, 250.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def compute_hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Computes the logistic hypothesis probabilities h_theta(X) = sigmoid(X * theta).

    Args:
        X (np.ndarray): Design matrix of shape (m_samples, d_features).
        theta (np.ndarray): Weight vector of shape (d_features,) or (d_features, 1).

    Returns:
        np.ndarray: Predicted probability vector of shape (m_samples,).

    Raises:
        ValueError: If dimensions of X and theta do not align for dot product.
    """
    X_arr = np.asarray(X, dtype=float)
    th_arr = np.asarray(theta, dtype=float).ravel()

    if X_arr.ndim != 2:
        raise ValueError(f"Feature matrix X must be 2D, got shape {X_arr.shape}.")
    if X_arr.shape[1] != th_arr.shape[0]:
        raise ValueError(
            f"Dimension mismatch: X has {X_arr.shape[1]} columns, "
            f"but theta has {th_arr.shape[0]} elements."
        )

    linear_comb = np.dot(X_arr, th_arr)
    return sigmoid(linear_comb)


def compute_loss(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """Calculates the Binary Cross-Entropy (Log-Loss) cost J(theta).

    Formula:
        J(theta) = - (1 / m) * sum( y * log(p + eps) + (1 - y) * log(1 - p + eps) )

    Args:
        y_true (np.ndarray): Binary ground truth labels (0 or 1) of shape (m_samples,).
        y_pred_proba (np.ndarray): Predicted probabilities in (0, 1) of shape (m_samples,).
        epsilon (float): Small constant to avoid log(0) numerical instability.

    Returns:
        float: Non-negative binary cross-entropy loss value.

    Raises:
        ValueError: If shapes mismatch or arrays are empty.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred_proba, dtype=float).ravel()

    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError(f"Length mismatch: y_true ({y_t.shape[0]}) vs y_pred ({y_p.shape[0]}).")
    if y_t.shape[0] == 0:
        raise ValueError("Cannot compute loss on empty vectors.")

    # Clip probabilities for numeric stability
    p = np.clip(y_p, epsilon, 1.0 - epsilon)
    m = float(y_t.shape[0])

    loss = -(1.0 / m) * np.sum(y_t * np.log(p) + (1.0 - y_t) * np.log(1.0 - p))
    return float(loss)


def compute_gradient(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> np.ndarray:
    """Calculates the exact analytical gradient vector of the Log-Loss cost function.

    Formula:
        nabla J(theta) = (1 / m) * X^T * (h_theta(X) - y)

    Args:
        X (np.ndarray): Design matrix of shape (m_samples, d_features).
        y_true (np.ndarray): Binary ground truth labels of shape (m_samples,).
        y_pred_proba (np.ndarray): Predicted hypothesis probabilities of shape (m_samples,).

    Returns:
        np.ndarray: Gradient vector of shape (d_features,).

    Raises:
        ValueError: If shapes are inconsistent.
    """
    X_arr = np.asarray(X, dtype=float)
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred_proba, dtype=float).ravel()

    m = float(X_arr.shape[0])
    if m == 0:
        raise ValueError("Cannot compute gradient with zero samples.")
    if X_arr.ndim != 2:
        raise ValueError(f"Feature matrix X must be 2D, got shape {X_arr.shape}.")
    if y_t.shape[0] != m or y_p.shape[0] != m:
        raise ValueError("Sample count in X does not match target vector lengths.")

    errors = y_p - y_t
    gradient = (1.0 / m) * np.dot(X_arr.T, errors)
    return gradient


def gradient_descent_step(
    theta: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    """Executes a single parameter update step according to Batch Gradient Descent.

    Formula:
        theta := theta - alpha * nabla J(theta)

    Args:
        theta (np.ndarray): Current weight vector of shape (d,).
        gradient (np.ndarray): Gradient vector of shape (d,).
        learning_rate (float): Learning rate alpha (> 0).

    Returns:
        np.ndarray: Updated weight vector of shape (d,).

    Raises:
        ValueError: If dimensions mismatch or learning rate is non-positive.
    """
    if learning_rate <= 0.0:
        raise ValueError(f"Learning rate must be positive, got {learning_rate}.")

    th_arr = np.asarray(theta, dtype=float).ravel()
    grad_arr = np.asarray(gradient, dtype=float).ravel()

    if th_arr.shape[0] != grad_arr.shape[0]:
        raise ValueError(
            f"Shape mismatch: theta has {th_arr.shape[0]} elements, "
            f"gradient has {grad_arr.shape[0]}."
        )

    return th_arr - (learning_rate * grad_arr)


class BinaryLogisticRegression:
    """Single-class Binary Logistic Regression classifier with configurable optimizers.

    Supports:
        - 'batch': Batch Gradient Descent (exact gradient over all m samples).
        - 'sgd': Stochastic Gradient Descent (updates weights per single random observation).
        - 'minibatch': Mini-Batch Gradient Descent (updates weights over small random batches).

    Estimates weights theta minimizing Binary Cross-Entropy Loss via analytical gradients.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 1000,
        fit_intercept: bool = True,
        tolerance: float = 1e-7,
        method: str = "batch",
        batch_size: int = 32,
        random_state: Optional[int] = 42,
    ) -> None:
        """Initializes the BinaryLogisticRegression model.

        Args:
            learning_rate (float): Step size alpha for gradient descent. Defaults to 0.1.
            epochs (int): Maximum number of training iterations. Defaults to 1000.
            fit_intercept (bool): If True, automatically prepends a bias column of 1s.
            tolerance (float): Convergence threshold on loss change. Defaults to 1e-7.
            method (str): Optimization method ('batch', 'sgd', 'minibatch'). Defaults to 'batch'.
            batch_size (int): Size of batches when method is 'minibatch'. Defaults to 32.
            random_state (Optional[int]): Random seed for shuffling in SGD/Mini-Batch.
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
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance
        self.method = method.lower()
        self.batch_size = 1 if self.method == "sgd" else batch_size
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryLogisticRegression":
        """Fits binary logistic regression weights on training data (X, y).

        Args:
            X (np.ndarray): Feature matrix of shape (m_samples, n_features).
            y (np.ndarray): Binary labels (0 or 1) of shape (m_samples,).

        Returns:
            BinaryLogisticRegression: The fitted estimator instance.
        """
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=float).ravel()

        if self.fit_intercept:
            X_mat = add_bias(X_mat)

        m_samples, d_features = X_mat.shape
        self.weights_ = np.zeros(d_features, dtype=float)
        self.loss_history_ = []

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(m_samples)

        effective_batch_size = (
            1
            if self.method == "sgd"
            else (self.batch_size if self.method == "minibatch" else m_samples)
        )

        for epoch in range(self.epochs):
            # Compute full epoch loss for tracking & convergence
            y_pred_full = compute_hypothesis(X_mat, self.weights_)
            loss = compute_loss(y_vec, y_pred_full)
            self.loss_history_.append(loss)

            if epoch > 0 and abs(self.loss_history_[-2] - loss) < self.tolerance:
                break

            if self.method == "batch":
                grad = compute_gradient(X_mat, y_vec, y_pred_full)
                self.weights_ = gradient_descent_step(self.weights_, grad, self.learning_rate)
            else:
                rng.shuffle(indices)
                for start_idx in range(0, m_samples, effective_batch_size):
                    end_idx = min(start_idx + effective_batch_size, m_samples)
                    batch_idx = indices[start_idx:end_idx]

                    X_batch = X_mat[batch_idx]
                    y_batch = y_vec[batch_idx]
                    y_pred_batch = compute_hypothesis(X_batch, self.weights_)

                    grad = compute_gradient(X_batch, y_batch, y_pred_batch)
                    self.weights_ = gradient_descent_step(self.weights_, grad, self.learning_rate)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimates positive class posterior probability P(Y=1 | X).

        Args:
            X (np.ndarray): Feature matrix of shape (m_samples, n_features).

        Returns:
            np.ndarray: Predicted probability vector of shape (m_samples,).

        Raises:
            RuntimeError: If called before model has been fitted.
        """
        if not self.is_fitted_ or self.weights_ is None:
            raise RuntimeError("Cannot predict with an unfitted BinaryLogisticRegression model.")

        X_mat = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X_mat = add_bias(X_mat)

        return compute_hypothesis(X_mat, self.weights_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Classifies samples into discrete binary classes (0 or 1) by decision threshold.

        Args:
            X (np.ndarray): Feature matrix of shape (m_samples, n_features).
            threshold (float): Decision threshold in (0, 1). Defaults to 0.5.

        Returns:
            np.ndarray: Integer class label array (0 or 1) of shape (m_samples,).
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes model parameters and learned weights to a dictionary.

        Returns:
            Dict[str, Any]: Serialized representation.
        """
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "fit_intercept": self.fit_intercept,
            "tolerance": self.tolerance,
            "method": self.method,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted_,
            "weights": self.weights_.tolist() if self.weights_ is not None else None,
            "loss_history": self.loss_history_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BinaryLogisticRegression":
        """Reconstructs a BinaryLogisticRegression estimator from a parameter dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing serialized model configuration.

        Returns:
            BinaryLogisticRegression: Reconstructed model instance.
        """
        model = cls(
            learning_rate=data.get("learning_rate", 0.1),
            epochs=data.get("epochs", 1000),
            fit_intercept=data.get("fit_intercept", True),
            tolerance=data.get("tolerance", 1e-7),
            method=data.get("method", "batch"),
            batch_size=data.get("batch_size", 32),
            random_state=data.get("random_state", 42),
        )
        model.is_fitted_ = data.get("is_fitted", False)
        weights_raw = data.get("weights")
        if weights_raw is not None:
            model.weights_ = np.asarray(weights_raw, dtype=float)
        model.loss_history_ = data.get("loss_history", [])
        return model
