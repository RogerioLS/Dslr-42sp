"""Unit Test Suite for Logistic Regression Core Mathematical Foundations (DSLR-09).

Validates:
1. Bias addition and dimension augmentation.
2. Sigmoid activation axioms, symmetry, monotonicity, and numerical stability.
3. Hypothesis computation against linear algebra invariants.
4. Binary Cross-Entropy Loss (Log-Loss) edge cases and theoretical baselines.
5. Analytical gradient computation accuracy and shape consistency.
6. Parameter update rules for Batch Gradient Descent.
7. BinaryLogisticRegression fitting, loss decrease, predictions, and serialization.
"""

import unittest

import numpy as np

from src.model.logistic_regression import (
    BinaryLogisticRegression,
    add_bias,
    compute_gradient,
    compute_hypothesis,
    compute_loss,
    gradient_descent_step,
    sigmoid,
)


class TestLogisticRegressionMath(unittest.TestCase):
    """Test suite for handcrafted vectorized math functions in logistic regression."""

    def test_add_bias_success(self) -> None:
        """Tests that add_bias prepends a column of ones correctly."""
        X = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        X_augmented = add_bias(X)

        self.assertEqual(X_augmented.shape, (3, 3))
        # Column 0 must be all 1.0
        np.testing.assert_allclose(X_augmented[:, 0], np.ones(3))
        # Remaining columns must match original X
        np.testing.assert_allclose(X_augmented[:, 1:], X)

    def test_add_bias_validation_errors(self) -> None:
        """Tests that add_bias raises ValueError for non-2D or empty inputs."""
        with self.assertRaises(ValueError):
            add_bias(np.array([1.0, 2.0, 3.0]))  # 1D

        with self.assertRaises(ValueError):
            add_bias(np.empty((0, 2)))  # empty samples

        with self.assertRaises(ValueError):
            add_bias(np.empty((3, 0)))  # empty features

    def test_sigmoid_axioms_and_symmetry(self) -> None:
        """Validates fundamental sigmoid axioms: g(0) = 0.5 and g(-z) = 1 - g(z)."""
        # g(0) == 0.5
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5, places=10)

        # Symmetry property: g(-z) == 1 - g(z)
        z_vals = np.array([-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0])
        sig_pos = sigmoid(z_vals)
        sig_neg = sigmoid(-z_vals)
        np.testing.assert_allclose(sig_pos, 1.0 - sig_neg, atol=1e-12)

    def test_sigmoid_monotonicity(self) -> None:
        """Validates that sigmoid is strictly monotonically increasing."""
        z = np.linspace(-10.0, 10.0, 100)
        sig = sigmoid(z)
        self.assertTrue(np.all(np.diff(sig) > 0))

    def test_sigmoid_numerical_stability(self) -> None:
        """Tests extreme values to ensure no floating-point overflow occurs."""
        extreme_z = np.array([-1000.0, -500.0, 500.0, 1000.0])
        result = sigmoid(extreme_z)

        self.assertFalse(np.isnan(result).any())
        self.assertFalse(np.isinf(result).any())
        self.assertAlmostEqual(float(result[0]), 0.0, places=7)
        self.assertAlmostEqual(float(result[-1]), 1.0, places=7)

    def test_compute_hypothesis(self) -> None:
        """Validates linear combination and sigmoid mapping for hypothesis computation."""
        X = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, -2.0]])
        theta = np.array([0.0, 1.0])

        h = compute_hypothesis(X, theta)
        self.assertEqual(h.shape, (3,))
        self.assertAlmostEqual(h[0], 0.5)  # 0 * 1 + 0 * 0 = 0 -> g(0) = 0.5
        self.assertAlmostEqual(h[1], float(sigmoid(2.0)))
        self.assertAlmostEqual(h[2], float(sigmoid(-2.0)))

    def test_compute_hypothesis_dimension_mismatch(self) -> None:
        """Tests that hypothesis computation fails when dimensions do not align."""
        X = np.ones((5, 3))
        theta_wrong = np.ones(4)
        with self.assertRaises(ValueError):
            compute_hypothesis(X, theta_wrong)

    def test_compute_loss_baselines(self) -> None:
        """Tests Log-Loss calculation on known theoretical baselines."""
        # Baseline 1: Uninformative guess (p = 0.5 everywhere) -> Loss = ln(2) ~ 0.693147
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_half = np.array([0.5, 0.5, 0.5, 0.5])
        loss_half = compute_loss(y_true, y_half)
        self.assertAlmostEqual(loss_half, np.log(2.0), places=5)

        # Baseline 2: Perfect confidence -> Loss near 0
        y_perfect = np.array([0.0, 1.0, 0.0, 1.0])
        p_perfect = np.array([1e-12, 1.0 - 1e-12, 1e-12, 1.0 - 1e-12])
        loss_perfect = compute_loss(y_perfect, p_perfect)
        self.assertLess(loss_perfect, 1e-4)

        # Baseline 3: High penalty for confident wrong predictions
        p_wrong = np.array([0.999, 0.001, 0.999, 0.001])
        loss_wrong = compute_loss(y_true, p_wrong)
        self.assertGreater(loss_wrong, 5.0)

    def test_compute_loss_shape_mismatch(self) -> None:
        """Tests that compute_loss validates input length parity."""
        with self.assertRaises(ValueError):
            compute_loss(np.array([1.0, 0.0]), np.array([0.5]))

        with self.assertRaises(ValueError):
            compute_loss(np.array([]), np.array([]))

    def test_compute_gradient_zero_error(self) -> None:
        """Tests that zero prediction error yields a zero gradient vector."""
        X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([1.0, 0.0, 1.0])

        grad = compute_gradient(X, y_true, y_pred)
        self.assertEqual(grad.shape, (2,))
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-12)

    def test_compute_gradient_numerical_accuracy(self) -> None:
        """Tests analytical gradient against manual dot product formula."""
        X = np.array([[1.0, 1.0], [1.0, 2.0]])
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.8, 0.4])

        # errors = [0.8 - 1.0, 0.4 - 0.0] = [-0.2, 0.4]
        # grad_0 = (1/2) * (1 * -0.2 + 1 * 0.4) = 0.2 / 2 = 0.1
        # grad_1 = (1/2) * (1 * -0.2 + 2 * 0.4) = 0.6 / 2 = 0.3
        grad = compute_gradient(X, y_true, y_pred)
        np.testing.assert_allclose(grad, np.array([0.1, 0.3]), atol=1e-10)

    def test_gradient_descent_step(self) -> None:
        """Validates that gradient descent step subtracts alpha * grad correctly."""
        theta = np.array([1.0, 2.0])
        grad = np.array([0.5, -1.0])
        alpha = 0.1

        updated_theta = gradient_descent_step(theta, grad, alpha)
        # Expected: [1.0 - 0.1*0.5, 2.0 - 0.1*(-1.0)] = [0.95, 2.1]
        np.testing.assert_allclose(updated_theta, np.array([0.95, 2.1]), atol=1e-10)

        with self.assertRaises(ValueError):
            gradient_descent_step(theta, grad, -0.01)


class TestBinaryLogisticRegressionModel(unittest.TestCase):
    """Test suite for the BinaryLogisticRegression estimator class."""

    def setUp(self) -> None:
        """Sets up a linearly separable 2D synthetic dataset for classifier testing."""
        np.random.seed(42)
        # Class 0: centered around (-2, -2)
        X0 = np.random.randn(40, 2) - 2.0
        y0 = np.zeros(40)

        # Class 1: centered around (2, 2)
        X1 = np.random.randn(40, 2) + 2.0
        y1 = np.ones(40)

        self.X = np.vstack((X0, X1))
        self.y = np.hstack((y0, y1))

    def test_fit_and_loss_decrease(self) -> None:
        """Tests that model trains, loss strictly decreases, and achieves high accuracy."""
        clf = BinaryLogisticRegression(learning_rate=0.5, epochs=200, fit_intercept=True)
        clf.fit(self.X, self.y)

        self.assertTrue(clf.is_fitted_)
        self.assertIsNotNone(clf.weights_)
        self.assertEqual(len(clf.weights_), 3)  # bias + 2 features

        # Loss must have decreased significantly
        self.assertLess(clf.loss_history_[-1], clf.loss_history_[0])
        self.assertLess(clf.loss_history_[-1], 0.1)

        # Predictions must match ground truth on linearly separable data
        preds = clf.predict(self.X)
        accuracy = np.mean(preds == self.y)
        self.assertGreaterEqual(accuracy, 0.98)

    def test_predict_unfitted_raises(self) -> None:
        """Tests that calling predict on an unfitted model raises RuntimeError."""
        clf = BinaryLogisticRegression()
        with self.assertRaises(RuntimeError):
            clf.predict(self.X)

    def test_serialization_roundtrip(self) -> None:
        """Validates that to_dict and from_dict preserve exact weights and predictions."""
        clf = BinaryLogisticRegression(learning_rate=0.2, epochs=50, fit_intercept=True)
        clf.fit(self.X, self.y)

        serialized = clf.to_dict()
        restored = BinaryLogisticRegression.from_dict(serialized)

        self.assertTrue(restored.is_fitted_)
        np.testing.assert_allclose(restored.weights_, clf.weights_)

        probas_orig = clf.predict_proba(self.X)
        probas_rest = restored.predict_proba(self.X)
        np.testing.assert_allclose(probas_orig, probas_rest)


if __name__ == "__main__":
    unittest.main()
