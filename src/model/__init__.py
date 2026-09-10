"""Logistic regression classification model package for 42 DSLR.

Exposes core vectorized mathematical operations, gradient solvers,
and single-class binary logistic regression estimators.
"""

from src.model.logistic_regression import (
    BinaryLogisticRegression,
    add_bias,
    compute_gradient,
    compute_hypothesis,
    compute_loss,
    gradient_descent_step,
    sigmoid,
)

__all__ = [
    "add_bias",
    "sigmoid",
    "compute_hypothesis",
    "compute_loss",
    "compute_gradient",
    "gradient_descent_step",
    "BinaryLogisticRegression",
]
