"""Logistic regression classification model package for 42 DSLR.

Exposes core vectorized mathematical operations, gradient solvers,
single-class binary estimators, and the One-vs-Rest multiclass classifier.
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
from src.model.multiclass import DEFAULT_HOGWARTS_HOUSES, OneVsRestLogisticRegression

__all__ = [
    "add_bias",
    "sigmoid",
    "compute_hypothesis",
    "compute_loss",
    "compute_gradient",
    "gradient_descent_step",
    "BinaryLogisticRegression",
    "OneVsRestLogisticRegression",
    "DEFAULT_HOGWARTS_HOUSES",
]
