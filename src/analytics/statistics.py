"""Handcrafted mathematical statistics module for 42 DSLR.

Implements fundamental descriptive statistics from first principles without
using any external statistical libraries or built-in pandas/numpy aggregators.
Adheres strictly to the 42 Norm and Anti-Cheating protocols.
"""

from typing import Sequence


def compute_count(values: Sequence[float]) -> int:
    """Computes the total count of valid numerical observations.

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        int: Number of elements in the collection.
    """
    total = 0
    for _ in values:
        total += 1
    return total


def compute_mean(values: Sequence[float]) -> float:
    """Computes the arithmetic mean (expected value) from scratch.

    Formula:
        mu = (1 / N) * sum(x_i)

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        float: Arithmetic mean of the values.

    Raises:
        ValueError: If the input collection is empty.
    """
    n = compute_count(values)
    if n == 0:
        raise ValueError("Cannot compute mean of an empty collection.")

    total_sum = 0.0
    for val in values:
        total_sum += float(val)

    return total_sum / n


def compute_std(values: Sequence[float]) -> float:
    """Computes the sample standard deviation with Bessel's correction (N - 1).

    Formula:
        sigma = sqrt( (1 / (N - 1)) * sum((x_i - mu)^2) )

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        float: Sample standard deviation.

    Raises:
        ValueError: If the input collection has fewer than 2 elements.
    """
    n = compute_count(values)
    if n < 2:
        raise ValueError("Sample standard deviation requires at least 2 observations.")

    mu = compute_mean(values)
    sum_squared_diff = 0.0
    for val in values:
        diff = float(val) - mu
        sum_squared_diff += diff * diff

    variance = sum_squared_diff / (n - 1)
    return variance**0.5


def compute_min(values: Sequence[float]) -> float:
    """Finds the minimum value through handcrafted linear scan.

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        float: Smallest numerical value found.

    Raises:
        ValueError: If the input collection is empty.
    """
    n = compute_count(values)
    if n == 0:
        raise ValueError("Cannot compute minimum of an empty collection.")

    smallest = float(values[0])
    for val in values:
        f_val = float(val)
        if f_val < smallest:
            smallest = f_val

    return smallest


def compute_max(values: Sequence[float]) -> float:
    """Finds the maximum value through handcrafted linear scan.

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        float: Largest numerical value found.

    Raises:
        ValueError: If the input collection is empty.
    """
    n = compute_count(values)
    if n == 0:
        raise ValueError("Cannot compute maximum of an empty collection.")

    largest = float(values[0])
    for val in values:
        f_val = float(val)
        if f_val > largest:
            largest = f_val

    return largest
