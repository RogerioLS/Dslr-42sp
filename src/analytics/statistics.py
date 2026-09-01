"""Handcrafted mathematical statistics module for 42 DSLR.

Implements fundamental descriptive statistics from first principles without
using any external statistical libraries or built-in pandas/numpy aggregators.
Adheres strictly to the 42 Norm and Anti-Cheating protocols.
"""

from typing import Dict, Sequence


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


def compute_percentile(values: Sequence[float], p: float) -> float:
    """Computes quantile percentile using continuous linear interpolation (Method 7).

    Follows the canonical Method 7 formulation (the default standard in Pandas & NumPy):
        idx = p * (N - 1)
        k = floor(idx)
        d = idx - k
        Q(p) = x_(k) + d * (x_(k+1) - x_(k))

    Args:
        values (Sequence[float]): Collection of non-null numerical values.
        p (float): Quantile proportion between 0.0 and 1.0 (e.g. 0.25, 0.50, 0.75).

    Returns:
        float: Interpolated quantile value.

    Raises:
        ValueError: If the collection is empty or p is outside [0.0, 1.0].
    """
    n = compute_count(values)
    if n == 0:
        raise ValueError("Cannot compute percentile of an empty collection.")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Percentile p must be between 0.0 and 1.0, got {p}.")

    if n == 1:
        return float(values[0])

    sorted_vals = sorted(float(val) for val in values)
    idx = p * (n - 1)
    k = int(idx)
    d = idx - k

    if k >= n - 1:
        return sorted_vals[-1]

    return sorted_vals[k] + d * (sorted_vals[k + 1] - sorted_vals[k])


def compute_stats_summary(values: Sequence[float]) -> Dict[str, float]:
    """Computes all 8 canonical descriptive statistics required for describe.py.

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        Dict[str, float]: Dictionary mapping metric names to computed values:
            'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'.

    Raises:
        ValueError: If the input collection has fewer than 2 elements.
    """
    return {
        "Count": float(compute_count(values)),
        "Mean": compute_mean(values),
        "Std": compute_std(values),
        "Min": compute_min(values),
        "25%": compute_percentile(values, 0.25),
        "50%": compute_percentile(values, 0.50),
        "75%": compute_percentile(values, 0.75),
        "Max": compute_max(values),
    }
