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


def compute_bonus_stats_summary(values: Sequence[float], total_rows: int = 0) -> Dict[str, float]:
    """Computes full descriptive statistics including bonus metrics and NaN counts.

    Args:
        values (Sequence[float]): Collection of non-null numerical values.
        total_rows (int): Total number of rows including nulls in the feature column.

    Returns:
        Dict[str, float]: Dictionary with all 8 core metrics plus:
            'Variance', 'IQR', 'Skewness', 'Kurtosis', 'NaNs'.
    """
    summary = compute_stats_summary(values)
    count_val = len(values)
    nans_count = max(0, total_rows - count_val) if total_rows > 0 else 0

    summary["Variance"] = compute_variance(values)
    summary["IQR"] = compute_iqr(values)
    summary["Skewness"] = compute_skewness(values) if count_val >= 3 else 0.0
    summary["Kurtosis"] = compute_kurtosis(values) if count_val >= 4 else 0.0
    summary["NaNs"] = float(nans_count)
    return summary


def compute_covariance(x: Sequence[float], y: Sequence[float]) -> float:
    """Computes sample covariance between two paired numerical sequences.

    Formula:
        cov(x, y) = (1 / (N - 1)) * sum((x_i - mu_x) * (y_i - mu_y))

    Args:
        x (Sequence[float]): First numerical sequence.
        y (Sequence[float]): Second numerical sequence (must match length of x).

    Returns:
        float: Sample covariance between x and y.

    Raises:
        ValueError: If lengths do not match or sample size is fewer than 2.
    """
    n = compute_count(x)
    n_y = compute_count(y)
    if n != n_y:
        raise ValueError(f"Sequences must have identical length: {n} vs {n_y}.")
    if n < 2:
        raise ValueError("Covariance requires at least 2 paired observations.")

    mu_x = compute_mean(x)
    mu_y = compute_mean(y)

    sum_prod_diff = 0.0
    for xi, yi in zip(x, y):
        sum_prod_diff += (float(xi) - mu_x) * (float(yi) - mu_y)

    return sum_prod_diff / (n - 1)


def compute_pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Computes Pearson correlation coefficient (r) between two sequences.

    Formula:
        r = cov(x, y) / (std_x * std_y)

    Args:
        x (Sequence[float]): First numerical sequence.
        y (Sequence[float]): Second numerical sequence.

    Returns:
        float: Pearson correlation coefficient in range [-1.0, 1.0].

    Raises:
        ValueError: If sample size < 2 or either variable has zero variance.
    """
    std_x = compute_std(x)
    std_y = compute_std(y)

    if std_x == 0.0 or std_y == 0.0:
        raise ValueError("Cannot compute correlation for variables with zero variance.")

    cov = compute_covariance(x, y)
    r = cov / (std_x * std_y)

    # Clamp potential floating point rounding artifacts
    if r > 1.0:
        return 1.0
    if r < -1.0:
        return -1.0
    return r


def compute_variance(values: Sequence[float]) -> float:
    """Computes the sample variance with Bessel's correction (N - 1).

    Formula:
        s^2 = (1 / (N - 1)) * sum((x_i - mu)^2)

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        float: Sample variance.

    Raises:
        ValueError: If fewer than 2 elements.
    """
    n = compute_count(values)
    if n < 2:
        raise ValueError("Sample variance requires at least 2 observations.")

    mu = compute_mean(values)
    sum_sq = 0.0
    for val in values:
        diff = float(val) - mu
        sum_sq += diff * diff

    return sum_sq / (n - 1)


def compute_iqr(values: Sequence[float]) -> float:
    """Computes the Interquartile Range (IQR = Q3 - Q1).

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        float: Difference between 75th and 25th percentiles.
    """
    q75 = compute_percentile(values, 0.75)
    q25 = compute_percentile(values, 0.25)
    return q75 - q25


def compute_skewness(values: Sequence[float]) -> float:
    """Computes the sample skewness (Fisher-Pearson standardized 3rd moment).

    Formula (unbiased sample skewness):
        skew = (N / ((N - 1) * (N - 2))) * sum(((x_i - mu) / s)^3)

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        float: Sample skewness coefficient.

    Raises:
        ValueError: If fewer than 3 observations or standard deviation is zero.
    """
    n = compute_count(values)
    if n < 3:
        raise ValueError("Skewness computation requires at least 3 observations.")

    mu = compute_mean(values)
    std = compute_std(values)
    if std == 0.0:
        return 0.0

    sum_cubed = 0.0
    for val in values:
        z = (float(val) - mu) / std
        sum_cubed += z * z * z

    factor = float(n) / ((n - 1) * (n - 2))
    return factor * sum_cubed


def compute_kurtosis(values: Sequence[float]) -> float:
    """Computes sample excess kurtosis (Fisher definition, normal distribution = 0).

    Formula:
        kurt = (N * (N + 1) / ((N - 1) * (N - 2) * (N - 3))) * sum(z_i^4)
               - (3 * (N - 1)^2 / ((N - 2) * (N - 3)))

    Args:
        values (Sequence[float]): Collection of non-null numerical values.

    Returns:
        float: Sample excess kurtosis.

    Raises:
        ValueError: If fewer than 4 observations or standard deviation is zero.
    """
    n = compute_count(values)
    if n < 4:
        raise ValueError("Kurtosis computation requires at least 4 observations.")

    mu = compute_mean(values)
    std = compute_std(values)
    if std == 0.0:
        return 0.0

    sum_fourth = 0.0
    for val in values:
        z = (float(val) - mu) / std
        sum_fourth += z * z * z * z

    n_f = float(n)
    term1 = (n_f * (n_f + 1.0)) / ((n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0)) * sum_fourth
    term2 = (3.0 * (n_f - 1.0) ** 2) / ((n_f - 2.0) * (n_f - 3.0))
    return term1 - term2
