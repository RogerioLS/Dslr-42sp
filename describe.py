"""Command-line interface for 42 DSLR descriptive statistics.

Calculates and formats the 8 fundamental descriptive statistics (Count, Mean, Std,
Min, 25%, 50%, 75%, Max) for all numerical features from first principles,
strictly adhering to the 42 Norm and Anti-Cheating guidelines.
"""

import argparse
import sys
from typing import Dict, List, Sequence

from src.analytics.loader import extract_valid_feature_values, get_numerical_features, load_csv
from src.analytics.statistics import compute_bonus_stats_summary, compute_stats_summary

CORE_METRIC_LABELS: Sequence[str] = (
    "Count",
    "Mean",
    "Std",
    "Min",
    "25%",
    "50%",
    "75%",
    "Max",
)

BONUS_METRIC_LABELS: Sequence[str] = (
    "Count",
    "Mean",
    "Std",
    "Min",
    "25%",
    "50%",
    "75%",
    "Max",
    "Variance",
    "IQR",
    "Skewness",
    "Kurtosis",
    "NaNs",
)


def format_describe_table(
    features: Sequence[str],
    summaries: Dict[str, Dict[str, float]],
    metric_labels: Sequence[str] = CORE_METRIC_LABELS,
) -> str:
    """Formats calculated statistics into a clean, aligned tabular string.

    Args:
        features (Sequence[str]): Ordered list of numerical feature names.
        summaries (Dict[str, Dict[str, float]]): Mapping of feature name to
            its dictionary of computed metrics.
        metric_labels (Sequence[str]): Sequence of metric names to include in table rows.

    Returns:
        str: Multi-line formatted ASCII table matching the 42 subject.
    """
    if not features:
        return ""

    label_col_width = 10 if "Variance" in metric_labels else 8
    col_widths: Dict[str, int] = {}
    for feat in features:
        col_widths[feat] = max(len(feat) + 2, 16)

    # Header Row
    lines: List[str] = []
    header_cells = [f"{'':<{label_col_width}}"]
    for feat in features:
        header_cells.append(f"{feat:>{col_widths[feat]}}")
    lines.append("".join(header_cells))

    # Metric Rows
    for metric in metric_labels:
        row_cells = [f"{metric:<{label_col_width}}"]
        for feat in features:
            val = summaries[feat][metric]
            row_cells.append(f"{val:>{col_widths[feat]}.6f}")
        lines.append("".join(row_cells))

    return "\n".join(lines)


def run_describe(filepath: str, bonus: bool = False) -> int:
    """Executes descriptive statistics analysis pipeline for a given CSV file.

    Args:
        filepath (str): Path to the input dataset CSV.
        bonus (bool): If True, computes and displays extended statistics (Variance, IQR, etc.).

    Returns:
        int: 0 if execution succeeded, 1 on failure.
    """
    try:
        df = load_csv(filepath)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Unexpected error loading dataset: {err}", file=sys.stderr)
        return 1

    features = get_numerical_features(df)
    if not features:
        print("Error: No numerical features found in dataset.", file=sys.stderr)
        return 1

    total_rows = len(df)
    summaries: Dict[str, Dict[str, float]] = {}
    for feature in features:
        values = extract_valid_feature_values(df, feature)
        if not values:
            print(f"Warning: Feature '{feature}' has no valid values.", file=sys.stderr)
            continue
        if bonus:
            summaries[feature] = compute_bonus_stats_summary(values, total_rows=total_rows)
        else:
            summaries[feature] = compute_stats_summary(values)

    valid_features = [f for f in features if f in summaries]
    metric_labels = BONUS_METRIC_LABELS if bonus else CORE_METRIC_LABELS
    table = format_describe_table(valid_features, summaries, metric_labels=metric_labels)
    print(table)
    return 0


def main() -> None:
    """Main entry point for describe.py CLI."""
    parser = argparse.ArgumentParser(
        description="42 DSLR — Descriptive statistics calculator from first principles."
    )
    parser.add_argument("dataset", help="Path to the dataset CSV file.")
    parser.add_argument(
        "--bonus",
        "-b",
        action="store_true",
        help="Display extended bonus statistics (Variance, IQR, Skewness, Kurtosis, NaNs).",
    )

    args = parser.parse_args()
    exit_code = run_describe(args.dataset, bonus=args.bonus)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
