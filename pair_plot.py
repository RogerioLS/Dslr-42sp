"""Pair Plot CLI for 42 DSLR.

Generates a multivariate scatter plot matrix across Hogwarts courses,
answering the question: From this visualization, which features are you
going to use for your logistic regression?
"""

import sys
from pathlib import Path

from src.analytics.loader import load_csv
from src.visualization.pair import DISCARDED_FEATURES, SELECTED_FEATURES, plot_pair_plot


def main() -> None:
    """CLI entrypoint for pair plot multivariate analysis."""
    if len(sys.argv) != 2:
        print("Usage: python3 pair_plot.py <dataset_train.csv>", file=sys.stderr)
        sys.exit(1)

    dataset_path = sys.argv[1]
    if not Path(dataset_path).exists():
        print(f"Error: File '{dataset_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        df = load_csv(dataset_path)
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" 🧙‍♂️ 42 DSLR — PAIR PLOT MULTIVARIATE FEATURE SELECTION")
    print("=" * 70)
    print(" 🎯 Selected Features for Logistic Regression:")
    for feat in SELECTED_FEATURES:
        print(f"    ✔ {feat}")

    print("\n 🚫 Excluded Features & Mathematical Rationale:")
    for feat, reason in DISCARDED_FEATURES.items():
        print(f"    ✖ {feat:30s} -> {reason}")
    print("=" * 70 + "\n")

    plot_pair_plot(df, output_path="pair_plot.png", show=True)


if __name__ == "__main__":
    main()
