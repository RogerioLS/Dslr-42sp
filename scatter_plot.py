"""Scatter Plot CLI for 42 DSLR.

Analyzes bivariate relationships and feature correlations across Hogwarts courses,
answering the question: What are the two features that are similar?
"""

import sys
from pathlib import Path

from src.analytics.loader import load_csv
from src.visualization.scatter import find_most_correlated_pair, plot_scatter


def main() -> None:
    """CLI entrypoint for scatter plot feature correlation analysis."""
    if len(sys.argv) != 2:
        print("Usage: python3 scatter_plot.py <dataset_train.csv>", file=sys.stderr)
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

    feat_x, feat_y, corr = find_most_correlated_pair(df)
    print("\n" + "=" * 65)
    print(" 🧙‍♂️ 42 DSLR — SCATTER PLOT BIVARIATE ANALYSIS")
    print("=" * 65)
    print(f" 📊 Most Similar Features: {feat_x} <-> {feat_y}")
    print(f" 📉 Pearson Correlation (r): {corr:.6f} (|r| = {abs(corr):.6f})")
    print("=" * 65 + "\n")

    plot_scatter(df, feat_x, feat_y, output_path="scatter_plot.png", show=True)


if __name__ == "__main__":
    main()
