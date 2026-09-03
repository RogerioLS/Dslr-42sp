"""Histogram CLI for 42 DSLR.

Analyzes score distributions per Hogwarts house for each course,
answering the question: Which Hogwarts course has a homogeneous score
distribution between all four houses?
"""

import sys
from pathlib import Path

from src.analytics.loader import load_csv
from src.visualization.histogram import find_most_homogeneous_course, plot_histograms_grid


def main() -> None:
    """CLI entrypoint for histogram feature distribution analysis."""
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset_train.csv>", file=sys.stderr)
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

    homogeneous_course, variance = find_most_homogeneous_course(df)
    print("\n" + "=" * 65)
    print(" 🧙‍♂️ 42 DSLR — HISTOGRAM DISTRIBUTION ANALYSIS")
    print("=" * 65)
    print(f" 📊 Most Homogeneous Course: {homogeneous_course}")
    print(f" 📉 Variance between house means: {variance:.4f}")
    print("=" * 65 + "\n")

    plot_histograms_grid(df, output_path="histogram_grid.png", show=True)


if __name__ == "__main__":
    main()
