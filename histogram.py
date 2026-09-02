"""Histogram CLI for 42 DSLR.

Plots the score distribution per Hogwarts house for each course,
to identify which course has a homogeneous distribution across houses.
"""

import math
import sys

import matplotlib.pyplot as plt

from src.analytics.loader import load_csv, HOGWARTS_COURSES, extract_valid_feature_values
from src.analytics.statistics import compute_stats_summary

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset.csv>")
        sys.exit()

    dataset_path = sys.argv[1]
    df = load_csv(dataset_path)

    candidates = ["Arithmancy", "Care of Magical Creatures"]
    for course in candidates:
        print(f"\n{course}:")
        for house in HOUSES:
            house_df = df[df["Hogwarts House"] == house]
            values = extract_valid_feature_values(house_df, course)
            stats = compute_stats_summary(values)
            print(f"  {house}: mean={stats['Mean']:.2f}  std={stats['Std']:.2f}")

    n_courses = len(HOGWARTS_COURSES)
    n_cols = 4
    n_rows = math.ceil(n_courses / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()

    for i, course in enumerate(HOGWARTS_COURSES):
        ax = axes[i]
        for house in HOUSES:
            house_df = df[df["Hogwarts House"] == house]
            values = extract_valid_feature_values(house_df, course)
            ax.hist(values, bins=20, alpha=0.5, label=house)
        ax.set_title(course, fontsize=9)

    for j in range(n_courses, len(axes)):
        axes[j].axis("off")

    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("histogram_grid.png")
    plt.show()
        
if __name__ == "__main__":
    main()  