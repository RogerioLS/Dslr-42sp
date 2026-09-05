"""Scatter Plot Visualization and Feature Similarity Engine for 42 DSLR.

Analyzes bivariate relationships and feature correlations across Hogwarts courses
to answer: What are the two features that are similar?
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.loader import HOGWARTS_COURSES, extract_paired_feature_values
from src.analytics.statistics import compute_pearson_correlation

HOUSES: List[str] = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

HOUSE_COLORS: Dict[str, str] = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}


def compute_correlation_matrix(
    df: pd.DataFrame, courses: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Computes the pairwise Pearson correlation matrix across numerical courses.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        courses (Optional[List[str]]): List of course columns to evaluate.
            Defaults to all available courses in HOGWARTS_COURSES.

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary representing the correlation matrix.
    """
    target_courses = courses or [c for c in HOGWARTS_COURSES if c in df.columns]
    matrix: Dict[str, Dict[str, float]] = {c1: {} for c1 in target_courses}

    for i, c1 in enumerate(target_courses):
        matrix[c1][c1] = 1.0
        for j in range(i + 1, len(target_courses)):
            c2 = target_courses[j]
            vals_a, vals_b = extract_paired_feature_values(df, c1, c2)
            if len(vals_a) >= 2:
                r = compute_pearson_correlation(vals_a, vals_b)
            else:
                r = 0.0
            matrix[c1][c2] = r
            matrix[c2][c1] = r

    return matrix


def find_most_correlated_pair(
    df: pd.DataFrame, courses: Optional[List[str]] = None
) -> Tuple[str, str, float]:
    """Identifies the two features with the highest absolute Pearson correlation.

    In statistical modeling, collinear features (|r| close to 1.0) provide redundant
    information and should be detected during exploratory data analysis.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        courses (Optional[List[str]]): List of courses to analyze.

    Returns:
        Tuple[str, str, float]: (feature_1, feature_2, pearson_correlation_r).

    Raises:
        ValueError: If fewer than 2 courses are available for correlation analysis.
    """
    target_courses = courses or [c for c in HOGWARTS_COURSES if c in df.columns]
    if len(target_courses) < 2:
        raise ValueError("At least two courses are required to find correlated features.")

    best_pair: Tuple[str, str] = ("", "")
    best_abs_r: float = -1.0
    best_r: float = 0.0

    for i in range(len(target_courses)):
        for j in range(i + 1, len(target_courses)):
            c1, c2 = target_courses[i], target_courses[j]
            vals_a, vals_b = extract_paired_feature_values(df, c1, c2)
            if len(vals_a) < 2:
                continue
            r = compute_pearson_correlation(vals_a, vals_b)
            abs_r = abs(r)
            if abs_r > best_abs_r:
                best_abs_r = abs_r
                best_r = r
                best_pair = (c1, c2)

    return best_pair[0], best_pair[1], best_r


def plot_scatter(
    df: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    output_path: Optional[str] = "scatter_plot.png",
    show: bool = True,
) -> plt.Figure:
    """Generates a bivariate scatter plot colored by Hogwarts house.

    Args:
        df (pd.DataFrame): Dataset DataFrame containing houses and features.
        feat_x (str): Feature column name for X-axis.
        feat_y (str): Feature column name for Y-axis.
        output_path (Optional[str]): Path to save image file (or None).
        show (bool): Whether to invoke plt.show().

    Returns:
        plt.Figure: The generated Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for house in HOUSES:
        house_df = df[df["Hogwarts House"] == house]
        x_vals, y_vals = extract_paired_feature_values(house_df, feat_x, feat_y)
        if x_vals and y_vals:
            ax.scatter(
                x_vals,
                y_vals,
                alpha=0.6,
                label=house,
                color=HOUSE_COLORS.get(house, "gray"),
                edgecolors="black",
                linewidths=0.4,
                s=35,
            )

    ax.set_xlabel(feat_x, fontsize=12, fontweight="bold")
    ax.set_ylabel(feat_y, fontsize=12, fontweight="bold")
    ax.set_title(
        f"42 DSLR — Bivariate Scatter Analysis: {feat_x} vs {feat_y}",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="best", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)

    if show and plt.get_backend().lower() != "agg":
        plt.show()

    return fig
