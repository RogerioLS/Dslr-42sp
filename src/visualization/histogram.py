"""Histogram Visualization and Distribution Analysis Engine for 42 DSLR.

Analyzes score distributions per Hogwarts house across courses to identify
which course features a homogeneous score distribution across all four houses.
"""

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.loader import HOGWARTS_COURSES, extract_valid_feature_values
from src.analytics.statistics import compute_mean, compute_std

HOUSES: List[str] = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

HOUSE_COLORS: Dict[str, str] = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}


def compute_house_course_stats(df: pd.DataFrame, course: str) -> Dict[str, Dict[str, float]]:
    """Computes mean and std per Hogwarts house for a specific course.

    Args:
        df (pd.DataFrame): Dataset dataframe containing 'Hogwarts House' and course column.
        course (str): Course name column.

    Returns:
        Dict[str, Dict[str, float]]: Mapping of house name to {'mean': float, 'std': float}.
    """
    stats_per_house: Dict[str, Dict[str, float]] = {}
    for house in HOUSES:
        house_df = df[df["Hogwarts House"] == house]
        values = extract_valid_feature_values(house_df, course)
        if values:
            stats_per_house[house] = {
                "mean": compute_mean(values),
                "std": compute_std(values),
            }
        else:
            stats_per_house[house] = {"mean": 0.0, "std": 0.0}
    return stats_per_house


def compute_course_variance_between_houses(df: pd.DataFrame, course: str) -> float:
    """Computes the variance of house means to quantify homogeneity across houses.

    A lower variance between house means indicates a more homogeneous distribution.

    Args:
        df (pd.DataFrame): Input dataframe.
        course (str): Numerical course column name.

    Returns:
        float: Variance of house means.
    """
    house_stats = compute_house_course_stats(df, course)
    means = [h_stat["mean"] for h_stat in house_stats.values() if h_stat["std"] > 0]
    if len(means) < 2:
        return float("inf")
    return compute_std(means) ** 2


def find_most_homogeneous_course(df: pd.DataFrame) -> Tuple[str, float]:
    """Identifies which Hogwarts course has the most homogeneous distribution across houses.

    Args:
        df (pd.DataFrame): Hogwarts dataset dataframe.

    Returns:
        Tuple[str, float]: Name of the most homogeneous course and its variance score.
    """
    scores: Dict[str, float] = {}
    for course in HOGWARTS_COURSES:
        if course in df.columns:
            scores[course] = compute_course_variance_between_houses(df, course)

    if not scores:
        return "", 0.0

    most_homogeneous = min(scores, key=scores.get)  # type: ignore
    return most_homogeneous, scores[most_homogeneous]


def plot_histograms_grid(
    df: pd.DataFrame,
    output_path: Optional[str] = "histogram_grid.png",
    show: bool = True,
) -> plt.Figure:
    """Generates an NxM subplot grid of score histograms per Hogwarts house.

    Args:
        df (pd.DataFrame): Dataframe with Hogwarts House and course features.
        output_path (Optional[str]): Filepath to save generated figure (or None).
        show (bool): Whether to invoke plt.show().

    Returns:
        plt.Figure: The generated Matplotlib figure.
    """
    available_courses = [c for c in HOGWARTS_COURSES if c in df.columns]
    n_courses = len(available_courses)
    if n_courses == 0:
        raise ValueError("No Hogwarts courses found in dataframe to plot.")

    n_cols = 4
    n_rows = math.ceil(n_courses / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, course in enumerate(available_courses):
        ax = axes_flat[i]
        for house in HOUSES:
            house_df = df[df["Hogwarts House"] == house]
            values = extract_valid_feature_values(house_df, course)
            if values:
                ax.hist(
                    values,
                    bins=20,
                    alpha=0.55,
                    label=house,
                    color=HOUSE_COLORS.get(house, "gray"),
                    edgecolor="black",
                    linewidth=0.5,
                )
        ax.set_title(course, fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for j in range(n_courses, len(axes_flat)):
        axes_flat[j].axis("off")

    if len(axes_flat) > 0:
        axes_flat[0].legend(loc="upper right", fontsize=8)

    plt.suptitle(
        "Hogwarts Courses Score Distribution per House (42 DSLR Histogram Analysis)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)

    if show:
        plt.show()

    return fig
