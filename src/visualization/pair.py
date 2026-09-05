"""Pair Plot Visualization and Feature Selection Analysis for 42 DSLR.

Generates a multivariate scatter plot matrix (pair plot) across Hogwarts courses
colored by house, answering: From this visualization, which features are you
going to use for your logistic regression?
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analytics.loader import HOGWARTS_COURSES

HOUSES: List[str] = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

HOUSE_COLORS: Dict[str, str] = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}

# Features selected for Logistic Regression training based on visual separation
# and exclusion of homogeneous (Arithmancy, Care of Magical Creatures) and
# collinear (Defense Against the Dark Arts) features.
SELECTED_FEATURES: List[str] = [
    "Astronomy",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Charms",
    "Flying",
]

DISCARDED_FEATURES: Dict[str, str] = {
    "Arithmancy": "Homogeneous score distribution across all houses (F=0.38)",
    "Care of Magical Creatures": "Near-identical overlapping distribution across houses (F=1.57)",
    "Defense Against the Dark Arts": "Strict collinearity with Astronomy (r = -1.0000)",
    "Potions": "Low class discriminative separation relative to remaining courses",
}


def plot_pair_plot(
    df: pd.DataFrame,
    courses: Optional[List[str]] = None,
    output_path: Optional[str] = "pair_plot.png",
    show: bool = True,
) -> sns.PairGrid:
    """Generates a scatter plot matrix (pair plot) across numerical courses.

    Args:
        df (pd.DataFrame): Dataset DataFrame containing 'Hogwarts House' and course columns.
        courses (Optional[List[str]]): List of course column names to plot.
            Defaults to all available courses in HOGWARTS_COURSES.
        output_path (Optional[str]): Path to save generated image (or None).
        show (bool): Whether to invoke plt.show().

    Returns:
        sns.PairGrid: The Seaborn PairGrid object containing the matrix.

    Raises:
        ValueError: If 'Hogwarts House' is missing or fewer than 2 courses are available.
    """
    if "Hogwarts House" not in df.columns:
        raise ValueError("DataFrame missing mandatory 'Hogwarts House' column.")

    target_courses = [c for c in (courses or HOGWARTS_COURSES) if c in df.columns]
    if len(target_courses) < 2:
        raise ValueError("Pair plot requires at least 2 numerical course features.")

    plot_cols = ["Hogwarts House"] + target_courses
    plot_df = df[plot_cols].dropna()

    grid = sns.pairplot(
        plot_df,
        hue="Hogwarts House",
        hue_order=HOUSES,
        palette=HOUSE_COLORS,
        diag_kind="hist",
        plot_kws={"alpha": 0.45, "s": 10, "rasterized": True},
        diag_kws={"alpha": 0.55, "common_norm": False},
        corner=False,
    )

    grid.fig.suptitle(
        "42 DSLR — Multivariate Pair Plot Matrix (Hogwarts Courses vs Houses)",
        y=1.005,
        fontsize=14,
        fontweight="bold",
    )

    if output_path:
        grid.savefig(output_path, dpi=120)

    if show and plt.get_backend().lower() != "agg":
        plt.show()

    return grid
