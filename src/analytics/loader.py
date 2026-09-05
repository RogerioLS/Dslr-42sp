"""Data loading and feature extraction module for 42 DSLR.

Handles reading dataset CSV files, separating metadata from numerical features,
and filtering missing values (NaNs) for downstream mathematical processing.
"""

from pathlib import Path
from typing import Union

import pandas as pd

# Metadata columns to exclude from numerical feature analysis
METADATA_COLUMNS: list[str] = [
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
]

# Hogwarts courses evaluated as numerical features
HOGWARTS_COURSES: list[str] = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
]


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Loads a CSV dataset from the given file path.

    Args:
        filepath (Union[str, Path]): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file is empty or cannot be parsed as a CSV.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file at {path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    return df


def get_numerical_features(df: pd.DataFrame) -> list[str]:
    """Identifies and returns all numerical feature column names from the dataset.

    Args:
        df (pd.DataFrame): Dataset DataFrame.

    Returns:
        list[str]: List of numerical feature column names.
    """
    numerical_cols: list[str] = []
    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        # Verify column can be treated as numeric float data
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
    return numerical_cols


def extract_valid_feature_values(df: pd.DataFrame, feature_name: str) -> list[float]:
    """Extracts non-NaN numerical values for a specific feature as a raw float list.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        feature_name (str): Name of the feature column.

    Returns:
        list[float]: List of valid (non-null) numerical values.

    Raises:
        KeyError: If feature_name is not present in the DataFrame.
    """
    if feature_name not in df.columns:
        raise KeyError(f"Feature '{feature_name}' not found in dataset columns.")

    series = df[feature_name]
    valid_values: list[float] = []
    for val in series:
        if pd.notna(val):
            try:
                valid_values.append(float(val))
            except (ValueError, TypeError):
                continue

    return valid_values


def extract_paired_feature_values(
    df: pd.DataFrame, feat_a: str, feat_b: str
) -> tuple[list[float], list[float]]:
    """Extracts paired non-NaN numerical values for two features.

    Only rows where both features contain valid non-null numerical values
    are included, preserving observation alignment for bivariate analysis.

    Args:
        df (pd.DataFrame): Dataset DataFrame.
        feat_a (str): Name of first feature column.
        feat_b (str): Name of second feature column.

    Returns:
        tuple[list[float], list[float]]: Two parallel lists of valid paired floats.

    Raises:
        KeyError: If either feature is not present in the DataFrame.
    """
    if feat_a not in df.columns:
        raise KeyError(f"Feature '{feat_a}' not found in dataset columns.")
    if feat_b not in df.columns:
        raise KeyError(f"Feature '{feat_b}' not found in dataset columns.")

    series_a = df[feat_a]
    series_b = df[feat_b]

    paired_a: list[float] = []
    paired_b: list[float] = []

    for val_a, val_b in zip(series_a, series_b):
        if pd.notna(val_a) and pd.notna(val_b):
            try:
                f_a = float(val_a)
                f_b = float(val_b)
                paired_a.append(f_a)
                paired_b.append(f_b)
            except (ValueError, TypeError):
                continue

    return paired_a, paired_b
