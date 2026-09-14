#!/usr/bin/env python3
"""Inference Entrypoint for 42 DSLR Logistic Regression Model (logreg_predict.py).

Loads calibrated One-vs-Rest logistic regression weights and StandardScaler parameters,
classifies students from a test dataset, and generates the houses.csv prediction file
strictly conforming to École 42 subject specifications:
    Index,Hogwarts House
    0,Gryffindor
    1,Hufflepuff
    ...

Usage:
    python3 logreg_predict.py datasets/dataset_test.csv weights.json
    python3 logreg_predict.py datasets/dataset_test.csv weights.json --output houses.csv
"""

import argparse
import sys
import time
from collections import Counter
from typing import List, Optional

import pandas as pd

from src.analytics.loader import load_csv
from src.model.multiclass import OneVsRestLogisticRegression

# ANSI color escape sequences
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def print_prediction_banner(
    dataset_path: str,
    weights_path: str,
    samples_count: int,
    output_path: str,
) -> None:
    """Prints the styled inference configuration banner to stdout.

    Args:
        dataset_path (str): Path to input test dataset.
        weights_path (str): Path to serialized weights JSON file.
        samples_count (int): Number of student samples to classify.
        output_path (str): Destination path for predictions CSV.
    """
    print(f"{CYAN}============================================================{RESET}")
    print(f"{BOLD}{MAGENTA} 🔮 42 DSLR — ONE-VS-REST LOGISTIC REGRESSION INFERENCE      {RESET}")
    print(f"{CYAN}============================================================{RESET}")
    print(f"{BLUE}• Test Dataset   :{RESET} {dataset_path}")
    print(f"{BLUE}• Weights Source :{RESET} {weights_path}")
    print(f"{BLUE}• Samples Count  :{RESET} {samples_count}")
    print(f"{BLUE}• Output File    :{RESET} {output_path}")
    print(f"{CYAN}------------------------------------------------------------{RESET}")


def save_predictions_csv(
    indices: pd.Series,
    predictions: List[str],
    output_path: str,
) -> None:
    """Saves predictions to a CSV file matching the exact subject format.

    Args:
        indices (pd.Series): Student indices.
        predictions (List[str]): Predicted Hogwarts house names.
        output_path (str): Destination file path.
    """
    out_df = pd.DataFrame(
        {
            "Index": indices,
            "Hogwarts House": predictions,
        }
    )
    out_df.to_csv(output_path, index=False)


def _load_test_data(dataset_path: str) -> Optional[pd.DataFrame]:
    """Loads and validates non-emptiness of test dataset."""
    try:
        df = load_csv(dataset_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error loading test dataset: {err}", file=sys.stderr)
        return None
    except Exception as err:
        print(f"Unexpected error loading dataset: {err}", file=sys.stderr)
        return None

    if len(df) == 0:
        print("Error: Test dataset is empty.", file=sys.stderr)
        return None
    return df


def _load_model(weights_path: str) -> Optional[OneVsRestLogisticRegression]:
    """Loads calibrated OneVsRestLogisticRegression model from weights JSON."""
    try:
        return OneVsRestLogisticRegression.load_weights(weights_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error loading model weights: {err}", file=sys.stderr)
        return None
    except Exception as err:
        print(f"Unexpected error loading weights: {err}", file=sys.stderr)
        return None


def _validate_features(df: pd.DataFrame, model: OneVsRestLogisticRegression) -> bool:
    """Verifies that all model feature columns exist in test dataset."""
    missing = [col for col in model.features_ if col not in df.columns]
    if missing:
        print(
            f"Error: Missing required feature columns in test dataset: {missing}",
            file=sys.stderr,
        )
        return False
    return True


def _print_completion_summary(
    predictions: List[str],
    elapsed: float,
    output_path: str,
) -> None:
    """Prints breakdown of predictions across Hogwarts houses."""
    counts = Counter(predictions)
    house_colors = {
        "Gryffindor": "\033[31m",
        "Hufflepuff": "\033[33m",
        "Ravenclaw": "\033[34m",
        "Slytherin": "\033[32m",
    }
    print(f"{BOLD}{GREEN}✔ INFERENCE COMPLETED IN {elapsed:.3f}s!{RESET}")
    print(f"• Total Classified : {BOLD}{len(predictions)}{RESET} students")
    print(f"• Output Saved To  : {BOLD}{output_path}{RESET}\n")
    print(f"{BOLD}Sorting Hat Distribution:{RESET}")
    for house in sorted(counts.keys()):
        color = house_colors.get(house, RESET)
        pct = (counts[house] / len(predictions)) * 100.0
        print(f"  [{color}{house:<11}{RESET}] {counts[house]:>4} students ({pct:5.1f}%)")
    print(f"{CYAN}============================================================{RESET}")


def run_prediction(
    dataset_path: str,
    weights_path: str,
    output_path: str = "houses.csv",
    quiet: bool = False,
) -> int:
    """Executes the inference pipeline and writes predictions to CSV.

    Args:
        dataset_path (str): Filepath to the CSV test dataset.
        weights_path (str): Filepath to the serialized weights.json file.
        output_path (str): Output destination path for houses.csv.
        quiet (bool): If True, suppresses banner and summary output.

    Returns:
        int: 0 on success, 1 on error.
    """
    df = _load_test_data(dataset_path)
    if df is None:
        return 1

    model = _load_model(weights_path)
    if model is None:
        return 1

    if not _validate_features(df, model):
        return 1

    if not quiet:
        print_prediction_banner(dataset_path, weights_path, len(df), output_path)

    t0 = time.time()
    try:
        predictions = model.predict(df)
    except Exception as err:
        print(f"Error during inference: {err}", file=sys.stderr)
        return 1
    t1 = time.time()

    indices = df["Index"] if "Index" in df.columns else pd.Series(range(len(df)))

    try:
        save_predictions_csv(indices, predictions, output_path)
    except Exception as err:
        print(f"Error writing predictions to '{output_path}': {err}", file=sys.stderr)
        return 1

    if not quiet:
        _print_completion_summary(predictions, t1 - t0, output_path)

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command-line argument parser for logreg_predict.py.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="logreg_predict.py",
        description=(
            "42 DSLR — Predict Hogwarts Houses for student records using a trained "
            "One-vs-Rest Logistic Regression model and generate houses.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the test CSV dataset (e.g. datasets/dataset_test.csv).",
    )
    parser.add_argument(
        "weights",
        type=str,
        help="Path to the trained weights JSON file (e.g. weights.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="houses.csv",
        help="Destination path for predictions CSV (default: houses.csv).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress banner and distribution breakdown output.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI main entry point for logreg_predict.py.

    Args:
        argv (Optional[List[str]]): Command-line argument list (default: sys.argv[1:]).

    Returns:
        int: Exit status code (0 for success, non-zero for failure).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_prediction(
        dataset_path=args.dataset,
        weights_path=args.weights,
        output_path=args.output,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
