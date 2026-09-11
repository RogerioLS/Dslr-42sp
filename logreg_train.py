#!/usr/bin/env python3
"""Training Entrypoint for 42 DSLR Logistic Regression Model (logreg_train.py).

Trains a multiclass One-vs-Rest (OvR) logistic regression classifier on Hogwarts student
grades using handcrafted Batch Gradient Descent, and serializes the optimized weights
and standardization parameters into a weights JSON file for subsequent inference.

Usage:
    python3 logreg_train.py datasets/dataset_train.csv
    python3 logreg_train.py datasets/dataset_train.csv --output weights.json --epochs 2000 --lr 0.5
"""

import argparse
import sys
import time
from typing import List

import numpy as np
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


def print_training_banner(
    samples_count: int,
    features_count: int,
    epochs: int,
    learning_rate: float,
    output_path: str,
) -> None:
    """Prints the styled training configuration banner to stdout.

    Args:
        samples_count (int): Number of training samples loaded.
        features_count (int): Number of numerical features selected.
        epochs (int): Total training epochs.
        learning_rate (float): Learning rate alpha.
        output_path (str): Destination path for serialized weights.
    """
    print(f"{CYAN}============================================================{RESET}")
    print(f"{BOLD}{MAGENTA} 🧙‍♂️ 42 DSLR — ONE-VS-REST LOGISTIC REGRESSION TRAINING      {RESET}")
    print(f"{CYAN}============================================================{RESET}")
    print(f"{BLUE}• Training Samples  :{RESET} {samples_count}")
    print(f"{BLUE}• Feature Count     :{RESET} {features_count}")
    print(f"{BLUE}• Learning Rate (α) :{RESET} {learning_rate}")
    print(f"{BLUE}• Training Epochs   :{RESET} {epochs}")
    print(f"{BLUE}• Weights Target    :{RESET} {output_path}")
    print(f"{CYAN}------------------------------------------------------------{RESET}")


def progress_callback(class_name: str, epoch: int, loss: float) -> None:
    """Callback hook displaying loss updates at regular epoch intervals.

    Args:
        class_name (str): The Hogwarts house being trained.
        epoch (int): Current epoch number.
        loss (float): Binary Cross-Entropy loss at this epoch.
    """
    house_colors = {
        "Gryffindor": "\033[31m",  # Red
        "Hufflepuff": "\033[33m",  # Yellow
        "Ravenclaw": "\033[34m",  # Blue
        "Slytherin": "\033[32m",  # Green
    }
    color = house_colors.get(class_name, RESET)
    print(
        f"  [{color}{class_name:<11}{RESET}] "
        f"Epoch {epoch:>5} ➔ Log-Loss: {BOLD}{loss:.6f}{RESET}"
    )


def run_training(
    dataset_path: str,
    output_path: str = "weights.json",
    learning_rate: float = 0.5,
    epochs: int = 2000,
    features_mode: str = "all",
    quiet: bool = False,
) -> int:
    """Executes the end-to-end model training workflow.

    Args:
        dataset_path (str): Filepath to the CSV training dataset.
        output_path (str): Output destination path for weights.json.
        learning_rate (float): Step size for Batch Gradient Descent.
        epochs (int): Number of iterations.
        features_mode (str): Feature selection mode ('all' or 'selected').
        quiet (bool): If True, suppresses epoch-by-epoch loss printing.

    Returns:
        int: 0 on success, 1 on error.
    """
    try:
        df = load_csv(dataset_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error loading dataset: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Unexpected error: {err}", file=sys.stderr)
        return 1

    if "Hogwarts House" not in df.columns:
        print("Error: Required column 'Hogwarts House' missing from dataset.", file=sys.stderr)
        return 1

    # Select feature subset
    if features_mode == "selected":
        chosen_features = [f for f in SELECTED_FEATURES if f in df.columns]
    else:
        metadata = {
            "Index",
            "Hogwarts House",
            "First Name",
            "Last Name",
            "Birthday",
            "Best Hand",
        }
        chosen_features = [
            col
            for col in df.columns
            if col not in metadata and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not chosen_features:
        print("Error: No numeric course features found for training.", file=sys.stderr)
        return 1

    if not quiet:
        print_training_banner(len(df), len(chosen_features), epochs, learning_rate, output_path)

    classifier = OneVsRestLogisticRegression(
        learning_rate=learning_rate,
        epochs=epochs,
        features=chosen_features,
    )

    cb = None if quiet else progress_callback

    t0 = time.time()
    try:
        classifier.fit(df, target_column="Hogwarts House", callback=cb)
    except Exception as err:
        print(f"Error during training optimization: {err}", file=sys.stderr)
        return 1
    t1 = time.time()

    # Compute training accuracy
    predictions = classifier.predict(df)
    correct_count = (np.array(predictions) == df["Hogwarts House"].values).sum()
    train_accuracy = (correct_count / len(df)) * 100.0

    # Save weights payload
    try:
        classifier.save_weights(output_path)
    except Exception as err:
        print(f"Error saving weights to '{output_path}': {err}", file=sys.stderr)
        return 1

    if not quiet:
        print(f"{CYAN}============================================================{RESET}")
        print(f"{BOLD}{GREEN}✔ TRAINING COMPLETED SUCCESSFULLY IN {t1 - t0:.2f}s!{RESET}")
        print(
            f"• Training Set Accuracy : {BOLD}{train_accuracy:.2f}%{RESET} "
            f"({correct_count}/{len(df)})"
        )
        print(f"• Model Weights Saved   : {BOLD}{output_path}{RESET}")
        print(f"{CYAN}============================================================{RESET}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command-line argument parser for logreg_train.py.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="42 DSLR — Train Multiclass One-vs-Rest Logistic Regression Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="Path to training CSV file (e.g. datasets/dataset_train.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="weights.json",
        help="Path to output serialized weights JSON file",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.5,
        help="Learning rate alpha for Batch Gradient Descent",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=2000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--features",
        choices=["all", "selected"],
        default="all",
        help="Feature selection strategy: 'all' 13 courses or 'selected' 9 courses",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress intermediate training loss progress",
    )
    return parser


def main() -> None:
    """Main CLI entrypoint for logreg_train.py."""
    parser = build_arg_parser()
    args = parser.parse_args()

    exit_code = run_training(
        dataset_path=args.dataset,
        output_path=args.output,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        features_mode=args.features,
        quiet=args.quiet,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
