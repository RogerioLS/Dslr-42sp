#!/usr/bin/env python3
"""Accuracy Evaluator for 42 DSLR Predictions.

Compares a generated houses.csv prediction file against a ground truth CSV
using Scikit-Learn accuracy_score to verify the 98% threshold required by 42.
"""

import sys

import pandas as pd


def evaluate(predictions_path: str, truth_path: str) -> float:
    """Calculates accuracy score between predictions and truth.

    Args:
        predictions_path (str): Path to houses.csv.
        truth_path (str): Path to truth CSV (e.g. dataset_truth.csv).

    Returns:
        float: Accuracy score (0.0 to 1.0).
    """
    pred_df = pd.read_csv(predictions_path)
    truth_df = pd.read_csv(truth_path)

    if "Hogwarts House" not in pred_df.columns:
        raise ValueError("Missing 'Hogwarts House' column in predictions.")
    if "Hogwarts House" not in truth_df.columns:
        raise ValueError("Missing 'Hogwarts House' column in ground truth.")

    total = len(pred_df)
    correct = (pred_df["Hogwarts House"] == truth_df["Hogwarts House"]).sum()
    accuracy = correct / total

    print("==================================================")
    print(" 🎯  42 DSLR CLASSIFIER ACCURACY REPORT          ")
    print("==================================================")
    print(f"Total samples:     {total}")
    print(f"Correct sorted:    {correct}")
    print(f"Accuracy Score:    {accuracy * 100:.2f}%")
    print("Required Target:   >= 98.00%")
    print("--------------------------------------------------")

    if accuracy >= 0.98:
        print("✅ PASSED: Sorting Hat meets Professor McGonagall's standard!")
    else:
        print("❌ FAILED: Accuracy below 98%. Adjust learning rate, epochs, or features.")

    return accuracy


def main() -> int:
    """Main execution function for evaluate_accuracy."""
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/evaluate_accuracy.py <houses.csv> <ground_truth.csv>")
        return 1

    try:
        acc = evaluate(sys.argv[1], sys.argv[2])
        return 0 if acc >= 0.98 else 1
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
