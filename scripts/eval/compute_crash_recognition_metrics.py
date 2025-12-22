#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute classification metrics for crash_recognition (Yes/No).


Jiǎshè yùcè JSON wénjiàn shì yīgè list, měi gè yuánsù lèisì: { "Data_id": 208, "Video_path": "...", "Prompt": "...", "Gt_label": "No", "pred_label": "No", "raw_response": "No" } wǒmen jiāng"Yes" shì wéi zhèng lèi (Positive),"No" shì wéi fù lèi (Negative), jìsuàn: - N, TP, TN, FP, FN - Accuracy, Precision, Recall, F1 bìng jiāng jiéguǒ yǐ “Metric/ Value” de jǔzhèn xíngshì bǎocún dào běndì (txt + csv).
Show more
320
Assume the prediction JSON file is a list, with each element similar to:
{
"data_id": 208,
"video_path": "...",
"prompt": "...",
"gt_label": "No",
"pred_label": "No",
"raw_response": "No"
}

We consider "Yes" as the positive class and "No" as the negative class.
We will calculate:
- N, TP, TN, FP, FN
- Accuracy, Precision, Recall, F1
And save the results locally in a matrix format ("Metric / Value") (txt + csv).
"""

import json
import os
from typing import List, Dict, Tuple

PRED_JSON = (
    "/outputs/crash_recognition_evaluation_results/independent_monotask_models_crash_recognition/crashchat_dada_video_total_crash_recognition_test_predict.json"
)

OUTPUT_DIR = (
    "/outputs/crash_recognition_evaluation_results/independent_monotask_models_crash_recognition"
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "crashchat_dada_video_crash_recognition_test_metrics.txt"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "crashchat_dada_video_crash_recognition_test_metrics.csv"
)


def normalize_label(label: str) -> str:
    """Standardize the labels to either 'yes' or 'no'; all other values ​​will be considered invalid."""
    if label is None:
        return ""
    l = str(label).strip().lower()
    if l in ["yes", "y", "1", "true"]:
        return "yes"
    if l in ["no", "n", "0", "false"]:
        return "no"
    return ""


def compute_confusion_and_metrics(records: List[Dict]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Calculate the confusion matrix and metrics based on gt_label / pred_label.
    Positive class = "yes", Negative class = "no".
    """
    TP = TN = FP = FN = 0

    for rec in records:
        gt_raw = rec.get("gt_label", "")
        pred_raw = rec.get("pred_label", "")

        gt = normalize_label(gt_raw)
        pred = normalize_label(pred_raw)

        if gt not in ["yes", "no"] or pred not in ["yes", "no"]:
            continue

        if gt == "yes" and pred == "yes":
            TP += 1
        elif gt == "no" and pred == "no":
            TN += 1
        elif gt == "no" and pred == "yes":
            FP += 1
        elif gt == "yes" and pred == "no":
            FN += 1

    N = TP + TN + FP + FN

    if N == 0:
        raise ValueError("No valid (gt_label, pred_label) pairs found in the JSON file.")

    accuracy = (TP + TN) / N if N > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    confusion = {
        "N": N,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    return confusion, metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PRED_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    confusion, metrics = compute_confusion_and_metrics(records)

    table_rows = [
        ("N", confusion["N"]),
        ("TP", confusion["TP"]),
        ("TN", confusion["TN"]),
        ("FP", confusion["FP"]),
        ("FN", confusion["FN"]),
        ("Accuracy", round(metrics["Accuracy"], 4)),
        ("Precision", round(metrics["Precision"], 4)),
        ("Recall", round(metrics["Recall"], 4)),
        ("F1", round(metrics["F1"], 4)),
    ]

    print(f"Prediction file: {PRED_JSON}\n")
    print("{:<10s}{}".format("Metric", "Value"))
    print("-" * 22)
    for k, v in table_rows:
        print("{:<10s}{}".format(k, v))

    lines = [
        f"Prediction file: {PRED_JSON}",
        "",
        "{:<10s}{}".format("Metric", "Value"),
        "-" * 22,
    ]
    lines.extend("{:<10s}{}".format(k, v) for k, v in table_rows)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    import csv

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in table_rows:
            writer.writerow([k, v])

    print(f"\nMetrics (txt) saved to: {OUTPUT_TXT}")
    print(f"Metrics (csv) saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
