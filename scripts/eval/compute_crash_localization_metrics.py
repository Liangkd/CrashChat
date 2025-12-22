#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute AP@30, AP@50, AP@70 and mIoU for crash_localization results.

Assume the JSON file is a list, and each element looks like this:
{
    "data_id": 148,
    "video_path": "...",
    "prompt": "...",
    "gt_start": 8.3,
    "gt_end": 9.1,
    "pred_start": 8.0,
    "pred_end": 9.1,
    "raw_response": "8.0 - 9.1 seconds"
}
"""

import json
import os
from typing import List, Dict, Tuple

PRED_JSON = (
    "/outputs/crash_localization_evaluation_results/independent_monotask_models_crash_localization/crashchat_dada_video_total_crash_localization_test_predict.json"
)

OUTPUT_DIR = (
    "/outputs/crash_localization_evaluation_results/independent_monotask_models_crash_localization"
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "crashchat_dada_video_crash_localization_test_metrics.txt"
)


def compute_iou(gt_start: float, gt_end: float,
                pred_start: float, pred_end: float) -> float:
    """Calculate the IoU (Intersection over Union) for the time intervals."""
    if gt_end < gt_start:
        gt_start, gt_end = gt_end, gt_start
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start

    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(gt_start, pred_start)
    union_end = max(gt_end, pred_end)
    union = max(0.0, union_end - union_start)

    if union <= 0.0:
        return 0.0
    return intersection / union


def evaluate_temporal_localization(
    records: List[Dict],
    iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
):
    """
    Calculate mIoU and "AP" (actually Accuracy@IoU>=thr) at various IoU thresholds.
    """
    total = 0
    sum_iou = 0.0

    hit_counts = {thr: 0 for thr in iou_thresholds}

    for rec in records:
        if ("gt_start" not in rec or "gt_end" not in rec or
                "pred_start" not in rec or "pred_end" not in rec):
            continue

        gt_start = float(rec["gt_start"])
        gt_end = float(rec["gt_end"])
        pred_start = float(rec["pred_start"])
        pred_end = float(rec["pred_end"])

        iou = compute_iou(gt_start, gt_end, pred_start, pred_end)

        total += 1
        sum_iou += iou

        for thr in iou_thresholds:
            if iou >= thr:
                hit_counts[thr] += 1

    if total == 0:
        raise ValueError("No valid records found in the JSON file.")

    miou = sum_iou / total
    ap_results = {thr: hit_counts[thr] / total for thr in iou_thresholds}

    return miou, ap_results, total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PRED_JSON, "r") as f:
        records = json.load(f)

    miou, ap_results, total = evaluate_temporal_localization(records)

    print(f"Prediction file: {PRED_JSON}")
    print(f"Total samples: {total}")
    print(f"mIoU: {miou:.4f}")
    print(f"AP@30 (IoU>=0.3): {ap_results[0.3]:.4f}")
    print(f"AP@50 (IoU>=0.5): {ap_results[0.5]:.4f}")
    print(f"AP@70 (IoU>=0.7): {ap_results[0.7]:.4f}")

    lines = [
        f"Prediction file: {PRED_JSON}",
        f"Total samples: {total}",
        f"mIoU: {miou:.4f}",
        f"AP@30 (IoU>=0.3): {ap_results[0.3]:.4f}",
        f"AP@50 (IoU>=0.5): {ap_results[0.5]:.4f}",
        f"AP@70 (IoU>=0.7): {ap_results[0.7]:.4f}",
        "",
    ]
    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(lines))

    print(f"\nMetrics saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
