#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute TTA-based metrics for pre_crash_localization.


     Example of each record:
     {
         "data_id": 442,
         "video_path": "/lustre/.../data/videos/cap_11_003078.mp4",
         "prompt": "...",
         "gt_time": 4.6,
         "pred_time": 3.0,
         "raw_response": "3.0 seconds"
     }

  2) TL GT JSON (Time interval of the crash):

     example：
     {
       "video": ["videos/cap_11_010721.mp4"],
       "conversations": [
         {...},
         {"from": "gpt", "value": "7.5 - 15.0 seconds"}
       ],
       "source": "MM-AU-CAP"
     }

    Processing Logic:
       - For each AL sample, find the corresponding video in the TL ground truth using basename(video_path),
          and obtain the accident start time t_acc (7.5 in the example). 
       - Calculate:
          TTA_gt   = t_acc - gt_time
          TTA_pred = max(t_acc - pred_time, 0)   # If the prediction is after the accident, TTA_pred = 0

    - To avoid "premature false alarms," ​​an early tolerance window delta is introduced:
          delta = 0.5  # seconds

    If pred_time >= t_acc:
        -> Prediction is after the accident, considered anticipation failure, IoU = 0

    If pred_time < t_acc:
        1) pred_time < gt_time - delta:
               Prediction is too early, considered unrealistic/false alarm
               -> IoU = 0 (but TTA_pred is still calculated as t_acc - pred_time and included in mTTA for analysis)

        2) gt_time - delta <= pred_time < gt_time:
               Within a small early window, considered reasonable anticipation
               -> In IoU calculation, pred is treated as simultaneous with gt:
                  TTA_pred_for_iou = TTA_gt  => IoU = 1

        3) gt_time <= pred_time < t_acc:
               Normal slightly late prediction
               -> TTA_pred_for_iou = TTA_pred (t_acc - pred_time)
               -> IoU = min(TTA_gt, TTA_pred_for_iou) / max(TTA_gt, TTA_pred_for_iou)

  - Statistics:
        mTTA_gt   = mean(TTA_gt)
        mTTA_pred = mean(TTA_pred)
        TTA_MAE   = mean(|TTA_pred - TTA_gt|)
        mIoU      = mean(IoU)
        AP@30/50/70 = mean(1[IoU >= thr]) for thr in {0.3, 0.5, 0.7}
"""

import json
import os
import re
from typing import Dict, List, Tuple

PRED_JSON = (
    "/outputs/pre_crash_localization_evaluation_results/independent_monotask_models_pre_crash_localization/crashchat_dada_video_total_pre_crash_localization_test_predict.json"
)

TL_GT_JSON = (
    "/data/crashchat_dada_video_total_crash_localization_test.json"
)

OUTPUT_DIR = (
    "/outputs/pre_crash_localization_evaluation_results/independent_monotask_models_pre_crash_localization"
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "crashchat_dada_video_pre_crash_localization_test_metrics.txt"
)

# Early tolerance window (seconds): Predictions within delta seconds before gt_time are considered reasonably early.
EARLY_TOLERANCE_DELTA = 0.5

# IoU threshold, used to calculate AP@thr
IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def parse_time_range(text: str) -> Tuple[float, float]:
    """
    Parse (start, end) from the TL string, for example:
      "7.5 - 15.0 seconds"
      "6.0-7.0s"
    If parsing fails, it returns (0.0, 0.0).
    """
    if not isinstance(text, str):
        return 0.0, 0.0

    m = re.search(r"([\d\.]+)\s*-\s*([\d\.]+)", text)
    if m:
        try:
            s = float(m.group(1))
            e = float(m.group(2))
            return s, e
        except Exception:
            pass

    nums = re.findall(r"[\d\.]+", text)
    if len(nums) >= 2:
        try:
            s = float(nums[0])
            e = float(nums[1])
            return s, e
        except Exception:
            pass

    return 0.0, 0.0


def load_accident_start_times(tl_json_path: str) -> Dict[str, float]:
    """
    Read the accident start time t_acc (start of crash) for each video from the TL GT JSON file.

    return:
        video_basename -> t_acc
    """
    with open(tl_json_path, "r") as f:
        records = json.load(f)

    video2acc_start: Dict[str, float] = {}

    for rec in records:
        video_list = rec.get("video", [])
        if not video_list:
            continue
        rel_video_path = video_list[0]  
        basename = os.path.basename(rel_video_path)

        convs = rec.get("conversations", [])
        if len(convs) < 2:
            continue
        gt_str = convs[1].get("value", "")
        start, _ = parse_time_range(gt_str)

        if start <= 0.0:
            continue

        
        if basename not in video2acc_start:
            video2acc_start[basename] = start

    return video2acc_start


def evaluate_anticipation_localization(
    al_records: List[Dict],
    video2acc_start: Dict[str, float],
    iou_thresholds: Tuple[float, ...] = IOU_THRESHOLDS,
    early_tolerance_delta: float = EARLY_TOLERANCE_DELTA,
):
    """
    Based on the AL results, calculate metrics based on TTA.

    return:
      metrics: dict
      stats:   Some supplementary statistical information
    """
    total = 0
    sum_iou = 0.0

    sum_tta_gt = 0.0
    sum_tta_pred = 0.0
    sum_tta_abs_err = 0.0

    hit_counts = {thr: 0 for thr in iou_thresholds}

    missing_video_count = 0
    bad_gt_tta_count = 0
    late_pred_count = 0
    too_early_pred_count = 0

    for rec in al_records:
        video_path = rec.get("video_path", "")
        basename = os.path.basename(video_path)

        if basename not in video2acc_start:
            missing_video_count += 1
            continue

        t_acc = float(video2acc_start[basename])

        if "gt_time" not in rec or "pred_time" not in rec:
            continue

        gt_time = float(rec["gt_time"])
        pred_time = float(rec["pred_time"])

        
        TTA_gt = t_acc - gt_time
        TTA_pred_raw = t_acc - pred_time
        if TTA_pred_raw < 0.0:
            TTA_pred_raw = 0.0

        if TTA_gt <= 0.0:
            bad_gt_tta_count += 1
            continue

        sum_tta_gt += TTA_gt
        sum_tta_pred += TTA_pred_raw
        sum_tta_abs_err += abs(TTA_pred_raw - TTA_gt)

        if pred_time >= t_acc:
            IoU = 0.0
            late_pred_count += 1

        else:
            if pred_time < gt_time - early_tolerance_delta:
                IoU = 0.0
                too_early_pred_count += 1
            elif pred_time < gt_time:
                TTA_pred_for_iou = TTA_gt
                # IoU = 1
                IoU = 1.0
            else:
                TTA_pred_for_iou = TTA_pred_raw
                if TTA_pred_for_iou <= 0.0:
                    IoU = 0.0
                else:
                    IoU = min(TTA_gt, TTA_pred_for_iou) / max(TTA_gt, TTA_pred_for_iou)

        total += 1
        sum_iou += IoU

        for thr in iou_thresholds:
            if IoU >= thr:
                hit_counts[thr] += 1

    if total == 0:
        raise ValueError("No valid AL records found for evaluation.")

    mIoU = sum_iou / total
    ap_results = {thr: hit_counts[thr] / total for thr in iou_thresholds}

    mTTA_gt = sum_tta_gt / total
    mTTA_pred = sum_tta_pred / total
    TTA_mae = sum_tta_abs_err / total

    metrics = {
        "Num_samples": float(total),
        "mIoU": float(mIoU),
        "AP@30": float(ap_results[0.3]),
        "AP@50": float(ap_results[0.5]),
        "AP@70": float(ap_results[0.7]),
        "mTTA_gt": float(mTTA_gt),
        "mTTA_pred": float(mTTA_pred),
        "TTA_MAE": float(TTA_mae),
    }

    stats = {
        "missing_video_count": int(missing_video_count),
        "bad_gt_tta_count": int(bad_gt_tta_count),
        "late_pred_count": int(late_pred_count),
        "too_early_pred_count": int(too_early_pred_count),
    }

    return metrics, stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video2acc_start = load_accident_start_times(TL_GT_JSON)

    with open(PRED_JSON, "r") as f:
        al_records = json.load(f)

    metrics, stats = evaluate_anticipation_localization(
        al_records,
        video2acc_start=video2acc_start,
        iou_thresholds=IOU_THRESHOLDS,
        early_tolerance_delta=EARLY_TOLERANCE_DELTA,
    )

    print(f"Prediction file: {PRED_JSON}")
    print(f"Total valid samples: {int(metrics['Num_samples'])}")
    print(f"mIoU (TTA-IoU): {metrics['mIoU']:.4f}")
    print(f"AP@30 (IoU>=0.3): {metrics['AP@30']:.4f}")
    print(f"AP@50 (IoU>=0.5): {metrics['AP@50']:.4f}")
    print(f"AP@70 (IoU>=0.7): {metrics['AP@70']:.4f}")
    print(f"mTTA_gt:   {metrics['mTTA_gt']:.4f} seconds")
    print(f"mTTA_pred: {metrics['mTTA_pred']:.4f} seconds")
    print(f"TTA_MAE:   {metrics['TTA_MAE']:.4f} seconds")
    print()
    print(f"Missing videos in TL GT: {stats['missing_video_count']}")
    print(f"Bad GT TTA (TTA_gt<=0): {stats['bad_gt_tta_count']}")
    print(f"Late predictions (pred_time>=t_acc): {stats['late_pred_count']}")
    print(f"Too-early predictions (pred_time<gt_time-delta): {stats['too_early_pred_count']}")
    print(f"(early_tolerance_delta = {EARLY_TOLERANCE_DELTA} seconds)")

    lines = [
        f"Prediction file: {PRED_JSON}",
        f"Temporal GT file: {TL_GT_JSON}",
        f"Num_samples: {int(metrics['Num_samples'])}",
        f"mIoU (TTA-IoU): {metrics['mIoU']:.4f}",
        f"AP@30 (IoU>=0.3): {metrics['AP@30']:.4f}",
        f"AP@50 (IoU>=0.5): {metrics['AP@50']:.4f}",
        f"AP@70 (IoU>=0.7): {metrics['AP@70']:.4f}",
        f"mTTA_gt:   {metrics['mTTA_gt']:.4f} seconds",
        f"mTTA_pred: {metrics['mTTA_pred']:.4f} seconds",
        f"TTA_MAE:   {metrics['TTA_MAE']:.4f} seconds",
        "",
        f"Missing videos in TL GT: {stats['missing_video_count']}",
        f"Bad GT TTA (TTA_gt<=0): {stats['bad_gt_tta_count']}",
        f"Late predictions (pred_time>=t_acc): {stats['late_pred_count']}",
        f"Too-early predictions (pred_time<gt_time-delta): {stats['too_early_pred_count']}",
        f"early_tolerance_delta: {EARLY_TOLERANCE_DELTA} seconds",
        "",
    ]
    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(lines))

    print(f"\nMetrics saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
