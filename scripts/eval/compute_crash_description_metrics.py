#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute BLEU, ROUGE-1/2/L and BERTScore for accident captioning results.

Assume the prediction JSON file is a list, where each element looks like this:
{
    "data_id": 134,
    "video_path": "...",
    "prompt": "...",
    "gt_caption": "ego-car hits a car",
    "pred_caption": "ego-car hits a car",
    "raw_response": "ego-car hits a car"
}
"""

import json
import os
from typing import List, Dict, Tuple

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score

PRED_JSON = (
    "/outputs/crash_description_evaluation_results/independent_monotask_models_crash_description/crashchat_dada_video_total_crash_description_test_predict.json"
)

OUTPUT_DIR = (
    "/outputs/crash_description_evaluation_results/independent_monotask_models_crash_description"
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "crashchat_dada_video_crash_description_test_metrics.txt"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "crashchat_dada_video_crash_description_test_metrics.csv"
)


def load_refs_preds(records: List[Dict]) -> Tuple[List[str], List[str]]:
    """Extract gt_caption and pred_caption from the JSON data."""
    refs, preds = [], []
    for rec in records:
        gt = rec.get("gt_caption", "")
        pred = rec.get("pred_caption", "")
        gt = str(gt).strip()
        pred = str(pred).strip()
        if gt == "" or pred == "":
            continue
        refs.append(gt)
        preds.append(pred)
    if not refs or not preds:
        raise ValueError("No valid (gt_caption, pred_caption) pairs found.")
    assert len(refs) == len(preds)
    return refs, preds


def compute_bleu(refs: List[str], preds: List[str]) -> float:
    """Calculate corpus BLEU."""
    # 简单按空格切词
    refs_tok = [[r.split()] for r in refs]  # list of list of references
    preds_tok = [p.split() for p in preds]
    ch = SmoothingFunction()
    bleu = corpus_bleu(refs_tok, preds_tok, smoothing_function=ch.method1)
    return float(bleu)


def compute_rouge(refs: List[str], preds: List[str]) -> Dict[str, float]:
    """Calculate the average P/R/F values ​​for ROUGE-1, ROUGE-2, and ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    sums = {
        "rouge1_p": 0.0, "rouge1_r": 0.0, "rouge1_f": 0.0,
        "rouge2_p": 0.0, "rouge2_r": 0.0, "rouge2_f": 0.0,
        "rougeL_p": 0.0, "rougeL_r": 0.0, "rougeL_f": 0.0,
    }

    n = len(refs)
    for ref, pred in zip(refs, preds):
        scores = scorer.score(ref, pred)
        r1 = scores["rouge1"]
        r2 = scores["rouge2"]
        rL = scores["rougeL"]

        sums["rouge1_p"] += r1.precision
        sums["rouge1_r"] += r1.recall
        sums["rouge1_f"] += r1.fmeasure

        sums["rouge2_p"] += r2.precision
        sums["rouge2_r"] += r2.recall
        sums["rouge2_f"] += r2.fmeasure

        sums["rougeL_p"] += rL.precision
        sums["rougeL_r"] += rL.recall
        sums["rougeL_f"] += rL.fmeasure

    for k in sums:
        sums[k] /= n

    return sums


def compute_bertscore(refs: List[str], preds: List[str]) -> Dict[str, float]:
    """
    Calculate the average P/R/F1 scores for BERTScore.
    The English model (roberta-large) is used by default and will be downloaded automatically the first time it's used.
    """
    P, R, F1 = bertscore_score(preds, refs, lang="en")
    return {
        "P": float(P.mean().item()),
        "R": float(R.mean().item()),
        "F1": float(F1.mean().item()),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PRED_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    refs, preds = load_refs_preds(records)

    bleu = compute_bleu(refs, preds)
    rouge = compute_rouge(refs, preds)
    bert = compute_bertscore(refs, preds)

    table_rows = [
        ("BLEU", round(bleu, 4)),
        ("ROUGE-1_P", round(rouge["rouge1_p"], 4)),
        ("ROUGE-1_R", round(rouge["rouge1_r"], 4)),
        ("ROUGE-1_F", round(rouge["rouge1_f"], 4)),
        ("ROUGE-2_P", round(rouge["rouge2_p"], 4)),
        ("ROUGE-2_R", round(rouge["rouge2_r"], 4)),
        ("ROUGE-2_F", round(rouge["rouge2_f"], 4)),
        ("ROUGE-L_P", round(rouge["rougeL_p"], 4)),
        ("ROUGE-L_R", round(rouge["rougeL_r"], 4)),
        ("ROUGE-L_F", round(rouge["rougeL_f"], 4)),
        ("BERTScore_P", round(bert["P"], 4)),
        ("BERTScore_R", round(bert["R"], 4)),
        ("BERTScore_F1", round(bert["F1"], 4)),
    ]

    print(f"Prediction file: {PRED_JSON}\n")
    print("{:<15s}{}".format("Metric", "Value"))
    print("-" * 28)
    for k, v in table_rows:
        print("{:<15s}{}".format(k, v))

    lines = [
        f"Prediction file: {PRED_JSON}",
        "",
        "{:<15s}{}".format("Metric", "Value"),
        "-" * 28,
    ]
    lines.extend("{:<15s}{}".format(k, v) for k, v in table_rows)

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
