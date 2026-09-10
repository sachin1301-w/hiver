"""
Evaluation harness. For each predictions CSV (from pipeline.py):

1. Automated metrics: intent accuracy/F1, escalation accuracy/precision/recall
   (against the golden set's true_intent / true_escalate columns)
2. LLM-as-judge: scores each generated reply 1-5 on relevance, groundedness,
   correctness, and tone, using a structured rubric, and writes per-row
   scores to <predictions>_judged.csv for later human-agreement checking.

Usage:
    python src/eval_harness.py --predictions eval/predictions_trivial.csv \
        eval/predictions_simple.csv eval/predictions_llm.csv \
        --golden eval/golden_set.csv --out reports/eval_results.json
"""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llm_client import complete_json, LLMNotConfiguredError

JUDGE_SYSTEM = """You are grading a customer support reply for quality. Score it on each
dimension from 1 (poor) to 5 (excellent):

- relevance: does the reply actually address what the customer asked?
- groundedness: does it avoid inventing policies/facts not supported by the
  brand's historical replies, and stay appropriately general when unsure?
- correctness: is the reply's guidance the kind of thing this brand would
  actually do (based on the reference reply / historical pattern)?
- tone: is it appropriately professional, empathetic, and on-brand?

Return JSON: {"relevance": <1-5>, "groundedness": <1-5>, "correctness": <1-5>,
"tone": <1-5>, "overall": <1-5>, "notes": "<one short sentence>"}"""


def judge_reply(customer_text: str, reference_reply: str, generated_reply: str) -> dict:
    user = (
        f"Customer message: {customer_text}\n\n"
        f"Reference (how the brand historically actually replied to similar issues): {reference_reply}\n\n"
        f"Generated reply to grade: {generated_reply}"
    )
    return complete_json(JUDGE_SYSTEM, user, max_tokens=900)


def automated_metrics(preds: pd.DataFrame) -> dict:
    metrics = {}
    has_true_intent = "true_intent" in preds.columns and preds["true_intent"].notna().any()
    has_true_escalate = "true_escalate" in preds.columns and preds["true_escalate"].notna().any()

    if has_true_intent:
        y_true = preds["true_intent"].astype(str)
        y_pred = preds["pred_intent"].astype(str)
        metrics["intent_accuracy"] = accuracy_score(y_true, y_pred)
        metrics["intent_macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if has_true_escalate:
        y_true_esc = preds["true_escalate"].astype(str).str.lower().isin(["true", "1", "yes"])
        y_pred_esc = preds["pred_escalate"].astype(str).str.lower().isin(["true", "1", "yes"])
        metrics["escalation_accuracy"] = accuracy_score(y_true_esc, y_pred_esc)
        metrics["escalation_precision"] = precision_score(y_true_esc, y_pred_esc, zero_division=0)
        metrics["escalation_recall"] = recall_score(y_true_esc, y_pred_esc, zero_division=0)

    return metrics


def run_judge(preds: pd.DataFrame, limit: int = None) -> pd.DataFrame:
    rows = []
    subset = preds.head(limit) if limit else preds
    for i, row in subset.iterrows():
        try:
            score = judge_reply(row["customer_text"], row.get("reference_reply", ""), row["pred_reply"])
        except LLMNotConfiguredError:
            raise
        except Exception as e:
            score = {"relevance": None, "groundedness": None, "correctness": None, "tone": None, "overall": None, "notes": f"judge error: {e}"}
        merged = {**row.to_dict(), **{f"judge_{k}": v for k, v in score.items()}}
        rows.append(merged)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--golden", default="eval/golden_set.csv")
    ap.add_argument("--out", default="reports/eval_results.json")
    ap.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge (no API calls, metrics only)")
    ap.add_argument("--judge-limit", type=int, default=None, help="Only judge first N rows (save API cost)")
    args = ap.parse_args()

    results = {}
    for pred_path in args.predictions:
        name = os.path.splitext(os.path.basename(pred_path))[0].replace("predictions_", "")
        preds = pd.read_csv(pred_path)
        metrics = automated_metrics(preds)

        if not args.skip_judge:
            try:
                judged = run_judge(preds, limit=args.judge_limit)
                judged_path = pred_path.replace(".csv", "_judged.csv")
                judged.to_csv(judged_path, index=False)
                for col in ["judge_relevance", "judge_groundedness", "judge_correctness", "judge_tone", "judge_overall"]:
                    if col in judged.columns:
                        metrics[col + "_mean"] = judged[col].dropna().astype(float).mean() if judged[col].notna().any() else None
                print(f"[eval_harness] Wrote per-row judge scores to {judged_path}")
            except LLMNotConfiguredError as e:
                print(f"[eval_harness] Skipping LLM-judge for {name}: {e}")

        results[name] = metrics
        print(f"\n=== {name} ===")
        print(json.dumps(metrics, indent=2, default=str))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[eval_harness] Wrote combined results to {args.out}")

    # also write a quick human-readable markdown table
    md_path = args.out.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        all_keys = sorted({k for m in results.values() for k in m.keys()})
        f.write("| metric | " + " | ".join(results.keys()) + " |\n")
        f.write("|---|" + "---|" * len(results) + "\n")
        for k in all_keys:
            row = [f"{results[sys_name].get(k, ''):.3f}" if isinstance(results[sys_name].get(k), float) else str(results[sys_name].get(k, "")) for sys_name in results]
            f.write(f"| {k} | " + " | ".join(row) + " |\n")
    print(f"[eval_harness] Wrote markdown summary to {md_path}")


if __name__ == "__main__":
    main()
