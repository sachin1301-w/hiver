"""
Runs one full system end-to-end (classify -> generate reply -> escalate)
over every row in the golden set, and writes predictions to a CSV that
eval_harness.py can score.

Usage:
    python src/pipeline.py --system trivial --golden eval/golden_set.csv \
        --data data/processed_threads.csv --out eval/predictions_trivial.csv

    python src/pipeline.py --system simple --golden eval/golden_set.csv \
        --data data/processed_threads.csv --out eval/predictions_simple.csv \
        --model models/intent_clf.joblib

    python src/pipeline.py --system llm --golden eval/golden_set.csv \
        --data data/processed_threads.csv --out eval/predictions_llm.csv
"""

import argparse
import sys
import time

import pandas as pd

from classify import TrivialClassifier, TfidfLogRegClassifier, LLMClassifier
from generate_reply import generate_template_reply, generate_llm_reply, TRIVIAL_REPLY
from escalate import trivial_escalate, rule_based_escalate, llm_escalate
from retrieve import TfidfRetriever
from llm_client import LLMNotConfiguredError


def run(system: str, golden_path: str, data_path: str, out_path: str, model_path: str = None, limit: int = None):
    golden = pd.read_csv(golden_path)
    if "customer_text" not in golden.columns:
        raise ValueError("golden set must have a 'customer_text' column")
    if limit:
        golden = golden.sample(n=min(limit, len(golden)), random_state=42).reset_index(drop=True)

    try:
        threads = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"WARNING: {data_path} not found; reply grounding/retrieval will be empty.", file=sys.stderr)
        threads = pd.DataFrame(columns=["customer_text", "brand_reply_text"])

    retriever = TfidfRetriever(threads) if len(threads) > 0 else None

    if system == "trivial":
        clf = TrivialClassifier().fit(golden["customer_text"], golden.get("true_intent", golden["customer_text"]))
    elif system == "simple":
        if model_path:
            clf = TfidfLogRegClassifier.load(model_path)
        else:
            clf = TfidfLogRegClassifier().fit(golden["customer_text"], golden["true_intent"])
    elif system == "llm":
        clf = LLMClassifier()
    else:
        raise ValueError(f"Unknown system: {system}")

    rows = []
    n_failed = 0
    for i, row in golden.iterrows():
        text = row["customer_text"]

        try:
            pred = clf.predict_one(text)
            intent = pred["intent"]
            confidence = pred.get("confidence", 0.5)

            if system == "trivial":
                reply = TRIVIAL_REPLY
            elif system == "simple":
                reply = generate_template_reply(intent)
            else:  # llm
                retrieved = retriever.retrieve(text, k=3) if retriever else []
                reply = generate_llm_reply(text, intent, retrieved)

            if system == "trivial":
                esc = trivial_escalate(text, intent, confidence)
            elif system == "simple":
                esc = rule_based_escalate(text, intent, confidence)
            else:
                esc = llm_escalate(text, intent, confidence)

        except LLMNotConfiguredError as e:
            # No key at all — every remaining row would fail identically, so
            # stop the whole run instead of burning 200 rows on the same error.
            print(f"\n[pipeline] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            # A single row that fails even after llm_client's own retries
            # (persistent empty completions, an unrecoverable API error) is
            # not a reason to lose the other ~199 rows of a paid/rate-limited
            # run. Record it as a system-level escalation with the failure
            # visible, and keep going.
            n_failed += 1
            print(f"[pipeline:{system}] row {i} failed: {e}", file=sys.stderr)
            intent, confidence = "other", 0.0
            reply = ""
            esc = {
                "escalate": True,
                "reason": f"System error during generation, routed to a human: {e}",
            }

        rows.append(
            {
                "customer_text": text,
                "true_intent": row.get("true_intent", ""),
                "pred_intent": intent,
                "confidence": confidence,
                "pred_reply": reply,
                "reference_reply": row.get("reference_reply", ""),
                "pred_escalate": esc["escalate"],
                "true_escalate": row.get("true_escalate", ""),
                "escalate_reason": esc["reason"],
            }
        )

        if system == "llm":
            time.sleep(1.0)  # be polite to free-tier shared pools
            if (i + 1) % 10 == 0:
                print(f"[pipeline:{system}] {i + 1}/{len(golden)} done ({n_failed} failed so far)")

    if n_failed:
        print(f"[pipeline:{system}] {n_failed}/{len(golden)} rows failed and were recorded as system-error escalations.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"[pipeline:{system}] Wrote {len(out_df)} predictions to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["trivial", "simple", "llm"])
    ap.add_argument("--golden", default="eval/golden_set.csv")
    ap.add_argument("--data", default="data/processed_threads.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="Path to trained TF-IDF/LogReg model (for --system simple)")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N golden-set rows (save API cost/time)")
    args = ap.parse_args()

    run(args.system, args.golden, args.data, args.out, args.model, args.limit)
