"""
Convenience script: samples N rows from eval/predictions_llm_judged.csv into
eval/human_ratings.csv, with an empty human_overall column for you to fill in
by hand before running src/judge_agreement.py.

Usage:
    python scripts/sample_for_human_rating.py --n 40
"""
import argparse
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default="eval/predictions_llm_judged.csv")
    ap.add_argument("--out", default="eval/human_ratings.csv")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.judged)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)[["customer_text", "pred_reply"]].copy()
    sample["human_overall"] = ""  # fill in 1-5 by hand, based on YOUR read of the reply
    sample["human_notes"] = ""
    sample.to_csv(args.out, index=False)
    print(f"Wrote {len(sample)} rows to {args.out}. Fill in human_overall (1-5) by hand.")
