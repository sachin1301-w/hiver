"""
Measures how well the LLM-as-judge agrees with a real human rater. This is
the piece the assignment explicitly requires ("evidence of how well your
judge agrees with a human") and the piece most people skip.

Workflow:
1. Run eval_harness.py (produces eval/predictions_llm_judged.csv with
   judge_overall scores per row).
2. Copy ~30-50 rows of that file into eval/human_ratings.csv and fill in
   YOUR OWN 1-5 "human_overall" score for each, by actually reading them.
3. Run this script to get Pearson correlation + mean absolute difference
   between your scores and the judge's scores.

Usage:
    python src/judge_agreement.py --judge eval/predictions_llm_judged.csv --human eval/human_ratings.csv
"""

import argparse
import pandas as pd
from scipy.stats import pearsonr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="eval/predictions_llm_judged.csv")
    ap.add_argument("--human", default="eval/human_ratings.csv")
    args = ap.parse_args()

    judge_df = pd.read_csv(args.judge)
    human_df = pd.read_csv(args.human)

    if "human_overall" not in human_df.columns:
        raise ValueError("human_ratings.csv must have a 'human_overall' column (your 1-5 scores)")

    merged = human_df.merge(
        judge_df[["customer_text", "judge_overall"]],
        on="customer_text",
        how="left",
    ).dropna(subset=["human_overall", "judge_overall"])

    if len(merged) < 3:
        print(
            f"Only {len(merged)} rows have both human and judge scores — need at least a "
            f"handful to compute meaningful agreement. Fill in more of human_ratings.csv."
        )
        return

    r, p = pearsonr(merged["human_overall"], merged["judge_overall"])
    mad = (merged["human_overall"] - merged["judge_overall"]).abs().mean()
    exact_match_rate = (merged["human_overall"] == merged["judge_overall"]).mean()

    print(f"N examples compared: {len(merged)}")
    print(f"Pearson correlation (human vs LLM-judge overall score): r={r:.3f} (p={p:.4f})")
    print(f"Mean absolute difference: {mad:.3f} (on a 1-5 scale)")
    print(f"Exact match rate: {exact_match_rate:.1%}")
    print(
        "\nInterpretation guide: r > 0.6 is generally considered reasonable agreement for "
        "subjective quality rubrics; r < 0.3 means the judge shouldn't be trusted as a proxy "
        "for human judgment without more rubric refinement."
    )


if __name__ == "__main__":
    main()
