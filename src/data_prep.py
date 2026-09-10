"""
Loads the raw Twitter customer-support CSV (real Kaggle twcs.csv, or the
bundled synthetic sample as a fallback), filters to one brand, reconstructs
customer -> brand-reply thread pairs, cleans text, and writes a subsample.

Usage:
    python src/data_prep.py --input data/raw/twcs.csv --brand AmazonHelp --out data/processed_threads.csv
    python src/data_prep.py --sample-fallback --out data/processed_threads.csv
"""

import argparse
import re
import sys
import pandas as pd

HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("", text)
    text = HANDLE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_threads(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """
    Reconstruct (customer_message -> brand_reply) pairs.
    A brand reply is any inbound=False tweet authored by the brand handle
    whose in_response_to_tweet_id points at a customer (inbound=True) tweet.
    """
    df = df.copy()
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype(str)

    by_id = df.set_index("tweet_id")

    brand_replies = df[
        (df["author_id"] == brand_handle) & (df["inbound"].astype(str) == "False")
    ]

    records = []
    for _, reply in brand_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id in ("", "nan", "None") or parent_id not in by_id.index:
            continue
        cust_msg = by_id.loc[parent_id]
        if isinstance(cust_msg, pd.DataFrame):  # duplicate ids, take first
            cust_msg = cust_msg.iloc[0]
        if str(cust_msg.get("inbound")) != "True":
            continue
        records.append(
            {
                "customer_tweet_id": parent_id,
                "customer_text": clean_text(cust_msg["text"]),
                "customer_created_at": cust_msg.get("created_at", ""),
                "brand_tweet_id": reply["tweet_id"],
                "brand_reply_text": clean_text(reply["text"]),
            }
        )

    out = pd.DataFrame(records)
    out = out[out["customer_text"].str.len() > 3]
    out = out.drop_duplicates(subset=["customer_text", "brand_reply_text"])
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/twcs.csv", help="Path to raw Kaggle twcs.csv")
    ap.add_argument("--brand", default=None, help="Brand author_id/handle to filter to, e.g. AmazonHelp")
    ap.add_argument("--out", default="data/processed_threads.csv")
    ap.add_argument("--max-threads", type=int, default=5000)
    ap.add_argument(
        "--sample-fallback",
        action="store_true",
        help="If set, use the bundled synthetic sample instead of --input (for demo/dev).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--emit-eval-candidates",
        action="store_true",
        help="Also write a stratified-by-keyword sample to eval/golden_candidates.csv "
        "for you to hand-label into the real golden set.",
    )
    ap.add_argument("--eval-candidates-n", type=int, default=220)
    args = ap.parse_args()

    if args.sample_fallback:
        path = "data/sample/sample_tweets.csv"
        brand = "GadgetCoSupport"
        print(f"[data_prep] --sample-fallback set: using synthetic sample at {path} (brand={brand})")
    else:
        path = args.input
        brand = args.brand
        if brand is None:
            print("ERROR: --brand is required unless --sample-fallback is set.", file=sys.stderr)
            sys.exit(1)

    try:
        df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        print(
            f"ERROR: {path} not found.\n"
            f"Either download the real Kaggle dataset to data/raw/twcs.csv, "
            f"or run with --sample-fallback to use the bundled synthetic sample.",
            file=sys.stderr,
        )
        sys.exit(1)

    threads = build_threads(df, brand)
    print(f"[data_prep] Reconstructed {len(threads)} customer->brand reply pairs for brand={brand}")

    if len(threads) > args.max_threads:
        threads = threads.sample(n=args.max_threads, random_state=args.seed).reset_index(drop=True)
        print(f"[data_prep] Subsampled down to {args.max_threads} threads")

    threads.to_csv(args.out, index=False)
    print(f"[data_prep] Wrote {len(threads)} rows to {args.out}")

    if args.emit_eval_candidates:
        emit_eval_candidates(threads, args.eval_candidates_n, args.seed)


def emit_eval_candidates(threads: pd.DataFrame, n: int, seed: int):
    """
    Writes a diverse sample of customer_text to eval/golden_candidates.csv for
    you to hand-label (fill in true_intent, true_escalate, escalate_reason).
    Uses simple keyword-based stratification so rare issue types aren't
    drowned out by common ones, instead of pure random sampling.
    """
    import os

    keyword_groups = {
        "delivery": ["track", "deliver", "arrived", "package", "shipping", "lost"],
        "refund": ["refund", "return", "exchange", "money back"],
        "defect": ["broken", "defect", "wrong item", "not working", "damaged", "cracked"],
        "account": ["login", "password", "account", "hacked", "locked out"],
        "billing": ["charge", "billed", "billing", "payment"],
    }

    def group_of(text):
        t = str(text).lower()
        for g, kws in keyword_groups.items():
            if any(kw in t for kw in kws):
                return g
        return "other"

    threads = threads.copy()
    threads["_group"] = threads["customer_text"].apply(group_of)

    per_group = max(1, n // max(1, threads["_group"].nunique()))
    parts = []
    for g, sub in threads.groupby("_group"):
        parts.append(sub.sample(n=min(per_group, len(sub)), random_state=seed))
    candidates = pd.concat(parts).sample(frac=1, random_state=seed).head(n).reset_index(drop=True)

    candidates_out = candidates[["customer_text", "brand_reply_text"]].copy()
    candidates_out["true_intent"] = ""  # fill in by hand
    candidates_out["reference_reply"] = candidates_out["brand_reply_text"]
    candidates_out["true_escalate"] = ""  # fill in TRUE/FALSE by hand
    candidates_out["escalate_reason"] = ""  # fill in by hand

    os.makedirs("eval", exist_ok=True)
    candidates_out.to_csv("eval/golden_candidates.csv", index=False)
    print(
        f"[data_prep] Wrote {len(candidates_out)} stratified candidates to "
        f"eval/golden_candidates.csv — hand-label true_intent/true_escalate/escalate_reason, "
        f"then save as eval/golden_set.csv"
    )


if __name__ == "__main__":
    main()
