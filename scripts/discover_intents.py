"""
Reproduces the unsupervised clustering that the 9-intent taxonomy in
src/intents.py was actually derived from. This is NOT part of the runtime
pipeline — it's the exploratory step a human reads before naming intents.

Method: TF-IDF (word 1-2 grams) -> TruncatedSVD(200) (LSA) -> MiniBatchKMeans,
plus an NMF cross-check over the same TF-IDF matrix. For each k-means cluster
we print the terms with the highest lift over the corpus-wide mean (so common
support vocabulary like "please"/"help" doesn't dominate every cluster) and a
sample of real messages.

Usage:
    python scripts/discover_intents.py --data data/processed_threads.csv --k 16
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_threads.csv")
    ap.add_argument("--text-col", default="customer_text")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed)
    texts = df[args.text_col].fillna("").astype(str).tolist()

    vec = TfidfVectorizer(
        min_df=8, max_df=0.35, ngram_range=(1, 2), sublinear_tf=True,
        stop_words="english",
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    print(f"[discover] {X.shape[0]:,} docs x {X.shape[1]:,} terms\n")

    lsa = make_pipeline(TruncatedSVD(200, random_state=args.seed), Normalizer(copy=False))
    Z = lsa.fit_transform(X)
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=10, batch_size=2048)
    labels = km.fit_predict(Z)

    global_mean = np.asarray(X.mean(axis=0)).ravel()
    print("=" * 78)
    print("K-MEANS CLUSTERS (terms ranked by lift over the global mean)")
    print("=" * 78)
    for c in range(args.k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        centroid = np.asarray(X[mask].mean(axis=0)).ravel()
        lift = centroid - global_mean
        top = terms[np.argsort(lift)[::-1][:14]]
        print(f"\n--- cluster {c}  (n={mask.sum():,}, {mask.mean():.1%}) ---")
        print("terms:", ", ".join(top))
        for msg in df.loc[mask, args.text_col].head(args.examples):
            print("   *", str(msg)[:150])

    print("\n" + "=" * 78)
    print("NMF TOPICS (soft, overlapping — a cross-check on k-means)")
    print("=" * 78)
    nmf = NMF(n_components=args.k, random_state=args.seed, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    for i, comp in enumerate(nmf.components_):
        top = terms[np.argsort(comp)[::-1][:12]]
        print(f"topic {i:>2} (mass={W[:, i].sum():8.1f}): {', '.join(top)}")


if __name__ == "__main__":
    main()
