"""
Retrieval for reply-grounding: given a new customer message, find the most
similar historical customer messages (for this brand) and return how the
brand actually replied to them. This is the "grounded in how the brand has
historically resolved similar issues" requirement from the assignment.

Default implementation: TF-IDF + cosine similarity (fast, no extra deps,
good enough for a few thousand threads). If you have more time/compute, swap
in EmbeddingRetriever which uses sentence-transformers embeddings for denser
semantic matching (better on paraphrases the TF-IDF misses).
"""

from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedExample:
    customer_text: str
    brand_reply_text: str
    similarity: float


class TfidfRetriever:
    def __init__(self, threads_df: pd.DataFrame):
        """threads_df must have columns: customer_text, brand_reply_text"""
        self.df = threads_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words="english", min_df=1
        )
        self.matrix = self.vectorizer.fit_transform(self.df["customer_text"].fillna(""))

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedExample]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [
            RetrievedExample(
                customer_text=self.df.iloc[i]["customer_text"],
                brand_reply_text=self.df.iloc[i]["brand_reply_text"],
                similarity=float(sims[i]),
            )
            for i in top_idx
            if sims[i] > 0
        ]


class EmbeddingRetriever:
    """
    Optional denser retriever using sentence-transformers. Not used by
    default (extra heavy dependency); swap TfidfRetriever -> EmbeddingRetriever
    in pipeline.py if you install `sentence-transformers` and want better
    recall on paraphrased/semantically-similar-but-lexically-different
    messages.
    """

    def __init__(self, threads_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.np = np
        self.df = threads_df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(self.df["customer_text"].fillna("").tolist())

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedExample]:
        q_emb = self.model.encode([query])
        sims = cosine_similarity(q_emb, self.embeddings).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [
            RetrievedExample(
                customer_text=self.df.iloc[i]["customer_text"],
                brand_reply_text=self.df.iloc[i]["brand_reply_text"],
                similarity=float(sims[i]),
            )
            for i in top_idx
            if sims[i] > 0
        ]
