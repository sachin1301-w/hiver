"""
Three intent classifiers, matching the three systems compared in the report:

1. TrivialClassifier   -- always predicts the majority class from training data
2. TfidfLogRegClassifier -- TF-IDF + multinomial Logistic Regression, trained
   on the golden set (or any labeled data you provide)
3. LLMClassifier -- few-shot prompts the configured LLM with the intent
   taxonomy + examples, asks for a single intent label + confidence

Run this file directly to train & persist the TF-IDF/LogReg baseline:
    python src/classify.py --train --data data/processed_threads.csv \
        --golden eval/golden_set.csv --model-out models/intent_clf.joblib
"""

import argparse
import json
from collections import Counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from intents import INTENT_LIST, intents_prompt_block
from llm_client import complete_json


class TrivialClassifier:
    def __init__(self):
        self.majority_label = None

    def fit(self, texts, labels):
        self.majority_label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.majority_label for _ in texts]

    def predict_one(self, text: str) -> dict:
        return {"intent": self.majority_label, "confidence": 1.0}


class TfidfLogRegClassifier:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_one(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_idx = proba.argmax()
        return {"intent": classes[best_idx], "confidence": float(proba[best_idx])}

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.pipeline = joblib.load(path)
        return obj


class LLMClassifier:
    """Few-shot LLM intent classification with a confidence self-report."""

    SYSTEM = (
        "You are an intent classifier for customer support tweets. "
        "Classify the customer's message into exactly one of these intents:\n\n"
        f"{intents_prompt_block()}\n\n"
        f"Valid intent values: {json.dumps(INTENT_LIST)}\n"
        "Return JSON: {\"intent\": <one of the valid values>, \"confidence\": <0.0-1.0>, "
        "\"reasoning\": <one short sentence>}"
    )

    def predict_one(self, text: str) -> dict:
        result = complete_json(self.SYSTEM, f"Customer message: {text}")
        if result.get("intent") not in INTENT_LIST:
            result["intent"] = "other"
            result["confidence"] = min(result.get("confidence", 0.5), 0.5)
        return result


def train_and_eval(data_path: str, golden_path: str, model_out: str):
    golden = pd.read_csv(golden_path)
    if "true_intent" not in golden.columns:
        raise ValueError("golden set must have a 'true_intent' column")

    train_df, test_df = train_test_split(
        golden, test_size=0.25, random_state=42, stratify=golden["true_intent"]
    )

    trivial = TrivialClassifier().fit(train_df["customer_text"], train_df["true_intent"])
    trivial_preds = trivial.predict(test_df["customer_text"])

    tfidf = TfidfLogRegClassifier().fit(train_df["customer_text"], train_df["true_intent"])
    tfidf_preds = tfidf.predict(test_df["customer_text"])

    print("=== Trivial (majority-class) baseline ===")
    print(classification_report(test_df["true_intent"], trivial_preds, zero_division=0))

    print("=== TF-IDF + Logistic Regression baseline ===")
    print(classification_report(test_df["true_intent"], tfidf_preds, zero_division=0))

    tfidf.save(model_out)
    print(f"Saved trained TF-IDF/LogReg model to {model_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", default="data/processed_threads.csv")
    ap.add_argument("--golden", default="eval/golden_set.csv")
    ap.add_argument("--model-out", default="models/intent_clf.joblib")
    args = ap.parse_args()

    if args.train:
        train_and_eval(args.data, args.golden, args.model_out)
