"""
Flask backend for the support agent demo UI.

IMPORTANT: the LLM API key lives only here, server-side, read from the .env
file via src/llm_client.py. It is never sent to, or exposed in, the browser.
The frontend only ever talks to this backend's /api/* routes.

Run:
    cd backend
    python app.py
Then open http://localhost:5000 in a browser.

Endpoints:
    GET  /api/health           -> {status, llm_configured, data_loaded}
    POST /api/respond          -> {intent, confidence, reply, escalate, escalate_reason, retrieved}
                                   body: {"text": "...", "system": "llm"|"simple"|"trivial"}
    GET  /api/metrics          -> contents of reports/eval_results.json, or 404 if not generated yet
    GET  /api/intents          -> the intent taxonomy (for the frontend to display)
"""

import os
import sys
import traceback

from flask import Flask, request, jsonify, send_from_directory

# Make src/ importable regardless of where this is run from
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

import pandas as pd  # noqa: E402

from intents import INTENTS  # noqa: E402
from classify import TrivialClassifier, TfidfLogRegClassifier, LLMClassifier  # noqa: E402
from generate_reply import generate_template_reply, generate_llm_reply, TRIVIAL_REPLY  # noqa: E402
from escalate import trivial_escalate, rule_based_escalate, llm_escalate  # noqa: E402
from retrieve import TfidfRetriever  # noqa: E402
from llm_client import LLMNotConfiguredError, PROVIDER  # noqa: E402
import os as _os
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT_DIR, ".env"))

FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# --- Load data + models once at startup ---------------------------------

DATA_PATH = os.path.join(ROOT_DIR, "data", "processed_threads.csv")
SAMPLE_PATH = os.path.join(ROOT_DIR, "data", "sample", "sample_tweets.csv")
GOLDEN_PATH = os.path.join(ROOT_DIR, "eval", "golden_set.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "intent_clf.joblib")
METRICS_PATH = os.path.join(ROOT_DIR, "reports", "eval_results.json")

_threads_df = None
_retriever = None
_tfidf_clf = None
_llm_clf = None
_trivial_clf = None


def _load_threads():
    global _threads_df, _retriever
    if os.path.exists(DATA_PATH):
        _threads_df = pd.read_csv(DATA_PATH)
    else:
        # fall back to building threads from the bundled sample on the fly
        from data_prep import build_threads
        raw = pd.read_csv(SAMPLE_PATH, dtype=str)
        _threads_df = build_threads(raw, os.getenv("HSA_BRAND", "AppleSupport"))
    _retriever = TfidfRetriever(_threads_df) if len(_threads_df) > 0 else None


def _load_classifiers():
    global _tfidf_clf, _llm_clf, _trivial_clf
    _llm_clf = LLMClassifier()

    if os.path.exists(MODEL_PATH):
        _tfidf_clf = TfidfLogRegClassifier.load(MODEL_PATH)
    elif os.path.exists(GOLDEN_PATH):
        golden = pd.read_csv(GOLDEN_PATH)
        if "true_intent" in golden.columns and golden["true_intent"].notna().any():
            _tfidf_clf = TfidfLogRegClassifier().fit(golden["customer_text"], golden["true_intent"])

    if os.path.exists(GOLDEN_PATH):
        golden = pd.read_csv(GOLDEN_PATH)
        if "true_intent" in golden.columns and golden["true_intent"].notna().any():
            _trivial_clf = TrivialClassifier().fit(golden["customer_text"], golden["true_intent"])
    if _trivial_clf is None:
        _trivial_clf = TrivialClassifier().fit(["placeholder"], ["general_inquiry_other"])


_load_threads()
_load_classifiers()


# --- Routes ---------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    llm_configured = bool(_os.getenv("ANTHROPIC_API_KEY") or _os.getenv("OPENAI_API_KEY"))
    return jsonify(
        {
            "status": "ok",
            "llm_provider": PROVIDER,
            "llm_configured": llm_configured,
            "data_loaded": _threads_df is not None,
            "num_historical_threads": 0 if _threads_df is None else len(_threads_df),
            "simple_model_trained": _tfidf_clf is not None,
        }
    )


@app.route("/api/intents")
def intents():
    return jsonify(
        {name: {"definition": meta["definition"], "default_risk": meta["default_risk"]} for name, meta in INTENTS.items()}
    )


@app.route("/api/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "No eval results yet. Run scripts/run_all.sh first."}), 404
    import json

    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/respond", methods=["POST"])
def respond():
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("text") or "").strip()
    system = body.get("system", "llm")

    if not text:
        return jsonify({"error": "text is required"}), 400
    if system not in ("trivial", "simple", "llm"):
        return jsonify({"error": "system must be one of trivial, simple, llm"}), 400

    try:
        # 1. classify
        if system == "trivial":
            pred = _trivial_clf.predict_one(text)
        elif system == "simple":
            if _tfidf_clf is None:
                return jsonify({"error": "Simple classifier not trained yet. Run: python src/classify.py --train ..."}), 400
            pred = _tfidf_clf.predict_one(text)
        else:
            pred = _llm_clf.predict_one(text)

        intent = pred["intent"]
        confidence = pred.get("confidence", 0.5)

        # 2. retrieve grounding examples (always shown to the user for transparency)
        retrieved = _retriever.retrieve(text, k=3) if _retriever else []
        retrieved_payload = [
            {"customer_text": r.customer_text, "brand_reply_text": r.brand_reply_text, "similarity": round(r.similarity, 3)}
            for r in retrieved
        ]

        # 3. generate reply
        if system == "trivial":
            reply = TRIVIAL_REPLY
        elif system == "simple":
            reply = generate_template_reply(intent)
        else:
            reply = generate_llm_reply(text, intent, retrieved)

        # 4. escalate
        if system == "trivial":
            esc = trivial_escalate(text, intent, confidence)
        elif system == "simple":
            esc = rule_based_escalate(text, intent, confidence)
        else:
            esc = llm_escalate(text, intent, confidence)

        return jsonify(
            {
                "system": system,
                "intent": intent,
                "confidence": confidence,
                "reply": reply,
                "escalate": esc["escalate"],
                "escalate_reason": esc["reason"],
                "retrieved": retrieved_payload,
            }
        )

    except LLMNotConfiguredError as e:
        return jsonify({"error": str(e), "code": "LLM_NOT_CONFIGURED"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {e}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"\nSupport agent demo running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
