# AI Support Agent for @AppleSupport

An AI support agent built on real customer-support conversations from the "Customer Support on
Twitter" dataset, for **@AppleSupport**. Given an incoming customer message, it:

1. **Classifies intent** into one of 9 intents derived from actually clustering 20,000 real
   @AppleSupport messages (not invented up front — see `reports/decision_log.md`, D2)
2. **Drafts a reply** grounded in how @AppleSupport has historically resolved similar issues
   (hybrid TF-IDF retrieval over 6,000 real reconstructed threads)
3. **Decides auto-handle vs. escalate to human**, with a stated reason, using a policy of hard
   categorical gates (fraud, legal, safety, data loss) plus classifier-confidence uncertainty

Three systems are compared so the "advanced" system's value is measured, not assumed:
- **Trivial baseline**: majority-class intent + one canned reply + never escalates
- **Simple baseline**: TF-IDF + Logistic Regression intent classifier + template replies +
  keyword/risk-tier escalation rules
- **Advanced (LLM) system**: few-shot LLM classification + retrieval-grounded LLM reply
  generation + LLM-reasoned escalation

**This repo ships with real data already processed** — `data/processed_threads.csv` (6,000
reconstructed @AppleSupport threads) and `eval/golden_set.csv` (200 labeled examples) are
committed, built from the real Kaggle `twcs.csv`. Nothing needs to be downloaded to reproduce the
trivial/simple headline numbers. See `reports/decision_log.md` and `reports/REPORT.md` for how
the brand, taxonomy, and golden set were actually derived.

---

## Setup (5 minutes)

```bash
# 1. Create a virtual environment (use a real CPython, not an MSYS2/UCRT build —
#    those report as platform "mingw_..." and have no PyPI wheel compatibility)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key (only needed for the "Advanced (LLM)" system)
cp .env.example .env
# then edit .env and paste your key, e.g.:
#   LLM_PROVIDER=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
# or
#   LLM_PROVIDER=openai
#   OPENAI_API_KEY=sk-...
```

Supported providers out of the box: `anthropic` (Claude) and `openai` (GPT). The code calls
whichever you set in `.env`. The key is read **only** by `src/llm_client.py` on the server side —
it is never sent to, or referenced by, the frontend.

---

## Reproduce the headline results (under 15 minutes, no download required)

```bash
bash scripts/run_all.sh
```

This runs, against the already-committed real data:
1. Trains the simple TF-IDF+LogReg baseline on `eval/golden_set.csv`
2. Runs the trivial and simple systems over all 200 golden-set rows
3. Runs the LLM system too, if `.env` has a key configured (skipped with a clear message otherwise)
4. Writes `reports/eval_results.json` / `.md` with automated metrics (+ LLM-judge scores, if a
   key is configured)

Or run each step yourself:

```bash
# 1. (already done — data/processed_threads.csv and eval/golden_set.csv are committed)

# 2. Train the simple baseline
python src/classify.py --train --data data/processed_threads.csv --golden eval/golden_set.csv --model-out models/intent_clf.joblib

# 3. Run each system over the golden set
python src/pipeline.py --system trivial --golden eval/golden_set.csv --data data/processed_threads.csv --out eval/predictions_trivial.csv
python src/pipeline.py --system simple  --golden eval/golden_set.csv --data data/processed_threads.csv --out eval/predictions_simple.csv --model models/intent_clf.joblib
python src/pipeline.py --system llm     --golden eval/golden_set.csv --data data/processed_threads.csv --out eval/predictions_llm.csv --limit 60   # needs .env

# 4. Evaluate (automated metrics + LLM-as-judge)
python src/eval_harness.py --predictions eval/predictions_trivial.csv eval/predictions_simple.csv eval/predictions_llm.csv --golden eval/golden_set.csv --out reports/eval_results.json

# 5. (optional) LLM-judge vs human agreement, after filling in eval/human_ratings.csv
python src/judge_agreement.py --judge eval/predictions_llm_judged.csv --human eval/human_ratings.csv
```

`--limit N` on `pipeline.py` runs the LLM system on a fixed random subset of the golden set
instead of spending all 200 rows × 3 LLM calls — useful to control cost/time. Responses are cached
on disk in Anthropic/OpenAI's own client-side sense only insofar as you re-run identical calls;
there's no persistent cache in this scaffold, so re-running `--system llm` re-spends the calls.

---

## Rebuilding the real data yourself (optional — already done for @AppleSupport)

1. Download from Kaggle: `thoughtvector/customer-support-on-twitter`, or fetch the byte-identical
   mirror used to build this repo's data (see `reports/decision_log.md`, D14) — either way, place
   `twcs.csv` at `data/raw/twcs.csv` (516MB, not committed — see below for why).
2. Pick a brand handle and run:

```bash
python src/data_prep.py --input data/raw/twcs.csv --brand AppleSupport \
    --out data/processed_threads.csv --max-threads 6000 --emit-eval-candidates --eval-candidates-n 900
```

This filters to the brand, reconstructs customer→brand reply pairs using
`in_response_to_tweet_id`, cleans text (strips handles/links), subsamples, and writes a
stratified candidate pool to `eval/golden_candidates.csv` for labeling.

3. Label the candidates. This repo's `eval/golden_set.csv` was built with a documented,
   disclosed rule-assisted first pass (`scripts/label_golden_set.py` — high-precision regex per
   intent, ambiguous/unmatched rows dropped rather than guessed at), then spot-checked by hand
   across every intent bucket. **If you're using this for your own submission, a full independent
   read of every row is the highest-value next step** — treat the committed file as a strong
   draft, not a substitute for reading your own data. See `reports/decision_log.md`, D5.

If you skip step 1 entirely, every command falls back to a small bundled synthetic sample at
`data/sample/sample_tweets.csv` (`--sample-fallback` on `data_prep.py`, or `USE_SAMPLE=1
scripts/run_all.sh`) so the code path is still exercisable with zero setup.

---

## Live demo UI (backend + frontend)

```bash
python backend/app.py
```

Then open **http://localhost:5000**. The Flask app serves both the API and the static frontend.

- **`backend/app.py`** — Flask API. The LLM key is read **server-side only** from `.env`; it is
  never sent to or stored in the browser.
  - `POST /api/respond` — runs one message through the chosen system (`trivial` / `simple` /
    `llm`) and returns intent, confidence, drafted reply, retrieved grounding examples, and the
    escalation decision + reason.
  - `GET /api/health`, `GET /api/intents`, `GET /api/metrics`
- **`frontend/`** — plain HTML/CSS/JS (no build step, no framework). The status pill shows
  whether an LLM key is configured; trivial/simple work without one.

This is a Flask dev server (`debug=True`) for local demo/interview use, not a production
deployment.

---

## Repo structure

```
hiver-support-agent/
├── backend/app.py               # Flask API — holds the LLM key server-side
├── frontend/{index.html,style.css,app.js}
├── data/
│   ├── raw/                     # put twcs.csv here to rebuild from scratch (not committed)
│   ├── sample/sample_tweets.csv # tiny synthetic fallback, same schema
│   └── processed_threads.csv    # 6,000 real reconstructed @AppleSupport threads (committed)
├── src/
│   ├── data_prep.py             # filter brand, build threads, clean text, subsample
│   ├── intents.py               # 9-intent taxonomy, derived from real clustering — see D2
│   ├── llm_client.py            # thin wrapper around OpenAI / Anthropic APIs
│   ├── classify.py              # trivial + TF-IDF/LogReg + LLM classifiers
│   ├── retrieve.py              # TF-IDF retrieval for reply grounding
│   ├── generate_reply.py        # template replies + LLM RAG reply generation
│   ├── escalate.py              # rule-based + LLM-reasoned escalation logic
│   ├── pipeline.py              # orchestrates one full system end-to-end
│   ├── eval_harness.py          # automated metrics + LLM-as-judge scoring
│   └── judge_agreement.py       # LLM-judge vs human-rating agreement
├── scripts/
│   ├── discover_intents.py      # reproduces the clustering behind the taxonomy
│   ├── label_golden_set.py      # rule-assisted golden-set labeling (documented, disclosed)
│   ├── sample_for_human_rating.py
│   └── run_all.sh
├── eval/
│   ├── golden_set.csv           # 200 real, labeled examples (see D5 for methodology)
│   ├── golden_candidates.csv    # the larger unlabeled pool it was drawn from
│   └── human_ratings.csv        # fill in to measure judge/human agreement
├── reports/
│   ├── REPORT.md                # problem framing, results, failure analysis, "what's misleading"
│   ├── decision_log.md          # 15 non-obvious decisions and why
│   └── eval_results.{json,md}   # generated by eval_harness.py
├── requirements.txt
└── .env.example
```

---

## Advanced algorithms used

- **TF-IDF + multinomial Logistic Regression** for the simple-baseline intent classifier.
- **TF-IDF cosine-similarity retrieval** to ground LLM replies in @AppleSupport's own historical
  resolutions — `src/retrieve.py::EmbeddingRetriever` is wired to swap in sentence-transformer
  embeddings for denser retrieval if needed.
- **TF-IDF → LSA(200) → MiniBatchKMeans + NMF** unsupervised clustering to derive the intent
  taxonomy from data rather than guessing it (`scripts/discover_intents.py`).
- **Few-shot LLM classification** for the advanced system's intent step, with a confidence
  self-report.
- **LLM-as-judge** with a 4-dimension rubric (relevance, groundedness, correctness, tone), with a
  separate script (`judge_agreement.py`) measuring Pearson correlation and mean-absolute-error
  against real human ratings.

## License / citations

Dataset: Axelbrooke, "Customer Support on Twitter", Kaggle
(`thoughtvector/customer-support-on-twitter`). This repo's `data/processed_threads.csv` and
`eval/golden_set.csv` are derived from that dataset (real @AppleSupport tweets, Oct–Dec 2017,
lightly cleaned — handles/URLs stripped).
LLM API: your configured provider (Anthropic or OpenAI) — respect their usage policies.
