#!/usr/bin/env bash
# Runs the entire pipeline end-to-end. This repo ships with real, already-
# processed @AppleSupport data (data/processed_threads.csv, 6,000 threads
# reconstructed from the real Kaggle twcs.csv) and a real hand/rule-labeled
# golden set (eval/golden_set.csv, 200 rows) already committed, so this runs
# against real data by default with no download required.
#
# To rebuild that data yourself from the raw Kaggle file, or to point this at
# a different brand, run:
#   python src/data_prep.py --input data/raw/twcs.csv --brand YOUR_BRAND \
#       --out data/processed_threads.csv --emit-eval-candidates
# then hand-label eval/golden_candidates.csv into eval/golden_set.csv (see
# scripts/label_golden_set.py for the rule-assisted first pass used here),
# and re-run this script.
#
# Set USE_SAMPLE=1 to force the tiny bundled synthetic sample instead (useful
# for a sanity check that the code runs at all with no real data present).
set -e

USE_SAMPLE="${USE_SAMPLE:-0}"

echo "== Step 1: data prep =="
if [ "$USE_SAMPLE" = "1" ]; then
  python src/data_prep.py --sample-fallback --out data/processed_threads.csv
elif [ -f data/processed_threads.csv ] && [ -f eval/golden_set.csv ]; then
  echo "Using committed real data: data/processed_threads.csv + eval/golden_set.csv"
else
  echo "Real processed data not found — falling back to the synthetic sample."
  python src/data_prep.py --sample-fallback --out data/processed_threads.csv
fi

echo "== Step 2: train simple baseline (TF-IDF + Logistic Regression) =="
python src/classify.py --train --data data/processed_threads.csv --golden eval/golden_set.csv --model-out models/intent_clf.joblib

echo "== Step 3: run trivial baseline =="
python src/pipeline.py --system trivial --golden eval/golden_set.csv --data data/processed_threads.csv --out eval/predictions_trivial.csv

echo "== Step 4: run simple baseline =="
python src/pipeline.py --system simple --golden eval/golden_set.csv --data data/processed_threads.csv --out eval/predictions_simple.csv --model models/intent_clf.joblib

echo "== Step 5: run LLM system (requires .env with a valid API key) =="
if python src/pipeline.py --system llm --golden eval/golden_set.csv --data data/processed_threads.csv --out eval/predictions_llm.csv; then
  LLM_OK=1
else
  echo "LLM system skipped (no API key configured in .env). Set one up to include it in the comparison."
  LLM_OK=0
fi

echo "== Step 6: evaluate =="
if [ "$LLM_OK" = "1" ]; then
  python src/eval_harness.py --predictions eval/predictions_trivial.csv eval/predictions_simple.csv eval/predictions_llm.csv --golden eval/golden_set.csv --out reports/eval_results.json
else
  python src/eval_harness.py --predictions eval/predictions_trivial.csv eval/predictions_simple.csv --golden eval/golden_set.csv --out reports/eval_results.json --skip-judge
fi

echo ""
echo "Done. See reports/eval_results.md for the results table."
echo "Next: read reports/REPORT.md and reports/decision_log.md and fill them in with your real findings."
echo ""
echo "To try the live demo UI: python backend/app.py   then open http://localhost:5000"
