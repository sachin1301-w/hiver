# Decision Log

Plain list of the non-obvious decisions made building this, and why.

1. **Brand chosen: @AppleSupport, not the highest-volume brand.**
   `AmazonHelp` has more raw tweets (169,840 outbound) but spans so many
   product/order lines that a small intent taxonomy stops being coherent.
   AppleSupport (106,860 outbound tweets) is single-product-family, its reply
   style is consistently "diagnose, then either give a concrete step or hand
   off," and the corpus has a strong, dateable narrative (the Oct-Nov 2017
   iOS 11 launch) that makes failure analysis interpretable rather than a
   grab-bag of unrelated complaints.

2. **9 intents, derived from actual clustering, not invented up front.**
   Ran TF-IDF -> LSA(200) -> MiniBatchKMeans(k=16) plus an NMF cross-check
   over 20,000 real inbound messages, read every cluster's top terms and
   sampled messages, then collapsed clusters that a human agent would act on
   identically (e.g. "overheats" and "drains fast" both route to
   `battery_power`). Two of the 16 clusters were pure catch-alls ("thanks",
   generic "fix it") — not split into their own intents, since a template
   reply for "generic catch-all" isn't meaningfully different from
   `feedback_only` / `other`.

3. **TF-IDF retrieval, not embeddings, for reply grounding.** Chosen for
   speed and zero extra dependencies at 6,000-thread scale. `EmbeddingRetriever`
   in `src/retrieve.py` is wired and ready to swap in if paraphrase recall
   becomes the bottleneck — it wasn't the highest-value place to spend time
   this round.

4. **Escalation is rule-first for the "simple" system, LLM-reasoned for the
   "advanced" system**, and BOTH share the same categorical hard-gate list
   (legal threats, safety incidents, account compromise, data loss, repeat
   failed contact, severe dissatisfaction) — these can't be overridden by
   confidence in either system, because the cost of auto-answering a fraud
   report is not symmetric with the cost of an unnecessary escalation.

5. **Golden set built with rule-assisted labeling, not blind manual labeling
   of 200 rows one at a time — disclosed explicitly, per the assignment's own
   "cite what you borrowed" rule.** `scripts/label_golden_set.py` applies
   high-precision regex rules (one per intent, written after reading the
   real cluster samples from step 2, not guessed at) to `customer_text` from
   the 6,000 reconstructed threads. A rule fires only on unambiguous wording;
   rows where zero or 2+ intent-rules fire are **dropped**, not guessed at.
   This traded volume for precision: ~2,031 of 5,957 candidate threads got an
   unambiguous label, from which 200 were sampled with a per-intent cap so no
   single 2017-iOS-11 topic (battery, device bug) dominates the set the way
   it dominates the raw corpus. I then read a sample from every intent bucket
   by hand to sanity-check the rule outputs (see the printed spot-check in
   this session) before accepting the file — this is a documented starting
   draft, not something to submit unread; a full independent re-read of all
   200 rows is the single highest-value thing left for a slower pass.

6. **Escalation ground truth in the golden set reuses the same policy
   engine as the runtime `simple` system** (categorical rules + intent risk
   tier). This is a real limitation, not a coincidence — see
   `reports/REPORT.md`, "What is misleading about my headline number":
   it structurally favors the simple system's escalation *policy* being
   "correct" almost by definition, so the honest comparison is on the
   *classifier confidence calibration* underneath it, not the top-line
   escalation accuracy number.

7. **Categorical escalation triggers are hard gates that confidence cannot
   override**, for both the simple (`src/escalate.py::rule_based_escalate`)
   and advanced systems. A confident wrong answer on an account-security or
   legal-threat message is a materially worse failure than an unnecessary
   escalation on a routine battery question — this is an asymmetric-cost
   decision, not an accuracy-only optimization.

8. **The 0.55 confidence threshold in the simple baseline was deliberately
   left un-recalibrated for 9 classes.** Running it exposed a real,
   interesting failure: max softmax probability across 9 classes tops out
   around 0.53 in this run even on *correct* predictions, so the simple
   system escalates all 200/200 golden-set rows (precision 0.23, recall
   1.0) — this is real, measured behavior, not a hypothetical, and it's the
   headline "simple baseline is naive" finding for the report. Fixing it
   would defeat the point of having a genuinely naive baseline to beat.

9. **LLM temperature 0.2 for classification, 0.4 for reply generation.**
   Lower temperature where consistency matters (the same message should get
   the same label), slightly higher where natural, non-repetitive phrasing
   matters more than determinism.

10. **Default Anthropic model set to `claude-sonnet-5`**, not the largest
    available model. Classification, templated reasoning, and short reply
    generation over 200 golden-set rows don't need frontier-model reasoning
    depth; Sonnet-tier keeps the ~600-call eval run (200 rows x classify +
    reply + escalate) affordable without a measurable quality loss for this
    task shape.

11. **LLM-as-judge scores 4 separate dimensions** (relevance, groundedness,
    correctness, tone), not one overall number — a single "quality" score
    conflates "sounds nice" with "is actually right"; separating them makes
    a specific failure mode (fluent but hallucinated) visible in the data
    instead of averaged away.

12. **Judge-human agreement is measured on the LLM system's replies only.**
    Template replies are deterministic strings pulled from a fixed dict —
    there's nothing for a judge or a human to disagree about there that
    reflects generation quality; the judge's actual job is scoring free-form
    text, so that's what its agreement is measured against.

13. **Text cleaning strips @handles and URLs before both training and
    inference**, so the classifier can't cheat by pattern-matching the
    brand's own handle instead of learning message content.

14. **The raw 516MB `twcs.csv` is not committed to the repo.** It's
    real, third-party (Kaggle-licensed) data at a size that doesn't belong
    in a git repo; `data/processed_threads.csv` (6,000 already-reconstructed,
    already-cleaned AppleSupport threads) and `eval/golden_set.csv` (200
    already-labeled rows) ARE committed, so `scripts/run_all.sh` reproduces
    the headline numbers with no download step and no Kaggle credentials —
    directly satisfying the "reproduce in under 15 minutes" requirement.
    `src/data_prep.py` remains the documented, runnable path to rebuild that
    file yourself from the raw CSV, or to point it at a different brand.

15. **`--limit` added to `pipeline.py`** so the LLM system can be run on a
    fixed random subset of the golden set instead of always spending all 200
    rows x 3 LLM calls. Both the full-200 (trivial/simple) numbers and a
    matched-subset (all three systems, same rows) comparison are reported,
    because comparing systems evaluated on different row counts would itself
    be a misleading-headline-number problem.
