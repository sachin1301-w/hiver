# Report: AI Support Agent for @AppleSupport

## 1. Problem framing

**What "good" means for this brand.** @AppleSupport's real volume (Oct–Dec 2017 window, the iOS
11 launch) is dominated by troubleshooting: battery regressions, a keyboard Unicode bug, failed
updates, and connectivity problems, plus a smaller but higher-stakes stream of Apple ID lockouts
and hardware/warranty issues. "Good" here means two different things for two different slices:

- For the 7 low-risk troubleshooting intents (`software_update`, `battery_power`, `device_bug`,
  `connectivity`, `app_service`, `howto_feature`, `feedback_only`): a correct intent label and a
  reply that gives a genuine first-line diagnostic step (not just "DM us"), auto-handled, because
  the cost of a wrong guess here is low (the customer replies "that didn't work" and gets another
  shot) and the cost of needlessly escalating routine volume is real (it's most of the traffic).
- For the 2 high-risk intents (`account_billing`, `hardware_repair`) and the cross-cutting
  categorical triggers (fraud, legal threats, safety incidents, data loss): **recall on
  escalation matters far more than precision.** A false negative here (auto-answering an account
  compromise) is a materially worse failure than a false positive (escalating a routine question).

**What I chose not to build:**
- Multi-turn conversation memory — each message is scored independently, though
  `data_prep.py` does reconstruct the full thread and `n_customer_turns`/`resolved` signals are
  available for a future pass.
- Real account/order lookups — replies ask the customer to DM specifics (device model, serial
  number) rather than pretending to query a real Apple backend that doesn't exist here.
- Sentence-embedding retrieval — `EmbeddingRetriever` is wired in `src/retrieve.py` but not the
  default; TF-IDF was accurate enough at 6,000-thread scale to not justify the extra dependency
  and latency this round.
- Fine-tuning a transformer classifier — out of scope by design; the assignment wants a
  genuinely *simple* baseline distinct from the *advanced* (LLM) system, and a fine-tuned BERT
  would blur that line.

**Sampling note for the golden set.** 6,000 real @AppleSupport customer→reply threads were
reconstructed from the raw Kaggle `twcs.csv` (105,416 raw reply pairs found for this brand,
subsampled to 6,000). A rule-assisted labeling pass (`scripts/label_golden_set.py`) applied
high-precision, per-intent regex rules — written after reading real cluster samples from
`scripts/discover_intents.py`, not guessed — to all 6,000 threads; only unambiguous single-rule
matches were kept (2,031 of 5,957 candidates), and 200 were sampled with a per-intent cap so the
two largest 2017 topics (battery, device bugs) don't dominate the golden set the way they
dominate the raw corpus. Every intent bucket was then spot-checked by hand before acceptance. This
is disclosed, not hidden — see `reports/decision_log.md`, D5, for exactly what "hand-labeled"
means here and what a slower, fully-independent re-read would add.

---

## 2. Results vs. baselines

Measured on the full 200-row golden set (trivial, simple) — see §4 for why the LLM row, when
present, is measured on a different-sized subset and shouldn't be read against these two rows
without that caveat in mind.

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalation Precision | Escalation Recall |
|---|---|---|---|---|---|
| Trivial (majority class, never escalate) | 0.12 | 0.024 | 0.77 | 0.00 | 0.00 |
| Simple (TF-IDF+LogReg, rule escalation) | 0.91* | 0.91* | 0.23 | 0.23 | 1.00 |
| Advanced (LLM few-shot + RAG + LLM escalation) | _pending .env key_ | | | | |

\* **Read this number with the in-sample caveat in §4 before trusting it** — the simple
classifier was trained on a stratified split of this same 200-row golden set, so this is closer
to a training-fit score than a held-out one. `src/classify.py`'s own held-out 25% split (50 rows
the model never saw during training) scores **64% accuracy, 0.60 macro-F1** — that is the honest
apples-to-apples number to compare an LLM system against.

**Reading these numbers:**
- The trivial baseline's 77% escalation "accuracy" is the classic vacuous-baseline trap: it never
  escalates, and 77% of the golden set doesn't need escalation, so silence alone gets 77% "right"
  while its precision/recall of 0.00/0.00 prove it has no actual escalation capability at all.
  This is exactly why accuracy alone is not the metric to trust — see §4.
- The simple baseline's escalation numbers (precision 0.23, recall 1.00) look like "perfectly
  safe, catches everything" until you see *why*: its fixed 0.55 confidence threshold, tuned for
  intuition rather than measurement, is above the maximum softmax probability the model ever
  produces across 9 classes in this run (max observed: 0.53, even on correct predictions) — so it
  escalates **all 200/200** golden-set rows. Perfect recall by escalating everything is not a
  capability; it's the same failure as the trivial baseline's "0% escalate," just at the opposite
  extreme, and it would be caught immediately by anyone timing how long the "auto-handle" queue
  actually stays empty.

---

## 3. Failure analysis (top 5)

Real examples from the simple baseline's 200-row run (18/200 misclassified overall).

1. **Symptom vs. root-cause conflation (the single biggest source of error).**
   Example: *"why am I being charged a Dollar every damn week for iCloud storage I wana
   cancelled"* → predicted `device_bug` (true: `account_billing`), confidence 0.14. The word
   "charged" and general complaint tone pattern-match device symptoms; only "cancelled" and
   "iCloud storage" carry the actual billing signal, and they're outweighed in a bag-of-words
   model by more frequent device-complaint vocabulary. Hypothesis: word-level TF-IDF has no way
   to weight "charged [money]" against "charg[ing] [battery]" — this is exactly the kind of
   lexical ambiguity a few-shot LLM classifier should resolve from context, and worth checking
   directly once the LLM row is filled in.

2. **Battery/hardware/charging three-way confusion.**
   Example: *"my iphone 7 plus isn't charging. What's the deal?"* → predicted `connectivity`
   (true: `battery_power`), confidence 0.13. *"My Watch series 2 was charging overnight... now at
   13%"* → predicted `hardware_repair` (true: `battery_power`). The taxonomy itself makes a
   judgment call here (a charging port failure is `hardware_repair`; a charging *behavior*
   complaint is `battery_power`) that's genuinely hard even for a careful reader without more
   context ("is the cable definitely fine?"). Hypothesis: partially a taxonomy-boundary problem,
   not purely a classifier failure — worth watching whether the LLM system, which can ask itself
   "would swapping the cable fix this?", does meaningfully better or hits the same wall.

3. **`app_service` is the taxonomy's weakest intent** — it's defined by "which technical layer is
   broken" (an Apple *service* vs. the device), which is often not explicit in the customer's own
   words. Example: *"since #ios11 upgrade my screen has frozen countless times, landscape mode
   switches in..."* → predicted `app_service` (true: `device_bug`) — plausible either way. Of 24
   `app_service` golden-set rows, several of its misclassifications go to `device_bug` and
   vice versa. Hypothesis: these two intents may deserve a shared reply template rather than
   forcing a hard boundary, if the LLM system doesn't show it can reliably tell them apart either.

4. **Uniformly low confidence, not calibration noise.** All 200 confidence scores in the simple
   system's run fall in a narrow 0.13–0.53 band (mean 0.25). This isn't scattered noise — it's a
   systematic property of a 9-way softmax where even confidently-correct predictions rarely clear
   0.5. Hypothesis: a threshold *and* margin-to-runner-up check (not just a raw top-probability
   cutoff) would separate "the model is right but the distribution is naturally spread over 9
   classes" from "the model is actually unsure" — the simple baseline deliberately doesn't do
   this (see decision_log D8), which is the point: it's the naive thing to beat.

5. **`feedback_only` vs. an embedded real request.** Example candidates during golden-set
   spot-checking included messages that open with thanks/praise but end with an unanswered
   question ("thanks for the last fix, but is there one coming for X too?") — the taxonomy
   definition explicitly excludes these from `feedback_only`, but a bag-of-words model can still
   be pulled toward the intent whose vocabulary (thanks, appreciate) dominates the message length.
   Hypothesis: this is the single case where an LLM's ability to find "the actual ask" buried in
   a longer message should show the clearest measurable advantage over TF-IDF.

---

## 4. What is misleading about my headline number?

*(Mandatory section.)*

- **The simple system's 91% intent accuracy is inflated by evaluating on its own training data.**
  `src/classify.py::train_and_eval` correctly holds out 25% for its own internal report (64%
  accuracy there), but `pipeline.py --system simple` re-fits/reuses the model against the *full*
  200-row golden set for the cross-system comparison table, so 150 of those 200 rows are ones the
  model was trained on. **64% (the honest held-out number), not 91%, is the real simple-baseline
  number to beat.**
- **The golden set's escalation ground truth is generated by the same policy logic the simple
  system uses at runtime** (categorical rules + intent risk tier — see decision_log D6). This
  structurally advantages any system whose escalation *policy* matches that logic almost by
  definition; the honest comparison across systems is on classifier *confidence calibration*
  feeding into that shared policy, not on the top-line escalation-accuracy number in isolation.
- **The golden set (200 rows, rule-assisted first-pass labeling) is not a substitute for reading
  every row by hand.** Rows where two intent rules fired ambiguously, or none fired, were dropped
  — meaning the golden set is systematically *easier* than the raw traffic distribution (which
  includes plenty of genuinely ambiguous, multi-issue, or borderline messages the rules
  deliberately excluded). Real-world accuracy on unfiltered incoming messages will be lower than
  any number in this report.
- **The corpus is a single 3-month window (Oct–Dec 2017, iOS 11 launch).** Two topics unique to
  that release (the "I" keyboard bug, iOS 11 battery regression) are large enough to dominate
  several clusters. A taxonomy and classifier tuned on this window may not generalize to a
  different iOS release with a different bug profile — this is a real risk, not a hypothetical
  one, given how concentrated the corpus is.
- **Twitter-support data only captures issues that got a public reply at all.** DM-only
  resolutions, unresolved complaints, and anything the brand chose not to answer publicly are
  systematically missing — the golden set can only be as representative as the platform's own
  selection bias allows.
- **LLM-judge scores (once populated) may reward fluency over correctness** — a confident,
  well-punctuated reply can score well on tone/relevance while still being ungrounded; this is
  why groundedness is scored as its own dimension rather than folded into one overall number, but
  the judge is itself an LLM and can share blind spots with the generation model. See
  `judge_agreement.py` output for the actual measured human-agreement number once available —
  don't trust the judge's score without it.

---

## 5. What I'd do next with one more week

- Fully re-read all 200 golden-set rows by hand (not rule-assisted) and measure how much the
  labels actually change — that delta is the real answer to "how good is the rule-assisted
  approach," which this report can currently only argue for, not measure.
- Re-run `classify.py` with proper train/held-out separation baked into `pipeline.py` itself
  (rather than only in the standalone training script), so the cross-system comparison table
  can't accidentally repeat the in-sample-accuracy mistake documented in §4.
- Recalibrate the escalation confidence threshold empirically (a margin-to-runner-up check, or a
  small labeled "should this have escalated" set used to pick a threshold by cost-weighted
  F-beta) instead of a fixed, untuned 0.55.
- Add multi-turn context: `data_prep.py` already reconstructs `n_customer_turns` and a resolved/
  unresolved signal from the customer's follow-up turns — neither is used by the classifier yet.
- Swap TF-IDF retrieval for sentence-embedding retrieval and measure the delta on the LLM
  system's groundedness score specifically (the `app_service`/`device_bug` boundary confusion in
  §3 is exactly the kind of paraphrase gap embeddings should help with).
- Get a second independent human labeler on a subset of the golden set to measure inter-annotator
  agreement — right now there is effectively one labeling pass (rule-assisted + one spot-check),
  which is a real limitation for a golden set meant to anchor every other number in this report.
