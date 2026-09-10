"""
Builds eval/golden_set.csv from eval/golden_candidates.csv.

METHODOLOGY (documented here + reports/decision_log.md, D5 — read before you
trust this file):

1. Each candidate's `customer_text` is matched against a set of HIGH-PRECISION,
   hand-written regex rules, one block per intent in src/intents.py. These
   rules were written by reading the actual cluster samples from
   scripts/discover_intents.py, not guessed at — each pattern targets wording
   that is genuinely diagnostic for that intent in this corpus.
2. A rule fires only on unambiguous wording. Rows where no rule fires, or
   where two different intents' rules both fire, are DROPPED rather than
   guessed at — a golden set with forced labels on ambiguous rows is worse
   than a smaller golden set with confident ones.
3. `true_escalate` / `escalate_reason` come from the same categorical policy
   rules used at runtime in src/escalate.py, PLUS the intent's default risk
   tier. This is deliberately the same policy the "simple" baseline uses —
   see reports/REPORT.md, "What is misleading about my headline number", for
   why that makes the simple baseline's escalation score look better than it
   should and how the LLM system's number should be read against it instead.

THIS IS RULE-ASSISTED LABELING, NOT BLIND HUMAN LABELING. Every row was
additionally read by a human during a spot-check pass (see the "reviewed"
note in decision_log.md D5) but the rules did the first pass at this volume.
If you are using this repo for your own submission, the assignment expects
YOU to have read every row yourself — treat this script's output as a strong
starting draft to review and correct, not a final answer to submit unread.

Usage:
    python scripts/label_golden_set.py --candidates eval/golden_candidates.csv --out eval/golden_set.csv --n 200
"""

import argparse
import re

import pandas as pd

RULES = [
    ("account_billing", re.compile(
        r"\bapple\s?id\b|\bicloud (storage|account|plan|lock)|\bsign(ed)? ?in\b|"
        r"\blog(ged)? ?in\b|\bpassword\b|\b2fa\b|two.?factor|"
        r"\b(refund|charged?|charging me|billing|invoice|receipt|subscription)\b|"
        r"reportaproblem|\baccount (is )?(locked|disabled|hacked|compromised)\b|"
        r"\bgift card\b|\bapple pay\b|\bitunes account\b", re.I)),
    ("hardware_repair", re.compile(
        r"\bcracked?\b|\bwater damage\b|\bshattered\b|\bgenius bar\b|"
        r"\b(repair|replacement|replace(d)? (my|the) (phone|screen|device))\b|"
        r"\bwarrant(y|ies)\b|\bapple ?care\b|\bserial number\b|"
        r"\b(dead|broken|faulty|defective) (pixel|button|port|charger|cable|screen|device|phone)\b|"
        r"\bservice (centre|center|appointment)\b|\bout of the box\b", re.I)),
    ("battery_power", re.compile(
        r"\bbatter(y|ies)\b|\bcharg(e|es|ing|er)\b|\boverheat|\btoo hot\b|"
        r"\bdrain(s|ing|ed)?\b|\b(shuts?|shutting|turns?) (down|off) (randomly|by itself|on its own)\b|"
        r"\bpower bank\b|\b\d+% battery\b", re.I)),
    ("connectivity", re.compile(
        r"\bwi[- ]?fi\b|\bbluetooth\b|\bairdrop\b|\bhotspot\b|"
        r"\bcellular\b|\b(no|lost) (signal|service)\b|\bairpods?\b.*\b(connect|pair)|"
        r"\b(won'?t|can'?t|cannot|not) (connect|pair)\b|\bnetwork settings\b|"
        r"\bdrops? (the )?(connection|wifi|signal)\b", re.I)),
    ("software_update", re.compile(
        r"\b(can'?t|cannot|won'?t|unable to|trying to|tried to) (update|install|download) (to |the )?(ios|the update|software)?|"
        r"\bupdate (is )?(stuck|failed|failing|won'?t install|not installing)\b|"
        r"\bnot enough storage\b|\bdowngrade\b|\brecovery mode\b|"
        r"\bstuck (on|at) the apple logo\b|\bverifying update\b|\brestore (my )?(phone|iphone|ipad)\b", re.I)),
    ("app_service", re.compile(
        r"\bapp ?store\b|\bitunes\b(?!.{0,15}account)|\bapple music\b|\bimessage\b|"
        r"\bfacetime\b|\bsiri\b|\bapple ?tv\b|\bicloud (sync|backup|photos|drive)\b|"
        r"\b(netflix|spotify|instagram|snapchat|whatsapp|youtube|facebook)\b|"
        r"\bmail app\b|\bsystem status\b|\bpodcast", re.I)),
    ("device_bug", re.compile(
        r"question mark|\bbox(es)? (instead|where)\b|"
        r"\bghost touch\b|\b(freez(e|es|ing)|frozen)\b|\b(lag|lagging|laggy|slow)\b|"
        r"\bcrash(es|ing|ed)?\b|\bglitch\b|\bkeyboard\b|\bautocorrect\b|"
        r"\b(black|white|blank) screen\b|\bunresponsive\b|\brandom(ly)? restart|"
        r"\bbug\b|\bstutter", re.I)),
    ("howto_feature", re.compile(
        r"^\s*how (do|can|would) (i|you|we)\b|\bhow do i\b|\bis there a way to\b|"
        r"\bwhen (will|is|are) (you|apple|it|this)\b.*\b(fix|release|come|available|out)\b|"
        r"\b(please|pls|plz) (add|bring back|make|allow)\b|\bfeature request\b|"
        r"\bany (plans|way) to\b|\bwill (you|apple) (ever|be)\b", re.I)),
    ("feedback_only", re.compile(
        r"^\s*(thank(s| you)|cheers|much appreciated|appreciate it)\b[^?]*$|"
        r"\b(worst|best) (customer service|company|phone|support)\b[^?]*$", re.I)),
]

ESCALATION_RULES = [
    ("legal_or_regulatory", re.compile(
        r"\b(lawyer|attorney|sue|suing|lawsuit|legal action|small claims|class action)\b", re.I),
     "Customer raised legal action — must not be answered by an automated reply."),
    ("safety_incident", re.compile(
        r"\b(caught fire|exploded|exploding|burn(ed|t|s)? me|smoking|injur(y|ed)|battery (swell|swollen|bulg))\b", re.I),
     "Possible physical-safety incident — needs immediate human handling."),
    ("security_compromise", re.compile(
        r"\b(hacked|compromised|unauthorized|fraud(ulent)?|stolen|phish|someone (is )?(using|accessed|got into) my)\b", re.I),
     "Possible account compromise or fraud — requires identity verification."),
    ("data_loss", re.compile(
        r"\b(lost|deleted|gone|wiped|erased)\b.{0,25}\b(photo|contact|note|message|data|backup|everything)\b", re.I),
     "Reported data loss — irreversible if mishandled, needs a specialist."),
    ("repeat_contact", re.compile(
        r"\b(third|3rd|fourth|4th|fifth|5th) time (i'?ve |i have )?(contact|tried|asked|call|email|dm)|"
        r"no ?one (has )?(replied|responded|helped)|weeks? (with )?no (response|reply|answer|fix)\b", re.I),
     "Customer already tried and failed to get help — another templated reply will make it worse."),
    ("severe_dissatisfaction", re.compile(
        r"\b(never buying|switching to android|done with apple|scam(mers)?|rip ?off|disgraceful|cancel(ling)? my)\b", re.I),
     "Churn or accusation-level dissatisfaction — a human should own the recovery."),
]

HIGH_RISK_INTENTS = {"account_billing", "hardware_repair"}


def label_intent(text: str) -> str | None:
    hits = [name for name, pat in RULES if pat.search(text or "")]
    if len(hits) == 1:
        return hits[0]
    return None  # none fired, or ambiguous (2+) — drop rather than guess


def label_escalation(text: str, intent: str) -> tuple[bool, str]:
    for _name, pat, reason in ESCALATION_RULES:
        if pat.search(text or ""):
            return True, reason
    if intent in HIGH_RISK_INTENTS:
        return True, f"Intent '{intent}' is tagged high-risk by default (requires identity verification or billing/warranty review)."
    return False, f"Intent '{intent}' is low/medium risk with a clear standard troubleshooting process."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="eval/golden_candidates.csv")
    ap.add_argument("--out", default="eval/golden_set.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.candidates)
    if "reference_reply" not in df.columns and "brand_reply_text" in df.columns:
        # sourcing straight from data/processed_threads.csv (much larger pool
        # than the stratified golden_candidates.csv) rather than duplicating it
        df["reference_reply"] = df["brand_reply_text"]
    df = df.drop_duplicates(subset=["customer_text"])
    df["true_intent"] = df["customer_text"].apply(label_intent)

    labeled = df.dropna(subset=["true_intent"]).copy()
    print(f"[label_golden_set] {len(labeled)}/{len(df)} candidates got an unambiguous rule label")

    esc = labeled["customer_text"].combine(labeled["true_intent"], label_escalation)
    labeled["true_escalate"] = esc.apply(lambda t: t[0])
    labeled["escalate_reason"] = esc.apply(lambda t: t[1])

    # Balance across intents a little instead of taking whatever order they
    # happen to appear in — otherwise the two largest 2017 topics (battery,
    # device_bug) would dominate the golden set the way they dominate the raw
    # corpus, and rarer-but-important intents (account_billing) would barely
    # be represented in the eval.
    labeled = labeled.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    per_intent_cap = max(20, args.n // labeled["true_intent"].nunique() + 5)
    parts = [g.head(per_intent_cap) for _, g in labeled.groupby("true_intent")]
    capped = pd.concat(parts, ignore_index=True)
    if len(capped) > args.n:
        capped = capped.sample(n=args.n, random_state=args.seed)
    capped = capped.sort_values("true_intent").reset_index(drop=True)

    out_cols = ["customer_text", "true_intent", "reference_reply", "true_escalate", "escalate_reason"]
    capped[out_cols].to_csv(args.out, index=False)
    print(f"[label_golden_set] Wrote {len(capped)} labeled rows to {args.out}")
    print(capped["true_intent"].value_counts().to_string())
    print(f"escalate rate: {capped['true_escalate'].mean():.1%}")


if __name__ == "__main__":
    main()
