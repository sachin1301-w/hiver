"""
Escalation decision: should this message be auto-handled or handed to a
human? Every decision must come with a stated reason (assignment requirement).

Three tiers, matching the three systems:
1. trivial: never escalates (auto-handles everything) — this is exactly why
   it's a bad system, and the eval harness should make that visible.
2. simple: keyword + confidence rule. Fast, explainable, no LLM call.
3. llm: LLM reasons over the message + intent + confidence and produces a
   decision + one-sentence justification.
"""

import re

from intents import INTENTS
from llm_client import complete_json

# High-precision categorical triggers. Each maps to a *specific* reason so an
# escalation always tells a human queue why, not just that it fired.
# Confidence can never override these — see reports/decision_log.md, D7.
CATEGORICAL_ESCALATION_RULES = [
    ("legal_or_regulatory",
     re.compile(r"\b(lawyer|attorney|sue|suing|lawsuit|legal action|small claims|"
                r"class action|trading standards|consumer rights)\b", re.I),
     "Customer raised legal action — must not be answered by an automated reply."),
    ("safety_incident",
     re.compile(r"\b(caught fire|exploded|exploding|burn(ed|t|s)? me|smoking|injur(y|ed)|"
                r"battery (swell|swollen|bulg))\b", re.I),
     "Possible physical-safety incident — needs immediate human handling."),
    ("security_compromise",
     re.compile(r"\b(hacked|compromised|unauthorized|fraud(ulent)?|stolen|phish|"
                r"someone (is )?(using|accessed|got into) my)\b", re.I),
     "Possible account compromise or fraud — requires identity verification."),
    ("data_loss",
     re.compile(r"\b(lost|deleted|gone|wiped|erased)\b.{0,25}\b(photo|contact|note|"
                r"message|data|backup|everything)\b", re.I),
     "Reported data loss — irreversible if mishandled, needs a specialist."),
    ("repeat_contact",
     re.compile(r"\b(third|3rd|fourth|4th|fifth|5th) time (i'?ve |i have )?"
                r"(contact|tried|asked|call|email|dm)|no ?one (has )?(replied|responded|helped)|"
                r"weeks? (with )?no (response|reply|answer|fix)\b", re.I),
     "Customer already tried and failed to get help — another templated reply will make it worse."),
    ("severe_dissatisfaction",
     re.compile(r"\b(never buying|switching to android|done with apple|scam(mers)?|"
                r"rip ?off|disgraceful|cancel(ling)? my)\b", re.I),
     "Churn or accusation-level dissatisfaction — a human should own the recovery."),
]

# Kept for the simple/legacy checks below; a rule above already gives the
# specific reason when one of these fires.
HIGH_RISK_KEYWORDS = [
    "hacked", "fraud", "unauthorized", "lawyer", "legal", "sue", "police",
    "chargeback", "dispute", "scam", "stolen",
]

LOW_CONFIDENCE_THRESHOLD = 0.55


def trivial_escalate(text: str, intent: str, confidence: float) -> dict:
    return {"escalate": False, "reason": "Trivial baseline never escalates."}


def rule_based_escalate(text: str, intent: str, confidence: float) -> dict:
    default_risk = INTENTS.get(intent, {}).get("default_risk", "low")

    for _name, pattern, reason in CATEGORICAL_ESCALATION_RULES:
        if pattern.search(text or ""):
            return {"escalate": True, "reason": reason}

    if default_risk == "high":
        return {
            "escalate": True,
            "reason": f"Intent '{intent}' is tagged high-risk by default (requires identity verification or billing review).",
        }

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return {
            "escalate": True,
            "reason": f"Classifier confidence ({confidence:.2f}) is below threshold ({LOW_CONFIDENCE_THRESHOLD}).",
        }

    return {"escalate": False, "reason": f"Intent '{intent}' is low/medium risk and confidence is sufficient."}


def llm_escalate(text: str, intent: str, confidence: float) -> dict:
    system = (
        "You decide whether a customer support message to @AppleSupport should be auto-handled "
        "by an AI agent or escalated to a human agent. Escalate anything involving: Apple ID / "
        "account security or fraud, legal threats, billing disputes or refunds, physical-safety "
        "incidents (overheating, fire, injury), data loss, hardware repair/warranty decisions "
        "(these cost money and need serial-number lookups), a customer who has already tried and "
        "failed to get help, or genuine ambiguity about what the customer needs. Auto-handle "
        "routine, low-risk troubleshooting the brand has a clear standard process for (software "
        "update failures, battery/performance symptoms, connectivity issues, how-to questions).\n"
        "Return JSON: {\"escalate\": true|false, \"reason\": <one short sentence, specific to this message>}"
    )
    user = f"Message: {text}\nClassified intent: {intent}\nClassifier confidence: {confidence:.2f}"
    result = complete_json(system, user)
    result["escalate"] = bool(result.get("escalate", True))
    result.setdefault("reason", "No reason provided by model.")
    return result
