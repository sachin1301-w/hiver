"""
Intent taxonomy for the support agent — brand: @AppleSupport.

HOW THESE 9 INTENTS WERE CHOSEN
--------------------------------
I ran TF-IDF -> LSA(200) -> MiniBatchKMeans(k=16) plus an NMF cross-check over
20,000 real @AppleSupport inbound tweets (see reports/decision_log.md, D3, and
scripts/discover_intents.py which reproduces this). The corpus is dominated by
the Oct-Dec 2017 iOS 11 launch window, so the clusters are very concrete:
battery drain after the update, a Unicode bug where typing "I" renders as a
boxed question mark, failed/stuck installs, and repeated mentions of Apple ID
lockouts, Genius Bar / repairs, and reportaproblem.apple.com.

I collapsed 16 clusters into 9 intents by merging any two clusters that a
human agent would act on identically (e.g. "battery drains fast" and "phone
overheats" both go to battery_power because the triage step is the same).
Two of the 16 clusters were generic catch-alls TF-IDF couldn't split further
("thanks" / "did you get my DM") — those aren't a separate intent, they're
absorbed into feedback_only / whichever real intent the surrounding text
implies, and the classifier (not the taxonomy) is responsible for telling them
apart from a real complaint.

DESIGN RULE: an intent earns its own label only if it changes what the agent
*does* — a different resolution template, or a different auto/escalate
default. Intents that would get the same reply are merged even if the surface
wording looks very different.

Each intent has:
- a definition (used in the LLM few-shot prompt)
- example phrases (used for the TF-IDF baseline's keyword seeding, and to
  sanity check your golden-set labeling) — these are real, lightly-redacted
  customer messages from the dataset, not invented.
- a default escalation risk tier (used by src/escalate.py as one signal
  among several — see that file for the full logic)
"""

INTENTS = {
    "software_update": {
        "definition": (
            "The update itself is the problem: it won't download or install, "
            "is stuck/verifying, there isn't enough storage to install it, the "
            "device is stuck on the Apple logo or in recovery mode after "
            "updating, or the customer wants to downgrade. NOT a symptom that "
            "merely appeared after an update — that goes to device_bug or "
            "battery_power instead."
        ),
        "examples": [
            "I have been trying to update to iOS 11 all day but it still isn't working",
            "it keeps saying i don't have enough storage so what can i do?",
            "any way to downgrade an iPhone 5s iOS 11.0.3 to iOS 10.3.3??",
        ],
        "default_risk": "low",
    },
    "battery_power": {
        "definition": (
            "Battery drains too fast, won't charge, device overheats, or the "
            "device shuts down unexpectedly at a non-zero charge."
        ),
        "examples": [
            "it shouldn't lose 80% of battery in 6 hrs if I'm not using it and it's on standby",
            "what's up with ios 11 slowing down my phone and giving me 2 hours of battery life?",
            "why does my iphone keep shutting down since the update",
        ],
        "default_risk": "low",
    },
    "device_bug": {
        "definition": (
            "The device or iOS itself misbehaves: lag, freezing, random "
            "restarts, the keyboard/autocorrect bug where typing 'I' produces "
            "a boxed question mark, display glitches, ghost touch, or a "
            "feature that stopped working after an update. Hardware is fine."
        ),
        "examples": [
            "so the letter I is still showing as a box and question mark, what's happening?",
            "please fix the ghost touch issue asap, it's irritating on an expensive phone",
            "every time I try to restart my phone the white screen comes on then it shuts off",
        ],
        "default_risk": "low",
    },
    "connectivity": {
        "definition": (
            "Wi-Fi, cellular data, Bluetooth, AirDrop, personal hotspot, or "
            "AirPods/accessory pairing won't connect or keeps dropping."
        ),
        "examples": [
            "wifi has been slow since the update, same connection I've always used",
            "my airpods keep disconnecting from my phone since the update",
            "bluetooth in my car doesn't work anymore after ios 11",
        ],
        "default_risk": "low",
    },
    "account_billing": {
        "definition": (
            "Apple ID sign-in, locked/disabled accounts, password and 2FA "
            "resets, iCloud storage plans, subscriptions, unexpected charges, "
            "refunds, and reportaproblem.apple.com — anything touching money "
            "or account access."
        ),
        "examples": [
            "my Apple ID is denying me access even after resetting it multiple times",
            "I can't log in to reportaproblem.apple.com, help!",
            "why was I charged twice for the same app",
        ],
        "default_risk": "high",
    },
    "hardware_repair": {
        "definition": (
            "Physical damage or hardware failure: cracked/black screens, "
            "water damage, dead buttons or ports, faulty units out of the "
            "box, plus warranty coverage, repair status, replacements, and "
            "Genius Bar appointments."
        ),
        "examples": [
            "waited weeks for my iPhone X and it arrived with a malfunctioning screen",
            "I processed a screen replacement service and the box still hasn't arrived",
            "third pair of lightning headphones broken in a year, only one side works",
        ],
        "default_risk": "high",
    },
    "app_service": {
        "definition": (
            "An Apple service rather than the device itself: App Store, "
            "iTunes/Apple Music, iMessage/FaceTime delivery, Siri, iCloud "
            "sync/backup, Apple TV, Mail — also third-party apps failing on "
            "an Apple platform."
        ),
        "examples": [
            "since I updated my iPad, Netflix won't open anymore",
            "tones don't play at all, default ringtone plays instead of the one I picked",
            "screen recording sound cancels out when I try to play music during it",
        ],
        "default_risk": "low",
    },
    "howto_feature": {
        "definition": (
            "The customer wants to know how to do something, whether a "
            "feature exists, when a fix is shipping, or is requesting Apple "
            "build or change something. Nothing is broken for them right now."
        ),
        "examples": [
            "how do I stop this update from downloading automatically overnight",
            "any plans to improve how audio apps sync to the control center?",
            "when can we expect a fix for the storage bug on the new update?",
        ],
        "default_risk": "low",
    },
    "feedback_only": {
        "definition": (
            "Venting, praise, sarcasm, jokes, or a bare thank-you with no "
            "question and nothing to resolve. If the message contains any "
            "answerable question or reproducible symptom it is NOT this "
            "intent — classify it by that symptom instead."
        ),
        "examples": [
            "customer service and apple support are the worst",
            "I am so thankful for this help! I can now use the letter I again, thank you!",
            "get it together, my keyboard was fine until the new update",
        ],
        "default_risk": "low",
    },
    "other": {
        "definition": (
            "Not in a supported language, unintelligible, spam, or genuinely "
            "not about Apple support. Also the fallback bucket when a "
            "classifier cannot confidently place a message anywhere else."
        ),
        "examples": [
            "corrige na atualizacao a bateria ai p nois",
            "crazy how this happened the day the iPhone X came out too",
        ],
        "default_risk": "medium",
    },
}

INTENT_LIST = list(INTENTS.keys())


def intents_prompt_block() -> str:
    """Render the intent taxonomy as a block of text for LLM few-shot prompts."""
    lines = []
    for name, meta in INTENTS.items():
        lines.append(f"- {name}: {meta['definition']}")
    return "\n".join(lines)
