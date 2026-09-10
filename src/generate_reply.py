"""
Reply generation for the three systems:

1. trivial: one canned generic reply regardless of intent
2. simple: intent -> template reply (filled from most common brand phrasing
   per intent, no LLM)
3. llm: LLM generates a reply grounded in retrieved historical brand replies
   to similar past customer messages (lightweight RAG)
"""

from llm_client import complete
from retrieve import RetrievedExample

TRIVIAL_REPLY = (
    "Thanks for reaching out! A member of our support team will get back to "
    "you as soon as possible."
)

TEMPLATE_REPLIES = {
    "software_update": (
        "Sorry the update isn't going through. Connect to Wi-Fi and power, then check "
        "Settings > General > iPhone Storage for a partial update file to remove, and retry "
        "from Settings > General > Software Update. Updating through a computer usually gets "
        "past a stuck install."
    ),
    "battery_power": (
        "Battery drain like that isn't expected. Check Settings > Battery for which app is "
        "using the most power over the last 10 days, and check Battery Health while you're "
        "there. If one app dominates, offloading and reinstalling it usually helps."
    ),
    "device_bug": (
        "Thanks for flagging that. A force restart clears most of these: press and release "
        "Volume Up, press and release Volume Down, then hold the Side button until the Apple "
        "logo appears. Let us know if it comes back, with your model and iOS version."
    ),
    "connectivity": (
        "Let's reset the radios: toggle Airplane Mode on and off, then if it persists go to "
        "Settings > General > Reset > Reset Network Settings. For Bluetooth accessories, forget "
        "the device and re-pair it."
    ),
    "account_billing": (
        "For anything on your Apple ID or a charge we'll need to look at the account, which we "
        "can't do over a public tweet. Send us a DM and we'll pick it up from there."
    ),
    "hardware_repair": (
        "That sounds like it needs hands on the device. support.apple.com will show your "
        "coverage and the nearest service options. DM us your serial number and we'll check "
        "what you're entitled to."
    ),
    "app_service": (
        "Let's isolate it: sign out and back into the service in Settings, then check Apple's "
        "System Status page in case it's service-side. If it's one app, offloading and "
        "reinstalling it clears most cases."
    ),
    "howto_feature": (
        "Happy to point you the right way — tell us your device and iOS version and exactly "
        "what you're trying to do. Feature suggestions are welcome too at apple.com/feedback."
    ),
    "feedback_only": (
        "Thanks for taking the time to tell us — we're passing this on. If there's anything "
        "specific we can look at for you, we're here."
    ),
    "other": (
        "We want to make sure we point you the right way — could you tell us a bit more about "
        "what's happening and which device you're using?"
    ),
}


def generate_template_reply(intent: str) -> str:
    return TEMPLATE_REPLIES.get(intent, TRIVIAL_REPLY)


def generate_llm_reply(customer_text: str, intent: str, retrieved: list[RetrievedExample]) -> str:
    context_block = "\n\n".join(
        f"Similar past customer message: {ex.customer_text}\nBrand's actual reply: {ex.brand_reply_text}"
        for ex in retrieved
    ) or "(no similar historical examples found)"

    system = (
        "You are @AppleSupport, replying to a customer on Twitter. "
        "Write a short (1-3 sentence), on-brand, helpful reply to the customer's message below. "
        "Ground your reply in how Apple has historically resolved similar issues, shown in the "
        "examples. Do not invent policies, refund amounts, warranty terms, or timelines not "
        "supported by the examples. If you are unsure of a specific fact, keep the reply general "
        "and ask for more info (device model, iOS version) instead of guessing.\n\n"
        f"The customer's message intent has been classified as: {intent}\n\n"
        f"Historical examples of how this brand resolved similar issues:\n{context_block}"
    )
    user = f"Customer's message: {customer_text}\n\nWrite the reply."
    # max_tokens is generous (not "200 for a 1-3 sentence reply") because some
    # free-tier models (e.g. reasoning-tuned ones on OpenRouter) spend a
    # hidden reasoning budget before the visible reply — too tight a cap
    # truncates mid-thought and returns empty content, not a short answer.
    return complete(system, user, max_tokens=1200, temperature=0.4).strip()
