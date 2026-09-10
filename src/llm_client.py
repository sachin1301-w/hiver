"""
Thin, provider-agnostic LLM client.

No API keys live in this file or anywhere in this repo. Set LLM_PROVIDER and
the matching *_API_KEY in a local .env file (see .env.example). This module
just reads them from the environment at call time.
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

# Free-tier endpoints (OpenRouter's shared :free pool especially) rate-limit
# under load and expect the caller to back off and retry, not give up.
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))
RETRY_BASE_DELAY_SEC = float(os.getenv("LLM_RETRY_BASE_DELAY", "3"))


class LLMNotConfiguredError(RuntimeError):
    pass


class EmptyCompletionError(RuntimeError):
    """A reasoning-heavy model spent its whole max_tokens budget thinking."""

    pass


def _require_key(var_name: str) -> str:
    key = os.getenv(var_name)
    if not key:
        raise LLMNotConfiguredError(
            f"{var_name} is not set. Copy .env.example to .env and add your own "
            f"API key before running any LLM-based command."
        )
    return key


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if "RateLimit" in name:
        return True
    status = getattr(exc, "status_code", None)
    return status == 429


def complete(system: str, user: str, max_tokens: int = 1000, temperature: float = 0.2) -> str:
    """
    Send a single-turn (system, user) request to the configured provider and
    return the plain text response. Raises LLMNotConfiguredError with a clear
    message if no key is set, instead of failing with a confusing traceback.

    Retries with exponential backoff on 429 (rate limit) errors — free-tier
    shared pools (e.g. OpenRouter's :free models) throttle under load and
    recover within seconds; a paid key rarely hits this path at all.

    Separately, if a reasoning-tuned model spends its entire max_tokens budget
    on hidden reasoning and returns empty content (EmptyCompletionError),
    retries with a doubled budget instead of backing off — that failure mode
    has nothing to do with load, so waiting would not help.
    """
    last_exc = None
    budget = max_tokens
    for attempt in range(MAX_RETRIES + 1):
        try:
            if PROVIDER == "anthropic":
                return _complete_anthropic(system, user, budget, temperature)
            elif PROVIDER == "openai":
                return _complete_openai(system, user, budget, temperature)
            else:
                raise LLMNotConfiguredError(
                    f"Unknown LLM_PROVIDER='{PROVIDER}'. Use 'anthropic' or 'openai' in .env."
                )
        except LLMNotConfiguredError:
            raise
        except EmptyCompletionError as e:
            if attempt == MAX_RETRIES:
                raise
            budget = min(budget * 2, 8000)
            print(f"[llm_client] empty completion, retrying with max_tokens={budget} (attempt {attempt + 1}/{MAX_RETRIES})...")
            last_exc = e
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY_SEC * (2**attempt)
            print(f"[llm_client] rate-limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(delay)
            last_exc = e
    raise last_exc  # pragma: no cover — loop always returns or raises above


def _complete_anthropic(system: str, user: str, max_tokens: int, temperature: float) -> str:
    _require_key("ANTHROPIC_API_KEY")
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    # NOTE: temperature/top_p/top_k sampling controls were removed from
    # messages.create() on current-generation models (adaptive thinking is on
    # by default instead), so `temperature` is accepted here for interface
    # symmetry with the OpenAI path but intentionally not forwarded.
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError("Model declined the request (safety refusal).")
    return "".join(block.text for block in resp.content if block.type == "text")


def _complete_openai(system: str, user: str, max_tokens: int, temperature: float) -> str:
    _require_key("OPENAI_API_KEY")
    from openai import OpenAI

    # OPENAI_BASE_URL lets this same code path hit any OpenAI-compatible
    # endpoint (OpenRouter, Groq, a local Ollama server) by pointing the
    # official OpenAI SDK's base_url elsewhere — no separate provider needed.
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    extra_headers = {}
    if base_url and "openrouter.ai" in base_url:
        # Optional but recommended by OpenRouter for routing/rankings.
        extra_headers = {
            "HTTP-Referer": "https://github.com/hiver-support-agent",
            "X-Title": "Hiver AI Support Agent",
        }
    client = OpenAI(base_url=base_url)  # reads OPENAI_API_KEY from env
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_headers=extra_headers or None,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason
    if not content:
        # Some free-tier models spend their entire max_tokens budget on a
        # hidden reasoning channel (message.reasoning) and leave the visible
        # `content` empty, reporting finish_reason="length". That is a
        # legitimate response, not a network error — retrying the exact same
        # request would just repeat it — so the caller (`complete`, which
        # retries on rate limits) needs a distinct signal to grow the budget
        # and try again rather than looping forever on the same cap.
        raise EmptyCompletionError(
            f"Model returned empty content (finish_reason={finish_reason!r}); "
            f"its hidden reasoning likely consumed the max_tokens={max_tokens} budget."
        )
    return content


def complete_json(system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> dict:
    """
    Same as complete(), but instructs the model to return ONLY JSON and parses
    it. Strips markdown code fences if the model adds them anyway.
    """
    system_json = system + "\n\nRespond with ONLY valid JSON. No markdown, no preamble."
    raw = complete(system_json, user, max_tokens=max_tokens, temperature=temperature)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if cleaned.lower().startswith("json") else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{raw}") from e
