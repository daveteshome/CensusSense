"""Shared LLM client setup for gatekeeper.py and responder.py.

Provider/model history, for context on why this isn't the "obvious"
default: Gemini was tried first. gemini-flash-latest took 15-26s per
call (once with a 503 "high demand"); several pinned Gemini versions
returned 404 "no longer available to new users" or 429 quota=0/depleted
prepaid credits depending on the API key's project, and enabling billing
required a minimum top-up the user didn't want to pay for a 24h project.
Switched to Groq: a free developer tier with no card required, an
OpenAI-compatible API (so the standard `openai` SDK works unmodified
against Groq's endpoint), and fast LPU-backed inference.
"""

from openai import OpenAI

from config import Config

MODEL_NAME = "llama-3.3-70b-versatile"
TIMEOUT_SECONDS = 12.0

_cached_client: OpenAI | None = None


class LLMError(Exception):
    pass


def get_client(cfg: Config) -> OpenAI:
    global _cached_client
    if _cached_client is None:
        _cached_client = OpenAI(
            api_key=cfg.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=TIMEOUT_SECONDS,
        )
    return _cached_client
