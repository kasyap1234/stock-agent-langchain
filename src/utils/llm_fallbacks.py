import os
from typing import Any, Optional

from langchain_groq import ChatGroq

_cerebras_warned = False


def _build_cerebras_fallback(model: str, temperature: float):
    """
    Lazily build a Cerebras chat model if API key is present and SDK is available.
    Returns None if not configured to avoid hard failures.
    """
    global _cerebras_warned
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        return None
    try:
        # Correct import path for Cerebras chat model
        from langchain_community.chat_models.cerebras import ChatCerebras

        return ChatCerebras(model=model, temperature=temperature)
    except Exception as e:
        if not _cerebras_warned:
            print(f"Warning: Cerebras fallback unavailable: {e}")
            _cerebras_warned = True
        return None


class LLMWithFallback:
    """
    Lightweight wrapper that tries primary LLM and falls back to secondary on 429/ratelimit errors.
    Implements minimal LCEL interop so it can be used in chains.
    """

    def __init__(self, primary: Any, fallback: Optional[Any] = None):
        self.primary = primary
        self.fallback = fallback

    def invoke(self, *args, **kwargs):
        try:
            return self.primary.invoke(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "Too Many Requests" in msg) and self.fallback:
                try:
                    return self.fallback.invoke(*args, **kwargs)
                except Exception:
                    # If fallback also fails, re-raise original error
                    raise e
            raise

    def __getattr__(self, name: str):
        # Delegate other attributes to primary
        return getattr(self.primary, name)

    # Minimal LCEL compatibility: allow prompt | llm sequences to work
    def __ror__(self, other):
        # delegate to primary for runnable composition
        if hasattr(self.primary, "__ror__"):
            return self.primary.__ror__(other)
        raise TypeError("Primary LLM does not support LCEL composition")

    def __or__(self, other):
        if hasattr(self.primary, "__or__"):
            return self.primary.__or__(other)
        raise TypeError("Primary LLM does not support LCEL composition")


def groq_with_cerebras_fallback(model: str, temperature: float = 0.0, max_retries: int = 5):
    """
    Build a Groq chat model with a Cerebras fallback if configured.
    """
    primary = ChatGroq(model=model, temperature=temperature, max_retries=max_retries)
    fallback = _build_cerebras_fallback(model=model, temperature=temperature)
    return LLMWithFallback(primary, fallback)

