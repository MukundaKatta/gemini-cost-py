"""gemini-cost: cost calculator for Google Gemini API calls.

Handles Gemini's context-size-dependent pricing (≤128 k vs >128 k tokens),
thinking tokens, and grounding surcharge. Zero external dependencies.

Part of the @mukundakatta cost-calculator trio:
  - claude-cost-py   https://github.com/MukundaKatta/claude-cost-py
  - openai-cost-py   https://github.com/MukundaKatta/openai-cost-py
  - gemini-cost-py   (this library)

Quick start::

    from gemini_cost import cost

    # Standard call
    usd = cost(
        model="gemini-2.0-flash",
        prompt_tokens=1000,
        completion_tokens=500,
    )

    # Long-context call (>128 k prompt tokens — higher rate on some models)
    usd = cost(
        model="gemini-1.5-pro",
        prompt_tokens=200_000,
        completion_tokens=2000,
        long_context=True,
    )

    # Thinking tokens (billed at a separate rate on Flash-Thinking / Pro)
    usd = cost(
        model="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=500,
        thinking_tokens=1500,
    )
"""

from .core import (
    DEFAULT_PRICING_TABLE,
    Pricing,
    Usage,
    cost,
    default_pricing,
    known_models,
    normalize_model_id,
    usage,
)

__all__ = [
    "DEFAULT_PRICING_TABLE",
    "Pricing",
    "Usage",
    "cost",
    "default_pricing",
    "known_models",
    "normalize_model_id",
    "usage",
]

__version__ = "0.1.0"
