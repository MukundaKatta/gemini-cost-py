"""Cost calculator for Google Gemini API calls.

Price table last updated: 2026-05-24.
Source: https://ai.google.dev/pricing

All prices are USD per 1,000,000 tokens (per-million).

Gemini-specific pricing notes:
  * Some models (1.5 Flash, 1.5 Pro) have two price tiers based on whether
    the *total context* is ≤128 k or >128 k tokens.  Pass ``long_context=True``
    to ``cost()`` to use the higher tier.
  * Thinking tokens (internal chain-of-thought) are billed separately at
    a different rate on models like Gemini 2.5 Flash.  Pass ``thinking_tokens``
    to include them.
  * Grounding with Google Search incurs a per-1 k requests surcharge.
    Pass ``grounding_requests`` to include it (default 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Pricing model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    """Per-million-token prices for a single Gemini model.

    Args:
        prompt_per_m: USD per 1M prompt tokens (standard / ≤128 k context).
        completion_per_m: USD per 1M completion tokens (standard / ≤128 k).
        prompt_per_m_long: USD per 1M prompt tokens when context > 128 k.
            ``None`` means the same rate as ``prompt_per_m``.
        completion_per_m_long: USD per 1M completion tokens when context > 128 k.
            ``None`` means the same rate as ``completion_per_m``.
        thinking_per_m: USD per 1M *thinking* (chain-of-thought) tokens.
            ``None`` means thinking tokens are not supported / not priced
            separately.
        grounding_per_k_requests: USD per 1 000 grounding requests (Google
            Search).  ``None`` means no grounding surcharge.
    """

    prompt_per_m: float
    completion_per_m: float
    prompt_per_m_long: float | None = None
    completion_per_m_long: float | None = None
    thinking_per_m: float | None = None
    grounding_per_k_requests: float | None = None


class Usage(NamedTuple):
    """Token usage for one Gemini API call.

    Attributes:
        model: Canonical model id.
        prompt_tokens: Total prompt (input) tokens.
        completion_tokens: Completion (output) tokens.
        thinking_tokens: Chain-of-thought tokens (0 if none).
        grounding_requests: Number of grounding requests (0 if none).
        long_context: Whether the long-context tier was applied.
        usd: Computed USD cost.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    grounding_requests: int
    long_context: bool
    usd: float


# ---------------------------------------------------------------------------
# Built-in price table (2026-05-24)
# ---------------------------------------------------------------------------

#: Built-in price table.  Keys are canonical model ids.
DEFAULT_PRICING_TABLE: dict[str, Pricing] = {
    # ---- Gemini 2.5 Flash ----
    # Flat pricing regardless of context length
    "gemini-2.5-flash": Pricing(
        prompt_per_m=0.30,
        completion_per_m=2.50,
        thinking_per_m=3.50,
    ),
    "gemini-2.5-flash-preview-05-20": Pricing(
        prompt_per_m=0.30,
        completion_per_m=2.50,
        thinking_per_m=3.50,
    ),
    # ---- Gemini 2.5 Pro ----
    "gemini-2.5-pro": Pricing(
        prompt_per_m=1.25,
        completion_per_m=10.00,
        prompt_per_m_long=2.50,
        completion_per_m_long=15.00,
        thinking_per_m=3.50,
    ),
    "gemini-2.5-pro-preview-05-06": Pricing(
        prompt_per_m=1.25,
        completion_per_m=10.00,
        prompt_per_m_long=2.50,
        completion_per_m_long=15.00,
        thinking_per_m=3.50,
    ),
    # ---- Gemini 2.0 Flash ----
    "gemini-2.0-flash": Pricing(
        prompt_per_m=0.10,
        completion_per_m=0.40,
    ),
    "gemini-2.0-flash-001": Pricing(
        prompt_per_m=0.10,
        completion_per_m=0.40,
    ),
    "gemini-2.0-flash-lite": Pricing(
        prompt_per_m=0.075,
        completion_per_m=0.30,
    ),
    "gemini-2.0-flash-lite-001": Pricing(
        prompt_per_m=0.075,
        completion_per_m=0.30,
    ),
    # ---- Gemini 2.0 Flash Thinking ----
    "gemini-2.0-flash-thinking-exp": Pricing(
        prompt_per_m=0.10,
        completion_per_m=0.40,
        thinking_per_m=3.50,
    ),
    # ---- Gemini 1.5 Flash ----
    # Context-length-dependent pricing
    "gemini-1.5-flash": Pricing(
        prompt_per_m=0.075,
        completion_per_m=0.30,
        prompt_per_m_long=0.15,
        completion_per_m_long=0.60,
    ),
    "gemini-1.5-flash-001": Pricing(
        prompt_per_m=0.075,
        completion_per_m=0.30,
        prompt_per_m_long=0.15,
        completion_per_m_long=0.60,
    ),
    "gemini-1.5-flash-002": Pricing(
        prompt_per_m=0.075,
        completion_per_m=0.30,
        prompt_per_m_long=0.15,
        completion_per_m_long=0.60,
    ),
    "gemini-1.5-flash-8b": Pricing(
        prompt_per_m=0.0375,
        completion_per_m=0.15,
        prompt_per_m_long=0.075,
        completion_per_m_long=0.30,
    ),
    # ---- Gemini 1.5 Pro ----
    "gemini-1.5-pro": Pricing(
        prompt_per_m=1.25,
        completion_per_m=5.00,
        prompt_per_m_long=2.50,
        completion_per_m_long=10.00,
    ),
    "gemini-1.5-pro-001": Pricing(
        prompt_per_m=1.25,
        completion_per_m=5.00,
        prompt_per_m_long=2.50,
        completion_per_m_long=10.00,
    ),
    "gemini-1.5-pro-002": Pricing(
        prompt_per_m=1.25,
        completion_per_m=5.00,
        prompt_per_m_long=2.50,
        completion_per_m_long=10.00,
    ),
    # ---- Gemini 1.0 Pro ----
    "gemini-1.0-pro": Pricing(
        prompt_per_m=0.50,
        completion_per_m=1.50,
    ),
    # ---- Gemma (open models via AI Studio, approximate prices) ----
    "gemma-3-27b-it": Pricing(
        prompt_per_m=0.10,
        completion_per_m=0.20,
    ),
    "gemma-3-12b-it": Pricing(
        prompt_per_m=0.05,
        completion_per_m=0.10,
    ),
}

# ---------------------------------------------------------------------------
# Alias map
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # Gemini 2.5
    "gemini-2.5-flash-latest": "gemini-2.5-flash",
    "gemini-2.5-pro-latest": "gemini-2.5-pro",
    # Gemini 2.0
    "gemini-2.0-flash-latest": "gemini-2.0-flash",
    "gemini-2.0-flash-exp": "gemini-2.0-flash",
    "gemini-2.0-flash-lite-latest": "gemini-2.0-flash-lite",
    # Gemini 1.5
    "gemini-1.5-flash-latest": "gemini-1.5-flash",
    "gemini-1.5-pro-latest": "gemini-1.5-pro",
    "gemini-1.5-flash-8b-latest": "gemini-1.5-flash-8b",
    # Gemini 1.0
    "gemini-pro": "gemini-1.0-pro",
    "gemini-1.0-pro-latest": "gemini-1.0-pro",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_model_id(model: str) -> str:
    """Return the canonical model id for *model*.

    Resolves aliases. Returns the input unchanged if already canonical or
    unknown.

    Args:
        model: A raw model id string.

    Returns:
        Canonical model id.

    Example::

        >>> normalize_model_id("gemini-pro")
        'gemini-1.0-pro'
    """
    m = model.strip()
    return _ALIASES.get(m, m)


def default_pricing(model: str) -> Pricing | None:
    """Return the built-in :class:`Pricing` for *model*, or ``None``.

    Performs :func:`normalize_model_id` before lookup.
    """
    return DEFAULT_PRICING_TABLE.get(normalize_model_id(model))


def known_models() -> list[str]:
    """Return all canonical model ids in the built-in table (sorted)."""
    return sorted(DEFAULT_PRICING_TABLE)


def cost(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    thinking_tokens: int = 0,
    grounding_requests: int = 0,
    long_context: bool = False,
    pricing_table: dict[str, Pricing] | None = None,
) -> float:
    """Calculate the USD cost of a Gemini API call.

    Args:
        model: Model id (alias or canonical).
        prompt_tokens: Prompt (input) token count.
        completion_tokens: Completion (output) token count.
        thinking_tokens: Chain-of-thought (thinking) token count.
            Billed separately on models that support thinking.
        grounding_requests: Number of Google Search grounding requests.
            Billed per 1 000 requests.
        long_context: ``True`` to use the >128 k pricing tier.
            Has no effect on models without context-length-dependent pricing.
        pricing_table: Optional override for the built-in price table.

    Returns:
        Cost in USD as a ``float``.

    Raises:
        KeyError: If the model is not in the price table.

    Example::

        >>> from gemini_cost import cost
        >>> round(cost(model="gemini-2.0-flash", prompt_tokens=1000, completion_tokens=200), 8)
        0.00018
    """
    table = pricing_table or DEFAULT_PRICING_TABLE
    canonical = normalize_model_id(model)
    try:
        p = table[canonical]
    except KeyError:
        raise KeyError(
            f"Model {model!r} (canonical: {canonical!r}) not found in pricing table. "
            f"Pass a custom pricing_table or use one of: {known_models()}"
        ) from None

    # Choose per-million rates based on context tier
    if long_context and p.prompt_per_m_long is not None:
        p_rate = p.prompt_per_m_long
        c_rate = (
            p.completion_per_m_long if p.completion_per_m_long is not None else p.completion_per_m
        )  # noqa: E501
    else:
        p_rate = p.prompt_per_m
        c_rate = p.completion_per_m

    prompt_cost = prompt_tokens * p_rate / 1_000_000
    completion_cost = completion_tokens * c_rate / 1_000_000

    # Thinking tokens at separate rate (or fallback to completion rate)
    if thinking_tokens > 0:
        think_rate = p.thinking_per_m if p.thinking_per_m is not None else c_rate
        thinking_cost = thinking_tokens * think_rate / 1_000_000
    else:
        thinking_cost = 0.0

    # Grounding surcharge
    if grounding_requests > 0 and p.grounding_per_k_requests is not None:
        grounding_cost = grounding_requests * p.grounding_per_k_requests / 1_000
    else:
        grounding_cost = 0.0

    return prompt_cost + completion_cost + thinking_cost + grounding_cost


def usage(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    thinking_tokens: int = 0,
    grounding_requests: int = 0,
    long_context: bool = False,
    pricing_table: dict[str, Pricing] | None = None,
) -> Usage:
    """Like :func:`cost` but returns a :class:`Usage` named-tuple.

    Args:
        model: Model id (alias or canonical).
        prompt_tokens: Prompt tokens.
        completion_tokens: Completion tokens.
        thinking_tokens: Thinking tokens.
        grounding_requests: Grounding request count.
        long_context: Whether to use the long-context tier.
        pricing_table: Optional override.

    Returns:
        :class:`Usage` with all fields and computed ``usd``.
    """
    usd = cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        thinking_tokens=thinking_tokens,
        grounding_requests=grounding_requests,
        long_context=long_context,
        pricing_table=pricing_table,
    )
    return Usage(
        model=normalize_model_id(model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        thinking_tokens=thinking_tokens,
        grounding_requests=grounding_requests,
        long_context=long_context,
        usd=usd,
    )
