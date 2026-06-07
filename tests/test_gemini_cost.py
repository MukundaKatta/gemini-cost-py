"""Tests for gemini-cost-py."""

from __future__ import annotations

import pytest

from gemini_cost import (
    DEFAULT_PRICING_TABLE,
    Pricing,
    Usage,
    cost,
    default_pricing,
    known_models,
    normalize_model_id,
    usage,
)

# ---------------------------------------------------------------------------
# normalize_model_id
# ---------------------------------------------------------------------------


def test_normalize_known_alias():
    assert normalize_model_id("gemini-pro") == "gemini-1.0-pro"


def test_normalize_gemini25_latest():
    assert normalize_model_id("gemini-2.5-flash-latest") == "gemini-2.5-flash"


def test_normalize_canonical_passthrough():
    assert normalize_model_id("gemini-2.0-flash") == "gemini-2.0-flash"


def test_normalize_unknown_passthrough():
    assert normalize_model_id("my-custom-gemini-model") == "my-custom-gemini-model"


def test_normalize_strips_whitespace():
    assert normalize_model_id("  gemini-pro  ") == "gemini-1.0-pro"


def test_normalize_15_flash_latest():
    assert normalize_model_id("gemini-1.5-flash-latest") == "gemini-1.5-flash"


def test_normalize_15_pro_latest():
    assert normalize_model_id("gemini-1.5-pro-latest") == "gemini-1.5-pro"


def test_normalize_20_flash_exp():
    assert normalize_model_id("gemini-2.0-flash-exp") == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# default_pricing
# ---------------------------------------------------------------------------


def test_default_pricing_known_model():
    p = default_pricing("gemini-2.0-flash")
    assert p is not None
    assert isinstance(p, Pricing)


def test_default_pricing_via_alias():
    p = default_pricing("gemini-pro")
    assert p is not None
    assert p == default_pricing("gemini-1.0-pro")


def test_default_pricing_unknown_returns_none():
    assert default_pricing("not-a-gemini-model-xyz") is None


def test_default_pricing_gemma():
    p = default_pricing("gemma-3-27b-it")
    assert p is not None


# ---------------------------------------------------------------------------
# known_models
# ---------------------------------------------------------------------------


def test_known_models_sorted():
    models = known_models()
    assert models == sorted(models)


def test_known_models_includes_flash():
    assert "gemini-2.0-flash" in known_models()


def test_known_models_includes_pro():
    assert "gemini-1.5-pro" in known_models()


# ---------------------------------------------------------------------------
# cost — basic
# ---------------------------------------------------------------------------


def test_cost_gemini20_flash():
    # $0.10/M prompt, $0.40/M completion
    # 1000 + 200 = 0.0001 + 0.00008 = 0.00018
    usd = cost(model="gemini-2.0-flash", prompt_tokens=1000, completion_tokens=200)
    assert usd == pytest.approx(0.00018)


def test_cost_gemini15_flash_short():
    # ≤128k: $0.075/M prompt, $0.30/M completion
    # 1000 + 500 = 0.000075 + 0.000150 = 0.000225
    usd = cost(model="gemini-1.5-flash", prompt_tokens=1000, completion_tokens=500)
    assert usd == pytest.approx(0.000225)


def test_cost_gemini15_pro_short():
    # ≤128k: $1.25/M prompt, $5.00/M completion
    # 1000 + 200 = 0.00125 + 0.001 = 0.00225
    usd = cost(model="gemini-1.5-pro", prompt_tokens=1000, completion_tokens=200)
    assert usd == pytest.approx(0.00225)


def test_cost_gemini10_pro():
    # $0.50/M prompt, $1.50/M completion
    # 1000 + 1000 = 0.0005 + 0.0015 = 0.002
    usd = cost(model="gemini-1.0-pro", prompt_tokens=1000, completion_tokens=1000)
    assert usd == pytest.approx(0.002)


def test_cost_zero_tokens():
    usd = cost(model="gemini-2.0-flash", prompt_tokens=0, completion_tokens=0)
    assert usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cost — long context
# ---------------------------------------------------------------------------


def test_cost_15_flash_long_context_higher():
    short = cost(model="gemini-1.5-flash", prompt_tokens=200_000, completion_tokens=1000)
    long = cost(
        model="gemini-1.5-flash",
        prompt_tokens=200_000,
        completion_tokens=1000,
        long_context=True,
    )
    assert long > short


def test_cost_15_pro_long_context_rate():
    # >128k: $2.50/M prompt, $10.00/M completion
    # 200_000 prompt + 1000 completion
    # = 200_000 * 2.50/1e6 + 1000 * 10.00/1e6
    # = 0.5 + 0.01 = 0.51
    usd = cost(
        model="gemini-1.5-pro",
        prompt_tokens=200_000,
        completion_tokens=1000,
        long_context=True,
    )
    assert usd == pytest.approx(0.51)


def test_cost_15_flash_long_context_rate():
    # >128k: $0.15/M prompt, $0.60/M completion
    # 200_000 prompt + 500 completion
    # = 200_000 * 0.15/1e6 + 500 * 0.60/1e6
    # = 0.03 + 0.0003 = 0.0303
    usd = cost(
        model="gemini-1.5-flash",
        prompt_tokens=200_000,
        completion_tokens=500,
        long_context=True,
    )
    assert usd == pytest.approx(0.0303)


def test_cost_long_context_no_effect_on_flat_models():
    # gemini-2.0-flash has no long-context pricing → same rate
    short = cost(model="gemini-2.0-flash", prompt_tokens=200_000, completion_tokens=1000)
    long = cost(
        model="gemini-2.0-flash",
        prompt_tokens=200_000,
        completion_tokens=1000,
        long_context=True,
    )
    assert long == pytest.approx(short)


# ---------------------------------------------------------------------------
# cost — thinking tokens
# ---------------------------------------------------------------------------


def test_cost_gemini25_flash_thinking():
    # $0.30/M prompt, $2.50/M completion, $3.50/M thinking
    # 1000 prompt + 500 completion + 1500 thinking
    # = 0.0003 + 0.00125 + 0.00525 = 0.0068
    usd = cost(
        model="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=500,
        thinking_tokens=1500,
    )
    assert usd == pytest.approx(0.0068)


def test_cost_thinking_zero_no_thinking_charge():
    no_think = cost(model="gemini-2.5-flash", prompt_tokens=1000, completion_tokens=500)
    with_zero = cost(
        model="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=500,
        thinking_tokens=0,
    )
    assert no_think == pytest.approx(with_zero)


def test_cost_thinking_on_model_without_thinking_rate():
    # gemini-2.0-flash has no thinking_per_m → falls back to completion rate
    usd_think = cost(
        model="gemini-2.0-flash",
        prompt_tokens=1000,
        completion_tokens=500,
        thinking_tokens=200,
    )
    # 1000 * 0.10/1e6 + 500 * 0.40/1e6 + 200 * 0.40/1e6
    # = 0.0001 + 0.0002 + 0.00008 = 0.00038
    assert usd_think == pytest.approx(0.00038)


# ---------------------------------------------------------------------------
# cost — grounding
# ---------------------------------------------------------------------------


def test_cost_grounding_not_configured():
    # gemini-2.0-flash has no grounding_per_k_requests → no surcharge
    usd = cost(
        model="gemini-2.0-flash",
        prompt_tokens=100,
        completion_tokens=50,
        grounding_requests=1000,
    )
    normal = cost(model="gemini-2.0-flash", prompt_tokens=100, completion_tokens=50)
    assert usd == pytest.approx(normal)


def test_cost_grounding_zero_requests_no_charge():
    usd = cost(
        model="gemini-2.0-flash",
        prompt_tokens=100,
        completion_tokens=50,
        grounding_requests=0,
    )
    normal = cost(model="gemini-2.0-flash", prompt_tokens=100, completion_tokens=50)
    assert usd == pytest.approx(normal)


def test_cost_grounding_surcharge_applied():
    # Custom model with a grounding surcharge of $35 / 1000 requests.
    # 2000 requests → 2000 * 35 / 1000 = 70.0 surcharge on top of token cost.
    custom = {
        "grounded-model": Pricing(
            prompt_per_m=0.0,
            completion_per_m=0.0,
            grounding_per_k_requests=35.0,
        )
    }
    usd = cost(
        model="grounded-model",
        prompt_tokens=0,
        completion_tokens=0,
        grounding_requests=2000,
        pricing_table=custom,
    )
    assert usd == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# cost — alias resolution
# ---------------------------------------------------------------------------


def test_cost_via_alias():
    usd_alias = cost(model="gemini-pro", prompt_tokens=1000, completion_tokens=100)
    usd_canonical = cost(model="gemini-1.0-pro", prompt_tokens=1000, completion_tokens=100)
    assert usd_alias == pytest.approx(usd_canonical)


def test_cost_unknown_model_raises():
    with pytest.raises(KeyError, match="not found in pricing table"):
        cost(model="not-a-gemini-model-xyz", prompt_tokens=100, completion_tokens=50)


# ---------------------------------------------------------------------------
# cost — custom pricing table
# ---------------------------------------------------------------------------


def test_cost_custom_pricing_table():
    custom = {"my-model": Pricing(prompt_per_m=1.00, completion_per_m=2.00)}
    usd = cost(
        model="my-model",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        pricing_table=custom,
    )
    assert usd == pytest.approx(3.00)


# ---------------------------------------------------------------------------
# usage()
# ---------------------------------------------------------------------------


def test_usage_is_publicly_exported():
    import gemini_cost

    assert "usage" in gemini_cost.__all__
    assert gemini_cost.usage is usage


def test_usage_returns_usage_namedtuple():
    u = usage(model="gemini-2.0-flash", prompt_tokens=500, completion_tokens=100)
    assert isinstance(u, Usage)


def test_usage_canonical_model():
    u = usage(model="gemini-pro", prompt_tokens=100, completion_tokens=10)
    assert u.model == "gemini-1.0-pro"


def test_usage_usd_matches_cost():
    u = usage(
        model="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=500,
        thinking_tokens=800,
        long_context=False,
    )
    expected = cost(
        model="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=500,
        thinking_tokens=800,
    )
    assert u.usd == pytest.approx(expected)


def test_usage_fields_preserved():
    u = usage(
        model="gemini-1.5-pro",
        prompt_tokens=200_000,
        completion_tokens=1000,
        thinking_tokens=0,
        grounding_requests=500,
        long_context=True,
    )
    assert u.prompt_tokens == 200_000
    assert u.completion_tokens == 1000
    assert u.grounding_requests == 500
    assert u.long_context is True


# ---------------------------------------------------------------------------
# Pricing dataclass
# ---------------------------------------------------------------------------


def test_pricing_frozen():
    p = Pricing(prompt_per_m=1.0, completion_per_m=2.0)
    with pytest.raises(Exception):
        p.prompt_per_m = 99.0  # type: ignore[misc]


def test_pricing_optional_fields_default_none():
    p = Pricing(prompt_per_m=0.10, completion_per_m=0.40)
    assert p.prompt_per_m_long is None
    assert p.thinking_per_m is None
    assert p.grounding_per_k_requests is None


# ---------------------------------------------------------------------------
# Default table sanity
# ---------------------------------------------------------------------------


def test_all_prices_non_negative():
    for model_id, p in DEFAULT_PRICING_TABLE.items():
        assert p.prompt_per_m >= 0, f"{model_id}: prompt price must be >= 0"
        assert p.completion_per_m >= 0, f"{model_id}: completion price must be >= 0"


def test_25_flash_has_thinking_rate():
    p = DEFAULT_PRICING_TABLE["gemini-2.5-flash"]
    assert p.thinking_per_m is not None


def test_15_pro_has_long_context_rates():
    p = DEFAULT_PRICING_TABLE["gemini-1.5-pro"]
    assert p.prompt_per_m_long is not None
    assert p.prompt_per_m_long > p.prompt_per_m
