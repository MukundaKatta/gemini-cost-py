# gemini-cost-py

USD cost calculator for Google Gemini API calls. Handles long-context pricing tiers, thinking tokens, and grounding surcharge. Zero dependencies.

Completes the cost-calculator trio with [claude-cost-py](https://github.com/MukundaKatta/claude-cost-py) and [openai-cost-py](https://github.com/MukundaKatta/openai-cost-py).

## Install

```bash
pip install gemini-cost
```

## Quickstart

```python
from gemini_cost import cost

# Standard call
usd = cost(model="gemini-2.0-flash", prompt_tokens=1000, completion_tokens=500)

# Long-context tier (>128k tokens — higher rate on 1.5 Flash / 1.5 Pro / 2.5 Pro)
usd = cost(
    model="gemini-1.5-pro",
    prompt_tokens=200_000,
    completion_tokens=2000,
    long_context=True,
)

# Thinking tokens (2.5 Flash, 2.5 Pro)
usd = cost(
    model="gemini-2.5-flash",
    prompt_tokens=1000,
    completion_tokens=500,
    thinking_tokens=1500,
)
```

## Models (2026-05-24 prices)

| Model | Prompt/M | Completion/M | Notes |
| --- | --- | --- | --- |
| `gemini-2.5-flash` | $0.30 | $2.50 | Thinking: $3.50/M |
| `gemini-2.5-pro` | $1.25 / $2.50 | $10 / $15 | Long-ctx + thinking |
| `gemini-2.0-flash` | $0.10 | $0.40 | |
| `gemini-2.0-flash-lite` | $0.075 | $0.30 | |
| `gemini-1.5-flash` | $0.075 / $0.15 | $0.30 / $0.60 | Long-ctx tier |
| `gemini-1.5-pro` | $1.25 / $2.50 | $5 / $10 | Long-ctx tier |
| `gemini-1.0-pro` | $0.50 | $1.50 | |
| `gemma-3-27b-it` | $0.10 | $0.20 | |

Aliases resolved: `gemini-pro`, `gemini-2.5-flash-latest`, `gemini-1.5-flash-latest`, etc.

## API

### `cost(...) -> float`

```python
cost(
    model="gemini-2.5-flash",
    prompt_tokens=1000,
    completion_tokens=500,
    thinking_tokens=0,          # chain-of-thought tokens
    grounding_requests=0,       # Google Search requests
    long_context=False,         # use >128k pricing tier
    pricing_table=None,         # override built-in table
) -> float  # USD
```

### `usage(...) -> Usage`

Same args, returns a `Usage` NamedTuple with all fields plus `usd`.

### Helpers

- `normalize_model_id(model)` — alias → canonical
- `default_pricing(model)` → `Pricing | None`
- `known_models()` → sorted list of canonical ids

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

43 tests. Covers standard pricing, long-context tiers, thinking tokens, grounding, alias resolution, custom tables, and `Pricing` immutability.

## License

MIT.
