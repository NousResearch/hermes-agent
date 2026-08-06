#!/usr/bin/env python3
"""
Cost Optimizer Tool - Analyze and optimize API costs

Provides cost analysis, provider cost comparison, and optimization recommendations.
Uses agent/usage_pricing.py for route-aware, versioned pricing instead of a static table.
"""

import json
from typing import Any, Dict, List, Optional


def _get_pricing(model: str, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Look up pricing via the existing usage_pricing module.

    Returns dict with input_cost_per_million, output_cost_per_million, or None.
    """
    try:
        from agent.usage_pricing import get_pricing_entry
        entry = get_pricing_entry(model, provider=provider)
        if entry is None:
            return None
        result = {}
        if entry.input_cost_per_million is not None:
            result["input"] = float(entry.input_cost_per_million)
        if entry.output_cost_per_million is not None:
            result["output"] = float(entry.output_cost_per_million)
        if entry.cache_read_cost_per_million is not None:
            result["cache_read"] = float(entry.cache_read_cost_per_million)
        if entry.cache_write_cost_per_million is not None:
            result["cache_write"] = float(entry.cache_write_cost_per_million)
        return result if result else None
    except ImportError:
        return None


# Fallback catalog when usage_pricing is not importable (e.g. tests).
# Covers the most common models with approximate rates ($/MTok).
_FALLBACK_RATES: Dict[str, Dict[str, Any]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "provider": "anthropic"},
    "gpt-4o": {"input": 2.5, "output": 10.0, "provider": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "openai"},
    "deepseek-v3": {"input": 0.27, "output": 1.10, "provider": "deepseek"},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30, "provider": "google"},
    "gemini-2.5-pro": {"input": 0.50, "output": 1.50, "provider": "google"},
}


def _resolve_rates(model: str, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve pricing for a model, trying usage_pricing then fallback."""
    pricing = _get_pricing(model, provider=provider)
    if pricing and "input" in pricing and "output" in pricing:
        return pricing
    fb = _FALLBACK_RATES.get(model)
    if fb:
        return fb
    return None


def _validate_tokens(value: Any, name: str) -> int:
    """Validate a token count is a non-negative integer."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 0
    if v < 0:
        return 0
    return v


def cost_optimizer(
    model: Optional[str] = None,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    provider: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    """
    Analyze API costs and provide optimization recommendations.

    Args:
        model: Specific model to analyze (omit for all models comparison)
        input_tokens: Estimated input tokens (default 1000, must be >= 0)
        output_tokens: Estimated output tokens (default 500, must be >= 0)
        provider: Compare costs across specific provider

    Returns:
        JSON string with cost analysis and optimization recommendations
    """
    input_tokens = _validate_tokens(input_tokens, "input_tokens")
    output_tokens = _validate_tokens(output_tokens, "output_tokens")

    result: Dict[str, Any] = {"success": True}

    if model:
        rates = _resolve_rates(model, provider=provider)
        if rates is None:
            return json.dumps({
                "success": False,
                "error": f"Unknown model: {model}. Try a full model ID like 'claude-sonnet-4-6' or 'gpt-4o'.",
            })
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        total_cost = input_cost + output_cost

        result["analysis"] = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "costs": {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6),
                "cost_per_1k_input": round(rates["input"] / 1000, 6),
                "cost_per_1k_output": round(rates["output"] / 1000, 6),
            },
        }

        alt_models = []
        for alt_name in _FALLBACK_RATES:
            if alt_name == model:
                continue
            alt_rates = _resolve_rates(alt_name)
            if alt_rates is None:
                continue
            alt_input = (input_tokens / 1_000_000) * alt_rates["input"]
            alt_output = (output_tokens / 1_000_000) * alt_rates["output"]
            alt_total = alt_input + alt_output
            savings = total_cost - alt_total

            if savings > 0.001:
                alt_models.append({
                    "model": alt_name,
                    "total_cost": round(alt_total, 6),
                    "savings": round(savings, 6),
                    "savings_pct": round((savings / total_cost) * 100, 1) if total_cost > 0 else 0,
                })

        alt_models.sort(key=lambda x: x["total_cost"])
        if alt_models:
            result["alternatives"] = alt_models[:5]
            cheapest = alt_models[0]
            result["recommendation"] = (
                f"Switch to {cheapest['model']} to save ${cheapest['savings']:.4f} "
                f"({cheapest['savings_pct']}%) on this request."
            )

    else:
        comparisons = []
        for m_name, m_rates in _FALLBACK_RATES.items():
            if provider and m_rates.get("provider") != provider:
                continue
            input_cost = (input_tokens / 1_000_000) * m_rates["input"]
            output_cost = (output_tokens / 1_000_000) * m_rates["output"]
            total = input_cost + output_cost
            comparisons.append({
                "model": m_name,
                "provider": m_rates.get("provider", "unknown"),
                "total_cost": round(total, 6),
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
            })

        comparisons.sort(key=lambda x: x["total_cost"])
        result["comparison"] = comparisons

        if comparisons:
            cheapest = comparisons[0]
            most_expensive = comparisons[-1]
            ratio = most_expensive["total_cost"] / cheapest["total_cost"] if cheapest["total_cost"] > 0 else 1

            result["summary"] = {
                "models_compared": len(comparisons),
                "cheapest": {"model": cheapest["model"], "cost": cheapest["total_cost"]},
                "most_expensive": {"model": most_expensive["model"], "cost": most_expensive["total_cost"]},
                "cost_ratio": round(ratio, 1),
            }

    return json.dumps(result, ensure_ascii=False)


def check_cost_optimizer_requirements() -> bool:
    return True


COST_OPTIMIZER_SCHEMA = {
    "name": "cost_optimizer",
    "description": (
        "Analyze API costs and provide optimization recommendations.\n\n"
        "Compare costs across models, identify savings opportunities,\n"
        "and get recommendations for cost-effective alternatives.\n\n"
        "Uses route-aware pricing from the agent's pricing catalog.\n\n"
        "Parameters:\n"
        "- model: Specific model to analyze (omit for all models comparison)\n"
        "- input_tokens: Estimated input tokens (default 1000, must be >= 0)\n"
        "- output_tokens: Estimated output tokens (default 500, must be >= 0)\n"
        "- provider: Filter by provider"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Specific model to analyze (omit for all models)",
            },
            "input_tokens": {
                "type": "integer",
                "description": "Estimated input tokens (must be >= 0)",
                "default": 1000,
            },
            "output_tokens": {
                "type": "integer",
                "description": "Estimated output tokens (must be >= 0)",
                "default": 500,
            },
            "provider": {
                "type": "string",
                "description": "Filter by provider",
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
    },
}


from tools.registry import registry

registry.register(
    name="cost_optimizer",
    toolset="cost",
    schema=COST_OPTIMIZER_SCHEMA,
    handler=lambda args, **kw: cost_optimizer(
        model=args.get("model"),
        input_tokens=args.get("input_tokens", 1000),
        output_tokens=args.get("output_tokens", 500),
        provider=args.get("provider"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_cost_optimizer_requirements,
    emoji="📊",
)
