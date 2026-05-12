#!/usr/bin/env python3
"""
Cost Optimizer Tool - Analyze and optimize API costs

Provides cost analysis, provider cost comparison, and optimization recommendations.
"""

import json
from typing import Any, Dict, List, Optional


MODEL_RATES = {
    "claude-sonnet-4": {"input": 15.0, "output": 75.0, "provider": "anthropic"},
    "claude-haiku": {"input": 0.80, "output": 4.0, "provider": "anthropic"},
    "gpt-4o": {"input": 10.0, "output": 30.0, "provider": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "openai"},
    "deepseek-v3": {"input": 0.27, "output": 1.10, "provider": "deepseek"},
    "gemini-flash": {"input": 0.075, "output": 0.30, "provider": "google"},
    "gemini-pro": {"input": 0.50, "output": 1.50, "provider": "google"},
}

PROVIDER_MULTIPLIERS = {
    "openrouter": 1.15,
    "anthropic": 1.0,
    "openai": 1.0,
    "google": 1.0,
    "deepseek": 1.0,
}


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
        input_tokens: Estimated input tokens (default 1000)
        output_tokens: Estimated output tokens (default 500)
        provider: Compare costs across specific provider

    Returns:
        JSON string with cost analysis and optimization recommendations
    """
    result: Dict[str, Any] = {"success": True}

    if model:
        if model not in MODEL_RATES:
            return json.dumps({
                "success": False,
                "error": f"Unknown model: {model}. Known: {', '.join(MODEL_RATES.keys())}",
            })
        rates = MODEL_RATES[model]
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        total_cost = input_cost + output_cost

        result["analysis"] = {
            "model": model,
            "provider": rates["provider"],
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

        if provider:
            provider_cost = total_cost * PROVIDER_MULTIPLIERS.get(provider, 1.15)
            result["analysis"]["provider_multiplier"] = PROVIDER_MULTIPLIERS.get(provider, 1.15)
            result["analysis"]["costs"]["via_provider"] = round(provider_cost, 6)

        alt_models = []
        for alt_name, alt_rates in MODEL_RATES.items():
            if alt_name == model:
                continue
            alt_input = (input_tokens / 1_000_000) * alt_rates["input"]
            alt_output = (output_tokens / 1_000_000) * alt_rates["output"]
            alt_total = alt_input + alt_output
            savings = total_cost - alt_total

            if savings > 0.001:
                alt_models.append({
                    "model": alt_name,
                    "provider": alt_rates["provider"],
                    "total_cost": round(alt_total, 6),
                    "savings": round(savings, 6),
                    "savings_pct": round((savings / total_cost) * 100, 1),
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
        for m_name, m_rates in MODEL_RATES.items():
            input_cost = (input_tokens / 1_000_000) * m_rates["input"]
            output_cost = (output_tokens / 1_000_000) * m_rates["output"]
            total = input_cost + output_cost
            comparisons.append({
                "model": m_name,
                "provider": m_rates["provider"],
                "total_cost": round(total, 6),
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
            })

        comparisons.sort(key=lambda x: x["total_cost"])
        result["comparison"] = comparisons

        cheapest = comparisons[0]
        most_expensive = comparisons[-1]
        ratio = most_expensive["total_cost"] / cheapest["total_cost"] if cheapest["total_cost"] > 0 else 1

        result["summary"] = {
            "models_compared": len(comparisons),
            "cheapest": {"model": cheapest["model"], "cost": cheapest["total_cost"]},
            "most_expensive": {"model": most_expensive["model"], "cost": most_expensive["total_cost"]},
            "cost_ratio": round(ratio, 1),
            "estimated_savings": {
                "switching_from_expensive": f"Switch from {most_expensive['model']} to {cheapest['model']} saves {round((most_expensive['total_cost'] - cheapest['total_cost']), 4)} per request",
                "tip": "Use gemini-flash or gpt-4o-mini for simple tasks to reduce costs by 10-100x",
            },
        }

        per_provider: Dict[str, list] = {}
        for c in comparisons:
            p = c["provider"]
            if p not in per_provider:
                per_provider[p] = []
            per_provider[p].append(c)

        if provider and provider in per_provider:
            result["provider_filter"] = {
                "provider": provider,
                "models": per_provider[provider],
                "cheapest_for_provider": min(per_provider[provider], key=lambda x: x["total_cost"]),
            }

    return json.dumps(result, ensure_ascii=False)


def check_cost_optimizer_requirements() -> bool:
    return True


COST_OPTIMIZER_SCHEMA = {
    "name": "cost_optimizer",
    "description": (
        "Analyze API costs and provide optimization recommendations.\n\n"
        "Compare costs across models and providers, identify savings opportunities,\n"
        "and get recommendations for cost-effective alternatives.\n\n"
        "Note: Pricing is approximate and may change. Check provider docs for current rates.\n\n"
        "Parameters:\n"
        "- model: Specific model to analyze (omit for all models comparison)\n"
        "- input_tokens: Estimated input tokens (default 1000)\n"
        "- output_tokens: Estimated output tokens (default 500)\n"
        "- provider: Filter by provider (anthropic, openai, google, deepseek)"
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
                "description": "Estimated input tokens",
                "default": 1000,
            },
            "output_tokens": {
                "type": "integer",
                "description": "Estimated output tokens",
                "default": 500,
            },
            "provider": {
                "type": "string",
                "description": "Filter by provider (anthropic, openai, google, deepseek)",
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