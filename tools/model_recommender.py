#!/usr/bin/env python3
"""
Model Recommender Tool - Suggest optimal model per task based on cost and complexity

Analyzes task type and complexity, recommends cost-effective models,
and supports auto-downgrade for simple tasks.
Note: Pricing is approximate and may change. Check provider docs for current rates.
"""

import json
from typing import Any, Dict, List, Optional, Tuple


TASK_CATEGORIES = {
    "code-gen": {"models": ["claude-sonnet-4", "gpt-4o", "deepseek-v3"], "complexity_weight": 0.8},
    "code-review": {"models": ["claude-sonnet-4", "gpt-4o", "deepseek-v3"], "complexity_weight": 0.7},
    "debug": {"models": ["claude-sonnet-4", "gpt-4o"], "complexity_weight": 0.9},
    "explain": {"models": ["gpt-4o-mini", "claude-haiku", "gemini-flash"], "complexity_weight": 0.4},
    "refactor": {"models": ["claude-sonnet-4", "gpt-4o", "deepseek-v3"], "complexity_weight": 0.7},
    "test-write": {"models": ["gpt-4o-mini", "claude-haiku", "gemini-flash"], "complexity_weight": 0.5},
    "config": {"models": ["gpt-4o-mini", "gemini-flash"], "complexity_weight": 0.3},
    "search": {"models": ["gemini-flash", "gpt-4o-mini"], "complexity_weight": 0.2},
    "summarize": {"models": ["claude-haiku", "gpt-4o-mini"], "complexity_weight": 0.3},
    "documentation": {"models": ["claude-haiku", "gpt-4o-mini", "gemini-flash"], "complexity_weight": 0.4},
}

MODEL_COST_PER_MTOK = {
    "claude-sonnet-4": {"input": 15.0, "output": 75.0},
    "claude-haiku": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 10.0, "output": 30.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
    "gemini-flash": {"input": 0.075, "output": 0.30},
    "gemini-pro": {"input": 0.50, "output": 1.50},
}

QUALITY_SCORES = {
    "claude-sonnet-4": 0.95, "gpt-4o": 0.9, "deepseek-v3": 0.85,
    "claude-haiku": 0.7, "gpt-4o-mini": 0.65, "gemini-flash": 0.6,
    "gemini-pro": 0.75,
}

SPEED_SCORES = {
    "claude-haiku": 0.9, "gpt-4o-mini": 0.8, "gemini-flash": 0.95,
    "gpt-4o": 0.6, "claude-sonnet-4": 0.5, "deepseek-v3": 0.7,
    "gemini-pro": 0.7,
}

PROVIDER_MODELS = {
    "anthropic": ["claude-sonnet-4", "claude-haiku"],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "google": ["gemini-flash", "gemini-pro"],
    "deepseek": ["deepseek-v3"],
}


def _classify_task(prompt: str) -> Tuple[str, float]:
    prompt_lower = prompt.lower()
    debug_kw = ["bug", "error", "fail", "crash", "fix", "issue", "wrong", "broken"]
    code_kw = ["write", "create", "implement", "add", "function", "class", "method"]
    review_kw = ["review", "check", "audit", "analyze", "quality"]
    test_kw = ["test", "unit test", "pytest", "coverage", "assert"]
    config_kw = ["config", "setting", "yaml", "json", "toml", "env"]
    search_kw = ["find", "search", "lookup", "query", "where"]
    doc_kw = ["document", "readme", "docs", "explain", "comment"]

    scores = {
        "debug": sum(1 for k in debug_kw if k in prompt_lower) * 3,
        "code-gen": sum(1 for k in code_kw if k in prompt_lower) * 2,
        "code-review": sum(1 for k in review_kw if k in prompt_lower) * 2,
        "test-write": sum(1 for k in test_kw if k in prompt_lower) * 2,
        "config": sum(1 for k in config_kw if k in prompt_lower),
        "search": sum(1 for k in search_kw if k in prompt_lower),
        "documentation": sum(1 for k in doc_kw if k in prompt_lower),
        "explain": 2 if len(prompt_lower.split()) > 50 else 0,
    }

    if not any(scores.values()):
        return "explain", 0.5

    best = max(scores, key=scores.get)
    complexity = min(1.0, (len(prompt_lower.split()) / 200) + (scores[best] / 10))
    return best, round(complexity, 2)


def model_recommend(
    prompt: str,
    preferred_provider: Optional[str] = None,
    max_cost_per_mtok: Optional[float] = None,
    prefer_speed: bool = False,
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    task_type, complexity = _classify_task(prompt)
    if task_type not in TASK_CATEGORIES:
        task_type = "explain"

    cat = TASK_CATEGORIES[task_type]
    candidates = list(cat["models"])

    filtered_by_provider = False
    if preferred_provider:
        allowed = PROVIDER_MODELS.get(preferred_provider)
        if allowed:
            filtered = [m for m in candidates if m in allowed]
            if filtered:
                candidates = filtered
            else:
                filtered_by_provider = True

    scored: List[Dict[str, Any]] = []
    for model in candidates:
        cost = MODEL_COST_PER_MTOK.get(model, {"input": 5.0, "output": 15.0})
        avg_cost = (cost["input"] + cost["output"]) / 2
        speed_score = SPEED_SCORES.get(model, 0.5)

        if complexity < 0.4:
            cw = 0.7 if not prefer_speed else 0.3
            sw = 0.3 if not prefer_speed else 0.7
            score = (1.0 / (avg_cost + 0.1)) * cw + speed_score * sw
        elif complexity < 0.7:
            cw = 0.4 if not prefer_speed else 0.2
            sw = 0.3 if not prefer_speed else 0.5
            qw = 0.3
            qs = QUALITY_SCORES.get(model, 0.5)
            score = (1.0 / (avg_cost + 0.1)) * cw + speed_score * sw + qs * qw
        else:
            qs = QUALITY_SCORES.get(model, 0.5)
            score = qs * 0.7 + speed_score * 0.3

        if max_cost_per_mtok and avg_cost > max_cost_per_mtok:
            continue

        scored.append({
            "model": model,
            "avg_cost_per_mtok": avg_cost,
            "speed_score": speed_score,
            "score": round(score, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    result: Dict[str, Any] = {
        "success": True,
        "task_type": task_type,
        "complexity": complexity,
        "recommendations": scored[:5],
    }

    if filtered_by_provider:
        result["note"] = f"No {preferred_provider} models for this task type. Showing all compatible models."

    if complexity < 0.4 and scored:
        cheap = [m for m in scored if m["avg_cost_per_mtok"] < 1.0]
        if cheap:
            diff = scored[0]["avg_cost_per_mtok"] - cheap[0]["avg_cost_per_mtok"]
            if diff > 5.0:
                result["auto_downgrade"] = {
                    "from": scored[0]["model"],
                    "to": cheap[0]["model"],
                    "savings_per_mtok": round(diff, 2),
                    "reason": f"Task complexity is low ({complexity}). Cheaper model sufficient.",
                }

    if complexity < 0.4:
        result["tip"] = "Use gpt-4o-mini or claude-haiku for simple tasks to save 10-50x cost."

    return json.dumps(result, ensure_ascii=False)


def check_model_recommend_requirements() -> bool:
    return True


MODEL_RECOMMEND_SCHEMA = {
    "name": "model_recommend",
    "description": (
        "Recommend the most cost-effective model for a given task.\n\n"
        "Analyzes task type (code-gen, debug, explain, refactor, test, config, search, docs)\n"
        "and complexity to suggest the optimal model balancing cost, speed, and quality.\n"
        "Supports auto-downgrade suggestions when expensive models are unnecessary.\n\n"
        "Note: Pricing is approximate and may change. Check provider docs for current rates.\n\n"
        "Parameters:\n"
        "- prompt: Task prompt to analyze\n"
        "- preferred_provider: Filter by provider\n"
        "- max_cost_per_mtok: Maximum acceptable cost per million tokens\n"
        "- prefer_speed: Prefer faster models over cheaper ones"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Task prompt to analyze"},
            "preferred_provider": {
                "type": "string", "description": "Preferred provider",
                "enum": ["openrouter", "anthropic", "openai", "google", "deepseek"],
            },
            "max_cost_per_mtok": {"type": "number", "description": "Max cost per million tokens"},
            "prefer_speed": {"type": "boolean", "description": "Prefer faster models", "default": False},
            "task_id": {"type": "string", "description": "Optional task ID"},
        },
        "required": ["prompt"],
    },
}


from tools.registry import registry

registry.register(
    name="model_recommend",
    toolset="cost",
    schema=MODEL_RECOMMEND_SCHEMA,
    handler=lambda args, **kw: model_recommend(
        prompt=args.get("prompt", ""),
        preferred_provider=args.get("preferred_provider"),
        max_cost_per_mtok=args.get("max_cost_per_mtok"),
        prefer_speed=args.get("prefer_speed", False),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_model_recommend_requirements,
    emoji="💰",
)