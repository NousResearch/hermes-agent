#!/usr/bin/env python3
"""
Delegation Cost Profiles

Provides preset profiles (minimal, balanced, full) that override
delegation config for cost optimization. Backward compatible:
if no `profile` key is present, behavior is unchanged.
"""

from typing import Any, Dict, Optional

# Schema trim level constants
SCHEMA_TRIM_NONE = "none"
SCHEMA_TRIM_MODERATE = "moderate"
SCHEMA_TRIM_AGGRESSIVE = "aggressive"

# System prompt level constants
PROMPT_FULL = "full"
PROMPT_ESSENTIAL = "essential"
PROMPT_GOAL_ONLY = "goal_only"

# Profile definitions
PROFILES: Dict[str, Dict[str, Any]] = {
    "minimal": {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "max_iterations": 20,
        "schema_trim": SCHEMA_TRIM_AGGRESSIVE,
        "system_prompt": PROMPT_GOAL_ONLY,
    },
    "balanced": {
        "model": "deepseek-chat",
        "provider": "deepseek",
        "max_iterations": 35,
        "schema_trim": SCHEMA_TRIM_MODERATE,
        "system_prompt": PROMPT_ESSENTIAL,
    },
    "full": {
        "model": None,  # use config value
        "provider": None,
        "max_iterations": 50,
        "schema_trim": SCHEMA_TRIM_NONE,
        "system_prompt": PROMPT_FULL,
    },
}


def get_available_profiles() -> Dict[str, Dict[str, Any]]:
    """Return a copy of all defined profiles."""
    return {k: v.copy() for k, v in PROFILES.items()}


def apply_profile(profile_name: str, delegation_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a named profile's overrides to the delegation config.

    Returns a new dict with profile values merged in.
    Unknown profile names return the original config unchanged.
    """
    if profile_name not in PROFILES:
        return delegation_config.copy()

    profile = PROFILES[profile_name]
    result = delegation_config.copy()

    for key in ("model", "provider", "max_iterations", "schema_trim", "system_prompt"):
        if profile.get(key) is not None:
            result[key] = profile[key]

    return result


def get_profile_cost_estimate(profile_name: str, avg_turns: int = 3) -> float:
    """
    Rough cost estimate per subagent (USD) for the given profile.

    Uses per-1M-token pricing from DESIGN.md and simplified token model.
    """
    if profile_name not in PROFILES:
        return 0.0

    # Approximate tokens for a 3-turn subagent (from DESIGN breakdown)
    base_input = 12568
    base_output = 1100

    # Schema trim savings (rough)
    trim = PROFILES[profile_name].get("schema_trim", SCHEMA_TRIM_NONE)
    if trim == SCHEMA_TRIM_AGGRESSIVE:
        input_tokens = int(base_input * 0.4)
    elif trim == SCHEMA_TRIM_MODERATE:
        input_tokens = int(base_input * 0.6)
    else:
        input_tokens = base_input

    output_tokens = base_output

    # Pricing per 1M tokens (input, output)
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "deepseek-chat": (0.27, 1.10),
        # full uses parent's model (assume Grok)
        None: (2.00, 10.00),
    }

    model = PROFILES[profile_name].get("model") or "grok-4-latest"
    if "gpt-4o-mini" in model:
        pin, pout = pricing["gpt-4o-mini"]
    elif "deepseek" in model:
        pin, pout = pricing["deepseek-chat"]
    else:
        pin, pout = pricing[None]

    cost = (input_tokens * pin + output_tokens * pout) / 1_000_000
    # Scale by actual turns if different from default
    return round(cost * (avg_turns / 3), 6)