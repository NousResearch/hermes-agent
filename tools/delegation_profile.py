"""
Delegation Profile - config knobs only (no model/provider routing per review).
"""

PROFILES = {
    "minimal": {
        "schema_trim": "aggressive",
        "system_prompt": "minimal",
        "max_iterations": 3,
    },
    "balanced": {
        "schema_trim": "moderate",
        "system_prompt": "balanced",
        "max_iterations": 8,
    },
    "full": {
        "schema_trim": "none",
        "system_prompt": "full",
        "max_iterations": 20,
    },
}

def apply_profile(profile_name: str, config: dict) -> dict:
    if profile_name not in PROFILES:
        return config
    profile = PROFILES[profile_name]
    result = config.copy()
    for key in ["schema_trim", "system_prompt", "max_iterations"]:
        if key in profile:
            result[key] = profile[key]
    return result

def get_profile_cost_estimate(profile_name: str) -> dict:
    # No model pricing - removed per review
    if profile_name not in PROFILES:
        return {"error": "unknown profile"}
    return {
        "profile": profile_name,
        "note": "cost depends on explicit delegation.provider/model + usage"
    }

def get_available_profiles():
    return list(PROFILES.keys())
