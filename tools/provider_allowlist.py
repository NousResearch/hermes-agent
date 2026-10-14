#!/usr/bin/env python3
"""
Provider Allowlist Guard — runtime enforcement of the banned provider policy.

Intercepts model/provider configuration at startup and rejects anything
not on the explicit allowlist. Prevents accidental Anthropic/OpenAI
fallback from ever reaching inference.

Usage:
    Import and call validate_provider() at agent startup:
    
    from provider_allowlist import validate_provider, ProviderBanned
    
    try:
        validate_provider(config["provider"], config["model"])
    except ProviderBanned as e:
        logger.error(f"Banned provider: {e}")
        sys.exit(1)

The allowlist is intentionally conservative. Add new providers here
only after human review and a merged PR.
"""
import re
import os

class ProviderBanned(Exception):
    """Raised when a banned provider is detected."""
    pass

# ═══ ALLOWLIST ═══
# Only these provider/model combinations are permitted.
# Everything else is rejected by default.
ALLOWED_PROVIDERS = {
    "ollama": {
        "description": "Local inference via Ollama",
        "model_patterns": [
            r"^hermes.*",        # All Hermes variants
            r"^gemma.*",         # Gemma family
            r"^qwen.*",         # Qwen family
            r"^llama.*",        # Llama family
            r"^mistral.*",      # Mistral family
            r"^codestral.*",    # Codestral
            r"^deepseek.*",     # DeepSeek family
            r"^phi.*",          # Phi family
            r"^nomic-embed.*",  # Embedding models
        ]
    },
    "openrouter": {
        "description": "OpenRouter relay (temporary falsework)",
        "model_patterns": [
            r"^google/.*",       # Google models via relay
            r"^meta-llama/.*",   # Meta models via relay
            r"^mistralai/.*",    # Mistral via relay
            r"^qwen/.*",        # Qwen via relay
            r"^deepseek/.*",    # DeepSeek via relay
            r"^nousresearch/.*", # NousResearch (Hermes) via relay
        ]
    },
    "kimi": {
        "description": "Kimi/Moonshot (wizard-specific)",
        "model_patterns": [
            r"^kimi.*",
            r"^moonshot.*",
        ]
    },
    "gemini": {
        "description": "Google Gemini direct (temporary falsework)",
        "model_patterns": [
            r"^gemini.*",
        ]
    },
}

# ═══ BANLIST ═══
# These are permanently banned. No exceptions.
BANNED_PROVIDERS = {
    "anthropic": "Permanently banned — hard policy since 2026-04-09",
    "claude": "Alias for Anthropic — permanently banned",
    "openai": "Not sovereign — use local models or OpenRouter relay",
}

BANNED_MODEL_PATTERNS = [
    r"^claude.*",          # Any Claude model
    r"^gpt-4.*",          # GPT-4 variants
    r"^gpt-3.*",          # GPT-3 variants  
    r"^o1.*",             # O1 variants
    r"^o3.*",             # O3 variants
]


def validate_provider(provider: str, model: str = "") -> bool:
    """
    Validate that a provider/model combination is allowed.
    
    Args:
        provider: The provider name (e.g., "ollama", "anthropic")
        model: The model name (e.g., "hermes3:latest", "claude-3-opus")
    
    Returns:
        True if allowed
        
    Raises:
        ProviderBanned: If the provider or model is banned
    """
    provider_lower = provider.lower().strip()
    model_lower = model.lower().strip()
    
    # Check banlist first
    if provider_lower in BANNED_PROVIDERS:
        raise ProviderBanned(
            f"Provider '{provider}' is permanently banned: {BANNED_PROVIDERS[provider_lower]}"
        )
    
    # Check model against banned patterns
    for pattern in BANNED_MODEL_PATTERNS:
        if re.match(pattern, model_lower):
            raise ProviderBanned(
                f"Model '{model}' matches banned pattern '{pattern}'"
            )
    
    # Check allowlist
    if provider_lower not in ALLOWED_PROVIDERS:
        raise ProviderBanned(
            f"Provider '{provider}' is not on the allowlist. "
            f"Allowed: {', '.join(sorted(ALLOWED_PROVIDERS.keys()))}"
        )
    
    # If model specified, validate against provider's allowed patterns
    if model_lower:
        allowed = ALLOWED_PROVIDERS[provider_lower]
        for pattern in allowed["model_patterns"]:
            if re.match(pattern, model_lower):
                return True
        
        raise ProviderBanned(
            f"Model '{model}' is not allowed for provider '{provider}'. "
            f"Allowed patterns: {allowed['model_patterns']}"
        )
    
    return True


def scan_config(config: dict) -> list:
    """
    Scan an entire config dict for banned provider references.
    Returns a list of violations.
    """
    violations = []
    
    def _scan(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            val = obj.lower()
            for banned in BANNED_PROVIDERS:
                if banned in val:
                    violations.append(f"{path}: contains banned provider '{banned}' in value '{obj}'")
            for pattern in BANNED_MODEL_PATTERNS:
                if re.match(pattern, val):
                    violations.append(f"{path}: matches banned model pattern '{pattern}' with value '{obj}'")
    
    _scan(config)
    return violations


if __name__ == "__main__":
    # Self-test
    import sys
    
    tests = [
        ("ollama", "hermes3:latest", True),
        ("ollama", "gemma4:latest", True),
        ("anthropic", "claude-3-opus", False),
        ("claude", "", False),
        ("openai", "gpt-4", False),
        ("openrouter", "google/gemini-2.5-pro", True),
        ("openrouter", "anthropic/claude-3", False),
        ("kimi", "kimi-k2.5", True),
        ("unknown_provider", "", False),
    ]
    
    passed = 0
    for provider, model, should_pass in tests:
        try:
            validate_provider(provider, model)
            if should_pass:
                passed += 1
                print(f"  ✓ {provider}/{model} — allowed (expected)")
            else:
                print(f"  ✗ {provider}/{model} — allowed (SHOULD HAVE BEEN BLOCKED)")
        except ProviderBanned as e:
            if not should_pass:
                passed += 1
                print(f"  ✓ {provider}/{model} — blocked: {e}")
            else:
                print(f"  ✗ {provider}/{model} — blocked (SHOULD HAVE BEEN ALLOWED): {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
