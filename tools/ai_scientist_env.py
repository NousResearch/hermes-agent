"""Bridge Hermes-resolved credentials into AI-Scientist subprocesses and harness runner.

Sakana upstream ``create_client(model)`` reads provider API keys from ``os.environ``
(``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, etc.). This module
loads ``~/.hermes/.env`` and Hermes OAuth stores, routes free-tier / OAuth providers
in priority order (Codex, Nous, NVIDIA, Groq, xAI, …), and maps them onto env vars
without introducing new user-facing ``HERMES_*`` config keys.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

ModelFamily = str

# Free-tier / OAuth friendly order when the requested model family has no direct key.
DEFAULT_PROVIDER_PRIORITY: tuple[str, ...] = (
    "openai-codex",
    "nous",
    "nvidia",
    "groq",
    "xai-oauth",
    "openrouter",
    "gemini",
    "anthropic",
    "deepseek",
    "ollama",
)

DEFAULT_CODEX_BASE = "https://chatgpt.com/backend-api/codex"
DEFAULT_NOUS_BASE = "https://inference-api.nousresearch.com/v1"
DEFAULT_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_XAI_BASE = "https://api.x.ai/v1"
DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Sakana ``AVAILABLE_LLMS`` names used as CLI/aider aliases for OpenAI-compatible shims.
SAKANA_OPENAI_SHIM_MODEL = "gpt-4o-mini"
SAKANA_GEMINI_MODEL = "gemini-2.0-flash-lite"
SAKANA_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"


@dataclass(frozen=True)
class ResolvedAiScientistLLM:
    provider_id: str
    sakana_model: str
    api_model: str
    api_key: str
    base_url: str
    source: str
    routing: str  # "native" | "openai_shim" | "ollama"


def infer_model_family(model: str | None) -> ModelFamily:
    """Classify a model string the way Sakana ``create_client`` does."""
    raw = (model or "").strip()
    if raw.lower() in {"", "auto"}:
        return "auto"
    if raw.startswith(("ollama/", "openai-compatible/")):
        return "ollama"
    lower = raw.lower()
    if lower.startswith("claude-"):
        return "anthropic"
    if lower.startswith("bedrock") and "claude" in lower:
        return "bedrock"
    if lower.startswith("vertex_ai") and "claude" in lower:
        return "vertex"
    if "gpt" in lower or "o1" in lower or "o3" in lower:
        return "openai"
    if lower in {"deepseek-chat", "deepseek-reasoner", "deepseek-coder"}:
        return "deepseek"
    if lower == "llama3.1-405b":
        return "openrouter"
    if "gemini" in lower:
        return "gemini"
    return "unknown"


def _ensure_hermes_env_loaded() -> None:
    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv()
    except Exception:
        logger.debug("Could not load Hermes dotenv", exc_info=True)


def _get_env_secret(name: str) -> str:
    try:
        from hermes_cli.config import get_env_value

        return (get_env_value(name) or "").strip()
    except Exception:
        return (os.environ.get(name) or "").strip()


def _read_ai_scientist_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        aux = cfg.get("auxiliary") if isinstance(cfg.get("auxiliary"), dict) else {}
        section = aux.get("ai_scientist") if isinstance(aux.get("ai_scientist"), dict) else {}
        return section
    except Exception:
        return {}


def _provider_priority() -> tuple[str, ...]:
    section = _read_ai_scientist_config()
    raw = section.get("provider_priority")
    if isinstance(raw, list) and raw:
        cleaned = [str(item).strip().lower() for item in raw if str(item).strip()]
        if cleaned:
            return tuple(cleaned)
    override = (section.get("provider") or "").strip().lower()
    if override:
        return (override, *DEFAULT_PROVIDER_PRIORITY)
    return DEFAULT_PROVIDER_PRIORITY


def _openai_shim_overlay(
    *,
    provider_id: str,
    api_key: str,
    base_url: str,
    api_model: str,
    sakana_model: str = SAKANA_OPENAI_SHIM_MODEL,
) -> dict[str, str]:
    base = base_url.strip().rstrip("/")
    return {
        "AI_SCIENTIST_HERMES_BRIDGE": "1",
        "AI_SCIENTIST_FORCE_OPENAI_SHIM": "1",
        "AI_SCIENTIST_API_MODEL": api_model,
        "AI_SCIENTIST_HERMES_PROVIDER": provider_id,
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base,
        "OPENAI_API_BASE": base,
    }


def _try_openai_codex() -> ResolvedAiScientistLLM | None:
    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials

        creds = resolve_codex_runtime_credentials()
    except Exception:
        return None
    api_key = (creds.get("api_key") or "").strip()
    if not api_key:
        return None
    base_url = (creds.get("base_url") or DEFAULT_CODEX_BASE).strip().rstrip("/")
    api_model = (creds.get("model") or SAKANA_OPENAI_SHIM_MODEL).strip()
    return ResolvedAiScientistLLM(
        provider_id="openai-codex",
        sakana_model=SAKANA_OPENAI_SHIM_MODEL,
        api_model=api_model,
        api_key=api_key,
        base_url=base_url,
        source=str(creds.get("source") or "codex_auth"),
        routing="openai_shim",
    )


def _try_nous() -> ResolvedAiScientistLLM | None:
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
    except Exception:
        return None
    api_key = (creds.get("api_key") or "").strip()
    if not api_key:
        return None
    base_url = (creds.get("base_url") or DEFAULT_NOUS_BASE).strip().rstrip("/")
    api_model = (creds.get("model") or "").strip()
    if not api_model:
        try:
            from hermes_cli.models import get_nous_recommended_aux_model

            api_model = (get_nous_recommended_aux_model() or "").strip()
        except Exception:
            api_model = ""
    if not api_model:
        api_model = "DeepHermes-3-Llama-3-8B-Preview"
    return ResolvedAiScientistLLM(
        provider_id="nous",
        sakana_model=SAKANA_OPENAI_SHIM_MODEL,
        api_model=api_model,
        api_key=api_key,
        base_url=base_url,
        source=str(creds.get("source") or "nous_auth"),
        routing="openai_shim",
    )


def _try_nvidia() -> ResolvedAiScientistLLM | None:
    api_key = _get_env_secret("NVIDIA_API_KEY")
    if not api_key:
        return None
    section = _read_ai_scientist_config()
    api_model = (section.get("nvidia_model") or "meta/llama-3.1-70b-instruct").strip()
    base_url = (section.get("nvidia_base_url") or DEFAULT_NVIDIA_BASE).strip().rstrip("/")
    return ResolvedAiScientistLLM(
        provider_id="nvidia",
        sakana_model=SAKANA_OPENAI_SHIM_MODEL,
        api_model=api_model,
        api_key=api_key,
        base_url=base_url,
        source="nvidia_api_key",
        routing="openai_shim",
    )


def _try_groq() -> ResolvedAiScientistLLM | None:
    api_key = _get_env_secret("GROQ_API_KEY")
    if not api_key:
        return None
    section = _read_ai_scientist_config()
    api_model = (section.get("groq_model") or "llama-3.1-8b-instant").strip()
    return ResolvedAiScientistLLM(
        provider_id="groq",
        sakana_model=SAKANA_OPENAI_SHIM_MODEL,
        api_model=api_model,
        api_key=api_key,
        base_url=DEFAULT_GROQ_BASE,
        source="groq_api_key",
        routing="openai_shim",
    )


def _try_xai_oauth() -> ResolvedAiScientistLLM | None:
    try:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        creds = resolve_xai_oauth_runtime_credentials()
    except Exception:
        return None
    api_key = (creds.get("api_key") or "").strip()
    if not api_key:
        return None
    base_url = (creds.get("base_url") or DEFAULT_XAI_BASE).strip().rstrip("/")
    api_model = (creds.get("model") or "grok-3-mini").strip()
    return ResolvedAiScientistLLM(
        provider_id="xai-oauth",
        sakana_model=SAKANA_OPENAI_SHIM_MODEL,
        api_model=api_model,
        api_key=api_key,
        base_url=base_url,
        source=str(creds.get("source") or "xai_oauth"),
        routing="openai_shim",
    )


def _try_openrouter() -> ResolvedAiScientistLLM | None:
    api_key = _get_env_secret("OPENROUTER_API_KEY")
    if not api_key:
        try:
            from hermes_cli.auth import resolve_api_key_provider_credentials

            creds = resolve_api_key_provider_credentials("openrouter")
            api_key = (creds.get("api_key") or "").strip()
        except Exception:
            api_key = ""
    if not api_key:
        return None
    section = _read_ai_scientist_config()
    api_model = (section.get("openrouter_model") or "meta-llama/llama-3.1-405b-instruct").strip()
    return ResolvedAiScientistLLM(
        provider_id="openrouter",
        sakana_model="llama3.1-405b",
        api_model=api_model,
        api_key=api_key,
        base_url=DEFAULT_OPENROUTER_BASE,
        source="openrouter_api_key",
        routing="native",
    )


def _try_gemini() -> ResolvedAiScientistLLM | None:
    api_key = _get_env_secret("GEMINI_API_KEY") or _get_env_secret("GOOGLE_API_KEY")
    if not api_key:
        return None
    section = _read_ai_scientist_config()
    api_model = (section.get("gemini_model") or SAKANA_GEMINI_MODEL).strip()
    return ResolvedAiScientistLLM(
        provider_id="gemini",
        sakana_model=api_model,
        api_model=api_model,
        api_key=api_key,
        base_url=DEFAULT_GEMINI_OPENAI_BASE,
        source="gemini_api_key",
        routing="native",
    )


def _try_anthropic() -> ResolvedAiScientistLLM | None:
    api_key = _get_env_secret("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            from agent.anthropic_adapter import resolve_anthropic_token

            api_key = (resolve_anthropic_token() or "").strip()
        except Exception:
            api_key = ""
    if not api_key:
        return None
    section = _read_ai_scientist_config()
    api_model = (section.get("anthropic_model") or SAKANA_ANTHROPIC_MODEL).strip()
    return ResolvedAiScientistLLM(
        provider_id="anthropic",
        sakana_model=api_model,
        api_model=api_model,
        api_key=api_key,
        base_url="",
        source="anthropic_oauth_or_key",
        routing="native",
    )


def _try_deepseek() -> ResolvedAiScientistLLM | None:
    api_key = _get_env_secret("DEEPSEEK_API_KEY")
    if not api_key:
        try:
            from hermes_cli.auth import resolve_api_key_provider_credentials

            creds = resolve_api_key_provider_credentials("deepseek")
            api_key = (creds.get("api_key") or "").strip()
        except Exception:
            api_key = ""
    if not api_key:
        return None
    api_model = "deepseek-chat"
    return ResolvedAiScientistLLM(
        provider_id="deepseek",
        sakana_model=api_model,
        api_model=api_model,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        source="deepseek_api_key",
        routing="native",
    )


def _try_ollama() -> ResolvedAiScientistLLM | None:
    section = _read_ai_scientist_config()
    base_raw = _get_env_secret("OLLAMA_BASE_URL")
    configured_model = _get_env_secret("OLLAMA_MODEL")
    allow_fallback = bool(section.get("allow_ollama_fallback"))
    if not base_raw and not configured_model and not allow_fallback:
        return None
    base_url = (base_raw or "http://localhost:11434/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    api_model = (section.get("ollama_model") or configured_model or "qwen-hakua-core:latest").strip()
    sakana_model = f"openai-compatible/{api_model}"
    return ResolvedAiScientistLLM(
        provider_id="ollama",
        sakana_model=sakana_model,
        api_model=api_model,
        api_key=_get_env_secret("OPENAI_API_KEY") or "ollama",
        base_url=base_url,
        source="ollama_local",
        routing="ollama",
    )


_PROVIDER_RESOLVERS = {
    "openai-codex": _try_openai_codex,
    "codex": _try_openai_codex,
    "gpt": _try_openai_codex,
    "nous": _try_nous,
    "nvidia": _try_nvidia,
    "groq": _try_groq,
    "xai-oauth": _try_xai_oauth,
    "xai": _try_xai_oauth,
    "openrouter": _try_openrouter,
    "gemini": _try_gemini,
    "anthropic": _try_anthropic,
    "deepseek": _try_deepseek,
    "ollama": _try_ollama,
}


def resolved_to_overlay(resolved: ResolvedAiScientistLLM) -> dict[str, str]:
    if resolved.routing == "openai_shim":
        return _openai_shim_overlay(
            provider_id=resolved.provider_id,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            api_model=resolved.api_model,
            sakana_model=resolved.sakana_model,
        )
    if resolved.routing == "ollama":
        return {
            "OPENAI_API_KEY": resolved.api_key,
            "OPENAI_BASE_URL": resolved.base_url,
            "OPENAI_API_BASE": resolved.base_url,
            "OLLAMA_BASE_URL": resolved.base_url,
            "AI_SCIENTIST_HERMES_PROVIDER": resolved.provider_id,
        }
    if resolved.provider_id == "anthropic":
        return {"ANTHROPIC_API_KEY": resolved.api_key, "AI_SCIENTIST_HERMES_PROVIDER": resolved.provider_id}
    if resolved.provider_id == "gemini":
        return {
            "GEMINI_API_KEY": resolved.api_key,
            "AI_SCIENTIST_HERMES_PROVIDER": resolved.provider_id,
        }
    if resolved.provider_id == "deepseek":
        return {"DEEPSEEK_API_KEY": resolved.api_key, "AI_SCIENTIST_HERMES_PROVIDER": resolved.provider_id}
    if resolved.provider_id == "openrouter":
        return {
            "OPENROUTER_API_KEY": resolved.api_key,
            "AI_SCIENTIST_HERMES_PROVIDER": resolved.provider_id,
        }
    return {}


def resolve_ai_scientist_llm(model: str | None = None) -> ResolvedAiScientistLLM | None:
    """Pick the first Hermes provider with credentials (free-tier / OAuth friendly)."""
    _ensure_hermes_env_loaded()
    section = _read_ai_scientist_config()
    explicit_model = (section.get("model") or "").strip()
    requested = (model or "").strip()
    if requested.lower() in {"", "auto"}:
        requested = ""

    family = infer_model_family(requested or explicit_model or None)
    family_to_provider = {
        "openai": "openai-codex",
        "anthropic": "anthropic",
        "gemini": "gemini",
        "deepseek": "deepseek",
        "openrouter": "openrouter",
        "ollama": "ollama",
    }
    if family in family_to_provider:
        resolver = _PROVIDER_RESOLVERS.get(family_to_provider[family])
        if resolver is not None:
            try:
                resolved = resolver()
            except Exception:
                resolved = None
            if resolved is not None:
                if explicit_model and family not in {"ollama"}:
                    return ResolvedAiScientistLLM(
                        provider_id=resolved.provider_id,
                        sakana_model=requested or explicit_model or resolved.sakana_model,
                        api_model=explicit_model or resolved.api_model,
                        api_key=resolved.api_key,
                        base_url=resolved.base_url,
                        source=resolved.source,
                        routing=resolved.routing,
                    )
                if requested and family != "auto":
                    return ResolvedAiScientistLLM(
                        provider_id=resolved.provider_id,
                        sakana_model=requested,
                        api_model=resolved.api_model,
                        api_key=resolved.api_key,
                        base_url=resolved.base_url,
                        source=resolved.source,
                        routing=resolved.routing,
                    )
                return resolved

    for provider_id in _provider_priority():
        resolver = _PROVIDER_RESOLVERS.get(provider_id)
        if resolver is None:
            continue
        try:
            resolved = resolver()
        except Exception:
            resolved = None
        if resolved is None:
            continue
        if explicit_model:
            return ResolvedAiScientistLLM(
                provider_id=resolved.provider_id,
                sakana_model=requested or explicit_model or resolved.sakana_model,
                api_model=explicit_model,
                api_key=resolved.api_key,
                base_url=resolved.base_url,
                source=resolved.source,
                routing=resolved.routing,
            )
        return resolved
    return None


def resolve_ai_scientist_run_config(model: str | None = None) -> dict[str, Any]:
    """Full launch config: Sakana CLI model + env overlay for subprocess / runner."""
    resolved = resolve_ai_scientist_llm(model)
    if resolved is None:
        overlay = resolve_ai_scientist_credential_overlay(model)
        sakana_model = (model or "").strip() or SAKANA_OPENAI_SHIM_MODEL
        return {
            "sakana_model": sakana_model,
            "overlay": overlay,
            "provider_id": None,
            "source": None,
            "routing": None,
            "has_credentials": bool(overlay),
        }

    overlay = resolved_to_overlay(resolved)
    return {
        "sakana_model": resolved.sakana_model,
        "api_model": resolved.api_model,
        "overlay": overlay,
        "provider_id": resolved.provider_id,
        "source": resolved.source,
        "routing": resolved.routing,
        "has_credentials": True,
    }


def apply_ai_scientist_run_config(model: str | None = None) -> dict[str, Any]:
    """Apply resolved credentials + routing metadata to the current process."""
    config = resolve_ai_scientist_run_config(model)
    os.environ.update(config.get("overlay") or {})
    return config


def _resolve_anthropic_overlay() -> dict[str, str]:
    resolved = _try_anthropic()
    return resolved_to_overlay(resolved) if resolved else {}


def _resolve_openai_overlay() -> dict[str, str]:
    overlay: dict[str, str] = {}
    api_key = _get_env_secret("OPENAI_API_KEY")
    base_url = _get_env_secret("OPENAI_BASE_URL")

    if not api_key:
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials()
            api_key = (creds.get("api_key") or "").strip()
            if not base_url:
                base_url = (creds.get("base_url") or "").strip().rstrip("/")
        except Exception:
            logger.debug("OpenAI/Codex credential resolution failed", exc_info=True)

    if api_key:
        overlay["OPENAI_API_KEY"] = api_key
    if base_url:
        overlay["OPENAI_BASE_URL"] = base_url.rstrip("/")
        overlay["OPENAI_API_BASE"] = base_url.rstrip("/")
    return overlay


def _resolve_deepseek_overlay() -> dict[str, str]:
    resolved = _try_deepseek()
    return resolved_to_overlay(resolved) if resolved else {}


def _resolve_openrouter_overlay() -> dict[str, str]:
    resolved = _try_openrouter()
    return resolved_to_overlay(resolved) if resolved else {}


def _resolve_gemini_overlay() -> dict[str, str]:
    resolved = _try_gemini()
    return resolved_to_overlay(resolved) if resolved else {}


_FAMILY_RESOLVERS = {
    "anthropic": _resolve_anthropic_overlay,
    "openai": _resolve_openai_overlay,
    "deepseek": _resolve_deepseek_overlay,
    "openrouter": _resolve_openrouter_overlay,
    "gemini": _resolve_gemini_overlay,
}


def resolve_ai_scientist_credential_overlay(model: str | None = None) -> dict[str, str]:
    """Return env vars to inject for the given model (or routed fallback)."""
    _ensure_hermes_env_loaded()
    overlay: dict[str, str] = {}

    family = infer_model_family(model)
    if family == "ollama":
        resolved = _try_ollama()
        return resolved_to_overlay(resolved) if resolved else {}
    if family in _FAMILY_RESOLVERS:
        overlay.update(_FAMILY_RESOLVERS[family]())
        if overlay:
            return overlay

    if family in {"bedrock", "vertex"}:
        return overlay

    routed = resolve_ai_scientist_llm(model)
    if routed is not None:
        return resolved_to_overlay(routed)

    if model and infer_model_family(model) not in {"auto", "unknown"}:
        return overlay

    for resolver in _FAMILY_RESOLVERS.values():
        for key, value in resolver().items():
            overlay.setdefault(key, value)
    if overlay:
        return overlay

    routed = resolve_ai_scientist_llm(None)
    return resolved_to_overlay(routed) if routed else {}


def build_ai_scientist_env(
    *,
    base: dict[str, str] | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Merge Hermes credential overlay onto a child-process environment."""
    config = resolve_ai_scientist_run_config(model)
    env = dict(base if base is not None else os.environ)
    env.update(config.get("overlay") or {})
    return env


def apply_ai_scientist_credentials(model: str | None = None) -> dict[str, str]:
    """Apply Hermes credentials to the current process environment (in-process runner)."""
    config = apply_ai_scientist_run_config(model)
    return config.get("overlay") or {}


def describe_credential_resolution(model: str | None = None) -> dict[str, Any]:
    """Non-secret summary for diagnostics and tests."""
    config = resolve_ai_scientist_run_config(model)
    overlay = config.get("overlay") or {}
    return {
        "model": model,
        "sakana_model": config.get("sakana_model"),
        "family": infer_model_family(config.get("sakana_model")),
        "provider_id": config.get("provider_id"),
        "routing": config.get("routing"),
        "source": config.get("source"),
        "env_keys": sorted(overlay.keys()),
        "has_credentials": bool(config.get("has_credentials")),
    }
