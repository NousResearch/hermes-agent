#!/usr/bin/env python3
"""
Consultation Tool - Frontier Model Advisory

A single-turn, cost-bounded, user-approved way for an agent to ask one
question to one expensive frontier model and get back one answer. The agent's
primary model continues normally; the consultant is invoked selectively for
specific, difficult problems.

Calls go directly to provider APIs (OpenAI, Anthropic) with no aggregator
or intermediary. Each model in config specifies its own provider, model
name, env var, and pricing.

Design:
  - Tool visibility is opt-in via config (consultation.enabled: false default)
  - Every call requires user approval (same path as dangerous commands)
  - Hard budget caps: per-call and per-session
  - Cost tracking: in-memory + on-disk persistence
  - Dynamic schema: model catalog + budget status injected per turn
  - Direct provider calls: OpenAI SDK or Anthropic SDK, chosen by provider key

See docs/consultation-tool-proposal.md for full design rationale.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSULTATION_SYSTEM_PROMPT = """You are an expert consultant called upon for a specific, difficult problem.
The calling agent has tried to solve this but needs your advanced reasoning
capabilities. Provide a focused, high-quality response. Do not ask clarifying
questions -- this is a single-turn advisory. Be specific and actionable.

IMPORTANT: The context you receive may contain sensitive intellectual property
(financial signals, trading strategies, proprietary algorithms). Treat all
information as confidential. Do not repeat or restate proprietary details
unnecessarily in your response — focus on your analysis and recommendations."""

MAX_CONTEXT_CHARS = 50_000
DEFAULT_SESSION_BUDGET = 20.00
DEFAULT_BUDGET_PER_CALL = 3.00
DEFAULT_SESSION_BUDGET_ALERT = 0.80
DEFAULT_MAX_INPUT_TOKENS = 100_000
DEFAULT_MAX_OUTPUT_TOKENS = 16_000
DEFAULT_INPUT_PRICE_PER_M = 15.00
DEFAULT_OUTPUT_PRICE_PER_M = 75.00
DEFAULT_APPROVAL_TIMEOUT = 300  # 5 minutes

# Provider types
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
SUPPORTED_PROVIDERS = (PROVIDER_OPENAI, PROVIDER_ANTHROPIC)

# ---------------------------------------------------------------------------
# Budget file path
# ---------------------------------------------------------------------------


def _budget_file_path() -> Path:
    state_dir = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    hermes_state = Path(state_dir) / "hermes"
    hermes_state.mkdir(parents=True, exist_ok=True)
    return hermes_state / "consultation_budget.json"


# ---------------------------------------------------------------------------
# In-memory budget tracker (thread-safe)
# ---------------------------------------------------------------------------

_budget_lock = threading.Lock()
_budget_state: Optional[Dict[str, Any]] = None


def _load_budget_state() -> Dict[str, Any]:
    global _budget_state
    if _budget_state is not None:
        return _budget_state

    with _budget_lock:
        if _budget_state is not None:
            return _budget_state

        bf = _budget_file_path()
        try:
            if bf.exists():
                data = json.loads(bf.read_text(encoding="utf-8"))
                session_start = data.get("session_start", "")
                if session_start:
                    try:
                        start_dt = datetime.fromisoformat(session_start.replace("Z", "+00:00"))
                        age_hours = (datetime.now(timezone.utc) - start_dt).total_seconds() / 3600
                        if age_hours > 24:
                            logger.info("Consultation budget file is stale (%.1fh old), resetting", age_hours)
                            raise ValueError("stale session")
                    except (ValueError, TypeError):
                        pass
                _budget_state = data
                return _budget_state
        except Exception as e:
            logger.debug("Could not load budget file (%s), starting fresh", e)

        _budget_state = {
            "session_start": datetime.now(timezone.utc).isoformat(),
            "total_spent": 0.0,
            "by_model": {},
        }
        return _budget_state


def _save_budget_state() -> None:
    global _budget_state
    if _budget_state is None:
        return
    with _budget_lock:
        current = copy.deepcopy(_budget_state)
    bf = _budget_file_path()
    try:
        bf.write_text(json.dumps(current, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save budget file: %s", e)


def _get_session_spent() -> float:
    state = _load_budget_state()
    return state.get("total_spent", 0.0)


def _record_cost(model_name: str, cost: float) -> None:
    global _budget_state
    with _budget_lock:
        if _budget_state is None:
            _budget_state = {"session_start": "", "total_spent": 0.0, "by_model": {}}
        _budget_state["total_spent"] = round(_budget_state.get("total_spent", 0.0) + cost, 4)
        by_model = _budget_state.setdefault("by_model", {})
        entry = by_model.setdefault(model_name, {"calls": 0, "spent": 0.0})
        entry["calls"] += 1
        entry["spent"] = round(entry.get("spent", 0.0) + cost, 4)
    _save_budget_state()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_consultation_config() -> Optional[Dict[str, Any]]:
    """Load consultation config from config.yaml.

    Returns None if consultation is disabled or misconfigured.
    Raises ValueError on config errors.
    """
    try:
        from hermes_cli.config import load_config, cfg_get

        config = load_config()
        consultation = cfg_get(config, "consultation", default=None)
        if consultation is None or not isinstance(consultation, dict):
            return None

        enabled = consultation.get("enabled", False)
        if not enabled:
            return None

        models_raw = consultation.get("models", [])
        if not isinstance(models_raw, list) or len(models_raw) == 0:
            raise ValueError("consultation requires at least one model in config")

        models = []
        for m in models_raw:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            provider = m.get("provider")
            model = m.get("model")
            if not name or not provider or not model:
                continue
            if provider not in SUPPORTED_PROVIDERS:
                logger.warning("Consultation model %s: unsupported provider %s (use openai or anthropic)", name, provider)
                continue

            budget_per_call = m.get("budget_per_call", DEFAULT_BUDGET_PER_CALL)
            if not isinstance(budget_per_call, (int, float)) or budget_per_call <= 0:
                continue

            models.append({
                "name": str(name),
                "provider": str(provider),
                "model": str(model),
                "budget_per_call": float(budget_per_call),
                "max_input_tokens": int(m.get("max_input_tokens", DEFAULT_MAX_INPUT_TOKENS)),
                "max_output_tokens": int(m.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
                "api_key_env": str(m["api_key_env"]) if m.get("api_key_env") else None,
                "description": str(m.get("description", "")),
                "input_price_per_m": float(m.get("input_price_per_m", DEFAULT_INPUT_PRICE_PER_M)),
                "output_price_per_m": float(m.get("output_price_per_m", DEFAULT_OUTPUT_PRICE_PER_M)),
            })

        if not models:
            raise ValueError("no valid consultation models configured")

        return {
            "enabled": True,
            "models": models,
            "session_budget": float(consultation.get("session_budget", DEFAULT_SESSION_BUDGET)),
            "session_budget_alert": float(consultation.get("session_budget_alert", DEFAULT_SESSION_BUDGET_ALERT)),
        }

    except ValueError:
        raise
    except Exception as e:
        logger.debug("Failed to load consultation config: %s", e)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_cost(input_chars: int, max_output_tokens: int,
                   input_price: float, output_price: float) -> float:
    input_tokens = max(input_chars // 4, 1)
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (max_output_tokens / 1_000_000) * output_price
    return round(input_cost + output_cost, 4)


def _resolve_api_key(model_config: Dict[str, Any]) -> str:
    """Resolve the API key for a model config."""
    env_var = model_config.get("api_key_env")
    if not env_var:
        # Default env vars by provider
        if model_config["provider"] == PROVIDER_OPENAI:
            env_var = "OPENAI_API_KEY"
        elif model_config["provider"] == PROVIDER_ANTHROPIC:
            env_var = "ANTHROPIC_API_KEY"
    key = os.environ.get(env_var)
    if not key:
        raise ValueError(f"{env_var} environment variable not set (needed for {model_config['name']})")
    return key


# ---------------------------------------------------------------------------
# Provider clients (lazy, cached)
# ---------------------------------------------------------------------------

_openai_clients: Dict[str, Any] = {}
_anthropic_clients: Dict[str, Any] = {}
_client_lock = threading.Lock()


def _get_openai_client(api_key: str):
    """Get or create an async OpenAI client."""
    with _client_lock:
        if api_key in _openai_clients:
            return _openai_clients[api_key]
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        _openai_clients[api_key] = client
        return client


def _get_anthropic_client(api_key: str):
    """Get or create an async Anthropic client."""
    with _client_lock:
        if api_key in _anthropic_clients:
            return _anthropic_clients[api_key]
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        _anthropic_clients[api_key] = client
        return client


# ---------------------------------------------------------------------------
# API call logic
# ---------------------------------------------------------------------------


async def _call_openai(model: str, system: str, user: str, max_tokens: int) -> tuple:
    """Call OpenAI API directly. Returns (content, input_tokens, output_tokens)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = _get_openai_client(api_key)
    # GPT-5+ reasoning models require max_completion_tokens instead of max_tokens
    kwargs: dict = {"max_completion_tokens": max_tokens}
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
    except Exception as e:
        # Fallback to max_tokens for older models
        if "max_completion_tokens" in str(e) or "max_tokens" in str(e):
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
        else:
            raise
    content = response.choices[0].message.content or ""
    usage = response.usage or type('U', (), {"prompt_tokens": 0, "completion_tokens": 0})()
    return content, usage.prompt_tokens, usage.completion_tokens


async def _call_anthropic(model: str, system: str, user: str, max_tokens: int) -> tuple:
    """Call Anthropic API directly. Returns (content, input_tokens, output_tokens)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = _get_anthropic_client(api_key)
    response = await client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    content = "\n".join(block.text for block in response.content if hasattr(block, "text"))
    usage = response.usage
    return content, usage.input_tokens, usage.output_tokens


# ---------------------------------------------------------------------------
# Tool visibility check
# ---------------------------------------------------------------------------


def _check_consultation() -> bool:
    try:
        cfg = _load_consultation_config()
        return cfg is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dynamic schema builder
# ---------------------------------------------------------------------------


def _build_dynamic_schema() -> Dict[str, Any]:
    try:
        cfg = _load_consultation_config()
    except Exception:
        cfg = None

    if not cfg or not cfg.get("models"):
        return {}

    models = cfg["models"]
    spent = _get_session_spent()
    budget = cfg.get("session_budget", DEFAULT_SESSION_BUDGET)

    model_lines = []
    for m in models:
        desc_line = m["description"].strip().split("\n")[0]
        model_lines.append(
            f"  - {m['name']} ({m['provider']}:{m['model']}): "
            f"{desc_line} ~${m['input_price_per_m']}/M in, "
            f"~${m['output_price_per_m']}/M out. Budget: ${m['budget_per_call']:.2f}/call."
        )

    model_catalog = "\n".join(model_lines)

    budget_line = ""
    if budget > 0:
        pct = min((spent / budget) * 100, 999)
        budget_line = f"\n\nCurrent session budget: ${spent:.2f}/${budget:.2f} ({pct:.0f}% used)."
        if spent >= budget:
            budget_line += " BUDGET EXHAUSTED - no more calls possible."

    description = (
        "Ask a frontier or peer model for expert advisory on a specific, "
        "difficult problem. Each call costs money and requires user approval. "
        "Use consultants (opus, gpt55) for hard problems; use peers (sonnet, gpt41) "
        "for cheaper second opinions and sanity checks. "
        "The consultant model receives your prompt and optional context and "
        "returns a single advisory response. Keep prompts focused and specific.\n\n"
        "WARNING: You are working with sensitive IP (financial signals, trading "
        "strategies, proprietary algorithms). Share ONLY what the consultant needs "
        "to answer the question — never dump large code files, full feature lists, "
        "or complete signal logic. Summarize and abstract when possible.\n\n"
        f"Available models:\n{model_catalog}{budget_line}"
    )

    model_choices = [m["name"] for m in models]

    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": model_choices,
                    "description": f"Consultation model name. Available: {', '.join(model_choices)}",
                },
                "prompt": {
                    "type": "string",
                    "description": "The specific question to ask the expert model. Be focused and specific.",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Optional additional context: code, file contents, "
                        "error traces, etc. Max 50,000 characters. "
                        "SENSITIVE: share only what is strictly necessary — "
                        "summarize and abstract; never dump full source files or "
                        "complete signal logic."
                    ),
                },
            },
            "required": ["model", "prompt"],
        },
    }


# ---------------------------------------------------------------------------
# Core tool handler
# ---------------------------------------------------------------------------


async def consultation_ask(args: Dict[str, Any]) -> str:
    """Handle a consultation.ask tool call."""
    model_name = args.get("model", "")
    prompt = args.get("prompt", "")
    context = args.get("context", "") or ""

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]
        logger.warning("Consultation context truncated to %d chars", MAX_CONTEXT_CHARS)

    if not model_name:
        return json.dumps({"error": "model parameter is required"})
    if not prompt:
        return json.dumps({"error": "prompt parameter is required"})

    # 1. Load config
    try:
        cfg = _load_consultation_config()
    except Exception as e:
        return json.dumps({"error": f"Consultation config error: {e}"})

    if not cfg:
        return json.dumps({"error": "Consultation is not enabled in config"})

    # Find model config
    model_cfg = None
    for m in cfg["models"]:
        if m["name"] == model_name:
            model_cfg = m
            break

    if not model_cfg:
        available = ", ".join(m["name"] for m in cfg["models"])
        return json.dumps({"error": f"Unknown consultation model '{model_name}'. Available: {available}"})

    provider = model_cfg["provider"]
    model_id = model_cfg["model"]
    budget_per_call = model_cfg["budget_per_call"]
    max_output_tokens = model_cfg["max_output_tokens"]
    input_price = model_cfg["input_price_per_m"]
    output_price = model_cfg["output_price_per_m"]

    # 2. Build the full message
    user_message = prompt
    if context:
        user_message = f"{prompt}\n\nContext:\n{context}"

    # Estimate cost
    full_text = CONSULTATION_SYSTEM_PROMPT + "\n" + user_message
    estimated_cost = _estimate_cost(len(full_text), max_output_tokens, input_price, output_price)

    # 3. Check session budget (fail fast)
    session_spent = _get_session_spent()
    session_budget = cfg.get("session_budget", DEFAULT_SESSION_BUDGET)

    if session_budget > 0 and session_spent >= session_budget:
        return json.dumps({
            "error": (
                f"Session consultation budget exhausted (${session_spent:.2f}/${session_budget:.2f}). "
                f"No more consultation calls possible this session."
            ),
        })

    # 4. Approval gate
    prompt_preview = prompt[:200] + ("..." if len(prompt) > 200 else "")

    approval_description = (
        f"[CONSULTATION] Agent wants to consult {model_name} ({provider}:{model_id}).\n"
        f"Estimated cost: ${estimated_cost:.2f} (call cap: ${budget_per_call:.2f}).\n"
        f"Session budget: ${session_spent:.2f}/${session_budget:.2f} "
        f"({(session_spent / session_budget * 100) if session_budget > 0 else 0:.0f}% used).\n"
        f"Question: {prompt_preview}"
    )

    try:
        from tools.approval import prompt_dangerous_approval

        choice = prompt_dangerous_approval(
            command="consultation.ask",
            description=approval_description,
            timeout_seconds=DEFAULT_APPROVAL_TIMEOUT,
            allow_permanent=False,
        )

        if choice not in ("once", "session"):
            return json.dumps({"error": "User declined this consultation call."})

    except Exception as e:
        logger.error("Approval failed: %s", e)
        return json.dumps({"error": f"Could not get user approval: {e}"})

    # 5. Final budget check
    session_spent = _get_session_spent()
    if session_budget > 0 and session_spent >= session_budget:
        return json.dumps({"error": "Session consultation budget exhausted. Call cancelled."})

    # Calculate effective max_output_tokens based on remaining budget
    remaining_budget = min(
        budget_per_call,
        session_budget - session_spent if session_budget > 0 else budget_per_call,
    )
    max_output_by_budget = int((remaining_budget * 1_000_000) / output_price)
    effective_max_output = min(max_output_tokens, max_output_by_budget, int(remaining_budget * 1000))
    effective_max_output = max(effective_max_output, 256)

    # 6. API call
    try:
        _resolve_api_key(model_cfg)  # validates key exists
    except ValueError as e:
        return json.dumps({"error": str(e)})

    try:
        logger.info(
            "Consultation call: model=%s provider=%s model_id=%s max_output=%d estimated=$%.2f",
            model_name, provider, model_id, effective_max_output, estimated_cost,
        )

        if provider == PROVIDER_OPENAI:
            content, input_tokens, output_tokens = await _call_openai(
                model_id, CONSULTATION_SYSTEM_PROMPT, user_message, effective_max_output
            )
        elif provider == PROVIDER_ANTHROPIC:
            content, input_tokens, output_tokens = await _call_anthropic(
                model_id, CONSULTATION_SYSTEM_PROMPT, user_message, effective_max_output
            )
        else:
            return json.dumps({"error": f"Unsupported provider: {provider}"})

        if not content:
            logger.warning("Consultation model returned empty response")
            return json.dumps({"error": f"{model_name} returned an empty response."})

        # Calculate actual cost
        actual_cost = round(
            (input_tokens / 1_000_000) * input_price +
            (output_tokens / 1_000_000) * output_price,
            4,
        )

        # 7. Track cost
        _record_cost(model_name, actual_cost)
        new_total = _get_session_spent()

        # Budget alert
        alert_threshold = cfg.get("session_budget_alert", DEFAULT_SESSION_BUDGET_ALERT)
        if session_budget > 0 and (new_total / session_budget) >= alert_threshold:
            logger.warning(
                "Consultation budget alert: %.0f%% consumed ($%.2f/$%.2f)",
                (new_total / session_budget) * 100, new_total, session_budget,
            )

        logger.info(
            "Consultation complete: model=%s cost=$%.2f input=%d output=%d session=$%.2f/$%.2f",
            model_name, actual_cost, input_tokens, output_tokens,
            new_total, session_budget,
        )

        return json.dumps({
            "response": content,
            "model": model_name,
            "provider": f"{provider}:{model_id}",
            "cost": actual_cost,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "session_spent": round(new_total, 2),
            "session_budget": session_budget,
            "session_remaining": round(max(session_budget - new_total, 0), 2),
        }, ensure_ascii=False)

    except Exception as e:
        error_msg = str(e)
        logger.error("Consultation API call failed: %s", error_msg)
        return json.dumps({"error": f"Consultation call failed: {error_msg}"})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

CONSULTATION_SCHEMA = {
    "name": "consultation.ask",
    "description": "Ask an expensive frontier model for expert advisory.",
    "parameters": {
        "type": "object",
        "properties": {
            "model": {"type": "string", "description": "Consultation model name"},
            "prompt": {"type": "string", "description": "The specific question to ask"},
            "context": {"type": "string", "description": "Optional context: code, errors, etc."},
        },
        "required": ["model", "prompt"],
    },
}

registry.register(
    name="consultation.ask",
    toolset="consultation",
    schema=CONSULTATION_SCHEMA,
    handler=lambda args, **kw: consultation_ask(args),
    check_fn=_check_consultation,
    is_async=True,
    description="Ask an expensive frontier model for expert advisory on a specific, difficult problem. Each call costs money and requires user approval.",
    emoji="\U0001f393",
    dynamic_schema_overrides=_build_dynamic_schema,
)
