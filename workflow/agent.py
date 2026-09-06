"""Run one agent step as a short-lived AIAgent turn.

A workflow run is its own session — it does not mutate the user's canvas
chat, so prompt-cache on that conversation stays intact.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable


def build_prompt(goal: str, context: str, payload: Any, profile: str | None = None) -> str:
    parts = []
    if profile:
        parts.append(f"You are the {profile} specialist on this workflow.")
    parts.append(goal.strip() or "Complete this step.")
    if context.strip():
        parts.append("Upstream output:\n" + context.strip())
    if payload not in (None, "", {}, []):
        blob = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
        parts.append("Trigger payload:\n" + blob)
    parts.append(
        "When you finish, reply with a short summary. If this step is a "
        "check, end with a line that is exactly PASS or FAIL."
    )
    return "\n\n".join(parts)


def parse_result(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    verdict = None
    output: dict[str, Any] = {"text": raw}
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            output = {**parsed, "text": raw}
            value = parsed.get("verdict")
            if isinstance(value, str) and value.upper() in {"PASS", "FAIL"}:
                verdict = value.upper()
    if verdict is None:
        tail = re.findall(r"\b(PASS|FAIL)\b", raw.upper())
        if tail:
            verdict = tail[-1]
    return {"summary": raw[:400] or "done", "verdict": verdict, "output": output}


def _arg_preview(args: Any) -> str:
    if isinstance(args, dict):
        for value in args.values():
            if value not in (None, ""):
                return str(value)[:80]
        return ""
    return str(args or "")[:80]


# Demo slugs the starter canvas used to ship. They are not in any catalog —
# sending them made every step 404 in ~300ms and the run skipped to approval.
_PLACEHOLDER_MODELS = {
    "claude-opus-4.8",
    "gpt-5.6-sol",
    "gpt-5.3-codex",
    "deepseek-v3.2",
    "kimi-k2-thinking",
}


_FAIL_MARKERS = (
    "requested model does not exist",
    "could not load the agent",
    "could not resolve authentication",
    "api_key or auth_token",
    "model parameter is required",
)


def _reply_text(text: str) -> str:
    return (text or "").strip().strip('"').strip("'")


def is_failed_reply(text: str) -> bool:
    """True when the model call never happened and the text is the error."""
    raw = _reply_text(text)
    if raw.startswith("HTTP 4") or raw.startswith("HTTP 5"):
        return True
    low = raw.lower()
    return any(mark in low for mark in _FAIL_MARKERS)


def is_user_fixable(error: str) -> bool:
    """Missing model / catalog miss — retrying the same slug will not help."""
    low = _reply_text(error).lower()
    return (
        any(mark in low for mark in _FAIL_MARKERS)
        or "http 404" in low
        or "http 401" in low
        or "http 403" in low
        or "http 400" in low
    )


def _default_model() -> tuple[str | None, str | None]:
    try:
        from hermes_cli.profiles import _read_config_model
        from hermes_constants import get_hermes_home

        return _read_config_model(get_hermes_home())
    except Exception:
        return None, None


def resolve_step_model(config: dict | None) -> tuple[str, str | None, str | None]:
    """Card model wins; else the named profile; else this Hermes home."""
    cfg = config or {}
    model = str(cfg.get("model") or "").strip()
    if model in _PLACEHOLDER_MODELS:
        model = ""
    provider = None
    profile = str(cfg.get("profile") or "").strip() or None
    if profile:
        try:
            from hermes_cli.profiles import _read_config_model, get_profile_dir, profile_exists

            if profile_exists(profile):
                profile_model, profile_provider = _read_config_model(get_profile_dir(profile))
                if not model and profile_model:
                    model = str(profile_model)
                if profile_provider:
                    provider = str(profile_provider)
        except Exception:
            # A profile that won't read is not a reason to fail the step —
            # fall through to this home's default, same as no profile at all.
            pass
    if not model or not provider:
        default_model, default_provider = _default_model()
        if not model and default_model:
            model = str(default_model)
        if not provider and default_provider:
            provider = str(default_provider)
    return model, provider, profile


def execute_agent_step(
    goal: str,
    context: str,
    payload: Any,
    config: dict | None = None,
    *,
    on_tool: Callable[[str, str], None] | None = None,
    session_id: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Call a real model. Tests inject their own execute_fn and never hit this."""
    cfg = config or {}
    model, provider, profile = resolve_step_model(cfg)
    if not model:
        return {
            "ok": False,
            "error": "Model parameter is required. Set a model on this step or in Hermes settings.",
        }
    prompt = build_prompt(goal, context, payload, profile)
    if resume:
        prompt = "Continue from where you left off. The last turn was interrupted.\n\n" + prompt
    try:
        from run_agent import AIAgent
    except Exception as exc:
        return {"ok": False, "error": f"could not load the agent: {exc}"}

    iterations = cfg.get("maxIterations") or 20
    try:
        iterations = max(1, min(int(iterations), 200))
    except (TypeError, ValueError):
        iterations = 20

    timeout_mins = cfg.get("timeoutMins") or 0
    try:
        timeout_mins = max(0, int(timeout_mins))
    except (TypeError, ValueError):
        timeout_mins = 0

    def started(_call_id, name, args):
        if on_tool is not None:
            on_tool(str(name or ""), _arg_preview(args))

    kwargs: dict[str, Any] = {
        "model": model,
        "quiet_mode": True,
        "skip_memory": True,
        "skip_context_files": True,
        "max_iterations": iterations,
        "run_budget_seconds": (timeout_mins * 60) if timeout_mins else None,
        "tool_start_callback": started,
        "platform": "workflow",
        "session_id": session_id,
    }
    if provider:
        kwargs["provider"] = provider

    try:
        agent = AIAgent(**kwargs)
        text = agent.chat(prompt)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    raw = str(text or "")
    if is_failed_reply(raw):
        return {"ok": False, "error": raw.strip() or "model call failed"}

    parsed = parse_result(raw)
    return {"ok": True, "sessionId": session_id, **parsed}
