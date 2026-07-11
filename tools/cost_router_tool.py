#!/usr/bin/env python3
"""Cost-aware routing tool.

Provides a small, explicit router surface for controller sessions.  Unlike
``delegate_task`` (which uses one configured delegation model), this tool routes
bounded worker-like slices to named Hermes worker profiles via the Hermes CLI.
The controller keeps final judgment; workers return evidence/drafts/local
analysis only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tools.registry import registry
from tools.project_intake import build_project_card
from utils import is_truthy_value
from agent.redact import redact_sensitive_text

try:
    import yaml
except Exception:  # pragma: no cover - yaml is present in Hermes runtime
    yaml = None


logger = logging.getLogger(__name__)

_ALLOWED_ROUTES: Dict[str, str] = {
    "luna": "worker-luna",
    "worker_luna": "worker-luna",
    "luna_economy": "worker-luna-economy",
    "economy": "worker-luna-economy",
    "worker_luna_economy": "worker-luna-economy",
    "terra": "worker-terra",
    "worker_terra": "worker-terra",
    "sol": "worker-sol",
    "worker_sol": "worker-sol",
}

_ALLOWED_PROFILES = frozenset(_ALLOWED_ROUTES.values())
_DEFAULT_TIMEOUT_SECONDS = 600
_MAX_TIMEOUT_SECONDS = 1800
_MAX_PROMPT_CHARS = 24000
_WORKER_RESULT_FOOTER_RE = re.compile(
    r"<cost_router_result>\s*(.*?)\s*</cost_router_result>\s*$",
    re.DOTALL,
)
_WORKER_RESULT_OPEN_TAG = "<cost_router_result>"
_WORKER_RESULT_STATUSES = frozenset({"complete", "partial", "blocked"})
_WORKER_RESULT_FIELDS = frozenset(
    {"status", "deliverable", "evidence", "unverified_items", "controller_decisions"}
)
_RETRYABLE_TIERS = frozenset({"luna", "luna_economy"})


def _safe_text(value: Any, limit: Optional[int] = 4000) -> str:
    """Redact worker-controlled text and optionally bound diagnostic fields."""
    try:
        redacted = redact_sensitive_text(str(value or ""), force=True)
        return redacted[-limit:] if limit is not None else redacted
    except Exception:
        return "[REDACTED - redaction failed]"


def _safe_json(payload: Dict[str, Any]) -> str:
    """Serialize a return payload through forced secret redaction."""
    return _safe_text(json.dumps(payload, ensure_ascii=False), limit=None)


def _classify_failure(text: str, *, timed_out: bool = False) -> tuple[bool, str]:
    """Classify only narrow infrastructure failures as controller-retryable."""
    if timed_out:
        return True, "timeout"
    lowered = (text or "").lower()
    if re.search(r"(?:http(?:/\d(?:\.\d)?)?\s*)?429\b|too many requests|rate limit", lowered):
        return True, "http_429"
    if re.search(r"(?:http(?:/\d(?:\.\d)?)?\s*)?503\b|service unavailable", lowered):
        return True, "http_503"
    if any(marker in lowered for marker in (
        "connectionerror", "connection error", "connection refused", "connection reset",
        "failed to connect", "could not connect", "network is unreachable",
        "temporary failure in name resolution",
    )):
        return True, "connection_failure"
    return False, "non_retryable_failure"


def _tool_error(message: str) -> str:
    return _safe_json({"error": message})


def _route_from_explicit(route: Optional[str]) -> Optional[tuple[str, str]]:
    """Return a validated route/profile pair, or ``None`` when not supplied."""
    raw = (route or "").strip().lower().replace("-", "_")
    if not raw:
        return None
    profile = _ALLOWED_ROUTES.get(raw)
    if raw in _ALLOWED_PROFILES:
        return raw.replace("worker-", "").replace("-", "_"), raw
    if profile:
        return ("luna_economy" if profile == "worker-luna-economy" else raw.replace("worker_", "")), profile
    raise ValueError(
        "Unknown cost_router route/profile %r. Use one of: luna, luna_economy, terra, sol, "
        "worker-luna, worker-luna-economy, worker-terra, worker-sol." % route
    )


_TASK_TYPE_ROUTES: Dict[str, tuple[str, str]] = {
    "router": ("luna", "worker-luna"),
    "classify": ("luna", "worker-luna"),
    "dedupe": ("luna_economy", "worker-luna-economy"),
    "bulk_preprocess": ("luna_economy", "worker-luna-economy"),
    "rag_answer": ("terra", "worker-terra"),
    "draft": ("terra", "worker-terra"),
    "coding": ("terra", "worker-terra"),
    "final_review": ("sol", "worker-sol"),
    "architecture": ("sol", "worker-sol"),
}


def _select_route(
    route: Optional[str],
    goal: str,
    context: Optional[str],
    task_type: Optional[str] = None,
    project: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Select a route with declared metadata taking precedence over keywords."""
    selected = _route_from_explicit(route)
    if selected:
        tier, profile = selected
        return {"tier": tier, "profile": profile, "selection_mode": "explicit", "matched_rule": f"route:{route}"}

    project_route = (project or {}).get("route") or (project or {}).get("route_candidate")
    selected = _route_from_explicit(str(project_route) if project_route else None)
    if selected:
        tier, profile = selected
        return {"tier": tier, "profile": profile, "selection_mode": "explicit", "matched_rule": f"project.route_candidate:{project_route}"}

    task = (task_type or "").strip().lower().replace("-", "_")
    selected = _TASK_TYPE_ROUTES.get(task)
    if selected:
        tier, profile = selected
        return {"tier": tier, "profile": profile, "selection_mode": "task_type", "matched_rule": f"task_type:{task}"}

    text = f"{goal}\n{context or ''}".lower()
    keyword_rules = (
        ("sol", "worker-sol", ("final", "audit", "review", "裁决", "终审", "最终审核", "架构", "系统升级", "conflict", "冲突", "low confidence", "低置信", "prompt design", "深度推理")),
        ("terra", "worker-terra", ("draft", "rewrite", "rag_answer", "article", "script", "coding", "mother", "母本", "正文", "创作", "生成", "长文", "影视分析", "电影解说", "obsidian 总结")),
        ("luna_economy", "worker-luna-economy", ("bulk", "batch", "many", "大量", "批量", "cheap", "economy", "low cost", "低成本", "log", "日志", "dedupe", "去重", "json", "metadata", "tag", "标签")),
        ("luna", "worker-luna", ("router", "classify", "dedupe", "tag", "metadata", "ocr", "compression", "query", "json", "cache", "filter", "dispatch", "日志", "分类", "去重", "标签")),
    )
    for tier, profile, keywords in keyword_rules:
        for keyword in keywords:
            if keyword in text:
                return {"tier": tier, "profile": profile, "selection_mode": "keyword", "matched_rule": f"keyword:{keyword}"}
    return {"tier": "terra", "profile": "worker-terra", "selection_mode": "default", "matched_rule": "default:terra"}


def _normalise_route(route: Optional[str], goal: str, context: Optional[str], task_type: Optional[str] = None) -> tuple[str, str]:
    """Compatibility wrapper returning only the historical tier/profile pair."""
    decision = _select_route(route, goal, context, task_type)
    return decision["tier"], decision["profile"]


def _safe_decision_metadata(project: Optional[Dict[str, Any]], parent_session_id: Optional[str]) -> Dict[str, str]:
    """Expose only non-secret correlation identifiers in routing responses/logs."""
    metadata: Dict[str, str] = {}
    for key in ("project_id", "id", "slice_id"):
        value = (project or {}).get(key)
        if value is not None and str(value).strip():
            metadata["project_id" if key in {"project_id", "id"} else "slice_id"] = str(value).strip()[:256]
    if parent_session_id and str(parent_session_id).strip():
        metadata["parent_session_id"] = str(parent_session_id).strip()[:256]
    return metadata


def _string_list(value: Any) -> list[Any]:
    """Normalize a worker-result list while retaining structured handles."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, (str, dict))]


def _fallback_worker_result(
    prose: str,
    contract_status: str,
    message: str,
    exit_code: int,
) -> Dict[str, Any]:
    """Represent legacy or malformed output without claiming completion."""
    unverified_items = [message]
    status = "partial"
    if exit_code != 0:
        status = "blocked"
        unverified_items.append("Worker process returned a non-zero exit status.")
    return {
        "status": status,
        "deliverable": prose or None,
        "evidence": [],
        "unverified_items": unverified_items,
        "controller_decisions": ["Controller must assess the free-text worker output."],
        "contract_status": contract_status,
        "controller_acceptance_required": True,
    }


def _valid_worker_result(parsed: Any) -> bool:
    """Validate the required footer shape without assigning controller acceptance."""
    if not isinstance(parsed, dict) or not _WORKER_RESULT_FIELDS.issubset(parsed):
        return False
    if not isinstance(parsed["status"], str) or parsed["status"] not in _WORKER_RESULT_STATUSES:
        return False
    if parsed["deliverable"] is not None and not isinstance(parsed["deliverable"], (str, dict)):
        return False
    return all(
        isinstance(parsed[field], list)
        and all(isinstance(item, (str, dict)) for item in parsed[field])
        for field in ("evidence", "unverified_items", "controller_decisions")
    )


def _parse_worker_result(output: str, exit_code: Any) -> tuple[str, Dict[str, Any]]:
    """Split prose from an optional structured footer and normalize its contract."""
    raw = (output or "").strip()
    # Only the final opening tag can begin the authoritative terminal footer.
    # Earlier tags may appear in ordinary prose as examples or quoted data.
    footer_start = raw.rfind(_WORKER_RESULT_OPEN_TAG)
    match = _WORKER_RESULT_FOOTER_RE.fullmatch(raw[footer_start:]) if footer_start >= 0 else None
    if match is None:
        if footer_start >= 0:
            prose = raw[:footer_start].strip()
            return prose, _fallback_worker_result(
                prose,
                "malformed",
                "Structured worker-result footer was malformed.",
                exit_code,
            )
        return raw, _fallback_worker_result(
            raw,
            "absent",
            "Worker returned legacy free text without a structured result footer.",
            exit_code,
        )

    prose = raw[:footer_start].strip()
    try:
        parsed = json.loads(match.group(1))
    except (TypeError, ValueError):
        return prose, _fallback_worker_result(
            prose,
            "malformed",
            "Structured worker-result footer was malformed.",
            exit_code,
        )

    if not _valid_worker_result(parsed):
        return prose, _fallback_worker_result(
            prose,
            "malformed",
            "Structured worker-result footer was malformed.",
            exit_code,
        )

    status = parsed["status"]
    unverified_items = _string_list(parsed.get("unverified_items"))
    if exit_code != 0:
        status = "blocked"
        unverified_items.append("Worker process returned a non-zero exit status.")
    return prose, {
        "status": status,
        "deliverable": parsed.get("deliverable"),
        "evidence": _string_list(parsed.get("evidence")),
        "unverified_items": unverified_items,
        "controller_decisions": _string_list(parsed.get("controller_decisions")),
        "contract_status": "valid",
        "controller_acceptance_required": True,
    }


def _build_acceptance_check(
    project_card: Dict[str, Any],
    worker_result: Dict[str, Any],
    exit_code: Any,
) -> Dict[str, Any]:
    """Build a controller-owned review checklist without making the decision.

    The checklist is deliberately advisory: it surfaces the questions and
    worker-provided handles a controller must verify.  It never publishes,
    promotes, accepts, or dispatches a Sol review.
    """
    evidence = _string_list(worker_result.get("evidence"))
    unresolved = _string_list(worker_result.get("unverified_items"))
    decisions = _string_list(worker_result.get("controller_decisions"))
    deliverable = worker_result.get("deliverable")
    acceptance_evidence = _string_list(project_card.get("acceptance_evidence"))

    sol_reasons: list[str] = []
    decision_text = json.dumps(decisions, ensure_ascii=False).lower()
    if any(marker in decision_text for marker in (
        "conflict", "architecture", "security", "high impact", "low confidence",
        "冲突", "架构", "安全", "低置信",
    )):
        sol_reasons.append("Worker surfaced a conflict or high-judgment controller decision.")
    if worker_result.get("status") == "blocked" and decisions:
        sol_reasons.append("Blocked work includes a controller decision that may need deep review.")

    checklist = {
        "deliverable_coverage": {
            "status": "review" if deliverable is not None else "missing",
            "expected": acceptance_evidence,
            "reported_deliverable": deliverable,
        },
        "evidence_sufficiency": {
            "status": "review" if evidence else "insufficient",
            "reported_evidence": evidence,
        },
        "unsupported_claims": {
            "status": "review",
            "instruction": "Compare material claims with the reported evidence; worker claims are not self-verifying.",
        },
        "cross_slice_conflicts": {
            "status": "review",
            "instruction": "Compare this result with sibling slices before final synthesis.",
            "reported_controller_decisions": decisions,
        },
        "verifiable_artifacts": {
            "status": "review" if deliverable is not None or evidence else "missing",
            "handles": ([deliverable] if deliverable is not None else []) + evidence,
        },
        "unresolved_items": {
            "status": "attention" if unresolved else "none_reported",
            "items": unresolved,
        },
        "sol_review_warranted": {
            "status": "consider" if sol_reasons else "not_warranted",
            "reasons": sol_reasons,
        },
    }
    blocking_reasons = ["worker_exit_success" if exit_code == 0 else "worker_execution_failed"]
    if worker_result.get("status") != "complete":
        blocking_reasons.append("worker_result_not_complete")
    if unresolved:
        blocking_reasons.append("unresolved_items_reported")
    if worker_result.get("contract_status") != "valid":
        blocking_reasons.append("worker_result_contract_not_valid")

    return {
        "accepted": False,
        "controller_decision_required": True,
        "blocking_reasons": blocking_reasons,
        "checklist": checklist,
        "sol_review": {
            "recommended": bool(sol_reasons),
            "automatic": False,
            "reasons": sol_reasons,
        },
    }


def _build_prompt(goal: str, context: Optional[str], tier: str, profile: str, task_type: Optional[str], project: Optional[Dict[str, Any]] = None) -> str:
    parts = [
        "You are a routed worker profile invoked by Hermes cost_router.",
        "Return only the requested worker output: evidence, structure, draft, or local judgment.",
        "Do NOT make final blocker/risk/publication/stage-promotion decisions; the controller keeps final authority.",
        "You may write ordinary prose, but end with exactly one machine-readable footer:",
        '<cost_router_result>{"status":"complete|partial|blocked","deliverable":"summary or handle","evidence":[],"unverified_items":[],"controller_decisions":[]}</cost_router_result>',
        "Use evidence for verifiable handles/results, list anything not verified, and leave acceptance to the controller.",
        f"Tier: {tier}; profile: {profile}; task_type: {task_type or 'auto'}.",
        "",
        "Goal:",
        goal.strip(),
    ]
    if context and str(context).strip():
        parts.extend(["", "Context:", str(context).strip()])
    prompt = "\n".join(parts)
    if len(prompt) > _MAX_PROMPT_CHARS:
        prompt = prompt[: _MAX_PROMPT_CHARS - 80] + "\n… [cost_router prompt truncated]"
    return prompt


def _strip_session_banner(output: str) -> tuple[Optional[str], str]:
    """Return (session_id, cleaned_output) from `hermes chat -Q` stream text."""
    session_id = None
    cleaned = []
    for line in (output or "").splitlines():
        m = re.match(r"^\s*session_id:\s*(\S+)\s*$", line)
        if m:
            session_id = m.group(1)
            continue
        cleaned.append(line)
    return session_id, "\n".join(cleaned).strip()


def _profile_env_overrides(profile: str) -> Dict[str, str]:
    """Return env overrides needed by provider SDKs for a worker profile."""
    if yaml is None:
        return {}
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    cfg_path = home / "profiles" / profile / "config.yaml"
    try:
        loaded = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}
    cfg = loaded if isinstance(loaded, dict) else {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    provider = str(model_cfg.get("provider") or "").strip().lower()
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    provider_cfg = providers.get(provider) if isinstance(providers.get(provider), dict) else {}
    api_key = str(provider_cfg.get("api_key") or model_cfg.get("api_key") or "").strip()
    if not api_key:
        return {}
    if provider == "deepseek":
        return {"DEEPSEEK_API_KEY": api_key}
    return {}


def _run_attempt(
    hermes_bin: str, prompt: str, tier: str, profile: str, effective_timeout: int,
) -> Dict[str, Any]:
    """Run one worker attempt and return a safe, structured result."""
    env = os.environ.copy()
    env.update(_profile_env_overrides(profile))
    cmd = [hermes_bin, "chat", "--profile", profile, "-q", prompt, "-Q"]
    try:
        completed = subprocess.run(
            cmd, text=True, capture_output=True, timeout=effective_timeout,
            env=env, cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "tier": tier, "profile": profile, "session_id": None,
            "exit_code": "timeout", "output": _safe_text(stdout),
            "stderr": _safe_text(stderr),
            "error": f"cost_router timed out after {effective_timeout}s",
            "retryable": True, "failure_reason": "timeout",
        }
    except Exception as exc:
        retryable, reason = _classify_failure(str(exc))
        return {
            "tier": tier, "profile": profile, "session_id": None,
            "exit_code": None, "output": "", "stderr": "",
            "error": _safe_text(f"cost_router failed to launch Hermes CLI: {exc}"),
            "retryable": retryable, "failure_reason": reason,
        }

    session_id, stdout = _strip_session_banner(completed.stdout)
    stderr_session_id, stderr = _strip_session_banner(completed.stderr)
    # Worker stdout is untrusted content; only stderr may trigger fallback.
    retryable, reason = _classify_failure(stderr) if completed.returncode else (False, "none")
    attempt: Dict[str, Any] = {
        "tier": tier, "profile": profile,
        "session_id": session_id or stderr_session_id,
        "exit_code": completed.returncode, "output": _safe_text(stdout),
        "retryable": retryable, "failure_reason": reason,
    }
    if stderr.strip():
        attempt["stderr"] = _safe_text(stderr)
    if completed.returncode != 0:
        attempt["error"] = "worker profile returned non-zero exit status"
    return attempt


def check_cost_router_requirements() -> bool:
    return shutil.which("hermes") is not None


def cost_router(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    route: Optional[str] = None,
    task_type: Optional[str] = None,
    project: Optional[Dict[str, Any]] = None,
    parent_session_id: Optional[str] = None,
    allow_fallback: Optional[bool] = None,
    timeout: Optional[int] = None,
    background: Optional[bool] = None,
    parent_agent=None,
) -> str:
    """Route a bounded execution slice to a named worker profile."""
    del parent_agent  # currently reserved for future in-process delegation integration

    if not goal or not str(goal).strip():
        return _tool_error("cost_router requires a non-empty 'goal'.")

    if is_truthy_value(background, default=False):
        return _tool_error(
            "cost_router background mode is not implemented in the minimal profile-router. "
            "Use delegate_task for async generic delegation or call cost_router synchronously."
        )

    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        return _tool_error("Hermes CLI not found on PATH; cannot route to worker profiles.")

    # Build and decide on intake before selecting or invoking a worker route.
    # Structured project metadata can select a route but never bypasses split gating.
    project_metadata = project if isinstance(project, dict) else {}
    resolved_task_type = task_type or project_metadata.get("task_type")
    project_card = build_project_card(
        str(goal),
        context=context,
        task_type=resolved_task_type,
    )
    decision_metadata = _safe_decision_metadata(project_metadata, parent_session_id)
    if project_card["split_required"]:
        return json.dumps(
            {
                "routing_status": "split_required",
                "project_card": project_card,
                "decision_metadata": decision_metadata,
                "message": "Request requires controller-side task splitting before worker routing.",
            },
            ensure_ascii=False,
        )

    try:
        decision = _select_route(route, str(goal), str(context or ""), resolved_task_type, project_metadata)
    except ValueError as exc:
        return _tool_error(str(exc))
    tier, profile = decision["tier"], decision["profile"]
    project_card["route_candidate"] = tier

    try:
        effective_timeout = int(timeout or _DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        effective_timeout = _DEFAULT_TIMEOUT_SECONDS
    effective_timeout = max(1, min(effective_timeout, _MAX_TIMEOUT_SECONDS))

    prompt = _build_prompt(str(goal), context, tier, profile, resolved_task_type, project_metadata)
    started_at = time.monotonic()
    attempts = [_run_attempt(hermes_bin, prompt, tier, profile, effective_timeout)]
    fallback_from = None
    fallback_reason = None
    fallback_allowed = is_truthy_value(
        allow_fallback if allow_fallback is not None else project_metadata.get("allow_fallback"),
        default=False,
    )
    first = attempts[0]
    if first["retryable"] and tier in _RETRYABLE_TIERS and fallback_allowed:
        fallback_from = tier
        fallback_reason = first["failure_reason"]
        fallback_prompt = _build_prompt(
            str(goal), context, "terra", "worker-terra", resolved_task_type, project_metadata
        )
        attempts.append(_run_attempt(
            hermes_bin, fallback_prompt, "terra", "worker-terra", effective_timeout
        ))

    final = attempts[-1]
    duration_seconds = time.monotonic() - started_at
    cleaned_stdout, worker_result = _parse_worker_result(
        final.get("output", ""), final.get("exit_code")
    )
    if final["exit_code"] == 0:
        routing_status = "completed"
    elif final["retryable"]:
        routing_status = "retryable"
    else:
        routing_status = "failed"
    result: Dict[str, Any] = {
        "routing_status": routing_status,
        "tier": final["tier"],
        "route": final["tier"],
        "profile": final["profile"],
        "selection_mode": decision["selection_mode"],
        "matched_rule": decision["matched_rule"],
        "decision_metadata": decision_metadata,
        "session_id": final.get("session_id"),
        "exit_code": final.get("exit_code"),
        "execution_status": "succeeded" if final.get("exit_code") == 0 else "failed",
        "output": cleaned_stdout,
        "worker_result": worker_result,
        "acceptance_check": _build_acceptance_check(
            project_card,
            worker_result,
            final.get("exit_code"),
        ),
        "project_card": project_card,
        "attempts": attempts,
        "fallback_from": fallback_from,
        "fallback_reason": fallback_reason,
    }
    if final.get("stderr"):
        result["stderr"] = final["stderr"]
    if final.get("error"):
        result["error"] = final["error"]
    if final.get("exit_code") == "timeout":
        result["partial_output"] = _safe_text(
            f"{final.get('output', '')}{final.get('stderr', '')}"
        )
    logger.info(
        "cost_router.result tier=%s route=%s profile=%s selection_mode=%s matched_rule=%s worker_session=%s exit_code=%s duration=%.2fs output_chars=%d stderr_chars=%d timeout=%s task_type=%s",
        final["tier"],
        final["tier"],
        final["profile"],
        decision["selection_mode"],
        decision["matched_rule"],
        final.get("session_id") or "",
        final.get("exit_code"),
        duration_seconds,
        len(cleaned_stdout or ""),
        len(final.get("stderr", "")),
        effective_timeout,
        task_type or "auto",
    )
    return _safe_json(result)


COST_ROUTER_SCHEMA = {
    "name": "cost_router",
    "description": (
        "Route one bounded, non-trivial worker slice after first translating the request into a project card. "
        "If mandatory split triggers are found, returns routing_status=split_required and does not dispatch; the controller must split before routing. "
        "Simple one-shot requests remain synchronous and include project_card.no_split_reason. "
        "Use Luna for lightweight routing/classification/cleanup, Luna Economy for bulk low-risk preprocessing, Terra for default production work, "
        "and Sol for final review, deep reasoning, conflicts, and system architecture. "
        "The controller keeps final judgment and final user-facing synthesis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Self-contained worker task. Include the concrete deliverable and constraints.",
            },
            "context": {
                "type": "string",
                "description": "Optional background the worker needs: file paths, snippets, evidence, output language/tone.",
            },
            "route": {
                "type": "string",
                "enum": [
                    "luna", "luna_economy", "terra", "sol",
                    "worker-luna", "worker-luna-economy", "worker-terra", "worker-sol",
                ],
                "description": "Optional explicit tier/profile. Omit for task-shape routing.",
            },
            "task_type": {
                "type": "string",
                "description": "Optional task type such as router, classify, dedupe, rag_answer, draft, coding, final_review, architecture.",
            },
            "project": {
                "type": "object",
                "description": "Optional structured project-card metadata. Supported routing fields: route or route_candidate, task_type, allow_fallback, project_id/id, slice_id. Values are not treated as credentials.",
            },
            "parent_session_id": {
                "type": "string",
                "description": "Optional non-secret parent session identifier included in decision telemetry for correlation.",
            },
            "allow_fallback": {
                "type": "boolean",
                "description": "Allow one visible Luna/Luna Economy to Terra fallback for retryable infrastructure failures. Default false.",
                "default": False,
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait for the worker profile. Default 600; capped at 1800.",
            },
            "background": {
                "type": "boolean",
                "description": "Reserved for future async support; currently must be false/omitted.",
                "default": False,
            },
        },
        "required": ["goal"],
    },
}


registry.register(
    name="cost_router",
    toolset="delegation",
    schema=COST_ROUTER_SCHEMA,
    handler=lambda args, **kw: cost_router(
        goal=args.get("goal"),
        context=args.get("context"),
        route=args.get("route"),
        task_type=args.get("task_type"),
        project=args.get("project"),
        parent_session_id=args.get("parent_session_id"),
        allow_fallback=args.get("allow_fallback"),
        timeout=args.get("timeout"),
        background=args.get("background"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_cost_router_requirements,
    emoji="🧭",
)
