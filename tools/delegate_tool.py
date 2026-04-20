#!/usr/bin/env python3
"""
Delegate Tool -- Subagent Architecture

Spawns child AIAgent instances with isolated context, restricted toolsets,
and their own terminal sessions. Supports single-task and batch (parallel)
modes. The parent blocks until all children complete.

Each child gets:
  - A fresh conversation (no parent history)
  - Its own task_id (own terminal session, file ops cache)
  - A restricted toolset (configurable, with blocked tools always stripped)
  - A focused system prompt built from the delegated goal + context

The parent's context only sees the delegation call and the summary result,
never the child's intermediate tool calls or reasoning.
"""

import json
import logging
logger = logging.getLogger(__name__)
import os
import shlex
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.archetypes import resolve_archetype, resolve_archetype_defaults, resolve_specialist_mapping
from agent.continuation_engine import apply_bounded_continuation_engine, build_continuation_snapshot
from agent.prompt_builder import build_wave1_overlay_prompt_from_normalized, normalize_wave1_overlay_inputs
from agent.route_categories import BUILTIN_ROUTE_CATEGORIES, DEFAULT_ROUTE_CATEGORY
from agent.runtime_modes import DEFAULT_RUNTIME_MODE_NAME, resolve_runtime_mode
from agent.task_contracts import (
    build_named_workflow_artifact,
    validate_named_workflow_artifact,
    validate_task_contract,
)
from agent.task_store import TaskStatus, TaskStore
from tools.background_delegate_tools import (
    BackgroundDelegateLaunchError,
    launch_background_delegate_task,
)
from toolsets import TOOLSETS


# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset([
    "delegate_task",   # no recursive delegation
    "clarify",         # no user interaction
    "memory",          # no writes to shared MEMORY.md
    "send_message",    # no cross-platform side effects
    "execute_code",    # children should reason step-by-step, not write scripts
])

# Build a description fragment listing toolsets available for subagents.
# Excludes toolsets where ALL tools are blocked, composite/platform toolsets
# (hermes-* prefixed), and scenario toolsets.
_EXCLUDED_TOOLSET_NAMES = frozenset({"debugging", "safe", "delegation", "moa", "rl"})
_SUBAGENT_TOOLSETS = sorted(
    name for name, defn in TOOLSETS.items()
    if name not in _EXCLUDED_TOOLSET_NAMES
    and not name.startswith("hermes-")
    and not all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
)
_TOOLSET_LIST_STR = ", ".join(f"'{n}'" for n in _SUBAGENT_TOOLSETS)

_DEFAULT_MAX_CONCURRENT_CHILDREN = 3
MAX_DEPTH = 2  # parent (0) -> child (1) -> grandchild rejected (2)


def _get_max_concurrent_children() -> int:
    """Read delegation.max_concurrent_children from config, falling back to
    DELEGATION_MAX_CONCURRENT_CHILDREN env var, then the default (3).

    Uses the same ``_load_config()`` path that the rest of ``delegate_task``
    uses, keeping config priority consistent (config.yaml > env > default).
    """
    cfg = _load_config()
    val = cfg.get("max_concurrent_children")
    if val is not None:
        try:
            return max(1, int(val))
        except (TypeError, ValueError):
            logger.warning(
                "delegation.max_concurrent_children=%r is not a valid integer; "
                "using default %d", val, _DEFAULT_MAX_CONCURRENT_CHILDREN,
            )
    env_val = os.getenv("DELEGATION_MAX_CONCURRENT_CHILDREN")
    if env_val:
        try:
            return max(1, int(env_val))
        except (TypeError, ValueError):
            pass
    return _DEFAULT_MAX_CONCURRENT_CHILDREN
DEFAULT_MAX_ITERATIONS = 50
_HEARTBEAT_INTERVAL = 30  # seconds between parent activity heartbeats during delegation
DEFAULT_TOOLSETS = ["terminal", "file", "web"]
DEFAULT_DELEGATION_PROFILE = "general"
_REVIEWER_ARCHETYPES = frozenset({"verifier"})
_REVIEWER_SPECIALISTS = frozenset({"code_reviewer", "qa_guard"})
_REVIEWER_DELEGATION_PROFILES = frozenset({"verification"})
_REVIEWER_READ_ONLY_TOOLS = frozenset({
    "browser_console",
    "browser_get_images",
    "browser_navigate",
    "browser_scroll",
    "browser_snapshot",
    "browser_vision",
    "clarify",
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "process",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "task",
    "terminal",
    "vision_analyze",
    "web_extract",
    "web_search",
})
_REVIEWER_MUTATING_REQUIRED_TOOLS = frozenset({
    "write_file",
    "patch",
    "memory",
    "send_message",
})
DEFAULT_DELEGATION_PROFILES = {
    "general": {"max_concurrent_children": _DEFAULT_MAX_CONCURRENT_CHILDREN},
    "research": {
        "max_concurrent_children": 3,
        "max_iterations": 25,
        "enabled_tools": [
            "read_file", "search_files", "session_search", "skills_list", "skill_view",
            "web_search", "web_extract", "browser_navigate", "browser_snapshot",
            "browser_console", "browser_scroll", "browser_get_images", "vision_analyze",
            "browser_vision",
        ],
    },
    "implementation": {
        "max_concurrent_children": 2,
        "max_iterations": 35,
        "toolsets": ["terminal", "file"],
    },
    "verification": {
        "max_concurrent_children": 3,
        "max_iterations": 20,
        "toolsets": ["terminal", "file", "web"],
    },
}
# Back-compat export for pre-Wave-1 tests/imports.
DEFAULT_DELEGATION_CATEGORY = DEFAULT_DELEGATION_PROFILE
DEFAULT_CATEGORY_PROFILES = DEFAULT_DELEGATION_PROFILES


def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    wave1_overlay_prompt: Optional[str] = None,
    workspace_path: Optional[str] = None,
    named_workflow: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a focused system prompt for a child agent."""
    parts = [
        "You are a focused subagent working on a specific delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")
    if wave1_overlay_prompt and wave1_overlay_prompt.strip():
        parts.append(f"\nWAVE 1 DELEGATION INPUTS:\n{wave1_overlay_prompt.strip()}")
    if isinstance(named_workflow, dict) and named_workflow:
        parts.append(
            "\nNAMED WORKFLOW ARTIFACT:\n"
            f"{json.dumps(named_workflow, indent=2, ensure_ascii=False)}\n"
            "Consume this structured artifact behaviorally. If it includes an execution_task_contract, follow that contract before freeform execution."
        )
    if workspace_path and str(workspace_path).strip():
        parts.append(
            "\nWORKSPACE PATH:\n"
            f"{workspace_path}\n"
            "Use this exact path for local repository/workdir operations unless the task explicitly says otherwise."
        )
    parts.append(
        "\nComplete this task using the tools available to you. "
        "When finished, provide a clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered\n\n"
        "Important workspace rule: Never assume a repository lives at /workspace/... or any other container-style path unless the task/context explicitly gives that path. "
        "If no exact local path is provided, discover it first before issuing git/workdir-specific commands.\n\n"
        "Be thorough but concise -- your response is returned to the "
        "parent agent as a summary."
    )
    return "\n".join(parts)


def _resolve_workspace_hint(parent_agent) -> Optional[str]:
    """Best-effort local workspace hint for child prompts.

    We only inject a path when we have a concrete absolute directory. This avoids
    teaching subagents a fake container path while still helping them avoid
    guessing `/workspace/...` for local repo tasks.
    """
    candidates = [
        os.getenv("TERMINAL_CWD"),
        getattr(getattr(parent_agent, "_subdirectory_hints", None), "working_dir", None),
        getattr(parent_agent, "terminal_cwd", None),
        getattr(parent_agent, "cwd", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            text = os.path.abspath(os.path.expanduser(str(candidate)))
        except Exception:
            continue
        if os.path.isabs(text) and os.path.isdir(text):
            return text
    return None


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets that contain only blocked tools."""
    blocked_toolset_names = {
        "delegation", "clarify", "memory", "code_execution",
    }
    return [t for t in toolsets if t not in blocked_toolset_names]


def _normalize_category_name(value: Optional[str]) -> str:
    if not value or not isinstance(value, str):
        return ""
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_named_string_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _canonicalize_archetype_name(value: Optional[str]) -> str:
    return resolve_archetype(value).name


def _get_parent_runtime_activation_defaults(parent_agent) -> Dict[str, Any]:
    if parent_agent is None:
        return {}

    state = None
    getter = getattr(parent_agent, "get_runtime_activation_state", None)
    if callable(getter):
        try:
            state = getter()
        except Exception as exc:
            logger.debug("Could not read parent runtime activation state via getter: %s", exc)

    if not isinstance(state, dict):
        state = getattr(parent_agent, "runtime_activation_state", None)
    if not isinstance(state, dict):
        state = getattr(parent_agent, "_runtime_activation_state", None)
    if not isinstance(state, dict):
        return {}

    return {
        "specialist": str(state.get("specialist") or "").strip() or None,
        "archetype": str(state.get("archetype") or "").strip() or None,
        "route_category": str(state.get("route_category") or "").strip() or None,
        "delegation_profile": str(state.get("delegation_profile") or "").strip() or None,
        "runtime_mode": str(state.get("runtime_mode") or "").strip() or None,
        "task_contract": state.get("task_contract"),
        "named_workflow": state.get("named_workflow"),
        "activation_applied": bool(state.get("activation_applied")),
    }


def _resolve_contract_tool_requirements(
    task_contract: Optional[Dict[str, Any]],
    *,
    parent_agent=None,
) -> Dict[str, List[str]]:
    if not isinstance(task_contract, dict):
        return {"toolsets": [], "enabled_tools": [], "required_tools": []}

    required_tools = _normalize_named_string_list(task_contract.get("required_tools"))
    if not required_tools:
        return {"toolsets": [], "enabled_tools": [], "required_tools": []}

    parent_tool_names = set(getattr(parent_agent, "valid_tool_names", set()) or [])
    resolved_toolsets: List[str] = []
    resolved_enabled_tools: List[str] = []

    for required in required_tools:
        if required in TOOLSETS:
            resolved_toolsets.append(required)
            toolset_tools = TOOLSETS.get(required, {}).get("tools", [])
            resolved_enabled_tools.extend(
                tool_name
                for tool_name in toolset_tools
                if tool_name not in DELEGATE_BLOCKED_TOOLS
                and (not parent_tool_names or tool_name in parent_tool_names)
            )
            continue

        if required in DELEGATE_BLOCKED_TOOLS:
            continue
        if not parent_tool_names or required in parent_tool_names:
            resolved_enabled_tools.append(required)

    return {
        "toolsets": list(dict.fromkeys(resolved_toolsets)),
        "enabled_tools": list(dict.fromkeys(resolved_enabled_tools)),
        "required_tools": required_tools,
    }


def _is_reviewer_like_resolution(resolved_inputs: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(resolved_inputs, dict):
        return False
    specialist = str(resolved_inputs.get("specialist") or "").strip().lower()
    return bool(specialist in _REVIEWER_SPECIALISTS)


def _apply_named_role_tool_policy(
    *,
    resolved_inputs: Dict[str, Any],
    toolsets: Optional[List[str]],
    enabled_tools: Optional[List[str]],
    parent_agent=None,
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    if not _is_reviewer_like_resolution(resolved_inputs):
        return toolsets, enabled_tools

    contract = resolved_inputs.get("task_contract") if isinstance(resolved_inputs, dict) else None
    required_tools = []
    if isinstance(contract, dict):
        required_tools = _normalize_named_string_list(contract.get("required_tools"))
    forbidden_required = sorted(tool for tool in required_tools if tool in _REVIEWER_MUTATING_REQUIRED_TOOLS)
    if forbidden_required:
        raise ValueError(
            "Reviewer/verifier delegations are read-only and cannot require mutating tools: "
            + ", ".join(forbidden_required)
        )

    parent_tool_names = set(getattr(parent_agent, "valid_tool_names", set()) or [])
    available_read_only = sorted(
        tool_name for tool_name in parent_tool_names if tool_name in _REVIEWER_READ_ONLY_TOOLS
    )

    requested_tools = _normalize_named_string_list(enabled_tools)
    if requested_tools:
        filtered_tools = [tool for tool in requested_tools if tool in _REVIEWER_READ_ONLY_TOOLS]
    else:
        filtered_tools = list(available_read_only)

    if not filtered_tools and available_read_only:
        filtered_tools = list(available_read_only)

    existing_hints = resolved_inputs.get("orchestration_hints")
    resolved_inputs["orchestration_hints"] = dict(existing_hints) if isinstance(existing_hints, dict) else {}
    resolved_inputs["orchestration_hints"].update(
        {
            "behavior_boundary": "reviewer_read_only",
            "completion_gate": "verification_evidence_required",
            "read_only_tools": filtered_tools,
        }
    )

    normalized_toolsets = _normalize_named_string_list(toolsets)
    if "terminal" in filtered_tools and "terminal" not in normalized_toolsets:
        normalized_toolsets.append("terminal")
    if any(tool in filtered_tools for tool in ("read_file", "search_files")) and "file" not in normalized_toolsets:
        normalized_toolsets.append("file")
    if any(tool in filtered_tools for tool in ("web_search", "web_extract", "browser_navigate", "browser_snapshot", "browser_console", "browser_scroll", "browser_get_images", "browser_vision", "vision_analyze")) and "web" not in normalized_toolsets:
        normalized_toolsets.append("web")
    if any(tool in filtered_tools for tool in ("task",)) and "orchestration" not in normalized_toolsets:
        normalized_toolsets.append("orchestration")

    return normalized_toolsets or None, filtered_tools or None


def _apply_named_role_completion_gate(
    entry: Dict[str, Any],
    *,
    resolved_inputs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not _is_reviewer_like_resolution(resolved_inputs):
        return entry

    tool_trace = entry.get("tool_trace") or []
    tool_names = {
        str(item.get("tool") or "").strip()
        for item in tool_trace
        if isinstance(item, dict)
    }
    evidence_tools = tool_names.intersection(_REVIEWER_READ_ONLY_TOOLS)
    if entry.get("status") == "completed" and not evidence_tools:
        entry["status"] = "failed"
        entry["error"] = (
            "Reviewer/verifier completion gate blocked success: no verification evidence tool "
            "was used in this run."
        )
        entry["exit_reason"] = "verification_evidence_missing"
    return entry


def _resolve_route_category_entry(category_name: Optional[str], cfg: Dict[str, Any]) -> Dict[str, str]:
    normalized_name = _normalize_category_name(category_name) or DEFAULT_ROUTE_CATEGORY
    registry = cfg.get("route_categories") if isinstance(cfg, dict) else {}
    if isinstance(registry, dict) and normalized_name in registry and isinstance(registry[normalized_name], dict):
        entry = registry[normalized_name]
        return {
            "name": normalized_name,
            "summary": str(entry.get("summary") or "").strip(),
            "intensity": str(entry.get("intensity") or "").strip(),
        }
    builtin = BUILTIN_ROUTE_CATEGORIES.get(normalized_name)
    if builtin is None:
        raise ValueError(
            f"Unknown route_category '{normalized_name}'. "
            f"Expected one of: {', '.join(sorted(BUILTIN_ROUTE_CATEGORIES))}"
        )
    return {
        "name": builtin.name,
        "summary": builtin.summary,
        "intensity": builtin.intensity,
    }


def _resolve_runtime_mode_entry(
    runtime_mode_name: Optional[str],
    cfg: Dict[str, Any],
) -> Dict[str, str]:
    resolved_builtin = resolve_runtime_mode(runtime_mode_name)
    normalized_name = str(runtime_mode_name or resolved_builtin.name).strip() or resolved_builtin.name
    registry = cfg.get("runtime_modes") if isinstance(cfg, dict) else {}
    if isinstance(registry, dict) and normalized_name in registry and isinstance(registry[normalized_name], dict):
        entry = registry[normalized_name]
        return {
            "name": normalized_name,
            "description": str(entry.get("description") or resolved_builtin.description).strip(),
            "operating_posture": str(entry.get("operating_posture") or resolved_builtin.operating_posture).strip(),
            "kind": str(entry.get("kind") or resolved_builtin.kind).strip() or resolved_builtin.kind,
        }
    return {
        "name": resolved_builtin.name,
        "description": resolved_builtin.description,
        "operating_posture": resolved_builtin.operating_posture,
        "kind": resolved_builtin.kind,
    }


def _resolve_profile_runtime_mode(
    task: Dict[str, Any],
    top_level_runtime_mode: Optional[str],
    cfg: Dict[str, Any],
    delegation_profile: str,
) -> Dict[str, str]:
    explicit_runtime_mode = str(task.get("runtime_mode") or "").strip()
    if explicit_runtime_mode:
        return _resolve_runtime_mode_entry(explicit_runtime_mode, cfg)

    inherited_runtime_mode = str(top_level_runtime_mode or "").strip()
    if inherited_runtime_mode:
        return _resolve_runtime_mode_entry(inherited_runtime_mode, cfg)

    profile = _resolve_category_profile(cfg, delegation_profile)
    profile_runtime_mode = str(profile.get("runtime_mode") or "").strip()
    if profile_runtime_mode:
        return _resolve_runtime_mode_entry(profile_runtime_mode, cfg)

    configured_runtime_mode = str(cfg.get("runtime_mode") or "").strip()
    return _resolve_runtime_mode_entry(configured_runtime_mode or DEFAULT_RUNTIME_MODE_NAME, cfg)


def _resolve_task_delegation_profile_details(
    task: Dict[str, Any],
    top_level_delegation_profile: Optional[str],
    top_level_category: Optional[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    explicit_profile = _normalize_category_name(task.get("delegation_profile"))
    if explicit_profile:
        return {
            "value": explicit_profile,
            "source": "delegation_profile",
            "compatibility_only": False,
            "legacy_category_input": None,
        }

    legacy_task_category = _normalize_category_name(task.get("category"))
    if legacy_task_category:
        return {
            "value": legacy_task_category,
            "source": "legacy_task_category",
            "compatibility_only": True,
            "legacy_category_input": str(task.get("category") or "").strip() or None,
        }

    inherited_profile = _normalize_category_name(top_level_delegation_profile)
    if inherited_profile:
        return {
            "value": inherited_profile,
            "source": "inherited_delegation_profile",
            "compatibility_only": False,
            "legacy_category_input": None,
        }

    inherited_legacy_category = _normalize_category_name(top_level_category)
    if inherited_legacy_category:
        return {
            "value": inherited_legacy_category,
            "source": "inherited_legacy_category",
            "compatibility_only": True,
            "legacy_category_input": str(top_level_category or "").strip() or None,
        }

    configured_profile = _normalize_category_name(cfg.get("default_delegation_profile"))
    if configured_profile:
        return {
            "value": configured_profile,
            "source": "default_delegation_profile",
            "compatibility_only": False,
            "legacy_category_input": None,
        }

    configured_legacy_category = _normalize_category_name(cfg.get("default_category"))
    if configured_legacy_category:
        return {
            "value": configured_legacy_category,
            "source": "default_legacy_category",
            "compatibility_only": True,
            "legacy_category_input": str(cfg.get("default_category") or "").strip() or None,
        }

    return {
        "value": DEFAULT_DELEGATION_CATEGORY,
        "source": "builtin_default_delegation_profile",
        "compatibility_only": False,
        "legacy_category_input": None,
    }


def _resolve_task_delegation_profile(
    task: Dict[str, Any],
    top_level_delegation_profile: Optional[str],
    top_level_category: Optional[str],
    cfg: Dict[str, Any],
) -> str:
    return _resolve_task_delegation_profile_details(
        task,
        top_level_delegation_profile=top_level_delegation_profile,
        top_level_category=top_level_category,
        cfg=cfg,
    )["value"]


def _resolve_wave1_task_inputs(
    task: Dict[str, Any],
    *,
    cfg: Dict[str, Any],
    top_level_archetype: Optional[str] = None,
    inherited_parent_archetype: Optional[str] = None,
    inherited_parent_specialist: Optional[str] = None,
    top_level_route_category: Optional[str] = None,
    inherited_parent_route_category: Optional[str] = None,
    top_level_delegation_profile: Optional[str] = None,
    inherited_parent_delegation_profile: Optional[str] = None,
    top_level_runtime_mode: Optional[str] = None,
    top_level_skills: Any = None,
    top_level_task_contract: Optional[Dict[str, Any]] = None,
    top_level_named_workflow: Optional[Dict[str, Any]] = None,
    top_level_category: Optional[str] = None,
) -> Dict[str, Any]:
    explicit_task_archetype = str(task.get("archetype") or "").strip()
    explicit_task_specialist = str(task.get("specialist") or "").strip()
    inherited_named_workflow = top_level_named_workflow if isinstance(top_level_named_workflow, dict) else None
    explicit_top_level_archetype = str(top_level_archetype or "").strip()
    inherited_archetype = str(inherited_parent_archetype or "").strip()
    requested_specialist = explicit_task_specialist or str(inherited_parent_specialist or "").strip()
    specialist_mapping = resolve_specialist_mapping(requested_specialist)
    resolved_specialist = specialist_mapping.name if specialist_mapping is not None else None
    if specialist_mapping is not None and not explicit_task_archetype and not explicit_top_level_archetype:
        inherited_archetype = specialist_mapping.archetype_name
    configured_archetype = str(cfg.get("archetype") or "").strip()
    requested_archetype = (
        explicit_task_archetype
        or explicit_top_level_archetype
        or inherited_archetype
        or configured_archetype
    )
    archetype_override_present = bool(explicit_task_archetype or explicit_top_level_archetype)
    resolved_archetype = _canonicalize_archetype_name(requested_archetype)
    archetype_defaults = resolve_archetype_defaults(resolved_archetype)

    if archetype_override_present:
        default_route_category = str(archetype_defaults.get("default_route_category") or DEFAULT_ROUTE_CATEGORY).strip()
        default_delegation_profile = _normalize_category_name(archetype_defaults.get("default_delegation_profile"))
        default_skills = _normalize_named_string_list(archetype_defaults.get("default_skills"))
    else:
        default_route_category = str(
            cfg.get("route_category")
            or cfg.get("default_route_category")
            or archetype_defaults.get("default_route_category")
            or DEFAULT_ROUTE_CATEGORY
        ).strip()
        default_delegation_profile = _normalize_category_name(
            cfg.get("default_delegation_profile")
            or archetype_defaults.get("default_delegation_profile")
        )
        default_skills = _normalize_named_string_list(
            cfg.get("default_skills") or archetype_defaults.get("default_skills")
        )

    explicit_route_category = _normalize_category_name(task.get("route_category"))
    explicit_top_level_route_category = _normalize_category_name(top_level_route_category)
    inherited_route_category = _normalize_category_name(inherited_parent_route_category)
    resolved_route_category_name = (
        explicit_route_category
        or explicit_top_level_route_category
        or ("" if archetype_override_present else inherited_route_category)
        or _normalize_category_name(default_route_category)
        or DEFAULT_ROUTE_CATEGORY
    )
    resolved_route_category = _resolve_route_category_entry(resolved_route_category_name, cfg)

    delegation_profile_resolution = _resolve_task_delegation_profile_details(
        task,
        top_level_delegation_profile=top_level_delegation_profile,
        top_level_category=None if archetype_override_present else top_level_category,
        cfg={
            **cfg,
            "default_delegation_profile": default_delegation_profile or cfg.get("default_delegation_profile"),
        },
    )
    if (
        not top_level_delegation_profile
        and not archetype_override_present
        and inherited_parent_delegation_profile
        and not _normalize_category_name(task.get("category"))
        and not _normalize_category_name(top_level_category)
    ):
        delegation_profile_resolution = _resolve_task_delegation_profile_details(
            task,
            top_level_delegation_profile=inherited_parent_delegation_profile,
            top_level_category=top_level_category,
            cfg={
                **cfg,
                "default_delegation_profile": default_delegation_profile or cfg.get("default_delegation_profile"),
            },
        )
    resolved_delegation_profile = delegation_profile_resolution["value"]

    resolved_skills = list(default_skills)
    resolved_skills.extend(_normalize_named_string_list(top_level_skills))
    resolved_skills.extend(_normalize_named_string_list(task.get("skills")))
    resolved_skills = list(dict.fromkeys(skill for skill in resolved_skills if skill))

    raw_task_contract = task.get("task_contract")
    if raw_task_contract is None:
        raw_task_contract = top_level_task_contract if top_level_task_contract is not None else cfg.get("task_contract")
    resolved_task_contract = None
    if raw_task_contract not in (None, "", {}):
        try:
            resolved_task_contract = validate_task_contract(raw_task_contract).model_dump()
        except Exception as exc:
            raise ValueError(f"Invalid task_contract for delegated task '{task.get('goal', '')}': {exc}") from exc

    explicit_named_workflow = task.get("named_workflow")
    resolved_named_workflow = None
    for source_name, candidate_named_workflow in (
        ("explicit", explicit_named_workflow),
        ("inherited", inherited_named_workflow),
    ):
        if not isinstance(candidate_named_workflow, dict) or not candidate_named_workflow:
            continue
        try:
            resolved_named_workflow = validate_named_workflow_artifact(candidate_named_workflow).model_dump(by_alias=True)
            break
        except Exception as exc:
            if source_name == "explicit":
                raise ValueError(f"Invalid named_workflow for delegated task '{task.get('goal', '')}': {exc}") from exc
            logger.debug(
                "Ignoring invalid inherited named_workflow for delegated task '%s': %s",
                task.get("goal", ""),
                exc,
            )

    resolved_runtime_mode = _resolve_profile_runtime_mode(
        task,
        top_level_runtime_mode=top_level_runtime_mode,
        cfg=cfg,
        delegation_profile=resolved_delegation_profile,
    )
    orchestration_hints = {
        "permission_preset": str(cfg.get("permission_preset") or archetype_defaults.get("permission_preset") or "inherit").strip(),
        "fallback_policy": str(cfg.get("fallback_policy") or archetype_defaults.get("fallback_policy") or "legacy_default_mapping").strip(),
    }
    normalized_overlay_inputs = normalize_wave1_overlay_inputs(
        archetype_name=resolved_archetype,
        route_category=resolved_route_category,
        delegation_profile=resolved_delegation_profile,
        runtime_mode=resolved_runtime_mode,
        skills=resolved_skills,
        task_contract=resolved_task_contract,
        orchestration_hints=orchestration_hints,
    )
    if resolved_named_workflow is None:
        resolved_named_workflow = build_named_workflow_artifact(
            objective=str(task.get("goal") or "").strip(),
            specialist=resolved_specialist,
            archetype=normalized_overlay_inputs["archetype"],
            route_category=normalized_overlay_inputs["route_category"],
            runtime_mode=normalized_overlay_inputs["runtime_mode"],
            delegation_profile=normalized_overlay_inputs["delegation_profile"],
            task_contract=normalized_overlay_inputs["task_contract"],
        )
    overlay_prompt = build_wave1_overlay_prompt_from_normalized(normalized_overlay_inputs)

    return {
        "specialist": resolved_specialist,
        "archetype": normalized_overlay_inputs["archetype"],
        "route_category": normalized_overlay_inputs["route_category"],
        "route_category_definition": normalized_overlay_inputs["route_category_definition"],
        "route_category_source": "explicit_or_inherited_route_category" if (
            explicit_route_category or explicit_top_level_route_category or inherited_route_category
        ) else "default_route_category",
        "delegation_profile": normalized_overlay_inputs["delegation_profile"],
        "delegation_profile_source": delegation_profile_resolution["source"],
        "delegation_profile_compatibility_only": bool(delegation_profile_resolution["compatibility_only"]),
        "legacy_category_input": delegation_profile_resolution["legacy_category_input"],
        "skills": normalized_overlay_inputs["skills"],
        "task_contract": normalized_overlay_inputs["task_contract"],
        "named_workflow": resolved_named_workflow,
        "runtime_mode": normalized_overlay_inputs["runtime_mode"],
        "runtime_mode_definition": normalized_overlay_inputs["runtime_mode_definition"],
        "permission_preset": orchestration_hints["permission_preset"],
        "fallback_policy": orchestration_hints["fallback_policy"],
        "orchestration_hints": orchestration_hints,
        "overlay_prompt": overlay_prompt,
    }


def _normalize_category_profile(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    profile: Dict[str, Any] = {}
    if isinstance(raw.get("toolsets"), list):
        profile["toolsets"] = [str(t).strip() for t in raw["toolsets"] if str(t).strip()]
    if isinstance(raw.get("enabled_tools"), list):
        profile["enabled_tools"] = [str(t).strip() for t in raw["enabled_tools"] if str(t).strip()]
    for key in ("model", "provider", "base_url", "api_key", "reasoning_effort", "acp_command", "routing_description"):
        if isinstance(raw.get(key), str) and raw.get(key).strip():
            profile[key] = raw.get(key).strip()
    runtime_mode = str(raw.get("runtime_mode") or "").strip()
    if runtime_mode:
        profile["runtime_mode"] = _resolve_runtime_mode_entry(runtime_mode, {}).get("name", runtime_mode)
    if isinstance(raw.get("acp_args"), list):
        profile["acp_args"] = [str(v) for v in raw["acp_args"]]
    for key in ("max_iterations", "max_concurrent_children"):
        if raw.get(key) is not None:
            try:
                profile[key] = max(1, int(raw.get(key)))
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid delegation profile %s=%r", key, raw.get(key))
    return profile


def _merge_category_profile(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if value in (None, "", [], {}):
            continue
        merged[key] = list(value) if isinstance(value, list) else value
    return merged


def _normalize_delegation_config(cfg: Any) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        cfg = {}
    normalized = dict(cfg)
    profiles = {name: dict(profile) for name, profile in DEFAULT_DELEGATION_PROFILES.items()}

    raw_profiles = cfg.get("delegation_profiles") if isinstance(cfg, dict) else None
    if isinstance(raw_profiles, dict):
        for name, raw_profile in raw_profiles.items():
            normalized_name = _normalize_category_name(name)
            if not normalized_name:
                continue
            profiles[normalized_name] = _merge_category_profile(
                profiles.get(normalized_name, {}),
                _normalize_category_profile(raw_profile),
            )

    raw_categories = cfg.get("categories") if isinstance(cfg, dict) else None
    if isinstance(raw_categories, dict):
        for name, raw_profile in raw_categories.items():
            normalized_name = _normalize_category_name(name)
            if not normalized_name:
                continue
            profiles[normalized_name] = _merge_category_profile(
                profiles.get(normalized_name, {}),
                _normalize_category_profile(raw_profile),
            )

    normalized["delegation_profiles"] = profiles
    normalized["categories"] = profiles

    default_profile = _normalize_category_name(cfg.get("default_delegation_profile")) if isinstance(cfg, dict) else ""
    default_category = _normalize_category_name(cfg.get("default_category")) if isinstance(cfg, dict) else ""
    resolved_default_profile = default_profile or default_category or DEFAULT_DELEGATION_PROFILE
    normalized["default_delegation_profile"] = resolved_default_profile
    normalized["default_category"] = resolved_default_profile
    return normalized


def _resolve_task_category(
    task: Dict[str, Any],
    top_level_category: Optional[str],
    cfg: Dict[str, Any],
    top_level_delegation_profile: Optional[str] = None,
) -> str:
    return _resolve_task_delegation_profile(
        task,
        top_level_delegation_profile=top_level_delegation_profile,
        top_level_category=top_level_category,
        cfg=cfg,
    )


def _resolve_category_profile(cfg: Dict[str, Any], category: str) -> Dict[str, Any]:
    profiles = {}
    if isinstance(cfg, dict):
        profiles = cfg.get("delegation_profiles") or cfg.get("categories") or {}
    profile = profiles.get(category) if isinstance(profiles, dict) else None
    if profile:
        return _merge_category_profile({}, profile)
    fallback_name = cfg.get("default_delegation_profile") or DEFAULT_DELEGATION_PROFILE if isinstance(cfg, dict) else DEFAULT_DELEGATION_PROFILE
    if isinstance(profiles, dict):
        return _merge_category_profile({}, profiles.get(fallback_name, {}) or profiles.get(DEFAULT_DELEGATION_PROFILE, {}))
    return {}


def _enforce_category_concurrency(
    task_list: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    top_level_category: Optional[str] = None,
    top_level_delegation_profile: Optional[str] = None,
) -> Optional[str]:
    categories = [
        _resolve_task_category(task, top_level_category, cfg, top_level_delegation_profile)
        for task in task_list
    ]
    counts = Counter(categories)
    for category, count in counts.items():
        profile = _resolve_category_profile(cfg, category)
        cap = profile.get("max_concurrent_children")
        if cap is None:
            continue
        try:
            cap_int = max(1, int(cap))
        except (TypeError, ValueError):
            continue
        if count > cap_int:
            return (
                f"Too many '{category}' delegation tasks: {count} provided, but "
                f"category policy caps this category at {cap_int}."
            )
    return None


def _resolve_batch_concurrency_limit(
    task_list: List[Dict[str, Any]],
    top_level_category: Optional[str],
    cfg: Dict[str, Any],
    top_level_delegation_profile: Optional[str] = None,
) -> tuple[int, Optional[str]]:
    """Return the effective concurrency limit for one delegation batch.

    The root ``max_concurrent_children`` remains the default/global guardrail.
    When every task is explicitly assigned to the same category, that category's
    higher concurrency cap may override the root default for that batch.
    """
    base_limit = _get_max_concurrent_children()
    if not task_list:
        return base_limit, None

    resolved_categories = [
        _resolve_task_category(task, top_level_category, cfg, top_level_delegation_profile)
        for task in task_list
    ]
    if len(set(resolved_categories)) != 1:
        return base_limit, None

    category_name = resolved_categories[0]
    explicit_top_level = _normalize_category_name(top_level_delegation_profile) or _normalize_category_name(top_level_category)
    explicit_per_task = all(
        (
            _normalize_category_name(task.get("delegation_profile"))
            or _normalize_category_name(task.get("category"))
        ) == category_name
        for task in task_list
    )
    if explicit_top_level != category_name and not explicit_per_task:
        return base_limit, None

    profile = _resolve_category_profile(cfg, category_name)
    category_limit = profile.get("max_concurrent_children")
    try:
        category_limit = max(1, int(category_limit))
    except (TypeError, ValueError):
        return base_limit, None

    if category_limit > base_limit:
        return category_limit, category_name
    return base_limit, None


def _build_child_progress_callback(task_index: int, goal: str, parent_agent, task_count: int = 1) -> Optional[callable]:
    """Build a callback that relays child agent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    Returns None if no display mechanism is available, in which case the
    child agent runs with no progress callback (identical to current behavior).
    """
    spinner = getattr(parent_agent, '_delegate_spinner', None)
    parent_cb = getattr(parent_agent, 'tool_progress_callback', None)

    if not spinner and not parent_cb:
        return None  # No display → no callback → zero behavior change

    # Show 1-indexed prefix only in batch mode (multiple tasks)
    prefix = f"[{task_index + 1}] " if task_count > 1 else ""
    goal_label = (goal or "").strip()

    # Gateway: batch tool names, flush periodically
    _BATCH_SIZE = 5
    _batch: List[str] = []

    def _relay(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
        if not parent_cb:
            return
        try:
            parent_cb(
                event_type,
                tool_name,
                preview,
                args,
                task_index=task_index,
                task_count=task_count,
                goal=goal_label,
                **kwargs,
            )
        except Exception as e:
            logger.debug("Parent callback failed: %s", e)

    def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
        # event_type is one of: "tool.started", "tool.completed",
        # "reasoning.available", "_thinking", "subagent.*"

        if event_type == "subagent.start":
            if spinner and goal_label:
                short = (goal_label[:55] + "...") if len(goal_label) > 55 else goal_label
                try:
                    spinner.print_above(f" {prefix}├─ 🔀 {short}")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.start", preview=preview or goal_label or "", **kwargs)
            return

        if event_type == "subagent.complete":
            _relay("subagent.complete", preview=preview, **kwargs)
            return

        # "_thinking" / reasoning events
        if event_type in ("_thinking", "reasoning.available"):
            text = preview or tool_name or ""
            if spinner:
                short = (text[:55] + "...") if len(text) > 55 else text
                try:
                    spinner.print_above(f" {prefix}├─ 💭 \"{short}\"")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.thinking", preview=text)
            return

        # tool.completed — no display needed here (spinner shows on started)
        if event_type == "tool.completed":
            return

        # tool.started — display and batch for parent relay
        if spinner:
            short = (preview[:35] + "...") if preview and len(preview) > 35 else (preview or "")
            from agent.display import get_tool_emoji
            emoji = get_tool_emoji(tool_name or "")
            line = f" {prefix}├─ {emoji} {tool_name}"
            if short:
                line += f"  \"{short}\""
            try:
                spinner.print_above(line)
            except Exception as e:
                logger.debug("Spinner print_above failed: %s", e)

        if parent_cb:
            _relay("subagent.tool", tool_name, preview, args)
            _batch.append(tool_name or "")
            if len(_batch) >= _BATCH_SIZE:
                summary = ", ".join(_batch)
                _relay("subagent.progress", preview=f"🔀 {prefix}{summary}")
                _batch.clear()

    def _flush():
        """Flush remaining batched tool names to gateway on completion."""
        if parent_cb and _batch:
            summary = ", ".join(_batch)
            _relay("subagent.progress", preview=f"🔀 {prefix}{summary}")
            _batch.clear()

    _callback._flush = _flush
    return _callback


def _build_child_agent(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    enabled_tools: Optional[List[str]] = None,
    model: Optional[str] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    task_count: int = 1,
    parent_agent=None,
    # Credential overrides from delegation config (provider:model resolution)
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    # ACP transport overrides — lets a non-ACP parent spawn ACP child agents
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
    wave1_overlay_prompt: Optional[str] = None,
    delegate_resolution: Optional[Dict[str, Any]] = None,
):
    """
    Build a child AIAgent on the main thread (thread-safe construction).
    Returns the constructed child agent without running it.

    When override_* params are set (from delegation config), the child uses
    those credentials instead of inheriting from the parent.  This enables
    routing subagents to a different provider:model pair (e.g. cheap/fast
    model on OpenRouter while the parent runs on Nous Portal).
    """
    from run_agent import AIAgent

    # When no explicit toolsets given, inherit from parent's enabled toolsets
    # so disabled tools (e.g. web) don't leak to subagents.
    # Note: enabled_toolsets=None means "all tools enabled" (the default),
    # so we must derive effective toolsets from the parent's loaded tools.
    parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
    if parent_enabled is not None:
        parent_toolsets = set(parent_enabled)
    elif parent_agent and hasattr(parent_agent, "valid_tool_names"):
        # enabled_toolsets is None (all tools) — derive from loaded tool names
        import model_tools
        parent_toolsets = {
            ts for name in parent_agent.valid_tool_names
            if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        parent_toolsets = set(DEFAULT_TOOLSETS)

    if toolsets:
        # Intersect with parent — subagent must not gain tools the parent lacks
        child_toolsets = _strip_blocked_tools([t for t in toolsets if t in parent_toolsets])
    elif parent_agent and parent_enabled is not None:
        child_toolsets = _strip_blocked_tools(parent_enabled)
    elif parent_toolsets:
        child_toolsets = _strip_blocked_tools(sorted(parent_toolsets))
    else:
        child_toolsets = _strip_blocked_tools(DEFAULT_TOOLSETS)

    child_enabled_tools = None
    if enabled_tools is not None:
        parent_tool_names = set(getattr(parent_agent, "valid_tool_names", set()) or [])
        child_enabled_tools = sorted({
            str(name).strip()
            for name in enabled_tools
            if isinstance(name, str)
            and str(name).strip()
            and str(name).strip() in parent_tool_names
            and str(name).strip() not in DELEGATE_BLOCKED_TOOLS
        })

    workspace_hint = _resolve_workspace_hint(parent_agent)
    resolved_named_workflow = None
    if isinstance(delegate_resolution, dict):
        candidate_named_workflow = delegate_resolution.get("named_workflow")
        if isinstance(candidate_named_workflow, dict) and candidate_named_workflow:
            try:
                resolved_named_workflow = validate_named_workflow_artifact(candidate_named_workflow).model_dump(by_alias=True)
            except Exception as exc:
                logger.debug("Ignoring invalid delegate_resolution named_workflow while building child prompt: %s", exc)
    child_prompt = _build_child_system_prompt(
        goal,
        context,
        wave1_overlay_prompt=wave1_overlay_prompt,
        workspace_path=workspace_hint,
        named_workflow=resolved_named_workflow,
    )
    # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    # Build progress callback to relay tool calls to parent display
    child_progress_cb = _build_child_progress_callback(task_index, goal, parent_agent, task_count)

    # Each subagent gets its own iteration budget capped at max_iterations
    # (configurable via delegation.max_iterations, default 50).  This means
    # total iterations across parent + subagents can exceed the parent's
    # max_iterations.  The user controls the per-subagent cap in config.yaml.

    child_thinking_cb = None
    if child_progress_cb:
        def _child_thinking(text: str) -> None:
            if not text:
                return
            try:
                child_progress_cb("_thinking", text)
            except Exception as e:
                logger.debug("Child thinking callback relay failed: %s", e)

        child_thinking_cb = _child_thinking

    # Resolve effective credentials: config override > parent inherit
    effective_model = model or parent_agent.model
    effective_provider = override_provider or getattr(parent_agent, "provider", None)
    effective_base_url = override_base_url or parent_agent.base_url
    effective_api_key = override_api_key or parent_api_key
    effective_api_mode = override_api_mode or getattr(parent_agent, "api_mode", None)
    effective_acp_command = override_acp_command or getattr(parent_agent, "acp_command", None)
    effective_acp_args = list(override_acp_args if override_acp_args is not None else (getattr(parent_agent, "acp_args", []) or []))

    # Resolve reasoning config: delegation override > parent inherit
    parent_reasoning = getattr(parent_agent, "reasoning_config", None)
    child_reasoning = parent_reasoning
    try:
        delegation_cfg = _load_config()
        delegation_effort = str(delegation_cfg.get("reasoning_effort") or "").strip()
        if delegation_effort:
            from hermes_constants import parse_reasoning_effort
            parsed = parse_reasoning_effort(delegation_effort)
            if parsed is not None:
                child_reasoning = parsed
            else:
                logger.warning(
                    "Unknown delegation.reasoning_effort '%s', inheriting parent level",
                    delegation_effort,
                )
    except Exception as exc:
        logger.debug("Could not load delegation reasoning_effort: %s", exc)

    child = AIAgent(
        base_url=effective_base_url,
        api_key=effective_api_key,
        model=effective_model,
        provider=effective_provider,
        api_mode=effective_api_mode,
        acp_command=effective_acp_command,
        acp_args=effective_acp_args,
        max_iterations=max_iterations,
        max_tokens=getattr(parent_agent, "max_tokens", None),
        reasoning_config=child_reasoning,
        prefill_messages=getattr(parent_agent, "prefill_messages", None),
        enabled_toolsets=child_toolsets,
        enabled_tools=child_enabled_tools,
        quiet_mode=True,
        ephemeral_system_prompt=child_prompt,
        log_prefix=f"[subagent-{task_index}]",
        platform=parent_agent.platform,
        skip_context_files=True,
        skip_memory=True,
        clarify_callback=None,
        thinking_callback=child_thinking_cb,
        session_db=getattr(parent_agent, '_session_db', None),
        parent_session_id=getattr(parent_agent, 'session_id', None),
        providers_allowed=parent_agent.providers_allowed,
        providers_ignored=parent_agent.providers_ignored,
        providers_order=parent_agent.providers_order,
        provider_sort=parent_agent.provider_sort,
        tool_progress_callback=child_progress_cb,
        iteration_budget=None,  # fresh budget per subagent
    )
    child._print_fn = getattr(parent_agent, '_print_fn', None)
    child._delegate_resolution = dict(delegate_resolution or {})
    # Set delegation depth so children can't spawn grandchildren
    child._delegate_depth = getattr(parent_agent, '_delegate_depth', 0) + 1

    # Share a credential pool with the child when possible so subagents can
    # rotate credentials on rate limits instead of getting pinned to one key.
    child_pool = _resolve_child_credential_pool(effective_provider, parent_agent)
    if child_pool is not None:
        child._credential_pool = child_pool

    # Register child for interrupt propagation
    if hasattr(parent_agent, '_active_children'):
        lock = getattr(parent_agent, '_active_children_lock', None)
        if lock:
            with lock:
                parent_agent._active_children.append(child)
        else:
            parent_agent._active_children.append(child)

    return child

def _run_single_child(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Run a pre-built child agent. Called from within a thread.
    Returns a structured result dict.
    """
    child_start = time.monotonic()

    # Get the progress callback from the child agent
    child_progress_cb = getattr(child, 'tool_progress_callback', None)

    # Restore parent tool names using the value saved before child construction
    # mutated the global. This is the correct parent toolset, not the child's.
    import model_tools
    _saved_tool_names = getattr(child, "_delegate_saved_tool_names",
                                list(model_tools._last_resolved_tool_names))

    child_pool = getattr(child, '_credential_pool', None)
    leased_cred_id = None
    if child_pool is not None:
        leased_cred_id = child_pool.acquire_lease()
        if leased_cred_id is not None:
            try:
                leased_entry = child_pool.current()
                if leased_entry is not None and hasattr(child, '_swap_credential'):
                    child._swap_credential(leased_entry)
            except Exception as exc:
                logger.debug("Failed to bind child to leased credential: %s", exc)

    # Heartbeat: periodically propagate child activity to the parent so the
    # gateway inactivity timeout doesn't fire while the subagent is working.
    # Without this, the parent's _last_activity_ts freezes when delegate_task
    # starts and the gateway eventually kills the agent for "no activity".
    _heartbeat_stop = threading.Event()

    def _heartbeat_loop():
        while not _heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
            if parent_agent is None:
                continue
            touch = getattr(parent_agent, '_touch_activity', None)
            if not touch:
                continue
            # Pull detail from the child's own activity tracker
            desc = f"delegate_task: subagent {task_index} working"
            try:
                child_summary = child.get_activity_summary()
                child_tool = child_summary.get("current_tool")
                child_iter = child_summary.get("api_call_count", 0)
                child_max = child_summary.get("max_iterations", 0)
                if child_tool:
                    desc = (f"delegate_task: subagent running {child_tool} "
                            f"(iteration {child_iter}/{child_max})")
                else:
                    child_desc = child_summary.get("last_activity_desc", "")
                    if child_desc:
                        desc = (f"delegate_task: subagent {child_desc} "
                                f"(iteration {child_iter}/{child_max})")
            except Exception:
                pass
            try:
                touch(desc)
            except Exception:
                pass

    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    _heartbeat_thread.start()

    try:
        if child_progress_cb:
            try:
                child_progress_cb("subagent.start", preview=goal)
            except Exception as e:
                logger.debug("Progress callback start failed: %s", e)

        result = child.run_conversation(user_message=goal)
        resolution = getattr(child, "_delegate_resolution", None) or {}
        runtime_mode = resolution.get("runtime_mode") if isinstance(resolution, dict) else None
        continuation_state = apply_bounded_continuation_engine(
            child,
            result,
            runtime_mode=runtime_mode,
        )
        result = continuation_state["result"]
        final_snapshot = continuation_state["snapshot"]

        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, '_flush'):
            try:
                child_progress_cb._flush()
            except Exception as e:
                logger.debug("Progress callback flush failed: %s", e)

        duration = round(time.monotonic() - child_start, 2)

        summary = result.get("final_response") or ""
        completed = result.get("completed", False)
        interrupted = result.get("interrupted", False)
        api_calls = int(result.get("api_calls", 0) or 0)
        api_calls += sum(int(snapshot.get("apiCalls", 0) or 0) for snapshot in continuation_state.get("snapshots", [])[1:])
        outcome_status = str(final_snapshot.get("outcomeStatus") or "").strip().lower()
        has_open_todos = bool(final_snapshot.get("activeTodos"))

        if outcome_status == "completed" and not has_open_todos:
            status = "completed"
        elif outcome_status == "interrupted" or interrupted:
            status = "interrupted"
        elif outcome_status == "failed":
            status = "failed"
        elif has_open_todos:
            status = "interrupted"
        elif summary:
            status = "completed"
        else:
            status = "failed"

        # Build tool trace from conversation messages (already in memory).
        # Uses tool_call_id to correctly pair parallel tool calls with results.
        tool_trace: list[Dict[str, Any]] = []
        trace_by_id: Dict[str, Dict[str, Any]] = {}
        messages = result.get("messages") or []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "assistant":
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function", {})
                        entry_t = {
                            "tool": fn.get("name", "unknown"),
                            "args_bytes": len(fn.get("arguments", "")),
                        }
                        tool_trace.append(entry_t)
                        tc_id = tc.get("id")
                        if tc_id:
                            trace_by_id[tc_id] = entry_t
                elif msg.get("role") == "tool":
                    content = msg.get("content", "")
                    is_error = bool(
                        content and "error" in content[:80].lower()
                    )
                    result_meta = {
                        "result_bytes": len(content),
                        "status": "error" if is_error else "ok",
                    }
                    # Match by tool_call_id for parallel calls
                    tc_id = msg.get("tool_call_id")
                    target = trace_by_id.get(tc_id) if tc_id else None
                    if target is not None:
                        target.update(result_meta)
                    elif tool_trace:
                        # Fallback for messages without tool_call_id
                        tool_trace[-1].update(result_meta)

        # Determine exit reason
        if status == "completed":
            exit_reason = "completed"
        elif continuation_state.get("exhausted"):
            exit_reason = "continuation_exhausted"
        elif outcome_status == "interrupted" or interrupted:
            exit_reason = "interrupted"
        elif completed:
            exit_reason = "completed"
        else:
            exit_reason = "max_iterations"

        # Extract token counts (safe for mock objects)
        _input_tokens = getattr(child, "session_prompt_tokens", 0)
        _output_tokens = getattr(child, "session_completion_tokens", 0)
        _model = getattr(child, "model", None)

        entry: Dict[str, Any] = {
            "task_index": task_index,
            "status": status,
            "summary": summary,
            "api_calls": api_calls,
            "duration_seconds": duration,
            "model": _model if isinstance(_model, str) else None,
            "exit_reason": exit_reason,
            "tokens": {
                "input": _input_tokens if isinstance(_input_tokens, (int, float)) else 0,
                "output": _output_tokens if isinstance(_output_tokens, (int, float)) else 0,
            },
            "tool_trace": tool_trace,
            "orchestration": final_snapshot,
            "continuation": {
                "mode": continuation_state.get("mode"),
                "attempt_count": continuation_state.get("attempt_count"),
                "resume_count": continuation_state.get("resume_count"),
                "final_outcome_status": outcome_status,
                "open_todos": final_snapshot.get("activeTodos") or [],
                "exhausted": bool(continuation_state.get("exhausted")),
            },
        }
        if resolution:
            entry["resolved_inputs"] = resolution
        entry = _apply_named_role_completion_gate(entry, resolved_inputs=resolution)
        if entry.get("status") == "failed" and not entry.get("error"):
            entry["error"] = result.get("error", "Subagent did not produce a response.")
        elif entry.get("status") == "interrupted" and not summary:
            entry["error"] = result.get("error") or "Subagent stopped with unfinished work remaining."

        if child_progress_cb:
            try:
                child_progress_cb(
                    "subagent.complete",
                    preview=summary[:160] if summary else entry.get("error", ""),
                    status=status,
                    duration_seconds=duration,
                    summary=summary[:500] if summary else entry.get("error", ""),
                )
            except Exception as e:
                logger.debug("Progress callback completion failed: %s", e)

        return entry

    except Exception as exc:
        duration = round(time.monotonic() - child_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        if child_progress_cb:
            try:
                child_progress_cb(
                    "subagent.complete",
                    preview=str(exc),
                    status="failed",
                    duration_seconds=duration,
                    summary=str(exc),
                )
            except Exception as e:
                logger.debug("Progress callback failure relay failed: %s", e)
        return {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
        }

    finally:
        # Stop the heartbeat thread so it doesn't keep touching parent activity
        # after the child has finished (or failed).
        _heartbeat_stop.set()
        _heartbeat_thread.join(timeout=5)

        if child_pool is not None and leased_cred_id is not None:
            try:
                child_pool.release_lease(leased_cred_id)
            except Exception as exc:
                logger.debug("Failed to release credential lease: %s", exc)

        # Restore the parent's tool names so the process-global is correct
        # for any subsequent execute_code calls or other consumers.
        import model_tools

        saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
        if isinstance(saved_tool_names, list):
            model_tools._last_resolved_tool_names = list(saved_tool_names)

        # Remove child from active tracking

        # Unregister child from interrupt propagation
        if hasattr(parent_agent, '_active_children'):
            try:
                lock = getattr(parent_agent, '_active_children_lock', None)
                if lock:
                    with lock:
                        parent_agent._active_children.remove(child)
                else:
                    parent_agent._active_children.remove(child)
            except (ValueError, UnboundLocalError) as e:
                logger.debug("Could not remove child from active_children: %s", e)

        # Close tool resources (terminal sandboxes, browser daemons,
        # background processes, httpx clients) so subagent subprocesses
        # don't outlive the delegation.
        try:
            if hasattr(child, 'close'):
                child.close()
        except Exception:
            logger.debug("Failed to close child agent after delegation")


def _build_persistent_launch_spec(
    *,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    enabled_tools: Optional[List[str]],
    resolved_inputs: Dict[str, Any],
    creds: Dict[str, Any],
    max_iterations: int,
    parent_agent,
    acp_command: Optional[str],
    acp_args: Optional[List[str]],
    wave1_overlay_prompt: Optional[str],
) -> Dict[str, Any]:
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    return {
        "runner": "delegate",
        "task_index": 0,
        "goal": goal,
        "context": context,
        "toolsets": list(toolsets or []),
        "enabled_tools": list(enabled_tools or []),
        "model": creds.get("model") or getattr(parent_agent, "model", None),
        "provider": creds.get("provider") or getattr(parent_agent, "provider", None),
        "base_url": creds.get("base_url") or getattr(parent_agent, "base_url", None),
        "api_key": creds.get("api_key") or parent_api_key,
        "api_mode": creds.get("api_mode") or getattr(parent_agent, "api_mode", None),
        "acp_command": acp_command,
        "acp_args": list(acp_args or []),
        "max_iterations": max_iterations,
        "wave1_overlay_prompt": wave1_overlay_prompt,
        "delegate_resolution": dict(resolved_inputs or {}),
        "parent_enabled_toolsets": list(getattr(parent_agent, "enabled_toolsets", None) or []),
        "parent_valid_tool_names": sorted(getattr(parent_agent, "valid_tool_names", set()) or []),
        "platform": getattr(parent_agent, "platform", None),
        "providers_allowed": getattr(parent_agent, "providers_allowed", None),
        "providers_ignored": getattr(parent_agent, "providers_ignored", None),
        "providers_order": getattr(parent_agent, "providers_order", None),
        "provider_sort": getattr(parent_agent, "provider_sort", None),
        "reasoning_config": getattr(parent_agent, "reasoning_config", None),
        "max_tokens": getattr(parent_agent, "max_tokens", None),
        "prefill_messages": getattr(parent_agent, "prefill_messages", None),
        "parent_session_id": getattr(parent_agent, "session_id", None),
    }


def _build_persistent_parent_agent(launch_spec: Dict[str, Any]):
    parent = SimpleNamespace()
    parent.base_url = launch_spec.get("base_url")
    parent.api_key = launch_spec.get("api_key")
    parent.provider = launch_spec.get("provider")
    parent.api_mode = launch_spec.get("api_mode")
    parent.model = launch_spec.get("model")
    parent.platform = launch_spec.get("platform")
    parent.providers_allowed = launch_spec.get("providers_allowed")
    parent.providers_ignored = launch_spec.get("providers_ignored")
    parent.providers_order = launch_spec.get("providers_order")
    parent.provider_sort = launch_spec.get("provider_sort")
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = list(launch_spec.get("parent_enabled_toolsets") or []) or None
    parent.valid_tool_names = set(launch_spec.get("parent_valid_tool_names") or [])
    parent.reasoning_config = launch_spec.get("reasoning_config")
    parent.max_tokens = launch_spec.get("max_tokens")
    parent.prefill_messages = launch_spec.get("prefill_messages")
    parent.session_id = launch_spec.get("parent_session_id")
    parent.acp_command = launch_spec.get("acp_command")
    parent.acp_args = list(launch_spec.get("acp_args") or [])
    return parent


def launch_persistent_delegate_task(task_id: str, *, store: Optional[TaskStore] = None, process_registry_obj=None) -> dict[str, Any]:
    return launch_background_delegate_task(
        task_id,
        store=store,
        process_registry_obj=process_registry_obj,
    )


def run_persistent_delegate_task(task_id: str, store_root: Optional[str] = None) -> None:
    task_store = TaskStore(store_root)
    record = task_store.require_task(task_id)
    launch_spec = dict(record.launch_spec or {})
    if launch_spec.get("runner") != "delegate":
        raise ValueError(f"task {task_id} is not a delegate-backed persistent task")

    task_store.transition_task(task_id, TaskStatus.running)
    parent = _build_persistent_parent_agent(launch_spec)
    try:
        child = _build_child_agent(
            task_index=0,
            goal=str(launch_spec.get("goal") or record.goal),
            context=launch_spec.get("context") or record.context,
            toolsets=launch_spec.get("toolsets"),
            enabled_tools=launch_spec.get("enabled_tools"),
            model=launch_spec.get("model"),
            max_iterations=int(launch_spec.get("max_iterations") or DEFAULT_MAX_ITERATIONS),
            task_count=1,
            parent_agent=parent,
            override_provider=launch_spec.get("provider"),
            override_base_url=launch_spec.get("base_url"),
            override_api_key=launch_spec.get("api_key"),
            override_api_mode=launch_spec.get("api_mode"),
            override_acp_command=launch_spec.get("acp_command"),
            override_acp_args=launch_spec.get("acp_args"),
            wave1_overlay_prompt=launch_spec.get("wave1_overlay_prompt"),
            delegate_resolution=launch_spec.get("delegate_resolution") or record.resolved_inputs,
        )
        result = _run_single_child(0, record.goal, child=child, parent_agent=parent)
        continuation = dict(result.get("continuation") or {})
        orchestration = dict(result.get("orchestration") or {})
        open_todos = continuation.get("open_todos") or orchestration.get("activeTodos") or []
        if result.get("status") == "completed" and not open_todos:
            task_store.clear_continuation(task_id)
            final_status = TaskStatus.completed
        else:
            task_store.update_continuation(
                task_id,
                mode=continuation.get("mode") or record.runtime_mode,
                status="retry_requested" if open_todos else "resolved",
                open_todos=open_todos,
                latest_response_preview=orchestration.get("responsePreview") or result.get("summary"),
                last_outcome_status=continuation.get("final_outcome_status") or orchestration.get("outcomeStatus"),
                resume_count=continuation.get("resume_count"),
                attempt_count=continuation.get("attempt_count"),
            )
            final_status = TaskStatus.failed
        task_store.record_result(
            task_id,
            status=final_status,
            result=result,
            summary=result.get("summary"),
            error=result.get("error") or ("unfinished work remains" if open_todos else None),
        )
    except Exception as exc:
        task_store.record_result(
            task_id,
            status=TaskStatus.failed,
            result={"error": str(exc)},
            summary=None,
            error=str(exc),
        )
        raise


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    category: Optional[str] = None,
    archetype: Optional[str] = None,
    route_category: Optional[str] = None,
    delegation_profile: Optional[str] = None,
    runtime_mode: Optional[str] = None,
    skills: Optional[List[str]] = None,
    task_contract: Optional[Dict[str, Any]] = None,
    max_iterations: Optional[int] = None,
    persistent: bool = False,
    background: bool = False,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
    parent_agent=None,
) -> str:
    """
    Spawn one or more child agents to handle delegated tasks.

    Supports two modes:
      - Single: provide goal (+ optional context, toolsets)
      - Batch:  provide tasks array [{goal, context, toolsets}, ...]

    Returns JSON with results array, one entry per task.
    """
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    # Depth limit
    depth = getattr(parent_agent, '_delegate_depth', 0)
    if depth >= MAX_DEPTH:
        return json.dumps({
            "error": (
                f"Delegation depth limit reached ({MAX_DEPTH}). "
                "Subagents cannot spawn further subagents."
            )
        })

    # Load config
    cfg = _normalize_delegation_config(_load_config())
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    effective_max_iter = max_iterations or default_max_iter

    # Resolve delegation credentials (provider:model pair).
    # When delegation.provider is configured, this resolves the full credential
    # bundle (base_url, api_key, api_mode) via the same runtime provider system
    # used by CLI/gateway startup.  When unconfigured, returns None values so
    # children inherit from the parent.
    try:
        creds = _resolve_delegation_credentials(cfg, parent_agent)
    except ValueError as exc:
        return tool_error(str(exc))

    parent_activation_defaults = _get_parent_runtime_activation_defaults(parent_agent)
    explicit_top_level_archetype = str(archetype or "").strip() or None
    explicit_top_level_route_category = str(route_category or "").strip() or None
    explicit_top_level_delegation_profile = str(delegation_profile or "").strip() or None
    inherited_parent_specialist = parent_activation_defaults.get("specialist")
    inherited_parent_archetype = parent_activation_defaults.get("archetype")
    inherited_parent_route_category = parent_activation_defaults.get("route_category")
    inherited_parent_delegation_profile = parent_activation_defaults.get("delegation_profile")
    effective_archetype = explicit_top_level_archetype or inherited_parent_archetype
    effective_route_category = explicit_top_level_route_category or inherited_parent_route_category
    effective_delegation_profile = explicit_top_level_delegation_profile or inherited_parent_delegation_profile
    effective_runtime_mode = runtime_mode or parent_activation_defaults.get("runtime_mode")
    effective_task_contract = task_contract if task_contract is not None else parent_activation_defaults.get("task_contract")
    effective_named_workflow = parent_activation_defaults.get("named_workflow")
    effective_category = category or effective_delegation_profile

    # Normalize to task list
    max_children, limit_category = _resolve_batch_concurrency_limit(
        task_list=tasks or [],
        top_level_category=effective_category,
        cfg=cfg,
        top_level_delegation_profile=effective_delegation_profile,
    )
    if tasks and isinstance(tasks, list):
        if len(tasks) > max_children:
            if limit_category:
                return tool_error(
                    f"Too many tasks: {len(tasks)} provided, but category '{limit_category}' allows at most "
                    f"{max_children} concurrent tasks in a single batch. Either reduce the task count or increase "
                    f"delegation.categories.{limit_category}.max_concurrent_children in config.yaml."
                )
            return tool_error(
                f"Too many tasks: {len(tasks)} provided, but "
                f"max_concurrent_children is {max_children}. "
                f"Either reduce the task count, split into multiple "
                f"delegate_task calls, or increase "
                f"delegation.max_concurrent_children in config.yaml."
            )
        task_list = tasks
    elif goal and isinstance(goal, str) and goal.strip():
        task_list = [{
            "goal": goal,
            "context": context,
            "toolsets": toolsets,
            "acp_command": acp_command,
            "acp_args": acp_args,
        }]
    else:
        return tool_error("Provide either 'goal' (single task) or 'tasks' (batch).")

    if not task_list:
        return tool_error("No tasks provided.")

    # Validate each task has a goal
    for i, task in enumerate(task_list):
        if not task.get("goal", "").strip():
            return tool_error(f"Task {i} is missing a 'goal'.")

    category_error = _enforce_category_concurrency(
        task_list,
        cfg,
        top_level_category=effective_category,
        top_level_delegation_profile=effective_delegation_profile,
    )
    if category_error:
        return tool_error(category_error)

    overall_start = time.monotonic()
    results = []

    n_tasks = len(task_list)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for t in task_list]

    # Save parent tool names BEFORE any child construction mutates the global.
    # _build_child_agent() calls AIAgent() which calls get_tool_definitions(),
    # which overwrites model_tools._last_resolved_tool_names with child's toolset.
    import model_tools as _model_tools
    _parent_tool_names = list(_model_tools._last_resolved_tool_names)

    # Build all child agents on the main thread (thread-safe construction)
    # Wrapped in try/finally so the global is always restored even if a
    # child build raises (otherwise _last_resolved_tool_names stays corrupted).
    children = []
    try:
        for i, t in enumerate(task_list):
            try:
                resolved_inputs = _resolve_wave1_task_inputs(
                    t,
                    cfg=cfg,
                    top_level_archetype=explicit_top_level_archetype,
                    inherited_parent_archetype=inherited_parent_archetype,
                    inherited_parent_specialist=inherited_parent_specialist,
                    top_level_route_category=explicit_top_level_route_category,
                    inherited_parent_route_category=inherited_parent_route_category,
                    top_level_delegation_profile=explicit_top_level_delegation_profile,
                    inherited_parent_delegation_profile=inherited_parent_delegation_profile,
                    top_level_runtime_mode=effective_runtime_mode,
                    top_level_skills=skills,
                    top_level_task_contract=effective_task_contract,
                    top_level_named_workflow=effective_named_workflow,
                    top_level_category=effective_category,
                )
            except (TypeError, ValueError, KeyError) as exc:
                return tool_error(str(exc))
            resolved_profile = resolved_inputs["delegation_profile"]
            category_profile = _resolve_category_profile(cfg, resolved_profile)
            merged_cfg = _merge_category_profile(cfg, category_profile)
            category_creds = _resolve_delegation_credentials(merged_cfg, parent_agent)
            contract_tool_policy = _resolve_contract_tool_requirements(
                resolved_inputs.get("task_contract"),
                parent_agent=parent_agent,
            )
            task_toolsets = list(dict.fromkeys(
                _normalize_named_string_list(t.get("toolsets") or category_profile.get("toolsets") or toolsets)
                + contract_tool_policy["toolsets"]
            )) or None
            task_enabled_tools = list(dict.fromkeys(
                _normalize_named_string_list(category_profile.get("enabled_tools"))
                + contract_tool_policy["enabled_tools"]
            )) or None
            task_toolsets, task_enabled_tools = _apply_named_role_tool_policy(
                resolved_inputs=resolved_inputs,
                toolsets=task_toolsets,
                enabled_tools=task_enabled_tools,
                parent_agent=parent_agent,
            )
            task_max_iter = int(category_profile.get("max_iterations") or effective_max_iter)
            task_overlay_prompt = build_wave1_overlay_prompt_from_normalized(
                normalize_wave1_overlay_inputs(
                    archetype_name=resolved_inputs["archetype"],
                    route_category=resolved_inputs.get("route_category_definition") or resolved_inputs["route_category"],
                    delegation_profile=resolved_inputs["delegation_profile"],
                    runtime_mode=resolved_inputs.get("runtime_mode_definition") or resolved_inputs["runtime_mode"],
                    skills=resolved_inputs.get("skills"),
                    task_contract=resolved_inputs.get("task_contract"),
                    orchestration_hints=resolved_inputs.get("orchestration_hints"),
                )
            )
            resolved_inputs["overlay_prompt"] = task_overlay_prompt
            child = _build_child_agent(
                task_index=i, goal=t["goal"], context=t.get("context"),
                toolsets=task_toolsets, enabled_tools=task_enabled_tools, model=category_creds["model"] or creds["model"],
                max_iterations=task_max_iter, task_count=n_tasks, parent_agent=parent_agent,
                override_provider=category_creds["provider"] or creds["provider"], override_base_url=category_creds["base_url"] or creds["base_url"],
                override_api_key=category_creds["api_key"] or creds["api_key"],
                override_api_mode=category_creds["api_mode"] or creds["api_mode"],
                override_acp_command=t.get("acp_command") or category_profile.get("acp_command") or acp_command,
                override_acp_args=t.get("acp_args") or category_profile.get("acp_args") or acp_args,
                wave1_overlay_prompt=task_overlay_prompt,
                delegate_resolution={key: value for key, value in resolved_inputs.items() if key != "overlay_prompt"},
            )
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            children.append((i, t, child, task_max_iter, task_overlay_prompt))
    finally:
        # Authoritative restore: reset global to parent's tool names after all children built
        _model_tools._last_resolved_tool_names = _parent_tool_names

    if persistent or background:
        if n_tasks != 1:
            return tool_error("Persistent/background delegation currently supports exactly one task.")
        _i, _t, child, task_max_iter, task_overlay_prompt = children[0]
        delegate_resolution = dict(getattr(child, "_delegate_resolution", {}) or {})
        resolved_hints = dict(delegate_resolution.get("orchestration_hints") or {})
        launch_spec = _build_persistent_launch_spec(
            goal=_t["goal"],
            context=_t.get("context"),
            toolsets=list(getattr(child, "enabled_toolsets", None) or []),
            enabled_tools=list(getattr(child, "enabled_tools", None) or []),
            resolved_inputs=delegate_resolution,
            creds={
                "model": getattr(child, "model", None),
                "provider": getattr(child, "provider", None),
                "base_url": getattr(child, "base_url", None),
                "api_key": getattr(child, "api_key", None),
                "api_mode": getattr(child, "api_mode", None),
            },
            max_iterations=task_max_iter,
            parent_agent=parent_agent,
            acp_command=getattr(child, "acp_command", None),
            acp_args=getattr(child, "acp_args", None),
            wave1_overlay_prompt=task_overlay_prompt,
        )
        store = TaskStore()
        record = store.create_task(
            goal=_t["goal"],
            context=_t.get("context"),
            owner_session_id=getattr(parent_agent, "session_id", None),
            parent_session_id=getattr(parent_agent, "session_id", None),
            archetype=delegate_resolution.get("archetype"),
            specialist=delegate_resolution.get("specialist"),
            route_category=delegate_resolution.get("route_category"),
            delegation_profile=delegate_resolution.get("delegation_profile"),
            runtime_mode=delegate_resolution.get("runtime_mode"),
            skills=list(delegate_resolution.get("skills") or []),
            task_contract=delegate_resolution.get("task_contract"),
            permissions={
                "permission_preset": delegate_resolution.get("permission_preset") or resolved_hints.get("permission_preset") or cfg.get("permission_preset"),
                "fallback_policy": delegate_resolution.get("fallback_policy") or resolved_hints.get("fallback_policy") or cfg.get("fallback_policy"),
            },
            resolved_inputs=delegate_resolution,
            launch_spec=launch_spec,
        )
        if background:
            try:
                launch_result = launch_background_delegate_task(record.id, store=store)
            except BackgroundDelegateLaunchError as exc:
                try:
                    child.close()
                except Exception:
                    logger.debug("Failed to close prebuilt child after background launch failure")
                try:
                    if hasattr(parent_agent, "_active_children") and child in parent_agent._active_children:
                        parent_agent._active_children.remove(child)
                except Exception:
                    pass
                return tool_error(f"Background delegate launch failed: {exc}")
            try:
                child.close()
            except Exception:
                logger.debug("Failed to close prebuilt child after persistent launch")
            try:
                if hasattr(parent_agent, "_active_children") and child in parent_agent._active_children:
                    parent_agent._active_children.remove(child)
            except Exception:
                pass
            return json.dumps(launch_result, ensure_ascii=False)

        store.transition_task(record.id, TaskStatus.queued)
        store.transition_task(record.id, TaskStatus.running)
        try:
            result = _run_single_child(0, _t["goal"], child, parent_agent)
            status = TaskStatus.completed if result.get("status") == "completed" else TaskStatus.failed
            store.record_result(
                record.id,
                status=status,
                result=result,
                summary=result.get("summary"),
                error=result.get("error"),
            )
        except Exception as exc:
            store.record_result(
                record.id,
                status=TaskStatus.failed,
                result={"error": str(exc)},
                summary=None,
                error=str(exc),
            )
            raise
        return json.dumps(
            {
                "task_id": record.id,
                "persistent": True,
                "background": False,
                "results": [result],
                "total_duration_seconds": round(time.monotonic() - overall_start, 2),
            },
            ensure_ascii=False,
        )

    if n_tasks == 1:
        # Single task -- run directly (no thread pool overhead)
        _i, _t, child, _task_max_iter, _task_overlay_prompt = children[0]
        result = _run_single_child(0, _t["goal"], child, parent_agent)
        results.append(result)
    else:
        # Batch -- run in parallel with per-task progress lines
        completed_count = 0
        spinner_ref = getattr(parent_agent, '_delegate_spinner', None)

        with ThreadPoolExecutor(max_workers=min(max_children, n_tasks)) as executor:
            futures = {}
            for i, t, child, _task_max_iter, _task_overlay_prompt in children:
                future = executor.submit(
                    _run_single_child,
                    task_index=i,
                    goal=t["goal"],
                    child=child,
                    parent_agent=parent_agent,
                )
                futures[future] = i

            # Poll futures with interrupt checking.  as_completed() blocks
            # until ALL futures finish — if a child agent gets stuck,
            # the parent blocks forever even after interrupt propagation.
            # Instead, use wait() with a short timeout so we can bail
            # when the parent is interrupted.
            pending = set(futures.keys())
            while pending:
                if getattr(parent_agent, "_interrupt_requested", False) is True:
                    # Parent interrupted — collect whatever finished and
                    # abandon the rest.  Children already received the
                    # interrupt signal; we just can't wait forever.
                    for f in pending:
                        idx = futures[f]
                        if f.done():
                            try:
                                entry = f.result()
                            except Exception as exc:
                                entry = {
                                    "task_index": idx,
                                    "status": "error",
                                    "summary": None,
                                    "error": str(exc),
                                    "api_calls": 0,
                                    "duration_seconds": 0,
                                }
                        else:
                            entry = {
                                "task_index": idx,
                                "status": "interrupted",
                                "summary": None,
                                "error": "Parent agent interrupted — child did not finish in time",
                                "api_calls": 0,
                                "duration_seconds": 0,
                            }
                        results.append(entry)
                        completed_count += 1
                    break

                from concurrent.futures import wait as _cf_wait, FIRST_COMPLETED
                done, pending = _cf_wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        entry = future.result()
                    except Exception as exc:
                        idx = futures[future]
                        entry = {
                            "task_index": idx,
                            "status": "error",
                            "summary": None,
                            "error": str(exc),
                            "api_calls": 0,
                            "duration_seconds": 0,
                        }
                    results.append(entry)
                    completed_count += 1

                    # Print per-task completion line above the spinner
                    idx = entry["task_index"]
                    label = task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
                    dur = entry.get("duration_seconds", 0)
                    status = entry.get("status", "?")
                    icon = "✓" if status == "completed" else "✗"
                    remaining = n_tasks - completed_count
                    completion_line = f"{icon} [{idx+1}/{n_tasks}] {label}  ({dur}s)"
                    if spinner_ref:
                        try:
                            spinner_ref.print_above(completion_line)
                        except Exception:
                            print(f"  {completion_line}")
                    else:
                        print(f"  {completion_line}")

                    # Update spinner text to show remaining count
                    if spinner_ref and remaining > 0:
                        try:
                            spinner_ref.update_text(f"🔀 {remaining} task{'s' if remaining != 1 else ''} remaining")
                        except Exception as e:
                            logger.debug("Spinner update_text failed: %s", e)

        # Sort by task_index so results match input order
        results.sort(key=lambda r: r["task_index"])

    # Notify parent's memory provider of delegation outcomes
    if parent_agent and hasattr(parent_agent, '_memory_manager') and parent_agent._memory_manager:
        for entry in results:
            try:
                _task_goal = task_list[entry["task_index"]]["goal"] if entry["task_index"] < len(task_list) else ""
                parent_agent._memory_manager.on_delegation(
                    task=_task_goal,
                    result=entry.get("summary", "") or "",
                    child_session_id=getattr(children[entry["task_index"]][2], "session_id", "") if entry["task_index"] < len(children) else "",
                )
            except Exception:
                pass

    total_duration = round(time.monotonic() - overall_start, 2)

    return json.dumps({
        "results": results,
        "total_duration_seconds": total_duration,
    }, ensure_ascii=False)


def _resolve_child_credential_pool(effective_provider: Optional[str], parent_agent):
    """Resolve a credential pool for the child agent.

    Rules:
    1. Same provider as the parent -> share the parent's pool so cooldown state
       and rotation stay synchronized.
    2. Different provider -> try to load that provider's own pool.
    3. No pool available -> return None and let the child keep the inherited
       fixed credential behavior.
    """
    if not effective_provider:
        return getattr(parent_agent, "_credential_pool", None)

    parent_provider = getattr(parent_agent, "provider", None) or ""
    parent_pool = getattr(parent_agent, "_credential_pool", None)
    if parent_pool is not None and effective_provider == parent_provider:
        return parent_pool

    try:
        from agent.credential_pool import load_pool
        pool = load_pool(effective_provider)
        if pool is not None and pool.has_credentials():
            return pool
    except Exception as exc:
        logger.debug(
            "Could not load credential pool for child provider '%s': %s",
            effective_provider,
            exc,
        )
    return None


def _resolve_delegation_credentials(cfg: dict, parent_agent) -> dict:
    """Resolve credentials for subagent delegation.

    If ``delegation.base_url`` is configured, subagents use that direct
    OpenAI-compatible endpoint. Otherwise, if ``delegation.provider`` is
    configured, the full credential bundle (base_url, api_key, api_mode,
    provider) is resolved via the runtime provider system — the same path used
    by CLI/gateway startup. This lets subagents run on a completely different
    provider:model pair.

    If neither base_url nor provider is configured, returns None values so the
    child inherits everything from the parent agent.

    Raises ValueError with a user-friendly message on credential failure.
    """
    configured_model = str(cfg.get("model") or "").strip() or None
    configured_provider = str(cfg.get("provider") or "").strip() or None
    configured_base_url = str(cfg.get("base_url") or "").strip() or None
    configured_api_key = str(cfg.get("api_key") or "").strip() or None

    if configured_base_url:
        api_key = (
            configured_api_key
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError(
                "Delegation base_url is configured but no API key was found. "
                "Set delegation.api_key or OPENAI_API_KEY."
            )

        base_lower = configured_base_url.lower()
        provider = "custom"
        api_mode = "chat_completions"
        if "chatgpt.com/backend-api/codex" in base_lower:
            provider = "openai-codex"
            api_mode = "codex_responses"
        elif "api.anthropic.com" in base_lower:
            provider = "anthropic"
            api_mode = "anthropic_messages"

        return {
            "model": configured_model,
            "provider": provider,
            "base_url": configured_base_url,
            "api_key": api_key,
            "api_mode": api_mode,
        }

    if not configured_provider:
        # No provider override — child inherits everything from parent
        return {
            "model": configured_model,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

    parent_provider = str(getattr(parent_agent, "provider", "") or "").strip()
    parent_base_url = str(getattr(parent_agent, "base_url", "") or "").strip()
    parent_api_key = str(getattr(parent_agent, "api_key", "") or "").strip()
    parent_api_mode = str(getattr(parent_agent, "api_mode", "") or "").strip()
    runtime_parent_providers = {
        "openai-codex",
        "nous",
        "qwen-oauth",
        "google-gemini-cli",
        "copilot-acp",
    }
    if (
        configured_provider == parent_provider
        and configured_provider in runtime_parent_providers
        and parent_base_url
        and parent_api_key
    ):
        return {
            "model": configured_model,
            "provider": parent_provider,
            "base_url": parent_base_url,
            "api_key": parent_api_key,
            "api_mode": parent_api_mode or None,
            "command": getattr(parent_agent, "acp_command", None),
            "args": list(getattr(parent_agent, "acp_args", []) or []),
        }

    # Provider is configured — resolve full credentials
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested=configured_provider)
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve delegation provider '{configured_provider}': {exc}. "
            f"Check that the provider is configured (API key set, valid provider name), "
            f"or set delegation.base_url/delegation.api_key for a direct endpoint. "
            f"Available providers: openrouter, nous, zai, kimi-coding, minimax."
        ) from exc

    api_key = runtime.get("api_key", "")
    if not api_key:
        raise ValueError(
            f"Delegation provider '{configured_provider}' resolved but has no API key. "
            f"Set the appropriate environment variable or run 'hermes auth'."
        )

    return {
        "model": configured_model,
        "provider": runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


def _load_config() -> dict:
    """Load delegation config from CLI_CONFIG or persistent config.

    Prefer the live CLI runtime config when it was loaded for the current
    HERMES_HOME. If ``cli`` was imported earlier under a different home/profile
    (common in tests that isolate HERMES_HOME after collection), its module-level
    ``CLI_CONFIG`` is stale and must not leak into delegation.
    """
    try:
        import cli as cli_mod
        from hermes_constants import get_hermes_home

        current_home = Path(get_hermes_home()).resolve()
        cli_home = getattr(cli_mod, "_hermes_home", None)
        cli_home_resolved = Path(cli_home).resolve() if cli_home is not None else None
        if cli_home_resolved == current_home:
            cfg = getattr(cli_mod, "CLI_CONFIG", {}).get("delegation", {})
            if cfg:
                return _normalize_delegation_config(cfg)
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config
        full = load_config()
        return _normalize_delegation_config(full.get("delegation", {}))
    except Exception:
        return _normalize_delegation_config({})


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    "description": (
        "Spawn one or more subagents to work on tasks in isolated contexts. "
        "Each subagent gets its own conversation, terminal session, and toolset. "
        "Only the final summary is returned -- intermediate tool results "
        "never enter your context window.\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        "2. Batch (parallel): provide 'tasks' array with up to 3 items. "
        "All run concurrently and results are returned together.\n\n"
        "WHEN TO USE delegate_task:\n"
        "- Reasoning-heavy subtasks (debugging, code review, research synthesis)\n"
        "- Tasks that would flood your context with intermediate data\n"
        "- Parallel independent workstreams (research A and B simultaneously)\n\n"
        "WHEN NOT TO USE (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> use execute_code\n"
        "- Single tool call -> just call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot use clarify\n\n"
        "IMPORTANT:\n"
        "- Subagents have NO memory of your conversation. Pass all relevant "
        "info (file paths, error messages, constraints) via the 'context' field.\n"
        "- Subagents CANNOT call: delegate_task, clarify, memory, send_message, "
        "execute_code.\n"
        "- Each subagent gets its own terminal session (separate working directory and state).\n"
        "- Results are always returned as an array, one entry per task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What the subagent should accomplish. Be specific and "
                    "self-contained -- the subagent knows nothing about your "
                    "conversation history."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Background information the subagent needs: file paths, "
                    "error messages, project structure, constraints. The more "
                    "specific you are, the better the subagent performs."
                ),
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Toolsets to enable for this subagent. "
                    "Default: inherits your enabled toolsets. "
                    f"Available toolsets: {_TOOLSET_LIST_STR}. "
                    "Common patterns: ['terminal', 'file'] for code work, "
                    "['web'] for research, ['browser'] for web interaction, "
                    "['terminal', 'file', 'web'] for full-stack tasks."
                ),
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Task goal"},
                        "context": {"type": "string", "description": "Task-specific context"},
                        "toolsets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Toolsets for this specific task. Available: {_TOOLSET_LIST_STR}. Use 'web' for network access, 'terminal' for shell, 'browser' for web interaction.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Legacy delegation-profile alias (e.g. research, implementation, verification). Category profiles can still constrain concurrency, tool access, and model/runtime defaults.",
                        },
                        "archetype": {
                            "type": "string",
                            "description": "Optional Wave 1 archetype override for this task.",
                        },
                        "route_category": {
                            "type": "string",
                            "description": "Optional Wave 1 route category override for this task. Distinct from delegation_profile/category.",
                        },
                        "delegation_profile": {
                            "type": "string",
                            "description": "Optional Wave 1 delegation profile override for this task. Distinct from route_category.",
                        },
                        "runtime_mode": {
                            "type": "string",
                            "description": "Optional Wave 1 runtime mode override for this task. Distinct from route_category and delegation_profile/category.",
                        },
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional additive Wave 1 skills overlay for this task.",
                        },
                        "task_contract": {
                            "type": "object",
                            "description": "Optional canonical structured Wave 1 task contract for this task.",
                        },
                        "acp_command": {
                            "type": "string",
                            "description": "Per-task ACP command override (e.g. 'claude'). Overrides the top-level acp_command for this task only.",
                        },
                        "acp_args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Per-task ACP args override.",
                        },
                    },
                    "required": ["goal"],
                },
                # No maxItems — the runtime limit is configurable via
                # delegation.max_concurrent_children (default 3) and
                # enforced with a clear error in delegate_task().
                "description": (
                    "Batch mode: tasks to run in parallel (limit configurable via delegation.max_concurrent_children, default 3). Each gets "
                    "its own subagent with isolated context and terminal session. "
                    "When provided, top-level goal/context/toolsets are ignored."
                ),
            },
            "category": {
                "type": "string",
                "description": "Legacy delegation-profile alias. Applies to all tasks unless a task overrides it. Preserves pre-Wave-1 delegation behavior when archetype/task_contract are absent.",
            },
            "archetype": {
                "type": "string",
                "description": "Optional Wave 1 archetype/task blueprint to seed defaults without replacing route category, delegation profile, skills, or task_contract.",
            },
            "route_category": {
                "type": "string",
                "description": "Optional Wave 1 route category (for example: ultrabrain, deep, quick, visual, writing, unspecified_low, unspecified_high). Distinct from delegation_profile/category.",
            },
            "delegation_profile": {
                "type": "string",
                "description": "Optional Wave 1 delegation profile. Distinct from route_category; falls back to the legacy category/default-mapping path when omitted.",
            },
            "runtime_mode": {
                "type": "string",
                "description": "Optional Wave 1 runtime mode. Resolved separately from route_category and delegation_profile/category, then rendered as its own prompt layer.",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional additive Wave 1 skills overlay. These remain separate from route category, delegation profile, and task_contract.",
            },
            "task_contract": {
                "type": "object",
                "description": "Optional canonical structured Wave 1 task contract. When omitted, legacy delegation behavior is preserved through default mapping.",
            },
            "max_iterations": {
                "type": "integer",
                "description": (
                    "Max tool-calling turns per subagent (default: 50). "
                    "Only set lower for simple tasks."
                ),
            },
            "persistent": {
                "type": "boolean",
                "description": "Persist the delegated task in the Wave 3 task store, even when run in the foreground.",
            },
            "background": {
                "type": "boolean",
                "description": "Launch the delegated task as a persistent background task using Hermes process_registry and return immediately with task/process IDs.",
            },
            "acp_command": {
                "type": "string",
                "description": (
                    "Override ACP command for child agents (e.g. 'claude', 'copilot'). "
                    "When set, children use ACP subprocess transport instead of inheriting "
                    "the parent's transport. Enables spawning Claude Code (claude --acp --stdio) "
                    "or other ACP-capable agents from any parent, including Discord/Telegram/CLI."
                ),
            },
            "acp_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Arguments for the ACP command (default: ['--acp', '--stdio']). "
                    "Only used when acp_command is set. Example: ['--acp', '--stdio', '--model', 'claude-opus-4-6']"
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        tasks=args.get("tasks"),
        category=args.get("category"),
        archetype=args.get("archetype"),
        route_category=args.get("route_category"),
        delegation_profile=args.get("delegation_profile"),
        runtime_mode=args.get("runtime_mode"),
        skills=args.get("skills"),
        task_contract=args.get("task_contract"),
        max_iterations=args.get("max_iterations"),
        persistent=bool(args.get("persistent", False)),
        background=bool(args.get("background", False)),
        acp_command=args.get("acp_command"),
        acp_args=args.get("acp_args"),
        parent_agent=kw.get("parent_agent")),
    check_fn=check_delegate_requirements,
    emoji="🔀",
)
