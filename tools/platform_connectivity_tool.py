"""Platform connectivity snapshot tool.

Surfaces lightweight reachability state from the local gateway/runtime state
files and built-in platform constants under ``gateway/``.  No outbound network
calls are made.

Success envelope contains:
* ``reachable`` mapping by platform name with bools
* ``issues`` list with short plain-language notes, capped small
* ``summary_counts`` with totals for included issues

Failure envelope:
* ``code: platform_connectivity_error`` when local runtime state is
  missing/unreadable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.config import get_hermes_home
from tools.validation_helper import relay_failure, relay_success

logger = logging.getLogger(__name__)

_ISSUE_CAP = 30
_HERMES_ERROR_CODE = "platform_connectivity_error"

_BUILTIN_PLATFORM_LABELS: Dict[str, str] = {
    "telegram": "Telegram",
    "photon": "Photon",
    "discord": "Discord",
    "whatsapp": "WhatsApp",
    "whatsapp_cloud": "WhatsApp Cloud",
    "slack": "Slack",
    "signal": "Signal",
    "mattermost": "Mattermost",
    "matrix": "Matrix",
    "homeassistant": "HomeAssistant",
    "email": "Email",
    "sms": "SMS",
    "dingtalk": "DingTalk",
    "api_server": "API Server",
    "webhook": "Webhook",
    "msgraph_webhook": "MSGraph Webhook",
    "feishu": "Feishu",
    "wecom": "WeCom",
    "wecom_callback": "WeCom Callback",
    "weixin": "Weixin",
    "bluebubbles": "BlueBubbles",
    "qqbot": "QQ Bot",
    "yuanbao": "Yuanbao",
    "relay": "Relay",
}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _coerce_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _phrase(text: Any) -> str:
    out = _safe_str(text)
    if not out:
        return "unknown error"
    out = out.replace("_", " ")
    return out.strip()


def _get_issue_cap() -> int:
    return _ISSUE_CAP


def _default_gateway_state_path() -> Optional[Path]:
    """Return the default gateway state JSON path, or ``None`` if unresolvable."""
    try:
        home = Path(get_hermes_home())
        return home.resolve() / "gateway_state.json"
    except (OSError, ImportError, AttributeError) as exc:
        logger.debug("Failed to resolve default gateway state file: %s", exc)
        return None


def _read_json_state(state_file: Optional[Path]) -> Any:
    if state_file is None or not state_file.exists():
        return None
    text = state_file.read_text(encoding="utf-8")
    if not text.strip():
        return None
    return json.loads(text)


def _collect_state_issues(
    raw_state: Dict[str, Any],
) -> Tuple[Dict[str, bool], List[str]]:
    """Return ``(reachable, issues)`` parsed from gateway_state JSON."""

    reachable: Dict[str, bool] = {}
    issues: List[str] = []
    aggregate = _safe_str(raw_state.get("gateway_state"))

    if aggregate:
        issues.append(f"gateway state: {aggregate}")

    state_platforms = raw_state.get("platforms")
    if not isinstance(state_platforms, dict):
        return reachable, issues

    top_error = _safe_str(raw_state.get("error"))
    if top_error:
        issues.append(f"gateway reported: {top_error}")

    for name, value in state_platforms.items():
        if not isinstance(name, str) or not name.strip():
            continue
        key = name.strip().lower()
        label = _BUILTIN_PLATFORM_LABELS.get(key, name.strip())
        if not isinstance(value, dict):
            continue

        state_value = _safe_str(value.get("state"))
        enabled_value = value.get("enabled")
        is_connected = state_value == "connected" or _coerce_boolish(enabled_value)
        reachable[key] = is_connected

        error_code = _safe_str(value.get("error_code"))
        error_message = _safe_str(value.get("error_message"))
        tools_present = _coerce_boolish(value.get("tools_present"))
        not_ready = _coerce_boolish(value.get("not_connected_reason"))

        if error_code:
            issues.append(f"{label} error code: {_phrase(error_code)}")
        if error_message:
            issues.append(f"{label}: {error_message}")
        if not tools_present and is_connected:
            issues.append(f"{label} adapter tools not detected")
        if not_ready:
            issues.append(f"{label} integration not ready")

    return reachable, issues


def _load_builtin_platforms() -> Dict[str, bool]:
    """Return a reachability mapping from local built-in platform constants.

    Malformed `config.yaml` entries are surfaced in ``issues`` instead of
    crashing the tool.
    """
    returning: Dict[str, bool] = {}
    try:
        from gateway.config import _BUILTIN_PLATFORM_VALUES  # type: ignore[attr-defined]

        for value in _BUILTIN_PLATFORM_VALUES:
            returning[str(value).strip().lower()] = False
    except (OSError, ImportError, AttributeError):
        for name in _BUILTIN_PLATFORM_LABELS:
            returning[name] = False
    return returning


def _validate_platform_config_note(name: str, path: Path) -> List[str]:
    if not path.exists() or not path.is_file():
        return []
    returning: List[str] = []
    try:
        text = path.read_text(encoding="utf-8")
        if text.strip():
            import yaml

            yaml.safe_load(text)
    except Exception as exc:
        returning.append(f"{name} config malformed: {exc}")
    return returning


def _merge_issue_sets(prior: List[str], incoming: List[str]) -> List[str]:
    returning = list(prior)
    seen = set(returning)
    returning.extend(item for item in incoming if item not in seen and not seen.add(item))  # type: ignore[func-returns-value]
    return returning


def _cap_issues(issues: List[str], cap: int) -> List[str]:
    unique = sorted(set(issues))
    return unique[:cap]


def _summarize_issues(
    issues: List[str],
) -> Tuple[List[str], Dict[str, int]]:
    unique = sorted(set(issues))
    capped = unique[:_get_issue_cap()]
    count_malformed = sum(1 for note in unique if "malformed" in note.lower())
    count_tools_missing = sum(
        1 for note in unique if "tools not detected" in note.lower() or "tools missing" in note.lower()
    )
    summary = {
        "missing_tools": count_tools_missing,
        "malformed_configs": count_malformed,
        "total_issues": len(unique),
    }
    return capped, summary


def summarize_platform_connectivity() -> Dict[str, Any]:
    """Return a structured connectivity summary for local runtime state.

    Uses `gateway_state.json` plus local built-in constants/config under
    `gateway/` and `cron/`. Never makes outbound network calls.
    """
    try:
        gateway_state_path = _default_gateway_state_path()
        state = _read_json_state(gateway_state_path)
        if state is None:
            return relay_failure(
                "Gateway state is not configured or available",
                code=_HERMES_ERROR_CODE,
            )
    except OSError as exc:
        logger.debug("Local gateway state is unreadable: %s", exc)
        return relay_failure(
            f"Gateway state is inaccessible or unreadable ({exc})",
            code=_HERMES_ERROR_CODE,
        )
    except ValueError as exc:
        logger.debug("Local gateway state is malformed: %s", exc)
        return relay_failure(
            f"Gateway state is malformed ({exc})",
            code=_HERMES_ERROR_CODE,
        )

    if not isinstance(state, dict):
        return relay_failure(
            "Local runtime state is missing or unreadable",
            code=_HERMES_ERROR_CODE,
        )

    try:
        state_reachable, state_issues = _collect_state_issues(state)
    except Exception as exc:
        logger.debug("Failed to parse runtime state: %s", exc)
        return relay_failure(
            f"Connectivity payload could not be built ({exc})",
            code=_HERMES_ERROR_CODE,
        )

    platforms = dict(_load_builtin_platforms())
    platforms.update(state_reachable)

    issue_pool = list(state_issues)
    count_malformed_from_state = 0
    for note in state_issues:
        if "malformed" in note.lower():
            count_malformed_from_state += 1

    counts_temp = {
        "missing_tools": sum(1 for n in state_issues if "tools not detected" in n.lower()),
        "malformed_configs": count_malformed_from_state,
        "total_issues": len(set(state_issues)),
    }

    if not platforms and not state_reachable and not issue_pool:
        return relay_failure(
            "Local runtime state presents no platform config or state, connectivity is unknown.",
            code=_HERMES_ERROR_CODE,
        )

    capped_issues, summary_counts = _summarize_issues(issue_pool)

    data = {
        "reachable": platforms,
        "issues": capped_issues,
        "summary_counts": summary_counts,
    }

    return relay_success(data)


def check_platform_connectivity_requirements() -> bool:
    """Return ``False`` when no platform config/state is present."""
    try:
        gateway_state_path = _default_gateway_state_path()
        if gateway_state_path is None:
            return False
        state = _read_json_state(gateway_state_path)
        if not isinstance(state, dict):
            return False
        platforms = state.get("platforms")
        return isinstance(platforms, dict) and bool(platforms)
    except Exception:
        return False


def platform_connectivity_tool(_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Tool handler wrapping :func:`summarize_platform_connectivity`."""
    return summarize_platform_connectivity()


_PLATFORM_CONNECTIVITY_SCHEMA: Dict[str, Any] = {
    "name": "platform_connectivity",
    "description": (
        "Return structured local platform connectivity: reachability by platform name, "
        "plain-language issues, and summary counts. No outbound network calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def register(registry):  # pragma: no cover - registration side-effect
    registry.register(
        name="platform_connectivity",
        toolset="ops",
        schema=_PLATFORM_CONNECTIVITY_SCHEMA,
        handler=platform_connectivity_tool,
        check_fn=check_platform_connectivity_requirements,
        emoji="🔌",
    )
