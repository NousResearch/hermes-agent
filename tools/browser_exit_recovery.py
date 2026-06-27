"""Safe browser network-exit recovery.

This module helps Hermes recover from genuine network, proxy, tunnel, DNS, or
TLS failures by switching to a configured backup exit. It deliberately refuses
to change exits for CAPTCHA, human verification, bot walls, bans, rate limits,
or other access-control blocks.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.browser_autonomy import assess_browser_state
from tools.registry import registry
from utils import is_truthy_value


_KNOWN_EXIT_PROFILES: Tuple[str, ...] = ("residential", "vpn", "direct")
_DEFAULT_EXIT_PROFILES: Tuple[str, ...] = ("residential", "vpn")
_DEFAULT_TIMEOUT_S = 45
_MAX_OUTPUT = 1200


_RECOVERABLE_NETWORK_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "proxy_connection_failed",
        re.compile(
            r"ERR_PROXY_CONNECTION_FAILED|ERR_TUNNEL_CONNECTION_FAILED|"
            r"proxy\s+(?:server\s+)?(?:connection\s+)?(?:failed|refused|unreachable)|"
            r"tunnel\s+connection\s+failed",
            re.I,
        ),
    ),
    (
        "socks_or_exit_unreachable",
        re.compile(
            r"socks(?:5|5h)?\s+(?:connection\s+)?(?:failed|refused|unreachable)|"
            r"exit\s+(?:proxy|tunnel|node)\s+(?:failed|down|unreachable)",
            re.I,
        ),
    ),
    (
        "connection_timeout_or_reset",
        re.compile(
            r"ERR_CONNECTION_(?:RESET|TIMED_OUT|CLOSED|ABORTED)|"
            r"connection\s+(?:reset|timed\s*out|timeout|closed|aborted)",
            re.I,
        ),
    ),
    (
        "dns_failure",
        re.compile(
            r"ERR_NAME_NOT_RESOLVED|DNS_PROBE_FINISHED|"
            r"temporary\s+failure\s+in\s+name\s+resolution|"
            r"could\s+not\s+resolve\s+host|name\s+or\s+service\s+not\s+known",
            re.I,
        ),
    ),
    (
        "network_unreachable",
        re.compile(
            r"ERR_INTERNET_DISCONNECTED|ERR_NETWORK_CHANGED|"
            r"network\s+(?:is\s+)?(?:unreachable|changed)|"
            r"site\s+can(?:not|'t)\s+be\s+reached",
            re.I,
        ),
    ),
    (
        "tls_or_gateway_failure",
        re.compile(
            r"ERR_SSL_PROTOCOL_ERROR|ERR_CERT_(?:AUTHORITY_INVALID|DATE_INVALID)|"
            r"TLS\s+handshake\s+(?:failed|timeout)|"
            r"\b(?:502|503|504)\s+(?:bad\s+gateway|service\s+unavailable|gateway\s+timeout)",
            re.I,
        ),
    ),
)


_ACCESS_CONTROL_ERROR_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("captcha", re.compile(r"\b(?:captcha|hcaptcha|recaptcha|turnstile)\b", re.I)),
    (
        "human_verification",
        re.compile(
            r"verify\s+(?:you\s+are|that\s+you\s+are)\s+(?:a\s+)?human|"
            r"i\s+am\s+not\s+a\s+robot|security\s+challenge",
            re.I,
        ),
    ),
    (
        "access_denied",
        re.compile(r"\b(?:403\s+forbidden|access\s+denied|request\s+blocked)\b", re.I),
    ),
    (
        "rate_limited",
        re.compile(
            r"\b(?:429\s+too\s+many\s+requests|rate\s+limit(?:ed)?|too\s+many\s+requests)\b",
            re.I,
        ),
    ),
    (
        "bot_wall",
        re.compile(
            r"bot\s+detected|automated\s+(?:traffic|queries)|unusual\s+traffic|temporarily\s+blocked",
            re.I,
        ),
    ),
    (
        "challenge_wall",
        re.compile(
            r"checking\s+your\s+browser|just\s+a\s+moment|attention\s+required|"
            r"\bcf-chl\b|challenge-platform|ddos\s+protection|\bcloudflare\b",
            re.I,
        ),
    ),
)


_SECRETISH_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+|"
    r"(xox[baprs]-)[A-Za-z0-9-]+|"
    r"(Authorization:\s*Bearer\s+)[A-Za-z0-9._-]+|"
    r"([A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASS|KEY)[A-Za-z0-9_]*=)[^\s]+",
    re.I,
)


def _clip(text: str, max_chars: int = _MAX_OUTPUT) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _redact(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        for idx in (1, 2, 3, 4):
            prefix = match.group(idx)
            if prefix:
                return f"{prefix}<redacted>"
        return "<redacted>"

    return _SECRETISH_RE.sub(repl, text or "")


def _fields(
    *,
    error: str = "",
    url: str = "",
    title: str = "",
    snapshot: str = "",
    console_text: str = "",
) -> Dict[str, str]:
    return {
        "error": error or "",
        "url": url or "",
        "title": title or "",
        "snapshot": snapshot or "",
        "console": console_text or "",
    }


def _first_match(
    fields: Dict[str, str],
    rules: Iterable[Tuple[str, re.Pattern[str]]],
) -> Optional[Dict[str, str]]:
    for field, text in fields.items():
        if not text:
            continue
        for rule_name, pattern in rules:
            match = pattern.search(text)
            if match is None:
                continue
            start = max(match.start() - 60, 0)
            end = min(match.end() + 60, len(text))
            return {
                "field": field,
                "rule": rule_name,
                "excerpt": _clip(text[start:end], 180),
            }
    return None


def classify_exit_issue(
    *,
    error: str = "",
    url: str = "",
    title: str = "",
    snapshot: str = "",
    console_text: str = "",
) -> Dict[str, Any]:
    """Classify whether changing the browser exit is safe and useful."""

    autonomy_console = "\n".join(part for part in (console_text, error) if part)
    state = assess_browser_state(
        url=url,
        title=title,
        snapshot=snapshot,
        console_text=autonomy_console,
    )
    if state.get("status") in {"needs_human", "blocked"}:
        return {
            "status": "not_allowed",
            "kind": state.get("kind", "access_control"),
            "safe_to_change_exit": False,
            "policy": state.get("policy", "do_not_evade_access_controls"),
            "message": (
                "The page appears to require human verification or has blocked "
                "automated access. Do not change IPs or exits to evade it."
            ),
            "matches": state.get("matches", []),
        }

    field_map = _fields(
        error=error,
        url=url,
        title=title,
        snapshot=snapshot,
        console_text=console_text,
    )

    access_match = _first_match(field_map, _ACCESS_CONTROL_ERROR_RULES)
    if access_match:
        return {
            "status": "not_allowed",
            "kind": "access_control_or_rate_limit",
            "safe_to_change_exit": False,
            "policy": "do_not_evade_access_controls",
            "message": (
                "The error looks like an access-control, rate-limit, CAPTCHA, "
                "or bot-wall response. Do not change IPs or exits to bypass it."
            ),
            "matches": [access_match],
        }

    network_match = _first_match(field_map, _RECOVERABLE_NETWORK_RULES)
    if network_match:
        return {
            "status": "recoverable",
            "kind": "network_or_exit_health",
            "safe_to_change_exit": True,
            "policy": "network_health_failover_only",
            "message": (
                "The error looks like a network, proxy, tunnel, DNS, TLS, or "
                "gateway health issue. A configured backup exit can be tried."
            ),
            "matches": [network_match],
        }

    return {
        "status": "none",
        "kind": "unclassified",
        "safe_to_change_exit": False,
        "policy": "no_exit_change_without_network_health_signal",
        "message": "No clear network or exit-health failure was detected.",
        "matches": [],
    }


def build_exit_recovery_hint(
    *,
    error: str = "",
    url: str = "",
    title: str = "",
    snapshot: str = "",
    console_text: str = "",
    auto_recovery_enabled: bool = False,
) -> Dict[str, Any]:
    classification = classify_exit_issue(
        error=error,
        url=url,
        title=title,
        snapshot=snapshot,
        console_text=console_text,
    )
    hint: Dict[str, Any] = {
        "status": classification["status"],
        "kind": classification["kind"],
        "eligible": classification["status"] == "recoverable",
        "auto_recovery_enabled": bool(auto_recovery_enabled),
        "safe_to_change_exit": bool(classification.get("safe_to_change_exit")),
        "policy": classification["policy"],
        "message": classification["message"],
        "matches": classification.get("matches", []),
    }
    if hint["eligible"]:
        hint["tool"] = "browser_exit_recover"
        hint["recommended_args"] = {
            "reason": error or classification["message"],
            "url": url,
            "preferred_exit": "residential",
            "allow_direct": False,
        }
    return hint


def _read_exit_recovery_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if not isinstance(browser_cfg, dict):
            return {}
        exit_cfg = browser_cfg.get("exit_recovery", {})
        return exit_cfg if isinstance(exit_cfg, dict) else {}
    except Exception:
        return {}


def _configured_controller() -> str:
    env_path = os.getenv("HERMES_BROWSER_EXIT_CONTROLLER", "").strip()
    if env_path:
        return env_path
    cfg_path = _read_exit_recovery_config().get("controller", "")
    return str(cfg_path or "").strip()


def resolve_exit_controller() -> Optional[str]:
    """Return an executable local exit-controller path, if one is available."""

    configured = _configured_controller()
    candidates: list[Path] = []
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            return None
        candidates.append(path)
    else:
        home = Path.home()
        candidates.append(home / ".hermes" / "bin" / "hermes-control.sh")

    for candidate in candidates:
        try:
            if (
                candidate.name == "hermes-control.sh"
                and candidate.is_file()
                and os.access(candidate, os.X_OK)
            ):
                return str(candidate)
        except OSError:
            continue
    return None


def _timeout_s() -> int:
    raw = _read_exit_recovery_config().get("timeout_s", _DEFAULT_TIMEOUT_S)
    try:
        return max(5, min(int(raw), 180))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


def _configured_profiles(allow_direct: bool) -> Tuple[str, ...]:
    cfg = _read_exit_recovery_config()
    raw_profiles = cfg.get("fallback_exits", _DEFAULT_EXIT_PROFILES)
    profiles: Sequence[Any]
    if isinstance(raw_profiles, (list, tuple)):
        profiles = raw_profiles
    elif isinstance(raw_profiles, str):
        profiles = [p.strip() for p in raw_profiles.split(",")]
    else:
        profiles = _DEFAULT_EXIT_PROFILES

    allowed: list[str] = []
    for item in profiles:
        profile = str(item or "").strip().lower()
        if profile not in _KNOWN_EXIT_PROFILES:
            continue
        if profile == "direct" and not allow_direct:
            continue
        if profile not in allowed:
            allowed.append(profile)

    if allow_direct and "direct" not in allowed:
        allowed.append("direct")
    return tuple(allowed)


def _allow_direct_fallback(explicit_allow: bool) -> bool:
    cfg = _read_exit_recovery_config()
    config_allows_direct = is_truthy_value(cfg.get("allow_direct_fallback"), default=False)
    return bool(explicit_allow and config_allows_direct)


def _select_exit(preferred_exit: str, allow_direct: bool) -> Tuple[Optional[str], Optional[str]]:
    allowed = _configured_profiles(allow_direct)
    preferred = (preferred_exit or "").strip().lower()
    if preferred:
        if preferred not in _KNOWN_EXIT_PROFILES:
            return None, f"Unknown exit profile {preferred_exit!r}."
        if preferred == "direct" and not allow_direct:
            return None, (
                "Direct exit fallback is disabled unless "
                "browser.exit_recovery.allow_direct_fallback and allow_direct are both true."
            )
        if preferred not in allowed:
            return None, f"Exit profile {preferred!r} is not allowed by browser.exit_recovery.fallback_exits."
        return preferred, None
    return (allowed[0] if allowed else None), None


def _run_controller(controller: str, args: Sequence[str], timeout_s: int) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            [controller, *args],
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
            creationflags=windows_hide_flags(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "ok": False,
            "stdout": _clip(_redact(exc.stdout or ""), _MAX_OUTPUT),
            "stderr": f"exit controller timed out after {timeout_s} seconds",
        }
    except OSError as exc:
        return {
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": _clip(_redact(str(exc)), _MAX_OUTPUT),
        }

    stdout = _clip(_redact(completed.stdout), _MAX_OUTPUT)
    stderr = _clip(_redact(completed.stderr), _MAX_OUTPUT)
    return {
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
    }


def recover_browser_exit(
    *,
    reason: str = "",
    url: str = "",
    preferred_exit: str = "",
    allow_direct: bool = False,
    verify: bool = True,
) -> Dict[str, Any]:
    """Attempt a safe backup-exit switch for a classified network failure."""

    classification = classify_exit_issue(error=reason, url=url)
    base: Dict[str, Any] = {
        "classification": classification,
        "policy": classification["policy"],
    }
    if classification["status"] == "not_allowed":
        return {
            **base,
            "status": "not_allowed",
            "success": False,
            "message": classification["message"],
        }
    if classification["status"] != "recoverable":
        return {
            **base,
            "status": "not_eligible",
            "success": False,
            "message": (
                "Exit recovery requires a concrete network, proxy, tunnel, DNS, "
                "TLS, or gateway-health failure signal."
            ),
        }

    direct_allowed = _allow_direct_fallback(allow_direct)
    selected_exit, selection_error = _select_exit(preferred_exit, direct_allowed)
    if selection_error or not selected_exit:
        return {
            **base,
            "status": "invalid_exit",
            "success": False,
            "message": selection_error or "No safe exit profile is configured.",
        }

    controller = resolve_exit_controller()
    if controller is None:
        return {
            **base,
            "status": "no_controller",
            "success": False,
            "selected_exit": selected_exit,
            "message": (
                "No executable hermes-control.sh exit controller was found. "
                "Configure browser.exit_recovery.controller or "
                "HERMES_BROWSER_EXIT_CONTROLLER."
            ),
        }

    timeout_s = _timeout_s()
    set_result = _run_controller(controller, ["exit", "set", selected_exit], timeout_s)
    result: Dict[str, Any] = {
        **base,
        "status": "changed" if set_result["ok"] else "set_failed",
        "success": bool(set_result["ok"]),
        "selected_exit": selected_exit,
        "controller": controller,
        "set_result": set_result,
        "message": (
            f"Changed browser network exit to {selected_exit}."
            if set_result["ok"]
            else f"Failed to change browser network exit to {selected_exit}."
        ),
    }
    if not set_result["ok"] or not verify:
        return result

    verify_result = _run_controller(controller, ["exit", "verify", selected_exit], timeout_s)
    result["verify_result"] = verify_result
    if not verify_result["ok"]:
        result["status"] = "verify_failed"
        result["success"] = False
        result["message"] = f"Changed exit to {selected_exit}, but verification failed."
    return result


BROWSER_EXIT_RECOVERY_SCHEMA: Dict[str, Any] = {
    "name": "browser_exit_recover",
    "description": (
        "Safely recover browser navigation from genuine network/proxy/tunnel/"
        "DNS/TLS exit-health failures by switching to a configured backup exit. "
        "This must not be used for CAPTCHA, human verification, bot walls, bans, "
        "rate limits, or access-control bypass."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "The browser error or diagnostic text that indicates a network/exit-health failure.",
            },
            "url": {
                "type": "string",
                "description": "Optional URL being navigated when the failure occurred.",
            },
            "preferred_exit": {
                "type": "string",
                "enum": list(_KNOWN_EXIT_PROFILES),
                "description": "Optional configured exit profile to try first.",
            },
            "allow_direct": {
                "type": "boolean",
                "default": False,
                "description": "Allow falling back to the direct/raw exit. Disabled by default.",
            },
            "verify": {
                "type": "boolean",
                "default": True,
                "description": "Verify the selected exit after switching.",
            },
        },
        "required": ["reason"],
    },
}


def browser_exit_recover(
    reason: Optional[str] = None,
    url: Optional[str] = None,
    preferred_exit: Optional[str] = None,
    allow_direct: bool = False,
    verify: bool = True,
) -> str:
    recovery = recover_browser_exit(
        reason=reason or "",
        url=url or "",
        preferred_exit=preferred_exit or "",
        allow_direct=allow_direct,
        verify=verify,
    )
    return json.dumps(
        {
            "success": bool(recovery.get("success")),
            "ip_recovery": recovery,
        },
        ensure_ascii=False,
    )


registry.register(
    name="browser_exit_recover",
    toolset="browser",
    schema=BROWSER_EXIT_RECOVERY_SCHEMA,
    handler=lambda args, **_kw: browser_exit_recover(
        reason=args.get("reason"),
        url=args.get("url"),
        preferred_exit=args.get("preferred_exit"),
        allow_direct=bool(args.get("allow_direct", False)),
        verify=bool(args.get("verify", True)),
    ),
    emoji="🌐",
)
