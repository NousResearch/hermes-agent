"""Computer-use/browser capability status and UI action risk policy.

This module is deliberately side-effect-light: it does not click, type, drag,
open native GUI windows, or launch browsers. It only classifies actions and
reports already-configured browser/computer-use availability so CLI/status
surfaces can explain the safe route before any mutation-capable tool is used.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Any

from hermes_cli.browser_connect import DEFAULT_BROWSER_CDP_URL, is_browser_debug_ready

from .routing import route_desktop_request

LOW_RISK_BROWSER_ACTIONS = frozenset(
    {
        "browser_snapshot",
        "browser_get_images",
        "browser_vision",
    }
)
MEDIUM_RISK_BROWSER_ACTIONS = frozenset(
    {
        "browser_navigate",
        "browser_back",
        "browser_scroll",
    }
)
HIGH_RISK_BROWSER_ACTIONS = frozenset(
    {
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_console",
    }
)
READ_ONLY_NATIVE_ACTIONS = frozenset({"capture", "wait", "list_apps"})
NATIVE_GUI_MUTATION_ACTIONS = frozenset(
    {
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "drag",
        "scroll",
        "type",
        "key",
        "set_value",
        "focus_app",
    }
)
WINDOWS_NATIVE_GUI_FORBIDDEN_ACTIONS = tuple(sorted(NATIVE_GUI_MUTATION_ACTIONS))


@dataclass(frozen=True)
class ActionRiskDecision:
    surface: str
    action: str
    risk: str
    allowed: bool
    requires_approval: bool
    reason: str
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrowserAvailability:
    available: bool
    mode: str
    reason: str
    next_step: str
    cdp_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_action_risk(
    *,
    surface: str,
    action: str,
    platform: str | None = None,
    task: str = "",
    read_only: bool | None = None,
) -> ActionRiskDecision:
    """Classify the execution risk for a GUI/browser action.

    The classifier is policy-only. It never performs the requested action.
    """

    normalized_surface = (surface or "").strip().lower().replace("-", "_")
    normalized_action = (action or "").strip().lower()
    platform_id = platform or sys.platform
    task_text = task or ""

    if normalized_surface in {"native", "native_gui", "computer_use", "desktop"}:
        return _classify_native_gui_action(normalized_action, platform_id, read_only)
    if normalized_surface in {"browser", "web", "web_ui"}:
        return _classify_browser_action(normalized_action, task_text)
    return ActionRiskDecision(
        surface=normalized_surface or "unknown",
        action=normalized_action,
        risk="blocked",
        allowed=False,
        requires_approval=False,
        reason=f"Unknown action surface {surface!r}; refusing to route by default.",
        next_step="Choose an explicit surface such as browser or native_gui before routing.",
    )


def diagnose_browser_availability(*, probe_cdp: bool = True) -> BrowserAvailability:
    """Return a structured diagnosis of browser tool availability.

    The probe is bounded and only checks an already-configured CDP endpoint. It
    does not launch Chrome, start Gateway, or call paid/cloud APIs.
    """

    cdp_url = _configured_cdp_url()
    if cdp_url:
        if not probe_cdp:
            return BrowserAvailability(
                available=True,
                mode="cdp",
                reason="CDP endpoint is configured; reachability probe skipped.",
                next_step="Run with probing enabled or use `hermes /browser status` to verify the endpoint.",
                cdp_url=cdp_url,
            )
        if is_browser_debug_ready(cdp_url, timeout=1.0):
            return BrowserAvailability(
                available=True,
                mode="cdp",
                reason="Configured CDP endpoint is reachable.",
                next_step="Use the browser toolset for cross-platform web UI automation.",
                cdp_url=cdp_url,
            )
        return BrowserAvailability(
            available=False,
            mode="cdp",
            reason="Configured CDP endpoint is not reachable.",
            next_step="Run `hermes /browser connect` or start Chrome with remote debugging, then retry.",
            cdp_url=cdp_url,
        )

    try:
        from tools import browser_tool
    except Exception as exc:
        return BrowserAvailability(
            available=False,
            mode="unknown",
            reason=f"browser tool module could not be imported: {exc}",
            next_step="Verify the Hermes browser tool installation.",
        )

    try:
        if browser_tool.check_browser_requirements():
            return BrowserAvailability(
                available=True,
                mode=_browser_mode(browser_tool),
                reason="Browser tool requirements are satisfied.",
                next_step="Use the browser toolset for cross-platform web UI automation.",
            )
    except Exception as exc:
        return BrowserAvailability(
            available=False,
            mode="unknown",
            reason=f"Browser requirement check failed: {exc}",
            next_step="Run `hermes doctor` or `hermes tools` to repair browser tooling.",
        )

    return BrowserAvailability(
        available=False,
        mode=_browser_mode(browser_tool),
        reason=_browser_missing_reason(browser_tool),
        next_step=_browser_next_step(browser_tool),
    )


def computer_use_capability_status(
    *,
    platform: str | None = None,
    probe_browser: bool = True,
) -> dict[str, Any]:
    """Return structured computer-use + browser status for CLI/readback."""

    platform_id = platform or sys.platform
    computer_use_available = _computer_use_available(platform_id)
    browser = diagnose_browser_availability(probe_cdp=probe_browser)
    routes = [
        route_desktop_request(
            target="https://example.com",
            platform=platform_id,
            browser_available=browser.available,
        ),
        route_desktop_request(
            target="C:/Users/example/AppData/Local/hermes/logs/agent.log",
            platform=platform_id,
        ),
        route_desktop_request(target="screenshot", platform=platform_id),
        route_desktop_request(
            intent="operate",
            target="app:Settings",
            read_only=False,
            platform=platform_id,
            windows_uia_readonly_available=True,
        ),
    ]
    return {
        "platform": platform_id,
        "computer_use": {
            "available": computer_use_available,
            "backend": "cua-driver",
            "reason": _computer_use_reason(platform_id, computer_use_available),
            "native_gui_mutation_allowed": bool(
                computer_use_available and not _is_windows(platform_id)
            ),
        },
        "browser": browser.to_dict(),
        "input_proposals": _input_proposal_capability_status(),
        "browser_input_execution": _browser_input_execution_capability_status(),
        "windows_uia_readonly": _windows_uia_readonly_capability_status(platform_id),
        "routes": [asdict(route) for route in routes],
        "risk_policy": {
            "windows_native_gui_mutation_allowed": False,
            "windows_native_gui_forbidden_actions": list(WINDOWS_NATIVE_GUI_FORBIDDEN_ACTIONS),
            "browser_high_risk_actions": sorted(HIGH_RISK_BROWSER_ACTIONS),
            "browser_low_risk_actions": sorted(LOW_RISK_BROWSER_ACTIONS),
        },
    }


def _classify_native_gui_action(
    action: str,
    platform_id: str,
    read_only: bool | None,
) -> ActionRiskDecision:
    if _is_windows(platform_id) and action in NATIVE_GUI_MUTATION_ACTIONS:
        return ActionRiskDecision(
            surface="native_gui",
            action=action,
            risk="blocked",
            allowed=False,
            requires_approval=False,
            reason=(
                "Windows native GUI click/type/drag and other mutation actions "
                "are forbidden in Wave 1."
            ),
            next_step="Use browser, terminal/file, or vision routes instead; do not click/type/drag native Windows UI.",
        )
    if action in READ_ONLY_NATIVE_ACTIONS or read_only is True:
        return ActionRiskDecision(
            surface="native_gui",
            action=action,
            risk="low",
            allowed=True,
            requires_approval=False,
            reason="Native GUI read-only inspection action.",
            next_step="Proceed only through an available read-only backend/capture path.",
        )
    if action in NATIVE_GUI_MUTATION_ACTIONS:
        return ActionRiskDecision(
            surface="native_gui",
            action=action,
            risk="high",
            allowed=True,
            requires_approval=True,
            reason="Native GUI action mutates user-visible state and requires approval.",
            next_step="Use the platform-supported backend with existing approval gates.",
        )
    return ActionRiskDecision(
        surface="native_gui",
        action=action,
        risk="blocked",
        allowed=False,
        requires_approval=False,
        reason=f"Unknown native GUI action {action!r}.",
        next_step="Choose an explicit supported action before routing.",
    )


def _classify_browser_action(action: str, task: str) -> ActionRiskDecision:
    if action in HIGH_RISK_BROWSER_ACTIONS and _is_sensitive_task(task):
        return ActionRiskDecision(
            surface="browser",
            action=action,
            risk="blocked",
            allowed=False,
            requires_approval=False,
            reason="Sensitive browser GUI task detected before execution.",
            next_step="Ask the user for explicit guidance or use an audited non-GUI workflow.",
        )
    if action in LOW_RISK_BROWSER_ACTIONS:
        return ActionRiskDecision(
            surface="browser",
            action=action,
            risk="low",
            allowed=True,
            requires_approval=False,
            reason="Browser read-only inspection action.",
            next_step="Proceed with browser readback/snapshot tooling.",
        )
    if action in MEDIUM_RISK_BROWSER_ACTIONS:
        return ActionRiskDecision(
            surface="browser",
            action=action,
            risk="medium",
            allowed=True,
            requires_approval=False,
            reason="Browser navigation/scroll action changes page context but not form values by itself.",
            next_step="Proceed when the target URL/task is in scope.",
        )
    if action in HIGH_RISK_BROWSER_ACTIONS:
        return ActionRiskDecision(
            surface="browser",
            action=action,
            risk="high",
            allowed=True,
            requires_approval=True,
            reason="Browser action mutates browser page state or executes page-context input.",
            next_step="Use browser tooling only within the approved task scope and verify after execution.",
        )
    return ActionRiskDecision(
        surface="browser",
        action=action,
        risk="blocked",
        allowed=False,
        requires_approval=False,
        reason=f"Unknown browser action {action!r}.",
        next_step="Choose an explicit supported browser action before routing.",
    )


def _configured_cdp_url() -> str:
    env_url = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_url:
        return env_url
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url") or "").strip()
    except Exception:
        return ""
    return ""


def _browser_mode(browser_tool: Any) -> str:
    try:
        if browser_tool._is_camofox_mode():
            return "camofox"
    except Exception:
        pass
    try:
        provider = browser_tool._get_cloud_provider()
        if provider is not None:
            return f"cloud:{provider.provider_name()}"
    except Exception:
        pass
    try:
        if browser_tool._using_lightpanda_engine():
            return "local:lightpanda"
    except Exception:
        pass
    return "local"


def _browser_missing_reason(browser_tool: Any) -> str:
    try:
        browser_tool._find_agent_browser()
    except FileNotFoundError:
        return "agent-browser CLI is not installed or not on PATH."
    except Exception as exc:
        return f"agent-browser lookup failed: {exc}"
    try:
        provider = browser_tool._get_cloud_provider()
        if provider is not None and not provider.is_configured():
            return f"{provider.provider_name()} browser provider credentials are not configured."
    except Exception:
        pass
    try:
        if not browser_tool._using_lightpanda_engine() and not browser_tool._chromium_installed():
            return "Chromium browser binary is not installed for local browser mode."
    except Exception:
        pass
    return "Browser requirements are not satisfied."


def _browser_next_step(browser_tool: Any) -> str:
    try:
        hint = browser_tool._browser_install_hint()
    except Exception:
        hint = "npx agent-browser install --with-deps"
    return f"Install/repair browser tooling: {hint}; or connect an existing browser with `hermes /browser connect`."


def _computer_use_available(platform_id: str) -> bool:
    if platform_id != "darwin":
        return False
    return bool(shutil.which("cua-driver"))


def _computer_use_reason(platform_id: str, available: bool) -> str:
    if available:
        return "cua-driver is installed and computer_use can use the macOS backend."
    if platform_id != "darwin":
        return "computer_use/cua-driver is macOS-only; use browser/terminal/file/vision alternatives on this platform."
    return "cua-driver is not installed."


def _input_proposal_capability_status() -> dict[str, Any]:
    """Return dry-run input proposal capability metadata.

    Imported lazily because proposals.py itself uses classify_action_risk from
    this module. The fallback keeps `computer-use status` robust if a partial
    install is missing the proposal module.
    """

    try:
        from .proposals import input_proposal_capability_status

        return input_proposal_capability_status()
    except Exception as exc:
        return {
            "available": False,
            "dry_run_only": True,
            "will_execute": False,
            "schemas": [],
            "reason": f"Input proposal capability module unavailable: {exc}",
        }


def _browser_input_execution_capability_status() -> dict[str, Any]:
    try:
        from .browser_input import browser_input_execution_capability_status

        return browser_input_execution_capability_status()
    except Exception as exc:
        return {
            "available": False,
            "requires_injected_backend": True,
            "native_gui_mutation_allowed": False,
            "reason": f"Browser input execution capability module unavailable: {exc}",
        }


def _windows_uia_readonly_capability_status(platform_id: str) -> dict[str, Any]:
    try:
        from .windows_uia_readonly import windows_uia_readonly_capability_status

        return windows_uia_readonly_capability_status(platform=platform_id)
    except Exception as exc:
        return {
            "available": False,
            "backend": "pywinauto-uia-readonly",
            "platform": platform_id,
            "read_only": True,
            "mutation_allowed": False,
            "input_fallback_enabled": False,
            "reason": f"Windows UIA read-only capability module unavailable: {exc}",
        }


def _is_windows(platform_id: str) -> bool:
    return platform_id.lower().startswith("win")


def _is_sensitive_task(task: str) -> bool:
    lowered = task.lower()
    sensitive_terms = (
        "password",
        "passcode",
        "2fa",
        "mfa",
        "totp",
        "otp",
        "payment",
        "checkout",
        "bank",
        "wire transfer",
        "permission dialog",
        "approve login",
        "login prompt",
    )
    return any(term in lowered for term in sensitive_terms)


__all__ = [
    "ActionRiskDecision",
    "BrowserAvailability",
    "HIGH_RISK_BROWSER_ACTIONS",
    "LOW_RISK_BROWSER_ACTIONS",
    "NATIVE_GUI_MUTATION_ACTIONS",
    "WINDOWS_NATIVE_GUI_FORBIDDEN_ACTIONS",
    "classify_action_risk",
    "computer_use_capability_status",
    "diagnose_browser_availability",
]
