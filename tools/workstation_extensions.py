"""Desktop-session Chrome extension capability tools for Hermes Workstation.

The manager owns durable CRX files, Electron owns the loaded Chromium
extension, the scoped policy engine owns authorization, and ExecutionJournal
owns evidence.  This module only composes those existing owners; it is never a
second browser or extension store.
"""

from __future__ import annotations

import json
from typing import Any

from tools.registry import registry
from workstation.contracts import ExecutionEventKind, RiskLevel
from workstation.extensions import ChromeExtensionManager
from workstation.journal import ExecutionJournal
from workstation.policy import ActionScope, PolicyDecision, ScopedPolicyEngine


_TOOLSET = "desktop_ui"


def _desktop_session() -> bool:
    try:
        from gateway.session_context import get_session_env
        source = str(get_session_env("HERMES_SESSION_SOURCE", "") or "").strip().lower()
        platform = str(get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
        return (source or platform) == "desktop"
    except Exception:
        return False


def _journal(task_id: str | None, session_id: str | None) -> ExecutionJournal:
    return ExecutionJournal(str(task_id or session_id or "workstation-extension"), str(session_id or ""))


def _approval(reason: str, rule_key: str) -> tuple[bool, str]:
    from tools.approval import request_tool_approval

    result = request_tool_approval("browser_extension_install", reason, rule_key=rule_key)
    return bool(result.get("approved")), str(result.get("message") or reason)


def _controller(action: str, args: dict[str, Any], *, task_id: str | None, session_id: str | None) -> dict[str, Any]:
    from tools.browser_workstation import _dispatch

    response = _dispatch(action, args, task_id=task_id, session_id=session_id)
    return json.loads(response) if isinstance(response, str) else response


def _error(message: str, *, code: str) -> str:
    return json.dumps({"success": False, "code": code, "error": message}, ensure_ascii=False)


def _install(args: dict[str, Any], *, task_id: str | None, session_id: str | None) -> str:
    if not _desktop_session():
        return _error("Chrome extensions are available only to a Hermes Desktop session.", code="desktop_session_required")
    identifier = str(args.get("extension") or "").strip()
    manager = ChromeExtensionManager()
    journal = _journal(task_id, session_id)
    try:
        extension_id = manager.extract_extension_id(identifier)
        journal.record(ExecutionEventKind.LIFECYCLE, "Extension download started", metadata={"event": "EXTENSION_DOWNLOAD_STARTED", "extension_id": extension_id})
        crx = manager.download_crx(extension_id)
        manifest, assessment = manager.inspect_crx(extension_id, crx)
        risk = RiskLevel(assessment["risk_level"])
        evaluation = ScopedPolicyEngine().evaluate(ActionScope(
            task_id=journal.task_id,
            session_id=str(session_id or ""),
            capability="browser_extension",
            action_name="install_extension",
            target=extension_id,
            parameters={"extension_risk": risk.value, "assessment": assessment},
            human_in_the_loop=True,
        ))
        journal.record(ExecutionEventKind.LIFECYCLE, "Extension policy evaluated", risk=evaluation.risk_level, metadata={"event": "EXTENSION_POLICY_CHECK", "extension_id": extension_id, "assessment": assessment, "policy": evaluation.to_dict()})
        if evaluation.decision == PolicyDecision.DENY:
            journal.record(ExecutionEventKind.ERROR, "Extension denied by policy", risk=evaluation.risk_level, metadata={"event": "EXTENSION_POLICY_DENIED", "extension_id": extension_id, "reason": evaluation.reason})
            return _error(evaluation.reason, code="policy_denied")
        if evaluation.decision == PolicyDecision.REQUIRE_APPROVAL:
            journal.record(ExecutionEventKind.APPROVAL_REQUESTED, "Extension installation requires approval", risk=evaluation.risk_level, metadata={"event": "EXTENSION_APPROVAL_REQUESTED", "extension_id": extension_id, "assessment": assessment})
            approved, message = _approval(
                f"Install Chrome extension {manifest.get('name') or extension_id} with {risk.value} risk: " + "; ".join(assessment["reasons"]),
                f"workstation-extension:{extension_id}:{risk.value}",
            )
            journal.record(ExecutionEventKind.APPROVAL_RESOLVED, "Extension approval resolved", risk=evaluation.risk_level, metadata={"event": "EXTENSION_APPROVED" if approved else "EXTENSION_APPROVAL_DENIED", "extension_id": extension_id})
            if not approved:
                return _error(message, code="approval_denied")
        installed = manager.install_from_bytes(extension_id, crx)
        journal.record(ExecutionEventKind.LIFECYCLE, "Extension installed to Workstation store", risk=risk, metadata={"event": "EXTENSION_INSTALLED", "extension_id": extension_id, "version": installed["version"]})
        loaded = _controller("browser_extension_load", {"extension_id": extension_id, "path": installed["path"]}, task_id=task_id, session_id=session_id)
        if not loaded.get("loaded"):
            manager.uninstall_extension(extension_id)
            journal.record(ExecutionEventKind.ERROR, "Extension failed verification and was rolled back", risk=risk, metadata={"event": "EXTENSION_LOAD_FAILED", "extension_id": extension_id})
            return _error("Electron did not verify the extension as loaded; the local installation was rolled back.", code="extension_load_failed")
        journal.record(ExecutionEventKind.LIFECYCLE, "Extension loaded and verified in Chromium", risk=risk, metadata={"event": "EXTENSION_VERIFIED", "extension_id": extension_id, "version": loaded.get("version", installed["version"])})
        return json.dumps({"success": True, "extension": installed, "runtime": loaded}, ensure_ascii=False)
    except Exception as exc:
        journal.record(ExecutionEventKind.ERROR, "Extension installation failed", metadata={"event": "EXTENSION_INSTALL_FAILED", "error": str(exc)[:500]})
        return _error(str(exc), code="extension_install_failed")


def _list(args: dict[str, Any], *, task_id: str | None, session_id: str | None) -> str:
    if not _desktop_session():
        return _error("Chrome extensions are available only to a Hermes Desktop session.", code="desktop_session_required")
    manager = ChromeExtensionManager()
    extensions = []
    for extension in manager.list_installed_extensions():
        try:
            runtime = _controller("browser_extension_verify", {"extension_id": extension["id"]}, task_id=task_id, session_id=session_id)
        except Exception as exc:
            runtime = {"loaded": False, "error": str(exc)[:300]}
        extensions.append({**extension, "runtime": runtime})
    return json.dumps({"success": True, "extensions": extensions}, ensure_ascii=False)


def _uninstall(args: dict[str, Any], *, task_id: str | None, session_id: str | None) -> str:
    if not _desktop_session():
        return _error("Chrome extensions are available only to a Hermes Desktop session.", code="desktop_session_required")
    manager = ChromeExtensionManager()
    journal = _journal(task_id, session_id)
    try:
        extension_id = manager.extract_extension_id(str(args.get("extension") or ""))
        evaluation = ScopedPolicyEngine().evaluate(ActionScope(task_id=journal.task_id, session_id=str(session_id or ""), capability="browser_extension", action_name="uninstall_extension", target=extension_id, parameters={"extension_risk": "medium"}, human_in_the_loop=True))
        if evaluation.decision == PolicyDecision.REQUIRE_APPROVAL:
            approved, message = _approval(f"Remove Chrome extension {extension_id} from the Hermes Workstation browser profile.", f"workstation-extension-remove:{extension_id}")
            if not approved:
                return _error(message, code="approval_denied")
        _controller("browser_extension_remove", {"extension_id": extension_id}, task_id=task_id, session_id=session_id)
        removed = manager.uninstall_extension(extension_id)
        journal.record(ExecutionEventKind.LIFECYCLE, "Extension uninstalled", metadata={"event": "EXTENSION_UNINSTALLED", "extension_id": extension_id, "removed": removed})
        return json.dumps({"success": True, "extension_id": extension_id, "removed": removed}, ensure_ascii=False)
    except Exception as exc:
        journal.record(ExecutionEventKind.ERROR, "Extension uninstall failed", metadata={"event": "EXTENSION_UNINSTALL_FAILED", "error": str(exc)[:500]})
        return _error(str(exc), code="extension_uninstall_failed")


def _open_options(args: dict[str, Any], *, task_id: str | None, session_id: str | None) -> str:
    if not _desktop_session():
        return _error("Chrome extensions are available only to a Hermes Desktop session.", code="desktop_session_required")
    manager = ChromeExtensionManager()
    try:
        extension_id = manager.extract_extension_id(str(args.get("extension") or ""))
        url = manager.get_options_url(extension_id)
        if not url:
            return _error("The installed extension does not declare an options page.", code="options_page_missing")
        options_path = url.split(f"chrome-extension://{extension_id}/", 1)[-1]
        result = _controller("browser_extension_open_options", {"extension_id": extension_id, "options_path": options_path}, task_id=task_id, session_id=session_id)
        _journal(task_id, session_id).record(ExecutionEventKind.LIFECYCLE, "Extension options opened", metadata={"event": "EXTENSION_USED", "extension_id": extension_id, "operation": "open_options"})
        return json.dumps({"success": True, "runtime": result}, ensure_ascii=False)
    except Exception as exc:
        return _error(str(exc), code="extension_options_failed")


def _schema(description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"name": "", "description": description, "parameters": {"type": "object", "properties": properties, "required": required or []}}


registry.register("browser_extension_install", _TOOLSET, _schema("Install a Chrome Web Store extension only when it is needed. Hermes inspects its manifest, applies policy and asks for approval when permissions are sensitive; success means Electron loaded and verified it.", {"extension": {"type": "string", "description": "Chrome Web Store URL or 32-character extension ID."}}, ["extension"]), _install, check_fn=lambda: False, emoji="🧩")
registry.register("browser_extension_list", _TOOLSET, _schema("List Hermes Workstation Chrome extensions with their current Electron loaded state.", {}), _list, check_fn=lambda: False, emoji="🧩")
registry.register("browser_extension_uninstall", _TOOLSET, _schema("Remove an installed Hermes Workstation Chrome extension after human approval.", {"extension": {"type": "string", "description": "Chrome Web Store URL or 32-character extension ID."}}, ["extension"]), _uninstall, check_fn=lambda: False, emoji="🧩")
registry.register("browser_extension_open_options", _TOOLSET, _schema("Open the options page of an already loaded Hermes Workstation Chrome extension in the task-owned internal browser.", {"extension": {"type": "string", "description": "Chrome Web Store URL or 32-character extension ID."}}, ["extension"]), _open_options, check_fn=lambda: False, emoji="🧩")
