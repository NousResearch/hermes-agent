"""Browser input execution v1.

This module executes browser input proposals only through an explicitly injected
browser backend. It does not launch browsers, call paid/cloud providers, access
native Windows GUI automation, or use SendInput/pyautogui/pynput. The default
capability status is therefore conservative: the executor exists, but no live
backend is configured by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

from .proposals import BrowserActionProposal, dry_run_action_proposal

_APPROVAL_TOKENS = frozenset({"approved", "approve_once", "approve_session", "always_approve"})
_SUPPORTED_ACTIONS = ("click", "type_text", "key_combo", "drag", "scroll")
_SENSITIVE_FIELD_MARKERS = frozenset({"password", "passwd", "pwd", "secret", "token", "oauth", "2fa", "mfa", "otp", "credential", "cookie"})
_DESTRUCTIVE_LABEL_MARKERS = frozenset({"delete", "remove", "destroy", "drop", "revoke", "reset", "wipe", "erase", "purchase", "buy", "pay", "결제", "삭제", "초기화"})


class BrowserInputBackend(Protocol):
    def is_available(self) -> bool: ...

    def click(self, target: dict[str, Any], *, button: str = "left") -> dict[str, Any]: ...

    def type_text(self, target: dict[str, Any] | None, text: str) -> dict[str, Any]: ...

    def key_combo(self, keys: tuple[str, ...]) -> dict[str, Any]: ...

    def drag(self, source: dict[str, Any] | None, target: dict[str, Any] | None) -> dict[str, Any]: ...

    def scroll(self, target: dict[str, Any] | None, delta_x: int = 0, delta_y: int = 0) -> dict[str, Any]: ...

class CdpTransport(Protocol):
    """Tiny transport seam for tests and future CDP clients.

    The real network/WebSocket adapter is intentionally not implemented here:
    unit tests inject a fake transport, and production code must explicitly pass
    a vetted transport instance.
    """

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]: ...


def _redact_url(raw: str) -> str:
    parsed = urlparse(raw or "")
    if not parsed.scheme or not parsed.netloc:
        return raw or ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _domain_from_url(raw: str) -> str:
    return (urlparse(raw or "").hostname or "").lower()


def _target_text(target: dict[str, Any] | None) -> str:
    if not target:
        return ""
    pieces = [str(target.get("selector") or ""), str(target.get("role") or ""), str(target.get("name") or "")]
    metadata = target.get("metadata") or {}
    if isinstance(metadata, dict):
        pieces.extend(str(v) for v in metadata.values())
    return " ".join(pieces).lower()


def _has_sensitive_marker(text: str) -> bool:
    lower = (text or "").lower()
    return any(marker in lower for marker in _SENSITIVE_FIELD_MARKERS)


def _has_destructive_marker(text: str) -> bool:
    lower = (text or "").lower()
    return any(marker in lower for marker in _DESTRUCTIVE_LABEL_MARKERS)


@dataclass(frozen=True)
class LocalCdpBrowserBackend:
    """Explicitly injected local CDP backend with conservative safety gates.

    This class does not launch browsers, discover endpoints, open sockets by
    itself, or use native OS input. Callers must pass a fake or vetted CDP
    transport object. Unit tests should use fake transports only.
    """

    endpoint: str
    domain_allowlist: tuple[str, ...]
    transport: CdpTransport | None = None
    redact_sensitive: bool = True

    def is_available(self) -> bool:
        if self.transport is None:
            return False
        parsed = urlparse(self.endpoint or "")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return False
        if not self.domain_allowlist:
            return False
        try:
            result = self.transport.call("Browser.getVersion", {})
        except Exception:
            return False
        return bool(result.get("ok", True))

    def _current_page(self) -> dict[str, Any]:
        assert self.transport is not None
        result = self.transport.call("Hermes.getActivePage", {})
        url = str(result.get("url") or "")
        domain = _domain_from_url(url)
        if domain not in {d.lower() for d in self.domain_allowlist}:
            return {"ok": False, "error": "domain_not_allowed", "domain": domain, "url": _redact_url(url)}
        return {"ok": True, "domain": domain, "url": _redact_url(url), "tab_id_hash": str(result.get("tab_id_hash") or "")}

    def _safe_target(self, target: dict[str, Any] | None, *, allow_destructive: bool = False) -> tuple[bool, str, dict[str, Any]]:
        text = _target_text(target)
        if _has_sensitive_marker(text):
            return False, "target appears to be a password/secret/token/OAuth/2FA field", self._redacted_target(target)
        if not allow_destructive and _has_destructive_marker(text):
            return False, "target label appears destructive or payment-related", self._redacted_target(target)
        return True, "ok", self._redacted_target(target)

    def _redacted_target(self, target: dict[str, Any] | None) -> dict[str, Any]:
        if not target:
            return {}
        redacted = dict(target)
        if self.redact_sensitive:
            metadata = redacted.get("metadata")
            if isinstance(metadata, dict):
                redacted["metadata"] = {k: ("[REDACTED]" if _has_sensitive_marker(str(k)) or _has_sensitive_marker(str(v)) else v) for k, v in metadata.items()}
            if _has_sensitive_marker(str(redacted.get("name") or "")):
                redacted["name"] = "[REDACTED]"
        return redacted

    def _action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.transport is None:
            return {"ok": False, "error": "missing_transport", "action": action}
        page = self._current_page()
        if not page.get("ok"):
            return {"ok": False, "action": action, **page}
        result = self.transport.call(f"Hermes.{action}", params)
        return {"ok": bool(result.get("ok", True)), "action": action, "domain": page.get("domain"), "url": page.get("url"), "tab_id_hash": page.get("tab_id_hash"), "transport_result": dict(result)}

    def click(self, target: dict[str, Any], *, button: str = "left") -> dict[str, Any]:
        ok, reason, redacted = self._safe_target(target)
        if not ok:
            return {"ok": False, "action": "click", "error": reason, "target": redacted}
        return self._action("click", {"target": redacted, "button": button})

    def type_text(self, target: dict[str, Any] | None, text: str) -> dict[str, Any]:
        ok, reason, redacted = self._safe_target(target)
        if not ok or _has_sensitive_marker(text):
            return {"ok": False, "action": "type_text", "error": reason if not ok else "text appears to contain sensitive credential-like content", "target": redacted}
        return self._action("typeText", {"target": redacted, "text_length": len(text)})

    def key_combo(self, keys: tuple[str, ...]) -> dict[str, Any]:
        return self._action("keyCombo", {"keys": list(keys)})

    def drag(self, source: dict[str, Any] | None, target: dict[str, Any] | None) -> dict[str, Any]:
        ok_s, reason_s, redacted_s = self._safe_target(source)
        ok_t, reason_t, redacted_t = self._safe_target(target)
        if not ok_s or not ok_t:
            return {"ok": False, "action": "drag", "error": reason_s if not ok_s else reason_t, "source": redacted_s, "target": redacted_t}
        return self._action("drag", {"source": redacted_s, "target": redacted_t})

    def scroll(self, target: dict[str, Any] | None, delta_x: int = 0, delta_y: int = 0) -> dict[str, Any]:
        ok, reason, redacted = self._safe_target(target, allow_destructive=True)
        if not ok:
            return {"ok": False, "action": "scroll", "error": reason, "target": redacted}
        return self._action("scroll", {"target": redacted, "delta_x": delta_x, "delta_y": delta_y})



@dataclass(frozen=True)
class BrowserInputExecutionResult:
    ok: bool
    executed: bool
    action: str
    risk: str
    requires_approval: bool
    reason: str
    proposal: dict[str, Any]
    backend_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrowserInputExecutor:
    """Execute browser proposals through an injected backend after dry-run gates."""

    def __init__(self, *, backend: BrowserInputBackend) -> None:
        self.backend = backend

    def execute(
        self,
        proposal: BrowserActionProposal,
        *,
        platform: str | None = None,
        approval_token: str = "",
    ) -> BrowserInputExecutionResult:
        dry_run = dry_run_action_proposal(proposal, platform=platform)
        proposal_dict = dry_run.proposal

        if not dry_run.allowed:
            return BrowserInputExecutionResult(
                ok=False,
                executed=False,
                action=proposal.action,
                risk=dry_run.risk,
                requires_approval=dry_run.requires_approval,
                reason=dry_run.reason,
                proposal=proposal_dict,
            )

        if dry_run.requires_approval and approval_token not in _APPROVAL_TOKENS:
            return BrowserInputExecutionResult(
                ok=False,
                executed=False,
                action=proposal.action,
                risk=dry_run.risk,
                requires_approval=True,
                reason="Browser input action requires explicit approval before execution.",
                proposal=proposal_dict,
            )

        try:
            available = bool(self.backend.is_available())
        except Exception as exc:
            return BrowserInputExecutionResult(
                ok=False,
                executed=False,
                action=proposal.action,
                risk=dry_run.risk,
                requires_approval=dry_run.requires_approval,
                reason=f"Browser input backend availability check failed: {exc}",
                proposal=proposal_dict,
            )
        if not available:
            return BrowserInputExecutionResult(
                ok=False,
                executed=False,
                action=proposal.action,
                risk=dry_run.risk,
                requires_approval=dry_run.requires_approval,
                reason="Browser input backend is not available.",
                proposal=proposal_dict,
            )

        target = proposal.target.to_dict() if proposal.target is not None else None
        backend_result = self._execute_backend_action(proposal, target)
        return BrowserInputExecutionResult(
            ok=bool(backend_result.get("ok", True)),
            executed=True,
            action=proposal.action,
            risk=dry_run.risk,
            requires_approval=dry_run.requires_approval,
            reason="Browser input action executed through injected backend.",
            proposal=proposal_dict,
            backend_result=dict(backend_result),
        )

    def _execute_backend_action(
        self,
        proposal: BrowserActionProposal,
        target: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if proposal.action == "click":
            return self.backend.click(target or {}, button="left")
        if proposal.action == "type_text":
            return self.backend.type_text(target, proposal.text)
        if proposal.action == "key_combo":
            return self.backend.key_combo(tuple(proposal.keys))
        if proposal.action == "drag":
            source = proposal.target.to_dict() if proposal.target is not None else None
            destination = proposal.target.to_dict() if proposal.target is not None else None
            return self.backend.drag(source, destination)
        if proposal.action == "scroll":
            return self.backend.scroll(target, delta_x=0, delta_y=0)
        return {"ok": False, "action": proposal.action, "error": "unsupported action"}


def browser_input_execution_capability_status() -> dict[str, Any]:
    return {
        "available": False,
        "requires_injected_backend": True,
        "native_gui_mutation_allowed": False,
        "launches_browser": False,
        "uses_paid_or_cloud_provider": False,
        "supported_actions": list(_SUPPORTED_ACTIONS),
        "local_cdp_backend_supported": True,
        "local_cdp_requires_explicit_transport": True,
        "local_cdp_loopback_only_by_default": True,
        "approval_required_for_high_risk": True,
        "sensitive_tasks_blocked": True,
        "next_step": "Inject a vetted local CDP transport explicitly; this module will not launch, auto-discover, or mutate native GUI state.",
    }
