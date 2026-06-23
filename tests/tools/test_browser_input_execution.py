"""Tests for safe browser input execution v1."""

from __future__ import annotations


class RecordingBrowserBackend:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.calls = []

    def is_available(self) -> bool:
        return self.available

    def click(self, target, *, button="left"):
        self.calls.append(("click", target, button))
        return {"ok": True, "action": "click", "target": target}

    def type_text(self, target, text: str):
        self.calls.append(("type_text", target, text))
        return {"ok": True, "action": "type_text", "text_length": len(text)}

    def key_combo(self, keys):
        self.calls.append(("key_combo", tuple(keys)))
        return {"ok": True, "action": "key_combo"}

    def drag(self, source, target):
        self.calls.append(("drag", source, target))
        return {"ok": True, "action": "drag"}

    def scroll(self, target, delta_x=0, delta_y=0):
        self.calls.append(("scroll", target, delta_x, delta_y))
        return {"ok": True, "action": "scroll"}


def test_browser_input_executor_requires_approval_for_high_risk_type():
    from tools.computer_use.browser_input import BrowserInputExecutor
    from tools.computer_use.proposals import BrowserActionProposal, ElementRef

    backend = RecordingBrowserBackend()
    executor = BrowserInputExecutor(backend=backend)
    proposal = BrowserActionProposal(
        action="type_text",
        target=ElementRef(selector="#search", role="textbox"),
        text="hello",
        origin="https://example.com",
    )

    denied = executor.execute(proposal, platform="win32")
    assert denied.ok is False
    assert denied.executed is False
    assert denied.requires_approval is True
    assert backend.calls == []

    approved = executor.execute(proposal, platform="win32", approval_token="approved")
    assert approved.ok is True
    assert approved.executed is True
    assert approved.backend_result["action"] == "type_text"
    assert backend.calls == [("type_text", {"selector": "#search", "role": "textbox", "name": "", "point": None, "frame": "", "metadata": {}}, "hello")]


def test_sensitive_browser_input_is_blocked_even_with_approval():
    from tools.computer_use.browser_input import BrowserInputExecutor
    from tools.computer_use.proposals import BrowserActionProposal, ElementRef

    backend = RecordingBrowserBackend()
    executor = BrowserInputExecutor(backend=backend)
    proposal = BrowserActionProposal(
        action="click",
        target=ElementRef(selector="button.approve", role="button", name="Approve"),
        task="approve the 2FA login prompt",
    )

    result = executor.execute(proposal, platform="win32", approval_token="approved")

    assert result.ok is False
    assert result.executed is False
    assert result.risk == "blocked"
    assert "sensitive" in result.reason.lower()
    assert backend.calls == []


def test_browser_input_executor_runs_low_or_medium_risk_without_approval():
    from tools.computer_use.browser_input import BrowserInputExecutor
    from tools.computer_use.proposals import BrowserActionProposal, ElementRef

    backend = RecordingBrowserBackend()
    executor = BrowserInputExecutor(backend=backend)
    proposal = BrowserActionProposal(
        action="scroll",
        target=ElementRef(selector="main"),
        origin="https://example.com",
    )

    result = executor.execute(proposal, platform="win32")

    assert result.ok is True
    assert result.executed is True
    assert result.risk == "medium"
    assert backend.calls == [("scroll", {"selector": "main", "role": "", "name": "", "point": None, "frame": "", "metadata": {}}, 0, 0)]


def test_browser_input_executor_refuses_unavailable_backend():
    from tools.computer_use.browser_input import BrowserInputExecutor
    from tools.computer_use.proposals import BrowserActionProposal, ElementRef

    backend = RecordingBrowserBackend(available=False)
    executor = BrowserInputExecutor(backend=backend)
    proposal = BrowserActionProposal(action="click", target=ElementRef(selector="#go"))

    result = executor.execute(proposal, platform="win32", approval_token="approved")

    assert result.ok is False
    assert result.executed is False
    assert "not available" in result.reason.lower()
    assert backend.calls == []


def test_status_json_exposes_browser_input_execution_capability(monkeypatch):
    import tools.computer_use.capabilities as capabilities

    fake_browser = capabilities.BrowserAvailability(
        available=False,
        mode="cloud:Browser Use",
        reason="Browser provider credentials are not configured.",
        next_step="Configure browser provider credentials.",
    )
    monkeypatch.setattr(capabilities, "diagnose_browser_availability", lambda probe_cdp=True: fake_browser)

    status = capabilities.computer_use_capability_status(platform="win32", probe_browser=False)
    execution = status["browser_input_execution"]

    assert execution["available"] is False
    assert execution["requires_injected_backend"] is True
    assert execution["native_gui_mutation_allowed"] is False
    assert execution["local_cdp_backend_supported"] is True
    assert execution["local_cdp_requires_explicit_transport"] is True
    assert execution["local_cdp_loopback_only_by_default"] is True
    assert "click" in execution["supported_actions"]
    assert "drag" in execution["supported_actions"]
