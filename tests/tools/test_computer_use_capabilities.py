"""Capability/status and action-risk tests for computer-use/browser UIA wave 1."""

from __future__ import annotations


def test_windows_native_gui_click_type_drag_are_blocked():
    from tools.computer_use.capabilities import classify_action_risk

    for action in ("click", "type", "drag"):
        decision = classify_action_risk(
            surface="native_gui",
            action=action,
            platform="win32",
        )

        assert decision.allowed is False
        assert decision.risk == "blocked"
        assert "Windows native GUI" in decision.reason
        assert "forbidden" in decision.reason


def test_browser_read_only_actions_are_low_risk():
    from tools.computer_use.capabilities import classify_action_risk

    decision = classify_action_risk(surface="browser", action="browser_snapshot")

    assert decision.allowed is True
    assert decision.risk == "low"
    assert decision.requires_approval is False
    assert "read-only" in decision.reason


def test_browser_mutation_is_high_risk_and_requires_approval():
    from tools.computer_use.capabilities import classify_action_risk

    decision = classify_action_risk(surface="browser", action="browser_type")

    assert decision.allowed is True
    assert decision.risk == "high"
    assert decision.requires_approval is True
    assert "mutates browser page state" in decision.reason


def test_sensitive_browser_mutation_is_blocked_before_execution():
    from tools.computer_use.capabilities import classify_action_risk

    decision = classify_action_risk(
        surface="browser",
        action="browser_click",
        task="approve the 2FA login prompt",
    )

    assert decision.allowed is False
    assert decision.risk == "blocked"
    assert "sensitive" in decision.reason.lower()


def test_browser_cdp_diagnosis_reports_reachable_override(monkeypatch):
    import tools.computer_use.capabilities as capabilities

    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    monkeypatch.setattr(capabilities, "is_browser_debug_ready", lambda url, timeout=1.0: True)

    diagnosis = capabilities.diagnose_browser_availability(probe_cdp=True)

    assert diagnosis.available is True
    assert diagnosis.mode == "cdp"
    assert diagnosis.cdp_url == "http://127.0.0.1:9222"
    assert "reachable" in diagnosis.reason


def test_browser_cdp_diagnosis_reports_unreachable_override(monkeypatch):
    import tools.computer_use.capabilities as capabilities

    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    monkeypatch.setattr(capabilities, "is_browser_debug_ready", lambda url, timeout=1.0: False)

    diagnosis = capabilities.diagnose_browser_availability(probe_cdp=True)

    assert diagnosis.available is False
    assert diagnosis.mode == "cdp"
    assert "not reachable" in diagnosis.reason
    assert "hermes /browser connect" in diagnosis.next_step


def test_structured_status_json_includes_browser_routes_and_risk_policy(monkeypatch):
    import tools.computer_use.capabilities as capabilities

    fake_browser = capabilities.BrowserAvailability(
        available=True,
        mode="cdp",
        reason="CDP endpoint is reachable.",
        next_step="Use the browser toolset.",
        cdp_url="http://127.0.0.1:9222",
    )
    monkeypatch.setattr(
        capabilities,
        "diagnose_browser_availability",
        lambda probe_cdp=True: fake_browser,
    )

    status = capabilities.computer_use_capability_status(platform="win32", probe_browser=True)

    assert status["platform"] == "win32"
    assert status["computer_use"]["available"] is False
    assert status["computer_use"]["native_gui_mutation_allowed"] is False
    assert status["browser"]["available"] is True
    assert status["browser"]["mode"] == "cdp"
    forbidden = status["risk_policy"]["windows_native_gui_forbidden_actions"]
    assert all(action in forbidden for action in ["click", "drag", "type"])
    assert any(route["route"] == "browser" for route in status["routes"])


def test_structured_status_routes_url_as_unsupported_when_browser_unavailable(monkeypatch):
    import tools.computer_use.capabilities as capabilities

    fake_browser = capabilities.BrowserAvailability(
        available=False,
        mode="cloud:Browser Use",
        reason="Browser provider credentials are not configured.",
        next_step="Configure browser provider credentials.",
    )
    monkeypatch.setattr(
        capabilities,
        "diagnose_browser_availability",
        lambda probe_cdp=True: fake_browser,
    )

    status = capabilities.computer_use_capability_status(platform="win32", probe_browser=True)
    url_route = status["routes"][0]

    assert status["browser"]["available"] is False
    assert url_route["route"] == "unsupported"
    assert url_route["available"] is False
    assert "browser automation is not available" in url_route["reason"]


def test_action_proposal_dry_run_schema_never_executes_native_input():
    from tools.computer_use.proposals import (
        BrowserActionProposal,
        DesktopActionProposal,
        ElementRef,
        dry_run_action_proposal,
    )

    browser = BrowserActionProposal(
        action="type_text",
        target=ElementRef(selector="#search", role="textbox", name="Search"),
        text="hello",
        origin="https://example.com",
        expected_result="query text appears",
    )
    browser_result = dry_run_action_proposal(browser, platform="win32")

    assert browser_result.will_execute is False
    assert browser_result.surface == "browser"
    assert browser_result.action == "type_text"
    assert browser_result.risk == "high"
    assert browser_result.requires_approval is True
    assert browser_result.proposal["target"]["selector"] == "#search"

    native_drag = DesktopActionProposal(
        surface="windows_input_fallback",
        action="drag_coord",
        target_selector={"x": 10, "y": 10, "to_x": 20, "to_y": 20},
        expected_result="drag item",
    )
    native_result = dry_run_action_proposal(native_drag, platform="win32")

    assert native_result.will_execute is False
    assert native_result.allowed is False
    assert native_result.risk == "blocked"
    assert "disabled" in native_result.reason.lower()


def test_sensitive_browser_action_proposal_is_blocked():
    from tools.computer_use.proposals import BrowserActionProposal, ElementRef, dry_run_action_proposal

    proposal = BrowserActionProposal(
        action="click",
        target=ElementRef(selector="button.approve", role="button", name="Approve"),
        origin="https://accounts.example",
        task="approve the MFA login prompt",
    )

    result = dry_run_action_proposal(proposal, platform="win32")

    assert result.will_execute is False
    assert result.allowed is False
    assert result.risk == "blocked"
    assert "sensitive" in result.reason.lower()


def test_windows_uia_readonly_backend_skeleton_has_no_mutation_methods():
    from tools.computer_use.windows_uia_readonly import WindowsUiaReadOnlyBackend

    backend = WindowsUiaReadOnlyBackend(platform="win32")
    caps = backend.capabilities()

    assert caps.read_only is True
    assert caps.mutation_allowed is False
    assert "list_windows" in caps.supports
    assert "snapshot_tree" in caps.supports
    assert "element_capabilities" in caps.supports
    for method_name in ("click", "drag", "type_text", "key", "set_value"):
        assert not hasattr(backend, method_name)

    windows = backend.list_windows()
    assert windows["read_only"] is True
    assert windows["would_execute_native_input"] is False


def test_status_json_exposes_input_proposal_and_uia_readonly_capabilities(monkeypatch):
    import tools.computer_use.capabilities as capabilities

    fake_browser = capabilities.BrowserAvailability(
        available=False,
        mode="cloud:Browser Use",
        reason="Browser provider credentials are not configured.",
        next_step="Configure browser provider credentials.",
    )
    monkeypatch.setattr(
        capabilities,
        "diagnose_browser_availability",
        lambda probe_cdp=True: fake_browser,
    )

    status = capabilities.computer_use_capability_status(platform="win32", probe_browser=False)

    proposals = status["input_proposals"]
    assert proposals["available"] is True
    assert proposals["dry_run_only"] is True
    assert "BrowserActionProposal" in proposals["schemas"]
    assert "DesktopActionProposal" in proposals["schemas"]
    assert "click" in proposals["browser_actions"]
    assert "drag" in proposals["browser_actions"]

    uia = status["windows_uia_readonly"]
    assert uia["backend"] == "pywinauto-uia-readonly"
    assert uia["read_only"] is True
    assert uia["mutation_allowed"] is False
    assert uia["input_fallback_enabled"] is False
