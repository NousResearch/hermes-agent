"""Tests for safe desktop-control route decisions."""

from __future__ import annotations


def test_url_target_routes_to_browser_on_windows_when_available():
    from tools.computer_use.routing import route_desktop_request

    decision = route_desktop_request(
        target="https://example.com/dashboard",
        platform="win32",
        browser_available=True,
    )

    assert decision.route == "browser"
    assert decision.available is True
    assert decision.read_only is True
    assert "URL" in decision.reason


def test_local_path_routes_to_file_terminal():
    from tools.computer_use.routing import route_desktop_request

    decision = route_desktop_request(
        target="C:/Users/82109/AppData/Local/hermes/logs/agent.log",
        platform="win32",
    )

    assert decision.route == "file_terminal"
    assert decision.available is True
    assert "local path" in decision.reason


def test_screenshot_target_routes_to_vision():
    from tools.computer_use.routing import route_desktop_request

    decision = route_desktop_request(
        target="screenshot",
        platform="win32",
        vision_available=True,
    )

    assert decision.route == "vision"
    assert decision.available is True
    assert "screenshot" in decision.reason.lower()


def test_windows_native_read_only_uses_uia_only_when_capability_injected():
    from tools.computer_use.routing import route_desktop_request

    unavailable = route_desktop_request(
        target="app:Settings",
        platform="win32",
        read_only=True,
        windows_uia_readonly_available=False,
    )
    available = route_desktop_request(
        target="app:Settings",
        platform="win32",
        read_only=True,
        windows_uia_readonly_available=True,
    )

    assert unavailable.route == "unsupported"
    assert unavailable.available is False
    assert "Windows UIA read-only backend is not available" in unavailable.reason
    assert available.route == "windows_uia_readonly"
    assert available.available is True
    assert "read-only" in available.reason


def test_windows_native_mutation_is_blocked_even_if_uia_exists():
    from tools.computer_use.routing import route_desktop_request

    decision = route_desktop_request(
        intent="operate",
        target="app:Settings",
        platform="win32",
        read_only=False,
        windows_uia_readonly_available=True,
    )

    assert decision.route == "unsupported"
    assert decision.available is False
    assert "Windows native GUI mutation is not supported" in decision.reason


def test_macos_native_routes_to_computer_use_when_available():
    from tools.computer_use.routing import route_desktop_request

    decision = route_desktop_request(
        target="app:TextEdit",
        platform="darwin",
        read_only=False,
        computer_use_available=True,
    )

    assert decision.route == "computer_use"
    assert decision.available is True
    assert "macOS" in decision.reason


def test_sensitive_native_task_is_unsupported_before_backend_choice():
    from tools.computer_use.routing import route_desktop_request

    decision = route_desktop_request(
        intent="operate",
        target="app:Browser",
        task="approve the 2FA login prompt",
        platform="darwin",
        read_only=False,
        computer_use_available=True,
    )

    assert decision.route == "unsupported"
    assert decision.available is False
    assert "sensitive" in decision.reason.lower()
