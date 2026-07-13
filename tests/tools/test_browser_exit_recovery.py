import json
import subprocess
from unittest.mock import patch

import pytest

from tools.browser_exit_recovery import (
    browser_exit_recover,
    build_exit_recovery_hint,
    classify_exit_issue,
    recover_browser_exit,
    resolve_exit_controller,
)


@pytest.mark.parametrize(
    ("message", "rule"),
    [
        ("net::ERR_PROXY_CONNECTION_FAILED while navigating", "proxy_connection_failed"),
        ("ERR_TUNNEL_CONNECTION_FAILED", "proxy_connection_failed"),
        ("Temporary failure in name resolution", "dns_failure"),
        ("socks5 connection refused", "socks_or_exit_unreachable"),
        ("site can't be reached because connection timed out", "connection_timeout_or_reset"),
        ("504 gateway timeout", "tls_or_gateway_failure"),
    ],
)
def test_classifies_network_proxy_and_exit_failures_as_recoverable(message, rule):
    classification = classify_exit_issue(error=message)

    assert classification["status"] == "recoverable"
    assert classification["safe_to_change_exit"] is True
    assert classification["policy"] == "network_health_failover_only"
    assert classification["matches"][0]["rule"] == rule


@pytest.mark.parametrize(
    "message",
    [
        "Please complete the reCAPTCHA to continue",
        "Verify you are human before accessing this site",
        "Cloudflare access denied error 1020",
        "403 Forbidden",
        "bot detected due to automated traffic",
        "429 too many requests",
    ],
)
def test_classifies_captcha_access_control_and_bot_walls_as_not_allowed(message):
    classification = classify_exit_issue(error=message)

    assert classification["status"] == "not_allowed"
    assert classification["safe_to_change_exit"] is False
    assert classification["policy"] in {
        "do_not_solve_or_bypass_captcha",
        "do_not_evade_access_controls",
    }


def test_autonomy_blockers_win_over_network_signals():
    classification = classify_exit_issue(
        error="net::ERR_PROXY_CONNECTION_FAILED",
        snapshot='text: Verify you are human\nbutton "I am not a robot" [ref=e1]',
    )

    assert classification["status"] == "not_allowed"
    assert classification["kind"] == "human_verification"


@pytest.mark.parametrize(
    "message",
    [
        "Cloudflare 503 Service Unavailable checking your browser",
        "Just a moment... cf-chl challenge-platform",
        "Attention required: DDoS protection",
    ],
)
def test_challenge_wall_errors_do_not_trigger_exit_recovery(message):
    classification = classify_exit_issue(error=message)

    assert classification["status"] == "not_allowed"
    assert classification["safe_to_change_exit"] is False
    assert classification["policy"] in {
        "do_not_solve_or_bypass_captcha",
        "do_not_evade_access_controls",
    }


def test_build_exit_recovery_hint_recommends_manual_tool_for_recoverable_error():
    hint = build_exit_recovery_hint(
        error="ERR_PROXY_CONNECTION_FAILED",
        url="https://example.test",
        auto_recovery_enabled=False,
    )

    assert hint["eligible"] is True
    assert hint["auto_recovery_enabled"] is False
    assert hint["tool"] == "browser_exit_recover"
    assert hint["recommended_args"]["preferred_exit"] == "residential"
    assert hint["recommended_args"]["allow_direct"] is False


def test_resolve_exit_controller_requires_absolute_executable_hermes_control(tmp_path, monkeypatch):
    wrong_name = tmp_path / "switch-exit.sh"
    wrong_name.write_text("#!/bin/sh\n", encoding="utf-8")
    wrong_name.chmod(0o755)
    monkeypatch.setenv("HERMES_BROWSER_EXIT_CONTROLLER", str(wrong_name))

    assert resolve_exit_controller() is None

    controller = tmp_path / "hermes-control.sh"
    controller.write_text("#!/bin/sh\n", encoding="utf-8")
    controller.chmod(0o755)
    monkeypatch.setenv("HERMES_BROWSER_EXIT_CONTROLLER", str(controller))

    assert resolve_exit_controller() == str(controller)


def test_recover_browser_exit_refuses_forbidden_failure_without_subprocess(monkeypatch):
    import tools.browser_exit_recovery as recovery

    def fail_run(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("forbidden failures must not run recovery")

    monkeypatch.setattr(recovery, "_run_controller", fail_run)

    result = recover_browser_exit(reason="verify you are human", url="https://example.test")

    assert result["success"] is False
    assert result["status"] == "not_allowed"


def test_recover_browser_exit_runs_fixed_controller_args(monkeypatch, tmp_path):
    import tools.browser_exit_recovery as recovery

    controller = tmp_path / "hermes-control.sh"
    controller.write_text("#!/bin/sh\n", encoding="utf-8")
    controller.chmod(0o755)
    calls = []

    def fake_run(controller_path, args, timeout_s):
        calls.append((controller_path, args, timeout_s))
        return {
            "returncode": 0,
            "ok": True,
            "stdout": "ok",
            "stderr": "",
        }

    monkeypatch.setattr(recovery, "resolve_exit_controller", lambda: str(controller))
    monkeypatch.setattr(recovery, "_run_controller", fake_run)
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {
        "browser": {
            "exit_recovery": {
                "fallback_exits": ["residential", "vpn"],
                "timeout_s": 7,
            }
        }
    })

    result = recover_browser_exit(
        reason="ERR_PROXY_CONNECTION_FAILED",
        url="https://example.test",
        preferred_exit="residential",
    )

    assert result["success"] is True
    assert result["selected_exit"] == "residential"
    assert calls == [
        (str(controller), ["exit", "set", "residential"], 7),
        (str(controller), ["exit", "verify", "residential"], 7),
    ]


def test_direct_exit_requires_operator_config_even_when_tool_arg_allows(monkeypatch):
    import tools.browser_exit_recovery as recovery

    monkeypatch.setattr(recovery, "resolve_exit_controller", lambda: "/tmp/hermes-control.sh")
    monkeypatch.setattr(
        recovery,
        "_run_controller",
        lambda *args, **kwargs: pytest.fail("direct must not run without config allow"),
    )
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {
        "browser": {
            "exit_recovery": {
                "fallback_exits": ["residential", "vpn", "direct"],
                "allow_direct_fallback": False,
            }
        }
    })

    result = recover_browser_exit(
        reason="ERR_PROXY_CONNECTION_FAILED",
        preferred_exit="direct",
        allow_direct=True,
    )

    assert result["success"] is False
    assert result["status"] == "invalid_exit"
    assert "Direct exit fallback is disabled" in result["message"]


@pytest.mark.parametrize("fallback_exits", [[], ["unknown"], ["direct"]])
def test_restrictive_fallback_exits_do_not_fail_open(monkeypatch, fallback_exits):
    import tools.browser_exit_recovery as recovery

    monkeypatch.setattr(recovery, "resolve_exit_controller", lambda: "/tmp/hermes-control.sh")
    monkeypatch.setattr(
        recovery,
        "_run_controller",
        lambda *args, **kwargs: pytest.fail("no configured safe exit should run"),
    )
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {
        "browser": {
            "exit_recovery": {
                "fallback_exits": fallback_exits,
                "allow_direct_fallback": False,
            }
        }
    })

    result = recover_browser_exit(reason="ERR_PROXY_CONNECTION_FAILED")

    assert result["success"] is False
    assert result["status"] == "invalid_exit"
    assert result["message"] == "No safe exit profile is configured."


def test_run_controller_uses_subprocess_without_shell(monkeypatch, tmp_path):
    import tools.browser_exit_recovery as recovery

    controller = tmp_path / "hermes-control.sh"
    controller.write_text("#!/bin/sh\n", encoding="utf-8")
    controller.chmod(0o755)
    calls = []

    def fake_subprocess_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="ready", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    result = recovery._run_controller(str(controller), ["exit", "set", "residential"], 9)

    assert result["ok"] is True
    args, kwargs = calls[0]
    assert args == [str(controller), "exit", "set", "residential"]
    assert kwargs["check"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 9
    assert "shell" not in kwargs


def test_browser_exit_recover_tool_shape(monkeypatch):
    import tools.browser_exit_recovery as recovery

    monkeypatch.setattr(recovery, "recover_browser_exit", lambda **kwargs: {
        "success": True,
        "status": "changed",
        "selected_exit": kwargs["preferred_exit"],
    })

    response = json.loads(
        browser_exit_recover(
            reason="ERR_PROXY_CONNECTION_FAILED",
            preferred_exit="vpn",
        )
    )

    assert response["success"] is True
    assert response["ip_recovery"]["selected_exit"] == "vpn"


def test_browser_exit_tool_is_in_browser_toolset():
    from hermes_cli.config import DEFAULT_CONFIG, OPTIONAL_ENV_VARS
    from toolsets import TOOLSETS, _HERMES_CORE_TOOLS
    from tools.registry import registry
    import tools.browser_exit_recovery  # noqa: F401

    assert "browser_exit_recover" in TOOLSETS["browser"]["tools"]
    assert "browser_exit_recover" in _HERMES_CORE_TOOLS
    assert "browser_exit_recover" in registry._tools
    assert DEFAULT_CONFIG["browser"]["exit_recovery"]["auto_recover"] is False
    assert DEFAULT_CONFIG["browser"]["exit_recovery"]["allow_direct_fallback"] is False
    # Non-secret feature flags live in config.yaml (browser.exit_recovery.*),
    # never as user-facing HERMES_* env vars.
    assert "HERMES_BROWSER_AUTO_EXIT_RECOVERY" not in OPTIONAL_ENV_VARS
    assert "HERMES_BROWSER_EXIT_CONTROLLER" not in OPTIONAL_ENV_VARS


def test_browser_navigate_failure_adds_ip_recovery_hint_without_auto(monkeypatch):
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_tool, "_browser_exit_auto_recovery_enabled", lambda: False)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda task_id: {
            "session_name": "test",
            "_first_nav": False,
            "features": {"local": True, "proxies": True},
        },
    )

    with patch(
        "tools.browser_tool._run_browser_command",
        return_value={
            "success": False,
            "error": "net::ERR_PROXY_CONNECTION_FAILED",
        },
    ), patch(
        "tools.browser_tool.recover_browser_exit",
        side_effect=AssertionError("auto recovery is disabled"),
    ):
        response = json.loads(
            browser_tool.browser_navigate("https://example.test", task_id="ip-hint")
        )

    assert response["success"] is False
    assert response["ip_recovery"]["eligible"] is True
    assert response["ip_recovery"]["auto_recovery_enabled"] is False
    assert response["ip_recovery"]["tool"] == "browser_exit_recover"
    browser_tool._last_active_session_key.pop("ip-hint", None)


def test_browser_navigate_auto_recovers_and_retries_once(monkeypatch):
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_tool, "_browser_exit_auto_recovery_enabled", lambda: True)
    monkeypatch.setattr(browser_tool, "_browser_exit_recovery_preferred_exit", lambda: "residential")
    monkeypatch.setattr(browser_tool, "_browser_exit_recovery_allow_direct", lambda: False)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda task_id: {
            "session_name": "test",
            "_first_nav": False,
            "features": {"local": True, "proxies": True},
        },
    )

    with patch(
        "tools.browser_tool.recover_browser_exit",
        return_value={"success": True, "status": "changed", "selected_exit": "residential"},
    ) as recover, patch(
        "tools.browser_tool._run_browser_command",
        side_effect=[
            {
                "success": False,
                "error": "net::ERR_PROXY_CONNECTION_FAILED",
            },
            {
                "success": True,
                "data": {"title": "Example", "url": "https://example.test/"},
            },
            {
                "success": True,
                "data": {
                    "snapshot": 'heading "Example" [ref=e1]',
                    "refs": {"e1": {}},
                },
            },
        ],
    ):
        response = json.loads(
            browser_tool.browser_navigate("https://example.test", task_id="ip-auto")
        )

    assert response["success"] is True
    assert response["ip_recovery"]["attempt"]["success"] is True
    assert response["ip_recovery"]["retried_navigation"] is True
    assert response["ip_recovery"]["retry_success"] is True
    recover.assert_called_once()
    browser_tool._last_active_session_key.pop("ip-auto", None)


def test_browser_navigate_does_not_auto_recover_access_control(monkeypatch):
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_tool, "_browser_exit_auto_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda task_id: {
            "session_name": "test",
            "_first_nav": False,
            "features": {"local": True, "proxies": True},
        },
    )

    with patch(
        "tools.browser_tool._run_browser_command",
        return_value={
            "success": False,
            "error": "403 Forbidden: verify you are human",
        },
    ), patch(
        "tools.browser_tool.recover_browser_exit",
        side_effect=AssertionError("access-control failures must not recover"),
    ):
        response = json.loads(
            browser_tool.browser_navigate("https://example.test", task_id="ip-blocked")
        )

    assert response["success"] is False
    assert response["ip_recovery"]["eligible"] is False
    assert response["ip_recovery"]["policy"] in {
        "do_not_solve_or_bypass_captcha",
        "do_not_evade_access_controls",
    }
    browser_tool._last_active_session_key.pop("ip-blocked", None)
