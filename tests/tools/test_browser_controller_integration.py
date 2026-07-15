from __future__ import annotations

import base64
import hashlib
import os
import sys
from types import SimpleNamespace

from tools import browser_tool
from tools import browser_controller_client


def _client_mapping(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(mode=0o700)
    return {
        "schema": browser_controller_client.CLIENT_CONFIG_SCHEMA,
        "socket_path": str(runtime / "controller.sock"),
        "server_uid": os.geteuid(),
        "artifact_root": str(artifacts),
        "connect_timeout_seconds": 1,
        "request_timeout_seconds": 5,
    }


def test_controller_error_never_falls_back_to_path_npx_cloud_or_cdp(
    tmp_path, monkeypatch
) -> None:
    mapping = _client_mapping(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"controller": mapping, "cdp_url": "ws://secret"}},
    )
    monkeypatch.setenv("BROWSER_CDP_URL", "ws://also-secret")
    monkeypatch.setattr(
        browser_controller_client,
        "maybe_run_browser_controller_command",
        lambda *_args, **_kwargs: {
            "success": False,
            "error": "browser_controller_transport_failed",
        },
    )
    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("PATH fallback")),
    )
    monkeypatch.setattr(
        browser_tool,
        "_get_cloud_provider",
        lambda: (_ for _ in ()).throw(AssertionError("cloud fallback")),
    )

    result = browser_tool._run_browser_command("session", "snapshot", ["-c"])

    assert result == {
        "success": False,
        "error": "browser_controller_transport_failed",
    }
    assert browser_tool._get_cdp_override() == ""


def test_malformed_controller_config_fails_closed_before_local_discovery(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"controller": {"schema": "drifted"}}},
    )
    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("local fallback")),
    )

    result = browser_tool._run_browser_command("session", "open", ["https://example.com"])

    assert result["success"] is False
    assert "browser_controller_client_config" in result["error"]
    assert browser_tool.check_browser_requirements() is False


def test_absent_controller_preserves_generic_browser_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {}},
    )
    monkeypatch.setattr(
        browser_controller_client,
        "maybe_run_browser_controller_command",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("generic marker")),
    )

    result = browser_tool._run_browser_command("session", "snapshot", [])

    assert result["success"] is False
    assert result["error"] == "generic marker"


def test_required_controller_config_read_failure_never_restores_generic_path(
    monkeypatch,
) -> None:
    monkeypatch.setattr(browser_controller_client, "_controller_required", False)
    browser_controller_client.activate_browser_controller_required()
    monkeypatch.setattr(
        "hermes_cli.config.effective_config_projection_is_pinned",
        lambda: True,
    )
    monkeypatch.setattr(
        "hermes_cli.config.attest_pinned_effective_config_projection",
        lambda: {"pinned": True},
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: (_ for _ in ()).throw(OSError("drifted")),
    )
    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("local fallback")),
    )

    result = browser_tool._run_browser_command("session", "snapshot", [])

    assert result == {
        "success": False,
        "error": "browser_controller_client_config_unavailable",
    }
    assert browser_controller_client.controller_mode_requested() is True


def test_required_controller_key_removal_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(browser_controller_client, "_controller_required", False)
    browser_controller_client.activate_browser_controller_required()
    monkeypatch.setattr(
        "hermes_cli.config.effective_config_projection_is_pinned",
        lambda: True,
    )
    monkeypatch.setattr(
        "hermes_cli.config.attest_pinned_effective_config_projection",
        lambda: {"pinned": True},
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"browser": {}},
    )
    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("local fallback")),
    )

    result = browser_tool._run_browser_command("session", "snapshot", [])

    assert result == {
        "success": False,
        "error": "browser_controller_client_config_missing",
    }


def test_required_controller_rejects_unpinned_effective_config(monkeypatch) -> None:
    monkeypatch.setattr(browser_controller_client, "_controller_required", False)
    browser_controller_client.activate_browser_controller_required()
    monkeypatch.setattr(
        "hermes_cli.config.effective_config_projection_is_pinned",
        lambda: False,
    )

    result = browser_controller_client.maybe_run_browser_controller_command(
        "session",
        "snapshot",
        [],
    )

    assert result == {
        "success": False,
        "error": "browser_controller_effective_config_not_pinned",
    }


def test_controller_eval_never_reaches_stale_cdp_supervisor(monkeypatch) -> None:
    class ForbiddenSupervisor:
        def evaluate_runtime(self, _expression):
            raise AssertionError("stale CDP supervisor executed")

    monkeypatch.setattr(
        browser_controller_client,
        "controller_mode_requested",
        lambda: True,
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.browser_supervisor",
        SimpleNamespace(SUPERVISOR_REGISTRY={"session": ForbiddenSupervisor()}),
    )
    monkeypatch.setattr(
        browser_tool,
        "_last_session_key",
        lambda _task_id: "session",
    )
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: {
            "success": False,
            "error": "browser_controller_command_forbidden",
        },
    )

    result = browser_tool._browser_eval("document.cookie", "session")

    assert result == (
        '{"success": false, "error": '
        '"browser_controller_command_forbidden"}'
    )


def test_failed_pool_eviction_unlinks_materialized_artifacts(
    tmp_path, monkeypatch
) -> None:
    config = browser_controller_client.BrowserControllerClientConfig.from_mapping(
        _client_mapping(tmp_path)
    )
    identity = "a" * 64
    client = browser_controller_client.BrowserControllerClient(config, identity)
    png = b"\x89PNG\r\n\x1a\n"
    path = client._materialize_artifact(
        {
            "encoding": "base64",
            "media_type": "image/png",
            "sha256": hashlib.sha256(png).hexdigest(),
            "size": len(png),
            "data": base64.b64encode(png).decode("ascii"),
        }
    )
    key = (config, identity)
    with browser_controller_client._pool_lock:
        browser_controller_client._clients.clear()
        browser_controller_client._clients[key] = client
    monkeypatch.setattr(
        browser_controller_client,
        "load_controller_client_config",
        lambda: config,
    )
    monkeypatch.setattr(
        browser_controller_client,
        "_session_identity",
        lambda _task_id: identity,
    )
    monkeypatch.setattr(
        client,
        "command",
        lambda *_args: {
            "success": False,
            "error": "browser_controller_transport_failed",
        },
    )

    result = browser_controller_client.maybe_run_browser_controller_command(
        "task", "snapshot", []
    )

    assert result["error"] == "browser_controller_transport_failed"
    assert not path.exists()
    assert client._closed is True
    with browser_controller_client._pool_lock:
        assert key not in browser_controller_client._clients
