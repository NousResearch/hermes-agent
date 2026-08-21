"""Profile-scoped config writes for the desktop/TUI gateway."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import tui_gateway.server as server


def _reset_cfg_cache() -> None:
    server._cfg_cache = None
    server._cfg_mtime = None
    server._cfg_path = None


@pytest.fixture
def profile_config_home(tmp_path, monkeypatch):
    launch = tmp_path
    worker = launch / "profiles" / "worker"
    worker.mkdir(parents=True)
    workdir = launch / "worker-cwd"
    workdir.mkdir()

    (launch / "config.yaml").write_text(
        "\n".join(
            [
                "display:",
                "  busy_input_mode: interrupt",
                "  tool_progress: all",
                "terminal:",
                "  cwd: /tmp/default",
                "approvals:",
                "  mode: manual",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (worker / "config.yaml").write_text(
        "\n".join(
            [
                "display:",
                "  busy_input_mode: queue",
                "  tool_progress: new",
                "terminal:",
                "  cwd: /tmp/worker",
                "approvals:",
                "  mode: smart",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(server, "_hermes_home", launch)
    _reset_cfg_cache()
    yield launch, worker, workdir
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    _reset_cfg_cache()


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _set(params: dict) -> dict:
    return server._methods["config.set"]("rid-1", params)


@pytest.mark.parametrize(
    ("key", "value", "section", "field", "expected"),
    [
        ("busy", "steer", "display", "busy_input_mode", "steer"),
        ("verbose", "off", "display", "tool_progress", "off"),
        ("approval_mode", "off", "approvals", "mode", "off"),
    ],
)
def test_config_set_writes_session_profile_config(
    profile_config_home, key, value, section, field, expected
) -> None:
    launch, worker, _workdir = profile_config_home
    session = {
        "agent": None,
        "profile_home": str(worker),
        "session_key": "worker-session",
        "tool_progress_mode": "new",
    }

    with patch.dict(server._sessions, {"s-worker": session}, clear=False):
        resp = _set({"session_id": "s-worker", "key": key, "value": value})

    assert resp["result"]["value"] == expected
    launch_cfg = _read_config(launch / "config.yaml")
    worker_cfg = _read_config(worker / "config.yaml")
    assert worker_cfg[section][field] == expected
    assert launch_cfg[section][field] != expected


def test_config_set_cwd_writes_session_profile_config(
    profile_config_home, monkeypatch
) -> None:
    launch, worker, workdir = profile_config_home
    monkeypatch.setenv("TERMINAL_CWD", "/tmp/default")
    session = {
        "agent": None,
        "profile_home": str(worker),
        "session_key": "worker-session",
    }

    with patch.dict(server._sessions, {"s-worker": session}, clear=False):
        resp = _set(
            {"session_id": "s-worker", "key": "cwd", "value": str(workdir)}
        )

    assert resp["result"]["cwd"] == str(workdir)
    launch_cfg = _read_config(launch / "config.yaml")
    worker_cfg = _read_config(worker / "config.yaml")
    assert worker_cfg["terminal"]["cwd"] == str(workdir)
    assert launch_cfg["terminal"]["cwd"] == "/tmp/default"
    assert server.os.environ["TERMINAL_CWD"] == "/tmp/default"


def test_config_set_without_session_still_writes_launch_config(
    profile_config_home,
) -> None:
    launch, worker, _workdir = profile_config_home

    resp = _set({"key": "busy", "value": "steer"})

    assert resp["result"]["value"] == "steer"
    launch_cfg = _read_config(launch / "config.yaml")
    worker_cfg = _read_config(worker / "config.yaml")
    assert launch_cfg["display"]["busy_input_mode"] == "steer"
    assert worker_cfg["display"]["busy_input_mode"] == "queue"


def test_config_set_without_session_cwd_still_updates_launch_env(
    profile_config_home, monkeypatch
) -> None:
    launch, worker, workdir = profile_config_home
    monkeypatch.setenv("TERMINAL_CWD", "/tmp/default")

    resp = _set({"key": "cwd", "value": str(workdir)})

    assert resp["result"]["cwd"] == str(workdir)
    launch_cfg = _read_config(launch / "config.yaml")
    worker_cfg = _read_config(worker / "config.yaml")
    assert launch_cfg["terminal"]["cwd"] == str(workdir)
    assert worker_cfg["terminal"]["cwd"] == "/tmp/worker"
    assert server.os.environ["TERMINAL_CWD"] == str(workdir)


def test_raw_config_save_honors_profile_home_override(profile_config_home) -> None:
    launch, worker, _workdir = profile_config_home
    token = server.set_hermes_home_override(str(worker))
    try:
        cfg = server._load_cfg_raw()
        cfg.setdefault("display", {})["busy_input_mode"] = "steer"
        server._save_cfg(cfg)
    finally:
        server.reset_hermes_home_override(token)

    launch_cfg = _read_config(launch / "config.yaml")
    worker_cfg = _read_config(worker / "config.yaml")
    assert worker_cfg["display"]["busy_input_mode"] == "steer"
    assert launch_cfg["display"]["busy_input_mode"] == "interrupt"
