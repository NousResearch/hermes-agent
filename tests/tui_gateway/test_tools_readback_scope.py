"""tools.list / tools.show must read back the session's effective toolsets (issue #117977).

``profiles.configure`` persists an ``enabled_toolsets`` pin in the profile's own home,
but the live agent of an already-built session keeps its boot-time set, and a
session-less call resolved against the launch home. Both read-back RPCs therefore
reported the gateway-global catalog even for a narrowly configured session.

The fix resolves through the session's home scope, so an agent-less session (pin
saved by the profile editor) and a foreign-profile session served by one backend
both answer with the toolsets that session actually runs with.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

import tui_gateway.server as server
import tui_gateway.methods_tools as methods_tools


LAUNCH_HOME_CFG = {"platform_toolsets": {"cli": ["web", "browser", "terminal"]}}
WORKER_HOME_CFG = {"platform_toolsets": {"cli": ["file", "clarify"]}}


def _write_cfg(home: Path, cfg: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _rows(session) -> list[dict]:
    return server._session_scoped_toolsets(session)


def test_built_agent_answer_wins_over_config(monkeypatch, tmp_path):
    agent = SimpleNamespace(enabled_toolsets=["skills"])
    session = {"agent": agent, "profile_home": str(tmp_path)}
    assert _rows(session) == ["skills"]


def test_agent_less_session_reads_its_own_home_pin(monkeypatch, tmp_path):
    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "code"
    _write_cfg(launch, LAUNCH_HOME_CFG)
    _write_cfg(worker, WORKER_HOME_CFG)

    monkeypatch.setattr(server, "_load_enabled_toolsets", _real_loader(launch, worker))
    session = {"agent": None, "profile_home": str(worker), "cwd": "", "source": ""}
    resolved = _rows(session)
    assert resolved is not None
    assert set(WORKER_HOME_CFG["platform_toolsets"]["cli"]) <= set(resolved)
    assert "browser" not in resolved


def test_session_without_profile_home_falls_back_to_launch(monkeypatch, tmp_path):
    launch = tmp_path / "launch"
    _write_cfg(launch, LAUNCH_HOME_CFG)
    monkeypatch.setattr(server, "_load_enabled_toolsets", _real_loader(launch, None))
    session = {"agent": None, "profile_home": None, "cwd": "", "source": ""}
    resolved = _rows(session)
    assert resolved is not None
    assert "browser" in resolved


def test_session_none_uses_launch_resolver(monkeypatch, tmp_path):
    launch = tmp_path / "launch"
    _write_cfg(launch, LAUNCH_HOME_CFG)
    calls = []

    def fake_loader(platform=None):
        calls.append(platform)
        return ["web"]

    monkeypatch.setattr(server, "_load_enabled_toolsets", fake_loader)
    assert _rows(None) == ["web"]
    assert calls == [None]


def _real_loader(launch: Path, worker: Path | None):
    def load(platform=None):
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        token = set_hermes_home_override(str(worker if worker else launch))
        try:
            from hermes_cli.config import load_config
            from hermes_cli.tools_config import _get_platform_tools
            cfg = load_config() or {}
            return sorted(_get_platform_tools(cfg, "cli", include_default_mcp_servers=False)) or None
        finally:
            reset_hermes_home_override(token)

    return load
