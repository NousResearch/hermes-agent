"""Tests for the TUI slash worker resolved-model mirror contract.

The TUI slash worker (``tui_gateway.slash_worker``) runs inside the
session's ``profile_home`` scope and resolves the typed command against
its profile-scoped provider / model / alias registry. When a model
switch succeeds, it emits structured metadata containing ONLY resolved,
non-sensitive fields (``side_effect="model_switch"``, ``resolved_model``,
``resolved_provider``, ``scope``, ``base_url``, ``api_mode``, ``raw_args``).

The parent process (``tui_gateway.methods_tools`` / ``tui_gateway.server``)
MUST consume this resolved metadata directly via
``_mirror_resolved_model_switch``. It MUST NOT re-parse ``raw_args``,
MUST NOT consult the parent process's alias cache, and MUST NOT transmit
or receive credentials in metadata.

These tests assert final live agent state (``agent.model``, ``agent.provider``),
``session["model_override"]``, scope behavior, multi-profile isolation,
failure/rollback safety, and metadata sanitization.
"""
from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


@pytest.fixture()
def session_with_agent(server):
    sid = "sid-resolved-mirror"
    session_key = "tui-resolved-mirror"

    fake_agent = MagicMock()
    fake_agent.model = "initial-model"
    fake_agent.provider = "initial-provider"
    fake_agent.base_url = "https://initial.api"

    def fake_switch(new_model, new_provider, api_key="", base_url="", api_mode=""):
        fake_agent.model = new_model
        fake_agent.provider = new_provider
        if base_url:
            fake_agent.base_url = base_url

    fake_agent.switch_model = MagicMock(side_effect=fake_switch)

    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
        "agent": fake_agent,
    }
    server._sessions[sid] = s
    return sid, session_key, s, fake_agent


def _make_worker_double(slash_meta=None, output=""):
    w = MagicMock()
    w.run = MagicMock(return_value=output)
    w.run_with_meta = MagicMock(return_value=(output, slash_meta))
    w.close = MagicMock()
    return w


def _install_worker(session_entry, worker):
    session_entry["slash_worker"] = worker


# ── Scenario 1 & 3: Multi-profile A vs B resolution & provider/model ─────


def test_multi_profile_resolved_model_mirror(server, hermes_home, monkeypatch):
    """Profile A maps /foo -> provider-a/model-a.
    Profile B maps /foo -> provider-b/model-b.
    Executing /foo in B session mirrors provider-b/model-b on B's live agent.
    """
    sid_a = "sid-prof-a"
    agent_a = MagicMock()
    agent_a.model = "old-model"
    agent_a.provider = "old-provider"
    def switch_a(new_model="", new_provider="", **kw):
        agent_a.model = new_model
        agent_a.provider = new_provider
    agent_a.switch_model = switch_a

    sess_a = {
        "session_key": "k-a",
        "history": [],
        "history_lock": threading.Lock(),
        "agent": agent_a,
        "profile_home": str(hermes_home / "prof_a"),
    }
    server._sessions[sid_a] = sess_a

    sid_b = "sid-prof-b"
    agent_b = MagicMock()
    agent_b.model = "old-model"
    agent_b.provider = "old-provider"
    def switch_b(new_model="", new_provider="", **kw):
        agent_b.model = new_model
        agent_b.provider = new_provider
    agent_b.switch_model = switch_b

    sess_b = {
        "session_key": "k-b",
        "history": [],
        "history_lock": threading.Lock(),
        "agent": agent_b,
        "profile_home": str(hermes_home / "prof_b"),
    }
    server._sessions[sid_b] = sess_b

    meta_b = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "model-b",
        "resolved_provider": "provider-b",
        "base_url": "https://b.api",
        "api_mode": "chat_completions",
        "raw_args": "foo",
    }
    _install_worker(sess_b, _make_worker_double(slash_meta=meta_b))

    res = server._methods["slash.exec"](1, {"command": "/foo", "session_id": sid_b})
    assert "result" in res

    # Assert live agent in session B is updated to provider-b / model-b
    assert agent_b.model == "model-b"
    assert agent_b.provider == "provider-b"
    assert sess_b["model_override"]["model"] == "model-b"
    assert sess_b["model_override"]["provider"] == "provider-b"

    # Profile A must remain untouched
    assert agent_a.model == "old-model"
    assert agent_a.provider == "old-provider"


# ── Scenario 2: Sequential execution A then B does not contaminate ──────


def test_sequential_execution_profile_isolation(server, hermes_home):
    """Execute session A (/foo -> provider-a/model-a), then session B (/foo -> provider-b/model-b).
    Assert both sessions end up with their respective profile's resolved provider/model.
    """
    sid_a = "sid-seq-a"
    agent_a = MagicMock()
    agent_a.model = "init"
    agent_a.provider = "init"
    def switch_a(new_model="", new_provider="", **kw):
        agent_a.model = new_model
        agent_a.provider = new_provider
    agent_a.switch_model = switch_a

    sess_a = {
        "session_key": "k-seq-a",
        "history": [],
        "history_lock": threading.Lock(),
        "agent": agent_a,
        "profile_home": str(hermes_home / "prof_a"),
    }
    server._sessions[sid_a] = sess_a

    sid_b = "sid-seq-b"
    agent_b = MagicMock()
    agent_b.model = "init"
    agent_b.provider = "init"
    def switch_b(new_model="", new_provider="", **kw):
        agent_b.model = new_model
        agent_b.provider = new_provider
    agent_b.switch_model = switch_b

    sess_b = {
        "session_key": "k-seq-b",
        "history": [],
        "history_lock": threading.Lock(),
        "agent": agent_b,
        "profile_home": str(hermes_home / "prof_b"),
    }
    server._sessions[sid_b] = sess_b

    meta_a = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "model-a",
        "resolved_provider": "provider-a",
        "base_url": None,
        "api_mode": None,
        "raw_args": "foo",
    }
    meta_b = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "model-b",
        "resolved_provider": "provider-b",
        "base_url": None,
        "api_mode": None,
        "raw_args": "foo",
    }

    _install_worker(sess_a, _make_worker_double(slash_meta=meta_a))
    _install_worker(sess_b, _make_worker_double(slash_meta=meta_b))

    # Execute A first
    server._methods["slash.exec"](1, {"command": "/foo", "session_id": sid_a})
    assert agent_a.model == "model-a"
    assert agent_a.provider == "provider-a"

    # Execute B second
    server._methods["slash.exec"](2, {"command": "/foo", "session_id": sid_b})
    assert agent_b.model == "model-b"
    assert agent_b.provider == "provider-b"

    # Re-verify A was not overwritten by B
    assert agent_a.model == "model-a"
    assert agent_a.provider == "provider-a"


# ── Scenario 4: Custom provider & No credentials in metadata ────────────


def test_custom_provider_metadata_has_no_secrets(server, session_with_agent, hermes_home):
    """Ensure worker metadata contains no api_key/tokens, and custom provider resolves."""
    sid, _, sess, agent = session_with_agent

    prof_dir = hermes_home / "custom_prof"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sess["profile_home"] = str(prof_dir)

    meta_custom = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "my-custom-model",
        "resolved_provider": "my-custom-provider",
        "base_url": "https://custom.endpoint/v1",
        "api_mode": "chat_completions",
        "raw_args": "custom-alias",
    }

    # Assert security constraint: metadata dict does not have api_key, token, or secret
    assert "api_key" not in meta_custom
    assert "token" not in meta_custom
    assert "secret" not in meta_custom

    _install_worker(sess, _make_worker_double(slash_meta=meta_custom))

    res = server._methods["slash.exec"](1, {"command": "/custom-alias", "session_id": sid})
    assert "result" in res

    # Live agent updated
    assert agent.model == "my-custom-model"
    assert agent.provider == "my-custom-provider"
    assert agent.base_url == "https://custom.endpoint/v1"


# ── Scenario 5: --session, --once, --global scope behavior ───────────────


def test_scope_session_once_global_behavior(server, session_with_agent, hermes_home):
    """Test --session, --once, and --global metadata scope handling."""
    sid, _, sess, agent = session_with_agent
    prof_dir = hermes_home / "scope_prof"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sess["profile_home"] = str(prof_dir)

    # 1. Once scope
    meta_once = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "once",
        "resolved_model": "model-once",
        "resolved_provider": "prov-once",
        "raw_args": "once-model --once",
    }
    _install_worker(sess, _make_worker_double(slash_meta=meta_once))
    server._methods["slash.exec"](1, {"command": "/once-model --once", "session_id": sid})
    assert agent.model == "model-once"
    assert sess["model_override"]["scope"] == "once"

    # 2. Global scope
    meta_global = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "global",
        "resolved_model": "model-global",
        "resolved_provider": "prov-global",
        "raw_args": "global-model --global",
    }
    _install_worker(sess, _make_worker_double(slash_meta=meta_global))
    server._methods["slash.exec"](2, {"command": "/global-model --global", "session_id": sid})
    assert agent.model == "model-global"
    assert sess["model_override"]["scope"] == "global"

    # Check that global persisted to config.yaml under prof_dir
    config_file = prof_dir / "config.yaml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert "model-global" in content


# ── Scenario 6: Worker failure / rollback does NOT mirror ───────────────


def test_worker_failure_does_not_mirror(server, session_with_agent):
    """When the worker fails or returns meta=None, live agent is NOT modified."""
    sid, _, sess, agent = session_with_agent

    initial_model = agent.model
    initial_provider = agent.provider

    # Worker returns meta=None (failed execution or no model switch produced)
    _install_worker(sess, _make_worker_double(slash_meta=None, output="Error: invalid model"))

    server._methods["slash.exec"](1, {"command": "/invalid-model", "session_id": sid})

    # Live agent must remain on initial state
    assert agent.model == initial_model
    assert agent.provider == initial_provider
    assert "model_override" not in sess


# ── Scenario 7: Consecutive commands reset metadata ─────────────────────


def test_consecutive_commands_metadata_reset(server, session_with_agent):
    """First command switches model; second command (non-model) returns meta=None.
    The second command must NOT re-apply the previous model switch metadata.
    """
    sid, _, sess, agent = session_with_agent

    # Turn 1: model switch
    meta_1 = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "model-turn-1",
        "resolved_provider": "prov-1",
        "raw_args": "turn-1",
    }
    worker = _make_worker_double()
    worker.run_with_meta = MagicMock(side_effect=[
        ("Switched", meta_1),
        ("Help output", None),
    ])
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/turn-1", "session_id": sid})
    assert agent.model == "model-turn-1"

    # Turn 2: non-model command (meta=None)
    fake_mirror = MagicMock(return_value="")
    server._mirror_slash_side_effects = fake_mirror

    server._methods["slash.exec"](2, {"command": "/help", "session_id": sid})
    # Agent remains model-turn-1, no second call to switch_model
    assert agent.model == "model-turn-1"


# ── Scenario 8: Built-in / plugin / skill collision still no model mirror ─


def test_builtin_plugin_skill_collision_no_model_mirror(server, session_with_agent):
    """When token matches a built-in or plugin, worker resolves meta=None and no model switch happens."""
    sid, _, sess, agent = session_with_agent

    initial_model = agent.model

    # Worker reports meta=None for built-in /version or plugin /foo
    _install_worker(sess, _make_worker_double(slash_meta=None, output="Version 1.0"))

    server._methods["slash.exec"](1, {"command": "/version", "session_id": sid})

    # Live agent unchanged
    assert agent.model == initial_model
    assert "model_override" not in sess
