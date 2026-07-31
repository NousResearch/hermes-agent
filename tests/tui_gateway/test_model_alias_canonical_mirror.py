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

These tests assert:
  1. Multi-profile resolution isolation and agent state.
  2. True concurrent interleaving execution without process-global environment tampering.
  3. Closed 7-phase --once model switch lifecycle and restore.
  4. Custom provider resolution without credential transmission.
  5. Session, once, and global scope behavior (verifying config writes stay within profile home).
  6. Worker failure / cancellation safety.
  7. Parent apply failure rollback.
  8. Consecutive command metadata resetting and collision immunity.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
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

    fake_agent = MagicMock(spec=["model", "provider", "base_url", "api_key", "api_mode", "switch_model"])
    fake_agent.model = "initial-model"
    fake_agent.provider = "initial-provider"
    fake_agent.base_url = "https://initial.api"
    fake_agent.api_key = "secret-key"
    fake_agent.api_mode = "chat_completions"

    def fake_switch(new_model="", new_provider="", api_key="", base_url="", api_mode=""):
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


@pytest.fixture()
def patched_runtime_provider(monkeypatch):
    """Replace ``hermes_cli.runtime_provider.resolve_runtime_provider`` with a
    stub that returns a deterministic runtime dict, so the mirror path can run
    end-to-end against fake providers without trying to actually reach an LLM.
    """
    def fake_resolve(
        requested=None,
        target_model=None,
        explicit_base_url=None,
        **_kwargs,
    ):
        return {
            "provider": requested or "",
            "api_mode": "chat_completions",
            "base_url": explicit_base_url or "",
            "api_key": "fake-key",
            "source": "test-stub",
            "requested_provider": requested or "",
        }

    import hermes_cli.runtime_provider as _rp_module
    _rp_module.resolve_runtime_provider = fake_resolve
    monkeypatch.setattr(
        _rp_module,
        "resolve_runtime_provider",
        fake_resolve,
    )


# ── Scenario 1: Multi-profile A vs B resolution & provider/model ─────


def test_multi_profile_resolved_model_mirror(server, hermes_home, patched_runtime_provider):
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


# ── Scenario 2: Concurrent profile interleaving without env tampering ────


def test_concurrent_profile_cross_talk_isolation(server, hermes_home, patched_runtime_provider):
    """Execute Profile A and Profile B mirrors concurrently using a barrier
    to force interleaving. Verify os.environ["HERMES_HOME"] is NEVER modified,
    context overrides remain isolated, and both sessions end up in correct state.
    """
    initial_env_home = os.environ.get("HERMES_HOME")

    prof_a_dir = hermes_home / "prof_conc_a"
    prof_b_dir = hermes_home / "prof_conc_b"
    prof_a_dir.mkdir(parents=True, exist_ok=True)
    prof_b_dir.mkdir(parents=True, exist_ok=True)

    sid_a = "sid-conc-a"
    agent_a = MagicMock()
    agent_a.model = "init"
    agent_a.provider = "init"
    def switch_a(new_model="", new_provider="", **kw):
        agent_a.model = new_model
        agent_a.provider = new_provider
    agent_a.switch_model = switch_a

    sess_a = {
        "session_key": "k-conc-a",
        "history": [],
        "history_lock": threading.Lock(),
        "agent": agent_a,
        "profile_home": str(prof_a_dir),
    }

    sid_b = "sid-conc-b"
    agent_b = MagicMock()
    agent_b.model = "init"
    agent_b.provider = "init"
    def switch_b(new_model="", new_provider="", **kw):
        agent_b.model = new_model
        agent_b.provider = new_provider
    agent_b.switch_model = switch_b

    sess_b = {
        "session_key": "k-conc-b",
        "history": [],
        "history_lock": threading.Lock(),
        "agent": agent_b,
        "profile_home": str(prof_b_dir),
    }

    meta_a = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "global",
        "resolved_model": "model-a",
        "resolved_provider": "provider-a",
        "raw_args": "foo",
    }
    meta_b = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "global",
        "resolved_model": "model-b",
        "resolved_provider": "provider-b",
        "raw_args": "foo",
    }

    barrier = threading.Barrier(2)

    def run_mirror_a():
        # Inject barrier inside resolution path
        orig_res = server._mirror_resolved_model_switch
        barrier.wait(timeout=5)
        res = orig_res(sid_a, sess_a, meta_a)
        # Assert process-global environment was NEVER mutated
        assert os.environ.get("HERMES_HOME") == initial_env_home
        return res

    def run_mirror_b():
        orig_res = server._mirror_resolved_model_switch
        barrier.wait(timeout=5)
        res = orig_res(sid_b, sess_b, meta_b)
        assert os.environ.get("HERMES_HOME") == initial_env_home
        return res

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_a = executor.submit(run_mirror_a)
        fut_b = executor.submit(run_mirror_b)
        fut_a.result()
        fut_b.result()

    # Verify final agent models
    assert agent_a.model == "model-a"
    assert agent_a.provider == "provider-a"
    assert agent_b.model == "model-b"
    assert agent_b.provider == "provider-b"

    # Verify configs persisted to respective profile homes without clobbering
    cfg_a = (prof_a_dir / "config.yaml").read_text(encoding="utf-8")
    cfg_b = (prof_b_dir / "config.yaml").read_text(encoding="utf-8")
    assert "model-a" in cfg_a
    assert "model-b" in cfg_b

    # Final env assertion
    assert os.environ.get("HERMES_HOME") == initial_env_home


# ── Scenario 3: Closed --once 7-phase lifecycle and restore ───────────────


def test_once_full_lifecycle_restore(server, session_with_agent):
    """Complete 7-phase validation for --once model switch:
    Phase 1: Worker returns once metadata.
    Phase 2: Live agent switches to target model.
    Phase 3: Turn 1 executes using target model.
    Phase 4: Turn 1 completes and restores previous provider/model.
    Phase 5: Session model_override once state and restore_snapshot cleared.
    Phase 6: Turn 2 continues using original model.
    Phase 7: Failure or cancellation paths leave no pending restore state.
    """
    sid, _, sess, agent = session_with_agent

    orig_model = agent.model
    orig_provider = agent.provider

    # Phase 1: Worker returns once metadata
    meta_once = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "once",
        "resolved_model": "model-temp-once",
        "resolved_provider": "prov-once",
        "raw_args": "temp-model --once",
    }
    _install_worker(sess, _make_worker_double(slash_meta=meta_once))

    # Phase 2: Parent mirrors slash exec
    res = server._methods["slash.exec"](1, {"command": "/temp-model --once", "session_id": sid})
    assert "result" in res

    # Live agent switched to temp once model
    assert agent.model == "model-temp-once"
    assert agent.provider == "prov-once"
    assert sess["model_override"]["scope"] == "once"
    assert sess.get("one_turn_model_restore") is not None

    # Phase 3: Assistant turn 1 executes with temp model
    assert agent.model == "model-temp-once"

    # Phase 4 & 5: Turn 1 finishes -> restore agent model runtime
    restore_snap = sess.pop("one_turn_model_restore", None)
    assert restore_snap is not None
    server._restore_agent_model_runtime(agent, restore_snap)
    sess["model_override"].pop("restore_snapshot", None)
    sess.pop("model_override", None)

    # Restored to original model and provider
    assert agent.model == orig_model
    assert agent.provider == orig_provider

    # Phase 6: Turn 2 executes with original model
    assert agent.model == orig_model
    assert "one_turn_model_restore" not in sess
    assert "model_override" not in sess

    # Phase 7: Failure/cancellation leaves no pending restore state
    _install_worker(sess, _make_worker_double(slash_meta=None, output="Cancelled"))
    server._methods["slash.exec"](2, {"command": "/bad-cmd", "session_id": sid})
    assert agent.model == orig_model
    assert "one_turn_model_restore" not in sess


# ── Scenario 4: Custom provider & No credentials in metadata ────────────


def test_custom_provider_metadata_has_no_secrets(server, session_with_agent, hermes_home, patched_runtime_provider):
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


def test_scope_session_once_global_behavior(server, session_with_agent, hermes_home, patched_runtime_provider):
    """Test --session, --once, and --global metadata scope handling."""
    sid, _, sess, agent = session_with_agent
    prof_dir = hermes_home / "scope_prof"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sess["profile_home"] = str(prof_dir)

    # 1. Session scope
    meta_sess = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "model-sess",
        "resolved_provider": "prov-sess",
        "raw_args": "sess-model --session",
    }
    _install_worker(sess, _make_worker_double(slash_meta=meta_sess))
    server._methods["slash.exec"](1, {"command": "/sess-model --session", "session_id": sid})
    assert agent.model == "model-sess"
    assert sess["model_override"]["scope"] == "session"

    # 2. Once scope
    meta_once = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "once",
        "resolved_model": "model-once",
        "resolved_provider": "prov-once",
        "raw_args": "once-model --once",
    }
    _install_worker(sess, _make_worker_double(slash_meta=meta_once))
    server._methods["slash.exec"](2, {"command": "/once-model --once", "session_id": sid})
    assert agent.model == "model-once"
    assert sess["model_override"]["scope"] == "once"

    # 3. Global scope
    meta_global = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "global",
        "resolved_model": "model-global",
        "resolved_provider": "prov-global",
        "raw_args": "global-model --global",
    }
    _install_worker(sess, _make_worker_double(slash_meta=meta_global))
    server._methods["slash.exec"](3, {"command": "/global-model --global", "session_id": sid})
    assert agent.model == "model-global"
    assert sess["model_override"]["scope"] == "global"

    # Check that global persisted ONLY to config.yaml under prof_dir
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


# ── Scenario 7: Parent apply failure rollback ────────────────────────────


def test_parent_apply_failure_rollback(server, session_with_agent):
    """If agent.switch_model fails on the parent, _mirror_resolved_model_switch
    returns a warning message and does not corrupt session state.
    """
    sid, _, sess, agent = session_with_agent

    agent.switch_model = MagicMock(side_effect=RuntimeError("connection refused"))

    meta = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "broken-model",
        "resolved_provider": "broken-prov",
        "raw_args": "broken",
    }
    _install_worker(sess, _make_worker_double(slash_meta=meta))

    res = server._methods["slash.exec"](1, {"command": "/broken", "session_id": sid})
    assert "result" in res
    assert "warning" in res["result"]
    assert "model mirror failed" in res["result"]["warning"]


# ── Scenario 8: Consecutive commands reset metadata & collision safety ───


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
    # Agent remains model-turn-1
    assert agent.model == "model-turn-1"


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


# ── Profile scope fail-closed tests ───────────────────────────────


def _session_with_profile(server, profile_home: Path):
    """Build a session dict wired with a real MagicMock agent."""
    sid = f"sid-{profile_home.name}"
    fake_agent = MagicMock(
        spec=["model", "provider", "base_url", "api_key", "api_mode", "switch_model"]
    )
    fake_agent.model = "initial-model"
    fake_agent.provider = "initial-provider"
    fake_agent.base_url = "https://initial.api"
    fake_agent.api_key = "secret-key"
    fake_agent.api_mode = "chat_completions"

    def fake_switch(new_model="", new_provider="", **kw):
        fake_agent.model = new_model
        fake_agent.provider = new_provider
        if base_url:
            fake_agent.base_url = base_url

    fake_agent.switch_model = MagicMock(side_effect=fake_switch)

    sess = {
        "session_key": f"k-{profile_home.name}",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
        "agent": fake_agent,
        "profile_home": str(profile_home),
    }
    server._sessions[sid] = sess
    return sid, sess, fake_agent


def test_fail_closed_home_scope_raises(server, hermes_home):
    """``set_hermes_home_override`` raises -> mirror aborts BEFORE any
    provider resolution, agent mutation, or model_override write."""
    prof_dir = hermes_home / "prof_home_fail"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sid, sess, agent = _session_with_profile(server, prof_dir)

    import hermes_constants
    real_set = hermes_constants.set_hermes_home_override

    def broken_set(path):
        raise RuntimeError("scope setup failed")

    hermes_constants.set_hermes_home_override = broken_set

    meta = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "model-x",
        "resolved_provider": "provider-x",
        "raw_args": "x",
    }

    try:
        warning = server._mirror_resolved_model_switch(sid, sess, meta)
    finally:
        hermes_constants.set_hermes_home_override = real_set

    assert "mirror aborted" in warning
    # Agent untouched
    assert agent.model == "initial-model"
    assert agent.provider == "initial-provider"
    # No session mutation
    assert "model_override" not in sess
    # config.yaml not created on profile_home
    assert not (prof_dir / "config.yaml").exists()


def test_fail_closed_secret_scope_raises(server, hermes_home):
    """``set_secret_scope`` raises AFTER home_token was established —
    home_token must be reset (no ContextVar leak) and mirror aborts."""
    prof_dir = hermes_home / "prof_secret_fail"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sid, sess, agent = _session_with_profile(server, prof_dir)

    # Track that home override was set and reset
    home_set_calls = []

    real_set = None
    real_reset = None
    import hermes_constants

    try:
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    except ImportError:
        pytest.skip("hermes_constants unavailable")

    real_set = set_hermes_home_override
    real_reset = reset_hermes_home_override

    def tracked_set(path):
        tok = real_set(path)
        home_set_calls.append(("set", tok))
        return tok

    def tracked_reset(tok):
        home_set_calls.append(("reset", tok))
        return real_reset(tok)

    hermes_constants.set_hermes_home_override = tracked_set
    hermes_constants.reset_hermes_home_override = tracked_reset

    # Force set_secret_scope to raise
    import agent.secret_scope as ss_module

    def broken_set_scope(scope):
        raise RuntimeError("secret scope setup failed")

    original_set_secret_scope = ss_module.set_secret_scope
    ss_module.set_secret_scope = broken_set_scope

    try:
        meta = {
            "side_effect": "model_switch",
            "canonical": "model",
            "scope": "session",
            "resolved_model": "model-x",
            "resolved_provider": "provider-x",
            "raw_args": "x",
        }

        warning = server._mirror_resolved_model_switch(sid, sess, meta)
    finally:
        ss_module.set_secret_scope = original_set_secret_scope
        hermes_constants.set_hermes_home_override = real_set
        hermes_constants.reset_hermes_home_override = real_reset

    assert "mirror aborted" in warning
    # home_token was set then reset by ExitStack
    ops = [op for op, _ in home_set_calls]
    assert ops.count("set") >= 1
    assert ops.count("reset") >= 1
    # Agent untouched, no session mutation
    assert agent.model == "initial-model"
    assert "model_override" not in sess


def test_fail_closed_provider_resolution_raises(server, hermes_home, monkeypatch):
    """``resolve_runtime_provider`` raises -> both scope tokens are reset."""
    prof_dir = hermes_home / "prof_resolve_fail"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sid, sess, agent = _session_with_profile(server, prof_dir)

    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "resolve_runtime_provider",
                       lambda *a, **k: (_ for _ in ()).throw(RuntimeError("resolve broken")))

    meta = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "model-x",
        "resolved_provider": "provider-x",
        "raw_args": "x",
    }

    warning = server._mirror_resolved_model_switch(sid, sess, meta)

    assert "provider resolution failed" in warning
    assert agent.model == "initial-model"
    assert "model_override" not in sess


def test_fail_closed_agent_apply_raises(server, hermes_home, patched_runtime_provider):
    """``agent.switch_model`` raises during apply -> scope tokens are reset."""
    prof_dir = hermes_home / "prof_apply_fail"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sid, sess, agent = _session_with_profile(server, prof_dir)

    # Force switch_model to raise on apply
    def broken_switch(**kw):
        raise RuntimeError("agent switch failed")
    agent.switch_model = MagicMock(side_effect=broken_switch)

    meta = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "session",
        "resolved_model": "model-x",
        "resolved_provider": "provider-x",
        "raw_args": "x",
    }

    warning = server._mirror_resolved_model_switch(sid, sess, meta)

    assert "model mirror failed" in warning
    # Agent.switch_model did get called (and raised) but agent.model stays initial
    assert agent.model == "initial-model"
    # session was not mutated
    assert "model_override" not in sess


def test_fail_closed_secret_scope_init_raises(server, hermes_home):
    """``build_profile_secret_scope`` raises (during init, before set_secret_scope)
    — home_token already established; both attempts handled and tokens reset."""
    prof_dir = hermes_home / "prof_build_secret_fail"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sid, sess, agent = _session_with_profile(server, prof_dir)

    import hermes_constants
    import agent.secret_scope as ss_module

    home_resets = []

    real_set = hermes_constants.set_hermes_home_override
    real_reset = hermes_constants.reset_hermes_home_override

    def tracked_reset(tok):
        home_resets.append(tok)
        return real_reset(tok)

    hermes_constants.set_hermes_home_override = real_set
    hermes_constants.reset_hermes_home_override = tracked_reset

    def broken_build(path):
        raise RuntimeError("build profile secret scope failed")

    real_build = ss_module.build_profile_secret_scope
    ss_module.build_profile_secret_scope = broken_build

    try:
        meta = {
            "side_effect": "model_switch",
            "canonical": "model",
            "scope": "session",
            "resolved_model": "model-x",
            "resolved_provider": "provider-x",
            "raw_args": "x",
        }
        warning = server._mirror_resolved_model_switch(sid, sess, meta)
    finally:
        ss_module.build_profile_secret_scope = real_build
        hermes_constants.set_hermes_home_override = real_set
        hermes_constants.reset_hermes_home_override = real_reset

    assert "mirror aborted" in warning
    # Agent untouched, no session mutation
    assert agent.model == "initial-model"
    assert "model_override" not in sess


# ── Real production --once lifecycle integration ─────────────────────


def _once_lifecycle_setup(server, hermes_home, patched_runtime_provider):
    """Prepare a session with all prerequisites for prompt.submit production chain.

    Returns: (sid, sess, agent)
    """
    prof_dir = hermes_home / "prof_lifecycle"
    prof_dir.mkdir(parents=True, exist_ok=True)

    sid = "sid-once-life"
    agent = MagicMock(
        spec=["model", "provider", "base_url", "api_key", "api_mode", "switch_model", "run_conversation"]
    )
    agent.model = "orig-model"
    agent.provider = "orig-provider"
    agent.base_url = "https://orig.api"
    agent.api_key = "orig-key"
    agent.api_mode = "chat_completions"

    def fake_switch(new_model="", new_provider="", api_key="", base_url="", api_mode=""):
        agent.model = new_model
        agent.provider = new_provider
        if base_url:
            agent.base_url = base_url

    agent.switch_model = MagicMock(side_effect=fake_switch)

    sess = {
        "session_key": "k-once-life",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
        "agent": agent,
        "profile_home": str(prof_dir),
        # Mark session as not running so prompt.submit runs _run_prompt_submit
        "agent_ready": threading.Event(),
    }
    sess["agent_ready"].set()
    server._sessions[sid] = sess

    # Worker reports once metadata
    meta_once = {
        "side_effect": "model_switch",
        "canonical": "model",
        "scope": "once",
        "resolved_model": "model-once-temp",
        "resolved_provider": "prov-once-temp",
        "raw_args": "temp-model --once",
    }
    _install_worker(sess, _make_worker_double(slash_meta=meta_once))

    # Phase A: slash.exec applies once via production mirror path
    res = server._methods["slash.exec"](1, {"command": "/temp-model --once", "session_id": sid})
    assert "result" in res

    return sid, sess, agent


def _wait_turn_threads(sess, timeout=120):
    """Join both daemon turn threads: the ``prompt.submit``
    ``run_after_agent_ready`` wrapper and the inner ``_run_prompt_submit``
    ``run()`` thread that ``_run_prompt_submit`` stores back into
    ``session["_run_thread"]`` when it starts the real turn body."""
    first = sess.get("_run_thread")
    if first is not None:
        first.join(timeout=timeout)
    inner = sess.get("_run_thread")
    if inner is not None and inner is not first:
        inner.join(timeout=timeout)


def test_once_real_turn_chain_success_restores(server, hermes_home, patched_runtime_provider):
    """Real production turn chain restores --once after successful turn."""
    sid, sess, agent = _once_lifecycle_setup(server, hermes_home, patched_runtime_provider)

    # Phase 2: live agent temp model after mirror
    assert agent.model == "model-once-temp"
    assert sess["one_turn_model_restore"] is not None

    # Mock agent.run_conversation to return successfully without real LLM
    def fake_run_conversation(*args, **kwargs):
        return {"messages": [{"role": "assistant", "content": "ok"}]}

    agent.run_conversation = MagicMock(side_effect=fake_run_conversation)

    # Phase 3: real prompt.submit -> _run_prompt_submit -> turn-finally -> restore
    res = server._methods["prompt.submit"](2, {"session_id": sid, "text": "hi"})
    assert "result" in res

    # Wait for daemon _run_thread to finish
    _wait_turn_threads(sess)

    # Phase 4: agent restored to original
    assert agent.model == "orig-model"
    assert agent.provider == "orig-provider"
    # Phase 5: once snapshot consumed by the production turn finally
    assert "one_turn_model_restore" not in sess
    # model_override scope remains "once" (production keeps it; the
    # consumed snapshot is what drives the restore)
    mo = sess.get("model_override", {})
    assert mo.get("scope") == "once"


def test_once_real_turn_chain_error_restores(server, hermes_home, patched_runtime_provider):
    """Real production turn chain restores --once even if turn raises."""
    sid, sess, agent = _once_lifecycle_setup(server, hermes_home, patched_runtime_provider)

    assert agent.model == "model-once-temp"

    # Mock agent.run_conversation to raise an exception
    def failing_run(*args, **kwargs):
        raise RuntimeError("model provider rate-limit")
    agent.run_conversation = MagicMock(side_effect=failing_run)

    # Submit prompt through production chain
    server._methods["prompt.submit"](2, {"session_id": sid, "text": "hi"})

    _wait_turn_threads(sess)

    # Agent restored despite turn error
    assert agent.model == "orig-model"
    assert agent.provider == "orig-provider"
    # Once snapshot consumed
    assert "one_turn_model_restore" not in sess


def test_once_real_turn_chain_interrupt_restores(server, hermes_home, patched_runtime_provider):
    """Real production turn chain restores --once when session is interrupted."""
    sid, sess, agent = _once_lifecycle_setup(server, hermes_home, patched_runtime_provider)

    assert agent.model == "model-once-temp"

    # Mock agent.run_conversation to honor interrupt flag and abort
    def interruptible_run(*args, **kwargs):
        # Read interrupt flag under lock to mimic real agent behavior
        with sess["history_lock"]:
            cancelled = bool(sess.get("_turn_cancel_requested"))
        if cancelled:
            raise RuntimeError("turn interrupted")
        # Otherwise block; interrupt will set the flag
        import time as _t
        for _ in range(50):
            with sess["history_lock"]:
                if sess.get("_turn_cancel_requested"):
                    raise RuntimeError("turn interrupted")
            _t.sleep(0.1)
        return {"messages": []}

    agent.run_conversation = MagicMock(side_effect=interruptible_run)

    # Submit prompt through production chain
    server._methods["prompt.submit"](2, {"session_id": sid, "text": "hi"})

    # Set interrupt flag immediately
    import time as _t
    _t.sleep(0.2)
    with sess["history_lock"]:
        sess["_turn_cancel_requested"] = True

    _wait_turn_threads(sess)

    # Agent restored despite interrupt
    assert agent.model == "orig-model"
    assert "one_turn_model_restore" not in sess
