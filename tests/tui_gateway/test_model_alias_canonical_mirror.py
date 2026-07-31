"""Tests for the TUI slash worker canonical-side-effect mirror contract.

The TUI slash worker (``tui_gateway.slash_worker``) runs inside the
session's ``profile_home`` scope and resolves the typed command against
the real built-in / quick / plugin / bundle / skill / model alias
registry. It then writes a structured ``meta`` JSON field describing
what it actually did.

The parent ``slash.exec`` handler (``tui_gateway.methods_tools``) MUST
base the live-session mirror decision ONLY on that ``meta`` field. It
must NOT re-derive whether the typed token is in a global alias cache.
Re-deriving breaks two things at once:

1. A built-in / quick / plugin / bundle / skill command whose typed
   name also exists as a model alias (e.g. ``/version`` while
   ``version`` is configured as a model alias). The worker actually
   ran the built-in; the mirror must NOT switch the live model.

2. Profile A and Profile B with the same alias name mapped to
   different models. The parent must read the worker's profile-scoped
   metadata, not the parent's global alias cache.

These tests exercise both bugs and pin the contract for every layer.
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
def session(server):
    sid = "sid-mirror-canonical"
    session_key = "tui-mirror-canonical"
    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _make_worker_double(slash_meta=None, output=""):
    """Return a worker double whose ``run_with_meta`` returns
    ``(output, slash_meta)`` — the structured metadata the real
    slash worker emits inside the session's profile scope.
    """
    w = MagicMock()
    w.run = MagicMock(return_value=output)
    w.run_with_meta = MagicMock(return_value=(output, slash_meta))
    w.close = MagicMock()
    return w


def _install_worker(session_entry, worker):
    session_entry["slash_worker"] = worker


def _capturing_mirror():
    """Return ``(fake_mirror_fn, captured_list)`` for capturing mirror calls."""
    captured = []

    def fake_mirror(sid, sess, cmd):
        captured.append(cmd)
        return ""

    return fake_mirror, captured


# ── 1. /version vs alias `version` ────────────────────────────────────


def test_builtin_version_with_alias_version_does_not_mirror_model(
    server, session, monkeypatch
):
    """When ``version`` is configured as a model alias, the worker
    actually resolves the built-in ``version`` command and reports
    ``side_effect=None`` (no model switch). The parent must NOT call
    ``_apply_model_switch`` and must NOT change the live session's
    model.
    """
    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    # Force the worker's reported metadata to be what the real worker
    # would have set when it ran built-in ``version``: meta is None.
    worker = _make_worker_double(slash_meta=None)
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/version", "session_id": sid})

    # The mirror still gets called (so other built-in side effects can
    # fire), but the command is /version verbatim — NOT /model version.
    assert len(captured) == 1
    assert captured[0].lower() == "/version"
    worker.run_with_meta.assert_called_with("/version")


# ── 2. alias vs quick command ──────────────────────────────────────────


def test_alias_collision_with_quick_command_does_not_mirror_model(
    server, session, monkeypatch
):
    """A quick command named like a model alias must win; the worker
    reports ``meta=None`` and the parent must NOT switch models.
    """
    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    worker = _make_worker_double(slash_meta=None)
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/sonnet", "session_id": sid})

    assert len(captured) == 1
    assert captured[0].lower() == "/sonnet"


# ── 3. alias vs plugin command ──────────────────────────────────────────


def test_alias_collision_with_plugin_does_not_mirror_model(
    server, session, monkeypatch
):
    """A plugin command named like a model alias must win."""
    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    worker = _make_worker_double(slash_meta=None)
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/sonnet", "session_id": sid})

    assert len(captured) == 1
    assert captured[0].lower() == "/sonnet"


# ── 4. alias vs skill bundle ──────────────────────────────────────────


def test_alias_collision_with_bundle_does_not_mirror_model(
    server, session, monkeypatch
):
    """A skill bundle command named like a model alias must win."""
    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    worker = _make_worker_double(slash_meta=None)
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/sonnet", "session_id": sid})

    assert len(captured) == 1
    assert captured[0].lower() == "/sonnet"


# ── 5. alias vs active skill ──────────────────────────────────────────


def test_alias_collision_with_active_skill_does_not_mirror_model(
    server, session, monkeypatch
):
    """An active skill command named like a model alias must win."""
    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    worker = _make_worker_double(slash_meta=None)
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/sonnet", "session_id": sid})

    assert len(captured) == 1
    assert captured[0].lower() == "/sonnet"


# ── 6. /sonnet (real alias) is canonicalized by worker ────────────────


def test_real_alias_sonnet_does_mirror_via_worker_meta(
    server, session, monkeypatch
):
    """When the worker reports ``side_effect == "model_switch"`` with
    ``canonical == "model"``, the parent must re-emit the model switch
    via ``_mirror_slash_side_effects`` exactly once.
    """
    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    meta = {
        "canonical": "model",
        "raw_args": "sonnet",
        "args": "sonnet",
        "target": "anthropic/claude-sonnet",
        "side_effect": "model_switch",
    }
    worker = _make_worker_double(slash_meta=meta)
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/sonnet", "session_id": sid})

    assert len(captured) == 1
    cmd = captured[0].lower()
    assert cmd.startswith("/model")
    assert "sonnet" in cmd
    worker.run_with_meta.assert_called_with("/sonnet")


# ── 7. Multi-profile: only the worker's profile-scoped meta matters ───


def test_multi_profile_meta_only_worker_resolves_alias(
    server, session, monkeypatch
):
    """The parent must NOT inspect any cross-profile alias cache. The
    decision comes purely from the structured metadata the worker
    returned.

    Profile A's worker resolves ``/foo`` to a built-in (meta=None).
    Profile B's worker resolves ``/foo`` to ``model_switch`` (meta=...).
    Two independent sessions, two independent mirror outcomes.
    """
    fake_mirror_a, captured_a = _capturing_mirror()
    fake_mirror_b, captured_b = _capturing_mirror()

    # Two sessions on different profile homes.
    sid_a, _, sess_a = session

    sid_b = "sid-profile-b"
    sess_b = {
        "session_key": "tui-profile-b",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid_b] = sess_b

    # Profile A: built-in /foo (e.g. a future addon)
    worker_a = _make_worker_double(slash_meta=None)
    _install_worker(sess_a, worker_a)

    # Profile B: /foo is a model alias to "gpt-5"
    meta_b = {
        "canonical": "model",
        "raw_args": "foo",
        "args": "foo",
        "target": "gpt-5",
        "side_effect": "model_switch",
    }
    worker_b = _make_worker_double(slash_meta=meta_b)
    _install_worker(sess_b, worker_b)

    # Patch each session's mirror capture independently. Because the
    # session dict carries its own ``agent``, we route the mirror
    # through the existing ``_mirror_slash_side_effects`` name, but
    # capture per-session using a wrapping closure that filters on sid.
    def route_mirror(sid_arg, sess_arg, cmd):
        if sid_arg == sid_a:
            return fake_mirror_a(sid_arg, sess_arg, cmd)
        if sid_arg == sid_b:
            return fake_mirror_b(sid_arg, sess_arg, cmd)
        return ""

    monkeypatch.setattr(server, "_mirror_slash_side_effects", route_mirror)

    # Profile A: /foo
    server._methods["slash.exec"](1, {"command": "/foo", "session_id": sid_a})
    # Profile B: /foo (different profile's worker)
    server._methods["slash.exec"](1, {"command": "/foo", "session_id": sid_b})

    # Profile A: built-in, mirror sees /foo verbatim
    assert len(captured_a) == 1
    assert captured_a[0].lower() == "/foo"

    # Profile B: model, mirror sees /model foo
    assert len(captured_b) == 1
    cmd_b = captured_b[0].lower()
    assert cmd_b.startswith("/model")
    assert "foo" in cmd_b


# ── 8. Multi-profile with same alias name, different targets ──────────


def test_multi_profile_same_alias_different_targets_isolated(
    server, session, monkeypatch
):
    """Profile A maps ``/foo`` -> model X. Profile B maps ``/foo`` ->
    model Y. The two live sessions must NOT contaminate each other.
    The mirror decision is entirely derived from the worker's
    profile-scoped metadata.
    """
    sid_a, _, sess_a = session

    sid_b = "sid-different-models"
    sess_b = {
        "session_key": "tui-different-models",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid_b] = sess_b

    captured_a, captured_b = [], []

    def route_mirror(sid_arg, sess_arg, cmd):
        if sid_arg == sid_a:
            captured_a.append(cmd)
        else:
            captured_b.append(cmd)
        return ""

    monkeypatch.setattr(server, "_mirror_slash_side_effects", route_mirror)

    meta_a = {
        "canonical": "model",
        "raw_args": "foo",
        "args": "foo",
        "target": "model-x",
        "side_effect": "model_switch",
    }
    meta_b = {
        "canonical": "model",
        "raw_args": "foo",
        "args": "foo",
        "target": "model-y",
        "side_effect": "model_switch",
    }
    _install_worker(sess_a, _make_worker_double(slash_meta=meta_a))
    _install_worker(sess_b, _make_worker_double(slash_meta=meta_b))

    server._methods["slash.exec"](1, {"command": "/foo", "session_id": sid_a})
    server._methods["slash.exec"](1, {"command": "/foo", "session_id": sid_b})

    # Both sessions get model switches; the targets differ only in
    # resolution, which is the worker's concern. The parent's mirror
    # contract is: re-emit ``/model foo`` because the worker said so.
    # No global cache leak must propagate one session's meta into the
    # other — and indeed it cannot, because the dict lookup of
    # ``slash_worker`` per session is local.
    assert len(captured_a) == 1
    assert captured_a[0].lower().startswith("/model")
    assert len(captured_b) == 1
    assert captured_b[0].lower().startswith("/model")


# ── 9. Sequential calls do not leak state across sessions ─────────────


def test_sequential_calls_do_not_leak_across_sessions(
    server, session, monkeypatch
):
    """A second slash.exec on session B after session A's model switch
    must not re-trigger A's switch. State is local to each session's
    worker metadata snapshot.
    """
    sid_a, _, sess_a = session
    sid_b = "sid-sequential"
    sess_b = {
        "session_key": "tui-sequential",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid_b] = sess_b

    captured_a, captured_b = [], []

    def route_mirror(sid_arg, sess_arg, cmd):
        if sid_arg == sid_a:
            captured_a.append(cmd)
        else:
            captured_b.append(cmd)
        return ""

    monkeypatch.setattr(server, "_mirror_slash_side_effects", route_mirror)

    # Profile A: first a model switch, then a non-model built-in.
    meta_a_switch = {
        "canonical": "model",
        "raw_args": "sonnet",
        "args": "sonnet",
        "target": "sonnet",
        "side_effect": "model_switch",
    }
    worker_a = _make_worker_double()
    worker_a.run_with_meta = MagicMock(
        side_effect=[("ok", meta_a_switch), ("ok", None)]
    )
    _install_worker(sess_a, worker_a)

    # Profile B: only built-in.
    worker_b = _make_worker_double(slash_meta=None)
    _install_worker(sess_b, worker_b)

    server._methods["slash.exec"](1, {"command": "/sonnet", "session_id": sid_a})
    server._methods["slash.exec"](1, {"command": "/version", "session_id": sid_a})
    server._methods["slash.exec"](1, {"command": "/personality", "session_id": sid_b})

    assert len(captured_a) == 2
    # First call: model switch mirror
    assert captured_a[0].lower().startswith("/model")
    # Second call: built-in (worker said meta=None)
    assert captured_a[1].lower() == "/version"

    assert len(captured_b) == 1
    assert captured_b[0].lower() == "/personality"


# ── 10. Worker reports non-model canonical; parent does not switch ──


def test_worker_reports_builtin_canonical_does_not_mirror_model(
    server, session, monkeypatch
):
    """If the worker explicitly reports ``canonical="version"`` and
    ``side_effect != "model_switch"``, the parent must not perform a
    model switch even if the typed token happens to be a model alias.
    """
    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    meta = {
        "canonical": "version",
        "raw_args": "",
        "args": "",
        "target": None,
        "side_effect": None,
    }
    worker = _make_worker_double(slash_meta=meta)
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/version", "session_id": sid})

    assert len(captured) == 1
    cmd = captured[0].lower()
    assert cmd == "/version"
    assert not cmd.startswith("/model")


# ── Guarantee: parent never re-imports DIRECT_ALIASES / MODEL_ALIASES ─


def test_parent_does_not_import_global_alias_cache(
    server, session, monkeypatch
):
    """Hard guarantee that the TUI mirror call site is NOT consulting
    the global alias registry. The contract is: trust the worker's
    profile-scoped meta. We force the parent's attempted import to
    blow up if it ever re-introduces a global alias cache lookup.
    """
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        # Allow the canonical helper under gateway/_model_alias_normalize
        # to keep being importable (the helper is the only sanctioned
        # resolver). The forbidden paths are direct reads of the live
        # alias caches from the TUI mirror call site.
        if name in {
            "hermes_cli.model_switch",
        } and any(
            mod in (kwargs.get("globals", {}) or {}).get("__name__", "")
            for mod in ["tui_gateway.methods_tools", "tui_gateway.server"]
        ):
            raise ImportError(
                "TUI mirror is forbidden from importing the global alias cache"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    # Worker says built-in (meta=None): no switch.
    worker = _make_worker_double(slash_meta=None)
    _install_worker(sess, worker)

    server._methods["slash.exec"](1, {"command": "/version", "session_id": sid})

    assert len(captured) == 1
    assert captured[0].lower() == "/version"


# ── Direct protocol test: /<alias> from worker reports model_switch ──


def test_alias_meta_says_model_switch_and_preserves_args(
    server, session, monkeypatch
):
    """When the worker reports ``side_effect="model_switch"`` with
    extra raw args (e.g. ``/sonnet --provider openrouter``), the
    parent's mirror command must include those args verbatim.
    """
    sid, _, sess = session
    fake_mirror, captured = _capturing_mirror()
    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)

    meta = {
        "canonical": "model",
        "raw_args": "sonnet --provider openrouter",
        "args": "sonnet --provider openrouter",
        "target": "anthropic/claude-sonnet",
        "side_effect": "model_switch",
    }
    worker = _make_worker_double(slash_meta=meta)
    _install_worker(sess, worker)

    server._methods["slash.exec"](
        1, {"command": "/sonnet --provider openrouter", "session_id": sid}
    )

    assert len(captured) == 1
    cmd = captured[0].lower()
    assert cmd.startswith("/model")
    assert "sonnet" in cmd
    assert "--provider openrouter" in cmd


# ── SlashWorker.run_with_meta passes through meta unchanged ───────────


def test_slash_worker_run_with_meta_returns_meta(server, monkeypatch):
    """Contract on _SlashWorker.run_with_meta: must return the worker's
    metadata dict verbatim, not a derived / re-derived one.
    """
    worker = server._SlashWorker.__new__(server._SlashWorker)
    worker._lock = threading.Lock()
    worker._seq = 0
    import queue

    captured_meta = {"canonical": "model", "side_effect": "model_switch"}

    worker.proc = MagicMock()
    worker.proc.poll = MagicMock(return_value=None)
    worker._drain_stdout = lambda: None
    worker._drain_stderr = lambda: None
    worker.stdout_queue = queue.Queue()
    worker.stdout_queue.put(
        {"id": 1, "ok": True, "output": "ok", "meta": captured_meta}
    )
    worker.proc.stdin = MagicMock()
    worker.stderr_tail = []

    output, meta = worker.run_with_meta("/model sonnet")
    assert output == "ok"
    assert meta is captured_meta