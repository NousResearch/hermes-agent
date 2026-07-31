"""Tests for /<alias> model switching via the TUI slash.exec path.

Regression coverage for PR #59606 — a typed alias like /sonnet in the TUI
must update the live session's model the same way /model sonnet does.
Without the rewrite at the mirror call site, /sonnet would print
"switched" in the worker thread but leave the live agent's model
unchanged, because _mirror_slash_side_effects only recognises the
literal command name "model".
"""
from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    sid = "sid-alias-test"
    session_key = "tui-alias-session"
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
    """Return a MagicMock that stands in for _SlashWorker so we
    can drive slash.exec without spawning the real worker subprocess.

    ``run_with_meta`` is the canonical hook the parent uses to learn
    the worker's structured side-effect metadata; ``run`` is kept for
    backwards-compatible callers that only need the captured output.
    """
    w = MagicMock()
    w.run = MagicMock(return_value=output)
    w.run_with_meta = MagicMock(return_value=(output, slash_meta))
    w.close = MagicMock()
    return w


def _install_worker(server, session_entry, worker):
    """Attach the fake worker to a session dict."""
    session_entry["slash_worker"] = worker


def _apply_model_switch_double(server):
    """Return an AsyncMock stand-in for _apply_model_switch."""
    return MagicMock(return_value={"value": "sonnet-target", "warning": ""})


# ── slash.exec /<alias> rewrites mirror command to /model ────────────


def test_alias_mirror_command_is_canonical_model(server, session, monkeypatch):
    """When slash.exec receives /sonnet and the worker reports the
    canonical ``model`` side-effect, the command passed to
    _mirror_slash_side_effects must be rewritten to /model sonnet so
    the live agent actually switches. The parent's mirror decision is
    driven by the worker's structured metadata — not by re-deriving
    whether the typed token is in any global alias cache."""
    sid, _, sess = session

    captured_mirror_cmd: list[str] = []

    def fake_mirror(sid_arg, sess_arg, cmd):
        captured_mirror_cmd.append(cmd)
        return ""

    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)
    # The real worker reports model_switch because it routed /sonnet
    # through the canonical /model path inside the session's profile.
    fake_worker = _make_worker_double(
        slash_meta={
            "canonical": "model",
            "raw_args": "sonnet",
            "args": "sonnet",
            "target": "anthropic/claude-sonnet",
            "side_effect": "model_switch",
        }
    )
    _install_worker(server, sess, fake_worker)

    r = server._methods["slash.exec"](1, {"command": "/sonnet", "session_id": sid})
    assert "result" in r
    assert len(captured_mirror_cmd) == 1, (
        f"_mirror_slash_side_effects must be called exactly once, "
        f"got calls: {captured_mirror_cmd}"
    )
    cmd = captured_mirror_cmd[0].lower()
    assert cmd.startswith("/model"), (
        f"alias /sonnet must be rewritten to /model sonnet before "
        f"reaching _mirror_slash_side_effects, got: {captured_mirror_cmd[0]!r}"
    )
    assert "sonnet" in cmd


def test_alias_mirror_preserves_extra_args(server, session, monkeypatch):
    """If the user types /sonnet --provider openrouter, the worker's
    metadata ``raw_args`` field carries those flags verbatim and the
    mirror command must include them so the live-session model switch
    respects them."""
    sid, _, sess = session

    captured: list[str] = []

    def fake_mirror(sid_arg, sess_arg, cmd):
        captured.append(cmd)
        return ""

    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)
    fake_worker = _make_worker_double(
        slash_meta={
            "canonical": "model",
            "raw_args": "sonnet --provider openrouter",
            "args": "sonnet --provider openrouter",
            "target": "anthropic/claude-sonnet",
            "side_effect": "model_switch",
        }
    )
    _install_worker(server, sess, fake_worker)

    r = server._methods["slash.exec"](
        1, {"command": "/sonnet --provider openrouter", "session_id": sid}
    )
    assert "result" in r
    assert len(captured) == 1
    cmd = captured[0].lower()
    assert cmd.startswith("/model")
    assert "sonnet" in cmd
    assert "--provider openrouter" in cmd


def test_canonical_model_command_still_routes_to_model(
    server, session, monkeypatch
):
    """Sanity: typing /model sonnet directly must still route to
    /model sonnet in the mirror. The worker reports
    ``side_effect == "model_switch"`` for canonical /model too, so the
    parent re-emits ``/model sonnet`` unchanged."""
    sid, _, sess = session

    captured: list[str] = []

    def fake_mirror(sid_arg, sess_arg, cmd):
        captured.append(cmd)
        return ""

    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)
    fake_worker = _make_worker_double(
        slash_meta={
            "canonical": "model",
            "raw_args": "sonnet",
            "args": "sonnet",
            "target": "anthropic/claude-sonnet",
            "side_effect": "model_switch",
        }
    )
    _install_worker(server, sess, fake_worker)

    r = server._methods["slash.exec"](
        1, {"command": "/model sonnet", "session_id": sid}
    )
    assert "result" in r
    assert len(captured) == 1
    cmd = captured[0].lower()
    assert cmd.startswith("/model")
    assert "sonnet" in cmd


def test_non_alias_command_is_passed_through_unchanged(
    server, session, monkeypatch
):
    """Non-model commands must be passed to the mirror unchanged so
    the live-session sync continues to work for every other slash
    command (personality, prompt, fast, etc.). Worker reports
    ``meta=None`` for these, so the parent never rewrites."""
    sid, _, sess = session

    captured: list[str] = []

    def fake_mirror(sid_arg, sess_arg, cmd):
        captured.append(cmd)
        return ""

    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)
    fake_worker = _make_worker_double()  # meta=None
    _install_worker(server, sess, fake_worker)

    r = server._methods["slash.exec"](
        1, {"command": "/personality", "session_id": sid}
    )
    assert "result" in r
    assert len(captured) == 1
    assert captured[0].lower() == "/personality"


def test_alias_unknown_command_does_not_rewrite_to_model(
    server, session, monkeypatch
):
    """A non-alias unknown command must NOT be rewritten to /model —
    the alias rewrite must not intercept arbitrary /<x> calls and
    silently route them to a model switch."""
    sid, _, sess = session

    captured: list[str] = []

    def fake_mirror(sid_arg, sess_arg, cmd):
        captured.append(cmd)
        return ""

    monkeypatch.setattr(server, "_mirror_slash_side_effects", fake_mirror)
    fake_worker = _make_worker_double()
    _install_worker(server, sess, fake_worker)

    # /xyzzy-zzzz is not a model alias and not a recognized command.
    # Whatever slash.exec returns, the rewrite at the mirror call site
    # must NOT have promoted it to /model — that would silently route
    # arbitrary text into a model switch and mask typos.
    server._methods["slash.exec"](
        1, {"command": "/xyzzy-zzzz", "session_id": sid}
    )
    if captured:
        cmd = captured[0].lower()
        assert not cmd.startswith("/model "), (
            f"non-alias /xyzzy-zzzz must not be rewritten to /model "
            f"xyzzy-zzzz (silent model switch), got: {captured[0]!r}"
        )