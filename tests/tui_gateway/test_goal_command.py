"""Tests for /goal handling in tui_gateway.

The TUI routes ``/goal`` through ``command.dispatch`` (not ``slash.exec``)
because the CLI's ``_handle_goal_command`` queues the kickoff message onto
``_pending_input``, which the slash-worker subprocess has no reader for.
Instead we handle ``/goal`` directly in the server and return a
``{"type": "send", "notice": ..., "message": ...}`` payload the TUI client
uses to render a system line and fire the kickoff prompt.
"""

from __future__ import annotations

import importlib
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module DB cache so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home, monkeypatch):
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    # Pin config resolution to the isolated HERMES_HOME. Sibling test
    # files (test_billing_rpc, test_delegation_session_lifecycle,
    # test_gateway_owned_session_reap, ...) import tui_gateway.server at
    # collection time — BEFORE the conftest env isolation runs — so the
    # module-level ``_hermes_home = get_hermes_home()`` snapshot freezes
    # the developer's real home. When any of them precede this file in
    # the same process, ``importlib.import_module`` returns that cached
    # module and ``_load_cfg()`` would read the REAL config.yaml (e.g. a
    # local MoA preset) instead of the one ``_write_moa_config`` writes.
    # Also reset the mtime-keyed config cache; monkeypatch restores the
    # originals on teardown so nothing leaks to later tests either.
    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_mtime", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    # Reset module-level session state without re-importing. importlib.reload
    # would re-register the module's atexit hooks (ThreadPoolExecutor
    # shutdown, _shutdown_sessions); the duplicates race the stderr
    # buffer at interpreter shutdown and surface as Fatal Python error:
    # _enter_buffered_busy. Clearing the per-session dicts gives the
    # next test a clean slate.
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()


@pytest.fixture()
def session(server):
    sid = "sid-test"
    session_key = "tui-goal-session-1"
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


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


class _InlineThread:
    """Run a turn synchronously so its automatic follow-up is observable."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, **_extra):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


@pytest.fixture()
def turn_env(server, monkeypatch, tmp_path):
    """Neutralize side paths unrelated to post-turn goal continuation."""
    emitted = []
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload)),
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(
        server, "_sync_agent_model_with_config", lambda sid, session: None
    )
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *a, **k: None
    )
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    return emitted


def _turn_session(agent, session_key):
    return {
        "agent": agent,
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
    }


def _compression_failure():
    message = "Context length exceeded: max compression attempts (3) reached."
    return {
        "final_response": message,
        "error": message,
        "failed": True,
        "partial": True,
        "compression_exhausted": True,
        "completed": False,
    }


# ── command.dispatch /goal ────────────────────────────────────────────


def test_goal_bare_shows_status_when_none_set(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def _exhaust_budget(session_key: str, goal_text: str = "finish the benchmark"):
    """Set a 1-turn goal and drive it to budget-exhaustion auto-pause."""
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_key)
    mgr.set(goal_text, max_turns=1)
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "needs more steps", False, None, False),
    ):
        decision = mgr.evaluate_after_turn("worked a bit")
    assert decision["status"] == "paused"
    assert decision["should_continue"] is False
    return mgr


def test_goal_resume_after_budget_exhaustion_dispatches_continuation(
    server, session
):
    """#75362: /goal resume must restart work, not just flip state.

    The pre-fix handler returned a display-only `exec` payload, so the
    resumed goal sat idle until the user sent another message. Resume
    must return a sendable dispatch carrying the canonical continuation
    prompt, with a concise `/goal resume` transcript projection.
    """
    from hermes_cli.goals import GoalManager

    sid, session_key, _ = session
    _exhaust_budget(session_key)
    assert GoalManager(session_key).state.status == "paused"

    r = _call(server, "command.dispatch", name="goal", arg="resume", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"].startswith("[Continuing toward your standing goal]")
    assert result["display"] == "/goal resume"
    assert "Goal resumed" in result["notice"]

    state = GoalManager(session_key).state
    assert state.status == "active"
    assert state.turns_used == 0, "resume must reset the turn budget"


def test_goal_resume_without_goal_stays_exec(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="resume", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No goal to resume" in r["result"]["output"]


# ── slash.exec /goal routing ──────────────────────────────────────────


def test_slash_exec_routes_goal_to_command_dispatch(server, session):
    """slash.exec must route /goal directly to command.dispatch internally
    instead of returning an error.  Previously the 4018 error required the
    TUI client to retry via command.dispatch, but some clients failed the
    fallback, leaving the command empty ("empty command")."""
    sid, _, _ = session
    r = _call(server, "slash.exec", command="goal status", session_id=sid)
    # Should succeed by routing to command.dispatch internally
    assert "result" in r
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_pending_input_commands_includes_goal(server):
    """Guard: _PENDING_INPUT_COMMANDS must list 'goal' — removing it would
    silently re-break the TUI."""
    assert "goal" in server._PENDING_INPUT_COMMANDS


# ── active-goal recovery after compression exhaustion ───────────────


def test_active_goal_retries_once_without_judging_failed_turn(
    server, turn_env, monkeypatch
):
    from hermes_cli.goals import GoalManager

    session_key = "goal-compression-retry"
    mgr = GoalManager(session_key)
    mgr.set("finish the current task")
    continuation = mgr.next_continuation_prompt()
    seen_prompts = []
    results = iter([_compression_failure(), {"final_response": "recovered work"}])

    def run_conversation(message, **_kwargs):
        seen_prompts.append(message)
        return next(results)

    judged = []

    def evaluate(self, response, **_kwargs):
        judged.append(response)
        return {"message": "", "should_continue": False}

    monkeypatch.setattr(GoalManager, "evaluate_after_turn", evaluate)
    agent = types.SimpleNamespace(
        session_id=session_key,
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
    )
    session = _turn_session(agent, session_key)

    server._run_prompt_submit("rid", "sid", session, "initial work")

    assert seen_prompts == ["initial work", continuation]
    assert judged == ["recovered work"]
    assert GoalManager(session_key).state.turns_used == 0
    assert server._GOAL_COMPRESSION_RECOVERY_ATTEMPTS not in session
    completes = [p for event, _sid, p in turn_env if event == "message.complete"]
    assert [p["status"] for p in completes] == ["error", "complete"]


def test_second_consecutive_exhaustion_pauses_goal_instead_of_looping(
    server, turn_env, monkeypatch
):
    from hermes_cli.goals import GoalManager

    session_key = "goal-compression-pause"
    GoalManager(session_key).set("finish the current task")
    seen_prompts = []

    def run_conversation(message, **_kwargs):
        seen_prompts.append(message)
        return _compression_failure()

    judged = []
    monkeypatch.setattr(
        GoalManager,
        "evaluate_after_turn",
        lambda self, response, **kwargs: judged.append(response),
    )
    agent = types.SimpleNamespace(
        session_id=session_key,
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
    )
    session = _turn_session(agent, session_key)

    server._run_prompt_submit("rid", "sid", session, "initial work")

    assert len(seen_prompts) == 2
    assert judged == []
    state = GoalManager(session_key).state
    assert state.status == "paused"
    assert state.turns_used == 0
    assert "compression exhausted twice" in state.paused_reason
    assert server._GOAL_COMPRESSION_RECOVERY_ATTEMPTS not in session
    notices = [
        p["text"]
        for event, _sid, p in turn_env
        if event == "status.update" and p.get("kind") == "goal"
    ]
    assert any("Retrying the active goal once" in text for text in notices)
    assert any("Goal paused" in text for text in notices)


def test_real_queued_prompt_preempts_goal_compression_retry(
    server, turn_env, monkeypatch
):
    from hermes_cli.goals import GoalManager

    session_key = "goal-compression-user-preempts"
    mgr = GoalManager(session_key)
    mgr.set("finish the current task")
    continuation = mgr.next_continuation_prompt()
    seen_prompts = []
    session_holder = {}

    def run_conversation(message, **_kwargs):
        seen_prompts.append(message)
        if len(seen_prompts) == 1:
            server._enqueue_prompt(session_holder["session"], "real user input", None)
            return _compression_failure()
        return {"final_response": "handled the user's update"}

    monkeypatch.setattr(
        GoalManager,
        "evaluate_after_turn",
        lambda self, response, **kwargs: {"message": "", "should_continue": False},
    )
    agent = types.SimpleNamespace(
        session_id=session_key,
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
    )
    session = _turn_session(agent, session_key)
    session_holder["session"] = session

    server._run_prompt_submit("rid", "sid", session, "initial work")

    assert seen_prompts == ["initial work", "real user input"]
    assert continuation not in seen_prompts
    assert server._GOAL_COMPRESSION_RECOVERY_ATTEMPTS not in session


def test_compression_deferred_is_not_treated_as_exhaustion(server):
    from hermes_cli.goals import GoalManager

    session_key = "goal-compression-deferred"
    GoalManager(session_key).set("finish the current task")
    session = {"session_key": session_key}

    prompt, notice = server._plan_goal_compression_recovery(
        session,
        {"compression_deferred": True, "failed": True},
        status="error",
        raw="Compression is already in progress.",
    )

    assert prompt is None
    assert notice is None
    assert server._GOAL_COMPRESSION_RECOVERY_ATTEMPTS not in session


def test_exhaustion_without_active_goal_keeps_error_only_behavior(server):
    session = {"session_key": "goal-compression-none"}

    prompt, notice = server._plan_goal_compression_recovery(
        session,
        _compression_failure(),
        status="error",
        raw="Context length exceeded.",
    )

    assert prompt is None
    assert notice is None
    assert server._GOAL_COMPRESSION_RECOVERY_ATTEMPTS not in session


def test_new_goal_does_not_inherit_previous_goal_recovery_attempt(server):
    from hermes_cli.goals import GoalManager

    session_key = "goal-compression-replaced"
    mgr = GoalManager(session_key)
    mgr.set("first goal")
    session = {"session_key": session_key}

    first_prompt, _ = server._plan_goal_compression_recovery(
        session,
        _compression_failure(),
        status="error",
        raw="Context length exceeded.",
    )
    mgr.set("replacement goal")
    replacement_prompt, replacement_notice = server._plan_goal_compression_recovery(
        session,
        _compression_failure(),
        status="error",
        raw="Context length exceeded.",
    )

    assert first_prompt is not None
    assert replacement_prompt is not None
    assert "replacement goal" in replacement_prompt
    assert "Retrying the active goal once" in replacement_notice
    assert GoalManager(session_key).state.status == "active"


# ── command.dispatch /moa ────────────────────────────────────────────

def _write_moa_config(home, text):
    cfg_path = home / "config.yaml"
    cfg_path.write_text(text)


def test_moa_bare_returns_usage(server, session, hermes_home):
    _write_moa_config(hermes_home, """
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""")
    sid, _, s = session
    r = _call(server, "command.dispatch", name="moa", arg="", session_id=sid)
    # Bare /moa is usage-only now; switching to a preset is via the model picker.
    assert "error" in r
    assert "model_override" not in s
