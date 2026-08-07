"""CheckpointManager snapshot scope: turn (default) vs task (#68877).

scope="turn" resets the per-directory dedup on every agent iteration, so each
turn can take one snapshot (historical behavior). scope="task" makes new_turn()
a no-op — the dedup persists across the task's iterations so only the first
file mutation snapshots the pre-task baseline — while new_task() re-arms it at
the next task boundary. These tests exercise the dedup bookkeeping directly
(no git needed) by inspecting ``_checkpointed_dirs``.
"""

from tools.checkpoint_manager import CheckpointManager


def _redirect_agent():
    """Minimal real AIAgent for driving ``_apply_active_turn_redirect``.

    Mirrors the ``object.__new__`` stub pattern in tests/run_agent/test_steer.py,
    which already exercises this same function.
    """
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._current_streamed_assistant_text = ""
    agent._stream_needs_break = False
    agent._strip_think_blocks = lambda content: content
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    return agent


class TestScopeNormalization:
    def test_default_scope_is_turn(self):
        assert CheckpointManager().scope == "turn"

    def test_task_scope_honored(self):
        assert CheckpointManager(scope="task").scope == "task"

    def test_scope_case_insensitive_and_trimmed(self):
        assert CheckpointManager(scope="  TASK ").scope == "task"

    def test_unknown_scope_degrades_to_turn(self):
        assert CheckpointManager(scope="bogus").scope == "turn"
        assert CheckpointManager(scope="").scope == "turn"


class TestTurnScopeDedup:
    def test_new_turn_clears_dedup_each_iteration(self):
        mgr = CheckpointManager(enabled=True, scope="turn")
        # Simulate a snapshot having been taken this turn.
        mgr._checkpointed_dirs.add("/work")
        mgr.new_turn()
        # Cleared → the next iteration is free to snapshot again.
        assert "/work" not in mgr._checkpointed_dirs


class TestTaskScopeDedup:
    def test_new_turn_is_noop_in_task_scope(self):
        mgr = CheckpointManager(enabled=True, scope="task")
        mgr._checkpointed_dirs.add("/work")
        mgr.new_turn()
        # Persisted → later turns in the same task will skip re-snapshotting.
        assert "/work" in mgr._checkpointed_dirs

    def test_new_task_clears_even_in_task_scope(self):
        mgr = CheckpointManager(enabled=True, scope="task")
        mgr._checkpointed_dirs.add("/work")
        mgr.new_task()
        assert "/work" not in mgr._checkpointed_dirs

    def test_new_task_clears_in_turn_scope_too(self):
        mgr = CheckpointManager(enabled=True, scope="turn")
        mgr._checkpointed_dirs.add("/work")
        mgr.new_task()
        assert "/work" not in mgr._checkpointed_dirs


class TestEndToEndDedupSemantics:
    """Model the loop's calls (new_task once, new_turn per iteration) and
    assert how many times a directory would be eligible for a snapshot."""

    def _eligible_count(self, scope, iterations):
        """Count how many iterations would take a snapshot: a dir is eligible
        when it is NOT already in the dedup set. Mirrors ensure_checkpoint's
        ``if abs_dir in self._checkpointed_dirs: return False`` gate."""
        mgr = CheckpointManager(enabled=True, scope=scope)
        mgr.new_task()  # task boundary, before the loop
        took = 0
        for _ in range(iterations):
            mgr.new_turn()  # start of each agent iteration
            if "/work" not in mgr._checkpointed_dirs:
                took += 1
                mgr._checkpointed_dirs.add("/work")  # ensure_checkpoint records it
        return took

    def test_turn_scope_snapshots_every_iteration(self):
        assert self._eligible_count("turn", iterations=5) == 5

    def test_task_scope_snapshots_once_per_task(self):
        assert self._eligible_count("task", iterations=5) == 1

    def test_task_scope_rearms_on_next_task(self):
        mgr = CheckpointManager(enabled=True, scope="task")
        # Task 1
        mgr.new_task()
        mgr.new_turn()
        assert "/work" not in mgr._checkpointed_dirs
        mgr._checkpointed_dirs.add("/work")
        mgr.new_turn()
        assert "/work" in mgr._checkpointed_dirs  # no second snapshot in task 1
        # Task 2 — fresh baseline
        mgr.new_task()
        assert "/work" not in mgr._checkpointed_dirs


class TestRedirectReArmsTaskBaseline:
    """A mid-turn correction is a user-message boundary too (#68877 review).

    ``run_conversation`` calls ``new_task()`` once before the iteration loop,
    but a redirect appends a real user message *inside* that loop and keeps
    going. Without a reset there, a scope="task" run would keep rolling back
    to the state before the *original* instruction — discarding the work the
    correction just asked for.
    """

    def test_applying_a_redirect_rearms_the_task_baseline(self):
        """Drives the real redirect path with a real CheckpointManager."""
        from agent.conversation_loop import _apply_active_turn_redirect

        mgr = CheckpointManager(enabled=True, scope="task")
        mgr._checkpointed_dirs.add("/work")  # the task already snapshotted

        agent = _redirect_agent()
        agent._checkpoint_mgr = mgr
        agent._current_streamed_assistant_text = "partial answer"
        messages = [{"role": "user", "content": "do the thing"}]

        _apply_active_turn_redirect(agent, messages, "actually, do it this way")

        assert messages[-1] == {
            "role": "user", "content": "actually, do it this way",
        }
        assert "/work" not in mgr._checkpointed_dirs, (
            "a correction is a task boundary; without re-arming, a "
            "scope='task' rollback discards the corrected work"
        )

    def test_redirect_without_a_checkpoint_manager_is_harmless(self):
        from agent.conversation_loop import _apply_active_turn_redirect

        agent = _redirect_agent()  # no _checkpoint_mgr installed
        messages = [{"role": "user", "content": "x"}]
        _apply_active_turn_redirect(agent, messages, "y")
        assert messages[-1]["content"] == "y"

    def test_new_task_after_simulated_redirect_allows_a_fresh_baseline(self):
        mgr = CheckpointManager(enabled=True, scope="task")
        # First mutation of the task snapshots and dedups the dir.
        mgr._checkpointed_dirs.add("/work")
        # Later iterations in the same task must NOT re-snapshot...
        mgr.new_turn()
        assert "/work" in mgr._checkpointed_dirs
        # ...but a correction arrives, which is a new task boundary.
        mgr.new_task()
        assert "/work" not in mgr._checkpointed_dirs


class TestScopeReachesConstructionPaths:
    """checkpoints.scope must survive the trip to every agent surface."""

    def test_gateway_kwargs_forward_scope(self):
        from gateway.run import _checkpoint_agent_kwargs

        kwargs = _checkpoint_agent_kwargs({"checkpoints": {"enabled": True, "scope": "task"}})
        assert kwargs["checkpoint_scope"] == "task"

    def test_gateway_kwargs_default_scope_is_turn(self):
        from gateway.run import _checkpoint_agent_kwargs

        assert _checkpoint_agent_kwargs({"checkpoints": {"enabled": True}})["checkpoint_scope"] == "turn"

    def test_tui_reads_scope_from_config(self, monkeypatch):
        """HERMES_TUI_CHECKPOINTS only gates *whether* checkpoints run.

        The cadence still comes from config, so a TUI session must not force
        "turn" when the user configured "task".
        """
        from tui_gateway import server

        monkeypatch.setattr(server, "_load_cfg", lambda: {"checkpoints": {"scope": "task"}})
        assert server._load_checkpoint_scope() == "task"

    def test_tui_scope_defaults_to_turn_when_unset_or_malformed(self, monkeypatch):
        from tui_gateway import server

        for cfg in ({}, {"checkpoints": True}, {"checkpoints": {}}, {"checkpoints": {"scope": "  "}}):
            monkeypatch.setattr(server, "_load_cfg", lambda cfg=cfg: cfg)
            assert server._load_checkpoint_scope() == "turn"

    def test_tui_agent_construction_passes_the_scope(self, monkeypatch):
        """Reading config is useless if the constructor never receives it.

        Captures the kwargs `_make_agent` actually hands to AIAgent instead of
        inspecting its source.
        """
        from tui_gateway import server

        captured = {}

        class _FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def __getattr__(self, name):
                return lambda *a, **k: None

        import run_agent
        from types import SimpleNamespace

        monkeypatch.setattr(server, "_load_cfg", lambda: {"checkpoints": {"scope": "task"}})
        # An isolated HERMES_HOME has no provider configured, so stub the
        # resolution step; this test is about what reaches the constructor.
        monkeypatch.setattr(
            server, "_resolve_runtime_with_fallback",
            lambda *a, **k: SimpleNamespace(
                runtime={
                    "provider": "openrouter", "base_url": "https://x/v1",
                    "api_key": "sk-x", "api_mode": "chat_completions",
                },
                selected_model="some/model",
                used_fallback=False,
                notice=None,
            ),
        )
        # _make_agent imports AIAgent from run_agent inside the function body.
        monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)
        monkeypatch.setenv("HERMES_TUI_CHECKPOINTS", "1")

        error = None
        try:
            server._make_agent("sid", "key")
        except Exception as exc:  # noqa: BLE001
            # _make_agent does plenty besides constructing the agent (model
            # resolution, provider routing). Those are not what this test is
            # about — but if it blew up *before* the constructor, say so
            # rather than silently asserting on an empty dict.
            error = exc

        assert captured, (
            "AIAgent was never constructed, so this test proves nothing "
            f"about checkpoint_scope: {type(error).__name__}: {error}"
        )
        assert captured.get("checkpoint_scope") == "task", (
            f"checkpoint_scope never reached AIAgent; got {sorted(captured)}"
        )


class TestOneshotPathCarriesTheScope:
    """`hermes -z` builds its own AIAgent in hermes_cli/oneshot.py.

    That construction sat outside the propagation added for the interactive
    CLI and the gateway, so a profile with ``checkpoints.scope: task`` was
    silently downgraded to turn scope for every oneshot run -- and a oneshot
    worker is exactly where task scope matters, since the whole run is one
    task.

    Drives the real ``_run_agent`` and captures what AIAgent is constructed
    with, rather than re-deriving the resolution in the test.
    """

    def _captured_kwargs(self, monkeypatch, checkpoints_cfg):
        import hermes_cli.config as cfg_mod
        import hermes_cli.runtime_provider as rp_mod
        import hermes_cli.tools_config as tc_mod
        import run_agent as ra_mod
        from hermes_cli import oneshot

        captured = {}

        class _FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.suppress_status_output = False
                self.stream_delta_callback = None
                self.tool_gen_callback = None

            def run_conversation(self, *_a, **_k):
                return {"final_response": "ok"}

            def __getattr__(self, name):
                return lambda *a, **k: None

        # _run_agent imports these lazily inside the function, so the source
        # modules are what must be patched.
        monkeypatch.setattr(ra_mod, "AIAgent", _FakeAgent)
        monkeypatch.setattr(
            cfg_mod, "load_config",
            lambda *a, **k: {"checkpoints": checkpoints_cfg, "model": {"default": "m"}},
        )
        monkeypatch.setattr(
            rp_mod, "resolve_runtime_provider",
            lambda *a, **k: {"api_key": "k", "base_url": "u", "provider": "p",
                             "requested_provider": "p", "api_mode": "chat_completions"},
        )
        monkeypatch.setattr(tc_mod, "_get_platform_tools", lambda *a, **k: set())
        monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
        monkeypatch.setattr(oneshot, "get_fallback_chain", lambda _c: [])

        oneshot._run_agent("hello")
        return captured

    def test_task_scope_reaches_the_oneshot_agent(self, monkeypatch):
        kwargs = self._captured_kwargs(monkeypatch, {"scope": "task"})
        assert kwargs.get("checkpoint_scope") == "task", (
            "hermes -z ignored checkpoints.scope -- the whole run would take "
            f"turn-scoped snapshots despite the profile asking for task scope "
            f"(got {kwargs.get('checkpoint_scope')!r})"
        )

    def test_default_is_turn(self, monkeypatch):
        assert self._captured_kwargs(monkeypatch, {}).get("checkpoint_scope") == "turn"

    def test_legacy_boolean_form_does_not_crash(self, monkeypatch):
        """`checkpoints: true` is the old shape; it must still yield a scope."""
        assert self._captured_kwargs(monkeypatch, True).get("checkpoint_scope") == "turn"

    def test_malformed_checkpoints_block_falls_back_to_turn(self, monkeypatch):
        assert self._captured_kwargs(monkeypatch, "nonsense").get("checkpoint_scope") == "turn"
