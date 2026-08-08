"""Regression tests: a background spawn that dies must not be reported as started.

``terminal(background=True)`` returned ``"Background process started"`` plus a
``session_id`` unconditionally — even when the process had already died during
spawn. On a non-local backend (SSH, Docker, Modal, Daytona) the wrapper can fail
to produce a PID (unwritable redirect target, broken login shell, transport
error), and ``ProcessRegistry.spawn_via_env`` marks the session
``completion_reason == "failed_start"`` and never registers it as running. The
agent was still handed a handle it could never poll — ``process(action="poll")``
answers ``not_found`` — so the real failure stayed invisible until someone read
the logs by hand.

Note the local ``Popen`` path cannot reach this state: it always obtains a PID,
and a bad command merely exits 127 *after* a successful spawn (a real, pollable
session). ``failed_start`` is specific to the ``env.execute`` backends, which is
why these tests drive a fake environment.

The assertions target the contract, not message strings:

1. spawn death => an error payload, and no ``session_id`` the caller could
   mistake for a pollable handle;
2. the surfaced output carries no unredacted credential from the command line —
   asserted both with the global redaction preference ON and with it OFF, see
   ``TestSpawnFailureOutputIsRedacted``;
3. a healthy spawn still returns the normal started payload — without this an
   over-broad guard that fired on every spawn would satisfy (1) and (2).
"""

import json

import pytest

import tools.terminal_tool as terminal_tool


# A credential-shaped constant: ``sk-`` is a recognised vendor prefix, so this
# is masked by the PREFIX pass in agent/redact.py — the pass that runs
# unconditionally, before the ``code_file`` gate. Do NOT swap this for an
# ENV-assignment shape (``DEPLOY_API_KEY=...``): ``redact_terminal_output``
# computes ``code_file = not is_env_dump_command(command)``, and the command
# under test is not an env dump, so the ENV-assignment pass is skipped BY
# DESIGN. Such a constant leaks against correct production code and would make
# these tests fail for the wrong reason.
SECRET = "sk-live-51HxXaMPLe0000000000000000000000000000000000000000"
COMMAND_WITH_SECRET = f"./deploy.sh --api-key={SECRET}"


class _BrokenSpawnEnvironment:
    """Non-local backend whose background wrapper never emits a PID."""

    env: dict = {}

    def execute(self, command, timeout=None, rewrite_compound_background=False, **kwargs):
        # Echo the command back the way a failing shell wrapper does, so the
        # credential on the command line reaches the captured output.
        return {
            "output": f"bash: cannot create log file for: {command}",
            "returncode": 1,
        }


class _HealthySpawnEnvironment:
    """Non-local backend whose wrapper prints a PID, i.e. a real launch."""

    env: dict = {}

    def execute(self, command, timeout=None, rewrite_compound_background=False, **kwargs):
        return {"output": "4242", "returncode": 0}


@pytest.fixture
def background_env(monkeypatch):
    """Route terminal_tool at a caller-supplied fake non-local environment.

    Also cleans up after a *healthy* fake spawn. Such a spawn registers a
    ``running`` session against a PID that does not exist and starts a
    ``proc-poller-*`` thread which keeps calling ``execute()`` on the fake
    environment. Left alone that entry leaks into every later test in the
    process — ``count_running()``, ``any_running()``, and the checkpoint
    writer all read the same global registry — so the cleanup below is not
    cosmetic.
    """
    from tools.process_registry import process_registry

    def _install(environment, task_id="spawn-contract"):
        monkeypatch.setattr(
            terminal_tool,
            "_get_env_config",
            lambda: {
                "env_type": "ssh",
                "cwd": "/tmp",
                "timeout": 60,
                "lifetime_seconds": 3600,
            },
        )
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        monkeypatch.setattr(
            terminal_tool, "_active_environments", {task_id: environment}
        )
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        return task_id

    with process_registry._lock:
        pre_existing = set(process_registry._running)

    yield _install

    with process_registry._lock:
        leaked = [
            process_registry._running.pop(session_id)
            for session_id in list(process_registry._running)
            if session_id not in pre_existing
        ]
    for session in leaked:
        # Breaks the poller's ``while not session.exited`` loop; the thread
        # then returns at the next iteration instead of polling a fake
        # environment for the lifetime of the test process.
        session.exited = True
        thread = getattr(session, "_reader_thread", None)
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive(), f"poller thread {thread.name} outlived the test"


@pytest.fixture
def redaction_preference_disabled(monkeypatch):
    """Run with the global ``security.redact_secrets`` preference OFF.

    This is the only condition under which the ``force=True`` argument at the
    failed-start boundary is observable: ``redact_sensitive_text`` short-circuits
    on ``if not (force or _REDACT_ENABLED)``, so while the preference is ON the
    preference alone masks the secret and dropping ``force`` changes nothing.

    ``agent.redact._REDACT_ENABLED`` is snapshotted from ``HERMES_REDACT_SECRETS``
    / ``security.redact_secrets`` at *import* time — deliberately, so a runtime
    ``export HERMES_REDACT_SECRETS=false`` cannot disable redaction mid-session.
    Setting the env var from a test is therefore a no-op: the module is long
    imported by the time the test runs. Patching the snapshot is the established
    convention across this suite (``tests/agent/test_redact.py``,
    ``tests/tools/test_process_registry.py``) and, unlike an
    ``importlib.reload``, monkeypatch reverts it automatically instead of
    leaving a rebuilt module behind for every later test.
    """
    import agent.redact

    monkeypatch.setattr(agent.redact, "_REDACT_ENABLED", False)


class TestSpawnDeathIsReportedAsFailure:
    def test_dead_spawn_does_not_report_success(self, background_env):
        task_id = background_env(_BrokenSpawnEnvironment())

        result = json.loads(
            terminal_tool.terminal_tool(
                command="./run-server.sh", background=True, task_id=task_id
            )
        )

        assert result.get("error")
        assert result.get("exit_code") != 0

    def test_dead_spawn_hands_back_no_pollable_handle(self, background_env):
        """A session id the caller cannot poll is worse than no id at all."""
        from tools.process_registry import process_registry

        task_id = background_env(_BrokenSpawnEnvironment())

        result = json.loads(
            terminal_tool.terminal_tool(
                command="./run-server.sh", background=True, task_id=task_id
            )
        )

        session_id = result.get("session_id")
        if session_id is not None:
            # If an id is surfaced at all, it must actually resolve.
            assert process_registry.poll(session_id).get("status") != "not_found"


class TestSpawnFailureOutputIsRedacted:
    def test_credential_from_command_line_is_not_echoed(self, background_env):
        task_id = background_env(_BrokenSpawnEnvironment())

        raw = terminal_tool.terminal_tool(
            command=COMMAND_WITH_SECRET, background=True, task_id=task_id
        )

        assert SECRET not in raw

    def test_credential_is_not_echoed_when_redaction_preference_is_disabled(
        self, background_env, redaction_preference_disabled
    ):
        """The failed-start payload must stay clean even with redaction OFF.

        A spawn failure is a safety boundary: the captured output echoes the
        wrapper's own command line, which may carry a credential the operator
        never intended to surface. That is why the call site passes
        ``force=True``. This test is what makes ``force=True`` load-bearing —
        with the preference ON, the sibling test above passes even if ``force``
        is deleted.
        """
        from agent.redact import redact_terminal_output

        # Control: prove the preference really is off, so the assertion below
        # cannot pass for the wrong reason if this fixture ever stops biting.
        # A non-forced redaction must pass the secret straight through here.
        assert SECRET in redact_terminal_output(
            f"bash: cannot create log file for: {COMMAND_WITH_SECRET}",
            COMMAND_WITH_SECRET,
        )

        task_id = background_env(_BrokenSpawnEnvironment())

        raw = terminal_tool.terminal_tool(
            command=COMMAND_WITH_SECRET, background=True, task_id=task_id
        )

        assert SECRET not in raw


class TestHealthySpawnIsUnaffected:
    """Load-bearing: an over-broad guard firing on every spawn must fail here."""

    def test_healthy_spawn_still_returns_started_payload(self, background_env):
        task_id = background_env(_HealthySpawnEnvironment())

        result = json.loads(
            terminal_tool.terminal_tool(
                command="./run-server.sh", background=True, task_id=task_id
            )
        )

        assert result.get("error") is None
        assert result.get("exit_code") == 0
        assert result.get("session_id")

    def test_healthy_spawn_handle_is_pollable(self, background_env):
        from tools.process_registry import process_registry

        task_id = background_env(_HealthySpawnEnvironment())

        result = json.loads(
            terminal_tool.terminal_tool(
                command="./run-server.sh", background=True, task_id=task_id
            )
        )

        polled = process_registry.poll(result["session_id"])
        assert polled.get("status") != "not_found"
