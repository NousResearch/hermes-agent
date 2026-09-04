"""Remote session kernels (tools/code_kernel_remote.py) — hermes-agent#96873.

These tests drive execute_in_remote_kernel against a scripted fake env that
implements the same contract as docker/ssh/modal envs (run-to-completion
execute()), with canned outputs for the spawn/liveness/cell round-trips.
The REAL end-to-end behavior (actual detached processes, real files, real
kill) was verified live on Windows against a bash-backed env; these tests
pin the host-side protocol logic: spawn parsing, liveness handling,
state_lost/state_reset reporting, fail-open, and owner isolation.
"""
import json
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.code_kernel_remote import (
    _REMOTE_KERNELS,
    RemoteKernel,
    execute_in_remote_kernel,
    shutdown_all_remote_kernels,
    shutdown_remote_kernels_for_owner,
)


class ScriptedEnv:
    """Contract-faithful fake: answers env.execute() from a script table.

    Handlers are (substring, callable) pairs checked in order; the callable
    receives the command and returns the result dict.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.commands = []

    def get_temp_dir(self):
        return "/tmp"

    def execute(self, command, cwd=None, timeout=None, **_kwargs):
        self.commands.append(command)
        for needle, handler in self.handlers:
            if needle in command:
                return handler(command)
        return {"output": "", "returncode": 0}


def _spawn_ok_handlers(cell_results):
    """Handlers for a healthy kernel: spawn returns PID, liveness ALIVE,
    cat of a cell result file returns the next canned payload."""
    results = list(cell_results)

    def cat_handler(command):
        if results:
            return {"output": json.dumps(results.pop(0)), "returncode": 0}
        return {"output": "", "returncode": 0}

    return [
        ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
        ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
        ("cat ", cat_handler),
    ]


def _cell(status="ok", stdout="", execution_count=1, **kw):
    payload = {
        "id": "000001", "status": status, "stdout": stdout, "stderr": "",
        "stdout_clipped": False, "stderr_clipped": False, "traceback": "",
        "execution_count": execution_count,
    }
    payload.update(kw)
    return payload


def _run(
    env,
    code="print(1)",
    *,
    task="t1",
    reset=False,
    timeout=10,
    session_id="",
    enabled_toolsets=None,
    disabled_toolsets=None,
    idle_exit=1800,
):
    return execute_in_remote_kernel(
        code, env=env, env_type="ssh", task_env_id=task,
        sandbox_tools=frozenset({"read_file"}), timeout=timeout,
        max_tool_calls=5, reset=reset, session_id=session_id,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        idle_exit=idle_exit,
    )


class RemoteKernelBase(unittest.TestCase):
    def setUp(self):
        shutdown_all_remote_kernels()
        # No approval session key in tests → owner falls back to task id,
        # which is exactly the isolation-by-key behavior under test.
        self._ship = patch(
            "tools.code_execution_tool._ship_file_to_remote",
        )
        self._ship.start()
        self._poll = patch(
            "tools.code_execution_tool._rpc_poll_loop",
        )
        self.poll = self._poll.start()

    def tearDown(self):
        self._ship.stop()
        self._poll.stop()
        shutdown_all_remote_kernels()


class TestSpawnAndReuse(RemoteKernelBase):
    def test_cell_rpc_keeps_parent_session_scope(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell()]))

        result = _run(
            env,
            session_id="session-1",
            enabled_toolsets=["calendar"],
            disabled_toolsets=["private"],
        )

        self.assertEqual(result["status"], "success", result)
        poll_args = self.poll.call_args.args
        self.assertEqual(poll_args[9], "session-1")
        self.assertEqual(poll_args[10], ["calendar"])
        self.assertEqual(poll_args[11], ["private"])

    def test_first_call_spawns_second_reuses(self):
        env = ScriptedEnv(_spawn_ok_handlers(
            [_cell(stdout="one\n"), _cell(stdout="two\n", execution_count=2)],
        ))
        first = _run(env)
        self.assertEqual(first["status"], "success", first)
        self.assertFalse(first["kernel"]["reused"])
        second = _run(env)
        self.assertTrue(second["kernel"]["reused"])
        self.assertEqual(second["kernel"]["execution_count"], 2)
        # Exactly one spawn happened.
        self.assertEqual(
            sum(1 for c in env.commands if "nohup" in c), 1,
        )

    def test_spawn_failure_fails_open(self):
        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "sh: cannot fork\n", "returncode": 1}),
        ])
        self.assertIsNone(_run(env))
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_reset_kills_and_respawns(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env)
        result = _run(env, reset=True)
        self.assertTrue(result["kernel"].get("state_reset"))
        self.assertFalse(result["kernel"]["reused"])
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)


class TestDeathDetection(RemoteKernelBase):
    def test_dead_kernel_is_reported_and_respawned(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env)
        # Flip liveness to dead for the next probe only.
        original = env.handlers
        env.handlers = [("kill -0", lambda c: {"output": "", "returncode": 1})] \
            + [h for h in original if h[0] != "kill -0"]
        # Restore ALIVE after the respawn's own probe would run: the spawn
        # path probes liveness once — make the dead answer one-shot.
        state = {"dead_probes": 0}

        def flaky_liveness(command):
            state["dead_probes"] += 1
            if state["dead_probes"] == 1:
                return {"output": "", "returncode": 1}
            return {"output": "ALIVE\n", "returncode": 0}

        env.handlers = [("kill -0", flaky_liveness)] + \
            [h for h in original if h[0] != "kill -0"]
        result = _run(env)
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(result["kernel"].get("state_lost"))
        self.assertIn("state from earlier calls was lost",
                      result["kernel"].get("note", ""))

    def test_cell_timeout_kills_kernel_and_reports(self):
        # cat never returns a result file → cell deadline expires.
        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:77\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", lambda c: {"output": "", "returncode": 0}),
        ])
        result = _run(env, timeout=2)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["kernel"]["state_lost"])
        self.assertEqual(len(_REMOTE_KERNELS), 0)
        # The kernel was actually killed on the remote.
        self.assertTrue(any("kill " in c for c in env.commands))

    def test_idle_registry_entries_are_reaped_before_next_lookup(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))

        with patch(
            "tools.approval.get_current_session_key",
            side_effect=("owner-a", "owner-b"),
        ):
            first = _run(env, task="turn-a", idle_exit=1)
            assert first is not None
            first_kernel = next(iter(_REMOTE_KERNELS.values()))
            first_kernel.last_used = time.monotonic() - 10
            second = _run(env, task="turn-b", idle_exit=1)

        assert second is not None
        self.assertEqual(len(_REMOTE_KERNELS), 1)
        self.assertEqual(next(iter(_REMOTE_KERNELS))[0], "owner-b")

    def test_registry_evicts_lru_kernels_over_configured_cap(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell(), _cell()]))

        with patch(
            "tools.approval.get_current_session_key",
            side_effect=("owner-a", "owner-b", "owner-c"),
        ), patch(
            "tools.code_execution_tool._load_config",
            return_value={
                "max_session_kernels": 2,
                "kernel_idle_timeout": 1800,
            },
        ):
            _run(env, task="turn-a")
            _run(env, task="turn-b")
            _run(env, task="turn-c")

        self.assertEqual(len(_REMOTE_KERNELS), 2)
        self.assertEqual(
            {key[0] for key in _REMOTE_KERNELS},
            {"owner-b", "owner-c"},
        )


class TestOwnershipIsolation(RemoteKernelBase):
    def test_concurrent_first_calls_spawn_one_remote_kernel(self):
        class SlowSpawnEnv(ScriptedEnv):
            def __init__(self):
                self.nohup_calls = 0
                self._guard = threading.Lock()
                self._results = [_cell(), _cell(execution_count=2)]
                super().__init__([
                    ("nohup", self._spawn),
                    ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
                    ("cat ", self._cat),
                ])

            def _spawn(self, _command):
                with self._guard:
                    self.nohup_calls += 1
                time.sleep(0.15)
                return {"output": "PID:4242\n", "returncode": 0}

            def _cat(self, _command):
                with self._guard:
                    result = self._results.pop(0)
                return {"output": json.dumps(result), "returncode": 0}

        env = SlowSpawnEnv()
        results = []

        def run_turn(task):
            results.append(_run(env, task=task))

        with patch(
            "tools.approval.get_current_session_key",
            return_value="shared-conversation",
        ):
            workers = [
                threading.Thread(target=run_turn, args=("turn-1",)),
                threading.Thread(target=run_turn, args=("turn-2",)),
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=5)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(len(results), 2)
        self.assertEqual(env.nohup_calls, 1)
        self.assertEqual(len(_REMOTE_KERNELS), 1)

    def test_concurrent_cells_on_same_remote_kernel_are_serialized(self):
        class OverlapEnv(ScriptedEnv):
            def __init__(self):
                self._results = [_cell(), _cell(), _cell()]
                self._active_cats = 0
                self.max_active_cats = 0
                self._guard = threading.Lock()
                super().__init__([
                    ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
                    ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
                    ("cat ", self._cat),
                ])

            def _cat(self, _command):
                with self._guard:
                    self._active_cats += 1
                    self.max_active_cats = max(
                        self.max_active_cats,
                        self._active_cats,
                    )
                    result = self._results.pop(0)
                time.sleep(0.15)
                with self._guard:
                    self._active_cats -= 1
                return {"output": json.dumps(result), "returncode": 0}

        env = OverlapEnv()
        results = []

        def run_turn(task):
            results.append(_run(env, task=task))

        with patch(
            "tools.approval.get_current_session_key",
            return_value="shared-conversation",
        ):
            first = _run(env, task="turn-1")
            assert first is not None
            workers = [
                threading.Thread(target=run_turn, args=("turn-2",)),
                threading.Thread(target=run_turn, args=("turn-3",)),
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=5)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(len(results), 2)
        self.assertEqual(env.max_active_cats, 1)

    def test_same_session_reuses_remote_kernel_across_turn_task_ids(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell(execution_count=2)]))

        with patch(
            "tools.approval.get_current_session_key",
            return_value="shared-conversation",
        ):
            first = _run(env, task="turn-1")
            second = _run(env, task="turn-2")

        assert first is not None
        assert second is not None
        self.assertFalse(first["kernel"]["reused"])
        self.assertTrue(second["kernel"]["reused"])
        self.assertEqual(len(_REMOTE_KERNELS), 1)
        self.assertEqual(sum(1 for command in env.commands if "nohup" in command), 1)

    def test_profiles_with_same_task_get_distinct_remote_kernels(self):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        def run_in_profile(profile_home):
            token = set_hermes_home_override(profile_home)
            try:
                return _run(env, task="shared-task")
            finally:
                reset_hermes_home_override(token)

        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        first = run_in_profile("/tmp/remote-kernel-profile-a")
        second = run_in_profile("/tmp/remote-kernel-profile-b")

        assert first is not None
        assert second is not None
        self.assertFalse(first["kernel"]["reused"])
        self.assertFalse(second["kernel"]["reused"])
        self.assertEqual(len(_REMOTE_KERNELS), 2)
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)

    def test_delegated_children_get_their_own_remote_kernels(self):
        """Same invariant as local (#94647 review fix): the child context
        qualifier must key a DIFFERENT remote kernel."""
        from agent.delegation_context import delegated_child_context

        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, task="conv")
        with delegated_child_context("child-9"):
            _run(env, task="conv")
        # Two distinct kernels, two spawns.
        self.assertEqual(len(_REMOTE_KERNELS), 2)
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)

    def test_owner_disposal_reaps_only_that_owner(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, task="owner-a")
        _run(env, task="owner-b")
        self.assertEqual(len(_REMOTE_KERNELS), 2)
        shutdown_remote_kernels_for_owner("owner-a")
        self.assertEqual(len(_REMOTE_KERNELS), 1)
        remaining_owner = next(iter(_REMOTE_KERNELS))[0]
        self.assertEqual(remaining_owner, "owner-b")

    def test_profile_disposal_preserves_same_owner_in_other_profile(self):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        def in_profile(profile_home, callback):
            token = set_hermes_home_override(profile_home)
            try:
                return callback()
            finally:
                reset_hermes_home_override(token)

        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        profile_a = "/tmp/remote-kernel-cleanup-profile-a"
        profile_b = "/tmp/remote-kernel-cleanup-profile-b"
        in_profile(profile_a, lambda: _run(env, task="shared-owner"))
        in_profile(profile_b, lambda: _run(env, task="shared-owner"))

        in_profile(
            profile_a,
            lambda: shutdown_remote_kernels_for_owner("shared-owner"),
        )

        self.assertEqual(len(_REMOTE_KERNELS), 1)
        remaining_key = next(iter(_REMOTE_KERNELS))
        self.assertEqual(remaining_key[1], os.path.realpath(profile_b))


class TestIdleReapAndCapEviction(RemoteKernelBase):
    """Unlike local session kernels, remote kernels had no idle-reap or
    process-wide cap: _REMOTE_KERNELS grew one entry per distinct
    (owner, env_type, task_env_id) that was never revisited, for the life
    of the gateway process."""

    def test_idle_expired_kernel_is_reaped_on_next_call(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        execute_in_remote_kernel(
            "print(1)", env=env, env_type="ssh", task_env_id="stale",
            sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
            reset=False, idle_exit=1800,
        )
        self.assertEqual(len(_REMOTE_KERNELS), 1)
        # Backdate the kernel's last_used past the idle window — simulates
        # a key that is never revisited again.
        for kernel in _REMOTE_KERNELS.values():
            kernel.last_used -= 2000
        # A call for a DIFFERENT key must reap the stale entry on entry,
        # without ever touching or reviving it.
        execute_in_remote_kernel(
            "print(1)", env=env, env_type="ssh", task_env_id="fresh",
            sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
            reset=False, idle_exit=1800,
        )
        owners = {key[0] for key in _REMOTE_KERNELS}
        self.assertNotIn("stale", owners)
        self.assertIn("fresh", owners)

    def test_over_cap_evicts_least_recently_used(self):
        with patch("tools.code_kernel._lifecycle_limits", return_value=(2, 1800)):
            env = ScriptedEnv(_spawn_ok_handlers([_cell() for _ in range(10)]))
            for i in range(3):
                execute_in_remote_kernel(
                    "print(1)", env=env, env_type="ssh", task_env_id=f"owner-{i}",
                    sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
                    reset=False, idle_exit=1800,
                )
            self.assertEqual(len(_REMOTE_KERNELS), 2)
            owners = {key[0] for key in _REMOTE_KERNELS}
            self.assertNotIn("owner-0", owners)
            self.assertIn("owner-1", owners)
            self.assertIn("owner-2", owners)

    def test_eviction_skips_kernels_with_a_running_cell(self):
        """Cap eviction must never kill a kernel mid-cell (the local-kernel
        race from hermes-agent#101861): a busy kernel stays put and a
        settled one goes instead, even if the busy one is older."""
        import threading

        gate = threading.Event()

        def slow_cat(command):
            gate.wait(10)
            return {"output": json.dumps(_cell()), "returncode": 0}

        busy_env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", slow_cat),
        ])
        with patch("tools.code_kernel._lifecycle_limits", return_value=(1, 1800)):
            worker = threading.Thread(target=_run, args=(busy_env,), kwargs={"task": "busy"})
            worker.start()
            while not any(k.attached for k in _REMOTE_KERNELS.values()):
                pass
            env = ScriptedEnv(_spawn_ok_handlers([_cell()]))
            _run(env, task="settled")
            owners = {key[0] for key in _REMOTE_KERNELS}
            self.assertIn("busy", owners)
            gate.set()
            worker.join(10)
        self.assertFalse(any("kill 4242" in c for c in busy_env.commands))


class TestDispatchIntegration(unittest.TestCase):
    """_execute_remote prefers the kernel and falls open to per-call."""

    def test_execute_remote_uses_kernel_result(self):
        from tools.code_execution_tool import _execute_remote

        fake = {
            "status": "success", "stdout": "kernel says hi\n", "stderr": "",
            "traceback": "", "tool_calls_made": 0,
            "kernel": {"reused": True, "remote": True, "execution_count": 3},
        }
        env = ScriptedEnv([
            ("command -v python3", lambda c: {"output": "OK\n", "returncode": 0}),
        ])
        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")), \
             patch("tools.code_kernel_remote.execute_in_remote_kernel",
                   return_value=fake) as kernel:
            result = json.loads(_execute_remote(
                "print()",
                "t",
                ["read_file"],
                session_id="session-1",
                enabled_toolsets=["calendar"],
                disabled_toolsets=["private"],
            ))
        self.assertEqual(result["status"], "success")
        self.assertIn("kernel says hi", result["output"])
        self.assertEqual(result["kernel"]["execution_count"], 3)
        self.assertEqual(kernel.call_args.kwargs["session_id"], "session-1")
        self.assertEqual(kernel.call_args.kwargs["enabled_toolsets"], ["calendar"])
        self.assertEqual(kernel.call_args.kwargs["disabled_toolsets"], ["private"])

    def test_execute_remote_falls_open_to_per_call(self):
        from tools.code_execution_tool import _execute_remote
        from unittest.mock import MagicMock

        env = ScriptedEnv([
            ("command -v python3", lambda c: {"output": "OK\n", "returncode": 0}),
            ("python3 script.py", lambda c: {"output": "per-call ran\n",
                                             "returncode": 0}),
        ])
        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")), \
             patch("tools.code_kernel_remote.execute_in_remote_kernel",
                   return_value=None), \
             patch("tools.code_execution_tool._ship_file_to_remote"), \
             patch("tools.code_execution_tool.threading.Thread",
                   return_value=MagicMock()):
            result = json.loads(_execute_remote("print()", "t", ["read_file"]))
        self.assertEqual(result["status"], "success")
        self.assertIn("per-call ran", result["output"])


if __name__ == "__main__":
    unittest.main()
