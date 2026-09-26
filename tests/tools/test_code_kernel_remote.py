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

    def execute(self, command, cwd=None, timeout=None):
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


def _run(env, code="print(1)", *, task="t1", reset=False, timeout=10,
         tools=frozenset({"read_file"})):
    return execute_in_remote_kernel(
        code, env=env, env_type="ssh", task_env_id=task,
        sandbox_tools=tools, timeout=timeout,
        max_tool_calls=5, reset=reset,
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
        self._poll.start()

    def tearDown(self):
        self._ship.stop()
        self._poll.stop()
        shutdown_all_remote_kernels()


class TestSpawnAndReuse(RemoteKernelBase):
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


class TestOwnershipIsolation(RemoteKernelBase):
    def test_changed_tool_set_spawns_kernel_with_fresh_stubs(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, tools=frozenset({"read_file"}))
        _run(env, tools=frozenset({"web_search"}))

        self.assertEqual(len(_REMOTE_KERNELS), 2)
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)
        keyed_tool_sets = {key[-1] for key in _REMOTE_KERNELS}
        self.assertEqual(
            keyed_tool_sets,
            {("read_file",), ("web_search",)},
        )

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


class TestConcurrentCellsSerialize(RemoteKernelBase):
    """Cells on one remote kernel must serialize exactly like the local path
    (SessionKernel.lock): concurrent execute_code calls in one turn attach to
    the same kernel, and without a per-kernel lock they race cell_seq minting
    (same cell_req_NNNNNN.json, one request clobbered, both pollers on one
    cell_res file -> loser times out and the kernel is killed) and let one
    cell's `rm -f req_* res_*` cleanup delete a sibling's in-flight tool RPC."""

    def _kernel_dir_of(self, command):
        import re
        match = re.search(r"hermes_rkernel_\w+", command)
        return match.group(0) if match else None

    def _await_poll(self, env, needle="cell_res_"):
        """Spin until the env sees a cell_res poll command; fail rather than
        hang the suite if a regression stops the first cell from polling."""
        deadline = time.monotonic() + 10
        while not any(needle in c for c in env.commands):
            self.assertLess(time.monotonic(), deadline, "first cell never polled")
            time.sleep(0.005)

    def test_second_cell_waits_for_first_to_settle(self):
        import threading

        gate = threading.Event()
        dispatched = []  # cell seqs whose request ship started
        finished = []

        def cat_handler(command):
            # Cell 1's poll stalls until released; cell 2 returns at once.
            if "cell_res_000001" in command:
                gate.wait(10)
            return {"output": json.dumps(_cell()), "returncode": 0}

        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", cat_handler),
        ])

        def ship_spy(_env, path, _content):
            dispatched.append(path)
        self._ship.stop()
        self._ship = patch(
            "tools.code_execution_tool._ship_file_to_remote", side_effect=ship_spy,
        )
        self._ship.start()

        t1 = threading.Thread(target=lambda: finished.append(_run(env)))
        t1.start()
        # Wait until cell 1 is mid-flight: its res_000001 poll is parked on the gate.
        self._await_poll(env)
        t2 = threading.Thread(target=lambda: finished.append(_run(env)))
        t2.start()
        time.sleep(0.3)
        # Serialization: while cell 1 is still blocked, cell 2 must not have
        # shipped its request file (or finished) yet.
        self.assertTrue(t2.is_alive(), "second cell ran concurrently with the first")
        self.assertEqual(
            len([p for p in dispatched if "cell_req_" in p]), 1,
            "queued cell shipped its request while the first was still running")
        gate.set()
        t1.join(10)
        t2.join(10)
        self.assertFalse(t1.is_alive() or t2.is_alive())
        # BOTH cells ran on the kernel, with DISTINCT request seqs — a queued
        # cell that wrongly fell open would ship nothing and pass a mere
        # distinctness check.
        self.assertEqual(len(finished), 2)
        for result in finished:
            self.assertEqual(result["status"], "success", result)
        reqs = [p for p in dispatched if "cell_req_" in p]
        self.assertEqual(len(reqs), 2)
        self.assertEqual(len(set(reqs)), 2)

    def test_cell_queued_on_discarded_kernel_respawns(self):
        """A cell that queued behind one which timed out must not run on the
        discarded (killed) kernel and must not silently degrade to stateless
        per-call either — it re-acquires, respawns a fresh kernel, and runs
        there (the "next call starts a fresh kernel" note made true early)."""
        import threading

        dispatched = []
        dirs = []  # kernel dirs in first-seen order; dir 0 is the kernel under test

        def cat_handler(command):
            d = self._kernel_dir_of(command)
            if d and d not in dirs:
                dirs.append(d)
            # Kernel 1's cell never gets a result (times out and kills it); a
            # respawned kernel's res answers instantly.
            if dirs and d == dirs[0]:
                return {"output": "", "returncode": 0}
            return {"output": json.dumps(_cell()), "returncode": 0}

        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", cat_handler),
        ])

        def ship_spy(_env, path, _content):
            dispatched.append(path)
        self._ship.stop()
        self._ship = patch(
            "tools.code_execution_tool._ship_file_to_remote", side_effect=ship_spy,
        )
        self._ship.start()

        done = {}
        t1 = threading.Thread(
            target=lambda: done.__setitem__(1, _run(env, timeout=2)))
        t1.start()
        self._await_poll(env)
        t2 = threading.Thread(
            target=lambda: done.__setitem__(2, _run(env)))
        t2.start()
        t1.join(15)
        t2.join(15)
        self.assertFalse(t1.is_alive() or t2.is_alive())
        self.assertEqual(done[1]["status"], "timeout", done[1])
        # The queued cell saw the dead registry entry, respawned, and ran on a
        # FRESH kernel — not the dead dir, not a stateless per-call run.
        self.assertEqual(done[2]["status"], "success", done[2])
        self.assertFalse(done[2]["kernel"]["reused"])
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)
        reqs = [p for p in dispatched if "cell_req_" in p]
        self.assertEqual(len(reqs), 2)
        self.assertNotEqual(
            self._kernel_dir_of(reqs[0]), self._kernel_dir_of(reqs[1]),
            "queued cell must not ship a request to a discarded kernel")

    def test_queued_cell_on_remote_dead_kernel_respawns(self):
        """A kernel that dies remote-side while still registered (OOM,
        container restart, runner self-exit) passes the registry-membership
        check — the post-lock liveness re-probe must catch it, pop the corpse,
        and respawn rather than burn the full cell timeout on a dead dir."""
        import threading

        gate = threading.Event()
        dispatched, dirs = [], []
        dead = {"v": False}
        pids = iter(("4242", "5555"))

        def cat_handler(command):
            d = self._kernel_dir_of(command)
            if d and d not in dirs:
                dirs.append(d)
            if dirs and d == dirs[0]:
                gate.wait(10)
            return {"output": json.dumps(_cell()), "returncode": 0}

        def liveness(command):
            if "4242" in command and dead["v"]:
                return {"output": "", "returncode": 1}
            return {"output": "ALIVE\n", "returncode": 0}

        env = ScriptedEnv([
            ("nohup", lambda c: {"output": f"PID:{next(pids)}\n", "returncode": 0}),
            ("kill -0", liveness),
            ("cat ", cat_handler),
        ])

        def ship_spy(_env, path, _content):
            dispatched.append(path)
        self._ship.stop()
        self._ship = patch(
            "tools.code_execution_tool._ship_file_to_remote", side_effect=ship_spy,
        )
        self._ship.start()

        done = {}
        t1 = threading.Thread(target=lambda: done.__setitem__(1, _run(env)))
        t1.start()
        self._await_poll(env)
        kernel1 = next(iter(_REMOTE_KERNELS.values()))
        t2 = threading.Thread(target=lambda: done.__setitem__(2, _run(env)))
        t2.start()
        # Wait until t2 is attached and queued on kernel.lock, then mark the
        # runner dead remote-side and let cell 1 finish.
        deadline = time.monotonic() + 10
        while kernel1.attached < 2:
            self.assertLess(time.monotonic(), deadline, "second cell never queued")
            time.sleep(0.005)
        dead["v"] = True
        gate.set()
        t1.join(15)
        t2.join(15)
        self.assertFalse(t1.is_alive() or t2.is_alive())
        self.assertEqual(done[1]["status"], "success", done[1])
        # The queued cell re-probed, found the corpse, popped it, and respawned.
        self.assertEqual(done[2]["status"], "success", done[2])
        self.assertFalse(done[2]["kernel"]["reused"])
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)
        reqs = [p for p in dispatched if "cell_req_" in p]
        self.assertEqual(len(reqs), 2)
        self.assertNotEqual(
            self._kernel_dir_of(reqs[0]), self._kernel_dir_of(reqs[1]))
        # The dead kernel's dir was cleaned up by the last attached cell out.
        self.assertTrue(
            any("rm -rf" in c and dirs[0] in c for c in env.commands))

    def test_reset_during_running_cell_defers_teardown(self):
        """A concurrent reset must not kill a kernel under a running cell: the
        entry is popped (so the reset spawns fresh) but teardown defers to the
        last attached cell out — the local-kernel rule (hermes-agent#101861),
        missing remotely until now."""
        import threading

        gate = threading.Event()
        dirs = []

        def cat_handler(command):
            d = self._kernel_dir_of(command)
            if d and d not in dirs:
                dirs.append(d)
            if dirs and d == dirs[0] and "cell_res_" in command:
                gate.wait(10)
            return {"output": json.dumps(_cell()), "returncode": 0}

        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", cat_handler),
        ])

        done = {}
        t1 = threading.Thread(target=lambda: done.__setitem__(1, _run(env)))
        t1.start()
        self._await_poll(env)

        # reset=True pops the live kernel and runs on a fresh one, but must not
        # kill the old dir while cell 1 still polls it.
        result = _run(env, reset=True)
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(result["kernel"].get("state_reset"))
        self.assertFalse(
            any("rm -rf" in c and dirs[0] in c for c in env.commands),
            "reset killed the kernel dir under a running cell")
        self.assertFalse(
            any("pkill -TERM -P" in c for c in env.commands),
            "reset killed the runner under a running cell")

        gate.set()
        t1.join(10)
        self.assertFalse(t1.is_alive())
        self.assertEqual(done[1]["status"], "success", done[1])
        # The last attached cell out owns the teardown: dir 1 is removed now.
        self.assertTrue(
            any("rm -rf" in c and dirs[0] in c for c in env.commands),
            "orphaned kernel was never torn down")

    def test_queue_wait_beyond_own_timeout_falls_open(self):
        """A queued cell waits on kernel.lock only up to its own cell budget;
        past that it fails open to per-call instead of doubling latency."""
        env = ScriptedEnv(_spawn_ok_handlers([_cell()]))
        self.assertEqual(_run(env)["status"], "success")
        kernel = next(iter(_REMOTE_KERNELS.values()))
        kernel.lock.acquire()
        try:
            started = time.monotonic()
            self.assertIsNone(_run(env, timeout=1))
            self.assertLess(time.monotonic() - started, 5)
        finally:
            kernel.lock.release()

    def test_parallel_execute_remote_calls_serialize_e2e(self):
        """End-to-end through _execute_remote: two concurrent calls resolve one
        kernel and must serialize on it, not interleave cell dispatch."""
        import threading
        from tools.code_execution_tool import _execute_remote

        gate = threading.Event()
        dispatched, results = [], []

        def cat_handler(command):
            if "cell_res_000001" in command:
                gate.wait(10)
            return {"output": json.dumps(_cell()), "returncode": 0}

        env = ScriptedEnv([
            ("command -v python3", lambda c: {"output": "OK\n", "returncode": 0}),
            ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", cat_handler),
        ])

        def ship_spy(_env, path, _content):
            dispatched.append(path)
        self._ship.stop()
        self._ship = patch(
            "tools.code_execution_tool._ship_file_to_remote", side_effect=ship_spy,
        )
        self._ship.start()

        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")):
            t1 = threading.Thread(
                target=lambda: results.append(_execute_remote("print(1)", "t1", ["read_file"])))
            t1.start()
            self._await_poll(env)
            t2 = threading.Thread(
                target=lambda: results.append(_execute_remote("print(2)", "t1", ["read_file"])))
            t2.start()
            time.sleep(0.3)
            self.assertTrue(t2.is_alive(),
                            "second execute_code ran concurrently on one kernel")
            gate.set()
            t1.join(10)
            t2.join(10)
        self.assertFalse(t1.is_alive() or t2.is_alive())
        # Both calls ran ON THE KERNEL (kernel provenance in the result), not
        # silently on the stateless per-call path.
        self.assertEqual(len(results), 2)
        for r in results:
            parsed = json.loads(r)
            self.assertEqual(parsed["status"], "success", r)
            self.assertTrue(parsed.get("kernel"), r)
        reqs = [p for p in dispatched if "cell_req_" in p]
        self.assertEqual(len(reqs), 2)
        self.assertEqual(len(set(reqs)), 2)


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
            # Snapshot: the worker thread inserts into the registry concurrently and a live
            # dict iteration raises "dictionary changed size during iteration".
            while not any(k.attached for k in list(_REMOTE_KERNELS.values())):
                time.sleep(0.005)
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
                   return_value=fake):
            result = json.loads(_execute_remote("print()", "t", ["read_file"]))
        self.assertEqual(result["status"], "success")
        self.assertIn("kernel says hi", result["output"])
        self.assertEqual(result["kernel"]["execution_count"], 3)

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
