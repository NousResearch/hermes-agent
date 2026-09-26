#!/usr/bin/env python3
"""Tests for execute_code's session kernel.

Session kernels are always on (the ``code_execution.kernel_mode`` key is
retired): each (task, mode, interpreter, cwd, tool-set) owner keeps one
Python child alive so state survives across calls. These tests pin the
contract:

  - state persists across cells and reset=true discards it
  - a raised exception keeps the kernel (and its state) alive
  - a timeout kills the kernel; the next call gets a fresh one
  - fd-level output from user-spawned subprocesses reaches the result
  - sys.exit() inside a cell ends the kernel deliberately

Mode is sourced from ``code_execution.mode`` in config.yaml only;
tests patch ``_load_config`` directly, mirroring test_code_execution_modes.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Mirror test_code_execution.py — guarantee local backend."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


from tools.code_execution_tool import execute_code
from tools.code_kernel import _KERNELS, shutdown_all_kernels


@contextmanager
def _kernel_config(**overrides):
    """Pin code_execution config; strict mode keeps the test hermetic.
    ``mode`` (strict/project) is the only config knob — session kernels are
    always on; the retired ``kernel_mode`` key is ignored by the tool."""
    config = {"mode": "strict", "timeout": 30}
    config.update(overrides)
    with patch("tools.code_execution_tool._load_config", return_value=config):
        yield


@pytest.fixture(autouse=True)
def _fresh_kernel_registry():
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def _run(code, **kwargs):
    return json.loads(execute_code(code, task_id="kernel-test", **kwargs))


class TestSessionStatePersistence(unittest.TestCase):
    def test_state_persists_across_cells(self):
        with _kernel_config():
            first = _run("x = 41")
            self.assertEqual(first["status"], "success", first)
            self.assertEqual(first["kernel"]["reused"], False)
            second = _run("print(x + 1)")
        self.assertEqual(second["status"], "success", second)
        self.assertIn("42", second["output"])
        self.assertEqual(second["kernel"]["reused"], True)
        self.assertEqual(second["kernel"]["execution_count"], 2)

    def test_reset_discards_state(self):
        with _kernel_config():
            _run("x = 41")
            second = _run("print(x + 1)", reset=True)
        self.assertEqual(second["status"], "error", second)
        self.assertIn("NameError", second.get("error", ""))
        self.assertEqual(second["kernel"]["state_reset"], True)

    def test_exception_keeps_the_kernel_alive(self):
        with _kernel_config():
            _run("a = 7")
            boom = _run("1 / 0")
            self.assertEqual(boom["status"], "error")
            self.assertIn("ZeroDivisionError", boom["error"])
            after = _run("print(a)")
        self.assertEqual(after["status"], "success", after)
        self.assertIn("7", after["output"])
        self.assertEqual(after["kernel"]["reused"], True)

    def test_imports_persist(self):
        with _kernel_config():
            _run("import json as _j")
            second = _run("print(_j.dumps({'k': 1}))")
        self.assertIn('{"k": 1}', second["output"])


class TestKernelLifecycle(unittest.TestCase):
    def test_kernel_exits_when_its_backend_parent_dies(self):
        """A kernel must not outlive the host that spawned it, even when the
        host dies without cleanup (SIGKILL/OOM/crash). Windows: inherited
        SYNCHRONIZE handle; POSIX: inherited death pipe. Both are proven the
        same way — kill the host mid-cell, the kernel is gone within seconds."""
        import psutil

        repo_root = str(Path(__file__).resolve().parents[2])
        host_src = textwrap.dedent(f"""
            import json, os, sys, time
            os.environ["HERMES_HOME"] = sys.argv[1]
            sys.path.insert(0, {repo_root!r})
            from tools.code_kernel import SessionKernel, _spawn
            k = SessionKernel(("parent-death",))
            _spawn(k, task_id="parent-death", child_python=sys.executable,
                   child_cwd="", sandbox_tools=frozenset(), max_tool_calls=1)
            cell = json.dumps({{"id": "x", "code": "import os, time\\n"
                "assert 'HERMES_KERNEL_PARENT_PROCESS_HANDLE' not in os.environ\\n"
                "assert 'HERMES_KERNEL_PARENT_DEATH_FD' not in os.environ\\n"
                "time.sleep(300)"}}) + "\\n"
            k.proc.stdin.write(cell.encode()); k.proc.stdin.flush()
            print(k.proc.pid, flush=True)
            time.sleep(600)
        """)
        with tempfile.TemporaryDirectory() as home:
            host = subprocess.Popen(
                [sys.executable, "-c", host_src, home],
                stdout=subprocess.PIPE, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                kernel = psutil.Process(int(host.stdout.readline()))
                time.sleep(0.5)
                self.assertTrue(kernel.is_running(), "kernel never came up")
                host.kill()
                host.wait(timeout=10)
                try:
                    kernel.wait(timeout=10)
                except psutil.TimeoutExpired:
                    kernel.kill()
                    self.fail("session kernel survived its backend parent")
            finally:
                if host.poll() is None:
                    host.kill()

    def test_timeout_kills_the_kernel_and_reports_state_loss(self):
        with _kernel_config(timeout=1):
            slow = _run("import time\ntime.sleep(30)")
            self.assertEqual(slow["status"], "timeout", slow)
            self.assertIn("state was lost", slow["error"])
        self.assertEqual(len(_KERNELS), 0)
        with _kernel_config():
            fresh = _run("print('alive')")
        self.assertEqual(fresh["status"], "success", fresh)
        self.assertEqual(fresh["kernel"]["reused"], False)
        self.assertIn("alive", fresh["output"])

    def test_sys_exit_ends_the_kernel(self):
        with _kernel_config():
            done = _run("import sys\nsys.exit(0)")
            self.assertEqual(done["kernel"].get("ended"), True, done)
            self.assertEqual(len(_KERNELS), 0)
            fresh = _run("print('respawned')")
        self.assertEqual(fresh["kernel"]["reused"], False)
        self.assertIn("respawned", fresh["output"])

    def test_subprocess_fd_output_reaches_the_result(self):
        code = (
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, '-c', \"print('raw-passthrough')\"])\n"
        )
        with _kernel_config():
            result = _run(code)
        self.assertEqual(result["status"], "success", result)
        self.assertIn("raw-passthrough", result["output"])


class TestModelFacingReset(unittest.TestCase):
    def test_reset_is_reachable_from_a_model_call_despite_stale_kernel_mode(self):
        """Session kernels are always on (#96787), so ``reset`` is the model's only
        way out of poisoned state. A stale ``kernel_mode: per-call`` key must not
        drop it from the schema, and a model-shaped call routed through the
        registered handler must actually discard the kernel's state."""
        from tools.code_execution_tool import _execute_code_handler, build_execute_code_schema

        with _kernel_config(kernel_mode="per-call"):
            schema = build_execute_code_schema(mode="strict")
            self.assertEqual(schema["parameters"]["properties"]["reset"]["type"], "boolean")
            _execute_code_handler({"code": "x = 41"}, task_id="kernel-test")
            kept = json.loads(_execute_code_handler({"code": "print(x + 1)"}, task_id="kernel-test"))
            self.assertIn("42", kept["output"], kept)
            reset = json.loads(_execute_code_handler(
                {"code": "print(x + 1)", "reset": True}, task_id="kernel-test"))
        self.assertEqual(reset["status"], "error", reset)
        self.assertIn("NameError", reset.get("error", ""))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestKernelOwnershipAndLifecycle(unittest.TestCase):
    """The kernel belongs to the conversation, and its lifetime is bounded.

    run_agent mints a fresh task id per top-level turn, so a task-keyed
    kernel would neither survive the next user turn nor ever be disposed
    with anything. The owner is the approval session key; disposal rides
    the same session boundary that clears approval/yolo state, idle
    kernels are reaped, and the process-wide live count is capped (the
    lifecycle shape carried forward from hermes-agent#88637).
    """

    def _run_as(self, session_key, code, task_id, **kwargs):
        from tools.approval_context import reset_current_session_key, set_current_session_key

        token = set_current_session_key(session_key)
        try:
            return json.loads(execute_code(code, task_id=task_id, **kwargs))
        finally:
            reset_current_session_key(token)

    def test_state_survives_across_turns_of_one_conversation(self):
        # Two top-level turns: same session, different per-turn task ids.
        with _kernel_config():
            first = self._run_as("conv-a", "x = 41", task_id="turn-1")
            self.assertEqual(first["status"], "success", first)
            second = self._run_as("conv-a", "print(x + 1)", task_id="turn-2")
        self.assertEqual(second["status"], "success", second)
        self.assertIn("42", second["output"])
        self.assertEqual(second["kernel"]["reused"], True)

    def test_sessions_are_isolated_from_each_other(self):
        # Same task id, different sessions: no state may cross.
        with _kernel_config():
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            other = self._run_as("conv-b", "print(x + 1)", task_id="turn-1")
        self.assertEqual(other["status"], "error", other)
        self.assertIn("NameError", other.get("error", ""))

    def test_delegated_children_get_their_own_kernels(self):
        """A delegated child runs in a COPY of the parent's context and
        inherits the parent's approval session key — the naive owner
        resolution attached the child to the parent's kernel and leaked
        in-memory state across the delegation boundary (both directions,
        verified live). The owner must be qualified for child contexts."""
        from agent.delegation_context import delegated_child_context

        with _kernel_config():
            self._run_as("conv-a", "parent_secret = 'p'", task_id="turn-1")
            with delegated_child_context("child-1"):
                leak = self._run_as(
                    "conv-a",
                    "print(globals().get('parent_secret', 'ISOLATED'))",
                    task_id="child-task",
                )
                self._run_as("conv-a", "child_secret = 'c'", task_id="child-task")
            back = self._run_as(
                "conv-a",
                "print(globals().get('child_secret', 'ISOLATED'))",
                task_id="turn-2",
            )
        self.assertIn("ISOLATED", leak.get("output", ""), leak)
        self.assertIn("ISOLATED", back.get("output", ""), back)

    def test_two_delegated_children_are_isolated_from_each_other(self):
        """Sibling children in one batch must not share a kernel either —
        each child context carries its own delegation session id."""
        from agent.delegation_context import delegated_child_context

        with _kernel_config():
            with delegated_child_context("child-A"):
                self._run_as("conv-a", "sibling_secret = 'A'", task_id="t")
            with delegated_child_context("child-B"):
                peek = self._run_as(
                    "conv-a",
                    "print(globals().get('sibling_secret', 'ISOLATED'))",
                    task_id="t",
                )
        self.assertIn("ISOLATED", peek.get("output", ""), peek)

    def test_live_children_keep_their_kernels_past_the_lru_cap(self):
        """A fan-out wider than max_session_kernels used to evict LIVE children's kernels (each
        child's execute_code spawned a kernel, the cap reaped the oldest sibling's), so a child's
        second call hit NameError on state its first call had set — 48 NameErrors across 28 lanes,
        while the schema promised persistence. A live child's kernel is pinned for the child's life."""
        import contextvars

        from agent.delegation_context import delegated_child_context

        with _kernel_config(max_session_kernels=2):
            contexts = []
            for index in range(5):
                def _set(index=index):
                    with delegated_child_context(f"child-{index}"):
                        self._run_as("conv", f"v = {index}", task_id=f"child-{index}")
                ctx = contextvars.copy_context()
                ctx.run(_set)
                contexts.append(ctx)
            outcomes = {}
            for index, ctx in enumerate(contexts):
                def _read(index=index):
                    with delegated_child_context(f"child-{index}"):
                        outcomes[index] = self._run_as("conv", "print(v)", task_id=f"child-{index}")
                ctx.run(_read)
        for index, outcome in outcomes.items():
            self.assertEqual(outcome["status"], "success", outcome)
            self.assertTrue(outcome["kernel"]["reused"], outcome)
            self.assertIn(str(index), outcome["output"])

    def test_finished_children_release_their_kernels(self):
        """The pin is not a leak: when the child is torn down (the delegate_task cleanup path calls
        ``shutdown_kernels_for_delegated_child``) its kernels die and stop counting."""
        from agent.delegation_context import delegated_child_context
        from tools.code_kernel import shutdown_kernels_for_delegated_child

        with _kernel_config():
            with delegated_child_context("child-done"):
                self._run_as("conv", "v = 1", task_id="child-done")
            with delegated_child_context("child-live"):
                self._run_as("conv", "v = 2", task_id="child-live")
            doomed = [k for k in _KERNELS.values() if k.owner.endswith("::child::child-done")]
            self.assertEqual(len(doomed), 1)
            shutdown_kernels_for_delegated_child("child-done")
            self.assertEqual([k for k in _KERNELS.values() if k.owner.endswith("::child::child-done")], [])
            doomed[0].proc.wait(timeout=10)
            self.assertFalse(doomed[0].alive())
            # The sibling's kernel is untouched.
            with delegated_child_context("child-live"):
                still = self._run_as("conv", "print(v)", task_id="child-live")
        self.assertIn("2", still["output"])

    def test_session_clear_disposes_the_owners_kernels(self):
        from tools.approval import clear_session

        with _kernel_config():
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            self.assertEqual(len(_KERNELS), 1)
            kernel = next(iter(_KERNELS.values()))
            self.assertTrue(kernel.alive())
            clear_session("conv-a")
            self.assertEqual(len(_KERNELS), 0)
            kernel.proc.wait(timeout=10)
            self.assertFalse(kernel.alive())
            # The next turn in a cleared session starts fresh.
            after = self._run_as("conv-a", "print('x' in dir())", task_id="turn-2")
        self.assertEqual(after["status"], "success", after)
        self.assertIn("False", after["output"])

    def test_live_kernels_are_capped_lru_across_owners(self):
        with _kernel_config(max_session_kernels=2):
            kernels = []
            for index in range(4):
                self._run_as(f"conv-{index}", "x = 1", task_id=f"turn-{index}")
                kernels.append(list(_KERNELS.values()))
            self.assertLessEqual(len(_KERNELS), 2)
            live_owners = {key[0] for key in _KERNELS}
            # The two most recently used owners survive.
            self.assertEqual(live_owners, {"conv-2", "conv-3"})
        # Evicted kernels are actually dead, not orphaned.
        evicted = [
            kernel
            for snapshot in kernels
            for kernel in snapshot
            if kernel.key not in _KERNELS
        ]
        for kernel in evicted:
            kernel.proc.wait(timeout=10)
            self.assertFalse(kernel.alive())

    def test_idle_kernels_are_reaped(self):
        import time as time_module

        with _kernel_config(kernel_idle_timeout=1):
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            stale = next(iter(_KERNELS.values()))
            time_module.sleep(1.2)
            # Any owner's next call sweeps expired kernels process-wide.
            self._run_as("conv-b", "y = 1", task_id="turn-2")
            self.assertNotIn(stale.key, _KERNELS)
            stale.proc.wait(timeout=10)
            self.assertFalse(stale.alive())

    def test_parallel_cells_share_one_kernel_process(self):
        """Parallel cells for one owner race the first spawn. Each racer
        used to see proc=None as 'dead', replace the registry entry, and
        orphan the winner's process — 110 live kernels under a 4-capped
        process (Sep 2026). Every kernel process must stay registry-owned.

        Capture actual children so an unregistered spawn cannot hide behind
        the registry count. The owned process must exit on teardown."""
        import threading

        spawned = []
        real_popen = subprocess.Popen

        def _capturing_popen(args, **kwargs):
            proc = real_popen(args, **kwargs)
            spawned.append((proc, list(args)))
            return proc

        results = []
        with patch("tools.code_kernel.subprocess.Popen", side_effect=_capturing_popen):
            with _kernel_config():
                def _cell():
                    results.append(self._run_as("conv-a", "import time; time.sleep(0.3)", task_id="t"))
                threads = [threading.Thread(target=_cell) for _ in range(6)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        self.assertEqual([r["status"] for r in results], ["success"] * 6)
        self.assertEqual(len(_KERNELS), 1)
        runners = [proc for proc, args in spawned
                   if len(args) == 2 and Path(args[1]).name == "hermes_kernel_runner.py"]
        self.assertEqual(len(runners), 1, "parallel cells spawned an unowned kernel")
        kernel = next(iter(_KERNELS.values()))
        self.assertIs(kernel.proc, runners[0])
        self.assertIsNone(kernel.proc.poll())
        shutdown_all_kernels()
        for proc in runners:
            proc.wait(timeout=10)
            self.assertIsNotNone(proc.returncode)


class TestPerCellRpcAuthority(unittest.TestCase):
    """Interpreter state persists across cells; RPC authority must not."""

    def _recorder(self, seen):
        def _handle(tool_name, tool_args, task_id=None):
            from tools.thread_context import _callback_api

            (get_approval, _set_a), *_rest = _callback_api()
            seen.append(
                {
                    "tool": tool_name,
                    "task_id": task_id,
                    "approval_cb": get_approval(),
                }
            )
            return json.dumps({"ok": True})

        return _handle

    def test_a_later_cells_rpc_runs_under_that_cells_authority(self):
        from tools.terminal_tool import set_approval_callback

        seen = []
        cell = "import hermes_tools\nhermes_tools.web_search(query='q')\n"
        with _kernel_config(), patch(
            "model_tools.handle_function_call", new=self._recorder(seen)
        ):
            def cb_one():
                return "one"

            def cb_two():
                return "two"

            set_approval_callback(cb_one)
            try:
                first = _run(cell)
                set_approval_callback(cb_two)
                second = _run(cell)
            finally:
                set_approval_callback(None)
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(second["status"], "success", second)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0]["approval_cb"], cb_one)
        self.assertIs(seen[1]["approval_cb"], cb_two)
        self.assertEqual(seen[0]["task_id"], "kernel-test")

    def test_cross_cell_alias_dispatches_under_the_current_cell(self):
        # Adversarial cross-cell dataflow: a callable captured in cell 1 and
        # invoked by an opaque global name in cell 2 still crosses the RPC
        # boundary — under cell 2's authority, allow-list, and budget — the
        # operative enforcement a per-script static scan cannot provide once
        # state persists (composition contract with the execute-code guard).
        from tools.terminal_tool import set_approval_callback

        seen = []
        with _kernel_config(), patch(
            "model_tools.handle_function_call", new=self._recorder(seen)
        ):
            def cb_one():
                return "one"

            def cb_two():
                return "two"

            set_approval_callback(cb_one)
            try:
                first = _run("import hermes_tools\nalias = hermes_tools.web_search\n")
                set_approval_callback(cb_two)
                second = _run("alias(query='q')\n")
            finally:
                set_approval_callback(None)
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(second["status"], "success", second)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0]["approval_cb"], cb_two)

    def test_a_settled_cells_authority_refuses_dispatch(self):
        from tools.code_kernel import CellAuthority

        authority = CellAuthority("turn-1")
        authority.retire()
        result = authority.dispatch("web_search", {"query": "q"})
        self.assertIn("No active execute_code cell", result)

    def test_each_cell_installs_a_fresh_authority(self):
        with _kernel_config():
            _run("x = 1")
            kernel = next(iter(_KERNELS.values()))
            first_authority = kernel.cell_authority
            self.assertFalse(first_authority.active)
            _run("y = 2")
            self.assertIsNot(kernel.cell_authority, first_authority)
            self.assertFalse(kernel.cell_authority.active)


class TestBackgroundIdleReaper(unittest.TestCase):
    """#117169: the idle sweep must not depend on the next kernel acquire — a host
    that stays alive but wedged (e.g. pids exhaustion fail-closing every tool call)
    never acquires again, so a background reaper reapplies the acquire-path criteria
    on its own schedule, and staging dirs that outlived a dead host are swept by age."""

    def _run_as(self, session_key, code, task_id, **kwargs):
        from tools.approval_context import reset_current_session_key, set_current_session_key

        token = set_current_session_key(session_key)
        try:
            return json.loads(execute_code(code, task_id=task_id, **kwargs))
        finally:
            reset_current_session_key(token)

    def test_reap_once_sweeps_idle_kernels_without_a_new_acquire(self):
        import time as time_module

        from tools.code_kernel import _reap_once

        with _kernel_config(kernel_idle_timeout=1):
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            stale = next(iter(_KERNELS.values()))
            time_module.sleep(1.2)
            # No conv-b acquire here: the reaper pass alone must retire the kernel.
            _reap_once()
            self.assertNotIn(stale.key, _KERNELS)
            stale.proc.wait(timeout=10)
            self.assertFalse(stale.alive())

    def test_reap_once_spares_attached_and_fresh_kernels(self):
        from tools.code_kernel import _reap_once

        with _kernel_config(kernel_idle_timeout=1):
            fresh = self._run_as("conv-fresh", "x = 1", task_id="turn-1")
            self.assertEqual(fresh["status"], "success", fresh)
            kernel = next(iter(_KERNELS.values()))
            kernel.attached += 1  # a cell is mid-flight: reaping must skip it
            try:
                _reap_once()
                self.assertIn(kernel.key, _KERNELS)
                self.assertTrue(kernel.alive())
            finally:
                kernel.attached -= 1

class TestStaleStagingDirSweep(unittest.TestCase):
    def test_week_old_kernel_dirs_go_and_fresh_ones_stay(self):
        import time as time_module

        from tools.code_kernel import _sweep_stale_staging_dirs

        with tempfile.TemporaryDirectory() as tmp:
            with patch("tools.code_kernel.tempfile.gettempdir", return_value=tmp):
                old = Path(tmp, "hermes_kernel_old")
                young = Path(tmp, "hermes_kernel_young")
                bystander = Path(tmp, "unrelated_dir")
                for path in (old, young, bystander):
                    path.mkdir()
                week_and_a_bit = time_module.time() - 8 * 86400
                os.utime(old, (week_and_a_bit, week_and_a_bit))
                removed = _sweep_stale_staging_dirs()
                # Asserted inside the TemporaryDirectory: cleanup would flatten everything.
                self.assertEqual(removed, 1)
                self.assertFalse(old.exists())
                self.assertTrue(young.exists())
                self.assertTrue(bystander.exists())

class TestStdoutReaderFrameBound(unittest.TestCase):
    """A cell can print the frame sentinel; a forged length header must not
    make _stdout_reader buffer an unbounded frame body."""

    def _kernel_with_stream(self, data: bytes):
        import io
        import queue
        from types import SimpleNamespace
        from tools.code_kernel import _BoundedBuffer

        kernel = SimpleNamespace(
            proc=SimpleNamespace(stdout=io.BufferedReader(io.BytesIO(data))),
            sentinel="@@S@@",
            response_q=queue.Queue(),
            raw=_BoundedBuffer(),
        )
        return kernel

    def test_forged_huge_length_is_output_and_next_frame_parses(self):
        from tools.code_kernel import _stdout_reader
        from tools.code_execution_rpc import MAX_RPC_MESSAGE_BYTES

        payload = json.dumps({"status": "ok"}).encode()
        real_frame = b"\n@@S@@ " + str(len(payload)).encode() + b"\n" + payload
        data = (b"\n@@S@@ " + str(MAX_RPC_MESSAGE_BYTES + 1).encode() + b"\n"
                + real_frame)
        kernel = self._kernel_with_stream(data)
        _stdout_reader(kernel)

        got = kernel.response_q.get_nowait()
        self.assertEqual(got, {"status": "ok"})
        self.assertEqual(kernel.response_q.get_nowait(), {"status": "kernel-eof"})
        # The forged header is emitted as cell output, not consumed as a frame.
        self.assertIn(str(MAX_RPC_MESSAGE_BYTES + 1).encode(), kernel.raw.drain().encode())

    def test_negative_length_is_output_and_next_frame_parses(self):
        from tools.code_kernel import _stdout_reader

        payload = json.dumps({"status": "ok"}).encode()
        real_frame = b"\n@@S@@ " + str(len(payload)).encode() + b"\n" + payload
        data = b"\n@@S@@ -1\n" + real_frame
        kernel = self._kernel_with_stream(data)
        _stdout_reader(kernel)

        self.assertEqual(kernel.response_q.get_nowait(), {"status": "ok"})
        self.assertEqual(kernel.response_q.get_nowait(), {"status": "kernel-eof"})
        self.assertIn("-1", kernel.raw.drain())

    def test_zero_length_is_output_and_next_frame_parses(self):
        """length <= 0 is always forged: real frames carry a JSON object."""
        from tools.code_kernel import _stdout_reader

        payload = json.dumps({"status": "ok"}).encode()
        real_frame = b"\n@@S@@ " + str(len(payload)).encode() + b"\n" + payload
        data = b"\n@@S@@ 0\n" + real_frame
        kernel = self._kernel_with_stream(data)
        _stdout_reader(kernel)

        self.assertEqual(kernel.response_q.get_nowait(), {"status": "ok"})
        self.assertEqual(kernel.response_q.get_nowait(), {"status": "kernel-eof"})
        self.assertIn("0", kernel.raw.drain())

    def test_garbage_frame_body_is_output_and_resyncs(self):
        """A real frame is always valid JSON; an unparseable in-cap body is
        forged output, not a kernel-fatal protocol error."""
        from tools.code_kernel import _stdout_reader

        payload = json.dumps({"status": "ok"}).encode()
        real_frame = b"\n@@S@@ " + str(len(payload)).encode() + b"\n" + payload
        data = b"\n@@S@@ 3\nxyz" + real_frame
        kernel = self._kernel_with_stream(data)
        _stdout_reader(kernel)

        self.assertEqual(kernel.response_q.get_nowait(), {"status": "ok"})
        self.assertEqual(kernel.response_q.get_nowait(), {"status": "kernel-eof"})
        self.assertIn("xyz", kernel.raw.drain())

    def test_unterminated_oversized_header_is_bounded(self):
        """Marker plus a long newline-free tail is output, not a pending
        header: the bytes must reach raw output while the stream stays open
        and silent, not accumulate inside the parser's buffer."""
        import io
        import queue
        import threading
        from types import SimpleNamespace
        from tools.code_kernel import _BoundedBuffer, _stdout_reader

        release = threading.Event()

        class _Drip(io.RawIOBase):
            def __init__(self):
                self._sent = False

            def readable(self):
                return True

            def readinto(self, buf):
                if not self._sent:
                    self._sent = True
                    data = b"\n@@S@@ " + b"9" * 200
                    buf[:len(data)] = data
                    return len(data)
                release.wait(10)
                return 0

        kernel = SimpleNamespace(
            proc=SimpleNamespace(stdout=io.BufferedReader(_Drip())),
            sentinel="@@S@@", response_q=queue.Queue(), raw=_BoundedBuffer())
        t = threading.Thread(target=_stdout_reader, args=(kernel,), daemon=True)
        t.start()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if b"9" * 100 in b"".join(kernel.raw.chunks):
                    break
                time.sleep(0.05)
            self.assertIn(b"9" * 100, b"".join(kernel.raw.chunks))
        finally:
            release.set()
            t.join(timeout=5)

class TestAwaitCellCorrelation(unittest.TestCase):
    """Reply frames carry the request's id; stale or forged frames sitting in
    the queue must not be returned as the current cell's result."""

    def test_stale_and_forged_frames_are_skipped(self):
        import queue
        from types import SimpleNamespace
        from tools.code_kernel import _await_cell

        kernel = SimpleNamespace(response_q=queue.Queue())
        kernel.response_q.put({"id": "forged", "status": "ok", "stdout": "FORGED"})
        kernel.response_q.put({"id": "req-1", "status": "ok", "stdout": "REAL"})
        status, payload = _await_cell(kernel, timeout=5, is_interrupted=lambda: False,
                                      request_id="req-1")
        self.assertEqual(status, "success")
        self.assertEqual(payload["stdout"], "REAL")

    def test_host_signals_still_return_immediately(self):
        import queue
        from types import SimpleNamespace
        from tools.code_kernel import _await_cell

        kernel = SimpleNamespace(response_q=queue.Queue())
        kernel.response_q.put({"status": "kernel-eof"})
        status, payload = _await_cell(kernel, timeout=5, is_interrupted=lambda: False,
                                      request_id="req-1")
        self.assertEqual(status, "error")
        self.assertEqual(payload["status"], "kernel-eof")

class TestRpcBoundsE2E(unittest.TestCase):
    """Real-kernel e2e for the RPC bounds: kernel cells use the socket RPC
    server for tool calls and the framed stdout protocol for replies."""

    def test_oversized_tool_request_is_rejected_and_kernel_survives(self):
        with _kernel_config(timeout=30):
            from tools.code_execution_rpc import MAX_RPC_MESSAGE_BYTES
            with patch("model_tools.handle_function_call",
                       side_effect=lambda name, args, **kw: json.dumps({"dispatched": name})):
                code = (
                    "from hermes_tools import terminal\n"
                    f"r = terminal('x' * {MAX_RPC_MESSAGE_BYTES + 1024 * 1024})\n"
                    "print('ERR:' + str(r.get('error', 'none'))[:80])\n"
                )
                out = _run(code)
                self.assertEqual(out["status"], "success", out)
                self.assertIn("size limit", out["output"])
                # The kernel's RPC server keeps serving: a normal call still works.
                ok = _run("from hermes_tools import terminal\n"
                          "print(terminal('echo hi').get('dispatched'))")
                self.assertEqual(ok["status"], "success", ok)
                self.assertIn("terminal", ok["output"])
                self.assertTrue(ok["kernel"]["reused"])

    def test_forged_frame_header_is_output_and_kernel_survives(self):
        """HERMES_KERNEL_SENTINEL sits in the kernel env, so cell code can forge
        a frame header; the host must bound the claimed length. print() is
        captured into the reply's stdout field, so the forge must write fd 1
        directly to reach the frame channel."""
        with _kernel_config(timeout=30):
            code = (
                "import os\n"
                "s = os.environ['HERMES_KERNEL_SENTINEL']\n"
                "os.write(1, ('\\n' + s + ' 99999999999\\n').encode())\n"
                "print('AFTER')\n"
            )
            forged = _run(code)
            self.assertEqual(forged["status"], "success", forged)
            # Forged header bytes come back as cell output, not a consumed frame.
            self.assertIn("99999999999", forged["output"])
            self.assertIn("AFTER", forged["output"])
            second = _run("print('STILL-ALIVE')")
            self.assertEqual(second["status"], "success", second)
            self.assertIn("STILL-ALIVE", second["output"])
            self.assertTrue(second["kernel"]["reused"])

    def test_forged_complete_frame_cannot_replace_the_result(self):
        """A forged in-cap frame whose id does not match the live request is
        skipped: the cell's real reply is what lands, and the forged payload
        cannot bleed into the next cell either."""
        with _kernel_config(timeout=30):
            code = (
                "import os\n"
                "s = os.environ['HERMES_KERNEL_SENTINEL']\n"
                "body = b'{\"id\": \"forged\", \"status\": \"ok\", \"stdout\": \"FORGED\"}'\n"
                "os.write(1, b'\\n' + s.encode() + b' ' + str(len(body)).encode() + b'\\n' + body)\n"
                "print('REAL')\n"
            )
            out = _run(code)
            self.assertEqual(out["status"], "success", out)
            self.assertIn("REAL", out["output"])
            self.assertNotIn("FORGED", out["output"])
            second = _run("print('STILL-ALIVE')")
            self.assertEqual(second["status"], "success", second)
            self.assertIn("STILL-ALIVE", second["output"])
            self.assertNotIn("FORGED", second["output"])
