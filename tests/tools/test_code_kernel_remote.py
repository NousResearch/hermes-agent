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
import shutil
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

    _stdin_mode = "pipe"  # contract-faithful: ssh/docker/local deliver stdin

    def __init__(self, handlers):
        self.handlers = handlers
        self.commands = []
        self.stdin_payloads = []

    def get_temp_dir(self):
        return "/tmp"

    def execute(self, command, cwd=None, timeout=None, stdin_data=None):
        self.commands.append(command)
        self.stdin_payloads.append(stdin_data)
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


class TestSharedHostLockdown(RemoteKernelBase):
    """Shared-host hardening: the kernel dir lives under a shared temp dir, so
    every dir must be owner-only, every remote write owner-only, and the RPC
    token must travel in a sourced env file rather than a ps-visible argv."""

    def test_spawn_locks_down_dirs_and_hides_token(self):
        self._ship.stop()  # let the real ship commands reach env.commands
        env = ScriptedEnv(_spawn_ok_handlers([_cell(stdout="hi\n")]))
        result = _run(env)
        self.assertEqual(result["status"], "success", result)
        kernel = next(iter(_REMOTE_KERNELS.values()))
        # The token never rides a command line: a remote shell's argv is
        # world-readable via ps for the command's whole lifetime.
        self.assertFalse(
            any(kernel.rpc_token in c for c in env.commands),
            "rpc token appeared in a remote command line")
        spawn_cmd = next(c for c in env.commands if "nohup" in c)
        # The env file is sourced inside a subshell so set -a's exports never
        # reach the backend's session-snapshot dump (issue #71296 class).
        self.assertIn("( set -a", spawn_cmd)
        self.assertIn(". ./kernel.env", spawn_cmd)
        self.assertIn("rm -f ./kernel.env", spawn_cmd)
        self.assertNotIn("HERMES_RPC_TOKEN=", spawn_cmd)
        # Every dir under the shared temp dir is owner-only (mkdir -p's -m
        # applies only to the leaf, so the chmod must name all three; umask 077
        # covers the creation-time window).
        mkdir_cmd = next(c for c in env.commands if "mkdir -p" in c)
        self.assertIn("umask 077", mkdir_cmd)
        self.assertIn("chmod 700", mkdir_cmd)
        for d in (kernel.kernel_dir, f"{kernel.kernel_dir}/cells",
                  f"{kernel.kernel_dir}/rpc"):
            self.assertIn(d, mkdir_cmd)
        # Ships write owner-only; on a pipe-capable backend the payload rides
        # stdin, so the base64 (which decodes to the token for kernel.env)
        # never enters argv either.
        ship_cmds = [c for c in env.commands if "base64 -d" in c]
        self.assertTrue(ship_cmds)
        self.assertTrue(all("umask 077" in c for c in ship_cmds))
        self.assertTrue(any("kernel.env" in c for c in ship_cmds))
        self.assertFalse(any("echo '" in c for c in ship_cmds),
                         "payload echoed into argv on a stdin-capable backend")
        import base64
        env_ship = next(p for p, c in zip(env.stdin_payloads, env.commands)
                        if p and "kernel.env" in c)
        env_content = base64.b64decode(env_ship).decode()
        self.assertIn(f"HERMES_RPC_TOKEN={kernel.rpc_token}", env_content)

    def test_remote_write_transport_follows_stdin_mode(self):
        """Pipe-mode backends carry the payload on stdin (never argv); heredoc
        backends embed stdin in the command anyway, so they keep the echo pipe."""
        import base64
        from tools.code_execution_rpc import _remote_write_cmd

        class HeredocEnv:
            _stdin_mode = "heredoc"

        cmd, stdin = _remote_write_cmd(HeredocEnv(), "/x/f", "data")
        self.assertIsNone(stdin)
        self.assertIn("echo '", cmd)
        cmd2, stdin2 = _remote_write_cmd(ScriptedEnv([]), "/x/f", "data")
        self.assertIsNotNone(stdin2)
        self.assertNotIn("echo '", cmd2)
        self.assertEqual(base64.b64decode(stdin2).decode(), "data")

    def test_stub_req_files_are_owner_only(self):
        """The generated file-RPC stub writes req files mode 600: they carry
        the token + tool args and sit in a dir a same-uid kernel shares."""
        import tempfile
        import types
        from tools.code_execution_tool import generate_hermes_tools_module
        src = generate_hermes_tools_module(["read_file"], transport="file")
        rpc_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, rpc_dir, True)
        with patch.dict(os.environ, {"HERMES_RPC_DIR": rpc_dir,
                                     "HERMES_RPC_TOKEN": "test-token"}):
            mod = types.ModuleType("hermes_tools")
            exec(compile(src, "hermes_tools.py", "exec"), mod.__dict__)
            with open(os.path.join(rpc_dir, "res_000001"), "w") as f:
                f.write(json.dumps("ok"))
            mod.read_file("/etc/hostname")
            req = os.path.join(rpc_dir, "req_000001")
            self.assertTrue(os.path.exists(req))
            self.assertEqual(os.stat(req).st_mode & 0o777, 0o600)

    def test_runner_cell_res_files_are_owner_only(self):
        """The remote runner writes cell_res files mode 600: they carry the
        cell's output in a dir under shared temp."""
        import tempfile
        import threading
        import types
        from tools.code_kernel import RUNNER_CELL_SOURCE
        from tools.code_kernel_remote import REMOTE_KERNEL_RUNNER_SOURCE
        kdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, kdir, True)
        cells = os.path.join(kdir, "cells")
        os.makedirs(cells)
        src = REMOTE_KERNEL_RUNNER_SOURCE.format(
            cell_source=RUNNER_CELL_SOURCE, capture_limit=10000, idle_exit=2)
        with patch.dict(os.environ, {"HERMES_KERNEL_DIR": kdir,
                                     "HERMES_RPC_DIR": f"{kdir}/rpc",
                                     "HERMES_RPC_TOKEN": "t"}):
            mod = types.ModuleType("kernel_runner")
            exec(compile(src, "kernel_runner.py", "exec"), mod.__dict__)
            with open(os.path.join(cells, "cell_req_000001.json"), "w") as f:
                json.dump({"code": "print('hi')", "id": "000001"}, f)
            runner = threading.Thread(target=mod.main, daemon=True)
            runner.start()
            res = os.path.join(cells, "cell_res_000001.json")
            deadline = time.monotonic() + 10
            while not os.path.exists(res):
                self.assertLess(time.monotonic(), deadline,
                                "runner never wrote the cell result")
                time.sleep(0.02)
            self.assertEqual(os.stat(res).st_mode & 0o777, 0o600)
            runner.join(5)


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"),
                     "needs a POSIX bash transport")
class TestRemoteKernelLocalEnvE2E(RemoteKernelBase):
    """Real end-to-end: spawn the remote kernel through LocalEnvironment's real
    bash transport, run a cell, and stat the modes on disk. This is the same
    code path an ssh backend drives, pointed at this host."""

    def test_kernel_tree_is_owner_only_on_real_fs(self):
        self._ship.stop()
        self._poll.stop()
        from tools.environments.local import LocalEnvironment
        env = LocalEnvironment(cwd="/", timeout=60)
        try:
            result = _run(env, code=(
                "import os\n"
                "print('KDIR_MODE=%o' % (os.stat(os.environ['HERMES_KERNEL_DIR']).st_mode & 0o777))\n"
                "print('RPC_MODE=%o' % (os.stat(os.environ['HERMES_RPC_DIR']).st_mode & 0o777))\n"
                "print('PP_DELIVERED=%s' % (os.environ.get('PYTHONPATH') == os.environ['HERMES_KERNEL_DIR']))\n"
                "print('TOKEN_DELIVERED=%s' % bool(os.environ.get('HERMES_RPC_TOKEN')))\n"
            ), timeout=60)
            self.assertEqual(result["status"], "success", result)
            blob = json.dumps(result)
            self.assertIn("KDIR_MODE=700", blob)
            self.assertIn("RPC_MODE=700", blob)
            # The env file must actually deliver its vars to the runner — a
            # broken source would silently degrade cells to no-RPC.
            self.assertIn("PP_DELIVERED=True", blob)
            self.assertIn("TOKEN_DELIVERED=True", blob)
            kernel = next(iter(_REMOTE_KERNELS.values()))
            # kernel.env is consumed by the subshell source: the token file is
            # gone after launch while the runner keeps the values in its env.
            self.assertFalse(os.path.exists(
                os.path.join(kernel.kernel_dir, "kernel.env")))
            for name in ("kernel_runner.py", "hermes_tools.py", "runner.log"):
                p = os.path.join(kernel.kernel_dir, name)
                self.assertTrue(os.path.exists(p), p)
                self.assertEqual(os.stat(p).st_mode & 0o777, 0o600, p)
            # The subshell confinement keeps the token and execution-scoped vars
            # out of the backend's session snapshot: they must not leak into the
            # snapshot file or the environ of a later command on the same env
            # (issue #71296 snapshot-leak class).
            probe = env.execute(
                "printenv HERMES_RPC_TOKEN; printenv HERMES_KERNEL_DIR; "
                "printenv HERMES_RPC_DIR; printenv PYTHONPATH",
                cwd="/", timeout=15)
            self.assertNotIn(kernel.rpc_token, probe.get("output", ""))
            self.assertNotIn(kernel.kernel_dir, probe.get("output", ""))
            self.assertNotIn(
                kernel.rpc_token,
                open(env._snapshot_path).read() if os.path.exists(env._snapshot_path) else "")
        finally:
            shutdown_all_remote_kernels()

    def test_per_call_sandbox_is_owner_only_on_real_fs(self):
        self._ship.stop()
        self._poll.stop()
        from tools.environments.local import LocalEnvironment
        from tools.code_execution_tool import _run_remote_per_call
        env = LocalEnvironment(cwd="/", timeout=60)
        code = (
            "import os\n"
            "print('SANDBOX_MODE=%o' % (os.stat(os.path.dirname(os.environ['HERMES_RPC_DIR'])).st_mode & 0o777))\n"
            "print('RPC_MODE=%o' % (os.stat(os.environ['HERMES_RPC_DIR']).st_mode & 0o777))\n"
            "print('SCRIPT_MODE=%o' % (os.stat('script.py').st_mode & 0o777))\n"
            "print('TOOLS_MODE=%o' % (os.stat('hermes_tools.py').st_mode & 0o777))\n"
            "print('ENVFILE_MODE=%o' % (os.stat('sandbox.env').st_mode & 0o777))\n"
        )
        out = json.loads(_run_remote_per_call(
            env, "local", code, "t-e2e", frozenset({"read_file"}),
            timeout=60, max_tool_calls=5, exec_start=time.monotonic()))
        self.assertEqual(out["status"], "success", out)
        self.assertIn("SANDBOX_MODE=700", out["output"])
        self.assertIn("RPC_MODE=700", out["output"])
        self.assertIn("SCRIPT_MODE=600", out["output"])
        self.assertIn("TOOLS_MODE=600", out["output"])
        self.assertIn("ENVFILE_MODE=600", out["output"])


if __name__ == "__main__":
    unittest.main()
