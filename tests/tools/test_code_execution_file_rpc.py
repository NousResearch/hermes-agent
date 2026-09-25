"""Generated stubs and the production file poller over real shell/filesystem I/O."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

from tools.code_execution_tool import generate_hermes_tools_module
from tools.code_execution_rpc import _rpc_poll_loop


CALLS = {
    "web_search": {"query": "fixture", "limit": 3},
    "web_extract": {"urls": ["https://example.test"], "char_limit": 3000},
    "read_file": {"path": "reference", "offset": 2, "limit": 4},
    "write_file": {"path": "output", "content": "café", "cross_profile": False},
    "search_files": {"pattern": "x", "target": "files", "path": ".", "file_glob": "*.py",
                     "limit": 4, "offset": 2, "output_mode": "count", "context": 3, "order": "modified"},
    "patch": {"path": "output", "old_string": "old", "new_string": "new", "replace_all": True,
              "mode": "replace", "patch": None, "cross_profile": False},
    "terminal": {"command": "echo fixture", "timeout": 3, "workdir": "/tmp"},
}


@pytest.mark.platforms("posix")
def test_generated_file_rpc_kwargs_correlation_and_authority(tmp_path, monkeypatch):
    from pm.shell import bash
    from tools.registry import registry
    import tools.file_tools  # noqa: F401 - populate schemas
    import tools.web_tools  # noqa: F401
    import tools.terminal_tool  # noqa: F401

    shell = bash()
    assert shell
    rpc = tmp_path / "rpc with spaces"
    rpc.mkdir()
    monkeypatch.delenv("HERMES_RPC_DIR", raising=False)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    namespace = {}
    exec(generate_hermes_tools_module([], transport="file"), namespace)
    assert namespace["_RPC_DIR"] == str(tmp_path / "hermes_rpc")
    assert "terminal" not in namespace
    monkeypatch.setenv("HERMES_RPC_DIR", str(rpc))
    monkeypatch.setenv("HERMES_RPC_TOKEN", "right-token")
    exec(generate_hermes_tools_module(list(CALLS), transport="file"), namespace)
    seen, log, counter = [], [], [0]

    def dispatch(name, args, **kwargs):
        seen.append((name, args.copy()))
        return json.dumps({"name": name, "args": args})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)

    class Shell:
        def execute(self, command, cwd=None, timeout=None):
            result = subprocess.run([shell, "-c", command], cwd=cwd, timeout=timeout,
                                    env=dict(os.environ), stdin=subprocess.DEVNULL, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            return {"output": result.stdout}

    stop = threading.Event()
    budget = len(CALLS) + 8
    poller = threading.Thread(target=_rpc_poll_loop, args=(Shell(), str(rpc), "owner", log, counter,
                               budget, frozenset(CALLS), stop, "right-token"), daemon=True)
    poller.start()
    try:
        # Raw clients bypass stub visibility; neither missing nor wrong token may dispatch.
        for seq, token in [(9001, None), (9002, "wrong-token")]:
            request = {"seq": seq, "tool": "terminal", "args": {"command": "forbidden"}}
            if token is not None:
                request["token"] = token
            path = rpc / f"req_{seq}"
            path.write_text(json.dumps(request), encoding="utf-8")
            deadline = time.monotonic() + 5
            while path.exists() and time.monotonic() < deadline:
                stop.wait(.01)
            assert not path.exists()
            assert not (rpc / f"res_{seq:06d}").exists()
        assert seen == [] and counter == [0]
        for name, args in CALLS.items():
            blocked = {"background", "heartbeat", "pty", "notify", "notify_on_complete", "watch_patterns"} if name == "terminal" else set()
            schema_keys = set(registry.get_entry(name).schema["parameters"]["properties"]) - blocked
            assert schema_keys <= set(args), (name, schema_keys - set(args))
            assert namespace[name](**args) == {"name": name, "args": args}
        assert seen == list(CALLS.items())
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda i: namespace["terminal"](f"tag-{i}", 3, "/tmp"), range(8)))
        assert [r["args"]["command"] for r in results] == [f"tag-{i}" for i in range(8)]
        assert "not available" in namespace["_call"]("unauthorized-tool", {})["error"]
        assert "limit reached" in namespace["terminal"]("over-budget")["error"]
        assert counter == [budget] and len(seen) == len(log) == budget
    finally:
        stop.set()
        poller.join(timeout=10)
        assert not poller.is_alive()


def _local_shell():
    """An env.execute double backed by a real local bash, matching the real contract:
    returns {"output", "returncode"} and never raises on a nonzero exit."""
    from pm.shell import bash

    shell = bash()
    assert shell

    class Shell:
        def execute(self, command, cwd=None, timeout=None):
            result = subprocess.run([shell, "-c", command], cwd=cwd, timeout=timeout,
                                    env=dict(os.environ), stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True)
            return {"output": result.stdout, "returncode": result.returncode}

    return Shell()


def _start_poller(env, rpc, allowed, token="right-token"):
    log, counter, stop = [], [0], threading.Event()
    poller = threading.Thread(target=_rpc_poll_loop, args=(env, str(rpc), "owner", log, counter,
                              10, frozenset(allowed), stop, token), daemon=True)
    poller.start()
    return poller, stop


def _publish_req(rpc, name, request):
    """Atomic publish, matching the shipped stub's tmp+rename so the poller can
    never cat a half-written request."""
    tmp = rpc / (name + ".tmp")
    tmp.write_text(json.dumps(request))
    os.rename(tmp, rpc / name)


@pytest.mark.platforms("posix")
def test_poll_loop_dispatches_once_when_response_write_fails(tmp_path, monkeypatch):
    """A req file whose response cannot be written must not be re-dispatched on
    the next poll cycle: side-effecting calls are at-most-once. The dir is made
    unwritable before the poller starts, so the consume-rm fails (rc!=0, counted)
    and every delivery attempt lands in the pending map."""
    rpc = tmp_path / "rpc"
    rpc.mkdir()
    seen = []

    def dispatch(name, args, **kwargs):
        seen.append(name)
        return json.dumps({"ok": True})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    _publish_req(rpc, "req_000001", {"seq": 1, "token": "right-token",
                                   "tool": "terminal", "args": {"command": "x"}})
    req = rpc / "req_000001"
    os.chmod(rpc, 0o555)                      # reads/listings work, rm and res writes fail
    poller, stop = _start_poller(_local_shell(), rpc, {"terminal"})
    try:
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            stop.wait(.01)
        assert seen == ["terminal"]           # dispatched exactly once
        time.sleep(.6)                        # several more poll cycles
        assert seen == ["terminal"]           # no re-dispatch despite the surviving req
        assert req.exists()                   # rm could not run; suppressed, not re-dispatched
        assert not (rpc / "res_000001").exists()
    finally:
        os.chmod(rpc, 0o755)
        stop.set()
        poller.join(timeout=10)
        assert not poller.is_alive()


@pytest.mark.platforms("posix")
def test_poll_loop_answers_by_filename_seq_not_body_seq(tmp_path, monkeypatch):
    """The requester polls res_<file seq>, so the response name must come from the
    req filename, not the forgeable/malformed seq field in the body. A non-integer
    body seq used to raise after dispatch and leave the req file to re-dispatch."""
    rpc = tmp_path / "rpc"
    rpc.mkdir()
    # A planted res_<digits> DIRECTORY must be swept, not swallow the response via mv.
    (rpc / "res_000007").mkdir()
    seen = []

    def dispatch(name, args, **kwargs):
        seen.append(name)
        return json.dumps({"ok": True})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    poller, stop = _start_poller(_local_shell(), rpc, {"terminal"})
    try:
        req = rpc / "req_000007"
        _publish_req(rpc, "req_000007", {"seq": "abc", "token": "right-token",
                                       "tool": "terminal", "args": {"command": "x"}})
        res = rpc / "res_000007"
        deadline = time.monotonic() + 5
        while not res.is_file() and time.monotonic() < deadline:
            stop.wait(.01)
        assert res.is_file(), "res_000007 missing (planted dir not swept or write failed)"
        assert json.loads(res.read_text()) == {"ok": True}
        assert seen == ["terminal"]           # answered once, named for the file seq
        assert not req.exists()
        # A name outside req_<digits> can never be answered: removed, never dispatched.
        bad = rpc / "req_evil"
        _publish_req(rpc, "req_evil", {"seq": 1, "token": "right-token",
                                     "tool": "terminal", "args": {"command": "x"}})
        deadline = time.monotonic() + 5
        while bad.exists() and time.monotonic() < deadline:
            stop.wait(.01)
        assert not bad.exists()
        assert seen == ["terminal"]
    finally:
        stop.set()
        poller.join(timeout=10)
        assert not poller.is_alive()


@pytest.mark.platforms("posix")
def test_poll_loop_never_redispatches_when_rm_silently_fails(tmp_path, monkeypatch):
    """If the consume-rm reports success but the file persists (weird fs, raced
    recreate), the dispatched map must still keep dispatch at-most-once."""
    rpc = tmp_path / "rpc"
    rpc.mkdir()
    seen = []

    def dispatch(name, args, **kwargs):
        seen.append(name)
        return json.dumps({"ok": True})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    shell = _local_shell()

    class Shell:
        def execute(self, command, cwd=None, timeout=None):
            # Simulate a silent rm failure on the req file: report success, keep the file.
            if "rm -rf" in command and "req_000003" in command:
                return {"output": "", "returncode": 0}
            return shell.execute(command, cwd=cwd, timeout=timeout)

    poller, stop = _start_poller(Shell(), rpc, {"terminal"})
    try:
        _publish_req(rpc, "req_000003", {"seq": 3, "token": "right-token",
                                       "tool": "terminal", "args": {"command": "x"}})
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            stop.wait(.01)
        assert seen == ["terminal"]
        time.sleep(.6)                        # req_000003 keeps listing; must not re-dispatch
        assert seen == ["terminal"]
    finally:
        stop.set()
        poller.join(timeout=10)
        assert not poller.is_alive()


@pytest.mark.platforms("posix")
def test_poll_loop_retries_undispatched_request_after_rm_transport_error(tmp_path, monkeypatch):
    """An rm that raises (transport failure) leaves the request undispatched; the
    next cycle retries it, and it must still dispatch exactly once."""
    rpc = tmp_path / "rpc"
    rpc.mkdir()
    seen, rm_attempts = [], [0]

    def dispatch(name, args, **kwargs):
        seen.append(name)
        return json.dumps({"ok": True})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    shell = _local_shell()

    class Shell:
        def execute(self, command, cwd=None, timeout=None):
            if "rm -rf" in command and "req_000005" in command:
                rm_attempts[0] += 1
                if rm_attempts[0] == 1:
                    raise RuntimeError("transport drop")   # rm ack lost before it ran
            return shell.execute(command, cwd=cwd, timeout=timeout)

    poller, stop = _start_poller(Shell(), rpc, {"terminal"})
    try:
        req = rpc / "req_000005"
        _publish_req(rpc, "req_000005", {"seq": 5, "token": "right-token",
                                       "tool": "terminal", "args": {"command": "x"}})
        res = rpc / "res_000005"
        deadline = time.monotonic() + 5
        while not res.exists() and time.monotonic() < deadline:
            stop.wait(.01)
        assert res.exists(), "res_000005 never written after rm retry"
        assert seen == ["terminal"]           # retried once, dispatched once
        assert not req.exists()
    finally:
        stop.set()
        poller.join(timeout=10)
        assert not poller.is_alive()


@pytest.mark.platforms("posix")
def test_poll_loop_sweeps_non_dict_and_newline_named_reqs(tmp_path, monkeypatch):
    """JSON that parses to a non-dict (null) must not wedge the loop on
    request.get; a name containing a newline must be removed byte-exact, not
    mint a phantom req_<digits> spelling that survives forever. A later legit
    request still dispatches."""
    rpc = tmp_path / "rpc"
    rpc.mkdir()
    seen = []

    def dispatch(name, args, **kwargs):
        seen.append(name)
        return json.dumps({"ok": True})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    poller, stop = _start_poller(_local_shell(), rpc, {"terminal"})
    try:
        weird = rpc / "req_0000\nphantom"
        weird.write_text("null")
        nodict = rpc / "req_000004"
        nodict.write_text("null")
        _publish_req(rpc, "req_000006", {"seq": 6, "token": "right-token",
                                       "tool": "terminal", "args": {"command": "x"}})
        res = rpc / "res_000006"
        deadline = time.monotonic() + 5
        while not res.exists() and time.monotonic() < deadline:
            stop.wait(.01)
        assert res.exists(), "legit req starved behind a wedged non-dict file"
        assert seen == ["terminal"]
        deadline = time.monotonic() + 5
        while (nodict.exists() or weird.exists()) and time.monotonic() < deadline:
            stop.wait(.01)
        assert not nodict.exists() and not weird.exists()
    finally:
        stop.set()
        poller.join(timeout=10)
        assert not poller.is_alive()
