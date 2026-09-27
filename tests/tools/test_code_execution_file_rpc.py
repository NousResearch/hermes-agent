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
        def execute(self, command, cwd=None, timeout=None, stdin_data=None):
            result = subprocess.run([shell, "-c", command], cwd=cwd, timeout=timeout,
                                    env=dict(os.environ), input=stdin_data or "", capture_output=True, text=True)
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
            blocked = {"background", "heartbeat", "pty", "notify", "notify_on_complete", "watch_patterns", "persist_on_release"} if name == "terminal" else set()
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
