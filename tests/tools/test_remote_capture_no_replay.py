"""A published remote cell must not be replayed after capture failure."""
import json
from unittest.mock import Mock

import pytest

from tools.environments.capture import OutputCaptureError


def test_remote_capture_uncertainty_never_falls_back(monkeypatch):
    from tools import code_execution_tool as tool
    import tools.code_kernel_remote as remote
    env = Mock()
    env.execute.return_value = {"output": "OK", "returncode": 0}
    monkeypatch.setattr(tool, "_get_or_create_env", lambda _: (env, "ssh"))
    monkeypatch.setattr(tool, "_load_config", dict)
    executions = []
    def publish_then_lose_capture(*args, **kwargs):
        executions.append("published")
        raise OutputCaptureError("Output capture is incomplete; command may have taken effect")
    monkeypatch.setattr(remote, "execute_in_remote_kernel", publish_then_lose_capture)
    fallback = Mock(return_value='{"status":"success"}')
    monkeypatch.setattr(tool, "_run_remote_per_call", fallback)
    result = json.loads(tool._execute_remote("side_effect()", "capture", []))
    fallback.assert_not_called()
    assert executions == ["published"]
    assert result["status"] == "error"
    assert result["output_incomplete"] is True


@pytest.mark.parametrize("failure", ["capture", "transport", "response"])
def test_atomic_cell_publication_capture_failure_does_not_replay(monkeypatch, failure):
    from types import SimpleNamespace
    from tools import code_execution_tool as tool
    from tools import code_kernel_remote as remote

    publications = []
    env = Mock()
    def execute(command, **kwargs):
        if "cell_req_" in command and "mv " in command:
            publications.append(command)
            if failure == "transport":
                raise OSError("private transport failure after rename")
            if failure == "response":
                return None  # A lost/malformed response after successful publication.
            return {"output": "", "returncode": 0, "output_incomplete": True}
        return {"output": "OK", "returncode": 0}
    env.execute.side_effect = execute
    kernel = SimpleNamespace(env=env, kernel_dir="/tmp/test-kernel", cell_seq=0,
                             sh=lambda *a, **kw: "", rpc_token="test")
    monkeypatch.setattr(tool, "_get_or_create_env", lambda _: (env, "ssh"))
    monkeypatch.setattr(tool, "_load_config", dict)
    monkeypatch.setattr(tool, "_rpc_poll_loop", lambda *a, **kw: None)
    monkeypatch.setattr(remote._REGISTRY, "discard", lambda *a: None)
    def execute_kernel(code, **kwargs):
        return remote._run_attached_cell(
            kernel, ("test",), code, env=env, task_env_id="capture", sandbox_tools=frozenset(),
            timeout=2, max_tool_calls=1, reused=True, state_reset=False, state_lost=False)
    monkeypatch.setattr(remote, "execute_in_remote_kernel", execute_kernel)
    fallback = Mock()
    monkeypatch.setattr(tool, "_run_remote_per_call", fallback)
    result = json.loads(tool._execute_remote("side_effect()", "capture", []))
    assert len(publications) == 1
    assert result["status"] == "error"
    assert result["output_incomplete"] is True
    fallback.assert_not_called()
