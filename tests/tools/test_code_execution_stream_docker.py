"""Opt-in real registry → Docker kernel → tool RPC → file tools integration.

HERMES_RUN_E2E=1 scripts/run_tests.sh tests/tools/test_code_execution_stream_docker.py
Needs Docker and python:3.14-slim-bookworm; no provider credentials or inference.
"""
import concurrent.futures
import json
import os
import uuid
from unittest.mock import patch

import pytest


@pytest.mark.skipif(os.environ.get("HERMES_RUN_E2E") != "1", reason="opt-in live Docker integration")
def test_concurrent_docker_owners_use_real_tools_and_preserve_kernel_state(tmp_path):
    from agent import secret_scope
    import tools.code_execution_tool  # noqa: F401 — register handlers
    import tools.file_tools as files
    from model_tools import handle_function_call
    from tools.code_kernel_remote import shutdown_all_remote_kernels
    from tools.environments.docker import DockerEnvironment
    from tools import terminal_tool as terminal
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope

    def run_owner(owner):
        home = tmp_path / owner
        home.mkdir()
        secret_token = secret_scope.set_secret_scope({}, profile_home=str(home))
        terminal_token = set_terminal_scope({"TERMINAL_ENV": "docker", "TERMINAL_CWD": "/workspace"})
        task = "stream-e2e-" + uuid.uuid4().hex
        env = None
        try:
            env = DockerEnvironment(image="python:3.14-slim-bookworm", cwd="/workspace",
                                    network=False, task_id=task, persist_across_processes=False)
            terminal.register_task_env_overrides(task, {"env_type": "docker", "cwd": "/workspace"})
            terminal._active_environments[task] = env
            code = f'''
from hermes_tools import write_file, read_file
import json
owner = {owner!r}
for i in range(12):
    result = write_file('/workspace/row-' + str(i), owner + ':' + str(i))
    assert 'error' not in result, result
rows = [read_file('/workspace/row-' + str(i)) for i in range(12)]
assert all(owner + ':' + str(i) in row['content'] for i, row in enumerate(rows)), rows
total = len(rows)
print(json.dumps({{'owner': owner, 'rows': total}}))
'''
            with patch.object(env, "open_code_rpc", wraps=env.open_code_rpc) as opened:
                first = json.loads(handle_function_call("execute_code", {"code": code}, task_id=task,
                                                       enabled_tools=["read_file", "write_file"]))
                assert first["status"] == "success", first
                assert first["tool_calls_made"] == 24
                assert json.loads(first["output"]) == {"owner": owner, "rows": 12}
                second = json.loads(handle_function_call(
                    "execute_code", {"code": "print(json.dumps({'owner': owner, 'rows': total}))"},
                    task_id=task, enabled_tools=["read_file", "write_file"]))
                assert second["status"] == "success", second
                assert second["kernel"]["reused"] is True
                assert json.loads(second["output"]) == {"owner": owner, "rows": 12}
                assert opened.call_count == 2
        finally:
            # All resources belong to this test's unique task/container.
            terminal._active_environments.pop(task, None)
            terminal.clear_task_env_overrides(task)
            files.clear_file_ops_cache(task)
            if env is not None:
                from tools.code_kernel_remote import shutdown_remote_kernels_for_owner
                from tools.code_kernel import _resolve_owner
                shutdown_remote_kernels_for_owner(_resolve_owner(task))
                env.cleanup()
            reset_terminal_scope(terminal_token)
            secret_scope.reset_secret_scope(secret_token)

    secret_scope.set_multiplex_active(True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run_owner, ["A", "B"]))
    finally:
        shutdown_all_remote_kernels()
        assert DockerEnvironment.wait_for_all_teardowns(timeout=30)
        secret_scope.set_multiplex_active(False)
