"""Tests for the narrow Orca runtime RPC subprocess boundary."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.orca_hermes_bridge.rpc import OrcaRpcClient, OrcaRpcError


REAL_HELPER = Path(__file__).parents[3] / "tools" / "orca_hermes_bridge" / "runtime_rpc.cjs"


def _python_helper(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_rpc.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_select_emits_explicit_host_target_through_real_subprocess(tmp_path):
    helper = _python_helper(
        tmp_path,
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({'ok': True, 'response': {'id': '1', 'ok': True, "
        "'result': {'request': request}, '_meta': {'runtimeId': 'runtime-1'}}}))\n",
    )
    client = OrcaRpcClient(
        node_executable=Path(sys.executable),
        resources_path=tmp_path / "resources",
        helper_path=helper,
    )

    result = client.select_host_codex("managed-1")

    assert result["request"] == {
        "resourcesPath": str(tmp_path / "resources"),
        "method": "accounts.selectCodexForTarget",
        "params": {
            "accountId": "managed-1",
            "target": {"runtime": "host", "wslDistro": None},
        },
    }


def test_list_requests_no_usage_refresh(tmp_path):
    helper = _python_helper(
        tmp_path,
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({'ok': True, 'response': {'id': '1', 'ok': True, "
        "'result': request, '_meta': {'runtimeId': 'runtime-1'}}}))\n",
    )
    client = OrcaRpcClient(Path(sys.executable), tmp_path, helper)

    assert client.list_accounts() == {
        "resourcesPath": str(tmp_path),
        "method": "accounts.list",
        "params": {"refreshUsage": False},
    }


def test_timeout_error_does_not_echo_request_or_subprocess_details(tmp_path):
    helper = _python_helper(tmp_path, "import time\ntime.sleep(2)\n")
    client = OrcaRpcClient(
        Path(sys.executable), tmp_path / "resources", helper, timeout_seconds=0.05
    )

    with pytest.raises(OrcaRpcError) as exc:
        client.list_accounts()

    assert exc.value.code == "runtime_timeout"
    assert "stdin" not in str(exc.value).lower()
    assert "access_token" not in str(exc.value)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_commonjs_helper_calls_packaged_runtime_client_and_rejects_other_methods(tmp_path):
    cli_dir = tmp_path / "app.asar.unpacked" / "out" / "cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "runtime-client.js").write_text(
        "class RuntimeClient {\n"
        "  async call(method, params) {\n"
        "    return {id: 'rpc-1', ok: true, result: {method, params}, "
        "_meta: {runtimeId: 'runtime-1'}}\n"
        "  }\n"
        "}\n"
        "module.exports = {RuntimeClient}\n",
        encoding="utf-8",
    )
    allowed = subprocess.run(
        [shutil.which("node"), str(REAL_HELPER)],
        input=json.dumps({
            "resourcesPath": str(tmp_path),
            "method": "accounts.selectCodexForTarget",
            "params": {
                "accountId": None,
                "target": {"runtime": "host", "wslDistro": None},
            },
        }),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )
    rejected = subprocess.run(
        [shutil.which("node"), str(REAL_HELPER)],
        input=json.dumps({
            "resourcesPath": str(tmp_path),
            "method": "accounts.removeCodex",
            "params": {"accountId": "managed-1"},
        }),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )

    allowed_payload = json.loads(allowed.stdout)
    rejected_payload = json.loads(rejected.stdout)
    assert allowed.returncode == 0
    assert allowed_payload["response"]["result"]["method"] == "accounts.selectCodexForTarget"
    assert rejected.returncode == 1
    assert rejected_payload == {
        "ok": False,
        "error": {"code": "invalid_method", "message": "Orca RPC request rejected"},
    }
