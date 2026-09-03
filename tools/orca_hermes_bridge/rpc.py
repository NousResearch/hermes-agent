"""Python boundary for Orca's packaged authenticated runtime client."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .windows import hidden_process_flags


class OrcaRpcError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _default_node() -> Path:
    executable = shutil.which("node")
    if not executable:
        raise OrcaRpcError("node_unavailable", "Node.js is required for the Orca bridge")
    return Path(executable)


def _default_resources() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise OrcaRpcError("orca_unavailable", "LOCALAPPDATA is not configured")
    return Path(local_app_data) / "Programs" / "Orca" / "resources"


class OrcaRpcClient:
    def __init__(
        self,
        node_executable: Path | None = None,
        resources_path: Path | None = None,
        helper_path: Path | None = None,
        timeout_seconds: float = 10.0,
    ):
        self.node_executable = node_executable or _default_node()
        self.resources_path = resources_path or _default_resources()
        self.helper_path = helper_path or Path(__file__).with_name("runtime_rpc.cjs")
        self.timeout_seconds = timeout_seconds

    def list_accounts(self) -> dict[str, Any]:
        return self._call("accounts.list", {"refreshUsage": False})

    def select_host_codex(self, account_id: str | None) -> dict[str, Any]:
        return self._call(
            "accounts.selectCodexForTarget",
            {
                "accountId": account_id,
                "target": {"runtime": "host", "wslDistro": None},
            },
        )

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request = {
            "resourcesPath": str(self.resources_path),
            "method": method,
            "params": params,
        }
        try:
            completed = subprocess.run(
                [str(self.node_executable), str(self.helper_path)],
                input=json.dumps(request, separators=(",", ":")),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_seconds,
                check=False,
                creationflags=hidden_process_flags(),
            )
        except subprocess.TimeoutExpired as exc:
            raise OrcaRpcError("runtime_timeout", "Orca runtime request timed out") from exc
        except OSError as exc:
            raise OrcaRpcError("runtime_unavailable", "Could not start the Orca RPC helper") from exc

        try:
            payload = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise OrcaRpcError("invalid_response", "Orca RPC helper returned an invalid response") from exc
        if completed.returncode != 0 or payload.get("ok") is not True:
            error = payload.get("error") if isinstance(payload, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            raise OrcaRpcError(
                str(code or "runtime_error"),
                "Orca runtime request failed",
            )
        response = payload.get("response")
        result = response.get("result") if isinstance(response, dict) else None
        if not isinstance(result, dict):
            raise OrcaRpcError("invalid_response", "Orca RPC response has no result object")
        return result
