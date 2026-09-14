"""Managed Modal environment backed by tool-gateway.

Deliberately overrides :meth:`BaseEnvironment.execute`: the tool-gateway does command
preparation, CWD tracking and env-snapshot management server-side, so the base
``_wrap_command`` / ``_wait_for_process`` / snapshot machinery does not apply.
"""

from __future__ import annotations

import json
import logging
import os
import requests
import shlex
import time
import uuid
from typing import Any, Dict, Optional

from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted
from tools.managed_tool_gateway import resolve_managed_tool_gateway

logger = logging.getLogger(__name__)

_TERMINAL_EXEC_STATUSES = frozenset({"completed", "failed", "cancelled", "timeout"})
_CLIENT_TIMEOUT_GRACE_SECONDS = 10.0


def _coerce_number(value: Any, default: float) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _request_timeout_env(name: str, default: float) -> float:
    value = _coerce_number(os.getenv(name), default)
    return value if value > 0 else default


def _result(output: str, returncode: int = 1) -> dict:
    return {"output": output, "returncode": returncode}


class ManagedModalEnvironment(BaseEnvironment):
    """Gateway-owned Modal sandbox with Hermes-compatible execute/cleanup."""

    _stdin_mode = "payload"
    _CONNECT_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_CONNECT_TIMEOUT_SECONDS", 1.0)
    _POLL_READ_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_POLL_READ_TIMEOUT_SECONDS", 5.0)
    _CANCEL_READ_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_CANCEL_READ_TIMEOUT_SECONDS", 5.0)

    def __init__(self, image: str, cwd: str = "/root", timeout: int = 60,
                 modal_sandbox_kwargs: Optional[Dict[str, Any]] = None,
                 persistent_filesystem: bool = True, task_id: str = "default"):
        super().__init__(cwd=cwd, timeout=timeout)
        # Managed Modal does not sync or mount host credential files.
        try:
            from tools.credential_files import get_credential_file_mounts
        except Exception:
            get_credential_file_mounts = None
        if get_credential_file_mounts is not None and get_credential_file_mounts():
            raise ValueError(
                "Managed Modal does not support host credential-file passthrough. "
                "Use TERMINAL_MODAL_MODE=direct when skills or config require "
                "credential files inside the sandbox."
            )
        gateway = resolve_managed_tool_gateway("modal")
        if gateway is None:
            raise ValueError("Managed Modal requires a configured tool gateway and Nous user token")
        self._gateway_origin = gateway.gateway_origin.rstrip("/")
        self._nous_user_token = gateway.nous_user_token
        self._task_id, self._persistent, self._image = task_id, persistent_filesystem, image
        self._sandbox_kwargs = dict(modal_sandbox_kwargs or {})
        self._create_idempotency_key = str(uuid.uuid4())
        self._sandbox_id = self._create_sandbox()

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None, stdin_data: str | None = None,
                rewrite_compound_background: bool = True, bounded_capture: bool = False) -> dict:
        # Signature parity with BaseEnvironment.execute only: the gateway runs commands
        # explicitly (no shell background rewriting) and returns the remote result in one
        # payload, so streaming-time bounding does not apply (the terminal tool's final
        # truncation still caps it).
        del rewrite_compound_background, bounded_capture
        exec_command, sudo_stdin = self._prepare_command(command)
        if sudo_stdin is not None:
            # Feed sudo via a shell pipe: the transport has no direct stdin piping.
            exec_command = f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {exec_command}"
        timeout = timeout or self.timeout
        exec_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {"execId": exec_id, "command": exec_command, "cwd": cwd or self.cwd,
                                   "timeoutMs": int(timeout * 1000)}
        if stdin_data is not None:
            payload["stdinData"] = stdin_data
        try:
            response = self._request("POST", f"/v1/sandboxes/{self._sandbox_id}/execs", json=payload, timeout=10)
            body = response.json() if response.status_code < 400 else None
        except Exception as exc:
            return _result(f"Managed Modal exec failed: {exc}")
        if body is None:
            return _result(self._format_error("Managed Modal exec failed", response))
        if (final := self._result_from_body(body)) is not None:
            return final
        if body.get("execId") != exec_id:
            return _result("Managed Modal exec start did not return the expected exec id")
        deadline = time.monotonic() + timeout + _CLIENT_TIMEOUT_GRACE_SECONDS
        _now = time.monotonic()
        _activity_state = {"last_touch": _now, "start": _now}
        while True:
            if is_interrupted():
                self._cancel_exec(exec_id)
                return _result("[Command interrupted - Modal sandbox exec cancelled]", 130)
            try:
                if (result := self._poll_exec(exec_id)) is not None:
                    return result
            except Exception as exc:
                return _result(f"Managed Modal exec failed: {exc}")
            if time.monotonic() >= deadline:
                self._cancel_exec(exec_id)
                return _result(f"Managed Modal exec timed out after {timeout}s", 124)
            # Periodic activity touch so the gateway knows we're alive (lazy import:
            # tests stub tools.environments.base with only BaseEnvironment)
            try:
                from tools.environments.base import touch_activity_if_due
                touch_activity_if_due(_activity_state, "modal command running")
            except Exception:
                pass
            time.sleep(0.25)

    @staticmethod
    def _result_from_body(body: dict) -> dict | None:
        """Final result dict if the exec body reports a terminal status, else ``None``."""
        if body.get("status") in _TERMINAL_EXEC_STATUSES:
            return _result(body.get("output", ""), body.get("returncode", 1))

    def _poll_exec(self, exec_id: str) -> dict | None:
        try:
            response = self._request("GET", f"/v1/sandboxes/{self._sandbox_id}/execs/{exec_id}",
                                     timeout=(self._CONNECT_TIMEOUT_SECONDS, self._POLL_READ_TIMEOUT_SECONDS))
        except Exception as exc:
            return _result(f"Managed Modal exec poll failed: {exc}")
        if response.status_code == 404:
            return _result("Managed Modal exec not found")
        if response.status_code >= 400:
            return _result(self._format_error("Managed Modal exec poll failed", response))
        return self._result_from_body(response.json())

    def _cancel_exec(self, exec_id: str) -> None:
        try:
            self._request("POST", f"/v1/sandboxes/{self._sandbox_id}/execs/{exec_id}/cancel",
                          timeout=(self._CONNECT_TIMEOUT_SECONDS, self._CANCEL_READ_TIMEOUT_SECONDS))
        except Exception as exc:
            logger.warning("Managed Modal exec cancel failed: %s", exc)

    def cleanup(self):
        if not getattr(self, "_sandbox_id", None):
            return
        try:
            self._request("POST", f"/v1/sandboxes/{self._sandbox_id}/terminate",
                          json={"snapshotBeforeTerminate": self._persistent}, timeout=60)
        except Exception as exc:
            logger.warning("Managed Modal cleanup failed: %s", exc)
        finally:
            self._sandbox_id = None

    def _create_sandbox(self) -> str:
        kw = self._sandbox_kwargs
        cpu = _coerce_number(kw.get("cpu"), 1)
        memory = _coerce_number(kw.get("memoryMiB", kw.get("memory")), 5120)
        disk = _coerce_number(kw.get("ephemeral_disk", kw.get("diskMiB")), None)
        create_payload = {
            "image": self._image, "cwd": self.cwd, "cpu": cpu, "memoryMiB": memory, "timeoutMs": 3_600_000,
            "idleTimeoutMs": max(300_000, int(self.timeout * 1000)),
            "persistentFilesystem": self._persistent, "logicalKey": self._task_id,
        }
        if disk is not None:
            create_payload["diskMiB"] = disk
        response = self._request("POST", "/v1/sandboxes", json=create_payload, timeout=60,
                                 extra_headers={"x-idempotency-key": self._create_idempotency_key})
        if response.status_code >= 400:
            raise RuntimeError(self._format_error("Managed Modal create failed", response))
        sandbox_id = response.json().get("id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise RuntimeError("Managed Modal create did not return a sandbox id")
        return sandbox_id

    def _request(self, method: str, path: str, *, json: Dict[str, Any] | None = None, timeout: int = 30,
                 extra_headers: Dict[str, str] | None = None) -> requests.Response:
        headers = {"Authorization": f"Bearer {self._nous_user_token}", "Content-Type": "application/json",
                   **(extra_headers or {})}
        return requests.request(method, f"{self._gateway_origin}{path}", headers=headers, json=json, timeout=timeout)

    @staticmethod
    def _format_error(prefix: str, response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = payload.get("error") or payload.get("message") or payload.get("code")
                if isinstance(message, str) and message:
                    return f"{prefix}: {message}"
                return f"{prefix}: {json.dumps(payload, ensure_ascii=False)}"
        except Exception:
            pass
        text = response.text.strip()
        return f"{prefix}: {text}" if text else f"{prefix}: HTTP {response.status_code}"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import dataclass  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
