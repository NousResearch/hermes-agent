"""Managed Modal environment backed by tool-gateway."""

from __future__ import annotations

import json
import logging
import os
import requests
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.environments.modal_utils import (
    BaseModalExecutionEnvironment,
    ModalExecStart,
    PreparedModalExec,
)
from tools.managed_tool_gateway import resolve_managed_tool_gateway

logger = logging.getLogger(__name__)


def _request_timeout_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class _ManagedModalExecHandle:
    exec_id: str


class ManagedModalEnvironment(BaseModalExecutionEnvironment):
    """Gateway-owned Modal sandbox with Hermes-compatible execute/cleanup."""

    _CONNECT_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_CONNECT_TIMEOUT_SECONDS", 1.0)
    _POLL_READ_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_POLL_READ_TIMEOUT_SECONDS", 5.0)
    _CANCEL_READ_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_CANCEL_READ_TIMEOUT_SECONDS", 5.0)
    _client_timeout_grace_seconds = 10.0
    _interrupt_output = "[Command interrupted - Modal sandbox exec cancelled]"
    _unexpected_error_prefix = "Managed Modal exec failed"

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        modal_sandbox_kwargs: Optional[Dict[str, Any]] = None,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        self._guard_unsupported_credential_passthrough()

        gateway = resolve_managed_tool_gateway("modal")
        if gateway is None:
            raise ValueError("Managed Modal requires a configured tool gateway and Nous user token")

        self._gateway_origin = gateway.gateway_origin.rstrip("/")
        self._nous_user_token = gateway.nous_user_token
        self._task_id = task_id
        self._persistent = persistent_filesystem
        self._image = image
        self._sandbox_kwargs = dict(modal_sandbox_kwargs or {})
        self._create_idempotency_key = str(uuid.uuid4())
        self._sandbox_id = self._create_sandbox()

    def _start_modal_exec(self, prepared: PreparedModalExec) -> ModalExecStart:
        exec_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "execId": exec_id,
            "command": prepared.command,
            "cwd": prepared.cwd,
            "timeoutMs": int(prepared.timeout * 1000),
        }
        if prepared.stdin_data is not None:
            payload["stdinData"] = prepared.stdin_data

        try:
            response = self._request(
                "POST",
                f"/v1/sandboxes/{self._sandbox_id}/execs",
                json=payload,
                timeout=10,
            )
        except Exception as exc:
            return ModalExecStart(
                immediate_result=self._error_result(f"Managed Modal exec failed: {exc}")
            )

        if response.status_code >= 400:
            return ModalExecStart(
                immediate_result=self._error_result(
                    self._format_error("Managed Modal exec failed", response)
                )
            )

        body = response.json()
        status = body.get("status")
        if status in {"completed", "failed", "cancelled", "timeout"}:
            return ModalExecStart(
                immediate_result=self._result(
                    body.get("output", ""),
                    body.get("returncode", 1),
                )
            )

        if body.get("execId") != exec_id:
            return ModalExecStart(
                immediate_result=self._error_result(
                    "Managed Modal exec start did not return the expected exec id"
                )
            )

        return ModalExecStart(handle=_ManagedModalExecHandle(exec_id=exec_id))

    def _poll_modal_exec(self, handle: _ManagedModalExecHandle) -> dict | None:
        try:
            status_response = self._request(
                "GET",
                f"/v1/sandboxes/{self._sandbox_id}/execs/{handle.exec_id}",
                timeout=(self._CONNECT_TIMEOUT_SECONDS, self._POLL_READ_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            return self._error_result(f"Managed Modal exec poll failed: {exc}")

        if status_response.status_code == 404:
            return self._error_result("Managed Modal exec not found")

        if status_response.status_code >= 400:
            return self._error_result(
                self._format_error("Managed Modal exec poll failed", status_response)
            )

        status_body = status_response.json()
        status = status_body.get("status")
        if status in {"completed", "failed", "cancelled", "timeout"}:
            return self._result(
                status_body.get("output", ""),
                status_body.get("returncode", 1),
            )
        return None

    def _cancel_modal_exec(self, handle: _ManagedModalExecHandle) -> None:
        self._cancel_exec(handle.exec_id)

    def _timeout_result_for_modal(self, timeout: int) -> dict:
        return self._result(f"Managed Modal exec timed out after {timeout}s", 124)

    def cleanup(self):
        if not getattr(self, "_sandbox_id", None):
            return

        try:
            self._request(
                "POST",
                f"/v1/sandboxes/{self._sandbox_id}/terminate",
                json={
                    "snapshotBeforeTerminate": self._persistent,
                },
                timeout=60,
            )
        except Exception as exc:
            logger.warning("Managed Modal cleanup failed: %s", exc)
        finally:
            self._sandbox_id = None

    def _create_sandbox(self) -> str:
        cpu = self._coerce_number(self._sandbox_kwargs.get("cpu"), 1)
        memory = self._coerce_number(
            self._sandbox_kwargs.get("memoryMiB", self._sandbox_kwargs.get("memory")),
            5120,
        )
        disk = self._coerce_number(
            self._sandbox_kwargs.get("ephemeral_disk", self._sandbox_kwargs.get("diskMiB")),
            None,
        )

        create_payload = {
            "image": self._image,
            "cwd": self.cwd,
            "cpu": cpu,
            "memoryMiB": memory,
            "timeoutMs": 3_600_000,
            "idleTimeoutMs": max(300_000, int(self.timeout * 1000)),
            "persistentFilesystem": self._persistent,
            "logicalKey": self._task_id,
        }
        if disk is not None:
            create_payload["diskMiB"] = disk

        response = self._request(
            "POST",
            "/v1/sandboxes",
            json=create_payload,
            timeout=60,
            extra_headers={
                "x-idempotency-key": self._create_idempotency_key,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(self._format_error("Managed Modal create failed", response))

        body = response.json()
        sandbox_id = body.get("id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise RuntimeError("Managed Modal create did not return a sandbox id")
        return sandbox_id

    def _guard_unsupported_credential_passthrough(self) -> None:
        """Managed Modal does not sync or mount host credential files."""
        try:
            from tools.credential_files import get_credential_file_mounts
        except Exception:
            return

        mounts = get_credential_file_mounts()
        if mounts:
            raise ValueError(
                "Managed Modal does not support host credential-file passthrough. "
                "Use TERMINAL_MODAL_MODE=direct when skills or config require "
                "credential files inside the sandbox."
            )

    def _request(self, method: str, path: str, *,
                 json: Dict[str, Any] | None = None,
                 timeout: int = 30,
                 extra_headers: Dict[str, str] | None = None) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self._nous_user_token}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        return requests.request(
            method,
            f"{self._gateway_origin}{path}",
            headers=headers,
            json=json,
            timeout=timeout,
        )

    def _cancel_exec(self, exec_id: str) -> None:
        try:
            self._request(
                "POST",
                f"/v1/sandboxes/{self._sandbox_id}/execs/{exec_id}/cancel",
                timeout=(self._CONNECT_TIMEOUT_SECONDS, self._CANCEL_READ_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            logger.warning("Managed Modal exec cancel failed: %s", exc)

    @staticmethod
    def _coerce_number(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

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
        if text:
            return f"{prefix}: {text}"
        return f"{prefix}: HTTP {response.status_code}"
