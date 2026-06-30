"""Minimal Coder execution environment.

v1 bootstrap implementation: resolve a workspace agent over the Coder REST API,
optionally create or auto-start a stopped workspace, open the workspace PTY
websocket, and read terminal output until EOF.

Current intentional limitations for the bootstrap step:
- treats websocket EOF/close as successful completion
- no persistent shell/session snapshot integration yet
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import threading
import time
import urllib.parse
import uuid

import requests
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from hermes_state import SessionDB
from tools.environments.base import BaseEnvironment, _ThreadedProcessHandle
from tools.environments.forward_env import collect_forwarded_env_values, normalize_forward_env_names

logger = logging.getLogger(__name__)

_WORKSPACE_NAME_PREFIX = "hermes-"
_WORKSPACE_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,31})$")
_MAX_WORKSPACE_NAME_LEN = 32


def _coder_headers(api_key: str) -> dict[str, str]:
    return {"Coder-Session-Token": api_key}



def _resolve_lineage_root_session_id(session_id: str, db: SessionDB) -> str:
    """Resolve the lineage root for a Hermes session ID.

    If the session cannot be found, fall back to the provided ID unchanged.
    """
    if not session_id:
        raise ValueError("Coder workspace resolution requires a non-empty session/task id")

    current = session_id
    visited: set[str] = set()

    while current and current not in visited:
        visited.add(current)
        session = db.get_session(current)
        if not session:
            break
        parent = session.get("parent_session_id")
        if not parent:
            return current
        current = parent

    return current or session_id



def _sanitize_workspace_name_suffix(value: str, max_len: int) -> str:
    sanitized = value.lower().replace("_", "-")
    sanitized = re.sub(r"[^a-z0-9-]+", "-", sanitized)
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    if not sanitized:
        sanitized = "task"
    if not sanitized[0].isalnum():
        sanitized = f"t-{sanitized}"
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rstrip("-")
    if not sanitized:
        sanitized = "task"
    return sanitized



def coder_workspace_name_for_task(task_id: str, db: SessionDB | None = None) -> str:
    """Map a Hermes task/session to a deterministic Coder workspace name."""
    database = db or SessionDB()
    root_session_id = _resolve_lineage_root_session_id(task_id, db=database)
    max_suffix_len = _MAX_WORKSPACE_NAME_LEN - len(_WORKSPACE_NAME_PREFIX)
    workspace_name = f"{_WORKSPACE_NAME_PREFIX}{_sanitize_workspace_name_suffix(root_session_id, max_suffix_len)}"
    if not _WORKSPACE_NAME_PATTERN.fullmatch(workspace_name):
        raise ValueError(f"Derived Coder workspace name is invalid after sanitization: {workspace_name!r}")
    return workspace_name



def _workspace_search_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v2/workspaces"



def _user_organizations_url(base_url: str, user: str = "me") -> str:
    user_part = urllib.parse.quote(user, safe="")
    return f"{base_url.rstrip('/')}/api/v2/users/{user_part}/organizations"


def _user_organization_by_name_url(base_url: str, organization_name: str, user: str = "me") -> str:
    user_part = urllib.parse.quote(user, safe="")
    org_part = urllib.parse.quote(organization_name, safe="")
    return f"{base_url.rstrip('/')}/api/v2/users/{user_part}/organizations/{org_part}"


def _template_by_name_url(base_url: str, organization_id: str, template_name: str) -> str:
    org_part = urllib.parse.quote(organization_id, safe="")
    template_part = urllib.parse.quote(template_name, safe="")
    return f"{base_url.rstrip('/')}/api/v2/organizations/{org_part}/templates/{template_part}"



def _create_workspace_url(base_url: str, organization_id: str, user: str = "me") -> str:
    org_part = urllib.parse.quote(organization_id, safe="")
    user_part = urllib.parse.quote(user, safe="")
    return f"{base_url.rstrip('/')}/api/v2/organizations/{org_part}/members/{user_part}/workspaces"



def _find_workspace_by_name(*, base_url: str, workspace_name: str, api_key: str, timeout: int = 10) -> dict | None:
    response = requests.get(
        _workspace_search_url(base_url),
        headers=_coder_headers(api_key),
        params={"q": f"owner:me name:{workspace_name}", "limit": 100},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    workspaces = payload.get("workspaces") if isinstance(payload, dict) else None
    if not isinstance(workspaces, list):
        raise RuntimeError(f"Unexpected workspace search payload while looking up {workspace_name!r}")
    for workspace in workspaces:
        if isinstance(workspace, dict) and workspace.get("name") == workspace_name:
            owner = workspace.get("owner_name") or workspace.get("owner", {}).get("username")
            if owner in (None, "", "me") or owner == workspace.get("owner_name"):
                return workspace
    return None



def _get_organization_id(*, base_url: str, api_key: str, organization_name: str | None = None, timeout: int = 10) -> str:
    if organization_name:
        response = requests.get(
            _user_organization_by_name_url(base_url, organization_name),
            headers=_coder_headers(api_key),
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        org_id = payload.get("id") if isinstance(payload, dict) else None
        if not org_id:
            raise RuntimeError(f"Coder organization {organization_name!r} payload did not include an id")
        return org_id

    response = requests.get(
        _user_organizations_url(base_url),
        headers=_coder_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Coder user has no accessible organizations for workspace creation")
    default_org = next((org for org in payload if isinstance(org, dict) and org.get("is_default")), None)
    chosen = default_org or payload[0]
    org_id = chosen.get("id") if isinstance(chosen, dict) else None
    if not org_id:
        raise RuntimeError("Coder organization payload did not include an id")
    return org_id


def _resolve_template_id(*, base_url: str, organization_id: str, template_name: str, api_key: str, timeout: int = 10) -> str:
    response = requests.get(
        _template_by_name_url(base_url, organization_id, template_name),
        headers=_coder_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    template_id = payload.get("id") if isinstance(payload, dict) else None
    if not template_id:
        raise RuntimeError(f"Coder template {template_name!r} payload did not include an id")
    return template_id



def _create_workspace(
    *,
    base_url: str,
    workspace_name: str,
    template_name: str,
    api_key: str,
    organization_name: str | None = None,
    timeout: int = 10,
) -> dict:
    organization_id = _get_organization_id(
        base_url=base_url,
        api_key=api_key,
        organization_name=organization_name,
        timeout=timeout,
    )
    template_id = _resolve_template_id(
        base_url=base_url,
        organization_id=organization_id,
        template_name=template_name,
        api_key=api_key,
        timeout=timeout,
    )
    response = requests.post(
        _create_workspace_url(base_url, organization_id),
        headers={**_coder_headers(api_key), "Content-Type": "application/json", "Accept": "application/json"},
        json={"name": workspace_name, "template_id": template_id},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected create-workspace payload for Coder workspace {workspace_name!r}")
    return payload



def coder_workspace_exists(*, base_url: str, workspace_name: str, api_key: str, timeout: int = 10) -> bool:
    """Return True when a workspace with the given name exists for the current user."""
    return _find_workspace_by_name(
        base_url=base_url,
        workspace_name=workspace_name,
        api_key=api_key,
        timeout=timeout,
    ) is not None


class CoderEnvironment(BaseEnvironment):
    """Execute commands inside a Coder workspace via the /pty websocket."""

    _snapshot_timeout = 180
    _stdin_mode = "passthrough"
    _STDIN_CHUNK_SIZE = 32 * 1024
    _PTY_RECV_POLL_TIMEOUT = 1.0
    _PTY_EMPTY_EOF_RECONNECTS = 5
    _PTY_EMPTY_EOF_RECONNECT_WINDOW = 3.0
    _PTY_EMPTY_EOF_RECONNECT_DELAY = 0.2

    def __init__(
        self,
        *,
        base_url: str,
        template_name: str,
        task_id: str,
        api_key: str,
        organization_name: str | None = None,
        workspace_name: str | None = None,
        cwd: str = "~",
        timeout: int = 60,
        forward_env: list[str] | None = None,
        workspace_startup_timeout: int | None = None,
        init_session: bool = True,
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.template_name = template_name
        self.organization_name = organization_name or None
        self.task_id = task_id
        self.workspace = workspace_name or coder_workspace_name_for_task(task_id)
        self.api_key = api_key
        if workspace_startup_timeout is not None:
            self._snapshot_timeout = int(workspace_startup_timeout)
        self._workspace_id: str | None = None
        self._forward_env = normalize_forward_env_names(forward_env, config_name="coder_forward_env")

        # Safe to call here: init_session() uses _run_bash() directly, which
        # resolves the workspace/agent and opens a PTY without going back
        # through BaseEnvironment.execute(), so there is no recursive wrapping
        # or re-entry into init_session().
        if init_session:
            self.init_session()

    def _headers(self) -> dict[str, str]:
        return _coder_headers(self.api_key)

    def _workspace_url(self) -> str:
        if not self._workspace_id:
            raise RuntimeError(f"Coder workspace {self.workspace!r} has not been resolved yet")
        workspace_id = urllib.parse.quote(self._workspace_id, safe="")
        return f"{self.base_url}/api/v2/workspaces/{workspace_id}"

    def _workspace_build_url(self, build_id: str) -> str:
        return f"{self.base_url}/api/v2/workspacebuilds/{urllib.parse.quote(build_id, safe='')}"

    def _workspace_builds_url(self, workspace_id: str) -> str:
        return f"{self.base_url}/api/v2/workspaces/{urllib.parse.quote(workspace_id, safe='')}/builds"

    def _ensure_workspace(self) -> dict:
        payload = _find_workspace_by_name(
            base_url=self.base_url,
            workspace_name=self.workspace,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        if payload is None:
            payload = _create_workspace(
                base_url=self.base_url,
                workspace_name=self.workspace,
                template_name=self.template_name,
                api_key=self.api_key,
                organization_name=self.organization_name,
                timeout=self.timeout,
            )
        workspace_id = payload.get("id") if isinstance(payload, dict) else None
        if not workspace_id:
            raise RuntimeError(f"Coder workspace {self.workspace!r} did not include a workspace id")
        self._workspace_id = workspace_id
        return payload

    def _get_workspace_payload(self) -> dict:
        self._ensure_workspace()
        response = requests.get(
            self._workspace_url(),
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected workspace payload for Coder workspace {self.workspace!r}")
        return payload

    def _start_workspace(self, workspace_id: str) -> str:
        response = requests.post(
            self._workspace_builds_url(workspace_id),
            headers=self._headers(),
            json={"transition": "start"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        build_id = payload.get("id")
        if not build_id:
            raise RuntimeError(f"Coder start build for workspace {self.workspace!r} did not return a build id")
        return build_id

    def _wait_for_build_completion(self, build_id: str) -> None:
        deadline = time.time() + max(self.timeout, 300)
        while time.time() < deadline:
            response = requests.get(
                self._workspace_build_url(build_id),
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            job = payload.get("job") or {}
            status = (job.get("status") or "").lower()
            if job.get("completed_at"):
                if status != "succeeded":
                    raise RuntimeError(
                        f"Coder workspace build {build_id} finished with status {status or 'unknown'}"
                    )
                return
            time.sleep(2)
        raise TimeoutError(f"Timed out waiting for Coder workspace build {build_id} to complete")

    @staticmethod
    def _agent_startup_ready(agent: dict) -> bool:
        """Return True when an agent is ready for PTY execution."""
        agent_status = str(agent.get("status") or "").strip().lower()
        if agent_status != "connected":
            return False

        lifecycle_state = str(agent.get("lifecycle_state") or "").strip().lower()
        if lifecycle_state != "ready":
            return False

        return True

    def _wait_for_agent_ready(self, payload: dict) -> dict:
        deadline = time.time() + max(self.timeout, 120)
        while True:
            latest_build = payload.get("latest_build") or {}
            resources = latest_build.get("resources") or []
            for resource in resources:
                for agent in resource.get("agents") or []:
                    agent_id = agent.get("id")
                    if not agent_id:
                        continue
                    if self._agent_startup_ready(agent):
                        return agent

            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Coder workspace agent startup for {self.workspace!r}"
                )

            time.sleep(2)
            payload = self._get_workspace_payload()

    def _resolve_agent_id(self) -> str:
        payload = self._get_workspace_payload()
        latest_build = payload.get("latest_build") or {}
        transition = (latest_build.get("transition") or "").lower()

        if transition != "start":
            if transition == "delete":
                raise RuntimeError(f"Coder workspace {self.workspace!r} is deleted")
            if (latest_build.get("status") or "").lower() != "stopped":
                raise RuntimeError(
                    f"Coder workspace {self.workspace!r} must be started before terminal execution"
                )
            workspace_id = payload.get("id")
            if not workspace_id:
                raise RuntimeError(f"Coder workspace {self.workspace!r} did not include a workspace id")
            build_id = self._start_workspace(workspace_id)
            self._wait_for_build_completion(build_id)
            payload = self._get_workspace_payload()
        else:
            job = latest_build.get("job") or {}
            if latest_build.get("id") and not job.get("completed_at"):
                self._wait_for_build_completion(latest_build["id"])
                payload = self._get_workspace_payload()

        agent = self._wait_for_agent_ready(payload)
        agent_id = agent.get("id")
        if agent_id:
            return agent_id
        raise RuntimeError(f"No workspace agent found for Coder workspace {self.workspace!r}")

    @staticmethod
    def _exit_marker(reconnect_id: str) -> str:
        return f"__HERMES_EXIT_{reconnect_id}__"

    @staticmethod
    def _exit_marker_match(output: str, exit_marker: str) -> re.Match[str] | None:
        pattern = re.compile(
            rf"(?:\r?\n)?{re.escape(exit_marker)}(\d{{1,3}}){re.escape(exit_marker)}\r?\n?"
        )
        matches = list(pattern.finditer(output))
        return matches[-1] if matches else None

    @classmethod
    def _has_exit_marker(cls, output: str, exit_marker: str) -> bool:
        return cls._exit_marker_match(output, exit_marker) is not None

    @classmethod
    def _extract_exit_code(cls, output: str, exit_marker: str) -> tuple[str, int]:
        match = cls._exit_marker_match(output, exit_marker)
        if match is None:
            logger.error("[coder] PTY exit marker missing; treating command as failed")
            return output, 1

        exit_code = int(match.group(1))
        if not 0 <= exit_code <= 255:
            exit_code = 1
        cleaned = output[: match.start()] + output[match.end() :]
        return cleaned, exit_code

    def _pty_command(self, cmd_string: str, *, login: bool, exit_marker: str) -> str:
        inner_shell_flag = "-lc" if login else "-c"
        capture_script = "\n".join(
            [
                f"bash {inner_shell_flag} {shlex.quote(cmd_string)}",
                "__coder_ec=$?",
                f"printf '\\n{exit_marker}%s{exit_marker}\\n' \"$__coder_ec\"",
                "exit \"$__coder_ec\"",
            ]
        )
        return f"bash -c {shlex.quote(capture_script)}"

    def _build_init_env_exports(self) -> str:
        """Build shell exports that seed forwarded env vars into the snapshot."""
        env = collect_forwarded_env_values(self._forward_env, config_name="coder_forward_env")
        if not env:
            return ""
        return "\n".join(
            f"export {key}={shlex.quote(value)}"
            for key, value in sorted(env.items())
        )

    def _pty_url(self, agent_id: str, *, command: str, reconnect_id: str) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        query = urllib.parse.urlencode(
            {
                "reconnect": reconnect_id,
                "command": command,
                "height": 80,
                "width": 80,
            }
        )
        return urllib.parse.urlunparse(
            (
                scheme,
                parsed.netloc,
                f"/api/v2/workspaceagents/{agent_id}/pty",
                "",
                query,
                "",
            )
        )

    @classmethod
    def _stdin_frame(cls, data: str) -> bytes:
        return json.dumps({"data": data}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @classmethod
    def _send_stdin_data(cls, websocket: ClientConnection, stdin_data: str) -> None:
        if not stdin_data:
            return
        chunk_size = max(1, cls._STDIN_CHUNK_SIZE)
        for start in range(0, len(stdin_data), chunk_size):
            chunk = stdin_data[start : start + chunk_size]
            frame = cls._stdin_frame(chunk)
            websocket.send(frame)

    @classmethod
    def _send_stdin_eof(cls, websocket: ClientConnection) -> None:
        # EOT / Ctrl+D signals EOF for stdin-driven commands.
        frame = cls._stdin_frame("\u0004")
        websocket.send(frame)

    @classmethod
    def _interrupt_pty(cls, websocket) -> None:
        """Send Ctrl+C to a Coder PTY websocket and close it.

        Coder PTY expects binary WebSocket frames that carry JSON payloads.
        Interrupt is ETX (0x03) in the "data" field.
        """
        try:
            frame = cls._stdin_frame("\u0003")
            websocket.send(frame)
        except Exception:
            pass
        try:
            websocket.close()
        except Exception:
            pass

    @staticmethod
    def _cancel_requested(cancel_state: dict | None) -> bool:
        if cancel_state is None:
            return False
        with cancel_state["lock"]:
            return bool(cancel_state.get("cancelled"))

    def _execute_via_pty(
        self,
        cmd_string: str,
        *,
        login: bool,
        timeout: int,
        stdin_data: str | None = None,
        cancel_state: dict | None = None,
    ) -> tuple[str, int]:
        agent_id = self._resolve_agent_id()
        reconnect_id = str(uuid.uuid4())
        exit_marker = self._exit_marker(reconnect_id)
        pty_command = self._pty_command(cmd_string, login=login, exit_marker=exit_marker)
        pty_url = self._pty_url(agent_id, command=pty_command, reconnect_id=reconnect_id)
        output_parts: list[str] = []
        recv_poll_timeout = max(0.1, min(self._PTY_RECV_POLL_TIMEOUT, float(timeout)))
        max_empty_reconnects = self._PTY_EMPTY_EOF_RECONNECTS
        empty_reconnects = 0
        stdin_send_attempted = False

        while True:
            attempt_started = time.monotonic()
            attempt_start_chars = sum(len(part) for part in output_parts)

            with connect(
                pty_url,
                additional_headers=self._headers(),
                open_timeout=timeout,
                close_timeout=1,
            ) as websocket:
                if cancel_state is not None:
                    with cancel_state["lock"]:
                        cancel_state["websocket"] = websocket
                if self._cancel_requested(cancel_state):
                    self._interrupt_pty(websocket)

                if stdin_data is not None and not stdin_send_attempted:
                    # The reconnect id resumes the same PTY session.  Stdin
                    # must be forwarded at most once locally; resending on a
                    # reconnect can duplicate file writes or command input.
                    stdin_send_attempted = True
                    self._send_stdin_data(websocket, stdin_data)
                    self._send_stdin_eof(websocket)

                try:
                    while True:
                        try:
                            message = websocket.recv(timeout=recv_poll_timeout, decode=False)
                        except TimeoutError:
                            continue
                        except EOFError:
                            break
                        except ConnectionClosed:
                            break

                        if isinstance(message, bytes):
                            decoded = message.decode("utf-8", errors="replace")
                            output_parts.append(decoded)
                        else:
                            output_parts.append(message)
                finally:
                    if cancel_state is not None:
                        with cancel_state["lock"]:
                            if cancel_state.get("websocket") is websocket:
                                cancel_state["websocket"] = None

            combined_output = "".join(output_parts)
            if self._has_exit_marker(combined_output, exit_marker) or self._cancel_requested(cancel_state):
                break

            attempt_output_chars = len(combined_output) - attempt_start_chars
            attempt_elapsed = time.monotonic() - attempt_started
            if (
                attempt_output_chars == 0
                and attempt_elapsed <= self._PTY_EMPTY_EOF_RECONNECT_WINDOW
                and empty_reconnects < max_empty_reconnects
            ):
                empty_reconnects += 1
                logger.warning(
                    "[coder] PTY closed before output/marker; reconnecting same session: "
                    "workspace=%s reconnect_id=%s reconnect_attempt=%s elapsed_ms=%.1f",
                    self.workspace,
                    reconnect_id,
                    empty_reconnects,
                    attempt_elapsed * 1000,
                )
                time.sleep(self._PTY_EMPTY_EOF_RECONNECT_DELAY * empty_reconnects)
                continue

            logger.info("[coder] Reconnection break: attempt_output_chars=%s attempt_elapsed=%.1f empty_reconnects=%s", attempt_output_chars, attempt_elapsed, empty_reconnects)
            break

        combined_output = "".join(output_parts)
        cleaned_output, exit_code = self._extract_exit_code(combined_output, exit_marker)
        # Workaround: \r\n -> \n for pty
        cleaned_output = cleaned_output.replace("\r\n", "\n")
        return cleaned_output, exit_code

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        if login:
            exports = self._build_init_env_exports()
            if exports:
                cmd_string = f"{exports}\n{cmd_string}"

        cancel_state = {"lock": threading.Lock(), "websocket": None, "cancelled": False}

        def cancel_pty() -> None:
            websocket = None
            with cancel_state["lock"]:
                cancel_state["cancelled"] = True
                websocket = cancel_state.get("websocket")
            if websocket is not None:
                self._interrupt_pty(websocket)

        return _ThreadedProcessHandle(
            lambda: self._execute_via_pty(
                cmd_string,
                login=login,
                timeout=timeout,
                stdin_data=stdin_data,
                cancel_state=cancel_state,
            ),
            cancel_fn=cancel_pty,
        )

    def cleanup(self):
        return None
