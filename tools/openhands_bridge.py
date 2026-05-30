"""Experimental OpenHands worker bridge for Dev runtime routing.

Phase 18 keeps this adapter deliberately small. It discovers whether a local
OpenHands SDK/CLI/server exists, but only launches through a configured agent
server URL so Hermes does not implicitly install packages, start Docker, or
take ownership of OpenHands runtime setup.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

from gateway.status import _pid_exists, terminate_pid


DEFAULT_SERVER_PORT = 3000
DEFAULT_SERVER_URL = f"http://127.0.0.1:{DEFAULT_SERVER_PORT}"
INSTALL_INSTRUCTION = "uv tool install openhands --python 3.12"
DEFAULT_WORKSPACE_DIR = str(Path(__file__).resolve().parents[2])
SERVER_URL_ENV_NAMES = (
    "OPENHANDS_AGENT_SERVER_URL",
    "OPENHANDS_SERVER_URL",
    "OPENHANDS_BASE_URL",
)


TERMINAL_STATUSES = {"finished", "completed", "done", "error", "failed", "stuck", "killed", "terminated"}
FAILED_STATUSES = {"error", "failed", "stuck", "killed", "terminated"}


class OpenHandsBridgeError(RuntimeError):
    pass


@dataclass
class OpenHandsSession:
    id: str
    project_id: Optional[str] = None
    status: Optional[str] = None
    workspace_path: Optional[str] = None
    branch: Optional[str] = None
    agent: Optional[str] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    summary: Optional[str] = None
    output_tail: Optional[str] = None
    open_command: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "OpenHandsSession":
        state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
        workspace = payload.get("workspace") if isinstance(payload.get("workspace"), dict) else {}
        agent = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
        llm = agent.get("llm") if isinstance(agent.get("llm"), dict) else {}
        tags = payload.get("tags") if isinstance(payload.get("tags"), dict) else {}
        session_id = (
            payload.get("id")
            or payload.get("conversation_id")
            or payload.get("conversationId")
            or state.get("id")
        )
        return cls(
            id=str(session_id or ""),
            project_id=payload.get("project_id") or payload.get("projectId") or workspace.get("project_id") or tags.get("project"),
            status=payload.get("status") or payload.get("execution_status") or payload.get("executionStatus") or state.get("execution_status"),
            workspace_path=payload.get("workspace_path") or payload.get("workspacePath") or workspace.get("path") or workspace.get("working_dir"),
            branch=payload.get("branch") or tags.get("branch"),
            agent=payload.get("agent_name") or payload.get("agentName") or agent.get("name") or "openhands",
            model=payload.get("model") or llm.get("model"),
            reasoning_effort=payload.get("reasoning_effort") or payload.get("reasoningEffort"),
            summary=payload.get("summary"),
            output_tail=payload.get("output_tail") or payload.get("outputTail"),
            open_command=payload.get("open_command") or payload.get("openCommand"),
        )

    @property
    def display_status(self) -> str:
        status = (self.status or "").lower()
        if status in FAILED_STATUSES:
            return "failed"
        if status in {"finished", "completed", "done"}:
            return "completed"
        return "running"

    @property
    def is_terminal(self) -> bool:
        return (self.status or "").lower() in TERMINAL_STATUSES

    def event_fields(self) -> Dict[str, Any]:
        return {
            "runtime": "openhands",
            "runtime_session_id": self.id,
            "runtime_project_id": self.project_id,
            "workspace_path": self.workspace_path,
            "branch": self.branch,
            "open_command": self.open_command,
            "agent": self.agent,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "status": self.display_status,
            "summary": self.summary,
            "output_tail": self.output_tail,
        }


class OpenHandsBridge:
    def __init__(
        self,
        *,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        command: Optional[str] = None,
        timeout_seconds: float = 3.0,
    ):
        self.server_url = (server_url or self._env_server_url() or self._managed_server_url() or "").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENHANDS_API_KEY")
        self.command = command or shutil.which(os.getenv("OPENHANDS_COMMAND") or "openhands")
        self.timeout_seconds = timeout_seconds
        self.sdk_available = importlib.util.find_spec("openhands") is not None
        self.configured_mode = self._configured_mode()
        self._health_cache: Optional[Dict[str, Any]] = None

    def discovery(self) -> Dict[str, Any]:
        server_configured = bool(self.server_url)
        server_healthy = False
        server_warning = None
        if server_configured:
            health = self.runtime_health(None)
            server_healthy = health.get("runtime_health") == "ok"
            server_warning = health.get("runtime_warning")
        detected = bool(server_configured or self.sdk_available or self.command)
        setup_warning = None
        if server_configured and not server_healthy:
            setup_warning = server_warning or "Configured OpenHands agent server is not healthy."
        elif not server_configured:
            if self.sdk_available or self.command:
                setup_warning = (
                    "OpenHands SDK/CLI detected, but Phase 18 launch requires "
                    "OPENHANDS_AGENT_SERVER_URL pointing at a reachable agent server."
                )
            else:
                setup_warning = (
                    "OpenHands is not installed or configured. Set OPENHANDS_AGENT_SERVER_URL "
                    "to enable the experimental runtime."
                )
        return {
            "available": detected,
            "launch_supported": server_configured and server_healthy,
            "configured_mode": "server" if self.server_url else self.configured_mode,
            "setup_warning": setup_warning,
            "server_url": self.server_url or None,
            "sdk_available": self.sdk_available,
            "command": self.command,
        }

    def spawn(
        self,
        *,
        project_id: str,
        prompt: str,
        issue_id: Optional[str] = None,
        branch: Optional[str] = None,
        agent: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> OpenHandsSession:
        if not self.server_url:
            raise OpenHandsBridgeError(self.discovery().get("setup_warning") or "OpenHands runtime is unavailable.")
        health = self.runtime_health(None)
        if health.get("runtime_health") != "ok":
            raise OpenHandsBridgeError(health.get("runtime_warning") or "OpenHands runtime is unavailable.")
        payload = {
            "workspace": {
                "working_dir": project_id if project_id.startswith("/") else DEFAULT_WORKSPACE_DIR,
                "kind": "LocalWorkspace",
            },
            "agent": {
                "llm": {
                    "model": model or os.getenv("OPENHANDS_MODEL") or "gpt-4o-mini",
                    "api_key": (
                        os.getenv("OPENROUTER_API_KEY")
                        or os.getenv("OPENAI_API_KEY")
                        or os.getenv("ANTHROPIC_API_KEY")
                        or os.getenv("LLM_API_KEY")
                    ),
                    "max_input_tokens": 128000,
                    "max_output_tokens": 16384,
                },
                "system_prompt_kwargs": {"llm_security_analyzer": True},
                "kind": "Agent",
            },
            "initial_message": {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
                "run": True,
            },
            "max_iterations": 5,
            "autotitle": False,
            "tags": {
                "project": project_id,
                "createdby": "hermes",
                **({"issue": issue_id} if issue_id else {}),
                **({"branch": branch} if branch else {}),
            },
            "metadata": {
                "issue_id": issue_id,
                "branch": branch,
                "reasoning_effort": reasoning_effort,
                "created_by": "hermes",
            },
        }
        conversation = self._request("POST", "/conversations", payload)
        session = OpenHandsSession.from_payload(conversation.get("conversation") or conversation)
        if not session.id:
            raise OpenHandsBridgeError("OpenHands server did not return a conversation id.")
        session.project_id = session.project_id or project_id
        session.branch = session.branch or branch
        session.agent = session.agent or agent or "openhands"
        session.model = session.model or model
        session.reasoning_effort = session.reasoning_effort or reasoning_effort
        session.open_command = session.open_command or f"{self.server_url}/conversations/{session.id}"
        return session

    def status(self, session_id: str) -> Optional[OpenHandsSession]:
        if not self.server_url:
            return None
        try:
            payload = self._request("GET", f"/conversations/{session_id}")
        except OpenHandsBridgeError:
            return None
        session_payload = payload.get("conversation") or payload
        session = OpenHandsSession.from_payload(session_payload)
        if session.id and session.is_terminal:
            final_response = self._agent_final_response(session.id)
            if final_response and not session.summary:
                session.summary = final_response
            output_tail = self.capture_output(session, lines=80)
            if output_tail:
                session.output_tail = output_tail
        return session if session.id else None

    def list(self, project_id: Optional[str] = None) -> list[OpenHandsSession]:
        if not self.server_url:
            return []
        try:
            payload = self._request("GET", "/conversations")
        except OpenHandsBridgeError:
            return []
        raw_items = payload if isinstance(payload, list) else (payload.get("conversations") or payload.get("items") or payload.get("data") or [])
        sessions = [OpenHandsSession.from_payload(item) for item in raw_items if isinstance(item, dict)]
        if project_id:
            sessions = [session for session in sessions if session.project_id in {None, project_id}]
        return [session for session in sessions if session.id]

    def send(self, session_id: str, message: str) -> Optional[OpenHandsSession]:
        if not self.server_url:
            raise OpenHandsBridgeError("OpenHands runtime is unavailable.")
        self._request(
            "POST",
            f"/conversations/{session_id}/events",
            {"role": "user", "content": [{"type": "text", "text": message}], "run": True},
        )
        return self.status(session_id)

    def kill(self, session_id: str, **kwargs: Any) -> None:
        if not self.server_url:
            return None
        try:
            self._request("DELETE", f"/conversations/{session_id}")
        except OpenHandsBridgeError:
            return None
        return None

    def capture_output(self, session: Optional[OpenHandsSession], lines: int = 40) -> str:
        if session is None:
            return ""
        if session.output_tail:
            return session.output_tail
        if not self.server_url:
            return ""
        try:
            payload = self._request("GET", f"/conversations/{session.id}/events/search?limit={max(1, int(lines))}")
        except OpenHandsBridgeError:
            return ""
        events = payload.get("events") or payload.get("items") or payload.get("data") or []
        text_lines: list[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            content = self._event_text(event)
            if not content and event.get("kind") == "ConversationErrorEvent":
                code = event.get("code") or "OpenHandsError"
                detail = event.get("detail") or ""
                content = f"{code}: {detail}".strip()
            if not content and event.get("kind") == "ConversationStateUpdateEvent":
                content = f"{event.get('key')}: {event.get('value')}"
            if content:
                text_lines.extend(str(content).splitlines())
        final_response = self._agent_final_response(session.id)
        if final_response:
            text_lines.extend(final_response.splitlines())
        return "\n".join(text_lines[-max(1, int(lines)):])

    def _agent_final_response(self, session_id: str) -> str:
        try:
            payload = self._request("GET", f"/conversations/{session_id}/agent_final_response")
        except OpenHandsBridgeError:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            for key in ("final_response", "response", "message", "content", "text", "summary"):
                value = payload.get(key)
                text = self._content_text(value)
                if text:
                    return text
        return ""

    def _event_text(self, event: Dict[str, Any]) -> str:
        for key in ("message", "content", "text"):
            content = self._content_text(event.get(key))
            if content:
                return content
        if isinstance(event.get("llm_message"), dict):
            content = self._content_text(event["llm_message"].get("content"))
            if content:
                return content
        if isinstance(event.get("action"), dict):
            action = event["action"]
            content = self._content_text(action.get("message") or action.get("summary"))
            if content:
                return content
        if isinstance(event.get("observation"), dict):
            content = self._content_text(event["observation"].get("content"))
            if content:
                return content
        return ""

    @staticmethod
    def _content_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("message")
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "message", "summary"):
                text = value.get(key)
                if text:
                    return str(text).strip()
        return str(value).strip()

    def runtime_health(self, session: Optional[OpenHandsSession]) -> Dict[str, Any]:
        if self._health_cache is not None:
            return dict(self._health_cache)
        if not self.server_url:
            warning = None
            if not (self.sdk_available or self.command):
                warning = "OpenHands is not installed or configured."
            self._health_cache = {
                "runtime_health": "unavailable",
                "runtime_warning": warning,
                "configured_mode": self.configured_mode,
            }
            return dict(self._health_cache)
        try:
            self._request("GET", "/health", timeout=self.timeout_seconds)
            info = self._request("GET", "/server_info", timeout=self.timeout_seconds)
            title = str(info.get("title") or "")
            if "OpenHands" not in title:
                raise OpenHandsBridgeError("Server is healthy but is not an OpenHands agent server.")
            self._health_cache = {
                "runtime_health": "ok",
                "runtime_warning": None,
                "configured_mode": "server",
            }
        except Exception as exc:
            self._health_cache = {
                "runtime_health": "unavailable",
                "runtime_warning": f"OpenHands agent server is unreachable: {exc}",
                "configured_mode": "server",
            }
        return dict(self._health_cache)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.server_url:
            raise OpenHandsBridgeError("OPENHANDS_AGENT_SERVER_URL is not configured.")
        url = urljoin(f"{self.server_url}/", path.lstrip("/"))
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=timeout or 30) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 and path.startswith("/conversations"):
                return self._request(method, f"/api{path}", payload, timeout=timeout)
            raise OpenHandsBridgeError(f"OpenHands server returned HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise OpenHandsBridgeError(str(exc)) from exc
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenHandsBridgeError(f"OpenHands server returned invalid JSON: {raw[:200]}") from exc

    @staticmethod
    def _env_server_url() -> Optional[str]:
        for name in SERVER_URL_ENV_NAMES:
            value = os.getenv(name)
            if value:
                return value
        return None

    @staticmethod
    def _managed_server_url() -> Optional[str]:
        state = read_openhands_server_metadata()
        server_url = state.get("server_url") or DEFAULT_SERVER_URL
        if _pid_running(state.get("pid")):
            return server_url
        if state and OpenHandsBridge(server_url=server_url).runtime_health(None).get("runtime_health") == "ok":
            return server_url
        return None

    def _configured_mode(self) -> str:
        if self.server_url:
            return "server"
        if self.sdk_available:
            return "sdk"
        if self.command:
            return "cli"
        return "missing"


def openhands_server_status(
    *,
    metadata_path: Optional[Path] = None,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = read_openhands_server_metadata(metadata_path=metadata_path)
    env_url = OpenHandsBridge._env_server_url()
    resolved_command = command or shutil.which(os.getenv("OPENHANDS_COMMAND") or "openhands")
    install = INSTALL_INSTRUCTION

    if env_url:
        health = OpenHandsBridge(server_url=env_url).runtime_health(None)
        status = "external_running" if health.get("runtime_health") == "ok" else "external_unreachable"
        return {
            "ok": status == "external_running",
            "object": "hermes.dev_openhands_server",
            "status": status,
            "server_url": env_url,
            "source": "env",
            "command": resolved_command,
            "install_instruction": install,
            "runtime_health": health.get("runtime_health"),
            "runtime_warning": health.get("runtime_warning"),
        }

    pid = metadata.get("pid")
    server_url = metadata.get("server_url") or DEFAULT_SERVER_URL
    if pid:
        running = _pid_running(pid)
        health = OpenHandsBridge(server_url=server_url).runtime_health(None)
        if health.get("runtime_health") == "ok":
            status = "running"
        elif running:
            status = "starting"
        else:
            status = "stale"
        return {
            "ok": status == "running",
            "object": "hermes.dev_openhands_server",
            "status": status,
            "server_url": server_url,
            "source": "managed",
            "pid": pid,
            "cwd": metadata.get("cwd"),
            "log_path": metadata.get("log_path"),
            "started_at": metadata.get("started_at"),
            "command": metadata.get("command") or resolved_command,
            "install_instruction": install,
            "runtime_health": health.get("runtime_health"),
            "runtime_warning": health.get("runtime_warning"),
        }

    if not resolved_command:
        return {
            "ok": False,
            "object": "hermes.dev_openhands_server",
            "status": "missing_cli",
            "server_url": None,
            "source": "missing",
            "command": None,
            "install_instruction": install,
            "runtime_health": "unavailable",
            "runtime_warning": f"OpenHands CLI is not installed. Install with: {install}",
        }

    return {
        "ok": False,
        "object": "hermes.dev_openhands_server",
        "status": "stopped",
        "server_url": None,
        "source": "local",
        "command": resolved_command,
        "install_instruction": install,
        "runtime_health": "stopped",
        "runtime_warning": None,
    }


def start_openhands_server(
    *,
    cwd: Optional[str] = None,
    server_url: Optional[str] = None,
    metadata_path: Optional[Path] = None,
    command: Optional[str] = None,
    wait_seconds: float = 5.0,
) -> Dict[str, Any]:
    current = openhands_server_status(metadata_path=metadata_path, command=command)
    if current.get("status") in {"running", "starting", "external_running"}:
        current["message"] = "OpenHands server is already configured or running."
        return current
    resolved_command = command or shutil.which(os.getenv("OPENHANDS_COMMAND") or "openhands")
    if not resolved_command:
        current["message"] = f"OpenHands CLI is not installed. Install with: {INSTALL_INSTRUCTION}"
        return current

    workspace_dir = str(Path(cwd or os.getenv("OPENHANDS_WORKSPACE_DIR") or DEFAULT_WORKSPACE_DIR).expanduser())
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    runtime_dir = _openhands_runtime_dir(metadata_path=metadata_path)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "server.log"
    port = _server_port_from_url(server_url) if server_url else _find_available_port(DEFAULT_SERVER_PORT)
    url = (server_url or f"http://127.0.0.1:{port}").rstrip("/")
    python_executable = _python_executable_for_command(resolved_command)
    argv = [
        python_executable,
        "-m",
        "openhands.agent_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    log_handle = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            argv,
            cwd=workspace_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    metadata = {
        "pid": proc.pid,
        "server_url": url,
        "cwd": workspace_dir,
        "log_path": str(log_path),
        "started_at": time.time(),
        "command": resolved_command,
        "argv": argv,
        "server_mode": "agent_server",
    }
    write_openhands_server_metadata(metadata, metadata_path=metadata_path)

    deadline = time.time() + max(0.0, float(wait_seconds))
    status = openhands_server_status(metadata_path=metadata_path, command=command)
    while time.time() < deadline and status.get("status") != "running":
        time.sleep(0.5)
        status = openhands_server_status(metadata_path=metadata_path, command=command)
    status["message"] = (
        "OpenHands server is running."
        if status.get("status") == "running"
        else "OpenHands server process started; waiting for health check."
    )
    return status


def stop_openhands_server(*, metadata_path: Optional[Path] = None) -> Dict[str, Any]:
    metadata = read_openhands_server_metadata(metadata_path=metadata_path)
    pid = metadata.get("pid")
    if not pid:
        return {
            "ok": True,
            "object": "hermes.dev_openhands_server",
            "status": "stopped",
            "message": "No managed OpenHands server metadata found.",
        }
    if _pid_running(pid):
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(int(pid)), signal.SIGINT)  # windows-footgun: ok — POSIX-only process group signal
            else:
                terminate_pid(int(pid))
        except Exception:
            try:
                terminate_pid(int(pid))
            except Exception:
                pass
        deadline = time.time() + 8.0
        while time.time() < deadline and _pid_running(pid):
            time.sleep(0.25)
        if _pid_running(pid):
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(int(pid)), signal.SIGTERM)  # windows-footgun: ok — POSIX-only process group signal
                else:
                    terminate_pid(int(pid), force=True)
            except Exception:
                try:
                    terminate_pid(int(pid), force=True)
                except Exception:
                    pass
    clear_openhands_server_metadata(metadata_path=metadata_path)
    return {
        "ok": True,
        "object": "hermes.dev_openhands_server",
        "status": "stopped",
        "server_url": metadata.get("server_url"),
        "pid": pid,
        "message": "Managed OpenHands server stopped.",
    }


def read_openhands_server_metadata(*, metadata_path: Optional[Path] = None) -> Dict[str, Any]:
    path = _openhands_metadata_path(metadata_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_openhands_server_metadata(metadata: Dict[str, Any], *, metadata_path: Optional[Path] = None) -> None:
    path = _openhands_metadata_path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_openhands_server_metadata(*, metadata_path: Optional[Path] = None) -> None:
    path = _openhands_metadata_path(metadata_path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _openhands_runtime_dir(*, metadata_path: Optional[Path] = None) -> Path:
    if metadata_path:
        return _openhands_metadata_path(metadata_path).parent
    from hermes_state import DEFAULT_DB_PATH

    return DEFAULT_DB_PATH.parent / "runtime" / "openhands"


def _openhands_metadata_path(metadata_path: Optional[Path] = None) -> Path:
    if metadata_path:
        return Path(metadata_path)
    return _openhands_runtime_dir() / "server.json"


def _python_executable_for_command(command: str) -> str:
    try:
        first_line = Path(command).read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return sys.executable
    if first_line.startswith("#!"):
        executable = first_line[2:].strip().split(" ")[0]
        if executable:
            return executable
    return "python3"


def _find_available_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OpenHandsBridgeError(f"No available local port found starting at {start}.")


def _server_port_from_url(server_url: Optional[str]) -> int:
    if not server_url:
        return DEFAULT_SERVER_PORT
    try:
        from urllib.parse import urlparse

        return int(urlparse(server_url).port or DEFAULT_SERVER_PORT)
    except Exception:
        return DEFAULT_SERVER_PORT


def _pid_running(pid: Any) -> bool:
    try:
        return _pid_exists(int(pid))
    except Exception:
        return False
