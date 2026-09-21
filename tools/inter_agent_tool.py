"""Inter-agent messaging tool for cross-conversation communication.

Sends, receives, and lists messages between Hermes agents running in
different conversations on the same machine. Messages persist via a
local HTTP broker (see tools/comms/broker.py).
"""
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode

import psutil

from hermes_cli._subprocess_compat import (
    windows_detach_flags_without_breakaway,
    windows_detach_popen_kwargs,
)
from hermes_constants import get_default_hermes_root, get_hermes_home, profile_name_for_home
from tools.comms import BROKER_PROTOCOL_VERSION
from tools.comms.broker import _secure_state_permissions
from tools.registry import registry, tool_error

TOOLSET = "inter_agent"

SEND_SCHEMA = {
    "type": "object",
    "properties": {
        "to": {
            "type": "string",
            "description": "Recipient agent name",
        },
        "body": {
            "type": "string",
            "description": "Message body",
        },
        "from_": {
            "type": "string",
            "description": "Sender agent name (default: current agent)",
        },
    },
    "required": ["to", "body"],
}

RECEIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "to": {
            "type": "string",
            "description": "Agent name to receive messages for",
        },
        "since": {
            "type": "integer",
            "description": "Message ID to start from (default: 0)",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum messages to return (default: 100)",
        },
    },
    "required": ["to"],
}

HISTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "with_": {
            "type": "string",
            "description": "Filter by agent name",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum messages to return (default: 100)",
        },
    },
}

_DEFAULT_HISTORY_LIMIT = 100
_BROKER_STARTUP_THREAD_LOCK = threading.Lock()


def _default_agent_address() -> str:
    """Return the stable profile address used when a call omits its agent name."""

    for env_name in ("HERMES_SESSION_PROFILE", "HERMES_PROFILE"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return profile_name_for_home(get_hermes_home()) or "default"


def _broker_process_argv() -> list[str]:
    """Launch the broker from the verified Hermes runtime, not the caller's cwd."""

    return [sys.executable, "-P", "-m", "tools.comms.broker"]


def _broker_process_env() -> dict[str, str]:
    """Allow imports only from this runtime's package root and site packages."""

    env = dict(os.environ)
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return env


def _state_path(name: str) -> Path:
    return get_default_hermes_root() / name


def _broker_token() -> str:
    home = get_default_hermes_root()
    home.mkdir(parents=True, exist_ok=True)
    _secure_state_permissions(home, directory=True)
    path = _state_path("inter-agent-broker.token")
    try:
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    except FileNotFoundError:
        pass
    token = secrets.token_urlsafe(32)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(token)
        os.replace(temporary, path)
        _secure_state_permissions(path)
    finally:
        temporary.unlink(missing_ok=True)
    return token


def _broker_endpoint() -> Optional[dict[str, object]]:
    try:
        endpoint = json.loads(
            _state_path("inter-agent-broker.json").read_text(encoding="utf-8")
        )
        port = int(endpoint["port"])
        broker_id = endpoint["broker_id"]
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ):
        return None
    pid = endpoint.get("pid")
    if (
        not 1 <= port <= 65535
        or not isinstance(broker_id, str)
        or not broker_id
        or (pid is not None and (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0))
    ):
        return None
    return {
        "url": f"http://127.0.0.1:{port}",
        "broker_id": broker_id,
        "pid": pid,
        "pid_start_time_us": endpoint.get("pid_start_time_us"),
    }


def _broker_url() -> Optional[str]:
    endpoint = _broker_endpoint()
    return endpoint["url"] if endpoint is not None else None


def _request_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_broker_token()}"}


def _broker_is_ready() -> bool:
    endpoint = _broker_endpoint()
    if endpoint is None:
        return False
    try:
        request = urllib.request.Request(
            f"{endpoint['url']}/health", headers=_request_headers()
        )
        with urllib.request.urlopen(request, timeout=0.25) as resp:
            payload = json.loads(resp.read())
            return (
                resp.status == 200
                and payload.get("ok") is True
                and payload.get("broker_id") == endpoint["broker_id"]
                and payload.get("protocol_version") == BROKER_PROTOCOL_VERSION
            )
    except Exception:
        return False


def _retire_rejected_broker(endpoint: dict[str, object]) -> None:
    """Stop the broker named by an endpoint before replacing a rejected one."""

    pid = endpoint.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return
    try:
        process = psutil.Process(pid)
        command_line = process.cmdline()
        if "tools.comms.broker" not in command_line:
            return
        expected_start = endpoint.get("pid_start_time_us")
        if expected_start is not None:
            if isinstance(expected_start, bool) or not isinstance(expected_start, int):
                return
            actual_start = int(round(process.create_time() * 1_000_000))
            if actual_start != expected_start:
                return
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.5)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, OSError, ValueError):
        return


@contextmanager
def _broker_startup_lock():
    """Serialize broker publication across threads and Hermes processes."""
    with _BROKER_STARTUP_THREAD_LOCK:
        home = get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        path = _state_path("inter-agent-broker.startup.lock")
        with path.open("a+b") as handle:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ensure_broker() -> None:
    """Start the shared loopback broker on first use, if it is not already running."""
    with _broker_startup_lock():
        if _broker_is_ready():
            return
        endpoint = _broker_endpoint()
        if endpoint is not None:
            _retire_rejected_broker(endpoint)
        argv = _broker_process_argv()
        popen_kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": _broker_process_env(),
        }
        try:
            process = subprocess.Popen(
                argv,
                **popen_kwargs,
                **windows_detach_popen_kwargs(),
            )
        except OSError:
            if os.name != "nt":
                raise
            process = subprocess.Popen(
                argv,
                **popen_kwargs,
                creationflags=windows_detach_flags_without_breakaway(),
            )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if _broker_is_ready():
                return
            time.sleep(0.05)
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _broker_call(path: str, data: Optional[dict] = None, timeout: int = 5) -> dict:
    """Make a call to the broker."""
    _ensure_broker()
    base_url = _broker_url()
    if base_url is None:
        return {"error": "broker did not publish an endpoint"}
    url = f"{base_url}{path}"
    body = None
    if data is not None:
        body = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            **_request_headers(),
            **({"Content-Type": "application/json"} if body else {}),
        },
        method="POST" if body else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def send_message(to: str, body: str, from_: str = "hermes-agent") -> str:
    """Send a message to another agent."""
    result = _broker_call("/send", {"from": from_, "to": to, "body": body})
    if "error" in result:
        return tool_error(f"Failed to send: {result['error']}")
    return json.dumps({"ok": True, "id": result.get("id")})


def receive_messages(
    to: str, since: int = 0, limit: int = _DEFAULT_HISTORY_LIMIT
) -> str:
    """Receive messages for an agent."""
    query = urlencode({"to": to, "since": since, "limit": limit})
    result = _broker_call(f"/receive?{query}", timeout=30)
    if isinstance(result, dict) and "error" in result:
        return tool_error(f"Failed to receive: {result['error']}")
    return json.dumps(result, indent=2)


def list_history(
    with_: Optional[str] = None, limit: int = _DEFAULT_HISTORY_LIMIT
) -> str:
    """List message history."""
    path = "/history"
    query = {"limit": limit}
    if with_:
        query["with_"] = with_
    path += f"?{urlencode(query)}"
    result = _broker_call(path)
    if isinstance(result, dict) and "error" in result:
        return tool_error(f"Failed to get history: {result['error']}")
    return json.dumps(result, indent=2)


def inter_agent_tool(
    action: str = "send",
    to: Optional[str] = None,
    body: Optional[str] = None,
    from_: Optional[str] = None,
    since: int = 0,
    with_: Optional[str] = None,
    limit: int = _DEFAULT_HISTORY_LIMIT,
    sender: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Inter-agent messaging: send, receive, or list history."""
    if action == "send":
        if not to or not body:
            return tool_error("send requires 'to' and 'body'")
        return send_message(to, body, from_ or _default_agent_address())
    elif action == "receive":
        return receive_messages(to or _default_agent_address(), since, limit)
    elif action == "history":
        return list_history(with_, limit)
    else:
        return tool_error(f"Unknown action: {action}")


registry.register(
    name="inter_agent",
    toolset=TOOLSET,
    schema={
        "description": "Send persistent messages between Hermes agents.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send", "receive", "history"],
                    "description": "Action to perform",
                },
                "to": {
                    "type": "string",
                    "description": "Recipient agent name (for send/receive)",
                },
                "body": {
                    "type": "string",
                    "description": "Message body (for send)",
                },
                "from_": {
                    "type": "string",
                    "description": "Sender agent name (default: hermes-agent)",
                },
                "since": {
                    "type": "integer",
                    "description": "Message ID to start from (for receive)",
                },
                "with_": {
                    "type": "string",
                    "description": "Filter by agent name (for history)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages to return (default: 100)",
                },
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kw: inter_agent_tool(
        action=args.get("action", "send"),
        to=args.get("to"),
        body=args.get("body"),
        from_=args.get("from_"),
        since=args.get("since", 0),
        with_=args.get("with_"),
        limit=args.get("limit", _DEFAULT_HISTORY_LIMIT),
        sender=kw.get("session_id") or kw.get("task_id"),
        callback=kw.get("callback"),
    ),
    emoji="💬",
)
