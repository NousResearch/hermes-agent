#!/usr/bin/env python3
"""
Code Execution Tool -- Programmatic Tool Calling (PTC)

Lets the LLM write a Python script that calls Hermes tools via RPC,
collapsing multi-step tool chains into a single inference turn.

Architecture:
  1. Parent generates a `hermes_tools.py` stub module with RPC functions
  2. Parent opens a Unix domain socket and starts an RPC listener thread
  3. Parent spawns a child process that runs the LLM's script
  4. When the script calls a tool function, the call travels over the UDS
     back to the parent, which dispatches through handle_function_call
  5. Only the script's stdout is returned to the LLM; intermediate tool
     results never enter the context window

Platform: Linux / macOS only (Unix domain sockets). Disabled on Windows.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

# Availability gate: UDS requires a POSIX OS
SANDBOX_AVAILABLE = sys.platform != "win32"

# The 7 tools allowed inside the sandbox. The intersection of this list
# and the session's enabled tools determines which stubs are generated.
SANDBOX_ALLOWED_TOOLS = frozenset([
    "web_search",
    "web_extract",
    "read_file",
    "write_file",
    "search_files",
    "patch",
    "terminal",
])

# Resource limit defaults (overridable via config.yaml → code_execution.*)
DEFAULT_TIMEOUT = 300        # 5 minutes
DEFAULT_MAX_TOOL_CALLS = 50
MAX_STDOUT_BYTES = 50_000    # 50 KB
MAX_STDERR_BYTES = 10_000    # 10 KB


def check_sandbox_requirements() -> bool:
    """Code execution sandbox requires a POSIX OS for Unix domain sockets."""
    return SANDBOX_AVAILABLE


# ---------------------------------------------------------------------------
# hermes_tools.py code generator
# ---------------------------------------------------------------------------

# Per-tool stub templates: (function_name, signature, docstring, args_dict_expr)
# The args_dict_expr builds the JSON payload sent over the RPC socket.
_TOOL_STUBS = {
    "web_search": (
        "web_search",
        "query: str, limit: int = 5",
        '"""Search the web. Returns dict with data.web list of {url, title, description}."""',
        '{"query": query, "limit": limit}',
    ),
    "web_extract": (
        "web_extract",
        "urls: list",
        '"""Extract content from URLs. Returns dict with results list of {url, content, error}."""',
        '{"urls": urls}',
    ),
    "read_file": (
        "read_file",
        "path: str, offset: int = 1, limit: int = 500",
        '"""Read a file (1-indexed lines). Returns dict with "content" and "total_lines"."""',
        '{"path": path, "offset": offset, "limit": limit}',
    ),
    "write_file": (
        "write_file",
        "path: str, content: str",
        '"""Write content to a file (always overwrites). Returns dict with status."""',
        '{"path": path, "content": content}',
    ),
    "search_files": (
        "search_files",
        'pattern: str, target: str = "content", path: str = ".", file_glob: str = None, limit: int = 50',
        '"""Search file contents (target="content") or find files (target="files"). Returns dict with "matches"."""',
        '{"pattern": pattern, "target": target, "path": path, "file_glob": file_glob, "limit": limit}',
    ),
    "patch": (
        "patch",
        "path: str, old_string: str, new_string: str, replace_all: bool = False",
        '"""Replace old_string with new_string in a file. Returns dict with status."""',
        '{"path": path, "old_string": old_string, "new_string": new_string, "replace_all": replace_all}',
    ),
    "terminal": (
        "terminal",
        "command: str, timeout: int = None, workdir: str = None",
        '"""Run a shell command (foreground only). Returns dict with "output" and "exit_code"."""',
        '{"command": command, "timeout": timeout, "workdir": workdir}',
    ),
}


def generate_hermes_tools_module(enabled_tools: List[str]) -> str:
    """
    Build the source code for the hermes_tools.py stub module.

    Only tools in both SANDBOX_ALLOWED_TOOLS and enabled_tools get stubs.
    """
    tools_to_generate = sorted(SANDBOX_ALLOWED_TOOLS & set(enabled_tools))

    stub_functions = []
    export_names = []
    for tool_name in tools_to_generate:
        if tool_name not in _TOOL_STUBS:
            continue
        func_name, sig, doc, args_expr = _TOOL_STUBS[tool_name]
        stub_functions.append(
            f"def {func_name}({sig}):\n"
            f"    {doc}\n"
            f"    return _call({func_name!r}, {args_expr})\n"
        )
        export_names.append(func_name)

    header = '''\
"""Auto-generated Hermes tools RPC stubs."""
import json, os, socket

_sock = None

def _connect():
    global _sock
    if _sock is None:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.connect(os.environ["HERMES_RPC_SOCKET"])
        _sock.settimeout(300)
    return _sock

def _call(tool_name, args):
    """Send a tool call to the parent process and return the parsed result."""
    conn = _connect()
    request = json.dumps({"tool": tool_name, "args": args}) + "\\n"
    conn.sendall(request.encode())
    buf = b""
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            raise RuntimeError("Agent process disconnected")
        buf += chunk
        if buf.endswith(b"\\n"):
            break
    raw = buf.decode().strip()
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

'''

    return header + "\n".join(stub_functions)


# ---------------------------------------------------------------------------
# RPC server (runs in a thread inside the parent process)
# ---------------------------------------------------------------------------

# Terminal parameters that must not be used from ephemeral sandbox scripts
_TERMINAL_BLOCKED_PARAMS = {"background", "check_interval", "pty"}


def _rpc_server_loop(
    server_sock: socket.socket,
    task_id: str,
    tool_call_log: list,
    tool_call_counter: list,   # mutable [int] so the thread can increment
    max_tool_calls: int,
    allowed_tools: frozenset,
):
    """
    Accept one client connection and dispatch tool-call requests until
    the client disconnects or the call limit is reached.
    """
    from model_tools import handle_function_call

    conn = None
    try:
        server_sock.settimeout(5)
        conn, _ = server_sock.accept()
        conn.settimeout(300)

        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk

            # Process all complete newline-delimited messages in the buffer
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                call_start = time.monotonic()
                try:
                    request = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    resp = json.dumps({"error": f"Invalid RPC request: {exc}"})
                    conn.sendall((resp + "\n").encode())
                    continue

                tool_name = request.get("tool", "")
                tool_args = request.get("args", {})

                # Enforce the allow-list
                if tool_name not in allowed_tools:
                    available = ", ".join(sorted(allowed_tools))
                    resp = json.dumps({
                        "error": (
                            f"Tool '{tool_name}' is not available in execute_code. "
                            f"Available: {available}"
                        )
                    })
                    conn.sendall((resp + "\n").encode())
                    continue

                # Enforce tool call limit
                if tool_call_counter[0] >= max_tool_calls:
                    resp = json.dumps({
                        "error": (
                            f"Tool call limit reached ({max_tool_calls}). "
                            "No more tool calls allowed in this execution."
                        )
                    })
                    conn.sendall((resp + "\n").encode())
                    continue

                # Strip forbidden terminal parameters
                if tool_name == "terminal" and isinstance(tool_args, dict):
                    for param in _TERMINAL_BLOCKED_PARAMS:
                        tool_args.pop(param, None)

                # Dispatch through the standard tool handler.
                # Suppress stdout/stderr from internal tool handlers so
                # their status prints don't leak into the CLI spinner.
                try:
                    _real_stdout, _real_stderr = sys.stdout, sys.stderr
                    sys.stdout = open(os.devnull, "w")
                    sys.stderr = open(os.devnull, "w")
                    try:
                        result = handle_function_call(
                            tool_name, tool_args, task_id=task_id
                        )
                    finally:
                        sys.stdout.close()
                        sys.stderr.close()
                        sys.stdout, sys.stderr = _real_stdout, _real_stderr
                except Exception as exc:
                    result = json.dumps({"error": str(exc)})

                tool_call_counter[0] += 1
                call_duration = time.monotonic() - call_start

                # Log for observability
                args_preview = str(tool_args)[:80]
                tool_call_log.append({
                    "tool": tool_name,
                    "args_preview": args_preview,
                    "duration": round(call_duration, 2),
                })

                conn.sendall((result + "\n").encode())

    except socket.timeout:
        pass
    except OSError:
        pass
    finally:
        if conn:
            try:
                conn.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def execute_code(
    code: str,
    task_id: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
) -> str:
    """
    Run a Python script in a sandboxed child process with RPC access
    to a subset of Hermes tools.

    Args:
        code:          Python source code to execute.
        task_id:       Session task ID for tool isolation (terminal env, etc.).
        enabled_tools: Tool names enabled in the current session. The sandbox
                       gets the intersection with SANDBOX_ALLOWED_TOOLS.

    Returns:
        JSON string with execution results.
    """
    if not SANDBOX_AVAILABLE:
        return json.dumps({
            "error": "execute_code is not available on Windows. Use normal tool calls instead."
        })

    if not code or not code.strip():
        return json.dumps({"error": "No code provided."})

    # Import interrupt event from terminal_tool (cooperative cancellation)
    from tools.terminal_tool import _interrupt_event

    # Resolve config
    _cfg = _load_config()
    timeout = _cfg.get("timeout", DEFAULT_TIMEOUT)
    max_tool_calls = _cfg.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)

    # Determine which tools the sandbox can call
    session_tools = set(enabled_tools) if enabled_tools else set()
    sandbox_tools = frozenset(SANDBOX_ALLOWED_TOOLS & session_tools)

    if not sandbox_tools:
        sandbox_tools = SANDBOX_ALLOWED_TOOLS

    # --- Set up temp directory with hermes_tools.py and script.py ---
    tmpdir = tempfile.mkdtemp(prefix="hermes_sandbox_")
    sock_path = f"/tmp/hermes_rpc_{uuid.uuid4().hex}.sock"

    tool_call_log: list = []
    tool_call_counter = [0]  # mutable so the RPC thread can increment
    exec_start = time.monotonic()

    try:
        # Write the auto-generated hermes_tools module
        tools_src = generate_hermes_tools_module(
            list(sandbox_tools) if enabled_tools else list(SANDBOX_ALLOWED_TOOLS)
        )
        with open(os.path.join(tmpdir, "hermes_tools.py"), "w") as f:
            f.write(tools_src)

        # Write the user's script
        with open(os.path.join(tmpdir, "script.py"), "w") as f:
            f.write(code)

        # --- Start UDS server ---
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(sock_path)
        server_sock.listen(1)

        rpc_thread = threading.Thread(
            target=_rpc_server_loop,
            args=(
                server_sock, task_id, tool_call_log,
                tool_call_counter, max_tool_calls, sandbox_tools,
            ),
            daemon=True,
        )
        rpc_thread.start()

        # --- Spawn child process ---
        child_env = os.environ.copy()
        child_env["HERMES_RPC_SOCKET"] = sock_path
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"

        proc = subprocess.Popen(
            [sys.executable, "script.py"],
            cwd=tmpdir,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        # --- Poll loop: watch for exit, timeout, and interrupt ---
        deadline = time.monotonic() + timeout
        stdout_chunks: list = []
        stderr_chunks: list = []

        # Background readers to avoid pipe buffer deadlocks
        def _drain(pipe, chunks, max_bytes):
            total = 0
            try:
                while True:
                    data = pipe.read(4096)
                    if not data:
                        break
                    if total < max_bytes:
                        keep = max_bytes - total
                        chunks.append(data[:keep])
                    total += len(data)
            except (ValueError, OSError):
                pass

        stdout_reader = threading.Thread(
            target=_drain, args=(proc.stdout, stdout_chunks, MAX_STDOUT_BYTES), daemon=True
        )
        stderr_reader = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_chunks, MAX_STDERR_BYTES), daemon=True
        )
        stdout_reader.start()
        stderr_reader.start()

        status = "success"
        while proc.poll() is None:
            if _interrupt_event.is_set():
                _kill_process_group(proc)
                status = "interrupted"
                break
            if time.monotonic() > deadline:
                _kill_process_group(proc, escalate=True)
                status = "timeout"
                break
            time.sleep(0.2)

        # Wait for readers to finish draining
        stdout_reader.join(timeout=3)
        stderr_reader.join(timeout=3)

        stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        # Truncation notice
        if len(stdout_text) >= MAX_STDOUT_BYTES:
            stdout_text = stdout_text[:MAX_STDOUT_BYTES] + "\n[output truncated at 50KB]"

        exit_code = proc.returncode if proc.returncode is not None else -1
        duration = round(time.monotonic() - exec_start, 2)

        # Wait for RPC thread to finish
        server_sock.close()
        rpc_thread.join(timeout=3)

        # Build response
        result: Dict[str, Any] = {
            "status": status,
            "output": stdout_text,
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }

        if status == "timeout":
            result["error"] = f"Script timed out after {timeout}s and was killed."
        elif status == "interrupted":
            result["output"] = stdout_text + "\n[execution interrupted — user sent a new message]"
        elif exit_code != 0:
            result["status"] = "error"
            result["error"] = stderr_text or f"Script exited with code {exit_code}"
            # Include stderr in output so the LLM sees the traceback
            if stderr_text:
                result["output"] = stdout_text + "\n--- stderr ---\n" + stderr_text

        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logging.exception("execute_code failed")
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }, ensure_ascii=False)

    finally:
        # Cleanup temp dir and socket
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        try:
            os.unlink(sock_path)
        except OSError:
            pass


def _kill_process_group(proc, escalate: bool = False):
    """Kill the child and its entire process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:
            pass

    if escalate:
        # Give the process 5s to exit after SIGTERM, then SIGKILL
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except Exception:
                    pass


def _load_config() -> dict:
    """Load code_execution config from CLI_CONFIG if available."""
    try:
        from cli import CLI_CONFIG
        return CLI_CONFIG.get("code_execution", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

EXECUTE_CODE_SCHEMA = {
    "name": "execute_code",
    "description": (
        "Run a Python script that can call Hermes tools programmatically. "
        "Use this when you need 3+ tool calls with processing logic between them, "
        "need to filter/reduce large tool outputs before they enter your context, "
        "need conditional branching (if X then Y else Z), or need to loop "
        "(fetch N pages, process N files, retry on failure).\n\n"
        "Use normal tool calls instead when: single tool call with no processing, "
        "you need to see the full result and apply complex reasoning, "
        "or the task requires interactive user input.\n\n"
        "Available via `from hermes_tools import ...`:\n\n"
        "  web_search(query: str, limit: int = 5) -> dict\n"
        "    Returns {\"data\": {\"web\": [{\"url\", \"title\", \"description\"}, ...]}}\n"
        "  web_extract(urls: list[str]) -> dict\n"
        "    Returns {\"results\": [{\"url\", \"content\", \"error\"}, ...]} where content is markdown\n"
        "  read_file(path: str, offset: int = 1, limit: int = 500) -> dict\n"
        "    Lines are 1-indexed. Returns {\"content\": \"...\", \"total_lines\": N}\n"
        "  write_file(path: str, content: str) -> dict\n"
        "    Always overwrites the entire file.\n"
        "  search_files(pattern: str, target=\"content\", path=\".\", file_glob=None, limit=50) -> dict\n"
        "    target: \"content\" (grep) or \"files\" (find). Returns {\"matches\": [...]}\n"
        "  patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict\n"
        "    Replaces old_string with new_string in the file.\n"
        "  terminal(command: str, timeout=None, workdir=None) -> dict\n"
        "    Foreground only (no background/pty). Returns {\"output\": \"...\", \"exit_code\": N}\n\n"
        "Print your final result to stdout. Use Python stdlib (json, re, math, csv, "
        "datetime, collections, etc.) for processing between tool calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute. Import tools with "
                    "`from hermes_tools import web_search, terminal, ...` "
                    "and print your final result to stdout."
                ),
            },
        },
        "required": ["code"],
    },
}
