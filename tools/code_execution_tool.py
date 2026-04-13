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
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

_IS_WINDOWS = platform.system() == "Windows"
from typing import Any, Dict, List, Optional

# Availability gate: UDS requires a POSIX OS
logger = logging.getLogger(__name__)

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
        '"""Extract content from URLs. Returns dict with results list of {url, title, content, error}."""',
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
        'pattern: str, target: str = "content", path: str = ".", file_glob: str = None, limit: int = 50, offset: int = 0, output_mode: str = "content", context: int = 0',
        '"""Search file contents (target="content") or find files by name (target="files"). Returns dict with "matches"."""',
        '{"pattern": pattern, "target": target, "path": path, "file_glob": file_glob, "limit": limit, "offset": offset, "output_mode": output_mode, "context": context}',
    ),
    "patch": (
        "patch",
        'path: str = None, old_string: str = None, new_string: str = None, replace_all: bool = False, mode: str = "replace", patch: str = None',
        '"""Targeted find-and-replace (mode="replace") or V4A multi-file patches (mode="patch"). Returns dict with status."""',
        '{"path": path, "old_string": old_string, "new_string": new_string, "replace_all": replace_all, "mode": mode, "patch": patch}',
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
import json, os, socket, shlex, time

_sock = None


# ---------------------------------------------------------------------------
# Convenience helpers (avoid common scripting pitfalls)
# ---------------------------------------------------------------------------

def json_parse(text: str):
    """Parse JSON tolerant of control characters (strict=False).
    Use this instead of json.loads() when parsing output from terminal()
    or web_extract() that may contain raw tabs/newlines in strings."""
    return json.loads(text, strict=False)


def shell_quote(s: str) -> str:
    """Shell-escape a string for safe interpolation into commands.
    Use this when inserting dynamic content into terminal() commands:
        terminal(f"echo {shell_quote(user_input)}")
    """
    return shlex.quote(s)


def retry(fn, max_attempts=3, delay=2):
    """Retry a function up to max_attempts times with exponential backoff.
    Use for transient failures (network errors, API rate limits):
        result = retry(lambda: terminal("gh issue list ..."))
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (2 ** attempt))
    raise last_err

def _connect():
    global _sock
    if _sock is None:
        host = os.environ.get("HERMES_RPC_HOST", "127.0.0.1")
        port = int(os.environ["HERMES_RPC_PORT"])
        token = os.environ.get("HERMES_RPC_TOKEN", "")
        _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            _sock.connect((host, port))
        except ConnectionRefusedError:
            # Fallback for docker containers if host.docker.internal fails: use default gateway
            if host == "host.docker.internal":
                try:
                    gw_ip = os.popen("ip route | awk '/default/ { print $3 }'").read().strip()
                    if gw_ip:
                        _sock.connect((gw_ip, port))
                    else:
                        raise
                except Exception:
                    raise
            else:
                raise
                
        _sock.sendall((token + "\\n").encode())
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
_TERMINAL_BLOCKED_PARAMS = {"background", "check_interval", "pty", "notify_on_complete"}


def _rpc_server_loop(
    server_sock,
    task_id: str,
    tool_call_log: list,
    tool_call_counter: list,   # mutable [int] so the thread can increment
    max_tool_calls: int,
    allowed_tools: frozenset,
    auth_token: str,
):
    from model_tools import handle_function_call

    conn = None
    try:
        server_sock.settimeout(5)
        conn, _ = server_sock.accept()
        conn.settimeout(300)

        # Authenticate
        buf = b""
        import socket
        while b"\n" not in buf:
            try:
                chunk = conn.recv(1024)
            except socket.timeout:
                chunk = b""
            if not chunk: break
            buf += chunk
            
        if b"\n" in buf:
            token_line, buf = buf.split(b"\n", 1)
            if token_line.decode().strip() != auth_token:
                import logging
                logging.getLogger(__name__).warning("RPC authentication failed")
                conn.close()
                return
        else:
            conn.close()
            return

        while True:
            while b"\n" in buf:
                import json
                import time
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                call_start = time.monotonic()
                try:
                    request = json.loads(line.decode())
                except Exception as exc:
                    resp = json.dumps({"error": f"Invalid RPC request: {exc}"})
                    conn.sendall((resp + "\n").encode())
                    continue

                tool_name = request.get("tool", "")
                tool_args = request.get("args", {})

                if tool_name not in allowed_tools:
                    available = ", ".join(sorted(allowed_tools))
                    resp = json.dumps({"error": f"Tool '{tool_name}' is not available. Available: {available}"})
                    conn.sendall((resp + "\n").encode())
                    continue

                if tool_call_counter[0] >= max_tool_calls:
                    resp = json.dumps({"error": "Tool call limit reached."})
                    conn.sendall((resp + "\n").encode())
                    continue

                if tool_name == "terminal" and isinstance(tool_args, dict):
                    for param in _TERMINAL_BLOCKED_PARAMS:
                        tool_args.pop(param, None)

                try:
                    import sys, os
                    _real_stdout, _real_stderr = sys.stdout, sys.stderr
                    devnull = open(os.devnull, "w")
                    try:
                        sys.stdout = devnull
                        sys.stderr = devnull
                        result = handle_function_call(tool_name, tool_args, task_id=task_id)
                    finally:
                        sys.stdout, sys.stderr = _real_stdout, _real_stderr
                        devnull.close()
                except Exception as exc:
                    result = json.dumps({"error": str(exc)})

                tool_call_counter[0] += 1
                call_duration = time.monotonic() - call_start

                args_preview = str(tool_args)[:80]
                tool_call_log.append({
                    "tool": tool_name,
                    "args_preview": args_preview,
                    "duration": round(call_duration, 2),
                })

                conn.sendall((result + "\n").encode())

            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk

    except socket.timeout:
        pass
    except Exception:
        pass
    finally:
        if conn:
            try:
                conn.close()
            except OSError:
                pass


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

    # Determine if we are using a remote execution environment
    from tools.terminal_tool import _active_environments
    from tools.environments.local import LocalEnvironment
    terminal_env = _active_environments.get(task_id) if task_id else None
    is_sandboxed_remote = terminal_env is not None and not isinstance(terminal_env, LocalEnvironment)

    # Set up RPC
    import secrets
    import shlex
    
    auth_token = secrets.token_hex(16)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind to all interfaces so Docker containers can reach the host over the bridge network
    server_sock.bind(("0.0.0.0", 0))
    rpc_port = server_sock.getsockname()[1]
    server_sock.listen(1)

    tool_call_log: list = []
    tool_call_counter = [0]
    exec_start = time.monotonic()

    sandbox_session_id = uuid.uuid4().hex
    if is_sandboxed_remote:
        tmpdir = f"/tmp/hermes_sandbox_{sandbox_session_id}"
        rpc_host = "host.docker.internal"  # The default Mac docker bridge hostname
    else:
        tmpdir = tempfile.mkdtemp(prefix="hermes_sandbox_")
        rpc_host = "127.0.0.1"

    try:
        # Generate stubs
        tools_src = generate_hermes_tools_module(list(sandbox_tools))

        # Write files
        if is_sandboxed_remote:
            terminal_env.execute(f"mkdir -p {tmpdir}")
            # Base64 encode to safely write multiline code over command line
            import base64
            tools_b64 = base64.b64encode(tools_src.encode()).decode()
            code_b64 = base64.b64encode(code.encode()).decode()
            terminal_env.execute(f"echo {tools_b64} | base64 -d > {tmpdir}/hermes_tools.py")
            terminal_env.execute(f"echo {code_b64} | base64 -d > {tmpdir}/script.py")
        else:
            with open(os.path.join(tmpdir, "hermes_tools.py"), "w") as f:
                f.write(tools_src)
            with open(os.path.join(tmpdir, "script.py"), "w") as f:
                f.write(code)

        # Start RPC Thread
        rpc_thread = threading.Thread(
            target=_rpc_server_loop,
            args=(
                server_sock, task_id, tool_call_log,
                tool_call_counter, max_tool_calls, sandbox_tools, auth_token
            ),
            daemon=True,
        )
        rpc_thread.start()

        # Execute
        _SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                              "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                              "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
        _SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                              "PASSWD", "AUTH")
        try:
            from tools.env_passthrough import is_env_passthrough as _is_passthrough
        except Exception:
            _is_passthrough = lambda _: False  # noqa: E731
            
        child_env = {}
        for k, v in os.environ.items():
            if _is_passthrough(k):
                child_env[k] = v
                continue
            if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
                continue
            if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
                child_env[k] = v

        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        _tz_name = os.getenv("HERMES_TIMEZONE", "").strip()
        if _tz_name:
            child_env["TZ"] = _tz_name

        child_env["HERMES_RPC_PORT"] = str(rpc_port)
        child_env["HERMES_RPC_HOST"] = rpc_host
        child_env["HERMES_RPC_TOKEN"] = auth_token
        
        _hermes_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _existing_pp = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = _hermes_root + (os.pathsep + _existing_pp if _existing_pp else "")

        if is_sandboxed_remote:
            env_exports = []
            for k, v in child_env.items():
                if v is not None:
                    env_exports.append(f"export {k}={shlex.quote(str(v))}")
            env_cmd = " && ".join(env_exports)
            
            cmd = f"cd {tmpdir} && {env_cmd} && python3 script.py"
            
            # terminal_env.execute blocks and returns when finished, handling timeout internally
            res = terminal_env.execute(cmd, cwd=tmpdir, timeout=timeout)
            
            stdout_text = res.get("output", "")
            exit_code = res.get("returncode", -1)
            
            if "[Command interrupted]" in stdout_text:
                status = "interrupted"
            elif exit_code == 124 or "timeout" in stdout_text.lower():
                status = "timeout"
            elif exit_code != 0:
                status = "error"
            else:
                status = "success"
                
            duration = round(time.monotonic() - exec_start, 2)
            
            # Since terminal_env.execute blocks, we don't need real-time buffer management,
            # we just truncate the final output string here.
            total_stdout = len(stdout_text)
            if total_stdout > MAX_STDOUT_BYTES:
                omitted = total_stdout - MAX_STDOUT_BYTES
                _STDOUT_HEAD_BYTES = int(MAX_STDOUT_BYTES * 0.4)
                _STDOUT_TAIL_BYTES = MAX_STDOUT_BYTES - _STDOUT_HEAD_BYTES
                head = stdout_text[:_STDOUT_HEAD_BYTES]
                tail = stdout_text[-_STDOUT_TAIL_BYTES:]
                truncated_notice = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
                    f"out of {total_stdout:,} total] ...\n\n"
                )
                stdout_text = head + truncated_notice + tail

            # Wait for RPC thread
            server_sock.close()
            server_sock = None
            rpc_thread.join(timeout=3)
        else:
            proc = subprocess.Popen(
                [sys.executable, "script.py"],
                cwd=tmpdir,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                preexec_fn=None if _IS_WINDOWS else os.setsid,
            )

            deadline = time.monotonic() + timeout
            stderr_chunks: list = []
            _STDOUT_HEAD_BYTES = int(MAX_STDOUT_BYTES * 0.4)
            _STDOUT_TAIL_BYTES = MAX_STDOUT_BYTES - _STDOUT_HEAD_BYTES

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

            stdout_total_bytes = [0]

            def _drain_head_tail(pipe, head_chunks, tail_chunks, head_bytes, tail_bytes, total_ref):
                head_collected = 0
                from collections import deque
                tail_buf = deque()
                tail_collected = 0
                try:
                    while True:
                        data = pipe.read(4096)
                        if not data:
                            break
                        total_ref[0] += len(data)
                        if head_collected < head_bytes:
                            keep = min(len(data), head_bytes - head_collected)
                            head_chunks.append(data[:keep])
                            head_collected += keep
                            data = data[keep:]
                            if not data:
                                continue
                        tail_buf.append(data)
                        tail_collected += len(data)
                        while tail_collected > tail_bytes and tail_buf:
                            oldest = tail_buf.popleft()
                            tail_collected -= len(oldest)
                except (ValueError, OSError):
                    pass
                tail_chunks.extend(tail_buf)

            stdout_head_chunks: list = []
            stdout_tail_chunks: list = []

            stdout_reader = threading.Thread(
                target=_drain_head_tail,
                args=(proc.stdout, stdout_head_chunks, stdout_tail_chunks,
                      _STDOUT_HEAD_BYTES, _STDOUT_TAIL_BYTES, stdout_total_bytes),
                daemon=True
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

            stdout_reader.join(timeout=3)
            stderr_reader.join(timeout=3)

            stdout_head = b"".join(stdout_head_chunks).decode("utf-8", errors="replace")
            stdout_tail = b"".join(stdout_tail_chunks).decode("utf-8", errors="replace")
            stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

            total_stdout = stdout_total_bytes[0]
            if total_stdout > MAX_STDOUT_BYTES and stdout_tail:
                omitted = total_stdout - len(stdout_head) - len(stdout_tail)
                truncated_notice = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
                    f"out of {total_stdout:,} total] ...\n\n"
                )
                stdout_text = stdout_head + truncated_notice + stdout_tail
            else:
                stdout_text = stdout_head + stdout_tail

            exit_code = proc.returncode if proc.returncode is not None else -1
            duration = round(time.monotonic() - exec_start, 2)

            server_sock.close()
            server_sock = None
            rpc_thread.join(timeout=3)
            
            if exit_code != 0 and status == "success":
                status = "error"
            
            if stderr_text and status == "error":
                stdout_text = stdout_text + "\n--- stderr ---\n" + stderr_text

        # Strip ANSI escape sequences
        from tools.ansi_strip import strip_ansi
        stdout_text = strip_ansi(stdout_text)

        # Redact secrets
        from agent.redact import redact_sensitive_text
        stdout_text = redact_sensitive_text(stdout_text)

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
            if "error" not in result:
                result["error"] = f"Script exited with code {exit_code}"

        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logger.error(
            "execute_code failed after %ss with %d tool calls: %s: %s",
            duration,
            tool_call_counter[0],
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }, ensure_ascii=False)

    finally:
        if server_sock is not None:
            try:
                server_sock.close()
            except OSError as e:
                logger.debug("Server socket close error: %s", e)
        
        if is_sandboxed_remote:
            try:
                terminal_env.execute(f"rm -rf {tmpdir}")
            except Exception:
                pass
        else:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

def _kill_process_group(proc, escalate: bool = False):
    """Kill the child and its entire process group."""
    try:
        if _IS_WINDOWS:
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as e:
        logger.debug("Could not kill process group: %s", e, exc_info=True)
        try:
            proc.kill()
        except Exception as e2:
            logger.debug("Could not kill process: %s", e2, exc_info=True)

    if escalate:
        # Give the process 5s to exit after SIGTERM, then SIGKILL
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if _IS_WINDOWS:
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError) as e:
                logger.debug("Could not kill process group with SIGKILL: %s", e, exc_info=True)
                try:
                    proc.kill()
                except Exception as e2:
                    logger.debug("Could not kill process: %s", e2, exc_info=True)


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

# Per-tool documentation lines for the execute_code description.
# Ordered to match the canonical display order.
_TOOL_DOC_LINES = [
    ("web_search",
     "  web_search(query: str, limit: int = 5) -> dict\n"
     "    Returns {\"data\": {\"web\": [{\"url\", \"title\", \"description\"}, ...]}}"),
    ("web_extract",
     "  web_extract(urls: list[str]) -> dict\n"
     "    Returns {\"results\": [{\"url\", \"title\", \"content\", \"error\"}, ...]} where content is markdown"),
    ("read_file",
     "  read_file(path: str, offset: int = 1, limit: int = 500) -> dict\n"
     "    Lines are 1-indexed. Returns {\"content\": \"...\", \"total_lines\": N}"),
    ("write_file",
     "  write_file(path: str, content: str) -> dict\n"
     "    Always overwrites the entire file."),
    ("search_files",
     "  search_files(pattern: str, target=\"content\", path=\".\", file_glob=None, limit=50) -> dict\n"
     "    target: \"content\" (search inside files) or \"files\" (find files by name). Returns {\"matches\": [...]}"),
    ("patch",
     "  patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict\n"
     "    Replaces old_string with new_string in the file."),
    ("terminal",
     "  terminal(command: str, timeout=None, workdir=None) -> dict\n"
     "    Foreground only (no background/pty). Returns {\"output\": \"...\", \"exit_code\": N}"),
]


def build_execute_code_schema(enabled_sandbox_tools: set = None) -> dict:
    """Build the execute_code schema with description listing only enabled tools.

    When tools are disabled via ``hermes tools`` (e.g. web is turned off),
    the schema description should NOT mention web_search / web_extract —
    otherwise the model thinks they are available and keeps trying to use them.
    """
    if enabled_sandbox_tools is None:
        enabled_sandbox_tools = SANDBOX_ALLOWED_TOOLS

    # Build tool documentation lines for only the enabled tools
    tool_lines = "\n".join(
        doc for name, doc in _TOOL_DOC_LINES if name in enabled_sandbox_tools
    )

    # Build example import list from enabled tools
    import_examples = [n for n in ("web_search", "terminal") if n in enabled_sandbox_tools]
    if not import_examples:
        import_examples = sorted(enabled_sandbox_tools)[:2]
    if import_examples:
        import_str = ", ".join(import_examples) + ", ..."
    else:
        import_str = "..."

    description = (
        "Run a Python script that can call Hermes tools programmatically. "
        "Use this when you need 3+ tool calls with processing logic between them, "
        "need to filter/reduce large tool outputs before they enter your context, "
        "need conditional branching (if X then Y else Z), or need to loop "
        "(fetch N pages, process N files, retry on failure).\n\n"
        "Use normal tool calls instead when: single tool call with no processing, "
        "you need to see the full result and apply complex reasoning, "
        "or the task requires interactive user input.\n\n"
        f"Available via `from hermes_tools import ...`:\n\n"
        f"{tool_lines}\n\n"
        "Limits: 5-minute timeout, 50KB stdout cap, max 50 tool calls per script. "
        "terminal() is foreground-only (no background or pty). "
        "If the session uses a cloud sandbox backend, treat it as resumable task state rather than a durable always-on machine.\n\n"
        "Print your final result to stdout. Use Python stdlib (json, re, math, csv, "
        "datetime, collections, etc.) for processing between tool calls.\n\n"
        "Also available (no import needed — built into hermes_tools):\n"
        "  json_parse(text: str) — json.loads with strict=False; use for terminal() output with control chars\n"
        "  shell_quote(s: str) — shlex.quote(); use when interpolating dynamic strings into shell commands\n"
        "  retry(fn, max_attempts=3, delay=2) — retry with exponential backoff for transient failures"
    )

    return {
        "name": "execute_code",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. Import tools with "
                        f"`from hermes_tools import {import_str}` "
                        "and print your final result to stdout."
                    ),
                },
            },
            "required": ["code"],
        },
    }


# Default schema used at registration time (all sandbox tools listed)
EXECUTE_CODE_SCHEMA = build_execute_code_schema()


# --- Registry ---
from tools.registry import registry

registry.register(
    name="execute_code",
    toolset="code_execution",
    schema=EXECUTE_CODE_SCHEMA,
    handler=lambda args, **kw: execute_code(
        code=args.get("code", ""),
        task_id=kw.get("task_id"),
        enabled_tools=kw.get("enabled_tools")),
    check_fn=check_sandbox_requirements,
    emoji="🐍",
    mutates=True,
)
