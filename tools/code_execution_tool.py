#!/usr/bin/env python3
"""Code Execution Tool -- Programmatic Tool Calling (PTC).

The LLM writes a Python script that calls Hermes tools via RPC, collapsing
multi-step tool chains into one inference turn; only the script's stdout returns
to the LLM. Local backend: a persistent per-conversation session kernel
(tools/code_kernel.py) over a Unix socket (loopback TCP on Windows). Remote
backends: a remote session kernel (tools/code_kernel_remote.py) falling open to a
per-call script ship, tool calls as request files polled via env.execute()
(needs Python 3 on the backend). Siblings: tools/code_execution_env.py (env
scrubbing, interpreter/cwd), tools/code_execution_rpc.py (RPC servers).
"""

import base64
import json
import logging
import os
import re
import secrets
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tools.thread_context import propagate_context_to_thread
from tools.registry import registry, tool_error

from hermes_time import get_timezone_name
from tools.code_execution_env import _resolve_child_cwd, _resolve_child_python
from tools.code_execution_rpc import _rpc_poll_loop
from tools.tool_output_truncate import head_tail_split, truncation_notice

logger = logging.getLogger(__name__)

# Loopback TCP replaces AF_UNIX on Windows, so execute_code runs on every platform Hermes does.
SANDBOX_AVAILABLE = True

# Tools allowed inside the sandbox; ∩ the session's enabled tools decides which stubs are generated.
SANDBOX_ALLOWED_TOOLS = frozenset([
    "web_search", "web_extract", "read_file", "write_file", "search_files", "patch", "terminal",
])

# Resource limit defaults (overridable via config.yaml → code_execution.*)
DEFAULT_TIMEOUT = 300        # 5 minutes
DEFAULT_MAX_TOOL_CALLS = 50
MAX_STDOUT_BYTES = 50_000    # 50 KB
MAX_STDERR_BYTES = 10_000    # 10 KB
# Hard ceiling on the spilled file (as web_tools' MAX_STORED_TEXT_CHARS): a runaway print loop must not fill the disk.
MAX_SPILLED_STDOUT_BYTES = 5_000_000


def _truncate_stdout_text(stdout_text: str) -> Tuple[str, Dict[str, Any]]:
    """Cap stdout by bytes (40% head / 60% tail) with explicit truncation metadata: byte counts
    ride alongside the textual marker because a client layer can miss or re-truncate it. The
    omitted middle is spilled to cache/exec and the result carries the path (recover-don't-rerun)."""
    stdout_bytes = stdout_text.encode("utf-8", errors="replace")
    total = len(stdout_bytes)
    captured = min(total, MAX_STDOUT_BYTES)
    metadata: Dict[str, Any] = {"stdout_truncated": total > captured, "stdout_bytes_captured": captured,
                                "stdout_bytes_total": total, "stdout_bytes_omitted": total - captured}
    if total <= MAX_STDOUT_BYTES:
        return stdout_bytes.decode("utf-8", errors="replace"), metadata
    head_bytes, tail_bytes = head_tail_split(MAX_STDOUT_BYTES)
    text = (stdout_bytes[:head_bytes].decode("utf-8", errors="replace")
            + truncation_notice(total - captured, total, unit="bytes")
            + stdout_bytes[-tail_bytes:].decode("utf-8", errors="replace"))
    metadata["warning"] = ("execute_code stdout was truncated; the script did run, but only "
                           "the captured head/tail output is included. Re-run only with "
                           "narrower output if the omitted data is required.")
    spill_path = _spill_full_stdout(stdout_text)
    if spill_path:
        metadata["stdout_spill_path"] = spill_path
        metadata["warning"] = ("execute_code stdout was truncated (head/tail shown); the "
                               f"script did run. FULL output saved to {spill_path} — page it "
                               f'with read_file(path="{spill_path}", offset=...) instead of re-running.')
    return text, metadata


def _spill_full_stdout(stdout_text: str) -> Optional[str]:
    """Write full stdout to cache/exec; return its path (None on failure — best-effort,
    the truncated inline output is still returned). Keyed by content digest so identical
    reruns coalesce; the dir rides the cache/web remote bind-mount list (credential_files)."""
    try:
        import hashlib
        from hermes_constants import get_hermes_dir
        from tools.spill_safety import write_text_exclusive
        if len(stdout_text) > MAX_SPILLED_STDOUT_BYTES:
            stdout_text = (stdout_text[:MAX_SPILLED_STDOUT_BYTES]
                           + f"\n\n[... spill capped at {MAX_SPILLED_STDOUT_BYTES:,} bytes ...]")
        cache_dir = get_hermes_dir("cache/exec", "exec_spill")
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(stdout_text.encode("utf-8", errors="replace")).hexdigest()[:12]
        path = cache_dir / f"stdout-{digest}.txt"
        write_text_exclusive(path, stdout_text, private=False, overwrite=True)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to spill execute_code stdout: %s", exc)
        return None


def check_sandbox_requirements() -> bool:
    """check_fn: available unless the vercel_sandbox backend fails its own checks."""
    if not SANDBOX_AVAILABLE:
        return False
    try:
        from tools.terminal_tool import _get_env_config
        from tools.terminal_tool_backends import _check_vercel_sandbox_requirements
        config = _get_env_config()
    except Exception:
        logger.debug("Could not resolve terminal config for execute_code availability", exc_info=True)
        return False
    return config.get("env_type") != "vercel_sandbox" or _check_vercel_sandbox_requirements(config)


# ---- hermes_tools.py code generator ----

# Per-tool stub templates: (signature, docstring, args_dict_expr — the JSON payload sent over RPC).
_TOOL_STUBS = {
    "web_search": ("query: str, limit: int = 5",
        '"""Search the web. Returns dict with data.web list of {url, title, description}."""',
        '{"query": query, "limit": limit}'),
    "web_extract": ("urls: list, char_limit: int = None",
        '"""Extract content from URLs (no LLM summarization). Returns dict with results list of {url, title, content, error}. Pages over char_limit (default 15000) are head+tail truncated with the full text stored on disk; the content footer gives the path. content is markdown."""',
        '{"urls": urls, "char_limit": char_limit}'),
    "read_file": ("path: str, offset: int = 1, limit: int = 2000",
        '"""Read a file (1-indexed lines). Returns dict with "content" and "total_lines"."""',
        '{"path": path, "offset": offset, "limit": limit}'),
    "write_file": ("path: str, content: str, cross_profile: bool = False",
        '"""Write content to a file (always overwrites). Returns dict with status."""',
        '{"path": path, "content": content, "cross_profile": cross_profile}'),
    "search_files": ('pattern: str, target: str = "content", path: str = ".", file_glob: str = None, limit: int = 50, offset: int = 0, output_mode: str = "content", context: int = 0, order: str = "discovery"',
        '"""Search file contents (target="content") or find files by name (target="files"). Returns dict with "matches"."""',
        '{"pattern": pattern, "target": target, "path": path, "file_glob": file_glob, "limit": limit, "offset": offset, "output_mode": output_mode, "context": context, "order": order}'),
    "patch": ('path: str = None, old_string: str = None, new_string: str = None, replace_all: bool = False, mode: str = "replace", patch: str = None, cross_profile: bool = False',
        '"""Targeted find-and-replace (mode="replace") or V4A multi-file patches (mode="patch"). Returns dict with status."""',
        '{"path": path, "old_string": old_string, "new_string": new_string, "replace_all": replace_all, "mode": mode, "patch": patch, "cross_profile": cross_profile}'),
    "terminal": ("command: str, timeout: int = None, workdir: str = None",
        '"""Run a shell command (foreground only). Returns dict with "output" and "exit_code"."""',
        '{"command": command, "timeout": timeout, "workdir": workdir}'),
}


def _missing_hermes_tools_import_hint(m, enabled_tools) -> str:
    missing = m.group(1)
    if missing in {"json_parse", "shell_quote", "retry"}:
        return (f"Import helpers with `from hermes_tools import {missing}`. "
                "If that import failed, the generated module may be stale or another "
                "hermes_tools may be first on sys.path. Check hermes_tools.__file__ "
                "and retry with reset=true.")
    available = sorted(SANDBOX_ALLOWED_TOOLS & set(enabled_tools or SANDBOX_ALLOWED_TOOLS))
    return (f"'{missing}' is not available inside the execute_code sandbox. "
            f"Importable tools here: {', '.join(available)}. For anything "
            "else, use the normal tool call instead of execute_code.")


# (regex, formatter(match, enabled_tools)) — first match wins. Production mining (state.db) ranked
# these as the top execute_code failure classes: hermes_tools import misuse, missing helper
# imports, treating tool results as strings, importing third-party packages absent from the sandbox.
_FAILURE_HINT_RULES = (
    (r"cannot import name '(\w+)' from 'hermes_tools'", _missing_hermes_tools_import_hint),
    (r"NameError: name '(json_parse|shell_quote|retry)' is not defined",
     lambda m, _: f"Import {m.group(1)} before calling it: "
                  f"from hermes_tools import {m.group(1)}"),
    (r"ModuleNotFoundError: No module named '([\w.]+)'",
     lambda m, _: f"'{m.group(1)}' is not installed in the sandbox interpreter. "
                  "Use Python stdlib inside execute_code, or run the code via "
                  "terminal() with the project venv's python instead."),
    (r"TypeError: string indices must be integers|AttributeError: 'str' object has no attribute 'get'",
     lambda m, _: "Tool functions in the sandbox return DICTS (already parsed) — "
                  "do not json.loads() them or index them like strings. Example: read_file(path)['content']."),
)


def _sandbox_failure_hint(stderr_text: str, enabled_tools=None) -> Optional[str]:
    """Map well-known sandbox script failures to one actionable recovery hint
    (bounded scan, first match wins, never raises)."""
    if not stderr_text:
        return None
    window = stderr_text[:4000]
    try:
        for pattern, fmt in _FAILURE_HINT_RULES:
            m = re.search(pattern, window)
            if m:
                return fmt(m, enabled_tools)
    except Exception:
        return None
    return None


def generate_hermes_tools_module(enabled_tools: List[str],
                                 transport: str = "uds") -> str:
    """Source of the hermes_tools.py stub module for SANDBOX_ALLOWED_TOOLS ∩ *enabled_tools*.
    ``transport``: ``"uds"`` (local socket client) or ``"file"`` (file RPC, remote backends)."""
    header = _FILE_TRANSPORT_HEADER if transport == "file" else _UDS_TRANSPORT_HEADER
    return header + "\n".join(
        f"def {name}({sig}):\n    {doc}\n    return _call({name!r}, {args_expr})\n"
        for name, (sig, doc, args_expr) in sorted(_TOOL_STUBS.items()) if name in set(enabled_tools)
    )


# ---- Shared helpers section (embedded in both transport headers) ----------

_COMMON_HELPERS = '''\

# ---------------------------------------------------------------------------
# Convenience helpers (avoid common scripting pitfalls)
# ---------------------------------------------------------------------------

def json_parse(text: str):
    """Parse JSON tolerant of control characters and UTF-8 BOM (strict=False).
    Use this instead of json.loads() when parsing output from terminal()
    or web_extract() that may contain raw tabs/newlines in strings,
    or from tools/files that prepend a UTF-8 BOM (salvage #57870, credit @woxinwuhen713-bit)."""
    if isinstance(text, str) and text.startswith("\ufeff"):
        text = text[1:]
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

'''

# ---- UDS transport (local backend) ---------------------------------------

_UDS_TRANSPORT_HEADER = '''\
"""Auto-generated Hermes tools RPC stubs."""
import json, os, socket, shlex, threading, time

_sock = None
# The RPC server handles a single client connection serially and has no
# request-id in the protocol, so concurrent _call() invocations from multiple
# threads (e.g. ThreadPoolExecutor) would race on the shared socket and get
# each other's responses. Serialize the entire send+recv round-trip.
_call_lock = threading.Lock()
''' + _COMMON_HELPERS + '''\

def _connect():
    """Connect to the parent's RPC server via the transport it picked.

    HERMES_RPC_SOCKET can be either:
      - a filesystem path (POSIX Unix domain socket — the default on
        Linux and macOS)
      - a string of the form ``tcp://127.0.0.1:<port>`` (Windows, where
        AF_UNIX is unreliable — the parent falls back to loopback TCP)
    """
    global _sock
    if _sock is None:
        endpoint = os.environ["HERMES_RPC_SOCKET"]
        if endpoint.startswith("tcp://"):
            # tcp://host:port  (host is always 127.0.0.1 in practice — we
            # only bind loopback server-side)
            _host_port = endpoint[len("tcp://"):]
            _host, _, _port = _host_port.rpartition(":")
            _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _sock.connect((_host or "127.0.0.1", int(_port)))
        else:
            _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _sock.connect(endpoint)
        _sock.settimeout(300)
    return _sock

def _call(tool_name, args):
    """Send a tool call to the parent process and return the parsed result."""
    request = json.dumps({
        "tool": tool_name,
        "args": args,
        "token": os.environ.get("HERMES_RPC_TOKEN", ""),
    }) + "\\n"
    # Session kernels outlive the RPC server's 300s idle window, so their
    # connection can be legitimately gone by the next cell. The server
    # re-accepts (HERMES_RPC_PERSISTENT=1); retry once on a fresh socket.
    _attempts = 2 if os.environ.get("HERMES_RPC_PERSISTENT") == "1" else 1
    with _call_lock:
        for _attempt in range(_attempts):
            try:
                conn = _connect()
                conn.sendall(request.encode())
                buf = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        raise RuntimeError("Agent process disconnected")
                    buf += chunk
                    if buf.endswith(b"\\n"):
                        break
                break
            except (OSError, RuntimeError):
                global _sock
                try:
                    if _sock is not None:
                        _sock.close()
                except OSError:
                    pass
                _sock = None
                if _attempt + 1 >= _attempts:
                    raise
    raw = buf.decode().strip()
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

'''

# ---- File-based transport (remote backends) -------------------------------

_FILE_TRANSPORT_HEADER = '''\
"""Auto-generated Hermes tools RPC stubs (file-based transport)."""
import json, os, shlex, tempfile, threading, time

_RPC_DIR = os.environ.get("HERMES_RPC_DIR") or os.path.join(tempfile.gettempdir(), "hermes_rpc")
_seq = 0
# `_seq += 1` is not atomic (read-modify-write), so concurrent _call()
# invocations from multiple threads could allocate the same sequence number
# and clobber each other's request files. Guard seq allocation with a lock.
_seq_lock = threading.Lock()
''' + _COMMON_HELPERS + '''\

def _call(tool_name, args):
    """Send a tool call request via file-based RPC and wait for response."""
    global _seq
    with _seq_lock:
        _seq += 1
        seq = _seq
    seq_str = f"{seq:06d}"
    req_file = os.path.join(_RPC_DIR, f"req_{seq_str}")
    res_file = os.path.join(_RPC_DIR, f"res_{seq_str}")

    # Write request atomically (write to .tmp, then rename).
    # encoding="utf-8" is critical: on Windows-hosted remote backends
    # (or any non-UTF-8 locale) the default open() mode would mangle
    # non-ASCII chars in tool args when encoding them as JSON.
    tmp = req_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "tool": tool_name,
            "args": args,
            "seq": seq,
            "token": os.environ.get("HERMES_RPC_TOKEN", ""),
        }, f)
    os.rename(tmp, req_file)

    # Wait for response with adaptive polling
    deadline = time.monotonic() + 300  # 5-minute timeout per tool call
    poll_interval = 0.05  # Start at 50ms
    while not os.path.exists(res_file):
        if time.monotonic() > deadline:
            raise RuntimeError(f"RPC timeout: no response for {tool_name} after 300s")
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.2, 0.25)  # Back off to 250ms

    with open(res_file, encoding="utf-8") as f:
        raw = f.read()

    # Clean up response file
    try:
        os.unlink(res_file)
    except OSError:
        pass

    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

'''


# ---- Remote execution support (file-based RPC via terminal backend) ----

def _get_or_create_env(task_id: str):
    """``(env, env_type)`` — the environment the terminal/file tools share for *task_id*, created on
    first use (same double-checked per-task lock pattern as file_tools._get_file_ops)."""
    from tools.terminal_tool_backends import _container_config_from_config, _create_environment, _ssh_config_from_config
    from tools.terminal_tool import (
        _active_environments, _env_lock, _get_env_config, _last_activity,
        _start_cleanup_thread, _creation_locks, _creation_locks_lock, _task_env_overrides,
        _resolve_container_task_id, _resolve_task_host_cwd, _is_container_backend, _select_image,
    )
    effective_task_id = _resolve_container_task_id(task_id)
    def _cached():
        with _env_lock:
            env = _active_environments.get(effective_task_id)
            if env is not None:
                _last_activity[effective_task_id] = time.time()
        return env
    env = _cached()
    if env is not None:
        return env, _get_env_config()["env_type"]
    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(effective_task_id, threading.Lock())
    with task_lock:
        env = _cached()
        if env is not None:
            return env, _get_env_config()["env_type"]
        config = _get_env_config()
        env_type = config["env_type"]
        overrides = _task_env_overrides.get(effective_task_id, {})
        container_config = None
        from tools.terminal_tool import _is_container_backend as _is_container

        if _is_container(env_type):
            container_config = {
                "container_cpu": config.get("container_cpu", 1),
                "container_memory": config.get("container_memory", 5120),
                "container_disk": config.get("container_disk", 51200),
                "container_persistent": config.get("container_persistent", True),
                "vercel_runtime": config.get("vercel_runtime", ""),
                "docker_volumes": config.get("docker_volumes", []),
                "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
                "docker_network": config.get("docker_network", True),
            }

        ssh_config = None
        if env_type == "ssh":
            ssh_config = {
                "host": config.get("ssh_host", ""),
                "user": config.get("ssh_user", ""),
                "port": config.get("ssh_port", 22),
                "key": config.get("ssh_key", ""),
                "persistent": config.get("ssh_persistent", False),
            }

        local_config = None
        if env_type == "local":
            local_config = {
                "persistent": config.get("local_persistent", False),
            }

        logger.info("Creating new %s environment for execute_code task %s...",
                     env_type, effective_task_id[:8])
        env = _create_environment(
            env_type=env_type, image=_select_image(env_type, overrides, config),
            cwd=overrides.get("cwd") or config["cwd"], timeout=config["timeout"],
            ssh_config=_ssh_config_from_config(config) if env_type == "ssh" else None,
            container_config=container_config,
            local_config={"persistent": config.get("local_persistent", False)} if env_type == "local" else None,
            task_id=effective_task_id, host_cwd=_resolve_task_host_cwd(config, task_id),
        )
        with _env_lock:
            _active_environments[effective_task_id] = env
            _last_activity[effective_task_id] = time.time()
        _start_cleanup_thread()
        logger.info("%s environment ready for execute_code task %s",
                     env_type, effective_task_id[:8])
        return env, env_type


def _ship_file_to_remote(env, remote_path: str, content: str) -> None:
    """Write *content* to *remote_path* via ``echo … | base64 -d`` — some backends (Modal) don't
    reliably deliver stdin_data to chained commands; base64 is shell-safe inside single quotes."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    env.execute(f"echo '{encoded}' | base64 -d > {shlex.quote(remote_path)}", cwd="/", timeout=30)


def _env_temp_dir(env: Any) -> str:
    """Return a writable temp dir for env-backed execute_code sandboxes."""
    temp_dir = None
    get_temp_dir = getattr(env, "get_temp_dir", None)
    if callable(get_temp_dir):
        try:
            temp_dir = get_temp_dir()
        except Exception as exc:
            logger.debug("Could not resolve execute_code env temp dir: %s", exc)
    for candidate in (temp_dir, tempfile.gettempdir()):
        if isinstance(candidate, str) and candidate.startswith("/"):
            return candidate.rstrip("/") or "/"
    return "/tmp"


def _format_interrupted_output(stdout_text: str) -> str:
    """Append an interruption marker without guessing who caused it."""
    from tools.interrupt import get_interrupt_reason
    reason = get_interrupt_reason()
    marker = f"[execution interrupted — {reason}]" if reason else "[execution interrupted]"
    return f"{stdout_text}\n{marker}" if stdout_text else marker


def _format_interrupted_output(stdout_text: str) -> str:
    """Append an interruption marker without guessing who caused it."""
    from tools.interrupt import get_interrupt_reason

    reason = get_interrupt_reason()
    marker = (
        f"[execution interrupted — {reason}]"
        if reason
        else "[execution interrupted]"
    )
    return f"{stdout_text}\n{marker}" if stdout_text else marker


def _execute_remote(
    code: str,
    task_id: Optional[str],
    enabled_tools: Optional[List[str]],
) -> str:
    """Run a script on the remote terminal backend via file-based RPC.

    The script and the generated hermes_tools.py module are shipped to
    the remote environment, and tool calls are proxied through a polling
    thread that communicates via request/response files.
    """

    _cfg = _load_config()
    timeout = _cfg.get("timeout", DEFAULT_TIMEOUT)
    max_tool_calls = _cfg.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)

    session_tools = set(enabled_tools) if enabled_tools else set()
    sandbox_tools = frozenset(SANDBOX_ALLOWED_TOOLS & session_tools)
    if not sandbox_tools:
        sandbox_tools = SANDBOX_ALLOWED_TOOLS

    effective_task_id = task_id or "default"
    env, env_type = _get_or_create_env(effective_task_id)

    sandbox_id = uuid.uuid4().hex[:12]
    temp_dir = _env_temp_dir(env)
    sandbox_dir = f"{temp_dir}/hermes_exec_{sandbox_id}"
    quoted_sandbox_dir = shlex.quote(sandbox_dir)
    quoted_rpc_dir = shlex.quote(f"{sandbox_dir}/rpc")

    tool_call_log: list = []
    tool_call_counter = [0]
    exec_start = time.monotonic()
    stop_event = threading.Event()
    rpc_thread = None

    try:
        # Verify Python is available on the remote
        py_check = env.execute(
            "command -v python3 >/dev/null 2>&1 && echo OK",
            cwd="/", timeout=15,
        )
        if "OK" not in py_check.get("output", ""):
            return json.dumps({
                "status": "error",
                "error": (
                    f"Python 3 is not available in the {env_type} terminal "
                    "environment. Install Python to use execute_code with "
                    "remote backends."
                ),
                "tool_calls_made": 0,
                "duration_seconds": 0,
            })

        # Create sandbox directory on remote
        env.execute(
            f"mkdir -p {quoted_rpc_dir}", cwd="/", timeout=10,
        )

        rpc_token = secrets.token_urlsafe(32)

        # Generate and ship files
        tools_src = generate_hermes_tools_module(
            list(sandbox_tools), transport="file",
        )
        _ship_file_to_remote(env, f"{sandbox_dir}/hermes_tools.py", tools_src)
        _ship_file_to_remote(env, f"{sandbox_dir}/script.py", code)

        # Wrapped so the thread inherits the turn's approval context + callbacks
        # (see tools.thread_context) — else sandbox RPC tool calls lose approval
        # routing (#33057).
        rpc_thread = threading.Thread(
            target=propagate_context_to_thread(_rpc_poll_loop),
            args=(
                env, f"{sandbox_dir}/rpc", effective_task_id,
                tool_call_log, tool_call_counter, max_tool_calls,
                sandbox_tools, stop_event, rpc_token,
            ),
            daemon=True,
        )
        rpc_thread.start()

        # Build environment variable prefix for the script
        env_prefix = (
            f"HERMES_RPC_DIR={shlex.quote(f'{sandbox_dir}/rpc')} "
            f"HERMES_RPC_TOKEN={shlex.quote(rpc_token)} "
            f"PYTHONDONTWRITEBYTECODE=1"
        )
        tz = os.getenv("HERMES_TIMEZONE", "").strip()
        if tz:
            env_prefix += f" TZ={shlex.quote(tz)}"

        # Execute the script on the remote backend
        logger.info("Executing code on %s backend (task %s)...",
                     env_type, effective_task_id[:8])
        script_result = env.execute(
            f"cd {quoted_sandbox_dir} && {env_prefix} python3 script.py",
            timeout=timeout,
        )

        stdout_text = script_result.get("output", "") or ""
        exit_code = script_result.get("returncode", -1)
        status = "success"

        # Check for timeout/interrupt from the backend
        if exit_code == 124:
            status = "timeout"
        elif exit_code == 130:
            status = "interrupted"

    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logger.error(
            "execute_code remote failed after %ss with %d tool calls: %s: %s",
            duration, tool_call_counter[0], type(exc).__name__, exc,
            exc_info=True,
        )
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }, ensure_ascii=False)

    finally:
        # Stop the polling thread
        stop_event.set()
        if rpc_thread is not None:
            rpc_thread.join(timeout=5)

        # Clean up remote sandbox dir
        try:
            env.execute(
                f"rm -rf {quoted_sandbox_dir}", cwd="/", timeout=15,
            )
        except Exception:
            logger.debug("Failed to clean up remote sandbox %s", sandbox_dir)

    duration = round(time.monotonic() - exec_start, 2)

    # --- Post-process output (same as local path) ---

    stdout_text, stdout_metadata = _truncate_stdout_text(stdout_text)

    # Strip ANSI escape sequences
    from tools.ansi_strip import strip_ansi
    from agent.redact import redact_sensitive_text
    stdout_text, metadata = _truncate_stdout_text(stdout_text)
    return redact_sensitive_text(strip_ansi(stdout_text), code_file=True), metadata


def _with_timeout_notice(stdout_text: str, timeout_msg: str) -> str:
    """Timeout message goes in the output too — an empty result makes models answer as if
    nothing happened, and the gateway drops empty replies."""
    return stdout_text + f"\n\n⏰ {timeout_msg}" if stdout_text else f"⏰ {timeout_msg}"


def _error_result(error: str, *, tool_calls_made: int = 0, duration: float = 0,
                  user_summary: Optional[str] = None) -> str:
    body = {"status": "error", "error": error, "tool_calls_made": tool_calls_made, "duration_seconds": duration}
    if user_summary:
        body["user_summary"] = user_summary  # one human sentence; surfaces show it before the model text
    return json.dumps(body, ensure_ascii=False)


def _remote_failure(exc: BaseException, exec_start: float, tool_calls_made: int) -> str:
    duration = round(time.monotonic() - exec_start, 2)
    logger.error("execute_code remote failed after %ss with %d tool calls: %s: %s",
                 duration, tool_calls_made, type(exc).__name__, exc, exc_info=True)
    return _error_result(str(exc), tool_calls_made=tool_calls_made, duration=duration)


_REMOTE_EXIT_STATUS = {124: "timeout", 130: "interrupted"}


def _remote_result(status: str, raw_stdout: str, exec_start: float, fields: Dict[str, Any],
                   kernel: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Common remote reply shape: status, cleaned output, *fields*, duration, optional kernel
    info, then truncation metadata (key order is part of the result contract)."""
    stdout_text, stdout_metadata = _clean_output(raw_stdout)
    result: Dict[str, Any] = {"status": status, "output": stdout_text, **fields,
                              "duration_seconds": round(time.monotonic() - exec_start, 2)}
    if kernel is not None:
        result["kernel"] = kernel
    result.update(stdout_metadata)
    return result

    if status == "timeout":
        timeout_msg = f"Script timed out after {timeout}s and was killed."
        result["error"] = timeout_msg
        # Include timeout message in output so the LLM always surfaces it
        # to the user (see local path comment — same reasoning, #10807).
        if stdout_text:
            result["output"] = stdout_text + f"\n\n⏰ {timeout_msg}"
        else:
            result["output"] = f"⏰ {timeout_msg}"
        logger.warning(
            "execute_code (remote) timed out after %ss (limit %ss) with %d tool calls",
            duration, timeout, tool_call_counter[0],
        )
    elif status == "interrupted":
        result["output"] = _format_interrupted_output(stdout_text)
    elif exit_code != 0:
        result["status"] = "error"
        result["error"] = f"Script exited with code {exit_code}"

def _apply_timeout(result: Dict[str, Any], timeout_msg: str) -> None:
    result["error"] = timeout_msg
    result["output"] = _with_timeout_notice(result["output"], timeout_msg)


def _finish_remote_kernel_result(kernel_result: Dict[str, Any], *,
                                 timeout: int, exec_start: float) -> str:
    """Post-process a remote-kernel cell result into the tool's JSON reply. Timeout messaging
    mirrors the local kernel contract (kernel killed, state lost, next call fresh)."""
    stdout_text = kernel_result.get("stdout", "") or ""
    stderr_text = kernel_result.get("stderr", "") or ""
    traceback_text = kernel_result.get("traceback", "") or ""
    if stderr_text or traceback_text:
        # Same joining shape as the local kernel: stderr and traceback ride in
        # the output under one marker so the model sees the failure inline.
        stdout_text = stdout_text + "\n--- stderr ---\n" + stderr_text + traceback_text
    result = _remote_result(kernel_result.get("status", "error"), stdout_text, exec_start,
                            {"tool_calls_made": kernel_result.get("tool_calls_made", 0)},
                            kernel=kernel_result.get("kernel", {"remote": True}))
    if result["status"] == "timeout":
        _apply_timeout(result, f"Cell timed out after {timeout}s; the remote session kernel was "
                               "killed and its state was lost. The next call starts fresh.")
    elif result["status"] == "error" and kernel_result.get("error"):
        result["error"] = kernel_result["error"]
    return json.dumps(result, ensure_ascii=False)


def _sandbox_tools_for(enabled_tools: Optional[List[str]]) -> frozenset:
    """Enabled ∩ SANDBOX_ALLOWED_TOOLS, or every sandbox tool when the intersection is empty."""
    return frozenset(SANDBOX_ALLOWED_TOOLS & set(enabled_tools or ())) or SANDBOX_ALLOWED_TOOLS


def _run_remote_per_call(env, env_type: str, code: str, effective_task_id: str,
                         sandbox_tools: frozenset, *, timeout: int, max_tool_calls: int,
                         exec_start: float) -> str:
    """Per-call script ship: stage hermes_tools.py + script.py in a fresh remote sandbox dir,
    serve file-RPC from a polling thread, run, clean up."""
    sandbox_dir = f"{_env_temp_dir(env)}/hermes_exec_{uuid.uuid4().hex[:12]}"
    quoted_sandbox_dir = shlex.quote(sandbox_dir)
    quoted_rpc_dir = shlex.quote(f"{sandbox_dir}/rpc")
    tool_call_counter, stop_event, rpc_thread = [0], threading.Event(), None
    try:
        env.execute(f"mkdir -p {quoted_rpc_dir}", cwd="/", timeout=10)
        rpc_token = secrets.token_urlsafe(32)
        _ship_file_to_remote(env, f"{sandbox_dir}/hermes_tools.py",
                             generate_hermes_tools_module(list(sandbox_tools), transport="file"))
        _ship_file_to_remote(env, f"{sandbox_dir}/script.py", code)
        # Wrapped so the thread inherits the turn's approval context + callbacks
        # (tools.thread_context) — else sandbox RPC tool calls lose approval routing.
        # See #30882.
        rpc_thread = threading.Thread(
            target=propagate_context_to_thread(_rpc_poll_loop), daemon=True,
            args=(env, f"{sandbox_dir}/rpc", effective_task_id, [], tool_call_counter,
                  max_tool_calls, sandbox_tools, stop_event, rpc_token))
        rpc_thread.start()
        env_prefix = (f"HERMES_RPC_DIR={quoted_rpc_dir} HERMES_RPC_TOKEN={shlex.quote(rpc_token)} "
                      "PYTHONDONTWRITEBYTECODE=1")
        tz = get_timezone_name()  # routed profile's timezone, not the bridged default's
        if tz:
            env_prefix += f" TZ={shlex.quote(tz)}"
        logger.info("Executing code on %s backend (task %s)...", env_type, effective_task_id[:8])
        script_result = env.execute(f"cd {quoted_sandbox_dir} && {env_prefix} python3 script.py",
                                    timeout=timeout)
        stdout_text = script_result.get("output", "") or ""
        exit_code = script_result.get("returncode", -1)
        # Backend exit codes: 124 = timeout wrapper, 130 = SIGINT.
        status = _REMOTE_EXIT_STATUS.get(exit_code, "success")
    except Exception as exc:
        return _remote_failure(exc, exec_start, tool_call_counter[0])
    finally:
        stop_event.set()
        if rpc_thread is not None:
            rpc_thread.join(timeout=5)
        try:
            env.execute(f"rm -rf {quoted_sandbox_dir}", cwd="/", timeout=15)
        except Exception:
            logger.debug("Failed to clean up remote sandbox %s", sandbox_dir)
    result = _remote_result(status, stdout_text, exec_start,
                            {"exit_code": exit_code, "tool_calls_made": tool_call_counter[0]})
    if status == "timeout":
        _apply_timeout(result, f"Script timed out after {timeout}s and was killed.")
        logger.warning("execute_code (remote) timed out after %ss (limit %ss) with %d tool calls",
                       result["duration_seconds"], timeout, tool_call_counter[0])
    elif status == "interrupted":
        result["output"] = _format_interrupted_output(result["output"])
    elif exit_code != 0:
        result["status"] = "error"
        result["error"] = f"Script exited with code {exit_code}"
    return json.dumps(result, ensure_ascii=False)


def _execute_remote(code: str, task_id: Optional[str], enabled_tools: Optional[List[str]],
                    reset: bool = False) -> str:
    """Run code on the remote terminal backend: the owner's persistent remote session kernel
    (tools/code_kernel_remote.py) first, else the per-call script ship — the fail-open route when
    a kernel cannot be spawned and the only route for hosts that cannot sustain a background process."""
    _cfg = _load_config()
    timeout, max_tool_calls = _cfg.get("timeout", DEFAULT_TIMEOUT), _cfg.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
    sandbox_tools, effective_task_id = _sandbox_tools_for(enabled_tools), task_id or "default"
    env, env_type = _get_or_create_env(effective_task_id)
    exec_start = time.monotonic()
    try:
        py_check = env.execute("command -v python3 >/dev/null 2>&1 && echo OK", cwd="/", timeout=15)
        if "OK" not in py_check.get("output", ""):
            return _error_result(f"Python 3 is not available in the {env_type} terminal "
                                 "environment. Install Python to use execute_code with remote backends.")
        # Session-kernel path: one persistent kernel per owner on the
        # run-to-completion transport. Spawn failure falls OPEN to the per-call
        # path below so a degraded remote host never blocks execution.
        try:
            # --- Session-kernel path (hermes-agent#96873) ------------------- Same always-on model as
            # local: one persistent kernel per owner, rebuilt on the run-to-completion transport (detached
            # runner + file cell protocol).
            from tools.code_kernel_remote import execute_in_remote_kernel
            kernel_result = execute_in_remote_kernel(
                code, env=env, env_type=env_type, task_env_id=effective_task_id,
                sandbox_tools=frozenset(sandbox_tools), timeout=timeout,
                max_tool_calls=max_tool_calls, reset=bool(reset),
                idle_exit=int(_cfg.get("kernel_idle_timeout", 1800)),
            )
        except Exception:
            logger.warning("remote session-kernel path failed; falling back to per-call", exc_info=True)
            kernel_result = None
        if kernel_result is not None:
            return _finish_remote_kernel_result(kernel_result, timeout=timeout, exec_start=exec_start)
        logger.info("remote session kernel unavailable on %s; using per-call path", env_type)
    except Exception as exc:
        return _remote_failure(exc, exec_start, 0)
    return _run_remote_per_call(env, env_type, code, effective_task_id, sandbox_tools,
                                timeout=timeout, max_tool_calls=max_tool_calls, exec_start=exec_start)


# ---- Main entry point ----


def execute_code(
    code: str,
    task_id: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    reset: bool = False,
) -> str:
    """Run Python in the session's persistent kernel (local) or on the remote terminal backend,
    with RPC access to a subset of Hermes tools; returns the JSON result string. "Sandbox" means
    the security envelope (env scrubbing, tool whitelist + call budget, output redaction), not an
    isolation jail: default `project` mode runs in the session's cwd with the project venv.
    ``enabled_tools`` ∩ SANDBOX_ALLOWED_TOOLS; ``reset`` kills the existing kernel first."""
    if not SANDBOX_AVAILABLE:
        return tool_error("execute_code sandbox is unavailable in this environment. "
                          "Use normal tool calls (terminal, read_file, write_file, ...) instead.")
    # Fail closed under a terminal-policy refusal scope: the routed profile's terminal
    # policy is unresolved, so refuse rather than inherit the launch process's ambient policy.
    try:
        # See #68559.
        from tools.terminal_scope import enforce_no_refusal
        enforce_no_refusal()
    except Exception as refusal:
        return tool_error(f"execute_code refused: {refusal} "
                          "(profile terminal policy unresolved; fix the profile's config.yaml / .env and retry)")
    if not code or not code.strip():
        return tool_error(
            "No code provided. execute_code requires a non-empty 'code' "
            "parameter containing Python source. To run shell commands, use "
            "terminal(command=...) instead."
        )

    # Hard-block gateway-lifecycle commands, mirroring the terminal_tool
    # guard (#68289): without this, execute_code is a straight bypass — the
    # terminal() path refuses `launchctl bootout ai.hermes.gateway`, but the
    # identical command inside `os.system(...)` / `subprocess.run([...])`
    # here sailed through and SIGTERM'd the gateway mid-task. Gated on
    # PID-file ownership, not the inherited env marker (#92560).
    from tools.process_registry import _is_supervised_gateway_process
    if _is_supervised_gateway_process():
        from cron.lifecycle_guard import contains_gateway_lifecycle_command
        if contains_gateway_lifecycle_command(code):
            return tool_error(
                "Blocked: cannot restart or stop the gateway from inside the "
                "gateway process. The gateway would kill this script before "
                "it could complete (SIGTERM propagates to child processes). "
                "Run the lifecycle command from a shell outside the gateway."
            )

    # Dispatch: remote backends use file-based RPC, local uses UDS
    from tools.terminal_tool import _get_env_config, _docker_has_host_access
    _env_config = _get_env_config()
    env_type = _env_config["env_type"]
    # Arbitrary Python never passes through terminal()/DANGEROUS_PATTERNS, so guard the whole
    # script before either dispatch path spawns it — in this (tool-executor) thread, which holds
    # the session context. A Docker sandbox with host bind mounts gets no container fast-path.
    # See #30882.
    from tools.approval import check_execute_code_guard
    _guard = check_execute_code_guard(code, env_type, has_host_access=_docker_has_host_access(_env_config))
    if not _guard.get("approved", False):
        return _error_result(_guard.get("message") or "execute_code blocked by approval guard.",
                             user_summary=_guard.get("user_summary"))
    # Clear a stale interrupt bit that landed during the blocking approval-wait so it can't
    # kill the just-approved run on the first poll. A genuine post-clear interrupt re-sets it.
    if _guard.get("user_approved"):
        from tools.interrupt import clear_current_thread_interrupt
        clear_current_thread_interrupt()
    if env_type != "local":
        return _execute_remote(code, task_id, enabled_tools, reset=bool(reset))
    from tools.interrupt import is_interrupted as _is_interrupted
    # Session kernels are always on locally (one interpreter per conversation); the guards above
    # already ran for this cell, and the kernel path shares env builder, RPC server and redaction.
    from tools.code_kernel import execute_in_session_kernel
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
    # Use /tmp on macOS to avoid the long /var/folders/... path that pushes
    # Unix domain socket paths past the 104-byte macOS AF_UNIX limit.
    # On Linux, tempfile.gettempdir() already returns /tmp.
    #
    # Windows: Python 3.9+ added partial AF_UNIX support but the file-backed
    # variant is flaky across Windows builds (requires Windows 10 1803+,
    # still fails under some configurations, and the socket file can't live
    # on the same temp drive as the script).  Fall back to loopback TCP —
    # same ephemeral port, same 1-connection listen queue, same serialized
    # request/response framing.  The generated client reads the transport
    # selector from HERMES_RPC_SOCKET (path vs. ``tcp://host:port``).
    _sock_tmpdir = "/tmp" if sys.platform == "darwin" else tempfile.gettempdir()
    _use_tcp_rpc = _IS_WINDOWS
    if _use_tcp_rpc:
        sock_path = None  # not used on Windows; TCP endpoint stored below
        rpc_endpoint = None  # set after bind()
    else:
        sock_path = os.path.join(_sock_tmpdir, f"hermes_rpc_{uuid.uuid4().hex}.sock")
        rpc_endpoint = sock_path

    tool_call_log: list = []
    tool_call_counter = [0]  # mutable so the RPC thread can increment
    exec_start = time.monotonic()
    server_sock = None
    stop_event = threading.Event()

    try:
        # Write the auto-generated hermes_tools module.
        # encoding="utf-8" is required on Windows — the stub and user code
        # both contain non-ASCII characters (em-dashes in docstrings, plus
        # whatever the user script carries).  Python's default open() uses
        # the system locale on Windows (cp1252 typically), which corrupts
        # those bytes; the child then fails to import with a SyntaxError
        # ("'utf-8' codec can't decode byte 0x97 in position ...") because
        # Python source files are decoded as UTF-8 by default (PEP 3120).
        # sandbox_tools is already the correct set (intersection with session
        # tools, or SANDBOX_ALLOWED_TOOLS as fallback — see lines above).
        tools_src = generate_hermes_tools_module(list(sandbox_tools))
        with open(os.path.join(tmpdir, "hermes_tools.py"), "w", encoding="utf-8") as f:
            f.write(tools_src)

        # Write the user's script
        with open(os.path.join(tmpdir, "script.py"), "w", encoding="utf-8") as f:
            f.write(code)

        # --- Start RPC server ---
        rpc_token = secrets.token_urlsafe(32)
        # Two transports:
        #   POSIX: AF_UNIX stream socket on sock_path, chmod 0600 for
        #   owner-only access.  Filesystem permissions gate the socket.
        #   Windows: AF_INET stream socket on 127.0.0.1 with an ephemeral
        #   port.  No filesystem permission story, but loopback-only bind
        #   means only the current user's processes (not remote) can
        #   connect.  HERMES_RPC_SOCKET is set to ``tcp://127.0.0.1:<port>``
        #   which the generated client parses to pick AF_INET.
        if _use_tcp_rpc:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.bind(("127.0.0.1", 0))  # ephemeral port
            _host, _port = server_sock.getsockname()[:2]
            rpc_endpoint = f"tcp://{_host}:{_port}"
        else:
            server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_sock.bind(sock_path)
            os.chmod(sock_path, 0o600)
        server_sock.listen(1)

        # Wrapped so the thread inherits the turn's approval context + callbacks
        # (see tools.thread_context) — else gateway sandbox tool calls silently
        # auto-approve dangerous commands (#33057, #30882).
        rpc_thread = threading.Thread(
            target=propagate_context_to_thread(_rpc_server_loop),
            args=(
                server_sock, task_id, tool_call_log,
                tool_call_counter, max_tool_calls, sandbox_tools, stop_event, rpc_token,
            ),
            daemon=True,
        )
        rpc_thread.start()

        # --- Spawn child process ---
        # Build a minimal environment for the child. We intentionally exclude
        # API keys and tokens to prevent credential exfiltration from LLM-
        # generated scripts. The child accesses tools via RPC, not direct API.
        # Exception: env vars declared by loaded skills (via env_passthrough
        # registry) or explicitly allowed by the user in config.yaml
        # (terminal.env_passthrough) are passed through.  On Windows, a small
        # OS-essential allowlist (SYSTEMROOT, WINDIR, COMSPEC, ...) is also
        # passed through — without those, the child can't create a socket
        # or spawn a subprocess.  See ``_scrub_child_env`` for the rules.
        child_env = _scrub_child_env(os.environ)
        child_env["HERMES_RPC_SOCKET"] = rpc_endpoint
        child_env["HERMES_RPC_TOKEN"] = rpc_token
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Force UTF-8 for the child's stdio and default file encoding.
        #
        # Without this, on Windows sys.stdout is bound to the console code
        # page (cp1252 on US-locale installs), and any script that does
        # ``print("café")`` or ``print("→")`` crashes with:
        #
        #   UnicodeEncodeError: 'charmap' codec can't encode character
        #   '\u2192' in position N: character maps to <undefined>
        #
        # PYTHONIOENCODING fixes sys.stdin/stdout/stderr.
        # PYTHONUTF8=1 enables "UTF-8 mode" (PEP 540) which additionally
        # makes ``open()``'s default encoding UTF-8, so user scripts that
        # write files without specifying encoding= also work correctly.
        #
        # On POSIX both values usually match the locale default already,
        # so setting them is harmless belt-and-suspenders for environments
        # with a C/POSIX locale (containers, minimal base images).
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        # Inject user's configured timezone so datetime.now() in sandboxed
        # code reflects the correct wall-clock time.  Only TZ is set —
        # HERMES_TIMEZONE is an internal Hermes setting and must not leak
        # into child processes.
        _tz_name = os.getenv("HERMES_TIMEZONE", "").strip()
        if _tz_name:
            child_env["TZ"] = _tz_name
        child_env.pop("HERMES_TIMEZONE", None)

        from hermes_constants import apply_subprocess_home_env
        apply_subprocess_home_env(child_env)

        # Resolve interpreter + CWD based on execute_code mode.
        #   - strict : today's behavior (sys.executable + tmpdir CWD).
        #   - project: user's venv python + session's working directory, so
        #              project deps like pandas and user files resolve.
        # Env scrubbing and tool whitelist apply identically in both modes.
        _mode = _get_execution_mode()
        _child_python = _resolve_child_python(_mode)
        _child_cwd = _resolve_child_cwd(_mode, tmpdir, task_id=task_id or "")
        _script_path = os.path.join(tmpdir, "script.py")

        # ``hermes_tools.py`` always lives in the staging directory, so that
        # directory must be importable even when project mode changes CWD.
        # Hermes's own package root is useful too, but only when the child
        # uses the same Python environment. Project mode can select an
        # external venv; exposing Hermes's site-packages to that interpreter
        # can mix incompatible compiled extensions (for example, Python 3.12
        # NumPy with a Python 3.9 project interpreter).
        #
        # Before re-injecting PYTHONPATH, strip Hermes-owned entries that
        # leaked through _scrub_child_env (PYTHONPATH is in _SAFE_ENV_PREFIXES
        # so it passes the scrub).  They are redundant for same-Hermes-
        # environment children and may be incompatible with external
        # interpreters (project mode can select a different venv), so they
        # must not shadow or poison the child's sys.path (#74817).
        from tools.environments.local import _strip_hermes_owned_pythonpath
        _strip_hermes_owned_pythonpath(child_env)
        _hermes_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _existing_pp = child_env.get("PYTHONPATH", "")
        _pp_parts = [tmpdir]
        if _uses_hermes_python_environment(_child_python):
            _pp_parts.append(_hermes_root)
        elif _child_python not in _external_env_logged:
            # Import behavior changes silently otherwise — surface it (once
            # per interpreter path) so "import hermes_constants suddenly
            # fails" reports are diagnosable without log spam.
            _external_env_logged.add(_child_python)
            logger.info(
                "execute_code: child interpreter %s is outside the Hermes "
                "environment; hermes root omitted from PYTHONPATH",
                _child_python,
            )
        if _existing_pp:
            _pp_parts.append(_existing_pp)
        child_env["PYTHONPATH"] = os.pathsep.join(_pp_parts)

        proc = subprocess.Popen(
            [_child_python, _script_path],
            cwd=_child_cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
        )

        # --- Poll loop: watch for exit, timeout, and interrupt ---
        deadline = time.monotonic() + timeout
        stderr_chunks: list = []

        # Background readers to avoid pipe buffer deadlocks.
        # For stdout we use a head+tail strategy: keep the first HEAD_BYTES
        # and a rolling window of the last TAIL_BYTES so the final print()
        # output is never lost.  Stderr keeps head-only (errors appear early).
        _STDOUT_HEAD_BYTES = int(MAX_STDOUT_BYTES * 0.4)   # 40% head
        _STDOUT_TAIL_BYTES = MAX_STDOUT_BYTES - _STDOUT_HEAD_BYTES  # 60% tail

        def _drain(pipe, chunks, max_bytes):
            """Simple head-only drain (used for stderr)."""
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
            except (ValueError, OSError) as e:
                logger.debug("Error reading process output: %s", e, exc_info=True)

        stdout_total_bytes = [0]  # mutable ref for total bytes seen

        def _drain_head_tail(pipe, head_chunks, tail_chunks, head_bytes, tail_bytes, total_ref):
            """Drain stdout keeping both head and tail data."""
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
                    # Fill head buffer first
                    if head_collected < head_bytes:
                        keep = min(len(data), head_bytes - head_collected)
                        head_chunks.append(data[:keep])
                        head_collected += keep
                        data = data[keep:]  # remaining goes to tail
                        if not data:
                            continue
                    # Everything past head goes into rolling tail buffer
                    tail_buf.append(data)
                    tail_collected += len(data)
                    # Evict old tail data to stay within tail_bytes budget
                    while tail_collected > tail_bytes and tail_buf:
                        oldest = tail_buf.popleft()
                        tail_collected -= len(oldest)
            except (ValueError, OSError):
                pass
            # Transfer final tail to output list
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
        _activity_state = {
            "last_touch": time.monotonic(),
            "start": exec_start,
        }
        try:
            from tools.environments.base import touch_activity_if_due
        except Exception:
            touch_activity_if_due = None
        poll_interval = 0.005
        while proc.poll() is None:
            if _is_interrupted():
                _kill_process_group(proc)
                status = "interrupted"
                break
            now = time.monotonic()
            if now > deadline:
                _kill_process_group(proc, escalate=True)
                status = "timeout"
                break
            # Periodic activity touch so the gateway's inactivity timeout
            # doesn't kill the agent during long code execution (#10807).
            if touch_activity_if_due is not None:
                try:
                    touch_activity_if_due(_activity_state, "execute_code running")
                except Exception:
                    pass
            try:
                proc.wait(timeout=min(poll_interval, max(0.0, deadline - now)))
            except subprocess.TimeoutExpired:
                pass
            poll_interval = min(0.2, poll_interval * 1.5)

        # Wait for readers to finish draining
        stdout_reader.join(timeout=3)
        stderr_reader.join(timeout=3)

        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        stdout_text, stdout_metadata = _assemble_stdout_result(
            b"".join(stdout_head_chunks),
            b"".join(stdout_tail_chunks),
            total_bytes=stdout_total_bytes[0],
        )

        exit_code = proc.returncode if proc.returncode is not None else -1
        duration = round(time.monotonic() - exec_start, 2)

        # Wait for RPC thread to finish
        stop_event.set()
        server_sock.close()  # break accept() so thread exits promptly
        server_sock = None  # prevent double close in finally
        rpc_thread.join(timeout=3)

        # Strip ANSI escape sequences so the model never sees terminal
        # formatting — prevents it from copying escapes into file writes.
        from tools.ansi_strip import strip_ansi
        stdout_text = strip_ansi(stdout_text)
        stderr_text = strip_ansi(stderr_text)

        # Redact secrets (API keys, tokens, etc.) from sandbox output.
        # The sandbox env-var filter (lines 434-454) blocks os.environ access,
        # but scripts can still read secrets from disk (e.g. open('~/.hermes/.env')).
        # This ensures leaked secrets never enter the model context.
        # code_file=True: this is code-execution output — skip false-positive
        # ENV/JSON/f-string-template redaction; real credentials still masked.
        from agent.redact import redact_sensitive_text
        stdout_text = redact_sensitive_text(stdout_text, code_file=True)
        stderr_text = redact_sensitive_text(stderr_text, code_file=True)

        # Build response
        result: Dict[str, Any] = {
            "status": status,
            "output": stdout_text,
            "exit_code": exit_code,
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }
        result.update(stdout_metadata)

        if status == "timeout":
            timeout_msg = f"Script timed out after {timeout}s and was killed."
            result["error"] = timeout_msg
            # Include timeout message in output so the LLM always surfaces it
            # to the user.  When output is empty, models often treat the result
            # as "nothing happened" and produce an empty response, which the
            # gateway stream consumer silently drops (#10807).
            if stdout_text:
                result["output"] = stdout_text + f"\n\n⏰ {timeout_msg}"
            else:
                result["output"] = f"⏰ {timeout_msg}"
            logger.warning(
                "execute_code timed out after %ss (limit %ss) with %d tool calls",
                duration, timeout, tool_call_counter[0],
            )
        elif status == "interrupted":
            result["output"] = _format_interrupted_output(stdout_text)
        elif exit_code != 0:
            result["status"] = "error"
            result["error"] = stderr_text or f"Script exited with code {exit_code}"
            # Include stderr in output so the LLM sees the traceback
            if stderr_text:
                result["output"] = stdout_text + "\n--- stderr ---\n" + stderr_text
            # Known-failure-class recovery hint (import misuse, missing
            # module, dict-vs-string result handling) so the model fixes
            # the script on the next attempt instead of re-diagnosing.
            hint = _sandbox_failure_hint(stderr_text, enabled_tools=sandbox_tools)
            if hint:
                result["hint"] = hint

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
        # Cleanup temp dir and socket
        if server_sock is not None:
            try:
                server_sock.close()
            except OSError as e:
                logger.debug("Server socket close error: %s", e)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            # Only UDS has a filesystem socket to unlink; TCP sockets are
            # freed by server_sock.close() above.
            if sock_path:
                os.unlink(sock_path)
        except OSError:
            pass  # already cleaned up or never created


def _kill_process_group(proc, escalate: bool = False):
    """Kill the child and its entire process tree (cross-platform).

    Delegates to :func:`agent.deadline.kill_process_tree` (#85125 4d):
    SIGTERM to the whole tree first (killpg when the child leads its own
    group — it does, ``start_new_session=True`` — plus a psutil descendant
    sweep for setsid'd grandchildren; ``taskkill /T /F`` on Windows).
    With ``escalate=True`` the child gets 5s to exit after SIGTERM, then the
    surviving tree is SIGKILLed — same escalation the old psutil-local body
    implemented. Never raises; a delegation failure degrades to a plain
    ``proc.kill()`` like the old psutil-failure fallback.
    """
    import signal as _signal

    def _tree_signal(sig) -> None:
        try:
            from agent.deadline import kill_process_tree as _deadline_kill_tree

            _deadline_kill_tree(proc.pid, sig=sig)
        except Exception as e:
            logger.debug("Could not terminate process tree: %s", e, exc_info=True)
            try:
                proc.kill()
            except Exception as e2:
                logger.debug("Could not kill process: %s", e2, exc_info=True)

    # sig is ignored on Windows (taskkill /F is already forceful).
    _tree_signal(getattr(_signal, "SIGTERM", None))

    if escalate:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _tree_signal(getattr(_signal, "SIGKILL", None))


def _load_config() -> dict:
    """Effective ``code_execution`` section (defaults + user file + managed overlay) — runs while the
    module-level schema is built at tool discovery, so it must not import ``cli``."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly().get("code_execution", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


# ---- Execution mode resolution (strict vs project) ----

# Canonical code_execution.mode values (referenced by tests and the config layer). Session
# kernels are the only local execution model; a leftover kernel_mode config key is ignored.
EXECUTION_MODES = ("project", "strict")
DEFAULT_EXECUTION_MODE = "project"


def _get_execution_mode() -> str:
    """``code_execution.mode`` (invalid → default with a warning). ``project``: session cwd + active
    venv python; ``strict``: isolated temp dir + ``sys.executable``. Scrubbing/whitelist apply to both."""
    cfg_value = str(_load_config().get("mode", DEFAULT_EXECUTION_MODE)).strip().lower()
    if cfg_value in EXECUTION_MODES:
        return cfg_value
    logger.warning(
        "Ignoring code_execution.mode=%r (expected one of %s), falling back to %r",
        cfg_value, EXECUTION_MODES, DEFAULT_EXECUTION_MODE,
    )
    return DEFAULT_EXECUTION_MODE


# ---- OpenAI Function-Calling Schema ----

# Per-tool documentation lines for the execute_code description, in canonical display order.
_TOOL_DOC_LINES = [
    ("web_search", "  web_search(query: str, limit: int = 5) -> dict\n"
     "    Returns {\"data\": {\"web\": [{\"url\", \"title\", \"description\"}, ...]}}"),
    ("web_extract", "  web_extract(urls: list[str], char_limit: int = None) -> dict\n"
     "    Returns {\"results\": [{\"url\", \"title\", \"content\", \"error\"}, ...]} where content is markdown.\n"
     "    No LLM summarization. Pages over char_limit (default 15000) are head+tail truncated; full text stored on disk (path in the content footer)."),
    ("read_file", "  read_file(path: str, offset: int = 1, limit: int = 2000) -> dict\n"
     "    Lines are 1-indexed. Returns {\"content\": \"...\", \"total_lines\": N}"),
    ("write_file", "  write_file(path: str, content: str) -> dict\n    Always overwrites the entire file."),
    ("search_files", "  search_files(pattern: str, target=\"content\", path=\".\", file_glob=None, limit=50, order=\"discovery\") -> dict\n"
     "    target: \"content\" (search inside files) or \"files\" (find files by name). Returns {\"matches\": [...]}"),
    ("patch", "  patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict\n"
     "    Replaces old_string with new_string in the file."),
    ("terminal", "  terminal(command: str, timeout=None, workdir=None) -> dict\n"
     "    Foreground only (no background/pty). Returns {\"output\": \"...\", \"exit_code\": N}"),
]


def build_execute_code_schema(enabled_sandbox_tools: set = None,
                              mode: str = None) -> dict:
    """execute_code schema listing only *enabled_sandbox_tools* — a disabled tool (e.g. web off)
    must not appear or the model keeps trying it. ``mode`` (None → config) picks the cwd sentence."""
    if enabled_sandbox_tools is None:
        enabled_sandbox_tools = SANDBOX_ALLOWED_TOOLS
    if mode is None:
        mode = _get_execution_mode()
    tool_lines = "\n".join(doc for name, doc in _TOOL_DOC_LINES if name in enabled_sandbox_tools)
    import_examples = [n for n in ("web_search", "terminal") if n in enabled_sandbox_tools]
    import_examples = import_examples or sorted(enabled_sandbox_tools)[:2]
    import_str = ", ".join(import_examples) + ", ..." if import_examples else "..."
    if mode == "strict":
        cwd_note = (
            "Scripts run in their own temp dir, not the session's CWD — use absolute paths "
            "(os.path.expanduser('~/.hermes/.env')) or terminal()/read_file() for user files."
        )
    else:
        cwd_note = (
            "Scripts run in the session's working directory. Interpreter: "
            "the project's activated venv/conda python when one is active "
            "(VIRTUAL_ENV/CONDA_PREFIX — matches terminal()); otherwise "
            "Hermes's own python (the common case — stdlib plus Hermes's "
            "deps; check `import x` before relying on project packages)."
        )
    # Remote hosts that fail open to per-call are not worth schema words; the result's
    # `kernel` field tells the truth per call.
    # Session kernels are always on (kernel_mode retired in #96787): persistence is part of the tool's one
    # description, not a bolt-on paragraph behind a dead conditional.
    description = (
        "Run Python that calls Hermes tools programmatically. Use when you "
        "need 3+ tool calls with logic between them: filtering/reducing "
        "large outputs before they enter context, branching, or loops "
        "(N pages/files, retry on failure). Use normal tool calls for "
        "single calls, results you must reason over in full, or anything needing user interaction.\n\n"
        "Calls run in a persistent session kernel: variables, imports, and "
        "loaded data survive across execute_code calls, so build on earlier "
        "work instead of re-loading it. A timed-out or interrupted call loses that state.\n\n"
        f"Available via `from hermes_tools import ...`:\n\n"
        f"{tool_lines}\n\n"
        "Limits: 5-minute timeout, max 50 tool calls per call. Stdout over "
        "50KB shows head/tail inline; the FULL text is auto-saved to a file whose path rides in the result.\n\n"
        f"{cwd_note}\n\n"
        "Helpers require imports: `from hermes_tools import json_parse, shell_quote, retry`. "
        "json_parse(text) — tolerant "
        "json.loads for terminal() output; shell_quote(s) — shlex.quote for "
        "dynamic shell args; retry(fn, max_attempts=3, delay=2) — exponential backoff."
    )
    return {
        "name": "execute_code",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": (
                    "Python code to execute. Import tools with "
                    f"`from hermes_tools import {import_str}` "
                    "and print your final result to stdout.")},
                "reset": {"type": "boolean", "description": (
                    "Discard the kernel's persistent state and start fresh before running this code.")},
            },
            "required": ["code"],
        },
    }


# Registration-time schema (all sandbox tools, configured mode); model_tools.py rebuilds per-session.
EXECUTE_CODE_SCHEMA = build_execute_code_schema()


def _execute_code_handler(args: dict, **kwargs) -> str:
    """Redirect misdirected calls (terminal's ``command`` arg, non-string ``code``) with an
    actionable error before dispatching to ``execute_code``."""
    if "code" not in args and "command" in args:
        logger.warning("execute_code received 'command' instead of the required 'code' argument")
        return tool_error("execute_code received a 'command' parameter, but it requires "
                          "Python source in 'code'. Use terminal(command=...) for shell "
                          "commands; for Python, retry as execute_code(code=...).")
    code = args.get("code", "")
    if code is not None and not isinstance(code, str):
        return tool_error(f"execute_code received a {type(code).__name__} in 'code', but it "
                          "requires Python source as a string. Retry as execute_code(code=\"...\").")
    return execute_code(code=code or "", task_id=kwargs.get("task_id"),
                        enabled_tools=kwargs.get("enabled_tools"), reset=bool(args.get("reset", False)))


registry.register(
    name="execute_code", toolset="code_execution", schema=EXECUTE_CODE_SCHEMA,
    handler=_execute_code_handler, check_fn=check_sandbox_requirements, emoji="🐍",
    max_result_size_chars=100_000,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import platform  # noqa: F401,E402
import socket  # noqa: F401,E402
import sys  # noqa: F401,E402

DEFAULT_KERNEL_MODE = "session"

KERNEL_MODES = ("per-call", "session")  # legacy compat constant


_PLUGIN_COMPAT_LAZY = {
    'thread_scoped_silence': ('agent.thread_scoped_output', 'thread_scoped_silence'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
