"""Minimal client for the local DeepSeek Harness (dsh) agent process.

Pure transport/client layer -- no registry imports, safe to import from
tests or other tools. Mirrors what dsh 0.1.x actually exposes (see the kanban
dsh integration spec):

* ``run_headless`` -- one-shot task via ``dsh --profile headless "<task>"``:
  a fresh agent is started, the task is submitted as a user message, the
  final assistant text lands on stdout and the exit code (0/1) expresses
  success/failure. No port, no interaction, no built-in timeout -- the caller
  must kill the process tree once the deadline passes.
* ``rpc`` / ``list_sessions`` -- read-only JSON-RPC against the dsh *web*
  profile (``POST /api/<method>`` on ``dsh.url``, default
  http://127.0.0.1:3080), used for session inventory. dsh exposes no
  OpenAI-compatible endpoint, so task execution is process-level only.

All failures are normalized to :class:`DshError` subclasses carrying a
stable ``code`` (DSH_* string) that the tool layer surfaces to the model.
Configuration is read from the ``dsh:`` block of config.yaml (see
cli-config.yaml.example); dsh keeps its own upstream model credentials in
~/.dsh/.credentials.yaml -- Hermes never sees or forwards them.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_OUTPUT_CAP_CHARS = 8000
_RPC_TIMEOUT_SECONDS = 15.0
_DEFAULT_RUN_TIMEOUT_SECONDS = 900
_DEFAULT_PROFILE = "headless"
_DEFAULT_URL = "http://127.0.0.1:3080"


# ---------------------------------------------------------------------------
# Error types -- stable DSH_* codes surfaced via tool_error(code=...)
# ---------------------------------------------------------------------------

class DshError(Exception):
    """Base class for all dsh adapter failures."""

    code = "DSH_ERROR"

    def __init__(self, message: str, *, extra: Optional[dict] = None):
        super().__init__(str(message))
        self.message = str(message)
        self.extra = dict(extra or {})


class DshNotConfigured(DshError):
    code = "DSH_NOT_CONFIGURED"


class DshLaunchFailed(DshError):
    code = "DSH_LAUNCH_FAILED"


class DshUnreachable(DshError):
    code = "DSH_UNREACHABLE"


class DshTimeoutError(DshError):
    code = "DSH_TIMEOUT"


class DshAuthError(DshError):
    code = "DSH_AUTH_FAILED"


class DshRpcError(DshError):
    code = "DSH_RPC_ERROR"


class DshProtocolError(DshError):
    code = "DSH_PROTOCOL_ERROR"


class DshEmptyResult(DshError):
    code = "DSH_EMPTY_RESULT"


class DshExecFailed(DshError):
    code = "DSH_EXEC_FAILED"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Return the active profile's ``dsh:`` config block (never raises).

    Missing section / unreadable config yields ``{}``, which makes the
    toolset's check_fn return False (tool hidden, no model-visible noise).
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        block = config.get("dsh")
        return dict(block) if isinstance(block, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive, config is cached
        logger.debug("dsh config unreadable: %s", exc)
        return {}


def _cfg_str(config: Dict[str, Any], key: str, default: str = "") -> str:
    value = config.get(key, default)
    return str(value).strip() if value is not None else default


def _cfg_int(config: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_command(config: Dict[str, Any]) -> List[str]:
    raw = config.get("command")
    if isinstance(raw, str) and raw.strip():
        # YAML users may write a bare string; split on shell whitespace.
        return [part for part in raw.split() if part]
    if isinstance(raw, (list, tuple)):
        return [str(p) for p in raw if str(p).strip()]
    return []


def _text_cap(text: str, limit: int = _OUTPUT_CAP_CHARS) -> str:
    """Truncate long text for model-visible summaries, never mid-surrogate."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n...[truncated, {len(text) - limit} chars omitted]"


# ---------------------------------------------------------------------------
# Headless one-shot tasks (primary channel)
# ---------------------------------------------------------------------------

def interpret_run(
    exit_code: int,
    stdout_text: str,
    stderr_text: str,
    *,
    elapsed_s: float,
    cwd: Optional[str],
    sessions_dir: str,
) -> dict:
    """Classify a finished headless process into a result dict.

    Pure and synchronous so the full outcome matrix is unit-testable without
    spawning processes. Raises :class:`DshError` subclasses for every
    non-success outcome.
    """
    stdout_text = stdout_text or ""
    stderr_text = stderr_text or ""

    if exit_code != 0:
        stderr_l = stderr_text.lower()
        auth_hint = (
            "401" in stderr_l
            or "403" in stderr_l
            or "unauthori" in stderr_l
            or "invalid api key" in stderr_l
            or "authentication" in stderr_l
            or "credential" in stderr_l
            or "api key" in stderr_l
        )
        if auth_hint:
            raise DshAuthError(
                "dsh's upstream model call failed authentication. Check that "
                "~/.dsh/.credentials.yaml holds a valid DEEPSEEK_API_KEY and "
                "that settings.yaml's agent-default-model provider matches it.",
                extra={"exit_code": exit_code, "stderr": _text_cap(stderr_text, 2000)},
            )
        raise DshExecFailed(
            f"dsh exited with code {exit_code} without completing the task. "
            f"stderr: {_text_cap(stderr_text, 2000) or '(empty)'}",
            extra={"exit_code": exit_code, "stderr": _text_cap(stderr_text, 2000)},
        )

    if not stdout_text.strip():
        raise DshEmptyResult(
            "dsh finished (exit 0) but produced no assistant output. "
            "Retry with a clearer goal, or check the session log under "
            f"{sessions_dir}.",
            extra={"sessions_dir": sessions_dir},
        )

    return {
        "ok": True,
        "status": "completed",
        "exit_code": 0,
        "output": _text_cap(stdout_text.strip()),
        "cwd": cwd or "(inherited)",
        "sessions_dir": sessions_dir,
        "elapsed_s": int(round(elapsed_s)),
    }


def _default_sessions_dir() -> str:
    dsh_home = os.environ.get("DSH_HOME", "").strip() or os.path.join(
        os.path.expanduser("~"), ".dsh"
    )
    return os.path.join(dsh_home, "sessions")


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the process and its whole tree (headless spawns node -> agent)."""
    pid = proc.pid
    if pid is None:
        return
    if sys.platform == "win32":
        # taskkill /T walks the tree; /F force-kills. Fall back to proc.kill().
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=15,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)  # start_new_session=True
            return
        except (ProcessLookupError, PermissionError):
            pass
        except Exception:  # pragma: no cover - last resort on odd platforms
            pass
    try:
        proc.kill()
    except Exception:
        pass


async def run_headless(
    task: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Run one headless dsh task and wait for it to finish.

    Blocks up to ``timeout`` seconds (config ``dsh.timeout``, default 900);
    on expiry the whole process tree is killed and :class:`DshTimeoutError`
    is raised. Returns the :func:`interpret_run` summary dict on success.
    """
    config = config if config is not None else load_config()
    command = _cfg_command(config)
    if not command:
        raise DshNotConfigured(
            "dsh.command is empty in the dsh: config block, so action=run "
            "cannot launch the harness. Set it to the dsh CLI argv "
            "(e.g. [node.exe, --import, tsx/esm, <repo>/apps/cli/src/bin.ts]) "
            "or use action=list if you only need session inventory."
        )

    profile = _cfg_str(config, "profile", _DEFAULT_PROFILE) or _DEFAULT_PROFILE
    bash_dir = _cfg_str(config, "bash_dir")
    workdir = (cwd or "").strip() or _cfg_str(config, "cwd") or None
    effective_timeout = float(
        timeout if timeout is not None else _cfg_int(config, "timeout", _DEFAULT_RUN_TIMEOUT_SECONDS)
    )
    if effective_timeout <= 0:
        raise DshNotConfigured(f"dsh.timeout must be > 0, got {effective_timeout}")

    argv = list(command) + ["--profile", profile, task]
    env = os.environ.copy()
    if bash_dir:
        # Windows: never let the child resolve C:\\Windows\\System32\\bash.exe
        # (the WSL shim) when dsh spawns `bash` -- prepend the real Git Bash.
        env["PATH"] = bash_dir + os.pathsep + env.get("PATH", "")

    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=(sys.platform != "win32"),
        )
    except FileNotFoundError as exc:
        raise DshLaunchFailed(
            f"Could not start dsh: executable not found ({exc}). Check the "
            f"node/dsh path in the dsh.command config block."
        ) from exc
    except OSError as exc:
        raise DshLaunchFailed(
            f"Could not start dsh (cwd={workdir or '(inherited)'}): {exc}. "
            f"Check that the dsh.command path exists and dsh.cwd is valid."
        ) from exc

    try:
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            _kill_process_tree(proc)
            # Drain whatever the killed tree already emitted (bounded).
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=10
                )
            except Exception:
                stdout_bytes, stderr_bytes = b"", b""
            elapsed = time.monotonic() - started
            raise DshTimeoutError(
                f"dsh task exceeded the {int(effective_timeout)}s timeout and "
                "was killed (process tree terminated). Re-run with a larger "
                "timeout, or split the goal into smaller steps.",
                extra={
                    "elapsed_s": int(round(elapsed)),
                    "stdout_tail": _text_cap(
                        (stdout_bytes or b"").decode("utf-8", "replace"), 2000
                    ),
                },
            ) from None
    except DshError:
        raise
    except Exception as exc:
        _kill_process_tree(proc)
        raise DshLaunchFailed(f"dsh subprocess failed unexpectedly: {exc}") from exc

    stdout_text = (stdout_bytes or b"").decode("utf-8", "replace")
    stderr_text = (stderr_bytes or b"").decode("utf-8", "replace")
    return interpret_run(
        proc.returncode or 0,
        stdout_text,
        stderr_text,
        elapsed_s=time.monotonic() - started,
        cwd=workdir,
        sessions_dir=_default_sessions_dir(),
    )


# ---------------------------------------------------------------------------
# dsh web JSON-RPC (secondary channel: read-only session management)
# ---------------------------------------------------------------------------

def _parse_envelope(data: Any) -> Any:
    """Validate a Typert RPC response frame and return its ``value``.

    Raises :class:`DshProtocolError` (malformed frame) or
    :class:`DshRpcError` (server-reported error envelope).
    """
    if not isinstance(data, dict):
        raise DshProtocolError(f"dsh RPC returned a non-object payload: {type(data).__name__}")
    if data.get("type") != "server-response" or "result" not in data:
        raise DshProtocolError(
            f"dsh RPC returned an unexpected envelope: {_text_cap(json.dumps(data, ensure_ascii=False), 500)}"
        )
    result = data["result"]
    if not isinstance(result, dict):
        raise DshProtocolError("dsh RPC 'result' is not an object")
    if result.get("ok") is False:
        err = result.get("error") or {}
        code = str(err.get("code") or "unknown") if isinstance(err, dict) else "unknown"
        message = str(err.get("message") or err) if isinstance(err, dict) else str(err)
        raise DshRpcError(f"dsh RPC error {code}: {message}")
    if result.get("ok") is not True or "value" not in result:
        raise DshProtocolError(
            f"dsh RPC 'result' lacks ok/value: {_text_cap(json.dumps(result, ensure_ascii=False), 500)}"
        )
    return result["value"]


def _map_http_status(status: int, body_text: str) -> DshError:
    """Map a non-2xx HTTP response from the /api endpoint to a DshError."""
    snippet = _text_cap(body_text or "", 500)
    if status in (401, 403):
        return DshAuthError(f"dsh web rejected the request (HTTP {status}): {snippet}")
    if status == 404:
        return DshProtocolError(
            f"dsh web has no such API method (HTTP 404): {snippet} "
            f"-- the running dsh version may differ from this adapter."
        )
    return DshProtocolError(f"dsh web returned HTTP {status}: {snippet}")


async def rpc(
    method: str,
    payload: dict,
    *,
    config: Optional[Dict[str, Any]] = None,
    timeout: float = _RPC_TIMEOUT_SECONDS,
) -> Any:
    """POST one Typert RPC request to the dsh web profile and return value."""
    config = config if config is not None else load_config()
    url = _cfg_str(config, "url", _DEFAULT_URL).rstrip("/")
    if not url:
        raise DshNotConfigured(
            "dsh.url is not set in the dsh: config block; this action needs "
            "the dsh web profile (start it with start-harness.ps1)."
        )

    import aiohttp

    frame = {
        "type": "client-request",
        "rpcId": f"hermes-{uuid.uuid4().hex[:12]}",
        "method": method,
        "payload": payload,
    }
    endpoint = f"{url}/api/{method}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=frame,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                body_text = await resp.text()
                if resp.status >= 400:
                    raise _map_http_status(resp.status, body_text)
                try:
                    data = json.loads(body_text)
                except json.JSONDecodeError as exc:
                    raise DshProtocolError(
                        f"dsh web returned non-JSON for {method}: "
                        f"{_text_cap(body_text, 300)}"
                    ) from exc
                return _parse_envelope(data)
    except asyncio.TimeoutError:
        raise DshTimeoutError(
            f"dsh web call {method} timed out after {int(timeout)}s. Is the "
            f"web profile healthy at {url}?"
        ) from None
    except DshError:
        raise
    except (aiohttp.ClientConnectionError, OSError) as exc:
        raise DshUnreachable(
            f"dsh web is not listening at {url} ({exc.__class__.__name__}). "
            f"Start it (start-harness.ps1 / scheduled task) or fix dsh.url."
        ) from exc


async def list_sessions(
    *,
    config: Optional[Dict[str, Any]] = None,
    limit: int = 20,
    timeout: float = _RPC_TIMEOUT_SECONDS,
) -> dict:
    """Return a compact session inventory from dsh web's session.list."""
    value = await rpc("session.list", {}, config=config, timeout=timeout)
    items = value.get("items") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise DshProtocolError(
            "dsh session.list returned no 'items' array: "
            f"{_text_cap(json.dumps(value, ensure_ascii=False), 500)}"
        )
    compact = []
    for item in items:
        if not isinstance(item, dict):
            continue
        projections = item.get("projections") or {}
        pvalues = projections.get("values") if isinstance(projections, dict) else {}
        title = ""
        if isinstance(pvalues, dict):
            title = str(pvalues.get("title") or "")
        compact.append(
            {
                "sessionId": str(item.get("sessionId") or ""),
                "cwd": str(item.get("cwd") or ""),
                "agentPreset": str(item.get("agentPreset") or ""),
                "running": bool(item.get("running")),
                "updatedAt": item.get("updatedAt"),
                "title": title,
            }
        )
        if len(compact) >= limit:
            break
    return {
        "ok": True,
        "count": len(compact),
        "items": compact,
        "note": f"Most recent {len(compact)} sessions from dsh web (session.list).",
    }
