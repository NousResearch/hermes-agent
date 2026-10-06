"""Runtime evidence and identity binding for consequential smart approvals.

Extra reasoning time does not replace missing information.  This module reads
the small amount of host state needed by the two ambiguity classes that first
motivated it (bounded file deletion and force-killing a PID), keeps raw values
out of trusted prompt text, and binds an approval to stable object identity.

Observed strings are attacker-controlled data.  The trusted part is the schema
and the code-computed identity; process names, argv, paths, and VCS output never
become instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import threading
import time
from typing import Any, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from tools import approval_context as _ctx

logger = logging.getLogger("tools.approval")

_EXECUTION_WINDOW_SECONDS = 5.0
_STALE_RATE_WARNING_THRESHOLD = 0.03
_MAX_RAW_VALUE_CHARS = 4096
_PORT_REFERENCE_RE = re.compile(r"(?<!\d):(\d{2,5})\b")
_PORT_INSPECTION_RE = re.compile(r"(?i)\b(?:netstat(?:\.exe)?|get-nettcpconnection)\b")
_SHELL_BOUNDARY_RE = re.compile(r"\s*(?:&&|\|\||[;|\n])\s*")
_SHELL_META_RE = re.compile(r"[*?\[\]`$%~<>&#{}\\]")

_telemetry_lock = threading.Lock()
_telemetry_total = 0
_telemetry_stale = 0


@dataclass
class ApprovalPreflight:
    """One observation bound to a later execution-time identity check."""

    kind: str
    command_sha256: str
    env_type: str
    cwd: str
    status: str
    identity: dict[str, Any]
    observations: dict[str, Any]
    reason: str = ""
    observed_at_monotonic: float = field(default_factory=time.monotonic)
    approved_at_monotonic: Optional[float] = None
    queue_wait_ms: int = 0
    decision_latency_ms: int = 0


@dataclass(frozen=True)
class PreflightCheck:
    allowed: bool
    cause: str
    age_ms: int


def _command_hash(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8", "surrogatepass")).hexdigest()


def _bounded(value: Any) -> str:
    text = str(value or "")
    if len(text) <= _MAX_RAW_VALUE_CHARS:
        return text
    return text[:_MAX_RAW_VALUE_CHARS] + "...[truncated]"


def _normalized_path(value: str) -> str:
    return os.path.normcase(os.path.abspath(value))


def _first_shell_segment(command: str) -> str:
    return _SHELL_BOUNDARY_RE.split(command, maxsplit=1)[0].strip()


def _bounded_rm_operands(command: str) -> Optional[list[str]]:
    """Operands for a literal, non-recursive standalone ``rm``; else None."""
    if _SHELL_BOUNDARY_RE.search(command) or _SHELL_META_RE.search(command):
        return None
    segment = _first_shell_segment(command)
    try:
        argv = shlex.split(segment, posix=True)
    except ValueError:
        return None
    if not argv or argv[0].lower() not in {"rm", "rm.exe"}:
        return None
    operands: list[str] = []
    options_done = False
    for token in argv[1:]:
        if not options_done and token == "--":
            options_done = True
            continue
        if not options_done and token.startswith("-"):
            if "r" in token.lower():
                return None
            continue
        options_done = True
        if _SHELL_META_RE.search(token):
            return None
        operands.append(token)
    return operands or None


def _taskkill_pids(command: str) -> list[int]:
    segment = _first_shell_segment(command)
    try:
        argv = shlex.split(segment, posix=True)
    except ValueError:
        return []
    if not argv or argv[0].lower() not in {"taskkill", "taskkill.exe"}:
        return []
    pids = []
    force = False
    index = 1
    while index < len(argv):
        option = argv[index].upper()
        if option == "/F":
            force = True
        elif option == "/PID" and index + 1 < len(argv) and argv[index + 1].isdigit():
            index += 1
            pids.append(int(argv[index]))
        else:
            return []
        index += 1
    return pids if force else []


def requires_runtime_preflight(command: str) -> bool:
    """Whether approval needs live host identity unavailable to batched preparation."""
    return bool(_taskkill_pids(command) or _bounded_rm_operands(command))


def _run_git(args: list[str], *, cwd: str) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", cwd, *args], check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=2.0, stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        return completed.returncode, completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return -1, ""


def _git_observation(path: str, *, cwd: str) -> dict[str, Any]:
    root_rc, root = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if root_rc != 0 or not root:
        return {"repository": False, "tracked": None, "status": ""}
    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        return {"repository": True, "root": _bounded(root), "tracked": False, "status": "outside_repository"}
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return {"repository": True, "root": _bounded(root), "tracked": False, "status": "outside_repository"}
    tracked_rc, _ = _run_git(["ls-files", "--error-unmatch", "--", relative], cwd=root)
    status_rc, status_text = _run_git(["status", "--porcelain=v1", "--", relative], cwd=root)
    return {
        "repository": True,
        "root": _bounded(root),
        "tracked": tracked_rc == 0,
        "status": _bounded(status_text) if status_rc == 0 else "unavailable",
    }


def _disposable_roots() -> list[str]:
    raw = _ctx._get_approval_config().get("disposable_roots", [])
    if not isinstance(raw, list):
        return []
    roots = []
    for value in raw:
        if isinstance(value, str) and value.strip():
            roots.append(_normalized_path(os.path.realpath(os.path.expandvars(os.path.expanduser(value.strip())))))
    return roots


def _under_root(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _observe_files(command: str, cwd: str) -> ApprovalPreflight:
    operands = _bounded_rm_operands(command) or []
    disposable_roots = _disposable_roots()
    identities = []
    observations = []
    complete = bool(operands)
    for operand in operands:
        lexical = operand if os.path.isabs(operand) else os.path.join(cwd, operand)
        lexical = _normalized_path(lexical)
        try:
            info = os.lstat(lexical)
        except FileNotFoundError:
            identity = {"path": lexical, "exists": False}
            observed = {"input_path": _bounded(operand), "path": _bounded(lexical), "exists": False}
        except OSError as exc:
            complete = False
            identity = {"path": lexical, "exists": None}
            observed = {
                "input_path": _bounded(operand), "path": _bounded(lexical),
                "exists": None, "observation_error": type(exc).__name__,
            }
        else:
            identity = {
                "path": lexical,
                "exists": True,
                "device": int(info.st_dev),
                "file_id": int(info.st_ino),
                "kind": stat.S_IFMT(info.st_mode),
            }
            observed = {
                "input_path": _bounded(operand),
                "path": _bounded(lexical),
                "resolved_path": _bounded(str(Path(lexical).resolve(strict=False))),
                "exists": True,
                "is_symlink": stat.S_ISLNK(info.st_mode),
                "size": int(info.st_size),
                "mtime_ns": int(info.st_mtime_ns),
            }
        # rm unlinks the final component itself, but follows symlinked parents.
        effective_path = _normalized_path(os.path.join(os.path.realpath(os.path.dirname(lexical)), os.path.basename(lexical)))
        identity["effective_path"] = effective_path
        observed["under_disposable_root"] = any(_under_root(effective_path, root) for root in disposable_roots)
        observed["git"] = _git_observation(lexical, cwd=cwd)
        identities.append(identity)
        observations.append(observed)
    return ApprovalPreflight(
        kind="bounded_file_delete",
        command_sha256=_command_hash(command),
        env_type="local",
        cwd=cwd,
        status="ready" if complete else "incomplete",
        identity={"files": identities},
        observations={
            "trust": "UNTRUSTED_MACHINE_OBSERVATION",
            "files": observations,
            "configured_disposable_roots": [_bounded(root) for root in disposable_roots],
        },
        reason="" if complete else "one or more file identities could not be observed",
    )


def _observe_process(pid: int) -> tuple[dict[str, Any], dict[str, Any], bool]:
    try:
        import psutil

        process = psutil.Process(pid)
        with process.oneshot():
            create_time = process.create_time()
            executable = process.exe()
            name = process.name()
            commandline = process.cmdline()
        try:
            listeners = sorted({
                int(connection.laddr.port)
                for connection in process.net_connections(kind="inet")
                if connection.laddr and connection.status == psutil.CONN_LISTEN
            })
        except (psutil.AccessDenied, psutil.Error, OSError):
            listeners = []
    except Exception as exc:
        try:
            import psutil
            missing = isinstance(exc, psutil.NoSuchProcess)
        except Exception:
            missing = False
        if missing:
            return ({"pid": pid, "exists": False}, {"pid": pid, "exists": False}, True)
        return (
            {"pid": pid, "exists": None},
            {"pid": pid, "exists": None, "observation_error": type(exc).__name__},
            False,
        )
    normalized_executable = _normalized_path(executable) if executable else ""
    identity = {
        "pid": pid,
        "exists": True,
        "start_time_ns": int(create_time * 1_000_000_000),
        "executable_path": normalized_executable,
    }
    observation = {
        "pid": pid,
        "exists": True,
        "name": _bounded(name),
        "executable_path": _bounded(executable),
        "commandline": [_bounded(part) for part in commandline],
        "listening_ports": listeners,
    }
    return identity, observation, bool(normalized_executable and create_time)


def _argv_declared_ports(commandline: list[str]) -> set[int]:
    ports: set[int] = set()
    for index, value in enumerate(commandline):
        lowered = str(value).lower()
        candidate = ""
        if lowered.startswith("--port="):
            candidate = lowered.partition("=")[2]
        elif lowered in {"--port", "-p"} and index + 1 < len(commandline):
            candidate = str(commandline[index + 1])
        if candidate.isdigit() and 0 < int(candidate) <= 65535:
            ports.add(int(candidate))
    return ports


def _process_target_evidence(command: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a narrow, code-checkable intent match; raw names never count."""
    if not _PORT_INSPECTION_RE.search(command):
        return {"kind": "none", "identity_match": False}
    referenced_ports = {
        int(value) for value in _PORT_REFERENCE_RE.findall(command)
        if 0 < int(value) <= 65535
    }
    if len(referenced_ports) != 1 or len(observations) != 1:
        return {"kind": "none", "identity_match": False}
    port = next(iter(referenced_ports))
    process = observations[0]
    listening_ports = {int(value) for value in process.get("listening_ports", [])}
    argv_ports = _argv_declared_ports(process.get("commandline", []))
    return {
        "kind": "listening_port",
        "port": port,
        "listener_match": port in listening_ports,
        "argv_match": port in argv_ports,
        "identity_match": port in listening_ports and port in argv_ports,
    }


def _observe_processes(command: str, cwd: str) -> ApprovalPreflight:
    identities = []
    observations = []
    complete = True
    for pid in _taskkill_pids(command):
        identity, observation, observed = _observe_process(pid)
        identities.append(identity)
        observations.append(observation)
        complete = complete and observed
    return ApprovalPreflight(
        kind="force_kill_pid",
        command_sha256=_command_hash(command),
        env_type="local",
        cwd=cwd,
        status="ready" if complete else "incomplete",
        identity={"processes": identities},
        observations={
            "trust": "UNTRUSTED_MACHINE_OBSERVATION",
            "processes": observations,
            "single_command": not bool(_SHELL_BOUNDARY_RE.search(command) or _SHELL_META_RE.search(command)),
            "command_target": _process_target_evidence(command, observations),
        },
        reason="" if complete else "stable process identity could not be observed",
    )


def observe_preflight(command: str, *, env_type: str, cwd: str) -> Optional[ApprovalPreflight]:
    """Capture local live evidence; preserve the existing gate for remote backends."""
    pids = _taskkill_pids(command)
    operands = _bounded_rm_operands(command)
    if env_type != "local" or (not pids and not operands):
        return None
    normalized_cwd = _normalized_path(cwd or os.getcwd())
    if pids:
        return _observe_processes(command, normalized_cwd)
    return _observe_files(command, normalized_cwd)


def prompt_block(preflight: ApprovalPreflight) -> str:
    """Canonical JSON block whose free-text values cannot terminate its delimiter."""
    payload = {
        "schema": "hermes.smart_approval.preflight.v1",
        "trust": "UNTRUSTED_MACHINE_OBSERVATION",
        "kind": preflight.kind,
        "status": preflight.status,
        "reason": preflight.reason,
        "observations": preflight.observations,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f'<machine_observations encoding="canonical-json">\n{encoded}\n</machine_observations>'


def deterministic_preflight_verdict(preflight: ApprovalPreflight) -> Optional[str]:
    """Resolve only cases whose safety follows directly from the observed state.

    Existing files outside an operator-declared disposable root stay with the
    owner: neither extra model time nor an attractive filename proves they are
    recoverable.  Missing targets are harmless no-ops.  A live process has
    enough identity for the reviewer to assess, but is never approved here by
    name or command-line pattern alone.
    """
    if preflight.status != "ready":
        return "escalate"
    if preflight.kind == "bounded_file_delete":
        files = preflight.observations.get("files", [])
        if files and all(item.get("exists") is False for item in files):
            return "approve"
        if not files or any(
            item.get("exists") is not True or not item.get("under_disposable_root")
            for item in files
        ):
            return "escalate"
    elif preflight.kind == "force_kill_pid":
        processes = preflight.observations.get("processes", [])
        if processes and all(item.get("exists") is False for item in processes) and preflight.observations.get("single_command"):
            return "approve"
        target = preflight.observations.get("command_target", {})
        if not isinstance(target, dict) or target.get("identity_match") is not True:
            return "escalate"
    return None


def seal_preflight(preflight: Optional[ApprovalPreflight], *, now: Optional[float] = None) -> None:
    if preflight is not None:
        preflight.approved_at_monotonic = time.monotonic() if now is None else now


def verify_preflight(
    preflight: ApprovalPreflight, *, command: str, env_type: str, cwd: str,
    now: Optional[float] = None,
) -> PreflightCheck:
    """Re-observe stable identity immediately before execution; never re-run the readout."""
    current_time = time.monotonic() if now is None else now
    approved_at = preflight.approved_at_monotonic
    age_ms = max(0, int((current_time - approved_at) * 1000)) if approved_at is not None else 0
    if preflight.command_sha256 != _command_hash(command) or preflight.env_type != env_type:
        return PreflightCheck(False, "IDENTITY_MISMATCH", age_ms)
    current = observe_preflight(command, env_type=env_type, cwd=cwd)
    if current is None or current.kind != preflight.kind or current.status != "ready":
        return PreflightCheck(False, "OBSERVATION_FAILED", age_ms)
    if current.cwd != preflight.cwd or current.identity != preflight.identity:
        return PreflightCheck(False, "IDENTITY_MISMATCH", age_ms)
    if approved_at is None or current_time - approved_at > _EXECUTION_WINDOW_SECONDS:
        return PreflightCheck(False, "EXECUTION_DELAYED", age_ms)
    return PreflightCheck(True, "MATCH", age_ms)


def record_preflight_check(preflight: ApprovalPreflight, check: PreflightCheck) -> None:
    """Local operational telemetry: stale rate and cause, with no raw observed values."""
    global _telemetry_total, _telemetry_stale
    with _telemetry_lock:
        _telemetry_total += 1
        if not check.allowed:
            _telemetry_stale += 1
        total, stale = _telemetry_total, _telemetry_stale
    logger.info(
        "Smart approval preflight recheck: result=%s cause=%s kind=%s age_ms=%d "
        "queue_wait_ms=%d decision_latency_ms=%d total=%d stale=%d stale_rate=%.4f",
        "allow" if check.allowed else "deny", check.cause, preflight.kind, check.age_ms,
        preflight.queue_wait_ms, preflight.decision_latency_ms,
        total, stale, stale / total,
    )
    if not check.allowed and total >= 20 and stale / total > _STALE_RATE_WARNING_THRESHOLD:
        logger.warning(
            "Smart approval preflight stale rate exceeds %.1f%%: stale=%d total=%d; "
            "inspect fingerprint stability before treating the world as hostile",
            _STALE_RATE_WARNING_THRESHOLD * 100, stale, total,
        )


def reset_preflight_telemetry_for_tests() -> None:
    global _telemetry_total, _telemetry_stale
    with _telemetry_lock:
        _telemetry_total = _telemetry_stale = 0
