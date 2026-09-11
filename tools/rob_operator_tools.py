"""P1 — thin, purpose-built read-only operator tools for Rob.

Every tool here follows the same shape: validate its own narrow
parameters, construct exactly one command from a fixed template (never
from caller-supplied free text), run it through `read_only_command_guard`
(defense-in-depth even though the template is already fixed — a future
edit to a template that accidentally introduces an injectable segment is
caught here rather than silently shipping), execute with a bound timeout,
and redact the result before returning it.

No tool in this file accepts a raw command string from a caller. Where a
tool's whole purpose is closest to "run a shell command" (`git_inspect`),
its `operation` parameter is a closed enum, not free text.
"""

from __future__ import annotations

import functools
import json
import re
import shlex
import socket
import ssl
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

from tools.db_readonly_profile import DbProfileError, build_session_init_statements, load_profile
from tools.host_profiles import HostProfile, get_host_profile
from tools.read_only_command_guard import run_read_only_guard
from tools.secret_redaction import redact_mapping, redact_text

_DEFAULT_TIMEOUT_S = 15
_MAX_OUTPUT_CHARS = 200_000  # bound result size regardless of what the underlying command produced


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    duration_ms: int = 0


_SAFE_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-/"
)


def _validate_name(value: str, kind: str) -> None:
    if not value or any(ch not in _SAFE_NAME_CHARS for ch in value):
        raise ValueError(f"'{value}' is not a valid {kind} (unsafe characters)")


def _tool_boundary(fn):
    """Every public tool in this module is wrapped with this: a
    ValueError raised anywhere inside (parameter validation, mostly)
    becomes a clean ``ToolResult(ok=False, ...)`` instead of an
    uncaught exception reaching whatever calls these tools (an agent
    tool-dispatch layer should never have to know these functions can
    also raise) — "fail closed" means the command never runs, not that
    the failure mode has to be a Python exception in the caller's face."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))

    return wrapper


def _run(command: str, *, timeout: int = _DEFAULT_TIMEOUT_S, host: str | None = None) -> ToolResult:
    """The single execution chokepoint every tool in this file funnels
    through: guard check, then execute (locally or via the selected host
    profile's backend), then bound and redact the result."""
    guard = run_read_only_guard(command)
    if not guard.allowed:
        return ToolResult(ok=False, error=f"denied by read-only guard: {guard.reason} (offending: {guard.offending})")

    profile: HostProfile = get_host_profile(host)
    if not profile.enabled:
        return ToolResult(ok=False, error=f"host profile '{profile.name}' is not enabled: {profile.disabled_reason}")

    started = time.monotonic()
    try:
        proc = profile.run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error=f"command timed out after {timeout}s", duration_ms=int((time.monotonic() - started) * 1000))
    except Exception as exc:  # fail closed — never let a transport error look like empty success
        return ToolResult(ok=False, error=f"execution error: {exc}", duration_ms=int((time.monotonic() - started) * 1000))
    duration_ms = int((time.monotonic() - started) * 1000)

    combined = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    # Redact BEFORE truncating, not after — reverted from a prior version
    # of this line that truncated first. That reorder was itself a real,
    # live-proven regression: several redaction patterns need trailing
    # context to even RECOGNIZE a secret (_URI_CREDENTIAL_PATTERN needs
    # the closing `@`; the token/JWT/cookie patterns need a minimum
    # length), so cutting the string can remove that context while still
    # leaving the secret's own prefix in the truncated output — e.g. a
    # `_MAX_OUTPUT_CHARS`-length cut landing between a DATABASE_URL's
    # password and its `@` leaves the password in plaintext, since the
    # pattern that would have caught it never gets to see the `@` at all.
    # The actual fix for the CPU cost this was trying to address is
    # bounding the regexes themselves (see secret_redaction.py's
    # `_URI_CREDENTIAL_PATTERN`, whose previously-unbounded scheme group
    # was the real quadratic-blowup source) — with that done, redacting
    # the full output before truncating is no longer the pathological
    # case it once was, and is the only order that doesn't reintroduce a
    # leak at the cut point.
    combined = redact_text(combined)[:_MAX_OUTPUT_CHARS]
    if proc.returncode != 0:
        return ToolResult(ok=False, error=combined or f"exited {proc.returncode}", duration_ms=duration_ms)
    return ToolResult(ok=True, output=combined, duration_ms=duration_ms)


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

@_tool_boundary
def docker_ps(host: str | None = None) -> ToolResult:
    return _run("docker ps", host=host)


@_tool_boundary
def docker_inspect(container: str, host: str | None = None) -> ToolResult:
    _validate_name(container, "container name")
    return _run(f"docker inspect {shlex.quote(container)}", host=host)


@_tool_boundary
def docker_logs(container: str, since: str | None = None, tail: int | None = None, host: str | None = None) -> ToolResult:
    _validate_name(container, "container name")
    parts = ["docker", "logs"]
    if since:
        _validate_name(since.replace(" ", "_"), "since value")  # loose sanity check, real quoting below
        parts += ["--since", shlex.quote(since)]
    if tail is not None:
        if not isinstance(tail, int) or tail <= 0 or tail > 10_000:
            return ToolResult(ok=False, error="tail must be a positive integer <= 10000")
        parts += ["--tail", str(tail)]
    parts.append(shlex.quote(container))
    return _run(" ".join(parts), host=host)


@_tool_boundary
def docker_stats(container: str | None = None, host: str | None = None) -> ToolResult:
    cmd = "docker stats --no-stream"
    if container:
        _validate_name(container, "container name")
        cmd += f" {shlex.quote(container)}"
    return _run(cmd, host=host)


@_tool_boundary
def docker_network_inspect(network: str, host: str | None = None) -> ToolResult:
    _validate_name(network, "network name")
    return _run(f"docker network inspect {shlex.quote(network)}", host=host)


@_tool_boundary
def docker_volume_inspect(volume: str, host: str | None = None) -> ToolResult:
    _validate_name(volume, "volume name")
    return _run(f"docker volume inspect {shlex.quote(volume)}", host=host)


@_tool_boundary
def docker_compose_ps(project_dir: str, host: str | None = None) -> ToolResult:
    _validate_name(project_dir, "project directory")
    return _run(f"cd {shlex.quote(project_dir)} && docker compose ps", host=host)


# ---------------------------------------------------------------------------
# systemd / journal
# ---------------------------------------------------------------------------

@_tool_boundary
def systemd_status(unit: str, host: str | None = None) -> ToolResult:
    _validate_name(unit, "unit name")
    return _run(f"systemctl status {shlex.quote(unit)}", host=host)


@_tool_boundary
def systemd_show(unit: str, property: str | None = None, host: str | None = None) -> ToolResult:
    _validate_name(unit, "unit name")
    cmd = f"systemctl show {shlex.quote(unit)}"
    if property:
        _validate_name(property, "property name")
        cmd += f" --property={shlex.quote(property)}"
    return _run(cmd, host=host)


_JOURNAL_PRIORITIES = frozenset({"emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"})


@_tool_boundary
def journal_query(
    unit: str | None = None,
    since: str | None = None,
    until: str | None = None,
    priority: str | None = None,
    grep: str | None = None,
    host: str | None = None,
) -> ToolResult:
    parts = ["journalctl", "--no-pager"]
    if unit:
        _validate_name(unit, "unit name")
        parts += ["-u", shlex.quote(unit)]
    if since:
        parts += ["--since", shlex.quote(since)]
    if until:
        parts += ["--until", shlex.quote(until)]
    if priority:
        if priority not in _JOURNAL_PRIORITIES:
            return ToolResult(ok=False, error=f"priority must be one of {sorted(_JOURNAL_PRIORITIES)}")
        parts += ["-p", priority]
    cmd = " ".join(parts)
    if grep:
        # grep piped after journalctl is still an allowed final segment —
        # the guard validates the WHOLE pipeline, including this grep.
        cmd += f" | grep {shlex.quote(grep)}"
    return _run(cmd, host=host)


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

_GIT_OPERATIONS = {
    "status": "git status",
    "log": "git log -20",
    "diff": "git diff",
    "show": "git show HEAD",
    "rev-parse": "git rev-parse HEAD",
    "merge-base": None,  # requires args
    "worktree-list": "git worktree list",
    "branch-current": "git branch --show-current",
    "reflog": "git reflog -20",
    "tag": "git tag",
}


@_tool_boundary
def git_inspect(repo: str, operation: str, args: Optional[list] = None, host: str | None = None) -> ToolResult:
    """``operation`` is a closed enum, never free text — see
    ``_GIT_OPERATIONS``. ``args`` is only consulted for operations that
    need them (currently just merge-base) and is individually shell-quoted,
    never interpolated raw."""
    _validate_name(repo, "repo path")
    if operation not in _GIT_OPERATIONS:
        return ToolResult(ok=False, error=f"operation must be one of {sorted(_GIT_OPERATIONS)}")
    if operation == "merge-base":
        if not args or len(args) != 2:
            return ToolResult(ok=False, error="merge-base requires exactly two ref arguments")
        for a in args:
            _validate_name(a, "git ref")
        base_cmd = f"git merge-base {shlex.quote(args[0])} {shlex.quote(args[1])}"
    else:
        base_cmd = _GIT_OPERATIONS[operation]
    return _run(f"cd {shlex.quote(repo)} && {base_cmd}", host=host)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

@_tool_boundary
def network_probe(host_name: str, port: int, timeout: int = 5) -> ToolResult:
    """A bounded TCP connect test — deliberately NOT built on the shell
    guard at all, since a plain `socket.connect` is a narrower, more
    directly bounded primitive than shelling out to `nc`/`telnet` (both
    denied outright by the guard for exactly this reason)."""
    if not isinstance(port, int) or not (0 < port < 65536):
        return ToolResult(ok=False, error="port must be an integer in 1..65535")
    started = time.monotonic()
    try:
        with socket.create_connection((host_name, port), timeout=timeout):
            pass
    except Exception as exc:
        return ToolResult(ok=False, error=f"connect failed: {exc}", duration_ms=int((time.monotonic() - started) * 1000))
    return ToolResult(ok=True, output=f"TCP connect to {host_name}:{port} succeeded", duration_ms=int((time.monotonic() - started) * 1000))


@_tool_boundary
def http_probe(url: str, method: str = "GET", timeout: int = 10) -> ToolResult:
    """GET/HEAD only, enforced at the parameter level — never a raw curl
    passthrough, so there is no `-X`/body/output-file surface to police at
    all. Authorization response headers are stripped from the returned
    result even though this tool never sends one itself, purely as a
    forward-compatible safety margin if a redirect ever echoes one back."""
    method = method.upper()
    if method not in ("GET", "HEAD"):
        return ToolResult(ok=False, error="method must be GET or HEAD")
    if not (url.startswith("http://") or url.startswith("https://")):
        return ToolResult(ok=False, error="url must be http:// or https://")
    started = time.monotonic()
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — scheme validated above
            status = resp.status
            headers = {k: v for k, v in resp.getheaders() if k.lower() != "authorization"}
            body = "" if method == "HEAD" else resp.read(_MAX_OUTPUT_CHARS).decode("utf-8", errors="replace")
    except Exception as exc:
        # The exception text can itself embed the requested URL (e.g. a
        # DNS/connect failure message quoting it back), and a
        # credential-bearing URL is a caller mistake this tool must not
        # amplify into a leak — redact the error text same as any other
        # output.
        return ToolResult(
            ok=False,
            error=redact_text(f"request failed: {exc}"),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    duration_ms = int((time.monotonic() - started) * 1000)
    output = redact_text(json.dumps({"status": status, "headers": headers, "body": body}))
    return ToolResult(ok=True, output=output, duration_ms=duration_ms)


@_tool_boundary
def tls_inspect(host_name: str, port: int = 443, timeout: int = 10) -> ToolResult:
    if not isinstance(port, int) or not (0 < port < 65536):
        return ToolResult(ok=False, error="port must be an integer in 1..65535")
    started = time.monotonic()
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host_name, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host_name) as tls_sock:
                cert = tls_sock.getpeercert()
    except Exception as exc:
        return ToolResult(ok=False, error=f"TLS inspect failed: {exc}", duration_ms=int((time.monotonic() - started) * 1000))
    duration_ms = int((time.monotonic() - started) * 1000)
    if not cert:
        # getpeercert() returns None when no certificate was retrieved
        # (rare with a default context requiring verification, but the
        # stdlib type stub allows it — fail closed rather than crash).
        return ToolResult(ok=False, error="no certificate returned by peer", duration_ms=duration_ms)
    summary = {
        "subject": cert.get("subject"),
        "issuer": cert.get("issuer"),
        "notBefore": cert.get("notBefore"),
        "notAfter": cert.get("notAfter"),
        "subjectAltName": cert.get("subjectAltName"),
    }
    return ToolResult(ok=True, output=json.dumps(summary), duration_ms=duration_ms)


# ---------------------------------------------------------------------------
# Host / process
# ---------------------------------------------------------------------------

@_tool_boundary
def host_metrics(host: str | None = None) -> ToolResult:
    return _run("df -h / && free -h && uptime && uname -a", host=host)


@_tool_boundary
def process_inspect(pid: int, host: str | None = None) -> ToolResult:
    if not isinstance(pid, int) or pid <= 0:
        return ToolResult(ok=False, error="pid must be a positive integer")
    return _run(f"ps -o pid,ppid,user,etime,cmd -p {pid}", host=host)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_FORBIDDEN_SQL_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "copy", "call", "do", "vacuum", "reindex", "cluster",
    "lock", "merge", "execute", "prepare", "listen", "notify", "refresh",
    "into",  # blocks SELECT ... INTO, which creates a table
    # Function-call-based mutation/file-access: a bare SELECT wrapping one
    # of these is a mutation or filesystem read in disguise, not an
    # inspection query. This is a text-level convenience filter, not the
    # security boundary — the boundary is the dedicated least-privilege
    # role (sequence functions are the only one of these still callable
    # under a real read-only role; the rest require superuser and would be
    # refused server-side regardless).
    "setval", "nextval",
    "lo_import", "lo_export", "lo_get", "loread", "lowrite",
    "pg_read_file", "pg_read_binary_file", "pg_read_server_files",
    "pg_ls_dir", "pg_ls_logdir", "pg_ls_waldir", "pg_ls_tmpdir",
    "pg_stat_file",
    "dblink", "dblink_connect",
    "pg_sleep",
    "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "pg_rotate_logfile", "pg_switch_wal", "pg_promote",
    "pg_stat_reset", "set_config",
    "pg_file_write", "pg_file_unlink", "pg_file_rename",
    # "lock" (above) requires a word boundary before it, which underscore
    # does NOT provide in regex (`_lock` has no \b between `_` and `l`) —
    # these advisory-lock functions need their own explicit entries.
    "pg_advisory_lock", "pg_advisory_xact_lock",
    "pg_advisory_unlock", "pg_advisory_unlock_all",
)


def _reject_non_select(query: str) -> str | None:
    stripped = query.strip().rstrip(";").strip()
    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return "only SELECT (or a read-only WITH ... SELECT CTE) is permitted"
    if ";" in stripped:
        return "multiple statements are not permitted"
    for word in _FORBIDDEN_SQL_KEYWORDS:
        # Word-boundary check, not a naive substring match (so e.g. a
        # column literally named "deleted_at" isn't rejected).
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return f"query contains a forbidden keyword: {word}"
    return None


@_tool_boundary
def db_select(profile: str, query: str, params: Optional[list] = None, row_limit: int = 100) -> ToolResult:
    """Refuses to run at all unless a dedicated read-only profile is
    configured (never the app's own superuser credential — see
    ``db_readonly_profile.load_profile``'s own refusal logic), enforces
    SELECT-only at the query-text level as an additional layer, and caps
    both server-side (statement_timeout) and client-side (row_limit)
    resource use."""
    try:
        db_profile = load_profile(profile)
    except DbProfileError as exc:
        return ToolResult(ok=False, error=str(exc))

    rejection = _reject_non_select(query)
    if rejection:
        return ToolResult(ok=False, error=rejection)

    effective_limit = min(row_limit, db_profile.row_limit)
    started = time.monotonic()
    try:
        import psycopg  # local import: only required when this tool path is actually used
    except ImportError:
        return ToolResult(ok=False, error="psycopg is not installed — db_select cannot run")

    try:
        with psycopg.connect(db_profile.dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                for stmt in build_session_init_statements(db_profile):
                    cur.execute(stmt)
                bounded_query = f"SELECT * FROM ({query.rstrip(';')}) AS rob_bounded LIMIT {effective_limit}"
                cur.execute(bounded_query, params or [])
                columns = [c.name for c in cur.description] if cur.description else []
                rows = cur.fetchall()
    except Exception as exc:
        # Postgres error text can echo the failing statement, including any
        # literal values it contained — redact same as any other output
        # rather than trust that a DB driver's exception text is safe.
        return ToolResult(
            ok=False,
            error=redact_text(f"query failed: {exc}"),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    duration_ms = int((time.monotonic() - started) * 1000)

    payload = redact_mapping({"columns": columns, "rows": [list(r) for r in rows], "row_count": len(rows)})
    return ToolResult(ok=True, output=json.dumps(payload, default=str), duration_ms=duration_ms)


@_tool_boundary
def schema_inspect(profile: str, object: Optional[str] = None) -> ToolResult:
    if object:
        query = (
            "select column_name, data_type, is_nullable "
            "from information_schema.columns where table_name = %s order by ordinal_position"
        )
        params = [object]
    else:
        query = "select table_name from information_schema.tables where table_schema = 'public' order by table_name"
        params = []
    return db_select(profile, query, params=params, row_limit=500)


# ---------------------------------------------------------------------------
# Container read-only exec
# ---------------------------------------------------------------------------

_CONTAINER_EXEC_DENY_FLAGS = frozenset({"-i", "-t", "-it", "-ti", "--privileged", "--user", "-u"})


@_tool_boundary
def container_exec_readonly(container: str, command: str, host: str | None = None, timeout: int = 15) -> ToolResult:
    """No generic `docker exec` passthrough: the target container is
    validated, the interactive/privileged/user-override flags this
    function itself controls are hardcoded absent (never taken from the
    caller), and the COMMAND that runs inside the container goes through
    the exact same read-only guard as every other Rob tool — so `docker
    exec ctr rm -rf /` is denied at the same layer as a bare `rm -rf /`
    would be, not by some separate container-specific logic."""
    _validate_name(container, "container name")
    guard = run_read_only_guard(command)
    if not guard.allowed:
        return ToolResult(ok=False, error=f"denied by read-only guard: {guard.reason} (offending: {guard.offending})")
    # No -i/-t/--privileged/--user ever appended — plain non-interactive
    # exec as whatever user the container's own default is.
    full_command = f"docker exec {shlex.quote(container)} {command}"
    return _run(full_command, timeout=timeout, host=host)
