"""`hermes computer-use doctor` — thin client for cua-driver's `health_report` MCP tool. cua-driver owns the health
model; we drive the stdio JSON-RPC handshake, call `health_report` and render the stable ``schema_version="1"``
payload. cua-driver 0.10.x marks `health_report` risk-unclassified (isError=true, structuredContent
``{"exit_code": 1}``) — we detect that and synthesize a composite report from working probes (check_permissions,
list_apps, CLI --version). Exit codes: 0 overall=="ok"; 1 degraded/failed; 2 binary missing / protocol error."""

from __future__ import annotations

import json
import os
import platform as _platform_mod
import re
import subprocess
import sys
from contextlib import contextmanager, suppress
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.computer_use.permissions import _child_env as _sanitized_cua_env

# Match the ALLOWED_STATUS_VALUES + ALLOWED_OVERALL_VALUES the cua-driver integration test pins.
_STATUS_GLYPH = {"pass": "✅", "fail": "❌", "skip": "⏭️"}
_OVERALL_GLYPH = {"ok": "✅", "degraded": "⚠️", "failed": "❌"}
_SUPPORTED_PLATFORMS = ("darwin", "linux", "windows")
_TCC_HINT = "Grant {} to CuaDriver in System Settings → Privacy & Security."
_ZERO_DISPLAY_MSG = "ScreenCaptureKit reachable but 0 shareable display(s) — every capture will return 0x0."
_ZERO_DISPLAY_HINT = ("Wake the built-in display, connect a monitor or HDMI dummy dongle (e.g. Headless Ghost), or enable "
                      "a virtual display (Screen Sharing/VNC, BetterDisplay). Verify with `system_profiler SPDisplaysDataType`.")
_IO_EXC = (OSError, subprocess.TimeoutExpired)
Report = Dict[str, Any]
_Row = Tuple[str, str, Report]  # (status, message, extra {hint?, data?}) for one check


class HealthReportUnavailable(RuntimeError):
    """health_report denied or non-schema payload — ``run_doctor`` falls back to probes."""

def _run_cli(binary: str, *args: str, timeout: float) -> subprocess.CompletedProcess:
    """Run ``<binary> args`` with UTF-8 capture + sanitized env (raises on failure)."""
    return subprocess.run([binary, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          env=_sanitized_cua_env(), stdin=subprocess.DEVNULL)

def _cli_text(binary: str, *args: str, timeout: float, exc_types: Tuple[type, ...] = _IO_EXC) -> Union[subprocess.CompletedProcess, BaseException]:
    """``_run_cli`` that returns (not raises) any exception in *exc_types*."""
    try:
        return _run_cli(binary, *args, timeout=timeout)
    except exc_types as e:
        return e

def _combined_output(completed: subprocess.CompletedProcess) -> str:
    return ((completed.stdout or "") + (completed.stderr or "")).strip()

def _first_line(text: str) -> Optional[str]:
    return text.strip().splitlines()[0].strip() if text.strip() else None

def _read_cli_version(binary: str, *, timeout: float = 5.0) -> Optional[str]:
    """First line of ``--version`` or None; health_report's ``driver_version`` can disagree (seen on Windows)."""
    cp = _cli_text(binary, "--version", timeout=timeout, exc_types=_IO_EXC + (ValueError, TypeError))
    return None if isinstance(cp, BaseException) else _first_line(cp.stdout or cp.stderr or "")

def _cli_driver_version(binary: str, timeout: float = 5.0) -> Tuple[str, Optional[str]]:
    """(status, version_or_message) from ``cua-driver --version``."""
    cp = _cli_text(binary, "--version", timeout=timeout)
    if isinstance(cp, BaseException):
        return "fail", f"--version failed: {cp}"
    text, failed = _combined_output(cp), cp.returncode != 0
    if failed and not text:
        return "fail", f"--version exited {cp.returncode}"
    m = re.search(r"(\d+\.\d+\.\d+(?:[-+][\w.]+)?)", text)  # typical: "cua-driver 0.10.0"
    return ("fail" if failed else "pass"), m.group(1) if m else (_first_line(text) or "unknown")

def _cli_doctor_snippet(binary: str, timeout: float = 8.0) -> Optional[str]:
    """Optional one-shot ``cua-driver doctor`` text (best-effort, never fatal)."""
    cp = _cli_text(binary, "doctor", timeout=timeout)
    return None if isinstance(cp, BaseException) else (_combined_output(cp) or None)

def _build_identity(binary: str, report: Report) -> Report:
    """Hermes-side identity block comparing resolved binary vs health_report."""
    def token(text: str) -> str:  # dotted version-ish token out of a free-form string
        m = text and re.search(r"(\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?)", text)
        return m.group(1) if m else text.strip().lower()
    cli, report_v = _read_cli_version(binary) or "", str(report.get("driver_version") or "")
    return {"resolved_binary": binary, "cli_version": cli or None, "health_report_driver_version": report_v or None,
            "version_mismatch": bool(token(cli) and token(report_v) and token(cli) != token(report_v))}

def _is_valid_health_report(payload: Any) -> bool:  # looks like a schema_version=1 health_report
    return isinstance(payload, dict) and {"schema_version", "overall"} <= payload.keys() and isinstance(payload.get("checks"), list)

def _text_items(result: Report) -> Iterator[str]:  # text of every {"type": "text"} content item
    return (i.get("text") or "" for i in result.get("content") or [] if isinstance(i, dict) and i.get("type") == "text")

def _first_text(result: Report, default: str) -> str:
    return next((t.strip() for t in _text_items(result) if t.strip()), default)

def _extract_health_report_from_result(result: Report) -> Report:
    """Report from a tools/call result. ``HealthReportUnavailable`` when the tool denied the call (isError) or the
    payload is not a real report (0.10's ``{"exit_code": 1}``); ``RuntimeError`` when it carries no content."""
    if result.get("isError") is True:
        raise HealthReportUnavailable(_first_text(result, "health_report returned isError=true"))
    sc = result.get("structuredContent")
    if _is_valid_health_report(sc):
        return sc  # type: ignore[return-value]
    for text in _text_items(result):  # older builds: JSON text block with schema_version
        with suppress(ValueError, TypeError):
            if _is_valid_health_report(parsed := json.loads(text)):
                return parsed
    if isinstance(sc, dict):  # present but not a real report — unavailable, not fatal
        raise HealthReportUnavailable(f"health_report structuredContent lacks schema_version/overall/checks (keys={sorted(sc.keys())})")
    raise RuntimeError(f"health_report response carried neither structuredContent nor a parseable JSON text block. Result keys: {list(result.keys())}")

def _open_mcp(binary: str) -> subprocess.Popen:
    """Spawn ``<binary> mcp``; pin UTF-8 — cua-driver emits emoji/arbitrary paths and Windows' cp1252 would raise."""
    return subprocess.Popen([binary, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=windows_hide_flags(),
                            env=_sanitized_cua_env())

def _mcp_rpc(proc: subprocess.Popen, msg_id: int, method: str, params: Any = None) -> Report:
    """Write one JSON-RPC request and read one response line."""
    payload: Report = {"jsonrpc": "2.0", "id": msg_id, "method": method, **({"params": params} if params is not None else {})}
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        tail: List[str] = []
        with suppress(Exception):  # last 3 stderr lines, best-effort
            tail = [str(x) for x in (proc.stderr.read() or "").strip().splitlines()[-3:]]
        raise RuntimeError(f"cua-driver mcp produced no response for {method!r}. stderr tail: {tail or '(empty)'}")
    try:
        resp = json.loads(line)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"{method} response was not valid JSON: {e}\nraw: {line[:200]}")
    if "error" in resp:
        raise RuntimeError(f"{method} JSON-RPC error: {resp['error']}")
    return resp

def _call_tool(proc: subprocess.Popen, msg_id: int, name: str, arguments: Any = None) -> Any:
    """tools/call *name* → raw ``result`` value (``{}`` when absent)."""
    return _mcp_rpc(proc, msg_id, "tools/call", {"name": name, "arguments": arguments or {}}).get("result") or {}

@contextmanager
def _mcp_session(binary: str, timeout: float) -> Iterator[subprocess.Popen]:
    """Spawn ``<binary> mcp`` and always close stdin / wait / kill it on exit."""
    proc = _open_mcp(binary)
    try:
        yield proc
    finally:
        with suppress(Exception):
            proc.stdin.close()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

def _drive_health_report(binary: str, *, include: Sequence[str] = (), skip: Sequence[str] = (), timeout: float = 12.0) -> Report:
    """Handshake + `health_report` → report; HealthReportUnavailable (caller falls back) or RuntimeError."""
    args = {k: list(v) for k, v in (("include", include), ("skip", skip)) if v}
    with _mcp_session(binary, timeout) as proc:
        _mcp_rpc(proc, 1, "initialize", {})
        result = _call_tool(proc, 2, "health_report", args)
    if not isinstance(result, dict):
        raise RuntimeError(f"health_report result was not an object: {type(result).__name__}")
    return _extract_health_report_from_result(result)

def _structured(result: Report) -> Report:
    return result["structuredContent"] if isinstance(result.get("structuredContent"), dict) else {}

def _probe_tool(proc: subprocess.Popen, msg_id: int, name: str) -> Tuple[Optional[Report], Optional[str]]:
    """``(result, None)`` on success; ``(None, error_text)`` on isError or RPC failure."""
    try:
        result = _call_tool(proc, msg_id, name)
    except RuntimeError as e:
        return None, str(e)
    return (None, _first_text(result, f"{name} isError")) if result.get("isError") is True else (result, None)

def _drive_fallback_probes(binary: str, *, timeout: float = 12.0) -> Report:
    """One MCP session: initialize serverInfo version + check_permissions + list_apps probe results."""
    out: Report = dict.fromkeys(("init_version", "permissions", "permissions_error", "list_apps_ok", "list_apps_error", "list_apps_count"))
    with _mcp_session(binary, timeout) as proc:
        server_info = (_mcp_rpc(proc, 1, "initialize", {}).get("result") or {}).get("serverInfo") or {}
        out["init_version"] = server_info.get("version") if isinstance(server_info, dict) else None
        perms, out["permissions_error"] = _probe_tool(proc, 2, "check_permissions")  # primary TCC signal on 0.10
        out["permissions"] = _structured(perms) if perms is not None else None
        # list_apps — light AX capability probe; text-only success still counts as AX working
        apps, out["list_apps_error"] = _probe_tool(proc, 3, "list_apps")
        out["list_apps_ok"] = apps is not None
        app_list = _structured(apps).get("apps") if apps is not None else None
        out["list_apps_count"] = len(app_list) if isinstance(app_list, list) else None
    return out

def _platform_name() -> str:
    return (_platform_mod.system() or "").lower() or "unknown"

def _tcc_row(field: str, label: str, platform_bound: bool, ctx: Report) -> _Row:
    """tcc_* row for one check_permissions boolean *field* (Accessibility / Screen Recording)."""
    perms, err = ctx["perms"], ctx["perm_err"]
    if perms is None:
        return ("fail" if err else "skip"), err or "check_permissions unavailable", {}
    granted = perms.get(field)
    if not isinstance(granted, bool):  # only real booleans select a branch; anything else is "absent"
        off_platform = platform_bound and ctx["plat"] != "darwin"
        return "skip", f"not applicable on {ctx['plat']}" if off_platform else f"{field} field absent from check_permissions", {}
    if not granted:
        return "fail", f"{label} is not granted.", {"hint": _TCC_HINT.format(label), "data": {field: False}}
    data = {field: True, **({"screen_recording_capturable": perms.get("screen_recording_capturable")} if field == "screen_recording" else {})}
    if data.get("screen_recording_capturable") is False:  # the granted-but-not-capturable row wins over plain pass
        return "fail", "Screen Recording granted but not capturable.", {"hint": (
            "Screen Recording permission may need a restart of CuaDriver or a re-grant in System Settings."), "data": data}
    return "pass", f"{label} is granted.", {"data": data}

def _ax_capability_row(ctx: Report) -> _Row:
    """ax_capability — inferred from list_apps success or the accessibility grant."""
    probes, ax_granted = ctx["probes"], ctx["ax_granted"]
    list_ok, list_count = probes.get("list_apps_ok"), probes.get("list_apps_count")
    if list_ok is True:
        return "pass", "list_apps succeeded" + (f" ({list_count} apps)" if isinstance(list_count, int) else ""), {}
    if list_ok is False:
        return "fail", probes.get("list_apps_error") or ("list_apps failed" + (" despite accessibility grant" if ax_granted else "")), {}
    return ("pass", "inferred from accessibility grant (list_apps not probed)", {}) if ax_granted else ("skip", "not probed", {})

def _cli_doctor_row(txt: Optional[str]) -> Optional[_Row]:
    return None if not txt else (("pass" if "[ok" in txt.lower() or "ok  ]" in txt else "skip"), txt.splitlines()[0].strip(),
                                 {"data": {"snippet": txt[:2000]}})

# Fallback composite probe table, in emitted order: (check name, row builder(ctx) -> _Row | None to omit).
_FALLBACK_PROBES: Tuple[Tuple[str, Callable[[Report], Optional[_Row]]], ...] = (
    ("binary_version", lambda c: (c["ver_status"], c["ver_msg"], {})),
    ("platform_supported", lambda c: ("pass", f"platform={c['plat']}", {}) if c["plat"] in _SUPPORTED_PLATFORMS else ("fail", f"platform={c['plat']} (unsupported)", {})),
    # doctor does not start a session, so session_active is never probed
    ("session_active", lambda c: ("skip", "not probed (doctor does not open a cua session)", {})),
    ("tcc_accessibility", lambda c: _tcc_row("accessibility", "Accessibility", False, c)),
    ("tcc_screen_recording", lambda c: _tcc_row("screen_recording", "Screen Recording", True, c)),
    ("ax_capability", _ax_capability_row),
    ("health_report_path", lambda c: ("skip", "fallback composite (cua-driver 0.10 unclassified health_report); "
                                              f"cause: {c['reason_short']}", {})),
    ("cli_doctor", lambda c: _cli_doctor_row(c["doctor_txt"])),
)

def _overall_from(checks: List[Report]) -> str:
    """failed if binary missing/bad; ok if accessibility fine and nothing failed; else degraded."""
    by_name = {c.get("name"): c.get("status") for c in checks}
    if by_name.get("binary_version") != "pass":
        return "failed"
    ok = by_name.get("tcc_accessibility") in ("pass", "skip", None) and not any(c.get("status") == "fail" for c in checks)
    return "ok" if ok else "degraded"

def _compose_fallback_report(binary: str, *, reason: str = "", timeout: float = 12.0) -> Report:
    """schema_version=1 report from CLI + MCP probes (``_FALLBACK_PROBES``) when health_report is denied (0.10)."""
    plat = _platform_name()
    ver_status, ver_value = _cli_driver_version(binary)
    probes = _drive_fallback_probes(binary, timeout=timeout)
    if probes.get("init_version"):  # MCP initialize version beats a messy CLI parse
        ver_status, ver_value = "pass", str(probes["init_version"])
    perms = probes.get("permissions") if isinstance(probes.get("permissions"), dict) else None
    reason_short = (reason or "health_report unavailable").strip()
    reason_short = reason_short if len(reason_short) <= 160 else reason_short[:157] + "..."
    ctx: Report = {"plat": plat, "ver_status": ver_status, "probes": probes, "perms": perms,
                   "perm_err": probes.get("permissions_error"), "reason_short": reason_short,
                   "ver_msg": f"cua-driver {ver_value}" if ver_status == "pass" else (ver_value or "version unknown"),
                   "ax_granted": bool(perms and perms.get("accessibility") is True),
                   "doctor_txt": _cli_doctor_snippet(binary)}  # optional CLI doctor text (best-effort)
    checks = [{"name": name, "status": row[0], "message": row[1], **row[2]}  # hint/data only when present
              for name, build in _FALLBACK_PROBES if (row := build(ctx)) is not None]
    return {"schema_version": "1", "platform": plat,
            "driver_version": str(ver_value if ver_status == "pass" else (ver_value or "?")),
            "overall": _overall_from(checks), "checks": checks,
            "fallback": True, "fallback_reason": reason or "health_report unavailable"}

def _apply_display_count_guard(report: Report) -> Report:
    """Fail an 'ok' screen_capture_capability with ``display_count=0``: macOS ScreenCaptureKit reports 0 on headless
    / asleep panels — TCC fine, health_report ok, yet every capture is 0x0. Turns a silent failure actionable; applied
    at the report seam so the real and the fallback path both get it.

    Composed from PR #52949 (sujeet111) and PR #67259 (webtecnica).
    """
    checks = report.get("checks")
    for check in (c for c in (checks if isinstance(checks, list) else ()) if isinstance(c, dict) and c.get("name") == "screen_capture_capability"):
        data = check.get("data")
        if (data.get("display_count") if isinstance(data, dict) else None) == 0 and check.get("status") == "pass":
            check.update(status="fail", message=_ZERO_DISPLAY_MSG, hint=_ZERO_DISPLAY_HINT)
            if report.get("overall") == "ok":
                report["overall"] = "degraded"
    return report

def _wayland_environment_context(report: Report) -> Optional[Report]:
    """Linux+Wayland only: doctor probes the CLI process's environment, not the gateway's."""
    if report.get("platform") != "linux" or not os.environ.get("WAYLAND_DISPLAY"):
        return None
    return {"scope": "cli_process", "gateway_environment_checked": False}

def _print_text_report(report: Report, color: bool, *, identity: Optional[Report] = None,
                       environment: Optional[Report] = None) -> None:
    """Render like `cua-driver call health_report`: header (CLI --version preferred over health_report's stale
    ``driver_version``), identity block, environment note, one line per check + indented hint/``data`` rows
    (support staff need them)."""
    platform, report_v, overall = (report.get(k, "?") for k in ("platform", "driver_version", "overall"))
    identity = identity or {}
    cli_v = identity.get("cli_version") or ""
    header_v = cli_v or report_v  # binary's own --version wins when health_report is stale
    # No external color library — inline ANSI keeps doctor self-contained; colors only for known overall values.
    red, yellow, green, reset, dim = ("\033[31m", "\033[33m", "\033[32m", "\033[0m", "\033[2m") if color and overall in _OVERALL_GLYPH else ("",) * 5
    col_for = {"failed": red, "degraded": yellow, "ok": green}.get(overall, "")
    status_cols = {"pass": green, "fail": red, "skip": dim}
    lines = [f"{_OVERALL_GLYPH.get(overall, '•')} cua-driver {header_v} on {platform} — {col_for}{overall}{reset}"]
    if identity.get("resolved_binary"):
        lines.append(f"  {dim}binary: {identity['resolved_binary']}{reset}")
    if cli_v and report_v and str(report_v) not in str(cli_v) and str(cli_v) not in str(report_v):  # clearly differ
        lines += [f"  {dim}--version: {cli_v}{reset}", f"  {dim}health_report.driver_version: {report_v}{reset}"]
    if environment:
        lines += [f"  {dim}environment: current CLI process{reset}",
                  f"  {dim}gateway environment was not checked; active gateway computer_use sessions use that process environment{reset}"]
    if identity.get("version_mismatch"):
        lines += [f"  {yellow}⚠️ version mismatch: health_report says {report_v!r} but binary --version is {cli_v!r}{reset}",
                  f"  {dim}→ trust --version / packages/current for debugging; health_report's binary_version check can lag on Windows{reset}"]
    for check in report.get("checks", []):
        status = check.get("status", "?")
        lines.append(f"  {_STATUS_GLYPH.get(status, '•')} {status_cols.get(status, '')}{check.get('name', '?')}{reset}: {check.get('message') or ''}")
        if check.get("hint"):
            lines.append(f"      → {dim}{check['hint']}{reset}")
        data = check.get("data")
        for key, value in (data.items() if isinstance(data, dict) else ()):
            lines.append(f"      {dim}{key}={json.dumps(value) if isinstance(value, (dict, list)) else value}{reset}")
    print("\n".join(lines))

def run_doctor(driver_cmd: Optional[str] = None, *, include: Sequence[str] = (), skip: Sequence[str] = (), json_output: bool = False,
               color: Optional[bool] = None) -> int:
    """Resolve the binary via the shared runtime resolver (diagnose what `computer_use` actually invokes), call
    `health_report`, render; on 0.10.x (denied) a report is synthesized from probes."""
    # Windows' locale codec (cp1252, cp936, ...) cannot encode the ✅ ❌ ⚠️ ⏭️ glyphs — force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd
    binary = resolve_cua_driver_cmd(driver_cmd)
    if not binary:
        print(f"cua-driver: not installed (looked for {driver_cmd or 'cua-driver (PATH and canonical install paths)'!r}).\n  Run: hermes computer-use install")
        return 2
    try:  # prefer real health_report; on denial/non-schema, synthesize via probes
        try:
            report = _drive_health_report(binary, include=include, skip=skip, timeout=12.0)
        except HealthReportUnavailable as e:
            report = _compose_fallback_report(binary, reason=str(e), timeout=12.0)
    except RuntimeError as e:
        print(f"cua-driver health_report failed: {e}", file=sys.stderr)
        return 2
    report = _apply_display_count_guard(report)
    identity = _build_identity(binary, report)
    environment = _wayland_environment_context(report)
    if json_output:
        # Additive envelope: upstream keys preserved, identity under hermes_identity (and environment under
        # hermes_environment when present) so overall/checks parsers keep working.
        payload = {**report, "hermes_identity": identity}
        if environment:
            payload["hermes_environment"] = environment
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_text_report(report, color=sys.stdout.isatty() if color is None else bool(color), identity=identity,
                           environment=environment)
    return 0 if report.get("overall") == "ok" else 1  # unknown/missing overall must not look like success
