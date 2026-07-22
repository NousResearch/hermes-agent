"""
`hermes computer-use doctor` — thin client for cua-driver's `health_report` MCP tool.

cua-driver owns the health model (#1908 / be761fac on `main`). This module
just drives the stdio JSON-RPC handshake, calls `health_report`, and
renders the structured response. When the driver gets new checks, they
flow through here without code changes on the Hermes side — the only
contract is the stable `schema_version="1"` payload shape.

cua-driver 0.10.x marks `health_report` with risk.class='unclassified', so
MCP tools/call returns isError=true ("Permission denied: ... no reviewed
risk classification") with structuredContent ``{"exit_code": 1}``. That is
NOT a schema_version=1 report — we detect it and synthesize a composite
report via working probes (check_permissions, list_apps, CLI --version).

Exit code conventions:
- 0: overall == "ok"
- 1: overall in ("degraded", "failed")
- 2: driver binary missing / unreachable / protocol error
"""

from __future__ import annotations

import json
import os
import platform as _platform_mod
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Match the ALLOWED_STATUS_VALUES + ALLOWED_OVERALL_VALUES the cua-driver
# integration test pins. If health_report widens its vocabulary, add here.
_STATUS_GLYPH = {
    "pass": "✅",
    "fail": "❌",
    "skip": "⏭️",
}
_OVERALL_GLYPH = {
    "ok":       "✅",
    "degraded": "⚠️",
    "failed":   "❌",
}


class HealthReportUnavailable(RuntimeError):
    """health_report MCP tool denied or returned a non-schema payload.

    Raised so ``run_doctor`` can fall back to composite probes that work on
    cua-driver builds where ``health_report`` is risk-unclassified (0.10.x).
    """


def _cua_child_env() -> Dict[str, str]:
    """cua-driver child env with the Hermes telemetry policy applied.

    Delegates to ``cua_backend.cua_driver_child_env`` (telemetry disabled by
    default unless the user opts in). Falls back to the current environment
    if that import fails, so doctor never breaks on a telemetry-helper error.
    """
    try:
        from tools.computer_use.cua_backend import cua_driver_child_env

        return cua_driver_child_env()
    except Exception:
        return dict(os.environ)


def _sanitized_cua_env() -> Dict[str, str]:
    """Telemetry-policy env with Hermes provider secrets stripped.

    cua-driver is a third-party binary — it must never inherit provider
    API keys (#53503/#55709/#58889 lineage). Falls back to the unsanitized
    telemetry env if the sanitizer can't be imported, so doctor keeps
    working in stripped-down environments.
    """
    env = _cua_child_env()
    try:
        from tools.environments.local import _sanitize_subprocess_env

        return _sanitize_subprocess_env(env)
    except Exception:
        return env


def _is_valid_health_report(payload: Any) -> bool:
    """True when *payload* looks like a schema_version=1 health_report."""
    if not isinstance(payload, dict):
        return False
    if "schema_version" not in payload:
        return False
    if "overall" not in payload:
        return False
    if not isinstance(payload.get("checks"), list):
        return False
    return True


def _extract_health_report_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pull a schema_version=1 report out of an MCP tools/call result.

    Raises ``HealthReportUnavailable`` when the tool denied the call
    (isError) or the payload is not a real health report (e.g. 0.10's
    ``{"exit_code": 1}`` structuredContent on unclassified denial).
    Raises ``RuntimeError`` when the response shape is unusable for other
    reasons (no content at all).
    """
    if result.get("isError") is True:
        # Prefer the human text; fall back to a generic denial message.
        denial = "health_report returned isError=true"
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                text = (item.get("text") or "").strip()
                if text:
                    denial = text
                    break
        raise HealthReportUnavailable(denial)

    sc = result.get("structuredContent")
    if _is_valid_health_report(sc):
        return sc  # type: ignore[return-value]

    # Older builds: JSON text block with schema_version.
    for item in result.get("content") or []:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text", "")
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            continue
        if _is_valid_health_report(parsed):
            return parsed

    # structuredContent present but not a real report (the 0.10 unclassified
    # path ships {"exit_code": 1}) — treat as unavailable, not fatal protocol.
    if isinstance(sc, dict):
        raise HealthReportUnavailable(
            "health_report structuredContent lacks schema_version/overall/checks "
            f"(keys={sorted(sc.keys())})"
        )

    raise RuntimeError(
        "health_report response carried neither structuredContent nor a parseable "
        f"JSON text block. Result keys: {list(result.keys())}"
    )


def _open_mcp(binary: str) -> subprocess.Popen:
    """Spawn ``<binary> mcp`` with UTF-8 + sanitized env."""
    # cua-driver emits UTF-8 (containing emoji in check messages on macOS
    # and arbitrary file paths on Windows). The Python default
    # text-mode encoding follows the system locale — `cp1252` on a
    # default Windows install — which raises UnicodeDecodeError on the
    # first non-ASCII byte. Pin the codec.
    return subprocess.Popen(
        [binary, "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=_sanitized_cua_env(),
    )


def _mcp_rpc(proc: subprocess.Popen, msg_id: int, method: str, params: Any = None) -> Dict[str, Any]:
    """Write one JSON-RPC request and read one response line."""
    assert proc.stdin is not None and proc.stdout is not None
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        stderr_tail: List[str] = []
        if proc.stderr is not None:
            try:
                raw_err = proc.stderr.read() or ""
                stderr_tail = [str(x) for x in raw_err.strip().splitlines()[-3:]]
            except Exception:
                pass
        raise RuntimeError(
            f"cua-driver mcp produced no response for {method!r}. "
            f"stderr tail: {stderr_tail or '(empty)'}"
        )
    try:
        resp = json.loads(line)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"{method} response was not valid JSON: {e}\nraw: {line[:200]}")
    if "error" in resp:
        raise RuntimeError(f"{method} JSON-RPC error: {resp['error']}")
    return resp


def _close_mcp(proc: subprocess.Popen, timeout: float) -> None:
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _drive_health_report(
    binary: str,
    *,
    include: Sequence[str] = (),
    skip: Sequence[str] = (),
    timeout: float = 12.0,
) -> Dict[str, Any]:
    """Spawn `<binary> mcp`, perform the JSON-RPC handshake, call
    `health_report`, and return the parsed schema_version=1 report.

    Raises:
      HealthReportUnavailable: tool denied (isError) or non-schema payload
        (cua-driver 0.10 unclassified). Caller should fall back.
      RuntimeError: protocol-level failure (binary crash, malformed JSON,
        JSON-RPC error, empty content).
    """
    args: Dict[str, Any] = {}
    if include:
        args["include"] = list(include)
    if skip:
        args["skip"] = list(skip)

    proc = _open_mcp(binary)
    try:
        # 1. initialize
        init_resp = _mcp_rpc(proc, 1, "initialize", {})
        _ = init_resp  # handshake only

        # 2. tools/call health_report
        call_resp = _mcp_rpc(
            proc,
            2,
            "tools/call",
            {"name": "health_report", "arguments": args},
        )
    finally:
        _close_mcp(proc, timeout)

    result = call_resp.get("result") or {}
    if not isinstance(result, dict):
        raise RuntimeError(f"health_report result was not an object: {type(result).__name__}")
    return _extract_health_report_from_result(result)


def _cli_driver_version(binary: str, timeout: float = 5.0) -> Tuple[str, Optional[str]]:
    """Return (status, version_or_message) from ``cua-driver --version``."""
    try:
        completed = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_sanitized_cua_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return "fail", f"--version failed: {e}"

    text = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if completed.returncode != 0 and not text:
        return "fail", f"--version exited {completed.returncode}"

    # Typical: "cua-driver 0.10.0"
    m = re.search(r"(\d+\.\d+\.\d+(?:[-+][\w.]+)?)", text)
    version = m.group(1) if m else (text.splitlines()[0] if text else "unknown")
    if completed.returncode != 0:
        return "fail", version
    return "pass", version


def _cli_doctor_snippet(binary: str, timeout: float = 8.0) -> Optional[str]:
    """Optional one-shot ``cua-driver doctor`` text (best-effort, never fatal)."""
    try:
        completed = subprocess.run(
            [binary, "doctor"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_sanitized_cua_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return out or None


def _drive_fallback_probes(
    binary: str,
    *,
    timeout: float = 12.0,
) -> Dict[str, Any]:
    """Call working MCP tools (check_permissions, list_apps) in one session.

    Returns a dict with keys:
      - init_version: str | None (from initialize serverInfo)
      - permissions: structuredContent dict | None
      - permissions_error: str | None
      - list_apps_ok: bool | None
      - list_apps_error: str | None
      - list_apps_count: int | None
    """
    out: Dict[str, Any] = {
        "init_version": None,
        "permissions": None,
        "permissions_error": None,
        "list_apps_ok": None,
        "list_apps_error": None,
        "list_apps_count": None,
    }
    proc = _open_mcp(binary)
    try:
        init_resp = _mcp_rpc(proc, 1, "initialize", {})
        server_info = ((init_resp.get("result") or {}).get("serverInfo") or {})
        if isinstance(server_info, dict):
            out["init_version"] = server_info.get("version")

        # check_permissions — primary TCC signal on 0.10
        try:
            perm_resp = _mcp_rpc(
                proc, 2, "tools/call", {"name": "check_permissions", "arguments": {}}
            )
            perm_result = perm_resp.get("result") or {}
            if perm_result.get("isError") is True:
                msg = "check_permissions isError"
                for item in perm_result.get("content") or []:
                    if isinstance(item, dict) and item.get("type") == "text":
                        t = (item.get("text") or "").strip()
                        if t:
                            msg = t
                            break
                out["permissions_error"] = msg
            else:
                sc = perm_result.get("structuredContent")
                out["permissions"] = sc if isinstance(sc, dict) else {}
        except RuntimeError as e:
            out["permissions_error"] = str(e)

        # list_apps — light AX capability probe
        try:
            apps_resp = _mcp_rpc(
                proc, 3, "tools/call", {"name": "list_apps", "arguments": {}}
            )
            apps_result = apps_resp.get("result") or {}
            if apps_result.get("isError") is True:
                msg = "list_apps isError"
                for item in apps_result.get("content") or []:
                    if isinstance(item, dict) and item.get("type") == "text":
                        t = (item.get("text") or "").strip()
                        if t:
                            msg = t
                            break
                out["list_apps_ok"] = False
                out["list_apps_error"] = msg
            else:
                sc = apps_result.get("structuredContent") or {}
                apps = sc.get("apps") if isinstance(sc, dict) else None
                if isinstance(apps, list):
                    out["list_apps_ok"] = True
                    out["list_apps_count"] = len(apps)
                else:
                    # text-only success still counts as AX working
                    out["list_apps_ok"] = True
                    out["list_apps_count"] = None
        except RuntimeError as e:
            out["list_apps_ok"] = False
            out["list_apps_error"] = str(e)
    finally:
        _close_mcp(proc, timeout)

    return out


def _platform_name() -> str:
    sysname = (_platform_mod.system() or "").lower()
    if sysname == "darwin":
        return "darwin"
    if sysname == "windows":
        return "windows"
    if sysname == "linux":
        return "linux"
    return sysname or "unknown"


def _compose_fallback_report(
    binary: str,
    *,
    reason: str = "",
    timeout: float = 12.0,
) -> Dict[str, Any]:
    """Build a schema_version=1 report from CLI + working MCP probes.

    Used when ``health_report`` is denied (unclassified risk on 0.10) or
    returns a non-schema payload. Compatible with ``_print_text_report``.
    """
    plat = _platform_name()
    checks: List[Dict[str, Any]] = []

    ver_status, ver_value = _cli_driver_version(binary)
    driver_version = ver_value if ver_status == "pass" else (ver_value or "?")
    # Prefer MCP initialize version when CLI parse is messy
    probes = _drive_fallback_probes(binary, timeout=timeout)
    if probes.get("init_version"):
        driver_version = str(probes["init_version"])
        ver_status = "pass"
        ver_msg = f"cua-driver {driver_version}"
    else:
        ver_msg = (
            f"cua-driver {ver_value}" if ver_status == "pass" else (ver_value or "version unknown")
        )

    checks.append({
        "name": "binary_version",
        "status": ver_status,
        "message": ver_msg,
    })

    # platform_supported — doctor runs wherever the binary runs
    supported = plat in ("darwin", "linux", "windows")
    checks.append({
        "name": "platform_supported",
        "status": "pass" if supported else "fail",
        "message": f"platform={plat}" + ("" if supported else " (unsupported)"),
    })

    # session_active — we don't start a session in doctor; mark skip
    checks.append({
        "name": "session_active",
        "status": "skip",
        "message": "not probed (doctor does not open a cua session)",
    })

    perms = probes.get("permissions") if isinstance(probes.get("permissions"), dict) else None
    perm_err = probes.get("permissions_error")

    if perms is not None:
        ax = perms.get("accessibility")
        scr = perms.get("screen_recording")
        capturable = perms.get("screen_recording_capturable")

        if ax is True:
            checks.append({
                "name": "tcc_accessibility",
                "status": "pass",
                "message": "Accessibility is granted.",
                "data": {"accessibility": True},
            })
        elif ax is False:
            checks.append({
                "name": "tcc_accessibility",
                "status": "fail",
                "message": "Accessibility is not granted.",
                "hint": "Grant Accessibility to CuaDriver in System Settings → Privacy & Security.",
                "data": {"accessibility": False},
            })
        else:
            checks.append({
                "name": "tcc_accessibility",
                "status": "skip",
                "message": "accessibility field absent from check_permissions",
            })

        if scr is True and capturable is False:
            checks.append({
                "name": "tcc_screen_recording",
                "status": "fail",
                "message": "Screen Recording granted but not capturable.",
                "hint": (
                    "Screen Recording permission may need a restart of CuaDriver "
                    "or a re-grant in System Settings."
                ),
                "data": {
                    "screen_recording": True,
                    "screen_recording_capturable": False,
                },
            })
        elif scr is True:
            checks.append({
                "name": "tcc_screen_recording",
                "status": "pass",
                "message": "Screen Recording is granted.",
                "data": {
                    "screen_recording": True,
                    "screen_recording_capturable": capturable,
                },
            })
        elif scr is False:
            checks.append({
                "name": "tcc_screen_recording",
                "status": "fail",
                "message": "Screen Recording is not granted.",
                "hint": "Grant Screen Recording to CuaDriver in System Settings → Privacy & Security.",
                "data": {"screen_recording": False},
            })
        else:
            # Non-macOS or field absent
            if plat == "darwin":
                checks.append({
                    "name": "tcc_screen_recording",
                    "status": "skip",
                    "message": "screen_recording field absent from check_permissions",
                })
            else:
                checks.append({
                    "name": "tcc_screen_recording",
                    "status": "skip",
                    "message": f"not applicable on {plat}",
                })
    else:
        checks.append({
            "name": "tcc_accessibility",
            "status": "fail" if perm_err else "skip",
            "message": perm_err or "check_permissions unavailable",
        })
        checks.append({
            "name": "tcc_screen_recording",
            "status": "fail" if perm_err else "skip",
            "message": perm_err or "check_permissions unavailable",
        })

    # ax_capability — infer from list_apps success or accessibility grant
    list_ok = probes.get("list_apps_ok")
    list_err = probes.get("list_apps_error")
    list_count = probes.get("list_apps_count")
    ax_granted = bool(perms and perms.get("accessibility") is True)
    if list_ok is True:
        count_msg = f" ({list_count} apps)" if isinstance(list_count, int) else ""
        checks.append({
            "name": "ax_capability",
            "status": "pass",
            "message": f"list_apps succeeded{count_msg}",
        })
    elif list_ok is False:
        checks.append({
            "name": "ax_capability",
            "status": "fail",
            "message": (
                list_err
                or (
                    "list_apps failed despite accessibility grant"
                    if ax_granted
                    else "list_apps failed"
                )
            ),
        })
    elif ax_granted:
        checks.append({
            "name": "ax_capability",
            "status": "pass",
            "message": "inferred from accessibility grant (list_apps not probed)",
        })
    else:
        checks.append({
            "name": "ax_capability",
            "status": "skip",
            "message": "not probed",
        })

    # Annotate that we used the fallback path
    reason_short = (reason or "health_report unavailable").strip()
    if len(reason_short) > 160:
        reason_short = reason_short[:157] + "..."
    checks.append({
        "name": "health_report_path",
        "status": "skip",
        "message": (
            "fallback composite (cua-driver 0.10 unclassified health_report); "
            f"cause: {reason_short}"
        ),
    })

    # Optional CLI doctor text (best-effort)
    doctor_txt = _cli_doctor_snippet(binary)
    if doctor_txt:
        first = doctor_txt.splitlines()[0].strip()
        cli_ok = "[ok" in doctor_txt.lower() or "ok  ]" in doctor_txt
        checks.append({
            "name": "cli_doctor",
            "status": "pass" if cli_ok else "skip",
            "message": first,
            "data": {"snippet": doctor_txt[:2000]},
        })

    # Normalize any accidental non-vocab status values
    for c in checks:
        if c.get("status") not in ("pass", "fail", "skip"):
            c["status"] = "fail"

    # overall: ok if TCC+binary ok; degraded if partial; failed if binary missing/bad
    status_by_name = {c.get("name"): c.get("status") for c in checks}
    binary_ok = status_by_name.get("binary_version") == "pass"
    tcc_ax_status = status_by_name.get("tcc_accessibility")
    tcc_ok = tcc_ax_status in ("pass", "skip", None)
    fail_count = sum(1 for c in checks if c.get("status") == "fail")

    if not binary_ok:
        overall = "failed"
    elif tcc_ok and fail_count == 0:
        overall = "ok"
    elif tcc_ok and fail_count > 0:
        # Binary + accessibility fine, but something else failed (e.g. screen
        # recording) → degraded rather than failed.
        overall = "degraded"
    else:
        # Accessibility denied or broken — computer-use is partially/fully blocked.
        overall = "degraded"

    return {
        "schema_version": "1",
        "platform": plat,
        "driver_version": str(driver_version),
        "overall": overall,
        "checks": checks,
        "fallback": True,
        "fallback_reason": reason or "health_report unavailable",
    }


def _drive_health_report_or_fallback(
    binary: str,
    *,
    include: Sequence[str] = (),
    skip: Sequence[str] = (),
    timeout: float = 12.0,
) -> Dict[str, Any]:
    """Prefer real health_report; on denial/non-schema, synthesize via probes."""
    try:
        return _drive_health_report(
            binary, include=include, skip=skip, timeout=timeout,
        )
    except HealthReportUnavailable as e:
        return _compose_fallback_report(
            binary, reason=str(e), timeout=timeout,
        )


def _print_text_report(report: Dict[str, Any], color: bool) -> None:
    """Render the report in the same style as `cua-driver call health_report`
    would (one line per check + a summary footer)."""
    schema = report.get("schema_version", "?")
    platform = report.get("platform", "?")
    driver_v = report.get("driver_version", "?")
    overall = report.get("overall", "?")

    header_glyph = _OVERALL_GLYPH.get(overall, "•")

    if color and overall in _OVERALL_GLYPH:
        # No external color library — keep ANSI inline so the doctor
        # command stays a single self-contained module.
        col_red = "\033[31m"
        col_yellow = "\033[33m"
        col_green = "\033[32m"
        col_reset = "\033[0m"
        col_dim = "\033[2m"
        col_for = {"failed": col_red, "degraded": col_yellow, "ok": col_green}.get(overall, "")
    else:
        col_red = col_yellow = col_green = col_reset = col_dim = ""
        col_for = ""

    print(
        f"{header_glyph} cua-driver {driver_v} on {platform} — "
        f"{col_for}{overall}{col_reset}"
    )

    for check in report.get("checks", []):
        name = check.get("name", "?")
        status = check.get("status", "?")
        glyph = _STATUS_GLYPH.get(status, "•")
        message = check.get("message") or ""
        if color:
            status_col = {
                "pass": col_green, "fail": col_red, "skip": col_dim,
            }.get(status, "")
            print(f"  {glyph} {status_col}{name}{col_reset}: {message}")
        else:
            print(f"  {glyph} {name}: {message}")
        hint = check.get("hint")
        if hint:
            print(f"      → {col_dim}{hint}{col_reset}")
        # `data` is the structured payload some checks attach (bundle id,
        # AX permission state, version triple, etc.). Surface when present
        # because users / support staff frequently need it.
        data = check.get("data")
        if isinstance(data, dict) and data:
            for key, value in data.items():
                rendered = value if not isinstance(value, (dict, list)) else json.dumps(value)
                print(f"      {col_dim}{key}={rendered}{col_reset}")
    _ = schema  # acknowledge field for forward-compat readers


def run_doctor(
    driver_cmd: Optional[str] = None,
    *,
    include: Sequence[str] = (),
    skip: Sequence[str] = (),
    json_output: bool = False,
    color: Optional[bool] = None,
) -> int:
    """Resolve the cua-driver binary, call `health_report`, render the result.

    Honors `HERMES_CUA_DRIVER_CMD` via the same `_cua_driver_cmd()` resolver
    that `install_cua_driver` + the runtime backend use, so the doctor
    diagnoses what your `computer_use` toolset will actually invoke.

    On cua-driver 0.10.x, ``health_report`` may be risk-unclassified and
    denied; doctor then synthesizes a schema_version=1 report from
    check_permissions / list_apps / CLI probes instead of printing
    ``• cua-driver ? on ? — ?``.
    """
    # Windows ships stdout/stderr wrapped with the system ANSI codec
    # (`cp1252` on a US locale, `cp936` on zh-CN, etc.). The check-matrix
    # output below contains ✅ ❌ ⚠️ ⏭️ glyphs — none of them encodable
    # in those codepages. Switch stdout to UTF-8 once, idempotently: every
    # supported TextIOWrapper (Py3.7+) has `.reconfigure`, and a no-op
    # re-encode is cheap if we were already UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    if driver_cmd is None:
        try:
            from hermes_cli.tools_config import _cua_driver_cmd
            driver_cmd = _cua_driver_cmd()
        except Exception:
            driver_cmd = os.environ.get("HERMES_CUA_DRIVER_CMD") or "cua-driver"

    binary = shutil.which(driver_cmd)
    if not binary:
        print(f"cua-driver: not installed (looked for {driver_cmd!r}).")
        print("  Run: hermes computer-use install")
        return 2

    try:
        report = _drive_health_report_or_fallback(
            binary, include=include, skip=skip,
        )
    except RuntimeError as e:
        print(f"cua-driver health_report failed: {e}", file=sys.stderr)
        return 2

    if json_output:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if color is None:
            color = sys.stdout.isatty()
        _print_text_report(report, color=bool(color))

    overall = report.get("overall")
    if overall in ("degraded", "failed"):
        return 1
    if overall != "ok":
        # Unknown / missing overall after fallback should not look like success.
        return 1
    return 0
