"""
`hermes computer-use doctor` — diagnostics adapter for cua-driver.

cua-driver owns the health model (#1908 / be761fac on `main`). This module
prefers the stable `health_report` MCP tool and renders its structured
response. Newer cua-driver permission policies can deny that tool before
dispatch; in that case this module falls back to the driver's supported
`doctor --json` CLI without bypassing the policy.

Exit code conventions:
- 0: overall == "ok"
- 1: overall in ("degraded", "failed")
- 2: driver binary missing / unreachable / protocol error
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence


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


class _HealthReportPolicyDenied(RuntimeError):
    """The MCP exchange worked, but policy denied health_report dispatch."""


def _is_health_report(value: Any) -> bool:
    """Return whether *value* satisfies the stable health-report envelope."""
    return (
        isinstance(value, dict)
        and value.get("schema_version") == "1"
        and isinstance(value.get("platform"), str)
        and isinstance(value.get("driver_version"), str)
        and value.get("overall") in _OVERALL_GLYPH
        and isinstance(value.get("checks"), list)
    )


def _is_cli_doctor_report(value: Any) -> bool:
    """Return whether *value* is the current `cua-driver doctor --json` shape."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("ok"), bool)
        and isinstance(value.get("probes"), list)
    )


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


def _drive_health_report(
    binary: str,
    *,
    include: Sequence[str] = (),
    skip: Sequence[str] = (),
    timeout: float = 12.0,
) -> Dict[str, Any]:
    """Spawn `<binary> mcp`, perform the JSON-RPC handshake, call
    `health_report`, and return the parsed `structuredContent` dict.

    Raises `RuntimeError` on a protocol-level failure (binary crash,
    malformed response, JSON-RPC error). Never raises on a `health_report`
    that has failing checks — the tool's contract is to always return a
    well-formed report with `overall` set, never to set `isError`.
    """
    args: Dict[str, Any] = {}
    if include:
        args["include"] = list(include)
    if skip:
        args["skip"] = list(skip)

    # cua-driver emits UTF-8 (containing emoji in check messages on macOS
    # and arbitrary file paths on Windows). The Python default
    # text-mode encoding follows the system locale — `cp1252` on a
    # default Windows install — which raises UnicodeDecodeError on the
    # first non-ASCII byte. Pin the codec.
    proc = subprocess.Popen(
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
    try:
        # 1. initialize
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "initialize", "params": {},
        }) + "\n")
        proc.stdin.flush()
        init_line = proc.stdout.readline()
        if not init_line:
            stderr_tail = (proc.stderr.read() or "").strip().splitlines()[-3:]
            raise RuntimeError(
                f"cua-driver mcp produced no initialize response. "
                f"stderr tail: {stderr_tail or '(empty)'}"
            )

        # 2. tools/call health_report
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 2,
            "method": "tools/call",
            "params": {"name": "health_report", "arguments": args},
        }) + "\n")
        proc.stdin.flush()
        call_line = proc.stdout.readline()
        if not call_line:
            raise RuntimeError("cua-driver mcp closed stdout without responding to health_report.")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    try:
        resp = json.loads(call_line)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"health_report response was not valid JSON: {e}\nraw: {call_line[:200]}")

    if "error" in resp:
        raise RuntimeError(f"health_report JSON-RPC error: {resp['error']}")

    result = resp.get("result") or {}

    # A denied MCP tool call is still a successful JSON-RPC exchange. Current
    # cua-driver versions encode that denial as result.isError plus a small
    # structuredContent object (for example {"exit_code": 1}). Never mistake
    # that error envelope for the documented health-report schema.
    if result.get("isError"):
        messages = [
            str(item.get("text", "")).strip()
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        detail = next((message for message in messages if message), "MCP tool call failed")
        if detail.lower().startswith("permission denied"):
            raise _HealthReportPolicyDenied(detail)
        raise RuntimeError(f"health_report MCP tool failed: {detail}")

    # Preferred: structuredContent (cua-driver-rs always emits it on the
    # health_report response). Fall back to parsing the first text item
    # as JSON for older cua-driver builds that didn't carry structuredContent.
    sc = result.get("structuredContent")
    if _is_health_report(sc):
        return sc

    for item in result.get("content", []):
        if item.get("type") == "text":
            text = item.get("text", "")
            try:
                # Many health_report payloads ship JSON in the text item too.
                parsed = json.loads(text)
                if _is_health_report(parsed):
                    return parsed
            except (ValueError, TypeError):
                pass

    raise RuntimeError(
        "health_report response carried neither structuredContent nor a parseable "
        f"JSON text block. Result keys: {list(result.keys())}"
    )


def _drive_cli_doctor(binary: str, *, timeout: float = 12.0) -> Dict[str, Any]:
    """Run the driver's supported machine-readable self-diagnostic command."""
    try:
        completed = subprocess.run(
            [binary, "doctor", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_sanitized_cua_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"cua-driver doctor could not run: {e}") from e

    try:
        report = json.loads(completed.stdout)
    except (ValueError, TypeError) as e:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(
            f"cua-driver doctor returned invalid JSON: {e}; stderr: {stderr or '(empty)'}"
        ) from e

    if not _is_cli_doctor_report(report):
        raise RuntimeError(
            "cua-driver doctor returned an unrecognised payload; "
            f"keys: {list(report.keys()) if isinstance(report, dict) else type(report).__name__}"
        )
    return report


def _cli_doctor_identity(report: Dict[str, Any]) -> tuple[str, str]:
    """Extract version and platform from the driver's stable binary probe."""
    for probe in report.get("probes", []):
        if isinstance(probe, dict) and probe.get("label") == "binary":
            message = str(probe.get("message") or "")
            match = re.search(r"\bcua-driver\s+(\S+)\s+\([^)]*-(windows|linux|macos)\)", message)
            if match:
                platform = {"windows": "win32", "macos": "darwin"}.get(
                    match.group(2), match.group(2)
                )
                return match.group(1), platform
    return "?", sys.platform


def _print_cli_doctor_report(report: Dict[str, Any], color: bool) -> None:
    """Render the current `cua-driver doctor --json` probe report."""
    probes = [probe for probe in report.get("probes", []) if isinstance(probe, dict)]
    has_warning = any(probe.get("status") == "warn" for probe in probes)
    ok = bool(report.get("ok"))
    overall = "ok" if ok else "failed"
    summary = "ok (warnings present)" if ok and has_warning else overall
    version, platform = _cli_doctor_identity(report)
    glyph = "✅" if ok else "❌"

    if color:
        col_red = "\033[31m"
        col_yellow = "\033[33m"
        col_green = "\033[32m"
        col_reset = "\033[0m"
        col_dim = "\033[2m"
        header_col = col_yellow if has_warning and ok else (col_green if ok else col_red)
    else:
        col_red = col_yellow = col_green = col_reset = col_dim = header_col = ""

    print(f"{glyph} cua-driver {version} on {platform} — {header_col}{summary}{col_reset}")
    glyphs = {"ok": "✅", "warn": "⚠️", "err": "❌"}
    colours = {"ok": col_green, "warn": col_yellow, "err": col_red}
    for probe in probes:
        status = str(probe.get("status") or "?")
        label = str(probe.get("label") or "?")
        message = str(probe.get("message") or "")
        status_col = colours.get(status, "") if color else ""
        print(f"  {glyphs.get(status, '•')} {status_col}{label}{col_reset}: {message}")
        detail = probe.get("detail")
        if detail:
            for line in str(detail).splitlines():
                print(f"      {col_dim}{line}{col_reset}")


def _print_text_report(report: Dict[str, Any], color: bool) -> None:
    """Render the report in the same style as `cua-driver call health_report`
    would (one line per check + a summary footer)."""
    if _is_cli_doctor_report(report):
        _print_cli_doctor_report(report, color=color)
        return

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
        report = _drive_health_report(binary, include=include, skip=skip)
    except _HealthReportPolicyDenied as health_error:
        # Current cua-driver policy can deny health_report before dispatch even
        # though it is read-only. Its own `doctor --json` command is the
        # supported local diagnostic path and does not weaken that policy.
        if include or skip:
            print(
                "cua-driver health_report failed and the CLI fallback does not "
                f"support include/skip filters: {health_error}",
                file=sys.stderr,
            )
            return 2
        try:
            report = _drive_cli_doctor(binary)
        except RuntimeError as doctor_error:
            print(
                f"cua-driver diagnostics failed: health_report: {health_error}; "
                f"doctor --json: {doctor_error}",
                file=sys.stderr,
            )
            return 2
    except RuntimeError as health_error:
        print(f"cua-driver health_report failed: {health_error}", file=sys.stderr)
        return 2

    if json_output:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if color is None:
            color = sys.stdout.isatty()
        _print_text_report(report, color=bool(color))

    if _is_cli_doctor_report(report):
        return 0 if report.get("ok") else 1

    overall = report.get("overall")
    if overall in ("degraded", "failed"):
        return 1
    return 0
