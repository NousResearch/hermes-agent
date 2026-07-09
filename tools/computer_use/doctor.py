"""
`hermes computer-use doctor` — thin client for cua-driver's `health_report` MCP tool.

cua-driver owns the health model (#1908 / be761fac on `main`). This module
just drives the stdio JSON-RPC handshake, calls `health_report`, and
renders the structured response. When the driver gets new checks, they
flow through here without code changes on the Hermes side — the only
contract is the stable `schema_version="1"` payload shape.

Exit code conventions:
- 0: overall == "ok"
- 1: overall in ("degraded", "failed")
- 2: driver binary missing / unreachable / protocol error
"""

from __future__ import annotations

import json
import os
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
            proc.wait(timeout=10)

    try:
        resp = json.loads(call_line)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"health_report response was not valid JSON: {e}\nraw: {call_line[:200]}")

    if "error" in resp:
        raise RuntimeError(f"health_report JSON-RPC error: {resp['error']}")

    result = resp.get("result") or {}

    # Preferred: structuredContent (cua-driver-rs always emits it on the
    # health_report response). Fall back to parsing the first text item
    # as JSON for older cua-driver builds that didn't carry structuredContent.
    sc = result.get("structuredContent")
    if isinstance(sc, dict):
        return sc

    for item in result.get("content", []):
        if item.get("type") == "text":
            text = item.get("text", "")
            try:
                # Many health_report payloads ship JSON in the text item too.
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "schema_version" in parsed:
                    return parsed
            except (ValueError, TypeError):
                pass

    raise RuntimeError(
        "health_report response carried neither structuredContent nor a parseable "
        f"JSON text block. Result keys: {list(result.keys())}"
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
    return 0
