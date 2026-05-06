"""Internal web acquisition helpers.

This module is intentionally not registered as a Hermes tool. It provides a
small internal adapter for the optional Scrapling pilot runtime so future web
acquisition code can call a difficult-extraction fallback without exposing a new
public tool surface.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRAPLING_EXTRACT_SCRIPT = (
    REPO_ROOT
    / "optional-skills"
    / "research"
    / "scrapling"
    / "scripts"
    / "scrapling_extract.py"
)
DEFAULT_SCRAPLING_RUNTIME_PYTHON = Path.home() / ".hermes" / "runtimes" / "scrapling" / "bin" / "python"
MAX_ERROR_FIELD_CHARS = 2_000


VALID_MODES = {"static", "dynamic", "stealth"}
VALID_SELECTOR_TYPES = {"css", "xpath", "text", "regex"}


def default_scrapling_runtime_python() -> Path:
    """Return the isolated Scrapling runtime Python path.

    Do not point this at the Hermes main venv; Scrapling remains an optional
    pilot/fallback dependency.
    """

    return DEFAULT_SCRAPLING_RUNTIME_PYTHON


def _truncate(value: str, limit: int = MAX_ERROR_FIELD_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _empty_receipt(
    *,
    url: str,
    selector: str,
    selector_type: str = "css",
    mode: str = "static",
    fallback_reason: str = "selector_required",
    error_type: str,
    message: str,
    **extra: str,
) -> dict:
    error = {"type": error_type, "message": message}
    for key, value in extra.items():
        error[key] = _truncate(str(value))
    return {
        "backend": "scrapling",
        "mode": mode,
        "url": url,
        "selector": selector,
        "selector_type": selector_type,
        "content": "",
        "elapsed_ms": 0,
        "fallback_reason": fallback_reason,
        "errors": [error],
    }


def build_scrapling_command(
    *,
    url: str,
    selector: str,
    selector_type: str = "css",
    mode: str = "static",
    fallback_reason: str = "selector_required",
    timeout: int = 20,
    wait_selector: str | None = None,
    network_idle: bool = False,
    max_chars: int = 50_000,
    runtime_python: str | Path | None = None,
) -> list[str]:
    """Build the command that invokes the optional Scrapling pilot runner."""

    runtime = Path(runtime_python) if runtime_python is not None else default_scrapling_runtime_python()
    command = [
        str(runtime),
        str(SCRAPLING_EXTRACT_SCRIPT),
        "--url",
        url,
        "--selector",
        selector,
        "--selector-type",
        selector_type,
        "--mode",
        mode,
        "--timeout",
        str(timeout),
        "--max-chars",
        str(max_chars),
        "--fallback-reason",
        fallback_reason,
    ]
    if wait_selector:
        command.extend(["--wait-selector", wait_selector])
    if network_idle:
        command.append("--network-idle")
    return command


def run_scrapling_command(
    command: Sequence[str],
    *,
    adapter_timeout: int,
    url: str,
    selector: str,
    selector_type: str,
    mode: str,
    fallback_reason: str,
) -> dict:
    """Run a Scrapling command and normalize success/failure to a receipt."""

    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=adapter_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return _empty_receipt(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            fallback_reason=fallback_reason,
            error_type="ScraplingAdapterTimeout",
            message=f"Scrapling adapter timed out after {adapter_timeout}s",
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
    except OSError as exc:
        return _empty_receipt(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            fallback_reason=fallback_reason,
            error_type="ScraplingAdapterError",
            message=str(exc),
        )

    if completed.returncode != 0:
        message_parts = [f"Scrapling runner exited with {completed.returncode}"]
        if completed.stderr:
            message_parts.append(completed.stderr.strip())
        if completed.stdout:
            message_parts.append(completed.stdout.strip())
        return _empty_receipt(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            fallback_reason=fallback_reason,
            error_type="ScraplingAdapterError",
            message=_truncate("; ".join(message_parts)),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return _empty_receipt(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            fallback_reason=fallback_reason,
            error_type="InvalidScraplingReceipt",
            message=str(exc),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    if not isinstance(receipt, dict):
        return _empty_receipt(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            fallback_reason=fallback_reason,
            error_type="InvalidScraplingReceipt",
            message="Scrapling runner stdout JSON was not an object",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return receipt


def difficult_web_extract(
    url: str,
    *,
    selector: str,
    selector_type: str = "css",
    mode: str = "static",
    fallback_reason: str = "selector_required",
    timeout: int = 20,
    wait_selector: str | None = None,
    network_idle: bool = False,
    max_chars: int = 50_000,
    runtime_python: str | Path | None = None,
) -> dict:
    """Run the optional Scrapling difficult-extraction fallback.

    This is an internal adapter only. It does not search, does not browse, does
    not manage login/cookies, and does not register a public Hermes tool.
    """

    if not url.startswith(("http://", "https://")):
        return _empty_receipt(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            fallback_reason=fallback_reason,
            error_type="InvalidURL",
            message="Scrapling fallback only accepts http:// or https:// URLs",
        )
    if selector_type not in VALID_SELECTOR_TYPES:
        return _empty_receipt(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            fallback_reason=fallback_reason,
            error_type="InvalidSelectorType",
            message=f"Unsupported selector_type: {selector_type}",
        )
    if mode not in VALID_MODES:
        return _empty_receipt(
            url=url,
            selector=selector,
            selector_type=selector_type,
            mode=mode,
            fallback_reason=fallback_reason,
            error_type="InvalidMode",
            message=f"Unsupported mode: {mode}",
        )

    command = build_scrapling_command(
        url=url,
        selector=selector,
        selector_type=selector_type,
        mode=mode,
        fallback_reason=fallback_reason,
        timeout=timeout,
        wait_selector=wait_selector,
        network_idle=network_idle,
        max_chars=max_chars,
        runtime_python=runtime_python,
    )
    return run_scrapling_command(
        command,
        adapter_timeout=timeout + 10,
        url=url,
        selector=selector,
        selector_type=selector_type,
        mode=mode,
        fallback_reason=fallback_reason,
    )
