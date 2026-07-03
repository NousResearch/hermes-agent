#!/usr/bin/env python3
"""Upload local files into a browser ``<input type=file>`` via Playwright/CDP.

CDP alone cannot set a local file chooser payload reliably because browsers
intentionally protect file inputs.  Playwright exposes the supported
``setInputFiles`` path while still connecting to the same user-authenticated
Chrome instance via CDP.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DEFAULT_SELECTOR = 'input[type="file"]'
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_cdp_endpoint() -> str:
    """Return a CDP endpoint suitable for Playwright ``connectOverCDP``."""
    raw_env = (os.environ.get("BROWSER_CDP_URL") or "").strip()
    if raw_env:
        return raw_env
    try:
        from tools.browser_tool import _get_cdp_override  # type: ignore[import-not-found]

        return (_get_cdp_override() or "").strip()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("browser_upload_files: failed to resolve CDP endpoint: %s", exc)
        return ""


def _node_command() -> Optional[str]:
    return shutil.which("node")


def _normalise_files(files: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in files:
        path = Path(str(item)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Upload file does not exist or is not a file: {path}")
        out.append(str(path.resolve()))
    return out


def _playwright_node_script() -> str:
    return r"""
const payload = JSON.parse(process.argv[1]);

async function main() {
  let playwright;
  try {
    playwright = require("playwright-core");
  } catch (error) {
    throw new Error(
      "Node package 'playwright-core' is not available. Install/sync it locally. " +
      error.message
    );
  }

  const browser = await playwright.chromium.connectOverCDP(payload.cdpEndpoint);
  try {
    const contexts = browser.contexts();
    const pages = contexts.flatMap((context) => context.pages());
    let page = null;
    for (const candidate of pages) {
      const url = candidate.url();
      const title = await candidate.title().catch(() => "");
      if (
        (!payload.targetUrlContains || url.includes(payload.targetUrlContains)) &&
        (!payload.targetTitleContains || title.includes(payload.targetTitleContains))
      ) {
        page = candidate;
        break;
      }
    }
    if (!page) {
      throw new Error("No matching page found for requested target filters");
    }

    const locator = page.locator(payload.selector).first();
    const count = await page.locator(payload.selector).count();
    if (count < 1) {
      throw new Error(`No file input matched selector: ${payload.selector}`);
    }

    await locator.setInputFiles(payload.files);
    await page.waitForTimeout(payload.settleMs);

    const state = await page.evaluate((selector) => {
      const norm = (value) => (value || "").replace(/\s+/g, " ").trim();
      const fileInputs = Array.from(document.querySelectorAll("input[type=file]")).map(
        (el, index) => ({
          index,
          selectorMatched: el.matches(selector),
          accept: el.getAttribute("accept") || "",
          multiple: !!el.multiple,
          disabled: !!el.disabled,
          files: el.files ? el.files.length : null,
        })
      );
      const buttons = Array.from(document.querySelectorAll('div[role="button"], button, a[role="button"]'))
        .map((el) => ({
          text: norm(el.innerText || el.getAttribute("aria-label")),
          aria: norm(el.getAttribute("aria-label")),
          ariaDisabled: el.getAttribute("aria-disabled") || "",
          disabled: !!el.disabled,
          visible: (() => {
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          })(),
        }))
        .filter((item) => /^(Next|Post|Publish|Submit|Done|下一步|刊登|發布|送出)$/.test(item.text || item.aria));
      return {
        href: location.href,
        title: document.title,
        fileInputs,
        actionButtons: buttons,
        textTail: norm(document.body ? document.body.innerText : "").slice(-1200),
      };
    }, payload.selector);

    return {
      success: true,
      uploadedFiles: payload.files.length,
      selector: payload.selector,
      targetUrl: page.url(),
      targetTitle: await page.title().catch(() => ""),
      state,
    };
  } finally {
    await browser.close().catch(() => {});
  }
}

main()
  .then((result) => process.stdout.write(JSON.stringify(result)))
  .catch((error) => {
    process.stdout.write(JSON.stringify({ success: false, error: String(error && error.message || error) }));
    process.exitCode = 1;
  });
"""


def _run_playwright_upload(payload: Dict[str, Any], timeout: float) -> subprocess.CompletedProcess[str]:
    node = _node_command()
    if not node:
        raise RuntimeError("Node.js is required for browser_upload_files but was not found on PATH")

    script = _playwright_node_script()
    command = [node, "-e", script, json.dumps(payload, ensure_ascii=False)]
    env = os.environ.copy()

    # Prefer explicit local dependencies.  The old npx fallback is deliberately
    # not used because npm/npx do not expose package imports reliably to
    # ``node -e`` across npm versions.
    configured_node_paths = [
        p.strip()
        for p in (env.get("HERMES_BROWSER_UPLOAD_NODE_PATHS") or "").split(os.pathsep)
        if p.strip()
    ]
    node_paths = configured_node_paths or [str(PROJECT_ROOT / "node_modules")]
    if node_paths:
        existing = env.get("NODE_PATH", "")
        env["NODE_PATH"] = os.pathsep.join(node_paths + ([existing] if existing else []))

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout),
        env=env,
    )


def browser_upload_files(
    files: List[str],
    selector: str = DEFAULT_SELECTOR,
    target_url_contains: Optional[str] = None,
    target_title_contains: Optional[str] = None,
    timeout: float = 60.0,
    settle_ms: int = 5000,
) -> str:
    """Set local files on a file input in a CDP-connected browser page.

    This tool does not click buttons or submit forms.  It only populates a
    file input and reports page state after the browser processes the upload.
    """
    if not isinstance(files, list) or not files:
        return tool_error("'files' must be a non-empty list of local file paths")
    if not isinstance(selector, str) or not selector.strip():
        return tool_error("'selector' must be a non-empty CSS selector")
    try:
        upload_files = _normalise_files(files)
    except FileNotFoundError as exc:
        return tool_error(str(exc))

    endpoint = _resolve_cdp_endpoint()
    if not endpoint:
        return tool_error(
            "No CDP endpoint is available. Set BROWSER_CDP_URL or run /browser connect first."
        )

    try:
        safe_timeout = float(timeout) if timeout else 60.0
    except (TypeError, ValueError):
        safe_timeout = 60.0
    safe_timeout = max(1.0, min(safe_timeout, 300.0))

    payload = {
        "cdpEndpoint": endpoint,
        "files": upload_files,
        "selector": selector.strip(),
        "targetUrlContains": target_url_contains or "",
        "targetTitleContains": target_title_contains or "",
        "settleMs": max(0, min(int(settle_ms or 0), 30000)),
    }

    try:
        proc = _run_playwright_upload(payload, safe_timeout)
    except subprocess.TimeoutExpired:
        return tool_error(f"browser_upload_files timed out after {safe_timeout}s")
    except Exception as exc:
        return tool_error(f"browser_upload_files failed before upload: {exc}")

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    try:
        data = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        data = {"success": False, "error": "Upload helper returned non-JSON output", "stdout": stdout}

    if proc.returncode != 0 or not data.get("success"):
        return tool_error(
            data.get("error") or stderr or f"Upload helper exited with rc={proc.returncode}",
            stderr=stderr[-1000:] if stderr else "",
        )

    if stderr:
        data["stderr_tail"] = stderr[-1000:]
    return json.dumps(data, ensure_ascii=False)


BROWSER_UPLOAD_FILES_SCHEMA: Dict[str, Any] = {
    "name": "browser_upload_files",
    "description": (
        "Upload one or more local files into a browser file input using "
        "Playwright connected to the current CDP browser session. This only "
        "sets files on an input[type=file] and reports resulting page state; "
        "it does not click Next/Post/Publish or submit any form. Use this when "
        "a web form requires photos or attachments and browser_cdp cannot set "
        "the local file chooser payload."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Absolute or user-relative local file paths to upload.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for the file input. Defaults to input[type=\"file\"].",
                "default": DEFAULT_SELECTOR,
            },
            "target_url_contains": {
                "type": "string",
                "description": "Optional substring used to select the browser tab by URL.",
            },
            "target_title_contains": {
                "type": "string",
                "description": "Optional substring used to select the browser tab by title.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 60, max 300).",
                "default": 60,
            },
            "settle_ms": {
                "type": "integer",
                "description": "Milliseconds to wait after setInputFiles before reading page state.",
                "default": 5000,
            },
        },
        "required": ["files"],
    },
}


def _browser_upload_files_check() -> bool:
    return bool(_resolve_cdp_endpoint() and _node_command())


registry.register(
    name="browser_upload_files",
    toolset="browser-cdp",
    schema=BROWSER_UPLOAD_FILES_SCHEMA,
    handler=lambda args, **kw: browser_upload_files(
        files=args.get("files", []),
        selector=args.get("selector", DEFAULT_SELECTOR),
        target_url_contains=args.get("target_url_contains"),
        target_title_contains=args.get("target_title_contains"),
        timeout=args.get("timeout", 60.0),
        settle_ms=args.get("settle_ms", 5000),
    ),
    check_fn=_browser_upload_files_check,
    emoji="📎",
)
