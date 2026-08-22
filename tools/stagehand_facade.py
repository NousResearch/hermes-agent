"""Stagehand V4 implementation of Hermes's single ``browser_exec`` tool.

The model-facing contract stays identical to Browser Use mode: one tool with
``code`` and ``timeout_s`` arguments, returning only ``success``,
``exit_code``, and ``output``. The implementation swaps Python/browser-harness
for a JavaScript Playwright-shaped facade executed by Stagehand V4's
``experimentalBatch`` on Browserbase.

This integration intentionally consumes the production facade from a built
Stagehand checkout instead of carrying a second copy in Hermes. Configure:

    browser:
      backend: stagehand
      stagehand_root: /absolute/path/to/stagehand

The checkout must contain built ``packages/sdk-ts/dist`` and
``packages/integrations/core/dist/facade`` artifacts.
"""

from __future__ import annotations

import shutil
from typing import Any, Mapping

from tools.registry import tool_result
from tools.stagehand_facade_client import call_stagehand_facade

_DEFAULT_TIMEOUT_S = 60
_MIN_TIMEOUT_S = 5
_MAX_TIMEOUT_S = 60


_STAGEHAND_DESCRIPTION = (
    "Drive one persistent Browserbase browser by writing JavaScript against "
    "Stagehand V4's Playwright-shaped facade. The `code` argument is the BODY "
    "of an async workflow, not Python and not an `async () => {}` wrapper. "
    "The variables `page`, `context`, and `browser` are prebound. Use familiar "
    "Playwright operations such as `await page.goto(url)`, "
    "`page.locator(selector)`, `page.getByRole(role, options)`, "
    "`await locator.click()`, and `await page.evaluate(...)`. Do not import "
    "packages, launch another browser, or close the provided browser.\n\n"
    "STATE: pages, cookies, and navigation persist across calls; JavaScript "
    "variables do not. Inspect an unfamiliar page before acting: return a "
    "compact result from `await page.accessibility.snapshot()`, `await "
    "page.content()`, or a targeted `page.evaluate(...)`, then use what you "
    "observed in the next call. For deterministic tasks, navigation, "
    "inspection, interaction, extraction, filtering, and aggregation may be "
    "combined in one workflow. Return only compact JSON-serializable data "
    "needed for the answer.\n\n"
    "The facade translates this Playwright-shaped workflow to Stagehand V4 "
    "experimental batch execution. Login walls: stop and ask the user; never "
    "guess credentials."
)


def _browser_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        value = cfg_get(read_raw_config(), "browser", default={})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _stagehand_root() -> str:
    return str(_browser_config().get("stagehand_root") or "").strip()


def _node_executable() -> str:
    configured = str(
        _browser_config().get("stagehand_node_executable") or ""
    ).strip()
    if configured:
        return configured
    return shutil.which("node") or "node"


def _timeout(value: Any) -> int:
    try:
        return max(_MIN_TIMEOUT_S, min(int(value), _MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


def stagehand_browser_exec(
    *,
    code: str,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    task_id: str | None = None,
) -> str:
    """Execute one JavaScript workflow and return Browser Use's envelope."""
    root = _stagehand_root()
    if not root:
        return tool_result(
            {
                "success": False,
                "exit_code": 1,
                "output": "",
                "stderr": (
                    "Stagehand browser_exec is selected but browser.stagehand_root "
                    "is not configured. Point it at a built Stagehand V4 checkout."
                ),
            }
        )

    try:
        response = call_stagehand_facade(
            code=code,
            timeout_s=_timeout(timeout_s),
            node_executable=_node_executable(),
            stagehand_root=root,
            task_key=str(task_id or "default"),
        )
    except Exception as error:
        return tool_result(
            {
                "success": False,
                "exit_code": 1,
                "output": "",
                "stderr": f"{type(error).__name__}: {str(error)[:1000]}",
            }
        )

    if response.get("success") is not True:
        return tool_result(
            {
                "success": False,
                "exit_code": 1,
                "output": "",
                "stderr": str(
                    response.get("error") or "Stagehand facade execution failed"
                )[:1000],
            }
        )

    return tool_result(
        {
            "success": True,
            "exit_code": 0,
            "output": str(response.get("output") or ""),
        }
    )


def stagehand_schema_overrides(
    browser_exec_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the Stagehand-specific schema while preserving one tool name."""
    parameters = dict(browser_exec_schema["parameters"])
    properties = dict(parameters["properties"])
    properties["code"] = {
        "type": "string",
        "description": (
            "JavaScript async-workflow body using the prebound Playwright-shaped "
            "page, context, and browser variables. Await operations and return "
            "compact JSON-serializable data."
        ),
    }
    properties.pop("session", None)
    properties["timeout_s"] = {
        "type": "integer",
        "description": (
            f"Max seconds to wait for the workflow (default {_DEFAULT_TIMEOUT_S}, "
            f"max {_MAX_TIMEOUT_S})."
        ),
        "default": _DEFAULT_TIMEOUT_S,
    }
    parameters["properties"] = properties
    return {"description": _STAGEHAND_DESCRIPTION, "parameters": parameters}
