"""LLM Guard integration for tool-result scanning.

Wraps protectai/llm-guard's output scanners to detect prompt injection in tool
results before they're fed back into the model context.

Configuration (config.yaml under ``security.llm_guard``):

  llm_guard_enabled: false        # master switch (env: HERMES_LLM_GUARD)
  llm_guard_fail_open: true       # true = pass through on scan error / library missing
                                  # false = block on error (env: HERMES_LLM_GUARD_FAIL_OPEN)
  llm_guard_block_action: "replace"  # what to do when injection detected:
                                  #   "replace" — substitute a blocked-content notice (default)
                                  #   "raise"   — raise LLMGuardInjectionError, stopping the job
                                  # (env: HERMES_LLM_GUARD_BLOCK_ACTION)

Scan pipeline (applied in order):
  1. PromptInjectionV2 — transformer-based injection classifier (threshold 0.85)
  2. BanSubstrings      — literal-match blocklist from threat_patterns.py

Install: pip install hermes-agent[llm-guard]
"""

from __future__ import annotations

import json as _json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMGuardInjectionError(RuntimeError):
    """Raised when llm_guard_block_action is 'raise' and injection is detected.

    Callers (tool executor, gateway) should catch this to stop the current job
    and surface the warning to the operator.
    """
    def __init__(self, tool_name: str, failed_scanners: list[str], scores: dict):
        self.tool_name = tool_name
        self.failed_scanners = failed_scanners
        self.scores = scores
        super().__init__(
            f"LLM Guard blocked tool result from '{tool_name}': "
            f"injection detected by {failed_scanners} (scores: "
            f"{{{', '.join(f'{k}: {round(v, 3)}' for k, v in scores.items())}}})"
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes"}


def _load_llm_guard_config() -> dict:
    """Load llm_guard settings from config.yaml, with env var overrides.

    Returns a dict with keys: enabled, fail_open, block_action.
    """
    defaults = {
        "enabled": False,
        "fail_open": True,
        "block_action": "replace",
    }
    try:
        from hermes_cli.config import load_config
        sec = load_config().get("security", {}) or {}
        cfg = sec.get("llm_guard", {}) or {}
    except Exception:
        cfg = {}

    enabled = _env_bool(
        "HERMES_LLM_GUARD",
        bool(cfg.get("enabled", defaults["enabled"])),
    )
    fail_open = _env_bool(
        "HERMES_LLM_GUARD_FAIL_OPEN",
        bool(cfg.get("fail_open", defaults["fail_open"])),
    )

    env_action = os.getenv("HERMES_LLM_GUARD_BLOCK_ACTION")
    block_action = env_action if env_action else cfg.get("block_action", defaults["block_action"])
    if block_action not in ("replace", "raise"):
        logger.warning(
            "llm_guard.block_action %r is not valid (expected 'replace' or 'raise'); "
            "falling back to 'replace'",
            block_action,
        )
        block_action = "replace"

    return {"enabled": enabled, "fail_open": fail_open, "block_action": block_action}


# ---------------------------------------------------------------------------
# Scanner cache — built once per type, reused across calls
#
# PromptInjection lives in llm_guard.input_scanners and is called via
# scan_prompt(). BanSubstrings lives in llm_guard.output_scanners and is
# called via scan_output(). They must be run separately.
# ---------------------------------------------------------------------------

_input_scanners: list | None = None
_output_scanners: list | None = None
_scanners_init_failed: bool = False

# Literal strings drawn from threat_patterns.py "all"-scope patterns
# that are unambiguously malicious and short enough for exact matching.
_BAN_SUBSTRINGS = [
    "ignore previous instructions",
    "ignore all instructions",
    "ignore prior instructions",
    "ignore above instructions",
    "system prompt override",
    "disregard your instructions",
    "disregard all instructions",
    "do not tell the user",
]


def _get_scanners() -> tuple[list, list]:
    """Build (and cache) the input and output scanner pipelines.

    Returns (input_scanners, output_scanners).  Both lists are empty on
    failure so callers can check ``if not input_scanners and not output_scanners``.
    """
    global _input_scanners, _output_scanners, _scanners_init_failed

    if _input_scanners is not None and _output_scanners is not None:
        return _input_scanners, _output_scanners
    if _scanners_init_failed:
        return [], []

    try:
        from llm_guard.input_scanners import PromptInjection
        from llm_guard.input_scanners.prompt_injection import V2_MODEL
        from llm_guard.output_scanners import BanSubstrings

        _input_scanners = [PromptInjection(model=V2_MODEL, threshold=0.85)]
        _output_scanners = [BanSubstrings(substrings=_BAN_SUBSTRINGS, match_type="str", case_sensitive=False)]
        logger.debug("llm-guard scanners initialised (%d input, %d output)",
                     len(_input_scanners), len(_output_scanners))
        return _input_scanners, _output_scanners

    except ImportError:
        logger.debug("llm-guard not installed; tool-result scanning disabled")
        _scanners_init_failed = True
        return [], []
    except Exception as exc:
        logger.warning("llm-guard scanner init failed: %s", exc)
        _scanners_init_failed = True
        return [], []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _serialize_for_scan(content: Any) -> str | None:
    """Convert structured tool results to a string for scanning.

    Dicts and lists are serialized to JSON so tools like ``read_file``,
    ``terminal``, ``search_files``, and ``web_search`` are covered.
    Returns None for types that cannot carry injection payloads
    (None, bool, int, float).
    """
    if isinstance(content, (dict, list)):
        try:
            return _json.dumps(content, default=str)
        except Exception:
            return str(content)
    if isinstance(content, str):
        return content
    # None, bool, int, float — not injection vectors.
    return None


def scan_tool_result(tool_name: str, content: Any) -> Any:
    """Scan a tool result for prompt injection.

    Structured content (dicts / lists) is serialized to JSON before
    scanning so tools like ``read_file``, ``terminal``, ``search_files``,
    and ``web_search`` are covered.

    Behaviour depends on config:

    - Disabled → content returned unchanged.
    - Content that can't carry injection (None, bool, int, float) → unchanged.
    - llm-guard not installed or scanner raises:
        fail_open=true  → content returned unchanged (logged at WARNING)
        fail_open=false → raises LLMGuardInjectionError
    - Injection detected:
        block_action='replace' → returns a [BLOCKED …] notice string
        block_action='raise'   → raises LLMGuardInjectionError
    """
    cfg = _load_llm_guard_config()

    if not cfg["enabled"]:
        return content

    # Serialize structured content to a scan-able string.
    scan_text = _serialize_for_scan(content)
    if scan_text is None:
        return content

    input_scanners, output_scanners = _get_scanners()
    if not input_scanners and not output_scanners:
        if not cfg["fail_open"]:
            raise LLMGuardInjectionError(
                tool_name, ["(scanner unavailable)"], {}
            )
        return content

    try:
        from llm_guard import scan_prompt, scan_output

        all_results: dict = {}
        all_scores: dict = {}
        current = scan_text

        # PromptInjection (input scanner) — classify the text for injection
        if input_scanners:
            current, results, scores = scan_prompt(input_scanners, current)
            all_results.update(results)
            all_scores.update(scores)

        # BanSubstrings (output scanner) — literal-string blocklist
        if output_scanners:
            # scan_output needs a prompt argument; tool name is used as proxy
            current, results, scores = scan_output(output_scanners, tool_name, current)
            all_results.update(results)
            all_scores.update(scores)

        failed = [name for name, passed in all_results.items() if not passed]
        if not failed:
            return content

        logger.warning(
            "llm-guard flagged tool result from %r: failed scanners %s (scores: %s)",
            tool_name, failed, {k: round(v, 3) for k, v in all_scores.items()}
        )

        if cfg["block_action"] == "raise":
            raise LLMGuardInjectionError(tool_name, failed, all_scores)

        # block_action == "replace"
        return (
            f"[BLOCKED by llm-guard: tool result from '{tool_name}' was flagged "
            f"for potential prompt injection ({', '.join(failed)}). "
            f"Content not forwarded to model.]"
        )

    except LLMGuardInjectionError:
        raise
    except Exception as exc:
        logger.warning("llm-guard scan raised for tool %r: %s", tool_name, exc)
        if not cfg["fail_open"]:
            raise LLMGuardInjectionError(tool_name, [f"(scan error: {exc})"], {})
        return content