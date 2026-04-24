#!/usr/bin/env python3
"""
Compression Pipeline — token-saving layer for terminal command output.

This module intercepts terminal_tool JSON results and routes them through
the compressor chain before they are returned to the agent loop.

Architecture (every layer has a guaranteed fallback):
    terminal_tool() returns JSON string
        |
        v
    compress_tool_output(command, raw_json, exit_code)  <- entry point
        |
        +-- [RTK fast path] --> if RTK binary found on PATH
        |       fallback: skip, try native
        |
        +-- [Native compressor] --> if command in CompressorRegistry
        |       fallback: skip, try LLM summarization
        |
        +-- [LLM summarization] --> if output > COMPRESSION_THRESHOLD_TOKENS
        |       fallback: skip, return raw
        |
        v
    raw output (always safe — never crashes the agent)

Enable with HERMES_COMPRESS=1 (default: off).
Stats are tracked regardless (atomically incremented).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Optional

from tools.command_compressors import DEFAULT_COMPRESSORS, CompressorRegistry
from tools.tty_detector import should_skip_compression

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Feature flag — off by default to preserve existing behaviour
COMPRESSION_ENABLED = os.getenv("HERMES_COMPRESS", "0") == "1"

# Token threshold — only LLM-summarise if raw output exceeds this
COMPRESSION_THRESHOLD_TOKENS = int(os.getenv("HERMES_COMPRESSION_THRESHOLD_TOKENS", "500"))

# Try RTK binary as fast path if found on PATH
RTK_BINARY = os.getenv("HERMES_RTK_BINARY", "rtk")

# LLM summarization timeout (seconds)
LLM_SUMMARIZE_TIMEOUT = int(os.getenv("HERMES_COMPRESS_LLM_TIMEOUT", "5"))

# -------------------------------------------------------------------
# Token estimation
# -------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate. Used to decide whether to compress."""
    # Rough: 4 chars per token on average
    return max(1, len(text) // 4)


# -------------------------------------------------------------------
# RTK fast path (optional, no-op if RTK not installed)
# -------------------------------------------------------------------


def _try_rtk_compress(command: str, stdout: str, stderr: str, exit_code: int) -> Optional[str]:
    """
    Try to run `rtk <subcommand> --ultra-compact` as a subprocess.
    Returns compressed output string or None if RTK is not installed / fails.
    This is a pure no-op — no external network calls, no side effects.
    """
    try:
        # Determine RTK subcommand from the command
        # e.g. "git status" -> "rtk git status --ultra-compact"
        rtk_cmd = f"{RTK_BINARY} {command}"

        # Run with a short timeout — if RTK hangs, fall back immediately
        proc = subprocess.run(
            rtk_cmd.split(),
            input=stdout,
            capture_output=True,
            text=True,
            timeout=2,  # 2s max — if RTK doesn't respond quickly, skip it
        )
        if proc.returncode == 0 and proc.stdout.strip():
            compressed = proc.stdout.strip()
            if compressed and len(compressed) < len(stdout):
                return compressed
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


# -------------------------------------------------------------------
# LLM summarization (last-resort fallback)
# -------------------------------------------------------------------

# Lazy import to avoid circular dependency
_llm_client = None


def _get_llm_client():
    """Lazily get the auxiliary LLM client for summarization."""
    global _llm_client
    if _llm_client is None:
        try:
            from agent.auxiliary_client import call_llm
            _llm_client = call_llm
        except Exception as e:
            logger.debug("Could not import auxiliary_client: %s", e)
            _llm_client = False
    return _llm_client


def _llm_summarize(command: str, stdout: str, stderr: str, exit_code: int) -> Optional[str]:
    """
    Use the auxiliary LLM (cheap/fast model) to summarize command output.
    Only called when:
      1. No native compressor matched
      2. Output exceeds COMPRESSION_THRESHOLD_TOKENS
    Returns None on failure — caller falls back to raw output.
    """
    call_llm = _get_llm_client()
    if not call_llm:
        return None

    # Build a focused summarization prompt
    summary_prompt = (
        f"Compress this CLI command output to its essential signal.\n"
        f"Command: {command}\n"
        f"Exit code: {exit_code}\n"
        f"Output ({len(stdout)} chars):\n"
        f"---\n"
        f"{stdout[:3000]}\n"  # Cap at 3000 chars to keep summarizer fast
        f"---\n"
        f"Summarize in 1-3 lines. Include exit code if non-zero. "
        f"Format: '[status] summary text'."
    )

    try:
        response = call_llm(
            prompt=summary_prompt,
            model="fast",  # Use the fastest available auxiliary model
            max_tokens=100,
            timeout=LLM_SUMMARIZE_TIMEOUT,
        )
        if response and response.strip():
            result = response.strip()
            # Validate it's not wildly longer than the input
            if len(result) < len(stdout):
                return result
    except Exception as e:
        logger.debug("LLM summarization failed: %s", e)

    return None


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------


def compress_tool_output(
    command: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    *,
    registry: Optional[CompressorRegistry] = None,
) -> str:
    """
    Compress raw terminal command output.
    
    This is the single public entry point — called by the tool result hook
    in model_tools.py after terminal_tool returns.

    Args:
        command: The full command that was run (e.g. 'cargo test --lib')
        stdout: Raw stdout from the command
        stderr: Raw stderr from the command
        exit_code: Process exit code
        registry: CompressorRegistry to use (default: DEFAULT_COMPRESSORS)
        
    Returns:
        Compressed output string — never raises, always returns something usable.
        If all compression layers fail, returns original stdout.

    Compression layers (in order):
        1. RTK binary (if HERMES_COMPRESS=1 and RTK installed)
        2. Native Python compressor (if command matches)
        3. LLM summarization (if output > COMPRESSION_THRESHOLD_TOKENS)
        4. Raw output (guaranteed fallback)
    """
    # --- Fast path: compression disabled ---
    if not COMPRESSION_ENABLED:
        return stdout

    # --- TTY guard: skip compression for interactive programs ---
    should_skip, _ = should_skip_compression(command, stdout, stderr)
    if should_skip:
        logger.debug("Skipping compression for interactive command: %s", command)
        return stdout

    # --- Fast path: empty or tiny output ---
    raw_tokens = _estimate_tokens(stdout)
    if raw_tokens < 20:
        return stdout

    registry = registry or DEFAULT_COMPRESSORS

    # --- Layer 1: RTK fast path ---
    try:
        rtk_result = _try_rtk_compress(command, stdout, stderr, exit_code)
        if rtk_result is not None:
            _record_savings(command, raw_tokens, _estimate_tokens(rtk_result))
            return rtk_result
    except Exception as e:
        logger.debug("RTK compression layer failed (non-fatal): %s", e)

    # --- Layer 2: Native Python compressor ---
    try:
        native_result = registry.compress(command, stdout, stderr, exit_code)
        if native_result is not None:
            _record_savings(command, raw_tokens, _estimate_tokens(native_result))
            return native_result
    except Exception as e:
        logger.debug("Native compression failed (non-fatal): %s", e)

    # --- Layer 3: LLM summarization (only for large outputs) ---
    if raw_tokens >= COMPRESSION_THRESHOLD_TOKENS:
        try:
            llm_result = _llm_summarize(command, stdout, stderr, exit_code)
            if llm_result is not None:
                _record_savings(command, raw_tokens, _estimate_tokens(llm_result))
                return llm_result
        except Exception as e:
            logger.debug("LLM summarization failed (non-fatal): %s", e)

    # --- Layer 4: Guaranteed fallback ---
    return stdout


# -------------------------------------------------------------------
# Stats tracking (shared with hermes stats command)
# -------------------------------------------------------------------

_stats: dict[str, dict[str, int]] = {}
_stats_lock = __import__("threading").Lock()


def _record_savings(command: str, raw_tokens: int, compressed_tokens: int):
    """Thread-safe recording of compression savings."""
    import shlex

    cmd_key = shlex.split(command.strip())[0] if command.strip() else "unknown"
    with _stats_lock:
        if cmd_key not in _stats:
            _stats[cmd_key] = {"count": 0, "raw_tokens": 0, "compressed_tokens": 0}
        s = _stats[cmd_key]
        s["count"] += 1
        s["raw_tokens"] += raw_tokens
        s["compressed_tokens"] += compressed_tokens


def get_compression_stats() -> dict:
    """Return a copy of the stats dict for hermes stats command."""
    with _stats_lock:
        return dict(_stats)


def reset_compression_stats():
    """Clear stats (e.g., on session reset)."""
    with _stats_lock:
        _stats.clear()
