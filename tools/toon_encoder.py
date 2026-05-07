"""
TOON Output Encoder — AXI Principle 1 & 3

Transforms tool JSON output to TOON (Token-Oriented Object Notation) format
for ~30-40% token savings on structured data. Falls back to JSON on any error.

Config: output_format in config.yaml — "json" (default) or "toon"

This module sits at the registry dispatch boundary. Individual tool handlers
are NOT modified — they continue returning json.dumps(). This module converts
at the exit point.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy import — toon_format may not be installed
_encode = None
_encode_loaded = False


def _get_encoder():
    """Lazy-load the TOON encoder. Returns None if not available."""
    global _encode, _encode_loaded
    if not _encode_loaded:
        _encode_loaded = True
        try:
            from toon_format import encode as _enc
            _encode = _enc
            logger.debug("TOON encoder loaded successfully")
        except ImportError:
            logger.warning("toon_format not installed — TOON output disabled")
            _encode = None
    return _encode


# ---------------------------------------------------------------------------
# AXI Principle 3: Content truncation
# ---------------------------------------------------------------------------

_DEFAULT_TRUNCATE_LIMIT = 1500  # chars


def _truncate_value(value: Any, limit: int = _DEFAULT_TRUNCATE_LIMIT) -> Any:
    """Truncate large string values in a data structure with size hints."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"\n... (truncated, {len(value)} chars total)"
    if isinstance(value, dict):
        return {k: _truncate_value(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, limit) for item in value]
    return value


# ---------------------------------------------------------------------------
# AXI Principle 5: Definitive empty states
# ---------------------------------------------------------------------------

def _ensure_definitive_empty(data: Any) -> Any:
    """Convert ambiguous empty outputs to explicit '0 results' states."""
    if isinstance(data, dict):
        # Empty list results with no count
        for key in ("results", "matches", "skills", "items", "sessions"):
            val = data.get(key)
            if isinstance(val, list) and len(val) == 0:
                if "count" not in data and "total_count" not in data:
                    data["count"] = 0
        return data
    return data


# ---------------------------------------------------------------------------
# AXI Principle 6: Structured errors
# ---------------------------------------------------------------------------

_STANDARD_ERROR_KEYS = {"error", "success", "message"}


def _is_error_result(data: Any) -> bool:
    """Check if the result is an error."""
    if isinstance(data, dict):
        if "error" in data:
            return True
        if data.get("success") is False:
            return True
    return False


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

# Tools that should always stay in JSON (e.g., complex nested structures
# that TOON handles poorly, or tools where consumers expect JSON)
_SKIP_TOON_TOOLS = frozenset({
    # Add tool names here if they break with TOON
})


def encode_toon(json_str: str, tool_name: str = "", truncate: bool = True) -> str:
    """
    Convert a JSON tool output string to TOON format.

    Args:
        json_str: The JSON string returned by a tool handler
        tool_name: Tool name (for skip-list and metadata)
        truncate: Whether to apply AXI content truncation

    Returns:
        TOON string, or original JSON string on any failure
    """
    # Skip list check
    if tool_name in _SKIP_TOON_TOOLS:
        return json_str

    # Get encoder
    encoder = _get_encoder()
    if encoder is None:
        return json_str

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON — return as-is (plain text tool output)
        return json_str

    # Apply AXI transformations
    data = _ensure_definitive_empty(data)

    if truncate:
        data = _truncate_value(data)

    # Encode to TOON
    try:
        toon_str = encoder(data)
        return toon_str
    except Exception as e:
        logger.debug("TOON encoding failed for %s: %s", tool_name, e)
        return json_str


def should_use_toon() -> bool:
    """Check if TOON output is enabled in config."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg.get("output_format") == "toon"
    except Exception:
        return False
