"""Worker-scope memory cap derivation for tools/process_registry.py.

Kept as a topical sibling so the over-2k-line facade files do not grow (see
scripts/code_health config and AGENTS.md): the operator-facing cap resolution
lives here, and process_registry imports the one function it needs.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def operator_worker_memory_max_bytes(
    min_bytes: int = 64 * 1024 * 1024,
) -> Optional[int]:
    """Per-worker ``MemoryMax`` (bytes) explicitly requested by the operator.

    ``TERMINAL_LOCAL_MEMORY_MAX_MB`` (env) wins over ``terminal.local_memory_max_mb``
    (config.yaml). ``0``/absent means "stay on the conservative default". Values
    below *min_bytes* are rejected with a warning. Returns ``None`` when the
    operator did not request a limit (unreadable config included — the caller's
    safe bound then stands).
    """
    source: str
    raw = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if raw:
        source = "TERMINAL_LOCAL_MEMORY_MAX_MB"
    else:
        try:
            from hermes_cli.config import DEFAULT_CONFIG, cfg_get, read_raw_config

            val = cfg_get(read_raw_config(), "terminal", "local_memory_max_mb")
            if val is None:
                val = DEFAULT_CONFIG["terminal"]["local_memory_max_mb"]
            raw = "" if val is None else str(val).strip()
            source = "terminal.local_memory_max_mb"
        except (ImportError, KeyError, TypeError, ValueError):
            # Missing/blank key is identical to "not configured"; the caller's
            # safe bound then stands. read_raw_config itself never raises
            # (returns {} on unreadable), so only the import/schema lookups can.
            return None
    if not raw or raw == "0":
        return None
    try:
        parsed = int(raw) * 1024 * 1024
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; expected an integer number of MiB", source, raw)
        return None
    if parsed < min_bytes:
        logger.warning(
            "Ignoring %s=%r; expected at least %d MiB",
            source, raw, min_bytes // (1024 * 1024))
        return None
    return parsed
