"""headroom_retrieve tool — retrieve compressed tool output originals.

When tool output compression is enabled (context.headroom.enabled), large
tool outputs are compressed before entering the conversation context.  The
original is stored in an in-memory cache keyed by a short hash.

This tool lets the agent retrieve the full original when needed.  The hash
is embedded in the compressed output marker:

    [Full output cached: hash=abc123def456]

Usage:
    headroom_retrieve(hash="abc123def456")

The tool is only available when compression is enabled.
"""

import json
from typing import Optional

from tools.registry import registry


def check_requirements() -> bool:
    """Only available when headroom compression is enabled."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        return bool(cfg.get("context", {}).get("headroom", {}).get("enabled", False))
    except Exception:
        return False


def headroom_retrieve(hash: str) -> str:
    """Retrieve a full tool output that was compressed by headroom."""
    try:
        from agent.tool_output_compressor import retrieve_cached
        original = retrieve_cached(hash)
        if original is None:
            return json.dumps({
                "found": False,
                "message": f"No cached content for hash {hash}. "
                           f"The cache may have expired (max 50 items, process-bound).",
            })

        return json.dumps({
            "found": True,
            "content": original,
            "size_bytes": len(original),
        })
    except Exception as e:
        return json.dumps({
            "found": False,
            "error": str(e),
        })


# Tool schema — only shown when compression is active
SCHEMA = {
    "name": "headroom_retrieve",
    "description": (
        "Retrieve a full tool output that was compressed by headroom. "
        "When large tool outputs are compressed, the original is cached "
        "with a short hash. Use this tool to fetch the uncompressed version.\n\n"
        "Example usage: see the compressed output marker for a hash like "
        "'hash=abc123def456', then call headroom_retrieve(hash='abc123def456').\n\n"
        "Cache is process-bound (cleared on Hermes restart) with a max of 50 items "
        "(FIFO eviction).  If the hash is no longer cached, it may have been evicted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hash": {
                "type": "string",
                "description": "The 12-character hex hash from the compressed output marker (e.g. 'abc123def456')"
            }
        },
        "required": ["hash"]
    }
}


registry.register(
    name="headroom_retrieve",
    toolset="headroom",
    schema=SCHEMA,
    handler=lambda args, **kw: headroom_retrieve(hash=args.get("hash", "")),
    check_fn=check_requirements,
    emoji="📦",
)
