#!/usr/bin/env python3
"""
Caveman Compression Tool for Hermes Agent

Compresses LLM context text using the rust-cave-001 native library.
Reduces token count by applying semantic compression rules:
- Remove articles, intensifiers, connectives
- Normalize active voice
- Enforce 2-5 words per sentence
- Estimate token counts and compression ratio

Usage: from tools.caveman_compression import compress, caveman_compress, estimate_tokens
"""

import json
import importlib.util
from typing import Optional

# Lazy-load the rust library at first call (not at import time for auto-discovery)
_rust_cave_001 = None


def _get_rust_module():
    """Lazily import rust_cave_001, retrying via sys.path injection on failure."""
    global _rust_cave_001
    if _rust_cave_001 is None:
        # Try direct import first (works when installed via pip in Hermes venv)
        try:
            _rust_cave_001 = importlib.import_module("rust_cave_001")
        except ModuleNotFoundError:
            # Fall back to project target/release directory
            import sys, os
            wheel_dir = "/srv/sync/projects/rust-cave-001/target/release"
            if wheel_dir not in sys.path:
                sys.path.insert(0, wheel_dir)
            os.environ["LD_LIBRARY_PATH"] = wheel_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            _rust_cave_001 = importlib.import_module("rust_cave_001")
    return _rust_cave_001


def estimate_tokens(text: str) -> str:
    """Estimate token count for text."""
    try:
        mod = _get_rust_module()
        count = mod.estimate_tokens(text)
        return json.dumps({"tokens": count, "text": text[:100]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def compress(text: str) -> str:
    """Compress text using caveman compression rules.

    Args:
        text: The text to compress.
    """
    try:
        mod = _get_rust_module()
        compressed = mod.compress(text)
        original_tokens = mod.estimate_tokens(text)
        compressed_tokens = mod.estimate_tokens(compressed)
        saved = original_tokens - compressed_tokens
        ratio = round(compressed_tokens / original_tokens, 2) if original_tokens > 0 else 1.0
        return json.dumps(
            {
                "original_text": text,
                "compressed_text": compressed,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "tokens_saved": saved,
                "compression_ratio": ratio,
            },
            ensure_ascii=False,
        )
    except ValueError as e:
        # Logical completeness or other validation error
        return json.dumps({"error": str(e), "original_text": text}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def caveman_compress(text: str) -> str:
    """Alias for compress() — semantic text compression for LLM context reduction.

    Removes articles (the/a/an), intensifiers (very/extremely), connectives
    (because/however/therefore/but), converts passive to active voice,
    and enforces 2-5 words per sentence.
    """
    return compress(text)


def preprocess_text(text: str) -> str:
    """Preprocess text: active voice transformation only.

    Unlike compress(), this does not remove articles or intensifiers,
    only converts passive voice to active voice.
    """
    try:
        mod = _get_rust_module()
        result = mod.preprocess_text(text)
        return json.dumps(
            {"original_text": text, "processed_text": result}, ensure_ascii=False
        )
    except ValueError as e:
        return json.dumps({"error": str(e), "original_text": text}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# --- Tool Schema ---

_COMPRESS_SCHEMA = {
    "name": "compress",
    "description": (
        "Compresses text using caveman compression rules to reduce LLM token usage. "
        "Removes articles (a/an/the/this), intensifiers (very/extremely/quite), "
        "connectives (because/however/therefore/but), keeps active voice, and enforces 2-5 words per sentence. "
        "Returns compressed text, token counts, and compression ratio."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to compress. Should be complete sentences.",
            }
        },
        "required": ["text"],
    },
}

_PREPROCESS_SCHEMA = {
    "name": "preprocess_text",
    "description": (
        "Preprocess text to convert passive voice to active voice. "
        "E.g., 'The ball was thrown by John' becomes 'John threw the ball'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to preprocess.",
            }
        },
        "required": ["text"],
    },
}

_ESTIMATE_SCHEMA = {
    "name": "estimate_tokens",
    "description": "Estimate token count for a given text string.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to estimate tokens for.",
            }
        },
        "required": ["text"],
    },
}


# --- Check function ---

def _check_caveman_compression():
    """Check if rust_cave_001 library is available."""
    try:
        from rust_cave_001 import compress
        return True
    except ModuleNotFoundError:
        pass
    except ImportError:
        pass

    # Try the project wheel path
    import sys, os
    wheel_dir = "/srv/sync/projects/rust-cave-001/target/release"
    if wheel_dir not in sys.path:
        sys.path.insert(0, wheel_dir)
    os.environ["LD_LIBRARY_PATH"] = wheel_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    try:
        from rust_cave_001 import compress
        return True
    except (ModuleNotFoundError, ImportError):
        return False


# --- Registry ---
from tools.registry import registry

registry.register(
    name="compress",
    toolset="caveman",
    schema=_COMPRESS_SCHEMA,
    handler=lambda args, **kw: compress(text=args.get("text", "")),
    check_fn=_check_caveman_compression,
    emoji="🗜️",
)

registry.register(
    name="preprocess_text",
    toolset="caveman",
    schema=_PREPROCESS_SCHEMA,
    handler=lambda args, **kw: preprocess_text(text=args.get("text", "")),
    check_fn=_check_caveman_compression,
    emoji="✏️",
)

registry.register(
    name="estimate_tokens",
    toolset="caveman",
    schema=_ESTIMATE_SCHEMA,
    handler=lambda args, **kw: estimate_tokens(text=args.get("text", "")),
    check_fn=_check_caveman_compression,
    emoji="🔢",
)
