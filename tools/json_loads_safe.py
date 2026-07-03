"""BOM-tolerant JSON parser, drop-in replacement for ``json.loads``.

Hermes tools that ingest data from external sources — subprocess
output, web responses, file reads — frequently hit payloads prefixed
with a UTF-8 BOM (``\\ufeff``). Python's stdlib ``json.loads`` rejects
these with::

    json.decoder.JSONDecodeError: Unexpected UTF-8 BOM
    (decode using utf-8-sig): line 1 column 1 (char 0)

``json_loads_safe`` strips a single leading BOM before delegating to
``json.loads(text, strict=False)``. The ``strict=False`` behavior is
preserved so embedded control characters (tabs/newlines inside JSON
strings — common in shell output and pretty-printed APIs) still parse.

Use this instead of ``json.loads`` when parsing:

- Output from ``subprocess`` / ``terminal`` commands
- HTTP response bodies (some Windows-hosted APIs emit BOM)
- Files written by editors that default to UTF-8-with-BOM
  (Notepad, some Excel CSV exports, older macOS apps)

For project-internal ``.json`` config files (which never carry BOM),
``json.loads`` is fine and slightly faster — no need to swap.

Design notes:
- Mirrors the shape of ``tools/ansi_strip.py`` (single public function,
  fast-path early-return, zero dependencies, stdlib-only).
- Does NOT recursively strip BOMs from every string value — only the
  leading byte. Recursive stripping would mask real bugs in upstream
  producers.
- Preserves the input unchanged when no BOM is present (fast path).
"""
from __future__ import annotations

import json
from typing import Any, Union

__all__ = ["json_loads_safe", "JSONDecodeError"]

# Re-export so callers don't need a second import for the error type.
JSONDecodeError = json.JSONDecodeError

# Fast-path: skip the strip when the input has no leading BOM byte.
_HAS_LEADING_BOM = "\ufeff"


def json_loads_safe(text: Union[str, bytes, bytearray], **kwargs: Any) -> Any:
    """Parse JSON, tolerating a single leading UTF-8 BOM.

    Drop-in replacement for ``json.loads`` that strips one ``\\ufeff``
    from the start of ``text`` before parsing. ``strict=False`` is the
    default (preserves prior Hermes behavior of allowing embedded
    control characters); callers can override by passing ``strict=...``
    explicitly.

    Args:
        text: JSON text. If a ``str`` and starts with ``\\ufeff``, the
            BOM is stripped before parsing. ``bytes`` input is passed
            through unchanged (stdlib auto-decodes bytes already).
        **kwargs: Forwarded verbatim to ``json.loads``.

    Returns:
        The parsed JSON value (dict, list, str, int, float, bool, None).

    Raises:
        json.JSONDecodeError: If the input is not valid JSON after
            optional BOM stripping. The position reported is relative
            to the post-strip text — matches what the user sees.

    Examples:
        >>> json_loads_safe('\\ufeff{"k": 1}')
        {'k': 1}
        >>> json_loads_safe('{"k": 2}')
        {'k': 2}
        >>> json_loads_safe('{"k": 3}\\ufeff{"k": 4}')  # mid-string BOM
        Traceback (most recent call last):
            ...
        json.decoder.JSONDecodeError: Extra data: ...
    """
    if isinstance(text, str) and text.startswith(_HAS_LEADING_BOM):
        text = text[1:]
    # strict=False preserves Hermes' prior behavior of tolerating
    # embedded control characters (tabs, raw newlines) in JSON strings.
    kwargs.setdefault("strict", False)
    return json.loads(text, **kwargs)


class _BOMStrippingDecoder(json.JSONDecoder):
    """``json.JSONDecoder`` subclass that strips a leading BOM.

    Use when you need to pass a ``cls`` argument to ``json.loads`` or
    ``json.load``. Most callers should prefer the simpler
    ``json_loads_safe`` function above.
    """

    def decode(self, s: str) -> Any:  # type: ignore[override]
        if isinstance(s, str) and s.startswith(_HAS_LEADING_BOM):
            s = s[1:]
        return super().decode(s)