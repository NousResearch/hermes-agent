"""Per-server, opt-in filtering of MCP tool results before they are rendered for the model.

``mcp_servers.<name>.result_filter.drop_string_prefixes`` lists prefixes of opaque string values
to remove wherever they appear in a JSON text block or in ``structuredContent``: object entries
whose value starts with one are dropped, as are matching list items. Motivating case: gbrain stamps
every search hit with ``id: gbrain-page:v1:<base64url>``, and OpenRouter's prompt-injection screen
rejects the whole request as ``base64_encoded_injection`` once enough of them pile up. The hits keep
``slug``/``source_id``, which gbrain's tools accept instead.

The filter runs on the raw ``CallToolResult``, one content block at a time, before
``_render_call_tool_result`` joins the blocks and applies the hard size cap: a status line next to
the JSON block, or a result big enough to be truncated, would otherwise leave nothing parseable to
filter. Non-JSON text blocks, non-text blocks, ``_meta`` and error results pass through unchanged,
as does every block nothing was dropped from; without the config key the result is returned as-is."""

import copy
import json
from typing import Any, Optional

from pydantic import BaseModel

from tools.mcp_tool_common import mcp_field

_STRUCTURED_FIELDS = ("structured_content", "structuredContent")  # mcp 2.x / 1.x spelling


def apply_result_filter(result: Any, server_config: Optional[dict]) -> Any:
    """Return *result* with the configured values removed (*result* itself when nothing changes)."""
    prefixes = _configured_prefixes(server_config)
    if not prefixes or mcp_field(result, "is_error", "isError", False):
        return result
    updates: dict = {}
    blocks = list(getattr(result, "content", None) or [])
    filtered_blocks = [_filter_block(block, prefixes) for block in blocks]
    if any(new is not old for new, old in zip(filtered_blocks, blocks)):
        updates["content"] = filtered_blocks
    for field in _STRUCTURED_FIELDS:
        structured = getattr(result, field, None)
        if isinstance(structured, (dict, list)):
            dropped = _drop(structured, prefixes)
            if dropped != structured:
                updates[field] = dropped
            break
    return _replace(result, updates) if updates else result


def _configured_prefixes(server_config: Optional[dict]) -> tuple[str, ...]:
    cfg = server_config.get("result_filter") if isinstance(server_config, dict) else None
    raw = cfg.get("drop_string_prefixes") if isinstance(cfg, dict) else None
    if isinstance(raw, str):  # a bare YAML scalar is one prefix, not one prefix per character
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(p) for p in raw if p)


def _filter_block(block: Any, prefixes: tuple[str, ...]) -> Any:
    """A text block holding a JSON object/array is filtered and re-serialised (pretty-printed when
    it was); any other block, or one nothing was dropped from, is returned as the same object."""
    text = getattr(block, "text", None)
    if not isinstance(text, str):
        return block
    try:
        parsed = json.loads(text)
    except ValueError:
        return block
    if not isinstance(parsed, (dict, list)):
        return block
    dropped = _drop(parsed, prefixes)
    if dropped == parsed:
        return block
    return _replace(block, {"text": json.dumps(dropped, ensure_ascii=False, indent=2 if "\n" in text else None)})


def _drop(value: Any, prefixes: tuple[str, ...]) -> Any:
    def matches(v: Any) -> bool:
        return isinstance(v, str) and v.startswith(prefixes)

    if isinstance(value, dict):
        return {k: _drop(v, prefixes) for k, v in value.items() if not matches(v)}
    if isinstance(value, list):
        return [_drop(v, prefixes) for v in value if not matches(v)]
    return value


def _replace(obj: Any, updates: dict) -> Any:
    """A copy of *obj* with *updates* applied; the server's own result object is never mutated."""
    if isinstance(obj, BaseModel):
        return obj.model_copy(update=updates)
    clone = copy.copy(obj)
    for name, value in updates.items():
        setattr(clone, name, value)
    return clone
