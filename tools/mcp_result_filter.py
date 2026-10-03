"""Per-server, opt-in filtering of rendered MCP tool results before they reach the model.

``mcp_servers.<name>.result_filter.drop_string_prefixes`` lists prefixes of opaque string values
to remove wherever they appear in a JSON text block or in ``structuredContent``: object entries
whose value starts with one are dropped, as are matching list items. Motivating case: gbrain stamps
every search hit with ``id: gbrain-page:v1:<base64url>``, and OpenRouter's prompt-injection screen
rejects the whole request as ``base64_encoded_injection`` once enough of them pile up. The hits keep
``slug``/``source_id``, which gbrain's tools accept instead. Non-JSON text, ``_meta`` and error
results pass through unchanged; without the config key the rendered output is returned as-is."""

import json
from typing import Any, Optional, Tuple

_FILTERED_KEYS = ("result", "structuredContent")


def apply_result_filter(rendered: str, server_config: Optional[dict]) -> str:
    """Return *rendered* (``_render_call_tool_result`` JSON) with the configured values removed."""
    cfg = (server_config or {}).get("result_filter") or {}
    prefixes = tuple(str(p) for p in cfg.get("drop_string_prefixes") or () if p)
    if not prefixes:
        return rendered
    try:
        payload = json.loads(rendered)
    except (TypeError, ValueError):
        return rendered
    if not isinstance(payload, dict) or not any(k in payload for k in _FILTERED_KEYS):
        return rendered
    for key in _FILTERED_KEYS:
        if key in payload:
            payload[key] = _filter_text(payload[key], prefixes) if isinstance(payload[key], str) \
                else _drop(payload[key], prefixes)
    return json.dumps(payload, ensure_ascii=False)


def _filter_text(text: str, prefixes: Tuple[str, ...]) -> str:
    """A JSON object/array text block is filtered and re-serialised (pretty-printed when it was)."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(parsed, (dict, list)):
        return text
    return json.dumps(_drop(parsed, prefixes), ensure_ascii=False, indent=2 if "\n" in text else None)


def _drop(value: Any, prefixes: Tuple[str, ...]) -> Any:
    def matches(v: Any) -> bool:
        return isinstance(v, str) and v.startswith(prefixes)

    if isinstance(value, dict):
        return {k: _drop(v, prefixes) for k, v in value.items() if not matches(v)}
    if isinstance(value, list):
        return [_drop(v, prefixes) for v in value if not matches(v)]
    return value
