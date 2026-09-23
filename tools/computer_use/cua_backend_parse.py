"""Pure parsing helpers for the cua-driver backend: MCP result flattening, ``list_windows`` /
``get_window_state`` payload normalisation, key combos. No I/O, no module state — every function
depends only on its inputs, so the MCP and CLI transports share them safely."""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, UIElement, image_dimensions_from_bytes

# Linux/X11 surfaces GNOME Shell / desktop backdrop windows ahead of real app windows with no useful z-order; they
# are targetable but capture as empty, so default capture skips them.
_NON_APP_WINDOW_TITLE_PREFIXES = ("@!", "Desktop", "gnome-shell", "GNOME Shell")  # "@!" = GNOME helpers

_ELEMENT_LINE_RE = re.compile(
    r'^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)'
    r'(?:'
      r'\s*=\s*"([^"]*)"'              # = "value"
      r'|\s+"([^"]*)"'                 # "value"
      r'|\s+\((?!\d+\))([^)]*)\)'      # (value) but not a pure-digit (order) number
    r')?'
    r'(?:\s+(?:\(\d+\)\s+)?id=([^\s\[\]]+))?',  # optional id=value (after an optional (order))
    re.MULTILINE,
)
"""get_window_state AX-tree markdown line: ``[N] AXRole`` + label in one of four forms (``= "value"``, ``"quoted"``,
``(paren)``, ``id=Label`` optionally after an ``(order)`` number). A parenthesised pure-digit group is an ORDER
index, not a label, and is excluded so the id= label wins. Group 1 index, group 2 role, groups 3-6 the label."""

def _mcp_field(obj, snake: str, camel: str, default=None):
    """Read an MCP model field across the 1.x -> 2.x rename: mcp 2.0 exposes snake_case attributes and keeps
    camelCase only as a serialization alias, so ``getattr(result, "isError", False)`` is False for every 2.x result
    and a denied call looks like success. Deliberately duplicated from ``tools.mcp_tool.mcp_field`` so computer_use
    never loads the much larger config-driven MCP client module."""
    return getattr(obj, snake, getattr(obj, camel, default))

def _action_result_from(name: str, ok: bool, message: str, meta: Dict[str, Any],
                        structured: Dict[str, Any], *, requested_delivery: Optional[str] = None) -> ActionResult:
    """Build an ActionResult, lifting cua-driver's structured verdict. structuredContent is canonical, the flattened
    ``meta`` copy the fallback. Every structured field is additive: a driver that omits one leaves the attribute
    ``None`` so old drivers see unchanged behavior.

    See the action response shape in cua-driver's mcp-tool-notes and NousResearch/hermes-agent#67052.
    """
    sc = structured if isinstance(structured, dict) else {}

    def _raw(key: str) -> Any:
        return sc.get(key) if key in sc else meta.get(key)

    def _typed(value: Any, typ) -> Any:
        return value if isinstance(value, typ) else None

    return ActionResult(
        ok=ok, action=name, message=message, meta=meta,
        verified=_typed(_raw("verified"), bool), effect=_typed(_raw("effect"), str),
        escalation=_typed(_raw("escalation"), dict), path=_typed(_raw("path"), str),
        degraded=_typed(_raw("degraded"), bool),
        # What we asked for; the driver's `path` records the rung that ran.
        delivery_mode=_typed(requested_delivery, str),
        # Refusal/limitation code — drivers spell it "code" or "reason_code".
        code=_typed(_raw("code") or _raw("reason_code"), str),
    )

def _z_index_uninformative(windows: List[Dict[str, Any]]) -> bool:
    """True when every window shares the same z_index (common on Linux/X11)."""
    return len({w.get("z_index", 0) for w in windows}) <= 1

def _parse_xprop_net_active_window(stdout: str) -> Optional[int]:
    """Parse ``xprop -root _NET_ACTIVE_WINDOW`` stdout into a window id: the ``window id # 0x...`` form, falling
    back to the first hex token."""
    text = stdout or ""
    match = re.search(r"window id # (0x[0-9a-fA-F]+)", text) or re.search(r"(0x[0-9a-fA-F]+)", text)
    return int(match.group(1), 16) if match else None

def _is_real_app_window(w: Dict[str, Any]) -> bool:
    """Return False for desktop/shell helper windows that capture as empty."""
    title = w.get("title", "")
    return not any(title.startswith(p) or title.lower().startswith(p.lower()) for p in _NON_APP_WINDOW_TITLE_PREFIXES)

def _parse_elements_from_tree(markdown: str) -> List[UIElement]:
    """Parse UIElements from get_window_state AX-tree markdown — last-resort fallback for drivers without
    ``structuredContent.elements``. Bounds are always ``(0, 0, 0, 0)`` (the markdown carries none), fine for
    element-index clicks since the driver resolves the frame.

    Last-resort fallback for cua-driver builds that don't carry the canonical ``structuredContent.elements``
    array (see ``_parse_elements_from_structured`` — Surface 2 of #47072 prefers that path).
    """
    return [
        # groups 3-6: value / quoted / paren / id= label (first non-None wins)
        UIElement(index=int(m.group(1)), role=m.group(2),
                  label=m.group(3) or m.group(4) or m.group(5) or m.group(6) or "", bounds=(0, 0, 0, 0))
        for m in _ELEMENT_LINE_RE.finditer(markdown)
    ]

def _parse_elements_from_structured(raw_elements: List[Dict[str, Any]]) -> List[UIElement]:
    """Read the canonical ``structuredContent.elements`` array: ``element_index``, ``role``, ``label`` and, when
    AT-SPI / AXFrame returned usable bounds, ``frame`` ``{x, y, w, h}`` — so real pixel bounds survive (the
    markdown path loses them). Malformed entries are skipped.

    Surface 2 of NousResearch/hermes-agent#47072: read the canonical ``structuredContent.elements`` array
    cua-driver-rs emits on every ``get_window_state`` response (trycua/cua#1961).
    """
    elements: List[UIElement] = []
    for raw in raw_elements:
        idx = raw.get("element_index") if isinstance(raw, dict) else None
        if not isinstance(idx, int):
            continue
        role, label, frame, token = (raw.get(k) for k in ("role", "label", "frame", "element_token"))
        bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
        with contextlib.suppress(TypeError, ValueError):
            if isinstance(frame, dict) and frame:
                bounds = tuple(int(frame.get(k, 0)) for k in ("x", "y", "w", "h"))  # type: ignore[assignment]
        elements.append(UIElement(
            index=idx, role=role if isinstance(role, str) else "", label=label if isinstance(label, str) else "",
            bounds=bounds,
            # Opaque `s{snapshot_hex}:{index}` token — the driver owns parse + LRU semantics.
            element_token=token if isinstance(token, str) and token else None,
        ))
    return elements

def _image_dimensions_from_bytes(raw: bytes) -> Tuple[int, int]:
    """Best-effort PNG/JPEG dimension sniff; ``(0, 0)`` when unreadable or non-positive."""
    dims = image_dimensions_from_bytes(raw)
    return dims if dims and dims[0] > 0 and dims[1] > 0 else (0, 0)

def _split_tree_text(full_text: str) -> Tuple[str, str]:
    """Split get_window_state text into (summary_line, tree_markdown)."""
    summary, _, tree = full_text.partition("\n")
    return summary, tree

_MODIFIER_NAMES = frozenset({"cmd", "command", "shift", "option", "alt", "ctrl", "control", "fn"})
_KEY_ALIASES = {"command": "cmd", "alt": "option", "control": "ctrl"}

def _parse_key_combo(keys: str) -> Tuple[Optional[str], List[str]]:
    """Parse 'cmd+s' / 'ctrl-alt-t' into (key, modifiers); last non-modifier wins."""
    modifiers: List[str] = []
    key = None
    for part in (p.strip().lower() for p in re.split(r'[+\-]', keys) if p.strip()):
        normalized = _KEY_ALIASES.get(part, part)
        if normalized in _MODIFIER_NAMES:
            modifiers.append(normalized)
        else:
            key = part
    return key, modifiers

def _tool_envelope(data: Any, images: List[str], structured: Any, is_error: bool,
                   image_mime_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """The normalised tool-result dict every transport emits: ``{data, images, [image_mime_types,] structuredContent,
    isError}``. ``image_mime_types`` is only present when the transport can report it (MCP image parts)."""
    out: Dict[str, Any] = {"data": data, "images": images}
    if image_mime_types is not None:
        out["image_mime_types"] = image_mime_types
    out["structuredContent"], out["isError"] = structured, is_error
    return out

def _extract_tool_result(mcp_result: Any) -> Dict[str, Any]:
    """Flatten an mcp CallToolResult into ``{data, images, image_mime_types, structuredContent, isError}``. ``data``
    is the joined text parts (parsed as JSON when it looks like JSON); ``image_mime_types`` is parallel to
    ``images`` with ``""`` where the part carried no mimeType (older drivers — callers then sniff the base64 prefix).

    `image_mime_types` is the explicit `mimeType` cua-driver emits on every image part as of trycua/cua#1961
    (Surface 7 of NousResearch/hermes-agent#47072). Each entry corresponds index-for-index with `images`; an
    empty string entry signals the part carried no mimeType (older cua-driver build), and the caller should
    fall back to base64-prefix sniffing.
    """
    data: Any = None
    images: List[str] = []
    image_mime_types: List[str] = []
    text_chunks: List[str] = []
    for part in getattr(mcp_result, "content", []) or []:
        ptype = getattr(part, "type", None)
        if ptype == "text":
            text_chunks.append(getattr(part, "text", "") or "")
        elif ptype == "image" and getattr(part, "data", None):
            images.append(part.data)
            image_mime_types.append(_mcp_field(part, "mime_type", "mimeType") or "")
    if text_chunks:
        joined = "\n".join(t for t in text_chunks if t)
        try:
            data = json.loads(joined) if joined.strip().startswith(("{", "[")) else joined
        except json.JSONDecodeError:
            data = joined
    return _tool_envelope(
        data, images, _mcp_field(mcp_result, "structured_content", "structuredContent") or None,
        # Identity, not truthiness: mocks/proxies synthesize truthy attributes.
        _mcp_field(mcp_result, "is_error", "isError", False) is True, image_mime_types)

def _image_from_tool_result(out: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Pull ``(b64, mime_type)`` out of a flattened tool result. cua-driver delivers screenshots as an MCP ``image``
    part (``out["images"]``) or as ``screenshot_png_b64`` in structuredContent (newer builds, CLI transport);
    checking both keeps capture() robust to the driver moving it."""
    images = out.get("images") or []
    if images and images[0]:
        mimes = out.get("image_mime_types") or []
        return images[0], (mimes[0] if mimes and mimes[0] else None)
    structured = out.get("structuredContent") or {}
    b64 = structured.get("screenshot_png_b64") or structured.get("png_b64")
    if b64:
        return b64, (structured.get("screenshot_mime_type") or structured.get("mime_type") or None)
    return None, None

def _int_or_none(value: Any) -> Optional[int]:
    """``int(value)`` for int/numeric-str inputs; None for bools, other types and malformed strings."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        return int(value)
    except ValueError:
        return None

def _positive_int(value: Any) -> Optional[int]:
    """Return a positive integer, rejecting booleans and malformed values."""
    parsed = _int_or_none(value)
    return parsed if parsed is not None and parsed > 0 else None

def _is_placeholder_id(value: Any) -> bool:
    """True when *value* is a schema-filler id (``0`` / negative) rather than a target: some providers zero-fill
    every optional integer, and treating that as targeting would drop the caller's ``app=``. Non-numeric values
    are NOT placeholders — they still reach validation."""
    parsed = _int_or_none(value)
    return parsed is not None and parsed <= 0

def _ingest_windows(raw_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise cua-driver ``list_windows`` entries, dropping unusable ones. Every downstream call needs integer
    ``pid`` and ``window_id``; on X11 the PID comes from the optional ``_NET_WM_PID`` property, so root/panel/popup
    windows report ``pid: null`` — skip those instead of aborting the enumeration. ``z_index``: higher = closer to
    front; Wayland's null (undefined stacking) sorts lowest so real windows stay above the desktop."""
    windows: List[Dict[str, Any]] = []
    for w in raw_windows:
        if not isinstance(w, dict):  # untrusted compatibility envelopes
            continue
        pid_int, window_id_int = _positive_int(w.get("pid")), _positive_int(w.get("window_id"))
        if pid_int is None or window_id_int is None:
            continue
        z_raw, app_name, title = w.get("z_index"), w.get("app_name", ""), w.get("title", "")
        windows.append({
            "app_name": app_name if isinstance(app_name, str) else "",
            "pid": pid_int,
            "window_id": window_id_int,
            # Only explicit False means off-screen; null (Linux 0.6.x) means unknown.
            "off_screen": w.get("is_on_screen") is False,
            "title": title if isinstance(title, str) else "",
            "z_index": z_raw if isinstance(z_raw, (int, float)) and not isinstance(z_raw, bool) else 0,
        })
    return windows

def _windows_from_tool_result(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list_windows payloads across cua-driver result shapes: structuredContent.windows, then ``windows`` /
    ``_legacy_windows`` in the text payload, then on the envelope itself."""
    candidates = ((out.get("structuredContent"), ("windows",)),
                  (out.get("data"), ("windows", "_legacy_windows")),
                  (out, ("windows", "_legacy_windows")))
    for container, keys in candidates:
        if isinstance(container, dict):
            for key in keys:
                value = container.get(key)
                if isinstance(value, list) and value:
                    return value
    return []

def _apps_from_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    apps: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for summary in _ingest_windows(windows):
        name, key = summary["app_name"], (summary["app_name"], summary["pid"])
        if name and key not in seen:
            seen.add(key)
            apps.append({"name": name, "pid": summary["pid"]})
    return apps
