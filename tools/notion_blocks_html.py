"""Render a Notion block tree as Zendesk Help Center article HTML.

Used by ``zendesk_create_draft`` to build the article body FROM the approved
Release Assets row, so the model never supplies the HTML. The output is safe
by construction rather than sanitized after the fact: every piece of text is
HTML-escaped, the only tags emitted are the fixed set below, attributes are
limited to ``href`` on links whose scheme is ``https``/``http``/``mailto``,
and a block type this module doesn't know is reported in ``unsupported``
instead of being passed through. So no ``<script>``, ``<iframe>``,
``<style>`` or ``on*`` attribute can reach Zendesk from row content.

Images are never embedded: Notion-hosted file URLs expire within the hour,
and uploading attachments isn't supported yet, so their captions are returned
in ``images`` for the caller to refuse or report.

Not a tool module (registers nothing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from urllib.parse import urlsplit

_SAFE_SCHEMES = frozenset({"https", "http", "mailto"})
_LIST_TAG = {"bulleted_list_item": "ul", "numbered_list_item": "ol", "to_do": "ul"}
_HEADING = {"heading_1": "h1", "heading_2": "h2", "heading_3": "h3", "heading_4": "h4"}
_SKIP = frozenset({"table_of_contents", "breadcrumb"})
_MAX_DEPTH = 8


@dataclass
class Rendered:
    html: str
    images: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def _safe_href(href) -> str | None:
    if not isinstance(href, str) or not href.strip():
        return None
    href = href.strip()
    try:
        scheme = urlsplit(href).scheme.lower()
    except ValueError:
        return None
    if scheme not in _SAFE_SCHEMES or any(ch.isspace() for ch in href):
        return None
    return escape(href, quote=True)


def _rich(items) -> str:
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = escape(str(item.get("plain_text") or "")).replace("\n", "<br>")
        if not text:
            continue
        ann = item.get("annotations") or {}
        if ann.get("code"):
            text = f"<code>{text}</code>"
        if ann.get("bold"):
            text = f"<strong>{text}</strong>"
        if ann.get("italic"):
            text = f"<em>{text}</em>"
        if ann.get("strikethrough"):
            text = f"<s>{text}</s>"
        if ann.get("underline"):
            text = f"<u>{text}</u>"
        href = _safe_href(item.get("href"))
        if href:
            text = f'<a href="{href}">{text}</a>'
        out.append(text)
    return "".join(out)


def _plain(items) -> str:
    return "".join(str(i.get("plain_text") or "") for i in items or [] if isinstance(i, dict)).strip()


class _Renderer:
    def __init__(self) -> None:
        self.images: list[str] = []
        self.unsupported: list[str] = []

    def blocks(self, blocks, depth: int = 0) -> str:
        if depth > _MAX_DEPTH:
            self.unsupported.append("nesting too deep")
            return ""
        out: list[str] = []
        open_list: str | None = None
        for b in blocks or []:
            if not isinstance(b, dict):
                continue
            kind = str(b.get("type") or "")
            tag = _LIST_TAG.get(kind)
            if tag != open_list:
                if open_list:
                    out.append(f"</{open_list}>")
                if tag:
                    out.append(f"<{tag}>")
                open_list = tag
            out.append(self.block(b, kind, depth))
        if open_list:
            out.append(f"</{open_list}>")
        return "".join(out)

    def children(self, b: dict, depth: int) -> str:
        return self.blocks(b.get("children") or [], depth + 1)

    def block(self, b: dict, kind: str, depth: int) -> str:
        body = b.get(kind) if isinstance(b.get(kind), dict) else {}
        text = _rich(body.get("rich_text"))
        if kind == "paragraph":
            inner = f"<p>{text}</p>" if text else ""
            return inner + self.children(b, depth)
        if kind in _HEADING:
            h = _HEADING[kind]
            return (f"<{h}>{text}</{h}>" if text else "") + self.children(b, depth)
        if kind in _LIST_TAG:
            if kind == "to_do":
                text = ("☑ " if body.get("checked") else "☐ ") + text
            return f"<li>{text}{self.children(b, depth)}</li>"
        if kind in ("quote", "callout"):
            return f"<blockquote>{text}{self.children(b, depth)}</blockquote>"
        if kind == "code":
            return f"<pre><code>{escape(_plain(body.get('rich_text')))}</code></pre>"
        if kind == "divider":
            return "<hr>"
        if kind == "toggle":
            return (f"<p><strong>{text}</strong></p>" if text else "") + self.children(b, depth)
        if kind == "table":
            return self.table(b, body)
        if kind in ("column_list", "column"):
            return self.children(b, depth)
        if kind == "image":
            self.images.append(_plain(body.get("caption")) or "(image without caption)")
            return ""
        if kind == "bookmark":
            href = _safe_href(body.get("url"))
            if href:
                return f'<p><a href="{href}">{href}</a></p>'
            self.unsupported.append(kind)
            return ""
        if kind in _SKIP:
            return ""
        self.unsupported.append(kind or "unknown")
        return ""

    def table(self, b: dict, body: dict) -> str:
        header = bool(body.get("has_column_header"))
        rows = []
        for i, row in enumerate(b.get("children") or []):
            cells = ((row or {}).get("table_row") or {}).get("cells") or []
            cell_tag = "th" if header and i == 0 else "td"
            rows.append("<tr>" + "".join(
                f"<{cell_tag}>{_rich(c)}</{cell_tag}>" for c in cells) + "</tr>")
        return f"<table><tbody>{''.join(rows)}</tbody></table>"


def render_blocks(blocks) -> Rendered:
    """Render a block list (each block may carry a ``children`` list)."""
    r = _Renderer()
    html = r.blocks(blocks)
    return Rendered(html=html, images=r.images, unsupported=r.unsupported)
