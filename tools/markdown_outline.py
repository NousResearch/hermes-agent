"""Markdown heading-outline scanning for read_file's optional outline mode (#103374).

Pure stdlib text parsing — no model calls, network, or new dependency:

* Recognizes ATX (``#``..``######``) and Setext (``===`` / ``---`` underline)
  headings, in source order.
* Ignores heading-like lines inside fenced code blocks (backtick and tilde).
* Preserves duplicate headings as separate entries with their own line.
* Reports 1-based source line numbers (CRLF-safe via ``str.splitlines``).

The scanner is deliberately line-oriented and lenient (CommonMark-ish rather
than full CommonMark): it exists to orient the agent, not to render HTML.
"""

import re

# Lowercased file suffixes that outline mode treats as Markdown.
MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown", ".mdown", ".mkd"})

# Per-call output bounds. A bounded page keeps the JSON well under the tool
# registry's max_result_size_chars even for pathological documents; the caller
# marks truncation and continues with a new offset instead of dumping it all.
OUTLINE_MAX_ENTRIES = 500
OUTLINE_HEADING_MAX_CHARS = 150

# Up to 3 leading spaces are part of the syntax, not the content.
_ATX_RE = re.compile(r" {0,3}(#{1,6})(?:[ \t]+(.*))?$")
# A trailing run of hashes preceded by whitespace is a closing sequence
# (CommonMark), not heading text: "# Title ###" -> "Title".
_CLOSING_HASHES_RE = re.compile(r"[ \t]+#+[ \t]*$")
_FENCE_OPEN_RE = re.compile(r" {0,3}(`{3,}|~{3,})")
_FENCE_CLOSE_RE = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*$")
_SETEXT_UNDERLINE_RE = re.compile(r" {0,3}(=+|-+)[ \t]*$")


def markdown_outline(content: str) -> list:
    """Return every Markdown heading in *content*, in source order.

    Each entry is ``{"line": int, "level": int, "heading": str}`` where
    ``line`` is the 1-based source line of the heading text (the ATX marker
    line, or the Setext text line above its underline).
    """
    entries = []
    fence_char = None  # None = outside a fenced code block
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if fence_char is not None:
            close = _FENCE_CLOSE_RE.match(line)
            if close is not None and close.group(1)[0] == fence_char:
                fence_char = None
            continue
        open_fence = _FENCE_OPEN_RE.match(line)
        if open_fence is not None:
            fence_char = open_fence.group(1)[0]
            continue
        stripped = line.strip()
        if not stripped:
            continue
        atx = _ATX_RE.match(line)
        if atx is not None:
            text = _CLOSING_HASHES_RE.sub("", atx.group(2) or "").strip()
            entries.append({"line": i + 1, "level": len(atx.group(1)), "heading": text})
            continue
        # Setext: the previous line is paragraph text and this line is an
        # underline of "=" (H1) or "-" (H2). CommonMark requires the text line
        # to be a paragraph, so heading/fence/blank/underline lines are excluded.
        if i > 0:
            underline = _SETEXT_UNDERLINE_RE.match(line)
            if underline is not None:
                prev = lines[i - 1]
                if (prev.strip() and _ATX_RE.match(prev) is None
                        and _FENCE_OPEN_RE.match(prev) is None
                        and _SETEXT_UNDERLINE_RE.match(prev) is None):
                    level = 1 if underline.group(1).startswith("=") else 2
                    entries.append({"line": i, "level": level, "heading": prev.strip()})
    return entries
