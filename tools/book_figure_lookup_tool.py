"""Book Figure Lookup Tool — resolve "Figure 11-1" to a local image path.

Each indexed book lives at::

    ~/.openclaw/workspace-auto/<book>/figures_index.json

The index is produced by an off-line indexer (see scripts/build_figures_index.py).
This tool reads the indexes at call time, finds entries whose ``id`` or
``caption`` matches the user-supplied figure reference, and returns
``image_paths`` so the agent can attach them to a Slack/Telegram/etc. message
via the existing ``send_message`` ``MEDIA:`` tag mechanism.

Page-level images (not figure-level crops) are returned — multiple figures may
share a page, and the surrounding page text/captions are visible in the image.
The tool surfaces this fact in its response so the agent can warn the user.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


WORKSPACE_AUTO = Path("/Users/neuromark/.openclaw/workspace-auto")

# Patterns the agent (or end-user via the agent) might use to refer to a figure.
# Accepted: "Figure 11-1", "Fig. 11-1", "fig 11-1", "11-1", "그림 11-1",
#           "Table 3-2", "표 3-2", with - or . as separator.
_ID_PATTERNS = [
    re.compile(r"^\s*(?:figure|fig\.?|그림)\s*(?P<chap>\d+)[-\.](?P<num>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:table|표)\s*(?P<chap>\d+)[-\.](?P<num>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?P<chap>\d+)[-\.](?P<num>\d+)\s*$"),  # bare "11-1"
]


def _parse_figure_ref(ref: str) -> Optional[Dict[str, Any]]:
    """Return {kind, chapter, num} or None if the reference is unparseable."""
    if not ref:
        return None
    s = ref.strip()
    lower = s.lower()
    if lower.startswith(("table", "표")):
        kind = "Table"
    elif lower.startswith(("figure", "fig", "그림")):
        kind = "Figure"
    else:
        kind = "Figure"  # default
    for pat in _ID_PATTERNS:
        m = pat.match(s)
        if m:
            return {
                "kind": kind,
                "chapter": int(m.group("chap")),
                "num": int(m.group("num")),
            }
    return None


def _load_indexes() -> List[Dict[str, Any]]:
    """Read every figures_index.json under workspace-auto/."""
    indexes = []
    if not WORKSPACE_AUTO.exists():
        return indexes
    for path in WORKSPACE_AUTO.glob("*/figures_index.json"):
        try:
            indexes.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("Failed to read figure index %s: %s", path, e)
    return indexes


def _normalize_book_name(name: str) -> str:
    """Lowercase + collapse separators for fuzzy book matching."""
    return re.sub(r"[\s_\-]+", "", name.lower())


def _match_book(query: Optional[str], book: str) -> bool:
    if not query:
        return True
    return _normalize_book_name(query) in _normalize_book_name(book)


def book_figure_lookup(
    figure_id: str,
    book: Optional[str] = None,
    max_results: int = 5,
) -> Dict[str, Any]:
    """Look up a figure/table reference across indexed books.

    Args:
        figure_id: e.g. "Figure 11-1", "11-1", "Table 3-2", "그림 5-2".
        book: optional book-name substring filter (case-insensitive,
            ignores spaces/underscores/dashes).
        max_results: cap on returned matches.

    Returns:
        {
          "found": bool,
          "query": {...parsed...},
          "matches": [
            {
              "book": str,
              "id": str,
              "caption": str,
              "image_paths": [str, ...],   # absolute paths
              "missing_files": [str, ...], # paths that no longer exist
              "source_wiki": str,
            },
            ...
          ],
          "note": str,                     # always present; explains caveats
        }
    """
    parsed = _parse_figure_ref(figure_id)
    if not parsed:
        return {
            "found": False,
            "error": f"Could not parse figure reference: {figure_id!r}",
            "hint": "Try: 'Figure 11-1', '11-1', 'Table 3-2', or '그림 5-2'",
        }

    indexes = _load_indexes()
    if not indexes:
        return {
            "found": False,
            "query": parsed,
            "error": "No figure indexes found.",
            "hint": f"Build one at {WORKSPACE_AUTO}/<book>/figures_index.json",
        }

    matches: List[Dict[str, Any]] = []
    for idx in indexes:
        book_name = idx.get("book", "?")
        if not _match_book(book, book_name):
            continue
        for fig in idx.get("figures", []):
            if (fig.get("kind") == parsed["kind"]
                    and fig.get("chapter") == parsed["chapter"]
                    and fig.get("num") == parsed["num"]):
                paths = fig.get("image_paths") or []
                existing = [p for p in paths if os.path.exists(p)]
                missing = [p for p in paths if not os.path.exists(p)]
                matches.append({
                    "book": book_name,
                    "id": fig.get("id"),
                    "caption": fig.get("caption", ""),
                    "image_paths": existing,
                    "missing_files": missing,
                    "source_wiki": fig.get("source_wiki", ""),
                })
                if len(matches) >= max_results:
                    break

    note = (
        "image_paths are PAGE-level screenshots (not figure-level crops). "
        "A page may contain the requested figure plus adjacent figures, captions, "
        "and surrounding body text. Multiple figures from the same chapter often "
        "share a single page image."
    )

    return {
        "found": bool(matches),
        "query": parsed,
        "matches": matches,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Tool schema + registration
# ---------------------------------------------------------------------------

BOOK_FIGURE_LOOKUP_SCHEMA = {
    "name": "book_figure_lookup",
    "description": (
        "Resolve a textbook figure/table reference (e.g. 'Figure 11-1') to "
        "a local image file path that can be attached to a chat message via "
        "the send_message MEDIA: tag.\n\n"
        "USE THIS WHEN:\n"
        "- The user asks to see / show / display / send a specific figure or "
        "table from a clinical textbook.\n"
        "- You want to back up a textual answer with the source figure.\n\n"
        "RESULT:\n"
        "- 'image_paths' contains absolute paths to PAGE-level screenshots "
        "(not figure-level crops). A page may contain multiple figures.\n"
        "- To attach: append a line like 'MEDIA:<path>' to your reply; the "
        "send_message tool then uploads it natively (Telegram/Discord/Slack/"
        "Matrix/WeChat/Signal).\n\n"
        "ARGUMENTS:\n"
        "- figure_id (required): 'Figure 11-1', 'Fig. 11-1', '11-1', "
        "'Table 3-2', or Korean '그림 5-2' / '표 2-3'.\n"
        "- book (optional): substring filter on book name when the same "
        "figure ID exists in multiple books — e.g. 'Atlas of EEG', '뇌졸중'.\n\n"
        "If 'found' is false, the index does not yet contain the requested "
        "figure — tell the user the figure isn't available in the indexed "
        "library rather than fabricating an image."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "figure_id": {
                "type": "string",
                "description": "Figure or table reference, e.g. 'Figure 11-1', '11-1', '그림 5-2', 'Table 3-2'.",
            },
            "book": {
                "type": "string",
                "description": "Optional book-name substring filter (case-insensitive).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matches to return (default 5).",
                "default": 5,
            },
        },
        "required": ["figure_id"],
    },
}


from tools.registry import registry  # noqa: E402  (registry import after schema)


def _handler(args: Dict[str, Any], **_kw) -> str:
    result = book_figure_lookup(
        figure_id=args.get("figure_id", ""),
        book=args.get("book"),
        max_results=int(args.get("max_results", 5)),
    )
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="book_figure_lookup",
    toolset="knowledge",
    schema=BOOK_FIGURE_LOOKUP_SCHEMA,
    handler=_handler,
    emoji="📖",
)
