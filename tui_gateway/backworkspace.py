"""Back-workspace pages: the blank page on the back of the Desktop window.

One markdown file per page in ``<HERMES_HOME>/backworkspace/<id>.md``. Page ids share the
session-id shape (``hermes_state_ids.new_session_id``), so pages sort by creation time the
way sessions do. Plain files on purpose: the agent reads them with its ordinary file tools.

The home is resolved per call through ``get_hermes_home()``, so the caller's profile scope
(``@_profile_scoped`` in the RPC layer) decides whose pages these are.
"""

from __future__ import annotations

import re
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_state_ids import new_session_id
from utils import atomic_write_text

# The id is a file name the client sends back; nothing but the session-id shape may reach the
# filesystem, so a crafted id can never name a path outside the directory. Checked with
# ``fullmatch`` under ASCII rules: ``$`` would admit a trailing newline and ``\d`` other digits.
_PAGE_ID = re.compile(r"\d{8}_\d{6}_[0-9a-f]{6}", re.ASCII)


def pages_dir() -> Path:
    return get_hermes_home() / "backworkspace"


def latest_page() -> dict | None:
    """The most recently created page as ``{id, content}``, or None before the first save."""
    ids = sorted(p.stem for p in pages_dir().glob("*.md") if _PAGE_ID.fullmatch(p.stem))
    if not ids:
        return None
    path = pages_dir() / f"{ids[-1]}.md"
    return {"id": ids[-1], "content": path.read_text(encoding="utf-8")}


def save_page(page_id: str | None, content: str) -> str:
    """Write ``content`` to ``page_id`` (a new page when None) and return the page id."""
    if page_id is None:
        page_id = new_session_id()
    elif not _PAGE_ID.fullmatch(page_id):
        raise ValueError(f"invalid page id: {page_id!r}")
    directory = pages_dir()
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / f"{page_id}.md", content)
    return page_id
