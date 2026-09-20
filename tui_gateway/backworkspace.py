"""Back-workspace pages: the blank page on the back of the Desktop window.

One markdown file per page in ``<HERMES_HOME>/backworkspace/<id>.md``. Page ids share the
session-id shape (``hermes_state_ids.new_session_id``), so pages sort by creation time the
way sessions do. Plain files on purpose, and every result carries the page's path: the client
can name the file to an agent, which reads it with its ordinary file tools.

The home is resolved per call through ``get_hermes_home()``, so the caller's profile scope
(``@_profile_scoped`` in the RPC layer) decides whose pages these are.
"""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_state_ids import new_session_id
from utils import atomic_write_bytes, atomic_write_text

# The id is a file name the client sends back; nothing but the session-id shape may reach the
# filesystem, so a crafted id can never name a path outside the directory. Checked with
# ``fullmatch`` under ASCII rules: ``$`` would admit a trailing newline and ``\d`` other digits.
_PAGE_ID = re.compile(r"\d{8}_\d{6}_[0-9a-f]{6}", re.ASCII)


def pages_dir() -> Path:
    return get_hermes_home() / "backworkspace"


def page_path(page_id: str) -> Path:
    """The file a page id names. Ids only ever come from ``new_session_id`` or ``_PAGE_ID``."""
    return pages_dir() / f"{page_id}.md"


def latest_page() -> dict | None:
    """The most recently created page as ``{id, content, path}``, or None before the first save."""
    ids = sorted(p.stem for p in pages_dir().glob("*.md") if _PAGE_ID.fullmatch(p.stem))
    if not ids:
        return None
    path = page_path(ids[-1])
    return {
        "id": ids[-1],
        "content": path.read_text(encoding="utf-8"),
        "path": str(path),
    }


ASSETS_DIR = "assets"
# What a page may hold beside it. The set is the one an <img> renders and a vision model reads;
# SVG is left out on purpose — it is a document that can carry script, not a picture.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
# Well past any screenshot, and the same ceiling the Desktop reads a local file as a data URL
# with by default (``DATA_URL_READ_DEFAULT_MAX_MB``): storing a picture the client cannot then
# display would leave a link to something invisible.
_MAX_ATTACHMENT_BYTES = 16 * 1024 * 1024


def attach_image(name: str, data_base64: str) -> dict:
    """Store a pasted image beside the pages and return ``{path, href}``.

    ``href`` is what goes in the page — a relative link, so the page and its images stay one
    folder that can be moved or synced. The stored name is minted here: a client-supplied name
    is read for its suffix only, never used as a path.
    """
    suffix = Path(name).suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image type: {suffix or name!r}")
    try:
        blob = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image data is not valid base64") from exc
    if not blob:
        raise ValueError("image data is empty")
    if len(blob) > _MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"image is larger than {_MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB"
        )

    path = pages_dir() / ASSETS_DIR / f"{new_session_id()}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, blob)
    return {"path": str(path), "href": f"{ASSETS_DIR}/{path.name}"}


def save_page(page_id: str | None, content: str) -> str:
    """Write ``content`` to ``page_id`` (a new page when None) and return the page id."""
    if page_id is None:
        page_id = new_session_id()
    elif not _PAGE_ID.fullmatch(page_id):
        raise ValueError(f"invalid page id: {page_id!r}")
    path = page_path(page_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content)
    return page_id
