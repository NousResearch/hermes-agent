"""Back-workspace pages (methods_backworkspace): the page on the back of the Desktop window."""

from __future__ import annotations

from .base import Result
from .common import ProfileParams
from .registry import method


class BackworkspacePage(Result):
    id: str
    content: str
    path: str  # the page file on the backend host, for handing to an agent's file tools


class BackworkspaceOpenResult(Result):
    page: BackworkspacePage | None = None


method(
    "backworkspace.open",
    params=ProfileParams,
    result=BackworkspaceOpenResult,
    doc="The most recent back-workspace page, or null before the first save.",
)


class BackworkspaceSaveParams(ProfileParams):
    id: str | None = None
    content: str | None = None


class BackworkspaceSaveResult(Result):
    id: str
    path: str


method(
    "backworkspace.save",
    params=BackworkspaceSaveParams,
    result=BackworkspaceSaveResult,
    doc="Write a back-workspace page (a new page when no id is given); returns its id and file path.",
)


class BackworkspaceAttachParams(ProfileParams):
    """``name`` is read for its suffix only; ``data`` is the image, base64."""

    name: str | None = None
    data: str | None = None


class BackworkspaceAttachResult(Result):
    path: str
    href: str


method(
    "backworkspace.attach",
    params=BackworkspaceAttachParams,
    result=BackworkspaceAttachResult,
    doc="Store a pasted image beside the back-workspace pages; returns its path and the page's link to it.",
)
