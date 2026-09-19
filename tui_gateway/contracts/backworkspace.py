"""Back-workspace pages (methods_backworkspace): the page on the back of the Desktop window."""

from __future__ import annotations

from .base import Result
from .common import ProfileParams
from .registry import method


class BackworkspacePage(Result):
    id: str
    content: str


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


method(
    "backworkspace.save",
    params=BackworkspaceSaveParams,
    result=BackworkspaceSaveResult,
    doc="Write a back-workspace page (a new page when no id is given) and return its id.",
)
