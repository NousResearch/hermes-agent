"""Provide basic warnings used by setuptools modules.

Using custom classes (other than ``UserWarning``) allow users to set
``PYTHONWARNINGS`` filters to run tests and prepare for upcoming changes in
setuptools.
"""

from __future__ import annotations

import os
import warnings
from datetime import date
from inspect import cleandoc
from textwrap import indent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

_DueDate: TypeAlias = tuple[int, int, int]  # time tuple
_INDENT = 8 * " "
_TEMPLATE = f"""{80 * '*'}\n{{details}}\n{80 * '*'}"""


class SetuptoolsWarning(UserWarning):
    """Base class in ``setuptools`` warning hierarchy."""

    @classmethod
    def emit(
        cls,
        summary: str | None = None,
        details: str | None = None,
        due_date: _DueDate | None = None,
        see_docs: str | None = None,
        see_url: str | None = None,
        stacklevel: int = 2,
        **kwargs,
    ) -> None:
        """Private: reserved for ``setuptools`` internal use only"""
        # Default values:
        summary_ = summary or getattr(cls, "_SUMMARY", None) or ""
        details_ = details or getattr(cls, "_DETAILS", None) or ""
        due_date = due_date or getattr(cls, "_DUE_DATE", None)
        docs_ref = see_docs or getattr(cls, "_SEE_DOCS", None)
        docs_url = docs_ref and f"https://setuptools.pypa.io/en/latest/{docs_ref}"
        see_url = see_url or getattr(cls, "_SEE_URL", None)
        due = date(*due_date) if due_date else None

        text = cls._format(summary_, details_, due, see_url or docs_url, kwargs)
        if due and due < date.today() and _should_enforce():
            raise cls(text)
        warnings.warn(text, cls, stacklevel=stacklevel + 1)

    @classmethod
    def _format(
        cls,
        summary: str,
        details: str,
        due_date: date | None = None,
        see_url: str | None = None,
        format_args: dict | None = None,
    ) -> str:
        """Private: reserved for ``setuptools`` internal use only"""
        today = date.today()
        summary = cleandoc(summary).format_map(format_args or {})
        possible_parts = [
            cleandoc(details).format_map(format_args or {}),
            (
                f"\nBy {due_date:%Y-%b-%d}, you need to update your project and remove "
                "deprecated calls\nor your builds will no longer be supported."
                if due_date and due_date > today
                else None
            ),
            (
                "\nThis deprecation is overdue, please update your project and remove "
                "deprecated\ncalls to avoid build errors in the future."
                if due_date and due_date < today
                else None
            ),
            (f"\nSee {see_url} for details." if see_url else None),
        ]
        parts = [x for x in possible_parts if x]
        if parts:
            body = indent(_TEMPLATE.format(details="\n".join(parts)), _INDENT)
            return "\n".join([summary, "!!\n", body, "\n!!"])
        return summary


class InformationOnly(SetuptoolsWarning):
    """Currently there is no clear way of displaying messages to the users
    that use the setuptools backend directly via ``pip``.
    The only thing that might work is a warning, although it is not the
    most appropriate tool for the job...

    See pypa/packaging-problems#558.
    """


class SetuptoolsDeprecationWarning(SetuptoolsWarning):
    """
    Base class for warning deprecations in ``setuptools``

    This class is not derived from ``DeprecationWarning``, and as such is
    visible by default.
    """


def _should_enforce():
    enforce = os.getenv("SETUPTOOLS_ENFORCE_DEPRECATION", "false").lower()
    return enforce in ("true", "on", "ok", "1")
