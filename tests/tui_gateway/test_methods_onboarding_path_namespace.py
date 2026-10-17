"""Onboarding split handlers must bind Path in their own module (#138516).

``_shared_nous_store`` annotates ``tuple[Path, ...]`` and the due-stamp /
unfinished-run helpers construct ``Path(...)``, but the split module never
imported ``pathlib.Path`` and has no ``from __future__ import annotations``.

On Python 3.11–3.13 the return annotation is evaluated at ``def`` time, so
``import tui_gateway.methods_onboarding`` raises ``NameError``. ``server.py``
imports that module while splitting handlers, so the TUI / Desktop backend
fails to load. Python 3.14 defers annotation evaluation (PEP 649), which
hides the crash in some CI legs.

``typing.get_type_hints`` still evaluates the annotation against the module
globals on every supported version. Body-level ``Path(...)`` is rebound onto
``server.py`` after ``bind_module``, but Path is a real runtime dependency of
this module and must not exist only in the server namespace.
"""

from pathlib import Path
from typing import get_type_hints

from tui_gateway import methods_onboarding


def test_shared_nous_store_return_annotation_resolves_to_path_tuple() -> None:
    """Eager-annotation Pythons crash at import; get_type_hints catches 3.14 too."""
    hints = get_type_hints(methods_onboarding._shared_nous_store)
    assert hints["return"] == tuple[Path, ...]


def test_onboarding_handlers_bind_path_without_server_rebind() -> None:
    """Body-level Path(...) must resolve before bind_module copies server globals."""
    assert methods_onboarding.__dict__.get("Path") is Path
