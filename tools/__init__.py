#!/usr/bin/env python3
"""Tools package namespace.

Keep package import side effects minimal. Importing ``tools`` should not
eagerly import the full tool stack, because several subsystems load tools while
``hermes_cli.config`` is still initializing.

Callers should import concrete submodules directly, for example:

    import tools.web_tools
    from tools import browser_tool

Python will resolve those submodules via the package path without needing them
to be re-exported here.
"""

import importlib


def check_file_requirements():
    """File tools only require terminal backend availability."""
    from .terminal_tool import check_terminal_requirements

    return check_terminal_requirements()


def __getattr__(name: str):
    """Lazily resolve tools submodules for backward compatibility.

    Many call sites (and tests) reference attributes like ``tools.vision_tools``
    and ``tools.terminal_tool``. We keep import side effects minimal by loading
    submodules on demand instead of importing the entire stack at package import.
    """
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    globals()[name] = module
    return module


__all__ = ["check_file_requirements"]
