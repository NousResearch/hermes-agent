"""A failing import inside the Telegram send must name the module that broke.

``_send_telegram`` performs four imports inside one ``try``: the package
itself (``telegram``), its ``constants`` submodule, and two in-tree modules
(``plugins.platforms.telegram.telegram_ids``, ``gateway.platforms.base``).
The ``except ImportError`` used to answer every one of them with
``python-telegram-bot not installed. Run: hermes pm install --extra telegram``,
so an in-tree import failure was misdiagnosed as a missing dependency. A live
install followed exactly that trail — reinstall, environment rebuild, service
restarts — while the real cause was somewhere else entirely.

These tests pin the contract: the returned error names the module that
actually failed, and the install hint is only offered when ``telegram``
itself is what could not be imported.
"""

from __future__ import annotations

from typing import Any

import asyncio

import pytest


def _missing(module: str) -> ModuleNotFoundError:
    """A ``ModuleNotFoundError`` shaped like the interpreter's own (``.name`` set)."""
    return ModuleNotFoundError(f"No module named {module!r}", name=module)


def _raise_import_error(module: str):
    def _boom(_message: str) -> None:
        raise _missing(module)

    return _boom


class TestSendTelegramImportErrorReport:
    """The error text must identify the import that failed, not the package we assume."""

    def test_in_tree_import_failure_names_that_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing in-tree import is not the package's fault and must not say so."""
        from tools.send_message_tool import _send_telegram

        monkeypatch.setattr(
            "tools.send_message_senders._telegram_format",
            _raise_import_error("gateway.platforms.base"),
        )

        result: dict[str, Any] = asyncio.run(_send_telegram("tok", "123", "hello"))

        assert "gateway.platforms.base" in result["error"], result["error"]
        assert "python-telegram-bot" not in result["error"], (
            "an in-tree import failure must not be blamed on python-telegram-bot"
        )
        assert "hermes pm install --extra telegram" not in result["error"], (
            "the install hint must not be offered for a module that extra cannot provide"
        )

    def test_missing_package_still_offers_the_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``telegram`` itself is missing, the actionable hint stays."""
        from tools.send_message_tool import _send_telegram

        monkeypatch.setattr(
            "tools.send_message_senders._telegram_format",
            _raise_import_error("telegram"),
        )

        result: dict[str, Any] = asyncio.run(_send_telegram("tok", "123", "hello"))

        assert "telegram" in result["error"], result["error"]
        assert "hermes pm install --extra telegram" in result["error"], result["error"]
