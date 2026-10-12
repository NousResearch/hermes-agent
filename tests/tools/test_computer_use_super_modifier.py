"""Advertised Super/Win modifiers must stay on the hotkey, not be dropped.

cua-driver accepts super/meta/win/cmd as the same key. It does not accept the
schema token ``windows``. Dropping the modifier and pressing the bare key
reports success while the combo never happened.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.cua_backend_parse import _parse_key_combo


def test_super_number_keeps_the_modifier():
    assert _parse_key_combo("super+3") == ("3", ["super"])


def test_win_aliases_canonicalize_to_the_driver_name():
    assert _parse_key_combo("win+d") == ("d", ["super"])
    assert _parse_key_combo("meta+d") == ("d", ["super"])
    assert _parse_key_combo("windows+d") == ("d", ["super"])


def test_existing_modifiers_are_unchanged():
    assert _parse_key_combo("ctrl-alt-delete") == ("delete", ["ctrl", "option"])
    assert _parse_key_combo("cmd+s") == ("s", ["cmd"])


def test_super_combo_is_a_hotkey_not_a_bare_press():
    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.return_value = {
        "data": "ok",
        "images": [],
        "structuredContent": None,
        "isError": False,
    }
    backend._active_pid = 111
    backend._active_window_id = 222

    assert backend.key("super+3").ok is True
    name, args = backend._session.call_tool.call_args.args
    assert name == "hotkey"
    assert args["keys"] == ["super", "3"]
    assert args["pid"] == 111
    assert args["window_id"] == 222
