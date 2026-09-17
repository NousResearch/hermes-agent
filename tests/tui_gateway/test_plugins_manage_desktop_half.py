"""``plugins.manage`` ``desktop_half`` — the Desktop UI half's BYTES over the authenticated channel.

Why this action exists: the desktop app evaluates a Desktop half with the app's own authority, so a
client that is not on this machine (SSH / URL backend) must not fetch it from the dashboard's static
asset route, which is deliberately unauthenticated because the SPA loads plugin JS with ``<script src>``.
These arms pin the contract the app codes against: ``name``/``key`` in, ``{name, key, source, sha256,
bytes, text}`` out, with every refusal named.
"""

import hashlib
from unittest.mock import patch

from tui_gateway import server


def _with_plugins(tmp_path, rows):
    unified = tmp_path / "media"
    (unified / "desktop").mkdir(parents=True)
    (unified / "desktop" / "plugin.js").write_text(rows["desktop_half_text"], encoding="utf-8")
    plain = tmp_path / "snap"
    plain.mkdir()
    return [
        ("media", "1.0", "Media", "user", unified, "media"),
        ("snap", "1.0", "Snap", "user", plain, "snap"),
    ]


def _request(params):
    return server.handle_request({"id": "1", "method": "plugins.manage", "params": params})


def test_desktop_half_returns_text_and_digest(tmp_path):
    text = "export default { id: 'media', register() {} }\n"
    rows = _with_plugins(tmp_path, {"desktop_half_text": text})
    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=rows), \
         patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_pins", return_value={}), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_versions", return_value={}):
        resp = _request({"action": "desktop_half", "name": "media"})

    result = resp["result"]
    assert result["text"] == text
    assert result["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert result["bytes"] == len(text.encode("utf-8"))
    assert result["name"] == "media" and result["key"] == "media"
    assert result["source"] == "user"


def test_desktop_half_accepts_the_registry_key(tmp_path):
    """The desktop pairs rows by key OR name (names collide across category dirs)."""
    text = "export default { id: 'media' }\n"
    rows = _with_plugins(tmp_path, {"desktop_half_text": text})
    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=rows), \
         patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_pins", return_value={}), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_versions", return_value={}):
        resp = _request({"action": "desktop_half", "key": "media"})

    assert resp["result"]["text"] == text


def test_desktop_half_refuses_a_plugin_without_one(tmp_path):
    """``has_desktop_half: false`` in ``list`` must be a NAMED refusal here, never an empty payload."""
    rows = _with_plugins(tmp_path, {"desktop_half_text": "export default {}\n"})
    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=rows), \
         patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_pins", return_value={}), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_versions", return_value={}):
        resp = _request({"action": "desktop_half", "name": "snap"})

    assert "result" not in resp
    assert "ships no desktop half" in resp["error"]["message"]


def test_desktop_half_refuses_an_unknown_plugin():
    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=[]), \
         patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_pins", return_value={}), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_versions", return_value={}):
        resp = _request({"action": "desktop_half", "name": "nope"})

    assert "no plugin 'nope'" in resp["error"]["message"]


def test_desktop_half_requires_an_identifier():
    resp = _request({"action": "desktop_half"})

    assert "requires a 'key' or 'name'" in resp["error"]["message"]
