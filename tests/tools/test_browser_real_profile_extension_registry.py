"""The real-profile snapshot must not carry the user's extension registry.

Chrome keeps the extension REGISTRY (not the extension code) in ``Secure Preferences`` and the
keyboard-shortcut bindings in ``Preferences``. The snapshot's ignore list drops the
``Extensions/`` tree, so store extensions never arrive — but developer/unpacked extensions
(``location: 4``) live outside it and are registered by ABSOLUTE PATH into the user's real
filesystem, so a copied registry makes the snapshot browser load the user's own unpacked code
and inherit their global shortcuts (``global: true`` is broadcast per OS user, so a background
instance steals the keystroke from the visible browser).

These tests exercise the real strip function and the real snapshot flow against a synthetic
profile tree — no mocks beyond the filesystem.
"""
import json
import os

import pytest


def _write_json(path, payload):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _read_json(path):
    with open(os.fspath(path), encoding="utf-8") as fh:
        return json.load(fh)


def _profile_with_extensions(root):
    """A user-data-dir whose profile registers a user extension and binds a global shortcut."""
    default = root / "Default"
    default.mkdir(parents=True)
    (root / "Local State").write_text('{"os_crypt": {}}', encoding="utf-8")
    _write_json(default / "Preferences", {
        "extensions": {
            "commands": {
                "mac:Ctrl+Q": {
                    "command_name": "autofill",
                    "extension": "kmcfomidfpdkfieipokbalgegidffkal",
                    "global": True,
                },
            },
            "theme": 1,
        },
        "profile": {"name": "Default"},
    })
    _write_json(default / "Secure Preferences", {
        "extensions": {
            "settings": {
                "kmcfomidfpdkfieipokbalgegidffkal": {
                    "location": 4,
                    "path": "/Users/someone/ext/enpass-ext",
                },
                "mhjfbmdgcfjbbp": {"location": 5, "path": "/opt/chrome/pdf"},
            },
            "ui": {"developer_mode": True},
        },
        "protection": {"macs": {"extensions": {"install": "hash"}}},
        "browser": {"stuff": True},
    })
    return root


class TestStripExtensionRegistry:

    def test_strips_registry_and_global_bindings(self, tmp_path):
        import hermes_cli.browser_connect as bc
        dst = tmp_path / "copy"
        _profile_with_extensions(dst)

        bc._strip_extension_registry(str(dst))

        secure = _read_json(dst / "Default" / "Secure Preferences")
        assert "settings" not in secure["extensions"]
        prefs = _read_json(dst / "Default" / "Preferences")
        assert "commands" not in prefs["extensions"]

    def test_preserves_everything_else(self, tmp_path):
        """Only the registry and the bindings go; the rest of both files must survive intact."""
        import hermes_cli.browser_connect as bc
        dst = tmp_path / "copy"
        _profile_with_extensions(dst)

        bc._strip_extension_registry(str(dst))

        secure = _read_json(dst / "Default" / "Secure Preferences")
        assert secure["protection"]["macs"]["extensions"]["install"] == "hash"
        assert secure["browser"]["stuff"] is True
        assert secure["extensions"]["ui"] == {"developer_mode": True}
        prefs = _read_json(dst / "Default" / "Preferences")
        assert prefs["extensions"]["theme"] == 1
        assert prefs["profile"]["name"] == "Default"

    def test_missing_files_are_not_an_error(self, tmp_path):
        import hermes_cli.browser_connect as bc
        dst = tmp_path / "copy"
        (dst / "Default").mkdir(parents=True)
        bc._strip_extension_registry(str(dst))  # must not raise

    def test_malformed_files_are_not_an_error(self, tmp_path):
        import hermes_cli.browser_connect as bc
        dst = tmp_path / "copy"
        (dst / "Default").mkdir(parents=True)
        (dst / "Default" / "Secure Preferences").write_text("{not json", encoding="utf-8")
        (dst / "Default" / "Preferences").write_text("", encoding="utf-8")
        bc._strip_extension_registry(str(dst))  # must not raise

    def test_idempotent(self, tmp_path):
        import hermes_cli.browser_connect as bc
        dst = tmp_path / "copy"
        _profile_with_extensions(dst)
        bc._strip_extension_registry(str(dst))
        first = (dst / "Default" / "Secure Preferences").read_bytes()
        bc._strip_extension_registry(str(dst))
        assert (dst / "Default" / "Secure Preferences").read_bytes() == first


class TestSnapshotCarriesNoExtensionRegistry:

    def test_fresh_snapshot_has_no_user_extension_registered(self, tmp_path, monkeypatch):
        """A fresh snapshot must not register the user's extensions in the copy."""
        import hermes_cli.browser_connect as bc
        src = _profile_with_extensions(tmp_path / "real")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)

        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        assert dst is not None

        secure = _read_json(os.path.join(dst, "Default", "Secure Preferences"))
        settings = secure.get("extensions", {}).get("settings", {})
        # Chrome's own built-in components (location 5) are written by Chrome itself and are
        # fine; a USER extension (location 4 / 1) must not be registered.
        user_exts = {eid: e for eid, e in settings.items() if e.get("location") != 5}
        assert not user_exts, f"user extension(s) still registered: {user_exts}"

        prefs = _read_json(os.path.join(dst, "Default", "Preferences"))
        commands = prefs.get("extensions", {}).get("commands", {})
        globals_claimed = {k: v for k, v in commands.items() if isinstance(v, dict) and v.get("global")}
        assert not globals_claimed, f"global shortcut(s) still claimed: {globals_claimed}"

    def test_registry_is_stripped_on_every_launch(self, tmp_path, monkeypatch):
        """Preferences is re-synced from the live profile on EVERY launch, so the strip has to
        run every launch too — a one-shot strip would be undone by the next refresh."""
        import hermes_cli.browser_connect as bc
        src = _profile_with_extensions(tmp_path / "real")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)

        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        assert dst is not None

        # Simulate the per-launch re-sync of Preferences from the live profile.
        live_prefs = (src / "Default" / "Preferences").read_text(encoding="utf-8")
        (tmp_path / "hermes-home" / "browser-profile" / "chrome" / "Default"
         / "Preferences").write_text(live_prefs, encoding="utf-8")

        # The next snapshot pass must strip it again.
        bc._strip_extension_registry(dst)
        prefs = _read_json(os.path.join(dst, "Default", "Preferences"))
        assert "commands" not in prefs.get("extensions", {})

    def test_login_state_still_reaches_the_snapshot(self, tmp_path, monkeypatch):
        """The strip must not cost the consented login state: Cookies/Login Data still arrive."""
        import hermes_cli.browser_connect as bc
        import sqlite3
        from contextlib import closing

        src = _profile_with_extensions(tmp_path / "real")
        for name, value in (("Cookies", "sqlite-cookies"), ("Login Data", "sqlite-logins")):
            with closing(sqlite3.connect(src / "Default" / name)) as conn, conn:
                conn.execute("create table if not exists marker(value)")
                conn.execute("delete from marker")
                conn.execute("insert into marker values(?)", (value,))
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)

        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        assert dst is not None

        for name, value in (("Cookies", "sqlite-cookies"), ("Login Data", "sqlite-logins")):
            path = os.path.join(dst, "Default", name)
            assert os.path.exists(path), f"{name} missing from the snapshot"
            with closing(sqlite3.connect(path)) as conn:
                assert conn.execute("select value from marker").fetchone()[0] == value
