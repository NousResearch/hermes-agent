"""Regression test for #68053 — hub lock.json with Windows-1252 bytes.

`_read_hub_installed_names()` reads `~/.hermes/skills/.hub/lock.json` with a
strict UTF-8 decode. A hub skill description carrying a Windows-1252 typographic
byte (em-dash `0x97`, smart quotes, bullets) makes `read_text(encoding="utf-8")`
raise `UnicodeDecodeError` — a `ValueError` sibling that is NOT caught by the
function's `except (OSError, json.JSONDecodeError)`, so it escapes and returns
HTTP 500 from the entire `/api/skills` endpoint, blanking the desktop Skills
panel. The fix decodes with `errors="replace"`, so the offending byte degrades
to U+FFFD and every skill name in the (structurally valid) lock stays readable.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import tools.skill_usage as mod
    importlib.reload(mod)
    return home


def _write_hub_lock(home: Path, raw: bytes) -> None:
    hub_dir = home / "skills" / ".hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "lock.json").write_bytes(raw)


def test_windows_1252_em_dash_does_not_raise(skills_home):
    """A 0x97 em-dash byte in a description must not blow up the reader."""
    import tools.skill_usage as mod

    # Valid JSON structure, but the description value contains a raw cp1252
    # em-dash byte (0x97) instead of the UTF-8 sequence.
    raw = (
        b'{"installed": {"supply-chain": '
        b'{"description": "guidance: safe \x97 3 finding(s)"}}}'
    )
    _write_hub_lock(skills_home, raw)

    names = mod._read_hub_installed_names()

    # The skill name is still recovered; no UnicodeDecodeError / 500.
    assert "supply-chain" in names


def test_clean_utf8_lock_still_read(skills_home):
    """A well-formed UTF-8 lock keeps working unchanged."""
    import tools.skill_usage as mod

    raw = '{"installed": {"alpha": {}, "beta": {}}}'.encode("utf-8")
    _write_hub_lock(skills_home, raw)

    names = mod._read_hub_installed_names()

    assert {"alpha", "beta"} <= names
