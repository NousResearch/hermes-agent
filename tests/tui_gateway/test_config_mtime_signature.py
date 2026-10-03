"""config.get key=mtime must expose a change signal a pinned-mtime replacement can't dodge.

``ui-tui``'s poller re-hydrates the general display config (theme, bell, indicator style, ...)
only when this RPC's ``mtime`` field changes. A replacement that preserves mtime (``cp -p``,
``rsync -t``, a dotfile-sync tool) bumped the inode/ctime but left ``mtime`` untouched, so the
poller never re-hydrated non-MCP settings — ``mcp_rev`` doesn't help there, since it's a hash of
MCP-relevant sections only and is unrelated to a theme/bell change. ``_cfg_get_mtime`` now also
returns ``sig`` (mtime+size+inode+ctime, via ``utils.file_signature``), which the pinned-mtime
replacement still moves.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

import tui_gateway.server as server


def _write_cfg(home: Path, value: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"display": {"bell": value}}), encoding="utf-8")


def _get_mtime(monkeypatch, home: Path) -> dict:
    monkeypatch.setattr(server, "_hermes_home", home)
    return server._methods["config.get"]("rid", {"key": "mtime"})["result"]


def test_sig_present_alongside_mtime(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_cfg(home, "on")

    result = _get_mtime(monkeypatch, home)

    assert result["mtime"] > 0
    assert isinstance(result["sig"], str) and result["sig"]


def test_sig_changes_when_a_replacement_pins_mtime(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_cfg(home, "on")
    before = _get_mtime(monkeypatch, home)

    # Simulate cp -p / rsync -t / a dotfile-sync tool: new content, timestamp pinned back
    # to the original value (a fresh write, then utime restores mtime and atime).
    cfg_path = home / "config.yaml"
    pinned_stat = cfg_path.stat()
    _write_cfg(home, "off")
    os.utime(cfg_path, ns=(pinned_stat.st_atime_ns, pinned_stat.st_mtime_ns))

    after = _get_mtime(monkeypatch, home)

    assert after["mtime"] == before["mtime"]  # the replacement's whole premise
    assert after["sig"] != before["sig"]  # but the signature still catches it


def test_sig_stable_when_nothing_changed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_cfg(home, "on")

    first = _get_mtime(monkeypatch, home)
    second = _get_mtime(monkeypatch, home)

    assert first["sig"] == second["sig"]


def test_missing_config_yaml_has_no_sig(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True)

    result = _get_mtime(monkeypatch, home)

    assert result == {"mtime": 0}
