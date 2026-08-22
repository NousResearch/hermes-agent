"""Regression for #91587: _save_cfg must write the same home _load_cfg_raw reads.

_load_cfg_raw honors get_hermes_home_override(); _save_cfg used to ignore it
and dump into the launch-profile _hermes_home. That breaks the documented
read→mutate→_save_cfg pair whenever a session override is active.
"""

from __future__ import annotations

import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server


def test_save_cfg_writes_override_home_not_launch_home(tmp_path, monkeypatch):
    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "profile"
    launch_home.mkdir()
    profile_home.mkdir()
    (launch_home / "config.yaml").write_text("model:\n  default: launch-model\n", encoding="utf-8")
    (profile_home / "config.yaml").write_text("model:\n  default: profile-model\n", encoding="utf-8")

    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_mtime", None)
    monkeypatch.setattr(server, "_cfg_path", None)

    token = set_hermes_home_override(str(profile_home))
    try:
        raw = server._load_cfg_raw()
        assert raw["model"]["default"] == "profile-model"
        raw["model"]["default"] = "updated-profile"
        server._save_cfg(raw)
    finally:
        reset_hermes_home_override(token)

    profile = yaml.safe_load((profile_home / "config.yaml").read_text(encoding="utf-8"))
    launch = yaml.safe_load((launch_home / "config.yaml").read_text(encoding="utf-8"))
    assert profile["model"]["default"] == "updated-profile"
    assert launch["model"]["default"] == "launch-model"


def test_cfg_home_and_watcher_home_share_override(tmp_path, monkeypatch):
    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "profile"
    launch_home.mkdir()
    profile_home.mkdir()
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    token = set_hermes_home_override(str(profile_home))
    try:
        assert server._cfg_home() == profile_home
        assert server._watcher_home() == profile_home
    finally:
        reset_hermes_home_override(token)
    assert server._cfg_home() == launch_home
    assert server._watcher_home() == launch_home
