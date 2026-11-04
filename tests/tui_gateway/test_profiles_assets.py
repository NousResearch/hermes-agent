"""profiles.get_asset resolves only the asset names profiles.set_asset can write.

The asset name becomes a path segment under ``<profile>/assets/``; get_asset used to take it
unchecked, so a name with path separators resolved to image files outside the profile.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

import tui_gateway.server as server

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def profile_dir(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "hermes_home"
    path = root / "profiles" / "bot"
    path.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return path


def test_get_asset_serves_only_the_profiles_own_named_assets(profile_dir, tmp_path):
    stored = server._methods["profiles.set_asset"](
        1, {"name": "bot", "asset": "avatar", "data": base64.b64encode(_PNG).decode()})
    assert stored["result"]["ok"] is True

    avatar = server._methods["profiles.get_asset"](2, {"name": "bot", "asset": "avatar"})["result"]
    assert avatar["found"] is True and base64.b64decode(avatar["data"].split(",", 1)[1]) == _PNG

    (tmp_path / "elsewhere.png").write_bytes(_PNG)
    escaped = Path("..", "..", "..", "..", "elsewhere").as_posix()
    resp = server._methods["profiles.get_asset"](3, {"name": "bot", "asset": escaped})
    assert "result" not in resp or resp["result"].get("found") is not True
