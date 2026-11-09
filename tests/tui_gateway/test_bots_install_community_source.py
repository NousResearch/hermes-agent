"""A bot served by a configured community source installs through the same reviewed path."""
from __future__ import annotations

from pathlib import Path

import pytest
import hermes_yaml as yaml

from tests.hermes_cli.test_bot_catalog_sources import FEED_URL, _community
from tui_gateway import server


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  provider: test\n  default: test/model\n"
        f"bots:\n  catalog_sources:\n    - id: hermesmade\n      url: {FEED_URL}\n", encoding="utf-8")
    (home / ".env").write_text("TEST_API_KEY=source-value\n", encoding="utf-8")
    from hermes_cli import bot_catalog
    monkeypatch.setattr(bot_catalog, "fetch_live_bot_catalog", lambda **_: None)
    monkeypatch.setattr(bot_catalog, "_fetch_feed",
                        lambda url, cache, *, force=False: {"entries": [_community()]} if url == FEED_URL else None)
    return home


def _call(method, **params):
    return server._methods[method](7, params)


def test_community_bot_installs_with_source_provenance_and_no_credentials(home):
    catalog = _call("bots.catalog")["result"]
    listed = {entry["name"]: entry for entry in catalog["entries"]}
    assert listed["fact-check-desk"]["source"] == "hermesmade"
    assert listed["fact-check-desk"]["tier"] == "community"

    result = _call("bots.install", catalog_name="fact-check-desk", name="fact-checker",
                   source_profile="default")["result"]
    assert result["committed"] is True
    assert result["catalog_source"] == "hermesmade"
    assert result["copied_credentials"] == []

    profile = home / "profiles" / "fact-checker"
    assert not (profile / ".env").exists() or "source-value" not in (profile / ".env").read_text()
    meta = yaml.safe_load((profile / "profile.yaml").read_text())
    assert meta["provenance"]["catalog_source"] == "hermesmade"
    assert all(r["state"] == "paused" for r in meta["routines"])
    blueprint = yaml.safe_load((profile / "bot-blueprint.yaml").read_text())
    assert blueprint["catalog_entry"]["source"] == "hermesmade"
    # Exactly the declared toolsets: no platform composite (terminal, file, code, delegation).
    config = yaml.safe_load((profile / "config.yaml").read_text())
    for platform in ("cli", "desktop", "cron"):
        assert config["platform_toolsets"][platform] == ["web"], platform
    # Only the reviewed, client-shipped skill was copied in.
    assert (profile / "skills" / "research" / "grounded-citations" / "SKILL.md").is_file()


def test_install_refuses_a_name_no_configured_source_serves(home, monkeypatch):
    monkeypatch.setattr("hermes_cli.bot_catalog.configured_catalog_sources", lambda: [])
    envelope = _call("bots.install", catalog_name="fact-check-desk", name="fact-checker",
                     source_profile="default")
    assert envelope["error"]["code"] == 4071
    assert not (home / "profiles" / "fact-checker").exists()
