"""``plugins.catalog.lookup``: the deep link's catalog lookup, resolved by the backend (hermes-agent 136483)."""

import pytest

from hermes_cli import plugin_catalog as pc
from tui_gateway import server

SHA = "a" * 40


def _entry(name, repo=None, subdir="", tier="community"):
    return pc.entry_from_mapping({"name": name, "repo": repo or f"https://github.com/fx/{name}", "sha": SHA,
                                  "description": "d.", "maintainer": "fx", "subdir": subdir, "tier": tier}, name)


def _lookup(name, **extra):
    resp = server.handle_request({"id": "1", "method": "plugins.catalog.lookup", "params": {"name": name, **extra}})
    assert "error" not in resp, resp
    return resp["result"]


@pytest.fixture
def catalog(monkeypatch):
    entries = [_entry("weather"), _entry("aihubmix", "https://github.com/AIhubmix/mono", subdir="aihubmix", tier="official"),
               _entry("pulled")]
    monkeypatch.setattr(pc, "load_catalog_live", lambda: entries)
    monkeypatch.setattr(pc, "load_removed_list", lambda catalog_dir=None: [pc.RemovedEntry(name="pulled")])
    monkeypatch.setattr(pc, "live_removed_list", list)
    return entries


def test_a_listed_name_returns_what_the_install_dialog_needs(catalog):
    assert _lookup("aihubmix") == {"status": "found", "entry": {
        "name": "aihubmix", "repo": "https://github.com/AIhubmix/mono", "sha": SHA, "subdir": "aihubmix",
        "tier": "official"}}
    # The tier rides along so a deep link's dialog badges Official/Verified/Community like an in-app pick.
    assert _lookup(" weather ")["entry"]["subdir"] == ""
    assert _lookup("weather")["entry"]["tier"] == "community"


def test_an_unlisted_or_removed_name_is_unknown_never_a_guess(catalog):
    assert _lookup("weather-evil") == {"status": "unknown"}
    assert _lookup("pulled") == {"status": "unknown"}  # listed but on the kill list


@pytest.mark.parametrize("name", ["", "../etc", "https://github.com/x/y", "Weather", "a" * 65])
def test_anything_but_a_catalog_name_is_invalid(catalog, name):
    assert _lookup(name) == {"status": "invalid_name"}


def test_an_unreadable_catalog_is_unavailable(monkeypatch):
    monkeypatch.setattr(pc, "load_catalog_live", list)  # no live doc, no cache, no in-tree entries
    assert _lookup("weather") == {"status": "unavailable"}


def test_the_lookup_reads_the_backend_catalog_not_the_docs_redirect(monkeypatch, tmp_path):
    """The live fetch follows the docs host's redirect server-side (httpx, follow_redirects), where CORS
    does not apply; with the network down the in-tree catalog still answers."""
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda force=False: None)
    first = pc.load_catalog()[0]
    assert _lookup(first.name)["entry"]["sha"] == first.sha
