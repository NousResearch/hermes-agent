"""plugins.manage install (Desktop and Webapp) clones over https:// only; the CLI keeps every scheme."""

import pytest

from tui_gateway import server

REFUSED = ["file:///srv/plugin", "ssh://git@example.com/acme/plugin.git", "git@example.com:acme/plugin.git",
           "http://example.com/acme/plugin.git"]


def _install(**params):
    return server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "install", **params}})


@pytest.mark.parametrize("identifier", REFUSED)
def test_non_https_source_is_refused_before_any_clone(identifier, monkeypatch):
    from hermes_cli import plugins_cmd
    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", lambda *a, **k: pytest.fail("installer ran"))

    resp = _install(identifier=identifier)

    assert "https://" in resp["error"]["message"]


@pytest.mark.parametrize("params", [
    {"identifier": "https://example.com/acme/plugin.git"}, {"identifier": "acme/plugin"},
    # A catalog install clones the catalog's own source, whatever identifier rides along.
    {"identifier": "file:///srv/plugin", "catalog_name": "alpha"},
])
def test_https_shorthand_and_catalog_sources_reach_the_installer(params, monkeypatch):
    from hermes_cli import plugins_cmd
    seen = []
    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin",
                        lambda ident, **k: seen.append(ident) or {"ok": False, "error": "stub"})

    _install(**params)

    assert len(seen) == 1
