"""``i18n.languages`` lists bundled + pack languages; ``i18n.catalog`` returns the pack/overlay layer per
surface and nothing bundled; both answer for the requested profile's home."""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

from agent import i18n, i18n_layers


@pytest.fixture
def clean_layers():
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()
    yield
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


def _call(server, method, params):
    return server.handle_request({"jsonrpc": "2.0", "id": 3, "method": method, "params": params})


def test_languages_and_catalog_serve_pack_layers_per_surface(tmp_path, monkeypatch, clean_layers):
    from tui_gateway import server
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest

    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    (home / "locales" / "de.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Overlay-DE"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    i18n.reset_language_cache()

    ctx = PluginContext(PluginManifest(name="hermes-lang-pl", source="user", path=str(home)), PluginManager())
    ctx.register_locale("pl", {"approval": {"denied": "Odrzucono"}}, endonym="Polski")
    ctx.register_locale("pl", {"status": {"ready": "Gotowy"}}, surface="tui")
    ctx.register_locale("pl", {"settings": {"title": "Ustawienia"}}, surface="desktop", rtl=False)

    languages = _call(server, "i18n.languages", {})["result"]["languages"]
    ids = [row["id"] for row in languages]
    assert ids == ["en", *sorted(ids[1:])]
    assert {"id": "pl", "endonym": "Polski", "rtl": False, "source": "plugin:hermes-lang-pl"} in languages
    assert {"id": "ar", "endonym": "العربية", "rtl": True, "source": "bundled"} in languages

    core = _call(server, "i18n.catalog", {"lang": "pl"})["result"]
    assert core == {"lang": "pl", "surface": "core", "messages": {"approval.denied": "Odrzucono"}}
    tui = _call(server, "i18n.catalog", {"lang": "PL", "surface": "tui"})["result"]
    assert tui == {"lang": "pl", "surface": "tui", "messages": {"status.ready": "Gotowy"}}
    desktop = _call(server, "i18n.catalog", {"lang": "pl", "surface": "desktop"})["result"]
    assert desktop["messages"] == {"settings.title": "Ustawienia"}
    # The overlay layer is served too; bundled German stays out of the wire payload.
    de = _call(server, "i18n.catalog", {"lang": "de"})["result"]
    assert de["messages"] == {"approval.denied": "Overlay-DE"}
    # Unknown ids resolve to en with an empty layer; a bad surface is a client error.
    assert _call(server, "i18n.catalog", {"lang": "xx-nope"})["result"] == {"lang": "en", "surface": "core", "messages": {}}
    assert _call(server, "i18n.catalog", {"lang": "pl", "surface": "web"})["error"]["code"] in (4000, 4002)


def test_catalog_answers_for_the_requested_profile_home(tmp_path, monkeypatch, clean_layers):
    """Two on-disk homes; only the named profile overlays de. The method must read that home's overlay
    and leave the launch home's (empty) view intact."""
    from tui_gateway import server

    launch, named = tmp_path / "launch", tmp_path / "profiles" / "named"
    (launch / "locales").mkdir(parents=True)
    (named / "locales").mkdir(parents=True)
    (named / "locales" / "de.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Nur named"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_profile_home", lambda name: str(named) if name == "named" else None)
    i18n.reset_language_cache()

    assert _call(server, "i18n.catalog", {"lang": "de", "profile": "named"})["result"]["messages"] == {"approval.denied": "Nur named"}
    assert _call(server, "i18n.catalog", {"lang": "de"})["result"]["messages"] == {}
