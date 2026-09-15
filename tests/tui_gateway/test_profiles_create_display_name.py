"""Bot-created profiles persist their presentation identity without changing their canonical id."""

from __future__ import annotations

import hermes_cli.profiles as profiles
import tui_gateway.server as srv
from hermes_cli.web_routers.profiles import _profile_to_dict


def test_profiles_create_persists_display_name_and_lists_canonical_id(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(profiles.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda *_args, **_kwargs: None)

    created = srv._methods["profiles.create"](
        "create",
        {
            "name": "weather-man",
            "display_name": "Weather",
            "mirror_credentials": False,
            "no_alias": True,
            "no_skills": True,
        },
    )["result"]
    rows = srv._methods["profiles.list"]("list", {"include_sessions": False})["result"]["profiles"]
    row = next(item for item in rows if item["name"] == "weather-man")
    configured = srv._methods["profiles.configure"](
        "configure",
        {
            "name": "weather-man",
            "display_name": "Weather Bot",
            "ui_meta": {"hermes-bots": {"title": "Weather Bot"}},
        },
    )["result"]
    info = next(item for item in profiles.list_profiles() if item.name == "weather-man")
    rest_row = _profile_to_dict(info)
    rejected = srv._methods["profiles.configure"](
        "configure",
        {"name": "weather-man", "display_name": "x" * 65},
    )["result"]
    preserved = next(item for item in profiles.list_profiles() if item.name == "weather-man")

    assert created["name"] == "weather-man"
    assert row["display_name"] == "Weather"
    assert configured["applied"]["display_name"] is True
    assert configured["applied"]["ui_meta"] is True
    assert info.display_name == "Weather Bot"
    assert rejected["applied"]["display_name"] is False
    assert preserved.display_name == "Weather Bot"
    assert rest_row["name"] == "weather-man"
    assert rest_row["ui_meta"]["hermes-bots"]["title"] == "Weather Bot"