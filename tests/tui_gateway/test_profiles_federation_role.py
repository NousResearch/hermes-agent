"""Bot Mode receives governed federation identity metadata per profile."""

from __future__ import annotations

import json

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    profile = hermes_home / "profiles" / "librarian"
    profile.mkdir(parents=True)
    (profile / "federation_role.json").write_text(
        json.dumps(
            {
                "schema_name": "hermes_federation_role_v1",
                "role_id": "librarian",
                "display_name": "Federation Librarian",
                "department": "knowledge_commons",
                "authority": "advisory",
                "schedule": "daily",
                "skills": ["llm-wiki"],
                "toolsets": ["file"],
                "handoffs": ["cataloger"],
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def test_profiles_list_exposes_federation_role_identity(home):
    response = srv._methods["profiles.list"](1, {"include_sessions": False})
    rows = response["result"]["profiles"]
    row = next(item for item in rows if item["name"] == "librarian")

    assert row["federation_role"] == {
        "role_id": "librarian",
        "display_name": "Federation Librarian",
        "department": "knowledge_commons",
        "authority": "advisory",
        "schedule": "daily",
        "skills": ["llm-wiki"],
        "toolsets": ["file"],
        "handoffs": ["cataloger"],
    }


def test_profiles_list_ignores_invalid_federation_identity(home):
    profile = home / "profiles" / "librarian" / "federation_role.json"
    profile.write_text("not json")

    response = srv._methods["profiles.list"](1, {"include_sessions": False})
    row = next(item for item in response["result"]["profiles"] if item["name"] == "librarian")

    assert "federation_role" not in row
