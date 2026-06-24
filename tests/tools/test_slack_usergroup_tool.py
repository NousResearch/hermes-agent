import json

import pytest

from tools import slack_usergroup_tool as tool


def test_normalize_users_rejects_empty_membership():
    with pytest.raises(ValueError, match="refusing to empty"):
        tool._normalize_users(" , ,, ")


def test_normalize_users_deduplicates_and_trims():
    assert tool._normalize_users(" U1, U2, U1 ,, U3 ") == ["U1", "U2", "U3"]


def test_update_usergroup_users_posts_complete_normalized_membership(monkeypatch):
    calls = []

    def fake_slack_api(method, params):
        calls.append((method, params))
        return {"ok": True, "usergroup": {"id": params["usergroup"]}}

    monkeypatch.setattr(tool, "_slack_api", fake_slack_api)
    result = json.loads(tool.slack_update_usergroup_users(" S123 ", " U1, U2, U1 "))

    assert calls == [
        ("usergroups.users.update", {"usergroup": "S123", "users": "U1,U2"})
    ]
    assert result["ok"] is True
    assert result["users"] == ["U1", "U2"]


def test_list_usergroups_maps_response(monkeypatch):
    def fake_slack_api(method, params):
        assert method == "usergroups.list"
        assert params == {"include_users": "true", "include_disabled": "false"}
        return {
            "ok": True,
            "usergroups": [
                {
                    "id": "S123",
                    "handle": "jubeon",
                    "name": "주번",
                    "description": "weekly duty",
                    "date_delete": 0,
                    "users": ["U1"],
                }
            ],
        }

    monkeypatch.setattr(tool, "_slack_api", fake_slack_api)
    result = json.loads(tool.slack_list_usergroups(include_users=True))

    assert result["usergroups"][0] == {
        "id": "S123",
        "handle": "jubeon",
        "name": "주번",
        "description": "weekly duty",
        "is_disabled": False,
        "users": ["U1"],
    }
