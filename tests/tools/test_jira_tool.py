import json
from unittest.mock import patch

import pytest

from tools.jira_tool import extract_issue_key, jira_get_issue, normalize_issue


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("CPG-1489", "CPG-1489"),
        ("Please do https://flash-ai.atlassian.net/browse/CPG-1489", "CPG-1489"),
        ("https://flash-ai.atlassian.net/jira/software/c/projects/CPG/issues/CPG-1489", "CPG-1489"),
    ],
)
def test_extract_issue_key_accepts_key_and_urls(reference, expected):
    assert extract_issue_key(reference) == expected


def test_jira_get_issue_fetches_by_url_with_basic_auth(monkeypatch):
    monkeypatch.setenv("JIRA_EMAIL", "dev@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "secret-token")
    monkeypatch.delenv("JIRA_SITE_URL", raising=False)
    monkeypatch.delenv("ATLASSIAN_SITE_URL", raising=False)
    monkeypatch.delenv("JIRA_BEARER_TOKEN", raising=False)

    jira_payload = {
        "key": "CPG-1489",
        "names": {"customfield_10010": "Acceptance Criteria"},
        "fields": {
            "summary": "Hermes agent: complete Jira tickets from URL/key",
            "description": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Implement ticket-driven orchestration."}],
                    }
                ],
            },
            "labels": ["automation"],
            "components": [{"name": "agent"}],
            "status": {"name": "To Do"},
            "issuetype": {"name": "Story"},
            "priority": {"name": "High"},
            "assignee": {"displayName": "Devin"},
            "customfield_10010": {
                "type": "doc",
                "content": [
                    {
                        "type": "bulletList",
                        "content": [
                            {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Accept URL"}]}]},
                            {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Accept key"}]}]},
                        ],
                    }
                ],
            },
        },
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(jira_payload).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("tools.jira_tool.urllib.request.urlopen", side_effect=fake_urlopen):
        result = json.loads(jira_get_issue("https://flash-ai.atlassian.net/browse/CPG-1489"))

    assert result["success"] is True
    assert result["key"] == "CPG-1489"
    assert result["url"] == "https://flash-ai.atlassian.net/browse/CPG-1489"
    assert result["summary"] == "Hermes agent: complete Jira tickets from URL/key"
    assert result["description"] == "Implement ticket-driven orchestration."
    assert result["labels"] == ["automation"]
    assert result["components"] == ["agent"]
    assert result["custom_fields"] == {"Acceptance Criteria": "- Accept URL\n- Accept key"}
    assert captured["url"].startswith("https://flash-ai.atlassian.net/rest/api/3/issue/CPG-1489?")
    assert captured["auth"].startswith("Basic ")
    assert captured["timeout"] == 30


def test_jira_get_issue_requires_site_for_bare_key(monkeypatch):
    monkeypatch.setenv("JIRA_EMAIL", "dev@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "secret-token")
    monkeypatch.delenv("JIRA_SITE_URL", raising=False)
    monkeypatch.delenv("ATLASSIAN_SITE_URL", raising=False)
    monkeypatch.delenv("JIRA_BEARER_TOKEN", raising=False)

    result = json.loads(jira_get_issue("CPG-1489"))

    assert result["success"] is False
    assert "JIRA_SITE_URL" in result["error"]


def test_normalize_issue_includes_links_and_subtasks():
    issue = {
        "key": "CPG-1489",
        "fields": {
            "summary": "Parent work",
            "issuelinks": [
                {
                    "type": {"name": "Blocks"},
                    "outwardIssue": {
                        "key": "CPG-1490",
                        "fields": {"summary": "Child work", "status": {"name": "Open"}},
                    },
                }
            ],
            "subtasks": [
                {"key": "CPG-1491", "fields": {"summary": "Subtask", "status": {"name": "Done"}}}
            ],
        },
    }

    normalized = normalize_issue(issue, site_url="https://example.atlassian.net")

    assert normalized["issue_links"] == [
        {"type": "Blocks", "direction": "outward", "key": "CPG-1490", "summary": "Child work", "status": "Open"}
    ]
    assert normalized["subtasks"] == [{"key": "CPG-1491", "summary": "Subtask", "status": "Done"}]
