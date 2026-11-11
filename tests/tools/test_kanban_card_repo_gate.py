"""kanban_create refuses cards that span several GitHub repos (one card = one PR)."""
import pytest

from tools.kanban_tools import _card_repos


@pytest.fixture(autouse=True)
def repository_policy(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {"single_repo_owners": ["example-owner"]}})



@pytest.mark.parametrize("text,expected", [
    ("Fix example-owner/sample#944", ["example-owner/sample"]),
    ("https://github.com/example-owner/sample-docs/pull/23 and example-owner/sample",
     ["example-owner/sample", "example-owner/sample-docs"]),
    ("clone https://github.com/example-owner/garage.git", ["example-owner/garage"]),
    ("work in ~/code/example-owner/foo", []),
    ("upstream NousResearch/hermes-agent", []),
    ("", []),
])
def test_card_repos(text, expected):
    assert _card_repos(text) == expected


def test_title_and_body_are_combined():
    assert len(_card_repos("example-owner/a docs", "also example-owner/b")) == 2


def test_policy_is_disabled_unless_configured(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert _card_repos("example-owner/a example-owner/b") == []
