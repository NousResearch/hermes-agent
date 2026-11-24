from unittest.mock import MagicMock, patch

from tools.skills_hub_sources import WellKnownSkillSource


@patch("tools.skills_hub._write_index_cache")
@patch("tools.skills_hub._read_index_cache", return_value=None)
@patch("tools.skills_hub._ssrf_safe_http_get")
def test_identifiers_from_an_index_outside_base_path_resolve_again(
    mock_get, _mock_read_cache, _mock_write_cache, monkeypatch
):
    monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
    monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)
    index_url = "https://example.com/.well-known/agent-skills/index.json"

    def fake_get(url, *args, **kwargs):
        if url == index_url:
            return MagicMock(status_code=200, json=lambda: {
                "skills": [{"name": "code-review", "description": "Review code"}]
            })
        if url == "https://example.com/.well-known/agent-skills/code-review/SKILL.md":
            return MagicMock(status_code=200, text="# Code Review\n")
        raise AssertionError(url)

    mock_get.side_effect = fake_get
    source = WellKnownSkillSource()

    [listed] = source.search(index_url, limit=10)
    installed = source.fetch(listed.identifier)
    # An update check fetches by the identifier recorded at install.
    rechecked = source.fetch(installed.identifier)

    assert listed.identifier == f"well-known:{index_url}#code-review"
    assert installed.identifier == listed.identifier
    assert rechecked.files == {"SKILL.md": "# Code Review\n"}
