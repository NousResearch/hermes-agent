"""LobeHub Skills Hub fetch uses guarded HTTP + agent-id sanitization."""

from unittest.mock import MagicMock, patch

from tools.skills_hub import LobeHubSource


def test_lobehub_sanitize_rejects_traversal():
    src = LobeHubSource()
    assert src._sanitize_agent_id("../etc/passwd") is None
    assert src._sanitize_agent_id("http://evil.example/x") is None
    assert src._sanitize_agent_id("a/b") is None
    assert src._sanitize_agent_id("safe-agent_1") == "safe-agent_1"


def test_lobehub_fetch_index_uses_guarded_http():
    src = LobeHubSource()
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {"agents": []}
    with patch("tools.skills_hub._read_index_cache", return_value=None), patch(
        "tools.skills_hub._write_index_cache"
    ), patch("tools.skills_hub._guarded_http_get", return_value=fake) as guarded:
        assert src._fetch_index() == {"agents": []}
        guarded.assert_called_once_with(src.INDEX_URL, timeout=30)


def test_lobehub_fetch_agent_rejects_unsafe_id():
    src = LobeHubSource()
    with patch("tools.skills_hub._guarded_http_get") as guarded:
        assert src._fetch_agent("../../x") is None
        guarded.assert_not_called()
