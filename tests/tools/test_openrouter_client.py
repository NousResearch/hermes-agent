def test_openrouter_api_key_whitespace_is_unset(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")

    from tools.openrouter_client import check_api_key

    assert check_api_key() is False

def test_openrouter_api_key_valid(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    from tools.openrouter_client import check_api_key

    assert check_api_key() is True
