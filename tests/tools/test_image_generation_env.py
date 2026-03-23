def test_fal_key_whitespace_is_unset(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "   ")

    from tools.image_generation_tool import check_fal_api_key

    assert check_fal_api_key() is False


def test_fal_key_valid(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "sk-test")

    from tools.image_generation_tool import check_fal_api_key

    assert check_fal_api_key() is True
