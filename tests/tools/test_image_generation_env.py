"""FAL_KEY env var normalization (whitespace-only treated as unset)."""


def test_fal_key_whitespace_is_unset(monkeypatch):
    # Whitespace-only FAL_KEY must NOT register as configured, and the managed
    # gateway fallback must be disabled for this assertion to be meaningful.
    monkeypatch.setenv("FAL_KEY", "   ")

    from tools import image_generation_tool

    monkeypatch.setattr(
        image_generation_tool, "_resolve_managed_fal_gateway", lambda: None
    )

    assert image_generation_tool.check_fal_api_key() is False


def test_fal_key_valid(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "sk-test")

    from tools import image_generation_tool

    monkeypatch.setattr(
        image_generation_tool, "_resolve_managed_fal_gateway", lambda: None
    )

    assert image_generation_tool.check_fal_api_key() is True


def test_fal_key_empty_is_unset(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "")

    from tools import image_generation_tool

    monkeypatch.setattr(
        image_generation_tool, "_resolve_managed_fal_gateway", lambda: None
    )

    assert image_generation_tool.check_fal_api_key() is False
