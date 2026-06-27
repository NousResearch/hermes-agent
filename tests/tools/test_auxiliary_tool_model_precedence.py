import os
from unittest.mock import AsyncMock, MagicMock, patch


def test_browser_vision_config_model_beats_env_override():
    from tools import browser_tool

    with (
        patch("hermes_cli.config.read_raw_config", return_value={
            "auxiliary": {"vision": {"model": "config-vision"}}
        }),
        patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "stale-env"}, clear=False),
    ):
        assert browser_tool._get_vision_model() is None


def test_browser_vision_env_model_remains_fallback_without_config_model():
    from tools import browser_tool

    with (
        patch("hermes_cli.config.read_raw_config", return_value={"auxiliary": {"vision": {}}}),
        patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "env-vision"}, clear=False),
    ):
        assert browser_tool._get_vision_model() == "env-vision"


def test_browser_extraction_config_model_beats_env_override():
    from tools import browser_tool

    with (
        patch("hermes_cli.config.read_raw_config", return_value={
            "auxiliary": {"web_extract": {"model": "config-web"}}
        }),
        patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "stale-env"}, clear=False),
    ):
        assert browser_tool._get_extraction_model() is None


def test_browser_extraction_env_model_remains_fallback_without_config_model():
    from tools import browser_tool

    with (
        patch("hermes_cli.config.read_raw_config", return_value={"auxiliary": {"web_extract": {}}}),
        patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "env-web"}, clear=False),
    ):
        assert browser_tool._get_extraction_model() == "env-web"


def test_vision_handler_config_model_beats_env_override():
    from tools import vision_tools

    with (
        patch("hermes_cli.config.read_raw_config", return_value={
            "auxiliary": {"vision": {"model": "config-vision"}}
        }),
        patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "stale-env"}, clear=False),
        patch("tools.vision_tools._should_use_native_vision_fast_path", return_value=False),
        patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock) as mock_tool,
    ):
        coro = vision_tools._handle_vision_analyze({
            "image_url": "https://example.com/img.png",
            "question": "describe",
        })
        coro.close()

    assert mock_tool.call_args.args[2] is None


def test_vision_handler_env_model_remains_fallback_without_config_model():
    from tools import vision_tools

    with (
        patch("hermes_cli.config.read_raw_config", return_value={"auxiliary": {"vision": {}}}),
        patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "env-vision"}, clear=False),
        patch("tools.vision_tools._should_use_native_vision_fast_path", return_value=False),
        patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock) as mock_tool,
    ):
        coro = vision_tools._handle_vision_analyze({
            "image_url": "https://example.com/img.png",
            "question": "describe",
        })
        coro.close()

    assert mock_tool.call_args.args[2] == "env-vision"


def test_web_extract_config_model_beats_env_override():
    from tools import web_tools

    client = MagicMock(base_url="https://api.openrouter.ai/v1")
    with (
        patch("tools.web_tools.get_async_text_auxiliary_client", return_value=(client, "config-web")),
        patch("hermes_cli.config.read_raw_config", return_value={
            "auxiliary": {"web_extract": {"model": "config-web"}}
        }),
        patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "stale-env"}, clear=False),
    ):
        _, model, _ = web_tools._resolve_web_extract_auxiliary()

    assert model == "config-web"


def test_web_extract_env_model_remains_fallback_without_config_model():
    from tools import web_tools

    client = MagicMock(base_url="https://api.openrouter.ai/v1")
    with (
        patch("tools.web_tools.get_async_text_auxiliary_client", return_value=(client, "default-web")),
        patch("hermes_cli.config.read_raw_config", return_value={"auxiliary": {"web_extract": {}}}),
        patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "env-web"}, clear=False),
    ):
        _, model, _ = web_tools._resolve_web_extract_auxiliary()

    assert model == "env-web"


def test_web_extract_explicit_model_still_wins_over_config_and_env():
    from tools import web_tools

    client = MagicMock(base_url="https://api.openrouter.ai/v1")
    with (
        patch("tools.web_tools.get_async_text_auxiliary_client", return_value=(client, "config-web")),
        patch("hermes_cli.config.read_raw_config", return_value={
            "auxiliary": {"web_extract": {"model": "config-web"}}
        }),
        patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "stale-env"}, clear=False),
    ):
        _, model, _ = web_tools._resolve_web_extract_auxiliary("explicit-web")

    assert model == "explicit-web"
