"""Edge proxy routing contracts, with no network or optional SDK installation."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize(
    "edge_config,env,expected",
    [
        ({}, {}, None),
        ({}, {"HTTPS_PROXY": "http://secure.example:8080"}, "http://secure.example:8080"),
        ({}, {"https_proxy": "http://lower.example:8080"}, "http://lower.example:8080"),
        ({}, {"HTTP_PROXY": "http://http.example:8080"}, "http://http.example:8080"),
        ({}, {"ALL_PROXY": "http://all.example:8080"}, "http://all.example:8080"),
        ({}, {"http_proxy": "http://lower.example"}, "http://lower.example"),
        ({}, {"all_proxy": "https://lower.example"}, "https://lower.example"),
        ({}, {"HTTPS_PROXY": "http://first.example", "HTTP_PROXY": "http://second.example",
              "ALL_PROXY": "http://third.example"}, "http://first.example"),
        ({}, {"HTTP_PROXY": "http://second.example", "ALL_PROXY": "http://third.example"},
         "http://second.example"),
        ({}, {"HTTPS_PROXY": "http://upper.example", "https_proxy": "http://lower.example"},
         "http://upper.example"),
        ({}, {"HTTPS_PROXY": "http://proxy.example", "NO_PROXY": "speech.platform.bing.com"}, None),
        ({}, {"HTTPS_PROXY": "http://proxy.example", "no_proxy": "*.bing.com"}, None),
        ({}, {"HTTPS_PROXY": "http://proxy.example", "NO_PROXY": "speech.platform.bing.com:443"}, None),
        ({}, {"HTTPS_PROXY": "http://proxy.example", "NO_PROXY": "other.example"}, "http://proxy.example"),
        ({"proxy": "https://override.example:8443"}, {"HTTPS_PROXY": "http://env.example",
                                                      "NO_PROXY": "*"}, "https://override.example:8443"),
        ({"proxy": " http://user:secret@proxy.example:8080 "}, {}, "http://user:secret@proxy.example:8080"),
        ({"proxy": ""}, {"HTTPS_PROXY": "http://env.example"}, None),
        ({"proxy": "   "}, {"HTTPS_PROXY": "http://env.example"}, None),
        ({"proxy": "socks5://user:secret@proxy.example:1080"}, {}, ValueError),
        ({"proxy": "proxy.example:8080"}, {}, ValueError),
        ({"proxy": "http://user:secret@proxy.example:bad"}, {}, ValueError),
        ({"proxy": "http://user:secret@proxy.example:99999"}, {}, ValueError),
        ({"proxy": "http://"}, {}, ValueError),
        ({"proxy": "http://user:secret@[broken"}, {}, ValueError),
        ({"proxy": False}, {}, ValueError),
        ({"proxy": None}, {}, ValueError),
        ({}, {"HTTPS_PROXY": "socks5://user:secret@proxy.example:1080"}, ValueError),
    ],
)
def test_edge_proxy_routing(edge_config, env, expected, tmp_path, monkeypatch):
    from agent.proxy_bypass import PROXY_ENV_KEYS
    from tools import tts_tool

    for key in (*PROXY_ENV_KEYS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    client = Mock(return_value=SimpleNamespace(save=AsyncMock()))
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: SimpleNamespace(Communicate=client))
    output = str(tmp_path / "speech.mp3")
    config = {"speed": 1.1, "edge": {"voice": "en-GB-SoniaNeural", "speed": 1.25, **edge_config}}

    if expected is ValueError:
        with pytest.raises(ValueError, match="Edge TTS proxy") as error:
            asyncio.run(tts_tool._generate_edge_tts("Hello", output, config))
        assert "secret" not in str(error.value)
        assert "user" not in str(error.value)
        client.assert_not_called()
    else:
        assert asyncio.run(tts_tool._generate_edge_tts("Hello", output, config)) == output
        kwargs = {"voice": "en-GB-SoniaNeural", "rate": "+25%"}
        if expected is not None:
            kwargs["proxy"] = expected
        client.assert_called_once_with("Hello", **kwargs)
        client.return_value.save.assert_awaited_once_with(output)


@pytest.mark.parametrize("proxy", ["http://override.example:8080", "", "socks5://user:secret@proxy.example:1080"])
def test_edge_proxy_config_reaches_registered_handler(proxy, tmp_path, monkeypatch, caplog):
    import hermes_yaml
    from hermes_constants import get_hermes_home
    from tools import tts_tool
    from tools.registry import registry

    home = get_hermes_home()
    (home / "config.yaml").write_text(hermes_yaml.safe_dump({
        "tts": {"provider": "edge", "edge": {"proxy": proxy, "voice": "en-GB-SoniaNeural", "speed": 1.25}}
    }), encoding="utf-8")
    monkeypatch.setenv("HTTPS_PROXY", "http://env.example:8080")
    calls = []

    def communicate(text, **kwargs):
        calls.append((text, kwargs))

        async def save(output):
            Path(output).write_bytes(b"ID3offline-contract-audio")

        return SimpleNamespace(save=save)

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: SimpleNamespace(Communicate=communicate))
    result = json.loads(registry.get_entry("text_to_speech").handler({
        "text": "Hello", "output_path": str(tmp_path / "speech.mp3")
    }))
    if proxy.startswith("socks5:"):
        assert result["success"] is False
        assert "Edge TTS proxy" in result["error"]
        assert not calls
        assert "secret" not in result["error"] + caplog.text
        assert "user:secret" not in result["error"] + caplog.text
    else:
        assert result["success"] is True, result
        assert result["provider"] == "edge"
        assert Path(result["file_path"]).read_bytes() == b"ID3offline-contract-audio"
        kwargs = {"voice": "en-GB-SoniaNeural", "rate": "+25%"}
        if proxy:
            kwargs["proxy"] = proxy
        assert calls == [("Hello", kwargs)]
