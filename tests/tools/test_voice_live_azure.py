"""GPT-Live voice chat mode on Azure: endpoint detection and Azure Realtime session wiring.

When ``voice.gpt_live.base_url`` points at an Azure OpenAI resource
(``*.openai.azure.com`` / ``*.cognitiveservices.azure.com``), gpt-live switches from the
OpenAI WebRTC path (``POST /v1/live/sessions``) to the Azure Realtime API over WebSocket,
where ``voice.gpt_live.model`` is the Azure *deployment* name and ``api_key`` is the Azure
resource key. These tests pin the backend detection and the session-config shape; the
default (OpenAI) behavior is pinned as unchanged.
"""

import pytest

from tools import voice_live


class TestLiveBackendDetection:
    @pytest.mark.parametrize("base_url", [
        "https://foo.openai.azure.com",
        "https://foo.openai.azure.com/openai/v1",
        "https://foo.cognitiveservices.azure.com",
        "https://foo.cognitiveservices.azure.com/openai/v1/",
        "https://FOO.OPENAI.AZURE.COM",  # host matching is case-insensitive
    ])
    def test_azure_resource_hosts_detect_as_azure(self, base_url):
        assert voice_live.live_backend(base_url) == voice_live.BACKEND_AZURE

    @pytest.mark.parametrize("base_url", [
        voice_live.DEFAULT_LIVE_BASE_URL,  # https://api.openai.com/v1
        "https://api.openai.com/v1",
        "https://api.x.ai/v1",
        "https://example.com/v1",
        "https://foo.azure.com",  # azure.com but not an OpenAI resource host
        "",
        "not-a-url",
    ])
    def test_non_azure_hosts_stay_on_openai(self, base_url):
        assert voice_live.live_backend(base_url) == voice_live.BACKEND_OPENAI


class TestAzureRealtimeWsUrl:
    def test_bare_resource_url_becomes_realtime_websocket(self):
        url = voice_live.azure_realtime_ws_url("https://foo.openai.azure.com", "my-deploy")
        assert url == (
            "wss://foo.openai.azure.com/openai/realtime"
            f"?api-version={voice_live.AZURE_REALTIME_API_VERSION}&deployment=my-deploy"
        )

    def test_openai_path_suffix_is_tolerated(self):
        url = voice_live.azure_realtime_ws_url("https://foo.openai.azure.com/openai", "my-deploy")
        assert url.startswith("wss://foo.openai.azure.com/openai/realtime?")
        assert "deployment=my-deploy" in url

    def test_cognitiveservices_host_and_custom_api_version(self):
        url = voice_live.azure_realtime_ws_url(
            "https://foo.cognitiveservices.azure.com", "dep", api_version="2024-10-01-preview")
        assert url.startswith("wss://foo.cognitiveservices.azure.com/openai/realtime?")
        assert "api-version=2024-10-01-preview" in url
        assert "deployment=dep" in url

    def test_deployment_name_is_url_encoded(self):
        url = voice_live.azure_realtime_ws_url("https://foo.openai.azure.com", "my deploy")
        assert "deployment=my+deploy" in url

    def test_plain_http_maps_to_ws(self):
        url = voice_live.azure_realtime_ws_url("http://foo.openai.azure.com", "dep")
        assert url.startswith("ws://foo.openai.azure.com/openai/realtime?")


class TestAzureSessionConfig:
    def test_model_is_the_azure_deployment_name(self, monkeypatch):
        """On an Azure endpoint the configured ``model`` is sent as-is: it names the Azure
        deployment, not an OpenAI model id."""
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {
            "model": "my-gpt-realtime-deployment",
            "voice": "alloy",
            "base_url": "https://foo.openai.azure.com",
            "api_key": "azure-resource-key",
        })

        config = voice_live.build_session_config()

        assert config["model"] == "my-gpt-realtime-deployment"
        assert config["audio"]["output"]["voice"] == "alloy"
        assert config["delegation"] == {"type": "client"}
        assert config["instructions"] == voice_live.LIVE_PERSONA

    def test_default_model_is_the_openai_live_model(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {})
        assert voice_live.build_session_config()["model"] == voice_live.DEFAULT_LIVE_MODEL

    def test_status_reports_deployment_name_and_resolved_azure_key(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_voice_section", lambda: {
            "voice_chat_mode": "gpt-live",
            "gpt_live": {
                "model": "my-gpt-realtime-deployment",
                "base_url": "https://foo.openai.azure.com",
                "api_key": "azure-resource-key",
            },
        })

        status = voice_live.resolve_gpt_live_status()

        assert status["mode"] == "gpt-live"
        assert status["available"] is True
        assert status["model"] == "my-gpt-realtime-deployment"
        assert "azure-resource-key" not in str(status)  # never leaks the key
