"""GPT-Live voice chat mode on Azure: endpoint detection and Azure Realtime session wiring.

When ``voice.gpt_live.base_url`` points at an Azure OpenAI resource
(``*.openai.azure.com`` / ``*.cognitiveservices.azure.com``), gpt-live switches from the
OpenAI WebRTC path (``POST /v1/live/sessions``) to the Azure Realtime API over WebSocket,
where ``voice.gpt_live.model`` is the Azure *deployment* name and ``api_key`` is the Azure
resource key. These tests pin backend detection (including suffix-spoofing look-alikes),
the WebSocket URL and both session payloads, endpoint dispatch, and the WebSocket
handshake (api-key header, session event handling, errors/timeouts); the default
(OpenAI) behavior is pinned as unchanged.
"""

import json
import sys
import types
import urllib.parse

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
        # suffix spoofing: the azure label must be the actual end of the host
        "https://foo.openai.azure.com.evil.example.com/",
        "https://foo.cognitiveservices.azure.com.attacker.io/",
        "https://openai.azure.com.evil.com/",
        # look-alike labels that are not the azure suffix
        "https://notopenai.azure.com/",
        "https://evil-openai.azure.com/",
        # userinfo trick: the real host is the one after '@'
        "https://foo.openai.azure.com@evil.example.com/",
        # the apex is not a *.openai.azure.com resource host
        "https://openai.azure.com",
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

    def test_default_api_version_is_the_pinned_preview(self):
        url = voice_live.azure_realtime_ws_url("https://foo.openai.azure.com", "dep")
        assert "api-version=2025-04-01-preview" in url

    @pytest.mark.parametrize("path", ["/", "/openai", "/openai/"])
    def test_base_path_variants_keep_single_openai_segment(self, path):
        parsed = urllib.parse.urlparse(
            voice_live.azure_realtime_ws_url("https://foo.openai.azure.com" + path, "dep"))
        assert parsed.scheme == "wss"
        assert parsed.path == "/openai/realtime"

    def test_query_round_trips_through_parse_qs(self):
        url = voice_live.azure_realtime_ws_url("https://foo.openai.azure.com", "my dep/x?")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert query["deployment"] == ["my dep/x?"]
        assert query["api-version"] == [voice_live.AZURE_REALTIME_API_VERSION]


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


AZURE_KEY = "azure-test-key-0001"
AZURE_BASE = "https://myresource.openai.azure.com"
DEPLOYMENT = "gpt-4o-realtime-preview"
API_VERSION = "2025-04-01-preview"


class _FakeWS:
    """Scripted stand-in for the send/recv/close wrapper ``_azure_ws_connect`` returns."""

    def __init__(self, events):
        self._events = list(events)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self, timeout=None):
        if not self._events:
            raise ConnectionError("socket closed by peer")
        event = self._events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event if isinstance(event, str) else json.dumps(event)

    def close(self):
        self.closed = True


class _FakeRawSocket:
    """Scripted stand-in for the raw client socket object."""

    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout=None):
        return "{}"

    def close(self):
        self.closed = True


def _live_section(**overrides):
    section = {"api_key": AZURE_KEY, "base_url": AZURE_BASE, "model": DEPLOYMENT, "voice": "echo"}
    section.update(overrides)
    return section


def _install_fake_ws(monkeypatch, events):
    """Patch ``_azure_ws_connect`` to return a scripted ``_FakeWS``; captures connect args."""
    ws = _FakeWS(events)
    captured = {}

    def fake_connect(url, api_key, timeout):
        captured.update(url=url, api_key=api_key, timeout=timeout)
        return ws

    monkeypatch.setattr(voice_live, "_azure_ws_connect", fake_connect)
    monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: _live_section())
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: (AZURE_KEY, AZURE_BASE))
    return ws, captured


class TestOpenAiSessionConfigUnchanged:
    """The OpenAI ``/live/sessions`` payload must not gain Azure-flavored fields."""

    def test_payload_shape_and_history(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {
            "model": "gpt-live-1", "voice": "willow", "instructions": "Speak Spanish.",
        })
        history = [{"type": "message", "role": "user", "content": []}]
        config = voice_live.build_session_config(history)
        assert config["model"] == "gpt-live-1"
        assert config["delegation"] == {"type": "client"}
        assert config["audio"] == {"output": {"voice": "willow"}}
        assert config["instructions"].endswith("Speak Spanish.")
        assert config["input"] == history
        assert "voice" not in config
        assert "input_audio_format" not in config
        assert "output_audio_format" not in config
        assert "turn_detection" not in config

    def test_no_history_no_input_key(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {})
        assert "input" not in voice_live.build_session_config()


class TestAzureSessionUpdatePayload:
    """The Azure ``session.update`` payload: the deployment rides in the WS URL, not here."""

    def test_payload_shape(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {
            "model": "my-deployment", "voice": "echo", "instructions": "Be terse.",
        })
        history = [{"type": "message", "role": "user", "content": "hi"}]
        config = voice_live.build_azure_session_config(history)
        assert "model" not in config
        assert "delegation" not in config
        assert config["voice"] == "echo"
        assert config["input_audio_format"] == "pcm16"
        assert config["output_audio_format"] == "pcm16"
        assert config["turn_detection"] == {"type": "server_vad"}
        assert config["input_audio_transcription"] == {"model": "whisper-1"}
        assert config["instructions"].endswith("Be terse.")
        assert "input" not in config  # history uses conversation.item.create, not session fields

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {})
        config = voice_live.build_azure_session_config()
        assert config["voice"] == voice_live.DEFAULT_LIVE_VOICE
        assert "input" not in config


class TestCreateWebrtcSessionDispatch:
    def test_azure_endpoint_dispatches_to_websocket_session(self, monkeypatch):
        calls = []
        sentinel = {"session": {"id": "s1"}, "transport": {"type": "websocket"}}

        def fake_azure(sdp_offer=None, history=None, timeout=30.0):
            calls.append((sdp_offer, history))
            return sentinel

        monkeypatch.setattr(voice_live, "create_azure_realtime_session", fake_azure)
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: _live_section())
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: (AZURE_KEY, AZURE_BASE))
        monkeypatch.setattr(
            voice_live.urllib.request, "urlopen",
            lambda *a, **k: pytest.fail("the OpenAI HTTP exchange must not run on an Azure endpoint"),
        )
        history = [{"type": "message", "role": "user", "content": []}]
        assert voice_live.create_webrtc_session("v=0 offer", history=history) is sentinel
        assert calls == [("v=0 offer", history)]

    def test_openai_endpoint_keeps_http_sdp_exchange(self, monkeypatch):
        def fail_azure(*a, **k):
            pytest.fail("the Azure WebSocket path must not run on an OpenAI endpoint")

        monkeypatch.setattr(voice_live, "create_azure_realtime_session", fail_azure)
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {"voice": "willow"})
        monkeypatch.setattr(
            voice_live, "_resolve_credentials",
            lambda live: ("sk-test", voice_live.DEFAULT_LIVE_BASE_URL),
        )

        captured = {}

        class _Resp:
            def read(self):
                return json.dumps(
                    {"session": {"id": "x"}, "transport": {"type": "webrtc", "sdp": "answer"}}
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            captured["body"] = json.loads(req.data)
            return _Resp()

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)
        result = voice_live.create_webrtc_session("v=0 offer")
        assert result["transport"]["sdp"] == "answer"
        assert captured["url"] == voice_live.DEFAULT_LIVE_BASE_URL + "/live/sessions"
        assert captured["auth"] == "Bearer sk-test"
        assert captured["body"]["transport"] == {"type": "webrtc", "sdp": "v=0 offer"}
        assert captured["body"]["session"]["delegation"] == {"type": "client"}

    def test_missing_key_refuses_before_any_transport(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("", AZURE_BASE))
        monkeypatch.setattr(voice_live, "create_azure_realtime_session", lambda *a, **k: pytest.fail("must not connect"))
        monkeypatch.setattr(voice_live.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call the vendor"))
        with pytest.raises(ValueError):
            voice_live.create_webrtc_session("v=0 offer")


class TestAzureRealtimeHandshake:
    def test_session_created_update_ack_and_result_contract(self, monkeypatch):
        history = [{"type": "message", "role": "user", "content": "hi"}]
        ws, captured = _install_fake_ws(monkeypatch, [
            {"type": "session.created", "session": {"id": "sess_abc"}},
            {"type": "session.updated", "session": {"id": "sess_abc"}},
        ])
        result = voice_live.create_azure_realtime_session("v=0 offer", history=history, timeout=7.0)

        # the wss URL carries api-version + deployment; the key travels via the
        # api-key header argument, never inside the URL
        parsed = urllib.parse.urlparse(captured["url"])
        assert parsed.scheme == "wss"
        assert parsed.path == "/openai/realtime"
        query = urllib.parse.parse_qs(parsed.query)
        assert query["api-version"] == [API_VERSION]
        assert query["deployment"] == [DEPLOYMENT]
        assert captured["api_key"] == AZURE_KEY
        assert captured["timeout"] == 7.0
        assert AZURE_KEY not in captured["url"]

        # one session.update followed by conversation.item.create history seeding
        assert [m["type"] for m in ws.sent] == ["session.update", "conversation.item.create"]
        session = ws.sent[0]["session"]
        assert "model" not in session
        assert "delegation" not in session
        assert session["voice"] == "echo"
        assert session["input_audio_format"] == "pcm16"
        assert session["output_audio_format"] == "pcm16"
        assert session["turn_detection"] == {"type": "server_vad"}
        assert session["input_audio_transcription"] == {"model": "whisper-1"}
        assert "input" not in session
        assert ws.sent[1] == {"type": "conversation.item.create", "item": history[0]}

        # result mirrors the vendor contract; the key appears nowhere in it
        assert result["session"]["id"] == "sess_abc"
        assert result["transport"]["type"] == "websocket"
        assert result["transport"]["url"] == captured["url"]
        assert result["transport"]["api_version"] == API_VERSION
        assert AZURE_KEY not in json.dumps(result)
        assert ws.closed is True

    def test_non_json_frame_is_skipped(self, monkeypatch):
        ws, _ = _install_fake_ws(monkeypatch, [
            "not-json-at-all",
            {"type": "session.created", "session": {"id": "s1"}},
            {"type": "session.updated"},
        ])
        assert voice_live.create_azure_realtime_session(timeout=5.0)["session"]["id"] == "s1"
        assert [m["type"] for m in ws.sent] == ["session.update"]

    def test_error_event_before_created_raises(self, monkeypatch):
        ws, _ = _install_fake_ws(monkeypatch, [{"type": "error", "error": {"message": "invalid api-key"}}])
        with pytest.raises(RuntimeError, match="invalid api-key"):
            voice_live.create_azure_realtime_session()
        assert ws.closed is True

    def test_error_event_on_update_raises(self, monkeypatch):
        ws, _ = _install_fake_ws(monkeypatch, [
            {"type": "session.created", "session": {"id": "s1"}},
            {"type": "error", "error": {"message": "unknown voice"}},
        ])
        with pytest.raises(RuntimeError, match="unknown voice"):
            voice_live.create_azure_realtime_session()
        assert [m["type"] for m in ws.sent] == ["session.update"]
        assert ws.closed is True

    def test_socket_close_or_timeout_raises_runtime_error(self, monkeypatch):
        ws, _ = _install_fake_ws(monkeypatch, [TimeoutError("timed out")])
        with pytest.raises(RuntimeError, match="socket closed"):
            voice_live.create_azure_realtime_session()
        assert ws.closed is True

    def test_missing_key_raises_before_connect(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: _live_section(api_key=""))
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("", AZURE_BASE))
        monkeypatch.setattr(voice_live, "_azure_ws_connect", lambda *a, **k: pytest.fail("must not connect"))
        with pytest.raises(ValueError):
            voice_live.create_azure_realtime_session()


class TestAzureWsConnectAuthHeader:
    @staticmethod
    def _install_fake_websockets(monkeypatch, captured):
        raw = _FakeRawSocket()

        def fake_connect(url, additional_headers=None, open_timeout=None):
            captured.update(url=url, headers=additional_headers, timeout=open_timeout)
            return raw

        client_mod = types.ModuleType("websockets.sync.client")
        client_mod.connect = fake_connect
        sync_mod = types.ModuleType("websockets.sync")
        sync_mod.__path__ = []
        sync_mod.client = client_mod
        ws_mod = types.ModuleType("websockets")
        ws_mod.__path__ = []
        ws_mod.sync = sync_mod
        monkeypatch.setitem(sys.modules, "websockets", ws_mod)
        monkeypatch.setitem(sys.modules, "websockets.sync", sync_mod)
        monkeypatch.setitem(sys.modules, "websockets.sync.client", client_mod)
        return raw

    def test_websockets_lib_gets_api_key_header_not_url(self, monkeypatch):
        captured = {}
        raw = self._install_fake_websockets(monkeypatch, captured)
        url = f"wss://myresource.openai.azure.com/openai/realtime?api-version={API_VERSION}&deployment=d"
        wrapper = voice_live._azure_ws_connect(url, AZURE_KEY, 12.0)
        assert captured["url"] == url
        assert captured["headers"] == {"api-key": AZURE_KEY}
        assert captured["timeout"] == 12.0
        assert AZURE_KEY not in url
        # the returned wrapper still delegates send/recv/close
        wrapper.send("x")
        assert raw.sent == ["x"]
        assert wrapper.recv(timeout=1.0) == "{}"
        wrapper.close()
        assert raw.closed is True

    def test_websocket_client_fallback_uses_api_key_header(self, monkeypatch):
        captured = {}
        monkeypatch.setitem(sys.modules, "websockets", None)  # force ImportError on the primary lib
        client = types.ModuleType("websocket")
        raw = _FakeRawSocket()

        def fake_create_connection(url, header=None, timeout=None):
            captured.update(url=url, header=header, timeout=timeout)
            return raw

        client.create_connection = fake_create_connection
        monkeypatch.setitem(sys.modules, "websocket", client)
        url = f"wss://myresource.openai.azure.com/openai/realtime?api-version={API_VERSION}&deployment=d"
        voice_live._azure_ws_connect(url, AZURE_KEY, 9.0)
        assert captured["url"] == url
        assert captured["header"] == ["api-key: " + AZURE_KEY]
        assert captured["timeout"] == 9.0
        assert AZURE_KEY not in url

    def test_no_ws_library_available_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "websockets", None)
        monkeypatch.setitem(sys.modules, "websocket", None)
        with pytest.raises(RuntimeError, match="websockets"):
            voice_live._azure_ws_connect("wss://myresource.openai.azure.com/openai/realtime", AZURE_KEY, 1.0)
