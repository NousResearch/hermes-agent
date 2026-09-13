"""Discovery must not leave ephemeral aux clients in runtime caches (offline)."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("ordering", ["probe-first", "runtime-first", "repeated-probe", "provider-change", "error-recovery"])
def test_browser_aux_vision_remains_callable_after_discovery(tmp_path, monkeypatch, ordering):
    import socket

    def offline_connect(*args, **kwargs):
        raise OSError("Network disabled for offline regression")

    monkeypatch.setattr(socket.socket, "connect", offline_connect)
    from agent import auxiliary_client as aux
    from tools import browser_tool_vision, vision_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ZAI_API_KEY", "offline-test-key")
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  vision:\n    provider: zai\n    model: fixture-vision\n",
        encoding="utf-8",
    )
    aux.shutdown_cached_clients()
    constructed = []
    requests = []

    class OfflineClient:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]
            self.api_key = kwargs["api_key"]
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            constructed.append(self)

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="offline fixture response"))])

        def close(self):
            pass

    monkeypatch.setattr(aux, "OpenAI", OfflineClient)
    monkeypatch.setattr(aux, "_openai_http_client_kwargs", lambda *a, **k: {})
    image = tmp_path / "screenshot.png"
    from PIL import Image
    Image.new("RGB", (2, 2), "white").save(image)

    try:
        if ordering == "runtime-first":
            assert browser_tool_vision._analyze_screenshot_with_aux_llm(image, "test") == "offline fixture response"
        before = len(constructed)
        for _ in range(3 if ordering == "repeated-probe" else 1):
            assert vision_tools.check_vision_requirements()
        assert len(constructed) == before, "Discovery constructed an SDK client"
        if ordering == "provider-change":
            assert browser_tool_vision._analyze_screenshot_with_aux_llm(image, "test") == "offline fixture response"
            old_url = constructed[-1].base_url
            monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
            (tmp_path / "config.yaml").write_text(
                "auxiliary:\n  vision:\n    provider: deepseek\n    model: fixture-vision\n",
                encoding="utf-8",
            )
            assert vision_tools.check_vision_requirements()
        if ordering == "error-recovery":
            def failed_constructor(**kwargs):
                raise RuntimeError("offline constructor failure")
            monkeypatch.setattr(aux, "OpenAI", failed_constructor)
            with pytest.raises(RuntimeError, match="offline constructor failure"):
                aux.resolve_vision_provider_client()
            monkeypatch.setattr(aux, "OpenAI", OfflineClient)
        assert browser_tool_vision._analyze_screenshot_with_aux_llm(image, "test") == "offline fixture response"
        if ordering == "provider-change":
            assert constructed[-1].base_url != old_url
        assert requests and requests[-1]["model"] == "fixture-vision"
        assert not any(isinstance(entry[0], aux._AuxProbeClientStub) for entry in aux._client_cache.values())
    finally:
        # RED must report the runtime failure, not a second stub.close failure.
        with aux._client_cache_lock:
            aux._client_cache.clear()
