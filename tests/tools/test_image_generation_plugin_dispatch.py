from __future__ import annotations

import json
import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class _FakeCodexProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "codex"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {
            "success": True,
            "image": "/tmp/codex-test.png",
            "model": "gpt-5.2-codex",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "codex",
        }


class TestPluginDispatch:
    def test_dispatch_routes_to_codex_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")
        image_gen_registry.register_provider(_FakeCodexProvider())

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: _FakeCodexProvider() if name == "codex" else None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "square")
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["image"] == "/tmp/codex-test.png"
        assert payload["aspect_ratio"] == "square"

    def test_dispatch_reports_missing_registered_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: missing-codex\n")

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "missing-codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "landscape")
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["error_type"] == "provider_not_registered"
        assert "image_gen.provider='missing-codex'" in payload["error"]

    def test_dispatch_force_refreshes_plugins_when_provider_initially_missing(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")

        calls = []
        provider_state = {"provider": None}

        def fake_ensure_plugins_discovered(force=False):
            calls.append(force)
            if force:
                provider_state["provider"] = _FakeCodexProvider()

        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", fake_ensure_plugins_discovered)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider_state["provider"])

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw hammy", "portrait")
        payload = json.loads(dispatched)

        assert calls == [False, True]
        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["aspect_ratio"] == "portrait"


# A 1×1 PNG — smallest valid bytes that pass the magic-byte sniff in
# gateway.platforms.base.cache_image_from_bytes.
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8"
    b"\xcf\xc0\x00\x00\x00\x03\x00\x01\xa9\x12\x9b\xb1\x00\x00\x00\x00"
    b"IEND\xaeB`\x82"
)


class _FakeHTTPResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHTTPClient:
    """Minimal sync httpx.Client stand-in supporting the context-manager
    + .get() shape used by _cache_remote_image_url."""

    def __init__(self, response: _FakeHTTPResponse, calls: list):
        self._response = response
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, headers=None):
        self._calls.append(url)
        return self._response


class TestImageGenerateEphemeralUrlCache:
    """Regression guard for #26942 — image_generate URLs from providers like
    xAI Grok (`imgen.x.ai/xai-tmp-imgen-*`) 404 by the time the messaging
    adapter calls send_photo. Pre-caching at tool-completion time captures
    the bytes before the URL expires and emits a MEDIA: tag so the
    gateway's MEDIA-tag scanner delivers the local file."""

    def test_handle_caches_ephemeral_remote_url_and_emits_media_tag(
        self, monkeypatch, tmp_path
    ):
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        ephemeral_url = "https://imgen.x.ai/xai-tmp-imgen-deadbeef-cafe-feed.jpeg"

        def fake_dispatch(prompt, aspect_ratio):
            return json.dumps({
                "success": True,
                "image": ephemeral_url,
                "provider": "xai",
            })

        monkeypatch.setattr(
            image_generation_tool, "_dispatch_to_plugin_provider", fake_dispatch
        )

        import httpx
        calls: list = []
        monkeypatch.setattr(
            httpx, "Client",
            lambda *a, **kw: _FakeHTTPClient(_FakeHTTPResponse(_PNG_1x1), calls),
        )

        result = json.loads(image_generation_tool._handle_image_generate(
            {"prompt": "a Frühlingsmorgen im Garten"}
        ))

        assert result["success"] is True
        assert result["provider"] == "xai"
        assert calls == [ephemeral_url], (
            "Expected the ephemeral URL to be fetched exactly once at "
            "tool-completion time"
        )
        assert result["source_url"] == ephemeral_url
        # image field swapped to local path so extract_images() (which only
        # matches https?:// markdown) no longer routes through the failing
        # URL-based send_photo path.
        assert result["image"] != ephemeral_url
        # Extension is derived from the URL path (.jpeg → .jpeg, .png → .png)
        # and cache_image_from_bytes keeps it verbatim.
        assert result["image"].endswith(".jpeg")
        assert "media_tag" in result
        assert result["media_tag"] == f"MEDIA:{result['image']}"
        # Belt-and-suspenders: the gateway's MEDIA scanner looks for the
        # literal "MEDIA:" substring anywhere in the serialized tool
        # result, so the augmented payload must serialize to contain it.
        assert "MEDIA:" in json.dumps(result)

    def test_handle_preserves_local_path_results_unchanged(self, monkeypatch, tmp_path):
        """Plugin providers that already return a local path (the existing
        _FakeCodexProvider contract) must not be re-processed — no HTTP
        fetch, no field rewrite."""
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def fake_dispatch(prompt, aspect_ratio):
            return json.dumps({
                "success": True,
                "image": "/tmp/codex-test.png",
                "provider": "codex",
            })

        monkeypatch.setattr(
            image_generation_tool, "_dispatch_to_plugin_provider", fake_dispatch
        )

        import httpx

        def _no_http(*a, **kw):
            raise AssertionError("Local paths must not trigger an HTTP fetch")

        monkeypatch.setattr(httpx, "Client", _no_http)

        result = json.loads(image_generation_tool._handle_image_generate(
            {"prompt": "draw cat"}
        ))

        assert result["image"] == "/tmp/codex-test.png"
        assert "source_url" not in result
        assert "media_tag" not in result

    def test_handle_preserves_error_results_unchanged(self, monkeypatch, tmp_path):
        """A failed image_generate (success=False) has no URL to cache —
        the payload must round-trip unchanged so the error reaches the
        agent intact."""
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def fake_dispatch(prompt, aspect_ratio):
            return json.dumps({
                "success": False,
                "image": None,
                "error": "API quota exceeded",
                "error_type": "RateLimitError",
            })

        monkeypatch.setattr(
            image_generation_tool, "_dispatch_to_plugin_provider", fake_dispatch
        )

        import httpx

        def _no_http(*a, **kw):
            raise AssertionError("Error results must not trigger an HTTP fetch")

        monkeypatch.setattr(httpx, "Client", _no_http)

        result = json.loads(image_generation_tool._handle_image_generate(
            {"prompt": "x"}
        ))

        assert result["success"] is False
        assert result["image"] is None
        assert result["error_type"] == "RateLimitError"
        assert "source_url" not in result
        assert "media_tag" not in result

    def test_handle_falls_back_to_url_when_cache_fetch_fails(
        self, monkeypatch, tmp_path
    ):
        """When the URL is unreachable at tool-completion time (network
        flake, partial outage), the result must round-trip with the
        original URL intact — the messaging adapter's existing 3-tier
        fallback (URL → download → text) can still take a swing."""
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        flaky_url = "https://imgen.x.ai/xai-tmp-imgen-broken.jpeg"

        def fake_dispatch(prompt, aspect_ratio):
            return json.dumps({
                "success": True,
                "image": flaky_url,
                "provider": "xai",
            })

        monkeypatch.setattr(
            image_generation_tool, "_dispatch_to_plugin_provider", fake_dispatch
        )

        import httpx

        class _BoomClient:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def get(self, url, headers=None):
                raise RuntimeError("network refused")

        monkeypatch.setattr(httpx, "Client", lambda *a, **kw: _BoomClient())

        result = json.loads(image_generation_tool._handle_image_generate(
            {"prompt": "x"}
        ))

        assert result["success"] is True
        assert result["image"] == flaky_url, (
            "URL must be preserved verbatim when caching fails so the "
            "adapter can still attempt the legacy delivery chain"
        )
        assert "source_url" not in result
        assert "media_tag" not in result

    def test_handle_skips_non_image_response_bytes(self, monkeypatch, tmp_path):
        """A URL that returns an HTML error page instead of image bytes
        must be rejected by cache_image_from_bytes' magic-byte sniff;
        the original URL must be preserved so the failure is observable."""
        from tools import image_generation_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        url = "https://imgen.x.ai/xai-tmp-imgen-html.jpeg"

        def fake_dispatch(prompt, aspect_ratio):
            return json.dumps({"success": True, "image": url})

        monkeypatch.setattr(
            image_generation_tool, "_dispatch_to_plugin_provider", fake_dispatch
        )

        import httpx
        calls: list = []
        html_bytes = b"<!doctype html><html><body>404 Not Found</body></html>"
        monkeypatch.setattr(
            httpx, "Client",
            lambda *a, **kw: _FakeHTTPClient(_FakeHTTPResponse(html_bytes), calls),
        )

        result = json.loads(image_generation_tool._handle_image_generate(
            {"prompt": "x"}
        ))

        assert calls == [url]
        assert result["image"] == url
        assert "media_tag" not in result
