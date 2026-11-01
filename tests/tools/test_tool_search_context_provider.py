"""Regression coverage for provider-aware context sizing in the tool-search gate.

``model_tools._resolve_active_context_length()`` feeds ``should_activate``'s
window-fraction check. Providers like Codex OAuth enforce a lower context
window than the direct API for the same slug (e.g. gpt-5.5 is 1.05M on the
API but 272K on the Codex route), and ``get_model_context_length()`` only
applies those provider-aware resolutions when it receives the provider,
base_url, and credential. Before this coverage existed the gate called the
resolver with the model id alone, so Codex sessions sized activation against
generic direct-API metadata.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch


def _model_cfg(**overrides):
    cfg = {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": "",
    }
    cfg.update(overrides)
    return {"model": cfg}


class TestResolveActiveContextLengthProviderAware:
    def test_passes_provider_base_url_and_key_from_runtime(self):
        """Resolved runtime credentials must reach get_model_context_length."""
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured.update(
                model=model_id, base_url=base_url, api_key=api_key,
                config_ctx=config_context_length, provider=provider,
            )
            return 272_000

        with patch("hermes_cli.config.load_config", return_value=_model_cfg()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"base_url": "https://chatgpt.com/backend-api/codex",
                                 "api_key": "tok-live"}) as mock_rt, \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 272_000
        assert captured["provider"] == "openai-codex"
        assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert captured["api_key"] == "tok-live"
        mock_rt.assert_called_once_with(
            requested="openai-codex", target_model="gpt-5.6-sol"
        )

    def test_offline_credential_failure_degrades_to_config_values(self):
        """Runtime resolution raising must not zero the gate — the resolver is
        still called with the configured provider/base_url and an empty key so
        static provider-aware fallbacks apply."""
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured.update(base_url=base_url, api_key=api_key, provider=provider)
            return 272_000

        with patch("hermes_cli.config.load_config",
                   return_value=_model_cfg(base_url="https://chatgpt.com/backend-api/codex")), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   side_effect=RuntimeError("no credentials")), \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 272_000
        assert captured["provider"] == "openai-codex"
        assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert captured["api_key"] == ""

    def test_no_provider_configured_skips_runtime_resolution(self):
        """Without a provider in config, behavior matches the legacy path: no
        runtime resolution attempt, resolver called with empty routing."""
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured.update(base_url=base_url, provider=provider)
            return 200_000

        with patch("hermes_cli.config.load_config",
                   return_value={"model": {"model": "some-model"}}), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_rt, \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 200_000
        assert captured["provider"] == ""
        mock_rt.assert_not_called()

    def test_config_context_length_still_short_circuits(self):
        """Explicit model.context_length must keep winning (issue #46620)."""
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured["config_ctx"] = config_context_length
            return config_context_length or 0

        with patch("hermes_cli.config.load_config",
                   return_value=_model_cfg(context_length=150_000)), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"base_url": "https://chatgpt.com/backend-api/codex",
                                 "api_key": "tok"}), \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 150_000
        assert captured["config_ctx"] == 150_000


class TestResolveActiveContextLengthKeyCmd:
    """``key_cmd`` (and Entra ID) providers resolve to a callable token source, not a string.
    The gate used to ``str()`` the runtime credential, which turned the source into
    ``"<agent.command_token_source.CommandTokenSource object at 0x...>"``; the probe layer mints
    callables itself, so it sent that text verbatim as the bearer and every metadata probe at
    startup 401ed. The credential must reach the resolver in the shape the resolver understands."""

    @staticmethod
    def _key_cmd_source():
        from agent.command_token_source import CommandTokenSource

        return CommandTokenSource(f"{sys.executable} -c \"print('TOKEN-FROM-KEY-CMD')\"", "mingli")

    def test_callable_credential_reaches_resolver_unchanged(self):
        import model_tools

        source = self._key_cmd_source()
        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured.update(api_key=api_key, base_url=base_url, provider=provider)
            return 131_072

        with patch("hermes_cli.config.load_config",
                   return_value=_model_cfg(model="qwen-test", provider="mingli")), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"base_url": "http://127.0.0.1:8765/v1", "api_key": source}), \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 131_072
        assert captured["api_key"] is source
        assert captured["base_url"] == "http://127.0.0.1:8765/v1"
        assert captured["provider"] == "mingli"

    def test_static_credential_is_still_a_stripped_string(self):
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured["api_key"] = api_key
            return 131_072

        with patch("hermes_cli.config.load_config",
                   return_value=_model_cfg(model="qwen-test", provider="mingli")), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"base_url": "http://127.0.0.1:8765/v1", "api_key": "  static-key \n"}), \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            model_tools._resolve_active_context_length()

        assert captured["api_key"] == "static-key"

    def test_metadata_probes_carry_the_minted_token(self, tmp_path):
        """End to end against a local endpoint, no resolver mocked: every probe the gate triggers
        (LM Studio ``/api/v1/models``, ``/v1/models/<model>``, ``/v1/models``, Ollama ``/api/show``)
        must carry the token the ``key_cmd`` printed, never the source's repr."""
        import model_tools

        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self):
                seen.append((self.command, self.path, self.headers.get("Authorization")))
                body = json.dumps({"object": "list", "data": [{"id": "qwen-test", "object": "model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._reply()

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self._reply()

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        try:
            with patch("hermes_cli.config.load_config",
                       return_value=_model_cfg(model="qwen-test", provider="mingli")), \
                 patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       return_value={"base_url": base_url, "api_key": self._key_cmd_source()}), \
                 patch("agent.model_metadata.get_cached_context_length", return_value=None):
                model_tools._resolve_active_context_length()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        authorized = [(method, path, auth) for method, path, auth in seen if auth is not None]
        assert authorized, f"no authenticated probe reached the endpoint: {seen!r}"
        offenders = [entry for entry in authorized if entry[2] != "Bearer TOKEN-FROM-KEY-CMD"]
        assert not offenders, f"probes sent something other than the minted token: {offenders!r}"
        assert not any("CommandTokenSource" in (auth or "") for _, _, auth in seen)
