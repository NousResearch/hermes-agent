"""Tests for the Tinfoil EHBP transport.

These tests cover:
* Transport registration and api_mode
* build_kwargs (inherits from ChatCompletionsTransport)
* normalize_response (inherits from ChatCompletionsTransport)
* Fallback to plain TLS when SDK unavailable
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestTinfoilTransportBasics:
    """Registration, api_mode, and identity."""

    def test_api_mode(self):
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        assert t.api_mode == "tinfoil_ehbp"

    def test_registered(self):
        from agent.transports import get_transport
        t = get_transport("tinfoil_ehbp")
        assert t is not None
        assert t.api_mode == "tinfoil_ehbp"

    def test_inherits_from_chat_completions(self):
        from agent.transports.chat_completions import ChatCompletionsTransport
        from agent.transports.tinfoil import TinfoilTransport
        assert issubclass(TinfoilTransport, ChatCompletionsTransport)

    def test_get_transport_returns_instance(self):
        from agent.transports import get_transport
        t1 = get_transport("tinfoil_ehbp")
        t2 = get_transport("tinfoil_ehbp")
        assert t1 is not None and t2 is not None
        # Each call returns a fresh instance
        assert t1 is not t2


class TestTinfoilTransportBuildKwargs:
    """build_kwargs inherits ChatCompletionsTransport behaviour."""

    def test_basic_kwargs_no_tools(self):
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        kw = t.build_kwargs(
            model="kimi-k2-6",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert kw["model"] == "kimi-k2-6"
        assert len(kw["messages"]) == 1
        assert "tools" not in kw

    def test_kwargs_with_tools(self):
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_fn",
                    "description": "A test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        kw = t.build_kwargs(
            model="kimi-k2-6",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        assert kw["tools"] == tools

    def test_kwargs_with_provider_profile(self):
        """End-to-end: profile injects tinfoil_endpoint into extra_body."""
        from providers import get_provider_profile
        p = get_provider_profile("tinfoil")
        assert p is not None
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        kw = t.build_kwargs(
            model="tinfoil/kimi-k2-6",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            provider_profile=p,
        )
        extra = kw.get("extra_body", {})
        assert extra.get("tinfoil_endpoint") == "kimi-k2-6"

    def test_kwargs_provider_profile_headers(self):
        from providers import get_provider_profile
        p = get_provider_profile("tinfoil")
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        kw = t.build_kwargs(
            model="tinfoil/kimi-k2-6",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            provider_profile=p,
        )
        # Content-Type comes from profile's default_headers
        extra_headers = kw.get("extra_headers", {})
        assert extra_headers.get("Content-Type") == "application/json"


class TestTinfoilTransportNormalizeResponse:
    """normalize_response is identical to ChatCompletionsTransport."""

    def test_normalize_text_response(self):
        from openai.types.chat import ChatCompletion
        mock_response = MagicMock(spec=ChatCompletion)
        mock_response.choices = [
            MagicMock(
                finish_reason="stop",
                message=MagicMock(
                    content="Hello!",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                    model_extra=None,
                ),
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        nr = t.normalize_response(mock_response)

        assert nr.content == "Hello!"
        assert nr.finish_reason == "stop"
        assert nr.tool_calls is None
        assert nr.usage is not None
        assert nr.usage.prompt_tokens == 10
        assert nr.usage.completion_tokens == 5

    def test_normalize_no_usage(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                finish_reason="stop",
                message=MagicMock(
                    content="Hi",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                    model_extra=None,
                ),
            )
        ]
        mock_response.usage = None

        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        nr = t.normalize_response(mock_response)

        assert nr.content == "Hi"
        assert nr.usage is None  # No usage when response lacks it

    def test_normalize_invalid_response(self):
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        assert t.validate_response(None) is False
        assert t.validate_response(MagicMock(choices=None)) is False


class TestTinfoilTransportSecureClient:
    """SecureClient lifecycle and fallback."""

    def test_build_secure_client_sdk_not_available(self):
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        with patch("agent.transports.tinfoil._TINFOIL_AVAILABLE", False):
            client = t.build_secure_client(api_key="test-key")
        assert client is None

    def test_build_secure_client_init_failure(self):
        """SDK available but SecureClient constructor raises."""
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        with patch("agent.transports.tinfoil._TINFOIL_AVAILABLE", True):
            with patch(
                "agent.transports.tinfoil.SecureClient",
                side_effect=RuntimeError("no enclave"),
            ):
                client = t.build_secure_client(api_key="test-key")
        assert client is None

    def test_build_secure_openai_client_sdk_not_available(self):
        """Falls back to plain openai.OpenAI when SDK not available."""
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        fake_openai = MagicMock()
        fake_openai_instance = MagicMock()
        fake_openai.OpenAI.return_value = fake_openai_instance

        with (
            patch("agent.transports.tinfoil._TINFOIL_AVAILABLE", False),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            client = t.build_secure_openai_client(
                api_key="key",
                base_url="https://inference.tinfoil.sh/v1",
            )
        assert client is fake_openai_instance
        fake_openai.OpenAI.assert_called_once()

    def test_build_secure_openai_client_secure_client_none(self):
        """build_secure_client returns None → falls back to plain OpenAI."""
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        fake_openai = MagicMock()
        fake_openai_instance = MagicMock()
        fake_openai.OpenAI.return_value = fake_openai_instance

        with (
            patch.object(t, "build_secure_client", return_value=None),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            client = t.build_secure_openai_client(
                api_key="key",
                base_url="https://inference.tinfoil.sh/v1",
            )
        assert client is fake_openai_instance

    def test_build_kwargs_delegates_to_super(self):
        """build_kwargs still works even after SecureClient attempts."""
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        kw = t.build_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )
        assert kw["model"] == "test-model"
        assert kw["messages"][0]["content"] == "test"

    def test_build_secure_openai_client_uses_enclave_as_base_url(self):
        """Secure client's enclave hostname is used as base_url so TLS
        cert pinning matches the host we actually connect to."""
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()

        mock_secure = MagicMock()
        mock_secure.enclave = "router.inf6.tinfoil.sh"
        mock_secure.make_secure_http_client.return_value = MagicMock()

        fake_openai = MagicMock()
        fake_openai_instance = MagicMock()
        fake_openai.OpenAI.return_value = fake_openai_instance

        with (
            patch.object(t, "build_secure_client", return_value=mock_secure),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            client = t.build_secure_openai_client(
                api_key="key",
                base_url="https://inference.tinfoil.sh/v1",
            )

        assert client is fake_openai_instance
        fake_openai.OpenAI.assert_called_once_with(
            api_key="key",
            base_url="https://router.inf6.tinfoil.sh/v1/",
            http_client=mock_secure.make_secure_http_client.return_value,
            timeout=None,
            default_headers=None,
        )


class TestTinfoilRequireSecure:
    """Fail-closed behaviour gated by tinfoil.require_secure."""

    def test_require_secure_raises_when_sdk_unavailable(self):
        """require_secure=True + no secure client → raise, never fall back."""
        from agent.transports.tinfoil import (
            TinfoilSecureUnavailableError,
            TinfoilTransport,
        )
        t = TinfoilTransport()
        fake_openai = MagicMock()

        with (
            patch.object(t, "build_secure_client", return_value=None),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            with pytest.raises(TinfoilSecureUnavailableError):
                t.build_secure_openai_client(
                    api_key="key",
                    base_url="https://inference.tinfoil.sh/v1",
                    require_secure=True,
                )
        # Must NOT have constructed a plain-TLS client.
        fake_openai.OpenAI.assert_not_called()

    def test_require_secure_raises_when_http_client_fails(self):
        """Secure client built but transport creation fails → raise."""
        from agent.transports.tinfoil import (
            TinfoilSecureUnavailableError,
            TinfoilTransport,
        )
        t = TinfoilTransport()

        mock_secure = MagicMock()
        mock_secure.enclave = "router.inf6.tinfoil.sh"
        mock_secure.make_secure_http_client.side_effect = RuntimeError("handshake")

        fake_openai = MagicMock()

        with (
            patch.object(t, "build_secure_client", return_value=mock_secure),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            with pytest.raises(TinfoilSecureUnavailableError):
                t.build_secure_openai_client(
                    api_key="key",
                    base_url="https://inference.tinfoil.sh/v1",
                    require_secure=True,
                )
        fake_openai.OpenAI.assert_not_called()
        # is_secure() must report False after a failed handshake.
        assert t.is_secure() is False

    def test_require_secure_false_still_falls_back(self):
        """Default (require_secure=False) preserves plain-TLS fallback."""
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        fake_openai = MagicMock()
        fake_instance = MagicMock()
        fake_openai.OpenAI.return_value = fake_instance

        with (
            patch.object(t, "build_secure_client", return_value=None),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            client = t.build_secure_openai_client(
                api_key="key",
                base_url="https://inference.tinfoil.sh/v1",
                require_secure=False,
            )
        assert client is fake_instance

    def test_build_tinfoil_client_honors_require_secure_config(self):
        """_build_tinfoil_client reads tinfoil.require_secure from config and
        fails closed when the transport isn't registered."""
        from agent.agent_runtime_helpers import _build_tinfoil_client
        from agent.transports.tinfoil import TinfoilSecureUnavailableError

        agent = MagicMock()
        agent._get_transport.return_value = None

        with patch(
            "hermes_cli.config.load_config",
            return_value={"tinfoil": {"require_secure": True}},
        ):
            with pytest.raises(TinfoilSecureUnavailableError):
                _build_tinfoil_client(
                    agent,
                    {"api_key": "k", "base_url": "https://inference.tinfoil.sh/v1"},
                )

    def test_build_tinfoil_client_passes_require_secure_to_transport(self):
        """The config flag is forwarded to build_secure_openai_client."""
        from agent.agent_runtime_helpers import _build_tinfoil_client

        mock_transport = MagicMock()
        mock_transport.build_secure_openai_client.return_value = MagicMock()
        agent = MagicMock()
        agent._get_transport.return_value = mock_transport

        with patch(
            "hermes_cli.config.load_config",
            return_value={"tinfoil": {"require_secure": True}},
        ):
            _build_tinfoil_client(
                agent,
                {"api_key": "k", "base_url": "https://inference.tinfoil.sh/v1"},
            )

        _, kwargs = mock_transport.build_secure_openai_client.call_args
        assert kwargs["require_secure"] is True


class TestTinfoilTransportVerificationDoc:
    """get_verification_document accessor."""

    def test_no_doc_when_no_client(self):
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        assert t.get_verification_document() is None

    def test_returns_cached_doc(self):
        from agent.transports.tinfoil import TinfoilTransport
        t = TinfoilTransport()
        doc = {"status": "verified"}
        t._verification_document = doc
        assert t.get_verification_document() is doc


class TestTinfoilProviderProfileApiMode:
    """Provider profile declares tinfoil_ehbp as its api_mode."""

    def test_profile_api_mode(self):
        from providers import get_provider_profile
        p = get_provider_profile("tinfoil")
        assert p is not None
        assert p.api_mode == "tinfoil_ehbp"


class TestBuildTinfoilClient:
    """_build_tinfoil_client in agent_runtime_helpers."""

    def test_transport_not_registered_fallback(self):
        """When transport isn't registered, falls back to plain OpenAI."""
        from agent.agent_runtime_helpers import _build_tinfoil_client

        fake_instance = MagicMock()

        agent = MagicMock()
        agent._build_keepalive_http_client.return_value = None
        # The cached-transport lookup returns None when no transport is
        # registered for tinfoil_ehbp.
        agent._get_transport.return_value = None
        client_kwargs = {
            "api_key": "test-key",
            "base_url": "https://inference.tinfoil.sh/v1",
        }

        with patch(
            "run_agent.OpenAI",
            return_value=fake_instance,
        ):
            client = _build_tinfoil_client(agent, client_kwargs)

        assert client is fake_instance
        agent._get_transport.assert_called_once_with("tinfoil_ehbp")

    def test_transport_registered_delegates(self):
        """When transport is registered, delegates to build_secure_openai_client."""
        from agent.agent_runtime_helpers import _build_tinfoil_client

        mock_transport = MagicMock()
        mock_openai_client = MagicMock()
        mock_transport.build_secure_openai_client.return_value = mock_openai_client

        agent = MagicMock()
        # _build_tinfoil_client must read the agent's *cached* transport so
        # the attestation result it records is observable later via
        # is_secure() (status bar / TUI session-info probe).
        agent._get_transport.return_value = mock_transport
        client_kwargs = {
            "api_key": "test-key",
            "base_url": "https://inference.tinfoil.sh/v1",
            "timeout": 30.0,
            "default_headers": {"Content-Type": "application/json"},
        }

        client = _build_tinfoil_client(agent, client_kwargs)

        assert client is mock_openai_client
        agent._get_transport.assert_called_once_with("tinfoil_ehbp")
        mock_transport.build_secure_openai_client.assert_called_once_with(
            api_key="test-key",
            base_url="https://inference.tinfoil.sh/v1",
            timeout=30.0,
            default_headers={"Content-Type": "application/json"},
            require_secure=False,
        )

    def test_secure_state_observable_via_cached_transport(self):
        """Regression: the attestation result recorded during client build
        must be observable later via the same cached transport instance.

        Previously ``_build_tinfoil_client`` and the status-bar probe each
        created a throwaway transport via the module-level ``get_transport``,
        so ``is_secure()`` always reported ``False`` even on a verified
        connection. Both now share the agent's cached transport.
        """
        from agent.agent_runtime_helpers import _build_tinfoil_client
        from agent.transports.tinfoil import TinfoilTransport

        transport = TinfoilTransport()
        assert transport.is_secure() is False

        mock_secure = MagicMock()
        mock_secure.enclave = "router.inf6.tinfoil.sh"
        mock_secure.make_secure_http_client.return_value = MagicMock()

        agent = MagicMock()
        # The agent's transport cache always hands back the SAME instance.
        agent._get_transport.return_value = transport

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = MagicMock()

        # Drive the real build_secure_client path (which sets _secure_client)
        # by faking the SDK at the SecureClient level.
        with (
            patch("agent.transports.tinfoil._TINFOIL_AVAILABLE", True),
            patch(
                "agent.transports.tinfoil.SecureClient",
                return_value=mock_secure,
            ),
            patch.dict("sys.modules", {"openai": fake_openai}),
        ):
            _build_tinfoil_client(
                agent,
                {"api_key": "k", "base_url": "https://inference.tinfoil.sh/v1"},
            )

        # The cached instance the status bar / TUI would query now reports
        # the verified (secure) state.
        assert agent._get_transport("tinfoil_ehbp").is_secure() is True


class TestAgentInitApiModeDetection:
    """agent_init.py routes tinfoil provider to tinfoil_ehbp api_mode."""

    def test_provider_profile_has_correct_api_mode(self):
        """The TinfoilProvider profile declares the correct api_mode."""
        from providers import get_provider_profile
        p = get_provider_profile("tinfoil")
        assert p is not None
        assert p.api_mode == "tinfoil_ehbp"