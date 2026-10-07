"""Tests for trajectory_compressor.py -- init/provider-detection layer and summary retry/fallback.

Covers the regions of ``trajectory_compressor.py`` that exercise:

1. ``_effective_temperature_for_model`` import-success / import-failure branches.
2. ``_init_tokenizer`` success and runtime-error paths.
3. ``_init_summarizer`` provider-vs-custom-endpoint routing and the missing-key error.
4. ``_detect_provider`` base-url -> provider-name mapping.
5. ``count_tokens`` fallback on tokenizer failure and the empty-text short-circuit.
6. ``_generate_summary`` sync: call_llm path, temperature-omission, retry, exhaustion.
7. ``_generate_summary_async`` async: the same four shapes.

All tests are hermetic: no network, no real tokenizer, no API keys. The
``transformers`` package isn't installed in this venv, so ``_init_tokenizer``
tests inject a fake ``transformers`` module via ``sys.modules`` rather than
patching a module that does not exist.
"""

import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trajectory_compressor import (
    CompressionConfig,
    TrajectoryMetrics,
    TrajectoryCompressor,
    _effective_temperature_for_model,
)


# ---------------------------------------------------------------------------
# Shared construction helper
# ---------------------------------------------------------------------------


class FALLBACK_SUMMARY:
    """The exact fallback string returned when all summary retries are exhausted."""

    TEXT = (
        "[CONTEXT SUMMARY]: [Summary generation failed - previous turns "
        "contained tool calls and responses that have been compressed to save "
        "context space.]"
    )


def _new_compressor(config=None):
    """Build a compressor via ``__new__`` (skip __init__ network/tokenizer IO)."""
    comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
    comp.config = config if config is not None else CompressionConfig()
    comp.logger = MagicMock()
    return comp


def _summary_response(text):
    """A fake OpenAI chat-completions response carrying ``text`` as content."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _fake_transformers_module(return_value=None, side_effect=None):
    """A stub ``transformers`` module whose AutoTokenizer is a MagicMock.

    ``transformers`` is not installed in the CI venv, so the real module
    cannot be patched. Injecting a fake module lets ``_init_tokenizer``'s
    ``from transformers import AutoTokenizer`` resolve to an injectable mock.
    """
    fake_tf = ModuleType("transformers")
    auto_tokenizer = MagicMock()
    if return_value is not None:
        auto_tokenizer.from_pretrained.return_value = return_value
    if side_effect is not None:
        auto_tokenizer.from_pretrained.side_effect = side_effect
    fake_tf.AutoTokenizer = auto_tokenizer
    return fake_tf, auto_tokenizer


# ---------------------------------------------------------------------------
# _effective_temperature_for_model
# ---------------------------------------------------------------------------


class TestEffectiveTemperatureForModel:
    """``_effective_temperature_for_model`` applies model temperature contracts.

    The helper delegates to ``agent.auxiliary_client._fixed_temperature_for_model``;
    the three import-success outcomes are sensibly distinct, and the
    import-failure fallback must return the requested temperature untouched.
    """

    def test_omit_temperature_sentinel_returns_none(self):
        """Kimi models return the OMIT_TEMPERATURE sentinel -> effective None.

        The caller must drop the ``temperature`` kwarg entirely because the
        provider manages temperature server-side.
        """
        from agent.auxiliary_client import OMIT_TEMPERATURE, _fixed_temperature_for_model

        # The sentinel's identity is a singleton object, not None.
        assert OMIT_TEMPERATURE is not None
        assert _fixed_temperature_for_model("kimi-for-coding") is OMIT_TEMPERATURE
        assert _effective_temperature_for_model("kimi-for-coding", 0.3) is None

    def test_fixed_temperature_override(self):
        """Trinity Large Thinking carries a server-side fixed temperature of 0.5.

        ``_fixed_temperature_for_model`` returns a concrete float, which the
        effective helper forwards verbatim regardless of the requested value.
        """
        from agent.auxiliary_client import _fixed_temperature_for_model

        assert _fixed_temperature_for_model("arcee/trinity-large-thinking") == 0.5
        assert _effective_temperature_for_model("arcee/trinity-large-thinking", 0.3) == 0.5

    def test_no_override_passes_requested_through(self):
        """Models with no fixed contract return None -> requested temperature.

        A plain model (gemini default) has no override, so the requested
        temperature must survive untouched.
        """
        from agent.auxiliary_client import _fixed_temperature_for_model

        assert _fixed_temperature_for_model("google/gemini-3-flash-preview") is None
        assert _effective_temperature_for_model("google/gemini-3-flash-preview", 0.3) == 0.3

    def test_import_failure_falls_back_to_requested(self):
        """If ``agent.auxiliary_client`` cannot be imported, return requested.

        The lazy import lives in a try/except; any exception (simulated here by
        stubbing the module to None) must degrade to passing the requested
        temperature through — safe for callers never expecting to omit it.
        """
        with patch.dict(sys.modules, {"agent.auxiliary_client": None}):
            assert _effective_temperature_for_model("kimi-for-coding", 0.42) == 0.42


# ---------------------------------------------------------------------------
# _init_tokenizer
# ---------------------------------------------------------------------------


class TestInitTokenizer:
    """``_init_tokenizer`` loads a HuggingFace tokenizer or raises a RuntimeError."""

    def test_success_sets_tokenizer_and_prints(self, monkeypatch, capsys):
        """On success the tokenizer attribute is set and a status line prints.

        The fake ``transformers`` module supplies an AutoTokenizer whose
        ``from_pretrained`` returns the mock we assert on.
        """
        comp = _new_compressor()
        tokenizer_mock = MagicMock()
        fake_tf, _ = _fake_transformers_module(return_value=tokenizer_mock)
        monkeypatch.setitem(sys.modules, "transformers", fake_tf)

        comp._init_tokenizer()

        assert comp.tokenizer is tokenizer_mock
        assert "Loaded tokenizer" in capsys.readouterr().out

    def test_failure_raises_runtime_error_with_tokenizer_name(self, monkeypatch):
        """A tokenizer load failure becomes a RuntimeError naming the tokenizer.

        The error message must carry ``config.tokenizer_name`` so users can
        diagnose which model/tokenizer failed to load.
        """
        comp = _new_compressor()
        comp.config.tokenizer_name = "my/pretrained-tok"
        fake_tf, _ = _fake_transformers_module(side_effect=Exception("boom"))
        monkeypatch.setitem(sys.modules, "transformers", fake_tf)

        with pytest.raises(RuntimeError, match="my/pretrained-tok"):
            comp._init_tokenizer()


# ---------------------------------------------------------------------------
# _init_summarizer
# ---------------------------------------------------------------------------


class TestInitSummarizer:
    """``_init_summarizer`` routes to call_llm vs raw client construction."""

    def test_provider_path_uses_call_llm(self, capsys):
        """A known provider base_url enables call_llm routing.

        When ``_detect_provider`` returns a provider (openrouter), the
        compressor delegates to ``resolve_provider_client`` and keeps
        ``client``/``async_client`` as None — they are not used directly.
        """
        comp = _new_compressor()
        comp.config.base_url = "https://openrouter.ai/api/v1"
        rpc = MagicMock(return_value=(MagicMock(), None))

        with patch("agent.auxiliary_client.resolve_provider_client", rpc):
            comp._init_summarizer()

        assert comp._use_call_llm is True
        assert comp._llm_provider == "openrouter"
        assert comp.client is None
        assert comp.async_client is None
        rpc.assert_called_once_with(
            "openrouter", model=comp.config.summarization_model
        )
        assert "Initialized summarizer client" in capsys.readouterr().out

    def test_custom_endpoint_builds_raw_openai_client(self, monkeypatch, capsys):
        """An unknown base_url falls back to a raw OpenAI client.

        The custom-endpoint branch reads the key from ``api_key_env`` and
        constructs an OpenAI client, storing the key for lazy async creation.
        """
        comp = _new_compressor()
        comp.config.base_url = "http://localhost:8000/v1"
        comp.config.api_key_env = "TEST_TC_API_KEY"
        monkeypatch.setenv("TEST_TC_API_KEY", "secret-key")
        openai_mock = MagicMock()

        with patch("openai.OpenAI", openai_mock):
            comp._init_summarizer()

        assert comp._use_call_llm is False
        assert comp.client is openai_mock.return_value
        assert comp.async_client is None
        assert comp._async_client_api_key == "secret-key"
        openai_mock.assert_called_once_with(
            api_key="secret-key", base_url="http://localhost:8000/v1"
        )
        assert "Initialized summarizer client" in capsys.readouterr().out

    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        """A custom endpoint with no key in the env raises a descriptive error.

        Only the custom-endpoint branch demands a key; it must name the env var
        so the operator knows exactly what to export.
        """
        comp = _new_compressor()
        comp.config.base_url = "http://localhost:8000/v1"
        comp.config.api_key_env = "TEST_TC_API_KEY"
        monkeypatch.delenv("TEST_TC_API_KEY", raising=False)

        with patch("openai.OpenAI", MagicMock()):
            with pytest.raises(RuntimeError, match="TEST_TC_API_KEY"):
                comp._init_summarizer()


# ---------------------------------------------------------------------------
# _detect_provider
# ---------------------------------------------------------------------------


class TestDetectProvider:
    """``_detect_provider`` maps a configured base_url to a provider name."""

    @pytest.mark.parametrize(
        "base_url,expected",
        [
            ("https://openrouter.ai/api/v1", "openrouter"),
            ("https://nousresearch.com/api/v1", "nous"),
            ("https://chatgpt.com/backend-api/codex", "codex"),
            ("https://z.ai/api/v1", "zai"),
            ("https://api.moonshot.ai/v1", "kimi-coding"),
            ("https://api.moonshot.cn/v1", "kimi-coding"),
            ("https://api.kimi.com/v1", "kimi-coding"),
            ("https://api.arcee.ai/v1", "arcee"),
            ("https://api.minimaxi.com/v1", "minimax-cn"),
            ("https://api.minimax.io/v1", "minimax"),
            ("http://localhost:9/v1", ""),
        ],
        ids=[
            "openrouter",
            "nous",
            "codex",
            "zai",
            "moonshot-ai-kimi",
            "moonshot-cn-kimi",
            "kimi-com",
            "arcee",
            "minimax-cn",
            "minimax",
            "unknown-localhost",
        ],
    )
    def test_detects_provider_from_base_url(self, base_url, expected):
        """A base_url must resolve to exactly one provider (or '' if unknown).

        Uses the real ``base_url_host_matches``/``base_url_hostname`` helpers
        from ``utils`` (imported by the production module), so a URL fixture
        that fails a host match is a genuine deployment-detect regression.
        """
        comp = _new_compressor()
        comp.config.base_url = base_url
        assert comp._detect_provider() == expected


# ---------------------------------------------------------------------------
# count_tokens fallback
# ---------------------------------------------------------------------------


class TestCountTokensFallback:
    """``count_tokens`` character-approximation fallback and empty short-circuit."""

    def test_empty_text_returns_zero(self):
        """Empty input short-circuits to 0 without touching the tokenizer."""
        comp = _new_compressor()
        comp.tokenizer = MagicMock()
        comp.tokenizer.encode = MagicMock(side_effect=AssertionError("must not call"))
        assert comp.count_tokens("") == 0

    def test_tokenizer_error_falls_back_to_character_estimate(self):
        """A tokenizer failure degrades to ``len(text) // 4``.

        The fallback keeps compression working even when the tokenizer is
        unavailable, at the cost of a coarse approximation.
        """
        comp = _new_compressor()
        comp.tokenizer = MagicMock()
        comp.tokenizer.encode = MagicMock(side_effect=Exception("tokenizer down"))
        assert comp.count_tokens("12345678") == 2


# ---------------------------------------------------------------------------
# _generate_summary (sync)
# ---------------------------------------------------------------------------


class TestGenerateSummarySync:
    """``_generate_summary`` call_llm / raw-client routing, retry and fallback."""

    def test_call_llm_path_forwards_kwargs(self):
        """The call_llm path forwards provider/model/messages/temperature/max_tokens.

        The centralized router (``call_llm``) is the single seam for known
        providers, so the compressor must hand it the exact request it would
        have sent directly and count a single summarization API call.
        """
        comp = _new_compressor()
        comp._use_call_llm = True
        comp._llm_provider = "openrouter"
        call_llm_mock = MagicMock(
            return_value=_summary_response("[CONTEXT SUMMARY]: ok")
        )

        with patch("agent.auxiliary_client.call_llm", call_llm_mock):
            metrics = TrajectoryMetrics()
            result = comp._generate_summary("tool output", metrics)

        assert result == "[CONTEXT SUMMARY]: ok"
        kwargs = call_llm_mock.call_args.kwargs
        assert kwargs["provider"] == "openrouter"
        assert kwargs["model"] == comp.config.summarization_model
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == comp.config.summary_target_tokens * 2
        assert metrics.summarization_api_calls == 1

    def test_raw_client_kimi_omits_temperature(self):
        """Kimi models drop temperature from raw client create kwargs.

        ``_effective_temperature_for_model`` returns None for Kimi, and the
        raw-client branch must leave ``temperature`` out of the request.
        """
        comp = _new_compressor()
        comp.config.summarization_model = "kimi-for-coding"
        comp._use_call_llm = False
        client = MagicMock()
        client.chat.completions.create.return_value = _summary_response(
            "[CONTEXT SUMMARY]: kimi"
        )
        comp.client = client

        metrics = TrajectoryMetrics()
        result = comp._generate_summary("tool output", metrics)

        assert result == "[CONTEXT SUMMARY]: kimi"
        assert "temperature" not in client.chat.completions.create.call_args.kwargs

    def test_raw_client_non_kimi_includes_temperature(self):
        """Non-Kimi models pass the configured temperature into create kwargs.

        This is the complementary branch to the Kimi omission test: a default
        model keeps a concrete temperature value in the request.
        """
        comp = _new_compressor()
        comp._use_call_llm = False
        client = MagicMock()
        client.chat.completions.create.return_value = _summary_response(
            "[CONTEXT SUMMARY]: ok"
        )
        comp.client = client

        metrics = TrajectoryMetrics()
        comp._generate_summary("tool output", metrics)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == comp.config.summary_target_tokens * 2

    def test_retry_recovers_after_one_failure(self):
        """A single transient failure is retried and the summary returned.

        After a failed attempt the compressor sleeps a backoff (mocked to 0),
        retries, and on success must count two API calls, one error, and log a
        warning — while still returning the correct summary.
        """
        comp = _new_compressor()
        comp.config.max_retries = 2
        comp.config.retry_delay = 0
        comp._use_call_llm = False
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("transient"),
            _summary_response("[CONTEXT SUMMARY]: ok"),
        ]
        comp.client = client

        with patch("trajectory_compressor.jittered_backoff", return_value=0):
            metrics = TrajectoryMetrics()
            result = comp._generate_summary("tool output", metrics)

        assert result == "[CONTEXT SUMMARY]: ok"
        assert metrics.summarization_api_calls == 2
        assert metrics.summarization_errors == 1
        comp.logger.warning.assert_called()

    def test_exhaustion_returns_fallback_summary(self):
        """Every attempt failing returns the fixed fallback summary.

        With max_retries=2 and both calls failing, the compressor must count
        two errors and return the canned fallback rather than raising.
        """
        comp = _new_compressor()
        comp.config.max_retries = 2
        comp.config.retry_delay = 0
        comp._use_call_llm = False
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("permanent")
        comp.client = client

        with patch("trajectory_compressor.jittered_backoff", return_value=0):
            metrics = TrajectoryMetrics()
            result = comp._generate_summary("tool output", metrics)

        assert result == FALLBACK_SUMMARY.TEXT
        assert metrics.summarization_api_calls == 2
        assert metrics.summarization_errors == 2


# ---------------------------------------------------------------------------
# _generate_summary_async
# ---------------------------------------------------------------------------


class TestGenerateSummaryAsync:
    """``_generate_summary_async`` mirrors the sync shapes in an event loop."""

    @pytest.mark.asyncio
    async def test_async_call_llm_path(self):
        """The async call_llm path forwards kwargs and counts one API call."""
        comp = _new_compressor()
        comp._use_call_llm = True
        comp._llm_provider = "openrouter"
        async_call_llm_mock = AsyncMock(
            return_value=_summary_response("[CONTEXT SUMMARY]: ok")
        )

        with patch("agent.auxiliary_client.async_call_llm", async_call_llm_mock):
            metrics = TrajectoryMetrics()
            result = await comp._generate_summary_async("tool output", metrics)

        assert result == "[CONTEXT SUMMARY]: ok"
        assert async_call_llm_mock.call_args.kwargs["provider"] == "openrouter"
        assert len(async_call_llm_mock.call_args.kwargs["messages"]) == 1
        assert async_call_llm_mock.call_args.kwargs["temperature"] == 0.3
        assert metrics.summarization_api_calls == 1

    @pytest.mark.asyncio
    async def test_async_raw_client_kimi_omits_temperature(self):
        """Async raw client path omits temperature for Kimi models."""
        comp = _new_compressor()
        comp.config.summarization_model = "kimi-for-coding"
        comp._use_call_llm = False
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(
            return_value=_summary_response("[CONTEXT SUMMARY]: kimi")
        )
        comp._get_async_client = MagicMock(return_value=async_client)

        metrics = TrajectoryMetrics()
        result = await comp._generate_summary_async("tool output", metrics)

        assert result == "[CONTEXT SUMMARY]: kimi"
        assert (
            "temperature"
            not in async_client.chat.completions.create.call_args.kwargs
        )

    @pytest.mark.asyncio
    async def test_async_raw_client_non_kimi_includes_temperature(self):
        """Async raw client path passes temperature for non-Kimi models."""
        comp = _new_compressor()
        comp._use_call_llm = False
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(
            return_value=_summary_response("[CONTEXT SUMMARY]: ok")
        )
        comp._get_async_client = MagicMock(return_value=async_client)

        metrics = TrajectoryMetrics()
        await comp._generate_summary_async("tool output", metrics)

        kwargs = async_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_async_retry_recovers_after_one_failure(self):
        """Async path retries a transient failure and returns the summary."""
        comp = _new_compressor()
        comp.config.max_retries = 2
        comp.config.retry_delay = 0
        comp._use_call_llm = False
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("transient"),
                _summary_response("[CONTEXT SUMMARY]: ok"),
            ]
        )
        comp._get_async_client = MagicMock(return_value=async_client)

        with patch("trajectory_compressor.jittered_backoff", return_value=0):
            metrics = TrajectoryMetrics()
            result = await comp._generate_summary_async("tool output", metrics)

        assert result == "[CONTEXT SUMMARY]: ok"
        assert metrics.summarization_api_calls == 2
        assert metrics.summarization_errors == 1
        comp.logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_async_exhaustion_returns_fallback_summary(self):
        """Async exhaustion returns the canned fallback and counts two errors."""
        comp = _new_compressor()
        comp.config.max_retries = 2
        comp.config.retry_delay = 0
        comp._use_call_llm = False
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(side_effect=Exception("permanent"))
        comp._get_async_client = MagicMock(return_value=async_client)

        with patch("trajectory_compressor.jittered_backoff", return_value=0):
            metrics = TrajectoryMetrics()
            result = await comp._generate_summary_async("tool output", metrics)

        assert result == FALLBACK_SUMMARY.TEXT
        assert metrics.summarization_api_calls == 2
        assert metrics.summarization_errors == 2
