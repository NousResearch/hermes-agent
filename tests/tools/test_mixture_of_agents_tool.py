import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

moa = importlib.import_module("tools.mixture_of_agents_tool")


def test_moa_defaults_are_well_formed():
    # Invariants, not a catalog snapshot: the exact model list churns with
    # OpenRouter availability (see PR #6636 where gemini-3-pro-preview was
    # removed upstream). What we care about is that the defaults are present
    # and valid vendor/model slugs.
    assert isinstance(moa.REFERENCE_MODELS, list)
    assert len(moa.REFERENCE_MODELS) >= 1
    for m in moa.REFERENCE_MODELS:
        assert isinstance(m, str) and "/" in m and not m.startswith("/")
    assert isinstance(moa.AGGREGATOR_MODEL, str)
    assert "/" in moa.AGGREGATOR_MODEL


@pytest.mark.asyncio
async def test_reference_model_retry_warnings_avoid_exc_info_until_terminal_failure(monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=RuntimeError("rate limited"))
            )
        )
    )
    warn = MagicMock()
    err = MagicMock()

    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)
    monkeypatch.setattr(moa.logger, "warning", warn)
    monkeypatch.setattr(moa.logger, "error", err)

    model, message, success = await moa._run_reference_model_safe(
        "openai/gpt-5.4-pro", "hello", max_retries=2
    )

    assert model == "openai/gpt-5.4-pro"
    assert success is False
    assert "failed after 2 attempts" in message
    assert warn.call_count == 2
    assert all(call.kwargs.get("exc_info") is None for call in warn.call_args_list)
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_moa_top_level_error_logs_single_traceback_on_aggregator_failure(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(return_value=("anthropic/claude-opus-4.6", "ok", True)),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(side_effect=RuntimeError("aggregator boom")),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    err = MagicMock()
    monkeypatch.setattr(moa.logger, "error", err)

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "solve this",
            reference_models=["anthropic/claude-opus-4.6"],
        )
    )

    assert result["success"] is False
    assert "Error in MoA processing" in result["error"]
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


# ── Configurability (issue #38952) ─────────────────────────────────────────


def _set_moa_config(monkeypatch, section):
    """Point _get_moa_config() at a fixed ``moa:`` section for one test.

    _get_moa_config() reads live through load_config() (no local cache layer),
    so patching load_config is immediately reflected — no cache to clear.
    """
    monkeypatch.setattr(moa, "load_config", lambda: {"moa": section})


def test_get_moa_config_reads_moa_section(monkeypatch):
    # The resolver pulls the moa: section from the standard loader, not a
    # bespoke reader — patching load_config must be reflected in _get_moa_config.
    monkeypatch.setattr(moa, "load_config", lambda: {"moa": {"aggregator_model": "vendor/agg"}})
    assert moa._get_moa_config() == {"aggregator_model": "vendor/agg"}


def test_get_moa_config_empty_on_loader_failure(monkeypatch):
    # A config load failure must not break MoA — it falls back to module defaults.
    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(moa, "load_config", _boom)
    assert moa._get_moa_config() == {}


def test_get_moa_config_reflects_config_change_between_calls(monkeypatch):
    # Regression for the dropped @lru_cache(maxsize=1) staleness bug: a
    # long-lived process that serves sessions with different config (e.g. a new
    # HERMES_HOME) must see the SECOND config, not the first one pinned forever.
    monkeypatch.setattr(moa, "load_config", lambda: {"moa": {"aggregator_model": "first/agg"}})
    assert moa._get_moa_config() == {"aggregator_model": "first/agg"}

    # Swap the underlying config (simulates a HERMES_HOME / profile switch) and
    # confirm the next read reflects it — no stale process-lifetime cache.
    monkeypatch.setattr(moa, "load_config", lambda: {"moa": {"aggregator_model": "second/agg"}})
    assert moa._get_moa_config() == {"aggregator_model": "second/agg"}


def test_resolve_precedence_call_arg_over_config_over_default(monkeypatch):
    _set_moa_config(monkeypatch, {"aggregator_model": "cfg/model"})
    # call-arg beats config
    assert moa._resolve("arg/model", "aggregator_model", "default/model") == "arg/model"
    # config beats default when no call-arg
    assert moa._resolve(None, "aggregator_model", "default/model") == "cfg/model"
    # empty call-arg / empty config fall through to default
    assert moa._resolve("", "missing_key", "default/model") == "default/model"
    assert moa._resolve([], "missing_key", ["a"]) == ["a"]


def test_reasoning_extra_body_guard_omits_when_disabled():
    # Falsy or "none" => no extra_body block (LiteLLM / local models reject it).
    assert moa._reasoning_extra_body("") == {}
    assert moa._reasoning_extra_body("none") == {}
    assert moa._reasoning_extra_body("NONE") == {}
    assert moa._reasoning_extra_body(None) == {}
    # A real effort produces the OpenRouter reasoning payload.
    body = moa._reasoning_extra_body("xhigh")
    assert body["extra_body"]["reasoning"]["effort"] == "xhigh"
    assert body["extra_body"]["reasoning"]["enabled"] is True


@pytest.mark.asyncio
async def test_config_models_and_reasoning_reach_runners(monkeypatch):
    # End-to-end: a moa: config with no call-args must drive the models,
    # temperatures, and reasoning effort actually passed to the runners.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _set_moa_config(
        monkeypatch,
        {
            "reference_models": ["cfg/ref-a", "cfg/ref-b"],
            "aggregator_model": "cfg/agg",
            "reference_temperature": 0.9,
            "aggregator_temperature": 0.2,
            "reasoning_effort": "none",
        },
    )

    ref_mock = AsyncMock(return_value=("cfg/ref-a", "ok", True))
    agg_mock = AsyncMock(return_value="final answer")
    monkeypatch.setattr(moa, "_run_reference_model_safe", ref_mock)
    monkeypatch.setattr(moa, "_run_aggregator_model", agg_mock)
    monkeypatch.setattr(
        moa, "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(await moa.mixture_of_agents_tool("solve this"))

    assert result["success"] is True
    assert result["models_used"]["reference_models"] == ["cfg/ref-a", "cfg/ref-b"]
    assert result["models_used"]["aggregator_model"] == "cfg/agg"

    # Both config reference models were dispatched with the configured temp
    # and reasoning effort.
    assert ref_mock.call_count == 2
    ref_models_called = {c.args[0] for c in ref_mock.call_args_list}
    assert ref_models_called == {"cfg/ref-a", "cfg/ref-b"}
    first_ref = ref_mock.call_args_list[0]
    assert first_ref.args[2] == 0.9  # reference_temperature
    assert first_ref.kwargs["reasoning_effort"] == "none"

    # Aggregator received the configured model, temperature, and effort.
    assert agg_mock.call_args.kwargs["aggregator_model"] == "cfg/agg"
    assert agg_mock.call_args.args[2] == 0.2  # aggregator_temperature
    assert agg_mock.call_args.kwargs["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_call_arg_overrides_config(monkeypatch):
    # An explicit call-arg must win over the moa: config (precedence top tier).
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _set_moa_config(monkeypatch, {"aggregator_model": "cfg/agg"})

    ref_mock = AsyncMock(return_value=("arg/ref", "ok", True))
    agg_mock = AsyncMock(return_value="final")
    monkeypatch.setattr(moa, "_run_reference_model_safe", ref_mock)
    monkeypatch.setattr(moa, "_run_aggregator_model", agg_mock)
    monkeypatch.setattr(
        moa, "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "q",
            reference_models=["arg/ref"],
            aggregator_model="arg/agg",
        )
    )

    assert result["models_used"]["aggregator_model"] == "arg/agg"
    assert agg_mock.call_args.kwargs["aggregator_model"] == "arg/agg"
    assert ref_mock.call_count == 1
    assert ref_mock.call_args.args[0] == "arg/ref"


@pytest.mark.asyncio
async def test_no_config_matches_legacy_defaults(monkeypatch):
    # With an empty moa: section, the tool must behave exactly as before:
    # module-constant models, default temperatures, default reasoning effort.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _set_moa_config(monkeypatch, {})

    ref_mock = AsyncMock(return_value=("x", "ok", True))
    agg_mock = AsyncMock(return_value="final")
    monkeypatch.setattr(moa, "_run_reference_model_safe", ref_mock)
    monkeypatch.setattr(moa, "_run_aggregator_model", agg_mock)
    monkeypatch.setattr(
        moa, "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(await moa.mixture_of_agents_tool("q"))

    assert result["models_used"]["reference_models"] == moa.REFERENCE_MODELS
    assert result["models_used"]["aggregator_model"] == moa.AGGREGATOR_MODEL
    assert ref_mock.call_count == len(moa.REFERENCE_MODELS)
    # Default reference temperature + reasoning effort were used.
    assert ref_mock.call_args_list[0].args[2] == moa.REFERENCE_TEMPERATURE
    assert ref_mock.call_args_list[0].kwargs["reasoning_effort"] == "xhigh"
    assert agg_mock.call_args.kwargs["aggregator_model"] == moa.AGGREGATOR_MODEL


def test_registration_handler_forwards_model_overrides(monkeypatch):
    # Regression guard for the registration lambda that previously stripped
    # reference_models / aggregator_model: the handler must forward both so
    # caller-supplied overrides reach mixture_of_agents_tool().
    from tools.registry import registry

    captured = {}

    def _fake_tool(**kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(moa, "mixture_of_agents_tool", _fake_tool)

    handler = registry.get_entry("mixture_of_agents").handler
    result = handler({
        "user_prompt": "hi",
        "reference_models": ["a/b"],
        "aggregator_model": "c/d",
    })

    assert result == "sentinel"
    assert captured["user_prompt"] == "hi"
    assert captured["reference_models"] == ["a/b"]
    assert captured["aggregator_model"] == "c/d"


# ── Adversarial-review regressions ─────────────────────────────────────────


def _empty_content_response():
    """A chat-completions response that extract_content_or_reasoning sees as empty.

    content=None with no reasoning/reasoning_content/reasoning_details fields →
    extract_content_or_reasoning returns "". Mirrors a reasoning-only / blank
    upstream response.
    """
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )


@pytest.mark.asyncio
async def test_empty_content_on_last_attempt_returns_failure_not_blank_success(monkeypatch):
    # Bug: when the FINAL attempt yielded empty content, the code fell through
    # to `return model, content, True` — a "success" carrying "" that the
    # aggregator would then ingest as a blank reference. The last attempt must
    # instead fail explicitly. max_retries=1 makes the first attempt the last,
    # so there is no retry/backoff sleep to wait on.
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_empty_content_response())
            )
        )
    )
    # Patch accepts base_url=/api_key= kwargs (the runner always passes them).
    monkeypatch.setattr(moa, "_get_openrouter_client", lambda *a, **k: fake_client)

    model, message, success = await moa._run_reference_model_safe(
        "vendor/reasoner", "hello", max_retries=1
    )

    assert model == "vendor/reasoner"
    assert success is False           # NOT a blank success
    assert message != ""              # carries a diagnostic, not empty content
    assert "empty content" in message


@pytest.mark.asyncio
async def test_empty_content_retries_then_fails_across_attempts(monkeypatch):
    # The non-last-attempt path must still `continue` (retry), and only the
    # terminal attempt converts the empty content into a failure. Patch sleep so
    # the backoff between attempts is instant.
    create = AsyncMock(return_value=_empty_content_response())
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(moa, "_get_openrouter_client", lambda *a, **k: fake_client)

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(moa.asyncio, "sleep", _no_sleep)

    model, message, success = await moa._run_reference_model_safe(
        "vendor/reasoner", "hello", max_retries=3
    )

    assert success is False
    assert create.await_count == 3    # all attempts were made (retried, not short-circuited)
    assert "after 3 attempts" in message


def test_check_moa_requirements_true_with_base_url_no_openrouter_key(monkeypatch):
    # Headline use case: a local / LiteLLM deploy sets moa.base_url and has NO
    # OPENROUTER_API_KEY. The tool must report itself available (gating it on the
    # OpenRouter key made it silently disappear for exactly this deployment).
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(moa, "load_config", lambda: {"moa": {"base_url": "http://localhost:4000/v1"}})

    assert moa.check_moa_requirements() is True


def test_check_moa_requirements_false_with_no_key_and_no_base_url(monkeypatch):
    # The negative: with neither an OpenRouter key nor a base_url, the default
    # path has no usable auth and the tool must report itself unavailable.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(moa, "load_config", lambda: {"moa": {}})

    assert moa.check_moa_requirements() is False


def test_check_moa_requirements_true_with_openrouter_key_and_no_base_url(monkeypatch):
    # The default OpenRouter path stays intact: a key alone (no base_url) is
    # still sufficient — the base_url branch is additive, not a replacement.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(moa, "load_config", lambda: {"moa": {}})

    assert moa.check_moa_requirements() is True
