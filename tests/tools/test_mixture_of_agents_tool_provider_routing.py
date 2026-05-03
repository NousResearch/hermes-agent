import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import hermes_cli.models as models_mod
from hermes_cli.config import DEFAULT_CONFIG, get_config_path
from hermes_cli.model_normalize import normalize_model_for_provider

import tools.mixture_of_agents_tool as moa


VALID_PROVIDER_IDS = [
    "openrouter",
    "nous",
    "openai-codex",
    "lmstudio",
    "google-gemini-cli",
    "copilot",
    "copilot-acp",
    "gemini",
    "huggingface",
    "zai",
    "kimi-coding",
    "kimi-coding-cn",
    "minimax",
    "minimax-cn",
    "kilocode",
    "anthropic",
    "alibaba",
    "qwen-oauth",
    "xiaomi",
    "tencent-tokenhub",
    "opencode-zen",
    "opencode-go",
    "ai-gateway",
    "deepseek",
    "arcee",
    "xai",
    "nvidia",
    "ollama-cloud",
    "bedrock",
    "custom",
]


@pytest.fixture(autouse=True)
def _clear_moa_state():
    for name in ("_TEMPERATURE_UNSUPPORTED", "_REASONING_UNSUPPORTED", "_codex_warning_seen"):
        value = getattr(moa, name, None)
        if isinstance(value, (dict, set)):
            value.clear()


@pytest.fixture(autouse=True)
def _stable_provider_registry(monkeypatch):
    monkeypatch.setattr(
        models_mod,
        "list_available_providers",
        lambda: [{"id": provider_id} for provider_id in VALID_PROVIDER_IDS],
    )


def _write_config(data: dict):
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def fake_catalogs(monkeypatch):
    catalogs = {
        "openrouter": [
            "anthropic/claude-opus-4.7",
            "google/gemini-2.5-pro",
            "openai/gpt-5.5-pro",
            "deepseek/deepseek-v3.2",
            "qwen/qwen3.5-plus-02-15",
            "minimax/minimax-m2.5",
        ],
        "anthropic": ["claude-opus-4-6"],
        "openai-codex": ["gpt-5.4"],
        "copilot": ["gpt-5.4", "gpt-4.1"],
        "copilot-acp": ["gpt-5.4"],
        "nous": ["minimax/minimax-m2.5"],
        "ai-gateway": ["anthropic/claude-opus-4.7"],
        "lmstudio": ["local-reasoner"],
        "tencent-tokenhub": ["hy3-preview"],
        "custom": [],
    }

    def _provider_model_ids(provider, *, force_refresh=False):
        return list(catalogs.get(provider, []))

    monkeypatch.setattr(models_mod, "provider_model_ids", _provider_model_ids)
    return catalogs


@pytest.mark.parametrize("reasoning_value", [None, "", "   "])
def test_load_moa_config_absent_or_blank_reasoning_uses_defaults(fake_catalogs, reasoning_value):
    _write_config(
        {
            "moa": {
                "reference_models": [
                    {
                        "model": "anthropic/claude-opus-4.7",
                        "reasoning": reasoning_value,
                    }
                ],
                "aggregator_model": {"model": "anthropic/claude-opus-4.7"},
            }
        }
    )

    loaded = moa._load_moa_config()

    assert loaded["enabled"] is True
    assert loaded["reference_models"][0]["provider"] == "openrouter"
    assert loaded["reference_models"][0]["reasoning_config"] is None
    assert loaded["aggregator_model"]["provider"] == "openrouter"
    assert loaded["aggregator_model"]["reasoning_config"] is None
    assert loaded["min_successful_references"] == 1


def test_load_moa_config_defaults_when_block_absent(fake_catalogs):
    _write_config({"model": "anthropic/claude-opus-4.7"})

    loaded = moa._load_moa_config()

    assert [entry["model"] for entry in loaded["reference_models"]] == moa.REFERENCE_MODELS
    assert all(entry["provider"] == "openrouter" for entry in loaded["reference_models"])
    assert all(entry["reasoning_config"] == {"enabled": True, "effort": "xhigh"} for entry in loaded["reference_models"])
    assert loaded["aggregator_model"]["model"] == moa.AGGREGATOR_MODEL
    assert loaded["aggregator_model"]["provider"] == "openrouter"
    assert loaded["aggregator_model"]["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
    assert loaded["reference_temperature"] == moa.REFERENCE_TEMPERATURE
    assert loaded["aggregator_temperature"] == moa.AGGREGATOR_TEMPERATURE
    assert loaded["min_successful_references"] == 2


def test_load_moa_config_accepts_string_shorthand(fake_catalogs):
    _write_config(
        {
            "moa": {
                "reference_models": ["gpt-5.4"],
                "aggregator_model": "anthropic/claude-opus-4.7",
            }
        }
    )

    loaded = moa._load_moa_config()

    assert loaded["reference_models"][0]["model"] == "gpt-5.4"
    assert loaded["reference_models"][0]["provider"] == "openrouter"
    assert loaded["reference_models"][0]["reasoning_config"] is None
    assert loaded["aggregator_model"]["model"] == "anthropic/claude-opus-4.7"
    assert loaded["aggregator_model"]["provider"] == "openrouter"


def test_load_moa_config_preserves_dict_entries(fake_catalogs):
    _write_config(
        {
            "moa": {
                "reference_models": [
                    {
                        "model": "gpt-5.4",
                        "provider": "openai-codex",
                        "reasoning": "high",
                    }
                ],
                "aggregator_model": {
                    "model": "claude-opus-4-6",
                    "provider": "anthropic",
                    "reasoning": "none",
                },
            }
        }
    )

    loaded = moa._load_moa_config()

    assert loaded["reference_models"][0] == {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "reasoning_config": {"enabled": True, "effort": "high"},
    }
    assert loaded["aggregator_model"] == {
        "model": "claude-opus-4-6",
        "provider": "anthropic",
        "reasoning_config": {"enabled": False},
    }


def test_load_moa_config_normalizes_provider_aliases(fake_catalogs):
    _write_config(
        {
            "moa": {
                "reference_models": [{"model": "hy3-preview", "provider": "tokenhub"}],
                "aggregator_model": {"model": "local-reasoner", "provider": "lm-studio"},
            }
        }
    )

    loaded = moa._load_moa_config()

    assert loaded["reference_models"][0]["provider"] == "tencent-tokenhub"
    assert loaded["aggregator_model"]["provider"] == "lmstudio"


def test_load_moa_config_accepts_named_custom_provider_slug(fake_catalogs):
    _write_config(
        {
            "custom_providers": [
                {
                    "name": "internal",
                    "base_url": "http://localhost:1234/v1",
                    "api_key": "test",
                }
            ],
            "moa": {
                "reference_models": [{"model": "my-model", "provider": "custom:internal"}],
                "aggregator_model": {"model": "my-model", "provider": "custom:internal"},
            },
        }
    )

    loaded = moa._load_moa_config()

    assert loaded["reference_models"][0]["provider"] == "custom:internal"
    assert loaded["aggregator_model"]["provider"] == "custom:internal"


def test_load_moa_config_rejects_unknown_provider(fake_catalogs):
    _write_config(
        {
            "moa": {
                "reference_models": [
                    {"model": "anthropic/claude-opus-4.7", "provider": "totally-made-up"}
                ],
                "aggregator_model": {"model": "anthropic/claude-opus-4.7"},
            }
        }
    )

    with pytest.raises(ValueError, match="unknown provider"):
        moa._load_moa_config()


@pytest.mark.parametrize("reasoning", ["extreme", "max"])
def test_load_moa_config_rejects_invalid_reasoning(fake_catalogs, reasoning):
    _write_config(
        {
            "moa": {
                "reference_models": [
                    {"model": "anthropic/claude-opus-4.7", "reasoning": reasoning}
                ],
                "aggregator_model": {"model": "anthropic/claude-opus-4.7"},
            }
        }
    )

    with pytest.raises(ValueError) as exc_info:
        moa._load_moa_config()

    message = str(exc_info.value)
    for effort in ("minimal", "low", "medium", "high", "xhigh", "none"):
        assert effort in message
    assert "max" not in message.replace(repr(reasoning), "")


@pytest.mark.parametrize(
    ("config", "expected_match"),
    [
        (
            {
                "moa": {
                    "reference_models": [],
                    "aggregator_model": {"model": "anthropic/claude-opus-4.7"},
                }
            },
            "reference_models",
        ),
        (
            {
                "moa": {
                    "reference_models": ["anthropic/claude-opus-4.7"],
                    "aggregator_model": None,
                }
            },
            "aggregator_model",
        ),
        (
            {
                "moa": {
                    "reference_models": ["anthropic/claude-opus-4.7"],
                    "aggregator_model": {},
                }
            },
            "aggregator_model",
        ),
        (
            {
                "moa": {
                    "reference_models": [{}],
                    "aggregator_model": {"model": "anthropic/claude-opus-4.7"},
                }
            },
            "reference_models",
        ),
    ],
)
def test_load_moa_config_rejects_invalid_shapes(fake_catalogs, config, expected_match):
    _write_config(config)

    with pytest.raises(ValueError, match=expected_match):
        moa._load_moa_config()


@pytest.mark.parametrize(
    ("roster", "explicit", "expected"),
    [
        (["a", "b", "c"], None, 2),
        (["a"], None, 1),
        (["a", "b", "c"], 1, 1),
    ],
)
def test_min_successful_references_default_and_override(fake_catalogs, roster, explicit, expected):
    config = {
        "moa": {
            "reference_models": roster,
            "aggregator_model": "anthropic/claude-opus-4.7",
        }
    }
    if explicit is not None:
        config["moa"]["min_successful_references"] = explicit
    _write_config(config)

    loaded = moa._load_moa_config()

    assert loaded["min_successful_references"] == expected


@pytest.mark.parametrize("value", [0, 3, 1.5, True])
def test_min_successful_references_out_of_range(fake_catalogs, value):
    _write_config(
        {
            "moa": {
                "reference_models": ["anthropic/claude-opus-4.7", "gpt-5.4"],
                "aggregator_model": "anthropic/claude-opus-4.7",
                "min_successful_references": value,
            }
        }
    )

    with pytest.raises(ValueError, match="min_successful_references"):
        moa._load_moa_config()


def test_model_catalog_mismatch_warns_but_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        models_mod,
        "provider_model_ids",
        lambda provider, *, force_refresh=False: ["gpt-5.4"],
    )
    _write_config(
        {
            "moa": {
                "reference_models": [{"model": "not-in-catalog", "provider": "openai-codex"}],
                "aggregator_model": {"model": "gpt-5.4", "provider": "openai-codex"},
            }
        }
    )

    with patch.object(moa.logger, "warning") as warn:
        loaded = moa._load_moa_config(emit_warnings=True)

    assert loaded["reference_models"][0]["model"] == "not-in-catalog"
    warn.assert_any_call(
        "MoA: model %r not in %s catalog — may fail at call time",
        "not-in-catalog",
        "openai-codex",
    )


@pytest.mark.asyncio
async def test_kill_switch_short_circuits_fail_closed(fake_catalogs, monkeypatch):
    _write_config(
        {
            "moa": {
                "enabled": False,
                "reference_models": ["anthropic/claude-opus-4.7"],
                "aggregator_model": "anthropic/claude-opus-4.7",
            }
        }
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(await moa.mixture_of_agents_tool("solve this"))

    assert result == {
        "success": False,
        "response": "",
        "models_used": {"reference_models": [], "aggregator_model": ""},
        "error": "MoA disabled via moa.enabled=false",
    }
    assert moa.check_moa_requirements() is False


def test_check_moa_requirements_is_config_aware(fake_catalogs, monkeypatch):
    _write_config(
        {
            "moa": {
                "reference_models": [{"model": "gpt-5.4", "provider": "openai-codex"}],
                "aggregator_model": {"model": "gpt-5.4", "provider": "openai-codex"},
            }
        }
    )
    monkeypatch.setattr(
        moa,
        "_provider_has_credentials",
        lambda provider, model=None: provider == "openai-codex",
    )

    assert moa.check_moa_requirements() is True


def test_check_moa_requirements_reports_missing_provider(fake_catalogs, monkeypatch, caplog):
    _write_config(
        {
            "moa": {
                "reference_models": ["anthropic/claude-opus-4.7"],
                "aggregator_model": "anthropic/claude-opus-4.7",
            }
        }
    )
    monkeypatch.setattr(moa, "_provider_has_credentials", lambda provider, model=None: False)

    available, hint = moa.get_moa_preflight_status()

    assert available is False
    assert hint == "credentials for openrouter"
    assert "openrouter" in caplog.text.lower()


def test_custom_provider_slug_does_not_bypass_preflight(fake_catalogs, monkeypatch):
    _write_config(
        {
            "custom_providers": [
                {
                    "name": "internal",
                    "base_url": "http://localhost:1234/v1",
                    "api_key": "test",
                }
            ],
            "moa": {
                "reference_models": [{"model": "my-model", "provider": "custom:internal"}],
                "aggregator_model": {"model": "my-model", "provider": "custom:internal"},
            },
        }
    )
    monkeypatch.setattr(moa, "resolve_provider_client", lambda *args, **kwargs: (None, "my-model"))

    available, hint = moa.get_moa_preflight_status()

    assert available is False
    assert hint == "credentials for custom:internal"


def test_preflight_uses_resolver_and_closes_clients(fake_catalogs, monkeypatch):
    _write_config(
        {
            "moa": {
                "reference_models": [{"model": "gpt-5.4", "provider": "openai-codex"}],
                "aggregator_model": {"model": "claude-opus-4-6", "provider": "anthropic"},
            }
        }
    )
    client = SimpleNamespace(close=MagicMock())
    calls = []

    def _resolve(provider, model=None, async_mode=False):
        calls.append((provider, model, async_mode))
        return client, model

    monkeypatch.setattr(moa, "resolve_provider_client", _resolve)

    available, hint = moa.get_moa_preflight_status()

    assert available is True
    assert hint is None
    assert calls == [
        ("openai-codex", "gpt-5.4", False),
        ("anthropic", "claude-opus-4-6", False),
    ]
    assert client.close.call_count == 2


@pytest.mark.asyncio
async def test_create_chat_completion_supports_sync_create():
    calls = []

    def _create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    response = await moa._create_chat_completion(client, model="m", messages=[])

    assert response.choices[0].message.content == "ok"
    assert calls == [{"model": "m", "messages": []}]


@pytest.mark.asyncio
async def test_create_chat_completion_awaits_sync_wrapper_awaitable_result():
    calls = []

    async def _response():
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    def _create(**kwargs):
        calls.append(kwargs)
        return _response()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    response = await moa._create_chat_completion(client, model="m", messages=[])

    assert response.choices[0].message.content == "ok"
    assert calls == [{"model": "m", "messages": []}]


@pytest.mark.parametrize(
    ("provider", "model", "reasoning_config", "expected"),
    [
        ("openrouter", "anthropic/claude-opus-4.7", None, {}),
        (
            "openrouter",
            "qwen/qwen3.5-plus-02-15",
            {"enabled": True, "effort": "xhigh"},
            {"extra_body": {"reasoning": {"enabled": True, "effort": "xhigh"}}},
        ),
        (
            "openrouter",
            "tencent/hy3-preview",
            {"enabled": True, "effort": "high"},
            {"extra_body": {"reasoning": {"enabled": True, "effort": "high"}}},
        ),
        ("openrouter", "minimax/minimax-m2.5", {"enabled": True, "effort": "high"}, {}),
        ("nous", "minimax/minimax-m2.5", {"enabled": False}, {}),
        (
            "ai-gateway",
            "anthropic/claude-opus-4.7",
            {"enabled": False},
            {"extra_body": {"reasoning": {"enabled": False}}},
        ),
        (
            "copilot",
            "gpt-5.4",
            {"enabled": True, "effort": "xhigh"},
            {"extra_body": {"reasoning": {"effort": "high"}}},
        ),
        (
            "copilot",
            "gpt-5.4",
            {"enabled": True, "effort": "minimal"},
            {"extra_body": {"reasoning": {"effort": "low"}}},
        ),
        ("copilot", "gpt-5.4", {"enabled": False}, {}),
        ("copilot", "gpt-4.1", {"enabled": True, "effort": "high"}, {}),
        (
            "custom",
            "gpt-5.4",
            {"enabled": True, "effort": "high"},
            {"reasoning_effort": "high"},
        ),
        (
            "custom",
            "gpt-5.4",
            {"enabled": False},
            {"extra_body": {"think": False}},
        ),
        (
            "kimi-coding",
            "kimi-k2.6",
            {"enabled": True, "effort": "minimal"},
            {
                "reasoning_effort": "low",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        ),
        (
            "kimi-coding",
            "kimi-k2.6",
            {"enabled": True, "effort": "xhigh"},
            {
                "reasoning_effort": "high",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        ),
        (
            "kimi-coding-cn",
            "kimi-k2.6",
            {"enabled": False},
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "tencent-tokenhub",
            "hy3-preview",
            {"enabled": True, "effort": "medium"},
            {"reasoning_effort": "medium"},
        ),
        (
            "tencent-tokenhub",
            "hy3-preview",
            {"enabled": True, "effort": "xhigh"},
            {"reasoning_effort": "high"},
        ),
        ("tencent-tokenhub", "hy3-preview", {"enabled": False}, {}),
        (
            "openai-codex",
            "gpt-5.4",
            {"enabled": True, "effort": "high"},
            {"reasoning_config": {"enabled": True, "effort": "high"}},
        ),
        (
            "anthropic",
            "claude-opus-4-6",
            {"enabled": True, "effort": "high"},
            {"reasoning_config": {"enabled": True, "effort": "high"}},
        ),
    ],
)
def test_reasoning_kwargs_translation(monkeypatch, provider, model, reasoning_config, expected):
    efforts = {"gpt-5.4": ["low", "medium", "high"], "gpt-4.1": []}
    monkeypatch.setattr(
        models_mod,
        "github_model_reasoning_efforts",
        lambda model_id, catalog=None, api_key=None: list(efforts.get(model_id, [])),
    )

    assert moa._reasoning_kwargs(provider, model, reasoning_config) == expected


@pytest.mark.parametrize(
    ("options", "reasoning_config", "expected"),
    [
        ([], {"enabled": True, "effort": "high"}, {}),
        (["off"], {"enabled": True, "effort": "high"}, {}),
        (["off", "low", "medium"], {"enabled": True, "effort": "high"}, {}),
        (["off", "low", "medium"], {"enabled": True, "effort": "low"}, {"reasoning_effort": "low"}),
        (["off", "on"], {"enabled": True, "effort": "medium"}, {"reasoning_effort": "medium"}),
    ],
)
def test_reasoning_kwargs_lmstudio_uses_published_options(monkeypatch, options, reasoning_config, expected):
    monkeypatch.setattr(
        models_mod,
        "lmstudio_model_reasoning_options",
        lambda model, base_url=None, api_key=None: list(options),
    )

    assert moa._reasoning_kwargs(
        "lmstudio",
        "local-reasoner",
        reasoning_config,
        base_url="http://127.0.0.1:1234/v1",
        api_key="test",
    ) == expected


def test_request_cache_key_includes_endpoint_port():
    assert moa._request_cache_key(
        "lmstudio",
        "local-reasoner",
        "http://localhost:1234/v1",
    ) == ("lmstudio", "local-reasoner", "localhost:1234")
    assert moa._request_cache_key(
        "lmstudio",
        "local-reasoner",
        "http://localhost:5678/v1",
    ) == ("lmstudio", "local-reasoner", "localhost:5678")


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


@pytest.fixture
def agent_for_parity():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.auxiliary_client.OpenAI"),
    ):
        agent = AIAgent(
            base_url="https://example.test/v1",
            api_key="test-key-1234567890",
            provider="custom",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        yield agent


def _agent_reasoning_view(agent, provider: str) -> dict:
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
    if provider in {"openrouter", "nous", "copilot", "ai-gateway"}:
        reasoning = kwargs.get("extra_body", {}).get("reasoning")
        return {"extra_body": {"reasoning": reasoning}} if reasoning is not None else {}
    return {}


@pytest.mark.parametrize(
    ("provider", "base_url", "model", "reasoning_config", "supported_efforts"),
    [
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            "qwen/qwen3.5-plus-02-15",
            {"enabled": True, "effort": "xhigh"},
            [],
        ),
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            "minimax/minimax-m2.5",
            {"enabled": True, "effort": "high"},
            [],
        ),
        (
            "nous",
            "https://inference-api.nousresearch.com/v1",
            "minimax/minimax-m2.5",
            {"enabled": False},
            [],
        ),
        (
            "copilot",
            "https://api.githubcopilot.com",
            "gpt-5.4",
            {"enabled": True, "effort": "xhigh"},
            ["low", "medium", "high"],
        ),
        (
            "ai-gateway",
            "https://ai-gateway.vercel.sh/v1",
            "anthropic/claude-opus-4.7",
            {"enabled": False},
            [],
        ),
    ],
)
def test_reasoning_translation_matches_main_agent(
    monkeypatch,
    agent_for_parity,
    provider,
    base_url,
    model,
    reasoning_config,
    supported_efforts,
):
    monkeypatch.setattr(
        models_mod,
        "github_model_reasoning_efforts",
        lambda model_id, catalog=None, api_key=None: list(supported_efforts),
    )

    agent_for_parity.provider = provider
    agent_for_parity.base_url = base_url
    agent_for_parity._base_url_lower = base_url.lower()
    agent_for_parity.model = model
    agent_for_parity.reasoning_config = reasoning_config

    assert moa._reasoning_kwargs(provider, model, reasoning_config) == _agent_reasoning_view(
        agent_for_parity,
        provider,
    )


def test_omitted_moa_reasoning_intentionally_differs_from_main_agent_default(agent_for_parity):
    agent_for_parity.provider = "openrouter"
    agent_for_parity.base_url = "https://openrouter.ai/api/v1"
    agent_for_parity._base_url_lower = agent_for_parity.base_url.lower()
    agent_for_parity.model = "anthropic/claude-opus-4.7"
    agent_for_parity.reasoning_config = None

    assert moa._reasoning_kwargs("openrouter", "anthropic/claude-opus-4.7", None) == {}
    assert _agent_reasoning_view(agent_for_parity, "openrouter") == {
        "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
    }


def test_build_api_params_omits_temperature_for_kimi_model():
    params = moa._build_api_params(
        "kimi-k2.6",
        [{"role": "user", "content": "hi"}],
        "kimi-coding",
        {"enabled": True, "effort": "medium"},
        0.6,
        moa._request_cache_key("kimi-coding", "kimi-k2.6", "https://api.kimi.com/coding"),
        base_url="https://api.kimi.com/coding",
        max_tokens=32000,
    )

    assert "temperature" not in params
    assert params["max_tokens"] == 32000
    assert params["reasoning_effort"] == "medium"


def test_build_api_params_omits_temperature_for_anthropic_no_sampling_model():
    params = moa._build_api_params(
        "claude-opus-4-7",
        [{"role": "user", "content": "hi"}],
        "anthropic",
        None,
        0.4,
        moa._request_cache_key("anthropic", "claude-opus-4-7", "https://api.anthropic.com"),
        base_url="https://api.anthropic.com",
    )

    assert "temperature" not in params


def test_build_api_params_direct_openai_custom_uses_max_completion_tokens():
    params = moa._build_api_params(
        "gpt-5.4",
        [{"role": "user", "content": "hi"}],
        "custom",
        None,
        0.4,
        moa._request_cache_key("custom", "gpt-5.4", "https://api.openai.com/v1"),
        base_url="https://api.openai.com/v1",
        max_tokens=123,
    )

    assert params["max_completion_tokens"] == 123
    assert "max_tokens" not in params


def test_build_api_params_nous_adds_product_tag_and_omits_disabled_reasoning():
    params = moa._build_api_params(
        "minimax/minimax-m2.5",
        [{"role": "user", "content": "hi"}],
        "nous",
        {"enabled": False},
        0.4,
        moa._request_cache_key("nous", "minimax/minimax-m2.5", "https://inference-api.nousresearch.com/v1"),
        base_url="https://inference-api.nousresearch.com/v1",
    )

    assert params["extra_body"] == {"tags": ["product=hermes-agent"]}


def test_build_api_params_openrouter_keeps_max_tokens():
    params = moa._build_api_params(
        "anthropic/claude-opus-4.7",
        [{"role": "user", "content": "hi"}],
        "openrouter",
        None,
        0.4,
        moa._request_cache_key("openrouter", "anthropic/claude-opus-4.7", "https://openrouter.ai/api/v1"),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=456,
    )

    assert params["max_tokens"] == 456
    assert "max_completion_tokens" not in params


@pytest.mark.asyncio
async def test_reference_model_forwards_default_max_tokens_and_reasoning(monkeypatch):
    calls = []

    async def _create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    monkeypatch.setattr(
        moa,
        "resolve_provider_client",
        lambda *args, **kwargs: (fake_client, "anthropic/claude-opus-4.7"),
    )
    monkeypatch.setattr(
        moa,
        "extract_content_or_reasoning",
        lambda response: response.choices[0].message.content,
    )

    model_name, content, success = await moa._run_reference_model_safe(
        {
            "model": "anthropic/claude-opus-4.7",
            "provider": "openrouter",
            "reasoning_config": {"enabled": True, "effort": "xhigh"},
        },
        "hello",
        max_retries=1,
    )

    assert (model_name, content, success) == (
        "anthropic/claude-opus-4.7",
        "ok",
        True,
    )
    assert calls[0]["max_tokens"] == 32000
    assert calls[0]["extra_body"] == {
        "reasoning": {"enabled": True, "effort": "xhigh"}
    }
    assert "temperature" not in calls[0]


@pytest.mark.asyncio
async def test_structured_unsupported_reasoning_retries_and_caches_by_endpoint(monkeypatch):
    calls = []

    class FakeError(Exception):
        def __init__(self):
            super().__init__("Unsupported parameter: reasoning_effort")
            self.body = {"error": {"code": "unsupported_parameter", "param": "reasoning_effort"}}

    async def _create(**kwargs):
        calls.append(kwargs)
        if "reasoning_effort" in kwargs:
            raise FakeError()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    fake_client = SimpleNamespace(
        base_url="https://one.example/v1",
        api_key="test",
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
    )
    monkeypatch.setattr(moa, "resolve_provider_client", lambda *args, **kwargs: (fake_client, "gpt-5.4"))
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda response: response.choices[0].message.content)

    model_name, content, success = await moa._run_reference_model_safe(
        {"model": "gpt-5.4", "provider": "custom", "reasoning_config": {"enabled": True, "effort": "high"}},
        "hello",
        max_retries=2,
    )

    assert (model_name, content, success) == ("gpt-5.4", "ok", True)
    assert calls[0]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in calls[1]
    assert moa._REASONING_UNSUPPORTED[("custom", "gpt-5.4", "one.example")] is True

    calls.clear()
    model_name, content, success = await moa._run_reference_model_safe(
        {"model": "gpt-5.4", "provider": "custom", "reasoning_config": {"enabled": True, "effort": "high"}},
        "hello again",
        max_retries=2,
    )

    assert (model_name, content, success) == ("gpt-5.4", "ok", True)
    assert len(calls) == 1
    assert "reasoning_effort" not in calls[0]

    fake_client.base_url = "https://two.example/v1"
    calls.clear()
    model_name, content, success = await moa._run_reference_model_safe(
        {"model": "gpt-5.4", "provider": "custom", "reasoning_config": {"enabled": True, "effort": "high"}},
        "fresh endpoint",
        max_retries=2,
    )

    assert (model_name, content, success) == ("gpt-5.4", "ok", True)
    assert calls[0]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in calls[1]
    assert moa._REASONING_UNSUPPORTED[("custom", "gpt-5.4", "two.example")] is True


@pytest.mark.asyncio
async def test_structured_unsupported_temperature_retries_and_caches(monkeypatch):
    calls = []

    class FakeError(Exception):
        def __init__(self):
            super().__init__("Unsupported parameter: temperature")
            self.body = {"error": {"code": "unsupported_parameter", "param": "temperature"}}

    async def _create(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise FakeError()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    fake_client = SimpleNamespace(
        base_url="https://one.example/v1",
        api_key="test",
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
    )
    monkeypatch.setattr(moa, "resolve_provider_client", lambda *args, **kwargs: (fake_client, "gpt-5.4"))
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda response: response.choices[0].message.content)

    model_name, content, success = await moa._run_reference_model_safe(
        {"model": "gpt-5.4", "provider": "custom", "reasoning_config": None},
        "hello",
        max_retries=2,
    )

    assert (model_name, content, success) == ("gpt-5.4", "ok", True)
    assert calls[0]["temperature"] == moa.REFERENCE_TEMPERATURE
    assert "temperature" not in calls[1]
    assert moa._TEMPERATURE_UNSUPPORTED[("custom", "gpt-5.4", "one.example")] is True

    calls.clear()
    model_name, content, success = await moa._run_reference_model_safe(
        {"model": "gpt-5.4", "provider": "custom", "reasoning_config": None},
        "hello again",
        max_retries=2,
    )

    assert (model_name, content, success) == ("gpt-5.4", "ok", True)
    assert len(calls) == 1
    assert "temperature" not in calls[0]


@pytest.mark.asyncio
async def test_vague_unsupported_reasoning_retries_once_without_sticky_cache(monkeypatch):
    calls = []

    class FakeError(Exception):
        pass

    async def _create(**kwargs):
        calls.append(kwargs)
        if "reasoning_effort" in kwargs:
            raise FakeError("reasoning is not supported for this model")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    fake_client = SimpleNamespace(
        base_url="https://one.example/v1",
        api_key="test",
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
    )
    monkeypatch.setattr(moa, "resolve_provider_client", lambda *args, **kwargs: (fake_client, "gpt-5.4"))
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda response: response.choices[0].message.content)

    model_name, content, success = await moa._run_reference_model_safe(
        {"model": "gpt-5.4", "provider": "custom", "reasoning_config": {"enabled": True, "effort": "high"}},
        "hello",
        max_retries=2,
    )

    assert (model_name, content, success) == ("gpt-5.4", "ok", True)
    assert len(calls) == 2
    assert moa._REASONING_UNSUPPORTED == {}


@pytest.mark.parametrize(
    "message",
    [
        "function tool_x is not supported",
        "tool calls are not supported",
        "streaming is unsupported",
    ],
)
def test_vague_non_parameter_unsupported_errors_do_not_poison_sticky_cache(message):
    class FakeError(Exception):
        pass

    dropped, sticky = moa._handle_unsupported_param(
        FakeError(message),
        moa._request_cache_key("custom", "gpt-5.4", "https://one.example/v1"),
    )

    assert (dropped, sticky) == (None, False)
    assert moa._REASONING_UNSUPPORTED == {}
    assert moa._TEMPERATURE_UNSUPPORTED == {}


@pytest.mark.asyncio
async def test_reference_model_init_failure_is_reported_as_model_failure(fake_catalogs, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("bad provider state")

    monkeypatch.setattr(moa, "resolve_provider_client", _boom)

    model_name, content, success = await moa._run_reference_model_safe(
        {"model": "gpt-5.4", "provider": "openai-codex", "reasoning_config": None},
        "hello",
        max_retries=1,
    )

    assert model_name == "gpt-5.4"
    assert success is False
    assert "could not be initialized" in content


@pytest.mark.asyncio
async def test_run_reference_model_safe_supports_copilot_acp(fake_catalogs):
    from agent.copilot_acp_client import CopilotACPClient

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ok",
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                )
            )
        ]
    )

    with (
        patch("agent.auxiliary_client._read_main_model", return_value="gpt-5.4"),
        patch.object(CopilotACPClient, "_create_chat_completion", return_value=fake_response) as mock_create,
        patch(
            "hermes_cli.auth.resolve_external_process_provider_credentials",
            return_value={
                "provider": "copilot-acp",
                "api_key": "copilot-acp",
                "base_url": "acp://copilot",
                "command": "/usr/bin/copilot",
                "args": ["--acp", "--stdio"],
            },
        ),
    ):
        model_name, content, success = await moa._run_reference_model_safe(
            {"model": "gpt-5.4", "provider": "copilot-acp", "reasoning_config": None},
            "hello",
            max_retries=1,
        )

    assert (model_name, content, success) == ("gpt-5.4", "ok", True)
    mock_create.assert_called_once()


def test_default_config_moa_entries_exist_in_provider_catalogs(fake_catalogs):
    default_moa = DEFAULT_CONFIG["moa"]
    entries = list(default_moa["reference_models"]) + [default_moa["aggregator_model"]]

    for entry in entries:
        normalized_model = normalize_model_for_provider(entry["model"], entry["provider"])
        assert normalized_model in fake_catalogs[entry["provider"]]


def test_module_constant_fallbacks_exist_in_openrouter_catalog(fake_catalogs):
    for model in moa.REFERENCE_MODELS + [moa.AGGREGATOR_MODEL]:
        normalized_model = normalize_model_for_provider(model, "openrouter")
        assert normalized_model in fake_catalogs["openrouter"]


def test_debug_parameters_capture_provider_and_reasoning(fake_catalogs):
    _write_config(
        {
            "moa": {
                "reference_models": [
                    {
                        "model": "anthropic/claude-opus-4.7",
                        "provider": "openrouter",
                        "reasoning": "high",
                    },
                    {"model": "gpt-5.4", "provider": "openai-codex"},
                ],
                "aggregator_model": {
                    "model": "anthropic/claude-opus-4.7",
                    "provider": "ai-gateway",
                    "reasoning": "none",
                },
            }
        }
    )

    loaded = moa._load_moa_config()

    ref0 = loaded["reference_models"][0]
    ref1 = loaded["reference_models"][1]
    agg = loaded["aggregator_model"]

    assert ref0["provider"] == "openrouter"
    assert ref0["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert ref1["provider"] == "openai-codex"
    assert ref1["reasoning_config"] is None
    assert agg["provider"] == "ai-gateway"
    assert agg["reasoning_config"] == {"enabled": False}


def test_codex_adapter_receives_reasoning_config_from_moa_shape():
    from agent.auxiliary_client import _CodexCompletionsAdapter

    moa_kwargs = moa._reasoning_kwargs(
        "openai-codex", "gpt-5.4", {"enabled": True, "effort": "high"}
    )
    assert moa_kwargs == {"reasoning_config": {"enabled": True, "effort": "high"}}

    captured: dict = {}

    class FakeStream:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(())

        def get_final_response(self):
            return SimpleNamespace(output=[], usage=None, model="gpt-5.4")

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(stream=lambda **kwargs: FakeStream(**kwargs))
    )

    adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.4")
    adapter.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        **moa_kwargs,
    )

    assert captured["reasoning"] == {"effort": "high", "summary": "auto"}
    assert captured["include"] == ["reasoning.encrypted_content"]


def test_anthropic_adapter_receives_reasoning_config_from_moa_shape(monkeypatch):
    from agent import auxiliary_client as aux

    moa_kwargs = moa._reasoning_kwargs(
        "anthropic", "claude-opus-4-6", {"enabled": True, "effort": "high"}
    )
    assert moa_kwargs == {"reasoning_config": {"enabled": True, "effort": "high"}}

    captured_build_kwargs: dict = {}

    def fake_build_kwargs(**kwargs):
        captured_build_kwargs.update(kwargs)
        return {
            "model": kwargs["model"],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": kwargs["max_tokens"],
        }

    fake_transport = SimpleNamespace(
        normalize_response=lambda _response, strip_tool_prefix=False: SimpleNamespace(
            content="",
            tool_calls=None,
            reasoning=None,
            finish_reason="stop",
        )
    )
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            content=[],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        ))
    )

    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_kwargs", fake_build_kwargs)
    monkeypatch.setattr("agent.transports.get_transport", lambda _mode: fake_transport)

    adapter = aux._AnthropicCompletionsAdapter(fake_client, "claude-opus-4-6")
    adapter.create(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "hi"}],
        **moa_kwargs,
    )

    assert captured_build_kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_anthropic_adapter_does_not_clobber_thinking_temperature(monkeypatch):
    from agent import auxiliary_client as aux

    captured: dict = {}

    def fake_build_kwargs(**kwargs):
        return {
            "model": kwargs["model"],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": kwargs["max_tokens"],
            "temperature": 1,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }

    fake_transport = SimpleNamespace(
        normalize_response=lambda _response, strip_tool_prefix=False: SimpleNamespace(
            content="",
            tool_calls=None,
            reasoning=None,
            finish_reason="stop",
        )
    )
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: (captured.update(kw), SimpleNamespace(
            content=[],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        ))[1])
    )

    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_kwargs", fake_build_kwargs)
    monkeypatch.setattr("agent.transports.get_transport", lambda _mode: fake_transport)

    adapter = aux._AnthropicCompletionsAdapter(fake_client, "claude-opus-4-6")
    adapter.create(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.4,
        reasoning_config={"enabled": True, "effort": "high"},
    )

    assert captured["temperature"] == 1


@pytest.mark.asyncio
async def test_mixture_of_agents_uses_adaptive_quorum(fake_catalogs, monkeypatch):
    _write_config(
        {
            "moa": {
                "reference_models": ["anthropic/claude-opus-4.7", "openai/gpt-5.5-pro"],
                "aggregator_model": "anthropic/claude-opus-4.7",
            }
        }
    )
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(side_effect=[
            ("anthropic/claude-opus-4.7", "ok", True),
            ("openai/gpt-5.5-pro", "failed", False),
        ]),
    )
    monkeypatch.setattr(moa, "_run_aggregator_model", AsyncMock(return_value="final"))
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(await moa.mixture_of_agents_tool("solve this"))

    assert result["success"] is False
    assert "Need at least 2 successful responses" in result["error"]
