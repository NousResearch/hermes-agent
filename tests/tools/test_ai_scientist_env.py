"""Tests for Hermes → AI-Scientist credential bridge (no live API)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import ai_scientist_env as env_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_SCRIPTS = REPO_ROOT / "vendor" / "openclaw-mirror" / "extensions" / "hypura-harness" / "scripts"


@pytest.fixture()
def harness_scripts_path():
    repo_root = str(REPO_ROOT)
    path = str(HARNESS_SCRIPTS)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return HARNESS_SCRIPTS


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o-mini", "openai"),
        ("auto", "auto"),
        ("claude-3-5-sonnet-20241022", "anthropic"),
        ("deepseek-chat", "deepseek"),
        ("llama3.1-405b", "openrouter"),
        ("gemini-2.0-flash", "gemini"),
        ("ollama/qwen:latest", "ollama"),
        ("openai-compatible/foo", "ollama"),
        ("unknown-model", "unknown"),
    ],
)
def test_infer_model_family(model: str, expected: str) -> None:
    assert env_mod.infer_model_family(model) == expected


def test_resolve_openai_from_codex_oauth(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with patch.object(env_mod, "_ensure_hermes_env_loaded"), patch(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        return_value={
            "api_key": "codex-oauth-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "source": "codex_auth",
        },
    ):
        overlay = env_mod.resolve_ai_scientist_credential_overlay("gpt-4o-mini")

    assert overlay["OPENAI_API_KEY"] == "codex-oauth-token"
    assert overlay["OPENAI_BASE_URL"] == "https://chatgpt.com/backend-api/codex"


def test_resolve_anthropic_from_oauth(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch.object(env_mod, "_ensure_hermes_env_loaded"), patch.object(
        env_mod, "_get_env_secret", return_value=""
    ), patch(
        "agent.anthropic_adapter.resolve_anthropic_token",
        return_value="sk-ant-oauth-token",
    ):
        overlay = env_mod.resolve_ai_scientist_credential_overlay("claude-3-5-sonnet-20241022")

    assert overlay["ANTHROPIC_API_KEY"] == "sk-ant-oauth-token"


def test_resolve_nous_free_tier_routing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    nous_resolved = env_mod.ResolvedAiScientistLLM(
        provider_id="nous",
        sakana_model=env_mod.SAKANA_OPENAI_SHIM_MODEL,
        api_model="DeepHermes-3-Llama-3-8B-Preview",
        api_key="nous-jwt",
        base_url=env_mod.DEFAULT_NOUS_BASE,
        source="nous_auth",
        routing="openai_shim",
    )
    with patch.object(env_mod, "_ensure_hermes_env_loaded"), patch.object(
        env_mod, "_provider_priority", return_value=("nous",)
    ), patch.dict(
        env_mod._PROVIDER_RESOLVERS,
        {"nous": lambda: nous_resolved, "openai-codex": lambda: None},
        clear=False,
    ):
        config = env_mod.resolve_ai_scientist_run_config("auto")

    assert config["has_credentials"] is True
    assert config["provider_id"] == "nous"
    assert config["routing"] == "openai_shim"
    assert config["overlay"]["OPENAI_API_KEY"] == "nous-jwt"
    assert config["overlay"]["AI_SCIENTIST_API_MODEL"] == "DeepHermes-3-Llama-3-8B-Preview"


def test_resolve_groq_routing(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    with patch.object(env_mod, "_ensure_hermes_env_loaded"), patch.object(
        env_mod, "_try_openai_codex", return_value=None
    ), patch.object(env_mod, "_try_nous", return_value=None), patch.object(
        env_mod, "_try_nvidia", return_value=None
    ):
        config = env_mod.resolve_ai_scientist_run_config("auto")

    assert config["provider_id"] == "groq"
    assert config["overlay"]["OPENAI_BASE_URL"] == env_mod.DEFAULT_GROQ_BASE


def test_build_ai_scientist_env_merges_base(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    with patch.object(
        env_mod,
        "resolve_ai_scientist_run_config",
        return_value={
            "overlay": {"OPENAI_API_KEY": "test-key"},
            "has_credentials": True,
            "sakana_model": "gpt-4o-mini",
        },
    ):
        merged = env_mod.build_ai_scientist_env(base={"PATH": "/usr/bin", "FOO": "bar"}, model="auto")

    assert merged["PATH"] == "/usr/bin"
    assert merged["FOO"] == "bar"
    assert merged["OPENAI_API_KEY"] == "test-key"


def test_ai_scientist_tool_uses_credential_bridge(monkeypatch, tmp_path) -> None:
    from tools import ai_scientist_tool as mod

    entry = tmp_path / "launch_scientist.py"
    launcher = tmp_path / "launcher.py"
    entry.write_text("# stub\n", encoding="utf-8")
    launcher.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(mod, "AI_SCIENTIST_DIR", tmp_path)
    monkeypatch.setattr(mod, "AI_SCIENTIST_ENTRYPOINT", entry)
    monkeypatch.setattr(mod, "AI_SCIENTIST_LAUNCHER", launcher)

    captured: dict[str, object] = {}

    def _fake_run_config(model=None):
        return {
            "sakana_model": "gpt-4o-mini",
            "overlay": {"OPENAI_API_KEY": "bridged-key"},
            "has_credentials": True,
            "provider_id": "openai-codex",
            "routing": "openai_shim",
        }

    def _fake_build(*, base=None, model=None):
        captured["model"] = model
        env = dict(base or {})
        env["OPENAI_API_KEY"] = "bridged-key"
        return env

    def _fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        captured["cmd"] = cmd
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "resolve_ai_scientist_run_config", _fake_run_config)
    monkeypatch.setattr(mod, "build_ai_scientist_env", _fake_build)
    monkeypatch.setattr(mod, "ensure_ai_scientist_deps", lambda **kwargs: None)
    monkeypatch.setattr("subprocess.run", _fake_run)

    mod.ai_scientist_research(experiment="nc_kan", model="auto", task_id="cred-bridge")

    assert captured["model"] == "auto"
    assert captured["env"]["OPENAI_API_KEY"] == "bridged-key"
    assert any("launcher.py" in str(part) for part in captured["cmd"])


def test_runner_applies_credentials_before_create_client(harness_scripts_path) -> None:
    import ai_scientist_runner as runner_mod

    applied: list[str | None] = []

    def _track(model):
        applied.append(model)

    fake_client = object()

    class _FakeLLM:
        @staticmethod
        def create_client(model):
            return fake_client, model

    fake_module = type(sys)("ai_scientist.llm")
    fake_module.create_client = _FakeLLM.create_client
    with patch.object(runner_mod, "_apply_hermes_credentials", side_effect=_track), patch.dict(
        sys.modules, {"ai_scientist.llm": fake_module}
    ):
        client, model = runner_mod.resolve_ai_scientist_client("gpt-4o-mini")

    assert applied == ["gpt-4o-mini"]
    assert client is fake_client
    assert model == "gpt-4o-mini"


def test_lazy_deps_ai_scientist_feature_registered() -> None:
    from tools.ai_scientist_deps import AIDER_SPEC
    from tools.lazy_deps import LAZY_DEPS

    specs = LAZY_DEPS["tool.ai_scientist"]
    assert "backoff==2.2.1" in specs
    assert not any(s.startswith("aider-chat==") for s in specs)
    assert AIDER_SPEC.startswith("aider-chat==")
