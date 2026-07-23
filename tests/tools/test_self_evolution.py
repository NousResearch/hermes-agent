import json
from unittest.mock import MagicMock, patch

import tools.ai_scientist_tool as ai_scientist_tool
import tools.shinka_evolve_tool as shinka_evolve_tool
from tools.ai_scientist_tool import ai_scientist_research
from tools.shinka_evolve_tool import shinka_run_batch


def test_shinka_evolve_tool_registration():
    from tools.registry import registry

    assert "shinka_run" in registry.get_all_tool_names()


def test_ai_scientist_tool_registration():
    from tools.registry import registry

    assert "ai_scientist_research" in registry.get_all_tool_names()


def test_self_evolution_toolset_in_toolsets():
    from toolsets import TOOLSETS, resolve_toolset, validate_toolset

    assert "self_evolution" in TOOLSETS
    assert TOOLSETS["self_evolution"].get("opt_in") is True
    assert validate_toolset("self_evolution") is True
    resolved = resolve_toolset("self_evolution")
    assert "ai_scientist_research" in resolved
    assert "shinka_run" in resolved


def test_ai_scientist_requires_initialized_submodule(monkeypatch, tmp_path):
    monkeypatch.setattr(ai_scientist_tool, "AI_SCIENTIST_ENTRYPOINT", tmp_path / "launch_scientist.py")
    monkeypatch.setattr(ai_scientist_tool, "AI_SCIENTIST_LAUNCHER", tmp_path / "launcher.py")
    assert ai_scientist_tool.check_ai_scientist_available() is False

    (tmp_path / "launch_scientist.py").write_text("", encoding="utf-8")
    (tmp_path / "launcher.py").write_text("", encoding="utf-8")
    assert ai_scientist_tool.check_ai_scientist_available() is True


def test_shinka_requires_initialized_submodule(monkeypatch, tmp_path):
    sentinel = tmp_path / "shinka" / "__init__.py"
    cli = tmp_path / "shinka" / "cli" / "run.py"
    monkeypatch.setattr(shinka_evolve_tool, "SHINKA_PACKAGE_INIT", sentinel)
    monkeypatch.setattr(shinka_evolve_tool, "SHINKA_CLI_RUN", cli)
    assert shinka_evolve_tool.check_shinka_available() is False

    sentinel.parent.mkdir(parents=True)
    cli.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("", encoding="utf-8")
    cli.write_text("", encoding="utf-8")
    assert shinka_evolve_tool.check_shinka_available() is True


def test_shinka_vendor_path_points_at_shinka_osint():
    assert shinka_evolve_tool.SHINKA_DIR.name == "shinka-osint"
    assert shinka_evolve_tool.SHINKA_DIR.as_posix().endswith("vendor/shinka-osint")


@patch("subprocess.run")
def test_shinka_run_dispatch(mock_run, monkeypatch):
    mock_run.return_value = MagicMock(returncode=0, stdout="Batch completed", stderr="")
    monkeypatch.setattr(
        shinka_evolve_tool,
        "resolve_shinka_run_config",
        lambda model=None: {
            "overlay": {"OPENAI_API_KEY": "test"},
            "llm_models": ["gpt-4o-mini"],
            "has_credentials": True,
            "provider_id": "openai-codex",
            "routing": "openai_shim",
        },
    )
    monkeypatch.setattr(
        shinka_evolve_tool,
        "build_shinka_env",
        lambda **kwargs: {
            "OPENAI_API_KEY": "test",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": "x",
        },
    )

    result_json = shinka_run_batch(
        task_dir="examples/test_task", num_generations=1, task_id="test_session"
    )
    result = json.loads(result_json)

    assert result["success"] is True
    assert "results_dir" in result
    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert "-m" in cmd and "shinka.cli.run" in cmd
    assert mock_run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"


@patch("subprocess.run")
def test_ai_scientist_research_dispatch(mock_run, monkeypatch, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stdout="Research completed", stderr="")
    entry = tmp_path / "launch_scientist.py"
    launcher = tmp_path / "ai_scientist_launcher.py"
    entry.write_text("# stub", encoding="utf-8")
    launcher.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(ai_scientist_tool, "AI_SCIENTIST_DIR", tmp_path)
    monkeypatch.setattr(ai_scientist_tool, "AI_SCIENTIST_ENTRYPOINT", entry)
    monkeypatch.setattr(ai_scientist_tool, "AI_SCIENTIST_LAUNCHER", launcher)
    monkeypatch.setattr(
        ai_scientist_tool,
        "resolve_ai_scientist_run_config",
        lambda model=None: {
            "sakana_model": "gpt-4o-mini",
            "overlay": {"OPENAI_API_KEY": "test"},
            "has_credentials": True,
            "provider_id": "openai-codex",
            "routing": "openai_shim",
        },
    )
    monkeypatch.setattr(ai_scientist_tool, "ensure_ai_scientist_deps", lambda **kwargs: None)
    monkeypatch.setattr(
        ai_scientist_tool,
        "build_ai_scientist_env",
        lambda **kwargs: {"OPENAI_API_KEY": "test", "CUDA_VISIBLE_DEVICES": "0"},
    )

    result_json = ai_scientist_research(
        experiment="nanoGPT_lite", num_ideas=1, task_id="test_session"
    )
    result = json.loads(result_json)

    assert result["success"] is True
    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert any("ai_scientist_launcher.py" in str(part) for part in cmd)
    assert "--skip-novelty-check" in cmd
    assert "--experiment" in cmd and "nanoGPT_lite" in cmd
    env = mock_run.call_args.kwargs.get("env") or {}
    assert isinstance(env, dict)
