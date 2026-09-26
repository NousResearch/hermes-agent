"""Skills show the model the container path of their directory under a Docker terminal.

The Docker backend bind-mounts skill directories read-only under
``/root/.hermes`` inside the sandbox, so the directory the model is told about
must be that container path. On a local backend the host path is correct and
must be left alone.
"""

import json
from unittest.mock import patch

import pytest

import tools.container_paths as cp
from agent.skill_commands import build_skill_invocation_message, scan_skill_commands
from tools.skills_tool import skill_view


def _make_skill(skills_dir, name, body="Run scripts/run.sh."):
    skill_dir = skills_dir / name
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "run.sh").write_text("echo ok\n", encoding="utf-8")
    return skill_dir


@pytest.fixture
def skills_root(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    return root


@pytest.fixture
def docker_backend(monkeypatch, skills_root):
    """Docker terminal backend whose only skill mount is ``skills_root``."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "[]")
    monkeypatch.setattr(cp, "_config_terminal_value", lambda key: None)
    with patch(
        "tools.credential_files._skill_dir_roots",
        lambda base: iter([(skills_root, f"{base}/skills")]),
    ):
        yield


@pytest.fixture
def local_backend(monkeypatch, skills_root):
    monkeypatch.setattr(cp, "_config_terminal_value", lambda key: None)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    with patch(
        "tools.credential_files._skill_dir_roots",
        lambda base: iter([(skills_root, f"{base}/skills")]),
    ):
        yield


class TestSkillViewSkillDir:
    def test_docker_backend_reports_container_path(self, docker_backend, skills_root):
        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            skill_dir = _make_skill(skills_root, "demo")
            result = json.loads(skill_view("demo"))

        assert result["success"] is True
        assert result["skill_dir"] == "/root/.hermes/skills/demo"
        assert result["skill_dir_note"]
        # The model-facing payload carries no host path for the directory.
        assert str(skill_dir) not in json.dumps(
            {k: v for k, v in result.items() if k != "_source_path"})

    def test_host_paths_returns_host_dir_for_in_process_callers(self, docker_backend, skills_root):
        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            skill_dir = _make_skill(skills_root, "demo")
            result = json.loads(skill_view("demo", host_paths=True))

        assert result["success"] is True
        assert result["skill_dir"] == str(skill_dir)

    def test_local_backend_reports_host_path(self, local_backend, skills_root):
        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            skill_dir = _make_skill(skills_root, "demo")
            result = json.loads(skill_view("demo"))

        assert result["success"] is True
        assert result["skill_dir"] == str(skill_dir)
        assert result["skill_dir_note"] is None


class TestSkillInvocationMessage:
    def test_docker_backend_announces_container_dir(self, docker_backend, skills_root):
        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            skill_dir = _make_skill(skills_root, "demo")
            scan_skill_commands()
            msg = build_skill_invocation_message("/demo", "go")

        assert msg is not None
        assert "[Skill directory: /root/.hermes/skills/demo]" in msg
        assert f"[Skill directory: {skill_dir}]" not in msg
        # Linked files are still discovered host-side.
        assert "scripts/run.sh" in msg

    def test_local_backend_announces_host_dir(self, local_backend, skills_root):
        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            skill_dir = _make_skill(skills_root, "demo")
            scan_skill_commands()
            msg = build_skill_invocation_message("/demo", "go")

        assert msg is not None
        assert f"[Skill directory: {skill_dir}]" in msg
