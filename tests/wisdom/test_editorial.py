import sys
import types
import shutil
from pathlib import Path

from agent.skill_utils import parse_frontmatter
from hermes_wisdom.editorial import (
    apply_editorial_metadata_to_overlay,
    ensure_skill_editorial_metadata,
)


def _skill(tmp_path: Path, frontmatter: str) -> Path:
    skill = tmp_path / "legacy-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        f"---\n{frontmatter}---\n# Workflow\nFollow the safe procedure.\n",
        encoding="utf-8",
    )
    return skill


def _model(monkeypatch, *, name: str, description: str):
    captured = {}
    module = types.ModuleType("agent.auxiliary_client")

    def call_llm(**kwargs):
        captured.update(kwargs)
        return {
            "content": (
                '{"editorial_name": "'
                + name
                + '", "editorial_description": "'
                + description
                + '"}'
            )
        }

    module.call_llm = call_llm
    module.extract_content_or_reasoning = lambda response: response["content"]
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    return captured


def test_qualifying_legacy_skill_generates_editorial_copy_without_mutating_source(
    monkeypatch, tmp_path: Path
):
    skill = _skill(
        tmp_path,
        "name: legacy-skill\n"
        "description: Use when running a safe release.\n"
        "metadata:\n"
        "  hermes:\n"
        "    tags: [release]\n",
    )
    captured = _model(
        monkeypatch,
        name="Safe Release Workflow",
        description="Prepare and verify a reliable software release.",
    )

    original = (skill / "SKILL.md").read_text(encoding="utf-8")
    result = ensure_skill_editorial_metadata(skill)

    assert result == {
        "editorial_name": "Safe Release Workflow",
        "editorial_description": "Prepare and verify a reliable software release.",
        "changed": False,
    }
    assert (skill / "SKILL.md").read_text(encoding="utf-8") == original
    assert captured["task"] == "background_review"
    assert captured["tools"] == []
    assert "untrusted" in captured["messages"][0]["content"].lower()


def test_existing_editorial_copy_never_calls_model(monkeypatch, tmp_path: Path):
    skill = _skill(
        tmp_path,
        "name: legacy-skill\n"
        "description: Agent-facing description.\n"
        "metadata:\n"
        "  hermes:\n"
        "    editorial_name: Existing Name\n"
        "    editorial_description: Existing description.\n",
    )
    module = types.ModuleType("agent.auxiliary_client")
    module.call_llm = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("model should not be called")
    )
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)

    result = ensure_skill_editorial_metadata(skill)

    assert result == {
        "editorial_name": "Existing Name",
        "editorial_description": "Existing description.",
        "changed": False,
    }


def test_generation_failure_keeps_legacy_skill_and_falls_back(
    monkeypatch, tmp_path: Path
):
    skill = _skill(
        tmp_path,
        "name: legacy-skill\ndescription: Agent-facing description.\n",
    )
    original = (skill / "SKILL.md").read_text(encoding="utf-8")
    module = types.ModuleType("agent.auxiliary_client")
    module.call_llm = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
    module.extract_content_or_reasoning = lambda _response: ""
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)

    result = ensure_skill_editorial_metadata(skill)

    assert result == {
        "editorial_name": "legacy-skill",
        "editorial_description": "Agent-facing description.",
        "changed": False,
    }
    assert (skill / "SKILL.md").read_text(encoding="utf-8") == original


def test_generation_preserves_an_existing_editorial_field(monkeypatch, tmp_path: Path):
    skill = _skill(
        tmp_path,
        "name: legacy-skill\n"
        "description: Agent-facing description.\n"
        "metadata:\n"
        "  hermes:\n"
        "    editorial_name: Owner Chosen Name\n",
    )
    _model(
        monkeypatch,
        name="Generated Name",
        description="A clear description generated for people.",
    )

    original = (skill / "SKILL.md").read_text(encoding="utf-8")
    result = ensure_skill_editorial_metadata(skill)

    assert result["editorial_name"] == "Owner Chosen Name"
    assert result["editorial_description"] == (
        "A clear description generated for people."
    )
    assert (skill / "SKILL.md").read_text(encoding="utf-8") == original


def test_generated_editorial_copy_is_applied_only_to_review_overlay(tmp_path: Path):
    skill = _skill(
        tmp_path,
        "name: legacy-skill\ndescription: Agent-facing description.\n",
    )
    original = (skill / "SKILL.md").read_text(encoding="utf-8")
    overlay = tmp_path / "overlay"
    shutil.copytree(skill, overlay)

    assert apply_editorial_metadata_to_overlay(
        overlay,
        editorial_name="Human Friendly Skill",
        editorial_description="A clear explanation for people.",
    )

    assert (skill / "SKILL.md").read_text(encoding="utf-8") == original
    frontmatter, _body = parse_frontmatter(
        (overlay / "SKILL.md").read_text(encoding="utf-8")
    )
    assert frontmatter["metadata"]["hermes"] == {
        "editorial_name": "Human Friendly Skill",
        "editorial_description": "A clear explanation for people.",
    }
