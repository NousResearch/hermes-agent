"""Advisory leakage coverage for tools/skill_linter.py + the review prompts.

The post-turn background review fork (and manual /refine, which shares the
prompt) must GENERALIZE lessons before writing them to the shared skill
library. The linter rule stays advisory: WARNING only, never a gate.
"""

from agent.background_review import (
    _COMBINED_REVIEW_PROMPT,
    _MEMORY_REVIEW_PROMPT,
    _SKILL_REVIEW_PROMPT,
)
from tools.skill_linter import WARNING, has_errors, lint_content, lint_skill

LEAKY_BODY = """# Leaky Skill

## When to Use
- When reconciling shared state in myproject.

## Procedure
1. Read /home/alice/myproject/MYPROJECT_STATUS.json for state.
2. Reuse session 9f3b2c1d-4e5a-4a7b-8c9d-0e1f2a3b4c5d on 10.20.30.40
   (branch fix/123627-skill-review-distil) as the example.
"""

LEAKY = """---
name: leaky-skill
description: Reconcile shared project state across workers.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [coordination]
    related_skills: []
---
""" + LEAKY_BODY


def _rules(findings):
    return {f.rule for f in findings}


def test_leaky_body_warns_once_per_category():
    findings = [f for f in lint_content(LEAKY, project_tokens=("myproject",))
                if f.rule == "project-specific-content"]
    assert findings, "expected advisory leakage warnings"
    assert all(f.severity == WARNING for f in findings)
    # one WARNING per category, not one per match
    assert len(findings) == len({f.message.split(":")[1] for f in findings})


def test_leakage_never_blocks():
    findings = lint_content(LEAKY, project_tokens=("myproject",))
    assert not has_errors(findings)


def test_bare_project_name_needs_tokens():
    def _name_findings(content, **kw):
        return [f for f in lint_content(content, **kw)
                if f.rule == "project-specific-content" and ":project_name:" in f.message]
    assert _name_findings(LEAKY) == []
    assert _name_findings(LEAKY, project_tokens=("myproject",))


def test_skill_prompts_carry_scope_rule_memory_prompt_untouched():
    for prompt in (_SKILL_REVIEW_PROMPT, _COMBINED_REVIEW_PROMPT):
        assert "GLOBAL-SKILL SCOPE RULE" in prompt
        assert "GENERALIZE the lesson" in prompt
    assert "GLOBAL-SKILL SCOPE RULE" not in _MEMORY_REVIEW_PROMPT


def test_lint_skill_accepts_project_tokens_positionally(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(LEAKY, encoding="utf-8")
    assert "project-specific-content" in _rules(lint_skill(skill_md, ("myproject",)))
    # existing callers keep working unchanged
    assert isinstance(lint_skill(skill_md), list)


def test_manager_lint_path_stays_advisory(tmp_path):
    from tools.skill_manager_tool import _attach_lint_findings
    skill_dir = tmp_path / "leaky-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(LEAKY, encoding="utf-8")
    result = {"success": True}
    _attach_lint_findings(result, skill_md)
    assert result["success"] is True
    for w in result.get("lint_warnings", []):
        assert w["severity"] == WARNING
