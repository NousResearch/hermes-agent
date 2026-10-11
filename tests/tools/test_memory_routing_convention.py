"""Memory vs skill routing convention (#106786).

Procedures / workflows / recipes belong in SKILL.md via skill_manage
create/update — memory holds every-session declarative facts, not how-tos.
Guidance-only: no runtime classifier.
"""

from agent.prompt_builder import MEMORY_GUIDANCE, build_memory_guidance
from tools.memory_tool import MEMORY_SCHEMA


def test_memory_schema_routes_procedures_to_skill_manage():
    desc = MEMORY_SCHEMA["description"].lower()
    assert "skill_manage" in desc
    # must explicitly steer procedures/workflows/recipes away from memory how-tos
    assert any(k in desc for k in ("procedure", "workflow", "recipe"))
    assert "how-to" in desc or "howto" in desc.replace("-", "") or "declarative" in desc
    # concrete create/SKILL path signal from the issue
    assert "skill.md" in desc or "create" in desc


def test_memory_schema_routing_sentence_is_explicit():
    """Issue #106786: one schema sentence must name the trio, skill_manage,
    and that memory is declarative / not how-to content."""
    desc = MEMORY_SCHEMA["description"].lower()
    assert "procedure" in desc
    assert "workflow" in desc
    assert "recipe" in desc
    assert "skill_manage" in desc
    assert "skill.md" in desc
    assert "create" in desc
    assert "declarative" in desc
    assert "how-to" in desc or "howto" in desc.replace("-", "")
    assert "do not repeat" in desc or "not duplicate" in desc or "already cover" in desc


def test_memory_guidance_keeps_skills_first_for_procedures():
    assert MEMORY_GUIDANCE.index("Skills come first") < MEMORY_GUIDANCE.index(
        "Memory is the narrow exception"
    )
    assert "procedures and workflows belong" in MEMORY_GUIDANCE
    assert "declarative facts" in MEMORY_GUIDANCE
    # Narrow ~3-step temporary exception — no category curricula.
    assert "3-step" in MEMORY_GUIDANCE
    assert "temporary" in MEMORY_GUIDANCE
    # Diet (#95681): do not re-teach category curricula here.
    assert "PR numbers" not in MEMORY_GUIDANCE
    assert "tool quirks" not in MEMORY_GUIDANCE


def test_memory_guidance_unavailable_skill_write_still_routes_to_skills():
    """Fail-open: when skill writing is unavailable, still say procedures
    belong in skills — do not invent a hard block or widen memory."""
    text = build_memory_guidance(True, True, skill_manage_available=False)
    assert "belongs in skills" in text
    assert "even when skill writing is unavailable" in text
    assert "procedures" in text
