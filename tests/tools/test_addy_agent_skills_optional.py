from tools.skills_hub import OptionalSkillSource


def test_addy_agent_skills_pack_is_searchable_from_official_optional_source():
    source = OptionalSkillSource()

    results = source.search("addy-agent-skills", limit=50)
    names = {result.name for result in results}

    assert "addy-using-agent-skills" in names
    assert "addy-spec-driven-development" in names
    assert "addy-test-driven-development" in names
    assert len(names) >= 24


def test_addy_agent_skills_fetch_includes_shared_references_and_license():
    source = OptionalSkillSource()

    bundle = source.fetch("official/software-development/addy-agent-skills/skills/addy-using-agent-skills")

    assert bundle is not None
    assert "SKILL.md" in bundle.files
    assert "references/testing-patterns.md" in bundle.files
    assert "agents/code-reviewer.md" in bundle.files
    assert "LICENSE.agent-skills" in bundle.files
    skill_md = bundle.files["SKILL.md"]
    if isinstance(skill_md, bytes):
        skill_md = skill_md.decode("utf-8")
    assert "name: addy-using-agent-skills" in skill_md
    assert "addy-spec-driven-development" in skill_md
