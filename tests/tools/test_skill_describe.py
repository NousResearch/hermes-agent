import textwrap


def _setup_skills(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    for category, name, description in [
        ("cat-a", "alpha", "A description"),
        ("cat-a", "beta", "B description"),
        ("cat-b", "gamma", "C description"),
    ]:
        skill_dir = skills_dir / category / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent(
                f"""\
                ---
                name: {name}
                description: {description}
                ---
                body
                """
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [skills_dir])
    from agent.skill_inventory import clear_inventory_cache

    clear_inventory_cache(clear_snapshot=True)


def test_skill_describe_by_category(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe

    result = skill_describe(category="cat-a")
    assert result["success"] is True
    assert sorted(skill["name"] for skill in result["skills"]) == ["alpha", "beta"]


def test_skill_describe_by_names(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe

    result = skill_describe(names=["alpha", "gamma"])
    assert result["success"] is True
    assert sorted(skill["name"] for skill in result["skills"]) == ["alpha", "gamma"]


def test_skill_describe_unknown_name_errors(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe

    result = skill_describe(names=["does-not-exist"])
    assert result["success"] is False
    assert "does-not-exist" in result["error"]


def test_skill_describe_unknown_category_errors(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe

    result = skill_describe(category="missing")
    assert result["success"] is False


def test_skill_describe_empty_args_errors(tmp_path, monkeypatch):
    _setup_skills(tmp_path, monkeypatch)
    from tools.skills_tool import skill_describe

    result = skill_describe()
    assert result["success"] is False

