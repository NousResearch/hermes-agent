"""#108032: skills.external_dirs provenance + foreground write guard.

Uses a real tmp HERMES_HOME + config.yaml (not get_all_skills_dirs patches) so
get_external_skills_dirs() is the actual detection surface. The older
TestExternalSkillMutations._two_roots class documents #4759 in-place writes
and is left intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent.skill_utils as su


def _write_skill(root: Path, name: str, body: str = "Body with OLD_MARKER here.\n") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\n{body}",
        encoding="utf-8",
    )
    return skill_dir


def _write_config(home: Path, *, external_dirs: list[Path], extra_skills: str = "") -> None:
    lines = ["skills:", "  external_dirs:"]
    if external_dirs:
        lines.extend(f"    - {d}" for d in external_dirs)
    else:
        lines.append("    []")
    if extra_skills:
        lines.append(extra_skills.rstrip("\n"))
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    su._external_dirs_cache_clear()


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    su._external_dirs_cache_clear()
    yield {"home": home, "vault": vault, "local": home / "skills"}
    su._external_dirs_cache_clear()


def _patch(name: str, old: str = "OLD_MARKER", new: str = "NEW_MARKER") -> dict:
    from tools.skill_manager_tool import skill_manage
    return json.loads(skill_manage(action="patch", name=name, old_string=old, new_string=new))


# ---------------------------------------------------------------------------
# 1. Provenance: hub > bundled > external > agent
# ---------------------------------------------------------------------------


class TestExternalDirsProvenance:
    def test_external_only_skill_is_external_not_agent(self, hermes_env):
        _write_config(hermes_env["home"], external_dirs=[hermes_env["vault"]])
        _write_skill(hermes_env["vault"], "ext-only")
        from tools.skill_usage import provenance
        assert provenance("ext-only") == "external"

    def test_local_only_skill_stays_agent(self, hermes_env):
        _write_config(hermes_env["home"], external_dirs=[hermes_env["vault"]])
        _write_skill(hermes_env["local"], "local-only")
        from tools.skill_usage import provenance
        assert provenance("local-only") == "agent"

    def test_local_copy_wins_over_external_same_name(self, hermes_env):
        """Local first-wins: a name in ~/.hermes/skills AND external_dirs stays agent."""
        _write_config(hermes_env["home"], external_dirs=[hermes_env["vault"]])
        _write_skill(hermes_env["local"], "shared-name")
        _write_skill(hermes_env["vault"], "shared-name")
        from tools.skill_usage import provenance
        assert provenance("shared-name") == "agent"

    def test_hub_and_bundled_names_unchanged(self, hermes_env, monkeypatch):
        _write_config(hermes_env["home"], external_dirs=[hermes_env["vault"]])
        _write_skill(hermes_env["vault"], "hub-skill")
        _write_skill(hermes_env["vault"], "bundled-skill")
        from tools import skill_usage
        monkeypatch.setattr(skill_usage, "is_hub_installed", lambda n: n == "hub-skill")
        monkeypatch.setattr(skill_usage, "is_bundled", lambda n: n == "bundled-skill")
        assert skill_usage.provenance("hub-skill") == "hub"
        assert skill_usage.provenance("bundled-skill") == "bundled"

    def test_empty_external_dirs_fail_open_as_agent(self, hermes_env):
        _write_config(hermes_env["home"], external_dirs=[])
        _write_skill(hermes_env["vault"], "orphan")
        from tools.skill_usage import provenance
        assert provenance("orphan") == "agent"


# ---------------------------------------------------------------------------
# 2–4. Foreground write guard
# ---------------------------------------------------------------------------


class TestExternalDirsForegroundWriteGuard:
    def test_patch_external_skill_refused_and_file_unchanged(self, hermes_env):
        _write_config(hermes_env["home"], external_dirs=[hermes_env["vault"]])
        skill_dir = _write_skill(hermes_env["vault"], "ext-skill")
        before = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

        result = _patch("ext-skill")

        assert result["success"] is False, result
        err = (result.get("error") or "").lower()
        assert "external_dirs" in err or "external" in err
        assert "read-only" in err or "readonly" in err or "allow_mutations" in err
        assert "external_dirs_allow_mutations" in (result.get("error") or "")
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == before

    def test_allow_mutations_true_patches_in_place(self, hermes_env):
        """CONTROL / #4759: opt-in keeps in-place writes."""
        _write_config(
            hermes_env["home"],
            external_dirs=[hermes_env["vault"]],
            extra_skills="  external_dirs_allow_mutations: true",
        )
        skill_dir = _write_skill(hermes_env["vault"], "ext-skill")

        result = _patch("ext-skill")

        assert result["success"] is True, result
        assert "NEW_MARKER" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert not (hermes_env["local"] / "ext-skill").exists()

    def test_quoted_false_does_not_enable_mutations(self, hermes_env):
        skill_dir = _write_skill(hermes_env["vault"], "ext-skill")
        before = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        for quoted in ("false", "0", "no", "off"):
            _write_config(
                hermes_env["home"],
                external_dirs=[hermes_env["vault"]],
                extra_skills=f'  external_dirs_allow_mutations: "{quoted}"',
            )
            result = _patch("ext-skill")
            assert result["success"] is False, f"quoted {quoted!r} must not enable mutations: {result}"
            assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == before

    def test_local_skill_still_patches(self, hermes_env):
        _write_config(hermes_env["home"], external_dirs=[hermes_env["vault"]])
        skill_dir = _write_skill(hermes_env["local"], "local-skill")

        result = _patch("local-skill")

        assert result["success"] is True, result
        assert "NEW_MARKER" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. Forbidden heuristic: do not lift is_external_skill_path (project dirs)
# ---------------------------------------------------------------------------


class TestProjectSkillsNotBlocked:
    def test_trusted_project_skill_foreground_patch_allowed(self, hermes_env, tmp_path, monkeypatch):
        """is_external_skill_path is True via project dirs, but the path is NOT
        under get_external_skills_dirs — foreground patch must still succeed.
        create_dir points at the project skills dir so _find_skill locates it
        without patching get_all_skills_dirs.
        """
        repo = tmp_path / "proj"
        (repo / ".git").mkdir(parents=True)
        project_skills = repo / ".hermes" / "skills"
        skill_dir = _write_skill(project_skills, "repo-skill")
        _write_config(
            hermes_env["home"],
            external_dirs=[],
            extra_skills=(
                f"  trusted_project_dirs: ['{repo}']\n"
                f"  create_dir: '{project_skills}'"
            ),
        )
        monkeypatch.chdir(repo)
        su._external_dirs_cache_clear()

        assert su.is_external_skill_path(skill_dir) is True
        assert su.get_external_skills_dirs() == []

        result = _patch("repo-skill")

        assert result["success"] is True, result
        assert "NEW_MARKER" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
