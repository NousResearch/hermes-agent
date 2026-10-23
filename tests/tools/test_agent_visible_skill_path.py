"""Skill paths rendered to the model must resolve inside the active terminal backend (#135899).

``skill_view``'s ``skill_dir`` and ``${HERMES_SKILL_DIR}`` hand out host paths, but
docker/modal/ssh-style backends run the skill's scripts where the dir is mounted or
synced in — the host path dangles there. Mirrors the cache-path coverage in
``TestToAgentVisiblePathPerBackend`` (tests/tools/test_credential_files.py).
"""

from pathlib import Path

import pytest


def _local_skill(hermes_home: Path) -> Path:
    skill = hermes_home / "skills" / "xlsx"
    skill.mkdir(parents=True)
    return skill


class TestMapSkillPathToContainer:
    def test_local_skills_dir_maps(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        skill = _local_skill(hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        from tools.credential_files import map_skill_path_to_container
        assert map_skill_path_to_container(str(skill)) == "/root/.hermes/skills/xlsx"

    def test_external_skills_dir_maps_to_its_namespace(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        external = tmp_path / "bundled-skills"
        external.mkdir()
        monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", lambda: [external])

        from tools.credential_files import map_skill_path_to_container
        assert map_skill_path_to_container(str(external / "docx")) == "/root/.hermes/external_skills/0/docx"

    def test_path_outside_skill_roots_is_none(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        from tools.credential_files import map_skill_path_to_container
        assert map_skill_path_to_container(str(tmp_path / "plugin" / "skill")) is None

    def test_symlinked_home_maps_resolved_path(self, tmp_path, monkeypatch):
        real_home = tmp_path / "real-hermes"
        skill = _local_skill(real_home)
        link_home = tmp_path / ".hermes"
        link_home.symlink_to(real_home, target_is_directory=True)
        monkeypatch.setenv("HERMES_HOME", str(link_home))

        from tools.credential_files import map_skill_path_to_container
        # The configured spelling and the resolved spelling both map (#103147 shape).
        assert map_skill_path_to_container(str(link_home / "skills" / "xlsx")) == "/root/.hermes/skills/xlsx"
        assert map_skill_path_to_container(str(skill)) == "/root/.hermes/skills/xlsx"


class TestToAgentVisibleSkillPathPerBackend:
    def _skill(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        skill = _local_skill(hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        return str(skill)

    @pytest.mark.parametrize("backend", ["docker", "modal"])
    def test_container_backends_map_to_root_hermes(self, tmp_path, monkeypatch, backend):
        skill = self._skill(tmp_path, monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", backend)
        from tools.credential_files import to_agent_visible_skill_path
        assert to_agent_visible_skill_path(skill) == "/root/.hermes/skills/xlsx"

    def test_synced_backend_maps_to_tilde_hermes(self, tmp_path, monkeypatch):
        skill = self._skill(tmp_path, monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        from tools.credential_files import to_agent_visible_skill_path
        assert to_agent_visible_skill_path(skill) == "~/.hermes/skills/xlsx"

    @pytest.mark.parametrize("backend", ["local", "singularity", ""])
    def test_untranslated_backends_keep_host_path(self, tmp_path, monkeypatch, backend):
        skill = self._skill(tmp_path, monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", backend)
        from tools.credential_files import to_agent_visible_skill_path
        assert to_agent_visible_skill_path(skill) == skill

    def test_plugin_skill_dir_outside_mounts_keeps_host_path(self, tmp_path, monkeypatch):
        """Plugin-bundled skills are not mounted into the sandbox; the host path is
        what the model gets (unchanged behaviour, locked here)."""
        self._skill(tmp_path, monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        plugin_skill = str(tmp_path / "plugins" / "superpowers" / "writing-plans")
        from tools.credential_files import to_agent_visible_skill_path
        assert to_agent_visible_skill_path(plugin_skill) == plugin_skill


class TestSkillTemplateVarPerBackend:
    """${HERMES_SKILL_DIR} must expand to the backend-visible dir (#135899)."""

    def test_docker_expands_to_container_dir(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        skill = _local_skill(hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        from agent.skill_preprocessing import substitute_template_vars
        out = substitute_template_vars("run ${HERMES_SKILL_DIR}/scripts/convert.py", skill, None)
        assert out == "run /root/.hermes/skills/xlsx/scripts/convert.py"

    def test_local_keeps_host_dir(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        skill = _local_skill(hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        from agent.skill_preprocessing import substitute_template_vars
        out = substitute_template_vars("run ${HERMES_SKILL_DIR}/scripts/convert.py", skill, None)
        assert out == f"run {skill}/scripts/convert.py"

    def test_unresolvable_token_stays_in_place(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        from agent.skill_preprocessing import substitute_template_vars
        out = substitute_template_vars("${HERMES_SKILL_DIR} and ${HERMES_SESSION_ID}", None, None)
        assert out == "${HERMES_SKILL_DIR} and ${HERMES_SESSION_ID}"


class TestSkillViewReportsBackendDir:
    """skill_view's model-facing ``skill_dir`` field translates; internal bookkeeping
    paths stay on the host."""

    @pytest.fixture
    def skill(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        skill = _local_skill(hermes_home)
        (skill / "SKILL.md").write_text(
            "---\nname: xlsx\ndescription: Convert spreadsheets\n---\nbody\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        return skill

    def test_skill_dir_field_translates_under_docker(self, tmp_path, monkeypatch, skill):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        import json

        from tools.skills_tool import skill_view
        result = json.loads(skill_view(name="xlsx"))
        assert result["skill_dir"] == "/root/.hermes/skills/xlsx"
        # Internal dedup fingerprint keeps the host source path.
        assert result["_source_path"] == str(skill / "SKILL.md")

    def test_skill_dir_field_keeps_host_path_under_local(self, tmp_path, monkeypatch, skill):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        import json

        from tools.skills_tool import skill_view
        result = json.loads(skill_view(name="xlsx"))
        assert result["skill_dir"] == str(skill)
