"""``delegation.child_toolsets`` / ``delegation.child_bare``: operator-set child narrowing + bare prompt.

Both are operator-controlled, never model-controlled: the model never chooses capabilities (#56386).
``child_toolsets`` can only narrow (it is intersected with the parent's set), and unset keys must reproduce
today's child exactly.
"""

from types import SimpleNamespace

import pytest

from tools import delegate_tool_config
from tools.delegate_tool_toolsets import _resolve_child_toolsets

FAKE_ROUTE = dict(provider="openai-compat", base_url="http://127.0.0.1:9/v1", api_key="fake", model="m")


def _config(monkeypatch, **delegation):
    monkeypatch.setattr(delegate_tool_config, "_load_config", lambda: dict(delegation))


class TestChildToolsetsParsing:
    def test_unset_and_empty_inherit(self, monkeypatch):
        for value in ({}, {"child_toolsets": None}, {"child_toolsets": []}, {"child_toolsets": ["", "  "]}):
            _config(monkeypatch, **value)
            assert delegate_tool_config._get_child_toolsets() is None

    def test_a_bare_string_is_one_name_or_a_comma_list(self, monkeypatch):
        _config(monkeypatch, child_toolsets=" file ")
        assert delegate_tool_config._get_child_toolsets() == ["file"]
        _config(monkeypatch, child_toolsets="file, web")
        assert delegate_tool_config._get_child_toolsets() == ["file", "web"]

    def test_non_strings_and_blanks_are_dropped(self, monkeypatch):
        _config(monkeypatch, child_toolsets=["file", 3, "", " terminal "])
        assert delegate_tool_config._get_child_toolsets() == ["file", "terminal"]

    def test_child_bare_defaults_off_and_accepts_yaml_strings(self, monkeypatch):
        _config(monkeypatch)
        assert delegate_tool_config._get_child_bare() is False
        _config(monkeypatch, child_bare="true")
        assert delegate_tool_config._get_child_bare() is True

    def test_a_value_outside_the_parent_is_dropped(self, monkeypatch):
        """The config narrows only: a stale name can cost a toolset, never leak one."""
        _config(monkeypatch, child_toolsets=["file", "browser"])
        enabled, _disabled = _resolve_child_toolsets(
            SimpleNamespace(enabled_toolsets=["terminal", "file"]), delegate_tool_config._get_child_toolsets(), "leaf")
        assert enabled == ["file"]


    def test_a_list_that_strips_to_nothing_inherits(self, monkeypatch):
        """``memory`` is never granted to children: narrowing to it must not leave a toolless child."""
        _config(monkeypatch, child_toolsets=["memory"])
        enabled, _disabled = _resolve_child_toolsets(
            SimpleNamespace(enabled_toolsets=["terminal", "file"]), delegate_tool_config._get_child_toolsets(), "leaf")
        assert enabled == ["file", "terminal"]


@pytest.fixture()
def parent():
    from run_agent import AIAgent
    agent = AIAgent(**FAKE_ROUTE, enabled_toolsets=["terminal", "file", "skills"], quiet_mode=True,
                    skip_memory=True, skip_context_files=True, save_trajectories=False)
    yield agent
    agent.close()


class TestBuildChildAgent:
    def _build(self, parent, **overrides):
        from tools.delegate_tool import _build_child_agent
        return _build_child_agent(task_index=0, goal="g", context="c", toolsets=None, model=None,
                                  max_iterations=2, task_count=1, parent_agent=parent, **overrides)

    def test_unset_reproduces_the_inherited_child(self, parent, monkeypatch):
        """Equality against the no-config child, not just "some tools": a shrinking default must show."""
        _config(monkeypatch)
        baseline = self._build(parent)
        try:
            expected = list(baseline.enabled_toolsets)
        finally:
            baseline.close()
        child = self._build(parent)
        try:
            assert child.enabled_toolsets == expected and "terminal" in expected
            assert child.skip_skills_index is False
        finally:
            child.close()

    def test_a_narrowing_that_strips_to_nothing_falls_back_in_the_build(self, parent, monkeypatch):
        _config(monkeypatch, child_toolsets=["memory"])
        child = self._build(parent)
        try:
            assert "terminal" in child.enabled_toolsets and "file" in child.enabled_toolsets
        finally:
            child.close()

    def test_the_bare_override_wins_over_config_for_a_resume(self, parent, monkeypatch):
        """action='continue' pins the prompt shape the transcript was recorded under."""
        _config(monkeypatch, child_bare=False)
        child = self._build(parent, override_child_bare=True)
        try:
            assert child.skip_skills_index is True
        finally:
            child.close()

    def test_child_toolsets_narrows_the_child(self, parent, monkeypatch):
        _config(monkeypatch, child_toolsets=["file"])
        child = self._build(parent)
        try:
            assert child.enabled_toolsets == ["file"]
        finally:
            child.close()

    def test_child_bare_sets_the_prompt_lever(self, parent, monkeypatch):
        _config(monkeypatch, child_bare=True)
        child = self._build(parent)
        try:
            assert child.skip_skills_index is True
            assert child.load_soul_identity is False
        finally:
            child.close()


class TestHelpGuidanceSlot:
    def test_a_bare_child_keeps_the_skill_pointer(self, parent, monkeypatch):
        """It still has ``skill_view``; the no-skills variant ("point the user there") fits a chat, not a child."""
        from agent.system_prompt import HERMES_AGENT_HELP_GUIDANCE, build_system_prompt

        _config(monkeypatch, child_toolsets=["skills"], child_bare=True)
        child = _build_child_agent_quick(parent)
        try:
            assert "skill_view" in child.valid_tool_names
            assert HERMES_AGENT_HELP_GUIDANCE in build_system_prompt(child)
        finally:
            child.close()


def _build_child_agent_quick(parent):
    from tools.delegate_tool import _build_child_agent
    return _build_child_agent(task_index=0, goal="g", context="c", toolsets=None, model=None,
                              max_iterations=2, task_count=1, parent_agent=parent)


class TestSkillsIndexGate:
    def _agent(self, **extra):
        return SimpleNamespace(valid_tool_names={"skills_list"}, **extra)

    @staticmethod
    def _patch_builder(monkeypatch, value="INDEX"):
        from agent import system_prompt
        monkeypatch.setattr(system_prompt, "_pb",
                            SimpleNamespace(build_skills_system_prompt=lambda **_kw: value))
        return system_prompt

    def test_index_is_built_normally(self, monkeypatch):
        assert self._patch_builder(monkeypatch)._skills_prompt(self._agent()) == "INDEX"

    def test_bare_child_skips_the_index_without_losing_the_skills_tools(self, monkeypatch):
        system_prompt = self._patch_builder(monkeypatch)
        assert system_prompt._skills_prompt(self._agent(skip_skills_index=True)) == ""
        assert "skills_list" in self._agent(skip_skills_index=True).valid_tool_names
