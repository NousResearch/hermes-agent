"""Registry dispatch must keep instruction dedup state inside its profile."""
import json
from contextlib import contextmanager

from agent.secret_scope import (
    reset_multiplex_context, reset_secret_scope, set_multiplex_context, set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import skills_tool  # registers the real handler
from tools.registry import registry


@contextmanager
def _profile(home):
    token = set_hermes_home_override(home)
    secrets = set_secret_scope({}, profile_home=str(home))
    multiplex = set_multiplex_context(True)
    try:
        yield
    finally:
        reset_multiplex_context(multiplex)
        reset_secret_scope(secrets)
        reset_hermes_home_override(token)


def _homes(tmp_path):
    homes = [tmp_path / name for name in ("alpha", "beta")]
    for home in homes:
        skill = home / "skills" / "instructions"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: instructions\ndescription: Profile instructions\n---\n{home.name} procedure\n")
    return homes


def _view():
    return json.loads(registry.dispatch("skill_view", {"name": "instructions"}, task_id="same-task"))


def test_profile_switch_serves_its_own_instructions_and_preserves_each_cache(tmp_path):
    alpha, beta = _homes(tmp_path)
    skills_tool.reset_skill_view_dedup()
    try:
        for home in (alpha, beta):
            with _profile(home):
                first = _view()
                assert f"{home.name} procedure" in first.get("content", "")
                assert _view().get("dedup") is True
        with _profile(alpha):
            assert _view().get("dedup") is True
    finally:
        skills_tool.reset_skill_view_dedup()


def test_task_reset_only_invalidates_the_current_profiles_loaded_instructions(tmp_path):
    alpha, beta = _homes(tmp_path)
    skills_tool.reset_skill_view_dedup()
    try:
        for home in (alpha, beta):
            with _profile(home):
                assert f"{home.name} procedure" in _view().get("content", "")
        with _profile(alpha):
            skills_tool.reset_skill_view_dedup("same-task")
            assert "alpha procedure" in _view().get("content", "")
        with _profile(beta):
            assert _view().get("dedup") is True
    finally:
        skills_tool.reset_skill_view_dedup()
