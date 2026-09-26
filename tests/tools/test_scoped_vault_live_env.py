"""Skill grants must survive tool workers without crossing conversation/profile boundaries."""

import contextvars
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.secret_scope import load_env_file, set_secret_scope
from hermes_constants import set_hermes_home_override
from tools.code_execution_env import _scrub_child_env
from tools.environments.local import _sanitize_subprocess_env
from tools.thread_context import propagate_context_to_thread


def _profile(tmp_path, name):
    home = tmp_path / name
    skill = home / "skills" / "access"
    skill.mkdir(parents=True)
    (home / "config.yaml").write_text("terminal:\n  env_passthrough: []\n")
    (home / ".env").write_text(f"CLIENT_LOGIN={name}\nCLIENT_PASSWORD={name}-secret\n")
    (skill / "SKILL.md").write_text(
        "---\nname: access\ndescription: Client access\n"
        "required_environment_variables:\n"
        "  - name: CLIENT_LOGIN\n    optional: true\n"
        "  - name: CLIENT_PASSWORD\n    optional: true\n"
        "  - name: OPENAI_API_KEY\n    optional: true\n"
        "---\nUse client credentials without displaying them.\n"
    )
    ctx = contextvars.Context()  # No fixture-preinitialized passthrough set.
    ctx.run(set_hermes_home_override, home)
    ctx.run(set_secret_scope, load_env_file(home / ".env"), profile_home=str(home))
    return home, ctx


def _worker(ctx, callback):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run(propagate_context_to_thread, callback)).result(timeout=30)


def _load_skill():
    # Real registry dispatch, including readiness registration.
    import tools.skills_tool  # noqa: F401
    from tools.registry import registry
    result = json.loads(registry.get_entry("skill_view").handler({"name": "access"}))
    assert result["success"], result


def _read_child(builder):
    env = builder(dict(os.environ))
    assert "OPENAI_API_KEY" not in env
    code = "import os,json; print(json.dumps([os.getenv('CLIENT_LOGIN'),os.getenv('CLIENT_PASSWORD')]))"
    return json.loads(subprocess.check_output([sys.executable, "-c", code], env=env, text=True, timeout=10))


@pytest.mark.parametrize("builder", [_sanitize_subprocess_env, _scrub_child_env])
def test_skill_grant_survives_worker_and_live_rotation(tmp_path, monkeypatch, builder):
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.delenv("CLIENT_LOGIN", raising=False)
    monkeypatch.delenv("CLIENT_PASSWORD", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "provider-must-stay-private")
    home, ctx = _profile(tmp_path, "a")
    # The very first tool is skill_view; registration must reach the next worker.
    _worker(ctx, _load_skill)
    assert _worker(ctx, lambda: _read_child(builder)) == ["a", "a-secret"]
    (home / ".env").write_text("CLIENT_LOGIN=a\nCLIENT_PASSWORD=rotated\n")
    ctx.run(set_secret_scope, load_env_file(home / ".env"), profile_home=str(home))
    assert _worker(ctx, lambda: _read_child(builder)) == ["a", "rotated"]
    (home / ".env").write_text("CLIENT_LOGIN=a\n")
    ctx.run(set_secret_scope, load_env_file(home / ".env"), profile_home=str(home))
    assert _worker(ctx, lambda: _read_child(builder)) == ["a", None]


@pytest.mark.parametrize("builder", [_sanitize_subprocess_env, _scrub_child_env])
def test_grants_stay_in_their_conversation_and_values_in_their_profile(tmp_path, monkeypatch, builder):
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.delenv("CLIENT_LOGIN", raising=False)
    monkeypatch.delenv("CLIENT_PASSWORD", raising=False)
    _, a = _profile(tmp_path, "a")
    _, b = _profile(tmp_path, "b")
    _worker(a, _load_skill)
    assert _worker(b, lambda: _read_child(builder)) == [None, None]
    _worker(b, _load_skill)
    assert _worker(b, lambda: _read_child(builder)) == ["b", "b-secret"]
    assert _worker(a, lambda: _read_child(builder)) == ["a", "a-secret"]
