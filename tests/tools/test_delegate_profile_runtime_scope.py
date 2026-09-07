"""Phase 1 RED contract for target-profile runtime scope."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from agent.profile_runtime_scope import profile_runtime_scope
from agent.secret_scope import current_secret_scope
from hermes_constants import get_hermes_home
from tools import delegate_tool as dt


def _creds():
    return {
        "provider": "parent-provider",
        "base_url": "https://parent.invalid/v1",
        "api_key": "parent-secret",
        "api_mode": "chat_completions",
        "request_overrides": {},
        "max_output_tokens": None,
        "command": None,
        "args": None,
        "model": "",
    }


def test_target_profile_scope_applies_profile_tools_not_parent_defaults(monkeypatch):
    seen = []

    def fake_builder(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_builder)
    children, error = dt._build_children(
        [{"goal": "use target profile", "context": "task"}],
        [None],
        _creds(),
        top_role="leaf",
        max_iterations=1,
        parent_agent=SimpleNamespace(),
        live_deleg_id=None,
        live_writers=[],
        profile="target-profile",
        profile_content={
            "name": "target-profile",
            "config": {"model": {"default": "target-model"}, "provider": "target-provider"},
            "soul_md": "target identity",
            "skills": ["target-toolset"],
        },
    )
    assert error is None
    assert children
    assert seen[0]["model"] == "target-model"
    assert seen[0]["toolsets"] == ["target-toolset"]
    assert seen[0]["override_provider"] == "target-provider"


def test_target_profile_credentials_are_not_inherited_from_parent(monkeypatch):
    seen = []

    def fake_builder(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_builder)
    children, error = dt._build_children(
        [{"goal": "use target secret", "context": None}],
        [None],
        _creds(),
        top_role="leaf",
        max_iterations=1,
        parent_agent=SimpleNamespace(),
        live_deleg_id=None,
        live_writers=[],
        profile="target-profile",
        profile_content={
            "name": "target-profile",
            "config": {
                "provider": "target-provider",
                "credentials": {"api_key": "target-secret"},
            },
            "soul_md": None,
            "skills": [],
        },
    )
    assert error is None
    assert children
    assert seen[0]["override_api_key"] == "target-secret"
    assert seen[0]["override_api_key"] != "parent-secret"


def test_profile_scope_is_isolated_per_concurrent_child():
    barrier = Barrier(2)
    homes = [Path("/tmp/profile-a"), Path("/tmp/profile-b")]
    secrets = [{"TARGET_SECRET": "a"}, {"TARGET_SECRET": "b"}]

    def worker(index):
        with profile_runtime_scope(homes[index], secrets[index], hydrate_secrets=False):
            barrier.wait(timeout=5)
            scope = current_secret_scope()
            assert scope is not None
            return get_hermes_home(), scope["TARGET_SECRET"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        observed = [future.result(timeout=10) for future in [pool.submit(worker, 0), pool.submit(worker, 1)]]

    assert observed == [(homes[0], "a"), (homes[1], "b")]
    assert current_secret_scope() is None
    assert callable(getattr(dt, "_child_profile_scope", None))


def test_profile_scope_restores_home_and_secret_after_exception(tmp_path):
    before_home = get_hermes_home()
    before_scope = current_secret_scope()
    with pytest.raises(RuntimeError, match="scope failure"):
        with profile_runtime_scope(tmp_path / "target", {"TARGET_SECRET": "target"}, hydrate_secrets=False):
            assert get_hermes_home() == tmp_path / "target"
            assert current_secret_scope() == {"TARGET_SECRET": "target"}
            raise RuntimeError("scope failure")
    assert get_hermes_home() == before_home
    assert current_secret_scope() == before_scope


def test_profile_scope_does_not_mutate_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TARGET_SECRET", "ambient-parent-secret")
    before = dict(os.environ)
    with profile_runtime_scope(tmp_path / "target", {"TARGET_SECRET": "target-secret"}, hydrate_secrets=False):
        assert os.environ == before
        assert current_secret_scope() == {"TARGET_SECRET": "target-secret"}
    assert os.environ == before


def test_child_execution_enters_target_profile_scope(tmp_path, monkeypatch):
    child = SimpleNamespace(_delegate_profile_home=tmp_path / "target")
    observed = []

    def fake_run(*args, **kwargs):
        observed.append((get_hermes_home(), current_secret_scope()))
        return {"status": "completed"}

    monkeypatch.setattr(dt, "_run_single_child_impl", fake_run)
    result = dt._run_single_child(0, "profile task", child, SimpleNamespace())

    assert result == {"status": "completed"}
    assert observed[0][0] == tmp_path / "target"
    assert observed[0][1] == {}
    assert current_secret_scope() is None
