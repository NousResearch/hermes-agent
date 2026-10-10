"""Git consumers use the selected package environment without owning its layout."""
import importlib
import shutil
import subprocess

import pytest

import pm
from tools import checkpoint_manager, working_diff


@pytest.fixture(autouse=True)
def managed_git_available(monkeypatch):
    # Exercise acquisition on this POSIX test host without changing its OS.
    from pm.registry import get_package
    from hermes_cli import _subprocess_compat as compat

    monkeypatch.setattr(compat, "_git_fallback_env_cache", {})
    monkeypatch.setattr(get_package("git"), "gaps", {})


@pytest.mark.parametrize("target", [
    "linux-x64", "linux-arm64", "linux-x64-musl", "linux-arm64-musl",
    "darwin-x64", "darwin-arm64",
])
def test_platform_gap_skips_worker_even_with_changing_environment(monkeypatch, target):
    from hermes_cli import _subprocess_compat as compat
    from pm.registry import get_package
    from pm import store
    import pm.client

    package = get_package("git")
    monkeypatch.setattr(package, "gaps", type(package).gaps)
    assert package.missing_reason(target) is not None
    monkeypatch.setattr(store, "current_target", lambda: target)
    calls = []
    acquire = pm.ensure

    def ensure(*args, **kwargs):
        calls.append(True)
        return acquire(*args, **kwargs)

    def request(*args, **kwargs):
        raise RuntimeError("a platform gap must not spawn a worker")

    monkeypatch.setattr(pm, "ensure", ensure)
    monkeypatch.setattr(pm.client, "_request", request)
    for turn in range(110):
        base = {"PATH": "/usr/bin", "TURN": str(turn)}
        selected = compat.selected_git_env(base)
        assert selected == base
        assert selected is not base
    assert calls == []


@pytest.mark.platforms("posix")
def test_checkpoint_and_diff_follow_selected_git_environment(tmp_path, monkeypatch):
    git = shutil.which("git")
    assert git is not None
    selected = tmp_path / "selected-tools"
    selected.mkdir()
    observed = tmp_path / "git-environment"
    wrapper = selected / "git"
    import shlex

    wrapper.write_text(
        '#!/bin/sh\n'
        'test "$HERMES_GIT_HELPER" = "selected" || exit 77\n'
        f'printf "%s\\n" selected >> {shlex.quote(str(observed))}\n'
        f'exec {shlex.quote(git)} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([git, "init", str(repo)], check=True, capture_output=True)
    (repo / "new.txt").write_text("new content\n", encoding="utf-8")

    def selected_git(name, *, base_env):
        assert name == "git"
        return pm.Runner(name, {
            **base_env, "PATH": str(selected), "HERMES_GIT_HELPER": "selected"
        })

    monkeypatch.setattr(pm, "ensure", selected_git)
    monkeypatch.setenv("PATH", str(tmp_path / "missing-path"))
    monkeypatch.setattr(checkpoint_manager, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    manager = checkpoint_manager.CheckpointManager(enabled=True)
    assert manager.ensure_checkpoint(str(repo)) is True
    result = working_diff.collect_working_diff(str(repo))
    assert result["success"] is True
    assert "+new content" in result["diff"]
    assert observed.read_text(encoding="utf-8").splitlines()

    # A changed PM selection must affect this same process, not a cached command.
    monkeypatch.setattr(pm, "ensure", lambda name, *, base_env: pm.Runner(name, {
        **base_env, "PATH": str(tmp_path / "missing-path")
    }))
    assert working_diff.collect_working_diff(str(repo))["success"] is False
    manager.new_turn()
    assert manager.ensure_checkpoint(str(repo)) is False


@pytest.mark.parametrize("consumer", ["checkpoint", "diff"])
@pytest.mark.parametrize("failure", ["unsupported-target", "acquisition"])
def test_git_consumers_fall_back_to_system_git(tmp_path, monkeypatch, consumer, failure):
    git = shutil.which("git")
    assert git is not None
    repo = tmp_path / "repo"
    subprocess.run([git, "init", str(repo)], check=True, capture_output=True)
    (repo / "new.txt").write_text("system git content\n", encoding="utf-8")

    # Fail at the PM boundary reached by both ensure and env_for, without
    # pretending this interpreter is running on a different host OS/CPU.
    resolver = importlib.import_module("pm.install")
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(True)
        if failure == "unsupported-target":
            raise RuntimeError("unsupported architecture")
        raise pm.InstallError("git", "managed Git unavailable")

    boundary = "current_target" if failure == "unsupported-target" else "_installed_location"
    monkeypatch.setattr(resolver, boundary, unavailable)
    monkeypatch.setattr(checkpoint_manager, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    if consumer == "checkpoint":
        manager = checkpoint_manager.CheckpointManager(enabled=True)
        assert manager.ensure_checkpoint(str(repo)) is True
    else:
        result = working_diff.collect_working_diff(str(repo))
        assert result["success"] is True
        assert "+system git content" in result["diff"]
    assert calls


def test_failed_git_environment_lookup_is_retried_after_bounded_cache(monkeypatch):
    compat = importlib.import_module("hermes_cli._subprocess_compat")
    selected_git_env = compat.selected_git_env

    calls = []
    now = [100.0]
    base = {"PATH": "/missing", "HERMES_GIT_CACHE_TEST": "unique"}

    def unavailable_once(name, *, base_env):
        calls.append(name)
        if len(calls) < 3:
            raise RuntimeError("managed Git unavailable")
        return pm.Runner(name, {**base_env, "PATH": "/managed"})

    monkeypatch.setattr(pm, "ensure", unavailable_once)
    monkeypatch.setattr(compat.time, "monotonic", lambda: now[0])

    first = selected_git_env(base)
    second = selected_git_env(base)
    assert calls == ["git"]
    assert first == base
    assert second == base
    assert second is not first

    now[0] += compat._GIT_FALLBACK_CACHE_TTL
    third = selected_git_env(base)
    assert calls == ["git", "git"]
    assert third == base

    now[0] += compat._GIT_FALLBACK_CACHE_TTL
    recovered = selected_git_env(base)
    assert calls == ["git", "git", "git"]
    assert recovered["PATH"] == "/managed"
    assert recovered is not third
