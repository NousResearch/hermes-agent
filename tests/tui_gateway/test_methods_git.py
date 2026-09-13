"""Tests for the git.repo_refs JSON-RPC method (github.com ref autolinking)."""

from __future__ import annotations

import os
import subprocess

import pytest

import tui_gateway.server as server


def _call(method, params=None):
    handler = server._methods[method]
    resp = handler(1, params or {})
    assert "error" not in resp, resp.get("error")
    return resp["result"]


@pytest.fixture()
def _git_sessions(monkeypatch):
    """A fake live-session registry keyed by session id, plus a clean refs cache."""
    from tui_gateway import methods_git

    sessions = {}
    monkeypatch.setattr(server, "_sessions", sessions)
    methods_git._refs_cache.clear()
    return sessions


def _init_repo(path, origin=None):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    if origin:
        subprocess.run(
            ["git", "-C", str(path), "remote", "add", "origin", origin], check=True
        )


def test_methods_registered():
    assert "git.repo_refs" in server._methods


def test_session_id_required(_git_sessions):
    resp = server._methods["git.repo_refs"](1, {})
    assert "error" in resp


def test_unknown_session_is_unavailable(_git_sessions):
    result = _call("git.repo_refs", {"session_id": "nope"})
    assert result == {"available": False}


def test_resolves_github_origin(_git_sessions, tmp_path):
    repo = tmp_path / "newsAgg"
    repo.mkdir()
    _init_repo(repo, "https://github.com/KaptenKatthatt/newsAgg.git")
    _git_sessions["s1"] = {"cwd": str(repo)}

    result = _call("git.repo_refs", {"session_id": "s1"})
    assert result["available"] is True
    assert result["host"] == "github.com"
    assert result["owner"] == "KaptenKatthatt"
    assert result["repo"] == "newsAgg"


def test_resolves_ssh_origin(_git_sessions, tmp_path):
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    _init_repo(repo, "git@github.com:NousResearch/hermes-agent.git")
    _git_sessions["s1"] = {"cwd": str(repo)}

    result = _call("git.repo_refs", {"session_id": "s1"})
    assert result["available"] is True
    assert result["owner"] == "NousResearch"
    assert result["repo"] == "hermes-agent"


def test_worktree_cwd_folds_to_main_root(_git_sessions, tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    _init_repo(main, "https://github.com/o/r.git")
    subprocess.run(
        ["git", "-C", str(main), "commit", "--allow-empty", "-q", "-m", "init"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(main),
            "worktree",
            "add",
            "-q",
            str(tmp_path / "wt"),
            "-b",
            "side",
        ],
        check=True,
    )
    _git_sessions["s1"] = {"cwd": str(tmp_path / "wt")}

    result = _call("git.repo_refs", {"session_id": "s1"})
    assert result["available"] is True
    assert result["owner"] == "o"


def test_non_github_origin_is_unavailable(_git_sessions, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo, "https://gitlab.com/o/r.git")
    _git_sessions["s1"] = {"cwd": str(repo)}

    assert _call("git.repo_refs", {"session_id": "s1"}) == {"available": False}


def test_repo_without_origin_is_unavailable(_git_sessions, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    _git_sessions["s1"] = {"cwd": str(repo)}

    assert _call("git.repo_refs", {"session_id": "s1"}) == {"available": False}


def test_non_repo_cwd_is_unavailable(_git_sessions, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    _git_sessions["s1"] = {"cwd": str(plain)}

    assert _call("git.repo_refs", {"session_id": "s1"}) == {"available": False}


def test_refs_are_cached_per_repo_root(_git_sessions, tmp_path, monkeypatch):
    from tui_gateway import git_probe, methods_git

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo, "https://github.com/o/r.git")
    _git_sessions["s1"] = {"cwd": str(repo)}
    _call("git.repo_refs", {"session_id": "s1"})

    calls = {"n": 0}
    real_run_git = git_probe.run_git

    def counting_run_git(cwd, *args):
        if args[:2] == ("remote", "get-url"):
            calls["n"] += 1
        return real_run_git(cwd, *args)

    monkeypatch.setattr(git_probe, "run_git", counting_run_git)
    _call("git.repo_refs", {"session_id": "s1"})
    assert calls["n"] == 0
    assert methods_git._refs_cache


def test_kill_switch_disables(_git_sessions, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo, "https://github.com/o/r.git")
    _git_sessions["s1"] = {"cwd": str(repo)}

    import hermes_cli.config as config_mod

    real_load = config_mod.load_config

    def fake_load(*a, **kw):
        cfg = real_load(*a, **kw)
        cfg.setdefault("desktop", {})["autolink_issue_refs"] = False
        return cfg

    monkeypatch.setattr(config_mod, "load_config", fake_load)
    result = _call("git.repo_refs", {"session_id": "s1"})
    assert result == {"available": False}
