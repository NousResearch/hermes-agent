"""The terminal tool refuses whole-tree walks (find/du/grep -r/rg) over a broad root."""

import json
from unittest.mock import MagicMock

import pytest

from tools.terminal_broad_walk_guard import broad_walk_error


@pytest.fixture(autouse=True)
def _roots(monkeypatch):
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv("HERMES_SEARCH_BROAD_ROOTS", "/mnt/big/ws:/workspace/hh")


@pytest.mark.parametrize("command", [
    "find /home/u /workspace /mnt/big -name \"*.toml\" -path \"*memory*\" 2>/dev/null | head -20",
    "cd /tmp && find /mnt/big -maxdepth 6 -name x",
    "timeout 60 find ~ -name '*.py'",
    "find / -name foo",
    "du -sh /workspace/hh",
    "grep -n -H -rn -e 'needle' -- /home/u/.x /workspace/hh",
    "rg -l needle /mnt/big",
    "sudo du -sx /",
    "ls; find $HOME -type f",
])
def test_refuses_broad_walks(command):
    err = broad_walk_error(command)
    assert err and "Refusing" in err


@pytest.mark.parametrize("command", [
    "find /mnt/big/ws/products/app -name '*.toml'",
    "find /mnt/big -maxdepth 2 -type d",
    "rg -l needle /mnt/big/ws/oss/repo",
    "rg --max-depth 2 -l needle /home/u",
    "du -sh /mnt/big/ws/data/x",
    "grep -n foo /home/u/file.txt",
    "ls -la /mnt/big /home/u",
    "find . -name '*.py'",
    "echo 'find / -name x'",
    "git grep -r foo",
    "cat 'unbalanced",
])
def test_allows_bounded_commands(command):
    assert broad_walk_error(command) is None


def _pre_exec(monkeypatch, command, env_type="local"):
    from tools import terminal_tool as tt
    monkeypatch.setattr(tt, "gateway_lifecycle_block", lambda **_: None)
    monkeypatch.setattr(tt, "self_repo_block", lambda **_: None)
    monkeypatch.setattr(tt, "locked_skill_block", lambda **_: None)
    tt._pre_exec_block(command, env=MagicMock(), env_type=env_type, cwd="/tmp",
                       workdir=None, session_key="s")


def test_terminal_pre_exec_blocks_broad_walk(monkeypatch):
    from tools.terminal_tool import _Rejected
    with pytest.raises(_Rejected) as exc:
        _pre_exec(monkeypatch, "find / -name '*.toml'")
    payload = json.loads(exc.value.args[0])
    assert payload["status"] == "blocked"
    assert "Refusing to walk /" in payload["error"]


def test_terminal_pre_exec_allows_project_walk(monkeypatch):
    _pre_exec(monkeypatch, "find /mnt/big/ws/products/app -name '*.toml'")


def test_terminal_pre_exec_skips_non_local_backends(monkeypatch):
    _pre_exec(monkeypatch, "find / -name x", env_type="docker")
