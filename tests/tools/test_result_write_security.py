"""Public persistence boundary: private output without clobbering outside files."""

import stat
import subprocess
from pathlib import Path

import pytest

from tools import tool_result_storage as storage


class ShellSandbox:
    """Exercise the remote fallback with real shell I/O, not a live backend."""

    def __init__(self, root):
        self.root = root

    def get_temp_dir(self):
        return str(self.root)

    def execute(self, command, timeout, stdin_data=None):
        if command.startswith("test -r "):
            return {"returncode": 1, "output": ""}  # no cache mount
        result = subprocess.run(
            ["/bin/bash", "-c", command], input=stdin_data, text=True,
            capture_output=True, timeout=timeout, cwd=self.root, check=False,
        )
        return {"returncode": result.returncode, "output": result.stdout + result.stderr}


@pytest.fixture
def destinations(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return tmp_path, storage.get_spillover_dir(), tmp_path / "hermes-results"


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("acl", [pytest.param(False, marks=pytest.mark.platforms("posix")), pytest.param(True, marks=pytest.mark.platforms("macos"))])
def test_persisted_content_is_private_and_lossless(destinations, remote, acl):
    root, host, sandbox = destinations
    directory = sandbox if remote else host
    directory.mkdir(parents=True, mode=0o755)
    target = directory / "private.txt"
    target.write_text("stale", encoding="utf-8")
    target.chmod(0o644)
    if acl:
        subprocess.run(["chmod", "+a", "everyone allow read,write,execute,file_inherit,directory_inherit", str(directory)], check=True)
    content = "héllo sensitive output\n" * 100
    result = storage.maybe_persist_tool_result(content, "terminal", "private", ShellSandbox(root) if remote else None, threshold=0)
    assert storage.extract_persisted_path(result) == str(target)
    assert target.read_text(encoding="utf-8") == content
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    if acl:
        listing = subprocess.run(["ls", "-lde", str(directory), str(target)], capture_output=True, text=True, check=True).stdout
        assert "group:everyone" not in listing


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("linked_directory", [False, True])
def test_persistence_never_writes_through_preplanted_links(destinations, remote, linked_directory):
    root, host, sandbox = destinations
    outside = root / "outside"
    outside.mkdir()
    victim = outside / "victim.txt"
    victim.write_text("outside must survive", encoding="utf-8")
    directory = sandbox if remote else host
    directory.parent.mkdir(parents=True, exist_ok=True)
    if linked_directory:
        directory.symlink_to(outside, target_is_directory=True)
    else:
        directory.mkdir()
        (directory / "victim.txt").symlink_to(victim)
    result = storage.maybe_persist_tool_result("replacement", "terminal", "victim", ShellSandbox(root) if remote else None, threshold=0)
    assert victim.read_text(encoding="utf-8") == "outside must survive"
    if linked_directory:
        assert storage.extract_persisted_path(result) is None
    else:
        saved = storage.extract_persisted_path(result)
        assert saved is not None
        target = Path(saved)
        assert not target.is_symlink()
        assert target.read_text(encoding="utf-8") == "replacement"
