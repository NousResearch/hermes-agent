import json

import pytest

from tools import file_tools as ft
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


@pytest.fixture
def ops(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    backend = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": backend)
    return backend


@pytest.mark.platforms("posix")
def test_create_only_dispatch_refuses_existing_entries(ops, tmp_path):
    target = tmp_path / "original.txt"
    target.write_bytes(b"original\n")
    empty = tmp_path / "empty.txt"
    empty.touch()
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    dangling = tmp_path / "dangling.txt"
    dangling.symlink_to(tmp_path / "absent.txt")
    directory = tmp_path / "directory"
    directory.mkdir()
    for entry in (target, empty, link, dangling, directory):
        out = json.loads(registry.dispatch("write_file", {
            "path": str(entry), "content": "replacement\n", "create_only": True}, task_id=str(entry)))
        assert out.get("status") == "already_exists", out
    assert target.read_bytes() == b"original\n"
    assert empty.read_bytes() == b""
    assert link.is_symlink() and dangling.is_symlink() and directory.is_dir()
    assert not (tmp_path / "absent.txt").exists()


def test_concurrent_create_only_publishes_one_complete_file(ops, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    target = tmp_path / "nested" / "new.txt"
    candidates = [f"candidate-é-{i}\n" * 1000 for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda content: ops.write_file(str(target), content, create_only=True), candidates))
    winners = [i for i, result in enumerate(results) if not result.error]
    assert len(winners) == 1, results
    assert target.read_bytes() == candidates[winners[0]].encode()
    assert all(result.status == "already_exists" for result in results if result.error)
    assert not list(target.parent.glob(".hermes-create-*"))
