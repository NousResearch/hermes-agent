"""File-tool Python snippets must preserve filenames beyond the BMP."""

import json

import pytest

from tools import file_tools
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


class ShellTransport:
    """Remote-style shell contract executed locally; no cloud service is used."""

    def __init__(self, env):
        self.env = env
        self.cwd = env.cwd

    def execute(self, *args, **kwargs):
        return self.env.execute(*args, **kwargs)


@pytest.fixture(params=["local", "shell"])
def file_backend(request, tmp_path, monkeypatch):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    backend = env if request.param == "local" else ShellTransport(env)
    ops = ShellFileOperations(backend)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda *args, **kwargs: ops)
    try:
        yield
    finally:
        env.cleanup()


@pytest.mark.parametrize("name", ["plain.txt", "报告-😀-𠀀 'quoted'.txt"])
def test_patch_delete_preserves_filename(file_backend, tmp_path, name):
    target = tmp_path / name
    target.write_text("delete me\n", encoding="utf-8")
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep me\n", encoding="utf-8")
    patch = f"*** Begin Patch\n*** Delete File: {target}\n*** End Patch"

    result = json.loads(registry.get_entry("patch").handler(
        {"mode": "patch", "patch": patch}, task_id="unicode-delete"))

    assert result.get("success"), result
    assert not target.exists()
    assert sibling.read_text(encoding="utf-8") == "keep me\n"


@pytest.mark.parametrize("name", ["plain.txt", "报告-😀-𠀀 'quoted'.txt"])
def test_read_utf16_preserves_filename(file_backend, tmp_path, name):
    target = tmp_path / name
    original = "alpha\nbeta\n".encode("utf-16")
    target.write_bytes(original)

    result = json.loads(registry.get_entry("read_file").handler(
        {"path": str(target)}, task_id="unicode-read"))

    assert not result.get("error"), result
    assert "alpha" in result["content"] and "beta" in result["content"]
    assert "UTF-16" in result["hint"]
    assert target.read_bytes() == original
