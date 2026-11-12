"""Regression tests for #148767: a Windows device-name entry (`nul`) crashed `complete.path`.

`os.path.join(root, "nul")` abspaths to the `\\\\.\\nul` device, so `ntpath.relpath` raises
``ValueError: path is on mount '\\\\.\\nul', start on mount 'C:'``. One such entry in the listing
turned the keystroke into a 5021 error instead of items; the `~/` spelling now shares that one
guarded call instead of computing a second `relpath`, and the `@name` fuzzy listing (`git ls-files`)
skips the same entry.

Covers:
  - a real file named `nul` in the listed directory: the rest of the listing still comes back
  - the same entry under a `~/` word (home-relative text)
  - the same entry in the `@name` fuzzy listing, which reads `git ls-files` rather than `listdir`
  - the platform-agnostic contract: whatever relpath refuses to relate is skipped, nothing else is
"""

from __future__ import annotations

import os
import subprocess

import pytest

from tui_gateway import server


def _texts(word: str) -> list[str]:
    resp = server.handle_request({"id": "1", "method": "complete.path", "params": {"word": word}})
    assert "result" in resp, resp
    return [item["text"] for item in resp["result"]["items"]]


def _create_device_named_file(directory) -> None:
    """Create a file literally named ``nul``: NTFS only accepts it through the ``\\\\?\\`` prefix."""
    with open("\\\\?\\" + os.path.join(os.fspath(directory), "nul"), "w") as handle:
        handle.write("x")


@pytest.fixture(autouse=True)
def _completion_root(tmp_path, monkeypatch):
    # `_completion_cwd` otherwise falls back to the launch profile's configured terminal.cwd, which is
    # read from the developer's real config rather than the test home (see
    # tests/gateway/test_complete_path_at_filter.py).
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: None)
    monkeypatch.chdir(tmp_path)
    server._fuzzy_cache.clear()
    yield
    server._fuzzy_cache.clear()


@pytest.mark.platforms("windows")
def test_device_named_entry_is_skipped_not_fatal(tmp_path):
    for name in ("alpha.md", "zeta.md"):
        (tmp_path / name).write_text("x")
    _create_device_named_file(tmp_path)  # sorts between them: the listing must continue past it

    assert os.listdir(tmp_path).count("nul") == 1
    texts = _texts("./")
    assert "./alpha.md" in texts
    assert "./zeta.md" in texts
    assert not any("nul" in text for text in texts)


@pytest.mark.platforms("windows")
def test_device_named_entry_is_skipped_for_home_words(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # `~/` text is spelled against home
    for name in ("alpha.md", "zeta.md"):
        (tmp_path / name).write_text("x")
    _create_device_named_file(tmp_path)

    texts = _texts("~/")
    assert "~/alpha.md" in texts
    assert "~/zeta.md" in texts


def test_unrelatable_entry_is_skipped_without_dropping_the_listing(tmp_path, monkeypatch):
    """Platform-agnostic twin: the guard is about relpath's refusal, not about `nul` itself."""
    for name in ("alpha.md", "beta.md"):
        (tmp_path / name).write_text("x")
    real_relpath = os.path.relpath

    def _relpath(path, start=None):
        if os.path.basename(str(path)) == "alpha.md":
            raise ValueError("path is on mount '\\\\.\\nul', start on mount 'C:'")
        return real_relpath(path, start)

    monkeypatch.setattr(os.path, "relpath", _relpath)
    texts = _texts("./")
    assert "./beta.md" in texts
    assert not any("alpha.md" in text for text in texts)


def _git_init(directory) -> None:
    """A repo is what routes the `@name` listing through `git ls-files`; `nul` needs no index entry,
    ``--others`` lists it (git itself refuses to add the file)."""
    run = subprocess.run(["git", "init", "-q", os.fspath(directory)], check=False, capture_output=True)
    if run.returncode != 0:
        pytest.skip("git unavailable")


@pytest.mark.platforms("windows")
def test_device_named_entry_is_skipped_by_the_at_name_listing(tmp_path):
    """`@al` ranks what `git ls-files --others` reports, which includes the unrelatable `nul`."""
    for name in ("alpha.md", "zeta.md"):
        (tmp_path / name).write_text("x")
    _create_device_named_file(tmp_path)
    _git_init(tmp_path)

    assert os.listdir(tmp_path).count("nul") == 1  # the case is present, not merely un-crashed
    texts = _texts("@al")
    assert "@file:alpha.md" in texts, texts
    assert not any("nul" in text for text in texts), texts


def test_unrelatable_git_entry_is_skipped_without_dropping_the_at_name_listing(tmp_path, monkeypatch):
    """Same contract without the device: a real `nul` file on any host, relpath refusing its name."""
    for name in ("alpha.md", "zeta.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "nul").write_text("x")
    if "nul" not in os.listdir(tmp_path):
        # Windows resolves `nul` to the device: no entry exists, so the native test above owns this.
        pytest.skip("host resolves `nul` to the device; nothing to list")
    _git_init(tmp_path)
    real_relpath = os.path.relpath

    def _relpath(path, start=None):
        if os.path.basename(str(path)).lower() == "nul":
            raise ValueError("path is on mount '\\\\.\\nul', start on mount 'C:'")
        return real_relpath(path, start) if start is not None else real_relpath(path)

    monkeypatch.setattr(os.path, "relpath", _relpath)
    texts = _texts("@al")
    assert "@file:alpha.md" in texts, texts
    assert not any("nul" in text for text in texts), texts
