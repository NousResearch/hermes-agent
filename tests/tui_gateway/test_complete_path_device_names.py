"""Regression tests for #148767: a Windows device-name entry (`nul`) crashed `complete.path`.

`os.path.join(root, "nul")` abspaths to the `\\\\.\\nul` device, so `ntpath.relpath` raises
``ValueError: path is on mount '\\\\.\\nul', start on mount 'C:'``. One such entry in the listing
turned the keystroke into a 5021 error instead of items, and the `~/` text path called relpath a
second time, so both sites had to go.

Covers:
  - a real file named `nul` in the listed directory: the rest of the listing still comes back
  - the same entry under a `~/` word (home-relative text)
  - the platform-agnostic contract: whatever relpath refuses to relate is skipped, nothing else is
"""

from __future__ import annotations

import os

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
