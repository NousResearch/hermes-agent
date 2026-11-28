"""Regression for #156164 item 5: an archive pointer must survive a home or profile change.

A stub stores the archive as a home-relative display path when the prune runs. Resuming that
session under another ``HERMES_HOME`` (a profile, a different install) made ``read_file`` name a
path that never existed there and report the archive expired, although the bytes were untouched in
the original home. Resolving the file name against the current home keeps the promise; a pointer
whose file is genuinely gone still reports expired.
"""
from tools.tool_result_storage import rehome_archive_path


def _archive(home, subdir, name, text="archived body"):
    target = home / subdir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_a_pointer_from_another_home_resolves_to_this_homes_copy(tmp_path, monkeypatch):
    """The display path of a previous home resolves by file name in the current one."""
    current = tmp_path / "profile"
    current.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(current))
    # The transcript says ~/.hermes/cache/pruned/<name>; that literal path does not exist here.
    expected = _archive(current, "cache/pruned", "s_1_terminal_ab12cd34.json")

    resolved = rehome_archive_path("~/.hermes/cache/pruned/s_1_terminal_ab12cd34.json")

    assert resolved == str(expected)
    assert resolved != "~/.hermes/cache/pruned/s_1_terminal_ab12cd34.json"


def test_an_absolute_pointer_from_another_home_also_resolves(tmp_path, monkeypatch):
    """The absolute host form recorded by a different install resolves the same way."""
    current = tmp_path / "profile"
    current.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(current))
    expected = _archive(current, "cache/spillover", "result_9f.txt")

    assert rehome_archive_path("/home/other/.hermes/cache/spillover/result_9f.txt") == str(expected)


def test_a_genuinely_expired_archive_still_resolves_to_nothing(tmp_path, monkeypatch):
    """No file anywhere in this home: the caller must keep reporting it expired, not resurrect it."""
    current = tmp_path / "profile"
    current.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(current))

    assert rehome_archive_path("~/.hermes/cache/pruned/never_written.json") is None


def test_a_path_that_is_not_an_archive_is_left_alone(tmp_path, monkeypatch):
    """Ordinary file paths never enter the archive re-resolution."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert rehome_archive_path("~/notes/todo.md") is None
    assert rehome_archive_path("/etc/hosts") is None
    assert rehome_archive_path("") is None


def test_windows_separators_in_a_recorded_pointer_still_match(tmp_path, monkeypatch):
    """A transcript written on Windows carries backslashes; the name segment still matches."""
    current = tmp_path / "profile"
    current.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(current))
    expected = _archive(current, "cache/pruned", "win_case_0a.txt")

    assert rehome_archive_path(r"C:\Users\x\.hermes\cache\pruned\win_case_0a.txt") == str(expected)