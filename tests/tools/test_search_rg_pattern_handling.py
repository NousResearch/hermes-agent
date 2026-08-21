"""Regression tests for targeted ripgrep PCRE2 fallback."""

import shutil

import pytest

from tools.file_operations import ShellFileOperations
from tools.file_operations_search import _rg_diagnostic_requires_pcre2
from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="requires ripgrep")


def _ops(root):
    return ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.txt").write_text("alpha foo omega\nalpha bar omega\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("alpha foo foo omega\n")
    return tmp_path


def _rg(ops, pattern, root, **kwargs):
    return ops._search_with_rg(
        pattern,
        str(root),
        kwargs.get("file_glob"),
        kwargs.get("limit", 50),
        kwargs.get("offset", 0),
        kwargs.get("output_mode", "content"),
        kwargs.get("context", 0),
    )


def test_ordinary_rust_regex_keeps_single_fast_path(corpus, monkeypatch):
    ops = _ops(corpus)
    commands = []
    original = ops._exec

    def capture(command, *args, **kwargs):
        commands.append(command)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(ops, "_exec", capture)
    result = _rg(ops, r"alpha (foo|bar)", corpus)

    assert result.error is None
    assert result.total_count == 3
    assert len(commands) == 1
    assert "--pcre2" not in commands[0]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (r"alpha (?=foo)", 2),
        (r"alpha (foo) \1 omega", 1),
    ],
)
def test_unsupported_rust_regex_retries_once_with_pcre2(corpus, pattern, expected):
    result = _rg(_ops(corpus), pattern, corpus)

    assert result.error is None
    assert result.total_count == expected


def test_invalid_regex_preserves_normal_error_without_pcre2_retry(corpus, monkeypatch):
    ops = _ops(corpus)
    commands = []
    original = ops._exec

    def capture(command, *args, **kwargs):
        commands.append(command)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(ops, "_exec", capture)
    result = _rg(ops, "[", corpus)

    assert result.error is not None
    assert len(commands) == 1
    assert "--pcre2" not in commands[0]


def test_trigger_phrase_inside_pattern_does_not_enable_pcre2(corpus, monkeypatch):
    ops = _ops(corpus)
    commands = []
    original = ops._exec

    def capture(command, *args, **kwargs):
        commands.append(command)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(ops, "_exec", capture)
    pattern = r"(?P<x>a)(?P=x)(?# backreferences are not supported)"
    result = _rg(ops, pattern, corpus)

    assert result.error is not None
    assert len(commands) == 1
    assert "--pcre2" not in commands[0]


def test_indented_echoed_error_line_does_not_enable_pcre2():
    diagnostic = (
        "rg: regex parse error:\n"
        "    error: backreferences are not supported\n"
        "error: unclosed character class"
    )
    assert not _rg_diagnostic_requires_pcre2(diagnostic)


@pytest.mark.parametrize(
    "trigger",
    [
        "error: backreferences are not supported",
        "error: look-around, including look-ahead and look-behind, is not supported",
    ],
)
def test_trigger_phrase_in_multiline_missing_path_does_not_enable_pcre2(
    corpus, monkeypatch, trigger
):
    ops = _ops(corpus)
    commands = []
    original = ops._exec

    def capture(command, *args, **kwargs):
        commands.append(command)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(ops, "_exec", capture)
    missing = corpus / f"prefix\n{trigger}\nsuffix"
    result = _rg(ops, "alpha", missing)

    assert result.error is not None
    assert len(commands) == 1
    assert "--pcre2" not in commands[0]


def test_pcre2_retry_preserves_glob_context_and_path(corpus):
    result = _rg(
        _ops(corpus),
        r"alpha (?=foo)",
        corpus,
        file_glob="*.py",
        context=1,
    )

    assert result.error is None
    assert result.total_count == 1
    assert all(match.path.endswith("b.py") for match in result.matches)
