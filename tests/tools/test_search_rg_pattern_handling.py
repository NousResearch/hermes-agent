"""Regression tests for targeted ripgrep PCRE2 fallback."""

import shutil

import pytest

from tools.file_operations import (
    ExecuteResult,
    ShellFileOperations,
    _rg_diagnostic_requires_pcre2,
)
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


def test_rg_without_pcre2_is_probed_once_and_not_retried(corpus, monkeypatch):
    ops = _ops(corpus)
    commands = []
    parser_error = (
        "rg: regex parse error:\n"
        "error: look-around, including look-ahead and look-behind, is not supported\n"
        "consider enabling PCRE2 with the --pcre2 flag, which can handle "
        "backreferences and look-around."
    )

    def no_pcre2(command, *args, **kwargs):
        commands.append(command)
        if command == "rg --pcre2-version":
            return ExecuteResult(stdout="PCRE2 is not available", exit_code=2)
        return ExecuteResult(stdout=parser_error, exit_code=2)

    monkeypatch.setattr(ops, "_exec", no_pcre2)

    first = _rg(ops, r"alpha (?=foo)", corpus)
    second = _rg(ops, r"alpha (?=foo)", corpus)

    assert first.error is not None
    assert second.error is not None
    assert commands.count("rg --pcre2-version") == 1
    assert not any(" --pcre2 " in command for command in commands)


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


def test_recommendation_reflow_still_enables_pcre2():
    diagnostic = (
        "rg: regex parse error:\n"
        "error: backreferences are not supported\n"
        "consider enabling PCRE2 with the --pcre2 flag, which can handle\n"
        "backreferences and look-around."
    )
    assert _rg_diagnostic_requires_pcre2(diagnostic)


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


def test_pcre2_retry_preserves_offset_and_limit(corpus):
    ops = _ops(corpus)
    full = _rg(
        ops,
        r"alpha (?=foo)",
        corpus,
        limit=2,
    )
    result = _rg(
        ops,
        r"alpha (?=foo)",
        corpus,
        limit=1,
        offset=1,
    )

    assert full.error is None
    assert result.error is None
    assert result.total_count == 2
    assert len(result.matches) == 1
    assert result.matches[0] == full.matches[1]


def test_pcre2_retry_reuses_original_shell_template_and_timeout(corpus, monkeypatch):
    ops = _ops(corpus)
    calls = []
    original = ops._exec

    def capture(command, *args, **kwargs):
        calls.append((command, kwargs.get("timeout")))
        return original(command, *args, **kwargs)

    monkeypatch.setattr(ops, "_exec", capture)
    result = _rg(ops, r"alpha (?=foo)", corpus)

    assert result.error is None
    search_calls = [call for call in calls if call[0] != "rg --pcre2-version"]
    assert search_calls[1][0] == search_calls[0][0].replace("rg ", "rg --pcre2 ", 1)
    assert search_calls[0][1] == search_calls[1][1] == 60


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
