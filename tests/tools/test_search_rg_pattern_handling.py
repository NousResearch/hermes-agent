"""Regression tests for targeted ripgrep PCRE2 fallback."""

import shutil

import pytest

from tools.file_operations import ShellFileOperations
from tools.file_operations_common import ExecuteResult
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
        rg_executable="rg",
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


@pytest.mark.parametrize("header", ["rg: regex parse error:", "regex parse error:"])
def test_recommendation_reflow_still_enables_pcre2(header):
    diagnostic = (
        f"{header}\n"
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

    # Multiline I/O diagnostics can be classified as payload by the existing
    # output parser (depending on rg version and numeric path segments). This
    # regression guards retry routing, not that unrelated error-reporting bug.
    assert len(commands) == 1
    assert "--pcre2" not in commands[0]
    assert result.matches == []
    assert result.total_count == 0


@pytest.mark.parametrize("trigger", [
    "error: backreferences are not supported",
    "error: look-around, including look-ahead and look-behind, is not supported",
])
@pytest.mark.parametrize("prefix", ["", "rg: "])
def test_multiline_io_diagnostic_without_payload_does_not_retry(
    corpus, monkeypatch, trigger, prefix
):
    ops = _ops(corpus)
    calls = []
    diagnostic = f"{prefix}/missing path/prefix\n{trigger}\nsuffix: No such file or directory (os error 2)\n"

    def io_error(command, **kwargs):
        calls.append(command)
        return ExecuteResult(stdout=diagnostic, exit_code=2)

    monkeypatch.setattr(ops, "_exec", io_error)
    result = _rg(ops, "alpha", corpus)

    # No payload: exercise the diagnostic guard rather than short-circuiting
    # on a path fragment that the output parser mistakes for a match.
    assert result.error is not None
    assert len(calls) == 1
    assert "--pcre2" not in calls[0]


def test_pcre2_retry_preserves_offset_and_limit(corpus, monkeypatch):
    # A single file has stable line ordering; parallel directory traversal does
    # not promise the same order across two independent rg invocations.
    path = corpus / "pagination.txt"
    path.write_text("alpha foo first\nalpha bar excluded\nalpha foo second\nalpha foo third\n")
    ops = _ops(corpus)
    calls = []
    original = ops._exec

    def capture(command, *args, **kwargs):
        calls.append(command)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(ops, "_exec", capture)
    result = _rg(ops, r"alpha (?=foo)", path, limit=1, offset=1)

    assert result.error is None
    assert result.total_count == 2
    assert len(result.matches) == 1
    assert result.matches[0].path == str(path)
    assert result.matches[0].line_number == 3
    assert result.matches[0].content == "alpha foo second"
    search_calls = [command for command in calls if command != "rg --pcre2-version"]
    assert len(search_calls) == 2
    assert search_calls[1] == search_calls[0].replace("rg ", "rg --pcre2 ", 1)
    assert "head -n 2" in search_calls[0]


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


@pytest.mark.parametrize("output_mode", ["content", "files_only", "count"])
def test_public_search_pcre2_uses_resolved_executable(corpus, monkeypatch, output_mode):
    ops = _ops(corpus)
    calls = []
    original = ops._exec

    def capture(command, *args, **kwargs):
        calls.append(command)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(ops, "_exec", capture)
    for _ in range(2):
        result = ops.search(r"alpha (?=foo)", str(corpus), output_mode=output_mode)
        assert result.error is None
        assert result.total_count == 2
    executable = ops._rg_resolution_cache["rg"]
    quoted = ops._quote_executable(executable)
    assert calls.count(f"{quoted} --pcre2-version") == 1
    assert sum(command.startswith(f"set -o pipefail; {quoted} --pcre2 ")
               for command in calls) == 2


def test_pcre2_capability_cache_is_per_executable(corpus, monkeypatch):
    ops = _ops(corpus)
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        return ExecuteResult(stdout="", exit_code=0 if "enabled rg" in command else 2)

    monkeypatch.setattr(ops, "_exec", probe)
    for _ in range(2):
        assert ops._rg_supports_pcre2("/backend/enabled rg")
        assert not ops._rg_supports_pcre2("/backend/disabled rg")
    assert calls == [
        f"{ops._quote_executable('/backend/enabled rg')} --pcre2-version",
        f"{ops._quote_executable('/backend/disabled rg')} --pcre2-version",
    ]
