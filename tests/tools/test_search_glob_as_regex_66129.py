"""Regression tests for glob-as-regex recovery in content search (#66129).

Local models (e.g. Qwen3.6 35B A3B on llama.cpp) sometimes call
``search_files`` with ``target="content"`` and a glob-style pattern where rg
expects a regex.  rg fails with ``repetition operator missing expression``
and the session loops as the model retries the same call.

The fix detects that signature, converts the glob to a regex (``*`` ->
``[^/]*``, ``?`` -> ``[^/]``, anchor with ``.*`` on both ends), and retries
once.  These tests exercise the recovery end-to-end through the real local
backend so the rg error path is genuine, not mocked.
"""

import os
import re
import shutil

import pytest

from tools.file_operations import (
    ShellFileOperations,
)
from tools.environments.local import LocalEnvironment


def _ops(root):
    return ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))


@pytest.fixture
def glob_tree(tmp_path):
    """Tree with a couple of files whose names + contents rely on the glob."""
    f1 = tmp_path / "test_detector_scheduler.py"
    f1.write_text("def foo():\n    return 'detector scheduler'\n")
    f2 = tmp_path / "signal_scheduler.py"
    f2.write_text("SIGNAL_SCHEDULER = 1\n")
    return tmp_path


# ---------------------------------------------------------------------------
# _rg_error_looks_like_bad_glob / _glob_to_regex (unit-level)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("err,pat,expected", [
    # Canonical signature: leading ``*`` triggers rg's "repetition operator
    # missing expression" because ``*`` is a quantifier that needs an operand.
    ("rg: regex parse error:\n    *foo*\n       ^\nerror: repetition operator missing expression",
     "*foo*", True),
    # Some models wrap the glob in a non-capturing group; same root cause.
    ("rg: regex parse error:\n    (?:*foo*bar*)\n       ^\nerror: repetition operator missing expression",
     "(?:*foo*bar*)", True),
    # Doubly-wrapped form has appeared in the wild.
    ("rg: regex parse error:\n    (?:(?:*foo*))\n          ^\nerror: repetition operator missing expression",
     "(?:(?:*foo*))", True),
    # ``(*`` (quantifier over group start) surfaces the same way.
    ("rg: regex parse error:\n    (*foo)\n     ^\nerror: repetition operator missing expression",
     "(*foo)", True),
    # NOT a glob — a legitimately broken regex shouldn't trip the recovery.
    ("rg: regex parse error:\n    (?P<>)\n       ^\nerror: unrecognized flag",
     "(?P<>)", False),
    # No regex parse error at all.
    ("rg: bad flag", "*foo*", False),
    # Empty / None inputs.
    ("", "*foo*", False),
    (None, "*foo*", False),
    ("rg: regex parse error: *", "", False),
])
def test_rg_error_looks_like_bad_glob(err, pat, expected):
    assert ShellFileOperations._rg_error_looks_like_bad_glob(err, pat) is expected


@pytest.mark.parametrize("glob,expected", [
    # Bare ``*`` anchors become ``[^/]*`` and the whole thing is wrapped
    # with .* on both ends so the substring can match anywhere in the path
    # component (mirrors ``find -name '*foo*'``).
    ("*foo*", ".*[^/]*foo[^/]*.*"),
    # Multi-segment globs preserve the ``*`` as a non-slash run.
    ("*detector*scheduler*", ".*[^/]*detector[^/]*scheduler[^/]*.*"),
    # ``?`` becomes ``[^/]`` (single non-slash char, not bare ``.``).
    ("foo?bar", ".*foo[^/]bar.*"),
    # Wrapping ``(?:...)`` is stripped before translation (idempotent).
    ("(?:*foo*)", ".*[^/]*foo[^/]*.*"),
    # Doubly-wrapped too.
    ("(?:(?:*foo*))", ".*[^/]*foo[^/]*.*"),
    # Literal chars are regex-escaped.
    ("foo.bar", ".*foo\\.bar.*"),
    # Pre-escaped wildcards preserved (``\\*`` stays a literal ``*``);
    # re.escape() emits a literal backslash+star, which matches the input.
    (r"foo\*bar", ".*" + re.escape(r"foo\*bar") + ".*"),
    (r"foo\?bar", ".*" + re.escape(r"foo\?bar") + ".*"),
])
def test_glob_to_regex(glob, expected):
    assert ShellFileOperations._glob_to_regex(glob) == expected


# ---------------------------------------------------------------------------
# End-to-end recovery through ``_search_with_rg``
# ---------------------------------------------------------------------------

def test_search_content_recovers_from_glob_as_regex(glob_tree):
    """The canonical #66129 repro: glob pattern, content target -> 0 error + results."""
    ops = _ops(glob_tree)
    result = ops.search(
        pattern="*detector*scheduler*",
        path=str(glob_tree),
        target="content",
        file_glob=None, limit=50, offset=0,
        output_mode="content", context=0,
    )
    assert result.error is None, f"expected recovery, got error: {result.error!r}"
    assert result.total_count >= 1, "expected the test file to match"
    assert any("detector scheduler" in m.content for m in result.matches)
    # Warning tells the caller the pattern was auto-corrected.
    assert result.warning and "auto-converted" in result.warning


def test_search_content_recovers_from_doubly_wrapped_glob(glob_tree):
    """The wrapped form ``(?:(?:*foo*))`` is also recovered."""
    ops = _ops(glob_tree)
    result = ops.search(
        pattern="(?:(?:*detector*scheduler*))",
        path=str(glob_tree),
        target="content",
        file_glob=None, limit=50, offset=0,
        output_mode="content", context=0,
    )
    assert result.error is None, f"expected recovery, got error: {result.error!r}"
    assert result.total_count >= 1


def test_search_content_does_not_recover_legitimately_broken_regex(glob_tree):
    """A genuinely broken regex (not a glob) returns the original rg error."""
    ops = _ops(glob_tree)
    result = ops.search(
        pattern=r"(?P<> unnamed)",
        path=str(glob_tree),
        target="content",
        file_glob=None, limit=50, offset=0,
        output_mode="content", context=0,
    )
    assert result.error is not None, "broken regex must surface an error"
    assert "Search failed" in result.error
    # No auto-convert warning when recovery didn't fire.
    assert not getattr(result, "warning", None)


def test_search_content_legit_star_regex_unchanged(glob_tree):
    """A valid regex with ``*`` (e.g. ``a*b``) must keep matching normally.

    This is the false-positive guard: ``a*b`` parses fine in rg, so the
    recovery path should never fire and the matches payload should reflect
    the real result of the regex.
    """
    ops = _ops(glob_tree)
    result = ops.search(
        pattern="det.*tor",  # valid regex, matches "detector"
        path=str(glob_tree),
        target="content",
        file_glob=None, limit=50, offset=0,
        output_mode="content", context=0,
    )
    assert result.error is None, f"valid regex should not error: {result.error!r}"
    assert result.total_count >= 1
    # No auto-convert warning because no recovery was needed.
    assert not getattr(result, "warning", None)


def test_search_content_recovers_via_search_tool_entrypoint(glob_tree, monkeypatch):
    """End-to-end via ``search_tool`` (the public tool entry point)."""
    import json
    from tools import file_tools

    # Force the public tool to use a fresh FileOperations for our tree.
    ops = _ops(glob_tree)
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: ops)

    out = file_tools.search_tool(
        pattern="*detector*scheduler*",
        path=str(glob_tree),
        target="content",
        file_glob=None, limit=50, offset=0,
        output_mode="content", context=0,
        task_id="test_66129_e2e",
    )
    parsed = json.loads(out)
    assert "error" not in parsed or not parsed["error"], parsed
    assert parsed["total_count"] >= 1
