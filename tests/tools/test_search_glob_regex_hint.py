"""search_files: glob-vs-regex didactic error and guardrail-prevention tests.

Covers the case where a glob pattern (e.g. ``*.py``) is passed to
``target='content'`` (regex mode). A leading ``*`` is invalid regex
("nothing to repeat") -> rg parse error -> tool failure -> repeated-failure
guardrail. The tool now pre-validates and returns a didactic error naming the
corrected call BEFORE invoking rg/grep.
"""

import json

from tools.file_operations_search import _maybe_enrich_regex_error
from tools.file_tools import (
    _glob_as_regex_error,
    _looks_like_glob_pattern,
    search_tool,
)


# --- heuristic ---------------------------------------------------------------

def test_looks_like_glob_flags_only_invalid_regex():
    # leading '*' and '*' after '(' or '|' -> guaranteed regex parse error
    assert _looks_like_glob_pattern("*config*") is True
    assert _looks_like_glob_pattern("*.py") is True
    assert _looks_like_glob_pattern("(*x)") is True
    assert _looks_like_glob_pattern("a|*b") is True
    assert _looks_like_glob_pattern("**config") is True


def test_looks_like_glob_does_not_flag_valid_regex():
    # valid regex must NEVER be misreported (would break real searches)
    assert _looks_like_glob_pattern("config") is False
    assert _looks_like_glob_pattern("config.*py") is False
    assert _looks_like_glob_pattern("foo.*") is False
    assert _looks_like_glob_pattern("a.*b") is False
    assert _looks_like_glob_pattern("[a-z]*") is False
    assert _looks_like_glob_pattern("config*") is False   # valid: zero+ 'g'
    assert _looks_like_glob_pattern("foo*bar") is False
    assert _looks_like_glob_pattern("") is False
    assert _looks_like_glob_pattern("no_star") is False


# --- didactic error string ---------------------------------------------------

def test_glob_as_regex_error_names_corrected_call():
    msg = _glob_as_regex_error("*config*", "content", ".")
    assert "looks like a glob" in msg
    assert "target='content'" in msg
    assert "target='files'" in msg
    # files-mode example keeps the original glob
    assert "search_files(pattern='*config*', target='files'" in msg
    # content example strips wrapping * only
    assert "e.g. 'config'" in msg


def test_glob_as_regex_error_keeps_star_dot_py_as_glob():
    msg = _glob_as_regex_error("*.py", "content", "/tmp/repo")
    assert "search_files(pattern='*.py', target='files', path='/tmp/repo')" in msg
    # must NOT strip to '.py' (that is not a useful glob)
    assert "pattern='.py'" not in msg
    # content example must not be the leftover extension
    assert "e.g. 'foo'" in msg
    assert "e.g. '.py'" not in msg


# --- search_tool pre-validation (no rg/grep invocation) ---------------------

def test_search_tool_glob_in_content_mode_returns_didactic_error_without_running():
    # A glob pattern in content (regex) mode returns the didactic error and
    # never reaches the path-resolution / rg execution path, so an empty/
    # bogus path does not raise and no shell search runs.
    raw = search_tool(pattern="*config*", target="content", path="/nonexistent", task_id="t-glob")
    data = json.loads(raw)
    assert "error" in data
    assert "looks like a glob" in data["error"]
    assert "target='files'" in data["error"]


def test_search_tool_valid_regex_in_content_mode_is_not_blocked_by_hint():
    # Valid regex (config.*py) is NOT flagged by the pre-validation; it proceeds
    # to a real search. With a bogus path it surfaces "Path not found" (not the
    # glob hint), proving the pre-validation left it alone.
    raw = search_tool(pattern="config.*py", target="content", path="/nonexistent/here", task_id="t-regex")
    data = json.loads(raw)
    # Either no error key (matches path resolved) or a non-glob error.
    if "error" in data:
        assert "looks like a glob" not in data["error"]


def test_search_tool_files_mode_with_glob_is_not_flagged():
    # target='files' is glob mode by design — a glob pattern here is correct and
    # must NOT trigger the content-mode pre-validation.
    raw = search_tool(pattern="*config*", target="files", path="/nonexistent/here", task_id="t-files")
    data = json.loads(raw)
    # Path-not-found is expected (bogus path); the glob didactic hint is not.
    if "error" in data:
        assert "looks like a glob" not in data["error"]
        assert "target='content' uses REGEX" not in data["error"]


def test_search_tool_repeated_glob_call_keeps_didactic_error():
    # Even on repeat, the pre-validation fires first and returns the didactic
    # error (it runs before the consecutive-search tracker's own block), so the
    # agent keeps seeing the actionable message rather than a generic block.
    for _ in range(3):
        raw = search_tool(pattern="*config*", target="content", path=".", task_id="t-repeat")
        data = json.loads(raw)
        assert "looks like a glob" in data["error"]


# --- regex parse-error enrichment (fallback if pre-validation is skipped) ---

def test_enrich_regex_error_for_repetition_only():
    # "nothing to repeat" (leading '*') is the glob pitfall -> enriched.
    out = _maybe_enrich_regex_error(
        "Search failed: rg: regex parse error: missing operand, nothing to repeat",
        "*config*",
    )
    assert "looks like a glob" in out
    assert "target='files'" in out


def test_enrich_regex_error_leaves_non_repetition_parse_errors_alone():
    # An unclosed "[" is a parse error but NOT a glob mistake -> untouched, so
    # we never mislabel bracket/escape errors as glob confusion.
    bracket = "Search failed: rg: regex parse error: unclosed character class"
    assert _maybe_enrich_regex_error(bracket, "[") == bracket
    # Non-regex errors (permissions, etc.) are never enriched.
    perm = "Search failed: rg: permission denied"
    assert _maybe_enrich_regex_error(perm, "foo") == perm
