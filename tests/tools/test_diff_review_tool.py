"""Tests for tools/diff_review_tool.py - Automated code diff analysis."""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub optional heavy dependencies so tools/__init__.py can be imported in a
# minimal environment (CI / dev machines) that may not have them installed.
_STUB_MODULES = [
    "httpx",
    "anthropic",
    "openai",
    "firecrawl",
    "fal_client",
    "pydub",
    "edge_tts",
    "elevenlabs",
    "playwright",
    "playwright.async_api",
    "daytona_sdk",
    "modal",
    "docker",
    "paramiko",
    "pyperclip",
    "honcho",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from tools.diff_review_tool import (
    DIFF_REVIEW_SCHEMA,
    _WARN_PATTERNS,
    _get_git_diff,
    _parse_diff,
    check_diff_review_requirements,
    diff_review_tool,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SIMPLE_DIFF = """\
diff --git a/foo.py b/foo.py
index 0000000..1111111 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
+    print("hello")
-    pass
     return True
"""

_TWO_FILE_DIFF = """\
diff --git a/alpha.py b/alpha.py
index 0000000..aaaaaaa 100644
--- a/alpha.py
+++ b/alpha.py
@@ -1,2 +1,3 @@
 x = 1
+y = 2
-z = 3
diff --git a/beta.js b/beta.js
index 0000000..bbbbbbb 100644
--- a/beta.js
+++ b/beta.js
@@ -1,1 +1,2 @@
 const a = 1;
+console.log(a);
"""


# ---------------------------------------------------------------------------
# _parse_diff — structural parsing
# ---------------------------------------------------------------------------


class TestParseDiffStructure:
    """Tests for basic structural parsing of unified diffs."""

    def test_empty_string_returns_zero_counts(self):
        result = _parse_diff("")
        assert result["files_changed"] == 0
        assert result["total_added"] == 0
        assert result["total_removed"] == 0
        assert result["files"] == []
        assert result["warnings"] == []
        assert result["warning_count"] == 0

    def test_single_file_detected(self):
        result = _parse_diff(_SIMPLE_DIFF)
        assert result["files_changed"] == 1

    def test_single_file_path(self):
        result = _parse_diff(_SIMPLE_DIFF)
        assert result["files"][0]["path"] == "foo.py"

    def test_single_file_added_count(self):
        result = _parse_diff(_SIMPLE_DIFF)
        assert result["files"][0]["added"] == 1

    def test_single_file_removed_count(self):
        result = _parse_diff(_SIMPLE_DIFF)
        assert result["files"][0]["removed"] == 1

    def test_total_added(self):
        result = _parse_diff(_SIMPLE_DIFF)
        assert result["total_added"] == 1

    def test_total_removed(self):
        result = _parse_diff(_SIMPLE_DIFF)
        assert result["total_removed"] == 1

    def test_two_files_detected(self):
        result = _parse_diff(_TWO_FILE_DIFF)
        assert result["files_changed"] == 2

    def test_two_files_paths(self):
        result = _parse_diff(_TWO_FILE_DIFF)
        paths = [f["path"] for f in result["files"]]
        assert "alpha.py" in paths
        assert "beta.js" in paths

    def test_two_files_total_added(self):
        result = _parse_diff(_TWO_FILE_DIFF)
        assert result["total_added"] == 2  # +y=2 in alpha, +console.log in beta

    def test_two_files_total_removed(self):
        result = _parse_diff(_TWO_FILE_DIFF)
        assert result["total_removed"] == 1  # -z=3 in alpha

    def test_context_lines_not_counted_as_added(self):
        diff = """\
diff --git a/f.py b/f.py
--- a/f.py
+++ b/f.py
@@ -1,3 +1,3 @@
 context_line_one
+new_line
 context_line_two
"""
        result = _parse_diff(diff)
        assert result["total_added"] == 1
        assert result["total_removed"] == 0

    def test_plus_plus_plus_line_not_counted_as_addition(self):
        """The +++ b/... header line must not be counted as an added line."""
        result = _parse_diff(_SIMPLE_DIFF)
        # If +++ were counted, added would be 2 (header + actual added line)
        assert result["total_added"] == 1

    def test_minus_minus_minus_line_not_counted_as_removal(self):
        """The --- a/... header line must not be counted as a removed line."""
        result = _parse_diff(_SIMPLE_DIFF)
        assert result["total_removed"] == 1

    def test_multiple_hunks_same_file(self):
        diff = """\
diff --git a/multi.py b/multi.py
--- a/multi.py
+++ b/multi.py
@@ -1,2 +1,3 @@
 line_one
+added_one
-removed_one
@@ -10,2 +11,3 @@
 another_context
+added_two
"""
        result = _parse_diff(diff)
        assert result["total_added"] == 2
        assert result["total_removed"] == 1

    def test_file_with_no_additions_or_removals(self):
        diff = """\
diff --git a/empty_change.py b/empty_change.py
--- a/empty_change.py
+++ b/empty_change.py
@@ -1,1 +1,1 @@
 only context line
"""
        result = _parse_diff(diff)
        assert result["files_changed"] == 1
        assert result["total_added"] == 0
        assert result["total_removed"] == 0


# ---------------------------------------------------------------------------
# _parse_diff — line number tracking
# ---------------------------------------------------------------------------


class TestParseDiffLineNumbers:
    """Tests for line number tracking in warnings."""

    def test_warning_line_number_is_positive(self):
        diff = """\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,1 +1,2 @@
 existing = 1
+print("debug")
"""
        result = _parse_diff(diff)
        assert result["warnings"][0]["line"] > 0

    def test_warning_line_number_reflects_hunk_start(self):
        diff = """\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -50,1 +50,2 @@
 existing = 1
+print("debug")
"""
        result = _parse_diff(diff)
        # The @@ says new-file starts at line 50; parser sets line_number = 49.
        # Context line increments to 50, then the added line increments to 51.
        assert result["warnings"][0]["line"] == 51


# ---------------------------------------------------------------------------
# _parse_diff — warning detection
# ---------------------------------------------------------------------------


class TestParseDiffWarnings:
    """Tests that each _WARN_PATTERNS entry is detected on added lines."""

    def _diff_with_added(self, content: str, filename: str = "f.py") -> str:
        return (
            f"diff --git a/{filename} b/{filename}\n"
            f"--- a/{filename}\n"
            f"+++ b/{filename}\n"
            "@@ -1,1 +1,2 @@\n"
            " context\n"
            f"+{content}\n"
        )

    def test_debug_print_python(self):
        result = _parse_diff(self._diff_with_added('print("hello")'))
        issues = [w["issue"] for w in result["warnings"]]
        assert "debug print statement" in issues

    def test_console_log_javascript(self):
        result = _parse_diff(self._diff_with_added("console.log(x)", filename="f.js"))
        issues = [w["issue"] for w in result["warnings"]]
        assert "console.log statement" in issues

    def test_hardcoded_password(self):
        result = _parse_diff(self._diff_with_added("password = 'hunter2'"))
        issues = [w["issue"] for w in result["warnings"]]
        assert "possible hardcoded secret" in issues

    def test_hardcoded_secret(self):
        result = _parse_diff(self._diff_with_added('secret = "abc123"'))
        issues = [w["issue"] for w in result["warnings"]]
        assert "possible hardcoded secret" in issues

    def test_hardcoded_api_key(self):
        result = _parse_diff(self._diff_with_added('api_key = "mykey"'))
        issues = [w["issue"] for w in result["warnings"]]
        assert "possible hardcoded secret" in issues

    def test_hardcoded_token(self):
        result = _parse_diff(self._diff_with_added('token = "tok_abc"'))
        issues = [w["issue"] for w in result["warnings"]]
        assert "possible hardcoded secret" in issues

    def test_hardcoded_secret_case_insensitive(self):
        result = _parse_diff(self._diff_with_added('PASSWORD = "S3cr3t"'))
        issues = [w["issue"] for w in result["warnings"]]
        assert "possible hardcoded secret" in issues

    def test_bare_except(self):
        result = _parse_diff(self._diff_with_added("except:"))
        issues = [w["issue"] for w in result["warnings"]]
        assert "bare except clause" in issues

    def test_broad_exception_catch(self):
        result = _parse_diff(self._diff_with_added("except Exception:"))
        issues = [w["issue"] for w in result["warnings"]]
        assert "broad Exception catch" in issues

    def test_todo_comment(self):
        result = _parse_diff(self._diff_with_added("# TODO: fix this"))
        issues = [w["issue"] for w in result["warnings"]]
        assert "unresolved TODO/FIXME" in issues

    def test_fixme_comment(self):
        result = _parse_diff(self._diff_with_added("# FIXME: broken"))
        issues = [w["issue"] for w in result["warnings"]]
        assert "unresolved TODO/FIXME" in issues

    def test_hack_comment(self):
        result = _parse_diff(self._diff_with_added("# HACK: workaround"))
        issues = [w["issue"] for w in result["warnings"]]
        assert "unresolved TODO/FIXME" in issues

    def test_xxx_comment(self):
        result = _parse_diff(self._diff_with_added("# XXX: danger"))
        issues = [w["issue"] for w in result["warnings"]]
        assert "unresolved TODO/FIXME" in issues

    def test_no_warning_on_clean_line(self):
        result = _parse_diff(self._diff_with_added("x = 42"))
        assert result["warnings"] == []
        assert result["warning_count"] == 0

    def test_warning_not_triggered_on_removed_line(self):
        diff = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,2 +1,1 @@\n"
            ' context\n'
            '-print("removed debug")\n'
        )
        result = _parse_diff(diff)
        assert result["warnings"] == []

    def test_multiple_warnings_same_line(self):
        """A line matching multiple patterns should produce multiple warnings."""
        # bare except + a TODO comment: two separate patterns matched
        diff = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,1 +1,2 @@\n"
            " context\n"
            "+except:  # TODO remove\n"
        )
        result = _parse_diff(diff)
        issues = [w["issue"] for w in result["warnings"]]
        assert "bare except clause" in issues
        assert "unresolved TODO/FIXME" in issues

    def test_warning_count_matches_warnings_list_length(self):
        diff = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,1 +1,3 @@\n"
            " context\n"
            '+print("one")\n'
            '+print("two")\n'
        )
        result = _parse_diff(diff)
        assert result["warning_count"] == len(result["warnings"])

    def test_warning_file_field_matches_filename(self):
        result = _parse_diff(self._diff_with_added('print("x")', filename="mymod.py"))
        assert result["warnings"][0]["file"] == "mymod.py"


# ---------------------------------------------------------------------------
# _get_git_diff
# ---------------------------------------------------------------------------


class TestGetGitDiff:
    """Tests for _get_git_diff subprocess wrapper."""

    def test_returns_stdout_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "diff output"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            output = _get_git_diff()
        assert output == "diff output"
        mock_run.assert_called_once()

    def test_default_base_is_head(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _get_git_diff()
        cmd = mock_run.call_args[0][0]
        assert cmd[2] == "HEAD"

    def test_target_appended_when_given(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _get_git_diff(base="HEAD", target="origin/main")
        cmd = mock_run.call_args[0][0]
        assert "origin/main" in cmd

    def test_non_zero_returncode_returns_empty_string(self):
        mock_result = MagicMock(returncode=1, stdout="partial", stderr="error msg")
        with patch("subprocess.run", return_value=mock_result):
            result = _get_git_diff()
        assert result == ""

    def test_git_not_found_returns_empty_string(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _get_git_diff()
        assert result == ""

    def test_timeout_returns_empty_string(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=30)):
            result = _get_git_diff()
        assert result == ""

    def test_custom_base_passed_to_subprocess(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _get_git_diff(base="abc123")
        cmd = mock_run.call_args[0][0]
        assert "abc123" in cmd


# ---------------------------------------------------------------------------
# diff_review_tool — public API
# ---------------------------------------------------------------------------


class TestDiffReviewToolNoDiff:
    """Tests for the 'no diff' path."""

    def test_empty_string_diff_text(self):
        result = json.loads(diff_review_tool(diff_text=""))
        assert result["status"] == "no_diff"

    def test_whitespace_only_diff_text(self):
        result = json.loads(diff_review_tool(diff_text="   \n\t  "))
        assert result["status"] == "no_diff"

    def test_no_diff_message_field(self):
        result = json.loads(diff_review_tool(diff_text=""))
        assert "message" in result
        assert result["message"]

    def test_no_diff_zero_files_changed(self):
        result = json.loads(diff_review_tool(diff_text=""))
        assert result["files_changed"] == 0

    def test_no_diff_zero_added_removed(self):
        result = json.loads(diff_review_tool(diff_text=""))
        assert result["total_added"] == 0
        assert result["total_removed"] == 0

    def test_no_diff_empty_warnings(self):
        result = json.loads(diff_review_tool(diff_text=""))
        assert result["warnings"] == []
        assert result["warning_count"] == 0


class TestDiffReviewToolWithDiff:
    """Tests for the 'ok' path when a diff is provided."""

    def test_status_ok_when_diff_provided(self):
        result = json.loads(diff_review_tool(diff_text=_SIMPLE_DIFF))
        assert result["status"] == "ok"

    def test_returns_json_string(self):
        raw = diff_review_tool(diff_text=_SIMPLE_DIFF)
        assert isinstance(raw, str)
        json.loads(raw)  # must not raise

    def test_files_changed_populated(self):
        result = json.loads(diff_review_tool(diff_text=_TWO_FILE_DIFF))
        assert result["files_changed"] == 2

    def test_warnings_populated_for_debug_print(self):
        diff = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n+++ b/x.py\n"
            "@@ -1,1 +1,2 @@\n"
            " ctx\n"
            '+print("debug")\n'
        )
        result = json.loads(diff_review_tool(diff_text=diff))
        assert result["warning_count"] > 0

    def test_clean_diff_no_warnings(self):
        diff = (
            "diff --git a/clean.py b/clean.py\n"
            "--- a/clean.py\n+++ b/clean.py\n"
            "@@ -1,1 +1,2 @@\n"
            " ctx\n"
            "+x = 42\n"
        )
        result = json.loads(diff_review_tool(diff_text=diff))
        assert result["warning_count"] == 0
        assert result["warnings"] == []

    def test_calls_git_diff_when_no_diff_text(self):
        mock_result = MagicMock(returncode=0, stdout=_SIMPLE_DIFF, stderr="")
        with patch("subprocess.run", return_value=mock_result):
            result = json.loads(diff_review_tool())
        assert result["status"] == "ok"

    def test_diff_text_bypasses_git(self):
        """When diff_text is supplied, subprocess.run should not be called."""
        with patch("subprocess.run") as mock_run:
            diff_review_tool(diff_text=_SIMPLE_DIFF)
        mock_run.assert_not_called()

    def test_base_and_target_forwarded_to_git(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            diff_review_tool(base="abc", target="def")
        cmd = mock_run.call_args[0][0]
        assert "abc" in cmd
        assert "def" in cmd


# ---------------------------------------------------------------------------
# check_diff_review_requirements
# ---------------------------------------------------------------------------


class TestCheckDiffReviewRequirements:
    """Tests for the requirements check function."""

    def test_returns_true_when_git_found(self):
        mock_result = MagicMock(returncode=0, stdout="git version 2.x", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            assert check_diff_review_requirements() is True

    def test_returns_false_when_git_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert check_diff_review_requirements() is False

    def test_returns_false_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5)):
            assert check_diff_review_requirements() is False


# ---------------------------------------------------------------------------
# DIFF_REVIEW_SCHEMA
# ---------------------------------------------------------------------------


class TestDiffReviewSchema:
    """Tests for the OpenAI function-calling schema."""

    def test_schema_name(self):
        assert DIFF_REVIEW_SCHEMA["name"] == "diff_review"

    def test_schema_has_description(self):
        assert "description" in DIFF_REVIEW_SCHEMA
        assert len(DIFF_REVIEW_SCHEMA["description"]) > 20

    def test_schema_parameters_is_object(self):
        assert DIFF_REVIEW_SCHEMA["parameters"]["type"] == "object"

    def test_schema_has_base_property(self):
        assert "base" in DIFF_REVIEW_SCHEMA["parameters"]["properties"]

    def test_schema_has_target_property(self):
        assert "target" in DIFF_REVIEW_SCHEMA["parameters"]["properties"]

    def test_schema_has_diff_text_property(self):
        assert "diff_text" in DIFF_REVIEW_SCHEMA["parameters"]["properties"]

    def test_schema_no_required_params(self):
        """All parameters are optional — required list should be empty."""
        assert DIFF_REVIEW_SCHEMA["parameters"]["required"] == []

    def test_base_default_is_head(self):
        base_spec = DIFF_REVIEW_SCHEMA["parameters"]["properties"]["base"]
        assert base_spec.get("default") == "HEAD"


# ---------------------------------------------------------------------------
# _WARN_PATTERNS constant
# ---------------------------------------------------------------------------


class TestWarnPatterns:
    """Sanity checks on the _WARN_PATTERNS constant."""

    def test_warn_patterns_is_list(self):
        assert isinstance(_WARN_PATTERNS, list)

    def test_each_entry_is_two_tuple(self):
        for entry in _WARN_PATTERNS:
            assert len(entry) == 2, f"Expected 2-tuple, got: {entry}"

    def test_all_patterns_compile(self):
        import re
        for pattern, _label in _WARN_PATTERNS:
            re.compile(pattern)  # must not raise

    def test_all_labels_are_non_empty_strings(self):
        for _pattern, label in _WARN_PATTERNS:
            assert isinstance(label, str) and label.strip()
