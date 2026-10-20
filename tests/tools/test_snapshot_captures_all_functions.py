"""Bash snapshot must capture every shell function, not filter private helpers.

Regression coverage for the ``_safe_eval: command not found`` failure class: a
user shell tool (e.g. scm_breeze) aliases core commands (``ls``, ``cd``, ``cat``…)
to a wrapper function that delegates to a *private*, underscore-prefixed helper
(``_safe_eval``). The old snapshot bootstrap filtered function names with
``grep -vE '^_[^_]'``, dropping the helper while keeping its dependents, so
sourcing the snapshot left a dangling reference and every command touching those
aliases failed.

The fix captures ALL functions (``declare -F`` without a name filter); function
bodies are inert until called, so capturing them is safe.
"""

from tools.environments.base_session_env import _snapshot_bootstrap_script


def _capture_line(script: str) -> str:
    for line in script.splitlines():
        if "__hermes_fns=" in line:
            return line
    raise AssertionError("no __hermes_fns capture line in bootstrap script")


def test_bootstrap_captures_all_functions_without_name_filter():
    script = _snapshot_bootstrap_script(
        quoted_cwd="/tmp",
        quoted_snap="/tmp/snap.sh",
        snap_tmp_template="/tmp/snap.sh.tmp.XXXXXX",
        excluded_names=[],
        cwd_marker="__HERMES_TEST__",
    )
    line = _capture_line(script)
    # A captured alias/function may depend on a private underscore-prefixed helper
    # (scm_breeze's _safe_eval); filtering names by prefix drops the helper and leaves
    # the dependents dangling. The capture must be name-filter-free.
    assert "grep -vE" not in line
    assert "awk '{print $3}'" in line
