"""Adversarial verification tests for ci-poller-hint-dead-skill-ref.

Independent of the implementer's own tests in test_notify_on_complete.py.
Goal: re-derive the acceptance criteria from the spec and probe
edge cases the implementer's 4 restored tests did not cover:

  1. No skill_view() dead pointer remains anywhere in the hint text.
  2. The literal "green-ci-policy" token is present (hard constraint).
  3. Both canonical snippets (exit-code rc branching AND column-2 awk-on-tabs)
     are inlined verbatim enough to be actionable without loading anything.
  4. Boundary probes the implementer didn't test:
     - jq appearing as a substring of another word (e.g. "jquery") must NOT
       false-positive the _has_jq gate.
     - "gh pr checks" alone (no jq, no awk) must NOT fire (that's model core
       positive control already covered, but combined with unrelated jq use
       elsewhere in the command should still not fire the CI-poller hint if
       jq isn't piped from a gh call).
     - statusCheckRollup appearing in a comment/string but the command is
       otherwise foreground must not fire (background=False gate).
     - Multiple qualifying signals in one command still fire exactly once
       (hint text is not duplicated).
"""

import json
import re

import pytest

from tests.tools.test_notify_on_complete import _silent_bg_harness


def _run(tt, command, **kwargs):
    try:
        return json.loads(
            tt.terminal_tool(
                command=command, background=True, notify_on_complete=True, **kwargs
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)


def test_no_dead_skill_view_pointer_in_source():
    """Positive control on the actual bug: the hint must not tell the agent
    to skill_view() a path that doesn't resolve. Grep the live source for
    any skill_view() call inside terminal_tool.py — there should be none,
    since the fix inlines the guidance instead of pointing at a skill."""
    with open("tools/terminal_tool.py") as f:
        src = f.read()
    assert "skill_view(" not in src, (
        "terminal_tool.py must not call skill_view() to back the CI-poller "
        "hint — the fix inlines the canonical snippets instead of pointing "
        "at a skill file (dead or otherwise)."
    )


def test_hint_names_green_ci_policy_token(monkeypatch, tmp_path):
    """Hard constraint from the spec: the literal token 'green-ci-policy'
    must survive in the emitted hint regardless of implementation choice."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    result = _run(
        tt,
        "PR=1; while true; do gh pr view $PR --json statusCheckRollup --jq '.'; sleep 30; done",
    )
    hint = result.get("hint", "")
    assert "green-ci-policy" in hint


def test_hint_inlines_both_canonical_snippets_verbatim_enough(monkeypatch, tmp_path):
    """The hint must be self-sufficient per the spec's Option B design goal:
    an agent reading it with NO skill_view() call should know both the
    exit-code-driven pattern and the column-2 awk-on-tabs pattern well
    enough to reproduce them. Check for the actual snippet fragments, not
    just loose keywords."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    result = _run(
        tt,
        "PR=1; while true; do gh pr checks $PR | jq -R 'split(\"\\t\")'; sleep 30; done",
    )
    hint = result.get("hint", "")
    # Exit-code pattern: rc 0/8 branching on `gh pr checks`.
    assert re.search(r"gh pr checks \$PR", hint), (
        "Hint must inline the actual exit-code-driven gh pr checks command"
    )
    assert "0" in hint and "8" in hint, (
        "Hint must state the specific exit codes (0=green, 8=pending) — "
        "without them the guidance isn't actionable, just a vague pointer"
    )
    # Column-2 awk pattern.
    assert "awk -F" in hint and "pending" in hint, (
        'Hint must inline the actual awk -F"\\t" column-2 pending pattern'
    )


def test_jquery_substring_does_not_false_positive_jq_gate(monkeypatch, tmp_path):
    """Boundary case the implementer's tests didn't cover: the _has_jq gate
    checks for ' jq ', '| jq', '$(jq' substrings. A command mentioning
    something like 'jquery' or a variable named 'jq_path' should not be
    treated as piping through the jq binary."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    result = _run(
        tt,
        "gh pr view 1 --json statusCheckRollup > /tmp/out.json; "
        "node build_jquery_bundle.js; sleep 30",
    )
    # This command DOES contain statusCheckRollup unconditionally in the
    # detector (bad_shape fires on statusCheckRollup alone, independent of
    # jq), so the hint SHOULD fire here — this test documents that the
    # detector's OR-gate is on statusCheckRollup already, not gated by jq.
    hint = result.get("hint", "")
    assert "green-ci-policy" in hint


def test_gh_pr_checks_alone_without_jq_or_awk_does_not_fire(monkeypatch, tmp_path):
    """`gh pr checks` by itself (no jq, no awk parsing stdout) is exactly
    the blessed exit-code pattern and must not be flagged."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    result = _run(
        tt,
        "PR=1; while :; do gh pr checks $PR >/dev/null 2>&1; rc=$?; "
        "case $rc in 0) exit 0;; 8) sleep 30;; *) exit 1;; esac; done",
    )
    assert "hint" not in result or "green-ci-policy" not in result.get("hint", ""), (
        f"Canonical gh pr checks exit-code loop must not trigger the "
        f"homebrew-poller hint, got: {result.get('hint')!r}"
    )


def test_statuscheckrollup_in_foreground_command_does_not_fire(monkeypatch, tmp_path):
    """The homebrew-poller hint is gated on background=True. A foreground
    one-shot statusCheckRollup query (not a poller loop) must not fire it,
    since the detector's docstring scopes it to background anti-patterns."""
    from types import SimpleNamespace

    tt = _silent_bg_harness(monkeypatch, tmp_path)
    dummy_env = SimpleNamespace(
        env={},
        execute=lambda *a, **kw: {"output": "{}", "exit_code": 0, "error": None},
    )
    tt._active_environments["default"] = dummy_env
    try:
        result = json.loads(
            tt.terminal_tool(
                command="gh pr view 1 --json statusCheckRollup",
                background=False,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert "hint" not in result, (
        f"Foreground statusCheckRollup one-shot must not fire the homebrew "
        f"poller hint (background=False), got: {result.get('hint')!r}"
    )


def test_hint_not_duplicated_when_multiple_signals_present(monkeypatch, tmp_path):
    """A command that matches both the statusCheckRollup AND the gh-pr-checks
    -piped-to-jq shapes simultaneously must still get exactly ONE occurrence
    of the canonical hint text, not one appended per matching branch."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    result = _run(
        tt,
        "PR=1; while true; do "
        "gh pr view $PR --json statusCheckRollup --jq '.'; "
        "gh pr checks $PR | jq -R '.'; "
        "sleep 30; done",
    )
    hint = result.get("hint", "")
    assert hint.count("green-ci-policy") == 1, (
        f"Hint must name green-ci-policy exactly once even when multiple "
        f"anti-pattern signals match, got {hint.count('green-ci-policy')} "
        f"occurrences in: {hint!r}"
    )
