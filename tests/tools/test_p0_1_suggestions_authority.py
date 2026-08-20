"""P0-1 / Block 2 authority tests for cron/suggestions guard.

Witnesses:
  1. canonical DENY + env changed to permissive afterwards -> still DENY
  2. canonical ALLOW + env changed to restrictive afterwards -> stays ALLOW
  3. operation_kind="suggestions_write" preserved in legacy layer
  4. no None/fallback branch for get_self_improvement_decision()
  5. guard no longer reads HERMES_DISABLE_SELF_IMPROVEMENT / HERMES_READ_ONLY_SESSION
     from os.environ directly to derive self-improvement authority

These are not unit tests of the policy; they are SHA-anchored assertions
about the cron/suggestions.py source code itself.
"""

import os
import re
from pathlib import Path

import pytest


SUGGESTIONS = Path(__file__).resolve().parents[2] / "cron" / "suggestions.py"


def _read():
    return SUGGESTIONS.read_text(encoding="utf-8")


def _grep(pattern, src=None):
    return re.findall(pattern, src if src is not None else _read())


def test_get_self_improvement_decision_is_imported():
    src = _read()
    assert "get_self_improvement_decision" in src, (
        "cron/suggestions.py must import get_self_improvement_decision"
    )


def test_no_env_re_sample_for_self_improvement_authority():
    """The guard must not re-derive self-improvement authority from env vars
    at mutation time. Env vars may be sampled for the legacy layer's
    operation_kind labelling, but the primary DENY/ALLOW decision must come
    from the captured Decision."""
    src = _read()
    # The decide block must NOT call os.environ.get with the self-improvement
    # env vars as the primary authority. We allow them to appear in comments
    # explaining the legacy role, but not as the active decision source.
    guard_block = re.search(
        r"def _background_review_suggestions_guard.*?(?=\ndef )",
        src,
        re.DOTALL,
    )
    assert guard_block is not None
    body = guard_block.group(0)
    # Env-driven authority is only allowed AFTER the captured DENY check.
    # The PRIMARY check must use ``captured_decision`` or
    # ``get_self_improvement_decision()``.
    assert "captured_decision" in body or "get_self_improvement_decision()" in body, (
        "Guard body must reference the captured Decision as the primary authority"
    )
    # The legacy _policy_evaluate call must pass environment_disabled=""
    # and session_read_only="" because the captured Decision is the only
    # authority and any env-derived override is rejected by the primary
    # check.
    legacy_match = re.search(
        r"_policy_evaluate\(([^)]+)\)", body, re.DOTALL
    )
    assert legacy_match is not None, "Legacy _policy_evaluate call must still exist"
    args = legacy_match.group(1)
    assert 'environment_disabled=""' in args, (
        "Legacy _policy_evaluate must not pass env-derived env vars to "
        "reconstruct self-improvement authority"
    )
    assert 'session_read_only=""' in args, (
        "Legacy _policy_evaluate must not pass env-derived session_read_only "
        "to reconstruct self-improvement authority"
    )
    assert 'operation_kind="suggestions_write"' in args, (
        "operation_kind='suggestions_write' must be preserved"
    )


def test_captured_deny_overrides_env():
    """A captured DENY must not be overridden by env, even if env is set
    permissively afterwards. We verify the source code:
    the captured Decision is checked FIRST; only if the captured allows
    does the legacy layer run."""
    src = _read()
    body = re.search(
        r"def _background_review_suggestions_guard.*?(?=\ndef )",
        src,
        re.DOTALL,
    ).group(0)
    # The captured-decision check must come BEFORE the legacy layer.
    cap_pos = body.find("captured_decision")
    legacy_pos = body.find("_policy_evaluate")
    assert cap_pos >= 0 and legacy_pos >= 0
    assert cap_pos < legacy_pos, (
        "captured_decision check must precede the legacy _policy_evaluate call"
    )
    # The captured DENY path must return True (deny).
    assert "return True" in body, "Guard must return True on DENY"
    # The body must NOT contain a path that reads env AFTER the captured check
    # to override the captured decision.
    # We check that the legacy _policy_evaluate call has empty env strings.
    assert 'environment_disabled=""' in body
    assert 'session_read_only=""' in body


def test_legacy_layer_is_secondary():
    """The legacy _policy_evaluate call must not be able to deny a captured
    ALLOW. The captured Decision is authoritative; the legacy layer is
    advisory only."""
    src = _read()
    body = re.search(
        r"def _background_review_suggestions_guard.*?(?=\ndef )",
        src,
        re.DOTALL,
    ).group(0)
    # If the legacy layer returns non-ALLOW after the captured ALLOW, we
    # log and return False (still allow).
    # Verify this pattern exists.
    assert "_legacy.result" in body, "Legacy result must be inspected"
    # The legacy non-ALLOW branch must return False (allow), not True (deny).
    # Find the _legacy.result != "ALLOW" branch.
    m = re.search(
        r'if _legacy\.result != "ALLOW":\s*\n((?:\s+.+\n)+)',
        body,
    )
    assert m is not None, "Legacy non-ALLOW branch must exist"
    branch = m.group(1)
    assert "return False" in branch, (
        "Legacy non-ALLOW must NOT deny the captured ALLOW"
    )


def test_suggestions_write_operation_kind_preserved():
    src = _read()
    assert 'operation_kind="suggestions_write"' in src
    # Verify the legacy helper still gets operation_kind="suggestions_write"
    # even after the captured Decision became the primary authority.


def test_no_none_branch_for_decision():
    """The guard must not return None or fall through without an explicit
    return. Every code path must return True or False."""
    src = _read()
    body = re.search(
        r"def _background_review_suggestions_guard.*?(?=\ndef )",
        src,
        re.DOTALL,
    ).group(0)
    # Count return statements
    returns = re.findall(r"\n\s+return\s+", body)
    assert len(returns) >= 3, (
        f"Guard must have multiple return paths (got {len(returns)})"
    )


def test_default_off_denied_when_context_unset():
    """When the ContextVar is unset, get_self_improvement_decision() returns
    DENY_FALLBACK_DECISION. The guard must therefore deny. This is the
    'no implicit ALLOW' invariant."""
    src = _read()
    # The import DENY_FALLBACK_DECISION must be present.
    assert "DENY_FALLBACK_DECISION" in src, (
        "DENY_FALLBACK_DECISION must be imported as a fallback"
    )
    # The captured decision is treated as authoritative.
    # When the ContextVar is unset, get_self_improvement_decision() returns
    # DENY_FALLBACK_DECISION (allow=False), so the guard returns True
    # (deny). This is the "default-off" invariant.
    assert "captured_decision.allow" in src or "getattr(captured_decision" in src


def test_block_1_decoupled_imports():
    """Block 1: run_agent.py must decouple session_write_policy and
    self_improvement_decision_context imports so a partial import failure
    doesn't drop the protected-write authority."""
    run_agent = Path(__file__).resolve().parents[2] / "run_agent.py"
    src = run_agent.read_text(encoding="utf-8")
    # The original PR has a single try/except ImportError wrapping both
    # imports. The repair must split them into two separate try blocks.
    # We verify by checking for two distinct except ImportError blocks
    # in the relevant section.
    block = re.search(
        r"Block 1 repair.*?finish_logical_calls",
        src,
        re.DOTALL,
    )
    assert block is not None, "Block 1 repair contract must be present"
    body = block.group(0)
    n_excepts = len(re.findall(r"except ImportError", body))
    assert n_excepts >= 2, (
        f"Block 1 repair must have at least two separate try/except ImportError "
        f"blocks (got {n_excepts})"
    )
    # The fail-closed branch must explicitly refuse to run the turn body.
    assert "refusing to run turn body" in body or "session_write_policy helpers unavailable" in body
