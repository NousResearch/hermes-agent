"""The two delivery refusals are different conditions and now have different codes (#93091 follow-up).

Both shipped ``target_busy`` before, and the waiter only ever saw human prose:

* TARGET_BUSY — another delivery TURN holds this profile's cross-process turn lock.
* TARGET_SESSION_LIVE — the target's chat has a LIVE OWNER mid-turn. Nothing is queued behind a
  turn; the message was refused at the session LEASE.

That second one is what the 2026-09-20 orphan produced: a delivery session that outlived its
requester kept owning the target profile's chat, so every later delivery to it was refused with
the *lease* text while senders keyed on ``target_busy``. A sender cannot choose the right
operator action from one code for two conditions.
"""

from __future__ import annotations

import pytest

from hermes_cli import active_sessions as act
from tools import bot_failure_reasons as fr


def test_the_refusal_vocabulary_is_closed_and_names_both():
    assert fr.TARGET_BUSY == "target_busy"
    assert fr.TARGET_SESSION_LIVE == "target_session_live"
    assert fr.DELIVERY_REFUSAL_REASONS == frozenset({"target_busy", "target_session_live"})


def test_every_delivery_reason_is_a_known_code():
    """A consumer validating a delivery reason accepts both families, and nothing else."""
    assert fr.DELIVERY_REASONS == fr.ALL_REASONS | fr.DELIVERY_REFUSAL_REASONS
    assert "target_session_live" in fr.DELIVERY_REASONS
    assert "definitely-not-a-code" not in fr.DELIVERY_REASONS


def test_the_refusal_codes_are_not_turn_failure_reasons():
    """``ALL_REASONS`` stays the agent/provider vocabulary: the retry policy and the hosted-room
    ``turn.failed`` event schema are total over it, and a target-state refusal is neither."""
    assert fr.TARGET_BUSY not in fr.ALL_REASONS
    assert fr.TARGET_SESSION_LIVE not in fr.ALL_REASONS
    # And neither is auto-retryable: a supervisor must not silently re-send into the same wall.
    assert not fr.is_auto_retryable(fr.TARGET_SESSION_LIVE)
    assert not fr.is_auto_retryable(fr.TARGET_BUSY)
    assert fr.retry_action(fr.TARGET_SESSION_LIVE) == fr.RETRY_NONE


def test_the_mirrored_refusal_literal_matches_the_real_constant():
    """``bot_failure_reasons`` mirrors ``SESSION_NOT_OWNED`` as a literal (import cycle), so the
    two are pinned HERE: a rename on one side must fail loudly instead of silently classifying
    every lease refusal as 'unknown'."""
    assert fr.SESSION_NOT_OWNED_REASON == act.SESSION_NOT_OWNED


def test_the_refusal_marker_is_the_one_the_cli_writes():
    """The child writes ``hermes-refusal-reason: <code>`` (``format_refusal_stderr``); the lanes
    read exactly that. A drift here would make the code unreachable across the process boundary."""
    message = act.ActiveSessionRefusal("Ce chat est occupé.", reason=act.SESSION_NOT_OWNED)
    stderr = act.format_refusal_stderr(message)
    assert stderr.splitlines()[0] == f"{fr._REFUSAL_MARKER}{act.SESSION_NOT_OWNED}"


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        # The lease refusal, as the CLI child reports it: the marker, then the human text.
        (f"{fr._REFUSAL_MARKER}{act.SESSION_NOT_OWNED}\nCe chat est occupé.", fr.TARGET_SESSION_LIVE),
        # A bare code with NO marker is not a form the CLI produces (format_refusal_stderr only
        # emits the marker when a reason is set) — so it must not be read as one.
        (act.SESSION_NOT_OWNED, fr.UNKNOWN),
        # A different refusal code is not the lease case: fall through to the text classifier.
        (f"{fr._REFUSAL_MARKER}SESSION_COORDINATION_UNAVAILABLE", fr.UNKNOWN),
        # Older CLIs ship no marker; the historical wording is the only witness.
        ("Session abc already has a live owner (desktop, pid 1).", fr.TARGET_SESSION_LIVE),
        # An ordinary turn failure still classifies from its text.
        ("Error code: 429 - Too Many Requests", fr.PROVIDER_RATE_LIMIT),
        ("something nobody has a rule for", fr.UNKNOWN),
        ("", fr.UNKNOWN),
    ],
)
def test_delivery_detail_classification(detail, expected):
    assert fr.classify_delivery_detail(detail) == expected


def test_the_marker_wins_over_prose_that_merely_mentions_the_lease():
    """A refusal code beats the wording: the marker is the child's own statement of the reason."""
    detail = (
        f"{fr._REFUSAL_MARKER}{act.SESSION_NOT_OWNED}\n"
        "Delivery failed: @ops's Bot Chat is open on another surface right now."
    )
    assert fr.classify_delivery_detail(detail) == fr.TARGET_SESSION_LIVE


def test_a_lease_refusal_from_an_old_cli_still_names_the_second_flavour():
    """Mixed-version fleet: an OLD holder's child writes no marker, so the prose is all there is."""
    detail = "Session 9f2 already has a live owner (cli, pid 1). Details: session 9f2 opened by cli 2h ago."
    assert fr.classify_delivery_detail(detail) == fr.TARGET_SESSION_LIVE


# ── the refusal travels through an exception without leaking free text ──────

def test_a_refusal_exception_forwards_its_own_code():
    class _Refusal(RuntimeError):
        reason = fr.TARGET_SESSION_LIVE

    assert fr.delivery_failure_reason(_Refusal("x")) == fr.TARGET_SESSION_LIVE


def test_a_stdlib_reason_attribute_is_still_not_forwarded():
    """``reason`` is also a stdlib attribute (ssl/urllib): free text must not reach a closed set."""
    class _Ssl(RuntimeError):
        reason = "CERTIFICATE_VERIFY_FAILED"

    assert fr.delivery_failure_reason(_Ssl("ssl handshake failed")) == fr.UNKNOWN
