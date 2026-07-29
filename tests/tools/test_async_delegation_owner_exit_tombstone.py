"""Owner-exit tombstones must distinguish real loss from post-delivery bookkeeping.

The owner-liveness reaper used to emit a byte-identical terminal record for two
OPPOSITE situations:

* the owner died with children still unaccounted for — real lost work needing
  re-dispatch; and
* the owner died *after* the consolidated result was already delivered to the
  parent conversation — pure bookkeeping.

An operator could not tell them apart without hand-verifying deliverables on
disk. These are behavior contracts on the relationship between delivery state
and what the reaper reports, not snapshots of the message wording.
"""

import json
import queue

import pytest

import tools.async_delegation as ad


DEAD_PID = 99999999


@pytest.fixture(autouse=True)
def _sandboxed_home(tmp_path, monkeypatch):
    """Point the durable ledger at a temp HERMES_HOME (never production)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield


def _kill_owner(delegation_id):
    """Make the recorded owner pid unresolvable, simulating owner exit."""
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET owner_pid=?, owner_started_at=NULL "
            "WHERE delegation_id=?",
            (DEAD_PID, delegation_id),
        )


def _row(delegation_id):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT state, delivery_state FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


def _event(delegation_id):
    with ad._DB_LOCK, ad._transaction() as conn:
        row = conn.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def _dispatch(delegation_id, goals):
    ad._persist_dispatch({
        "delegation_id": delegation_id,
        "session_key": "owner",
        "origin_ui_session_id": "",
        "parent_session_id": None,
        "dispatched_at": 1.0,
        "goals": list(goals),
        "is_batch": True,
    })


def _set_results(delegation_id, results):
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET result_json=? WHERE delegation_id=?",
            (json.dumps({"results": results}), delegation_id),
        )


def _restored_ids(delegation_id=None):
    target = queue.Queue()
    ad.restore_undelivered_completions(target)
    ids = []
    while not target.empty():
        ids.append(target.get().get("delegation_id"))
    return ids


# ── Direction (a): owner died mid-flight → LOUD, names the lost children ──

def test_midflight_owner_exit_names_unaccounted_children_for_redispatch():
    _dispatch("deleg_lost", ["build parser", "write tests", "ship docs"])
    # The owner got one child home before dying; two never reached a terminal state.
    _set_results("deleg_lost", [
        {"status": "completed", "summary": "parser ok"},
        {"status": "running"},
    ])
    _kill_owner("deleg_lost")

    assert ad.recover_abandoned_delegations() == 1
    event = _event("deleg_lost")

    # The outcome is genuinely in doubt, so it must NOT be reported as delivered.
    assert event["status"] == "unknown"
    assert event["owner_exit_delivered"] is False

    # Per-child terminal state travels with the tombstone — the store already
    # had this and the old message discarded it.
    assert [child["status"] for child in event["child_states"]] == [
        "completed", "running", "unknown",
    ]
    # Only the children that never reached a terminal success are unaccounted.
    assert [child["index"] for child in event["unaccounted_children"]] == [1, 2]

    # The operator is told which ids to re-dispatch, and is NOT told to stand down.
    assert "re-dispatch" in event["error"].lower()
    assert "no action needed" not in event["error"].lower()

    # A genuinely-lost batch still re-enters the conversation.
    assert "deleg_lost" in _restored_ids()


def test_single_delegation_owner_exit_also_reports_child_state():
    """Sibling call path: a non-batch record stores one result dict, not a list."""
    ad._persist_dispatch({
        "delegation_id": "deleg_single",
        "session_key": "owner",
        "origin_ui_session_id": "",
        "parent_session_id": None,
        "dispatched_at": 1.0,
        "goal": "one job",
    })
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET result_json=? WHERE delegation_id=?",
            (json.dumps({"status": "running", "goal": "one job"}), "deleg_single"),
        )
    _kill_owner("deleg_single")

    assert ad.recover_abandoned_delegations() == 1
    event = _event("deleg_single")
    assert event["status"] == "unknown"
    assert len(event["child_states"]) == 1
    assert event["unaccounted_children"][0]["index"] == 0


# ── Direction (b): owner died after delivery → QUIET, no action needed ──

def test_delivery_makes_the_row_terminal_so_the_reaper_cannot_phantom_it():
    """The structural fix: delivery is the terminal moment.

    A delivered row left at ``state='running'`` is precisely what let the reaper
    manufacture a phantom "outcome unknown" for work that was fully delivered.
    """
    _dispatch("deleg_ok", ["a", "b"])
    ad._persist_completion(
        {"delegation_id": "deleg_ok", "status": "completed", "completed_at": 2.0},
        {"results": [{"status": "completed"}, {"status": "completed"}]},
    )
    # Reproduce the pathology: the durable branch never advanced ``state``.
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET state='running' WHERE delegation_id=?",
            ("deleg_ok",),
        )

    assert ad.claim_completion_delivery("deleg_ok", "claim-1") is True
    assert ad.complete_completion_delivery("deleg_ok", "claim-1") is True

    # Delivery advanced the row out of the reaper's scan window.
    assert _row("deleg_ok")[0] not in ("running", "finalizing")

    _kill_owner("deleg_ok")
    assert ad.recover_abandoned_delegations() == 0
    assert "deleg_ok" not in _restored_ids()


def test_mark_completion_delivered_is_also_terminal():
    """Sibling delivery path must carry the same contract as the claim path."""
    _dispatch("deleg_marked", ["a"])
    ad._persist_completion(
        {"delegation_id": "deleg_marked", "status": "completed", "completed_at": 2.0},
        {"results": [{"status": "completed"}]},
    )
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET state='running' WHERE delegation_id=?",
            ("deleg_marked",),
        )

    assert ad.mark_completion_delivered("deleg_marked") is True
    assert _row("deleg_marked")[0] not in ("running", "finalizing")

    _kill_owner("deleg_marked")
    assert ad.recover_abandoned_delegations() == 0


def test_delivered_row_reaching_the_reaper_is_settled_quietly():
    """A legacy row already delivered but still 'running' must not re-alarm."""
    _dispatch("deleg_legacy", ["x", "y"])
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET state='running', delivery_state='delivered', "
            "result_json=? WHERE delegation_id=?",
            (json.dumps({"results": [{"status": "completed"}, {"status": "completed"}]}),
             "deleg_legacy"),
        )
    _kill_owner("deleg_legacy")

    assert ad.recover_abandoned_delegations() == 1
    # Settled to a terminal state...
    assert _row("deleg_legacy")[0] not in ("running", "finalizing")
    # ...without re-entering the conversation...
    assert "deleg_legacy" not in _restored_ids()
    # ...and never reaped a second time.
    assert ad.recover_abandoned_delegations() == 0


def test_suppression_knob_false_emits_a_distinguishable_quiet_record(tmp_path):
    """With suppression off the record IS emitted — and reads nothing like a loss."""
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  suppress_delivered_owner_exit_tombstones: false\n"
    )
    assert ad._suppress_delivered_tombstones() is False

    _dispatch("deleg_visible", ["p", "q"])
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET state='running', delivery_state='delivered', "
            "result_json=? WHERE delegation_id=?",
            (json.dumps({"results": [{"status": "completed"}, {"status": "completed"}]}),
             "deleg_visible"),
        )
    _kill_owner("deleg_visible")
    assert ad.recover_abandoned_delegations() == 1

    event = _event("deleg_visible")
    assert event["owner_exit_delivered"] is True
    assert event["unaccounted_children"] == []
    assert len(event["child_states"]) == 2
    assert "no action needed" in event["error"].lower()
    assert "re-dispatch" not in event["error"].lower()


def test_delivered_and_lost_tombstones_are_never_the_same_message(tmp_path):
    """The core contract: the two opposite outcomes must not read identically."""
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  suppress_delivered_owner_exit_tombstones: false\n"
    )

    _dispatch("deleg_a", ["one", "two"])
    _set_results("deleg_a", [{"status": "completed"}, {"status": "running"}])
    _kill_owner("deleg_a")

    _dispatch("deleg_b", ["one", "two"])
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET state='running', delivery_state='delivered', "
            "result_json=? WHERE delegation_id=?",
            (json.dumps({"results": [{"status": "completed"}, {"status": "completed"}]}),
             "deleg_b"),
        )
    _kill_owner("deleg_b")

    ad.recover_abandoned_delegations()
    lost = _event("deleg_a")
    delivered = _event("deleg_b")

    assert lost["error"] != delivered["error"]
    assert lost["owner_exit_delivered"] is False
    assert delivered["owner_exit_delivered"] is True
    # And each carries the per-child detail, in BOTH directions.
    assert lost["child_states"] and delivered["child_states"]


def test_reaper_never_downgrades_a_delivered_row_at_either_knob_setting(tmp_path):
    """A delivered row must never be re-enqueued for delivery — knob-independent.

    Delivery is tracked on ``delivery_state``, orthogonal to ``state``. Any
    recovery path keying only on ``state`` can flip an already-delivered row
    back to ``pending``, and ``restore_undelivered_completions()`` will then
    re-deliver a result the conversation already has. Suppression must not be
    what protects this: it has to hold with the knob OFF too, which is exactly
    the branch that writes a fresh tombstone.
    """
    for suppress in ("true", "false"):
        (tmp_path / "config.yaml").write_text(
            f"delegation:\n  suppress_delivered_owner_exit_tombstones: {suppress}\n"
        )
        for index in range(3):
            delegation_id = f"deleg_{suppress}_{index}"
            _dispatch(delegation_id, ["g"])
            with ad._DB_LOCK, ad._transaction() as conn:
                conn.execute(
                    "UPDATE async_delegations SET state=?, delivery_state='delivered', "
                    "result_json=? WHERE delegation_id=?",
                    (
                        "running" if index % 2 == 0 else "finalizing",
                        json.dumps({"results": [{"status": "completed", "goal": "g"}]}),
                        delegation_id,
                    ),
                )
            _kill_owner(delegation_id)

        for _ in range(3):
            ad.recover_abandoned_delegations()

        with ad._DB_LOCK, ad._transaction() as conn:
            downgraded = conn.execute(
                "SELECT COUNT(*) FROM async_delegations "
                "WHERE delivery_state='pending' AND delegation_id LIKE ?",
                (f"deleg_{suppress}_%",),
            ).fetchone()[0]
        assert downgraded == 0, f"delivered rows downgraded with suppress={suppress}"

        # And nothing already-delivered is ever handed back for redelivery.
        assert not [
            did for did in _restored_ids() if did.startswith(f"deleg_{suppress}_")
        ]


def test_suppression_knob_defaults_to_config_default():
    """The knob is a real documented config surface, not a bare literal."""
    from hermes_cli.config import DEFAULT_CONFIG

    assert (
        DEFAULT_CONFIG["delegation"]["suppress_delivered_owner_exit_tombstones"]
        is ad._DEFAULT_SUPPRESS_DELIVERED_TOMBSTONES
    )
    # With no config.yaml written, the helper resolves to that documented default.
    assert ad._suppress_delivered_tombstones() is (
        DEFAULT_CONFIG["delegation"]["suppress_delivered_owner_exit_tombstones"]
    )
