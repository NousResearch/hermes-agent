"""The failure limit counts failed tasks, not more recent successful delegations."""
import time

from tools import async_delegation as ad


def test_recent_successes_do_not_hide_an_older_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    for index, status in enumerate(["error", "completed", "completed"]):
        ad._persist_dispatch({
            "delegation_id": f"history-{index}", "session_key": "", "origin_ui_session_id": "history-ui",
            "parent_session_id": "history-parent", "dispatched_at": now - 60, "goal": "review",
        })
        ad._persist_completion({
            "delegation_id": f"history-{index}", "status": status, "completed_at": now - 3 + index,
        }, {"status": status, "error": "review failed" if status == "error" else None})

    failures = ad.failed_delegations_for_session("history-ui", limit=1, now=now)

    assert [(row["delegation_id"], row["error"]) for row in failures] == [("history-0", "review failed")]
