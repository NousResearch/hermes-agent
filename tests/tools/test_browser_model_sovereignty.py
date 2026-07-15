"""Model-sovereignty invariants for the browser observation boundary."""

from __future__ import annotations

import json


def test_registry_never_forwards_user_task_to_snapshot_auxiliary_model(
    monkeypatch,
) -> None:
    """Primary GPT, not a hidden summarizer, interprets browser evidence."""

    from tools import browser_tool

    observed: dict[str, object] = {}

    def snapshot(**kwargs):
        observed.update(kwargs)
        return json.dumps({"success": True, "snapshot": "mechanical evidence"})

    monkeypatch.setattr(browser_tool, "browser_snapshot", snapshot)

    result = browser_tool.registry.dispatch(
        "browser_snapshot",
        {"full": True},
        task_id="task-1",
        user_task="decide which page details matter",
    )

    assert json.loads(result)["success"] is True
    assert observed == {"full": True, "task_id": "task-1"}
