"""Regression test: cronjob list action must tolerate a non-dict `repeat` value.

A raw string reaching `_repeat_display()` caused an `AttributeError` because
`.get()` was chained on the decoded payload. The list branch should coerce
that into a safe display string instead of crashing.
"""

import json

import pytest

from tools import cronjob_tools


def test_list_jobs_payload_tolerates_string_repeat(monkeypatch):
    bad_job = {
        "id": "job-bad-repeat",
        "prompt": "ping healthcheck",
        "repeat": "repeat-3-times",  # malformed payload: string where dict expected
        "enabled": True,
    }

    monkeypatch.setattr(cronjob_tools, "list_jobs", lambda include_disabled=False: [bad_job])

    result = cronjob_tools.cronjob(action="list")
    payload = json.loads(result)

    assert payload["success"] is True
    assert payload["count"] == 1
    assert payload["jobs"][0]["repeat"] in {"forever", "repeat-3-times"}
