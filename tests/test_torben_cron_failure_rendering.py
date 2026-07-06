from __future__ import annotations

from cron import scheduler


def test_cron_failure_delivery_uses_profile_renderer(tmp_path, monkeypatch) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "torben_comms_render.py").write_text(
        "\n".join(
            [
                "def render_failure_message(*, job, raw_error, consecutive_failures, detail_path, severity='yellow', handle=None):",
                "    return '\\n'.join([",
                "        '🟡 WHAT: ' + job + ' failed: sanitized cause',",
                "        'WHY IT MATTERS: Torben may be blind or noisy until this job is repaired.',",
                "        'DID/STAGED: Consecutive failures: ' + str(consecutive_failures) + '; details saved at ' + str(detail_path) + '.',",
                "        'CAN SAY NEXT: Investigate the saved run output before retrying or changing schedule.',",
                "    ]) + '\\n'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scheduler, "get_hermes_home", lambda: str(tmp_path))
    monkeypatch.setattr(scheduler, "_TORBEN_COMMS_RENDERER", None)
    monkeypatch.setattr(scheduler, "_TORBEN_COMMS_RENDERER_LOADED", False)

    text = scheduler._summarize_cron_failure_for_delivery(
        {"id": "abc123", "name": "torben-desk-v2-bar-refresh", "last_status": "error"},
        "Traceback (most recent call last):\n{\"stdout_tail\": \"raw json\"}",
    )

    assert len(text.strip().splitlines()) == 4
    assert text.startswith("🟡 WHAT: torben-desk-v2-bar-refresh failed")
    assert "Traceback" not in text
    assert "{\"stdout_tail\"" not in text
    assert "cron/output/abc123" in text
