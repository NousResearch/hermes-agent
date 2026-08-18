"""macOS ScreenCaptureKit display_count=0 diagnosability.

Composed from PR #52949 (sujeet111, doctor guard) and PR #67259
(webtecnica, actionable-hint direction): a headless Mac or asleep panel
leaves SCK with zero shareable displays while TCC grants pass, so
health_report says ok and every capture returns 0x0 silently. The guard
turns that into a failed check + degraded overall + recovery hint.
"""

from tools.computer_use.doctor import _apply_display_count_guard


def _report(status="pass", count=0, overall="ok", name="screen_capture_capability"):
    return {
        "overall": overall,
        "checks": [
            {
                "name": name,
                "status": status,
                "message": "ScreenCaptureKit reachable",
                "data": {"display_count": count},
            }
        ],
    }


def test_zero_displays_downgrades_ok_to_degraded():
    out = _apply_display_count_guard(_report(count=0))
    assert out["overall"] == "degraded"
    assert out["checks"][0]["status"] == "fail"
    assert "0 shareable" in out["checks"][0]["message"]
    assert "dongle" in out["checks"][0]["hint"]


def test_positive_display_count_untouched():
    out = _apply_display_count_guard(_report(count=1))
    assert out["overall"] == "ok"
    assert out["checks"][0]["status"] == "pass"


def test_already_failed_check_not_rewritten():
    out = _apply_display_count_guard(_report(status="fail", count=0, overall="degraded"))
    # Guard only flips pass->fail; an existing failure keeps its message.
    assert out["checks"][0]["message"] == "ScreenCaptureKit reachable"


def test_missing_data_and_other_checks_ignored():
    out = _apply_display_count_guard(
        {"overall": "ok", "checks": [{"name": "binary_version", "status": "pass"}]}
    )
    assert out["overall"] == "ok"
    out2 = _apply_display_count_guard({"overall": "ok", "checks": "bogus"})
    assert out2["overall"] == "ok"


def test_overall_failed_not_upgraded():
    out = _apply_display_count_guard(_report(count=0, overall="failed"))
    # fail-worse states are never softened to degraded.
    assert out["overall"] == "failed"
    assert out["checks"][0]["status"] == "fail"
