"""Exact shared-CDP target close regression tests."""

from tools import browser_cdp_tool, browser_tool


def _install_fake_cdp(monkeypatch, responses):
    calls = []
    queued = iter(responses)

    def fake_cdp_call(_url, method, params, target_id, timeout):
        calls.append((method, params, target_id, timeout))
        return method

    def fake_run_async(method):
        response = next(queued)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", fake_cdp_call)
    monkeypatch.setattr(browser_cdp_tool, "_run_async", fake_run_async)
    return calls


def test_exact_close_succeeds_only_after_target_disappears(monkeypatch):
    calls = _install_fake_cdp(
        monkeypatch,
        [
            {"success": True},
            {"targetInfos": [{"targetId": "OTHER"}]},
        ],
    )

    assert browser_tool._close_shared_cdp_target_confirmed(
        "ws://shared", "TARGET-OWNED"
    ) is True
    assert [call[0] for call in calls] == ["Target.closeTarget", "Target.getTargets"]
    assert calls[0][1] == {"targetId": "TARGET-OWNED"}
    assert all(call[2] is None for call in calls)


def test_exact_close_rejects_surface_success_while_target_remains(monkeypatch):
    calls = _install_fake_cdp(
        monkeypatch,
        [
            {"success": True},
            {"targetInfos": [{"targetId": "TARGET-OWNED"}]},
        ],
    )
    ticks = iter([10.0, 13.0])
    monkeypatch.setattr(browser_tool.time, "monotonic", lambda: next(ticks))

    assert browser_tool._close_shared_cdp_target_confirmed(
        "ws://shared", "TARGET-OWNED"
    ) is False
    assert [call[0] for call in calls] == ["Target.closeTarget", "Target.getTargets"]


def test_exact_close_accepts_lost_response_only_after_absence_proof(monkeypatch):
    calls = _install_fake_cdp(
        monkeypatch,
        [
            TimeoutError("close response lost"),
            {"targetInfos": []},
        ],
    )

    assert browser_tool._close_shared_cdp_target_confirmed(
        "ws://shared", "TARGET-OWNED"
    ) is True
    assert [call[0] for call in calls] == ["Target.closeTarget", "Target.getTargets"]


def test_exact_close_fails_when_target_list_cannot_be_verified(monkeypatch):
    calls = _install_fake_cdp(
        monkeypatch,
        [
            {"success": True},
            TimeoutError("target list unavailable"),
        ],
    )
    ticks = iter([20.0, 23.0])
    monkeypatch.setattr(browser_tool.time, "monotonic", lambda: next(ticks))

    assert browser_tool._close_shared_cdp_target_confirmed(
        "ws://shared", "TARGET-OWNED"
    ) is False
    assert [call[0] for call in calls] == ["Target.closeTarget", "Target.getTargets"]
