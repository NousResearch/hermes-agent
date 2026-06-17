from agent import runtime_status


def test_approval_hooks_record_and_clear_approval_wait():
    from tools.approval import _fire_approval_hook

    runtime_status.clear_session("s-approval")

    _fire_approval_hook("pre_approval_request", session_key="s-approval")
    snap = runtime_status.snapshot("s-approval")
    assert snap["phase"] == "waiting"
    assert snap["wait"]["reason"] == "approval"

    _fire_approval_hook("post_approval_response", session_key="s-approval", choice="once")
    snap = runtime_status.snapshot("s-approval")
    assert snap["phase"] == "running"
    assert snap["wait"] == {"reason": "none", "since": None}


def test_clarify_gateway_register_and_resolution_record_wait_state():
    from tools import clarify_gateway

    runtime_status.clear_session("s-clarify")

    clarify_gateway.register("cid-1", "s-clarify", "Choose", ["A", "B"])
    snap = runtime_status.snapshot("s-clarify")
    assert snap["phase"] == "waiting"
    assert snap["wait"]["reason"] == "clarify"

    assert clarify_gateway.resolve_gateway_clarify("cid-1", "A") is True
    snap = runtime_status.snapshot("s-clarify")
    assert snap["phase"] == "running"
    assert snap["wait"] == {"reason": "none", "since": None}
