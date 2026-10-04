"""Regress #127107 against the repository-pinned cua-driver 0.21.0 contract."""
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tests.tools.test_computer_use_delivery_ladder import _FakeSession, _make_backend
from tools.computer_use.schema import COMPUTER_USE_SCHEMA
from tools.computer_use.tool import _dispatch

CONTRACT = json.loads((Path(__file__).parents[1] / "fixtures" / "cua_driver_0_21_verify_state.json").read_text())
ELEMENT = {"element": {"selector": {"role": "CheckBox", "label_contains": "高级查看"}, "exists": True, "selected": True}}
WINDOW = {"window": {"exists": True, "bounds": {"x": 10, "y": 20, "width": 300, "height": 400, "tolerance_px": 1}}}


@pytest.mark.parametrize("status,stable,is_error", [
    ("satisfied", True, False), ("unsatisfied", False, False),
    ("unknown", False, False), ("satisfied", False, False),
    ("satisfied", True, True), (None, True, False), ("unexpected", True, False),
])
@pytest.mark.parametrize("surface", ["data", "structuredContent"])
def test_driver_status_controls_verdict_and_preserves_evidence(status, stable, is_error, surface):
    evidence = {"status": status, "stable": stable, "elapsed_ms": 544, "samples": 2,
                "predicates": [{"index": 0, "status": status, "unknown_reason": None,
                                "observed_json": '{"element_index":2,"selected":true}'}]}
    if status in {"satisfied", "unsatisfied", "unknown"}:
        Draft202012Validator(CONTRACT["success_output_schema"]).validate(evidence)
    out = {"isError": is_error, "data": {}, "structuredContent": {}, surface: evidence}
    backend = _make_backend(_FakeSession(out))
    result = json.loads(_dispatch(backend, "verify_state", {"expect": [ELEMENT]}))
    success = status == "satisfied" and stable and not is_error
    assert (result["verdict"]["decision"] == "done") is success
    assert result["verified"] is success
    assert result["effect"] == ("confirmed" if success else "unverifiable")
    assert result["ok"] is (not is_error)  # transport success is not predicate satisfaction
    assert result["meta"] == evidence
    # Conflicting generic action flags cannot override the verification contract;
    # canonical structuredContent must also win over flattened data.
    out["data"] = {"status": "satisfied", "stable": True, "verified": True, "effect": "confirmed"}
    out["structuredContent"] = evidence
    conflicted = json.loads(_dispatch(backend, "verify_state", {"expect": [ELEMENT]}))
    assert (conflicted["verdict"]["decision"] == "done") is success


@pytest.mark.parametrize("expect,valid", [
    ([ELEMENT], True), ([WINDOW], True), ([ELEMENT, WINDOW], True), ([ELEMENT] * 8, True),
    ([{"element": {"selector": {"role": "TextField"}, "value_equals": "", "enabled": False}}], True),
    ([{"window": {"exists": False}}], True),
    (None, False), ([], False), ([ELEMENT] * 9, False),
    ([{"element": {"exists": True}}], False), ([{"element": {"selector": {}, "exists": True}}], False),
    ([{"element": {"selector": {"role": ""}, "exists": True}}], False),
    ([{"element": {"selector": {"role": "CheckBox"}, "exists": False}}], False),
    ([{"element": {"selector": {"role": "CheckBox"}, "selected": "true"}}], False),
    ([{"window": {"bounds": {"x": 0}}}], False),
    ([{"window": {"bounds": {"x": 0, "y": 0, "width": 1, "height": 1, "tolerance_px": 101}}}], False),
    ([{"window": {}}], False), ([{"anything": True}], False), ([{}], False),
])
def test_public_predicates_match_driver_and_validate_before_transport(expect, valid):
    public = Draft202012Validator(COMPUTER_USE_SCHEMA["parameters"]["properties"]["expect"])
    assert public.is_valid(expect) is valid
    if valid:
        Draft202012Validator(CONTRACT["input_schema"]).validate({"pid": 4242, "window_id": 7, "expect": expect})
    session = _FakeSession({"isError": False, "data": {}, "structuredContent": {"status": "satisfied", "stable": True}})
    backend = _make_backend(session)
    result = json.loads(_dispatch(backend, "verify_state", {"expect": expect}))
    if valid:
        assert result["verdict"]["decision"] == "done"
        assert session.last_args == {"pid": 4242, "window_id": 7, "expect": expect, "session": "test-run"}
    else:
        assert "error" in result
        assert backend.verify_state(expect).ok is False
        assert session.calls == []
