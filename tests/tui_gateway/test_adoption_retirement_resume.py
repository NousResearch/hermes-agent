"""Private contract probe for real profile-resume retirement retries."""

import pytest

from hermes_state import SessionDB
from tests.tui_gateway import test_stranded_session_adoption as adoption_cases
from tests.hermes_state.test_adoption_transfer_records import _retire_fails_for, _seed_compression_lineage


gateway = adoption_cases.gateway
_seed_stranded = adoption_cases._seed_stranded


def _resume(mod, sid, request_id):
    return mod.handle_request({"id": request_id, "method": "session.resume",
                               "params": {"session_id": sid, "profile": "developer", "lazy": True}})


@pytest.mark.parametrize("fail_root", [True, False], ids=["first-retirement", "partial-retirement"])
def test_profile_resume_finishes_the_saved_transfer(gateway, monkeypatch, fail_root):
    mod, donor, profile_home = gateway
    root, tip = "resume-transfer-root", "resume-transfer-tip"
    _seed_compression_lineage(donor, root, tip)
    with monkeypatch.context() as patch:
        patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(root if fail_root else tip))
        _resume(mod, tip, "first")
    assert donor.get_session(tip)["end_reason"] is None
    response = _resume(mod, tip if fail_root else root, "retry")
    assert not response.get("error"), response.get("error")
    assert donor.get_session(root)["end_reason"] == "adopted_by_profile"
    assert donor.get_session(tip)["end_reason"] == "adopted_by_profile"
    target = SessionDB(profile_home / "state.db")
    try:
        target.append_message(tip, "user", "after profile resume")
        assert len(target.get_messages(tip)) == 3
    finally:
        target.close()


def test_profile_resume_does_not_retire_a_copy_only_donor(gateway):
    mod, donor, profile_home = gateway
    sid = "copy-only-resume"
    _seed_stranded(donor, sid)
    target = SessionDB(profile_home / "state.db")
    try:
        assert target.adopt_session_lineage_from(donor, sid, retire_donor=False)["adopted"]
    finally:
        target.close()
    _resume(mod, sid, "view-copy")
    assert donor.get_session(sid)["end_reason"] is None
    assert not donor.get_session(sid)["archived"]

"""Additional controls for a stale resume hint and completed transfers."""


def test_stale_resume_hint_cannot_override_copy_only_intent(gateway, monkeypatch):
    import hermes_state_adoption_records as records
    mod, donor, profile_home = gateway
    sid = "stale-copy-only-resume"
    _seed_stranded(donor, sid)
    target = SessionDB(profile_home / "state.db")
    try:
        assert target.adopt_session_lineage_from(donor, sid, retire_donor=False)["adopted"]
    finally:
        target.close()
    monkeypatch.setattr(records, "retirement_pending_source", lambda db, member: str(donor.db_path.resolve()))
    _resume(mod, sid, "stale-hint")
    assert donor.get_session(sid)["end_reason"] is None
    assert not donor.get_session(sid)["archived"]


def test_completed_resume_does_not_repeat_adoption(gateway, monkeypatch):
    mod, donor, profile_home = gateway
    sid = "completed-transfer-resume"
    _seed_stranded(donor, sid)
    target = SessionDB(profile_home / "state.db")
    try:
        assert target.adopt_session_lineage_from(donor, sid)["donor_retired"]
    finally:
        target.close()

    def no_adoption(*args, **kwargs):
        raise AssertionError("completed transfers must not revisit the donor")

    monkeypatch.setattr(SessionDB, "adopt_session_lineage_from", no_adoption)
    response = _resume(mod, sid, "completed")
    assert not response.get("error"), response.get("error")
