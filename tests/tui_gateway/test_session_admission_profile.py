"""Admission must use the owning profile's effective policy and writer registry (profile-scoped session admission)."""

from pathlib import Path

import pytest

from hermes_constants import get_hermes_home_override, reset_hermes_home_override, set_hermes_home_override
from hermes_cli.active_sessions import active_session_registry_snapshot, try_acquire_active_session
from tui_gateway import server


@pytest.mark.parametrize("named_launch", [False, True])
@pytest.mark.parametrize("owner", ["launch", "secondary"])
@pytest.mark.parametrize("foreign_context", [False, True])
def test_owning_profile_controls_admission(tmp_path, monkeypatch, named_launch, owner, foreign_context):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ("profiles/work" if named_launch else ".hermes")
    secondary = tmp_path / "profiles/secondary"
    foreign = tmp_path / "profiles/other"
    for home, cap in [(launch, 2), (secondary, 6), (foreign, 1)]:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(f"max_concurrent_sessions: {cap}\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    owning_home = secondary if owner == "secondary" else launch
    leases = []
    token = set_hermes_home_override(str(foreign) if foreign_context else None)
    ambient = get_hermes_home_override()
    try:
        # Two Desktop windows already own sessions in the selected profile.
        for i in range(2):
            lease, refusal = try_acquire_active_session(
                session_id=f"held-{i}", surface="desktop", config={"max_concurrent_sessions": 6},
                registry_home=owning_home, metadata={"live_session_id": f"held-runtime-{i}"},
                track_liveness=True)
            assert refusal is None
            leases.append(lease)
        lease, refusal = server._claim_active_session_slot(
            "new-chat", live_session_id="new-runtime", surface="desktop",
            profile_home=secondary if owner == "secondary" else None)
        if owner == "secondary":
            assert refusal is None, refusal
            leases.append(lease)
            assert len(active_session_registry_snapshot(registry_home=secondary)) == 3
            assert active_session_registry_snapshot(registry_home=launch) == []
        else:
            assert lease is None
            assert "limit (2/2)" in refusal
        assert get_hermes_home_override() == ambient
        assert active_session_registry_snapshot(registry_home=foreign) == []
    finally:
        for lease in reversed(leases):
            if lease is not None:
                lease.release()
        reset_hermes_home_override(token)


def test_profile_admission_keeps_writer_exclusivity_and_reentry(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 6\n")
    leases = []
    try:
        for _ in range(2):
            lease, refusal = server._claim_active_session_slot(
                "same-chat", live_session_id="same-writer", surface="desktop", profile_home=tmp_path)
            assert refusal is None
            leases.append(lease)
        assert len(active_session_registry_snapshot(registry_home=tmp_path)) == 1
        lease, refusal = server._claim_active_session_slot(
            "same-chat", live_session_id="other-writer", surface="desktop", profile_home=tmp_path)
        assert lease is None
        assert refusal.reason == "SESSION_NOT_OWNED"
    finally:
        for lease in reversed(leases):
            lease.release()
