"""Hosted-room profile discovery excludes deletion metadata."""

from pathlib import Path

from tests.tui_gateway.test_hosted_room_service import _server
from tui_gateway.hosted_room_service import HostedRoomService


def test_local_profiles_skips_delete_tombstones_and_dot_dirs(tmp_path: Path):
    """`hermes profile delete` leaves ``profiles/.deleted/<name>``; neither the tombstone dir, a
    tombstoned profile, nor a marker-less cron shell is a roster member (#106847: ``.deleted``
    failed validate_roster every cycle; #99392: side-effect dirs listed as bots)."""
    from hermes_constants import mark_named_profile_deleted

    profiles = tmp_path / "profiles"
    (profiles / "ops").mkdir(parents=True)
    (profiles / "ops" / "config.yaml").write_text("{}\n", encoding="utf-8")
    (profiles / "gone").mkdir()
    (profiles / "gone" / "config.yaml").write_text("{}\n", encoding="utf-8")
    mark_named_profile_deleted(profiles / "gone")
    assert (profiles / ".deleted").is_dir()
    (profiles / "shell" / "cron").mkdir(parents=True)

    service = HostedRoomService(_server(), db_path=tmp_path / "shared-state.db")

    assert service.local_profiles() == ("default", "ops")
