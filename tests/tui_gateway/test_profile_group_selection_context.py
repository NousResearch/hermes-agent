"""The group context a profile model switch warns with, at its edges: a ``bot_room`` session on another model
without a usage anchor gives none, and a profile without a store gives none and gets no store."""

from __future__ import annotations

import persistence_assembly
import tui_gateway.server as srv
from tests.fixtures.storage_seed import seed_messages, seed_session

OLD_MODEL = "old-local-model"
NEW_MODEL = "new-local-model"


def test_group_member_without_a_usage_anchor_gives_no_context(tmp_path):
    profile_dir = tmp_path / "bot"
    profile_dir.mkdir()
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=profile_dir)) as store:
        seed_session(store, "unanchored-room", "bot_room", model=OLD_MODEL)
        seed_messages(store, "unanchored-room", {"role": "user", "content": "Existing history"})

    assert srv._profile_group_selection_context(profile_dir, NEW_MODEL) is None


def test_profile_without_a_store_has_no_group_context(tmp_path):
    profile_dir = tmp_path / "fresh"
    profile_dir.mkdir()

    assert srv._profile_group_selection_context(profile_dir, NEW_MODEL) is None
    assert not (profile_dir / "state.db").exists()
