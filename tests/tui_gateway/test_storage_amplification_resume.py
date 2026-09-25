"""Cold resume must not treat a physical-row safety refusal as a long lineage."""

import pytest

from hermes_state import SessionResumeTooLargeError, SessionStorageAmplificationError
from tui_gateway.server import _load_resume_transcript


class _BlockedStore:
    def __init__(self, *, block_at):
        self.block_at = block_at
        self.calls = []

    def assert_storage_safe(self, session_id):
        self.calls.append("storage")
        if self.block_at == "storage":
            raise SessionStorageAmplificationError(session_id, 5, 0, 4)
        return 0

    def assert_resume_safe(self, session_id):
        self.calls.append("resume")
        if self.block_at == "resume":
            raise SessionStorageAmplificationError(session_id, 5, 0, 4)

    def get_messages_as_conversation(self, *args, **kwargs):
        pytest.fail("physical-row refusal must not load model history")

    def get_resume_conversations(self, *args, **kwargs):
        pytest.fail("physical-row refusal must not load the lineage")


@pytest.mark.parametrize("model_history_only", [False, True])
def test_physical_archive_guard_blocks_both_resume_projections(model_history_only):
    store = _BlockedStore(block_at="storage")
    with pytest.raises(SessionStorageAmplificationError):
        _load_resume_transcript(store, "synthetic-session", model_history_only=model_history_only)
    assert store.calls == ["storage"]


def test_amplification_refusal_from_lineage_guard_does_not_fallback_to_tip():
    store = _BlockedStore(block_at="resume")
    with pytest.raises(SessionStorageAmplificationError):
        _load_resume_transcript(store, "synthetic-session")
    assert store.calls == ["storage", "resume"]


def test_active_lineage_limit_still_falls_back_to_tip():
    class LongLineage:
        def assert_storage_safe(self, session_id):
            return 0

        def assert_resume_safe(self, session_id):
            raise SessionResumeTooLargeError(5, 4)

        def get_messages_as_conversation(self, *args, **kwargs):
            return [{"role": "user", "content": "synthetic tip"}]

    history, display, prefix = _load_resume_transcript(LongLineage(), "synthetic-session")
    assert history == display == [{"role": "user", "content": "synthetic tip"}]
    assert prefix == []
