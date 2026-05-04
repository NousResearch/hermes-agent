
import os

import pytest

from agent.transcript_capture.config import TranscriptCaptureConfig
from agent.transcript_capture.writer import TranscriptWriter, stable_short_hash


def _cfg(tmp_path):
    return TranscriptCaptureConfig(active_dir=tmp_path/"active", corpus_dir=tmp_path/"corpus", state_dir=tmp_path/"state")


def test_stable_short_hash_is_deterministic_and_short():
    assert stable_short_hash("raw-chat-id") == stable_short_hash("raw-chat-id")
    assert stable_short_hash("raw-chat-id") != stable_short_hash("other")
    assert len(stable_short_hash("raw-chat-id")) == 16


def test_final_filename_is_flat_and_hash_based(tmp_path):
    writer = TranscriptWriter(_cfg(tmp_path))
    path = writer.final_path("2026-05-03", "discord", "raw-session-key", "raw-session-id")
    assert path.parent == tmp_path / "corpus"
    assert path.name.startswith("2026-05-03-discord-")
    assert path.name.endswith(".txt")
    assert "raw-session" not in path.name


def test_publish_uses_active_part_then_flat_corpus(tmp_path):
    writer = TranscriptWriter(_cfg(tmp_path))
    final = writer.publish("2026-05-03", "discord", "key", "sid", "body\nEND_SESSION\n")
    assert final.exists()
    assert final.parent == tmp_path / "corpus"
    assert final.read_text() == "body\nEND_SESSION\n"
    assert not list((tmp_path / "corpus").glob("*.part"))
    assert not list((tmp_path / "active").glob("*.part"))


def test_publish_rejects_missing_end_session(tmp_path):
    with pytest.raises(ValueError):
        TranscriptWriter(_cfg(tmp_path)).publish("2026-05-03", "discord", "key", "sid", "body")


def test_publish_is_idempotent_for_same_content(tmp_path):
    writer = TranscriptWriter(_cfg(tmp_path))
    first = writer.publish("2026-05-03", "discord", "key", "sid", "body\nEND_SESSION\n")
    second = writer.publish("2026-05-03", "discord", "key", "sid", "body\nEND_SESSION\n")
    assert first == second
    assert len(list((tmp_path / "corpus").glob("*.txt"))) == 1


def test_publish_forces_redaction_at_disk_write_boundary(tmp_path):
    writer = TranscriptWriter(_cfg(tmp_path))
    raw_secret = "Bearer " + "A" * 32
    final = writer.publish("2026-05-03", "discord", "key-redact", "sid-redact", f"token={raw_secret}\nEND_SESSION\n")
    body = final.read_text()
    assert raw_secret not in body
    assert "Bearer ***" in body
    assert body.endswith("END_SESSION\n")


def test_publish_redacts_configured_raw_chat_identifiers_from_body(tmp_path):
    raw_chat_id = "1498618695046660267"
    cfg = TranscriptCaptureConfig(
        active_dir=tmp_path / "active",
        corpus_dir=tmp_path / "corpus",
        state_dir=tmp_path / "state",
        chat_allowlist=frozenset({raw_chat_id}),
    )
    writer = TranscriptWriter(cfg)

    final = writer.publish(
        "2026-05-04",
        "discord",
        "session-key",
        "session-id",
        f"quoted rollout note mentioned channel {raw_chat_id}\nEND_SESSION\n",
    )

    body = final.read_text()
    assert raw_chat_id not in body
    assert "[REDACTED CHAT ID]" in body
    assert body.endswith("END_SESSION\n")
