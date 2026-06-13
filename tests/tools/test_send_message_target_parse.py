"""Parser-only tests for send_message targets.

These stay separate from ``test_send_message_tool.py`` because that module
skips wholesale when optional Telegram dependencies are not installed.
"""

from tools.send_message_tool import _parse_target_ref


def test_photon_e164_target_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref("photon", "+15551234567")

    assert chat_id == "+15551234567"
    assert thread_id is None
    assert is_explicit is True


def test_e164_target_still_requires_phone_platform() -> None:
    assert _parse_target_ref("matrix", "+15551234567")[2] is False


def test_whatsapp_group_jid_is_explicit() -> None:
    """WhatsApp group JIDs (e.g. 120363001234567890@g.us) must be recognized as explicit targets."""
    chat_id, thread_id, is_explicit = _parse_target_ref("whatsapp", "120363001234567890@g.us")

    assert chat_id == "120363001234567890@g.us"
    assert thread_id is None
    assert is_explicit is True


def test_whatsapp_individual_jid_is_explicit() -> None:
    """WhatsApp individual JIDs (e.g. 15551234567@s.whatsapp.net) must be explicit."""
    chat_id, thread_id, is_explicit = _parse_target_ref("whatsapp", "15551234567@s.whatsapp.net")

    assert chat_id == "15551234567@s.whatsapp.net"
    assert thread_id is None
    assert is_explicit is True


def test_whatsapp_lid_jid_is_explicit() -> None:
    """WhatsApp LID JIDs (e.g. 1234567890@lid) must be explicit."""
    chat_id, thread_id, is_explicit = _parse_target_ref("whatsapp", "1234567890@lid")

    assert chat_id == "1234567890@lid"
    assert thread_id is None
    assert is_explicit is True


def test_whatsapp_jid_not_matched_for_other_platforms() -> None:
    """JIDs should not be treated as explicit for non-WhatsApp platforms."""
    assert _parse_target_ref("signal", "120363001234567890@g.us")[2] is False
    assert _parse_target_ref("telegram", "15551234567@s.whatsapp.net")[2] is False


def test_whatsapp_bare_phone_still_uses_e164() -> None:
    """Bare phone numbers for WhatsApp should still match E.164 (with + prefix)."""
    chat_id, _, is_explicit = _parse_target_ref("whatsapp", "+15551234567")

    assert chat_id == "+15551234567"
    assert is_explicit is True

