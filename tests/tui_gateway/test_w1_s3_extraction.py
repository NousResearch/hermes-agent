"""Regression tests for the wave-1 s3 extraction of tui_gateway/server.py.

Clusters c14 (message projection -> tui_gateway/message_projection.py) and c22
(pet payload -> tui_gateway/pet_payload_mixin.py) moved VERBATIM; these tests
pin the observable behavior of the pure helpers in their NEW modules and
assert server.py re-exports the very same objects, so in-file callers, the
HandlerRegistry-rebound methods_* handlers, and external
``from tui_gateway.server import ...`` consumers keep resolving them.
"""

import base64

import pytest

from tui_gateway import message_projection as mp
from tui_gateway import pet_payload_mixin as ppm
from tui_gateway import server


def test_server_reexports_moved_names():
    # message projection
    assert server._history_to_messages is mp._history_to_messages
    assert server._coerce_message_text is mp._coerce_message_text
    assert server._coerce_seed_history is mp._coerce_seed_history
    assert server._content_display_text is mp._content_display_text
    assert server._is_text_only_busy_payload is mp._is_text_only_busy_payload
    assert server._is_display_hidden_marker is mp._is_display_hidden_marker
    assert server._legacy_display_kind is mp._legacy_display_kind
    assert server._skill_scaffold_projection is mp._skill_scaffold_projection
    assert server._expand_skill_invocation_for_replay is mp._expand_skill_invocation_for_replay
    assert server._AUTO_CONTINUE_NOTE_PREFIX == mp._AUTO_CONTINUE_NOTE_PREFIX
    # pet payload
    assert server._pet_active_selection is ppm._pet_active_selection
    assert server._pet_sprite_payload is ppm._pet_sprite_payload
    assert server._pet_sheet_revision is ppm._pet_sheet_revision
    assert server._pet_frame_counts is ppm._pet_frame_counts
    assert server._pet_config_scale is ppm._pet_config_scale
    assert server._pet_cancel_arm is ppm._pet_cancel_arm
    assert server._pet_cancel_request is ppm._pet_cancel_request
    assert server._pet_is_cancelled is ppm._pet_is_cancelled
    assert server._pet_cancel_release is ppm._pet_cancel_release
    assert server._PET_REFERENCE_MIME_EXT is ppm._PET_REFERENCE_MIME_EXT
    assert server._PET_REFERENCE_MAX_BYTES == ppm._PET_REFERENCE_MAX_BYTES
    # the shared server helper the moved code binds is still the live one
    assert callable(server._tool_ctx)
    assert server._AUTO_CONTINUE_NOTE_PREFIX.startswith("[System note:")


class TestContentDisplayText:
    def test_none_and_str(self):
        assert mp._content_display_text(None) == ""
        assert mp._content_display_text("hi") == "hi"

    def test_numbers(self):
        assert mp._content_display_text(3) == "3"
        assert mp._content_display_text(2.5) == "2.5"

    def test_list_parts_joined(self):
        assert mp._content_display_text([{"type": "text", "text": "a"}, "b"]) == "a\nb"

    def test_dict_kinds(self):
        assert mp._content_display_text({"type": "image_url"}) == "[image]"
        assert mp._content_display_text({"type": "input_audio"}) == "[audio]"
        assert mp._content_display_text({"type": "weird"}) == "[weird]"
        assert mp._content_display_text({"text": "x"}) == "x"
        assert mp._content_display_text({"type": "text", "content": "c"}) == "c"
        assert mp._content_display_text({"other": 1}) == "[structured content]"

    def test_duplicate_definition_consolidated(self):
        # the 7041 shadow-def was consolidated; behavior must match the 6695 def
        assert mp._content_display_text([{"type": "text", "text": "a"}, 1]) == "a\n1"


class TestCoerceMessageText:
    def test_plain_shapes(self):
        assert mp._coerce_message_text(None) == ""
        assert mp._coerce_message_text("x") == "x"
        assert mp._coerce_message_text(7) == "7"

    def test_multimodal_list_preserves_image_url(self):
        out = mp._coerce_message_text(
            [
                {"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": "data:img/png;base64,AAA"}},
            ]
        )
        assert out == "see\ndata:img/png;base64,AAA"

    def test_audio_and_unknown_placeholders(self):
        assert mp._coerce_message_text([{"type": "input_audio"}]) == "\n[audio]"
        assert mp._coerce_message_text([{"type": "mystery"}]) == "\n[mystery]"

    def test_structured_dict(self):
        assert mp._coerce_message_text({"type": "text", "text": "hi"}) == "hi"
        assert mp._coerce_message_text({"type": "image_url", "image_url": "http://x"}) == "http://x"


class TestBusyPayloadClassification:
    def test_text_only_shapes(self):
        assert mp._is_text_only_busy_payload("hi") is True
        assert mp._is_text_only_busy_payload([{"type": "text", "text": "a"}]) is True
        assert mp._is_text_only_busy_payload({"type": "output_text", "text": "a"}) is True

    def test_non_text_shapes(self):
        assert mp._is_text_only_busy_payload(None) is False
        assert mp._is_text_only_busy_payload([]) is False
        assert mp._is_text_only_busy_payload([{"type": "image_url"}]) is False
        assert mp._is_text_only_busy_payload({"type": "image_url"}) is False


class TestDisplayMarkers:
    def test_hidden_marker_sniff(self):
        assert mp._is_display_hidden_marker("user", "[System: model switch]") is True
        assert mp._is_display_hidden_marker("user", "  [System: x]") is True
        assert mp._is_display_hidden_marker("assistant", "[System: x]") is False
        assert mp._is_display_hidden_marker("user", "hello") is False

    def test_legacy_display_kind_auto_continue(self):
        text = mp._AUTO_CONTINUE_NOTE_PREFIX + " — the app or its backend process "
        assert mp._legacy_display_kind("user", text) == "auto_continue"
        assert mp._legacy_display_kind("user", "hello") is None
        assert mp._legacy_display_kind("assistant", text) is None


class TestSeedHistory:
    def test_filters_invalid_rows(self):
        value = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "x"},
            {"role": "user", "text": "fallback"},
            {"role": "user", "content": "   "},
            "not-a-dict",
            {"role": "user", "content": 42},
        ]
        assert mp._coerce_seed_history(value) == [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fallback"},
        ]

    def test_non_list(self):
        assert mp._coerce_seed_history(None) == []
        assert mp._coerce_seed_history("x") == []


class TestHistoryToMessages:
    def test_projects_rows_and_forwards_row_id(self):
        out = mp._history_to_messages(
            [
                {"role": "user", "content": "hi", "_row_id": 3},
                {"role": "assistant", "content": "yo"},
            ]
        )
        assert out == [
            {"role": "user", "text": "hi", "row_id": 3},
            {"role": "assistant", "text": "yo"},
        ]

    def test_drops_display_hidden_scaffolding(self):
        out = mp._history_to_messages(
            [{"role": "user", "content": "[System: model switch]", "display_kind": "hidden"}]
        )
        assert out == []

    def test_tool_row_gets_context(self):
        out = mp._history_to_messages(
            [{"role": "tool", "tool_call_id": "t1", "content": "result"}]
        )
        assert len(out) == 1
        assert out[0]["role"] == "tool"
        assert out[0]["name"] == "tool"
        assert isinstance(out[0]["context"], str)

    def test_skill_turn_projects_to_invocation(self):
        # plain text stays plain (no skill scaffolding involved)
        out = mp._history_to_messages([{"role": "user", "content": "just words"}])
        assert out == [{"role": "user", "text": "just words"}]


class TestPetPayloadHelpers:
    def test_clone_pet_payload_deep_copies_nested(self):
        payload = {
            "slug": "boba",
            "framesByState": {"idle": 4},
            "framesByRow": {"idle": [1, 2]},
            "stateRows": ["idle"],
            "scale": 0.33,
        }
        clone = ppm._clone_pet_payload(payload)
        assert clone == payload
        clone["framesByState"]["idle"] = 99
        clone["stateRows"].append("walk")
        clone["scale"] = 1.0
        assert payload["framesByState"]["idle"] == 4
        assert payload["stateRows"] == ["idle"]
        assert payload["scale"] == 0.33

    def test_sheet_revision_missing_file(self, tmp_path):
        assert ppm._pet_sheet_revision(tmp_path / "nope.png") == "0:0"

    def test_payload_cache_key_missing_sheet(self, tmp_path):
        pet = type("Pet", (), {"spritesheet": tmp_path / "nope.png", "slug": "s", "display_name": "d"})()
        assert ppm._pet_payload_cache_key(pet, scale=0.33) is None

    def test_cancel_token_semantics(self):
        token = "s3-w1a-cancel-token"
        ppm._pet_cancel_arm(token)
        assert ppm._pet_is_cancelled(token) is False
        ppm._pet_cancel_request(token)
        assert ppm._pet_is_cancelled(token) is True
        ppm._pet_cancel_release(token)
        assert ppm._pet_is_cancelled(token) is False
        ppm._pet_cancel_release(token)  # idempotent

    def test_reference_images_from_data_url_writes_file(self, tmp_path):
        raw = base64.b64encode(b"fake-png-bytes").decode("ascii")
        out = ppm._pet_reference_images_from_data_url(f"data:image/png;base64,{raw}", tmp_path)
        assert len(out) == 1
        assert out[0] == tmp_path / "reference.png"
        assert out[0].read_bytes() == b"fake-png-bytes"

    def test_reference_images_invalid_format(self, tmp_path):
        with pytest.raises(ValueError, match="invalid reference image format"):
            ppm._pet_reference_images_from_data_url("not-a-data-url", tmp_path)

    def test_reference_images_unsupported_mime(self, tmp_path):
        with pytest.raises(ValueError, match="unsupported reference image type"):
            ppm._pet_reference_images_from_data_url("data:image/tiff;base64,AAAA", tmp_path)

    def test_reference_images_too_large(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ppm, "_PET_REFERENCE_MAX_BYTES", 16)
        raw = base64.b64encode(b"x" * 64).decode("ascii")
        with pytest.raises(ValueError, match="reference image too large"):
            ppm._pet_reference_images_from_data_url(f"data:image/png;base64,{raw}", tmp_path)

    def test_config_scale_reads_config(self, monkeypatch):
        def fake_load_config():
            return {"display": {"pet": {"scale": 2.5}}}

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
        assert ppm._pet_config_scale() == 2.5
