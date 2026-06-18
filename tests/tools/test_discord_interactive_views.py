# tests/tools/test_discord_interactive_views.py
"""Unit tests for tools/discord_interactive_views.py.

The views (InteractivePromptView, InteractivePromptModal) require ``discord.py``
at class-definition time (they inherit from ``discord.ui.View`` / ``discord.ui.Modal``),
so the entire test module is skipped when ``discord`` is not installed.

The pure helper ``_component_check_auth`` does **not** require discord and is
always tested.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Set
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# _component_check_auth — pure function, always testable
# ---------------------------------------------------------------------------
from tools.discord_interactive_views import _component_check_auth, unwrap_modal_children


# -- mock interaction helpers ------------------------------------------------

def _make_interaction(
    user_id: str = "123",
    role_ids: Optional[list[str]] = None,
) -> MagicMock:
    """Build a lightweight mock that quacks like discord.Interaction."""
    user = MagicMock()
    user.id = int(user_id)
    if role_ids is not None:
        user.roles = [MagicMock(id=int(rid)) for rid in role_ids]
    else:
        user.roles = []
    interaction = MagicMock()
    interaction.user = user
    return interaction


# ===========================================================================
# Tests for _component_check_auth
# ===========================================================================


class TestComponentCheckAuth:
    """Pure-function auth checker — no discord import required."""

    def test_both_empty_allows(self):
        """Both allowlists empty → allow everyone."""
        assert _component_check_auth(_make_interaction(), None, None) is True

    def test_user_allowed(self):
        """User ID in allowed_user_ids → allow."""
        assert _component_check_auth(
            _make_interaction("42"), {"42"}, None,
        ) is True

    def test_user_not_allowed(self):
        """User ID not in allowed_user_ids, no roles → reject."""
        assert _component_check_auth(
            _make_interaction("99"), {"42"}, None,
        ) is False

    def test_role_allowed(self):
        """User has a role in allowed_role_ids → allow."""
        assert _component_check_auth(
            _make_interaction("99", role_ids=["55"]),
            None, {55},
        ) is True

    def test_role_not_allowed(self):
        """User's roles don't intersect allowed_role_ids → reject."""
        assert _component_check_auth(
            _make_interaction("99", role_ids=["10", "20"]),
            None, {55},
        ) is False

    def test_user_or_role_combined(self):
        """User in user_set OR role in role_set → allow."""
        # User matches
        assert _component_check_auth(
            _make_interaction("42", role_ids=["10"]),
            {"42"}, {55},
        ) is True
        # Role matches
        assert _component_check_auth(
            _make_interaction("99", role_ids=["55"]),
            {"42"}, {55},
        ) is True
        # Neither matches
        assert _component_check_auth(
            _make_interaction("99", role_ids=["10"]),
            {"42"}, {55},
        ) is False

    def test_no_user_attribute(self):
        """Interaction without .user → reject."""
        interaction = MagicMock(spec=[])  # no attributes
        assert _component_check_auth(interaction, {"42"}, None) is False

    def test_user_without_id(self):
        """User object without .id → reject gracefully."""
        user = MagicMock(spec=[])  # no .id
        interaction = MagicMock()
        interaction.user = user
        assert _component_check_auth(interaction, {"42"}, None) is False

    def test_roles_not_iterable(self):
        """User.roles is not iterable → reject gracefully."""
        user = MagicMock()
        user.id = 99
        user.roles = "not-a-list"  # string is iterable but items have no .id
        interaction = MagicMock()
        interaction.user = user
        # This should not crash; return False
        result = _component_check_auth(interaction, None, {42})
        assert result is False

    def test_string_role_ids(self):
        """Role IDs are compared as integers (adapter stores them as int)."""
        assert _component_check_auth(
            _make_interaction("1", role_ids=["100"]),
            None, {100},
        ) is True


# ===========================================================================
# Tests for view construction (requires discord.py)
# ===========================================================================

discord = pytest.importorskip("discord")
_ui = discord.ui

from tools.discord_interactive_views import (
    InteractivePromptView,
    InteractivePromptModal,
    build_prompt_embed,
)


def _make_options(n: int = 2) -> list[dict[str, Any]]:
    return [
        {"label": f"Opt {i}", "value": f"opt_{i}"}
        for i in range(n)
    ]


class TestInteractivePromptViewConstruction:
    """Test view construction and configuration (no Discord connection)."""

    def test_custom_id_button_length(self):
        """Button custom_id must be ≤ 100 chars (Discord limit)."""
        long_prompt_id = "a" * 80  # 80-char prompt_id
        view = InteractivePromptView(
            prompt_id=long_prompt_id,
            question="Q?",
            options=_make_options(3),
            allowed_user_ids=set(),
            allowed_role_ids=set(),
            auth_policy="session_owner_only",
            origin_user_id="123",
            timeout_seconds=900,
        )
        for child in view.children:
            if isinstance(child, discord.ui.Button):
                assert len(child.custom_id) <= 100, (
                    f"Button custom_id too long: {len(child.custom_id)} chars"
                )

    def test_custom_id_modal_length(self):
        """Modal custom_id triggered from a button must be ≤ 100 chars."""
        long_prompt_id = "b" * 80
        # Simulate what _resolve_choice does: create a modal with option_index
        modal = InteractivePromptModal(
            prompt_id=long_prompt_id,
            option_index=2,
            modal_spec={
                "title": "Test",
                "fields": [{"key": "k", "label": "L", "type": "text"}],
            },
            original_view=None,
        )
        assert len(modal.custom_id) <= 100, (
            f"Modal custom_id too long: {len(modal.custom_id)} chars"
        )

    def test_view_timeout_default(self):
        """Default timeout is 900 seconds."""
        view = InteractivePromptView(
            prompt_id="test",
            question="Q?",
            options=_make_options(1),
            allowed_user_ids=set(),
            allowed_role_ids=set(),
            auth_policy="session_owner_only",
            origin_user_id="123",
            timeout_seconds=900,
        )
        assert view.timeout == 900

    def test_view_timeout_custom(self):
        """Custom timeout is forwarded correctly."""
        view = InteractivePromptView(
            prompt_id="test",
            question="Q?",
            options=_make_options(1),
            allowed_user_ids=set(),
            allowed_role_ids=set(),
            auth_policy="session_owner_only",
            origin_user_id="123",
            timeout_seconds=1800,
        )
        assert view.timeout == 1800

    def test_view_timeout_capped_at_3600(self):
        """Timeout > 3600 is capped to 3600."""
        view = InteractivePromptView(
            prompt_id="test",
            question="Q?",
            options=_make_options(1),
            allowed_user_ids=set(),
            allowed_role_ids=set(),
            auth_policy="session_owner_only",
            origin_user_id="123",
            timeout_seconds=9999,
        )
        assert view.timeout == 3600

    def test_view_max_25_buttons(self):
        """View with 25 options creates exactly 25 buttons."""
        opts = _make_options(25)
        view = InteractivePromptView(
            prompt_id="test",
            question="Q?",
            options=opts,
            allowed_user_ids=set(),
            allowed_role_ids=set(),
            auth_policy="session_owner_only",
            origin_user_id="123",
            timeout_seconds=900,
        )
        buttons = [c for c in view.children if isinstance(c, discord.ui.Button)]
        assert len(buttons) == 25

    def test_modal_uses_index_not_value(self):
        """Modal receives option_index, not the value string."""
        modal = InteractivePromptModal(
            prompt_id="test",
            option_index=7,
            modal_spec={
                "title": "Test",
                "fields": [{"key": "k", "label": "L", "type": "text"}],
            },
            original_view=None,
        )
        assert modal.option_index == 7
        # custom_id should contain the index, not a value string
        assert ":7" in modal.custom_id


class TestInteractivePromptModalFieldTypes:
    """Test modal construction with select, radio, and checkbox field types."""

    def _modal_with_fields(self, fields: list[dict[str, Any]]) -> InteractivePromptModal:
        return InteractivePromptModal(
            prompt_id="test",
            option_index=0,
            modal_spec={"title": "Test", "fields": fields},
            original_view=None,
        )

    def test_select_field_creates_select_component(self):
        """A 'select' field type creates a discord.ui.Select child (wrapped in Label)."""
        modal = self._modal_with_fields([
            {"key": "priority", "label": "Priority", "type": "select",
             "options": ["Low", "Medium", "High"]},
        ])
        inner = unwrap_modal_children(modal.children)
        selects = [c for c in inner if isinstance(c, discord.ui.Select)]
        assert len(selects) == 1
        assert len(selects[0].options) == 3
        assert [o.label for o in selects[0].options] == ["Low", "Medium", "High"]

    def test_select_field_custom_id_uses_key(self):
        """Select component's custom_id matches the field key."""
        modal = self._modal_with_fields([
            {"key": "my_select", "label": "Pick", "type": "select",
             "options": ["A", "B"]},
        ])
        inner = unwrap_modal_children(modal.children)
        select = [c for c in inner if isinstance(c, discord.ui.Select)][0]
        assert select.custom_id == "my_select"

    def test_select_field_capped_at_25_options(self):
        """Select options are capped at Discord's 25 max."""
        opts = [f"Option {i}" for i in range(30)]
        modal = self._modal_with_fields([
            {"key": "sel", "label": "Pick", "type": "select", "options": opts},
        ])
        inner = unwrap_modal_children(modal.children)
        select = [c for c in inner if isinstance(c, discord.ui.Select)][0]
        assert len(select.options) == 25

    def test_select_field_empty_options(self):
        """Select with no options creates an empty dropdown."""
        modal = self._modal_with_fields([
            {"key": "sel", "label": "Pick", "type": "select", "options": []},
        ])
        inner = unwrap_modal_children(modal.children)
        select = [c for c in inner if isinstance(c, discord.ui.Select)][0]
        assert len(select.options) == 0
        assert "sel" in modal._field_keys

    def test_radio_field_creates_radio_group(self):
        """A 'radio' field type creates a discord.ui.RadioGroup child (wrapped in Label)."""
        modal = self._modal_with_fields([
            {"key": "category", "label": "Category", "type": "radio",
             "options": ["Bug", "Feature"]},
        ])
        inner = unwrap_modal_children(modal.children)
        radios = [c for c in inner if isinstance(c, discord.ui.RadioGroup)]
        assert len(radios) == 1
        assert len(radios[0].options) == 2

    def test_radio_field_capped_at_10_options(self):
        """Radio options are capped at Discord's 10 max."""
        opts = [f"Choice {i}" for i in range(15)]
        modal = self._modal_with_fields([
            {"key": "rg", "label": "Pick one", "type": "radio", "options": opts},
        ])
        inner = unwrap_modal_children(modal.children)
        radio = [c for c in inner if isinstance(c, discord.ui.RadioGroup)][0]
        assert len(radio.options) == 10

    def test_checkbox_field_creates_checkbox_group(self):
        """A 'checkbox' field type creates a discord.ui.CheckboxGroup child (wrapped in Label)."""
        modal = self._modal_with_fields([
            {"key": "tags", "label": "Tags", "type": "checkbox",
             "options": ["urgent", "backend"]},
        ])
        inner = unwrap_modal_children(modal.children)
        checkboxes = [c for c in inner
                     if isinstance(c, discord.ui.CheckboxGroup)]
        assert len(checkboxes) == 1
        assert len(checkboxes[0].options) == 2

    def test_checkbox_field_capped_at_10_options(self):
        """Checkbox options are capped at Discord's 10 max."""
        opts = [f"Tag {i}" for i in range(12)]
        modal = self._modal_with_fields([
            {"key": "cb", "label": "Tags", "type": "checkbox", "options": opts},
        ])
        inner = unwrap_modal_children(modal.children)
        cb = [c for c in inner
              if isinstance(c, discord.ui.CheckboxGroup)][0]
        assert len(cb.options) == 10

    def test_mixed_field_types_all_present(self):
        """Modal with text, select, radio, and checkbox creates all components (wrapped in Labels)."""
        modal = self._modal_with_fields([
            {"key": "name", "label": "Name", "type": "text"},
            {"key": "priority", "label": "Priority", "type": "select",
             "options": ["Low", "High"]},
            {"key": "category", "label": "Category", "type": "radio",
             "options": ["Bug", "Feature"]},
            {"key": "tags", "label": "Tags", "type": "checkbox",
             "options": ["A"]},
        ])
        assert len(modal.children) == 4
        inner = unwrap_modal_children(modal.children)
        assert isinstance(inner[0], discord.ui.TextInput)
        assert isinstance(inner[1], discord.ui.Select)
        assert isinstance(inner[2], discord.ui.RadioGroup)
        assert isinstance(inner[3], discord.ui.CheckboxGroup)
        assert modal._field_keys == ["name", "priority", "category", "tags"]

    def test_mixed_fields_at_modal_limit(self):
        """5 fields exactly fills Discord's 5-component limit."""
        modal = self._modal_with_fields([
            {"key": "f1", "label": "F1", "type": "text"},
            {"key": "f2", "label": "F2", "type": "select", "options": ["A"]},
            {"key": "f3", "label": "F3", "type": "radio", "options": ["X"]},
            {"key": "f4", "label": "F4", "type": "checkbox", "options": ["P"]},
            {"key": "f5", "label": "F5", "type": "text"},
        ])
        assert len(modal.children) == 5
        assert modal._field_keys == ["f1", "f2", "f3", "f4", "f5"]

    def test_modal_limit_enforces_5_children(self):
        """6th field is dropped — Discord's modal max is 5 children."""
        modal = self._modal_with_fields([
            {"key": "f1", "label": "F1", "type": "text"},
            {"key": "f2", "label": "F2", "type": "select", "options": ["A"]},
            {"key": "f3", "label": "F3", "type": "radio", "options": ["X"]},
            {"key": "f4", "label": "F4", "type": "checkbox", "options": ["P"]},
            {"key": "f5", "label": "F5", "type": "text"},
            {"key": "f6", "label": "F6", "type": "text"},
        ])
        assert len(modal.children) == 5
        assert "f6" not in modal._field_keys

    def test_select_optional_min_values_zero(self):
        """Select with required=False should allow skipping (min_values=0)."""
        modal = self._modal_with_fields([
            {"key": "sel", "label": "Pick", "type": "select",
             "required": False, "options": ["A", "B"]},
        ])
        inner = unwrap_modal_children(modal.children)
        select = [c for c in inner if isinstance(c, discord.ui.Select)][0]
        assert select.min_values == 0

    def test_select_required_min_values_one(self):
        """Select with required=True should enforce selection (min_values=1)."""
        modal = self._modal_with_fields([
            {"key": "sel", "label": "Pick", "type": "select",
             "required": True, "options": ["A", "B"]},
        ])
        inner = unwrap_modal_children(modal.children)
        select = [c for c in inner if isinstance(c, discord.ui.Select)][0]
        assert select.min_values == 1

    def test_custom_id_truncated_at_100_chars(self):
        """Field custom_id is truncated to Discord's 100-char limit."""
        long_key = "k" * 120
        modal = self._modal_with_fields([
            {"key": long_key, "label": "Pick", "type": "select",
             "options": ["A"]},
        ])
        inner = unwrap_modal_children(modal.children)
        select = [c for c in inner if isinstance(c, discord.ui.Select)][0]
        assert len(select.custom_id) <= 100

    def test_file_upload_field_creates_file_upload_component(self):
        """file_upload field type creates a FileUpload wrapped in Label."""
        modal = self._modal_with_fields([
            {"key": "file", "label": "Upload", "type": "file_upload"},
        ])
        assert len(modal.children) == 1
        label = modal.children[0]
        assert isinstance(label, _ui.Label)
        assert label.text == "Upload"
        assert isinstance(label.component, _ui.FileUpload)
        assert label.component.custom_id == "file"
        assert label.component.required is False
        assert label.component.max_values == 1
        assert "file" in modal._field_keys

    def test_file_upload_with_file_policy(self):
        """file_policy.max_files and min_files are forwarded to FileUpload."""
        modal = self._modal_with_fields([
            {
                "key": "files",
                "label": "Reports",
                "type": "file_upload",
                "required": True,
                "file_policy": {"max_files": 5, "min_files": 1},
            },
        ])
        fu = modal.children[0].component
        assert fu.required is True
        assert fu.max_values == 5
        assert fu.min_values == 1


# ===========================================================================
# Label wrapper tests — TDD for Discord modal API migration (Sep 2025)
# ===========================================================================


class TestLabelWrapperMigration:
    """Tests for discord.ui.Label wrapping in InteractivePromptModal.

    Discord's Sep 2025 modal API change requires ALL interactive components
    inside modals to be wrapped in discord.ui.Label (type 18).
    """

    def _modal_with_fields(self, fields: list[dict[str, Any]]) -> InteractivePromptModal:
        return InteractivePromptModal(
            prompt_id="test",
            option_index=0,
            modal_spec={"title": "Test", "fields": fields},
            original_view=None,
        )

    # -- T1: Label wrapper in modal payload --------------------------------

    def test_label_wraps_text_input_in_payload(self):
        """Modal with text field → to_dict() has Label(type:18) wrapping TextInput(type:4)."""
        modal = self._modal_with_fields([
            {"key": "name", "label": "Your Name", "type": "text"},
        ])
        d = modal.to_dict()
        components = d["components"]
        assert len(components) == 1

        label_comp = components[0]
        assert label_comp["type"] == 18, "Top-level should be Label (type 18)"
        assert label_comp["label"] == "Your Name"

        inner = label_comp["component"]
        assert inner["type"] == 4, "Inner should be TextInput (type 4)"
        # Label should own the label text; inner TextInput label should be null
        assert inner.get("label") is None, "Inner TextInput label should be null"

    def test_label_wraps_select_in_payload(self):
        """Modal with select field → to_dict() has Label(type:18) wrapping Select(type:3)."""
        modal = self._modal_with_fields([
            {"key": "priority", "label": "Priority", "type": "select",
             "options": ["Low", "High"]},
        ])
        d = modal.to_dict()
        components = d["components"]
        assert len(components) == 1

        label_comp = components[0]
        assert label_comp["type"] == 18
        assert label_comp["label"] == "Priority"

        inner = label_comp["component"]
        assert inner["type"] == 3, "Inner should be Select (type 3)"

    # -- T2: Label.description mapping --------------------------------------

    def test_label_description_in_payload(self):
        """Field description maps to Label.description in modal payload."""
        modal = self._modal_with_fields([
            {"key": "email", "label": "Email", "type": "text",
             "description": "Enter your work email"},
        ])
        d = modal.to_dict()
        label_comp = d["components"][0]
        assert label_comp.get("description") == "Enter your work email"

    def test_label_description_truncated_to_100(self):
        """Description >100 chars is truncated to 100 characters."""
        long_desc = "x" * 150
        modal = self._modal_with_fields([
            {"key": "f", "label": "F", "type": "text",
             "description": long_desc},
        ])
        d = modal.to_dict()
        label_comp = d["components"][0]
        assert len(label_comp["description"]) == 100

    def test_label_no_description_when_absent(self):
        """No description key in payload when field has no description."""
        modal = self._modal_with_fields([
            {"key": "name", "label": "Name", "type": "text"},
        ])
        d = modal.to_dict()
        label_comp = d["components"][0]
        assert "description" not in label_comp

    # -- T3: Submit value extraction with Labels ---------------------------

    def test_submit_extracts_values_through_labels(self):
        """on_submit correctly extracts values from Label-wrapped components."""
        modal = self._modal_with_fields([
            {"key": "name", "label": "Name", "type": "text"},
            {"key": "priority", "label": "Priority", "type": "select",
             "options": ["Low", "High"]},
        ])
        # Simulate user input by setting internal values on inner components
        inner_components = unwrap_modal_children(modal.children)
        inner_components[0]._value = "Alice"
        inner_components[1]._values = ["High"]

        # Extract fields using the same logic as on_submit
        fields: dict[str, Any] = {}
        for idx, child in enumerate(modal.children):
            if idx >= len(modal._field_keys):
                break
            field_key = modal._field_keys[idx]
            inner = child.component if isinstance(child, discord.ui.Label) else child
            if isinstance(inner, (discord.ui.TextInput, discord.ui.RadioGroup)):
                fields[field_key] = inner.value
            elif isinstance(inner, (discord.ui.Select, discord.ui.CheckboxGroup)):
                fields[field_key] = inner.values
            else:
                fields[field_key] = getattr(inner, "value", None)

        assert fields == {"name": "Alice", "priority": ["High"]}

    def test_submit_extracts_radio_value_through_label(self):
        """RadioGroup value extracted through Label wrapper."""
        modal = self._modal_with_fields([
            {"key": "category", "label": "Category", "type": "radio",
             "options": ["Bug", "Feature"]},
        ])
        inner = unwrap_modal_children(modal.children)[0]
        inner._value = "Bug"

        fields = {}
        for idx, child in enumerate(modal.children):
            if idx >= len(modal._field_keys):
                break
            field_key = modal._field_keys[idx]
            inner = child.component if isinstance(child, discord.ui.Label) else child
            if isinstance(inner, (discord.ui.TextInput, discord.ui.RadioGroup)):
                fields[field_key] = inner.value
            elif isinstance(inner, (discord.ui.Select, discord.ui.CheckboxGroup)):
                fields[field_key] = inner.values
            else:
                fields[field_key] = getattr(inner, "value", None)

        assert fields == {"category": "Bug"}

    # -- T4: Select.disabled NOT in modal payload --------------------------

    def test_select_no_disabled_in_modal_payload(self):
        """Select component inside modal payload must not have 'disabled' key."""
        modal = self._modal_with_fields([
            {"key": "pick", "label": "Pick", "type": "select",
             "options": ["A", "B"]},
        ])
        d = modal.to_dict()
        inner = d["components"][0]["component"]
        assert "disabled" not in inner, (
            "disabled must not be present in Select inside modal"
        )

    # -- T5: 5-child limit with Labels -------------------------------------

    def test_5_child_limit_with_label_wrapping(self):
        """6 fields → only 5 Labels added, _field_keys length 5."""
        modal = self._modal_with_fields([
            {"key": "f1", "label": "F1", "type": "text"},
            {"key": "f2", "label": "F2", "type": "select", "options": ["A"]},
            {"key": "f3", "label": "F3", "type": "radio", "options": ["X"]},
            {"key": "f4", "label": "F4", "type": "checkbox", "options": ["P"]},
            {"key": "f5", "label": "F5", "type": "text"},
            {"key": "f6", "label": "F6", "type": "text"},
        ])
        # All children should be Labels
        for child in modal.children:
            assert isinstance(child, discord.ui.Label), (
                f"Expected Label, got {type(child).__name__}"
            )
        assert len(modal.children) == 5
        assert len(modal._field_keys) == 5
        assert "f6" not in modal._field_keys

    # -- T6: Backward compat text-only -------------------------------------

    def test_all_text_fields_all_wrapped_in_labels(self):
        """All text fields → all wrapped in Labels, payload is valid."""
        modal = self._modal_with_fields([
            {"key": "name", "label": "Name", "type": "text"},
            {"key": "email", "label": "Email", "type": "text"},
            {"key": "notes", "label": "Notes", "type": "text",
             "multiline": True},
        ])
        d = modal.to_dict()
        assert len(d["components"]) == 3
        for comp in d["components"]:
            assert comp["type"] == 18, "Each top-level should be Label"
            assert "component" in comp
            assert comp["component"]["type"] == 4, "Inner should be TextInput"

    def test_all_children_are_labels(self):
        """Every modal child should be a discord.ui.Label instance."""
        modal = self._modal_with_fields([
            {"key": "name", "label": "Name", "type": "text"},
            {"key": "priority", "label": "Priority", "type": "select",
             "options": ["Low", "High"]},
            {"key": "category", "label": "Category", "type": "radio",
             "options": ["Bug", "Feature"]},
            {"key": "tags", "label": "Tags", "type": "checkbox",
             "options": ["A"]},
        ])
        for child in modal.children:
            assert isinstance(child, discord.ui.Label), (
                f"Expected Label wrapper, got {type(child).__name__}"
            )
