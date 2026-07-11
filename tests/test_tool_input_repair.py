#!/usr/bin/env python3
"""Unit tests for tool_input_repair module."""

import pytest

from agent.tool_input_repair import (
    FIELD_ALIAS_MAP,
    validate_tool_args,
    repair_tool_args,
    _rename_aliased_fields,
    append_repair_note,
)


class TestValidateToolArgs:
    """Test validate_tool_args() functionality."""

    def test_missing_required_field(self):
        """Missing required field should be reported."""
        errors = validate_tool_args("read_file", {})
        assert len(errors) > 0
        assert any("'path'" in err for err in errors)

    def test_valid_args_no_errors(self):
        """Valid arguments should pass validation."""
        args = {"path": "/etc/hosts", "offset": 1, "limit": 100}
        errors = validate_tool_args("read_file", args)
        assert errors == []

    def test_unknown_fields_not_flagged(self):
        """Unknown fields should not be flagged (tools may handle them)."""
        args = {"path": "/etc/hosts", "unknown_field": "value"}
        errors = validate_tool_args("read_file", args)
        # We only check for missing required fields, not unknown fields
        assert errors == []


class TestRenameAliasedFields:
    """Test _rename_aliased_fields() rule."""

    def test_alias_rename(self):
        """Aliased field should be renamed to canonical name."""
        args = {"file_path": "/etc/hosts"}
        result = _rename_aliased_fields("read_file", args)
        assert result == {"path": "/etc/hosts"}
        assert "file_path" not in result

    def test_canonical_present_alias_dropped(self):
        """When both canonical and alias present, canonical wins."""
        args = {"path": "/etc/hosts", "file_path": "/etc/passwd"}
        result = _rename_aliased_fields("read_file", args)
        assert result == {"path": "/etc/hosts"}
        assert "file_path" not in result

    def test_no_aliases_unchanged(self):
        """Args without aliases should be returned unchanged."""
        args = {"path": "/etc/hosts"}
        result = _rename_aliased_fields("read_file", args)
        assert result == args

    def test_unknown_tool_unchanged(self):
        """Unknown tool name should return args unchanged."""
        args = {"file_path": "/etc/hosts"}
        result = _rename_aliased_fields("unknown_tool", args)
        assert result == args

    def test_multiple_aliases_same_tool(self):
        """Multiple aliases for same tool should all be renamed."""
        args = {"file_path": "/etc/hosts", "filePath": "/etc/passwd"}
        result = _rename_aliased_fields("read_file", args)
        # Both aliases map to "path", last one wins in dict
        assert "path" in result
        assert "file_path" not in result
        assert "filePath" not in result


class TestRepairToolArgs:
    """Test repair_tool_args() main entry point."""

    def test_no_errors_returns_unchanged(self):
        """Valid args with no errors should return unchanged."""
        args = {"path": "/etc/hosts"}
        result = repair_tool_args("read_file", args, [])
        assert result == args

    def test_alias_triggers_repair(self):
        """Missing required field due to alias should trigger repair."""
        args = {"file_path": "/etc/hosts"}
        # This would fail validation because "path" is required
        errors = validate_tool_args("read_file", args)
        assert len(errors) > 0

        result = repair_tool_args("read_file", args, errors)
        # After repair, should have canonical field
        assert "path" in result
        assert "file_path" not in result

    def test_unknown_fields_pass_through(self):
        """Unknown fields should pass through unchanged."""
        args = {"path": "/etc/hosts", "unknown": "value"}
        errors = validate_tool_args("read_file", args)
        result = repair_tool_args("read_file", args, errors)
        assert result == {"path": "/etc/hosts", "unknown": "value"}

    def test_repair_does_not_mutate_file_contents(self):
        """File content strings should never be mutated by repair rules."""
        args = {
            "path": "/tmp/test.txt",
            "content": "Hello [world](http://example.com)"
        }
        errors = validate_tool_args("write_file", args)
        result = repair_tool_args("write_file", args, errors)
        assert result["content"] == "Hello [world](http://example.com)"


class TestAppendRepairNote:
    """Test append_repair_note() functionality."""

    def test_no_repairs_returns_unchanged(self):
        """Empty repairs list should return result unchanged."""
        result = '{"success": true}'
        output = append_repair_note(result, [])
        assert output == result

    def test_single_repair_appends_note(self):
        """Single repair should append note."""
        result = '{"success": true}'
        repairs = ["file_path → path"]
        output = append_repair_note(result, repairs)
        assert "[Tool input repair applied: file_path → path]" in output
        assert '{"success": true}' in output

    def test_multiple_repairs_appends_note(self):
        """Multiple repairs should append note with all repairs."""
        result = '{"success": true}'
        repairs = ["file_path → path", "oldValue → old_string"]
        output = append_repair_note(result, repairs)
        assert "[Tool input repair applied: file_path → path, oldValue → old_string]" in output
        assert '{"success": true}' in output


class TestFieldAliasMap:
    """Test FIELD_ALIAS_MAP structure."""

    def test_read_file_aliases(self):
        """read_file should have expected aliases."""
        assert "read_file" in FIELD_ALIAS_MAP
        aliases = FIELD_ALIAS_MAP["read_file"]
        assert aliases.get("file_path") == "path"
        assert aliases.get("filePath") == "path"
        assert aliases.get("filepath") == "path"

    def test_write_file_aliases(self):
        """write_file should have expected aliases."""
        assert "write_file" in FIELD_ALIAS_MAP
        aliases = FIELD_ALIAS_MAP["write_file"]
        assert aliases.get("file_path") == "path"
        assert aliases.get("filePath") == "path"
        assert aliases.get("filepath") == "path"

    def test_patch_aliases(self):
        """patch should have expected aliases."""
        assert "patch" in FIELD_ALIAS_MAP
        aliases = FIELD_ALIAS_MAP["patch"]
        assert aliases.get("oldValue") == "old_string"
        assert aliases.get("oldvalue") == "old_string"
        assert aliases.get("newValue") == "new_string"
        assert aliases.get("newvalue") == "new_string"

    def test_search_files_aliases(self):
        """search_files should have expected aliases."""
        assert "search_files" in FIELD_ALIAS_MAP
        aliases = FIELD_ALIAS_MAP["search_files"]
        assert aliases.get("file_path") == "path"
        assert aliases.get("filePath") == "path"
        assert aliases.get("filepath") == "path"


class TestIntegration:
    """Integration tests for the full repair flow."""

    def test_full_repair_flow_alias_only(self):
        """Full flow: alias triggers repair, validation passes after."""
        args = {"file_path": "/etc/hosts", "offset": 1}
        # Validate: should fail due to missing "path"
        errors = validate_tool_args("read_file", args)
        assert len(errors) > 0

        # Repair: should rename file_path → path
        repaired = repair_tool_args("read_file", args, errors)

        # Re-validate: should pass
        new_errors = validate_tool_args("read_file", repaired)
        assert new_errors == []

        # Verify the rename happened
        assert repaired == {"path": "/etc/hosts", "offset": 1}

    def test_full_repair_flow_no_alias_unchanged(self):
        """Full flow: no aliases present, args unchanged."""
        args = {"path": "/etc/hosts", "offset": 1}
        errors = validate_tool_args("read_file", args)
        # Should pass validation (no errors)
        assert errors == []

        repaired = repair_tool_args("read_file", args, errors)

        # Should be unchanged
        assert repaired == args

    def test_full_repair_flow_both_canonical_and_alias(self):
        """Full flow: valid input with canonical field is untouched."""
        args = {"path": "/etc/hosts", "file_path": "/etc/passwd", "offset": 1}
        errors = validate_tool_args("read_file", args)
        # Should pass validation (has canonical field)
        assert errors == []

        repaired = repair_tool_args("read_file", args, errors)

        # Valid input is untouched - redundant alias stays
        # (per acceptance criteria: "Valid inputs must be untouched")
        assert repaired == args