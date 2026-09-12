import json
import os
import pytest
from unittest.mock import patch, MagicMock

from agent.context_compressor import _truncate_tool_call_args_json
from tools.file_tools import write_file_tool, patch_tool


class TestTruncationPlaceholderGuard:
    """Issue #83714: Guard against copyable truncation markers in compressor and file writes."""

    def test_compressor_does_not_emit_copyable_truncation_marker(self):
        """Producer: _truncate_tool_call_args_json must not append '...[truncated]' to shrunken args."""
        payload = json.dumps({
            "path": "test.py",
            "content": "def foo():\n" + "    pass\n" * 100,
        })
        assert len(payload) > 500
        shrunk = _truncate_tool_call_args_json(payload, head_chars=200)
        parsed = json.loads(shrunk)
        assert parsed["path"] == "test.py"
        assert len(parsed["content"]) <= 200
        assert "...[truncated]" not in parsed["content"]
        assert "[truncated]" not in parsed["content"]

    def test_write_file_rejects_truncation_placeholder(self, tmp_path):
        """Consumer: write_file_tool must fail closed when content contains truncation marker."""
        target = tmp_path / "sample.py"
        bad_content = "def foo():\n    ...[truncated]\n"
        result_raw = write_file_tool(str(target), bad_content)
        result = json.loads(result_raw)
        assert "error" in result
        assert "truncation placeholder" in result["error"].lower()
        assert not target.exists()

    def test_patch_replace_rejects_truncation_placeholder(self, tmp_path):
        """Consumer: patch_tool (replace mode) must reject new_string with truncation placeholder."""
        target = tmp_path / "sample.py"
        target.write_text("def foo():\n    pass\n", encoding="utf-8")

        bad_new = "def foo():\n    ...[truncated]\n"
        result_raw = patch_tool(
            mode="replace",
            path=str(target),
            old_string="def foo():\n    pass\n",
            new_string=bad_new,
        )
        result = json.loads(result_raw)
        assert "error" in result
        assert "truncation placeholder" in result["error"].lower()
        # Ensure file content was not modified
        assert target.read_text(encoding="utf-8") == "def foo():\n    pass\n"

    def test_patch_replace_allows_preexisting_placeholder(self, tmp_path):
        """Consumer: patch_tool must allow marker in new_string if already present in old_string."""
        target = tmp_path / "sample.py"
        original = "# Note: ...[truncated] in docs\ndef foo():\n    pass\n"
        target.write_text(original, encoding="utf-8")

        # Edit line near the marker without removing or duplicating it
        old_str = "# Note: ...[truncated] in docs\ndef foo():\n    pass\n"
        new_str = "# Note: ...[truncated] in docs\ndef foo():\n    return 42\n"
        result_raw = patch_tool(
            mode="replace",
            path=str(target),
            old_string=old_str,
            new_string=new_str,
        )
        result = json.loads(result_raw)
        assert "error" not in result
        assert target.read_text(encoding="utf-8") == new_str

    def test_patch_v4a_rejects_truncation_placeholder(self, tmp_path):
        """Consumer: patch_tool (V4A patch mode) must reject patch adding truncation placeholder."""
        target = tmp_path / "sample.py"
        target.write_text("def foo():\n    pass\n", encoding="utf-8")

        v4a_patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            "-def foo():\n"
            "-    pass\n"
            "+def foo():\n"
            "+    ...[truncated]\n"
            "*** End Patch\n"
        )
        result_raw = patch_tool(mode="patch", patch=v4a_patch)
        result = json.loads(result_raw)
        assert "error" in result
        assert "truncation placeholder" in result["error"].lower()
        assert target.read_text(encoding="utf-8") == "def foo():\n    pass\n"

    def test_patch_v4a_allows_preexisting_context_placeholder(self, tmp_path):
        """Consumer: V4A patch with context line having placeholder is not rejected."""
        target = tmp_path / "sample.py"
        original = "# ...[truncated]\ndef foo():\n    old_call()\n"
        target.write_text(original, encoding="utf-8")

        v4a_patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            " # ...[truncated]\n"
            " def foo():\n"
            "-    old_call()\n"
            "+    new_call()\n"
            "*** End Patch\n"
        )
        result_raw = patch_tool(mode="patch", patch=v4a_patch)
        result = json.loads(result_raw)
        assert "error" not in result
        assert "new_call()" in target.read_text(encoding="utf-8")

    def test_patch_replace_rejects_additional_placeholder(self, tmp_path):
        """Consumer: patch_tool must reject when new_string introduces an ADDITIONAL placeholder."""
        target = tmp_path / "sample.py"
        original = "# Note: ...[truncated] in docs\ndef foo():\n    return 1\n"
        target.write_text(original, encoding="utf-8")

        # new_string retains the original placeholder BUT also adds a second one in place of code
        bad_new = "# Note: ...[truncated] in docs\ndef foo():\n    ...[truncated]\n"
        result_raw = patch_tool(
            mode="replace",
            path=str(target),
            old_string=original,
            new_string=bad_new,
        )
        result = json.loads(result_raw)
        assert "error" in result
        assert "truncation placeholder" in result["error"].lower()
        assert target.read_text(encoding="utf-8") == original

    def test_patch_v4a_add_file_combined_with_context_placeholder_allowed(self, tmp_path):
        """Consumer: V4A patch with Add File and an Update File with context placeholder must succeed."""
        new_file = tmp_path / "new_module.py"
        existing_file = tmp_path / "existing.py"
        existing_file.write_text("# ...[truncated]\ndef old():\n    return 1\n", encoding="utf-8")

        v4a_patch = (
            "*** Begin Patch\n"
            f"*** Add File: {new_file}\n"
            "+def added():\n"
            "+    return 42\n"
            f"*** Update File: {existing_file}\n"
            "@@\n"
            " # ...[truncated]\n"
            " def old():\n"
            "-    return 1\n"
            "+    return 2\n"
            "*** End Patch\n"
        )
        result_raw = patch_tool(mode="patch", patch=v4a_patch)
        result = json.loads(result_raw)
        assert "error" not in result
        assert new_file.read_text(encoding="utf-8") == "def added():\nreturn 42" or "return 42" in new_file.read_text(encoding="utf-8")
        assert "return 2" in existing_file.read_text(encoding="utf-8")

    def test_patch_v4a_add_file_rejects_placeholder(self, tmp_path):
        """Consumer: V4A patch with Add File containing truncation placeholder must be rejected."""
        new_file = tmp_path / "new_module.py"
        v4a_patch = (
            "*** Begin Patch\n"
            f"*** Add File: {new_file}\n"
            "+def added():\n"
            "+    ...[truncated]\n"
            "*** End Patch\n"
        )
        result_raw = patch_tool(mode="patch", patch=v4a_patch)
        result = json.loads(result_raw)
        assert "error" in result
        assert "truncation placeholder" in result["error"].lower()
        assert not new_file.exists()

    @pytest.mark.parametrize("placeholder", [
        "// ... unchanged ...",
        "// ... rest unchanged ...",
        "// ... rest of file unchanged ...",
        "// ... rest of code unchanged ...",
        "// ... unchanged",
        "# ... unchanged",
        "# ... rest of file ...",
        "/* ... unchanged ... */",
        "<!-- ... unchanged ... -->",
        "⟪HERMES-CONTEXT-COMPRESSION: 100 of 500 chars omitted here⟫",
    ])
    def test_write_file_rejects_broad_truncation_variants(self, tmp_path, placeholder):
        """Consumer: write_file_tool must reject common comment and compression markers."""
        target = tmp_path / "variant.py"
        content = f"def foo():\n    {placeholder}\n"
        result = json.loads(write_file_tool(str(target), content))
        assert "error" in result
        assert "truncation placeholder" in result["error"].lower()
        assert not target.exists()
