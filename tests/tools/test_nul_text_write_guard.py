"""A NUL (U+0000) in a text edit is refused before it is stored (port of cloudflare/cloudflare-os#1010).

read_file treats any NUL as binary, so a stored one locks the agent out of the file it just wrote.
"""

import json

from tools.file_tools import patch_tool, write_file_tool


def test_text_edits_with_nul_are_refused_and_leave_the_file_untouched(tmp_path):
    target = tmp_path / "f.py"
    target.write_text("print(x)\n")

    results = [
        write_file_tool(str(target), 'x = "a\x00b"\n'),
        patch_tool("replace", path=str(target), old_string="print(x)", new_string="print('\x00')"),
        patch_tool("patch", patch=f"*** Begin Patch\n*** Update File: {target}\n@@\n-print(x)\n+print('\x00')\n*** End Patch\n"),
    ]

    for raw in results:
        assert "NUL" in json.loads(raw)["error"]
    assert target.read_bytes() == b"print(x)\n"


def test_escaped_nul_written_as_text_still_works(tmp_path):
    target = tmp_path / "f.py"
    assert "error" not in json.loads(write_file_tool(str(target), 'x = "a\\x00b"\n'))
    assert target.read_bytes() == b'x = "a\\x00b"\n'
