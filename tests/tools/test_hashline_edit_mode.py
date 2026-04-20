import hashlib
import json

from tools.file_tools import patch_tool


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replace_mode_accepts_matching_hashline(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("hello world\n", encoding="utf-8")

    result = json.loads(
        patch_tool(
            mode="replace",
            path=str(target),
            old_string=f"# HASHLINE sha256:{_sha256(target)}\nhello world",
            new_string="hello hermes",
            task_id="hashline-ok",
        )
    )

    assert not result.get("error")
    assert "hello hermes\n" == target.read_text(encoding="utf-8")


def test_replace_mode_blocks_mismatched_hashline(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("hello world\n", encoding="utf-8")

    result = json.loads(
        patch_tool(
            mode="replace",
            path=str(target),
            old_string=f"# HASHLINE sha256:{'0' * 64}\nhello world",
            new_string="hello hermes",
            task_id="hashline-bad",
        )
    )

    assert "HASHLINE validation failed" in result["error"]
    assert target.read_text(encoding="utf-8") == "hello world\n"


def test_patch_mode_validates_all_touched_files(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    patch_text = f"""*** Begin Patch
*** Hashline: {first} sha256:{_sha256(first)}
*** Update File: {first}
@@
-alpha
+alpha patched
*** Hashline: {second} sha256:{_sha256(second)}
*** Update File: {second}
@@
-beta
+beta patched
*** End Patch
"""

    result = json.loads(patch_tool(mode="patch", patch=patch_text, task_id="hashline-multi"))

    assert not result.get("error")
    assert first.read_text(encoding="utf-8") == "alpha patched\n"
    assert second.read_text(encoding="utf-8") == "beta patched\n"


def test_patch_mode_requires_hashline_for_each_patched_file(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    patch_text = f"""*** Begin Patch
*** Hashline: {first} sha256:{_sha256(first)}
*** Update File: {first}
@@
-alpha
+alpha patched
*** Update File: {second}
@@
-beta
+beta patched
*** End Patch
"""

    result = json.loads(patch_tool(mode="patch", patch=patch_text, task_id="hashline-missing"))

    assert "missing '*** Hashline:" in result["error"]
    assert first.read_text(encoding="utf-8") == "alpha\n"
    assert second.read_text(encoding="utf-8") == "beta\n"
