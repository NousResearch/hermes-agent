"""write_file tells the caller when it just re-sent a large file that was already on disk.

In one 1,393-agent run 661 read->whole-file-rewrites of >20k-char files cost ~25M output chars (~$155)
while `patch` (1.3k chars/call) succeeded 99.7% of the time; the tool result is the only place to say so.
"""
import json

from tools.file_tools import write_file_tool


def _big(n_lines=800):
    return "\n".join(f"line {i}: " + "x" * 40 for i in range(n_lines)) + "\n"  # ~40k chars


def test_rewrite_of_a_large_file_with_few_changes_gets_a_patch_hint(tmp_path):
    f = tmp_path / "mod.py"
    old = _big()
    f.write_text(old, encoding="utf-8")
    new = old.replace("line 400:", "line 400 (edited):").replace("line 401:", "line 401 (edited):")
    r = json.loads(write_file_tool(str(f), new, task_id="t"))
    assert r.get("error") is None and f.read_text(encoding="utf-8") == new  # the write still happens
    assert "use patch" in r["hint"] and "798 of 800 lines were already on disk" in r["hint"]


def test_new_files_small_files_and_real_rewrites_get_no_hint(tmp_path):
    new_file = tmp_path / "new.py"
    assert "hint" not in json.loads(write_file_tool(str(new_file), _big(), task_id="t"))
    small = tmp_path / "small.py"
    small.write_text("a\nb\n", encoding="utf-8")
    assert "hint" not in json.loads(write_file_tool(str(small), "a\nc\n", task_id="t"))
    big = tmp_path / "big.py"
    big.write_text(_big(), encoding="utf-8")
    rewritten = "\n".join(f"other {i}: " + "y" * 40 for i in range(800)) + "\n"
    assert "hint" not in json.loads(write_file_tool(str(big), rewritten, task_id="t"))
