"""Post-edit diff is bounded: a huge edit still lands, its preview becomes a notice."""

import time

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_operations_common import bounded_unified_diff


def test_large_replace_all_lands_without_a_slow_diff(tmp_path):
    # 20k lines, every other one changed: difflib alone takes ~10 s on this.
    target = tmp_path / "big.txt"
    target.write_text("".join(f"keep {i}\nOLD\n" for i in range(10_000)), encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))

    started = time.monotonic()
    result = ops.patch_replace(str(target), "OLD", "NEW", replace_all=True)
    elapsed = time.monotonic() - started

    assert result.success, result.error
    assert target.read_text(encoding="utf-8-sig").count("NEW") == 10_000
    assert result.diff.startswith(f"# diff omitted for {target}:")
    assert elapsed < 5
    # Repeated lines reordered: every line occurs twice, so a count-only estimate sees no change.
    lines = "".join(f"row {i}\n" for i in range(10_000)) * 2
    assert bounded_unified_diff(lines, "".join(sorted(lines.splitlines(keepends=True))), "a/f", "b/f") \
        .startswith("# diff omitted for f:")


def test_ordinary_edit_keeps_its_unified_diff(tmp_path):
    target = tmp_path / "small.py"
    target.write_text("".join(f"line {i}\n" for i in range(5_000)), encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))

    result = ops.patch_replace(str(target), "line 2500\n", "changed\n")

    assert result.success, result.error
    assert "-line 2500\n+changed\n" in result.diff
    assert result.diff.startswith("--- a/")
    # Small insert among repeated (blank) lines keeps its preview too.
    rows = "".join(f"\nrow {i}\n" for i in range(5_000))
    assert bounded_unified_diff(rows, "\n" + rows + "end\n", "a/f", "b/f").startswith("--- a/f")
