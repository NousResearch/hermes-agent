"""write_file near-duplicate hint (tools/file_tools_near_duplicate.py)."""

import json
from unittest.mock import MagicMock, patch

from tools.environments.local import LocalEnvironment
from tools.file_tools_near_duplicate import find_near_duplicate


def _module(seed: int, lines: int = 40) -> str:
    return "".join(f"def fn_{seed}_{i}(x):\n    return x + {i}\n\n" for i in range(lines))


def test_new_file_that_copies_a_sibling_is_flagged_and_a_fresh_one_is_not(tmp_path):
    original = _module(seed=1)
    (tmp_path / "utils.py").write_text(original, encoding="utf-8")
    # utils2.py: same module with two functions changed — the accidental-parallel-copy shape.
    copy = original.replace("return x + 3\n", "return x * 3\n").replace("return x + 7\n", "return x - 7\n")

    match = find_near_duplicate(str(tmp_path / "utils2.py"), copy)
    assert match is not None
    sibling, ratio, _shared = match
    assert sibling == "utils.py" and ratio >= 0.85

    # Different content of similar size, and a different extension, both stay silent.
    assert find_near_duplicate(str(tmp_path / "other.py"), _module(seed=2)) is None
    assert find_near_duplicate(str(tmp_path / "utils.txt"), copy) is None
    # An EXISTING target is an overwrite, not a new copy: never flagged here.
    assert find_near_duplicate(str(tmp_path / "utils.py"), copy) is None


def test_write_file_result_carries_hint_only_for_local_env(tmp_path):
    from tools.file_tools import write_file_tool

    original = _module(seed=3)
    (tmp_path / "handler.py").write_text(original, encoding="utf-8")
    target = tmp_path / "handler_new.py"

    ops = MagicMock()
    ops.env = MagicMock(spec=LocalEnvironment)
    ops.write_file.return_value.to_dict.side_effect = lambda: {"bytes_written": len(original)}
    ops.read_file_raw.return_value = MagicMock(error="not found", content=None)
    with patch("tools.file_tools._get_file_ops", return_value=ops), \
            patch("tools.file_tools._resolve_or_none", return_value=str(target)):
        result = json.loads(write_file_tool(str(target), original))
    assert "handler.py" in result["hint"] and "parallel copy" in result["hint"]
    ops.write_file.assert_called_once()  # advisory: the write still happened

    # Remote/sandbox backends target a different filesystem: the host scan must not run.
    ops.env = object()
    with patch("tools.file_tools._get_file_ops", return_value=ops), \
            patch("tools.file_tools._resolve_or_none", return_value=str(target)):
        result = json.loads(write_file_tool(str(target), original))
    assert "hint" not in result
