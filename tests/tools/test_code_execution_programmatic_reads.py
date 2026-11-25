"""Tool calls made from inside an execute_code cell are consumed by code, not shown to the model.

Regression for #82711 / #44843: a cell's repeated ``read_file(p)["content"]`` raised ``KeyError``
because it was answered with the conversation's "unchanged since last read" stub.
"""

import json

import pytest

from model_tools import handle_function_call
from tools.code_execution_rpc import _default_dispatch
from tools.code_kernel import CellAuthority
from tools.file_tools_read_tracking import reset_file_dedup


def _call(dispatch, name, **args):
    return json.loads(dispatch(name, args))


# Local cells dispatch through the kernel's CellAuthority; remote backends through the RPC loop.
@pytest.mark.parametrize("cell_dispatch", [lambda task: CellAuthority(task).dispatch, _default_dispatch],
                         ids=["session-kernel", "rpc"])
def test_cell_reads_always_carry_content_and_do_not_count_as_shown_to_the_model(tmp_path, cell_dispatch):
    task = "cell-reads"
    reset_file_dedup(task)
    seen, unseen = tmp_path / "seen.txt", tmp_path / "unseen.txt"
    seen.write_text("alpha\n", encoding="utf-8")
    unseen.write_text("beta\n", encoding="utf-8")
    cell = cell_dispatch(task)

    def model(name, **args):
        return json.loads(handle_function_call(name, args, task_id=task))

    assert "alpha" in model("read_file", path=str(seen))["content"]
    # The model already has this file; the cell still needs the text, however often it asks.
    assert all("alpha" in _call(cell, "read_file", path=str(seen))["content"] for _ in range(5))
    assert all("beta" in _call(cell, "read_file", path=str(unseen))["content"] for _ in range(5))
    assert all("total_count" in _call(cell, "search_files", pattern="beta", path=str(tmp_path)) for _ in range(5))
    # Only the cell saw unseen.txt, so the model's own first read gets the text, not the stub...
    assert "beta" in model("read_file", path=str(unseen))["content"]
    # ...and the model's repeat of a file it was shown is still deduplicated.
    assert model("read_file", path=str(seen)).get("content_returned") is False


def test_cell_read_still_counts_as_a_read_before_overwriting(tmp_path):
    task = "cell-read-then-write"
    reset_file_dedup(task)
    target = tmp_path / "notes.txt"
    target.write_text("old\n", encoding="utf-8")
    cell = _default_dispatch(task)

    refused = _call(cell, "write_file", path=str(target), content="new\n")
    assert "error" in refused and target.read_text(encoding="utf-8") == "old\n"

    assert "old" in _call(cell, "read_file", path=str(target))["content"]
    written = _call(cell, "write_file", path=str(target), content="new\n")
    assert "error" not in written and target.read_text(encoding="utf-8") == "new\n"
