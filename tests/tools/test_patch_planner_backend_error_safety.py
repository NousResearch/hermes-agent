"""Backend errors must not disclose arbitrary source text or credentials."""
from types import SimpleNamespace

import pytest

from tools.patch_parser import apply_v4a_operations, parse_v4a_patch


def test_identical_missing_search_is_not_success():
    marker = "*" * 3
    patch = (f"{marker} Begin Patch\n{marker} Update File: f\n"
             f"@@\n-missing\n+missing\n{marker} End Patch\n")
    operations, error = parse_v4a_patch(patch)
    assert error is None
    from tools.v4a_planner import plan_hunks

    result = plan_hunks("unrelated\n", operations[0].hunks)
    assert result.errors


def test_escape_drift_exception_text_is_never_exposed(monkeypatch):
    from tools import escape_drift_autocorrect as drift
    from tools.v4a_planner import plan_hunks

    token = "SYNTHETIC_DRIFT_SECRET_9XYZ"

    def fail(*args):
        raise drift.AmbiguousEscapeDriftError(token)

    monkeypatch.setattr(drift, "maybe_correct_escape_drift", fail)
    marker = "*" * 3
    patch = f"{marker} Begin Patch\n{marker} Update File: f\n@@\n-old\n+new\n{marker} End Patch\n"
    operations, error = parse_v4a_patch(patch)
    assert error is None
    result = plan_hunks("old\n", operations[0].hunks)
    assert result.errors and token not in repr(result)


def test_resolved_hunk_then_consumed_search_refuses_without_writes():
    class Store:
        def __init__(self):
            self.files = {"f": "total_count = 0\ntotal_count2 = 0\n"}
            self.writes = []

        def read_file_raw(self, path):
            return SimpleNamespace(error=None, content=self.files[path])

        def write_file(self, path, content, pre_content=None):
            self.writes.append((path, content))
            self.files[path] = content
            return SimpleNamespace(error=None)

    marker = "*" * 3
    start = f"{marker} Begin Patch\n{marker} Update File: f\n"
    first = "@@\n-total_count = 0\n+grand_sum = 42\n"
    second = "@@\n-total_count = 0\n+total_count = 5\n"
    end = f"{marker} End Patch\n"
    ops, error = parse_v4a_patch(start + first + end)
    assert error is None
    control = Store()
    assert apply_v4a_operations(ops, control).success
    assert control.files["f"] == "grand_sum = 42\ntotal_count2 = 0\n"
    ops, error = parse_v4a_patch(start + first + second + end)
    assert error is None
    store = Store()
    assert not apply_v4a_operations(ops, store).success
    assert store.writes == [] and store.files["f"] == "total_count = 0\ntotal_count2 = 0\n"


@pytest.mark.parametrize("transition", ["same", "alias", "move"])
def test_separate_updates_cannot_reedit_inserted_bytes(transition):
    class Store:
        def __init__(self):
            self.files = {"f": "old\n"}
            self.writes = []

        def read_file_raw(self, path):
            if path not in self.files:
                return SimpleNamespace(error="missing", not_found=True)
            return SimpleNamespace(error=None, content=self.files[path])

        def write_file(self, path, content, pre_content=None):
            self.writes.append((path, content))
            self.files[path] = content
            return SimpleNamespace(error=None)

    marker = "*" * 3
    second_path = {"same": "f", "alias": "./f", "move": "g"}[transition]
    move = f"{marker} Move File: f -> g\n" if transition == "move" else ""
    patch = (f"{marker} Begin Patch\n{marker} Update File: f\n@@\n-old\n+new\n"
             f"{move}{marker} Update File: {second_path}\n@@\n-new\n+BAD\n{marker} End Patch\n")
    operations, error = parse_v4a_patch(patch)
    assert error is None
    store = Store()
    from tools.patch_parser import _resolve_operations

    control = _resolve_operations(operations[:-1], store)
    assert control.errors == []
    result = apply_v4a_operations(operations, store)
    assert not result.success
    assert store.writes == [] and store.files == {"f": "old\n"}


@pytest.mark.parametrize("stage", ["read", "write"])
def test_backend_error_is_not_echoed(stage):
    secret = "synthetic-token-XYZ987"

    class Store:
        def read_file_raw(self, path):
            if stage == "read":
                return SimpleNamespace(error=secret, content=None)
            return SimpleNamespace(error=None, content="old\n")

        def write_file(self, path, content, pre_content=None):
            return SimpleNamespace(error=secret)

    patch = "*** Begin Patch\n*** Update File: f\n@@\n-old\n+new\n*** End Patch\n"
    operations, error = parse_v4a_patch(patch)
    assert error is None
    result = apply_v4a_operations(operations, Store())
    assert not result.success
    assert secret not in (result.error or "")
