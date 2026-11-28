"""Spec-open hinted safety of the production V4A planner.

The targeting contract does not fix hint-window boundaries. These tests
call ``apply_v4a_operations`` and assert only properties that hold
whichever boundary an implementation uses:

- a refusal writes nothing and leaves every original byte unchanged
- a write never lands in a file the patch did not name
- diagnostics do not quote source lines or hint text

They do not assert which site a ``value = 1`` or ``value<TAB>=<TAB>1``
hint selects.
"""

import pytest

from tools.file_operations_common import ReadResult, WriteResult
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

SPACE_VALUE = "value = 1"
TAB_VALUE = "value\t=\t1"
# Distances on both sides of any fixed before/after window. Assertions do
# not branch on these values.
GAPS = (0, 40, 600, 3000)
HINTS = (SPACE_VALUE, TAB_VALUE)


class MemoryFiles:
    def __init__(self, files):
        self.files = dict(files)
        self.writes = []

    def read_file_raw(self, path):
        if path not in self.files:
            return ReadResult(error="file not found", not_found=True)
        text = self.files[path]
        return ReadResult(
            content=text,
            total_lines=text.count("\n") + 1,
            file_size=len(text.encode()),
        )

    def write_file(self, path, content, pre_content=None):
        self.writes.append((path, content))
        self.files[path] = content
        return WriteResult(bytes_written=len(content.encode()), verified=True)


def _patch(parts):
    chunks = ["*** Begin Patch"]
    for path, body in parts:
        chunks.append(f"*** Update File: {path}")
        chunks.append(body)
    chunks.append("*** End Patch")
    return "\n".join(chunks) + "\n"


def _apply(files, parts):
    ops, error = parse_v4a_patch(_patch(parts))
    assert not error, error
    store = MemoryFiles(files)
    return apply_v4a_operations(ops, store), store, dict(files)


def _forms(gap, token):
    """Exact ``value = 1`` and tab-normalized ``value<TAB>=<TAB>1``, gap apart."""
    return f"{SPACE_VALUE}\n" + ("x" * gap) + f"\n{token}\n{TAB_VALUE}\n"


def _secrets(*extra):
    return (
        SPACE_VALUE,
        TAB_VALUE,
        "value = 10",
        "OPAQUE_GAP_7X4Y",
        "OPAQUE_FOOTER_7X4Y",
        "OPAQUE_OTHER_7X4Y",
        "OPAQUE_SRC_7X4Y",
        "OPAQUE_SECRET_7X4Y",
        *extra,
    )


def _diagnostics(result):
    return "\n".join(part for part in (result.error, result.note, result.diff) if part)


def _assert_no_excerpts(result, secrets):
    diagnostic = _diagnostics(result)
    for secret in secrets:
        assert secret not in diagnostic, secret


def _assert_safety(result, store, before, targets, secrets):
    written = [path for path, _content in store.writes]
    assert set(written) <= set(targets)
    reported = (
        result.files_modified + result.files_created + result.files_deleted
    )
    assert set(reported) <= set(targets)
    for path, original in before.items():
        if path not in targets:
            assert path not in written
            assert store.files[path] == original
    if result.note:
        for secret in secrets:
            assert secret not in result.note
    if result.success:
        for path, original in before.items():
            if path not in written:
                assert store.files[path] == original
        return
    assert not store.writes
    assert store.files == before
    assert not reported
    _assert_no_excerpts(result, secrets)


@pytest.mark.parametrize("gap", GAPS)
def test_spec_open_double_space_search_with_tab_prefix_hint_is_atomic(gap):
    """Do not fix a site for the boundary-dependent normalized example."""
    files = {"a.py": _forms(gap, "OPAQUE_GAP_7X4Y"), "b.py": "untouched\n"}
    result, store, before = _apply(
        files,
        (("a.py", "@@ value\t= @@\n-value  =  1\n+CHANGED\n@@\n-missing\n+new"),),
    )
    assert not result.success
    _assert_safety(result, store, before, ("a.py",), _secrets("value\t="))


@pytest.mark.parametrize("hint", HINTS, ids=["space-hint", "tab-hint"])
@pytest.mark.parametrize("gap", GAPS)
def test_failing_other_file_is_atomic_for_space_or_tab_hint(hint, gap):
    """A later file that cannot match must not leave the hinted file written."""
    files = {
        "a.py": _forms(gap, "OPAQUE_GAP_7X4Y") + "footer = OPAQUE_FOOTER_7X4Y\n",
        "b.py": "keep = OPAQUE_OTHER_7X4Y\n",
    }
    result, store, before = _apply(
        files,
        (
            ("a.py", f"@@ {hint} @@\n-{SPACE_VALUE}\n+CHANGED"),
            ("b.py", "@@\n-missing = old\n+missing = new"),
        ),
    )
    assert not result.success
    _assert_safety(result, store, before, ("a.py", "b.py"), _secrets(hint))


@pytest.mark.parametrize("hint", HINTS, ids=["space-hint", "tab-hint"])
@pytest.mark.parametrize("gap", GAPS)
def test_later_miss_is_atomic_at_any_hint_distance(hint, gap):
    """A missing later hunk refuses the whole plan, at every hint distance."""
    source = _forms(gap, "OPAQUE_GAP_7X4Y") + "footer = OPAQUE_FOOTER_7X4Y\n"
    result, store, before = _apply(
        {"a.py": source, "b.py": "keep = OPAQUE_OTHER_7X4Y\n"},
        (
            (
                "a.py",
                f"@@ {hint} @@\n-{SPACE_VALUE}\n+CHANGED\n"
                "@@\n-missing = old\n+missing = new",
            ),
        ),
    )
    assert not result.success
    _assert_safety(result, store, before, ("a.py",), _secrets(hint))


@pytest.mark.parametrize("hint", HINTS, ids=["space-hint", "tab-hint"])
def test_hint_text_in_another_file_does_not_write_that_file(hint):
    """A hint that also occurs in a file the patch does not name stays there."""
    files = {
        "a.py": f"keep = OPAQUE_SRC_7X4Y\n{SPACE_VALUE}\n",
        "b.py": f"{TAB_VALUE}\nkeep = OPAQUE_OTHER_7X4Y\n{SPACE_VALUE}\n",
    }
    result, store, before = _apply(
        files,
        (("a.py", f"@@ {hint} @@\n-{SPACE_VALUE}\n+CHANGED"),),
    )
    _assert_safety(result, store, before, ("a.py",), _secrets(hint))


@pytest.mark.parametrize("hint", HINTS, ids=["space-hint", "tab-hint"])
@pytest.mark.parametrize("gap", GAPS)
def test_space_or_tab_hint_does_not_write_bystander(hint, gap):
    """Either representation may be the hint; the other file is never written."""
    files = {
        "a.py": "value = 10\n" + _forms(gap, "OPAQUE_GAP_7X4Y"),
        "b.py": f"{hint}\nkeep = OPAQUE_OTHER_7X4Y\n{TAB_VALUE}\n",
    }
    result, store, before = _apply(
        files,
        (("a.py", f"@@ {hint} @@\n-{SPACE_VALUE}\n+CHANGED"),),
    )
    _assert_safety(result, store, before, ("a.py",), _secrets(hint))


@pytest.mark.parametrize("hint", HINTS, ids=["space-hint", "tab-hint"])
@pytest.mark.parametrize("gap", GAPS)
def test_hinted_miss_does_not_excerpt_nearby_source(hint, gap):
    """A hinted miss quotes neither the hint nor source at any distance."""
    source = (
        f"{hint}\n"
        + ("q" * gap)
        + '\npassword = "OPAQUE_SECRET_7X4Y"\n'
        f"{SPACE_VALUE}\n{TAB_VALUE}\n"
    )
    result, store, before = _apply(
        {"a.py": source, "b.py": "keep = OPAQUE_OTHER_7X4Y\n"},
        (("a.py", f'@@ {hint} @@\n-password = "different"\n+changed'),),
    )
    assert not result.success
    _assert_safety(
        result,
        store,
        before,
        ("a.py",),
        _secrets(hint, 'password = "OPAQUE_SECRET_7X4Y"', "different"),
    )
