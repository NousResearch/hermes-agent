import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.verified_patch_core import (  # noqa: E402
    VerifiedPatchError,
    apply_operations,
    make_replace_operation,
)


def test_verified_patch_applies_fresh_snapshot():
    snapshot = 'header = "case"\nvalue = 3\ntail = 1\n'
    op = make_replace_operation(snapshot, 2, 2, ["value = 10"])

    assert apply_operations(snapshot, [op]) == 'header = "case"\nvalue = 10\ntail = 1\n'


def test_verified_patch_rejects_semantic_target_drift():
    snapshot = 'header = "case"\nvalue = 3\ntail = 1\n'
    actual = 'header = "case"\nvalue = 5\ntail = 1\n'
    op = make_replace_operation(snapshot, 2, 2, ["value = 10"])

    try:
        apply_operations(actual, [op])
    except VerifiedPatchError as e:
        assert "precondition" in str(e)
    else:
        raise AssertionError("semantic drift must reject")


def test_verified_patch_allows_unrelated_line_drift():
    snapshot = 'header = "case"\nvalue = 3\ntail = 1\n'
    actual = 'header = "changed"\nvalue = 3\ntail = 1\n'
    op = make_replace_operation(snapshot, 2, 2, ["value = 10"])

    assert apply_operations(actual, [op]) == 'header = "changed"\nvalue = 10\ntail = 1\n'


def test_verified_patch_allows_unique_layout_drift():
    snapshot = 'header = "case"\nconfigure(retries=3)\ntail = 1\n'
    actual = 'header = "case"\nconfigure( retries = 3 )\ntail = 1\n'
    op = make_replace_operation(snapshot, 2, 2, ["configure(retries=10)"])

    assert apply_operations(actual, [op]) == 'header = "case"\nconfigure(retries=10)\ntail = 1\n'


def test_verified_patch_rejects_semantic_whitespace_inside_string_literal():
    snapshot = 'msg = "a b"\n'
    actual = 'msg = "ab"\n'
    op = make_replace_operation(snapshot, 1, 1, ['msg = "new"'], context=0)

    try:
        apply_operations(actual, [op])
    except VerifiedPatchError as e:
        assert "precondition" in str(e)
    else:
        raise AssertionError("semantic whitespace drift must reject")


def test_verified_patch_uses_context_to_pick_duplicate_target():
    snapshot = (
        'scope = "first"\nflag = True\n'
        'scope = "second"\nflag = True\ntail = 1\n'
    )
    op = make_replace_operation(snapshot, 4, 4, ["flag = False"])

    assert apply_operations(snapshot, [op]) == (
        'scope = "first"\nflag = True\n'
        'scope = "second"\nflag = False\ntail = 1\n'
    )


def test_verified_patch_rejects_unresolved_duplicate_target():
    snapshot = "flag = True\nflag = True\n"
    op = make_replace_operation(snapshot, 1, 1, ["flag = False"], context=0)

    try:
        apply_operations(snapshot, [op])
    except VerifiedPatchError as e:
        assert "ambiguous" in str(e)
    else:
        raise AssertionError("ambiguous duplicate must reject")


def test_context_beats_exact_duplicate_when_intended_target_has_layout_drift():
    snapshot = "alpha\nvalue = 1\nomega\nnoise\nvalue = 1\n"
    actual = "alpha\nvalue=1\nomega\nnoise\nvalue = 1\n"
    op = make_replace_operation(snapshot, 2, 2, ["value = 10"], context=1)

    assert apply_operations(actual, [op]) == "alpha\nvalue = 10\nomega\nnoise\nvalue = 1\n"


def test_verified_patch_rejects_semantic_python_indentation_drift():
    snapshot = "if cond:\n    do_a()\ndo_b()\n"
    actual = "if cond:\n    do_a()\n    do_b()\n"
    op = make_replace_operation(snapshot, 3, 3, ["do_c()"], context=1)

    try:
        apply_operations(actual, [op])
    except VerifiedPatchError as e:
        assert "precondition" in str(e)
    else:
        raise AssertionError("semantic indentation drift must reject")
