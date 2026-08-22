"""Behavior contract for patch tool argument validation."""

import json
from unittest.mock import patch

import pytest

from tools.file_tools import _handle_patch


def _failure(args):
    with patch("tools.file_tools.patch_tool") as handler:
        result = json.loads(_handle_patch(args))
    handler.assert_not_called()
    assert result["success"] is False
    assert result["failure"]["kind"] == "malformed_input"
    assert result["failure"]["class"] == "patch_contract"
    assert result["failure"]["tool"] == "patch"
    assert "args" not in result["failure"]
    return result


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", None),
        ("path", 7),
        ("old_string", None),
        ("old_string", ["x"]),
        ("new_string", None),
        ("new_string", {"x": 1}),
    ],
)
def test_replace_rejects_missing_or_wrong_type_required_fields(field, value):
    args = {"mode": "replace", "path": "x.py", "old_string": "old", "new_string": "new"}
    if value is None:
        args.pop(field)
    else:
        args[field] = value

    result = _failure(args)

    assert result["failure"]["code"] == f"patch.replace.{field}.invalid"
    assert field in result["error"]


def test_replace_allows_empty_new_string_deletion():
    args = {"mode": "replace", "path": "x.py", "old_string": "old", "new_string": ""}
    with patch("tools.file_tools.patch_tool", return_value='{"ok":true}') as handler:
        assert json.loads(_handle_patch(args)) == {"ok": True}
    handler.assert_called_once()


def test_replace_rejects_identical_old_and_new_strings():
    result = _failure(
        {"mode": "replace", "path": "x.py", "old_string": "same", "new_string": "same"}
    )
    assert result["failure"]["code"] == "patch.replace.no_change"


@pytest.mark.parametrize("patch_payload", ["diff", ""])
def test_replace_rejects_incompatible_patch_payload(patch_payload):
    result = _failure(
        {
            "mode": "replace",
            "path": "x.py",
            "old_string": "old",
            "new_string": "new",
            "patch": patch_payload,
        }
    )
    assert result["failure"]["code"] == "patch.replace.incompatible_fields"


@pytest.mark.parametrize("patch_text", [None, "", "   ", 42])
def test_patch_mode_requires_nonempty_patch_text(patch_text):
    args = {"mode": "patch"}
    if patch_text is not None:
        args["patch"] = patch_text
    result = _failure(args)
    assert result["failure"]["code"] == "patch.patch.patch.invalid"


@pytest.mark.parametrize("field,value", [("path", "x.py"), ("old_string", "old"), ("new_string", "new"), ("replace_all", True)])
def test_patch_mode_rejects_ambiguous_replace_fields(field, value):
    result = _failure({"mode": "patch", "patch": "*** Begin Patch\n*** End Patch", field: value})
    assert result["failure"]["code"] == "patch.patch.incompatible_fields"


@pytest.mark.parametrize("mode", ["", "merge", 7])
def test_unknown_or_invalid_mode_is_rejected(mode):
    args = {"mode": mode}
    result = _failure(args)
    assert result["failure"]["code"] == "patch.mode.invalid"


def test_validation_precedes_path_resolution_approval_and_mutation():
    args = {"mode": "replace", "path": 7, "old_string": "old", "new_string": "new"}
    with (
        patch("tools.file_tools._resolve_path_for_task") as resolve,
        patch("tools.file_tools._check_approval_required_write") as approval,
        patch("tools.file_tools._get_file_ops") as mutation,
    ):
        result = json.loads(_handle_patch(args))
    assert result["failure"]["kind"] == "malformed_input"
    resolve.assert_not_called()
    approval.assert_not_called()
    mutation.assert_not_called()
