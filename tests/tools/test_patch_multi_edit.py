"""Ordered, atomic one-file MultiEdit contracts (salvage of #76413)."""

import json
from unittest.mock import Mock

import pytest

import tools.file_tools as ft
from acp_adapter.edit_approval import build_edit_proposal
from tools.code_execution_tool import generate_hermes_tools_module
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


EDITS = [
    {"old_string": "alpha", "new_string": "intermediate"},
    {"old_string": "intermediate", "new_string": "omega"},
]


@pytest.fixture
def target(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"alpha\nbravo\n")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path), timeout=15), cwd=str(tmp_path))
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id: ops)
    return path, ops


def call_patch(path, **kwargs):
    return json.loads(registry.dispatch("patch", {"path": str(path), **kwargs}, task_id=str(path)))


def test_ordered_round_trip_has_one_write_and_diff(target, monkeypatch):
    path, ops = target
    writer = Mock(wraps=ops.write_file)
    monkeypatch.setattr(ops, "write_file", writer)
    result = json.loads(ft.patch_tool(path=str(path), edits=EDITS, task_id=str(path)))
    assert not result.get("error"), result
    assert path.read_bytes() == b"omega\nbravo\n"
    assert writer.call_count == 1
    assert result["diff"].count("--- ") == 1
    assert "-alpha" in result["diff"] and "+omega" in result["diff"]
    assert result["files_modified"] == [str(path.resolve())]
    assert result["resolved_path"] == str(path.resolve())


def test_later_failure_is_atomic_and_named(target, monkeypatch):
    path, ops = target
    before = path.read_bytes()
    writer = Mock(wraps=ops.write_file)
    monkeypatch.setattr(ops, "write_file", writer)
    result = call_patch(path, edits=[EDITS[0], {"old_string": "missing anchor", "new_string": "no"}])
    assert result["error"].startswith("edit 2 of 2:"), result
    assert "Could not find" in result["error"]
    assert path.read_bytes() == before
    writer.assert_not_called()


@pytest.mark.parametrize("edits", [
    [], "bad", [None], [{"old_string": 1, "new_string": "x"}],
    [{"old_string": "alpha", "new_string": None}],
    [{"old_string": "alpha"}],
    [{"old_string": "alpha", "new_string": "x", "replace_all": 1}],
    [{"old_string": "alpha", "new_string": "alpha"}],
    [EDITS[0]] * 51,
])
def test_invalid_edits_do_not_write(target, edits):
    path, _ = target
    before = path.read_bytes()
    result = call_patch(path, edits=edits)
    assert result.get("error"), result
    assert path.read_bytes() == before


@pytest.mark.parametrize("extra", [
    {"old_string": "alpha", "new_string": "x"},
    {"old_string": "alpha"}, {"new_string": "x"},
])
def test_shapes_are_mutually_exclusive(target, extra):
    path, _ = target
    before = path.read_bytes()
    result = call_patch(path, edits=EDITS, **extra)
    assert "one or the other" in result["error"], result
    assert path.read_bytes() == before


@pytest.mark.parametrize("prefix,ending", [(b"", b"\r\n"), (b"\xef\xbb\xbf", b"\n")])
def test_replace_all_delete_and_preserve_encoding(target, prefix, ending):
    path, _ = target
    path.write_bytes(prefix + ending.join([b"alpha", b"alpha", b"bravo", b""]))
    result = call_patch(path, edits=[
        {"old_string": "alpha", "new_string": "omega", "replace_all": True},
        {"old_string": "bravo", "new_string": ""},
    ])
    assert not result.get("error"), result
    assert path.read_bytes() == prefix + ending.join([b"omega", b"omega", b"", b""])


def test_full_read_baseline_carries_through_multi_edit(target):
    path, _ = target
    task_id = str(path)
    assert not json.loads(ft.read_file_tool(str(path), task_id=task_id)).get("error")
    result = call_patch(path, edits=EDITS)
    assert not result.get("error"), result
    result = json.loads(ft.write_file_tool(str(path), "whole replacement\n", task_id=task_id))
    assert not result.get("error"), result
    assert path.read_text() == "whole replacement\n"


@pytest.mark.parametrize("provider,model,required", [
    ("anthropic", "claude-sonnet-4", ["path"]),
    ("openai", "gpt-5.2", ["mode"]),
])
def test_served_schemas_include_edits_without_unions(monkeypatch, provider, model, required):
    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: provider)
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: model)
    monkeypatch.setattr("tools.registry._memo_check", lambda *args: True)
    schema = registry.get_definitions({"patch"})[0]["function"]
    assert "edits" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == required
    assert "anyOf" not in json.dumps(schema) and "oneOf" not in json.dumps(schema)


def test_acp_proposes_ordered_result_without_writing(target):
    path, _ = target
    before = path.read_bytes()
    proposal = build_edit_proposal("patch", {"path": str(path), "edits": EDITS})
    assert proposal.old_text == "alpha\nbravo\n"
    assert proposal.new_text == "omega\nbravo\n"
    assert path.read_bytes() == before


def test_acp_failure_names_edit_and_does_not_write(target):
    path, _ = target
    before = path.read_bytes()
    with pytest.raises(ValueError, match="edit 2 of 2:"):
        build_edit_proposal("patch", {"path": str(path), "edits": [
            EDITS[0], {"old_string": "missing anchor", "new_string": "no"},
        ]})
    assert path.read_bytes() == before


@pytest.mark.parametrize("transport", ["uds", "file"])
def test_execute_code_stub_forwards_edits(target, transport):
    path, _ = target
    namespace = {}
    exec(generate_hermes_tools_module(["patch"], transport=transport), namespace)
    calls = []

    def dispatch(name, args):
        calls.append((name, args))
        return json.loads(registry.dispatch(name, args, task_id=str(path)))

    namespace["_call"] = dispatch
    result = namespace["patch"](path=str(path), edits=EDITS)
    assert calls[0][0] == "patch" and calls[0][1]["edits"] == EDITS
    assert not result.get("error"), result
    assert path.read_bytes() == b"omega\nbravo\n"


def test_unknown_mode_still_fails(target):
    path, _ = target
    result = call_patch(path, mode="multi_edit", edits=EDITS)
    assert "Unknown mode: multi_edit" in result["error"]
    assert build_edit_proposal("patch", {"path": str(path), "mode": "multi_edit", "edits": EDITS}) is None
    assert path.read_bytes() == b"alpha\nbravo\n"


def test_resent_batch_fails_without_writing(target, monkeypatch):
    path, ops = target
    edits = [{"old_string": "alpha", "new_string": "omega first"},
             {"old_string": "bravo", "new_string": "omega second"}]
    assert not call_patch(path, edits=edits).get("error")
    after = path.read_bytes()
    writer = Mock(wraps=ops.write_file)
    monkeypatch.setattr(ops, "write_file", writer)
    result = call_patch(path, edits=edits)
    assert result["error"].startswith("edit 1 of 2:"), result
    assert path.read_bytes() == after
    writer.assert_not_called()


@pytest.mark.parametrize("content,replace_all", [
    (b"first completed\nsetting  = 0\nsetting = 1\n", True),  # edit 2 still fuzzy-matches
    (b"first completed\nsetting  = 0\nsetting  = 0\nsetting = 1\n", False),  # edit 2 is ambiguous
])
def test_batch_with_a_pending_edit_is_never_reported_applied(target, content, replace_all):
    path, _ = target
    path.write_bytes(content)
    result = call_patch(path, edits=[
        {"old_string": "zzzzzzzz", "new_string": "first completed"},
        {"old_string": "setting = 0", "new_string": "setting = 1", "replace_all": replace_all},
    ])
    assert result["error"].startswith("edit 1 of 2:"), result
    assert path.read_bytes() == content


@pytest.mark.parametrize("blank", [
    {"old_string": "", "new_string": ""}, {"old_string": ""}, {"new_string": ""},
])
def test_blank_single_edit_fields_count_as_absent(target, blank):
    path, _ = target
    proposal = build_edit_proposal("patch", {"path": str(path), "edits": EDITS, **blank})
    assert proposal.new_text == "omega\nbravo\n"
    result = call_patch(path, edits=EDITS, **blank)
    assert not result.get("error"), result
    assert path.read_bytes() == b"omega\nbravo\n"
