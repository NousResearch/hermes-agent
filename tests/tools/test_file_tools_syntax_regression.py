import json
import os
import shutil
import tempfile

import pytest


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home

    from tools.file_tools import clear_file_ops_cache, _patch_failure_lock, _patch_failure_tracker
    from tools.file_tools import _read_tracker_lock, _read_tracker
    from tools.terminal_tool import _active_environments, _env_lock

    clear_file_ops_cache()
    with _read_tracker_lock:
        _read_tracker.clear()
    with _patch_failure_lock:
        _patch_failure_tracker.clear()
    with _env_lock:
        _active_environments.clear()


@pytest.fixture
def safe_temp_path():
    base = tempfile.mkdtemp(prefix="hermes-file-tools-regression-", dir="/tmp")
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_write_file_invalid_json_is_reported_as_failure(isolated_hermes_home, safe_temp_path):
    from tools.file_tools import write_file_tool

    target = os.path.join(safe_temp_path, "bad.json")
    result = json.loads(write_file_tool(str(target), '{"a":'))

    assert result.get("error"), result
    assert "files_modified" not in result, result


def test_patch_tool_replace_invalidates_file_with_syntax_regression(
    isolated_hermes_home, safe_temp_path
):
    from tools.file_tools import patch_tool

    target = os.path.join(safe_temp_path, "valid.json")
    with open(target, "w", encoding="utf-8") as fp:
        fp.write('{"a": 1}')

    result = json.loads(
        patch_tool(
            mode="replace",
            path=str(target),
            old_string='"a": 1',
            new_string='"a":',
        )
    )

    assert result.get("error"), result
    assert "files_modified" not in result, result


def test_write_file_valid_json_still_reports_success(isolated_hermes_home, safe_temp_path):
    from tools.file_tools import write_file_tool

    target = os.path.join(safe_temp_path, "good.json")
    resolved_target = os.path.realpath(target)
    with open(target, "w", encoding="utf-8") as fp:
        fp.write('{"a": 1}')

    result = json.loads(write_file_tool(str(target), '{"a": 1, "b": 2}'))

    assert not result.get("error"), result
    assert result.get("files_modified") == [resolved_target], result


def test_pre_existing_syntax_error_is_not_recast_as_new_regression(
    isolated_hermes_home, safe_temp_path
):
    from tools.file_tools import write_file_tool

    target = os.path.join(safe_temp_path, "broken.json")
    payload = '{"a":\n'
    resolved_target = os.path.realpath(target)
    with open(target, "w", encoding="utf-8") as fp:
        fp.write(payload)

    result = json.loads(write_file_tool(str(target), payload))

    assert not result.get("error"), result
    assert result.get("files_modified") == [resolved_target], result
