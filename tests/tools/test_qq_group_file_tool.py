"""Tests for tools/qq_group_file_tool.py."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.qq_group_file_tool import qq_group_file_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_qq_napcat_config(home_chat_id=None):
    platform = getattr(Platform, "QQ_NAPCAT")
    qq_cfg = SimpleNamespace(
        enabled=True,
        token=None,
        api_key=None,
        extra={"ws_url": "ws://127.0.0.1:3001"},
    )
    home = SimpleNamespace(chat_id=home_chat_id) if home_chat_id else None
    return SimpleNamespace(
        platforms={platform: qq_cfg},
        get_home_channel=lambda _platform: home,
    ), qq_cfg


def test_list_group_root_files_uses_root_listing_api():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"files": [{"file_id": "fid-1"}], "folders": []}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "list",
                    "target": "group:987654321",
                }
            )
        )

    assert result["success"] is True
    assert result["group_id"] == "987654321"
    assert result["files"] == [{"file_id": "fid-1"}]
    assert result["folders"] == []
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_list_group_files_by_folder_uses_folder_listing_api():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"files": [], "folders": [{"folder_id": "/docs"}]}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "list",
                    "target": "qq_napcat:group:987654321",
                    "folder_id": "/docs",
                }
            )
        )

    assert result["success"] is True
    assert result["folder_id"] == "/docs"
    assert result["folders"] == [{"folder_id": "/docs"}]
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_files_by_folder",
        {"group_id": 987654321, "folder_id": "/docs"},
    )


def test_list_group_root_files_treats_slash_folder_id_as_root():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"files": [], "folders": []}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "list",
                    "target": "group:987654321",
                    "folder_id": "/",
                }
            )
        )

    assert result["success"] is True
    assert result["folder_id"] == "/"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_upload_group_file_uses_absolute_path_and_default_name(tmp_path):
    config, qq_cfg = _make_qq_napcat_config()
    upload_path = tmp_path / "测试 文档.txt"
    upload_path.write_text("hello")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"file_id": "fid-1"}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "upload",
                    "target": "group:987654321",
                    "file_path": str(upload_path),
                    "folder_id": "/docs",
                }
            )
        )

    assert result["success"] is True
    assert result["uploaded"] is True
    assert result["file_name"] == "测试 文档.txt"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "upload_group_file",
        {
            "group_id": 987654321,
            "file": str(upload_path.resolve()),
            "name": "测试 文档.txt",
            "folder": "/docs",
        },
    )


def test_upload_group_file_treats_slash_folder_id_as_root(tmp_path):
    config, qq_cfg = _make_qq_napcat_config()
    upload_path = tmp_path / "root.txt"
    upload_path.write_text("hello")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"file_id": "fid-root"}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "upload",
                    "target": "group:987654321",
                    "file_path": str(upload_path),
                    "folder_id": "/",
                }
            )
        )

    assert result["success"] is True
    assert result["folder_id"] == "/"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "upload_group_file",
        {
            "group_id": 987654321,
            "file": str(upload_path.resolve()),
            "name": "root.txt",
        },
    )


def test_delete_group_file_uses_file_id_and_busid():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "delete",
                    "target": "group:987654321",
                    "file_id": "fid-1",
                    "busid": 102,
                }
            )
        )

    assert result["success"] is True
    assert result["deleted"] is True
    assert result["file_id"] == "fid-1"
    assert result["busid"] == 102
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "delete_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "busid": 102},
    )


def test_create_group_folder_uses_root_parent_id():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"folder_id": "/docs"}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "create_folder",
                    "target": "group:987654321",
                    "folder_name": "文档",
                }
            )
        )

    assert result["success"] is True
    assert result["created"] is True
    assert result["folder_name"] == "文档"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "create_group_file_folder",
        {"group_id": 987654321, "name": "文档", "parent_id": "/"},
    )


def test_rejects_non_root_parent_for_group_folder_creation():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "create_folder",
                    "target": "group:987654321",
                    "folder_name": "文档",
                    "parent_id": "/nested",
                }
            )
        )

    assert "root" in result["error"].lower()
    assert "/" in result["error"]


def test_delete_group_folder_uses_folder_id():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "delete_folder",
                    "target": "group:987654321",
                    "folder_id": "/docs",
                }
            )
        )

    assert result["success"] is True
    assert result["deleted"] is True
    assert result["folder_id"] == "/docs"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "delete_group_folder",
        {"group_id": 987654321, "folder_id": "/docs"},
    )


def test_get_group_file_system_info_returns_payload():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"total_space": 123, "used_space": 45}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "system_info",
                    "target": "group:987654321",
                }
            )
        )

    assert result["success"] is True
    assert result["group_id"] == "987654321"
    assert result["raw_response"] == {"total_space": 123, "used_space": 45}
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_file_system_info",
        {"group_id": 987654321},
    )


def test_get_group_file_url_requires_file_identity():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"url": "https://files.example.com/fid-1"}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "get_url",
                    "target": "group:987654321",
                    "file_id": "fid-1",
                    "busid": 102,
                }
            )
        )

    assert result["success"] is True
    assert result["url"] == "https://files.example.com/fid-1"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_file_url",
        {"group_id": 987654321, "file_id": "fid-1", "busid": 102},
    )


def test_move_group_file_uses_target_dir():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "move",
                    "target": "group:987654321",
                    "file_id": "fid-1",
                    "target_dir": "/docs",
                }
            )
        )

    assert result["success"] is True
    assert result["moved"] is True
    assert result["target_dir"] == "/docs"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "move_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "target_dir": "/docs"},
    )


def test_rename_group_file_requires_current_parent_directory():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "rename",
                    "target": "group:987654321",
                    "file_id": "fid-1",
                    "current_parent_directory": "/",
                    "new_name": "新文件.pdf",
                }
            )
        )

    assert result["success"] is True
    assert result["renamed"] is True
    assert result["new_name"] == "新文件.pdf"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "rename_group_file",
        {
            "group_id": 987654321,
            "file_id": "fid-1",
            "current_parent_directory": "/",
            "new_name": "新文件.pdf",
        },
    )


def test_forward_group_file_uses_target_group_id():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=({}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "forward",
                    "target": "group:987654321",
                    "file_id": "fid-1",
                    "target_group_id": "group:123456",
                }
            )
        )

    assert result["success"] is True
    assert result["forwarded"] is True
    assert result["target_group_id"] == "123456"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "trans_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "target_group_id": 123456},
    )


def test_find_group_files_recursively_returns_nested_file_matches():
    config, qq_cfg = _make_qq_napcat_config()

    responses = [
        (
            {
                "files": [{"file_id": "fid-root", "file_name": "预算.xlsx", "busid": 101}],
                "folders": [{"folder_id": "/docs", "folder_name": "文档"}],
            },
            None,
        ),
        (
            {
                "files": [{"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102}],
                "folders": [{"folder_id": "/docs/archive", "folder_name": "归档"}],
            },
            None,
        ),
        (
            {
                "files": [{"file_id": "fid-2", "file_name": "周报-最终.pdf", "busid": 103}],
                "folders": [],
            },
            None,
        ),
    ]

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(side_effect=responses),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "find",
                    "target": "group:987654321",
                    "query": "周报",
                }
            )
        )

    assert result["success"] is True
    assert result["match_count"] == 2
    assert result["searched_folder_ids"] == ["/", "/docs", "/docs/archive"]
    assert [item["file_id"] for item in result["matches"]] == ["fid-1", "fid-2"]
    assert result["matches"][0]["current_parent_directory"] == "/docs"
    assert result["matches"][1]["current_parent_directory"] == "/docs/archive"
    assert call_mock.await_args_list[0].args == (
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )
    assert call_mock.await_args_list[1].args == (
        qq_cfg.extra,
        "get_group_files_by_folder",
        {"group_id": 987654321, "folder_id": "/docs"},
    )
    assert call_mock.await_args_list[2].args == (
        qq_cfg.extra,
        "get_group_files_by_folder",
        {"group_id": 987654321, "folder_id": "/docs/archive"},
    )


def test_find_group_files_can_include_folder_matches_without_recursing():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [],
                     "folders": [{"folder_id": "/docs", "folder_name": "文档资料"}],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "find",
                    "target": "group:987654321",
                    "query": "文档",
                    "include_folders": True,
                    "recursive": False,
                }
            )
        )

    assert result["success"] is True
    assert result["match_count"] == 1
    assert result["matches"][0]["entry_type"] == "folder"
    assert result["matches"][0]["folder_id"] == "/docs"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_find_group_files_parses_string_false_booleans_correctly():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [],
                     "folders": [{"folder_id": "/docs", "folder_name": "文档资料"}],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "find",
                    "target": "group:987654321",
                    "query": "文档",
                    "include_folders": "false",
                    "recursive": "false",
                    "exact": "false",
                }
            )
        )

    assert result["success"] is True
    assert result["include_folders"] is False
    assert result["recursive"] is False
    assert result["exact"] is False
    assert result["match_count"] == 0
    assert result["matches"] == []
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_find_group_files_does_not_mark_truncated_when_matches_equal_limit():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [{"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102}],
                     "folders": [],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "find",
                    "target": "group:987654321",
                    "query": "周报",
                    "max_results": 1,
                    "recursive": False,
                }
            )
        )

    assert result["success"] is True
    assert result["match_count"] == 1
    assert result["truncated"] is False
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_find_group_files_marks_truncated_only_when_more_matches_exist():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [
                         {"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102},
                         {"file_id": "fid-2", "file_name": "周报-最终.pdf", "busid": 103},
                     ],
                     "folders": [],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "find",
                    "target": "group:987654321",
                    "query": "周报",
                    "max_results": 1,
                    "recursive": False,
                }
            )
        )

    assert result["success"] is True
    assert result["match_count"] == 1
    assert result["truncated"] is True
    assert [item["file_id"] for item in result["matches"]] == ["fid-1"]
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_resolve_group_file_returns_unique_match():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [{"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102}],
                     "folders": [],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "resolve",
                    "target": "group:987654321",
                    "query": "周报",
                }
            )
        )

    assert result["success"] is True
    assert result["match_count"] == 1
    assert result["match"]["file_id"] == "fid-1"
    assert result["match"]["current_parent_directory"] == "/"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_resolve_group_file_rejects_ambiguous_matches():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [
                         {"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102},
                         {"file_id": "fid-2", "file_name": "周报-最终.pdf", "busid": 103},
                     ],
                     "folders": [],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "resolve",
                    "target": "group:987654321",
                    "query": "周报",
                }
            )
        )

    assert "multiple" in result["error"].lower()
    assert result["match_count"] == 2
    assert [item["file_id"] for item in result["matches"]] == ["fid-1", "fid-2"]
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_resolve_group_file_mixed_ambiguity_mentions_find_with_include_folders():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [{"file_id": "fid-1", "file_name": "文档.txt", "busid": 102}],
                     "folders": [{"folder_id": "/docs", "folder_name": "文档"}],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "resolve",
                    "target": "group:987654321",
                    "query": "文档",
                    "recursive": False,
                }
            )
        )

    assert "include_folders=true" in result["error"]
    assert result["match_count"] == 2
    assert [item["entry_type"] for item in result["matches"]] == ["file", "folder"]
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_resolve_group_file_can_return_unique_folder_match():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [],
                     "folders": [{"folder_id": "/docs", "folder_name": "文档"}],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "resolve",
                    "target": "group:987654321",
                    "query": "文档",
                    "recursive": False,
                }
            )
        )

    assert result["success"] is True
    assert result["match"]["entry_type"] == "folder"
    assert result["match"]["folder_id"] == "/docs"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_resolve_folder_returns_unique_folder_match():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [],
                     "folders": [{"folder_id": "/docs", "folder_name": "文档"}],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "resolve_folder",
                    "target": "group:987654321",
                    "query": "文档",
                    "recursive": False,
                }
            )
        )

    assert result["success"] is True
    assert result["match_count"] == 1
    assert result["match"]["entry_type"] == "folder"
    assert result["match"]["folder_id"] == "/docs"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_resolve_folder_rejects_file_match():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [{"file_id": "fid-1", "file_name": "文档", "busid": 102}],
                     "folders": [],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "resolve_folder",
                    "target": "group:987654321",
                    "query": "文档",
                    "recursive": False,
                }
            )
        )

    assert "folder match" in result["error"].lower()
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_resolve_folder_ignores_same_name_file_when_folder_is_unique():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [{"file_id": "fid-1", "file_name": "文档.txt", "busid": 102}],
                     "folders": [{"folder_id": "/docs", "folder_name": "文档"}],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "resolve_folder",
                    "target": "group:987654321",
                    "query": "文档",
                    "recursive": False,
                }
            )
        )

    assert result["success"] is True
    assert result["match"]["entry_type"] == "folder"
    assert result["match"]["folder_id"] == "/docs"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_resolve_folder_ignores_max_results_when_folder_match_is_unique():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [
                         {"file_id": "fid-1", "file_name": "文档.txt", "busid": 102},
                         {"file_id": "fid-2", "file_name": "文档-备份.txt", "busid": 103},
                     ],
                     "folders": [{"folder_id": "/docs", "folder_name": "文档"}],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "resolve_folder",
                    "target": "group:987654321",
                    "query": "文档",
                    "recursive": False,
                    "max_results": 1,
                }
            )
        )

    assert result["success"] is True
    assert result["match"]["entry_type"] == "folder"
    assert result["match"]["folder_id"] == "/docs"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )


def test_get_url_resolved_resolves_then_fetches_url():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 (
                     {
                         "files": [{"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102}],
                         "folders": [],
                     },
                     None,
                 ),
                 (
                     {"url": "https://files.example.com/fid-1"},
                     None,
                 ),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "get_url_resolved",
                    "target": "group:987654321",
                    "query": "周报",
                }
            )
        )

    assert result["success"] is True
    assert result["url"] == "https://files.example.com/fid-1"
    assert result["resolved_match"]["file_id"] == "fid-1"
    assert call_mock.await_args_list[0].args == (
        qq_cfg.extra,
        "get_group_root_files",
        {"group_id": 987654321},
    )
    assert call_mock.await_args_list[1].args == (
        qq_cfg.extra,
        "get_group_file_url",
        {"group_id": 987654321, "file_id": "fid-1", "busid": 102},
    )


def test_delete_resolved_resolves_then_deletes():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 (
                     {
                         "files": [{"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102}],
                         "folders": [],
                     },
                     None,
                 ),
                 ({}, None),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "delete_resolved",
                    "target": "group:987654321",
                    "query": "周报",
                }
            )
        )

    assert result["success"] is True
    assert result["deleted"] is True
    assert result["resolved_match"]["file_id"] == "fid-1"
    assert call_mock.await_args_list[1].args == (
        qq_cfg.extra,
        "delete_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "busid": 102},
    )


def test_forward_resolved_resolves_then_forwards():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 (
                     {
                         "files": [{"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102}],
                         "folders": [],
                     },
                     None,
                 ),
                 ({}, None),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "forward_resolved",
                    "target": "group:987654321",
                    "query": "周报",
                    "target_group_id": "group:123456",
                }
            )
        )

    assert result["success"] is True
    assert result["forwarded"] is True
    assert result["resolved_match"]["file_id"] == "fid-1"
    assert result["target_group_id"] == "123456"
    assert call_mock.await_args_list[1].args == (
        qq_cfg.extra,
        "trans_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "target_group_id": 123456},
    )


def test_delete_resolved_rejects_ambiguous_matches_without_deleting():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [
                         {"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102},
                         {"file_id": "fid-2", "file_name": "周报-最终.pdf", "busid": 103},
                     ],
                     "folders": [],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "delete_resolved",
                    "target": "group:987654321",
                    "query": "周报",
                }
            )
        )

    assert "multiple" in result["error"].lower()
    assert result["match_count"] == 2
    assert len(call_mock.await_args_list) == 1


def test_delete_resolved_rejects_ambiguous_matches_even_with_max_results_one():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [
                         {"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102},
                         {"file_id": "fid-2", "file_name": "周报-最终.pdf", "busid": 103},
                     ],
                     "folders": [],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "delete_resolved",
                    "target": "group:987654321",
                    "query": "周报",
                    "max_results": 1,
                }
            )
        )

    assert "multiple" in result["error"].lower()
    assert result["match_count"] == 2
    assert len(call_mock.await_args_list) == 1


def test_delete_folder_resolved_resolves_then_deletes_folder():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 (
                     {
                         "files": [{"file_id": "fid-1", "file_name": "文档.txt", "busid": 102}],
                         "folders": [{"folder_id": "/docs", "folder_name": "文档"}],
                     },
                     None,
                 ),
                 ({}, None),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "delete_folder_resolved",
                    "target": "group:987654321",
                    "query": "文档",
                    "recursive": False,
                }
            )
        )

    assert result["success"] is True
    assert result["deleted"] is True
    assert result["folder_id"] == "/docs"
    assert result["resolved_match"]["entry_type"] == "folder"
    assert result["resolved_match"]["folder_id"] == "/docs"
    assert call_mock.await_args_list[1].args == (
        qq_cfg.extra,
        "delete_group_folder",
        {"group_id": 987654321, "folder_id": "/docs"},
    )


def test_delete_folder_resolved_rejects_ambiguous_matches_without_deleting():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(return_value=(
                 {
                     "files": [],
                     "folders": [
                         {"folder_id": "/docs", "folder_name": "文档"},
                         {"folder_id": "/docs-archive", "folder_name": "文档-归档"},
                     ],
                 },
                 None,
             )),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "delete_folder_resolved",
                    "target": "group:987654321",
                    "query": "文档",
                    "recursive": False,
                }
            )
        )

    assert "multiple" in result["error"].lower()
    assert result["match_count"] == 2
    assert len(call_mock.await_args_list) == 1


def test_move_resolved_resolves_then_moves():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 (
                     {
                         "files": [{"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102}],
                         "folders": [],
                     },
                     None,
                 ),
                 ({}, None),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "move_resolved",
                    "target": "group:987654321",
                    "query": "周报",
                    "target_dir": "/docs",
                }
            )
        )

    assert result["success"] is True
    assert result["moved"] is True
    assert result["resolved_match"]["file_id"] == "fid-1"
    assert result["target_dir"] == "/docs"
    assert call_mock.await_args_list[1].args == (
        qq_cfg.extra,
        "move_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "target_dir": "/docs"},
    )


def test_rename_resolved_resolves_then_renames():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_file_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 (
                     {
                         "files": [{"file_id": "fid-1", "file_name": "周报.pdf", "busid": 102}],
                         "folders": [],
                     },
                     None,
                 ),
                 ({}, None),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "rename_resolved",
                    "target": "group:987654321",
                    "query": "周报",
                    "new_name": "周报-归档.pdf",
                }
            )
        )

    assert result["success"] is True
    assert result["renamed"] is True
    assert result["resolved_match"]["file_id"] == "fid-1"
    assert result["current_parent_directory"] == "/"
    assert result["new_name"] == "周报-归档.pdf"
    assert call_mock.await_args_list[1].args == (
        qq_cfg.extra,
        "rename_group_file",
        {
            "group_id": 987654321,
            "file_id": "fid-1",
            "current_parent_directory": "/",
            "new_name": "周报-归档.pdf",
        },
    )


def test_rejects_dm_target_for_group_file_actions():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(
            qq_group_file_tool(
                {
                    "action": "list",
                    "target": "dm:123456",
                }
            )
        )

    assert "group" in result["error"].lower()
    assert "dm" in result["error"].lower()
