"""Contracts for the user-scoped Feishu message tools (#11540).

Both handlers are driven end to end against a local stand-in for ``open.feishu.cn``, so the request
shape Feishu actually receives — query string vs. JSON body, filter nesting, page-size caps — is
asserted rather than assumed.
"""

import json

import pytest

from tests.tools.feishu_user_helpers import (
    FakeFeishu, configure_app, point_open_host, store_grant)
from tools import feishu_user_auth, feishu_user_tool


@pytest.fixture
def signed_in(monkeypatch):
    configure_app(monkeypatch)
    store_grant()


def _search_response(items, **extra):
    data = {"items": items, "has_more": False, "page_token": "", "total": len(items)}
    data.update(extra)
    return (200, {"code": 0, "msg": "success", "data": data})


def test_search_sends_the_filter_in_the_body_and_paging_in_the_query(signed_in, monkeypatch):
    """Feishu splits this call: ``query``/``filter`` are JSON, ``page_size``/``page_token`` are query."""
    with FakeFeishu([_search_response([])]) as server:
        point_open_host(monkeypatch, server)
        feishu_user_tool._handle_feishu_message_search({
            "query": "team dinner", "chat_ids": ["oc_1", "oc_2"], "from_ids": ["ou_a"],
            "start_time": "2026-03-21T16:15:30+08:00", "page_size": 7, "page_token": "tok-1",
        })
        request = server.requests[0]
        body = server.json_body(0)

    assert request["method"] == "POST"
    assert request["path"] == "/open-apis/im/v1/messages/search"
    assert request["query"] == {"page_size": "7", "page_token": "tok-1"}
    assert request["headers"]["Authorization"] == "Bearer u-access-1"
    assert body == {
        "query": "team dinner",
        "filter": {
            "chat_ids": ["oc_1", "oc_2"], "from_ids": ["ou_a"],
            "time_range": {"start_time": "2026-03-21T16:15:30+08:00"},
        },
    }


def test_search_omits_the_filter_when_nothing_narrows_it(signed_in, monkeypatch):
    """An empty ``filter`` object is not the same request as no filter; don't send one."""
    with FakeFeishu([_search_response([])]) as server:
        point_open_host(monkeypatch, server)
        feishu_user_tool._handle_feishu_message_search({"query": "anything"})
        assert server.json_body(0) == {"query": "anything"}


def test_search_page_size_is_clamped_to_the_api_cap(signed_in, monkeypatch):
    """Feishu 400s above 30; clamping keeps an over-eager model's call working."""
    with FakeFeishu([_search_response([]), _search_response([])]) as server:
        point_open_host(monkeypatch, server)
        feishu_user_tool._handle_feishu_message_search({"query": "q", "page_size": 500})
        feishu_user_tool._handle_feishu_message_search({"query": "q", "page_size": 0})
        sizes = [server.requests[0]["query"]["page_size"], server.requests[1]["query"]["page_size"]]
    assert sizes == [str(feishu_user_tool._SEARCH_PAGE_SIZE_MAX), "1"]


def test_search_rows_carry_the_chat_and_sender_and_drop_highlight_markup(signed_in, monkeypatch):
    """``display_info`` arrives wrapped in <h> tags; a follow-up read needs chat_id and message_id."""
    item = {
        "id": "om_1", "display_info": "where is <h>dinner</h>",
        "meta_data": {
            "message_id": "om_1", "chat_id": "oc_9", "from_id": "ou_b", "type": "text",
            "create_time": "2026-03-21T16:15:30+08:00", "is_p2p_chat": False,
        },
    }
    with FakeFeishu([_search_response([item], has_more=True, page_token="next", total=42)]) as server:
        point_open_host(monkeypatch, server)
        result = json.loads(feishu_user_tool._handle_feishu_message_search({"query": "dinner"}))

    assert result["success"] is True
    assert result["total"] == 42
    assert result["has_more"] is True
    assert result["page_token"] == "next"
    assert result["messages"] == [{
        "message_id": "om_1", "chat_id": "oc_9", "from_id": "ou_b", "msg_type": "text",
        "create_time": "2026-03-21T16:15:30+08:00", "thread_id": None, "is_p2p_chat": False,
        "snippet": "where is dinner",
    }]


def test_a_missing_scope_error_names_the_fix_instead_of_echoing_feishus_code(signed_in, monkeypatch):
    """99991672 means the grant lacks a scope — re-running login is the only fix, so say so."""
    with FakeFeishu([(200, {"code": 99991672, "msg": "no permission"})]) as server:
        point_open_host(monkeypatch, server)
        result = json.loads(feishu_user_tool._handle_feishu_message_search({"query": "q"}))
    assert "hermes feishu login" in result["error"]


def test_list_reads_one_chat_and_extracts_the_text_body(signed_in, monkeypatch):
    """The wire carries ``body.content`` as a JSON string; a model should get the words."""
    item = {
        "message_id": "om_2", "chat_id": "oc_9", "msg_type": "text",
        "create_time": "1742544930", "deleted": False, "root_id": "om_0",
        "sender": {"id": "cli_other_bot", "sender_type": "app"},
        "body": {"content": json.dumps({"text": "hello from another agent"})},
    }
    response = (200, {"code": 0, "msg": "ok", "data": {"items": [item], "has_more": False}})
    with FakeFeishu([response]) as server:
        point_open_host(monkeypatch, server)
        result = json.loads(feishu_user_tool._handle_feishu_message_list({
            "chat_id": "oc_9", "sort_type": "ByCreateTimeDesc", "start_time": "1742544000"}))
        request = server.requests[0]

    assert request["method"] == "GET"
    assert request["query"] == {
        "container_id_type": "chat", "container_id": "oc_9", "start_time": "1742544000",
        "sort_type": "ByCreateTimeDesc", "page_size": "15"}
    assert result["messages"][0]["text"] == "hello from another agent"
    # The point of the user grant: another app's message is visible where the bot's own view is not.
    assert result["messages"][0]["sender_type"] == "app"


def test_list_drops_an_unsupported_sort_type_rather_than_400ing(signed_in, monkeypatch):
    with FakeFeishu([(200, {"code": 0, "data": {"items": []}})]) as server:
        point_open_host(monkeypatch, server)
        feishu_user_tool._handle_feishu_message_list({"chat_id": "oc_9", "sort_type": "newest"})
        assert "sort_type" not in server.requests[0]["query"]


def test_a_non_text_body_is_previewed_not_dropped(signed_in, monkeypatch):
    """A post/card payload has no ``text`` field; returning nothing would hide the message."""
    content = json.dumps({"title": "t", "content": [[{"tag": "text", "text": "rich"}]]})
    response = (200, {"code": 0, "data": {"items": [
        {"message_id": "om_3", "msg_type": "post", "body": {"content": content}}]}})
    with FakeFeishu([response]) as server:
        point_open_host(monkeypatch, server)
        result = json.loads(feishu_user_tool._handle_feishu_message_list({"chat_id": "oc_9"}))
    assert "rich" in result["messages"][0]["text"]


def test_both_tools_stay_out_of_the_schema_until_a_grant_is_stored(monkeypatch):
    """The gate is the stored grant, so an install that never authorized pays no schema cost."""
    configure_app(monkeypatch)
    assert feishu_user_auth.has_user_token() is False
    store_grant()
    assert feishu_user_auth.has_user_token() is True


def test_the_tools_are_registered_in_the_feishu_user_toolset_and_the_bot_bundle():
    """Reachability contract: opt-in toolset plus the Feishu gateway bundle, never a core tool."""
    from toolsets import TOOLSETS, _HERMES_CORE_TOOLS
    from tools.registry import registry

    names = ("feishu_message_search", "feishu_message_list")
    for name in names:
        entry = registry._tools[name]
        assert entry.toolset == "feishu_user"
        assert entry.check_fn is feishu_user_auth.has_user_token
        assert name not in _HERMES_CORE_TOOLS
        assert name in TOOLSETS["feishu_user"]["tools"]
        assert name in TOOLSETS["hermes-feishu"]["tools"]
    # The positional slice behind ``feishu_drive`` must not have absorbed them.
    assert not set(names) & set(TOOLSETS["feishu_drive"]["tools"])
