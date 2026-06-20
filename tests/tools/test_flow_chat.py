"""共享对话流判据 tools.flow_chat —— 资源采集与注册口共用,判定「ChatInput + ChatOutput」。"""

from __future__ import annotations

from tools.flow_chat import flow_node_types, is_chat_flow


def _node(type_=None, node_id=None):
    n = {}
    if type_ is not None:
        n["data"] = {"type": type_}
    if node_id is not None:
        n["id"] = node_id
    return n


def _flow(nodes):
    return {"id": "f", "name": "flow", "data": {"nodes": nodes}}


def test_requires_both_chat_input_and_output():
    assert is_chat_flow(_flow([_node("ChatInput"), _node("ChatOutput")])) is True
    assert is_chat_flow(_flow([_node("ChatInput")])) is False
    assert is_chat_flow(_flow([_node("ChatOutput")])) is False
    assert is_chat_flow(_flow([_node("TextInput"), _node("ChatOutput")])) is False
    assert is_chat_flow(_flow([])) is False


def test_falls_back_to_node_id_prefix():
    # 没 data.type 时从 id 前缀("ChatInput-xxxx")判定。
    assert is_chat_flow(_flow([_node(node_id="ChatInput-a1"), _node(node_id="ChatOutput-b2")])) is True


def test_node_types_collects_and_ignores_garbage():
    types = flow_node_types(_flow([_node("ChatInput"), {"not": "a node"}, _node("ChatOutput"), "bad"]))
    assert types == {"ChatInput", "ChatOutput"}


def test_non_dict_or_missing_data_is_not_chat():
    assert is_chat_flow(None) is False
    assert is_chat_flow({}) is False
    assert is_chat_flow({"data": {}}) is False
