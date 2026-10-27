"""Group-chat memory writes stay out of the owner's MEMORY.md / USER.md."""

import json

import pytest

from tools import memory_tool as mt
from tools.memory_tool_channel import CHANNEL_MEMORY, ChannelMemoryBinding, channel_prompt_block


@pytest.fixture()
def mem_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    return tmp_path


def _add(store, target, content):
    return json.loads(mt.memory_tool(target=target, operations=[{"action": "add", "content": content}], store=store))


def test_group_member_writes_land_in_channel_memory_not_owner_files(mem_dir):
    store = mt.MemoryStore()
    store.load_from_disk()
    token = CHANNEL_MEMORY.set(ChannelMemoryBinding(scope="telegram/-1001", owner_writes=False))
    try:
        user_result = _add(store, "user", "User is Mallory and prefers terse replies")
        memory_result = _add(store, "memory", "Standup is at 9am")
    finally:
        CHANNEL_MEMORY.reset(token)

    assert user_result["success"] and user_result["target"] == "channel"
    assert memory_result["routed_to"] == "channel"
    assert not (mem_dir / "USER.md").exists() and not (mem_dir / "MEMORY.md").exists()
    channel_file = (mem_dir / "channels" / "telegram" / "-1001.md").read_text(encoding="utf-8-sig")
    assert "Mallory" in channel_file and "Standup" in channel_file

    # Owner-trusted sender in the same group, and any write outside a group, keep the legacy target.
    token = CHANNEL_MEMORY.set(ChannelMemoryBinding(scope="telegram/-1001", owner_writes=True))
    try:
        assert _add(store, "memory", "Owner uses fish shell")["target"] == "memory"
    finally:
        CHANNEL_MEMORY.reset(token)
    assert _add(store, "user", "Owner is Ada")["target"] == "user"
    assert "Owner is Ada" in (mem_dir / "USER.md").read_text(encoding="utf-8-sig")


def test_channel_notes_load_only_in_their_chat(mem_dir):
    class Agent:
        _memory_store = mt.MemoryStore()

    token = CHANNEL_MEMORY.set(ChannelMemoryBinding(scope="slack/T1_C1", owner_writes=False))
    try:
        _add(Agent._memory_store, "memory", "Deploys freeze on Fridays")
        assert "Deploys freeze on Fridays" in channel_prompt_block(Agent())
    finally:
        CHANNEL_MEMORY.reset(token)

    other = CHANNEL_MEMORY.set(ChannelMemoryBinding(scope="slack/T1_C2", owner_writes=False))
    try:
        assert channel_prompt_block(Agent()) is None
    finally:
        CHANNEL_MEMORY.reset(other)
    assert channel_prompt_block(Agent()) is None  # DM / CLI: no binding, no block


def test_turn_start_gives_channel_writes_a_fresh_retry_budget(mem_dir):
    store = mt.MemoryStore()
    store.load_from_disk()
    token = CHANNEL_MEMORY.set(ChannelMemoryBinding(scope="telegram/-1001", owner_writes=False, char_limit=20))
    try:
        for _ in range(4):  # past the per-turn cap: terminal "stop retrying"
            last = _add(store, "memory", "x" * 30)
        assert last.get("done") is True and "current_entries" not in last
        store.reset_consolidation_failures()  # what agent/turn_context.py does at every turn start
        fresh = _add(store, "memory", "x" * 30)
    finally:
        CHANNEL_MEMORY.reset(token)
    assert "current_entries" in fresh and "Stop retrying" not in fresh["error"]


def test_redundant_nested_target_still_routes_to_channel_memory(mem_dir):
    store = mt.MemoryStore()
    store.load_from_disk()
    token = CHANNEL_MEMORY.set(ChannelMemoryBinding(scope="telegram/-1001", owner_writes=False))
    try:
        result = json.loads(mt.memory_tool(
            target="memory", operations=[{"target": "memory", "action": "add", "content": "Standup is at 9am"}],
            store=store))
        mixed = json.loads(mt.memory_tool(
            target="memory", operations=[{"target": "user", "action": "add", "content": "two stores"}], store=store))
    finally:
        CHANNEL_MEMORY.reset(token)
    assert result["success"] and result["routed_to"] == "channel"
    assert not (mem_dir / "MEMORY.md").exists()
    assert mixed["success"] is False  # a different nested target is still one call asking for two stores
