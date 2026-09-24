"""RPC history must project compaction messages without NameError."""

from __future__ import annotations

from agent.context_compressor import SUMMARY_PREFIX, _SUMMARY_END_MARKER
import tui_gateway.server as server


def test_history_to_messages_compaction_projection():
    history = [
        {"role": "user", "content": "Hello"},
        {
            "role": "user",
            "content": f"{SUMMARY_PREFIX}\nSome compaction summary\n{_SUMMARY_END_MARKER}",
        },
        {"role": "assistant", "content": "Hi there!"},
    ]

    messages = server._history_to_messages(history)
    assert len(messages) == 2
    assert messages[0]["text"] == "Hello"
    assert messages[1]["text"] == "Hi there!"
