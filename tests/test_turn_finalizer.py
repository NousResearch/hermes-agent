from agent.turn_finalizer import _ensure_final_response_message


def test_ensure_final_response_appends_after_reasoning_only_assistant_tail():
    messages = [{
        "role": "assistant",
        "content": "",
        "reasoning_content": "I verified the implementation.",
    }]

    appended = _ensure_final_response_message(messages, "The implementation is complete.", False)

    assert appended is True
    assert messages == [{
        "role": "assistant",
        "content": "",
        "reasoning_content": "I verified the implementation.",
    }, {
        "role": "assistant",
        "content": "The implementation is complete.",
    }]


def test_ensure_final_response_does_not_duplicate_visible_assistant_tail():
    messages = [{"role": "assistant", "content": "The implementation is complete."}]

    appended = _ensure_final_response_message(messages, "The implementation is complete.", False)

    assert appended is False
    assert messages == [{"role": "assistant", "content": "The implementation is complete."}]


def test_ensure_final_response_appends_after_non_assistant_tail():
    messages = [{"role": "tool", "content": "All checks passed."}]

    appended = _ensure_final_response_message(messages, "The implementation is complete.", False)

    assert appended is True
    assert messages[-1] == {"role": "assistant", "content": "The implementation is complete."}


def test_ensure_final_response_leaves_interrupted_turn_unchanged():
    messages = [{"role": "assistant", "content": "", "reasoning_content": "Partial reasoning"}]

    appended = _ensure_final_response_message(messages, "Partial answer", True)

    assert appended is False
    assert len(messages) == 1
