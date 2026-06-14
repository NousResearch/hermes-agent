from tools.process_registry import format_process_notification


def test_format_codex_learning_staged_event():
    evt = {
        "type": "codex_learning_staged",
        "message": "Codex learning staged 1 proposal.",
    }

    result = format_process_notification(evt)

    assert result == "[IMPORTANT: Codex learning staged 1 proposal.]"
