"""Secret capture behavior at the TUI/Desktop gateway callback boundary."""

from agent.secret_sources import bitwarden_write
from tools.secret_capture_tool import get_secret_capture_callback
from tui_gateway import server


def test_bitwarden_failure_is_safe_and_destination_is_bound_to_prompt(monkeypatch):
    prompts = []

    def ask(kind, sid, payload):
        prompts.append((kind, sid, payload))
        return "captured-secret"

    def fail_storage(_name, _value):
        raise ValueError("provider detail containing captured-secret")

    monkeypatch.setattr(server, "_ask", ask)
    monkeypatch.setattr(bitwarden_write, "store_bitwarden_secret", fail_storage)

    server._wire_callbacks("session-a")
    callback = get_secret_capture_callback()
    assert callback is not None

    result = callback(
        "OPENROUTER_API_KEY",
        "Replacement OpenRouter key",
        {"destination": "bitwarden_sm", "source": "secret_capture"},
    )

    assert prompts == [(
        "secret",
        "session-a",
        {
            "prompt": "Replacement OpenRouter key",
            "env_var": "OPENROUTER_API_KEY",
            "destination": "bitwarden_sm",
            "metadata": {"destination": "bitwarden_sm", "source": "secret_capture"},
        },
    )]
    assert result == {
        "success": False,
        "stored_as": "OPENROUTER_API_KEY",
        "validated": False,
        "skipped": False,
        "error": "Bitwarden secret storage failed.",
    }
