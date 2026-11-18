"""One reader per background job — the always-loaded description guard.

A completion notification injects a synthetic message into the session
(``gateway/run.py``), so it STARTS A NEW TURN.  That is why a background job
must have exactly one reader: if the agent both arms the notification and
reads the result itself, the same result is delivered twice — once by the
agent, once as a fresh turn that re-reads the whole context.

These descriptions are the ALWAYS-LOADED surface: they are in the tool
schema of every session, whereas the devops skill is opt-in.  So the rule
has to live here, or an agent that never opens the skill gets the opposite
instruction — which is exactly what happened.

The guard is deliberately ABOUT THE TEXT, because the text is the delivery
mechanism.  Reverting any one description must fail it.
"""

from __future__ import annotations


def _always_loaded_background_descriptions() -> dict[str, str]:
    """Every surface that tells an agent how to run a background job."""
    from tools.process_registry import PROCESS_SCHEMA
    from tools.terminal_tool import TERMINAL_SCHEMA, TERMINAL_TOOL_DESCRIPTION

    props = TERMINAL_SCHEMA["parameters"]["properties"]
    return {
        "terminal_tool_description": TERMINAL_TOOL_DESCRIPTION,
        "schema.background": props["background"]["description"],
        "schema.notify": props["notify"]["description"],
        "process_tool_description": PROCESS_SCHEMA["description"],
    }


class TestOneReaderPerBackgroundJob:
    """The always-loaded descriptions must teach ONE reader, not two."""

    def test_no_description_tells_the_agent_to_must_arm_notify(self) -> None:
        """The old text said 'MUST set notify_on_complete=true' for bounded
        tasks.  That is half the rule: it never said the notification is a
        new TURN, so agents armed it AND polled — and got the result twice."""
        for name, text in _always_loaded_background_descriptions().items():
            assert "MUST set notify" not in text, (
                f"{name} still mandates the notification without stating the "
                f"one-reader rule: {text!r}"
            )
            assert "Almost always pair with notify" not in text, (
                f"{name} still says 'almost always pair' with the "
                f"notification: {text!r}"
            )

    def test_every_description_states_the_one_reader_rule(self) -> None:
        """Each surface must name the rule, so it is findable without
        loading an opt-in skill."""
        for name, text in _always_loaded_background_descriptions().items():
            assert "one reader" in text.lower(), (
                f"{name} does not state the one-reader rule: {text!r}"
            )

    def test_descriptions_state_that_a_notification_is_a_new_turn(self) -> None:
        """The cost is the point.  Without it, 'notify' reads as a passive
        message and gets combined with polling."""
        for name, text in _always_loaded_background_descriptions().items():
            assert "new turn" in text.lower(), (
                f"{name} does not say the notification starts a new turn: {text!r}"
            )

    def test_the_silent_background_hint_does_not_teach_both(self) -> None:
        """The hint that fires when a bounded job was launched silently.

        It used to say "re-launch with notify_on_complete=true, OR call
        poll/wait yourself" — and an agent that did BOTH armed the
        notification AND read the job, so the result arrived twice.  The
        hint must offer the CHOICE, not both options at once.
        """
        from tools.terminal_tool_background import _SILENT_BACKGROUND_HINT

        lowered = _SILENT_BACKGROUND_HINT.lower()
        assert "one reader" in lowered, (
            f"the silent-background hint does not state the one-reader rule: "
            f"{_SILENT_BACKGROUND_HINT!r}"
        )
        assert "new turn" in lowered, (
            f"the silent-background hint does not say the notification starts a "
            f"new turn: {_SILENT_BACKGROUND_HINT!r}"
        )
        assert "re-launch with notify" not in lowered, (
            f"the silent-background hint still offers 're-launch with notify' "
            f"as an alternative to reading the job yourself: "
            f"{_SILENT_BACKGROUND_HINT!r}"
        )
