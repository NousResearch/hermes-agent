"""A failed TUI turn must say why in its own record (#89117).

#89117 is a report made entirely of two log lines::

    tui_turn finished: ui_session=0dfcee58 status=error error_retained=True duration=0.9s
    tui_turn finished: ui_session=093285e9 status=error error_retained=True duration=0.9s

That is the whole evidence, and it is not enough to act on: a provider 4xx, a
budget wall, a billing block and a crashed finalizer all produce exactly those
characters. The bookend was added by #86865 to trace compression rotations, so
it carries identities and a coarse status by design — but it is also the *only*
record the returned-error path writes. A sub-second failure almost always takes
that path (the provider rejected the request before any work happened), so the
quietest failures are precisely the ones with nothing to read. The exception
path at least prints ``[gateway-turn] <Type>: <msg>`` to stderr.

These tests pin the cause into the record on both failure paths, and pin the
content discipline #86865 established while doing it: prompts are never logged,
and the provider's message is redacted and length-capped, because a 4xx body
can quote the request that produced it.
"""

from __future__ import annotations

import logging
import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "gw-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _finished(caplog):
    records = [r for r in caplog.records if "tui turn finished" in r.getMessage()]
    assert len(records) == 1, f"expected exactly one bookend, got {len(records)}"
    return records[0].getMessage()


def _run(session, prompt="go"):
    server._run_prompt_submit("rid", "ui-sid", session, prompt)


def _agent_returning(result):
    return types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: result,
        clear_interrupt=lambda: None,
    )


class TestTheReportedRecordNowNamesItsCause:

    def test_returned_error_carries_the_provider_message(self, turn_env, caplog):
        """The reporter's exact line shape, with the missing half filled in."""
        session = _session(agent=_agent_returning({
            "final_response": "",
            "error": "Error code: 402 - {'error': {'message': 'insufficient credits'}}",
            "failed": True,
        }))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session)

        msg = _finished(caplog)
        assert "status=error" in msg
        assert "error_retained=True" in msg
        assert "insufficient credits" in msg, (
            "a record that says only status=error is what #89117 is about"
        )

    def test_structured_failure_reason_is_logged_when_present(self, turn_env, caplog):
        """The billing wall already ships a machine-readable reason; use it.

        ``failure_reason`` is the field the client renders a billing-specific
        recovery surface from, so it is the one field guaranteed to be stable
        enough to grep a log for across releases.
        """
        session = _session(agent=_agent_returning({
            "final_response": "",
            "error": "payment required",
            "failure_reason": "billing_wall",
            "failed": True,
        }))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session)

        assert "failure_reason=billing_wall" in _finished(caplog)

    def test_exception_path_carries_the_exception(self, turn_env, caplog):
        """The other failure path, so one grep covers both."""
        def _boom(*a, **k):
            raise RuntimeError("connection reset mid-stream")

        session = _session(agent=types.SimpleNamespace(
            session_id="agent-sid-1",
            run_conversation=_boom,
            clear_interrupt=lambda: None,
        ))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session)

        msg = _finished(caplog)
        assert "status=error" in msg
        assert "failure_reason=RuntimeError" in msg
        assert "connection reset mid-stream" in msg

    def test_successful_turn_stays_exactly_as_it_was(self, turn_env, caplog):
        """No cost to the common case: a clean turn gains no new fields."""
        session = _session(agent=_agent_returning({"final_response": "done"}))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session)

        msg = _finished(caplog)
        assert "status=complete" in msg
        assert "cause=" not in msg
        assert "failure_reason=" not in msg


class TestContentDiscipline:
    """#86865's rule — the record logs identities, never content."""

    SECRETISH_PROMPT = "please rotate QDRANT_API_KEY=hunter2-super-secret now"

    def test_prompt_is_never_logged_even_when_the_turn_fails(self, turn_env, caplog):
        session = _session(agent=_agent_returning({
            "final_response": "",
            "error": "provider rejected the request",
            "failed": True,
        }))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session, self.SECRETISH_PROMPT)

        msg = _finished(caplog)
        assert "hunter2" not in msg
        assert "QDRANT_API_KEY" not in msg

    def test_secrets_echoed_back_by_the_provider_are_redacted(self, turn_env, caplog):
        """The load-bearing safety test.

        A 4xx body frequently quotes the request. Without redaction, adding the
        cause to a log record would take a header the user never chose to log
        and write it to disk — turning a diagnostics improvement into a secret
        leak. This is why the cause goes through ``redact_sensitive_text`` and
        not ``str()``.
        """
        session = _session(agent=_agent_returning({
            "final_response": "",
            "error": (
                "400 from provider; request headers were "
                "Authorization: Bearer sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
            ),
            "failed": True,
        }))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session)

        msg = _finished(caplog)
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789" not in msg
        # The diagnostic value survives the redaction — this is the point.
        assert "400 from provider" in msg

    SENTINEL = "the marmalade inventory for Q3 was discontinued in March"

    def test_a_prompt_the_provider_quotes_back_does_not_reach_the_record(
        self, turn_env, caplog
    ):
        """Secret redaction is not prompt omission, and this is the difference.

        A provider that rejects a request routinely quotes it back. The quoted
        material is the user's own prose: it matches no credential pattern, so
        ``redact_sensitive_text`` passes it through untouched, and adding the
        cause to this record would newly persist user content that #86865
        deliberately kept out of it. The sentinel here is deliberately benign
        for that reason: nothing about it looks like a secret.
        """
        session = _session(agent=_agent_returning({
            "final_response": "",
            "error": (
                "400 Bad Request from provider: messages[0].content was "
                "rejected: '" + self.SENTINEL + "'"
            ),
            "failed": True,
        }))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session, "Summarise this: " + self.SENTINEL)

        msg = _finished(caplog)
        assert self.SENTINEL not in msg
        assert "marmalade" not in msg
        assert "<prompt>" in msg, "the removal should be visible, not silent"
        # The whole point of the cause survives the removal.
        assert "400 Bad Request from provider" in msg

    def test_a_provider_message_that_shares_nothing_is_untouched(
        self, turn_env, caplog
    ):
        """The echo strip must not eat diagnostics that merely sit near a prompt."""
        session = _session(agent=_agent_returning({
            "final_response": "",
            "error": "429 rate limited; retry after 30s",
            "failed": True,
        }))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session, "Summarise this: " + self.SENTINEL)

        msg = _finished(caplog)
        assert "429 rate limited; retry after 30s" in msg
        assert "<prompt>" not in msg

    def test_a_huge_provider_body_cannot_flood_the_log(self, turn_env, caplog):
        """An HTML error page or a full request echo is a log-volume problem."""
        session = _session(agent=_agent_returning({
            "final_response": "",
            "error": "upstream said: " + ("x" * 9000),
            "failed": True,
        }))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session)

        msg = _finished(caplog)
        assert len(msg) < 700
        assert "upstream said" in msg
        assert "…" in msg, "truncation should be visible, not silent"

    def test_a_multiline_traceback_stays_one_record(self, turn_env, caplog):
        """One accepted prompt, one finished record — including its cause.

        A cause spanning lines would break every log pipeline that treats the
        bookend as a single greppable line, which is the only reason it is
        useful for an intermittent bug like this one.
        """
        def _boom(*a, **k):
            raise RuntimeError("first line\nsecond line\n\tthird")

        session = _session(agent=types.SimpleNamespace(
            session_id="agent-sid-1",
            run_conversation=_boom,
            clear_interrupt=lambda: None,
        ))

        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            _run(session)

        msg = _finished(caplog)
        assert "\n" not in msg
        assert "first line second line third" in msg


class TestDetailHelperDirectly:
    """``_turn_failure_detail`` in isolation — the branches the paths can't reach."""

    def test_nothing_to_say_produces_nothing(self):
        assert server._turn_failure_detail("", None) == ""
        assert server._turn_failure_detail(None) == ""

    def test_fragment_carries_its_own_leading_space(self):
        """The bookend appends it unconditionally, so it must self-format."""
        out = server._turn_failure_detail("boom")
        assert out.startswith(" ")

    def test_an_exception_with_no_message_still_names_its_type(self):
        assert "KeyError" in server._turn_failure_detail(KeyError())

    def test_the_prompt_argument_is_optional(self):
        """Callers without a prompt in scope still get the secret contract."""
        out = server._turn_failure_detail("Bearer sk-proj-supersecretvalue1234")
        assert "supersecretvalue1234" not in out

    def test_a_broken_redactor_fails_closed(self, monkeypatch):
        """If redaction cannot run, the raw message must not reach the log.

        Failing open here would be worse than logging nothing: the whole reason
        the cause is safe to log is that it went through the redactor.
        """
        import agent.redact

        def _explode(*a, **k):
            raise RuntimeError("redactor unavailable")

        monkeypatch.setattr(agent.redact, "redact_sensitive_text", _explode)

        out = server._turn_failure_detail("Bearer sk-proj-supersecretvalue")
        assert "supersecretvalue" not in out
        assert "unredactable" in out


class TestPromptEchoStripping:
    """``_strip_prompt_echo`` in isolation: the boundaries of the guarantee."""

    def test_an_overlap_below_the_window_is_not_an_echo(self):
        """Short shared phrases are coincidence, and eating them costs detail."""
        out = server._strip_prompt_echo("400: invalid model", "invalid model")
        assert out == "400: invalid model"

    def test_a_json_escaped_echo_is_stripped_too(self):
        """A provider handing back its own request body often hands it escaped."""
        prompt = "please summarise the Q3 marmalade inventory memo for me"
        message = 'upstream body: {"messages": [{"content": "' + prompt + '"}]}'
        out = server._strip_prompt_echo(message, prompt)
        assert "marmalade" not in out
        assert "<prompt>" in out

    def test_an_echo_is_removed_before_the_length_cap_applies(self):
        """A quote must not survive by starting inside the kept prefix."""
        prompt = "the confidential merger memorandum for the northern division"
        error = ("x" * 200) + " echoed request: " + prompt
        out = server._turn_failure_detail(error, None, prompt)
        assert "merger memorandum" not in out
        assert "confidential" not in out

    def test_a_prompt_shorter_than_the_window_cannot_blank_the_message(self):
        """A one-word prompt must not turn every message into <prompt>."""
        out = server._strip_prompt_echo("provider said no", "hi")
        assert out == "provider said no"

    def test_whitespace_shape_does_not_hide_an_echo(self):
        """Both sides are collapsed, so a re-wrapped quote still matches."""
        prompt = "the marmalade inventory for Q3 was discontinued in March"
        error = (
            "rejected:    the marmalade inventory\n"
            " for Q3 was discontinued in March"
        )
        out = server._turn_failure_detail(error, None, prompt)
        assert "marmalade" not in out
