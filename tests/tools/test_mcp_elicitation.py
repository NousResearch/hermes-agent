"""Tests for the MCP elicitation handler in tools.mcp_tool_sampling.

These tests exercise ElicitationHandler in isolation -- the underlying
approval system and the MCP transport layer are mocked, so no real MCP
server or user input is required.

Tests skip cleanly if the optional `mcp` SDK is not installed (it is an
optional dependency under the `[mcp]` extra).
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import patch

import pytest


pytest.importorskip("mcp.types")

from mcp.types import ElicitResult  # noqa: E402  -- after importorskip

import tools.mcp_tool_sampling as mcp_tool  # noqa: E402

from tools.mcp_tool_sampling import (  # noqa: E402
    ElicitationHandler,
    _format_elicitation_schema_summary,
    set_elicitation_input_callback,
)


@contextmanager
def _input_callback(cb):
    """Stand in for the surface that asks the user a question.

    Restores whatever was registered, so one test's fake answerer can't make
    the next test's unwired-surface case quietly pass.
    """
    previous = mcp_tool._elicitation_input_cb

    set_elicitation_input_callback(cb)
    try:
        yield
    finally:
        set_elicitation_input_callback(previous)


def _form_params(message="please confirm", schema=None):
    """Build a stand-in for ElicitRequestFormParams.

    We use a plain object (not the SDK type directly) so the test doesn't
    couple to optional Pydantic validation -- the handler reads fields via
    getattr() and tolerates duck-typed inputs.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        mode="form",
        message=message,
        requested_schema=schema or {},
    )


def _url_params(message="open this url", url="https://example.com/auth", elicitation_id="e1"):
    from types import SimpleNamespace
    return SimpleNamespace(
        mode="url",
        message=message,
        url=url,
        elicitation_id=elicitation_id,
    )


class TestSchemaSummary:
    def test_empty_schema_falls_back_to_generic_message(self):
        out = _format_elicitation_schema_summary({}, "pay")
        assert "pay" in out
        assert "Approval requested" in out

    def test_properties_render_with_type_and_description(self):
        schema = {
            "type": "object",
            "properties": {
                "amount": {"type": "string", "description": "USD amount"},
                "recipient": {"type": "string"},
            },
        }
        out = _format_elicitation_schema_summary(schema, "pay")
        assert "amount (string): USD amount" in out
        assert "recipient (string)" in out


class TestElicitationHandlerFormMode:
    def test_a_bare_confirmation_accepts(self):
        """No properties means the server asked a yes/no, which consent answers."""
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params("authorize a payment of $0.50", {"type": "object"})

        with patch("tools.approval_prompt.request_elicitation_consent", return_value="accept"):
            result = asyncio.run(handler(context=None, params=params))

        assert isinstance(result, ElicitResult)
        assert result.action == "accept"
        assert result.content == {}
        assert handler.metrics["accepted"] == 1
        assert handler.metrics["declined"] == 0

    def test_a_request_for_fields_is_answered_with_the_fields(self):
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params(
            "how much should I send?",
            {"properties": {"amount": {"type": "string"}}, "required": ["amount"]},
        )

        with _input_callback(lambda question, choices: "12.50"):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "accept"
        assert result.content == {"amount": "12.50"}
        assert handler.metrics["accepted"] == 1

    def test_the_server_sentence_leads_the_first_question_only(self):
        """It explains why any of this is being asked. Repeating it above
        every field turns a two-field form into the same paragraph twice."""
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params(
            "authorizing your payment",
            {"properties": {"amount": {"type": "string"}, "note": {"type": "string"}}},
        )
        asked: list = []

        with _input_callback(lambda question, choices: asked.append(question) or "x"):
            asyncio.run(handler(context=None, params=params))

        assert asked[0].startswith("authorizing your payment")
        assert "authorizing your payment" not in asked[1]

    def test_declared_types_survive_the_round_trip(self):
        """The user answers in text; the server declared types. Handing back
        the string "3" for an integer field is a contract violation the server
        has no way to see coming."""
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params(
            "details",
            {
                "properties": {
                    "count": {"type": "integer"},
                    "rate": {"type": "number"},
                    "confirm": {"type": "boolean"},
                    "label": {"type": "string"},
                }
            },
        )
        answers = {"count": "3", "rate": "1.5", "confirm": "Yes", "label": "hi"}

        def answer(question, choices):
            return next(value for field, value in answers.items() if field in question)

        with _input_callback(answer):
            result = asyncio.run(handler(context=None, params=params))

        assert result.content == {"count": 3, "rate": 1.5, "confirm": True, "label": "hi"}

    def test_an_enum_is_offered_as_choices_rather_than_typed_out(self):
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params(
            "pick one",
            {"properties": {"tier": {"type": "string", "enum": ["basic", "pro"]}}},
        )
        offered: list = []

        with _input_callback(lambda question, choices: (offered.append(choices), "pro")[1]):
            result = asyncio.run(handler(context=None, params=params))

        assert offered == [["basic", "pro"]]
        assert result.content == {"tier": "pro"}

    def test_a_blank_required_field_declines_the_whole_form(self):
        """Half a form is not an answer: the server asked for this one."""
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params(
            "how much?",
            {"properties": {"amount": {"type": "string"}}, "required": ["amount"]},
        )

        with _input_callback(lambda question, choices: "   "):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "decline"
        assert handler.metrics["declined"] == 1

    def test_a_blank_optional_field_is_simply_absent(self):
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params(
            "details",
            {
                "properties": {"amount": {"type": "string"}, "note": {"type": "string"}},
                "required": ["amount"],
            },
        )

        with _input_callback(lambda question, choices: "" if "note" in question else "5"):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "accept"
        assert result.content == {"amount": "5"}

    def test_a_surface_that_cannot_ask_declines_instead_of_inventing_data(self):
        """Accepting with no content would tell the server the user filled the
        form in and left every field blank."""
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params(
            "how much should I send?",
            {"properties": {"amount": {"type": "string"}}, "required": ["amount"]},
        )

        with _input_callback(None):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "decline"
        assert handler.metrics["accepted"] == 0
        assert handler.metrics["declined"] == 1

    def test_an_oversized_form_is_declined_rather_than_marched_through(self):
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params(
            "everything about you",
            {"properties": {f"field_{i}": {"type": "string"} for i in range(20)}},
        )
        asked: list = []

        with _input_callback(lambda question, choices: asked.append(question) or "x"):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "decline"
        assert asked == []


    def test_schema_read_from_real_sdk_params_reaches_the_summary(self):
        """The requested schema must be read off the *real* SDK model.

        Every other test here builds a duck-typed ``SimpleNamespace``, which
        cannot catch a field rename in the SDK — and 2.0 renamed this field
        (``requestedSchema`` -> ``requested_schema``). Pinning one case to the
        actual model is what proves the elicitation path still reads the
        schema after the migration, rather than silently summarising an empty
        one.
        """
        from mcp.types import ElicitRequestFormParams

        params = ElicitRequestFormParams(
            message="authorize a payment of $0.50",
            requested_schema={
                "type": "object",
                "properties": {"card_number": {"type": "string"}},
            },
        )
        handler = ElicitationHandler("pay", {"timeout": 5})
        asked: list = []

        with _input_callback(lambda question, choices: asked.append(question) or "4111"):
            asyncio.run(handler(context=None, params=params))

        assert any("card_number" in question for question in asked), asked

    def test_cancel_propagates_through(self):
        """request_elicitation_consent returns 'cancel' when the gateway
        wait times out (resolved=False). The handler should propagate
        that as ElicitResult(action='cancel') so the server can
        distinguish 'no answer' from 'no'."""
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params()

        with patch("tools.approval_prompt.request_elicitation_consent", return_value="cancel"):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "cancel"
        assert handler.metrics["errors"] == 1


class TestUrlMode:
    """Handing the user off to a page the server owns."""

    def test_the_destination_host_is_in_what_the_user_is_asked(self):
        """The domain IS the security decision. A user consenting to 'open a
        page' has been told nothing; a user consenting to open evil.example
        can refuse."""
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _url_params(url="https://checkout.stripe.com/pay/abc")
        asked: dict = {}

        def _capture(message, description, **kwargs):
            asked["message"] = message
            asked["description"] = description
            return "decline"

        with patch("tools.approval_prompt.request_elicitation_consent", _capture):
            asyncio.run(handler(context=None, params=params))

        assert "checkout.stripe.com" in asked["description"]

    def test_accepting_opens_the_browser(self):
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _url_params(url="https://accounts.google.com/o/oauth2/auth")

        with (
            patch("tools.approval_prompt.request_elicitation_consent", return_value="accept"),
            patch("webbrowser.open", return_value=True) as opened,
        ):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "accept"
        opened.assert_called_once_with("https://accounts.google.com/o/oauth2/auth")
        assert handler.metrics["accepted"] == 1

    def test_declining_opens_nothing(self):
        handler = ElicitationHandler("pay", {"timeout": 5})

        with (
            patch("tools.approval_prompt.request_elicitation_consent", return_value="decline"),
            patch("webbrowser.open") as opened,
        ):
            result = asyncio.run(handler(context=None, params=_url_params()))

        assert result.action == "decline"
        assert not opened.called
        assert handler.metrics["declined"] == 1

    def test_consent_is_still_honoured_when_no_browser_opens(self):
        """Headless: the user agreed and knows the address, so the server is
        told the hand-off happened rather than being sent a false refusal."""
        handler = ElicitationHandler("pay", {"timeout": 5})

        with (
            patch("tools.approval_prompt.request_elicitation_consent", return_value="accept"),
            patch("webbrowser.open", return_value=False),
        ):
            result = asyncio.run(handler(context=None, params=_url_params()))

        assert result.action == "accept"

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "not-a-url",
            "",
        ],
    )
    def test_an_unopenable_url_is_refused_without_asking(self, url):
        """No answer the user gives makes these safe, so they are not asked."""
        handler = ElicitationHandler("pay", {"timeout": 5})

        with (
            patch(
                "tools.approval_prompt.request_elicitation_consent",
                side_effect=AssertionError("must not prompt for an unusable URL"),
            ),
            patch("webbrowser.open") as opened,
        ):
            result = asyncio.run(handler(context=None, params=_url_params(url=url)))

        assert result.action == "decline"
        assert not opened.called


class TestElicitationHandlerFailureModes:
    def test_exception_in_approval_fails_closed_to_decline(self):
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _form_params()

        with patch(
            "tools.approval_prompt.request_elicitation_consent",
            side_effect=RuntimeError("approval system blew up"),
        ):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "decline"
        assert handler.metrics["errors"] == 1

    def test_timeout_returns_cancel(self, monkeypatch):
        # Shrink the outer grace window so the test budget is just the
        # handler timeout. Default grace is 5s, which makes stall durations
        # tight and the test flaky.
        monkeypatch.setattr(
            ElicitationHandler, "_OUTER_TIMEOUT_GRACE_SECONDS", 0
        )
        # _safe_numeric clamps `timeout` to a minimum of 1s, so the
        # effective wait_for budget is 1s here. Stall longer than that
        # so the wait_for reliably fires TimeoutError.
        handler = ElicitationHandler("pay", {"timeout": 0.05})
        params = _form_params()

        def stall(*_args, **_kwargs):
            import time as _t
            _t.sleep(2)
            return "accept"

        with patch("tools.approval_prompt.request_elicitation_consent", side_effect=stall):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "cancel"
        assert handler.metrics["errors"] == 1


class TestElicitationHandlerWiring:
    def test_session_kwargs_returns_callback(self):
        handler = ElicitationHandler("pay", {})
        kwargs = handler.session_kwargs()
        assert kwargs == {"elicitation_callback": handler}


    def test_disabled_config_does_not_construct_handler(self):
        """The server task initializer checks ``elicitation.enabled`` --
        an explicit ``False`` should suppress handler creation. The unit
        of that decision lives in MCPServerTask, but the handler itself
        must remain harmless to instantiate with arbitrary config."""
        handler = ElicitationHandler("pay", {"enabled": False, "timeout": 10})
        # Just confirm it instantiates and reads timeout; the gate lives
        # at the higher layer.
        assert handler.timeout == 10


class TestElicitationHandlerContextBridge:
    """The MCP recv-loop task that fires elicitation callbacks does NOT
    inherit the agent's contextvars (HERMES_SESSION_PLATFORM etc.). The
    handler reads ``owner._pending_call_context`` -- a snapshot captured
    by the MCP tool wrapper around ``session.call_tool`` -- and replays
    it before invoking the approval router so gateway-session detection
    survives the task hop. Regression tests for that bridge."""

    def test_captured_context_is_replayed_in_consent_call(self):
        """The captured context's contextvar values must be observable
        when ``request_elicitation_consent`` runs -- otherwise the
        gateway-platform detection in approval.py sees an empty platform
        string and falls back to the CLI path (the bug this fixes)."""
        import contextvars
        from types import SimpleNamespace

        probe: contextvars.ContextVar[str] = contextvars.ContextVar(
            "elicitation_test_probe", default=""
        )
        seen: list[str] = []

        def fake_consent(*_args, **_kwargs):
            seen.append(probe.get())
            return "accept"

        token = probe.set("gateway:telegram")
        try:
            captured = contextvars.copy_context()
        finally:
            probe.reset(token)
        assert probe.get() == "", (
            "Sanity check: the probe must be empty outside the captured "
            "context, otherwise the test would pass even without replay."
        )

        owner = SimpleNamespace(_pending_call_context=captured)
        handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)
        params = _form_params()

        with patch("tools.approval_prompt.request_elicitation_consent", side_effect=fake_consent):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "accept"
        assert seen == ["gateway:telegram"], (
            f"Expected the captured contextvar to be visible inside the "
            f"consent call; got {seen!r}"
        )

    def test_missing_captured_context_falls_back_to_direct_call(self):
        """Without an owner (or with an owner that hasn't entered a tool
        call) the handler must still invoke the consent router -- just
        without the contextvar replay. Otherwise CLI/TUI sessions, which
        don't set HERMES_SESSION_PLATFORM, would break."""
        handler = ElicitationHandler("pay", {"timeout": 5}, owner=None)
        params = _form_params()

        with patch("tools.approval_prompt.request_elicitation_consent", return_value="accept") as m:
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "accept"
        assert m.call_count == 1


    def test_pending_call_context_none_does_not_crash(self):
        """``owner._pending_call_context`` is set to None between tool
        calls. An elicitation arriving in that window must not crash."""
        from types import SimpleNamespace

        owner = SimpleNamespace(_pending_call_context=None)
        handler = ElicitationHandler("pay", {"timeout": 5}, owner=owner)
        params = _form_params()

        with patch("tools.approval_prompt.request_elicitation_consent", return_value="decline"):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "decline"


class TestRequestedSchemaFieldName:
    """The requested schema must be read off the *real* SDK model.

    Every other test in this file builds a duck-typed ``SimpleNamespace``
    stand-in for the params object. That keeps them cheap, but it means none
    of them can catch the handler reading a field name the SDK model does not
    actually have -- the stand-in simply has whatever name the test wrote.

    The SDK spells this field ``requestedSchema`` on mcp 1.x and
    ``requested_schema`` on 2.0 (which renamed model fields to snake_case and
    kept camelCase only as a serialization alias, which pydantic does not
    expose to attribute access). Constructing with the camelCase spelling
    works on both -- 2.0 accepts it as the alias -- so this test pins the
    behaviour to the real model on whichever SDK is installed.
    """

    def test_real_sdk_params_schema_reaches_the_consent_description(self):
        from mcp.types import ElicitRequestFormParams

        params = ElicitRequestFormParams(
            message="authorize a payment of $0.50",
            requestedSchema={
                "type": "object",
                "properties": {
                    "card_number": {
                        "type": "string",
                        "description": "card to charge",
                    },
                },
            },
        )
        handler = ElicitationHandler("pay", {"timeout": 5})
        asked: list = []

        # A schema we failed to read is an empty one, which asks nothing at
        # all — so the field reaching the question is the proof.
        with _input_callback(lambda question, choices: asked.append(question) or "4111"):
            asyncio.run(handler(context=None, params=params))

        assert any("card_number" in question for question in asked), asked
