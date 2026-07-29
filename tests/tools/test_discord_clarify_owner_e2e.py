"""End-to-end test for Discord rich-clarify session-owner authorization (#8).

Exercises the REAL production path that carries the session initiator's
identity from registration through to a resolved rich response:

  register(session_owner_user_id=<source.user_id>)   # gateway _clarify_callback_sync
    -> DiscordAdapter.send_clarify(options=..., auth_policy="session_owner_only")
      -> REAL InteractivePromptView(origin_user_id=<owner>)
        -> button.callback(interaction)               # real _resolve_choice
          -> resolve_gateway_clarify(prompt_id, json)
            -> wait_for_response(...)                  # what the agent thread reads

The sibling suites already cover the pieces in isolation:

  * ``tests/tools/test_clarify_rich_options.py`` asserts the owner field is
    persisted on the entry and that ``_clarify_callback_sync`` carries the
    kwarg (an AST structural check, not a runtime one).
  * ``tests/tools/test_discord_interactive_views.py`` unit-tests
    ``InteractivePromptView._check_auth`` directly.
  * ``tests/gateway/test_discord_clarify_buttons.py`` verifies
    ``send_clarify`` threads ``origin_user_id`` into the view — but with the
    view STUBBED (the gateway conftest installs a discord mock that lacks
    ``discord.ui.Modal``, so the real view class is never defined there), so
    the real button callback and resolution never run.

This file closes the gap.  It lives under ``tests/tools/`` (alongside
``test_discord_interactive_views.py``) because only here — without the
gateway conftest's discord mock — is the REAL ``discord.ui.View`` available.
It drives the real ``InteractivePromptView`` button callback (the exact
entry point discord.py invokes on a click) and reads back the actual JSON
envelope the agent would receive via ``wait_for_response``.  That proves
the owner is admitted end to end and that a real rich response resolves —
not just that a constructor received a keyword argument.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# The real InteractivePromptView subclasses discord.ui.View, so the real
# discord.py must be importable.  Skip cleanly when it isn't installed so
# this file never crashes under an environment without the messaging extra.
def _real_discord_available() -> bool:
    mod = sys.modules.get("discord")
    if mod is not None and hasattr(mod, "__file__"):
        return True
    try:
        import discord  # noqa: F401

        return hasattr(discord, "__file__")
    except ImportError:
        return False


discord = pytest.importorskip("discord")
if not _real_discord_available():
    pytest.skip("real discord.py not installed", allow_module_level=True)

# Repo root importable (mirrors the sibling test files).
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from tools import clarify_gateway as cm  # noqa: E402
from tools.discord_interactive_views import InteractivePromptView  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_clarify_state() -> None:
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _make_adapter() -> "DiscordAdapter":
    """A DiscordAdapter wired to a mocked client/channel that records the
    sent message.  Only the Discord *client* is mocked — the view, the
    clarify primitive, and the resolution path are all the real production
    code."""
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = DiscordAdapter(config)
    adapter._client = MagicMock()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    channel = MagicMock()
    sent_msg = MagicMock()
    sent_msg.id = 424242
    channel.send = AsyncMock(return_value=sent_msg)
    adapter._client.get_channel = MagicMock(return_value=channel)
    # Expose the channel so a test can grab the view off channel.send.
    adapter._channel = channel
    return adapter


def _make_interaction(*, user_id: str, display_name: str = "Tester") -> Any:
    """A mock discord.Interaction carrying the attributes the real
    ``InteractivePromptView._resolve_choice`` reads: ``user.id``,
    ``user.display_name``, ``response.edit_message`` / ``send_message``,
    and ``message.embeds``."""
    user = SimpleNamespace(id=int(user_id), display_name=display_name)
    embed = SimpleNamespace(color=None, set_footer=MagicMock())
    message = SimpleNamespace(embeds=[embed])
    response = SimpleNamespace(
        edit_message=AsyncMock(),
        send_message=AsyncMock(),
        defer=AsyncMock(),
    )
    return SimpleNamespace(user=user, response=response, message=message)


def _register_as_gateway(
    *,
    clarify_id: str,
    session_key: str,
    question: str,
    options: list,
    owner_user_id: Optional[str],
) -> None:
    """Mirror exactly the ``register(...)`` call inside
    ``gateway/run.py::_clarify_callback_sync``: the owner comes from
    ``source.user_id`` in closure scope, threaded as
    ``session_owner_user_id``.

    Every rich clarify is gated by ``session_owner_only`` — this whole
    module exists to prove that policy's owner fast-path — so the policy
    is fixed here rather than parameterised.
    """
    source = SimpleNamespace(user_id=owner_user_id)
    cm.register(
        clarify_id=clarify_id,
        session_key=session_key,
        question=question,
        choices=None,
        options=options,
        display_type="buttons",
        auth_policy="session_owner_only",
        session_owner_user_id=(
            str(source.user_id) if source and source.user_id else None
        ),
    )


_OPTIONS = [
    {"label": "Approve", "value": "yes", "style": "success"},
    {"label": "Reject", "value": "no", "style": "danger"},
]


async def _send_and_click(
    *,
    adapter: "DiscordAdapter",
    clarify_id: str,
    session_key: str,
    options: list,
    owner_user_id: Optional[str],
    clicker_user_id: str,
    button_index: int = 0,
) -> tuple[InteractivePromptView, Any, Optional[str]]:
    """Drive the full production path and return
    ``(view, interaction, resolved_response)``.

    * ``resolved_response`` is the actual string the agent thread would
      receive from ``wait_for_response`` — ``None`` when the click was
      rejected (the entry stays pending, exactly as in production).
    """
    _register_as_gateway(
        clarify_id=clarify_id,
        session_key=session_key,
        question="Deploy to production?",
        options=options,
        owner_user_id=owner_user_id,
    )

    result = await adapter.send_clarify(
        chat_id="9001",
        question="Deploy to production?",
        choices=None,
        clarify_id=clarify_id,
        session_key=session_key,
        options=options,
        display_type="buttons",
        auth_policy="session_owner_only",
        timeout_seconds=900,
    )
    assert result.success is True, f"send_clarify failed: {result.error}"

    # The real InteractivePromptView is attached to the sent message.
    sent_kwargs = adapter._channel.send.call_args.kwargs
    view = sent_kwargs["view"]
    assert isinstance(view, InteractivePromptView), (
        f"expected the REAL InteractivePromptView, got {type(view).__name__}; "
        "send_clarify must not stub the view on the rich path"
    )

    interaction = _make_interaction(user_id=clicker_user_id)
    # This is the exact entry point discord.py invokes when a button is
    # clicked: the button's ``callback`` coroutine, with the interaction.
    await view.children[button_index].callback(interaction)

    # Read back what the agent thread would receive.  The entry is only
    # resolved when the click was authorized; a rejected click leaves it
    # pending, so wait_for_response would block — inspect the entry
    # directly to distinguish "resolved" from "rejected".
    with cm._lock:
        entry = cm._entries.get(clarify_id)
    if entry is not None and entry.event.is_set():
        # Resolved — drain it the way the agent thread does so the return
        # value is the real response string.
        response = cm.wait_for_response(clarify_id, timeout=0.1)
        return view, interaction, response
    return view, interaction, None


# ===========================================================================
# End-to-end: session owner authorization through the real view
# ===========================================================================


class TestDiscordRichClarifyOwnerAuthE2E:
    """The session initiator is admitted through the real production path
    and a real rich response resolves; everyone else is rejected when no
    allowlist applies; union semantics hold when one does."""

    def setup_method(self) -> None:
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_owner_admitted_and_resolves_real_rich_response(self) -> None:
        """AC1 + AC2 + AC5: the initiating user (captured at registration
        from ``source.user_id``) is carried into the real Discord prompt
        and can answer under ``session_owner_only`` with NO static
        allowlist.  The response read back is the actual rich JSON envelope
        carrying the chosen option's value — not a copied registration
        field or a source-structure assertion."""
        adapter = _make_adapter()  # no static allowlist

        view, interaction, response = await _send_and_click(
            adapter=adapter,
            clarify_id="cid-owner-e2e",
            session_key="sk-owner-e2e",
            options=_OPTIONS,
            owner_user_id="42",  # the session initiator
            clicker_user_id="42",  # same user clicks
            button_index=0,  # "Approve"
        )

        # Owner was authorized: edit_message (success path), not the
        # ephemeral rejection send_message.
        interaction.response.edit_message.assert_called_once()
        interaction.response.send_message.assert_not_called()

        # A real rich response resolved end to end.
        assert response is not None, "owner click should resolve the prompt"
        parsed = json.loads(response)
        assert parsed["status"] == "answered"
        assert parsed["value"] == "yes"  # the option's value, not its label
        assert parsed["label"] == "Approve"
        assert parsed["user_id"] == "42"
        # The view is locked after resolution.
        assert view.resolved is True
        assert all(getattr(c, "disabled", False) for c in view.children)

    @pytest.mark.asyncio
    async def test_non_owner_rejected_without_allowlist(self) -> None:
        """AC3: a user who is neither the owner nor in any allowlist is
        rejected, and the prompt stays pending (no resolution lands for
        the agent thread)."""
        adapter = _make_adapter()  # no static allowlist

        view, interaction, response = await _send_and_click(
            adapter=adapter,
            clarify_id="cid-non-owner",
            session_key="sk-non-owner",
            options=_OPTIONS,
            owner_user_id="42",  # initiator
            clicker_user_id="99",  # someone else
            button_index=1,  # "Reject"
        )

        # Rejected via ephemeral message; no edit, no resolution.
        interaction.response.send_message.assert_called_once()
        assert interaction.response.send_message.call_args.kwargs.get("ephemeral") is True
        interaction.response.edit_message.assert_not_called()
        assert response is None, "non-owner click must not resolve the prompt"
        assert view.resolved is False
        # Entry remains pending for the real owner to answer later.
        with cm._lock:
            entry = cm._entries["cid-non-owner"]
        assert entry is not None
        assert not entry.event.is_set()

    @pytest.mark.asyncio
    async def test_no_owner_no_allowlist_fails_closed(self) -> None:
        """The documented fail-closed invariant: with no owner AND no
        allowlist, ``session_owner_only`` rejects everyone — including the
        initiator.  This is the regression that motivated threading the
        owner in the first place; the E2E path must preserve it."""
        adapter = _make_adapter()

        view, interaction, response = await _send_and_click(
            adapter=adapter,
            clarify_id="cid-fail-closed",
            session_key="sk-fail-closed",
            options=_OPTIONS,
            owner_user_id=None,  # owner never captured (anonymous platform)
            clicker_user_id="42",  # the person who started the turn
            button_index=0,
        )

        assert response is None, "fail-closed must not admit anyone"
        interaction.response.send_message.assert_called_once()
        assert interaction.response.send_message.call_args.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_owner_and_allowlist_union_semantics(self) -> None:
        """AC4: when a static allowlist IS configured, the owner fast-path
        and the allowlist gate independently (union).  An allowlisted user
        is admitted regardless of ownership; the owner is admitted
        regardless of the allowlist; an unrelated user is rejected.

        Uses THREE prompts (one per principal) so each authorization
        decision is observed on a fresh, unresolved view — mirroring how
        distinct clarify prompts behave in production.
        """
        allowed = {"1000"}  # a configured allowlist user
        adapter = _make_adapter()
        adapter._allowed_user_ids = set(allowed)
        # The session initiator is someone NOT in the allowlist.
        owner = "7"

        # (a) allowlisted user (not the owner) is admitted via the allowlist.
        _, ix_allowed, resp_allowed = await _send_and_click(
            adapter=adapter,
            clarify_id="cid-union-allowed",
            session_key="sk-union-allowed",
            options=_OPTIONS,
            owner_user_id=owner,
            clicker_user_id="1000",
            button_index=0,
        )
        assert resp_allowed is not None, "allowlisted user must be admitted"
        assert json.loads(resp_allowed)["user_id"] == "1000"
        ix_allowed.response.edit_message.assert_called_once()

        # (b) the owner (not in the allowlist) is admitted via the owner
        # fast-path — proving the allowlist is not the only gate.
        _, ix_owner, resp_owner = await _send_and_click(
            adapter=adapter,
            clarify_id="cid-union-owner",
            session_key="sk-union-owner",
            options=_OPTIONS,
            owner_user_id=owner,
            clicker_user_id=owner,
            button_index=0,
        )
        assert resp_owner is not None, "owner must be admitted via the fast-path"
        assert json.loads(resp_owner)["user_id"] == owner
        ix_owner.response.edit_message.assert_called_once()

        # (c) an unrelated user is rejected by BOTH gates.
        _, ix_other, resp_other = await _send_and_click(
            adapter=adapter,
            clarify_id="cid-union-other",
            session_key="sk-union-other",
            options=_OPTIONS,
            owner_user_id=owner,
            clicker_user_id="9999",
            button_index=0,
        )
        assert resp_other is None, "unrelated user must be rejected"
        ix_other.response.send_message.assert_called_once()
        assert ix_other.response.send_message.call_args.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_owner_carried_per_prompt_from_registration(self) -> None:
        """AC1 (focused): the identity captured at registration
        (``source.user_id``) is the identity the real view compares
        against.  Register with owner "42", click as "42" -> admitted;
        on a FRESH prompt registered with owner "55", clicking as "42" is
        rejected.  This proves the owner is carried per-prompt from the
        production registration call, not from a global or a stub."""
        adapter = _make_adapter()

        # Prompt owned by 42 -> 42 admitted.
        _, _, resp_a = await _send_and_click(
            adapter=adapter,
            clarify_id="cid-identity-a",
            session_key="sk-identity-a",
            options=_OPTIONS,
            owner_user_id="42",
            clicker_user_id="42",
            button_index=0,
        )
        assert resp_a is not None

        # Fresh prompt owned by 55 -> 42 is now the non-owner and rejected.
        _, ix_b, resp_b = await _send_and_click(
            adapter=adapter,
            clarify_id="cid-identity-b",
            session_key="sk-identity-b",
            options=_OPTIONS,
            owner_user_id="55",
            clicker_user_id="42",
            button_index=0,
        )
        assert resp_b is None
        ix_b.response.send_message.assert_called_once()
