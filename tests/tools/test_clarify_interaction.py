"""Durable clarify interaction storage semantics."""

import time

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_choice_resolution_is_persisted_and_authorized(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import create_clarify_interaction, get_interaction, resolve_choice

        interaction = create_clarify_interaction(
            session_key="telegram:chat:thread:user",
            platform="telegram",
            question="Deploy where?",
            choices=["staging", "production"],
            chat_id="chat",
            thread_id="thread",
            user_id="user",
            ttl_seconds=3600,
        )
        denied = resolve_choice(
            interaction.interaction_id, 1, user_id="intruder", chat_id="chat", thread_id="thread"
        )
        assert denied.ok is False
        assert denied.status == "unauthorized"

        wrong_context = resolve_choice(
            interaction.interaction_id, 1, user_id="user", chat_id="other-chat", thread_id="thread"
        )
        assert wrong_context.ok is False
        assert wrong_context.status == "unauthorized"

        wrong_thread = resolve_choice(
            interaction.interaction_id, 1, user_id="user", chat_id="chat", thread_id="other-thread"
        )
        assert wrong_thread.ok is False
        assert wrong_thread.status == "unauthorized"

        resolved = resolve_choice(
            interaction.interaction_id, 1, user_id="user", chat_id="chat", thread_id="thread"
        )
        assert resolved.ok is True
        assert resolved.answer == "production"

        loaded = get_interaction(interaction.interaction_id)
        assert loaded is not None
        assert loaded.status == "resolved"

        duplicate = resolve_choice(
            interaction.interaction_id, 0, user_id="user", chat_id="chat", thread_id="thread"
        )
        assert duplicate.ok is False
        assert duplicate.status == "already_resolved"
    finally:
        reset_hermes_home_override(token)


def test_text_resolution_prefers_awaiting_text_and_skips_choice_prompts(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import (
            create_clarify_interaction,
            get_interaction,
            mark_awaiting_text,
            resolve_text_for_session,
        )

        choice_prompt = create_clarify_interaction(
            session_key="session",
            platform="telegram",
            question="Pick?",
            choices=["one", "two"],
            chat_id="chat",
            user_id="user",
            ttl_seconds=3600,
        )
        text_prompt = create_clarify_interaction(
            session_key="session",
            platform="telegram",
            question="Explain?",
            choices=None,
            chat_id="chat",
            user_id="user",
            ttl_seconds=3600,
        )
        assert mark_awaiting_text(text_prompt.interaction_id, user_id="user", chat_id="chat").ok

        resolved = resolve_text_for_session(
            "session", "custom answer", user_id="user", chat_id="chat"
        )
        assert resolved is not None
        assert resolved.interaction_id == text_prompt.interaction_id

        choice_loaded = get_interaction(choice_prompt.interaction_id)
        text_loaded = get_interaction(text_prompt.interaction_id)
        assert choice_loaded is not None
        assert text_loaded is not None
        assert choice_loaded.status == "pending"
        assert text_loaded.answer == "custom answer"
    finally:
        reset_hermes_home_override(token)


def test_actor_scoped_cancellation_does_not_cancel_other_user(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import cancel_session_for_actor, create_clarify_interaction, get_interaction

        owner = create_clarify_interaction(
            session_key="shared",
            platform="telegram",
            question="Owner?",
            choices=["yes"],
            chat_id="chat",
            thread_id="topic",
            user_id="owner",
            ttl_seconds=3600,
        )
        other = create_clarify_interaction(
            session_key="shared",
            platform="telegram",
            question="Other?",
            choices=["no"],
            chat_id="chat",
            thread_id="topic",
            user_id="other",
            ttl_seconds=3600,
        )

        assert cancel_session_for_actor(
            "shared", user_id="owner", chat_id="chat", thread_id="topic"
        ) == 1
        owner_loaded = get_interaction(owner.interaction_id)
        other_loaded = get_interaction(other.interaction_id)
        assert owner_loaded is not None
        assert other_loaded is not None
        assert owner_loaded.status == "cancelled"
        assert other_loaded.status == "pending"
    finally:
        reset_hermes_home_override(token)


def test_expired_choice_cannot_be_resolved(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import create_clarify_interaction, resolve_choice

        interaction = create_clarify_interaction(
            session_key="session",
            platform="telegram",
            question="Pick?",
            choices=["one"],
            user_id="user",
            ttl_seconds=-1,
            now=time.time() - 10,
        )
        result = resolve_choice(interaction.interaction_id, 0, user_id="user")
        assert result.ok is False
        assert result.status == "expired"
    finally:
        reset_hermes_home_override(token)
