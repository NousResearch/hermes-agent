import time

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _use_home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    return token


def test_create_and_resolve_choice_round_trips_from_sqlite(tmp_path):
    token = _use_home(tmp_path)
    try:
        from tools.clarify_interaction import (
            create_clarify_interaction,
            get_interaction,
            resolve_choice,
        )

        created = create_clarify_interaction(
            session_key="telegram:chat:thread:user",
            platform="telegram",
            question="Deploy where?",
            choices=["staging", "production"],
            chat_id="123",
            thread_id="456",
            user_id="789",
            ttl_seconds=3600,
        )

        loaded = get_interaction(created.interaction_id)
        assert loaded is not None
        assert loaded.status == "pending"
        assert loaded.question == "Deploy where?"
        assert loaded.choices == ["staging", "production"]

        resolved = resolve_choice(
            created.interaction_id,
            1,
            user_id="789",
            chat_id="123",
            thread_id="456",
        )
        assert resolved.ok is True
        assert resolved.status == "resolved"
        assert resolved.answer == "production"

        loaded_again = get_interaction(created.interaction_id)
        assert loaded_again is not None
        assert loaded_again.status == "resolved"
        assert loaded_again.answer == "production"
    finally:
        reset_hermes_home_override(token)


def test_resolve_choice_is_idempotent_and_authorized(tmp_path):
    token = _use_home(tmp_path)
    try:
        from tools.clarify_interaction import create_clarify_interaction, resolve_choice

        created = create_clarify_interaction(
            session_key="s1",
            platform="telegram",
            question="Pick?",
            choices=["a", "b"],
            chat_id="chat",
            thread_id="topic",
            user_id="owner",
            ttl_seconds=3600,
        )

        wrong_user = resolve_choice(
            created.interaction_id,
            0,
            user_id="intruder",
            chat_id="chat",
            thread_id="topic",
        )
        assert wrong_user.ok is False
        assert wrong_user.status == "unauthorized"

        first = resolve_choice(
            created.interaction_id,
            0,
            user_id="owner",
            chat_id="chat",
            thread_id="topic",
        )
        second = resolve_choice(
            created.interaction_id,
            1,
            user_id="owner",
            chat_id="chat",
            thread_id="topic",
        )

        assert first.ok is True
        assert first.answer == "a"
        assert second.ok is False
        assert second.status == "already_resolved"
    finally:
        reset_hermes_home_override(token)


def test_awaiting_text_resolves_oldest_matching_session(tmp_path):
    token = _use_home(tmp_path)
    try:
        from tools.clarify_interaction import (
            create_clarify_interaction,
            get_interaction,
            mark_awaiting_text,
            resolve_text_for_session,
        )

        first = create_clarify_interaction(
            session_key="s1",
            platform="telegram",
            question="Explain?",
            choices=["short"],
            chat_id="chat",
            thread_id="topic",
            user_id="owner",
            ttl_seconds=3600,
        )
        second = create_clarify_interaction(
            session_key="s1",
            platform="telegram",
            question="Later?",
            choices=None,
            chat_id="chat",
            thread_id="topic",
            user_id="owner",
            ttl_seconds=3600,
        )
        assert mark_awaiting_text(first.interaction_id, user_id="owner", chat_id="chat", thread_id="topic").ok

        resolved = resolve_text_for_session(
            "s1",
            "custom answer",
            user_id="owner",
            chat_id="chat",
            thread_id="topic",
        )

        assert resolved is not None
        assert resolved.interaction_id == first.interaction_id
        assert resolved.answer == "custom answer"
        assert get_interaction(first.interaction_id).status == "resolved"
        assert get_interaction(second.interaction_id).status == "pending"
    finally:
        reset_hermes_home_override(token)


def test_cancel_interaction_only_cancels_one_row(tmp_path):
    token = _use_home(tmp_path)
    try:
        from tools.clarify_interaction import (
            cancel_interaction,
            create_clarify_interaction,
            get_interaction,
        )

        first = create_clarify_interaction(
            session_key="s1",
            platform="telegram",
            question="First?",
            choices=["a"],
            ttl_seconds=3600,
        )
        second = create_clarify_interaction(
            session_key="s1",
            platform="telegram",
            question="Second?",
            choices=["b"],
            ttl_seconds=3600,
        )

        assert cancel_interaction(first.interaction_id) is True
        assert get_interaction(first.interaction_id).status == "cancelled"
        assert get_interaction(second.interaction_id).status == "pending"
    finally:
        reset_hermes_home_override(token)


def test_expired_interaction_cannot_resolve(tmp_path):
    token = _use_home(tmp_path)
    try:
        from tools.clarify_interaction import create_clarify_interaction, resolve_choice

        created = create_clarify_interaction(
            session_key="s1",
            platform="telegram",
            question="Pick?",
            choices=["a"],
            user_id="owner",
            ttl_seconds=-1,
            now=time.time() - 10,
        )

        result = resolve_choice(created.interaction_id, 0, user_id="owner")
        assert result.ok is False
        assert result.status == "expired"
    finally:
        reset_hermes_home_override(token)
