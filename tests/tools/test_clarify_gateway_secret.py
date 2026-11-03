"""The clarify registry's side of a secret prompt: who may answer it, what a busy session hands
it, and the short window after it ends unanswered, during which its owner's next message is taken
as a late secret (``gateway/run_inbound_secret.py`` deletes it instead of routing it)."""

import pytest

from tools import clarify_gateway as cm

SESSION = "agent:main:telegram:group:-100777:42"


@pytest.fixture(autouse=True)
def _clean_registry():
    def clear():
        with cm._lock:
            cm._entries.clear()
            cm._session_index.clear()
            cm._secret_graces.clear()
    clear()
    yield
    clear()


@pytest.fixture
def later(monkeypatch):
    """Move the registry's clock forward by ``seconds`` (after any wait has returned)."""
    real = cm.time.monotonic

    def jump(seconds):
        monkeypatch.setattr(cm.time, "monotonic", lambda: real() + seconds)
    return jump


def _secret(cid="cl-secret", owner="1001"):
    return cm.register(cid, SESSION, "password?", None, secret=True, owner_user_id=owner)


def test_an_unanswered_secret_prompt_opens_a_window_for_its_owner_only():
    _secret()
    assert cm.wait_for_response("cl-secret", 0.01) is None
    assert cm.take_secret_grace(SESSION, "2002") is False
    assert cm.take_secret_grace(SESSION, "1001") is True
    assert cm.take_secret_grace(SESSION, "1001") is False, "one message, once"


def test_the_window_closes(later):
    _secret()
    cm.wait_for_response("cl-secret", 0.01)
    later(cm.SECRET_GRACE_S + 1)
    assert cm.take_secret_grace(SESSION, "1001") is False


def test_no_window_after_an_answer_or_a_public_prompt():
    _secret()
    cm.resolve_gateway_clarify("cl-secret", "hunter2")
    cm.wait_for_response("cl-secret", 0.01)
    cm.register("cl-public", SESSION, "email?", None, owner_user_id="1001")
    cm.wait_for_response("cl-public", 0.01)
    assert cm.take_secret_grace(SESSION, "1001") is False


def test_a_new_prompt_supersedes_the_window():
    _secret("cl-1")
    cm.wait_for_response("cl-1", 0.01)
    _secret("cl-2")
    assert cm.take_secret_grace(SESSION, "1001") is False


def test_a_busy_session_hands_slash_text_to_a_secret_prompt_only():
    cm.register("cl-open", SESSION, "which file?", None)
    assert cm.pending_takes_message(SESSION, is_command=False) is True
    assert cm.pending_takes_message(SESSION, is_command=True) is False
    cm.clear_session(SESSION)
    _secret()
    assert cm.pending_takes_message(SESSION, is_command=True) is True
