"""Idempotent ``prompt.submit`` retries: one client submission id, one turn.

A client that loses the reply to ``prompt.submit`` — an app relaunch, a mobile network drop, a
reconnect mid-turn — resends the same words, and nothing on this surface told that retry apart
from a second intent: the session ran a second turn, with a second reply and a second bill. The
canonical session runtime already keys its admission on exactly this id
(``gateway/session_controls.py::submission`` → ``Submission.request_id``), and the sidecar already
answers a retried ``session.create`` / ``session.branch`` from a keyed registry (#65410). This is
the same registry discipline for prompts.

The window is deliberately a retry window, not a ledger: the reply is remembered for
``_TTL_SECONDS`` and dropped after it, so the user genuinely repeating the same words later gets a
new turn. A retry that arrives after the window, or after this process restarted, is admitted as a
new turn — the client's own id is its only defence there.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

#: How long an accepted submit keeps answering retries of its own id. Mirrors
#: ``server._IDEMPOTENCY_KEY_TTL`` (5 min): longer than any realistic reconnect, short enough that
#: the same words typed again much later are a new turn.
_TTL_SECONDS = 300.0
#: Bound on remembered replies, evicted oldest-first. A submit is remembered at most once, so this
#: caps memory, not correctness.
_MAX_ENTRIES = 256

_lock = threading.Lock()
_answers: OrderedDict[tuple[str, str], tuple[dict, float]] = OrderedDict()


def _key(session_id: str, submission_id: str) -> tuple[str, str]:
    # The key is the PAIR: two sessions may legitimately carry the same client id (a client that
    # mints ids per conversation, or a copied draft), and each must run its own turn.
    return (session_id, submission_id)


def replay(session_id: str, submission_id: str) -> dict | None:
    """The reply the first submit with this id received, flagged ``duplicate``; None if unseen."""
    key = _key(session_id, submission_id)
    with _lock:
        entry = _answers.get(key)
        if entry is None:
            return None
        answer, stamped = entry
        if time.monotonic() - stamped > _TTL_SECONDS:
            _answers.pop(key, None)
            return None
        # Refresh the window so back-to-back retries of one dropped reply cannot age out mid-retry.
        _answers[key] = (answer, time.monotonic())
        _answers.move_to_end(key)
        return {**answer, "duplicate": True}


def remember(session_id: str, submission_id: str, answer: dict) -> None:
    """Remember the acceptance body a turn's first submit returned.

    Only an accepted turn has a ``status``; the typed-stop-phrase reply (``voice_stopped``) admits
    nothing and is not remembered, and neither is a body that is already a replay.
    """
    if "status" not in answer or answer.get("duplicate"):
        return
    key = _key(session_id, submission_id)
    with _lock:
        _answers[key] = (dict(answer), time.monotonic())
        _answers.move_to_end(key)
        while len(_answers) > _MAX_ENTRIES:
            _answers.popitem(last=False)

