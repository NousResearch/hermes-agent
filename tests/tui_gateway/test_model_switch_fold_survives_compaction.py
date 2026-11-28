"""After a model switch, a compaction that carries the user's next turn must keep that turn on screen.

The switch notice is a ``role=user`` row (#48338), so the turn-start alternation repair folds it with the prompt
right after it. The fold used to write the notice text first into ``content``. A compaction that carried the turn
then stored that composite as the turn's only live row: ``[System: …]`` first and no display kind, so the
Desktop's projection hid it as a gateway notice. A summary compaction also rewinds the prompt's own row as a
carried original, so the prompt vanished from the thread while the model still read it.
"""

from __future__ import annotations

import copy

from agent.context_compressor import SUMMARY_PREFIX, stamp_db_persisted_markers
from agent.conversation_compression_archive import ABSORBED_ROW_IDS, coverage_for_commit
from agent.message_metadata import stamp_message_uid
from hermes_cli.web_routers.sessions import _project_for_display
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message, archive_and_compact, deactivate_messages_by_display_kind, get_active_message_watermark,
    get_message_role, get_messages, get_resume_conversations,
)
from persistence.sqlite.conversation._store.sessions import create_session
from session_rules.turns import repair_message_sequence
from tui_gateway.server import _append_model_switch_marker

SID = "chat"
PROMPT = "1. I've sent him the API keys."
CALL = [{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]


class _Roles:
    """The ``get_message_role`` read ``coverage_for_commit`` makes on a store without the interface handle."""

    def __init__(self, db: StoreHolder) -> None:
        self.db = db

    def get_message_role(self, session_id: str, row_id: int):
        return get_message_role(self.db, session_id, row_id)


def _durable(db: StoreHolder, live: list, message: dict, **row) -> None:
    message["_row_id"] = append_message(db, SID, message["role"], message["content"],
                                        message_uid=stamp_message_uid(message), **row)
    message["_db_persisted"] = True
    live.append(message)


def _switch_then_prompt(tmp_path) -> tuple[StoreHolder, list]:
    """A thread the Desktop reopened, a switch to model-b, the user's prompt, and the turn-start repair."""
    db = StoreHolder(tmp_path / "state.db")
    create_session(db, SID, source="desktop")
    t = 1795700000.0
    append_message(db, SID, "user", "what does he need?", timestamp=t)
    append_message(db, SID, "assistant", "Two things.", timestamp=t + 1, finish_reason="stop")
    append_message(db, SID, "user", "which first?", timestamp=t + 1.5)
    append_message(db, SID, "assistant", "The key.", timestamp=t + 2, finish_reason="stop")
    session = {"session_key": SID, "history": get_resume_conversations(db, SID)[0]}
    # _append_model_switch_marker: the real in-memory half, and the durable half its persistence route writes.
    _append_model_switch_marker(session, model="model-b", provider="p")
    deactivate_messages_by_display_kind(db, SID, "model_switch")
    live = session["history"]
    notice = live.pop()
    _durable(db, live, notice, display_kind="model_switch", timestamp=t + 3)
    _durable(db, live, {"role": "user", "content": PROMPT, "timestamp": t + 4}, timestamp=t + 4)
    assert repair_message_sequence(None, live) == 1  # turn start: the notice and the prompt fold
    # The turn runs a tool before the compaction triggers (post_tool).
    _durable(db, live, {"role": "assistant", "content": "", "tool_calls": CALL, "timestamp": t + 5},
             tool_calls=CALL, timestamp=t + 5, finish_reason="tool_calls")
    _durable(db, live, {"role": "tool", "content": "ok", "tool_call_id": "call_1", "timestamp": t + 6},
             tool_call_id="call_1", timestamp=t + 6)
    return db, live


def _commit(db: StoreHolder, live: list, compacted: list, **kwargs) -> list:
    covered, unresolved = coverage_for_commit(_Roles(db), SID, live)
    archive_and_compact(db, SID, compacted, watermark=get_active_message_watermark(db, SID), covered_ids=covered,
                        unresolved_held=unresolved, **kwargs)
    stamp_db_persisted_markers(compacted)  # the post-commit stamp every compaction path makes
    return compacted


def _summary_compaction(db: StoreHolder, live: list, keep: int) -> list:
    """The in-place summary commit (conversation_compression._commit_compaction): a summary row, then the
    newest *keep* messages carried verbatim; tail_count also counts each carried dict's absorbed live rows."""
    tail = [copy.deepcopy(m) for m in live[-keep:]]
    absorbed = {r for m in tail for r in m.get(ABSORBED_ROW_IDS) or () if get_message_role(db, SID, r) is not None}
    summary = {"role": "user", "content": f"{SUMMARY_PREFIX}\n\nEarlier turns.", "_compressed_summary": True}
    return _commit(db, live, [summary, *tail], tail_count=keep + len(absorbed))


def _full_rewrite(db: StoreHolder, live: list) -> list:
    """The proactive prune's fallback commit: the whole held list rewritten as a new generation."""
    return _commit(db, live, copy.deepcopy(live), held_for_cas=live)


def _shown(db: StoreHolder) -> list:
    """The bubbles the Desktop renders: the REST projection over the display read."""
    return [(m["role"], m["content"]) for m in _project_for_display(get_messages(db, SID, include_compacted=True))
            if m["role"] in ("user", "assistant") and m["content"] and not m.get("display_kind")]


def _model_user_turns(messages: list) -> list:
    """The text each user row sends the model (the replay substitutes ``api_content`` for ``content``)."""
    return [m.get("api_content") or m["content"] for m in messages if m["role"] == "user"]


THREAD = [("user", "what does he need?"), ("assistant", "Two things."), ("user", "which first?"),
          ("assistant", "The key."), ("user", PROMPT)]


def _assert_prompt_once(db: StoreHolder, live: list) -> None:
    for turns in (_model_user_turns(live), _model_user_turns(get_resume_conversations(db, SID)[0])):
        carrying = [text for text in turns if PROMPT in text]
        assert len(carrying) == 1 and "model-b" in carrying[0], turns


def test_summary_compaction_after_a_switch_keeps_the_prompt_on_screen(tmp_path) -> None:
    db, live = _switch_then_prompt(tmp_path)

    live = _summary_compaction(db, live, keep=4)  # "The key.", the folded turn, the tool call, its result

    assert _shown(db) == THREAD
    _assert_prompt_once(db, live)


def test_full_rewrite_then_summary_shows_the_prompt_once(tmp_path) -> None:
    db, live = _switch_then_prompt(tmp_path)

    live = _full_rewrite(db, live)
    assert _shown(db) == THREAD
    live = _summary_compaction(db, live, keep=4)

    assert _shown(db) == THREAD
    _assert_prompt_once(db, live)


def test_fold_keeps_the_prompts_typed_ask_stamp() -> None:
    """The prompt's content is unchanged by the fold (the notice rides its wire sidecar), so it is still the
    one ask the user typed and keeps the stamp ``previous_message`` reads."""
    from session_rules.turns import TYPED_ASK

    marker = {"role": "user", "content": "[System: The active model for this chat has changed to m.]",
              "display_kind": "model_switch", "_row_id": 1}
    prompt = {"role": "user", "content": PROMPT, "_row_id": 2, "display_metadata": {TYPED_ASK: PROMPT}}
    history = [{"role": "assistant", "content": "ok"}, marker, prompt]
    assert repair_message_sequence(None, history) == 1
    assert history[-1] is prompt and prompt["content"] == PROMPT
    assert prompt["display_metadata"][TYPED_ASK] == PROMPT
    assert prompt["api_content"].startswith(marker["content"]) and prompt["_absorbed_row_ids"] == [1]


def _notice(model: str) -> str:
    return f"[System: The active model for this chat has changed to {model}.]"


def _marker(model: str, row_id: int | None = None) -> dict:
    marker = {"role": "user", "content": _notice(model), "display_kind": "model_switch"}
    if row_id is not None:
        marker["_row_id"] = row_id
    return marker


def test_fold_keeps_an_injected_sidecar_after_the_notice() -> None:
    """A turn whose wire already carries injected context (memory, plugins) keeps it behind the notice."""
    prompt = {"role": "user", "content": PROMPT, "api_content": f"{PROMPT}\n\n<memory>", "_row_id": 2}
    history = [{"role": "assistant", "content": "ok"}, _marker("m", 1), prompt]
    assert repair_message_sequence(None, history) == 1
    assert prompt["content"] == PROMPT
    assert prompt["api_content"] == f"{_notice('m')}\n\n{PROMPT}\n\n<memory>"


def test_refolding_the_same_rows_does_not_stack_the_notice() -> None:
    """A reload folds the stored rows again: a turn whose stored sidecar already leads with the notice."""
    prompt = {"role": "user", "content": PROMPT, "api_content": f"{_notice('m')}\n\n{PROMPT}", "_row_id": 2}
    history = [{"role": "assistant", "content": "ok"}, _marker("m", 1), prompt]
    assert repair_message_sequence(None, history) == 1
    assert prompt["api_content"] == f"{_notice('m')}\n\n{PROMPT}"


def test_a_newer_switch_drops_the_older_notice_from_the_wire() -> None:
    """Only the newest notice names the active model (#65891): the older fold's notice leaves its turn's wire."""
    first = {"role": "user", "content": "first ask", "_row_id": 2}
    second = {"role": "user", "content": PROMPT, "_row_id": 5}
    history = [_marker("model-a", 1), first, {"role": "assistant", "content": "ok"}]
    assert repair_message_sequence(None, history) == 1
    assert first["api_content"].startswith(_notice("model-a"))
    history += [_marker("model-b", 4), second]
    assert repair_message_sequence(None, history) == 1
    assert "api_content" not in first and first["content"] == "first ask"
    assert second["api_content"] == f"{_notice('model-b')}\n\n{PROMPT}"
    assert [m["content"] for m in history] == ["first ask", "ok", PROMPT]


def test_a_personality_notice_folds_into_the_sidecar_too() -> None:
    """A personality notice lives only in memory (no row); it rides the turn's wire like a switch notice."""
    notice = "[System: The personality for this chat has changed to concise.]"
    prompt = {"role": "user", "content": PROMPT, "_row_id": 2}
    history = [{"role": "assistant", "content": "ok"},
               {"role": "user", "content": notice, "display_kind": "personality_switch"}, prompt]
    assert repair_message_sequence(None, history) == 1
    assert history[-1] is prompt and prompt["content"] == PROMPT
    assert prompt["api_content"] == f"{notice}\n\n{PROMPT}"


def test_two_markers_then_a_prompt_keep_the_prompt_and_retire_both_rows() -> None:
    """Two consecutive markers text-fold first (no plain row in them); the pair then folds into the prompt."""
    personality = {"role": "user", "content": "[System: The personality for this chat has changed to concise.]",
                   "display_kind": "personality_switch", "_row_id": 1}
    prompt = {"role": "user", "content": PROMPT, "_row_id": 3}
    history = [{"role": "assistant", "content": "ok"}, personality, _marker("m", 2), prompt]
    repair_message_sequence(None, history)
    assert history[-1] is prompt and prompt["content"] == PROMPT
    assert personality["content"] in prompt["api_content"] and _notice("m") in prompt["api_content"]
    assert prompt["api_content"].endswith(PROMPT)
    assert sorted(prompt["_absorbed_row_ids"]) == [1, 2]


def test_a_newer_switch_drops_an_older_notice_behind_a_personality_notice() -> None:
    """A personality notice leading the sidecar must not shield a stale model-switch notice behind it."""
    persona = "[System: The personality for this chat has changed to concise.]"
    first = {"role": "user", "content": "first ask", "_row_id": 3}
    history = [{"role": "user", "content": persona, "display_kind": "personality_switch"}, _marker("model-a", 2),
               first, {"role": "assistant", "content": "ok"}]
    repair_message_sequence(None, history)
    assert first["api_content"] == f"{persona}\n\n{_notice('model-a')}\n\nfirst ask"
    second = {"role": "user", "content": PROMPT, "_row_id": 5}
    history += [_marker("model-b", 4), second]
    repair_message_sequence(None, history)
    assert first["api_content"] == f"{persona}\n\nfirst ask"
    assert second["api_content"] == f"{_notice('model-b')}\n\n{PROMPT}"


def test_a_plain_turn_merged_after_a_fold_keeps_the_notice() -> None:
    """Two consecutive plain turns after a marker: the text merge keeps the folded notice ahead of both."""
    history = [{"role": "assistant", "content": "ok"}, _marker("m", 1),
               {"role": "user", "content": "part one", "_row_id": 2}, {"role": "user", "content": "part two", "_row_id": 3}]
    repair_message_sequence(None, history)
    turn = history[-1]
    assert turn["content"] == "part one\n\npart two"
    assert turn["api_content"] == f"{_notice('m')}\n\npart one\n\npart two"


def test_the_fold_never_strips_the_users_own_text() -> None:
    """A prompt whose own text starts like a notice keeps it on the wire; only a stored sidecar is stripped."""
    own = f"{_notice('fake')}\n\nreal question"
    prompt = {"role": "user", "content": own, "_row_id": 2}
    history = [{"role": "assistant", "content": "ok"}, _marker("m", 1), prompt]
    repair_message_sequence(None, history)
    assert prompt["content"] == own
    assert prompt["api_content"] == f"{_notice('m')}\n\n{own}"


def test_an_older_turns_injected_text_survives_the_stale_notice_drop() -> None:
    """Dropping an older notice keeps that turn's injected context."""
    first = {"role": "user", "content": "first ask", "api_content": f"{_notice('a')}\n\nfirst ask\n\n<memory>"}
    history = [first, {"role": "assistant", "content": "ok"}, _marker("b", 4),
               {"role": "user", "content": PROMPT, "_row_id": 5}]
    repair_message_sequence(None, history)
    assert first["api_content"] == "first ask\n\n<memory>"


def test_the_turn_start_stamp_keeps_a_folded_notice_ahead_of_injected_context() -> None:
    """The prologue rebuilds the sidecar from content plus injected context; a folded notice stays in front."""
    import types

    from agent.turn_context import _stamp_api_content_sidecar

    folded = f"{_notice('m')}\n\n{PROMPT}"
    prompt = {"role": "user", "content": PROMPT, "api_content": folded}
    _stamp_api_content_sidecar(types.SimpleNamespace(), [prompt], 0, "", "PLUGIN CONTEXT", preflight_compressed=False)
    assert prompt["api_content"].startswith(f"{folded}\n\n") and "PLUGIN CONTEXT" in prompt["api_content"]
    untouched = {"role": "user", "content": PROMPT, "api_content": folded}
    _stamp_api_content_sidecar(types.SimpleNamespace(), [untouched], 0, "", "", preflight_compressed=False)
    assert untouched["api_content"] == folded


def test_the_turn_start_stamp_keeps_a_folded_notice_under_a_persist_override() -> None:
    """With a clean-transcript override and nothing injected, the folded sidecar is still what the wire sends."""
    import types

    from agent.turn_context import _stamp_api_content_sidecar

    folded = f"{_notice('m')}\n\nlive ask"
    prompt = {"role": "user", "content": "live ask", "api_content": folded}
    agent = types.SimpleNamespace(_persist_user_message_override="clean ask")
    _stamp_api_content_sidecar(agent, [prompt], 0, "", "", preflight_compressed=False)
    assert prompt["api_content"] == folded


def test_the_stamp_does_not_repeat_a_prompts_own_leading_system_line() -> None:
    """A prompt whose own text starts like a notice is not mistaken for a fold when the sidecar is rebuilt."""
    import types

    from agent.turn_context import _stamp_api_content_sidecar

    own = "[System: my own note]\n\nask"
    prompt = {"role": "user", "content": own, "api_content": f"{own}\n\nOLD CONTEXT"}
    _stamp_api_content_sidecar(types.SimpleNamespace(), [prompt], 0, "", "PLUGIN CONTEXT", preflight_compressed=False)
    assert prompt["api_content"].startswith(f"{own}\n\n") and prompt["api_content"].count("[System: my own note]") == 1
    assert "OLD CONTEXT" not in prompt["api_content"]


def test_the_stale_notice_drop_skips_entries_that_are_not_messages() -> None:
    from session_rules.turns import _drop_stale_switch_notices

    older = {"role": "user", "content": "a", "api_content": f"{_notice('a')}\n\na"}
    _drop_stale_switch_notices(["not a message", older])
    assert "api_content" not in older
