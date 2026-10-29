"""``session_search(ref=...)``: exact recovery of one original message by its ``message_uid`` prefix.

Compaction summaries and demoted-tool stubs carry ``m:<12hex>`` refs. The ref shape resolves to the
EARLIEST row carrying that uid (the original, in its original position), whatever its visibility flags,
and returns it verbatim with a window of neighbours. It resolves in ONE store only, like the read shape.
"""

import copy
import inspect
import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search

BIG_OUTPUT = ("build line\n" * 400) + "fatal: EXACT_NEEDLE_4242"


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _rows(db, sid):
    return [dict(r) for r in db._conn.execute(
        "SELECT id, role, content, message_uid, active, compacted FROM messages WHERE session_id = ? ORDER BY id",
        (sid,)).fetchall()]


def _compacted_session(db, sid="s_live"):
    """In-place compaction with a carried tail whose tool output was demoted to a stub.

    Rows 0-3 are summarized (``active=0, compacted=1``); the carried tail's originals (the assistant
    call and its full tool result) become superseded duplicates (``active=0, compacted=0``) while
    their live copies keep the uids, the tool copy holding only a stub.
    """
    db.create_session(sid, source="cli")
    db.append_message(sid, role="user", content="first question about the build")
    db.append_message(sid, role="assistant", content="first answer")
    db.append_message(sid, role="user", content="second question")
    db.append_message(sid, role="assistant", content="compacted detail COMPACTED_NEEDLE_77")
    call = {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
    db.append_message(sid, role="assistant", content="", tool_calls=[call])
    db.append_message(sid, role="tool", content=BIG_OUTPUT, tool_name="terminal", tool_call_id="call_1")
    stored = _rows(db, sid)
    restored = db.get_messages_as_conversation(sid)
    stub = copy.copy(restored[5])
    stub["content"] = "[terminal output demoted at compaction — 4,423 chars preserved in session history.]"
    compacted = [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, copy.copy(restored[4]), stub]
    db.archive_and_compact(sid, compacted, watermark=stored[-1]["id"], tail_count=2)
    return stored


def _call(db, **kwargs):
    return json.loads(session_search(db=db, current_session_id=kwargs.pop("current", "s_live"), **kwargs))


class TestRefRecoversOriginals:
    def test_a_compacted_original_comes_back_verbatim(self, db):
        stored = _compacted_session(db)
        target = stored[3]
        after = {r["id"]: r for r in _rows(db, "s_live")}[target["id"]]
        assert (after["active"], after["compacted"]) == (0, 1)
        result = _call(db, ref="m:" + target["message_uid"][:12])
        assert result["success"] is True and result["mode"] == "ref"
        assert result["session_id"] == "s_live" and result["link"].endswith("s_live")
        assert result["ref"] == "m:" + target["message_uid"][:12]
        anchor = next(m for m in result["messages"] if m.get("anchor"))
        assert anchor["id"] == target["id"] and anchor["content"] == "compacted detail COMPACTED_NEEDLE_77"

    def test_a_hidden_carried_tail_original_comes_back_in_full(self, db):
        stored = _compacted_session(db)
        target = stored[5]
        after = {r["id"]: r for r in _rows(db, "s_live")}[target["id"]]
        assert (after["active"], after["compacted"]) == (0, 0)
        # Neither discovery nor scroll can reach it today.
        discovered = _call(db, query="EXACT_NEEDLE_4242", role_filter="tool")
        assert "fatal: EXACT_NEEDLE_4242" not in json.dumps(discovered)

        result = _call(db, ref=target["message_uid"][:12], window=1)
        anchor = next(m for m in result["messages"] if m.get("anchor"))
        assert anchor["id"] == target["id"] and anchor["content"] == BIG_OUTPUT
        assert "content_truncated" not in anchor
        assert "note" not in result

    def test_a_ref_whose_original_is_still_live_says_so(self, db):
        db.create_session("s_live", source="cli")
        db.append_message("s_live", role="user", content="still in context")
        uid = _rows(db, "s_live")[0]["message_uid"]
        result = _call(db, ref="m:" + uid[:12])
        assert result["success"] is True
        assert result["messages"][0]["content"] == "still in context"
        assert "active context" in result["note"]


class TestWindow:
    def test_neighbours_come_from_the_original_position_deduped_by_uid(self, db):
        stored = _compacted_session(db)
        result = _call(db, ref="m:" + stored[2]["message_uid"][:12], window=20)
        ids = [m["id"] for m in result["messages"]]
        # The original generation in order, then the summary; the live copies of rows 4 and 5 share
        # their originals' uids and are dropped, so the stub never shadows the full output.
        assert ids[:6] == [r["id"] for r in stored]
        contents = [m["content"] for m in result["messages"]]
        assert "[CONTEXT COMPACTION] summary" in contents
        assert not any("output demoted at compaction" in (c or "") for c in contents)
        assert len(ids) == len(set(ids)) == 7

    def test_large_neighbours_are_capped_but_the_anchor_is_whole(self, db):
        from tools.session_search_tool import _READ_MAX_CONTENT
        stored = _compacted_session(db)
        db.append_message("s_live", role="tool", content="late " + BIG_OUTPUT, tool_name="terminal",
                          tool_call_id="call_1")
        late = _rows(db, "s_live")[-1]
        # Anchor on the big output: whole. Its big neighbour (a later row) is capped at the read-shape cap.
        result = _call(db, ref="m:" + stored[5]["message_uid"][:12], window=20)
        by_id = {m["id"]: m for m in result["messages"]}
        assert by_id[stored[5]["id"]]["content"] == BIG_OUTPUT and "content_truncated" not in by_id[stored[5]["id"]]
        neighbour = by_id[late["id"]]
        assert neighbour["content_truncated"] is True and neighbour["original_content_chars"] == len(late["content"])
        assert len(neighbour["content"]) == _READ_MAX_CONTENT + 1
        # Anchored on the neighbour instead, it comes back whole.
        whole = _call(db, ref="m:" + late["message_uid"][:12], window=1)
        assert next(m for m in whole["messages"] if m.get("anchor"))["content"] == late["content"]

    def test_hint_offers_a_larger_window_or_scroll(self, db):
        stored = _compacted_session(db)
        hint = _call(db, ref="m:" + stored[0]["message_uid"][:12])["hint"]
        assert hint == ("More context: repeat with a larger window (max 20), or scroll with "
                        "session_search(session_id='s_live', around_message_id=<first or last id above>).")

    def test_window_is_clamped_like_scroll(self, db):
        stored = _compacted_session(db)
        result = _call(db, ref="m:" + stored[0]["message_uid"][:12], window=0)
        assert result["window"] == 1 and [m["id"] for m in result["messages"]] == [stored[0]["id"], stored[1]["id"]]


class TestDispatchAndIsolation:
    def test_ref_overrides_query_and_session_id(self, db):
        stored = _compacted_session(db)
        db.create_session("s_other", source="cli")
        db.append_message("s_other", role="user", content="other session")
        result = _call(db, ref="m:" + stored[3]["message_uid"][:12], query="other", session_id="s_other",
                       around_message_id=_rows(db, "s_other")[0]["id"])
        assert result["mode"] == "ref" and result["session_id"] == "s_live"

    def test_a_ref_from_another_profiles_store_needs_profile(self, db, tmp_path, monkeypatch):
        other_home = tmp_path / "work_home"
        other_home.mkdir()
        other = SessionDB(other_home / "state.db")
        other.create_session("s_far", source="cli")
        other.append_message("s_far", role="user", content="secret far message")
        uid = _rows(other, "s_far")[0]["message_uid"]
        other.close()

        missed = _call(db, ref="m:" + uid[:12])
        assert missed["success"] is False and "secret" not in json.dumps(missed)
        assert "profile" in missed["error"]

        from hermes_cli import profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda n: None)
        monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: True)
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: other_home)
        named = _call(db, ref="m:" + uid[:12], profile="work")
        assert named["success"] is True and named["messages"][0]["content"] == "secret far message"
        assert named["link"] == "@session:work/s_far"


    def test_a_slash_session_id_never_switches_a_refs_store(self, db, tmp_path, monkeypatch):
        import tools.session_search_tool as tool_module
        stored = _compacted_session(db)
        requested = []
        real_resolve = tool_module._resolve_profile_db
        monkeypatch.setattr(tool_module, "_resolve_profile_db",
                            lambda profile: requested.append(profile) or real_resolve(profile))
        ref = "m:" + stored[3]["message_uid"][:12]
        result = _call(db, ref=ref, session_id="work/abc")
        assert result["mode"] == "ref" and result["session_id"] == "s_live"
        assert "work" not in result["link"] and requested == [None]

        # An explicit profile= still routes the ref to that store.
        other_home = tmp_path / "work_home"
        other_home.mkdir()
        other = SessionDB(other_home / "state.db")
        other.create_session("s_far", source="cli")
        other.append_message("s_far", role="user", content="far message")
        far_uid = _rows(other, "s_far")[0]["message_uid"]
        other.close()
        from hermes_cli import profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda n: None)
        monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: True)
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: other_home)
        routed = _call(db, ref="m:" + far_uid[:12], session_id="work/abc", profile="work")
        assert routed["session_id"] == "s_far" and requested[-1] == "work"


class TestErrors:
    @pytest.mark.parametrize("ref", ["m:abc", "m:not-hex-at-all", "x:0123456789ab"])
    def test_invalid(self, db, ref):
        _compacted_session(db)
        result = _call(db, ref=ref)
        assert result["success"] is False and "invalid ref" in result["error"]

    def test_not_found(self, db):
        _compacted_session(db)
        result = _call(db, ref="m:" + "0" * 12)
        assert result["success"] is False and "not found" in result["error"]

    def test_ambiguous(self, db):
        stored = _compacted_session(db)
        for row, tail in ((stored[0], "0"), (stored[1], "1")):
            db._conn.execute("UPDATE messages SET message_uid = ? WHERE message_uid = ?",
                             ("abcdef012345" + tail * 20, row["message_uid"]))
        db._conn.commit()
        result = _call(db, ref="m:abcdef012345")
        assert result["success"] is False and "ambiguous" in result["error"]
        assert "longer" in result["error"]

    def test_blank_ref_is_ignored(self, db):
        _compacted_session(db)
        assert _call(db, ref="  ")["mode"] == "browse"


class TestSchema:
    def test_ref_param_is_one_short_string(self, db):
        prop = SESSION_SEARCH_SCHEMA["parameters"]["properties"]["ref"]
        assert prop["type"] == "string"
        assert len(prop["description"].split()) <= 35

    def test_ref_is_appended_after_the_frozen_positional_params(self):
        params = list(inspect.signature(session_search).parameters)
        assert params[-1] == "ref" and params[:-1][-1] == "exclude_session_ids"

    def test_registry_handler_forwards_ref(self, db):
        from tools.registry import registry
        stored = _compacted_session(db)
        handler = registry.get_entry("session_search").handler
        result = json.loads(handler({"ref": "m:" + stored[3]["message_uid"][:12]}, db=db,
                                    current_session_id="s_live"))
        assert result["mode"] == "ref"
