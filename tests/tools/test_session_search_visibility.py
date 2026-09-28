"""Search and browse must not expose messages excluded from display history."""

import json

import pytest

from hermes_state import SessionDB
from tools import session_search_tool  # noqa: F401  register the real handler
from tools.registry import registry


@pytest.mark.parametrize("excluded", ["rewound", "hidden", "model_only"])
@pytest.mark.parametrize("compacted", [False, True])
def test_previews_choose_visible_user_content(tmp_path, excluded, compacted):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli")
        kwargs = {"display_kind": "hidden"} if excluded == "hidden" else {}
        if excluded == "model_only":
            kwargs["display_metadata"] = {"model_only": True}
        first = db.append_message("s", "user", "excluded preview", **kwargs)
        if excluded == "rewound":
            db.rewind_to_message("s", first)
        db.append_message("s", "user", "visible original request")
        if compacted:
            db.archive_and_compact("s", [{"role": "assistant", "content": "summary"}])
        db.append_message("s", "assistant", "visible reply")

        browse = json.loads(registry.dispatch("session_search", {}, db=db))
        assert browse["success"] is True
        for row in (
            browse["results"][0],
            db.list_sessions_rich()[0],
            db.list_sessions_rich(order_by_last_active=True)[0],
            db.get_session_rich_row("s"),
        ):
            assert row["preview"] == "visible original request"


@pytest.mark.parametrize("surface", [
    "hits", "context", "window", "bookends", "discovery", "title", "read", "scroll",
])
def test_model_only_rows_stay_out_of_search_views(tmp_path, surface):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli")
        db.set_session_title("s", "Visibility Recall Title")

        def internal():
            return db.append_message("s", "user", "INTERNALMODEL needle",
                                     display_metadata={"model_only": True})

        first = internal()
        for i in range(8):
            db.append_message("s", "user", f"visible head {i}")
        internal()
        anchor = db.append_message("s", "user", "visible anchor needle",
                                   display_metadata={"model_only": False})
        internal()
        for i in range(8):
            db.append_message("s", "assistant", f"visible tail {i}")
        internal()

        # Raw model history retains the carrier, display history excludes it.
        assert db.get_messages("s")[0]["id"] == first
        assert first not in {m["id"] for m in db.get_messages("s", include_compacted=True)}
        if surface == "hits":
            for include_inactive in (False, True):
                assert db.search_messages("INTERNALMODEL", include_inactive=include_inactive) == []
            return
        if surface == "context":
            for include_inactive in (False, True):
                hit = db.search_messages("visible anchor needle", include_inactive=include_inactive)[0]
                assert [m["content"] for m in hit["context"]] == [
                    "visible head 7", "visible anchor needle", "visible tail 0",
                ]
            return
        if surface == "window":
            result = db.get_messages_around("s", anchor, window=1, search_visible=True)
            assert [m["content"] for m in result["window"]] == [
                "visible head 7", "visible anchor needle", "visible tail 0",
            ]
            assert db.get_messages_around("s", first, search_visible=True)["window"] == []
            assert db.get_messages_around("s", first)["window"][0]["id"] == first
        elif surface == "bookends":
            result = db.get_anchored_view("s", anchor, window=1)
            assert result["bookend_start"][0]["content"] == "visible head 0"
            assert result["bookend_end"][-1]["content"] == "visible tail 7"
        else:
            args = {
                "discovery": {"query": "visible anchor needle", "detail": "full"},
                "title": {"query": "Visibility Recall Title"},
                "read": {"session_id": "s"},
                "scroll": {"session_id": "s", "around_message_id": anchor},
            }[surface]
            result = json.loads(registry.dispatch("session_search", args, db=db))
            assert result["success"] is True
        payload = json.dumps(result)
        expected = "visible head 0" if surface == "title" else "visible anchor needle"
        assert expected in payload
        assert "INTERNALMODEL" not in payload
