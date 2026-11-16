"""zendesk_create_draft — the draft create gated by the profile's autonomy
level (off / signoff / veto / auto). Zendesk HTTP, the Notion row, Slack and
the clock are mocked; the pending store is real (HERMES_HOME is a tmp dir)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

import agent.autonomy as au
import tools.release_asset_gate as gate
import tools.zendesk_tool as zt
from model_tools import _run_async

BASE = "https://everafterai.zendesk.com"
DS = "11111111222233334444555555555555"
ROW = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
NOW = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)  # a Wednesday


def _page(status="Approved", asset_type="Help article",
          destination="everafterai.zendesk.com", parent_ds=DS):
    return {"url": "https://notion.so/row", "parent": {"type": "data_source_id",
                                                       "data_source_id": parent_ds},
            "properties": {"Status": {"type": "select", "select": {"name": status}},
                           "Asset type": {"type": "select", "select": {"name": asset_type}},
                           "Destination": {"type": "select", "select": {"name": destination}}}}


class World:
    def __init__(self, monkeypatch):
        self.level = "signoff"
        self.paused = False
        self.row = _page()
        self.requests = []
        self.posts = []
        self.vetoers = []
        self.audits = []
        self.now = NOW
        self.article = {"id": 77, "section_id": 42, "title": "Live article",
                        "html_url": f"{BASE}/hc/en-us/articles/77", "draft": False}
        monkeypatch.setattr(zt, "_zendesk_creds", lambda: (BASE, "bot@ex.com", "tok"))
        monkeypatch.setattr(zt, "_config", self.config)
        monkeypatch.setattr(zt, "_now", lambda: self.now)
        monkeypatch.setattr(zt, "_audit", lambda action, target: self.audits.append((action, target)))
        monkeypatch.setattr(zt, "_zendesk_request", self.zendesk)
        monkeypatch.setattr(gate, "fetch_notion_page", self.fetch_row)
        monkeypatch.setattr(gate, "post_announcement", self.post)
        monkeypatch.setattr(gate, "veto_reactions", self.reactions)

    def config(self):
        return {"autonomy": {"paused": self.paused, "announce_channel": "CHOME",
                             "veto_wait_working_hours": 4,
                             "actions": {"zendesk_create_draft": self.level}},
                "zendesk": {"release_assets_data_source_id": DS}}

    async def zendesk(self, base, email, token, path, method="GET", json_body=None):
        self.requests.append((method, path, json_body))
        if method == "GET":
            return {"status": 200, "text": json.dumps({"article": self.article})}
        return {"status": 201, "text": json.dumps({"article": {
            "id": 500, "html_url": f"{BASE}/hc/en-us/articles/500", "draft": True}})}

    async def fetch_row(self, page_id):
        if isinstance(self.row, Exception):
            raise self.row
        return self.row

    async def post(self, channel, text, thread_ts=""):
        self.posts.append((channel, text, thread_ts))
        return "111.222" if not thread_ts else "111.333"

    async def reactions(self, channel, ts):
        return list(self.vetoers)

    @property
    def writes(self):
        return [r for r in self.requests if r[0] != "GET"]


@pytest.fixture
def w(monkeypatch):
    return World(monkeypatch)


def _args(**over):
    a = {"section_id": "42", "title": "Account points", "body": "<h2>Overview</h2><p>x</p>",
         "permission_group_id": 7, "release_asset_row_id": ROW}
    a.update(over)
    return a


def _call(args):
    return json.loads(_run_async(zt._zendesk_create_draft_handler(args)))


def test_off_refuses(w):
    w.level = "off"
    out = _call(_args())
    assert "error" in out and "off" in out["error"]
    assert w.requests == [] and w.posts == []


def test_publish_level_config_error_refuses(w, monkeypatch):
    monkeypatch.setattr(zt, "_config", lambda: {"autonomy": {"actions": {
        "zendesk_create_draft": "publish"}}})
    out = _call(_args())
    assert "error" in out and "Built levels" in out["error"]
    assert w.requests == []


def test_signoff_requires_row(w):
    out = _call(_args(release_asset_row_id=""))
    assert "error" in out and "release_asset_row_id" in out["error"]
    assert w.writes == []


@pytest.mark.parametrize("page,why", [
    (_page(status="Draft"), "Approved"),
    (_page(status="In review"), "Approved"),
    (_page(status="Published"), "Published"),
    (_page(asset_type="Slack post"), "Help article"),
    (_page(destination="support.base.ai"), "everafterai.zendesk.com"),
    (_page(parent_ds="ffffffffffffffffffffffffffffffff"), "Release Assets"),
])
def test_signoff_refuses_unapproved_published_wrong_type_or_parent(w, page, why):
    w.row = page
    out = _call(_args())
    assert "error" in out and why in out["error"]
    assert w.writes == []


def test_signoff_notion_unreachable_fails_closed(w):
    w.row = gate.GateError("Notion API error (HTTP 500)")
    out = _call(_args())
    assert "error" in out and "500" in out["error"]
    assert w.writes == []


def test_signoff_creates_forced_draft(w):
    out = _call(_args(user_segment_id=12, locale="en-us"))
    assert out["ok"] is True and out["article_id"] == 500 and out["draft"] is True
    assert out["level"] == "signoff" and out["reconcile_required"] is False
    assert out["html_url"].endswith("/articles/500")
    [(method, path, body)] = w.writes
    assert method == "POST" and path == "api/v2/help_center/en-us/sections/42/articles.json"
    assert body == {"article": {"title": "Account points", "locale": "en-us",
                                "body": "<h2>Overview</h2><p>x</p>", "draft": True,
                                "permission_group_id": 7, "user_segment_id": 12},
                    "notify_subscribers": False}
    assert w.audits and w.audits[0][0] == "autonomy_signoff"


def test_model_cannot_unset_draft(w):
    # Extra/unknown args never reach the body.
    _call(_args(draft=False, notify_subscribers=True, article={"draft": False}))
    [(_m, _p, body)] = w.writes
    assert body["article"]["draft"] is True and body["notify_subscribers"] is False


@pytest.mark.parametrize("bad", [
    {"section_id": "42/../1"}, {"section_id": ""}, {"title": ""}, {"body": ""},
    {"permission_group_id": "x"}, {"permission_group_id": None}, {"locale": "../en"},
    {"user_segment_id": "abc"}, {"revision_of_article_id": "1/../2"},
])
def test_invalid_args_refused(w, bad):
    out = _call(_args(**bad))
    assert "error" in out
    assert w.writes == []


def test_revision_creates_separate_draft_and_flags_reconcile(w):
    out = _call(_args(section_id=None, revision_of_article_id="77"))
    assert out["ok"] is True and out["reconcile_required"] is True
    assert out["reconcile_with"] == {"id": 77, "title": "Live article",
                                     "html_url": f"{BASE}/hc/en-us/articles/77"}
    # Read the live article, then POST a NEW article into its section. Never PUT.
    assert w.requests[0][:2] == ("GET", "api/v2/help_center/articles/77.json")
    [(method, path, _body)] = w.writes
    assert (method, path) == ("POST", "api/v2/help_center/en-us/sections/42/articles.json")


def test_revision_of_missing_article_refused(w, monkeypatch):
    async def missing(base, email, token, path, method="GET", json_body=None):
        w.requests.append((method, path, json_body))
        return {"status": 404, "text": '{"error": "RecordNotFound"}'}
    monkeypatch.setattr(zt, "_zendesk_request", missing)
    out = _call(_args(revision_of_article_id="77"))
    assert "error" in out and "404" in out["error"]
    assert w.writes == []


def test_veto_announces_and_queues(w):
    w.level = "veto"
    w.row = _page(status="Draft")
    out = _call(_args())
    assert out["status"] == "pending" and w.writes == []
    pid = out["pending_id"]
    assert out["not_before"] == (NOW + timedelta(hours=4)).isoformat()
    [(channel, text, thread)] = w.posts
    assert channel == "CHOME" and thread == "" and "Account points" in text and pid in text
    rec = au.load_pending("zendesk_create_draft", pid)
    assert rec["status"] == "pending" and rec["announcement_ts"] == "111.222"
    listed = _call({"list_pending": True})
    assert [p["pending_id"] for p in listed["pending"]] == [pid]


def test_veto_requires_announce_channel(w, monkeypatch):
    w.level = "veto"
    w.row = _page(status="Draft")
    cfg = w.config()
    cfg["autonomy"]["announce_channel"] = ""
    monkeypatch.setattr(zt, "_config", lambda: cfg)
    out = _call(_args())
    assert "error" in out and "announce_channel" in out["error"]
    assert au.list_pending("zendesk_create_draft") == []


def test_veto_with_approved_row_acts_now(w):
    w.level = "veto"
    out = _call(_args())
    assert out["ok"] is True and len(w.writes) == 1 and w.posts == []


def test_veto_still_refuses_published_or_held_row(w):
    w.level = "veto"
    for status in ("Published", "Hold"):
        w.row = _page(status=status)
        assert "error" in _call(_args())
    assert w.writes == [] and w.posts == []


def _queue(w):
    w.level = "veto"
    w.row = _page(status="Draft")
    return _call(_args())["pending_id"]


def test_veto_execute_too_early(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=3)
    out = _call({"pending_id": pid})
    assert "error" in out and "veto window" in out["error"]
    assert w.writes == []


@pytest.mark.parametrize("veto", ["reaction", "hold"])
def test_veto_execute_refuses_on_hold_or_x_reaction(w, veto):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=5)
    if veto == "reaction":
        w.vetoers = ["UPM"]
    else:
        w.row = _page(status="Hold")
    out = _call({"pending_id": pid})
    assert out["status"] == "vetoed" and w.writes == []
    assert au.load_pending("zendesk_create_draft", pid)["status"] == "vetoed"
    # A vetoed action can't be re-run.
    assert "error" in _call({"pending_id": pid})


def test_veto_execute_uses_stored_request_and_audits(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=5)
    out = _call({"pending_id": pid, "title": "SWAPPED", "body": "<p>evil</p>"})
    assert out["ok"] is True and out["level"] == "veto"
    [(_m, _p, body)] = w.writes
    assert body["article"]["title"] == "Account points"
    assert body["article"]["body"] == "<h2>Overview</h2><p>x</p>"
    assert ("autonomy_veto" in [a for a, _ in w.audits])
    # Result is reported in the announcement thread.
    assert w.posts[-1][0] == "CHOME" and w.posts[-1][2] == "111.222"
    assert au.load_pending("zendesk_create_draft", pid)["status"] == "done"
    assert "error" in _call({"pending_id": pid})  # not twice


def test_pending_execution_rechecks_level(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=5)
    w.level = "off"
    assert "error" in _call({"pending_id": pid})
    w.level = "signoff"  # demoted: now needs the row Approved
    assert "Approved" in _call({"pending_id": pid})["error"]
    w.row = _page(status="Approved")
    assert _call({"pending_id": pid})["ok"] is True
    assert len(w.writes) == 1


def test_unknown_pending_id(w):
    assert "error" in _call({"pending_id": "0123456789ab"})
    assert "error" in _call({"pending_id": "../../x"})


def test_auto_creates_and_announces(w):
    w.level = "auto"
    w.row = _page(status="Draft")
    out = _call(_args())
    assert out["ok"] is True and out["level"] == "auto"
    assert len(w.writes) == 1
    assert w.posts and w.posts[0][0] == "CHOME" and "articles/500" in w.posts[0][1]
    assert [a for a, _ in w.audits] == ["autonomy_auto"]


def test_auto_row_optional_but_checked_when_given(w):
    w.level = "auto"
    assert _call(_args(release_asset_row_id=""))["ok"] is True
    w.row = _page(status="Published")
    assert "error" in _call(_args())
    assert len(w.writes) == 1


def test_auto_announce_failure_does_not_undo(w, monkeypatch):
    w.level = "auto"

    async def fail(*a, **k):
        raise gate.GateError("channel_not_found")
    monkeypatch.setattr(gate, "post_announcement", fail)
    out = _call(_args())
    assert out["ok"] is True and "channel_not_found" in out["announce_error"]


def test_paused_drops_auto_to_signoff(w):
    w.level = "auto"
    w.paused = True
    w.row = _page(status="Draft")
    out = _call(_args())
    assert "error" in out and "Approved" in out["error"]
    assert w.writes == []
