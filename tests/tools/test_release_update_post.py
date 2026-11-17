"""Gated #release-updates posting (education design 5b) through
slack_post_thread: allowlisted channels only; a gated channel takes a Release
Assets row id, never text; the row must be a Slack update, Approved (per the
`post_release_update` autonomy level) and every linked Jira key Done; the
post is the row's own body; then Published URL + Status=Published are written
back. Notion, Jira, Slack and the clock are mocked; the autonomy store is real
under the test HERMES_HOME."""
import json
from datetime import datetime, timedelta, timezone

import pytest

import agent.autonomy as au
import tools.release_asset_gate as gate
import tools.release_update_post as rup
import tools.slack_post_thread_tool as spt
from model_tools import _run_async

DS = "11111111222233334444555555555555"
ROW = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
ROW_KEY = ROW.replace("-", "")
RELEASE = "c" * 32
CHOME, CREL, COTHER = "CHOME0001", "CREL00001", "COTHER001"
NOW = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)  # a Wednesday


def rt(text, **ann):
    return {"type": "text", "plain_text": text, "href": None, "annotations": ann}


def _row(status="Approved", asset_type="Slack update", jira="BA-1", relation=(RELEASE,), has_more=False,
         status_type="select", parent=DS, **extra):
    page = {"id": ROW, "url": "https://www.notion.so/row-aaaa",
            "parent": {"type": "data_source_id", "data_source_id": parent},
            "properties": {
                "Name": {"type": "title", "title": [rt("Points <launch>")]},
                "Status": {"type": status_type, status_type: {"name": status}},
                "Asset type": {"type": "select", "select": {"name": asset_type}},
                "JIRA": {"type": "rich_text", "rich_text": [rt(jira)] if jira else []},
                "Release": {"type": "relation", "relation": [{"id": r} for r in (relation or [])],
                            "has_more": has_more},
                "Published URL": {"type": "url", "url": None}}}
    page.update(extra)
    return page


class World:
    def __init__(self, monkeypatch):
        self.level = "signoff"
        self.guard = True
        self.allowed = [CHOME, CREL]
        self.gated = [CREL]
        self.row = _row()
        self.pages = {RELEASE: {"id": RELEASE, "properties": {
            "JIRA": {"type": "multi_select", "multi_select": [{"name": "BA-2"}]}}}}
        self.blocks = [{"type": "paragraph", "paragraph": {"rich_text": [rt("Points are "), rt("live", bold=True)]}}]
        self.jira = {"BA-1": "Done", "BA-2": "Done"}
        self.posts, self.writebacks, self.audits = [], [], []
        self.vetoers = []
        self.writeback_error = None
        self.now = NOW
        self.free_posts = []
        monkeypatch.setattr(rup, "_config", self.config)
        monkeypatch.setattr(rup, "_now", lambda: self.now)
        monkeypatch.setattr(rup, "_audit", lambda a, t: self.audits.append((a, t)))
        monkeypatch.setattr(rup, "_jira_status", self.jira_status)
        monkeypatch.setattr(gate, "fetch_notion_page", self.fetch)
        monkeypatch.setattr(gate, "fetch_block_tree", self.fetch_blocks)
        monkeypatch.setattr(gate, "post_announcement", self.post)
        monkeypatch.setattr(gate, "veto_reactions", self.reactions)
        monkeypatch.setattr(gate, "update_page_properties", self.writeback)
        monkeypatch.setattr(gate, "permalink", self.permalink)

        async def free_post(token, channel, thread_ts, text):
            self.free_posts.append((channel, thread_ts, text))
            return {"ok": True, "ts": "9.9"}

        async def free_link(token, channel, ts):
            return ""
        monkeypatch.setattr(spt, "_post_message", free_post)
        monkeypatch.setattr(spt, "_get_permalink", free_link)
        monkeypatch.setattr(spt, "_resolve_slack_token", lambda: "xoxb")

    def config(self):
        cfg = {"autonomy": {"announce_channel": CHOME, "veto_wait_working_hours": 4,
                            "veto_expiry_hours": 72, "actions": {"post_release_update": self.level}},
               "slack": {"post_allowed_channels": self.allowed, "post_gated_channels": self.gated},
               "release_updates": {"release_assets_data_source_id": DS,
                                   "jira_key_properties": ["JIRA"],
                                   "jira_relation_properties": ["Release"]}}
        if self.allowed is None:
            del cfg["slack"]["post_allowed_channels"]
        if self.guard:
            cfg["notion"] = {"write_guard": {"enabled": True}}
            cfg["agent_write_guard"] = True
        return cfg

    async def fetch(self, page_id):
        key = page_id.replace("-", "")
        if key == ROW_KEY:
            return self.row
        return self.pages[key]

    async def fetch_blocks(self, page_id):
        return self.blocks

    async def post(self, channel, text, thread_ts=""):
        self.posts.append((channel, text, thread_ts))
        return f"{len(self.posts)}00.0001"

    async def permalink(self, channel, ts):
        return f"https://acme.slack.com/archives/{channel}/p{ts.replace('.', '')}"

    async def reactions(self, channel, ts):
        return list(self.vetoers)

    async def writeback(self, page_id, properties):
        if self.writeback_error:
            raise gate.GateError(self.writeback_error)
        self.writebacks.append((page_id, properties))
        return {}

    async def jira_status(self, key):
        if key not in self.jira:
            raise gate.GateError(f"Jira issue {key} not found")
        return self.jira[key]

    @property
    def release_posts(self):
        return [p for p in self.posts if p[0] == CREL]


@pytest.fixture
def w(monkeypatch):
    return World(monkeypatch)


def _call(args):
    return json.loads(_run_async(spt._slack_post_thread_handler(args)))


def _gated(**over):
    a = {"chat_id": CREL, "release_asset_row_id": ROW}
    a.update(over)
    return a


# ── channel policy ───────────────────────────────────────────────────────

def test_free_text_to_allowed_channel(w):
    out = _call({"chat_id": CHOME, "message": "hi"})
    assert out["success"] is True and w.free_posts == [(CHOME, "", "hi")]


def test_channel_outside_allowlist_refused(w):
    out = _call({"chat_id": COTHER, "message": "hi"})
    assert "post_allowed_channels" in out["error"] and w.free_posts == []


def test_free_text_to_gated_channel_refused(w):
    out = _call({"chat_id": CREL, "message": "hi"})
    assert "release_asset_row_id" in out["error"]
    out = _call({"chat_id": CREL, "message": "hi", "release_asset_row_id": ROW})
    assert "free text" in out["error"]
    out = _call(_gated(thread_ts="1.2"))
    assert "error" in out
    assert w.free_posts == [] and w.posts == []


def test_gated_channel_must_also_be_allowed(w):
    w.allowed = [CHOME]
    assert "post_allowed_channels" in _call(_gated())["error"]
    assert w.posts == []


def test_row_id_only_for_gated_channels(w):
    assert "gated" in _call({"chat_id": CHOME, "release_asset_row_id": ROW})["error"]


def test_guarded_profile_without_allowlist_posts_nowhere(w):
    w.allowed = None
    assert "post_allowed_channels" in _call({"chat_id": CHOME, "message": "hi"})["error"]


def test_unguarded_profile_without_policy_is_legacy(w):
    w.allowed, w.gated, w.guard = None, [], False
    assert _call({"chat_id": COTHER, "message": "hi"})["success"] is True


def test_gated_channel_still_gated_without_allowlist(w):
    w.allowed, w.guard = None, False
    assert "error" in _call({"chat_id": CREL, "message": "hi"})


# ── preconditions ────────────────────────────────────────────────────────

def test_off_refuses(w):
    w.level = "off"
    assert "off" in _call(_gated())["error"] and w.posts == []


@pytest.mark.parametrize("level", ["signoff", "veto", "auto"])
def test_refused_without_notion_write_guard(w, level):
    w.level, w.guard = level, False
    assert "write_guard" in _call(_gated())["error"] and w.posts == []


def test_invalid_level_refused(w):
    w.level = "autopilot"
    assert "post_release_update" in _call(_gated())["error"]


# ── signoff ──────────────────────────────────────────────────────────────

def test_signoff_posts_row_body_and_writes_back(w):
    out = _call(_gated())
    assert out["ok"] is True and out["message_ts"] == "100.0001"
    assert w.release_posts == [(CREL, "Points are *live*", "")]
    assert w.writebacks == [(ROW_KEY, {
        "Published URL": {"url": "https://acme.slack.com/archives/CREL00001/p1000001"},
        "Status": {"select": {"name": "Published"}}})]
    assert out["published_written"] is True
    assert out["jira"] == {"BA-1": "Done", "BA-2": "Done"}
    assert any(a == "autonomy_signoff" for a, _ in w.audits)


def test_status_type_property_written_back_as_status(w):
    w.row = _row(status_type="status")
    _call(_gated())
    assert w.writebacks[0][1]["Status"] == {"status": {"name": "Published"}}


def test_signoff_requires_approved(w):
    w.row = _row(status="In review")
    assert "Approved" in _call(_gated())["error"] and w.posts == []


@pytest.mark.parametrize("status", ["Hold", "Published"])
def test_hold_or_published_refused(w, status):
    w.row = _row(status=status)
    assert "error" in _call(_gated()) and w.posts == []


def test_wrong_type_refused(w):
    w.row = _row(asset_type="Help article")
    assert "Slack update" in _call(_gated())["error"] and w.posts == []


def test_wrong_parent_refused(w):
    w.row = _row(parent="9" * 32)
    assert "error" in _call(_gated()) and w.posts == []


def test_trashed_row_refused(w):
    w.row = _row(in_trash=True)
    assert "error" in _call(_gated()) and w.posts == []


def test_jira_not_done_refused(w):
    w.jira["BA-2"] = "In Staging"
    out = _call(_gated())
    assert "BA-2" in out["error"] and "In Staging" in out["error"] and w.posts == []


def test_no_jira_keys_refused(w):
    w.row = _row(jira="", relation=[])
    assert "Jira" in _call(_gated())["error"] and w.posts == []


def test_relation_has_more_refused(w):
    w.row = _row(relation=[RELEASE], has_more=True)
    assert "error" in _call(_gated()) and w.posts == []


def test_jira_keys_from_relations(w):
    w.row = _row(jira="", relation=[RELEASE])
    out = _call(_gated())
    assert out["jira"] == {"BA-2": "Done"}


def test_unsupported_content_and_images(w):
    w.blocks = [{"type": "table", "table": {}}]
    assert "table" in _call(_gated())["error"]
    w.blocks = [{"type": "image", "image": {"caption": [rt("shot")]}},
                {"type": "paragraph", "paragraph": {"rich_text": [rt("Body")]}}]
    assert "image" in _call(_gated())["error"]
    out = _call(_gated(omit_images=True))
    assert out["ok"] is True and out["images_omitted"] == ["shot"]
    assert w.release_posts[0][1] == "Body"


def test_empty_body_refused(w):
    w.blocks = []
    assert "error" in _call(_gated()) and w.posts == []


def test_one_post_per_row(w):
    _call(_gated())
    out = _call(_gated())
    assert out["already_posted"] is True
    assert len(w.release_posts) == 1 and len(w.writebacks) == 1


def test_writeback_retry_does_not_repost(w):
    w.writeback_error = "Notion 502"
    out = _call(_gated())
    assert out["published_written"] is False and "Notion 502" in out["writeback_error"]
    w.writeback_error = None
    out = _call(_gated())
    assert out["already_posted"] is True and out["published_written"] is True
    assert len(w.release_posts) == 1 and len(w.writebacks) == 1


def test_lock_held_refuses(w):
    with au.exclusive(rup.ACTION, ROW_KEY):
        assert "already running" in _call(_gated())["error"]
    assert w.posts == []


# ── veto ─────────────────────────────────────────────────────────────────

def _queue(w):
    w.level = "veto"
    w.row = _row(status="In review")
    out = _call(_gated())
    assert out["status"] == "pending"
    return out["pending_id"]


def test_veto_announces_then_posts_after_window(w):
    pid = _queue(w)
    assert w.release_posts == []
    announce = [p for p in w.posts if p[0] == CHOME]
    assert len(announce) == 1 and pid in announce[0][1] and "Points &lt;launch&gt;" in announce[0][1]
    w.now = NOW + timedelta(hours=3)
    assert "veto window" in _call({"pending_id": pid})["error"]
    w.now = NOW + timedelta(hours=5)
    results = _call({"run_due": True})["results"]
    assert results[0]["ok"] is True
    assert w.release_posts == [(CREL, "Points are *live*", "")]
    assert len(w.writebacks) == 1
    assert au.load_pending(rup.ACTION, pid)["status"] == "done"


def test_veto_with_approved_row_posts_now(w):
    w.level = "veto"
    out = _call(_gated())
    assert out["ok"] is True and len(w.release_posts) == 1


def test_veto_x_reaction_stops(w):
    pid = _queue(w)
    w.vetoers = ["U0VETO"]
    w.now = NOW + timedelta(hours=5)
    out = _call({"pending_id": pid})
    assert out["status"] == "vetoed" and w.release_posts == []


def test_veto_hold_stops(w):
    pid = _queue(w)
    w.row = _row(status="Hold")
    w.now = NOW + timedelta(hours=5)
    assert _call({"pending_id": pid})["status"] == "vetoed"
    assert w.release_posts == []


def test_veto_content_change_is_stale(w):
    pid = _queue(w)
    w.blocks = [{"type": "paragraph", "paragraph": {"rich_text": [rt("Something else")]}}]
    w.now = NOW + timedelta(hours=5)
    assert _call({"pending_id": pid})["status"] == "stale"
    assert w.release_posts == []


def test_veto_jira_reopened_stays_pending(w):
    pid = _queue(w)
    w.jira["BA-1"] = "In Progress"
    w.now = NOW + timedelta(hours=5)
    assert "BA-1" in _call({"pending_id": pid})["error"]
    assert au.load_pending(rup.ACTION, pid)["status"] == "pending"
    assert w.release_posts == []


def test_veto_queue_requires_jira_done(w):
    w.level = "veto"
    w.row = _row(status="In review")
    w.jira["BA-1"] = "In Staging"
    assert "error" in _call(_gated()) and w.posts == []


def test_veto_expiry(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=4 + 72, minutes=1)
    assert _call({"pending_id": pid})["status"] == "expired"


def test_veto_channel_dropped_from_allowlist_before_run(w):
    pid = _queue(w)
    w.allowed = [CHOME]
    w.now = NOW + timedelta(hours=5)
    assert "error" in _call({"pending_id": pid}) and w.release_posts == []


def test_list_pending(w):
    pid = _queue(w)
    pending = _call({"list_pending": True})["pending"]
    assert [p["pending_id"] for p in pending] == [pid]


def test_paused_drops_veto_to_signoff(w):
    w.level = "veto"
    w.row = _row(status="In review")
    cfg = w.config()
    cfg["autonomy"]["paused"] = True
    w.config = lambda: cfg
    rup._config = w.config  # monkeypatch fixture restores the attribute
    assert "Approved" in _call(_gated())["error"]


# ── auto ─────────────────────────────────────────────────────────────────

def test_auto_posts_without_approval_and_reports(w):
    w.level = "auto"
    w.row = _row(status="In review")
    out = _call(_gated())
    assert out["ok"] is True and len(w.release_posts) == 1
    report = [p for p in w.posts if p[0] == CHOME]
    assert len(report) == 1 and "autonomy: auto" in report[0][1]


def test_auto_still_needs_jira_done_and_not_hold(w):
    w.level = "auto"
    w.row = _row(status="Hold")
    assert "error" in _call(_gated())
    w.row = _row(status="Draft")
    w.jira["BA-1"] = "To Do"
    assert "error" in _call(_gated())
    assert w.release_posts == []



# ── review round ─────────────────────────────────────────────────────────

def test_requires_agent_write_guard(w, monkeypatch):
    cfg = w.config()
    del cfg["agent_write_guard"]
    monkeypatch.setattr(rup, "_config", lambda: cfg)
    assert "agent_write_guard" in _call(_gated())["error"] and w.posts == []


@pytest.mark.parametrize("level", ["veto", "auto"])
def test_rejected_row_never_posts_without_approval(w, level):
    w.level = level
    w.row = _row(status="Changes requested")
    assert "error" in _call(_gated())
    assert w.posts == []


def test_waiting_row_not_announced_twice(w):
    pid = _queue(w)
    out = _call(_gated())
    assert out["pending_id"] == pid
    assert len([p for p in w.posts if p[0] == CHOME]) == 1


def test_vetoed_row_not_requeued(w):
    pid = _queue(w)
    w.vetoers = ["U0VETO"]
    w.now = NOW + timedelta(hours=5)
    assert _call({"pending_id": pid})["status"] == "vetoed"
    out = _call(_gated())
    assert "vetoed" in out["error"]
    # a human approving it is the way forward
    w.row = _row(status="Approved")
    assert _call(_gated())["ok"] is True and len(w.release_posts) == 1


def test_channel_name_refused(w):
    assert "channel id" in _call({"chat_id": "#product-education", "message": "hi"})["error"]
    assert w.free_posts == []
