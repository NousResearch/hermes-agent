"""zendesk_create_draft — a draft built FROM the Release Assets row, gated by
the profile's autonomy level (off / signoff / veto / auto) and the Notion
write-guard precondition. Zendesk HTTP, the Notion row and its blocks, Slack
and the clock are mocked; the autonomy store (pending, ledger, locks, HMAC
key) is real under the test HERMES_HOME."""
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
ROW_KEY = ROW.replace("-", "")
NOW = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)  # a Wednesday


def rt(text):
    return {"type": "text", "plain_text": text, "href": None, "annotations": {}}


def _page(status="Approved", asset_type="Help article", title="Account points",
          destination="everafterai.zendesk.com", parent_ds=DS, **extra):
    page = {"url": "https://www.notion.so/row-aaaa",
            "parent": {"type": "data_source_id", "data_source_id": parent_ds},
            "properties": {
                "Name": {"type": "title", "title": [rt(title)]},
                "Status": {"type": "select", "select": {"name": status}},
                "Asset type": {"type": "select", "select": {"name": asset_type}},
                "Destination": {"type": "select", "select": {"name": destination}}}}
    page.update(extra)
    return page


def _blocks(*paras):
    return [{"type": "paragraph", "has_children": False, "paragraph": {"rich_text": [rt(p)]}}
            for p in paras]


class World:
    def __init__(self, monkeypatch):
        self.level = "signoff"
        self.paused = False
        self.guard = True
        self.row = _page()
        self.blocks = _blocks("Earn points.")
        self.requests = []
        self.posts = []
        self.vetoers = []
        self.audits = []
        self.now = NOW
        self.next_id = 500
        self.article = {"id": 77, "section_id": 42, "title": "Live <b>article</b>",
                        "locale": "en-us", "permission_group_id": 9, "user_segment_id": 33}
        monkeypatch.setattr(zt, "_zendesk_creds", lambda: (BASE, "cid", "secret"))

        async def token(scope, *, force=False):
            return "tok"
        monkeypatch.setattr(zt, "_access_token", token)
        monkeypatch.setattr(zt, "_config", self.config)
        monkeypatch.setattr(zt, "_now", lambda: self.now)
        monkeypatch.setattr(zt, "_audit", lambda action, target: self.audits.append((action, target)))
        monkeypatch.setattr(zt, "_zendesk_request", self.zendesk)
        monkeypatch.setattr(gate, "fetch_notion_page", self.fetch_row)
        monkeypatch.setattr(gate, "fetch_block_tree", self.fetch_blocks)
        monkeypatch.setattr(gate, "post_announcement", self.post)
        monkeypatch.setattr(gate, "veto_reactions", self.reactions)

    def config(self):
        cfg = {"autonomy": {"paused": self.paused, "announce_channel": "CHOME",
                            "veto_wait_working_hours": 4, "veto_expiry_hours": 72,
                            "actions": {"zendesk_create_draft": self.level}},
               "zendesk": {"release_assets_data_source_id": DS,
                           "default_permission_group_id": 7, "default_user_segment_id": 12}}
        if self.guard:
            cfg["notion"] = {"write_guard": {"enabled": True}}
        return cfg

    async def zendesk(self, base, token, path, method="GET", json_body=None):
        self.requests.append((method, path, json_body))
        if method == "GET":
            return {"status": 200, "text": json.dumps({"article": self.article})}
        aid = self.next_id
        self.next_id += 1
        return {"status": 201, "text": json.dumps({"article": {
            "id": aid, "html_url": f"{BASE}/hc/en-us/articles/{aid}", "draft": True}})}

    async def fetch_row(self, page_id):
        if isinstance(self.row, Exception):
            raise self.row
        return self.row

    async def fetch_blocks(self, page_id):
        return self.blocks

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
    a = {"section_id": "42", "release_asset_row_id": ROW}
    a.update(over)
    return a


def _call(args):
    return json.loads(_run_async(zt._zendesk_create_draft_handler(args)))


# ── preconditions ────────────────────────────────────────────────────────

def test_off_refuses(w):
    w.level = "off"
    out = _call(_args())
    assert "error" in out and "off" in out["error"]
    assert w.requests == [] and w.posts == []


@pytest.mark.parametrize("level", ["signoff", "veto", "auto"])
def test_refused_without_notion_write_guard(w, monkeypatch, level):
    w.level = level
    w.guard = False
    out = _call(_args())
    assert "notion.write_guard.enabled" in out["error"]
    assert w.requests == [] and w.posts == []
    assert "notion.write_guard.enabled" in _run_async(zt.run_due_pending(w.config()))[0]["error"]
    # A guard block that isn't `enabled: true` doesn't count.
    for guard in (True, {"enabled": "yes"}, {}):
        cfg = w.config()
        cfg["notion"] = {"write_guard": guard}
        monkeypatch.setattr(zt, "_config", lambda c=cfg: c)
        assert "notion.write_guard" in _call(_args())["error"]


def test_publish_level_config_error_refuses(w, monkeypatch):
    monkeypatch.setattr(zt, "_config", lambda: {"autonomy": {"actions": {
        "zendesk_create_draft": "publish"}}})
    out = _call(_args())
    assert "Built levels" in out["error"]
    assert w.requests == []


@pytest.mark.parametrize("bad", [
    {"release_asset_row_id": ""}, {"release_asset_row_id": "../x"},
    {"section_id": "42/../1"}, {"section_id": ""}, {"locale": "../en"},
    {"revision_of_article_id": "1/../2"},
])
def test_invalid_args_refused(w, bad):
    out = _call(_args(**bad))
    assert "error" in out
    assert w.writes == []


def test_missing_default_permission_group_refused(w, monkeypatch):
    cfg = w.config()
    del cfg["zendesk"]["default_permission_group_id"]
    monkeypatch.setattr(zt, "_config", lambda: cfg)
    assert "default_permission_group_id" in _call(_args())["error"]
    assert w.writes == []


# ── signoff: content from the row, once per row ──────────────────────────

def test_signoff_requires_row(w):
    out = _call(_args(release_asset_row_id=""))
    assert "release_asset_row_id" in out["error"]
    assert w.writes == []


@pytest.mark.parametrize("page,why", [
    (_page(status="Draft"), "Approved"),
    (_page(status="In review"), "Approved"),
    (_page(status="Published"), "Published"),
    (_page(asset_type="Slack post"), "Help article"),
    (_page(destination="support.base.ai"), "everafterai.zendesk.com"),
    (_page(parent_ds="ffffffffffffffffffffffffffffffff"), "Release Assets"),
    (_page(in_trash=True), "trash"),
    (_page(archived=True), "archived"),
    (_page(title=""), "title"),
])
def test_signoff_refuses_bad_rows(w, page, why):
    w.row = page
    out = _call(_args())
    assert why in out["error"]
    assert w.writes == []


def test_signoff_notion_unreachable_fails_closed(w):
    w.row = gate.GateError("Notion API error (HTTP 500)")
    assert "500" in _call(_args())["error"]
    assert w.writes == []


def test_body_comes_from_the_row_not_args(w):
    w.blocks = _blocks("Earn <points>.", "Redeem them.")
    out = _call(_args(title="INJECTED", body="<script>x</script>", draft=False,
                      notify_subscribers=True, permission_group_id=1, user_segment_id=2,
                      article={"draft": False}))
    assert out["ok"] is True and out["article_id"] == 500 and out["draft"] is True
    [(method, path, body)] = w.writes
    assert method == "POST" and path == "api/v2/help_center/en-us/sections/42/articles.json"
    assert body == {"article": {"title": "Account points", "locale": "en-us",
                                "body": "<p>Earn &lt;points&gt;.</p><p>Redeem them.</p>",
                                "draft": True, "permission_group_id": 7, "user_segment_id": 12},
                    "notify_subscribers": False}
    assert out["edit_url"] == f"{BASE}/knowledge/articles/500/en-us"
    assert [a for a, _ in w.audits] == ["autonomy_signoff"]


def test_reused_row_returns_existing_draft(w):
    first = _call(_args())
    second = _call(_args(section_id="43"))
    assert second["already_exists"] is True and second["article_id"] == first["article_id"]
    assert len(w.writes) == 1


def test_concurrent_create_for_same_row_is_refused(w):
    with au.exclusive("zendesk_create_draft", ROW_KEY):
        out = _call(_args())
    assert "already running" in out["error"]
    assert w.writes == []


def test_failed_post_leaves_no_ledger_and_can_retry(w, monkeypatch):
    async def boom(base, token, path, method="GET", json_body=None):
        return {"status": 422, "text": '{"error": "invalid section"}'}
    monkeypatch.setattr(zt, "_zendesk_request", boom)
    assert "422" in _call(_args())["error"]
    assert au.ledger_get("zendesk_create_draft", ROW_KEY) is None
    monkeypatch.setattr(zt, "_zendesk_request", w.zendesk)
    assert _call(_args())["ok"] is True


def test_images_refused_unless_omitted(w):
    img = {"type": "image", "has_children": False,
           "image": {"type": "file", "file": {"url": "https://s3/x.png"},
                     "caption": [rt("Points screen")]}}
    w.blocks = _blocks("Intro") + [img]
    out = _call(_args())
    assert "image" in out["error"] and "Points screen" in out["error"]
    assert w.writes == []
    out = _call(_args(omit_images=True))
    assert out["ok"] is True and out["images_omitted"] == ["Points screen"]
    [(_m, _p, body)] = w.writes
    assert body["article"]["body"] == "<p>Intro</p>"


def test_unsupported_blocks_refused(w):
    w.blocks = _blocks("x") + [{"type": "embed", "embed": {}, "has_children": False}]
    assert "embed" in _call(_args())["error"]
    assert w.writes == []


def test_revision_takes_visibility_from_live_article(w):
    out = _call(_args(section_id=None, revision_of_article_id="77"))
    assert out["reconcile_required"] is True
    assert out["reconcile_with"] == {"id": 77, "title": "Live <b>article</b>",
                                     "html_url": f"{BASE}/hc/en-us/articles/77"}
    assert w.requests[0][:2] == ("GET", "api/v2/help_center/articles/77.json")
    [(method, path, body)] = w.writes
    assert (method, path) == ("POST", "api/v2/help_center/en-us/sections/42/articles.json")
    assert body["article"]["permission_group_id"] == 9
    assert body["article"]["user_segment_id"] == 33


def test_revision_of_missing_article_refused(w, monkeypatch):
    async def missing(base, token, path, method="GET", json_body=None):
        w.requests.append((method, path, json_body))
        return {"status": 404, "text": '{"error": "RecordNotFound"}'}
    monkeypatch.setattr(zt, "_zendesk_request", missing)
    assert "404" in _call(_args(revision_of_article_id="77"))["error"]
    assert w.writes == []


# ── veto ─────────────────────────────────────────────────────────────────

def _queue(w, **over):
    w.level = "veto"
    w.row = _page(status="Draft", **over)
    out = _call(_args())
    assert out["status"] == "pending", out
    return out["pending_id"]


def test_veto_announces_and_queues(w):
    pid = _queue(w)
    assert w.writes == []
    out = _call({"list_pending": True})
    [p] = out["pending"]
    assert p["pending_id"] == pid and p["title"] == "Account points"
    assert p["not_before"] == (NOW + timedelta(hours=4)).isoformat()
    assert p["expires_at"] == (NOW + timedelta(hours=4 + 72)).isoformat()
    [(channel, text, thread)] = w.posts
    assert channel == "CHOME" and thread == "" and pid in text
    assert "<https://www.notion.so/row-aaaa|this Release Assets row>" in text
    rec = au.load_pending("zendesk_create_draft", pid)
    assert au.verify_pending(rec) and rec["signed"]["announcement_ts"] == "111.222"


def test_veto_announcement_escapes_slack_markup(w):
    w.article["title"] = "<!channel> & <https://evil.example|click>"
    w.level = "veto"
    w.row = _page(status="Draft", title="<!here> *x* <https://evil.example|y>",
                  url="https://evil.example/row")
    out = _call(_args(section_id=None, revision_of_article_id="77"))
    assert out["status"] == "pending"
    text = w.posts[0][1]
    assert "<!here>" not in text and "<!channel>" not in text
    assert "&lt;!here&gt;" in text and "&amp;" in text
    assert "<https://evil.example" not in text  # only escaped, never live markup
    assert "this Release Assets row" in text  # label kept, foreign link dropped


def test_veto_requires_announce_channel(w, monkeypatch):
    w.level = "veto"
    w.row = _page(status="Draft")
    cfg = w.config()
    cfg["autonomy"]["announce_channel"] = ""
    monkeypatch.setattr(zt, "_config", lambda: cfg)
    assert "announce_channel" in _call(_args())["error"]
    assert au.list_pending("zendesk_create_draft") == []


def test_veto_with_approved_row_acts_now(w):
    w.level = "veto"
    out = _call(_args())
    assert out["ok"] is True and len(w.writes) == 1 and w.posts == []


def test_veto_still_refuses_published_held_or_trashed_row(w):
    w.level = "veto"
    for page in (_page(status="Published"), _page(status="Hold"), _page(in_trash=True)):
        w.row = page
        assert "error" in _call(_args())
    assert w.writes == [] and w.posts == []


def test_veto_execute_too_early(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=3)
    assert "veto window" in _call({"pending_id": pid})["error"]
    assert w.writes == []


@pytest.mark.parametrize("veto", ["reaction", "hold", "trash"])
def test_veto_execute_refuses_on_veto(w, veto):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=5)
    if veto == "reaction":
        w.vetoers = ["UPM"]
    elif veto == "hold":
        w.row = _page(status="Hold")
    else:
        w.row = _page(status="Draft", in_trash=True)
    out = _call({"pending_id": pid})
    assert out["status"] == "vetoed" and w.writes == []
    assert au.load_pending("zendesk_create_draft", pid)["status"] == "vetoed"
    assert "error" in _call({"pending_id": pid})  # never re-runs


def test_veto_execute_uses_row_content_at_execution_and_audits(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=5)
    out = _call({"pending_id": pid, "release_asset_row_id": "b" * 32, "section_id": "1"})
    assert out["ok"] is True and out["level"] == "veto"
    [(_m, path, body)] = w.writes
    assert path.endswith("/sections/42/articles.json")
    assert body["article"]["body"] == "<p>Earn points.</p>"
    assert "autonomy_veto" in [a for a, _ in w.audits]
    assert w.posts[-1][0] == "CHOME" and w.posts[-1][2] == "111.222"
    assert au.load_pending("zendesk_create_draft", pid)["status"] == "done"
    assert "error" in _call({"pending_id": pid})  # not twice


def test_veto_content_changed_after_announcement_is_stale(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=5)
    w.blocks = _blocks("Something the team never saw.")
    out = _call({"pending_id": pid})
    assert out["status"] == "stale" and w.writes == []
    assert au.load_pending("zendesk_create_draft", pid)["status"] == "stale"


def test_expired_pending_never_executes(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=4 + 72, minutes=1)
    out = _call({"pending_id": pid})
    assert out["status"] == "expired" and w.writes == []
    assert au.load_pending("zendesk_create_draft", pid)["status"] == "expired"


def test_tampered_pending_record_refused(w):
    pid = _queue(w)
    path = au.autonomy_dir() / "pending" / "zendesk_create_draft" / f"{pid}.json"
    rec = json.loads(path.read_text())
    rec["signed"]["not_before"] = NOW.isoformat()  # skip the window
    path.write_text(json.dumps(rec))
    w.now = NOW + timedelta(minutes=1)
    assert "integrity" in _call({"pending_id": pid})["error"]
    assert w.writes == []
    assert ("autonomy_tampered", f"pending={pid}") in w.audits


def test_double_execution_creates_once(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=5)
    # A direct signoff-style call for the same row already created the draft.
    w.row = _page(status="Approved")
    w.level = "signoff"
    first = _call(_args())
    assert first["ok"] is True
    w.level = "veto"
    out = _call({"pending_id": pid})
    assert out["already_exists"] is True and out["article_id"] == first["article_id"]
    assert len(w.writes) == 1


def test_pending_execution_rechecks_level(w):
    pid = _queue(w)
    w.now = NOW + timedelta(hours=5)
    w.level = "off"
    assert "error" in _call({"pending_id": pid})
    w.level = "signoff"  # demoted: now needs the row Approved
    assert "Approved" in _call({"pending_id": pid})["error"]
    assert au.load_pending("zendesk_create_draft", pid)["status"] == "pending"
    w.row = _page(status="Approved")
    assert _call({"pending_id": pid})["ok"] is True
    assert len(w.writes) == 1


def test_run_due_executes_due_and_expires_old(w):
    early = _queue(w)
    w.now = NOW + timedelta(hours=1)
    late = _queue(w, title="Account points")  # same row → one draft at most
    w.now = NOW + timedelta(hours=4, minutes=30)
    out = _call({"run_due": True})
    results = {r["pending_id"]: r for r in out["results"]}
    assert results[early]["ok"] is True and late not in results  # late not due yet
    assert len(w.writes) == 1
    w.now = NOW + timedelta(hours=200)
    out = _call({"run_due": True})
    assert {r["pending_id"]: r["status"] for r in out["results"]} == {late: "expired"}


def test_unknown_pending_id(w):
    assert "error" in _call({"pending_id": "0123456789ab"})
    assert "error" in _call({"pending_id": "../../x"})


# ── auto ─────────────────────────────────────────────────────────────────

def test_auto_creates_and_announces(w):
    w.level = "auto"
    w.row = _page(status="Draft")
    out = _call(_args())
    assert out["ok"] is True and out["level"] == "auto" and len(w.writes) == 1
    assert w.posts and w.posts[0][0] == "CHOME"
    assert f"<{BASE}/knowledge/articles/500/en-us|open the draft>" in w.posts[0][1]
    assert [a for a, _ in w.audits] == ["autonomy_auto"]


def test_auto_still_needs_row(w):
    w.level = "auto"
    assert "release_asset_row_id" in _call(_args(release_asset_row_id=""))["error"]
    w.row = _page(status="Published")
    assert "Published" in _call(_args())["error"]
    assert w.writes == []


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
    assert "Approved" in _call(_args())["error"]
    assert w.writes == []
