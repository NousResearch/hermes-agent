"""Notion write guard (education design, dependency 5a): the rule engine.

Every Notion lookup goes through a fake ``lookup`` so these tests pin the
rules, the target resolution (block → page → row → data source) and the
fail-closed behaviour without a Notion workspace."""
import pytest

import tools.notion_write_guard as g

KITS = "1" * 32
ASSETS = "2" * 32
PRDS = "3" * 32
IDEAS = "4" * 32
OTHER_DS = "5" * 32
HOME = "6" * 32
BOT = "bot-user-id"


def _dash(h):
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def cfg(**guard_over):
    guard = {
        "enabled": True,
        "release_kits": {"data_sources": [KITS, _dash(ASSETS)]},
        "prds": {"data_sources": [PRDS]},
        "ideas": {"data_sources": [IDEAS]},
        "writable_pages": [HOME],
    }
    guard.update(guard_over)
    return {"notion": {"write_guard": guard}, "agent_write_guard": True}


def _row(page_id, ds, status, *, status_type="select", created_by=BOT, extra_props=None, **kw):
    props = {
        "Name": {"id": "title", "type": "title", "title": []},
        "Status": {"id": "st%3Aa", "type": status_type, status_type: {"name": status} if status else None},
    }
    if ds == IDEAS:
        props["Rationale"] = {"id": "ra", "type": "rich_text", "rich_text": []}
        props["Decision note"] = {"id": "dn", "type": "rich_text", "rich_text": []}
    props.update(extra_props or {})
    page = {"object": "page", "id": _dash(page_id),
            "parent": {"type": "data_source_id", "data_source_id": _dash(ds), "database_id": "9" * 32},
            "created_by": {"object": "user", "id": created_by},
            "properties": props, "in_trash": False, "archived": False}
    page.update(kw)
    return page


def _prd(page_id):
    return _row(page_id, PRDS, "In review", extra_props={
        "Release Comms Status": {"id": "rcs", "type": "select", "select": {"name": "Not started"}},
        "JIRA": {"id": "jira", "type": "rich_text", "rich_text": []}})


def _ds_schema(ds):
    props = {"Name": {"id": "title", "type": "title"}, "Status": {"id": "st%3Aa", "type": "select"}}
    if ds == IDEAS:
        props["Decision note"] = {"id": "dn", "type": "rich_text"}
        props["Rationale"] = {"id": "ra", "type": "rich_text"}
    return {"object": "data_source", "id": ds, "properties": props}


class FakeLookup:
    def __init__(self):
        self.pages = {}
        self.blocks = {}
        self.calls = []
        self.fail = False

    def _norm(self, i):
        return i.replace("-", "").lower()

    def page(self, page_id):
        self.calls.append(("page", page_id))
        if self.fail:
            raise g.GuardLookupError("boom")
        try:
            return self.pages[self._norm(page_id)]
        except KeyError:
            raise g.GuardLookupError("not found")

    def block(self, block_id):
        self.calls.append(("block", block_id))
        if self.fail:
            raise g.GuardLookupError("boom")
        key = self._norm(block_id)
        if key in self.blocks:
            return self.blocks[key]
        if key in self.pages:
            return {"object": "block", "id": block_id, "type": "child_page",
                    "parent": self.pages[key]["parent"]}
        raise g.GuardLookupError("not found")

    def data_source(self, ds):
        self.calls.append(("ds", ds))
        return _ds_schema(self._norm(ds))

    def me(self):
        return BOT


@pytest.fixture
def lk():
    lookup = FakeLookup()
    lookup.pages["a" * 32] = _row("a" * 32, ASSETS, "Draft")
    lookup.pages["b" * 32] = _row("b" * 32, ASSETS, "In review")
    lookup.pages["c" * 32] = _prd("c" * 32)
    lookup.pages["d" * 32] = _row("d" * 32, IDEAS, "Accepted")
    lookup.pages["e" * 32] = _row("e" * 32, IDEAS, "Accepted", created_by="human")
    lookup.pages["f" * 32] = _row("f" * 32, OTHER_DS, "Draft")
    lookup.pages[HOME] = {"object": "page", "id": HOME, "parent": {"type": "workspace", "workspace": True},
                          "properties": {"title": {"id": "title", "type": "title", "title": []}}}
    lookup.pages["7" * 32] = {"object": "page", "id": "7" * 32,
                              "parent": {"type": "page_id", "page_id": _dash(HOME)},
                              "properties": {"title": {"id": "title", "type": "title", "title": []}}}
    # a block inside the Draft kit row, nested under another block
    lookup.blocks["0" * 31 + "1"] = {"object": "block", "type": "paragraph",
                                     "parent": {"type": "block_id", "block_id": "0" * 31 + "2"}}
    lookup.blocks["0" * 31 + "2"] = {"object": "block", "type": "toggle",
                                     "parent": {"type": "page_id", "page_id": "a" * 32}}
    lookup.blocks["0" * 31 + "3"] = {"object": "block", "type": "paragraph",
                                     "parent": {"type": "page_id", "page_id": "b" * 32}}
    return lookup


def check(lk, method, path, body=None, config=None):
    return g.check_request(method, path, body, cfg=config or cfg(), lookup=lk)


def status(name, kind="select"):
    return {kind: {"name": name}}


# ── activation ───────────────────────────────────────────────────────────

def test_not_configured_is_inert(lk):
    assert not g.is_enforcing({})
    assert not g.is_enforcing({"notion": {"write_guard": {"enabled": False, "release_kits": {}}}})
    assert g.check_request("DELETE", f"v1/pages/{'b' * 32}", None, cfg={}, lookup=lk) is None
    assert lk.calls == []


@pytest.mark.parametrize("block", [
    {"enabled": True},                                      # incomplete
    {"enabled": True, "release_kits": {"data_sources": ["nope"]},
     "prds": {"data_sources": [PRDS]}, "ideas": {"data_sources": [IDEAS]}},  # bad id
    True,                                                   # bare scalar
    {"release_kits": {}},                                   # enabled missing
    {"enabled": "yes please"},                              # not a bool
])
def test_enabled_incomplete_block_denies_writes_allows_reads(lk, block):
    config = {"notion": {"write_guard": block}}
    assert g.is_enforcing(config)
    guard = g.load_guard(config)
    assert guard is not None and guard.error
    denial = g.check_request("PATCH", f"v1/pages/{'a' * 32}", {"properties": {}}, cfg=config, lookup=lk)
    assert denial and "write_guard" in denial
    assert g.check_request("GET", f"v1/pages/{'a' * 32}", None, cfg=config, lookup=lk) is None
    assert g.check_request("POST", f"v1/data_sources/{KITS}/query", {}, cfg=config, lookup=lk) is None
    assert g.validate(config)


def test_valid_block_validates_clean():
    assert g.validate(cfg()) == []
    assert g.validate({}) == []


def test_overlapping_data_sources_rejected():
    assert g.load_guard(cfg(prds={"data_sources": [KITS]})).error


# ── universal rules ──────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [f"v1/pages/{'a' * 32}", f"v1/blocks/{'0' * 31}1", f"v1/data_sources/{KITS}"])
def test_delete_denied(lk, path):
    assert "DELETE" in check(lk, "DELETE", path)


@pytest.mark.parametrize("key,val", [("archived", True), ("in_trash", True), ("in_trash", False),
                                     ("is_archived", True)])
def test_archived_flag_denied(lk, key, val):
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {key: val})
    assert check(lk, "PATCH", f"v1/blocks/{'0' * 31}1", {key: val})


@pytest.mark.parametrize("method,path", [
    ("POST", "v1/databases"), ("PATCH", f"v1/databases/{KITS}"), ("POST", "v1/data_sources"),
    ("PATCH", f"v1/data_sources/{KITS}"), ("POST", f"v1/pages/{'a' * 32}/move"),
    ("PATCH", f"v1/pages/{'a' * 32}/properties/st"), ("POST", "v1/views"), ("PATCH", "v1/users/me"),
])
def test_schema_and_unknown_endpoints_denied(lk, method, path):
    assert check(lk, method, path, {})


@pytest.mark.parametrize("method,path", [
    ("POST", "v1/search"), ("POST", f"v1/data_sources/{KITS}/query"),
    ("POST", f"v1/databases/{KITS}/query"), ("POST", "v1/file_uploads"),
    ("POST", f"v1/file_uploads/{'a' * 32}/send"), ("GET", f"v1/pages/{'c' * 32}?filter_properties=x"),
])
def test_reads_and_uploads_allowed(lk, method, path):
    assert check(lk, method, path, {}) is None


@pytest.mark.parametrize("path", [
    f"v1/pages/{'a' * 32}/../../databases/{KITS}", f"v1/pages/{'a' * 32}?x=1",
    "v1/pages/not-an-id", "v1//pages", f"v1/pages/%2e%2e/{KITS}", "v2/pages",
])
def test_path_traversal_denied(lk, path):
    assert check(lk, "PATCH", path, {"properties": {}})


def test_lookup_error_denies(lk):
    lk.fail = True
    assert "could not" in check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {}}).lower()


def test_unknown_target_denied(lk):
    assert check(lk, "PATCH", f"v1/pages/{'9' * 32}", {"properties": {}})   # not found
    assert check(lk, "PATCH", f"v1/pages/{'f' * 32}", {"properties": {}})   # other data source
    assert check(lk, "POST", "v1/pages", {"parent": {"workspace": True}, "properties": {}})
    assert check(lk, "POST", "v1/pages", {"parent": {"database_id": KITS}, "properties": {}})


# ── Release Kits ─────────────────────────────────────────────────────────

def test_kit_create_requires_draft(lk):
    parent = {"data_source_id": ASSETS}
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {"Status": status("Draft")},
                                          "markdown": "# hi"}) is None
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {"Status": status("Approved")}})
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {"Name": {"title": []}}})
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {"Status": status("Draft")},
                                          "template": {"type": "default"}})


def test_kit_update_only_while_draft(lk):
    body = {"properties": {"Name": {"title": [{"text": {"content": "x"}}]}}}
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", body) is None
    assert "read-only" in check(lk, "PATCH", f"v1/pages/{'b' * 32}", body)


@pytest.mark.parametrize("value", ["In review", "Approved", "Published"])
def test_kit_status_beyond_draft_denied(lk, value):
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"Status": status(value)}})
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"Status": status("Draft")}}) is None


def test_status_by_option_id_denied(lk):
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"Status": {"select": {"id": "x"}}}})
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}",
                 {"properties": {"Status": {"select": {"id": "x", "name": "Draft"}}}})
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"Status": {"select": None}}})


def test_property_by_id_is_normalised(lk):
    # "st%3Aa" is the Status property's id; both encodings must be recognised.
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"st%3Aa": status("Approved")}})
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"st:a": status("Approved")}})
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"zzz": {"rich_text": []}}})


def test_kit_status_type_property(lk):
    lk.pages["a" * 32] = _row("a" * 32, ASSETS, "Draft", status_type="status")
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"Status": status("Draft", "status")}}) is None
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {"Status": status("Approved", "status")}})


def test_kit_content_edit_via_block_children_resolves_parent(lk):
    md = {"type": "insert_content", "insert_content": {"content": "x"}}
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}/markdown", md) is None
    assert check(lk, "PATCH", f"v1/pages/{'b' * 32}/markdown", md)
    # block nested two levels inside the Draft row
    assert check(lk, "PATCH", f"v1/blocks/{'0' * 31}1/children", {"children": []}) is None
    assert check(lk, "PATCH", f"v1/blocks/{'0' * 31}1", {"paragraph": {"rich_text": []}}) is None
    # block inside the In-review row
    assert check(lk, "PATCH", f"v1/blocks/{'0' * 31}3", {"paragraph": {"rich_text": []}})
    # the page id used as a block id
    assert check(lk, "PATCH", f"v1/blocks/{'a' * 32}/children", {"children": []}) is None
    assert check(lk, "PATCH", f"v1/blocks/{'b' * 32}/children", {"children": []})


def test_notion_lookup_caches_blocks_not_pages():
    g.clear_caches()
    seen = []

    def run(path):
        seen.append(path)
        if path == "v1/users/me":
            return {"id": BOT}
        return {"object": "x", "id": path}

    lookup = g.NotionLookup(run=run)
    for _ in range(2):
        lookup.block("0" * 32)
        lookup.page("a" * 32)
        lookup.data_source(KITS)
        lookup.me()
    assert seen.count(f"v1/blocks/{'0' * 32}") == 1
    assert seen.count(f"v1/pages/{'a' * 32}") == 2      # page state is always fresh
    assert seen.count(f"v1/data_sources/{KITS}") == 1
    assert seen.count("v1/users/me") == 1


def test_notion_lookup_error_shapes():
    def run(path):
        raise g.GuardLookupError("HTTP 404")
    with pytest.raises(g.GuardLookupError):
        g.NotionLookup(run=run).page("a" * 32)


def test_trashed_row_denied(lk):
    lk.pages["a" * 32] = _row("a" * 32, ASSETS, "Draft", in_trash=True)
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {}})


def test_comments(lk):
    assert check(lk, "POST", "v1/comments", {"parent": {"page_id": "a" * 32}, "rich_text": []}) is None
    assert check(lk, "POST", "v1/comments", {"parent": {"page_id": "c" * 32}, "rich_text": []})
    assert check(lk, "POST", "v1/comments", {"discussion_id": "x", "rich_text": []})


# ── PRDs ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("current,value,ok", [
    ("Not started", "In progress", True), ("Not started", "Done", False),
    ("In progress", "Done", True), ("In progress", "In progress", True),
    ("Not started", "Not needed", False), ("Not needed", "In progress", False),
    ("A part of a bigger module", "In progress", False), ("Done", "In progress", False),
])
def test_prd_release_comms_status_transitions(lk, current, value, ok):
    lk.pages["c" * 32]["properties"]["Release Comms Status"]["select"] = {"name": current}
    out = check(lk, "PATCH", f"v1/pages/{'c' * 32}", {"properties": {"Release Comms Status": status(value)}})
    assert (out is None) is ok


def test_prd_other_properties_and_content_denied(lk):
    assert check(lk, "PATCH", f"v1/pages/{'c' * 32}", {"properties": {"Status": status("Done")}})
    assert check(lk, "PATCH", f"v1/pages/{'c' * 32}", {"properties": {
        "Release Comms Status": status("Done"), "JIRA": {"rich_text": []}}})
    assert check(lk, "PATCH", f"v1/pages/{'c' * 32}", {"icon": {"emoji": "x"}})
    assert check(lk, "PATCH", f"v1/pages/{'c' * 32}/markdown", {"type": "insert_content"})
    assert check(lk, "POST", "v1/pages", {"parent": {"data_source_id": PRDS}, "properties": {}})


# ── Ideas ────────────────────────────────────────────────────────────────

def test_ideas_create_proposed(lk):
    parent = {"data_source_id": IDEAS}
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {"Status": status("Proposed")}}) is None
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {"Status": status("Accepted")}})
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {}})
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {
        "Status": status("Proposed"), "Decision note": {"rich_text": []}}})


def test_ideas_own_transitions(lk):
    own = f"v1/pages/{'d' * 32}"
    assert check(lk, "PATCH", own, {"properties": {"Status": status("In progress")}}) is None
    assert check(lk, "PATCH", own, {"properties": {"Status": status("Done")}})  # skips In progress
    lk.pages["d" * 32] = _row("d" * 32, IDEAS, "In progress")
    assert check(lk, "PATCH", own, {"properties": {"Status": status("Done")}}) is None
    lk.pages["d" * 32] = _row("d" * 32, IDEAS, "Proposed")
    assert check(lk, "PATCH", own, {"properties": {"Status": status("In progress")}})
    assert check(lk, "PATCH", own, {"properties": {"Rationale": {"rich_text": []}}}) is None


@pytest.mark.parametrize("value", ["Accepted", "Rejected", "Parked"])
def test_ideas_human_status_denied(lk, value):
    lk.pages["d" * 32] = _row("d" * 32, IDEAS, "Proposed")
    assert check(lk, "PATCH", f"v1/pages/{'d' * 32}", {"properties": {"Status": status(value)}})


def test_ideas_human_property_denied(lk):
    assert check(lk, "PATCH", f"v1/pages/{'d' * 32}", {"properties": {"Decision note": {"rich_text": []}}})


def test_ideas_not_own_read_only(lk):
    assert check(lk, "PATCH", f"v1/pages/{'e' * 32}", {"properties": {"Rationale": {"rich_text": []}}})
    assert check(lk, "PATCH", f"v1/pages/{'e' * 32}/markdown", {"type": "insert_content"})


# ── free pages, autonomy ─────────────────────────────────────────────────

def test_free_pages_writable(lk):
    assert check(lk, "PATCH", f"v1/pages/{HOME}/markdown", {"type": "insert_content"}) is None
    assert check(lk, "PATCH", f"v1/pages/{'7' * 32}", {"properties": {"title": {"title": []}}}) is None
    assert check(lk, "POST", "v1/pages", {"parent": {"page_id": HOME}, "properties": {}}) is None
    assert check(lk, "PATCH", f"v1/pages/{'7' * 32}", {"in_trash": True})
    assert check(lk, "POST", "v1/pages", {"parent": {"page_id": "c" * 32}, "properties": {}})  # under a PRD


@pytest.mark.parametrize("action", ["advance_asset_status", "ideas_accept_own"])
@pytest.mark.parametrize("level", ["signoff", "veto", "auto"])
def test_unbuilt_autonomy_level_breaks_guard(lk, action, level):
    config = cfg()
    config["autonomy"] = {"actions": {action: level}}
    guard = g.load_guard(config)
    assert guard.error and action in guard.error
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", {"properties": {}}, config=config)


def test_off_autonomy_levels_are_fine(lk):
    config = cfg()
    config["autonomy"] = {"actions": {"advance_asset_status": "off", "ideas_accept_own": "off"}}
    assert g.load_guard(config).error == ""


# ── review round: relations, hidden deletes, self-edit tie ───────────────

def _with_relation(row, ids, has_more=False):
    row["properties"]["Release"] = {"id": "rel", "type": "relation",
                                    "relation": [{"id": i} for i in ids], "has_more": has_more}
    return row


def test_relation_change_must_touch_only_writable_pages(lk):
    approved = "b" * 32                      # In review kit row
    draft = "a" * 32
    other_draft = "8" * 32
    lk.pages[other_draft] = _row(other_draft, ASSETS, "Draft")
    _with_relation(lk.pages[draft], [other_draft])
    rel = lambda ids: {"properties": {"Release": {"relation": [{"id": i} for i in ids]}}}
    # adding a Draft row: fine; unchanged list: fine
    assert check(lk, "PATCH", f"v1/pages/{draft}", rel([other_draft])) is None
    # adding a non-Draft row (its back-relation would change) → refused
    assert check(lk, "PATCH", f"v1/pages/{draft}", rel([other_draft, approved]))
    # removing a link to an approved row → refused
    _with_relation(lk.pages[draft], [approved])
    assert check(lk, "PATCH", f"v1/pages/{draft}", rel([]))
    # linking a PRD → refused
    assert check(lk, "PATCH", f"v1/pages/{draft}", rel(["c" * 32]))
    # truncated current list → refused
    _with_relation(lk.pages[draft], [other_draft], has_more=True)
    assert check(lk, "PATCH", f"v1/pages/{draft}", rel([other_draft]))
    # malformed value
    assert check(lk, "PATCH", f"v1/pages/{draft}", {"properties": {"Release": {"relation": "x"}}})


def test_relation_on_create_checked(lk, monkeypatch):
    orig = _ds_schema

    def schema(ds):
        out = orig(ds)
        out["properties"]["Release"] = {"id": "rel", "type": "relation"}
        return out
    monkeypatch.setattr(lk, "data_source", lambda ds: schema(ds.replace("-", "")))
    body = lambda ids: {"parent": {"data_source_id": ASSETS}, "properties": {
        "Status": status("Draft"), "Release": {"relation": [{"id": i} for i in ids]}}}
    assert check(lk, "POST", "v1/pages", body(["a" * 32])) is None
    assert check(lk, "POST", "v1/pages", body(["b" * 32]))


@pytest.mark.parametrize("body", [
    {"type": "replace_content", "replace_content": {"new_str": "x", "allow_deleting_content": True}},
    {"children": [{"type": "child_database", "child_database": {"title": "x"}}]},
    {"children": [{"type": "toggle", "toggle": {"children": [{"child_database": {}}]}}]},
])
def test_hidden_deletes_and_schema_denied(lk, body):
    assert check(lk, "PATCH", f"v1/pages/{HOME}/markdown", body)
    assert check(lk, "PATCH", f"v1/blocks/{'a' * 32}/children", body)


def test_guard_requires_agent_write_guard(lk):
    config = cfg()
    del config["agent_write_guard"]
    guard = g.load_guard(config)
    assert guard.error and "agent_write_guard" in guard.error
    assert check(lk, "PATCH", f"v1/pages/{HOME}/markdown", {}, config=config)
    config["agent_write_guard"] = {"enabled": True, "extra_paths": ["x"]}
    assert g.load_guard(config).error == ""
    config["agent_write_guard"] = {"enabled": False}
    assert g.load_guard(config).error



# ── second review round: the Release → PRD link, strict parents, synced blocks ──

def _kit_schema_with_prd(kind="dual_property", synced="Release Kit", target=PRDS):
    def schema(ds):
        out = _ds_schema(ds.replace("-", ""))
        rel = {"data_source_id": target, "type": kind}
        if kind == "dual_property":
            rel["dual_property"] = {"synced_property_name": synced, "synced_property_id": "x"}
        if kind:
            out["properties"]["PRD"] = {"id": "prd", "type": "relation", "relation": rel}
        return out
    return schema


def _link_prd(lk, ids):
    lk.pages["a" * 32]["properties"]["PRD"] = {"id": "prd", "type": "relation",
                                               "relation": [{"id": i} for i in ids], "has_more": False}


PRD_CFG = cfg(prds={"data_sources": [PRDS], "back_relation_property": "Release Kit"})


def _create_with_prd(ids):
    return {"parent": {"data_source_id": KITS},
            "properties": {"Status": status("Draft"), "PRD": {"relation": [{"id": i} for i in ids]}}}


def test_prd_link_only_in_the_create_body(lk, monkeypatch):
    monkeypatch.setattr(lk, "data_source", _kit_schema_with_prd())
    assert check(lk, "POST", "v1/pages", _create_with_prd(["c" * 32]), config=PRD_CFG) is None
    assert check(lk, "POST", "v1/pages", _create_with_prd(["b" * 32]), config=PRD_CFG)  # not a PRD
    # any later change — adding to an empty link included — is refused
    path = f"v1/pages/{'a' * 32}"
    body = lambda ids: {"properties": {"PRD": {"relation": [{"id": i} for i in ids]}}}
    for current, new in (([], ["c" * 32]), (["c" * 32], []), (["c" * 32], ["c" * 31 + "d"])):
        _link_prd(lk, current)
        assert "set only when the row is created" in check(lk, "PATCH", path, body(new), config=PRD_CFG)
    _link_prd(lk, ["c" * 32])
    assert check(lk, "PATCH", path, body(["c" * 32]), config=PRD_CFG) is None  # unchanged: no-op


def test_prd_link_back_relation_must_be_the_configured_one(lk, monkeypatch):
    create = _create_with_prd(["c" * 32])
    monkeypatch.setattr(lk, "data_source", _kit_schema_with_prd(synced="Owner notes"))
    assert check(lk, "POST", "v1/pages", create, config=PRD_CFG)
    monkeypatch.setattr(lk, "data_source", _kit_schema_with_prd())
    assert check(lk, "POST", "v1/pages", create, config=cfg())          # back relation not configured
    monkeypatch.setattr(lk, "data_source", _kit_schema_with_prd(kind="single_property"))
    assert check(lk, "POST", "v1/pages", create, config=cfg()) is None  # far side untouched
    monkeypatch.setattr(lk, "data_source", _kit_schema_with_prd(kind=""))
    assert check(lk, "POST", "v1/pages", create, config=PRD_CFG)        # schema silent
    monkeypatch.setattr(lk, "data_source", _kit_schema_with_prd(target=IDEAS))
    assert check(lk, "POST", "v1/pages", create, config=PRD_CFG)        # points elsewhere


def test_single_property_relation_targets_unchecked(lk, monkeypatch):
    def schema(ds):
        out = _ds_schema(ds.replace("-", ""))
        out["properties"]["Release"] = {"id": "rel", "type": "relation",
                                        "relation": {"data_source_id": KITS, "type": "single_property"}}
        return out
    monkeypatch.setattr(lk, "data_source", schema)
    _with_relation(lk.pages["a" * 32], [])
    body = {"properties": {"Release": {"relation": [{"id": "b" * 32}]}}}  # an In-review row
    assert check(lk, "PATCH", f"v1/pages/{'a' * 32}", body) is None


@pytest.mark.parametrize("parent", [
    {"type": "page_id", "data_source_id": ASSETS},
    {"data_source_id": ASSETS, "page_id": HOME},
    {"data_source_id": ASSETS, "extra": 1},
    {},
])
def test_create_parent_must_be_exact(lk, parent):
    assert check(lk, "POST", "v1/pages", {"parent": parent, "properties": {"Status": status("Draft")}})


def test_create_parent_with_matching_type_ok(lk):
    assert check(lk, "POST", "v1/pages", {"parent": {"type": "data_source_id", "data_source_id": ASSETS},
                                          "properties": {"Status": status("Draft")}}) is None


def test_synced_blocks_denied(lk):
    body = {"children": [{"type": "synced_block", "synced_block": {"synced_from": {"block_id": "x"}}}]}
    assert "synced" in check(lk, "PATCH", f"v1/blocks/{'a' * 32}/children", body)
