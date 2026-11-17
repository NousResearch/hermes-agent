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
    return {"notion": {"write_guard": guard}}


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

@pytest.mark.parametrize("value,ok", [("In progress", True), ("Done", True), ("Not needed", False),
                                      ("Not started", False)])
def test_prd_only_release_comms_status(lk, value, ok):
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
