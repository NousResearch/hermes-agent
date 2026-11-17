"""zendesk_api (read) and the draft-only write chokepoint. Network (_zendesk_request)
and credentials are mocked; these pin the Help Center path allowlist, host
handling, get_secret-only credentials, and the write invariants."""
import json

import pytest

import tools.zendesk_tool as zt
from model_tools import _run_async

BASE = "https://everafterai.zendesk.com"


@pytest.fixture
def zd_ok(monkeypatch):
    monkeypatch.setattr(zt, "_zendesk_creds", lambda: (BASE, "cid", "secret"))

    async def token(scope, *, force=False):
        return "tok"
    monkeypatch.setattr(zt, "_access_token", token)


@pytest.fixture
def no_network(monkeypatch):
    async def boom(*a, **k):
        pytest.fail("must not make a request")
    monkeypatch.setattr(zt, "_zendesk_request", boom)


def _read(args):
    return json.loads(_run_async(zt._zendesk_api_handler(args)))


def test_registered_toolsets_are_separate():
    import toolsets
    from tools.registry import registry
    assert registry.get_toolset_for_tool("zendesk_api") == "zendesk"
    assert registry.get_toolset_for_tool("zendesk_create_draft") == "zendesk_write"
    assert toolsets.TOOLSETS["zendesk"]["tools"] == ["zendesk_api"]
    assert toolsets.TOOLSETS["zendesk_write"]["tools"] == ["zendesk_create_draft"]
    # Not in any built-in role.
    from gateway.tool_access import BUILTIN_ROLES
    for role, grant in BUILTIN_ROLES.items():
        if "*" in grant:
            continue
        assert "zendesk" not in grant and "zendesk_write" not in grant, role


def test_read_passes_path_and_parses_json(monkeypatch, zd_ok):
    seen = {}

    async def fake(base, token, path, method="GET", json_body=None):
        seen.update(base=base, token=token, path=path, method=method)
        return {"status": 200, "text": '{"categories": [{"id": 1}]}'}

    monkeypatch.setattr(zt, "_zendesk_request", fake)
    out = _read({"path": "api/v2/help_center/en-us/categories.json?page[size]=100"})
    assert out["ok"] is True and out["data"]["categories"] == [{"id": 1}]
    assert seen == {"base": BASE, "token": "tok",
                    "path": "api/v2/help_center/en-us/categories.json?page[size]=100",
                    "method": "GET"}


@pytest.mark.parametrize("path", [
    "api/v2/help_center/categories.json",
    "api/v2/help_center/en-us/sections.json",
    "api/v2/help_center/categories/360001/sections.json",
    "api/v2/help_center/en-us/categories/360001/articles.json",
    "api/v2/help_center/sections/42/articles.json",
    "api/v2/help_center/articles/99.json",
    "api/v2/help_center/en-us/articles/99",
    "api/v2/help_center/articles/99/translations.json",
    "api/v2/help_center/articles/99/translations/en-us.json",
    "api/v2/help_center/articles/99/attachments.json",
    "api/v2/help_center/articles/search.json?query=points",
    "api/v2/help_center/user_segments.json",
    "api/v2/help_center/locales.json",
    "api/v2/guide/permission_groups.json",
    "/api/v2/help_center/sections.json",
])
def test_read_allowlist_accepts_help_center(path):
    assert zt._normalize_read_path(path, BASE) is not None


def test_read_accepts_own_host_full_url_and_refuses_other_hosts(monkeypatch, zd_ok):
    seen = {}

    async def fake(base, token, path, method="GET", json_body=None):
        seen["path"] = path
        return {"status": 200, "text": "{}"}

    monkeypatch.setattr(zt, "_zendesk_request", fake)
    nxt = f"{BASE}/api/v2/help_center/en-us/articles.json?page[after]=abc&page[size]=100"
    assert _read({"path": nxt})["ok"] is True
    assert seen["path"] == "api/v2/help_center/en-us/articles.json?page[after]=abc&page[size]=100"
    for bad in ("https://evil.example.com/api/v2/help_center/articles.json",
                "https://everafterai.zendesk.com.evil.io/api/v2/help_center/articles.json",
                "http://everafterai.zendesk.com/api/v2/help_center/articles.json",
                "https://other.zendesk.com/api/v2/help_center/articles.json"):
        assert zt._normalize_read_path(bad, BASE) is None, bad


@pytest.mark.parametrize("path", [
    "api/v2/tickets.json",
    "api/v2/tickets/1/comments.json",
    "api/v2/users.json",
    "api/v2/help_center/../tickets.json",
    "api/v2/help_center/articles/1/../../../users.json",
    "api/v2/help_center//articles.json",
    "api/v2/help_center/articles/1%2F..%2F..%2Ftickets.json",
    "api/v2/help_center/articles/abc.json",
    "api/v2/help_center/articles.json#frag",
    "api/v2/help_center/articles .json",
    "api/v2/help_center/articles/1/votes.json",
    "api/v2/help_center/articles/1/comments.json",
    "api/v2/help_center/incremental/articles.json",
    "",
])
def test_read_refuses_tickets_and_traversal(path, zd_ok, no_network):
    assert zt._normalize_read_path(path, BASE) is None
    assert "error" in _read({"path": path})


def test_read_refuses_non_get(zd_ok, no_network):
    out = _read({"path": "api/v2/help_center/articles.json", "method": "POST"})
    assert "error" in out


def test_read_unconfigured(monkeypatch, no_network):
    monkeypatch.setattr(zt, "_zendesk_creds", lambda: ("", "", ""))
    out = _read({"path": "api/v2/help_center/articles.json"})
    assert "not configured" in out["error"]
    assert zt._check_zendesk() is False


def test_creds_use_get_secret_not_environ(monkeypatch):
    import agent.secret_scope as ss
    monkeypatch.setenv("ZENDESK_SUBDOMAIN", "from-environ")
    vals = {"ZENDESK_SUBDOMAIN": "everafterai", "ZENDESK_OAUTH_CLIENT_ID": "cid",
            "ZENDESK_OAUTH_CLIENT_SECRET": "secret"}
    monkeypatch.setattr(ss, "get_secret", lambda name, default=None: vals.get(name, default))
    assert zt._zendesk_creds() == (BASE, "cid", "secret")
    assert zt._check_zendesk() is True


def test_creds_unscoped_secret_is_not_configured(monkeypatch):
    import agent.secret_scope as ss

    def unscoped(name, default=None):
        raise ss.UnscopedSecretError(name)

    monkeypatch.setattr(ss, "get_secret", unscoped)
    assert zt._zendesk_creds() == ("", "", "")
    assert zt._check_zendesk() is False


def test_creds_reject_bad_subdomain(monkeypatch):
    import agent.secret_scope as ss
    for sub in ("evil.com/x", "a b", "everafterai.zendesk.com", "-x"):
        vals = {"ZENDESK_SUBDOMAIN": sub, "ZENDESK_OAUTH_CLIENT_ID": "c",
                "ZENDESK_OAUTH_CLIENT_SECRET": "s"}
        monkeypatch.setattr(ss, "get_secret", lambda n, d=None, v=vals: v.get(n, d))
        assert zt._zendesk_creds() == ("", "", ""), sub


def _draft_body(**article):
    art = {"title": "T", "locale": "en-us", "body": "<p>x</p>", "draft": True,
           "permission_group_id": 7}
    art.update(article)
    return {"article": art, "notify_subscribers": False}


def test_write_allowlist_is_section_article_create_only():
    assert zt._write_allowed("POST", "api/v2/help_center/en-us/sections/42/articles.json")
    assert zt._write_allowed("POST", "api/v2/help_center/sections/42/articles.json")
    for method, path in [
        ("PUT", "api/v2/help_center/articles/1.json"),
        ("PUT", "api/v2/help_center/articles/1/translations/en-us.json"),
        ("DELETE", "api/v2/help_center/articles/1.json"),
        ("POST", "api/v2/help_center/articles/1/translations.json"),
        ("POST", "api/v2/help_center/en-us/sections/42/articles.json?x=1"),
        ("POST", "api/v2/help_center/sections/42/../../articles.json"),
        ("POST", "api/v2/tickets.json"),
        ("PUT", "api/v2/help_center/articles/1/source_locale.json"),
        ("POST", "api/v2/guide/permission_groups.json"),
    ]:
        assert not zt._write_allowed(method, path), (method, path)


def test_write_chokepoint_refuses_non_draft_and_other_paths(monkeypatch, zd_ok, no_network):
    path = "api/v2/help_center/en-us/sections/42/articles.json"
    bad = [
        ("PUT", path, _draft_body()),
        ("POST", path, _draft_body(draft=False)),
        ("POST", path, {"article": {k: v for k, v in _draft_body()["article"].items()
                                    if k != "draft"}, "notify_subscribers": False}),
        ("POST", path, {"article": _draft_body()["article"], "notify_subscribers": True}),
        ("POST", path, {"article": _draft_body()["article"]}),
        ("POST", "api/v2/help_center/articles/1.json", _draft_body()),
    ]
    for method, p, body in bad:
        with pytest.raises(zt.ZendeskWriteRefused):
            _run_async(zt._zendesk_write(method, p, body))


def test_write_chokepoint_sends_allowed_draft(monkeypatch, zd_ok):
    seen = {}

    async def fake(base, token, path, method="GET", json_body=None):
        seen.update(path=path, method=method, body=json_body)
        return {"status": 201, "text": '{"article": {"id": 5}}'}

    monkeypatch.setattr(zt, "_zendesk_request", fake)
    path = "api/v2/help_center/en-us/sections/42/articles.json"
    resp = _run_async(zt._zendesk_write("POST", path, _draft_body()))
    assert resp["status"] == 201
    assert seen["method"] == "POST" and seen["body"]["article"]["draft"] is True


@pytest.mark.parametrize("query", [
    "page[size]=100", "page%5Bsize%5D=100&page%5Bafter%5D=abc", "per_page=30&page=2",
    "query=points&locale=en-us&category=1&section=2&label_names=a,b",
    "sort_by=updated_at&sort_order=desc", "page[before]=x",
])
def test_read_allows_pagination_and_search_keys(query):
    assert zt._normalize_read_path(f"api/v2/help_center/articles.json?{query}", BASE)


@pytest.mark.parametrize("query", [
    "include=users", "include=users,sections", "page[size]=1&include=users",
    "role=admin", "x", "&&", "query=a;include=users",
])
def test_read_refuses_include_and_unknown_query_keys(query, zd_ok, no_network):
    path = f"api/v2/help_center/articles.json?{query}"
    assert zt._normalize_read_path(path, BASE) is None
    assert "error" in _read({"path": path})
    # Same rule for a links.next URL.
    assert zt._normalize_read_path(f"{BASE}/{path}", BASE) is None
