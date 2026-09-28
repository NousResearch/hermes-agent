from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from tools.mcp_skills_protocol import (
    DirectoryReadResult,
    SkillEntry,
    SkillsListResult,
    advertised_skills_settings,
    decode_read_resource_result,
    get_skill,
    list_skills,
    manifest_fingerprint,
    read_directory,
    skills_opted_in,
    validate_skill_entry,
)


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _entry(name: str = "remote-demo", *, root: str = "skill://fixture") -> dict:
    body = f"---\nname: {name}\ndescription: Remote demo\n---\n\n# Demo\n".encode()
    ref = b"support"
    uri = f"{root}/{name}/SKILL.md"
    return {
        "uri": uri,
        "frontmatter": {"name": name, "description": "Remote demo"},
        "resources": [
            {"uri": uri, "digest": _digest(body), "size": len(body)},
            {"uri": f"{root}/{name}/references/info.md", "digest": _digest(ref), "size": len(ref)},
        ],
    }


def test_opt_in_and_capability_are_both_explicit():
    assert skills_opted_in({"skills": {"enabled": True}})
    assert not skills_opted_in({"skills": {"enabled": False}})
    advertised = SimpleNamespace(capabilities=SimpleNamespace(
        extensions={"io.modelcontextprotocol/skills": {"directoryRead": True}}))
    assert advertised_skills_settings(advertised) == {"directoryRead": True}
    assert advertised_skills_settings(SimpleNamespace(capabilities=SimpleNamespace(extensions={}))) is None


def test_manifest_validation_rejects_duplicate_and_traversal_resources():
    duplicate = _entry()
    duplicate["resources"].append(dict(duplicate["resources"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_skill_entry(duplicate)

    traversal = _entry()
    traversal["resources"][1]["uri"] = "skill://fixture/remote-demo/%2e%2e/secret.txt"
    with pytest.raises(ValueError, match="traversal|escapes"):
        validate_skill_entry(traversal)


class _ListOnlySession:
    def __init__(self):
        self.requests = []

    async def send_request(self, request, adapter):
        self.requests.append(request)
        assert request.method == "skills/list"
        return SkillsListResult(skills=[SkillEntry.model_validate(_entry())])

    async def read_resource(self, uri):
        raise AssertionError("catalog discovery must not fetch bodies")


async def _run_list_only():
    session = _ListOnlySession()
    entries, metadata = await list_skills(session, "fixture")
    return session, entries, metadata


def test_list_only_server_catalog_is_sufficient_and_body_lazy():
    session, entries, metadata = asyncio.run(_run_list_only())
    assert [entry.frontmatter["name"] for entry in entries] == ["remote-demo"]
    assert metadata == {"ttl_ms": None, "cache_scope": None}
    assert len(session.requests) == 1


def test_repeated_cursor_fails_closed():
    class Cyclic:
        async def send_request(self, request, adapter):
            return SkillsListResult(skills=[], nextCursor="same")

    with pytest.raises(ValueError, match="repeated cursor"):
        asyncio.run(list_skills(Cyclic(), "fixture"))


def test_resource_decode_requires_exact_uri_and_strict_base64():
    uri = "skill://fixture/remote-demo/SKILL.md"
    text = SimpleNamespace(uri=uri, text="hello", blob=None, mimeType="text/markdown")
    raw, mime, is_text = decode_read_resource_result(SimpleNamespace(contents=[text]), uri)
    assert (raw, mime, is_text) == (b"hello", "text/markdown", True)

    bad = SimpleNamespace(uri=uri, text=None, blob="%%%", mimeType="application/octet-stream")
    with pytest.raises(ValueError, match="strict base64"):
        decode_read_resource_result(SimpleNamespace(contents=[bad]), uri)


def test_directory_result_preserves_extension_fields():
    result = DirectoryReadResult.model_validate({
        "resources": [{"uri": "skill://fixture/remote-demo/references/info.md",
                       "name": "info.md", "mimeType": "text/markdown"}],
        "nextCursor": "opaque", "vendorExtra": 1})
    assert result.next_cursor == "opaque"
    assert result.resources[0].uri.endswith("references/info.md")


def test_optional_get_and_directory_helpers_send_exact_wire_methods():
    class Session:
        def __init__(self):
            self.requests = []

        async def send_request(self, request, adapter):
            self.requests.append(request)
            if request.method == "skills/get":
                from tools.mcp_skills_protocol import SkillsGetResult
                return SkillsGetResult(skill=SkillEntry.model_validate(_entry()))
            return DirectoryReadResult.model_validate({"resources": [{
                "uri": "skill://fixture/remote-demo/references/info.md",
                "name": "info.md", "mimeType": "text/markdown"}]})

    async def run():
        session = Session()
        uri = _entry()["uri"]
        skill = await get_skill(session, uri)
        directory = await read_directory(session, "skill://fixture/remote-demo", "next")
        return session, skill, directory

    session, skill, directory = asyncio.run(run())
    assert [request.method for request in session.requests] == ["skills/get", "resources/directory/read"]
    assert session.requests[0].params == {"uri": _entry()["uri"]}
    assert session.requests[1].params == {"uri": "skill://fixture/remote-demo", "cursor": "next"}
    assert skill.frontmatter["name"] == "remote-demo"
    assert directory.resources[0].uri.endswith("references/info.md")


def test_get_rejects_mismatched_return_uri():
    class Session:
        async def send_request(self, request, adapter):
            from tools.mcp_skills_protocol import SkillsGetResult
            return SkillsGetResult(skill=SkillEntry.model_validate(_entry("other")))

    with pytest.raises(ValueError, match="different skill URI"):
        asyncio.run(get_skill(Session(), _entry()["uri"]))


@pytest.mark.parametrize("uri", ["remote-demo", "skill://fixture/remote-demo", "skill://fixture/other.txt"])
def test_malformed_get_uri_is_rejected_before_request(uri):
    class Session:
        async def send_request(self, request, adapter):
            raise AssertionError("malformed URI reached the network")

    with pytest.raises(ValueError):
        asyncio.run(get_skill(Session(), uri))


class _WireSession:
    def __init__(self, *results):
        self.results = iter(results)
        self.requests = []
        self.parsed = []

    async def send_request(self, request, adapter):
        self.requests.append(request)
        result = adapter.validate_python(next(self.results))
        self.parsed.append(result)
        return result

    async def read_resource(self, uri):
        raise AssertionError("cache hints must not prefetch content")


def _envelope(method, **hints):
    return {("skills" if method == "list" else "skill"):
            ([_entry()] if method == "list" else _entry()), **hints}


def _call(method, session):
    return asyncio.run(list_skills(session, "fixture") if method == "list"
                       else get_skill(session, _entry()["uri"]))


@pytest.mark.parametrize("method", ["list", "get"])
@pytest.mark.parametrize("hints", [
    {}, {"ttlMs": 0, "cacheScope": "private"},
    {"ttlMs": 30, "cacheScope": "public"},
    {"ttlMs": 0.25, "cacheScope": "private"},
    {"ttlMs": 10**400, "cacheScope": "public"},
    {"ttl_ms": 0.25, "cache_scope": "public"},
])
def test_cache_hints_preserve_entry_identity_and_legacy_omission(method, hints):
    session = _WireSession(_envelope(method, vendorExtra={"opaque": True}, **hints))
    returned = _call(method, session)
    entry = returned[0][0] if isinstance(returned, tuple) else returned
    assert entry == validate_skill_entry(_entry())
    assert manifest_fingerprint(entry) == manifest_fingerprint(_entry())
    parsed = session.parsed[0]
    assert parsed.ttl_ms == hints.get("ttlMs", hints.get("ttl_ms"))
    assert parsed.cache_scope == hints.get("cacheScope", hints.get("cache_scope"))
    assert parsed.model_dump(by_alias=True)["vendorExtra"] == {"opaque": True}
    assert len(session.requests) == 1


@pytest.mark.parametrize("method", ["list", "get"])
@pytest.mark.parametrize("hints", [
    {"ttlMs": value} for value in [-1, -0.25, "1", True, False, None,
                                  float("nan"), float("inf"), -float("inf")]
] + [{"cacheScope": value} for value in ["shared", "PUBLIC", 1, True, None, [], {}]])
def test_malformed_advertised_cache_hints_fail_adapter_validation(method, hints):
    with pytest.raises(ValueError):
        _call(method, _WireSession(_envelope(method, **hints)))


@pytest.mark.parametrize("method", ["list", "get"])
def test_explicit_result_type_must_be_complete(method):
    _call(method, _WireSession(_envelope(method, resultType="complete")))
    with pytest.raises(ValueError, match="incomplete"):
        _call(method, _WireSession(_envelope(method, resultType="partial")))


@pytest.mark.parametrize("later_hints", [
    {"ttlMs": -1}, {"cacheScope": "shared"}, {"ttlMs": None},
])
def test_later_page_malformed_hints_reject_whole_listing(later_hints):
    session = _WireSession(
        {"skills": [], "nextCursor": "second", "ttlMs": 1000, "cacheScope": "public"},
        _envelope("list", **later_hints))
    with pytest.raises(ValueError):
        _call("list", session)
    assert session.requests[1].params == {"cursor": "second"}


def test_paginated_hints_are_diagnostics_not_response_reuse_or_prefetch():
    pages = [{"skills": [], "nextCursor": "second", "ttlMs": 1000, "cacheScope": "public"},
             _envelope("list", ttlMs=0, cacheScope="private")]
    session = _WireSession(*pages, *pages)
    first = _call("list", session)
    second = _call("list", session)
    assert first == second
    assert isinstance(first, tuple)
    assert first[1] == {"ttl_ms": 1000, "cache_scope": "public"}
    assert [request.method for request in session.requests] == ["skills/list"] * 4
    assert session.parsed[1].ttl_ms == 0
    assert session.parsed[1].cache_scope == "private"
