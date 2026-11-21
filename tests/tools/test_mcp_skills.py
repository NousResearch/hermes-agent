"""Skills extension contracts through real registry handlers against a remote-session boundary."""

import asyncio
import base64
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import mcp_tool as core
from tools import mcp_tool_skill_loading as loading
from tools.mcp_tool_registration import _register_server_tools, _select_utility_schemas
from tools.mcp_tool_skills import EXTENSION, SkillVerificationError, parse_entry, read_verified
from tools.registry import registry

URI = "skill://review/SKILL.md"
TEXT = "---\nname: review\ndescription: Review changes.\nallowed-tools: terminal\n---\n\nRead references/check.md.\n"
REFERENCE = "skill://review/references/check.md"


def make_entry(files=None, uri=URI, **frontmatter):
    files = files or {uri: TEXT, REFERENCE: "Check correctness.\n"}
    return {"uri": uri, "frontmatter": {"name": "review", "description": "Review changes.",
                                         "allowed-tools": "terminal", **frontmatter},
            "resources": [{"uri": path, "digest": "sha256:" + hashlib.sha256(
                data.encode() if isinstance(data, str) else data).hexdigest(),
                "size": len(data.encode() if isinstance(data, str) else data)} for path, data in files.items()]}


class RemoteSession:
    def __init__(self):
        self.files = {URI: TEXT, REFERENCE: "Check correctness.\n"}
        self.entries = [make_entry()]
        self.calls = []
        self.pages = None

    async def send_request(self, request, result_type):
        self.calls.append((request.method, request.params))
        result = {"resultType": "complete", "ttlMs": 0, "cacheScope": "private"}
        if request.method == "skills/get":
            return {**result, "skill": next(e for e in self.entries if e["uri"] == request.params["uri"])}
        if self.pages is not None:
            return {**result, **self.pages[len([c for c in self.calls if c[0] == "skills/list"]) - 1]}
        return {**result, "skills": copy.deepcopy(self.entries)}

    async def read_resource(self, uri):
        self.calls.append(("resources/read", uri))
        data = self.files[uri]
        content = ({"text": data} if isinstance(data, str) else {"blob": base64.b64encode(data).decode()})
        return SimpleNamespace(contents=[SimpleNamespace(uri=uri, **content)])


@pytest.fixture
def connected(tmp_path, monkeypatch):
    from tools import approval_prompt, mcp_tool_loop
    from tools.approval_context import _approval_session_key
    token = _approval_session_key.set("skills-conversation")
    loading.clear_session("skills-conversation")
    server = core.MCPServerTask("docs")
    server.session = RemoteSession()
    server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(
        resources=SimpleNamespace(), prompts=None, extensions={EXTENSION: {}}))
    monkeypatch.setattr(core, "_servers", {"docs": server})
    monkeypatch.setattr(core, "_server_scope_keys", {})
    monkeypatch.setattr(core, "_server_tool_scopes", {})
    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda factory, **kw: asyncio.run(factory()))
    consent = Mock(return_value="accept")
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", consent)
    names = _register_server_tools("docs", server, {})
    try:
        yield server, consent
    finally:
        for name in names:
            registry.deregister(name)
        loading.clear_session("skills-conversation")
        _approval_session_key.reset(token)


def call(tool="skill_view", **args):
    return json.loads(registry.dispatch(f"mcp__docs__{tool}", args))


def test_discovery_is_lazy_paginated_and_preserves_colliding_names(connected):
    server, consent = connected
    second = make_entry({"github://team/skills/review/SKILL.md": TEXT}, uri="github://team/skills/review/SKILL.md")
    server.session.pages = [{"skills": [make_entry()], "nextCursor": "next"}, {"skills": [second]}]
    assert server.session.calls == []  # registration retrieves no skill content
    result = call("list_skills")
    assert {s["uri"] for s in result["skills"]} == {URI, second["uri"]}
    assert all(s["server"] == "docs" for s in result["skills"])
    assert server.session.calls == [("skills/list", {}), ("skills/list", {"cursor": "next"})]
    consent.assert_not_called()


def test_direct_load_and_supporting_reads_are_bound_to_manifest_and_origin(connected):
    server, consent = connected
    loaded = call(uri=URI)
    assert loaded["success"] and loaded["origin"] == "mcp" and loaded["server"] == "docs"
    assert loaded["content"] == TEXT
    assert [c[0] for c in server.session.calls] == ["skills/get", "resources/read"]
    assert "local code execution" in consent.call_args.args[0]
    reference = call(uri=URI, revision=loaded["revision"], file_path="references/check.md")
    assert reference["content"] == "Check correctness.\n"
    assert consent.call_count == 1
    reads = len(server.session.calls)
    directory = call(uri=URI, revision=loaded["revision"], file_path="references/")
    assert directory["resources"] == [{"uri": REFERENCE, "name": "check.md", "is_directory": False}]
    assert len(server.session.calls) == reads


def test_consent_denial_and_missing_revision_never_retrieve_files(connected):
    server, consent = connected
    consent.return_value = "decline"
    assert "not approve" in call(uri=URI)["error"]
    assert [c[0] for c in server.session.calls] == ["skills/get"]
    assert "revision" in call(uri=URI, file_path="references/check.md")["error"]
    assert len(server.session.calls) == 1


def test_digest_failure_rejects_content_and_refresh_requires_new_consent(connected):
    server, consent = connected
    loaded = call(uri=URI)
    server.session.files[REFERENCE] = "Changed content.\n"
    rejected = call(uri=URI, revision=loaded["revision"], file_path="references/check.md")
    assert "error" in rejected and "content" not in rejected
    server.session.entries = [make_entry(server.session.files)]
    refreshed = call(uri=URI)
    assert refreshed["revision"] != loaded["revision"]
    assert consent.call_count == 2
    assert call(uri=URI, revision=refreshed["revision"], file_path="references/check.md")["content"] == "Changed content.\n"


def test_changed_listing_revokes_old_approval_without_replacing_held_entry(connected):
    server, consent = connected
    loaded = call(uri=URI)
    server.session.files[REFERENCE] = "New checklist.\n"
    server.session.entries = [make_entry(server.session.files)]
    call("list_skills")
    assert "changed" in call(uri=URI, revision=loaded["revision"], file_path="references/check.md")["error"]
    consent.return_value = "decline"
    assert "error" in call(uri=URI)
    assert consent.call_count == 2


def test_nested_skill_is_supporting_data_and_needs_its_own_activation_consent(connected):
    server, consent = connected
    nested_uri = "skill://review/nested/SKILL.md"
    nested_text = "---\nname: nested\ndescription: Nested review.\n---\n\nReview subprojects.\n"
    server.session.files[nested_uri] = nested_text
    server.session.entries = [make_entry(server.session.files), make_entry(
        {nested_uri: nested_text}, uri=nested_uri, name="nested", description="Nested review.")]
    server.session.entries[1]["frontmatter"].pop("allowed-tools")
    loaded = call(uri=URI)
    assert call(uri=URI, revision=loaded["revision"], file_path="nested/SKILL.md")["content"] == nested_text
    assert consent.call_count == 1
    assert call(uri=nested_uri)["success"]
    assert consent.call_count == 2


@pytest.mark.parametrize("file_path", ["../secret", "%2e%2e/secret", "/etc/passwd", "C:/secret", "a/../../x",
                                      "references/new.md", "https://other/secret", "references\\check.md"])
def test_unlisted_or_escaping_file_never_reaches_server(connected, file_path):
    server, _ = connected
    loaded = call(uri=URI)
    calls = len(server.session.calls)
    assert "error" in call(uri=URI, revision=loaded["revision"], file_path=file_path)
    assert len(server.session.calls) == calls


def test_frontmatter_is_compared_field_by_field_before_loading(connected):
    server, _ = connected
    server.session.entries[0]["frontmatter"]["allowed-tools"] = "execute_code"
    result = call(uri=URI)
    assert "error" in result and "content" not in result
    assert loading._origins() == set()


def test_generic_resource_reader_does_not_bypass_manifest_or_cross_server_consent(connected):
    server, consent = connected
    loaded = call(uri=URI)
    calls = len(server.session.calls)
    assert "skill_view" in call("read_resource", uri=REFERENCE)["error"]
    assert len(server.session.calls) == calls
    consent.return_value = "decline"
    assert loading.resource_read_gate("other", "resource://private")
    assert "docs" in consent.call_args.args[0] and "other" in consent.call_args.args[0]
    assert loaded["revision"]


def test_approvals_and_retained_entries_are_scoped_to_profile_session_and_route(connected, monkeypatch, tmp_path):
    from tools.approval_context import _approval_session_key
    server, _ = connected
    loaded = call(uri=URI)
    args = {"uri": URI, "revision": loaded["revision"], "file_path": "references/check.md"}
    with monkeypatch.context() as scoped:
        scoped.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
        assert "error" in call(**args)
    token = _approval_session_key.set("other-chat")
    try:
        assert "error" in call(**args)
    finally:
        _approval_session_key.reset(token)
    server._config = {"url": "https://different.example/mcp"}
    assert "error" in call(**args)


def test_same_uri_on_different_servers_requires_separate_approval(connected, monkeypatch):
    server, consent = connected
    loaded = call(uri=URI)
    other = core.MCPServerTask("other")
    other.session = RemoteSession()
    other.initialize_result = server.initialize_result
    monkeypatch.setitem(core._servers, "other", other)
    names = _register_server_tools("other", other, {})
    try:
        rejected = json.loads(registry.dispatch("mcp__other__skill_view", {
            "uri": URI, "revision": loaded["revision"], "file_path": "references/check.md"}))
        assert "error" in rejected and other.session.calls == []
        other_loaded = json.loads(registry.dispatch("mcp__other__skill_view", {"uri": URI}))
        assert other_loaded["server"] == "other" and consent.call_count == 2
        consent.return_value = "decline"
        assert "error" in call(uri=URI, revision=loaded["revision"], file_path="references/check.md")
        assert "other" in consent.call_args.args[0] and "docs" in consent.call_args.args[0]
    finally:
        for name in names:
            registry.deregister(name)


def test_profile_cleanup_does_not_revoke_another_profiles_held_skills(connected, monkeypatch, tmp_path):
    call(uri=URI)
    with monkeypatch.context() as scoped:
        scoped.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
        loading.clear_session("skills-conversation")
    assert loading._origins() == {"docs"}


@pytest.mark.parametrize("resources,extensions,config,expected", [
    (None, {EXTENSION: {}}, {}, False), ({}, {}, {}, False), ({}, {EXTENSION: {}}, {}, True),
    ({}, {EXTENSION: {}}, {"tools": {"skills": False}}, False),
    ({}, {EXTENSION: {}}, {"tools": {"resources": False}}, False),
    ({}, {EXTENSION: False}, {}, False),
])
def test_only_advertised_enabled_skills_get_utilities(resources, extensions, config, expected):
    server = SimpleNamespace(initialize_result=SimpleNamespace(capabilities=SimpleNamespace(
        resources=resources, extensions=extensions, prompts=None)))
    keys = {s["handler_key"] for s in _select_utility_schemas("docs", server, config)}
    assert ({"list_skills", "skill_view"} <= keys) == expected


@pytest.mark.parametrize("family", ["skills", "resources", "prompts"])
def test_disabling_utility_family_invalidates_lazy_schema_cache(connected, family):
    from tools.mcp_schema_cache import config_fingerprint, get_cached_entry
    assert get_cached_entry("docs", config_fingerprint({})) is not None
    assert get_cached_entry("docs", config_fingerprint({"tools": {family: False}})) is None


def test_supports_maximum_manifest_and_rejects_larger_or_invalid_manifests():
    raw = make_entry()
    resource = raw["resources"][-1]
    raw["resources"] += [{**resource, "uri": f"skill://review/files/{i}"} for i in range(510)]
    raw["resources"][0]["size"] = 16 * 1024 * 1024 - sum(r["size"] for r in raw["resources"][1:])
    assert len(parse_entry(raw).value["resources"]) == 512
    raw["resources"].append({**resource, "uri": "skill://review/files/extra"})
    with pytest.raises(SkillVerificationError):
        parse_entry(raw)


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(resources="dynamic"), lambda e: e.pop("resources"),
    lambda e: e["resources"][0].update(digest="sha256:wrong"),
    lambda e: e["resources"][0].update(size=True),
    lambda e: e["resources"].append(e["resources"][0]),
    lambda e: e["resources"][1].update(uri="skill://other/secret"),
    lambda e: e["resources"][1].update(uri="skill://review/%2e%2e/secret"),
    lambda e: e["frontmatter"].update(name="wrong"),
    lambda e: e["resources"][0].update(size=16 * 1024 * 1024),
])
def test_invalid_entries_fail_closed(mutation):
    entry = make_entry()
    mutation(entry)
    with pytest.raises(SkillVerificationError):
        parse_entry(entry)


def test_binary_and_unicode_bytes_are_verified_without_transformation():
    session = RemoteSession()
    session.files[REFERENCE] = b"\x00\xff\x80"
    entry = parse_entry(make_entry(session.files))
    result = asyncio.run(read_verified(session, entry, REFERENCE))
    assert base64.b64decode(result["blob"]) == b"\x00\xff\x80"
    session.files[REFERENCE] = "\u00e9\r\n"
    result = asyncio.run(read_verified(session, parse_entry(make_entry(session.files)), REFERENCE))
    assert result["content"] == "\u00e9\r\n"


def test_session_cleanup_releases_entries(connected):
    from tools.approval import clear_session
    call(uri=URI)
    assert loading._origins() == {"docs"}
    clear_session("skills-conversation")
    assert loading._origins() == set()
