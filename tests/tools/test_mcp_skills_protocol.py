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
