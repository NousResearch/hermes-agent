"""Behaviour contracts for loading remote skills through ``skill_view`` / ``skills_list``.

Routing is where a new skill source can quietly break an old one, so the ordering
rules are asserted directly: a local skill keeps its name, a bare name that two
servers both offer is refused rather than guessed, and nothing reaches the network
unless a skill that exists somewhere was actually asked for.
"""

import json

import pytest

from tests.fakes.fake_mcp_skills import FakeSkill, FakeSkillsServer, FakeSkillsSender
from tools import skills_tool, skills_tool_mcp
from tools.mcp_skills_manager import SkillsOverMcpManager, get_manager, set_manager

SERVER = "acme"
SERVER_B = "other"
SCOPE = ""

# One conversation id per test: the repeat-view dedup is keyed by (task, server, uri,
# revision) process-wide, and two tests loading identical fake content under one id
# would see the second call deduped against the first.
_TASK = ["task-2640"]


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A live manager backed by a fake server, with the profile's skill list stubbed empty."""
    server = FakeSkillsServer([
        FakeSkill(uri="skill://team/deploy/SKILL.md", name="deploy", description="ship it",
                  files={"SKILL.md": "---\nname: deploy\n---\nrun the deploy\n",
                         "references/steps.md": "step one\n"}),
        FakeSkill(uri="skill://team/rollback/SKILL.md", name="rollback", description="undo it",
                  files={"SKILL.md": "---\nname: rollback\n---\nundo\n"}),
    ])
    manager = SkillsOverMcpManager(sender=FakeSkillsSender(server))
    manager.observe_connection(SERVER, SCOPE, initialize_result=server.initialize_result(), session=object())
    import asyncio
    asyncio.run(manager.discover(SERVER, SCOPE))

    _TASK[0] = f"task-{tmp_path.name.replace('-', '')[:12]}"
    set_manager(manager)
    monkeypatch.setattr(skills_tool_mcp, "known_server_labels", lambda: {SERVER, SERVER_B})
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda *a, **k: set())
    try:
        yield server, manager
    finally:
        set_manager(None)


def _home():
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _view(**kwargs) -> dict:
    kwargs.setdefault("task_id", _TASK[0])
    return json.loads(skills_tool.skill_view(**kwargs))


# ── locators ────────────────────────────────────────────────────────────────

def test_a_qualified_name_loads_the_remote_skill(remote):
    result = _view(name=f"{SERVER}::team/deploy")
    assert result["success"] is True
    assert result["name"] == f"{SERVER}::team/deploy"
    assert result["content"].endswith("run the deploy\n")
    assert result["verified"] is True
    assert result["source"] == {"backend": "mcp", "server": SERVER,
                               "uri": "skill://team/deploy/SKILL.md", "remote": True}


def test_server_and_uri_load_the_same_skill(remote):
    result = _view(server=SERVER, uri="skill://team/deploy/SKILL.md")
    assert result["success"] is True and result["verified"] is True


def test_a_remote_result_never_pretends_to_have_a_local_directory(remote):
    result = _view(server=SERVER, uri="skill://team/deploy/SKILL.md")
    assert result["skill_dir"] is None
    assert result["skill_root_uri"] == "skill://team/deploy"
    assert result["identity"]["server"] == SERVER


def test_a_uri_without_a_server_refuses_to_guess_the_origin(remote):
    result = _view(uri="skill://team/deploy/SKILL.md")
    assert result["success"] is False
    assert result["error_code"] == "ambiguous_identity"


def test_a_server_without_a_uri_says_what_is_missing(remote):
    result = _view(server=SERVER)
    assert result["success"] is False
    assert result["error_code"] == "not_provided"
    assert "uri=" in result["error"]


def test_an_empty_call_is_told_how_to_locate_something():
    result = json.loads(skills_tool.skill_view(""))
    assert result["success"] is False
    assert "skill name" in result["error"]


def test_a_bare_name_found_on_exactly_one_remote_server_resolves(remote):
    result = _view(name="rollback")
    assert result["success"] is True
    assert result["name"] == f"{SERVER}::team/rollback"


def test_a_bare_name_two_servers_both_offer_is_refused_with_candidates(remote):
    server_b = FakeSkillsServer([FakeSkill(uri="skill://team/deploy/SKILL.md", name="deploy",
                                           files={"SKILL.md": "---\nname: deploy\n---\nother\n"})])
    manager = get_manager()
    manager.observe_connection(SERVER_B, SCOPE, initialize_result=server_b.initialize_result(),
                               session=object())
    import asyncio
    asyncio.run(manager.discover(SERVER_B, SCOPE))

    result = _view(name="deploy")
    assert result["success"] is False
    assert result["error_code"] == "ambiguous_identity"
    assert {m["server"] for m in result["matches"]} == {SERVER, SERVER_B}


def test_a_name_no_server_offers_keeps_the_local_not_found_answer(remote):
    result = _view(name="no-such-skill-anywhere")
    assert result["success"] is False
    assert "not found" in result["error"]
    assert result.get("error_code") is None, "this is the local error, not a remote one"


def test_an_on_disk_skill_keeps_its_name_even_when_a_server_offers_the_same_one(remote):
    """Adding a server must never hijack a call that already resolved."""
    skill_dir = _home() / "skills" / "deploy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: deploy\ndescription: local one\n---\nlocal body\n",
                                        encoding="utf-8")
    skills_tool.clear_skills_cache()

    result = _view(name="deploy")
    assert result["success"] is True
    assert result["content"].endswith("local body\n")
    assert result.get("source") is None


def test_a_locally_colliding_name_is_not_papered_over_by_a_remote_match(remote):
    """An ambiguous local name is a refusal, not a miss — the remote catalog must not fill in."""
    for tier in ("a", "b"):
        target = _home() / "skills" / "tier" / tier / "dup"
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text(f"---\nname: dup\n---\n{tier}\n", encoding="utf-8")
    skills_tool.clear_skills_cache()

    result = _view(name="dup")
    assert result["success"] is False
    assert "match" in result or "name" in result["error"].lower()


# ── enablement ──────────────────────────────────────────────────────────────

def test_an_opted_in_server_loads_without_a_separate_approval(remote):
    """The whole path: enabling the server is the consent, and a load just works."""
    _server, manager = remote
    result = _view(name=f"{SERVER}::team/deploy")
    assert result["success"] is True
    assert result["verified"] is True
    assert manager.instance_view(SERVER, SCOPE).support.value == "supported"


def test_no_approval_record_is_consulted_or_written(remote, tmp_path):
    """Nothing in the load path persists a per-skill decision any more."""
    _server, _manager = remote
    _view(name=f"{SERVER}::team/deploy")
    assert not list(tmp_path.glob("*approval*")), "the load path must not create approval state"


def test_a_disabled_skill_is_refused_before_any_read(remote, monkeypatch):
    server, _manager = remote
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda *a, **k: {f"{SERVER}::team/deploy"})
    before = server.calls.count("resources/read")
    result = _view(name=f"{SERVER}::team/deploy")
    assert result["success"] is False
    assert "disabled" in result["error"]
    assert server.calls.count("resources/read") == before, "a switched-off skill must not fetch a body"


def test_a_disabled_skill_is_refused_through_a_structured_locator_too(remote, monkeypatch):
    """A direct URI must not be a way around the enable/disable list."""
    _server, _manager = remote
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda *a, **k: {f"{SERVER}::team/deploy"})
    result = _view(server=SERVER, uri="skill://team/deploy/SKILL.md")
    assert result["success"] is False
    assert "disabled" in result["error"]


def test_a_disabled_skill_is_absent_from_the_listing_and_the_index(remote, monkeypatch):
    _server, _manager = remote
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda *a, **k: {f"{SERVER}::team/deploy"})
    listing = json.loads(skills_tool.skills_list())
    names = {s["name"] for s in listing["skills"] if s.get("source", {}).get("remote")}
    assert names == {f"{SERVER}::team/rollback"}

    from agent.skills_prompt_mcp import mcp_skill_rows
    rows, _token = mcp_skill_rows({f"{SERVER}::team/deploy"}, None, None, None)
    assert {r[1] for r in rows} == {f"{SERVER}::team/rollback"}


def test_the_bare_name_does_not_switch_off_a_remote_skill(remote, monkeypatch):
    """A `skills.disabled` entry written for a local skill must not reach a remote one."""
    _server, _manager = remote
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda *a, **k: {"deploy"})
    assert _view(name=f"{SERVER}::team/deploy")["success"] is True


# ── attachments ─────────────────────────────────────────────────────────────

def test_a_support_file_inside_the_manifest_is_served_and_verified(remote):
    result = _view(server=SERVER, uri="skill://team/deploy/SKILL.md", file_path="references/steps.md")
    assert result["success"] is True
    assert result["content"] == "step one\n"
    assert result["verified"] is True


def test_a_file_outside_the_manifest_is_refused_with_what_is_available(remote):
    result = _view(server=SERVER, uri="skill://team/deploy/SKILL.md", file_path="references/secret.md")
    assert result["success"] is False
    assert result["available_files"] == ["SKILL.md", "references/steps.md"]


def test_a_nested_skill_cannot_be_reached_through_its_parents_file_path(remote):
    """A nested SKILL.md is its own skill: it has its own identity and is loaded as one."""
    result = _view(server=SERVER, uri="skill://team/deploy/SKILL.md", file_path="sub/SKILL.md")
    assert result["success"] is False
    assert "nested skill" in result["error"]
    assert "skill_view(server=" in result["hint"]


def test_attachment_bytes_that_no_longer_match_retire_the_catalog(remote):
    """A server that keeps declaring the approved digest but serves other bytes must fail.

    The manifest is deliberately held at the *old* digest while the bytes change, so the
    mismatch is the server's inconsistency rather than a legitimate republication.
    """
    server, manager = remote
    declared = manager.entries(SCOPE)[0].manifest.find("skill://team/deploy/references/steps.md")
    server.skills[0].files["references/steps.md"] = "different bytes entirely\n"
    server.digest_overrides[server.skills[0].uri] = {"references/steps.md": declared.digest}

    result = _view(server=SERVER, uri="skill://team/deploy/SKILL.md", file_path="references/steps.md")
    assert result["success"] is False
    assert result["error_code"] == "verification_failed"
    assert manager.snapshot(SERVER, SCOPE) is None


# ── repeats and platform ────────────────────────────────────────────────────

def test_a_repeat_view_of_the_same_revision_is_deduped(remote):
    first = _view(name=f"{SERVER}::team/deploy")
    assert first["success"] is True
    again = _view(name=f"{SERVER}::team/deploy")
    assert again.get("dedup") is True and again.get("content_returned") is False


def test_a_platform_mismatched_remote_skill_is_refused_on_load(remote):
    server, manager = remote
    server.skills.append(FakeSkill(uri="skill://team/linuxonly/SKILL.md", name="linuxonly",
                                   files={"SKILL.md": "---\nname: linuxonly\n---\nx\n"},
                                   frontmatter_extra={"platforms": ["linux"]}))
    import asyncio
    asyncio.run(manager.discover(SERVER, SCOPE, reason="test"))

    result = _view(name=f"{SERVER}::team/linuxonly")
    assert result["success"] is False
    assert result["readiness_status"] == "unsupported"


# ── listing, projection, mutation ───────────────────────────────────────────

def test_skills_list_carries_the_identity_of_every_remote_skill(remote):
    listing = json.loads(skills_tool.skills_list())
    rows = [s for s in listing["skills"] if s.get("source", {}).get("remote")]
    assert {r["name"] for r in rows} == {f"{SERVER}::team/deploy", f"{SERVER}::team/rollback"}
    assert all(r["server"] == SERVER and r["uri"].endswith("SKILL.md") for r in rows)


def test_the_index_projects_remote_skills_under_a_qualified_key(remote):
    from agent.skills_prompt_mcp import mcp_skill_rows
    rows, token = mcp_skill_rows(set(), {"skill_view"}, {"skills"}, None)
    assert {(r[1]) for r in rows} == {f"{SERVER}::team/deploy", f"{SERVER}::team/rollback"}
    assert all(r[0] == f"mcp/{SERVER}" for r in rows)
    assert token == ((SERVER, 1),), "the projection re-renders when the revision moves"


def test_the_index_hides_platform_mismatched_remote_skills(remote):
    server, manager = remote
    server.skills.append(FakeSkill(uri="skill://team/linuxonly/SKILL.md", name="linuxonly",
                                   files={"SKILL.md": "---\nname: linuxonly\n---\nx\n"},
                                   frontmatter_extra={"platforms": ["linux"]}))
    import asyncio
    asyncio.run(manager.discover(SERVER, SCOPE, reason="test"))
    from agent.skills_prompt_mcp import mcp_skill_rows
    rows, _token = mcp_skill_rows(set(), None, None, None)
    assert f"{SERVER}::team/linuxonly" not in {r[1] for r in rows}, "the offer gate withholds it"


def test_a_disabled_remote_skill_is_not_projected(remote):
    from agent.skills_prompt_mcp import mcp_skill_rows
    rows, _token = mcp_skill_rows({f"{SERVER}::team/deploy"}, None, None, None)
    assert {r[1] for r in rows} == {f"{SERVER}::team/rollback"}


def test_skill_manage_refuses_to_mutate_a_remote_skill(remote):
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage(action="patch", name=f"{SERVER}::team/deploy",
                                     old_string="undo", new_string="redo"))
    assert result["success"] is False
    assert result["error_code"] == "mutation_unsupported"


def test_skill_manage_still_accepts_a_local_name_that_merely_looks_qualified(remote):
    """The guard is about remote ownership, not about the presence of a colon."""
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage(action="patch", name="notaserver:thing",
                                     old_string="a", new_string="b"))
    assert result.get("error_code") != "mutation_unsupported"
