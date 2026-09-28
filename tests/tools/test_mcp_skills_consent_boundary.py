"""Cross-origin consent cannot outlive the authority shown in its prompt."""
import asyncio
import copy

import pytest

from tests.tools.test_mcp_skills_auth_context import oauth_skill  # noqa: F401


def test_generic_cross_origin_rechecks_after_approval(oauth_skill, monkeypatch):
    s = oauth_skill
    from tools import mcp_skills_consent as consent, approval_prompt
    from tools import mcp_tool as core
    s.registry.mark_active(s.record, s.home, 'conversation')
    monkeypatch.setitem(core._mcp_tool_server_names, 'mcp__other__read_resource', 'other')
    def approve(*a, **kw):
        asyncio.run(s.switch('account'))
        return 'accept'
    monkeypatch.setattr(approval_prompt, 'request_elicitation_consent', approve)
    result = consent.enforce_remote_skill_gate('mcp__other__read_resource', {'uri': 'resource://other/doc'}, session_id='conversation')
    assert result is not None, 'cross-origin read proceeded after active origin authorization changed during approval'


@pytest.mark.parametrize("route", ["generic", "native"])
@pytest.mark.parametrize("answer", ["accept", "decline"])
@pytest.mark.parametrize("change", ["none", "active_auth", "active_source", "target_auth", "same_origin"])
def test_cross_origin_consent_revalidates_its_authority(oauth_skill, monkeypatch, route, answer, change):
    s = oauth_skill
    from tools import approval_prompt, mcp_skills_consent as consent, mcp_tool as core

    s.registry.mark_active(s.record, s.home, "conversation")
    target = copy.deepcopy(s.record)
    if change != "same_origin":
        target["uri"] = "skill://fixture/other/SKILL.md"
    name = "mcp__fixture__read_resource"
    monkeypatch.setitem(core._mcp_tool_server_names, name, "fixture.team")
    prompts = []

    def approve(*args, **kwargs):
        prompts.append(kwargs["surface"])
        if change == "active_auth":
            asyncio.run(s.switch("account"))
        elif change == "active_source":
            s.server._skills_connected = False
        elif change == "target_auth":
            # Only the native gate owns a target manifest; generic reads own
            # active origins but leave target-resource validation to dispatch.
            target["authorization_context"] = "invalidated-during-approval"
        return answer

    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", approve)

    def gate():
        if route == "native":
            return consent.enforce_native_skill_read_gate(
                target, target["resources"][0], session_id="conversation")
        uri = s.uri if change == "same_origin" else "resource://other/doc"
        return consent.enforce_remote_skill_gate(name, {"uri": uri}, session_id="conversation")

    result = gate()
    stale = change in {"active_auth", "active_source"} or (route == "native" and change == "target_auth")
    if stale:
        assert result is not None
        assert "authorization context or source is unavailable" in result
        assert s.uri not in result and target["uri"] not in result
    elif change == "same_origin" or answer == "accept":
        assert result is None
    else:
        assert result is not None
        assert "not explicitly approved for this call" in result
    assert prompts == ([] if change == "same_origin" else ["mcp-skill-cross-origin-read"])
    if not stale and change != "same_origin":
        assert gate() == result
        assert len(prompts) == 2, "cross-origin consent must remain per-call"
    assert not s.calls, "consent checks must not fetch remote bytes"
