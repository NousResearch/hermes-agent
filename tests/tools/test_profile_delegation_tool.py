import json
import yaml


def test_delegate_to_profile_tool_accepts_auto_executor_and_structured_policy(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    p = home / "profiles" / "cto"
    (p / "mcp-tokens").mkdir(parents=True)
    p.joinpath("config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"vercel": {"enabled": True, "auth": "oauth"}}}), encoding="utf-8")
    p.joinpath("mcp-tokens", "vercel.json").write_text("secret", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "cmo")

    from hermes_cli.profile_delegation import ProfileDelegationResult
    import tools.profile_delegation_tool as tool

    def fake_delegate(req):
        assert req.profile is None
        assert req.requester_profile == "cmo"
        assert req.tool_action.tool_name == "mcp:vercel"
        return ProfileDelegationResult(
            status="queued", delegation_id="pd_test", task_id="t_test",
            executor_profile="cto", requester_profile="cmo", capability="mcp:vercel",
            risk="READ", ranking=[])

    monkeypatch.setattr(tool, "delegate_to_profile", fake_delegate)
    data = json.loads(tool.delegate_to_profile_tool(
        task="Inspect Vercel", required_capability="mcp:vercel", tool_name="mcp:vercel", action_name="list_projects"
    ))
    assert data["executor_profile"] == "cto"
    assert data["status"] == "queued"
