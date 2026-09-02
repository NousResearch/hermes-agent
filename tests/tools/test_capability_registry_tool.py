import json
import yaml


def test_find_capability_tool_returns_workload_ranking(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    p = home / "profiles" / "cto"
    (p / "mcp-tokens").mkdir(parents=True)
    p.joinpath("config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"vercel": {"enabled": True, "auth": "oauth"}}}), encoding="utf-8")
    p.joinpath("mcp-tokens", "vercel.json").write_text("secret", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))

    from tools.capability_registry_tool import find_capability_tool

    data = json.loads(find_capability_tool("mcp:vercel", requester_profile="cmo"))
    assert data["recommendation"]["best_profile"] == "cto"
    assert data["profiles"][0]["credential_present"] is True
    assert "secret" not in json.dumps(data)
