"""An env=None spill must be announced by the path the sandbox will see (#121024)."""

import tools.tool_result_storage as trs


def _spill(monkeypatch, translate_to, env=None):
    monkeypatch.setattr(trs, "_write_to_spillover", lambda content, filename: "/home/u/.hermes/cache/spillover/call_1.txt")
    monkeypatch.setattr(trs, "_is_host_side_env", lambda e: True)
    import tools.credential_files as cf

    monkeypatch.setattr(cf, "to_agent_visible_cache_path", lambda p, container_base="/root/.hermes": translate_to)
    return trs.maybe_persist_tool_result("x" * 5000, "get", "id1", env=env, threshold=100)


def test_remote_backend_env_none_announces_sandbox_path(monkeypatch):
    msg = _spill(monkeypatch, "/root/.hermes/cache/spillover/call_1.txt")
    assert "/root/.hermes/cache/spillover/call_1.txt" in msg
    assert "/home/u/.hermes/cache/spillover/call_1.txt" not in msg


def test_local_backend_env_none_keeps_host_path(monkeypatch):
    msg = _spill(monkeypatch, "/home/u/.hermes/cache/spillover/call_1.txt")
    assert "/home/u/.hermes/cache/spillover/call_1.txt" in msg
