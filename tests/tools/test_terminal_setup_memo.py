"""Per-call terminal setup is memoised and spill pruning is throttled."""
from tools.environments import base_output, local


def test_path_memo_reuses_value_within_ttl(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(local, "_path_memo", {})
    calls = []
    monkeypatch.setattr(local, "_user_tools_first_uncached", lambda p: calls.append(p) or p + ":x")
    assert local._user_tools_first("/a") == "/a:x"
    assert local._user_tools_first("/a") == "/a:x"
    assert local._user_tools_first("/b") == "/b:x"
    assert calls == ["/a", "/b"]


def test_path_memo_is_keyed_by_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setattr(local, "_path_memo", {})
    calls = []
    monkeypatch.setattr(local, "_managed_runtime_path_entries_uncached", lambda: calls.append(1) or ["/m"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "one"))
    local._managed_runtime_path_entries()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "two"))
    local._managed_runtime_path_entries()
    assert len(calls) == 2


def test_path_memo_expires(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(local, "_path_memo", {})
    monkeypatch.setattr(local, "_PATH_MEMO_TTL_S", 0.0)
    calls = []
    monkeypatch.setattr(local, "_user_tools_first_uncached", lambda p: calls.append(p) or p)
    local._user_tools_first("/a")
    local._user_tools_first("/a")
    assert len(calls) == 2


def test_spill_prune_runs_once_per_interval(monkeypatch, tmp_path):
    monkeypatch.setattr(base_output, "_spill_pruned_at", {})
    assert base_output._claim_spill_prune(tmp_path) is True
    assert base_output._claim_spill_prune(tmp_path) is False
    monkeypatch.setattr(base_output, "_SPILL_PRUNE_INTERVAL_S", 0.0)
    assert base_output._claim_spill_prune(tmp_path) is True
