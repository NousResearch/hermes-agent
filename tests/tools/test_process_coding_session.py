"""Slice-1 contracts for the typed coding-agent descriptor (#103194).

Covers: constructor defaults (ordinary processes unchanged), normalize
hygiene (unknown backends degrade, never raises), launch plumbing through
``_new_session`` (the single writer both spawn paths share), checkpoint
round-trip + backward compat with pre-field checkpoints, redaction (no env
or token material ever lands in the descriptor), the effective-model
readback stub, and the terminal dispatch foreground guard.
"""

import json

import pytest

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    _CHECKPOINT_DEFAULTS,
    _CHECKPOINT_FIELDS,
    coding_session_effective_model,
    normalize_coding_descriptor,
)


def test_descriptor_defaults_to_untyped():
    """Ordinary processes behave exactly as before: all coding_* empty."""
    s = ProcessSession(id="proc_x", command="echo hi")
    assert s.coding_backend == ""
    assert s.coding_backend_version == ""
    assert s.coding_session_id == ""
    assert s.coding_model == ""
    assert s.coding_capabilities == []
    assert s.coding_resume_command == ""
    assert coding_session_effective_model(s) == ""


def test_normalize_coding_descriptor_passthrough():
    out = normalize_coding_descriptor(
        backend="Opencode", version="v1.2", session_id="ses_abc",
        model="gpt-5", capabilities=["hot_switch", " structured_api ", ""],
        resume_command="opencode attach ses_abc")
    assert out == {
        "coding_backend": "opencode",
        "coding_backend_version": "v1.2",
        "coding_session_id": "ses_abc",
        "coding_model": "gpt-5",
        "coding_capabilities": ["hot_switch", "structured_api"],
        "coding_resume_command": "opencode attach ses_abc",
    }


def test_normalize_coding_descriptor_degrades_safely():
    """Unknown backends become untyped; garbage never raises."""
    assert normalize_coding_descriptor(backend="tmux")["coding_backend"] == ""
    degraded = normalize_coding_descriptor(backend="tmux", model="x")
    assert degraded["coding_model"] == ""
    assert normalize_coding_descriptor(backend=None)["coding_backend"] == ""
    assert normalize_coding_descriptor(
        backend="codex", capabilities="hot_switch")["coding_capabilities"] == []
    # Never raises, whatever rides in.
    assert normalize_coding_descriptor(
        backend=object(), version=object(), session_id=object(), model=object(),
        capabilities=object(), resume_command=object())["coding_backend"] == ""


def test_new_session_plumbs_descriptor():
    """The single writer both spawn paths share normalizes once, here."""
    s = ProcessRegistry._new_session(
        "codex --ask", "t1", "sa-1", "sk", "/tmp",
        coding_backend="codex", coding_session_id="thr_1", coding_model="gpt-5.2",
        coding_capabilities=["hot_switch"], coding_resume_command="codex resume thr_1")
    assert (s.coding_backend, s.coding_session_id, s.coding_model) == ("codex", "thr_1", "gpt-5.2")
    assert s.coding_capabilities == ["hot_switch"]
    assert s.coding_resume_command == "codex resume thr_1"
    assert s.owner_task_id == "sa-1"  # ownership untouched
    assert coding_session_effective_model(s) == "gpt-5.2"


def test_spawn_signatures_accept_descriptor():
    """spawn_local/spawn_via_env expose the same optional kwargs (one writer)."""
    import inspect

    from tools.process_registry import ProcessRegistry as PR

    for name in ("spawn_local", "spawn_via_env"):
        params = inspect.signature(getattr(PR, name)).parameters
        for field in ("coding_backend", "coding_backend_version", "coding_session_id",
                      "coding_model", "coding_capabilities", "coding_resume_command"):
            assert field in params, f"{name} missing {field}"
            assert params[field].default in ("", None), f"{name}.{field} must default empty"


def test_checkpoint_fields_cover_descriptor():
    """Persisted verbatim; old checkpoints (keys absent) load as untyped."""
    for field in ("coding_backend", "coding_backend_version", "coding_session_id",
                  "coding_model", "coding_capabilities", "coding_resume_command"):
        assert field in _CHECKPOINT_FIELDS
    assert _CHECKPOINT_DEFAULTS["coding_backend"] == ""
    assert _CHECKPOINT_DEFAULTS["coding_capabilities"] == []


def test_recover_restores_descriptor_and_tolerates_old_entries(tmp_path, monkeypatch):
    """Crash recovery round-trips the descriptor; pre-field entries default."""
    import tools.process_registry as pr

    monkeypatch.setattr(pr, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(ProcessRegistry, "_host_pid_is_ours", lambda self, pid, st: True)
    entries = [
        {"session_id": "proc_new", "pid": 424242, "pid_scope": "host",
         "command": "codex", "task_id": "t", "owner_task_id": "t",
         "coding_backend": "codex", "coding_backend_version": "v9",
         "coding_session_id": "thr_9", "coding_model": "gpt-5",
         "coding_capabilities": ["hot_switch"], "coding_resume_command": "codex resume thr_9"},
        {"session_id": "proc_old", "pid": 424243, "pid_scope": "host",
         "command": "echo hi", "task_id": "t", "owner_task_id": "t"},
    ]
    (tmp_path / "processes.json").write_text(json.dumps(entries), encoding="utf-8")
    reg = ProcessRegistry()
    assert reg.recover_from_checkpoint() == 2
    new = reg._running["proc_new"]
    assert (new.coding_backend, new.coding_session_id, new.coding_model) == ("codex", "thr_9", "gpt-5")
    assert new.coding_capabilities == ["hot_switch"]
    assert new.detached is True
    old = reg._running["proc_old"]
    assert old.coding_backend == "" and old.coding_capabilities == []


def test_descriptor_never_captures_env_or_tokens():
    """Launch env and token-bearing argv must not leak into the descriptor:
    only explicitly passed ids/model names are stored."""
    secret_env = {"OPENAI_API_KEY": "sk-super-secret", "GH_TOKEN": "gho_secret"}
    s = ProcessRegistry._new_session(
        "codex --key gho_secret-value", "t1", "sa-1", "sk", "/tmp",
        coding_backend="codex", coding_model="gpt-5", coding_session_id="thr_1")
    blob = "|".join([s.coding_backend, s.coding_backend_version, s.coding_session_id,
                     s.coding_model, ",".join(s.coding_capabilities), s.coding_resume_command])
    assert "sk-super-secret" not in blob and "gho_secret" not in blob
    for key in secret_env:
        assert key not in blob
    # And the checkpoint entry carries no env mapping either.
    entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
    assert "OPENAI_API_KEY" not in json.dumps(entry)


def test_background_spawn_forwards_descriptor():
    """terminal_tool_background._spawn passes coding_* to the registry call."""
    from tools.terminal_tool_background import _spawn

    seen = {}

    class FakeRegistry:
        def spawn_local(self, **kwargs):
            seen.update(kwargs)
            session = ProcessSession(id="proc_fake", command=kwargs["command"])
            for key in ("coding_backend", "coding_backend_version", "coding_session_id",
                        "coding_model", "coding_capabilities", "coding_resume_command"):
                setattr(session, key, kwargs.get(key) or ([] if key == "coding_capabilities" else ""))
            return session

    session = _spawn(FakeRegistry(), env=None, env_type="local", command="codex",
                     cwd="/tmp", effective_task_id="t", task_id="t", session_key="sk",
                     effective_pty=False, coding_backend="codex", coding_session_id="thr_1",
                     coding_model="gpt-5")
    assert seen["coding_backend"] == "codex"
    assert seen["coding_session_id"] == "thr_1"
    assert session.coding_model == "gpt-5"


def test_terminal_schema_documents_coding_params():
    """Model-facing schema exposes the six optional params (background-only)."""
    from tools.terminal_tool import TERMINAL_SCHEMA

    props = TERMINAL_SCHEMA["parameters"]["properties"]
    for field in ("coding_backend", "coding_backend_version", "coding_session_id",
                  "coding_model", "coding_capabilities", "coding_resume_command"):
        assert field in props, f"schema missing {field}"
        assert "background" in props[field]["description"]
    assert TERMINAL_SCHEMA["parameters"].get("required") == ["command"]


def test_terminal_dispatch_rejects_coding_params_on_foreground():
    """coding_* on a foreground call fails with the corrected call (same shape
    as the notify/pty guards), never silently ignored."""
    from tools.terminal_tool import _handle_terminal

    result = _handle_terminal(
        {"command": "echo hi", "coding_backend": "codex"}, task_id="t")
    import json as _json

    body = _json.loads(result)
    assert body.get("exit_code", 0) != 0 or "error" in body
    assert "background" in _json.dumps(body)
