"""File policy contracts on native tools; the final backend never writes."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools import file_tools as ft, file_tools_paths as ftp


@pytest.fixture
def policy(tmp_path, monkeypatch):
    assert "pre_file_write" in plugins.VALID_HOOKS
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    ctx = PluginContext(PluginManifest(name="file-policy-test"), manager)
    anchor = [str(tmp_path)]
    monkeypatch.setattr(ftp, "_terminal_env_type_for_task", lambda task: "local")
    monkeypatch.setattr(ftp, "_authoritative_workspace_root", lambda task: anchor[0])
    # Isolate the new gate from unrelated native write/read-baseline policies.
    for name in ("_check_sensitive_path", "_check_binary_document_write", "_check_protected_instruction_write",
                 "_check_approval_required_write", "_check_cross_profile_path", "_stale_overwrite_blocker"):
        monkeypatch.setattr(ft, name, lambda *a, **kw: None)
    monkeypatch.setattr(ft, "_edit_warnings", lambda *a: [])
    monkeypatch.setattr(ft, "_whole_file_rewrite_hint", lambda *a: None)
    reached = []
    def finish(kind, *args):
        reached.append((kind, args))
        return SimpleNamespace(to_dict=lambda: {"error": "simulated backend result"})
    backend = SimpleNamespace(env=None, write_file=lambda *a: finish("write", *a),
                              patch_replace=lambda *a: finish("replace", *a),
                              patch_v4a=lambda *a: finish("v4a", *a))
    monkeypatch.setattr(ft, "_get_file_ops", lambda task: backend)
    yield SimpleNamespace(ctx=ctx, manager=manager, reached=reached, anchor=anchor,
                          backend=backend, root=tmp_path)
    manager.unload()


def call(kind, path="notes.txt"):
    if kind == "write":
        return json.loads(ft.write_file_tool(path, "new", task_id="task", session_id="session"))
    if kind == "replace":
        return json.loads(ft.patch_tool(path=path, old_string="old", new_string="new", task_id="task", session_id="session"))
    return json.loads(ft.patch_tool(mode="patch", patch=f"*** Begin Patch\n*** Update File: {path}\n@@\n-old\n+new\n*** End Patch",
                                   task_id="task", session_id="session"))


@pytest.mark.parametrize("kind", ["write", "replace", "v4a"])
def test_veto_stops_native_backend_and_removing_hook_removes_veto(policy, kind):
    policy.ctx.register_hook("pre_file_write", lambda **kw: {"action": "block", "message": "policy veto"})
    assert "policy veto" in call(kind)["error"]
    assert not policy.reached
    policy.manager._hooks["pre_file_write"].clear()
    assert "simulated" in call(kind)["error"]
    assert len(policy.reached) == 1


@pytest.mark.parametrize("kind", ["write", "replace", "v4a"])
def test_checked_paths_and_content_survive_callback_cwd_backend_and_payload_changes(policy, monkeypatch, kind):
    seen = []
    def inspect(**kw):
        seen.append(kw)
        assert kw["targets"][0]["path"] == str(policy.root / "notes.txt")
        assert kw["namespace"] == "host" and kw["task_id"] == "task" and kw["session_id"] == "session"
        assert kw["write_id"]
        kw["targets"][0]["path"] = "/tmp/decoy"
        kw["args"].update(content="mutated", new_string="mutated", patch="mutated")
        policy.anchor[0] = str(policy.root / "other")
        monkeypatch.setattr(ft, "_get_file_ops", lambda task: pytest.fail("backend was selected again after policy"))
    policy.ctx.register_hook("pre_file_write", inspect)
    assert "simulated" in call(kind)["error"]
    assert len(seen) == 1 and len(policy.reached) == 1
    _kind, args = policy.reached[0]
    if kind == "v4a":
        assert f"*** Update File: {policy.root}/notes.txt" in args[0]
        assert "+new" in args[0] and "mutated" not in args[0]
    else:
        assert args[0] == str(policy.root / "notes.txt")
        assert args[1 if kind == "write" else 2] == "new"


def test_every_v4a_operation_and_both_move_endpoints_use_entry_semantics(policy):
    target = policy.root / "real.txt"
    target.write_text("old")
    (policy.root / "link.txt").symlink_to(target)
    patch = "\n".join(["*** Begin Patch", "*** Add File: added.txt", "+new", "*** Update File: link.txt",
                        "@@", "-old", "+new", "*** Delete File: link.txt", "*** Move File: link.txt -> moved.txt", "*** End Patch"])
    seen = []
    policy.ctx.register_hook("pre_file_write", lambda **kw: seen.append(kw) or {"action": "block", "message": "veto"})
    result = json.loads(ft.patch_tool(mode="patch", patch=patch, task_id="task"))
    assert "veto" in result["error"] and not policy.reached
    pairs = [(t["operation"], t["path"]) for t in seen[0]["targets"]]
    assert pairs == [("add", str(policy.root / "added.txt")), ("update", str(target)),
                     ("delete", str(policy.root / "link.txt")), ("move_source", str(policy.root / "link.txt")),
                     ("move_destination", str(policy.root / "moved.txt"))]


@pytest.mark.parametrize("kind", ["write", "replace", "v4a"])
@pytest.mark.parametrize("host_namespace", [True, False])
def test_unknown_resolution_and_foreign_namespace_never_fall_back_to_unchecked_write(policy, monkeypatch, kind, host_namespace):
    policy.ctx.register_hook("pre_file_write", lambda **kw: None)
    if host_namespace:
        monkeypatch.setattr(ft, "_resolve_or_none", lambda *a, **kw: None)
    else:
        monkeypatch.setattr(ft, "_file_ops_uses_host_paths", lambda ops: False)
    assert "BLOCKED" in call(kind)["error"] and not policy.reached


@pytest.mark.parametrize("directive", [False, "allow", {}, {"action": "modify", "args": {}}, {"action": "approve", "approval_policy": "unknown"}])
def test_malformed_policy_directives_fail_closed(policy, directive):
    policy.ctx.register_hook("pre_file_write", lambda **kw: directive)
    assert "BLOCKED" in call("write")["error"] and not policy.reached


def test_callback_exception_and_timeout_fail_closed(policy, monkeypatch):
    policy.ctx.register_hook("pre_file_write", lambda **kw: 1 / 0)
    assert "error" in call("write") and not policy.reached
    policy.manager._hooks["pre_file_write"].clear()
    from hermes_cli import plugins_dispatch
    monkeypatch.setattr(policy.manager, "_run_hook_callback_bounded", lambda *a: plugins_dispatch._HOOK_SKIPPED)
    policy.ctx.register_hook("pre_file_write", lambda **kw: None)
    assert "error" in call("write") and not policy.reached


@pytest.mark.parametrize("approved", [True, False])
def test_native_approval_is_used_and_a_veto_wins_before_prompt(policy, monkeypatch, approved):
    from tools import approval
    prompts = []
    monkeypatch.setattr(approval, "request_tool_approval", lambda *a, **kw: prompts.append((a, kw)) or {"approved": approved})
    policy.ctx.register_hook("pre_file_write", lambda **kw: {"action": "approve", "message": "review"})
    call("write")
    assert len(prompts) == 1 and bool(policy.reached) is approved
    policy.reached.clear();prompts.clear()
    policy.ctx.register_hook("pre_file_write", lambda **kw: {"action": "block", "message": "veto"})
    assert "veto" in call("write")["error"]
    assert not prompts and not policy.reached


def test_concurrent_effects_are_distinct_policy_calls(policy):
    barrier = threading.Barrier(2)
    ids = []
    def inspect(**kw):
        ids.append(kw["write_id"])
        barrier.wait(timeout=5)
    policy.ctx.register_hook("pre_file_write", inspect)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: call("write", name), ["first.txt", "second.txt"]))
    assert all("simulated" in r["error"] for r in results)
    assert len(set(ids)) == 2 and len(policy.reached) == 2


def test_approval_failure_stops_effect(policy, monkeypatch):
    from tools import approval
    def broken(*a, **kw):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(approval, "request_tool_approval", broken)
    policy.ctx.register_hook("pre_file_write", lambda **kw: {"action": "approve"})
    assert "BLOCKED" in call("write")["error"] and not policy.reached


def test_mandatory_without_host_support_never_uses_default_gate(policy, monkeypatch):
    from tools import approval
    monkeypatch.setattr(plugins, "VALID_TOOL_APPROVAL_POLICIES", frozenset(), raising=False)
    monkeypatch.setattr(approval, "request_tool_approval", lambda *a, **kw: pytest.fail("mandatory fell back"))
    policy.ctx.register_hook("pre_file_write", lambda **kw: {"action": "approve", "approval_policy": "mandatory"})
    assert "unavailable" in call("write")["error"] and not policy.reached


@pytest.mark.skipif("mandatory" not in getattr(plugins, "VALID_TOOL_APPROVAL_POLICIES", ()), reason="P1 is not present")
@pytest.mark.parametrize("choice", ["once", "deny"])
def test_p1_mandatory_uses_the_bound_snapshot_and_survives_yolo_off(policy, monkeypatch, choice):
    from tools import approval
    seen = []
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (cb, True, False, False))
    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "off")
    monkeypatch.setattr(approval, "is_approved", lambda *a: True)
    monkeypatch.setattr(approval, "_present_with_selected_transport", lambda **kw: {"selected": False})
    def prompt(command, description, **kwargs):
        seen.append((json.loads(command), kwargs))
        return choice
    monkeypatch.setattr(approval, "prompt_dangerous_approval", prompt)
    policy.ctx.register_hook("pre_file_write", lambda **kw: {"action": "approve", "approval_policy": "mandatory"})
    for _ in range(2):call("write")
    assert len(seen) == 2 and len(policy.reached) == (2 if choice == "once" else 0)
    assert all(item[0]["args"]["targets"][0]["path"] == str(policy.root / "notes.txt") for item in seen)
    assert all(not item[1]["allow_session"] and not item[1]["allow_permanent"] for item in seen)
