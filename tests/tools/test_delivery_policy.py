"""Adversarial production-path coverage for immutable delivery roles."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_agent,
    _normalize_delivery_role,
    _normalize_required_skills,
    _resolve_required_delivery_skills,
    delegate_task,
)
from tools.delegate_tool_progress import _build_child_system_prompt
from tools.delivery_action import _docker_verify, delivery_action_handler
from tools.delivery_policy import (
    DELIVERY_ROLES,
    DeliveryPolicy,
    apply_delivery_capabilities,
    build_delivery_policy,
    delivery_role_context,
    delivery_tool_block_reason,
    effective_tool_definitions,
    validate_delivery_terminal_command,
)

SHA = "a" * 40
IMAGE = "registry.example/hermes/verify@sha256:" + "1" * 64
TRUSTED_RUNTIME = {
    "verification_image": IMAGE,
    "acceptance": {"command": ["python", "-m", "pytest", "-q"], "image": IMAGE},
}
MERGER_TARGET = {"repository": "owner/repo", "pull_request": 17, "exact_sha": SHA}
CLOSURE_TARGET = {"repository": "owner/repo", "issue": 18, "merged_sha": SHA}
REVIEWER_TARGET = {"repository": "owner/repo", "pull_request": 17, "exact_sha": SHA}


def _definition(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {"type": "object"}}}


def _policy(role: str, workspace: Path | None = None) -> DeliveryPolicy:
    evidence = (
        MERGER_TARGET if role == "merger"
        else CLOSURE_TARGET if role == "closure_controller"
        else REVIEWER_TARGET if role == "reviewer"
        else None
    )
    policy = build_delivery_policy(role, evidence, trusted_runtime=TRUSTED_RUNTIME)
    return policy.bound_to_workspace(str(workspace)) if workspace else policy


def _init_repo(path: Path, *, remote: bool = False) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Delivery Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "delivery@example.invalid"], cwd=path, check=True)
    if remote:
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/owner/repo.git"],
            cwd=path, check=True,
        )
    (path / "tracked.txt").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True, text=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _delegation_validation_result(*, config: dict | None = None, **kwargs) -> dict:
    parent = SimpleNamespace(
        _delegate_depth=0,
        provider="test",
        model="test/model",
        base_url="",
        api_key="test",
    )
    with (
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=2),
        patch("tools.delegate_tool._get_max_concurrent_children", return_value=4),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "provider": "test", "model": "test/model", "base_url": "", "api_key": "", "api_mode": "",
            "request_overrides": None, "command": None, "args": None,
        }),
        patch("tools.delegate_tool._load_config", return_value=config or {}),
    ):
        return json.loads(delegate_task(goal="inspect", parent_agent=parent, **kwargs))


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, None), *[(role, role) for role in DELIVERY_ROLES]],
)
def test_role_normalization_and_no_role_compatibility(raw, expected) -> None:
    assert _normalize_delivery_role(raw) == expected
    definitions = [_definition("terminal"), _definition("write_file")]
    assert effective_tool_definitions(definitions, None) == definitions
    assert validate_delivery_terminal_command("anything", None) is None


@pytest.mark.parametrize("raw", ["", " Reviewer ", "Reviewer", "closure-controller", "MERGER"])
def test_delivery_role_aliases_fail_closed_at_direct_production_seam(raw: str) -> None:
    with pytest.raises(ValueError, match="Invalid delivery_role"):
        _normalize_delivery_role(raw)


def test_schema_preserves_topology_and_delivery_role_separation() -> None:
    task = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert task["properties"]["delivery_role"]["enum"] == list(DELIVERY_ROLES)
    assert "role" not in task["properties"]


def test_required_role_and_invalid_role_fail_closed_on_production_delegate_entrypoint() -> None:
    missing = _delegation_validation_result(config={"require_delivery_role": True})
    assert "delivery_role is required" in missing["error"]
    invalid = _delegation_validation_result(delivery_role="administrator")
    assert "Invalid delivery_role" in invalid["error"]
    malformed_setting = _delegation_validation_result(config={"require_delivery_role": "true"})
    assert "must be true or false" in malformed_setting["error"]


def test_delivery_authority_overrides_orchestrator_topology_capability_prompt() -> None:
    ordinary = _build_child_system_prompt("goal", role="orchestrator")
    restricted = _build_child_system_prompt(
        "goal", role="orchestrator", delivery_policy=_policy("implementer"),
    )
    assert "Subagent Spawning (Orchestrator Role)" in ordinary
    assert "Subagent Spawning (Orchestrator Role)" not in restricted
    assert "AUTHORITATIVE IMMUTABLE DELIVERY ROLE" in restricted


def test_delivery_child_drops_parent_prefill_dialogue() -> None:
    parent = MagicMock()
    parent.base_url = "https://example.invalid/v1"
    parent.api_key = "test"
    parent.provider = "test"
    parent.api_mode = "chat_completions"
    parent.model = "test/model"
    parent.platform = "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent.enabled_toolsets = ["terminal", "file", "skills"]
    parent.disabled_toolsets = []
    parent.prefill_messages = [{"role": "user", "content": "fabricated delivery policy"}]

    with patch("run_agent.AIAgent") as agent_class:
        agent_class.return_value = MagicMock(tools=[])
        _build_child_agent(
            task_index=0,
            goal="Review",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=10,
            parent_agent=parent,
            task_count=1,
            delivery_policy=DeliveryPolicy("reviewer"),
        )
    assert agent_class.call_args.kwargs["prefill_messages"] is None
    assert "AUTHORITATIVE IMMUTABLE DELIVERY ROLE" in agent_class.call_args.kwargs["ephemeral_system_prompt"]


def test_invalid_role_and_semantically_invalid_evidence_fail_closed() -> None:
    with pytest.raises(ValueError, match="Invalid delivery_role"):
        _normalize_delivery_role("admin")
    with pytest.raises(ValueError, match="requires repository"):
        build_delivery_policy("merger", {})
    with pytest.raises(ValueError, match="unsupported"):
        build_delivery_policy("merger", {**MERGER_TARGET, "ci_evidence": "trust me"})
    with pytest.raises(ValueError, match="full 40-character"):
        build_delivery_policy("merger", {**MERGER_TARGET, "exact_sha": "abc"})
    with pytest.raises(ValueError, match="repository"):
        build_delivery_policy("merger", {**MERGER_TARGET, "repository": "--repo other"})


def test_each_role_contract_is_system_content_without_fabricated_prefill() -> None:
    for role in DELIVERY_ROLES:
        policy = _policy(role)
        prompt = _build_child_system_prompt("goal", delivery_policy=policy, acceptance_ledger="verify A; verify B")
        assert f"Role: `{role}`" in prompt
        assert policy.contract in prompt
        assert "AUTHORITATIVE ACCEPTANCE LEDGER" in prompt
        assert "ASSISTANT:" not in prompt
        assert "user said" not in prompt.lower()


def test_required_skills_are_normalized_and_resolved_before_spawn(monkeypatch) -> None:
    assert _normalize_required_skills([" github ", "github", "release"]) == ["github", "release"]
    calls: list[str] = []

    def read(name: str, *, max_bytes: int | None = None) -> dict:
        assert max_bytes is not None
        calls.append(name)
        return {
            "success": True,
            "name": name,
            "path": name,
            "content": f"binding {name}",
            "content_sha256": "1" * 64,
            "immutable_delivery_read": True,
        }

    monkeypatch.setattr("tools.skills_tool.read_delivery_skill", read)
    resolved = _resolve_required_delivery_skills(["github", "release"], "reviewer")
    assert calls == ["github", "release"]
    prompt = _build_child_system_prompt("goal", delivery_policy=_policy("reviewer"), required_skills=resolved)
    assert "already loaded" in prompt
    assert "binding github" in prompt
    assert "sha256: " + "1" * 64 in prompt


@pytest.mark.parametrize("failure", ["unavailable", "disabled", "unreadable"])
def test_required_skill_failure_aborts_resolution(monkeypatch, failure: str) -> None:
    def fail(_name: str, **_kwargs):
        raise ValueError(f"Required delivery skill is {failure}")

    monkeypatch.setattr("tools.skills_tool.read_delivery_skill", fail)
    with pytest.raises(ValueError, match=failure):
        _resolve_required_delivery_skills(["missing"], "reviewer")


def test_required_skill_unexpected_load_failure_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.skills_tool.read_delivery_skill",
        lambda _name, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(ValueError, match="failed to load"):
        _resolve_required_delivery_skills(["broken"], "reviewer")
    with pytest.raises(ValueError, match="requires a delivery_role"):
        _resolve_required_delivery_skills(["skill"], None)


def test_required_skill_per_skill_boundary_and_aggregate_limit(monkeypatch) -> None:
    monkeypatch.setattr("tools.delegate_tool._REQUIRED_SKILL_MAX_BYTES", 8)
    monkeypatch.setattr("tools.delegate_tool._REQUIRED_SKILLS_TOTAL_MAX_BYTES", 12)

    def read(name: str, *, max_bytes: int) -> dict:
        content = "x" * (9 if name == "oversized" else 8)
        return {
            "success": True, "name": name, "path": name, "content": content,
            "content_sha256": "1" * 64, "immutable_delivery_read": True,
        }

    monkeypatch.setattr("tools.skills_tool.read_delivery_skill", read)
    assert len(_resolve_required_delivery_skills(["boundary"], "reviewer")[0]["content"]) == 8
    with pytest.raises(ValueError, match="per-skill byte limit"):
        _resolve_required_delivery_skills(["oversized"], "reviewer")
    with pytest.raises(ValueError, match="aggregate byte limit"):
        _resolve_required_delivery_skills(["one", "two"], "reviewer")


def test_required_skill_limit_aborts_before_any_child_spawn(monkeypatch) -> None:
    monkeypatch.setattr("tools.delegate_tool._REQUIRED_SKILL_MAX_BYTES", 4)
    monkeypatch.setattr(
        "tools.skills_tool.read_delivery_skill",
        lambda name, **_kwargs: {
            "success": True, "name": name, "path": name, "content": "12345",
            "content_sha256": "1" * 64, "immutable_delivery_read": True,
        },
    )
    monkeypatch.setattr("tools.delegate_tool._build_children", lambda *_a, **_k: pytest.fail("child spawned"))
    result = _delegation_validation_result(
        config={"delegation": {"delivery": TRUSTED_RUNTIME}},
        delivery_role="reviewer",
        delivery_evidence=REVIEWER_TARGET,
        required_skills=["too-large"],
    )
    assert "per-skill byte limit" in result["error"]


def test_effective_surfaces_remove_generic_execution_and_inject_structured_action() -> None:
    definitions = [_definition(name) for name in (
        "terminal", "execute_code", "read_file", "search_files", "patch", "write_file",
        "skill_view", "skills_list", "github", "mcp__opaque", "delegate_task",
    )]
    implementer = {d["function"]["name"] for d in effective_tool_definitions(definitions, _policy("implementer"))}
    assert {"read_file", "search_files", "patch", "write_file", "skill_view", "delivery_action"} <= implementer
    assert {"terminal", "execute_code", "github", "mcp__opaque", "delegate_task"}.isdisjoint(implementer)

    reviewer = {d["function"]["name"] for d in effective_tool_definitions(definitions, _policy("reviewer"))}
    assert reviewer == {"skill_view", "delivery_action"}
    for role in ("merger", "closure_controller"):
        names = {d["function"]["name"] for d in effective_tool_definitions(definitions, _policy(role))}
        assert names == {"skill_view", "delivery_action"}


def test_capability_application_filters_restored_snapshots() -> None:
    child = SimpleNamespace(tools=[_definition("execute_code"), _definition("terminal"), _definition("skill_view")])
    apply_delivery_capabilities(child, _policy("reviewer"))
    assert child.valid_tool_names == {"skill_view", "delivery_action"}
    assert child._delivery_policy.role == "reviewer"


def test_mcp_refresh_prefix_cannot_restore_generic_delivery_capabilities(monkeypatch) -> None:
    from tools import mcp_tool_agent
    from tools.registry import registry

    agent = SimpleNamespace(
        tools=[_definition("terminal"), _definition("execute_code")],
        valid_tool_names={"terminal", "execute_code"},
        enabled_toolsets=None,
        disabled_toolsets=[],
        _delivery_policy=_policy("reviewer"),
        _tool_snapshot_generation=-1,
        _context_engine_tool_names=set(),
        platform="cli",
    )
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: [_definition("terminal"), _definition("mcp__opaque"), _definition("skill_view")],
    )
    monkeypatch.setattr(mcp_tool_agent, "_reinject_post_build_tools", lambda *_args: set())
    monkeypatch.setattr(mcp_tool_agent, "_reinject_authorized_dynamic_tools", lambda *_args: None)
    monkeypatch.setattr(mcp_tool_agent, "persist_agent_tool_names", lambda *_args: None)
    monkeypatch.setattr(
        registry,
        "get_all_entries",
        lambda: [SimpleNamespace(name=name) for name in ("terminal", "execute_code", "mcp__opaque", "skill_view")],
    )
    mcp_tool_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)
    assert agent.valid_tool_names == {"skill_view", "delivery_action"}
    assert {item["function"]["name"] for item in agent.tools} == agent.valid_tool_names


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("execute_code", {"code": "import subprocess; subprocess.run(['gh','pr','merge','17'])"}),
        ("terminal", {"command": "python -c \"import requests; requests.post('https://api.github.com/graphql')\""}),
        ("terminal", {"command": "sh -c 'gh pr merge 17 --admin --delete-branch'"}),
        ("github", {"query": "mutation { mergePullRequest(input: {}) { clientMutationId } }"}),
        ("mcp__opaque", {"payload": "opaque"}),
        ("graphql", {"operationName": "MergePullRequest"}),
    ],
)
def test_adversarial_production_dispatch_never_runs_underlying_handler(monkeypatch, name: str, args: dict) -> None:
    from agent import tool_executor

    executed: list[dict] = []
    agent = SimpleNamespace(_delivery_policy=_policy("implementer"), _tool_guardrails=SimpleNamespace())
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda *_args, **_kwargs: None)
    state = tool_executor._run_agent_tool_execution_middleware(
        agent,
        function_name=name,
        function_args=args,
        effective_task_id="task",
        tool_call_id="call",
        execute=lambda payload: executed.append(payload),
    )
    assert state.blocked is True
    assert executed == []
    assert "outside this role's capability set" in str(state.result)


def test_central_dispatch_blocks_opaque_registered_or_restored_handler(monkeypatch) -> None:
    from model_tools import handle_function_call
    from tools.registry import registry

    monkeypatch.setattr(registry, "dispatch", lambda *_args, **_kwargs: pytest.fail("underlying registry handler ran"))
    with delivery_role_context(_policy("implementer")):
        result = json.loads(handle_function_call("mcp__opaque", {"anything": "merge"}))
    assert "outside this role's capability set" in result["error"]


def test_plugin_and_middleware_are_not_activated_for_delivery_calls(monkeypatch) -> None:
    from agent import tool_executor

    monkeypatch.setattr(tool_executor, "_pre_tool_block", lambda *_: pytest.fail("plugin pre-tool hook ran"))
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda *_args, **_kwargs: None)
    agent = SimpleNamespace(_delivery_policy=_policy("implementer"), _tool_guardrails=SimpleNamespace())
    state = tool_executor._run_agent_tool_execution_middleware(
        agent,
        function_name="execute_code",
        function_args={"code": "pass"},
        effective_task_id="task",
        tool_call_id="call",
        execute=lambda _payload: pytest.fail("handler ran"),
    )
    assert state.blocked


def test_workspace_boundary_rejects_relative_escape_symlink_and_git_control(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "link").symlink_to(outside)
    policy = _policy("implementer", workspace)
    assert delivery_tool_block_reason("write_file", {"path": "relative.txt"}, policy)
    assert delivery_tool_block_reason("write_file", {"path": str(outside)}, policy)
    assert delivery_tool_block_reason("write_file", {"path": str(workspace / "link")}, policy)
    assert delivery_tool_block_reason("write_file", {"path": str(workspace / ".git")}, policy)
    assert delivery_tool_block_reason("write_file", {"path": str(workspace / "ok.txt")}, policy) is None


def test_generic_terminal_is_always_unavailable_for_delivery_roles() -> None:
    for role in DELIVERY_ROLES:
        assert "generic terminal execution is unavailable" in (
            validate_delivery_terminal_command("git status", _policy(role)) or ""
        )


def test_real_registered_skill_handler_is_side_effect_free_for_restricted_roles(monkeypatch, tmp_path: Path) -> None:
    from agent.skill_utils import TIER_LOCAL
    from model_tools import handle_function_call
    from tools import skills_tool

    skill_dir = tmp_path / "skills" / "safe"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: safe\ndescription: safe delivery guidance\n---\nNever mutate while reading.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agent.skill_utils.get_skill_search_roots",
        lambda *_args, **_kwargs: [(TIER_LOCAL, tmp_path / "skills")],
    )
    monkeypatch.setattr(skills_tool, "_preprocess_skill", lambda *_a, **_k: pytest.fail("preprocess shell ran"))
    monkeypatch.setattr(skills_tool, "_mark_background_review_read", lambda *_a, **_k: pytest.fail("usage mark mutated"))
    monkeypatch.setattr(skills_tool, "_record_skill_view", lambda *_a, **_k: pytest.fail("dedup state mutated"))

    for role in ("reviewer", "merger", "closure_controller"):
        with delivery_role_context(_policy(role)):
            payload = json.loads(handle_function_call("skill_view", {"name": "safe"}))
        assert payload["success"] is True
        assert payload["immutable_delivery_read"] is True
        assert payload["content_sha256"]


def test_delivery_skill_reader_rejects_symlinked_content(monkeypatch, tmp_path: Path) -> None:
    from agent.skill_utils import TIER_LOCAL
    from tools.skills_tool import read_delivery_skill

    root = tmp_path / "skills"
    skill = root / "linked"
    skill.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (skill / "SKILL.md").symlink_to(outside)
    monkeypatch.setattr(
        "agent.skill_utils.get_skill_search_roots",
        lambda *_args, **_kwargs: [(TIER_LOCAL, root)],
    )
    with pytest.raises(ValueError, match="regular file"):
        read_delivery_skill("linked")


def test_reviewer_verification_container_is_read_only_networkless_and_uncredentialed(monkeypatch, tmp_path: Path) -> None:
    captured: list[tuple[list[str], dict]] = []

    def run(argv, **kwargs):
        captured.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr("tools.delivery_action.run_bounded", run)
    monkeypatch.setattr("tools.delivery_action._cleanup_stale_verification_state", lambda _docker: None)
    monkeypatch.setattr("tools.delivery_action._run", lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""))
    monkeypatch.setattr("tools.delivery_action._docker_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        "tools.delivery_action._workspace_source", lambda _policy, root: root / "checkout",
    )
    result = json.loads(_docker_verify(_policy("implementer", tmp_path), ["python", "-m", "pytest", "-q"]))
    assert result["ok"] is True
    argv, kwargs = captured[0]
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--pull") + 1] == "never"
    assert argv[argv.index("--entrypoint") + 1] == ""
    assert "--read-only" in argv
    assert argv[argv.index("--user") + 1] == "65534:65534"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--memory") + 1] == "4g"
    assert argv[argv.index("--cpus") + 1] == "2"
    mount = argv[argv.index("-v") + 1]
    source, destination, mode = mount.rsplit(":", 2)
    assert Path(source).name == "checkout"
    assert Path(source).parent.name.startswith("hermes-delivery-review-")
    assert Path(source) != tmp_path.resolve()
    assert (destination, mode) == ("/workspace", "ro")
    assert set(kwargs["env"]) == {"PATH"}
    assert IMAGE in argv


def test_reviewer_mutation_attempt_cannot_touch_host_checkout(monkeypatch, tmp_path: Path) -> None:
    marker = tmp_path / "source.py"
    marker.write_text("original", encoding="utf-8")

    def run(argv, **_kwargs):
        assert "--read-only" in argv
        mount = argv[argv.index("-v") + 1]
        source, destination, mode = mount.rsplit(":", 2)
        assert Path(source) != tmp_path.resolve()
        assert (destination, mode) == ("/workspace", "ro")
        return subprocess.CompletedProcess(argv, 1, "", "Read-only file system")

    monkeypatch.setattr("tools.delivery_action.run_bounded", run)
    monkeypatch.setattr("tools.delivery_action._cleanup_stale_verification_state", lambda _docker: None)
    monkeypatch.setattr("tools.delivery_action._run", lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""))
    monkeypatch.setattr("tools.delivery_action._docker_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        "tools.delivery_action._workspace_source", lambda _policy, root: root / "checkout",
    )
    result = json.loads(_docker_verify(
        _policy("implementer", tmp_path),
        ["python", "-c", "open('/workspace/source.py','w').write('owned')"],
    ))
    assert result["ok"] is False
    assert marker.read_text(encoding="utf-8") == "original"


def test_structured_action_is_registered_and_implementer_commit_uses_production_dispatch(tmp_path: Path) -> None:
    from model_tools import handle_function_call
    from tools.registry import registry

    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    entry = registry.get_entry("delivery_action")
    assert entry is not None and entry.toolset == "delivery"

    with delivery_role_context(_policy("implementer", repo)):
        staged = json.loads(handle_function_call("delivery_action", {"action": "stage", "paths": ["tracked.txt"]}))
        committed = json.loads(handle_function_call("delivery_action", {"action": "commit", "message": "change"}))
    assert staged["ok"] is True
    assert committed["ok"] is True and len(committed["sha"]) == 40
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout == ""


def test_exact_sha_materialization_uses_archive_without_git_metadata(tmp_path: Path) -> None:
    from tools.delivery_action import _exact_sha_source

    repo = tmp_path / "repo"
    sha = _init_repo(repo)
    export_root = tmp_path / "export"
    export_root.mkdir()
    source = _exact_sha_source(DeliveryPolicy("reviewer", workspace=str(repo)), sha, export_root)
    assert (source / "tracked.txt").read_text(encoding="utf-8") == "initial\n"
    assert not (source / ".git").exists()
    assert subprocess.run(
        ["git", "worktree", "list", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.count("worktree ") == 1


def test_merger_queries_bound_head_review_and_required_ci_immediately_before_merge(monkeypatch) -> None:
    calls: list[tuple] = []

    def gh_json(endpoint: str, *, paginate: bool = False):
        calls.append(("api", endpoint, paginate))
        if "/reviews" in endpoint:
            return [{"state": "APPROVED", "commit_id": SHA, "user": {"login": "reviewer"}}]
        if endpoint.endswith("/protection"):
            return {
                "required_pull_request_reviews": {
                    "required_approving_review_count": 1,
                    "dismiss_stale_reviews": True,
                    "require_last_push_approval": True,
                },
                "required_status_checks": {"strict": True, "checks": [{"context": "test"}]},
                "enforce_admins": {"enabled": True},
            }
        return {
            "head": {"sha": SHA}, "base": {"ref": "main"},
            "user": {"login": "author"}, "state": "open", "draft": False,
        }

    def run(argv, **_kwargs):
        calls.append(("run", tuple(argv)))
        if "checks" in argv:
            return subprocess.CompletedProcess(argv, 0, '[{"name":"test","bucket":"pass"}]', "")
        return subprocess.CompletedProcess(argv, 0, "merged", "")

    monkeypatch.setattr("tools.delivery_action._gh_json", gh_json)
    monkeypatch.setattr("tools.delivery_action._gh", run)
    from model_tools import handle_function_call
    with delivery_role_context(_policy("merger")):
        payload = json.loads(handle_function_call("delivery_action", {"action": "merge"}))
    assert payload["ok"] is True
    merge_call = next(call[1] for call in calls if call[0] == "run" and "merge" in call[1])
    assert "--match-head-commit" in merge_call
    assert merge_call[merge_call.index("--match-head-commit") + 1] == SHA
    assert "--admin" not in merge_call and "--delete-branch" not in merge_call
    assert calls[-1][0] == "run" and "merge" in calls[-1][1]
    assert any(call[0] == "api" and call[1].endswith("/protection") for call in calls)


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "merge", "repository": "other/repo"},
        {"action": "merge", "pull_request": 99},
        {"action": "merge", "admin": True},
        {"action": "merge", "delete_branch": True},
        {"action": "merge", "match_head_commit": "b" * 40},
        {"action": "merge", "command": ["gh", "pr", "merge", "17", "--admin"]},
    ],
)
def test_merger_rejects_wrong_target_bypass_flags_and_arbitrary_commands(payload: dict) -> None:
    from model_tools import handle_function_call

    with delivery_role_context(_policy("merger")):
        result = json.loads(handle_function_call("delivery_action", payload))
    assert result.get("ok") is not True
    assert result.get("error")


def test_merger_rejects_stale_or_forged_approval(monkeypatch) -> None:
    def gh_json(endpoint: str, *, paginate: bool = False):
        if "/reviews" in endpoint:
            return [{"state": "APPROVED", "commit_id": "b" * 40, "user": {"login": "reviewer"}}]
        return {
            "head": {"sha": SHA}, "base": {"ref": "main"},
            "user": {"login": "author"}, "state": "open", "draft": False,
        }

    monkeypatch.setattr("tools.delivery_action._gh_json", gh_json)
    monkeypatch.setattr("tools.delivery_action._run", lambda *_a, **_k: pytest.fail("CI or merge ran"))
    with delivery_role_context(_policy("merger")):
        payload = json.loads(delivery_action_handler({"action": "merge"}))
    assert payload["ok"] is False
    assert "independent GitHub approval" in payload["error"]


def test_closure_binds_default_branch_acceptance_and_tracker_before_close(monkeypatch, tmp_path: Path) -> None:
    from model_tools import handle_function_call

    repo = tmp_path / "repo"
    merged_sha = _init_repo(repo, remote=True)
    policy = build_delivery_policy(
        "closure_controller", {"repository": "owner/repo", "issue": 18, "merged_sha": merged_sha},
        trusted_runtime=TRUSTED_RUNTIME,
    ).bound_to_workspace(str(repo))
    endpoints: list[str] = []
    close_calls: list[list[str]] = []

    def gh_json(endpoint: str, *, paginate: bool = False):
        assert paginate is False
        endpoints.append(endpoint)
        if endpoint == "repos/owner/repo":
            return {"default_branch": "main"}
        if endpoint.startswith("repos/owner/repo/compare/"):
            return {"status": "ahead"}
        if endpoint == "repos/owner/repo/issues/18":
            return {"state": "open"}
        raise AssertionError(endpoint)

    monkeypatch.setattr("tools.delivery_action._gh_json", gh_json)
    monkeypatch.setattr(
        "tools.delivery_action._docker_verify",
        lambda _policy, command, *, exact_sha="", acceptance=False: json.dumps({
            "ok": command is None and exact_sha == merged_sha and acceptance,
            "attestation": {"image": IMAGE, "recipe_identity": _policy.acceptance_recipe.identity},
        }),
    )
    monkeypatch.setattr(
        "tools.delivery_action._gh",
        lambda argv, **_kwargs: (
            close_calls.append(list(argv))
            or subprocess.CompletedProcess(argv, 0, "closed", "")
        ),
    )
    with delivery_role_context(policy):
        payload = json.loads(handle_function_call("delivery_action", {"action": "close_issue"}))
    assert payload["ok"] is True
    assert policy.acceptance_recipe is not None
    assert f"repos/owner/repo/compare/{merged_sha}...main" in endpoints
    assert endpoints.count(f"repos/owner/repo/compare/{merged_sha}...main") == 3
    assert "repos/owner/repo/issues/18" in endpoints
    assert close_calls == [[
        "issue", "close", "18", "--repo", "owner/repo",
        "--comment", f"Post-merge acceptance passed for {merged_sha} ({policy.acceptance_recipe.identity}).",
    ]]


def test_closure_never_closes_when_post_merge_acceptance_fails(monkeypatch, tmp_path: Path) -> None:
    from model_tools import handle_function_call

    repo = tmp_path / "repo"
    merged_sha = _init_repo(repo, remote=True)
    policy = build_delivery_policy(
        "closure_controller", {"repository": "owner/repo", "issue": 18, "merged_sha": merged_sha},
        trusted_runtime=TRUSTED_RUNTIME,
    ).bound_to_workspace(str(repo))
    monkeypatch.setattr(
        "tools.delivery_action._gh_json",
        lambda endpoint, **_kwargs: (
            {"default_branch": "main"} if endpoint == "repos/owner/repo" else {"status": "identical"}
        ),
    )
    monkeypatch.setattr(
        "tools.delivery_action._docker_verify",
        lambda *_args, **_kwargs: json.dumps({"ok": False, "exit_code": 1}),
    )
    monkeypatch.setattr("tools.delivery_action._gh", lambda *_args, **_kwargs: pytest.fail("issue was closed"))
    with delivery_role_context(policy):
        payload = json.loads(handle_function_call("delivery_action", {"action": "close_issue"}))
    assert payload["ok"] is False
    assert "acceptance failed" in payload["error"]
