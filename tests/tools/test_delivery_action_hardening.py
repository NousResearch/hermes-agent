"""Second-round adversarial hardening tests for structured delivery effects."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import psutil
import pytest

from tools.delivery_action import (
    DeliveryActionError,
    _cleanup_stale_verification_state,
    _docker_verify,
    _exact_sha_source,
    _extract_git_archive,
    _gh,
    _workspace_source,
    delivery_action_handler,
)
from tools.delivery_action_runtime import DeliveryRuntimeError, require_free_disk, run_bounded
from tools.delivery_policy import build_delivery_policy, delivery_role_context


SHA = "a" * 40
IMAGE = "registry.example/hermes/verify@sha256:" + "2" * 64
RUNTIME = {
    "verification_image": IMAGE,
    "acceptance": {"command": ["python", "-m", "pytest", "-q"], "image": IMAGE},
}


def _init_repo(path: Path, *, remote: bool = False) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Delivery Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "delivery@example.invalid"], cwd=path, check=True)
    if remote:
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/owner/repo.git"],
            cwd=path, check=True,
        )
    (path / "tracked.txt").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _closure_policy(repo: Path, sha: str):
    return build_delivery_policy(
        "closure_controller",
        {"repository": "owner/repo", "issue": 18, "merged_sha": sha},
        trusted_runtime=RUNTIME,
    ).bound_to_workspace(repo)


@pytest.mark.parametrize(
    "runtime",
    [
        {"verification_image": "python:3.12"},
        {"verification_image": "local-image@sha256:" + "1" * 64},
        {"verification_image": "registry.example/image@sha256:not-a-digest"},
    ],
)
def test_verification_image_rejects_mutable_tags_and_untrusted_local_names(runtime: dict) -> None:
    with pytest.raises(ValueError, match="immutable image reference"):
        build_delivery_policy("implementer", trusted_runtime=runtime)


def test_worker_schema_has_no_image_and_closure_cannot_override_recipe(monkeypatch, tmp_path: Path) -> None:
    from tools.delivery_action import DELIVERY_ACTION_SCHEMA

    assert "image" not in DELIVERY_ACTION_SCHEMA["parameters"]["properties"]
    repo = tmp_path / "repo"
    sha = _init_repo(repo, remote=True)
    policy = _closure_policy(repo, sha)
    monkeypatch.setattr("tools.delivery_action._docker_verify", lambda *_a, **_k: pytest.fail("override reached runner"))
    with delivery_role_context(policy):
        result = json.loads(delivery_action_handler({
            "action": "close_issue", "command": ["true"], "image": "evil:latest",
        }))
    assert result["ok"] is False
    assert "unsupported argument" in result["error"]


def test_policy_bound_digest_and_neutral_entrypoint_are_used(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    policy = build_delivery_policy("implementer", trusted_runtime=RUNTIME).bound_to_workspace(tmp_path)

    def bounded(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr("tools.delivery_action.run_bounded", bounded)
    monkeypatch.setattr("tools.delivery_action._cleanup_stale_verification_state", lambda _docker: None)
    monkeypatch.setattr("tools.delivery_action._run", lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""))
    monkeypatch.setattr("tools.delivery_action._docker_path", lambda: "/usr/bin/docker")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr("tools.delivery_action._workspace_source", lambda _policy, _root: snapshot)
    result = json.loads(_docker_verify(policy, ["trusted-runner", "verify"]))
    argv = calls[0]
    assert result["attestation"] == {
        "image": IMAGE, "recipe_identity": None, "entrypoint": "neutralized",
    }
    assert argv[argv.index("--entrypoint") + 1] == ""
    assert argv[argv.index("-w") + 2] == IMAGE
    assert argv[-2:] == ["trusted-runner", "verify"]


def test_gh_uses_absolute_binary_github_host_and_minimal_profile_secret_env(monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setenv("PATH", "/attacker/bin")
    monkeypatch.setenv("GH_HOST", "evil.example")
    monkeypatch.setenv("GH_CONFIG_DIR", "/attacker/config")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-for-gh")
    monkeypatch.setattr("tools.delivery_action._trusted_gh_path", lambda: "/usr/bin/gh")
    monkeypatch.setattr(
        "agent.secret_scope.get_secret",
        lambda name, *args: "served-profile-token" if name == "GH_TOKEN" else None,
    )

    def run(argv, **kwargs):
        seen.update(argv=list(argv), **kwargs)
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    monkeypatch.setattr("tools.delivery_action._run", run)
    _gh(["api", "--hostname", "github.com", "repos/owner/repo"])
    assert seen["argv"][0] == "/usr/bin/gh"
    assert seen["env"] == {
        "GH_TOKEN": "served-profile-token",
        "GH_HOST": "github.com",
        "GH_PROMPT_DISABLED": "1",
        "GH_PAGER": "cat",
        "PAGER": "cat",
        "LANG": "C.UTF-8",
    }


def _approved_pr(endpoint: str, *, protection: object) -> object:
    if "/reviews" in endpoint:
        return [{"state": "APPROVED", "commit_id": SHA, "user": {"login": "reviewer"}}]
    if endpoint.endswith("/protection"):
        return protection
    if "/rules/branches/" in endpoint:
        return []
    return {
        "head": {"sha": SHA}, "base": {"ref": "main"},
        "user": {"login": "author"}, "state": "open", "draft": False,
    }


@pytest.mark.parametrize("protection", [{}, {"required_pull_request_reviews": {}, "required_status_checks": {}}])
def test_merge_fails_closed_when_repository_protection_is_absent_or_changed(monkeypatch, protection) -> None:
    monkeypatch.setattr(
        "tools.delivery_action._gh_json",
        lambda endpoint, **_kwargs: _approved_pr(endpoint, protection=protection),
    )

    def gh(argv):
        if "checks" in argv:
            return subprocess.CompletedProcess(argv, 0, '[{"name":"test","bucket":"pass"}]', "")
        pytest.fail("merge ran without proven repository protection")

    monkeypatch.setattr("tools.delivery_action._gh", gh)
    policy = build_delivery_policy(
        "merger", {"repository": "owner/repo", "pull_request": 17, "exact_sha": SHA},
        trusted_runtime=RUNTIME,
    )
    with delivery_role_context(policy):
        result = json.loads(delivery_action_handler({"action": "merge"}))
    assert result["ok"] is False
    assert "protection does not provably enforce" in result["error"]


def test_merge_rechecks_protection_and_refuses_a_midflight_change(monkeypatch) -> None:
    protection_calls = 0
    commands: list[list[str]] = []

    def protection(_repository: str, _branch: str) -> dict[str, object]:
        nonlocal protection_calls
        protection_calls += 1
        if protection_calls == 2:
            raise DeliveryActionError("repository protection changed")
        return {"kind": "branch_protection", "required_checks": 1}

    def gh_json(endpoint: str, **_kwargs: object) -> object:
        if "/reviews" in endpoint:
            return [{"state": "APPROVED", "commit_id": SHA, "user": {"login": "reviewer"}}]
        return {
            "head": {"sha": SHA}, "base": {"ref": "main"},
            "user": {"login": "author"}, "state": "open", "draft": False,
        }

    def gh(argv: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        if "checks" in argv:
            return subprocess.CompletedProcess(argv, 0, '[{"name":"test","bucket":"pass"}]', "")
        pytest.fail("merge ran after repository protection changed")

    monkeypatch.setattr("tools.delivery_action._assert_merge_protection", protection)
    monkeypatch.setattr("tools.delivery_action._gh_json", gh_json)
    monkeypatch.setattr("tools.delivery_action._gh", gh)
    policy = build_delivery_policy(
        "merger", {"repository": "owner/repo", "pull_request": 17, "exact_sha": SHA},
        trusted_runtime=RUNTIME,
    )
    with delivery_role_context(policy):
        result = json.loads(delivery_action_handler({"action": "merge"}))
    assert result["ok"] is False
    assert result["error"] == "repository protection changed"
    assert protection_calls == 2
    assert all("merge" not in argv for argv in commands)


@pytest.mark.parametrize(
    ("details", "accepted"),
    [
        ({"enforcement": "active", "bypass_actors": []}, True),
        ({"enforcement": "active"}, False),
        ({"enforcement": "active", "bypass_actors": [{"actor_id": 1}]}, False),
    ],
)
def test_ruleset_protection_requires_strict_checks_and_explicitly_no_bypass(
    monkeypatch, details: dict, accepted: bool,
) -> None:
    from tools.delivery_action import _assert_merge_protection

    rules = [
        {
            "type": "pull_request", "ruleset_id": 1,
            "parameters": {
                "required_approving_review_count": 1,
                "dismiss_stale_reviews_on_push": True,
                "require_last_push_approval": True,
            },
        },
        {
            "type": "required_status_checks", "ruleset_id": 1,
            "parameters": {
                "required_status_checks": [{"context": "test"}],
                "strict_required_status_checks_policy": True,
            },
        },
    ]

    def gh_json(endpoint: str, **_kwargs):
        if endpoint.endswith("/protection"):
            raise DeliveryActionError("classic protection absent")
        if "/rules/branches/" in endpoint:
            return rules
        if endpoint.endswith("/rulesets/1"):
            return details
        raise AssertionError(endpoint)

    monkeypatch.setattr("tools.delivery_action._gh_json", gh_json)
    if accepted:
        assert _assert_merge_protection("owner/repo", "main") == {
            "kind": "ruleset", "rulesets": 1,
        }
    else:
        with pytest.raises(DeliveryActionError, match="does not provably enforce"):
            _assert_merge_protection("owner/repo", "main")


def test_ruleset_protection_rejects_non_strict_required_checks(monkeypatch) -> None:
    from tools.delivery_action import _assert_merge_protection

    rules = [
        {
            "type": "pull_request", "ruleset_id": 1,
            "parameters": {
                "required_approving_review_count": 1,
                "dismiss_stale_reviews_on_push": True,
                "require_last_push_approval": True,
            },
        },
        {
            "type": "required_status_checks", "ruleset_id": 1,
            "parameters": {"required_status_checks": [{"context": "test"}]},
        },
    ]

    def gh_json(endpoint: str, **_kwargs):
        if endpoint.endswith("/protection"):
            raise DeliveryActionError("classic protection absent")
        if "/rules/branches/" in endpoint:
            return rules
        raise AssertionError(endpoint)

    monkeypatch.setattr("tools.delivery_action._gh_json", gh_json)
    with pytest.raises(DeliveryActionError, match="does not provably enforce"):
        _assert_merge_protection("owner/repo", "main")


def test_bounded_runner_terminates_on_aggregate_output_overflow() -> None:
    with pytest.raises(DeliveryRuntimeError, match="output limit"):
        run_bounded(
            [sys.executable, "-c", "import os; os.write(1,b'x'*4096); os.write(2,b'y'*4096)"],
            timeout=10, output_limit=1024,
        )


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_bounded_runner_terminates_pipe_holding_child_after_parent_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    script = f"sleep 30 & echo $! > {shlex.quote(str(pid_file))}"
    with pytest.raises(DeliveryRuntimeError, match="timed out"):
        run_bounded(["/bin/sh", "-c", script], timeout=1, output_limit=1024)
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not psutil.pid_exists(child_pid):
            break
        try:
            if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    try:
        final_status = psutil.Process(child_pid).status()
    except psutil.NoSuchProcess:
        final_status = "gone"
    assert final_status in {"gone", psutil.STATUS_ZOMBIE}


def test_exact_sha_archive_quota_is_enforced(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    sha = _init_repo(repo)
    export = tmp_path / "export"
    export.mkdir()
    monkeypatch.setattr("tools.delivery_action._ARCHIVE_MAX_BYTES", 16)
    with pytest.raises(DeliveryActionError, match="output limit"):
        _exact_sha_source(build_delivery_policy("reviewer", {
            "repository": "owner/repo", "pull_request": 1, "exact_sha": sha,
        }, trusted_runtime=RUNTIME).bound_to_workspace(repo), sha, export)


def test_archive_extracted_size_quota_is_enforced(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo("large.bin")
        info.size = 32
        bundle.addfile(info, io.BytesIO(b"x" * 32))
    destination = tmp_path / "checkout"
    destination.mkdir()
    monkeypatch.setattr("tools.delivery_action._EXTRACTED_MAX_BYTES", 16)
    with pytest.raises(DeliveryActionError, match="extracted-size quota"):
        _extract_git_archive(archive, destination)


def test_low_disk_fails_closed_without_writing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "tools.delivery_action_runtime.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )
    with pytest.raises(DeliveryRuntimeError, match="insufficient free disk"):
        require_free_disk(tmp_path, 2, reserve=1)


def test_stale_temp_and_container_cleanup_reaps_interrupted_runs(monkeypatch, tmp_path: Path) -> None:
    stale = tmp_path / "hermes-delivery-review-999999-dead"
    stale.mkdir()
    removed: list[list[str]] = []
    monkeypatch.setattr("tools.delivery_action.tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("tools.delivery_action._pid_is_live", lambda _pid: False)

    def run(argv, **_kwargs):
        if argv[1] == "ps":
            return subprocess.CompletedProcess(argv, 0, "container-id 999999\n", "")
        removed.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("tools.delivery_action._run", run)
    _cleanup_stale_verification_state("/usr/bin/docker")
    assert not stale.exists()
    assert removed == [["/usr/bin/docker", "rm", "--force", "container-id"]]


def test_gitlink_fails_closed_instead_of_materializing_empty_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base_sha = _init_repo(repo)
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"160000,{base_sha},submodule"],
        cwd=repo, check=True,
    )
    subprocess.run(["git", "commit", "-m", "gitlink"], cwd=repo, check=True, capture_output=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    export = tmp_path / "export"
    export.mkdir()
    policy = build_delivery_policy(
        "reviewer", {"repository": "owner/repo", "pull_request": 1, "exact_sha": sha},
        trusted_runtime=RUNTIME,
    ).bound_to_workspace(repo)
    with pytest.raises(DeliveryActionError, match="does not support git submodules"):
        _exact_sha_source(policy, sha, export)


def test_live_workspace_gitlink_fails_closed_before_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base_sha = _init_repo(repo)
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"160000,{base_sha},submodule"],
        cwd=repo, check=True,
    )
    export = tmp_path / "export"
    export.mkdir()
    policy = build_delivery_policy("implementer", trusted_runtime=RUNTIME).bound_to_workspace(repo)
    with pytest.raises(DeliveryActionError, match="does not support git submodules"):
        _workspace_source(policy, export)


def test_closure_rechecks_containment_after_acceptance(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    sha = _init_repo(repo, remote=True)
    policy = _closure_policy(repo, sha)
    comparisons = iter([{"status": "ahead"}, {"status": "behind"}])

    def gh_json(endpoint: str, **_kwargs):
        if endpoint == "repos/owner/repo":
            return {"default_branch": "main"}
        if "/compare/" in endpoint:
            return next(comparisons)
        pytest.fail("issue queried after containment changed")

    monkeypatch.setattr("tools.delivery_action._gh_json", gh_json)
    monkeypatch.setattr(
        "tools.delivery_action._docker_verify",
        lambda *_args, **_kwargs: json.dumps({"ok": True, "attestation": {"image": IMAGE}}),
    )
    monkeypatch.setattr("tools.delivery_action._gh", lambda *_args: pytest.fail("issue closed"))
    with delivery_role_context(policy):
        result = json.loads(delivery_action_handler({"action": "close_issue"}))
    assert result["ok"] is False
    assert "not present" in result["error"]


def test_closed_issue_retry_is_explicitly_idempotent_after_fresh_acceptance(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    sha = _init_repo(repo, remote=True)
    policy = _closure_policy(repo, sha)
    assert policy.acceptance_recipe is not None
    comparisons = 0

    def gh_json(endpoint: str, **_kwargs):
        nonlocal comparisons
        if endpoint == "repos/owner/repo":
            return {"default_branch": "main"}
        if "/compare/" in endpoint:
            comparisons += 1
            return {"status": "identical"}
        if endpoint.endswith("/issues/18"):
            return {"state": "closed"}
        raise AssertionError(endpoint)

    monkeypatch.setattr("tools.delivery_action._gh_json", gh_json)
    monkeypatch.setattr(
        "tools.delivery_action._docker_verify",
        lambda *_args, **_kwargs: json.dumps({
            "ok": True,
            "attestation": {"image": IMAGE, "recipe_identity": policy.acceptance_recipe.identity},
        }),
    )
    monkeypatch.setattr("tools.delivery_action._gh", lambda *_args: pytest.fail("closed issue was reclosed"))
    with delivery_role_context(policy):
        result = json.loads(delivery_action_handler({"action": "close_issue"}))
    assert result["ok"] is True
    assert result["already_complete"] is True
    assert result["acceptance"]["image"] == IMAGE
    assert comparisons == 2
