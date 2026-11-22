"""opencode_task: governance mirrors terminal, OpenCode stays confined."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml
from hermes_cli.dashboard_governance.context import DashboardGovernanceContext, governance_context
from hermes_cli.dashboard_governance.loader import load_governance_policy
from hermes_cli.dashboard_governance.models import GovernanceSubject
from hermes_cli.dashboard_governance.resolver import resolve_effective_access
from hermes_cli.dashboard_governance.tool_policy import decide_tool_access, decide_tool_argument_access

from tools import opencode_task_runner as runner
from tools import opencode_tool

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "deploy" / "opencode" / "opencode-wrapper.sh"
FAKE_REGISTRY = SimpleNamespace(get_entry=lambda name: SimpleNamespace(toolset="terminal"),
                                get_toolset_for_tool=lambda name: "terminal")


def _access(tmp_path, user: dict, role: dict, email="dev@example.test"):
    policy = {"mode": "enforce", "roles": {"member": {"grants": role}},
              "users": {email: {"roles": ["member"], **user}}}
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy))
    subject = GovernanceSubject(email=email)
    return resolve_effective_access(load_governance_policy(path=path), subject), subject, path


def _blacklist(tmp_path, workroot: Path, **extra_user):
    role = {"permissions": ["chat:use", "terminal:use"], "profiles": ["default"],
            "tools": {"builtins": ["terminal"], "toolsets": ["terminal", "delegation"]},
            "cli": {"commands": ["*"], "workdir_roots": [str(workroot)]},
            "files": {"read_roots": [str(workroot)], "write_roots": [str(workroot)]}}
    user = {"access_level": "elevated", "access_mode": "blacklist", **extra_user}
    return _access(tmp_path, user, role)


def _whitelist_with_opencode(tmp_path, workroot: Path, **extra_user):
    role = {"permissions": ["chat:use", "terminal:use"], "profiles": ["default"],
            "tools": {"builtins": ["terminal"], "toolsets": ["terminal", "delegation"]},
            "cli": {"commands": ["opencode", "gh", "pytest"], "workdir_roots": [str(workroot)]},
            "files": {"read_roots": [str(workroot)], "write_roots": [str(workroot)]}}
    return _access(tmp_path, {"access_level": "elevated", **extra_user}, role)


def _whitelist_without_opencode(tmp_path, workroot: Path):
    role = {"permissions": ["chat:use", "terminal:use"], "profiles": ["default"],
            "tools": {"builtins": ["terminal"], "toolsets": ["terminal", "delegation"]},
            "cli": {"commands": ["gmail", "gh"], "workdir_roots": [str(workroot)]},
            "files": {"read_roots": [str(workroot)], "write_roots": [str(workroot)]}}
    return _access(tmp_path, {"access_level": "elevated"}, role)


@pytest.fixture
def workroot(tmp_path):
    root = tmp_path / "work"
    (root / "repo").mkdir(parents=True)
    return root


@pytest.mark.parametrize("maker", [_blacklist, _whitelist_with_opencode])
def test_allowed_workdir_passes_governance(tmp_path, workroot, maker):
    access, _, _ = maker(tmp_path, workroot)
    assert decide_tool_access(access, "opencode_task", FAKE_REGISTRY).allowed
    decision = decide_tool_argument_access(
        access, "opencode_task", {"task": "add a unit test", "workdir": str(workroot / "repo")})
    assert decision.allowed, decision


def test_workdir_outside_roots_is_denied(tmp_path, workroot):
    # Blacklist accounts have no workdir roots, exactly like terminal; the roots
    # bind whitelist accounts.
    access, _, _ = _whitelist_with_opencode(tmp_path, workroot)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    decision = decide_tool_argument_access(access, "opencode_task", {"task": "x", "workdir": str(outside)})
    assert not decision.allowed
    assert decision.reason in {"cli_workdir_not_allowed", "file_write_root_not_allowed", "file_root_not_allowed"}


def test_relative_workdir_is_denied(tmp_path, workroot):
    access, _, _ = _blacklist(tmp_path, workroot)
    decision = decide_tool_argument_access(access, "opencode_task", {"task": "x", "workdir": "repo"})
    assert not decision.allowed and decision.reason == "opencode_workdir_required"


def test_user_without_opencode_cli_grant_is_refused(tmp_path, workroot):
    access, _, _ = _whitelist_without_opencode(tmp_path, workroot)
    assert decide_tool_access(access, "terminal", FAKE_REGISTRY).allowed  # terminal itself is fine
    tool = decide_tool_access(access, "opencode_task", FAKE_REGISTRY)
    assert not tool.allowed and tool.reason == "opencode_cli_not_allowed"
    args = decide_tool_argument_access(access, "opencode_task", {"task": "x", "workdir": str(workroot / "repo")})
    assert not args.allowed and args.reason == "opencode_cli_not_allowed"


def test_explicit_cli_deny_blocks_blacklist_account(tmp_path, workroot):
    access, _, _ = _blacklist(tmp_path, workroot, deny={"cli": {"commands": ["opencode"]}})
    assert not decide_tool_access(access, "opencode_task", FAKE_REGISTRY).allowed


def test_task_text_naming_a_denied_path_is_refused(tmp_path, workroot):
    private = workroot / "private"
    access, _, _ = _blacklist(tmp_path, workroot, grants={"files": {"denied_globs": [str(private), str(private) + "/**"]}})
    decision = decide_tool_argument_access(
        access, "opencode_task", {"task": f"copy {private}/notes.txt into the repo", "workdir": str(workroot / "repo")})
    assert not decision.allowed
    access, _, _ = _whitelist_with_opencode(tmp_path, workroot, deny={"files": {"denied_globs": [str(private) + "/**"]}})
    decision = decide_tool_argument_access(
        access, "opencode_task", {"task": f"read {private}/notes.txt", "workdir": str(workroot / "repo")})
    assert not decision.allowed


def test_test_command_is_checked_like_terminal(tmp_path, workroot):
    access, _, _ = _whitelist_without_opencode(tmp_path, workroot)
    # whitelist account: opencode refused before the test command matters
    assert not decide_tool_argument_access(
        access, "opencode_task", {"task": "x", "workdir": str(workroot), "test_command": "rm -rf /"}).allowed
    access, _, _ = _blacklist(tmp_path, workroot, deny={"cli": {"commands": ["rm"]}})
    decision = decide_tool_argument_access(
        access, "opencode_task", {"task": "x", "workdir": str(workroot / "repo"), "test_command": "rm -rf build"})
    assert not decision.allowed


def _rules(cfg, action, effect="deny"):
    return [r["resource"] for r in cfg["permissions"] if r["action"] == action and r["effect"] == effect]


def test_owner_sees_keys_and_only_the_secret_mount_is_masked(tmp_path, workroot):
    dirs, files, patterns = opencode_tool.mask_plan(None)
    assert dirs == ["/run/synthpulse/secrets"] and files == []
    cfg = opencode_tool.opencode_config(patterns)
    assert cfg["permissions"][0] == {"action": "*", "resource": "*", "effect": "allow"}  # last match wins
    assert "*" in _rules(cfg, "external_directory")
    assert "*/.env*" not in _rules(cfg, "read")
    assert "*opencode*run*" in _rules(cfg, "shell")
    assert all(s.get("disabled") is True for s in cfg["mcp"]["servers"].values())  # no MCP server active
    assert cfg["providers"]["omniroute"]["settings"]["apiKey"] == "{env:OMNIROUTE_API_KEY}"
    assert cfg["providers"]["omniroute"]["package"] == "aisdk:@ai-sdk/openai-compatible"
    assert cfg["update"] == "disable" and "permission" not in cfg and "provider" not in cfg


def test_caller_denied_paths_are_masked(tmp_path, workroot):
    creds = os.path.join(str(opencode_tool._host_home()), ".config/synthwave/clients")
    access, _, _ = _blacklist(tmp_path, workroot,
                              grants={"files": {"denied_globs": [creds, creds + "/**"]}})
    dirs, _, patterns = opencode_tool.mask_plan(access)
    assert creds in dirs and creds + "/**" in patterns
    cfg = opencode_tool.opencode_config(patterns)
    assert creds + "/*" in _rules(cfg, "read") and creds + "/*" in _rules(cfg, "edit")
    assert "*" + creds.lstrip("/") + "*" in _rules(cfg, "shell")
    (tmp_path / "b").mkdir()
    allowed, _, _ = _blacklist(tmp_path / "b", workroot)
    dirs_ok, _, _ = opencode_tool.mask_plan(allowed)
    assert creds not in dirs_ok


def test_deny_globs_become_opencode2_resources_relative_and_absolute():
    v = opencode_tool._path_variants
    assert v("/w/repo/secret/**", "/w/repo") == ["/w/repo/secret/*", "secret/*"]
    assert v("**/.ssh/**", "/w/repo") == ["*/.ssh/*", ".ssh/*"]
    assert v("/other/x/**", "/w/repo") == ["/other/x/*"]
    cfg = opencode_tool.opencode_config(["/w/repo/secret/**"], workdir="/w/repo")
    assert cfg["permissions"].index({"action": "read", "resource": "secret/*", "effect": "deny"}) > 0
    assert {"action": "skill", "resource": "*", "effect": "deny"} in cfg["permissions"]
    assert {"action": "skill", "resource": "*", "effect": "deny"} not in opencode_tool.opencode_config([], lean=False)["permissions"]


def test_runner_reports_only_files_this_job_changed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    git("init", "-q")
    (repo / "a.py").write_text("a = 1\n")
    (repo / "b.py").write_text("b = 1\n")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (repo / "b.py").write_text("b = 2\n")  # dirty before the job, untouched by it
    before = runner.snapshot(repo, str(repo))
    (repo / "a.py").write_text("a = 2\n")
    (repo / "new.py").write_text("n = 1\n")
    after = runner.snapshot(repo, str(repo))
    changed = runner.changed_files(before, after)
    assert changed == ["a.py", "new.py"]
    patch = runner.collect_git({"top": str(repo)}, repo, changed)
    assert "+a = 2" in patch and "+n = 1" in patch and "b = 2" not in patch


def test_slot_limit(tmp_path):
    held = [runner.acquire_slot(tmp_path, 2, 0) for _ in range(2)]
    assert all(held)
    assert runner.acquire_slot(tmp_path, 2, 0) is None
    held[0].close()
    again = runner.acquire_slot(tmp_path, 2, 0)
    assert again is not None
    again.close()
    held[1].close()


def test_tool_refuses_inside_opencode(monkeypatch, workroot):
    monkeypatch.setenv("OPENCODE_RECURSION_GUARD", "1")
    out = json.loads(opencode_tool.opencode_task(task="x", workdir=str(workroot / "repo")))
    assert "recursion guard" in json.dumps(out)


def test_runner_refuses_inside_opencode(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    (run_dir / "job.json").write_text(json.dumps({"id": "t", "task": "x"}))
    (run_dir / "omniroute.key").write_text("k")
    monkeypatch.setenv("OPENCODE_RECURSION_GUARD", "1")
    assert runner.main(str(run_dir)) == 2
    assert "recursion guard" in capsys.readouterr().out
    assert not (run_dir / "omniroute.key").exists()  # one-shot key is gone either way


def test_tool_blocks_governed_caller_outside_scope(tmp_path, workroot):
    access, subject, path = _whitelist_with_opencode(tmp_path, workroot)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ctx = DashboardGovernanceContext(subject=subject, access=access, active_profile="default",
                                     session_id="oc-test", approval_policy_path=str(path))
    with governance_context(ctx):
        out = json.loads(opencode_tool.opencode_task(task="x", workdir=str(outside)))
    assert "Blocked by governance" in json.dumps(out)


def test_tool_blocks_non_granted_caller(tmp_path, workroot):
    access, subject, path = _whitelist_without_opencode(tmp_path, workroot)
    ctx = DashboardGovernanceContext(subject=subject, access=access, active_profile="default",
                                     session_id="oc-test", approval_policy_path=str(path))
    with governance_context(ctx):
        out = json.loads(opencode_tool.opencode_task(task="x", workdir=str(workroot / "repo")))
    assert "opencode_cli_not_allowed" in json.dumps(out)


def _wrapper_copy(tmp_path, slots=1):
    fake = tmp_path / "fake-opencode"
    fake.write_text("#!/usr/bin/env bash\necho \"ARGS:$*\"\necho \"KEY:${OMNIROUTE_API_KEY:-}\"\n"
                    'if [ -n "${READY:-}" ]; then touch "$READY"; fi\n'
                    'if [ -n "${RELEASE:-}" ]; then while [ ! -f "$RELEASE" ]; do sleep 0.01; done; fi\nexit 0\n')
    fake.chmod(0o755)
    keyfile = tmp_path / "omniroute.key"
    keyfile.write_text("secret-key\n")
    keyfile.chmod(0o600)
    text = WRAPPER.read_text()
    text = text.replace("__OPENCODE_REAL_BIN__", str(fake)).replace("__OPENCODE_SLOT_DIR__", str(tmp_path / "slots"))
    text = text.replace("__OPENCODE_KEY_FILE__", str(keyfile)).replace('OPENCODE_MAX_SLOTS="${OPENCODE_MAX_SLOTS:-12}"', f"OPENCODE_MAX_SLOTS={slots}")
    wrapper = tmp_path / "opencode"
    wrapper.write_text(text)
    wrapper.chmod(0o755)
    return wrapper


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock missing")
def test_wrapper_guard_default_model_key_and_slots(tmp_path):
    wrapper = _wrapper_copy(tmp_path, slots=1)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    done = subprocess.run([str(wrapper), "run", "hello"], env=env, capture_output=True, text=True, timeout=20, check=False)
    assert done.returncode == 0, done.stderr
    assert "-m omniroute/synthwave-auto" in done.stdout
    assert "KEY:secret-key" in done.stdout
    explicit = subprocess.run([str(wrapper), "run", "-m", "omniroute/auto/coding", "hi"], env=env,
                              capture_output=True, text=True, timeout=20, check=False)
    assert "synthwave-auto" not in explicit.stdout
    nested = subprocess.run([str(wrapper), "run", "x"], env={**env, "OPENCODE_RECURSION_GUARD": "1"},
                            capture_output=True, text=True, timeout=20, check=False)
    assert nested.returncode == 1 and "recursion guard" in nested.stderr
    ready, release = tmp_path / "ready", tmp_path / "release"
    busy = subprocess.Popen([str(wrapper), "run", "long"],
                            env={**env, "READY": str(ready), "RELEASE": str(release)},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        from tests.waiting import wait_until
        wait_until(ready.exists, timeout=10, what="first OpenCode process holds the slot")
        second = subprocess.run([str(wrapper), "run", "x"], env={**env, "OPENCODE_SLOT_WAIT": "0"},
                                capture_output=True, text=True, timeout=20, check=False)
        assert second.returncode == 75 and "slots" in second.stderr
    finally:
        release.touch()
        busy.communicate(timeout=20)


def test_whitelist_gets_allowlist_home(tmp_path, workroot):
    access, _, _ = _whitelist_with_opencode(tmp_path, workroot)
    layout = opencode_tool.home_layout(access)
    assert layout["home_mode"] == "allowlist" and str(workroot) in layout["read_roots"]
    access, _, _ = _blacklist(tmp_path, workroot)
    assert opencode_tool.home_layout(access) == {"home_mode": "masked"}
    assert opencode_tool.home_layout(None) == {"home_mode": "masked"}


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap missing")
def test_sandbox_hides_unlisted_home_and_blocks_outside_writes(tmp_path):
    home = tmp_path / "home"
    allowed = home / "allowed"
    hidden = home / "hidden"
    for d in (allowed, hidden):
        d.mkdir(parents=True)
        (d / "f.txt").write_text("x")
    worktree = allowed / "wt"
    worktree.mkdir()
    job = {"home": str(home), "home_mode": "allowlist", "read_roots": [str(allowed)], "runtime_paths": [],
           "mask_dirs": [], "mask_files": []}
    argv = runner.bwrap_argv(job, tmp_path / "run", worktree, {"git": False})
    def sh(cmd):
        return subprocess.run(argv + ["bash", "-c", cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False)
    assert sh(f"cat {allowed}/f.txt").stdout == "x"
    assert sh(f"cat {hidden}/f.txt").returncode != 0
    assert sh(f"touch {allowed}/new.txt").returncode != 0
    assert sh(f"touch {worktree}/ok.txt").returncode == 0


def test_runner_worktree_mode_leaves_live_tree_alone(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout
    git("init", "-q")
    (repo / "a.py").write_text("a = 1\n")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    job = {"git_top": str(repo), "workdir": str(repo), "branch": "opencode/t1",
           "worktree": str(repo / ".opencode-worktrees" / "t1")}
    workroot, info = runner.prepare_worktree(job)
    before = runner.snapshot(workroot, info["top"])
    (workroot / "a.py").write_text("a = 2\n")
    changed = runner.changed_files(before, runner.snapshot(workroot, info["top"]))
    assert changed == ["a.py"]
    assert "+a = 2" in runner.collect_git(info, workroot, changed)
    runner.commit_worktree(info, "bump a")
    assert (repo / "a.py").read_text() == "a = 1\n"
    assert "a = 2" in git("show", "opencode/t1:a.py")
    argv = runner.bwrap_argv({"home": str(tmp_path), "deny_patterns": []}, tmp_path / "run", workroot, info)
    assert str(repo / ".git" / "objects") in argv and str(repo) not in [argv[i + 1] for i, x in enumerate(argv) if x == "--bind" and argv[i + 1] == str(repo)]


def test_directory_denies_do_not_mask_unrelated_project_files(tmp_path):
    paths = ['app/collector.py', 'requirements.txt', '.ssh/id_test',
             'nested/.ssh/id_test', '.config/bunq/config.json', 'nested/private-sa.json']
    for rel in paths:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('synthetic fixture')
    denied = set(runner._denied_files_under(
        tmp_path, ['**/.ssh/**', '**/.config/bunq/**', '**/*-sa.json']))
    assert denied == {tmp_path / rel for rel in paths[2:]}


# ── 07-11-2026: ideas from dsh-superagent ─────────────────────────────────────

FAKE_OPENCODE_DB = r'''#!/usr/bin/env python3
"""Fake OpenCode 2 `run --format json`: fails the model call once, then finishes on --session."""
import json, os, sys
args = sys.argv[1:]
log = os.path.join(os.environ["XDG_DATA_HOME"], "calls.log")
os.makedirs(os.path.dirname(log), exist_ok=True)
with open(log, "a") as fh:
    fh.write(json.dumps(args) + "\n")
def emit(**event):
    print(json.dumps(event))
if "--session" in args:
    sid = args[args.index("--session") + 1]
    emit(type="step_start", sessionID=sid, part={"messageID": "m2", "type": "step-start"})
    emit(type="text", sessionID=sid, part={"messageID": "m2", "type": "text",
         "text": "Changed a.txt, verified with cat.\nSTATUS: blocked\nQUESTION: Keep the old API?"})
    emit(type="step_finish", sessionID=sid, part={"messageID": "m2", "type": "step-finish", "reason": "stop"})
    open("a.txt", "w").write("done\n")
else:
    emit(type="step_start", sessionID="ses_test1", part={"messageID": "m1", "type": "step-start"})
    emit(type="error", sessionID="ses_test1",
         error={"type": "provider.error", "message": "Maximum combo retry limit reached"})
    sys.exit(1)
'''


def test_parse_status_and_transient_errors():
    assert runner.parse_status("did x\nSTATUS: ready\nQUESTION: none") == ("ready", "")
    assert runner.parse_status("**STATUS:** blocked\nQUESTION: Which DB?") == ("blocked", "Which DB?")
    assert runner.parse_status("no contract here") == ("", "")
    assert runner.is_transient("Maximum combo retry limit reached")
    assert runner.is_transient("[codex/x] [503]: busy")
    assert not runner.is_transient("SyntaxError in a.py")
    assert not runner.is_transient("")


@pytest.mark.skipif(not shutil.which("bwrap"), reason="needs bubblewrap")
def test_runner_resumes_same_session_after_gateway_error(tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    work.mkdir()
    fake = tmp_path / "fake-opencode"
    fake.write_text(FAKE_OPENCODE_DB)
    fake.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    job = {"id": "j1", "task": "make a.txt", "workdir": str(work), "model": "m/x", "opencode_bin": str(fake),
           "home": str(tmp_path), "slot_dir": str(tmp_path / "slots"), "slots": 2, "slot_wait": 5,
           "timeout": 600, "test_command": "", "mask_dirs": [], "mask_files": [], "deny_patterns": [],
           "opencode_config": {}, "path": os.environ.get("PATH", "/usr/bin:/bin")}
    (run_dir / "job.json").write_text(json.dumps(job))
    monkeypatch.setattr(runner, "RETRY_BACKOFF", (0, 0, 0))
    monkeypatch.delenv("OPENCODE_RECURSION_GUARD", raising=False)
    assert runner.main(str(run_dir)) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[2] == "STATUS: blocked"
    assert "QUESTION: Keep the old API?" in out
    assert "retry 1" in out and "a.txt" in out
    calls = [json.loads(l) for l in (run_dir / "state" / "data" / "calls.log").read_text().splitlines()]
    assert len(calls) == 2 and "--session" not in calls[0] and calls[1][calls[1].index("--session") + 1] == "ses_test1"
    assert calls[0][:6] == ["run", "--standalone", "--auto", "--format", "json", "-m"] and "--dir" not in calls[0]
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "blocked" and result["retries"] == 1 and result["session"] == "ses_test1"
    assert (run_dir / "report.md").read_text().startswith("# opencode_task j1")


def test_handoff_contract_is_appended_and_continue_job_checks_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(opencode_tool, "_job_root", lambda: tmp_path)
    (tmp_path / "j0").mkdir()
    (tmp_path / "j0" / "job.json").write_text(json.dumps({"id": "j0", "owner_email": "a@x", "workdir": "/w"}))
    assert "someone else" in opencode_tool.load_previous_job("j0", "b@x")[1]
    assert "no OpenCode session" in opencode_tool.load_previous_job("j0", "a@x")[1]
    (tmp_path / "j0" / "state").mkdir()
    (tmp_path / "j0" / "result.json").write_text(json.dumps({"session": "ses_x"}))
    prev, err = opencode_tool.load_previous_job("j0", "a@x")
    assert err is None and prev["_session"] == "ses_x" and prev["_state_dir"] == str(tmp_path / "j0" / "state")
    assert "earlier" in opencode_tool.load_previous_job("../etc", "a@x")[1]
    assert "no earlier job" in opencode_tool.load_previous_job("nope", "a@x")[1]
    assert "STATUS: ready | blocked | failed | dropped" in opencode_tool.HANDOFF_CONTRACT


def test_lean_jobs_switch_off_host_mcp_servers_and_external_skills(tmp_path, monkeypatch):
    cfg = tmp_path / ".config" / "opencode"
    cfg.mkdir(parents=True)
    (cfg / "opencode.jsonc").write_text('{\n  // comment\n  "mcp": {"mailchimp": {"type": "local", "command": ["m"]},'
                                        ' "ragflow": {"type": "remote", "url": "http://r"}, "old": {}}\n}')
    monkeypatch.setattr(opencode_tool, "_host_home", lambda: tmp_path)
    assert opencode_tool.opencode_config([], lean=True)["mcp"]["servers"] == {
        "mailchimp": {"type": "local", "command": ["m"], "disabled": True},
        "ragflow": {"type": "remote", "url": "http://r", "disabled": True}}
    assert opencode_tool.opencode_config([], lean=False)["mcp"]["servers"] == {}
    job = {"home": str(tmp_path), "opencode_config": {}}
    env = runner.sandbox_env({**job, "lean": True}, tmp_path / "run", "k")
    assert env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1" and env["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] == "1"
    assert "OPENCODE_DISABLE_EXTERNAL_SKILLS" not in runner.sandbox_env({**job, "lean": False}, tmp_path / "run", "k")
    assert opencode_tool.wants_lean(None, None) and not opencode_tool.wants_lean(True, None)
    assert not opencode_tool.wants_lean(None, {"lean": False})


class _Access:
    def __init__(self, allowed: bool):
        self._allowed = allowed

    def allows(self, dimension, value, *, path=False):
        return self._allowed and dimension == "mcp_servers" and value == "plainboard"


def test_plainboard_token_only_with_the_grant_and_from_the_profile_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OMNIROUTE_API_KEY=x\nPLAINBOARD_MCP_TOKEN=pbb_partner\n")
    monkeypatch.setattr(opencode_tool, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("PLAINBOARD_MCP_TOKEN", "pbb_process_env_must_not_leak")
    assert opencode_tool.plainboard_token(_Access(True)) == "pbb_partner"
    assert opencode_tool.plainboard_token(_Access(False)) == ""
    (tmp_path / ".env").write_text("OMNIROUTE_API_KEY=x\n")
    assert opencode_tool.plainboard_token(_Access(True)) == ""


def test_plainboard_mcp_stays_on_when_lean_and_is_off_without_token(monkeypatch):
    host = {"plainboard": {"type": "remote", "url": "http://127.0.0.1:18156/mcp"}, "twenty": {"type": "remote", "url": "http://t"}}
    monkeypatch.setattr(opencode_tool, "host_mcp_servers", lambda: host)
    on = opencode_tool.opencode_config([], lean=True, plainboard=True)["mcp"]["servers"]
    assert on["twenty"]["disabled"] is True
    assert on["plainboard"]["type"] == "remote" and not on["plainboard"].get("disabled")
    assert on["plainboard"]["headers"]["Authorization"] == "Bearer {env:PLAINBOARD_MCP_TOKEN}"
    assert opencode_tool.opencode_config([], lean=True, plainboard=False)["mcp"]["servers"]["plainboard"]["disabled"] is True
    assert opencode_tool.opencode_config([], lean=False, plainboard=False)["mcp"]["servers"] == {
        "plainboard": {"type": "remote", "url": opencode_tool.PLAINBOARD_MCP_URL, "disabled": True}}
    monkeypatch.setattr(opencode_tool, "host_mcp_servers", lambda: {"twenty": host["twenty"]})
    assert opencode_tool.opencode_config([], lean=False, plainboard=False)["mcp"]["servers"] == {}
    models = opencode_tool.opencode_config([])["providers"]["omniroute"]["models"]
    assert "synthwave-auto" in models and opencode_tool.DEFAULT_MODEL == "omniroute/synthwave-auto"


def test_stream_facts_and_readable_output():
    out = "\n".join([
        "some stderr line",
        json.dumps({"type": "step_start", "sessionID": "ses_1", "part": {"messageID": "a"}}),
        json.dumps({"type": "tool_use", "sessionID": "ses_1", "part": {"tool": "shell", "messageID": "a",
                    "state": {"status": "completed", "input": {"command": "ls"}}}}),
        json.dumps({"type": "step_finish", "sessionID": "ses_1", "part": {"messageID": "a"}}),
        json.dumps({"type": "text", "sessionID": "ses_1", "part": {"messageID": "b", "text": "Done.\nSTATUS: ready"}}),
        json.dumps({"type": "step_finish", "sessionID": "ses_1", "part": {"messageID": "b"}})])
    facts = runner.stream_facts(out)
    assert facts == {"session": "ses_1", "error": "", "text": "Done.\nSTATUS: ready", "steps": 2}
    assert "[shell]" in runner.readable_output(out) and "STATUS: ready" in runner.readable_output(out)
    err = json.dumps({"type": "error", "sessionID": "ses_2", "error": {"type": "provider.no-route", "message": "Model unavailable"}})
    assert runner.stream_facts(err)["error"] == "Model unavailable" and runner.stream_facts(err)["session"] == "ses_2"


def test_runner_passes_the_plainboard_token_once(tmp_path):
    (tmp_path / "plainboard.token").write_text("pbb_caller\n")
    job = {"home": str(tmp_path), "opencode_config": {}, "lean": True}
    env = runner.sandbox_env(job, tmp_path, "k")
    assert env["PLAINBOARD_MCP_TOKEN"] == "pbb_caller"
    assert not (tmp_path / "plainboard.token").exists()
    assert "PLAINBOARD_MCP_TOKEN" not in runner.sandbox_env(job, tmp_path, "k")
