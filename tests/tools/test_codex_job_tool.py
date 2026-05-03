"""Tests for tools/codex_job_tool.py."""

import json
import subprocess


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _make_repo(tmp_path):
    repo = tmp_path / "FocusLock"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test User"], check=True)
    (repo / "README.md").write_text("# FocusLock\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo


def test_start_worktree_creates_codex_shaped_worktree_without_launching(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import codex_job_tool

    result = json.loads(codex_job_tool({
        "action": "start",
        "title": "Fix onboarding bug",
        "prompt": "Inspect only; do not modify files.",
        "repo_path": str(repo),
        "workspace_mode": "worktree",
        "launch": False,
        "discord": False,
    }))

    assert result["success"] is True
    assert result["workspace_mode"] == "worktree"
    assert result["job_id"]
    assert result["worktree_path"].startswith(str(tmp_path / ".codex" / "worktrees" / result["job_id"]))
    assert result["branch"] == f"codex/fix-onboarding-bug-{result['job_id']}"
    assert result["tmux_session"] == f"codex-{result['job_id']}"
    assert result["attach_command"] == f"tmux attach -t codex-{result['job_id']}"
    assert result["profile"] == "base-rich"
    assert result["profile_source"] == "built-in"
    assert result["sandbox"] == "workspace-write"
    assert result["search"] is False
    assert result["codex_profile"] is None
    assert "--search" not in result["codex_launch_flags"]
    assert _git(result["worktree_path"], "rev-parse", "--show-toplevel") == result["worktree_path"]
    assert _git(result["worktree_path"], "branch", "--show-current") == result["branch"]


def test_start_local_uses_existing_checkout_without_worktree(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import codex_job_tool

    result = json.loads(codex_job_tool({
        "action": "start",
        "title": "Local FocusLock task",
        "prompt": "Inspect only.",
        "repo_path": str(repo),
        "workspace_mode": "local",
        "launch": False,
        "discord": False,
    }))

    assert result["success"] is True
    assert result["workspace_mode"] == "local"
    assert result["worktree_path"] is None
    assert result["workspace_path"] == str(repo.resolve())
    assert result["branch"] == "main"


def test_review_profile_uses_readonly_launch_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import codex_job_tool

    result = json.loads(codex_job_tool({
        "action": "start",
        "title": "Review branch safely",
        "prompt": "Review only.",
        "repo_path": str(repo),
        "workspace_mode": "local",
        "profile": "review-rich-readonly",
        "launch": False,
        "discord": False,
    }))

    assert result["success"] is True
    assert result["profile"] == "review-rich-readonly"
    assert result["sandbox"] == "read-only"
    assert result["approval"] == "never"
    assert any("file mutation" in item for item in result["omitted_capabilities"])
    assert "-s read-only" in "\n".join(result["tmux_commands"])


def test_codex_profile_config_overrides_and_search_are_rendered(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import codex_job_tool

    result = json.loads(codex_job_tool({
        "action": "start",
        "title": "Web job",
        "prompt": "Build and verify.",
        "repo_path": str(repo),
        "workspace_mode": "local",
        "profile": "web-full",
        "codex_profile": "web",
        "codex_config_overrides": {
            "features.browser": True,
            "model_reasoning_effort": "high",
        },
        "launch": False,
        "discord": False,
    }))

    joined = "\n".join(result["tmux_commands"])
    assert result["success"] is True
    assert result["profile"] == "web-full"
    assert result["search"] is True
    assert result["codex_profile"] == "web"
    assert "-p web" in joined
    assert "--search" in joined
    assert "-c features.browser=true" in joined
    assert "model_reasoning_effort=" in joined
    assert "high" in joined
    assert "--search" in result["codex_launch_flags"]


def test_start_scratch_creates_documents_codex_git_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from tools.codex_job_tool import codex_job_tool

    result = json.loads(codex_job_tool({
        "action": "start",
        "title": "Research calorie app idea",
        "prompt": "Research only.",
        "workspace_mode": "scratch",
        "launch": False,
        "discord": False,
    }))

    assert result["success"] is True
    assert result["workspace_mode"] == "scratch"
    assert result["workspace_path"].startswith(str(tmp_path / "Documents" / "Codex"))
    assert result["branch"] == "main"
    assert _git(result["workspace_path"], "rev-parse", "--is-inside-work-tree") == "true"
    for name in ["inputs", "outputs", "scripts", "scratch"]:
        assert (tmp_path / "Documents" / "Codex").joinpath(*result["workspace_path"].split("/Documents/Codex/", 1)[1].split("/"), name).exists()


def test_build_tmux_commands_include_no_alt_screen_log_and_prompt(tmp_path):
    from tools.codex_job_tool import _build_tmux_commands

    workspace = tmp_path / "FocusLock"
    workspace.mkdir()
    log_path = tmp_path / "job.log"
    commands = _build_tmux_commands(
        session="codex-abcd",
        workspace_path=workspace,
        prompt="Do the task safely",
        log_path=log_path,
        model="gpt-5.5",
        approval="never",
        sandbox="workspace-write",
    )

    joined = "\n".join(commands)
    assert "tmux new-session -d -s codex-abcd" in joined
    assert "codex -C" in joined
    assert "--no-alt-screen" in joined
    assert "-m gpt-5.5" in joined
    assert "-a never" in joined
    assert "-s workspace-write" in joined
    assert "Do the task safely" in joined
    assert "tmux pipe-pane -o -t codex-abcd" in joined
    assert str(log_path) in joined


def test_codex_job_registration_does_not_gate_terminal_toolset():
    from tools.codex_job_tool import registry

    entry = registry.get_entry("codex_job")
    assert entry is not None
    assert entry.toolset == "terminal"
    assert entry.check_fn is None


def test_status_reads_job_record(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import codex_job_tool

    start = json.loads(codex_job_tool({
        "action": "start",
        "title": "Status test",
        "prompt": "Inspect only.",
        "repo_path": str(repo),
        "workspace_mode": "local",
        "launch": False,
        "discord": False,
    }))
    status = json.loads(codex_job_tool({"action": "status", "job_id": start["job_id"]}))

    assert status["success"] is True
    assert status["job"]["job_id"] == start["job_id"]
    assert status["job"]["workspace_path"] == str(repo.resolve())
    assert status["summary"]["profile"] == "base-rich"
    assert status["summary"]["model"] == "gpt-5.5"
    assert status["summary"]["effort"] == "xhigh"
    assert status["summary"]["workspace_path"] == str(repo.resolve())
    assert status["summary"]["tests"] == "not_run"
    assert status["tmux_alive"] is False


def test_monitor_status_surfaces_key_findings_and_workspace_changes(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "README.md").write_text("# FocusLock\n\nChanged\n", encoding="utf-8")

    from tools.codex_job_tool import _render_monitor_status

    job = {
        "job_id": "abcd",
        "title": "Critical bug hunt",
        "workspace_path": str(repo),
        "branch": "codex/critical-bug-abcd",
        "model": "gpt-5.5",
        "effort": "xhigh",
        "attach_command": "tmux attach -t codex-abcd",
    }
    output = """
• Explored
  └ Read HabitServices.swift

• I found one high-confidence Major bug to fix:

Bug: Free-tier habit access is only enforced at creation time.
Severity: Major. Existing over-limit habits remain user-visible after downgrade.
Root cause: PlanLimitEnforcer only has schedule/mode enforcement.
Planned fix: Apply habit limits at presentation and reminder boundaries.

• Ran swift test
"""

    message = _render_monitor_status(job, alive=True, output=output)

    assert "**Task summary / key findings**" in message
    assert "**Profile:** `base-rich`" in message
    assert "**Capabilities:**" in message
    assert "**Tests/verification:** Ran swift test" in message
    assert "Bug: Free-tier habit access" in message
    assert "Planned fix: Apply habit limits" in message
    assert "Root cause: PlanLimitEnforcer" in message
    assert "**Workspace changes**" in message
    assert "README.md" in message
    assert "**Recent useful activity**" in message
    assert len(message) <= 2000


def test_monitor_status_surfaces_generic_task_summary_for_non_bug_tasks(tmp_path):
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import _render_monitor_status

    job = {
        "job_id": "ef01",
        "title": "Build onboarding analytics",
        "workspace_path": str(repo),
        "branch": "codex/onboarding-analytics-ef01",
        "model": "gpt-5.5",
        "effort": "high",
        "attach_command": "tmux attach -t codex-ef01",
    }
    output = """
• Explored
  └ Read AnalyticsService.swift and OnboardingView.swift

Objective: Add privacy-safe onboarding funnel instrumentation.
Approach: Introduce an AnalyticsEvent enum and emit step-complete events from the view model.
Decision: Keep event payloads aggregate-only; no user-entered text.
Next step: Add tests for event emission and wire the sink into app services.

• Edited AnalyticsService.swift
"""

    message = _render_monitor_status(job, alive=True, output=output)

    assert "**Task summary / key findings**" in message
    assert "Objective: Add privacy-safe onboarding funnel instrumentation." in message
    assert "Approach: Introduce an AnalyticsEvent enum" in message
    assert "Decision: Keep event payloads aggregate-only" in message
    assert "Next step: Add tests for event emission" in message
    assert "**Bug / key findings**" not in message
    assert "**Recent useful activity**" in message


def test_start_defaults_completion_summary_to_discord_home_when_discord_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    import tools.codex_job_tool as tool
    monkeypatch.setattr(tool, "_send_message", lambda args: {"success": True, "message_id": "status-1", "chat_id": "123"})

    result = json.loads(tool.codex_job_tool({
        "action": "start",
        "title": "Discord summary default",
        "prompt": "Inspect only.",
        "repo_path": str(repo),
        "workspace_mode": "local",
        "launch": False,
        "discord": True,
        "discord_target": "discord:123:456",
    }))

    assert result["success"] is True
    assert result["summary_target"] == "discord"
    assert result["notify_on_completion"] is True


def test_start_can_disable_completion_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import codex_job_tool

    result = json.loads(codex_job_tool({
        "action": "start",
        "title": "No summary",
        "prompt": "Inspect only.",
        "repo_path": str(repo),
        "workspace_mode": "local",
        "launch": False,
        "discord": False,
        "summary_target": "local",
    }))

    assert result["success"] is True
    assert result["summary_target"] is None
    assert result["notify_on_completion"] is False


def test_completion_summary_is_generic_and_short(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "README.md").write_text("# FocusLock\n\nChanged\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fix: repair onboarding gate"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    from tools.codex_job_tool import _render_completion_summary

    job = {
        "job_id": "done1",
        "title": "Critical major bug fix",
        "workspace_path": str(repo),
        "branch": "codex/critical-major-done1",
        "model": "gpt-5.5",
        "effort": "xhigh",
        "discord_thread_target": "discord:123:456",
    }
    output = """
Bug: Onboarding can mark completion before required permissions are saved.
Severity: Major because users can reach the app with missing scheduling state.
Root cause: The completion flag is written before persistence succeeds.
Result: Moved completion after durable save and added a focused regression test.
Tests: OnboardingStateTests passed.
"""

    message = _render_completion_summary(job, output, status="completed")

    assert "Codex job `done1` completed" in message
    assert "fix: repair onboarding gate" in message
    assert "**Result / key findings**" in message
    assert "Root cause: The completion flag" in message
    assert "discord:123:456" in message
    assert len(message) <= 2000


def test_completion_summary_send_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    sent = []

    import tools.codex_job_tool as tool

    def fake_send(args):
        sent.append(args)
        return {"success": True, "message_id": "msg-1"}

    monkeypatch.setattr(tool, "_send_message", fake_send)
    job = {
        "job_id": "done2",
        "title": "Feature task",
        "workspace_path": str(tmp_path),
        "branch": "codex/feature-done2",
        "model": "gpt-5.5",
        "effort": "xhigh",
        "summary_target": "discord",
        "notify_on_completion": True,
        "monitor_log_path": str(tmp_path / "monitor.log"),
    }

    tool._send_completion_summary(job, "Result: Finished successfully", status="completed")
    tool._send_completion_summary(job, "Result: Finished successfully", status="completed")

    assert len(sent) == 1
    assert sent[0]["target"] == "discord"
    assert job["completion_summary_message_id"] == "msg-1"
    assert job["completion_summary_sent_at"]


def test_completion_record_persists_worker_handoff_and_distillation_marker():
    from tools.codex_job_tool import _detect_completion_state, _record_completion_state

    output = """
Result: Implemented profile-aware job metadata.
Files changed: tools/codex_job_tool.py, tests/tools/test_codex_job_tool.py
Tests run: scripts/run_tests.sh tests/tools/test_codex_job_tool.py -q passed.
Known blockers: MCP allowlists are metadata only unless Codex config profiles enforce them.
Notable lessons for future runs: Keep profile dashboards generic, not bug-specific.
Suggested docs/skill updates: Add a Hermes orchestration skill note.

HERMES_JOB_DONE {"status":"completed","recommendation":"open_pr","summary":"Profile support landed","tests":"passed"}
"""
    job = {
        "job_id": "done3",
        "status": "running",
        "phase": "running",
        "monitor_status": "running",
    }
    state = _detect_completion_state(output, tmux_alive=True)

    _record_completion_state(job, state, output)

    assert job["status"] == "completed"
    assert job["final_handoff"]["files_changed"].startswith("tools/codex_job_tool.py")
    assert job["tests"].startswith("scripts/run_tests.sh")
    assert "MCP allowlists are metadata only" in job["blockers"]
    assert job["distillation"]["recommended"] is True
    assert "worker suggested docs or skill updates" in job["distillation"]["reasons"]


def test_start_appends_machine_readable_completion_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import codex_job_tool

    result = json.loads(codex_job_tool({
        "action": "start",
        "title": "Review branch",
        "prompt": "Review the branch and give a normal human-readable summary.",
        "repo_path": str(repo),
        "workspace_mode": "local",
        "launch": False,
        "discord": False,
    }))

    assert result["success"] is True
    assert "Review the branch and give a normal human-readable summary." in result["prompt"]
    assert "Lightweight final handoff" in result["prompt"]
    assert "Files changed" in result["prompt"]
    assert "Suggested docs/skill updates" in result["prompt"]
    assert "Hermes completion protocol" in result["prompt"]
    assert "HERMES_JOB_DONE" in result["prompt"]
    assert "Do the task normally" in result["prompt"]
    assert "exactly one line" in result["prompt"]
    assert "HERMES_JOB_DONE" in "\n".join(result["tmux_commands"])


def test_start_can_disable_worker_handoff_template(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    repo = _make_repo(tmp_path)

    from tools.codex_job_tool import codex_job_tool

    result = json.loads(codex_job_tool({
        "action": "start",
        "title": "Plain protocol",
        "prompt": "Return only the completion sentinel.",
        "repo_path": str(repo),
        "workspace_mode": "local",
        "launch": False,
        "discord": False,
        "append_handoff_template": False,
    }))

    assert result["success"] is True
    assert "Lightweight final handoff" not in result["prompt"]
    assert "Hermes completion protocol" in result["prompt"]


def test_completion_detector_parses_sentinel_json():
    from tools.codex_job_tool import _detect_completion_state

    output = """
Verdict: REQUEST_CHANGES

The fix still misses deep-link handling for hidden over-limit habits.

HERMES_JOB_DONE {"status":"request_changes","verdict":"REQUEST_CHANGES","recommendation":"send_to_fixer","summary":"Habit limit fix has blocking deep-link gap","tests":"not_run"}
"""

    state = _detect_completion_state(output, tmux_alive=True)

    assert state["is_complete"] is True
    assert state["method"] == "sentinel"
    assert state["status"] == "request_changes"
    assert state["payload"]["verdict"] == "REQUEST_CHANGES"
    assert state["payload"]["recommendation"] == "send_to_fixer"


def test_completion_detector_marks_codex_idle_prompt_after_final_answer_as_heuristic_complete():
    from tools.codex_job_tool import _detect_completion_state

    output = """
Verdict: APPROVE

Blocking Issues
None.

Recommendation
Push/open a draft PR after CI runs in a writable environment.

─ Worked for 5m 30s ─────────────────────────────────────────────────────────

› Use /skills to list available skills

gpt-5.5 xhigh · Context 39% left · ~/.codex/worktrees/2353/FocusLock · FocusLock
"""

    state = _detect_completion_state(output, tmux_alive=True)

    assert state["is_complete"] is True
    assert state["method"] == "heuristic_idle_prompt"
    assert state["status"] == "approved"
    assert state["payload"]["verdict"] == "APPROVE"



def test_completion_detector_does_not_mark_environmental_test_blocker_as_blocked():
    from tools.codex_job_tool import _detect_completion_state

    output = """
Commit
abc123 fix: enforce limits

Verification
Core tests passed. Focused simulator tests were blocked by CoreSimulatorService timeout.

Recommendation
Start independent review.

─ Worked for 12m 10s ─────────────────────────────────────────────────────────

› Use /skills to list available skills

gpt-5.5 xhigh · Context 45% left · ~/.codex/worktrees/abcd/FocusLock · FocusLock
"""

    state = _detect_completion_state(output, tmux_alive=True)

    assert state["is_complete"] is True
    assert state["method"] == "heuristic_idle_prompt"
    assert state["status"] == "completed"
