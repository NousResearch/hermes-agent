"""Codex job runner tool for Discord/Hermes control-room workflows.

Creates Codex-app-compatible workspaces, launches interactive Codex in tmux,
and records enough metadata for Discord thread/status monitors.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error


CODEX_JOB_SCHEMA = {
    "name": "codex_job",
    "description": (
        "Start, inspect, list, or stop interactive Codex jobs managed by Hermes. "
        "Project jobs use isolated git worktrees by default, can explicitly run "
        "in a local checkout, or can create scratch workspaces under ~/Documents/Codex. "
        "Jobs launch Codex inside tmux with --no-alt-screen so output can be captured "
        "and mirrored into Discord status threads."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "status", "list", "stop"],
                "description": "Action to perform. Defaults to 'start'.",
            },
            "job_id": {
                "type": "string",
                "description": "Existing job id for status/stop, or optional explicit id for start.",
            },
            "title": {
                "type": "string",
                "description": "Human-readable job title. Used for branch/thread/session naming.",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt to pass to Codex for a started job.",
            },
            "repo_path": {
                "type": "string",
                "description": "Canonical git checkout path for project jobs. Required for worktree/local modes.",
            },
            "workspace_mode": {
                "type": "string",
                "enum": ["worktree", "local", "scratch"],
                "description": "Workspace mode. 'worktree' is the default for project jobs; 'local' uses repo_path directly; 'scratch' uses ~/Documents/Codex/<date>/<slug>.",
            },
            "base_ref": {
                "type": "string",
                "description": "Git ref to base a new worktree branch on. Defaults to the repo's current branch, then HEAD.",
            },
            "branch": {
                "type": "string",
                "description": "Optional branch name for worktree mode. Defaults to codex/<title-slug>-<job_id>.",
            },
            "launch": {
                "type": "boolean",
                "description": "Whether to actually launch Codex in tmux. Defaults true. Set false to prepare/record only.",
            },
            "model": {
                "type": "string",
                "description": "Codex model to pass with -m. Defaults to gpt-5.5.",
            },
            "effort": {
                "type": "string",
                "description": "Reasoning effort label to record in job metadata/status. The current Codex CLI may infer this from model rather than accept a flag.",
            },
            "approval": {
                "type": "string",
                "description": "Codex approval policy passed with -a. Defaults to never.",
            },
            "sandbox": {
                "type": "string",
                "description": "Codex sandbox passed with -s. Defaults to workspace-write; use read-only for smoke/audit jobs.",
            },
            "discord": {
                "type": "boolean",
                "description": "Whether to create/send Discord job status messages. Defaults true when a Discord target is provided.",
            },
            "discord_parent_target": {
                "type": "string",
                "description": "Parent Discord target for a new job thread, e.g. discord:#codex-control or discord:<channel_id>.",
            },
            "discord_target": {
                "type": "string",
                "description": "Existing Discord channel/thread target for status messages, e.g. discord:<channel_id>:<thread_id>.",
            },
            "thread_name": {
                "type": "string",
                "description": "Optional Discord thread name. Defaults to a Codex job title.",
            },
            "monitor": {
                "type": "boolean",
                "description": "Whether to launch a lightweight background monitor that edits the live Discord status message. Defaults true when Discord status is enabled and Codex is launched.",
            },
            "summary_target": {
                "type": "string",
                "description": "Optional target for concise completion/blocker summaries back to the orchestrator chat. Defaults to the Discord home/origin channel when Discord status is enabled. Use 'local' or empty to disable.",
            },
            "notify_on_completion": {
                "type": "boolean",
                "description": "Whether the monitor should send one concise summary to summary_target when the Codex tmux session exits. Defaults true when summary_target is set.",
            },
        },
        "required": [],
    },
}


_VALID_WORKSPACE_MODES = {"worktree", "local", "scratch"}
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _jobs_dir() -> Path:
    path = get_hermes_home() / "codex_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _logs_dir() -> Path:
    path = _jobs_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(text: str, fallback: str = "codex-job", max_len: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    value = re.sub(r"-+", "-", value)
    if not value:
        value = fallback
    return value[:max_len].strip("-") or fallback


def _generate_job_id() -> str:
    for _ in range(100):
        job_id = secrets.token_hex(2)
        if not _job_path(job_id).exists() and not (Path.home() / ".codex" / "worktrees" / job_id).exists():
            return job_id
    return secrets.token_hex(4)


def _run_command(command: str, cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=True,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _git_output(repo: str | Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def _ensure_git_repo(repo_path: str | Path) -> Path:
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists():
        raise ValueError(f"repo_path does not exist: {repo}")
    top = _git_output(repo, "rev-parse", "--show-toplevel")
    return Path(top).resolve()


def _current_branch_or_head(repo: str | Path) -> str:
    try:
        branch = _git_output(repo, "branch", "--show-current")
        if branch:
            return branch
    except subprocess.CalledProcessError:
        pass
    return "HEAD"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = _now_iso()
    _write_json(_job_path(job["job_id"]), job)


def _tmux_alive(session: str | None) -> bool:
    if not session or not shutil.which("tmux"):
        return False
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode == 0


def _build_tmux_commands(
    *,
    session: str,
    workspace_path: str | Path,
    prompt: str,
    log_path: str | Path,
    model: str = "gpt-5.5",
    approval: str = "never",
    sandbox: str = "workspace-write",
) -> list[str]:
    codex_parts = [
        "codex",
        "-C", str(workspace_path),
        "--no-alt-screen",
    ]
    if model:
        codex_parts.extend(["-m", model])
    if approval:
        codex_parts.extend(["-a", approval])
    if sandbox:
        codex_parts.extend(["-s", sandbox])
    codex_parts.append(prompt)
    codex_command = " ".join(shlex.quote(part) for part in codex_parts)
    pipe_command = "cat >> " + shlex.quote(str(log_path))
    return [
        f"tmux new-session -d -s {shlex.quote(session)} -x 140 -y 40 {shlex.quote(codex_command)}",
        f"tmux pipe-pane -o -t {shlex.quote(session)} {shlex.quote(pipe_command)}",
    ]


def _launch_tmux(job: dict[str, Any]) -> None:
    missing = [name for name in ("tmux", "codex") if not shutil.which(name)]
    if missing:
        raise RuntimeError(f"Missing required command(s) for launch: {', '.join(missing)}")
    commands = _build_tmux_commands(
        session=job["tmux_session"],
        workspace_path=job["workspace_path"],
        prompt=job["prompt"],
        log_path=job["log_path"],
        model=job.get("model", "gpt-5.5"),
        approval=job.get("approval", "never"),
        sandbox=job.get("sandbox", "workspace-write"),
    )
    # Replace a stale same-name session if one exists. Job ids are generated to
    # be unique, so this should only hit after a failed partial launch/retry.
    if _tmux_alive(job["tmux_session"]):
        _run_command(f"tmux kill-session -t {shlex.quote(job['tmux_session'])}")
    for command in commands:
        _run_command(command)
    job["tmux_commands"] = commands
    job["status"] = "running"
    job["started_at"] = _now_iso()


def _create_worktree(repo: Path, job_id: str, title: str, base_ref: str | None, branch: str | None) -> tuple[Path, str]:
    repo_name = repo.name
    slug = _slugify(title)
    branch_name = branch or f"codex/{slug}-{job_id}"
    wt = Path.home() / ".codex" / "worktrees" / job_id / repo_name
    wt.parent.mkdir(parents=True, exist_ok=True)
    if wt.exists():
        raise ValueError(f"worktree path already exists: {wt}")
    base = base_ref or _current_branch_or_head(repo)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", branch_name, str(wt), base],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return wt.resolve(), branch_name


def _create_scratch_workspace(title: str) -> tuple[Path, str]:
    today = datetime.now().strftime("%Y-%m-%d")
    slug = _slugify(title, fallback="scratch")
    base = Path.home() / "Documents" / "Codex" / today / slug
    path = base
    suffix = 2
    while path.exists():
        path = Path(f"{base}-{suffix}")
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    for child in ["inputs", "outputs", "scripts", "scratch"]:
        (path / child).mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-b", "main"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (path / ".gitignore").write_text(".DS_Store\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", ".gitignore"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Commit if git identity is available; Codex only needs a git repo, not a commit.
    subprocess.run(["git", "-C", str(path), "commit", "-m", "initial scratch workspace"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return path.resolve(), "main"


def _initial_status_message(job: dict[str, Any]) -> str:
    return (
        f"**Codex job `{job['job_id']}` started**\n"
        f"- Title: {job['title']}\n"
        f"- Status: {job['status']}\n"
        f"- Workspace mode: `{job['workspace_mode']}`\n"
        f"- Workspace: `{job['workspace_path']}`\n"
        f"- Branch: `{job.get('branch') or 'n/a'}`\n"
        f"- Model/effort: `{job.get('model')}` / `{job.get('effort')}`\n"
        f"- tmux: `{job['tmux_session']}`\n"
        f"- Attach: `{job['attach_command']}`\n"
    )


def _load_hermes_env_for_standalone_process() -> None:
    """Load ~/.hermes/.env for monitor subprocesses before messaging calls.

    Gateway-launched agent turns already have platform tokens in their process
    environment.  A detached ``python -m tools.codex_job_tool monitor ...``
    process may not, even though ~/.hermes/.env is configured.  Loading it here
    lets the monitor use the same Discord config as the gateway without copying
    or storing secrets in job records.
    """
    env_path = get_hermes_home() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(str(env_path), override=False, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            from dotenv import load_dotenv

            load_dotenv(str(env_path), override=False, encoding="latin-1")
        except Exception:
            pass
    except Exception:
        pass


def _send_message(args: dict[str, Any]) -> dict[str, Any]:
    _load_hermes_env_for_standalone_process()

    from tools.send_message_tool import send_message_tool

    return json.loads(send_message_tool(args))


def _setup_discord(job: dict[str, Any], args: dict[str, Any]) -> None:
    parent_target = args.get("discord_parent_target")
    target = args.get("discord_target")
    thread_name = args.get("thread_name") or f"codex-{job['job_id']}-{_slugify(job['title'], max_len=32)}"

    if parent_target and not target:
        created = _send_message({
            "action": "create_thread",
            "target": parent_target,
            "thread_name": thread_name,
            "thread_auto_archive_duration": 1440,
        })
        if created.get("error"):
            job["discord_error"] = created.get("error")
            return
        target = created.get("target")
        job["discord_thread_id"] = created.get("thread_id")
        job["discord_parent_target"] = parent_target
        job["discord_thread_target"] = target

    if target:
        sent = _send_message({
            "action": "send",
            "target": target,
            "message": _initial_status_message(job),
        })
        if sent.get("error"):
            job["discord_error"] = sent.get("error")
            return
        job["discord_thread_target"] = target
        job["discord_status_message_id"] = sent.get("message_id")
        job["discord_chat_id"] = sent.get("chat_id")


def _start_monitor(job: dict[str, Any]) -> None:
    module_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "tools.codex_job_tool", "monitor", job["job_id"]]
    log = _logs_dir() / f"{job['job_id']}.monitor.log"
    with log.open("ab") as fh:
        proc = subprocess.Popen(
            command,
            cwd=str(module_root),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    job["monitor_pid"] = proc.pid
    job["monitor_log_path"] = str(log)


def _handle_start(args: dict[str, Any]) -> str:
    title = (args.get("title") or "Codex job").strip()
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required when action='start'")

    mode = (args.get("workspace_mode") or "worktree").strip().lower()
    if mode not in _VALID_WORKSPACE_MODES:
        return tool_error(f"workspace_mode must be one of {sorted(_VALID_WORKSPACE_MODES)}")

    job_id = (args.get("job_id") or _generate_job_id()).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", job_id):
        return tool_error("job_id must be 3-32 characters of letters, numbers, underscore, or dash")
    if _job_path(job_id).exists():
        return tool_error(f"codex job already exists: {job_id}")

    repo = None
    branch = None
    worktree_path = None
    try:
        if mode == "scratch":
            workspace_path, branch = _create_scratch_workspace(title)
        else:
            if not args.get("repo_path"):
                return tool_error("repo_path is required for worktree/local workspace modes")
            repo = _ensure_git_repo(args["repo_path"])
            if mode == "worktree":
                workspace_path, branch = _create_worktree(
                    repo,
                    job_id,
                    title,
                    args.get("base_ref"),
                    args.get("branch"),
                )
                worktree_path = workspace_path
            else:
                workspace_path = repo
                branch = _current_branch_or_head(repo)
    except subprocess.CalledProcessError as exc:
        return tool_error(f"workspace setup failed: {exc.stderr.strip() or exc.stdout.strip() or exc}")
    except Exception as exc:
        return tool_error(f"workspace setup failed: {exc}")

    session = f"codex-{job_id}"
    log_path = _logs_dir() / f"{job_id}.tmux.log"
    model = args.get("model") or "gpt-5.5"
    effort = args.get("effort") or "xhigh"
    approval = args.get("approval") or "never"
    sandbox = args.get("sandbox") or "workspace-write"
    launch = bool(args.get("launch", True))
    discord_enabled = bool(args.get("discord", bool(args.get("discord_parent_target") or args.get("discord_target"))))
    summary_target = args.get("summary_target")
    if summary_target is None and discord_enabled:
        # In Discord gateway use, bare "discord" resolves to the configured home/origin-like channel.
        # Callers can pass an explicit platform:chat_id target to route elsewhere.
        summary_target = "discord"
    if isinstance(summary_target, str) and summary_target.strip().lower() in {"", "local", "none", "false"}:
        summary_target = None
    notify_on_completion = bool(args.get("notify_on_completion", bool(summary_target)))

    job = {
        "job_id": job_id,
        "title": title,
        "prompt": prompt,
        "workspace_mode": mode,
        "repo_path": str(repo) if repo else None,
        "workspace_path": str(Path(workspace_path).resolve()),
        "worktree_path": str(worktree_path) if worktree_path else None,
        "branch": branch,
        "base_ref": args.get("base_ref"),
        "status": "prepared",
        "model": model,
        "effort": effort,
        "approval": approval,
        "sandbox": sandbox,
        "tmux_session": session,
        "attach_command": f"tmux attach -t {session}",
        "log_path": str(log_path),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "summary_target": summary_target,
        "notify_on_completion": notify_on_completion,
    }

    try:
        if launch:
            _launch_tmux(job)
        else:
            job["tmux_commands"] = _build_tmux_commands(
                session=session,
                workspace_path=workspace_path,
                prompt=prompt,
                log_path=log_path,
                model=model,
                approval=approval,
                sandbox=sandbox,
            )
        if discord_enabled:
            _setup_discord(job, args)
        monitor = bool(args.get("monitor", discord_enabled and launch))
        if monitor and job.get("discord_thread_target") and job.get("discord_status_message_id"):
            _start_monitor(job)
        _save_job(job)
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        _save_job(job)
        return tool_error(f"codex job start failed after workspace setup: {exc}")

    return json.dumps({"success": True, **job}, ensure_ascii=False)


def _handle_status(args: dict[str, Any]) -> str:
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return tool_error("job_id is required when action='status'")
    job = _read_job(job_id)
    if not job:
        return tool_error(f"codex job not found: {job_id}")
    alive = _tmux_alive(job.get("tmux_session"))
    if job.get("status") == "running" and not alive:
        job["status"] = "exited"
        _save_job(job)
    return json.dumps({"success": True, "job": job, "tmux_alive": alive}, ensure_ascii=False)


def _handle_list(args: dict[str, Any]) -> str:
    jobs = []
    for path in sorted(_jobs_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            job["tmux_alive"] = _tmux_alive(job.get("tmux_session"))
            jobs.append(job)
        except Exception:
            continue
    limit = int(args.get("limit") or 20)
    return json.dumps({"success": True, "jobs": jobs[:limit]}, ensure_ascii=False)


def _handle_stop(args: dict[str, Any]) -> str:
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return tool_error("job_id is required when action='stop'")
    job = _read_job(job_id)
    if not job:
        return tool_error(f"codex job not found: {job_id}")
    alive_before = _tmux_alive(job.get("tmux_session"))
    if alive_before:
        _run_command(f"tmux kill-session -t {shlex.quote(job['tmux_session'])}")
    job["status"] = "stopped"
    job["stopped_at"] = _now_iso()
    _save_job(job)
    return json.dumps({"success": True, "job": job, "tmux_was_alive": alive_before}, ensure_ascii=False)


def codex_job_tool(args: dict[str, Any], **kw) -> str:
    action = (args.get("action") or "start").strip().lower()
    if action == "start":
        return _handle_start(args)
    if action == "status":
        return _handle_status(args)
    if action == "list":
        return _handle_list(args)
    if action == "stop":
        return _handle_stop(args)
    return tool_error(f"Unknown codex_job action: {action}")


def _strip_terminal_sequences(text: str) -> str:
    text = _ANSI_RE.sub("", text or "")
    text = text.replace("\r", "\n")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _capture_tmux_pane(job: dict[str, Any]) -> str:
    session = job.get("tmux_session")
    if not session or not shutil.which("tmux"):
        return ""
    proc = subprocess.run(
        ["tmux", "capture-pane", "-t", session, "-p", "-J", "-S", "-160"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _is_codex_status_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("› "):
        return True
    if stripped.startswith("─"):
        return True
    if stripped.startswith("Use /skills to list available skills"):
        return True
    if "Context" in stripped and "worktrees" in stripped and "gpt-" in stripped:
        return True
    if "esc to interrupt" in stripped and ("Starting MCP server" in stripped or "Booting MCP server" in stripped):
        return True
    return False


def _clean_codex_output(text: str, max_chars: int = 1800) -> str:
    text = _strip_terminal_sequences(text)
    cleaned: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if _is_codex_status_noise(line):
            continue
        blank = not line.strip()
        if blank and previous_blank:
            continue
        cleaned.append(line)
        previous_blank = blank
    output = "\n".join(cleaned).strip()
    if len(output) <= max_chars:
        return output
    selected: list[str] = []
    total = 0
    for line in reversed(cleaned):
        line_len = len(line) + 1
        if selected and total + line_len > max_chars:
            break
        selected.append(line)
        total += line_len
    return "\n".join(reversed(selected)).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _workspace_summary(job: dict[str, Any], max_chars: int = 520) -> str:
    workspace = job.get("workspace_path")
    if not workspace:
        return ""
    workspace_path = Path(workspace)
    if not workspace_path.exists() or not shutil.which("git"):
        return ""
    try:
        status = subprocess.run(
            ["git", "-C", str(workspace_path), "status", "--short"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        diff_stat = subprocess.run(
            ["git", "-C", str(workspace_path), "diff", "--stat"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except Exception:
        return ""

    parts: list[str] = []
    if status.returncode == 0:
        status_text = status.stdout.strip() or "clean"
        parts.append("Status:\n" + status_text)
    if diff_stat.returncode == 0 and diff_stat.stdout.strip():
        parts.append("Diff stat:\n" + diff_stat.stdout.strip())
    return _truncate_text("\n\n".join(parts), max_chars)


def _extract_key_findings(cleaned_output: str, max_chars: int = 700) -> str:
    """Pull durable task facts out of Codex's scrolling TUI output.

    This intentionally supports more than bug-fix work.  Bug hunts usually emit
    labels such as ``Bug:``, ``Severity:``, and ``Root cause:``; implementation,
    research, refactor, and release tasks more often emit labels such as
    ``Objective:``, ``Decision:``, ``Approach:``, ``Plan:``, ``Blocker:``, or
    ``Next step:``.  The live dashboard should surface whichever durable facts
    Codex has produced instead of assuming the job is always about a bug.
    """
    labels = (
        "Objective:",
        "Goal:",
        "Task:",
        "Requirement:",
        "Summary:",
        "Finding:",
        "Findings:",
        "Decision:",
        "Approach:",
        "Plan:",
        "Planned fix:",
        "Implementation:",
        "Current status:",
        "Current changes:",
        "Progress:",
        "Blocker:",
        "Risk:",
        "Next step:",
        "Next steps:",
        "Tests:",
        "Verification:",
        "Result:",
        "Bug:",
        "Severity:",
        "Root cause:",
    )
    lines = cleaned_output.splitlines()
    findings: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not any(stripped.startswith(label) for label in labels):
            index += 1
            continue
        block = [stripped]
        index += 1
        while index < len(lines):
            continuation = lines[index].strip()
            if not continuation:
                break
            if any(continuation.startswith(label) for label in labels):
                break
            if continuation.startswith(("•", "└", "Read ", "Search ", "Ran ", "Edited ", "List ")):
                break
            block.append(continuation)
            index += 1
        findings.append(" ".join(block))
        if len("\n".join(findings)) >= max_chars:
            break
    return _truncate_text("\n".join(findings), max_chars)


def _recent_activity(cleaned_output: str, max_chars: int = 700) -> str:
    lines: list[str] = []
    for raw in cleaned_output.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped in {"Explored", "Ran", "Edited"}:
            continue
        if stripped.startswith("└ "):
            stripped = stripped[2:].strip()
        if stripped.startswith("│ "):
            stripped = stripped[2:].strip()
        lines.append(stripped)
    return _truncate_text("\n".join(lines[-12:]), max_chars) or "(no useful captured activity yet)"


def _render_monitor_status(job: dict[str, Any], alive: bool, output: str) -> str:
    phase = "running" if alive else "exited"
    cleaned = _clean_codex_output(output, max_chars=5000)
    key_findings = _extract_key_findings(cleaned)
    recent = _recent_activity(cleaned)
    workspace_summary = _workspace_summary(job)

    message = (
        f"**Codex job `{job['job_id']}` live status**\n"
        f"- **Title:** {job['title']}\n"
        f"- **Phase:** `{phase}`\n"
        f"- **Model/effort:** `{job.get('model')}` / `{job.get('effort')}`\n"
        f"- **Branch:** `{job.get('branch') or 'n/a'}`\n"
        f"- **Workspace:** `{job['workspace_path']}`\n"
        f"- **Attach:** `{job['attach_command']}`\n"
    )
    if key_findings:
        message += f"\n**Task summary / key findings**\n```text\n{key_findings}\n```\n"
    if workspace_summary:
        message += f"\n**Workspace changes**\n```text\n{workspace_summary}\n```\n"
    message += f"\n**Recent useful activity**\n```text\n{recent}\n```"
    # Discord hard-limits message content to 2,000 characters. Leave margin for
    # Unicode/code-fence edge cases so background status edits do not fail.
    return _truncate_text(message, 1900)


def _append_monitor_log(job: dict[str, Any], message: str) -> None:
    log_path = job.get("monitor_log_path") or str(_logs_dir() / f"{job['job_id']}.monitor.log")
    try:
        with Path(log_path).open("a", encoding="utf-8") as fh:
            fh.write(f"{_now_iso()} {message}\n")
    except Exception:
        pass


def _last_commit_summary(job: dict[str, Any]) -> str:
    workspace = job.get("workspace_path")
    if not workspace:
        return ""
    workspace_path = Path(workspace)
    if not workspace_path.exists() or not shutil.which("git"):
        return ""
    try:
        sha = subprocess.run(
            ["git", "-C", str(workspace_path), "rev-parse", "--short", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        subject = subprocess.run(
            ["git", "-C", str(workspace_path), "log", "-1", "--pretty=%s"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        return f"{sha} {subject}".strip()
    except Exception:
        return ""


def _render_completion_summary(job: dict[str, Any], output: str, status: str = "exited") -> str:
    cleaned = _clean_codex_output(output, max_chars=5000)
    key_findings = _extract_key_findings(cleaned, max_chars=650)
    recent = _recent_activity(cleaned, max_chars=450)
    workspace_summary = _workspace_summary(job, max_chars=360)
    commit = _last_commit_summary(job)
    detail_target = job.get("discord_thread_target") or job.get("discord_parent_target") or "#codex-control thread"

    message = (
        f"**Codex job `{job['job_id']}` {status}**\n"
        f"- **Title:** {job.get('title')}\n"
        f"- **Model/effort:** `{job.get('model')}` / `{job.get('effort')}`\n"
        f"- **Branch:** `{job.get('branch') or 'n/a'}`\n"
    )
    if commit:
        message += f"- **Latest commit:** `{commit}`\n"
    message += f"- **Details:** `{detail_target}`\n"

    if key_findings:
        message += f"\n**Result / key findings**\n```text\n{key_findings}\n```\n"
    if workspace_summary:
        message += f"\n**Workspace changes**\n```text\n{workspace_summary}\n```\n"
    message += f"\n**Recent activity**\n```text\n{recent}\n```"
    return _truncate_text(message, 1900)


def _send_completion_summary(job: dict[str, Any], output: str, status: str = "exited") -> None:
    if not job.get("notify_on_completion") or job.get("completion_summary_sent_at"):
        return
    target = job.get("summary_target")
    if not target:
        return
    message = _render_completion_summary(job, output, status=status)
    try:
        result = _send_message({"action": "send", "target": target, "message": message})
        if result.get("error"):
            job["completion_summary_error"] = result.get("error")
            _append_monitor_log(job, f"completion summary error: {result.get('error')}")
            return
        job["completion_summary_sent_at"] = _now_iso()
        job["completion_summary_message_id"] = result.get("message_id")
        _append_monitor_log(job, "completion summary sent")
    except Exception as exc:
        job["completion_summary_error"] = str(exc)
        _append_monitor_log(job, f"completion summary exception: {exc}")


def monitor_job(job_id: str, interval: int = 30, max_seconds: int = 60 * 60 * 6) -> None:
    started = time.time()
    while time.time() - started < max_seconds:
        job = _read_job(job_id)
        if not job:
            return
        target = job.get("discord_thread_target")
        message_id = job.get("discord_status_message_id")
        if not target or not message_id:
            return
        alive = _tmux_alive(job.get("tmux_session"))
        output = _capture_tmux_pane(job) if alive else ""
        if not output:
            log_path = Path(job.get("log_path") or "")
            if log_path.exists():
                try:
                    output = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                except Exception:
                    output = "(failed to read log)"
        try:
            result = _send_message({
                "action": "edit",
                "target": target,
                "message_id": message_id,
                "message": _render_monitor_status(job, alive, output),
            })
            if result.get("error"):
                _append_monitor_log(job, f"edit error: {result.get('error')}")
            else:
                _append_monitor_log(job, "edit ok")
        except Exception as exc:
            _append_monitor_log(job, f"edit exception: {exc}")
        if not alive:
            job["status"] = "exited"
            job["ended_at"] = _now_iso()
            _send_completion_summary(job, output, status="completed")
            _save_job(job)
            return
        time.sleep(interval)


def _check_codex_job_requirements() -> bool:
    return bool(shutil.which("git") and shutil.which("tmux") and shutil.which("codex"))


registry.register(
    name="codex_job",
    toolset="terminal",
    schema=CODEX_JOB_SCHEMA,
    handler=codex_job_tool,
    emoji="🤖",
)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "monitor":
        monitor_job(sys.argv[2])
    else:
        print("Usage: python -m tools.codex_job_tool monitor <job_id>", file=sys.stderr)
        raise SystemExit(2)
