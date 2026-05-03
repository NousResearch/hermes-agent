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
            "profile": {
                "type": "string",
                "description": "Transparent Hermes job profile to record and use for safe launch defaults. Built-ins include base-rich, review-rich-readonly, ios-full, web-full, ops-full, and hermes-full. Defaults to base-rich.",
            },
            "profile_overrides": {
                "type": "object",
                "description": "Optional metadata-only profile overrides such as summary, included_capabilities, omitted_capabilities, runner_limitations, or capability_enforcement. These do not hard-enforce MCP allowlists.",
            },
            "codex_profile": {
                "type": "string",
                "description": "Existing Codex CLI config.toml profile to pass with --profile. Defaults to none; Hermes job profiles are metadata/policy bundles unless mapped here.",
            },
            "codex_config_overrides": {
                "type": "object",
                "description": "Codex CLI config overrides to pass as repeated -c key=value flags. Values are rendered as TOML-ish literals. Use only for supported Codex config keys.",
            },
            "search": {
                "type": "boolean",
                "description": "Whether to pass --search to Codex. Defaults from the selected Hermes job profile.",
            },
            "append_handoff_template": {
                "type": "boolean",
                "description": "Whether to append the lightweight worker final handoff request to the prompt. Defaults true.",
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
_DEFAULT_JOB_PROFILE = "base-rich"
_COMMON_RUNNER_LIMITATIONS = [
    "MCP and plugin availability comes from the Codex CLI configuration unless codex_profile or codex_config_overrides are supplied.",
    "codex_job records profile intent and coarse launch policy; it does not synthesize a hard per-MCP allowlist.",
]
_BUILTIN_JOB_PROFILES: dict[str, dict[str, Any]] = {
    "base-rich": {
        "summary": "Default coding and research worker with broad safe workspace capabilities.",
        "included_capabilities": [
            "workspace shell/file/git access",
            "repo inspection and edits",
            "configured Codex MCP servers and plugins",
            "web search when enabled by profile, config, or caller",
            "GitHub/browser/API tools when configured in Codex",
        ],
        "omitted_capabilities": [
            "danger-full-access sandbox by default",
            "secret printing or raw auth/log exfiltration",
            "production mutations without explicit task instructions",
        ],
        "defaults": {
            "model": "gpt-5.5",
            "effort": "xhigh",
            "approval": "never",
            "sandbox": "workspace-write",
            "search": False,
        },
    },
    "review-rich-readonly": {
        "summary": "Broad review/audit profile with read-only launch defaults.",
        "included_capabilities": [
            "repo-wide inspection",
            "git diff/status/log analysis",
            "test and CI log review from existing files",
            "configured read-only browser/search/GitHub tools",
        ],
        "omitted_capabilities": [
            "file mutation",
            "commit, push, PR creation",
            "destructive shell commands",
            "long-running simulator or deployment actions unless explicitly reconfigured",
        ],
        "defaults": {
            "model": "gpt-5.5",
            "effort": "xhigh",
            "approval": "never",
            "sandbox": "read-only",
            "search": False,
        },
    },
    "ios-full": {
        "summary": "iOS/Xcode/simulator-oriented implementation profile.",
        "included_capabilities": [
            "Swift and Xcode project editing",
            "SwiftPM/xcodebuild commands",
            "simulator automation when configured",
            "App Store/TestFlight tooling when configured",
            "mobile and Xcode MCPs when configured and healthy",
        ],
        "omitted_capabilities": [
            "forced startup of fragile or unavailable Xcode MCPs",
            "physical-device actions without explicit operator setup",
            "secret or signing material disclosure",
        ],
        "runner_limitations": [
            "XcodeBuildMCP and mobile MCPs are config-driven and may be skipped if unavailable or stuck.",
        ],
        "defaults": {
            "model": "gpt-5.5",
            "effort": "xhigh",
            "approval": "never",
            "sandbox": "workspace-write",
            "search": False,
        },
    },
    "web-full": {
        "summary": "Web/frontend/browser/API-oriented implementation profile.",
        "included_capabilities": [
            "frontend and backend source edits",
            "npm/pnpm/yarn and local dev servers",
            "browser/devtools automation when configured",
            "API inspection and web search",
            "Playwright-style verification when available",
        ],
        "omitted_capabilities": [
            "persistent browser account mutation unless requested",
            "secret capture",
            "production deployment mutations by default",
        ],
        "defaults": {
            "model": "gpt-5.5",
            "effort": "xhigh",
            "approval": "never",
            "sandbox": "workspace-write",
            "search": True,
        },
    },
    "ops-full": {
        "summary": "GitHub, CI, release, and devops-oriented profile.",
        "included_capabilities": [
            "GitHub and CI inspection when configured",
            "branch, commit, push, and draft PR work when requested",
            "release/devops CLIs when configured",
            "web search for current platform docs when needed",
        ],
        "omitted_capabilities": [
            "production deploys without explicit approval",
            "credential display",
            "danger-full-access sandbox by default",
            "destructive infrastructure commands unless explicitly requested",
        ],
        "defaults": {
            "model": "gpt-5.5",
            "effort": "xhigh",
            "approval": "never",
            "sandbox": "workspace-write",
            "search": True,
        },
    },
    "hermes-full": {
        "summary": "Hermes source, gateway, tool, plugin, and skill development profile.",
        "included_capabilities": [
            "Hermes Python source edits",
            "gateway/tool/plugin/skill code paths",
            "repo docs and plan files",
            "focused pytest wrapper runs",
            "configured Codex MCP servers and plugins",
        ],
        "omitted_capabilities": [
            "raw runtime logs or token files",
            "gateway restart loops",
            "private ops-doc mutation outside the workspace",
        ],
        "defaults": {
            "model": "gpt-5.5",
            "effort": "xhigh",
            "approval": "never",
            "sandbox": "workspace-write",
            "search": False,
        },
    },
}


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


def _default_discord_thread_name(title: str, job_id: str, max_len: int = 90) -> str:
    """Prefer human-readable Discord thread titles; keep technical IDs in status."""
    value = re.sub(r"\s+", " ", (title or "").strip())
    if not value or value.lower() in {"codex job", "job", "review", "implementation"}:
        value = f"Codex job {job_id}"
    value = _suppress_dangerous_mentions(value)
    if len(value) <= max_len:
        return value
    return value[: max(0, max_len - 1)].rstrip(" -–—·") + "…"


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _resolve_job_profile(args: dict[str, Any]) -> dict[str, Any]:
    requested = str(args.get("profile") or _DEFAULT_JOB_PROFILE).strip() or _DEFAULT_JOB_PROFILE
    profile_key = requested.lower()
    base = dict(_BUILTIN_JOB_PROFILES.get(profile_key) or {})
    source = "built-in" if base else "custom"
    defaults = dict(base.get("defaults") or _BUILTIN_JOB_PROFILES[_DEFAULT_JOB_PROFILE]["defaults"])
    runner_limitations = [*_COMMON_RUNNER_LIMITATIONS, *_coerce_string_list(base.get("runner_limitations"))]

    profile = {
        "name": profile_key if source == "built-in" else requested,
        "source": source,
        "summary": base.get("summary") or "Custom Hermes job profile. Capability metadata is caller-provided; Codex enforcement is config-driven.",
        "included_capabilities": _coerce_string_list(base.get("included_capabilities")),
        "omitted_capabilities": _coerce_string_list(base.get("omitted_capabilities")),
        "runner_limitations": runner_limitations,
        "capability_enforcement": base.get("capability_enforcement") or "metadata_plus_coarse_codex_cli_policy",
        "defaults": defaults,
        "codex_profile": base.get("codex_profile"),
        "codex_config_overrides": dict(base.get("codex_config_overrides") or {}),
    }

    overrides = args.get("profile_overrides") or {}
    if isinstance(overrides, dict):
        if overrides.get("summary"):
            profile["summary"] = str(overrides["summary"]).strip()
        if "included_capabilities" in overrides:
            profile["included_capabilities"] = _coerce_string_list(overrides.get("included_capabilities"))
        if "omitted_capabilities" in overrides:
            profile["omitted_capabilities"] = _coerce_string_list(overrides.get("omitted_capabilities"))
        if "runner_limitations" in overrides:
            profile["runner_limitations"] = [
                *_COMMON_RUNNER_LIMITATIONS,
                *_coerce_string_list(overrides.get("runner_limitations")),
            ]
        if overrides.get("capability_enforcement"):
            profile["capability_enforcement"] = str(overrides["capability_enforcement"]).strip()
    return profile


def _tomlish_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (
            stripped in {"true", "false"}
            or re.fullmatch(r"-?\d+(\.\d+)?", stripped)
            or stripped.startswith(("[", "{", '"', "'"))
        ):
            return stripped
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_tomlish_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{key} = {_tomlish_literal(item)}" for key, item in value.items())
        return "{ " + items + " }"
    return json.dumps(value)


def _config_override_arg(key: str, value: Any) -> str:
    return f"{key}={_tomlish_literal(value)}"


def _normalize_config_overrides(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("codex_config_overrides must be an object")
    overrides: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        overrides[key_text] = item
    return overrides


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


_COMPLETION_PROTOCOL = """

Hermes completion protocol:
Do the task normally and provide the usual human-readable final summary. When you are fully done or blocked, append exactly one line as the final line:
HERMES_JOB_DONE {"status":"completed|blocked|request_changes|approved|needs_manual_verification","recommendation":"short_next_step","summary":"one sentence","commit":null,"tests":"passed|failed|blocked|not_run|partial"}
Use valid single-line JSON after HERMES_JOB_DONE. After that line, stop and wait at the prompt.
""".strip()

_WORKER_HANDOFF_TEMPLATE = """

Lightweight final handoff:
When you finish, keep the human-readable final summary concise and include these sections:
- Result
- Files changed
- Tests run
- Known blockers
- Notable lessons for future runs
- Suggested docs/skill updates
""".strip()

_MCP_STARTUP_STALL_SECONDS = 45


def _append_worker_handoff_template(prompt: str) -> str:
    if "Lightweight final handoff:" in prompt:
        return prompt
    return f"{prompt.rstrip()}\n\n{_WORKER_HANDOFF_TEMPLATE}"


def _append_completion_protocol(prompt: str) -> str:
    """Add a tiny machine-readable footer without replacing the human summary."""
    if "Hermes completion protocol:" in prompt:
        return prompt
    return f"{prompt.rstrip()}\n\n{_COMPLETION_PROTOCOL}"


def _prepare_prompt(prompt: str, append_handoff_template: bool = True) -> str:
    if append_handoff_template:
        prompt = _append_worker_handoff_template(prompt)
    return _append_completion_protocol(prompt)


def _build_codex_parts(
    *,
    workspace_path: str | Path,
    prompt: str | None,
    model: str = "gpt-5.5",
    approval: str = "never",
    sandbox: str = "workspace-write",
    codex_profile: str | None = None,
    codex_config_overrides: dict[str, Any] | None = None,
    search: bool = False,
) -> list[str]:
    parts = [
        "codex",
        "-C", str(workspace_path),
        "--no-alt-screen",
    ]
    if codex_profile:
        parts.extend(["-p", codex_profile])
    if model:
        parts.extend(["-m", model])
    if approval:
        parts.extend(["-a", approval])
    if sandbox:
        parts.extend(["-s", sandbox])
    if search:
        parts.append("--search")
    for key, value in (codex_config_overrides or {}).items():
        parts.extend(["-c", _config_override_arg(key, value)])
    if prompt is not None:
        parts.append(prompt)
    return parts


def _codex_launch_flags(
    *,
    workspace_path: str | Path,
    model: str,
    approval: str,
    sandbox: str,
    codex_profile: str | None,
    codex_config_overrides: dict[str, Any],
    search: bool,
) -> list[str]:
    parts = _build_codex_parts(
        workspace_path=workspace_path,
        prompt=None,
        model=model,
        approval=approval,
        sandbox=sandbox,
        codex_profile=codex_profile,
        codex_config_overrides=codex_config_overrides,
        search=search,
    )
    return parts[1:]


def _build_tmux_commands(
    *,
    session: str,
    workspace_path: str | Path,
    prompt: str,
    log_path: str | Path,
    model: str = "gpt-5.5",
    approval: str = "never",
    sandbox: str = "workspace-write",
    codex_profile: str | None = None,
    codex_config_overrides: dict[str, Any] | None = None,
    search: bool = False,
) -> list[str]:
    codex_parts = _build_codex_parts(
        workspace_path=workspace_path,
        prompt=prompt,
        model=model,
        approval=approval,
        sandbox=sandbox,
        codex_profile=codex_profile,
        codex_config_overrides=codex_config_overrides,
        search=search,
    )
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
        codex_profile=job.get("codex_profile"),
        codex_config_overrides=job.get("codex_config_overrides") or {},
        search=bool(job.get("search")),
    )
    # Replace a stale same-name session if one exists. Job ids are generated to
    # be unique, so this should only hit after a failed partial launch/retry.
    if _tmux_alive(job["tmux_session"]):
        _run_command(f"tmux kill-session -t {shlex.quote(job['tmux_session'])}")
    for command in commands:
        _run_command(command)
    job["tmux_commands"] = commands
    job["status"] = "running"
    job["phase"] = "running"
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


def _format_capability_summary(job: dict[str, Any], max_chars: int = 360) -> str:
    included = "; ".join(_coerce_string_list(job.get("included_capabilities"))[:4]) or "not recorded"
    omitted = "; ".join(_coerce_string_list(job.get("omitted_capabilities"))[:3]) or "none recorded"
    return _truncate_text(f"Included: {included}. Omitted: {omitted}.", max_chars)


def _format_launch_policy(job: dict[str, Any], max_chars: int = 220) -> str:
    codex_profile = job.get("codex_profile") or "none"
    overrides = job.get("codex_config_overrides") or {}
    override_count = len(overrides) if isinstance(overrides, dict) else 0
    search = "on" if job.get("search") else "off"
    return _truncate_text(
        f"sandbox={job.get('sandbox')}; approval={job.get('approval')}; search={search}; codex_profile={codex_profile}; -c overrides={override_count}",
        max_chars,
    )


def _initial_status_message(job: dict[str, Any]) -> str:
    return (
        f"**Codex job `{job['job_id']}` started**\n"
        f"- Title: {job['title']}\n"
        f"- Status: {job['status']}\n"
        f"- Phase: `{job.get('phase') or job['status']}`\n"
        f"- Profile: `{job.get('profile')}` — {_truncate_text(job.get('profile_summary') or '', 160)}\n"
        f"- Capabilities: {_format_capability_summary(job, max_chars=260)}\n"
        f"- Launch policy: `{_format_launch_policy(job)}`\n"
        f"- Workspace mode: `{job['workspace_mode']}`\n"
        f"- Repo: `{job.get('repo_path') or 'n/a'}`\n"
        f"- Workspace: `{job['workspace_path']}`\n"
        f"- Worktree: `{job.get('worktree_path') or 'n/a'}`\n"
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
    thread_name = args.get("thread_name") or _default_discord_thread_name(job.get("title") or "Codex job", job["job_id"])

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
        job["discord_thread_name"] = thread_name
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
    prompt = _prepare_prompt(prompt, append_handoff_template=bool(args.get("append_handoff_template", True)))

    mode = (args.get("workspace_mode") or "worktree").strip().lower()
    if mode not in _VALID_WORKSPACE_MODES:
        return tool_error(f"workspace_mode must be one of {sorted(_VALID_WORKSPACE_MODES)}")
    try:
        profile = _resolve_job_profile(args)
        codex_config_overrides = {
            **_normalize_config_overrides(profile.get("codex_config_overrides")),
            **_normalize_config_overrides(args.get("codex_config_overrides")),
        }
    except Exception as exc:
        return tool_error(str(exc))

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
    profile_defaults = profile.get("defaults") or {}
    model = args.get("model") or profile_defaults.get("model") or "gpt-5.5"
    effort = args.get("effort") or profile_defaults.get("effort") or "xhigh"
    approval = args.get("approval") or profile_defaults.get("approval") or "never"
    sandbox = args.get("sandbox") or profile_defaults.get("sandbox") or "workspace-write"
    search = bool(args.get("search", profile_defaults.get("search", False)))
    codex_profile = (args.get("codex_profile") or profile.get("codex_profile") or "").strip() or None
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
        "phase": "prepared",
        "model": model,
        "effort": effort,
        "profile": profile["name"],
        "profile_source": profile["source"],
        "profile_summary": profile["summary"],
        "profile_policy": profile_defaults,
        "included_capabilities": profile["included_capabilities"],
        "omitted_capabilities": profile["omitted_capabilities"],
        "runner_limitations": profile["runner_limitations"],
        "capability_enforcement": profile["capability_enforcement"],
        "codex_profile": codex_profile,
        "codex_config_overrides": codex_config_overrides,
        "search": search,
        "codex_launch_flags": _codex_launch_flags(
            workspace_path=workspace_path,
            model=model,
            approval=approval,
            sandbox=sandbox,
            codex_profile=codex_profile,
            codex_config_overrides=codex_config_overrides,
            search=search,
        ),
        "approval": approval,
        "sandbox": sandbox,
        "tmux_session": session,
        "attach_command": f"tmux attach -t {session}",
        "log_path": str(log_path),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "latest_activity": "job prepared",
        "key_findings": "",
        "tests": "not_run",
        "blockers": "",
        "final_handoff": {},
        "distillation": {"recommended": False, "status": "not_needed", "reasons": []},
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
                codex_profile=codex_profile,
                codex_config_overrides=codex_config_overrides,
                search=search,
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
    output = ""
    if job.get("status") == "running":
        output = _capture_tmux_pane(job) if alive else ""
        if not output:
            log_path = Path(job.get("log_path") or "")
            if log_path.exists():
                try:
                    output = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                except Exception:
                    output = ""
        state = _detect_completion_state(output, tmux_alive=alive)
        if state.get("is_complete"):
            _record_completion_state(job, state, output)
            if not alive:
                job["ended_at"] = _now_iso()
            _save_job(job)
        elif not alive:
            job["status"] = "exited"
            job["phase"] = "exited"
            job["ended_at"] = _now_iso()
            _save_job(job)
        elif output:
            _update_launch_health(job, output, tmux_alive=alive)
            _update_runtime_observations(job, output)
            _save_job(job)
    return json.dumps({"success": True, "job": job, "tmux_alive": alive, "summary": _job_status_summary(job, alive)}, ensure_ascii=False)


def _handle_list(args: dict[str, Any]) -> str:
    jobs = []
    for path in sorted(_jobs_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            job["tmux_alive"] = _tmux_alive(job.get("tmux_session"))
            job["summary"] = _job_status_summary(job, bool(job["tmux_alive"]))
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
    job["phase"] = "stopped"
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


def _detect_launch_health(output: str, *, tmux_alive: bool, elapsed_seconds: float = 0) -> dict[str, Any] | None:
    """Detect early Codex startup states that need operator attention."""
    if not tmux_alive:
        return None
    cleaned = _strip_terminal_sequences(output or "")
    recent_lines = [line.strip() for line in cleaned.splitlines() if line.strip()][-24:]
    lowered_lines = [line.lower() for line in recent_lines]
    lowered = re.sub(r"\s+", " ", "\n".join(recent_lines)).strip().lower()
    if not lowered:
        return None

    last_progress_line = max((index for index, line in enumerate(recent_lines) if _is_launch_progress_line(line)), default=-1)

    trust_line = max(
        (
            index
            for index, line in enumerate(lowered_lines)
            if (
                ("do you trust" in line and ("folder" in line or "files" in line or "repo" in line))
                or "trust the files in this folder" in line
                or "trust this repo" in line
            )
        ),
        default=-1,
    )
    if trust_line > last_progress_line:
        return {
            "kind": "repo_trust_prompt",
            "phase": "awaiting_user_input",
            "severity": "needs_input",
            "summary": "Codex is waiting at a repository trust prompt.",
            "recommendation": "Inspect with tmux attach and approve only if this is the expected trusted repo/worktree.",
        }

    permission_patterns = (
        r"\ballow\b.{0,80}\b(command|tool|operation|edit|write|run)\b",
        r"\b(approve|deny)\b.{0,80}\b(command|tool|operation|request)\b",
        r"\brequires approval\b",
        r"\bwaiting for approval\b",
        r"\bdo you want to allow\b",
        r"\bwould you like to run\b",
    )
    permission_line = max(
        (
            index
            for index, line in enumerate(lowered_lines)
            if any(re.search(pattern, line) for pattern in permission_patterns)
        ),
        default=-1,
    )
    if permission_line > last_progress_line:
        return {
            "kind": "permission_prompt",
            "phase": "awaiting_user_input",
            "severity": "needs_input",
            "summary": "Codex appears to be waiting for an approval or permission prompt.",
            "recommendation": "Inspect with tmux attach and approve or deny the prompt so the job does not sit idle.",
        }

    mcp_line = max(
        (
            index
            for index, line in enumerate(lowered_lines)
            if (
                "starting mcp server" in line
                or "booting mcp server" in line
                or ("mcp server" in line and ("starting" in line or "booting" in line or "handshaking" in line))
            )
        ),
        default=-1,
    )
    mcp_starting = mcp_line >= 0
    if elapsed_seconds >= _MCP_STARTUP_STALL_SECONDS and mcp_line > last_progress_line and mcp_starting and "esc to interrupt" in lowered:
        return {
            "kind": "mcp_startup_stall",
            "phase": "startup_stalled",
            "severity": "startup_delay",
            "summary": "Codex appears stuck during MCP startup.",
            "recommendation": "Inspect with tmux attach; press Esc only when the stalled MCP is irrelevant to this job, otherwise wait or fix the MCP configuration.",
        }

    return None


def _job_elapsed_seconds(job: dict[str, Any]) -> float:
    raw = job.get("started_at") or job.get("created_at")
    if not raw:
        return 0.0
    try:
        started = datetime.strptime(str(raw).replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return 0.0
    return max(0.0, (datetime.utcnow() - started).total_seconds())


def _normalize_codex_activity_line(line: str) -> str:
    value = line.strip()
    while value.startswith(("•", "└", "├", "│")):
        value = value[1:].strip()
    return value


def _is_launch_progress_line(line: str) -> bool:
    lowered = _normalize_codex_activity_line(line).lower()
    return bool(
        re.match(r"^(ran|edited|read|search|searched|listed|opened)\b", lowered)
        or lowered.startswith(("result:", "objective:", "approach:", "plan:"))
        or "hermes_job_done" in lowered
    )


def _looks_like_launch_progress(output: str) -> bool:
    recent_lines = [line.strip() for line in _strip_terminal_sequences(output or "").splitlines() if line.strip()][-24:]
    return any(_is_launch_progress_line(line) for line in recent_lines)


def _resolve_launch_health(job: dict[str, Any]) -> None:
    existing = job.get("health_check") if isinstance(job.get("health_check"), dict) else {}
    if not existing or existing.get("resolved_at"):
        return
    existing["status"] = "resolved"
    existing["resolved_at"] = _now_iso()
    job["health_check"] = existing
    if job.get("phase") in {"awaiting_user_input", "startup_stalled"}:
        job["phase"] = "running"
    if job.get("blockers") == existing.get("summary"):
        job["blockers"] = ""


def _update_launch_health(job: dict[str, Any], output: str, *, tmux_alive: bool, elapsed_seconds: float | None = None) -> dict[str, Any] | None:
    elapsed = _job_elapsed_seconds(job) if elapsed_seconds is None else elapsed_seconds
    health = _detect_launch_health(output, tmux_alive=tmux_alive, elapsed_seconds=elapsed)
    if not health:
        if _looks_like_launch_progress(output):
            _resolve_launch_health(job)
        return None
    existing = job.get("health_check") if isinstance(job.get("health_check"), dict) else {}
    is_continuing_same_detection = existing.get("kind") == health.get("kind") and existing.get("detected_at") and not existing.get("resolved_at")
    if is_continuing_same_detection:
        health["detected_at"] = existing["detected_at"]
    else:
        health["detected_at"] = _now_iso()
        for key in (
            "health_alert_sent_at",
            "health_alert_kind",
            "health_alert_message_id",
            "health_alert_target",
            "health_alert_error",
        ):
            job.pop(key, None)
    health["elapsed_seconds"] = int(elapsed)
    health["attach_command"] = job.get("attach_command")
    job["health_check"] = health
    job["phase"] = health.get("phase") or job.get("phase") or "running"
    if health.get("summary"):
        job["blockers"] = health["summary"]
    return health


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _suppress_dangerous_mentions(text: str) -> str:
    return (text or "").replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")


def _completion_status_from_payload(payload: dict[str, Any], cleaned_output: str = "") -> str:
    status = str(payload.get("status") or "").strip().lower()
    if status:
        return status
    verdict = str(payload.get("verdict") or "").strip().upper()
    if verdict == "APPROVE":
        return "approved"
    if verdict == "REQUEST_CHANGES":
        return "request_changes"
    if verdict == "NEEDS_MANUAL_VERIFICATION":
        return "needs_manual_verification"
    upper = cleaned_output.upper()
    if "REQUEST_CHANGES" in upper or "REQUEST CHANGES" in upper:
        return "request_changes"
    if "NEEDS_MANUAL_VERIFICATION" in upper or "NEEDS MANUAL VERIFICATION" in upper:
        return "needs_manual_verification"
    if re.search(r"(?im)^\s*Verdict:\s*APPROVE\b", cleaned_output):
        return "approved"
    return "completed"


def _extract_completion_sentinel(cleaned_output: str) -> dict[str, Any] | None:
    for match in reversed(list(re.finditer(r"HERMES_JOB_DONE", cleaned_output or ""))):
        tail = cleaned_output[match.end():]
        start = tail.find("{")
        if start < 0:
            continue
        chunk = tail[start:start + 4000]
        depth = 0
        in_string = False
        escape = False
        raw_candidate = None
        for index, char in enumerate(chunk):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    raw_candidate = chunk[:index + 1]
                    break
        if raw_candidate is None:
            continue
        candidates = [re.sub(r"\s*\n\s*", " ", raw_candidate).strip(), raw_candidate]
        for candidate in candidates:
            try:
                payload = json.loads(candidate, strict=False)
            except Exception:
                continue
            return payload if isinstance(payload, dict) else {"status": "completed", "value": payload}
        return {
            "status": "blocked",
            "summary": "Completion sentinel was present but JSON could not be parsed.",
            "parse_error": True,
            "raw": _truncate_text(raw_candidate, 500),
        }
    return None


def _has_bare_completion_sentinel(cleaned_output: str) -> bool:
    return any(line.strip() == "HERMES_JOB_DONE" for line in (cleaned_output or "").splitlines())


def _idle_prompt_after_final_answer(cleaned_output: str) -> bool:
    if not cleaned_output:
        return False
    has_idle_prompt = (
        "Use /skills to list available skills" in cleaned_output
        or re.search(r"(?m)^›\s+", cleaned_output) is not None
    )
    has_worked_footer = "Worked for" in cleaned_output
    has_status_bar = "Context" in cleaned_output and "gpt-" in cleaned_output
    has_final_content = re.search(
        r"(?im)^\s*(Verdict:|Recommendation\s*$|Recommendation:|Commit\s*$|Commit:|Verification\s*Performed|Tests:|Result:|Blocking Issues\s*$|Remaining Risk\s*$)",
        cleaned_output,
    ) is not None
    return bool(has_idle_prompt and has_worked_footer and has_status_bar and has_final_content)


def _heuristic_completion_payload(cleaned_output: str) -> dict[str, Any]:
    upper = cleaned_output.upper()
    payload: dict[str, Any] = {}
    if "REQUEST_CHANGES" in upper or "REQUEST CHANGES" in upper:
        payload["verdict"] = "REQUEST_CHANGES"
    elif "NEEDS_MANUAL_VERIFICATION" in upper or "NEEDS MANUAL VERIFICATION" in upper:
        payload["verdict"] = "NEEDS_MANUAL_VERIFICATION"
    elif re.search(r"(?im)^\s*Verdict:\s*APPROVE\b", cleaned_output):
        payload["verdict"] = "APPROVE"
    payload["summary"] = _recent_activity(_clean_codex_output(cleaned_output, max_chars=2500), max_chars=220)
    return payload


def _detect_completion_state(output: str, tmux_alive: bool) -> dict[str, Any]:
    """Classify task completion separately from whether tmux is still alive."""
    cleaned = _strip_terminal_sequences(output or "")
    payload = _extract_completion_sentinel(cleaned)
    if payload is not None:
        status = _completion_status_from_payload(payload, cleaned)
        return {
            "is_complete": True,
            "method": "sentinel",
            "status": status,
            "phase": "idle_complete" if tmux_alive else "exited",
            "payload": payload,
        }
    if _has_bare_completion_sentinel(cleaned):
        return {
            "is_complete": True,
            "method": "bare_sentinel",
            "status": "needs_manual_verification",
            "phase": "idle_complete" if tmux_alive else "exited",
            "payload": {
                "status": "needs_manual_verification",
                "summary": "Completion sentinel was present without JSON.",
                "parse_error": True,
            },
        }
    if _idle_prompt_after_final_answer(cleaned):
        payload = _heuristic_completion_payload(cleaned)
        status = _completion_status_from_payload(payload, cleaned)
        return {
            "is_complete": True,
            "method": "heuristic_idle_prompt",
            "status": status,
            "phase": "idle_complete",
            "payload": payload,
        }
    if not tmux_alive:
        return {
            "is_complete": True,
            "method": "process_exit",
            "status": "exited",
            "phase": "exited",
            "payload": {},
        }
    return {
        "is_complete": False,
        "method": "active",
        "status": "running",
        "phase": "running",
        "payload": {},
    }


def _extract_labeled_blocks(cleaned_output: str, label_map: dict[str, tuple[str, ...]], max_chars: int = 500) -> dict[str, str]:
    all_labels = tuple(label for labels in label_map.values() for label in labels)
    result: dict[str, str] = {}
    lines = cleaned_output.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        lower = stripped.lower()
        matched_key = None
        matched_label = None
        for key, labels in label_map.items():
            for label in labels:
                if lower.startswith(label.lower()):
                    matched_key = key
                    matched_label = label
                    break
            if matched_key:
                break
        if not matched_key or not matched_label:
            index += 1
            continue

        value = stripped[len(matched_label):].strip()
        block = [value] if value else []
        index += 1
        while index < len(lines):
            continuation = lines[index].strip()
            if not continuation:
                break
            continuation_lower = continuation.lower()
            if any(continuation_lower.startswith(label.lower()) for label in all_labels):
                break
            if continuation.startswith(("•", "└", "Read ", "Search ", "Ran ", "Edited ", "List ", "Open ")):
                break
            block.append(continuation)
            index += 1
        if block:
            result[matched_key] = _truncate_text(" ".join(block), max_chars)
        else:
            result[matched_key] = ""
    return result


def _extract_worker_handoff(cleaned_output: str) -> dict[str, str]:
    return _extract_labeled_blocks(
        cleaned_output,
        {
            "result": ("Result:",),
            "files_changed": ("Files changed:", "Files Changed:"),
            "tests_run": ("Tests run:", "Tests:", "Verification:"),
            "known_blockers": ("Known blockers:", "Known Blockers:", "Blockers:", "Blocker:"),
            "notable_lessons": ("Notable lessons for future runs:", "Notable lessons:", "Lessons:"),
            "suggested_docs_skill_updates": (
                "Suggested docs/skill updates:",
                "Suggested docs updates:",
                "Docs/skill updates:",
                "Suggested skill updates:",
            ),
        },
        max_chars=700,
    )


def _extract_blockers_decisions(cleaned_output: str, handoff: dict[str, str] | None = None) -> str:
    handoff = handoff or {}
    parts: list[str] = []
    if handoff.get("known_blockers"):
        parts.append(handoff["known_blockers"])
    decision_blocks = _extract_labeled_blocks(
        cleaned_output,
        {"decisions": ("Decision:", "Decisions:", "Blocker:", "Risk:")},
        max_chars=320,
    )
    for value in decision_blocks.values():
        if value and value not in parts:
            parts.append(value)
    return _truncate_text(" | ".join(parts), 420)


def _distillation_recommendation(job: dict[str, Any], cleaned_output: str, status: str) -> dict[str, Any]:
    reasons: list[str] = []
    lowered = cleaned_output.lower()
    if status in {"blocked", "request_changes", "needs_manual_verification", "exited"}:
        reasons.append(f"terminal status was {status}")
    if len(cleaned_output) > 12000:
        reasons.append("long job transcript")
    if any(token in lowered for token in ("mcp", "xcodebuildmcp", "laravel-boost", "tool issue", "startup hangs")):
        reasons.append("tool or MCP issue mentioned")
    if any(token in lowered for token in ("repeated failure", "retry", "rerun", "fix loop", "review comment")):
        reasons.append("review/fix or repeated-failure loop mentioned")
    if job.get("final_handoff", {}).get("suggested_docs_skill_updates"):
        reasons.append("worker suggested docs or skill updates")
    return {
        "recommended": bool(reasons),
        "status": "pending" if reasons else "not_needed",
        "reasons": reasons,
    }


def _update_runtime_observations(job: dict[str, Any], output: str) -> None:
    cleaned = _clean_codex_output(output, max_chars=8000)
    if not cleaned:
        return
    handoff = _extract_worker_handoff(cleaned)
    key_findings = _extract_key_findings(cleaned)
    recent = _recent_activity(cleaned)
    if key_findings:
        job["key_findings"] = key_findings
    if recent:
        job["latest_activity"] = recent
    if handoff:
        existing = job.get("final_handoff") if isinstance(job.get("final_handoff"), dict) else {}
        job["final_handoff"] = {**existing, **handoff}
    if handoff.get("tests_run"):
        job["tests"] = handoff["tests_run"]
    else:
        tests = _extract_test_activity(cleaned, max_chars=500)
        if tests:
            job["tests"] = tests
    blockers = _extract_blockers_decisions(cleaned, handoff)
    if blockers:
        job["blockers"] = blockers


def _record_completion_state(job: dict[str, Any], state: dict[str, Any], output: str = "") -> None:
    job["status"] = state.get("status") or "completed"
    job["phase"] = state.get("phase") or "idle_complete"
    job["completion_detected_at"] = _now_iso()
    job["completion_method"] = state.get("method")
    job["completion_payload"] = state.get("payload") or {}
    if output:
        _update_runtime_observations(job, output)
    _resolve_launch_health(job)
    payload = job.get("completion_payload") or {}
    if isinstance(payload, dict):
        existing_tests = str(job.get("tests") or "").strip().lower()
        if payload.get("tests") and existing_tests in {"", "not_run", "not captured yet", "unknown"}:
            job["tests"] = str(payload["tests"])
        if payload.get("summary") and not job.get("key_findings"):
            job["key_findings"] = str(payload["summary"])
    cleaned = _clean_codex_output(output, max_chars=16000) if output else ""
    job["distillation"] = _distillation_recommendation(job, cleaned, job["status"])
    job["monitor_status"] = "stopped_after_completion"


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


def _job_status_summary(job: dict[str, Any], alive: bool) -> dict[str, Any]:
    return {
        "job_id": job.get("job_id"),
        "title": job.get("title"),
        "status": job.get("status"),
        "phase": job.get("phase") or ("running" if alive else job.get("status")),
        "tmux_alive": alive,
        "model": job.get("model"),
        "effort": job.get("effort"),
        "profile": job.get("profile"),
        "profile_summary": job.get("profile_summary"),
        "capability_enforcement": job.get("capability_enforcement"),
        "included_capabilities": job.get("included_capabilities") or [],
        "omitted_capabilities": job.get("omitted_capabilities") or [],
        "runner_limitations": job.get("runner_limitations") or [],
        "workspace_mode": job.get("workspace_mode"),
        "repo_path": job.get("repo_path"),
        "workspace_path": job.get("workspace_path"),
        "worktree_path": job.get("worktree_path"),
        "branch": job.get("branch"),
        "approval": job.get("approval"),
        "sandbox": job.get("sandbox"),
        "search": bool(job.get("search")),
        "codex_profile": job.get("codex_profile"),
        "codex_launch_flags": job.get("codex_launch_flags") or [],
        "latest_activity": job.get("latest_activity"),
        "key_findings": job.get("key_findings"),
        "tests": job.get("tests"),
        "blockers": job.get("blockers"),
        "health_check": job.get("health_check") or {},
        "final_handoff": job.get("final_handoff") or {},
        "distillation": job.get("distillation") or {},
        "attach_command": job.get("attach_command"),
        "discord_thread_target": job.get("discord_thread_target"),
        "discord_status_message_id": job.get("discord_status_message_id"),
        "log_path": job.get("log_path"),
    }


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
        "Files changed:",
        "Tests run:",
        "Known blockers:",
        "Notable lessons:",
        "Notable lessons for future runs:",
        "Suggested docs/skill updates:",
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


def _extract_test_activity(cleaned_output: str, max_chars: int = 260) -> str:
    labeled = _extract_labeled_blocks(cleaned_output, {"tests": ("Tests run:", "Tests:", "Verification:")}, max_chars=max_chars)
    if labeled.get("tests"):
        return labeled["tests"]
    lines: list[str] = []
    for raw in cleaned_output.splitlines():
        stripped = raw.strip()
        if stripped.startswith("└ "):
            stripped = stripped[2:].strip()
        if stripped.startswith("• "):
            stripped = stripped[2:].strip()
        lowered = stripped.lower()
        if lowered.startswith(("ran ", "run ")) and any(token in lowered for token in ("test", "pytest", "swift test", "xcodebuild")):
            lines.append(stripped)
    return _truncate_text("; ".join(lines[-3:]), max_chars)


def _render_monitor_status(job: dict[str, Any], alive: bool, output: str) -> str:
    phase = job.get("phase") or ("running" if alive else "exited")
    cleaned = _clean_codex_output(output, max_chars=5000)
    handoff = _extract_worker_handoff(cleaned)
    key_findings = _extract_key_findings(cleaned)
    recent = _recent_activity(cleaned)
    tests = handoff.get("tests_run") or _extract_test_activity(cleaned) or job.get("tests") or "not captured yet"
    blockers = _extract_blockers_decisions(cleaned, handoff) or job.get("blockers") or "none captured"
    workspace_summary = _workspace_summary(job)

    message = (
        f"**Codex job `{job['job_id']}` live status**\n"
        f"- **Title:** {job['title']}\n"
        f"- **Phase:** `{phase}`\n"
        f"- **Model/effort:** `{job.get('model')}` / `{job.get('effort')}`\n"
        f"- **Profile:** `{job.get('profile') or 'base-rich'}` — {_truncate_text(job.get('profile_summary') or '', 120)}\n"
        f"- **Launch policy:** `{_format_launch_policy(job, max_chars=170)}`\n"
        f"- **Capabilities:** {_format_capability_summary(job, max_chars=220)}\n"
        f"- **Repo:** `{job.get('repo_path') or 'n/a'}`\n"
        f"- **Worktree:** `{job.get('worktree_path') or 'n/a'}`\n"
        f"- **Branch:** `{job.get('branch') or 'n/a'}`\n"
        f"- **Workspace:** `{job['workspace_path']}`\n"
        f"- **Tests/verification:** {_truncate_text(tests, 150)}\n"
        f"- **Blockers/decisions:** {_truncate_text(blockers, 150)}\n"
        f"- **Attach:** `{job['attach_command']}`\n"
    )
    health = job.get("health_check") if isinstance(job.get("health_check"), dict) else {}
    if health and not health.get("resolved_at"):
        health_text = "\n".join(
            part for part in (
                health.get("summary"),
                health.get("recommendation"),
                f"Attach: {health.get('attach_command') or job.get('attach_command')}",
            ) if part
        )
        message += f"\n**Launch health check**\n```text\n{_truncate_text(health_text, 420)}\n```\n"
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
    handoff = _extract_worker_handoff(cleaned)
    key_findings = _extract_key_findings(cleaned, max_chars=650)
    recent = _recent_activity(cleaned, max_chars=450)
    workspace_summary = _workspace_summary(job, max_chars=360)
    commit = _last_commit_summary(job)
    detail_target = job.get("discord_thread_target") or job.get("discord_parent_target") or "#codex-control thread"
    tests = handoff.get("tests_run") or job.get("tests") or "not captured"
    blockers = _extract_blockers_decisions(cleaned, handoff) or job.get("blockers") or "none captured"

    message = (
        f"**Codex job `{job['job_id']}` {status}**\n"
        f"- **Title:** {job.get('title')}\n"
        f"- **Model/effort:** `{job.get('model')}` / `{job.get('effort')}`\n"
        f"- **Profile:** `{job.get('profile') or 'base-rich'}`\n"
        f"- **Branch:** `{job.get('branch') or 'n/a'}`\n"
        f"- **Tests/verification:** {_truncate_text(tests, 140)}\n"
        f"- **Blockers/decisions:** {_truncate_text(blockers, 140)}\n"
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


def _render_health_alert(job: dict[str, Any]) -> str:
    health = job.get("health_check") if isinstance(job.get("health_check"), dict) else {}
    summary = _suppress_dangerous_mentions(health.get("summary") or "Codex needs attention during startup.")
    recommendation = _suppress_dangerous_mentions(health.get("recommendation") or "Inspect the tmux session before taking action.")
    title = _suppress_dangerous_mentions(str(job.get("title") or ""))
    attach = health.get("attach_command") or job.get("attach_command") or "n/a"
    return _truncate_text(
        (
            f"**Codex job `{job['job_id']}` needs attention**\n"
            f"- **Title:** {title}\n"
            f"- **Health check:** `{health.get('kind') or 'unknown'}`\n"
            f"- **Phase:** `{health.get('phase') or job.get('phase') or 'running'}`\n"
            f"- **Summary:** {summary}\n"
            f"- **Recommendation:** {recommendation}\n"
            f"- **Attach:** `{attach}`"
        ),
        1900,
    )


def _send_health_alert_if_needed(job: dict[str, Any]) -> None:
    health = job.get("health_check") if isinstance(job.get("health_check"), dict) else {}
    kind = health.get("kind")
    if not kind or health.get("resolved_at"):
        return
    if job.get("health_alert_sent_at") and job.get("health_alert_kind") == kind:
        return
    target = job.get("summary_target") or job.get("discord_thread_target")
    if not target:
        return
    try:
        result = _send_message({"action": "send", "target": target, "message": _render_health_alert(job)})
        if result.get("error"):
            job["health_alert_error"] = result.get("error")
            _append_monitor_log(job, f"health alert error: {result.get('error')}")
            return
        job["health_alert_sent_at"] = _now_iso()
        job["health_alert_kind"] = kind
        job["health_alert_message_id"] = result.get("message_id")
        job["health_alert_target"] = target
        _append_monitor_log(job, f"health alert sent: {kind}")
    except Exception as exc:
        job["health_alert_error"] = str(exc)
        _append_monitor_log(job, f"health alert exception: {exc}")


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
        completion_state = _detect_completion_state(output, tmux_alive=alive)
        if completion_state.get("is_complete"):
            _record_completion_state(job, completion_state, output)
            if not alive:
                job["ended_at"] = _now_iso()
        else:
            if output:
                _update_launch_health(job, output, tmux_alive=alive, elapsed_seconds=max(time.time() - started, _job_elapsed_seconds(job)))
                _update_runtime_observations(job, output)
            _send_health_alert_if_needed(job)
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
        if completion_state.get("is_complete"):
            _send_completion_summary(job, output, status=job.get("status") or "completed")
            _save_job(job)
            return
        _save_job(job)
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
