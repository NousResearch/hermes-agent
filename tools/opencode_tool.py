"""opencode_task: hand a bounded repository job to OpenCode, under governance.

One call replaces the five terminal steps a model needed before (worktree,
key, model flag, background run, diff). The job runs in the background and
the completion notification carries the review report: diff stat, patch,
test output and OpenCode's own output tail.

Governance (15-10-2026, Michael):

* the tool rides the terminal grant plus the ``opencode`` CLI grant
  (``tool_policy._decide_opencode_tool_access``), so an account that may not
  run ``opencode run`` through terminal never sees this tool;
* workdir, task text and test command are checked like terminal arguments
  (``tool_policy._opencode_argument_decision``) by the dispatcher, and again
  here against the live context, failing closed when the context cannot be
  resolved;
* the profile folderguard checks the workdir;
* OpenCode itself runs confined (see ``tools/opencode_task_runner.py``):
  read-only filesystem outside the workdir, the caller's denied paths masked,
  per-job OpenCode state, global slot limit.

16-10-2026 (Michael): OpenCode works directly in the workdir (no worktree),
sees keys and credentials the caller may read, and runs up to
HERMES_OPENCODE_MAX_SLOTS jobs at once (default 12). Only what the caller has
no access to stays masked.

07-11-2026 (Michael, ideas from DeepSeek's dsh-superagent): every task gets a
short handoff contract (stop by impact, one concrete question, honest status,
a closing STATUS line), and ``continue_job`` sends a follow-up into the
OpenCode session of an earlier job, in the same workdir or worktree, so a
fix-up round keeps the context instead of re-reading the repository. Passing
gateway failures are resumed by the runner.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_default_hermes_root, get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

TOOL_NAME = "opencode_task"
DEFAULT_MODEL = "omniroute/synthwave-auto"
# PlainBoard (the agent team on this VPS): its MCP bridge takes one bearer token per caller and acts as
# that caller's own PlainBoard account. The token lives in the caller's profile .env.
PLAINBOARD_MCP_URL = "http://127.0.0.1:18156/mcp"
RUNNER = Path(__file__).resolve().with_name("opencode_task_runner.py")

# Appended to every task. Kept short: OpenCode reads it on every job.
HANDOFF_CONTRACT = """

---
How to work and hand back (from the agent that delegated this job):
- Work unattended to a reviewable result. Nobody answers questions while you run.
- Stop by impact, not effort: finish local changes completely; if the job needs a design
  decision, a credential, or access you do not have, stop and ask one concrete question.
- Stay inside the task's scope; mention anything else you noticed in one line, do not fix it.
- Verify what you changed (run the relevant tests or the command that proves it). Never claim
  a check you did not run. Do not commit, push or deploy.
- End your final message with these lines:
  STATUS: ready | blocked | failed | dropped
  (ready = done and verified; blocked = needs the QUESTION answered; failed = could not do it;
  dropped = nothing to do, e.g. already done or not reproducible)
  QUESTION: <one concrete question, only when blocked>
  Before them, at most five lines: what changed and how you verified it."""


def _job_root() -> Path:
    return get_hermes_home() / "cache/opencode-tasks"


def load_previous_job(job_id: str, owner_email: str) -> tuple[Optional[dict], Optional[str]]:
    """The earlier job a follow-up continues, or an error. Same owner only."""
    job_id = (job_id or "").strip()
    if not job_id or "/" in job_id or job_id.startswith("."):
        return None, "continue_job must be the id of an earlier opencode_task job."
    path = _job_root() / job_id / "job.json"
    try:
        prev = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None, f"continue_job: no earlier job {job_id} found."
    if (prev.get("owner_email") or "") != (owner_email or ""):
        return None, "continue_job: that job belongs to someone else."
    state = Path(prev.get("state_dir") or (_job_root() / job_id / "state"))
    if not (state / "data" / "opencode" / "opencode.db").is_file():
        return None, f"continue_job: job {job_id} has no OpenCode session to continue."
    prev["_state_dir"] = str(state)
    from tools.opencode_task_runner import session_facts
    prev["_session"] = session_facts(state).get("session", "")
    if not prev["_session"]:
        return None, f"continue_job: job {job_id} has no OpenCode session to continue."
    return prev, None


def _host_home() -> Path:
    # Sandbox masks refer to the host's passwd home, even under the multiplexer.
    if sys.platform == "win32":
        return Path.home()
    import pwd
    return Path(pwd.getpwuid(os.getuid()).pw_dir) if hasattr(os, "getuid") else Path.home()


def _opencode_binary() -> str:
    from hermes_cli.synthpulse_settings import setting
    configured = setting("opencode.binary")
    return str(configured or (_host_home() / ".npm-global/lib/node_modules/opencode-ai/bin/opencode.exe"))


def _max_slots() -> int:
    from hermes_cli.synthpulse_settings import setting
    try:
        return max(1, int(setting("opencode.max_slots", 12)))
    except (TypeError, ValueError):
        return 12


# Candidate locations with credentials or other people's state. Each one is
# masked only when the caller may not read it through Hermes; an owner sees
# them all, like in their own terminal.
CANDIDATE_MASK_DIRS = (
    ".hermes", ".hermes-mux", ".ssh", ".gnupg", ".claude", ".codex", ".aws", ".azure",
    ".azure-devops", ".kube", ".docker", ".config/gcloud", ".config/gh", ".config/bunq",
    ".config/wealth-automation", ".local/share/opencode", ".IdentityService", ".apify",
    "bunq-agentic", "ObsidianVault", ".config/synthwave/clients",
)
CANDIDATE_MASK_FILES = (
    ".env", ".netrc", ".git-credentials", ".npmrc", ".pypirc", ".claude.json",
    ".config/opencode/opencode.jsonc", ".config/opencode/opencode.json",
)
# Hermes refuses this mount to every account, so OpenCode never sees it.
HARD_MASK_DIRS = ("/run/synthpulse/secrets",)

OPENCODE_TASK_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Hand a bounded repository job (feature, refactor, tests, review, script) to OpenCode, "
        "a coding agent, and get a review report back. Runs in the background directly in the "
        "workdir: it edits the files in place. You are notified with the changed files, diff, test "
        "output and OpenCode's output when it finishes. Use it for code work that does not need your "
        "session context; run several in parallel (one per task, on separate files or folders, or with "
        "worktree=true for parallel jobs in one repository). "
        "OpenCode can read what the caller may read, keys included, and writes only inside the workdir."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Self-contained instructions: goal, files, constraints, how to verify."},
            "workdir": {"type": "string", "description": "Absolute path of the repository or folder to work in."},
            "model": {"type": "string", "description": f"provider/model for OpenCode (default {DEFAULT_MODEL})."},
            "background": {"type": "boolean", "description": "Run in the background and notify on completion (default true)."},
            "worktree": {"type": "boolean", "description": "Work in a separate git worktree on branch opencode/<id> instead of the live tree (default false). Use it for parallel jobs in one repository, or when the live tree has work in progress."},
            "test_command": {"type": "string", "description": "Optional command run in the workdir after OpenCode, e.g. 'pytest -q tests/x'."},
            "timeout": {"type": "integer", "description": "Seconds before OpenCode is stopped (default 3600, max 7200)."},
            "full_context": {"type": "boolean", "description": "Load every installed skill and MCP server into OpenCode (default false). That costs about ten times more tokens per step and is slower; use it only when the job needs a specific skill or MCP tool."},
            "continue_job": {"type": "string", "description": "Id of an earlier opencode_task job (opencode_job in its result). Sends this task as a follow-up into that job's OpenCode session, in the same workdir or worktree, so it keeps everything it already read. Use it for review fixes, failed tests and answers to its QUESTION instead of starting a fresh job. workdir may then be the same workdir or omitted."},
        },
        "required": ["task"],
    },
}


def _current_context():
    try:
        from hermes_cli.dashboard_governance.context import current_governance_context
        return current_governance_context(), None
    except Exception as exc:
        logging.getLogger(__name__).error("_current_context: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(exc).__name__), exc.__traceback__))
        return None, exc


def _governed_without_context() -> bool:
    """True when this process should have a governance context but has none."""
    from hermes_cli.dashboard_governance.context import GOVERNANCE_CONTEXT_ENV
    if os.environ.get(GOVERNANCE_CONTEXT_ENV):
        return True
    try:
        from tools import profile_scope_bridge
        mod = profile_scope_bridge.load_profile_scope()
        if mod is not None and mod.is_scoped(mod.resolve_profile()):
            return True
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("_governed_without_context: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return True
    return False


def check_governance(args: dict[str, Any]):
    """(context, error). Fail closed whenever the caller's access cannot be resolved."""
    ctx, exc = _current_context()
    if exc is not None:
        return None, "Blocked by governance: access could not be resolved (fail-closed)."
    if ctx is None:
        if _governed_without_context():
            return None, "Blocked by governance: no governance context for a governed profile (fail-closed)."
        return None, None
    try:
        from hermes_cli.dashboard_governance.tool_policy import (
            tool_allowed_for_context, tool_arguments_allowed_for_context,
        )
        tool = tool_allowed_for_context(ctx, TOOL_NAME, registry)
        if not tool.allowed:
            return ctx, f"Blocked by governance: {tool.reason}."
        decision = tool_arguments_allowed_for_context(ctx, TOOL_NAME, args)
        if not decision.allowed:
            detail = f" ({decision.detail})" if getattr(decision, "detail", "") else ""
            return ctx, f"Blocked by governance: {decision.reason}{detail}."
    except Exception:
        logger.debug("opencode_task governance check failed", exc_info=True)
        return ctx, "Blocked by governance: policy check failed (fail-closed)."
    return ctx, None


def check_folderguard(workdir: str) -> Optional[str]:
    try:
        from tools import profile_scope_bridge
        mod = profile_scope_bridge.load_profile_scope()
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("check_folderguard: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        mod = None
    if mod is None:
        return "Blocked: profile scope guard unavailable (fail-closed)." if _governed_without_context() else None
    try:
        profile = mod.resolve_profile()
        ok, reason = mod.check_path(profile, workdir, "write")
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("check_folderguard: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return "Blocked: profile scope check failed (fail-closed)."
    return None if ok else f"Blocked by folder scope: {reason}"


def _git_top(workdir: Path) -> Optional[Path]:
    try:
        done = subprocess.run(["git", "-C", str(workdir), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    return Path(done.stdout.strip())


def _literal_path(glob: str) -> Optional[str]:
    """A denied glob that names one concrete place (optionally with /**)."""
    raw = os.path.expanduser(str(glob).strip())
    for suffix in ("/**", "/*"):
        raw = raw.removesuffix(suffix)
    if not raw.startswith("/") or any(ch in raw for ch in "*?["):
        return None
    return raw.rstrip("/") or None


def _caller_may_read(access, path: str) -> bool:
    if access is None:
        return True
    try:
        from hermes_cli.dashboard_governance.tool_policy import decide_tool_argument_access
        return decide_tool_argument_access(access, "read_file", {"path": path}).allowed
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("_caller_may_read: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return False


def mask_plan(access) -> tuple[list[str], list[str], list[str]]:
    """(dirs to mask, files to mask, OpenCode deny patterns) for this caller.

    Everything the caller may read stays visible; a location is masked only
    when the policy refuses it to them."""
    home = str(_host_home())
    dirs = list(HARD_MASK_DIRS)
    files: list[str] = []
    patterns: list[str] = ["/run/synthpulse/secrets/**"]
    for rel in CANDIDATE_MASK_DIRS:
        full = os.path.join(home, rel)
        if not _caller_may_read(access, os.path.join(full, "probe")):
            dirs.append(full)
            patterns.append(full + "/**")
    for rel in CANDIDATE_MASK_FILES:
        full = os.path.join(home, rel)
        if not _caller_may_read(access, full):
            files.append(full)
            patterns.append(full)
    if access is not None:
        globs = set(getattr(access.grants, "file_denied_globs", ()) or ()) | set(
            getattr(access.deny, "file_denied_globs", ()) or ())
        allow = set(getattr(access.grants, "file_allow_globs", ()) or ())
        for glob in sorted(globs - allow):
            literal = _literal_path(glob)
            if literal:
                if _caller_may_read(access, literal):
                    continue
                (dirs if Path(literal).is_dir() or glob.endswith(("/**", "/*")) else files).append(literal)
                patterns.append(literal + ("/**" if Path(literal).is_dir() else ""))
            elif str(glob).startswith("**/") or str(glob).startswith("*"):
                patterns.append(str(glob))
        for raw in sorted(set(getattr(access.deny, "file_read_roots", ()) or ())):
            literal = _literal_path(raw)
            if literal:
                dirs.append(literal)
                patterns.append(literal + "/**")
    return sorted(set(dirs)), sorted(set(files)), list(dict.fromkeys(patterns))


RUNTIME_PATHS = (".npm-global", ".config/opencode", ".local/bin", ".gitconfig", ".bun")


def home_layout(access) -> dict:
    """Blacklist accounts and the ungoverned owner see the home read-only with
    secrets masked. Whitelist accounts see an empty home with only their own
    read roots mounted back: what they cannot read through Hermes, OpenCode
    cannot read either."""
    if access is None or getattr(access, "access_mode", "") == "blacklist":
        return {"home_mode": "masked"}
    roots = sorted(str(Path(os.path.expanduser(r))) for r in (getattr(access.grants, "file_read_roots", ()) or ())
                   if r and r != "*" and not any(ch in str(r) for ch in "*?["))
    return {"home_mode": "allowlist", "read_roots": roots,
            "runtime_paths": [str(_host_home() / p) for p in RUNTIME_PATHS]}


def user_mcp_names() -> list[str]:
    """MCP servers the host's OpenCode config defines. Inline config merges with
    it, so an empty "mcp" does not switch them off; each needs enabled=false."""
    import re

    names: set[str] = set()
    for name in ("opencode.jsonc", "opencode.json"):
        path = _host_home() / ".config" / "opencode" / name
        try:
            text = re.sub(r"(?m)^\s*//.*$", "", path.read_text(encoding="utf-8-sig"))
            names.update((json.loads(text).get("mcp") or {}).keys())
        except (OSError, ValueError, AttributeError):
            continue
    return sorted(names)


def plainboard_token(access) -> str:
    """The caller's own PlainBoard bridge token, only when governance grants MCP 'plainboard'.

    Read from this profile's .env (never from the shared process environment), so a partner's job
    gets the partner's token and a caller without the grant gets none."""
    if access is not None:
        try:
            if not access.allows("mcp_servers", "plainboard"):
                return ""
        except Exception:
            return ""
    try:
        text = (get_hermes_home() / ".env").read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("PLAINBOARD_MCP_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def opencode_config(deny_patterns: list[str], lean: bool = True, plainboard: bool = False) -> dict:
    """Inline config (highest precedence): provider, the caller's denies, no edits outside the workdir.

    lean (07-11-2026): the host's MCP servers off. With the external skills also
    off (runner env) a step costs ~13k input tokens instead of ~134k.
    plainboard (22-11-2026): the PlainBoard bridge stays on, also when lean, with the caller's token."""
    rules = {"*": "allow"}
    bash_rules = {"*": "allow", "*opencode*run*": "deny"}
    for pattern in deny_patterns:
        rules[pattern] = "deny"
        stem = pattern.replace("/**", "").lstrip("*").strip("/")
        if stem:
            bash_rules[f"*{stem}*"] = "deny"
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "omniroute": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "OmniRoute (Synthwave gateway)",
                "options": {"baseURL": "http://127.0.0.1:20128/v1", "apiKey": "{env:OMNIROUTE_API_KEY}"},
                "models": {m: {"name": m} for m in (
                    "synthwave-auto", "synthpulse-sonnet", "synthpulse-opus",
                    "auto/best-coding", "auto/coding", "auto/best-coding-fast", "auto/best-reasoning")},
            }
        },
        "mcp": _mcp_section(lean, plainboard),
        "permission": {
            "external_directory": "deny",
            "read": dict(rules),
            "edit": dict(rules),
            "bash": bash_rules,
            "webfetch": "allow",
        },
    }


def _mcp_section(lean: bool, plainboard: bool) -> dict:
    host = user_mcp_names()
    mcp: dict = {name: {"enabled": False} for name in host} if lean else {}
    if plainboard:
        mcp["plainboard"] = {"type": "remote", "url": PLAINBOARD_MCP_URL, "enabled": True, "oauth": False,
                             "headers": {"Authorization": "Bearer {env:PLAINBOARD_MCP_TOKEN}"}}
    elif "plainboard" in host:
        # The host's own entry would run with the host user's token: never for a caller without the grant.
        mcp["plainboard"] = {"enabled": False}
    return mcp


def resolve_continuation(continue_job: str, workdir: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """The earlier job a follow-up continues, checked against the caller and workdir."""
    ctx, _ = _current_context()
    access = getattr(ctx, "access", None) if ctx is not None else None
    owner = getattr(getattr(access, "subject", None), "email", "") if access is not None else ""
    prev, err = load_previous_job(continue_job, owner)
    if err:
        return None, err
    given = (workdir or "").strip()
    if given and os.path.realpath(os.path.expanduser(given)) != prev["workdir"]:
        return None, f"continue_job {continue_job} worked in {prev['workdir']}; pass that workdir or none."
    return prev, None


def continuation_args(continue_job, workdir, model):
    """(earlier job or None, workdir, model, error) for an optional follow-up."""
    if not continue_job:
        return None, workdir, model, None
    prev, err = resolve_continuation(continue_job, workdir)
    if err:
        return None, workdir, model, err
    return prev, prev["workdir"], model or prev.get("model"), None


def apply_continuation(job: dict, prev: Optional[dict]) -> None:
    """Point a new job at the earlier job's OpenCode session and worktree."""
    if prev is None:
        return
    job.update({"continue_of": prev["id"], "state_dir": prev["_state_dir"],
                "continue_session": prev.get("_session", "")})
    if prev.get("worktree"):
        job.update({"branch": prev["branch"], "worktree": prev["worktree"], "reuse_worktree": True})


def wants_lean(full_context: Optional[bool], prev: Optional[dict]) -> bool:
    """Lean unless asked for full context; a follow-up keeps its session's choice."""
    if full_context is None:
        return (prev or {}).get("lean", True)
    return not full_context


def opencode_task(task: str, workdir: str, model: Optional[str] = None, background: bool = True,
                  test_command: Optional[str] = None, timeout: Optional[int] = None, worktree: bool = False,
                  task_id: Optional[str] = None, session_id: Optional[str] = None,
                  continue_job: Optional[str] = None, full_context: Optional[bool] = None) -> str:
    task = (task or "").strip()
    if not task:
        return tool_error("opencode_task needs a task.")
    if os.environ.get("OPENCODE_RECURSION_GUARD"):
        return tool_error("Refused: already running inside OpenCode (recursion guard). Do the work directly.")
    prev, workdir, model, err = continuation_args(continue_job, workdir, model)
    if err:
        return tool_error(err)
    raw_workdir = (workdir or "").strip()
    if not raw_workdir or not os.path.isabs(os.path.expanduser(raw_workdir)):
        return tool_error("workdir must be an absolute path.")
    resolved = Path(os.path.realpath(os.path.expanduser(raw_workdir)))
    if not resolved.is_dir():
        return tool_error(f"workdir does not exist: {resolved}")
    args = {"task": task, "workdir": str(resolved), "test_command": test_command or ""}
    ctx, blocked = check_governance(args)
    if blocked:
        return tool_error(blocked)
    blocked = check_folderguard(str(resolved))
    if blocked:
        return tool_error(blocked)
    opencode_bin = _opencode_binary()
    if not Path(opencode_bin).is_file():
        return tool_error(f"OpenCode binary not found at {opencode_bin}.")
    # The runner is a Hermes control-plane process on the host. A profile whose
    # terminal is sandboxed (docker, ssh) must not gain a host process this way.
    try:
        from tools.terminal_tool import _get_env_config
        env_type = _get_env_config().get("env_type", "local")
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("opencode_task: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        env_type = "unknown"
    if env_type != "local":
        return tool_error(f"opencode_task needs the host terminal backend; this profile uses '{env_type}'.")

    access = getattr(ctx, "access", None) if ctx is not None else None
    top = _git_top(resolved)
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    job: dict[str, Any] = {
        "id": job_id, "task": task + ("" if prev else HANDOFF_CONTRACT), "brief": task, "workdir": str(resolved), "model": (model or DEFAULT_MODEL).strip(),
        "opencode_bin": opencode_bin, "home": str(_host_home()), "slot_dir": str(get_default_hermes_root() / "run/opencode-slots"), "slots": _max_slots(),
        "slot_wait": 600, "timeout": max(60, min(int(timeout or 3600), 7200)),
        "test_command": (test_command or "").strip(),
        "path": ":".join([str(_host_home() / ".npm-global" / "bin"), str(_host_home() / ".local" / "bin"),
                          "/usr/local/bin", "/usr/bin", "/bin"]),
        "owner_email": getattr(getattr(access, "subject", None), "email", "") if access is not None else "",
    }
    if top is not None:
        job["git_top"] = str(top)
        # The repository root (and so .git) is writable only when the caller may
        # write there; otherwise OpenCode can write the workdir alone.
        top_writable = True
        if access is not None and top != resolved:
            from hermes_cli.dashboard_governance.tool_policy import decide_tool_argument_access
            top_writable = decide_tool_argument_access(
                access, "write_file", {"path": str(top / ".opencode-task-probe")}).allowed
        job["git_writable"] = top_writable
        if worktree:
            if access is not None:
                from hermes_cli.dashboard_governance.tool_policy import decide_tool_argument_access
                probe = decide_tool_argument_access(
                    access, "write_file", {"path": str(top / ".opencode-worktrees" / "probe")})
                if not probe.allowed:
                    return tool_error(f"Blocked by governance: the repository root {top} is outside your write scope.")
            job.update({"branch": f"opencode/{job_id}", "worktree": str(top / ".opencode-worktrees" / job_id)})
    elif worktree:
        return tool_error("worktree=true needs a git repository as workdir.")
    apply_continuation(job, prev)
    mask_dirs, mask_files, deny_patterns = mask_plan(access)
    # Never mask the place OpenCode has to work in.
    keep = Path(job.get("worktree") or resolved)
    job["mask_dirs"] = [d for d in mask_dirs if not str(keep).startswith(d.rstrip("/") + "/") and str(keep) != d]
    job["mask_files"] = mask_files
    job["deny_patterns"] = deny_patterns
    job["lean"] = wants_lean(full_context, prev)
    pb_token = plainboard_token(access)
    job["plainboard"] = bool(pb_token)
    job["opencode_config"] = opencode_config(deny_patterns, lean=job["lean"], plainboard=job["plainboard"])
    job.update(home_layout(access))

    run_dir = get_hermes_home() / "cache/opencode-tasks" / job_id
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (run_dir / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    key = os.environ.get("OMNIROUTE_API_KEY", "")
    if key:
        key_path = run_dir / "omniroute.key"
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key)

    if pb_token:
        fd = os.open(run_dir / "plainboard.token", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(pb_token)

    command = f"{shlex.quote(sys.executable)} -I {shlex.quote(str(RUNNER))} {shlex.quote(str(run_dir))}"
    from tools.terminal_tool import terminal_tool
    result = terminal_tool(command=command, background=bool(background), timeout=None if background else job["timeout"] + 900,
                           task_id=task_id, session_id=session_id, workdir=str(resolved),
                           notify_on_complete=bool(background), _host_local=True,
                           _completion_output_chars=40000)
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        data = {"output": result}
    data.update({"opencode_job": job_id, "model": job["model"], "workdir": str(resolved),
                 "worktree": job.get("worktree", ""), "branch": job.get("branch", "")})
    if background:
        data["note"] = ("OpenCode is working in the background directly in the workdir; you will be notified "
                        "with the changed files and diff. Review them before committing.")
    return json.dumps(data)


def check_opencode_requirements() -> bool:
    # The worker depends on Linux bubblewrap; keep discovery importable on
    # other hosts without advertising an unavailable sandbox.
    return sys.platform == "linux" and Path(_opencode_binary()).is_file() and RUNNER.is_file()


def _handle_opencode_task(args, **kw):
    return opencode_task(
        task=args.get("task"), workdir=args.get("workdir"), model=args.get("model"),
        background=args.get("background", True), test_command=args.get("test_command"),
        timeout=args.get("timeout"), worktree=bool(args.get("worktree", False)), task_id=kw.get("task_id"), session_id=kw.get("session_id"),
        continue_job=args.get("continue_job"), full_context=args.get("full_context"),
    )


registry.register(
    name=TOOL_NAME,
    toolset="delegation",
    schema=OPENCODE_TASK_SCHEMA,
    handler=_handle_opencode_task,
    check_fn=check_opencode_requirements,
    emoji="🧑‍💻",
    max_result_size_chars=60_000,
)
