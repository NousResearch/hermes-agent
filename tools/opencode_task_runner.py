"""Run one governed OpenCode job and print a review report.

Started by ``tools/opencode_tool.py`` as a background process, never by the
model directly. Everything that decides what OpenCode may touch arrives in the
job file the tool wrote after its governance checks; the model only supplied
the task text.

What this process guarantees (15-10-2026):

* a global limit on concurrent OpenCode jobs (flock slots, default 12);
* OpenCode works directly in the workdir (16-10-2026); the report lists
  exactly the files this job changed, with their diff. With worktree=true
  (18-10-2026) it works in its own git worktree on branch opencode/<id>
  instead, so parallel jobs in one repository cannot collide;
* OpenCode runs inside bubblewrap: the filesystem is read-only except the
  workdir and OpenCode's own per-job state, and what the caller may not read
  is masked with empty mounts. That holds for OpenCode's bash tool too, which
  a permission file alone cannot promise;
* the OmniRoute key travels in a one-shot file that is deleted before
  OpenCode starts and is masked inside the sandbox;
* the recursion guard is set, so a nested ``opencode run`` is refused.

07-11-2026 (ideas taken from DeepSeek's dsh-superagent): a passing model or
gateway failure (combo retry limit, 408/429/5xx, unavailable, reset) no
longer kills the job. The runner resumes the same OpenCode session up to
three times with a growing backoff, so the work done so far is kept. A job
can continue the session of an earlier job (``continue_job``), so a fix-up
round does not re-read the repository. OpenCode ends with a structured
``STATUS:`` line (ready, blocked, failed, dropped) and at most one
``QUESTION:``; the report starts with them. The report is also kept as
``report.md`` and ``result.json`` in the job directory.

The report (stdout) is what the completion notification delivers to the agent.
"""

from __future__ import annotations

import logging

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import re

REPORT_DIFF_LINES = 400
RETRY_BACKOFF = (15, 45, 90)
TRANSIENT_PATTERNS = (
    "combo retry limit", "model is unavailable", "upstream request failed", "overloaded",
    "rate limit", "too many requests", "timed out", "timeout", "econnreset", "econnrefused",
    "socket hang up", "connection reset", "connection refused", "fetch failed", "network",
    "[408]", "[429]", "[500]", "[502]", "[503]", "[504]", "[520]", "[522]", "[524]", "[529]",
)
STATUS_RE = re.compile(r"^\s*\**STATUS\**\s*:\s*\**\s*(ready|blocked|failed|dropped)\b", re.IGNORECASE | re.MULTILINE)
QUESTION_RE = re.compile(r"^\s*\**QUESTION\**\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
CONTINUE_NOTE = ("The previous model call failed on a passing gateway error. Continue the same task "
                 "from where you stopped; do not redo finished steps.")
REPORT_TAIL_CHARS = 6000
SNAPSHOT_FILE_LIMIT = 20000


def _tail(text: str, limit: int = REPORT_TAIL_CHARS) -> str:
    # OpenCode prints its whole permission rule set when it refuses a path;
    # the refusal line matters, the multi-kilobyte rule dump does not.
    text = "\n".join("[permission rules omitted: OpenCode refused an access outside its scope]"
                     if len(line) > 1500 and '"permission"' in line else line
                     for line in (text or "").splitlines())
    return text if len(text) <= limit else "[...]\n" + text[-limit:]


def _run(argv, cwd=None, env=None, timeout=None, input_text=None):
    try:
        # stdin is never inherited: `opencode run` reads a piped stdin as part of
        # the message and would wait for EOF forever.
        extra = {"input": input_text} if input_text is not None else {"stdin": subprocess.DEVNULL}
        done = subprocess.run(argv, cwd=cwd, env=env, timeout=timeout,
                              capture_output=True, text=True, **extra, check=False)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return 124, out + f"\n[timeout after {timeout}s]"
    except OSError as exc:
        return 127, str(exc)


def acquire_slot(slot_dir: Path, slots: int, wait_seconds: int):
    """Hold one of *slots* flock slots; None when all stay busy for wait_seconds."""
    import fcntl

    slot_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        for index in range(max(1, slots)):
            handle = open(slot_dir / f"slot-{index}.lock", "w", encoding="utf-8")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle.write(str(os.getpid()))
                handle.flush()
                return handle
            except OSError:
                handle.close()
        if time.monotonic() >= deadline:
            return None
        time.sleep(2)


def _git(repo: Path, *args, timeout=120):
    return _run(["git", "-C", str(repo), *args], timeout=timeout)


def _hash(path: Path) -> str:
    try:
        digest = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def snapshot(root: Path, git_top: str = "") -> dict:
    """Content fingerprints of the files a job could change.

    In a git repo only the dirty and untracked files are hashed (the rest is
    HEAD); elsewhere every file under root, up to SNAPSHOT_FILE_LIMIT."""
    state = {}
    if git_top:
        code, out = _git(Path(root), "status", "--porcelain=v1", "-z", "-uall", "--", ".")
        if code == 0:
            entries = out.split("\0")
            index = 0
            while index < len(entries):
                entry = entries[index]
                index += 1
                if len(entry) < 4:
                    continue
                if entry[0] in "RC":
                    index += 1  # the rename source follows
                rel = entry[3:]
                state[rel] = _hash(Path(git_top) / rel)
            return state
    skip = {".git", "node_modules", ".venv", "venv", "__pycache__"}
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            count += 1
            if count > SNAPSHOT_FILE_LIMIT:
                return state
            full = Path(dirpath) / name
            state[str(full.relative_to(root))] = _hash(full)
    return state


def changed_files(before: dict, after: dict) -> list[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def state_dir(job: dict, run_dir: Path) -> Path:
    """OpenCode's per-job state; a continued job reuses the earlier job's."""
    return Path(job.get("state_dir") or (run_dir / "state"))


def stream_facts(output: str) -> dict:
    """Session id, last error, last answer text and step count from ``opencode run --format json``.

    OpenCode 2 writes one JSON event per line (step_start, text, tool_use, step_finish, error, each
    with sessionID). Reading that stream replaces reading OpenCode 1's internal sqlite database,
    whose layout OpenCode 2 changed. Non-JSON lines (stderr, notes) are skipped."""
    facts = {"session": "", "error": "", "text": "", "steps": 0}
    texts: dict[str, list[str]] = {}
    order: list[str] = []
    for line in (output or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        facts["session"] = event.get("sessionID") or facts["session"]
        kind = event.get("type")
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        if kind == "step_finish":
            facts["steps"] += 1
        elif kind == "text":
            mid = part.get("messageID") or ""
            if mid not in texts:
                texts[mid] = []
                order.append(mid)
            texts[mid].append(part.get("text") or "")
        elif kind == "error":
            err = event.get("error") or {}
            facts["error"] = str(err.get("message") or err.get("type") or err)[:600] if isinstance(err, dict) else str(err)[:600]
    for mid in reversed(order):
        text = "\n".join(texts[mid]).strip()
        if text:
            facts["text"] = text
            break
    return facts


def readable_output(output: str) -> str:
    """The JSON event stream as short lines for the report: tool calls, answers, errors."""
    lines = []
    for raw in (output or "").splitlines():
        raw = raw.strip()
        try:
            event = json.loads(raw) if raw.startswith("{") else None
        except ValueError:
            event = None
        if not isinstance(event, dict):
            if raw:
                lines.append(raw)
            continue
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        kind = event.get("type")
        if kind == "tool_use":
            state = part.get("state") or {}
            arg = json.dumps(state.get("input") or {}, ensure_ascii=False)[:160]
            lines.append(f"[{part.get('tool')}] {arg} -> {state.get('status')}")
        elif kind == "text" and (part.get("text") or "").strip():
            lines.append(part["text"].strip())
        elif kind == "error":
            lines.append(f"ERROR: {json.dumps(event.get('error'), ensure_ascii=False)[:400]}")
    return "\n".join(lines)


def is_transient(error: str) -> bool:
    low = (error or "").lower()
    return bool(low) and any(p in low for p in TRANSIENT_PATTERNS)


def parse_status(text: str) -> tuple[str, str]:
    """(status, question) from OpenCode's closing lines; '' when absent."""
    found = STATUS_RE.findall(text or "")
    status = found[-1].lower() if found else ""
    q = QUESTION_RE.findall(text or "")
    question = q[-1].strip() if q else ""
    if question.lower() in {"none", "-", "n/a", "geen"}:
        question = ""
    return status, question[:500]


def prepare_worktree(job: dict) -> tuple[Path, dict]:
    """A fresh worktree on branch opencode/<id>; returns (work path, git info)."""
    top = Path(job["git_top"])
    worktree = Path(job["worktree"])
    branch = job["branch"]
    if job.get("reuse_worktree") and worktree.is_dir():
        workdir = Path(job["workdir"])
        rel = workdir.relative_to(top) if workdir != top else Path(".")
        _, base = _git(worktree, "rev-parse", "HEAD")
        return worktree / rel, {"git": True, "top": str(worktree), "repo": str(top), "worktree": str(worktree),
                                "branch": branch, "base": base.strip(), "writable": False}
    worktree.parent.mkdir(parents=True, exist_ok=True)
    exclude = top / ".git" / "info" / "exclude"
    try:
        marker = "/.opencode-worktrees/"
        current = exclude.read_text(encoding="utf-8-sig") if exclude.exists() else ""
        if marker not in current:
            exclude.parent.mkdir(parents=True, exist_ok=True)
            with open(exclude, "a", encoding="utf-8") as fh:
                fh.write(("\n" if current and not current.endswith("\n") else "") + marker + "\n")
    except OSError:
        pass
    code, out = _git(top, "worktree", "add", "-b", branch, str(worktree), "HEAD")
    if code != 0:
        raise RuntimeError("git worktree add failed: " + out.strip()[-800:])
    _, base = _git(top, "rev-parse", "HEAD")
    workdir = Path(job["workdir"])
    rel = workdir.relative_to(top) if workdir != top else Path(".")
    info = {"git": True, "top": str(worktree), "repo": str(top), "worktree": str(worktree),
            "branch": branch, "base": base.strip(), "writable": False}
    return worktree / rel, info


def commit_worktree(git_info: dict, task: str) -> None:
    worktree = Path(git_info["worktree"])
    _git(worktree, "add", "-A")
    title = "opencode_task: " + " ".join(task.split())[:60]
    _run(["git", "-C", str(worktree), "-c", "user.name=OpenCode task", "-c", "user.email=opencode@localhost",
          "commit", "-q", "--no-verify", "-m", title])


def bwrap_argv(job: dict, run_dir: Path, workroot: Path, git_info: dict) -> list[str]:
    home = Path(job["home"])
    argv = ["bwrap", "--die-with-parent", "--unshare-pid", "--new-session",
            # no-tmp: ok — this is a private tmpfs inside the Linux sandbox, not host scratch.
            "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    visible = None
    if job.get("home_mode") == "allowlist":
        # Whitelist accounts: an empty home with only their read roots and
        # OpenCode's own runtime mounted back, read-only.
        argv += ["--tmpfs", str(home)]
        visible = []
        for raw in list(job.get("read_roots", [])) + list(job.get("runtime_paths", [])):
            if Path(raw).exists():
                argv += ["--ro-bind", raw, raw]
                visible.append(raw.rstrip("/"))

    def shown(raw: str) -> bool:
        return visible is None or any(raw == v or raw.startswith(v + "/") for v in visible)

    for raw in job.get("mask_dirs", []):
        if Path(raw).is_dir() and shown(raw):
            argv += ["--tmpfs", raw]
    for raw in job.get("mask_files", []):
        if Path(raw).is_file() and shown(raw):
            argv += ["--ro-bind", "/dev/null", raw]
    state = state_dir(job, run_dir)
    for sub in ("data", "state", "tmp"):
        (state / sub).mkdir(parents=True, exist_ok=True)
    cache = home / ".cache" / "opencode"
    cache.mkdir(parents=True, exist_ok=True)
    argv += ["--bind", str(state), str(state), "--bind", str(cache), str(cache)]
    if git_info.get("worktree"):
        repo = Path(git_info["repo"])
        worktree = Path(git_info["worktree"])
        argv += ["--bind", str(worktree), str(worktree)]
        for extra in (repo / ".git" / "worktrees" / worktree.name, repo / ".git" / "objects"):
            if extra.is_dir():
                argv += ["--bind", str(extra), str(extra)]
    elif git_info.get("git") and git_info.get("writable"):
        top = Path(git_info["top"])
        argv += ["--bind", str(top), str(top)]
    else:
        argv += ["--bind", str(workroot), str(workroot)]
    # Files inside the workdir the caller may not read (a denied .env, say).
    for found in _denied_files_under(workroot, job.get("deny_patterns", [])):
        argv += ["--ro-bind", "/dev/null", str(found)]
    # Masks again on top of the writable bind, for denied folders inside it.
    for raw in job.get("mask_dirs", []):
        if Path(raw).is_dir() and raw.startswith(str(workroot).rstrip("/") + "/"):
            argv += ["--tmpfs", raw]
    argv += ["--chdir", str(workroot)]
    return argv


def _denied_files_under(root: Path, patterns, limit: int = 20000):
    patterns = [str(p) for p in patterns or () if p]
    if not patterns:
        return
    checked = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", ".venv", "venv", "__pycache__"}]
        for name in filenames:
            checked += 1
            if checked > limit:
                return
            full = os.path.join(dirpath, name)
            # Match complete paths: reducing **/.ssh/** to its final **
            # masks every ordinary source file in the coding workspace.
            if any(fnmatch.fnmatch(full, pat) for pat in patterns):
                yield Path(full)


def sandbox_env(job: dict, run_dir: Path, key: str) -> dict:
    home = job["home"]
    state = state_dir(job, run_dir)
    env = {
        "HOME": home,
        "PATH": job.get("path") or "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "TERM": "dumb",
        "XDG_DATA_HOME": str(state / "data"),
        "XDG_STATE_HOME": str(state / "state"),
        "XDG_CACHE_HOME": str(Path(home) / ".cache"),
        "XDG_CONFIG_HOME": str(Path(home) / ".config"),
        "TMPDIR": "/tmp",  # no-tmp: ok — points to the private sandbox tmpfs above.
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENCODE_RECURSION_GUARD": "1",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_CONFIG_CONTENT": json.dumps(job["opencode_config"]),
        "OMNIROUTE_API_KEY": key,
    }
    plainboard = take_plainboard_token(run_dir)
    if plainboard:
        env["PLAINBOARD_MCP_TOKEN"] = plainboard
    if job.get("lean", True):
        # ~2,000 installed skills otherwise ride along in every step (~90k tokens).
        env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] = "1"
        env["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] = "1"
    return env


def take_plainboard_token(run_dir: Path) -> str:
    """The caller's PlainBoard bridge token written by opencode_task; read once, then removed."""
    path = run_dir / "plainboard.token"
    try:
        token = path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""
    try:
        path.unlink()
    except OSError:
        pass
    return token


def take_key(run_dir: Path) -> str:
    path = run_dir / "omniroute.key"
    try:
        key = path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        key = ""
    try:
        path.unlink()
    except OSError:
        pass
    if not key:
        fallback = Path(os.path.expanduser("~/.config/opencode/omniroute.key"))
        try:
            key = fallback.read_text(encoding="utf-8-sig").strip()
        except OSError:
            key = ""
    return key


def collect_git(git_info: dict, workroot: Path, files: list[str]) -> str:
    """Diff of exactly the files this job changed, against HEAD (untracked as new)."""
    if not files:
        return ""
    top = Path(git_info["top"])
    _, patch = _run(["git", "-C", str(top), "diff", "HEAD", "--", *files], timeout=120)
    _, untracked = _run(["git", "-C", str(top), "ls-files", "--others", "--exclude-standard", "--", *files], timeout=60)
    for rel in [u for u in untracked.splitlines() if u.strip()]:
        _, extra = _run(["git", "-C", str(top), "diff", "--no-index", "--", "/dev/null", rel], timeout=60)
        patch += extra
    lines = patch.splitlines()
    shown = "\n".join(lines[:REPORT_DIFF_LINES])
    if len(lines) > REPORT_DIFF_LINES:
        shown += f"\n[... {len(lines) - REPORT_DIFF_LINES} more diff lines; run git diff in the workdir]"
    return shown


def lock_session(job: dict, run_dir: Path):
    """One run per OpenCode session: a second follow-up is refused, not queued."""
    import fcntl

    state = state_dir(job, run_dir)
    state.mkdir(parents=True, exist_ok=True)
    handle = open(state / ".session.lock", "w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def run_with_resume(job: dict, run_dir: Path, workroot: Path, git_info: dict, env: dict, deadline: float):
    """Run OpenCode; resume the same session after a passing gateway failure.

    Returns (exit code, output, session facts, session id, retry notes)."""
    session = job.get("continue_session", "")
    message = job["task"]
    retries: list[str] = []
    outputs: list[str] = []
    while True:
        # OpenCode 2: a private server per run (no shared background service inside the sandbox),
        # permissions decided by the inline rules (--auto approves what they do not deny), one JSON
        # event per line on stdout; the workdir is the sandbox's cwd (v2 has no --dir).
        argv = bwrap_argv(job, run_dir, workroot, git_info) + [
            job["opencode_bin"], "run", "--standalone", "--auto", "--format", "json", "-m", job["model"]]
        if session:
            argv += ["--session", session]
        argv += [message]
        code, output = _run(argv, cwd=str(workroot), env=env, timeout=max(60, int(deadline - time.time())))
        outputs.append(output)
        facts = stream_facts(output)
        session = facts["session"] or session
        error = facts["error"] or (output.strip()[-600:] if code not in (0, 124) else "")
        if code == 124 or not is_transient(error) or len(retries) >= len(RETRY_BACKOFF) or not session:
            break
        wait = RETRY_BACKOFF[len(retries)]
        if time.time() + wait + 120 > deadline:
            break
        retries.append(f"{' '.join(error.split())[:160]} (retry {len(retries) + 1} after {wait}s)")
        time.sleep(wait)
        message = CONTINUE_NOTE
    if facts.get("error") and code == 0:
        code = 1  # OpenCode exits 0 even when its last model call failed
    return code, "\n".join(outputs), facts, session, retries


def main(job_dir: str) -> int:
    run_dir = Path(job_dir)
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8-sig"))
    started = time.time()
    report = [f"# opencode_task {job['id']}", f"task: {' '.join(job.get('brief', job['task']).split())[:300]}"]

    key = take_key(run_dir)
    if os.environ.get("OPENCODE_RECURSION_GUARD"):
        print("\n".join(report + ["status: refused", "reason: already inside an OpenCode run (recursion guard)"]))
        return 2
    if not shutil.which("bwrap"):
        print("\n".join(report + ["status: refused", "reason: bubblewrap (bwrap) is not installed; refusing to run OpenCode unconfined"]))
        return 2
    slot = acquire_slot(Path(job["slot_dir"]), int(job.get("slots", 12)), int(job.get("slot_wait", 600)))
    if slot is None:
        print("\n".join(report + ["status: busy",
                                  f"reason: all {job.get('slots', 12)} OpenCode slots stayed busy; retry later"]))
        return 75
    session_lock = lock_session(job, run_dir)
    if session_lock is None:
        slot.close()
        print("\n".join(report + ["STATUS: failed",
                                  f"reason: job {job.get('continue_of') or job['id']} is still running in that session; wait for its report"]))
        return 75
    try:
        if job.get("worktree"):
            try:
                workroot, git_info = prepare_worktree(job)
            except Exception as exc:
                logging.getLogger(__name__).error("main: operation failed",
                    exc_info=(RuntimeError, RuntimeError(type(exc).__name__), exc.__traceback__))
                print("\n".join(report + ["status: failed", f"reason: {exc}"]))
                return 1
        else:
            workroot = Path(job["workdir"])
            git_info = {"git": bool(job.get("git_top")), "top": job.get("git_top", ""),
                        "writable": bool(job.get("git_writable"))}
        before = snapshot(workroot, git_info["top"])
        env = sandbox_env(job, run_dir, key)
        key = ""
        code, output, facts, session, retries = run_with_resume(
            job, run_dir, workroot, git_info, env, deadline=started + int(job.get("timeout", 3600)))
        report.append(f"opencode exit: {code}")
        test_out = ""
        tcode = None
        if job.get("test_command"):
            test_argv = bwrap_argv(job, run_dir, workroot, git_info) + ["bash", "-c", job["test_command"]]
            env_test = dict(env)
            env_test.pop("OMNIROUTE_API_KEY", None)
            tcode, test_out = _run(test_argv, cwd=str(workroot), env=env_test, timeout=int(job.get("test_timeout", 900)))
            report.append(f"test exit: {tcode}")
        status, question = parse_status(facts.get("text", ""))
        if facts.get("error") or code not in (0,):
            status = "failed" if status != "blocked" else status
        if not status:
            status = "ready" if code == 0 else "failed"
        head = [f"STATUS: {status}"] + ([f"QUESTION: {question}"] if question else [])
        if status == "ready" and tcode not in (None, 0):
            status = "failed"
            head = ["STATUS: failed", "NOTE: OpenCode reported ready, but test_command failed (see Test output)."]
        head += [f"session: {session or '-'} (follow-up: opencode_task continue_job={job['id']})"]
        if retries:
            head += ["gateway retries:"] + [f"- {r}" for r in retries]
        report[2:2] = head
        after = snapshot(workroot, git_info["top"])
        changed = changed_files(before, after)
        where = f"worktree {workroot}" if git_info.get("worktree") else f"edited in place in {workroot}"
        report += ["", f"## Changed files ({where})"] + (changed or ["(none)"])
        if git_info.get("git"):
            patch = collect_git(git_info, workroot, changed)
            if patch:
                report += ["", "## Diff (against HEAD, only files this job changed)", patch]
            if git_info.get("worktree"):
                if changed:
                    commit_worktree(git_info, job["task"])
                repo = git_info["repo"]
                report += ["", f"branch: {git_info['branch']} (base {git_info['base'][:12]})",
                           f"merge after review: git -C {repo} merge {git_info['branch']}",
                           f"discard: git -C {repo} worktree remove --force {git_info['worktree']} && "
                           f"git -C {repo} branch -D {git_info['branch']}"]
            else:
                report += ["", "Nothing is committed: review, then commit or `git checkout -- <file>` to undo."]
        if job.get("test_command"):
            report += ["", "## Test output", _tail(test_out)]
        report += ["", "## OpenCode output", _tail(readable_output(output))]
        took = int(time.time() - started)
        report.append(f"\nstatus: {'ok' if code == 0 else 'failed'} in {took}s")
        text = "\n".join(report)
        (run_dir / "report.md").write_text(text, encoding="utf-8")
        (run_dir / "result.json").write_text(json.dumps({
                "id": job["id"], "status": status, "exit": code, "seconds": took, "retries": len(retries),
                "session": session, "steps": facts.get("steps", 0), "changed": len(changed),
                "question": question, "continue_of": job.get("continue_of", ""),
                "test_exit": tcode}, indent=1), encoding="utf-8")
        print(text)
        return 0 if code == 0 else 1
    finally:
        slot.close()
        session_lock.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: opencode_task_runner.py <job-dir>", file=sys.stderr)
        sys.exit(64)
    sys.exit(main(sys.argv[1]))
