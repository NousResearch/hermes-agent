"""End-of-turn beacon for Hazel's Codex/Hermes runtime.

Writes a current-state JSON file on turn start/end and posts Slack only for
active blockers, milestone commits, or upward test-threshold crossings.  The
file intentionally contains only structured operational metadata, no prompts,
credentials, or tool output.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

STATE_PATH = Path(os.environ.get("HERMES_TURN_STATUS_PATH", str(get_hermes_home() / "state" / "turn-status.json")))
DEFAULT_REPO = Path(os.environ.get("HERMES_ACTIVE_REPO") or os.environ.get("HERMES_WATCHER_REPO") or "/root/.hermes/skills/hazel-statusbrew-social")
SLACK_CHANNEL = os.environ.get("HERMES_TURN_BEACON_SLACK_CHANNEL", "C0B3AKEGZM3")
QUEUE_PATH = Path(os.environ.get("HERMES_SLICE_QUEUE_PATH", str(get_hermes_home() / "state" / "slice-queue.yaml")))
PAUSE_FLAG_PATH = Path(os.environ.get("HERMES_QUEUE_PAUSE_FLAG", str(get_hermes_home() / "state" / "queue-paused.flag")))
AUTO_DISPATCH_COUNT_PATH = Path(os.environ.get("HERMES_AUTO_DISPATCH_COUNT_PATH", str(get_hermes_home() / "state" / "auto-dispatch-count")))
AUTO_DISPATCH_LIMIT = int(os.environ.get("HERMES_AUTO_DISPATCH_LIMIT", "5"))
CHICAGO_TZ_NAME = "America/Chicago"
MILESTONE_COMMIT_RE = re.compile(r"^(complete|ship|integrate|Plan)", re.I)
TEST_THRESHOLD_START = 175
TEST_THRESHOLD_STEP = 25



def _now() -> tuple[dt.datetime, dt.datetime]:
    utc = dt.datetime.now(dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        chicago = utc.astimezone(ZoneInfo(CHICAGO_TZ_NAME))
    except Exception:
        chicago = utc.astimezone(dt.timezone(dt.timedelta(hours=-5), name=CHICAGO_TZ_NAME))
    return utc, chicago


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 10) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except Exception as exc:
        return 1, str(exc)


def _find_repo() -> Path:
    candidates: list[Path] = []
    env_repo = os.environ.get("HERMES_ACTIVE_REPO") or os.environ.get("HERMES_WATCHER_REPO")
    if env_repo:
        candidates.append(Path(env_repo))
    try:
        candidates.append(Path.cwd())
    except Exception:
        pass
    candidates.append(DEFAULT_REPO)
    for candidate in candidates:
        rc, top = _run(["git", "rev-parse", "--show-toplevel"], candidate)
        if rc == 0 and top:
            return Path(top).resolve()
    return DEFAULT_REPO.resolve()


def git_snapshot(repo: Path | None = None) -> dict[str, Any]:
    repo = (repo or _find_repo()).resolve()
    rc, head = _run(["git", "rev-parse", "--short=12", "HEAD"], repo)
    if rc != 0:
        head = ""
    rc, subject = _run(["git", "log", "-1", "--pretty=%s"], repo)
    if rc != 0:
        subject = ""
    rc, status = _run(["git", "status", "--porcelain"], repo)
    dirty_lines = [ln for ln in status.splitlines() if ln.strip()] if rc == 0 else []
    return {
        "active_repo": str(repo),
        "last_commit_sha": head.strip(),
        "last_commit_subject": subject.strip(),
        "working_tree": "clean" if not dirty_lines else f"dirty ({len(dirty_lines)} files)",
        "dirty_count": len(dirty_lines),
    }


def commit_count_since(repo: Path, start_sha: str | None) -> int:
    if not start_sha:
        return 0
    rc, count = _run(["git", "rev-list", "--count", f"{start_sha}..HEAD"], repo)
    if rc == 0 and count.isdigit():
        return int(count)
    return 0



def _read_counter() -> int:
    try:
        raw = AUTO_DISPATCH_COUNT_PATH.read_text(encoding="utf-8").strip()
        return max(0, int(raw or "0"))
    except Exception:
        return 0


def _write_counter(value: int) -> None:
    AUTO_DISPATCH_COUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUTO_DISPATCH_COUNT_PATH.with_suffix(".tmp")
    tmp.write_text(str(max(0, value)) + "\n", encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    tmp.replace(AUTO_DISPATCH_COUNT_PATH)


def reset_auto_dispatch_counter() -> None:
    _write_counter(0)


def _queue_empty_payload() -> dict[str, Any]:
    return {"queue": []}


def _locked_queue_file():
    if not QUEUE_PATH.exists():
        return None
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = QUEUE_PATH.open("r+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def _load_queue_from_handle(fh) -> list[dict[str, Any]]:
    fh.seek(0)
    raw = fh.read()
    data = yaml.safe_load(raw) if raw.strip() else _queue_empty_payload()
    queue = data.get("queue") if isinstance(data, dict) else None
    if not isinstance(queue, list):
        return []
    return [item for item in queue if isinstance(item, dict)]


def _atomic_write_queue_locked(queue: list[dict[str, Any]]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=QUEUE_PATH.name + ".", suffix=".tmp", dir=str(QUEUE_PATH.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump({"queue": queue}, f, sort_keys=False, allow_unicode=True)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        tmp.replace(QUEUE_PATH)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _pop_next_slice_locked() -> tuple[dict[str, Any] | None, int]:
    fh = _locked_queue_file()
    if fh is None:
        return None, -1
    try:
        queue = _load_queue_from_handle(fh)
        if not queue:
            return None, 0
        item = queue.pop(0)
        _atomic_write_queue_locked(queue)
        return item, len(queue)
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _write_prompt_temp(slice_item: dict[str, Any]) -> Path:
    scope = str(slice_item.get("scope") or "")
    title = str(slice_item.get("title") or "Untitled slice")
    slice_id = str(slice_item.get("id") or "slice-unknown")
    prompt = f"Auto-dispatched slice {slice_id}: {title}\n\n{scope}".strip() + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix="hermes-auto-slice-", suffix=".prompt", dir="/tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(prompt)
    os.chmod(tmp_name, stat.S_IRUSR | stat.S_IWUSR)
    return Path(tmp_name)


def _spawn_next_turn(prompt_path: Path, slice_item: dict[str, Any]) -> int:
    hermes_bin = os.environ.get("HERMES_AUTO_DISPATCH_CMD") or shutil.which("hermes") or "/usr/local/bin/hermes"
    env = os.environ.copy()
    env["HERMES_AUTO_DISPATCH"] = "1"
    env["HERMES_AUTO_DISPATCH_SLICE_ID"] = str(slice_item.get("id") or "")
    if os.environ.get("HERMES_AUTO_DISPATCH_DRY_RUN") == "1":
        log_path = get_hermes_home() / "logs" / "turn-beacon-auto-dispatch-dry-run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(prompt_path.read_text(encoding="utf-8"))
        prompt_path.unlink(missing_ok=True)
        return 0
    cmd = f'{shlex.quote(hermes_bin)} --yolo -z "$(cat {shlex.quote(str(prompt_path))})"; rm -f {shlex.quote(str(prompt_path))}'
    proc = subprocess.Popen(["/usr/bin/env", "bash", "-lc", cmd], env=env, cwd=str(Path.cwd()), start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return int(proc.pid)


def _post_auto_dispatch(slice_item: dict[str, Any], remaining: int, count: int) -> str | None:
    slice_id = str(slice_item.get("id") or "slice-unknown")
    title = str(slice_item.get("title") or "Untitled slice")
    text = f':robot_face: Auto-dispatching slice {slice_id} "{title}" ({remaining} remaining in queue, {count} of {AUTO_DISPATCH_LIMIT} consecutive)'
    posted = _slack_api("chat.postMessage", {"channel": SLACK_CHANNEL, "text": text, "unfurl_links": False, "unfurl_media": False})
    return _slack_permalink(SLACK_CHANNEL, posted["ts"])


def _post_queue_empty() -> str | None:
    # Queue-empty pings are intentionally silenced. Chris only gets active needs,
    # milestone commits, and upward 25-multiple test threshold crossings.
    return None


def _append_slack_sidecar(kind: str, permalink: str | None, utc: dt.datetime, extra: dict[str, Any] | None = None) -> None:
    if not permalink:
        return
    sidecar = STATE_PATH.with_suffix(".slack.json")
    payload = {"kind": kind, "permalink": permalink, "posted_at_iso": utc.isoformat()}
    if extra:
        payload.update(extra)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def _maybe_auto_dispatch(state: dict[str, Any], utc: dt.datetime) -> tuple[dict[str, Any], bool]:
    if state.get("status") != "idle_awaiting_prompt":
        reset_auto_dispatch_counter()
        return state, False
    if PAUSE_FLAG_PATH.exists():
        state["auto_dispatch_paused_reason"] = "queue-paused.flag set"
        return state, False
    count = _read_counter()
    if count >= AUTO_DISPATCH_LIMIT:
        state["auto_dispatch_paused_reason"] = "5 consecutive auto-dispatches reached — human ack required"
        return state, False
    if not QUEUE_PATH.exists():
        state["auto_dispatch_paused_reason"] = "queue empty"
        return state, False
    item, remaining = _pop_next_slice_locked()
    if remaining == -1:
        state["auto_dispatch_paused_reason"] = "slice-queue.yaml locked"
        return state, False
    if item is None:
        state["auto_dispatch_paused_reason"] = "queue empty"
        return state, False

    new_count = count + 1
    _write_counter(new_count)
    prompt_path = _write_prompt_temp(item)
    permalink = None
    # Routine auto-dispatch pings are silenced by policy.
    pid = _spawn_next_turn(prompt_path, item)
    running = dict(state)
    running.update({
        "status": "running",
        "turn_id": f"auto-{item.get('id') or 'slice'}",
        "summary": "Auto-dispatched next queued slice.",
        "next_pending_intent": f"Running queued slice {item.get('id')}: {item.get('title')}",
        "auto_dispatch": {
            "slice_id": item.get("id"),
            "title": item.get("title"),
            "remaining_queue": remaining,
            "consecutive_count": new_count,
            "prompt_path": str(prompt_path),
            "pid": pid,
            "slack_permalink": permalink,
        },
        "auto_dispatch_paused_reason": None,
    })
    return running, True


def _log_slack_failure(utc: dt.datetime, turn_id: str | None, message: str) -> None:
    log_path = get_hermes_home() / "logs" / "turn-beacon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{utc.isoformat()} {message} for turn {turn_id}\n")


def _read_existing() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    tmp.replace(STATE_PATH)
    os.chmod(STATE_PATH, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def _parse_test_counts(messages: list[dict[str, Any]] | None) -> tuple[int, int]:
    text = "\n".join(str(m.get("content", "")) for m in (messages or []) if isinstance(m, dict) and m.get("role") == "tool")
    passed = failed = 0
    for number, word in re.findall(r"(\d+)\s+(passed|failed|errors?)", text, flags=re.I):
        n = int(number)
        key = word.lower()
        if key == "passed":
            passed = max(passed, n)
        else:
            failed = max(failed, n)
    return passed, failed


def _one_line(value: str, limit: int = 180) -> str:
    clean = re.sub(r"\s+", " ", value or "").strip()
    return clean[: limit - 1] + "…" if len(clean) > limit else clean


def _summary_from_response(final_response: str | None) -> str:
    text = re.sub(r"```.*?```", "", final_response or "", flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Turn ended without a visible assistant summary."
    parts = re.split(r"(?<=[.!?])\s+", text)
    return _one_line(" ".join(parts[:3]), 500)


def _next_intent(final_response: str | None, user_message: str | None) -> str:
    text = final_response or ""
    for pat in (r"(?:^|\n)\s*Next intent\s*:\s*(.+)", r"(?:^|\n)\s*Next\s*:\s*(.+)", r"(?:^|\n)\s*Next step\s*:\s*(.+)"):
        m = re.search(pat, text, flags=re.I)
        if m:
            return _one_line(m.group(1), 180)
    if user_message:
        return _one_line("Await Chris's next prompt to continue: " + user_message, 180)
    return "Await Chris's next prompt."


def _slack_api(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_TOKEN")
    if not token:
        env_path = get_hermes_home() / ".env"
        if env_path.exists():
            for raw in env_path.read_text(errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = val.strip().strip('"').strip("'")
        token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_TOKEN")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN/SLACK_TOKEN missing")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"https://slack.com/api/{method}",
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {body}")
    return body


def _slack_permalink(channel: str, ts: str) -> str | None:
    token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_TOKEN")
    if not token:
        # _slack_api loads ~/.hermes/.env before posting; mirror that token after post.
        token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_TOKEN")
    if not token:
        return None
    query = urllib.parse.urlencode({"channel": channel, "message_ts": ts})
    req = urllib.request.Request(f"https://slack.com/api/chat.getPermalink?{query}", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if body.get("ok"):
        return body.get("permalink")
    return None


def _test_threshold_bucket(passed: int) -> int:
    if passed < TEST_THRESHOLD_START:
        return 0
    return ((passed - TEST_THRESHOLD_START) // TEST_THRESHOLD_STEP) * TEST_THRESHOLD_STEP + TEST_THRESHOLD_START


def _crossed_test_threshold(prev_passed: int, current_passed: int) -> int | None:
    prev_bucket = _test_threshold_bucket(prev_passed)
    current_bucket = _test_threshold_bucket(current_passed)
    if current_bucket > prev_bucket:
        return current_bucket
    return None


def _terminal_post_reasons(state: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    if state.get("status") == "blocked":
        return ["active need"]
    reasons: list[str] = []
    subject = str(state.get("last_commit_subject") or "")
    if state.get("last_commit_sha") and state.get("last_commit_sha") != previous.get("last_commit_sha") and MILESTONE_COMMIT_RE.match(subject):
        reasons.append("milestone commit")
    threshold = _crossed_test_threshold(int(previous.get("tests_pass", 0) or 0), int(state.get("tests_pass", 0) or 0))
    if threshold is not None:
        reasons.append(f"tests crossed {threshold} passed")
    return reasons


def _post_slack(state: dict[str, Any]) -> str | None:
    status = state.get("status")
    if status == "blocked":
        lines = [
            f":warning: Turn ended blocked — `{status}`",
            f"Blocker: {state.get('blocker') or 'unspecified'}",
        ]
    else:
        lines = [":white_check_mark: Hazel allowed milestone/status signal"]
    lines.extend([
        f"Last: `{state.get('last_commit_sha') or 'unknown'}` {state.get('last_commit_subject') or ''}".rstrip(),
        f"Tests: {state.get('tests_pass', 0)}p / {state.get('tests_fail', 0)}f · Tree: {state.get('working_tree') or 'unknown'}",
        "Next intent: " + str(state.get("next_pending_intent") or "Await Chris's next prompt."),
    ])
    posted = _slack_api("chat.postMessage", {"channel": SLACK_CHANNEL, "text": "\n".join(lines), "unfurl_links": False, "unfurl_media": False})
    return _slack_permalink(SLACK_CHANNEL, posted["ts"])


def mark_running(turn_id: str, user_message: str | None = None) -> dict[str, Any]:
    if os.environ.get("HERMES_AUTO_DISPATCH") != "1":
        reset_auto_dispatch_counter()
    repo = _find_repo()
    git = git_snapshot(repo)
    utc, chi = _now()
    state = {
        "status": "running",
        "turn_id": turn_id,
        "ended_at_iso": utc.isoformat(),
        "ended_at_iso_chicago": chi.isoformat(),
        "last_commit_sha": git["last_commit_sha"],
        "last_commit_subject": git["last_commit_subject"],
        "commits_this_turn": 0,
        "tests_pass": 0,
        "tests_fail": 0,
        "active_repo": git["active_repo"],
        "working_tree": git["working_tree"],
        "next_pending_intent": _next_intent(None, user_message),
        "blocker": None,
        "summary": "Turn is running.",
    }
    _atomic_write_state(state)
    return {"repo": repo, "start_sha": git["last_commit_sha"], "state": state}


def mark_finished(
    *,
    turn_id: str,
    start_sha: str | None = None,
    status: str = "idle_awaiting_prompt",
    blocker: str | None = None,
    final_response: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    user_message: str | None = None,
) -> dict[str, Any]:
    previous = _read_existing()
    repo = _find_repo()
    git = git_snapshot(repo)
    utc, chi = _now()
    tests_pass, tests_fail = _parse_test_counts(messages)
    state = {
        "status": status,
        "turn_id": turn_id,
        "ended_at_iso": utc.isoformat(),
        "ended_at_iso_chicago": chi.isoformat(),
        "last_commit_sha": git["last_commit_sha"],
        "last_commit_subject": git["last_commit_subject"],
        "commits_this_turn": commit_count_since(repo, start_sha),
        "tests_pass": tests_pass,
        "tests_fail": tests_fail,
        "active_repo": git["active_repo"],
        "working_tree": git["working_tree"],
        "next_pending_intent": _next_intent(final_response, user_message),
        "blocker": blocker,
        "summary": _summary_from_response(final_response),
    }
    state, dispatched = _maybe_auto_dispatch(state, utc)
    _atomic_write_state(state)
    if dispatched:
        return state
    post_reasons = _terminal_post_reasons(state, previous)
    if post_reasons:
        state["slack_post_reasons"] = post_reasons
        try:
            permalink = _post_slack(state)
            _append_slack_sidecar("terminal", permalink, utc, {"turn_id": turn_id, "reasons": post_reasons})
        except Exception as exc:
            _log_slack_failure(utc, turn_id, f"Slack post failed: {exc}")
    return state


def classify_status(turn_exit_reason: str | None, interrupted: bool, final_response: str | None, last_msg_role: str | None) -> tuple[str, str | None]:
    reason = turn_exit_reason or "unknown"
    lower_resp = (final_response or "").lower()
    if interrupted:
        return "blocked", "turn interrupted before completion"
    if last_msg_role == "tool":
        return "blocked", "turn ended after a tool result without a final assistant response"
    if any(key in reason for key in ("max_iterations", "error_near_max_iterations", "empty_response_exhausted", "budget_exhausted")):
        return "blocked", reason
    if "blocker:" in lower_resp[:1200]:
        # Short, conservative blocker extraction for explicit assistant reports.
        m = re.search(r"blocker:\s*(.+)", final_response or "", flags=re.I)
        return "blocked", _one_line(m.group(1), 180) if m else "blocker reported in final response"
    return "idle_awaiting_prompt", None
