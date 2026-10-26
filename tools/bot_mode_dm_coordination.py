"""``message_agent`` coordination: a fan-out to several teammates and cancelling one's own request.

Fan-out: each target is resolved and admitted exactly like a single ``message_agent`` call, but its
reply runner is not started on its own. One combined runner (``bot_mode_dm.py --fan-out <spec>``)
runs those per-target runners side by side and prints ONE result once every one has settled or hit
its bound, so the sender wakes once with a typed outcome per teammate instead of N separate turns.

Cancel: the sender withdraws its own queued request, or stops the turn that request started, on the
teammate's owner through the same ``prompt.cancel`` a viewer's Stop uses. Only the request's author
may: the receipt records who sent it.

The runner half (``run_fan_out``) is stdlib only: it runs as a background process whose stdout is
the sender's completion notification, and each sub-runner boots Hermes itself.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

# Every sub-runner bounds its own wait (the live lane at 30 min, the relay waiter by its budget);
# this only reaps one that outlives that, reported as pending rather than lost.
FAN_OUT_WAIT_SECONDS = 32 * 60
_HEX_ID_RE = re.compile(r"[0-9a-f]{32,64}")


def fan_out(targets: list[str], content: str, sender: Any, *, expects_reply: bool) -> str:
    """Dispatch ``content`` to every target; with a reply expected, start the one combined runner."""
    from tools.bot_mode_dm import _dispatch

    collected: list[dict] = []
    acks = []
    for target in targets:
        ack = json.loads(_dispatch(target, content, sender, wake=collected if expects_reply else "none"))
        acks.append({"to": f"@{target}", **ack})
    sent = [ack for ack in acks if not ack.get("error")]
    if not collected:
        return json.dumps({"status": "queued" if sent else "failed", "fan_out": acks,
                           "detail": ("Nothing to wait for. " if sent else "Nothing was sent. ")
                           + "Do not resend to a teammate whose entry has no error."})
    started = _start_combined(collected, sender)
    if started.get("error"):
        # The local admissions stand (their owner runs them); only the joint wake is lost. A peer
        # entry was never sent: its runner owned the send.
        for ack in acks:
            if any(entry["delivery_id"] == ack.get("delivery_id") and entry["dm_file"] for entry in collected):
                ack.update(status="failed", error="NOT sent: the fan-out runner did not start")
        return json.dumps({"status": "queued", "fan_out": acks, "notification_error": started["error"],
                           "detail": "The replies will NOT wake you. Do not resend to entries without an error."})
    detail = (f"One combined completion carries every reply (or why not) once all {len(collected)} settle "
              "or hit their bound — do NOT wait, poll or resend; finish your turn.")
    if started.get("reply_delivery") == "poll":
        detail = started["detail"]
    return json.dumps({"status": "queued", "fan_out": acks, "process_id": started.get("process_id"),
                       "reply_delivery": started.get("reply_delivery"), "detail": detail})


def _start_combined(collected: list[dict], sender: Any) -> dict:
    from tools.bot_mode_dm import REPLY_COMPLETION_CHARS, _dm_dir, _spawn_delivery, _unlink_dm_file

    fd, spec = tempfile.mkstemp(prefix="fanout-", suffix=".json", dir=_dm_dir(), text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump([{k: entry[k] for k in ("command", "label", "delivery_id")} for entry in collected], stream)
    argv = [sys.executable, str(Path(__file__).resolve().parent / "bot_mode_dm.py"), "--fan-out", spec]
    if sys.platform == "win32":
        argv = [part.replace("\\", "/") for part in argv]  # Git Bash runs the tracked command (see _delivery_command)
    started = json.loads(_spawn_delivery(shlex.join(argv), f"{len(collected)} teammates", dm_file=spec,
                                         task_id=sender.task_id, agent=sender.agent,
                                         completion_chars=REPLY_COMPLETION_CHARS * len(collected)))
    if started.get("error"):
        for entry in collected:
            if entry["dm_file"]:
                _unlink_dm_file(entry["dm_file"])
    return started


def run_fan_out(spec_path: str) -> int:
    """Runner entry: run each target's runner concurrently, print one JSON outcome list."""
    with open(spec_path, encoding="utf-8") as stream:
        entries = json.load(stream)
    with contextlib.suppress(OSError):
        os.unlink(spec_path)
    results: list[dict] = [{} for _ in entries]

    def run(index: int, entry: dict) -> None:
        base = {"to": entry["label"], "delivery_id": entry["delivery_id"]}
        try:
            done = subprocess.run(shlex.split(entry["command"]), capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=FAN_OUT_WAIT_SECONDS, check=False)
        except subprocess.TimeoutExpired:
            results[index] = {**base, "status": "pending", "detail": "No outcome within the bound. Do not resend."}
            return
        results[index] = {**base, **_outcome(done.returncode, done.stdout)}

    threads = [threading.Thread(target=run, args=item, daemon=True) for item in enumerate(entries)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    settled = sum(result.get("status") == "settled" for result in results)
    print(json.dumps({"replies": results, "settled": settled, "of": len(results)}, ensure_ascii=False))
    return 0 if settled == len(results) else 1


def _outcome(returncode: int, stdout: str) -> dict:
    """A sub-runner's stdout as a typed outcome: the live lane prints one JSON payload; the relay
    waiter and ``hermes peer dm`` print the reply (or the failure) as text."""
    text = stdout.strip()
    try:
        parsed = json.loads(text.splitlines()[-1]) if text else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        outcome = {k: v for k, v in parsed.items() if k != "delivery_id"}
        outcome.setdefault("status", "failed" if parsed.get("error") else "settled")
        return outcome
    if returncode == 0:
        return {"status": "settled", "reply": text or "(empty reply)"}
    return {"status": "failed", "error": text or f"delivery runner exited {returncode}"}


def cancel_own_delivery(delivery_id: str, raw_target: str, sender: Any) -> str:
    """Withdraw the caller's own queued request, or stop the turn it started, on a local teammate."""
    from tools.bot_live_delivery import _TERMINAL, _authority_rpc, read_delivery_result
    from tools.bot_mode_dm import _err, _resolve_local_name

    if not _HEX_ID_RE.fullmatch(delivery_id):
        return _err("cancel takes the delivery_id from your own message_agent ack. Do not retry.")
    named = raw_target.strip().lstrip("@")
    resolved = _resolve_local_name(named, list(sender.roster_homes), sender.root) if named else None
    homes = ([sender.roster_homes[resolved]] if resolved else
             [home for name, home in sender.roster_homes.items() if name != sender.me])
    found = next(((home, record) for home in homes if (record := read_delivery_result(home, delivery_id))), None)
    if found is None:
        return _err("No request with that delivery_id to a teammate on this install (requests to another "
                    "machine cannot be cancelled). Do not retry.")
    home, record = found
    if (record.get("author") or {}).get("id") != f"bot:{sender.me}":
        return _err("Only the agent that sent a request can cancel it. Do not retry.")
    if record["status"] in _TERMINAL:
        return json.dumps({"status": record["status"], "delivery_id": delivery_id,
                           "detail": "That request already finished; nothing to cancel."})
    admission = record.get("retry_admission_id") or record.get("admission_id")
    if not admission:
        return _err("That request has no admission to cancel (its outcome is unknown). Do not retry.")
    try:
        answer = _authority_rpc(home, "prompt.cancel", {"session_id": record["session_id"], "admission_id": admission})
    except Exception as exc:
        return _err(f"Cancel failed: {exc}")
    if answer.get("status") == "terminal" and answer.get("outcome") == "cancelled":
        return json.dumps({"status": "cancelled", "delivery_id": delivery_id,
                           "detail": "Withdrawn before it ran. Its delivery process reports the cancellation."})
    if answer.get("status") == "terminal":
        return json.dumps({"status": answer.get("outcome") or "terminal", "delivery_id": delivery_id,
                           "detail": "That request already finished; nothing to cancel."})
    return json.dumps({"status": "stopping", "delivery_id": delivery_id,
                       "detail": "Its turn was running and was told to stop; its delivery process reports how it ended."})
