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
import logging
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
logger = logging.getLogger(__name__)


def split_targets(target: str, sender: Any) -> list[str]:
    """``target`` as the recipients it names: one name, or a comma-separated list deduplicated by
    the bot each name resolves to (``hermes`` and ``default`` are one recipient). A friendly name
    that itself contains a comma ('QA, Release') still resolves whole."""
    from tools.bot_mode_dm import _resolve_local_name

    raw = str(target or "").strip().lstrip("@")
    roster = list(sender.roster_homes)
    if "," not in raw or _resolve_local_name(raw, roster, sender.root) is not None:
        return [raw] if raw else []
    named: dict[str, str] = {}
    for part in raw.split(","):
        name = part.strip().lstrip("@")
        if name:
            named.setdefault(_resolve_local_name(name, roster, sender.root) or name.lower(), name)
    return list(named.values())


def fan_out(targets: list[str], content: str, sender: Any, *, expects_reply: bool) -> str:
    """Dispatch ``content`` to every target; with a reply expected, start the one combined runner."""
    from tools.bot_mode_dm import _dispatch

    collected: list[dict] = []
    acks = []
    for target in targets:
        try:
            ack = json.loads(_dispatch(target, content, sender, wake=collected if expects_reply else "none"))
        except Exception as exc:
            # One target's setup failure stays that target's: requests already admitted still get
            # their combined waiter below.
            logger.debug("message_agent fan-out to %s failed", target, exc_info=True)
            ack = {"error": f"NOT sent: {exc}"}
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
        json.dump([{**{k: entry[k] for k in ("command", "label", "delivery_id")},
                    "output": _runner_output(entry["command"])} for entry in collected], stream)
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


def _runner_output(command: str) -> str:
    """How a sub-runner reports: the live lane (``--run-delivery query-file``) prints one JSON
    payload; the relay waiter and the ``hermes peer dm`` transport print the reply as text."""
    from tools.bot_mode_dm import _runner_argv

    argv = shlex.split(command)
    marker = argv.index("--run-delivery") if "--run-delivery" in argv else len(argv)
    parsed = _runner_argv(argv[marker:])
    return "json" if parsed is not None and parsed[1] == "query-file" else "text"


def run_fan_out(spec_path: str) -> int:
    """Runner entry: run each target's runner concurrently, print one JSON outcome list."""
    with open(spec_path, encoding="utf-8-sig") as stream:
        entries = json.load(stream)
    with contextlib.suppress(OSError):
        os.unlink(spec_path)
    results: list[dict] = [{} for _ in entries]

    def run(index: int, entry: dict) -> None:
        base = {"to": entry["label"], "delivery_id": entry["delivery_id"]}
        try:
            done = subprocess.run(shlex.split(entry["command"]), stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=FAN_OUT_WAIT_SECONDS,
                                  check=False)
        except subprocess.TimeoutExpired:
            results[index] = {**base, "status": "pending", "detail": "No outcome within the bound. Do not resend."}
            return
        except (OSError, ValueError) as exc:
            results[index] = {**base, "status": "failed", "error": f"Its delivery runner could not start: {exc}"}
            return
        results[index] = {**base, **_outcome(entry.get("output", "text"), done.returncode, done.stdout, done.stderr)}

    threads = [threading.Thread(target=run, args=item, daemon=True) for item in enumerate(entries)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    settled = sum(result.get("status") == "settled" for result in results)
    print(json.dumps({"replies": results, "settled": settled, "of": len(results)}, ensure_ascii=False))
    return 0 if settled == len(results) else 1


def _outcome(output: str, returncode: int, stdout: str, stderr: str = "") -> dict:
    """A sub-runner's result as a typed outcome. Only a ``json`` runner's stdout is a delivery
    record; a ``text`` runner's successful stdout is the teammate's answer verbatim (even one that
    looks like JSON). A failing text runner may still have been accepted: every runner says "do not
    resend" exactly when the request is still in flight, which is pending, never failed."""
    text = stdout.strip()
    if output == "json" or returncode != 0:
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
    said = "\n".join(part for part in (text, stderr.strip()) if part)
    if "do not resend" in said.lower():
        return {"status": "pending", "detail": said}
    return {"status": "failed", "error": said or f"delivery runner exited {returncode}"}


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
    if not record.get("admission_id"):
        return _err("That request has no admission to cancel (its outcome is unknown). Do not retry.")
    try:
        # The owner decides which admission is the request now, under the lock that admits its one
        # automatic retry: a retry not admitted yet is withdrawn instead of racing this cancel.
        answer = _authority_rpc(home, "bot_relay.cancel", {
            "id": delivery_id, "author_id": f"bot:{sender.me}",
            "profile": Path(home).name if Path(home).parent.name == "profiles" else "default"})
    except Exception as exc:
        # The owner may be down or refuse: the sender gets that as a tool error, never a crash.
        logger.debug("message_agent cancel of %s failed", delivery_id, exc_info=True)
        return _err(f"Cancel failed: {exc}")
    detail = {"cancelled": "Withdrawn before it ran (or before its automatic retry); it will not run.",
              "stopping": "Its turn was running and was told to stop; its delivery process reports how it ended.",
              }.get(answer.get("status"), "That request already finished; nothing to cancel.")
    status = answer.get("status") if answer.get("status") in {"cancelled", "stopping"} else answer.get("receipt_status")
    return json.dumps({"status": status or "terminal", "delivery_id": delivery_id, "detail": detail})
