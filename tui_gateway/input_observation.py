"""Bounded, volatile input correlation for shared clients; never model or DB state.

Mutators and projections run under the session history lock. Queue envelopes own
span sidecars against their existing text. Evicting evidence cannot change work.
"""

from collections import OrderedDict, deque
from copy import deepcopy
import json
import threading
from uuid import uuid4

MAX_RECORDS = 256
MAX_BYTES = 64 * 1024


class InputState:
    def __init__(self):
        self.records = OrderedDict()
        self.record_bytes = 0
        self.outcomes = deque()
        self.outcome_bytes = 0
        self.revision = 0
        self.truncated_before = None
        self.control = threading.RLock()
        self.view = {"revision": 0, "queued": [], "queued_complete": True,
                     "outcomes": [], "outcomes_truncated_before_revision": None}

    def changed(self):
        self.revision += 1
        return self.revision


def state(session):
    owner = session.get("_input_state")
    return owner if owner is not None else session.setdefault("_input_state", InputState())


def _size(value):
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii"))


def valid_ref(value):
    return isinstance(value, str) and 1 <= len(value) <= 64 and all(32 <= ord(c) <= 126 for c in value)


def new_input(session, text, ref=None):
    occurrence = {"id": uuid4().hex}
    if isinstance(ref, str) and valid_ref(ref):
        occurrence["ref"] = ref
    owner = state(session)
    size = _size(occurrence)
    owner.records[occurrence["id"]] = (occurrence, size)
    owner.record_bytes += size
    while len(owner.records) > MAX_RECORDS or owner.record_bytes + len(owner.records) + 1 > MAX_BYTES:
        _, (_, removed) = owner.records.popitem(last=False)
        owner.record_bytes -= removed
    owner.changed()
    snapshot(session)
    return {"parts": [{**occurrence, "start": 0, "end": len(text) if isinstance(text, str) else 1}],
            "complete": True}


def adopt_inputs(session, batch):
    """Only the trusted parent/child protocol imports IDs; RPC never accepts them."""
    owner = state(session)
    for part in batch.get("parts", []):
        occurrence = {k: part[k] for k in ("id", "ref") if k in part}
        if occurrence["id"] not in owner.records:
            size = _size(occurrence)
            owner.records[occurrence["id"]] = (occurrence, size)
            owner.record_bytes += size
    while len(owner.records) > MAX_RECORDS or owner.record_bytes + len(owner.records) + 1 > MAX_BYTES:
        _, (_, removed) = owner.records.popitem(last=False)
        owner.record_bytes -= removed


def project_inputs(session, batch):
    if not batch:
        return {"inputs": [], "inputs_complete": False}
    owner = state(session)
    parts = batch["parts"]
    inputs = [dict(owner.records[p["id"]][0]) for p in parts if p["id"] in owner.records]
    return {"inputs": inputs, "inputs_complete": bool(batch["complete"]) and len(inputs) == len(parts)}


def merge_inputs(first, second, offset):
    parts = [*first["parts"], *[{**p, "start": p["start"] + offset, "end": p["end"] + offset}
                               for p in second["parts"]]]
    return {"parts": parts[-MAX_RECORDS:],
            "complete": first["complete"] and second["complete"] and len(parts) <= MAX_RECORDS}


def reply_submission(response, batch, disposition, turn=None):
    # A fresh RPC occurrence has one member; preserve it even if another thread
    # evicts the snapshot evidence before the reply is written.
    part = batch["parts"][0]
    submission = {"input_id": part["id"], "disposition": disposition}
    if "ref" in part:
        submission["ref"] = part["ref"]
    if turn:
        submission["turn"] = deepcopy(turn)
    if "error" in response:
        error = response["error"]
        error["data"] = {**(error.get("data") or {}), "submission": submission}
    else:
        response["result"]["submission"] = submission
    return response


def record_outcome(session, batch, disposition, **details):
    owner = state(session)
    if disposition == "absorbed":
        inflight = session.get("inflight_turn") or {}
        if inflight.get("streaming") and inflight.get("turn"):
            details.setdefault("turn", inflight["turn"])
            details.setdefault("into_inputs", project_inputs(session, inflight.get("input_batch"))["inputs"])
    for occurrence in project_inputs(session, batch)["inputs"]:
        record = {"revision": owner.changed(), "input": occurrence, "disposition": disposition, **deepcopy(details)}
        size = _size(record)
        owner.outcomes.append((record, size))
        owner.outcome_bytes += size
    while len(owner.outcomes) > MAX_RECORDS or owner.outcome_bytes + len(owner.outcomes) + 1 > MAX_BYTES:
        removed, size = owner.outcomes.popleft()
        owner.outcome_bytes -= size
        owner.truncated_before = removed["revision"]
    snapshot(session)


def queue_entries(session):
    head = session.get("queued_prompt")
    return ([head] if head else []) + list(session.get("queued_prompts") or [])


def clear_queue(session, reason):
    for entry in queue_entries(session):
        record_outcome(session, entry.get("input_batch"), "cancelled", reason=reason)
    session["queued_prompt"] = None
    session.pop("queued_prompts", None)
    state(session).changed()
    snapshot(session)


def sanitize_inputs(session, original_entry, cleaned):
    """Mirror the existing prefix sanitizer using spans, not text-based identity."""
    batch = original_entry.get("input_batch")
    if not batch or cleaned is original_entry:
        return cleaned
    old = original_entry["text"]
    if cleaned is None:
        record_outcome(session, batch, "absorbed", reason="duplicate_of_inflight")
        return None
    new = cleaned["text"]
    # The sanitizer only removes a prefix and strips whitespace. rfind locates
    # the retained suffix; identities themselves never come from text matching.
    start = old.rfind(new)
    end = start + len(new)
    kept, removed = [], []
    for part in batch["parts"]:
        if part["end"] <= start or part["start"] >= end:
            removed.append(part)
        else:
            kept.append({**part, "start": max(0, part["start"] - start), "end": min(end, part["end"]) - start})
    if removed:
        record_outcome(session, {"parts": removed, "complete": batch["complete"]}, "absorbed",
                       reason="duplicate_of_inflight")
    return {**cleaned, "input_batch": {"parts": kept, "complete": batch["complete"]}}


def snapshot(session):
    owner = state(session)
    entries = queue_entries(session)
    # Drop evicted IDs from envelope sidecars as well as the lookup. Empty
    # sidecars retain only an incompleteness bit, never hidden copies of refs.
    batches = [entry.get("input_batch") for entry in entries]
    inflight = session.get("inflight_turn") or {}
    batches.append(inflight.get("input_batch"))
    batches.extend(item.get("input_batch") for item in inflight.get("input_observations", []))
    if observation := session.get("_turn_observation"):
        batches.append(observation.inputs)
    for batch in batches:
        if batch:
            kept = [part for part in batch["parts"] if part["id"] in owner.records]
            if len(kept) != len(batch["parts"]):
                batch.update(parts=kept, complete=False)
    queued = [project_inputs(session, entry.get("input_batch")) for entry in entries[:MAX_RECORDS]]
    owner.view = {"revision": owner.revision, "queued": queued,
            "queued_complete": len(entries) <= MAX_RECORDS and all(entry["inputs_complete"] for entry in queued),
            "outcomes": [deepcopy(item) for item, _ in owner.outcomes],
            "outcomes_truncated_before_revision": owner.truncated_before}
    return deepcopy(owner.view)


def cached_snapshot(session):
    # Writers replace this immutable view under history_lock. Slow info builders
    # may already hold that lock; they must not acquire it recursively.
    return deepcopy(state(session).view)


def publish_state(sid, session):
    from tui_gateway import server
    server._emit("session.info", sid, server._session_info(session.get("agent"), session))
