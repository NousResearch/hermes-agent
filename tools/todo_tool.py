#!/usr/bin/env python3
"""
Todo Tool Module - Planning & Task Management

Provides an in-memory task list the agent uses to decompose complex tasks,
track progress, and maintain focus across long conversations. The state
lives on the AIAgent instance (one per session) and is re-injected into
the conversation after context compression events.

Design:
- Single `todo` tool: provide `todos` to write, omit it to read or control
- Durable control fields are isolated: exactly one; only a Canonical checkpoint
  may accompany `todos`, as one readback-verified atomic update
- Every call returns the full current list
- No system prompt mutation, no tool response modification
- Behavioral guidance lives entirely in the tool schema description
"""

import copy
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional


# Valid status values for todo items
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled", "blocked"}

# Bounds on persisted todo state. The todo list is a planning aid the model
# re-reads after every context-compression event (see format_for_injection),
# so unbounded item content or count defeats the compression it rides through.
# These caps keep a single oversized item (whether authored by the model or
# replayed from caller-supplied history on the API server) from inflating the
# re-injection block. Generous relative to real plans — a todo item is a short
# task description, and active lists are a handful of items, not hundreds.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
# Upper bound on a single todo tool-result payload accepted during history
# hydration. The gateway/API server replays caller-supplied conversation
# history to rebuild the store, so an oversized forged result is dropped
# before it is parsed and re-injected (see AIAgent._hydrate_todo_store).
MAX_TODO_RESULT_CHARS = 512_000
_TRUNCATION_MARKER = "… [truncated]"

CANONICAL_BINDING_SCHEMA = "hermes.todo-canonical-binding.v1"
CANONICAL_SYNC_BLOCKED_SCHEMA = "hermes.todo-canonical-sync-blocked.v1"
_CANONICAL_BINDING_KEYS = frozenset({
    "schema",
    "case_id",
    "plan_id",
    "plan_revision",
    "plan_state",
    "plan_event_id",
    "canonical_content_sha256",
    "workspace_todos_sha256",
    "todo_items_sha256",
    "binding_sha256",
})
_CANONICAL_SYNC_BLOCKED_KEYS = frozenset({
    "schema",
    "error_code",
    "todo_items_sha256",
    "binding_sha256",
    "sync_sha256",
})
_CANONICAL_SYNC_BLOCKED_CODES = frozenset(
    {
        "canonical_brain_unavailable",
        "canonical_truth_diverged",
    }
)
_CANONICAL_PREDISPATCH_UNAVAILABLE_CODES = frozenset(
    {
        "connection_closed",
        "dispatch_unavailable",
        "timeout",
        "transport_error",
        "unauthorized_peer",
    }
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _todo_items_sha256(items: List[Dict[str, str]]) -> str:
    return _sha256_json(items)


def _is_sha256(value: Any, *, allow_empty: bool = False) -> bool:
    if allow_empty and value == "":
        return True
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _canonical_binding_state(
    *,
    case_id: str,
    plan_id: str,
    plan_revision: int,
    plan_state: str,
    plan_event_id: str,
    canonical_content_sha256: str,
    workspace_todos_sha256: str,
    items: List[Dict[str, str]],
) -> Dict[str, Any]:
    unsigned: Dict[str, Any] = {
        "schema": CANONICAL_BINDING_SCHEMA,
        "case_id": str(case_id or "").strip(),
        "plan_id": str(plan_id or "").strip(),
        "plan_revision": plan_revision,
        "plan_state": str(plan_state or "").strip(),
        "plan_event_id": str(plan_event_id or "").strip(),
        "canonical_content_sha256": str(canonical_content_sha256 or "").strip(),
        "workspace_todos_sha256": str(workspace_todos_sha256 or "").strip(),
        "todo_items_sha256": _todo_items_sha256(items),
    }
    if (
        not unsigned["case_id"]
        or not unsigned["plan_id"]
        or type(plan_revision) is not int
        or plan_revision < 1
        or unsigned["plan_state"] not in {"active", "blocked"}
        or not unsigned["plan_event_id"]
        or not _is_sha256(
            unsigned["canonical_content_sha256"],
            allow_empty=True,
        )
        or not _is_sha256(unsigned["workspace_todos_sha256"])
    ):
        raise ValueError("canonical todo binding is invalid")
    return {**unsigned, "binding_sha256": _sha256_json(unsigned)}


def _validate_canonical_binding_state(
    value: Any,
    items: List[Dict[str, str]],
) -> Dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != _CANONICAL_BINDING_KEYS:
        raise ValueError("canonical todo binding schema is invalid")
    expected = _canonical_binding_state(
        case_id=value.get("case_id", ""),
        plan_id=value.get("plan_id", ""),
        plan_revision=value.get("plan_revision"),
        plan_state=value.get("plan_state", ""),
        plan_event_id=value.get("plan_event_id", ""),
        canonical_content_sha256=value.get("canonical_content_sha256", ""),
        workspace_todos_sha256=value.get("workspace_todos_sha256", ""),
        items=items,
    )
    if expected != value:
        raise ValueError("canonical todo binding checksum is invalid")
    return copy.deepcopy(expected)


def _todo_snapshot_fragment(body: str) -> str:
    from agent.message_provenance import TODO_SNAPSHOT_END, TODO_SNAPSHOT_START

    return f"{TODO_SNAPSHOT_START}\n{body}\n{TODO_SNAPSHOT_END}"


class TodoStoreFencedError(RuntimeError):
    """Local todo state is unavailable until Canonical truth is reconciled."""


class TodoStore:
    """
    In-memory todo list. One instance per AIAgent (one per session).

    Items are ordered -- list position is priority. Each item has:
      - id: unique string identifier (agent-chosen)
      - content: task description
      - status: pending | in_progress | completed | cancelled
    """

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        self._canonical_fence: Optional[Dict[str, Any]] = None
        self._canonical_binding: Optional[Dict[str, Any]] = None
        self._canonical_sync_blocked: Optional[Dict[str, Any]] = None

    def _require_unfenced(self) -> None:
        if self._canonical_fence is not None:
            raise TodoStoreFencedError(
                "TodoStore is fenced after an unverified Canonical checkpoint; "
                "retry the exact checkpoint before reading or writing local state"
            )

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status} dicts
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        self._require_unfenced()
        if self._canonical_sync_blocked is not None:
            raise TodoStoreFencedError(
                "TodoStore is waiting for a Canonical checkpoint retry"
            )
        previous_items = [item.copy() for item in self._items]
        if not merge:
            # Replace mode: new list entirely
            self._items = [self._validate(t) for t in self._dedupe_by_id(todos)]
        else:
            # Merge mode: update existing items by id, append new ones
            existing = {item["id"]: item for item in self._items}
            for t in self._dedupe_by_id(todos):
                item_id = str(t.get("id", "")).strip()
                if not item_id:
                    continue  # Can't merge without an id

                if item_id in existing:
                    # Update only the fields the LLM actually provided
                    if "content" in t and t["content"]:
                        existing[item_id]["content"] = self._cap_content(str(t["content"]).strip())
                    if "status" in t and t["status"]:
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                else:
                    # New item -- validate fully and append to end
                    validated = self._validate(t)
                    existing[validated["id"]] = validated
                    self._items.append(validated)
            # Rebuild _items preserving order for existing items
            seen = set()
            rebuilt = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = rebuilt
        # Bound total item count so a replayed/oversized list can't grow the
        # re-injection block without limit. Keep the highest-priority head
        # (list order is priority).
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        if (
            self._canonical_binding is not None
            and _todo_items_sha256(self._items)
            != self._canonical_binding["todo_items_sha256"]
        ):
            self._items = previous_items
            raise TodoStoreFencedError(
                "Canonical-bound TodoStore updates require an exact "
                "readback-verified canonical_checkpoint"
            )
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        self._require_unfenced()
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        # A fence counts as occupied state so history hydration cannot replace
        # an unresolved Canonical transition with an older tool-result snapshot.
        return (
            self._canonical_fence is not None
            or self._canonical_sync_blocked is not None
            or bool(self._items)
        )

    def canonical_binding_state(self) -> Optional[Dict[str, Any]]:
        """Return the exact bounded binding carried in todo tool receipts."""

        return copy.deepcopy(self._canonical_binding)

    def canonical_sync_blocked_state(self) -> Optional[Dict[str, Any]]:
        """Return the exact typed pre-dispatch blocker, if one is active."""

        return copy.deepcopy(self._canonical_sync_blocked)

    def is_canonical_dirty(self) -> bool:
        """Return exact snapshot divergence without inspecting task meaning."""

        return (
            self._canonical_binding is not None
            and _todo_items_sha256(self._items)
            != self._canonical_binding["todo_items_sha256"]
        )

    def canonical_sync_receipt(self) -> Optional[Dict[str, Any]]:
        """Return bounded mechanical sync state for the model and hydrator."""

        if self._canonical_fence is not None:
            return self.canonical_fence_receipt()
        current_sha256 = _todo_items_sha256(self._items)
        if self._canonical_sync_blocked is not None:
            return {
                "state": "sync_blocked",
                "error_code": self._canonical_sync_blocked["error_code"],
                "current_todos_sha256": current_sha256,
                "canonical_retry_required": True,
                "durable_completion_verified": False,
            }
        if self._canonical_binding is None:
            return None
        dirty = current_sha256 != self._canonical_binding["todo_items_sha256"]
        return {
            "state": "dirty" if dirty else "clean",
            "case_id": self._canonical_binding["case_id"],
            "plan_id": self._canonical_binding["plan_id"],
            "plan_revision": self._canonical_binding["plan_revision"],
            "plan_state": self._canonical_binding["plan_state"],
            "plan_event_id": self._canonical_binding["plan_event_id"],
            "canonical_todos_sha256": self._canonical_binding[
                "todo_items_sha256"
            ],
            "current_todos_sha256": current_sha256,
            "canonical_checkpoint_required": dirty,
            "durable_completion_verified": not dirty,
        }

    def bind_canonical_workspace(
        self,
        *,
        case_id: str,
        plan_id: str,
        plan_revision: int,
        plan_state: str,
        plan_event_id: str,
        canonical_content_sha256: str,
        workspace_todos_sha256: str,
        items: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Bind exact local items to one readback-proven Canonical plan."""

        self._require_unfenced()
        if _stable_json(items) != _stable_json(self._items):
            raise ValueError("canonical todo binding items do not match live state")
        binding = _canonical_binding_state(
            case_id=case_id,
            plan_id=plan_id,
            plan_revision=plan_revision,
            plan_state=plan_state,
            plan_event_id=plan_event_id,
            canonical_content_sha256=canonical_content_sha256,
            workspace_todos_sha256=workspace_todos_sha256,
            items=self._items,
        )
        self._canonical_binding = binding
        self._canonical_sync_blocked = None
        return copy.deepcopy(binding)

    def restore_canonical_binding(
        self,
        state: Dict[str, Any],
    ) -> None:
        """Restore an exact checksum-bound Canonical binding from history."""

        self._require_unfenced()
        self._canonical_binding = _validate_canonical_binding_state(
            state,
            self._items,
        )
        self._canonical_sync_blocked = None

    def install_verified_canonical_snapshot(
        self,
        items: List[Dict[str, str]],
        *,
        binding: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Install exactly the candidate whose Canonical readback just passed."""

        normalized = [self._validate(item) for item in items]
        if len(normalized) > MAX_TODO_ITEMS:
            normalized = normalized[:MAX_TODO_ITEMS]
        if _stable_json(normalized) != _stable_json(items):
            raise ValueError("verified Canonical todo snapshot is not normalized")
        validated_binding = (
            _validate_canonical_binding_state(binding, normalized)
            if binding is not None
            else None
        )
        self._items = normalized
        self._canonical_fence = None
        self._canonical_binding = validated_binding
        self._canonical_sync_blocked = None
        return self.read()

    def mark_canonical_sync_blocked(self, error_code: str) -> Dict[str, Any]:
        """Record one typed proven-pre-dispatch Canonical blocker."""

        if error_code not in _CANONICAL_SYNC_BLOCKED_CODES:
            raise ValueError("unsupported canonical sync blocker")
        unsigned = {
            "schema": CANONICAL_SYNC_BLOCKED_SCHEMA,
            "error_code": error_code,
            "todo_items_sha256": _todo_items_sha256(self._items),
            "binding_sha256": (
                self._canonical_binding["binding_sha256"]
                if self._canonical_binding is not None
                else ""
            ),
        }
        self._canonical_sync_blocked = {
            **unsigned,
            "sync_sha256": _sha256_json(unsigned),
        }
        return copy.deepcopy(self._canonical_sync_blocked)

    def restore_canonical_sync_blocked(self, state: Dict[str, Any]) -> None:
        """Restore a checksum-bound typed blocker from paired tool history."""

        if (
            not isinstance(state, dict)
            or frozenset(state) != _CANONICAL_SYNC_BLOCKED_KEYS
        ):
            raise ValueError("canonical sync blocker schema is invalid")
        unsigned = {
            key: state[key]
            for key in _CANONICAL_SYNC_BLOCKED_KEYS
            if key != "sync_sha256"
        }
        expected_binding_sha256 = (
            self._canonical_binding["binding_sha256"]
            if self._canonical_binding is not None
            else ""
        )
        if (
            state.get("schema") != CANONICAL_SYNC_BLOCKED_SCHEMA
            or state.get("error_code") not in _CANONICAL_SYNC_BLOCKED_CODES
            or state.get("todo_items_sha256")
            != _todo_items_sha256(self._items)
            or state.get("binding_sha256") != expected_binding_sha256
            or state.get("sync_sha256") != _sha256_json(unsigned)
        ):
            raise ValueError("canonical sync blocker checksum is invalid")
        self._canonical_sync_blocked = copy.deepcopy(state)

    def clear_canonical_sync_blocked(self) -> None:
        """Clear a prior typed blocker immediately before an explicit retry."""

        self._canonical_sync_blocked = None

    def replace_canonical_fence_with_truth_divergence(self) -> Dict[str, Any]:
        """Replace an uncertain retry fence with a proven content conflict.

        A later exact reconciliation can prove that the deterministic event
        identity is occupied by different Canonical content. At that point an
        exact retry is permanently wrong: discard the untrusted candidate,
        expose no executable local plan, and require an authoritative read.
        """

        if self._canonical_fence is None:
            raise TodoStoreFencedError("TodoStore has no Canonical fence")
        self._canonical_fence = None
        self._items = []
        self._canonical_binding = None
        self._canonical_sync_blocked = None
        return self.mark_canonical_sync_blocked("canonical_truth_diverged")

    def has_active_items(self) -> bool:
        """Return whether the model-owned plan still requires execution.

        The runtime does not interpret task meaning.  It only observes the
        structured statuses chosen by the model.  ``pending`` and
        ``in_progress`` require another tool call; ``blocked`` is terminal for
        the current turn so the model can ask for genuinely unavailable input.
        """
        if self._canonical_fence is not None:
            # Keep the model in the tool loop until it performs exact
            # idempotent reconciliation.  No stale item meaning is inspected.
            return True
        if self._canonical_sync_blocked is not None:
            # A proven writer-unavailable pre-dispatch failure is allowed to
            # surface one focused blocker reply instead of forcing a tool loop
            # until the aggregate execution lease is exhausted.
            return False
        if self.is_canonical_dirty():
            return True
        return any(
            item["status"] in {"pending", "in_progress"}
            for item in self._items
        )

    def format_for_injection(self) -> Optional[str]:
        """
        Render the todo list for post-compression injection.

        Returns a provenance-marked reference for attachment to the existing
        final real user turn, or None if the list is empty.
        """
        if self._canonical_fence is not None:
            public = self.canonical_fence_receipt() or {}
            return _todo_snapshot_fragment(
                "[Canonical Task Workspace reconciliation required]\n"
                "A task.plan.updated append may have committed, but its readback "
                "receipt was not verified. Local todo state is fenced and must not "
                "be used. Retry the exact same canonical_checkpoint with no todos "
                "until an idempotent readback receipt is verified.\n"
                f"- checkpoint_sha256: {public.get('checkpoint_sha256', '')}\n"
                f"- case_id: {public.get('case_id', '')}\n"
                f"- plan_id: {public.get('plan_id', '')}\n"
                f"- plan_revision: {public.get('plan_revision', '')}"
            )
        if self._canonical_sync_blocked is not None:
            if (
                self._canonical_sync_blocked["error_code"]
                == "canonical_truth_diverged"
            ):
                return _todo_snapshot_fragment(
                    "[Canonical Task Workspace truth diverged]\n"
                    "The exact plan-transition identity already exists in the "
                    "Canonical Brain with different content. The local candidate "
                    "was not advanced and must not be treated as Canonical truth. "
                    "Query the exact Canonical workspace, then author a new "
                    "checkpoint from that authoritative state (using the next "
                    "valid revision or an explicit supersession).\n"
                    "- error_code: canonical_truth_diverged\n"
                    f"- todo_items_sha256: "
                    f"{self._canonical_sync_blocked['todo_items_sha256']}"
                )
            return _todo_snapshot_fragment(
                "[Canonical Task Workspace sync blocked]\n"
                "The exact local Todo snapshot was not advanced because the "
                "Canonical Writer was proven unavailable before dispatch. "
                "Do not claim durable completion. Retry a model-authored "
                "canonical_checkpoint when the writer is available.\n"
                f"- error_code: {self._canonical_sync_blocked['error_code']}\n"
                f"- todo_items_sha256: "
                f"{self._canonical_sync_blocked['todo_items_sha256']}"
            )
        if self.is_canonical_dirty():
            binding = self._canonical_binding or {}
            return _todo_snapshot_fragment(
                "[Canonical Task Workspace checkpoint required]\n"
                "The recovered local Todo snapshot differs from the last exact "
                "Canonical readback. Call todo with no parameters if you need "
                "the full snapshot, then submit a model-authored "
                "canonical_checkpoint before claiming durable progress.\n"
                f"- case_id: {binding.get('case_id', '')}\n"
                f"- plan_id: {binding.get('plan_id', '')}\n"
                f"- plan_revision: {binding.get('plan_revision', '')}\n"
                f"- current_todos_sha256: {_todo_items_sha256(self._items)}"
            )
        if not self._items:
            return None

        # Status markers for compact display
        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
            "blocked": "[!]",
        }

        # Preserve every unfinished model-authored state.  A blocked item does
        # not keep the current tool loop alive (see ``has_active_items``), but
        # omitting it here would erase the exact work waiting for new user
        # input when compression happens between the blocker and the reply.
        # Completed/cancelled items stay omitted so finished work is not redone.
        active_items = [
            item for item in self._items
            if item["status"] in {"pending", "in_progress", "blocked"}
        ]
        if not active_items:
            return None

        lines = ["[Your active task list was preserved across context compression]"]
        if any(item["status"] == "blocked" for item in active_items):
            lines.append(
                "Blocked items are exact prior model state, not a runtime decision: "
                "inspect the current user turn and live evidence, then update or "
                "continue them if their blocker has cleared."
            )
        for item in active_items:
            marker = markers.get(item["status"], "[?]")
            lines.append(f"- {marker} {item['id']}. {item['content']} ({item['status']})")

        return _todo_snapshot_fragment("\n".join(lines))

    def fence_canonical_uncertainty(
        self,
        *,
        checkpoint: Dict[str, Any],
        effective_checkpoint: Dict[str, Any],
        items: List[Dict[str, str]],
        checkpoint_sha256: str,
        details: Dict[str, Any],
    ) -> None:
        """Hide all local items after an append with an unknown terminal state.

        The exact candidate and checkpoint remain private in memory solely so a
        later call can repeat the same idempotent append.  Ordinary reads and
        writes are rejected, and prior local items are discarded rather than
        being misrepresented as Canonical truth.
        """

        plan = checkpoint.get("plan") if isinstance(checkpoint, dict) else {}
        plan = plan if isinstance(plan, dict) else {}
        self._items = []
        self._canonical_binding = None
        self._canonical_sync_blocked = None
        self._canonical_fence = {
            "checkpoint": copy.deepcopy(checkpoint),
            "effective_checkpoint": copy.deepcopy(effective_checkpoint),
            "items": copy.deepcopy(items),
            "checkpoint_sha256": str(checkpoint_sha256),
            "case_id": str(checkpoint.get("case_id") or ""),
            "plan_id": str(plan.get("plan_id") or ""),
            "plan_revision": plan.get("revision"),
            "details": copy.deepcopy(details),
        }

    def canonical_fence_receipt(self) -> Optional[Dict[str, Any]]:
        """Return bounded non-content metadata for an active fence."""

        if self._canonical_fence is None:
            return None
        return {
            "status": "canonical_reconciliation_required",
            "checkpoint_sha256": self._canonical_fence["checkpoint_sha256"],
            "case_id": self._canonical_fence["case_id"],
            "plan_id": self._canonical_fence["plan_id"],
            "plan_revision": self._canonical_fence["plan_revision"],
            "canonical_write_may_have_occurred": True,
        }

    def _canonical_fence_state(self) -> Optional[Dict[str, Any]]:
        """Return the private exact reconciliation payload to ``todo_tool``."""

        return copy.deepcopy(self._canonical_fence)

    def restore_canonical_fence(self, state: Dict[str, Any]) -> None:
        """Restore an exact fence from a paired todo result in session history.

        Gateway workers may construct a fresh ``AIAgent`` for the next message.
        The uncertain append therefore has to survive history hydration; an
        in-memory-only fence would allow the hydrator to roll back to an older
        successful todo result.  The checksum binds the frozen checkpoint and
        candidate items before either is accepted.
        """

        if not isinstance(state, dict):
            raise ValueError("canonical fence state must be an object")
        checkpoint = state.get("checkpoint")
        effective_checkpoint = state.get("effective_checkpoint")
        items = state.get("items")
        details = state.get("details")
        checkpoint_sha256 = str(state.get("checkpoint_sha256") or "")
        if not isinstance(checkpoint, dict):
            raise ValueError("canonical fence checkpoint must be an object")
        if not isinstance(effective_checkpoint, dict):
            raise ValueError(
                "canonical fence effective_checkpoint must be an object"
            )
        if not isinstance(items, list):
            raise ValueError("canonical fence items must be a list")
        if not isinstance(details, dict):
            raise ValueError("canonical fence details must be an object")

        validated_items = [self._validate(item) for item in items]
        if len(validated_items) > MAX_TODO_ITEMS:
            raise ValueError("canonical fence items exceed the bounded maximum")
        if _stable_json(validated_items) != _stable_json(items):
            raise ValueError("canonical fence items are not normalized")
        expected_sha256 = _canonical_checkpoint_sha256(
            effective_checkpoint,
            validated_items,
        )
        if checkpoint_sha256 != expected_sha256:
            raise ValueError("canonical fence checksum mismatch")

        self.fence_canonical_uncertainty(
            checkpoint=checkpoint,
            effective_checkpoint=effective_checkpoint,
            items=validated_items,
            checkpoint_sha256=checkpoint_sha256,
            details=details,
        )

    def resolve_canonical_fence(self, items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Install exact readback-verified items and clear the fence."""

        if self._canonical_fence is None:
            raise TodoStoreFencedError("TodoStore has no Canonical fence to resolve")
        # Validate into a fresh list before changing either field.
        # ``items`` is the already-validated exact snapshot used in the
        # Canonical append. Do not dedupe it a second time: legacy coercion can
        # preserve multiple invalid raw items as separate ``?`` entries, and
        # local state must match the bytes that were checkpointed.
        validated = [self._validate(item) for item in items]
        if len(validated) > MAX_TODO_ITEMS:
            validated = validated[:MAX_TODO_ITEMS]
        self._items = validated
        self._canonical_fence = None
        return self.read()

    @staticmethod
    def _cap_content(content: str) -> str:
        """Truncate oversized todo content to MAX_TODO_CONTENT_CHARS.

        A single huge item would otherwise inflate the post-compression
        re-injection block (format_for_injection) without bound. Keep the
        head — the actionable part of a task description — plus a marker.
        """
        if len(content) > MAX_TODO_CONTENT_CHARS:
            keep = MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)
            return content[:keep] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate and normalize a todo item.

        Ensures required fields exist and status is valid.
        Returns a clean dict with only {id, content, status}.
        """
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}

        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"
        else:
            content = TodoStore._cap_content(content)

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        return {"id": item_id, "content": content, "status": status}

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                # Non-dict items get a synthetic key so _validate can handle them
                last_index[f"__invalid_{i}"] = i
                continue
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]

def _candidate_todo_items(
    previous_items: List[Dict[str, str]],
    todos: Optional[List[Dict[str, Any]]],
    merge: bool,
) -> List[Dict[str, str]]:
    """Validate a prospective update without mutating the live TodoStore."""

    if todos is None:
        return copy.deepcopy(previous_items)
    candidate = TodoStore()
    candidate.write(previous_items, merge=False)
    return candidate.write(todos, merge=merge)


def _todo_result(
    items: List[Dict[str, str]],
    *,
    store: Optional[TodoStore] = None,
) -> Dict[str, Any]:
    """Build the stable full-list response for one validated snapshot."""

    result: Dict[str, Any] = {
        "todos": copy.deepcopy(items),
        "summary": {
            "total": len(items),
            "pending": sum(1 for item in items if item["status"] == "pending"),
            "in_progress": sum(
                1 for item in items if item["status"] == "in_progress"
            ),
            "completed": sum(
                1 for item in items if item["status"] == "completed"
            ),
            "cancelled": sum(
                1 for item in items if item["status"] == "cancelled"
            ),
            "blocked": sum(1 for item in items if item["status"] == "blocked"),
        },
    }
    if store is not None:
        binding = store.canonical_binding_state()
        sync_blocked = store.canonical_sync_blocked_state()
        sync = store.canonical_sync_receipt()
        if binding is not None:
            result["canonical_binding_state"] = binding
        if sync_blocked is not None:
            result["canonical_sync_blocked_state"] = sync_blocked
        if sync is not None:
            result["canonical_sync"] = sync
    return result


def _canonical_checkpoint_sha256(
    checkpoint: Dict[str, Any],
    items: List[Dict[str, str]],
) -> str:
    return hashlib.sha256(
        _stable_json({"checkpoint": checkpoint, "todos": items}).encode("utf-8")
    ).hexdigest()


def _effective_canonical_checkpoint(
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """Freeze runtime-observed source refs before the first append attempt.

    A later reconciliation can run in a new user turn whose observed message id
    differs. Persisting the first turn's exact augmented refs keeps both the
    idempotency key and canonical content hash byte-stable across that boundary.
    """

    effective = copy.deepcopy(checkpoint)
    source_refs = effective.get("source_refs")
    if isinstance(source_refs, dict):
        from tools.canonical_brain_tool import (
            _augment_source_refs_from_session_context,
        )

        effective["source_refs"] = _augment_source_refs_from_session_context(
            source_refs
        )
    return effective


def _binding_for_verified_checkpoint(
    checkpoint: Dict[str, Any],
    receipt: Dict[str, Any],
    items: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Build a mechanical binding, or release it for a terminal plan state."""

    plan = checkpoint.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("canonical checkpoint plan is unavailable")
    plan_state = str(plan.get("state") or "").strip()
    if plan_state in {"completed", "cancelled"}:
        return None
    return _canonical_binding_state(
        case_id=str(receipt.get("case_id") or ""),
        plan_id=str(receipt.get("plan_id") or ""),
        plan_revision=receipt.get("plan_revision"),
        plan_state=plan_state,
        plan_event_id=str(receipt.get("event_id") or ""),
        canonical_content_sha256=str(
            receipt.get("canonical_content_sha256") or ""
        ),
        workspace_todos_sha256=str(
            receipt.get("workspace_todos_sha256") or ""
        ),
        items=items,
    )


def _canonical_fence_error(
    store: TodoStore,
    message: str,
    **extra: Any,
) -> str:
    """Return a restart-safe tombstone for an unresolved Canonical append."""

    return tool_error(
        message,
        # An explicit empty snapshot prevents legacy hydration from skipping
        # this result and restoring an older, now-stale local plan.
        todos=[],
        canonical_fence=store.canonical_fence_receipt() or {},
        # Preserve the exact frozen append inputs across fresh gateway agents.
        # The hydrator verifies checkpoint_sha256 before restoring the fence.
        canonical_fence_state=store._canonical_fence_state() or {},
        todo_update_applied=False,
        **extra,
    )


def todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    store: Optional[TodoStore] = None,
    plan_approval: Optional[Dict[str, Any]] = None,
    goal_outcome: Optional[Dict[str, Any]] = None,
    goal_contract: Optional[Dict[str, Any]] = None,
    canonical_checkpoint: Optional[Dict[str, Any]] = None,
    delivery_outcome: Optional[Dict[str, Any]] = None,
    delivery_outcome_recorder: Optional[
        Callable[[Dict[str, Any]], Dict[str, Any]]
    ] = None,
    session_key: str = "",
    goal_session_id: str = "",
    originating_turn_id: str = "",
    goal_generation_id: str = "",
    user_id: str = "",
) -> str:
    """
    Read or write todos, or execute one isolated canonical control side effect.

    Args:
        todos: if provided, write these items. If None, read current list.
        merge: if True, update by id. If False (default), replace entire list.
        store: the TodoStore instance from the AIAgent.
        session_key: stable gateway authority key used for approval leases.
        goal_session_id: conversation session id used by persistent /goal state.
        originating_turn_id: exact model turn mechanically captured by dispatch.
        goal_generation_id: durable standing-goal generation bound to that turn.

    Returns:
        JSON string with the full current list and summary metadata.
    """
    if store is None:
        return tool_error("TodoStore not initialized")

    control_fields = (
        ("canonical_checkpoint", canonical_checkpoint),
        ("plan_approval", plan_approval),
        ("goal_outcome", goal_outcome),
        ("goal_contract", goal_contract),
        ("delivery_outcome", delivery_outcome),
    )
    requested_controls = [
        name for name, value in control_fields if value is not None
    ]
    if len(requested_controls) > 1:
        return tool_error(
            "canonical control fields must be submitted as separate todo tool "
            "calls; submit exactly one of canonical_checkpoint, plan_approval, "
            "goal_outcome, goal_contract, or delivery_outcome per call",
            requested_control_fields=requested_controls,
            todo_update_applied=False,
            control_side_effect_applied=False,
        )
    atomic_checkpoint_update = (
        todos is not None
        and requested_controls == ["canonical_checkpoint"]
    )
    if requested_controls and todos is not None and not atomic_checkpoint_update:
        return tool_error(
            "todos and canonical control fields must be submitted as separate "
            "todo tool calls; only canonical_checkpoint may accompany todos "
            "to perform one readback-verified atomic Canonical update",
            requested_control_fields=requested_controls,
            todo_update_applied=False,
            control_side_effect_applied=False,
        )
    if (goal_outcome is not None or goal_contract is not None) and (
        not str(originating_turn_id or "").strip()
        or not str(goal_generation_id or "").strip()
    ):
        return tool_error(
            "goal_outcome and goal_contract require exact turn and goal-generation authority"
        )

    # Validate every model-authored container before performing any local or
    # external side effect.  This makes malformed calls a proven pre-dispatch
    # failure and leaves the prior TodoStore snapshot untouched.
    for label, value in control_fields:
        if value is not None and not isinstance(value, dict):
            return tool_error(
                f"{label} must be an object",
                todo_update_applied=False,
            )
    if delivery_outcome is not None and delivery_outcome_recorder is None:
        return tool_error(
            "delivery outcome recorder not initialized",
            todo_update_applied=False,
        )

    fence = store._canonical_fence_state()
    if fence is not None:
        if (
            canonical_checkpoint is None
            or todos is not None
            or any(
                value is not None
                for name, value in control_fields
                if name != "canonical_checkpoint"
            )
        ):
            return _canonical_fence_error(
                store,
                "TodoStore is fenced: retry the exact same canonical_checkpoint "
                "with no todos or other side effects",
            )
        if _stable_json(canonical_checkpoint) != _stable_json(fence["checkpoint"]):
            return _canonical_fence_error(
                store,
                "TodoStore is fenced: canonical_checkpoint does not match the "
                "exact uncertain append",
            )
        try:
            checkpoint_receipt = _record_canonical_checkpoint(
                fence["effective_checkpoint"],
                fence["items"],
            )
        except Exception as exc:
            details = getattr(exc, "checkpoint_details", {})
            if (
                isinstance(details, dict)
                and details.get("canonical_sync_blocked") is True
                and details.get("canonical_sync_error_code")
                == "canonical_truth_diverged"
            ):
                store.replace_canonical_fence_with_truth_divergence()
                return tool_error(
                    f"canonical checkpoint conflicts with Canonical truth: {exc}",
                    **_todo_result([], store=store),
                    todo_update_applied=False,
                    todo_preserved=False,
                    **details,
                )
            return _canonical_fence_error(
                store,
                f"canonical checkpoint reconciliation not verified: {exc}",
                **(details if isinstance(details, dict) else {}),
            )
        binding = _binding_for_verified_checkpoint(
            fence["effective_checkpoint"],
            checkpoint_receipt,
            fence["items"],
        )
        items = store.install_verified_canonical_snapshot(
            fence["items"],
            binding=binding,
        )
        result = _todo_result(items, store=store)
        result["canonical_checkpoint"] = checkpoint_receipt
        result["canonical_reconciliation"] = {
            "resolved": True,
            "checkpoint_sha256": fence["checkpoint_sha256"],
        }
        return json.dumps(result, ensure_ascii=False)

    if todos is not None:
        # Guard: models sometimes send todos as a JSON string instead of a list.
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except (json.JSONDecodeError, TypeError):
                return tool_error(
                    "todos must be a list of objects, got unparseable string",
                    todo_update_applied=False,
                )
        if not isinstance(todos, list):
            return tool_error(
                f"todos must be a list, got {type(todos).__name__}",
                todo_update_applied=False,
            )

    if (
        store.canonical_sync_blocked_state() is not None
        and canonical_checkpoint is None
    ):
        current = store.read()
        return tool_error(
            "Canonical Task Workspace sync is blocked by a proven "
            "pre-dispatch writer outage; retry canonical_checkpoint before "
            "any further todo/control mutation",
            **_todo_result(current, store=store),
            todo_update_applied=False,
            control_side_effect_applied=False,
        )
    if (
        (store.canonical_binding_state() is not None or store.is_canonical_dirty())
        and todos is not None
        and canonical_checkpoint is None
    ):
        current = store.read()
        return tool_error(
            "Canonical-bound TodoStore updates require todos and a "
            "model-authored canonical_checkpoint in the same call",
            **_todo_result(current, store=store),
            todo_update_applied=False,
            canonical_checkpoint_required=True,
        )
    # Build the candidate snapshot in an isolated store. The control-field
    # preflight above guarantees the only combined operation is the exact
    # candidate -> Canonical append/readback -> local install transaction.
    previous_items = store.read()
    items = _candidate_todo_items(previous_items, todos, merge)
    result = _todo_result(items)
    checkpoint_binding: Optional[Dict[str, Any]] = None

    if canonical_checkpoint is not None:
        effective_checkpoint = _effective_canonical_checkpoint(
            canonical_checkpoint
        )
        try:
            checkpoint_receipt = _record_canonical_checkpoint(
                effective_checkpoint,
                items,
            )
        except Exception as exc:
            details = getattr(exc, "checkpoint_details", {})
            details = details if isinstance(details, dict) else {}
            if details.get("canonical_write_may_have_occurred") is True:
                checkpoint_sha256 = _canonical_checkpoint_sha256(
                    effective_checkpoint,
                    items,
                )
                store.fence_canonical_uncertainty(
                    checkpoint=canonical_checkpoint,
                    effective_checkpoint=effective_checkpoint,
                    items=items,
                    checkpoint_sha256=checkpoint_sha256,
                    details=details,
                )
                return _canonical_fence_error(
                    store,
                    f"canonical checkpoint not verified: {exc}",
                    **details,
                )
            if details.get("canonical_sync_blocked") is True:
                error_code = str(
                    details.get("canonical_sync_error_code") or ""
                )
                store.mark_canonical_sync_blocked(error_code)
                return tool_error(
                    f"canonical checkpoint not recorded: {exc}",
                    **_todo_result(previous_items, store=store),
                    todo_update_applied=False,
                    todo_preserved=True,
                    **details,
                )
            return tool_error(
                f"canonical checkpoint not recorded: {exc}",
                todo_update_applied=False,
                todo_preserved=True,
                **details,
            )
        checkpoint_binding = _binding_for_verified_checkpoint(
            effective_checkpoint,
            checkpoint_receipt,
            items,
        )
        result["canonical_checkpoint"] = checkpoint_receipt

    if plan_approval is not None:
        try:
            from tools.approval import (
                get_current_session_key,
                grant_plan_capability,
            )

            result["plan_capability"] = grant_plan_capability(
                # Normal model-tool dispatch does not carry gateway routing
                # authority as handler kwargs.  Resolve the stable key from
                # the context bound around the agent turn; never substitute
                # the transcript/session id.
                session_key=(
                    str(session_key or "").strip()
                    or get_current_session_key(default="")
                ),
                plan_id=plan_approval.get("plan_id", ""),
                plan_revision=plan_approval.get("plan_revision"),
                exact_commands=plan_approval.get("exact_commands") or [],
                approved_by_user_id=user_id,
                ttl_seconds=plan_approval.get("ttl_seconds", 3600),
                max_uses_per_command=plan_approval.get("max_uses_per_command", 3),
                canonical_case_id=plan_approval.get("canonical_case_id", ""),
                source_refs=plan_approval.get("source_refs") or {},
            )
        except Exception as exc:
            return tool_error(
                f"plan capability not granted: {exc}",
                todo_update_applied=False,
                todo_preserved=True,
            )
    if goal_outcome is not None:
        try:
            from hermes_cli.goals import GoalManager

            recorded = GoalManager(goal_session_id or session_key).record_model_outcome(
                goal_outcome.get("status", ""),
                goal_outcome.get("reason", ""),
                originating_turn_id=originating_turn_id,
                goal_generation_id=goal_generation_id,
            )
            result["goal_outcome"] = {"recorded": recorded}
        except Exception as exc:
            return tool_error(
                f"goal outcome not recorded: {exc}",
                todo_update_applied=False,
                todo_preserved=True,
            )
    if goal_contract is not None:
        try:
            from hermes_cli.goals import GoalManager

            recorded = GoalManager(goal_session_id or session_key).record_model_contract(
                goal_contract,
                originating_turn_id=originating_turn_id,
                goal_generation_id=goal_generation_id,
            )
            result["goal_contract"] = {"recorded": recorded}
        except Exception as exc:
            return tool_error(
                f"goal contract not recorded: {exc}",
                todo_update_applied=False,
                todo_preserved=True,
            )
    if delivery_outcome is not None:
        try:
            result["delivery_outcome"] = delivery_outcome_recorder(
                delivery_outcome
            )
        except Exception as exc:
            return tool_error(
                f"delivery outcome not recorded: {exc}",
                todo_update_applied=False,
                todo_preserved=True,
            )

    if canonical_checkpoint is not None:
        try:
            items = store.install_verified_canonical_snapshot(
                items,
                binding=checkpoint_binding,
            )
        except Exception as exc:
            details = {
                "canonical_write_may_have_occurred": True,
                "canonical_local_install_failed": True,
            }
            checkpoint_sha256 = _canonical_checkpoint_sha256(
                effective_checkpoint,
                items,
            )
            store.fence_canonical_uncertainty(
                checkpoint=canonical_checkpoint,
                effective_checkpoint=effective_checkpoint,
                items=items,
                checkpoint_sha256=checkpoint_sha256,
                details=details,
            )
            return _canonical_fence_error(
                store,
                f"canonical checkpoint local install not verified: {exc}",
                **details,
            )
        result = {
            **result,
            **_todo_result(items, store=store),
        }
    elif todos is not None:
        # Repeat the same deterministic normalization against the still-
        # unchanged live snapshot. Using the original raw list preserves the
        # historical coercion contract for multiple non-dict items.
        items = store.write(todos, merge=merge)
        if items != result["todos"]:
            raise RuntimeError("TodoStore candidate changed before final commit")
        result = {**result, **_todo_result(items, store=store)}
    elif not requested_controls:
        result = _todo_result(items, store=store)
    elif store.canonical_binding_state() is not None:
        result = {**result, **_todo_result(items, store=store)}
    return json.dumps(result, ensure_ascii=False)


class _CanonicalCheckpointError(RuntimeError):
    """Fail-closed checkpoint error carrying bounded mechanical evidence."""

    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.checkpoint_details = details


def _record_canonical_checkpoint(
    checkpoint: Dict[str, Any],
    items: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Persist the exact TodoStore snapshot as a Canonical Task Workspace plan.

    GPT authors the plan metadata and dependencies.  The runtime only attaches
    the exact in-memory todo items, validates the service receipt, and returns
    a compact correlation receipt.  No task text or user message is inspected.
    """
    from tools.canonical_brain_tool import (
        canonical_event_append_tool,
        check_canonical_brain_requirements,
    )

    if not check_canonical_brain_requirements():
        raise _CanonicalCheckpointError(
            "Canonical Brain writer is unavailable",
            canonical_sync_blocked=True,
            canonical_sync_error_code="canonical_brain_unavailable",
            canonical_write_may_have_occurred=False,
        )

    case_id = str(checkpoint.get("case_id") or "").strip()
    summary = str(checkpoint.get("summary") or "").strip()
    source_refs = checkpoint.get("source_refs")
    plan = checkpoint.get("plan")
    if not isinstance(source_refs, dict):
        raise _CanonicalCheckpointError("canonical_checkpoint.source_refs must be an object")
    if not isinstance(plan, dict):
        raise _CanonicalCheckpointError("canonical_checkpoint.plan must be an object")
    if "steps" in plan:
        raise _CanonicalCheckpointError(
            "canonical_checkpoint.plan.steps is runtime-owned; use todos"
        )

    canonical_plan = copy.deepcopy(plan)
    raw_dependencies = canonical_plan.pop("step_dependencies", {})
    if not isinstance(raw_dependencies, dict):
        raise _CanonicalCheckpointError(
            "canonical_checkpoint.plan.step_dependencies must be an object"
        )

    item_ids = {item["id"] for item in items}
    unknown_dependency_owners = sorted(
        str(item_id) for item_id in raw_dependencies if str(item_id) not in item_ids
    )
    if unknown_dependency_owners:
        raise _CanonicalCheckpointError(
            "step_dependencies contains unknown todo ids:"
            + ",".join(unknown_dependency_owners)
        )

    canonical_steps: List[Dict[str, Any]] = []
    for item in items:
        dependencies = raw_dependencies.get(item["id"], [])
        if not isinstance(dependencies, list):
            raise _CanonicalCheckpointError(
                f"step_dependencies.{item['id']} must be an array"
            )
        canonical_steps.append(
            {
                **item,
                "depends_on": [str(value).strip() for value in dependencies],
            }
        )
    canonical_plan["steps"] = canonical_steps

    payload: Dict[str, Any] = {"plan": canonical_plan}
    decision = checkpoint.get("decision")
    if decision is not None:
        if not isinstance(decision, dict):
            raise _CanonicalCheckpointError(
                "canonical_checkpoint.decision must be an object"
            )
        payload["decision"] = copy.deepcopy(decision)

    requested_idempotency_key = str(
        checkpoint.get("idempotency_key") or ""
    ).strip()
    if not requested_idempotency_key:
        # Bind every attempt to one exact key before the first dispatch.  The
        # writer already derives deterministic event identity for plan
        # revisions; this explicit key also keeps retry/readback reconciliation
        # stable if the first response is lost or malformed.
        requested_idempotency_key = "todo-checkpoint:" + hashlib.sha256(
            _stable_json(
                {
                    "case_id": case_id,
                    "summary": summary,
                    "source_refs": source_refs,
                    "actors": checkpoint.get("actors"),
                    "payload": payload,
                    "safety": checkpoint.get("safety"),
                }
            ).encode("utf-8")
        ).hexdigest()

    append_kwargs = {
        "event_type": "task.plan.updated",
        "case_id": case_id,
        "summary": summary,
        "source_refs": source_refs,
        "actors": checkpoint.get("actors"),
        "payload": payload,
        "safety": checkpoint.get("safety"),
        "idempotency_key": requested_idempotency_key,
    }
    response: Dict[str, Any] | None = None
    ambiguous_seen = False
    reconciliation_attempts = 0
    last_error = "readback not verified"
    last_status = ""
    for attempt in range(2):
        try:
            raw_response = canonical_event_append_tool(**append_kwargs)
        except Exception as exc:
            ambiguous_seen = True
            last_error = "Canonical Brain append raised before a receipt was observed"
            if attempt == 0:
                reconciliation_attempts += 1
                continue
            raise _CanonicalCheckpointError(
                last_error,
                canonical_status=last_status,
                canonical_write_may_have_occurred=True,
                canonical_reconciliation_attempts=reconciliation_attempts,
            ) from exc
        try:
            decoded = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError, UnicodeError) as exc:
            ambiguous_seen = True
            last_error = "Canonical Brain returned an invalid receipt"
            if attempt == 0:
                reconciliation_attempts += 1
                continue
            raise _CanonicalCheckpointError(
                last_error,
                canonical_status=last_status,
                canonical_write_may_have_occurred=True,
                canonical_reconciliation_attempts=reconciliation_attempts,
            ) from exc
        if not isinstance(decoded, dict):
            ambiguous_seen = True
            last_error = "Canonical Brain returned an invalid receipt"
            if attempt == 0:
                reconciliation_attempts += 1
                continue
            raise _CanonicalCheckpointError(
                last_error,
                canonical_status=last_status,
                canonical_write_may_have_occurred=True,
                canonical_reconciliation_attempts=reconciliation_attempts,
            )

        response = decoded
        last_status = str(response.get("status") or "")
        last_error = str(
            response.get("error") or last_status or "readback not verified"
        )
        verified = (
            response.get("success") is True
            and response.get("readback_verified") is True
            and response.get("event_type") == "task.plan.updated"
            and response.get("case_id") == case_id
            and response.get("idempotency_key") == requested_idempotency_key
            and bool(str(response.get("event_id") or "").strip())
            and len(str(response.get("canonical_content_sha256") or "")) == 64
            and all(
                char in "0123456789abcdef"
                for char in str(response.get("canonical_content_sha256") or "")
            )
        )
        if verified:
            break

        # A deterministic plan-transition identity occupied by different
        # Canonical content is not a transient writer outage and not an
        # uncertain append.  The current invocation provably did not write,
        # but authoritative truth has diverged from the model's candidate.
        # Preserve the prior local snapshot mechanically, block execution, and
        # require the model to read Canonical state before authoring a new
        # revision/supersession. Retrying the same checkpoint can never fix a
        # permanent content conflict.
        proven_content_conflict = (
            last_status == "CANONICAL_EVENT_APPEND_IDEMPOTENCY_CONFLICT"
            and response.get("success") is False
            and response.get("readback_verified") is False
            and response.get("write_may_have_occurred") is False
            and response.get("inserted") is False
            and response.get("deduped") is True
        )
        if proven_content_conflict:
            raise _CanonicalCheckpointError(
                last_error,
                canonical_status=last_status,
                canonical_write_may_have_occurred=False,
                canonical_reconciliation_attempts=reconciliation_attempts,
                canonical_sync_blocked=True,
                canonical_sync_error_code="canonical_truth_diverged",
            )

        # Missing certainty is unknown, never proof of no write. Only an
        # explicit ``False`` produced by the mechanical writer boundary may
        # preserve the prior local snapshot without reconciliation/fencing.
        current_ambiguous = bool(
            response.get("write_may_have_occurred") is not False
            or response.get("inserted") is True
            or response.get("success") is True
            or response.get("readback_verified") is True
        )
        ambiguous_seen = ambiguous_seen or current_ambiguous
        if current_ambiguous and attempt == 0:
            reconciliation_attempts += 1
            continue
        raise _CanonicalCheckpointError(
            last_error,
            canonical_status=last_status,
            canonical_write_may_have_occurred=ambiguous_seen,
            canonical_reconciliation_attempts=reconciliation_attempts,
            canonical_sync_blocked=(
                not ambiguous_seen
                and str(response.get("canonical_error_code") or "")
                in _CANONICAL_PREDISPATCH_UNAVAILABLE_CODES
            ),
            canonical_sync_error_code=(
                "canonical_brain_unavailable"
                if (
                    not ambiguous_seen
                    and str(response.get("canonical_error_code") or "")
                    in _CANONICAL_PREDISPATCH_UNAVAILABLE_CODES
                )
                else ""
            ),
            canonical_writer_error_code=str(
                response.get("canonical_error_code") or ""
            ),
        )

    if response is None:
        raise _CanonicalCheckpointError(
            last_error,
            canonical_status=last_status,
            canonical_write_may_have_occurred=True,
            canonical_reconciliation_attempts=reconciliation_attempts,
        )

    workspace_sha256 = hashlib.sha256(
        json.dumps(
            canonical_steps,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "success": True,
        "status": str(response.get("status") or ""),
        "event_id": str(response.get("event_id") or ""),
        "event_type": "task.plan.updated",
        "case_id": case_id,
        "plan_id": str(canonical_plan.get("plan_id") or ""),
        "plan_revision": canonical_plan.get("revision"),
        "plan_state": str(canonical_plan.get("state") or ""),
        "idempotency_key": str(response.get("idempotency_key") or ""),
        "canonical_content_sha256": str(
            response.get("canonical_content_sha256") or ""
        ),
        "workspace_todos_sha256": workspace_sha256,
        "todo_items_sha256": _todo_items_sha256(items),
        "readback_verified": True,
        "inserted": bool(response.get("inserted")),
        "deduped": bool(response.get("deduped")),
        "canonical_reconciliation_attempts": reconciliation_attempts,
    }


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================
# Behavioral guidance is baked into the description so it's part of the
# static tool schema (cached, never changes mid-conversation).

TODO_SCHEMA = {
    "name": "todo",
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled|blocked}\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If one approach fails, "
        "add a revised item and keep working. Use blocked only when every safe "
        "available approach is exhausted and user/external input is genuinely required.\n\n"
        "Canonical controls:\n"
        "- canonical_checkpoint, plan_approval, goal_outcome, goal_contract, and "
        "delivery_outcome each perform a durable control side effect\n"
        "- Submit exactly one control field per todo call. Never combine todos "
        "with a control field except canonical_checkpoint\n"
        "- For an already Canonical-bound plan, submit the updated todos and "
        "the model-authored canonical_checkpoint in the SAME call. Runtime "
        "validates the candidate, requires Canonical readback, then installs "
        "the exact local snapshot atomically\n"
        "- An unbound draft may still be written first and checkpointed in a "
        "following canonical_checkpoint-only call\n\n"
        "Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": (
                    "Task items to write. Omit to read current list. For a "
                    "Canonical-bound plan, combine them with canonical_checkpoint "
                    "for one atomic readback-verified update. Never combine them "
                    "with plan_approval, goal_outcome, goal_contract, or "
                    "delivery_outcome."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique item identifier"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled", "blocked"],
                            "description": "Current status"
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list."
                ),
                "default": False
            },
            "reasoning": {
                "type": "object",
                "description": (
                    "For exact GPT-5.6 on the verified OpenAI Codex Responses "
                    "backend only: request reasoning depth for later model calls "
                    "in this current turn. You decide the effort; "
                    "the runtime only validates the operator's baseline/cap and returns "
                    "a receipt. You may choose max for the most demanding work. "
                    "The mechanical policy receipt rejects any effort unavailable "
                    "under the configured runtime cap."
                ),
                "properties": {
                    "effort": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "xhigh", "max"],
                    },
                },
                "required": ["effort"],
                "additionalProperties": False,
            },
            "plan_approval": {
                "type": "object",
                "description": (
                    "Use only after the authenticated owner explicitly approves this plan. "
                    "Hermes decides approval meaning and authors the lease bounds; runtime grants "
                    "only exact expiring commands. One grant remains valid across monotonic progress "
                    "revisions of the same active plan, but never authorizes a newly added command."
                ),
                "properties": {
                    "plan_id": {"type": "string"},
                    "plan_revision": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Exact active Canonical Task Workspace revision approved by the owner."
                        ),
                    },
                    "exact_commands": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64},
                    "ttl_seconds": {
                        "type": "integer",
                        "minimum": 60,
                        "maximum": 28800,
                        "description": (
                            "Choose the smallest 60..28800 second lease that can cover the already-"
                            "approved plan end to end without a time-only reapproval. This does not "
                            "expand the exact command set, active plan, owner, or session epoch."
                        ),
                    },
                    "max_uses_per_command": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": (
                            "Choose the smallest per-exact-command use count that covers planned "
                            "retries and verification without step-by-step reapproval. Uses remain "
                            "bound to each approved command hash."
                        ),
                    },
                    "canonical_case_id": {
                        "type": "string",
                        "description": (
                            "Exact case: id for a durable Canonical Brain approval receipt. "
                            "Omit only when Canonical Brain is unavailable."
                        ),
                    },
                    "source_refs": {
                        "type": "object",
                        "description": "Exact approval message/session refs; runtime fills observed refs when omitted.",
                    },
                },
                "required": [
                    "plan_id",
                    "plan_revision",
                    "exact_commands",
                    "ttl_seconds",
                    "max_uses_per_command",
                ],
                "additionalProperties": False,
            },
            "goal_outcome": {
                "type": "object",
                "description": (
                    "When a standing /goal is active, record your own outcome for this turn. "
                    "Use complete only with concrete verification. Use blocked only after all "
                    "safe available approaches are exhausted; otherwise continue."
                ),
                "properties": {
                    "status": {"type": "string", "enum": ["continue", "complete", "blocked"]},
                    "reason": {"type": "string"},
                },
                "required": ["status", "reason"],
            },
            "goal_contract": {
                "type": "object",
                "description": (
                    "When a standing /goal needs a concrete completion contract, author it "
                    "yourself from the full conversation context. This primary-model-authored "
                    "object becomes the durable definition of done; no auxiliary planner or "
                    "judge rewrites it."
                ),
                "properties": {
                    "outcome": {"type": "string"},
                    "verification": {"type": "string"},
                    "constraints": {"type": "string"},
                    "boundaries": {"type": "string"},
                    "stop_when": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "canonical_checkpoint": {
                "type": "object",
                "description": (
                    "Bind the current exact TodoStore snapshot to a durable "
                    "Canonical Task Workspace task.plan.updated event. You author "
                    "the plan meaning, dependencies, success criteria, cursor, and "
                    "state; the runtime only attaches the exact candidate todos, "
                    "validates, writes, and requires a readback receipt. For an "
                    "already bound plan, include the updated todos in this same call: "
                    "the local snapshot is installed only after Canonical readback. "
                    "An unbound draft can instead be checkpointed alone. A proven "
                    "pre-dispatch failure preserves current local state; an uncertain "
                    "append is retried with the exact idempotency key and fences local "
                    "state until verified."
                ),
                "properties": {
                    "case_id": {
                        "type": "string",
                        "description": "Exact Canonical Brain case id starting with case:",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short model-authored operational summary",
                    },
                    "source_refs": {
                        "type": "object",
                        "description": (
                            "Exact source refs. Runtime fills observed gateway refs "
                            "when available."
                        ),
                    },
                    "plan": {
                        "type": "object",
                        "description": (
                            "Model-authored durable plan metadata. Do not provide steps; "
                            "runtime attaches the exact todos from the current store."
                        ),
                        "properties": {
                            "plan_id": {"type": "string"},
                            "revision": {"type": "integer", "minimum": 1},
                            "objective": {"type": "string"},
                            "state": {
                                "type": "string",
                                "enum": ["active", "completed", "blocked", "cancelled"],
                            },
                            "success_criteria": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 32,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "content": {"type": "string"},
                                    },
                                    "required": ["id", "content"],
                                },
                            },
                            "current_step_id": {"type": "string"},
                            "step_dependencies": {
                                "type": "object",
                                "description": (
                                    "Model-authored mapping of todo id to prerequisite "
                                    "todo ids. Omitted ids have no dependencies."
                                ),
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "resume_cursor": {
                                "type": "object",
                                "properties": {
                                    "summary": {"type": "string"},
                                    "next_step_id": {"type": "string"},
                                },
                                "required": ["summary"],
                            },
                            "attempts": {
                                "type": "array",
                                "items": {"type": "object"},
                                "maxItems": 64,
                            },
                            "decisions": {
                                "type": "array",
                                "items": {"type": "object"},
                                "maxItems": 64,
                            },
                            "artifacts": {
                                "type": "array",
                                "items": {"type": "object"},
                                "maxItems": 64,
                            },
                            "verification_event_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 64,
                            },
                            "blocker": {"type": "object"},
                            "supersedes_plan_id": {"type": "string"},
                            "supersedes_plan_revision": {
                                "type": "integer",
                                "minimum": 1,
                            },
                        },
                        "required": [
                            "plan_id",
                            "revision",
                            "objective",
                            "state",
                            "success_criteria",
                            "resume_cursor",
                        ],
                        "additionalProperties": False,
                    },
                    "decision": {
                        "type": "object",
                        "description": "Optional model-authored decision/evidence metadata",
                    },
                    "actors": {"type": "object"},
                    "safety": {"type": "object"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["case_id", "summary", "source_refs", "plan"],
                "additionalProperties": False,
            },
            "delivery_outcome": {
                "type": "object",
                "description": (
                    "Choose whether this exact completed turn should be delivered "
                    "to the user. This is your semantic decision: the runtime only "
                    "validates and executes it. Use suppress when the turn should "
                    "produce no outbound final message (for example, a scheduled "
                    "check found nothing worth reporting). Use deliver to replace "
                    "an earlier suppress choice in the same turn. The choice resets "
                    "at every new user/cron turn; response text is never inspected "
                    "for control words. Failed turns are always delivered."
                ),
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["deliver", "suppress"],
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                    },
                },
                "required": ["action", "reason"],
                "additionalProperties": False,
            },
        },
        "required": []
    }
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="todo",
    toolset="todo",
    schema=TODO_SCHEMA,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"), merge=args.get("merge", False), store=kw.get("store"),
        plan_approval=args.get("plan_approval"),
        goal_outcome=args.get("goal_outcome"),
        goal_contract=args.get("goal_contract"),
        canonical_checkpoint=args.get("canonical_checkpoint"),
        delivery_outcome=args.get("delivery_outcome"),
        delivery_outcome_recorder=kw.get("delivery_outcome_recorder"),
        session_key=str(kw.get("session_key") or ""),
        goal_session_id=str(
            kw.get("goal_session_id")
            or kw.get("session_id")
            or kw.get("session_key")
            or ""
        ),
        originating_turn_id=str(
            kw.get("originating_turn_id") or kw.get("turn_id") or ""
        ),
        goal_generation_id=str(kw.get("goal_generation_id") or ""),
        user_id=str(kw.get("user_id") or "")),
    check_fn=check_todo_requirements,
    emoji="📋",
)
