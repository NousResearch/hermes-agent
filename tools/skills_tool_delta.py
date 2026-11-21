"""skill_view changed-file deltas, verified against the live transcript.

After the agent edits a skill with ``skill_manage``, it usually re-reads it with ``skill_view`` to
check the edit. The file changed, so the repeat-view dedup stub cannot apply and the whole skill is
sent again, often tens of KB to show a few hundred characters of change.

``maybe_skill_view_delta`` replaces such a full result with a unified diff, but only when the
baseline the diff applies to is *demonstrably* in the transcript the model will see: it rebuilds the
latest text of that skill file by replaying the ``skill_view`` results present in ``messages`` (the
last full body plus every later delta, each hash-checked) and diffs the fresh result against that.
The dedup cache is never trusted for this. So rewinds, edit-and-resend, prunes, compaction, session
forks and subagents (each has its own ``messages``), plugin/guardrail annotations, persisted
previews, and concurrent writers all degrade to the full result instead of a diff against content
the model does not have.

It is applied in one place — ``agent.tool_executor._commit_tool_result``, where results join the
transcript. Every other caller of the ``skill_view`` handler (execute_code RPC, cron, plugins,
slash/preload paths) keeps receiving full content. ``skill_view(full=true)`` always returns the
complete file.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A delta is served only when it is at most this fraction of the full result; larger rewrites
# read better whole and save little.
DELTA_MAX_RATIO = 0.40
_CONTEXT_LINES = 3
_NO_NL = "\\ No newline at end of file\n"
_HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

DELTA_MESSAGE = (
    "This skill file changed since the copy already in this conversation. Apply `diff` to the most "
    "recent skill_view result for this file (its text hashes to `base_sha256`) to get the complete "
    "current file; `changed_fields` lists other payload fields that changed. Nothing else changed. "
    "If that earlier result is not fully visible to you, call skill_view again with full=true.")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def skill_view_delta(base: str, new: str) -> str:
    """Unified diff ``base`` -> ``new`` with explicit no-trailing-newline markers."""
    out = []
    for line in difflib.unified_diff(base.splitlines(keepends=True), new.splitlines(keepends=True),
                                     "earlier", "current", n=_CONTEXT_LINES):
        out.append(line if line.endswith("\n") else line + "\n" + _NO_NL)
    return "".join(out)


def apply_skill_view_delta(base: str, diff: str) -> str:
    """Strictly apply a diff from :func:`skill_view_delta`; ValueError on any mismatch."""
    src, lines = base.splitlines(keepends=True), diff.splitlines(keepends=True)
    out: list[str] = []
    pos, i = 0, 0
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1
    while i < len(lines):
        m = _HUNK_RE.match(lines[i])
        if not m:
            raise ValueError(f"bad hunk header: {lines[i]!r}")
        start = int(m.group(1)) - (0 if m.group(2) == "0" else 1)
        if start < pos:
            raise ValueError("overlapping hunks")
        out.extend(src[pos:start])
        pos = start
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            tag, body = lines[i][:1], lines[i][1:]
            if tag not in (" ", "-", "+"):
                raise ValueError(f"bad diff line: {lines[i]!r}")
            if i + 1 < len(lines) and lines[i + 1] == _NO_NL:
                body = body[:-1]
                i += 1
            if tag in " -":
                if pos >= len(src) or src[pos] != body:
                    raise ValueError(f"context mismatch at base line {pos + 1}")
                pos += 1
            if tag in " +":
                out.append(body)
            i += 1
    out.extend(src[pos:])
    return "".join(out)


def _changed_sections(new: str, diff: str) -> list[str]:
    """Markdown heading above the first changed line of each hunk (diagnostic aid)."""
    heads: list[str] = []
    lines, n = new.splitlines(), None
    for line in diff.splitlines():
        if m := _HUNK_RE.match(line):
            n = int(m.group(3))
            continue
        if n is None or line.startswith(("---", "+++", "\\")):
            continue
        if line[:1] == " ":
            n += 1
            continue
        head = next((lines[k].strip() for k in range(min(n, len(lines)) - 1, -1, -1)
                     if lines[k].startswith("#")), "(top of file)")
        if head not in heads:
            heads.append(head)
        n = None
    return heads


def _key(payload: dict) -> tuple[str, str]:
    """(resolved skill name, file) — full SKILL.md bodies carry no ``file``; stubs say SKILL.md."""
    f = payload.get("file") or ""
    return str(payload.get("name") or ""), "" if f == "SKILL.md" else str(f)


def _base_name(name: Any) -> str:
    return str(name or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def _skill_view_call_names(messages: list) -> dict[str, str]:
    names: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or ():
            fn = (tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)) or {}
            fname = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if fname != "skill_view":
                continue
            raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (TypeError, ValueError):
                args = {}
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id:
                names[str(tc_id)] = _base_name(args.get("name") if isinstance(args, dict) else "")
    return names


def transcript_skill_view_state(messages: Optional[list], key: tuple[str, str]
                                ) -> Optional[tuple[str, dict]]:
    """Rebuild ``(text, metadata)`` of skill file ``key`` as visible in ``messages``, or None.

    Replays ``skill_view`` tool results in order: a full body sets the state, a delta must apply
    exactly to it (both hashes checked), an "unchanged" stub keeps it. Any ``skill_view`` row for the
    same skill that is not a parseable result (pruned/demoted marker, persisted preview, annotation)
    clears it — the model may not have that content.
    """
    if not messages:
        return None
    call_names = None
    state: Optional[tuple[str, dict]] = None
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        if (msg.get("tool_name") or msg.get("name")) != "skill_view":
            continue
        content = msg.get("content")
        payload = None
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except ValueError:
                payload = None
        if not isinstance(payload, dict):
            if call_names is None:
                call_names = _skill_view_call_names(messages)
            called = call_names.get(str(msg.get("tool_call_id") or ""))
            if called is None or called == _base_name(key[0]):
                state = None
            continue
        if not payload.get("success") or _key(payload) != key:
            continue
        if payload.get("delta"):
            if state is None:
                continue
            try:
                text, meta = state
                if _sha(text)[:16] != payload.get("base_sha256"):
                    raise ValueError("base hash mismatch")
                new = apply_skill_view_delta(text, str(payload.get("diff") or ""))
                if _sha(new)[:16] != payload.get("current_sha256"):
                    raise ValueError("result hash mismatch")
                meta = {**meta, **(payload.get("changed_fields") or {})}
                state = (new, meta)
            except (ValueError, TypeError):
                state = None
        elif payload.get("dedup"):
            continue
        elif isinstance(payload.get("content"), str):
            state = (payload["content"], {k: v for k, v in payload.items() if k != "content"})
    return state


def maybe_skill_view_delta(result: Any, messages: Optional[list], args: Optional[dict] = None) -> Any:
    """Return a delta payload for a full ``skill_view`` result when the transcript holds the
    baseline and the delta is small; otherwise return ``result`` unchanged."""
    if not isinstance(result, str) or not messages or (args or {}).get("full"):
        return result
    try:
        payload = json.loads(result)
    except ValueError:
        return result
    if (not isinstance(payload, dict) or not payload.get("success")
            or not isinstance(payload.get("content"), str) or payload.get("dedup")
            or payload.get("delta") or payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"):
        return result
    key = _key(payload)
    state = transcript_skill_view_state(messages, key)
    if state is None:
        return result
    base_text, base_meta = state
    text = payload["content"]
    meta = {k: v for k, v in payload.items() if k != "content"}
    if base_text == text:
        return result  # unchanged text: the repeat-view dedup stub owns that case
    diff = skill_view_delta(base_text, text)
    changed = {k: meta.get(k) for k in sorted(set(meta) | set(base_meta)) if meta.get(k) != base_meta.get(k)}
    delta = json.dumps({
        "success": True, "name": key[0], "file": key[1] or "SKILL.md", "status": "changed",
        "delta": True, "content_returned": "delta", "base_sha256": _sha(base_text)[:16],
        "current_sha256": _sha(text)[:16], "current_chars": len(text),
        "changed_sections": _changed_sections(text, diff), "changed_fields": changed,
        "diff": diff, "message": DELTA_MESSAGE}, ensure_ascii=False)
    return delta if len(delta) <= DELTA_MAX_RATIO * len(result) else result


def skill_view_result_for_transcript(result: Any, messages: Optional[list], args: Any = None) -> Any:
    """Executor entry point (``agent.tool_executor._commit_tool_result``): never in the
    background-review fork, whose read-before-write guard needs real bodies (#95976); any
    unexpected payload shape keeps the full result."""
    from tools.skill_provenance import is_background_review

    if is_background_review():
        return result
    try:
        return maybe_skill_view_delta(result, messages, args if isinstance(args, dict) else None)
    except (ValueError, TypeError, KeyError, AttributeError):
        logger.debug("skill_view delta skipped; serving the full result", exc_info=True)
        return result
