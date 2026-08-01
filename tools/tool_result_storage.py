"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written INTO THE
   SANDBOX temp dir (for example /tmp/hermes-results/{tool_use_id}.txt on
   standard Linux, or $TMPDIR/hermes-results/{tool_use_id}.txt on Termux)
   via env.execute(). The in-context content is replaced with a preview +
   file path reference. The model can read_file to access the full output
   on any backend.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.
"""

import hashlib
import json
import logging
import os
import re
import shlex
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from tools.budget_config import (
    DEFAULT_RESULT_TOKEN_LIMIT,
    MAX_RESULT_TOKEN_LIMIT,
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120
# Tool-result images use provider-native vision accounting rather than the
# textual token budget below. They still need a hard availability bound: a
# poisoned or malformed tool must not attach an arbitrarily large data URI or
# opaque multipart value to one result. Four MiB matches the existing image
# shrink target used by conversation_compression.
MAX_NON_TEXT_RESULT_BYTES = 4 * 1024 * 1024
# Across one tool turn, keep at most two normal screenshot-sized payloads. The
# newest parts win; older opaque parts become bounded textual markers.
MAX_TURN_NON_TEXT_RESULT_BYTES = 8 * 1024 * 1024


class TokenCounter(Protocol):
    """Counter contract used by the model-visible tool-result hardline."""

    method: str
    exact: bool

    def count(self, text: str) -> int:
        """Return the number of budget units in ``text``."""


class Utf8ByteTokenCounter:
    """Conservative local fallback for byte-backed provider tokenizers."""

    method = "utf8_bytes"
    exact = False

    def count(self, text: str) -> int:
        return len(_sanitize_unicode(text).encode("utf-8"))


DEFAULT_TOKEN_COUNTER: TokenCounter = Utf8ByteTokenCounter()


@dataclass(frozen=True)
class ToolResultBudgetOutcome:
    """Non-sensitive evidence for one final model-visible result."""

    schema_version: int
    limit_tokens: int
    counter_method: str
    exact_counter: bool
    initial_count: int
    final_count: int
    initial_utf8_bytes: int
    final_utf8_bytes: int
    truncated: bool
    override_requested: bool
    persisted: bool
    persisted_path: str | None

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _sanitize_unicode(text: str) -> str:
    """Return valid Unicode without normalization or surrogate exceptions."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _is_multimodal_envelope(content: Any) -> bool:
    """Return whether content is the registry's recognized multimodal envelope."""
    return (
        isinstance(content, dict)
        and content.get("_multimodal") is True
        and isinstance(content.get("content"), list)
    )


def _text_values(content: Any) -> list[str]:
    if isinstance(content, str):
        return [_sanitize_unicode(content)]
    if _is_multimodal_envelope(content):
        nested = _text_values(content["content"])
        summary = content.get("text_summary")
        if not isinstance(summary, str):
            return nested
        sanitized_summary = _sanitize_unicode(summary)
        nested_bytes = sum(len(text.encode("utf-8")) for text in nested)
        if len(sanitized_summary.encode("utf-8")) > nested_bytes:
            return [sanitized_summary]
        return nested or [sanitized_summary]
    if isinstance(content, list):
        values = []
        for part in content:
            if isinstance(part, str):
                values.append(_sanitize_unicode(part))
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                values.append(_sanitize_unicode(part["text"]))
        return values
    return []


def _is_text_only_part(part: Any) -> bool:
    """Return whether a multipart value contains text and no opaque payload."""
    return (
        isinstance(part, dict)
        and isinstance(part.get("text"), str)
        and set(part).issubset({"type", "text"})
    )


def _non_text_omission_marker(max_bytes: int) -> dict[str, str]:
    return {
        "type": "text",
        "text": (
            "[Non-text tool-result part omitted: aggregate payload "
            f"exceeded {max_bytes:,} bytes.]"
        ),
    }


def _replace_text_values(content: Any, replacement: str) -> Any:
    """Replace aggregate list text once while preserving all non-text parts."""
    if isinstance(content, str):
        return replacement
    if _is_multimodal_envelope(content):
        result = {
            **content,
            "content": _replace_text_values(content["content"], replacement),
        }
        if isinstance(content.get("text_summary"), str):
            result["text_summary"] = replacement
        return result
    if not isinstance(content, list):
        return content
    replaced = False
    result = []
    for part in content:
        if isinstance(part, str):
            if not replaced:
                result.append(replacement)
                replaced = True
            else:
                result.append("")
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            if not replaced:
                result.append({**part, "text": replacement})
                replaced = True
            else:
                result.append({**part, "text": ""})
        else:
            result.append(part)
    return result


def _normalize_model_visible_text(content: Any) -> Any:
    if isinstance(content, str):
        return _sanitize_unicode(content)
    if _is_multimodal_envelope(content):
        result = {
            **content,
            "content": _normalize_model_visible_text(content["content"]),
        }
        if isinstance(content.get("text_summary"), str):
            result["text_summary"] = _sanitize_unicode(content["text_summary"])
        return result
    if not isinstance(content, list):
        return content
    return [
        _sanitize_unicode(part)
        if isinstance(part, str)
        else (
            {**part, "text": _sanitize_unicode(part["text"])}
            if isinstance(part, dict) and isinstance(part.get("text"), str)
            else part
        )
        for part in content
    ]


def _serialized_part_size(part: Any) -> int:
    """Return a conservative UTF-8 size for one non-text content part."""
    try:
        rendered = json.dumps(
            part,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        rendered = repr(part)
    return len(_sanitize_unicode(rendered).encode("utf-8"))


def _non_text_payload_size(content: Any) -> int:
    """Return serialized bytes for opaque multipart parts only."""
    if _is_multimodal_envelope(content):
        return _non_text_payload_size(content["content"])
    if isinstance(content, dict):
        return _serialized_part_size(content)
    if not isinstance(content, list):
        return 0
    return sum(
        _serialized_part_size(part)
        for part in content
        if not (
            isinstance(part, str)
            or _is_text_only_part(part)
        )
    )


def _bound_non_text_parts(
    content: Any,
    *,
    max_bytes: int | None = None,
) -> tuple[Any, bool]:
    """Bound aggregate opaque multipart payload while preserving normal images."""
    if max_bytes is None:
        max_bytes = MAX_NON_TEXT_RESULT_BYTES
    max_bytes = max(0, max_bytes)
    if _is_multimodal_envelope(content):
        bounded, omitted = _bound_non_text_parts(
            content["content"],
            max_bytes=max_bytes,
        )
        if not omitted:
            return content, False
        return {**content, "content": bounded}, True
    if isinstance(content, dict):
        if _serialized_part_size(content) <= max_bytes:
            return content, False
        return [_non_text_omission_marker(max_bytes)], True
    if not isinstance(content, list):
        return content, False

    used = 0
    omitted = False
    result = []
    for part in content:
        if isinstance(part, str) or _is_text_only_part(part):
            result.append(part)
            continue
        size = _serialized_part_size(part)
        if used + size <= max_bytes:
            result.append(part)
            used += size
            continue
        omitted = True
        result.append(_non_text_omission_marker(max_bytes))
    return result, omitted


def _count_texts(
    texts: list[str],
    counter: TokenCounter,
) -> tuple[int, TokenCounter]:
    try:
        counts = [counter.count(text) for text in texts]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("invalid token counter result")
        return sum(counts), counter
    except Exception as exc:
        logger.debug("Tool-result token counter failed; using UTF-8 fallback: %s", exc)
        fallback = DEFAULT_TOKEN_COUNTER
        return sum(fallback.count(text) for text in texts), fallback


def model_visible_text_count(
    content: Any,
    counter: TokenCounter | None = None,
) -> int:
    """Count text in a string, content-part list, or multimodal envelope."""
    count, _ = _count_texts(_text_values(content), counter or DEFAULT_TOKEN_COUNTER)
    return count


def _utf8_prefix(raw: bytes, length: int) -> str:
    return raw[:max(0, length)].decode("utf-8", errors="ignore")


def _utf8_suffix(raw: bytes, length: int) -> str:
    if length <= 0:
        return ""
    return raw[-length:].decode("utf-8", errors="ignore")


def _compose_bounded_text(text: str, marker: str, byte_budget: int) -> str:
    """Keep both ends so structural closing delimiters survive truncation."""
    text_bytes = _sanitize_unicode(text).encode("utf-8")
    marker_bytes = _sanitize_unicode(marker).encode("utf-8")
    if len(marker_bytes) >= byte_budget:
        return _utf8_prefix(marker_bytes, byte_budget)
    remaining = byte_budget - len(marker_bytes)
    head_budget = (remaining * 3) // 4
    tail_budget = remaining - head_budget
    return (
        _utf8_prefix(text_bytes, head_budget)
        + marker_bytes.decode("utf-8")
        + _utf8_suffix(text_bytes, tail_budget)
    )


def _truncate_with_counter(
    text: str,
    marker: str,
    limit_tokens: int,
    counter: TokenCounter,
) -> tuple[str, TokenCounter]:
    """Build and verify a bounded head/marker/tail representation."""
    # Exact counters may use up to four UTF-8 bytes per requested token; the
    # fallback itself is stricter and uses one byte per budget unit.
    byte_budget = (
        limit_tokens
        if counter.method == DEFAULT_TOKEN_COUNTER.method and not counter.exact
        else limit_tokens * 4
    )
    candidate = _compose_bounded_text(text, marker, byte_budget)
    count, effective_counter = _count_texts([candidate], counter)

    for _ in range(20):
        if count <= limit_tokens:
            return candidate, effective_counter
        next_budget = max(
            0,
            min(byte_budget - 1, int(byte_budget * limit_tokens / max(count, 1) * 0.9)),
        )
        if next_budget >= byte_budget:
            next_budget = byte_budget - 1
        byte_budget = next_budget
        candidate = _compose_bounded_text(text, marker, byte_budget)
        count, effective_counter = _count_texts([candidate], effective_counter)

    # A broken/non-monotone injected counter must never weaken the hardline.
    fallback = DEFAULT_TOKEN_COUNTER
    return _compose_bounded_text(text, marker, limit_tokens), fallback


def _read_file_continuation_marker(
    *,
    limit_tokens: int,
    source_args: Mapping[str, Any] | None,
) -> str:
    source_args = source_args or {}
    path = source_args.get("path")
    offset = source_args.get("offset", 0)
    original_limit = source_args.get("limit")
    target = f" path={path!r}," if path else ""
    page = f" offset={offset!r}"
    if original_limit is not None:
        page += f", prior_limit={original_limit!r}"
    return (
        "\n\n[read_file page truncated to the model-visible "
        f"{limit_tokens:,}-unit limit; re-run read_file with{target}{page} "
        "and a smaller limit to continue from the original source. "
        "This page was not persisted as a new handle.]\n\n"
    )


def finalize_model_visible_tool_result(
    content: Any,
    *,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    limit_tokens: int = DEFAULT_RESULT_TOKEN_LIMIT,
    override_requested: bool = False,
    source_args: Mapping[str, Any] | None = None,
    token_counter: TokenCounter | None = None,
    trusted_persisted: bool = False,
) -> tuple[Any, ToolResultBudgetOutcome]:
    """Apply the universal per-result bound after every result transformation."""
    if (
        isinstance(limit_tokens, bool)
        or not isinstance(limit_tokens, int)
        or not 1 <= limit_tokens <= MAX_RESULT_TOKEN_LIMIT
    ):
        limit_tokens = DEFAULT_RESULT_TOKEN_LIMIT
        override_requested = False

    normalized = _normalize_model_visible_text(content)
    normalized, non_text_omitted = _bound_non_text_parts(normalized)
    texts = _text_values(normalized)
    initial_utf8_bytes = sum(len(text.encode("utf-8")) for text in texts)
    requested_counter = token_counter or DEFAULT_TOKEN_COUNTER
    if not requested_counter.exact and requested_counter is not DEFAULT_TOKEN_COUNTER:
        requested_counter = DEFAULT_TOKEN_COUNTER
    initial_count, effective_counter = _count_texts(texts, requested_counter)
    byte_limit = (
        limit_tokens
        if effective_counter.method == DEFAULT_TOKEN_COUNTER.method
        and not effective_counter.exact
        else limit_tokens * 4
    )
    text_needs_truncation = (
        initial_count > limit_tokens or initial_utf8_bytes > byte_limit
    )
    needs_truncation = text_needs_truncation or non_text_omitted
    persisted = False
    persisted_path = None
    final_content = normalized

    if text_needs_truncation and texts:
        full_text = "\n\n".join(texts)
        already_persisted = trusted_persisted
        remote_path = None
        if tool_name != "read_file" and not already_persisted and env is not None:
            remote_path = (
                f"{_resolve_storage_dir(env)}/{_unique_result_filename(tool_use_id)}"
            )
            try:
                persisted = _write_to_sandbox(full_text, remote_path, env)
                if persisted:
                    persisted_path = remote_path
            except Exception as exc:
                logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)
                persisted = False

        if tool_name == "read_file":
            marker = _read_file_continuation_marker(
                limit_tokens=limit_tokens,
                source_args=source_args,
            )
        elif persisted and remote_path:
            marker = (
                f"\n\n{PERSISTED_OUTPUT_TAG}\n"
                "[Truncated tool result to the model-visible "
                f"{limit_tokens:,}-unit limit. Full textual output saved to: "
                f"{remote_path}. Use read_file with offset and limit to retrieve "
                "specific sections.]\n"
                f"{PERSISTED_OUTPUT_CLOSING_TAG}\n\n"
            )
        elif already_persisted:
            marker = (
                "\n\n[Persisted-output preview further reduced to the "
                f"model-visible {limit_tokens:,}-unit limit; use the existing "
                "handle above rather than creating another one.]\n\n"
            )
        else:
            marker = (
                "\n\n[Truncated tool result to the model-visible "
                f"{limit_tokens:,}-unit limit. Full output could not be saved "
                "to the active sandbox.]\n\n"
            )

        truncated_text, effective_counter = _truncate_with_counter(
            full_text,
            marker,
            limit_tokens,
            effective_counter,
        )
        final_content = _replace_text_values(normalized, truncated_text)

    final_texts = _text_values(final_content)
    final_count, effective_counter = _count_texts(final_texts, effective_counter)
    final_utf8_bytes = sum(len(text.encode("utf-8")) for text in final_texts)

    # Last-resort enforcement uses the declared conservative counter. This is
    # intentionally redundant: no injected counter failure can release an
    # oversized string.
    if final_count > limit_tokens or final_utf8_bytes > (
        limit_tokens
        if effective_counter.method == DEFAULT_TOKEN_COUNTER.method
        and not effective_counter.exact
        else limit_tokens * 4
    ):
        fallback_text = "\n\n".join(final_texts)
        fallback_marker = (
            f"\n\n[Tool result reduced to the {limit_tokens:,}-byte fallback limit.]\n\n"
        )
        bounded, effective_counter = _truncate_with_counter(
            fallback_text,
            fallback_marker,
            limit_tokens,
            DEFAULT_TOKEN_COUNTER,
        )
        final_content = _replace_text_values(final_content, bounded)
        final_texts = _text_values(final_content)
        final_count = sum(DEFAULT_TOKEN_COUNTER.count(text) for text in final_texts)
        final_utf8_bytes = sum(len(text.encode("utf-8")) for text in final_texts)

    outcome = ToolResultBudgetOutcome(
        schema_version=1,
        limit_tokens=limit_tokens,
        counter_method=effective_counter.method,
        exact_counter=effective_counter.exact,
        initial_count=initial_count,
        final_count=final_count,
        initial_utf8_bytes=initial_utf8_bytes,
        final_utf8_bytes=final_utf8_bytes,
        truncated=needs_truncation,
        override_requested=override_requested,
        persisted=persisted,
        persisted_path=persisted_path,
    )
    if needs_truncation or override_requested:
        logger.info(
            "tool_result_budget tool=%s call=%s method=%s initial=%d limit=%d "
            "final=%d initial_bytes=%d final_bytes=%d truncated=%s "
            "override=%s persisted=%s",
            tool_name,
            tool_use_id,
            outcome.counter_method,
            outcome.initial_count,
            outcome.limit_tokens,
            outcome.final_count,
            outcome.initial_utf8_bytes,
            outcome.final_utf8_bytes,
            outcome.truncated,
            outcome.override_requested,
            outcome.persisted,
        )
    return final_content, outcome


def enforce_model_visible_tool_result_limits(
    messages: list[dict],
) -> list[dict]:
    """Return a wire-safe copy with every direct ``role=tool`` text bounded.

    Call-local metadata is consumed here to preserve a validated override, then
    removed. It remains on canonical history for auditability but must never be
    sent to strict providers as an unknown message field.
    """
    bounded_messages: list[dict] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            bounded_messages.append(message)
            continue

        prior_metadata = message.get("_tool_result_budget")
        limit_tokens = DEFAULT_RESULT_TOKEN_LIMIT
        override_requested = False
        if isinstance(prior_metadata, dict) and prior_metadata.get("override_requested") is True:
            candidate = prior_metadata.get("limit_tokens")
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and 1 <= candidate <= MAX_RESULT_TOKEN_LIMIT
            ):
                limit_tokens = candidate
                override_requested = True

        content, _outcome = finalize_model_visible_tool_result(
            message.get("content", ""),
            tool_name=message.get("tool_name") or message.get("name") or "synthetic",
            tool_use_id=message.get("tool_call_id") or "synthetic",
            limit_tokens=limit_tokens,
            override_requested=override_requested,
            trusted_persisted=(
                isinstance(prior_metadata, dict)
                and prior_metadata.get("persisted") is True
            ),
        )
        original_content = message.get("content")
        internal_fields = {"tool_name", "effect_disposition", "timestamp"}
        dirty_keys = {
            key
            for key in message
            if isinstance(key, str)
            and (key.startswith("_") or key in internal_fields)
        }
        if content == original_content and not dirty_keys:
            bounded_messages.append(message)
            continue
        bounded = {
            key: value
            for key, value in message.items()
            if not (
                isinstance(key, str)
                and (key.startswith("_") or key in internal_fields)
            )
        }
        bounded["content"] = content
        bounded_messages.append(bounded)
    return bounded_messages


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    if env is not None:
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
            except Exception as exc:
                logger.debug("Could not resolve env temp dir: %s", exc)
            else:
                if temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    return f"{temp_dir}/hermes-results"
    return STORAGE_DIR


def _safe_result_filename(tool_use_id: str) -> str:
    """Return a single safe filename for a tool result id."""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id

    if not safe_stem:
        safe_stem = "tool_result"
        changed = True

    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"

    return f"{safe_stem}.txt"


def _unique_result_filename(tool_use_id: str) -> str:
    """Return an unguessable per-write filename for a persisted result."""
    safe_name = _safe_result_filename(tool_use_id)
    safe_stem = safe_name.removesuffix(".txt")
    return f"{safe_stem}_{uuid.uuid4().hex[:12]}.txt"


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def _heredoc_marker(content: str) -> str:
    """Return a heredoc delimiter that doesn't collide with content."""
    if HEREDOC_MARKER not in content:
        return HEREDOC_MARKER
    return f"HERMES_PERSIST_{uuid.uuid4().hex[:8]}"


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(). Returns True on success.

    Pushes ``content`` through stdin rather than embedding it in the command
    string. Linux's ``MAX_ARG_STRLEN`` caps any single argv element at 128 KB
    (32 * PAGE_SIZE), so the previous heredoc-in-the-command-string approach
    silently failed with ``OSError: [Errno 7] Argument list too long`` for any
    tool result over ~128 KB — exactly the case persistence exists to handle.
    Routing through stdin removes that ceiling on local + ssh (``_stdin_mode
    == "pipe"``); remote backends with ``_stdin_mode == "heredoc"`` keep their
    existing API-body sized limit, which is orders of magnitude larger than
    the exec-arg ceiling.
    """
    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
) -> str:
    """Layer 2: persist oversized result into the sandbox, return preview + path.

    Writes via env.execute() so the file is accessible from any backend
    (local, Docker, SSH, Modal, Daytona). Falls back to inline truncation
    if write fails or no env is available.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (used for threshold lookup).
        tool_use_id: Unique ID for this tool call (used as filename).
        env: The active BaseEnvironment instance, or None.
        config: BudgetConfig controlling thresholds and preview size.
        threshold: Explicit override; takes precedence over config resolution.

    Returns:
        Original content if small, or <persisted-output> replacement.
    """
    # A read_file page must never become a fresh persisted handle that asks the
    # model to call read_file again. The final model-visible budget handles this
    # tool inline and points continuation back to the original path/offset.
    if tool_name == "read_file":
        return content

    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    storage_dir = _resolve_storage_dir(env)
    remote_path = f"{storage_dir}/{_unique_result_filename(tool_use_id)}"
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(preview, has_more, len(content), remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def _content_text_size(content: Any) -> int:
    """Return aggregate model-visible characters for turn-budget accounting."""
    return sum(len(text) for text in _text_values(content))


def _turn_budget_marker(
    *,
    limit_chars: int,
    persisted: bool,
    persisted_path: str | None,
) -> str:
    if persisted_path:
        detail = (
            f" Full output saved to: {persisted_path}. Use read_file with offset "
            "and limit to retrieve specific sections."
        )
    elif persisted:
        detail = " Use the existing persisted-output handle above to retrieve details."
    else:
        detail = " Full output could not be saved to the active sandbox."
    return (
        f"\n\n{PERSISTED_OUTPUT_TAG}\n"
        "[Tool-result preview reduced to fit the aggregate "
        f"{limit_chars:,}-character turn budget.{detail}]\n"
        f"{PERSISTED_OUTPUT_CLOSING_TAG}\n\n"
    )


def _reduce_content_for_turn_budget(
    content: Any,
    *,
    limit_chars: int,
    marker: str,
) -> Any:
    normalized = _normalize_model_visible_text(content)
    texts = _text_values(normalized)
    if not texts:
        return normalized
    bounded, _counter = _truncate_with_counter(
        "\n\n".join(texts),
        marker,
        max(1, limit_chars),
        DEFAULT_TOKEN_COUNTER,
    )
    return _replace_text_values(normalized, bounded)


def _enforce_turn_non_text_budget(tool_messages: list[dict]) -> None:
    """Bound opaque multipart bytes across a turn, keeping newest parts first."""
    remaining = MAX_TURN_NON_TEXT_RESULT_BYTES
    for msg in reversed(tool_messages):
        content = msg.get("content", "")
        size = _non_text_payload_size(content)
        if size <= remaining:
            remaining -= size
            continue
        bounded, omitted = _bound_non_text_parts(
            content,
            max_bytes=remaining,
        )
        if not omitted:
            remaining = max(0, remaining - _non_text_payload_size(bounded))
            continue
        msg["content"] = bounded
        remaining = max(0, remaining - _non_text_payload_size(bounded))
        prior_metadata = msg.get("_tool_result_budget")
        if isinstance(prior_metadata, dict):
            msg["_tool_result_budget"] = {
                **prior_metadata,
                "truncated": True,
                "non_text_omitted": True,
            }


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate text and opaque-payload turn budgets.

    Opaque multipart bytes are bounded first, keeping the newest parts. If
    aggregate text then exceeds the configured budget, persist the largest
    non-persisted results first until under budget. Already-persisted results
    are reduced without rewriting their authenticated handles.

    Mutates the list in-place and returns it.
    """
    _enforce_turn_non_text_budget(tool_messages)

    candidates: list[tuple[int, int]] = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = _content_text_size(content)
        total_size += size
        if size:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")
        prior_metadata = msg.get("_tool_result_budget")
        metadata_persisted = (
            isinstance(prior_metadata, dict)
            and prior_metadata.get("persisted") is True
        )
        already_persisted = metadata_persisted
        target_size = max(1, size - (total_size - config.turn_budget))
        budget_outcome = None

        if already_persisted:
            persisted_path = (
                prior_metadata.get("persisted_path")
                if isinstance(prior_metadata, dict)
                else None
            )
            if not isinstance(persisted_path, str):
                persisted_path = None
            marker = _turn_budget_marker(
                limit_chars=target_size,
                persisted=True,
                persisted_path=persisted_path,
            )
            replacement = _reduce_content_for_turn_budget(
                content,
                limit_chars=target_size,
                marker=marker,
            )
        else:
            full_text = "\n\n".join(_text_values(content))
            replacement_text, budget_outcome = finalize_model_visible_tool_result(
                full_text,
                tool_name=_BUDGET_TOOL_NAME,
                tool_use_id=f"{tool_use_id}_turn_budget",
                env=env,
                config=config,
                limit_tokens=min(DEFAULT_RESULT_TOKEN_LIMIT, target_size),
            )
            replacement = _replace_text_values(content, replacement_text)

        new_size = _content_text_size(replacement)
        if replacement != content and new_size < size:
            total_size -= size
            total_size += new_size
            msg["content"] = replacement
            if budget_outcome is not None:
                updated_metadata = budget_outcome.as_metadata()
            elif isinstance(prior_metadata, dict):
                updated_metadata = {**prior_metadata}
            else:
                updated_metadata = {}
            updated_metadata["final_count"] = model_visible_text_count(replacement)
            updated_metadata["final_utf8_bytes"] = sum(
                len(text.encode("utf-8")) for text in _text_values(replacement)
            )
            updated_metadata["truncated"] = True
            msg["_tool_result_budget"] = updated_metadata
            logger.info(
                "Budget enforcement: reduced tool result %s (%d -> %d chars, "
                "already_persisted=%s)",
                tool_use_id,
                size,
                new_size,
                already_persisted,
            )

    # Persistence previews are normally small enough to finish the job. For a
    # deliberately tiny aggregate budget, reduce those previews inline only
    # after all useful full-output handles have been created. This preserves
    # evidence and never writes to an existing handle a second time.
    if total_size > config.turn_budget:
        persisted_candidates = sorted(
            (
                (i, _content_text_size(msg.get("content", "")))
                for i, msg in enumerate(tool_messages)
                if _content_text_size(msg.get("content", ""))
                and isinstance(msg.get("_tool_result_budget"), dict)
                and msg["_tool_result_budget"].get("persisted") is True
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for idx, size in persisted_candidates:
            if total_size <= config.turn_budget:
                break
            msg = tool_messages[idx]
            content = msg.get("content", "")
            prior_metadata = msg.get("_tool_result_budget")
            persisted_path = (
                prior_metadata.get("persisted_path")
                if isinstance(prior_metadata, dict)
                else None
            )
            if not isinstance(persisted_path, str):
                persisted_path = None
            target_size = max(1, size - (total_size - config.turn_budget))
            marker = _turn_budget_marker(
                limit_chars=target_size,
                persisted=True,
                persisted_path=persisted_path,
            )
            replacement = _reduce_content_for_turn_budget(
                content,
                limit_chars=target_size,
                marker=marker,
            )
            new_size = _content_text_size(replacement)
            if replacement == content or new_size >= size:
                continue
            total_size -= size
            total_size += new_size
            msg["content"] = replacement
            if isinstance(prior_metadata, dict):
                updated_metadata = {**prior_metadata}
                updated_metadata["final_count"] = model_visible_text_count(replacement)
                updated_metadata["final_utf8_bytes"] = sum(
                    len(text.encode("utf-8")) for text in _text_values(replacement)
                )
                updated_metadata["truncated"] = True
                msg["_tool_result_budget"] = updated_metadata
            logger.info(
                "Budget enforcement: reduced persisted preview %s (%d -> %d chars)",
                msg.get("tool_call_id", f"budget_{idx}"),
                size,
                new_size,
            )

    return tool_messages
