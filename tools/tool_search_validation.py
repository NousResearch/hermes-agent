"""Local argument validation for ``tool_call`` against a deferred tool's schema."""

from __future__ import annotations

import copy
import itertools
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.registry import tool_error
from tools.tool_search_catalog import BRIDGE_TOOL_NAMES, _registry_entry

logger = logging.getLogger("tools.tool_search")

_SCHEMA_LITERAL_KEYS = frozenset({"const", "default", "enum", "example", "examples"})


def _schema_for_local_validation(node: Any) -> Any:
    """JSON-Schema-compatible copy honoring OpenAPI ``nullable: true`` (the normal coercion
    path accepts that shape, so local validation must too)."""
    if isinstance(node, list):
        return [_schema_for_local_validation(item) for item in node]
    if not isinstance(node, dict):
        return node
    # Literal keywords hold instance data, not schemas: copy byte-for-byte.
    normalized = {key: (copy.deepcopy(value) if key in _SCHEMA_LITERAL_KEYS
                        else _schema_for_local_validation(value))
                  for key, value in node.items() if key != "nullable"}
    if node.get("nullable") is not True:
        return normalized
    schema_type = normalized.get("type")
    if isinstance(schema_type, str):
        schema_type = [schema_type]
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            normalized["type"] = [*schema_type, "null"]
        return normalized
    # No ``type`` to extend ($ref/combinator): wrap so local refs still resolve from the
    # root while null stays an explicit alternative.
    return {"anyOf": [normalized, {"type": "null"}]}


def _schema_has_external_ref(node: Any) -> bool:
    """True when *node* contains a non-local ``$ref`` — local validation must never turn a
    tool call into an implicit network/file fetch (fail open)."""
    if isinstance(node, list):
        return any(_schema_has_external_ref(item) for item in node)
    if not isinstance(node, dict):
        return False
    ref = node.get("$ref")
    return (isinstance(ref, str) and not ref.startswith("#")) or any(
        _schema_has_external_ref(value) for key, value in node.items()
        if key not in _SCHEMA_LITERAL_KEYS)


def _validation_path(error: Any) -> str:
    """Format a jsonschema error path as a compact argument path."""
    path = "arguments"
    for part in getattr(error, "absolute_path", ()):
        if isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            path += f".{part}"
        else:
            path += f"[{part if isinstance(part, int) else json.dumps(part, ensure_ascii=False)}]"
    return path


def _validation_error(message: str, *, path: str, constraint: str, parameters: Any) -> str:
    return tool_error(
        message, path=path, constraint=constraint, parameters=parameters,
        hint="Retry tool_call with 'arguments' matching the parameters schema above.")


def validate_deferred_call_args(name: str, args: dict[str, Any]) -> Optional[str]:
    """Validate ``tool_call`` arguments against the deferred tool's schema. Models invoke
    deferred tools "blind" (schema unseen) and omit required args; without this, the opaque
    downstream failure makes cheap models loop. Required-field probe first, then the same
    schema-guided coercion normal dispatch applies, then jsonschema on the repaired copy.
    Missing/malformed schemas, no validator, and external refs all fail OPEN. Returns a JSON
    error string when invalid, ``None`` when the call should dispatch.

    This restores the concrete-schema checks that the provider cannot perform through the generic
    ``arguments: object`` bridge. See #5149.
    """
    try:
        from tools.registry import registry as _registry
        schema = _registry.get_schema(name)
        if not isinstance(schema, dict):
            return None
        fn = schema.get("function") if schema.get("type") == "function" else schema
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            return None
        required = params.get("required")
        missing = ([r for r in required if isinstance(r, str) and r not in args]
                   if isinstance(required, list) else [])
        if missing:
            return _validation_error(
                f"tool_call to '{name}' is missing required argument(s): "
                f"{', '.join(missing)}. The tool was NOT invoked.",
                path="arguments", constraint="required", parameters=params)
        validation_schema = _schema_for_local_validation(params)
        if _schema_has_external_ref(validation_schema):
            logger.debug("Skipping local deferred-argument validation for %s: external $ref", name)
            return None
        # Validate the repaired shape dispatch will see; copy because coerce_tool_args may
        # normalize in place (dispatch re-coerces canonically).
        try:
            from model_tools import coerce_tool_args
            candidate_args = coerce_tool_args(name, dict(args))
        except Exception:
            logger.debug("Deferred-argument coercion failed for %s", name, exc_info=True)
            candidate_args = dict(args)
        try:
            from jsonschema.exceptions import best_match
            from jsonschema.validators import validator_for
        except ImportError:
            logger.debug("jsonschema unavailable; keeping required-only validation for %s", name)
            return None
        validator_cls = validator_for(validation_schema)
        validator_cls.check_schema(validation_schema)
        validation_error = best_match(validator_cls(validation_schema).iter_errors(candidate_args))
        if validation_error is None:
            return None
        path = _validation_path(validation_error)
        constraint = str(getattr(validation_error, "validator", None) or "schema")
        detail = re.sub(r"\s+", " ", str(validation_error.message)).strip()
        if len(detail) > 600:
            detail = detail[:597] + "..."
        return _validation_error(
            f"tool_call to '{name}' failed argument validation at {path} "
            f"({constraint}): {detail}. The tool was NOT invoked.",
            path=path, constraint=constraint, parameters=params)
    except Exception:  # pragma: no cover — never block dispatch on validator bugs
        logger.debug("validate_deferred_call_args failed for %s", name, exc_info=True)
        return None


def _repair_swapped_close_brackets(text: str) -> Optional[str]:
    """Repair adjacent close brackets the model emitted out of order.

    A strict single-pass scan (string literals respected) that stops at the first
    closer not matching its stack counterpart. When the very next character is the
    closer that WOULD match the stack, swapping the two is the minimal repair and the
    scan resumes. Returns the repaired string, or None when the failure is not this
    exact pattern (no silent best-effort beyond one well-defined repair class).

    Measured cause: a long Qwen3.8-27B turn emitted the tool_call batch as a
    JSON string whose tail read `"]}]}"` instead of `"]}"` — the array's `]`
    landed one position too early, ahead of the arguments object's `}`
    (5/5 failed attempts identical in shape). Bracket counts stay balanced, so
    a naive parse fails right at the misplaced `]`.
    """
    pairs = {"}": "{", "]": "["}
    opens = {"{", "["}
    stack: list[str] = []
    out = list(text)
    in_str = False
    esc = False
    repairs = 0
    i = 0
    n = len(text)
    while i < n:
        ch = out[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch in opens:
            stack.append(ch)
        elif ch in pairs:
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
            elif (
                i + 1 < n
                and stack
                and pairs.get(out[i + 1]) == stack[-1]
            ):
                out[i], out[i + 1] = out[i + 1], ch
                repairs += 1
                stack.pop()  # the swapped-in closer at out[i] now matches the stack
            else:
                return None
        i += 1
    if in_str or esc or stack:
        return None
    return "".join(out) if repairs else None


def _is_valid_calls_batch(obj: Any) -> bool:
    """Semantic guard for the malformed-tail repairs: a recovered string must parse
    to a real calls batch (a non-empty list of dicts each carrying a non-empty
    ``name``). This is what stops a repair from "fixing" a genuinely corrupted or
    structurally wrong payload into a plausible-but-wrong one."""
    if not isinstance(obj, list) or not obj:
        return False
    return all(isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"].strip() for e in obj)


def _entry_args_dict(entry: Any) -> Optional[dict[str, Any]]:
    """The entry's ``arguments`` as a dict (a JSON string is parsed, ``None``/blank
    means ``{}``), or ``None`` when the value is not an object (after parsing)."""
    if not isinstance(entry, dict):
        return None
    raw = entry.get("arguments")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw if isinstance(raw, dict) else None


def _schema_params_of(name: str) -> tuple[Optional[dict], Optional[list]]:
    """``(properties, required)`` of a registered tool's parameter schema;
    ``(None, None)`` when the tool or its schema is unavailable (callers fail open)."""
    try:
        from tools.registry import registry
        schema = registry.get_schema(name)
    except Exception:
        return None, None
    if not isinstance(schema, dict):
        return None, None
    fn = schema.get("function") if schema.get("type") == "function" else schema
    params = fn.get("parameters") if isinstance(fn, dict) else None
    if not isinstance(params, dict):
        return None, None
    props = params.get("properties") if isinstance(params.get("properties"), dict) else None
    required = params.get("required") if isinstance(params.get("required"), list) else None
    return props, required


def _entry_consistent_with(name: str, args: dict[str, Any]) -> bool:
    """True when *args* fits *name*'s schema: every key is a declared property AND
    every declared required parameter is present. Fails OPEN (True) when the schema
    is unavailable — the parse gate then carries the whole no-guess burden."""
    props, required = _schema_params_of(name)
    if props is None:
        return True
    if set(args) - set(props):
        return False
    if required is not None and not set(required) <= set(args):
        return False
    return True


def _resolve_entry_name(args: dict[str, Any]) -> tuple[Optional[str], list[str]]:
    """Name resolution for a nameless entry: the UNIQUE registered non-bridge tool
    whose parameter schema fits the entry's keys — every key a declared property,
    every declared required parameter present. Uniqueness across ALL registered
    tools (not just deferrable ones) is what keeps this no-guess: a task-card
    session that defers core tools (read_file/terminal/session_search/... in the
    effective defer list) routes them through tool_call, so a nameless
    ``{"command": ...}`` entry must be recoverable exactly like a nameless MCP
    entry (measured 10-25, rowids 612161/613854/613927). Tools with no required
    parameters can match too — the measured {query} case is genuinely ambiguous
    between session_search (required=[]) and x_search (required=[query]) and must
    be rejected with BOTH named, not silently assigned to one. An empty key set
    never matches (the original "empty fingerprint fits too many" invariant).
    A resolved name the session cannot reach is still refused downstream by
    ``resolve_underlying_call``'s effective-defer gate, so registry generosity
    costs nothing.
    """
    if not args:
        return None, []
    try:
        from tools.registry import registry
        known = registry.get_all_tool_names()
    except Exception:
        return None, []
    keys = frozenset(args)
    candidates: list[str] = []
    for cand in known:
        if cand in BRIDGE_TOOL_NAMES:
            continue
        props, required = _schema_params_of(cand)
        if not props:
            continue
        if set(keys) - set(props):
            continue
        req = set(required) if isinstance(required, list) else set()
        if req - set(keys):
            continue
        candidates.append(cand)
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def _resolve_entry_name_with_move(args: dict[str, Any]) -> tuple[Optional[str], list[str], bool]:
    """_resolve_entry_name, plus a name LIFT: when the arguments carry a ``name`` key
    whose value is a registered tool and args-minus-that-key fit its schema, the value
    is the tool the entry meant — lift it to entry level. Measured 10-25 (rowids
    613854/613927/585819): Qwen3.8-27B wrote the name INSIDE arguments and dropped the
    arguments closer (``{"arguments": {"command": ..., "name": "terminal"}}``), so the
    entry parsed nameless with the tool name trapped one level too deep. A direct
    fingerprint and a lifted value naming DIFFERENT tools is two readings -> None."""
    name, candidates = _resolve_entry_name(args)
    lifted: Optional[str] = None
    value = args.get("name")
    if isinstance(value, str) and value.strip():
        value = value.strip()
        if value not in BRIDGE_TOOL_NAMES:
            try:
                from tools.registry import registry
                registered = value in registry.get_all_tool_names()
            except Exception:
                registered = False
            if registered:
                reduced = {k: v for k, v in args.items() if k != "name"}
                if _entry_consistent_with(value, reduced):
                    lifted = value
    if name is not None:
        if lifted is not None and lifted != name:
            return None, candidates + [lifted], False
        return name, candidates, False
    if lifted is not None:
        if candidates:
            # An ambiguous direct fingerprint AND a lift are two (or more)
            # readings — the lift must not win silently (measured 10-25
            # adversarial review: {name} key sets fingerprint several
            # no-required tools, so {"name": X} would otherwise always lift).
            return None, candidates + [lifted], False
        return lifted, [lifted], True
    return None, candidates, False


def _finalize_entry(entry: Any, assigned_name: Optional[str] = None) -> Optional[dict[str, Any]]:
    """A raw parsed entry into ``{"name", "arguments"}`` — or None when the name
    cannot be established without a guess. A nameless entry resolves via
    :func:`_resolve_entry_name_with_move` (unique match only, including the lift).
    ``assigned_name`` is a name the caller already arbitrated for a nameless entry
    (the dangle repair); stray entry-level keys (legal only as name/arguments) are
    then folded into arguments — but only when they fit that tool's schema."""
    if not isinstance(entry, dict):
        return None
    args = _entry_args_dict(entry)
    if args is None:
        return None
    strays = {k: v for k, v in entry.items() if k not in ("name", "arguments")}
    name = str(entry.get("name") or "").strip()
    if not name and assigned_name is not None:
        name = assigned_name
    if name:
        if name in BRIDGE_TOOL_NAMES:
            return None
        if strays:
            merged = dict(args)
            merged.update(strays)
            if not _entry_consistent_with(name, merged):
                return None  # strays don't belong to this tool -> no silent fold
            args = merged
        return {"name": name, "arguments": args}
    resolved, _cands, moved = _resolve_entry_name_with_move(args)
    if resolved is None:
        return None
    if moved:
        args = {k: v for k, v in args.items() if k != "name"}
    if strays:
        merged = dict(args)
        merged.update(strays)
        if not _entry_consistent_with(resolved, merged):
            return None
        args = merged
    return {"name": resolved, "arguments": args}


def _canonical_batch(entries: Any, assigned: Optional[tuple[int, str]] = None) -> Optional[str]:
    """Canonical JSON of a fully-resolved batch, or None when ANY entry is
    unresolvable — the no-guess gate every tail repair runs through. ``assigned``
    is ``(index, name)``: a dangle name the caller is testing on one nameless
    entry. Two repairs that parse to different names/args yield different
    canonical forms and both survive -> the caller sees >1 and rejects."""
    if not isinstance(entries, list) or not entries:
        return None
    finalized: list[dict[str, Any]] = []
    for i, e in enumerate(entries):
        a = assigned[1] if assigned is not None and assigned[0] == i else None
        f = _finalize_entry(e, a)
        if f is None:
            return None
        finalized.append(f)
    return json.dumps(finalized, sort_keys=True, ensure_ascii=False)


def _collect_tail_closer_run(text: str, max_k: int = 5, max_suffix: int = 2,
                             want_list: bool = True) -> dict[str, str]:
    """EVERY ``(canonical result, repaired text)`` pair reachable by dropping k
    trailing close brackets (``k <= max_k``; the run must be pure ``}]``) and
    appending at most ``max_suffix`` closers. Supersedes the old valid-prefix +
    stray-tail cut (``k=0, suffix=''``) and the bounded tail appender (``k=0``)
    with a name-aware gate: each candidate must re-parse to a list of dict
    entries (``want_list``) or a dict, and every entry must resolve to a name
    without a guess (:func:`_canonical_batch`) — so a corrupted tail cannot be
    "completed" into a plausible-but-wrong batch. Measured 10-25 family: the
    entry's closing ``}`` drifted to ``]`` (or vanished) and 1-3 surplus closers
    piled up at the batch tail, with the entry(s) nameless.
    """
    found: dict[str, str] = {}
    n = len(text)
    for k in range(max_k + 1):
        if k and any(ch not in "]}" for ch in text[n - k:]):
            break  # the trailing run is shorter than k — larger k cannot work
        head = text[:n - k]
        for depth in range(max_suffix + 1):
            for suffix in itertools.product("]}", repeat=depth):
                cand = head + "".join(suffix)
                try:
                    obj = json.loads(cand)
                except json.JSONDecodeError:
                    continue
                if want_list:
                    canon = _canonical_batch(obj)
                else:
                    if not isinstance(obj, dict):
                        continue
                    canon = json.dumps(obj, sort_keys=True, ensure_ascii=False)
                if canon:
                    found.setdefault(canon, cand)
    return found


def _repair_single_char_edit(text: str) -> Optional[str]:
    """Recover a stringified ``tool_call`` batch that needs exactly one character
    fixed: a missing or misplaced bracket/quote-escape, or one missing comma.

    Measured shapes (Qwen3.8-27B, tail region of the stringified batch):

    - the entry object closed one brace too early, leaving ``"name"`` dangling at
      array level: ``...}]}}, "name": "..."`` (10/94 rejected payloads);
    - the array closed before the entry object: ``...}]}`` at the very tail
      (missing one final ``]`` or ``}``);
    - a dropped comma between batch entries: ``...600}, {"arguments": ...``;
    - escape-level drift where ``arguments`` is itself a JSON string: the closing
      quote got over-escaped (``...\\\\\"}]}]}`` — an extra backslash makes the
      string never terminate).

    Unlike the single-shape strategies above, an one-character edit can have several
    legal-looking placements, so acceptance is stricter: every edit candidate that
    parses AND passes :func:`_is_valid_calls_batch` is kept, and the repair is used
    **only when exactly one semantic batch survives**. Two different valid readings
    means the slip is ambiguous (deleting ``]`` vs ``}`` at the tail yields
    structurally different batches) and must be left to the model's retry, never
    guessed. Two-or-more-edit damage (double slips, truncation) is out of scope on
    purpose — the combinatorics make unique-solution checking unsound.
    """
    results: dict[str, str] = {}  # canonical batch JSON -> repaired text

    def accept(candidate: str) -> None:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return
        if _is_valid_calls_batch(obj):
            results.setdefault(json.dumps(obj, sort_keys=True, ensure_ascii=False), candidate)

    # Editable positions: brackets outside string literals + backslashes inside
    # them (the escape-drift shape). A single pass that also tracks in-string state.
    positions: list[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
                positions.append(i)
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{}[]":
            positions.append(i)

    for i in positions:
        accept(text[:i] + text[i + 1:])  # delete the one char at i
    for i in positions + [len(text)]:
        accept(text[:i] + "]" + text[i:])  # insert ]
        accept(text[:i] + "}" + text[i:])  # insert }
        prev = text[i - 1] if i > 0 else ""
        # A comma only makes sense where a sibling element would follow — after a
        # closer, never glued to an identifier/number.
        if prev not in ("", "{", "[") and not prev.isalnum():
            accept(text[:i] + "," + text[i:])  # insert ,

    if len(results) == 1:
        return next(iter(results.values()))
    return None


def _repair_missing_tail_closers(text: str) -> Optional[str]:
    """Recover a batch whose tail is missing one or two final close brackets.

    Measured shape (Qwen3.8-27B, session musg34l4x2opyk, 8 identical failures
    over multiple days): the stringified batch stops right after closing the
    inner ``arguments`` object — the enclosing entry object and the batch array
    are never closed, so the tail reads ``...}}]}`` instead of ``...}}]}}]``.

    Appending is bounded to two closers (8 suffixes, shortest first). A repair is
    accepted only when it is the UNIQUE valid batch at the shortest working
    length; a payload truncated mid-content cannot parse after any tail append
    (dangling string/comma), so this cannot complete garbage.
    """
    for depth in (1, 2):
        found: dict[str, str] = {}
        for suffix in itertools.product("]}", repeat=depth):
            cand = text + "".join(suffix)
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if _is_valid_calls_batch(obj):
                found.setdefault(json.dumps(obj, sort_keys=True, ensure_ascii=False), cand)
        if len(found) == 1:
            return next(iter(found.values()))
        if found:  # several valid completions at this length -> ambiguous
            return None
    return None


def _resolve_entry_name_from_args(args: dict[str, Any]) -> tuple[Optional[str], list[str]]:
    """Infer the ``name`` a batch entry omitted, from its argument keys.

    Sound only on a UNIQUE fingerprint: exactly one registered deferrable tool
    whose parameter schema fits the entry's keys — every declared required
    parameter present AND every key a declared property. A tool with no
    required parameters is never a match (an empty fingerprint fits too many).
    Returns ``(name, candidates)``; ``name`` is None when the match is not
    unique. Never raises — resolution failure means "no recovery".
    """
    try:
        from tools.registry import registry
        from tools.tool_search import is_deferrable_tool_name
        known = registry.get_all_tool_names()
    except Exception:
        # Deliberate catch-all: recovery is best-effort and must never break
        # the normalization it is embedded in; a broken registry is an anomaly
        # worth a traceback, not a raised error.
        logger.exception("entry-name recovery: registry unavailable; skipping recovery")
        return None, []
    keys = frozenset(args)
    candidates: list[str] = []
    for cand in known:
        if cand in BRIDGE_TOOL_NAMES:
            continue
        try:
            if not is_deferrable_tool_name(cand):
                continue
            schema = registry.get_schema(cand)
            fn = schema.get("function") if isinstance(schema, dict) and schema.get("type") == "function" else schema
            params = fn.get("parameters") if isinstance(fn, dict) else None
            props = params.get("properties") if isinstance(params, dict) else None
            required = params.get("required") if isinstance(params, dict) else None
        except Exception:
            logger.warning("entry-name recovery: skipping tool %s (bad schema)", cand, exc_info=True)
            continue
        if not isinstance(props, dict) or not props:
            continue
        if not (isinstance(required, list) and required):
            continue
        if set(required) <= keys <= set(props):
            candidates.append(cand)
    return (candidates[0] if len(candidates) == 1 else None), candidates


def _recover_missing_name_entry(text: str) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """The 10-24 measured shape (session mvm57ofwgwwvvu, 8/8 attempts): the
    stringified batch's single entry dropped BOTH its ``"name"`` field and its
    closing ``}``, so the tail reads ``...<arguments>...]}]`` — the final
    ``]`` lands while the entry object is still open. No bracket repair can
    restore a missing key, so the name is resolved from the argument
    fingerprint (:func:`_resolve_entry_name_from_args`, unique match only).

    Bounded: insert one ``}`` at each trailing closer position (string-aware,
    last eight closers plus end); accept only parses to a single entry lacking
    ``"name"`` and carrying a non-empty ``"arguments"`` dict, and only when
    every such parse canonicalizes to the same batch (else ambiguous).
    Returns ``(entries, None)`` on rescue; ``(None, error)`` when the shape is
    recognized but the name is not uniquely resolvable (the error teaches the
    retry the exact fix); ``(None, None)`` when the shape does not apply.
    """
    in_str = False
    esc = False
    closers: list[int] = []
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "}]":
            closers.append(i)
    positions = sorted(set(closers[-8:] + [len(text)]))
    found: dict[str, dict] = {}
    for i in positions:
        cand = text[:i] + "}" + text[i:]
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not (isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict)):
            continue
        entry = obj[0]
        if isinstance(entry.get("name"), str) and entry["name"].strip():
            continue
        raw_args = entry.get("arguments")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue
        if not isinstance(raw_args, dict) or not raw_args:
            continue
        canon = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        found.setdefault(canon, raw_args)
    if not found:
        return None, None
    if len(found) > 1:
        return None, None  # ambiguous — leave to the model's retry
    raw_args = next(iter(found.values()))
    name, candidates = _resolve_entry_name_from_args(raw_args)
    if name is not None:
        return [{"name": name, "arguments": raw_args}], None
    detail = (" Closest matches: " + ", ".join(candidates) + " — pick the one you mean.") if candidates else ""
    return None, (
        "tool_call entry is missing its 'name' field: each entry must be "
        "{\"name\": \"<exact tool name>\", \"arguments\": {...}} — pass the name "
        "tool_search returned. The call was NOT made; retry with 'name'." + detail)


_DANGLE_TEACH = (
    "A 'name' field landed OUTSIDE its entry — after the entry (or the whole batch) "
    "had already closed. Every call must be one object carrying BOTH fields inside "
    "it: {\"name\": \"<tool>\", \"arguments\": {...}}. The call was NOT made; retry "
    "with each name inside its own entry object.")


def _find_dangling_name(text: str) -> Optional[tuple[str, int]]:
    """Locate the measured dangle: a ``"name"`` KEY at bracket depth 0 or 1 (root or
    array level — never inside an entry object) whose string value is followed by
    NOTHING but close brackets. Returns ``(value, key_position)`` or None. Valid
    JSON cannot hold a key-value pair at array/root level, so this only matches
    corrupted text; the closers-only tail anchors the match to the actual end.
    """
    in_str = False
    esc = False
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True  # the scanner re-enters string state on the key itself
            if depth <= 1 and text.startswith('"name"', i):
                j = i + len('"name"')
                while j < n and text[j] == " ":
                    j += 1
                if j < n and text[j] == ":":
                    j += 1
                    while j < n and text[j] == " ":
                        j += 1
                    if j < n and text[j] == '"':
                        k = j + 1
                        esc2 = False
                        closed = False
                        while k < n:
                            if esc2:
                                esc2 = False
                            elif text[k] == "\\":
                                esc2 = True
                            elif text[k] == '"':
                                closed = True
                                break
                            k += 1
                        if closed and all(c in "]}" for c in text[k + 1:]):
                            return text[j + 1:k], i
            i += 1
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        i += 1
    return None


def _repair_dangling_name(text: str) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """The 10-25 measured shape (rowids 612161/612965/582641): an entry's
    ``"name"`` leaked out of its entry object — emitted at array level
    (``...}, "name": "X"}]``, the entry having closed one brace too early) or
    after the batch's own close (``...]}, "name": "X"}]``). Recovery is a semantic
    reconstruction, not a text edit:

    1. the dangle value X is located (:func:`_find_dangling_name`);
    2. the prefix before X's key is repaired with the bounded k-run — its last
       entry usually dropped (or drifted) a closer at the same time;
    3. X is attached to the ONE nameless entry whose arguments (+ stray
       entry-level keys, which legally belong nowhere) fit X's schema; every
       other nameless entry resolves by unique fingerprint / name lift;
    4. only when exactly one final batch survives ALL of that is it returned.

    Returns ``(entries, None)`` on rescue; ``(None, teach)`` when the shape is
    recognized but the assignment is not unique (or fits nowhere); ``(None, None)``
    when no dangle applies.
    """
    hit = _find_dangling_name(text)
    if hit is None:
        return None, None
    x_name, key_pos = hit
    # A dangle naming a tool that cannot be dispatched (typo, unregistered,
    # or a bridge name) is not a name-position slip — refuse to "recover"
    # it: other strategies or the plain rejection handle the text, and an
    # unregistered name must never ride into a sibling-valid batch.
    try:
        from tools.registry import registry
        if x_name in BRIDGE_TOOL_NAMES or x_name not in registry.get_all_tool_names():
            return None, None
    except Exception:
        pass
    prefix = text[:key_pos]
    comma = re.search(r",\s*$", prefix)
    if comma:
        prefix = prefix[:comma.start()]
    cands = _collect_tail_closer_run(prefix)
    if not cands:
        return None, _DANGLE_TEACH
    finals: dict[str, list[dict[str, Any]]] = {}
    for _canon, fixed in cands.items():
        obj = json.loads(fixed)
        nameless_idx = [i for i, e in enumerate(obj)
                        if isinstance(e, dict) and not str(e.get("name") or "").strip()]
        if not nameless_idx:
            continue  # X fits nowhere
        targets: list[int] = []
        for i in nameless_idx:
            e = obj[i]
            args = _entry_args_dict(e)
            if args is None:
                continue
            strays = {k: v for k, v in e.items() if k not in ("name", "arguments")}
            merged = dict(args)
            merged.update(strays)
            if _entry_consistent_with(x_name, merged):
                targets.append(i)
        if len(targets) != 1:
            continue  # X fits several (or zero) entries -> this reading is a guess
        canon = _canonical_batch(obj, (targets[0], x_name))
        if canon:
            finals.setdefault(canon, json.loads(canon))
    if len(finals) == 1:
        return next(iter(finals.values())), None
    return None, _DANGLE_TEACH


def _deescape_structural(seg: str) -> Optional[str]:
    """Inverse of the model's over-escape on a STRUCTURAL region (keys, values,
    delimiters): ``\\"`` -> ``"`` and ``\\\\`` -> ``\\``. Any other escape means the
    region boundary was mis-guessed -> None, which rejects that candidate."""
    out: list[str] = []
    i = 0
    n = len(seg)
    while i < n:
        ch = seg[i]
        if ch == "\\":
            if i + 1 >= n:
                return None
            nxt = seg[i + 1]
            if nxt == "\\":
                out.append("\\")
            elif nxt == '"':
                out.append('"')
            else:
                return None
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_escape_drift(text: str) -> Optional[str]:
    """The 10-25 measured shape (rowid 613401): a string value that itself quotes
    nested quotes was opened with an ESCAPED quote (``\"`` where the raw ``"``
    belongs) — and the model then kept escaping every quote until the structure
    resumed raw, so the tail reads ``\", \"title\": \"...\"}], \"context_id\": ...``.
    The parse dies "Expecting value" at the first out-of-string backslash.

    The drift region runs from that backslash to the next RAW quote. Inside it,
    every escaped quote is a candidate boundary for the value's over-escaped close:
    an even backslash run ends the value with a raw close (the run's pairs are
    literal backslashes in the value); an odd run is the over-escaped close itself
    (all but the final backslash are value content). Everything between the close
    and the region end is structural and must de-escape cleanly.

    Measured 10-25 (rowid 613401): several boundaries can re-parse — a LATER one
    lets the value swallow a span that is itself complete key-value structure
    (the model's over-escaped ``", "title": "..."`` reads as value text). The
    intended close is the EARLIEST boundary whose candidate re-parses to a fully
    resolvable batch: inner boundaries fail to parse (the swallowed span leaves
    raw unquoted structure), and any later survivor converts more bytes from
    structure into value text — a strictly larger deviation. Boundaries are
    walked in ascending order, so the first success is the answer; no success
    means the drift is not this shape (or is ambiguous beyond recovery).
    """
    in_str = False
    esc = False
    s: Optional[int] = None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == "\\":
            s = i
            break
        elif ch == '"':
            in_str = True
    if s is None or s + 1 >= len(text) or text[s + 1] != '"':
        return None
    e: Optional[int] = None
    for j in range(s, len(text)):
        if text[j] == '"' and (j == 0 or text[j - 1] != "\\"):
            e = j
            break
    if e is None or e <= s + 1:
        return None
    q = s + 1
    while q < e:
        if text[q] == '"':
            run = 0
            t = q - 1
            while t >= s and text[t] == "\\":
                run += 1
                t -= 1
            if run:
                value = text[s + 2:q] if run % 2 == 0 else text[s + 2:q - 1]
                structural = _deescape_structural(text[q + 1:e])
                if structural is not None:
                    cand = text[:s] + '"' + value + '"' + structural + text[e:]
                    try:
                        obj = json.loads(cand)
                    except json.JSONDecodeError:
                        obj = None
                    if obj is not None and _canonical_batch(obj):
                        return cand  # earliest winning boundary
        q += 1
    return None


def _repair_malformed_calls_string(text: str) -> Optional[str]:
    """Recover a stringified ``tool_call`` batch from the small-model tail-bracket
    malformations measured on long Qwen3.8-27B turns. Five bounded strategies,
    tried in code order:

    1. adjacent close brackets emitted out of order (counts balanced) — handled by
       :func:`_repair_swapped_close_brackets` (09-29, 5/5 identical shape);
    2. a valid batch prefix followed by a trailing run of stray close brackets —
       cut at the prefix end;
    3. exactly one missing/misplaced character or a dropped comma (early entry
       close, escape drift, sibling comma loss) — :func:`_repair_single_char_edit`,
       applied only when the one-edit repair yields a UNIQUE valid batch;
    4. the batch tail missing its final one/two close brackets —
       :func:`_repair_missing_tail_closers`, unique completion required;
    5. exactly one surplus close bracket (counts off by +1) — delete it.

    Each candidate is accepted **only** if it re-parses to a valid calls batch
    (:func:`_is_valid_calls_batch`), so real corruption is never silently masked.
    Returns the repaired string, or ``None`` when no bounded repair applies.
    """
    # 1. adjacent close brackets out of order (balanced counts).
    repaired = _repair_swapped_close_brackets(text)
    if repaired is not None:
        try:
            if _is_valid_calls_batch(json.loads(repaired)):
                return repaired
        except json.JSONDecodeError:
            pass
    # 2. valid batch prefix + trailing stray close brackets -> cut.
    try:
        obj, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        obj, end = None, None
    if obj is not None:
        trail = text[end:]
        if trail and all(ch in "]}" for ch in trail) and _is_valid_calls_batch(obj):
            return text[:end]
    # 4. a single missing/misplaced character (one edit only; the repaired batch
    #    must be the UNIQUE semantic result, else reject).
    repaired = _repair_single_char_edit(text)
    if repaired is not None:
        return repaired
    # 5. the batch tail never got its final one/two close brackets — bounded
    #    append, unique-completion required.
    repaired = _repair_missing_tail_closers(text)
    if repaired is not None:
        return repaired
    # 3. exactly one surplus close bracket (forward stack scan, delete the mismatch).
    pairs = {"}": "{", "]": "["}
    opens = {"{", "["}
    stack: list[int] = []
    in_str = False
    esc = False
    mismatch: Optional[int] = None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in opens:
            stack.append(i)
        elif ch in pairs:
            if stack and text[stack[-1]] == pairs[ch]:
                stack.pop()
            else:
                mismatch = i
                break
    if mismatch is not None and not in_str:
        candidate = text[:mismatch] + text[mismatch + 1:]
        try:
            obj2 = json.loads(candidate)
        except json.JSONDecodeError:
            obj2 = None
        if _is_valid_calls_batch(obj2):
            return candidate
    return None


def normalize_tool_call_entries(args: dict[str, Any]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Normalize ``tool_call`` arguments into a ``calls[]`` list of entries.

    Accepts the advertised batch shape ``{"calls": [{"name", "arguments"}, ...]}``
    and, tolerantly, the legacy single shape ``{"name": ..., "arguments": ...}``
    (a single call is a batch of one). Each entry's ``arguments`` is coerced to
    a dict (JSON strings parsed, ``None`` → ``{}``). Returns ``(entries, None)``
    or ``([], error_message)``.
    """
    raw_calls = args.get("calls")
    if raw_calls is None:
        # Legacy single shape.
        if not str(args.get("name") or "").strip():
            return [], "tool_call requires 'calls' (an array of {name, arguments})"
        raw_calls = [{"name": args.get("name"), "arguments": args.get("arguments")}]
    if isinstance(raw_calls, str):
        # Tolerate the model emitting the batch envelope as a JSON string —
        # mirror the per-entry `arguments` handling below (#114484).
        try:
            raw_calls = json.loads(raw_calls)
        except json.JSONDecodeError as e:
            # Small-model structural slips in the stringified batch envelope
            # (measured on long Qwen3.8-27B turns). Five bounded no-guess repairs,
            # in order — each accepted ONLY when exactly one valid reading
            # survives (re-parse + every entry resolvable to a name):
            #   1. adjacent close brackets out of order, a trailing run of stray
            #      close brackets, one surplus close bracket, a single-char edit,
            #      missing tail closers — _repair_malformed_calls_string;
            #   2. a "name" dangle outside its entry (array or root level) —
            #      _repair_dangling_name (semantic re-attachment);
            #   3. the tail closer run mangled with nameless entries —
            #      _collect_tail_closer_run (drop k trailing closers, append <=2);
            #   4. escape drift in a value with nested quotes — _repair_escape_drift;
            #   5. one entry that dropped BOTH its name and its closing brace —
            #      _recover_missing_name_entry (carries the targeted teaching).
            recovered_obj: Any = None
            repaired = _repair_malformed_calls_string(raw_calls)
            if repaired is not None:
                try:
                    recovered_obj = json.loads(repaired)
                    logger.info("tool_call 'calls' recovered a malformed-tail slip")
                except json.JSONDecodeError:
                    recovered_obj = None
            if recovered_obj is None:
                recovered, targeted = _repair_dangling_name(raw_calls)
                if recovered is not None:
                    logger.info("tool_call 'calls' recovered a dangling-name slip")
                    return recovered, None
                if targeted is not None:
                    return [], targeted
            if recovered_obj is None:
                found = _collect_tail_closer_run(raw_calls)
                if len(found) == 1:
                    repaired = next(iter(found.values()))
                    recovered_obj = json.loads(repaired)
                    logger.info("tool_call 'calls' recovered a tail-closer-run slip")
            if recovered_obj is None:
                repaired = _repair_escape_drift(raw_calls)
                if repaired is not None:
                    recovered_obj = json.loads(repaired)
                    logger.info("tool_call 'calls' recovered an escape-drift slip")
            if recovered_obj is not None:
                raw_calls = recovered_obj
            else:
                recovered, targeted = _recover_missing_name_entry(raw_calls)
                if recovered is not None:
                    logger.info("tool_call 'calls' recovered a missing-entry-name slip")
                    return recovered, None
                if targeted is not None:
                    return [], targeted
                return [], f"tool_call 'calls' is not valid JSON: {e}"
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    if not isinstance(raw_calls, list) or not raw_calls:
        return [], "tool_call 'calls' must be a non-empty array of {name, arguments}"

    entries: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            return [], f"tool_call calls[{position}] must be an object with 'name' and 'arguments'"
        raw_args = raw.get("arguments")
        if raw_args is None or (isinstance(raw_args, str) and not raw_args.strip()):
            # "" / whitespace is how some OpenAI-compatible gateways spell "no arguments" for a
            # parameterless tool (#83937); the loop already treats an empty outer arguments string
            # as {} (turn_tool_validation), and a missing required param still surfaces below via
            # validate_deferred_call_args instead of an opaque JSON parse error.
            raw_args = {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                # Measured 10-24 backlog (mv2e5ly1bah9we, 4/5 attempts): the stringified
                # arguments object carries one surplus tail closer ("...plan...]}}").
                # Bounded k-run completion; accepted only on the UNIQUE reading.
                found = _collect_tail_closer_run(raw_args, max_k=3, max_suffix=1, want_list=False)
                if len(found) == 1:
                    raw_args = json.loads(next(iter(found.values())))
                    logger.info("tool_call calls[%d].arguments recovered a tail-closer slip", position)
                else:
                    return [], f"tool_call calls[{position}].arguments is not valid JSON: {e}"
        if not isinstance(raw_args, dict):
            return [], f"tool_call calls[{position}].arguments must be an object"
        name = str(raw.get("name") or "").strip()
        if not name:
            # Measured 10-24 shape (session mvm57ofwgwwvvu, 8/8 attempts): a small model
            # omits "name" on the single-entry batch. Recover it when the argument
            # fingerprint is UNIQUE across the registered deferrable tools — or across
            # all registered tools (stage 2: sessions that defer core tools route them
            # through this bridge too) — or by lifting a "name" key the model trapped
            # inside arguments. Otherwise teach the retry the exact shape instead of an
            # opaque JSON error.
            resolved, candidates, moved = _resolve_entry_name_with_move(raw_args)
            if resolved:
                if moved:
                    raw_args = {k: v for k, v in raw_args.items() if k != "name"}
                logger.info("tool_call calls[%d] 'name' recovered by unique schema match: %s%s",
                            position, resolved, " (lifted out of arguments)" if moved else "")
                name = resolved
            else:
                hint = (f" Closest matches: {', '.join(candidates)} — pick the one you mean.") if candidates else ""
                return [], (
                    f"tool_call calls[{position}] requires a 'name' (the exact tool name to "
                    f"invoke): each entry must be {{\"name\": \"<tool name>\", \"arguments\": "
                    f"{{...}}}} — pass the name tool_search returned.{hint} The call was NOT "
                    f"made; retry with 'name'.")
        if name in BRIDGE_TOOL_NAMES:
            return [], f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
        entries.append({"name": name, "arguments": raw_args})
    return entries, None


_ECHO_ARGS_MAX_CHARS = 1500


def local_batch_error(entries: list[dict[str, Any]]) -> str:
    """Rejection for a multi-entry batch that names a local tool. Restates the valid
    shape with the caller's OWN first entry: small models re-send an identical batch
    when told only the constraint, and the echoed payload is what gets them unstuck."""
    first = entries[0]
    args = json.dumps(first.get("arguments", {}), ensure_ascii=False, separators=(",", ":"))
    if len(args) > _ECHO_ARGS_MAX_CHARS:
        args = "{...}"  # keep the correction readable; the model still has its own arguments
    retry = '{{"calls":[{{"name":{},"arguments":{}}}]}}'.format(json.dumps(first["name"], ensure_ascii=False), args)
    remaining = (f" then issue the remaining {len(entries) - 1} call(s) as separate tool_call invocations"
                 if len(entries) > 1 else "")
    return (
        f"tool_call takes exactly one entry for local tools; you sent {len(entries)}. "
        f"Retry with only: {retry}{remaining}. Only connectors__ names may be batched together."
    )


def unavailable_in_session_reason(name: str) -> str:
    """Why a registered tool absent from THIS session cannot be reached, whatever its door.

    Deferral is a property of the tool, availability one of the session: a deferred tool the
    session does not carry (``defer: [delegate_task]`` in a worker) is no more findable through
    ``tool_search`` than a stripped core tool is callable directly."""
    return (f"'{name}' is not available in this execution context. "
            "Don't retry it — state the assumption and proceed with the work.")


def session_unavailable_error(name: str, session_tool_names: Optional[Iterable[str]]) -> Optional[str]:
    """The unavailability message when a core tool is missing from the session's own tool list,
    whether or not ``defer:`` moved it behind the bridge; None without session context, when
    the session carries it, or for MCP/plugin/unregistered names (an absent one may be a stale
    catalog that ``tool_search`` can still refresh). A missing GUI surface keeps its own reason."""
    if session_tool_names is None or name in set(session_tool_names):
        return None
    from tools.tool_search import _core_tool_names, out_of_scope_reason  # late: imports this module
    if name not in _core_tool_names() or _registry_entry(name) is None:
        return None
    return out_of_scope_reason(name) or unavailable_in_session_reason(name)


def not_deferrable_error(name: str, session_tool_names: Optional[Iterable[str]] = None) -> str:
    """Rejection for a ``tool_call`` naming something that is not a deferred tool.
    Two different mistakes reach here and need opposite corrections: a directly-listed
    tool (call it without the bridge) vs. an unknown name — typically a deferred MCP tool
    cited by its bare suffix instead of the full ``mcp__<server>__{tool}`` name. Telling
    the second group 'call it directly' is the opposite of what they must do.

    ``_HERMES_CORE_TOOLS`` is a *static* list, so a core tool stripped from this session's
    toolset (clarify in a subagent, kanban worker) otherwise gets told "call it directly"
    — an instruction that can never resolve, burning a turn on a confident wrong answer.
    When the caller passes the session-scoped names, absence is reported as unavailability
    instead. Stays None for callers with no session context; deliberately never consults the
    global blocklist, because a top-level session legitimately has these tools.
    """
    from tools.tool_search import _core_tool_names  # late: tool_search imports this module
    if name in _core_tool_names() or _registry_entry(name) is not None:
        if session_tool_names is not None and name not in set(session_tool_names):
            return unavailable_in_session_reason(name)
        return (f"'{name}' is a directly-listed tool, not a deferred one. "
                "Call it directly instead of via tool_call.")
    suffix = f"__{name}"
    try:
        from tools.registry import registry
        candidates = sorted(n for n in registry.get_all_tool_names() if n.endswith(suffix))
    except Exception:
        candidates = []
    hint = (f" Did you mean {', '.join(repr(c) for c in candidates)}?" if candidates
            else " Use tool_search to find the exact name.")
    return (f"'{name}' is not a known tool name. Deferred tools must be invoked through tool_call "
            f"by the exact name tool_search returns (e.g. mcp__<server>__{{tool}}).{hint}")
