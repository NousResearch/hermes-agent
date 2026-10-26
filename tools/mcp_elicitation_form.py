"""MCP form elicitation: ask the user for the fields a server's ``requestedSchema`` names and return
schema-valid content.

The questions go through the running agent's clarify callback, which every interactive surface
already implements (classic CLI panel, Ink TUI, Desktop card, messaging platforms' clarify prompt).
The executor binds that callback for each dispatched tool call (:func:`bound_form_asker`); the MCP
elicitation runs later on another thread inside the call's contextvars snapshot and reads it with
:func:`current_form_asker`. The answers go back to the MCP server only: nothing here writes them to
the transcript, and the model sees them only if the server echoes them in its tool result.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlparse

from tools.clarify_tool import MAX_CHOICES, MAX_QUESTIONS

FormAsker = Callable[[list], dict]
_asker: contextvars.ContextVar[Optional[FormAsker]] = contextvars.ContextVar("mcp_form_asker", default=None)

# Wrong answers are re-asked this many times before the form is declined.
MAX_REASKS = 2
_YES = ("yes", "y", "true", "t", "1", "on")
_NO = ("no", "n", "false", "f", "0", "off")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@contextlib.contextmanager
def bound_form_asker(callback: Optional[FormAsker]) -> Iterator[None]:
    """Bind the agent's clarify callback for one dispatched tool call."""
    token = _asker.set(callback)
    try:
        yield
    finally:
        _asker.reset(token)


def current_form_asker() -> Optional[FormAsker]:
    return _asker.get()


@dataclass
class FormField:
    name: str
    kind: str  # string | number | integer | boolean | enum | multi
    label: str
    description: str = ""
    required: bool = False
    default: Any = None
    # enum/multi: (value, label) pairs in schema order
    options: list[tuple[Any, str]] = field(default_factory=list)
    spec: dict = field(default_factory=dict)


def _options(spec: dict) -> list[tuple[Any, str]]:
    """``enum`` (+ legacy ``enumNames``) or titled ``oneOf``/``anyOf`` ``const`` lists."""
    values, names = spec.get("enum"), spec.get("enumNames")
    if isinstance(values, list):
        names = names if isinstance(names, list) else []
        return [(v, str(names[i]) if i < len(names) else str(v)) for i, v in enumerate(values)]
    titled = spec.get("oneOf") or spec.get("anyOf") or []
    return [(c["const"], str(c.get("title") or c["const"])) for c in titled if isinstance(c, dict) and "const" in c]


def _field_kind(spec: dict) -> tuple[str, list[tuple[Any, str]]]:
    if spec.get("type") == "array":
        items = spec.get("items")
        return "multi", _options(items if isinstance(items, dict) else {})
    options = _options(spec)
    if options:
        return "enum", options
    kind = spec.get("type")
    return (kind if kind in ("number", "integer", "boolean") else "string"), []


def parse_form(schema: Any) -> list[FormField]:
    """The flat object schema MCP elicitation allows -> fields in declaration order."""
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return []
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    fields = []
    for name, raw in props.items():
        spec = raw if isinstance(raw, dict) else {}
        kind, options = _field_kind(spec)
        fields.append(FormField(name=str(name), kind=kind, label=str(spec.get("title") or name),
                                description=str(spec.get("description") or ""), required=name in required,
                                default=spec.get("default"), options=options, spec=spec))
    return fields


def _hint(f: FormField) -> str:
    """Short type/constraint hint shown after the field label."""
    spec, parts = f.spec, []
    if f.kind in ("enum", "multi") and len(f.options) > MAX_CHOICES:
        parts.append(("one or more of" if f.kind == "multi" else "one of") + ": "
                     + ", ".join(label for _, label in f.options))
    elif f.kind == "multi":
        parts.append("pick one or more")
    if f.kind in ("number", "integer"):
        parts.append("a whole number" if f.kind == "integer" else "a number")
        if "minimum" in spec:
            parts.append(f"min {spec['minimum']}")
        if "maximum" in spec:
            parts.append(f"max {spec['maximum']}")
    if spec.get("format"):
        parts.append(str(spec["format"]))
    if f.default is not None:
        parts.append(f"default: {f.default}")
    parts.append("required" if f.required else "optional, skip to leave blank")
    return "; ".join(parts)


def _choices(f: FormField) -> Optional[list[str]]:
    if f.kind == "boolean":
        return ["Yes", "No"]
    if f.kind in ("enum", "multi") and len(f.options) <= MAX_CHOICES:
        return [label for _, label in f.options]
    return None


def _question(f: FormField, header: str, error: str) -> dict:
    text = f"{f.label}" + (f": {f.description}" if f.description else "") + f" ({_hint(f)})"
    if error:
        text += f"\n⚠ {error}"
    if header:
        text = f"{header}\n\n{text}"
    choices = _choices(f)
    return {"question": text, "choices": choices, "choices_offered": choices,
            "multi_select": f.kind == "multi" and bool(choices)}


def _match_option(f: FormField, text: str) -> tuple[bool, Any]:
    wanted = text.strip().casefold()
    for value, label in f.options:
        if wanted in (label.casefold(), str(value).casefold()):
            return True, value
    return False, None


def _coerce_multi(f: FormField, raw: Any) -> tuple[Any, str]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        raw = parsed if isinstance(parsed, list) else [p for p in raw.split(",") if p.strip()]
    values = []
    for item in raw:
        ok, value = _match_option(f, str(item))
        if not ok:
            return None, f"'{item}' is not one of: " + ", ".join(label for _, label in f.options)
        values.append(value)
    if len(values) < int(f.spec.get("minItems", 0)):
        return None, f"pick at least {f.spec['minItems']}"
    if "maxItems" in f.spec and len(values) > int(f.spec["maxItems"]):
        return None, f"pick at most {f.spec['maxItems']}"
    return values, ""


def _coerce_number(f: FormField, text: str) -> tuple[Any, str]:
    try:
        number = float(text)
    except ValueError:
        return None, f"'{text}' is not a number"
    if f.kind == "integer":
        if not number.is_integer():
            return None, f"'{text}' is not a whole number"
        number = int(number)
    if "minimum" in f.spec and number < f.spec["minimum"]:
        return None, f"must be at least {f.spec['minimum']}"
    if "maximum" in f.spec and number > f.spec["maximum"]:
        return None, f"must be at most {f.spec['maximum']}"
    return number, ""


def _format_error(fmt: str, text: str) -> str:
    checks = {
        "email": lambda: bool(_EMAIL.match(text)),
        "uri": lambda: bool(urlparse(text).scheme and urlparse(text).netloc),
        "date": lambda: bool(date.fromisoformat(text)),
        "date-time": lambda: bool(datetime.fromisoformat(text)),
    }
    check = checks.get(fmt)
    try:
        return "" if check is None or check() else f"'{text}' is not a valid {fmt}"
    except ValueError:
        return f"'{text}' is not a valid {fmt}"


def _coerce_string(f: FormField, text: str) -> tuple[Any, str]:
    if "minLength" in f.spec and len(text) < int(f.spec["minLength"]):
        return None, f"must be at least {f.spec['minLength']} characters"
    if "maxLength" in f.spec and len(text) > int(f.spec["maxLength"]):
        return None, f"must be at most {f.spec['maxLength']} characters"
    error = _format_error(str(f.spec.get("format") or ""), text)
    return (None, error) if error else (text, "")


def _coerce_boolean(_f: FormField, text: str) -> tuple[Any, str]:
    lowered = text.casefold()
    if lowered in _YES:
        return True, ""
    if lowered in _NO:
        return False, ""
    return None, f"'{text}' is not yes or no"


def _coerce_enum(f: FormField, text: str) -> tuple[Any, str]:
    ok, value = _match_option(f, text)
    return (value, "") if ok else (None, f"'{text}' is not one of: " + ", ".join(lb for _, lb in f.options))


_COERCERS = {"string": _coerce_string, "number": _coerce_number, "integer": _coerce_number,
             "boolean": _coerce_boolean, "enum": _coerce_enum}


def coerce_answer(f: FormField, raw: Any) -> tuple[bool, Any, str]:
    """One raw surface answer -> ``(present, value, error)``; a blank answer takes the default."""
    blank = raw is None or (isinstance(raw, (str, list)) and not (raw.strip() if isinstance(raw, str) else raw))
    if blank:
        if f.default is not None:
            return True, f.default, ""
        return False, None, ("this field is required" if f.required else "")
    if f.kind == "multi":
        value, error = _coerce_multi(f, raw)
    else:
        value, error = _COERCERS[f.kind](f, str(raw).strip())
    return error == "", value, error


@dataclass
class FormOutcome:
    action: str  # accept | decline | cancel
    content: Optional[dict] = None


# clarify ``outcome`` for a wait that ended without a submit -> elicitation action. Nobody could be
# reached (undelivered) reads as the user not providing the data; a dismissed or expired prompt is
# the spec's "cancel" (no explicit choice).
_UNSUBMITTED = {"cancelled": "cancel", "timed_out": "cancel", "undelivered": "decline"}


def _ask_batch(asker: FormAsker, batch: list[FormField], header: str, errors: dict) -> tuple[Optional[str], dict]:
    questions = []
    for index, f in enumerate(batch):
        entry = _question(f, header if index == 0 else "", errors.get(f.name, ""))
        questions.append({"qid": f"q{index}", **entry})
    reply = asker(questions) or {}
    outcome = str(reply.get("outcome") or "submitted")
    if outcome != "submitted":
        return _UNSUBMITTED.get(outcome, "cancel"), {}
    answers = reply.get("answers") or {}
    return None, {f.name: answers.get(f"q{i}") for i, f in enumerate(batch)}


def _collect(asker: FormAsker, fields: list[FormField], header: str) -> FormOutcome:
    content: dict[str, Any] = {}
    answered_any = False
    for start in range(0, len(fields), MAX_QUESTIONS):
        pending, errors = fields[start:start + MAX_QUESTIONS], {}
        for _attempt in range(MAX_REASKS + 1):
            stop, raw = _ask_batch(asker, pending, header if start == 0 and not errors else "", errors)
            if stop:
                return FormOutcome(stop)
            errors = {}
            for f in pending:
                answered_any = answered_any or raw.get(f.name) not in (None, "", [])
                present, value, error = coerce_answer(f, raw.get(f.name))
                if error:
                    errors[f.name] = error
                elif present:
                    content[f.name] = value
            pending = [f for f in pending if f.name in errors]
            if not pending:
                break
        if pending:  # still invalid after the re-asks
            return FormOutcome("decline")
    # A form whose every field was skipped and has no default is the user saying no.
    if not answered_any and not content and any(f.required for f in fields):
        return FormOutcome("decline")
    return FormOutcome("accept", content)


def ask_consent(asker: FormAsker, message: str) -> FormOutcome:
    """A form with no fields is a yes/no question: the server's own words, then Yes / No."""
    reply = asker([{"qid": "q0", "question": message, "choices": ["Yes", "No"],
                    "choices_offered": ["Yes", "No"], "multi_select": False}]) or {}
    outcome = str(reply.get("outcome") or "submitted")
    if outcome != "submitted":
        return FormOutcome(_UNSUBMITTED.get(outcome, "cancel"))
    answer = (reply.get("answers") or {}).get("q0")
    yes = isinstance(answer, str) and answer.strip().casefold() in _YES
    return FormOutcome("accept", {}) if yes else FormOutcome("decline")


def ask_form(asker: FormAsker, server_name: str, message: str, schema: Any) -> FormOutcome:
    """Ask the user for *schema*'s fields; returns accept + validated content, decline or cancel."""
    from tools.human_input_hooks import human_input_request

    fields = parse_form(schema)
    header = f"{server_name} asks: {message}"
    with human_input_request("elicitation", prompt=header) as human:
        result = _collect(asker, fields, header) if fields else ask_consent(asker, header)
        human.outcome = result.action
    return result
