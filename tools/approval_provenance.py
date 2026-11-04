"""Why a flagged command ran without a prompt.

When a prior user decision (a ``command_allowlist`` entry, an 'always' approval, or an
'approve for session' answer) covers every finding in a dangerous command, the guard still
approves silently; these helpers attach that provenance to the guard result and turn it into
the terminal tool's ``approval`` note, so the run is explained instead of looking like a bypass.
Provenance never decides anything: callers build it only after the approval check passed.
"""


def preapproved_result(findings: list[tuple[str, str]], scope_of) -> dict:
    """Approved guard result for *findings* ``[(pattern_key, description)]`` that are all already
    approved; ``scope_of(key)`` labels each one 'permanent' or 'session'."""
    return {
        "approved": True,
        "message": None,
        "pre_approved": [{"description": text, "scope": scope_of(key)} for key, text in findings],
    }


def allowlisted_result(rule: str, description: str) -> dict:
    """Approved guard result for a flagged command waved through by a ``command_allowlist`` entry."""
    return {
        "approved": True,
        "message": None,
        "pre_approved": [{"description": description, "scope": "allowlist", "rule": rule}],
    }


_SCOPE_PHRASES = {
    "allowlist": "the command_allowlist entry `{rule}` in config.yaml",
    "permanent": "an 'always' approval the user saved earlier",
    "session": "an 'approve for session' answer earlier in this session",
}


def preapproval_note(entries: list[dict]) -> str:
    """One-sentence ``approval`` note for a guard result's ``pre_approved`` entries."""
    flagged = "; ".join(entry["description"] for entry in entries)
    reasons = []
    for entry in entries:
        phrase = _SCOPE_PHRASES[entry["scope"]].format(rule=entry.get("rule", ""))
        if len(entries) > 1:
            phrase = f"{entry['description']}: {phrase}"
        if phrase not in reasons:
            reasons.append(phrase)
    return f"Command was flagged ({flagged}) and auto-approved by {'; '.join(reasons)}."
