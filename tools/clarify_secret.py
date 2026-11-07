"""Masked secret questions for the clarify tool (``clarify(secret_env=NAME)``).

The user types the value into the surface's masked secret prompt (the same one skill setup uses:
CLI panel, TUI/Desktop ``secret`` overlay, a local viewer's gateway turn). The surface stores it in
this profile's ``.env``; this module only ever sees the callback's metadata, so the value never
reaches the tool result, the transcript or the model. On success the name joins the terminal /
execute_code passthrough allowlist, so commands use it as ``$NAME``.
"""

import json

from hermes_constants import display_hermes_home
from tools.registry import tool_error
from tools.skills_tool_setup import _ENV_VAR_NAME_RE, resolve_secret_capture_callback


def _refusal(name: str) -> str:
    return tool_error(
        f"Masked secret entry is not available on this surface, so {name} cannot be collected here. "
        f"Tell the user to set it from a local Hermes CLI, TUI or Desktop session, or to add it to "
        f"{display_hermes_home()}/.env themselves. Never ask them to paste it into chat.")


def ask_secret(questions, name: str, interactive: bool) -> str:
    """Collect secret ``name`` through the masked prompt; the result names it, never carries it."""
    if not (isinstance(questions, list) and len(questions) == 1 and isinstance(questions[0], dict)):
        return tool_error("secret_env takes exactly one question saying what the secret is for.")
    question = str(questions[0].get("question") or "").strip()
    if not question:
        return tool_error("questions[0].question must say what the secret is for and where it is used.")
    if not _ENV_VAR_NAME_RE.match(name):
        return tool_error(f"secret_env must be an environment variable name, got {name!r}.")
    from gateway.session_context import session_is_messaging_surface
    # A chat channel never gets a prompt, even in a process that also serves a TUI/Desktop viewer.
    callback = resolve_secret_capture_callback() if interactive and not session_is_messaging_surface() else None
    if callback is None:
        return _refusal(name)
    try:
        stored = callback(name, question, {"source": "agent"})
    except ValueError as exc:  # a locked/denylisted key: the save refused it before writing
        return tool_error(f"{name} was not stored: {exc}")
    if not (isinstance(stored, dict) and stored.get("success") and not stored.get("skipped")):
        reason = str((stored or {}).get("reason") or "cancelled") if isinstance(stored, dict) else "cancelled"
        return json.dumps({"responses": [{"question": question, "status": "skipped", "stored_as": None}],
                           "outcome": "timed_out" if reason == "timeout" else "cancelled",
                           "notice": f"The user did not enter {name}. Do not ask for it in chat."})
    from tools.env_passthrough import is_env_passthrough, register_env_passthrough
    register_env_passthrough([name])
    usable = is_env_passthrough(name)
    notice = (f"Stored in this profile's secrets as {name}; you never see the value. Use it only as "
              f"${name} inside terminal commands or os.environ[{name!r}] in execute_code, and never print "
              f"it. An execute_code kernel already running in this session keeps its old environment: "
              f"pass reset=true on the next execute_code call (this clears its Python state) to see it."
              if usable else
              f"Stored in this profile's secrets as {name}; you never see the value. It is a Hermes "
              f"provider credential, so Hermes uses it directly and it is not passed to commands.")
    return json.dumps({"responses": [{"question": question, "status": "answered", "stored_as": name}],
                       "outcome": "submitted", "notice": notice})
