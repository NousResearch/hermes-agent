import json
from pathlib import Path

TITLE_LIMIT = 40


def _rejected(reason: str, retryable: bool = False) -> str:
    """``retryable``: the same arguments may start the chat on another try (the handoff card's Retry)."""
    return json.dumps({"status": "rejected", "reason": reason, "retryable": retryable})


def start_chat(args: dict, caller_id: str | None = None) -> str:
    from agent.onboarding import PROFILE_BUILD_FLAG, mark_seen
    from gateway.session_context import get_session_env
    from hermes_cli.profiles import SETUP_PROFILE_MARKER
    from hermes_cli.setup_profile import mark_completed
    from hermes_constants import profile_name_for_home
    from tui_gateway import server
    from tui_gateway.transport import bind_transport, reset_transport

    caller = server._sessions.get(get_session_env("HERMES_UI_SESSION_ID", "") if caller_id is None else caller_id)
    if caller is None:
        return _rejected("start_chat works only from a chat in the Hermes desktop app.")
    caller_home = Path(caller.get("profile_home") or server._hermes_home)
    message = str(args.get("message") or "").strip()
    if not message:
        return _rejected("message is empty: pass the new chat's first message.")
    title = str(args.get("title") or "").strip()
    if len(title) > TITLE_LIMIT:
        return _rejected(f"title has {len(title)} characters; the limit is {TITLE_LIMIT}.")
    profile = str(args.get("profile") or "").strip() or profile_name_for_home(caller.get("profile_home"))
    try:
        home = server._profile_home(profile)
    except server.ProfileUnavailableError as exc:
        return _rejected(str(exc))
    target_home = Path(home or server._hermes_home)
    token = bind_transport(caller.get("transport"))
    try:
        with server._session_profile_runtime_scope({"profile_home": str(home) if home else None}):
            created = server._create_session(
                None, {"profile": profile or "", "source": caller.get("source"), "title": title})
            if "error" in created:
                return _rejected(created["error"]["message"], retryable=True)
            result = created["result"]
            mark_seen(target_home / "config.yaml", PROFILE_BUILD_FLAG)
            submitted = server._methods["prompt.submit"](None, {"session_id": result["session_id"], "text": message})
            if "error" in submitted:
                server._methods["session.close"](None, {"session_id": result["session_id"]})
                return _rejected(submitted["error"]["message"], retryable=True)
    finally:
        reset_transport(token)
    if (caller_home / SETUP_PROFILE_MARKER).is_file() and target_home.resolve() != caller_home.resolve():
        mark_completed()
    name = result["info"]["profile_name"]
    return json.dumps({
        "status": "started", "session_id": result["stored_session_id"], "profile": name, "title": title or None,
        "message": f"Started '{title or message[:TITLE_LIMIT]}' in {name}. Do not call start_chat again for this task.",
    })
