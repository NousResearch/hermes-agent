"""Target parsing and resolution for send_message (platform:chat_id[:thread] → ids)."""

import logging
import re

logger = logging.getLogger("tools.send_message_tool")

# E.164 phone recipients ("+1555..."): the '+' fails the isdigit() rule and the channel directory
# cannot resolve a raw number, so keep the '+' and treat it as explicit.
_PHONE_PLATFORMS = frozenset({"photon", "signal", "sms", "whatsapp"})
_E164_TARGET_RE = re.compile(r"^\s*\+(\d{7,15})\s*$")
# Exceptions to "<PLATFORM>_HOME_CHANNEL" for error hints (email reads EMAIL_HOME_ADDRESS).
_HOME_CHANNEL_ENV_OVERRIDES = {"email": "EMAIL_HOME_ADDRESS"}


def _target_syntax_fn(platform_name: str):
    """The plugin's declared native target syntax, or None. A fresh CLI/cron process can parse a
    target before plugin discovery registered the platform, so discover first when the name is
    unknown (idempotent manifest scan; only this platform's adapter is materialized)."""
    from gateway.platform_registry import platform_registry
    if not platform_registry.is_registered(platform_name):
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    entry = platform_registry.get(platform_name)
    return entry.target_ref_syntax_fn if entry is not None else None


def _parse_target_ref(platform_name: str, target_ref: str):
    """(chat_id, thread_id, explicit): the plugin's ``target_ref_syntax_fn`` first, then in order
    E.164 phone numbers (keeping the '+'), bare numeric ids, Matrix ``!room``/``@user``, XMPP JIDs.
    Anything else goes to channel-directory resolution."""
    from gateway.platforms.send_targets import NOT_EXPLICIT
    parser = _target_syntax_fn(platform_name)
    if parser is not None:
        parsed = parser(target_ref)
        if parsed is NOT_EXPLICIT:
            return None, None, False
        if parsed is not None:
            return parsed[0], parsed[1], True
    if platform_name in _PHONE_PLATFORMS and _E164_TARGET_RE.fullmatch(target_ref):
        return target_ref.strip(), None, True
    if (target_ref.lstrip("-").isdigit() or (platform_name == "matrix" and target_ref.startswith(("!", "@")))
            or (platform_name == "xmpp" and "@" in target_ref)):
        return target_ref, None, True
    return None, None, False


def unknown_platform_error(platform_name: str) -> str:
    from hermes_cli.left_core_migration import platform_install_hint
    return f"Unknown or unregistered plugin platform: {platform_name}{platform_install_hint(platform_name)}"


def resolve_send_target(
    platform_name: str, target_ref: str, *, pass_unresolved_references: bool = False
) -> tuple[str | None, str | None, str | None]:
    """Resolve one send target the same way for every caller (model tool, CLI, cron).
    Channel-directory IDs are trusted; plugin parsers are the authority on native syntax. By
    default an unresolvable target is an error the model can act on. ``pass_unresolved_references``
    (no model in the loop: cron, react/unreact on native ids) hands an unresolvable target on a
    built-in platform, or a plugin platform without a parser, to the adapter as written; a plugin
    WITH a parser stays strict. The optional validator has the final say over every returned id."""
    from gateway.config import Platform
    from gateway.platform_registry import platform_registry
    entry = platform_registry.get(platform_name)

    def _validated(chat_id, thread_id):
        """``(chat_id, thread_id, None)`` when the plugin validator (if any) accepts, else an error."""
        if entry is None or entry.validate_target_ref_fn is None:
            return chat_id, thread_id, None
        try:
            verdict = entry.validate_target_ref_fn(chat_id)
        except Exception:
            logger.debug("Plugin target validator failed for %s", platform_name, exc_info=True)
            return None, None, f"Target validator failed for platform '{platform_name}'"
        if verdict is True:
            return chat_id, thread_id, None
        detail = f": {verdict}" if isinstance(verdict, str) and verdict else ""
        return None, None, f"Invalid target '{target_ref}' on {platform_name}{detail}"
    if entry is not None and entry.parse_target_ref_fn is not None:
        try:
            parsed = entry.parse_target_ref_fn(target_ref)
        except Exception:
            logger.debug("Plugin target parser failed for %s", platform_name, exc_info=True)
            return None, None, f"Target parser failed for platform '{platform_name}'"
        if parsed is not None:
            if (not isinstance(parsed, tuple) or len(parsed) != 2 or not isinstance(parsed[0], str)
                    or not parsed[0] or (parsed[1] is not None and not isinstance(parsed[1], str))):
                return None, None, f"Target parser for platform '{platform_name}' returned an invalid result"
            return _validated(*parsed)
    parsed_chat_id, parsed_thread_id, explicit = _parse_target_ref(platform_name, target_ref)
    if explicit and parsed_chat_id is not None:
        return _validated(parsed_chat_id, parsed_thread_id)
    resolution_failed = False
    try:
        from gateway.channel_directory import resolve_channel_name
        resolved = resolve_channel_name(platform_name, target_ref)
    except Exception:
        resolved = None
        resolution_failed = True
    if resolved:
        parsed_chat_id, parsed_thread_id, _ = _parse_target_ref(platform_name, resolved)
        return _validated(parsed_chat_id or resolved, parsed_thread_id)
    is_builtin = platform_name in {member.value for member in Platform}
    if entry is None and not is_builtin:
        return None, None, unknown_platform_error(platform_name)
    is_plugin = entry is not None and entry.source == "plugin" and not is_builtin
    if pass_unresolved_references and (not is_plugin or entry.parse_target_ref_fn is None):
        # Hand the raw target to the adapter unchanged (it validates).
        chat_id, thread_id, error = _validated(target_ref, None)
        if not error:
            logger.debug("Handing unresolved target '%s' to the %s adapter unchanged "
                         "(the adapter validates it)", target_ref, platform_name)
        return chat_id, thread_id, error
    if is_plugin:
        hint = "The plugin parser did not recognize it and no channel-directory entry matched."
    elif resolution_failed:
        hint = "Try using a numeric channel ID instead."
    else:
        hint = "Use send_message(action='list') to see available targets."
    return None, None, f"Could not resolve '{target_ref}' on {platform_name}. {hint}"
