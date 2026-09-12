"""Target parsing and resolution for send_message (platform:chat_id[:thread] → ids)."""

import logging
import re

logger = logging.getLogger("tools.send_message_tool")

_TELEGRAM_TOPIC_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")
_NUMERIC_TOPIC_RE = _TELEGRAM_TOPIC_TARGET_RE  # Discord snowflakes: numeric, same "<id>[:<thread>]" shape
_FEISHU_TARGET_RE = re.compile(r"^\s*((?:oc|ou|on|chat|open)_[-A-Za-z0-9]+)(?::([-A-Za-z0-9_]+))?\s*$")
# Slack conversation IDs: C (public), G (private/group), D (DM); uppercase alnum, 9+ chars. User IDs
# (U...) become ``user:U...`` and are opened as D... conversations first (posting straight to a U/W
# id fails); ``@handle`` -> ``user_name:...`` resolves via users.list.
_SLACK_TARGET_RE = re.compile(r"^\s*([CGD][A-Z0-9]{8,})\s*$")
_SLACK_USER_ID_RE = re.compile(r"^\s*(U[A-Z0-9]{8,})\s*$")
_SLACK_USER_NAME_RE = re.compile(r"^\s*@([A-Za-z0-9._-]{1,80})\s*$")
_SLACK_MENTION_RE = re.compile(r"^\s*<@(U[A-Z0-9]{8,})(?:\|[^>]+)?>\s*$")
# Session-derived Slack thread targets use "<conversation_id>:<thread_ts>".
_SLACK_THREAD_TARGET_RE = re.compile(r"^\s*([CGD][A-Z0-9]{8,}):([^\s:]+)\s*$")
_WEIXIN_TARGET_RE = re.compile(r"^\s*((?:wxid|gh|v\d+|wm|wb)_[A-Za-z0-9_-]+|[A-Za-z0-9._-]+@chatroom|filehelper)\s*$")
_YUANBAO_TARGET_RE = re.compile(r"^\s*((?:group|direct):[^:]+)\s*$")
# E.164 phone recipients ("+1555..."): the '+' fails the isdigit() rule and the channel directory
# cannot resolve a raw number, so keep the '+' and treat it as explicit.
_PHONE_PLATFORMS = frozenset({"photon", "signal", "sms", "whatsapp"})
_E164_TARGET_RE = re.compile(r"^\s*\+(\d{7,15})\s*$")
_PHOTON_DM_GUID_RE = re.compile(r"^any;-;\+\d{6,}$")  # mirrors _DM_CHAT_GUID_RE in the photon adapter
# WhatsApp JIDs (@g.us, @s.whatsapp.net, @lid, broadcast/newsletter) and Buzz UUIDs are native targets
# the adapter accepts verbatim — explicit, never home-channel. A valid email address likewise.
_WHATSAPP_JID_RE = re.compile(r"^\s*[\w-]+@(?:g\.us|s\.whatsapp\.net|lid|broadcast|newsletter)\s*$", re.IGNORECASE)
_BUZZ_UUID_RE = re.compile(r"^\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\s*$", re.IGNORECASE)
_EMAIL_TARGET_RE = re.compile(r"^\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\s*$")
# Exceptions to "<PLATFORM>_HOME_CHANNEL" for error hints (email reads EMAIL_HOME_ADDRESS).
_HOME_CHANNEL_ENV_OVERRIDES = {"email": "EMAIL_HOME_ADDRESS"}

_UNRESOLVED = object()  # sentinel: stop parsing, target is NOT explicit (skip generic rules)


# Per-platform explicit-target parsers: target_ref -> (chat_id, thread_id), None to fall
# through to the generic rules in _parse_target_ref, or _UNRESOLVED.
def _parse_regex_groups(regex, *, thread_group=True):
    """Explicit when ``regex`` fully matches: chat_id = group 1, thread = group 2 (or None)."""
    return lambda ref: ((m.group(1), m.group(2) if thread_group else None)
                        if (m := regex.fullmatch(ref)) else None)


def _parse_regex_stripped(regex):
    """Explicit when ``regex`` fully matches; returns the stripped ref verbatim."""
    return lambda ref: (ref.strip(), None) if regex.fullmatch(ref) else None


def _parse_nonempty(ref):
    # ntfy topics and WeCom ids (the adapter picks the send command) are explicit when non-empty.
    stripped = ref.strip()
    return (stripped, None) if stripped else None


def _parse_telegram(ref):
    # "<chat_id>[:<topic_id>]" or an @username (usernames must not be force-int'd).
    parsed = _parse_regex_groups(_TELEGRAM_TOPIC_TARGET_RE)(ref)
    if parsed:
        return parsed
    from plugins.platforms.telegram.telegram_ids import parse_telegram_username_target
    return _parse_nonempty(parse_telegram_username_target(ref) or "")


# (regex, chat_id template, thread comes from group 2) — thread form before bare id.
_SLACK_FORMS = ((_SLACK_THREAD_TARGET_RE, "{}", True), (_SLACK_TARGET_RE, "{}", False),
                (_SLACK_USER_ID_RE, "user:{}", False), (_SLACK_MENTION_RE, "user:{}", False),
                (_SLACK_USER_NAME_RE, "user_name:{}", False))


def _parse_slack(ref):
    return next(((template.format(m.group(1)), m.group(2) if has_thread else None)
                 for regex, template, has_thread in _SLACK_FORMS if (m := regex.fullmatch(ref))), None)


def _parse_matrix(ref):
    # "<room>:$<event_id>" addresses a thread (rfind: room ids contain ':'). Bare "!room" /
    # "@user" go via the generic rule so the numeric check keeps precedence.
    trimmed = ref.strip()
    split_idx = trimmed.rfind(":$")
    return (trimmed[:split_idx], trimmed[split_idx + 1:]) if split_idx > 0 else None


def _parse_yuanbao(ref):
    # "group:<code>" / "direct:<id>"; a bare number is a group code (never generic rules).
    match = _YUANBAO_TARGET_RE.fullmatch(ref)
    if match:
        return match.group(1), None
    return (f"group:{ref.strip()}", None) if ref.strip().isdigit() else _UNRESOLVED


def _parse_signal(ref):
    # "group:<id>" is a native group target; an empty id is not explicit.
    stripped = ref.strip()
    if not stripped.startswith("group:"):
        return None
    group_id = stripped[len("group:"):].strip()
    return (f"group:{group_id}", None) if group_id else _UNRESOLVED


_PLATFORM_PARSERS = {
    "telegram": _parse_telegram,
    "feishu": _parse_regex_groups(_FEISHU_TARGET_RE),
    "discord": _parse_regex_groups(_NUMERIC_TOPIC_RE),  # "<channel>[:<thread>]" snowflakes
    "slack": _parse_slack,
    "matrix": _parse_matrix,
    "weixin": _parse_regex_groups(_WEIXIN_TARGET_RE, thread_group=False),
    "yuanbao": _parse_yuanbao,
    "ntfy": _parse_nonempty,
    "email": _parse_regex_stripped(_EMAIL_TARGET_RE),
    # Native WhatsApp JIDs pass through verbatim; E.164 numbers use the phone rule.
    "whatsapp": _parse_regex_stripped(_WHATSAPP_JID_RE),
    "buzz": _parse_regex_stripped(_BUZZ_UUID_RE),
    "signal": _parse_signal,
    "wecom": _parse_nonempty,
    # Photon DM GUIDs are adapter-native ids (mirrors the react handler).
    "photon": _parse_regex_stripped(_PHOTON_DM_GUID_RE)}


def _parse_target_ref(platform_name: str, target_ref: str):
    """(chat_id, thread_id, explicit): platform parser first, then in order E.164 phone
    numbers (keeping the '+'), bare numeric ids, Matrix ``!room``/``@user``, XMPP JIDs.
    Anything else goes to channel-directory resolution."""
    parser = _PLATFORM_PARSERS.get(platform_name)
    if parser is not None:
        parsed = parser(target_ref)
        if parsed is _UNRESOLVED:
            return None, None, False
        if parsed is not None:
            return parsed[0], parsed[1], True
    if platform_name in _PHONE_PLATFORMS and _E164_TARGET_RE.fullmatch(target_ref):
        return target_ref.strip(), None, True
    if (target_ref.lstrip("-").isdigit() or (platform_name == "matrix" and target_ref.startswith(("!", "@")))
            or (platform_name == "xmpp" and "@" in target_ref)):
        return target_ref, None, True
    return None, None, False


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
        return None, None, f"Unknown or unregistered plugin platform: {platform_name}"
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
