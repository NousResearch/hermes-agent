"""Runtime toolset selection. Configuration and credential loading belong to callers."""

from collections.abc import Callable
import logging
from typing import List, Set
from tools.toolset_scope import (
    _TOOLSET_PLATFORM_RESTRICTIONS,
    parse_platform_toolsets_value,
    toolset_allowed_for_platform as _toolset_allowed_for_platform,
)

logger = logging.getLogger(__name__)

PLATFORM_DEFAULT_TOOLSETS = {
    "cli": "hermes-cli",
    "telegram": "hermes-telegram",
    "discord": "hermes-discord",
    "slack": "hermes-slack",
    "whatsapp": "hermes-whatsapp",
    "whatsapp_cloud": "hermes-whatsapp",
    "signal": "hermes-signal",
    "bluebubbles": "hermes-bluebubbles",
    "email": "hermes-email",
    "homeassistant": "hermes-homeassistant",
    "mattermost": "hermes-mattermost",
    "matrix": "hermes-matrix",
    "dingtalk": "hermes-dingtalk",
    "feishu": "hermes-feishu",
    "wecom": "hermes-wecom",
    "wecom_callback": "hermes-wecom-callback",
    "weixin": "hermes-weixin",
    "qqbot": "hermes-qqbot",
    "yuanbao": "hermes-yuanbao",
    "webhook": "hermes-webhook",
    "api_server": "hermes-api-server",
    "cron": "hermes-cron"
}

CONFIGURABLE_TOOLSET_KEYS = (
    'web', 'browser', 'terminal', 'file', 'code_execution',
    'vision', 'video', 'image_gen', 'video_gen', 'x_search',
    'tts', 'stt', 'skills', 'todo', 'kanban',
    'memory', 'context_engine', 'session_search', 'connections', 'clarify',
    'delegation', 'cronjob', 'homeassistant', 'spotify', 'discord',
    'discord_admin', 'yuanbao', 'computer_use',
)

_DEFAULT_OFF_TOOLSETS = {
    "homeassistant", "spotify", "discord", "discord_admin", "video",
    "video_gen", "x_search", "a2a", "kanban",
}


#: Toolsets young enough that absence from a saved ``platform_toolsets`` list means "never offered", not
#: "declined": saving ``hermes tools`` freezes a platform's composite into an explicit list nothing adds to, so
#: a later toolset stays off forever for picker users while ``[hermes-cli]`` users inherit it.
#: MUST ship in the same release as the toolset and be emptied in the next: once a released build has put the
#: toolset on a checklist, an unchecking user's config is byte-identical to one saved before it existed and this
#: rule would turn the opt-out back on (stuck checkbox). ``check_fn``-gated toolsets cost nothing here; never
#: probe a remote service from this path — it runs on every CLI start, gateway session and cron tick.
_RECENTLY_SHIPPED_TOOLSETS: frozenset = frozenset()


_warned_invalid_platform_toolsets: Set[str] = set()


def _homeassistant_credentials_present() -> bool:
    """Return whether the active profile has a Home Assistant token."""
    try:
        from agent.secret_scope import get_secret
        return bool((get_secret("HASS_TOKEN", "") or "").strip())
    except Exception:
        return False


def get_plugin_toolset_keys() -> set:
    """Return the set of toolset keys provided by plugins."""
    try:
        # Non-blocking on CLI startup: while background discovery is still importing, serve last
        # launch's persisted key set instead of joining the discovery thread.
        from plugin_runtime.lifecycle import get_plugin_toolset_keys_nowait
        return get_plugin_toolset_keys_nowait()
    except Exception:
        return set()


def platform_default_toolset(platform: str) -> str:
    """Composite toolset a platform falls back to (plugin platforms derive ``hermes-<platform>``)."""
    return PLATFORM_DEFAULT_TOOLSETS.get(platform, f"hermes-{platform}")


def enabled_mcp_server_names(config: dict) -> Set[str]:
    """MCP servers globally enabled in config.yaml or by a plugin (shared by platform + cron resolvers). Enabled
    unless ``enabled`` is explicitly falsey; portable-plugin servers (in-memory) count — enabling the plugin is
    the opt-in."""
    from tools.mcp_tool_common import mcp_server_enabled

    mcp_servers = (config or {}).get("mcp_servers") or {}
    names = {
        str(name) for name, server_cfg in mcp_servers.items()
        if isinstance(server_cfg, dict) and mcp_server_enabled(server_cfg)
    }
    try:
        from plugin_runtime.lifecycle import get_portable_mcp_server_names_nowait
        portable = get_portable_mcp_server_names_nowait()
        names |= portable - set(mcp_servers)  # native config wins on a name collision (mirrors _load_mcp_config)
    except Exception:
        logger.debug("Failed to include portable MCP servers", exc_info=True)
    return names


def _enable_recently_shipped_toolsets(enabled_toolsets: Set[str], config: dict, platform: str) -> None:
    """Turn on toolsets that shipped after this platform's saved list (mutates ``enabled_toolsets``). Both "no"s
    outlive this: unchecking records ``known_builtin_toolsets`` (declined), and ``agent.disabled_toolsets`` is
    subtracted last in :func:`get_platform_tools`."""
    from toolsets import resolve_toolset

    offered = (config.get("known_builtin_toolsets") or {}).get(platform)
    declined = {str(ts) for ts in offered} if isinstance(offered, list) else set()
    default_ts = platform_default_toolset(platform)
    composite_tools = None
    for ts_key in sorted(_RECENTLY_SHIPPED_TOOLSETS):
        if ts_key in enabled_toolsets or ts_key in declined or not _toolset_allowed_for_platform(ts_key, platform):
            continue
        # Only enable where staying on the composite would have enabled it anyway; deliberately narrow
        # composites (hermes-acp, hermes-webhook) stay narrow.
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if composite_tools is None:
            composite_tools = set(resolve_toolset(default_ts))
        if not ts_tools or not ts_tools.issubset(composite_tools):
            continue
        enabled_toolsets.add(ts_key)


def _configurable_subset_of(tool_names: Set[str], platform: str) -> Set[str]:
    """Configurable toolsets whose STATIC membership is within ``tool_names`` (``include_registry=False``: a
    runtime-registered tool the composite never listed must not drop the whole toolset)."""
    from toolsets import resolve_toolset

    return {
        ts_key for ts_key in CONFIGURABLE_TOOLSET_KEYS if _toolset_allowed_for_platform(ts_key, platform)
        and (ts_tools := set(resolve_toolset(ts_key, include_registry=False))) and ts_tools <= tool_names}


def _default_off_toolsets(platform: str, explicitly_configured: bool) -> Set[str]:
    """Toolsets to strip from an implicit (composite-derived) enable set. A platform named after a default-off
    toolset (``homeassistant``) keeps it, except platform-restricted ones (``discord`` on discord stays OFF); a
    configured HASS_TOKEN is an explicit opt-in that must survive platforms resolving without a saved list.
    Platform-native default-off toolsets (``discord`` on discord) are off for unconfigured platforms as a
    security opt-in — an explicitly saved list IS that opt-in and lets them through."""
    default_off = set(_DEFAULT_OFF_TOOLSETS)
    if platform in default_off and platform not in _TOOLSET_PLATFORM_RESTRICTIONS:
        default_off.remove(platform)
    # Home Assistant is already runtime-gated by its check_fn (requires HASS_TOKEN to register any tools).
    # When a user has configured HASS_TOKEN, they've explicitly opted in — don't also strip it via
    # _DEFAULT_OFF_TOOLSETS, which would silently drop HA from platforms (e.g. cron) that run through
    # get_platform_tools without an explicit saved toolset list. Without this, Norbert's HA cron jobs
    # regressed after #14798 made cron honor per-platform tool config.
    if "homeassistant" in default_off and _homeassistant_credentials_present():
        default_off.remove("homeassistant")
    if explicitly_configured:
        default_off -= {ts for ts in default_off if platform in (_TOOLSET_PLATFORM_RESTRICTIONS.get(ts) or ())}
    return default_off


def configurable_toolset_keys() -> Set[str]:
    return {ts_key for ts_key in CONFIGURABLE_TOOLSET_KEYS}


def _platform_default_keys() -> Set[str]:
    return set(PLATFORM_DEFAULT_TOOLSETS.values())


def _explicit_toolsets(
    toolset_names: List[str], explicit_known_keys: Set[str], config: dict, platform: str,
    explicitly_configured: bool) -> Set[str]:
    """Enabled set when the saved list names configurable/plugin keys directly (subset inference over
    ``hermes-cli`` would re-enable disabled toolsets). A mixed list (``[hermes-cli, spotify]``) still expands the
    composite; _DEFAULT_OFF_TOOLSETS applies to that implicit expansion only."""
    from toolsets import resolve_toolset, TOOLSETS

    enabled = {ts for ts in toolset_names if ts in explicit_known_keys and _toolset_allowed_for_platform(ts, platform)}
    composite_tools = {
        t for ts_name in toolset_names if ts_name not in explicit_known_keys and ts_name in TOOLSETS
        for t in resolve_toolset(ts_name)}
    if composite_tools:
        enabled |= _configurable_subset_of(composite_tools, platform) - _default_off_toolsets(platform, explicitly_configured)
    _enable_recently_shipped_toolsets(enabled, config, platform)
    return enabled


def _composite_toolsets(
    toolset_names: List[str], platform: str, explicitly_configured: bool,
    xai_credentials_present: Callable[[], bool] | bool = False,
) -> Set[str]:
    """Enabled set inferred from composite names by reverse-mapping tool names (only while no explicit list is
    saved). ``x_search`` is not in any composite, so inject it when xAI creds exist and exempt it from default-off."""
    from toolsets import resolve_toolset

    all_tool_names = {t for ts_name in toolset_names for t in resolve_toolset(ts_name)}
    enabled = _configurable_subset_of(all_tool_names, platform)
    default_off = _default_off_toolsets(platform, explicitly_configured)
    has_xai = xai_credentials_present() if callable(xai_credentials_present) else xai_credentials_present
    if _toolset_allowed_for_platform("x_search", platform) and has_xai:
        enabled.add("x_search")
        default_off.discard("x_search")
    return enabled - default_off


def _enabled_plugin_toolsets(config: dict, platform: str, toolset_names: List[str], plugin_ts_keys: Set[str]) -> Set[str]:
    """Plugin toolsets: on by default unless default-off (bundled spotify) or "known" for this platform
    (``known_plugin_toolsets``, written on every save) and absent from the saved list."""
    known_for_platform = set((config.get("known_plugin_toolsets", {}) or {}).get(platform, []) or [])
    return {
        pts for pts in plugin_ts_keys
        if pts in toolset_names or (pts not in _DEFAULT_OFF_TOOLSETS and pts not in known_for_platform)
    }


def _context_engine_active(config: dict) -> bool:
    context_cfg = config.get("context") or {}
    name = str(context_cfg.get("engine") or "compressor").strip().lower() if isinstance(context_cfg, dict) else "compressor"
    return bool(name) and name != "compressor"


def coerce_platform_toolsets_value(value, platform: str):
    """Read a list-literal string saved for ``platform_toolsets.<platform>`` as the list it encodes.

    The scope parser is also used by ``hermes doctor`` and ``hermes plugins``,
    so every surface agrees on the user's selection (#115866).
    Any other non-list value is warned about once (naming the expected shape) and left as-is, so
    the default fallback below is loud rather than silent.
    """
    if value is None:
        return None
    parsed = parse_platform_toolsets_value(value)
    if parsed is not None:
        return parsed
    if platform not in _warned_invalid_platform_toolsets:
        _warned_invalid_platform_toolsets.add(platform)
        logger.warning(
            "platform_toolsets.%s is %r, expected a YAML list of toolset names "
            "(e.g. [terminal, file, web]) - falling back to the platform default. "
            "Run `hermes tools` to reconfigure.", platform, value)
    return value


def get_platform_tools(
    config: dict,
    platform: str,
    *,
    include_default_mcp_servers: bool = True,
    xai_credentials_present: Callable[[], bool] | bool = False,
) -> Set[str]:
    """Resolve enabled toolset names from supplied settings.

    The optional xAI input preserves the application's offline opt-in check
    without loading configuration or auth state here. Runtime registry checks
    and execution authorization remain independent of this selection.
    """
    platform_toolsets = config.get("platform_toolsets") or {}
    toolset_names = coerce_platform_toolsets_value(platform_toolsets.get(platform), platform)
    # An explicitly saved list (even a composite like ``hermes-discord``) is an opt-in to the platform's
    # native default-off toolsets — see _default_off_toolsets.
    # Track whether the user explicitly saved a toolset list for this platform (vs. falling back to the
    # platform default). See #35527.
    explicitly_configured = isinstance(toolset_names, list)
    # A saved empty selection is authoritative, including native/plugin/MCP additions.
    if explicitly_configured and not toolset_names:
        return set()
    if not explicitly_configured:
        toolset_names = [platform_default_toolset(platform)]
    # YAML may parse bare numeric names (``12306:``) as int; normalise so sorted() never mixes types.
    toolset_names = [str(ts) for ts in toolset_names]

    configurable_keys = configurable_toolset_keys()
    plugin_ts_keys = get_plugin_toolset_keys()
    platform_default_keys = _platform_default_keys()
    # Plugin toolsets are first-class on a saved list: ``[hermes-cli, a2a]`` must survive filtering.
    # Plugin-provided toolsets are first-class on a platform-toolsets list — explicit config like
    # ``[hermes-cli, a2a]`` must survive filtering just like a built-in configurable toolset would. See
    # issue #81163.
    explicit_known_keys = configurable_keys | plugin_ts_keys

    if any(ts in explicit_known_keys for ts in toolset_names):
        enabled_toolsets = _explicit_toolsets(toolset_names, explicit_known_keys, config, platform, explicitly_configured)
    else:
        enabled_toolsets = _composite_toolsets(toolset_names, platform, explicitly_configured, xai_credentials_present)

    _recover_platform_native_toolsets(enabled_toolsets, platform, skip=configurable_keys | plugin_ts_keys | platform_default_keys)
    if plugin_ts_keys:
        enabled_toolsets |= _enabled_plugin_toolsets(config, platform, toolset_names, plugin_ts_keys)

    # Context-engine tools are runtime-provided, not in static composites.
    # The explicit-empty case has already returned above.
    if _context_engine_active(config):
        enabled_toolsets.add("context_engine")

    # Explicit non-configurable entries (custom toolsets, MCP server names) pass through.
    explicit_passthrough = {ts for ts in toolset_names if ts not in explicit_known_keys and ts not in platform_default_keys}
    enabled_toolsets |= _merge_mcp_servers(config, toolset_names, explicit_passthrough, include_default_mcp_servers)

    # Legacy profile opt-in is a fallback only. A saved platform list (even
    # empty) is authoritative, so a later disable cannot silently re-enable it.
    if not explicitly_configured and "kanban" in (config.get("toolsets") or []):
        enabled_toolsets.add("kanban")

    # agent.disabled_toolsets is a global suppression list (#86661) and runs LAST so it overrides everything
    # above. It may arrive as a JSON-array string ("['memory']") from `hermes config set` or a JSON-mode editor.
    disabled_toolsets = (config.get("agent") or {}).get("disabled_toolsets")
    if disabled_toolsets:
        from agent.skill_utils import parse_config_string_list
        disabled_names = [name.strip() for name in parse_config_string_list(disabled_toolsets) if name.strip()]
        enabled_toolsets = _prune_toolsets_stripped_by_disabled(enabled_toolsets, disabled_names)

    if explicitly_configured and toolset_names:
        _warn_all_invalid_platform_toolsets(platform, toolset_names)
    return enabled_toolsets


def _prune_toolsets_stripped_by_disabled(enabled_toolsets: Set[str], disabled_names: List[str]) -> Set[str]:
    """Drop disabled names AND every toolset whose tools the runtime would strip anyway.

    The agent subtracts ``agent.disabled_toolsets`` at TOOL granularity (``model_tools._select_tool_names``),
    so disabling a composite like ``debugging`` removes the terminal/web/file tools even though those names
    never appear in the list. A name-only subtraction here left inspection surfaces (``hermes tools
    --summary``, banner, ``/tools``) showing toolsets as enabled that no session could call (#97015).
    Passthrough entries (MCP server names) and toolsets with no static tools (``context_engine``) are kept.
    """
    from tools.toolset_selection import resolve_toolset_selection
    from toolsets import resolve_toolset, validate_toolset

    remaining = enabled_toolsets - set(disabled_names)
    resolved = {name: set(resolve_toolset(name)) if validate_toolset(name) else set() for name in remaining}
    surviving: Set[str] = set().union(*resolved.values())
    for name in disabled_names:
        resolved_disabled = resolve_toolset_selection(name, disable=True)
        if resolved_disabled is not None:
            surviving.difference_update(resolved_disabled)
    return {name for name, tools in resolved.items() if not tools or tools & surviving}


def _recover_platform_native_toolsets(enabled_toolsets: Set[str], platform: str, *, skip: Set[str]) -> None:
    """Add non-configurable platform toolsets (discord, feishu_*) in place: in the default composite but not in
    CONFIGURABLE_TOOLSETS, so never in a checklist or saved list. Runs for BOTH ``get_platform_tools`` branches."""
    from toolsets import resolve_toolset, TOOLSETS

    platform_tool_universe = set(resolve_toolset(platform_default_toolset(platform)))
    configurable_tool_universe = {t for ts_key in CONFIGURABLE_TOOLSET_KEYS for t in resolve_toolset(ts_key)}
    claimed = {t for ts_key in enabled_toolsets for t in resolve_toolset(ts_key)}
    skip = skip | {k for k in TOOLSETS if k.startswith("hermes-")} | (set(_DEFAULT_OFF_TOOLSETS) - {platform})
    for ts_key, ts_def in TOOLSETS.items():
        # Posture toolsets (``coding``) are session-level selections made by agent/coding_context.py, not
        # per-platform capabilities to recover.
        if ts_key in skip or ts_def.get("includes") or ts_def.get("posture"):
            continue
        # Static membership: a registry-added tool absent from the platform composite must not block recovery
        # of a non-configurable toolset whose authored tools the composite lists.
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if not ts_tools or not ts_tools <= platform_tool_universe or ts_tools <= configurable_tool_universe:
            continue
        if not ts_tools <= claimed:
            enabled_toolsets.add(ts_key)
            claimed.update(ts_tools)


def _merge_mcp_servers(
    config: dict, toolset_names: List[str], explicit_passthrough: Set[str], include_default_mcp_servers: bool
) -> Set[str]:
    """Explicit passthrough entries plus this platform's MCP servers: listed names form an allowlist, else every
    globally enabled server (when ``include_default_mcp_servers``); the ``no_mcp`` sentinel disables all."""
    enabled_mcp_servers = enabled_mcp_server_names(config)
    result = explicit_passthrough - enabled_mcp_servers
    if "no_mcp" in toolset_names:
        return result - {"no_mcp"}
    explicit_mcp_servers = explicit_passthrough & enabled_mcp_servers
    if include_default_mcp_servers and not explicit_mcp_servers:
        return result | enabled_mcp_servers
    return result | explicit_mcp_servers


def _warn_all_invalid_platform_toolsets(platform: str, explicit: list) -> None:
    """Warn once when an explicit platform list has only invalid names (``hermes`` for ``hermes-cli`` → no
    native tools), at session tool resolution rather than only in update/doctor."""
    from toolsets import validate_toolset

    named = [str(t) for t in explicit if isinstance(t, str) and t]
    if named and not any(validate_toolset(t) for t in named) and platform not in _warned_invalid_platform_toolsets:
        _warned_invalid_platform_toolsets.add(platform)
        logger.warning(
            "platform '%s' has no valid toolsets configured (unknown "
            "name(s): %s) - tools will be unavailable. Run `hermes tools` "
            "to reconfigure. See issue #38798.",
            platform, ", ".join(named))

