"""Send Gate Tool Registration

This tool implements Layer 2 of the send-gate feature: preventing the send tool
from being registered when send_gate is disabled on any platform.

Layer 1 (in gateway/platforms/base.py) makes send() raise SendGateDisabledException
when send_gate=disabled. Layer 2 (here) prevents the tool from appearing in the
available_tools list when any platform has send_gate=disabled.

The tool is intentionally NOT registered when send_gate is disabled globally,
keeping it out of the agent's toolset entirely rather than just raising at runtime.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_gateway_config() -> Optional[object]:
    """Load the gateway config if available.

    Returns None if not in gateway context or config is unavailable.
    """
    try:
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        return config
    except (ImportError, FileNotFoundError, Exception):
        # Not in gateway context, or config loading failed
        return None


def _check_send_gate_enabled() -> bool:
    """Check if send_gate is enabled for all platforms.

    Returns True (tool available) only if:
    - No gateway config exists (running in agent/CLI context, not gateway)
    - Gateway config exists and all connected platforms have send_gate enabled

    Returns False (tool unavailable) if:
    - Any connected platform has send_gate set to "disabled"

    In CLI context, send_message_tool functions remain available via direct
    import (cron, CLI send command, etc.). This check only prevents the tool
    from being registered as an agent-callable model tool.
    """
    config = _get_gateway_config()
    if config is None:
        # Not in gateway context; allow registration for CLI/cron use
        return True

    # Check all platforms for send_gate configuration
    try:
        platforms = getattr(config, "platforms", {})
        for platform, platform_config in platforms.items():
            if not getattr(platform_config, "enabled", True):
                # Skip disabled platforms
                continue

            extra = getattr(platform_config, "extra", {})
            send_gate = extra.get("send_gate", "enabled").lower()

            if send_gate == "disabled":
                # Any enabled platform with send_gate=disabled blocks tool registration
                platform_name = getattr(platform, "value", str(platform))
                logger.debug(
                    "Send tool registration blocked: platform '%s' has send_gate=disabled",
                    platform_name,
                )
                return False
    except Exception as e:
        logger.debug("Error checking send_gate config: %s; allowing send tool", e)
        # On any error, allow registration (fail-open rather than blocking)
        return True

    return True


# This is a placeholder schema for documentation purposes. The send tool is
# intentionally NOT registered via registry.register() to avoid cluttering
# the agent with a tool that should only be used via direct import (cron,
# CLI send command, gateway notifier, MCP server).
#
# If in the future we want to expose send as an agent-callable tool, the
# schema would look like this:
#
# SEND_SCHEMA = {
#     "name": "send",
#     "description": (
#         "Send a message to a connected messaging platform.\n\n"
#         "Supports sending to specific channels/users on Telegram, Discord, Slack, "
#         "and other connected platforms. When send_gate is disabled, this tool "
#         "is not available."
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "platform": {
#                 "type": "string",
#                 "description": "Target platform (e.g., 'telegram', 'discord')",
#             },
#             "chat_id": {
#                 "type": "string",
#                 "description": "Chat/channel ID or name on the target platform",
#             },
#             "message": {
#                 "type": "string",
#                 "description": "Message content to send",
#             },
#         },
#         "required": ["platform", "chat_id", "message"],
#     },
# }


# NOTE: The send tool is intentionally NOT registered as an agent-callable
# model tool. Send functionality remains available via direct import by:
#   - CLI (hermes send command)
#   - Cron delivery
#   - Gateway notifier
#   - MCP server
#
# This module exists to establish the _check_send_gate_enabled() function
# and serve as the attachment point for Layer 2 send-gate filtering if/when
# the send tool becomes agent-callable in the future.
