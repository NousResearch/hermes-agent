#!/usr/bin/env python3
"""
Toolsets Module

This module provides a flexible system for defining and managing tool aliases/toolsets.
Toolsets allow you to group tools together for specific scenarios and can be composed
from individual tools or other toolsets.

Features:
- Define custom toolsets with specific tools
- Compose toolsets from other toolsets
- Built-in common toolsets for typical use cases
- Easy extension for new toolsets
- Support for dynamic toolset resolution

Usage:
    from toolsets import get_toolset, resolve_toolset, get_all_toolsets
    
    # Get tools for a specific toolset
    tools = get_toolset("research")
    
    # Resolve a toolset to get all tool names (including from composed toolsets)
    all_tools = resolve_toolset("full_stack")
"""

from typing import List, Dict, Any, Set, Optional


# GWS operations exposed to the model as native tools — two layers built on the
# shared engine in tools/gws_ops.py:
#   gws_*     → operate on the SESSION USER's own account (OAuth; trusted process)
#   gws_dwd_* → domain-wide delegation on ANY account in an allowed domain
#               (impersonation; trusted process; data-only returns; never sends)
# Both are registered by tools/gws_ops_tools.py and tools/gws_dwd_tools.py.
_GWS_OPS = [
    "gws_gmail_search",
    "gws_gmail_get",
    "gws_gmail_thread_get",
    "gws_gmail_list_labels",
    "gws_gmail_labels_modify",
    "gws_gmail_draft_create",
    "gws_gmail_reply_draft",
    "gws_gmail_draft_list",
    "gws_gmail_draft_delete",
    "gws_gmail_trash",
    "gws_gmail_untrash",
    # Auto-registered by tools/gws_ops_tools.py from the OPERATIONS table in
    # tools/gws_ops.py. Added there 2026-09-13 but not here, which silently
    # disabled the whole `messaging` toolset -- see static_toolset_tools().
    "gws_gmail_attachment_get",
    "gws_drive_list",
    "gws_drive_get",
    "gws_drive_create_folder",
    "gws_drive_create_file",
    "gws_drive_move",
    "gws_drive_update",
    "gws_drive_delete",
    "gws_drive_transfer_ownership",
    "gws_drive_permissions_list",
    "gws_drive_permissions_add",
    "gws_drive_permissions_remove",
]

_GWS_DWD_OPS = [
    "gws_dwd_resolve",
    "gws_dwd_gmail_search",
    "gws_dwd_gmail_get",
    "gws_dwd_gmail_thread_get",
    "gws_dwd_gmail_list_labels",
    "gws_dwd_gmail_labels_modify",
    "gws_dwd_gmail_draft_create",
    "gws_dwd_gmail_reply_draft",
    "gws_dwd_gmail_draft_list",
    "gws_dwd_gmail_draft_delete",
    "gws_dwd_gmail_trash",
    "gws_dwd_gmail_untrash",
    # Auto-registered from the same OPERATIONS table as its OAuth twin; see
    # the note on "gws_gmail_attachment_get" above.
    "gws_dwd_gmail_attachment_get",
    "gws_dwd_drive_list",
    "gws_dwd_drive_get",
    "gws_dwd_drive_create_folder",
    "gws_dwd_drive_create_file",
    "gws_dwd_drive_move",
    "gws_dwd_drive_update",
    "gws_dwd_drive_delete",
    "gws_dwd_drive_transfer_ownership",
    "gws_dwd_drive_permissions_list",
    "gws_dwd_drive_permissions_add",
    "gws_dwd_drive_permissions_remove",
]

# Tools that are registered + callable internally (sandbox RPC stubs, e.g.
# gws_fetch_token) but must NEVER be presented to the LLM as callable tools.
# They stay in _HERMES_CORE_TOOLS so the execute_code sandbox still generates
# their RPC stubs; model_tools._compute_tool_definitions filters them out of
# the LLM-facing schema list at the very end.
_RPC_ONLY_TOOLS = frozenset({
    "gws_fetch_token",
})


# Shared tool list for CLI and all messaging platform toolsets.
# Edit this once to update all platforms simultaneously.
_HERMES_CORE_TOOLS = [
    # Web
    "web_search", "web_extract", "apify_run_actor",
    # Terminal + process management
    "terminal", "process",
    # File manipulation
    "read_file", "write_file", "patch", "search_files",
    # Vision + image generation
    "vision_analyze", "image_generate",
    # Skills
    "skills_list", "skill_view", "skill_manage",
    # Browser automation
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_get_images",
    "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
    "nopecha_recognition",
    # Text-to-speech
    "text_to_speech",
    # Planning & memory
    "todo", "memory",
    # Session history search
    "session_search",
    # Clarifying questions
    "clarify",
    # Code execution + delegation
    "execute_code", "delegate_task",
    # Cronjob management
    "cronjob",
    # Cross-platform messaging (gated on gateway running via check_fn)
    "send_message",
    # OAuth / auth-link delivery (gated on HERMES_OAUTH_CLIENT_ID via check_fn)
    "send_oauth_url",
    # Kelsa CRM per-user OAuth (2026-07-19 fix) -- these are registered under
    # toolset "oauth" in tools/kelsa_tool.py but were never listed in any
    # resolvable toolset here, so kelsa_login/kelsa_complete_login were
    # invisible to the agent. It fell back to the broken legacy
    # `hermes mcp --auth oauth` flow and to hand-rolled execute_code calls
    # that lost a freshly-authorized token to a vault-write crash. See
    # tools/kelsa_auth.py and tools/kelsa_tool.py.
    "kelsa_login", "kelsa_complete_login", "kelsa_list_tools", "kelsa_call_tool",
    # TailorTalk MCP per-user OAuth (2026-10-07) -- same shape as Kelsa; see
    # tools/tailortalk_auth.py and tools/tailortalk_tool.py.
    "tailortalk_login", "tailortalk_list_tools", "tailortalk_call_tool",
    # Home Assistant smart home control (gated on HASS_TOKEN via check_fn)
    "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
    # Kanban multi-agent coordination — only in schema when the agent is
    # spawned as a kanban worker (HERMES_KANBAN_TASK env set) or the current
    # profile explicitly enables the kanban toolset. Gated via check_fn in
    # tools/kanban_tools.py.
    "kanban_show", "kanban_list",
    "kanban_complete", "kanban_block", "kanban_heartbeat",
    "kanban_comment", "kanban_create", "kanban_link",
    "kanban_unblock",
    # Computer use (macOS, gated on cua-driver being installed via check_fn)
    "computer_use",
    # Smart browser and cloud browser tools
    "smart_browser",
    "browser_use_cloud",
    # OpenRouter ad-hoc model routing (explicit-trigger only)
    "call_openrouter_model",
    # User management is removed from LLM-accessible tools.
    # Admin user management is handled via admin.ahfl.in.
    # The manage_user module at tools/user_mgmt_tool.py is preserved
    # for the admin app to import directly (admin-app/app/vault_client.py).
    # WhatsApp deep-link generator -- sole sanctioned way to build wa.me URLs
    "whatsapp_link",
    # GWS multi-account resolver -- maps account email/label -> vault service_name
    "gws_resolve_account",
    # GWS token fetch RPC tool (sandbox-only path to vault credentials via
    # execute_code; registered under toolset "oauth"). RPC-only: filtered out of
    # the LLM-facing schema by model_tools via _RPC_ONLY_TOOLS, but kept in core
    # so the execute_code sandbox still generates its stub (build_service works).
    "gws_fetch_token",
    # GWS native operations (trusted-process, data-only returns). OAuth =
    # session user's own account; DWD = impersonate any allowed-domain account.
    *_GWS_OPS,
    *_GWS_DWD_OPS,
    # Sarvam AI tools (vault-backed key, session-scoped; tools/sarvam_tools.py)
    "sarvam_translate",
    "sarvam_transliterate",
    "sarvam_identify_language",
    "sarvam_llm_complete",
    "sarvam_stt_transcribe",
    "sarvam_text_analytics",
    "sarvam_tts",
    "sarvam_voice",
    # Contacts/entities registry lookup (fuzzy/ranked search across the
    # Google Sheets registry) -- per-user OAuth via gws_auth, no SA/DWD.
    "contact_resolver",
    "entity_resolver",
    "noun_learner",
    # Document OCR (Mistral OCR 4) -- gated on MISTRAL_API_KEY via check_fn
    "ocr_mistral",
    # Certificate renewal (tools/cert_status_tool.py): reads the status
    # file written by the host self-heal script; can request forced renewal.
    "cert_status", "cert_renew_request",
]

# Webhook events may originate from untrusted third-party content (for example,
# public PR titles/comments). Keep the default webhook toolset intentionally
# constrained to avoid local file/system execution by prompt injection.
_HERMES_WEBHOOK_SAFE_TOOLS = [
    "web_search",
    "web_extract",
    "vision_analyze",
    "clarify",
]

# Interactive-UI tools that have no sensible rendering on HTTP/API/editor
# surfaces (they render multi-choice prompts / cross-channel send buttons).
_HERMES_INTERACTIVE_UI_TOOLS = {"clarify", "send_message"}

# Shared core bundle for the editor (ACP) surface: every other platform
# composite minus the interactive-UI tools.
#
# NOTE: the API server (OpenWebUI) used to share this bundle. It no longer
# does — it carries the full _HERMES_CORE_TOOLS, identical to Telegram. See
# the "hermes-api-server" entry below.
_HERMES_CORE_TOOLS_NONINTERACTIVE = [
    t for t in _HERMES_CORE_TOOLS if t not in _HERMES_INTERACTIVE_UI_TOOLS
]


# Core toolset definitions
# These can include individual tools or reference other toolsets
TOOLSETS = {
    # Basic toolsets - individual tool categories
    "web": {
        "description": "Web research and content extraction tools",
        "tools": ["web_search", "web_extract"],
        "includes": []  # No other toolsets included
    },
    
    "search": {
        "description": "Web search only (no content extraction/scraping)",
        "tools": ["web_search"],
        "includes": []
    },
    "sarvam": {
        "description": "Sarvam AI (vault-backed key, session-scoped; tools/sarvam_tools.py) - STT, TTS, voice loop, LLM, translate, transliterate, language ID, text analytics.",
        "tools": [
            "sarvam_translate",
            "sarvam_transliterate",
            "sarvam_identify_language",
            "sarvam_llm_complete",
            "sarvam_stt_transcribe",
            "sarvam_text_analytics",
            "sarvam_tts",
            "sarvam_voice",
        ],
        "includes": []
    },



    "x_search": {
        "description": (
            "Search X (Twitter) posts and threads via xAI's built-in "
            "x_search Responses tool. Available when xAI credentials are "
            "configured (SuperGrok OAuth or XAI_API_KEY). Off by default; "
            "enable in `hermes tools` → X (Twitter) Search."
        ),
        "tools": ["x_search"],
        "includes": []
    },
    
    "vision": {
        "description": "Image analysis and vision tools",
        "tools": ["vision_analyze"],
        "includes": []
    },

    "video": {
        "description": "Video analysis and understanding tools (opt-in, not in default toolset)",
        "tools": ["video_analyze"],
        "includes": []
    },
    
    "image_gen": {
        "description": "Creative generation tools (images)",
        "tools": ["image_generate"],
        "includes": []
    },

    "video_gen": {
        "description": (
            "Video generation tools. Single ``video_generate`` tool covers "
            "text-to-video (prompt only) and image-to-video (prompt + "
            "image_url) — the active backend auto-routes. Configure via "
            "``hermes tools`` → Video Generation."
        ),
        "tools": ["video_generate"],
        "includes": []
    },

    "computer_use": {
        "description": (
            "Background macOS desktop control via cua-driver — screenshots, "
            "mouse, keyboard, scroll, drag. Does NOT steal the user's cursor "
            "or keyboard focus. Works with any tool-capable model."
        ),
        "tools": ["computer_use"],
        "includes": []
    },

    "terminal": {
        "description": "Terminal/command execution and process management tools",
        "tools": ["terminal", "process"],
        "includes": []
    },
    
    "moa": {
        "description": "Advanced reasoning and problem-solving tools",
        "tools": ["mixture_of_agents"],
        "includes": []
    },
    
    "skills": {
        "description": "Access, create, edit, and manage skill documents with specialized instructions and knowledge",
        "tools": ["skills_list", "skill_view", "skill_manage"],
        "includes": []
    },
    
    "browser": {
        "description": "Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs",
        "tools": [
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console", "browser_cdp",
            "browser_dialog", "web_search",
            "smart_browser",
            "browser_use_cloud", "nopecha_recognition"
        ],
        "includes": []
    },

    "external_model": {
        "description": "Route a sub-task to a user-specified model via OpenRouter (explicit request only)",
        "tools": ["call_openrouter_model"],
        "includes": []
    },

    "cronjob": {
        "description": "Cronjob management tool - create, list, update, pause, resume, remove, and trigger scheduled tasks",
        "tools": ["cronjob"],
        "includes": []
    },
    
    "messaging": {
        "description": "Cross-platform messaging: send messages to Telegram, Discord, Slack, SMS, etc.",
        "tools": ["send_message"],
        "includes": ["oauth"]
    },

    "oauth": {
        "description": "OAuth-related tools (e.g. send authorization links as Telegram buttons, "
                       "CLI prints, or pre-formatted markdown links). Source-of-truth URLs are "
                       "computed inside the tool — the LLM never sees them, which prevents the "
                       "class of failures where small chat models silently truncate substrings "
                       "like 'google.' from OAuth client_ids.",
        "tools": ["send_oauth_url", "kelsa_login", "kelsa_complete_login",
                  "kelsa_list_tools", "kelsa_call_tool",
                  "tailortalk_login", "tailortalk_list_tools", "tailortalk_call_tool"],
        "includes": []
    },

    "gws-dwd": {
        "description": "Domain-wide delegation (DWD): act on ANY Google Workspace account "
                       "in an allowed domain (including non-vault accounts like ex-employees) "
                       "via the system service-account key. Trusted-process, data-only returns; "
                       "sending email is permanently blocked (draft-only).",
        "tools": _GWS_DWD_OPS,
        "includes": []
    },

    "general": {
        "description": "General-purpose registry lookup tools: fuzzy/ranked contact and entity "
                       "resolution plus noun-learner write-back across the Google Sheets "
                       "registry (per-user OAuth via gws_auth)",
        "tools": ["contact_resolver", "entity_resolver", "noun_learner"],
        "includes": []
    },

    
    "file": {
        "description": "File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)",
        "tools": ["read_file", "write_file", "patch", "search_files"],
        "includes": []
    },
    
    "tts": {
        "description": "Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, OpenAI, or xAI",
        "tools": ["text_to_speech"],
        "includes": []
    },
    
    "todo": {
        "description": "Task planning and tracking for multi-step work",
        "tools": ["todo"],
        "includes": []
    },
    
    "memory": {
        "description": "Persistent memory across sessions (personal notes + user profile)",
        "tools": ["memory"],
        "includes": []
    },

    "context_engine": {
        "description": "Runtime tools exposed by the active context engine",
        "tools": [],
        "includes": []
    },
    
    "session_search": {
        "description": "Search and recall past conversations with summarization",
        "tools": ["session_search"],
        "includes": []
    },
    
    "clarify": {
        "description": "Ask the user clarifying questions (multiple-choice or open-ended)",
        "tools": ["clarify"],
        "includes": []
    },
    
    "code_execution": {
        "description": "Run Python scripts that call tools programmatically (reduces LLM round trips)",
        "tools": ["execute_code"],
        "includes": []
    },
    
    "delegation": {
        "description": "Spawn subagents with isolated context for complex subtasks",
        "tools": ["delegate_task"],
        "includes": []
    },

    # "honcho" toolset removed — Honcho is now a memory provider plugin.
    # Tools are injected via MemoryManager, not the toolset system.

    "homeassistant": {
        "description": "Home Assistant smart home control and monitoring",
        "tools": ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"],
        "includes": []
    },

    "kanban": {
        "description": (
            "Kanban multi-agent coordination — only active when the agent "
            "is spawned by the kanban dispatcher (HERMES_KANBAN_TASK env "
            "set). The dispatcher runs inside the gateway by default; see "
            "`kanban.dispatch_in_gateway` in config.yaml. Lets workers mark "
            "tasks done with structured handoffs, block for human input, "
            "heartbeat during long ops, comment on threads, and (for "
            "orchestrators) list, unblock, and fan out tasks."
        ),
        "tools": [
            "kanban_show", "kanban_list", "kanban_complete", "kanban_block",
            "kanban_heartbeat", "kanban_comment",
            "kanban_create", "kanban_link",
            "kanban_unblock",
        ],
        "includes": [],
    },

    "dra_content": {
        "description": (
            "DRA Content artifact publishing -- create, update, search, "
            "share and export websites (HTML plus its CSS, JavaScript, "
            "images and data files) through the DRA Content platform "
            "(content.ahfl.in). Standalone documents -- PDF, Word, Excel, "
            "PowerPoint -- go to Google Drive instead and are rejected here. "
            "See hermes-data/skills/productivity/dra-content-artifacts/SKILL.md."
        ),
        "tools": [
            "content_publish", "content_update", "content_find", "content_get",
            "content_list_versions", "content_share", "content_revoke",
            "content_access_list", "content_user_access", "content_archive",
            "content_export",
        ],
        "includes": [],
    },
    "whatsapp": {
        "description": (
            "NDR's personal WhatsApp -- read and search every chat/group, learn how he "
            "writes to a specific counterparty, and reply in his voice. Sending is a "
            "two-step draft/confirm flow: the bridge stores the draft and sends only the "
            "stored text, so NDR must approve the exact message first. See "
            "hermes-data/skills/communication/whatsapp-reply/SKILL.md."
        ),
        "tools": [
            "wa_list_chats", "wa_read_messages", "wa_search_messages", "wa_style_corpus",
            "wa_draft_reply", "wa_send_confirmed", "wa_draft_cancel", "wa_draft_status",
            "wa_health",
        ],
        "includes": [],
    },
    "next_action": {
        "description": (
            "Next Agent Action sheet CRUD -- schedule future-dated actions "
            "the cron agent executes automatically, park async clarifying "
            "questions when the cron agent can't proceed, and mark tasks "
            "complete/cancelled. See "
            "hermes-data/skills/productivity/next-agent-action/SKILL.md."
        ),
        "tools": [
            "next_action_add", "next_action_update", "next_action_complete",
            "next_action_cancel", "next_action_ask", "next_action_check_answers",
        ],
        "includes": [],
    },


    "discord": {
        "description": "Discord read and participate tools (fetch messages, search members, create threads)",
        "tools": ["discord"],
        "includes": [],
    },

    "discord_admin": {
        "description": "Discord server management (list channels/roles, pin messages, assign roles)",
        "tools": ["discord_admin"],
        "includes": [],
    },

    "yuanbao": {
        "description": "Yuanbao platform tools - group info, member queries, DM, stickers",
        "tools": [
            "yb_query_group_info",
            "yb_query_group_members",
            "yb_send_dm",
            "yb_search_sticker",
            "yb_send_sticker",
        ],
        "includes": []
    },

    "feishu_doc": {
        "description": "Read Feishu/Lark document content",
        "tools": ["feishu_doc_read"],
        "includes": []
    },

    "feishu_drive": {
        "description": "Feishu/Lark document comment operations (list, reply, add)",
        "tools": [
            "feishu_drive_list_comments", "feishu_drive_list_comment_replies",
            "feishu_drive_reply_comment", "feishu_drive_add_comment",
        ],
        "includes": []
    },

    "spotify": {
        "description": "Native Spotify playback, search, playlist, album, and library tools",
        "tools": [
            "spotify_playback", "spotify_devices", "spotify_queue", "spotify_search",
            "spotify_playlists", "spotify_albums", "spotify_library",
        ],
        "includes": []
    },


    # Scenario-specific toolsets
    
    "debugging": {
        "description": "Debugging and troubleshooting toolkit",
        "tools": ["terminal", "process"],
        "includes": ["web", "file"]  # For searching error messages and solutions, and file operations
    },
    
    "safe": {
        "description": "Safe toolkit without terminal access",
        "tools": [],
        "includes": ["web", "vision", "image_gen"]
    },
    
    # ==========================================================================
    # Full Hermes toolsets (CLI + messaging platforms)
    #
    # All platforms share the same core tools (including send_message,
    # which is gated on gateway running via its check_fn).
    # ==========================================================================

    "hermes-acp": {
        "description": "Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without messaging, audio, or clarify UI",
        "tools": _HERMES_CORE_TOOLS_NONINTERACTIVE,
        "includes": []
    },

    "hermes-api-server": {
        "description": "OpenAI-compatible API server (OpenWebUI / chat.ahfl.in) — exactly the same toolset as Telegram, including clarify and send_message",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-cli": {
        "description": "Full interactive CLI toolset - all default tools plus cronjob management",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-cron": {
        # Mirrors hermes-cli so cron's "default" toolset is the same set of
        # core tools users see interactively — then `hermes tools` filters
        # them down per the platform config. _DEFAULT_OFF_TOOLSETS (moa,
        # homeassistant) are excluded by _get_platform_tools() unless
        # the user explicitly enables them.
        "description": "Default cron toolset - same core tools as hermes-cli; gated by `hermes tools`",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-telegram": {
        "description": "Telegram bot toolset - full access for personal use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-discord": {
        "description": "Discord bot toolset - full access (terminal has safety checks via dangerous command approval)",
        "tools": _HERMES_CORE_TOOLS + [
            "discord",
            "discord_admin",
        ],
        "includes": []
    },
    
    "hermes-whatsapp": {
        "description": "WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-slack": {
        "description": "Slack bot toolset - full access for workspace use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-signal": {
        "description": "Signal bot toolset - encrypted messaging platform (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-bluebubbles": {
        "description": "BlueBubbles iMessage bot toolset - Apple iMessage via local BlueBubbles server",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-homeassistant": {
        "description": "Home Assistant bot toolset - smart home event monitoring and control",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-email": {
        "description": "Email bot toolset - interact with Hermes via email (IMAP/SMTP)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-mattermost": {
        "description": "Mattermost bot toolset - self-hosted team messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-matrix": {
        "description": "Matrix bot toolset - decentralized encrypted messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-dingtalk": {
        "description": "DingTalk bot toolset - enterprise messaging platform (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-feishu": {
        "description": "Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access)",
        "tools": _HERMES_CORE_TOOLS + [
            "feishu_doc_read",
            "feishu_drive_list_comments",
            "feishu_drive_list_comment_replies",
            "feishu_drive_reply_comment",
            "feishu_drive_add_comment",
        ],
        "includes": []
    },

    "hermes-weixin": {
        "description": "Weixin bot toolset - personal WeChat messaging via iLink (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-qqbot": {
        "description": "QQBot toolset - QQ messaging via Official Bot API v2 (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-wecom": {
        "description": "WeCom bot toolset - enterprise WeChat messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-wecom-callback": {
        "description": "WeCom callback toolset - enterprise self-built app messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-yuanbao": {
        "description": "Yuanbao Bot 元宝消息平台工具集 - 群信息、成员查询、私聊、贴纸表情",
        "tools": _HERMES_CORE_TOOLS + [
            "yb_query_group_info",
            "yb_query_group_members",
            "yb_send_dm",
            "yb_search_sticker",
            "yb_send_sticker",
        ],
        "module": "tools.yuanbao_tools",
        "includes": []
    },

    "hermes-sms": {
        "description": "SMS bot toolset - interact with Hermes via SMS (Twilio)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-webhook": {
        "description": "Webhook toolset - receive and process external webhook events",
        "tools": _HERMES_WEBHOOK_SAFE_TOOLS,
        "includes": []
    },

    "hermes-gateway": {
        "description": "Gateway toolset - union of all messaging platform tools",
        "tools": [],
        "includes": ["hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack", "hermes-signal", "hermes-bluebubbles", "hermes-homeassistant", "hermes-email", "hermes-sms", "hermes-mattermost", "hermes-matrix", "hermes-dingtalk", "hermes-feishu", "hermes-wecom", "hermes-wecom-callback", "hermes-weixin", "hermes-qqbot", "hermes-webhook", "hermes-yuanbao"]
    }
}



def get_toolset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a toolset definition by name.
    
    Args:
        name (str): Name of the toolset
        
    Returns:
        Dict: Toolset definition with description, tools, and includes
        None: If toolset not found
    """
    toolset = TOOLSETS.get(name)

    try:
        from tools.registry import registry
    except Exception:
        return toolset if toolset else None

    if toolset:
        merged_tools = sorted(
            set(toolset.get("tools", []))
            | set(registry.get_tool_names_for_toolset(name))
        )
        return {**toolset, "tools": merged_tools}

    registry_toolset = name
    description = f"Plugin toolset: {name}"
    alias_target = registry.get_toolset_alias_target(name)

    if name not in _get_plugin_toolset_names():
        registry_toolset = alias_target
        if not registry_toolset:
            return None
        description = f"MCP server '{name}' tools"
    else:
        reverse_aliases = {
            canonical: alias
            for alias, canonical in _get_registry_toolset_aliases().items()
            if alias not in TOOLSETS
        }
        alias = reverse_aliases.get(name)
        if alias:
            description = f"MCP server '{alias}' tools"

    return {
        "description": description,
        "tools": registry.get_tool_names_for_toolset(registry_toolset),
        "includes": [],
    }


def static_toolset_tools(name: str, visited: Optional[Set[str]] = None) -> Set[str]:
    """Resolve *name* using ONLY the static ``TOOLSETS`` definitions.

    ``resolve_toolset()`` merges in whatever the live tool registry has filed
    under the same toolset name, so its result grows every time a module
    registers a new operation or a plugin files itself under an existing
    toolset. That makes it the wrong basis for "does this composite intend
    this toolset?" questions, because the composites it gets compared against
    (``_HERMES_CORE_TOOLS`` and friends) are hand-maintained lists.

    Concretely: on 2026-09-13 ``gws_gmail_attachment_get`` was auto-registered
    into ``oauth`` without being added to ``_HERMES_CORE_TOOLS``. ``messaging``
    includes ``oauth``, so the strict subset test in
    ``hermes_cli.tools_config._get_platform_tools`` stopped believing the
    platform composite implied ``messaging`` -- and every gateway session lost
    ``whatsapp_link``, ``send_message``, ``send_oauth_url`` and every
    ``gws_*``/``kelsa_*`` tool. One unlisted name, 32 tools gone, for ten days.
    ``apify_run_actor`` did the same thing to ``web`` (taking ``web_extract``).

    Returns an empty set for registry-only toolsets (plugins, MCP servers) --
    they have no static definition to reason about, so callers fall back to
    the resolved view for those.
    """
    if visited is None:
        visited = set()
    if name in visited:
        return set()
    visited.add(name)

    toolset = TOOLSETS.get(name)
    if not toolset:
        return set()

    tools = set(toolset.get("tools", []))
    for included_name in toolset.get("includes", []):
        tools |= static_toolset_tools(included_name, visited)
    return tools


def resolve_toolset(name: str, visited: Set[str] = None) -> List[str]:
    """
    Recursively resolve a toolset to get all tool names.
    
    This function handles toolset composition by recursively resolving
    included toolsets and combining all tools.
    
    Args:
        name (str): Name of the toolset to resolve
        visited (Set[str]): Set of already visited toolsets (for cycle detection)
        
    Returns:
        List[str]: List of all tool names in the toolset
    """
    if visited is None:
        visited = set()
    
    # Special aliases that represent all tools across every toolset
    # This ensures future toolsets are automatically included without changes.
    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            # Use a fresh visited set per branch to avoid cross-branch contamination
            resolved = resolve_toolset(toolset_name, visited.copy())
            all_tools.update(resolved)
        return sorted(all_tools)

    # Check for cycles / already-resolved (diamond deps).
    # Silently return [] — either this is a diamond (not a bug, tools already
    # collected via another path) or a genuine cycle (safe to skip).
    if name in visited:
        return []

    visited.add(name)

    # Get toolset definition
    toolset = get_toolset(name)
    if not toolset:
        # Auto-generate a toolset for plugin platforms (hermes-<name>).
        # Gives them _HERMES_CORE_TOOLS plus any tools the plugin registered
        # into a toolset matching the platform name.
        if name.startswith("hermes-"):
            platform_name = name[len("hermes-"):]
            try:
                from gateway.platform_registry import platform_registry
                if platform_registry.is_registered(platform_name):
                    plugin_tools = set(_HERMES_CORE_TOOLS)
                    try:
                        from tools.registry import registry
                        plugin_tools.update(
                            e.name for e in registry._tools.values()
                            if e.toolset == platform_name
                        )
                    except Exception:
                        pass
                    return list(plugin_tools)
            except Exception:
                pass

        return []

    # Collect direct tools
    tools = set(toolset.get("tools", []))

    # Recursively resolve included toolsets, sharing the visited set across
    # sibling includes so diamond dependencies are only resolved once and
    # cycle warnings don't fire multiple times for the same cycle.
    for included_name in toolset.get("includes", []):
        included_tools = resolve_toolset(included_name, visited)
        tools.update(included_tools)
    
    return sorted(tools)


def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    """
    Resolve multiple toolsets and combine their tools.
    
    Args:
        toolset_names (List[str]): List of toolset names to resolve
        
    Returns:
        List[str]: Combined list of all tool names (deduplicated)
    """
    all_tools = set()
    
    for name in toolset_names:
        tools = resolve_toolset(name)
        all_tools.update(tools)
    
    return sorted(all_tools)


def _get_plugin_toolset_names() -> Set[str]:
    """Return toolset names registered by plugins (from the tool registry).

    These are toolsets that exist in the registry but not in the static
    ``TOOLSETS`` dict — i.e. they were added by plugins at load time.
    """
    try:
        from tools.registry import registry
        return {
            toolset_name
            for toolset_name in registry.get_registered_toolset_names()
            if toolset_name not in TOOLSETS
        }
    except Exception:
        return set()


def _get_registry_toolset_aliases() -> Dict[str, str]:
    """Return explicit toolset aliases registered in the live registry."""
    try:
        from tools.registry import registry
        return registry.get_registered_toolset_aliases()
    except Exception:
        return {}


def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """
    Get all available toolsets with their definitions.

    Includes both statically-defined toolsets and plugin-registered ones.
    
    Returns:
        Dict: All toolset definitions
    """
    result = dict(TOOLSETS)
    aliases = _get_registry_toolset_aliases()
    for ts_name in _get_plugin_toolset_names():
        display_name = ts_name
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                display_name = alias
                break
        if display_name in result:
            continue
        toolset = get_toolset(display_name)
        if toolset:
            result[display_name] = toolset
    return result


def get_toolset_names() -> List[str]:
    """
    Get names of all available toolsets (excluding aliases).

    Includes plugin-registered toolset names.
    
    Returns:
        List[str]: List of toolset names
    """
    names = set(TOOLSETS.keys())
    aliases = _get_registry_toolset_aliases()
    for ts_name in _get_plugin_toolset_names():
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                names.add(alias)
                break
        else:
            names.add(ts_name)
    return sorted(names)




def validate_toolset(name: str) -> bool:
    """
    Check if a toolset name is valid.
    
    Args:
        name (str): Toolset name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Accept special alias names for convenience
    if name in {"all", "*"}:
        return True
    if name in TOOLSETS:
        return True
    if name in _get_plugin_toolset_names():
        return True
    return name in _get_registry_toolset_aliases()


def create_custom_toolset(
    name: str,
    description: str,
    tools: List[str] = None,
    includes: List[str] = None
) -> None:
    """
    Create a custom toolset at runtime.
    
    Args:
        name (str): Name for the new toolset
        description (str): Description of the toolset
        tools (List[str]): Direct tools to include
        includes (List[str]): Other toolsets to include
    """
    TOOLSETS[name] = {
        "description": description,
        "tools": tools or [],
        "includes": includes or []
    }




def get_toolset_info(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a toolset including resolved tools.
    
    Args:
        name (str): Toolset name
        
    Returns:
        Dict: Detailed toolset information
    """
    toolset = get_toolset(name)
    if not toolset:
        return None
    
    resolved_tools = resolve_toolset(name)
    
    return {
        "name": name,
        "description": toolset["description"],
        "direct_tools": toolset["tools"],
        "includes": toolset["includes"],
        "resolved_tools": resolved_tools,
        "tool_count": len(resolved_tools),
        "is_composite": bool(toolset["includes"])
    }




if __name__ == "__main__":
    print("Toolsets System Demo")
    print("=" * 60)
    
    print("\nAvailable Toolsets:")
    print("-" * 40)
    for name, toolset in get_all_toolsets().items():
        info = get_toolset_info(name)
        composite = "[composite]" if info["is_composite"] else "[leaf]"
        print(f"  {composite} {name:20} - {toolset['description']}")
        print(f"     Tools: {len(info['resolved_tools'])} total")
    
    print("\nToolset Resolution Examples:")
    print("-" * 40)
    for name in ["web", "terminal", "safe", "debugging"]:
        tools = resolve_toolset(name)
        print(f"\n  {name}:")
        print(f"    Resolved to {len(tools)} tools: {', '.join(sorted(tools))}")
    
    print("\nMultiple Toolset Resolution:")
    print("-" * 40)
    combined = resolve_multiple_toolsets(["web", "vision", "terminal"])
    print("  Combining ['web', 'vision', 'terminal']:")
    print(f"    Result: {', '.join(sorted(combined))}")
    
    print("\nCustom Toolset Creation:")
    print("-" * 40)
    create_custom_toolset(
        name="my_custom",
        description="My custom toolset for specific tasks",
        tools=["web_search"],
        includes=["terminal", "vision"]
    )
    custom_info = get_toolset_info("my_custom")
    print("  Created 'my_custom' toolset:")
    print(f"    Description: {custom_info['description']}")
    print(f"    Resolved tools: {', '.join(custom_info['resolved_tools'])}")
