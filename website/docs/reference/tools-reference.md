---
sidebar_position: 3
title: "Built-in Tools Reference"
description: "Authoritative reference for Hermes built-in tools, grouped by toolset"
---

# Built-in Tools Reference

This page documents Hermes' built-in tools, grouped by toolset. Availability varies by platform, credentials, and enabled toolsets.

**Quick counts (current registry):** ~70 tools — 10 browser tools (core) + 2 CDP-gated browser tools, 4 file tools, 10 RL tools, 4 Home Assistant tools, 2 terminal tools, 2 web tools, 5 Feishu tools, 7 Spotify tools (registered by the bundled `spotify` plugin), 5 Yuanbao tools, 7 kanban tools (registered when the kanban dispatcher spawns the agent), 2 Discord tools, and a handful of standalone tools (`memory`, `clarify`, `delegate_task`, `execute_code`, `cronjob`, `session_search`, `skill_view`/`skill_manage`/`skills_list`, `text_to_speech`, `image_generate`, `video_generate`, `vision_analyze`, `video_analyze`, `mixture_of_agents`, `send_message`, `todo`, `computer_use`, `process`).

:::tip MCP Tools
In addition to built-in tools, Hermes can load tools dynamically from MCP servers. MCP tools appear with the prefix `mcp_<server>_` (e.g., `mcp_github_create_issue` for the `github` MCP server). See [MCP Integration](/docs/user-guide/features/mcp) for configuration.
:::

## `browser` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `browser_back` | Navigate back to the previous page in browser history. Requires browser_navigate to be called first. | — |
| `browser_click` | Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first. | — |
| `browser_console` | Get browser console output and JavaScript errors from the current page. Returns console.log/warn/error/info messages and uncaught JS exceptions. Use this to detect silent JavaScript errors, failed API calls, and application warnings. Requi… | — |
| `browser_get_images` | Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first. | — |
| `browser_navigate` | Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). Use browser tools when you need… | — |
| `browser_press` | Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first. | — |
| `browser_scroll` | Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first. | — |
| `browser_snapshot` | Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: comp… | — |
| `browser_type` | Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first. | — |
| `browser_vision` | Take a screenshot of the current page and analyze it with vision AI. Use this when you need to visually understand what's on the page - especially useful for CAPTCHAs, visual verification challenges, complex layouts, or when the text snaps… | — |

## `browser` toolset (CDP-gated tools)

These two tools live in the `browser` toolset but only register when a Chrome DevTools Protocol endpoint is reachable at session start — via `/browser connect`, `browser.cdp_url` config, a Browserbase session, or Camofox.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `browser_cdp` | Send a raw Chrome DevTools Protocol command. Escape hatch for browser operations not covered by the higher-level `browser_*` tools. See https://chromedevtools.github.io/devtools-protocol/ | CDP endpoint |
| `browser_dialog` | Respond to a native JavaScript dialog (alert / confirm / prompt / beforeunload). Call `browser_snapshot` first — pending dialogs appear in its `pending_dialogs` field. Then call `browser_dialog(action='accept'\|'dismiss')`. | CDP endpoint |

## `clarify` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `clarify` | Ask the user a question when you need clarification, feedback, or a decision before proceeding. Supports two modes: 1. **Multiple choice** — provide up to 4 choices. The user picks one or types their own answer via a 5th 'Other' option. 2.… | — |

## `code_execution` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `execute_code` | Run a Python script that can call Hermes tools programmatically. Use this when you need 3+ tool calls with processing logic between them, need to filter/reduce large tool outputs before they enter your context, need conditional branching (… | — |

## `cronjob` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `cronjob` | Unified scheduled-task manager. Use `action="create"`, `"list"`, `"update"`, `"pause"`, `"resume"`, `"run"`, or `"remove"` to manage jobs. Supports skill-backed jobs with one or more attached skills, and `skills=[]` on update clears attached skills. Cron runs happen in fresh sessions with no current-chat context. | — |

## `delegation` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `delegate_task` | Spawn one or more subagents to work on tasks in isolated contexts. Each subagent gets its own conversation, terminal session, and toolset. Only the final summary is returned -- intermediate tool results never enter your context window. TWO… | — |

## `feishu_doc` toolset

Scoped to the Feishu document-comment intelligent-reply handler (`gateway/platforms/feishu_comment.py`). Not exposed on `hermes-cli` or the regular Feishu chat adapter.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `feishu_doc_read` | Read the full text content of a Feishu/Lark document (Docx, Doc, or Sheet) given its file_type and token. | Feishu app credentials |

## `feishu_drive` toolset

Scoped to the Feishu document-comment handler. Drives comment read/write operations on drive files.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `feishu_drive_add_comment` | Add a top-level comment on a Feishu/Lark document or file. | Feishu app credentials |
| `feishu_drive_list_comments` | List whole-document comments on a Feishu/Lark file, most recent first. | Feishu app credentials |
| `feishu_drive_list_comment_replies` | List replies on a specific Feishu comment thread (whole-doc or local-selection). | Feishu app credentials |
| `feishu_drive_reply_comment` | Post a reply on a Feishu comment thread, with optional `@`-mention. | Feishu app credentials |

## `file` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `patch` | Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. Returns a unified diff. Auto-runs syntax checks after editing… | — |
| `read_file` | Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM\|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. NOTE: Cannot read images o… | — |
| `search_files` | Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents. Content search (target='content'): Regex search inside files. Output modes: full matches with line… | — |
| `write_file` | Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. | — |

## `homeassistant` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `ha_call_service` | Call a Home Assistant service to control a device. Use ha_list_services to discover available services and their parameters for each domain. | — |
| `ha_get_state` | Get the detailed state of a single Home Assistant entity, including all attributes (brightness, color, temperature setpoint, sensor readings, etc.). | — |
| `ha_list_entities` | List Home Assistant entities. Optionally filter by domain (light, switch, climate, sensor, binary_sensor, cover, fan, etc.) or by area name (living room, kitchen, bedroom, etc.). | — |
| `ha_list_services` | List available Home Assistant services (actions) for device control. Shows what actions can be performed on each device type and what parameters they accept. Use this to discover how to control devices found via ha_list_entities. | — |

## `computer_use` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `computer_use` | Background macOS desktop control via cua-driver — screenshots (SOM / vision / AX), click / drag / scroll / type / key / wait, list_apps, focus_app. Does NOT steal the user's cursor or keyboard focus. Works with any tool-capable model. macOS only. | `cua-driver` on `$PATH` (install via `hermes tools`). |


:::note
**Honcho tools** (`honcho_profile`, `honcho_search`, `honcho_context`, `honcho_reasoning`, `honcho_conclude`) are no longer built-in. They are available via the Honcho memory provider plugin at `plugins/memory/honcho/`. See [Memory Providers](../user-guide/features/memory-providers.md) for installation and usage.
:::

## `image_gen` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `image_generate` | Generate high-quality images from text prompts using FAL.ai. The underlying model is user-configured (default: FLUX 2 Klein 9B, sub-1s generation) and is not selectable by the agent. Returns a single image URL. Display it using… | FAL_KEY |

## `kanban` toolset

Registered only when the agent is spawned by the kanban dispatcher (`HERMES_KANBAN_TASK` env set). Lets workers mark tasks done with structured handoffs, block for human input, heartbeat during long ops, comment on threads, and (for orchestrators) fan out into child tasks. See [Kanban Multi-Agent](/docs/user-guide/features/kanban) for the full workflow.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `kanban_show` | Show the active kanban task assigned to this worker (title, description, comments, dependencies). | `HERMES_KANBAN_TASK` |
| `kanban_complete` | Mark the current task done with a structured handoff payload (results, artifacts, follow-ups). | `HERMES_KANBAN_TASK` |
| `kanban_block` | Block the current task on a question for the user — the dispatcher pauses, surfaces the question, and resumes once a human replies. | `HERMES_KANBAN_TASK` |
| `kanban_heartbeat` | Send a progress heartbeat during a long-running operation so the dispatcher knows the worker is still alive. | `HERMES_KANBAN_TASK` |
| `kanban_comment` | Add a comment to the task thread without changing its state — useful for surfacing intermediate findings. | `HERMES_KANBAN_TASK` |
| `kanban_create` | (Orchestrator only) Fan out child tasks from the current task. | `HERMES_KANBAN_TASK` + orchestrator role |
| `kanban_link` | (Orchestrator only) Link related tasks together (blocks/blocked-by/related). | `HERMES_KANBAN_TASK` + orchestrator role |

## `memory` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `memory` | Save important information to persistent memory that survives across sessions. Your memory appears in your system prompt at session start -- it's how you remember things about the user and your environment between conversations. WHEN TO SA… | — |

## `messaging` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `send_message` | Send a message to a connected messaging platform, or list available targets. IMPORTANT: When the user asks to send to a specific channel or person (not just a bare platform name), call send_message(action='list') FIRST to see available tar… | — |

## `moa` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `mixture_of_agents` | Route a hard problem through multiple frontier LLMs collaboratively. Makes 5 API calls (4 reference models + 1 aggregator) with maximum reasoning effort — use sparingly for genuinely difficult problems. Best for: complex math, advanced alg… | OPENROUTER_API_KEY |

## `session_search` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `session_search` | Search your long-term memory of past conversations. This is your recall -- every past session is searchable, and this tool summarizes what happened. USE THIS PROACTIVELY when: - The user says 'we did this before', 'remember when', 'last ti… | — |

## `skills` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `skill_manage` | Manage skills (create, update, delete). Skills are your procedural memory — reusable approaches for recurring task types. New skills go to ~/.hermes/skills/; existing skills can be modified wherever they live. Actions: create (full SKILL.m… | — |
| `skill_view` | Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a… | — |
| `skills_list` | List available skills (name + description). Use skill_view(name) to load full content. | — |

## `terminal` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `process` | Manage background processes started with terminal(background=true). Actions: 'list' (show all), 'poll' (check status + new output), 'log' (full output with pagination), 'wait' (block until done or timeout), 'kill' (terminate), 'write' (sen… | — |
| `terminal` | Execute shell commands on a Linux environment. Filesystem persists between calls. Set `background=true` for long-running servers. Set `notify_on_complete=true` (with `background=true`) to get an automatic notification when the process finishes — no polling needed. Do NOT use cat/head/tail — use read_file. Do NOT use grep/rg/find — use search_files. | — |

## `todo` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `todo` | Manage your task list for the current session. Use for complex tasks with 3+ steps or when the user provides multiple tasks. Call with no parameters to read the current list. Writing: - Provide 'todos' array to create/update items - merge=… | — |

## `vision` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `vision_analyze` | Analyze images using AI vision. Provides a comprehensive description and answers a specific question about the image content. | — |

## `video` toolset

Opt-in toolset (not loaded in the default `hermes-cli` set). Add via `--toolsets video` or include `video` in your `toolsets:` config.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `video_analyze` | Analyze video content from a URL or file path — captions, scene breakdowns, key timestamps, and visual descriptions. | — |

## `video_gen` toolset

Opt-in toolset (not loaded in the default `hermes-cli` set). Add via `--toolsets video_gen` or enable it in `hermes tools` → Video Generation, which also walks you through picking a backend.

Backends ship as plugins under `plugins/video_gen/<name>/`:

- **xAI Grok-Imagine** — text-to-video and image-to-video (SuperGrok OAuth or `XAI_API_KEY`).
- **FAL.ai** — Veo 3.1, Pixverse v6, Kling O3 (requires `FAL_KEY`).

The single `video_generate` tool covers both modalities — pass `image_url` to animate a still, omit it to generate from text alone. The active backend auto-routes to the right endpoint. The tool's description is rebuilt at session start to reflect the active backend's actual capabilities (modalities, aspect ratios, resolutions, duration range, max reference images, audio support). See [Video Generation Provider Plugins](/docs/developer-guide/video-gen-provider-plugin) for backend authoring.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `video_generate` | Generate a video from a text prompt (text-to-video) or animate a still image (image-to-video) using the user's configured video generation backend. Pass `image_url` to animate that image; omit it to generate from text alone. The backend auto-routes to the right endpoint. Returns either an HTTP URL or an absolute file path in the `video` field. | Active `video_gen` plugin + its credential (e.g. `XAI_API_KEY`, `FAL_KEY`) |

## `web` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `web_search` | Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. Accepts an optional `limit` (1-100, default 5). The query is passed through to the configured backend, so operators such as `site:domain`, `filetype:pdf`, `intitle:word`, `-term`, and `"exact phrase"` may work when the backend supports them. | EXA_API_KEY or PARALLEL_API_KEY or FIRECRAWL_API_KEY or TAVILY_API_KEY |
| `web_extract` | Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs — pass the PDF link directly and it converts to markdown text. Pages under 5000 chars return full markdown; larger pages are LLM-summarized. | EXA_API_KEY or PARALLEL_API_KEY or FIRECRAWL_API_KEY or TAVILY_API_KEY |

## `x_search` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `x_search` | Search X (Twitter) posts, profiles, and threads using xAI's built-in `x_search` Responses tool. Use this for current discussion, reactions, or claims on X rather than general web pages. Off by default — opt in via `hermes tools` → 🐦 X (Twitter) Search. Schema is only registered when xAI credentials are configured (check_fn-gated). | XAI_API_KEY **or** xAI Grok OAuth (SuperGrok Subscription) login |

## `tts` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `text_to_speech` | Convert text to speech audio. Returns a MEDIA: path that the platform delivers as a voice message. On Telegram it plays as a voice bubble, on Discord/WhatsApp as an audio attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider… | — |

## `discord` toolset

Registered on the `hermes-discord` platform toolset (gateway only). Uses the same bot token as the messaging adapter.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `discord` | Read and participate in a Discord server. Actions include `search_members`, `fetch_messages`, `send_message`, `react`, `fetch_channel`, `list_channels`, and more. | `DISCORD_BOT_TOKEN` |

## `discord_admin` toolset

Registered on the `hermes-discord` platform toolset. Moderation actions require the bot to hold the matching Discord permissions.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `discord_admin` | Manage a Discord server via the REST API: list guilds/channels/roles, create/edit/delete channels, manage role grants, timeouts, kicks, and bans. | `DISCORD_BOT_TOKEN` + bot permissions |

## `spotify` toolset

Registered by the bundled `spotify` plugin. Requires an OAuth token — run `hermes spotify setup` once to authorize.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `spotify_playback` | Control Spotify playback, inspect the active playback state, or fetch recently played tracks. | Spotify OAuth |
| `spotify_devices` | List Spotify Connect devices or transfer playback to a different device. | Spotify OAuth |
| `spotify_queue` | Inspect the user's Spotify queue or add an item to it. | Spotify OAuth |
| `spotify_search` | Search the Spotify catalog for tracks, albums, artists, playlists, shows, or episodes. | Spotify OAuth |
| `spotify_playlists` | List, inspect, create, update, and modify Spotify playlists. | Spotify OAuth |
| `spotify_albums` | Fetch Spotify album metadata or album tracks. | Spotify OAuth |
| `spotify_library` | List, save, or remove the user's saved Spotify tracks or albums. | Spotify OAuth |

## `hermes-yuanbao` toolset

Registered only on the `hermes-yuanbao` platform toolset. Yuanbao is Tencent's chat app; these tools drive its DM/group/sticker APIs.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `yb_query_group_info` | Query basic info about a group (called "派/Pai" in the app): name, owner, member count. | Yuanbao credentials |
| `yb_query_group_members` | Query members of a group (for `@`-mentions, finding a user by name, listing bots). | Yuanbao credentials |
| `yb_send_dm` | Send a private/direct message to a user in a group, with optional media files. | Yuanbao credentials |
| `yb_search_sticker` | Search the built-in Yuanbao sticker (TIM face) catalogue by keyword. | Yuanbao credentials |
| `yb_send_sticker` | Send a built-in sticker to the current Yuanbao chat. | Yuanbao credentials |


