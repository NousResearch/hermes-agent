---
sidebar_position: 3
title: "Built-in Tools Reference"
description: "Authoritative reference for Hermes built-in tools, grouped by toolset"
---

# Built-in Tools Reference

This page documents Hermes' built-in tools, grouped by toolset. Availability varies by platform, credentials, and enabled toolsets.

**Quick counts (current registry):** ~86 tools — 10 browser tools (core) + 2 CDP-gated browser tools, 4 file tools, 4 Home Assistant tools, 2 terminal tools (`terminal`, `process`), 12 desktop-GUI tools (`read_terminal`, `close_terminal`, `open_preview`, `close_preview`, `read_preview`, `drive_preview`, `annotate_preview`, `read_window_below`, `focus_pane`, `react_to_message`, `tour`, `tip` — desktop-app sessions only), 2 web tools, 5 Feishu tools, 7 Spotify tools (registered by the bundled `spotify` plugin), 5 Yuanbao tools, 12 kanban tools (registered when the kanban dispatcher spawns the agent), 3 project tools (desktop/GUI sessions), 2 Discord tools, 3 video tools (`video_generate`, `xai_video_edit`, `xai_video_extend`), and a handful of standalone tools (`memory`, `clarify`, `delegate_task`, `execute_code`, `cronjob`, `session_search`, `skill_view`/`skill_manage`/`skills_list`, `text_to_speech`, `image_generate`, `vision_analyze`, `video_analyze`, `todo`, `computer_use`, `x_search`).

:::tip MCP Tools
In addition to built-in tools, Hermes can load tools dynamically from MCP servers. MCP tools appear with the prefix `mcp__<server>__` (e.g., `mcp__github__create_issue` for the `github` MCP server). See [MCP Integration](/user-guide/features/mcp) for configuration.
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
| `browser_vision` | Take a screenshot of the current page so you can inspect it visually. Use this when you need to understand what the page looks like — especially for CAPTCHAs, visual verification challenges, complex layouts, or cases where the text snapshot misses important visual information. On native-vision models the screenshot is attached directly; otherwise falls back to an auxiliary vision mo… | — |

## `browser` toolset (CDP-gated tools)

These two tools live in the `browser` toolset but only register when a Chrome DevTools Protocol endpoint is reachable at session start — via `/browser connect`, `browser.cdp_url` config, a Browserbase session, or Camofox.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `browser_cdp` | Send a raw Chrome DevTools Protocol command. Escape hatch for browser operations not covered by the higher-level `browser_*` tools. See https://chromedevtools.github.io/devtools-protocol/ | CDP endpoint |
| `browser_dialog` | Respond to a native JavaScript dialog (alert / confirm / prompt / beforeunload). Call `browser_snapshot` first — pending dialogs appear in its `pending_dialogs` field. Then call `browser_dialog(action='accept'\|'dismiss')`. | CDP endpoint |

## `clarify` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `clarify` | Ask the user a question when you need clarification, feedback, or a decision before proceeding. Supports three modes: 1. **Single-select multiple choice** — up to 4 choices; the user picks one or types their own answer via a 5th 'Other' option. 2. **Multi-select multiple choice** — `multi_select=true` renders checkboxes and returns a list of selected choices. 3. **Open-ended** — no choices; the user types a free-form response. Choices are ordered best-first, so the first one is labelled `(Recommended)` on every surface and is the default highlight; the label is presentation only and is stripped from the answer the agent reads. On the classic CLI multi-select uses Space-to-toggle checkboxes; on messaging platforms without native checkbox UIs the user replies with comma/space-separated numbers (e.g. "1, 3") or the option text. | — |

### Asking multiple questions at once

The `clarify` tool also accepts a `questions` array (2–5 independent questions, each with its own `choices` and `multi_select`) so the agent can batch several clarification needs into a single prompt instead of asking sequentially. The result is a `responses` array in the same order, with each question's `id` (when supplied) echoed back.

Per-surface behavior:

- **Desktop** shows every question on one card. Picks and typed answers stage locally, and one **Confirm and continue** button (enabled once every question has an answer) submits the whole batch. Staged answers stay editable until that confirm. Skip cancels the whole batch.
- **TUI and CLI** show a compact status list (`✓` answered / `▸` active / `·` pending) with only the active question's choices expanded. Enter locks the active answer and jumps to the next unanswered question; Tab moves between questions to answer in any order; Esc cancels the batch.
- **Messaging platforms** (Telegram, Discord, …) fall back to asking the questions one at a time through the existing single-question prompt. If the user stops responding, the remaining questions are not sent.

If the prompt times out part-way, answers the user already locked are kept: the tool result carries them plus `"timed_out": true`, with the unanswered entries left blank, so the agent can distinguish a deliberate skip from an absent user.

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
| `delegate_task` | Spawn subagents in isolated contexts; each gets its own conversation, terminal session, and toolset, and only its final summary returns to you. Provide 'goal' for a single task or 'tasks' for a parallel batch (limits and nesting rules… | — |

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
| `read_file` | Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM\|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are truncated on a line boundary and return a next_offset. Jupyter notebooks (.ipynb), Word documents (.docx), and Excel workbooks (.xlsx) a… | — |
| `search_files` | Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents. Content search (target='content'): Regex search inside files. Output modes: full matches with line… | — |
| `write_file` | Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by the write are surfaced. | — |

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
| `computer_use` | Background desktop control via cua-driver — screenshots (SOM / vision / AX), click / drag / scroll / type / key / wait, list_apps, focus_app. Does NOT steal the user's cursor or keyboard focus. Works with any tool-capable model. macOS, Windows, and Linux. | `cua-driver` on `$PATH` (install via `hermes tools`). |


:::note
**Honcho tools** (`honcho_profile`, `honcho_search`, `honcho_context`, `honcho_reasoning`, `honcho_conclude`) are no longer built-in. They are available via the Honcho memory provider plugin at `plugins/memory/honcho/`. See [Memory Providers](../user-guide/features/memory-providers.md) for installation and usage.
:::

## `image_gen` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `image_generate` | Generate images from text prompts (text-to-image) or edit/transform an existing image (image-to-image) via the user-configured backend (FAL.ai, OpenAI, OpenAI Codex auth, xAI, Krea). Pass `image_url` to edit an image and `reference_image_urls` for style references; omit both for text-to-image. The model is user-configured and not selectable by the agent. Returns a single image URL or local path. | FAL_KEY / OPENAI_API_KEY / Codex OAuth / xAI OAuth / KREA_API_KEY |

## `kanban` toolset

Registered when the agent is either (a) spawned by the kanban dispatcher (`HERMES_KANBAN_TASK` env set) or (b) running in a profile that explicitly enables the `kanban` toolset. Task-scoped workers use lifecycle tools for their assigned task; orchestrator profiles additionally get board-routing tools like `kanban_list` and `kanban_unblock`. See [Kanban Multi-Agent](/user-guide/features/kanban) for the full workflow.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `kanban_show` | Show the active kanban task assigned to this worker (title, description, comments, dependencies). | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_list` | List board tasks with filters. Orchestrator-only; hidden from dispatcher-spawned task workers. | profile with `kanban` toolset |
| `kanban_complete` | Mark the current task done with a structured handoff payload (results, artifacts, follow-ups). | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_block` | Block the current task on a question for the user — the dispatcher pauses, surfaces the question, and resumes once a human replies. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_request_review` | Hand the implementation to a reviewer with `summary`, optional structured `metadata`, and an optional reviewer profile. Moves the same task to `review`; it is not a block and does not affect block-loop accounting. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_request_changes` | Reviewer verdict for an actively claimed review run. Closes the review run, reapplies parent gating, and routes the task back to the original implementer without using a block. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_heartbeat` | Send a progress heartbeat during a long-running operation so the dispatcher knows the worker is still alive. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_comment` | Add a comment to the task thread without changing its state — useful for surfacing intermediate findings. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_create` | Fan out child tasks from the current task. Used by orchestrators and follow-up-spawning workers. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_link` | Link tasks with a parent → child dependency edge. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_unblock` | Move a blocked task to `ready` when all parents are done, or `todo` while any parent remains open. Orchestrator-only; hidden from dispatcher-spawned task workers. | profile with `kanban` toolset |
| `kanban_attach` | Attach a file to a task by passing its bytes inline (base64). Stored as a real attachment under the task's attachments dir, capped at 25 MB. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_attach_url` | Attach a file to a task by URL — Hermes downloads it server-side and stores it as a real attachment (capped at 25 MB). Only http/https URLs. | `HERMES_KANBAN_TASK` or `kanban` toolset |
| `kanban_attachments` | List the files attached to a task: id, filename, content_type, size, uploader, and the absolute on-disk path. | `HERMES_KANBAN_TASK` or `kanban` toolset |

## `project` toolset

Tools for driving desktop [Projects](../user-guide/cli.md) — named, multi-folder workspaces. Registered when the `project` toolset is enabled (primarily the desktop app / dashboard surfaces).

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `project_create` | Create a desktop Project (a named workspace) and switch this chat into it. Pass `path` to anchor it to a repo/folder. | — |
| `project_list` | List the desktop Projects and which one is active. | — |
| `project_switch` | Switch this chat into an existing Project (by name, slug, or id); moves the session workspace to the project's primary folder. | — |

## `memory` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `memory` | Save important information to persistent memory that survives across sessions. Your memory appears in your system prompt at session start -- it's how you remember things about the user and your environment between conversations. WHEN TO SA… | — |

## `session_search` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `session_search` | Search past sessions stored in the local session DB, or scroll inside one. FTS5-backed retrieval; returns actual messages from the DB (no LLM calls). Four shapes: discovery (pass `query`), scroll (pass `session_id` + `around_message_id`), read (pass `session_id` only), browse (no args). | — |

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

## `desktop_ui` toolset

Enabled for sessions whose source is the Hermes desktop app, on any backend it
is connected to (local, SSH, URL, or Hermes Cloud). Absent from CLI, TUI,
messaging, and cron sessions.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `read_terminal` | Read what's currently shown in the in-app terminal pane of the Hermes desktop GUI (the embedded shell beside this chat). | — |
| `close_terminal` | Close the read-only terminal tab for a background process in the Hermes desktop GUI. Does NOT kill the process — only drops the tab/view; use process(action='kill') to stop it. | — |
| `open_preview` | Open a web URL, localhost dev-server URL, or file path in the preview pane beside the chat in the Hermes desktop app. | — |
| `close_preview` | Close the preview pane beside the chat, or one tab inside it. Omit `url` to close the whole pane; pass a URL or file path to close that tab. | — |
| `read_preview` | Read what's currently shown in the preview pane of the Hermes desktop GUI — the in-app Browser's page text (URL + title + rendered text, pageable with `start`/`count`), or a file/artifact tab's identity. | — |
| `drive_preview` | Interact with the page open in the in-app browser: `elements` inventories what's clickable and typable (each with a ref that names it, like `btn-sign-in` or `inp-email`, plus role, label, and value), then `click`, `hover`, `type`, `scroll`, and `press` act on a ref, and `back`/`forward`/`reload` drive the pane's history. The pointer and keyboard are real input, so hover menus open. A ref lasts until the page navigates, including across a re-render that rebuilds the element, so after the first inventory every action answers with just a delta — what was added, removed, changed, or rebound — instead of the whole page again. | — |
| `annotate_preview` | Outline an element in the in-app browser and leave the mark up until it's removed — the deliberate counterpart to the transient cues `drive_preview` draws as it works. `add` marks a ref with an optional short label, `remove` takes one down, `clear` takes them all. Marks follow their element and vanish with it, so a navigation clears them. | — |
| `read_window_below` | Identify the OS window directly underneath the Hermes desktop window — app name, title, bounds (metadata only, never pixels). On macOS, other apps' titles appear only when Screen Recording is already granted; the tool never prompts for it. | — |
| `focus_pane` | Reveal and focus a pane in the Hermes desktop app (chat, files, terminal, review, sessions). | — |
| `react_to_message` | React to a message with a single emoji, iMessage-tapback style. Opt-in via Settings → Appearance (`display.message_reactions`). | — |
| `tour` | Give a live guided tour: dim the screen, highlight an element, and attach a narrated popover (driver.js). Works on the Hermes app's own UI and on any page open in the preview pane; `targets` discovers what's on screen, `show` narrates step-by-step, `start` hands the user Next/Prev controls. | — |
| `tip` | Point at one element with a small accent bubble and an arrow — the quiet sibling of `tour`, with no dimming, no spotlight, and no Next/Prev. Same `data-tour` handles and the same `tour(action='targets')` discovery call. | — |

### Tours

The `tour` tool discovers its own targets — call `action='targets'` and it returns every addressable element on screen with a selector, a label, and a `stable` flag. Stable selectors key off identity (`data-tour`, `id`, `data-testid`, `aria-label`) and survive a re-render; positional `nth-child` paths don't, so stable ones sort first and should be preferred.

To give an element a durable handle of your own, mark it up:

```html
<div data-tour="composer">…</div>
```

Handles are applied at the **primitive**, not the call site, so one edit names every instance. The ones that already exist:

| Handle | What it names |
|---|---|
| `overlay-nav` | the left nav of any route overlay (settings, cron, profiles, agents) |
| `nav-<id>` | one row in that nav — `nav-models`, `nav-appearance`, … |
| `field-<schemaKey>` | one settings row, by its config key — `field-model`, `field-provider`, … |
| `page-tabs` | the filter tabs on any `PageSearchShell` page (artifacts, skills, …) |
| `artifact-card` | an artifact card in the grid |

When adding a surface, tag its shared primitive the same way rather than tagging screens one by one — that keeps the tour vocabulary small and stops selectors from rotting.

The same engine backs curated (non-agent) tours in the desktop app, so a feature can ship its own walkthrough:

```ts
import { startTour, showTourStep, stopTour } from '@/lib/tour'

startTour([
  { selector: '[data-tour="composer"]', title: 'Composer', text: 'Type here.' },
  { selector: '[data-tour="files"]', title: 'Files', text: 'Browse your project.' }
])
```

A step can also move the app to where its target lives, and the tour puts things back when it ends:

```ts
startTour([
  { navigate: '/artifacts', selector: '[data-tour="page-tabs"]', title: 'Filters', text: '…' },
  { pane: 'sessions', selector: '[data-slot="sidebar"]', title: 'Sessions', text: '…' }
])
```

`navigate` takes a route path and `pane` a desktop pane name. Both run as the step is entered, targets that mount late are waited for, and closing the tour — by any route, including Esc — returns to wherever it started.

Pass `'preview'` as the second argument to run against the page in the preview pane instead of the app.

### Tips

A tip is a tour step without the production: one bubble, one arrow, no scrim and
nothing to page through. It's the right weight for a sentence that would be
clearer with a finger on the thing it's about — "the model name is a button" —
where dimming the whole app would not be.

The `tip` tool takes the same selectors `tour(action='targets')` reports, so
discovery is one call for both, and the durable `data-tour` handles above name
targets for either. One tip is on screen at a time; a new one replaces the last.

The app can also show its own, walking a built-in catalog of app features in
order, paced like a game's loading-screen tips rather than a notification: a few
minutes into a launch at the earliest, then at most one every six hours, and
only at a genuinely idle moment. A tip from Hermes shares that cooldown, so it
also buys the user six hours of quiet from the rotation. Closing a rotation tip
with the ✕ retires that tip for good, and the settings row brings them back.

Both tips and tours are on by default and switched off in Settings → Appearance
(`display.in_app_tips`, `display.in_app_tours`). Off covers Hermes as well as
the app: the switch reaches the connected gateway's config and the tool leaves
the model's schema, so the agent is never told about a surface it isn't allowed
to use. Like every schema change, that lands on the next session — a running
conversation keeps the toolset it started with, and the app declines the call in
the meantime.

## `todo` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `todo` | Manage your task list for the current session. Use for complex tasks with 3+ steps or when the user provides multiple tasks. Call with no parameters to read the current list. Items may nest: an item's optional `parent` field points at another item's id, making it a subtask — surfaces render the tree indented. | — |

## `vision` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `vision_analyze` | Analyze images using AI vision. On vision-capable main models, returns the raw image pixels as a multimodal tool result so the model sees them natively on its next turn. On text-only main models, falls back to an auxiliary vision model that describes the image and returns the description as text. Tool signature is identical either way. | — |

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

The single `video_generate` tool covers both modalities — pass `image_url` to animate a still, omit it to generate from text alone. The active backend auto-routes to the right endpoint. The tool's description is rebuilt at session start to reflect the active backend's actual capabilities (modalities, aspect ratios, resolutions, duration range, max reference images, audio support). See [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) for backend authoring.

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `video_generate` | Generate a video from a text prompt (text-to-video) or animate a still image (image-to-video) using the user's configured video generation backend. Pass `image_url` to animate that image; omit it to generate from text alone. The backend auto-routes to the right endpoint. Returns either an HTTP URL or an absolute file path in the `video` field. | Active `video_gen` plugin + its credential (e.g. `XAI_API_KEY`, `FAL_KEY`) |
| `xai_video_edit` | Edit an existing video with xAI Imagine. Provider-specific (separate from `video_generate`). `video_url` must be the public HTTPS MP4 URL from a prior Imagine result. | xAI Imagine credentials (SuperGrok OAuth or `XAI_API_KEY`) |
| `xai_video_extend` | Extend an existing video with xAI Imagine. Provider-specific (separate from `video_generate`). `video_url` must be the public HTTPS MP4 URL from a prior Imagine result. | xAI Imagine credentials (SuperGrok OAuth or `XAI_API_KEY`) |

## `web` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `web_search` | Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. Accepts an optional `limit` (1-100, default 5). The query is passed through to the configured backend, so operators such as `site:domain`, `filetype:pdf`, `intitle:word`, `-term`, and `"exact phrase"` may work when the backend supports them. | EXA_API_KEY or PARALLEL_API_KEY or FIRECRAWL_API_KEY or TAVILY_API_KEY or KEENABLE_API_KEY |
| `web_extract` | Extract content from web page URLs. Returns clean page content in markdown/text (no LLM summarization — fast). Also works with PDF URLs (arxiv papers, documents) — pass the PDF link directly. Pages within the char budget (default 15000) return whole; larger pages return a head+tail window with a footer pointing at the full text saved on disk. Max 5 URLs per call. | EXA_API_KEY or PARALLEL_API_KEY or FIRECRAWL_API_KEY or TAVILY_API_KEY or KEENABLE_API_KEY |

## `x_search` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `x_search` | Search X (Twitter) posts, profiles, and threads using xAI's built-in `x_search` Responses tool. Read-only public X discovery for current discussion, reactions, or claims on public X (not general web pages). Does not post, reply, like, DM, upload media, delete, or inspect the authenticated X account — those need a separate authenticated X API surface (e.g. the `xurl` skill). Off by default — opt in via `hermes tools` → 🐦 X (Twitter) Search. Schema is only registered when xAI credentials are configured (check_fn-gated). | XAI_API_KEY **or** xAI Grok OAuth (SuperGrok / Premium+) login |

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

Registered by the bundled `spotify` plugin. Requires an OAuth token — run `hermes auth spotify` once to authorize.

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


