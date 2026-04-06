---
sidebar_position: 3
title: "Built-in Tools Reference"
description: "Authoritative reference for Hermes built-in tools, grouped by toolset"
---

# Built-in Tools Reference

This page documents all 47 built-in tools in the Hermes tool registry, grouped by toolset. Availability varies by platform, credentials, and enabled toolsets.

**Quick counts:** 11 browser tools, 4 file tools, 10 RL tools, 4 Home Assistant tools, 2 terminal tools, 2 web tools, and 14 standalone tools across other toolsets.

:::tip MCP Tools
In addition to built-in tools, Hermes can load tools dynamically from MCP servers. MCP tools appear with a server-name prefix (e.g., `github_create_issue` for the `github` MCP server). See [MCP Integration](/docs/user-guide/features/mcp) for configuration.
:::

## `browser` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `browser_back` | Navigate back to the previous page in browser history. Requires browser_navigate to be called first. | — |
| `browser_click` | Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first. | — |
| `browser_close` | Close the browser session and release resources. Call this when done with browser tasks to free up Browserbase session quota. | — |
| `browser_console` | Get browser console output and JavaScript errors from the current page. Returns console.log/warn/error/info messages and uncaught JS exceptions. Use this to detect silent JavaScript errors, failed API calls, and application warnings. Requi… | — |
| `browser_get_images` | Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first. | — |
| `browser_navigate` | Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). Use browser tools when you need… | — |
| `browser_press` | Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first. | — |
| `browser_scroll` | Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first. | — |
| `browser_snapshot` | Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: comp… | — |
| `browser_type` | Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first. | — |
| `browser_vision` | Take a screenshot of the current page and analyze it with vision AI. Use this when you need to visually understand what's on the page - especially useful for CAPTCHAs, visual verification challenges, complex layouts, or when the text snaps… | — |

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

:::note
**Honcho tools** (`honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`) are no longer built-in. They are available via the Honcho memory provider plugin at `plugins/memory/honcho/`. See [Plugins](../user-guide/features/plugins.md) for installation and usage.
:::

## `image_gen` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `image_generate` | Generate high-quality images from text prompts using FLUX 2 Pro model with automatic 2x upscaling. Creates detailed, artistic images that are automatically upscaled for hi-rez results. Returns a single upscaled image URL. Display it using… | FAL_KEY |

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

## `rl` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `rl_check_status` | Get status and metrics for a training run. RATE LIMITED: enforces 30-minute minimum between checks for the same run. Returns WandB metrics: step, state, reward_mean, loss, percent_correct. | TINKER_API_KEY, WANDB_API_KEY |
| `rl_edit_config` | Update a configuration field. Use rl_get_current_config() first to see all available fields for the selected environment. Each environment has different configurable options. Infrastructure settings (tokenizer, URLs, lora_rank, learning_ra… | TINKER_API_KEY, WANDB_API_KEY |
| `rl_get_current_config` | Get the current environment configuration. Returns only fields that can be modified: group_size, max_token_length, total_steps, steps_per_eval, use_wandb, wandb_name, max_num_workers. | TINKER_API_KEY, WANDB_API_KEY |
| `rl_get_results` | Get final results and metrics for a completed training run. Returns final metrics and path to trained weights. | TINKER_API_KEY, WANDB_API_KEY |
| `rl_list_environments` | List all available RL environments. Returns environment names, paths, and descriptions. TIP: Read the file_path with file tools to understand how each environment works (verifiers, data loading, rewards). | TINKER_API_KEY, WANDB_API_KEY |
| `rl_list_runs` | List all training runs (active and completed) with their status. | TINKER_API_KEY, WANDB_API_KEY |
| `rl_select_environment` | Select an RL environment for training. Loads the environment's default configuration. After selecting, use rl_get_current_config() to see settings and rl_edit_config() to modify them. | TINKER_API_KEY, WANDB_API_KEY |
| `rl_start_training` | Start a new RL training run with the current environment and config. Most training parameters (lora_rank, learning_rate, etc.) are fixed. Use rl_edit_config() to set group_size, batch_size, wandb_project before starting. WARNING: Training… | TINKER_API_KEY, WANDB_API_KEY |
| `rl_stop_training` | Stop a running training job. Use if metrics look bad, training is stagnant, or you want to try different settings. | TINKER_API_KEY, WANDB_API_KEY |
| `rl_test_inference` | Quick inference test for any environment. Runs a few steps of inference + scoring using OpenRouter. Default: 3 steps x 16 completions = 48 rollouts per model, testing 3 models = 144 total. Tests environment loading, prompt construction, in… | TINKER_API_KEY, WANDB_API_KEY |

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
| `terminal` | Execute shell commands on a Linux environment. Filesystem persists between calls. Do NOT use cat/head/tail to read files — use read_file instead. Do NOT use grep/rg/find to search — use search_files instead. Do NOT use ls to list directori… | — |

## `todo` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `todo` | Manage your task list for the current session. Use for complex tasks with 3+ steps or when the user provides multiple tasks. Call with no parameters to read the current list. Writing: - Provide 'todos' array to create/update items - merge=… | — |

## `vision` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `vision_analyze` | Analyze images using AI vision. Provides a comprehensive description and answers a specific question about the image content. | — |

## `web` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `web_search` | Search the web for information on any topic. Returns up to 5 relevant results with titles, URLs, and descriptions. | EXA_API_KEY or PARALLEL_API_KEY or FIRECRAWL_API_KEY or TAVILY_API_KEY |
| `web_extract` | Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs — pass the PDF link directly and it converts to markdown text. Pages under 5000 chars return full markdown; larger pages are LLM-summarized. | EXA_API_KEY or PARALLEL_API_KEY or FIRECRAWL_API_KEY or TAVILY_API_KEY |

## `tts` toolset

| Tool | Description | Requires environment |
|------|-------------|----------------------|
| `text_to_speech` | Convert text to speech audio. Returns a MEDIA: path that the platform delivers as a voice message. On Telegram it plays as a voice bubble, on Discord/WhatsApp as an audio attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider… | — |


