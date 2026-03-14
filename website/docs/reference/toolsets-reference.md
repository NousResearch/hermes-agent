---
sidebar_position: 4
title: "Toolsets Reference"
description: "Reference for Hermes core, composite, platform, and dynamic toolsets"
---

# Toolsets Reference

Toolsets are named bundles of tools that you can enable with `hermes chat --toolsets ...`, configure per platform, or resolve inside the agent runtime.

| Toolset | Kind | Resolves to |
|---------|------|-------------|
| `browser` | core | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `web_search` |
| `clarify` | core | `clarify` |
| `code_execution` | core | `execute_code` |
| `cronjob` | core | `list_cronjobs`, `remove_cronjob`, `schedule_cronjob` |
| `debugging` | composite | `patch`, `process`, `read_file`, `search_files`, `terminal`, `web_extract`, `web_search`, `write_file` |
| `delegation` | core | `delegate_task` |
| `file` | core | `patch`, `read_file`, `search_files`, `write_file` |
| `hermes-cli` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `hermes-discord` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `hermes-email` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `hermes-gateway` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `hermes-homeassistant` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `hermes-signal` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `hermes-slack` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `hermes-telegram` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `hermes-whatsapp` | platform | `browser_back`, `browser_click`, `browser_close`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `clarify`, `delegate_task`, `execute_code`, `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services`, `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search`, `image_generate`, `list_cronjobs`, `memory`, `mixture_of_agents`, `patch`, `process`, `read_file`, `remove_cronjob`, `schedule_cronjob`, `search_files`, `send_message`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `text_to_speech`, `todo`, `vision_analyze`, `web_extract`, `web_search`, `write_file` |
| `homeassistant` | core | `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services` |
| `honcho` | core | `honcho_conclude`, `honcho_context`, `honcho_profile`, `honcho_search` |
| `image_gen` | core | `image_generate` |
| `memory` | core | `memory` |
| `moa` | core | `mixture_of_agents` |
| `rl` | core | `rl_check_status`, `rl_edit_config`, `rl_get_current_config`, `rl_get_results`, `rl_list_environments`, `rl_list_runs`, `rl_select_environment`, `rl_start_training`, `rl_stop_training`, `rl_test_inference` |
| `safe` | composite | `image_generate`, `mixture_of_agents`, `vision_analyze`, `web_extract`, `web_search` |
| `search` | core | `web_search` |
| `session_search` | core | `session_search` |
| `skills` | core | `skill_manage`, `skill_view`, `skills_list` |
| `terminal` | core | `process`, `terminal` |
| `todo` | core | `todo` |
| `tts` | core | `text_to_speech` |
| `vision` | core | `vision_analyze` |
| `web` | core | `web_extract`, `web_search` |

## Dynamic toolsets

- `mcp-<server>` — generated at runtime for each configured MCP server.
- Custom toolsets can be created in configuration and resolved at startup.
- Wildcards: `all` and `*` expand to every registered toolset.