---
sidebar_position: 3
title: "Limits and Defaults"
description: "Where Hermes caps sizes, counts and timeouts: the default value, the config.yaml key that changes it, and the source that enforces it"
---

# Limits and Defaults

Hermes caps output sizes, concurrency, retries and timeouts in many places. This page lists the ones you are most likely to hit, with the shipped default, the `config.yaml` key that changes it, and the source that defines it (`path::symbol`). A value marked **constant, not configurable** is a code constant; changing it means changing the source.

Config keys are dotted paths into `config.yaml` (`delegation.max_concurrent_children` is `delegation:` → `max_concurrent_children:`). Defaults come from `DEFAULT_CONFIG` in `hermes_cli/config_defaults.py`; run `hermes config` to see your effective values. Where a comment in the code says `0` turns a limit off, the table says so.

This page is a curated subset, not an exhaustive list. For every key, see [Configuration](../user-guide/configuration.md); for environment-variable overrides, see [Environment Variables](./environment-variables.md).

## Agent turn and retries

| Limit | Default | Config key | Source |
|---|---|---|---|
| API retry attempts per call (connection drops, timeouts, 5xx) | `3` | `agent.api_max_retries` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Turn iteration budget | unlimited (`null`) | `agent.max_turns` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Primary rate-limit cooldown (no provider reset time) | `60 × 2^n` s, capped at `14400` s | constant, not configurable | `agent/fallback_cooldown.py::_arm_rate_limit_cooldown` |
| Skip fallback when primary resets within N seconds | `0` (off) | `fallback.min_switch_reset_seconds` | `agent/fallback_cooldown.py` |
| Cooldown after the whole fallback chain fails (non-rate-limit) | `5.0` s | constant, not configurable | `agent/chat_completion_helpers.py::_FALLBACK_EXHAUSTED_COOLDOWN_S` |
| `/goal` continuation turns before auto-pause | `20` | `goals.max_turns` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| `/loop` wakeups before auto-pause | `100` (`0` = unlimited) | `loops.max_ticks` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| `/loop` smallest fixed interval | `30` s | `loops.min_interval_seconds` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |

## Credential pool

| Limit | Default | Config key | Source |
|---|---|---|---|
| Concurrent requests leased per credential | `1` | constant, not configurable | `agent/credential_pool.py::DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL` |
| Cooldown after a 401 | `300` s | constant, not configurable | `agent/credential_pool.py::EXHAUSTED_TTL_401_SECONDS` |
| Cooldown after a 429 | `3600` s | constant, not configurable | `agent/credential_pool.py::EXHAUSTED_TTL_429_SECONDS` |
| Cooldown after other exhaustion errors | `3600` s | constant, not configurable | `agent/credential_pool.py::EXHAUSTED_TTL_DEFAULT_SECONDS` |
| Cooldown when the failing key is the only usable one | `60` s | constant, not configurable | `agent/credential_pool.py::EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS` |

A provider-supplied reset time overrides these cooldowns.

## Context compression

| Limit | Default | Config key | Source |
|---|---|---|---|
| Compress when context usage exceeds | `0.50` of the window | `compression.threshold` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Threshold floor for windows below 512K tokens | `0.75` (raise-only) | constant, not configurable | `agent/context_compressor.py::_SMALL_CTX_THRESHOLD_PERCENT` |
| Window size below which the floor applies | `512000` tokens | constant, not configurable | `agent/context_compressor.py::_SMALL_CTX_WINDOW_LIMIT` |
| Absolute token trigger | off (`null`) | `compression.threshold_tokens` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Fraction of threshold kept as recent tail | `0.20` | `compression.target_ratio` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Minimum recent messages kept uncompressed | `20` | `compression.protect_last_n` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Head messages always kept verbatim (besides the system prompt) | `3` | `compression.protect_first_n` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Compression attempts per trigger | `3` | `compression.max_attempts` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Gateway force-compress threshold (message count) | `5000` | `compression.hygiene_hard_message_limit` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |

## Tool output and file reads

| Limit | Default | Config key | Source |
|---|---|---|---|
| Terminal output kept per call (chars, head + tail) | `50000` | `tool_output.max_bytes` | `tools/tool_output_limits.py::DEFAULT_MAX_BYTES` |
| Max `limit` one `read_file` call may request | `2000` lines | `tool_output.max_lines` | `tools/tool_output_limits.py::DEFAULT_MAX_LINES` |
| Per-line cap in `read_file` output | `2000` chars | `tool_output.max_line_length` | `tools/tool_output_limits.py::DEFAULT_MAX_LINE_LENGTH` |
| Max chars per `read_file` call (larger reads are rejected) | `100000` | `file_read_max_chars` | `tools/file_tools.py::_DEFAULT_MAX_READ_CHARS` |
| `search_files` default result limit | `50` | constant, not configurable | `tools/file_operations_common.py::DEFAULT_SEARCH_LIMIT` |
| Background process rolling output buffer | `200000` chars | constant, not configurable | `tools/process_registry.py::MAX_OUTPUT_CHARS` |
| `web_extract` per-page char budget | `15000` | `web.extract_char_limit` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| `web_search` calls per turn | `50` (`0` = unlimited) | `tool_loop_guardrails.loop_caps.max_web_searches` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Subagents spawned per turn | `50` (`0` = unlimited) | `tool_loop_guardrails.loop_caps.max_subagents` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |

## Terminal and code execution

| Limit | Default | Config key | Source |
|---|---|---|---|
| Foreground command timeout | `180` s | `terminal.timeout` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Grace between SIGTERM and SIGKILL for host process trees | `2.0` s | `terminal.daemon_term_grace_seconds` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Container memory (docker, modal, daytona, …) | `5120` MB | `terminal.container_memory` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Container disk | `51200` MB | `terminal.container_disk` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| `execute_code` kernel idle timeout | `1800` s | `code_execution.kernel_idle_timeout` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| `execute_code` kernels per session | `4` | `code_execution.max_session_kernels` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Dangerous-command approval wait | `300` s | `approvals.timeout` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |

## Delegation (subagents)

| Limit | Default | Config key | Source |
|---|---|---|---|
| Parallel children per batch / concurrent background units | `10` (floor 1, no ceiling) | `delegation.max_concurrent_children` | `tools/delegate_tool_config.py::_DEFAULT_MAX_CONCURRENT_CHILDREN` |
| Spawn depth (parent is depth 0) | `1` (floor 1, no ceiling) | `delegation.max_spawn_depth` | `tools/delegate_tool_config.py::MAX_DEPTH` |
| Iteration budget per child | `250` | `delegation.max_iterations` | `tools/delegate_tool.py::DEFAULT_MAX_ITERATIONS` |
| Child summary ceiling returned to the parent | `24000` chars (`0` = off) | `delegation.max_summary_chars` | `tools/delegate_tool_results.py::DEFAULT_MAX_SUMMARY_CHARS` |
| Child inactivity timeout | `0` (no cap; floor 30 s when set) | `delegation.child_timeout_seconds` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Subagents per one-shot run (`hermes chat -q`) | `2` (`0` = unlimited) | `delegation.oneshot_max_children` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Images attached per task | `8` | constant, not configurable | `tools/delegate_tool_tasks.py::_MAX_TASK_IMAGES` |

## Gateway

| Limit | Default | Config key | Source |
|---|---|---|---|
| Inactivity timeout for a turn (idle only, not while tools run) | `1800` s (`0` = unlimited) | `agent.gateway_timeout` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Inactivity warning before the timeout | `900` s (`0` = off) | `agent.gateway_timeout_warning` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Chat-turn interrupt budget on shutdown | `0` s | `agent.restart_drain_timeout` | `gateway/restart.py::DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT` |
| Wait for running cron jobs on shutdown | `30` s (`0` = opt out) | `agent.cron_drain_timeout` | `gateway/restart.py::DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT` |
| `/restart` wait for turns to finish | `1800` s | `agent.restart_after_turn_timeout` | `gateway/restart.py::DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT` |
| Post-interrupt grace on shutdown | `5.0` s | constant, not configurable | `gateway/restart.py::DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT` |
| Per-platform connect timeout | `30` s (`0` = wait indefinitely) | `gateway.platform_connect_timeout` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Cached per-session agents | `128` | `agent.agent_cache.max_size` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Inbound media download cap | `134217728` bytes (128 MiB; `0` = no cap) | `gateway.max_inbound_media_bytes` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| API server concurrent `/v1/runs` | `10` (`0` = no cap) | `gateway.api_server.max_concurrent_runs` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| In-memory TUI/desktop/dashboard sessions (soft LRU) | `16` (`0` = off) | `max_live_sessions` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Minimum seconds between streaming edits | `0.8` | `streaming.edit_interval` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |

## Platform message length

Replies longer than these are split into multiple messages. An adapter without its own value uses `4096` (`gateway/platforms/base.py::BasePlatformAdapter.max_message_length_for_chat`).

| Platform | Max chars per message | Config key | Source |
|---|---|---|---|
| Telegram | `4096` | constant, not configurable | `plugins/platforms/telegram/adapter.py::TelegramAdapter.MAX_MESSAGE_LENGTH` |
| Discord | `2000` | constant, not configurable | `plugins/platforms/discord/adapter.py::DiscordAdapter.MAX_MESSAGE_LENGTH` |
| Slack | `39000` | constant, not configurable | `plugins/platforms/slack/adapter.py::SlackAdapter.MAX_MESSAGE_LENGTH` |
| WhatsApp | `4096` | constant, not configurable | `gateway/platforms/whatsapp_common.py::WhatsAppBehaviorMixin.MAX_MESSAGE_LENGTH` |
| Signal | `8000` | constant, not configurable | `plugins/platforms/signal/adapter.py::MAX_MESSAGE_LENGTH` |
| Matrix | `15000` UTF-8 bytes (also the maximum) | `matrix.max_message_length` | `plugins/platforms/matrix/adapter.py::DEFAULT_MAX_MESSAGE_LENGTH` |
| Microsoft Teams | `28000` | constant, not configurable | `plugins/platforms/teams/adapter.py::TeamsAdapter.MAX_MESSAGE_LENGTH` |
| Google Chat | `4000` | constant, not configurable | `plugins/platforms/google_chat/adapter.py::_MAX_TEXT_LENGTH` |
| Feishu / Lark | `8000` | constant, not configurable | `plugins/platforms/feishu/adapter.py::FeishuAdapter.MAX_MESSAGE_LENGTH` |
| SMS | `1600` | constant, not configurable | `plugins/platforms/sms/adapter.py::MAX_SMS_LENGTH` |
| Email | `50000` | constant, not configurable | `plugins/platforms/email/adapter.py::MAX_MESSAGE_LENGTH` |

## Slack slash commands

| Limit | Default | Config key | Source |
|---|---|---|---|
| Slash commands in a generated Slack app manifest | `50` | constant, not configurable | `hermes_cli/commands_platforms.py::_SLACK_MAX_SLASH_COMMANDS` |
| Slash command name length | `32` chars | constant, not configurable | `hermes_cli/commands_platforms.py::_SLACK_NAME_LIMIT` |

The manifest generator stops at 50, but Slack accepts at most 25 slash commands per app, so a manifest with more than 25 is rejected at install time (tracked in [#124762](https://github.com/NousResearch/hermes-agent/issues/124762)).

## Cron

| Limit | Default | Config key | Source |
|---|---|---|---|
| Scheduler tick interval | `60` s | constant, not configurable | `cron/jobs.py::TICKER_INTERVAL_SECONDS` |
| No-agent script timeout | `3600` s | `cron.script_timeout_seconds` | `cron/scheduler.py::_DEFAULT_SCRIPT_TIMEOUT` |
| Saved output files kept per job | `50` (`0` = no pruning) | `cron.output_retention` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Re-alert window for a repeating failure | `6` h (`0` = every run) | `cron.failure_repeat_alert_hours` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Per-job late-run grace | max `1440` min | per-job `grace_minutes` | `cron/jobs_time.py::MAX_GRACE_MINUTES` |
| Persisted error text per run | `2000` chars | constant, not configurable | `cron/jobs_error_text.py::MAX_PERSISTED_ERROR_CHARS` |

## Memory, sessions and storage

| Limit | Default | Config key | Source |
|---|---|---|---|
| Built-in memory size | `2200` chars | `memory.memory_char_limit` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| User profile memory size | `1375` chars | `memory.user_char_limit` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Messages loaded on interactive resume | `20000` (`0` = off) | `sessions.max_resume_messages` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Days of ended-session history kept by prune | `90` | `sessions.retention_days` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Checkpoints per working directory | `20` | `checkpoints.max_snapshots` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Total checkpoint storage | `500` MB (`0` = off) | `checkpoints.max_total_size_mb` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Log file size before rotation | `5` MB | `logging.max_size_mb` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
| Rotated log backups | `3` | `logging.backup_count` | `hermes_cli/config_defaults.py::DEFAULT_CONFIG` |
