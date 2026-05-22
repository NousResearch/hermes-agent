---
sidebar_position: 3
title: "内置工具参考"
description: "Hermes 内置工具总览（按 toolset 分组）"
---

# 内置工具参考

本页是内置工具的中文速查版，按 toolset 分组。工具可用性会受平台、凭据和启用的 toolset 影响。

> 说明：工具名保持英文原样，便于在 CLI 与配置文件中直接使用。

:::tip MCP 工具
除内置工具外，Hermes 还能动态加载 MCP 工具。命名形式为 `mcp_<server>_<tool>`。见 [MCP 集成](/user-guide/features/mcp)。
:::

## 高频 toolset

| Toolset | 代表工具 | 用途 |
|---|---|---|
| `browser` | `browser_navigate`、`browser_snapshot`、`browser_click`、`browser_type`、`browser_scroll` | 浏览器导航、交互、页面理解 |
| `file` | `read_file`、`search_files`、`patch`、`write_file` | 文件读取、搜索与编辑 |
| `terminal` | `terminal`、`process` | 命令执行与后台进程管理 |
| `web` | `web_search`、`web_extract` | 搜索网页与提取正文 |
| `memory` | `memory` | 跨会话持久记忆 |
| `skills` | `skills_list`、`skill_view`、`skill_manage` | 技能查看与管理 |
| `todo` | `todo` | 会话任务清单 |
| `vision` | `vision_analyze` | 图像分析 |
| `tts` | `text_to_speech` | 文本转语音 |

## 其他常见 toolset

| Toolset | 代表工具 | 说明 |
|---|---|---|
| `code_execution` | `execute_code` | 用 Python 脚本编排多步工具调用 |
| `delegation` | `delegate_task` | 启动子 agent 并行执行任务 |
| `cronjob` | `cronjob` | 定时任务管理 |
| `kanban` | `kanban_show`、`kanban_complete` | 多 agent 看板调度场景 |
| `homeassistant` | `ha_list_entities`、`ha_call_service` | 家居自动化 |
| `image_gen` | `image_generate` | 文生图 |
| `rl` | `rl_start_training` 等 | RL 训练任务 |
| `messaging` | `send_message` | 跨平台发送消息 |
| `session_search` | `session_search` | 历史会话检索 |

## 平台差异

- `hermes-cli`：默认最完整。
- `hermes-acp`：偏 IDE 编码场景，裁剪部分交互/消息工具。
- `hermes-discord`、`hermes-telegram` 等：在基础集上附加平台特有工具。

完整平台清单见 [Toolsets 参考](/reference/toolsets-reference)。

## 与 `hermes tools` 的关系

`hermes tools` 允许你在平台级启用/禁用单个工具。即使某个 toolset 已启用，被显式禁用的工具仍不会注册。
