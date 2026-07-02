---
sidebar_position: 4
title: "工具集参考"
description: "Hermes 核心、复合、平台及动态工具集参考"
---

# 工具集参考

工具集（Toolset）是工具的命名集合，用于控制 agent 可以执行的操作。它是按平台、按会话或按任务配置工具可用性的主要机制。

## 工具集的工作原理

每个工具恰好属于一个工具集。启用某个工具集后，该集合中的所有工具都将对 agent 可用。工具集分为三种类型：

- **核心（Core）** — 一组相关工具的逻辑分组（例如，`file` 包含 `read_file`、`write_file`、`patch`、`search_files`）
- **复合（Composite）** — 将多个核心工具集组合用于常见场景（例如，`debugging` 包含 file、terminal 和 web 工具）
- **平台（Platform）** — 针对特定部署环境的完整工具配置（例如，`hermes-cli` 是交互式 CLI 会话的默认配置）

## 配置工具集

### 按会话（CLI）

```bash
hermes chat --toolsets web,file,terminal
hermes chat --toolsets debugging        # composite — expands to file + terminal + web
hermes chat --toolsets all              # everything
```

### 按平台（config.yaml）

```yaml
toolsets:
  - hermes-cli          # default for CLI
  # - hermes-telegram   # override for Telegram gateway
```

### 交互式管理

```bash
hermes tools                            # curses UI to enable/disable per platform
```

或在会话中：

```
/tools list
/tools disable browser
/tools enable homeassistant
```

## 核心工具集

| 工具集 | 工具 | 用途 |
|--------|------|------|
| `browser` | `browser_back`, `browser_cdp`, `browser_click`, `browser_console`, `browser_dialog`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `web_search` | 核心浏览器自动化。包含 `web_search` 作为快速查询的备用方案。`browser_cdp` 和 `browser_dialog` 在运行时受限——仅在会话启动时 CDP 端点可达（通过 `/browser connect`、`browser.cdp_url` 配置、Browserbase 或 Camofox）时才注册。`browser_dialog` 与 `browser_snapshot` 在附加 CDP supervisor 时添加的 `pending_dialogs` 和 `frame_tree` 字段配合使用。 |
| `clarify` | `clarify` | 当 agent 需要澄清时向用户提问。 |
| `code_execution` | `execute_code` | 运行以编程方式调用 Hermes 工具的 Python 脚本。 |
| `cronjob` | `cronjob` | 调度和管理周期性任务。 |
| `debugging` | 复合（`file` + `terminal` + `web`） | 调试套件——文件、进程/终端、网页提取/搜索。 |
| `delegation` | `delegate_task` | 生成隔离的子 agent 实例以并行执行工作，并支持按调用或按 task 设置 `model`/`provider` 路由。 |
| `discord` | `discord` | 核心 Discord 文本/嵌入/私信操作（仅限 gateway）。在 `hermes-discord` 工具集上激活。 |
| `discord_admin` | `discord_admin` | Discord 管理操作（封禁、角色变更、频道管理）。在 `hermes-discord` 工具集上激活；需要 bot 持有相关 Discord 权限。 |
| `feishu_doc` | `feishu_doc_read` | 读取飞书/Lark 文档内容。由飞书文档评论智能回复处理器使用。 |
| `feishu_drive` | `feishu_drive_add_comment`, `feishu_drive_list_comments`, `feishu_drive_list_comment_replies`, `feishu_drive_reply_comment` | 飞书/Lark 云盘评论操作。仅限评论 agent 使用；不在 `hermes-cli` 或其他消息工具集上暴露。 |
| `file` | `patch`, `read_file`, `search_files`, `write_file` | 文件读取、写入、搜索和编辑。 |
| `homeassistant` | `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services` | 通过 Home Assistant 进行智能家居控制。仅在设置 `HASS_TOKEN` 时可用。 |
| `computer_use` | `computer_use` | 通过 cua-driver 进行后台 macOS 桌面控制——不抢占光标/焦点。适用于任何支持工具调用的模型。仅限 macOS；需要 `cua-driver` 在 `$PATH` 中。 |
| `image_gen` | `image_generate` | 通过 FAL.ai 进行文本生成图像（支持可选的 OpenAI / xAI 后端）。 |
| `video_gen` | `video_generate` | 通过插件注册的后端（xAI Grok-Imagine、FAL.ai Veo 3.1 / Pixverse v6 / Kling O3）进行文本生成视频和图像生成视频。传入 `image_url` 可对图像进行动画化；省略则为文本生成视频。 |
| `kanban` | `kanban_block`, `kanban_comment`, `kanban_complete`, `kanban_create`, `kanban_heartbeat`, `kanban_link`, `kanban_list`, `kanban_show`, `kanban_unblock` | 多 agent 协调工具。为调度器生成的任务工作者（`HERMES_KANBAN_TASK`）以及显式启用 `kanban` 工具集的 profile 注册。工作者可标记任务完成、阻塞、心跳、评论以及创建/关联后续任务；编排器 profile 还额外获得看板路由工具，如 list/unblock。 |
| `memory` | `memory` | 持久化跨会话记忆管理。 |
| `messaging` | `send_message` | 在会话中向其他平台（Telegram、Discord 等）发送消息。 |
| `safe` | `image_generate`, `vision_analyze`, `web_extract`, `web_search`（通过 `includes`） | 只读研究 + 媒体生成。无文件写入、无终端、无代码执行。 |
| `search` | `web_search` | 仅网页搜索（不含提取）。 |
| `session_search` | `session_search` | 搜索历史会话记录。 |
| `skills` | `skill_manage`, `skill_view`, `skills_list` | 技能的增删改查与浏览。 |
| `spotify` | `spotify_albums`, `spotify_devices`, `spotify_library`, `spotify_playback`, `spotify_playlists`, `spotify_queue`, `spotify_search` | 原生 Spotify 控制（播放、队列、搜索、播放列表、专辑、音乐库）。由内置 `spotify` 插件注册。 |
| `terminal` | `process`, `terminal` | Shell 命令执行和后台进程管理。 |
| `todo` | `todo` | 会话内任务列表管理。 |
| `tts` | `text_to_speech` | 文本转语音音频生成。 |
| `vision` | `vision_analyze` | 通过视觉能力模型进行图像分析。 |
| `video` | `video_analyze` | 视频分析与理解工具（需手动启用，不在默认工具集中——通过 `--toolsets` 显式添加）。 |
| `web` | `web_extract`, `web_search` | 网页搜索和页面内容提取。 |
| `x_search` | `x_search` | 通过 xAI 内置的 `x_search` Responses 工具搜索 X（Twitter）帖子和话题。默认关闭；通过 `hermes tools` 启用。仅在配置了 xAI 凭据（SuperGrok OAuth 或 `XAI_API_KEY`）时注册 schema。 |
| `yuanbao` | `yb_query_group_info`, `yb_query_group_members`, `yb_search_sticker`, `yb_send_dm`, `yb_send_sticker` | 元宝私信/群组操作和表情包搜索。仅在 `hermes-yuanbao` 上注册。 |

## 平台工具集

平台工具集定义了部署目标的完整工具配置。大多数消息平台使用与 `hermes-cli` 相同的配置：

| 工具集 | 与 `hermes-cli` 的差异 |
|--------|------------------------|
| `hermes-cli` | 完整工具集——交互式 CLI 会话的默认配置。包含 file、terminal、web、browser、memory、skills、vision、image_gen、todo、tts、delegation、code_execution、cronjob、session_search、clarify 和 `safe`（只读）套件，以及标准消息工具。 |
| `hermes-acp` | 移除了 `clarify`、`cronjob`、`image_generate`、`send_message`、`text_to_speech` 以及全部四个 Home Assistant 工具。专注于 IDE 环境中的编码任务。 |
| `hermes-api-server` | 移除了 `clarify`、`send_message` 和 `text_to_speech`。保留其他所有工具——适用于无法进行用户交互的程序化访问场景。 |
| `hermes-cron` | 与 `hermes-cli` 相同。 |
| `hermes-telegram` | 与 `hermes-cli` 相同。 |
| `hermes-discord` | 在 `hermes-cli` 基础上添加了 `discord` 和 `discord_admin`。 |
| `hermes-slack` | 与 `hermes-cli` 相同。 |
| `hermes-whatsapp` | 与 `hermes-cli` 相同。 |
| `hermes-signal` | 与 `hermes-cli` 相同。 |
| `hermes-matrix` | 与 `hermes-cli` 相同。 |
| `hermes-mattermost` | 与 `hermes-cli` 相同。 |
| `hermes-email` | 与 `hermes-cli` 相同。 |
| `hermes-sms` | 与 `hermes-cli` 相同。 |
| `hermes-bluebubbles` | 与 `hermes-cli` 相同。 |
| `hermes-dingtalk` | 与 `hermes-cli` 相同。 |
| `hermes-feishu` | 添加了五个 `feishu_doc_*` / `feishu_drive_*` 工具（仅由文档评论处理器使用，不用于常规聊天适配器）。 |
| `hermes-qqbot` | 与 `hermes-cli` 相同。 |
| `hermes-wecom` | 与 `hermes-cli` 相同。 |
| `hermes-wecom-callback` | 与 `hermes-cli` 相同。 |
| `hermes-weixin` | 与 `hermes-cli` 相同。 |
| `hermes-yuanbao` | 在 `hermes-cli` 基础上添加了五个 `yb_*` 工具（私信/群组/表情包）。 |
| `hermes-homeassistant` | 与 `hermes-cli` 相同（Home Assistant 工具默认已存在，在设置 `HASS_TOKEN` 时激活）。 |
| `hermes-webhook` | 与 `hermes-cli` 相同。 |
| `hermes-gateway` | 内部 gateway 编排器工具集——所有 `hermes-<platform>` 工具集的并集；当 gateway 需要接受任意消息来源时使用。 |

## 动态工具集

### MCP server 工具集

每个已配置的 MCP server 在运行时会生成一个 `mcp-<server>` 工具集。例如，若配置了 `github` MCP server，则会创建包含该 server 所有暴露工具的 `mcp-github` 工具集。

```yaml
# config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
```

这将创建一个 `mcp-github` 工具集，可在 `--toolsets` 或平台配置中引用。

### 插件工具集

插件可在初始化期间通过 `ctx.register_tool()` 注册自己的工具集。这些工具集与内置工具集并列显示，可以用相同方式启用/禁用。

### 自定义工具集

在 `config.yaml` 中定义自定义工具集，以创建项目专属的工具集合：

```yaml
toolsets:
  - hermes-cli
custom_toolsets:
  data-science:
    - file
    - terminal
    - code_execution
    - web
    - vision
```

### 通配符

- `all` 或 `*` — 展开为所有已注册的工具集（内置 + 动态 + 插件）

## 与 `hermes tools` 的关系

`hermes tools` 命令提供基于 curses 的 UI，用于按平台切换单个工具的启用/禁用状态。该操作在工具级别进行（比工具集更细粒度），并持久化到 `config.yaml`。即使工具集已启用，被禁用的工具也会被过滤掉。

另请参阅：[工具参考](./tools-reference.md)，获取所有单个工具及其参数的完整列表。