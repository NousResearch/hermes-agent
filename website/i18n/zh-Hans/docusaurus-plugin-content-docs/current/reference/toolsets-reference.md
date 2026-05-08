---
sidebar_position: 4
title: "工具集参考"
description: "Hermes 核心、复合、平台和动态工具集参考"
---

# 工具集参考

工具集是命名的工具捆绑包，用于控制 agent 可以执行的操作。它们是按平台、会话或任务配置工具可用性的主要机制。

## 工具集如何工作

每个工具恰好属于一个工具集。当您启用一个工具集时，该捆绑包中的所有工具都可供 agent 使用。工具集有三种类型：

- **核心** —— 单一逻辑相关的工具组（例如，`file` 捆绑 `read_file`、`write_file`、`patch`、`search_files`）
- **复合** —— 为常见场景组合多个核心工具集（例如，`debugging` 捆绑 file、terminal 和 web 工具）
- **平台** —— 特定部署上下文的完整工具配置（例如，`hermes-cli` 是交互式 CLI 会话的默认配置）

## 配置工具集

### 按会话（CLI）

```bash
hermes chat --toolsets web,file,terminal
hermes chat --toolsets debugging        # 复合 —— 展开为 file + terminal + web
hermes chat --toolsets all              # 所有工具
```

### 按平台（config.yaml）

```yaml
toolsets:
  - hermes-cli          # CLI 默认值
  # - hermes-telegram   # Telegram 网关覆盖
```

### 交互式管理

```bash
hermes tools                            # curses UI，按平台启用/禁用
```

或在会话中：

```
/tools list
/tools disable browser
/tools enable rl
```

## 核心工具集

| 工具集 | 工具 | 用途 |
|---------|-------|---------|
| `browser` | `browser_back`, `browser_click`, `browser_console`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `web_search` | 核心浏览器自动化。包括 `web_search` 作为快速查找的备用。`browser_cdp` 和 `browser_dialog` 位于单独的 `browser-cdp` 工具集中，仅当会话启动时可访问 CDP 端点时才注册 —— 通过 `/browser connect`、`browser.cdp_url` 配置、Browserbase 或 Camofox。`browser_dialog` 与 `browser_snapshot` 在附加 CDP 监督器时添加的 `pending_dialogs` 和 `frame_tree` 字段配合工作。 |
| `clarify` | `clarify` | 当 agent 需要澄清时向用户提问。 |
| `code_execution` | `execute_code` | 运行编程调用 Hermes 工具的 Python 脚本。 |
| `cronjob` | `cronjob` | 调度和管理定期任务。 |
| `debugging` | 复合（`file` + `terminal` + `web`） | 调试捆绑包 —— 文件、进程/终端、网页提取/搜索。 |
| `delegation` | `delegate_task` | 生成隔离的子 agent 实例用于并行工作。 |
| `discord` | `discord` | 核心 Discord 文本/嵌入/私信操作（仅网关）。在 `hermes-discord` 工具集上活跃。 |
| `discord_admin` | `discord_admin` | Discord 管理（封禁、角色更改、频道管理）。在 `hermes-discord` 工具集上活跃；需要 bot 持有相关 Discord 权限。 |
| `feishu_doc` | `feishu_doc_read` | 读取飞书/Lark 文档内容。由飞书文档评论智能回复处理器使用。 |
| `feishu_drive` | `feishu_drive_add_comment`, `feishu_drive_list_comments`, `feishu_drive_list_comment_replies`, `feishu_drive_reply_comment` | 飞书/Lark 云盘评论操作。限于评论 agent；不在 `hermes-cli` 或其他消息工具集上暴露。 |
| `file` | `patch`, `read_file`, `search_files`, `write_file` | 文件读取、写入、搜索和编辑。 |
| `homeassistant` | `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services` | 通过 Home Assistant 控制智能家居。仅在设置 `HASS_TOKEN` 时可用。 |
| `image_gen` | `image_generate` | 通过 FAL.ai 进行文本到图像生成（可选 OpenAI / xAI 后端）。 |
| `memory` | `memory` | 持久跨会话内存管理。 |
| `messaging` | `send_message` | 在会话内发送消息到其他平台（Telegram、Discord 等）。 |
| `moa` | `mixture_of_agents` | 通过 Mixture of Agents 实现多模型共识。 |
| `rl` | `rl_check_status`, `rl_edit_config`, `rl_get_current_config`, `rl_get_results`, `rl_list_environments`, `rl_list_runs`, `rl_select_environment`, `rl_start_training`, `rl_stop_training`, `rl_test_inference` | RL 训练环境管理（Atropos）。 |
| `safe` | `image_generate`, `vision_analyze`, `web_extract`, `web_search`（通过 `includes`） | 只读研究 + 媒体生成。无文件写入、无终端、无代码执行。 |
| `search` | `web_search` | 仅网页搜索（无提取）。 |
| `session_search` | `session_search` | 搜索过去的对话会话。 |
| `skills` | `skill_manage`, `skill_view`, `skills_list` | 技能增删改查和浏览。 |
| `spotify` | `spotify_albums`, `spotify_devices`, `spotify_library`, `spotify_playback`, `spotify_playlists`, `spotify_queue`, `spotify_search` | 原生 Spotify 控制（播放、队列、搜索、播放列表、专辑、资料库）。由捆绑的 `spotify` 插件注册。 |
| `terminal` | `process`, `terminal` | Shell 命令执行和后台进程管理。 |
| `todo` | `todo` | 会话内任务列表管理。 |
| `tts` | `text_to_speech` | 文本到语音音频生成。 |
| `vision` | `vision_analyze` | 通过支持视觉的模型进行图像分析。 |
| `web` | `web_extract`, `web_search` | 网页搜索和页面内容提取。 |
| `yuanbao` | `yb_query_group_info`, `yb_query_group_members`, `yb_search_sticker`, `yb_send_dm`, `yb_send_sticker` | 元宝私信/群组操作和贴纸搜索。仅注册在 `hermes-yuanbao`。 |

## 平台工具集

平台工具集定义部署目标的完整工具配置。大多数消息平台使用与 `hermes-cli` 相同的工具集：

| 工具集 | 与 `hermes-cli` 的区别 |
|---------|-------------------------------|
| `hermes-cli` | 完整工具集 —— 38 个工具。交互式 CLI 会话的默认配置。 |
| `hermes-acp` | 移除 `clarify`、`cronjob`、`image_generate`、`send_message`、`text_to_speech` 和全部四个 Home Assistant 工具。专注于 IDE 上下文中的编码任务。 |
| `hermes-api-server` | 移除 `clarify`、`send_message` 和 `text_to_speech`。保留其他所有工具 —— 适用于无法进行用户交互的程序化访问。 |
| `hermes-cron` | 与 `hermes-cli` 相同。 |
| `hermes-telegram` | 与 `hermes-cli` 相同。 |
| `hermes-discord` | 在 `hermes-cli` 基础上添加 `discord` 和 `discord_admin`。 |
| `hermes-slack` | 与 `hermes-cli` 相同。 |
| `hermes-whatsapp` | 与 `hermes-cli` 相同。 |
| `hermes-signal` | 与 `hermes-cli` 相同。 |
| `hermes-matrix` | 与 `hermes-cli` 相同。 |
| `hermes-mattermost` | 与 `hermes-cli` 相同。 |
| `hermes-email` | 与 `hermes-cli` 相同。 |
| `hermes-sms` | 与 `hermes-cli` 相同。 |
| `hermes-bluebubbles` | 与 `hermes-cli` 相同。 |
| `hermes-dingtalk` | 与 `hermes-cli` 相同。 |
| `hermes-feishu` | 添加五个 `feishu_doc_*` / `feishu_drive_*` 工具（仅供文档评论处理器使用，不在常规聊天适配器上）。 |
| `hermes-qqbot` | 与 `hermes-cli` 相同。 |
| `hermes-wecom` | 与 `hermes-cli` 相同。 |
| `hermes-wecom-callback` | 与 `hermes-cli` 相同。 |
| `hermes-weixin` | 与 `hermes-cli` 相同。 |
| `hermes-yuanbao` | 在 `hermes-cli` 基础上添加五个 `yb_*` 工具（私信/群组/贴纸）。 |
| `hermes-homeassistant` | 与 `hermes-cli` 相同（Home Assistant 工具已默认存在，在设置 `HASS_TOKEN` 时激活）。 |
| `hermes-webhook` | 与 `hermes-cli` 相同。 |
| `hermes-gateway` | 内部网关编排器工具集 —— 每个 `hermes-<platform>` 工具集的并集；当网关需要接受任何消息源时使用。 |

## 动态工具集

### MCP 服务器工具集

每个配置的 MCP 服务器在运行时生成一个 `mcp-<server>` 工具集。例如，如果您配置了 `github` MCP 服务器，会创建一个包含该服务器暴露的所有工具的 `mcp-github` 工具集。

```yaml
# config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
```

这会创建一个可在 `--toolsets` 或平台配置中引用的 `mcp-github` 工具集。

### 插件工具集

插件可以通过插件初始化期间的 `ctx.register_tool()` 注册自己的工具集。这些工具集与内置工具集并列显示，可以相同方式启用/禁用。

### 自定义工具集

在 `config.yaml` 中定义自定义工具集以创建项目特定的捆绑包：

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

- `all` 或 `*` —— 展开为每个已注册工具集（内置 + 动态 + 插件）

## 与 `hermes tools` 的关系

`hermes tools` 命令提供一个基于 curses 的 UI，用于按平台切换单个工具的开/关。这在工具级别操作（比工具集更细粒度）并持久化到 `config.yaml`。禁用的工具即使其工具集已启用也会被过滤掉。

另请参阅：[工具参考](./tools-reference.md) 了解完整工具列表及其参数。
