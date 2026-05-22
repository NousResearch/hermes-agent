---
sidebar_position: 4
title: "Toolsets 参考"
description: "Hermes 核心、组合、平台与动态 toolsets 的参考文档"
---

# Toolsets 参考

toolset 是一组具名工具集合，用来控制 agent 能做什么。它是按平台、会话、任务配置工具可用性的核心机制。

## Toolset 如何工作

每个工具都只属于一个 toolset。启用某个 toolset 后，该集合里的全部工具都会对 agent 可用。toolset 分三类：

- **Core** - 逻辑上相关的一组工具（例如 `file` 打包 `read_file`、`write_file`、`patch`、`search_files`）
- **Composite** - 为常见场景组合多个 core toolset（例如 `debugging` 组合 file、terminal、web）
- **Platform** - 某个部署场景的完整工具配置（例如 `hermes-cli` 是交互式 CLI 的默认配置）

## 配置 Toolset

### 会话级（CLI）

```bash
hermes chat --toolsets web,file,terminal
hermes chat --toolsets debugging        # composite，会展开为 file + terminal + web
hermes chat --toolsets all              # 全部 toolset
```

### 平台级（config.yaml）

```yaml
toolsets:
  - hermes-cli          # CLI 默认
  # - hermes-telegram   # Telegram 网关可单独覆盖
```

### 交互管理

```bash
hermes tools                            # curses UI，按平台启用/禁用
```

或在会话中：

```
/tools list
/tools disable browser
/tools enable rl
```

## Core Toolset

> 说明：以下表格是英文源文档的中文化摘要；工具名保持原文。

| Toolset | 工具 | 用途 |
|---------|------|------|
| `browser` | `browser_back`、`browser_cdp`、`browser_click`、`browser_console`、`browser_dialog`、`browser_get_images`、`browser_navigate`、`browser_press`、`browser_scroll`、`browser_snapshot`、`browser_type`、`browser_vision`、`web_search` | 浏览器自动化核心集。`browser_cdp` 与 `browser_dialog` 会在运行时按 CDP 可达性动态注册。 |
| `clarify` | `clarify` | 需要澄清时向用户提问。 |
| `code_execution` | `execute_code` | 运行可程序化调用 Hermes 工具的 Python 脚本。 |
| `cronjob` | `cronjob` | 定时任务创建与管理。 |
| `debugging` | 组合（`file` + `terminal` + `web`） | 调试常用组合。 |
| `delegation` | `delegate_task` | 启动隔离的子 agent 并行处理任务。 |
| `discord` | `discord` | Discord 核心操作（仅网关）。 |
| `discord_admin` | `discord_admin` | Discord 管理操作（封禁、角色、频道等，需要机器人权限）。 |
| `feishu_doc` | `feishu_doc_read` | 读取 Feishu/Lark 文档内容。 |
| `feishu_drive` | `feishu_drive_add_comment`、`feishu_drive_list_comments`、`feishu_drive_list_comment_replies`、`feishu_drive_reply_comment` | Feishu/Lark 文档评论相关操作。 |
| `file` | `patch`、`read_file`、`search_files`、`write_file` | 文件读写、检索与局部编辑。 |
| `homeassistant` | `ha_call_service`、`ha_get_state`、`ha_list_entities`、`ha_list_services` | Home Assistant 智能家居控制。 |
| `computer_use` | `computer_use` | macOS 后台桌面控制（不抢光标/焦点，需 `cua-driver`）。 |
| `image_gen` | `image_generate` | 文生图能力（默认走 FAL.ai）。 |
| `kanban` | `kanban_block`、`kanban_comment`、`kanban_complete`、`kanban_create`、`kanban_heartbeat`、`kanban_link`、`kanban_show` | 多 agent 看板协作工具，仅 dispatcher 场景动态注册。 |
| `memory` | `memory` | 跨会话持久记忆管理。 |
| `messaging` | `send_message` | 向其他平台发送消息。 |
| `moa` | `mixture_of_agents` | 多模型协同求解复杂问题。 |
| `rl` | `rl_check_status`、`rl_edit_config`、`rl_get_current_config`、`rl_get_results`、`rl_list_environments`、`rl_list_runs`、`rl_select_environment`、`rl_start_training`、`rl_stop_training`、`rl_test_inference` | RL 训练环境与任务控制。 |
| `safe` | `image_generate`、`vision_analyze`、`web_extract`、`web_search`（通过 includes） | 只读研究与媒体生成。 |
| `search` | `web_search` | 仅网页搜索。 |
| `session_search` | `session_search` | 检索历史会话记忆。 |
| `skills` | `skill_manage`、`skill_view`、`skills_list` | skill 的查看与管理。 |
| `spotify` | `spotify_albums`、`spotify_devices`、`spotify_library`、`spotify_playback`、`spotify_playlists`、`spotify_queue`、`spotify_search` | Spotify 播放/检索/歌单能力。 |
| `terminal` | `process`、`terminal` | Shell 执行与后台进程管理。 |
| `todo` | `todo` | 会话内任务清单管理。 |
| `tts` | `text_to_speech` | 文本转语音。 |
| `vision` | `vision_analyze` | 图像分析。 |
| `video` | `video_analyze` | 视频分析（默认不在标准 toolset 中，需显式添加）。 |
| `web` | `web_extract`、`web_search` | 网页搜索与提取。 |
| `yuanbao` | `yb_query_group_info`、`yb_query_group_members`、`yb_search_sticker`、`yb_send_dm`、`yb_send_sticker` | Yuanbao 平台 DM/群组/贴纸能力。 |

## Platform Toolset

平台 toolset 定义某个部署目标的完整工具配置。多数消息平台与 `hermes-cli` 保持一致：

| Toolset | 与 `hermes-cli` 的差异 |
|---------|-------------------------|
| `hermes-cli` | 完整工具集，交互式 CLI 默认值。 |
| `hermes-acp` | 移除 `clarify`、`cronjob`、`image_generate`、`send_message`、`text_to_speech` 及 Home Assistant 四个工具。更偏 IDE 代码场景。 |
| `hermes-api-server` | 移除 `clarify`、`send_message`、`text_to_speech`。 |
| `hermes-cron` | 与 `hermes-cli` 相同。 |
| `hermes-telegram` | 与 `hermes-cli` 相同。 |
| `hermes-discord` | 在 `hermes-cli` 上额外加入 `discord` 与 `discord_admin`。 |
| `hermes-slack` | 与 `hermes-cli` 相同。 |
| `hermes-whatsapp` | 与 `hermes-cli` 相同。 |
| `hermes-signal` | 与 `hermes-cli` 相同。 |
| `hermes-matrix` | 与 `hermes-cli` 相同。 |
| `hermes-mattermost` | 与 `hermes-cli` 相同。 |
| `hermes-email` | 与 `hermes-cli` 相同。 |
| `hermes-sms` | 与 `hermes-cli` 相同。 |
| `hermes-bluebubbles` | 与 `hermes-cli` 相同。 |
| `hermes-dingtalk` | 与 `hermes-cli` 相同。 |
| `hermes-feishu` | 增加 `feishu_doc_*` / `feishu_drive_*` 系列工具（主要用于文档评论处理）。 |
| `hermes-qqbot` | 与 `hermes-cli` 相同。 |
| `hermes-wecom` | 与 `hermes-cli` 相同。 |
| `hermes-wecom-callback` | 与 `hermes-cli` 相同。 |
| `hermes-weixin` | 与 `hermes-cli` 相同。 |
| `hermes-yuanbao` | 在 `hermes-cli` 上额外加入 `yb_*` 工具。 |
| `hermes-homeassistant` | 与 `hermes-cli` 相同（`HASS_TOKEN` 配置后自动激活 HA 工具）。 |
| `hermes-webhook` | 与 `hermes-cli` 相同。 |
| `hermes-gateway` | 网关内部编排 toolset：所有 `hermes-<platform>` 的并集。 |

## Dynamic Toolset

### MCP server toolset

每个已配置 MCP server 会在运行时生成一个 `mcp-<server>` toolset。例如配置 `github` server 后，会生成 `mcp-github`，其中包含该 server 暴露的所有工具。

```yaml
# config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
```

这样就会生成 `mcp-github`，可用于 `--toolsets` 或平台配置。

### Plugin toolset

插件可以在初始化阶段通过 `ctx.register_tool()` 注册自己的 toolset。它们会和内建 toolset 一样出现，也能同样被启用/禁用。

### Custom toolset

你可以在 `config.yaml` 里定义自定义 toolset，创建项目化组合：

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

### 通配

- `all` 或 `*` - 展开为所有已注册 toolset（内建 + 动态 + 插件）

## 与 `hermes tools` 的关系

`hermes tools` 提供基于 curses 的 UI，用来按平台启用/禁用单个工具。它的粒度比 toolset 更细，并会持久化写入 `config.yaml`。即使某个 toolset 已启用，只要其中某个工具被显式禁用，它也不会对 agent 可用。

另见：[Tools 参考](/reference/tools-reference) 获取全部工具与参数详情。
