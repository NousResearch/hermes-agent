---
sidebar_position: 3
title: "内置工具参考"
description: "Hermes 内置工具权威参考，按工具集分组"
---

# 内置工具参考

本页面记录了 Hermes 工具注册表中的全部 68 个内置工具，按工具集分组。不同平台、凭证和启用的工具集可用性有所不同。

**快速统计：** 10 个浏览器工具（核心）+ 2 个 browser-cdp 工具、4 个文件工具、10 个 RL 工具、4 个 Home Assistant 工具、2 个终端工具、2 个 Web 工具、5 个 Feishu 工具、7 个 Spotify 工具、5 个 Yuanbao 工具、2 个 Discord 工具，以及其他工具集中的 15 个独立工具。

:::tip MCP 工具
除了内置工具外，Hermes 还可以从 MCP 服务器动态加载工具。MCP 工具带有服务器名称前缀（例如，`github` MCP 服务器的 `github_create_issue`）。参见 [MCP 集成](/docs/user-guide/features/mcp) 了解配置方法。
:::

## `browser` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `browser_back` | 在浏览器历史记录中导航回上一页。需要先调用 browser_navigate。 | — |
| `browser_click` | 点击快照中通过 ref ID 标识的元素（例如 '@e5'）。ref ID 在快照输出中以方括号显示。需要先调用 browser_navigate 和 browser_snapshot。 | — |
| `browser_console` | 获取当前页面的浏览器控制台输出和 JavaScript 错误。返回 console.log/warn/error/info 消息和未捕获的 JS 异常。用此检测静默 JavaScript 错误、失败的 API 调用和应用警告。需要先调用 browser_navigate... | — |
| `browser_get_images` | 获取当前页面上所有图像及其 URL 和 alt 文本的列表。可用于查找图像以供视觉工具分析。需要先调用 browser_navigate。 | — |
| `browser_navigate` | 在浏览器中导航到 URL。初始化会话并加载页面。必须在调用其他浏览器工具之前调用。对于简单信息检索，优先使用 web_search 或 web_extract（更快、更便宜）。当您需要...时使用浏览器工具... | — |
| `browser_press` | 按下键盘键。可用于提交表单（Enter）、导航（Tab）或键盘快捷键。需要先调用 browser_navigate。 | — |
| `browser_scroll` | 在某个方向上滚动页面。用此显示可能在当前视口上方或下方的更多内容。需要先调用 browser_navigate。 | — |
| `browser_snapshot` | 获取当前页面可访问性树的文本快照。返回带有 ref ID（如 @e1、@e2）的交互元素供 browser_click 和 browser_type 使用。full=false（默认）：紧凑视图，仅交互元素。full=true：完整视图... | — |
| `browser_type` | 在通过 ref ID 标识的输入字段中键入文本。先清除字段，然后输入新文本。需要先调用 browser_navigate 和 browser_snapshot。 | — |
| `browser_vision` | 对当前页面进行截图并使用视觉 AI 分析。当您需要直观理解页面内容时使用此工具 —— 特别是对于 CAPTCHA、可视化验证挑战、复杂布局或文本快照... | — |

## `browser-cdp` 工具集

仅当会话启动时可访问 Chrome DevTools Protocol 端点时注册 —— 通过 `/browser connect`、`browser.cdp_url` 配置、Browserbase 会话或 Camofox。

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `browser_cdp` | 发送原始 Chrome DevTools Protocol 命令。用于浏览器操作未被子工具覆盖时的后备方案。参见 https://chromedevtools.github.io/devtools-protocol/ | CDP 端点 |
| `browser_dialog` | 响应原生 JavaScript 对话框（alert / confirm / prompt / beforeunload）。先调用 `browser_snapshot` —— 待处理对话框会出现在其 `pending_dialogs` 字段中。然后调用 `browser_dialog(action='accept'|'dismiss')`。 | CDP 端点 |

## `clarify` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `clarify` | 当 agent 需要澄清、反馈或在继续之前做决定时向用户提问。支持两种模式：1. **多选** —— 提供最多 4 个选项。用户选择一个或通过第 5 个"其他"选项输入自己的答案。2.... | — |

## `code_execution` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `execute_code` | 运行可以编程方式调用 Hermes 工具的 Python 脚本。当您需要 3+ 个工具调用且调用之间有处理逻辑、需要过滤/缩减大型工具输出后再进入上下文、需要条件分支（… | — |

## `cronjob` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `cronjob` | 统一的计划任务管理器。使用 `action="create"`、`"list"`、`"update"`、`"pause"`、`"resume"`、`"run"` 或 `"remove"` 管理作业。支持附加一个或多个技能的后台技能作业，更新时 `skills=[]` 清除附加技能。Cron 运行在全新会话中进行，无当前聊天上下文。 | — |

## `delegation` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `delegate_task` | 生成一个或多个子 agent 在隔离上下文中处理任务。每个子 agent 获得自己的对话、终端会话和工具集。只返回最终摘要 —— 中间工具结果永远不会进入您的上下文窗口。两个… | — |

## `feishu_doc` 工具集

仅限于 Feishu 文档评论智能回复处理器（`gateway/platforms/feishu_comment.py`）。不在 `hermes-cli` 或常规 Feishu 聊天适配器上暴露。

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `feishu_doc_read` | 读取给定文件类型和 token 的 Feishu/Lark 文档（Docx、Doc 或 Sheet）的完整文本内容。 | Feishu 应用凭证 |

## `feishu_drive` 工具集

仅限于 Feishu 文档评论处理器。驱动云盘文件的评论读写操作。

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `feishu_drive_add_comment` | 在 Feishu/Lark 文档或文件上添加顶级评论。 | Feishu 应用凭证 |
| `feishu_drive_list_comments` | 列出 Feishu/Lark 文件的整文档评论，最新在前。 | Feishu 应用凭证 |
| `feishu_drive_list_comment_replies` | 列出特定 Feishu 评论线程上的回复（整文档或本地选择）。 | Feishu 应用凭证 |
| `feishu_drive_reply_comment` | 在 Feishu 评论线程上发布回复，可选 @ 提及。 | Feishu 应用凭证 |

## `file` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `patch` | 对文件进行定向查找和替换编辑。用此代替终端中的 sed/awk。使用模糊匹配（9 种策略），因此轻微的空格/缩进差异不会破坏它。返回统一 diff。编辑后自动运行语法检查… | — |
| `read_file` | 读取带行号和分页的文本文件。用此代替终端中的 cat/head/tail。输出格式：'LINE_NUM\|CONTENT'。如果未找到文件会建议相似文件名。使用 offset 和 limit 处理大文件。注：无法读取图像… | — |
| `search_files` | 搜索文件内容或按名称查找文件。用此代替终端中的 grep/rg/find/ls。基于 Ripgrep，比 shell 等效命令更快。内容搜索（target='content'）：文件内正则搜索。输出模式：完整匹配及行… | — |
| `write_file` | 将内容写入文件，完全替换现有内容。用此代替终端中的 echo/cat heredoc。自动创建父目录。**覆盖整个文件** —— 定向编辑请使用 'patch'。 | — |

## `homeassistant` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `ha_call_service` | 调用 Home Assistant 服务来控制设备。使用 ha_list_services 发现每个域的可用服务及其参数。 | — |
| `ha_get_state` | 获取单个 Home Assistant 实体的详细状态，包括所有属性（亮度、颜色、温度设定值、传感器读数等）。 | — |
| `ha_list_entities` | 列出 Home Assistant 实体。可按域（light、switch、climate、sensor、binary_sensor、cover、fan 等）或按区域名称（客厅、厨房、卧室等）过滤。 | — |
| `ha_list_services` | 列出可用的 Home Assistant 服务（操作）以控制设备。显示每种设备类型可执行的操作及其接受的参数。用此发现如何控制通过 ha_list_entities 找到的设备。 | — |

:::note
**Honcho 工具**（`honcho_profile`、`honcho_search`、`honcho_context`、`honcho_reasoning`、`honcho_conclude`）不再是内置工具。可通过 `plugins/memory/honcho/` 中的 Honcho 内存提供程序插件获取。参见[内存提供程序](../user-guide/features/memory-providers.md)了解安装和使用方法。
:::

## `image_gen` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `image_generate` | 使用 FAL.ai 从文本提示生成高质量图像。底层模型由用户配置（默认：FLUX 2 Klein 9B，亚秒级生成），agent 无法选择。返回单个图像 URL。使用…显示 | FAL_KEY |

## `memory` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `memory` | 将重要信息保存到持久内存中，在会话之间保持存活。您的内存在会话开始时出现在系统提示中 —— 这是您在对话之间记住用户和环境信息的方式。何时使用… | — |

## `messaging` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `send_message` | 向已连接的消息平台发送消息，或列出可用目标。**重要：**当用户要求发送到特定频道或人员（而不仅仅是裸平台名称）时，先调用 send_message(action='list') 查看可用目标… | — |

## `moa` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `mixture_of_agents` | 通过多个前沿 LLM 协作处理难题。进行 5 次 API 调用（4 个参考模型 + 1 个聚合器），最大推理投入 —— 谨慎使用，仅用于真正困难的问题。最适合：复杂数学、高级算法… | OPENROUTER_API_KEY |

## `rl` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `rl_check_status` | 获取训练运行的状态和指标。速率限制：同一运行检查间隔至少 30 分钟。返回 WandB 指标：step、state、reward_mean、loss、percent_correct。 | TINKER_API_KEY、WANDB_API_KEY |
| `rl_edit_config` | 更新配置字段。先使用 rl_get_current_config() 查看所选环境的所有可用字段。每个环境有不同的可配置选项。基础设施设置（tokenizer、URL、lora_rank、learning_rate 等）由 Tinker 管理。 | TINKER_API_KEY、WANDB_API_KEY |
| `rl_get_current_config` | 获取当前环境配置。仅返回可修改的字段：group_size、max_token_length、total_steps、steps_per_eval、use_wandb、wandb_name、max_num_workers。 | TINKER_API_KEY、WANDB_API_KEY |
| `rl_get_results` | 获取已完成训练运行的最终结果和指标。返回最终指标和训练权重路径。 | TINKER_API_KEY、WANDB_API_KEY |
| `rl_list_environments` | 列出所有可用的 RL 环境。返回环境名称、路径和描述。提示：使用文件工具读取 file_path 以了解每个环境的工作方式（验证器、数据加载、奖励）。 | TINKER_API_KEY、WANDB_API_KEY |
| `rl_list_runs` | 列出所有训练运行（活跃和已完成）及其状态。 | TINKER_API_KEY、WANDB_API_KEY |
| `rl_select_environment` | 选择 RL 环境进行训练。加载环境的默认配置。选择后，使用 rl_get_current_config() 查看设置，使用 rl_edit_config() 修改。 | TINKER_API_KEY、WANDB_API_KEY |
| `rl_start_training` | 使用当前环境和配置启动新的 RL 训练运行。大多数训练参数（lora_rank、learning_rate 等）是固定的。使用 rl_edit_config() 设置 group_size、batch_size、wandb_project 后再启动。警告：训练… | TINKER_API_KEY、WANDB_API_KEY |
| `rl_stop_training` | 停止正在运行的训练作业。如果指标看起来不好、训练停滞或您想尝试不同设置，请使用此工具。 | TINKER_API_KEY、WANDB_API_KEY |
| `rl_test_inference` | 任何环境的快速推理测试。使用 OpenRouter 运行几步推理 + 评分。默认：3 步 x 16 个完成 = 每模型 48 次 rollout，测试 3 个模型 = 144 总计。测试环境加载、提示构建、在… | TINKER_API_KEY、WANDB_API_KEY |

## `session_search` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `session_search` | 在过去的对话长期记忆中搜索。这是您的回忆功能 —— 每个过去的会话都是可搜索的，此工具总结发生了什么。**主动使用此工具**当： - 用户说"我们以前做过这个"、"还记得吗"、"上次…" | — |

## `skills` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `skill_manage` | 管理技能（创建、更新、删除）。技能是您的程序记忆 —— 适用于重复任务类型的可复用方法。新技能放到 ~/.hermes/skills/；现有技能可在其所在位置修改。操作：create（完整 SKILL.m… | — |
| `skill_view` | 技能允许加载关于特定任务和工作流程的信息，以及脚本和模板。加载技能的全部内容或访问其链接的文件（参考资料、模板、脚本）。首次调用返回 SKILL.md 内容加上… | — |
| `skills_list` | 列出可用技能（名称 + 描述）。使用 skill_view(name) 加载完整内容。 | — |

## `terminal` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `process` | 管理使用 terminal(background=true) 启动的后台进程。操作：'list'（显示全部）、'poll'（检查状态 + 新输出）、'log'（带分页的完整输出）、'wait'（阻塞直到完成或超时）、'kill'（终止）、'write'（发送… | — |
| `terminal` | 在 Linux 环境执行 shell 命令。文件系统在调用之间保持持久。设置 `background=true` 用于长时间运行的服务器。设置 `notify_on_complete=true`（配合 `background=true`）可在进程完成时获得自动通知 —— 无需轮询。**不要**使用 cat/head/tail —— 使用 read_file。**不要**使用 grep/rg/find —— 使用 search_files。 | — |

## `todo` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `todo` | 管理当前会话的任务列表。用于 3+ 步的复杂任务或用户提供多个任务时。无参数调用可读取当前列表。写入： - 提供 'todos' 数组创建/更新项目 - merge=… | — |

## `vision` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `vision_analyze` | 使用 AI 视觉分析图像。提供全面描述并回答关于图像内容的特定问题。 | — |

## `web` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `web_search` | 在网络上搜索信息。默认返回最多 5 个结果，包含标题、URL 和描述。接受可选的 `limit`（1-100，默认 5）。查询会传递给配置的后端，因此后端支持时可能使用 `site:domain`、`filetype:pdf`、`intitle:word`、`-term` 和 `"exact phrase"` 等操作符。 | EXA_API_KEY 或 PARALLEL_API_KEY 或 FIRECRAWL_API_KEY 或 TAVILY_API_KEY |
| `web_extract` | 从网页 URL 提取内容。以 markdown 格式返回页面内容。也适用于 PDF URL —— 直接传递 PDF 链接会转换为 markdown 文本。5000 字符以下的页面返回完整 markdown；更大的页面由 LLM 总结。 | EXA_API_KEY 或 PARALLEL_API_KEY 或 FIRECRAWL_API_KEY 或 TAVILY_API_KEY |

## `tts` 工具集

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `text_to_speech` | 将文本转换为语音音频。返回平台作为语音消息传递的 MEDIA: 路径。在 Telegram 上播放为语音气泡，在 Discord/WhatsApp 上作为音频附件。CLI 模式下保存到 ~/voice-memos/。语音和提供商… | — |

## `discord` 工具集

注册在 `hermes-discord` 平台工具集上（仅网关）。使用与消息适配器相同的 bot token。

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `discord` | 阅读并参与 Discord 服务器。操作包括 `search_members`、`fetch_messages`、`send_message`、`react`、`fetch_channel`、`list_channels` 等。 | `DISCORD_BOT_TOKEN` |

## `discord_admin` 工具集

注册在 `hermes-discord` 平台工具集上。审核操作需要 bot 持有匹配的 Discord 权限。

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `discord_admin` | 通过 REST API 管理 Discord 服务器：列出服务器/频道/角色、创建/编辑/删除频道、管理角色授权、超时、踢出和封禁。 | `DISCORD_BOT_TOKEN` + bot 权限 |

## `spotify` 工具集

由捆绑的 `spotify` 插件注册。需要 OAuth token —— 运行 `hermes spotify setup` 一次进行授权。

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `spotify_playback` | 控制 Spotify 播放、检查活跃播放状态或获取最近播放的曲目。 | Spotify OAuth |
| `spotify_devices` | 列出 Spotify Connect 设备或将播放转移到其他设备。 | Spotify OAuth |
| `spotify_queue` | 检查用户的 Spotify 队列或在队列中添加项目。 | Spotify OAuth |
| `spotify_search` | 在 Spotify 目录中搜索曲目、专辑、艺术家、播放列表、节目或剧集。 | Spotify OAuth |
| `spotify_playlists` | 列出、检查、创建、更新和修改 Spotify 播放列表。 | Spotify OAuth |
| `spotify_albums` | 获取 Spotify 专辑元数据或专辑曲目。 | Spotify OAuth |
| `spotify_library` | 列出、保存或移除用户保存的 Spotify 曲目或专辑。 | Spotify OAuth |

## `hermes-yuanbao` 工具集

仅注册在 `hermes-yuanbao` 平台工具集上。元宝是腾讯的聊天应用；这些工具驱动其 DM/群组/贴纸 API。

| 工具 | 描述 | 需要环境 |
|------|-------------|----------------------|
| `yb_query_group_info` | 查询群组基本信息（应用内称为"派/Pai"）：名称、群主、成员数量。 | 元宝凭证 |
| `yb_query_group_members` | 查询群组成员（用于 @ 提及、按名称查找用户、列出机器人）。 | 元宝凭证 |
| `yb_send_dm` | 向群组中的用户发送私信，可选附带媒体文件。 | 元宝凭证 |
| `yb_search_sticker` | 按关键词搜索内置元宝贴纸（TIM 表情）目录。 | 元宝凭证 |
| `yb_send_sticker` | 向当前元宝聊天发送内置贴纸。 | 元宝凭证 |
