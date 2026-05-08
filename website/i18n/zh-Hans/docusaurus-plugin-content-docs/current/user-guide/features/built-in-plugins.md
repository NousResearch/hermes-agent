---
sidebar_position: 12
sidebar_label: "内置插件"
title: "内置插件"
description: "随 Hermes Agent 一起提供的插件，通过生命周期钩子自动运行——磁盘清理等"
---

# 内置插件

Hermes 附带了一小组与仓库捆绑的插件。它们位于 `<repo>/plugins/<name>/`，与 `~/.hermes/plugins/` 中的用户安装插件一起自动加载。它们使用与第三方插件相同的插件表面——钩子、工具、斜杠命令——只是在树内维护。

有关通用插件系统，请参阅[插件](/docs/user-guide/features/plugins)页面，以及[构建 Hermes 插件](/docs/guides/build-a-hermes-plugin)来编写你自己的插件。

## 发现工作原理

`PluginManager` 按顺序扫描四个来源：

1. **捆绑** — `<repo>/plugins/<name>/`（本文档记录的内容）
2. **用户** — `~/.hermes/plugins/<name>/`
3. **项目** — `./.hermes/plugins/<name>/`（需要 `HERMES_ENABLE_PROJECT_PLUGINS=1`）
4. **Pip 入口点** — `hermes_agent.plugins`

当名称冲突时，后来的来源优先——名为 `disk-cleanup` 的用户插件将替换捆绑的插件。

`plugins/memory/` 和 `plugins/context_engine/` 被故意排除在捆绑扫描之外。这些目录使用自己的发现路径，因为内存提供者和上下文引擎是通过 `hermes memory setup` / config 中的 `context.engine` 配置的单选提供者。

## 捆绑插件是可选加入的

捆绑插件默认禁用。发现会找到它们（它们出现在 `hermes plugins list` 和交互式 `hermes plugins` UI 中），但在你明确启用之前不会加载：

```bash
hermes plugins enable disk-cleanup
```

或通过 `~/.hermes/config.yaml`：

```yaml
plugins:
  enabled:
    - disk-cleanup
```

这与用户安装插件使用的机制相同。捆绑插件永远不会自动启用——不是在全新安装时，不是为现有用户升级到更新的 Hermes 时。你总是明确选择加入。

再次关闭捆绑插件：

```bash
hermes plugins disable disk-cleanup
# 或：从 config.yaml 中的 plugins.enabled 移除它
```

## 当前提供的

仓库在 `plugins/` 下提供这些捆绑插件。所有都是可选加入的——通过 `hermes plugins enable <name>` 启用。

| 插件 | 类型 | 用途 |
|---|---|---|
| `disk-cleanup` | 钩子 + 斜杠命令 | 自动跟踪临时文件并在会话结束时清理 |
| `observability/langfuse` | 钩子 | 将轮次/LLM 调用/工具跟踪到 [Langfuse](https://langfuse.com) |
| `spotify` | 后端（7 个工具） | 原生 Spotify 播放、队列、搜索、播放列表、专辑、资料库 |
| `google_meet` | 独立 | 加入 Meet 通话、实时字幕转录、可选的实时双工音频 |
| `image_gen/openai` | 图像后端 | OpenAI `gpt-image-2` 图像生成后端（FAL 的替代方案） |
| `image_gen/openai-codex` | 图像后端 | 通过 Codex OAuth 的 OpenAI 图像生成 |
| `image_gen/xai` | 图像后端 | xAI `grok-2-image` 后端 |
| `hermes-achievements` | 仪表板选项卡 | 基于真实 Hermes 会话历史生成的 Steam 风格可收集徽章 |
| `example-dashboard` | 仪表板示例 | [扩展仪表板](./extending-the-dashboard.md) 的参考仪表板插件 |
| `strike-freedom-cockpit` | 仪表板皮肤 | 示例自定义仪表板皮肤 |

内存提供者（`plugins/memory/*`）和上下文引擎（`plugins/context_engine/*`）在[内存提供者](./memory-providers.md)上单独列出——它们分别通过 `hermes memory` 和 `hermes plugins` 管理。以下是两个长期运行的基于钩子插件的完整每个插件详情。

### disk-cleanup

自动跟踪和删除会话期间创建的临时文件——测试脚本、临时输出、cron 日志、过时的 chrome 配置文件——而无需 agent 记住调用工具。

**工作原理：**

| 钩子 | 行为 |
|---|---|
| `post_tool_call` | 当 `write_file` / `terminal` / `patch` 在 `HERMES_HOME` 或 `/tmp/hermes-*` 内创建匹配 `test_*`、`tmp_*` 或 `*.test.*` 的文件时，静默将其跟踪为 `test` / `temp` / `cron-output`。 |
| `on_session_end` | 如果在轮次期间自动跟踪了任何测试文件，运行安全的 `quick` 清理并记录一行摘要。否则保持静默。 |

**删除规则：**

| 类别 | 阈值 | 确认 |
|---|---|---|
| `test` | 每次会话结束 | 从不 |
| `temp` | 跟踪后超过 7 天 | 从不 |
| `cron-output` | 跟踪后超过 14 天 | 从不 |
| HERMES_HOME 下的空目录 | 始终 | 从不 |
| `research` | 超过 30 天，超出 10 个最新 | 始终（深度清理） |
| `chrome-profile` | 跟踪后超过 14 天 | 始终（深度清理） |
| 大于 500 MB 的文件 | 从不自动 | 始终（深度清理） |

**斜杠命令** — `/disk-cleanup` 在 CLI 和 gateway 会话中都可用：

```
/disk-cleanup status                     # 细分 + 前 10 个最大
/disk-cleanup dry-run                    # 预览而不删除
/disk-cleanup quick                      # 现在运行安全清理
/disk-cleanup deep                       # quick + 列出需要确认的项目
/disk-cleanup track <path> <category>    # 手动跟踪
/disk-cleanup forget <path>              # 停止跟踪（不删除）
```

**状态** — 所有内容位于 `$HERMES_HOME/disk-cleanup/`：

| 文件 | 内容 |
|---|---|
| `tracked.json` | 带有类别、大小和时间戳的跟踪路径 |
| `tracked.json.bak` | 上述内容的原子写入备份 |
| `cleanup.log` | 每个跟踪/跳过/拒绝/删除的仅追加审计跟踪 |

**安全** — 清理仅触及 `HERMES_HOME` 或 `/tmp/hermes-*` 下的路径。Windows 挂载（`/mnt/c/...`）被拒绝。众所周知的顶级状态目录（`logs/`、`memories/`、`sessions/`、`cron/`、`cache/`、`skills/`、`plugins/`、`disk-cleanup/` 本身）即使为空也永远不会被删除——全新安装不会在第一次会话结束时被清空。

**启用：** `hermes plugins enable disk-cleanup`（或在 `hermes plugins` 中勾选框）。

**再次禁用：** `hermes plugins disable disk-cleanup`。

### observability/langfuse

将 Hermes 轮次、LLM 调用和工具调用跟踪到 [Langfuse](https://langfuse.com)——一个开源 LLM 可观测性平台。每轮一个 span，每个 API 调用一次生成，每个工具调用一次工具观察。使用总量、每种类型的令牌计数和成本估算来自 Hermes 的规范 `agent.usage_pricing` 数字，因此 Langfuse 仪表板看到与 `hermes logs` 中相同的细分（input / output / `cache_read_input_tokens` / `cache_creation_input_tokens` / `reasoning_tokens`）。

插件是故障开放的：没有安装 SDK、没有凭据或瞬态 Langfuse 错误——所有这些都会在钩子中变成静默无操作。Agent 循环永远不会受到影响。

**设置（交互式——推荐）：**

```bash
hermes tools          # → Langfuse Observability → Cloud or Self-Hosted
```

向导收集你的密钥，`pip install`s `langfuse` SDK，并为你将 `observability/langfuse` 添加到 `plugins.enabled`。重启 Hermes，下一轮将发送一个跟踪。

**设置（手动）：**

```bash
pip install langfuse
hermes plugins enable observability/langfuse
```

然后将凭据放入 `~/.hermes/.env`：

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # 或你的自托管 URL
```

**工作原理：**

| 钩子 | 行为 |
|---|---|
| `pre_api_request` / `pre_llm_call` | 打开（或重用）每轮根 span"Hermes 轮"。为此 API 调用启动一个 `generation` 子观察，序列化的最近消息作为输入。 |
| `post_api_request` / `post_llm_call` | 关闭生成，附加 `usage_details`、`cost_details`、`finish_reason`、助手输出 + 工具调用。如果没有工具调用且内容非空，关闭轮次。 |
| `pre_tool_call` | 使用清理后的 `args` 启动 `tool` 子观察。 |
| `post_tool_call` | 使用清理后的 `result` 关闭工具观察。`read_file` 有效载荷被总结（头 + 尾 + 省略行数），以便大型文件读取保持在 `HERMES_LANGFUSE_MAX_CHARS` 以下。 |

会话按 Hermes 会话 ID（对于子代理为任务 ID）通过 `langfuse.propagate_attributes` 分组，因此单个 `hermes chat` 会话中的所有内容都位于一个 Langfuse 会话下。

**验证：**

```bash
hermes plugins list                 # observability/langfuse 应该显示"enabled"
hermes chat -q "hello"              # 在 Langfuse UI 中检查"Hermes turn"跟踪
```

**可选调优**（在 `.env` 中）：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `HERMES_LANGFUSE_ENV` | — | 跟踪上的环境标签（`production`、`staging`、…） |
| `HERMES_LANGFUSE_RELEASE` | — | 发布/版本标签 |
| `HERMES_LANGFUSE_SAMPLE_RATE` | `1.0` | 传递给 SDK 的采样率（0.0–1.0） |
| `HERMES_LANGFUSE_MAX_CHARS` | `12000` | 消息内容/工具参数/工具结果的单字段截断 |
| `HERMES_LANGFUSE_DEBUG` | `false` | 向 `agent.log` 的详细插件日志记录 |

Hermes 前缀和标准 SDK 环境变量（`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`）都被接受——当两者都设置时，Hermes 前缀优先。

**性能：** Langfuse 客户端在第一次钩子调用后被缓存。如果缺少凭据或 SDK，该决定也被缓存——后续钩子快速返回而不重新检查环境变量或重新加载配置。

**禁用：** `hermes plugins disable observability/langfuse`。插件模块仍被发现，但在你重新启用之前不会运行任何模块代码。

### google_meet

让 agent **加入、转录和参与 Google Meet 通话**——记录会议笔记，总结来回对话，跟进具体要点，并（可选）通过 TTS 将语音回复说回通话中。

**它添加的内容：**

- 使用浏览器自动化加入 Meet URL 的无头虚拟参与者
- 通过配置的 STT 提供商对会议音频进行实时转录
- Agent 调用以对其听到的内容采取行动的 `meet_summarize` / `meet_speak` / `meet_followup` 工具集
- 会议结束后保存到 `~/.hermes/cache/google_meet/<meeting_id>/` 的产物（转录稿、带说话人归属的笔记、行动项目）

**设置：**

```bash
hermes plugins enable google_meet
# 首次使用时提示你通过插件的 OAuth 流程登录——
# 需要一个具有 Meet 访问权限的 Google 账户。如果会议
# 强制执行"仅受邀参与者可以加入"，可能需要主持人批准。
```

从聊天中使用：

> "加入 meet.google.com/abc-defg-hij 并记录笔记。通话结束后，发送给我带有行动项目的摘要。"

Agent 启动会议加入，在通话进行时将转录流式传输到其上下文中，并在会议结束（或你告诉它停止）时生成结构化摘要。

**何时使用：** 你希望机器人转录+总结异步参与者的定期站会；你希望获得结构化笔记的取证风格面试；任何你否则需要 Fireflies / Otter / Grain 的情况。当你不希望 AI 听的时候——不要启用它。

**禁用：** `hermes plugins disable google_meet`。任何缓存的转录稿和录音都保留在 `~/.hermes/cache/google_meet/` 中，直到你移除它们。

### hermes-achievements

向仪表板添加 **Steam 风格的成就选项卡**——基于你的真实 Hermes 会话历史生成 60+ 可收集的分层徽章。工具链壮举、调试模式、vibe-coding 连胜、技能/记忆使用、模型/提供商多样性、生活怪癖（周末和夜间会话）。最初由 [@PCinkusz](https://github.com/PCinkusz) 作为外部插件创作；引入树内以便与 Hermes 功能变化保持同步。

**工作原理：**

- 在仪表板后端扫描你整个 `~/.hermes/state.db` 会话历史
- 每个会话统计通过 `(started_at, last_active)` 指纹缓存，因此只有新的或更改的会话在后续扫描时重新分析
- 首次扫描在后台线程运行——仪表板永远不会阻塞等待它，即使在有数千个会话的数据库上
- 解锁状态持久化到 `$HERMES_HOME/plugins/hermes-achievements/state.json`

**层级进阶：** 铜 → 银 → 金 → 钻石 → 奥运。每个卡片都显示一个"什么算数"部分，列出正在跟踪的确切指标。

**成就状态：**

| 状态 | 含义 |
|---|---|
| 已解锁 | 至少达到一个层级 |
| 已发现 | 已知成就，可见进度，尚未获得 |
| 秘密 | 直到 Hermes 在你的历史中检测到第一个相关信号之前隐藏 |

**API** — 路由挂载在 `/api/plugins/hermes-achievements/`：

| 端点 | 用途 |
|---|---|
| `GET /achievements` | 完整目录，每个徽章带有解锁状态（在首次冷扫描运行时返回待处理占位符） |
| `GET /scan-status` | 后台扫描器状态：`idle` / `running` / `failed`、最后持续时间、运行计数 |
| `GET /recent-unlocks` | 最近解锁的 20 个徽章，最新的在前 |
| `GET /sessions/{id}/badges` | 主要在特定会话中获得的徽章 |
| `POST /rescan` | 手动同步重新扫描（阻塞；当用户点击重新扫描按钮时使用） |
| `POST /reset-state` | 清除解锁历史和缓存快照 |

**状态文件** — 位于 `$HERMES_HOME/plugins/hermes-achievements/`：

| 文件 | 内容 |
|---|---|
| `state.json` | 解锁历史：你获得了哪些徽章以及何时。跨 Hermes 更新稳定。 |
| `scan_snapshot.json` | 上次完成的扫描有效载荷（仪表板加载时立即提供） |
| `scan_checkpoint.json` | 按指纹缓存的每个会话统计（使热重新扫描快速） |

**性能说明：**

- 冷扫描约 8,000 个会话需要几分钟。它在首次仪表板请求时在后台线程运行；UI 看到待处理占位符并轮询 `/scan-status`。
- **冷扫描期间的增量结果** — 扫描器每约 250 个会话发布一个部分快照，以便每次仪表板刷新都显示随着扫描进行解锁的更多徽章。没有盯着零的长等待。
- 热重新扫描为每个 `started_at` + `last_active` 指纹与检查点匹配的会话重用每个会话统计——即使在大历史上也在几秒内完成。
- 内存快照 TTL 为 120 秒；过时请求立即提供旧快照并触发后台刷新。你永远不会因为 TTL 过期而等待旋转器。

**启用：** 无需启用——`hermes-achievements` 是一个仅仪表板插件（无生命周期钩子、无模型可见工具）。它在首次启动时自动注册为 `hermes dashboard` 中的选项卡。`plugins.enabled` 配置仅控制生命周期/工具插件；仪表板插件仅通过其 `dashboard/manifest.json` 发现。

**选择退出：** 删除或重命名 `plugins/hermes-achievements/dashboard/manifest.json`，或者用 `~/.hermes/plugins/hermes-achievements/` 中同名的用户插件覆盖它，该插件不提供仪表板。`$HERMES_HOME/plugins/hermes-achievements/` 下的插件状态文件存活——重新安装会保留你的解锁历史。

## 添加捆绑插件

捆绑插件的编写方式与任何其他 Hermes 插件完全相同——请参阅[构建 Hermes 插件](/docs/guides/build-a-hermes-plugin)。唯一的区别是：

- 目录位于 `<repo>/plugins/<name>/` 而不是 `~/.hermes/plugins/<name>/`
- 清单源在 `hermes plugins list` 中报告为 `bundled`
- 同名用户插件覆盖捆绑版本

插件成为捆绑的良好候选时：

- 它没有可选依赖项（或者它们已经是 `pip install .[all]` 依赖项）
- 该行为使大多数用户受益，并且是选择退出而不是选择加入
- 该逻辑连接到 agent 否则必须记住调用的生命周期钩子
- 它补充了核心功能而不扩展模型可见的工具表面

反例——应该保持为用户可安装插件而不是捆绑的东西：带有 API 密钥的第三方集成、利基工作流、大型依赖树、任何默认情况下会实质性改变 agent 行为的东西。
