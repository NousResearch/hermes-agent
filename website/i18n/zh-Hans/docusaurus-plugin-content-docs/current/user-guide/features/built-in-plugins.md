---
sidebar_position: 12
sidebar_label: "内置插件"
title: "内置插件"
description: "随 Hermes Agent 一起发布的插件，通过生命周期钩子自动运行 —— disk-cleanup 等"
---

# 内置插件

Hermes 附带了一组与仓库捆绑的小型插件。它们位于 `<repo>/plugins/<name>/` 下，与 `~/.hermes/plugins/` 中的用户安装插件一起自动加载。它们使用与第三方插件相同的插件表面 —— 钩子、工具、斜杠命令 —— 只是在仓库内维护。

有关通用插件系统，请参阅 [插件](/user-guide/features/plugins) 页面，有关编写您自己的插件，请参阅 [构建 Hermes 插件](/guides/build-a-hermes-plugin)。

## 发现机制

`PluginManager` 按顺序扫描四个来源：

1. **捆绑** — `<repo>/plugins/<name>/`（本页文档的内容）
2. **用户** — `~/.hermes/plugins/<name>/`
3. **项目** — `./.hermes/plugins/<name>/`（需要 `HERMES_ENABLE_PROJECT_PLUGINS=1`）
4. **Pip 入口点** — `hermes_agent.plugins`

在名称冲突时，后面的来源获胜 —— 名为 `disk-cleanup` 的用户插件将替换捆绑的版本。

`plugins/memory/` 和 `plugins/context_engine/` 被故意从捆绑扫描中排除。这些目录使用自己的发现路径，因为记忆提供者和上下文引擎是通过 `hermes memory setup` / config 中的 `context.engine` 配置的单选提供者。

## 捆绑插件是可选的

捆绑插件默认禁用。发现会找到它们（它们出现在 `hermes plugins list` 和交互式 `hermes plugins` UI 中），但在您显式启用之前都不会加载：

```bash
hermes plugins enable disk-cleanup
```

或通过 `~/.hermes/config.yaml`：

```yaml
plugins:
  enabled:
    - disk-cleanup
```

这与用户安装插件使用的机制相同。捆绑插件永远不会自动启用 —— 无论是在全新安装时，还是现有用户升级到较新的 Hermes 时。您始终显式选择加入。

要再次关闭捆绑插件：

```bash
hermes plugins disable disk-cleanup
# 或：从 config.yaml 中的 plugins.enabled 移除它
```

## 当前已发布

仓库在 `plugins/` 下发布了这些捆绑插件。所有都是可选的 —— 通过 `hermes plugins enable <name>` 启用它们。

| 插件 | 类型 | 用途 |
|---|---|---|
| `disk-cleanup` | 钩子 + 斜杠命令 | 自动跟踪临时文件并在会话结束时清理它们 |
| `observability/langfuse` | 钩子 | 将轮次 / LLM 调用 / 工具追踪到 [Langfuse](https://langfuse.com) |
| `spotify` | 后端（7 个工具） | 原生 Spotify 播放、队列、搜索、播放列表、专辑、库 |
| `google_meet` | 独立 | 加入 Meet 通话、实时字幕转录、可选实时双工音频 |
| `image_gen/openai` | 图像后端 | OpenAI `gpt-image-2` 图像生成后端（FAL 的替代方案） |
| `image_gen/openai-codex` | 图像后端 | 通过 Codex OAuth 的 OpenAI 图像生成 |
| `image_gen/xai` | 图像后端 | xAI `grok-2-image` 后端 |
| `hermes-achievements` | 仪表板标签页 | 从您的真实 Hermes 会话历史生成的 Steam 风格可收集徽章 |
| `kanban/dashboard` | 仪表板标签页 | 多智能体调度器的看板 UI —— 任务、评论、扇出、看板切换。请参阅 [Kanban 多智能体](./kanban.md)。 |

记忆提供者（`plugins/memory/*`）和上下文引擎（`plugins/context_engine/*`）在 [记忆提供者](./memory-providers.md) 上单独列出 —— 它们分别通过 `hermes memory` 和 `hermes plugins` 管理。下面详细介绍两个长期运行的基于钩子的插件。

### disk-cleanup

自动跟踪和移除会话期间创建的临时文件 —— 测试脚本、临时输出、cron 日志、过时的 chrome 配置文件 —— 无需智能体记住调用工具。

**工作原理：**

| 钩子 | 行为 |
|---|---|
| `post_tool_call` | 当 `write_file` / `terminal` / `patch` 在 `HERMES_HOME` 或 `/tmp/hermes-*` 内创建匹配 `test_*`、`tmp_*` 或 `*.test.*` 的文件时，静默将其跟踪为 `test` / `temp` / `cron-output`。 |
| `on_session_end` | 如果本轮期间有任何测试文件被自动跟踪，则运行安全的 `quick` 清理并记录一行摘要。否则保持静默。 |

**删除规则：**

| 类别 | 阈值 | 确认 |
|---|---|---|
| `test` | 每次会话结束 | 从不 |
| `temp` | 跟踪后 >7 天 | 从不 |
| `cron-output` | 跟踪后 >14 天 | 从不 |
| HERMES_HOME 下的空目录 | 始终 | 从不 |
| `research` | >30 天，超过 10 个最新 | 始终（仅限深度清理） |
| `chrome-profile` | 跟踪后 >14 天 | 始终（仅限深度清理） |
| >500 MB 的文件 | 从不自动 | 始终（仅限深度清理） |

**斜杠命令** —— `/disk-cleanup` 在 CLI 和网关会话中都可用：

```
/disk-cleanup status                     # 明细 + 前 10 个最大文件
/disk-cleanup dry-run                    # 预览而不删除
/disk-cleanup quick                      # 立即运行安全清理
/disk-cleanup deep                       # quick + 列出需要确认的项目
/disk-cleanup track <path> <category>    # 手动跟踪
/disk-cleanup forget <path>              # 停止跟踪（不删除）
```

**状态** —— 所有内容位于 `$HERMES_HOME/disk-cleanup/`：

| 文件 | 内容 |
|---|---|
| `tracked.json` | 带有类别、大小和时间戳的跟踪路径 |
| `tracked.json.bak` | 上述文件的原子写入备份 |
| `cleanup.log` | 每次跟踪 / 跳过 / 拒绝 / 删除的仅追加审计追踪 |

**安全性** —— 清理只触及 `HERMES_HOME` 或 `/tmp/hermes-*` 下的路径。Windows 挂载（`/mnt/c/...`）被拒绝。众所周知的顶级状态目录（`logs/`、`memories/`、`sessions/`、`cron/`、`cache/`、`skills/`、`plugins/`、`disk-cleanup/` 本身）即使为空也永远不会被移除 —— 全新安装不会在第一次会话结束时被清空。

**启用：** `hermes plugins enable disk-cleanup`（或在 `hermes plugins` 中勾选复选框）。

**再次禁用：** `hermes plugins disable disk-cleanup`。

### observability/langfuse

将 Hermes 轮次、LLM 调用和工具调用追踪到 [Langfuse](https://langfuse.com) —— 一个开源的 LLM 可观测性平台。每个轮次一个 span，每次 API 调用一个 generation，每次工具调用一个 tool observation。使用总量、每类型令牌计数和成本估算来自 Hermes 规范的 `agent.usage_pricing` 数字，因此 Langfuse 仪表板看到的细分（input / output / `cache_read_input_tokens` / `cache_creation_input_tokens` / `reasoning_tokens`）与 `hermes logs` 中出现的相同。

该插件是故障开放的：未安装 SDK、没有凭证或瞬态 Langfuse 错误 —— 所有这些都变成钩子中的静默无操作。智能体循环永远不会受到影响。

**设置（交互式 —— 推荐）：**

```bash
hermes tools          # → Langfuse Observability → Cloud 或 Self-Hosted
```

向导收集您的密钥，`pip install` `langfuse` SDK，并为您将 `observability/langfuse` 添加到 `plugins.enabled`。重启 Hermes，下一轮就会发送一个 trace。

**设置（手动）：**

```bash
pip install langfuse
hermes plugins enable observability/langfuse
```

然后将凭证放入 `~/.hermes/.env`：

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # 或您的自托管 URL
```

**工作原理：**

| 钩子 | 行为 |
|---|---|
| `pre_api_request` / `pre_llm_call` | 打开（或重用）每个轮次的根 span "Hermes turn"。为这次 API 调用启动一个 `generation` 子 observation，序列化的最近消息作为输入。 |
| `post_api_request` / `post_llm_call` | 关闭 generation，附加 `usage_details`、`cost_details`、`finish_reason`、助手输出 + 工具调用。如果没有工具调用且内容非空，则关闭轮次。 |
| `pre_tool_call` | 启动一个带有脱敏 `args` 的 `tool` 子 observation。 |
| `post_tool_call` | 关闭带有脱敏 `result` 的 tool observation。`read_file` 负载被摘要（头部 + 尾部 + 省略行数），因此大文件读取保持在 `HERMES_LANGFUSE_MAX_CHARS` 以下。 |

会话分组通过 `langfuse.propagate_attributes` 使用 Hermes 会话 ID（子智能体使用任务 ID），因此单个 `hermes chat` 会话中的所有内容都位于一个 Langfuse 会话下。

**验证：**

```bash
hermes plugins list                 # observability/langfuse 应显示 "enabled"
hermes chat -q "hello"              # 检查 Langfuse UI 中是否有 "Hermes turn" trace
```

**可选调优**（在 `.env` 中）：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `HERMES_LANGFUSE_ENV` | — | trace 上的环境标签（`production`、`staging`、…） |
| `HERMES_LANGFUSE_RELEASE` | — | 发布/版本标签 |
| `HERMES_LANGFUSE_SAMPLE_RATE` | `1.0` | 传递给 SDK 的采样率（0.0–1.0） |
| `HERMES_LANGFUSE_MAX_CHARS` | `12000` | 消息内容 / 工具参数 / 工具结果的每字段截断 |
| `HERMES_LANGFUSE_DEBUG` | `false` | 向 `agent.log` 输出详细插件日志 |

Hermes 前缀和标准 SDK 环境变量（`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`）都被接受 —— 当两者都设置时，Hermes 前缀优先。

**性能：** Langfuse 客户端在第一次钩子调用后缓存。如果凭证或 SDK 缺失，该决定也会被缓存 —— 后续钩子快速返回，无需重新检查环境变量或重新加载配置。

**禁用：** `hermes plugins disable observability/langfuse`。插件模块仍然被发现，但在您重新启用之前不会运行任何模块代码。

### google_meet

让智能体**加入、转录和参与 Google Meet 通话** —— 做会议笔记、事后总结来回对话、跟进具体要点，以及（可选地）通过 TTS 将回复说回通话中。

**它添加了什么：**

- 一个使用浏览器自动化加入 Meet URL 的无头虚拟参与者
- 通过配置的 STT 提供商进行会议音频的实时转录
- 智能体调用的 `meet_summarize` / `meet_speak` / `meet_followup` 工具集，用于对其听到的内容采取行动
- 会议后产物（转录、按发言者分类的笔记、行动项）保存在 `~/.hermes/cache/google_meet/<meeting_id>/` 下

**设置：**

```bash
hermes plugins enable google_meet
# 首次使用时提示您通过插件的 OAuth 流程登录 ——
# 需要具有 Meet 访问权限的 Google 账户。如果会议强制执行
# "仅受邀参与者可以加入"，可能需要主持人批准。
```

从聊天中使用：

> "加入 meet.google.com/abc-defg-hij 并做笔记。通话结束后，给我发一份带行动项的摘要。"

智能体启动会议加入，在通话进行时实时将转录流回其上下文，并在会议结束时（或当您告诉它停止时）生成结构化摘要。

**何时使用：** 您希望机器人转录 + 摘要的定期站会；您希望结构化笔记的取证式访谈；任何您本来需要 Fireflies / Otter / Grain 的情况。当您不希望有 AI 在旁听时 —— 不要启用它。

**禁用：** `hermes plugins disable google_meet`。任何缓存的转录和录音会保留在 `~/.hermes/cache/google_meet/` 中，直到您移除它们。

### hermes-achievements

在仪表板上添加一个 **Steam 风格的成就标签页** —— 60+ 个可从您的真实 Hermes 会话历史生成的可收集、分层徽章。工具链壮举、调试模式、氛围编码连胜、技能/记忆使用、模型/提供商多样性、生活方式怪癖（周末和夜间会话）。最初由 [@PCinkusz](https://github.com/PCinkusz) 作为外部插件编写；引入仓库内以便与 Hermes 功能变更保持同步。

**工作原理：**

- 在仪表板后端扫描您的整个 `~/.hermes/state.db` 会话历史
- 每会话统计按 `(started_at, last_active)` 指纹缓存，因此只有新的或更改的会话在后续扫描时重新分析
- 首次扫描在后台线程中运行 —— 仪表板永远不会等待它，即使在有数千个会话的数据库上
- 解锁状态持久保存到 `$HERMES_HOME/plugins/hermes-achievements/state.json`

**层级进阶：** Copper → Silver → Gold → Diamond → Olympian。每张卡片都暴露一个"What counts"部分，列出正在跟踪的确切指标。

**成就状态：**

| 状态 | 含义 |
|---|---|
| Unlocked | 至少达到一个层级 |
| Discovered | 已知成就，进度可见，尚未获得 |
| Secret | 隐藏，直到 Hermes 在您的历史中检测到第一个相关信号 |

**API** —— 路由挂载在 `/api/plugins/hermes-achievements/` 下：

| 端点 | 用途 |
|---|---|
| `GET /achievements` | 带有每个徽章解锁状态的完整目录（首次冷扫描运行时返回待处理占位符） |
| `GET /scan-status` | 后台扫描器的状态：`idle` / `running` / `failed`，上次持续时间，运行次数 |
| `GET /recent-unlocks` | 二十个最近解锁的徽章，最新的在前 |
| `GET /sessions/{id}/badges` | 主要在一个特定会话中获得的徽章 |
| `POST /rescan` | 手动同步重新扫描（阻塞；当用户点击重新扫描按钮时使用） |
| `POST /reset-state` | 清除解锁历史和缓存快照 |

**状态文件** —— 位于 `$HERMES_HOME/plugins/hermes-achievements/` 下：

| 文件 | 内容 |
|---|---|
| `state.json` | 解锁历史：您获得了哪些徽章以及何时获得。在 Hermes 更新之间稳定。 |
| `scan_snapshot.json` | 上次完成的扫描负载（仪表板加载时立即提供） |
| `scan_checkpoint.json` | 按指纹键控的每会话统计缓存（使热重新扫描快速） |

**性能说明：**

- 约 8,000 个会话的冷扫描需要几分钟。它在第一次仪表板请求时在后台线程中运行；UI 看到待处理占位符并轮询 `/scan-status`。
- **冷扫描期间的增量结果** —— 扫描器每约 250 个会话发布一个部分快照，因此每次仪表板刷新都会随着扫描进度显示更多已解锁的徽章。不会有盯着零看一分钟的情况。
- 热重新扫描为每个 `started_at` + `last_active` 指纹与检查点匹配的会话重用每会话统计 —— 即使在大型历史记录上也能在几秒钟内完成。
- 内存快照 TTL 为 120 秒；过期请求立即提供旧快照并启动后台刷新。您永远不会因为 TTL 过期而等待加载指示器。

**启用：** 无需启用 —— `hermes-achievements` 是一个仅仪表板的插件（无生命周期钩子，无模型可见工具）。它在首次启动时自动注册为 `hermes dashboard` 中的标签页。`plugins.enabled` 配置只控制生命周期/工具插件；仪表板插件纯粹通过其 `dashboard/manifest.json` 发现。

**退出：** 删除或重命名 `plugins/hermes-achievements/dashboard/manifest.json`，或在 `~/.hermes/plugins/hermes-achievements/` 中用同名的用户插件覆盖它，该插件不提供仪表板。插件在 `$HERMES_HOME/plugins/hermes-achievements/` 下的状态文件会保留 —— 重新安装会保留您的解锁历史。

## 添加捆绑插件

捆绑插件的编写方式与任何其他 Hermes 插件完全相同 —— 请参阅 [构建 Hermes 插件](/guides/build-a-hermes-plugin)。唯一的区别是：

- 目录位于 `<repo>/plugins/<name>/` 而不是 `~/.hermes/plugins/<name>/`
- 清单来源在 `hermes plugins list` 中报告为 `bundled`
- 同名用户插件会覆盖捆绑版本

当以下情况时，插件是捆绑的良好候选：

- 它没有可选依赖项（或它们已经是 `pip install .[all]` 的依赖项）
- 该行为使大多数用户受益，并且是可选退出而非可选加入
- 逻辑与智能体否则必须记住调用的生命周期钩子相关联
- 它补充了核心能力，而不会扩展模型可见的工具表面

反例 —— 应保持为用户可安装插件而非捆绑的内容：需要 API 密钥的第三方集成、小众工作流、大型依赖树、任何会默认显著改变智能体行为的内容。
