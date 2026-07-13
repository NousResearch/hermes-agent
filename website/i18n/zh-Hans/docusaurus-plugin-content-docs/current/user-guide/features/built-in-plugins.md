---
sidebar_position: 12
sidebar_label: "内置插件"
title: "内置插件"
description: "随 Hermes Agent 附带并通过生命周期 hook 自动运行的插件——disk-cleanup 等"
---

# 内置插件

Hermes 随仓库附带了一小组插件。它们位于 `<repo>/plugins/<name>/`，与用户安装在 `~/.hermes/plugins/` 中的插件一同自动加载。它们使用与第三方插件相同的插件接口——hook、工具、斜杠命令——只是在仓库内维护。

请参阅 [插件](/user-guide/features/plugins) 页面了解通用插件系统，以及 [构建 Hermes 插件](/developer-guide/plugins) 了解如何编写自己的插件。

## 发现机制

`PluginManager` 按顺序扫描四个来源：

1. **内置（Bundled）** — `<repo>/plugins/<name>/`（本页所记录的内容）
2. **用户（User）** — `~/.hermes/plugins/<name>/`
3. **项目（Project）** — `./.hermes/plugins/<name>/`（需要 `HERMES_ENABLE_PROJECT_PLUGINS=1`）
4. **Pip 入口点（Entry points）** — `hermes_agent.plugins`

名称冲突时，后面的来源优先——名为 `disk-cleanup` 的用户插件会替换内置版本。

`plugins/memory/` 和 `plugins/context_engine/` 被刻意排除在内置扫描之外。这两个目录使用各自的发现路径，因为内存提供者和上下文引擎是通过 `hermes memory setup` / 配置中的 `context.engine` 进行单选配置的提供者。

## 内置插件默认不启用

内置插件随附时处于禁用状态。发现机制会找到它们（它们会出现在 `hermes plugins list` 和交互式 `hermes plugins` UI 中），但在你明确启用之前不会加载：

```bash
hermes plugins enable disk-cleanup
```

或通过 `~/.hermes/config.yaml`：

```yaml
plugins:
  enabled:
    - disk-cleanup
```

这与用户安装的插件使用的机制相同。内置插件永远不会自动启用——无论是全新安装，还是现有用户升级到更新版本的 Hermes，都需要你明确选择启用。

要再次关闭内置插件：

```bash
hermes plugins disable disk-cleanup
# 或：从 config.yaml 的 plugins.enabled 中移除它
```

## 当前附带的插件

仓库在 `plugins/` 下附带了以下内置插件。所有插件均需手动启用——通过 `hermes plugins enable <name>` 启用。

| 插件 | 类型 | 用途 |
|---|---|---|
| `disk-cleanup` | hook + 斜杠命令 | 自动追踪临时文件并在会话结束时清理 |
| `observability/langfuse` | hook | 将轮次 / LLM 调用 / 工具追踪到 [Langfuse](https://langfuse.com) |
| `spotify` | 后端（7 个工具） | 原生 Spotify 播放、队列、搜索、播放列表、专辑、曲库 |
| `google_meet` | 独立插件 | 加入 Meet 通话、实时字幕转录、可选实时双工音频 |
| `image_gen/openai` | 图像后端 | OpenAI `gpt-image-2` 图像生成后端（FAL 的替代方案） |
| `image_gen/openai-codex` | 图像后端 | 通过 Codex OAuth 使用 OpenAI 图像生成 |
| `image_gen/xai` | 图像后端 | xAI `grok-imagine-image` 后端 |
| `image_gen/krea` | 图像后端 | Krea `Krea 2`（Medium + Medium Turbo + Large）图像生成后端 |
| `hermes-achievements` | 仪表盘标签页 | Steam 风格的可收集徽章，根据你真实的 Hermes 会话历史生成 |
| `kanban/dashboard` | 仪表盘标签页 | 多智能体调度器的看板（Kanban）UI——任务、评论、扇出、切换看板。参见 [Kanban 多智能体](./kanban.md)。 |

内存提供者（`plugins/memory/*`）和上下文引擎（`plugins/context_engine/*`）在 [内存提供者](./memory-providers.md) 中单独列出——它们分别通过 `hermes memory` 和 `hermes plugins` 管理。以下是两个长期运行的基于 hook 的插件的详细说明。

### disk-cleanup

自动追踪并删除会话期间创建的临时文件——测试脚本、临时输出、cron 日志、过期的 Chrome 配置文件——无需 agent 记住调用工具。

**工作原理：**

| Hook | 行为 |
|---|---|
| `post_tool_call` | 当 `write_file` / `terminal` / `patch` 在 `HERMES_HOME` 或 `/tmp/hermes-*` 内创建匹配 `test_*`、`tmp_*` 或 `*.test.*` 的文件时，静默追踪为 `test` / `temp` / `cron-output`。 |
| `on_session_end` | 如果本轮中有任何测试文件被自动追踪，则执行安全的 `quick` 清理并记录一行摘要。否则保持静默。 |

**删除规则：**

| 类别 | 阈值 | 确认 |
|---|---|---|
| `test` | 每次会话结束 | 从不 |
| `temp` | 追踪后超过 7 天 | 从不 |
| `cron-output` | 追踪后超过 14 天 | 从不 |
| HERMES_HOME 下的空目录 | 始终 | 从不 |
| `research` | 超过 30 天，且超出最新 10 个 | 始终（仅 deep 模式） |
| `chrome-profile` | 追踪后超过 14 天 | 始终（仅 deep 模式） |
| 超过 500 MB 的文件 | 从不自动删除 | 始终（仅 deep 模式） |

**斜杠命令** — `/disk-cleanup` 在 CLI 和 gateway 会话中均可用：

```
/disk-cleanup status                     # 分类明细 + 最大的 10 个文件
/disk-cleanup dry-run                    # 预览，不实际删除
/disk-cleanup quick                      # 立即执行安全清理
/disk-cleanup deep                       # quick + 列出需要确认的项目
/disk-cleanup track <path> <category>    # 手动追踪
/disk-cleanup forget <path>              # 停止追踪（不删除）
```

**状态** — 所有内容存储在 `$HERMES_HOME/disk-cleanup/`：

| 文件 | 内容 |
|---|---|
| `tracked.json` | 已追踪路径，包含类别、大小和时间戳 |
| `tracked.json.bak` | 上述文件的原子写入备份 |
| `cleanup.log` | 每次追踪 / 跳过 / 拒绝 / 删除操作的仅追加审计日志 |

**安全性** — 清理操作仅涉及 `HERMES_HOME` 或 `/tmp/hermes-*` 下的路径。Windows 挂载点（`/mnt/c/...`）会被拒绝。已知的顶级状态目录（`logs/`、`memories/`、`sessions/`、`cron/`、`cache/`、`skills/`、`plugins/`、`disk-cleanup/` 本身）即使为空也不会被删除——全新安装不会在第一次会话结束时被清空。

**启用：** `hermes plugins enable disk-cleanup`（或在 `hermes plugins` 中勾选复选框）。

**再次禁用：** `hermes plugins disable disk-cleanup`。

### observability/langfuse

将 Hermes 的轮次、LLM 调用和工具调用追踪到 [Langfuse](https://langfuse.com)——一个开源 LLM 可观测性平台。每轮一个 span，每次 API 调用一个 generation，每次工具调用一个 tool observation。用量总计、各类型 token 数量和成本估算来自 Hermes 的标准 `agent.usage_pricing` 数据，因此 Langfuse 仪表盘看到的分类（input / output / `cache_read_input_tokens` / `cache_creation_input_tokens` / `reasoning_tokens`）与 `hermes logs` 中显示的一致。

该插件采用失败开放（fail-open）策略：未安装 SDK、无凭据或 Langfuse 出现瞬时错误——所有情况都会在 hook 中静默处理为无操作。agent 循环不受任何影响。

**设置：**

```bash
pip install langfuse
hermes plugins enable observability/langfuse
```

或在交互式 `hermes plugins` UI 中勾选复选框。然后将凭据写入 `~/.hermes/.env`：

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # 或你的自托管 URL
```

**工作原理：**

| Hook | 行为 |
|---|---|
| `pre_api_request` / `pre_llm_call` | 打开（或复用）每轮的根 span "Hermes turn"。为本次 API 调用启动一个 `generation` 子 observation，将最近的消息序列化为输入。 |
| `post_api_request` / `post_llm_call` | 关闭 generation，附加 `usage_details`、`cost_details`、`finish_reason`、助手输出和工具调用。如果没有工具调用且内容非空，则关闭本轮。 |
| `pre_tool_call` | 启动一个带有经过清理的 `args` 的 `tool` 子 observation。 |
| `post_tool_call` | 关闭 tool observation，附加经过清理的 `result`。`read_file` 的内容会被摘要化（头部 + 尾部 + 省略行数），以使大文件读取保持在 `HERMES_LANGFUSE_MAX_CHARS` 以内。 |

会话分组基于 Hermes 会话 ID（或子 agent 的任务 ID），通过 `langfuse.propagate_attributes` 实现，因此单次 `hermes chat` 会话中的所有内容都归属于同一个 Langfuse session。

**验证：**

```bash
hermes plugins list                 # observability/langfuse 应显示 "enabled"
hermes chat -q "hello"              # 在 Langfuse UI 中检查是否有 "Hermes turn" trace
```

**可选调优**（在 `.env` 中）：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `HERMES_LANGFUSE_ENV` | — | trace 上的环境标签（`production`、`staging` 等） |
| `HERMES_LANGFUSE_RELEASE` | — | 发布/版本标签 |
| `HERMES_LANGFUSE_SAMPLE_RATE` | `1.0` | 传递给 SDK 的采样率（0.0–1.0） |
| `HERMES_LANGFUSE_MAX_CHARS` | `12000` | 消息内容 / 工具参数 / 工具结果的单字段截断长度 |
| `HERMES_LANGFUSE_DEBUG` | `false` | 向 `agent.log` 输出详细插件日志 |

Hermes 前缀的环境变量和标准 SDK 环境变量（`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`）均被接受——两者同时设置时，Hermes 前缀的优先。

**性能：** Langfuse 客户端在第一次 hook 调用后被缓存。如果凭据或 SDK 缺失，该决定也会被缓存——后续 hook 会快速返回，不再重新检查环境变量或重新加载配置。

**禁用：** `hermes plugins disable observability/langfuse`。插件模块仍会被发现，但在你重新启用之前不会运行任何模块代码。

### google_meet

让 agent **加入、转录并参与 Google Meet 通话**——记录会议笔记、事后总结对话内容、跟进特定要点，并可选择通过 TTS 将回复发回通话中。

**新增功能：**

- 使用浏览器自动化加入 Meet URL 的无头虚拟参与者
- 通过配置的 STT 提供者对会议音频进行实时转录
- agent 调用的 `meet_summarize` / `meet_speak` / `meet_followup` 工具集，用于对所听内容采取行动
- 会后产物（转录、带发言人归属的笔记、行动项）保存在 `~/.hermes/cache/google_meet/<meeting_id>/`

**设置：**

```bash
hermes plugins enable google_meet
# 首次使用时会提示你通过插件的 OAuth 流程登录——
# 需要有 Meet 访问权限的 Google 账号。如果会议强制要求
# "仅受邀参与者可加入"，可能需要主持人批准。
```

在聊天中使用：

> "加入 meet.google.com/abc-defg-hij 并记录笔记。通话结束后，给我发一份包含行动项的摘要。"

agent 会启动会议加入流程，在通话进行时将转录内容流式传输到其上下文中，并在会议结束（或你告知停止）时生成结构化摘要。

**适用场景：** 需要机器人转录并为异步参与者总结的定期站会；需要结构化笔记的访谈式会议；任何原本需要 Fireflies / Otter / Grain 的场景。如果你不希望有 AI 在旁监听——请勿启用。

**禁用：** `hermes plugins disable google_meet`。已缓存的转录和录音保留在 `~/.hermes/cache/google_meet/`，直到你手动删除。

### hermes-achievements

在仪表盘中添加一个 **Steam 风格的成就标签页**——60 多个可收集的分级徽章，根据你真实的 Hermes 会话历史生成。工具链成就、调试模式、vibe-coding 连击、技能/内存使用、模型/提供者多样性、生活方式特征（周末和夜间会话）。最初由 [@PCinkusz](https://github.com/PCinkusz) 作为外部插件编写；已并入仓库，以便与 Hermes 功能变更保持同步。

**工作原理：**

- 在仪表盘后端扫描你的整个 `~/.hermes/state.db` 会话历史
- 每个会话的统计数据按 `(started_at, last_active)` 指纹缓存，因此后续扫描只重新分析新增或变更的会话
- 首次扫描在后台线程中运行——即使数据库有数千个会话，仪表盘也不会阻塞等待
- 解锁状态持久化到 `$HERMES_HOME/plugins/hermes-achievements/state.json`

**等级进阶：** 铜 → 银 → 金 → 钻石 → 奥林匹斯。每张卡片都有"计算方式"部分，列出所追踪的确切指标。

**成就状态：**

| 状态 | 含义 |
|---|---|
| 已解锁 | 至少达到一个等级 |
| 已发现 | 已知成就，进度可见，尚未获得 |
| 隐藏 | 在 Hermes 检测到你历史中的第一个相关信号之前保持隐藏 |

**API** — 路由挂载在 `/api/plugins/hermes-achievements/` 下：

| 端点 | 用途 |
|---|---|
| `GET /achievements` | 完整目录，包含每个徽章的解锁状态（首次冷扫描运行期间返回待处理占位符） |
| `GET /scan-status` | 后台扫描器状态：`idle` / `running` / `failed`，上次耗时，运行次数 |
| `GET /recent-unlocks` | 最近解锁的 20 个徽章，最新的在前 |
| `GET /sessions/{id}/badges` | 主要在某个特定会话中获得的徽章 |
| `POST /rescan` | 手动同步重新扫描（阻塞；在用户点击重新扫描按钮时使用） |
| `POST /reset-state` | 清除解锁历史和缓存快照 |

**状态文件** — 位于 `$HERMES_HOME/plugins/hermes-achievements/`：

| 文件 | 内容 |
|---|---|
| `state.json` | 解锁历史：你获得了哪些徽章以及获得时间。在 Hermes 更新间保持稳定。 |
| `scan_snapshot.json` | 上次完成的扫描载荷（在仪表盘加载时立即提供） |
| `scan_checkpoint.json` | 按指纹键控的每会话统计缓存（使热重扫描更快） |

**性能说明：**

- 约 8,000 个会话的冷扫描需要几分钟。它在首次仪表盘请求时在后台线程中运行；UI 显示待处理占位符并轮询 `/scan-status`。
- **冷扫描期间的增量结果** — 扫描器每约 250 个会话发布一次部分快照，因此每次仪表盘刷新都会显示更多已解锁的徽章。不会出现盯着零数字等待一分钟的情况。
- 热重扫描对每个 `started_at` + `last_active` 指纹与检查点匹配的会话复用每会话统计——即使在大型历史记录上也能在几秒内完成。
- 内存快照 TTL 为 120 秒；过期请求立即提供旧快照并触发后台刷新。不会因为 TTL 过期就让你等待加载动画。

**启用：** 无需启用——`hermes-achievements` 是一个仅限仪表盘的插件（无生命周期 hook，无模型可见工具）。它在 `hermes dashboard` 首次启动时自动注册为标签页。`plugins.enabled` 配置仅控制生命周期/工具插件；仪表盘插件完全通过其 `dashboard/manifest.json` 发现。

**退出：** 删除或重命名 `plugins/hermes-achievements/dashboard/manifest.json`，或在 `~/.hermes/plugins/hermes-achievements/` 中用同名用户插件覆盖它（该插件不包含仪表盘）。`$HERMES_HOME/plugins/hermes-achievements/` 下的插件状态文件会保留——重新安装后你的解锁历史依然存在。

## 添加内置插件

内置插件的编写方式与其他 Hermes 插件完全相同——参见 [构建 Hermes 插件](/developer-guide/plugins)。唯一的区别是：

- 目录位于 `<repo>/plugins/<name>/`，而非 `~/.hermes/plugins/<name>/`
- 在 `hermes plugins list` 中，manifest 来源显示为 `bundled`
- 同名用户插件会覆盖内置版本

以下情况适合将插件纳入内置：

- 没有可选依赖项（或它们已经是 `pip install .[all]` 的依赖）
- 该行为对大多数用户有益，且是默认启用、需要主动关闭的
- 逻辑与生命周期 hook 紧密结合，否则 agent 需要记住手动调用
- 在不扩展模型可见工具接口的前提下补充核心能力

反例——应作为用户可安装插件而非内置插件的情况：需要 API 密钥的第三方集成、小众工作流、大型依赖树、任何会默认改变 agent 行为的内容。