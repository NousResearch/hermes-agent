---
sidebar_position: 99
title: "Honcho Memory"
description: "通过 Honcho 实现 AI 原生持久化记忆 —— 辩证推理、多智能体用户建模与深度个性化"
---

# Honcho Memory

[Honcho](https://github.com/plastic-labs/honcho) 是一个 AI 原生记忆后端，它在 Hermes 内置记忆系统之上增加了辩证推理和深度用户建模。与简单的键值存储不同，Honcho 通过在对话结束后进行推理，持续维护一个关于用户是谁的运行模型 —— 包括他们的偏好、沟通风格、目标和模式。

:::info Honcho 是一个 Memory Provider Plugin
Honcho 已集成到 [Memory Providers](./memory-providers.md) 系统中。以下所有功能均可通过统一的 memory provider 接口使用。
:::

## Honcho 带来的增强

| 能力 | 内置 Memory | Honcho |
|-----------|----------------|--------|
| 跨会话持久化 | ✔ 基于文件的 MEMORY.md/USER.md | ✔ 通过 API 在服务端实现 |
| 用户画像 | ✔ 手动 agent 维护 | ✔ 自动辩证推理 |
| 会话摘要 | — | ✔ 会话级上下文注入 |
| 多智能体隔离 | — | ✔ 按 peer 的画像分离 |
| 观察模式 | — | ✔ 统一或定向观察 |
| 结论（派生洞察） | — | ✔ 在服务端对模式进行推理 |
| 历史搜索 | ✔ FTS5 会话搜索 | ✔ 对结论进行语义搜索 |

**Dialectic reasoning**：每次对话轮次结束后（由 `dialecticCadence` 控制），Honcho 会分析对话内容并推导出关于用户偏好、习惯和目标的洞察。这些洞察会随时间累积，使 agent 对用户形成比用户明确陈述内容更深的理解。Dialectic 支持多轮深度（1–3 轮），并自动选择冷启动/温启动 prompt —— 冷启动查询侧重于一般用户事实，而温启动查询优先关注会话级上下文。

**Session-scoped context**：基础上下文现在包含会话摘要，以及用户画像和 peer card。这让 agent 能够感知当前会话中已经讨论过的内容，减少重复并实现连续性。

**Multi-agent profiles**：当多个 Hermes 实例与同一用户对话时（例如编程助手和个人助手），Honcho 会维护独立的 "peer" 画像。每个 peer 只能看到自己的观察和结论，防止上下文交叉污染。

## 设置

```bash
hermes memory setup    # 从 provider 列表中选择 "honcho"
```

或手动配置：

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo 'HONCHO_API_KEY=[REDACTED]' >> ~/.hermes/.env
```

在 [honcho.dev](https://honcho.dev) 获取 API key。

## 架构

### 双层上下文注入

每轮对话（在 `hybrid` 或 `context` 模式下），Honcho 会组装两层上下文注入到 system prompt 中：

1. **Base context** —— 会话摘要、用户画像、用户 peer card、AI 自我画像和 AI identity card。按 `contextCadence` 刷新。这是 "这个用户是谁" 层。
2. **Dialectic supplement** —— LLM 合成的关于用户当前状态和需求的推理。按 `dialecticCadence` 刷新。这是 "目前什么最重要" 层。

两层内容会拼接在一起，并在设置了 `contextTokens` 预算时进行截断。

### 冷启动/温启动 Prompt 选择

Dialectic 会自动在两种 prompt 策略之间选择：

- **冷启动**（尚无 base context）：通用查询 —— "这个人是谁？他们的偏好、目标和工作风格是什么？"
- **温启动会话**（base context 已存在）：会话级查询 —— "根据本次会话到目前为止的讨论内容，关于这个用户最相关的上下文是什么？"

这会根据 base context 是否已填充自动发生。

### 三个正交配置旋钮

成本和深度由三个独立的旋钮控制：

| 旋钮 | 控制内容 | 默认值 |
|------|----------|---------|
| `contextCadence` | 两次 `context()` API 调用之间的轮数（基础层刷新） | `1` |
| `dialecticCadence` | 两次 `peer.chat()` LLM 调用之间的轮数（dialectic 层刷新） | `2`（推荐 1–5） |
| `dialecticDepth` | 每次 dialectic 调用时 `.chat()` 的轮数（1–3） | `1` |

这些选项相互独立 —— 你可以频繁刷新上下文但较少运行 dialectic，或者低频运行深度多轮 dialectic。例如：`contextCadence: 1, dialecticCadence: 5, dialecticDepth: 2` 表示每轮刷新基础上下文，每 5 轮运行一次 dialectic，每次 dialectic 运行 2 轮。

### Dialectic Depth（多轮）

当 `dialecticDepth` > 1 时，每次 dialectic 调用会运行多轮 `.chat()`：

- **Pass 0**：冷启动或温启动 prompt（见上文）
- **Pass 1**：自我审计 —— 识别初始评估中的差距，并从近期会话中综合证据
- **Pass 2**：调和 —— 检查先前轮次之间的矛盾并生成最终综合结果

每轮使用成比例的 reasoning level（早期轮次较轻，主轮次使用基础级别）。使用 `dialecticDepthLevels` 覆盖每轮级别 —— 例如，深度为 3 的运行可设置为 `["minimal", "medium", "high"]`。

如果前一轮返回了强信号（长且结构化的输出），后续轮次会提前退出，因此深度 3 并不总是意味着 3 次 LLM 调用。

### 会话启动预热身

在会话初始化时，Honcho 会在后台以完整配置的 `dialecticDepth` 触发一次 dialectic 调用，并将结果直接交给第 1 轮的上下文组装。对冷 peer 的单轮预热通常返回较薄的输出 —— 多轮深度会在用户开口之前运行审计/调和循环。如果预热在第 1 轮时尚未完成，第 1 轮将回退到带有限制超时的同步调用。

### 查询自适应 Reasoning Level

自动注入的 dialectic 会根据查询长度缩放 `dialecticReasoningLevel`：≥120 字符时 +1 级，≥400 字符时 +2 级，上限为 `reasoningLevelCap`（默认 `"high"`）。将 `reasoningHeuristic` 设为 `false` 可将每次自动调用固定为 `dialecticReasoningLevel`。可用级别：`minimal`、`low`、`medium`、`high`、`max`。

## 配置选项

Honcho 的配置文件位于 `~/.honcho/config.json`（全局）或 `$HERMES_HOME/honcho.json`（profile 本地）。设置向导会为你处理这些配置。

### 完整配置参考

| Key | 默认值 | 说明 |
|-----|---------|-------------|
| `contextTokens` | `null`（无上限） | 每轮自动注入上下文的 token 预算。设为整数（如 1200）可限制。按词边界截断 |
| `contextCadence` | `1` | 两次 `context()` API 调用之间的最小轮数（基础层刷新） |
| `dialecticCadence` | `2` | 两次 `peer.chat()` LLM 调用之间的最小轮数（dialectic 层）。推荐 1–5。在 `tools` 模式下不适用 —— 由模型显式调用 |
| `dialecticDepth` | `1` | 每次 dialectic 调用时 `.chat()` 的轮数。限制在 1–3 之间 |
| `dialecticDepthLevels` | `null` | 可选的每轮 reasoning level 数组，例如 `["minimal", "low", "medium"]`。覆盖默认比例 |
| `dialecticReasoningLevel` | `'low'` | 基础 reasoning level：`minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 为 `true` 时，模型可通过 tool 参数覆盖每次调用的 reasoning level |
| `dialecticMaxChars` | `600` | 注入 system prompt 的 dialectic 结果最大字符数 |
| `recallMode` | `'hybrid'` | `hybrid`（自动注入 + tools）、`context`（仅注入）、`tools`（仅 tools） |
| `writeFrequency` | `'async'` | 消息刷新时机：`async`（后台线程）、`turn`（同步）、`session`（会话结束时批量）或整数 N |
| `saveMessages` | `true` | 是否将消息持久化到 Honcho API |
| `observationMode` | `'directional'` | `directional`（全部开启）或 `unified`（共享池）。使用 `observation` 对象进行细粒度控制 |
| `messageMaxChars` | `25000` | 通过 `add_messages()` 发送的每条消息最大字符数。超出则分块 |
| `dialecticMaxInputChars` | `10000` | dialectic 查询输入到 `peer.chat()` 的最大字符数 |
| `sessionStrategy` | `'per-directory'` | `per-directory`、`per-repo`、`per-session` 或 `global` |

**Session strategy** 控制 Honcho 会话如何映射到你的工作：
- `per-session` —— 每次运行 `hermes` 都会获得一个新会话。干净的开始，通过 tools 访问记忆。推荐给新用户。
- `per-directory` —— 每个工作目录一个 Honcho 会话。上下文在多次运行之间累积。
- `per-repo` —— 每个 git 仓库一个会话。
- `global` —— 所有目录共享一个会话。

**Recall mode** 控制记忆如何流入对话：
- `hybrid` —— 上下文自动注入 system prompt，且 tools 可用（由模型决定何时查询）。
- `context` —— 仅自动注入，隐藏 tools。
- `tools` —— 仅 tools，无自动注入。Agent 必须显式调用 `honcho_reasoning`、`honcho_search` 等。

**各 recall mode 的设置：**

| 设置 | `hybrid` | `context` | `tools` |
|---------|----------|-----------|---------|
| `writeFrequency` | 刷新消息 | 刷新消息 | 刷新消息 |
| `contextCadence` | 控制基础上下文刷新 | 控制基础上下文刷新 | 不适用 —— 无注入 |
| `dialecticCadence` | 控制自动 LLM 调用 | 控制自动 LLM 调用 | 不适用 —— 由模型显式调用 |
| `dialecticDepth` | 每次调用多轮 | 每次调用多轮 | 不适用 —— 由模型显式调用 |
| `contextTokens` | 限制注入 | 限制注入 | 不适用 —— 无注入 |
| `dialecticDynamic` | 控制模型覆盖 | N/A（无 tools） | 控制模型覆盖 |

在 `tools` 模式下，模型完全自主控制 —— 它在需要时调用 `honcho_reasoning`，并自行选择 `reasoning_level`。Cadence 和预算设置仅适用于带有自动注入的模式（`hybrid` 和 `context`）。

## Observation（定向 vs. 统一） {#observation-directional-vs-unified}

Honcho 将对话建模为 peers 交换消息。每个 peer 有两个观察开关，与 Honcho 的 `SessionPeerConfig` 一一对应：

| 开关 | 效果 |
|--------|--------|
| `observeMe` | Honcho 根据该 peer 自己的消息构建其画像 |
| `observeOthers` | 该 peer 观察另一个 peer 的消息（用于跨 peer 推理） |

两个 peers × 两个开关 = 四个标志。`observationMode` 是一个简写预设：

| 预设 | User 标志 | AI 标志 | 语义 |
|--------|-----------|----------|-----------|
| `"directional"`（默认） | me: on, others: on | me: on, others: on | 完全相互观察。支持跨 peer dialectic —— "基于用户所说和 AI 回复的内容，AI 了解用户什么。" |
| `"unified"` | me: on, others: off | me: off, others: on | 共享池语义 —— AI 仅观察用户消息，user peer 仅自我建模。单一观察者池。 |

使用显式的 `observation` 块覆盖预设以实现逐 peer 控制：

```json
"observation": {
  "user": { "observeMe": true,  "observeOthers": true },
  "ai":   { "observeMe": true,  "observeOthers": false }
}
```

常见模式：

| 意图 | 配置 |
|--------|--------|
| 完全观察（大多数用户） | `"observationMode": "directional"` |
| AI 不应从自己的回复中重新建模用户 | `"ai": {"observeMe": true, "observeOthers": false}` |
| AI peer 不应从自我观察中更新强人设 | `"ai": {"observeMe": false, "observeOthers": true}` |

通过 [Honcho dashboard](https://app.honcho.dev) 设置的服务端开关优先于本地默认值 —— Hermes 在会话初始化时会同步它们。

## Tools

当 Honcho 作为活跃的 memory provider 时，以下五个 tools 可用：

| Tool | 用途 |
|------|---------|
| `honcho_profile` | 读取或更新 peer card —— 传入 `card`（事实列表）以更新，省略则读取 |
| `honcho_search` | 对上下文进行语义搜索 —— 原始摘录，无 LLM 合成 |
| `honcho_context` | 完整会话上下文 —— 摘要、画像、card、近期消息 |
| `honcho_reasoning` | 来自 Honcho LLM 的合成回答 —— 传入 `reasoning_level`（minimal/low/medium/high/max）控制深度 |
| `honcho_conclude` | 创建或删除结论 —— 传入 `conclusion` 以创建，`delete_id` 以移除（仅限 PII） |

## CLI 命令

`hermes honcho` 子命令**仅在 Honcho 是活跃 memory provider 时注册**（`config.yaml` 中设置 `memory.provider: honcho`）。先运行 `hermes memory setup` 并选择 Honcho；下次调用时该子命令就会出现。

```bash
hermes honcho status          # 连接状态、配置和关键设置
hermes honcho setup           # 重定向到 `hermes memory setup`
hermes honcho strategy        # 显示或设置 session strategy（per-session/per-directory/per-repo/global）
hermes honcho peer            # 显示或更新 peer 名称 + dialectic reasoning level
hermes honcho mode            # 显示或设置 recall mode（hybrid/context/tools）
hermes honcho tokens          # 显示或设置上下文和 dialectic 的 token 预算
hermes honcho identity        # 设置或显示 AI peer 的 Honcho identity
hermes honcho sync            # 将 Honcho 配置同步到所有现有 profiles
hermes honcho peers           # 显示所有 profiles 的 peer identity
hermes honcho sessions        # 列出已知的 Honcho session 映射
hermes honcho map             # 将当前目录映射到 Honcho session 名称
hermes honcho enable          # 为活跃 profile 启用 Honcho
hermes honcho disable         # 为活跃 profile 禁用 Honcho
hermes honcho migrate         # 从 openclaw-honcho 迁移的分步指南
```

## 从 `hermes honcho` 迁移

如果你之前使用过独立的 `hermes honcho setup`：

1. 你现有的配置（`honcho.json` 或 `~/.honcho/config.json`）会被保留
2. 你的服务端数据（记忆、结论、用户画像）完好无损
3. 在 config.yaml 中设置 `memory.provider: honcho` 即可重新激活

无需重新登录或重新设置。运行 `hermes memory setup` 并选择 "honcho" —— 向导会检测到你现有的配置。

## 完整文档

请参阅 [Memory Providers — Honcho](./memory-providers.md#honcho) 获取完整参考。
