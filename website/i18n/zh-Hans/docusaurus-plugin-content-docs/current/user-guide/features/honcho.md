---
sidebar_position: 99
title: "Honcho 记忆"
description: "通过 Honcho 实现的 AI 原生的持久化记忆 —— 辩证推理、多代理用户建模和深度个性化"
---

# Honcho 记忆

[Honcho](https://github.com/plastic-labs/honcho) 是一个 AI 原生的记忆后端，在 Hermes 内置记忆系统之上增加了辩证推理和用户建模能力。Honcho 不是简单的键值存储，而是通过对话结束后对对话进行推理，持续构建关于用户是谁的模型——包括偏好、沟通风格、目标和行为模式。

:::info Honcho 是一个记忆提供器插件
Honcho 集成在[记忆提供器](./memory-providers.md)系统中。以下所有功能都通过统一的记忆提供器接口可用。
:::

## Honcho 增加了什么

| 能力 | 内置记忆 | Honcho |
|-----------|----------------|--------|
| 跨会话持久化 | ✔ 基于文件的 MEMORY.md/USER.md | ✔ 服务端通过 API |
| 用户画像 | ✔ agent 手动整理 | ✔ 自动辩证推理 |
| 会话摘要 | — | ✔ 会话范围的上下文注入 |
| 多代理隔离 | — | ✔ 按 peer 画像隔离 |
| 观察模式 | — | ✔ 统一或定向观察 |
| 结论（推导洞察） | — | ✔ 服务端推理模式 |
| 跨历史搜索 | ✔ FTS5 会话搜索 | ✔ 对结论的语义搜索 |

**辩证推理**：每一轮对话结束后（受 `dialecticCadence` 节制），Honcho 分析交流内容并推导关于用户偏好、习惯和目标的洞察。这些洞察随时间积累，让 agent 对用户的理解不断加深，超越了用户明确说明的内容。"辩证"支持多遍深度（1-3 遍），通过冷启动/热启动提示选择——冷启动查询侧重用户通用事实，而热启动查询优先会话范围的上下文。

**会话范围的上下文**：基础上下文现在包含了会话摘要以及用户表示、用户 peer 卡片和 AI 自我表示。这让 agent 知道本次会话中已经讨论了什么，减少重复，实现连续性。

**多代理画像**：当多个 Hermes 实例与同一用户对话时（例如编码助手和个人助手），Honcho 维护独立的 "peer" 画像。每个 peer 只看到自己的观察和结论，防止上下文互相污染。

## 设置

```bash
hermes memory setup    # 从提供器列表中选择 "honcho"
```

或手动配置：

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo "HONCHO_API_KEY=*** >> ~/.hermes/.env
```

在 [honcho.dev](https://honcho.dev) 获取 API 密钥。

## 架构

### 双层上下文注入

每一轮（在 `hybrid` 或 `context` 模式下），Honcho 组装两层注入到系统提示中的上下文：

1. **基础上下文** —— 会话摘要、用户表示、用户 peer 卡片、AI 自我表示和 AI 身份卡片。按 `contextCadence` 刷新。这是"这个用户是谁"的层次。
2. **辩证补充** —— LLM 合成的关于用户当前状态和需求的推理。按 `dialecticCadence` 刷新。这是"此刻最相关的内容"的层次。

两层串联并以 `contextTokens` 预算（如设置）为限进行截断。

### 冷启动/热启动提示选择

辩证会自动在两种提示策略之间选择：

- **冷启动**（尚无基础上下文）：通用查询——"这个人是谁？他们的偏好、目标和做事风格是什么？"
- **热会话**（已有基础上下文）：会话范围的查询——"根据此会话到目前为止讨论的内容，关于这个用户最相关的上下文是什么？"

这会根据基础上下文是否已填充自动发生。

### 三个正交配置旋钮

成本和深度由三个独立的旋钮控制：

| 旋钮 | 控制 | 默认值 |
|------|----------|---------|
| `contextCadence` | `context()` API 调用之间的轮次数（基础层刷新） | `1` |
| `dialecticCadence` | `peer.chat()` LLM 调用之间的轮次数（辩证层）。推荐 1-5。在 `tools` 模式下无关——模型显式调用 | `2`（推荐 1-5） |
| `dialecticDepth` | 每次辩证调用的 `.chat()` 遍数（1-3）。限制在 1-3。第 0 遍：冷或热提示（见上）；第 1 遍：自我审计——识别初始评估中的空白并从最近会话中合成证据；第 2 遍：对账——检查前后遍之间的矛盾并产生最终合成。 | `1` |

它们是正交的——你可以让基础上下文频繁刷新但辩证不频繁，或者以低频进行深度多遍辩证。示例：`contextCadence: 1, dialecticCadence: 5, dialecticDepth: 2` 每轮刷新基础上下文，每 5 轮运行一次辩证，每次辩证运行 2 遍。

### 辩证深度（多遍）

当 `dialecticDepth` > 1 时，每次辩证调用运行多个 `.chat()` 遍：

- **第 0 遍**：冷或热提示（见上）
- **第 1 遍**：自我审计——识别初始评估中的空白并从最近会话中合成证据
- **第 2 遍**：对账——检查前后遍之间的矛盾并产生最终合成

每遍使用比例推理级别（较早的遍较轻，主遍为基准级别）。通过 `dialecticDepthLevels` 覆盖每遍的级别——例如 `["minimal", "medium", "high"]` 用于深度 3 运行。

如果前一遍返回了强信号（长且结构化的输出），遍会提前退出，因此深度 3 并不总是意味着 3 次 LLM 调用。

### 会话启动预热

会话初始化时，Honcho 在后台以完整配置的 `dialecticDepth` 发起辩证调用，并将结果直接交给第 1 轮的上下文组装。冷 peer 上的单次预热通常返回较薄的输；多遍深度会在用户说话之前运行审计/对账周期。如果预热在第 1 轮之前尚未完成，第 1 轮回退到带有限定超时的同步调用。

### 查询自适应推理级别

自动注入的辩证会根据查询长度缩放 `dialecticReasoningLevel`：≥120 字符时 +1 级，≥400 时 +2，以 `reasoningLevelCap`（默认 `"high"`）为限截断。通过 `reasoningHeuristic: false` 禁用，以将每次自动调用固定到 `dialecticReasoningLevel`。可用级别：`minimal`, `low`, `medium`, `high`, `max`。

## 配置选项

Honcho 在 `~/.honcho/config.json`（全局）或 `$HERMES_HOME/honcho.json`（profile 本地）中配置。设置向导会自动处理。

### 完整配置参考

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `apiKey` | -- | 来自 [app.honcho.dev](https://app.honcho.dev) 的 API 密钥 |
| `baseUrl` | -- | 自托管的 Honcho 的基础 URL |
| `peerName` | -- | 用户 peer 身份 |
| `aiPeer` | host key | AI peer 身份（每个 profile 一个） |
| `workspace` | host key | 共享工作区 ID |
| `contextTokens` | `null`（无上限） | 每轮自动注入的上下文的令牌预算。设为整数（例如 1200）以限制。在单词边界处截断 |
| `contextCadence` | `1` | `context()` API 调用之间的最小轮次数（基础层刷新） |
| `dialecticCadence` | `2` | `peer.chat()` LLM 调用之间的最小轮次数。推荐 1–5。仅适用于 `hybrid`/`context` 模式 |
| `dialecticDepth` | `1` | 每次辩证调用的 `.chat()` 遍数。限制在 1–3。第 0 遍：冷/热提示，第 1 遍：自我审计，第 2 遍：对账 |
| `dialecticDepthLevels` | `null` | 每遍的可选推理级别数组，例如 `["minimal", "low", "medium"]`。覆盖比例默认值 |
| `dialecticReasoningLevel` | `'low'` | 基准推理级别：`minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | 为 `true` 时，模型可以通过工具参数按调用覆盖推理级别 |
| `dialecticMaxChars` | `600` | 注入到系统提示中的辩证结果的最大字符数 |
| `recallMode` | `'hybrid'` | `hybrid`（自动注入 + 工具），`context`（仅注入），`tools`（仅工具） |
| `writeFrequency` | `'async'` | 何时刷新消息：`async`（后台线程），`turn`（同步），`session`（批处理结束时），或整数 N |
| `saveMessages` | `true` | 是否将消息持久化到 Honcho API |
| `observationMode` | `'directional'` | `directional`（全部开启）或 `unified`（共享池）。使用 `observation` 对象进行细粒度控制 |
| `messageMaxChars` | `25000` | 通过 `add_messages()` 发送的每个消息的最大字符数。如超出则分块 |
| `dialecticMaxInputChars` | `10000` | 对 `peer.chat()` 的辩证查询输入的最大字符数 |
| `sessionStrategy` | `'per-directory'` | `per-directory`，`per-repo`，`per-session`，或 `global` |

**会话策略** 控制 Honcho 会话如何映射到你的工作：

- `per-session`——每次 `hermes` 运行获得一个新的会话。全新启动，通过工具记忆。推荐给新用户。
- `per-directory`——每个工作目录一个 Honcho 会话。跨运行积累上下文。
- `per-repo`——每个 git 仓库一个会话。
- `global`——所有目录的单一会话。

**召回模式** 控制记忆如何流入对话：

- `hybrid`——上下文自动注入到系统提示**并且**工具可用（模型决定何时查询）。
- `context`——仅自动注入，工具隐藏。
- `tools`——仅工具，无自动注入。Agent 必须显式调用 `honcho_reasoning`、`honcho_search` 等。

**每种召回模式的设置：**

| 设置 | `hybrid` | `context` | `tools` |
|---------|----------|-----------|---------|
| `writeFrequency` | 刷新消息 | 刷新消息 | 刷新消息 |
| `contextCadence` | 节制基础上下文刷新 | 节制基础上下文刷新 | 无关——无注入 |
| `dialecticCadence` | 节制自动 LLM 调用 | 节制自动 LLM 调用 | 无关——模型显式调用 |
| `dialecticDepth` | 每次调用的多遍 | 每次调用的多遍 | 无关——模型显式调用 |
| `contextTokens` | 限制注入 | 限制注入 | 无关——无注入 |
| `dialecticDynamic` | 节制模型覆盖 | N/A（无工具） | 节制模型覆盖 |

在 `tools` 模式下，模型完全受控——它在想要时调用 `honcho_reasoning`，使用它选择的任何 `reasoning_level`。节奏和预算设置仅适用于带自动注入的模式（`hybrid` 和 `context`）。

## 观察（定向 vs 统一）

Honcho 将对话建模为交换消息的 peer。每个 peer 有两个观察开关，一对一映射到 Honcho 的 `SessionPeerConfig`：

| 开关 | 效果 |
|--------|--------|
| `observeMe` | Honcho 从其自己的消息中构建此 peer 的表示 |
| `observeOthers` | 此 peer 观察其他 peer 的消息（供给交叉 peer 推理） |

两个 peers × 两个开关 = 四个标志。`observationMode` 是一个预设简写：

| 预设 | 用户标志 | AI 标志 | 语义 |
|--------|-----------|----------|-----------|
| `"directional"`（默认） | me: on, others: on | me: on, others: on | 完全相互观察。启用交叉 peer 辩证——"AI 根据用户的发言和 AI 的回复对用户了解什么？" |
| `"unified"` | me: on, others: off | me: off, others: on | 共享池语义——AI 仅观察用户的消息，用户 peer 仅自我建模。单一观察者池。 |

使用显式的 `observation` 块覆盖预设以进行每 peer 控制：

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
| AI 不应根据其自己的回复重新建模用户 | `"ai": {"observeMe": true, "observeOthers": false}` |
| 强人格的 AI peer 不应从自我观察更新 | `"ai": {"observeMe": false, "observeOthers": true}` |

服务端开关通过 [Honcho 仪表板](https://app.honcho.dev) 设置，优先于本地默认值——Hermes 在会话初始化时将它们同步回来。

## 工具

当 Honcho 作为记忆提供器激活时，五个工具变为可用：

| 工具 | 用途 |
|------|---------|
| `honcho_profile` | 读取或更新 peer 卡片——传递 `card`（事实列表）以更新，省略以读取 |
| `honcho_search` | 对上下文进行语义搜索——原始摘录，无 LLM 合成 |
| `honcho_context` | 完整会话上下文——摘要、表示、卡片、最近消息 |
| `honcho_reasoning` | 来自 Honcho 的 LLM 的合成答案——传递 `reasoning_level`（minimal/low/medium/high/max）以控制深度 |
| `honcho_conclude` | 创建或删除结论——传递 `conclusion` 以创建，`delete_id` 以移除（仅 PII） |

## CLI 命令

```bash
hermes honcho status          # 连接状态、配置和关键设置
hermes honcho setup           # 交互式设置向导
hermes honcho strategy        # 显示或设置会话策略
hermes honcho peer            # 更新多代理设置的 peer 名称
hermes honcho mode            # 显示或设置召回模式
hermes honcho tokens          # 显示或设置上下文令牌预算
hermes honcho identity        # 显示 Honcho peer 身份
hermes honcho sync            # 为所有画像同步 host 块
hermes honcho enable          # 启用 Honcho
hermes honcho disable         # 禁用 Honcho
```

## 从 `hermes honcho` 迁移

如果你之前使用了独立的 `hermes honcho setup`：

1. 你现有的配置（`honcho.json` 或 `~/.honcho/config.json`）被保留
2. 你的服务端数据（记忆、结论、用户画像）完好无损
3. 在 `config.yaml` 中设置 `memory.provider: honcho` 以重新激活

无需重新登录或重新设置。运行 `hermes memory setup` 并选择 "honcho"——向导会检测到你现有的配置。

## 完整文档

参见[记忆提供器——Honcho](./memory-providers.md#honcho)以获取完整参考。
