---
title: "Honcho"
sidebar_label: "Honcho"
description: "配置并使用 Honcho 记忆功能与 Hermes -- 跨会话用户建模、多配置文件 peer 隔离、观察配置、辩证推理、会话摘要及上下文预算控制。"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Honcho

配置并使用 Honcho 记忆功能与 Hermes -- 跨会话用户建模、多配置文件 peer 隔离、观察配置、辩证推理、会话摘要及上下文预算控制。适用于设置 Honcho、排查记忆问题、通过 Honcho peers 管理配置文件，或调整观察、召回和辩证设置。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/autonomous-ai-agents/honcho` 安装 |
| 路径 | `optional-skills/autonomous-ai-agents/honcho` |
| 版本 | `2.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Honcho`, `Memory`, `Profiles`, `Observation`, `Dialectic`, `User-Modeling`, `Session-Summary` |
| 相关 skills | [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Hermes 的 Honcho 记忆

Honcho 提供 AI 原生的跨会话用户建模。它在多次对话中学习用户特征，并为每个 Hermes 配置文件提供独立的 peer 身份，同时共享统一的用户视图。

## 使用场景

- 设置 Honcho（云端或自托管）
- 排查记忆不工作 / peers 未同步的问题
- 创建多配置文件设置，使每个 agent 拥有自己的 Honcho peer
- 调整观察、召回、辩证深度或写入频率设置
- 了解 5 个 Honcho 工具的功能及使用时机
- 配置上下文预算和会话摘要注入

## 设置

### 云端（app.honcho.dev）

```bash
hermes honcho setup
# select "cloud", paste API key from https://app.honcho.dev
```

### 自托管

```bash
hermes honcho setup
# select "local", enter base URL (e.g. http://localhost:8000)
```

参见：https://docs.honcho.dev/v3/guides/integrations/hermes#running-honcho-locally-with-hermes

### 验证

```bash
hermes honcho status    # shows resolved config, connection test, peer info
```

## 架构

### 基础上下文注入

当 Honcho 将上下文注入系统 prompt（在 `hybrid` 或 `context` 召回模式下）时，按以下顺序组装基础上下文块：

1. **会话摘要** -- 当前会话的简短摘要（置于首位，使模型立即获得对话连续性）
2. **用户表示** -- Honcho 积累的用户模型（偏好、事实、行为模式）
3. **AI peer 卡片** -- 此 Hermes 配置文件的 AI peer 身份卡片

会话摘要由 Honcho 在每轮开始时自动生成（当存在先前会话时）。它为模型提供热启动，无需重放完整历史。

### 冷启动 / 热启动 Prompt 选择

Honcho 自动在两种 prompt 策略之间选择：

| 条件 | 策略 | 行为 |
|-----------|----------|--------------|
| 无先前会话或表示为空 | **冷启动** | 轻量级介绍 prompt；跳过摘要注入；鼓励模型了解用户 |
| 存在表示和/或会话历史 | **热启动** | 完整基础上下文注入（摘要 → 表示 → 卡片）；更丰富的系统 prompt |

无需配置此项 -- 它根据会话状态自动选择。

### Peers

Honcho 将对话建模为 **peers** 之间的交互。Hermes 每个会话创建两个 peers：

- **用户 peer**（`peerName`）：代表人类用户。Honcho 从观察到的消息中构建用户表示。
- **AI peer**（`aiPeer`）：代表此 Hermes 实例。每个配置文件拥有自己的 AI peer，使 agents 形成独立视角。

### 观察

每个 peer 有两个观察开关，控制 Honcho 从哪些内容中学习：

| 开关 | 功能 |
|--------|-------------|
| `observeMe` | 观察 peer 自身的消息（构建自我表示） |
| `observeOthers` | 观察其他 peers 的消息（构建跨 peer 理解） |

默认：所有四个开关均**开启**（完全双向观察）。

在 `honcho.json` 中按 peer 配置：

```json
{
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": true, "observeOthers": true }
  }
}
```

或使用简写预设：

| 预设 | 用户 | AI | 使用场景 |
|--------|------|----|----------|
| `"directional"`（默认） | me:on, others:on | me:on, others:on | 多 agent，完整记忆 |
| `"unified"` | me:on, others:off | me:off, others:on | 单 agent，仅用户建模 |

在 [Honcho 控制台](https://app.honcho.dev) 中更改的设置会在会话初始化时同步回来 -- 服务端配置优先于本地默认值。

### 会话

Honcho 会话限定消息和观察的落点。策略选项：

| 策略 | 行为 |
|----------|----------|
| `per-directory`（默认） | 每个工作目录一个会话 |
| `per-repo` | 每个 git 仓库根目录一个会话 |
| `per-session` | 每次 Hermes 运行创建新的 Honcho 会话 |
| `global` | 跨所有目录使用单一会话 |

手动覆盖：`hermes honcho map my-project-name`

### 召回模式

agent 访问 Honcho 记忆的方式：

| 模式 | 自动注入上下文？ | 工具可用？ | 使用场景 |
|------|---------------------|-----------------|----------|
| `hybrid`（默认） | 是 | 是 | agent 自行决定使用工具还是自动上下文 |
| `context` | 是 | 否（隐藏） | 最小 token 消耗，无工具调用 |
| `tools` | 否 | 是 | agent 显式控制所有记忆访问 |

## 三个正交调节维度

Honcho 的辩证行为由三个独立维度控制。每个维度可单独调整，互不影响：

### 节奏（何时）

控制辩证和上下文调用的**频率**。

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `contextCadence` | `1` | 上下文 API 调用之间的最小轮次间隔 |
| `dialecticCadence` | `2` | 辩证 API 调用之间的最小轮次间隔。建议 1–5 |
| `injectionFrequency` | `every-turn` | 基础上下文注入频率：`every-turn` 或 `first-turn` |

节奏值越高，辩证 LLM 触发越少。`dialecticCadence: 2` 表示每隔一轮触发一次。设为 `1` 则每轮触发。

### 深度（多少轮）

控制 Honcho 每次查询执行**多少轮**辩证推理。

| 键 | 默认值 | 范围 | 描述 |
|-----|---------|-------|-------------|
| `dialecticDepth` | `1` | 1-3 | 每次查询的辩证推理轮数 |
| `dialecticDepthLevels` | -- | 数组 | 可选的每轮级别覆盖（见下文） |

`dialecticDepth: 2` 表示 Honcho 运行两轮辩证合成。第一轮产生初始答案，第二轮进行精炼。

`dialecticDepthLevels` 允许为每轮独立设置推理级别：

```json
{
  "dialecticDepth": 3,
  "dialecticDepthLevels": ["low", "medium", "high"]
}
```

若省略 `dialecticDepthLevels`，各轮使用从 `dialecticReasoningLevel`（基准）派生的**比例级别**：

| 深度 | 各轮级别 |
|-------|-------------|
| 1 | [base] |
| 2 | [minimal, base] |
| 3 | [minimal, base, low] |

这使早期轮次成本较低，同时在最终合成时使用完整深度。

**会话开始时的深度。** 会话开始时的预热在第 1 轮之前在后台运行完整配置的 `dialecticDepth`。对冷 peer 进行单轮预热通常返回较薄的输出 -- 多轮深度在用户开口之前运行审计/协调周期。第 1 轮直接消费预热结果；若预热未在时限内完成，第 1 轮将回退到有界超时的同步调用。

### 级别（强度）

控制每轮辩证推理的**强度**。

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 为 `true` 时，模型可向 `honcho_reasoning` 传递 `reasoning_level` 以覆盖每次调用的默认值。`false` = 始终使用 `dialecticReasoningLevel`，忽略模型覆盖 |

级别越高，合成越丰富，但在 Honcho 后端消耗的 token 也越多。

## 多配置文件设置

每个 Hermes 配置文件拥有自己的 Honcho AI peer，同时共享同一工作区（用户上下文）。这意味着：

- 所有配置文件看到相同的用户表示
- 每个配置文件构建自己的 AI 身份和观察
- 一个配置文件写入的结论通过共享工作区对其他配置文件可见

### 创建带 Honcho peer 的配置文件

```bash
hermes profile create coder --clone
# creates host block hermes.coder, AI peer "coder", inherits config from default
```

`--clone` 对 Honcho 的作用：
1. 在 `honcho.json` 中创建 `hermes.coder` host 块
2. 设置 `aiPeer: "coder"`（配置文件名称）
3. 从默认值继承 `workspace`、`peerName`、`writeFrequency`、`recallMode` 等
4. 在 Honcho 中预先创建 peer，使其在第一条消息之前就已存在

### 为现有配置文件补充创建

```bash
hermes honcho sync    # creates host blocks for all profiles that don't have one yet
```

### 按配置文件配置

在 host 块中覆盖任意设置：

```json
{
  "hosts": {
    "hermes.coder": {
      "aiPeer": "coder",
      "recallMode": "tools",
      "dialecticDepth": 2,
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    }
  }
}
```

## 工具

agent 拥有 5 个双向 Honcho 工具（在 `context` 召回模式下隐藏）：

| 工具 | LLM 调用？ | 成本 | 使用时机 |
|------|-----------|------|----------|
| `honcho_profile` | 否 | 极低 | 对话开始时的快速事实快照，或快速查询姓名/角色/偏好 |
| `honcho_search` | 否 | 低 | 获取特定历史事实以自行推理 -- 原始摘录，无合成 |
| `honcho_context` | 否 | 低 | 完整会话上下文快照：摘要、表示、卡片、近期消息 |
| `honcho_reasoning` | 是 | 中–高 | 由 Honcho 辩证引擎合成的自然语言问答 |
| `honcho_conclude` | 否 | 极低 | 写入或删除持久化事实；传递 `peer: "ai"` 用于 AI 自我知识 |

### `honcho_profile`
读取或更新 peer 卡片 -- 精选关键事实（姓名、角色、偏好、沟通风格）。传递 `card: [...]` 进行更新；省略则为读取。无 LLM 调用。

### `honcho_search`
对特定 peer 的存储上下文进行语义搜索。返回按相关性排序的原始摘录，无合成。默认 800 token，最大 2000。适用于需要获取特定历史事实以自行推理而非合成答案的场景。

### `honcho_context`
来自 Honcho 的完整会话上下文快照 -- 会话摘要、peer 表示、peer 卡片和近期消息。无 LLM 调用。适用于一次性查看 Honcho 对当前会话和 peer 所知的全部内容。

### `honcho_reasoning`
由 Honcho 辩证推理引擎（Honcho 后端的 LLM 调用）回答的自然语言问题。成本较高，质量较高。传递 `reasoning_level` 控制深度：`minimal`（快速/低成本）→ `low` → `medium` → `high` → `max`（深度）。省略则使用配置的默认值（`low`）。适用于对用户模式、目标或当前状态的合成理解。

### `honcho_conclude`
写入或删除关于 peer 的持久化结论。传递 `conclusion: "..."` 进行创建。传递 `delete_id: "..."` 删除结论（用于 PII 删除 -- Honcho 会随时间自动修复错误结论，因此删除仅在 PII 场景下需要）。必须且只能传递两者之一。

### 双向 peer 定向

所有 5 个工具接受可选的 `peer` 参数：
- `peer: "user"`（默认）-- 操作用户 peer
- `peer: "ai"` -- 操作此配置文件的 AI peer

无法识别的自由文本值会解析为用户 peer，而不会创建新的 peer，因此模型无法将
同一用户的记忆分散到不同的显示名称变体中。如需有意针对其他 peer，请在 Honcho
配置中通过 `user_peer_aliases` 声明它们。

示例：
```
honcho_profile                        # read user's card
honcho_profile peer="ai"              # read AI peer's card
honcho_reasoning query="What does this user care about most?"
honcho_reasoning query="What are my interaction patterns?" peer="ai" reasoning_level="medium"
honcho_conclude conclusion="Prefers terse answers"
honcho_conclude conclusion="I tend to over-explain code" peer="ai"
honcho_conclude delete_id="abc123"    # PII removal
```

## Agent 使用模式

Honcho 记忆激活时 Hermes 的使用指南。

### 对话开始时

```
1. honcho_profile                  → fast warmup, no LLM cost
2. If context looks thin → honcho_context  (full snapshot, still no LLM)
3. If deep synthesis needed → honcho_reasoning  (LLM call, use sparingly)
```

不要在每轮都调用 `honcho_reasoning`。自动注入已处理持续的上下文刷新。仅在真正需要基础上下文未提供的合成洞察时才使用推理工具。

### 当用户分享需要记住的内容时

```
honcho_conclude conclusion="<specific, actionable fact>"
```

好的结论："Prefers code examples over prose explanations"、"Working on a Rust async project through April 2026"
差的结论："User said something about Rust"（过于模糊）、"User seems technical"（已在表示中）

### 当用户询问历史上下文 / 需要召回具体内容时

```
honcho_search query="<topic>"       → fast, no LLM, good for specific facts
honcho_context                       → full snapshot with summary + messages
honcho_reasoning query="<question>"  → synthesized answer, use when search isn't enough
```

### 何时使用 `peer: "ai"`

使用 AI peer 定向来构建和查询 agent 自身的自我知识：
- `honcho_conclude conclusion="I tend to be verbose when explaining architecture" peer="ai"` -- 自我纠正
- `honcho_reasoning query="How do I typically handle ambiguous requests?" peer="ai"` -- 自我审计
- `honcho_profile peer="ai"` -- 查看自身身份卡片

### 何时不调用工具

在 `hybrid` 和 `context` 模式下，基础上下文（用户表示 + 卡片 + 会话摘要）在每轮之前自动注入。不要重新获取已注入的内容。仅在以下情况调用工具：
- 需要注入上下文中没有的内容
- 用户明确要求召回或检查记忆
- 正在写入关于新内容的结论

### 节奏感知

工具侧的 `honcho_reasoning` 与自动注入辩证的成本相同。显式工具调用后，自动注入节奏重置 -- 避免同一轮被双重计费。

## 配置参考

配置文件：`$HERMES_HOME/honcho.json`（配置文件本地）或 `~/.honcho/config.json`（全局）。

### 关键设置

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `apiKey` | -- | API 密钥（[获取](https://app.honcho.dev)） |
| `baseUrl` | -- | 自托管 Honcho 的 Base URL |
| `peerName` | -- | 用户 peer 身份 |
| `aiPeer` | host 键 | AI peer 身份 |
| `workspace` | host 键 | 共享工作区 ID |
| `recallMode` | `hybrid` | `hybrid`、`context` 或 `tools` |
| `observation` | 全部开启 | 每个 peer 的 `observeMe`/`observeOthers` 布尔值 |
| `writeFrequency` | `async` | `async`、`turn`、`session` 或整数 N |
| `sessionStrategy` | `per-directory` | `per-directory`、`per-repo`、`per-session`、`global` |
| `messageMaxChars` | `25000` | 每条消息的最大字符数（超出时自动分块） |

### 辩证设置

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 根据查询复杂度自动提升推理级别。`false` = 固定级别 |
| `dialecticDepth` | `1` | 每次查询的辩证轮数（1-3） |
| `dialecticDepthLevels` | -- | 可选的每轮级别数组，例如 `["low", "high"]` |
| `dialecticMaxInputChars` | `10000` | 辩证查询输入的最大字符数 |

### 上下文预算与注入

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `contextTokens` | 无上限 | 组合基础上下文注入（摘要 + 表示 + 卡片）的最大 token 数。可选上限 -- 省略则不限，设为整数则限制注入大小。 |
| `injectionFrequency` | `every-turn` | `every-turn` 或 `first-turn` |
| `contextCadence` | `1` | 上下文 API 调用之间的最小轮次间隔 |
| `dialecticCadence` | `2` | 辩证 LLM 调用之间的最小轮次间隔（建议 1–5） |

`contextTokens` 预算在注入时强制执行。若会话摘要 + 表示 + 卡片超出预算，Honcho 优先裁剪摘要，然后裁剪表示，保留卡片。这防止长会话中的上下文膨胀。

### 记忆上下文净化

Honcho 在注入前对 `memory-context` 块进行净化，以防止 prompt 注入和格式错误内容：

- 从用户编写的结论中剥离 XML/HTML 标签
- 规范化空白字符和控制字符
- 截断超过 `messageMaxChars` 的单条结论
- 转义可能破坏系统 prompt 结构的分隔符序列

此修复解决了包含标记或特殊字符的原始用户结论可能损坏注入上下文块的边缘情况。

## 故障排查

### "Honcho not configured"
运行 `hermes honcho setup`。确保 `~/.hermes/config.yaml` 中包含 `memory.provider: honcho`。

### 记忆未跨会话持久化
检查 `hermes honcho status` -- 验证 `saveMessages: true` 且 `writeFrequency` 不是 `session`（该选项仅在退出时写入）。

### 配置文件未获得自己的 peer
创建时使用 `--clone`：`hermes profile create <name> --clone`。对于现有配置文件：`hermes honcho sync`。

### 控制台中的观察更改未生效
观察配置在每次会话初始化时从服务器同步。在 Honcho UI 中更改设置后，启动新会话。

### 消息被截断
超过 `messageMaxChars`（默认 25k）的消息会自动分块并添加 `[continued]` 标记。若频繁触发，检查工具结果或 skill 内容是否导致消息体积膨胀。

### 上下文注入过大
若看到上下文预算超出的警告，降低 `contextTokens` 或减少 `dialecticDepth`。预算紧张时优先裁剪会话摘要。

### 会话摘要缺失
会话摘要需要当前 Honcho 会话中至少有一轮先前记录。冷启动时（新会话，无历史），摘要被省略，Honcho 改用冷启动 prompt 策略。

## CLI 命令

| 命令 | 描述 |
|---------|-------------|
| `hermes honcho setup` | 交互式设置向导（云端/本地、身份、观察、召回、会话） |
| `hermes honcho status` | 显示当前配置文件的已解析配置、连接测试、peer 信息 |
| `hermes honcho enable` | 为当前配置文件启用 Honcho（如需则创建 host 块） |
| `hermes honcho disable` | 为当前配置文件禁用 Honcho |
| `hermes honcho peer` | 显示或更新 peer 名称（`--user <name>`、`--ai <name>`、`--reasoning <level>`） |
| `hermes honcho peers` | 显示所有配置文件的 peer 身份 |
| `hermes honcho mode` | 显示或设置召回模式（`hybrid`、`context`、`tools`） |
| `hermes honcho tokens` | 显示或设置 token 预算（`--context <N>`、`--dialectic <N>`） |
| `hermes honcho sessions` | 列出已知的目录到会话名称映射 |
| `hermes honcho map <name>` | 将当前工作目录映射到 Honcho 会话名称 |
| `hermes honcho identity` | 为 AI peer 身份播种，或显示两个 peer 的表示 |
| `hermes honcho sync` | 为所有尚未拥有 host 块的 Hermes 配置文件创建 host 块 |
| `hermes honcho migrate` | 从 OpenClaw 原生记忆迁移到 Hermes + Honcho 的分步指南 |
| `hermes memory setup` | 通用记忆提供商选择器（选择 "honcho" 运行相同向导） |
| `hermes memory status` | 显示当前活跃的记忆提供商及配置 |
| `hermes memory off` | 禁用外部记忆提供商 |