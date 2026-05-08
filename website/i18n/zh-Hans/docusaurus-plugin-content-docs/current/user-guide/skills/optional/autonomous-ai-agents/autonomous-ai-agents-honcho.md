---
title: "Honcho"
sidebar_label: "Honcho"
description: "配置和使用 Honcho 记忆功能 — 跨会话用户建模、多配置文件隔离、观察配置、辩证推理、会话摘要和上下文预算强制执行。"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 skill 的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Honcho

配置和使用 Honcho 记忆功能 — 跨会话用户建模、多配置文件隔离、观察配置、辩证推理、会话摘要和上下文预算强制执行。在设置 Honcho、排查记忆问题、使用 Honcho 同行管理配置文件或调优观察、召回和辩证设置时使用。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/autonomous-ai-agents/honcho` 安装 |
| 路径 | `optional-skills/autonomous-ai-agents/honcho` |
| 版本 | `2.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `Honcho`, `Memory`, `Profiles`, `Observation`, `Dialectic`, `User-Modeling`, `Session-Summary` |
| 相关技能 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Honcho Memory for Hermes

Honcho 提供 AI 原生的跨会话用户建模。它跨对话学习用户是谁，并为每个 Hermes 配置文件提供自己的同行身份，同时共享统一的用户视图。

## 使用场景

- 设置 Honcho（云端或自托管）
- 排查记忆不工作/同行不同步的问题
- 创建多配置文件设置，其中每个代理有其自己的 Honcho 同行
- 调优观察、召回、辩证深度或写入频率设置
- 了解 5 个 Honcho 工具的功能及使用时机
- 配置上下文预算和会话摘要注入

## 设置

### 云端（app.honcho.dev）

```bash
hermes honcho setup
# 选择"cloud"，粘贴来自 https://app.honcho.dev 的 API 密钥
```

### 自托管

```bash
hermes honcho setup
# 选择"local"，输入 base URL（例如 http://localhost:8000）
```

参见：https://docs.honcho.dev/v3/guides/integrations/hermes#running-honcho-locally-with-hermes

### 验证

```bash
hermes honcho status    # 显示解析后的配置、连接测试、同行信息
```

## 架构

### 基础上下文注入

当 Honcho 向系统提示词注入上下文时（在 `hybrid` 或 `context` 召回模式下），它按以下顺序组装基础上下文块：

1. **会话摘要** — 迄今为止当前会话的简短摘要（放在首位，以便模型立即获得对话连续性）
2. **用户表示** — Honcho 积累的用户模型（偏好、事实、模式）
3. **AI 同行卡片** — 此 Hermes 配置文件 AI 同行的身份卡片

会话摘要在每个回合开始时由 Honcho 自动生成（当存在先前的会话时）。它为模型提供热启动，而无需重放完整历史。

### 冷/暖提示选择

Honcho 自动选择两种提示策略：

| 条件 | 策略 | 发生什么 |
|-----------|----------|--------------|
| 无先就会话或空表示 | **冷启动** | 轻量级介绍提示；跳过摘要注入；鼓励模型学习用户 |
| 存在表示和/或会话历史 | **暖启动** | 完整基础上下文注入（摘要 → 表示 → 卡片）；更丰富的系统提示词 |

您无需配置此选项 — 它基于会话状态自动运行。

### 同行

Honcho 将对话建模为**同行**之间的交互。Hermes 为每个会话创建两个同行：

- **用户同行**（`peerName`）：代表人类。Honcho 从观察到的消息中构建用户表示。
- **AI 同行**（`aiPeer`）：代表此 Hermes 实例。每个配置文件有其自己的 AI 同行，以便代理形成独立的视图。

### 观察

每个同行有两个观察开关，控制 Honcho 从中学到什么：

| 开关 | 功能 |
|--------|-------------|
| `observeMe` | 观察同行的自身消息（建立自我表示） |
| `observeOthers` | 观察其他同行的消息（建立跨同行理解） |

默认：全部四个开关**开启**（完全双向观察）。

在 `honcho.json` 中按同行配置：

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
| `"directional"`（默认） | 我:开，其他:开 | 我:开，其他:开 | 多代理，完整记忆 |
| `"unified"` | 我:开，其他:关 | 我:关，其他:开 | 单代理，仅用户建模 |

在 [Honcho 仪表盘](https://app.honcho.dev) 中更改的设置会在会话初始化时同步回来 — 服务器端配置优先于本地默认值。

### 会话

Honcho 会话限定消息和观察的位置。策略选项：

| 策略 | 行为 |
|----------|-------------|
| `per-directory`（默认） | 每个工作目录一个会话 |
| `per-repo` | 每个 git 仓库根目录一个会话 |
| `per-session` | 每次 Hermes 运行一个新 Honcho 会话 |
| `global` | 跨所有目录的单一会话 |

手动覆盖：`hermes honcho map my-project-name`

### 召回模式

代理访问 Honcho 记忆的方式：

| 模式 | 自动注入上下文？ | 工具可用？ | 使用场景 |
|------|---------------------|-----------------|----------|
| `hybrid`（默认） | 是 | 是 | 代理决定何时使用工具 vs 自动上下文 |
| `context` | 是 | 否（隐藏） | 最小 token 成本，无工具调用 |
| `tools` | 否 | 是 | 代理显式控制所有记忆访问 |

## 三个独立旋钮

Honcho 的辩证行为由三个独立维度控制。每个都可以独立调优而不影响其他：

### 节拍（何时）

控制辩证和上下文调用发生的**频率**。

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `contextCadence` | `1` | 上下文 API 调用之间的最小回合数 |
| `dialecticCadence` | `2` | 辩证 API 调用之间的最小回合数。推荐 1–5 |
| `injectionFrequency` | `every-turn` | `every-turn` 或 `first-turn`，用于基础上下文注入 |

更高的节拍值使辩证 LLM 触发频率降低。`dialecticCadence: 2` 表示引擎每两个回合触发一次。设置为 `1` 则每个回合都触发。

### 深度（多少轮）

控制 Honcho 每次查询执行的辩证推理**轮数**。

| 键 | 默认值 | 范围 | 描述 |
|-----|---------|-------|-------------|
| `dialecticDepth` | `1` | 1-3 | 每次查询的辩证推理轮数 |
| `dialecticDepthLevels` | -- | 数组 | 可选的每深度轮级别覆盖（见下文） |

`dialecticDepth: 2` 表示 Honcho 运行两轮辩证综合。第一轮产生初始答案；第二轮改进它。

`dialecticDepthLevels` 允许您独立设置每轮的推理级别：

```json
{
  "dialecticDepth": 3,
  "dialecticDepthLevels": ["low", "medium", "high"]
}
```

如果省略 `dialecticDepthLevels`，轮次使用从 `dialecticReasoningLevel`（基础）派生的**比例级别**：

| 深度 | 通过级别 |
|-------|-------------|
| 1 | [基础] |
| 2 | [最小, 基础] |
| 3 | [最小, 基础, 低] |

这使得早期轮次成本低廉，同时在最终综合中使用完整深度。

**会话开始时的深度。** 会话开始预热在后台运行完整配置的 `dialecticDepth`，在第 1 回合之前。冷同行上的单轮预热通常返回较薄的输出 — 多轮深度运行在用户说话之前执行审计/协调循环。第 1 回合直接消耗预热结果；如果预热未及时落地，第 1 回合回退到带有限定超时时间的同步调用。

### 级别（多难）

控制每轮辩证推理的**强度**。

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 当为 `true` 时，模型可以向 `honcho_reasoning` 传递 `reasoning_level` 以覆盖每次调用的默认值。`false` = 始终使用 `dialecticReasoningLevel`，忽略模型覆盖 |

更高的级别产生更丰富的综合，但会在 Honcho 后端消耗更多 token。

## 多配置文件设置

每个 Hermes 配置文件获取其自己的 Honcho AI 同行，同时共享相同的工作区（用户上下文）。这意味着：

- 所有配置文件看到相同的用户表示
- 每个配置文件构建自己的 AI 身份和观察
- 一个配置文件写入的结论通过共享工作区对其他配置文件可见

### 使用 Honcho 同行创建配置文件

```bash
hermes profile create coder --clone
# 创建 hermes.coder 主机块，AI 同行"coder"，从默认配置继承
```

`--clone` 对 Honcho 的作用：
1. 在 `honcho.json` 中创建 `hermes.coder` 主机块
2. 设置 `aiPeer: "coder"`（配置文件名称）
3. 从默认配置继承 `workspace`、`peerName`、`writeFrequency`、`recallMode` 等
4. 热切地在 Honcho 中创建同行，以便在第一条消息之前就存在

### 回填现有配置文件

```bash
hermes honcho sync    # 为所有尚无主机块的配置文件创建主机块
```

### 按配置文件配置

在主机块中覆盖任何设置：

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

代理有 5 个双向 Honcho 工具（在 `context` 召回模式下隐藏）：

| 工具 | LLM 调用？ | 成本 | 使用时机 |
|------|-----------|------|----------|
| `honcho_profile` | 否 | 极小 | 会话开始时的快速事实快照，或用于快速姓名/角色/偏好查找 |
| `honcho_search` | 否 | 低 | 获取特定的过去事实以供自己推理 — 原始摘录，无综合 |
| `honcho_context` | 否 | 低 | 完整会话上下文快照：摘要、表示、卡片、最近消息 |
| `honcho_reasoning` | 是 | 中-高 | 由 Honcho 辩证推理引擎合成的自然语言问题 |
| `honcho_conclude` | 否 | 极小 | 写入或删除持久结论；传递 `peer: "ai"` 用于 AI 自我知识 |

### `honcho_profile`
读取或更新同行卡片 — 精选的关键事实（姓名、角色、偏好、沟通风格）。传递 `card: [...]` 更新；省略以读取。无 LLM 调用。

### `honcho_search`
对存储的上下文进行语义搜索，针对特定同行。返回按相关性排序的原始摘录，无综合。默认 800 token，最大 2000。当您需要特定的过去事实以供自己推理而非综合答案时使用。

### `honcho_context`
来自 Honcho 的完整会话上下文快照 — 会话摘要、同行表示、同行卡片和最近消息。无 LLM 调用。当您想一次性查看 Honcho 知道的关于当前会话和同行的所有内容时使用。

### `honcho_reasoning`
由 Honcho 的辩证推理引擎回答的自然语言问题（Honcho 后端上的 LLM 调用）。成本较高，质量较高。传递 `reasoning_level` 控制深度：`minimal`（快速/廉价）→ `low` → `medium` → `high` → `max`（详尽）。省略以使用配置的默认值（`low`）。用于综合理解用户的模式、目标或当前状态。

### `honcho_conclude`
写入或删除关于同行的持久结论。传递 `conclusion: "..."` 创建。传递 `delete_id: "..."` 删除（用于 PII 删除 — Honcho 会随时间自我修复不正确的结论，因此删除仅在 PII 情况下需要）。您必须准确传递两者之一。

### 双向同行定位

所有 5 个工具接受可选的 `peer` 参数：
- `peer: "user"`（默认）— 操作用户同行
- `peer: "ai"` — 操作此配置文件的 AI 同行
- `peer: "<explicit-id>"` — 工作区中的任何同行 ID

示例：
```
honcho_profile                        # 读取用户的卡片
honcho_profile peer="ai"              # 读取 AI 同行的卡片
honcho_reasoning query="这个用户最关心什么？"
honcho_reasoning query="我的互动模式是什么？" peer="ai" reasoning_level="medium"
honcho_conclude conclusion="偏好简洁的回答"
honcho_conclude conclusion="我倾向于过度解释代码" peer="ai"
honcho_conclude delete_id="abc123"    # PII 删除
```

## 代理使用模式

Honcho 记忆激活时的 Hermes 指南。

### 会话开始时

```
1. honcho_profile                  → 快速预热，无 LLM 成本
2. 如果上下文看起来薄弱 → honcho_context  （完整快照，仍无 LLM）
3. 如果需要深度综合 → honcho_reasoning  （LLM 调用，谨慎使用）
```

不要在每个回合调用 `honcho_reasoning`。自动注入已经处理持续的上下文刷新。仅在您真正需要基础上下文未提供的综合洞察时才使用推理工具。

### 当用户分享需要记住的内容时

```
honcho_conclude conclusion="<具体、可操作的事实>"
```

好的结论："偏好代码示例而非文字解释"、"2026 年 4 月前正在做一个 Rust 异步项目"
差的结论："用户提到了 Rust"（太模糊）、"用户看起来很技术"（已在表示中）

### 当用户询问过去的上下文/需要回忆细节时

```
honcho_search query="<主题>"       → 快速，无 LLM，适合具体事实
honcho_context                       → 带摘要 + 消息的完整快照
honcho_reasoning query="<问题>"  → 综合答案，在搜索不够时使用
```

### 何时使用 `peer: "ai"`

使用 AI 同行定位来构建和查询代理自己的自我知识：
- `honcho_conclude conclusion="我在解释架构时倾向于冗长" peer="ai"` — 自我纠正
- `honcho_reasoning query="我通常如何处理模糊请求？" peer="ai"` — 自我审计
- `honcho_profile peer="ai"` — 审查自己的身份卡片

### 何时不调用工具

在 `hybrid` 和 `context` 模式下，基础上下文（用户表示 + 卡片 + 会话摘要）在每个回合之前自动注入。不要重新获取已注入的内容。仅在以下情况下调用工具：
- 您需要注入上下文没有的内容
- 用户明确要求您回忆或检查记忆
- 您正在写入关于新事物的结论

### 节拍意识

工具端的 `honcho_reasoning` 与自动注入辩证共享相同成本。在显式工具调用之后，自动注入节拍会重置 — 避免在同一回合重复计费。

## 配置参考

配置文件：`$HERMES_HOME/honcho.json`（配置文件本地）或 `~/.honcho/config.json`（全局）。

### 关键设置

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `apiKey` | -- | API 密钥（[获取一个](https://app.honcho.dev)） |
| `baseUrl` | -- | 自托管 Honcho 的基础 URL |
| `peerName` | -- | 用户同行身份 |
| `aiPeer` | 主机键 | AI 同行身份 |
| `workspace` | 主机键 | 共享工作区 ID |
| `recallMode` | `hybrid` | `hybrid`、`context` 或 `tools` |
| `observation` | 全部开启 | 按同行的 `observeMe`/`observeOthers` 布尔值 |
| `writeFrequency` | `async` | `async`、`turn`、`session` 或整数 N |
| `sessionStrategy` | `per-directory` | `per-directory`、`per-repo`、`per-session`、`global` |
| `messageMaxChars` | `25000` | 每条消息的最大字符数（超出时拆分） |

### 辩证设置

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 按查询复杂度自动提升推理。`false` = 固定级别 |
| `dialecticDepth` | `1` | 每次查询的辩证轮数（1-3） |
| `dialecticDepthLevels` | -- | 可选的每轮级别数组，例如 `["low", "high"]` |
| `dialecticMaxInputChars` | `10000` | 辩证查询输入的最大字符数 |

### 上下文预算和注入

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `contextTokens` | 无上限 | 组合基础上下文注入的最大 token 数（摘要 + 表示 + 卡片）。选择性上限 — 省略以保持无上限，设置为整数以限制注入大小。 |
| `injectionFrequency` | `every-turn` | `every-turn` 或 `first-turn` |
| `contextCadence` | `1` | 上下文 API 调用之间的最小回合数 |
| `dialecticCadence` | `2` | 辩证 LLM 调用之间的最小回合数（推荐 1–5） |

`contextTokens` 预算在注入时强制执行。如果会话摘要 + 表示 + 卡片超出预算，Honcho 首先修剪摘要，然后是表示，保留卡片。这可以防止长会话中的上下文膨胀。

### 记忆上下文清理

Honcho 在注入前清理 `memory-context` 块，以防止提示注入和格式错误的内容：

- 从用户编写的结论中剥离 XML/HTML 标签
- 规范化空白和控制字符
- 截断超出 `messageMaxChars` 的单个结论
- 转义可能破坏系统提示词结构的分隔符序列

此修复解决了原始用户结论包含可能损坏注入上下文块的标记或特殊字符的边缘情况。

## 故障排除

### "Honcho 未配置"
运行 `hermes honcho setup`。确保 `memory.provider: honcho` 在 `~/.hermes/config.yaml` 中。

### 记忆不跨会话持久化
检查 `hermes honcho status` — 验证 `saveMessages: true` 且 `writeFrequency` 不是 `session`（后者仅在退出时写入）。

### 配置文件未获得自己的同行
创建时使用 `--clone`：`hermes profile create <name> --clone`。对于现有配置文件：`hermes honcho sync`。

### 仪表盘中的观察更改未反映
观察配置在每个会话初始化时从服务器同步。在 Honcho UI 中更改设置后开始新会话。

### 消息被截断
超过 `messageMaxChars`（默认 25k）的消息会自动拆分并标记 `[continued]`。如果您经常遇到此问题，请检查工具结果或技能内容是否使消息膨胀。

### 上下文注入过大
如果您看到关于上下文预算超限的警告，请降低 `contextTokens` 或减少 `dialecticDepth`。预算紧张时首先修剪会话摘要。

### 会话摘要缺失
会话摘要需要当前 Honcho 会话中至少有一个先前的回合。在冷启动（新会话，无历史）时，摘要被省略，Honcho 使用冷启动提示策略代替。

## CLI 命令

| 命令 | 描述 |
|---------|-------------|
| `hermes honcho setup` | 交互式设置向导（云端/本地、身份、观察、召回、会话） |
| `hermes honcho status` | 显示活动配置文件的解析配置、连接测试、同行信息 |
| `hermes honcho enable` | 为活动配置文件启用 Honcho（如需要则创建主机块） |
| `hermes honcho disable` | 为活动配置文件禁用 Honcho |
| `hermes honcho peer` | 显示或更新同行名称（`--user <name>`、`--ai <name>`、`--reasoning <level>`） |
| `hermes honcho peers` | 显示所有配置文件中的同行身份 |
| `hermes honcho mode` | 显示或设置召回模式（`hybrid`、`context`、`tools`） |
| `hermes honcho tokens` | 显示或设置 token 预算（`--context <N>`、`--dialectic <N>`） |
| `hermes honcho sessions` | 列出已知的目录到会话名称映射 |
| `hermes honcho map <name>` | 将当前工作目录映射到 Honcho 会话名称 |
| `hermes honcho identity` | 为 AI 同行身份种子或显示两个同行表示 |
| `hermes honcho sync` | 为所有尚无主机块的 Hermes 配置文件创建主机块 |
| `hermes honcho migrate` | 从 OpenClaw 原生记忆到 Hermes + Honcho 的分步迁移指南 |
| `hermes memory setup` | 通用记忆提供者选择器（选择"honcho"运行相同的向导） |
| `hermes memory status` | 显示活动记忆提供者和配置 |
| `hermes memory off` | 禁用外部记忆提供者 |
