---
sidebar_position: 99
title: "Honcho Memory"
description: "通过 Honcho 实现 AI 原生持久记忆——辩证推理、多智能体用户建模与深度个性化"
---

# Honcho Memory

[Honcho](https://github.com/plastic-labs/honcho) 是一个 AI 原生记忆后端，在 Hermes 内置记忆系统之上增加了辩证推理（dialectic reasoning）和深度用户建模能力。它不是简单的键值存储，而是通过对对话事后推理，持续维护一个关于用户的动态模型——涵盖其偏好、沟通风格、目标与行为模式。

:::info Honcho 是一个 Memory Provider 插件
Honcho 已集成到 [Memory Providers](./memory-providers.md) 系统中。以下所有功能均可通过统一的 memory provider 接口使用。
:::

## Honcho 新增了什么

| 能力 | 内置记忆 | Honcho |
|-----------|----------------|--------|
| 跨会话持久化 | ✔ 基于文件的 MEMORY.md/USER.md | ✔ 服务端 API |
| 用户画像 | ✔ 手动 agent 维护 | ✔ 自动辩证推理 |
| 会话摘要 | — | ✔ 会话级上下文注入 |
| 多 agent 隔离 | — | ✔ 按 peer 分离画像 |
| 观察模式 | — | ✔ 统一或定向观察 |
| 结论（派生洞察） | — | ✔ 服务端模式推理 |
| 历史搜索 | ✔ FTS5 会话搜索 | ✔ 基于结论的语义搜索 |

**辩证推理**：每轮对话后（由 `dialecticCadence` 控制频率），Honcho 分析交流内容，推导出关于用户偏好、习惯和目标的洞察。这些洞察随时间积累，使 agent 对用户的理解不断加深，超越用户明确表述的内容。辩证过程支持多轮深度（1–3 轮），并自动选择冷启动/热启动 prompt——冷启动查询聚焦于通用用户事实，热启动查询优先处理会话级上下文。

**会话级上下文**：基础上下文现在包含会话摘要，以及用户表示和 peer 卡片。这使 agent 能感知当前会话中已讨论的内容，减少重复并保持连贯性。

**多 agent 画像**：当多个 Hermes 实例与同一用户交互时（例如编程助手和个人助手），Honcho 为每个 peer 维护独立画像。每个 peer 只能看到自己的观察和结论，防止上下文交叉污染。

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
echo 'HONCHO_API_KEY=***' >> ~/.hermes/.env
```

在 [honcho.dev](https://honcho.dev) 获取 API key。

## 架构

### 双层上下文注入

每轮对话（在 `hybrid` 或 `context` 模式下），Honcho 组装两层上下文注入到系统 prompt 中：

1. **基础上下文** — 会话摘要、用户表示、用户 peer 卡片、AI 自我表示和 AI 身份卡片。按 `contextCadence` 刷新。这是"这个用户是谁"层。
2. **辩证补充** — LLM 合成的关于用户当前状态和需求的推理。按 `dialecticCadence` 刷新。这是"当前最重要的是什么"层。

两层内容拼接后，按 `contextTokens` 预算截断（如已设置）。

### 冷启动/热启动 Prompt 选择

辩证过程自动在两种 prompt 策略之间切换：

- **冷启动**（尚无基础上下文）：通用查询——"这个人是谁？他们的偏好、目标和工作方式是什么？"
- **热启动会话**（已有基础上下文）：会话级查询——"结合本次会话已讨论的内容，关于该用户哪些上下文最相关？"

是否已填充基础上下文决定了自动选择哪种策略。

### 三个正交配置旋钮

成本和深度由三个独立旋钮控制：

| 旋钮 | 控制内容 | 默认值 |
|------|----------|---------|
| `contextCadence` | `context()` API 调用之间的最小轮数（基础层刷新） | `1` |
| `dialecticCadence` | `peer.chat()` LLM 调用之间的最小轮数（辩证层刷新） | `2`（推荐 1–5） |
| `dialecticDepth` | 每次辩证调用的 `.chat()` 轮数（1–3） | `1` |

三者相互独立——可以频繁刷新上下文而不频繁运行辩证，也可以低频运行深度多轮辩证。示例：`contextCadence: 1, dialecticCadence: 5, dialecticDepth: 2` 表示每轮刷新基础上下文，每 5 轮运行一次辩证，每次辩证运行 2 轮。

### 辩证深度（多轮）

当 `dialecticDepth` > 1 时，每次辩证调用运行多轮 `.chat()`：

- **第 0 轮**：冷启动或热启动 prompt（见上文）
- **第 1 轮**：自我审计——识别初始评估中的不足，并综合近期会话的证据
- **第 2 轮**：调和——检查前几轮之间的矛盾，生成最终综合结论

每轮使用按比例分配的推理级别（早期轮次较轻，主轮次使用基础级别）。通过 `dialecticDepthLevels` 可逐轮覆盖——例如，深度 3 运行时使用 `["minimal", "medium", "high"]`。

如果前一轮返回了强信号（长且结构化的输出），后续轮次会提前退出，因此深度 3 并不总是意味着 3 次 LLM 调用。

### 会话启动预热

会话初始化时，Honcho 在后台以完整配置的 `dialecticDepth` 触发一次辩证调用，并将结果直接传递给第 1 轮的上下文组装。对冷 peer 进行单轮预热通常返回较少内容——多轮深度会在用户开口之前完成审计/调和周期。如果预热在第 1 轮前未完成，第 1 轮将回退到有超时限制的同步调用。

### 查询自适应推理级别

自动注入的辩证会根据查询长度调整 `dialecticReasoningLevel`：≥120 字符时 +1 级，≥400 字符时 +2 级，上限为 `reasoningLevelCap`（默认 `"high"`）。设置 `reasoningHeuristic: false` 可禁用此功能，将所有自动调用固定在 `dialecticReasoningLevel`。可用级别：`minimal`、`low`、`medium`、`high`、`max`。

## 配置选项

Honcho 在 `~/.honcho/config.json`（全局）或 `$HERMES_HOME/honcho.json`（profile 本地）中配置。设置向导会自动处理。

### 完整配置参考

| 键 | 默认值 | 说明 |
|-----|---------|-------------|
| `contextTokens` | `null`（不限制） | 每轮自动注入上下文的 token 预算。设为整数（如 1200）以限制上限，按词边界截断 |
| `contextCadence` | `1` | `context()` API 调用之间的最小轮数（基础层刷新） |
| `dialecticCadence` | `2` | `peer.chat()` LLM 调用之间的最小轮数（辩证层）。推荐 1–5。在 `tools` 模式下无关——由模型显式调用 |
| `dialecticDepth` | `1` | 每次辩证调用的 `.chat()` 轮数，限制在 1–3 |
| `dialecticDepthLevels` | `null` | 可选的每轮推理级别数组，如 `["minimal", "low", "medium"]`，覆盖按比例分配的默认值 |
| `dialecticReasoningLevel` | `'low'` | 基础推理级别：`minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 为 `true` 时，模型可通过 tool 参数逐次覆盖推理级别 |
| `dialecticMaxChars` | `600` | 注入系统 prompt 的辩证结果最大字符数 |
| `recallMode` | `'hybrid'` | `hybrid`（自动注入 + tools）、`context`（仅注入）、`tools`（仅 tools） |
| `writeFrequency` | `'async'` | 消息刷新时机：`async`（后台线程）、`turn`（同步）、`session`（会话结束时批量）或整数 N |
| `saveMessages` | `true` | 是否将消息持久化到 Honcho API |
| `observationMode` | `'directional'` | `directional`（全部开启）或 `unified`（共享池）。可用 `observation` 对象进行精细控制 |
| `messageMaxChars` | `25000` | 通过 `add_messages()` 发送的每条消息最大字符数，超出时分块 |
| `dialecticMaxInputChars` | `10000` | 传入 `peer.chat()` 的辩证查询输入最大字符数 |
| `sessionStrategy` | `'per-directory'` | `per-directory`、`per-repo`、`per-session` 或 `global` |

**会话策略**控制 Honcho 会话与工作内容的映射方式：
- `per-session` — 每次 `hermes` 运行获得一个新会话。干净启动，通过 tools 访问记忆。推荐新用户使用。
- `per-directory` — 每个工作目录对应一个 Honcho 会话，上下文跨运行积累。
- `per-repo` — 每个 git 仓库对应一个会话。
- `global` — 所有目录共用一个会话。

**Recall 模式**控制记忆如何流入对话：
- `hybrid` — 上下文自动注入系统 prompt，同时提供 tools（由模型决定何时查询）。
- `context` — 仅自动注入，隐藏 tools。
- `tools` — 仅 tools，不自动注入。agent 必须显式调用 `honcho_reasoning`、`honcho_search` 等。

**各 recall 模式下的设置行为：**

| 设置 | `hybrid` | `context` | `tools` |
|---------|----------|-----------|---------|
| `writeFrequency` | 刷新消息 | 刷新消息 | 刷新消息 |
| `contextCadence` | 控制基础上下文刷新 | 控制基础上下文刷新 | 无关——不注入 |
| `dialecticCadence` | 控制自动 LLM 调用 | 控制自动 LLM 调用 | 无关——由模型显式调用 |
| `dialecticDepth` | 每次调用的多轮数 | 每次调用的多轮数 | 无关——由模型显式调用 |
| `contextTokens` | 限制注入量 | 限制注入量 | 无关——不注入 |
| `dialecticDynamic` | 控制模型覆盖 | 不适用（无 tools） | 控制模型覆盖 |

在 `tools` 模式下，模型完全自主——它在需要时调用 `honcho_reasoning`，并自行选择 `reasoning_level`。Cadence 和预算设置仅适用于有自动注入的模式（`hybrid` 和 `context`）。

## 观察模式（定向 vs. 统一）

Honcho 将对话建模为 peer 之间的消息交换。每个 peer 有两个观察开关，与 Honcho 的 `SessionPeerConfig` 一一对应：

| 开关 | 效果 |
|--------|--------|
| `observeMe` | Honcho 根据该 peer 自身的消息构建其表示 |
| `observeOthers` | 该 peer 观察另一 peer 的消息（用于跨 peer 推理） |

两个 peer × 两个开关 = 四个标志。`observationMode` 是快捷预设：

| 预设 | 用户标志 | AI 标志 | 语义 |
|--------|-----------|----------|-----------|
| `"directional"`（默认） | me: 开，others: 开 | me: 开，others: 开 | 完全互相观察。启用跨 peer 辩证——"AI 根据用户所说和 AI 回复，对用户了解多少。" |
| `"unified"` | me: 开，others: 关 | me: 关，others: 开 | 共享池语义——AI 仅观察用户消息，用户 peer 仅自我建模。单观察者池。 |

使用显式 `observation` 块覆盖预设，实现逐 peer 精细控制：

```json
"observation": {
  "user": { "observeMe": true,  "observeOthers": true },
  "ai":   { "observeMe": true,  "observeOthers": false }
}
```

常见配置模式：

| 意图 | 配置 |
|--------|--------|
| 完全观察（大多数用户） | `"observationMode": "directional"` |
| AI 不应根据自身回复重新建模用户 | `"ai": {"observeMe": true, "observeOthers": false}` |
| AI peer 不应通过自我观察更新的强人设 | `"ai": {"observeMe": false, "observeOthers": true}` |

通过 [Honcho 控制台](https://app.honcho.dev) 设置的服务端开关优先于本地默认值——Hermes 在会话初始化时同步回本地。

## Tools

当 Honcho 作为 memory provider 激活时，以下五个 tools 可用：

| Tool | 用途 |
|------|---------|
| `honcho_profile` | 读取或更新 peer 卡片——传入 `card`（事实列表）以更新，省略则读取 |
| `honcho_search` | 对上下文进行语义搜索——返回原始摘录，不经 LLM 合成 |
| `honcho_context` | 完整会话上下文——摘要、表示、卡片、近期消息 |
| `honcho_reasoning` | Honcho LLM 合成的答案——传入 `reasoning_level`（minimal/low/medium/high/max）控制深度 |
| `honcho_conclude` | 创建或删除结论——传入 `conclusion` 创建，传入 `delete_id` 删除（仅限 PII） |

## CLI 命令

`hermes honcho` 子命令**仅在 Honcho 为当前活跃 memory provider 时注册**（`config.yaml` 中 `memory.provider: honcho`）。先运行 `hermes memory setup` 并选择 Honcho，子命令将在下次调用时出现。

```bash
hermes honcho status          # 连接状态、配置及关键设置
hermes honcho setup           # 重定向到 `hermes memory setup`
hermes honcho strategy        # 查看或设置会话策略（per-session/per-directory/per-repo/global）
hermes honcho peer            # 查看或更新 peer 名称及辩证推理级别
hermes honcho mode            # 查看或设置 recall 模式（hybrid/context/tools）
hermes honcho tokens          # 查看或设置上下文和辩证的 token 预算
hermes honcho identity        # 初始化或查看 AI peer 的 Honcho 身份
hermes honcho sync            # 将 Honcho 配置同步到所有现有 profile
hermes honcho peers           # 查看所有 profile 中的 peer 身份
hermes honcho sessions        # 列出已知的 Honcho 会话映射
hermes honcho map             # 将当前目录映射到 Honcho 会话名称
hermes honcho enable          # 为当前 profile 启用 Honcho
hermes honcho disable         # 为当前 profile 禁用 Honcho
hermes honcho migrate         # 从 openclaw-honcho 迁移的分步指南
```

## 从 `hermes honcho` 迁移

如果你之前使用了独立的 `hermes honcho setup`：

1. 你的现有配置（`honcho.json` 或 `~/.honcho/config.json`）已保留
2. 你的服务端数据（记忆、结论、用户画像）完好无损
3. 在 config.yaml 中设置 `memory.provider: honcho` 即可重新激活

无需重新登录或重新设置。运行 `hermes memory setup` 并选择"honcho"——向导会自动检测你的现有配置。

## 完整文档

参见 [Memory Providers — Honcho](./memory-providers.md#honcho) 获取完整参考文档。