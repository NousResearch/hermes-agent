---
sidebar_position: 4
title: "Memory Providers"
description: "外部记忆提供者插件 — Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory"
---

# Memory Providers

Hermes Agent 内置 8 个外部记忆提供者插件，为 Agent 提供跨会话的持久化知识，超越内置的 MEMORY.md 和 USER.md。同一时间只能激活**一个**外部提供者——内置记忆始终与其并行工作。

## 快速开始

```bash
hermes memory setup      # 交互式选择器 + 配置
hermes memory status     # 查看当前激活状态
hermes memory off        # 禁用外部提供者
```

也可以通过 `hermes plugins` → Provider Plugins → Memory Provider 选择激活的记忆提供者。

或在 `~/.hermes/config.yaml` 中手动设置：

```yaml
memory:
  provider: openviking   # 或 honcho, mem0, hindsight, holographic, retaindb, byterover, supermemory
```

## 工作原理

当记忆提供者激活时，Hermes 会自动：

1. **注入提供者上下文**到系统 prompt（提示词）中（提供者已知的内容）
2. **在每轮对话前预取相关记忆**（后台非阻塞）
3. **在每次响应后将对话轮次同步**到提供者
4. **在会话结束时提取记忆**（适用于支持此功能的提供者）
5. **将内置记忆写入镜像**到外部提供者
6. **添加提供者专属工具**，使 Agent 能够搜索、存储和管理记忆

内置记忆（MEMORY.md / USER.md）继续按原有方式工作。外部提供者是增量叠加的。

## 可用提供者

### Honcho

AI 原生的跨会话用户建模，具备辩证推理、会话范围上下文注入、语义搜索和持久化结论。基础上下文现在包含会话摘要以及用户表示和 peer card，使 Agent 能感知已讨论的内容。

| | |
|---|---|
| **适合场景** | 具有跨会话上下文的多 Agent 系统、用户-Agent 对齐 |
| **依赖** | `pip install honcho-ai` + [API key](https://app.honcho.dev) 或自托管实例 |
| **数据存储** | Honcho Cloud 或自托管 |
| **费用** | Honcho 定价（云端）/ 免费（自托管） |

**工具（5 个）：** `honcho_profile`（读取/更新 peer card）、`honcho_search`（语义搜索）、`honcho_context`（会话上下文——摘要、表示、card、消息）、`honcho_reasoning`（LLM 合成）、`honcho_conclude`（创建/删除结论）

**架构：** 双层上下文注入——基础层（会话摘要 + 表示 + peer card，按 `contextCadence` 刷新）加上辩证补充层（LLM 推理，按 `dialecticCadence` 刷新）。辩证层根据基础上下文是否存在，自动选择冷启动 prompt（通用用户事实）或热 prompt（会话范围上下文）。

**三个正交配置项**独立控制成本和深度：

- `contextCadence` — 基础层刷新频率（API 调用频率）
- `dialecticCadence` — 辩证 LLM 触发频率（LLM 调用频率）
- `dialecticDepth` — 每次辩证调用的 `.chat()` 轮数（1–3，推理深度）

**安装向导：**
```bash
hermes memory setup        # 选择 "honcho" — 运行 Honcho 专属的安装后配置
```

旧版 `hermes honcho setup` 命令仍然有效（现在会重定向到 `hermes memory setup`），但只有在 Honcho 被选为激活记忆提供者后才会注册。

**配置：** `$HERMES_HOME/honcho.json`（profile 本地）或 `~/.honcho/config.json`（全局）。解析顺序：`$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`。参见[配置参考](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md)和 [Honcho 集成指南](https://docs.honcho.dev/v3/guides/integrations/hermes)。

<details>
<summary>完整配置参考</summary>

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `apiKey` | -- | 来自 [app.honcho.dev](https://app.honcho.dev) 的 API key |
| `baseUrl` | -- | 自托管 Honcho 的 Base URL |
| `peerName` | -- | 用户 peer 身份 |
| `aiPeer` | host key | AI peer 身份（每个 profile 一个） |
| `workspace` | host key | 共享 workspace ID |
| `contextTokens` | `null`（无上限） | 每轮自动注入上下文的 token 预算。按词边界截断 |
| `contextCadence` | `1` | `context()` API 调用之间的最小轮数（基础层刷新） |
| `dialecticCadence` | `2` | `peer.chat()` LLM 调用之间的最小轮数。建议 1–5。仅适用于 `hybrid`/`context` 模式 |
| `dialecticDepth` | `1` | 每次辩证调用的 `.chat()` 轮数。限制在 1–3。第 0 轮：冷/热 prompt，第 1 轮：自我审计，第 2 轮：调和 |
| `dialecticDepthLevels` | `null` | 可选的每轮推理级别数组，例如 `["minimal", "low", "medium"]`。覆盖比例默认值 |
| `dialecticReasoningLevel` | `'low'` | 基础推理级别：`minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 为 `true` 时，模型可通过工具参数在每次调用时覆盖推理级别 |
| `dialecticMaxChars` | `600` | 注入系统 prompt 的辩证结果最大字符数 |
| `recallMode` | `'hybrid'` | `hybrid`（自动注入 + 工具）、`context`（仅注入）、`tools`（仅工具） |
| `writeFrequency` | `'async'` | 消息刷新时机：`async`（后台线程）、`turn`（同步）、`session`（会话结束时批量）或整数 N |
| `saveMessages` | `true` | 是否将消息持久化到 Honcho API |
| `observationMode` | `'directional'` | `directional`（全部开启）或 `unified`（共享池）。通过 `observation` 对象覆盖 |
| `messageMaxChars` | `25000` | 每条消息的最大字符数（超出时分块） |
| `dialecticMaxInputChars` | `10000` | 传入 `peer.chat()` 的辩证查询输入最大字符数 |
| `sessionStrategy` | `'per-directory'` | `per-directory`、`per-repo`、`per-session`、`global` |

</details>

<details>
<summary>最简 honcho.json（云端）</summary>

```json
{
  "apiKey": "your-key-from-app.honcho.dev",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

<details>
<summary>最简 honcho.json（自托管）</summary>

```json
{
  "baseUrl": "http://localhost:8000",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

:::tip 从 `hermes honcho` 迁移
如果你之前使用过 `hermes honcho setup`，你的配置和所有服务端数据均完好无损。只需通过安装向导重新启用，或手动设置 `memory.provider: honcho`，即可通过新系统重新激活。
:::

**多 peer 配置：**

Honcho 将对话建模为 peer 之间的消息交换——每个 Hermes profile 对应一个用户 peer 加一个 AI peer，共享同一个 workspace。workspace 是共享环境：用户 peer 在各 profile 间全局共享，每个 AI peer 拥有独立身份。每个 AI peer 从自身的观察中独立构建表示/card，因此 `coder` profile 保持代码导向，而 `writer` profile 针对同一用户保持编辑导向。

映射关系：

| 概念 | 含义 |
|---------|-----------|
| **Workspace** | 共享环境。同一 workspace 下的所有 Hermes profile 共享同一用户身份。 |
| **用户 peer**（`peerName`） | 人类用户。在 workspace 内跨 profile 共享。 |
| **AI peer**（`aiPeer`） | 每个 Hermes profile 一个。host key `hermes` → 默认；其他 profile 使用 `hermes.<profile>`。 |
| **Observation** | 每个 peer 的开关，控制 Honcho 从哪些消息中建模。`directional`（默认，全部开启）或 `unified`（单一观察者池）。 |

### 新建 profile，创建新 Honcho peer

```bash
hermes profile create coder --clone
```

`--clone` 在 `honcho.json` 中创建一个 `hermes.coder` host 块，包含 `aiPeer: "coder"`、共享的 `workspace`、继承的 `peerName`、`recallMode`、`writeFrequency`、`observation` 等。AI peer 会在 Honcho 中提前创建，确保在第一条消息之前就已存在。

### 为现有 profile 补充 Honcho peer

```bash
hermes honcho sync
```

扫描所有 Hermes profile，为没有 host 块的 profile 创建 host 块，从默认 `hermes` 块继承设置，并提前创建新的 AI peer。幂等操作——跳过已有 host 块的 profile。

### 每个 profile 的 observation 配置

每个 host 块可以独立覆盖 observation 配置。示例：一个以代码为中心的 profile，AI peer 观察用户但不自我建模：

```json
"hermes.coder": {
  "aiPeer": "coder",
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": false, "observeOthers": true }
  }
}
```

**Observation 开关（每个 peer 一组）：**

| 开关 | 效果 |
|--------|--------|
| `observeMe` | Honcho 根据该 peer 自身的消息构建其表示 |
| `observeOthers` | 该 peer 观察另一 peer 的消息（用于跨 peer 推理） |

通过 `observationMode` 使用预设：

- **`"directional"`**（默认）——四个标志全部开启。完全互相观察；启用跨 peer 辩证。
- **`"unified"`**——用户 `observeMe: true`，AI `observeOthers: true`，其余为 false。单一观察者池；AI 对用户建模但不自我建模，用户 peer 仅自我建模。

通过 [Honcho 控制台](https://app.honcho.dev) 设置的服务端开关优先于本地默认值——在会话初始化时同步回来。

参见 [Honcho 页面](./honcho.md#observation-directional-vs-unified) 获取完整的 observation 参考。

<details>
<summary>完整 honcho.json 示例（多 profile）</summary>

```json
{
  "apiKey": "your-key",
  "workspace": "hermes",
  "peerName": "eri",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "hybrid",
      "writeFrequency": "async",
      "sessionStrategy": "per-directory",
      "observation": {
        "user": { "observeMe": true, "observeOthers": true },
        "ai": { "observeMe": true, "observeOthers": true }
      },
      "dialecticReasoningLevel": "low",
      "dialecticDynamic": true,
      "dialecticCadence": 2,
      "dialecticDepth": 1,
      "dialecticMaxChars": 600,
      "contextCadence": 1,
      "messageMaxChars": 25000,
      "saveMessages": true
    },
    "hermes.coder": {
      "enabled": true,
      "aiPeer": "coder",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "tools",
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    },
    "hermes.writer": {
      "enabled": true,
      "aiPeer": "writer",
      "workspace": "hermes",
      "peerName": "eri"
    }
  },
  "sessions": {
    "/home/user/myproject": "myproject-main"
  }
}
```

</details>

参见[配置参考](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md)和 [Honcho 集成指南](https://docs.honcho.dev/v3/guides/integrations/hermes)。


---

### OpenViking

由 Volcengine（ByteDance）提供的上下文数据库，具备文件系统式知识层级、分层检索，以及自动将记忆提取为 6 个类别的功能。

| | |
|---|---|
| **适合场景** | 具有结构化浏览功能的自托管知识管理 |
| **依赖** | `pip install openviking` + 运行中的服务器 |
| **数据存储** | 自托管（本地或云端） |
| **费用** | 免费（开源，AGPL-3.0） |

**工具：** `viking_search`（语义搜索）、`viking_read`（分层：摘要/概览/全文）、`viking_browse`（文件系统导航）、`viking_remember`（存储事实）、`viking_add_resource`（导入 URL/文档）

**安装：**
```bash
# 先启动 OpenViking 服务器
pip install openviking
openviking-server

# 然后配置 Hermes
hermes memory setup    # 选择 "openviking"
# 或手动配置：
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

**主要特性：**
- 分层上下文加载：L0（约 100 tokens）→ L1（约 2k）→ L2（完整）
- 会话提交时自动提取记忆（profile、偏好、实体、事件、案例、模式）
- `viking://` URI 方案用于层级知识浏览

---

### Mem0

服务端 LLM 事实提取，具备语义搜索、重排序和自动去重功能。

| | |
|---|---|
| **适合场景** | 免维护的记忆管理——Mem0 自动处理提取 |
| **依赖** | `pip install mem0ai` + API key |
| **数据存储** | Mem0 Cloud |
| **费用** | Mem0 定价 |

**工具：** `mem0_profile`（所有已存储记忆）、`mem0_search`（语义搜索 + 重排序）、`mem0_conclude`（逐字存储事实）

**安装：**
```bash
hermes memory setup    # 选择 "mem0"
# 或手动配置：
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**配置：** `$HERMES_HOME/mem0.json`

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `user_id` | `hermes-user` | 用户标识符 |
| `agent_id` | `hermes` | Agent 标识符 |

---

### Hindsight

具备知识图谱、实体解析和多策略检索的长期记忆。`hindsight_reflect` 工具提供其他提供者均不具备的跨记忆合成能力。自动保留完整对话轮次（包括工具调用），并进行会话级文档追踪。

| | |
|---|---|
| **适合场景** | 基于知识图谱的实体关系召回 |
| **依赖** | 云端：来自 [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io) 的 API key。本地：LLM API key（OpenAI、Groq、OpenRouter 等） |
| **数据存储** | Hindsight Cloud 或本地嵌入式 PostgreSQL |
| **费用** | Hindsight 定价（云端）或免费（本地） |

**工具：** `hindsight_retain`（带实体提取的存储）、`hindsight_recall`（多策略搜索）、`hindsight_reflect`（跨记忆合成）

**安装：**
```bash
hermes memory setup    # 选择 "hindsight"
# 或手动配置：
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

安装向导会自动安装依赖，并仅安装所选模式所需的内容（云端用 `hindsight-client`，本地用 `hindsight-all`）。需要 `hindsight-client >= 0.4.22`（会话启动时若版本过旧则自动升级）。

**本地模式 UI：** `hindsight-embed -p hermes ui start`

**配置：** `$HERMES_HOME/hindsight/config.json`

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` 或 `local` |
| `bank_id` | `hermes` | 记忆库标识符 |
| `recall_budget` | `mid` | 召回彻底程度：`low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid`（上下文 + 工具）、`context`（仅自动注入）、`tools`（仅工具） |
| `auto_retain` | `true` | 自动保留对话轮次 |
| `auto_recall` | `true` | 每轮对话前自动召回记忆 |
| `retain_async` | `true` | 在服务器上异步处理保留操作 |
| `retain_context` | `conversation between Hermes Agent and the User` | 保留记忆的上下文标签 |
| `retain_tags` | — | 应用于保留记忆的默认标签；与每次工具调用的标签合并 |
| `retain_source` | — | 附加到保留记忆的可选 `metadata.source` |
| `retain_user_prefix` | `User` | 自动保留的对话记录中用户轮次前的标签 |
| `retain_assistant_prefix` | `Assistant` | 自动保留的对话记录中助手轮次前的标签 |
| `recall_tags` | — | 召回时用于过滤的标签 |

完整配置参考参见[插件 README](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md)。

---

### Holographic

本地 SQLite 事实存储，具备 FTS5 全文搜索、信任评分和 HRR（Holographic Reduced Representations，全息降维表示）用于组合代数查询。

| | |
|---|---|
| **适合场景** | 无外部依赖的纯本地高级检索记忆 |
| **依赖** | 无（SQLite 始终可用）。NumPy 可选，用于 HRR 代数。 |
| **数据存储** | 本地 SQLite |
| **费用** | 免费 |

**工具：** `fact_store`（9 个动作：add、search、probe、related、reason、contradict、update、remove、list）、`fact_feedback`（有用/无用评分，用于训练信任评分）

**安装：**
```bash
hermes memory setup    # 选择 "holographic"
# 或手动配置：
hermes config set memory.provider holographic
```

**配置：** `plugins.hermes-memory-store` 下的 `config.yaml`

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite 数据库路径 |
| `auto_extract` | `false` | 会话结束时自动提取事实 |
| `default_trust` | `0.5` | 默认信任评分（0.0–1.0） |

**独特能力：**
- `probe` — 针对特定实体的代数召回（某人/某物的所有事实）
- `reason` — 跨多个实体的组合 AND 查询
- `contradict` — 自动检测冲突事实
- 信任评分，带非对称反馈（有用 +0.05 / 无用 -0.10）

---

### RetainDB

云端记忆 API，具备混合搜索（向量 + BM25 + 重排序）、7 种记忆类型和增量压缩。

| | |
|---|---|
| **适合场景** | 已使用 RetainDB 基础设施的团队 |
| **依赖** | RetainDB 账号 + API key |
| **数据存储** | RetainDB Cloud |
| **费用** | $20/月 |

**工具：** `retaindb_profile`（用户 profile）、`retaindb_search`（语义搜索）、`retaindb_context`（任务相关上下文）、`retaindb_remember`（带类型和重要性的存储）、`retaindb_forget`（删除记忆）

**安装：**
```bash
hermes memory setup    # 选择 "retaindb"
# 或手动配置：
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

---

### ByteRover

通过 `brv` CLI 实现持久化记忆——具备分层知识树和分层检索（模糊文本 → LLM 驱动搜索）。本地优先，可选云端同步。

| | |
|---|---|
| **适合场景** | 希望使用可移植、本地优先记忆和 CLI 的开发者 |
| **依赖** | ByteRover CLI（`npm install -g byterover-cli` 或[安装脚本](https://byterover.dev)） |
| **数据存储** | 本地（默认）或 ByteRover Cloud（可选同步） |
| **费用** | 免费（本地）或 ByteRover 定价（云端） |

**工具：** `brv_query`（搜索知识树）、`brv_curate`（存储事实/决策/模式）、`brv_status`（CLI 版本 + 树状统计）

**安装：**
```bash
# 先安装 CLI
curl -fsSL https://byterover.dev/install.sh | sh

# 然后配置 Hermes
hermes memory setup    # 选择 "byterover"
# 或手动配置：
hermes config set memory.provider byterover
```

**主要特性：**
- 自动预压缩提取（在上下文压缩丢弃内容前保存洞察）
- 知识树存储于 `$HERMES_HOME/byterover/`（profile 范围隔离）
- SOC2 Type II 认证的云端同步（可选）

---

### Supermemory

语义长期记忆，具备 profile 召回、语义搜索、显式记忆工具，以及通过 Supermemory graph API 进行会话结束时的对话导入。

| | |
|---|---|
| **适合场景** | 带用户 profile 和会话级图谱构建的语义召回 |
| **依赖** | `pip install supermemory` + [API key](https://supermemory.ai) |
| **数据存储** | Supermemory Cloud |
| **费用** | Supermemory 定价 |

**工具：** `supermemory_store`（保存显式记忆）、`supermemory_search`（语义相似度搜索）、`supermemory_forget`（按 ID 或最佳匹配查询遗忘）、`supermemory_profile`（持久化 profile + 近期上下文）

**安装：**
```bash
hermes memory setup    # 选择 "supermemory"
# 或手动配置：
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

**配置：** `$HERMES_HOME/supermemory.json`

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `container_tag` | `hermes` | 用于搜索和写入的容器标签。支持 `{identity}` 模板用于 profile 范围隔离。 |
| `auto_recall` | `true` | 在每轮对话前注入相关记忆上下文 |
| `auto_capture` | `true` | 每次响应后存储清理过的用户-助手轮次 |
| `max_recall_results` | `10` | 格式化为上下文的最大召回条目数 |
| `profile_frequency` | `50` | 在第一轮及每 N 轮包含 profile 事实 |
| `capture_mode` | `all` | 默认跳过过短或无意义的轮次 |
| `search_mode` | `hybrid` | 搜索模式：`hybrid`、`memories` 或 `documents` |
| `api_timeout` | `5.0` | SDK 和导入请求的超时时间 |

**环境变量：** `SUPERMEMORY_API_KEY`（必填）、`SUPERMEMORY_CONTAINER_TAG`（覆盖配置）。

**主要特性：**
- 自动上下文隔离——从捕获的轮次中剥离已召回的记忆，防止递归记忆污染
- 会话结束时的对话导入，用于构建更丰富的图谱级知识
- 在第一轮及可配置间隔注入 profile 事实
- 无意义消息过滤（跳过"ok"、"thanks"等）
- **Profile 范围容器**——在 `container_tag` 中使用 `{identity}`（例如 `hermes-{identity}` → `hermes-coder`），按 Hermes profile 隔离记忆
- **多容器模式**——启用 `enable_custom_container_tags` 并配置 `custom_containers` 列表，让 Agent 跨命名容器读写。自动操作（同步、预取）保持在主容器上。

<details>
<summary>多容器示例</summary>

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "shared-knowledge"],
  "custom_container_instructions": "Use project-alpha for coding context."
}
```

</details>

**支持：** [Discord](https://supermemory.link/discord) · [support@supermemory.com](mailto:support@supermemory.com)

---

## 提供者对比

| 提供者 | 存储 | 费用 | 工具数 | 依赖 | 独特特性 |
|----------|---------|------|-------|-------------|----------------|
| **Honcho** | 云端 | 付费 | 5 | `honcho-ai` | 辩证用户建模 + 会话范围上下文 |
| **OpenViking** | 自托管 | 免费 | 5 | `openviking` + 服务器 | 文件系统层级 + 分层加载 |
| **Mem0** | 云端 | 付费 | 3 | `mem0ai` | 服务端 LLM 提取 |
| **Hindsight** | 云端/本地 | 免费/付费 | 3 | `hindsight-client` | 知识图谱 + reflect 合成 |
| **Holographic** | 本地 | 免费 | 2 | 无 | HRR 代数 + 信任评分 |
| **RetainDB** | 云端 | $20/月 | 5 | `requests` | 增量压缩 |
| **ByteRover** | 本地/云端 | 免费/付费 | 3 | `brv` CLI | 预压缩提取 |
| **Supermemory** | 云端 | 付费 | 4 | `supermemory` | 上下文隔离 + 会话图谱导入 + 多容器 |

## Profile 隔离

每个提供者的数据按 [profile](/user-guide/profiles) 隔离：

- **本地存储提供者**（Holographic、ByteRover）使用 `$HERMES_HOME/` 路径，各 profile 路径不同
- **配置文件提供者**（Honcho、Mem0、Hindsight、Supermemory）将配置存储在 `$HERMES_HOME/` 中，每个 profile 拥有独立凭证
- **云端提供者**（RetainDB）自动派生 profile 范围的项目名称
- **环境变量提供者**（OpenViking）通过每个 profile 的 `.env` 文件配置

## 构建记忆提供者

参见[开发者指南：Memory Provider 插件](/developer-guide/memory-provider-plugin)了解如何创建自己的提供者。