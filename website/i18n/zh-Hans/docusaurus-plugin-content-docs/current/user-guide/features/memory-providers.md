---
sidebar_position: 4
title: "内存提供者"
description: "外部内存提供者插件 — Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory"
---

# 内存提供者

Hermes Agent 随附 8 个外部内存提供者插件，提供持久的跨会话知识，超出内置的 MEMORY.md 与 USER.md。一次只能激活 **一个** 外部提供者——内置内存始终并行工作。

## 快速开始

```bash
hermes memory setup      # 交互式选择与配置
hermes memory status     # 检查当前激活情况
hermes memory off        # 禁用外部提供者
```

你也可以通过 `hermes plugins` 菜单选择，或在 `~/.hermes/config.yaml` 中手动设置：

```yaml
memory:
  provider: openviking   # 或 honcho, mem0, hindsight, holographic, retaindb, byterover, supermemory
```

## 工作原理

当外部内存提供者激活时，Hermes 会自动：

1. 将提供者的上下文注入系统提示（提供者已知内容）
2. 在每个回合前预取相关记忆（后台非阻塞）
3. 在每次回应后将会话回合同步到提供者
4. 在会话结束时从提供者提取记忆（若支持）
5. 将内置内存写入镜像到外部提供者
6. 添加提供者特定的工具，供代理搜索、存储和管理记忆

内置内存（MEMORY.md / USER.md）继续正常工作，外部提供者是附加的。

## 可用提供者（概要）

### Honcho

面向 AI 的跨会话用户建模，支持方言式推理、会话范围上下文注入、语义检索与持久结论。适合需要跨会话上下文与用户对齐的场景。需要 `pip install honcho-ai` 并使用 Honcho 云 API 或自托管。

主要工具：`honcho_profile`、`honcho_search`、`honcho_context`、`honcho_reasoning`、`honcho_conclude`。

### OpenViking

Volcengine 的上下文数据库，支持分层检索与文件系统式知识层次，适合自托管知识管理。

主要工具：`viking_search`、`viking_read`、`viking_browse`、`viking_remember`、`viking_add_resource`。

### Mem0

服务器端事实抽取与语义检索服务，自动去重与重排序，适合免维护的记忆管理。需要 Mem0 云账号或 API 密钥。

主要工具：`mem0_profile`、`mem0_search`、`mem0_conclude`。

### Hindsight

长期记忆与知识图谱，支持实体解析与多策略检索。`hindsight_reflect` 提供跨记忆综合能力，适合需要复杂关系检索的场景。

主要工具：`hindsight_retain`、`hindsight_recall`、`hindsight_reflect`。

### Holographic

基于本地 SQLite + FTS5 的事实存储，带信任评分与 HRR（用于组合代数查询），适合纯本地部署场景。

主要工具：`fact_store`、`fact_feedback`。无需外部依赖。

（下略：原文对各提供者的配置、示例与细节非常丰富，已在英文源文档保留完整参考）

---
---
sidebar_position: 4
title: "Memory Providers"
description: "外部 memory provider 插件 — Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory"
---

# Memory Providers

Hermes Agent 内置了 8 个外部 memory provider 插件，可为 agent 提供持久的、跨 session 的知识，超越内置的 MEMORY.md 和 USER.md。同一时间只能激活 **一个** 外部 provider —— 内置 memory 始终与其同时运行。

## 快速开始

```bash
hermes memory setup      # 交互式选择 + 配置
hermes memory status     # 查看当前激活的 provider
hermes memory off        # 禁用外部 provider
```

你也可以通过 `hermes plugins` → Provider Plugins → Memory Provider 来选择激活的 memory provider。

或者在 `~/.hermes/config.yaml` 中手动设置：

```yaml
memory:
  provider: openviking   # 或 honcho、mem0、hindsight、holographic、retaindb、byterover、supermemory
```

## 工作原理

当 memory provider 激活时，Hermes 会自动：

1. **将 provider 上下文注入** system prompt（provider 已知的信息）
2. **在每次 turn 前预取相关记忆**（后台、非阻塞）
3. **在每次响应后将对话 turn 同步**到 provider
4. **在 session 结束时提取记忆**（对于支持的 provider）
5. **将内置 memory 的写入镜像**到外部 provider
6. **添加 provider 专属工具**，使 agent 能够搜索、存储和管理记忆

内置 memory（MEMORY.md / USER.md）完全保持原有工作方式。外部 provider 是增量补充。

## 可用 Providers

### Honcho

AI-native 跨 session 用户建模，支持 dialectic reasoning、session-scoped 上下文注入、semantic search 和持久化结论。基础上下文现在包含 session summary，以及用户画像和 peer card，使 agent 能够了解已经讨论过的内容。

| | |
|---|---|
| **最佳适用场景** | 需要跨 session 上下文的多 agent 系统、用户-agent 对齐 |
| **需要** | `pip install honcho-ai` + [API key](https://app.honcho.dev) 或自托管实例 |
| **数据存储** | Honcho Cloud 或自托管 |
| **成本** | Honcho 定价（云端）/ 免费（自托管） |

**工具（5 个）：** `honcho_profile`（读取/更新 peer card）、`honcho_search`（semantic search）、`honcho_context`（session 上下文 — summary、representation、card、messages）、`honcho_reasoning`（LLM 合成）、`honcho_conclude`（创建/删除结论）

**架构：** 双层上下文注入 —— 基础层（session summary + representation + peer card，按 `contextCadence` 刷新）和 dialectic 补充层（LLM reasoning，按 `dialecticCadence` 刷新）。Dialectic 会根据基础上下文是否存在，自动选择 cold-start prompts（通用用户事实）或 warm prompts（session-scoped 上下文）。

**三个独立的配置旋钮**，分别控制成本和深度：

- `contextCadence` —— 基础层刷新的频率（API 调用频率）
- `dialecticCadence` —— dialectic LLM 触发的频率（LLM 调用频率）
- `dialecticDepth` —— 每次 dialectic 调用中 `.chat()` 的轮数（1–3，推理深度）

**设置向导：**
```bash
hermes memory setup        # 选择 "honcho" —— 运行 Honcho 专属的后置设置
```

旧的 `hermes honcho setup` 命令仍然可用（现在会重定向到 `hermes memory setup`），但仅在 Honcho 被选为激活的 memory provider 后才会注册。

**配置：** `$HERMES_HOME/honcho.json`（profile 本地）或 `~/.honcho/config.json`（全局）。解析顺序：`$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`。参见 [配置参考](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md) 和 [Honcho 集成指南](https://docs.honcho.dev/v3/guides/integrations/hermes)。

<details>
<summary>完整配置参考</summary>

| Key | Default | Description |
|-----|---------|-------------|
| `apiKey` | -- | 来自 [app.honcho.dev](https://app.honcho.dev) 的 API key |
| `baseUrl` | -- | 自托管 Honcho 的基础 URL |
| `peerName` | -- | 用户 peer 身份 |
| `aiPeer` | host key | AI peer 身份（每个 profile 一个） |
| `workspace` | host key | 共享 workspace ID |
| `contextTokens` | `null`（无上限） | 每 turn 自动注入上下文的 token 预算。按词边界截断 |
| `contextCadence` | `1` | 两次 `context()` API 调用之间的最小 turn 数（基础层刷新） |
| `dialecticCadence` | `2` | 两次 `peer.chat()` LLM 调用之间的最小 turn 数。推荐 1–5。仅适用于 `hybrid`/`context` 模式 |
| `dialecticDepth` | `1` | 每次 dialectic 调用中 `.chat()` 的轮数。限制在 1–3。Pass 0：cold/warm prompt，pass 1：self-audit，pass 2：reconciliation |
| `dialecticDepthLevels` | `null` | 可选的每 pass reasoning level 数组，例如 `["minimal", "low", "medium"]`。覆盖比例默认值 |
| `dialecticReasoningLevel` | `'low'` | 基础 reasoning level：`minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 当为 `true` 时，模型可以通过 tool 参数覆盖每次调用的 reasoning level |
| `dialecticMaxChars` | `600` | 注入 system prompt 的 dialectic 结果最大字符数 |
| `recallMode` | `'hybrid'` | `hybrid`（自动注入 + 工具）、`context`（仅注入）、`tools`（仅工具） |
| `writeFrequency` | `'async'` | 何时 flush messages：`async`（后台线程）、`turn`（同步）、`session`（结束时批量），或整数 N |
| `saveMessages` | `true` | 是否将消息持久化到 Honcho API |
| `observationMode` | `'directional'` | `directional`（全部开启）或 `unified`（共享池）。可通过 `observation` 对象覆盖 |
| `messageMaxChars` | `25000` | 每条消息的最大字符数（超出则分块） |
| `dialecticMaxInputChars` | `10000` | 传给 `peer.chat()` 的 dialectic 查询输入最大字符数 |
| `sessionStrategy` | `'per-directory'` | `per-directory`、`per-repo`、`per-session`、`global` |

</details>

<details>
<summary>最小化 honcho.json（云端）</summary>

```json
{
  "apiKey": "[REDACTED]",
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
<summary>最小化 honcho.json（自托管）</summary>

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
如果你之前使用过 `hermes honcho setup`，你的配置和所有服务端数据都保持不变。只需再次通过设置向导重新启用，或手动设置 `memory.provider: honcho` 即可通过新系统重新激活。
:::

**多 peer 设置：**

Honcho 将对话建模为 peer 之间交换消息 —— 一个用户 peer 加上每个 Hermes profile 一个 AI peer，全部共享一个 workspace。Workspace 是共享环境：用户 peer 跨 profile 全局共享，每个 AI peer 是其独立身份。每个 AI peer 从自身观察中构建独立的 representation / card，因此 `coder` profile 保持代码导向，而 `writer` profile 保持编辑导向，针对同一用户。

映射关系：

| 概念 | 说明 |
|---------|-----------|
| **Workspace** | 共享环境。同一 workspace 下的所有 Hermes profile 看到相同的用户身份。 |
| **User peer** (`peerName`) | 人类。在 workspace 中跨 profile 共享。 |
| **AI peer** (`aiPeer`) | 每个 Hermes profile 一个。Host key `hermes` → 默认；其他 profile 为 `hermes.<profile>`。 |
| **Observation** | 每个 peer 的开关，控制 Honcho 从谁的消息中建模什么。`directional`（默认，四个全开启）或 `unified`（单一观察者池）。 |

### 新建 profile，创建新的 Honcho peer

```bash
hermes profile create coder --clone
```

`--clone` 在 `honcho.json` 中创建一个 `hermes.coder` host 块，`aiPeer: "coder"`，共享 `workspace`，继承 `peerName`、`recallMode`、`writeFrequency`、`observation` 等。AI peer 会在 Honcho 中预先创建，确保在第一条消息发送前已存在。

### 已有 profile，回填 Honcho peer

```bash
hermes honcho sync
```

扫描每个 Hermes profile，为没有 host 块的 profile 创建 host 块，从默认的 `hermes` 块继承设置，并预先创建新的 AI peer。幂等 —— 跳过已有 host 块的 profile。

### 每个 profile 的 observation

每个 host 块可以独立覆盖 observation 配置。示例：一个专注于代码的 profile，其中 AI peer 观察用户但不自我建模：

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
| `observeMe` | Honcho 从此 peer 自身的消息中构建其 representation |
| `observeOthers` | 此 peer 观察另一方的消息（用于跨 peer reasoning） |

通过 `observationMode` 的预设：

- **`"directional"`**（默认）— 四个标志全部开启。完全相互观察；启用跨 peer dialectic。
- **`"unified"`** — user `observeMe: true`，AI `observeOthers: true`，其余为 false。单一观察者池；AI 建模用户但不建模自身，用户 peer 仅自我建模。

通过 [Honcho dashboard](https://app.honcho.dev) 设置的服务端开关优先于本地默认值 —— 在 session 初始化时同步回来。

参见 [Honcho 页面](./honcho.md#observation-directional-vs-unified) 获取完整的 observation 参考。

<details>
<summary>完整 honcho.json 示例（多 profile）</summary>

```json
{
  "apiKey": "[REDACTED]",
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

参见 [配置参考](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md) 和 [Honcho 集成指南](https://docs.honcho.dev/v3/guides/integrations/hermes)。


---

### OpenViking

由 Volcengine（ByteDance）提供的上下文数据库，具有类文件系统的知识层级、分层检索，以及自动将记忆提取到 6 个类别中。

| | |
|---|---|
| **最佳适用场景** | 需要结构化浏览的自托管知识管理 |
| **需要** | `pip install openviking` + 运行中的 server |
| **数据存储** | 自托管（本地或云端） |
| **成本** | 免费（开源，AGPL-3.0） |

**工具：** `viking_search`（semantic search）、`viking_read`（分层：abstract/overview/full）、`viking_browse`（文件系统导航）、`viking_remember`（存储事实）、`viking_add_resource`（摄取 URL/docs）

**设置：**
```bash
# 首先启动 OpenViking server
pip install openviking
openviking-server

# 然后配置 Hermes
hermes memory setup    # 选择 "openviking"
# 或手动：
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

**关键特性：**
- 分层上下文加载：L0（~100 tokens）→ L1（~2k）→ L2（完整）
- Session commit 时自动提取记忆（profile、preferences、entities、events、cases、patterns）
- `viking://` URI scheme 用于层级知识浏览

---

### Mem0

服务端 LLM 事实提取，支持 semantic search、reranking 和自动去重。

| | |
|---|---|
| **最佳适用场景** | 无需干预的 memory 管理 —— Mem0 自动处理提取 |
| **需要** | `pip install mem0ai` + API key |
| **数据存储** | Mem0 Cloud |
| **成本** | Mem0 定价 |

**工具：** `mem0_profile`（所有存储的记忆）、`mem0_search`（semantic search + reranking）、`mem0_conclude`（存储逐字事实）

**设置：**
```bash
hermes memory setup    # 选择 "mem0"
# 或手动：
hermes config set memory.provider mem0
echo "MEM0_API_KEY=[REDACTED]" >> ~/.hermes/.env
```

**配置：** `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `user_id` | `hermes-user` | 用户标识符 |
| `agent_id` | `hermes` | Agent 标识符 |

---

### Hindsight

长期记忆，具有知识图谱、实体解析和多策略检索。`hindsight_reflect` 工具提供跨记忆综合，这是其他 provider 不具备的。自动保留完整的对话 turn（包括 tool calls），并支持 session 级别的文档追踪。

| | |
|---|---|
| **最佳适用场景** | 基于知识图谱的回忆，支持实体关系 |
| **需要** | 云端：来自 [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io) 的 API key。本地：LLM API key（OpenAI、Groq、OpenRouter 等） |
| **数据存储** | Hindsight Cloud 或本地嵌入式 PostgreSQL |
| **成本** | Hindsight 定价（云端）或免费（本地） |

**工具：** `hindsight_retain`（存储并提取实体）、`hindsight_recall`（多策略搜索）、`hindsight_reflect`（跨记忆综合）

**设置：**
```bash
hermes memory setup    # 选择 "hindsight"
# 或手动：
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=[REDACTED]" >> ~/.hermes/.env
```

设置向导会自动安装依赖，并且仅安装所选模式所需的依赖（云端安装 `hindsight-client`，本地安装 `hindsight-all`）。需要 `hindsight-client >= 0.4.22`（如果过时会在 session 开始时自动升级）。

**本地模式 UI：** `hindsight-embed -p hermes ui start`

**配置：** `$HERMES_HOME/hindsight/config.json`

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` 或 `local` |
| `bank_id` | `hermes` | Memory bank 标识符 |
| `recall_budget` | `mid` | 回忆彻底程度：`low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid`（上下文 + 工具）、`context`（仅自动注入）、`tools`（仅工具） |
| `auto_retain` | `true` | 自动保留对话 turn |
| `auto_recall` | `true` | 在每次 turn 前自动回忆记忆 |
| `retain_async` | `true` | 在服务端异步处理 retain |
| `retain_context` | `conversation between Hermes Agent and the User` | 保留记忆的上下文标签 |
| `retain_tags` | — | 应用于保留记忆的默认标签；与每次调用的 tool 标签合并 |
| `retain_source` | — | 附加到保留记忆的可选 `metadata.source` |
| `retain_user_prefix` | `User` | 自动保留 transcript 中用户 turn 前的标签 |
| `retain_assistant_prefix` | `Assistant` | 自动保留 transcript 中 assistant turn 前的标签 |
| `recall_tags` | — | 回忆时过滤用的标签 |

参见 [plugin README](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md) 获取完整配置参考。

---

### Holographic

本地 SQLite 事实存储，支持 FTS5 全文搜索、信任评分，以及 HRR（Holographic Reduced Representations）用于组合代数查询。

| | |
|---|---|
| **最佳适用场景** | 纯本地 memory，支持高级检索，无外部依赖 |
| **需要** | 无需任何依赖（SQLite 始终可用）。NumPy 可选，用于 HRR 代数运算。 |
| **数据存储** | 本地 SQLite |
| **成本** | 免费 |

**工具：** `fact_store`（9 个动作：add、search、probe、related、reason、contradict、update、remove、list）、`fact_feedback`（helpful/unhelpful 评分，用于训练信任分数）

**设置：**
```bash
hermes memory setup    # 选择 "holographic"
# 或手动：
hermes config set memory.provider holographic
```

**配置：** `plugins.hermes-memory-store` 下的 `config.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite 数据库路径 |
| `auto_extract` | `false` | Session 结束时自动提取事实 |
| `default_trust` | `0.5` | 默认信任评分（0.0–1.0） |

**独特能力：**
- `probe` —— 针对特定实体的代数回忆（关于某个人/事物的所有事实）
- `reason` —— 跨多个实体的组合 AND 查询
- `contradict` —— 自动检测冲突事实
- 信任评分，非对称反馈（helpful +0.05 / unhelpful -0.10）

---

### RetainDB

云端 memory API，支持混合搜索（Vector + BM25 + Reranking）、7 种 memory 类型和 delta 压缩。

| | |
|---|---|
| **最佳适用场景** | 已在使用 RetainDB 基础设施的团队 |
| **需要** | RetainDB 账户 + API key |
| **数据存储** | RetainDB Cloud |
| **成本** | $20/月 |

**工具：** `retaindb_profile`（用户画像）、`retaindb_search`（semantic search）、`retaindb_context`（任务相关上下文）、`retaindb_remember`（按类型 + 重要性存储）、`retaindb_forget`（删除记忆）

**设置：**
```bash
hermes memory setup    # 选择 "retaindb"
# 或手动：
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=[REDACTED]" >> ~/.hermes/.env
```

---

### ByteRover

通过 `brv` CLI 实现持久化记忆 —— 层级知识树，支持分层检索（fuzzy text → LLM-driven search）。本地优先，可选云端同步。

| | |
|---|---|
| **最佳适用场景** | 希望拥有可移植、本地优先 memory 并附带 CLI 的开发者 |
| **需要** | ByteRover CLI（`npm install -g byterover-cli` 或 [install script](https://byterover.dev)） |
| **数据存储** | 本地（默认）或 ByteRover Cloud（可选同步） |
| **成本** | 免费（本地）或 ByteRover 定价（云端） |

**工具：** `brv_query`（搜索知识树）、`brv_curate`（存储事实/决策/模式）、`brv_status`（CLI 版本 + 树统计）

**设置：**
```bash
# 首先安装 CLI
curl -fsSL https://byterover.dev/install.sh | sh

# 然后配置 Hermes
hermes memory setup    # 选择 "byterover"
# 或手动：
hermes config set memory.provider byterover
```

**关键特性：**
- 自动 pre-compression extraction（在上下文压缩丢弃内容前保存洞察）
- 知识树存储在 `$HERMES_HOME/byterover/`（按 profile 隔离）
- SOC2 Type II 认证的云端同步（可选）

---

### Supermemory

Semantic 长期记忆，支持 profile recall、semantic search、显式 memory 工具，以及通过 Supermemory graph API 在 session 结束时摄取对话。

| | |
|---|---|
| **最佳适用场景** | 需要用户画像和 session 级别图谱构建的 semantic recall |
| **需要** | `pip install supermemory` + [API key](https://supermemory.ai) |
| **数据存储** | Supermemory Cloud |
| **成本** | Supermemory 定价 |

**工具：** `supermemory_store`（保存显式记忆）、`supermemory_search`（semantic similarity search）、`supermemory_forget`（按 ID 或 best-match query 遗忘）、`supermemory_profile`（持久化画像 + 最近上下文）

**设置：**
```bash
hermes memory setup    # 选择 "supermemory"
# 或手动：
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=[REDACTED]' >> ~/.hermes/.env
```

**配置：** `$HERMES_HOME/supermemory.json`

| Key | Default | Description |
|-----|---------|-------------|
| `container_tag` | `hermes` | 用于搜索和写入的 container tag。支持 `{identity}` 模板用于按 profile 隔离的 tag。 |
| `auto_recall` | `true` | 在 turn 前注入相关记忆上下文 |
| `auto_capture` | `true` | 在每次响应后存储清理后的 user-assistant turn |
| `max_recall_results` | `10` | 格式化为上下文的最大回忆条目数 |
| `profile_frequency` | `50` | 在首次 turn 和每 N 个 turn 时包含 profile 事实 |
| `capture_mode` | `all` | 默认跳过微小或琐碎的 turn |
| `search_mode` | `hybrid` | 搜索模式：`hybrid`、`memories` 或 `documents` |
| `api_timeout` | `5.0` | SDK 和 ingest 请求的超时时间 |

**环境变量：** `SUPERMEMORY_API_KEY`（必需）、`SUPERMEMORY_CONTAINER_TAG`（覆盖配置）。

**关键特性：**
- 自动上下文围栏 —— 从捕获的 turn 中剥离回忆的记忆，防止递归记忆污染
- Session 结束时的对话摄取，用于更丰富的图谱级知识构建
- Profile 事实在首次 turn 和可配置间隔时注入
- 琐碎消息过滤（跳过 "ok"、"thanks" 等）
- **按 profile 隔离的 container** —— 在 `container_tag` 中使用 `{identity}`（例如 `hermes-{identity}` → `hermes-coder`）以按 Hermes profile 隔离记忆
- **多 container 模式** —— 启用 `enable_custom_container_tags` 并配合 `custom_containers` 列表，允许 agent 跨命名 container 读写。自动操作（sync、prefetch）仍保留在主 container 上。

<details>
<summary>多 container 示例</summary>

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

## Provider 对比

| Provider | 存储 | 成本 | 工具 | 依赖 | 独特特性 |
|----------|---------|------|-------|-------------|----------------|
| **Honcho** | Cloud | 付费 | 5 | `honcho-ai` | Dialectic 用户建模 + session-scoped 上下文 |
| **OpenViking** | 自托管 | 免费 | 5 | `openviking` + server | 文件系统层级 + 分层加载 |
| **Mem0** | Cloud | 付费 | 3 | `mem0ai` | 服务端 LLM 提取 |
| **Hindsight** | Cloud/Local | 免费/付费 | 3 | `hindsight-client` | 知识图谱 + reflect 综合 |
| **Holographic** | Local | 免费 | 2 | 无 | HRR 代数 + 信任评分 |
| **RetainDB** | Cloud | $20/月 | 5 | `requests` | Delta 压缩 |
| **ByteRover** | Local/Cloud | 免费/付费 | 3 | `brv` CLI | Pre-compression extraction |
| **Supermemory** | Cloud | 付费 | 4 | `supermemory` | 上下文围栏 + session 图谱摄取 + 多 container |

## Profile 隔离

每个 provider 的数据按 [profile](/user-guide/profiles) 隔离：

- **本地存储 provider**（Holographic、ByteRover）使用按 profile 不同的 `$HERMES_HOME/` 路径
- **配置文件 provider**（Honcho、Mem0、Hindsight、Supermemory）将配置存储在 `$HERMES_HOME/` 中，因此每个 profile 有自己的凭证
- **云端 provider**（RetainDB）自动推导按 profile 隔离的项目名称
- **环境变量 provider**（OpenViking）通过每个 profile 的 `.env` 文件配置

## 构建 Memory Provider

参见 [Developer Guide: Memory Provider Plugins](/developer-guide/memory-provider-plugin) 了解如何创建你自己的 provider。
