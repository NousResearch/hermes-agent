---
sidebar_position: 4
title: "记忆提供者"
description: "外部记忆提供者插件 — Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory"
---

# 记忆提供者

Hermes Agent 附带 8 个外部记忆提供者插件，为 Agent 提供超越内置 MEMORY.md 和 USER.md 的持久化跨会话知识。**一次只能激活一个**外部提供者——内置记忆始终与其一起激活。

## 快速开始

```bash
hermes memory setup      # 交互式选择器 + 配置
hermes memory status     # 检查当前激活的
hermes memory off        # 禁用外部提供者
```

你也可以通过 `hermes plugins` → Provider Plugins → Memory Provider 选择活动的记忆提供者。

或在 `~/.hermes/config.yaml` 中手动设置：

```yaml
memory:
  provider: openviking   # 或 honcho、mem0、hindsight、holographic、retaindb、byterover、supermemory
```

## 工作原理

当记忆提供者激活时，Hermes 自动：

1. **注入提供者上下文**到系统提示（提供者知道什么）
2. **在每轮之前预取**相关记忆（后台、非阻塞）
3. **在每轮响应后同步**对话轮到提供者
4. **在会话结束时提取**记忆（对于支持的提供者）
5. **镜像内置记忆写入**到外部提供者
6. **添加提供者特定工具**以便 Agent 可以搜索、存储和管理记忆

内置记忆（MEMORY.md / USER.md）继续完全像以前一样工作。外部提供者是附加的。

## 可用提供者

### Honcho

具有辩证推理的 AI 原生跨会话用户建模、会话作用域上下文注入、语义搜索和持久化结论。基础上下文现在包括会话摘要以及用户表示和对等卡片，让 Agent 能够了解已经讨论过的内容。

| | |
|---|---|
| **最适合** | 具有跨会话上下文的多 Agent 系统、用户-Agent 对齐 |
| **需要** | `pip install honcho-ai` + [API 密钥](https://app.honcho.dev) 或自托管实例 |
| **数据存储** | Honcho Cloud 或自托管 |
| **费用** | Honcho 定价（云）/ 免费（自托管） |

**工具（5个）：** `honcho_profile`（读取/更新对等卡片）、`honcho_search`（语义搜索）、`honcho_context`（会话上下文——摘要、表示、卡片、消息）、`honcho_reasoning`（LLM 综合）、`honcho_conclude`（创建/删除结论）

**架构：** 两层上下文注入——基础层（会话摘要 + 表示 + 对等卡片，按 `contextCadence` 刷新）加上辩证补充（LLM 推理，按 `dialecticCadence` 刷新）。辩证自动根据是否存在基础上下文选择冷启动提示（一般用户事实）还是暖提示（会话作用域上下文）。

**三个独立配置旋钮**独立控制成本和深度：

- `contextCadence` — 基础层刷新频率（API 调用频率）
- `dialecticCadence` — 辩证 LLM 触发频率（LLM 调用频率）
- `dialecticDepth` — 每次辩证调用中的 `.chat()` 传递次数（1-3，推理深度）

**设置向导：**
```bash
hermes honcho setup        # （遗留命令）
# 或
hermes memory setup        # 选择 "honcho"
```

**配置：** `$HERMES_HOME/honcho.json`（profile 本地）或 `~/.honcho/config.json`（全局）。解析顺序：`$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`。参见 [配置参考](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md) 和 [Honcho 集成指南](https://docs.honcho.dev/v3/guides/integrations/hermes)。

<details>
<summary>完整配置参考</summary>

| 键 | 默认 | 描述 |
|-----|---------|-------------|
| `apiKey` | -- | 来自 [app.honcho.dev](https://app.honcho.dev) 的 API 密钥 |
| `baseUrl` | -- | 自托管 Honcho 的基础 URL |
| `peerName` | -- | 用户对等身份 |
| `aiPeer` | host key | AI 对等身份（每个 profile 一个） |
| `workspace` | host key | 共享工作区 ID |
| `contextTokens` | `null`（无上限） | 每轮自动注入上下文的令牌预算。按词边界截断 |
| `contextCadence` | `1` | `context()` API 调用之间的最小轮数（基础层刷新） |
| `dialecticCadence` | `2` | `peer.chat()` LLM 调用之间的最小轮数。推荐 1-5。仅适用于 `hybrid`/`context` 模式 |
| `dialecticDepth` | `1` | 每次辩证调用的 `.chat()` 传递次数。夹紧 1-3。传递 0：冷/暖提示，传递 1：自我审计，传递 2：调停 |
| `dialecticDepthLevels` | `null` | 每次传递的可选推理级别数组，例如 `["minimal", "low", "medium"]`。覆盖比例默认值 |
| `dialecticReasoningLevel` | `'low'` | 基础推理级别：`minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | 为 `true` 时，模型可以通过工具参数按调用覆盖推理级别 |
| `dialecticMaxChars` | `600` | 注入到系统提示的辩证结果的最大字符数 |
| `recallMode` | `'hybrid'` | `hybrid`（自动注入 + 工具）、`context`（仅注入）、`tools`（仅工具） |
| `writeFrequency` | `'async'` | 何时刷新消息：`async`（后台线程）、`turn`（同步）、`session`（结束时批量）或整数 N |
| `saveMessages` | `true` | 是否将消息持久化到 Honcho API |
| `observationMode` | `'directional'` | `directional`（全部打开）或 `unified`（共享池）。用 `observation` 对象覆盖 |
| `messageMaxChars` | `25000` | 每条消息的最大字符数（超过时分段） |
| `dialecticMaxInputChars` | `10000` | 辩证查询输入到 `peer.chat()` 的最大字符数 |
| `sessionStrategy` | `'per-directory'` | `per-directory`、`per-repo`、`per-session`、`global` |

</details>

<details>
<summary>最小 honcho.json（云）</summary>

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
<summary>最小 honcho.json（自托管）</summary>

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
如果你之前用过 `hermes honcho setup`，你的配置和所有服务器端数据都完好。只需通过设置向导重新启用，或手动设置 `memory.provider: honcho` 通过新系统重新激活。
:::

**多对等设置：**

Honcho 将对话建模为交换消息的对等——一个用户对等加上每个 Hermes profile 的一个 AI 对等，全部共享一个工作区。工作区是共享环境：所有 Hermes profile 在一个工作区下看到相同的用户身份。

映射：

| 概念 | 它是什么 |
|---------|-----------|
| **工作区** | 共享环境。一个工作区下的所有 Hermes profile 看到相同的用户身份。 |
| **用户对等**（`peerName`） | 人类。工作区内跨 profile 共享。 |
| **AI 对等**（`aiPeer`） | 每个 Hermes profile 一个。Host key `hermes` → 默认；其他为 `hermes.<profile>`。 |
| **观察** | 每个对等的切换，控制 Honcho 从谁的消息中建模什么。`directional`（默认，全部四个打开）或 `unified`（单观察者池）。 |

### 新 profile，新 Honcho 对等

```bash
hermes profile create coder --clone
```

`--clone` 在 `honcho.json` 中创建带有 `aiPeer: "coder"` 的 `hermes.coder` host 块，共享 `workspace`、继承的 `peerName`、`recallMode`、`writeFrequency`、`observation` 等。AI 对等被急切地在 Honcho 中创建，所以在第一条消息之前就存在。

### 现有 profile，回填 Honcho 对等

```bash
hermes honcho sync
```

扫描每个 Hermes profile，为任何没有 host 块的 profile 创建 host 块，从默认 `hermes` 块继承设置，并急切地创建新的 AI 对等。幂等——跳过已有 host 块的 profile。

### 每 profile 观察

每个 host 块可以独立覆盖观察配置。示例：一个代码聚焦的 profile，AI 对等观察用户但不自我建模：

```json
"hermes.coder": {
  "aiPeer": "coder",
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": false, "observeOthers": true }
  }
}
```

**观察切换（每对等一组）：**

| 切换 | 效果 |
|--------|--------|
| `observeMe` | Honcho 从此对等自己的消息中构建表示 |
| `observeOthers` | 此对等观察另一个对等的消息（提供跨对等辩证） |

通过 `observationMode` 的预设：

- **`"directional"`**（默认）— 全部四个标志打开。完全相互观察；启用跨对等辩证。
- **`"unified"`** — 用户 `observeMe: true`，AI `observeOthers: true`，其余为 false。单观察者池；AI 为用户建模但不为自己建模，用户对等只自我建模。

通过 [Honcho 仪表板](https://app.honcho.dev) 设置的服务器端切换优先于本地默认值——在会话初始化时同步回来。

参见 [Honcho 页面](./honcho.md#observation-directional-vs-unified) 获取完整观察参考。

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

参见 [配置参考](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md) 和 [Honcho 集成指南](https://docs.honcho.dev/v3/guides/integrations/hermes)。

---

### OpenViking

字节跳动的 Volcano Engine 上下文数据库，具有文件系统式知识层次、分层检索和自动记忆提取为 6 个类别。

| | |
|---|---|
| **最适合** | 具有结构化浏览的自托管知识管理 |
| **需要** | `pip install openviking` + 运行中的服务器 |
| **数据存储** | 自托管（本地或云） |
| **费用** | 免费（开源，AGPL-3.0） |

**工具：** `viking_search`（语义搜索）、`viking_read`（分层：摘要/概述/完整）、`viking_browse`（文件系统导航）、`viking_remember`（存储事实）、`viking_add_resource`（摄入 URL/文档）

**设置：**
```bash
# 先启动 OpenViking 服务器
pip install openviking
openviking-server

# 然后配置 Hermes
hermes memory setup    # 选择 "openviking"
# 或手动：
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

**关键特性：**
- 分层上下文加载：L0（~100 令牌）→ L1（~2k）→ L2（完整）
- 会话提交时自动记忆提取（profile、偏好、实体、事件、案例、模式）
- `viking://` URI 方案用于层次知识浏览

---

### Mem0

服务器端 LLM 事实提取，具有语义搜索、重排和自动去重。

| | |
|---|---|
| **最适合** | 免手记忆管理 — Mem0 自动处理提取 |
| **需要** | `pip install mem0ai` + API 密钥 |
| **数据存储** | Mem0 Cloud |
| **费用** | Mem0 定价 |

**工具：** `mem0_profile`（所有存储的记忆）、`mem0_search`（语义搜索 + 重排）、`mem0_conclude`（存储逐字事实）

**设置：**
```bash
hermes memory setup    # 选择 "mem0"
# 或手动：
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**配置：** `$HERMES_HOME/mem0.json`

| 键 | 默认 | 描述 |
|-----|---------|-------------|
| `user_id` | `hermes-user` | 用户标识符 |
| `agent_id` | `hermes` | Agent 标识符 |

---

### Hindsight

具有知识图谱、实体解析和多策略检索的长期记忆。`hindsight_reflect` 工具提供其他提供者无法提供的跨记忆综合。自动保留完整对话轮次（包括工具调用），带有会话级文档跟踪。

| | |
|---|---|
| **最适合** | 具有实体关系的知识图谱召回 |
| **需要** | 云：来自 [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io) 的 API 密钥。本地：LLM API 密钥（OpenAI、Groq、OpenRouter 等） |
| **数据存储** | Hindsight Cloud 或本地嵌入式 PostgreSQL |
| **费用** | Hindsight 定价（云）或免费（本地） |

**工具：** `hindsight_retain`（存储并提取实体）、`hindsight_recall`（多策略搜索）、`hindsight_reflect`（跨记忆综合）

**设置：**
```bash
hermes memory setup    # 选择 "hindsight"
# 或手动：
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

设置向导自动安装依赖，并且只安装所选模式需要的内容（云用 `hindsight-client`，本地用 `hindsight-all`）。需要 `hindsight-client >= 0.4.22`（会话开始时如果过时会自动升级）。

**本地模式 UI：** `hindsight-embed -p hermes ui start`

**配置：** `$HERMES_HOME/hindsight/config.json`

| 键 | 默认 | 描述 |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` 或 `local` |
| `bank_id` | `hermes` | 记忆库标识符 |
| `recall_budget` | `mid` | 召回详尽程度：`low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid`（上下文 + 工具）、`context`（仅自动注入）、`tools`（仅工具） |
| `auto_retain` | `true` | 自动保留对话轮次 |
| `auto_recall` | `true` | 在每轮之前自动召回记忆 |
| `retain_async` | `true` | 在服务器上异步处理保留 |
| `retain_context` | `conversation between Hermes Agent and the User` | 保留记忆的上下文标签 |
| `retain_tags` | — | 应用于保留记忆的默认标签；与每调用工具标签合并 |
| `retain_source` | — | 附加到保留记忆的可选 `metadata.source` |
| `retain_user_prefix` | `User` | 自动保留转录中用户轮次前使用的标签 |
| `retain_assistant_prefix` | `Assistant` | 自动保留转录中助手轮次前使用的标签 |
| `recall_tags` | — | 召回时筛选的标签 |

参见 [插件 README](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md) 获取完整配置参考。

---

### Holographic

具有 FTS5 全文搜索、信任评分和 HRR（全息简化表示）用于组合代数查询的本地 SQLite 事实存储。

| | |
|---|---|
| **最适合** | 无外部依赖的高级检索本地-only 记忆 |
| **需要** | 无（SQLite 始终可用）。NumPy 可选用于 HRR 代数。 |
| **数据存储** | 本地 SQLite |
| **费用** | 免费 |

**工具：** `fact_store`（9 个操作：添加、搜索、探测、相关、推理、矛盾、更新、移除、列表）、`fact_feedback`（有用/无用评分，训练信任评分）

**设置：**
```bash
hermes memory setup    # 选择 "holographic"
# 或手动：
hermes config set memory.provider holographic
```

**配置：** `plugins.hermes-memory-store` 下的 `config.yaml`

| 键 | 默认 | 描述 |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite 数据库路径 |
| `auto_extract` | `false` | 会话结束时自动提取事实 |
| `default_trust` | `0.5` | 默认信任评分（0.0-1.0） |

**独特能力：**
- `probe` — 特定实体的代数召回（关于一个人/事物的所有事实）
- `reason` — 跨多个实体的组合 AND 查询
- `contradict` — 冲突事实的自动检测
- 具有非对称反馈的信任评分（+0.05 有用 / -0.10 无用）

---

### RetainDB

具有混合搜索（向量 + BM25 + 重排）、7 种记忆类型和增量压缩的云记忆 API。

| | |
|---|---|
| **最适合** | 已经使用 RetainDB 基础设施的团队 |
| **需要** | RetainDB 账户 + API 密钥 |
| **数据存储** | RetainDB Cloud |
| **费用** | $20/月 |

**工具：** `retaindb_profile`（用户 profile）、`retaindb_search`（语义搜索）、`retaindb_context`（任务相关上下文）、`retaindb_remember`（存储并分类重要性）、`retaindb_forget`（删除记忆）

**设置：**
```bash
hermes memory setup    # 选择 "retaindb"
# 或手动：
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

---

### ByteRover

通过 `brv` CLI 实现持久化记忆——具有分层检索的层次知识树（模糊文本 → LLM 驱动搜索）。本地优先，可选云同步。

| | |
|---|---|
| **最适合** | 想要便携本地优先记忆和 CLI 的开发者 |
| **需要** | ByteRover CLI（`npm install -g byterover-cli` 或[安装脚本](https://byterover.dev)） |
| **数据存储** | 本地（默认）或 ByteRover Cloud（可选同步） |
| **费用** | 免费（本地）或 ByteRover 定价（云） |

**工具：** `brv_query`（搜索知识树）、`brv_curate`（存储事实/决策/模式）、`brv_status`（CLI 版本 + 树统计）

**设置：**
```bash
# 先安装 CLI
curl -fsSL https://byterover.dev/install.sh | sh

# 然后配置 Hermes
hermes memory setup    # 选择 "byterover"
# 或手动：
hermes config set memory.provider byterover
```

**关键特性：**
- 自动预压缩提取（在上下文压缩丢弃之前保存洞察）
- 知识树存储在 `$HERMES_HOME/byterover/`（profile 作用域）
- SOC2 Type II 认证云同步（可选）

---

### Supermemory

具有 profile 召回、语义搜索、显式记忆工具和通过 Supermemory 图 API 的会话结束对话摄入的语义长期记忆。

| | |
|---|---|
| **最适合** | 具有用户 profile 和会话级图构建的语义召回 |
| **需要** | `pip install supermemory` + [API 密钥](https://supermemory.ai) |
| **数据存储** | Supermemory Cloud |
| **费用** | Supermemory 定价 |

**工具：** `supermemory_store`（保存显式记忆）、`supermemory_search`（语义相似搜索）、`supermemory_forget`（按 ID 或最佳匹配查询遗忘）、`supermemory_profile`（持久化 profile + 最近上下文）

**设置：**
```bash
hermes memory setup    # 选择 "supermemory"
# 或手动：
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

**配置：** `$HERMES_HOME/supermemory.json`

| 键 | 默认 | 描述 |
|-----|---------|-------------|
| `container_tag` | `hermes` | 用于搜索和写入的容器标签。支持 `{identity}` 模板用于 profile 作用域标签。 |
| `auto_recall` | `true` | 在轮次之前注入相关记忆上下文 |
| `auto_capture` | `true` | 在每轮响应后存储清理的用户-助手轮次 |
| `max_recall_results` | `10` | 格式化为上下文的最多召回项数 |
| `profile_frequency` | `50` | 在第一轮和每 N 轮包含 profile 事实 |
| `capture_mode` | `all` | 默认跳过微小或琐碎的轮次 |
| `search_mode` | `hybrid` | 搜索模式：`hybrid`、`memories` 或 `documents` |
| `api_timeout` | `5.0` | SDK 和摄入请求的超时 |

**环境变量：** `SUPERMEMORY_API_KEY`（必需）、`SUPERMEMORY_CONTAINER_TAG`（覆盖配置）。

**关键特性：**
- 自动上下文隔离——在捕获的轮次中剥离召回的记忆以防止递归记忆污染
- 会话结束对话摄入以获得更丰富的图级知识构建
- Profile 事实注入在第一轮和可配置间隔
- 琐碎消息过滤（跳过"好的"、"谢谢"等）
- **Profile 作用域容器** — 在 `container_tag` 中使用 `{identity}`（例如 `hermes-{identity}` → `hermes-coder`）隔离每个 Hermes profile 的记忆
- **多容器模式** — 启用 `enable_custom_container_tags` 配合 `custom_containers` 列表让 Agent 跨命名容器读写。自动操作（同步、预取）保持在主容器上。

<details>
<summary>多容器示例</summary>

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "shared-knowledge"],
  "custom_container_instructions": "使用 project-alpha 获取编码上下文。"
}
```

</details>

**支持：** [Discord](https://supermemory.link/discord) · [support@supermemory.com](mailto:support@supermemory.com)

---

## 提供者对比

| 提供者 | 存储 | 费用 | 工具 | 依赖 | 独特特性 |
|----------|---------|------|-------|-------------|----------------|
| **Honcho** | 云 | 付费 | 5 | `honcho-ai` | 辩证用户建模 + 会话作用域上下文 |
| **OpenViking** | 自托管 | 免费 | 5 | `openviking` + 服务器 | 文件系统层次 + 分层加载 |
| **Mem0** | 云 | 付费 | 3 | `mem0ai` | 服务器端 LLM 提取 |
| **Hindsight** | 云/本地 | 免费/付费 | 3 | `hindsight-client` | 知识图谱 + 反思综合 |
| **Holographic** | 本地 | 免费 | 2 | 无 | HRR 代数 + 信任评分 |
| **RetainDB** | 云 | $20/月 | 5 | `requests` | 增量压缩 |
| **ByteRover** | 本地/云 | 免费/付费 | 3 | `brv` CLI | 预压缩提取 |
| **Supermemory** | 云 | 付费 | 4 | `supermemory` | 上下文隔离 + 会话图摄入 + 多容器 |

## Profile 隔离

每个提供者的数据按 [profile](/docs/user-guide/profiles) 隔离：

- **本地存储提供者**（Holographic、ByteRover）使用因 profile 而异的 `$HERMES_HOME/` 路径
- **配置文件提供者**（Honcho、Mem0、Hindsight、Supermemory）在 `$HERMES_HOME/` 存储配置，所以每个 profile 有自己的凭据
- **云提供者**（RetainDB）自动派生 profile 作用域项目名称
- **环境变量提供者**（OpenViking）通过每个 profile 的 `.env` 文件配置

## 构建记忆提供者

参见 [开发者指南：记忆提供者插件](/docs/developer-guide/memory-provider-plugin) 了解如何创建你自己的提供者。
