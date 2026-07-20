# 会话存储

Hermes Agent 使用 SQLite 数据库（`~/.hermes/state.db`）跨 CLI 和 gateway 会话持久化会话元数据、完整消息历史及模型配置。这替代了早期的逐会话 JSONL 文件方案。

源文件：`hermes_state.py`


## 架构概览

```
~/.hermes/state.db (SQLite, WAL mode)
├── sessions              — 会话元数据、token 计数、计费信息
├── messages              — 每个会话的完整消息历史
├── session_model_usage   — 按模型和任务归因 token/成本
├── gateway_routing       — gateway 会话路由的规范索引
├── compression_locks     — 跨进程压缩租约
├── async_delegations     — 持久化异步委派状态
├── messages_fts          — FTS5 虚拟表（content + tool_name + tool_calls）
├── messages_fts_trigram  — 使用 trigram tokenizer 的 FTS5 虚拟表（CJK / 子串搜索）
├── messages_fts_cjk      — 原生 tokenizer 可用时启用的可选 CJK bigram FTS5 索引
├── state_meta            — 键值元数据表
└── schema_version        — 单行表，跟踪迁移状态
```

关键设计决策：
- **WAL 模式**：支持并发读取 + 单写入（gateway 多平台）
- **FTS5 虚拟表**：跨所有会话消息的快速全文搜索
- **会话血缘**：通过 `parent_session_id` 链实现（压缩触发的会话分割）
- **来源标记**（`cli`、`telegram`、`discord` 等）：用于平台过滤
- 批量运行器和 RL 轨迹不存储于此（独立系统）


## SQLite Schema

### Sessions 表

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    profile_name TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_source_id ON sessions(source, id);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_session_key
    ON sessions(session_key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique
    ON sessions(title) WHERE title IS NOT NULL;
```

### Messages 表

```sql
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_active_null
    ON messages(active) WHERE active IS NULL;
```

说明：
- `tool_calls` 以 JSON 字符串存储（序列化的 tool call 对象列表）
- `reasoning_details`、`codex_reasoning_items` 和 `codex_message_items` 以 JSON 字符串存储
- `reasoning` 存储提供商暴露的原始推理文本
- 时间戳为 Unix epoch 浮点数（`time.time()`）

### 辅助表

- `session_model_usage` 按会话、模型、计费路由和任务归因调用、token 与成本。
- `gateway_routing` 以 `(scope, session_key)` 为键，将每条 gateway 路由记录存为
  JSON。它是规范路由索引；`sessions.json` 仅作为兼容镜像/回退。
- `compression_locks` 用于跨进程协调压缩租约。
- `async_delegations` 持久化委派生命周期、结果、所有权与交付状态。

### FTS5 全文搜索

全新创建的 schema v23 数据库使用 external-content FTS 表，避免在基础索引中
重复存储可搜索文本：

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);
```

INSERT、UPDATE 和 DELETE 触发器使 `content`、`tool_name` 与 `tool_calls`
保持同步。重建期间，触发器使用 `state_meta` 中的
`fts_rebuild_high_water` 和 `fts_rebuild_progress` 标记，只修改已存在于
索引中的行。

trigram 索引及其可选的原生 CJK bigram 替代索引通过视图读取，并排除
`role='tool'` 的行，避免重复索引大型机器载荷。工具消息仍可通过
`messages_fts` 搜索。

已有的 v23 之前数据库继续使用正常工作的 v22 inline FTS 布局，直到用户显式
运行 `hermes sessions optimize-storage`。FTS 布局通过
`state_meta.fts_storage_version` 独立跟踪，因此不执行存储重写也不会阻塞后续
主 schema 迁移。完整触发器定义见 `hermes_state.py` 中的 `FTS_SQL`、
`FTS_TRIGRAM_SQL` 与 `FTS_CJK_*_SQL`。


## Schema 版本与迁移

当前 schema 版本：**23**

`schema_version` 表存储单个整数。简单的列添加由 `_reconcile_columns()` 声明式处理（对比实时列与 `SCHEMA_SQL` 并 ADD 缺失列）。版本门控链保留用于无法声明式表达的数据迁移及索引/FTS 变更：

| 版本 | 变更 |
|------|------|
| 1 | 初始 schema（sessions、messages、FTS5） |
| 2 | 向 messages 添加 `finish_reason` 列 |
| 3 | 向 sessions 添加 `title` 列 |
| 4 | 在 `title` 上添加唯一索引（允许 NULL，非 NULL 必须唯一） |
| 5 | 添加计费列：`cache_read_tokens`、`cache_write_tokens`、`reasoning_tokens`、`billing_provider`、`billing_base_url`、`billing_mode`、`estimated_cost_usd`、`actual_cost_usd`、`cost_status`、`cost_source`、`pricing_version` |
| 6 | 向 messages 添加推理列：`reasoning`、`reasoning_details`、`codex_reasoning_items` |
| 7 | 向 messages 添加 `reasoning_content` 列 |
| 8 | 向 sessions 添加 `api_call_count` 列 |
| 9 | 向 messages 添加 `codex_message_items` 列，用于 Codex Responses 消息 id/phase 重放 |
| 10 | 添加 `messages_fts_trigram` 虚拟表（trigram tokenizer，用于 CJK / 子串搜索）并回填现有行 |
| 11 | 重新索引 `messages_fts` 和 `messages_fts_trigram` 以覆盖 `tool_name` + `tool_calls`，从外部内容模式切换为内联模式；删除旧触发器并回填所有消息行 |
| 16 | 在 `model_config` 中标记委派子 agent 行（`$._delegate_from`），防止父会话删除后孤立的子会话出现在普通会话选择器中 |
| 18 | 整合 gateway 元数据——从 `sessions.json` 回填 `display_name` / `origin_json` / `expiry_finalized` |
| 20 | 按模型归因用量——根据历史 session 聚合总量生成 `session_model_usage` 行 |
| 22 | 按任务维度归因用量——重建 `session_model_usage`，使 `task` 参与主键；历史行保留为主循环用量（`task=''`） |
| 23 | FTS 存储重构——新数据库使用 external-content 索引；已有数据库继续使用 v22 inline 布局，直到运行 `hermes sessions optimize-storage`，FTS 布局版本由 `state_meta` 独立跟踪 |

未在上表列出的版本属于由 `_reconcile_columns()` 处理的声明式列添加（只递增版本，不执行数据迁移）。

声明式列添加使用 `ALTER TABLE ADD COLUMN`，包裹在 try/except 中以处理列已存在的情况（幂等）。必要的 schema 初始化完成后，主 schema 标记会更新为 `SCHEMA_VERSION`。从 v23 开始，可选的 FTS 存储重写使用独立的 `fts_storage_version` 标记，不再阻塞 `schema_version` 更新。


## 写入竞争处理

多个 hermes 进程（gateway + CLI 会话 + worktree agent）共享同一个 `state.db`。`SessionDB` 类通过以下方式处理写入竞争：

- **短 SQLite 超时**（1 秒），而非默认的 30 秒
- **应用层重试**，带随机抖动（20–150ms，最多 15 次重试）
- **BEGIN IMMEDIATE** 事务，在事务开始时暴露锁竞争
- **定期 WAL checkpoint**，每 50 次成功写入执行一次（PASSIVE 模式）

这避免了"护卫效应"——SQLite 确定性内部退避会导致所有竞争写入者在相同间隔重试。

```
_WRITE_MAX_RETRIES = 15
_WRITE_RETRY_MIN_S = 0.020   # 20ms
_WRITE_RETRY_MAX_S = 0.150   # 150ms
_CHECKPOINT_EVERY_N_WRITES = 50
```


## 常用操作

### 初始化

```python
from hermes_state import SessionDB

db = SessionDB()                           # 默认：~/.hermes/state.db
db = SessionDB(db_path=Path("/tmp/test.db"))  # 自定义路径
```

### 创建和管理会话

```python
# 创建新会话
db.create_session(
    session_id="sess_abc123",
    source="cli",
    model="anthropic/claude-sonnet-4.6",
    user_id="user_1",
    parent_session_id=None,  # 或用于血缘追踪的上一个会话 ID
)

# 结束会话
db.end_session("sess_abc123", end_reason="user_exit")

# 重新打开会话（清除 ended_at/end_reason）
db.reopen_session("sess_abc123")
```

### 存储消息

```python
msg_id = db.append_message(
    session_id="sess_abc123",
    role="assistant",
    content="Here's the answer...",
    tool_calls=[{"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}],
    token_count=150,
    finish_reason="stop",
    reasoning="Let me think about this...",
)
```

### 检索消息

```python
# 包含所有元数据的原始消息
messages = db.get_messages("sess_abc123")

# OpenAI 对话格式（用于 API 重放）
conversation = db.get_messages_as_conversation("sess_abc123")
# 返回：[{"role": "user", "content": "..."}, {"role": "assistant", ...}]
```

### 会话标题

```python
# 设置标题（非 NULL 标题中必须唯一）
db.set_session_title("sess_abc123", "Fix Docker Build")

# 按标题解析（返回血缘中最新的）
session_id = db.resolve_session_by_title("Fix Docker Build")

# 自动生成血缘中的下一个标题
next_title = db.get_next_title_in_lineage("Fix Docker Build")
# 返回："Fix Docker Build #2"
```


## 全文搜索

`search_messages()` 方法支持 FTS5 查询语法，并自动对用户输入进行清理。

### 基本搜索

```python
results = db.search_messages("docker deployment")
```

### FTS5 查询语法

| 语法 | 示例 | 含义 |
|------|------|------|
| 关键词 | `docker deployment` | 两个词均包含（隐式 AND） |
| 引号短语 | `"exact phrase"` | 精确短语匹配 |
| 布尔 OR | `docker OR kubernetes` | 任一词 |
| 布尔 NOT | `python NOT java` | 排除词 |
| 前缀 | `deploy*` | 前缀匹配 |

### 过滤搜索

```python
# 仅搜索 CLI 会话
results = db.search_messages("error", source_filter=["cli"])

# 排除 gateway 会话
results = db.search_messages("bug", exclude_sources=["telegram", "discord"])

# 仅搜索用户消息
results = db.search_messages("help", role_filter=["user"])
```

### 搜索结果格式

每条结果包含：
- `id`、`session_id`、`role`、`timestamp`
- `snippet` — FTS5 生成的片段，带 `>>>match<<<` 标记
- `context` — 匹配前后各 1 条消息（内容截断至 200 字符）
- `source`、`model`、`session_started` — 来自父会话

`_sanitize_fts5_query()` 方法处理边缘情况：
- 去除不匹配的引号和特殊字符
- 将含连字符的词包裹在引号中（`chat-send` → `"chat-send"`）
- 移除悬空的布尔运算符（`hello AND` → `hello`）


## 会话血缘

会话可通过 `parent_session_id` 形成链。这发生在 gateway 中上下文压缩触发会话分割时。

### 查询：查找会话血缘

```sql
-- 查找会话的所有祖先
WITH RECURSIVE lineage AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN lineage l ON s.id = l.parent_session_id
)
SELECT id, title, started_at, parent_session_id FROM lineage;

-- 查找会话的所有后代
WITH RECURSIVE descendants AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN descendants d ON s.parent_session_id = d.id
)
SELECT id, title, started_at FROM descendants;
```

### 查询：带预览的最近会话

```sql
SELECT s.*,
    COALESCE(
        (SELECT SUBSTR(m.content, 1, 63)
         FROM messages m
         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
         ORDER BY m.timestamp, m.id LIMIT 1),
        ''
    ) AS preview,
    COALESCE(
        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
        s.started_at
    ) AS last_active
FROM sessions s
ORDER BY s.started_at DESC
LIMIT 20;
```

### 查询：Token 使用统计

```sql
-- 按模型统计总 token 数
SELECT model,
       COUNT(*) as session_count,
       SUM(input_tokens) as total_input,
       SUM(output_tokens) as total_output,
       SUM(estimated_cost_usd) as total_cost
FROM sessions
WHERE model IS NOT NULL
GROUP BY model
ORDER BY total_cost DESC;

-- token 使用量最高的会话
SELECT id, title, model, input_tokens + output_tokens AS total_tokens,
       estimated_cost_usd
FROM sessions
ORDER BY total_tokens DESC
LIMIT 10;
```


## 导出与清理

```python
# 导出单个会话及其消息
data = db.export_session("sess_abc123")

# 导出所有会话（含消息）为字典列表
all_data = db.export_all(source="cli")

# 删除旧会话（仅删除已结束的会话）
deleted_count = db.prune_sessions(older_than_days=90)
deleted_count = db.prune_sessions(older_than_days=30, source="telegram")

# 清除消息但保留会话记录
db.clear_messages("sess_abc123")

# 删除会话及所有消息
db.delete_session("sess_abc123")
```


## 数据库位置

默认路径：`~/.hermes/state.db`

该路径由 `hermes_constants.get_hermes_home()` 推导，默认解析为 `~/.hermes/`，或 `HERMES_HOME` 环境变量的值。

数据库文件、WAL 文件（`state.db-wal`）和共享内存文件（`state.db-shm`）均创建于同一目录。
