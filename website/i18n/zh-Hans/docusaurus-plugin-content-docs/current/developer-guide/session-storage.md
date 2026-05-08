---
sidebar_position: 6
title: "会话存储"
description: "SQLite 架构、FTS5、会话谱系、SessionDB、SessionStore 和状态管理"
---

# 会话存储

Hermes 使用基于 SQLite 的会话存储，带全文搜索（FTS5）、谱系跟踪和跨压缩的父子关系。

## 主要文件

| 文件 | 用途 |
|------|------|
| `hermes_state.py` | SessionDB — SQLite 会话和状态存储，带 FTS5 |
| `gateway/session.py` | SessionStore — 网关会话管理和 SessionDB 包装 |
| `gateway/delivery.py` | 投递逻辑和会话隔离 |

## SQLite 模式

### sessions 表

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES sessions(session_id),
    lineage_id TEXT NOT NULL,  -- Shared by all compress children
    platform TEXT,
    chat_id TEXT,
    chat_type TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    model TEXT,
    provider TEXT,
    message_count INTEGER DEFAULT 0,
    token_count INTEGER,
    last_active REAL,
    terminated INTEGER DEFAULT 0,
    FOREIGN KEY (parent_id) REFERENCES sessions(session_id)
);

CREATE INDEX idx_sessions_platform_chat ON sessions(platform, chat_id, chat_type);
CREATE INDEX idx_sessions_lineage ON sessions(lineage_id);
CREATE INDEX idx_sessions_updated ON sessions(updated_at DESC);
```

### messages 表

```sql
CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    token_count INTEGER,
    created_at REAL NOT NULL,
    seq INTEGER NOT NULL,  -- Order within session
    metadata TEXT,  -- JSON: tool_calls, reasoning, etc.
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX idx_messages_session ON messages(session_id, seq);
CREATE INDEX idx_messages_role ON messages(session_id, role);
```

### FTS5 虚拟表

```sql
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
```

FTS5 提供快速全文搜索，而 porter stemmer 处理英语词干。

### 状态表

```sql
CREATE TABLE state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL
);
```

用于存储全局状态如会话计数。

## SessionDB API

```python
from hermes_state import SessionDB

db = SessionDB()

# Create session
session_id = db.create_session(
    platform="telegram",
    chat_id="123456",
    chat_type="private",
    model="anthropic/claude-sonnet-4.6",
)

# Add message
db.add_message(session_id, "user", "Hello", seq=1)
db.add_message(session_id, "assistant", "Hi there!", seq=2)

# Search
results = db.search_messages("hello", limit=10)

# Get history
messages = db.get_messages(session_id)

# Compression — create child session
child_id = db.create_session(
    parent_id=session_id,
    lineage_id=db.get_session(session_id)["lineage_id"],
    ...
)
```

## 会话谱系

每次压缩创建新会话并共享 `lineage_id`：

```text
Session A (lineage: abc123)
  └── Compress → Session B (lineage: abc123)
        └── Compress → Session C (lineage: abc123)
```

这允许：

- `/resume` 列出所有谱系成员
- `session_search` 在整个历史中搜索
- 将来可能的反向遍历

## FTS5 搜索

```python
# Full-text search across all messages
results = db.search_messages("fix bug", limit=20)

# Filter by lineage
session = db.get_session(session_id)
results = db.search_messages(
    "error",
    lineage_id=session["lineage_id"]
)
```

FTS5 使用porter stemmer，所以搜索"running"也匹配"ran"、"run"等。

## 事务和并发

SessionDB 使用：

- **WAL 模式**：允许并发读取而无需写入锁定
- **原子事务**：确保消息和会话创建一起成功或失败
- **超时**：防止长时间运行的写入阻塞

```python
# Atomic multi-table operation
db.with_transaction(lambda: [
    db.create_session(...),
    db.add_message(session_id, ...),
    db.update_session_stats(session_id, ...),
])
```

## 网关集成

`gateway/session.py` 包装 SessionDB：

```python
from gateway.session import SessionStore

store = SessionStore()

# Gateway uses session keys
session_key = store.build_session_key(
    platform="telegram",
    chat_type="private",
    chat_id="123456",
)
session = store.get_or_create_session(session_key)

# Messages are persisted automatically
store.add_message(session["session_id"], "user", "Hello")
```

### 会话键格式

```
agent:main:{platform}:{chat_type}:{chat_id}
```

例如：`agent:main:telegram:private:123456789`

线程感知平台（Telegram 论坛主题、Discord 线程）在 chat_id 中包含线程 ID。

## 内存刷新

内存更改在每次轮次后刷新到磁盘：

1. MEMORY.md 更新
2. USER.md 更新
3. SessionDB 消息持久化

这确保即使进程崩溃，最近的对话也被保留。

## 清理

会话在以下情况下被标记为已终止：

- 用户发送 `/new`
- 会话超时
- 显式 `/reset`

已终止的会话保留用于历史搜索但不会出现在活跃会话列表中。

## 迁移

SessionDB 在架构更改时自动迁移：

```python
db = SessionDB()
db.migrate()  # Runs any pending migrations
```

迁移是向后兼容的 — 旧版本可以读取新版数据库。

## 配置

会话存储位置由 `hermes_constants.py` 管理：

```python
from hermes_constants import get_hermes_home

db_path = get_hermes_home() / "state.db"
```

每个配置文件有自己的 `state.db`。
