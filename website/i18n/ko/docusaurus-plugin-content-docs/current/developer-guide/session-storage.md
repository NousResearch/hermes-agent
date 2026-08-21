# 세션 저장소

Hermes Agent는 SQLite 데이터베이스(`~/.hermes/state.db`)를 사용하여 CLI와 게이트웨이 세션 전반에서 세션 메타데이터, 전체 메시지 기록, 모델 구성을 영속화합니다. 이는 기존의 세션별 JSONL 파일 방식을 대체합니다.

소스 파일: `hermes_state.py`


## 아키텍처 개요

```
~/.hermes/state.db (SQLite, WAL mode)
├── sessions              — Session metadata, token counts, billing
├── messages              — Full message history per session
├── session_model_usage   — Per-model/per-task usage attribution rows
├── messages_fts          — FTS5 virtual table (content + tool_name + tool_calls)
├── messages_fts_trigram  — FTS5 virtual table with trigram tokenizer (CJK / substring search)
├── messages_fts_cjk      — FTS5 virtual table with cjk_unicode61 tokenizer
├── state_meta            — Key/value metadata table
├── gateway_routing       — Gateway routing metadata
├── compression_locks     — Cross-process compression locking
├── async_delegations     — Async delegation bookkeeping
└── schema_version        — Single-row table tracking migration state
```

핵심 설계 결정:
- 동시 독자와 단일 기록자(게이트웨이 다중 플랫폼)를 위한 **WAL 모드**
- 모든 세션 메시지를 빠르게 검색하기 위한 **FTS5 가상 테이블**
- `parent_session_id` 체인을 통한 **세션 계보**(압축으로 트리거된 분할)
- 플랫폼 필터링을 위한 **소스 태깅**(`cli`, `telegram`, `discord` 등)
- 배치 실행기와 RL trajectory는 여기에 저장하지 않음(별도 시스템)


## SQLite 스키마

### Sessions 테이블

요약본입니다. 전체 최신 컬럼 목록은 `hermes_state.py`의 `SCHEMA_SQL`을 참조하세요. 여기에는 `session_key`, `chat_id`, `chat_type`, `thread_id`, `display_name`, `origin_json`, `expiry_finalized`, 작업공간 필드 `cwd` / `git_branch` / `git_repo_root`, handoff 및 압축 실패 필드, `profile_name`, `rewind_count`, `archived`, `pinned` 등 게이트웨이 라우팅 메타데이터도 포함됩니다.

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
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
    -- ... additional gateway/workspace/handoff/compression columns ...
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique
    ON sessions(title) WHERE title IS NOT NULL;
```

### Messages 테이블

요약본입니다. 전체 스키마에는 `effect_disposition`, `platform_message_id`, `observed`, `active`, `compacted`, `api_content`, `display_kind`, `display_metadata`도 포함됩니다.

```sql
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT
    -- ... additional display/compaction columns ...
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
```

참고:
- `tool_calls`는 JSON 문자열(도구 호출 객체 목록을 직렬화한 값)로 저장됩니다.
- `reasoning_details`, `codex_reasoning_items`, `codex_message_items`는 JSON 문자열로 저장됩니다.
- `reasoning`은 이를 노출하는 provider의 원시 reasoning 텍스트를 저장합니다.
- `api_content`는 byte-fidelity sidecar입니다. `content`와 다를 때 이 메시지에 대해 API로 전송된 정확한 content 문자열(임시 메모리/플러그인 주입, persist override)을 저장합니다. prompt-cache-stable replay를 위해 전송된 wire bytes를 보존합니다. 단, sqlite3가 바인딩할 수 없는 lone surrogate는 예외이며, conversation loop가 모든 outgoing payload에서 이를 삭제합니다. `NULL`은 `content`가 그대로 전송되었음을 의미합니다.
- 타임스탬프는 Unix epoch 부동소수점 값입니다(`time.time()`).

### FTS5 전문 검색

`messages` 테이블의 INSERT, UPDATE, DELETE 시 실행되는 세 개의 trigger를 통해 FTS5 테이블이 동기화됩니다. 현재 trigger는 `state_meta`의 `fts_rebuild_high_water` / `fts_rebuild_progress` marker에 따라 동작하므로 background FTS rebuild가 중복 인덱싱 없이 진행될 수 있으며, 인덱싱된 세 컬럼을 모두 다룹니다. 정확한 SQL은 `hermes_state.py`의 `SCHEMA_SQL`을 참조하세요.

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);
```


## 스키마 버전 및 마이그레이션

현재 스키마 버전: **23**

`schema_version` 테이블은 단일 정수를 저장합니다. 단순한 컬럼 추가는 `_reconcile_columns()`가 live column과 `SCHEMA_SQL`을 비교하여 누락된 컬럼을 ADD하는 선언 방식으로 처리합니다. 버전으로 구분되는 chain은 선언 방식으로 표현할 수 없는 데이터 마이그레이션과 index/FTS 변경에 사용됩니다.

| 버전 | 변경 사항 |
|---------|--------|
| 1 | 초기 스키마(sessions, messages, FTS5) |
| 2 | messages에 `finish_reason` 컬럼 추가 |
| 3 | sessions에 `title` 컬럼 추가 |
| 4 | `title`에 unique index 추가(NULL 허용, non-NULL은 unique해야 함) |
| 5 | billing 컬럼 추가: `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`, `billing_provider`, `billing_base_url`, `billing_mode`, `estimated_cost_usd`, `actual_cost_usd`, `cost_status`, `cost_source`, `pricing_version` |
| 6 | messages에 reasoning 컬럼 추가: `reasoning`, `reasoning_details`, `codex_reasoning_items` |
| 7 | messages에 `reasoning_content` 컬럼 추가 |
| 8 | sessions에 `api_call_count` 컬럼 추가 |
| 9 | Codex Responses message id/phase replay를 위해 messages에 `codex_message_items` 컬럼 추가 |
| 10 | `messages_fts_trigram` 가상 테이블 추가(CJK / substring search용 trigram tokenizer) 및 기존 row backfill |
| 11 | `messages_fts`와 `messages_fts_trigram`을 re-index하여 `tool_name` + `tool_calls`를 포함하고 external-content에서 inline mode로 전환; 기존 trigger 삭제 및 모든 message row backfill |
| 16 | delegate subagent row를 `model_config`의 `$._delegate_from`으로 태깅하여 parent 삭제로 orphan이 된 뒤에도 session picker를 깔끔하게 유지 |
| 18 | Gateway metadata consolidation — `sessions.json`에서 `display_name` / `origin_json` / `expiry_finalized` backfill |
| 20 | 모델별 usage attribution — 과거 세션별 aggregate total에서 `session_model_usage` row seed |
| 22 | Task-dimension usage attribution — `task` 컬럼이 PRIMARY KEY에 참여하도록 `session_model_usage` 재구축 |
| 23 | FTS storage redesign — v11 inline-mode 복사본을 대체하는 external-content FTS table(기존 DB를 위한 opt-in transition) |

위에 나열되지 않은 버전은 `_reconcile_columns()`가 처리한 선언적 컬럼 추가입니다(버전 bump만 수행되며 데이터 마이그레이션은 없음).

선언적 컬럼 추가는 `ALTER TABLE ADD COLUMN`을 try/except로 감싸 이미 컬럼이 존재하는 경우(idempotent)를 처리합니다. 각 migration block이 성공한 뒤 버전 번호를 bump합니다.


## 쓰기 경합 처리

여러 hermes 프로세스(게이트웨이 + CLI 세션 + worktree agent)가 하나의 `state.db`를 공유합니다. `SessionDB` 클래스는 다음을 사용해 쓰기 경합을 처리합니다.

- 기본 30초 대신 **짧은 SQLite timeout**(1초)
- **애플리케이션 수준 retry** 및 random jitter(20–150ms, 최대 15회 retry)
- 트랜잭션 시작 시 lock contention을 드러내는 **BEGIN IMMEDIATE** transaction
- 성공적인 write 50회마다 **주기적인 WAL checkpoint**(PASSIVE mode)

이 방식은 SQLite의 deterministic internal backoff로 인해 경쟁하는 모든 writer가 같은 간격으로 retry하는 "convoy effect"를 방지합니다.

```
_WRITE_MAX_RETRIES = 15
_WRITE_RETRY_MIN_S = 0.020   # 20ms
_WRITE_RETRY_MAX_S = 0.150   # 150ms
_CHECKPOINT_EVERY_N_WRITES = 50
```


## 일반 작업

### 초기화

```python
from hermes_state import SessionDB

db = SessionDB()                           # Default: ~/.hermes/state.db
db = SessionDB(db_path=Path("/tmp/test.db"))  # Custom path
```

### 세션 생성 및 관리

```python
# Create a new session
db.create_session(
    session_id="sess_abc123",
    source="cli",
    model="anthropic/claude-sonnet-4.6",
    user_id="user_1",
    parent_session_id=None,  # or previous session ID for lineage
)

# End a session
db.end_session("sess_abc123", end_reason="user_exit")

# Reopen a session (clear ended_at/end_reason)
db.reopen_session("sess_abc123")
```

### 메시지 저장

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

### 메시지 검색

```python
# Raw messages with all metadata
messages = db.get_messages("sess_abc123")

# OpenAI conversation format (for API replay)
conversation = db.get_messages_as_conversation("sess_abc123")
# Returns: [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
```

### 세션 제목

```python
# Set a title (must be unique among non-NULL titles)
db.set_session_title("sess_abc123", "Fix Docker Build")

# Resolve by title (returns most recent in lineage)
session_id = db.resolve_session_by_title("Fix Docker Build")

# Auto-generate next title in lineage
next_title = db.get_next_title_in_lineage("Fix Docker Build")
# Returns: "Fix Docker Build #2"
```


## 전문 검색

`search_messages()` 메서드는 사용자 입력을 자동으로 정제하면서 FTS5 query syntax를 지원합니다.

### 기본 검색

```python
results = db.search_messages("docker deployment")
```

### FTS5 query syntax

| 구문 | 예시 | 의미 |
|--------|---------|---------|
| 키워드 | `docker deployment` | 두 용어 모두(암시적 AND) |
| 인용 구문 | `"exact phrase"` | 정확한 구문 일치 |
| Boolean OR | `docker OR kubernetes` | 둘 중 하나의 용어 |
| Boolean NOT | `python NOT java` | 해당 용어 제외 |
| Prefix | `deploy*` | prefix 일치 |

### 필터링 검색

```python
# Search only CLI sessions
results = db.search_messages("error", source_filter=["cli"])

# Exclude gateway sessions
results = db.search_messages("bug", exclude_sources=["telegram", "discord"])

# Search only user messages
results = db.search_messages("help", role_filter=["user"])
```

### 검색 결과 형식

각 결과에는 다음이 포함됩니다.
- `id`, `session_id`, `role`, `timestamp`
- `snippet` — `>>>match<<<` marker가 포함된 FTS5 생성 snippet
- `context` — 일치 항목 전후의 메시지 1개씩(content는 200자로 truncate됨)
- `source`, `model`, `session_started` — parent session에서 가져온 값

`_sanitize_fts5_query()` 메서드는 다음 edge case를 처리합니다.
- 일치하지 않는 quote와 특수 문자를 제거
- 하이픈이 있는 용어를 quote로 감쌈(`chat-send` → `"chat-send"`)
- 끝에 남은 Boolean operator를 제거(`hello AND` → `hello`)


## 세션 계보

세션은 `parent_session_id`를 통해 chain을 형성할 수 있습니다. 게이트웨이에서 context compression이 session split을 trigger할 때 발생합니다.

### Query: 세션 계보 찾기

```sql
-- Find all ancestors of a session
WITH RECURSIVE lineage AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN lineage l ON s.id = l.parent_session_id
)
SELECT id, title, started_at, parent_session_id FROM lineage;

-- Find all descendants of a session
WITH RECURSIVE descendants AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN descendants d ON s.parent_session_id = d.id
)
SELECT id, title, started_at FROM descendants;
```

### Query: preview가 포함된 최근 세션

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

### Query: token usage 통계

```sql
-- Total tokens by model
SELECT model,
       COUNT(*) as session_count,
       SUM(input_tokens) as total_input,
       SUM(output_tokens) as total_output,
       SUM(estimated_cost_usd) as total_cost
FROM sessions
WHERE model IS NOT NULL
GROUP BY model
ORDER BY total_cost DESC;

-- Sessions with highest token usage
SELECT id, title, model, input_tokens + output_tokens AS total_tokens,
       estimated_cost_usd
FROM sessions
ORDER BY total_tokens DESC
LIMIT 10;
```


## 내보내기 및 정리

```python
# Export a single session with messages
data = db.export_session("sess_abc123")

# Export all sessions (with messages) as list of dicts
all_data = db.export_all(source="cli")

# Delete old sessions (only ended sessions)
deleted_count = db.prune_sessions(older_than_days=90)
deleted_count = db.prune_sessions(older_than_days=30, source="telegram")

# Clear messages but keep the session record
db.clear_messages("sess_abc123")

# Delete session and all messages
db.delete_session("sess_abc123")
```


## 데이터베이스 위치

기본 경로: `~/.hermes/state.db`

이는 `hermes_constants.get_hermes_home()`에서 파생되며, 기본적으로 `~/.hermes/`로 resolve되거나 `HERMES_HOME` environment variable의 값을 사용합니다.

데이터베이스 파일, WAL 파일(`state.db-wal`), shared-memory 파일(`state.db-shm`)은 모두 같은 디렉터리에 생성됩니다.
