# 세션 스토리지 (Session Storage)

Hermes Agent는 CLI와 게이트웨이 세션 전반에 걸쳐 세션 메타데이터, 전체 메시지 기록, 모델 구성을 유지하기 위해 SQLite 데이터베이스(`~/.hermes/state.db`)를 사용합니다. 이는 이전의 세션별 JSONL 파일 저장 방식을 대체합니다.

소스 파일: `hermes_state.py`

## 아키텍처 개요

```
~/.hermes/state.db (SQLite, WAL 모드)
├── sessions              — 세션 메타데이터, 토큰 수, 청구(billing)
├── messages              — 세션별 전체 메시지 기록
├── messages_fts          — FTS5 가상 테이블 (콘텐츠 + 도구 이름 + 도구 호출)
├── messages_fts_trigram  — trigram 토크나이저를 사용한 FTS5 가상 테이블 (CJK / 부분 문자열 검색)
├── state_meta            — 키/값 메타데이터 테이블
└── schema_version        — 마이그레이션 상태를 추적하는 단일 행 테이블
```

주요 설계 결정:
- **WAL 모드**: 다수의 동시 읽기 + 단일 쓰기 (게이트웨이 다중 플랫폼)
- **FTS5 가상 테이블**: 모든 세션 메시지에 대한 빠른 전체 텍스트 검색
- **세션 계보(lineage)**: `parent_session_id` 체인 사용 (압축으로 인한 분할)
- **소스 태깅(Source tagging)** (`cli`, `telegram`, `discord` 등): 플랫폼별 필터링용
- 일괄 실행기(Batch runner) 및 강화학습(RL) 궤적은 여기에 저장되지 않음 (별도 시스템)

## SQLite 스키마

### Sessions 테이블

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
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique
    ON sessions(title) WHERE title IS NOT NULL;
```

### Messages 테이블

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
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
```

참고:
- `tool_calls`는 JSON 문자열(직렬화된 도구 호출 객체 목록)로 저장됩니다.
- `reasoning_details`, `codex_reasoning_items`, `codex_message_items`는 JSON 문자열로 저장됩니다.
- `reasoning`은 이를 노출하는 제공자에 대한 원시(raw) 추론 텍스트를 저장합니다.
- 타임스탬프는 Unix 에포크 부동 소수점(float) (`time.time()`)입니다.

### FTS5 전체 텍스트 검색

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=id
);
```

FTS5 테이블은 `messages` 테이블의 INSERT, UPDATE, DELETE 시 실행되는 3개의 트리거를 통해 동기화 상태를 유지합니다:

```sql
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
```

## 스키마 버전 및 마이그레이션

현재 스키마 버전: **11**

`schema_version` 테이블은 단일 정수를 저장합니다. 단순한 열 추가는 `_reconcile_columns()`(라이브 열을 `SCHEMA_SQL`과 비교(diff)하여 누락된 열을 `ADD` 함)를 통해 선언적으로 처리됩니다. 버전 제어 체인은 선언적으로 표현할 수 없는 데이터 마이그레이션 및 인덱스/FTS 변경에만 예약되어 있습니다:

| 버전 | 변경 사항 |
|---------|--------|
| 1 | 초기 스키마 (sessions, messages, FTS5) |
| 2 | messages에 `finish_reason` 열 추가 |
| 3 | sessions에 `title` 열 추가 |
| 4 | `title`에 고유 인덱스 추가 (NULL 허용, NULL이 아닌 값은 고유해야 함) |
| 5 | 청구 관련 열 추가: `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`, `billing_provider`, `billing_base_url`, `billing_mode`, `estimated_cost_usd`, `actual_cost_usd`, `cost_status`, `cost_source`, `pricing_version` |
| 6 | messages에 추론 관련 열 추가: `reasoning`, `reasoning_details`, `codex_reasoning_items` |
| 7 | messages에 `reasoning_content` 열 추가 |
| 8 | sessions에 `api_call_count` 열 추가 |
| 9 | Codex Responses 메시지 id/phase 리플레이를 위해 messages에 `codex_message_items` 열 추가 |
| 10 | `messages_fts_trigram` 가상 테이블 추가 (CJK / 부분 문자열 검색용 trigram 토크나이저) 및 기존 행 백필(backfill) |
| 11 | `messages_fts` 및 `messages_fts_trigram`을 재색인하여 `tool_name` + `tool_calls`를 포함하고 외부 콘텐츠에서 인라인 모드로 전환; 기존 트리거 삭제 및 모든 메시지 행 백필 |

선언적 열 추가는 이미 열이 존재하는 경우를 처리하기 위해(멱등성 보장) `try/except`로 래핑된 `ALTER TABLE ADD COLUMN`을 사용합니다. 버전 번호는 각 성공적인 마이그레이션 블록 이후에 증가합니다.

## 쓰기 경합 처리 (Write Contention Handling)

다수의 hermes 프로세스(게이트웨이 + CLI 세션 + 워크트리 에이전트)가 하나의 `state.db`를 공유합니다. `SessionDB` 클래스는 다음 방법으로 쓰기 경합을 처리합니다:

- **짧은 SQLite 타임아웃** (기본값 30초 대신 1초)
- 무작위 지연 시간(jitter)을 갖는 **애플리케이션 수준의 재시도** (20-150ms, 최대 15회)
- 트랜잭션 시작 시 락(lock) 경합을 노출하기 위한 **BEGIN IMMEDIATE** 트랜잭션
- 성공적인 쓰기가 50번 발생할 때마다 **주기적인 WAL 체크포인트** 실행 (PASSIVE 모드)

이를 통해 SQLite의 결정론적(deterministic)인 내부 백오프(backoff)로 인해 모든 경쟁 기록기가 동일한 간격으로 재시도하는 "호송 효과(convoy effect)"를 방지합니다.

```
_WRITE_MAX_RETRIES = 15
_WRITE_RETRY_MIN_S = 0.020   # 20ms
_WRITE_RETRY_MAX_S = 0.150   # 150ms
_CHECKPOINT_EVERY_N_WRITES = 50
```

## 일반적인 작업

### 초기화

```python
from hermes_state import SessionDB

db = SessionDB()                           # 기본값: ~/.hermes/state.db
db = SessionDB(db_path=Path("/tmp/test.db"))  # 사용자 지정 경로
```

### 세션 생성 및 관리

```python
# 새 세션 생성
db.create_session(
    session_id="sess_abc123",
    source="cli",
    model="anthropic/claude-sonnet-4.6",
    user_id="user_1",
    parent_session_id=None,  # 또는 계보를 위한 이전 세션 ID
)

# 세션 종료
db.end_session("sess_abc123", end_reason="user_exit")

# 세션 재개 (ended_at/end_reason 지우기)
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
# 모든 메타데이터를 포함한 원시 메시지
messages = db.get_messages("sess_abc123")

# OpenAI 대화 형식 (API 리플레이 용)
conversation = db.get_messages_as_conversation("sess_abc123")
# 반환값: [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
```

### 세션 제목

```python
# 제목 설정 (NULL이 아닌 제목들 사이에서 고유해야 함)
db.set_session_title("sess_abc123", "Fix Docker Build")

# 제목으로 해결 (계보에서 가장 최근 세션 반환)
session_id = db.resolve_session_by_title("Fix Docker Build")

# 계보에서 다음 제목 자동 생성
next_title = db.get_next_title_in_lineage("Fix Docker Build")
# 반환값: "Fix Docker Build #2"
```

## 전체 텍스트 검색

`search_messages()` 메서드는 사용자 입력에 대한 자동 살균(sanitization)과 함께 FTS5 쿼리 구문을 지원합니다.

### 기본 검색

```python
results = db.search_messages("docker deployment")
```

### FTS5 쿼리 구문

| 구문 | 예시 | 의미 |
|--------|---------|---------|
| 키워드 | `docker deployment` | 두 용어 모두 (암시적 AND) |
| 따옴표 붙은 구문 | `"exact phrase"` | 정확한 구문 일치 |
| 부울 OR | `docker OR kubernetes` | 두 용어 중 하나 |
| 부울 NOT | `python NOT java` | 용어 제외 |
| 접두사(Prefix) | `deploy*` | 접두사 일치 |

### 필터링된 검색

```python
# CLI 세션만 검색
results = db.search_messages("error", source_filter=["cli"])

# 게이트웨이 세션 제외
results = db.search_messages("bug", exclude_sources=["telegram", "discord"])

# 사용자 메시지만 검색
results = db.search_messages("help", role_filter=["user"])
```

### 검색 결과 형식

각 결과에는 다음이 포함됩니다:
- `id`, `session_id`, `role`, `timestamp`
- `snippet` — `>>>match<<<` 마커가 포함된 FTS5 생성 스니펫
- `context` — 일치 항목 전후의 1개 메시지 (내용은 200자로 잘림)
- `source`, `model`, `session_started` — 부모 세션 정보

`_sanitize_fts5_query()` 메서드는 엣지 케이스(edge cases)를 처리합니다:
- 짝이 맞지 않는 따옴표 및 특수 문자 제거
- 하이픈으로 연결된 용어를 따옴표로 묶음 (`chat-send` → `"chat-send"`)
- 매달려 있는 부울 연산자 제거 (`hello AND` → `hello`)

## 세션 계보 (Session Lineage)

세션은 `parent_session_id`를 통해 체인을 형성할 수 있습니다. 이는 게이트웨이에서 컨텍스트 압축이 세션 분할을 트리거할 때 발생합니다.

### 쿼리: 세션 계보 찾기

```sql
-- 세션의 모든 조상 찾기
WITH RECURSIVE lineage AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN lineage l ON s.id = l.parent_session_id
)
SELECT id, title, started_at, parent_session_id FROM lineage;

-- 세션의 모든 후손 찾기
WITH RECURSIVE descendants AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN descendants d ON s.parent_session_id = d.id
)
SELECT id, title, started_at FROM descendants;
```

### 쿼리: 미리보기가 포함된 최근 세션

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

### 쿼리: 토큰 사용량 통계

```sql
-- 모델별 총 토큰
SELECT model,
       COUNT(*) as session_count,
       SUM(input_tokens) as total_input,
       SUM(output_tokens) as total_output,
       SUM(estimated_cost_usd) as total_cost
FROM sessions
WHERE model IS NOT NULL
GROUP BY model
ORDER BY total_cost DESC;

-- 가장 높은 토큰 사용량을 기록한 세션
SELECT id, title, model, input_tokens + output_tokens AS total_tokens,
       estimated_cost_usd
FROM sessions
ORDER BY total_tokens DESC
LIMIT 10;
```

## 내보내기 및 정리 (Export and Cleanup)

```python
# 메시지와 함께 단일 세션 내보내기
data = db.export_session("sess_abc123")

# 모든 세션(메시지 포함)을 딕셔너리 목록으로 내보내기
all_data = db.export_all(source="cli")

# 오래된 세션 삭제 (종료된 세션만 해당)
deleted_count = db.prune_sessions(older_than_days=90)
deleted_count = db.prune_sessions(older_than_days=30, source="telegram")

# 세션 기록은 유지하되 메시지 지우기
db.clear_messages("sess_abc123")

# 세션과 모든 메시지 삭제
db.delete_session("sess_abc123")
```

## 데이터베이스 위치

기본 경로: `~/.hermes/state.db`

이 경로는 기본적으로 `~/.hermes/`로 결정되거나 `HERMES_HOME` 환경 변수의 값으로 결정되는 `hermes_constants.get_hermes_home()`에서 파생됩니다.

데이터베이스 파일, WAL 파일(`state.db-wal`) 및 공유 메모리 파일(`state.db-shm`)은 모두 동일한 디렉토리에 생성됩니다.
