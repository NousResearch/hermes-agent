---
sidebar_position: 9
title: "컨텍스트 엔진 플러그인"
description: "기본 제공 ContextCompressor를 대체하는 컨텍스트 엔진 플러그인을 만드는 방법"
---

# 컨텍스트 엔진 플러그인 빌드

컨텍스트 엔진 플러그인은 대화 컨텍스트를 관리하기 위한 대체 전략으로 기본 제공 `ContextCompressor`를 대체합니다. 예를 들어 손실이 있는 요약 대신 지식 DAG를 구축하는 Lossless Context Management(LCM) 엔진이 있습니다.

## 작동 방식

에이전트의 컨텍스트 관리는 `ContextEngine` ABC(`agent/context_engine.py`)를 기반으로 합니다. 기본 제공 `ContextCompressor`가 기본 구현입니다. 플러그인 엔진은 동일한 인터페이스를 구현해야 합니다.

한 번에 활성화할 수 있는 컨텍스트 엔진은 **하나뿐**입니다. 선택은 설정에 따라 결정됩니다.

```yaml
# config.yaml
context:
  engine: "compressor"    # default built-in
  engine: "lcm"           # activates a plugin engine named "lcm"
```

플러그인 엔진은 **자동으로 활성화되지 않습니다** — 사용자가 명시적으로 `context.engine`을 플러그인 이름으로 설정해야 합니다.

## 디렉터리 구조

각 컨텍스트 엔진은 `plugins/context_engine/<name>/`에 있습니다.

```
plugins/context_engine/lcm/
├── __init__.py      # exports the ContextEngine subclass
├── plugin.yaml      # metadata (name, description, version)
└── ...              # any other modules your engine needs
```

## ContextEngine ABC

엔진은 다음 **필수** 메서드를 구현해야 합니다.

```python
from agent.context_engine import ContextEngine

class LCMEngine(ContextEngine):

    @property
    def name(self) -> str:
        """Short identifier, e.g. 'lcm'. Must match config.yaml value."""
        return "lcm"

    def update_from_response(self, usage: dict) -> None:
        """Called after every LLM call with the usage dict.

        Update self.last_prompt_tokens, self.last_completion_tokens,
        self.last_total_tokens from the response.
        """

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""

    def compress(self, messages: list, current_tokens: int = None,
                 focus_topic: str = None) -> list:
        """Compact the message list and return a new (possibly shorter) list.

        The returned list must be a valid OpenAI-format message sequence.

        ``focus_topic`` is an optional topic string from manual
        ``/compress <focus>``; engines that support guided compression should
        prioritise preserving information related to it, others may ignore it.
        """
```

### 엔진이 유지해야 하는 클래스 속성

에이전트는 표시 및 로깅을 위해 다음 속성을 직접 읽습니다.

```python
last_prompt_tokens: int = 0
last_completion_tokens: int = 0
last_total_tokens: int = 0
threshold_tokens: int = 0        # when compression triggers
context_length: int = 0          # model's full context window
compression_count: int = 0       # how many times compress() has run
```

### 선택적 메서드

ABC에는 합리적인 기본값이 있습니다. 필요에 따라 재정의하세요.

| 메서드 | 기본값 | 다음 경우 재정의 |
|--------|--------|----------------|
| `on_session_start(session_id, **kwargs)` | No-op | 유지된 상태(DAG, DB)를 로드해야 할 때 |
| `on_session_end(session_id, messages)` | No-op | 상태를 플러시하거나 연결을 닫아야 할 때 |
| `on_session_reset()` | 토큰 카운터 재설정 | 세션별 상태를 지워야 할 때 |
| `update_model(model, context_length, ...)` | context_length + threshold 업데이트 | 모델 전환 시 예산을 다시 계산해야 할 때 |
| `get_tool_schemas()` | `[]` 반환 | 엔진이 에이전트가 호출할 수 있는 도구(예: `lcm_grep`)를 제공할 때 |
| `handle_tool_call(name, args, **kwargs)` | 오류 JSON 반환 | 도구 핸들러를 구현할 때 |
| `should_compress_preflight(messages)` | `False` 반환 | 저렴한 사전 API 호출 추정을 수행할 수 있을 때 |
| `get_status()` | 표준 토큰/임계값 딕셔너리 | 노출할 사용자 지정 메트릭이 있을 때 |
| `select_context(request_messages, *, conversation_messages, incoming_message, budget_tokens)` | `None` 반환(동작 없음) | **이 요청에** 들어갈 컨텍스트를 선택/라우팅할 때(아래 참조) |
| `on_turn_complete(messages, usage=None, **kwargs)` | No-op | 완료된 턴을 수집/인덱싱/관찰할 때(아래 참조) |

## 턴별 컨텍스트 선택 및 관찰

`compress()`는 "컨텍스트가 너무 김 → 더 짧게 만들기"에 답합니다. 두 개의 선택적이고 기본적으로 동작하지 않는 훅은 서로 독립적인 선택/관찰 축을 다룹니다. 따라서 엔진은 `should_compress()`를 `True`로 만들고 `compress()`를 턴별 콜백처럼 남용할 필요가 없습니다.

```python
def select_context(self, request_messages, *, conversation_messages=None,
                   incoming_message=None, budget_tokens=0):
    """Choose/replace the context for THIS request, before dispatch.

    Return a new message list to use for this one provider call (retrieval,
    topic routing, role/branch switching), or None to leave it unchanged.
    Request-only: the persisted conversation history is never mutated.
    """

def on_turn_complete(self, messages, usage=None, **kwargs):
    """Observe a finished turn after the assistant/tool loop completes.

    Receives a shallow copy of the finalized transcript plus the turn's
    canonical usage dict (or None if no provider response was reached), so the
    engine can ingest/index/summarize for the next select_context(). The return
    value is ignored.
    """
```

계약:

- **기본적으로 동작하지 않으며 장애가 발생해도 계속 진행합니다.** 두 기본 메서드는 모두 `return None`입니다. 훅이 없거나, 예외가 발생하거나, 반환값이 유효하지 않으면 요청을 변경하지 않은 채 진행됩니다 — 따라서 엔진 오류가 엔진을 설치하지 않은 것보다 나쁜 결과를 만들지 않습니다. 호스트는 상속된 ABC 기본 구현인지도 확인하여 완전히 건너뛰므로, 구현하지 않는 엔진(기본 제공 compressor 포함)은 요청마다 작업을 수행하지 않습니다.
- **`select_context()`는 요청 전용입니다.** 반환된 목록은 단일 제공자 호출의 메시지를 대체하며 저장된 기록은 절대 기록되지 않습니다. `None`, `[]`, 목록이 아닌 값 또는 딕셔너리가 아닌 항목을 포함한 목록을 반환하면 수정되지 않은 요청으로 처리됩니다.
- **순서 / 캐시 안정성.** 훅은 프롬프트 캐시 제어와 모든 요청 정제기 **전에** 실행되므로 (a) 대체된 값도 모든 요청과 동일한 검증을 통과하고, (b) 동작하지 않는 기본값은 요청을 바이트 단위로 동일하게 유지합니다 — 구현하지 않는 엔진에서는 프롬프트 캐시 동작이 변하지 않습니다. 목록을 대체하는 엔진은 자체 캐시 접두사만 변경합니다. 제공자 요청마다 평가됩니다(재시도 시 다시 실행).
- **`on_turn_complete()`**는 턴 이후 관찰 전용입니다. `messages`를 읽기 전용으로 취급하세요. **적용 범위는 최선의 노력입니다.** 표준 턴 마무리 지점에서 호출됩니다. 루프의 일부 비정상적인 조기 반환 경로(예: 콘텐츠 정책 차단 또는 제공자의 최종 실패)는 마무리를 거치지 않고 저장 및 반환하므로 현재 이 훅을 발생시키지 않습니다 — 모든 조기 종료에 대한 보장된 콜백이 아니라 완료된 턴에 대한 최선의 노력 기반 관찰로 취급하세요. 모든 종료 경로를 하나의 마무리 지점으로 통합하는 것은 별도의 후속 작업입니다.

### 이 훅을 사용할 때 — 사용하지 말아야 할 때

- 엔진이 **요청별 컨텍스트를 대체해야 할 때만** `select_context()`를 구현하세요 — 검색 증강 선택, 주제/브랜치 라우팅, 역할 전환 등이 해당합니다. 어떤 메시지가 요청에 들어갈지 바꿀 수 있는 유일한 동작입니다. `pre_llm_call` 플러그인 훅은 문서화된 설계상 주입 전용입니다(프롬프트 캐시 접두사를 보존하기 위해 사용자 메시지에 추가만 하고 목록을 다시 작성하지 않음). 대체가 필요하지 않다면 구현하지 마세요.
- **플러그인에 턴 이후 관찰/수집**(인덱싱, 메모리 동기화, 분석)만 필요하다면 컨텍스트 엔진 대신 **메모리 제공자**(`sync_turn()` — [메모리 제공자 플러그인](./memory-provider-plugin.md) 참조)를 구현하세요. 컨텍스트 엔진은 세션의 압축 정책을 맡고, 메모리 제공자는 아무것도 소유하지 않고 턴을 관찰합니다. `on_turn_complete()`는 이미 `select_context()`가 필요한 엔진이 라우팅한 턴에서 학습할 수 있도록 만든 관찰 대응 기능이지, 범용 턴 콜백이 아닙니다.
- **실제 `select_context()`가 프롬프트 캐시에 미치는 영향.** 동작하지 않는 상태가 아닌 선택은 선택이 바뀌는 턴의 프롬프트 캐시 접두사를 자연스럽게 변경합니다 — 해당 요청의 접두사는 제공자의 캐시된 접두사와 더 이상 일치하지 않으므로 해당 턴에는 캐시를 읽는 대신 다시 작성합니다. 엔진은 아무것도 변하지 않았을 때 **안정적인 선택을 반환**해야 합니다(동일한 객체 또는 동일한 목록). 라우팅 결정이 실제로 달라질 때만 컨텍스트를 재구성하세요. 매 턴 목록을 섞는 선택은 매 턴 캐시 재사용을 조용히 포기하게 만듭니다.

## 엔진 도구

컨텍스트 엔진은 에이전트가 직접 호출하는 도구를 노출할 수 있습니다. `get_tool_schemas()`에서 스키마를 반환하고 `handle_tool_call()`에서 호출을 처리하세요.

```python
def get_tool_schemas(self):
    return [{
        "name": "lcm_grep",
        "description": "Search the context knowledge graph",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    }]

def handle_tool_call(self, name, args, **kwargs):
    if name == "lcm_grep":
        results = self._search_dag(args["query"])
        return json.dumps({"results": results})
    return json.dumps({"error": f"Unknown tool: {name}"})
```

엔진 도구는 시작 시 에이전트의 도구 목록에 주입되고 자동으로 디스패치됩니다 — 레지스트리에 등록할 필요가 없습니다.

## 등록

### 디렉터리를 통한 등록(권장)

엔진을 `plugins/context_engine/<name>/`에 배치하세요. `__init__.py`는 `ContextEngine` 하위 클래스를 내보내야 합니다. 검색 시스템이 자동으로 엔진을 찾아 인스턴스화합니다.

### 일반 플러그인 시스템을 통한 등록

일반 플러그인도 컨텍스트 엔진을 등록할 수 있습니다.

```python
def register(ctx):
    engine = LCMEngine(context_length=200000)
    ctx.register_context_engine(engine)
```

등록할 수 있는 엔진은 하나뿐입니다. 두 번째 플러그인이 등록을 시도하면 경고와 함께 거부됩니다.

## 수명 주기

```
1. Engine instantiated (plugin load or directory discovery)
2. on_session_start() — conversation begins
3. update_from_response() — after each API call
4. should_compress() — checked each turn
5. compress() — called when should_compress() returns True
6. on_session_end() — session boundary (CLI exit, /reset, gateway expiry)
```

`on_session_reset()`은 `/new` 또는 `/reset`에서 호출되어 완전히 종료하지 않고 세션별 상태를 지웁니다.

## 설정

사용자는 `hermes plugins` → Provider Plugins → Context Engine을 통해 또는 `config.yaml`을 편집하여 엔진을 선택합니다.

```yaml
context:
  engine: "lcm"   # must match your engine's name property
```

`compression` 설정 블록(`compression.threshold`, `compression.protect_last_n` 등)은 기본 제공 `ContextCompressor`에만 해당합니다. 단 하나의 명시적 예외인 `compression.model_thresholds`(모델별 임계값 재정의)는 컨텍스트 엔진 계약의 일부입니다. 호스트는 최초 `update_model()` 호출 **전에** 해석된 맵을 `engine.model_thresholds`에 할당하고, 기본 클래스의 `update_model()`은 이를 적용합니다(가장 긴 부분 문자열이 일치하는 항목을 사용하며, 엔진에 설정된 임계값으로 대체). `update_model()`을 재정의하는 엔진은 자체 압축 정책을 소유하므로 맵을 적용하거나 무시할 수 있습니다 — 동일한 해석 로직을 재사용하려면 `from agent.context_compressor import resolve_model_threshold`를 사용하세요. 그 외의 경우 엔진은 필요하다면 자체 설정 형식을 정의하고 초기화 중 `config.yaml`에서 읽어야 합니다.

## 테스트

```python
from agent.context_engine import ContextEngine

def test_engine_satisfies_abc():
    engine = YourEngine(context_length=200000)
    assert isinstance(engine, ContextEngine)
    assert engine.name == "your-name"

def test_compress_returns_valid_messages():
    engine = YourEngine(context_length=200000)
    msgs = [{"role": "user", "content": "hello"}]
    result = engine.compress(msgs)
    assert isinstance(result, list)
    assert all("role" in m for m in result)
```

전체 ABC 계약 테스트 모음은 `tests/agent/test_context_engine.py`를 참조하세요.

## 스레드 안전성

`compression.context_timeout_seconds > 0`(기본값)일 때 Hermes는 전체 압축 과정 — 엔진의 `compress()` 및 경계 콜백, 그리고 메모리 제공자의 `on_pre_compress` / `on_session_switch` 포함 — 을 호스트 측 타임아웃이 있는 풀의 데몬 스레드에서 실행합니다. 따라서 엔진은 다음을 가정해야 합니다.

- 호출은 임의의 풀 스레드에서 도착할 수 있습니다. 대화 스레드와 공유되는 스레드 친화성이나 `threading.local` 상태에 의존하지 마세요.
- 수신하는 메시지 목록은 전용 딥 스냅샷입니다. 제자리에서 수정하는 것은 허용됩니다(레거시 계약). 그러나 패스가 커밋되는 경우에만 수정 내용이 표시됩니다. 호스트 타임아웃 후에도 작업이 계속 실행 중이라면 해당 작업은 폐기됩니다 — 커밋 전에 외부/영구 상태에 절대 게시하지 마세요.
- 서로 다른 세션의 패스는 풀의 형제 스레드에서 동시에 실행될 수 있습니다. 여러 세션이 공유하는 단일 엔진/제공자 인스턴스는 스레드 안전해야 합니다.

## 함께 보기

- [컨텍스트 압축 및 캐싱](/developer-guide/context-compression-and-caching) — 기본 제공 compressor의 작동 방식
- [메모리 제공자 플러그인](/developer-guide/memory-provider-plugin) — 단일 선택 플러그인 시스템과 유사한 메모리
- [플러그인](/user-guide/features/plugins) — 일반 플러그인 시스템 개요
