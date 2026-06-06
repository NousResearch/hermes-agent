---
sidebar_position: 9
title: "Context Engine Plugins"
description: "내장된 ContextCompressor를 대체하는 컨텍스트 엔진 플러그인을 빌드하는 방법"
---

# 컨텍스트 엔진 플러그인 빌드하기

컨텍스트 엔진 플러그인은 대화 컨텍스트를 관리하기 위한 대안적인 전략으로 내장된 `ContextCompressor`를 대체합니다. 예를 들어, 손실(lossy) 요약 대신 지식 DAG(Directed Acyclic Graph)를 구축하는 무손실 컨텍스트 관리(Lossless Context Management, LCM) 엔진이 있습니다.

## 작동 방식

에이전트의 컨텍스트 관리는 `ContextEngine` 추상 기본 클래스(ABC) (`agent/context_engine.py`)를 기반으로 구축됩니다. 내장된 `ContextCompressor`가 기본 구현입니다. 플러그인 엔진도 동일한 인터페이스를 구현해야 합니다.

한 번에 **단 하나**의 컨텍스트 엔진만 활성화될 수 있습니다. 선택은 구성 기반입니다:

```yaml
# config.yaml
context:
  engine: "compressor"    # 기본 내장 엔진
  engine: "lcm"           # 이름이 "lcm"인 플러그인 엔진 활성화
```

플러그인 엔진은 **절대 자동 활성화되지 않습니다**. 사용자가 명시적으로 `context.engine`을 플러그인 이름으로 설정해야 합니다.

## 디렉토리 구조

각 컨텍스트 엔진은 `plugins/context_engine/<name>/`에 위치합니다:

```
plugins/context_engine/lcm/
├── __init__.py      # ContextEngine 하위 클래스 내보내기
├── plugin.yaml      # 메타데이터 (이름, 설명, 버전)
└── ...              # 엔진에 필요한 기타 모든 모듈
```

## ContextEngine ABC

엔진은 다음 **필수** 메서드를 구현해야 합니다:

```python
from agent.context_engine import ContextEngine

class LCMEngine(ContextEngine):

    @property
    def name(self) -> str:
        """짧은 식별자, 예: 'lcm'. config.yaml 값과 일치해야 합니다."""
        return "lcm"

    def update_from_response(self, usage: dict) -> None:
        """모든 LLM 호출 후에 usage 딕셔너리와 함께 호출됩니다.

        응답에서 self.last_prompt_tokens, self.last_completion_tokens,
        self.last_total_tokens를 업데이트합니다.
        """

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """이번 턴에 압축(compaction)이 실행되어야 하면 True를 반환합니다."""

    def compress(self, messages: list, current_tokens: int = None,
                 focus_topic: str = None) -> list:
        """메시지 목록을 압축하고 새로운(아마도 더 짧은) 목록을 반환합니다.

        반환된 목록은 유효한 OpenAI 형식의 메시지 시퀀스여야 합니다.

        ``focus_topic``은 수동 ``/compress <focus>``에서 오는 선택적인 주제 문자열입니다; 
        안내식 압축을 지원하는 엔진은 이와 관련된 정보 보존을 우선시해야 하고, 
        그렇지 않은 엔진은 무시할 수 있습니다.
        """
```

### 엔진이 유지해야 하는 클래스 속성

에이전트는 디스플레이 및 로깅을 위해 이를 직접 읽습니다:

```python
last_prompt_tokens: int = 0
last_completion_tokens: int = 0
last_total_tokens: int = 0
threshold_tokens: int = 0        # 압축이 트리거되는 시점
context_length: int = 0          # 모델의 전체 컨텍스트 윈도우
compression_count: int = 0       # compress()가 실행된 횟수
```

### 선택적 메서드

이들은 ABC에서 합리적인 기본값을 가집니다. 필요에 따라 재정의(override)하세요:

| 메서드 | 기본값 | 재정의 시기 |
|--------|---------|--------------|
| `on_session_start(session_id, **kwargs)` | No-op | 유지된 상태(DAG, DB)를 로드해야 할 때 |
| `on_session_end(session_id, messages)` | No-op | 상태를 플러시(flush)하고 연결을 닫아야 할 때 |
| `on_session_reset()` | 토큰 카운터 재설정 | 지워야 할 세션별 상태가 있을 때 |
| `update_model(model, context_length, ...)` | context_length + threshold 업데이트 | 모델 전환 시 예산을 다시 계산해야 할 때 |
| `get_tool_schemas()` | `[]` 반환 | 엔진이 에이전트 호출 가능한 도구를 제공할 때 (예: `lcm_grep`) |
| `handle_tool_call(name, args, **kwargs)` | 오류 JSON 반환 | 도구 핸들러를 구현할 때 |
| `should_compress_preflight(messages)` | `False` 반환 | API 호출 전 저렴한 비용으로 추정이 가능할 때 |
| `get_status()` | 표준 토큰/임계값 딕셔너리 | 노출할 사용자 지정 메트릭이 있을 때 |

## 엔진 도구

컨텍스트 엔진은 에이전트가 직접 호출할 수 있는 도구를 노출할 수 있습니다. `get_tool_schemas()`에서 스키마를 반환하고 `handle_tool_call()`에서 호출을 처리합니다:

```python
def get_tool_schemas(self):
    return [{
        "name": "lcm_grep",
        "description": "컨텍스트 지식 그래프 검색",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"}
            },
            "required": ["query"],
        },
    }]

def handle_tool_call(self, name, args, **kwargs):
    if name == "lcm_grep":
        results = self._search_dag(args["query"])
        return json.dumps({"results": results})
    return json.dumps({"error": f"알 수 없는 도구: {name}"})
```

엔진 도구는 시작 시 에이전트의 도구 목록에 주입되며 자동으로 디스패치됩니다 — 레지스트리 등록이 필요하지 않습니다.

## 등록

### 디렉토리를 통한 등록 (권장)

엔진을 `plugins/context_engine/<name>/`에 배치합니다. `__init__.py`는 `ContextEngine` 하위 클래스를 내보내야(export) 합니다. 검색 시스템이 자동으로 찾아 인스턴스화합니다.

### 일반 플러그인 시스템을 통한 등록

일반 플러그인도 컨텍스트 엔진을 등록할 수 있습니다:

```python
def register(ctx):
    engine = LCMEngine(context_length=200000)
    ctx.register_context_engine(engine)
```

하나의 엔진만 등록할 수 있습니다. 두 번째 플러그인이 등록을 시도하면 경고와 함께 거부됩니다.

## 라이프사이클

```
1. 엔진 인스턴스화 (플러그인 로드 또는 디렉토리 검색)
2. on_session_start() — 대화 시작
3. update_from_response() — 각 API 호출 후
4. should_compress() — 각 턴마다 확인
5. compress() — should_compress()가 True를 반환할 때 호출됨
6. on_session_end() — 세션 경계 (CLI 종료, /reset, 게이트웨이 만료)
```

`on_session_reset()`은 `/new` 또는 `/reset`에서 호출되어 완전한 종료 없이 세션별 상태를 지웁니다.

## 구성 (Configuration)

사용자는 `hermes plugins` → Provider Plugins → Context Engine을 통해 엔진을 선택하거나 `config.yaml`을 편집합니다:

```yaml
context:
  engine: "lcm"   # 엔진의 name 속성과 일치해야 합니다
```

`compression` 구성 블록(`compression.threshold`, `compression.protect_last_n` 등)은 내장된 `ContextCompressor`에만 적용됩니다. 엔진은 필요한 경우 자체 구성 형식을 정의하고 초기화 중에 `config.yaml`에서 읽어와야 합니다.

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

전체 ABC 계약 테스트 스위트는 `tests/agent/test_context_engine.py`를 참조하세요.

## 참고

- [컨텍스트 압축 및 캐싱](/developer-guide/context-compression-and-caching) — 내장 압축기의 작동 방식
- [메모리 프로바이더 플러그인](/developer-guide/memory-provider-plugin) — 메모리를 위한 유사한 단일 선택 플러그인 시스템
- [플러그인](/user-guide/features/plugins) — 일반 플러그인 시스템 개요
