---
sidebar_position: 3
title: "에이전트 루프 내부 동작"
description: "AIAgent 실행, API 모드, 도구, 콜백 및 폴백 동작에 대한 상세 안내"
---

# 에이전트 루프 내부 동작

핵심 오케스트레이션 엔진은 `run_agent.py`의 `AIAgent` 클래스입니다. 프롬프트 조립부터 도구 디스패치, 프로바이더 장애 조치까지 모든 작업을 처리하는 대형 파일입니다.

## 핵심 책임

`AIAgent`의 책임은 다음과 같습니다.

- `prompt_builder.py`를 통해 유효한 시스템 프롬프트와 도구 스키마 조립
- 올바른 프로바이더/API 모드 선택(chat_completions, codex_responses, anthropic_messages)
- 취소를 지원하는 중단 가능한 모델 호출 수행
- 도구 호출 실행(순차 실행 또는 스레드 풀을 통한 동시 실행)
- OpenAI 메시지 형식으로 대화 기록 유지
- 압축, 재시도 및 폴백 모델 전환 처리
- 부모 에이전트와 자식 에이전트 전반의 반복 예산 추적
- 컨텍스트가 손실되기 전에 영구 메모리 플러시

## 두 가지 진입점

```python
# Simple interface — returns final response string
response = agent.chat("Fix the bug in main.py")

# Full interface — returns dict with messages, metadata, usage stats
result = agent.run_conversation(
    user_message="Fix the bug in main.py",
    system_message=None,           # auto-built if omitted
    conversation_history=None,      # auto-loaded from session if omitted
    task_id="task_abc123"
)
```

`chat()`은 `run_conversation()`을 감싸는 얇은 래퍼로, 결과 딕셔너리에서 `final_response` 필드를 추출합니다.

## API 모드

Hermes는 프로바이더 선택, 명시적 인자 및 base URL 휴리스틱을 바탕으로 결정되는 세 가지 API 실행 모드를 지원합니다.

| API 모드 | 사용 대상 | 클라이언트 유형 |
|----------|----------|-------------|
| `chat_completions` | OpenAI 호환 엔드포인트(OpenRouter, 사용자 지정 엔드포인트, 대부분의 프로바이더) | `openai.OpenAI` |
| `codex_responses` | OpenAI Codex / Responses API | Responses 형식을 사용하는 `openai.OpenAI` |
| `anthropic_messages` | 네이티브 Anthropic Messages API | 어댑터를 통한 `anthropic.Anthropic` |

모드는 메시지 형식, 도구 호출 구조, 응답 파싱 방식, 캐싱/스트리밍 동작을 결정합니다. 세 모드는 모두 API 호출 전후에 동일한 내부 메시지 형식(OpenAI 스타일 `role`/`content`/`tool_calls` 딕셔너리)으로 수렴합니다.

**모드 결정 순서:**
1. 명시적인 `api_mode` 생성자 인자(최우선)
2. 프로바이더별 감지(예: `anthropic` 프로바이더 → `anthropic_messages`)
3. Base URL 휴리스틱(예: `api.anthropic.com` → `anthropic_messages`)
4. 기본값: `chat_completions`

## 턴 수명 주기

에이전트 루프의 각 반복은 다음 순서로 진행됩니다.

```text
run_conversation()
  1. Generate task_id if not provided
  2. Append user message to conversation history
  3. Build or reuse cached system prompt (prompt_builder.py)
  4. Check if preflight compression is needed (>50% context)
  5. Build API messages from conversation history
     - chat_completions: OpenAI format as-is
     - codex_responses: convert to Responses API input items
     - anthropic_messages: convert via anthropic_adapter.py
  6. Inject ephemeral prompt layers (budget warnings, context pressure)
  7. Apply prompt caching markers if on Anthropic
  8. Make interruptible API call (_interruptible_api_call)
  9. Parse response:
     - If tool_calls: execute them, append results, loop back to step 5
     - If text response: persist session, flush memory if needed, return
```

### 메시지 형식

모든 메시지는 내부적으로 OpenAI 호환 형식을 사용합니다.

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

추론 콘텐츠(확장 사고를 지원하는 모델에서 제공)는 `assistant_msg["reasoning"]`에 저장되며 `reasoning_callback`을 통해 선택적으로 표시됩니다.

### 메시지 교대 규칙

에이전트 루프는 메시지 역할의 엄격한 교대를 강제합니다.

- 시스템 메시지 다음: `User → Assistant → User → Assistant → ...`
- 도구 호출 중: `Assistant (with tool_calls) → Tool → Tool → ... → Assistant`
- **assistant 메시지가 연속으로 두 개 올 수 없음**
- **user 메시지가 연속으로 두 개 올 수 없음**
- 연속 항목을 가질 수 있는 역할은 `tool`뿐임(병렬 도구 결과)

프로바이더는 이 시퀀스를 검증하며 형식이 잘못된 기록을 거부합니다.

## 중단 가능한 API 호출

API 요청은 `_interruptible_api_call()`로 감싸며, 이 함수는 실제 HTTP 호출을 백그라운드 스레드에서 실행하면서 인터럽트 이벤트를 모니터링합니다.

```text
┌────────────────────────────────────────────────────┐
│  Main thread                  API thread           │
│                                                    │
│   wait on:                     HTTP POST           │
│    - response ready     ───▶   to provider         │
│    - interrupt event                               │
│    - timeout                                       │
└────────────────────────────────────────────────────┘
```

인터럽트가 발생하면(사용자가 새 메시지를 보내거나, `/stop` 명령 또는 시그널이 발생할 때):
- API 스레드는 중단된 것으로 처리됩니다(응답은 폐기됨).
- 에이전트는 새 입력을 처리하거나 정상적으로 종료할 수 있습니다.
- 부분 응답은 대화 기록에 삽입되지 않습니다.

## 도구 실행

### 순차 실행과 동시 실행

모델이 도구 호출을 반환하면 다음과 같이 처리합니다.

- **단일 도구 호출** → 메인 스레드에서 직접 실행
- **여러 도구 호출** → `ThreadPoolExecutor`를 통해 동시에 실행
  - 예외: 대화형으로 표시된 도구(예: `clarify`)는 순차 실행을 강제함
  - 완료 순서와 관계없이 결과는 원래 도구 호출 순서로 다시 삽입됨

### 실행 흐름

```text
for each tool_call in response.tool_calls:
    1. Resolve handler from tools/registry.py
    2. Fire pre_tool_call plugin hook
    3. Check if dangerous command (tools/approval.py)
       - If dangerous: invoke approval_callback, wait for user
    4. Execute handler with args + task_id
    5. Fire post_tool_call plugin hook
    6. Append {"role": "tool", "content": result} to history
```

### 에이전트 수준 도구

일부 도구는 `handle_function_call()`에 도달하기 전에 `run_agent.py`가 가로챕니다.

| 도구 | 가로채는 이유 |
|------|----------------|
| `todo` | 에이전트 로컬 작업 상태를 읽고 씀 |
| `memory` | 글자 수 제한에 따라 영구 메모리 파일에 씀 |
| `session_search` | 에이전트의 세션 DB를 통해 세션 기록을 조회 |
| `delegate_task` | 격리된 컨텍스트로 하위 에이전트를 생성 |

이러한 도구는 에이전트 상태를 직접 수정하며 레지스트리를 거치지 않고 합성된 도구 결과를 반환합니다.

## 콜백 표면

`AIAgent`는 CLI, 게이트웨이 및 ACP 통합에서 실시간 진행 상황을 활성화하는 플랫폼별 콜백을 지원합니다.

| 콜백 | 호출 시점 | 사용처 |
|-----------|-------------|-----------|
| `tool_progress_callback` | 각 도구 실행 전/후 | CLI 스피너, 게이트웨이 진행 메시지 |
| `thinking_callback` | 모델이 사고를 시작/종료할 때 | CLI "thinking..." 표시기 |
| `reasoning_callback` | 모델이 추론 콘텐츠를 반환할 때 | CLI 추론 표시, 게이트웨이 추론 블록 |
| `clarify_callback` | `clarify` 도구가 호출될 때 | CLI 입력 프롬프트, 게이트웨이 대화형 메시지 |
| `step_callback` | 각 에이전트 턴이 완료된 후 | 게이트웨이 단계 추적, ACP 진행 상황 |
| `stream_delta_callback` | 스트리밍 토큰마다(활성화된 경우) | CLI 스트리밍 표시 |
| `tool_gen_callback` | 스트림에서 도구 호출을 파싱할 때 | 스피너의 CLI 도구 미리 보기 |
| `status_callback` | 상태가 변경될 때(사고, 실행 등) | ACP 상태 업데이트 |

## 예산 및 폴백 동작

### 반복 예산

에이전트는 `IterationBudget`을 통해 반복을 추적합니다.

- 기본값: 500회 반복(`agent.max_turns`로 구성 가능)
- 각 에이전트는 자체 예산을 가집니다. 하위 에이전트는 `delegation.max_iterations`(기본값 50)로 제한된 독립 예산을 가지므로 부모와 하위 에이전트의 전체 반복 횟수는 부모의 제한을 초과할 수 있습니다.
- 100%에 도달하면 에이전트가 중지되고 완료한 작업의 요약을 반환합니다.

### 폴백 모델

기본 모델이 실패하면(429 속도 제한, 5xx 서버 오류, 401/403 인증 오류) 다음과 같이 처리합니다.

1. 설정에서 `fallback_providers` 목록 확인
2. 순서대로 각 폴백 시도
3. 성공하면 새 프로바이더로 대화를 계속함
4. 401/403이면 장애 조치 전에 자격 증명 새로 고침 시도

폴백 시스템은 보조 작업도 독립적으로 다룹니다 — 비전, 압축 및 웹 추출에는 각각 `auxiliary.*` 설정 섹션에서 구성할 수 있는 별도의 폴백 체인이 있습니다.

## 압축 및 영속성

### 압축이 시작되는 시점

- **사전 점검**(API 호출 전): 대화가 모델 컨텍스트 창의 50%를 초과할 때
- **게이트웨이 자동 압축:** 대화가 85%를 초과할 때(더 공격적이며 턴 사이에 실행)

### 압축 중 발생하는 일

1. 먼저 메모리를 디스크로 플러시하여 데이터 손실을 방지합니다.
2. 대화 중간의 턴을 간결한 요약으로 만듭니다.
3. 마지막 N개 메시지는 그대로 보존합니다(`compression.protect_last_n`, 기본값: 20).
4. 도구 호출/결과 메시지 쌍은 함께 유지합니다(절대 분리하지 않음).
5. 새 세션 계보 ID를 생성합니다(압축은 하위 세션을 만듭니다).

### 세션 영속성

각 턴 후:
- 메시지가 세션 저장소에 저장됩니다(`hermes_state.py`를 통한 SQLite).
- 메모리 변경 사항이 `MEMORY.md` / `USER.md`로 플러시됩니다.
- 나중에 `/resume` 또는 `hermes chat --resume`을 통해 세션을 재개할 수 있습니다.

## 주요 소스 파일

| 파일 | 목적 |
|---------|---------|
| `run_agent.py` | AIAgent 클래스 — 전체 에이전트 루프 |
| `agent/prompt_builder.py` | 메모리, 스킬, 컨텍스트 파일, 성격에서 시스템 프롬프트 조립 |
| `agent/context_engine.py` | ContextEngine ABC — 플러그형 컨텍스트 관리 |
| `agent/context_compressor.py` | 기본 엔진 — 손실 요약 알고리즘 |
| `agent/prompt_caching.py` | Anthropic 프롬프트 캐싱 마커 및 캐시 지표 |
| `agent/auxiliary_client.py` | 보조 작업(비전, 요약)을 위한 보조 LLM 클라이언트 |
| `model_tools.py` | 도구 스키마 수집, `handle_function_call()` 디스패치 |

## 관련 문서

- [프로바이더 런타임 결정](./provider-runtime.md)
- [프롬프트 조립](./prompt-assembly.md)
- [컨텍스트 압축 및 프롬프트 캐싱](./context-compression-and-caching.md)
- [도구 런타임](./tools-runtime.md)
- [아키텍처 개요](./architecture.md)
