---
sidebar_position: 3
title: "Agent Loop Internals"
description: "AIAgent 실행, API 모드, 도구, 콜백 및 폴백 동작에 대한 상세한 가이드"
---

# 에이전트 루프 내부

핵심 오케스트레이션 엔진은 `run_agent.py`의 `AIAgent` 클래스입니다. 프롬프트 조합부터 도구 디스패치, 프로바이더 장애 조치(failover)에 이르기까지 모든 것을 처리하는 약 4,400줄의 거대한 파일입니다.

## 핵심 책임

`AIAgent`는 다음을 담당합니다:

- `prompt_builder.py`를 통한 효과적인 시스템 프롬프트 및 도구 스키마 조립
- 올바른 프로바이더/API 모드 선택 (chat_completions, codex_responses, anthropic_messages)
- 취소(cancellation)를 지원하는 중단 가능한(interruptible) 모델 호출 수행
- 도구 호출 실행 (순차적 또는 스레드 풀을 통한 동시 실행)
- OpenAI 메시지 형식으로 대화 기록 유지
- 압축, 재시도, 폴백 모델 전환 처리
- 부모 및 자식 에이전트 간의 반복(iteration) 예산 추적
- 컨텍스트 손실 전 영구 메모리 플러시

## 두 가지 진입점

```python
# 간단한 인터페이스 — 최종 응답 문자열 반환
response = agent.chat("main.py의 버그를 수정해줘")

# 전체 인터페이스 — 메시지, 메타데이터, 사용량 통계가 포함된 딕셔너리 반환
result = agent.run_conversation(
    user_message="main.py의 버그를 수정해줘",
    system_message=None,           # 생략 시 자동 생성
    conversation_history=None,      # 생략 시 세션에서 자동 로드
    task_id="task_abc123"
)
```

`chat()`은 결과 딕셔너리에서 `final_response` 필드를 추출하는 `run_conversation()`의 얇은 래퍼(wrapper)입니다.

## API 모드

Hermes는 세 가지 API 실행 모드를 지원하며, 이는 프로바이더 선택, 명시적 인자 및 기본 URL 휴리스틱(heuristics)에서 결정됩니다:

| API 모드 | 용도 | 클라이언트 유형 |
|----------|----------|-------------|
| `chat_completions` | OpenAI 호환 엔드포인트 (OpenRouter, 커스텀, 대부분의 프로바이더) | `openai.OpenAI` |
| `codex_responses` | OpenAI Codex / Responses API | Responses 형식의 `openai.OpenAI` |
| `anthropic_messages` | 네이티브 Anthropic Messages API | 어댑터를 통한 `anthropic.Anthropic` |

모드는 메시지 형식 지정 방식, 도구 호출 구성 방식, 응답 구문 분석 방식, 캐싱/스트리밍 작동 방식을 결정합니다. API 호출 전후에 이 세 가지 모드 모두 동일한 내부 메시지 형식(OpenAI 스타일의 `role`/`content`/`tool_calls` 딕셔너리)으로 수렴합니다.

**모드 해결 순서:**
1. 명시적 `api_mode` 생성자 인자 (가장 높은 우선순위)
2. 프로바이더별 감지 (예: `anthropic` 프로바이더 → `anthropic_messages`)
3. 기본 URL 휴리스틱 (예: `api.anthropic.com` → `anthropic_messages`)
4. 기본값: `chat_completions`

## 턴 라이프사이클 (Turn Lifecycle)

에이전트 루프의 각 반복은 다음 순서를 따릅니다:

```text
run_conversation()
  1. 제공되지 않은 경우 task_id 생성
  2. 대화 기록에 사용자 메시지 추가
  3. 캐시된 시스템 프롬프트 구축 또는 재사용 (prompt_builder.py)
  4. 사전 압축(preflight compression)이 필요한지 확인 (컨텍스트의 50% 초과)
  5. 대화 기록에서 API 메시지 구축
     - chat_completions: OpenAI 형식 그대로 사용
     - codex_responses: Responses API 입력 항목으로 변환
     - anthropic_messages: anthropic_adapter.py를 통해 변환
  6. 임시 프롬프트 레이어 주입 (예산 경고, 컨텍스트 압력)
  7. Anthropic인 경우 프롬프트 캐싱 마커 적용
  8. 중단 가능한 API 호출 생성 (_interruptible_api_call)
  9. 응답 분석:
     - tool_calls인 경우: 이를 실행하고, 결과를 추가한 뒤, 5단계로 루프백
     - 텍스트 응답인 경우: 세션 유지, 필요한 경우 메모리 플러시, 반환
```

### 메시지 형식

모든 메시지는 내부적으로 OpenAI 호환 형식을 사용합니다:

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

추론 콘텐츠(확장된 사고를 지원하는 모델의 경우)는 `assistant_msg["reasoning"]`에 저장되며 선택적으로 `reasoning_callback`을 통해 표시됩니다.

### 메시지 교대 규칙 (Message Alternation Rules)

에이전트 루프는 엄격한 메시지 역할 교대 규칙을 적용합니다:

- 시스템 메시지 이후: `User → Assistant → User → Assistant → ...`
- 도구 호출 중: `Assistant (tool_calls 포함) → Tool → Tool → ... → Assistant`
- **절대** 두 개의 Assistant 메시지가 연속될 수 없음
- **절대** 두 개의 User 메시지가 연속될 수 없음
- **오직** `tool` 역할만이 연속된 항목(병렬 도구 결과)을 가질 수 있음

프로바이더는 이러한 시퀀스를 검증하고 잘못된 형식의 기록을 거부합니다.

## 중단 가능한 API 호출

API 요청은 중단 이벤트(interrupt event)를 모니터링하면서 백그라운드 스레드에서 실제 HTTP 호출을 실행하는 `_interruptible_api_call()`로 감싸집니다:

```text
┌────────────────────────────────────────────────────┐
│  메인 스레드                  API 스레드           │
│                                                    │
│   대기 조건:                   HTTP POST           │
│    - 응답 준비됨        ───▶   프로바이더로 전송   │
│    - 중단 이벤트                                   │
│    - 시간 초과(timeout)                            │
└────────────────────────────────────────────────────┘
```

중단된 경우(사용자가 새 메시지를 보내거나, `/stop` 명령어 또는 신호):
- API 스레드는 버려집니다 (응답 폐기).
- 에이전트는 새 입력을 처리하거나 깔끔하게 종료할 수 있습니다.
- 대화 기록에 부분적인 응답이 주입되지 않습니다.

## 도구 실행

### 순차적 vs 동시 실행

모델이 도구 호출을 반환할 때:

- **단일 도구 호출** → 메인 스레드에서 직접 실행됨
- **다중 도구 호출** → `ThreadPoolExecutor`를 통해 동시에 실행됨
  - 예외: 대화형(interactive)으로 표시된 도구(예: `clarify`)는 강제로 순차 실행됨
  - 결과는 완료 순서에 관계없이 원래 도구 호출 순서대로 다시 삽입됨

### 실행 흐름

```text
response.tool_calls의 각 tool_call에 대해:
    1. tools/registry.py에서 핸들러 해석(resolve)
    2. pre_tool_call 플러그인 훅 실행
    3. 위험한 명령어인지 확인 (tools/approval.py)
       - 위험한 경우: approval_callback을 호출하여 사용자 대기
    4. args + task_id와 함께 핸들러 실행
    5. post_tool_call 플러그인 훅 실행
    6. 기록에 {"role": "tool", "content": result} 추가
```

### 에이전트 수준 도구

일부 도구는 `handle_function_call()`에 도달하기 *전에* `run_agent.py`에서 가로챕니다:

| 도구 | 가로채는 이유 |
|------|--------------------|
| `todo` | 에이전트 로컬의 작업(task) 상태를 읽거나 씁니다 |
| `memory` | 글자 수 제한이 있는 영구 메모리 파일에 씁니다 |
| `session_search` | 에이전트의 세션 DB를 통해 세션 기록을 조회합니다 |
| `delegate_task` | 격리된 컨텍스트를 가진 하위 에이전트를 생성합니다 |

이 도구들은 레지스트리를 거치지 않고 에이전트 상태를 직접 수정하며 합성된(synthetic) 도구 결과를 반환합니다.

## 콜백 표면 (Callback Surfaces)

`AIAgent`는 CLI, 게이트웨이 및 ACP 통합에서 실시간 진행 상황을 가능하게 하는 플랫폼별 콜백을 지원합니다:

| 콜백 | 실행 시점 | 사용처 |
|----------|-----------|---------|
| `tool_progress_callback` | 각 도구 실행 전/후 | CLI 스피너, 게이트웨이 진행 메시지 |
| `thinking_callback` | 모델이 생각을 시작/중지할 때 | CLI "thinking..." 표시기 |
| `reasoning_callback` | 모델이 추론 콘텐츠를 반환할 때 | CLI 추론 표시, 게이트웨이 추론 블록 |
| `clarify_callback` | `clarify` 도구가 호출될 때 | CLI 입력 프롬프트, 게이트웨이 대화형 메시지 |
| `step_callback` | 각 완료된 에이전트 턴 후 | 게이트웨이 단계 추적, ACP 진행 상황 |
| `stream_delta_callback` | 각 스트리밍 토큰 (활성화 시) | CLI 스트리밍 표시 |
| `tool_gen_callback` | 스트림에서 도구 호출이 분석될 때 | 스피너의 CLI 도구 미리보기 |
| `status_callback` | 상태 변경 시 (thinking, executing 등) | ACP 상태 업데이트 |

## 예산 및 폴백 동작

### 반복 예산 (Iteration Budget)

에이전트는 `IterationBudget`을 통해 반복을 추적합니다:

- 기본값: 90회 반복 (`agent.max_turns`를 통해 구성 가능)
- 각 에이전트는 자체 예산을 갖습니다. 하위 에이전트는 `delegation.max_iterations`(기본값 50)으로 제한된 독립적인 예산을 갖습니다. 부모와 하위 에이전트 전체의 총 반복 횟수는 부모의 한도를 초과할 수 있습니다.
- 100% 도달 시, 에이전트는 중지하고 완료된 작업의 요약을 반환합니다.

### 폴백 모델

기본 모델이 실패할 경우 (429 속도 제한, 5xx 서버 오류, 401/403 인증 오류):

1. 구성에서 `fallback_providers` 목록 확인
2. 순서대로 각 폴백 시도
3. 성공 시 새 프로바이더로 대화 계속
4. 401/403의 경우, 장애 조치 전 자격 증명 갱신 시도

폴백 시스템은 독립적인 보조 작업도 다룹니다. 비전, 압축 및 웹 추출은 각각 `auxiliary.*` 구성 섹션을 통해 구성 가능한 자체 폴백 체인을 가지고 있습니다.

## 압축 및 영속성 (Compression and Persistence)

### 압축 트리거 시점

- **사전 확인 (API 호출 전)**: 대화가 모델 컨텍스트 윈도우의 50%를 초과할 경우
- **게이트웨이 자동 압축**: 대화가 85%를 초과할 경우 (더 공격적, 턴 사이에 실행됨)

### 압축 중에 발생하는 일

1. 데이터 손실을 방지하기 위해 먼저 메모리를 디스크로 플러시합니다.
2. 중간 대화 턴을 요약하여 간결한 요약본으로 만듭니다.
3. 마지막 N개의 메시지는 온전하게 유지됩니다 (`compression.protect_last_n`, 기본값: 20).
4. 도구 호출/결과 메시지 쌍은 함께 유지됩니다 (절대 분할되지 않음).
5. 새로운 세션 계보 ID가 생성됩니다 (압축은 "자식" 세션을 생성합니다).

### 세션 영속성

각 턴 이후:
- 메시지가 세션 저장소에 저장됩니다 (`hermes_state.py`를 통한 SQLite).
- 메모리 변경 사항이 `MEMORY.md` / `USER.md`로 플러시됩니다.
- `/resume` 또는 `hermes chat --resume`을 통해 나중에 세션을 재개할 수 있습니다.

## 주요 소스 파일

| 파일 | 목적 |
|------|---------|
| `run_agent.py` | AIAgent 클래스 — 전체 에이전트 루프 |
| `agent/prompt_builder.py` | 메모리, 스킬, 컨텍스트 파일, 성격에서 시스템 프롬프트 조립 |
| `agent/context_engine.py` | ContextEngine ABC — 플러그형 컨텍스트 관리 |
| `agent/context_compressor.py` | 기본 엔진 — 손실(lossy) 요약 알고리즘 |
| `agent/prompt_caching.py` | Anthropic 프롬프트 캐싱 마커 및 캐시 메트릭 |
| `agent/auxiliary_client.py` | 보조 작업을 위한 보조 LLM 클라이언트 (비전, 요약) |
| `model_tools.py` | 도구 스키마 수집, `handle_function_call()` 디스패치 |

## 관련 문서

- [프로바이더 런타임 해석 (Provider Runtime Resolution)](./provider-runtime.md)
- [프롬프트 조합 (Prompt Assembly)](./prompt-assembly.md)
- [컨텍스트 압축 및 캐싱 (Context Compression & Prompt Caching)](./context-compression-and-caching.md)
- [도구 런타임 (Tools Runtime)](./tools-runtime.md)
- [아키텍처 개요 (Architecture Overview)](./architecture.md)
