---
sidebar_position: 14
title: "API 서버"
description: "hermes-agent를 모든 프론트엔드를 위한 OpenAI 호환 API로 노출합니다"
---

# API 서버 (API Server)

API 서버는 hermes-agent를 OpenAI 호환 HTTP 엔드포인트로 노출합니다. OpenAI 형식을 사용하는 모든 프론트엔드(Open WebUI, LobeChat, LibreChat, NextChat, ChatBox 등 수백 개)는 hermes-agent에 연결하여 이를 백엔드로 사용할 수 있습니다.

에이전트는 전체 도구 세트(터미널, 파일 작업, 웹 검색, 메모리, 스킬)를 활용하여 요청을 처리하고 최종 응답을 반환합니다. 스트리밍 시, 도구 진행 상태 표시기가 인라인으로 나타나 프론트엔드에서 에이전트가 어떤 작업을 하고 있는지 보여줄 수 있습니다.

:::tip 하나의 백엔드로 모델과 도구를 모두 커버합니다
API 서버를 유용하게 사용하려면 Hermes 자체에 구성된 제공자(provider)와 도구 백엔드가 필요합니다. [Nous Portal](/user-guide/features/tool-gateway) 구독은 300개 이상의 모델과 함께 Tool Gateway를 통한 웹/이미지/TTS/브라우저 도구를 모두 제공합니다. API 서버를 시작하기 전에 `hermes setup --portal`을 한 번 실행하면 Open WebUI나 LobeChat과 같은 프론트엔드에서 도구가 완벽하게 갖춰진 백엔드를 얻을 수 있습니다.
:::

## 빠른 시작 (Quick Start)

### 1. API 서버 활성화

`~/.hermes/.env`에 다음을 추가하세요:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# 선택 사항: 브라우저가 직접 Hermes를 호출해야 하는 경우에만 설정
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. 게이트웨이 시작

```bash
hermes gateway
```

다음 메시지가 표시됩니다:

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. 프론트엔드 연결

OpenAI 호환 클라이언트를 `http://localhost:8642/v1`에 연결하세요:

```bash
# curl로 테스트하기
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

또는 Open WebUI, LobeChat 등의 프론트엔드에 연결하세요. 단계별 지침은 [Open WebUI 연동 가이드](/user-guide/messaging/open-webui)를 참조하세요.

## 엔드포인트 (Endpoints)

### POST /v1/chat/completions

표준 OpenAI Chat Completions 형식입니다. 무상태(Stateless) 방식이며 — 각 요청의 `messages` 배열에 전체 대화가 포함됩니다.

**요청 (Request):**
```json
{
  "model": "hermes-agent",
  "messages": [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "Write a fibonacci function"}
  ],
  "stream": false
}
```

**응답 (Response):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "hermes-agent",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Here's a fibonacci function..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
}
```

**인라인 이미지 입력:** user 메시지는 `content`를 `text` 및 `image_url` 부분으로 구성된 배열 형태로 보낼 수 있습니다. 원격 `http(s)` URL 및 `data:image/...` URL 모두 지원됩니다:

```json
{
  "model": "hermes-agent",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}}
      ]
    }
  ]
}
```

업로드된 파일(`file` / `input_file` / `file_id`)과 이미지가 아닌 `data:` URL은 `400 unsupported_content_type`을 반환합니다.

**스트리밍** (`"stream": true`): 토큰 단위의 응답 청크를 포함하는 Server-Sent Events (SSE)를 반환합니다. **Chat Completions**의 경우, 스트림은 도구 시작 UX를 위한 Hermes의 사용자 정의 `hermes.tool.progress` 이벤트와 함께 표준 `chat.completion.chunk` 이벤트를 사용합니다. **Responses**의 경우, 스트림은 `response.created`, `response.output_text.delta`, `response.output_item.added`, `response.output_item.done`, `response.completed`와 같은 OpenAI Responses 이벤트 유형을 사용합니다.

**스트림에서의 도구 진행 상태 (Tool progress in streams)**:
- **Chat Completions**: Hermes는 영구적인 assistant 텍스트를 오염시키지 않으면서 도구 시작 시각화를 제공하기 위해 `event: hermes.tool.progress`를 방출합니다.
- **Responses**: Hermes는 SSE 스트림 동안 스펙의 네이티브(spec-native) 출력 항목인 `function_call` 및 `function_call_output`을 방출하므로 클라이언트는 구조화된 도구 UI를 실시간으로 렌더링할 수 있습니다.

### POST /v1/responses

OpenAI Responses API 형식입니다. `previous_response_id`를 통해 서버 측 대화 상태 유지를 지원합니다 — 서버가 (도구 호출 및 결과를 포함한) 전체 대화 기록을 저장하므로 클라이언트가 관리하지 않아도 다중 턴 컨텍스트가 보존됩니다.

**요청 (Request):**
```json
{
  "model": "hermes-agent",
  "input": "What files are in my project?",
  "instructions": "You are a helpful coding assistant.",
  "store": true
}
```

**응답 (Response):**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "hermes-agent",
  "output": [
    {"type": "function_call", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

**인라인 이미지 입력:** `input[].content`에는 `input_text` 및 `input_image` 부분이 포함될 수 있습니다. 원격 URL 및 `data:image/...` URL 모두 지원됩니다:

```json
{
  "model": "hermes-agent",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this screenshot."},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K..."}
      ]
    }
  ]
}
```

업로드된 파일(`input_file` / `file_id`)과 이미지가 아닌 `data:` URL은 `400 unsupported_content_type`을 반환합니다.

#### previous_response_id를 사용한 다중 턴 대화

턴에 걸쳐 모든 컨텍스트(도구 호출 포함)를 유지하려면 응답을 연결하세요:

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

서버는 저장된 응답 체인에서 전체 대화를 재구성합니다 — 모든 이전 도구 호출 및 결과가 보존됩니다. 연결된 요청들은 동일한 세션을 공유하므로, 다중 턴 대화는 대시보드와 세션 기록에 단일 항목으로 표시됩니다.

#### 이름이 지정된 대화 (Named conversations)

응답 ID를 추적하는 대신 `conversation` 매개변수를 사용하세요:

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

서버는 해당 대화의 최신 응답에 자동으로 연결합니다. 게이트웨이 세션의 `/title` 명령어와 유사하게 동작합니다.

### GET /v1/responses/\{id\}

ID로 이전에 저장된 응답을 검색합니다.

### DELETE /v1/responses/\{id\}

저장된 응답을 삭제합니다.

### GET /v1/models

에이전트를 사용 가능한 모델로 나열합니다. 광고되는 모델 이름은 기본적으로 [프로필(profile)](/user-guide/profiles) 이름으로 지정됩니다 (또는 기본 프로필의 경우 `hermes-agent`). 대부분의 프론트엔드에서 모델을 찾기 위해 필수적인 기능입니다.

### GET /v1/capabilities

외부 UI, 오케스트레이터 및 플러그인 브리지를 위해 API 서버의 안정적인 표면 영역을 설명하는 기계 판독 가능한(machine-readable) 데이터를 반환합니다.

```json
{
  "object": "hermes.api_server.capabilities",
  "platform": "hermes-agent",
  "model": "hermes-agent",
  "auth": {"type": "bearer", "required": true},
  "features": {
    "chat_completions": true,
    "responses_api": true,
    "run_submission": true,
    "run_status": true,
    "run_events_sse": true,
    "run_stop": true
  }
}
```

대시보드, 브라우저 UI 또는 제어 평면(control plane)을 연동할 때 이 엔드포인트를 사용하여 실행 중인 Hermes 버전이 내부 Python 코드에 의존하지 않고도 실행(runs), 스트리밍, 취소 및 세션 연속성을 지원하는지 확인할 수 있습니다.

### GET /health

상태 확인. `{"status": "ok"}`를 반환합니다. `/v1/` 접두사를 요구하는 OpenAI 호환 클라이언트를 위해 **GET /v1/health**로도 사용할 수 있습니다.

### GET /health/detailed

활성 세션, 실행 중인 에이전트 및 리소스 사용량도 함께 보고하는 확장된 상태 확인 기능입니다. 모니터링/관찰 도구에 유용합니다.

## Runs API (스트리밍에 친화적인 대안)

서버는 `/v1/chat/completions` 및 `/v1/responses` 외에도, 클라이언트가 스트리밍을 직접 관리하는 대신 진행 이벤트에 가입(subscribe)하려는 장기 세션을 위한 **runs** API를 노출합니다.

### POST /v1/runs

새로운 에이전트 실행(run)을 생성합니다. 진행 이벤트에 가입하는 데 사용할 수 있는 `run_id`를 반환합니다.

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

실행은 간단한 `input` 문자열과 선택적 `session_id`, `instructions`, `conversation_history` 또는 `previous_response_id`를 허용합니다. `session_id`가 제공되면 Hermes는 실행 상태에 이를 표출하므로 외부 UI가 자체 대화 ID와 실행을 연관 지을 수 있습니다.

### GET /v1/runs/\{run_id\}

현재 실행 상태를 폴링합니다. SSE 연결을 계속 열어두지 않고 상태 정보가 필요한 대시보드나 탐색 후 다시 연결하는 UI에 유용합니다.

```json
{
  "object": "hermes.run",
  "run_id": "run_abc123",
  "status": "completed",
  "session_id": "space-session",
  "model": "hermes-agent",
  "output": "Done.",
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

종료 상태(`completed`, `failed` 또는 `cancelled`)에 도달한 후에도 폴링 및 UI 조정을 위해 일정 시간 동안 상태가 유지됩니다.

### GET /v1/runs/\{run_id\}/events

실행의 도구 호출 진행 상황, 토큰 델타(delta) 및 수명 주기(lifecycle) 이벤트의 Server-Sent Events 스트림입니다. 상태 손실 없이 연결/분리하려는 대시보드 및 리치 클라이언트(thick clients)를 위해 설계되었습니다.

### POST /v1/runs/\{run_id\}/stop

실행 중인 에이전트 턴을 중단합니다. 이 엔드포인트는 `{"status": "stopping"}`을 즉시 반환하며, 그동안 Hermes는 활성 에이전트에게 안전하게 중단 가능한 다음 지점에서 작업을 멈추도록 요청합니다.

## Jobs API (예약된 백그라운드 작업)

서버는 원격 클라이언트에서 백그라운드 예약 에이전트 실행을 관리하기 위해 가벼운 작업(jobs) CRUD 표면을 노출합니다. 모든 엔드포인트는 동일한 베어러(bearer) 인증 게이트에 의해 보호됩니다.

### GET /api/jobs

예약된 모든 작업을 나열합니다.

### POST /api/jobs

예약된 새 작업을 생성합니다. 본문은 프롬프트, 일정, 스킬, 제공자 오버라이드, 전달 대상과 같은 `hermes cron`과 동일한 형태를 허용합니다.

### GET /api/jobs/\{job_id\}

단일 작업의 정의와 마지막 실행 상태를 가져옵니다.

### PATCH /api/jobs/\{job_id\}

기존 작업의 필드(프롬프트, 일정 등)를 업데이트합니다. 부분 업데이트(partial updates)가 병합됩니다.

### DELETE /api/jobs/\{job_id\}

작업을 제거합니다. 실행 중인 작업도 함께 취소합니다.

### POST /api/jobs/\{job_id\}/pause

작업을 삭제하지 않고 일시 중지합니다. 다음 예약된 실행 타임스탬프는 재개될 때까지 연기됩니다.

### POST /api/jobs/\{job_id\}/resume

이전에 일시 중지된 작업을 재개합니다.

### POST /api/jobs/\{job_id\}/run

일정에 관계없이 즉시 작업을 실행하도록 트리거합니다.

## Sessions API (REST를 통한 세션 제어)

외부 UI는 대시보드를 구축하지 않고도 REST를 통해 Hermes 세션을 관리할 수 있습니다. 모든 엔드포인트는 `API_SERVER_KEY`에 의해 보호되며 `/api/sessions/*` 아래에 위치합니다.

| 메서드 | 경로 | 설명 |
|--------|------|-------------|
| `GET` | `/api/sessions` | 세션 나열 (페이지 지정 — `limit`, `offset`, `source`, `include_children`) |
| `POST` | `/api/sessions` | 빈 세션 생성 |
| `GET` | `/api/sessions/{id}` | 세션 메타데이터 읽기 |
| `PATCH` | `/api/sessions/{id}` | 제목 또는 `end_reason` 업데이트 |
| `DELETE` | `/api/sessions/{id}` | 세션 삭제 |
| `GET` | `/api/sessions/{id}/messages` | 세션의 메시지 기록 |
| `POST` | `/api/sessions/{id}/fork` | `SessionDB` 계보를 통해 세션 분기 (CLI `/branch` 문법과 일치) |
| `POST` | `/api/sessions/{id}/chat` | 하나의 동기적 에이전트 턴 실행 |
| `POST` | `/api/sessions/{id}/chat/stream` | 단일 턴에 대한 SSE 래퍼 — `assistant.delta`, `tool.started`, `tool.completed`, `run.completed` 이벤트를 방출합니다. |

`/v1/capabilities`는 `session_*` 기능 플래그 및 `endpoints.session_*` 항목을 통해 전체 기능을 알리므로 외부 UI가 지원 여부를 감지하고 안전하게 폴백할 수 있습니다. 인라인 이미지는 `chat` 및 `chat/stream` 페이로드(멀티모달 인식 경로)에서 지원됩니다.

```bash
# 세션을 분기하고 하나의 턴을 실행
curl -X POST http://localhost:8642/api/sessions/$ID/fork \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"title": "explore alt path"}'

# SSE를 통해 턴을 스트리밍
curl -N -X POST http://localhost:8642/api/sessions/$ID/chat/stream \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"input": "what files changed in the last hour?"}'
```

## 스킬 및 도구 세트 검색 (Skills and toolsets discovery)

`GET /v1/skills` 및 `GET /v1/toolsets`를 사용하면 외부 클라이언트가 모델에 묻지 않고도 REST를 통해 에이전트의 기능을 확정적으로(deterministically) 나열할 수 있습니다. 둘 다 읽기 전용이며 `API_SERVER_KEY`로 보호됩니다.

```bash
curl http://localhost:8642/v1/skills \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "github-pr-workflow", "description": "...", "category": "..."}, ...]

curl http://localhost:8642/v1/toolsets \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "core", "label": "...", "description": "...", "enabled": true,
#     "configured": true, "tools": ["read_file", "write_file", ...]}, ...]
```

`/v1/skills`는 스킬 허브가 내부적으로 사용하는 것과 동일한 메타데이터를 반환합니다. `/v1/toolsets`는 `api_server` 플랫폼에 대해 해석된(resolved) 도구 세트와, 각 도구 세트가 전개되는 구체적인 `tools` 목록을 함께 반환합니다. 둘 다 `/v1/capabilities`의 `endpoints.*` 아래에 광고됩니다.

## 장기 메모리 범위 설정 (`X-Hermes-Session-Key`)

Open WebUI와 같은 다중 사용자 프론트엔드는 (새로운 채팅 시 바뀌는) 트랜스크립트(transcript) 단위의 `X-Hermes-Session-Id`와는 **독립적인** 장기 메모리(Honcho 등)를 위한 안정적인 채널별 식별자가 필요합니다. `/v1/chat/completions`, `/v1/responses`, 또는 `/v1/runs` 호출 시 `X-Hermes-Session-Key` 헤더를 전달하면, Hermes가 이를 `AIAgent(gateway_session_key=...)`로 넘겨주고 Honcho 메모리 제공자는 이 값을 사용해 안정적인 메모리 범위를 형성합니다.

```http
POST /v1/chat/completions HTTP/1.1
Authorization: Bearer ***
X-Hermes-Session-Id: transcript-alpha
X-Hermes-Session-Key: agent:main:webui:dm:user-42
```

규칙: 최대 256자, 제어 문자(`\r`, `\n`, `\x00`)는 거부되며 값은 응답(JSON + SSE)에 다시 반환(echo)됩니다. `/v1/capabilities`는 `"session_key_header": "X-Hermes-Session-Key"`를 통해 이 기능의 지원 여부를 알립니다. 이 키가 없으면 Honcho의 `per-session` 전략은 `session_id`마다 다른 범위를 생성하며 — 이는 이전의 Hermes 동작과 정확히 일치합니다.

## 시스템 프롬프트 처리 (System Prompt Handling)

프론트엔드가 `system` 메시지(Chat Completions) 또는 `instructions` 필드(Responses API)를 보낼 때, hermes-agent는 이를 핵심 시스템 프롬프트의 **가장 위 계층에 추가(layer it on top)** 합니다. 에이전트는 모든 도구, 메모리, 스킬을 유지하며 — 프론트엔드의 시스템 프롬프트가 추가적인 지시사항을 덧붙이는 형태가 됩니다.

이는 기능 손실 없이 각 프론트엔드별로 동작 방식을 커스터마이징할 수 있음을 의미합니다:
- Open WebUI 시스템 프롬프트 예: "당신은 파이썬 전문가입니다. 항상 타입 힌트(type hints)를 포함시키세요."
- 이렇게 해도 에이전트는 여전히 터미널, 파일 도구, 웹 검색, 메모리 등의 기능을 모두 갖추고 있습니다.

## 인증 (Authentication)

`Authorization` 헤더를 통한 Bearer 토큰 인증을 사용합니다:

```
Authorization: Bearer ***
```

`API_SERVER_KEY` 환경 변수를 통해 키를 설정합니다. 브라우저가 Hermes를 직접 호출해야 하는 경우, `API_SERVER_CORS_ORIGINS`에 명시적인 허용 목록도 설정해야 합니다.

:::warning 보안
API 서버는 **터미널 명령을 포함한** hermes-agent 도구 세트에 대한 전체 접근 권한을 부여합니다. `127.0.0.1` 기본 루프백 바인딩을 포함한 **모든 배포 환경에서 `API_SERVER_KEY` 설정은 필수**입니다. 브라우저 호출을 명시적으로 허용할 때는 `API_SERVER_CORS_ORIGINS` 범위를 좁게 유지하여 브라우저의 접근 권한을 통제하세요.
:::

## 구성 (Configuration)

### 환경 변수 (Environment Variables)

| 변수 | 기본값 | 설명 |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | API 서버 활성화 |
| `API_SERVER_PORT` | `8642` | HTTP 서버 포트 |
| `API_SERVER_HOST` | `127.0.0.1` | 바인딩 주소 (기본값은 로컬 호스트만 허용) |
| `API_SERVER_KEY` | _(필수)_ | 인증용 Bearer 토큰 |
| `API_SERVER_CORS_ORIGINS` | _(없음)_ | 허용할 브라우저 출처(origin) (쉼표로 구분) |
| `API_SERVER_MODEL_NAME` | _(프로필 이름)_ | `/v1/models`에 노출되는 모델 이름. 기본값은 프로필 이름이며 기본 프로필은 `hermes-agent`입니다. |

### config.yaml

```yaml
# 아직 지원되지 않습니다 — 환경 변수를 사용하세요.
# config.yaml 지원은 향후 릴리스에서 추가될 예정입니다.
```

## 보안 헤더 (Security Headers)

모든 응답에는 보안 헤더가 포함됩니다:
- `X-Content-Type-Options: nosniff` — MIME 타입 스니핑 방지
- `Referrer-Policy: no-referrer` — 리퍼러 유출 방지

## CORS

API 서버는 기본적으로 브라우저 CORS를 **활성화하지 않습니다**.

브라우저에서 직접 접근하려면 명시적 허용 목록을 설정해야 합니다:

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

CORS가 활성화된 경우:
- **Preflight 응답**에 `Access-Control-Max-Age: 600`(10분 캐시)가 포함됩니다.
- **SSE 스트리밍 응답**에 CORS 헤더가 포함되므로 브라우저 EventSource 클라이언트가 올바르게 작동합니다.
- **`Idempotency-Key`** 헤더가 허용된 요청 헤더로 취급됩니다 — 클라이언트는 중복 방지를 위해 이를 전송할 수 있습니다(응답은 키를 기준으로 5분간 캐시됨).

Open WebUI와 같이 문서화된 대부분의 프론트엔드는 서버 대 서버로 연결하므로 CORS가 전혀 필요하지 않습니다.

## 호환되는 프론트엔드 (Compatible Frontends)

OpenAI API 형식을 지원하는 모든 프론트엔드와 작동합니다. 테스트 및 문서화된 연동:

| 프론트엔드 | 별점(Stars) | 연결 설정 |
|----------|-------|------------|
| [Open WebUI](/user-guide/messaging/open-webui) | 126k | 전체 가이드 이용 가능 |
| LobeChat | 73k | 사용자 지정 제공자 엔드포인트 |
| LibreChat | 34k | librechat.yaml 내 사용자 지정 엔드포인트 |
| AnythingLLM | 56k | 일반 OpenAI 제공자 |
| NextChat | 87k | `BASE_URL` 환경 변수 |
| ChatBox | 39k | API Host 설정 |
| Jan | 26k | 원격 모델(Remote model) 구성 |
| HF Chat-UI | 8k | `OPENAI_BASE_URL` |
| big-AGI | 7k | 사용자 지정 엔드포인트 |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | 직접 HTTP 요청 전송 |

## 프로필을 활용한 다중 사용자 설정 (Multi-User Setup with Profiles)

여러 사용자에게 (독립적인 구성, 메모리, 스킬을 가진) 격리된 개별 Hermes 인스턴스를 제공하려면, [프로필(profiles)](/user-guide/profiles) 기능을 사용하세요:

```bash
# 사용자별 프로필 생성
hermes profile create alice
hermes profile create bob

# 각 프로필의 API 서버를 다른 포트로 구성. API_SERVER_* 변수는
# (config.yaml 키가 아닌) 환경 변수이므로 각 프로필의 .env에 작성:
cat >> ~/.hermes/profiles/alice/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
API_SERVER_KEY=alice-secret
EOF

cat >> ~/.hermes/profiles/bob/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8644
API_SERVER_KEY=bob-secret
EOF

# 각 프로필의 게이트웨이 시작
hermes -p alice gateway &
hermes -p bob gateway &
```

각 프로필의 API 서버는 자동으로 프로필 이름을 모델 ID로 알립니다:

- `http://localhost:8643/v1/models` → 모델명 `alice`
- `http://localhost:8644/v1/models` → 모델명 `bob`

Open WebUI에서는 각각을 별도의 연결로 추가하세요. 모델 선택 드롭다운에 `alice`와 `bob`이 각각 별개의 모델로 표시되며, 둘 다 완전히 격리된 Hermes 인스턴스에 의해 구동됩니다. 자세한 내용은 [Open WebUI 연동 가이드](/user-guide/messaging/open-webui#multi-user-setup-with-profiles)를 참조하세요.

## 제한 사항 (Limitations)

- **응답 보관** — (`previous_response_id` 활용을 위한) 응답 보관 기능은 SQLite에 유지되므로 게이트웨이 재시작 후에도 남아있습니다. 최대 100개의 응답이 저장됩니다 (LRU 정책에 따라 초과분 삭제).
- **파일 업로드 불가** — `/v1/chat/completions` 및 `/v1/responses` 모두 인라인 이미지를 지원하지만, 업로드된 파일(`file`, `input_file`, `file_id`)이나 이미지가 아닌 문서 입력을 API를 통해 전송하는 것은 지원하지 않습니다.
- **모델 필드는 장식입니다** — 요청의 `model` 필드 값은 받아들여지긴 하지만, 실제 사용되는 LLM 모델은 서버 측의 config.yaml 설정을 따릅니다.

## 프록시 모드 (Proxy Mode)

API 서버는 **게이트웨이 프록시 모드**의 백엔드 역할도 수행합니다. 다른 Hermes 게이트웨이 인스턴스가 이 API 서버를 가리키도록 `GATEWAY_PROXY_URL`을 설정하면, 해당 게이트웨이는 자체 에이전트를 실행하는 대신 모든 메시지를 API 서버로 전달합니다. 이를 통해, 예를 들어 Matrix의 E2EE(종단 간 암호화)를 처리하는 Docker 컨테이너가 호스트 측 에이전트와 통신하는 식의 분리 배포(split deployment)가 가능해집니다.

전체 설정 가이드는 [Matrix 프록시 모드](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)를 참조하세요.
