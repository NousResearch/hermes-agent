---
sidebar_position: 14
title: "API 서버"
description: "hermes-agent를 모든 프런트엔드에서 사용할 수 있는 OpenAI 호환 API로 노출"
---

# API 서버

API 서버는 hermes-agent를 OpenAI 호환 HTTP 엔드포인트로 노출합니다. Open WebUI, LobeChat, LibreChat, NextChat, ChatBox 등 OpenAI 형식을 지원하는 모든 프런트엔드와 수백 개의 다른 클라이언트가 hermes-agent에 연결하여 백엔드로 사용할 수 있습니다.

에이전트는 전체 도구 세트(터미널, 파일 작업, 웹 검색, 메모리, 스킬)로 요청을 처리하고 최종 응답을 반환합니다. 스트리밍 중에는 도구 진행 표시가 인라인으로 나타나므로 프런트엔드에서 에이전트가 수행 중인 작업을 표시할 수 있습니다.

:::tip 모델과 도구를 모두 지원하는 하나의 백엔드
API 서버를 유용하게 사용하려면 Hermes 자체에 프로바이더와 도구 백엔드가 구성되어 있어야 합니다. [Nous Portal](/user-guide/features/tool-gateway) 구독을 사용하면 두 가지를 모두 처리할 수 있으며, 300개 이상의 모델과 Tool Gateway를 통한 웹/이미지/TTS/브라우저를 제공합니다. API 서버를 시작하기 전에 `hermes setup --portal`을 한 번 실행하면 Open WebUI나 LobeChat 같은 프런트엔드에서 도구가 완비된 백엔드를 사용할 수 있습니다.
:::

## 빠른 시작

### 1. API 서버 활성화

`~/.hermes/.env`에 다음을 추가합니다.

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# Optional: only if a browser must call Hermes directly
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. 게이트웨이 시작

```bash
hermes gateway
```

다음과 같은 메시지가 표시됩니다.

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. 프런트엔드 연결

OpenAI 호환 클라이언트가 `http://localhost:8642/v1`을 가리키도록 설정합니다.

```bash
# Test with curl
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

또는 Open WebUI, LobeChat 및 다른 프런트엔드에 연결합니다. 단계별 안내는 [Open WebUI 통합 가이드](/user-guide/messaging/open-webui)를 참조하세요.

## 엔드포인트

### POST /v1/chat/completions

표준 OpenAI Chat Completions 형식입니다. 상태 비저장 방식으로 동작하며, 전체 대화가 `messages` 배열을 통해 각 요청에 포함됩니다.

**요청:**
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

**응답:**
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

**인라인 이미지 입력:** 사용자 메시지는 `content`를 `text`와 `image_url` 부분의 배열로 보낼 수 있습니다. 원격 `http(s)` URL과 `data:image/...` URL을 모두 지원합니다.

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

**스트리밍**(`"stream": true`): 토큰 단위 응답 청크가 포함된 Server-Sent Events(SSE)를 반환합니다. **Chat Completions**에서는 표준 `chat.completion.chunk` 이벤트와 도구 시작 UX를 위한 Hermes의 사용자 지정 `hermes.tool.progress` 이벤트를 함께 사용합니다. **Responses**에서는 `response.created`, `response.output_text.delta`, `response.output_item.added`, `response.output_item.done`, `response.completed` 같은 OpenAI Responses 이벤트 유형을 사용합니다.

**스트림의 도구 진행 상황:**
- **Chat Completions**: Hermes는 저장된 어시스턴트 텍스트를 오염시키지 않고 도구 시작 상태를 표시하기 위해 `event: hermes.tool.progress`를 내보냅니다.
- **Responses**: Hermes는 SSE 스트림 중 사양에 맞는 `function_call` 및 `function_call_output` 출력 항목을 내보내므로 클라이언트가 구조화된 도구 UI를 실시간으로 렌더링할 수 있습니다.

### POST /v1/responses

OpenAI Responses API 형식입니다. `previous_response_id`를 통한 서버 측 대화 상태를 지원하므로, 서버가 전체 대화 기록(도구 호출과 결과 포함)을 저장합니다. 따라서 클라이언트가 이를 직접 관리하지 않아도 여러 턴의 컨텍스트가 유지됩니다.

**요청:**
```json
{
  "model": "hermes-agent",
  "input": "What files are in my project?",
  "instructions": "You are a helpful coding assistant.",
  "store": true
}
```

**응답:**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "hermes-agent",
  "output": [
    {"type": "function_call", "status": "completed", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "status": "completed", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

`output` 배열의 도구 호출은 Hermes 에이전트가 이미 서버 측에서 실행했습니다. 구조화된 도구 UI를 위해 `"status": "completed"`로 재생되며, 클라이언트가 실행해야 하는 대기 중 호출로 전달되지 않습니다.

**인라인 이미지 입력:** `input[].content`에는 `input_text`와 `input_image` 부분을 포함할 수 있습니다. 원격 URL과 `data:image/...` URL을 모두 지원합니다.

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

#### previous_response_id를 사용한 여러 턴

응답을 연결하여 도구 호출을 포함한 전체 컨텍스트를 여러 턴에 걸쳐 유지합니다.

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

서버는 저장된 응답 체인에서 전체 대화를 재구성하므로 이전 도구 호출과 결과가 모두 보존됩니다. 연결된 요청은 동일한 세션도 공유하므로 여러 턴의 대화가 대시보드와 세션 기록에 하나의 항목으로 표시됩니다.

#### 이름이 지정된 대화

응답 ID를 추적하는 대신 `conversation` 매개변수를 사용합니다.

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

서버는 해당 대화의 최신 응답에 자동으로 연결합니다. 이는 게이트웨이 세션의 `/title` 명령과 유사합니다.

### GET /v1/responses/\{id\}

ID로 이전에 저장된 응답을 조회합니다.

### DELETE /v1/responses/\{id\}

저장된 응답을 삭제합니다.

### GET /v1/models

에이전트를 사용 가능한 모델로 나열합니다. 표시되는 모델 이름은 기본적으로 [프로필](/user-guide/profiles) 이름이며, 기본 프로필에서는 `hermes-agent`입니다. 대부분의 프런트엔드에서 모델 검색을 위해 필요합니다.

`/v1/models`는 의도적으로 저비용 OpenAI 호환 표면입니다. Hermes가 라우팅할 수 있는 인증된 모든 프로바이더/모델 조합을 열거하지 않으며, 가격이나 기능 정보를 보강하지도 않습니다.

### GET /api/model/options

Hermes를 인식하는 클라이언트는 대시보드와 TUI에서 사용하는 동일한 선별된 프로바이더/모델 목록을 요청할 수 있습니다. 이 경로는 API 서버의 일반 bearer 인증을 사용하며, OpenAI 호환 `/v1/models` 응답에 포함되지 않는 프로바이더 행, 모델 기능 힌트, 가격 메타데이터를 반환합니다.

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options"
```

해당 페이로드는 대시보드 Models 페이지와 TUI의 `model.options` RPC가 사용하는 동일한 기반 데이터입니다. 인증된 프로바이더, 선별된 모델 목록, 모델별 가격, 모델 기능 힌트를 반환합니다.

사용자 지정 프로바이더에 대해서는 일반적인 열기 작업이 의도적으로 보수적으로 동작합니다. 오래되었거나 오프라인인 저장된 엔드포인트가 선택기를 차단하지 않도록 **현재 선택된** 사용자 지정 엔드포인트만 프로브합니다. 명시적 새로 고침을 수행하면 전체 프로빙으로 전환되고 프로바이더 모델 캐시가 무효화됩니다.

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options?refresh=1"
```

OpenAI 호환 클라이언트가 채팅/Responses 요청에 보낼 모델 이름만 필요하다면 `/v1/models`를 사용합니다. 인증된 UI에서 더 풍부한 Hermes 전용 선택기 메타데이터가 필요하다면 `/api/model/options`를 사용합니다.

### GET /v1/capabilities

외부 UI, 오케스트레이터, 플러그인을 위한 API 서버의 안정적인 표면을 기계가 읽을 수 있는 형태로 반환합니다.

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

대시보드, 브라우저 UI, 제어 플레인을 통합할 때 이 엔드포인트를 사용하면 비공개 Python 내부 구현에 의존하지 않고 실행 중인 Hermes 버전이 runs, 스트리밍, 취소, 세션 연속성을 지원하는지 검색할 수 있습니다.

## 요청별 모델 선택

인증된 클라이언트는 각 요청에 다음을 보내 Hermes의 기본 모델 선택을 재정의할 수 있습니다.

- `model` — 이 턴의 대상 모델 ID
- `provider` — 이 턴의 자격 증명/런타임을 확인할 Hermes 프로바이더 슬러그
- `model_options` — 요청 범위의 추론/서비스 계층 제어

동일한 요청 필드는 다음에도 허용됩니다.

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/runs`
- `POST /api/sessions/{session_id}/chat`
- `POST /api/sessions/{session_id}/chat/stream`

우선순위는 결정론적입니다.

1. 해당 세션에 이미 설정된 세션 `/model` 재정의
2. 요청의 `model`이 구성된 경로 별칭인 경우 선택되는 정적 `gateway.platforms.api_server.model_routes` 매핑
3. 일치하는 경로 별칭이 없을 때 직접 요청의 `model` / `provider`
4. 전역 게이트웨이 구성 / 환경 기본값

승리한 모델/프로바이더와 관계없이 `model_options`는 요청 범위로 유지됩니다. 요청에서 구성된 `model_routes` 별칭과 충돌하는 `provider`를 보내면 Hermes는 경로 자격 증명을 다른 프로바이더와 조용히 섞는 대신 `400`으로 요청을 거부합니다.

**OpenAI 호환 엔드포인트의 단순 `model` 값은 선택 사항입니다.** 일반적인 OpenAI 클라이언트는 모델 이름(`gpt-4o`, ...)을 하드코딩하는 경우가 많고 기존 배포는 이러한 값이 게이트웨이 기본값으로 대체되는 동작에 의존합니다. 따라서 `POST /v1/chat/completions`와 `POST /v1/responses`에서 `provider` 없이 전송된 `model` 값은 다음을 활성화하지 않는 한 무시됩니다.

```yaml
gateway:
  platforms:
    api_server:
      direct_model_requests: true
```

명시적인 `provider`를 포함한 요청과 Hermes 네이티브 `/v1/runs` 및 세션 채팅 엔드포인트는 이 플래그와 관계없이 요청된 모델을 항상 사용합니다.

예시:

```json
{
  "model": "MiniMax-M3",
  "provider": "minimax",
  "model_options": {
    "reasoning_effort": "high",
    "service_tier": "priority"
  },
  "messages": [
    {"role": "user", "content": "Summarize the repo status."}
  ]
}
```

### GET /health

상태 확인입니다. `{"status": "ok"}`를 반환합니다. `/v1/` 접두사를 요구하는 OpenAI 호환 클라이언트를 위해 **GET /v1/health**에서도 사용할 수 있습니다.

### GET /health/detailed

모니터링 및 제어 플레인을 위한 인증된 준비 상태 확인입니다. 활성 프로필의 구성, 상태 데이터베이스, 구성된 모델, 디스크 공간, 게이트웨이/플랫폼 상태, 활성 API runs, 대기 중인 프로세스 완료 작업, 활성 위임 작업의 제한된 상태를 보고합니다. 응답에는 구성 값, 자격 증명, 경로, 명령, 큐 페이로드, 원시 오류가 아닌 상태와 개수만 노출됩니다.

공개 `/health` 경로는 저비용 활성 상태 프로브로 유지되며 준비 상태 확인을 수행하지 않습니다. 준비 상태가 저하되어도 HTTP 200을 사용하며, 최상위 `status` 및 `readiness.checks` 필드를 확인해야 합니다.

## Runs API (스트리밍에 적합한 대안)

`/v1/chat/completions`와 `/v1/responses` 외에도 서버는 **runs** API를 제공합니다. 클라이언트가 직접 스트리밍을 관리하는 대신 진행 이벤트를 구독하려는 장기 세션을 위한 API입니다.

### POST /v1/runs

새 에이전트 실행을 생성합니다. 진행 이벤트를 구독하는 데 사용할 수 있는 `run_id`를 반환합니다.

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

runs는 간단한 `input` 문자열과 선택적 `session_id`, `instructions`, `conversation_history`, `previous_response_id`를 허용합니다. `session_id`가 제공되면 Hermes는 실행 상태에 이를 표시하므로 외부 UI가 자체 대화 ID와 실행을 연결할 수 있습니다.

### GET /v1/runs/\{run_id\}

현재 실행 상태를 폴링합니다. SSE 연결을 계속 열어 두지 않고 상태를 확인해야 하는 대시보드나, 탐색 후 다시 연결하는 UI에 유용합니다.

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

터미널 상태(`completed`, `failed`, `cancelled`)가 된 후에도 폴링과 UI 조정을 위해 상태가 잠시 유지됩니다.

### GET /v1/runs/\{run_id\}/events

실행의 도구 호출 진행 상황, 토큰 델타, 수명 주기 이벤트를 제공하는 Server-Sent Events 스트림입니다. 상태를 잃지 않고 연결하거나 연결을 해제하려는 대시보드와 고급 클라이언트를 위해 설계되었습니다.

에이전트가 백그라운드 서브에이전트에 작업을 위임하면 스트림에는 `subagent.start` 및 `subagent.complete` 수명 주기 이벤트도 전달됩니다. 따라서 자식이 작업하는 동안 실행이 아무 응답도 하지 않는 대신, 시간 초과와 실패를 포함한 위임 결과를 클라이언트에서 확인할 수 있습니다. `subagent.complete` 페이로드에는 자식의 상태, 요약, 소요 시간, 토큰/비용 수치, 상관관계 확인을 위한 `child_session_id`가 포함됩니다. 자유 형식 텍스트 필드는 프로세스를 벗어나기 전에 강제 비밀 정보 삭제를 거칩니다. 자식별 도구 이벤트(`subagent.tool`, 진행 틱)는 고빈도 UI 잡음이므로 의도적으로 전달하지 않습니다. 작업의 상세 진행은 자식별 실시간 트랜스크립트 파일을 사용하세요.

소비되지 않은 이벤트 버퍼는 5분 후 만료되므로 연결이 끊긴 클라이언트가 메모리를 무한히 늘릴 수 없습니다. 이는 전송 상태만 만료시키며, 계속 실행 중인 run은 실행기 작업이 실제로 종료될 때까지 상태 폴링, 승인, 중지 제어, 동시성 계산에 계속 표시됩니다. 연결된 SSE 구독자는 정상적으로 계속 소진합니다.

### POST /v1/runs/\{run_id\}/stop

실행 중인 에이전트 턴을 중단합니다. Hermes가 활성 에이전트에 다음 안전한 중단 지점에서 멈추도록 요청하는 동안 엔드포인트는 `{"status": "stopping"}`을 즉시 반환합니다. 실행기는 실행기 기반 작업이 종료될 때까지 `stopping`으로 추적되며 이후 `cancelled`로 확정됩니다. 중지를 요청해도 아직 실행 중인 작업자가 숨겨지지 않습니다.

### POST /v1/runs/\{run_id\}/approval

사람의 결정을 기다리고 있는 실행의 대기 중 승인을 해결합니다(예: 승인 정책에 의해 게이트된 도구 호출). 본문에는 승인 결정이 포함되며, 결정이 기록되면 실행이 재개됩니다. 이 엔드포인트는 `/v1/capabilities`에서 `run_approval` 기능으로 광고되므로 외부 UI가 승인 프롬프트를 표시하기 전에 지원 여부를 확인할 수 있습니다.

## Jobs API (백그라운드 예약 작업)

서버는 원격 클라이언트에서 예약/백그라운드 에이전트 실행을 관리할 수 있는 간단한 jobs CRUD 표면을 제공합니다. 모든 엔드포인트는 동일한 bearer 인증으로 보호됩니다.

### GET /api/jobs

예약된 모든 작업을 나열합니다.

### POST /api/jobs

새 예약 작업을 생성합니다. 본문은 `hermes cron`과 동일한 형식(프롬프트, 일정, 스킬, 프로바이더 재정의, 전달 대상)을 허용합니다.

### GET /api/jobs/\{job_id\}

단일 작업의 정의와 최근 실행 상태를 가져옵니다.

### PATCH /api/jobs/\{job_id\}

기존 작업의 필드(프롬프트, 일정 등)를 수정합니다. 부분 업데이트는 병합됩니다.

### DELETE /api/jobs/\{job_id\}

작업을 제거합니다. 진행 중인 실행도 취소합니다.

### POST /api/jobs/\{job_id\}/pause

작업을 삭제하지 않고 일시 중지합니다. 재개할 때까지 다음 예약 실행 시각이 중단됩니다.

### POST /api/jobs/\{job_id\}/resume

이전에 일시 중지한 작업을 재개합니다.

### POST /api/jobs/\{job_id\}/run

일정과 관계없이 작업을 즉시 실행하도록 트리거합니다.

## Sessions API (REST를 통한 세션 제어)

외부 UI는 대시보드를 별도로 실행하지 않고도 REST를 통해 Hermes 세션을 관리할 수 있습니다. 모든 엔드포인트는 `API_SERVER_KEY`로 보호되며 `/api/sessions/*` 아래에 있습니다.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions` | 세션 나열(페이지 매김 — `limit`, `offset`, `source`, `include_children`) |
| `POST` | `/api/sessions` | 빈 세션 생성 |
| `GET` | `/api/sessions/{id}` | 세션 메타데이터 읽기 |
| `PATCH` | `/api/sessions/{id}` | 제목 또는 `end_reason` 업데이트 |
| `DELETE` | `/api/sessions/{id}` | 세션 삭제 |
| `GET` | `/api/sessions/{id}/messages` | 세션의 메시지 기록 |
| `POST` | `/api/sessions/{id}/fork` | `SessionDB` 계보를 통해 세션 분기(CLI `/branch` 의미 체계와 일치) |
| `POST` | `/api/sessions/{id}/chat` | 동기 에이전트 턴 1회 실행 |
| `POST` | `/api/sessions/{id}/chat/stream` | 단일 턴을 감싸는 SSE — `assistant.delta`, `tool.started`, `tool.completed`, `run.completed` 이벤트를 내보냄 |

`/v1/capabilities`는 `session_*` 기능 플래그와 `endpoints.session_*` 항목을 통해 전체 표면을 광고하므로 외부 UI가 지원 여부를 확인하고 안전하게 대체 경로를 사용할 수 있습니다. 인라인 이미지는 `chat` 및 `chat/stream` 페이로드에서 지원됩니다(멀티모달 인식 경로).

```bash
# fork a session and run one turn
curl -X POST http://localhost:8642/api/sessions/$ID/fork \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"title": "explore alt path"}'

# stream a turn over SSE
curl -N -X POST http://localhost:8642/api/sessions/$ID/chat/stream \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"input": "what files changed in the last hour?"}'
```

## 스킬 및 도구 세트 검색

`GET /v1/skills`와 `GET /v1/toolsets`를 사용하면 외부 클라이언트가 모델에 묻지 않고 REST를 통해 에이전트의 기능을 결정론적으로 열거할 수 있습니다. 둘 다 읽기 전용이며 `API_SERVER_KEY`로 보호됩니다.

```bash
curl http://localhost:8642/v1/skills \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "github-pr-workflow", "description": "...", "category": "..."}, ...]

curl http://localhost:8642/v1/toolsets \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "core", "label": "...", "description": "...", "enabled": true,
#     "configured": true, "tools": ["read_file", "write_file", ...]}, ...]
```

`/v1/skills`는 스킬 허브가 내부적으로 사용하는 것과 동일한 메타데이터를 반환합니다. `/v1/toolsets`는 `api_server` 플랫폼에 대해 확인된 도구 세트를 반환하며, 각 도구 세트가 확장되는 구체적인 `tools` 목록을 포함합니다. 둘 다 `/v1/capabilities`의 `endpoints.*` 아래에 광고됩니다.

## 장기 메모리 범위 지정(`X-Hermes-Session-Key`)

Open WebUI 같은 다중 사용자 프런트엔드는 트랜스크립트 범위의 `X-Hermes-Session-Id`(`/new`에서 순환)와 독립적인 장기 메모리(Honcho 등)용 안정적인 채널별 식별자가 필요합니다. `/v1/chat/completions`, `/v1/responses`, `/v1/runs`에 `X-Hermes-Session-Key`를 전달하면 Hermes가 이를 `AIAgent(gateway_session_key=...)`까지 전달하고, Honcho 메모리 프로바이더가 이를 사용해 안정적인 범위를 도출합니다.

```http
POST /v1/chat/completions HTTP/1.1
Authorization: Bearer ***
X-Hermes-Session-Id: transcript-alpha
X-Hermes-Session-Key: agent:main:webui:dm:user-42
```

규칙: 최대 256자이며, 제어 문자(`\r`, `\n`, `\x00`)는 거부되고 값은 응답(JSON + SSE)에 다시 포함됩니다. `/v1/capabilities`는 `"session_key_header": "X-Hermes-Session-Key"`를 통해 지원 여부를 광고합니다. 키가 없으면 Honcho의 `per-session` 전략은 `session_id`마다 서로 다른 범위를 생성합니다. 이는 Hermes가 이전에 사용하던 동작과 정확히 같습니다.

## 시스템 프롬프트 처리

프런트엔드가 `system` 메시지(Chat Completions) 또는 `instructions` 필드(Responses API)를 보내면 hermes-agent는 이를 핵심 시스템 프롬프트 위에 계층화합니다. 에이전트는 모든 도구, 메모리, 스킬을 유지하며 프런트엔드의 시스템 프롬프트는 추가 지침을 제공합니다.

따라서 기능을 잃지 않고 프런트엔드별 동작을 사용자 지정할 수 있습니다.
- Open WebUI 시스템 프롬프트: "You are a Python expert. Always include type hints."
- 에이전트에는 여전히 터미널, 파일 도구, 웹 검색, 메모리 등이 있습니다.

## 인증

`Authorization` 헤더를 통한 Bearer 토큰 인증입니다.

```
Authorization: Bearer ***
```

`API_SERVER_KEY` 환경 변수로 키를 구성합니다. 브라우저가 Hermes를 직접 호출해야 한다면 `API_SERVER_CORS_ORIGINS`도 명시적인 허용 목록으로 설정합니다.

### 멀티 프로필 라우팅(`/p/<profile>/…`)

[멀티 프로필 게이트웨이 라우팅](/user-guide/multi-profile-gateways)이 활성화된 경우(`gateway.multiplex_profiles`), 공유 리스너는 `/p/<profile>/` URL 접두사를 통해 모든 프로필을 제공하며 **인증은 라우팅된 프로필에 연결됩니다**.

- `/p/<profile>/v1/...` 요청은 해당 프로필의 자체 `API_SERVER_KEY`(`~/.hermes/profiles/<profile>/.env`)를 제시해야 합니다. 기본 리스너의 키는 이름이 지정된 프로필 접두사에서 거부됩니다.
- 접두사가 없는 경로와 `/p/default/...`는 기본 프로필의 키를 계속 사용합니다.
- 자체 `API_SERVER_KEY`가 없는 이름 지정 프로필은 안전하게 거부됩니다. 키를 설정할 때까지 해당 접두사에 접근할 수 없습니다.

:::warning 호환성을 깨는 변경 사항(2026년 7월)
이 수정 전에는 유효한 기본 프로필 키가 모든 `/p/<profile>/` 접두사에서 허용되었습니다. 프로필 접두사 전체에서 하나의 공유 키에 의존하고 있었다면 각 프로필의 `.env`에 별도의 `API_SERVER_KEY`를 설정하세요. 이제 이름 지정 프로필 접두사에서 재사용된 기본 키는 `401`을 반환합니다.
:::

:::warning 보안
API 서버는 **터미널 명령을 포함하여** hermes-agent 도구 세트에 대한 전체 접근 권한을 제공합니다. 기본 루프백 바인딩인 `127.0.0.1`을 포함한 **모든 배포에서** `API_SERVER_KEY`가 **필수입니다**. 브라우저 호출자를 명시적으로 허용할 때는 브라우저 접근을 제어할 수 있도록 `API_SERVER_CORS_ORIGINS`의 범위를 좁게 유지하세요.
:::

## 구성

### 환경 변수

| 변수 | 기본값 | 설명 |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | API 서버 활성화 |
| `API_SERVER_PORT` | `8642` | HTTP 서버 포트 |
| `API_SERVER_HOST` | `127.0.0.1` | 바인딩 주소(기본값은 localhost만) |
| `API_SERVER_KEY` | _(필수)_ | 인증용 Bearer 토큰 |
| `API_SERVER_CORS_ORIGINS` | _(없음)_ | 쉼표로 구분한 허용 브라우저 출처 |
| `API_SERVER_MODEL_NAME` | _(프로필 이름)_ | `/v1/models`의 모델 이름. 기본 프로필에서는 프로필 이름 또는 `hermes-agent`가 기본값입니다. |

### config.yaml

동일한 설정을 `~/.hermes/config.yaml`의 중첩된 `gateway.api_server:` 섹션에 둘 수 있습니다.

```yaml
gateway:
  api_server:
    enabled: true
    port: 8642
    host: 127.0.0.1
    key: your-secret-key
    cors_origins: http://localhost:3000
    model_name: my-hermes
    max_concurrent_runs: 10   # concurrent-run cap; 0 disables the limit
```

`port`, `key`, `host`, `cors_origins`, `model_name`은 자동으로 플랫폼의 `extra` 설정에 연결되므로 `API_SERVER_*` 환경 변수에 대응하는 설정과 정확히 동일하게 동작합니다. 환경 변수가 `config.yaml` 값보다 우선합니다. 이 블록은 `gateway.platforms.api_server:` 또는 최상위 `platforms.api_server:` 섹션에서도 허용됩니다.

### 동시 실행 제한

API 서버는 OpenAI 호환 엔드포인트와 Runs 엔드포인트 전체에서 동시에 실행할 수 있는 에이전트 실행 수를 제한합니다. 제한은 `gateway.api_server.max_concurrent_runs`에서 읽으며 기본값은 **10**이고, `0`은 제한을 비활성화하며 음수 값은 0으로 제한됩니다. 제한에 도달하면 새 실행 시작 요청은 **HTTP 429** `Too many concurrent runs (max N)`으로 거부됩니다. 클라이언트는 대기 후 재시도해야 합니다.

## 보안 헤더

모든 응답에는 보안 헤더가 포함됩니다.
- `X-Content-Type-Options: nosniff` — MIME 유형 스니핑 방지
- `Referrer-Policy: no-referrer` — 리퍼러 유출 방지

## CORS

API 서버는 기본적으로 브라우저 CORS를 활성화하지 **않습니다**.

브라우저에서 직접 접근하려면 명시적인 허용 목록을 설정합니다.

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

CORS가 활성화되면 다음과 같이 동작합니다.
- **Preflight 응답**에는 `Access-Control-Max-Age: 600`(10분 캐시)이 포함됩니다.
- **SSE 스트리밍 응답**에는 CORS 헤더가 포함되므로 브라우저 EventSource 클라이언트가 올바르게 작동합니다.
- **`Idempotency-Key`**는 허용되는 요청 헤더이므로 클라이언트가 중복 제거를 위해 보낼 수 있습니다(응답은 5분 동안 키별로 캐시됨).

Open WebUI 같은 대부분의 문서화된 프런트엔드는 서버 간 연결을 사용하므로 CORS가 전혀 필요하지 않습니다.

## 호환되는 프런트엔드

OpenAI API 형식을 지원하는 모든 프런트엔드가 작동합니다. 다음은 테스트 및 문서화된 통합입니다.

| 프런트엔드 | 별 | 연결 |
|----------|-------|------------|
| [Open WebUI](/user-guide/messaging/open-webui) | 126k | 전체 가이드 제공 |
| LobeChat | 73k | 사용자 지정 프로바이더 엔드포인트 |
| LibreChat | 34k | librechat.yaml의 사용자 지정 엔드포인트 |
| AnythingLLM | 56k | 일반 OpenAI 프로바이더 |
| NextChat | 87k | BASE_URL 환경 변수 |
| ChatBox | 39k | API Host 설정 |
| Jan | 26k | 원격 모델 구성 |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | 사용자 지정 엔드포인트 |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | 직접 HTTP 요청 |

## 프로필을 사용한 다중 사용자 설정

여러 사용자에게 각각 격리된 Hermes 인스턴스(별도의 구성, 메모리, 스킬)를 제공하려면 [프로필](/user-guide/profiles)을 사용합니다.

```bash
# Create a profile per user
hermes profile create alice
hermes profile create bob

# Configure each profile's API server on a different port. API_SERVER_* are env
# vars (not config.yaml keys), so write them to each profile's .env:
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

# Start each profile's gateway
hermes -p alice gateway &
hermes -p bob gateway &
```

각 프로필의 API 서버는 프로필 이름을 모델 ID로 자동 광고합니다.

- `http://localhost:8643/v1/models` → 모델 `alice`
- `http://localhost:8644/v1/models` → 모델 `bob`

Open WebUI에서 각각을 별도의 연결로 추가합니다. 모델 드롭다운에는 `alice`와 `bob`이 서로 다른 모델로 표시되며, 각 모델은 완전히 격리된 Hermes 인스턴스를 기반으로 합니다. 자세한 내용은 [Open WebUI 가이드](/user-guide/messaging/open-webui#multi-user-setup-with-profiles)를 참조하세요.

## 제한 사항

- **응답 저장** — `previous_response_id`에 사용되는 저장된 응답은 SQLite에 유지되며 게이트웨이를 다시 시작해도 보존됩니다. 저장되는 응답은 최대 100개입니다(LRU 제거).
- **파일 업로드 없음** — `/v1/chat/completions`와 `/v1/responses` 모두에서 인라인 이미지를 지원하지만, 업로드된 파일(`file`, `input_file`, `file_id`)과 이미지가 아닌 문서 입력은 API를 통해 지원하지 않습니다.
- **단순 OpenAI 클라이언트에도 별칭이 표시됨** — `/v1/models`는 안정적인 Hermes 별칭(`hermes-agent` 또는 활성 프로필 이름)을 광고합니다. 더 풍부한 클라이언트는 요청에서 명시적인 `provider` / `model_options` 재정의를 보낼 수 있습니다.

## 프록시 모드

API 서버는 **게이트웨이 프록시 모드**의 백엔드 역할도 합니다. 다른 Hermes 게이트웨이 인스턴스가 `GATEWAY_PROXY_URL`을 이 API 서버를 가리키도록 구성되면 자체 에이전트를 실행하는 대신 모든 메시지를 이곳으로 전달합니다. 이를 통해 분리된 배포가 가능합니다. 예를 들어 Docker 컨테이너가 Matrix E2EE를 처리하고 호스트 측 에이전트로 중계할 수 있습니다.

전체 설정 가이드는 [Matrix 프록시 모드](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)를 참조하세요.
