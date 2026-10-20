---
sidebar_position: 8
title: "프로그래밍 방식 통합"
description: "외부 프로그램에서 hermes-agent를 구동하는 세 가지 프로토콜: ACP, TUI 게이트웨이 JSON-RPC, OpenAI 호환 HTTP API"
---

# 프로그래밍 방식 통합

Hermes는 IDE 플러그인, 사용자 지정 UI, CI 파이프라인, 임베디드 서브 에이전트 등 외부 프로그램에서 에이전트를 구동할 수 있는 세 가지 프로토콜을 제공합니다. 사용하는 전송 방식과 소비자에 맞는 것을 선택하세요.

| 프로토콜 | 전송 방식 | 적합한 용도 | 정의 위치 |
|----------|----------|------------|------------|
| **ACP** | stdio를 통한 JSON-RPC | 이미 [Agent Client Protocol](https://github.com/zed-industries/agent-client-protocol)을 사용하는 IDE 클라이언트(VS Code, Zed, JetBrains) | `acp_adapter/` |
| **TUI 게이트웨이** | stdio(또는 WebSocket)를 통한 JSON-RPC | 세션, 슬래시 명령, 승인, 스트리밍 이벤트를 세밀하게 제어하려는 사용자 지정 호스트 | `tui_gateway/server.py` |
| **API 서버** | HTTP + Server-Sent Events | OpenAI 호환 프런트엔드(Open WebUI, LobeChat, LibreChat…) 및 언어에 종속되지 않는 웹 클라이언트 | `gateway/platforms/api_server.py` |

세 프로토콜 모두 동일한 `AIAgent` 코어를 구동합니다. 차이는 유선 형식과 노출하는 기능 집합뿐입니다.

---

## ACP(Agent Client Protocol)

`hermes acp`는 ACP를 사용하는 stdio JSON-RPC 서버를 시작합니다. VS Code(Zed Industries의 ACP 확장), Zed, ACP 플러그인이 있는 모든 JetBrains IDE에서 프로덕션 용도로 사용됩니다.

노출되는 기능: 세션 생성, 프롬프트 제출, 에이전트 메시지 청크 스트리밍, 도구 호출 이벤트, 권한 요청, 세션 포크, 취소, 인증. 도구 출력은 IDE가 이해할 수 있는 ACP `Diff`/`ToolCall` 콘텐츠 블록으로 렌더링됩니다.

전체 수명 주기, 이벤트 브리지, 승인 흐름은 [ACP 내부 구조](./acp-internals)를 참조하세요.

```bash
hermes acp                  # serve ACP on stdio
hermes acp --check          # verify ACP dependencies and adapter imports
hermes acp --setup          # interactive provider/model setup for ACP terminal auth
```

---

## TUI 게이트웨이 JSON-RPC

`tui_gateway/server.py`는 Ink TUI(`hermes --tui`)와 임베디드 대시보드 PTY 브리지가 통신하는 프로토콜입니다. 모든 외부 호스트는 stdio(또는 `tui_gateway/ws.py`를 통한 WebSocket)로 동일한 프로토콜을 사용할 수 있습니다.

### 메서드 카탈로그(일부)

```
prompt.submit           prompt.background       session.steer
session.create          session.list            session.active_list
session.activate        session.close           session.interrupt
session.history         session.compress        session.branch
session.title           session.usage           session.status
clarify.respond         sudo.respond            secret.respond
approval.respond        config.set / config.get commands.catalog
command.resolve         command.dispatch        cli.exec
reload.mcp              reload.env              process.stop
delegation.status       subagent.interrupt      subagent.steer
spawn_tree.save / list / load
terminal.resize         clipboard.paste         image.attach
```

`session.active_list`, `session.activate`, `session.close`는 TUI 세션 전환기에서 사용하는 프로세스 로컬 실시간 세션 제어 기능입니다. 저장된 대화 기록을 찾으려면 `session.list` / `/resume`을 사용하고, 현재 TUI 게이트웨이 프로세스에서 열려 있는 세션에는 실시간 세션 메서드만 사용하세요.

### `prompt.submit`에서 기록 되감기

되감기 / 편집 / 재생성은 새 턴을 실행하기 전에 저장된 대화 기록의 일부를 삭제하는 `prompt.submit`입니다. 이 작업은 세션의 영속 행을 파괴적으로 다시 쓰므로, 게이트웨이는 클라이언트가 의도를 명시한 경우에만 이를 처리합니다.

| 매개변수 | 의미 |
|-----------|---------|
| `truncate_before_user_ordinal` | 잘라낼 사용자 턴의 0부터 시작하는 인덱스입니다. 해당 턴부터 이후의 모든 항목이 삭제됩니다. 표시 전용 타임라인 행(`display_kind`)은 계산하지 않습니다. 실제 정수여야 하며 JSON 불리언은 코드 `4004`로 거부됩니다. |
| `truncate_before_row_id` | 잘라낼 대상 사용자 턴의 정수 SQLite 행 ID(`messages.id` / `row_id`)입니다. 영속 주소로 권장됩니다. 서수와 행 ID를 모두 제공하면 게이트웨이가 일치 여부를 검증하며(불일치 시 `4030` 반환), 알 수 없거나 오래된 행 ID는 `4018`로 거부되고 서수로 대체되지 않습니다. |
| `confirm_truncate` | 서수, 메시지 ID 또는 행 ID를 보낼 때마다 필요합니다. 이 제출이 남은 매개변수를 우연히 포함한 일반 전송이 아니라 실제 되감기임을 선언합니다. 대상 없이 보내면 코드 `4004`로 거부됩니다. |
| `confirm_empty_truncate` | 잘라낸 결과 대화 기록이 비게 되는 경우(서수 `0`) 추가로 필요합니다. |

`confirm_truncate` 없이 잘라내기 매개변수를 보내면 코드 `4004` 또는 `4029`로 거부되며 아무것도 기록되지 않습니다. 되감기를 구현하는 호스트는 사용자가 되감기를 요청하는 순간 플래그를 설정해야 하며, 일반 제출 사이에 잘라내기 매개변수를 상태로 유지해서는 안 됩니다. `truncate_before_row_id`(resume의 `row_id` / `_row_id`에서 가져옴)를 서수보다 우선 사용하고, 아직 영속 ID를 사용할 수 없을 때만 이전 버전 호환 / 낙관적 행 경로로 서수를 유지하세요.

영속 세션에 대한 잘라내기 제출이 성공하면 `prompt.submit` 결과에 `survivor_user_row_ids`도 포함됩니다. 이는 살아남은 사용자 턴의 새 다시 쓰기 후 행 ID를 표시되는 사용자 서수 순서로 담습니다. 다시 쓰기는 보존된 접두사를 새 행으로 재삽입하므로, 되감기 전에 호스트가 캐시한 모든 행 ID는 이후 오래된 값이 됩니다. 이 목록에서 캐시된 ID를 다시 연결하세요(해당 턴에 영속 ID가 없음을 뜻하는 `null` 항목은 캐시된 ID를 삭제). 그렇지 않으면 더 오래된 생존 턴을 대상으로 하는 다음 되감기가 `4018`로 거부됩니다.

### 스트리밍으로 반환되는 이벤트

`message.delta`, `message.complete`, `tool.start`, `tool.progress`, `tool.complete`, `approval.request`, `clarify.request`, `sudo.request`, `sudo.expire`, `secret.request`, `secret.expire`, `gateway.ready`와 세션 수명 주기 및 오류 이벤트가 반환됩니다. 만료 이벤트에는 원래 `{ request_id }`가 포함되므로 외부 호스트는 일치하는 대기 중 프롬프트만 삭제해야 합니다.

### Pi 스타일 RPC 매핑

Pi-mono RPC 사양의 모든 명령([이슈 #360](https://github.com/NousResearch/hermes-agent/issues/360))에는 TUI 게이트웨이 대응 항목이 있습니다.

| Pi 명령 | Hermes 대응 항목 |
|------------|-------------------|
| `prompt` | `prompt.submit`(또는 ACP `session/prompt`) |
| `steer` | `session.steer` |
| `follow_up` | 현재 턴 이후 대기열에 추가되는 `prompt.submit` |
| `abort` | `session.interrupt` |
| `set_model` | `/model <provider:model>`에 대한 `command.dispatch`(세션 중간에 영속적으로 적용) |
| `compact` | `session.compress` |
| `get_state` | `session.status` |
| `get_messages` | `session.history` |
| `switch_session` | `session.resume` |
| `fork` | `session.branch` |
| `ui_request` / `ui_response` | `clarify.respond` / `sudo.respond` / `secret.respond` / `approval.respond` |

---

## OpenAI 호환 API 서버

`gateway/platforms/api_server.py`는 이미 OpenAI 형식을 사용하는 모든 클라이언트에 HTTP로 Hermes를 제공합니다. 웹 프런트엔드, curl 기반 CI 실행기 또는 Python을 사용하지 않는 소비자에 유용합니다.

엔드포인트:

```
POST /v1/chat/completions        OpenAI Chat Completions (streaming via SSE)
POST /v1/responses               OpenAI Responses API (stateful)
POST /v1/runs                    Start a run, returns run_id (202)
GET  /v1/runs/{id}               Run status
GET  /v1/runs/{id}/events        SSE stream of lifecycle events
POST /v1/runs/{id}/approval      Resolve a pending approval
POST /v1/runs/{id}/steer         Inject mid-run guidance at the next tool boundary
POST /v1/runs/{id}/stop          Interrupt the run
GET  /v1/capabilities            Machine-readable feature flags
GET  /v1/models                  Lists hermes-agent
GET  /api/model/options          Provider-aware picker inventory
GET  /health, /health/detailed
```

설정, 헤더(`X-Hermes-Session-Id`, `X-Hermes-Session-Key`), 프런트엔드 연결은 [API 서버](../user-guide/features/api-server)를 참조하세요.

### 모델 카탈로그 표면

OpenAI 호환 API는 `GET /v1/models`를 의도적으로 최소화합니다. 이 엔드포인트는 전체 Hermes 프로바이더/모델 선택기 카탈로그가 아니라 프런트엔드가 기대하는 호환성 엔드포인트입니다.

외부 제어 플레인에서 Hermes가 선별한 프로바이더 행, 모델별 가격 또는 기능 힌트가 필요하다면 인증이 필요한 선택기 표면 중 하나를 사용하세요.

- API 서버 REST: API 서버 bearer 키와 함께 `GET /api/model/options`
- 대시보드 백엔드 REST: `X-Hermes-Session-Token`과 함께 `GET /api/model/options`
- TUI 게이트웨이 RPC: `model.options`

이 표면들은 동일한 페이로드 빌더와 동일한 사용자 지정 프로바이더 탐색 정책을 공유합니다.

- 일반 열기: 현재 사용자 지정 프로바이더만 탐색하여 오프라인에 저장된 엔드포인트가 선택기를 멈추게 하지 않습니다.
- 명시적 새로 고침(`refresh=1` 또는 `refresh: true`): 프로바이더-모델 캐시를 무효화하고 저장된 모든 사용자 지정 프로바이더를 탐색하여 실시간 카탈로그를 완전히 다시 채웁니다.

OpenAI 클라이언트 호환성에는 `/v1/models`를 사용하세요. Hermes를 인식하는 모델 선택기를 만들 때는 `/api/model/options` 또는 `model.options`를 사용하세요.

`POST /v1/runs/{id}/steer`는 Hermes `/steer`의 HTTP 대응 항목입니다. 새 사용자 턴을 만들거나 현재 진행 중인 어시스턴트 출력을 즉시 다시 쓰지 않습니다. 대신 텍스트를 실행 중인 run에 추가하고 다음 도구 경계 이후 에이전트에 표시하므로, 현재 도구 호출 루프를 폐기하지 않고 방향을 수정할 수 있습니다.

`/v1/runs/{id}/steer`는 run 상태가 `running`일 때만 허용됩니다. 대기 중, 승인 일시 중지, 중지 중, 취소, 실패, 완료 상태의 run은 서버가 협력적 종료 중에도 내부 에이전트 참조를 보유하고 있더라도 `409 run_not_accepting_steer`를 반환합니다.

`200`(및 `run.steered` 이벤트)은 텍스트가 **대기열에 추가되었다**는 뜻이지 에이전트가 이를 소비했다는 뜻은 아닙니다. 에이전트의 최종 응답 이후에 steer가 도착하여 전달할 다음 도구 경계가 없다면, 전달되지 않은 텍스트는 종료된 `run.completed` 이벤트와 run 상태의 `pending_steer`로 반환됩니다. 따라서 클라이언트는 이를 잃지 않고 다음 사용자 턴으로 재생할 수 있습니다.

---

## 어떤 것을 사용해야 하나요?

- **IDE 플러그인을 작성 중이고 IDE가 이미 ACP를 사용함** → ACP. IDE 측에서 프로토콜 작업이 필요하지 않습니다.
- **사용자 지정 데스크톱 / 웹 / TUI 호스트를 작성 중이며 모든 Hermes 기능**(슬래시 명령, 승인, 명확화, 멀티 에이전트, 세션 분기)을 원함 → TUI 게이트웨이 JSON-RPC.
- **OpenAI 호환 프런트엔드, 언어에 종속되지 않는 HTTP 클라이언트 또는 curl 기반 자동화**를 원함 → API 서버.
- **서브프로세스 없이 Python 프로세스 내부에 임베드**하려 함 → `run_agent.AIAgent`를 직접 가져오세요. [에이전트 루프](./agent-loop)를 참조하세요.

---

## 모델 핫 스와핑

세션 중간의 모델 전환은 모든 표면에서 작동하며, 내부적으로는 `/model` 슬래시 명령입니다.

- **CLI / TUI:** `/model claude-sonnet-4` 또는 `/model openrouter:anthropic/claude-sonnet-4.6`
- **TUI 게이트웨이 RPC:** `{"command": "/model claude-sonnet-4"}`와 함께 `command.dispatch`
- **ACP:** IDE가 슬래시 명령을 프롬프트로 보내면 에이전트가 이를 디스패치합니다
- **API 서버:** 요청 본문에 `model` 필드를 포함합니다

프로바이더를 인식하는 해석(현재 프로바이더에 맞는 형식을 동일한 모델 이름으로 선택)은 기본 제공됩니다. `hermes_cli/model_switch.py`를 참조하세요.

---

## `--mode rpc`에 대한 참고

Hermes에는 `--mode rpc` 플래그가 없습니다. 위의 세 프로토콜이 이미 사용 사례를 다룹니다. IDE 프로토콜 클라이언트에는 ACP, stdio JSON-RPC 호스트에는 TUI 게이트웨이, HTTP에는 API 서버를 사용하세요. 어느 것도 충족하지 못하는 실제 공백을 발견했다면, 구축 중인 구체적인 소비자와 함께 이슈를 열어 주세요.
