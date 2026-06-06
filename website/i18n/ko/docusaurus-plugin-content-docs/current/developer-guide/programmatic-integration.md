---
sidebar_position: 8
title: "프로그래밍 방식의 통합 (Programmatic Integration)"
description: "외부 프로그램에서 hermes-agent를 제어하기 위한 세 가지 프로토콜: ACP, TUI 게이트웨이 JSON-RPC, 그리고 OpenAI 호환 HTTP API"
---

# 프로그래밍 방식의 통합 (Programmatic Integration)

Hermes는 IDE 플러그인, 사용자 지정 UI, CI 파이프라인, 임베디드 하위 에이전트 등 외부 프로그램에서 에이전트를 구동하기 위해 세 가지 프로토콜을 제공합니다. 사용 중인 트랜스포트와 소비자에 맞는 프로토콜을 선택하세요.

| 프로토콜 | 트랜스포트 | 적합한 대상 | 정의 위치 |
|----------|-----------|----------|------------|
| **ACP** | stdio를 통한 JSON-RPC | [Agent Client Protocol](https://github.com/zed-industries/agent-client-protocol)과 이미 통신할 수 있는 IDE 클라이언트 (VS Code, Zed, JetBrains) | `acp_adapter/` |
| **TUI 게이트웨이** | stdio (또는 WebSocket)를 통한 JSON-RPC | 세션, 슬래시 명령어, 승인 및 스트리밍 이벤트에 대한 세분화된 제어를 원하는 사용자 지정 호스트 | `tui_gateway/server.py` |
| **API 서버** | HTTP + Server-Sent Events | OpenAI 호환 프론트엔드 (Open WebUI, LobeChat, LibreChat 등) 및 언어에 구애받지 않는 웹 클라이언트 | `gateway/platforms/api_server.py` |

이 세 가지 모두 동일한 `AIAgent` 코어를 구동합니다. 단지 전송 형식(wire format)과 노출하는 기능 세트가 다를 뿐입니다.

---

## ACP (에이전트 클라이언트 프로토콜, Agent Client Protocol)

`hermes acp`는 ACP 통신을 수행하는 stdio 기반 JSON-RPC 서버를 시작합니다. 이는 프로덕션 환경에서 VS Code (Zed Industries의 ACP 확장 프로그램), Zed, 그리고 ACP 플러그인이 설치된 모든 JetBrains IDE에 사용됩니다.

노출된 기능: 세션 생성, 프롬프트 제출, 스트리밍 에이전트 메시지 청크, 도구 호출 이벤트, 권한 요청, 세션 포크(fork), 취소 및 인증. 도구 출력은 IDE가 이해할 수 있는 ACP `Diff`/`ToolCall` 콘텐츠 블록으로 렌더링됩니다.

전체 라이프사이클, 이벤트 브릿지 및 승인 흐름: [ACP 내부 동작(ACP Internals)](./acp-internals).

```bash
hermes acp                  # stdio로 ACP 서버 제공
hermes acp --bootstrap      # ACP 지원 IDE를 위한 설치 스니펫 출력
```

---

## TUI 게이트웨이 JSON-RPC (TUI Gateway JSON-RPC)

`tui_gateway/server.py`는 Ink TUI(`hermes --tui`)와 임베디드 대시보드 PTY 브릿지가 통신하는 프로토콜입니다. 모든 외부 호스트는 stdio (또는 `tui_gateway/ws.py`를 통한 WebSocket)를 통해 동일한 프로토콜로 통신할 수 있습니다.

### 메서드 카탈로그 (선택적)

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
delegation.status       subagent.interrupt      spawn_tree.save / list / load
terminal.resize         clipboard.paste         image.attach
```

`session.active_list`, `session.activate`, `session.close`는 TUI 세션 전환기가 사용하는 프로세스 로컬 활성 세션 제어입니다. 저장된 기록을 찾으려면 `session.list` / `/resume`을 사용하세요. 활성 세션 메서드는 TUI 게이트웨이 프로세스에서 현재 열려 있는 세션에만 사용하세요.

### 스트리밍 이벤트 응답

`message.delta`, `message.complete`, `tool.start`, `tool.progress`, `tool.complete`, `approval.request`, `clarify.request`, `sudo.request`, `secret.request`, `gateway.ready`, 그리고 세션 라이프사이클 및 오류 이벤트가 전송됩니다.

### Pi 스타일 RPC 매핑

Pi-mono RPC 사양([이슈 #360](https://github.com/NousResearch/hermes-agent/issues/360))의 모든 명령어는 TUI 게이트웨이에 동일하게 매핑됩니다:

| Pi 명령어 | Hermes 대응 |
|------------|-------------------|
| `prompt` | `prompt.submit` (또는 ACP `session/prompt`) |
| `steer` | `session.steer` |
| `follow_up` | 현재 턴 종료 후 대기 중인 `prompt.submit` |
| `abort` | `session.interrupt` |
| `set_model` | `/model <provider:model>`을 위한 `command.dispatch` (세션 진행 중 적용, 영구적) |
| `compact` | `session.compress` |
| `get_state` | `session.status` |
| `get_messages` | `session.history` |
| `switch_session` | `session.resume` |
| `fork` | `session.branch` |
| `ui_request` / `ui_response` | `clarify.respond` / `sudo.respond` / `secret.respond` / `approval.respond` |

---

## OpenAI 호환 API 서버 (OpenAI-Compatible API Server)

`gateway/platforms/api_server.py`는 이미 OpenAI 형식을 사용하는 모든 클라이언트가 HTTP를 통해 Hermes를 사용할 수 있도록 노출합니다. 웹 프론트엔드, curl 기반 CI 실행기, 파이썬 이외의 소비자를 사용할 때 유용합니다.

엔드포인트:

```
POST /v1/chat/completions        OpenAI 채팅 완료 (SSE 스트리밍)
POST /v1/responses               OpenAI 응답 API (상태 유지)
POST /v1/runs                    실행 시작, run_id 반환 (202)
GET  /v1/runs/{id}               실행 상태
GET  /v1/runs/{id}/events        라이프사이클 이벤트의 SSE 스트림
POST /v1/runs/{id}/approval      대기 중인 승인 처리
POST /v1/runs/{id}/stop          실행 인터럽트
GET  /v1/capabilities            기계가 판독 가능한 기능 플래그
GET  /v1/models                  hermes-agent 목록 제공
GET  /health, /health/detailed
```

설정, 헤더 (`X-Hermes-Session-Id`, `X-Hermes-Session-Key`) 및 프론트엔드 연결: [API 서버](../user-guide/features/api-server) 참고.

---

## 어떤 것을 사용해야 하나요?

- **IDE 플러그인을 작성 중이며 IDE가 이미 ACP를 지원하는 경우** → ACP. IDE 측에서 추가 프로토콜 작업이 전혀 필요하지 않습니다.
- **슬래시 명령어, 승인, 질문 구체화(clarify), 다중 에이전트, 세션 분기 등 Hermes의 모든 기능을 원하는 사용자 지정 데스크톱 / 웹 / TUI 호스트를 작성 중인 경우** → TUI 게이트웨이 JSON-RPC.
- **OpenAI 호환 프론트엔드, 언어에 구애받지 않는 HTTP 클라이언트 또는 curl 기반 자동화가 필요한 경우** → API 서버.
- **하위 프로세스 없이 파이썬 내장 프로세스에 직접 포함하려는 경우** → 직접 `run_agent.AIAgent`를 가져오세요(import). [Agent Loop](./agent-loop)를 참고하세요.

---

## 모델 핫 스와핑 (Model hot-swapping)

세션 진행 중 모델 전환은 모든 표면에서 동작합니다 — 이는 보이지 않는 곳에서 작동하는 `/model` 슬래시 명령어입니다.

- **CLI / TUI:** `/model claude-sonnet-4` 또는 `/model openrouter:anthropic/claude-sonnet-4.6`
- **TUI 게이트웨이 RPC:** `{"command": "/model claude-sonnet-4"}`를 포함하는 `command.dispatch` 호출
- **ACP:** IDE는 슬래시 명령어를 프롬프트로 전송하고, 에이전트가 이를 처리합니다.
- **API 서버:** 요청 본문에 `model` 필드를 포함하거나 `X-Hermes-Model` 헤더를 설정합니다.

제공자를 인식하는 결정 논리(같은 모델 이름이 사용 중인 제공자에 맞춰 적절한 형식을 선택함)가 내장되어 있습니다. `hermes_cli/model_switch.py`를 참조하세요.

---

## `--mode rpc`에 대한 참고 사항

Hermes에는 `--mode rpc` 플래그가 없습니다. 앞서 설명한 세 가지 프로토콜이 이미 주요 활용 사례를 다루고 있기 때문입니다. — IDE 프로토콜 클라이언트에는 ACP, stdio JSON-RPC 호스트에는 TUI 게이트웨이, HTTP 통신에는 API 서버. 이 세 가지가 포괄하지 못하는 실제 부족한 부분을 발견했다면, 개발 중인 구체적인 소비자와 함께 이슈를 열어 주세요.
