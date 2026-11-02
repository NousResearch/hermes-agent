---
sidebar_position: 18
title: "브라우저 CDP 감독자"
description: "Hermes가 네이티브 JS 대화 상자를 감지하고 응답하며, 영구 CDP 연결을 통해 출처가 다른 iframe과 상호작용하는 방식입니다."
---

# 브라우저 CDP 감독자

CDP 감독자는 Hermes 브라우저 도구의 오랜 두 가지 공백을 해소합니다.

1. **네이티브 JS 대화 상자**(`alert`/`confirm`/`prompt`/`beforeunload`)는 페이지의 JS 스레드를 차단합니다. 감독이 없으면 에이전트는 대화 상자가 열려 있는지 알 수 없으며, 이후 도구 호출이 멈추거나 불투명한 오류를 발생시킵니다.
2. **출처가 다른 iframe(OOPIF)**은 최상위 `Runtime.evaluate`에서 보이지 않습니다. 에이전트는 DOM 스냅샷에서 iframe 노드를 볼 수 있지만, 자식 대상에 CDP 세션을 연결하지 않으면 그 안에서 클릭하거나 입력하거나 평가할 수 없습니다.

감독자는 브라우저 작업별로 백엔드의 CDP 엔드포인트에 영구 WebSocket을 유지하여 두 문제를 모두 해결합니다. 또한 대기 중인 대화 상자와 프레임 구조를 `browser_snapshot`에 표시하고, 명시적인 응답을 위한 `browser_dialog` 도구를 제공합니다.

## 백엔드 지원

| 백엔드 | 대화 상자 감지 | 대화 상자 응답 | 프레임 트리 | `browser_cdp(frame_id=...)`를 통한 OOPIF `Runtime.evaluate` |
|---|---|---|---|---|
| 로컬 Chrome(`--remote-debugging-port`) / `/browser connect` | ✓ | ✓ 전체 워크플로 | ✓ | ✓ |
| Browserbase | ✓(브리지 사용) | ✓ 전체 워크플로(브리지 사용) | ✓ | ✓ |
| Camofox | ✗ CDP 없음(REST 전용) | ✗ | DOM 스냅샷을 통한 부분 지원 | ✗ |

**Browserbase 특이 사항.** Browserbase의 CDP 프록시는 내부적으로 Playwright를 사용하며 네이티브 대화 상자를 약 10ms 이내에 자동으로 닫기 때문에 `Page.handleJavaScriptDialog`가 따라잡을 수 없습니다. 감독자는 `Page.addScriptToEvaluateOnNewDocument`를 통해 브리지 스크립트를 주입하고, `window.alert`/`confirm`/`prompt`를 매직 호스트(`hermes-dialog-bridge.invalid`)에 대한 동기 XHR로 재정의합니다. `Fetch.enable`은 이 XHR이 네트워크에 도달하기 전에 가로채며, 대화 상자는 감독자가 캡처하는 `Fetch.requestPaused` 이벤트가 됩니다. 이후 `respond_to_dialog`가 `Fetch.fulfillRequest`를 통해 주입된 스크립트가 디코딩하는 JSON 본문으로 응답을 완료합니다.

페이지 관점에서 `prompt()`는 여전히 에이전트가 제공한 문자열을 반환합니다. 에이전트 관점에서는 어느 경우든 동일한 `browser_dialog(action=...)` API입니다.

Camofox는 지원되지 않습니다 — CDP 표면이 없고 REST만 지원합니다.

## 아키텍처

### CDPSupervisor

Hermes `task_id`마다 백그라운드 데몬 스레드에서 실행되는 `asyncio.Task` 하나를 둡니다. 백엔드의 CDP 엔드포인트에 영구 WebSocket을 유지하며 다음을 관리합니다.

- **대화 상자 큐** — `{id, type, message, default_prompt, session_id, opened_at}`을 포함하는 `List[PendingDialog]`
- **프레임 트리** — 부모 관계, URL, 출처, 출처가 다른 자식 세션인지 여부를 포함하는 `Dict[frame_id, FrameInfo]`
- **세션 맵** — OOPIF 작업을 위해 상호작용 도구를 올바른 연결된 세션으로 라우팅하는 `Dict[session_id, SessionInfo]`
- **최근 콘솔 오류** — 진단을 위한 최근 50개 링 버퍼

연결 시 다음을 구독합니다.

- `Page.enable` — `javascriptDialogOpening`, `frameAttached`, `frameNavigated`, `frameDetached`
- `Runtime.enable` — `executionContextCreated`, `consoleAPICalled`, `exceptionThrown`
- `Target.setAutoAttach {autoAttach: true, flatten: true}` — 자식 OOPIF 대상을 표시하며, 감독자는 각 대상에서 `Page`와 `Runtime`을 활성화합니다.

스냅샷 잠금을 통해 스레드 안전한 상태 접근을 제공합니다. 동기식 도구 핸들러는 대기 없이 고정된 스냅샷을 읽습니다.

### 수명 주기

- **시작:** `SupervisorRegistry.get_or_start(task_id, cdp_url)` — `browser_navigate`, Browserbase 세션 생성, `/browser connect`에서 호출됩니다. 멱등적입니다.
- **중지:** 세션 해제 또는 `/browser disconnect`. asyncio 작업을 취소하고 WebSocket을 닫은 뒤 상태를 폐기합니다.
- **재연결:** CDP URL이 변경되면(사용자가 새 Chrome에 재연결할 때) 기존 감독자를 중지하고 새 감독자를 시작합니다 — 엔드포인트 간에 상태를 재사용하지 않습니다.

### 대화 상자 정책

`config.yaml`의 `browser.dialog_policy`에서 구성할 수 있습니다.

- **`must_respond`**(기본값) — 캡처하여 `browser_snapshot`에 표시하고, 명시적인 `browser_dialog(action=...)` 호출을 기다립니다. 응답 없이 300초의 안전 시간 초과가 지나면 자동으로 닫고 기록합니다. 문제가 있는 에이전트가 영원히 멈추는 것을 방지합니다.
- `auto_dismiss` — 즉시 기록하고 닫습니다. 에이전트는 이후 `browser_snapshot` 내부의 `browser_state`를 통해 이를 확인합니다.
- `auto_accept` — 기록하고 수락합니다(`beforeunload`에서 워크플로가 깔끔하게 이동하려 할 때 유용).

정책은 작업별로 적용되며 대화 상자별 재정의는 지원하지 않습니다.

## 에이전트 표면

### `browser_dialog` 도구

```
browser_dialog(action, prompt_text=None, dialog_id=None)
```

- `action="accept"` / `"dismiss"` → 지정한 대화 상자 또는 대기 중인 유일한 대화 상자에 응답(필수)
- `prompt_text=...` → `prompt()` 대화 상자에 입력할 텍스트
- `dialog_id=...` → 여러 대화 상자가 큐에 있을 때 구분(드묾)

이 도구는 응답 전용입니다. 호출하기 전에 에이전트는 `browser_snapshot` 출력에서 대기 중인 대화 상자를 읽습니다.

### `browser_snapshot` 확장

감독자가 연결되어 있으면 기존 스냅샷 출력에 다음 세 가지 선택적 필드를 추가합니다.

```json
{
  "pending_dialogs": [
    {"id": "d-1", "type": "alert", "message": "Hello", "opened_at": 1650000000.0}
  ],
  "recent_dialogs": [
    {"id": "d-1", "type": "alert", "message": "...", "opened_at": 1650000000.0,
     "closed_at": 1650000000.1, "closed_by": "remote"}
  ],
  "frame_tree": {
    "top": {"frame_id": "FRAME_A", "url": "https://example.com/", "origin": "https://example.com"},
    "children": [
      {"frame_id": "FRAME_B", "url": "about:srcdoc", "is_oopif": false},
      {"frame_id": "FRAME_C", "url": "https://ads.example.net/", "is_oopif": true, "session_id": "SID_C"}
    ],
    "truncated": false
  }
}
```

- **`pending_dialogs`** — 현재 페이지의 JS 스레드를 차단하는 대화 상자입니다. 에이전트는 응답하기 위해 `browser_dialog(action=...)`을 호출해야 합니다. Browserbase에서는 CDP 프록시가 약 10ms 이내에 자동으로 닫기 때문에 비어 있습니다.

- **`recent_dialogs`** — 최근 닫힌 대화 상자를 최대 20개까지 저장하는 링 버퍼이며 `closed_by` 태그를 포함합니다: `"agent"`(에이전트가 응답), `"auto_policy"`(로컬 `auto_dismiss`/`auto_accept`), `"watchdog"`(`must_respond` 시간 초과), 또는 `"remote"`(브라우저/백엔드가 닫음, 예: Browserbase). 이를 통해 Browserbase의 에이전트도 발생한 일을 확인할 수 있습니다.

- **`frame_tree`** — 출처가 다른(OOPIF) 자식을 포함한 프레임 구조입니다. 광고가 많은 페이지에서 스냅샷 크기를 제한하기 위해 항목은 30개, OOPIF 깊이는 2로 제한됩니다. 제한에 도달하면 `truncated: true`가 표시됩니다. 전체 트리가 필요한 에이전트는 `Page.getFrameTree`와 함께 `browser_cdp`를 사용할 수 있습니다.

이 중 어떤 기능에도 새로운 도구 스키마 표면은 필요하지 않습니다 — 에이전트는 이미 요청하는 스냅샷을 읽습니다.

### 사용 가능성 게이팅

두 표면 모두 `_browser_cdp_check`로 게이팅됩니다(CDP 엔드포인트에 연결할 수 있을 때만 감독자를 실행할 수 있음). Camofox 또는 백엔드가 없는 세션에서는 대화 상자 도구가 숨겨지고 스냅샷에서 새 필드가 생략됩니다 — 스키마가 비대해지지 않습니다.

## 출처가 다른 iframe 상호작용

`browser_cdp(frame_id=...)`는 감독자가 이미 연결한 WebSocket을 통해 CDP 호출(특히 `Runtime.evaluate`)을 라우팅하며, OOPIF의 자식 `sessionId`를 사용합니다. 에이전트는 `browser_snapshot.frame_tree.children[]`에서 `is_oopif=true`인 frame_id를 선택해 `browser_cdp`에 전달합니다. 동일 출처 iframe(전용 CDP 세션이 없음)의 경우 에이전트는 최상위 `Runtime.evaluate`에서 `contentWindow`/`contentDocument`를 대신 사용합니다. `frame_id`가 OOPIF가 아닌 프레임에 속하면 감독자는 이 대체 경로를 안내하는 오류를 표시합니다.

Browserbase에서는 이것이 iframe 상호작용을 위한 유일하게 신뢰할 수 있는 경로입니다 — `browser_cdp` 호출마다 여는 무상태 CDP 연결은 서명된 URL 만료를 겪지만, 감독자의 장기 연결은 유효한 세션을 유지합니다.

## 파일 구성

- `tools/browser_supervisor.py` — `CDPSupervisor`, `SupervisorRegistry`, `PendingDialog`, `FrameInfo`
- `tools/browser_dialog_tool.py` — `browser_dialog` 도구 핸들러
- `tools/browser_tool.py` — `browser_navigate` 시작 훅, `browser_snapshot` 병합, `/browser connect` 재연결, `_cleanup_browser_session` 해제
- `toolsets.py` — `browser`, `hermes-acp`, `hermes-api-server`, 핵심 도구 세트에 `browser_dialog` 등록(CDP 연결 가능 여부에 따라 게이팅)
- `hermes_cli/config.py` — `browser.dialog_policy` 및 `browser.dialog_timeout_s` 기본값

## 목표에 포함되지 않는 사항

- Camofox의 감지/상호작용(상위 프로젝트의 공백이며 별도로 추적)
- 대화 상자/프레임 이벤트를 사용자에게 실시간 스트리밍(게이트웨이 훅이 필요)
- 세션 간 대화 상자 기록 유지(메모리 내에서만 유지)
- iframe별 대화 상자 정책(`dialog_id`로 에이전트가 표현 가능)
- `browser_cdp` 대체 — 장기적인 예외 상황(쿠키, 뷰포트, 네트워크 스로틀링)을 위한 탈출구로 계속 유지

## 테스트

단위 테스트(`tests/tools/test_browser_supervisor.py`)는 모든 상태 전이를 실행할 수 있을 만큼의 프로토콜을 구사하는 asyncio 모의 CDP 서버를 사용합니다.
연결, 활성화, 이동, 대화 상자 발생, 대화 상자 닫기, 프레임 연결/해제, 자식 대상 연결, 세션 해제를 테스트합니다. 실제 백엔드 E2E(Browserbase + 로컬 Chromium 계열 브라우저)는 수동으로 수행합니다 — `/browser connect`를 통해 실행 중인 Chromium 계열 브라우저에 연결하고 위에 설명한 대화 상자/프레임 테스트 사례를 실행하세요.
