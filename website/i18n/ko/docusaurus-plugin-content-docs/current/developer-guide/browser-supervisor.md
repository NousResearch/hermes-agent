---
sidebar_position: 18
title: "브라우저 CDP 감독관 (Browser CDP Supervisor)"
description: "Hermes가 기본 JS 대화 상자를 감지 및 응답하고 영구적인 CDP 연결을 통해 교차 출처(cross-origin) iframe과 상호작용하는 방법입니다."
---

# 브라우저 CDP 감독관 (Browser CDP Supervisor)

CDP 감독관은 Hermes의 브라우저 도구에 있던 두 가지 오랜 간극을 해결합니다:

1. **기본 JS 대화 상자**(`alert`/`confirm`/`prompt`/`beforeunload`)는 페이지의 JS 스레드를 차단합니다. 감독 없이는 에이전트가 대화 상자가 열려 있는지 알 방법이 없으며, 후속 도구 호출이 중단되거나 불투명한 오류가 발생합니다.
2. **교차 출처 iframe(OOPIF, Out-of-Process iframes)** 은 최상위 `Runtime.evaluate`에 보이지 않습니다. 에이전트는 DOM 스냅샷에서 iframe 노드를 볼 수는 있지만, 하위 대상에 CDP 세션이 연결되지 않으면 그 안에서 클릭, 입력 또는 평가(eval)를 할 수 없습니다.

감독관은 브라우저 작업마다 백엔드의 CDP 엔드포인트에 대한 영구적인 WebSocket을 유지하고, 대기 중인 대화 상자와 프레임 구조를 `browser_snapshot`에 노출하며, 명시적인 응답을 위한 `browser_dialog` 도구를 제공하여 이 두 가지를 모두 해결합니다.

## 백엔드 지원

| 백엔드 | 대화 상자 감지 | 대화 상자 응답 | 프레임 트리 | `browser_cdp(frame_id=...)`를 통한 OOPIF `Runtime.evaluate` |
|---|---|---|---|---|
| Local Chrome (`--remote-debugging-port`) / `/browser connect` | ✓ | ✓ 전체 워크플로 | ✓ | ✓ |
| Browserbase | ✓ (브리지 사용) | ✓ 전체 워크플로 (브리지 사용) | ✓ | ✓ |
| Camofox | ✗ CDP 없음 (REST 전용) | ✗ | DOM 스냅샷을 통한 부분 지원 | ✗ |

**Browserbase의 특이점.** Browserbase의 CDP 프록시는 내부적으로 Playwright를 사용하며 약 10ms 이내에 기본 대화 상자를 자동 닫기(auto-dismiss) 하므로, `Page.handleJavaScriptDialog`가 따라잡을 수 없습니다. 감독관은 `Page.addScriptToEvaluateOnNewDocument`를 통해 브리지 스크립트를 주입하여 `window.alert`/`confirm`/`prompt`를 특정 호스트(`hermes-dialog-bridge.invalid`)에 대한 동기식 XHR로 재정의합니다. `Fetch.enable`은 이러한 XHR이 네트워크에 닿기 전에 차단합니다. 즉, 대화 상자는 감독관이 캡처하는 `Fetch.requestPaused` 이벤트가 되고, `respond_to_dialog`는 주입된 스크립트가 디코딩하는 JSON 본문과 함께 `Fetch.fulfillRequest`를 통해 요청을 이행합니다.

페이지 관점에서는 `prompt()`가 여전히 에이전트가 제공한 문자열을 반환합니다. 에이전트 관점에서는 어떤 쪽이든 동일한 `browser_dialog(action=...)` API입니다.

Camofox는 지원되지 않습니다 — CDP 표면이 없으며 REST 전용입니다.

## 아키텍처 (Architecture)

### CDPSupervisor

Hermes `task_id`당 백그라운드 데몬 스레드에서 실행되는 하나의 `asyncio.Task`입니다. 백엔드의 CDP 엔드포인트에 대한 영구적인 WebSocket을 유지합니다. 다음을 유지합니다:

- **대화 상자 큐 (Dialog queue)** — `{id, type, message, default_prompt, session_id, opened_at}`를 포함하는 `List[PendingDialog]`
- **프레임 트리 (Frame tree)** — 부모 관계, URL, 출처(origin), 교차 출처 하위 세션 여부를 포함하는 `Dict[frame_id, FrameInfo]`
- **세션 맵 (Session map)** — 상호작용 도구가 OOPIF 작업을 위해 연결된 올바른 세션으로 라우팅할 수 있도록 하는 `Dict[session_id, SessionInfo]`
- **최근 콘솔 오류** — 진단을 위한 최근 50개의 링 버퍼

연결(attach) 시 구독 항목:

- `Page.enable` — `javascriptDialogOpening`, `frameAttached`, `frameNavigated`, `frameDetached`
- `Runtime.enable` — `executionContextCreated`, `consoleAPICalled`, `exceptionThrown`
- `Target.setAutoAttach {autoAttach: true, flatten: true}` — 하위 OOPIF 대상을 노출합니다; 감독관은 각각에 대해 `Page`+`Runtime`을 활성화합니다.

스레드로부터 안전한 상태 접근은 스냅샷 잠금을 통해 이루어지며, 도구 핸들러(동기식)는 대기(await) 없이 고정된 스냅샷을 읽습니다.

### 라이프사이클 (Lifecycle)

- **시작:** `SupervisorRegistry.get_or_start(task_id, cdp_url)` — `browser_navigate`, Browserbase 세션 생성, `/browser connect`에 의해 호출됩니다. 멱등성(Idempotent)을 가집니다.
- **중지:** 세션 해제(teardown) 또는 `/browser disconnect`. asyncio 작업을 취소하고, WebSocket을 닫고, 상태를 폐기합니다.
- **재결합(Rebind):** CDP URL이 변경되는 경우 (사용자가 새 Chrome에 다시 연결), 이전 감독관이 중지되고 새 감독관이 시작됩니다 — 상태는 엔드포인트 간에 재사용되지 않습니다.

### 대화 상자 정책 (Dialog policy)

`config.yaml`의 `browser.dialog_policy` 항목을 통해 구성 가능합니다:

- **`must_respond`** (기본값) — 캡처하고, `browser_snapshot`에 노출하며, 명시적인 `browser_dialog(action=...)` 호출을 기다립니다. 응답 없이 300초의 안전 제한 시간이 지나면 자동 닫기(auto-dismiss)하고 기록합니다. 버그가 있는 에이전트가 영원히 멈춰 있는 것을 방지합니다.
- `auto_dismiss` — 기록하고 즉시 닫습니다. 에이전트는 `browser_snapshot` 내부의 `browser_state`를 통해 사후에 이를 봅니다.
- `auto_accept` — 기록하고 수락합니다 (`beforeunload` 시 워크플로가 깨끗하게 다른 곳으로 이동하길 원할 때 유용함).

정책은 태스크별로 적용되며, 개별 대화 상자에 대한 재정의는 불가능합니다.

## 에이전트 인터페이스 (Agent surface)

### `browser_dialog` 도구

```
browser_dialog(action, prompt_text=None, dialog_id=None)
```

- `action="accept"` / `"dismiss"` → 지정된 또는 유일하게 대기 중인 대화 상자에 응답합니다. (필수)
- `prompt_text=...` → `prompt()` 대화 상자에 제공할 텍스트입니다.
- `dialog_id=...` → 큐에 여러 대화 상자가 있을 때 모호성을 해결합니다. (드묾)

응답 전용(response-only) 도구입니다. 에이전트는 호출하기 전에 `browser_snapshot` 출력에서 대기 중인 대화 상자를 읽습니다.

### `browser_snapshot` 확장

감독관이 연결되면 기존 스냅샷 출력에 세 가지 선택적 필드를 추가합니다:

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

- **`pending_dialogs`** — 현재 페이지의 JS 스레드를 차단하고 있는 대화 상자들입니다. 에이전트는 응답하기 위해 반드시 `browser_dialog(action=...)`를 호출해야 합니다. Browserbase에서는 프록시가 ~10ms 내에 자동 닫기를 하므로 비어 있습니다.

- **`recent_dialogs`** — `closed_by` 태그가 포함된 최대 20개의 최근에 닫힌 대화 상자 링 버퍼입니다. 태그: `"agent"` (우리가 응답함), `"auto_policy"` (로컬에서 auto_dismiss/auto_accept 적용됨), `"watchdog"` (must_respond 타임아웃 발생), 또는 `"remote"` (브라우저/백엔드가 우리 몰래 닫음, 예: Browserbase). 이것이 Browserbase에 있는 에이전트가 어떤 일이 일어났는지 파악하는 방법입니다.

- **`frame_tree`** — 교차 출처 (OOPIF) 하위 항목을 포함한 프레임 구조입니다. 광고가 많은 페이지에서 스냅샷 크기를 제한하기 위해 30개 항목 + OOPIF 깊이 2로 제한됩니다. 제한에 도달하면 `truncated: true`가 표시됩니다. 전체 트리가 필요한 에이전트는 `Page.getFrameTree`와 함께 `browser_cdp`를 사용할 수 있습니다.

이 중 어느 것도 새로운 도구 스키마 표면을 필요로 하지 않습니다 — 에이전트는 이미 요청한 스냅샷을 읽기만 하면 됩니다.

### 가용성 게이트 (Availability gating)

두 표면 모두 `_browser_cdp_check`를 기반으로 활성화됩니다 (감독관은 CDP 엔드포인트에 도달할 수 있을 때만 실행될 수 있음). Camofox / 백엔드가 없는 세션에서는 대화 상자 도구가 숨겨지고 스냅샷에서 새 필드가 생략됩니다 — 스키마 비대화(bloat)가 없습니다.

## 교차 출처 iframe 상호작용 (Cross-origin iframe interaction)

`browser_cdp(frame_id=...)`는 OOPIF의 하위 `sessionId`를 사용하여 이미 연결된 감독관의 WebSocket을 통해 CDP 호출(특히 `Runtime.evaluate`)을 라우팅합니다. 에이전트는 `browser_snapshot.frame_tree.children[]`에서 `is_oopif=true`인 `frame_id`를 골라내어 `browser_cdp`로 전달합니다. 동일 출처(same-origin) iframe(전용 CDP 세션이 없음)의 경우, 에이전트는 대신 최상위 `Runtime.evaluate`에서 `contentWindow`/`contentDocument`를 사용합니다 — 감독관은 `frame_id`가 OOPIF가 아닌 경우 해당 대체 방법(fallback)을 가리키는 오류를 표출합니다.

Browserbase에서는 이것이 iframe 상호작용을 위한 유일하게 신뢰할 수 있는 경로입니다. (`browser_cdp` 호출마다 열리는) 무상태(stateless) CDP 연결은 서명된 URL 만료에 부딪히지만, 감독관의 수명이 긴 연결은 유효한 세션을 유지합니다.

## 파일 배치 (File layout)

- `tools/browser_supervisor.py` — `CDPSupervisor`, `SupervisorRegistry`, `PendingDialog`, `FrameInfo`
- `tools/browser_dialog_tool.py` — `browser_dialog` 도구 핸들러
- `tools/browser_tool.py` — `browser_navigate` 시작 훅(start-hook), `browser_snapshot` 병합(merge), `/browser connect` 재결합(reattach), `_cleanup_browser_session` 해제(teardown)
- `toolsets.py` — `browser`, `hermes-acp`, `hermes-api-server` 및 코어 도구 세트에 `browser_dialog` 등록 (CDP 도달 가능성에 따라 제한됨)
- `hermes_cli/config.py` — `browser.dialog_policy` 및 `browser.dialog_timeout_s` 기본값

## 목표가 아닌 것 (Non-goals)

- Camofox에 대한 감지/상호작용 (업스트림 차이; 별도로 추적됨)
- 사용자에게 실시간으로 대화 상자/프레임 이벤트 스트리밍 (게이트웨이 훅 필요)
- 세션 간 대화 상자 기록 유지 (메모리 내에만 저장됨)
- iframe별 대화 상자 정책 (에이전트가 `dialog_id`를 통해 표현할 수 있음)
- `browser_cdp` 교체 — 롱테일 작업(쿠키, 뷰포트, 네트워크 스로틀링)을 위한 탈출구(escape hatch)로 유지됨

## 테스트 (Testing)

단위 테스트 (`tests/tools/test_browser_supervisor.py`)는 연결(attach), 활성화(enable), 탐색(navigate), 대화 상자 발생(dialog fire), 대화 상자 닫기(dialog dismiss), 프레임 연결/해제(frame attach/detach), 하위 대상 연결(child target attach), 세션 해제(session teardown) 등 모든 상태 전환을 테스트하기에 충분한 프로토콜을 통신하는 asyncio 모의(mock) CDP 서버를 사용합니다. 실제 백엔드 E2E (Browserbase + 로컬 Chromium 계열 브라우저)는 수동으로 이루어집니다 — 활성 Chromium 계열 브라우저에 `/browser connect`를 통해 실행하고 위에 설명된 대화 상자/프레임 테스트 케이스를 실행합니다.
