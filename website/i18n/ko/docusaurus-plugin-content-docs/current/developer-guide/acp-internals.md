---
sidebar_position: 2
title: "ACP 내부 구조"
description: "ACP 어댑터의 작동 방식: 수명 주기, 세션, 이벤트 브리지, 승인 및 도구 렌더링"
---

# ACP 내부 구조

ACP 어댑터는 Hermes의 동기식 `AIAgent`를 비동기 JSON-RPC stdio 서버로 감쌉니다.

주요 구현 파일:

- `acp_adapter/entry.py`
- `acp_adapter/server.py`
- `acp_adapter/session.py`
- `acp_adapter/events.py`
- `acp_adapter/permissions.py`
- `acp_adapter/tools.py`
- `acp_adapter/auth.py`

## 부팅 흐름

```text
hermes acp / hermes-acp / python -m acp_adapter
  -> acp_adapter.entry.main()
  -> parse --version / --check / --setup before server startup
  -> load ~/.hermes/.env
  -> configure stderr logging
  -> construct HermesACPAgent
  -> acp.run_agent(agent, use_unstable_protocol=True)
```

stdout는 ACP JSON-RPC 전송 전용입니다. 사람이 읽는 로그는 stderr로 전송됩니다.

## 주요 구성 요소

### `HermesACPAgent`

`acp_adapter/server.py`는 ACP 에이전트 프로토콜을 구현합니다.

책임:

- 초기화 / 인증
- 세션 생성/로드/재개/분기/목록/취소 메서드
- 프롬프트 실행
- 세션 모델 전환
- 동기식 AIAgent 콜백을 ACP 비동기 알림에 연결

### `SessionManager`

`acp_adapter/session.py`는 활성 ACP 세션을 추적합니다.

각 세션에는 다음이 저장됩니다.

- `session_id`
- `agent`
- `cwd`
- `model`
- `history`
- `cancel_event`

관리자는 스레드 안전하며 다음 작업을 지원합니다.

- 생성
- 가져오기
- 제거
- 분기
- 목록
- 정리
- cwd 업데이트

### 이벤트 브리지

`acp_adapter/events.py`는 AIAgent 콜백을 ACP `session_update` 이벤트로 변환합니다.

연결되는 콜백:

- `tool_progress_callback`
- `thinking_callback` (현재 ACP 브리지에서는 `None`으로 설정됨 — 추론은 대신 `step_callback`을 통해 전달됨)
- `step_callback`

AIAgent는 작업자 스레드에서 실행되고 ACP I/O는 메인 이벤트 루프에서 실행되므로 브리지는 다음을 사용합니다.

```python
asyncio.run_coroutine_threadsafe(...)
```

### 권한 브리지

`acp_adapter/permissions.py`는 위험한 터미널 승인 프롬프트를 ACP 권한 요청에 맞게 변환합니다.

매핑:

- `allow_once` -> Hermes `once`
- `allow_always` -> Hermes `always`
- 거부 옵션 -> Hermes `deny`

시간 초과와 브리지 오류가 발생하면 기본적으로 거부합니다.

### 도구 렌더링 헬퍼

`acp_adapter/tools.py`는 Hermes 도구를 ACP 도구 종류에 매핑하고 편집기용 콘텐츠를 구성합니다.

예:

- `patch` / `write_file` -> 파일 diff
- `terminal` -> 셸 명령 텍스트
- `read_file` / `search_files` -> 텍스트 미리보기
- 큰 결과 -> UI 안전성을 위한 잘린 텍스트 블록

## 세션 수명 주기

```text
new_session(cwd)
  -> create SessionState
  -> create AIAgent(platform="acp", enabled_toolsets=["hermes-acp"])
  -> bind task_id/session_id to cwd override

prompt(..., session_id)
  -> extract text from ACP content blocks
  -> reset cancel event
  -> install callbacks + approval bridge
  -> run AIAgent in ThreadPoolExecutor
  -> update session history
  -> emit final agent message chunk
```

### 취소

`cancel(session_id)`:

- 세션 취소 이벤트를 설정합니다.
- 사용 가능한 경우 `agent.interrupt()`를 호출합니다.
- 프롬프트 응답이 `stop_reason="cancelled"`를 반환하게 합니다.

### 분기

`fork_session()`은 메시지 기록을 새 활성 세션에 깊은 복사하여, 대화 상태는 유지하면서 분기 세션에 별도의 세션 ID와 cwd를 제공합니다.

## 제공업체/인증 동작

ACP는 자체 인증 저장소를 구현하지 않습니다.

대신 Hermes의 런타임 확인자를 재사용합니다.

- `acp_adapter/auth.py`
- `hermes_cli/runtime_provider.py`

따라서 ACP는 현재 구성된 Hermes 제공업체/자격 증명을 알리고 사용합니다. 또한 항상 터미널 설정 인증 방식(`hermes-setup`, 인수 `--setup`)을 알리므로, 최초 실행 ACP 클라이언트가 일반 ACP 세션을 시작하기 전에 Hermes의 대화형 모델/제공업체 설정을 열 수 있습니다.

## 작업 디렉터리 연결

ACP 세션은 편집기의 cwd를 전달합니다.

세션 관리자는 작업별 터미널/파일 재정의를 통해 해당 cwd를 ACP 세션 ID에 연결하므로, 파일 및 터미널 도구가 편집기 작업 공간을 기준으로 작동합니다.

## 동일한 이름의 도구 호출 중복

이벤트 브리지는 도구 이름마다 하나의 ID만 저장하는 대신 도구 ID를 FIFO 방식으로 추적합니다. 이는 다음에 중요합니다.

- 동일한 이름의 병렬 호출
- 한 단계에서 반복되는 동일한 이름의 호출

FIFO 큐가 없으면 완료 이벤트가 잘못된 도구 호출에 연결됩니다.

## 승인 콜백 복원

ACP는 프롬프트 실행 중 터미널 도구에 승인 콜백을 임시로 설치한 다음, 이후 이전 콜백을 복원합니다. 이를 통해 ACP 세션별 승인 핸들러가 전역에 영구적으로 설치된 상태로 남는 것을 방지합니다.

## 현재 제한 사항

- ACP 세션은 공유 `~/.hermes/state.db`(SessionDB)에 저장되며 프로세스 재시작 후 투명하게 복원됩니다. `session_search`에 표시됩니다.
- 텍스트가 아닌 프롬프트 블록은 현재 요청 텍스트 추출에서 무시됩니다.
- 편집기별 UX는 ACP 클라이언트 구현에 따라 다릅니다.

## 관련 파일

- `tests/acp/` — ACP 테스트 모음
- `toolsets.py` — `hermes-acp` 도구 세트 정의
- `hermes_cli/main.py` — `hermes acp` CLI 하위 명령
- `pyproject.toml` — `[acp]` 선택적 의존성 + `hermes-acp` 스크립트
