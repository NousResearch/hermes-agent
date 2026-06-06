---
sidebar_position: 2
title: "ACP Internals"
description: "ACP 어댑터 작동 방식: 라이프사이클, 세션, 이벤트 브리지, 승인 및 도구 렌더링"
---

# ACP 내부

ACP 어댑터는 비동기 JSON-RPC stdio 서버에 Hermes의 동기식 `AIAgent`를 래핑합니다.

주요 구현 파일:

- `acp_adapter/entry.py`
- `acp_adapter/server.py`
- `acp_adapter/session.py`
- `acp_adapter/events.py`
- `acp_adapter/permissions.py`
- `acp_adapter/tools.py`
- `acp_adapter/auth.py`
- `acp_registry/agent.json`

## 부팅 흐름 (Boot flow)

```text
hermes acp / hermes-acp / python -m acp_adapter
  -> acp_adapter.entry.main()
  -> 서버 시작 전 --version / --check / --setup 분석
  -> ~/.hermes/.env 로드
  -> stderr 로깅 구성
  -> HermesACPAgent 구성
  -> acp.run_agent(agent, use_unstable_protocol=True)
```

Zed ACP 레지스트리 경로는 `uvx --from 'hermes-agent[acp]==<version>' hermes-acp`를 통해 `hermes-agent` PyPI 릴리스를 가리키는 동일한 어댑터를 시작합니다.

Stdout은 ACP JSON-RPC 전송을 위해 예약되어 있습니다. 사람이 읽을 수 있는 로그는 stderr로 이동합니다.

## 주요 구성 요소

### `HermesACPAgent`

`acp_adapter/server.py`는 ACP 에이전트 프로토콜을 구현합니다.

역할:

- 초기화 / 인증
- 세션 메서드: new/load/resume/fork/list/cancel
- 프롬프트 실행
- 세션 모델 전환
- 동기 AIAgent 콜백을 비동기 ACP 알림으로 연결

### `SessionManager`

`acp_adapter/session.py`는 실시간 ACP 세션을 추적합니다.

각 세션은 다음을 저장합니다:

- `session_id`
- `agent`
- `cwd`
- `model`
- `history`
- `cancel_event`

관리자는 스레드 안전(thread-safe)하며 다음을 지원합니다:

- create (생성)
- get (가져오기)
- remove (제거)
- fork (포크)
- list (목록)
- cleanup (정리)
- cwd 업데이트

### 이벤트 브리지 (Event bridge)

`acp_adapter/events.py`는 AIAgent 콜백을 ACP `session_update` 이벤트로 변환합니다.

연결된 콜백:

- `tool_progress_callback`
- `thinking_callback` (현재 ACP 브리지에서는 `None`으로 설정됨 — 추론은 대신 `step_callback`을 통해 전달됨)
- `step_callback`

`AIAgent`는 작업자 스레드에서 실행되고 ACP I/O는 메인 이벤트 루프에 상주하므로, 브리지는 다음을 사용합니다:

```python
asyncio.run_coroutine_threadsafe(...)
```

### 권한 브리지 (Permission bridge)

`acp_adapter/permissions.py`는 위험한 터미널 승인 프롬프트를 ACP 권한 요청에 맞게 조정합니다.

매핑:

- `allow_once` -> Hermes `once`
- `allow_always` -> Hermes `always`
- reject 옵션 -> Hermes `deny`

시간 초과 및 브리지 실패 시 기본적으로 거부(deny)됩니다.

### 도구 렌더링 헬퍼 (Tool rendering helpers)

`acp_adapter/tools.py`는 Hermes 도구를 ACP 도구 종류에 매핑하고 에디터용 콘텐츠를 작성합니다.

예:

- `patch` / `write_file` -> 파일 diff
- `terminal` -> 셸 명령어 텍스트
- `read_file` / `search_files` -> 텍스트 미리보기
- 큰 결과 -> UI 안전을 위한 잘린(truncated) 텍스트 블록

## 세션 라이프사이클

```text
new_session(cwd)
  -> SessionState 생성
  -> AIAgent 생성(platform="acp", enabled_toolsets=["hermes-acp"])
  -> cwd 재정의에 task_id/session_id 바인딩

prompt(..., session_id)
  -> ACP 콘텐츠 블록에서 텍스트 추출
  -> cancel 이벤트 초기화
  -> 콜백 + 승인 브리지 설치
  -> ThreadPoolExecutor에서 AIAgent 실행
  -> 세션 기록 업데이트
  -> 최종 에이전트 메시지 청크(chunk) 발신
```

### 취소 (Cancelation)

`cancel(session_id)`:

- 세션 취소(cancel) 이벤트를 설정합니다
- 가능할 때 `agent.interrupt()`를 호출합니다
- 프롬프트 응답이 `stop_reason="cancelled"`를 반환하도록 합니다

### 포크 (Forking)

`fork_session()`은 메시지 기록을 새로운 라이브 세션으로 깊은 복사(deep-copy)하여, 포크에 자체 세션 ID와 cwd를 제공하는 동시에 대화 상태를 보존합니다.

## 프로바이더/인증 동작

ACP는 자체 인증 저장소를 구현하지 않습니다.

대신 Hermes의 런타임 리졸버를 재사용합니다:

- `acp_adapter/auth.py`
- `hermes_cli/runtime_provider.py`

따라서 ACP는 현재 구성된 Hermes 프로바이더/자격 증명을 광고하고 사용합니다. 또한 첫 실행 레지스트리 클라이언트가 일반 ACP 세션을 시작하기 전에 Hermes의 대화형 모델/프로바이더 구성을 열 수 있도록 터미널 설정 인증 방식(`hermes-setup`, 인자 `--setup`)을 항상 광고합니다.

## 작업 디렉토리 바인딩

ACP 세션은 에디터의 cwd(현재 작업 디렉토리)를 전달합니다.

세션 관리자는 작업(task) 범위의 터미널/파일 재정의를 통해 해당 cwd를 ACP 세션 ID에 바인딩하므로, 파일 및 터미널 도구가 에디터 작업 공간을 기준으로 작동합니다.

## 동일한 이름의 도구 호출 중복

이벤트 브리지는 이름당 하나의 ID가 아닌, 도구 이름별로 FIFO 방식으로 도구 ID를 추적합니다. 이는 다음 사항에 중요합니다:

- 병렬로 발생한 동일한 이름 호출
- 한 단계에서 반복된 동일한 이름 호출

FIFO 큐가 없으면 완료 이벤트가 잘못된 도구 호출에 연결될 수 있습니다.

## 승인 콜백 복원

ACP는 프롬프트 실행 중에 터미널 도구에 임시로 승인 콜백을 설치한 다음, 완료 후 이전 콜백을 복원합니다. 이렇게 하면 ACP 세션 전용 승인 핸들러가 전역적으로 영원히 설치된 상태로 남는 것을 방지할 수 있습니다.

## 현재 한계

- ACP 세션은 공유된 `~/.hermes/state.db` (SessionDB)에 유지되며 프로세스 재시작 간에 투명하게 복원됩니다. 이는 `session_search`에 나타납니다.
- 비텍스트 프롬프트 블록은 현재 요청 텍스트 추출 시 무시됩니다.
- 에디터별 UX는 ACP 클라이언트 구현에 따라 다릅니다.

## 관련 파일

- `tests/acp/` — ACP 테스트 스위트
- `toolsets.py` — `hermes-acp` 도구 세트 정의
- `hermes_cli/main.py` — `hermes acp` CLI 하위 명령
- `pyproject.toml` — `[acp]` 선택적 종속성 + `hermes-acp` 스크립트
