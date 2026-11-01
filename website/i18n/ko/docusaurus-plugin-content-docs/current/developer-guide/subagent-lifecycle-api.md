---
title: 공개 서브에이전트 수명 주기 API
sidebar_label: 서브에이전트 수명 주기 API
---

# 공개 서브에이전트 수명 주기 API

플러그인은 `tools.delegate_tool`, 게이트웨이 내부 구현, TUI 상태 또는
`AIAgent` 필드를 가져오지 않고도 새로운 Hermes 자식 세션을 시작하고
감독할 수 있습니다. 이 서비스는 현재 에이전트 턴에서 부모를 확인하므로
CLI, 게이트웨이, 비대화형 세션 및 kanban-worker 세션에서 모두 작동합니다.
활성 에이전트 턴 외부에서 시작하면 `No active Hermes parent session`과 함께
안전하게 실패합니다.

```python
from agent.subagent_lifecycle import SubagentLaunchRequest

def launch_review(ctx):
    # Call from a plugin tool or hook while an agent turn is active.
    service = ctx.subagent_lifecycle
    handle = service.launch(SubagentLaunchRequest(
        goal="Review this change for regressions.",
        context="Only inspect the supplied repository.",
        role="leaf",
        correlation_id="review-42",
        allowed_toolsets=("file",),
    ))
    # Persist handle.to_dict() if desired.
    if service.wait(handle, timeout_seconds=2).timed_out:
        return handle.to_dict()
    return service.result(handle)
```

`SubagentHandle`은 직렬화할 수 있으며 버전이 지정된 불투명 capability를
담고 있습니다. 이를 `status`, `wait`, `cancel`, `result` 또는 `reconnect`에
전달할 수 있습니다. 잘못되었거나 위조된 핸들은 `UNKNOWN`/`UNKNOWN_HANDLE`을
반환하며 자식에 접근할 수 없습니다.

안정적으로 사용되는 상태는 `PENDING`, `STARTING`, `RUNNING`, `SUCCEEDED`,
`FAILED`, `INTERRUPTED`, `CANCEL_REQUESTED`, `CANCELLED` 및 `UNKNOWN`입니다.

`cancel(handle, reason=...)`은 협력적 방식으로 동작합니다. 자식 에이전트에
다음 안전 경계에서 중단하도록 요청하고 `CANCEL_REQUESTED`를 반환합니다.
`wait` 또는 `result`가 터미널 상태를 확인하기 전에는 완료를 주장하지
않습니다. 터미널 결과는 변경되지 않고 멱등적이며 32k자로 제한되고,
트랜스크립트와 숨겨진 추론은 제외하며, 안정적인 결과 해시를 포함합니다.

이 API는 수명 주기로 관리되는 비동기 실행을 제공합니다. 자식 생성과 완료는
`delegate_task`와 동일한 호스트 소유 경로를 사용합니다. 여기에는 부모 도구
확인 복원, 메모리 알림, 직렬화된 `subagent_stop` 훅, 리소스 정리 및 자식
비용 집계가 포함됩니다. 동기식 `delegate_task` 도구, 일괄 위임 또는 해당
게이트웨이/TUI 표시 방식은 변경하지 않습니다. 최초 구현에서는 메타데이터와
터미널 결과를 프로세스 내에서 한 시간 동안 보존합니다.
프로세스가 재시작된 후 `reconnect`는 `RECONNECT_UNAVAILABLE`을 반환하며
대체 자식을 절대로 시작하지 않습니다. 실행 중인 Python 스레드도 프로세스
종료 후에는 유지될 수 없으므로, 호출자는 해당 핸들을 프로세스 종료로 인해
중단된 것으로 처리해야 합니다.

요청은 안전하게 실패합니다. goal/context/metadata 크기에는 상한이 적용되고,
알 수 없거나 부모의 범위를 넓히는 도구 세트는 거부됩니다. 또한 Hermes가
격리를 약화시키지 않고 지원할 수 있을 때까지 도구별 차단, 작업 디렉터리
재정의 및 실행별 타임아웃은 명시적으로 거부됩니다. `allowed_toolsets`를
사용해 자식의 범위를 좁힐 수 있으며, Hermes의 기존 위험 도구 차단은 계속
적용됩니다.
