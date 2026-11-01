---
sidebar_position: 17
title: "세션 하트비트"
description: "세션이 유휴 상태일 때마다 현재 세션에 다시 진입하는 반복 프롬프트 — /heartbeat every 10m Check the deployment."
---

# 세션 하트비트(`/heartbeat`)

`/heartbeat`는 **현재 세션**에 하나의 반복 지시를 부여합니다. 세션이 유휴 상태이고 간격이 지나면 프롬프트가 일반 사용자 턴으로 실행됩니다. 동일한 대화, 동일한 컨텍스트, 동일한 프롬프트 캐시를 사용합니다.

```
/heartbeat every 10m Check the deployment and report meaningful changes
```

Prime-Agent의 `/heartbeat`에서 영감을 받았습니다. Hermes의 구현은 엄격한 메시지 흐름 불변 조건을 유지합니다. 하트비트는 턴 사이에만 삽입되며(실행 중간에는 삽입되지 않음), 일반 사용자 역할 메시지로 처리됩니다.

## 하트비트와 cron: 어느 것을 사용해야 할까요?

둘은 비슷해 보이지만 서로 다른 작업을 수행합니다.

| | `/heartbeat` | [`hermes cron`](./cron) |
|---|---|---|
| 실행 위치 | **이 대화** — 전체 컨텍스트와 대화의 기억 사용 | 틱마다 새로 생성되는 격리된 세션 |
| 프로세스 재시작 후 유지 | 상태가 유지됨(SessionDB); 다음에 세션이 구동될 때 실행 재개 | 예 — 완전히 영속적인 스케줄러 |
| 개수 | 세션당 하나 | 작업 수 제한 없음 |
| 적합한 용도 | "작업하는 동안 *이 스레드에서* X를 지켜봐" | 상시 작업, 보고서, 감시, 전달 |

간단한 기준은 다음과 같습니다. 반복 프롬프트에 대화의 컨텍스트가 필요하면 `/heartbeat`를 사용하세요. 독립적인 작업이라면 cron을 사용하세요.

## 명령

| 명령 | 동작 |
|---|---|
| `/heartbeat every <interval> <prompt>` | 세션의 하트비트를 설정(또는 교체)합니다. 간격: `90s`, `10m`, `2h`, `1d`(최소 60초). |
| `/heartbeat` 또는 `/heartbeat status` | 하트비트, 간격, 다음 실행까지의 시간을 표시합니다. |
| `/heartbeat pause` | 삭제하지 않고 실행을 중지합니다. |
| `/heartbeat resume` | 재개합니다(타이머를 다시 기준점에 맞추므로 오래된 실행이 즉시 발생하지 않음). |
| `/heartbeat clear` | 하트비트를 제거합니다. |

`/hb`는 별칭입니다. CLI와 게이트웨이 플랫폼에서 작동합니다(Slack에서는 `/hermes heartbeat …` 사용).

## 동작 세부 사항

- **유휴 상태에서만 실행됩니다.** 하트비트는 실행 중인 턴을 절대 중단하지 않습니다. 틱이 도래했을 때 에이전트가 바쁘면 다음 유휴 폴링 때 실행됩니다.
- **놓친 틱은 합쳐집니다.** 여러 간격 동안 세션이 바빴거나 프로세스가 실행 중이 아니었다면 하트비트 턴을 여러 개가 아닌 **하나**만 받습니다. 타이머는 실행될 때마다 기준점을 다시 잡습니다.
- **사용자 메시지가 우선입니다.** 대기 중인 사용자 메시지가 있으면 항상 우선 처리되며, 하트비트는 입력 큐가 비워질 때까지 기다립니다.
- **캐시 안전성을 지킵니다.** 삽입되는 프롬프트는 일반 사용자 메시지입니다. 시스템 프롬프트 변경이나 도구 세트 변경이 없습니다.
- **영속성.** 상태는 `heartbeat:<session_id>`를 키로 사용하는 `SessionDB.state_meta`에 저장됩니다. `/resume` 후에도 유지되며 컨텍스트 압축에 따른 세션 교체를 거쳐 전달됩니다. 실행하려면 소유 프로세스(CLI 세션 또는 게이트웨이)가 실행 중이어야 합니다. 어떤 상황에서도 유지되어야 하는 일정에는 cron을 사용하세요.
- **작업을 지어내지 않는 보호 장치.** 삽입되는 프롬프트는 의미 있는 변경이 없을 때 에이전트가 짧게 답하고 중지하도록 지시합니다. 따라서 유휴 하트비트가 쓸데없는 작업을 만들어 내지 않습니다.

## 예시

```
You: /heartbeat every 15m Check whether the CI run for PR #1234 finished; summarize the result when it does

  ♥ Heartbeat set (every 15m): Check whether the CI run for PR #1234 finished; ...

[15 minutes of you working on other things in the same session]

Hermes: [Heartbeat — recurring instruction, fires every 15m]
  💻 gh pr checks 1234   (1.2s)
  CI is still running (14/37 checks complete). Nothing to report yet.
```

결과가 더 이상 변하지 않으면 `/heartbeat clear`로 삭제하거나 계속 지켜보도록 두세요.
