---
sidebar_position: 11
title: "크론 내부 (Cron Internals)"
description: "Hermes가 크론 작업을 저장, 예약, 편집, 일시 중지, 스킬 로드 및 전달하는 방법"
---

# 크론 내부 (Cron Internals)

크론 하위 시스템은 단순한 1회성 지연부터 스킬 주입 및 크로스 플랫폼 전달 기능이 포함된 반복적인 크론 표현식 작업까지 예약된 작업 실행을 제공합니다.

## 주요 파일 (Key Files)

| 파일 | 목적 |
|------|---------|
| `cron/jobs.py` | 작업 모델, 저장소, `jobs.json`에 대한 원자적(atomic) 읽기/쓰기 |
| `cron/scheduler.py` | 스케줄러 루프 — 기한이 된 작업 감지, 실행, 반복 추적 |
| `tools/cronjob_tools.py` | 모델을 향한 `cronjob` 도구 등록 및 핸들러 |
| `gateway/run.py` | 게이트웨이 통합 — 장기 실행 루프에서의 크론 틱킹(cron ticking) |
| `hermes_cli/cron.py` | CLI `hermes cron` 하위 명령어 |

## 예약 모델 (Scheduling Model)

네 가지 예약 형식을 지원합니다:

| 형식 | 예시 | 동작 |
|--------|---------|----------|
| **상대적 지연 (Relative delay)** | `30m`, `2h`, `1d` | 지정된 기간 이후에 실행되는 1회성 작업 |
| **간격 (Interval)** | `every 2h`, `every 30m` | 규칙적인 간격으로 실행되는 반복 작업 |
| **크론 표현식 (Cron expression)** | `0 9 * * *` | 표준 5-필드 크론 구문 (분, 시, 일, 월, 요일) |
| **ISO 타임스탬프** | `2025-01-15T09:00:00` | 정확한 시간에 실행되는 1회성 작업 |

모델이 바라보는 표면(Model-facing surface)은 `create`, `list`, `update`, `pause`, `resume`, `run`, `remove`와 같은 액션 스타일 작업을 갖춘 단일 `cronjob` 도구입니다.

## 작업 저장소 (Job Storage)

작업은 원자적 쓰기 시맨틱(임시 파일에 쓰고 이름을 바꾸는 방식)을 사용하여 `~/.hermes/cron/jobs.json`에 저장됩니다. 각 작업 레코드는 다음을 포함합니다:

```json
{
  "id": "a1b2c3d4e5f6",
  "name": "Daily briefing",
  "prompt": "Summarize today's AI news and funding rounds",
  "schedule": {
    "kind": "cron",
    "expr": "0 9 * * *",
    "display": "0 9 * * *"
  },
  "skills": ["ai-funding-daily-report"],
  "deliver": "telegram:-1001234567890",
  "repeat": {
    "times": null,
    "completed": 42
  },
  "state": "scheduled",
  "enabled": true,
  "next_run_at": "2025-01-16T09:00:00Z",
  "last_run_at": "2025-01-15T09:00:00Z",
  "last_status": "ok",
  "created_at": "2025-01-01T00:00:00Z",
  "model": null,
  "provider": null,
  "script": null
}
```

### 작업 수명 주기 상태 (Job Lifecycle States)

| 상태 | 의미 |
|-------|---------|
| `scheduled` | 활성 상태, 다음 예약된 시간에 실행됨 |
| `paused` | 일시 중지됨 — 재개될 때까지 실행되지 않음 |
| `completed` | 반복 횟수가 모두 소진되었거나 실행이 완료된 1회성 작업 |
| `running` | 현재 실행 중 (일시적인 상태) |

### 하위 호환성 (Backward Compatibility)

오래된 작업은 `skills` 배열 대신 단일 `skill` 필드를 가질 수 있습니다. 스케줄러는 로드 시 이를 정규화합니다 — 단일 `skill`은 `skills: [skill]`로 승격됩니다.

## 스케줄러 런타임 (Scheduler Runtime)

### 틱 사이클 (Tick Cycle)

스케줄러는 주기적인 틱 단위(기본값: 60초마다)로 실행됩니다:

```text
tick()
  1. Acquire scheduler lock (prevents overlapping ticks)
  2. Load all jobs from jobs.json
  3. Filter to due jobs (next_run <= now AND state == "scheduled")
  4. For each due job:
     a. Set state to "running"
     b. Create fresh AIAgent session (no conversation history)
     c. Load attached skills in order (injected as user messages)
     d. Run the job prompt through the agent
     e. Deliver the response to the configured target
     f. Update run_count, compute next_run
     g. If repeat count exhausted → state = "completed"
     h. Otherwise → state = "scheduled"
  5. Write updated jobs back to jobs.json
  6. Release scheduler lock
```

### 게이트웨이 통합 (Gateway Integration)

게이트웨이 모드에서 스케줄러는 메시지 처리와 나란히 60초마다 `scheduler.tick()`을 호출하는 전용 백그라운드 스레드(`gateway/run.py`의 `_start_cron_ticker`)에서 실행됩니다.

CLI 모드에서 크론 작업은 `hermes cron` 명령어가 실행될 때나 활성 CLI 세션 중에만 실행됩니다.

### 새로운 세션 격리 (Fresh Session Isolation)

각 크론 작업은 완전히 새로운 에이전트 세션에서 실행됩니다:

- 이전 실행의 대화 기록 없음
- 이전 크론 실행에 대한 기억 없음 (메모리나 파일에 유지된 경우는 예외)
- 프롬프트는 자기 완결적이어야 함 — 크론 작업은 명확화를 위한 질문(clarifying questions)을 할 수 없습니다.
- `cronjob` 도구 모음은 비활성화됨 (재귀 가드)

## 스킬 기반 작업 (Skill-Backed Jobs)

크론 작업은 `skills` 필드를 통해 하나 이상의 스킬을 연결할 수 있습니다. 실행 시점에:

1. 스킬은 지정된 순서대로 로드됩니다.
2. 각 스킬의 SKILL.md 콘텐츠가 컨텍스트로 주입됩니다.
3. 작업의 프롬프트가 태스크 지시사항(task instruction)으로 추가됩니다.
4. 에이전트는 결합된 스킬 컨텍스트 + 프롬프트를 처리합니다.

이를 통해 크론 프롬프트에 전체 지시사항을 붙여넣지 않고도 재사용 가능하고 검증된 워크플로우를 사용할 수 있습니다. 예시:

```
매일 펀딩 보고서 만들기 → "ai-funding-daily-report" 스킬 연결
```

### 스크립트 기반 작업 (Script-Backed Jobs)

작업은 `script` 필드를 통해 파이썬 스크립트를 연결할 수도 있습니다. 스크립트는 각 에이전트 턴(turn) *이전에* 실행되며, 스크립트의 표준 출력(stdout)은 컨텍스트로서 프롬프트에 주입됩니다. 이를 통해 데이터 수집 및 변경 감지 패턴을 구현할 수 있습니다:

```python
# ~/.hermes/scripts/check_competitors.py
import requests, json
# 경쟁사 릴리스 노트를 가져오고 이전 실행과 비교(diff)
# 요약 내용을 stdout으로 출력 — 에이전트가 이를 분석하고 보고함
```

스크립트 시간 초과의 기본값은 120초입니다. `_get_script_timeout()`은 다음 세 가지 계층 체인을 통해 제한 시간을 확인합니다:

1. **모듈 수준 재정의 (Module-level override)** — `_SCRIPT_TIMEOUT` (테스트/몽키패칭용). 기본값과 다를 때만 사용됩니다.
2. **환경 변수 (Environment variable)** — `HERMES_CRON_SCRIPT_TIMEOUT`
3. **설정 (Config)** — `config.yaml`의 `cron.script_timeout_seconds` (`load_config()`를 통해 읽음)
4. **기본값 (Default)** — 120초

### 제공자 복구 (Provider Recovery)

`run_job()`은 사용자가 구성한 대체 제공자(fallback providers)와 자격 증명 풀(credential pool)을 `AIAgent` 인스턴스에 전달합니다:

- **대체 제공자 (Fallback providers)** — 게이트웨이의 `_load_fallback_model()` 패턴과 일치하게 `config.yaml`에서 `fallback_providers` (리스트) 또는 `fallback_model` (기존의 딕셔너리)을 읽습니다. 이는 `AIAgent.__init__`에 `fallback_model=`로 전달되며, 여기에서 두 형식을 모두 대체 체인(fallback chain)으로 정규화합니다.
- **자격 증명 풀 (Credential pool)** — 확인된 런타임 제공자 이름을 사용하여 `agent.credential_pool`에서 `load_pool(provider)`를 통해 로드합니다. 풀에 자격 증명이 있을 때만(`pool.has_credentials()`) 전달됩니다. 429/속도 제한 오류가 발생했을 때 동일한 제공자의 키 교체(key rotation)를 가능하게 합니다.

이는 게이트웨이의 동작을 반영한 것입니다 — 이것이 없다면 크론 에이전트는 복구를 시도하지 않고 속도 제한 오류에 실패할 것입니다.

## 전달 모델 (Delivery Model)

크론 작업의 결과는 지원되는 모든 플랫폼으로 전달할 수 있습니다:

| 대상 | 구문 | 예시 |
|--------|--------|---------|
| 원본 채팅 (Origin chat) | `origin` | 작업이 생성된 채팅으로 전달 |
| 로컬 파일 (Local file) | `local` | `~/.hermes/cron/output/` 에 저장 |
| Telegram | `telegram` 또는 `telegram:<chat_id>` | `telegram:-1001234567890` |
| Discord | `discord` 또는 `discord:#channel` | `discord:#engineering` |
| Slack | `slack` | Slack 기본 채널로 전달 |
| WhatsApp | `whatsapp` | WhatsApp 기본 채팅으로 전달 |
| Signal | `signal` | Signal로 전달 |
| Matrix | `matrix` | Matrix 기본 룸으로 전달 |
| Mattermost | `mattermost` | Mattermost 기본 채팅으로 전달 |
| Email | `email` | 이메일로 전달 |
| SMS | `sms` | SMS로 전달 |
| Home Assistant | `homeassistant` | HA 대화로 전달 |
| DingTalk | `dingtalk` | DingTalk으로 전달 |
| Feishu | `feishu` | Feishu로 전달 |
| WeCom | `wecom` | WeCom으로 전달 |
| Weixin | `weixin` | Weixin (WeChat)으로 전달 |
| BlueBubbles | `bluebubbles` | BlueBubbles를 통해 iMessage로 전달 |
| QQ Bot | `qqbot` | 공식 API v2를 통해 QQ (Tencent)로 전달 |

Telegram 토픽의 경우 `telegram:<chat_id>:<thread_id>` 형식을 사용하세요 (예: `telegram:-1001234567890:17585`).

### 응답 래핑 (Response Wrapping)

기본적으로 (`cron.wrap_response: true`), 크론 전달은 다음 사항과 함께 래핑됩니다:
- 크론 작업 이름과 태스크를 식별하는 헤더
- 에이전트가 대화 내에서 전달된 메시지를 볼 수 없다는 내용의 푸터

크론 응답의 `[SILENT]` 접두사는 전달을 완전히 억제합니다 — 파일에 기록하거나 부수 효과(side effects)를 수행하기만 하면 되는 작업에 유용합니다.

### 세션 격리 (Session Isolation)

크론 전달 결과는 게이트웨이 세션의 대화 기록에 미러링되지 않습니다. 결과는 크론 작업 자체 세션에만 존재합니다. 이는 대상 채팅 대화에서 메시지 교대(message alternation) 위반을 방지하기 위함입니다.

## 재귀 가드 (Recursion Guard)

크론으로 실행되는 세션은 `cronjob` 도구 모음이 비활성화되어 있습니다. 이는 다음을 방지합니다:
- 예약된 작업이 새로운 크론 작업을 생성하는 것
- 토큰 사용량을 폭발시킬 수 있는 재귀적 예약
- 작업 내에서 실수로 작업 일정을 변경하는 것

## 잠금 (Locking)

스케줄러는 프로세스 간 파일 기반 잠금(Unix의 경우 `fcntl.flock`, Windows의 경우 `msvcrt.locking`)을 사용하여 동일한 기한 작업 배치를 중복 틱이 두 번 실행하는 것을 방지합니다 — 심지어 게이트웨이의 인프로세스 틱커와 독립 실행형 `hermes cron` / 수동 `tick()` 호출 사이에서도 방지합니다. 잠금을 획득할 수 없는 경우 `tick()`은 즉시 0을 반환합니다.

## CLI 인터페이스 (CLI Interface)

`hermes cron` CLI는 직접적인 작업 관리를 제공합니다:

```bash
hermes cron list                    # 모든 작업 표시
hermes cron create                  # 인터랙티브 작업 생성 (별칭: add)
hermes cron edit <job_id>           # 작업 설정 편집
hermes cron pause <job_id>          # 실행 중인 작업 일시 중지
hermes cron resume <job_id>         # 일시 중지된 작업 재개
hermes cron run <job_id>            # 즉시 실행 트리거
hermes cron remove <job_id>         # 작업 삭제
```

## 관련 문서 (Related Docs)

- [크론 기능 가이드](/user-guide/features/cron)
- [게이트웨이 내부](./gateway-internals.md)
- [에이전트 루프 내부](./agent-loop.md)
