---
sidebar_position: 11
title: "Cron 내부 구조"
description: "Hermes가 cron 작업을 저장, 예약, 편집, 일시 중지하고 스킬을 로드하며 전달하는 방법"
---

# Cron 내부 구조

cron 서브시스템은 단순한 일회성 지연부터 스킬 주입 및 플랫폼 간 전달을 지원하는 반복 cron 표현식 작업까지, 예약된 작업 실행을 제공합니다.

## 주요 파일

| 파일 | 용도 |
|------|---------|
| `cron/jobs.py` | 작업 모델, 저장소, `jobs.json`에 대한 원자적 읽기/쓰기 |
| `cron/scheduler.py` | 스케줄러 루프 — 기한이 된 작업 감지, 실행, 반복 추적 |
| `tools/cronjob_tools.py` | 모델용 `cronjob` 도구 등록 및 핸들러 |
| `gateway/run.py` | 게이트웨이 통합 — 장시간 실행 루프에서 cron 틱 처리 |
| `hermes_cli/cron.py` | CLI `hermes cron` 하위 명령 |

## 예약 모델

네 가지 예약 형식을 지원합니다.

| 형식 | 예시 | 동작 |
|--------|---------|----------|
| **상대 지연** | `30m`, `2h`, `1d` | 일회성, 지정된 기간 후 실행 |
| **간격** | `every 2h`, `every 30m` | 반복, 일정한 간격으로 실행 |
| **Cron 표현식** | `0 9 * * *` | 표준 5개 필드 cron 문법(분, 시, 일, 월, 요일) |
| **ISO 타임스탬프** | `2025-01-15T09:00:00` | 일회성, 정확한 시각에 실행 |

모델용 표면은 작업 스타일 연산을 사용하는 단일 `cronjob` 도구입니다: `create`, `list`, `update`, `pause`, `resume`, `run`, `remove`.

## 작업 저장

작업은 원자적 쓰기 의미론(임시 파일에 쓴 다음 이름 변경)을 사용하여 `~/.hermes/cron/jobs.json`에 저장됩니다. 각 작업 레코드는 다음을 포함합니다.

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

### 작업 수명 주기 상태

| 상태 | 의미 |
|-------|---------|
| `scheduled` | 활성 상태이며 다음 예약 시각에 실행됨 |
| `paused` | 일시 중지됨 — 재개할 때까지 실행되지 않음 |
| `completed` | 반복 횟수를 모두 소진했거나 실행된 일회성 작업 |
| `running` | 현재 실행 중(일시적 상태) |

### 이전 버전 호환성

이전 작업에는 `skills` 배열 대신 단일 `skill` 필드가 있을 수 있습니다. 스케줄러는 로드 시 이를 정규화합니다 — 단일 `skill`을 `skills: [skill]`로 승격합니다.

## 스케줄러 런타임

### 틱 주기

스케줄러는 주기적인 틱(기본값: 60초마다)으로 실행됩니다.

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

### 게이트웨이 통합

게이트웨이 모드에서 cron **트리거**(기한이 된 작업을 언제 실행할지 결정하는 부분 — "Axis B")는 플러그형 `CronScheduler` 공급자를 통해 선택됩니다. 게이트웨이는 `resolve_cron_scheduler()`(`cron/scheduler_provider.py`)를 호출하고, 별도의 게이트웨이 하우스키핑 스레드와 함께 확인된 공급자의 `start()`를 전용 백그라운드 스레드에서 실행합니다.

활성 공급자는 `cron.provider` config 키로 선택됩니다.

- **비어 있음(기본값)** → 내장 `InProcessCronScheduler`가 60초마다 `scheduler.tick()`을 호출하는 기존 인프로세스 루프를 실행합니다. 이는 공급자 도입 전 동작과 바이트 단위로 동일합니다.
- **이름이 지정된 공급자**(예: 스케일 투 제로 배포를 위한 관리형 cron 공급자 `chronos`) → `plugins/cron_providers/<name>/` 또는 `$HERMES_HOME/plugins/<name>/`에서 검색됩니다.

이름이 지정된 공급자가 없거나, 로드에 실패하거나, `is_available() == False`를 보고하면 확인자는 경고와 함께 내장 공급자로 대체합니다 — **cron에 트리거가 없는 상태는 절대 발생하지 않습니다.** 내장 공급자는 `plugins/`가 아니라 코어(`cron/scheduler_provider.py`)에 있으므로 대체 동작이 실수로 제거될 수 없습니다.

작업을 "실행"한다는 의미(작업 실행 + 전달)는 변경되지 않으며 모든 공급자가 공유합니다 — `scheduler.run_job()` / `scheduler._deliver_result()`에 그대로 유지됩니다. 공급자는 트리거만 제어하고 실행은 제어하지 않습니다.

CLI 모드에서는 `hermes cron` 명령을 실행하거나 활성 CLI 세션 중일 때만 cron 작업이 실행됩니다.

### 스케일 투 제로를 위한 관리형 cron (Chronos)

호스팅된 게이트웨이는 내장 틱커 대신 **Chronos** 공급자(`cron.provider: chronos`)를 실행할 수 있습니다. Chronos를 사용하면 유휴 게이트웨이가 **스케일 투 제로** 상태가 되어도 cron 작업을 실행할 수 있습니다. 프로세스를 깨어 있게 만드는 60초 인프로세스 루프 대신, Nous 인프라에 각 작업의 실제 다음 실행 시각에 정확히 **하나의 관리형 일회성 작업**을 예약하도록 요청합니다. 실행 시각이 되면 Nous가 인증된 웹훅(`POST /api/cron/fire`)으로 게이트웨이를 호출합니다. 게이트웨이는 내장 기능과 동일한 `run_one_job` 경로로 작업을 실행한 다음 다음 일회성 작업을 다시 예약합니다. 실행 사이에는 프로세스가 완전히 중지될 수 있으며, 주기적 타이머가 아니라 실제 실행 시각에만 깨어납니다.

흐름(관리형 스케줄러는 Nous가 제공하며, 에이전트는 스케줄러 자격 증명을 보유하지 않음)은 다음과 같습니다.

```
create/update a cron job
  → Chronos asks Nous to arm a one-shot at the job's next_run_at
      (authenticated with the agent's existing Nous token)
  → at fire time Nous calls the gateway: POST {callback_url}/api/cron/fire
      (authenticated with a short-lived, purpose-scoped Nous-minted JWT)
  → the gateway verifies the token, claims the job (store compare-and-set so
    multi-replica deployments fire at-most-once), runs it, and re-arms the next
    one-shot
```

구성(모두 비밀이 아닌 값이며, 호스팅된 에이전트에서는 Nous가 프로비저닝 시 설정함):

| key | 의미 |
|---|---|
| `cron.provider` | 활성화하려면 `chronos`(비어 있으면 내장 틱커) |
| `cron.chronos.portal_url` | Nous 기본 URL(예약 + 실행 토큰 발급자) |
| `cron.chronos.callback_url` | 인바운드 실행을 위한 게이트웨이 자체의 공개 기본 URL |
| `cron.chronos.expected_audience` | 이 에이전트의 실행 토큰 대상 |
| `cron.chronos.nas_jwks_url` | 인바운드 실행 토큰 검증을 위한 키 집합 |

Chronos가 잘못 구성되었거나 에이전트가 Nous에 로그인하지 않은 경우 `resolve_cron_scheduler()`는 내장 틱커로 대체됩니다(경고가 기록됨) — cron은 트리거를 잃지 않습니다. 반복 작업은 실행할 때마다 다시 예약되며, `repeat`-N 작업은 횟수를 모두 소진하면 깔끔하게 중지됩니다(고아 일회성 작업 없음). 전체 에이전트↔Nous 와이어 계약은 `docs/chronos-managed-cron-contract.md`에 있습니다.

### 새 세션 격리

각 cron 작업은 완전히 새로운 에이전트 세션에서 실행됩니다.

- 이전 실행의 대화 기록 없음
- 이전 cron 실행의 메모리 없음(메모리/파일에 저장된 경우 제외)
- 프롬프트는 자체적으로 완결되어야 함 — cron 작업은 명확화 질문을 할 수 없음
- `cronjob` 도구 세트는 비활성화됨(재귀 방지)

## 스킬 기반 작업

cron 작업은 `skills` 필드를 통해 하나 이상의 스킬을 연결할 수 있습니다. 실행 시:

1. 지정된 순서대로 스킬이 로드됩니다.
2. 각 스킬의 SKILL.md 내용이 컨텍스트로 주입됩니다.
3. 작업의 프롬프트가 작업 지침으로 추가됩니다.
4. 에이전트가 결합된 스킬 컨텍스트와 프롬프트를 처리합니다.

이를 통해 cron 프롬프트에 전체 지침을 붙여 넣지 않고도 재사용 가능하고 테스트된 워크플로를 사용할 수 있습니다. 예를 들면 다음과 같습니다.

```
Create a daily funding report → attach "ai-funding-daily-report" skill
```

### 스크립트 기반 작업

작업은 `script` 필드를 통해 Python 스크립트를 연결할 수도 있습니다. 스크립트는 각 에이전트 턴 **전에** 실행되며, 표준 출력이 컨텍스트로 프롬프트에 주입됩니다. 이를 통해 데이터 수집 및 변경 감지 패턴을 사용할 수 있습니다.

```python
# ~/.hermes/scripts/check_competitors.py
import requests, json
# Fetch competitor release notes, diff against last run
# Print summary to stdout — agent analyzes and reports
```

스크립트 시간 초과의 기본값은 3600초(1시간)입니다. `_get_script_timeout()`은 다음 세 계층의 체인을 통해 제한을 확인합니다.

1. **모듈 수준 재정의** — `_SCRIPT_TIMEOUT`(테스트/monkeypatching용). 기본값과 다를 때만 사용됩니다.
2. **환경 변수** — `HERMES_CRON_SCRIPT_TIMEOUT`
3. **구성** — `config.yaml`의 `cron.script_timeout_seconds`(`load_config()`를 통해 읽음)
4. **기본값** — 3600초(1시간)

이 시간 초과는 에이전트가 아니라 **사전 실행 스크립트만** 제한합니다. 스킬 기반/LLM 기반 작업은 별도의 *비활성* 기반 예산(`HERMES_CRON_TIMEOUT`, 유휴 시간 기본값 600초, `0` = 무제한)으로 실행됩니다. 도구를 계속 호출하거나 토큰을 스트리밍하는 동안에는 몇 시간 동안 실행될 수 있으며, 활동 없이 설정된 유휴 시간이 지난 경우에만 종료됩니다. 스크립트는 영구 스레드 풀에 전달됩니다(틱 잠금 아래에서 실행되지 않음). 따라서 오래 실행되는 스크립트가 다른 기한 도래 작업의 실행을 막지 않습니다.

### 공급자 복구

`run_job()`은 사용자가 구성한 대체 공급자와 자격 증명 풀을 `AIAgent` 인스턴스에 전달합니다.

- **대체 공급자** — `config.yaml`에서 `fallback_providers`(목록) 또는 `fallback_model`(이전 버전의 딕셔너리)을 읽으며, 게이트웨이의 `_load_fallback_model()` 패턴과 일치합니다. `AIAgent.__init__`에 `fallback_model=`로 전달되며, 두 형식을 대체 체인으로 정규화합니다.
- **자격 증명 풀** — 확인된 런타임 공급자 이름을 사용하여 `agent.credential_pool`의 `load_pool(provider)`로 로드됩니다. 풀에 자격 증명이 있을 때(`pool.has_credentials()`)만 전달됩니다. 이를 통해 429/속도 제한 오류 발생 시 동일 공급자 키를 순환할 수 있습니다.

이는 게이트웨이의 동작을 반영합니다 — 이 기능이 없으면 cron 에이전트는 복구를 시도하지 않고 속도 제한으로 실패합니다.

## 전달 모델

cron 작업 결과는 지원되는 모든 플랫폼으로 전달할 수 있습니다.

플랫폼 이름만 입력하면(`slack`, `telegram`, …) 해당 플랫폼에 구성된 **홈 채널**로 전달됩니다. 대신 **특정** 목적지를 지정하려면 콜론 뒤에 대상을 추가합니다: `platform:<target>`. 대상은 작업이 생성될 때가 아니라 실행 시점에 확인되므로, 아직 연결되지 않은 플랫폼의 목적지를 작업에 지정해 두었다가 해당 플랫폼이 온라인 상태가 되면 전달을 시작할 수 있습니다.

대부분의 플랫폼은 세 번째 세그먼트로 선택적 스레드/토픽도 받습니다: `platform:<chat_id>:<thread_id>`.

| 대상 | 문법 | 예시 |
|--------|--------|---------|
| 시작 채팅 | `origin` | 작업이 생성된 채팅으로 전달 |
| 로컬 파일 | `local` | `~/.hermes/cron/output/`에 저장 |
| Telegram | `telegram`, `telegram:<chat_id>`, `telegram:<chat_id>:<thread_id>`, `telegram:@username` | `telegram:-1001234567890:17585` |
| Discord | `discord`, `discord:#channel`, `discord:<channel_id>`, `discord:<channel_id>:<thread_id>` | `discord:#engineering` |
| Slack | `slack`, `slack:#channel`, `slack:<channel_id>`, `slack:<channel_id>:<thread_ts>` | `slack:#engineering` |
| Matrix | `matrix`, `matrix:<!room_id:server>`, `matrix:<@user:server>` | `matrix:!abc123:example.org` |
| Feishu | `feishu`, `feishu:<chat_id>`, `feishu:<chat_id>:<thread_id>` | `feishu:oc_abc123def` |
| WhatsApp | `whatsapp`, `whatsapp:<jid>`, `whatsapp:+<E.164>` | `whatsapp:123456@g.us` |
| Signal | `signal`, `signal:group:<id>`, `signal:+<E.164>` | `signal:group:aBcD==` |
| SMS | `sms`, `sms:+<E.164>` | `sms:+<E.164 number>` |
| Email | `email`, `email:<address>` | `email:alerts@example.com` |
| Weixin | `weixin`, `weixin:<wxid>` | `weixin:wxid_abc123` |
| Mattermost | `mattermost` 또는 `mattermost:<channel_id>` | 이름만 입력하면 Mattermost 홈으로 전달 |
| Home Assistant | `homeassistant` 또는 `homeassistant:<conversation>` | 이름만 입력하면 HA 대화로 전달 |
| DingTalk | `dingtalk` 또는 `dingtalk:<chat_id>` | 이름만 입력하면 DingTalk로 전달 |
| WeCom | `wecom` 또는 `wecom:<chat_id>` | 이름만 입력하면 WeCom으로 전달 |
| BlueBubbles | `bluebubbles` 또는 `bluebubbles:<chat_guid>` | 이름만 입력하면 BlueBubbles를 통해 iMessage로 전달 |
| QQ Bot | `qqbot` 또는 `qqbot:<chat_id>` | 이름만 입력하면 공식 API v2를 통해 QQ(Tencent)로 전달 |

첫 번째 그룹의 플랫폼은 이름이 지정된 채널(`#channel`), 토픽/스레드, 방/사용자 ID, 그룹 ID 또는 전화번호처럼 명시적이고 검증된 대상 문법을 사용합니다. 나머지 플랫폼은 일반적인 `platform:<chat_id>` 형식을 사용합니다(콜론 뒤의 값이 목적지 ID로 그대로 사용됨). 플랫폼 이름만 입력하면 항상 홈 채널로 전달됩니다.

**이름이 지정된 채널**(`slack:#engineering`, `discord:#engineering` 또는 `slack:engineering`과 같은 친숙한 이름)은 게이트웨이가 연결된 어댑터에서 구성한 채널 디렉터리를 기준으로 확인됩니다. 따라서 이름 확인이 성공하려면 게이트웨이가 해당 채널을 검색한 상태여야 합니다. 원시 ID(`slack:C0123ABCD45`)는 항상 작동합니다.

**Telegram 토픽**에는 `telegram:<chat_id>:<thread_id>`(예: `telegram:-1001234567890:17585`)를 사용합니다. **Slack 스레드**에서는 세 번째 세그먼트가 상위 메시지의 `thread_ts`(예: `slack:C0123ABCD45:1700000000.000100`)이며, 기존 메시지 아래에 답장할 때만 적용됩니다.

### 응답 래핑

기본값(`cron.wrap_response: true`)에 따라 cron 전달에는 다음이 래핑됩니다.

- cron 작업 이름과 작업을 식별하는 헤더
- 에이전트가 전달된 메시지를 대화에서 볼 수 없음을 알리는 푸터

cron 응답의 `[SILENT]` 접두사는 전달을 완전히 억제합니다 — 파일에 쓰거나 부수 효과만 필요한 작업에 유용합니다.

### 세션 격리

cron 전달은 게이트웨이 세션 대화 기록에 미러링되지 **않습니다**. 전달은 cron 작업 자체의 세션에만 존재합니다. 이를 통해 대상 채팅의 대화에서 메시지 교대 규칙 위반을 방지합니다.

## 재귀 방지

cron 실행 세션에서는 `cronjob` 도구 세트가 비활성화됩니다. 이를 통해 다음을 방지합니다.

- 예약된 작업이 새 cron 작업을 생성하는 것
- 토큰 사용량을 폭증시킬 수 있는 재귀적 예약
- 작업 내부에서 작업 일정을 실수로 변경하는 것

## 잠금

스케줄러는 프로세스 간 파일 기반 잠금(Unix에서는 `fcntl.flock`, Windows에서는 `msvcrt.locking`)을 사용하여 게이트웨이의 인프로세스 틱커와 독립 실행형 `hermes cron` / 수동 `tick()` 호출 사이에서도 겹치는 틱이 동일한 기한 도래 작업 묶음을 두 번 실행하지 않도록 합니다. 잠금을 획득할 수 없으면 `tick()`은 즉시 0을 반환합니다.

## CLI 인터페이스

`hermes cron` CLI는 직접적인 작업 관리를 제공합니다.

```bash
hermes cron list                    # Show all jobs
hermes cron create                  # Interactive job creation (alias: add)
hermes cron edit <job_id>           # Edit job configuration
hermes cron pause <job_id>          # Pause a running job
hermes cron resume <job_id>         # Resume a paused job
hermes cron run <job_id>            # Trigger immediate execution
hermes cron remove <job_id>         # Delete a job
```

## 관련 문서

- [Cron 기능 가이드](/user-guide/features/cron)
- [게이트웨이 내부 구조](./gateway-internals.md)
- [에이전트 루프 내부 구조](./agent-loop.md)
