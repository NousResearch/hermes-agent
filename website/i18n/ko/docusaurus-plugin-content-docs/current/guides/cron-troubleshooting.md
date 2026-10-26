---
sidebar_position: 12
title: "Cron 문제 해결"
description: "일정이 실행되지 않거나, 전달이 실패하거나, 스킬 로딩에 오류가 발생하거나, 성능 문제가 생기는 등 일반적인 Hermes cron 문제를 진단하고 해결합니다"
---

# Cron 문제 해결

cron 작업이 예상대로 동작하지 않을 때는 다음 점검을 순서대로 진행하세요. 대부분의 문제는 타이밍, 전달, 권한 또는 스킬 로딩이라는 네 가지 범주에 속합니다.

---

## 작업이 실행되지 않음

### 점검 1: 작업이 존재하고 활성 상태인지 확인

```bash
hermes cron list
```

작업을 찾아 상태가 `[active]`인지(`[paused]` 또는 `[completed]`가 아닌지) 확인하세요. `[completed]`로 표시된다면 반복 횟수를 모두 사용했을 수 있으므로 작업을 편집해 초기화하세요.

### 점검 2: 일정이 올바른지 확인

형식이 잘못된 일정은 조용히 일회성 일정으로 기본 설정되거나 완전히 거부됩니다. 표현식을 테스트하세요.

| 일정 표현식 | 다음과 같이 평가되어야 함 |
|----------------|-------------------|
| `0 9 * * *` | 매일 오전 9:00 |
| `0 9 * * 1` | 매주 월요일 오전 9:00 |
| `every 2h` | 지금부터 2시간마다 |
| `30m` | 지금부터 30분 후 |
| `2025-06-01T09:00:00` | 2025년 6월 1일 오전 9:00 UTC |

작업이 한 번 실행된 후 목록에서 사라진다면 일회성 일정(`30m`, `1d` 또는 ISO 타임스탬프)인 것이므로 정상적인 동작입니다.

### 점검 3: 게이트웨이가 실행 중인지 확인

cron 작업은 60초마다 실행되는 게이트웨이의 백그라운드 ticker 스레드에 의해 실행됩니다. 일반 CLI 채팅 세션은 cron 작업을 자동으로 실행하지 않습니다.

작업이 자동으로 실행되기를 기대한다면 실행 중인 게이트웨이가 필요합니다(포그라운드에서는 `hermes gateway`, 설치된 서비스에서는 `hermes gateway start`). 일회성 디버깅을 위해서는 `hermes cron tick`으로 tick을 수동 실행할 수 있습니다.

### 점검 4: 시스템 시계와 시간대 확인

작업은 로컬 시간대를 사용합니다. 컴퓨터의 시계가 잘못되었거나 예상한 시간대와 다르면 작업이 잘못된 시간에 실행됩니다. 다음을 확인하세요.

```bash
date
hermes cron list   # Compare next_run times with local time
```

---

## 전달 실패

### 점검 1: deliver 대상이 올바른지 확인

전달 대상은 대소문자를 구분하며 올바른 플랫폼이 구성되어 있어야 합니다. 잘못 구성된 대상은 응답을 조용히 삭제합니다.

| 대상 | 필요 항목 |
|--------|----------|
| `telegram` | `~/.hermes/.env`의 `TELEGRAM_BOT_TOKEN` |
| `discord` | `~/.hermes/.env`의 `DISCORD_BOT_TOKEN` |
| `slack` | `~/.hermes/.env`의 `SLACK_BOT_TOKEN` |
| `whatsapp` | WhatsApp 게이트웨이 구성 |
| `signal` | Signal 게이트웨이 구성 |
| `matrix` | Matrix homeserver 구성 |
| `email` | `config.yaml`의 SMTP 구성 |
| `sms` | SMS provider 구성 |
| `local` | `~/.hermes/cron/output/`에 대한 쓰기 권한 |
| `origin` | 작업이 생성된 채팅으로 전달 |

그 밖에 지원되는 플랫폼으로는 `mattermost`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`, `webhook`이 있습니다. `platform:chat_id` 구문을 사용해 특정 채팅을 대상으로 지정할 수도 있습니다(예: `telegram:-1001234567890`).

전달에 실패해도 작업 자체는 실행됩니다. 단지 아무 곳에도 전송되지 않을 뿐입니다. `hermes cron list`에서 업데이트된 `last_error` 필드(있는 경우)를 확인하세요.

### 점검 2: `[SILENT]` 사용 확인

cron 작업이 출력을 생성하지 않으면 전달이 억제됩니다. 에이전트 응답에 cron 무음 마커 `[SILENT]`가 포함되어 있어도 전달이 억제됩니다. 이는 모니터링 작업을 위한 의도된 동작이지만, 프롬프트가 실수로 모든 출력을 억제하고 있지는 않은지 확인하세요.

다음과 같은 프롬프트를 사용하세요. "변경된 내용이 없으면 `[SILENT]`만 응답하세요." 에이전트에게 더 긴 설명 안에 `[SILENT]`를 포함하도록 요청하지 마세요. cron은 해당 마커를 억제 신호로 처리하기 때문입니다.

### 점검 3: 플랫폼 토큰 권한

각 메시징 플랫폼 봇은 메시지를 받기 위해 특정 권한이 필요합니다. 전달이 조용히 실패한다면 다음을 확인하세요.

- **Telegram**: 봇이 대상 그룹/채널의 관리자여야 함
- **Discord**: 봇에 대상 채널에서 전송할 권한이 있어야 함
- **Slack**: 봇이 워크스페이스에 추가되어 있고 `chat:write` 범위가 있어야 함

### 점검 4: 응답 래핑

기본적으로 cron 응답은 헤더와 푸터로 래핑됩니다(`config.yaml`의 `cron.wrap_response: true`). 일부 플랫폼이나 통합 기능은 이를 제대로 처리하지 못할 수 있습니다. 비활성화하려면 다음과 같이 하세요.

```yaml
cron:
  wrap_response: false
```

---

## 스킬 로딩 실패

### 점검 1: 스킬이 설치되어 있는지 확인

```bash
hermes skills list
```

스킬은 cron 작업에 연결하기 전에 설치되어 있어야 합니다. 스킬이 없다면 먼저 `hermes skills install <skill-name>`으로 설치하거나 CLI에서 `/skills`를 사용하세요.

### 점검 2: 스킬 이름과 스킬 폴더 이름 확인

스킬 이름은 대소문자를 구분하며 설치된 스킬의 폴더 이름과 일치해야 합니다. 작업에서 `ai-funding-report`를 지정했는데 스킬 폴더가 `ai-funding-daily-report`라면 `hermes skills list`에서 정확한 이름을 확인하세요.

### 점검 3: 대화형 도구가 필요한 스킬

cron 작업은 `cronjob`, `messaging`, `clarify` 도구 세트를 비활성화한 상태로 실행됩니다. 이렇게 하면 재귀적인 cron 생성, 직접 메시지 전송(전달은 스케줄러가 처리), 대화형 프롬프트가 방지됩니다. 스킬이 이러한 도구 세트에 의존한다면 cron 컨텍스트에서 작동하지 않습니다.

스킬 문서를 확인해 비대화형(헤드리스) 모드에서 작동하는지 확인하세요.

### 점검 4: 여러 스킬의 순서

여러 스킬을 사용할 때는 스킬이 순서대로 로드됩니다. 스킬 A가 스킬 B의 컨텍스트에 의존한다면 B를 먼저 로드하세요.

```bash
/cron add "0 9 * * *" "..." --skill context-skill --skill target-skill
```

이 예시에서는 `context-skill`이 `target-skill`보다 먼저 로드됩니다.

---

## 작업 오류 및 실패

### 점검 1: 최근 작업 출력 검토

작업이 실행되었지만 실패했다면 다음 위치에서 오류 컨텍스트를 확인할 수 있습니다.

1. 작업이 전달되는 채팅(전달에 성공한 경우)
2. 스케줄러 메시지가 기록되는 `~/.hermes/logs/agent.log` (경고는 `errors.log`)
3. `hermes cron list`를 통해 확인하는 작업의 `last_run` 메타데이터

### 점검 2: 일반적인 오류 패턴

**스크립트에서 발생한 "No such file or directory"**
`script` 경로는 절대 경로(또는 Hermes 구성 디렉터리를 기준으로 한 상대 경로)여야 합니다. 다음을 확인하세요.
```bash
ls ~/.hermes/scripts/your-script.py   # Must exist
hermes cron edit <job_id> --script ~/.hermes/scripts/your-script.py
```

**작업 실행 시 "Skill not found"**
스케줄러를 실행하는 컴퓨터에 스킬이 설치되어 있어야 합니다. 컴퓨터를 바꿔도 스킬은 자동으로 동기화되지 않으므로 `hermes skills install <skill-name>`으로 다시 설치하세요.

**작업은 실행되지만 아무것도 전달되지 않음**
전달 대상 문제(위의 전달 실패 참조), 출력 없음 또는 cron 무음 마커 `[SILENT]`가 포함된 응답이 원인일 가능성이 높습니다.

**작업이 멈추거나 시간 초과됨**
스케줄러는 비활동 기반 시간 초과를 사용합니다(기본 600초, `HERMES_CRON_TIMEOUT` 환경 변수로 구성 가능하며 `0`은 무제한). 에이전트가 도구를 활발히 호출하는 동안에는 작업을 계속 실행할 수 있으며, 타이머는 지속적인 비활동 이후에만 작동합니다. 오래 실행되는 작업은 스크립트로 데이터 수집을 처리하고 결과만 전달해야 합니다.

### 점검 3: 잠금 경합

스케줄러는 tick이 겹쳐 실행되는 것을 막기 위해 파일 기반 잠금을 사용합니다. 두 개의 게이트웨이 인스턴스가 실행 중이거나 CLI 세션이 게이트웨이와 충돌하면 작업이 지연되거나 건너뛸 수 있습니다.

중복 게이트웨이 프로세스를 종료하세요.
```bash
ps aux | grep hermes
# Kill duplicate processes, keep only one
```

### 점검 4: jobs.json 권한

작업은 `~/.hermes/cron/jobs.json`에 저장됩니다. 이 파일을 사용자가 읽거나 쓸 수 없다면 스케줄러가 조용히 실패합니다.

```bash
ls -la ~/.hermes/cron/jobs.json
chmod 600 ~/.hermes/cron/jobs.json   # Your user should own it
```

---

## 성능 문제

### 작업 시작이 느림

각 cron 작업은 새로운 AIAgent 세션을 생성하며, 이 과정에서 provider 인증과 모델 로딩이 수행될 수 있습니다. 시간에 민감한 일정에는 버퍼 시간을 추가하세요(예: `0 9 * * *` 대신 `0 8 * * *`).

### 겹치는 작업이 너무 많음

스케줄러는 각 tick 안에서 작업을 순차적으로 실행합니다. 여러 작업의 예정 시간이 같으면 차례대로 실행됩니다. 지연을 피하려면 일정을 분산하는 것을 고려하세요(예: 둘 다 `0 9 * * *`로 설정하는 대신 `0 9 * * *`와 `5 9 * * *`로 설정).

### 스크립트 출력이 너무 큼

메가바이트 단위의 출력을 쏟아내는 스크립트는 에이전트 속도를 늦추고 토큰 제한에 걸릴 수 있습니다. 스크립트 수준에서 필터링하거나 요약해 에이전트가 추론하는 데 필요한 내용만 출력하세요.

---

## 진단 명령

```bash
hermes cron list                    # Show all jobs, states, next_run times
hermes cron run <job_id>            # Schedule for next tick (for testing)
hermes cron edit <job_id>           # Fix configuration issues
hermes logs                         # View recent Hermes logs
hermes skills list                  # Verify installed skills
```

---

## 추가 도움말

이 가이드를 모두 확인했는데도 문제가 지속된다면 다음을 수행하세요.

1. `hermes cron run <job_id>`로 작업을 실행하고(다음 게이트웨이 tick에서 실행됨) 채팅 출력에서 오류를 확인합니다.
2. 스케줄러 메시지는 `~/.hermes/logs/agent.log`에서, 경고는 `~/.hermes/logs/errors.log`에서 확인합니다.
3. 다음 정보를 포함해 [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)에 이슈를 엽니다.
   - 작업 ID와 일정
   - 전달 대상
   - 예상한 동작과 실제 발생한 동작
   - 로그의 관련 오류 메시지

---

*전체 cron 참조는 [Cron으로 무엇이든 자동화하기](/guides/automate-with-cron) 및 [예약 작업(Cron)](/user-guide/features/cron)을 참조하세요.*
