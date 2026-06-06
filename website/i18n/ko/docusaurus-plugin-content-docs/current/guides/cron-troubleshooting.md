---
sidebar_position: 12
title: "크론(Cron) 문제 해결 (Cron Troubleshooting)"
description: "일반적인 Hermes 크론 문제 진단 및 수정 — 실행되지 않는 작업, 전송 실패, 스킬 로딩 오류 및 성능 문제"
---

# 크론(Cron) 문제 해결

크론 작업이 예상대로 작동하지 않을 때, 다음 확인 사항들을 순서대로 진행해 보세요. 대부분의 문제는 타이밍(timing), 전송(delivery), 권한(permissions), 스킬 로딩(skill loading) 중 하나의 범주에 속합니다.

---

## 작업이 실행되지 않음 (Jobs Not Firing)

### 확인 1: 작업이 존재하고 활성 상태인지 확인

```bash
hermes cron list
```

작업을 찾아 상태가 `[active]`인지 확인하세요 (`[paused]`나 `[completed]`가 아니어야 함). 만약 `[completed]`로 표시된다면 반복 횟수(repeat count)가 소진된 것일 수 있습니다 — 작업을 편집하여 초기화하세요.

### 확인 2: 일정이 올바른지 확인

잘못된 형식의 일정은 조용히 일회성(one-shot)으로 기본 설정되거나 완전히 거부됩니다. 표현식을 테스트해 보세요:

| 사용자 표현식 | 평가 결과 |
|----------------|-------------------|
| `0 9 * * *` | 매일 오전 9:00 |
| `0 9 * * 1` | 매주 월요일 오전 9:00 |
| `every 2h` | 지금부터 2시간마다 |
| `30m` | 지금부터 30분 후 |
| `2025-06-01T09:00:00` | 2025년 6월 1일 오전 9:00 (UTC) |

작업이 한 번 실행된 후 목록에서 사라진다면 일회성 일정(`30m`, `1d` 또는 ISO 타임스탬프)이며, 이는 정상적인 동작입니다.

### 확인 3: 게이트웨이가 실행 중인가요?

크론 작업은 게이트웨이의 백그라운드 티커(ticker) 스레드에 의해 실행되며, 이 스레드는 60초마다 틱(tick)을 발생시킵니다. 일반적인 CLI 채팅 세션은 크론 작업을 자동으로 실행하지 **않습니다**.

작업이 자동으로 실행되기를 기대한다면 실행 중인 게이트웨이가 필요합니다 (포그라운드 실행을 위해서는 `hermes gateway`, 설치된 서비스의 경우 `hermes gateway start`). 일회성 디버깅을 위해서는 `hermes cron tick`을 사용하여 수동으로 틱을 트리거할 수 있습니다.

### 확인 4: 시스템 시계 및 시간대(timezone) 확인

작업은 로컬 시간대를 사용합니다. 컴퓨터의 시계가 잘못되었거나 예상과 다른 시간대인 경우, 작업이 잘못된 시간에 실행됩니다. 확인 방법:

```bash
date
hermes cron list   # next_run 시간과 로컬 시간을 비교하세요
```

---

## 전송 실패 (Delivery Failures)

### 확인 1: 전송 대상(deliver target)이 올바른지 확인

전송 대상은 대소문자를 구분하며 올바른 플랫폼이 구성되어 있어야 합니다. 잘못 구성된 대상은 조용히 응답을 무시(drop)합니다.

| 대상 (Target) | 필요 조건 |
|--------|----------|
| `telegram` | `~/.hermes/.env` 파일 내 `TELEGRAM_BOT_TOKEN` |
| `discord` | `~/.hermes/.env` 파일 내 `DISCORD_BOT_TOKEN` |
| `slack` | `~/.hermes/.env` 파일 내 `SLACK_BOT_TOKEN` |
| `whatsapp` | WhatsApp 게이트웨이 구성 |
| `signal` | Signal 게이트웨이 구성 |
| `matrix` | Matrix 홈서버 구성 |
| `email` | `config.yaml` 내 SMTP 구성 |
| `sms` | SMS 제공자 구성 |
| `local` | `~/.hermes/cron/output/`에 대한 쓰기 권한 |
| `origin` | 작업이 생성된 채팅방으로 전송 |

지원되는 다른 플랫폼으로는 `mattermost`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`, `webhook` 등이 있습니다. `platform:chat_id` 구문(예: `telegram:-1001234567890`)을 사용하여 특정 채팅방을 지정할 수도 있습니다.

전송에 실패하더라도 작업 자체는 실행됩니다 — 단지 아무 곳으로도 전송되지 않을 뿐입니다. `hermes cron list`에서 업데이트된 `last_error` 필드(있는 경우)를 확인하세요.

### 확인 2: `[SILENT]` 사용 확인

크론 작업이 아무런 출력도 생성하지 않거나 에이전트가 `[SILENT]`로 응답하면, 전송이 억제됩니다. 이는 모니터링 작업을 위해 의도된 것이지만 — 프롬프트가 실수로 모든 응답을 억제하고 있지는 않은지 확인하세요.

"변경된 사항이 없으면 [SILENT]로 응답해"라는 식의 프롬프트는 비어 있지 않은 응답도 조용히 삼켜버릴 수 있습니다. 조건부 로직을 확인하세요.

### 확인 3: 플랫폼 토큰 권한

각 메시징 플랫폼 봇은 메시지를 수신하기 위한 특정 권한이 필요합니다. 전송이 아무런 경고 없이 실패하는 경우:

- **Telegram**: 봇이 대상 그룹/채널의 관리자여야 합니다.
- **Discord**: 봇이 대상 채널에 메시지를 보낼 권한을 가지고 있어야 합니다.
- **Slack**: 봇이 워크스페이스에 추가되어 있어야 하고 `chat:write` 스코프를 가지고 있어야 합니다.

### 확인 4: 응답 래핑 (Response wrapping)

기본적으로 크론 응답은 헤더와 푸터로 래핑됩니다 (`config.yaml`에서 `cron.wrap_response: true`). 일부 플랫폼이나 통합(integration)은 이를 잘 처리하지 못할 수 있습니다. 비활성화하려면:

```yaml
cron:
  wrap_response: false
```

---

## 스킬 로딩 실패 (Skill Loading Failures)

### 확인 1: 스킬이 설치되어 있는지 확인

```bash
hermes skills list
```

스킬을 크론 작업에 연결하기 전에 반드시 설치되어 있어야 합니다. 스킬이 누락된 경우, `hermes skills install <skill-name>` 또는 CLI의 `/skills`를 통해 먼저 설치하세요.

### 확인 2: 스킬 이름 vs. 스킬 폴더 이름 확인

스킬 이름은 대소문자를 구분하며 설치된 스킬의 폴더 이름과 일치해야 합니다. 작업에서 `ai-funding-daily-report`를 지정했는데 스킬 폴더 이름이 다른 경우, `hermes skills list`에서 정확한 이름을 확인하세요.

### 확인 3: 대화형 도구(interactive tools)가 필요한 스킬

크론 작업은 `cronjob`, `messaging`, `clarify` 도구 모음(toolsets)이 비활성화된 상태로 실행됩니다. 이는 재귀적인 크론 생성, 직접 메시지 전송(전송은 스케줄러가 처리함), 그리고 대화형 프롬프트를 방지합니다. 스킬이 이러한 도구 모음에 의존한다면 크론 컨텍스트에서는 작동하지 않습니다.

스킬이 비대화형(헤드리스) 모드에서 작동하는지 확인하려면 스킬 문서를 참조하세요.

### 확인 4: 다중 스킬 순서 (Multi-skill ordering)

여러 스킬을 사용할 때, 이들은 지정된 순서대로 로드됩니다. 스킬 A가 스킬 B의 컨텍스트에 의존한다면 B가 먼저 로드되도록 하세요:

```bash
/cron add "0 9 * * *" "..." --skill context-skill --skill target-skill
```

이 예제에서는 `context-skill`이 `target-skill`보다 먼저 로드됩니다.

---

## 작업 오류 및 실패 (Job Errors and Failures)

### 확인 1: 최근 작업 출력 검토

작업이 실행되었다가 실패한 경우, 다음 위치에서 오류 컨텍스트를 볼 수 있습니다:

1. 작업이 전송되는 채팅방 (전송이 성공한 경우)
2. 스케줄러 메시지가 기록되는 `~/.hermes/logs/agent.log` (또는 경고의 경우 `errors.log`)
3. `hermes cron list`를 통한 작업의 `last_run` 메타데이터

### 확인 2: 일반적인 오류 패턴

**스크립트에서 "No such file or directory" 발생**
`script` 경로는 절대 경로여야 합니다 (또는 Hermes 설정 디렉토리 기준 상대 경로). 확인 방법:
```bash
ls ~/.hermes/scripts/your-script.py   # 반드시 존재해야 함
hermes cron edit <job_id> --script ~/.hermes/scripts/your-script.py
```

**작업 실행 시 "Skill not found" 발생**
스케줄러를 실행하는 기기에 스킬이 설치되어 있어야 합니다. 기기를 이동하는 경우 스킬이 자동으로 동기화되지 않으므로, `hermes skills install <skill-name>`으로 다시 설치하세요.

**작업은 실행되지만 아무것도 전송하지 않음**
전송 대상 문제(위의 전송 실패 참조) 또는 조용히 억제된 응답(`[SILENT]`)일 가능성이 높습니다.

**작업이 멈추거나 시간 초과(timeout)됨**
스케줄러는 비활동 기반 시간 초과를 사용합니다 (기본값 600초, `HERMES_CRON_TIMEOUT` 환경 변수로 설정 가능, `0`으로 설정 시 무제한). 에이전트가 적극적으로 도구를 호출하는 한 계속 실행될 수 있으며 — 타이머는 지속적인 비활동 후에만 트리거됩니다. 오래 실행되는 작업은 데이터를 수집하고 결과만 전송하도록 스크립트를 사용해야 합니다.

### 확인 3: 잠금 경합 (Lock contention)

스케줄러는 중첩되는 틱을 방지하기 위해 파일 기반 잠금을 사용합니다. 두 개의 게이트웨이 인스턴스가 실행 중이거나 (또는 CLI 세션이 게이트웨이와 충돌하는 경우), 작업이 지연되거나 건너뛸 수 있습니다.

중복된 게이트웨이 프로세스를 종료하세요:
```bash
ps aux | grep hermes
# 중복된 프로세스를 찾아 종료하고 하나만 남깁니다
```

### 확인 4: jobs.json에 대한 권한

작업은 `~/.hermes/cron/jobs.json`에 저장됩니다. 이 파일을 사용자가 읽거나 쓸 수 없는 경우, 스케줄러는 아무런 알림 없이 실패합니다:

```bash
ls -la ~/.hermes/cron/jobs.json
chmod 600 ~/.hermes/cron/jobs.json   # 해당 사용자가 소유자여야 합니다
```

---

## 성능 문제 (Performance Issues)

### 작업 시작이 느림

각 크론 작업은 새로운 AIAgent 세션을 생성하며, 여기에는 제공자 인증과 모델 로딩이 포함될 수 있습니다. 시간에 민감한 일정의 경우 버퍼 시간을 추가하세요 (예: `0 9 * * *` 대신 `0 8 * * *`).

### 겹치는 작업이 너무 많음

스케줄러는 각 틱 내에서 작업을 순차적으로 실행합니다. 여러 작업의 실행 시간이 같은 경우, 하나씩 차례로 실행됩니다. 지연을 방지하려면 일정을 엇갈리게 설정하는 것을 고려하세요 (예: 두 작업을 모두 `0 9 * * *`에 두는 대신 `0 9 * * *`와 `5 9 * * *`로 설정).

### 대용량 스크립트 출력

수 메가바이트의 출력을 덤프하는 스크립트는 에이전트의 속도를 늦추고 토큰 제한에 도달하게 할 수 있습니다. 스크립트 수준에서 필터링하거나 요약하세요 — 에이전트가 추론하는 데 필요한 내용만 출력하도록 하세요.

---

## 진단 명령어 (Diagnostic Commands)

```bash
hermes cron list                    # 모든 작업, 상태, next_run 시간 표시
hermes cron run <job_id>            # 다음 틱에 실행하도록 예약 (테스트용)
hermes cron edit <job_id>           # 구성 문제 수정
hermes logs                         # 최근 Hermes 로그 보기
hermes skills list                  # 설치된 스킬 확인
```

---

## 추가 지원받기

이 가이드를 따라 확인했는데도 문제가 지속되는 경우:

1. `hermes cron run <job_id>`(다음 게이트웨이 틱에 실행)로 작업을 실행하고 채팅 출력에서 오류를 확인하세요.
2. 스케줄러 메시지는 `~/.hermes/logs/agent.log`에서, 경고는 `~/.hermes/logs/errors.log`에서 확인하세요.
3. [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)에서 다음 정보와 함께 이슈를 여세요:
   - 작업 ID와 일정
   - 전송 대상 (delivery target)
   - 예상한 결과 vs. 실제 발생한 결과
   - 로그의 관련 오류 메시지

---

*전체 크론 레퍼런스는 [Automate Anything with Cron](/guides/automate-with-cron) 및 [Scheduled Tasks (Cron)](/user-guide/features/cron)을 참조하세요.*
