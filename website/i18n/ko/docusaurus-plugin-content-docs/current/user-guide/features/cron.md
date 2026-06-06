---
sidebar_position: 5
title: "예약된 작업 (Scheduled Tasks / Cron)"
description: "자연어로 자동화된 작업을 예약하고, 하나의 cron 도구로 관리하며, 하나 이상의 스킬을 연결합니다"
---

# 예약된 작업 (Scheduled Tasks / Cron)

자연어 또는 cron 표현식으로 자동으로 실행될 작업을 예약하세요. Hermes는 별도의 예약/목록/제거 도구 대신 작업(action) 스타일의 연산을 수행하는 단일 `cronjob` 도구를 통해 cron 관리를 노출합니다.

## cron으로 현재 할 수 있는 것들

Cron 작업은 다음을 수행할 수 있습니다:

- 일회성(one-shot) 또는 반복 작업 예약
- 작업 일시 중지, 재개, 편집, 트리거 및 제거
- 작업에 0개, 1개 또는 여러 개의 스킬(skills) 연결
- 결과를 원본 채팅, 로컬 파일, 또는 구성된 플랫폼 타겟으로 전달
- 일반적인 정적 도구 목록과 함께 새로운 에이전트 세션에서 실행
- **에이전트 없는 모드(no-agent mode)**로 실행 — 정해진 일정에 스크립트만 실행하며, LLM 개입 없이 표준 출력(stdout)을 그대로 전달합니다 (아래 [에이전트 없는 모드 (스크립트 전용 작업)](#no-agent-mode-script-only-jobs) 섹션 참조)

이 모든 기능은 통합된 `cronjob` 도구를 통해 Hermes 자체에 제공되므로, CLI가 없어도 평범한 언어로 요청하여 작업을 생성, 일시 중지, 편집, 제거할 수 있습니다.

:::tip
Cron 작업은 `hermes model`에서 선택한 공급자를 사용합니다. 자동 OAuth 갱신을 지원하므로, 사용자가 지켜보지 않는 무인 실행(unattended runs)의 경우 `hermes setup --portal`이 가장 마찰이 적은 옵션입니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

:::warning
Cron으로 실행된 세션은 재귀적으로 더 많은 cron 작업을 생성할 수 없습니다. Hermes는 무한 예약 루프를 방지하기 위해 cron 실행 내부에서 cron 관리 도구를 비활성화합니다.
:::

## 예약된 작업 생성하기

### 채팅에서 `/cron` 사용

```bash
/cron add 30m "빌드를 확인하라고 나에게 알려줘"
/cron add "every 2h" "서버 상태 확인해줘"
/cron add "every 1h" "새로운 피드 항목을 요약해줘" --skill blogwatcher
/cron add "every 1h" "두 스킬을 모두 사용하고 결과를 합쳐줘" --skill blogwatcher --skill maps
```

### 독립 실행형 CLI에서

```bash
hermes cron create "every 2h" "Check server status"
hermes cron create "every 1h" "Summarize new feed items" --skill blogwatcher
hermes cron create "every 1h" "Use both skills and combine the result" \
  --skill blogwatcher \
  --skill maps \
  --name "Skill combo"
```

### 자연스러운 대화를 통해

Hermes에게 평소처럼 요청하세요:

```text
매일 아침 9시에 Hacker News에서 AI 뉴스를 확인하고 텔레그램으로 요약본을 보내줘.
```

Hermes는 내부적으로 통합된 `cronjob` 도구를 사용합니다.

## 스킬이 연계된 cron 작업 (Skill-backed cron jobs)

Cron 작업은 프롬프트를 실행하기 전에 하나 이상의 스킬을 로드할 수 있습니다.

### 단일 스킬

```python
cronjob(
    action="create",
    skill="blogwatcher",
    prompt="구성된 피드를 확인하고 새로운 내용을 요약하세요.",
    schedule="0 9 * * *",
    name="Morning feeds",
)
```

### 다중 스킬

스킬은 순서대로 로드됩니다. 프롬프트는 해당 스킬들 위에 계층화된(layered) 작업 지시사항이 됩니다.

```python
cronjob(
    action="create",
    skills=["blogwatcher", "maps"],
    prompt="새로운 지역 이벤트와 흥미로운 주변 장소를 찾은 다음, 이를 하나의 짧은 브리핑으로 통합하세요.",
    schedule="every 6h",
    name="Local brief",
)
```

이 기능은 전체 스킬 텍스트를 cron 프롬프트 자체에 우겨넣지 않고 예약된 에이전트가 재사용 가능한 워크플로우를 상속받게 하려는 경우에 유용합니다.

## 프로젝트 디렉터리 내에서 작업 실행하기

Cron 작업은 기본적으로 어떤 레포지토리와도 분리되어 실행됩니다 — `AGENTS.md`, `CLAUDE.md`, 또는 `.cursorrules`가 로드되지 않으며, 터미널/파일/코드 실행 도구들은 게이트웨이가 시작된 작업 디렉터리에서 실행됩니다. 이를 변경하려면 `--workdir` (CLI) 또는 `workdir=` (도구 호출)을 전달하세요:

```bash
# 독립 실행형 CLI (일정과 프롬프트는 위치 인자입니다)
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# 채팅에서 cronjob 도구를 통한 예시
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="열려있는 PR을 감사하고, CI 상태를 요약한 후 #eng에 게시해",
)
```

`workdir`이 설정된 경우:

- 해당 디렉터리의 `AGENTS.md`, `CLAUDE.md`, `.cursorrules`가 시스템 프롬프트에 주입됩니다 (대화형 CLI와 동일한 발견 순서).
- `terminal`, `read_file`, `write_file`, `patch`, `search_files`, `execute_code` 모두 해당 디렉터리를 작업 디렉터리로 사용합니다.
- 경로는 반드시 존재하는 절대 디렉터리여야 합니다 — 생성/업데이트 시 상대 경로 및 존재하지 않는 디렉터리는 거부됩니다.
- 편집(edit)할 때 `--workdir ""` (또는 도구 통해 `workdir=""`)를 전달하여 이를 지우고 기존 동작으로 복원할 수 있습니다.

:::note 직렬화 (Serialization)
`workdir`이 지정된 작업들은 스케줄러 틱(tick)에서 병렬 풀이 아닌 순차적으로 실행됩니다. 이는 의도적인 설계입니다: cron 작업자(worker)는 프로세스 전역(process-global) 터미널 상태를 통해 작업 workdir을 적용하므로, 두 개의 workdir 작업이 동시에 실행되면 서로의 현재 작업 디렉터리(cwd)를 오염시킬 수 있기 때문입니다. workdir이 없는 작업은 이전처럼 병렬로 실행됩니다.
:::

## 특정 프로필에서 cron 작업 실행하기

기본적으로 cron 작업은 작업을 생성한 게이트웨이/CLI를 소유한 Hermes 프로필을 상속받습니다. `--profile <name>` (CLI) 또는 `profile=` (cronjob 도구)을 전달하여 작업 대상을 다른 프로필로 변경할 수 있습니다 — 스케줄러는 해당 프로필의 `HERMES_HOME`을 확인하고, 실행되는 동안 임시로 해당 프로필로 전환하여 `.env` 및 `config.yaml`을 로드한 후, 그곳에서 작업을 실행합니다:

```bash
# 어디서 예약되었는지에 관계없이 `night-ops` 프로필에 작업을 고정(pin)합니다.
hermes cron create "every 1d at 03:00" \
  "보안 로그를 테일링(tail)하고 이상 징후를 플래그해" \
  --profile night-ops
```

```python
# 채팅에서 cronjob 도구를 통한 예시
cronjob(
    action="create",
    schedule="every 1d at 03:00",
    prompt="보안 로그를 테일링하고 이상 징후를 플래그해",
    profile="night-ops",
)
```

최상위 기본 Hermes 프로필에 명시적으로 고정하려면 `--profile default`를 사용하세요. 지정된 프로필은 이미 존재해야 합니다; 스케줄러는 실행 중에 프로필을 임의로 생성하는 것을 거부합니다. `cron edit` 중에 프로필 고정을 해제하려면 빈 문자열(`--profile ""` 또는 `profile=""`)을 전달하세요 — 작업은 스케줄러 자체가 있는 프로필에서 실행되는 것으로 되돌아갑니다.

고정된 프로필이 나중에 삭제된 경우 스케줄러는 경고를 기록하고 크래시(crash)되는 대신 현재 프로필에서 작업을 실행하는 것으로 폴백합니다 — 따라서 오래된 `profile` 참조로 인해 작업이 멈추는(wedge) 일은 결코 발생하지 않습니다.

:::note 직렬화 (Serialization)
`profile`이 설정된 작업도 `workdir`이 고정된 작업과 동일한 이유로 순차적으로 실행됩니다: `HERMES_HOME`을 전환하는 것은 프로세스 전역 상태의 변경이므로 프로필이 고정된 두 작업이 병렬로 실행되면 경쟁 상태(race condition)가 발생합니다. 고정되지 않은 작업들은 계속 일반 병렬 풀에서 실행됩니다.
:::

## 작업 편집 (Editing jobs)

작업을 변경하기 위해 삭제하고 다시 만들 필요가 없습니다.

:::tip 작업 참조 (Job reference)
아래(및 [수명 주기 동작](#lifecycle-actions))의 `<job_id>` 플레이스홀더는 작업의 이름(대소문자 구분 없음)도 허용합니다 — 16진수 ID 대신 `morning-digest`와 같이 이름을 기억할 때 편리합니다. 정확한 작업 ID가 이름 일치보다 우선합니다; 참조값이 ID가 아니고 이름이 두 개 이상의 작업과 일치하는 경우, 명령이 거부되고 모호함을 해소할 수 있도록 후보 ID 목록을 인쇄합니다.
:::

### 채팅

```bash
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Use the revised task"
/cron edit <job_id> --skill blogwatcher --skill maps
/cron edit <job_id> --remove-skill blogwatcher
/cron edit <job_id> --clear-skills
```

### 독립 실행형 CLI

```bash
hermes cron edit <job_id> --schedule "every 4h"
hermes cron edit <job_id> --prompt "Use the revised task"
hermes cron edit <job_id> --skill blogwatcher --skill maps
hermes cron edit <job_id> --add-skill maps
hermes cron edit <job_id> --remove-skill blogwatcher
hermes cron edit <job_id> --clear-skills
```

참고:

- `--skill`을 반복하면 작업에 첨부된 스킬 목록이 바뀝니다.
- `--add-skill`은 기존 목록을 교체하지 않고 추가(append)합니다.
- `--remove-skill`은 연결된 특정 스킬을 제거합니다.
- `--clear-skills`는 첨부된 모든 스킬을 제거합니다.

## 수명 주기 동작 (Lifecycle actions)

Cron 작업은 이제 단순한 생성/제거보다 훨씬 완전한 라이프사이클을 가집니다.

### 채팅

```bash
/cron list
/cron pause <job_id>
/cron resume <job_id>
/cron run <job_id>
/cron remove <job_id>
```

### 독립 실행형 CLI

```bash
hermes cron list
hermes cron pause <job_id_or_name>
hermes cron resume <job_id_or_name>
hermes cron run <job_id_or_name>
hermes cron remove <job_id_or_name>
hermes cron edit <job_id_or_name> [...flags]
hermes cron status
hermes cron tick
```

수행하는 역할:

- `pause` — 작업은 유지하되 예약 실행을 중지합니다.
- `resume` — 작업을 다시 활성화하고 다음 실행 시간을 계산합니다.
- `run` — 다음 스케줄러 틱(tick)에서 작업을 트리거합니다.
- `remove` — 작업을 완전히 삭제합니다.
- `edit` — 일정, 프롬프트, 프로필, 전송 대상 등을 수정합니다.

**이름 기반 조회.** 4개의 변경 동사(`pause`, `resume`, `run`, `remove`, `edit`)와 에이전트의 `cronjob` 도구는 모두 16진수 ID 대신 작업 **이름**(대소문자 구분 안 함)을 허용합니다. 에이전트와 CLI는 모두 정확한 ID가 존재하는 경우 이를 선호합니다; 모호한 이름 일치(여러 작업이 같은 이름을 공유하는 경우)는 사용자가 명시적으로 하나를 선택할 수 있도록 전체 후보 ID 목록과 함께 거부됩니다. 이름은 고유하지 않으므로, 이 보호 기능이 꽤 중요합니다 — 동일한 이름을 공유할 때 조용히 엉뚱한 작업을 수정하는 것을 방지합니다.

## 작동 방식

**Cron 실행은 게이트웨이 데몬에서 처리합니다.** 게이트웨이는 스케줄러를 60초마다 틱(tick)하여 격리된 에이전트 세션에서 예정된 모든 작업을 실행합니다.

```bash
hermes gateway install     # 사용자 서비스로 설치
sudo hermes gateway install --system   # Linux: 서버용 부팅 시 시스템 서비스
hermes gateway             # 또는 포그라운드에서 실행

hermes cron list
hermes cron status
```

### 게이트웨이 스케줄러 동작

각 틱(tick)마다 Hermes는 다음을 수행합니다:

1. `~/.hermes/cron/jobs.json`에서 작업을 로드합니다.
2. `next_run_at`과 현재 시간을 비교 확인합니다.
3. 기한이 된 각 작업에 대해 새로운 `AIAgent` 세션을 시작합니다.
4. (선택 사항) 해당 새 세션에 연결된 하나 이상의 스킬을 주입합니다.
5. 프롬프트를 끝까지 실행합니다.
6. 최종 응답을 전송합니다.
7. 실행 메타데이터와 다음 예약 시간을 업데이트합니다.

`~/.hermes/cron/.tick.lock`에 있는 파일 잠금(lock)은 스케줄러 틱이 겹쳐서 동일한 작업 배치가 두 번 실행되는 것을 방지합니다.

## 전달(Delivery) 옵션

작업을 예약할 때 출력 결과가 갈 곳을 지정합니다:

| 옵션 | 설명 | 예시 |
|--------|-------------|---------|
| `"origin"` | 작업이 생성된 곳으로 되돌려 보냄 | 메시징 플랫폼의 기본값 |
| `"local"` | 로컬 파일에만 저장 (`~/.hermes/cron/output/`) | CLI의 기본값 |
| `"telegram"` | Telegram 홈 채널 | `TELEGRAM_HOME_CHANNEL` 사용 |
| `"telegram:123456"` | 특정 Telegram 채팅 ID | 직접 전송 |
| `"telegram:-100123:17585"` | 특정 Telegram 주제(topic) | `chat_id:thread_id` 형식 |
| `"discord"` | Discord 홈 채널 | `DISCORD_HOME_CHANNEL` 사용 |
| `"discord:#engineering"` | 특정 Discord 채널 | 채널명 지정 |
| `"slack"` | Slack 홈 채널 | |
| `"whatsapp"` | WhatsApp 홈 | |
| `"signal"` | Signal | |
| `"matrix"` | Matrix 홈 룸(room) | |
| `"mattermost"` | Mattermost 홈 채널 | |
| `"email"` | 이메일 | |
| `"sms"` | Twilio를 통한 SMS | |
| `"homeassistant"` | Home Assistant | |
| `"dingtalk"` | DingTalk | |
| `"feishu"` | Feishu/Lark | |
| `"wecom"` | WeCom | |
| `"weixin"` | Weixin (WeChat) | |
| `"bluebubbles"` | BlueBubbles (iMessage) | |
| `"qqbot"` | QQ Bot (Tencent QQ) | |
| `"all"` | 연결된 모든 홈 채널로 확산(Fan out) | 실행 시점에 확인됨 |
| `"telegram,discord"` | 특정 채널 세트로 확산 | 쉼표로 구분된 목록 |
| `"origin,all"` | 원본 **그리고** 그 밖의 모든 연결된 홈 채널로 전달 | 어떤 토큰이든 결합 가능 |

에이전트의 최종 응답은 자동으로 전송됩니다. cron 프롬프트에서 `send_message`를 호출할 필요가 없습니다.

### 라우팅 인텐트 (`all`)

`all`을 사용하면 여러 채널을 일일이 지정할 필요 없이 구성된 모든 메시징 채널로 단일 cron 작업을 배포(ship)할 수 있습니다. 이는 **실행 시점에 분석(resolved at fire time)**되므로, Telegram 연동 이전에 생성된 작업도 `TELEGRAM_HOME_CHANNEL`을 설정하고 나면 그 다음 틱부터 Telegram을 인식하게 됩니다.

동작 방식: `all`은 홈 채널이 설정된 모든 플랫폼으로 확장됩니다. 대상이 하나도 없어도(Zero) 괜찮습니다; 그저 전달 대상(delivery targets)을 생성하지 않고 상위 레벨에 전송 실패로 기록될 뿐입니다.

`all`은 명시적인 대상(explicit targets)과 조합하여 쓸 수 있습니다. `origin,all`은 원본(origin) 채팅과 그 밖의 모든 연결된 홈 채널로 전달하되, `(platform, chat_id, thread_id)` 단위로 중복을 제거(de-duplicating)합니다.

### Telegram cron 스레드(topic) (`TELEGRAM_CRON_THREAD_ID`)

Telegram 토픽(topic) 모드가 활성화된 경우 기본 DM은 시스템 로비로 예약됩니다 — 그곳으로 전송된 응답은 로비 안내 메시지와 함께 튕겨져 나오며 `reply_to_message_id`가 버려지므로, 메인 채팅에 도착한 cron 메시지에 직접 답장할 수 없습니다.

대신 cron을 전용 포럼 토픽으로 지정하세요:

1. Telegram에서 봇 DM을 열고 예컨대 `Cron`이라는 이름의 토픽을 만듭니다. 토픽 헤더를 길게 누른 후 → **링크 복사(Copy link)**를 선택합니다; 끝에 붙어 있는 정수가 해당 토픽의 `message_thread_id`입니다.
2. `.env` 파일에 `TELEGRAM_CRON_THREAD_ID=<복사한 ID>`를 설정합니다.

이 설정은 오직 cron 전송에만 적용됩니다. 다른 곳(예: 재시작 알림)에서 사용하는 `TELEGRAM_HOME_CHANNEL_THREAD_ID`는 변경되지 않습니다. 명시적으로 `deliver="telegram:chat_id:thread_id"` 대상을 지정한 경우에는 환경변수보다 우선 적용됩니다. 이제 cron 메시지에 대한 답장이 기존의 토픽 세션 내로 들어오므로, 즉시 후속 조치를 취할 수 있습니다.

### 응답 감싸기 (Response wrapping)

기본적으로 전송되는 cron 출력은 수신자가 예약된 작업에서 온 것임을 알 수 있도록 헤더와 푸터로 감싸집니다:

```
Cronjob Response: Morning feeds
-------------

<여기에 에이전트 출력>

Note: The agent cannot see this message, and therefore cannot respond to it.
(참고: 에이전트는 이 메시지를 볼 수 없으므로, 이에 응답할 수 없습니다.)
```

래퍼 없이 에이전트의 순수(raw) 출력을 그대로 전달하려면 `cron.wrap_response`를 `false`로 설정하세요:

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### 무음 억제 (Silent suppression)

에이전트의 최종 응답이 `[SILENT]`로 시작하면 전송이 완전히 억제됩니다. 출력은 감사(audit)를 위해 로컬(`~/.hermes/cron/output/` 아래)에 계속 저장되지만, 전송 대상으로는 어떠한 메시지도 전송되지 않습니다.

이는 뭔가 잘못되었을 때만 보고해야 하는 모니터링 작업에 유용합니다:

```text
nginx가 실행 중인지 확인하세요. 모든 것이 정상이면 오직 [SILENT]로만 응답하세요.
그렇지 않으면 문제를 보고하세요.
```

실패한 작업은 `[SILENT]` 마커와 관계없이 항상 메시지를 전송합니다 — 오직 성공적인 실행만 침묵 처리될 수 있습니다.

## 스크립트 시간 제한 (Script timeout)

( `script` 파라미터를 통해 연결된) 사전 실행 스크립트의 기본 시간 제한은 120초입니다. 스크립트에 봇과 같은 타이밍 패턴을 피하기 위해 무작위 지연(delay)을 포함하는 등 더 긴 시간이 필요한 경우 이 제한을 늘릴 수 있습니다:

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 300   # 5분
```

또는 환경 변수 `HERMES_CRON_SCRIPT_TIMEOUT`을 설정할 수도 있습니다. 해결 순서는: 환경 변수 → config.yaml → 120초(기본값)입니다.

## 에이전트 없는 모드 (스크립트 전용 작업) (no-agent mode)

LLM 추론이 필요 없는 반복 작업 — 고전적인 워치독(watchdogs), 디스크/메모리 경고, 하트비트, CI 핑 — 을 수행할 때는 작업 생성 시 `no_agent=True`를 전달하세요. 스케줄러가 예정된 일정에 스크립트를 실행하고 그 표준 출력(stdout)을 그대로 전달하여 에이전트를 완전히 건너뜁니다:

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

동작 방식:

- 스크립트의 표준 출력(트리밍됨) → 그대로 메시지로 전달됨.
- **빈 표준 출력(Empty stdout) → 침묵 틱(silent tick)**, 전달 안 함. 이것이 바로 워치독 패턴입니다: "무언가 잘못되었을 때만 말해라".
- 0이 아닌 코드(Non-zero)로 종료 또는 시간 초과 → 에러 알림이 전달되므로 고장난 워치독이 조용히 실패하는 일은 없음.
- 마지막 줄에 `{"wakeAgent": false}` 출력 → 침묵 틱 (LLM 작업이 사용하는 게이트와 동일).
- 토큰 사용 없음, 모델 없음, 공급자 폴백 없음 — 이 작업은 절대로 추론 계층(inference layer)을 건드리지 않습니다.

`.sh` / `.bash` 파일은 `/bin/bash` 환경에서 실행되며, 그 외의 모든 것은 현재의 Python 인터프리터(`sys.executable`) 아래에서 실행됩니다. 스크립트들은 반드시 `~/.hermes/scripts/`에 존재해야 합니다(사전 실행 스크립트 관문의 샌드박싱 규칙과 동일).

### 에이전트가 알아서 설정해 줍니다

`cronjob` 도구의 스키마가 `no_agent` 옵션을 Hermes에게 직접 노출하므로, 채팅에서 워치독에 대해 설명하면 에이전트가 알아서 연결해 줍니다:

```text
RAM 사용량이 85%를 넘으면 5분마다 텔레그램으로 알려줘.
```

Hermes는 `write_file`을 통해 `~/.hermes/scripts/`에 확인용 스크립트를 작성한 다음 아래처럼 호출합니다:

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

메시지의 내용이 오직 스크립트에 의해서만 결정되는 경우(워치독, 임계값 경고, 하트비트) Hermes가 `no_agent=True`를 자동으로 선택합니다. 같은 도구로 에이전트가 작업을 일시 중지, 재개, 편집, 제거할 수도 있으므로 전체 라이프사이클이 사용자가 CLI를 만지지 않고도 채팅을 주도하여 이루어질 수 있습니다.

실제 적용 예제는 [스크립트 전용 Cron 작업 가이드(Script-Only Cron Jobs guide)](/guides/cron-script-only)를 참조하세요.

## `context_from`으로 작업 연결하기 (Chaining jobs)

Cron 작업은 이전 실행의 메모리가 없는 격리된 세션에서 실행됩니다. 하지만 때로는 한 작업의 출력이 다음 작업에서 정확히 필요한 내용일 수 있습니다. `context_from` 매개변수는 이러한 연결을 자동으로 연결합니다 — '작업 B'의 프롬프트는 런타임에 '작업 A'의 최신 출력을 컨텍스트로 앞에 추가받습니다.

```python
# 작업 1: 원시 데이터 수집
cronjob(
    action="create",
    prompt="Hacker News에서 상위 10개의 AI/ML 기사를 가져옵니다. 제목, URL, 점수를 포함하여 ~/.hermes/data/briefs/raw.md에 마크다운 형식으로 저장하세요.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# 작업 2: 분류 (Triage) — 작업 1의 출력을 컨텍스트로 받습니다
# cronjob(action="list")에서 작업 1의 ID를 가져옵니다
cronjob(
    action="create",
    prompt="~/.hermes/data/briefs/raw.md를 읽으세요. 각 기사의 참여 잠재력과 참신성에 대해 1-10점 사이의 점수를 매기세요. 상위 5개를 ~/.hermes/data/briefs/ranked.md에 출력하세요.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# 작업 3: 전달(Ship) — 작업 2의 출력을 컨텍스트로 받습니다
cronjob(
    action="create",
    prompt="~/.hermes/data/briefs/ranked.md를 읽으세요. 3개의 트윗 초안(훅 + 본문 + 해시태그)을 작성하세요. telegram:7976161601로 전달하세요.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**작동 방식:**

- 작업 2가 실행될 때 Hermes는 `~/.hermes/cron/output/{job1_id}/*.md`에서 작업 1의 가장 최근 출력을 읽습니다.
- 해당 출력은 작업 2의 프롬프트 상단에 자동으로 추가됩니다.
- 작업 2는 "이 파일을 읽어라"라고 하드코딩할 필요가 없습니다 — 내용을 컨텍스트로 전달받기 때문입니다.
- 이 체인은 원하는 길이만큼 계속 연결될 수 있습니다: 작업 1 → 작업 2 → 작업 3 → ...

**`context_from`이 허용하는 형식:**

| 형식 | 예시 |
|--------|---------|
| 단일 작업 ID (문자열) | `context_from="a1b2c3d4"` |
| 다중 작업 ID (목록) | `context_from=["job_a", "job_b"]` |

출력 결과들은 나열된 순서대로 하나로 연결됩니다(concatenated).

**언제 사용하는가:**

- 다단계(Multi-stage) 파이프라인 (수집 → 필터링 → 서식 지정 → 전달)
- N단계의 작업이 N−1단계의 출력에 의존하는 종속적인 작업들
- 한 작업이 다른 여러 작업들의 결과를 집계하는 확산/수집(Fan-out/fan-in) 패턴

## 공급자 복구 (Provider recovery)

Cron 작업은 귀하가 설정한 폴백 공급자(fallback providers) 및 자격 증명 풀(credential pool) 회전을 상속받습니다. 주 API 키에 속도 제한(rate-limited)이 걸리거나 공급자가 에러를 반환할 때 cron 에이전트는 다음을 수행할 수 있습니다:

- `config.yaml`에 `fallback_providers` (또는 레거시 `fallback_model`)가 설정된 경우 **대체 공급자로 폴백**합니다.
- 동일한 공급자에 대해 [자격 증명 풀 (credential pool)](/user-guide/configuration#credential-pool-strategies)의 **다음 자격 증명으로 회전(rotate)**합니다.

이는 높은 빈도로 실행되거나 피크 시간대에 실행되는 cron 작업이 더 높은 복원력을 갖게 됨을 의미합니다 — 단일 키의 속도 제한이 전체 실행을 실패로 만들지 않습니다.

## 일정 형식 (Schedule formats)

에이전트의 최종 응답은 자동으로 전달됩니다 — 즉, 동일한 대상을 지정하기 위해 cron 프롬프트에 `send_message`를 포함시킬 필요가 **없습니다**. cron 실행이 스케줄러가 이미 전달하려는 정확한 목적지를 향해 `send_message`를 호출하는 경우, Hermes는 그 중복된 발송을 건너뛰고, 사용자 대면 콘텐츠를 그저 최종 응답 안에 넣도록 모델에게 지시합니다. `send_message`는 추가적이거나 아예 다른 대상일 때만 사용하세요.

### 상대적 지연 (일회성) (Relative delays (one-shot))

```text
30m     → 30분 후 1번 실행
2h      → 2시간 후 1번 실행
1d      → 하루 후 1번 실행
```

### 간격 (반복) (Intervals (recurring))

```text
every 30m    → 30분마다
every 2h     → 2시간마다
every 1d     → 매일
```

### Cron 표현식 (Cron expressions)

```text
0 9 * * *       → 매일 오전 9:00
0 9 * * 1-5     → 평일(월-금) 오전 9:00
0 */6 * * *     → 6시간마다
30 8 1 * *      → 매월 1일 오전 8:30
0 0 * * 0       → 매주 일요일 자정
```

### ISO 타임스탬프 (ISO timestamps)

```text
2026-03-15T09:00:00    → 일회성, 2026년 3월 15일 오전 9:00
```

## 반복 동작 (Repeat behavior)

| 일정(Schedule) 유형 | 기본 반복 횟수 | 동작 방식 |
|--------------|----------------|----------|
| 일회성 (`30m`, timestamp) | 1 | 한 번만 실행됨 |
| 간격 (`every 2h`) | forever (무한) | 제거될 때까지 계속 실행됨 |
| Cron 표현식 | forever (무한) | 제거될 때까지 계속 실행됨 |

다음을 통해 덮어쓸 수 있습니다:

```python
cronjob(
    action="create",
    prompt="...",
    schedule="every 2h",
    repeat=5,
)
```

## 프로그래밍 방식으로 작업 관리하기

에이전트 대상 API는 단일 도구입니다:

```python
cronjob(action="create", ...)
cronjob(action="list")
cronjob(action="update", job_id="...")
cronjob(action="pause", job_id="...")
cronjob(action="resume", job_id="...")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

`update`의 경우, 첨부된 모든 스킬을 제거하려면 `skills=[]`를 전달하세요.

## Cron 작업에서 사용할 수 있는 도구 세트

Cron은 채팅 플랫폼이 연결되지 않은 완전히 새로운 에이전트 세션에서 각 작업을 실행합니다. 기본적으로 cron 에이전트는 — CLI 기본값이나 수많은 모든 도구가 아닌 — **`hermes tools`에서 `cron` 플랫폼에 대해 당신이 구성한 도구 세트**를 받습니다.

```bash
hermes tools
# → curses UI에서 "cron" 플랫폼 선택
# → Telegram/Discord 등과 마찬가지로 도구 세트 켜기/끄기 토글
```

`cronjob.create`의 `enabled_toolsets` 필드를 통해 더 세밀한 작업별 제어가 가능합니다 (또는 기존 작업에 대해 `cronjob.update` 사용):

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # 터미널/브라우저/등등 없이, 웹 + 파일만 허용.
        prompt="Summarize this week's AI news: ...")
```

작업에 `enabled_toolsets`이 설정된 경우 이 설정이 우선합니다. 그렇지 않으면 `hermes tools`의 cron 플랫폼 설정이 적용되며, 이조차도 없으면 Hermes 내장 기본값으로 폴백(fallback)합니다. 이는 비용 통제에 매우 중요합니다: 단순한 "뉴스 가져오기" 작업에 매번 `moa`, `browser`, `delegation` 같은 도구들을 함께 싣고 다니면 LLM 호출마다 도구 스키마 프롬프트 용량이 불필요하게 커지기 때문입니다.

### 에이전트를 완전히 건너뛰기: `wakeAgent`

만약 cron 작업에 (`script=`를 통해) 사전 점검(pre-check) 스크립트가 연결되어 있다면, 해당 스크립트가 런타임에 Hermes가 에이전트를 호출해야 할지 말지를 결정할 수 있습니다. 스크립트의 마지막 줄(stdout)을 다음 형식으로 출력하세요:

```text
{"wakeAgent": false}
```

…그러면 cron은 이번 틱(tick)의 에이전트 실행을 완전히 건너뜁니다. 상태가 실제로 변했을 때만 LLM을 깨워야 하는 잦은 빈도의 폴링(1~5분마다)에 유용합니다 — 그렇지 않으면 의미 없는 에이전트 턴(turn)에 대한 비용을 계속해서 지불해야 하니까요.

```python
# 사전 점검(pre-check) 스크립트
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # 이번 틱은 건너뜀
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

`wakeAgent` 값이 출력되지 않으면 기본값은 `true`(평소처럼 에이전트를 깨움)입니다.

#### 레시피: 저비용(cheap) 사전 실행 관문

`wakeAgent` 관문(gate)은 예정된 작업이 LLM 토큰을 아예 소비해야 하는지 여부를 0원의 비용으로 결정할 수 있는 방법을 제공합니다. 이 세 가지 패턴으로 대부분의 사용 사례를 처리할 수 있습니다.

**파일 변경 관문 (File-change gate)** — 감시 중인 파일에 마지막 성공 틱(tick) 이후의 새로운 콘텐츠가 있을 때만 실행합니다. 스케줄러는 각 작업의 `last_run_at`을 기록합니다. 이를 파일의 수정 시간(mtime)과 비교하세요.

```bash
#!/bin/bash
# ~/.hermes/scripts/feed-changed.sh
FEED="$HOME/data/feed.json"
STATE="$HOME/.hermes/scripts/.feed-changed.last"
test -f "$FEED" || { echo '{"wakeAgent": false}'; exit 0; }
mtime=$(stat -c %Y "$FEED")
last=$(cat "$STATE" 2>/dev/null || echo 0)
if [ "$mtime" -le "$last" ]; then
  echo '{"wakeAgent": false}'
else
  echo "$mtime" > "$STATE"
  echo '{"wakeAgent": true}'
fi
```

```text
cronjob(action="create", name="process-feed",
        schedule="every 30m",
        script="feed-changed.sh",
        prompt="새로운 ~/data/feed.json 파일이 도착했습니다. 변경된 내용을 요약하세요.")
```

**외부 플래그 관문 (External-flag gate)** — 다른 프로세스가 준비되었다는 신호를 보냈을 때만 실행합니다 (예: 배포 훅(deploy hook)이 파일을 떨구거나, CI 작업이 상태 저장소에 값을 설정한 경우).

```bash
#!/bin/bash
# ~/.hermes/scripts/flag-ready.sh
if test -f /tmp/new-data-ready; then
  rm -f /tmp/new-data-ready
  echo '{"wakeAgent": true}'
else
  echo '{"wakeAgent": false}'
fi
```

```text
cronjob(action="create", name="nightly-analysis",
        schedule="0 9 * * *",
        script="flag-ready.sh",
        prompt="오늘자 배치 데이터에 대해 야간 분석(nightly analysis)을 실행해.")
```

**SQL 카운트 관문 (SQL-count gate)** — 자신의 데이터베이스에 처리할 새로운 행(row)이 있을 때만 실행합니다. 스크립트는 `context`를 통해 에이전트에게 개수를 전달할 수 있으므로, 에이전트가 데이터베이스를 다시 조회(re-querying)하지 않아도 얼마나 많은 항목을 보게 될지 미리 알 수 있습니다.

```python
#!/usr/bin/env python
# ~/.hermes/scripts/new-rows.py
import json, sqlite3
conn = sqlite3.connect("/home/me/data/app.db")
n = conn.execute(
    "SELECT COUNT(*) FROM messages WHERE ts > strftime('%s','now','-2 hours')"
).fetchone()[0]
if n < 1:
    print(json.dumps({"wakeAgent": False}))
else:
    print(json.dumps({"wakeAgent": True, "context": {"new_rows": n}}))
```

```text
cronjob(action="create", name="summarize-new-msgs",
        schedule="every 2h",
        script="new-rows.py",
        prompt="지난 2시간 동안의 새로운 메시지를 요약해.")
```

cron 서브시스템 내부에 별도의 SQL 평가기를 구현해 넣을 필요 없이, 스크립트를 통해 조회할 수 있는 모든 데이터 소스(Postgres, HTTP API, 자체 상태 저장소 등)에 동일한 패턴을 사용할 수 있습니다.

:::tip
Hermes 자체의 `~/.hermes/state.db`는 릴리스 버전 간에 변경될 수 있는 내부 스키마입니다. 사전-실행(pre-run) 관문 스크립트에서 이를 직접 조회(query)하지 마세요 — 대신 사용자 본인의 데이터베이스나 피드(feed)를 가리키도록 하세요.
:::

크레딧: 이 레시피 세트는 @iankar8 님이 [#2654](https://github.com/NousResearch/hermes-agent/pull/2654)에서 탐구한 내용(sql/파일/명령어 트리거를 병렬 매커니즘으로 추가하자는 제안)으로부터 비롯되었습니다. `script` + `wakeAgent` 게이트만으로도 이미 3가지 케이스를 모두 $0 비용으로 커버할 수 있었기 때문에, 해당 제안은 코드 대신 문서화 내용으로 반영되었습니다.

### 작업 연결: `context_from` (Chaining jobs)

Cron 작업은 `context_from`에 하나 이상의 작업 이름(또는 ID)을 나열함으로써 해당 작업들의 가장 최근에 성공한 출력을 사용할(consume) 수 있습니다:

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="위의 출력 결과들을 사용하여 일일 요약(daily digest)을 작성하세요.")
```

참조된 작업들의 가장 최근에 완료된 출력 결과들이 이번 실행의 프롬프트 위쪽에 컨텍스트(context)로 주입됩니다. 각 업스트림 항목은 유효한 작업 ID이거나 이름이어야 합니다 (`cronjob action="list"` 참조). 주의할 점: 체이닝(chaining)은 *가장 최근에 완료된* 출력물을 읽습니다 — 동일한 틱(tick) 내에서 동시에 실행 중인 업스트림 작업을 기다리지(wait)는 않습니다.

## 작업 저장 위치 (Job storage)

작업은 `~/.hermes/cron/jobs.json`에 저장됩니다. 작업 실행 결과물(Output)은 `~/.hermes/cron/output/{job_id}/{timestamp}.md`에 저장됩니다.

작업은 `model`과 `provider`를 `null`로 저장할 수 있습니다. 이 필드들이 생략된 경우, Hermes는 실행 시점에 전역(global) 구성에서 이들을 확인(resolves)합니다. 이는 작업별로 덮어쓰기 설정(override)이 적용되었을 때만 작업 레코드(job record)에 표시됩니다.

저장소는 원자적 파일 쓰기(atomic file writes)를 사용하므로 쓰기가 중단되더라도 부분적으로 쓰인 작업 파일이 남지 않습니다.

## 독립적인(Self-contained) 프롬프트의 중요성

:::warning 중요
Cron 작업은 완전히 새로운 에이전트 세션에서 실행됩니다. 프롬프트에는 첨부된 스킬들로 제공되지 않는, 에이전트가 필요로 하는 모든 것이 반드시 포함되어야 합니다.
:::

**나쁨 (BAD):** `"서버 이슈 확인해봐"`

**좋음 (GOOD):** `"192.168.1.100 서버에 'deploy' 사용자로 SSH 접속하고, 'systemctl status nginx' 명령으로 nginx가 실행 중인지 확인한 다음, https://example.com 이 HTTP 200 응답을 반환하는지 검증하세요."`

## 보안 (Security)

예약된 작업의 프롬프트는 생성 및 업데이트 시에 프롬프트-인젝션(prompt-injection) 및 자격증명-유출(credential-exfiltration) 패턴에 대한 스캔을 거칩니다. 보이지 않는 유니코드 트릭(invisible Unicode tricks), SSH 백도어 시도, 또는 명백한 비밀-유출 페이로드(secret-exfiltration payloads)를 포함한 프롬프트는 차단됩니다.
