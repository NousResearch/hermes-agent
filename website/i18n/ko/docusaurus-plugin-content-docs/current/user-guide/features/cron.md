---
sidebar_position: 5
title: "예약 작업 (Cron)"
description: "자연어로 자동화된 작업을 예약하고, 하나의 cron 도구로 관리하며, 하나 이상의 스킬을 연결하세요"
---

# 예약 작업 (Cron)

자연어나 cron 표현식을 사용해 작업이 자동으로 실행되도록 예약하세요. Hermes는 별도의 schedule/list/remove 도구 대신 작업 기반 작업을 제공하는 단일 `cronjob` 도구를 통해 cron 관리를 노출합니다.

## 현재 cron으로 할 수 있는 작업

Cron 작업은 다음을 수행할 수 있습니다.

- 일회성 또는 반복 작업 예약
- 작업 일시 중지, 재개, 편집, 트리거 및 제거
- 작업에 스킬을 0개, 1개 또는 여러 개 연결
- 결과를 원본 채팅, 로컬 파일 또는 구성된 플랫폼 대상으로 전달
- 일반적인 정적 도구 목록을 사용하는 새 에이전트 세션에서 실행
- **에이전트 없음 모드**에서 실행 — 일정에 따라 스크립트를 실행하고 stdout을 그대로 전달하며 LLM은 전혀 관여하지 않음 (아래 [에이전트 없음 모드](#no-agent-mode-script-only-jobs) 섹션 참고)

이 모든 기능은 `cronjob` 도구를 통해 Hermes 자체에서 사용할 수 있으므로, CLI 없이도 일반 언어로 요청해 작업을 만들고, 일시 중지하고, 편집하고, 제거할 수 있습니다.

:::tip
**cron 작업은 어떤 모델로 실행되나요?** 실행 시점의 결정 순서는 작업별 고정 → `cron.model`의 `config.yaml` 설정 → `hermes model`의 전역 기본값입니다.

- **작업별 고정** — 대시보드, `hermes cron create/edit --model … --provider …` 또는 `~/.hermes/cron/jobs.json` 편집을 통해 *사용자*가 설정합니다. 한 번 설정하면 변경할 때까지 유지됩니다. 에이전트의 `cronjob` 도구는 작업별 모델을 설정하거나 변경할 수 없습니다 — 추론 고정은 사용자가 관리합니다.
- **`cron.model` / `cron.model_provider`** — cron 작업군의 기본값입니다. 고정되지 않은 모든 작업은 채팅 모델과 무관하게 이 모델로 실행됩니다. 한 번 설정하면 (`hermes config set cron.model <name>`) `hermes model` 또는 `/model`로 채팅 모델을 바꿔도 cron 작업군에는 영향을 주지 않습니다.
- **전역 기본값** — 위 두 설정이 모두 없을 때만 작업이 `hermes model`을 따릅니다. 이 경우 Hermes는 생성 시점의 provider와 모델을 **스냅샷**하며, 이후 전역 기본값이 변경되면 작업이 **fail closed**됩니다. 즉 실행을 건너뛰고, 추론 호출을 하지 않으며, 한 번만 알립니다 — 사용자가 조치하거나 구성이 복원될 때까지 이후 틱에서도 작업은 건너뛴 채 조용히 유지됩니다 (#44585). 반복 작업이나 그 밖의 재실행 가능한 작업을 계속 실행하려면 provider/model을 명시적으로 고정하세요 (`cronjob action=update job_id=… provider=… model=…`). 이미 소비된 유한 일회성 작업은 업데이트할 수 없으므로, 대신 provider와 model을 명시한 새로운 미래 일회성 작업을 만드세요. 이렇게 하면 방치된 작업이 유료 provider/model 전환을 조용히 물려받는 일을 막을 수 있습니다. `cron.model` (또는 작업별 고정)을 설정하는 것이 cron 비용을 라우팅하는 명시적인 방법이며, 해당 설정이 적용되는 축에는 drift guard가 작동하지 않습니다. 반대로 고정되지 않은 작업이 변경되는 전역 기본값을 따르도록 하려면 [drift guard 비활성화](#letting-unpinned-jobs-track-global-defaults)를 참고하세요.

`hermes setup --portal`은 OAuth 갱신이 자동으로 이루어지므로 방치된 실행에 가장 간편한 옵션입니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

:::warning
Cron 실행 세션은 재귀적으로 더 많은 cron 작업을 만들 수 없습니다. 무한 예약 루프를 방지하기 위해 Hermes는 cron 실행 중 cron 관리 도구를 비활성화합니다.
:::

## 예약 작업 만들기

### `/cron`으로 채팅에서 만들기

```bash
/cron add 30m "Remind me to check the build"
/cron add "every 2h" "Check server status"
/cron add "every 1h" "Summarize new feed items" --skill blogwatcher
/cron add "every 1h" "Use both skills and combine the result" --skill blogwatcher --skill maps
```

### 독립 실행형 CLI에서 만들기

```bash
hermes cron create "every 2h" "Check server status"
hermes cron create "every 1h" "Summarize new feed items" --skill blogwatcher
hermes cron create "every 1h" "Use both skills and combine the result" \
  --skill blogwatcher \
  --skill maps \
  --name "Skill combo"
```

### 자연스러운 대화로 만들기

Hermes에 평소처럼 요청하세요.

```text
Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

Hermes는 내부적으로 통합 `cronjob` 도구를 사용합니다.

## 디스패치 전 구성 검증

예약 실행을 위한 에이전트 구성 요소를 만들기 전에 스케줄러는 작업 구성이 실제로 성공적인 실행을 만들 수 있는지 검증합니다.

- provider API 키가 확인되는지 검증합니다 (`fallback_providers` 체인이 구성된 경우에는 fallback 경로가 기본 키 누락을 복구할 수 있으므로 건너뜁니다).
- 연결된 스킬이 준비되었는지 검증합니다 (필수 환경 변수, 명령 또는 credential 파일 누락 여부).
- 전달 플랫폼 대상이 알려져 있고 gateway credential이 구성되어 있는지 검증합니다 (`local`/`origin` 대상은 확인하지 않습니다).

검증에 실패하면 작업의 `last_status`가 `blocked_config`가 되고, 알림이 한 번 전달되며 (틱마다 반복되지 않음), **LLM 호출이 이루어지지 않습니다** — 잘못 구성된 작업은 토큰을 사용하지 않습니다. 다음 정상 실행에서 차단 상태가 해제되므로, 이후 구성이 다시 잘못되면 알림이 다시 전달됩니다.

검증을 비활성화하고 이전 동작으로 되돌리려면 (실행이 진행된 후 실행 중 실패):

```yaml
cron:
  preflight: false
```

또는 `hermes config set cron.preflight false`를 실행합니다.

## 고정되지 않은 작업이 전역 기본값을 따르도록 허용

모델/provider drift guard는 기본적으로 활성화되어 있습니다. 고정되지 않은 cron 작업이 모든 전역 모델 또는 provider 변경을 의도적으로 따르도록 하려면 `config.yaml`에서 비활성화하세요.

```yaml
cron:
  model_drift_guard: false
```

또는 구성 명령을 사용합니다.

```bash
hermes config set cron.model_drift_guard false
```

이렇게 하면 전역 추론 설정이 변경될 때 표시되는 경고와 런타임 차단이 모두 비활성화됩니다. 기존 스냅샷은 계속 저장되므로 옵션을 다시 `true`로 설정하면 작업을 다시 만들지 않고도 보호 기능을 다시 활성화할 수 있습니다.

:::warning
guard를 비활성화하면 방치된 고정되지 않은 작업이 변경된 전역 기본값을 즉시 물려받습니다. 따라서 유료 provider나 모델로 전환하면 예약 실행마다 비용이 발생할 수 있습니다.
:::

## 스킬 기반 cron 작업

Cron 작업은 프롬프트를 실행하기 전에 하나 이상의 스킬을 로드할 수 있습니다.

### 단일 스킬

```python
cronjob(
    action="create",
    skill="blogwatcher",
    prompt="Check the configured feeds and summarize anything new.",
    schedule="0 9 * * *",
    name="Morning feeds",
)
```

### 여러 스킬

스킬은 순서대로 로드됩니다. 프롬프트는 이러한 스킬 위에 추가되는 작업 지침이 됩니다.

```python
cronjob(
    action="create",
    skills=["blogwatcher", "maps"],
    prompt="Look for new local events and interesting nearby places, then combine them into one short brief.",
    schedule="every 6h",
    name="Local brief",
)
```

전체 스킬 텍스트를 cron 프롬프트에 직접 넣지 않고도 예약된 에이전트가 재사용 가능한 워크플로를 상속하도록 만들고 싶을 때 유용합니다.

## 프로젝트 디렉터리 안에서 작업 실행

Cron 작업은 기본적으로 어떤 저장소와도 분리되어 실행됩니다 — `AGENTS.md`, `CLAUDE.md` 또는 `.cursorrules`가 로드되지 않으며, terminal / file / code-exec 도구는 gateway가 시작된 작업 디렉터리에서 실행됩니다. `--workdir` (CLI) 또는 `workdir=` (도구 호출)을 전달해 이를 변경하세요.

```bash
# Standalone CLI (schedule and prompt are positional)
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# From a chat, via the cronjob tool
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="Audit open PRs, summarize CI health, and post to #eng",
)
```

`workdir`가 설정되면:

- 해당 디렉터리의 `AGENTS.md`, `CLAUDE.md`, `.cursorrules`가 시스템 프롬프트에 주입됩니다 (대화형 CLI와 동일한 검색 순서).
- `terminal`, `read_file`, `write_file`, `patch`, `search_files`, `execute_code`가 모두 해당 디렉터리를 작업 디렉터리로 사용합니다.
- 경로는 존재하는 절대 디렉터리여야 합니다 — 상대 경로와 존재하지 않는 디렉터리는 생성/업데이트 시 거부됩니다.
- 편집 시 `--workdir ""` (또는 도구에서 `workdir=""`)을 전달하면 이를 지우고 이전 동작으로 되돌립니다.

:::note Serialization
`workdir`가 있는 작업은 스케줄러 틱에서 병렬 풀을 사용하지 않고 순차적으로 실행됩니다. 이는 cron worker가 프로세스 전역 terminal 상태를 통해 작업 디렉터리를 적용하기 때문이며, 두 개의 workdir 작업을 동시에 실행하면 서로의 cwd가 손상됩니다. workdir이 없는 작업은 이전처럼 계속 병렬 실행됩니다.
:::

## 작업 편집

변경만을 위해 작업을 삭제하고 다시 만들 필요는 없습니다.

:::tip 작업 참조
아래 (및 [수명 주기 작업](#lifecycle-actions))의 `<job_id>` 자리 표시자는 작업 이름(대소문자 구분 없음)도 허용합니다 — `morning-digest`는 기억나지만 hex ID가 기억나지 않을 때 편리합니다. 정확한 작업 ID가 이름 일치보다 우선하며, 참조가 ID가 아니고 하나 이상의 작업 이름과 일치하면 명령은 후보 ID를 출력하고 거부하므로 구분할 수 있습니다.
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

- 반복되는 `--skill`은 작업에 연결된 스킬 목록을 대체합니다.
- `--add-skill`은 기존 목록을 대체하지 않고 뒤에 추가합니다.
- `--remove-skill`은 지정한 연결 스킬을 제거합니다.
- `--clear-skills`는 연결된 모든 스킬을 제거합니다.

## 수명 주기 작업

이제 cron 작업은 생성/제거만이 아니라 더 완전한 수명 주기를 가집니다.

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

각 작업의 기능:

- `pause` — 작업을 유지하지만 예약을 중지합니다.
- `resume` — 작업을 다시 활성화하고 다음 미래 실행을 계산합니다.
- `run` — 다음 스케줄러 틱에 작업을 트리거합니다.
- `remove` — 작업을 완전히 삭제합니다.
- `edit` — 일정, 프롬프트, 전달 등을 수정합니다.

**이름 기반 조회.** 네 가지 변경 동사(`pause`, `resume`, `run`, `remove`, `edit`)와 에이전트의 `cronjob` 도구는 이제 hex ID 대신 작업 **이름**(대소문자 구분 없음)을 허용합니다. 에이전트와 CLI 모두 정확한 ID가 있으면 이를 우선하며, 이름이 여러 작업과 일치하면 후보 ID 전체 목록과 함께 거부되므로 명시적으로 하나를 선택할 수 있습니다. 이름은 고유하지 않으므로 이 보호 장치는 중요합니다 — 이름이 같은 두 작업이 있을 때 잘못된 작업을 조용히 변경하는 일을 막습니다.

## 에이전트가 관리하는 예약 (cron 작업을 관리하는 cron 작업)

기본적으로 스케줄러가 *실행한* 에이전트는 `cronjob` 도구를 사용할 수 없습니다 —
예약된 작업은 다른 작업을 생성, 편집 또는 제거할 수 없습니다. `config.yaml`에서 선택적으로 활성화하세요.

```yaml
cron:
  allow_agent_scheduling: true   # default: false
```

활성화하면 예약된 에이전트가 모든 채팅 세션과 마찬가지로 cron 테이블을 관리할 수 있습니다. 예약 작업 중 후속 일회성 작업을 예약하거나, 자체 실행 주기를 조정하거나, 전체 테이블을 조정하는 "cron librarian" 작업을 실행할 수 있습니다 (필요에 따라 목록 조회 후 업데이트/제거/생성). 다음 두 가지 속성이 이를 안전하게 유지합니다.

- **하나의 평평한 사용자 소유 테이블.** Cron 실행에서 생성된 작업은 특별한 소유권 없이 다른 모든 작업과 동일한 `jobs.json`에 저장됩니다 — 사용자가 직접 만든 것처럼 목록 조회, 편집 또는 제거할 수 있습니다.
- **끊어진 전달 없음.** Cron 실행은 일시적이므로 내부의 `deliver: origin`은 생성 시점에 생성한 작업 자체의 구체적인 대상(`platform:chat_id[:thread_id]`, 또는 생성한 작업이 아무 곳에도 전달하지 않으면 `local`)으로 확인됩니다. 예약된 에이전트가 만든 작업은 더 이상 존재하지 않는 세션으로 출력을 가리킬 수 없습니다. 명시적 대상(`local`, `all`, `telegram:<chat_id>`)은 입력한 그대로 적용됩니다.

매 실행마다 새 작업을 만드는 프롬프트보다는 기존 작업을 업데이트하는 프롬프트(먼저 목록을 조회한 후 ID로 업데이트)를 우선하세요.

## 작동 방식

**Cron 실행은 gateway 데몬이 처리합니다.** Gateway는 60초마다 스케줄러를 틱하여, 격리된 에이전트 세션에서 기한이 된 모든 작업을 실행합니다.

```bash
hermes gateway install     # Install as a user service
sudo hermes gateway install --system   # Linux: boot-time system service for servers
hermes gateway             # Or run in foreground

hermes cron list
hermes cron status
```

### Gateway 스케줄러 동작

각 틱에서 Hermes는 다음을 수행합니다.

1. `~/.hermes/cron/jobs.json`에서 작업을 로드합니다.
2. `next_run_at`을 현재 시간과 비교합니다.
3. 기한이 된 각 작업에 대해 새 `AIAgent` 세션을 시작합니다.
4. 필요하면 연결된 하나 이상의 스킬을 새 세션에 주입합니다.
5. 프롬프트를 끝까지 실행합니다.
6. 최종 응답을 전달합니다.
7. 실행 메타데이터와 다음 예약 시간을 업데이트합니다.

`~/.hermes/cron/.tick.lock`의 파일 잠금은 스케줄러 틱이 겹쳐 동일한 작업 배치를 두 번 실행하지 않도록 합니다.

### 실행 기록

Hermes는 executor 또는 provider 디스패치 전에 profile-local
`~/.hermes/cron/executions.db`에 클레임된 각 cron 시도를 기록합니다. 시도는 `claimed`, `running`을 거쳐 변경할 수 없는 하나의 최종 상태(`completed`, `failed` 또는 `unknown`)로 이동합니다. 재시작 후 Hermes는 원래 PID와 프로세스 시작 fingerprint로 소유자가 종료되었음을 입증할 수 있을 때만 중단된 시도를 `unknown`으로 표시합니다. 알 수 없는 시도는 감사 기록이며 자동으로 다시 실행되지 않습니다.

`hermes cron runs [job-id] --limit 20` (별칭: `history`)로 최근 시도를 확인하세요. 터미널 기록의 크기는 제한되며, 활성 시도는 절대 정리되지 않습니다. 이 ledger는 빠른 백업에 포함됩니다.

## 전달 옵션

작업을 예약할 때 출력이 전달될 위치를 지정합니다.

| 옵션 | 설명 | 예시 |
|--------|-------------|---------|
| `"origin"` | 작업이 생성된 곳으로 돌려보냄 | 메시징 플랫폼의 기본값 |
| `"local"` | 로컬 파일에만 저장 (`~/.hermes/cron/output/`) | CLI의 기본값 |
| `"telegram"` | Telegram 홈 채널 | `TELEGRAM_HOME_CHANNEL` 사용 |
| `"telegram:123456"` | ID로 지정한 Telegram 채팅 | 직접 전달 |
| `"telegram:-100123:17585"` | 지정한 Telegram 토픽 | `chat_id:thread_id` 형식 |
| `"discord"` | Discord 홈 채널 | `DISCORD_HOME_CHANNEL` 사용 |
| `"discord:#engineering"` | 지정한 Discord 채널 | 채널 이름으로 지정 |
| `"slack"` | Slack 홈 채널 | |
| `"whatsapp"` | WhatsApp 홈 | |
| `"signal"` | Signal | |
| `"matrix"` | Matrix 홈 룸 | |
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
| `"all"` | 연결된 모든 홈 채널로 fan out | 실행 시점에 확인 |
| `"telegram,discord"` | 지정한 채널 집합으로 fan out | 쉼표로 구분한 목록 |
| `"origin,all"` | 원본과 연결된 다른 모든 채널로 전달 | 토큰 조합 |

에이전트의 최종 응답은 구성된 `deliver:` 대상에 자동으로 전달됩니다 — 에이전트가 직접 메시지를 보내지 않으므로 cron 프롬프트에서 호출할 것이 없습니다.

### 라우팅 의도 (`all`)

`all`을 사용하면 이름을 일일이 열거하지 않고도 구성된 모든 메시징 채널로 하나의 cron 작업을 보낼 수 있습니다. **실행 시점에 확인**되므로 Telegram을 연결하기 전에 만든 작업도 `TELEGRAM_HOME_CHANNEL`을 설정한 다음 틱에서 Telegram을 사용합니다.

의미: `all`은 구성된 홈 채널이 있는 모든 플랫폼으로 확장됩니다. 0개여도 문제없습니다. 작업은 단순히 전달 대상 없이 실행되고, upstream에 전달 실패로 기록됩니다.

`all`은 명시적 대상과 조합할 수 있습니다. `origin,all`은 원본 채팅과 연결된 다른 모든 홈 채널로 전달하며, `(platform, chat_id, thread_id)`를 기준으로 중복을 제거합니다.

### Telegram cron 토픽 (`TELEGRAM_CRON_THREAD_ID`)

Telegram 토픽 모드가 활성화되면 루트 DM은 시스템 로비로 예약됩니다 — 그곳으로 보낸 응답은 로비 알림으로 반려되고 `reply_to_message_id`가 삭제되므로, 메인 채팅에 도착한 cron 메시지에 답장할 수 없습니다.

대신 cron을 전용 포럼 토픽으로 지정하세요.

1. Telegram에서 봇 DM을 열고 `Cron`과 같은 이름의 토픽을 만드세요. 토픽 헤더를 길게 누른 뒤 **링크 복사**를 선택하세요. 끝의 정수가 토픽의 `message_thread_id`입니다.
2. `TELEGRAM_CRON_THREAD_ID=<that id>`를 `.env`에 설정하세요.

이는 cron 전달에만 적용됩니다. `TELEGRAM_HOME_CHANNEL_THREAD_ID` (예: 재시작 알림 등 다른 곳에서 사용)는 변경되지 않습니다. 명시적 `deliver="telegram:chat_id:thread_id"` 대상은 계속 환경 변수를 우선합니다. 이제 cron 메시지에 대한 답장은 기존 토픽 세션에 도착하므로 바로 조치할 수 있습니다.

### 응답 래핑

기본적으로 전달되는 cron 출력은 수신자가 예약 작업에서 온 메시지임을 알 수 있도록 헤더와 푸터로 감쌉니다.

```
Cronjob Response: Morning feeds
-------------

<agent output here>

Note: The agent cannot see this message, and therefore cannot respond to it.
```

래퍼 없이 에이전트의 원시 출력을 전달하려면 `cron.wrap_response`를 `false`로 설정하세요.

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### 계속 가능한 작업 (cron 전달에 답장하기)

기본적으로 cron 전달은 fire-and-forget입니다. 메시지는 전송되지만 채팅의 대화 기록에는 포함되지 않으므로, 여기에 답장해도 에이전트는 자신이 무엇을 말했는지 알 수 없습니다. 작업을 **continuable**로 설정하면 전달된 brief가 답장할 수 있는 대화가 됩니다 — 에이전트가 "Task #2가 무엇이죠?"라고 묻는 대신 brief를 컨텍스트에 포함합니다.

선택 사항이며 **기본값은 꺼짐**입니다. 구성에서 전역으로 활성화하거나, `cronjob` 도구의 `attach_to_session`을 통해 작업별로 활성화할 수 있습니다 (해당 작업에 대해서는 전역 설정을 덮어씀).

```yaml
# ~/.hermes/config.yaml
cron:
  mirror_delivery: false   # set true to make cron deliveries continuable
```

동작은 작업 원본 채팅에 한정된 **스레드 우선** 방식입니다.

- **스레드를 지원하는 플랫폼** (Telegram 토픽, Discord/Slack 스레드): 각 전달마다 전용 스레드를 열고 brief를 해당 스레드의 세션에 주입하므로, 스레드에서 답장하면 전체 컨텍스트를 이어갑니다. 반복 작업 (예: 매일 brief)은 실행마다 새 스레드를 열어 각 전달의 후속 대화를 분리합니다.
- **DM 전용 플랫폼** (WhatsApp, Signal, SMS): 스레드가 없으므로 brief가 원본 DM 세션에 미러링됩니다 — DM 자체가 계속하기 위한 표면입니다.

원본 채팅만 건드립니다. fan-out / 브로드캐스트 대상(`all`, 명시적인 다른 채팅 전달)은 절대 continuable로 만들지 않습니다. 미러는 레이블이 있는 사용자 턴(`[Cron delivery: <task name>]`)으로 기록되어 모든 모델 provider에서 대화 기록의 교대 규칙을 안전하게 유지합니다.

#### 인채널 평면 계속하기 (Slack)

위의 스레드 우선 동작은 전달마다 전용 스레드를 만듭니다. continuable 작업이 **채널 타임라인에 평면적으로** 도착하도록 하려면 — 스레드 없이 — Slack **continuable surface**를 `in_channel`로 설정하세요.

```yaml
# ~/.hermes/config.yaml
slack:
  cron_continuable_surface: in_channel   # default: thread
  reply_in_thread: false                 # required pairing (see below)
  require_mention: false                 # so a plain reply continues the job
```

`in_channel` 모드에서는 brief가 일반적인 최상위 채널 메시지로 전달되고 (스레드는 열리지 않음), 채널의 공유 세션을 통해 답장이 작업을 계속합니다. 세 설정이 함께 작동합니다.

- **`cron_continuable_surface: in_channel`** — 전달 시 스레드 생성을 건너뜁니다.
- **`reply_in_thread: false`** (필수) — 봇이 답장에 평면적으로 답하고, brief가 주입된 동일한 전체 채널 세션에 연결되도록 합니다. 이 설정이 없으면 계속하기 자체는 작동하지만 스레드로 도착합니다 (안전하게 fallback하여 절대 답장을 삭제하지 않음 — gateway는 시작 시 경고를 기록하므로 불일치를 확인할 수 있음).
- **`require_mention: false`** (또는 채널을 `free_response_channels`에 추가) — 일반 메시지로 답장할 수 있도록 합니다. 그렇지 않으면 각 답장에서 `@` 멘션할 때만 봇이 깨어납니다.

계속하기가 **전체 채널** 세션이므로 공유됩니다. 채널의 다른 대화와 두 번째 continuable 인채널 작업이 동일한 롤링 대화에 참여합니다. 이는 "채널에서 평면적으로" 동작하는 것의 본질적인 절충이며, `reply_in_thread: false` 사용자가 이미 받아들이는 것과 같습니다. 각 전달의 후속 대화를 분리하려면 기본값인 `thread` 표면을 사용하세요.

이는 현재 Slack에서 지원하는 기능입니다. 다른 플랫폼은 해당 키를 허용하지만 `thread` 표면으로 fallback합니다 (계속하기 기본 요소가 다름). 선택은 플랫폼별이며 각 플랫폼 구성 아래에 설정합니다. gateway 측 구성 플래그이므로 `/restart`가 이를 적용하며 Slack 앱을 다시 설치할 필요가 없습니다.

:::note 1:1 DM
`cron_continuable_surface`는 **채널** 설정입니다 — 1:1 DM에는 스레드와 타임라인 중 선택할 구분이 없으며 (DM은 이미 평면적임), 따라서 이 키는 여기서 효과가 없습니다. DM cron 전달을 continuable로 만들지 여부는 별도의 기존 설정인 **`slack.dm_top_level_threads_as_sessions`**가 결정합니다.

- **`false`** — 모든 최상위 DM이 하나의 롤링 DM 세션을 공유하므로, continuable cron brief와 답장이 **동일한** 세션에 도착하고 작업이 컨텍스트에서 계속됩니다. DM에서 continuable cron에 원하는 설정입니다.
- **`true`** (기본값) — 각 최상위 DM 메시지가 자체 세션이므로, 전달된 brief에 답장하면 brief 기록이 없는 *새* 세션이 시작됩니다. 이 모드에서는 (cron이든 다른 평면 전달이든) 계속하기가 작동하지 않습니다.

따라서 1:1 DM으로 전달되는 continuable cron 작업에서는 `slack.dm_top_level_threads_as_sessions: false`를 설정하세요. DM에서는 `cron_continuable_surface`가 필요하지 않으며 무시됩니다.
:::

### 무음 억제

에이전트의 최종 응답에 `[SILENT]`가 포함되면 전달이 완전히 억제됩니다. 감사 목적으로 출력은 로컬에 계속 저장되지만 (`~/.hermes/cron/output/`), 전달 대상에는 메시지가 전송되지 않습니다.

이는 문제가 있을 때만 보고해야 하는 모니터링 작업에 유용합니다.

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

실패한 작업은 `[SILENT]` 마커가 있어도 항상 전달됩니다 — 성공한 실행만 무음 처리할 수 있습니다. 조용한 모니터링 작업에서는 보고할 내용이 없을 때 에이전트가 `[SILENT]`만 응답하도록 프롬프트하세요.

## 스크립트 시간 초과

`script` 매개변수로 연결된 사전 실행 스크립트의 기본 시간 초과는 3600초(1시간)입니다. 이는 **스크립트만** 제한합니다 — 스킬 기반 / LLM 기반 작업은 별도의 비활성 예산으로 실행되며 이 값의 제한을 받지 않습니다. 스크립트에 다른 제한이 필요하면 변경할 수 있습니다.

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 1800   # 30 minutes
```

또는 `HERMES_CRON_SCRIPT_TIMEOUT` 환경 변수를 설정하세요. 결정 순서는 env var → config.yaml → 기본값 3600초입니다.

Cron은 실행 후 세션과 에이전트 리소스 정리에도 제한을 둡니다. 이는 LLM 턴이 반환된 후 발생하므로 비활성 시간 초과와는 별개입니다. 기본값은 정리 작업당 10초입니다. 스토리지 또는 클라이언트 finalizer가 계속 반환하지 않으면 스케줄러가 오류를 기록하고 작업의 in-flight guard를 해제하며, 해당 작업이 영원히 건너뛰어지지 않도록 이후 실행이 디스패치되게 합니다.

```yaml
# ~/.hermes/config.yaml
cron:
  cleanup_timeout_seconds: 10
```

`cleanup_timeout_seconds: 0`으로 설정하면 기존의 무제한 정리 동작이 복원됩니다.

## 에이전트 없음 모드 (스크립트만 실행하는 작업)

LLM 추론이 필요 없는 반복 작업 — 고전적인 watchdog, 디스크/메모리 알림, heartbeat, CI ping — 에는 생성 시 `no_agent=True`를 전달하세요. 스케줄러가 일정에 따라 스크립트를 실행하고 stdout을 직접 전달하며 에이전트를 완전히 건너뜁니다.

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

의미:

- 스크립트 stdout(앞뒤 공백 제거) → 메시지로 그대로 전달됩니다.
- **빈 stdout → 무음 틱**, 전달하지 않습니다. 이는 "문제가 있을 때만 알리기" watchdog 패턴입니다.
- 0이 아닌 종료 코드 또는 시간 초과 → 오류 알림이 전달되므로 고장 난 watchdog이 조용히 실패할 수 없습니다.
- 마지막 줄의 `{"wakeAgent": false}` → 무음 틱 (LLM 작업이 사용하는 것과 같은 게이트).
- 토큰, 모델, provider fallback이 없음 — 작업은 추론 계층에 전혀 접근하지 않습니다.

`.sh` / `.bash` 파일은 사용 가능한 경우 `PATH`의 `bash`에서 실행되고, 그렇지 않으면 `/bin/bash`에서 실행됩니다 (Windows Git Bash에서 중요). 그 외 파일은 현재 Python 인터프리터(`sys.executable`)에서 실행됩니다. 스크립트는 `$HERMES_HOME/scripts/` 안에서 확인되어야 합니다 — 확인된 대상이 해당 디렉터리에 남아 있는 경우 상대 이름, 절대 경로 및 `~` 접두 경로가 허용됩니다. 해당 디렉터리 밖으로 나가는 경로는 거부됩니다. 하위 프로세스 환경은 정리됩니다 (`_sanitize_subprocess_env`). provider API credential과 Hermes가 관리하는 기타 secret은 cron 스크립트에 상속되지 않습니다.

### 에이전트가 이를 대신 설정합니다

`cronjob` 도구의 스키마는 Hermes에 `no_agent`를 직접 노출하므로, 채팅에서 watchdog을 설명하고 에이전트가 설정하도록 할 수 있습니다.

```text
Ping me on Telegram if RAM is over 85%, every 5 minutes.
```

Hermes는 `write_file`을 통해 확인 스크립트를 `~/.hermes/scripts/`에 작성한 다음 다음을 호출합니다.

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

메시지 내용이 스크립트로 완전히 결정되는 경우 (watchdog, 임계값 알림, heartbeat) `no_agent=True`를 자동으로 선택합니다. 동일한 도구를 사용해 에이전트가 작업을 일시 중지, 재개, 편집 및 제거할 수도 있으므로, 전체 수명 주기를 CLI를 건드리지 않고 채팅으로 처리할 수 있습니다.

작동 예시는 [스크립트 전용 Cron 작업 가이드](/guides/cron-script-only)를 참고하세요.

## `context_from`으로 작업 연결

Cron 작업은 격리된 세션에서 실행되므로 이전 실행을 기억하지 못합니다. 하지만 한 작업의 출력이 다음 작업에 정확히 필요한 경우가 있습니다. `context_from` 매개변수가 이 연결을 자동으로 구성합니다 — 작업 B의 프롬프트는 런타임에 작업 A의 가장 최근 출력이 컨텍스트로 앞에 추가된 상태가 됩니다.

```python
# Job 1: Collect raw data
cronjob(
    action="create",
    prompt="Fetch the top 10 AI/ML stories from Hacker News. Save them to ~/.hermes/data/briefs/raw.md in markdown format with title, URL, and score.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# Job 2: Triage — receives Job 1's output as context
# Get Job 1's ID from: cronjob(action="list")
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/raw.md. Score each story 1–10 for engagement potential and novelty. Output the top 5 to ~/.hermes/data/briefs/ranked.md.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# Job 3: Ship — receives Job 2's output as context
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/ranked.md. Write 3 tweet drafts (hook + body + hashtags). Deliver to telegram:7976161601.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**작동 방식:**

- 작업 2가 실행되면 Hermes는 `~/.hermes/cron/output/{job1_id}/*.md`에서 작업 1의 가장 최근 출력을 읽습니다.
- 해당 출력이 작업 2의 프롬프트 앞에 자동으로 추가됩니다.
- 작업 2는 "이 파일을 읽어라"를 하드코딩할 필요가 없습니다 — 콘텐츠를 컨텍스트로 받습니다.
- 체인은 어떤 길이든 가능합니다: 작업 1 → 작업 2 → 작업 3 → ...

**`context_from`이 허용하는 형식:**

| 형식 | 예시 |
|--------|---------|
| 단일 작업 ID (문자열) | `context_from="a1b2c3d4"` |
| 여러 작업 ID (목록) | `context_from=["job_a", "job_b"]` |

출력은 목록에 지정된 순서대로 연결됩니다.

**사용할 때:**

- 다단계 파이프라인 (수집 → 필터링 → 형식 지정 → 전달)
- 단계 N의 작업이 단계 N−1의 출력에 의존하는 종속 작업
- 하나의 작업이 여러 작업의 결과를 집계하는 fan-out/fan-in 패턴

## Provider 복구

Cron 작업은 구성된 fallback provider와 credential pool 순환을 상속합니다. 기본 API 키가 rate limit에 걸리거나 provider가 오류를 반환하면 cron 에이전트는 다음을 수행할 수 있습니다.

- `config.yaml`에 `fallback_providers` (또는 기존의 `fallback_model`)가 구성된 경우 **대체 provider로 fallback**
- [credential pool](/user-guide/configuration#credential-pool-strategies)의 다음 credential로 **같은 provider에서 순환**

따라서 높은 빈도나 피크 시간에 실행되는 cron 작업은 더 높은 복원력을 가집니다 — rate limit에 걸린 키 하나 때문에 전체 실행이 실패하지 않습니다.

## 일정 형식

에이전트의 최종 응답은 `deliver:` 대상에 자동으로 전달됩니다 — 에이전트가 더 이상 직접 메시지를 전송하지 않으므로 사용자에게 보이는 내용은 최종 응답에 넣으면 됩니다. **추가 또는 다른** 대상으로 전달하려면 에이전트가 직접 보내도록 하는 대신 cron 작업에 여러 `deliver:` 대상을 쉼표로 구분해 나열하세요 (예: `deliver: "telegram,discord"`).

### 상대 지연 (일회성)

```text
30m     → Run once in 30 minutes
2h      → Run once in 2 hours
1d      → Run once in 1 day
```

### 간격 (반복)

```text
every 30m    → Every 30 minutes
every 2h     → Every 2 hours
every 1d     → Every day
```

### Cron 표현식

```text
0 9 * * *       → Daily at 9:00 AM
0 9 * * 1-5     → Weekdays at 9:00 AM
0 */6 * * *     → Every 6 hours
30 8 1 * *      → First of every month at 8:30 AM
0 0 * * 0       → Every Sunday at midnight
```

### ISO 타임스탬프

```text
2026-03-15T09:00:00    → One-time at March 15, 2026 9:00 AM
```

## 반복 동작

| 일정 유형 | 기본 반복 | 동작 |
|--------------|----------------|----------|
| 일회성 (`30m`, 타임스탬프) | 1 | 한 번 실행 |
| 간격 (`every 2h`) | 계속 | 제거할 때까지 실행 |
| Cron 표현식 | 계속 | 제거할 때까지 실행 |

반복 횟수를 덮어쓸 수 있습니다.

```python
cronjob(
    action="create",
    prompt="...",
    schedule="every 2h",
    repeat=5,
)
```

## 프로그래밍 방식으로 작업 관리

에이전트용 API는 하나의 도구입니다.

```python
cronjob(action="create", ...)
cronjob(action="list")
cronjob(action="update", job_id="...")
cronjob(action="pause", job_id="...")
cronjob(action="resume", job_id="...")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

`update`에서는 `skills=[]`를 전달해 연결된 모든 스킬을 제거합니다.

### 수동 실행은 비동기입니다

`cronjob(action="run")`은 작업을 즉시 **백그라운드에서** 실행합니다 (`delegate_task`와 유사). 도구 호출은 핸들을 반환하고 즉시 끝나며, 작업 결과 — 성공/실패, 전달 대상, 다음 예약 실행, 출력 발췌 — 는 실행이 끝나면 새 메시지로 대화에 다시 들어옵니다. 그동안 에이전트(및 사용자)는 계속 작업할 수 있으며, 이미 실행 중인 작업은 중복 실행 대신 "already running"과 함께 거부됩니다.

`action="run"`에 `prompt`를 전달해 실행별 임시 컨텍스트를 주입할 수도 있습니다.

```python
cronjob(action="run", job_id="...", prompt="CONTEXT: focus on the EU region today")
```

컨텍스트는 해당 한 번의 실행에만 저장된 프롬프트 아래 `## Run Context` 헤더와 함께 추가됩니다 — 작업 정의에 저장되지 않으며 저장된 프롬프트와 동일한 prompt-injection 검사를 거칩니다.

분리된 결과를 받을 수 없는 런타임(일회성 `hermes -z`, CLI에서 실행한 `hermes cron run`, cron 자식 세션, Kanban worker)은 자동으로 동기 실행으로 fallback합니다.

## Cron 작업에서 사용할 수 있는 도구 세트

Cron은 각 작업을 연결된 채팅 플랫폼이 없는 새 에이전트 세션에서 실행합니다. 기본적으로 cron 에이전트는 `hermes tools`에서 **`cron` 플랫폼에 구성한 도구 세트**를 받습니다 — CLI 기본값도 아니고, 모든 도구도 아닙니다.

```bash
hermes tools
# → pick the "cron" platform in the curses UI
# → toggle toolsets on/off just like you would for Telegram/Discord/etc.
```

더 세밀한 작업별 제어는 `cronjob.create`의 `enabled_toolsets` 필드 (또는 기존 작업의 `cronjob.update`)를 통해 사용할 수 있습니다.

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # just web + file, no terminal/browser/etc.
        prompt="Summarize this week's AI news: ...")
```

작업에 `enabled_toolsets`가 설정되면 이것이 우선합니다. 그렇지 않으면 `hermes tools`의 cron 플랫폼 구성이 적용되고, 그마저 없으면 Hermes가 기본 제공 기본값으로 fallback합니다. 이는 비용 관리에 중요합니다. 모든 작은 "뉴스 가져오기" 작업의 모든 LLM 호출에 `browser`, `delegation`을 포함하면 도구 스키마 프롬프트가 커집니다.

### 에이전트 완전히 건너뛰기: `wakeAgent`

cron 작업에 사전 검사 스크립트를 연결하면 (`script=`를 통해) 스크립트가 런타임에 Hermes가 에이전트를 호출해야 하는지조차 결정할 수 있습니다. 다음 형식의 stdout 마지막 줄을 출력하세요.

```text
{"wakeAgent": false}
```

…그러면 cron이 해당 틱의 에이전트 실행을 완전히 건너뜁니다. 실제로 상태가 변경될 때만 LLM을 깨우면 되는 빈번한 폴링(1~5분마다)에 유용합니다 — 그렇지 않으면 내용이 없는 에이전트 턴에 계속 비용을 지불하게 됩니다.

```python
# pre-check script
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # skip this tick
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

`wakeAgent`가 없으면 기본값은 `true`입니다 (평소처럼 에이전트를 깨움).

#### 레시피: 저렴한 사전 실행 게이트

`wakeAgent` 게이트는 예약된 작업이 LLM 토큰을 사용할지 여부를 0달러로 결정할 수 있게 합니다. 세 가지 패턴이 대부분의 사용 사례를 다룹니다.

**파일 변경 게이트** — 마지막 성공 틱 이후 감시 중인 파일에 새 콘텐츠가 있을 때만 실행합니다. 스케줄러가 각 작업의 `last_run_at`을 기록하므로 파일의 mtime과 비교하세요.

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
        prompt="A new ~/data/feed.json has landed. Summarize what changed.")
```

**외부 플래그 게이트** — 다른 프로세스가 준비 상태를 알렸을 때만 실행합니다 (예: 배포 hook이 파일을 남기거나 CI 작업이 상태 저장소에 값을 설정).

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
        prompt="Run the nightly analysis over today's batch.")
```

**SQL 개수 게이트** — 자체 데이터베이스에서 처리할 새 행이 있을 때만 실행합니다. 스크립트는 개수를 `context`를 통해 에이전트에 전달할 수도 있으므로, 에이전트가 다시 조회하지 않아도 처리할 양을 알 수 있습니다.

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
        prompt="Summarize the new messages from the last 2 hours.")
```

동일한 패턴은 스크립트에서 조회할 수 있는 모든 데이터 소스 — Postgres, HTTP API, 자체 상태 저장소 — 에 적용할 수 있으므로 cron subsystem에 SQL evaluator를 내장할 필요가 없습니다.

:::tip
Hermes 자체의 `~/.hermes/state.db`는 릴리스마다 변경되는 내부 스키마입니다. 사전 실행 게이트에서 이를 조회하지 말고 자체 데이터베이스나 feed를 대상으로 하세요.
:::

기여: 이 레시피 모음은 병렬 메커니즘으로 sql/file/command 트리거 추가를 제안한 @iankar8의 [#2654](https://github.com/NousResearch/hermes-agent/pull/2654) 탐색에서 시작되었습니다. `script` + `wakeAgent` 게이트가 세 가지 경우를 모두 0달러로 이미 처리하므로, 작업은 문서로 반영되었습니다.

### 작업 연결: `context_from`

Cron 작업은 `context_from`에 다른 작업의 이름(또는 ID)을 하나 이상 나열해 가장 최근에 성공한 출력을 사용할 수 있습니다.

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="Write the daily digest using the outputs above.")
```

참조된 작업의 가장 최근 완료 출력은 이번 실행의 컨텍스트로 프롬프트 위에 주입됩니다. 각 upstream 항목은 유효한 작업 ID 또는 이름이어야 합니다 (`cronjob action="list"` 참고). 참고: 연결은 가장 최근의 *완료된* 출력을 읽으며, 같은 틱에서 실행 중인 upstream 작업을 기다리지 않습니다.

## 작업 저장소

작업은 `~/.hermes/cron/jobs.json`에 저장됩니다. 작업 실행의 출력은 `~/.hermes/cron/output/{job_id}/{timestamp}.md`에 저장됩니다.

작업 정의는 디스크의 일반 JSON입니다. `hermes update`, gateway 재시작 및 시스템 재부팅 후에도 유지됩니다. 재시작 중 실행 중이던 작업은 실행 ledger에서 `unknown`으로 표시됩니다 — 자동으로 재시도되지 않지만 작업의 다음 예약 틱은 정상적으로 실행됩니다. 자세한 내용은 [실행 기록](#execution-history)을 참고하세요.

:::tip
`jobs.json`을 직접 패치하지 말고 `cronjob` 도구, `hermes cron edit` 또는 `/cron`을 통해 에이전트에게 작업을 관리하도록 요청하세요. [파일 쓰기 안전성](../security.md#file-write-safety)이 경로를 차단하면 직접 편집이 조용히 실패할 수 있습니다 (예: `HERMES_WRITE_SAFE_ROOT`가 설정된 경우). 또한 [파일 변경 검증기](../configuration.md#file-mutation-verifier) footer가 아무것도 저장되지 않았음을 나타내는 권위 있는 신호입니다.
:::

작업은 `model`과 `provider`를 `null`로 저장할 수 있습니다. 이 필드가 생략되면 Hermes는 전역 구성에서 실행 시점에 이를 확인합니다. 작업별 재정의가 설정된 경우에만 작업 레코드에 나타납니다.

저장소는 원자적 파일 쓰기를 사용하므로 중단된 쓰기가 일부만 작성된 작업 파일을 남기지 않습니다.

## 독립적인 프롬프트가 여전히 중요한 이유

:::warning Important
Cron 작업은 완전히 새로운 에이전트 세션에서 실행됩니다. 프롬프트에는 연결된 스킬이 이미 제공하지 않는 모든 필수 정보를 포함해야 합니다.
:::

**나쁜 예:** `"Check on that server issue"`

**좋은 예:** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`

## 보안

예약 작업 프롬프트는 생성 및 업데이트 시 prompt-injection과 credential-exfiltration 패턴을 검사합니다. 보이지 않는 Unicode 트릭, SSH 백도어 시도 또는 명백한 secret-exfiltration payload가 포함된 프롬프트는 차단됩니다.
