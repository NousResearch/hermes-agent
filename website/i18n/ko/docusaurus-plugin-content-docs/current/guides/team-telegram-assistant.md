---
sidebar_position: 4
title: "튜토리얼: 팀용 Telegram 어시스턴트"
description: "팀 전체가 코드 지원, 리서치, 시스템 관리 등을 위해 사용할 수 있는 Telegram 봇을 설정하는 단계별 가이드"
---

# 팀용 Telegram 어시스턴트 설정

이 튜토리얼에서는 여러 팀원이 사용할 수 있는 Hermes Agent 기반 Telegram 봇을 설정하는 방법을 안내합니다. 이 과정을 마치면 팀은 코드, 리서치, 시스템 관리 및 그 밖의 다양한 작업에 도움을 요청할 수 있는 공유 AI 어시스턴트를 갖게 됩니다. 사용자별 인증으로 안전하게 보호됩니다.

## 만들게 될 것

다음 기능을 갖춘 Telegram 봇입니다.

- **승인된 모든 팀원**이 DM으로 도움을 요청할 수 있습니다 — 코드 리뷰, 리서치, 셸 명령, 디버깅
- **서버에서 실행**되며 모든 도구에 접근할 수 있습니다 — 터미널, 파일 편집, 웹 검색, 코드 실행
- **사용자별 세션** — 각자 별도의 대화 컨텍스트를 가집니다
- **기본적으로 안전** — 승인된 사용자만 상호작용할 수 있으며, 두 가지 인증 방법을 제공합니다
- **예약 작업** — 일일 스탠드업, 상태 점검, 리마인더를 팀 채널로 전달합니다

---

## 사전 요구 사항

시작하기 전에 다음을 준비하세요.

- **서버 또는 VPS에 설치된 Hermes Agent** (노트북이 아님 — 봇은 계속 실행되어야 합니다). 아직 설치하지 않았다면 [설치 가이드](/getting-started/installation)를 따르세요.
- **자신의 Telegram 계정** (봇 소유자)
- **구성된 LLM 제공업체** — 최소한 OpenAI, Anthropic 또는 지원되는 다른 제공업체의 API 키를 `~/.hermes/.env`에 설정해야 합니다

:::tip
게이트웨이를 실행하는 데는 월 5달러짜리 VPS면 충분합니다. Hermes 자체는 가볍고, 비용이 드는 것은 원격으로 수행되는 LLM API 호출입니다.
::

---

## 1단계: Telegram 봇 만들기

모든 Telegram 봇은 봇을 만들기 위한 Telegram 공식 봇인 **@BotFather**에서 시작합니다.

1. **Telegram을 열고** `@BotFather`를 검색하거나 [t.me/BotFather](https://t.me/BotFather)로 이동합니다.

2. **`/newbot`을 보냅니다** — BotFather가 다음 두 가지를 묻습니다.
   - **표시 이름** — 사용자가 보게 될 이름입니다 (예: `Team Hermes Assistant`)
   - **사용자 이름** — `bot`으로 끝나야 합니다 (예: `myteam_hermes_bot`)

3. **봇 토큰을 복사합니다** — BotFather가 다음과 비슷한 내용을 답장합니다.
   ```
   Use this token to access the HTTP API:
   7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...
   ```
   이 토큰을 저장하세요 — 다음 단계에서 필요합니다.

4. **설명을 설정합니다** (선택 사항이지만 권장):
   ```
   /setdescription
   ```
   봇을 선택한 다음 다음과 비슷한 내용을 입력합니다.
   ```
   Team AI assistant powered by Hermes Agent. DM me for help with code, research, debugging, and more.
   ```

5. **봇 명령을 설정합니다** (선택 사항 — 사용자에게 명령 메뉴를 제공합니다):
   ```
   /setcommands
   ```
   봇을 선택한 다음 다음을 붙여 넣습니다.
   ```
   new - Start a fresh conversation
   model - Show or change the AI model
   status - Show session info
   help - Show available commands
   stop - Stop the current task
   ```

:::warning
봇 토큰을 안전하게 보관하세요. 토큰을 가진 사람은 누구나 봇을 제어할 수 있습니다. 토큰이 유출되었다면 BotFather에서 `/revoke`를 사용해 새 토큰을 생성하세요.
:::

---

## 2단계: 게이트웨이 구성

대화형 설정 마법사(권장)를 사용하거나 수동으로 구성할 수 있습니다.

### 옵션 A: 대화형 설정 (권장)

```bash
hermes gateway setup
```

화살표 키로 선택하면서 모든 과정을 진행합니다. **Telegram**을 선택하고 봇 토큰을 붙여 넣은 다음, 메시지가 표시되면 사용자 ID를 입력하세요.

### 옵션 B: 수동 구성

다음 줄을 `~/.hermes/.env`에 추가합니다.

```bash
# Telegram bot token from BotFather
TELEGRAM_BOT_TOKEN=7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...

# Your Telegram user ID (numeric)
TELEGRAM_ALLOWED_USERS=123456789
```

### 사용자 ID 찾기

Telegram 사용자 ID는 숫자 값이며 사용자 이름이 아닙니다. 다음과 같이 찾을 수 있습니다.

1. Telegram에서 [@userinfobot](https://t.me/userinfobot)에게 메시지를 보냅니다.
2. 봇이 숫자로 된 사용자 ID에 즉시 답장합니다.
3. 해당 숫자를 `TELEGRAM_ALLOWED_USERS`에 복사합니다.

:::info
Telegram 사용자 ID는 `123456789`와 같은 영구적인 숫자입니다. 변경될 수 있는 `@username`과는 다릅니다. 허용 목록에는 항상 숫자 ID를 사용하세요.
:::

---

## 3단계: 게이트웨이 시작

### 빠른 테스트

먼저 포그라운드에서 게이트웨이를 실행해 모든 것이 작동하는지 확인합니다.

```bash
hermes gateway
```

다음과 비슷한 출력이 표시되어야 합니다.

```
[Gateway] Starting Hermes Gateway...
[Gateway] Telegram adapter connected
[Gateway] Cron scheduler started (tick every 60s)
```

Telegram을 열고 봇을 찾은 다음 메시지를 보내세요. 답장이 오면 정상적으로 작동하는 것입니다. 중지하려면 `Ctrl+C`를 누르세요.

### 운영 환경: 서비스로 설치

재부팅 후에도 유지되는 배포를 위해 다음을 실행합니다.

```bash
hermes gateway install
sudo hermes gateway install --system   # Linux only: boot-time system service
```

이 명령은 백그라운드 서비스를 생성합니다. 기본적으로 Linux에서는 사용자 수준의 **systemd** 서비스, macOS에서는 **launchd** 서비스가 생성되며, `--system`을 전달하면 부팅 시 시작되는 Linux 시스템 서비스가 생성됩니다.

```bash
# Linux — manage the default user service
hermes gateway start
hermes gateway stop
hermes gateway status

# View live logs
journalctl --user -u hermes-gateway -f

# Keep running after SSH logout
sudo loginctl enable-linger $USER

# Linux servers — explicit system-service commands
sudo hermes gateway start --system
sudo hermes gateway status --system
journalctl -u hermes-gateway -f
```

```bash
# macOS — manage the service
hermes gateway start
hermes gateway stop
tail -f ~/.hermes/logs/gateway.log
```

:::tip macOS PATH
launchd plist는 설치 시점의 셸 PATH를 저장하므로 게이트웨이 하위 프로세스가 Node.js와 ffmpeg 같은 도구를 찾을 수 있습니다. 나중에 새 도구를 설치했다면 `hermes gateway install`을 다시 실행해 plist를 업데이트하세요.
:::

### 실행 여부 확인

```bash
hermes gateway status
```

그런 다음 Telegram에서 봇에 테스트 메시지를 보내세요. 몇 초 안에 답장을 받아야 합니다.

---

## 4단계: 팀 액세스 설정

이제 팀원에게 액세스 권한을 부여합니다. 두 가지 방법이 있습니다.

### 방법 A: 정적 허용 목록

각 팀원의 Telegram 사용자 ID를 수집한 뒤([@userinfobot](https://t.me/userinfobot)에게 메시지를 보내도록 안내) 쉼표로 구분된 목록으로 추가합니다.

```bash
# In ~/.hermes/.env
TELEGRAM_ALLOWED_USERS=123456789,987654321,555555555
```

변경 후 게이트웨이를 다시 시작합니다.

```bash
hermes gateway stop && hermes gateway start
```

### 방법 B: DM 페어링 (팀에 권장)

DM 페어링은 더 유연합니다 — 미리 사용자 ID를 수집할 필요가 없습니다. 작동 방식은 다음과 같습니다.

1. **팀원이 봇에 DM을 보냅니다** — 허용 목록에 없으므로 봇이 일회성 페어링 코드를 답장합니다.
   ```
   🔐 Pairing code: XKGH5N7P
   Send this code to the bot owner for approval.
   ```

2. **팀원이 코드를 전달합니다** (Slack, 이메일, 직접 만남 등 어떤 채널이든 가능)

3. **서버에서 승인합니다**.
   ```bash
   hermes pairing approve telegram XKGH5N7P
   ```

4. **접속 완료** — 봇이 즉시 해당 팀원의 메시지에 답하기 시작합니다.

**페어링된 사용자 관리:**

```bash
# See all pending and approved users
hermes pairing list

# Revoke someone's access
hermes pairing revoke telegram 987654321

# Clear expired pending codes
hermes pairing clear-pending
```

:::tip
DM 페어링은 새 사용자를 추가할 때 게이트웨이를 다시 시작할 필요가 없으므로 팀에 이상적입니다. 승인은 즉시 적용됩니다.
:::

### 보안 고려 사항

- **터미널 액세스가 있는 봇에는 절대로 `GATEWAY_ALLOW_ALL_USERS=true`를 설정하지 마세요** — 봇을 찾은 누구나 서버에서 명령을 실행할 수 있습니다
- 페어링 코드는 **1시간** 후 만료되며 암호학적으로 안전한 난수를 사용합니다
- 속도 제한으로 무차별 대입 공격을 방지합니다: 사용자당 10분에 1회 요청, 플랫폼당 대기 중인 코드 최대 3개
- 승인에 5회 실패하면 플랫폼이 1시간 동안 잠깁니다
- 모든 페어링 데이터는 `chmod 0600` 권한으로 저장됩니다

---

## 5단계: 봇 구성

### 홈 채널 설정

**홈 채널**은 봇이 cron 작업 결과와 사전 알림 메시지를 전달하는 곳입니다. 홈 채널이 없으면 예약 작업이 출력을 보낼 곳이 없습니다.

**옵션 1:** 봇이 참여한 Telegram 그룹이나 채팅에서 `/sethome` 명령을 사용합니다.

**옵션 2:** `~/.hermes/.env`에서 수동으로 설정합니다.

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="Team Updates"
```

채널 ID를 찾으려면 [@userinfobot](https://t.me/userinfobot)을 그룹에 추가하세요 — 그룹의 채팅 ID를 알려 줍니다.

### 도구 진행 상황 표시 구성

봇이 도구를 사용할 때 표시하는 세부 정보의 양을 제어합니다. `~/.hermes/config.yaml`에서 설정합니다.

```yaml
display:
  tool_progress: new    # off | new | all | verbose
```

| 모드 | 표시 내용 |
|------|---------|
| `off` | 도구 활동 없이 깔끔한 응답만 표시 |
| `new` | 새 도구 호출마다 간단한 상태 표시 (메시징에 권장) |
| `all` | 세부 정보와 함께 모든 도구 호출 표시 |
| `verbose` | 명령 결과를 포함한 전체 도구 출력 |

사용자는 채팅에서 `/verbose` 명령을 사용해 세션별로 이 설정을 변경할 수도 있습니다.

### SOUL.md로 성격 설정

`~/.hermes/SOUL.md`를 편집해 봇의 대화 방식을 사용자 지정합니다.

전체 가이드는 [Hermes에서 SOUL.md 사용하기](/guides/use-soul-with-hermes)를 참조하세요.

```markdown
# Soul
You are a helpful team assistant. Be concise and technical.
Use code blocks for any code. Skip pleasantries — the team
values directness. When debugging, always ask for error logs
before guessing at solutions.
```

### 프로젝트 컨텍스트 추가

팀이 특정 프로젝트를 진행한다면 컨텍스트 파일을 만들어 봇이 기술 스택을 알 수 있도록 하세요.

```markdown
<!-- ~/.hermes/AGENTS.md -->
# Team Context
- We use Python 3.12 with FastAPI and SQLAlchemy
- Frontend is React with TypeScript
- CI/CD runs on GitHub Actions
- Production deploys to AWS ECS
- Always suggest writing tests for new code
```

:::info
컨텍스트 파일은 모든 세션의 시스템 프롬프트에 삽입됩니다. 간결하게 유지하세요 — 모든 문자는 토큰 예산을 차감합니다.
:::

---

## 6단계: 예약 작업 설정

게이트웨이가 실행 중이면 팀 채널로 결과를 전달하는 반복 작업을 예약할 수 있습니다.

### 일일 스탠드업 요약

Telegram에서 봇에게 다음과 같이 메시지를 보냅니다.

```
Every weekday at 9am, check the GitHub repository at
github.com/myorg/myproject for:
1. Pull requests opened/merged in the last 24 hours
2. Issues created or closed
3. Any CI/CD failures on the main branch
Format as a brief standup-style summary.
```

에이전트가 cron 작업을 자동으로 만들고, 요청한 채팅(또는 홈 채널)으로 결과를 전달합니다.

### 서버 상태 점검

```
Every 6 hours, check disk usage with 'df -h', memory with 'free -h',
and Docker container status with 'docker ps'. Report anything unusual —
partitions above 80%, containers that have restarted, or high memory usage.
```

### 예약 작업 관리

```bash
# From the CLI
hermes cron list          # View all scheduled jobs
hermes cron status        # Check if scheduler is running

# From Telegram chat
/cron list                # View jobs
/cron remove <job_id>     # Remove a job
```

:::warning
Cron 작업 프롬프트는 이전 대화의 기억이 전혀 없는 완전히 새로운 세션에서 실행됩니다. 각 프롬프트에 에이전트가 필요로 하는 모든 컨텍스트 — 파일 경로, URL, 서버 주소, 명확한 지침 — 를 포함하세요.
:::

---

## 운영 환경 팁

### 안전을 위해 Docker 사용

공유 팀 봇에서는 Docker를 터미널 백엔드로 사용해 에이전트 명령이 호스트가 아닌 컨테이너에서 실행되도록 하세요.

```bash
# In ~/.hermes/.env
TERMINAL_ENV=docker
TERMINAL_DOCKER_IMAGE=nikolaik/python-nodejs:python3.11-nodejs20
```

또는 `~/.hermes/config.yaml`에 다음을 설정합니다.

```yaml
terminal:
  backend: docker
  container_cpu: 1
  container_memory: 5120
  container_persistent: true
```

이렇게 하면 누군가 봇에게 파괴적인 작업을 요청하더라도 호스트 시스템이 보호됩니다.

### 게이트웨이 모니터링

```bash
# Check if the gateway is running
hermes gateway status

# Watch live logs (Linux)
journalctl --user -u hermes-gateway -f

# Watch live logs (macOS)
tail -f ~/.hermes/logs/gateway.log
```

### Hermes 최신 상태 유지

Telegram에서 봇에게 `/update`를 보내 최신 버전을 가져오고 다시 시작하도록 하세요. 또는 서버에서 실행합니다.

```bash
hermes update
hermes gateway stop && hermes gateway start
```

### 로그 위치

| 항목 | 위치 |
|------|------|
| 게이트웨이 로그 | `journalctl --user -u hermes-gateway` (Linux) 또는 `~/.hermes/logs/gateway.log` (macOS) |
| Cron 작업 출력 | `~/.hermes/cron/output/{job_id}/{timestamp}.md` |
| Cron 작업 정의 | `~/.hermes/cron/jobs.json` |
| 페어링 데이터 | `~/.hermes/pairing/` |
| 세션 기록 | `~/.hermes/sessions/` |

---

## 더 알아보기

이제 작동하는 팀용 Telegram 어시스턴트를 갖추었습니다. 다음 단계로 다음을 살펴보세요.

- **[보안 가이드](/user-guide/security)** — 인증, 컨테이너 격리, 명령 승인 자세히 알아보기
- **[메시징 게이트웨이](/user-guide/messaging)** — 게이트웨이 아키텍처, 세션 관리, 채팅 명령 전체 레퍼런스
- **[Telegram 설정](/user-guide/messaging/telegram)** — 음성 메시지와 TTS를 포함한 플랫폼별 세부 정보
- **[예약 작업](/user-guide/features/cron)** — 전달 옵션과 cron 표현식을 사용한 고급 cron 예약
- **[컨텍스트 파일](/user-guide/features/context-files)** — 프로젝트 지식을 위한 AGENTS.md, SOUL.md, .cursorrules
- **[성격](/user-guide/features/personality)** — 기본 제공 성격 프리셋과 사용자 지정 페르소나 정의
- **플랫폼 추가** — 같은 게이트웨이에서 [Discord](/user-guide/messaging/discord), [Slack](/user-guide/messaging/slack), [WhatsApp](/user-guide/messaging/whatsapp)을 동시에 실행할 수 있습니다

---

*질문이나 문제가 있나요? GitHub에서 이슈를 열어 주세요 — 기여를 환영합니다.*
