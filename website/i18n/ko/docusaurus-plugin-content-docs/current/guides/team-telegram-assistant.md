---
sidebar_position: 4
title: "튜토리얼: 팀 Telegram 어시스턴트"
description: "전체 팀이 코드 도움, 조사, 시스템 관리 등에 사용할 수 있는 Telegram 봇 설정에 대한 단계별 가이드입니다."
---

# 팀 Telegram 어시스턴트 설정

이 튜토리얼은 팀원 여러 명이 사용할 수 있는 Hermes Agent 기반의 Telegram 봇 설정 과정을 안내합니다. 이 튜토리얼이 끝날 때쯤이면 팀원들은 코드, 조사, 시스템 관리 등 어떤 것이든 도움을 받을 수 있는 공유 AI 어시스턴트를 갖게 되며, 이는 사용자별 권한 부여로 안전하게 보호됩니다.

## 구축할 내용

다음과 같은 기능을 갖춘 Telegram 봇:

- **승인된 팀원 누구나** DM을 보내 코드 리뷰, 조사, 셸 명령, 디버깅 등의 도움을 받을 수 있습니다.
- 터미널, 파일 편집, 웹 검색, 코드 실행 등 완전한 도구 액세스 권한을 가지고 **서버에서 실행**됩니다.
- **사용자별 세션** — 각 개인은 자신만의 대화 컨텍스트를 가집니다.
- **기본적으로 안전함** — 승인된 사용자만 두 가지 권한 부여 방식을 통해 상호 작용할 수 있습니다.
- **예약된 작업** — 일일 스탠드업, 상태 점검 및 알림이 팀 채널로 전달됩니다.

---

## 사전 요구 사항

시작하기 전에 다음이 준비되어 있는지 확인하세요:

- 서버 또는 VPS에 **Hermes Agent 설치됨** (랩탑은 권장하지 않습니다 — 봇은 계속 실행되어야 합니다). 아직 설치하지 않았다면 [설치 가이드](/getting-started/installation)를 따르세요.
- 봇 소유자를 위한 **Telegram 계정**
- **구성된 LLM 제공자** — 최소한 `~/.hermes/.env`에 OpenAI, Anthropic 또는 지원되는 다른 제공자의 API 키가 필요합니다.

:::tip
월 5달러의 VPS면 게이트웨이를 실행하기에 충분합니다. Hermes 자체는 가볍습니다 — 비용이 드는 것은 LLM API 호출이며, 이는 원격으로 발생합니다.
:::

---

## 1단계: Telegram 봇 생성

모든 Telegram 봇은 봇 생성을 위한 Telegram의 공식 봇인 **@BotFather**에서 시작합니다.

1. **Telegram을 열고** `@BotFather`를 검색하거나 [t.me/BotFather](https://t.me/BotFather)로 이동합니다.

2. **`/newbot` 전송** — BotFather가 두 가지를 묻습니다:
   - **표시 이름(Display name)** — 사용자가 보는 이름 (예: `Team Hermes Assistant`)
   - **사용자 이름(Username)** — 반드시 `bot`으로 끝나야 합니다 (예: `myteam_hermes_bot`)

3. **봇 토큰 복사** — BotFather가 다음과 같이 답장합니다:
   ```
   Use this token to access the HTTP API:
   7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...
   ```
   이 토큰을 저장해 두세요 — 다음 단계에서 필요합니다.

4. **설명 설정** (선택 사항이지만 권장):
   ```
   /setdescription
   ```
   봇을 선택한 후 다음과 같이 입력합니다:
   ```
   Hermes Agent 기반의 팀 AI 어시스턴트입니다. 코드, 조사, 디버깅 등의 도움을 받으려면 저에게 DM을 보내세요.
   ```

5. **봇 명령어 설정** (선택 사항 — 사용자에게 명령어 메뉴 제공):
   ```
   /setcommands
   ```
   봇을 선택한 후 다음을 붙여넣습니다:
   ```
   new - 새로운 대화 시작
   model - AI 모델 표시 또는 변경
   status - 세션 정보 표시
   help - 사용 가능한 명령어 표시
   stop - 현재 작업 중지
   ```

:::warning
봇 토큰을 비밀로 유지하세요. 토큰을 가진 누구나 봇을 제어할 수 있습니다. 만약 유출되었다면, BotFather에서 `/revoke`를 사용하여 새 토큰을 생성하세요.
:::

---

## 2단계: 게이트웨이 구성

대화형 설정 마법사(권장) 또는 수동 구성의 두 가지 옵션이 있습니다.

### 옵션 A: 대화형 설정 (권장)

```bash
hermes gateway setup
```

방향키 선택으로 모든 과정을 안내합니다. **Telegram**을 선택하고, 봇 토큰을 붙여넣고, 메시지가 표시되면 사용자 ID를 입력하세요.

### 옵션 B: 수동 구성

`~/.hermes/.env`에 다음 줄을 추가합니다:

```bash
# BotFather에서 받은 Telegram 봇 토큰
TELEGRAM_BOT_TOKEN=7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...

# 본인의 Telegram 사용자 ID (숫자)
TELEGRAM_ALLOWED_USERS=123456789
```

### 사용자 ID 찾기

Telegram 사용자 ID는 숫자 값입니다(사용자 이름이 아닙니다). 이를 찾으려면:

1. Telegram에서 [@userinfobot](https://t.me/userinfobot)에게 메시지를 보냅니다.
2. 숫자 사용자 ID를 즉시 답장으로 받습니다.
3. 그 숫자를 `TELEGRAM_ALLOWED_USERS`에 복사합니다.

:::info
Telegram 사용자 ID는 `123456789`와 같은 영구적인 숫자입니다. 변경될 수 있는 `@username`과는 다릅니다. 허용 목록에는 항상 숫자 ID를 사용하세요.
:::

---

## 3단계: 게이트웨이 시작

### 빠른 테스트

모든 것이 작동하는지 확인하기 위해 먼저 게이트웨이를 포그라운드에서 실행합니다:

```bash
hermes gateway
```

다음과 같은 출력이 보여야 합니다:

```
[Gateway] Starting Hermes Gateway...
[Gateway] Telegram adapter connected
[Gateway] Cron scheduler started (tick every 60s)
```

Telegram을 열고 봇을 찾아 메시지를 보냅니다. 답장이 오면 성공입니다. `Ctrl+C`를 눌러 중지합니다.

### 프로덕션: 서비스로 설치

재부팅 후에도 유지되는 영구 배포의 경우:

```bash
hermes gateway install
sudo hermes gateway install --system   # Linux 전용: 부팅 시 시스템 서비스로 설치
```

이 명령어는 백그라운드 서비스를 생성합니다: 기본적으로 Linux에서는 사용자 수준 **systemd** 서비스, macOS에서는 **launchd** 서비스, 또는 `--system`을 전달하면 부팅 시 Linux 시스템 서비스를 생성합니다.

```bash
# Linux — 기본 사용자 서비스 관리
hermes gateway start
hermes gateway stop
hermes gateway status

# 실시간 로그 보기
journalctl --user -u hermes-gateway -f

# SSH 로그아웃 후에도 계속 실행
sudo loginctl enable-linger $USER

# Linux 서버 — 명시적인 시스템 서비스 명령어
sudo hermes gateway start --system
sudo hermes gateway status --system
journalctl -u hermes-gateway -f
```

```bash
# macOS — 서비스 관리
hermes gateway start
hermes gateway stop
tail -f ~/.hermes/logs/gateway.log
```

:::tip macOS PATH
launchd plist는 게이트웨이 하위 프로세스가 Node.js 및 ffmpeg와 같은 도구를 찾을 수 있도록 설치 시점의 셸 PATH를 캡처합니다. 나중에 새로운 도구를 설치하는 경우 `hermes gateway install`을 다시 실행하여 plist를 업데이트하세요.
:::

### 실행 확인

```bash
hermes gateway status
```

그런 다음 Telegram에서 봇에게 테스트 메시지를 보냅니다. 몇 초 안에 응답을 받아야 합니다.

---

## 4단계: 팀 액세스 설정

이제 팀원들에게 권한을 부여해 보겠습니다. 두 가지 방법이 있습니다.

### 방법 A: 정적 허용 목록

각 팀원의 Telegram 사용자 ID를 수집하여([@userinfobot](https://t.me/userinfobot)에게 메시지를 보내도록 요청) 쉼표로 구분된 목록으로 추가합니다:

```bash
# ~/.hermes/.env 파일에서
TELEGRAM_ALLOWED_USERS=123456789,987654321,555555555
```

변경 후 게이트웨이를 재시작합니다:

```bash
hermes gateway stop && hermes gateway start
```

### 방법 B: DM 페어링 (팀에 권장)

DM 페어링은 더 유연합니다 — 사용자 ID를 미리 수집할 필요가 없습니다. 작동 방식은 다음과 같습니다:

1. **팀원이 봇에게 DM 전송** — 허용 목록에 없으므로 봇이 일회성 페어링 코드로 답장합니다:
   ```
   🔐 Pairing code: XKGH5N7P
   Send this code to the bot owner for approval.
   ```

2. **팀원이 코드를 당신에게 전송** (Slack, 이메일, 직접 전달 등 어떤 채널을 통해서든)

3. **서버에서 승인**:
   ```bash
   hermes pairing approve telegram XKGH5N7P
   ```

4. **접속 완료** — 봇이 즉시 그들의 메시지에 응답하기 시작합니다.

**페어링된 사용자 관리:**

```bash
# 보류 중이거나 승인된 모든 사용자 보기
hermes pairing list

# 누군가의 액세스 권한 취소
hermes pairing revoke telegram 987654321

# 만료된 보류 코드 지우기
hermes pairing clear-pending
```

:::tip
DM 페어링은 새 사용자를 추가할 때 게이트웨이를 재시작할 필요가 없으므로 팀에 이상적입니다. 승인은 즉시 적용됩니다.
:::

### 보안 고려 사항

- **절대 터미널 액세스가 있는 봇에서 `GATEWAY_ALLOW_ALL_USERS=true`로 설정하지 마세요** — 봇을 찾은 누구나 서버에서 명령을 실행할 수 있습니다.
- 페어링 코드는 **1시간** 후에 만료되며 암호화된 무작위성을 사용합니다.
- 속도 제한은 무차별 대입 공격을 방지합니다: 사용자당 10분마다 1회 요청, 플랫폼당 최대 3개의 보류 코드
- 승인 시도가 5번 실패하면 해당 플랫폼은 1시간 동안 잠금 상태가 됩니다.
- 모든 페어링 데이터는 `chmod 0600` 권한으로 저장됩니다.

---

## 5단계: 봇 구성

### 홈 채널 설정

**홈 채널(home channel)**은 봇이 크론 작업 결과와 선제적인 메시지를 전달하는 곳입니다. 이것이 없으면 예약된 작업이 출력을 보낼 곳이 없습니다.

**옵션 1:** 봇이 멤버로 있는 Telegram 그룹이나 채팅에서 `/sethome` 명령어를 사용합니다.

**옵션 2:** `~/.hermes/.env`에서 수동으로 설정합니다:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="Team Updates"
```

채널 ID를 찾으려면 [@userinfobot](https://t.me/userinfobot)을 그룹에 추가하세요 — 그룹의 채팅 ID를 알려줄 것입니다.

### 도구 진행 상황 표시 구성

봇이 도구를 사용할 때 얼마나 많은 세부 정보를 표시할지 제어합니다. `~/.hermes/config.yaml` 파일에서:

```yaml
display:
  tool_progress: new    # off | new | all | verbose
```

| 모드 | 표시 내용 |
|------|-------------|
| `off` | 깔끔한 응답만 — 도구 활동 없음 |
| `new` | 각 새 도구 호출에 대한 간략한 상태 (메시징에 권장) |
| `all` | 세부 정보가 포함된 모든 도구 호출 |
| `verbose` | 명령어 결과를 포함한 전체 도구 출력 |

사용자는 채팅에서 `/verbose` 명령을 통해 세션별로 이를 변경할 수도 있습니다.

### SOUL.md로 페르소나 설정

`~/.hermes/SOUL.md`를 편집하여 봇이 소통하는 방식을 사용자 정의하세요:

전체 가이드는 [Hermes와 함께 SOUL.md 사용](/guides/use-soul-with-hermes)을 참조하세요.

```markdown
# Soul
당신은 도움이 되는 팀 어시스턴트입니다. 간결하고 기술적으로 작성하세요.
모든 코드에는 코드 블록을 사용하세요. 팀은 직설적인 것을 중시하므로 인사말은 생략하세요.
디버깅할 때는 해결책을 추측하기 전에 항상 오류 로그를 먼저 요청하세요.
```

### 프로젝트 컨텍스트 추가

팀이 특정 프로젝트를 작업하는 경우, 봇이 스택을 알 수 있도록 컨텍스트 파일을 생성하세요:

```markdown
<!-- ~/.hermes/AGENTS.md -->
# Team Context
- 우리는 Python 3.12와 FastAPI, SQLAlchemy를 사용합니다.
- 프론트엔드는 React와 TypeScript입니다.
- CI/CD는 GitHub Actions에서 실행됩니다.
- 프로덕션은 AWS ECS에 배포됩니다.
- 새 코드에 대한 테스트 작성을 항상 제안하세요.
```

:::info
컨텍스트 파일은 모든 세션의 시스템 프롬프트에 주입됩니다. 간결하게 유지하세요 — 모든 문자가 토큰 예산에 반영됩니다.
:::

---

## 6단계: 예약된 작업 설정

게이트웨이가 실행 중이면 팀 채널로 결과를 전달하는 반복 작업을 예약할 수 있습니다.

### 일일 스탠드업 요약

Telegram에서 봇에게 메시지를 보냅니다:

```
매주 평일 오전 9시에 github.com/myorg/myproject에 있는 GitHub 저장소에서 다음을 확인해 줘:
1. 지난 24시간 동안 열리거나 병합된 풀 리퀘스트
2. 생성되거나 닫힌 이슈
3. main 브랜치의 CI/CD 실패
스탠드업 스타일의 간략한 요약으로 포맷해 줘.
```

에이전트는 자동으로 크론 작업을 생성하고 요청한 채팅(또는 홈 채널)으로 결과를 전달합니다.

### 서버 상태 점검

```
6시간마다 'df -h'로 디스크 사용량, 'free -h'로 메모리, 'docker ps'로 Docker 컨테이너 상태를 확인해 줘.
파티션 80% 초과, 재시작된 컨테이너, 높은 메모리 사용량 등 비정상적인 사항이 있으면 보고해.
```

### 예약된 작업 관리

```bash
# CLI에서
hermes cron list          # 예약된 모든 작업 보기
hermes cron status        # 스케줄러 실행 여부 확인

# Telegram 채팅에서
/cron list                # 작업 보기
/cron remove <job_id>     # 작업 제거
```

:::warning
크론 작업 프롬프트는 이전 대화의 기억이 전혀 없는 완전히 새로운 세션에서 실행됩니다. 각 프롬프트에 에이전트가 필요로 하는 **모든** 컨텍스트(파일 경로, URL, 서버 주소 및 명확한 지침)가 포함되어 있는지 확인하세요.
:::

---

## 프로덕션 팁

### 안전을 위한 Docker 사용

공유 팀 봇의 경우 Docker를 터미널 백엔드로 사용하여 에이전트 명령이 호스트가 아닌 컨테이너에서 실행되도록 하세요:

```bash
# ~/.hermes/.env 파일에서
TERMINAL_BACKEND=docker
TERMINAL_DOCKER_IMAGE=nikolaik/python-nodejs:python3.11-nodejs20
```

또는 `~/.hermes/config.yaml` 파일에서:

```yaml
terminal:
  backend: docker
  container_cpu: 1
  container_memory: 5120
  container_persistent: true
```

이렇게 하면 누군가 봇에게 파괴적인 실행을 요청하더라도 호스트 시스템이 보호됩니다.

### 게이트웨이 모니터링

```bash
# 게이트웨이가 실행 중인지 확인
hermes gateway status

# 실시간 로그 보기 (Linux)
journalctl --user -u hermes-gateway -f

# 실시간 로그 보기 (macOS)
tail -f ~/.hermes/logs/gateway.log
```

### Hermes 업데이트 유지

Telegram에서 봇에게 `/update`를 보내면 — 최신 버전을 가져와 재시작합니다. 또는 서버에서:

```bash
hermes update
hermes gateway stop && hermes gateway start
```

### 로그 위치

| 항목 | 위치 |
|------|----------|
| 게이트웨이 로그 | `journalctl --user -u hermes-gateway` (Linux) 또는 `~/.hermes/logs/gateway.log` (macOS) |
| 크론 작업 출력 | `~/.hermes/cron/output/{job_id}/{timestamp}.md` |
| 크론 작업 정의 | `~/.hermes/cron/jobs.json` |
| 페어링 데이터 | `~/.hermes/pairing/` |
| 세션 기록 | `~/.hermes/sessions/` |

---

## 더 나아가기

작동하는 팀 Telegram 어시스턴트를 구축했습니다. 다음 단계는 다음과 같습니다:

- **[보안 가이드](/user-guide/security)** — 권한 부여, 컨테이너 격리 및 명령 승인에 대한 심층 분석
- **[메시징 게이트웨이](/user-guide/messaging)** — 게이트웨이 아키텍처, 세션 관리 및 채팅 명령어에 대한 전체 레퍼런스
- **[Telegram 설정](/user-guide/messaging/telegram)** — 음성 메시지 및 TTS를 포함한 플랫폼별 세부 정보
- **[예약된 작업](/user-guide/features/cron)** — 전달 옵션 및 크론 표현식이 포함된 고급 크론 예약
- **[컨텍스트 파일](/user-guide/features/context-files)** — 프로젝트 지식을 위한 AGENTS.md, SOUL.md 및 .cursorrules
- **[페르소나](/user-guide/features/personality)** — 내장된 페르소나 프리셋 및 사용자 정의 페르소나 정의
- **플랫폼 추가** — 동일한 게이트웨이에서 [Discord](/user-guide/messaging/discord), [Slack](/user-guide/messaging/slack) 및 [WhatsApp](/user-guide/messaging/whatsapp)을 동시에 실행할 수 있습니다.

---

*질문이나 문제가 있나요? GitHub에서 이슈를 열어주세요 — 기여를 환영합니다.*
