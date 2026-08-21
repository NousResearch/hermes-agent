---
sidebar_position: 1
title: "Telegram"
description: "Hermes Agent를 Telegram 봇으로 설정"
---

# Telegram 설정

Hermes Agent는 Telegram과 연동되어 모든 기능을 갖춘 대화형 봇으로 작동합니다. 연결하면 어떤 기기에서든 에이전트와 대화하고, 자동으로 텍스트로 변환되는 음성 메모를 보내며, 예약된 작업 결과를 받고, 그룹 채팅에서 에이전트를 사용할 수 있습니다. 이 연동은 [python-telegram-bot](https://python-telegram-bot.org/)을 기반으로 하며 텍스트, 음성, 이미지, 파일 첨부를 지원합니다.

## 1단계: BotFather를 통해 봇 만들기

모든 Telegram 봇에는 Telegram의 공식 봇 관리 도구인 [@BotFather](https://t.me/BotFather)가 발급한 API 토큰이 필요합니다.

1. Telegram을 열고 **@BotFather**를 검색하거나 [t.me/BotFather](https://t.me/BotFather)를 방문합니다.
2. `/newbot`을 보냅니다.
3. **표시 이름**을 정합니다(예: "Hermes Agent") — 원하는 이름을 사용할 수 있습니다.
4. **사용자 이름**을 정합니다 — 고유해야 하며 `bot`으로 끝나야 합니다(예: `my_hermes_bot`).
5. BotFather가 **API 토큰**으로 답장합니다. 토큰은 다음과 같은 형식입니다.

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
봇 토큰을 비밀로 유지하세요. 이 토큰을 가진 사람은 누구나 봇을 제어할 수 있습니다. 토큰이 유출되면 BotFather에서 `/revoke`를 사용해 즉시 폐기하세요.
:::

## 2단계: 봇 사용자 지정(선택 사항)

다음 BotFather 명령은 사용자 경험을 개선합니다. @BotFather에게 메시지를 보내 다음 명령을 사용하세요.

| 명령 | 용도 |
|---------|---------|
| `/setdescription` | 사용자가 채팅을 시작하기 전에 표시되는 "이 봇은 무엇을 할 수 있나요?" 텍스트 설정 |
| `/setabouttext` | 봇 프로필 페이지에 표시되는 짧은 텍스트 설정 |
| `/setuserpic` | 봇 아바타 업로드 |
| `/setcommands` | 명령 메뉴(채팅의 `/` 버튼) 정의 |
| `/setprivacy` | 봇이 모든 그룹 메시지를 볼 수 있는지 제어(3단계 참조) |

:::tip
`/setcommands`에 사용할 수 있는 시작 명령 세트:

```
help - Show help information
new - Start a new conversation
sethome - Set this chat as the home channel
```
:::

### 온라인/오프라인 상태 표시기(선택 사항)

Telegram 봇에는 실제 온라인/오프라인 상태 점이 없습니다. 초록색 점은 *봇 API가 제공하지 않는* *사용자 계정* 기능입니다. 가장 가까운 표시 수단은 봇의 **짧은 설명**(봇 프로필에서 이름 아래에 표시되는 줄)입니다.

`status_indicator`를 활성화하면 게이트웨이 연결 시 Hermes가 짧은 설명을 **Online**으로 설정하고, 정상적으로 종료할 때 **Offline**으로 설정합니다.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        status_indicator: true
        # Optional custom strings (defaults: "Online" / "Offline"):
        status_online: "🟢 Online"
        status_offline: "🔴 Offline"
```

참고:

- 짧은 설명은 채팅별이 아니라 봇에 **전역적으로** 적용됩니다(모든 사용자에게 표시됨). 사용자는 열린 채팅 안의 실시간 배지가 아니라 봇 프로필 페이지에서 이 설명을 봅니다.
- "Offline"을 기록하는 것은 **정상적인** 게이트웨이 종료(`/stop`, `disconnect`)뿐입니다. 강제 종료가 발생하면 마지막으로 알려진 상태가 유지되는데, 이는 프로필 텍스트 표시기의 본질적인 한계입니다.
- 봇의 전역 프로필을 변경하므로 기본적으로 비활성화되어 있습니다.

### 명령 메뉴 우선순위 및 제한(선택 사항)

Hermes는 Telegram 게이트웨이가 시작될 때 명령 메뉴를 자동으로 등록합니다. 메뉴는 중앙 슬래시 명령 레지스트리와 사용 가능한 플러그인/스킬 명령으로 구성된 후, Telegram이 페이로드를 안정적으로 수락할 수 있도록 제한됩니다. 기본 제한은 60개 명령이며, 모든 기본 제공 명령과 자주 사용하는 스킬 명령을 표시하기에 충분합니다.

Telegram의 `/` 선택기에 표시할 로컬 또는 플러그인 명령이 있다면 `~/.hermes/config.yaml`에서 우선순위를 지정하세요.

```yaml
platforms:
  telegram:
    extra:
      command_menu:
        max_commands: 60
        priority_mode: prepend  # prepend | append | replace
        priority:
          - my_plugin_command
```

`priority_mode`는 목록을 Hermes의 기본 제공 우선순위 목록과 결합하는 방식을 제어합니다.

- `prepend`: 사용자의 명령을 먼저 배치한 다음 Hermes 기본값을 배치
- `append`: Hermes 기본값을 먼저 유지한 다음 사용자의 명령을 배치
- `replace`: 우선순위 정렬에 사용자의 목록만 사용

Telegram은 최대 100개의 BotCommand를 허용하지만, 명령 페이로드가 크면 실패할 수 있습니다. Hermes는 안정성을 위해 기본값을 60으로 사용하고, 설정된 값을 `1..100` 범위로 제한합니다. 전체 명령 목록을 보려면 `/commands`를 사용하세요.

## 3단계: 개인정보 보호 모드(그룹에서 중요)

Telegram 봇에는 **개인정보 보호 모드**가 있으며 기본적으로 **활성화**되어 있습니다. 봇을 그룹에서 사용할 때 가장 흔히 혼동하는 원인입니다.

**개인정보 보호 모드가 켜져 있으면** 봇은 다음만 볼 수 있습니다.
- `/` 명령으로 시작하는 메시지
- 봇 자신의 메시지에 직접 답장한 메시지
- 서비스 메시지(멤버 참여/퇴장, 고정 메시지 등)
- 봇이 관리자인 채널의 메시지

**개인정보 보호 모드가 꺼져 있으면** 봇은 그룹의 모든 메시지를 받습니다.

### 개인정보 보호 모드 비활성화 방법

1. **@BotFather**에게 메시지를 보냅니다.
2. `/mybots`를 보냅니다.
3. 봇을 선택합니다.
4. **Bot Settings → Group Privacy → Turn off**로 이동합니다.

:::warning
개인정보 보호 설정을 변경한 후에는 모든 그룹에서 **봇을 제거했다가 다시 추가해야 합니다**. Telegram은 봇이 그룹에 참여할 때 개인정보 보호 상태를 캐시하며, 봇을 제거했다가 다시 추가하기 전에는 상태를 업데이트하지 않습니다.
:::

:::tip
개인정보 보호 모드를 비활성화하는 대신 봇을 **그룹 관리자**로 승격할 수도 있습니다. 관리자 봇은 개인정보 보호 설정과 관계없이 항상 모든 메시지를 받으므로 전역 개인정보 보호 모드를 전환할 필요가 없습니다.
:::

### 자동으로 답장하지 않고 그룹 대화 관찰하기

OpenClaw/Yuanbao 스타일의 그룹 동작을 사용하려면, 봇이 일반 그룹 메시지를 **볼 수는 있지만 직접 트리거될 때만 응답**하도록 Telegram을 구성하세요.

```yaml
telegram:
  allowed_chats:
    - "-1001234567890"
  group_allowed_chats:
    - "-1001234567890"
  require_mention: true
  observe_unmentioned_group_messages: true
```

이 모드를 활성화하면 명시적으로 허용 목록에 추가된 채팅/토픽에서 멘션되지 않은 그룹 메시지가 관찰된 컨텍스트로 공유 채팅/토픽 세션 기록에 추가되지만, 에이전트를 실행하지는 않습니다. `allowed_chats`는 봇이 응답할 수 있는 위치를 제한하고, `group_allowed_chats`는 관찰된 컨텍스트에 사용할 공유 그룹 세션을 승인하므로 이 모드에서는 동일한 채팅 ID를 사용하세요. 이후 동일한 허용 목록 채팅/토픽에서 `@botname` 멘션, 봇에 대한 답장 또는 구성된 멘션 패턴이 발생하면 관찰된 컨텍스트를 사용할 수 있습니다. 트리거된 메시지에는 `[nickname|user_id]` 태그도 붙으며, 모델이 이전에 관찰된 줄을 봇에게 전달된 지시가 아니라 컨텍스트로 처리하도록 턴별 안전 프롬프트가 적용됩니다.

이에 해당하는 환경 변수:

```bash
TELEGRAM_ALLOWED_CHATS=-1001234567890
TELEGRAM_GROUP_ALLOWED_CHATS=-1001234567890
TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES=true
```

이를 사용하려면 Telegram이 일반 그룹 메시지를 게이트웨이로 전달해야 하므로, 위 설명대로 BotFather 개인정보 보호 모드를 비활성화하거나 봇을 그룹 관리자로 승격하세요.

## 4단계: 사용자 ID 확인

Hermes Agent는 숫자로 된 Telegram 사용자 ID를 사용해 접근을 제어합니다. 사용자 ID는 **사용자 이름**이 아니라 `123456789`와 같은 숫자입니다.

**방법 1(권장):** [@userinfobot](https://t.me/userinfobot)에게 메시지를 보내면 사용자 ID를 즉시 답장으로 알려 줍니다.

**방법 2:** [@get_id_bot](https://t.me/get_id_bot)에게 메시지를 보내세요. 이 방법도 신뢰할 수 있습니다.

이 번호를 저장해 두세요. 다음 단계에서 필요합니다.

## 5단계: Hermes 구성

### 옵션 A: 대화형 설정(권장)

```bash
hermes gateway setup
```

메시지가 표시되면 **Telegram**을 선택합니다. 마법사가 봇 토큰과 허용된 사용자 ID를 묻고 설정을 대신 작성합니다.

### 옵션 B: 수동 구성

다음 내용을 `~/.hermes/.env`에 추가합니다.

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789    # Comma-separated for multiple users
```

### 게이트웨이 시작

```bash
hermes gateway
```

몇 초 안에 봇이 온라인 상태가 됩니다. Telegram에서 봇에게 메시지를 보내 확인하세요.

## Docker 기반 터미널에서 생성된 파일 보내기

터미널 백엔드가 `docker`인 경우 Telegram 첨부 파일은 컨테이너 내부가 아니라 **게이트웨이 프로세스**에서 전송된다는 점에 유의하세요. 따라서 최종 `MEDIA:/...` 경로는 게이트웨이가 실행 중인 호스트에서 읽을 수 있어야 합니다.

흔히 발생하는 문제:

- 에이전트가 Docker 내부의 `/workspace/report.txt`에 파일을 작성
- 모델이 `MEDIA:/workspace/report.txt`를 출력
- `/workspace/report.txt`가 호스트가 아니라 컨테이너 내부에만 존재하므로 Telegram 전송 실패

권장 패턴:

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/.hermes/cache/documents:/output"
```

그런 다음:

- Docker 내부의 `/output/...`에 파일을 작성
- `MEDIA:`에 **호스트에서 볼 수 있는** 경로를 출력합니다. 예:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`

이미 `docker_volumes:` 섹션이 있다면 새 마운트를 동일한 목록에 추가하세요. YAML의 중복 키는 앞서 나온 값을 조용히 덮어씁니다.

### 지원되는 `MEDIA:` 파일 확장자

게이트웨이는 에이전트 답변에서 `MEDIA:/path/to/file` 태그를 추출하고, 참조된 파일을 플랫폼의 기본 첨부 파일로 전송합니다. 모든 게이트웨이 플랫폼에서 지원되는 확장자는 다음과 같습니다.

| 범주 | 확장자 |
|---|---|
| 이미지 | `png`, `jpg`, `jpeg`, `gif`, `webp`, `bmp`, `tiff`, `svg` |
| 오디오 | `mp3`, `wav`, `ogg`, `m4a`, `opus`, `flac`, `aac` |
| 비디오 | `mp4`, `mov`, `webm`, `mkv`, `avi` |
| **문서** | `pdf`, `txt`, `md`, `csv`, `json`, `xml`, `html`, `yaml`, `yml`, `log` |
| **오피스** | `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp` |
| **압축 파일** | `zip`, `rar`, `7z`, `tar`, `gz`, `bz2` |
| **전자책 / 패키지** | `epub`, `apk`, `ipa` |

이 목록에 있는 파일은 지원하는 플랫폼(Telegram, Discord, Signal, Slack, WhatsApp, Feishu, Matrix 등)에서 기본 첨부 파일로 전송됩니다. 기본 첨부를 지원하지 않는 플랫폼에서는 링크 또는 일반 텍스트 표시로 대체됩니다. **굵게 표시된** 범주는 최근 몇 차례 릴리스에서 추가되었습니다. 이전처럼 모델이 `here is the file: /path/to/report.docx`라고 말하도록 하고 있었다면, 기본 첨부 파일 전송을 위해 `MEDIA:/path/to/report.docx`로 바꾸세요.

## 웹훅 모드

기본적으로 Hermes는 **롱 폴링**을 사용해 Telegram에 연결합니다. 게이트웨이가 Telegram 서버에 아웃바운드 요청을 보내 새 업데이트를 가져오는 방식입니다. 로컬 환경과 항상 실행 중인 배포에 적합합니다.

**클라우드 배포**(Fly.io, Railway, Render 등)에서는 **웹훅 모드**가 비용 면에서 더 효율적입니다. 이러한 플랫폼은 인바운드 HTTP 트래픽이 발생하면 일시 중지된 머신을 자동으로 깨울 수 있지만, 아웃바운드 연결로는 그렇게 할 수 없습니다. 폴링은 아웃바운드 방식이므로 폴링 봇은 절대 절전 상태로 전환할 수 없습니다. 웹훅 모드는 방향을 반대로 바꿉니다. Telegram이 봇의 HTTPS URL로 업데이트를 푸시하므로 유휴 상태일 때 절전하는 배포가 가능합니다.

| | 폴링(기본값) | 웹훅 |
|---|---|---|
| 방향 | 게이트웨이 → Telegram(아웃바운드) | Telegram → 게이트웨이(인바운드) |
| 적합한 환경 | 로컬, 상시 실행 서버 | 자동 깨우기를 지원하는 클라우드 플랫폼 |
| 설정 | 추가 설정 없음 | `TELEGRAM_WEBHOOK_URL` 설정 |
| 유휴 비용 | 머신을 계속 실행해야 함 | 메시지 사이에 머신이 절전할 수 있음 |

### 구성

다음 내용을 `~/.hermes/.env`에 추가합니다.

```bash
TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
TELEGRAM_WEBHOOK_SECRET="$(openssl rand -hex 32)"  # required
# TELEGRAM_WEBHOOK_PORT=8443        # optional, default 8443
```

| 변수 | 필수 여부 | 설명 |
|----------|----------|-------------|
| `TELEGRAM_WEBHOOK_URL` | 예 | Telegram이 업데이트를 전송할 공개 HTTPS URL입니다. URL 경로는 자동으로 추출됩니다(예시의 `/telegram` 등). |
| `TELEGRAM_WEBHOOK_SECRET` | **예** (`TELEGRAM_WEBHOOK_URL`이 설정된 경우) | 확인을 위해 Telegram이 모든 웹훅 요청에 그대로 담아 보내는 비밀 토큰입니다. 게이트웨이는 이 값이 없으면 시작을 거부합니다 — [GHSA-3vpc-7q5r-276h](https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h)를 참조하세요. `openssl rand -hex 32`로 생성합니다. |
| `TELEGRAM_WEBHOOK_PORT` | 아니요 | 웹훅 서버가 수신 대기하는 로컬 포트입니다(기본값: `8443`). |

`TELEGRAM_WEBHOOK_URL`이 설정되면 게이트웨이는 폴링 대신 HTTP 웹훅 서버를 시작합니다. 설정되지 않으면 폴링 모드가 사용되며, 이전 버전과 동작이 달라지지 않습니다.

### 클라우드 배포 예시(Fly.io)

1. Fly.io 앱 시크릿에 환경 변수를 추가합니다.

```bash
fly secrets set TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
fly secrets set TELEGRAM_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

2. `fly.toml`에서 웹훅 포트를 공개합니다.

```toml
[[services]]
  internal_port = 8443
  protocol = "tcp"

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

3. 배포합니다.

```bash
fly deploy
```

게이트웨이 로그에 다음이 표시되어야 합니다: `[telegram] Connected to Telegram (webhook mode)`.

## 프록시 지원

Telegram의 API가 차단되었거나 프록시를 통해 라우팅해야 하는 경우, Telegram 전용 프록시 URL을 설정하세요. 이 설정은 일반적인 `HTTPS_PROXY` / `HTTP_PROXY` 환경 변수보다 우선합니다.

**옵션 1: config.yaml (권장)**

```yaml
telegram:
  proxy_url: "socks5://127.0.0.1:1080"
```

**옵션 2: 환경 변수**

```bash
TELEGRAM_PROXY=socks5://127.0.0.1:1080
```

지원되는 스킴: `http://`, `https://`, `socks5://`.

프록시는 기본 Telegram 연결과 대체 IP 전송 모두에 적용됩니다. Telegram 전용 프록시를 설정하지 않으면 게이트웨이는 `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` (또는 macOS 시스템 프록시 자동 감지)로 대체합니다.

호스트에서 대체 IP 검색 경로가 불안정하다면 `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS=true`를 설정해 일반 `api.telegram.org` 경로에서 콜드 연결을 유지할 수 있습니다. 대체 DNS-over-HTTPS 검색에 대한 제한 시간(초)은 `HERMES_TELEGRAM_FALLBACK_DISCOVERY_TIMEOUT`으로 설정할 수도 있으며, 기본값은 `5`입니다.

## 홈 채널

어떤 Telegram 채팅(DM 또는 그룹)에서든 `/sethome` 명령을 사용해 **홈 채널**로 지정하세요. 예약된 작업(cron 작업)의 결과가 이 채널로 전달됩니다.

`~/.hermes/.env`에서 직접 설정할 수도 있습니다.

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

:::tip
그룹 채팅 ID는 음수입니다(예: `-1001234567890`). 개인 DM 채팅 ID는 사용자 ID와 같습니다.
:::

### 토픽 모드에서의 Cron 전달

봇 DM에서 토픽 모드를 활성화한 경우, 루트 채팅으로 전달된 cron 메시지는 시스템 전용 로비에 도착합니다. 그곳에서 답장하면 세션이 열리지 않고 "main chat is reserved for system commands"라는 알림이 표시됩니다. 전용 포럼 토픽(예: `Cron`)을 만들고 다음을 설정하세요.

```bash
TELEGRAM_CRON_THREAD_ID=<topic_thread_id>
```

`TELEGRAM_CRON_THREAD_ID`는 cron 전달에 한해 `TELEGRAM_HOME_CHANNEL_THREAD_ID`보다 우선합니다. 해당 토픽의 답장은 기존 토픽 세션을 계속 사용합니다.

## 음성 메시지

### 수신 음성(음성-텍스트 변환)

Telegram에서 보낸 음성 메시지는 Hermes에 설정된 STT 제공자가 자동으로 텍스트로 변환한 뒤 대화에 삽입합니다.

- `local`은 Hermes를 실행하는 컴퓨터에서 `faster-whisper`를 사용하며 API 키가 필요하지 않습니다.
- `groq`는 Groq Whisper를 사용하며 `GROQ_API_KEY`가 필요합니다.
- `openai`는 OpenAI Whisper를 사용하며 `VOICE_TOOLS_OPENAI_KEY`가 필요합니다.

#### STT 건너뛰기: 원본 오디오 파일을 에이전트에 전달

오디오를 **에이전트 자체가** 처리하도록 하려면(화자 분리, 사용자 지정 전사 도구, 녹음 보관 등에 활용), `~/.hermes/config.yaml`에서 `stt.enabled: false`를 설정하세요.

```yaml
stt:
  enabled: false
```

STT를 비활성화해도 게이트웨이는 음성/오디오 첨부 파일을 Hermes의 오디오 캐시에 다운로드하지만 **전사하지는 않습니다**. 에이전트는 다음과 같은 마커와 함께 메시지를 받습니다.

```
[The user sent a voice message: /home/<user>/.hermes/cache/audio/<hash>.ogg]
```

그런 다음 도구나 스킬이 해당 경로를 직접 읽을 수 있습니다(예: 로컬 화자 분리 파이프라인이나 더 풍부한 전사 모델로 넘기거나 장기 저장소에 업로드). 파일 확장자는 Telegram이 전달한 원본 형식을 반영합니다(음성 메모는 `.ogg`, 오디오 첨부 파일은 `.mp3`/`.m4a`/등).

이는 아래의 [로컬 Bot API 서버](#large-files-20mb-via-local-bot-api-server) 섹션과 자연스럽게 연동됩니다. Telegram의 20MB `getFile` 제한을 2GB로 높여 몇 분보다 긴 녹음도 처리할 수 있습니다.

### 발신 음성(텍스트-음성 변환)

에이전트가 TTS를 통해 오디오를 생성하면 기본 Telegram **음성 말풍선**으로 전달됩니다. 둥글고 인라인 재생이 가능한 형태입니다.

- **OpenAI와 ElevenLabs**는 Opus를 기본으로 생성하므로 추가 설정이 필요하지 않습니다.
- **Edge TTS**(기본 무료 제공자)는 MP3를 출력하며 Opus로 변환하려면 **ffmpeg**가 필요합니다.

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

ffmpeg가 없으면 Edge TTS 오디오는 일반 오디오 파일로 전송됩니다(재생할 수는 있지만 음성 말풍선 대신 직사각형 플레이어를 사용합니다).

`config.yaml`의 `tts.provider` 키에서 TTS 제공자를 설정하세요.

## 로컬 Bot API 서버를 통한 대용량 파일(>20MB)

Telegram의 **공개** Bot API는 `getFile` 다운로드를 **20 MB**로 제한하므로, 이보다 큰 음성 메모, 오디오 파일, 동영상 또는 문서는 모두 Hermes에서 "too large" 답장과 함께 조용히 거부됩니다. 문서화된 우회 방법은 **로컬** [telegram-bot-api](https://github.com/tdlib/telegram-bot-api) 데몬을 실행하는 것입니다. Telegram이 사용하는 동일한 서버 소프트웨어를 네트워크에서 직접 실행하는 방식입니다. 로컬 서버는 파일 제한을 **2 GB**로 높이며, Hermes는 사용자 지정 `base_url`이 설정된 것을 확인하면 자체 내부 제한도 자동으로 높입니다.

이를 통해 다음과 같은 작업을 할 수 있습니다.

- 긴 음성 메모(45분 회의, 팟캐스트)를 봇으로 보내기
- 비전 도구 처리용 대용량 동영상 업로드
- 화자 분리, 정렬 또는 학습 데이터와 같은 오프라인 파이프라인을 위한 원본 오디오 보관

### 1단계: Telegram API 자격 증명 가져오기

로컬 서버는 공개 Bot API가 아니라 Telegram의 MTProto 계층에 직접 연결하므로 **MTProto 자격 증명**이 필요합니다.

1. [my.telegram.org/apps](https://my.telegram.org/apps)를 방문해 Telegram 계정으로 로그인합니다.
2. 새 애플리케이션을 만듭니다(이름과 짧은 설명은 무엇이든 괜찮습니다).
3. `api_id`와 `api_hash`를 복사합니다. 둘 다 필요합니다.

### 2단계: telegram-bot-api 서버 실행

커뮤니티에서 유지 관리하는 [`aiogram/telegram-bot-api`](https://hub.docker.com/r/aiogram/telegram-bot-api) Docker 이미지가 가장 간단한 방법입니다. 최소 구성인 `docker-compose.yaml`은 다음과 같습니다(높은 제한을 활성화하려면 `--local` 모드를 사용하세요).

```yaml
services:
  tg-bot-api:
    image: aiogram/telegram-bot-api:latest
    container_name: tg-bot-api
    restart: unless-stopped
    ports:
      - "127.0.0.1:8081:8081"   # bind to loopback only; see security note
    environment:
      TELEGRAM_API_ID: "12345"           # your api_id from Step 1
      TELEGRAM_API_HASH: "abcdef..."     # your api_hash from Step 1
      TELEGRAM_LOCAL: "1"                # enable --local mode (raises 20MB → 2GB)
    volumes:
      - ./tg-bot-api-data:/var/lib/telegram-bot-api
```

실행합니다.

```bash
docker compose up -d tg-bot-api
docker logs --tail 20 tg-bot-api
```

:::warning 보안
로컬 Bot API 서버는 URL 경로(예: `/bot<TOKEN>/getMe`)에 봇 토큰을 포함하며 **추가 인증이 없습니다**. 포트에 접근할 수 있는 사람은 누구나 봇을 완전히 제어할 수 있습니다. 봇이 볼 수 있는 모든 메시지를 읽고, 봇으로 메시지를 보내는 등의 작업이 가능합니다. 컨테이너를 `127.0.0.1`에 바인딩하거나 사설 네트워크의 리버스 프록시 앞에 두세요. **포트 8081을 공용 인터넷에 절대 노출하지 마세요.**
:::

### 3단계: 공개 API에서 봇 로그아웃(최초 1회)

봇은 한 번에 **하나의** Bot API 서버에서만 활성 상태일 수 있습니다. 봇이 이미 `api.telegram.org`를 대상으로 실행 중이었다면(거의 확실히 그랬을 것입니다), 로컬 서버가 봇을 받아들이기 전에 해당 서버에서 명시적으로 로그아웃해야 합니다.

```bash
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/logOut"
# expected response: {"ok":true,"result":true}
```

이 작업은 한 번만 수행하는 마이그레이션 단계이며 재시작할 때마다 반복하지 않습니다. `logOut` 이후 수신된 메시지는 Telegram이 대신 새 서버로 전달합니다.

로컬 서버가 봇을 대신해 Telegram과 통신할 수 있는지 확인합니다.

```bash
curl "http://127.0.0.1:8081/bot<YOUR_BOT_TOKEN>/getMe"
# expected response: {"ok":true,"result":{"id":...,"is_bot":true,...}}
```

### 4단계: Hermes를 로컬 서버에 연결

`~/.hermes/config.yaml`의 `platforms.telegram.extra` 아래에 URL을 추가합니다.

```yaml
platforms:
  telegram:
    extra:
      base_url: "http://127.0.0.1:8081/bot"
      base_file_url: "http://127.0.0.1:8081/file/bot"
      local_mode: true        # see Step 5 below — only set this if the bot's data
                              # directory is readable by the Hermes process
```

:::caution `telegram.extra`가 아니라 `platforms.telegram.extra`를 사용하세요
현재 `platforms.<name>.extra` 형식만 플랫폼 구성에 깊이 병합됩니다. 최상위 `telegram.extra` 블록 바로 아래에 배치한 키는 조용히 삭제됩니다.
:::

`base_url`이 설정되면 Hermes는 다음을 수행합니다.

- 로컬 서버를 사용하도록 python-telegram-bot 클라이언트를 구성합니다.
- 내부 문서/오디오 크기 제한을 20 MB → 2 GB로 자동으로 높입니다.
- "too large" 오류 메시지에 현재 제한(`Maximum: 2048 MB.`)을 표시해 어떤 모드인지 명확히 알려줍니다.

게이트웨이를 재시작하고 확인 로그 줄을 찾습니다.

```bash
hermes gateway restart
grep -E "Using custom Telegram base_url|Using Telegram local_mode" ~/.hermes/logs/gateway.log | tail
```

### 5단계: `local_mode` — 디스크의 파일 접근

로컬 서버는 파일을 전달하는 방법이 **두 가지**입니다.

1. **`--local` 없이**(기본값): 파일은 공개 Bot API와 동일하게 `/file/bot<TOKEN>/<path>`의 HTTP를 통해 제공됩니다. 20MB 제한이 그대로 유지됩니다. 네트워크 문제 해결용으로만 유용합니다(예: `api.telegram.org`에 연결할 수 없지만 직접 호스팅은 가능한 경우). 크기 제한을 높이려는 경우에는 원하는 방식이 아닙니다.
2. **`--local` 사용**(위에서 `TELEGRAM_LOCAL=1`로 설정): 파일이 서버의 파일 시스템에 기록되고 `getFile` 응답은 HTTP URL 대신 **절대 경로**를 반환합니다. 20MB 제한이 해제됩니다. 따라서 Hermes는 HTTP가 아니라 **디스크에서** 바이트를 읽어야 합니다.

디스크 읽기 경로를 작동시키려면 위 구성에서 `local_mode: true`를 설정하고 Hermes 프로세스가 서버가 반환하는 경로를 읽을 수 있는지 확인하세요. 두 가지 경우가 있습니다.

- **동일한 컴퓨터** — telegram-bot-api와 Hermes가 같은 호스트에서 실행됩니다. 데이터 볼륨을 Hermes가 읽을 수 있는 디렉터리(예: `/var/lib/telegram-bot-api`)에 바인드 마운트하고 파일 소유자가 일치하는지 확인합니다. 컨테이너는 내부 `telegram-bot-api` 사용자로 권한을 낮춥니다(uid는 이미지에 따라 다름). 가장 간단한 해결책은 compose 서비스에 `user: "<UID>:<GID>"`를 추가해 Hermes가 이미 실행되는 uid가 파일을 소유하도록 하는 것입니다.
- **서로 다른 컴퓨터** — 봇 서버는 한 호스트(예: NAS 또는 별도의 VM)에서, Hermes는 다른 호스트에서 실행됩니다. 서버의 데이터 디렉터리는 서버가 보고하는 **동일한 절대 경로**(일반적으로 `/var/lib/telegram-bot-api`)로 Hermes 컴퓨터에 공유되어야 합니다. NFS가 이 용도에 잘 맞습니다. 파일 시스템 수준에서 uid 불일치를 처리하고 싶지 않다면 `uid=` 마운트 재매핑을 사용하는 CIFS/SMB가 더 편리합니다.

`local_mode: true`가 설정되었지만 Hermes가 반환된 파일 경로를 `stat`할 수 없으면(권한 또는 잘못된 마운트), python-telegram-bot은 로컬 서버에 대해 HTTP `getFile`을 수행하는 방식으로 조용히 대체됩니다. 그러나 `--local` 모드의 로컬 서버는 `404 Not Found`로 응답합니다. 증상은 `gateway.log`에 다음과 같이 나타납니다.

```
[Telegram] Failed to cache voice: Not Found
telegram.error.InvalidToken: Not Found
```

이 메시지가 보이면 제한 상향은 작동하지만 파일 공유가 작동하지 않는 것입니다. Hermes 호스트에서 게이트웨이를 실행하는 사용자로 `ls -la /var/lib/telegram-bot-api/<TOKEN>/voice/`를 확인하고, 단일 파일을 권한 오류 없이 `cat`할 수 있는지 확인하세요.

### 6단계: 테스트

20MB보다 큰 음성 메모나 오디오 파일을 봇으로 보냅니다. 게이트웨이 로그를 추적합니다.

```bash
tail -f ~/.hermes/logs/gateway.log | grep -iE "telegram|cache"
```

`[Telegram] Cached user voice at /home/<user>/.hermes/cache/audio/...` 줄이 표시되고 **too large 거부 메시지는 표시되지 않아야** 합니다. (위의) `stt.enabled: false`와 함께 사용하면 원본 오디오 파일의 경로가 이후 처리를 위해 에이전트의 수신 메시지에 들어갑니다.

## 그룹 채팅 사용

Hermes Agent는 몇 가지 고려 사항과 함께 Telegram 그룹 채팅에서 작동합니다.

- **개인정보 보호 모드**는 봇이 볼 수 있는 메시지를 결정합니다(그룹에 중요한 [3단계](#step-3-privacy-mode-critical-for-groups) 참고).
- 그룹에서도 `TELEGRAM_ALLOWED_USERS`가 적용됩니다. 승인된 사용자만 봇을 호출할 수 있습니다.
- `telegram.require_mention: true`로 설정하면 일반적인 그룹 대화에 봇이 응답하지 않도록 할 수 있습니다.
- `telegram.require_mention: true`일 때 그룹 메시지는 다음 조건에서 수락됩니다.
  - 봇 메시지 중 하나에 대한 답장
  - `@botusername` 멘션
  - `/command@botusername`(봇 이름을 포함하는 Telegram의 봇 메뉴 명령 형식)
  - `telegram.mention_patterns`에 설정한 정규식 깨우기 단어 중 하나와 일치
- Hermes 봇이 여러 개 있는 그룹에서는 `telegram.exclusive_bot_mentions`가 라우팅을 결정적으로 유지합니다. 메시지에서 하나 이상의 Telegram 봇 사용자 이름을 명시적으로 멘션하면, 멘션된 봇 프로필만 메시지를 처리하고 다른 Hermes 봇은 답장 전에 무시합니다. 이후 깨우기 단어 대체 동작이 실행됩니다. 이 기능은 기본적으로 활성화되어 있습니다.
- BotFather에서 봇의 `@username`을 변경하면 자동으로 반영됩니다. Hermes는 게이트웨이 재시작 없이 새 핸들을 따라 멘션 라우팅을 수행합니다. `bot`으로 끝나지 않는 수집형(Fragment) 사용자 이름도 지원됩니다.
- `telegram.ignored_threads`를 사용하면 그룹에서 일반 응답이나 멘션 기반 응답이 허용되는 경우에도 특정 Telegram 포럼 토픽에서 Hermes가 조용히 있도록 할 수 있습니다.
- `telegram.require_mention`을 설정하지 않거나 false로 두면 Hermes는 이전의 개방형 그룹 동작을 유지하며 볼 수 있는 일반 그룹 메시지에 응답합니다.
### 여러 Hermes 봇을 한 그룹에서 실행하기

같은 Telegram 그룹에서 여러 Hermes 프로필을 실행하는 경우, 프로필마다 Telegram 봇 토큰을 하나씩 만들고 프로필마다 게이트웨이 하나를 시작하세요. 여러 실행 중인 게이트웨이에서 같은 봇 토큰을 재사용하지 마세요. Telegram이 동일한 토큰에 대한 동시 폴링을 거부합니다.

권장 그룹 설정:

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns: []
```

이 설정에서는 `@research_bot @ops_bot summarize this`와 같은 그룹 메시지를 `research_bot`과 `ops_bot`만 처리합니다. 그룹에 있는 다른 Hermes 봇은 해당 봇의 이전 메시지에 대한 답장이거나 공유 wake word와 일치하더라도 조용히 대기합니다.

명시적 멘션이 답장 및 wake word 트리거보다 우선하도록 하지 않아야 하는 레거시 그룹에서만 `exclusive_bot_mentions: false`를 설정하세요.

여러 프로필을 운영하려면 프로필마다 게이트웨이 명령을 한 번씩 실행하세요. 예:

```bash
# default profile
hermes gateway start
hermes gateway status
hermes gateway stop

# named profiles
hermes -p research gateway start
hermes -p research gateway status
hermes -p research gateway stop
```

작은 고정 규모의 봇 묶음이라면 기본 프로필에는 `hermes gateway <action>`을, 이름이 지정된 각 프로필에는 `hermes -p <profile> gateway <action>`을 호출하는 셸 루프나 스크립트를 사용하세요. 서비스 관리자마다 하나의 프로세스 수준 명령이 모든 이름 지정 프로필을 제어한다고 가정하는 것보다 이 방법이 안정적입니다.

### 문제 해결: DM에서는 작동하지만 그룹에서는 작동하지 않음

봇이 개인 채팅에서는 응답하지만 그룹에서 조용히 대기한다면 다음 게이트를 순서대로 확인하세요.

1. **Telegram 전달:** BotFather 개인정보 보호 모드를 끄거나, 봇을 관리자로 승격하거나, 봇을 직접 멘션하세요. Telegram이 봇에 전달하지 않은 그룹 메시지에는 Hermes가 응답할 수 없습니다.
2. **개인정보 보호 설정 변경 후 재가입:** BotFather 개인정보 보호 설정을 변경한 뒤 봇을 그룹에서 제거하고 다시 추가하세요. Telegram은 기존 멤버십에 이전 전달 동작을 유지할 수 있습니다.
3. **Hermes 인증:** 발신자가 `TELEGRAM_ALLOWED_USERS` 또는 `TELEGRAM_GROUP_ALLOWED_USERS`에 등록되어 있는지 확인하거나, `TELEGRAM_GROUP_ALLOWED_CHATS`를 사용해 그룹 채팅을 허용하세요.
4. **멘션 필터:** `telegram.require_mention: true`가 설정된 경우, 메시지가 슬래시 명령이거나, 봇에 대한 답장이거나, `@botusername` 멘션이거나, 설정된 `mention_patterns`와 일치하지 않으면 일반적인 그룹 대화는 무시됩니다.
5. **다중 봇 라우팅:** 그룹에 여러 봇이 있다면 각 Hermes 프로필이 고유한 봇 토큰을 사용하는지 확인하고, 레거시 공유 트리거 동작을 의도적으로 사용하려는 경우가 아니라면 `exclusive_bot_mentions`를 활성화된 상태로 유지하세요.

음수 채팅 ID는 Telegram 그룹과 슈퍼그룹에서 정상입니다. 채팅 범위 인증을 사용하는 경우 해당 ID를 발신자 사용자 허용 목록이 아니라 `TELEGRAM_GROUP_ALLOWED_CHATS`에 넣으세요.

### 그룹 트리거 설정 예

`~/.hermes/config.yaml`에 다음을 추가하세요.

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns:
    - "^\\s*chompy\\b"
  ignored_threads:
    - 31
    - "42"
```

이 예에서는 일반적인 모든 직접 트리거와 함께 `@mention`을 사용하지 않았더라도 `chompy`로 시작하는 메시지를 허용합니다.

Telegram 토픽 `31`과 `42`의 메시지는 멘션 및 자유 응답 확인이 실행되기 전에 항상 무시됩니다.

### `mention_patterns` 참고 사항

- 패턴에는 Python 정규 표현식을 사용합니다.
- 대소문자를 구분하지 않고 일치시킵니다.
- 패턴은 텍스트 메시지와 미디어 캡션 모두에 대해 확인됩니다.
- 잘못된 정규 표현식 패턴은 봇을 중단시키는 대신 게이트웨이 로그에 경고를 남기고 무시됩니다.
- 메시지 시작 부분에서만 패턴이 일치하게 하려면 `^`로 고정하세요.

## 비공개 채팅 토픽 (Bot API 9.4)

Telegram Bot API 9.4(2026년 2월)에서는 **비공개 채팅 토픽(Private Chat Topics)**이 도입되었습니다. 봇은 슈퍼그룹 없이도 1:1 DM 채팅에서 포럼 스타일의 토픽 스레드를 직접 만들 수 있습니다. 이를 통해 기존 DM 안에서 Hermes와 함께 여러 개의 격리된 작업 공간을 운영할 수 있습니다.

### 사용 사례

여러 장기 프로젝트를 진행한다면 토픽을 사용해 각 프로젝트의 컨텍스트를 분리할 수 있습니다.

- **토픽 "Website"** — 프로덕션 웹 서비스 작업
- **토픽 "Research"** — 문헌 검토 및 논문 탐색
- **토픽 "General"** — 기타 작업과 간단한 질문

각 토픽에는 다른 토픽과 완전히 격리된 자체 대화 세션, 기록 및 컨텍스트가 있습니다.

### 설정

:::caution 사전 조건
설정에 토픽을 추가하기 전에 사용자는 봇과의 DM 채팅에서 **토픽(Topics) 모드**를 활성화해야 합니다.

1. Telegram에서 Hermes 봇과의 개인 채팅을 엽니다.
2. 상단에서 봇 이름을 눌러 채팅 정보를 엽니다.
3. **토픽(Topics)**을 활성화합니다(채팅을 포럼으로 전환하는 토글).

이 작업을 하지 않으면 Hermes가 시작 시 `The chat is not a forum`을 로그에 기록하고 토픽 생성을 건너뜁니다. 이는 Telegram 클라이언트 측 설정이며, 봇이 프로그래밍 방식으로 활성화할 수 없습니다.
:::

`~/.hermes/config.yaml`의 `platforms.telegram.extra.dm_topics` 아래에 토픽을 추가하세요.

```yaml
platforms:
  telegram:
    extra:
      dm_topics:
      - chat_id: 123456789        # Your Telegram user ID
        topics:
        - name: General
          icon_color: 7322096
        - name: Website
          icon_color: 9367192
        - name: Research
          icon_color: 16766590
          skill: arxiv              # Auto-load a skill in this topic
```

**필드:**

| 필드 | 필수 여부 | 설명 |
|-------|----------|-------------|
| `name` | 예 | 토픽에 표시할 이름 |
| `icon_color` | 아니요 | Telegram 아이콘 색상 코드(정수) |
| `icon_custom_emoji_id` | 아니요 | 토픽 아이콘에 사용할 사용자 지정 이모지 ID |
| `skill` | 아니요 | 이 토픽의 새 세션에서 자동으로 로드할 Skill |
| `thread_id` | 아니요 | 토픽 생성 후 자동으로 채워짐 — 수동으로 설정하지 마세요 |

### 작동 방식

1. 게이트웨이가 시작되면 Hermes는 아직 `thread_id`가 없는 각 토픽에 대해 `createForumTopic`을 호출합니다.
2. `thread_id`는 자동으로 `config.yaml`에 다시 저장되며, 이후 재시작에서는 API 호출을 건너뜁니다.
3. 각 토픽은 격리된 세션 키 `agent:main:telegram:dm:{chat_id}:{thread_id}`에 매핑됩니다.
4. 각 토픽의 메시지에는 자체 대화 기록, 메모리 플러시 및 컨텍스트 창이 있습니다.

### 루트 DM 처리

기본적으로 루트 DM(토픽 밖에서 전송된 메시지)은 정상적으로 처리됩니다. `ignore_root_dm: true`를 설정하면 루트 DM이 로비로 전환됩니다. DM 토픽이 설정된 사용자의 일반 메시지는 조용히 무시되지만, 시스템 명령(`/start`, `/help`, `/status` 등)은 계속 작동합니다.

```yaml
platforms:
  telegram:
    extra:
      ignore_root_dm: true
      dm_topics:
        - chat_id: 123456789
          topics:
            - name: General
```

이 확인은 **채팅별로** 이루어집니다. `dm_topics`에 항목이 하나 이상 있는 사용자에게만 루트 DM 설정이 적용됩니다. 토픽이 설정되지 않은 사용자는 영향을 받지 않습니다.

### Skill 연결

`skill` 필드가 있는 토픽은 해당 토픽에서 새 세션이 시작될 때 그 Skill을 자동으로 로드합니다. 이는 대화를 시작할 때 `/skill-name`을 입력하는 것과 정확히 같은 방식으로 작동합니다. Skill 콘텐츠가 첫 번째 메시지에 삽입되고, 이후 메시지는 대화 기록에서 해당 콘텐츠를 확인합니다.

예를 들어 `skill: arxiv`가 있는 토픽은 세션이 유휴 시간 초과, 일일 초기화 또는 수동 `/reset`으로 재설정될 때마다 arxiv Skill이 미리 로드됩니다.

:::tip
설정 외부에서 생성된 토픽(예: Telegram API를 직접 호출해 만든 토픽)은 `forum_topic_created` 서비스 메시지가 도착하면 자동으로 검색됩니다. 게이트웨이가 실행 중일 때 설정에 토픽을 추가할 수도 있으며, 다음 캐시 미스에서 해당 토픽이 선택됩니다.
:::

## 다중 세션 DM 모드(`/topic`)

ChatGPT 스타일의 다중 세션 DM — 봇 하나로 여러 병렬 대화를 운영합니다. 위의 운영자가 관리하는 `extra.dm_topics`와 달리 이 모드는 **사용자 주도**입니다. 설정이나 사전 선언된 토픽 이름이 필요하지 않습니다. 최종 사용자가 `/topic`으로 이 모드를 켠 다음 Telegram의 **+** 버튼을 눌러 원하는 만큼 토픽을 만들 수 있으며, 각 토픽은 완전히 독립된 Hermes 세션이 됩니다.

### `/topic` 하위 명령

| 형식 | 컨텍스트 | 동작 |
|------|---------|--------|
| `/topic` | 루트 DM, 아직 활성화되지 않음 | BotFather 기능을 확인하고, 다중 세션 모드를 활성화하며, 고정된 System 토픽 생성 |
| `/topic` | 루트 DM, 이미 활성화됨 | 상태 표시: 복원할 수 있는 연결되지 않은 세션 |
| `/topic` | 토픽 내부 | 현재 토픽의 세션 연결 표시 |
| `/topic help` | 어디서나 | 인라인 사용법 |
| `/topic off` | 루트 DM | 다중 세션 모드를 비활성화하고 이 채팅의 모든 토픽 연결 삭제 |
| `/topic <session-id>` | 토픽 내부 | 이전 Telegram 세션을 현재 토픽으로 복원 |

허용 목록(`TELEGRAM_ALLOWED_USERS` / 플랫폼 인증 설정)을 통해 인증된 사용자만 `/topic`을 실행할 수 있습니다. 인증되지 않은 발신자에게는 활성화 대신 거부 응답이 전송됩니다.

### DM 토픽과 다중 세션 DM 모드 비교

| | `extra.dm_topics`(설정 기반) | `/topic`(사용자 주도) |
|---|---|---|
| 활성화하는 사람 | 운영자, `config.yaml`에서 | 최종 사용자, `/topic` 전송 |
| 토픽 목록 | 설정에 선언된 고정 집합 | 사용자가 자유롭게 토픽 생성/삭제 |
| 토픽 이름 | 운영자가 선택 | 사용자가 선택하며 Hermes 세션 제목에 맞게 자동 변경 |
| 루트 DM 동작 | 일반 채팅(`ignore_root_dm: true`이면 로비) | 시스템 로비가 됨(명령이 아닌 메시지는 거부) |
| 주요 사용 사례 | 선택적인 Skill 연결이 있는 영구 작업 공간 | 임시 병렬 세션 |
| 지속성 | 설정의 `extra.dm_topics` | SQLite 테이블 `telegram_dm_topic_mode` + `telegram_dm_topic_bindings` |

두 기능은 같은 봇에서 함께 사용할 수 있습니다. 사용자의 DM에서 `/topic`을 실행하는 동시에 `extra.dm_topics`로 다른 채팅의 운영자 선언 토픽을 계속 관리할 수 있습니다.

### 사전 조건

**@BotFather**에서 봇을 열고 **Bot Settings → Threads Settings**로 이동하세요.

1. **Threaded Mode**를 켭니다(`has_topics_enabled` 활성화).
2. 사용자가 토픽을 만들 수 없도록 비활성화하지 마세요(`allows_users_to_create_topics`를 켠 상태로 유지).

사용자가 처음 `/topic`을 실행하면 Hermes는 `getMe`를 호출해 두 플래그를 확인합니다. 둘 중 하나라도 꺼져 있으면 Hermes가 BotFather Threads Settings 페이지의 스크린샷을 보내고 어떤 설정을 전환해야 하는지 설명합니다. 사전 조건이 충족될 때까지 활성화는 이루어지지 않습니다.

### 활성화 과정

루트 DM에서 다음을 전송하세요.

```
/topic
```

Hermes는 다음 작업을 수행합니다.

1. `getMe().has_topics_enabled` 및 `allows_users_to_create_topics`를 확인합니다.
2. 두 값이 모두 `true`이면 이 DM에서 다중 세션 토픽 모드를 활성화합니다.
3. 상태 및 명령용 **System** 토픽을 만들고 고정합니다(가능한 경우).
4. 복원할 수 있는 이전의 연결되지 않은 Telegram 세션 목록으로 응답합니다.

활성화 후 **루트 DM은 로비**가 됩니다. 일반 프롬프트는 **All Messages**를 가리키는 안내와 함께 거부됩니다. 시스템 명령(`/status`, `/sessions`, `/usage`, `/help` 등)은 루트에서 계속 작동합니다.

### 새 토픽 만들기(최종 사용자 흐름)

1. Telegram에서 봇 DM을 엽니다.
2. 봇 인터페이스 상단의 **All Messages**를 누른 다음 아무 메시지나 전송합니다.
3. Telegram이 해당 메시지를 위한 새 토픽을 만듭니다.
4. Hermes가 해당 토픽 안에서 응답하며, 토픽은 이제 독립 세션이 됩니다.

모든 토픽에는 자체 대화 기록, 모델 상태, 도구 실행 및 세션 ID가 있습니다. 격리 키는 `agent:main:telegram:dm:{chat_id}:{thread_id}`이며, 설정 기반 DM 토픽의 격리 방식과 동일합니다.

### 토픽 자동 이름 변경

Hermes가 토픽의 세션 제목을 생성하면(첫 번째 대화 후 자동 제목 파이프라인을 통해) Telegram 토픽 자체의 이름도 그 제목에 맞게 변경됩니다. 예를 들어 "New Topic"이 "Database migration plan"으로 바뀝니다. 이름 변경은 최선의 노력으로 수행됩니다. 실패해도 로그에 기록될 뿐 세션이 중단되지는 않습니다.

이를 비활성화하고 직접 선택한 토픽 이름을 그대로 유지하려면 다음을 설정하세요.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_topic_auto_rename: true
```

이 플래그가 켜져 있어도 Hermes는 내부 세션 제목(`hermes sessions`, TUI 등에서 사용)을 생성하지만 Telegram 토픽 이름은 절대 수정하지 않습니다. BotFather Threaded Mode에서 토픽을 직접 정리하고 첫 번째 답변이 제목을 덮어쓰지 않도록 하려는 경우 유용합니다.

### 토픽 안에서 `/new` 사용

다른 토픽에는 영향을 주지 않고 현재 토픽의 세션을 초기화합니다(새 세션 ID와 새로운 기록 생성). Hermes는 병렬 작업을 위해서는 보통 다른 토픽(**All Messages**를 통해)을 만드는 것이 바람직하다는 안내로 응답합니다.

### 이전 세션 복원

토픽 내부에서 다음을 전송합니다:

```
/topic <session-id>
```

이렇게 하면 새 Hermes 세션을 시작하는 대신 현재 토픽을 기존 Hermes 세션에 연결합니다. 토픽 모드를 활성화하기 전에 시작된 대화를 계속할 때 유용합니다. 제한 사항:

- 대상 세션은 동일한 Telegram 사용자에게 속해야 합니다.
- 대상 세션이 이미 다른 토픽에 연결되어 있어서는 안 됩니다.

Hermes는 세션 제목으로 확인하고, 컨텍스트를 위해 마지막 assistant 메시지를 재생합니다.

### 토픽 내부의 `/topic` (인수 없음)

현재 토픽의 연결 상태를 표시합니다: 세션 제목, 세션 ID, `/new`를 사용할지 새 토픽을 만들지에 대한 힌트입니다.

### 내부 동작

- 활성화 상태는 `telegram_dm_topic_mode(chat_id, user_id, enabled, ...)`에 `state.db`로 저장됩니다.
- 각 토픽 연결은 `telegram_dm_topic_bindings(chat_id, thread_id, session_id, ...)`에 `ON DELETE CASCADE`와 함께 저장됩니다. 세션을 정리하면 토픽 연결도 자동으로 삭제됩니다.
- 토픽 모드 SQLite 마이그레이션은 **옵트인**입니다. 첫 `/topic` 호출 시에만 실행되며 게이트웨이 시작 시에는 실행되지 않습니다. 사용자가 이 프로필에서 `/topic`을 실행하기 전까지 `state.db`는 변경되지 않습니다.
- 각 인바운드 DM은 자신의 `(chat_id, thread_id)` 연결을 조회합니다. 연결이 있으면 `SessionStore.switch_session()`을 통해 해당 세션으로 메시지를 라우팅하므로, 세션 키와 세션 ID 매핑이 디스크에서 일관되게 유지됩니다.
- 토픽 내부의 `/new`는 연결 행이 새 세션 ID를 가리키도록 다시 작성하므로, 다음 메시지도 새 세션에 계속 머뭅니다.
- `extra.dm_topics`에 선언된 토픽은 다중 세션 모드가 활성화되어도 **자동으로 이름을 바꾸지 않습니다**. 운영자가 선택한 이름이 보존됩니다.
- `extra.disable_topic_auto_rename: true`를 설정하면 채팅의 **모든** 토픽(스레드 모드로 만든 임시 토픽 포함)에 대한 자동 이름 변경이 꺼집니다.
- 포럼이 활성화된 DM의 General(상단에 고정된) 토픽은 Telegram이 메시지를 `message_thread_id=1`로 전달하든 thread_id 없이 전달하든 루트 로비로 취급됩니다.
- 루트 로비 알림은 채팅당 30초에 한 번으로 제한됩니다. 토픽 모드가 켜진 것을 잊고 루트에서 프롬프트를 열 번 입력해도 답변 열 개가 전송되지 않습니다.
- BotFather 설정 스크린샷은 채팅당 5분에 한 번으로 제한됩니다. Threads Settings가 여전히 비활성화된 상태에서 `/topic`을 반복 시도해도 같은 이미지가 다시 업로드되지 않습니다.
- 토픽 내부에서 시작한 `/background <prompt>`는 결과를 같은 토픽으로 전달합니다. 백그라운드 세션은 해당 토픽의 자동 이름 변경을 유발하지 않습니다.
- `/topic` 자체는 봇의 사용자 인증 확인으로 보호됩니다. 인증되지 않은 DM에는 활성화 대신 거부 응답이 전송됩니다.

### 다중 세션 모드 비활성화

루트 DM에서 `/topic off`를 전송합니다. Hermes는 해당 행을 끄고 채팅의 `(thread_id → session_id)` 연결을 지우며, 루트 DM을 일반 Hermes 채팅으로 되돌립니다. Telegram의 기존 토픽은 삭제되지 않으며, 독립 세션으로 보호되지 않게 될 뿐입니다. 나중에 `/topic`을 다시 실행하면 다시 활성화할 수 있습니다.

수동으로 정리해야 하는 경우(예: 여러 채팅을 일괄 초기화할 때) 행을 직접 삭제합니다:

```bash
sqlite3 ~/.hermes/state.db \
  "UPDATE telegram_dm_topic_mode SET enabled = 0 WHERE chat_id = '<your_chat_id>'; \
   DELETE FROM telegram_dm_topic_bindings WHERE chat_id = '<your_chat_id>';"
```

### Hermes 다운그레이드

`/topic`보다 오래된 Hermes 버전으로 다운그레이드하면 이 기능은 작동을 멈춥니다. `telegram_dm_topic_mode` 및 `telegram_dm_topic_bindings` 테이블은 `state.db`에 남아 있지만 이전 코드에서는 무시됩니다. DM은 기본 스레드별 격리로 되돌아갑니다(`message_thread_id`마다 `build_session_key`를 통해 자체 세션을 계속 받음). 따라서 기존 Telegram 토픽은 병렬 세션으로 계속 작동합니다. 루트 DM은 더 이상 로비가 아니며, 해당 메시지는 예전처럼 에이전트로 전달됩니다. 다시 업그레이드하면 다중 세션 모드가 기존 상태 그대로 재활성화됩니다.

## 그룹 포럼 토픽 스킬 연결

**토픽 모드**가 활성화된 슈퍼그룹(“포럼 토픽”이라고도 함)은 이미 토픽별 세션 격리를 적용합니다. 각 `thread_id`가 자체 대화에 매핑됩니다. 그러나 DM 토픽 스킬 연결과 마찬가지로 특정 그룹 토픽에 메시지가 도착할 때 스킬을 **자동으로 로드**하고 싶을 수 있습니다.

### 사용 사례

워크스트림별 포럼 토픽이 있는 팀 슈퍼그룹:

- **Engineering** 토픽 → `software-development` 스킬을 자동 로드
- **Research** 토픽 → `arxiv` 스킬을 자동 로드
- **General** 토픽 → 스킬 없음, 범용 어시스턴트

### 구성

`~/.hermes/config.yaml`의 `platforms.telegram.extra.group_topics` 아래에 토픽 연결을 추가합니다:

```yaml
platforms:
  telegram:
    extra:
      group_topics:
      - chat_id: -1001234567890       # Supergroup ID
        topics:
        - name: Engineering
          thread_id: 5
          skill: software-development
        - name: Research
          thread_id: 12
          skill: arxiv
        - name: General
          thread_id: 1
          # No skill — general purpose
```

**필드:**

| 필드 | 필수 | 설명 |
|-------|----------|-------------|
| `chat_id` | 예 | `-100`으로 시작하는 슈퍼그룹의 숫자 ID(음수) |
| `name` | 아니요 | 사람이 읽을 수 있는 토픽 레이블(정보 제공용) |
| `thread_id` | 예 | Telegram 포럼 토픽 ID — `t.me/c/<group_id>/<thread_id>` 링크에서 확인 가능 |
| `skill` | 아니요 | 이 토픽의 새 세션에서 자동으로 로드할 스킬 |

### 작동 방식

1. 매핑된 그룹 토픽에 메시지가 도착하면 Hermes가 `group_topics` 구성에서 `chat_id`와 `thread_id`를 조회합니다.
2. 일치하는 항목에 `skill` 필드가 있으면 해당 스킬이 세션에 자동으로 로드됩니다 — DM 토픽 스킬 연결과 동일합니다.
3. `skill` 키가 없는 토픽은 세션 격리만 적용합니다(기존 동작 그대로).
4. 매핑되지 않은 `thread_id` 또는 `chat_id` 값은 조용히 다음 단계로 넘어갑니다 — 오류도, 스킬도 없습니다.

### DM 토픽과의 차이

| | DM 토픽 | 그룹 토픽 |
|---|---|---|
| 구성 키 | `extra.dm_topics` | `extra.group_topics` |
| 토픽 생성 | `thread_id`가 없으면 Hermes가 API로 토픽 생성 | 관리자가 Telegram UI에서 토픽 생성 |
| `thread_id` | 생성 후 자동 입력 | 수동으로 설정해야 함 |
| `icon_color` / `icon_custom_emoji_id` | 지원됨 | 해당 없음(관리자가 외관 제어) |
| 스킬 연결 | ✓ | ✓ |
| 세션 격리 | ✓ | ✓ (포럼 토픽에 이미 기본 제공) |

:::tip
토픽의 `thread_id`를 확인하려면 Telegram Web 또는 Desktop에서 토픽을 열고 URL을 확인합니다: `https://t.me/c/1234567890/5` — 마지막 숫자(`5`)가 `thread_id`입니다. 슈퍼그룹의 `chat_id`는 그룹 ID 앞에 `-100`을 붙인 값입니다(예: 그룹 `1234567890`은 `-1001234567890`이 됨).
:::

## 최근 Bot API 기능

- **Bot API 9.4 (2026년 2월):** Private Chat Topics — 봇이 `createForumTopic`을 사용해 1:1 DM 채팅에 포럼 토픽을 만들 수 있습니다. Hermes는 이를 서로 다른 두 기능에 사용합니다. 운영자가 관리하는 [Private Chat Topics](#private-chat-topics-bot-api-94)(구성 기반의 고정 토픽 목록)와 사용자가 주도하는 [다중 세션 DM 모드](#multi-session-dm-mode-topic)(`/topic`으로 활성화되며 사용자가 토픽을 무제한 생성)입니다.
- **개인정보 보호 정책:** 이제 Telegram은 봇에 개인정보 보호 정책을 요구합니다. BotFather에서 `/setprivacy_policy`로 설정하지 않으면 Telegram이 자리 표시자를 자동으로 생성할 수 있습니다. 봇을 공개적으로 운영하는 경우 특히 중요합니다.
- **Bot API 9.5 (2026년 3월): `sendMessageDraft`를 사용한 네이티브 스트리밍.** Hermes는 Telegram의 네이티브 스트리밍 초안 API를 비공개 채팅에서 옵트인 전송 방식으로 지원합니다. 기본값은 레거시 `editMessageText` 경로입니다. 일부 Telegram 클라이언트에서 초안 미리보기가 눈에 띄게 접혔다가 다시 렌더링될 수 있기 때문입니다.

### 스트리밍 전송 방식(`gateway.streaming.transport`)

스트리밍이 활성화되면(`gateway.streaming.enabled: true`) Hermes는 네 가지 전송 방식 중 하나를 선택합니다:

| 값 | 동작 |
|---|---|
| `auto` (기본값) | 지원되는 채팅(현재는 Telegram DM)에서는 네이티브 초안 스트리밍을 사용하고, 그 외에는 레거시 편집 기반 경로를 사용합니다. 초안 프레임이 실패하면 원활하게 폴백합니다. |
| `draft` | 네이티브 초안을 강제합니다. 채팅이 초안을 지원하지 않으면(예: 그룹/토픽) 다운그레이드를 기록하고 편집 방식으로 폴백합니다. |
| `edit` | 모든 채팅 유형에서 레거시 방식으로 `editMessageText`를 점진적으로 폴링합니다. |
| `off` | 스트리밍을 완전히 비활성화합니다(최종 답변만 전송하며 점진적 업데이트 없음). |

`~/.hermes/config.yaml`에서:

```yaml
gateway:
  streaming:
    enabled: true
    transport: auto    # auto | draft | edit | off
```

**`edit`(기본값)를 사용하는 DM에서 보이는 것** — 게이트웨이가 일반 미리보기 메시지를 전송하고 `editMessageText`를 통해 점진적으로 업데이트하므로 Telegram의 초안 미리보기 접힘/롤백 현상을 피합니다.

**`auto` 또는 `draft`를 사용하는 DM에서 보이는 것** — Telegram이 토큰 단위로 업데이트되는 애니메이션 초안 미리보기를 표시합니다. 답변이 완료되면 일반 메시지로 전송되고 클라이언트에서 초안 미리보기가 자연스럽게 사라집니다. 초안에는 메시지 ID가 없으므로 채팅 기록에 남는 것은 최종 답변입니다.

**그룹, 슈퍼그룹, 포럼 토픽은 어떻게 되나요?** Telegram은 `sendMessageDraft`를 비공개 채팅(DM)으로 제한합니다. 게이트웨이는 그 외 모든 경우에 편집 기반 경로로 투명하게 폴백합니다 — 이전과 동일한 UX입니다.

**초안 프레임이 실패하면 어떻게 되나요?** 어떤 실패든(일시적인 네트워크 오류, 서버 측 거부, 오래된 `python-telegram-bot` 설치) 해당 스트림의 나머지 구간에서 그 응답을 편집 기반 경로로 전환합니다. 다음 응답에서는 새로 시도합니다.

## 렌더링: 서식 있는 메시지, 표 및 링크 미리보기

**서식 있는 메시지(Bot API 10.1).** 레거시 MarkdownV2 경로에서 제대로 처리하기 어려운 구성(표, 작업 목록, 접을 수 있는 `<details>`, 블록 수학)이 포함된 최종 답변은 에이전트의 **원시 markdown**을 사용해 Telegram 네이티브 [`sendRichMessage`](https://core.telegram.org/bots/api#sendrichmessage)로 전송되며, 클라이언트 측 평탄화 없이 네이티브로 렌더링됩니다. DM에서는 기본값인 `rich_drafts: false`가 클라이언트 호환성을 위해 편집 가능한 레거시 초안 경로에서 애니메이션 미리보기를 유지한 뒤, 영구적인 최종 메시지를 `sendRichMessage`로 전송합니다. `rich_drafts: true`로 설정하면 라이브 미리보기에도 `sendRichMessageDraft`를 사용합니다. 편집 기반 스트림은 `editMessageText`의 `rich_message` 매개변수를 통해 기존 미리보기를 제자리에서 최종 메시지로 확정할 수 있습니다. 일반적인 답변(일반 문장, 굵게/기울임, 간단한 목록)은 클라이언트 간 글꼴 굵기와 간격을 일관되게 유지하기 위해 MarkdownV2 경로를 사용합니다.

서식 있는 경로는 콘텐츠가 32,768자 서식 있는 텍스트 제한을 초과하면 자동으로 건너뛰며, Telegram의 거부(이전 `python-telegram-bot`에서 지원되지 않는 엔드포인트, 파서 오류, 너무 큰 블록/열)가 발생하면 MarkdownV2 경로로 **투명하게 폴백**합니다 — 메시지가 손실되지 않습니다. 일시적/네트워크 오류는 조용히 다시 전송하지 않습니다(최종 메시지 중복 없음).

**MarkdownV2 폴백.** 메시지에서 서식 있는 경로를 사용할 수 없으면 Hermes가 마크다운을 MarkdownV2로 변환합니다. MarkdownV2에는 네이티브 표 구문이 없으므로 파이프 표를 정규화합니다:

- **작은 표**는 **행 그룹 글머리 기호**로 평탄화됩니다 — 각 행이 열 제목 아래에서 읽기 쉬운 글머리 기호 목록이 됩니다. 2~4개 열과 짧은 셀에 적합합니다.
- **더 크거나 넓은 표**는 열이 정렬된 **펜스 코드 블록**으로 폴백되므로 내용이 접히지 않습니다.

서식 있는 메시지는 **옵트인**입니다. 현재 Telegram 클라이언트에서는 Bot API 서식 있는 메시지를 일반 텍스트로 복사하기 어려울 수 있고, 명령어 스니펫과 모바일 전달에서 특히 불편하기 때문에 기본값은 레거시 MarkdownV2 경로로 유지됩니다. 표/작업 목록/details/수학을 네이티브로 렌더링하려면 다음을 활성화합니다:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        rich_messages: true
        rich_drafts: false
```

이 설정은 클라이언트 렌더링/복사 호환성을 위한 것입니다. Telegram이 서식 있는 API 호출을 거부하면 Hermes가 이미 자동으로 폴백합니다. `rich_drafts`는 Telegram DM 스트리밍 중 실험적인 서식 있는 초안 미리보기 경로를 제어합니다. Telegram Desktop/macOS에서는 채팅이 다시 그려질 때까지 서식 있는 초안 프레임이 시각적으로 겹쳐 보일 수 있으므로 기본적으로 꺼져 있습니다. 서식 있는 메시지를 활성화한 상태에서 레거시 “항상 코드 블록” 표 동작만 원한다면 `config.yaml`에서 `telegram.pretty_tables: false`를 설정해 표 정규화를 비활성화합니다(기본값: `true`).

**링크 미리보기.** Telegram은 봇 메시지의 URL에 대해 링크 미리보기를 자동 생성합니다. 이를 억제하려면(긴 `/tools` 출력, 링크 열 개를 언급하는 에이전트 답변 등) 다음을 사용합니다:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_link_previews: true
```

활성화하면 Hermes는 모든 발신 메시지에 Telegram의 `LinkPreviewOptions(is_disabled=True)`를 첨부하고, 이전 `python-telegram-bot` 버전에서는 레거시 `disable_web_page_preview` 매개변수로 폴백합니다.
## 그룹 허용 목록

Telegram 그룹과 포럼 채팅에는 서로 독립적인 두 가지 게이트를 설정할 수 있습니다.

- **발신자 사용자 ID** (`group_allow_from` / `TELEGRAM_GROUP_ALLOWED_USERS`) — 그룹/포럼 메시지에만 적용되는 발신자 기준 허용 목록입니다. `TELEGRAM_ALLOWED_USERS`에 사용자를 추가하지 않고도 그룹에서 특정 사용자가 봇을 호출하도록 할 때 사용합니다. `TELEGRAM_ALLOWED_USERS`에 추가하면 DM 접근 권한도 부여됩니다.
- **채팅 ID** (`group_allowed_chats` / `TELEGRAM_GROUP_ALLOWED_CHATS`) — 채팅 기준 허용 목록입니다. 이 그룹/포럼의 모든 구성원이 봇과 상호작용할 수 있습니다. 그룹 구성원 자체를 접근 신호로 사용하는 팀/지원 봇에 유용합니다.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # Global access (DMs + groups). Users here can always invoke the bot.
        allow_from:
          - "123456789"
        # Sender IDs allowed in groups/forums only. Does NOT grant DM access.
        group_allow_from:
          - "987654321"
        # Entire groups/forums — any member is authorized.
        group_allowed_chats:
          - "-1001234567890"
```

동일한 환경 변수는 다음과 같습니다.

```bash
TELEGRAM_ALLOWED_USERS="123456789"
TELEGRAM_GROUP_ALLOWED_USERS="987654321"
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

동작 방식:

- `TELEGRAM_ALLOWED_USERS`는 모든 채팅 유형(DM, 그룹, 포럼)에 적용됩니다.
- `TELEGRAM_GROUP_ALLOWED_USERS`는 그룹/포럼에서 지정된 발신자만 인증합니다. `TELEGRAM_ALLOWED_USERS`에도 등록하지 않는 한 이들은 봇에 DM을 보낼 수 없습니다.
- `TELEGRAM_GROUP_ALLOWED_CHATS`에 있는 채팅은 발신자와 관계없이 모든 구성원을 인증합니다.
- 이 중 어느 항목에서든 `*`를 사용하면 모든 발신자/채팅을 허용합니다.
- 이 설정은 기존의 멘션/패턴 트리거와 `group_topics` + `ignored_threads` 위에 적용됩니다.

### PR #17686 이전 버전에서 마이그레이션

이 분리가 이루어지기 전에는 `TELEGRAM_GROUP_ALLOWED_USERS`가 유일한 설정이었으며, 사용자는 여기에 **채팅 ID**를 입력했습니다. 하위 호환성을 위해 `TELEGRAM_GROUP_ALLOWED_USERS`에 있는 채팅 ID 형식(숫자 `-`로 시작)의 값은 여전히 채팅 ID로 인식되며, 사용 중단 경고가 한 번 기록됩니다. 마이그레이션 방법은 다음과 같습니다.

```bash
# Old (still works, but deprecated)
TELEGRAM_GROUP_ALLOWED_USERS="-1001234567890"

# New
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

### 게스트 @멘션 우회 (`guest_mode`)

일반적인 설정에서 `group_allowed_chats`는 강제 게이트입니다. 목록에 없는 그룹의 메시지는 구성원이 봇을 명시적으로 @멘션하더라도 조용히 삭제됩니다. 지원/팀 봇에는 이것이 올바른 기본값입니다.

좀 더 편한 설정, 즉 봇을 **대부분 조용히 두되 명시적으로 호출할 때만 가끔 사용하고 싶은** 친구 그룹 채팅에서는 `guest_mode`를 활성화하세요.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        group_allowed_chats:
          - "-1001234567890"   # your main allowlisted group
        guest_mode: true       # non-allowlisted groups: allow on @mention only
```

환경 변수로 설정하는 방법:

```bash
TELEGRAM_GUEST_MODE=true
```

기본값: `false`.

`guest_mode: true`이면 허용 목록에 없는 그룹의 메시지는 봇을 명시적으로 @멘션한 경우에만 처리됩니다. 매번 멘션해야 합니다. 게스트 상호작용에는 세션 고정이 적용되지 않으므로, 봇은 호출되지 않은 친구 그룹 스레드에 자동으로 참여하지 않습니다.

DM과 허용 목록에 있는 그룹은 이전과 정확히 동일하게 동작합니다.

## 슬래시 명령어 접근 제어

기본적으로 허용된 모든 사용자는 모든 슬래시 명령어를 실행할 수 있습니다. 허용 목록을 **관리자**(모든 슬래시 명령어 접근)와 **일반 사용자**(명시적으로 활성화한 명령어만 접근)로 나누려면 플랫폼의 `extra` 블록에 `allow_admin_from`과 `user_allowed_commands`를 추가하세요.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # Existing allowlists (unchanged)
        allow_from:
          - "123456789"     # admin
          - "555555555"     # regular user
          - "777777777"     # regular user

        # NEW — admins get all slash commands (built-in + plugin)
        allow_admin_from:
          - "123456789"

        # NEW — non-admin allowed users can only run these slash commands.
        # /help and /whoami are always allowed so users can see their access.
        user_allowed_commands:
          - status
          - model
          - history

        # Optional: separate admin/command lists for groups
        group_allow_admin_from:
          - "123456789"
        group_user_allowed_commands:
          - status
```

**동작 방식:**

- 특정 범위(DM 또는 그룹)의 `allow_admin_from`에 등록된 사용자는 라이브 레지스트리를 통해 등록된 모든 슬래시 명령어(내장 명령어 및 플러그인 등록 명령어)를 실행할 수 있습니다.
- `allow_from`에는 있지만 `allow_admin_from`에는 없는 사용자는 `user_allowed_commands`에 등록된 명령어와 항상 허용되는 `/help`, `/whoami`만 실행할 수 있습니다.
- 일반 채팅(슬래시로 시작하지 않는 메시지)은 영향을 받지 않습니다. 일반 사용자는 평소처럼 에이전트와 대화할 수 있지만, 임의의 명령어를 호출할 수는 없습니다.
- **하위 호환성:** 범위에 `allow_admin_from`이 설정되지 않으면 해당 범위의 슬래시 명령어 제한이 비활성화됩니다. 기존 설치는 변경 없이 계속 작동합니다.
- DM 관리자 상태가 그룹 관리자 상태를 의미하지는 않습니다. 각 범위에는 자체 관리자 목록이 있습니다.
- `group_allow_admin_from`만 설정하면 DM 범위는 제한 없는 상태(하위 호환 모드)로 유지됩니다.

`/whoami`를 사용하면 현재 범위, 자신의 등급(관리자 / 사용자 / 제한 없음), 실행할 수 있는 슬래시 명령어를 확인할 수 있습니다.

## 대화형 모델 선택기

Telegram 채팅에서 인수 없이 `/model`을 보내면 Hermes가 모델 전환을 위한 대화형 인라인 키보드를 표시합니다.

1. **프로바이더 선택** — 사용 가능한 각 프로바이더와 모델 수를 표시하는 버튼입니다(예: 현재 프로바이더에는 `"✓ Anthropic (12)"`, 그 밖의 프로바이더에는 `"OpenAI (15)"`).
2. **모델 선택** — **Prev**/**Next** 탐색, 프로바이더로 돌아가는 **Back** 버튼, **Cancel**이 있는 페이지네이션 모델 목록입니다.

현재 모델과 프로바이더가 상단에 표시됩니다. 모든 탐색은 같은 메시지를 제자리에서 편집하는 방식으로 이루어지므로 채팅이 지저분해지지 않습니다.

:::tip
정확한 모델 이름을 알고 있다면 `/model <name>`을 직접 입력해 선택기를 건너뛸 수 있습니다. `/model <name> --global`을 입력하면 세션 전체에 변경 사항을 유지할 수도 있습니다.
:::

## DNS-over-HTTPS 폴백 IP

일부 제한된 네트워크에서는 `api.telegram.org`가 연결할 수 없는 IP로 확인될 수 있습니다. Telegram 어댑터에는 올바른 TLS 호스트 이름과 SNI를 유지하면서 대체 IP로 연결을 투명하게 재시도하는 **폴백 IP** 메커니즘이 포함되어 있습니다.

### 작동 방식

1. `TELEGRAM_FALLBACK_IPS`가 설정되어 있으면 해당 IP를 직접 사용합니다.
2. 그렇지 않으면 어댑터가 **Google DNS**와 **Cloudflare DNS**에 DNS-over-HTTPS(DoH)로 자동 질의하여 `api.telegram.org`의 대체 IP를 찾습니다.
3. DoH가 반환한 IP 중 시스템 DNS 결과와 다른 IP를 폴백으로 사용합니다.
4. DoH도 차단된 경우 하드코딩된 시드 IP(`149.154.167.220`)를 최후의 수단으로 사용합니다.
5. 폴백 IP로 연결에 성공하면 해당 IP가 "고정"되어 이후 요청에서는 먼저 기본 경로를 재시도하지 않고 직접 사용합니다.

### 설정

```bash
# Explicit fallback IPs (comma-separated)
TELEGRAM_FALLBACK_IPS=149.154.167.220,149.154.167.221
```

또는 `~/.hermes/config.yaml`에 다음과 같이 설정합니다.

```yaml
platforms:
  telegram:
    extra:
      fallback_ips:
        - "149.154.167.220"
```

:::tip
대부분의 경우 직접 설정할 필요가 없습니다. DoH를 통한 자동 검색이 제한된 네트워크 환경의 대부분을 처리합니다. `TELEGRAM_FALLBACK_IPS` 환경 변수는 네트워크에서 DoH까지 차단된 경우에만 필요합니다.
:::

## 프록시 지원

인터넷에 연결하려면 HTTP 프록시가 필요한 네트워크(기업 환경에서 흔히 사용됨)에서는 Telegram 어댑터가 표준 프록시 환경 변수를 자동으로 읽고 모든 연결을 프록시를 통해 라우팅합니다.

### 지원되는 변수

어댑터는 다음 환경 변수를 순서대로 확인하며, 설정된 첫 번째 변수를 사용합니다.

1. `HTTPS_PROXY`
2. `HTTP_PROXY`
3. `ALL_PROXY`
4. `https_proxy` / `http_proxy` / `all_proxy`(소문자 변형)

### 설정

게이트웨이를 시작하기 전에 환경에 프록시를 설정합니다.

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
hermes gateway
```

또는 `~/.hermes/.env`에 추가합니다.

```bash
HTTPS_PROXY=http://proxy.example.com:8080
```

프록시는 기본 전송과 모든 폴백 IP 전송에 적용됩니다. 별도의 Hermes 설정은 필요하지 않습니다. 환경 변수가 설정되어 있으면 자동으로 사용됩니다.

:::note
이 설정은 Hermes가 Telegram 연결에 사용하는 사용자 지정 폴백 전송 계층에 적용됩니다. 다른 곳에서 사용하는 표준 `httpx` 클라이언트는 기본적으로 프록시 환경 변수를 이미 준수합니다.
:::

## 메시지 리액션

봇은 처리 상태를 시각적으로 알려주기 위해 메시지에 이모지 리액션을 추가할 수 있습니다.

- 👀 봇이 메시지 처리를 시작할 때
- ✅ 응답이 성공적으로 전달되었을 때
- ❌ 처리 중 오류가 발생했을 때

리액션은 **기본적으로 비활성화**되어 있습니다. `config.yaml`에서 활성화하세요.

```yaml
telegram:
  reactions: true
```

또는 환경 변수를 사용합니다.

```bash
TELEGRAM_REACTIONS=true
```

:::note
리액션이 누적되는 Discord와 달리 Telegram Bot API는 한 번의 호출에서 봇의 모든 리액션을 대체합니다. 👀에서 ✅/❌로의 전환은 원자적으로 이루어지므로 두 리액션이 동시에 표시되는 일은 없습니다.
:::

:::tip
그룹에서 봇에 리액션을 추가할 권한이 없으면 리액션 호출은 조용히 실패하고 메시지 처리는 정상적으로 계속됩니다.
:::

## 채널별 프롬프트

특정 Telegram 그룹 또는 포럼 주제에 임시 시스템 프롬프트를 할당합니다. 프롬프트는 매 턴 런타임에 주입되며 대화 기록에는 절대 저장되지 않으므로, 변경 사항이 즉시 적용됩니다.

```yaml
telegram:
  channel_prompts:
    "-1001234567890": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "42":  |
      This topic is for creative writing feedback. Be warm and
      constructive.
```

키는 채팅 ID(그룹/슈퍼그룹) 또는 포럼 주제 ID입니다. 포럼 그룹에서는 주제 수준의 프롬프트가 그룹 수준의 프롬프트보다 우선합니다.

- `-1001234567890` 그룹의 주제 `42`에서 보낸 메시지 → 주제 `42`의 프롬프트 사용
- 주제 `99`에서 보낸 메시지(명시적인 항목 없음) → `-1001234567890` 그룹의 프롬프트로 대체
- 항목이 없는 그룹에서 보낸 메시지 → 채널 프롬프트를 적용하지 않음

숫자 형식의 YAML 키는 자동으로 문자열로 정규화됩니다.

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| 봇이 전혀 응답하지 않음 | `TELEGRAM_BOT_TOKEN`이 올바른지 확인합니다. `hermes gateway` 로그에서 오류를 확인합니다. |
| 봇이 "unauthorized"로 응답함 | 사용자 ID가 `TELEGRAM_ALLOWED_USERS`에 없는 것입니다. @userinfobot으로 다시 확인합니다. |
| 봇이 그룹 메시지를 무시함 | 개인정보 보호 모드가 켜져 있을 가능성이 높습니다. 이를 비활성화하거나(3단계) 봇을 그룹 관리자로 지정합니다. **개인정보 보호 설정을 변경한 후에는 봇을 삭제했다가 다시 추가해야 합니다.** |
| 음성 메시지가 텍스트로 변환되지 않음 | STT를 사용할 수 있는지 확인합니다. 로컬 변환에는 `faster-whisper`를 설치하거나, `~/.hermes/.env`에 `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY`를 설정합니다. |
| 음성 답변이 말풍선이 아니라 파일로 전송됨 | `ffmpeg`를 설치합니다(Edge TTS Opus 변환에 필요). |
| 봇 토큰이 폐기되었거나 유효하지 않음 | BotFather에서 `/revoke` 다음에 `/newbot` 또는 `/token`을 사용해 새 토큰을 생성합니다. `.env` 파일을 업데이트합니다. |
| 웹훅이 업데이트를 받지 못함 | `TELEGRAM_WEBHOOK_URL`에 공개적으로 접근할 수 있는지 확인합니다(`curl`로 테스트). 플랫폼/리버스 프록시가 URL의 포트에서 들어오는 HTTPS 트래픽을 `TELEGRAM_WEBHOOK_PORT`로 설정한 로컬 수신 포트로 전달하는지 확인합니다(두 포트 번호는 같을 필요가 없습니다). SSL/TLS가 활성화되어 있는지 확인합니다. Telegram은 HTTPS URL로만 전송합니다. 방화벽 규칙도 확인합니다. |

## 실행 승인

에이전트가 잠재적으로 위험한 명령어를 실행하려고 하면 채팅에서 승인을 요청합니다.

> ⚠️ 이 명령어는 잠재적으로 위험합니다(재귀 삭제). 승인하려면 "yes"라고 답하세요.

승인하려면 "yes"/"y", 거부하려면 "no"/"n"이라고 답합니다.

## 대화형 프롬프트(명확히 하기)

에이전트가 `clarify` 도구를 호출해 선호하는 접근 방식을 묻거나, 작업 후 피드백을 받거나, 사소하지 않은 결정을 내리기 전에 확인할 때 Telegram은 **인라인 키보드 버튼**으로 질문을 표시합니다.

> ❓ 대시보드에는 어떤 프레임워크를 사용해야 하나요?
>
> [1. Next.js] [2. Remix] [3. Astro]
> [✏️ 기타 (답변 입력)]

버튼을 눌러 답하거나 **기타**를 눌러 자유 형식의 응답을 입력할 수 있습니다(다음에 보내는 메시지가 답변이 됩니다). 미리 설정된 선택지가 없는 개방형 `clarify` 호출은 버튼을 건너뛰고 다음 메시지를 그대로 받습니다.

응답 제한 시간은 `~/.hermes/config.yaml`의 `agent.clarify_timeout`에서 설정할 수 있습니다(기본값 `600`초). 제한 시간 내에 응답하지 않으면 에이전트는 센티널 메시지와 함께 대기를 해제하고 멈추지 않고 상황에 맞게 조정합니다.
## 푸시 알림 빈도

Telegram은 봇이 보내는 모든 메시지마다 푸시 알림을 발생시킵니다. 도구 진행 상황 버블, 스트리밍 업데이트, 상태 콜백을 전송하는 긴 에이전트 턴에서는 알림이 빠르게 많아질 수 있습니다. Telegram 어댑터에는 두 가지 알림 모드가 있습니다:

| 모드 | 동작 |
|------|------|
| `important` (기본값) | 최종 응답, 승인 요청, 슬래시 명령 확인에만 알림이 울립니다. 도구 진행 상황, 스트리밍 청크, 상태 메시지는 `disable_notification=true`로 전송됩니다. |
| `all` | 모든 발신 메시지에서 푸시 알림이 발생합니다. 정말로 모든 도구 호출에 대한 알림을 받고 싶은 경우에만 사용하세요. |

`~/.hermes/config.yaml`에서 설정하세요:

```yaml
display:
  platforms:
    telegram:
      notifications: important   # or "all"
```

환경 변수로 재정의할 수도 있습니다(빠른 A/B 테스트에 유용):

```bash
HERMES_TELEGRAM_NOTIFICATIONS=all
```

알 수 없는 값은 경고를 기록하고 `important`로 대체됩니다.

## 상태 메시지의 제자리 편집

Telegram 어댑터는 반복적으로 발생하는 에이전트 상태 콜백(예: "컨텍스트 압축 중…", "도구 호출 중…")을 `send_or_update_status()`를 통해 라우팅합니다. 이 메서드는 후속 전송 시 새 버블을 계속 추가하는 대신 `{(chat_id, status_key) → message_id}` 캐시를 사용해 기존 버블을 편집합니다. 서로 다른 `status_key` 값은 각각 별도의 메시지를 사용하며, 서로 다른 채팅은 절대 충돌하지 않습니다. 편집에 실패하면(예: 사용자가 메시지를 삭제했거나, Telegram에서 편집을 허용하는 기간이 지난 경우) 캐시 항목이 삭제되고 다음 전송에서 새 메시지를 게시한 뒤 ID를 다시 캐시합니다. 별도의 설정은 필요하지 않습니다 — 이것이 Telegram의 기본 동작입니다. `send_or_update_status`를 구현하지 않은 다른 어댑터는 변경 없이 일반 `send()`로 대체됩니다.

## 에이전트 턴 중 수신된 사용자 메시지 고정

사용자가 에이전트 턴을 시작하는 메시지를 보내면 Telegram 어댑터는 해당 턴이 진행되는 동안 수신 메시지를 고정하고, 응답이 완료되면 고정을 해제합니다. 이는 봇이 메시지를 무시하는 것이 아니라 적극적으로 처리하고 있음을 보여 주는 간단한 시각적 표시입니다. 추가 핑을 방지하기 위해 고정에는 `disable_notification=true`가 사용됩니다. 별도의 설정은 필요하지 않습니다.

## 보안

:::warning
봇과 상호작용할 수 있는 사용자를 제한하려면 항상 `TELEGRAM_ALLOWED_USERS`를 설정하세요. 이 값이 없으면 안전 조치로 게이트웨이가 기본적으로 모든 사용자를 거부합니다.
:::

봇 토큰을 공개적으로 공유하지 마세요. 토큰이 노출되었다면 BotFather의 `/revoke` 명령을 통해 즉시 폐기하세요.

자세한 내용은 [보안 문서](/user-guide/security)를 참조하세요. 사용자 인증에 더 유연한 방식을 원한다면 [DM 페어링](/user-guide/messaging#dm-pairing-alternative-to-allowlists)도 사용할 수 있습니다.
