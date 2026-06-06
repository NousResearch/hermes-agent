---
sidebar_position: 1
title: "Telegram"
description: "Set up Hermes Agent as a Telegram bot"
---

# 텔레그램 (Telegram) 설정

Hermes Agent는 완전한 기능을 갖춘 대화형 봇으로 텔레그램(Telegram)과 통합됩니다. 연결이 완료되면 어느 기기에서나 에이전트와 채팅하고, 자동 전사되는 음성 메모를 보내고, 예약된 작업 결과를 수신하고, 그룹 채팅에서 에이전트를 사용할 수 있습니다. 이 통합은 [python-telegram-bot](https://python-telegram-bot.org/)을 기반으로 구축되었으며 텍스트, 음성, 이미지 및 파일 첨부를 지원합니다.

## 1단계: BotFather를 통해 봇 생성

모든 텔레그램 봇에는 텔레그램의 공식 봇 관리 도구인 [@BotFather](https://t.me/BotFather)에서 발급한 API 토큰이 필요합니다.

1. 텔레그램을 열고 **@BotFather**를 검색하거나 [t.me/BotFather](https://t.me/BotFather)를 방문하세요.
2. `/newbot`을 보냅니다.
3. **표시 이름(display name)**을 선택합니다 (예: "Hermes Agent") — 자유롭게 정할 수 있습니다.
4. **사용자 이름(username)**을 선택합니다 — 고유해야 하며 `bot`으로 끝나야 합니다 (예: `my_hermes_bot`).
5. BotFather가 **API 토큰**으로 답장합니다. 다음과 같이 생겼습니다:

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
봇 토큰을 비밀로 유지하세요. 이 토큰이 있는 사람은 누구나 봇을 제어할 수 있습니다. 토큰이 유출된 경우 BotFather에서 `/revoke`를 통해 즉시 취소(revoke)하세요.
:::

## 2단계: 봇 사용자 지정 (선택 사항)

다음 BotFather 명령어들은 사용자 경험을 향상시킵니다. @BotFather에게 메시지를 보내 사용하세요:

| 명령어 | 목적 |
|---------|---------|
| `/setdescription` | 사용자가 채팅을 시작하기 전에 표시되는 "이 봇이 할 수 있는 일은 무엇입니까?" 텍스트 |
| `/setabouttext` | 봇의 프로필 페이지에 표시되는 짧은 텍스트 |
| `/setuserpic` | 봇의 아바타 업로드 |
| `/setcommands` | 명령어 메뉴(채팅창의 `/` 버튼) 정의 |
| `/setprivacy` | 봇이 그룹의 모든 메시지를 볼 수 있는지 여부 제어 (3단계 참조) |

:::tip
`/setcommands`에 유용한 시작 세트:

```
help - Show help information
new - Start a new conversation
sethome - Set this chat as the home channel
```
:::

## 3단계: 개인정보 보호 모드 (그룹의 경우 중요)

텔레그램 봇에는 기본적으로 **개인정보 보호 모드(privacy mode)가 켜져(enabled)** 있습니다. 이는 그룹에서 봇을 사용할 때 가장 혼란을 일으키는 원인입니다.

**개인정보 보호 모드가 켜져 있으면(ON)** 봇은 다음 메시지만 볼 수 있습니다:
- `/` 명령어로 시작하는 메시지
- 봇 자신의 메시지에 대한 직접적인 답장
- 서비스 메시지(멤버 가입/탈퇴, 고정된 메시지 등)
- 봇이 관리자인 채널의 메시지

**개인정보 보호 모드가 꺼져 있으면(OFF)** 봇은 그룹의 모든 메시지를 수신합니다.

### 개인정보 보호 모드 비활성화 방법

1. **@BotFather**에게 메시지를 보냅니다.
2. `/mybots`를 보냅니다.
3. 내 봇을 선택합니다.
4. **Bot Settings → Group Privacy → Turn off**로 이동합니다.

:::warning
개인정보 보호 설정을 변경한 후에는 **그룹에서 봇을 제거했다가 다시 추가해야 합니다**. 텔레그램은 봇이 그룹에 가입할 때 개인정보 보호 상태를 캐시하며, 봇을 제거했다가 다시 추가할 때까지 업데이트되지 않습니다.
:::

:::tip
개인정보 보호 모드를 비활성화하는 대안: 봇을 **그룹 관리자(group admin)**로 승격시킵니다. 관리자 봇은 개인정보 보호 설정과 관계없이 항상 모든 메시지를 수신하므로, 전역 개인정보 보호 모드를 전환할 필요가 없습니다.
:::

### 자동 답장 없이 그룹 채팅 관찰하기

OpenClaw/Yuanbao 스타일의 그룹 동작을 위해, 봇이 일반 그룹 메시지를 **볼** 수는 있지만 명시적으로 트리거될 때만 **응답**하도록 텔레그램을 구성할 수 있습니다:

```yaml
telegram:
  allowed_chats:
    - "-1001234567890"
  group_allowed_chats:
    - "-1001234567890"
  require_mention: true
  observe_unmentioned_group_messages: true
```

이 모드가 활성화되면, 명시적으로 허용 목록에 있는 채팅/주제에서 멘션되지 않은 그룹 메시지는 관찰된 컨텍스트(observed context)로서 공유 채팅/주제 세션 스크립트에 추가되지만, 에이전트를 파견(dispatch)하지는 않습니다. `allowed_chats`는 봇이 응답하는 위치를 통제하고, `group_allowed_chats`는 관찰된 컨텍스트에 사용되는 공유 그룹 세션을 승인하므로, 이 모드에서는 동일한 채팅 ID를 사용하세요. 나중에 동일한 허용된 채팅/주제에서 `@botname` 멘션, 봇에 대한 답장 또는 구성된 멘션 패턴이 발생하면 해당 관찰된 컨텍스트를 사용할 수 있습니다. 또한, 트리거된 메시지는 `[nickname|user_id]`로 태그 지정되고 턴별 안전 프롬프트를 받게 되어, 모델이 이전의 관찰된 줄들을 봇에게 전달된 지시 사항이 아닌 컨텍스트로 취급하도록 합니다.

해당 환경 변수:

```bash
TELEGRAM_ALLOWED_CHATS=-1001234567890
TELEGRAM_GROUP_ALLOWED_CHATS=-1001234567890
TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES=true
```

이 기능을 사용하려면 텔레그램이 일반 그룹 메시지를 게이트웨이로 전달해야 하므로 위에서 설명한 대로 BotFather 개인정보 보호 모드를 비활성화하거나 봇을 그룹 관리자로 승격시키세요.

## 4단계: 사용자 ID(User ID) 찾기

Hermes Agent는 액세스를 제어하기 위해 숫자 형태의 텔레그램 사용자 ID를 사용합니다. 사용자 ID는 사용자 이름(username)이 **아니라** `123456789`와 같은 숫자입니다.

**방법 1 (권장):** [@userinfobot](https://t.me/userinfobot)에게 메시지를 보내면 사용자 ID를 즉시 답장해 줍니다.

**방법 2:** [@get_id_bot](https://t.me/get_id_bot)에게 메시지를 보냅니다 — 또 다른 안정적인 옵션입니다.

이 숫자를 저장해 두세요. 다음 단계에서 필요합니다.

## 5단계: Hermes 구성

### 옵션 A: 대화형 설정 (권장)

```bash
hermes gateway setup
```

메시지가 나타나면 **Telegram**을 선택하세요. 마법사가 봇 토큰과 허용된 사용자 ID를 묻고 구성을 작성해 줍니다.

### 옵션 B: 수동 구성

`~/.hermes/.env`에 다음을 추가하세요:

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789    # 여러 사용자의 경우 쉼표로 구분
```

### 게이트웨이 시작

```bash
hermes gateway
```

몇 초 안에 봇이 온라인 상태가 될 것입니다. 텔레그램에서 봇에게 메시지를 보내 확인해 보세요.

## Docker 지원 터미널에서 생성된 파일 보내기

터미널 백엔드가 `docker`인 경우, 텔레그램 첨부 파일은 컨테이너 내부가 아닌 **게이트웨이 프로세스**에 의해 전송된다는 점을 명심하세요. 즉, 최종 `MEDIA:/...` 경로는 게이트웨이가 실행 중인 호스트에서 읽을 수 있어야 합니다.

흔히 하는 실수:

- 에이전트가 Docker 내부에서 `/workspace/report.txt`에 파일을 씁니다.
- 모델이 `MEDIA:/workspace/report.txt`를 출력합니다.
- `/workspace/report.txt`는 호스트가 아닌 컨테이너 내부에만 존재하므로 텔레그램 전송이 실패합니다.

권장 패턴:

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/.hermes/cache/documents:/output"
```

그런 다음:

- Docker 내부에서 `/output/...`에 파일을 씁니다.
- `MEDIA:`에서 **호스트에서 볼 수 있는(host-visible)** 경로를 출력합니다. 예:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`

이미 `docker_volumes:` 섹션이 있는 경우 새 마운트를 동일한 목록에 추가하세요. YAML 중복 키는 이전 키를 조용히 덮어씁니다.

### 지원되는 `MEDIA:` 파일 확장자

게이트웨이는 에이전트 답장에서 `MEDIA:/path/to/file` 태그를 추출하고 참조된 파일을 플랫폼 기본 첨부 파일로 전송합니다. 모든 게이트웨이 플랫폼에서 지원되는 확장자:

| 범주 | 확장자 |
|---|---|
| 이미지 | `png`, `jpg`, `jpeg`, `gif`, `webp`, `bmp`, `tiff`, `svg` |
| 오디오 | `mp3`, `wav`, `ogg`, `m4a`, `opus`, `flac`, `aac` |
| 비디오 | `mp4`, `mov`, `webm`, `mkv`, `avi` |
| **문서** | `pdf`, `txt`, `md`, `csv`, `json`, `xml`, `html`, `yaml`, `yml`, `log` |
| **Office** | `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp` |
| **압축 파일** | `zip`, `rar`, `7z`, `tar`, `gz`, `bz2` |
| **도서 / 패키지** | `epub`, `apk`, `ipa` |

이 목록에 있는 모든 것은 이를 지원하는 플랫폼(Telegram, Discord, Signal, Slack, WhatsApp, Feishu, Matrix 등)에서 기본 첨부 파일로 제공됩니다. 네이티브 지원이 없는 플랫폼에서는 링크 또는 일반 텍스트 표시기로 대체됩니다. **굵게** 표시된 범주는 최근 릴리스에서 추가되었습니다. 모델이 `여기에 파일이 있습니다: /path/to/report.docx`라고 말하는 것에 의존했다면 네이티브 전송을 위해 `MEDIA:/path/to/report.docx`로 교체하세요.

## 웹훅 모드 (Webhook Mode)

기본적으로 Hermes는 **롱 폴링(long polling)**을 사용하여 텔레그램에 연결합니다. 게이트웨이가 새 업데이트를 가져오기 위해 텔레그램 서버에 아웃바운드 요청을 보냅니다. 이는 로컬 및 항상 켜져 있는 배포 환경에서 잘 작동합니다.

**클라우드 배포**(Fly.io, Railway, Render 등)의 경우 **웹훅 모드(webhook mode)**가 더 비용 효율적입니다. 이러한 플랫폼은 인바운드 HTTP 트래픽에서 일시 중지된 시스템을 자동으로 깨울 수 있지만 아웃바운드 연결에서는 깨울 수 없습니다. 폴링은 아웃바운드이므로 폴링 봇은 절대 절전 모드로 들어갈 수 없습니다. 웹훅 모드는 방향을 뒤집습니다 — 텔레그램이 업데이트를 봇의 HTTPS URL로 푸시하여, 유휴 상태일 때 절전(sleep-when-idle) 배포가 가능해집니다.

| | 폴링 (기본값) | 웹훅 |
|---|---|---|
| 방향 | 게이트웨이 → 텔레그램 (아웃바운드) | 텔레그램 → 게이트웨이 (인바운드) |
| 적합한 환경 | 로컬, 항상 켜져 있는 서버 | 자동 깨우기(auto-wake) 기능이 있는 클라우드 플랫폼 |
| 설정 | 추가 구성 없음 | `TELEGRAM_WEBHOOK_URL` 설정 |
| 유휴 비용 | 시스템이 계속 실행되어야 함 | 메시지 수신 사이에 시스템이 절전할 수 있음 |

### 구성

`~/.hermes/.env`에 다음을 추가하세요:

```bash
TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
TELEGRAM_WEBHOOK_SECRET="$(openssl rand -hex 32)"  # 필수
# TELEGRAM_WEBHOOK_PORT=8443        # 선택, 기본값 8443
```

| 변수 | 필수 | 설명 |
|----------|----------|-------------|
| `TELEGRAM_WEBHOOK_URL` | 예 | 텔레그램이 업데이트를 보낼 공개 HTTPS URL입니다. URL 경로는 자동 추출됩니다(예: 위 예제의 `/telegram`). |
| `TELEGRAM_WEBHOOK_SECRET` | **예** (`TELEGRAM_WEBHOOK_URL`이 설정된 경우) | 확인을 위해 모든 웹훅 요청에서 텔레그램이 에코하는 비밀 토큰입니다. 이 토큰이 없으면 게이트웨이 시작이 거부됩니다 — [GHSA-3vpc-7q5r-276h](https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h) 참조. `openssl rand -hex 32`로 생성하세요. |
| `TELEGRAM_WEBHOOK_PORT` | 아니요 | 웹훅 서버가 수신 대기하는 로컬 포트입니다 (기본값: `8443`). |

`TELEGRAM_WEBHOOK_URL`이 설정되면 게이트웨이는 폴링 대신 HTTP 웹훅 서버를 시작합니다. 설정되지 않은 경우 폴링 모드가 사용되며 이전 버전과 동작이 변경되지 않습니다.

### 클라우드 배포 예시 (Fly.io)

1. Fly.io 앱 secrets에 환경 변수를 추가합니다:

```bash
fly secrets set TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
fly secrets set TELEGRAM_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

2. `fly.toml`에서 웹훅 포트를 노출합니다:

```toml
[[services]]
  internal_port = 8443
  protocol = "tcp"

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

3. 배포합니다:

```bash
fly deploy
```

게이트웨이 로그에 `[telegram] Connected to Telegram (webhook mode)`가 표시되어야 합니다.

## 프록시 지원

텔레그램 API가 차단되었거나 트래픽을 프록시를 통해 라우팅해야 하는 경우 텔레그램 전용 프록시 URL을 설정하세요. 이는 일반적인 `HTTPS_PROXY` / `HTTP_PROXY` 환경 변수보다 우선합니다.

**옵션 1: config.yaml (권장)**

```yaml
telegram:
  proxy_url: "socks5://127.0.0.1:1080"
```

**옵션 2: 환경 변수**

```bash
TELEGRAM_PROXY=socks5://127.0.0.1:1080
```

지원되는 스킴(schemes): `http://`, `https://`, `socks5://`.

프록시는 기본 텔레그램 연결과 폴백(fallback) IP 전송 모두에 적용됩니다. 텔레그램 전용 프록시가 설정되지 않은 경우 게이트웨이는 `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` (또는 macOS 시스템 프록시 자동 감지)로 폴백합니다.

## 홈 채널 (Home Channel)

모든 텔레그램 채팅(DM 또는 그룹)에서 `/sethome` 명령을 사용하여 해당 채널을 **홈 채널(home channel)**로 지정할 수 있습니다. 예약된 작업(cron jobs)은 이 채널로 결과를 전달합니다.

`~/.hermes/.env`에서 수동으로 설정할 수도 있습니다:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

:::tip
그룹 채팅 ID는 음수(예: `-1001234567890`)입니다. 개인 DM 채팅 ID는 본인의 사용자 ID와 동일합니다.
:::

### 주제(topic) 모드에서의 Cron 전송

봇 DM에서 주제 모드가 활성화되어 있는 경우, 루트 채팅으로 전달되는 cron 메시지는 시스템 전용 로비에 도착합니다. 여기서 답장하면 세션이 열리지 않고 "기본 채팅은 시스템 명령용으로 예약되어 있습니다"라는 알림이 표시됩니다. 전용 포럼 주제(예: `Cron`)를 만들고 다음과 같이 설정하세요:

```bash
TELEGRAM_CRON_THREAD_ID=<topic_thread_id>
```

`TELEGRAM_CRON_THREAD_ID`는 cron 전송에 한해 `TELEGRAM_HOME_CHANNEL_THREAD_ID`를 덮어씁니다. 해당 주제의 답장은 주제의 기존 세션을 계속합니다.

## 음성 메시지

### 수신 음성 (Speech-to-Text)

텔레그램에서 보낸 음성 메시지는 Hermes에 구성된 STT 제공자를 통해 자동으로 전사(transcription)되어 텍스트로 대화에 주입됩니다.

- `local`은 Hermes를 실행하는 머신에서 `faster-whisper`를 사용합니다 — API 키가 필요하지 않습니다.
- `groq`는 Groq Whisper를 사용하며 `GROQ_API_KEY`가 필요합니다.
- `openai`는 OpenAI Whisper를 사용하며 `VOICE_TOOLS_OPENAI_KEY`가 필요합니다.

#### STT 건너뛰기: 원시 오디오 파일을 에이전트에 전달하기

화자 분리(diarization), 사용자 정의 전사 도구 또는 단순히 녹음 파일 보관을 위해 **에이전트가 직접** 오디오를 처리하게 하려면 `~/.hermes/config.yaml`에서 `stt.enabled: false`를 설정하세요:

```yaml
stt:
  enabled: false
```

STT가 비활성화된 경우, 게이트웨이는 여전히 음성/오디오 첨부 파일을 Hermes의 오디오 캐시로 다운로드하지만 **전사하지는 않습니다**. 에이전트는 다음과 같은 마커가 포함된 메시지를 받습니다:

```
[The user sent a voice message: /home/<user>/.hermes/cache/audio/<hash>.ogg]
```

그러면 사용자 도구나 스킬이 해당 경로를 직접 읽을 수 있습니다 (예: 로컬 화자 분리 파이프라인, 더 풍부한 전사 모델로 전달하거나, 장기 스토리지에 업로드). 파일 확장자는 텔레그램이 전송한 원래 형식을 반영합니다 (음성 메모의 경우 `.ogg`, 오디오 첨부 파일의 경우 `.mp3`/`.m4a` 등).

이 기능은 처리하려는 녹음 길이가 1~2분을 넘을 때 유용한 아래의 [로컬 Bot API 서버](#대용량-파일20mb-초과-로컬-bot-api-서버-경유) 섹션과 자연스럽게 연결되어, 텔레그램의 20MB `getFile` 한도를 2GB로 올려줍니다.

### 발신 음성 (Text-to-Speech)

에이전트가 TTS를 통해 오디오를 생성하면 기본 텔레그램 **음성 버블(voice bubbles)**(인라인 재생이 가능한 둥근 형태)로 전달됩니다.

- **OpenAI와 ElevenLabs**는 기본적으로 Opus를 생성합니다 — 추가 설정이 필요하지 않습니다.
- **Edge TTS** (기본 무료 제공자)는 MP3를 출력하며 Opus로 변환하려면 **ffmpeg**가 필요합니다:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

ffmpeg가 없으면 Edge TTS 오디오는 일반 오디오 파일로 전송됩니다 (재생 가능하지만 음성 버블 대신 직사각형 플레이어가 사용됨).

`config.yaml`의 `tts.provider` 키 아래에서 TTS 제공자를 구성하세요.

## 대용량 파일(>20MB) (로컬 Bot API 서버 경유)

텔레그램의 **공개** Bot API는 `getFile` 다운로드를 **20 MB**로 제한하므로, 이 한도를 초과하는 모든 음성 메모, 오디오 파일, 비디오 또는 문서는 Hermes에서 "너무 큼(too large)"이라는 응답과 함께 조용히 거부됩니다. 이를 피하기 위해 문서화된 방법은 네트워크에서 텔레그램이 사용하는 것과 동일한 서버 소프트웨어인 **로컬** [telegram-bot-api](https://github.com/tdlib/telegram-bot-api) 데몬을 실행하는 것입니다. 로컬 서버는 파일 한도를 **2 GB**로 올리며, 사용자 지정 `base_url`이 구성된 것을 확인하면 Hermes는 자체 내부 한도를 자동으로 해제합니다.

이를 통해 다음과 같은 작업 흐름이 가능해집니다:

- 긴 음성 메모(45분 회의, 팟캐스트)를 봇으로 전송
- 비전 도구(vision-tool) 처리를 위해 대용량 비디오 업로드
- 화자 분리, 정렬 또는 훈련 데이터와 같은 오프라인 파이프라인을 위해 원시 오디오 보관

### 1단계: 텔레그램 API 자격 증명 얻기

로컬 서버는 공개 Bot API가 아닌 텔레그램의 MTProto 계층과 직접 통신하므로 **MTProto 자격 증명**이 필요합니다:

1. [my.telegram.org/apps](https://my.telegram.org/apps)를 방문하여 텔레그램 계정으로 로그인합니다.
2. 새 애플리케이션을 만듭니다 (이름과 짧은 설명은 자유롭게 지정 가능).
3. `api_id`와 `api_hash`를 복사합니다 — 둘 다 필수입니다.

### 2단계: telegram-bot-api 서버 실행

커뮤니티에서 유지 관리하는 [`aiogram/telegram-bot-api`](https://hub.docker.com/r/aiogram/telegram-bot-api) Docker 이미지가 가장 쉬운 경로입니다. 최소한의 `docker-compose.yaml` (더 높은 한도를 활성화하려면 `--local` 모드 사용):

```yaml
services:
  tg-bot-api:
    image: aiogram/telegram-bot-api:latest
    container_name: tg-bot-api
    restart: unless-stopped
    ports:
      - "127.0.0.1:8081:8081"   # 루프백에만 바인딩; 보안 참고 사항 참조
    environment:
      TELEGRAM_API_ID: "12345"           # 1단계의 api_id
      TELEGRAM_API_HASH: "abcdef..."     # 1단계의 api_hash
      TELEGRAM_LOCAL: "1"                # --local 모드 활성화 (20MB → 2GB 한도 상승)
    volumes:
      - ./tg-bot-api-data:/var/lib/telegram-bot-api
```

컨테이너 실행:

```bash
docker compose up -d tg-bot-api
docker logs --tail 20 tg-bot-api
```

:::warning 보안
로컬 Bot API 서버는 **추가 인증 없이** URL 경로(예: `/bot<TOKEN>/getMe`)에 봇 토큰을 받습니다. 포트에 연결할 수 있는 사람은 누구나 봇을 완전히 제어할 수 있습니다(봇이 볼 수 있는 모든 메시지 읽기, 봇으로서 메시지 보내기 등). 컨테이너를 `127.0.0.1`에 바인딩하거나 사설 네트워크에서 리버스 프록시 뒤에 두세요. **절대 공용 인터넷에 포트 8081을 노출하지 마세요.**
:::

### 3단계: 공개 API에서 봇 로그아웃 (1회성)

봇은 한 번에 **하나의** Bot API 서버에서만 활성화될 수 있습니다. 봇이 이미 `api.telegram.org`에서 실행 중이었다면(거의 확실히 그랬을 것입니다), 로컬 서버가 이를 수락하기 전에 해당 서버에서 명시적으로 로그아웃해야 합니다:

```bash
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/logOut"
# 예상 응답: {"ok":true,"result":true}
```

이것은 1회성 마이그레이션 단계이므로 재시작할 때마다 반복할 필요가 없습니다. 텔레그램은 `logOut` 이후에 수신된 모든 메시지를 새 서버를 통해 전달합니다.

로컬 서버가 봇을 대신하여 텔레그램과 통신할 수 있는지 확인합니다:

```bash
curl "http://127.0.0.1:8081/bot<YOUR_BOT_TOKEN>/getMe"
# 예상 응답: {"ok":true,"result":{"id":...,"is_bot":true,...}}
```

### 4단계: Hermes가 로컬 서버를 가리키도록 지정

`~/.hermes/config.yaml`의 `platforms.telegram.extra` 아래에 URL을 추가합니다:

```yaml
platforms:
  telegram:
    extra:
      base_url: "http://127.0.0.1:8081/bot"
      base_file_url: "http://127.0.0.1:8081/file/bot"
      local_mode: true        # 아래 5단계 참조 — 봇의 데이터 디렉터리를 Hermes 프로세스가 읽을 수 있는 경우에만 이 항목을 설정하세요.
```

:::caution `telegram.extra`가 아닌 `platforms.telegram.extra`를 사용하세요
현재 `platforms.<name>.extra` 형태만 플랫폼 구성에 딥 병합(deep-merged)됩니다. 최상위 `telegram.extra` 블록 바로 아래에 위치한 키들은 조용히 삭제됩니다.
:::

`base_url`이 설정되면 Hermes는 다음과 같이 동작합니다:

- 로컬 서버를 대상으로 python-telegram-bot 클라이언트를 구축합니다.
- 문서/오디오 크기의 내부 한도를 20 MB에서 2 GB로 자동 상향 조정합니다.
- 현재 사용 중인 모드를 분명히 알 수 있도록 "너무 큼(too large)" 오류 메시지(`Maximum: 2048 MB.`)에 활성 한도를 보고합니다.

게이트웨이를 다시 시작하고 확인 로그 줄을 찾으세요:

```bash
hermes gateway restart
grep -E "Using custom Telegram base_url|Using Telegram local_mode" ~/.hermes/logs/gateway.log | tail
```

### 5단계: `local_mode` — 디스크 파일 액세스

로컬 서버가 파일을 전달하는 **두 가지 방법**이 있습니다:

1. **`--local` 없음** (기본값): 파일은 공개 Bot API와 마찬가지로 `/file/bot<TOKEN>/<path>`에서 HTTP를 통해 제공됩니다. 20MB 제한이 유지됩니다. 네트워크 수정 전용(예: `api.telegram.org`에 연결할 수 없지만 자체 호스팅할 수 있는 경우)으로 유용하며, 크기 한도를 올리고 싶은 경우에는 원치 않는 방식입니다.
2. **`--local` 있음** (위에서 `TELEGRAM_LOCAL=1`을 통해 설정됨): 파일이 서버의 파일 시스템에 기록되고 `getFile` 응답은 HTTP URL 대신 **절대 경로(absolute path)**를 반환합니다. 20MB 제한이 해제됩니다. 그런 다음 Hermes는 HTTP가 아닌 **디스크에서** 바이트를 읽어야 합니다.

디스크 읽기 경로가 작동하도록 하려면 위 구성에서 `local_mode: true`를 설정하고 **또한** Hermes 프로세스가 서버가 반환하는 경로를 읽을 수 있는지 확인해야 합니다. 두 가지 시나리오가 있습니다:

- **동일한 머신** — telegram-bot-api와 Hermes가 같은 호스트에서 실행됩니다. Hermes가 읽을 수 있는 디렉터리(예: `/var/lib/telegram-bot-api`)에 데이터 볼륨을 바인드 마운트하고 파일 소유권이 일치하는지 확인하세요. 컨테이너는 내부 `telegram-bot-api` 사용자(uid는 이미지에 따라 다름)로 권한을 삭제합니다. 가장 간단한 수정 방법은 작성 서비스에 `user: "<UID>:<GID>"`를 추가하여 파일이 Hermes가 이미 실행 중인 uid의 소유가 되도록 하는 것입니다.
- **다른 머신** — 봇 서버는 하나의 호스트(예: NAS, 별도의 VM)에서 실행되고 Hermes는 다른 호스트에서 실행됩니다. 서버의 데이터 디렉터리는 서버가 보고하는 **동일한 절대 경로**(일반적으로 `/var/lib/telegram-bot-api`)로 Hermes 머신과 공유되어야 합니다. NFS가 이에 잘 작동하며, 파일 시스템 수준의 uid 불일치를 처리하고 싶지 않다면 `uid=` 마운트 재매핑이 있는 CIFS/SMB가 더 친숙합니다.

`local_mode: true`가 설정되었지만 Hermes가 반환된 파일 경로를 `stat`할 수 없는 경우(권한 문제 또는 잘못된 마운트), python-telegram-bot은 로컬 서버에 대한 HTTP `getFile`로 조용히 폴백합니다. 이 서버는 `--local` 모드에서 `404 Not Found`로 응답합니다. `gateway.log`에 다음과 같이 증상이 나타납니다:

```
[Telegram] Failed to cache voice: Not Found
telegram.error.InvalidToken: Not Found
```

이 메시지가 보인다면 크기 한도 증가는 작동하지만 파일 공유가 작동하지 않는 것입니다. 게이트웨이가 실행되는 사용자로 Hermes 호스트에서 `ls -la /var/lib/telegram-bot-api/<TOKEN>/voice/`를 확인하고 단일 파일이 권한 오류 없이 `cat`으로 열리는지 확인하세요.

### 6단계: 테스트

20MB보다 큰 음성 메모나 오디오 파일을 봇에 보냅니다. 게이트웨이 로그를 추적(tail)하세요:

```bash
tail -f ~/.hermes/logs/gateway.log | grep -iE "telegram|cache"
```

`[Telegram] Cached user voice at /home/<user>/.hermes/cache/audio/...` 라인이 보여야 하며 "too large" 거부가 **없어야** 합니다. (위의) `stt.enabled: false`와 결합하면 원본 오디오 파일의 경로가 에이전트의 인바운드 메시지에 도착하여 후속 처리에 사용할 수 있습니다.

## 그룹 채팅 사용 (Group Chat Usage)

Hermes Agent는 몇 가지 고려 사항과 함께 텔레그램 그룹 채팅에서 작동합니다:

- **개인정보 보호 모드(Privacy mode)**는 봇이 볼 수 있는 메시지를 결정합니다. ([3단계](#3단계-개인정보-보호-모드-그룹의-경우-중요) 참조)
- `TELEGRAM_ALLOWED_USERS`는 여전히 적용됩니다 — 그룹 내에서도 승인된 사용자만 봇을 트리거할 수 있습니다.
- `telegram.require_mention: true`를 사용하여 봇이 일반 그룹 대화에 응답하지 않게 할 수 있습니다.
- `telegram.require_mention: true`일 때 그룹 메시지는 다음과 같은 경우에 허용됩니다:
  - 봇의 메시지 중 하나에 대한 답장
  - `@botusername` 멘션
  - `/command@botusername` (봇 이름이 포함된 텔레그램의 봇 메뉴 명령 형식)
  - `telegram.mention_patterns`에 구성된 정규식 웨이크 워드 중 하나와 일치
- 여러 Hermes 봇이 있는 그룹에서는 `telegram.exclusive_bot_mentions`이 라우팅을 확정적으로 유지합니다. 메시지가 텔레그램 봇의 사용자 이름을 하나 이상 명시적으로 멘션할 때 멘션된 봇 프로필만 이를 처리합니다. 다른 Hermes 봇은 답장(reply) 및 웨이크 워드(wake-word) 폴백이 실행되기 전에 이를 무시합니다. 이는 기본적으로 활성화되어 있습니다.
- 그룹에서 자유로운 응답이나 멘션 트리거 답장이 허용된 경우에도 특정 텔레그램 포럼 주제에서 Hermes를 침묵시키려면 `telegram.ignored_threads`를 사용하세요.
- `telegram.require_mention`이 설정되지 않거나 false이면 Hermes는 기존의 열린 그룹 동작(open-group behavior)을 유지하여 자신이 볼 수 있는 정상적인 그룹 메시지에 응답합니다.

### 한 그룹에 여러 Hermes 봇 사용하기

동일한 텔레그램 그룹에서 여러 Hermes 프로필을 실행하려면, 프로필당 하나의 텔레그램 봇 토큰을 만들고 프로필당 하나의 게이트웨이를 시작하세요. 여러 게이트웨이 실행에서 동일한 봇 토큰을 재사용하지 마세요. 텔레그램은 동일한 토큰에 대한 동시 폴링을 거부합니다.

권장 그룹 구성:

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns: []
```

이 설정을 사용하면 `@research_bot @ops_bot 이거 요약해줘`와 같은 그룹 메시지는 `research_bot`과 `ops_bot`에 의해서만 처리됩니다. 해당 메시지가 그들의 이전 메시지 중 하나에 대한 답장이거나 공유 웨이크 워드와 일치하더라도 그룹의 다른 Hermes 봇은 침묵을 유지합니다.

명시적인 멘션이 답장(reply)이나 웨이크 워드 트리거를 오버라이드하지 않아야 하는 레거시 공유 트리거 동작(legacy shared-trigger behavior)을 의도한 그룹에 대해서만 `exclusive_bot_mentions: false`를 설정하세요.

여러 프로필을 운영하려면 프로필당 한 번씩 게이트웨이 명령을 실행하세요. 예를 들어:

```bash
# 기본 프로필
hermes gateway start
hermes gateway status
hermes gateway stop

# 이름이 지정된 프로필
hermes -p research gateway start
hermes -p research gateway status
hermes -p research gateway stop
```

소규모 고정 플릿(fleet)의 경우, 기본 프로필에 대해 `hermes gateway <action>`을 호출하고 각 명명된 프로필에 대해 `hermes -p <profile> gateway <action>`을 호출하는 셸 루프 또는 스크립트를 사용하세요. 단일 프로세스 수준의 명령이 모든 서비스 관리자(service manager)의 명명된 모든 프로필을 제어한다고 가정하는 것보다 이것이 더 안정적입니다.

### 문제 해결: DM에서는 작동하지만 그룹에서는 작동하지 않음

봇이 비공개 채팅에서는 응답하지만 그룹에서는 침묵을 유지하는 경우 다음 게이트들을 순서대로 확인하세요:

1. **텔레그램 전송(Telegram delivery):** BotFather 개인정보 보호 모드를 끄거나 봇을 관리자로 승격시키거나 봇을 직접 멘션하세요. 텔레그램이 봇에게 아예 전송하지 않은 그룹 메시지에는 Hermes가 응답할 수 없습니다.
2. **개인정보 보호 설정 변경 후 재가입:** BotFather 개인정보 보호 설정을 변경한 후에는 그룹에서 봇을 제거했다가 다시 추가하세요. 텔레그램은 기존 멤버십에 대해 이전 전송 동작을 유지할 수 있습니다.
3. **Hermes 인증:** 발신자가 `TELEGRAM_ALLOWED_USERS` 또는 `TELEGRAM_GROUP_ALLOWED_USERS`에 나열되어 있는지 확인하거나, `TELEGRAM_GROUP_ALLOWED_CHATS`를 통해 그룹 채팅을 허용하세요.
4. **멘션 필터:** `telegram.require_mention: true`가 설정된 경우, 메시지가 슬래시 명령, 봇에 대한 답장, `@botusername` 멘션 또는 구성된 `mention_patterns` 일치가 아닌 한 일반 그룹 대화는 무시됩니다.
5. **다중 봇 라우팅:** 그룹에 여러 봇이 포함된 경우, 각 Hermes 프로필이 고유한 봇 토큰을 사용하고 레거시 공유 트리거 동작을 의도적으로 원하지 않는 한 `exclusive_bot_mentions`를 활성화된 상태로 유지하세요.

음수 채팅 ID는 텔레그램 그룹과 수퍼그룹에 대해 정상입니다. 채팅 범위(chat-scoped) 인증을 사용하는 경우 이러한 ID를 사용자 전송자 허용 목록이 아닌 `TELEGRAM_GROUP_ALLOWED_CHATS`에 넣으세요.

### 그룹 트리거 구성 예시

`~/.hermes/config.yaml`에 다음을 추가합니다:

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

이 예제는 모든 일반적인 직접 트리거에 더해, `@mention`을 사용하지 않더라도 `chompy`로 시작하는 메시지를 허용합니다.
텔레그램 주제(topics) `31` 및 `42`의 메시지는 멘션 및 자유-응답(free-response) 확인이 실행되기 전에 항상 무시됩니다.

### `mention_patterns` 참고 사항

- 패턴은 Python 정규 표현식을 사용합니다.
- 매칭은 대소문자를 구분하지 않습니다.
- 패턴은 텍스트 메시지와 미디어 자막 모두에 대해 확인됩니다.
- 잘못된 정규식 패턴은 봇을 충돌시키지 않고 게이트웨이 로그에 경고를 남긴 채 무시됩니다.
- 메시지의 시작 부분에서만 패턴이 일치하게 하려면 `^`로 고정(anchor)하세요.

## 개인 채팅 주제 (Private Chat Topics, Bot API 9.4)

텔레그램 Bot API 9.4(2026년 2월)는 **개인 채팅 주제(Private Chat Topics)**를 도입했습니다. 이제 수퍼그룹이 없어도 봇이 1대1 DM 채팅에서 직접 포럼 스타일 주제(topic) 스레드를 만들 수 있습니다. 이를 통해 Hermes로 기존 DM 내에서 격리된 워크스페이스 여러 개를 운영할 수 있습니다.

### 활용 사례

여러 장기 프로젝트에서 작업하는 경우, 주제별로 컨텍스트를 분리해 유지할 수 있습니다:

- **주제 "Website"** — 프로덕션 웹 서비스 작업
- **주제 "Research"** — 문헌 리뷰 및 논문 탐색
- **주제 "General"** — 기타 잡다한 작업 및 간단한 질문

각 주제는 다른 주제들과 완전히 격리된 자체 대화 세션, 대화 기록 및 컨텍스트를 갖습니다.

### 구성

:::caution 전제 조건
구성에 주제를 추가하기 전에, 사용자는 봇과의 DM 채팅에서 **주제 모드(Topics mode)**를 활성화해야 합니다:

1. 텔레그램에서 Hermes 봇과의 비공개 채팅을 엽니다.
2. 채팅 정보창을 열기 위해 상단의 봇 이름을 탭합니다.
3. **주제(Topics)**를 활성화합니다 (채팅을 포럼으로 바꾸는 토글).

이 과정이 없으면 Hermes는 시작 시 `The chat is not a forum`을 기록하고 주제 생성을 건너뜁니다. 이는 텔레그램 클라이언트 측 설정이므로 봇이 프로그래밍 방식으로 활성화할 수 없습니다.
:::

`~/.hermes/config.yaml`의 `platforms.telegram.extra.dm_topics` 아래에 주제를 추가합니다:

```yaml
platforms:
  telegram:
    extra:
      dm_topics:
      - chat_id: 123456789        # 여러분의 텔레그램 사용자 ID
        topics:
        - name: General
          icon_color: 7322096
        - name: Website
          icon_color: 9367192
        - name: Research
          icon_color: 16766590
          skill: arxiv              # 이 주제에서 스킬(skill) 자동 로드
```

**필드:**

| 필드 | 필수 | 설명 |
|-------|----------|-------------|
| `name` | 예 | 주제 표시 이름 |
| `icon_color` | 아니요 | 텔레그램 아이콘 색상 코드 (정수형) |
| `icon_custom_emoji_id` | 아니요 | 주제 아이콘을 위한 사용자 지정 이모지 ID |
| `skill` | 아니요 | 이 주제의 새 세션에서 자동 로드할 스킬 |
| `thread_id` | 아니요 | 주제 생성 후 자동으로 채워집니다 — 수동으로 설정하지 마세요. |

### 작동 방식

1. 게이트웨이 시작 시, Hermes는 `thread_id`가 아직 없는 각 주제에 대해 `createForumTopic`을 호출합니다.
2. `thread_id`가 `config.yaml`에 자동으로 다시 저장됩니다 — 이후 시작 시 API 호출을 건너뜁니다.
3. 각 주제는 격리된 세션 키에 매핑됩니다: `agent:main:telegram:dm:{chat_id}:{thread_id}`
4. 각 주제 내 메시지들은 자체적인 대화 기록, 메모리 플러시, 컨텍스트 창을 가집니다.

### 루트 DM 처리

기본적으로 루트 DM(모든 주제 외부)으로 전송된 메시지는 정상적으로 처리됩니다. `ignore_root_dm: true`를 설정하면 루트 DM을 로비(lobby)로 전환할 수 있습니다. — DM 주제가 구성된 사용자의 경우 루트 DM에서의 일반 메시지는 조용히 무시되지만, 시스템 명령(`/start`, `/help`, `/status` 등)은 계속 작동합니다.

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

이 확인 작업은 **채팅별**로 이뤄집니다: `dm_topics` 항목이 하나 이상 있는 사용자만 루트 DM이 영향을 받습니다. 주제가 구성되지 않은 사용자는 영향을 받지 않습니다.

### 스킬 바인딩 (Skill binding)

`skill` 필드가 있는 주제는 해당 주제에서 새 세션이 시작될 때 스킬을 자동으로 로드합니다. 이는 대화 시작 시 `/skill-name`을 입력하는 것과 동일하게 작동합니다. 스킬 콘텐츠가 첫 메시지에 주입되고, 이후 메시지는 대화 기록에서 이 스킬 콘텐츠를 참조합니다.

예를 들어, `skill: arxiv`를 설정한 주제는 세션이 초기화될 때마다(유휴 시간 초과, 일일 리셋, 또는 수동 `/reset`) arxiv 스킬이 사전 로드됩니다.

:::tip
구성 외부에서 (예: 텔레그램 API를 직접 호출하여 수동으로) 생성된 주제는 `forum_topic_created` 서비스 메시지가 도착할 때 자동으로 검색됩니다. 또한 게이트웨이가 실행 중일 때 구성에 주제를 추가할 수도 있습니다. 다음번 캐시 미스 시 등록됩니다.
:::

## 멀티-세션 DM 모드 (`/topic`)

ChatGPT 스타일의 다중 세션 DM입니다 — 하나의 봇에서 병렬로 진행되는 여러 개의 대화입니다. 위에서 연산자 선별적으로 설정하는 `extra.dm_topics`와는 다르게, 이 모드는 **사용자 중심(user-driven)**입니다. 구성이나 사전 정의된 주제 이름이 필요 없습니다. 최종 사용자가 `/topic`으로 이 기능을 켜고 텔레그램의 **+** 버튼을 눌러 원하는 만큼의 주제(topics)를 생성하면, 각각의 주제가 완전히 독립된 Hermes 세션으로 작동하게 됩니다.

### `/topic` 하위 명령어들 (subcommands)

| 형식 | 컨텍스트 | 효과 |
|------|---------|--------|
| `/topic` | 루트 DM, 활성화되지 않음 | BotFather 기능을 확인하고 멀티-세션 모드를 활성화한 뒤 고정(pinned)된 시스템 주제를 생성합니다. |
| `/topic` | 루트 DM, 이미 활성화됨 | 상태 표시: 복원할 수 있는 연결되지 않은 세션 목록을 보여줍니다. |
| `/topic` | 주제 내부 | 현재 주제의 세션 바인딩을 표시합니다. |
| `/topic help` | 모두 | 사용법 안내(Inline usage)를 보여줍니다. |
| `/topic off` | 루트 DM | 멀티-세션 모드를 비활성화하고, 이 채팅의 모든 주제 바인딩을 지웁니다. |
| `/topic <session-id>` | 주제 내부 | 텔레그램 세션을 현재 주제로 복원합니다. |

인증된 사용자만 (`TELEGRAM_ALLOWED_USERS` / 플랫폼 인증 설정) `/topic`을 실행할 수 있습니다. 권한이 없는 발신자에게는 활성화 대신 거부 메시지가 표시됩니다.

### DM 주제 vs 멀티-세션 DM 모드

| | `extra.dm_topics` (구성 주도형) | `/topic` (사용자 주도형) |
|---|---|---|
| 활성화 주체 | 시스템 운영자(config.yaml 설정) | 최종 사용자 (`/topic` 전송을 통해) |
| 주제 목록 | 구성에 명시된 고정 집합 | 사용자가 자유롭게 주제 생성/삭제 |
| 주제 이름 | 시스템 운영자가 선택 | 사용자가 선택 (이후 Hermes 세션 제목과 일치하도록 자동 이름 변경) |
| 루트 DM 동작 | 정상적인 채팅 (`ignore_root_dm: true`인 경우엔 로비) | 시스템 로비로 변환 (명령어가 아닌 메시지는 거부됨) |
| 주요 활용 사례 | 선택적 스킬 바인딩이 포함된 영구적 작업 공간 | 즉석에서 만드는(ad-hoc) 병렬 세션 |
| 지속성 보장 | config 내 `extra.dm_topics` 값 | `telegram_dm_topic_mode` + `telegram_dm_topic_bindings` SQLite 테이블 |

이 두 가지 기능은 동일한 봇에서 공존할 수 있습니다. 사용자의 DM에서 `/topic`을 실행하고, `extra.dm_topics`를 사용해 운영자가 구성한 주제들을 계속 관리할 수 있습니다.

### 전제 조건 (Prerequisites)

**@BotFather**에서 자신의 봇을 선택 후 → **Bot Settings → Threads Settings**:

1. **Threaded Mode**를 켭니다 (`has_topics_enabled` 활성화됨)
2. 사용자의 주제 생성을 비활성화하지 **마세요** (`allows_users_to_create_topics` 켜짐 상태 유지)

사용자가 처음 `/topic`을 실행할 때, Hermes는 `getMe`를 호출해 이 두 가지 플래그를 모두 확인합니다. 둘 중 하나라도 꺼져있다면, Hermes는 BotFather Threads Settings 화면의 스크린샷과 함께 무엇을 토글해야 하는지 안내하는 메시지를 보내고, 조건을 충족할 때까지 활성화를 보류합니다.

### 활성화 흐름 (Activation flow)

루트 DM에서 다음과 같이 보냅니다:

```
/topic
```

Hermes는 다음과 같이 작동합니다:

1. `getMe().has_topics_enabled`와 `allows_users_to_create_topics` 여부를 확인합니다.
2. 둘 다 참(true)인 경우, 이 DM에 대해 멀티-세션 주제 모드를 활성화합니다.
3. 상태/명령어를 위한 **System** 주제를 생성하여 고정(pinned) 시킵니다 (최선 노력 방식).
4. 사용자가 복원할 수 있는 이전에 연결되지 않은 텔레그램 세션 목록을 회신합니다.

활성화 이후에는 **루트 DM이 로비(lobby)가 됩니다**. 여기서 사용자가 보내는 일반 프롬프트들은 거부되며, **All Messages** 공간으로 이동하라는 안내를 받게 됩니다. 시스템 명령어(`/status`, `/sessions`, `/usage`, `/help` 등)들은 루트 공간에서 계속 작동합니다.

### 새 주제 만들기 (최종 사용자 작업 흐름)

1. 텔레그램에서 봇 DM을 엽니다.
2. 봇 인터페이스 상단에서 **All Messages**를 탭한 다음 메시지를 보냅니다.
3. 텔레그램이 그 메시지를 위한 새 주제를 만듭니다.
4. Hermes는 해당 주제 안에서 응답하며, 이제 그 주제는 독립적인 세션이 됩니다.

모든 주제는 각자의 대화 이력(history), 모델 상태, 도구 실행, 그리고 세션 ID를 가집니다. 격리 키는 `agent:main:telegram:dm:{chat_id}:{thread_id}` 이며 구성형(config-driven) DM 주제 격리 방식과 동일합니다.

### 주제 이름 자동 변경 (Auto-renamed topics)

Hermes가 해당 주제의 세션 제목(session title)을 생성하면 (첫 번째 대화 교환 후에 자동 제목 지정 파이프라인을 통해 이루어짐), 텔레그램 주제 이름 역시 이에 맞춰 함께 변경됩니다. 예를 들어 "New Topic"이 "Database migration plan"으로 바뀝니다. 이름 바꾸기는 최선 노력(best-effort) 방식으로 진행되며, 만약 실패하더라도 기록에 남을 뿐 세션이 중단되지는 않습니다.

이 기능을 끄고 직접 지정한 주제 이름을 유지하려면 다음과 같이 설정하세요:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_topic_auto_rename: true
```

이 플래그가 켜진 상태라면, Hermes는 내부적으로 세션 제목(`hermes sessions`나 TUI 등에서 사용)을 계속 생성하긴 하지만, 텔레그램 쪽 주제 이름은 절대 변경하지 않습니다. 이는 BotFather의 쓰레드 모드에서 수동으로 주제들을 정리하고 싶고, 첫 번째 답변이 달릴 때마다 제목이 덮어써지는 것을 방지하고 싶을 때 유용합니다.

### 주제 안에서의 `/new`

다른 주제 건드림 없이, 현재 주제의 세션(새로운 세션 ID 부여 및 기록 초기화)만 초기화(reset)합니다. 병렬 작업을 위해서는 (명령어를 치기보단) **All Messages**를 통해 새 주제를 생성하는 편이 주로 원하는 목적에 부합할 것이라는 점을 상기시키는 메시지를 Hermes가 회신합니다.

### 이전 세션 복원하기 (Restoring a previous session)

주제 내부에서, 다음과 같이 보냅니다:

```
/topic <session-id>
```

이 명령어는 현재 주제에서 완전히 새롭게 시작하는 대신, 기존의 Hermes 세션을 현재 주제에 연결(binding)합니다. 주제 모드를 활성화하기 전에 시작했던 대화를 계속 이어가고자 할 때 유용합니다. 주의사항:

- 대상 세션은 반드시 동일한 텔레그램 사용자의 소유여야 합니다.
- 대상 세션이 이미 다른 주제에 연결되어 있으면 안 됩니다.

Hermes는 세션 제목을 확인시켜 주며, 맥락을 이을 수 있도록 해당 세션의 가장 마지막 AI 메시지 내용을 회신합니다.

세션 ID를 찾으려면 루트 DM 공간에서 매개변수 없이 `/topic`을 보내세요. Hermes가 사용자의 연결되지 않은 텔레그램 세션 목록을 나열해 줄 것입니다.

### 주제 안에서의 `/topic` (매개변수 없음)

현재 주제가 어떻게 연결되어 있는지 표시합니다: 세션 제목, 세션 ID 정보와 함께 `/new`를 사용하는 것과 아예 다른 주제를 새로 생성하는 것에 대한 차이점을 힌트로 제공합니다.

### 내부 작동 구조 (Under the hood)

- 활성화 상태는 `state.db`의 `telegram_dm_topic_mode(chat_id, user_id, enabled, ...)` 테이블에 지속 기록됩니다.
- 각 주제와의 연결 상태는 `telegram_dm_topic_bindings(chat_id, thread_id, session_id, ...)` 테이블에 기록되며, `session_id` 컬럼에 대해 `ON DELETE CASCADE` 속성이 적용됩니다. — 즉, 세션을 삭제하면 해당 세션의 주제 연결 내용도 자동으로 지워집니다.
- 주제 모드(topic-mode) SQLite 마이그레이션은 **선택적(opt-in)**입니다. 게이트웨이가 시작할 때가 아니라, 사용자가 처음 `/topic`을 호출할 때 실행됩니다. 이 프로필 내에서 누군가 `/topic`을 실행하기 전까지는 `state.db` 내용이 전혀 변경되지 않습니다.
- 텔레그램 DM으로 들어오는 각각의 메시지는 먼저 자신의 `(chat_id, thread_id)` 연결 고리를 조회합니다. 만약 연결 고리가 존재하면, 조회 시스템이 `SessionStore.switch_session()`을 통해 이 메시지를 연결된 해당 세션으로 보내주기 때문에, 디스크(DB)의 세션키와 세션ID 사이의 매핑이 항상 일관성을 유지할 수 있습니다.
- 주제 안에서 `/new` 명령어를 쓰면 이 연결 정보(binding row)를 새로 생성된 세션 ID로 업데이트하여, 다음 메시지부터는 새로운 세션 상에서 대화가 이어질 수 있게 합니다.
- `extra.dm_topics`에 정의된 주제들은 **절대 자동 이름 변경(auto-rename)의 대상이 되지 않습니다** — 멀티-세션 모드가 켜져 있더라도 운영자가 직접 정해둔 이름이 보존됩니다.
- 채팅 내 **모든** 주제(쓰레드 모드를 통한 즉석에서 만들어진(ad-hoc) 주제 포함)의 자동 이름 변경 기능을 끄려면 `extra.disable_topic_auto_rename: true` 로 설정하세요.
- 포럼 활성화 DM의 일반(General, 상단 고정) 주제는, 텔레그램이 `message_thread_id=1`로 전달하든 `thread_id` 정보 없이 전달하든 상관없이 루트 로비(root lobby) 취급을 받습니다.
- 루트 로비 안내 알림은 채팅별로 30초에 1회만 전송되도록 속도 제한(rate-limited)이 걸려있습니다 — 사용자가 주제 모드가 켜져 있는 걸 잊고 루트 채팅에서 프롬프트를 열 번 쳤다고 해서 열 번의 회신을 받는 일은 발생하지 않습니다.
- BotFather 설정 스크린샷 안내는 채팅별 5분당 1회만 전송되도록 속도 제한이 걸려있습니다. 따라서 스레드 설정이 비활성화된 상태에서 반복적으로 `/topic` 시도를 하더라도 동일한 이미지를 연속해서 재업로드하지 않습니다.
- 주제 내에서 시작된 `/background <prompt>` 작업의 결과는 다시 해당 주제로 전달됩니다; 백그라운드 세션은 소유하고 있는 주제의 자동 이름 바꾸기 트리거를 작동시키지 않습니다.
- `/topic` 명령어 자체는 봇의 사용자 인가 검사를 거칩니다 — 권한이 없는 DM에서는 활성화 대신 거부 메시지를 받게 됩니다.

### 멀티-세션 모드 비활성화하기

루트 DM 공간에서 `/topic off`를 전송합니다. 그러면 Hermes는 이 모드 플래그를 비활성화하고 이 채팅에 걸려있던 `(thread_id → session_id)` 연결들을 모두 정리하여, 이 루트 DM 공간은 일반적인 기본 Hermes 채팅으로 되돌아옵니다. 텔레그램 상에 이미 만들어진 주제들이 삭제되는 것은 아닙니다 — 그저 각각의 독립된 세션으로서의 통제 및 관리 기능이 사라질 뿐입니다. 나중에 `/topic`을 다시 실행하여 언제든지 다시 켤 수 있습니다.

만약 수동으로 정리 작업이 필요하다면(예: 수많은 채팅에서의 일괄 재설정 작업 등), 직접 행(row)을 삭제하세요:

```bash
sqlite3 ~/.hermes/state.db \
  "UPDATE telegram_dm_topic_mode SET enabled = 0 WHERE chat_id = '<your_chat_id>'; \
   DELETE FROM telegram_dm_topic_bindings WHERE chat_id = '<your_chat_id>';"
```

### Hermes 다운그레이드 시

`/topic` 기능 도입 이전의 옛날 Hermes 버전으로 다운그레이드할 경우, 이 기능은 단순히 작동을 멈추게 됩니다. `state.db` 안에 들어있는 `telegram_dm_topic_mode` 와 `telegram_dm_topic_bindings` 테이블들은 구버전 코드가 무시할 뿐 삭제되진 않습니다. 이 경우 DM 동작은 기본적인 스레드당 격리 체계(각 `message_thread_id`가 `build_session_key`를 거쳐 독자적인 세션을 가짐)로 되돌아가기 때문에, 이미 있는 텔레그램 주제들은 여전히 병렬 대화 세션의 형태로 작동을 계속합니다. 루트 DM 역시 더 이상 로비 역할을 하지 않고 예전처럼 일반 메시지를 받아들이게 됩니다. 추후 다시 최신 버전으로 업그레이드하면 멀티-세션 모드가 원래 상태 그대로 다시 활성화됩니다.

## 그룹 포럼 주제 내 스킬 바인딩 (Group Forum Topic Skill Binding)

**주제 모드(Topics mode)**가 켜져 있는 수퍼그룹(이를 "포럼 주제"라고도 부릅니다)은 이미 주제별로 세션이 완전히 분리(격리)되어 있습니다 — 각 `thread_id`가 자신만의 개별 대화 세션과 매핑됩니다. 하지만 DM 주제에서의 스킬 바인딩처럼, 특정 그룹 주제(topic) 안에 메시지가 도착할 때마다 **특정 스킬(skill)을 자동으로 불러오고(auto-load)** 싶을 수 있습니다.

### 활용 사례

여러 업무 흐름(workstreams)을 위해 만들어진 팀의 수퍼그룹 포럼 주제 예시:

- **Engineering** 주제 → `software-development` 스킬 자동 로드
- **Research** 주제 → `arxiv` 스킬 자동 로드
- **General** 주제 → 어떤 스킬도 로드하지 않음 (범용 어시스턴트로 작동)

### 구성

`~/.hermes/config.yaml` 파일 내의 `platforms.telegram.extra.group_topics` 에 주제 바인딩 설정을 추가합니다:

```yaml
platforms:
  telegram:
    extra:
      group_topics:
      - chat_id: -1001234567890       # 수퍼그룹 ID
        topics:
        - name: Engineering
          thread_id: 5
          skill: software-development
        - name: Research
          thread_id: 12
          skill: arxiv
        - name: General
          thread_id: 1
          # 스킬 없음 — 범용 어시스턴트로 작동
```

**필드:**

| 필드 | 필수 | 설명 |
|-------|----------|-------------|
| `chat_id` | 예 | 수퍼그룹의 숫자형 ID (`-100`으로 시작하는 음수 값) |
| `name` | 아니요 | 주제를 구별하기 위해 사람이 알아볼 수 있게 적은 라벨 정보 (참고용으로만 쓰임) |
| `thread_id` | 예 | 텔레그램 포럼 주제 ID — `t.me/c/<group_id>/<thread_id>` 형식의 링크에서 확인할 수 있습니다. |
| `skill` | 아니요 | 이 주제 안에서 새로운 세션이 열릴 때 자동으로 불러올 스킬 |

### 작동 방식

1. 매핑된 그룹 주제로 메시지가 도착하면, Hermes는 환경 설정 파일의 `group_topics` 에서 `chat_id`와 `thread_id`가 일치하는 항목이 있는지 확인합니다.
2. 매핑 항목에 `skill` 필드가 지정되어 있다면, DM 주제 스킬 바인딩의 작동 방식과 똑같이 해당 세션에서 사용할 스킬을 자동 로드합니다.
3. `skill` 키가 없는 주제들은 원래 하던 대로 세션 분리(격리)만 작동합니다 (기존 동작 방식과 차이 없음).
4. 매핑되지 않은 `thread_id`나 `chat_id` 값을 가진 메시지는 그냥 별문제 없이 지나갑니다(fall through) — 오류도 없고 스킬 로딩도 없습니다.

### DM 주제와의 차이점

| | DM 주제 | 그룹 주제 |
|---|---|---|
| 구성 키워드(Config key) | `extra.dm_topics` | `extra.group_topics` |
| 주제 생성 | `thread_id`가 누락된 경우 Hermes가 API를 통해 주제 생성 | 시스템 관리자가 텔레그램 UI를 통해 직접 생성 |
| `thread_id` | 생성 후에 자동으로 채워짐 | 사용자가 수동으로 설정해야 함 |
| `icon_color` / `icon_custom_emoji_id` | 지원함 | 적용 불가 (시스템 관리자가 생김새를 통제) |
| 스킬 바인딩(Skill binding) | ✓ | ✓ |
| 세션 격리(Session isolation) | ✓ | ✓ (포럼 주제 특성상 이미 기본으로 내장된 동작) |

:::tip
어떤 주제의 `thread_id`를 찾고 싶다면, 텔레그램 웹이나 데스크톱 버전에서 해당 주제를 연 다음 URL을 확인하세요: `https://t.me/c/1234567890/5` — 이 주소의 맨 마지막 숫자(`5`)가 바로 `thread_id`입니다. 수퍼그룹용 `chat_id`는 그룹 ID 값의 앞에 `-100`을 붙인 형태입니다 (예: `1234567890` 그룹은 `-1001234567890`이 됩니다).
:::

## 최신 Bot API 기능 (Recent Bot API Features)

- **Bot API 9.4 (2026년 2월):** 개인 채팅 주제(Private Chat Topics) — 봇이 `createForumTopic` 명령어를 통해 1:1 채팅 환경인 DM 안에서도 포럼 방식의 주제방들을 만들 수 있습니다. Hermes는 두 가지 서로 다른 기능을 통해 이를 활용합니다. 첫째는 시스템 운영자가 `config.yaml`로 사전에 정해두는 [개인 채팅 주제(Private Chat Topics)](#개인-채팅-주제-private-chat-topics-bot-api-94) 방식이고, 둘째는 사용자가 `/topic` 명령어 하나로 주제를 무제한 개설할 수 있는 [멀티-세션 DM 모드(Multi-session DM mode)](#멀티-세션-dm-모드-topic) 방식입니다.
- **개인정보 처리방침 (Privacy policy):** 현재 텔레그램은 봇들에게 개인정보 처리방침을 요구합니다. BotFather 화면에서 `/setprivacy_policy` 명령어를 통해 직접 설정할 수 있습니다. 안 할 경우 텔레그램 쪽에서 자동으로 임시 방침 문구를 만들어 채워 넣을 수 있습니다. 봇을 대중에게 공개하고 운영할 계획이라면 이 조치는 특히 더 중요합니다.
- **Bot API 9.5 (2026년 3월): `sendMessageDraft`를 이용한 네이티브 스트리밍 기능.** Hermes는 개인 채팅 환경에서 네이티브 스트리밍 기능(draft-preview 방식)을 선택 옵션(opt-in transport)으로 지원합니다. 다만, 일부 텔레그램 클라이언트 앱에서는 임시 스트리밍 미리보기(draft previews) 창이 계속 열렸다 닫혔다 하며 화면을 다시 그리는 증상(collapse and re-render)이 나타나므로, 여전히 예전부터 쓰던 `editMessageText` 방식을 기본 경로로 사용합니다.

### 스트리밍 전송 (`gateway.streaming.transport`)

스트리밍이 활성화되어 있을 때 (`gateway.streaming.enabled: true`), Hermes는 다음 4개의 전송 방식 중 하나를 선택합니다:

| 설정값 | 작동 방식 |
|---|---|
| `auto` | 호환되는 채팅 환경(현재는 텔레그램 DM 채팅만 해당)에서는 네이티브 임시 스트리밍(draft) 방식을 씁니다. 호환 안 되는 곳에서는 `edit` 기반 구형 방식을 씁니다. 중간에 스트리밍 프레임에 문제가 생기면 우아하게 오류를 우회하여 전송을 이어나갑니다. |
| `draft` | 무조건 네이티브 draft 방식을 강제합니다. 그룹 채팅/주제방처럼 해당 기능이 지원 안 되는 곳이라면, 텔레그램이 경고 메시지를 기록하고 내부적으로 `edit` 방식으로 전환시킵니다. |
| `edit` (기본값) | 모든 유형의 채팅방에서 구형 방식(`editMessageText`를 이용해 문장들을 주기적으로 계속 수정해 나가는 방식)을 씁니다. |
| `off` | 스트리밍을 전면 중지합니다. (중간 과정 없이 마지막에 단 한 번의 완벽한 답변만 회신합니다). |

`~/.hermes/config.yaml` 환경 설정에서:

```yaml
gateway:
  streaming:
    enabled: true
    transport: edit    # edit | auto | draft | off
```

**DM 환경에서 `edit` (기본값) 방식일 때 화면에 보이는 현상** — 게이트웨이가 정상적인 메시지 창을 하나 띄운 후 `editMessageText`를 이용해 이 메시지 내용을 지속적으로 교체(업데이트)합니다. 이 방식은 텔레그램의 draft-preview 창이 열렸다 닫히며 화면이 흔들리는 현상을 피할 수 있습니다.

**DM 환경에서 `auto`나 `draft` 방식일 때 화면에 보이는 현상** — 텔레그램에 애니메이션 형태의 임시 미리보기(draft preview) 창이 띄워지고, 문장들이 단어 단위(token-by-token)로 갱신됩니다. 에이전트의 응답이 끝나면 완전히 하나의 정식 메시지로 바뀌어 도착하며, 사용자 클라이언트 단의 draft preview 창도 자연스레 사라집니다. 이런 임시 파일들은 메시지 ID도 부여받지 못하기 때문에, 당신의 채팅 기록에 최종적으로 남는 것은 마지막으로 완성된 답변 메시지 하나뿐입니다.

**그룹, 수퍼그룹, 포럼 주제방에서는 어떨까요?** 텔레그램은 `sendMessageDraft` 명령어 사용을 비공개 1:1 채팅(DM) 환경으로 엄격하게 제한합니다. 따라서 이들 채팅방에서는 게이트웨이가 사용자의 별도 조작 없이 자동으로 내부 구형 방식(`edit` 방식)으로 회신을 이어나갑니다. 결국 겉으로 보이는 현상은 예전과 동일합니다.

**중간에 draft 프레임에 오류가 생기면?** 만약 네트워크에 일시적인 끊김 현상이 발생하거나 서버 쪽 오류 또는 사용자의 `python-telegram-bot` 버전이 오래되는 등 어떠한 이유로든 오류가 발생한다면, 시스템이 자동으로 남은 나머지 메시지들은 `edit` 방식 경로로 되돌려(flip back) 마저 이어나갑니다. 그리고 다음 응답 때는 처음부터 다시 draft 방식을 시도합니다.

## 렌더링: 테이블 및 링크 미리보기 (Rendering: Tables and Link Previews)

텔레그램의 MarkdownV2 형식은 테이블(표) 표현을 지원하는 고유 문법(native syntax)을 가지고 있지 않습니다. 따라서 일반적인 `|` (파이프) 형태의 마크다운 테이블 구문을 원형 그대로 텔레그램에 전송하면 알아볼 수 없게 렌더링됩니다. Hermes는 이러한 마크다운 테이블을 다음과 같이 자동으로 보기 좋게 정규화(normalize)합니다:

- **작은 테이블들**은 **그룹화된 글머리 기호(row-group bullets)** 형태로 평면화됩니다 — 표의 각 행들이 열 제목 밑에 알아보기 쉬운 글머리 기호(bulleted list)의 형태로 나타납니다. 내용이 짧고 열이 2~4개 정도 되는 표를 표현할 때 아주 좋습니다.
- **크거나 옆으로 넓은 테이블들**은 열의 형태가 무너지지 않도록 세로줄이 잘 정렬된 **fenced code block** (코드 형태) 방식으로 우회하여 표현됩니다. 모델(에이전트)이 텔레그램에서 사용자에게 표(table) 방식보다 서술적인 줄글 형태의 응답을 주도록 유도하는 한 줄짜리 프롬프트 힌트 정보도 함께 추가됩니다.

따로 구성할 항목은 없습니다. — 어댑터가 알아서 매 메시지마다 적절한 폴백(fallback) 방식을 선택합니다. 하지만 예전처럼 "무조건 코드 블록 방식"으로 처리하고 싶다면, `config.yaml`에서 `telegram.pretty_tables: false` (기본값: `true`)로 설정하여 테이블 자동 정규화 기능을 끄세요.

**링크 미리보기(Link previews).** 텔레그램은 봇 메시지 내의 URL 주소들에 대해 자동으로 링크 미리보기 창을 만들어 보여줍니다. 만약 이것이 불필요해서 미리보기 창이 뜨는 것을 막고 싶다면 (예: `/tools` 명령에 대한 결과물이 너무 길거나, 에이전트 답변 안에 주소가 열 개씩 포함된 경우 등):

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_link_previews: true
```

이 옵션이 켜지면, Hermes는 외부로 보내는 모든 메시지에 텔레그램 옵션인 `LinkPreviewOptions(is_disabled=True)` 속성을 부여합니다. 오래된 버전의 `python-telegram-bot` 라이브러리를 쓰는 경우를 대비해 기존에 쓰이던 매개변수인 `disable_web_page_preview` 항목으로도 처리될 수 있도록 조치합니다.

## 그룹 허용 목록 (Group Allowlisting)

텔레그램 그룹과 포럼 채팅에는 독립적으로 구성할 수 있는 두 개의 통제 수단이 존재합니다:

- **발신 사용자 ID (Sender user IDs)** (`group_allow_from` / `TELEGRAM_GROUP_ALLOWED_USERS`) — 그룹/포럼의 메시지들에 대해서만 적용되는 발신자 단위(sender-scoped) 허용 목록. 특정 사용자들에게 그룹 내 봇 이용 권한을 주면서도, 그들을 `TELEGRAM_ALLOWED_USERS` 목록에는 넣지 않아서 DM 채팅 권한까지는 주지 않고 싶을 때 사용합니다.
- **채팅 ID (Chat IDs)** (`group_allowed_chats` / `TELEGRAM_GROUP_ALLOWED_CHATS`) — 채팅방 단위(chat-scoped) 허용 목록. 이 목록에 속한 그룹/포럼 방의 구성원은 누구든 봇과 상호작용할 수 있습니다. 특정 팀 전용 봇, 고객 지원방 봇처럼 그룹방 입장 그 자체로 봇 이용 권한이 부여되는 환경에 매우 유용합니다.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # 전역 접속 권한 (DM + 그룹). 여기에 적힌 사용자들은 언제 어디서든 봇을 호출할 수 있습니다.
        allow_from:
          - "123456789"
        # 그룹/포럼 방 안에서만 발신 권한을 줌. 이 사람들에게 DM 채팅 권한까지 부여하는 것은 아닙니다.
        group_allow_from:
          - "987654321"
        # 그룹/포럼 방 전체에 권한을 줌 — 발신자가 누구든 방 안에 있는 모든 멤버가 봇을 쓸 수 있습니다.
        group_allowed_chats:
          - "-1001234567890"
```

동일한 역할을 하는 환경변수:

```bash
TELEGRAM_ALLOWED_USERS="123456789"
TELEGRAM_GROUP_ALLOWED_USERS="987654321"
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

동작 방식:

- `TELEGRAM_ALLOWED_USERS` 목록은 모든 채팅 형태(DM, 그룹, 포럼)를 전부 통제합니다.
- `TELEGRAM_GROUP_ALLOWED_USERS` 목록은 여기서 허용된 발신자가 그룹/포럼 방 안에서 봇과 소통하는 것만을 허용합니다. 그들이 `TELEGRAM_ALLOWED_USERS` 에도 등록되어 있지 않는 한, 봇에게 직접 개인 DM을 보낼 수는 없습니다.
- `TELEGRAM_GROUP_ALLOWED_CHATS` 목록에 적힌 그룹 방에서 이뤄지는 대화는, 전송자가 누구든 해당 채팅방의 구성원이기만 하면 모두 허용 처리됩니다.
- 이 목록들 어디에든 `*` 기호를 써넣으면, 모든 발신자나 모든 채팅방이 예외 없이 허용되는 효과를 갖습니다.
- 이러한 통제 장치는 기존의 멘션/패턴 인식 방식(trigger)과, `group_topics` 및 `ignored_threads` 등과 같은 옵션과 겹겹이 함께 쌓여 작동합니다.

### PR #17686 이전 버전에서의 업그레이드 마이그레이션

이 두 개의 분리(split) 기능이 나오기 전에는 오직 `TELEGRAM_GROUP_ALLOWED_USERS` 조절 옵션 하나뿐이어서, 사람들은 거기에 **채팅 ID(chat IDs)** 값들을 넣곤 했습니다. 호환성을 유지하기 위하여, 예전처럼 `-` 부호로 시작하는 채팅 ID 형식의 값을 `TELEGRAM_GROUP_ALLOWED_USERS` 에 넣을 경우, 시스템은 इसे 여전히 정상적인 채팅 ID 값으로 취급하고 허용 처리합니다. (단, 오래된 방식을 쓴다는 경고 로그가 한 번 표출됩니다). 마이그레이션:

```bash
# 구 방식 (지금도 정상 작동하긴 하지만, 향후 삭제될(deprecated) 예정)
TELEGRAM_GROUP_ALLOWED_USERS="-1001234567890"

# 새 방식
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

### 멘션(ping)을 해야만 응답하는 게스트 우회 모드 (`guest_mode`)

전형적인 기본 설정 환경에서, `group_allowed_chats`는 아주 강력한 차단기(hard gate)로 작동합니다. 즉, 허용 목록에 등록되지 않은 외부 그룹방에서 어떤 멤버가 봇을 @태그하여 명시적으로 불렀다(mention) 해도 시스템은 그냥 말없이 그 메시지를 삭제하고 아무 응답도 하지 않습니다. 이런 기본 설정은 특정 팀 전용 봇, 고객 지원용 봇 환경에 아주 알맞습니다.

친구들끼리의 대화방처럼 **기본적으로 봇이 침묵**을 지키게 하면서, 특별히 **명시적으로 태그해서 불렀을(ping) 때만** 반응하길 바라는 가벼운 환경(casual setups)의 경우 `guest_mode` 옵션을 활성화하세요:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        group_allowed_chats:
          - "-1001234567890"   # 내 메인 허용 목록 그룹방들
        guest_mode: true       # 비 허용 목록 그룹방들: 오직 봇을 @태그(mention)했을 때만 반응하게 함
```

환경변수 등가(Equivalent env vars):

```bash
TELEGRAM_GUEST_MODE=true
```

기본값: `false`.

`guest_mode: true` 설정이 켜져 있으면, 허용 목록에 없는 그룹 방에서 메시지가 올 때, 해당 메시지가 명시적으로 봇을 @태그(mention)한 경우에**만** 시스템이 반응합니다. 이러한 게스트 상호작용의 경우 이전 대화 내용을 이어가는 세션 유지(session stickiness) 기능이 꺼져 있기 때문에 사용자는 매번 말을 걸 때마다 봇을 태그해야 하며, 봇 스스로가 자신이 태그되지 않은 방에서 먼저 대화를 거는 일은 절대 생기지 않습니다.

물론 DM이나 허용 목록에 명시된 그룹방 안에서의 대화는 언제나 기존과 똑같이 정상 처리됩니다.

## 슬래시 명령어 권한 제어 (Slash Command Access Control)

기본적으로 봇의 허용 목록에 있는 사용자는 봇의 모든 슬래시 명령어를 자유롭게 사용할 수 있습니다. 이 허용 목록의 멤버를 **어드민(admins)** (모든 슬래시 명령에 대한 전체 권한 부여)과 **일반 사용자(regular users)** (내가 지정한 슬래시 명령만 사용 가능) 등급으로 나눠서 통제하고 싶다면, 플랫폼의 `extra` 설정 항목에 `allow_admin_from` 과 `user_allowed_commands`를 추가해 주세요:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # 기존 허용 목록 (변동 없음)
        allow_from:
          - "123456789"     # 관리자 등급 부여 대상
          - "555555555"     # 일반 사용자 등급
          - "777777777"     # 일반 사용자 등급

        # NEW — 어드민 계정들에게는 등록된 모든 슬래시 명령 (내장 기본 명령들 + 기타 플러그인 등)의 권한 부여
        allow_admin_from:
          - "123456789"

        # NEW — 어드민 등급이 아닌 그냥 허용 목록 일반 사용자들이 쓸 수 있는 명령어들 제한하기.
        # /help 와 /whoami 는 사용자가 자기 권한 상태를 확인할 수 있도록 언제나 묻지도 따지지도 않고 열려 있습니다.
        user_allowed_commands:
          - status
          - model
          - history

        # 선택 사항: 그룹 방 환경에서의 어드민 권한/명령어 목록을 별도로 독립적으로 설정
        group_allow_admin_from:
          - "123456789"
        group_user_allowed_commands:
          - status
```

**동작 방식:**

- `allow_admin_from` 에 등록되어 특정 환경(DM인지 그룹인지)의 어드민 등급을 받은 사용자는, 실시간 명령어 관리 목록에 등록된 **모든** 슬래시 명령어 (내장(built-in) 명령어 + 플러그인 등록 명령어)를 문제없이 쓸 수 있습니다.
- `allow_from` 허용 목록에는 있지만 `allow_admin_from`에 포함되지 않아서 **일반 사용자**로 남게 된 사람들은, 오직 `user_allowed_commands` 설정에 쓰여진 명령어들만 쓸 수 있습니다. 단지 언제나 사용 권한이 열려있는 하한선(floor) 명령어인 `/help` 와 `/whoami`는 누구나 쓸 수 있습니다.
- 일반적인 슬래시 명령이 없는 일상적인 대화(Plain chat) 기능에는 권한 제한 효과가 없습니다. 어드민이 아닌 일반 사용자들도 항상 예전처럼 에이전트와 대화를 나눌 수 있지만, 임의의 제약된 특수 명령어 트리거 기능(arbitrary commands)만 못 쓸 뿐입니다.
- **이전 버전과 하위 호환(Backward compat):** 만약 `allow_admin_from` 항목을 아예 빈칸으로 놔두면, 그 환경에 대한 슬래시 명령어 통제 기능 자체가 꺼진 걸로 간주됩니다. 기존 사용자들이 아무것도 바꾸지 않고 쓰던 설정 그대로 써도 아무 문제가 발생하지 않습니다.
- DM 채팅에서의 어드민 자격이 그룹 채팅방 안의 어드민 자격까지 보장해주지 않습니다. 이 두 환경 모두 독립적으로 자신만의 어드민 목록 관리 체계를 씁니다.
- `group_allow_admin_from` 설정 항목만 작성되어 있는 상황이라면, 그룹 방이 아닌 DM 채팅 환경은 여전히 예전 시스템처럼 하위 호환 모드(unrestricted mode)로 작동하게 됩니다.

`/whoami` 명령을 통해 내가 지금 활성화하고 있는 채팅 환경(DM인지 그룹인지), 내 등급(어드민 / 일반 유저 / 권한 제한 없는 상태), 그리고 어떤 슬래시 명령어들을 쓸 권한이 있는지 확인할 수 있습니다.

## 대화형 모델 선택기 (Interactive Model Picker)

매개변수 없이 텔레그램 채팅 창에 `/model` 이라고만 입력하면, Hermes가 모델 설정을 바꿀 수 있도록 대화형 인라인(inline) 키보드 버튼 메뉴를 보여줍니다:

1. **서비스 제공자 선택 화면(Provider selection)** — 현재 지원 중인 공급자 이름과 그 안의 모델 개수 현황을 보여주는 버튼들 (예: "OpenAI (15)", "✓ Anthropic (12)" 등).
2. **모델 목록 화면(Model selection)** — 버튼을 통해 페이지를 넘겨보는 방식으로 전체 모델 리스트들을 보여주며 **이전(Prev)**/**다음(Next)** 네비게이션 기능, 전 단계로 돌아가는 **뒤로 가기(Back)** 버튼, 그리고 **취소(Cancel)** 버튼을 띄워줍니다.

현재 모델과 제공자 이름이 맨 위에 표시됩니다. 이 모든 선택과 조작 과정은 새로운 채팅 메시지를 계속 밑으로 남기는 방식이 아니라 띄워진 같은 창 하나 안에서 제자리 갱신(in-place)되는 방식이기 때문에 채팅방 공간을 지저분하게 만들지 않습니다.

:::tip
변경할 모델의 정확한 이름을 안다면, 중간 과정을 전부 무시하고 채팅 창에 `/model <name>` 이라고 직접 써넣어도 됩니다. 그리고 변경한 모델을 이번 1회성 채팅이 아니라 계속 유지하려면 `/model <name> --global` 이라고 입력하세요.
:::

## DNS-over-HTTPS 대체 IP (Fallback IPs)

일부 제한적인 네트워크 환경에서는 텔레그램 API 서버인 `api.telegram.org` 주소가 작동하지 않는 곳으로 연결될 수도 있습니다. 텔레그램 어댑터는 원래의 올바른 TLS 호스트 이름과 SNI를 유지하면서도 대체 IP를 연결하는 **대체 IP(fallback IP)** 시스템을 제공하여 이런 문제를 방지합니다.

### 어떻게 동작하나요?

1. 만일 환경 변수 `TELEGRAM_FALLBACK_IPS`가 등록되어 있으면, 거기로 바로 연결합니다.
2. 지정된 IP가 없을 경우 어댑터가 내부적으로 **구글 DNS(Google DNS)**와 **클라우드플레어 DNS(Cloudflare DNS)** 서버에 DNS-over-HTTPS (DoH) 형식으로 질문을 던져서 현재 우회해서 접속할 수 있는 `api.telegram.org` 대체 IP 정보 목록을 알아냅니다.
3. 시스템에 설정된 IP와 다르게 새롭게 알아낸 DoH 기반 IP들을 대체 IP 경로로 확보합니다.
4. 만약에 DNS-over-HTTPS (DoH) 조회 시도마저 가로막힌 열악한 망 환경일 경우, 하드코딩된 내부 비밀 대체 접속용 시스템 IP인 `149.154.167.220`을 최후의 수단(last resort)으로 사용하게 됩니다.
5. 이런 대체 IP 중 하나로 텔레그램 API 통신에 성공하면, 해당 IP 경로가 "끈끈하게 연결(sticky)"된 상태로 전환되며 — 이후의 요청들은 매번 기본 접속부터 시도하여 시간을 낭비하는 일 없이 즉시 해당 대체 접속 경로로 재통신하게 됩니다.

### 구성

```bash
# 명시적인 대체 IP 설정 (콤마(,)로 여러 개 작성 가능)
TELEGRAM_FALLBACK_IPS=149.154.167.220,149.154.167.221
```

아니면, `~/.hermes/config.yaml` 에서:

```yaml
platforms:
  telegram:
    extra:
      fallback_ips:
        - "149.154.167.220"
```

:::tip
이 설정은 보통은 사람이 손댈 필요가 없습니다. 대부분의 망 차단 상황은 DoH 자동 대체 경로 탐색 기술이 먼저 알아서 다 뚫고 지나가기 때문입니다. `TELEGRAM_FALLBACK_IPS` 옵션은 회사 망 같은 곳에서 아예 외부 DoH 경로마저 싹 다 막아버렸을 경우에만 수동으로 지정해서 씁니다.
:::

## 프록시 지원 (Proxy Support)

인터넷에 접속하기 위해 특정 HTTP 프록시 망 통과를 강제하는 네트워크 망(일반적인 기업 망 환경 등)의 경우, 텔레그램 어댑터는 일반적인 표준 프록시 환경변수를 자동 식별하여 시스템에서 일어나는 모든 외부 접속을 프록시를 통해 진행하도록 처리합니다.

### 지원하는 환경변수 목록

어댑터 시스템은 아래의 프록시 환경 변수 목록을 차례대로 탐색하고, 가장 먼저 식별된 변숫값을 채택합니다:

1. `HTTPS_PROXY`
2. `HTTP_PROXY`
3. `ALL_PROXY`
4. `https_proxy` / `http_proxy` / `all_proxy` (소문자 버전)

### 구성

게이트웨이 시스템을 시작하기 전(환경 설정 구역에서)에 환경변수 설정을 해 줍니다:

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
hermes gateway
```

또는 설정 파일인 `~/.hermes/.env` 에 직접 기입합니다:

```bash
HTTPS_PROXY=http://proxy.example.com:8080
```

이 프록시 우회망 설정값은 기본 접속용 시스템뿐만 아니라 앞서 언급했던 예비용 대체 접속망(Fallback IP transport)에도 예외 없이 다 적용됩니다. 이 작업 외에는 Hermes에 어떤 추가 작업이나 조치도 필요하지 않습니다. 시스템에 환경 변수가 세팅되어 있기만 하다면 알아서 프록시 모드로 굴러갑니다.

:::note
이 설명은 Hermes 시스템에서 텔레그램 통신 목적으로 직접 만들어 넣은 예비 우회망 통신 모듈 영역에 한정된 내용입니다. 여기서 말고 딴 데서 쓰는 표준 범용 `httpx` 모듈 통신 기능은 어차피 예전부터 이미 환경변수의 프록시 환경값을 다 반영하고 작동해왔습니다.
:::

## 메시지 반응 (Message Reactions)

봇이 현재 메시지를 처리 중이라는 사실을 겉으로도 시각적으로 표현할 수 있도록 메시지 창에 이모지 반응 스티커(emoji reactions)를 붙일 수 있습니다:

- 👀 봇이 사용자 메시지의 처리 분석에 들어갔을 때 띄웁니다
- ✅ 사용자에 대한 답장과 작업을 무사히 모두 처리 완료했을 때 띄웁니다
- ❌ 작업 도중 어딘가 오류나 문제가 터졌을 때 띄웁니다

이 기능은 기본적으로 꺼져(disabled) 있습니다. `config.yaml` 에서 켜주세요:

```yaml
telegram:
  reactions: true
```

또는 아래의 환경변수로도 가능합니다:

```bash
TELEGRAM_REACTIONS=true
```

:::note
여러 개의 이모지 스티커를 다 같이 달 수 있는 디스코드(Discord) 시스템과 다르게, 텔레그램의 Bot API는 오직 한 번의 명령 콜 하나당 전체를 갈아 끼우는 방식을 사용합니다. 시스템에서 👀 모양 이모지 스티커가 ✅/❌ 모양 이모지로 바뀌는 전환 작업은, 다른 어떤 방해 요소 없이 단일 분자 단위의 일관성 높은 작업 묶음(atomically)으로 이루어집니다. 따라서 스티커가 두 개가 동시에 붙어 있는 모습은 절대 나타날 일이 없습니다.
:::

:::tip
만약 봇의 관리자 권한이 부족하여 그룹 채팅창의 특정 메시지에 이모지 스티커를 못 다는 상황이 닥쳐도, 그냥 조용히 오류만 삼키고 남은 메시지 처리 작업은 아무런 이상 없이 이어나갑니다.
:::

## 채널별 프롬프트 제어 (Per-Channel Prompts)

특정 텔레그램 그룹이나 포럼 주제 창 안의 환경 조건에 맞춰, 잠깐만 쓸 시스템 프롬프트 규칙(ephemeral system prompts)을 지정할 수 있습니다. 이렇게 부여한 프롬프트는 영구적으로 대화 로그(transcript history)에 기록되진 않고 매 대화마다 시스템상(런타임)에서 실시간으로 삽입됩니다. 이 때문에 규칙 내용을 갱신하자마자 채팅방의 반응도 실시간으로 수정됩니다.

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

부여하는 기준 키워드(Keys)는 채팅 ID(채팅/수퍼그룹 ID)나 포럼 주제 ID 값을 씁니다. 포럼 방식 그룹 방의 경우 주제-레벨 프롬프트 정보가 그룹-레벨 프롬프트 정보보다 우선으로 덮어씌웁니다:

- `-1001234567890` 그룹 안에 있는 `42`번 주제 방에서의 대화 → `42` 주제 전용 프롬프트로 처리
- 그 외 `99`번 같은 별도 지정 안 된 방 → 그 윗단계인 `-1001234567890` 그룹 규칙 프롬프트로 처리
- 위 목록에 아예 등록 안 된 그룹 방 → 채널 프롬프트 자체가 씌워지지 않고 평소대로 처리됨

숫자로 표기된 YAML 설정값들은 자동으로 인식하기 쉬운 문자열 형태로 변환 정규화(normalized)되어 인식합니다.

## 문제 해결 (Troubleshooting)

| 문제 | 해결 방법 |
|---------|----------|
| 봇이 아무런 반응을 하지 않음 | `TELEGRAM_BOT_TOKEN` 값이 틀린지 확인하세요. `hermes gateway`의 로그 창에서 오류 원인을 찾아보세요. |
| 봇이 "unauthorized(권한 없음)"로 반응함 | 당신의 사용자 ID값이 허가 목록인 `TELEGRAM_ALLOWED_USERS` 에 없습니다. @userinfobot을 통해 자신의 ID를 다시블체크하세요. |
| 그룹 방에서 메시지를 무시함 | 개인정보 보호 모드(Privacy mode)가 켜진 상태일 겁니다. 모드를 끄거나(Step 3 참고) 아예 봇에게 그룹 관리자 권한을 줘버리세요. **주의! 보호 모드 설정을 변경했다면 해당 방에서 봇을 삭제 후 다시 초대(re-add)해야 적용됩니다.** |
| 음성 메시지가 텍스트로 안 바뀜(미작동) | STT 설정이 켜져 있는지 확인하세요: `faster-whisper` 모델을 PC에 깔아서 자체 변환을 하거나, `~/.hermes/.env` 파일 안에 `GROQ_API_KEY` 나 `VOICE_TOOLS_OPENAI_KEY` API 키를 기입해 넣어야 합니다. |
| 봇의 음성 응답이 버블 모양이 아니라 파일 형식으로 왔을 때 | 시스템에 `ffmpeg` 파일 변환기를 설치하세요. (Edge TTS 형식 파일을 텔레그램 Opus 형식으로 바꾸기 위해 필수입니다). |
| 봇 인증 토큰 정보가 망가지거나 폐기된 상태일 때 | BotFather에서 `/revoke`를 누른 뒤 `/newbot`이나 `/token` 명령어를 통해 새 토큰 값을 발급받으세요. 새로 발급받은 값을 `.env`에 업데이트 적용하세요. |
| 웹훅이 업데이트를 받지 않음 | `TELEGRAM_WEBHOOK_URL`에 밖에서 공개적으로 접속할 수 있는지 확인하세요(테스트 방법: `curl`). 당신의 시스템 플랫폼/리버스 프록시가 웹 URL 포트를 통해 들어오는 인바운드 HTTPS 통신 신호들을 내부적으로 `TELEGRAM_WEBHOOK_PORT` 쪽으로 제대로 라우팅해 주는지 확인하세요 (굳이 같은 포트 번호일 필요는 없습니다). 텔레그램 서버는 오직 HTTPS 기반의 URL 주소로만 정보를 보내기 때문에, 보안 암호 설정(SSL/TLS)이 올바르게 켜져 있는지 꼼꼼하게 따져보세요. 방화벽 규칙도 함께 확인하세요. |

## 실행 승인 (Exec Approval)

에이전트가 위험할 가능성이 있는(예: recursive delete 등) 치명적인 명령어를 내리려고 할 때마다, 채팅창 화면에 안내문을 띄우고 시스템 실행 허가를 요청합니다:

> ⚠️ 이 명령어는 위험할 수 있습니다(하위 디렉토리 전체 삭제). 허용하시려면 "yes"로 답장해 주세요.

이에 동의하려면 채팅창에 "yes" 또는 "y" 라고 입력하고, 거절하려면 "no" 또는 "n" 을 칩니다.

## 대화형 선택 프롬프트 (Interactive Prompts: clarify)

에이전트가 사용자에게 어떤 옵션이나 접근 방식, 결정 사항 등을 명확히 하라고 요구하는 상황(clarify 도구 사용 시)이 올 때, 텔레그램은 **인라인 버튼 키보드(inline keyboard buttons)** 형태로 질문과 선택지를 제공합니다:

> ❓ 대시보드 제작을 위해 어느 프레임워크 툴을 쓸까요?
>
> [1. Next.js] [2. Remix] [3. Astro]
> [✏️ 기타(Other) - (직접 적기)]

답을 하려면 그냥 버튼을 누르고, 그 외 다른 복잡한 설명을 적으려면 **Other** 버튼을 클릭하여 채팅창에서 직접 타이핑하세요 (방금 친 그 메시지가 다음 작업에 반영됩니다). 아예 객관식 선택지가 없는 열린 질문형 `clarify` 호출 시엔 버튼 기능 자체가 건너뛰어지고 바로 사용자의 채팅 답변만을 기다리게 됩니다.

응답 대기 시간 조절은 `~/.hermes/config.yaml` 의 `agent.clarify_timeout` 값으로 통제합니다 (기본값은 `600`초). 만약 정해진 대기시간 동안 사용자의 아무 응답도 들어오지 않으면, 에이전트는 무한정 얼어 있는 대신 경계 메시지(sentinel message)를 띄워 차단(unblocks) 상태를 스스로 해제하고 상황에 맞게 자체적으로 작업을 조정(adapt)해 나갑니다.

## 푸시 알림 수위 조절 (Push notification volume)

기본적으로 텔레그램 봇은 메시지를 보낼 때마다 꼬박꼬박 스마트폰 푸시(push) 알림을 울려댑니다. 특히 도구의 작업 경과(tool-progress bubbles) 표시라든지, 스트리밍 진행 알림 및 콜백 상태 알림 등을 빈번히 내놓는 수다스러운 에이전트의 턴이 돌아오면, 알림 폭탄이 떨어져 매우 시끄럽고 불편해집니다. 이를 통제하기 위해 텔레그램 어댑터에는 알림 조절 모드 2개가 존재합니다:

| 통제 모드 | 동작 방식 |
|------|----------|
| `important` (기본값) | 에이전트의 **최종 결괏값**, **승인 대기 프롬프트**, 그리고 **슬래시 명령어 처리 결과 확인문** 들만 소리를 내며(ring) 알람을 보냅니다. 자잘한 도구 작업 진척 상황과 스트리밍 분할 덩어리들(streaming chunks), 콜백 상태 정보들은 `disable_notification=true` 처리된 상태로 전송됩니다(무음으로 조용히 옴). |
| `all` | 밖으로 내보내는 봇의 모든 채팅 메시지가 빠짐없이 푸시 알림을 발생시킵니다. (구버전의 레거시 행동 방식. 에이전트의 일거수일투족을 소리로 다 감시하고 싶은 경우에만 쓰세요.) |

`~/.hermes/config.yaml` 환경 설정:

```yaml
display:
  platforms:
    telegram:
      notifications: important   # 또는 "all"
```

환경 변수로 오버라이드 제어 (빠른 A/B 테스트 목적 시 유용함):

```bash
HERMES_TELEGRAM_NOTIFICATIONS=all
```

지원하지 않는 엉뚱한 값을 기입하면 시스템 로그에 경고를 띄우고 `important` 모드로 돌아갑니다(fallback).

## 상태 메시지 제자리 갱신 (Status messages edited in place)

텔레그램 어댑터는 주기적으로 반복되는 에이전트의 상태 콜백 정보(예: "맥락 압축 중…(Compressing context…)", "도구 호출 중…(Calling tool…)" 등)를 처리할 때, 매번 새로운 말풍선을 만들고 지우는 것을 반복하는 대신 `send_or_update_status()` 경로를 활용하여 `{(chat_id, status_key) → message_id}` 캐시 내역을 담아두고 그 말풍선의 **원래 텍스트 내용을 덮어쓰며 수정(edits the existing bubble)**합니다. 구별되는 각각의 `status_key` 값들이 서로 각자 자신들만의 독립적인 메시지 공간을 확보하게 되고 이로 인해 서로 다른 채팅끼리 데이터가 충돌하는 엉키는 현상은 방지됩니다. 수정 작업 시도 중에 모종의 이유로 오류(메시지를 사용자가 이미 삭제해 버렸거나 너무 오래된 메시지라 텔레그램의 수정 작업 제한 한계를 넘어버린 등)가 발생하면 기존 캐시 항목 내용을 삭제한 후 아예 깔끔한 새로운 메시지를 투척(post a fresh message)하면서 이 메시지의 새 ID 값을 새로 캐시에 재저장시킵니다. 이 모든 과정들은 기본 동작이라 특별한 텔레그램 구성/설정이 불필요합니다. 한편 다른 어댑터들 중 내부적으로 `send_or_update_status` 구성을 갖추지 못한 시스템들은 이 작업이 일반적인 평범한 `send()` 동작 방식으로 그냥 치환되어 떨어집니다(fall through).

## 에이전트 동작 턴 동안 사용자 메시지 고정 (Pin incoming user message during agent turn)

사용자가 보낸 메시지가 에이전트를 가동시키는 트리거 턴(agent turn) 기능을 작동시킨 경우, 텔레그램 어댑터가 화면 안에서 사용자의 그 원본 인바운드 메시지를 에이전트 턴이 끝날 때까지 윗단에 핀으로 잠시 고정(pin)시켜 주었다가(시각적 인디케이터 용도) 응답 처리가 완료되면 고정 해제(unpins)해 줍니다. 이는 봇이 그 메시지를 씹거나 무시한 것이 아니라 현재 이 메시지를 기반으로 능동적으로 열심히 분석 및 작업을 가동하고 있다는 점을 사용자 측에 시각적으로 편하게 환기해주는 가벼운 장치입니다. 추가적인 알람 공해를 유발하지 않도록, 이 고정 시스템 내부에는 조용히 `disable_notification=true` 기능이 적용되어 있습니다. 따로 구성할 설정은 없습니다.

## 보안 (Security)

:::warning
기본적으로 봇과 상호작용할 수 있는 권한을 주려면 `TELEGRAM_ALLOWED_USERS` 옵션은 언제나 반드시 세팅하세요. 혹여 이 권한 옵션을 무시하고 지워버린다면, 게이트웨이가 스스로 자체적인 보안 수단으로써 기본적으로 모든 유저들의 접근을 다 막아버리는 안전 모드 조치를 강제 시행하게 됩니다.
:::

당신의 봇 토큰 번호를 절대 인터넷 공간 등 외부에 공용으로 노출하지 마세요. 혹시라도 정보가 타인에게 노출되었다는 느낌이 드는 즉시 바로 BotFather의 `/revoke` 명령어를 날려 그 즉시 기존 토큰 사용을 취소 처리하세요.

세부 내용에 대해서는 [Security documentation(보안 문서)](/user-guide/security)을 한 번 훑어보세요. 이와 별개로, [DM pairing](/user-guide/messaging#dm-pairing-alternative-to-allowlists) 방식을 활용하면 매번 리스트를 직접 치고 고치는 복잡한 allowlists 방식 말고도 사용자 권한 관리를 더욱 동적인 체계로 처리할 수도 있습니다.
