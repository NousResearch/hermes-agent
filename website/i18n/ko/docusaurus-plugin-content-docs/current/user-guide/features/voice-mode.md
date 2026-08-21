---
sidebar_position: 10
title: "음성 모드"
description: "Hermes Agent와 실시간 음성 대화 — CLI, Telegram, Discord(DM, 텍스트 채널, 음성 채널)"
---

# 음성 모드

Hermes Agent는 CLI와 메시징 플랫폼 전반에서 완전한 음성 상호작용을 지원합니다. 마이크로 에이전트와 대화하고, 음성으로 답변을 듣고, Discord 음성 채널에서 실시간 음성 대화를 나눌 수 있습니다.

권장 구성과 실제 사용 패턴을 포함한 실용적인 설정 안내가 필요하다면 [Hermes에서 음성 모드 사용하기](../../guides/use-voice-mode-with-hermes.md)를 참고하세요.

CLI, TUI 또는 데스크톱 앱에서 "hey hermes"(또는 어떤 문구든)라고 말해 새 음성 세션을 핸즈프리로 시작하려면 [Wake Word](/user-guide/features/wake-word)를 참고하세요.

## 사전 요구 사항

음성 기능을 사용하기 전에 다음을 확인하세요.

1. **Hermes Agent가 설치되어 있어야 합니다** — 설치 스크립트를 사용하세요([설치](/getting-started/installation) 참고).
2. **LLM 제공자가 구성되어 있어야 합니다** — `hermes model`을 실행하거나 `~/.hermes/.env`에 원하는 제공자의 인증 정보를 설정하세요.
3. **기본 설정이 정상적으로 작동해야 합니다** — 음성을 활성화하기 전에 `hermes`를 실행해 에이전트가 텍스트에 응답하는지 확인하세요.

:::tip
`~/.hermes/` 디렉터리와 기본 `config.yaml`은 처음 `hermes`를 실행할 때 자동으로 생성됩니다. API 키를 위해 `~/.hermes/.env`만 직접 생성하면 됩니다.
:::

:::tip Nous Portal은 두 기능을 모두 제공합니다
유료 [Nous Portal](/user-guide/features/tool-gateway) 구독은 LLM(2단계)과 Tool Gateway를 통한 OpenAI TTS를 모두 제공하므로 별도의 OpenAI 키가 필요하지 않습니다. 새로 설치한 경우 `hermes setup --portal`이 두 기능을 한 번에 연결합니다.
:::

## 개요

| 기능 | 플랫폼 | 설명 |
|---------|----------|-------------|
| **대화형 음성** | CLI | Ctrl+B를 눌러 녹음하고, 에이전트가 무음을 자동으로 감지한 뒤 응답합니다 |
| **자동 음성 답변** | Telegram, Discord | 에이전트가 텍스트 답변과 함께 음성 오디오를 보냅니다 |
| **음성 채널** | Discord | 봇이 VC에 참여해 사용자의 말을 듣고 음성으로 답변합니다 |

## 요구 사항

### Python 패키지

```bash
# CLI voice mode (microphone + audio playback)
cd ~/.hermes/hermes-agent && uv pip install -e ".[voice]"

# Discord + Telegram messaging (includes discord.py[voice] for VC support)
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"

# Premium TTS (ElevenLabs)
cd ~/.hermes/hermes-agent && uv pip install -e ".[tts-premium]"

# Local TTS (NeuTTS, optional)
python -m pip install -U neutts[all]

# Everything at once
cd ~/.hermes/hermes-agent && uv pip install -e ".[all]"
```

| 추가 기능 | 패키지 | 필요한 기능 |
|-------|----------|-------------|
| `voice` | `sounddevice`, `numpy` | CLI 음성 모드 |
| `messaging` | `discord.py[voice]`, `python-telegram-bot`, `aiohttp` | Discord 및 Telegram 봇 |
| `tts-premium` | `elevenlabs` | ElevenLabs TTS 제공자 |

선택 사항인 로컬 TTS 제공자는 `python -m pip install -U neutts[all]`로 `neutts`를 별도로 설치하세요. 처음 사용하면 모델이 자동으로 다운로드됩니다.

:::info
`discord.py[voice]`는 음성 암호화용 **PyNaCl**과 **opus 바인딩**을 자동으로 설치합니다. Discord 음성 채널을 지원하려면 반드시 필요합니다.
:::

### 시스템 종속성

```bash
# macOS
brew install portaudio ffmpeg opus
brew install espeak-ng   # for NeuTTS

# Ubuntu/Debian
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng   # for NeuTTS
```

| 종속성 | 용도 | 필요한 기능 |
|-----------|---------|-------------|
| **PortAudio** | 마이크 입력 및 오디오 재생 | CLI 음성 모드 |
| **ffmpeg** | 오디오 형식 변환(MP3 → Opus, PCM → WAV) | 모든 플랫폼 |
| **Opus** | Discord 음성 코덱 | Discord 음성 채널 |
| **espeak-ng** | 음소 변환 백엔드 | 로컬 NeuTTS 제공자 |

### API 키

`~/.hermes/.env`에 추가하세요.

```bash
# Speech-to-Text — local provider needs NO key at all
# pip install faster-whisper          # Free, runs locally, recommended
GROQ_API_KEY=your-key                 # Groq Whisper — fast, free tier (cloud)
VOICE_TOOLS_OPENAI_KEY=your-key       # OpenAI Whisper — paid (cloud)

# Text-to-Speech (optional — Edge TTS and NeuTTS work without any key)
ELEVENLABS_API_KEY=***           # ElevenLabs — premium quality
# VOICE_TOOLS_OPENAI_KEY above also enables OpenAI TTS
```

:::tip
`faster-whisper`가 설치되어 있으면 STT에 **API 키가 전혀 필요하지 않습니다**. 모델(`base` 기준 약 150MB)은 처음 사용할 때 자동으로 다운로드됩니다.
:::

---

## CLI 음성 모드

음성 모드는 **기존 CLI**(`hermes chat`)와 **TUI**(`hermes --tui`)에서 모두 사용할 수 있습니다. 두 환경의 동작은 동일합니다. 슬래시 명령, VAD 무음 감지, 스트리밍 TTS, 환각 필터를 모두 공유합니다. TUI는 여기에 충돌 포렌식 로그를 `~/.hermes/logs/`로 전달하므로, 특이한 오디오 백엔드에서 발생한 푸시 투 토크 오류도 조용히 사라지지 않고 전체 스택 트레이스와 함께 보고할 수 있습니다.

### 빠른 시작

CLI를 시작하고 음성 모드를 활성화하세요.

```bash
hermes                # Start the interactive CLI
```

그런 다음 CLI 안에서 다음 명령을 사용하세요.

```
/voice          Toggle voice mode on/off
/voice on       Enable voice mode
/voice off      Disable voice mode
/voice tts      Toggle TTS output
/voice status   Show current state
```

### 작동 방식

1. `hermes`로 CLI를 시작하고 `/voice on`으로 음성 모드를 활성화합니다.
2. **Ctrl+B를 누릅니다** — 삐 소리(880Hz)가 나고 녹음이 시작됩니다.
3. **말합니다** — 실시간 오디오 레벨 막대에 입력이 표시됩니다: `● [▁▂▃▅▇▇▅▂] ❯`
4. **말을 멈춥니다** — 3초간 무음이 지속되면 녹음이 자동으로 멈춥니다.
5. 녹음이 끝났음을 알리는 **두 번의 삐 소리**(660Hz)가 납니다.
6. Whisper로 오디오를 텍스트로 변환해 에이전트에 보냅니다.
7. TTS가 활성화되어 있으면 에이전트의 답변을 소리 내어 읽습니다.
8. 녹음이 **자동으로 다시 시작되므로** 키를 누르지 않고 계속 말할 수 있습니다.

이 루프는 녹음 중 **Ctrl+B**를 누르거나(연속 모드 종료), 연속으로 3번 녹음에서 음성이 감지되지 않을 때까지 계속됩니다.

:::tip
녹음 키는 `~/.hermes/config.yaml`의 `voice.record_key`로 설정할 수 있습니다(기본값: `ctrl+b`).
:::

### 무음 감지

말을 끝냈는지 감지하는 2단계 알고리즘을 사용합니다.

1. **음성 확인** — RMS 임계값(200)을 넘는 오디오가 최소 0.3초 동안 들어올 때까지 기다리며, 음절 사이의 짧은 끊김은 허용합니다.
2. **종료 감지** — 음성이 확인되면 3.0초 동안 연속으로 무음이 지속될 때 녹음을 종료합니다.

15초 동안 음성이 전혀 감지되지 않으면 녹음이 자동으로 멈춥니다.

`silence_threshold`와 `silence_duration`은 `config.yaml`에서 설정할 수 있습니다. `voice.beep_enabled: false`로 녹음 시작/종료 알림음을 끌 수도 있습니다.

### 음성으로 음성 채팅 종료하기

**"stop"**이라고, 다른 말 없이 말하면 핸즈프리로 음성 대화를 종료할 수 있습니다. 이 일치는 의도적으로 엄격합니다. 문장 전체가 대소문자 구분 없이, 앞뒤 구두점을 제외하고 구성된 문구와 같아야 하므로 "stop doing that and try X instead"는 정상적으로 에이전트에 전달됩니다. `config.yaml`의 `voice.stop_phrases`로 문구 목록을 변경하거나(예: `["stop", "goodbye hermes"]`), `[]`로 설정해 비활성화할 수 있습니다. 음성 대화는 음성이 감지되지 않는 무음 사이클이 3회 연속되어도 자동으로 종료됩니다.

음성 채팅 중 **stop 문구를 그대로 입력**해도 모든 환경(CLI, TUI, 데스크톱)에서 동일하게 동작합니다. 해당 메시지는 에이전트로 전송되지 않고 음성 채팅을 종료합니다. 음성 채팅 밖에서는 입력한 "stop"이 일반 메시지로 처리됩니다.

### 스트리밍 TTS

TTS가 활성화되어 있으면 에이전트가 답변 전체를 생성할 때까지 기다리지 않고, 텍스트를 생성하는 대로 **문장 단위로** 읽습니다. 이는 **모든 TTS 제공자**에서 작동합니다.

1. 텍스트 델타를 완전한 문장으로 버퍼링합니다(최소 20자).
2. Markdown 서식, 이모지, `<think>` 블록을 제거합니다.
3. 문장이 완성되는 즉시 실시간으로 문장별 오디오를 재생합니다. 청크 PCM API를 지원하는 제공자(ElevenLabs, OpenAI)는 첫 단어까지의 시간을 최소화하기 위해 원시 오디오를 스트리밍하고, 그 밖의 모든 제공자(기본 Edge 포함)는 문장이 완성될 때마다 합성하고 재생합니다.

동일한 파이프라인이 기존 CLI, TUI, 데스크톱 앱에서 실행됩니다. 데스크톱 음성 대화에서는 모델이 답변을 생성하는 동안 답변 텍스트가 **실시간으로** 답변별 음성 WebSocket에 전달되므로 음성이 생성과 겹쳐 실행됩니다. 답변마다 소켓 하나와 오디오 클록 하나를 사용하며, 문장마다 연결이 끊겼다 다시 연결되지 않습니다.

### 끼어들기

턴이 진행되는 **어느 시점에서든** 에이전트를 중단할 수 있습니다. 양방향 통신이므로 말을 끝낸 순간부터 답변이 완전히 재생될 때까지 마이크가 계속 켜져 있습니다.

- **에이전트가 생각하는 동안 끼어들기** — 연속 음성 모드에서 LLM이 생성하는 동안(오디오가 재생되기 전) 말하면 진행 중인 턴이 중단되고, 끼어든 말이 실행 중인 턴에 입력한 메시지와 같은 방식으로 다음 메시지가 됩니다.
- **말하는 중에 덮어쓰기** — 에이전트 답변이 재생되는 동안 말하면 말을 시작하는 즉시 재생이 중단되고, 말한 내용이 제출됩니다. 감지기는 턴 시작 시 *조용한 방*을 기준으로 소음 바닥을 보정하며(재생음 자체를 기준으로 삼지 않음), 스피커 소리가 감지기를 마비시키지 않고 일반적인 말소리는 안정적으로 감지하도록 합니다.
- **입력하거나 녹음 키 누르기** — 새 메시지를 보내거나 푸시 투 토크 키를 누르면 모든 환경에서 재생이 즉시 중단됩니다.
- **"stop"이라고 말하기** — stop 문구는 두 단계 모두에서 작동합니다. 생성 중에는 턴을 중단하고 음성 채팅을 종료하며, 재생 중에는 음성을 끊고 채팅을 종료합니다.

조정(`config.yaml`): `voice.barge_in: false`로 끼어들기를 비활성화하고, `voice.barge_in_threshold_multiplier`(기본값 `3.0`)로 조용한 방의 소음 바닥에 대한 음성 트리거 배율을 조정하며, `voice.barge_in_grace_seconds`(기본값 `0.5`)로 재생 시작 직후의 오작동을 억제할 수 있습니다. `HERMES_VOICE_DEBUG=1`로 블록별 VAD 진단(보정된 바닥값, RMS, 트리거 결정)을 stderr에 스트리밍해 실시간으로 조정할 수 있습니다.

에이전트는 자신이 중단되었다는 사실을 **인지합니다**. 다음 메시지에 음성 답변이 중간에 끊겼다는 짧은 안내가 포함되므로, 상황에 자연스럽게 반응하거나("무례하네요!") 중단된 부분부터 이어갈 수 있습니다.

### 환각 필터

Whisper는 때때로 무음이나 배경 소음에서 유령 텍스트("Thank you for watching", "Subscribe" 등)를 생성합니다. 에이전트는 여러 언어에 걸친 알려진 환각 문구 26개와 반복적인 변형을 포착하는 정규식 패턴을 사용해 이런 텍스트를 걸러냅니다.

---

## 게이트웨이 음성 답변(Telegram 및 Discord)

아직 메시징 봇을 설정하지 않았다면 플랫폼별 안내를 참고하세요.
- [Telegram 설정 안내](../messaging/telegram.md)
- [Discord 설정 안내](../messaging/discord.md)

게이트웨이를 시작해 메시징 플랫폼에 연결하세요.

```bash
hermes gateway        # Start the gateway (connects to configured platforms)
hermes gateway setup  # Interactive setup wizard for first-time configuration
```

### Discord: 채널과 DM

봇은 Discord에서 두 가지 상호작용 모드를 지원합니다.

| 모드 | 대화 방법 | 멘션 필요 | 설정 |
|------|------------|----------------|-------|
| **다이렉트 메시지(DM)** | 봇 프로필 열기 → "메시지" | 아니요 | 즉시 작동 |
| **서버 채널** | 봇이 있는 텍스트 채널에 입력 | 예(`@botname`) | 봇을 서버에 초대해야 함 |

**DM(개인 사용에 권장):** 봇과 DM을 열고 입력하기만 하면 됩니다. @멘션이 필요하지 않습니다. 음성 답변과 모든 명령은 채널에서와 동일하게 작동합니다.

**서버 채널:** 봇은 사용자가 @멘션할 때만 응답합니다(예: `@hermesbyt4 hello`). 멘션 팝업에서 같은 이름의 역할이 아니라 **봇 사용자**를 선택해야 합니다.

:::tip
서버 채널에서 멘션 요구 사항을 끄려면 `~/.hermes/.env`에 다음을 추가하세요.
```bash
DISCORD_REQUIRE_MENTION=false
```
또는 특정 채널을 자유 응답(멘션 불필요) 채널로 설정할 수 있습니다.
```bash
DISCORD_FREE_RESPONSE_CHANNELS=123456789,987654321
```
:::

### 명령

다음 명령은 Telegram과 Discord(DM 및 텍스트 채널)에서 모두 작동합니다.

```
/voice          Toggle voice mode on/off
/voice on       Voice replies only when you send a voice message
/voice tts      Voice replies for ALL messages
/voice off      Disable voice replies
/voice status   Show current setting
```

### 모드

| 모드 | 명령 | 동작 |
|------|---------|----------|
| `off` | `/voice off` | 텍스트만 사용(기본값) |
| `voice_only` | `/voice on` | 음성 메시지를 보낼 때만 음성으로 답변 |
| `all` | `/voice tts` | 모든 메시지에 음성으로 답변 |

음성 모드 설정은 게이트웨이를 다시 시작해도 유지됩니다.

### 플랫폼별 전송

| 플랫폼 | 형식 | 참고 |
|----------|--------|-------|
| **Telegram** | 음성 버블(Opus/OGG) | 채팅 안에서 바로 재생됩니다. 필요한 경우 ffmpeg가 MP3를 Opus로 변환합니다 |
| **Discord** | 네이티브 음성 버블(Opus/OGG) | 사용자의 음성 메시지처럼 바로 재생됩니다. 음성 버블 API가 실패하면 파일 첨부로 대체됩니다 |

---

## Discord 음성 채널

가장 몰입감 높은 음성 기능입니다. 봇이 Discord 음성 채널에 참여해 사용자의 말을 듣고, 음성을 텍스트로 변환하고, 에이전트를 통해 처리한 뒤, 음성 채널에서 답변을 읽어 줍니다.

### 설정

#### 1. Discord 봇 권한

텍스트용 Discord 봇을 이미 설정했다면([Discord 설정 안내](../messaging/discord.md) 참고), 음성 권한을 추가해야 합니다.

[Discord Developer Portal](https://discord.com/developers/applications) → 애플리케이션 → **Installation** → **Default Install Settings** → **Guild Install**로 이동하세요.

**기존 텍스트 권한에 다음 권한을 추가하세요.**

| 권한 | 목적 | 필수 |
|-----------|---------|----------|
| **Connect** | 음성 채널에 참여 | 예 |
| **Speak** | 음성 채널에서 TTS 오디오 재생 | 예 |
| **Use Voice Activity** | 사용자가 말하는지 감지 | 권장 |

**업데이트된 권한 정수:**

| 수준 | 정수 | 포함 항목 |
|-------|---------|----------------|
| 텍스트만 | `309237763136` | 채널 보기, 메시지 보내기, 기록 읽기, 임베드, 첨부 파일, 스레드, 리액션, 공개 스레드 만들기 |
| 텍스트 + 음성 | `309240908864` | 위 항목 모두 + Connect, Speak |

업데이트된 권한 URL로 **봇을 다시 초대하세요.**

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=309240908864
```

`YOUR_APP_ID`를 Developer Portal의 Application ID로 바꾸세요.

:::warning
봇이 이미 들어가 있는 서버에 봇을 다시 초대하면 봇을 제거하지 않고 권한만 업데이트합니다. 데이터나 설정은 손실되지 않습니다.
:::

#### 2. 권한이 필요한 Gateway Intent

[Developer Portal](https://discord.com/developers/applications) → 애플리케이션 → **Bot** → **Privileged Gateway Intents**에서 다음 세 가지를 모두 활성화하세요.

| Intent | 목적 |
|--------|---------|
| **Presence Intent** | 사용자의 온라인/오프라인 상태 감지 |
| **Server Members Intent** | `DISCORD_ALLOWED_USERS`의 사용자 이름을 숫자 ID로 확인(조건부) |
| **Message Content Intent** | 채널의 텍스트 메시지 내용 읽기 |

**Message Content Intent**는 필수입니다. `DISCORD_ALLOWED_USERS` 목록에서 사용자 이름을 사용하는 경우에만 **Server Members Intent**가 필요합니다. 숫자 사용자 ID를 사용한다면 끈 상태로 두어도 됩니다. 음성 채널의 SSRC → user_id 매핑은 음성 WebSocket의 Discord SPEAKING opcode에서 가져오므로 Server Members Intent가 필요하지 않습니다.

#### 3. Opus 코덱

게이트웨이를 실행하는 시스템에 Opus 코덱 라이브러리를 설치해야 합니다.

```bash
# macOS (Homebrew)
brew install opus

# Ubuntu/Debian
sudo apt install libopus0
```

봇은 다음 경로에서 코덱을 자동으로 불러옵니다.
- **macOS:** `/opt/homebrew/lib/libopus.dylib`
- **Linux:** `libopus.so.0`

#### 4. 환경 변수

```bash
# ~/.hermes/.env

# Discord bot (already configured for text)
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=your-user-id

# STT — local provider needs no key (pip install faster-whisper)
# GROQ_API_KEY=your-key            # Alternative: cloud-based, fast, free tier

# TTS — optional. Edge TTS and NeuTTS need no key.
# ELEVENLABS_API_KEY=***      # Premium quality
# VOICE_TOOLS_OPENAI_KEY=***  # OpenAI TTS / Whisper
```

### 게이트웨이 시작

```bash
hermes gateway        # Start with existing configuration
```

몇 초 안에 Discord에서 봇이 온라인 상태가 되어야 합니다.

### 명령

봇이 있는 Discord 텍스트 채널에서 다음 명령을 사용하세요.

```
/voice join      Bot joins your current voice channel
/voice channel   Alias for /voice join
/voice leave     Bot disconnects from voice channel
/voice status    Show voice mode and connected channel
```

:::info
`/voice join`을 실행하기 전에 음성 채널에 들어가 있어야 합니다. 봇은 사용자가 들어가 있는 동일한 VC에 참여합니다.
:::

### 작동 방식

봇이 음성 채널에 참여하면 다음을 수행합니다.

1. 각 사용자의 오디오 스트림을 독립적으로 **수신**합니다.
2. **무음을 감지**합니다 — 최소 0.5초의 음성 후 1.5초 동안 무음이면 처리를 시작합니다.
3. Whisper STT(로컬, Groq 또는 OpenAI)로 오디오를 **텍스트로 변환**합니다.
4. 전체 에이전트 파이프라인(세션, 도구, 메모리)을 통해 **처리**합니다.
5. TTS로 답변을 음성 채널에서 **읽어 줍니다**.

### 텍스트 채널 연동

봇이 음성 채널에 있을 때:

- 텍스트 채널에 변환 결과가 표시됩니다: `[Voice] @user: what you said`
- 에이전트의 답변은 채널에 텍스트로 전송되고 VC에서 음성으로도 재생됩니다.
- 텍스트 채널은 `/voice join`을 실행한 채널입니다.

### 에코 방지

봇은 TTS 답변을 재생하는 동안 오디오 리스너를 자동으로 일시 중지해 자신의 출력을 듣고 다시 처리하는 것을 방지합니다.

### 접근 제어

`DISCORD_ALLOWED_USERS`에 등록된 사용자만 음성으로 상호작용할 수 있습니다. 다른 사용자의 오디오는 조용히 무시됩니다.

```bash
# ~/.hermes/.env
DISCORD_ALLOWED_USERS=284102345871466496
```

---

## 구성 참고

### config.yaml

```yaml
# Voice recording (CLI)
voice:
  record_key: "ctrl+b"            # Key to start/stop recording
  max_recording_seconds: 120       # Maximum recording length
  auto_tts: false                  # Auto-enable TTS when voice mode starts
  beep_enabled: true               # Play record start/stop beeps
  silence_threshold: 200           # RMS level (0-32767) below which counts as silence
  silence_duration: 3.0            # Seconds of silence before auto-stop
  stop_phrases: ["stop"]           # Saying exactly one of these ends the voice chat; [] disables

# Speech-to-Text
stt:
  enabled: true                     # set to false to skip auto-transcription —
                                    # the gateway still caches the audio file and
                                    # passes its path to the agent as part of the
                                    # inbound message, useful for custom pipelines
                                    # (diarization, alignment, archival, etc.)
  provider: "local"                  # "local" (free) | "groq" | "openai" | "mistral" | "xai"
  local:
    model: "base"                    # tiny, base, small, medium, large-v3
    language: ""                     # optional ISO-639-1 hint; blank = use HERMES_LOCAL_STT_LANGUAGE if set, else auto-detect
  groq:
    language: ""                     # optional ISO-639-1 hint; blank = use HERMES_LOCAL_STT_LANGUAGE if set, else auto-detect
  # model: "whisper-1"              # Legacy: used when provider is not set

# Text-to-Speech
tts:
  provider: "edge"                 # "edge" (free) | "elevenlabs" | "openai" | "neutts" | "minimax" | "mistral" | "gemini" | "xai" | "kittentts" | "piper"
  edge:
    voice: "en-US-AriaNeural"      # 322 voices, 74 languages
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"    # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"                 # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # optional: override for self-hosted or OpenAI-compatible endpoints
    # The `text_to_speech` tool accepts an optional per-call `instructions`
    # argument (tone, emotion, pacing, accent, whispering) that is forwarded
    # to `gpt-4o-mini-tts` and to OpenAI-compatible voice-design servers
    # (e.g. Qwen3-TTS-VoiceDesign via oMLX). See OpenAI's voice-design guide:
    # https://platform.openai.com/docs/guides/text-to-speech
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

### 환경 변수

```bash
# Speech-to-Text providers (local needs no key)
# pip install faster-whisper        # Free local STT — no API key needed
GROQ_API_KEY=...                    # Groq Whisper (fast, free tier)
VOICE_TOOLS_OPENAI_KEY=...         # OpenAI Whisper (paid)

# STT advanced overrides (optional)
STT_GROQ_MODEL=whisper-large-v3-turbo    # Override default Groq STT model
STT_OPENAI_MODEL=whisper-1               # Override default OpenAI STT model
GROQ_BASE_URL=https://api.groq.com/openai/v1     # Custom Groq endpoint
STT_OPENAI_BASE_URL=https://api.openai.com/v1    # Custom OpenAI STT endpoint

# Text-to-Speech providers (Edge TTS and NeuTTS need no key)
ELEVENLABS_API_KEY=***             # ElevenLabs (premium quality)
# VOICE_TOOLS_OPENAI_KEY above also enables OpenAI TTS

# Discord voice channel
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=...
```

### STT 제공자 비교

| 제공자 | 모델 | 속도 | 품질 | 비용 | API 키 |
|----------|-------|-------|---------|------|---------|
| **로컬** | `base` | 빠름(CPU/GPU에 따라 다름) | 좋음 | 무료 | 아니요 |
| **로컬** | `small` | 보통 | 더 좋음 | 무료 | 아니요 |
| **로컬** | `large-v3` | 느림 | 최고 | 무료 | 아니요 |
| **Groq** | `whisper-large-v3-turbo` | 매우 빠름(~0.5초) | 좋음 | 무료 티어 | 예 |
| **Groq** | `whisper-large-v3` | 빠름(~1초) | 더 좋음 | 무료 티어 | 예 |
| **OpenAI** | `whisper-1` | 빠름(~1초) | 좋음 | 유료 | 예 |
| **OpenAI** | `gpt-4o-transcribe` | 보통(~2초) | 최고 | 유료 | 예 |
| **OpenAI** | `gpt-transcribe` | 빠름 | 최고 | 유료($0.0045/분) | 예 |
| **Mistral** | `voxtral-mini-latest` | 빠름 | 좋음 | 유료 | 예 |
| **xAI** | `grok-stt` | 빠름 | 좋음 | 유료 | 예 |

제공자 우선순위(자동 대체): **local** > **groq** > **openai**

### TTS 제공자 비교

| 제공자 | 품질 | 비용 | 지연 시간 | 키 필요 |
|----------|---------|------|-------------|-------------|
| **Edge TTS** | 좋음 | 무료 | ~1초 | 아니요 |
| **ElevenLabs** | 뛰어남 | 유료 | ~2초 | 예 |
| **OpenAI TTS** | 좋음 | 유료 | ~1.5초 | 예 |
| **NeuTTS** | 좋음 | 무료 | CPU/GPU에 따라 다름 | 아니요 |

NeuTTS는 위의 `tts.neutts` 구성 블록을 사용합니다.

`openai`의 경우 `text_to_speech` 도구가 선택적 `instructions` 인수를 지원하며, 이를 통해 `gpt-4o-mini-tts`의 음성 디자인 기능(어조, 감정, 속도, 억양, 속삭임)을 사용할 수 있습니다. 같은 필드는 `tts.openai.base_url`을 통해 연결한 OpenAI 호환 음성 디자인 서버(예: oMLX를 통한 Qwen3-TTS-VoiceDesign)에도 전달됩니다.

---

## 문제 해결

### "오디오 장치를 찾을 수 없음"(CLI)

PortAudio가 설치되어 있지 않습니다.

```bash
brew install portaudio    # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

Linux 데스크톱의 Docker에서 Hermes를 실행한다면 컨테이너가 호스트 오디오 소켓에 접근할 수 있어야 합니다. PulseAudio/PipeWire와 호환되는 설정은 [Docker 오디오 브리지](/user-guide/docker#optional-linux-desktop-audio-bridge) 참고 사항을 확인하세요.

### Discord 서버 채널에서 봇이 응답하지 않음

기본적으로 봇은 서버 채널에서 @멘션을 요구합니다. 다음을 확인하세요.

1. `@`를 입력하고 같은 이름의 **역할**이 아니라 `#discriminator`가 표시된 **봇 사용자**를 선택합니다.
2. 또는 DM을 사용합니다 — 멘션이 필요하지 않습니다.
3. 또는 `~/.hermes/.env`에서 `DISCORD_REQUIRE_MENTION=false`로 설정합니다.

### 봇이 VC에 참여하지만 내 말을 듣지 못함

- Discord 사용자 ID가 `DISCORD_ALLOWED_USERS`에 있는지 확인하세요.
- Discord에서 음소거되어 있지 않은지 확인하세요.
- 봇이 오디오를 매핑하려면 Discord의 SPEAKING 이벤트가 필요합니다. 참여한 뒤 몇 초 안에 말하기 시작하세요.

### 봇이 내 말을 듣지만 응답하지 않음

- STT를 사용할 수 있는지 확인하세요. `faster-whisper`를 설치하거나(API 키 불필요) `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY`를 설정하세요.
- LLM 모델이 구성되어 있고 접근 가능한지 확인하세요.
- 게이트웨이 로그를 확인하세요: `tail -f ~/.hermes/logs/gateway.log`

### 봇이 텍스트로는 응답하지만 음성 채널에서는 응답하지 않음

- TTS 제공자가 실패했을 수 있습니다 — API 키와 할당량을 확인하세요.
- Edge TTS(무료, 키 불필요)가 기본 대체 제공자입니다.
- 로그에서 TTS 오류를 확인하세요.

### Whisper가 엉뚱한 텍스트를 반환함

환각 필터가 대부분의 경우를 자동으로 처리합니다. 그래도 유령 변환 결과가 계속 나온다면 다음을 시도하세요.

- 더 조용한 환경을 사용합니다.
- 구성에서 `silence_threshold`를 조정합니다(값이 높을수록 덜 민감함).
- 다른 STT 모델을 사용합니다.
