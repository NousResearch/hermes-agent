---
sidebar_position: 10
title: "음성 모드 (Voice Mode)"
description: "Hermes Agent와 실시간 음성 대화 — CLI, Telegram, Discord (DM, 텍스트 채널 및 음성 채널)"
---

# 음성 모드 (Voice Mode)

Hermes Agent는 CLI 및 메시징 플랫폼 전반에서 완벽한 음성 상호작용을 지원합니다. 마이크를 사용하여 에이전트와 대화하고, 음성으로 된 답변을 들으며, Discord 음성 채널에서 실시간 음성 대화를 나눌 수 있습니다.

권장 구성 및 실제 사용 패턴이 포함된 실용적인 설정 연습을 원하시면 [Hermes와 함께 음성 모드 사용하기](/guides/use-voice-mode-with-hermes)를 참조하세요.

## 전제 조건

음성 기능을 사용하기 전에 다음이 준비되었는지 확인하세요:

1. **Hermes Agent 설치** — `pip install hermes-agent` ([설치](/getting-started/installation) 참조)
2. **LLM 공급자 구성** — `hermes model`을 실행하거나 `~/.hermes/.env`에 선호하는 공급자 자격 증명 설정
3. **작동하는 기본 설정** — `hermes`를 실행하여 음성을 활성화하기 전에 에이전트가 텍스트에 응답하는지 확인

:::tip
`~/.hermes/` 디렉토리와 기본 `config.yaml`은 `hermes`를 처음 실행할 때 자동으로 생성됩니다. API 키를 위해 수동으로 `~/.hermes/.env`만 생성하면 됩니다.
:::

:::tip Nous Portal은 두 가지를 모두 포함합니다
유료 [Nous Portal](/user-guide/features/tool-gateway) 구독은 Tool Gateway를 통해 LLM(2단계) **및** OpenAI TTS를 모두 제공하므로 별도의 OpenAI 키가 필요하지 않습니다. 새로 설치하는 경우, `hermes setup --portal`을 실행하여 두 가지를 한 번에 설정할 수 있습니다.
:::

## 개요

| 기능 | 플랫폼 | 설명 |
|---------|----------|-------------|
| **대화형 음성 (Interactive Voice)** | CLI | Ctrl+B를 눌러 녹음, 에이전트가 침묵을 자동 감지하고 응답 |
| **자동 음성 답장 (Auto Voice Reply)** | Telegram, Discord | 에이전트가 텍스트 응답과 함께 음성 오디오를 전송 |
| **음성 채널 (Voice Channel)** | Discord | 봇이 VC(음성 채널)에 참여하여 사용자의 말을 듣고 음성으로 응답 |

## 요구 사항

### Python 패키지

```bash
# CLI 음성 모드 (마이크 + 오디오 재생)
pip install "hermes-agent[voice]"

# Discord + Telegram 메시징 (VC 지원을 위한 discord.py[voice] 포함)
pip install "hermes-agent[messaging]"

# 프리미엄 TTS (ElevenLabs)
pip install "hermes-agent[tts-premium]"

# 로컬 TTS (NeuTTS, 선택 사항)
python -m pip install -U neutts[all]

# 한 번에 모든 패키지 설치
pip install "hermes-agent[all]"
```

| Extra | 패키지 | 필수 항목 |
|-------|----------|-------------|
| `voice` | `sounddevice`, `numpy` | CLI 음성 모드 |
| `messaging` | `discord.py[voice]`, `python-telegram-bot`, `aiohttp` | Discord 및 Telegram 봇 |
| `tts-premium` | `elevenlabs` | ElevenLabs TTS 공급자 |

선택적인 로컬 TTS 공급자: `python -m pip install -U neutts[all]`을 사용하여 `neutts`를 별도로 설치하세요. 처음 사용할 때 모델이 자동으로 다운로드됩니다.

:::info
`discord.py[voice]`는 **PyNaCl**(음성 암호화용)과 **opus 바인딩**을 자동으로 설치합니다. 이는 Discord 음성 채널 지원을 위해 필요합니다.
:::

### 시스템 종속성

```bash
# macOS
brew install portaudio ffmpeg opus
brew install espeak-ng   # NeuTTS용

# Ubuntu/Debian
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng   # NeuTTS용
```

| 종속성 | 목적 | 필수 항목 |
|-----------|---------|-------------|
| **PortAudio** | 마이크 입력 및 오디오 재생 | CLI 음성 모드 |
| **ffmpeg** | 오디오 포맷 변환 (MP3 → Opus, PCM → WAV) | 모든 플랫폼 |
| **Opus** | Discord 음성 코덱 | Discord 음성 채널 |
| **espeak-ng** | 음소 변환(Phonemizer) 백엔드 | 로컬 NeuTTS 공급자 |

### API 키

`~/.hermes/.env`에 추가하세요:

```bash
# Speech-to-Text (STT) — 로컬 공급자는 키가 전혀 필요하지 않습니다.
# pip install faster-whisper          # 무료, 로컬에서 실행, 권장
GROQ_API_KEY=your-key                 # Groq Whisper — 빠름, 무료 등급 지원 (클라우드)
VOICE_TOOLS_OPENAI_KEY=your-key       # OpenAI Whisper — 유료 (클라우드)

# Text-to-Speech (TTS) (선택 사항 — Edge TTS와 NeuTTS는 키 없이 작동)
ELEVENLABS_API_KEY=***           # ElevenLabs — 프리미엄 품질
# 위의 VOICE_TOOLS_OPENAI_KEY도 OpenAI TTS를 활성화합니다.
```

:::tip
`faster-whisper`가 설치되어 있으면, 음성 모드는 STT를 위해 **API 키 없이(zero API keys)** 작동합니다. 모델(`base`의 경우 ~150 MB)은 처음 사용할 때 자동으로 다운로드됩니다.
:::

---

## CLI 음성 모드

음성 모드는 **클래식 CLI**(`hermes chat`)와 **TUI**(`hermes --tui`) 모두에서 사용할 수 있습니다. 동일한 슬래시 명령어, 동일한 VAD 침묵 감지, 동일한 스트리밍 TTS, 동일한 환각(hallucination) 필터 등 동작은 두 환경에서 동일합니다. TUI는 충돌 포렌식 로그를 `~/.hermes/logs/`로 전달하므로, 특이한 오디오 백엔드에서 푸시 투 토크(push-to-talk)가 실패할 경우 조용히 사라지는 대신 전체 스택 트레이스와 함께 보고할 수 있습니다.

### 빠른 시작

CLI를 시작하고 음성 모드를 활성화하세요:

```bash
hermes                # 대화형 CLI 시작
```

그런 다음 CLI 내부에서 다음 명령어를 사용하세요:

```
/voice          음성 모드 켜기/끄기 토글
/voice on       음성 모드 활성화
/voice off      음성 모드 비활성화
/voice tts      TTS 출력 토글
/voice status   현재 상태 표시
```

### 작동 방식

1. `hermes`로 CLI를 시작하고 `/voice on`으로 음성 모드를 활성화합니다.
2. **Ctrl+B 누르기** — 비프음(880Hz)이 재생되고 녹음이 시작됩니다.
3. **말하기** — 실시간 오디오 레벨 바가 입력을 표시합니다: `● [▁▂▃▅▇▇▅▂] ❯`
4. **말하기 멈춤** — 3초간의 침묵 후 녹음이 자동 중지됩니다.
5. **두 번의 비프음** (660Hz)이 재생되어 녹음이 끝났음을 확인합니다.
6. 오디오는 Whisper를 통해 전사(transcribe)되어 에이전트에게 전송됩니다.
7. TTS가 활성화된 경우 에이전트의 답변을 소리 내어 읽어줍니다.
8. 녹음이 **자동으로 재시작**됩니다 — 아무 키도 누르지 않고 다시 말하면 됩니다.

이 루프는 녹음 중 **Ctrl+B**를 누르거나 (연속 모드 종료), 3번 연속으로 말소리가 감지되지 않을 때까지 계속됩니다.

:::tip
녹음 키는 `~/.hermes/config.yaml`의 `voice.record_key`를 통해 설정할 수 있습니다 (기본값: `ctrl+b`).
:::

### 침묵 감지 (Silence Detection)

2단계 알고리즘이 언제 말을 마쳤는지 감지합니다:

1. **말소리 확인 (Speech confirmation)** — 최소 0.3초 동안 RMS 임계값(200)을 초과하는 오디오를 기다리며, 음절 사이의 짧은 끊김을 허용합니다.
2. **종료 감지 (End detection)** — 말소리가 확인되면, 연속적인 침묵이 3.0초 동안 지속될 때 감지됩니다.

만약 15초 동안 아무 말소리도 감지되지 않으면 녹음이 자동으로 중지됩니다.

`silence_threshold`와 `silence_duration`은 모두 `config.yaml`에서 설정 가능합니다. 또한 `voice.beep_enabled: false`를 통해 녹음 시작/중지 비프음을 비활성화할 수 있습니다.

### 스트리밍 TTS

TTS가 활성화되면 에이전트는 텍스트를 생성하면서 **문장 단위로** 답변을 말합니다 — 전체 응답을 기다릴 필요가 없습니다:

1. 텍스트 델타(deltas)를 완전한 문장으로 버퍼링합니다 (최소 20자).
2. 마크다운 포맷팅과 `<think>` 블록을 제거합니다.
3. 실시간으로 문장별 오디오를 생성하고 재생합니다.

### 환각 필터 (Hallucination Filter)

Whisper는 때때로 침묵이나 배경 소음에서 유령 텍스트(phantom text)를 생성합니다("Thank you for watching", "Subscribe" 등). 에이전트는 여러 언어에 걸쳐 알려진 26개의 환각 문구와 반복되는 패턴을 잡는 정규 표현식을 사용하여 이를 필터링합니다.

---

## 게이트웨이 음성 답장 (Telegram & Discord)

메시징 봇을 아직 설정하지 않았다면 플랫폼별 가이드를 참조하세요:
- [Telegram 설정 가이드](../messaging/telegram.md)
- [Discord 설정 가이드](../messaging/discord.md)

메시징 플랫폼에 연결하기 위해 게이트웨이를 시작하세요:

```bash
hermes gateway        # 게이트웨이 시작 (구성된 플랫폼에 연결)
hermes gateway setup  # 첫 설정을 위한 대화형 설정 마법사
```

### Discord: 채널 vs DM

봇은 Discord에서 두 가지 상호작용 모드를 지원합니다:

| 모드 | 대화 방법 | 멘션 필요 | 설정 |
|------|------------|-----------------|-------|
| **다이렉트 메시지 (DM)** | 봇의 프로필 열기 → "메시지" | 아니오 | 즉시 작동 |
| **서버 채널** | 봇이 있는 텍스트 채널에 입력 | 예 (`@botname`) | 봇이 서버에 초대되어야 함 |

**DM (개인용 권장):** 봇과 DM을 열고 입력하기만 하면 됩니다 — @멘션이 필요 없습니다. 음성 답장 및 모든 명령어는 채널에서와 동일하게 작동합니다.

**서버 채널:** 봇은 당신이 @멘션(예: `@hermesbyt4 hello`)을 할 때만 응답합니다. 멘션 팝업에서 같은 이름의 역할이 아닌 **봇 사용자(bot user)**를 선택했는지 확인하세요.

:::tip
서버 채널에서 멘션 요구사항을 비활성화하려면 `~/.hermes/.env`에 다음을 추가하세요:
```bash
DISCORD_REQUIRE_MENTION=false
```
또는 특정 채널을 자유 응답(멘션 불필요) 채널로 설정하세요:
```bash
DISCORD_FREE_RESPONSE_CHANNELS=123456789,987654321
```
:::

### 명령어

이러한 명령어는 Telegram과 Discord(DM 및 텍스트 채널) 모두에서 작동합니다:

```
/voice          음성 모드 켜기/끄기 토글
/voice on       음성 메시지를 보낼 때만 음성 답장
/voice tts      모든 메시지에 대해 음성 답장
/voice off      음성 답장 비활성화
/voice status   현재 설정 표시
```

### 모드

| 모드 | 명령어 | 동작 |
|------|---------|----------|
| `off` | `/voice off` | 텍스트만 전송 (기본값) |
| `voice_only` | `/voice on` | 음성 메시지를 보낼 때만 음성으로 답장 |
| `all` | `/voice tts` | 모든 메시지에 음성으로 답장 |

음성 모드 설정은 게이트웨이가 재시작되어도 유지됩니다.

### 플랫폼 전송

| 플랫폼 | 포맷 | 참고 |
|----------|--------|-------|
| **Telegram** | 음성 버블 (Opus/OGG) | 채팅에서 인라인으로 재생됩니다. 필요시 ffmpeg가 MP3 → Opus로 변환합니다. |
| **Discord** | 기본 음성 버블 (Opus/OGG) | 사용자 음성 메시지처럼 인라인으로 재생됩니다. 음성 버블 API가 실패할 경우 파일 첨부로 폴백됩니다. |

---

## Discord 음성 채널

가장 몰입감 있는 음성 기능입니다: 봇이 Discord 음성 채널에 참여하여 사용자의 말을 듣고, 음성을 전사하고, 에이전트 파이프라인을 통해 처리한 다음, 음성 채널로 다시 답변을 말합니다.

### 설정

#### 1. Discord 봇 권한

텍스트용으로 이미 Discord 봇을 설정했다면([Discord 설정 가이드](../messaging/discord.md) 참조), 음성 권한을 추가해야 합니다.

[Discord Developer Portal](https://discord.com/developers/applications) → 해당 어플리케이션 → **Installation** → **Default Install Settings** → **Guild Install** 로 이동합니다:

**기존 텍스트 권한에 다음 권한을 추가하세요:**

| 권한 | 목적 | 필수 여부 |
|-----------|---------|----------|
| **Connect** | 음성 채널 참여 | 예 |
| **Speak** | 음성 채널에서 TTS 오디오 재생 | 예 |
| **Use Voice Activity** | 사용자가 말할 때 감지 | 권장 |

**업데이트된 권한 정수 (Permissions Integer):**

| 레벨 | 정수 (Integer) | 포함 내용 |
|-------|---------|----------------|
| 텍스트만 | `274878286912` | 채널 보기, 메시지 보내기, 기록 읽기, 임베드, 첨부파일, 스레드, 반응 |
| 텍스트 + 음성 | `274881432640` | 위의 모든 권한 + Connect, Speak |

업데이트된 권한 URL로 **봇을 다시 초대하세요**:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274881432640
```

`YOUR_APP_ID`를 Developer Portal의 Application ID로 바꾸세요.

:::warning
이미 참여 중인 서버에 봇을 다시 초대하면 봇을 제거하지 않고 권한만 업데이트됩니다. 데이터나 설정은 손실되지 않습니다.
:::

#### 2. Privileged Gateway Intents

[Developer Portal](https://discord.com/developers/applications) → 해당 어플리케이션 → **Bot** → **Privileged Gateway Intents** 에서 3개 모두 활성화하세요:

| Intent | 목적 |
|--------|---------|
| **Presence Intent** | 사용자 온라인/오프라인 상태 감지 |
| **Server Members Intent** | `DISCORD_ALLOWED_USERS`에 있는 사용자 이름을 숫자 ID로 확인 (조건부) |
| **Message Content Intent** | 채널의 텍스트 메시지 내용 읽기 |

**Message Content Intent**는 필수입니다. **Server Members Intent**는 `DISCORD_ALLOWED_USERS` 목록에서 사용자 이름을 사용하는 경우에만 필요합니다 — 숫자 사용자 ID를 사용하는 경우 끌 수 있습니다. 음성 채널 SSRC → user_id 매핑은 음성 웹소켓의 Discord의 SPEAKING opcode에서 오며 Server Members Intent를 **요구하지 않습니다**.

#### 3. Opus 코덱

게이트웨이를 실행하는 머신에 Opus 코덱 라이브러리가 설치되어 있어야 합니다:

```bash
# macOS (Homebrew)
brew install opus

# Ubuntu/Debian
sudo apt install libopus0
```

봇은 다음 위치에서 코덱을 자동으로 로드합니다:
- **macOS:** `/opt/homebrew/lib/libopus.dylib`
- **Linux:** `libopus.so.0`

#### 4. 환경 변수

```bash
# ~/.hermes/.env

# Discord 봇 (이미 텍스트용으로 설정됨)
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=your-user-id

# STT — 로컬 공급자는 키가 필요 없음 (pip install faster-whisper)
# GROQ_API_KEY=your-key            # 대안: 클라우드 기반, 빠름, 무료 등급

# TTS — 선택 사항. Edge TTS 및 NeuTTS는 키가 필요 없음.
# ELEVENLABS_API_KEY=***      # 프리미엄 품질
# VOICE_TOOLS_OPENAI_KEY=***  # OpenAI TTS / Whisper
```

### 게이트웨이 시작

```bash
hermes gateway        # 기존 설정으로 시작
```

봇은 몇 초 내에 Discord에서 온라인 상태가 되어야 합니다.

### 명령어

봇이 참여 중인 Discord 텍스트 채널에서 다음 명령어를 사용하세요:

```
/voice join      봇이 현재 있는 음성 채널에 참여합니다
/voice channel   /voice join의 별칭
/voice leave     봇이 음성 채널에서 연결 해제됩니다
/voice status    음성 모드 및 연결된 채널 표시
```

:::info
`/voice join`을 실행하기 전에 사용자가 음성 채널에 있어야 합니다. 봇은 사용자가 참여한 동일한 VC에 조인합니다.
:::

### 작동 방식

봇이 음성 채널에 참여하면 다음을 수행합니다:

1. **듣기 (Listens)**: 각 사용자의 오디오 스트림을 독립적으로 듣습니다.
2. **침묵 감지 (Detects silence)**: 최소 0.5초의 말소리 후 1.5초의 침묵이 감지되면 처리를 시작합니다.
3. **전사 (Transcribes)**: Whisper STT(Local, Groq 또는 OpenAI)를 통해 오디오를 전사합니다.
4. **처리 (Processes)**: 전체 에이전트 파이프라인(세션, 도구, 메모리)을 거칩니다.
5. **말하기 (Speaks)**: TTS를 통해 음성 채널로 다시 응답을 읽어줍니다.

### 텍스트 채널 연동

봇이 음성 채널에 있을 때:

- 텍스트 채널에 기록이 나타납니다: `[Voice] @user: 사용자가 한 말`
- 에이전트 응답은 채널에 텍스트로 전송되고 VC에서 음성으로 읽어줍니다.
- 텍스트 채널은 `/voice join` 명령어를 내린 채널입니다.

### 에코(메아리) 방지

봇은 TTS 답변을 재생하는 동안 자신의 오디오 청취를 자동으로 일시 중지하여, 자신의 출력을 다시 듣고 처리하는 것을 방지합니다.

### 접근 제어

`DISCORD_ALLOWED_USERS`에 나열된 사용자만 음성으로 상호작용할 수 있습니다. 다른 사용자의 오디오는 조용히 무시됩니다.

```bash
# ~/.hermes/.env
DISCORD_ALLOWED_USERS=284102345871466496
```

---

## 구성 참조 (Configuration Reference)

### config.yaml

```yaml
# 음성 녹음 (CLI)
voice:
  record_key: "ctrl+b"            # 녹음 시작/중지 키
  max_recording_seconds: 120       # 최대 녹음 길이 (초)
  auto_tts: false                  # 음성 모드 시작 시 TTS 자동 활성화
  beep_enabled: true               # 녹음 시작/중지 비프음 재생
  silence_threshold: 200           # 이 값 미만이면 침묵으로 간주하는 RMS 레벨 (0-32767)
  silence_duration: 3.0            # 자동 중지 전의 침묵 지속 시간 (초)

# Speech-to-Text (STT)
stt:
  enabled: true                     # 자동 전사를 건너뛰려면 false로 설정하세요 —
                                    # 게이트웨이는 여전히 오디오 파일을 캐시하고
                                    # 인바운드 메시지의 일부로 경로를 에이전트에게
                                    # 전달합니다. 이는 사용자 지정 파이프라인
                                    # (화자 분리, 정렬, 아카이빙 등)에 유용합니다.
  provider: "local"                  # "local" (무료) | "groq" | "openai"
  local:
    model: "base"                    # tiny, base, small, medium, large-v3
  # model: "whisper-1"              # 레거시: 공급자가 설정되지 않은 경우 사용됨

# Text-to-Speech (TTS)
tts:
  provider: "edge"                 # "edge" (무료) | "elevenlabs" | "openai" | "neutts" | "minimax"
  edge:
    voice: "en-US-AriaNeural"      # 322개 음성, 74개 언어
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"    # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"                 # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # 선택 사항: 자체 호스팅 또는 OpenAI 호환 엔드포인트용 재정의
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

### 환경 변수

```bash
# Speech-to-Text 공급자 (로컬은 키가 필요 없음)
# pip install faster-whisper        # 무료 로컬 STT — API 키 필요 없음
GROQ_API_KEY=...                    # Groq Whisper (빠름, 무료 등급)
VOICE_TOOLS_OPENAI_KEY=...         # OpenAI Whisper (유료)

# STT 고급 재정의 (선택 사항)
STT_GROQ_MODEL=whisper-large-v3-turbo    # 기본 Groq STT 모델 재정의
STT_OPENAI_MODEL=whisper-1               # 기본 OpenAI STT 모델 재정의
GROQ_BASE_URL=https://api.groq.com/openai/v1     # 사용자 지정 Groq 엔드포인트
STT_OPENAI_BASE_URL=https://api.openai.com/v1    # 사용자 지정 OpenAI STT 엔드포인트

# Text-to-Speech 공급자 (Edge TTS 및 NeuTTS는 키가 필요 없음)
ELEVENLABS_API_KEY=***             # ElevenLabs (프리미엄 품질)
# 위의 VOICE_TOOLS_OPENAI_KEY도 OpenAI TTS를 활성화합니다

# Discord 음성 채널
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=...
```

### STT 공급자 비교

| 공급자 | 모델 | 속도 | 품질 | 비용 | API 키 |
|----------|-------|-------|---------|------|---------|
| **Local** | `base` | 빠름 (CPU/GPU에 따라 다름) | 좋음 | 무료 | 아님 |
| **Local** | `small` | 보통 | 더 좋음 | 무료 | 아님 |
| **Local** | `large-v3` | 느림 | 최고 | 무료 | 아님 |
| **Groq** | `whisper-large-v3-turbo` | 매우 빠름 (~0.5s) | 좋음 | 무료 등급 | 예 |
| **Groq** | `whisper-large-v3` | 빠름 (~1s) | 더 좋음 | 무료 등급 | 예 |
| **OpenAI** | `whisper-1` | 빠름 (~1s) | 좋음 | 유료 | 예 |
| **OpenAI** | `gpt-4o-transcribe` | 보통 (~2s) | 최고 | 유료 | 예 |

공급자 우선순위 (자동 폴백): **local** > **groq** > **openai**

### TTS 공급자 비교

| 공급자 | 품질 | 비용 | 지연 시간(Latency) | 키 필수 |
|----------|---------|------|---------|-------------|
| **Edge TTS** | 좋음 | 무료 | ~1s | 아님 |
| **ElevenLabs** | 뛰어남 | 유료 | ~2s | 예 |
| **OpenAI TTS** | 좋음 | 유료 | ~1.5s | 예 |
| **NeuTTS** | 좋음 | 무료 | CPU/GPU에 따라 다름 | 아님 |

NeuTTS는 위의 `tts.neutts` 구성 블록을 사용합니다.

---

## 문제 해결 (Troubleshooting)

### "No audio device found" (CLI)

PortAudio가 설치되지 않았습니다:

```bash
brew install portaudio    # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

Linux 데스크톱의 Docker 내부에서 Hermes를 실행하는 경우, 컨테이너가 호스트 오디오 소켓에 액세스해야 합니다. PulseAudio/PipeWire 호환 설정을 위해 [Docker 오디오 브릿지](/user-guide/docker#optional-linux-desktop-audio-bridge) 노트를 참조하세요.

### Discord 서버 채널에서 봇이 응답하지 않음

봇은 기본적으로 서버 채널에서 @멘션이 필요합니다. 다음을 확인하세요:

1. `@`를 입력하고, 같은 이름의 **역할(role)**이 아닌 **봇 사용자(bot user)**(#discriminator 포함)를 선택했는지 확인하세요.
2. 또는 멘션이 필요 없는 DM을 사용하세요.
3. 또는 `~/.hermes/.env`에서 `DISCORD_REQUIRE_MENTION=false`를 설정하세요.

### 봇이 VC에 참여했지만 내 말을 듣지 못함

- `DISCORD_ALLOWED_USERS`에 내 Discord 사용자 ID가 있는지 확인하세요.
- Discord에서 음소거되지 않았는지 확인하세요.
- 봇이 오디오를 매핑하기 전에 Discord에서 SPEAKING 이벤트가 필요합니다 — 참여 후 몇 초 안에 말하기 시작하세요.

### 봇이 내 말을 듣지만 응답하지 않음

- STT를 사용할 수 있는지 확인하세요: `faster-whisper` 설치(키 필요 없음) 또는 `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY` 설정.
- LLM 모델이 구성되고 액세스 가능한지 확인하세요.
- 게이트웨이 로그 검토: `tail -f ~/.hermes/logs/gateway.log`

### 봇이 텍스트로는 응답하지만 음성 채널에서는 응답하지 않음

- TTS 공급자가 실패했을 수 있습니다 — API 키와 할당량을 확인하세요.
- Edge TTS(무료, 키 불필요)가 기본 폴백(fallback)입니다.
- 로그에서 TTS 오류를 확인하세요.

### Whisper가 쓰레기(의미 없는) 텍스트를 반환함

환각 필터가 대부분의 경우를 자동으로 잡습니다. 여전히 유령 텍스트가 발생하는 경우:

- 더 조용한 환경을 사용하세요.
- 설정에서 `silence_threshold`를 조정하세요 (높을수록 덜 민감함).
- 다른 STT 모델을 시도해 보세요.
