---
sidebar_position: 8
title: "Hermes와 함께 음성 모드 사용하기 (Use Voice Mode with Hermes)"
description: "CLI, Telegram, Discord 및 Discord 음성 채널 등에서 Hermes 음성 모드를 설정하고 사용하기 위한 실용 가이드"
---

# Hermes와 함께 음성 모드 사용하기 (Use Voice Mode with Hermes)

이 가이드는 [음성 모드 기능 참조 문서](/user-guide/features/voice-mode)의 실용적인 안내서(companion)입니다.

기능 페이지가 음성 모드로 할 수 있는 일들을 설명한다면, 이 가이드는 그것을 실제로 어떻게 잘 활용할 수 있는지를 보여줍니다.

:::tip
[Nous Portal](/integrations/nous-portal)은 LLM과 TTS(텍스트 음성 변환)를 하나의 OAuth를 통해 번들로 제공합니다 — 별도의 자격 증명(credentials) 없이도 음성 모드가 종단간(end-to-end)으로 작동합니다.
:::

## 음성 모드의 장점

음성 모드는 특히 다음과 같은 경우에 유용합니다:
- 코딩 중이거나 손을 쓸 수 없는 핸즈프리 CLI 워크플로우를 원할 때
- Telegram이나 Discord에서 음성 답변을 원할 때
- Hermes를 Discord 음성 채널에 참여시켜 실시간 대화를 나누고 싶을 때
- 타이핑 대신 걸어 다니면서 빠르게 아이디어를 기록하거나 디버깅, 티키타카를 원할 때

## 음성 모드 설정 선택하기

Hermes에는 세 가지 주요한 음성 모드 경험이 있습니다.

| 모드 | 가장 적합한 용도 | 플랫폼 |
|---|---|---|
| 인터랙티브 마이크 루프 | 코딩이나 연구 중 개인적인 핸즈프리 사용 | CLI |
| 채팅 음성 응답 | 일반 메시지와 함께 음성으로 된 응답 | Telegram, Discord |
| 라이브 음성 채널 봇 | VC(음성 채널)에서의 개인 또는 그룹 실시간 대화 | Discord 음성 채널 |

권장하는 진행 순서는 다음과 같습니다:
1. 먼저 텍스트 모드가 작동하도록 설정합니다.
2. 두 번째로 음성 응답(voice replies) 기능을 활성화합니다.
3. 완벽한 경험을 원할 경우 마지막에 Discord 음성 채널 연동으로 넘어갑니다.

## 1단계: 먼저 기본 Hermes가 작동하는지 확인하기

음성 모드를 건드리기 전에 다음 사항을 확인하세요:
- Hermes가 시작됩니다.
- 제공자(provider) 구성이 설정되어 있습니다.
- 에이전트가 텍스트 프롬프트에 정상적으로 답변할 수 있습니다.

```bash
hermes
```

단순한 질문을 해보세요:

```text
어떤 도구들을 사용할 수 있어?
```

만약 이것이 불안정하다면 텍스트 모드부터 먼저 해결하세요.

## 2단계: 알맞은 추가 패키지 설치하기

### CLI 마이크 + 재생

```bash
pip install "hermes-agent[voice]"
```

### 메시징 플랫폼

```bash
pip install "hermes-agent[messaging]"
```

### 프리미엄 ElevenLabs TTS

```bash
pip install "hermes-agent[tts-premium]"
```

### 로컬 NeuTTS (선택 사항)

```bash
python -m pip install -U neutts[all]
```

### 모두 설치

```bash
pip install "hermes-agent[all]"
```

## 3단계: 시스템 의존성 패키지 설치

### macOS

```bash
brew install portaudio ffmpeg opus
brew install espeak-ng
```

### Ubuntu / Debian

```bash
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng
```

이 패키지들이 중요한 이유:
- `portaudio` → CLI 음성 모드를 위한 마이크 입력 및 스피커 재생 지원
- `ffmpeg` → TTS 및 메시징 플랫폼 전송을 위한 오디오 변환 지원
- `opus` → Discord 음성 코덱 지원
- `espeak-ng` → NeuTTS를 위한 음소화(phonemizer) 백엔드 지원

## 4단계: STT 및 TTS 제공자 선택

Hermes는 로컬 및 클라우드 음성 스택을 모두 지원합니다.

### 가장 쉽고 저렴한 설정

로컬 STT와 무료 Edge TTS를 사용합니다:
- STT 제공자: `local`
- TTS 제공자: `edge`

일반적으로 시작하기 가장 좋은 설정입니다.

### 환경 변수 파일 예시

`~/.hermes/.env` 파일에 추가하세요:

```bash
# Cloud STT 옵션 (local의 경우 키가 필요 없습니다)
GROQ_API_KEY=***
VOICE_TOOLS_OPENAI_KEY=***

# Premium TTS (선택 사항)
ELEVENLABS_API_KEY=***
```

### 제공자 권장 사항

#### STT (음성을 텍스트로)

- `local` → 개인정보 보호와 완전 무료(zero-cost)를 위한 최고의 기본값
- `groq` → 매우 빠른 클라우드 전사(transcription)
- `openai` → 좋은 유료 대체 옵션

#### TTS (텍스트를 음성으로)

- `edge` → 무료이며 대부분의 사용자에게 충분히 좋은 품질 제공
- `neutts` → 무료 로컬/온디바이스 TTS
- `elevenlabs` → 최상의 품질 제공
- `openai` → 품질과 가격의 좋은 절충안
- `mistral` → 다국어 지원, 기본 Opus 코덱 지원

### `hermes setup`을 사용하는 경우

설정 마법사에서 NeuTTS를 선택하면 Hermes는 `neutts`가 이미 설치되어 있는지 확인합니다. 패키지가 누락된 경우, 마법사는 NeuTTS에 Python 패키지인 `neutts`와 시스템 패키지인 `espeak-ng`가 필요함을 알리고 대신 설치해 줄지 물어봅니다. 승인 시 사용 중인 OS의 패키지 관리자를 통해 `espeak-ng`를 설치하고 이어서 다음 명령어를 실행합니다:

```bash
python -m pip install -U neutts[all]
```

만약 설치를 건너뛰거나 실패하면, 마법사는 Edge TTS로 물러납니다(fall back).

## 5단계: 권장 구성 설정

```yaml
voice:
  record_key: "ctrl+b"
  max_recording_seconds: 120
  auto_tts: false
  beep_enabled: true
  silence_threshold: 200
  silence_duration: 3.0

stt:
  provider: "local"
  local:
    model: "base"

tts:
  provider: "edge"
  edge:
    voice: "en-US-AriaNeural"
```

위의 설정은 대부분의 사람들에게 적합하고 보수적인 기본값입니다.

로컬 TTS를 원한다면 `tts` 블록을 다음과 같이 변경하세요:

```yaml
tts:
  provider: "neutts"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

## 사용 사례 1: CLI 음성 모드

## 기능 켜기

Hermes를 시작합니다:

```bash
hermes
```

CLI 내부에서 다음 명령을 입력하세요:

```text
/voice on
```

### 녹음 흐름 (Recording flow)

기본 키:
- `Ctrl+B`

작업 흐름(Workflow):
1. `Ctrl+B`를 누릅니다.
2. 말을 합니다.
3. 침묵(silence) 감지 기능이 자동으로 녹음을 멈출 때까지 기다립니다.
4. Hermes가 이를 텍스트로 변환(transcribe)하고 응답합니다.
5. TTS가 켜져 있으면 답변을 소리내어 말해줍니다.
6. 이 루프는 연속적인 사용을 위해 자동으로 다시 시작될 수 있습니다.

### 유용한 명령어

```text
/voice
/voice on
/voice off
/voice tts
/voice status
```

### 추천하는 CLI 워크플로우

#### 즉각적인(Walk-up) 디버깅

말해보세요:

```text
계속 docker permission 오류가 나. 디버깅하는 걸 도와줘.
```

그런 다음 손을 대지 않고 계속 대화를 이어갑니다(hands-free):
- "방금 말한 마지막 오류 다시 읽어줘"
- "근본 원인을 좀 더 쉽게 설명해 봐"
- "이제 정확한 해결책을 줘"

#### 연구 / 브레인스토밍

다음과 같은 상황에 아주 좋습니다:
- 생각을 하면서 걸어 다닐 때
- 절반쯤 떠오른 아이디어를 구술할 때
- 실시간으로 Hermes에게 내 생각을 구조화해 달라고 요청할 때

#### 접근성 / 타이핑을 적게 해야 하는 환경

타이핑이 불편한 상황이라면, 음성 모드는 완전한 Hermes 루프에 머물 수 있는 가장 빠른 방법 중 하나입니다.

## CLI 동작 튜닝하기

### 침묵 임계값 (Silence threshold)

Hermes가 너무 공격적으로 시작/중지를 반복한다면, 다음 설정을 조정하세요:

```yaml
voice:
  silence_threshold: 250
```

임계값이 높을수록 덜 민감해집니다.

### 침묵 지속 시간 (Silence duration)

문장과 문장 사이에 말을 멈추는 시간이 길다면, 다음 설정값을 늘리세요:

```yaml
voice:
  silence_duration: 4.0
```

### 녹음 키 변경 (Record key)

`Ctrl+B`가 사용 중인 터미널이나 tmux 단축키와 겹친다면:

```yaml
voice:
  record_key: "ctrl+space"
```

## 사용 사례 2: Telegram 또는 Discord의 음성 응답

이 모드는 완전한 음성 채널(VC) 모드보다 간단합니다.

Hermes는 정상적인 챗봇으로 머물지만 응답을 음성으로 보내줍니다.

### 게이트웨이 시작

```bash
hermes gateway
```

### 음성 응답 켜기

Telegram이나 Discord 채팅방 내부에서:

```text
/voice on
```

또는

```text
/voice tts
```

### 모드 설명

| 모드 | 의미 |
|---|---|
| `off` | 텍스트만 전송 |
| `voice_only` | 사용자가 음성 메시지를 보냈을 때만 음성으로 응답 |
| `all` | 모든 응답을 음성으로 전송 |

### 언제 어떤 모드를 사용해야 할까요?

- 음성으로 보낸 메시지에 대해서만 음성 응답을 원한다면 `/voice on`을 사용하세요.
- 언제나 완전히 소리 내어 말해주는 어시스턴트를 원한다면 `/voice tts`를 사용하세요.

### 추천하는 메시징 워크플로우

#### 휴대전화 속 Telegram 어시스턴트

다음에 유용합니다:
- 컴퓨터에서 떨어져 있을 때
- 음성 메시지를 보내고 빠른 음성 응답을 받고 싶을 때
- Hermes를 휴대용 연구 및 운영 보조 수단으로 활용하고 싶을 때

#### 음성 출력을 지원하는 Discord DM

서버 채널의 멘션 동작 없이 개인적인 상호작용을 원할 때 유용합니다.

## 사용 사례 3: Discord 음성 채널

가장 고급 모드입니다.

Hermes가 Discord 음성 채널(VC)에 합류하여 사용자의 음성을 듣고, 전사(transcribe)하며, 일반적인 에이전트 파이프라인을 실행하고, 해당 채널에 응답을 음성으로 돌려줍니다.

## 필요한 Discord 권한

일반적인 텍스트 봇 설정에 더하여, 봇이 다음 권한을 가지는지 확인하세요:
- Connect (연결)
- Speak (말하기)
- (권장) Use Voice Activity (음성 감지 사용)

또한 Developer Portal에서 권한 있는 인텐트(privileged intents)를 활성화해야 합니다:
- Presence Intent
- Server Members Intent
- Message Content Intent

## 합류 및 퇴장

봇이 있는 Discord 텍스트 채널에서 입력하세요:

```text
/voice join
/voice leave
/voice status
```

### 합류 시 일어나는 일

- 사용자가 VC에서 말합니다.
- Hermes가 음성(말)의 시작과 끝을 감지합니다.
- 대화 기록(transcript)이 관련 텍스트 채널에 게시됩니다.
- Hermes가 텍스트와 오디오로 응답합니다.
- 여기서 텍스트 채널이란 `/voice join` 명령어를 입력했던 채널을 뜻합니다.

### Discord VC 사용 모범 사례

- `DISCORD_ALLOWED_USERS`를 엄격하게 관리하세요.
- 처음에는 봇/테스트 전용 채널을 사용하세요.
- VC 모드를 시도하기 전에, STT와 TTS가 일반 텍스트-채팅 기반의 음성 모드에서 정상 작동하는지 먼저 확인하세요.

## 음성 품질 권장 사항

### 최고 품질 설정

- STT: 로컬 `large-v3` 또는 Groq `whisper-large-v3`
- TTS: ElevenLabs

### 속도 / 편의성 위주 설정

- STT: 로컬 `base` 또는 Groq
- TTS: Edge

### 비용 제로(무료) 설정

- STT: 로컬
- TTS: Edge

## 일반적인 실패 유형 (Common failure modes)

### "오디오 기기를 찾을 수 없습니다 (No audio device found)"

`portaudio`를 설치하세요.

### "봇이 참가하긴 했는데 아무것도 듣지 못합니다 (Bot joins but hears nothing)"

다음을 확인하세요:
- Discord 사용자 ID가 `DISCORD_ALLOWED_USERS`에 있는지
- 본인이 마이크를 음소거(mute)하지 않았는지
- privileged intents(권한 있는 인텐트)가 활성화되어 있는지
- 봇이 Connect/Speak 권한을 가지고 있는지

### "글로는 변환하는데 음성으로 말을 하지 않습니다 (It transcribes but does not speak)"

다음을 확인하세요:
- TTS 제공자(provider) 설정
- ElevenLabs 또는 OpenAI의 API 키 / 사용 가능 할당량(quota)
- Edge 변환 경로에 필요한 `ffmpeg` 설치 여부

### "Whisper가 의미 없는 쓰레기 값을 출력합니다 (Whisper outputs garbage)"

다음을 시도하세요:
- 주변 환경을 조용하게 만들기
- `silence_threshold` 값 높이기
- STT 제공자/모델 변경
- 짧고 또렷하게 말하기

### "DM에서는 작동하는데 서버 채널에서는 작동하지 않습니다 (It works in DMs but not in server channels)"

종종 멘션 정책(mention policy)의 문제입니다.

설정을 변경하지 않았다면, 기본적으로 봇은 Discord 서버 텍스트 채널에서 자신을 지칭하는 `@mention`을 요구합니다.

## 권장하는 첫 주차 설정 (Suggested first-week setup)

가장 빠르고 확실하게 성공하려면 다음을 따르세요:

1. 텍스트 환경에서 Hermes가 정상 작동하게 합니다.
2. `hermes-agent[voice]` 패키지를 설치합니다.
3. 로컬 STT + Edge TTS로 구성하여 CLI 음성 모드를 사용해 봅니다.
4. 그 다음 Telegram이나 Discord에서 `/voice on`을 활성화해 봅니다.
5. 그 모든 것이 잘 된 후에야 Discord VC 모드를 시도해 보세요.

이러한 단계적 진행은 디버깅 범위를 좁게 유지해 줍니다.

## 다음에 읽을 문서 (Where to read next)

- [음성 모드 기능 참조 문서 (Voice Mode feature reference)](/user-guide/features/voice-mode)
- [메시징 게이트웨이 (Messaging Gateway)](/user-guide/messaging)
- [Discord 설정](/user-guide/messaging/discord)
- [Telegram 설정](/user-guide/messaging/telegram)
- [설정 문서 (Configuration)](/user-guide/configuration)
