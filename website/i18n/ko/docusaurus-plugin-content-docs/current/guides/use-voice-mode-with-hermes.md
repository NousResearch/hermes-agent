---
sidebar_position: 8
title: "Hermes에서 음성 모드 사용"
description: "CLI, Telegram, Discord 및 Discord 음성 채널에서 Hermes 음성 모드를 설정하고 사용하는 실용 가이드"
---

# Hermes에서 음성 모드 사용

이 가이드는 [음성 모드 기능 참조](/user-guide/features/voice-mode)의 실용적인 보충 자료입니다.

기능 페이지에서 음성 모드로 무엇을 할 수 있는지 설명한다면, 이 가이드에서는 실제로 잘 사용하는 방법을 보여줍니다.

:::tip
[Nous Portal](/integrations/nous-portal)은 하나의 OAuth를 통해 LLM과 TTS를 모두 제공합니다. 따라서 추가 자격 증명 없이 음성 모드를 처음부터 끝까지 사용할 수 있습니다.
:::

## 음성 모드가 유용한 경우

음성 모드는 특히 다음과 같은 경우에 유용합니다:
- 핸즈프리 CLI 워크플로를 원할 때
- Telegram 또는 Discord에서 음성 응답을 원할 때
- 실시간 대화를 위해 Discord 음성 채널에 Hermes를 대기시키고 싶을 때
- 입력 대신 걸어 다니면서 빠르게 아이디어를 기록하거나, 디버깅하거나, 대화를 주고받고 싶을 때

## 음성 모드 설정 선택

Hermes에는 실제로 세 가지 서로 다른 음성 경험이 있습니다.

| 모드 | 적합한 용도 | 플랫폼 |
|---|---|---|
| 대화형 마이크 루프 | 코딩이나 조사 중 개인적인 핸즈프리 사용 | CLI |
| 채팅 음성 응답 | 일반 메시징과 함께 음성 응답 사용 | Telegram, Discord |
| 실시간 음성 채널 봇 | VC에서 그룹 또는 개인 실시간 대화 | Discord 음성 채널 |

좋은 진행 순서는 다음과 같습니다:
1. 먼저 텍스트가 작동하도록 설정
2. 두 번째로 음성 응답 활성화
3. 전체 경험을 원한다면 마지막으로 Discord 음성 채널로 이동

## 1단계: 먼저 일반 Hermes가 작동하는지 확인

음성 모드를 설정하기 전에 다음을 확인하세요:
- Hermes가 시작됨
- 프로바이더가 구성됨
- 에이전트가 일반 텍스트 프롬프트에 정상적으로 응답함

```bash
hermes
```

간단한 질문을 해 보세요:

```text
What tools do you have available?
```

아직 안정적으로 작동하지 않는다면 먼저 텍스트 모드를 해결하세요.

## 2단계: 적절한 추가 패키지 설치

### CLI 마이크 + 재생

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[voice]"
```

### 메시징 플랫폼

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"
```

### 프리미엄 ElevenLabs TTS

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[tts-premium]"
```

### 로컬 NeuTTS(선택 사항)

```bash
python -m pip install -U neutts[all]
```

### 전체 설치

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[all]"
```

## 3단계: 시스템 종속성 설치

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

이 항목들이 중요한 이유:
- `portaudio` → CLI 음성 모드의 마이크 입력 / 재생
- `ffmpeg` → TTS 및 메시징 전송을 위한 오디오 변환
- `opus` → Discord 음성 코덱 지원
- `espeak-ng` → NeuTTS용 음소화 백엔드

## 4단계: STT 및 TTS 프로바이더 선택

Hermes는 로컬 및 클라우드 음성 스택을 모두 지원합니다.

### 가장 쉽고 저렴한 설정

로컬 STT와 무료 Edge TTS를 사용하세요:
- STT 프로바이더: `local`
- TTS 프로바이더: `edge`

대개 여기서 시작하는 것이 가장 좋습니다.

### 환경 파일 예시

`~/.hermes/.env`에 추가하세요:

```bash
# Cloud STT options (local needs no key)
GROQ_API_KEY=***
VOICE_TOOLS_OPENAI_KEY=***

# Premium TTS (optional)
ELEVENLABS_API_KEY=***
```

### 프로바이더 권장 사항

#### 음성-텍스트 변환

- `local` → 개인정보 보호와 무료 사용을 위한 최적의 기본값
- `groq` → 매우 빠른 클라우드 전사
- `openai` → 좋은 유료 대안

#### 텍스트-음성 변환

- `edge` → 무료이며 대부분의 사용자에게 충분한 품질
- `neutts` → 무료 로컬/온디바이스 TTS
- `elevenlabs` → 최고의 품질
- `openai` → 품질과 비용의 좋은 중간 지점
- `mistral` → 다국어 지원, 네이티브 Opus

### `hermes setup`을 사용하는 경우

설정 마법사에서 NeuTTS를 선택하면 Hermes는 `neutts`가 이미 설치되어 있는지 확인합니다. 설치되어 있지 않으면 마법사가 NeuTTS에 Python 패키지 `neutts`와 시스템 패키지 `espeak-ng`가 필요하다고 알려주고, 설치 여부를 묻습니다. 사용자를 대신해 플랫폼 패키지 관리자로 `espeak-ng`를 설치한 다음 다음 명령을 실행합니다:

```bash
python -m pip install -U neutts[all]
```

설치를 건너뛰거나 실패하면 마법사는 Edge TTS로 대체합니다.

## 5단계: 권장 구성

```yaml
voice:
  record_key: "ctrl+b"
  submit_mode: "direct"  # TUI: direct | draft
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

대부분의 사용자에게 적합한 보수적인 기본값입니다.

TUI에서는 `voice.submit_mode`가 전사 후 동작을 제어합니다:

- `direct`(기본값)는 전사된 텍스트를 즉시 제출합니다.
- `draft`는 전사된 텍스트를 컴포저에 넣어 Enter를 누르기 전에 편집하거나 취소할 수 있게 합니다.

편집 가능한 음성 초안을 사용하려면 다음과 같이 설정하세요:

```yaml
voice:
  submit_mode: "draft"
```

로컬 TTS를 사용하려면 `tts` 블록을 다음과 같이 변경하세요:

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

## 켜기

Hermes를 시작하세요:

```bash
hermes
```

CLI 내부에서:

```text
/voice on
```

### 녹음 흐름

기본 키:
- `Ctrl+B`

워크플로:
1. `Ctrl+B`를 누릅니다
2. 말합니다
3. 무음 감지가 녹음을 자동으로 멈출 때까지 기다립니다
4. Hermes가 전사하고 응답합니다
5. TTS가 켜져 있으면 답변을 음성으로 읽습니다
6. 연속 사용을 위해 루프가 자동으로 다시 시작될 수 있습니다

### 유용한 명령

```text
/voice
/voice on
/voice off
/voice tts
/voice status
```

### 유용한 CLI 워크플로

#### 걸어 다니며 디버깅

다음과 같이 말합니다:

```text
I keep getting a docker permission error. Help me debug it.
```

이후에도 핸즈프리로 계속할 수 있습니다:
- "마지막 오류를 다시 읽어 줘"
- "근본 원인을 더 쉬운 말로 설명해 줘"
- "이제 정확한 해결 방법을 알려 줘"

#### 조사 / 브레인스토밍

다음과 같은 경우에 좋습니다:
- 생각하며 걸어 다닐 때
- 완성되지 않은 아이디어를 받아 적을 때
- Hermes에게 실시간으로 생각을 정리해 달라고 요청할 때

#### 접근성 / 타이핑이 적은 세션

타이핑이 불편하다면 음성 모드는 전체 Hermes 루프를 계속 사용하는 가장 빠른 방법 중 하나입니다.

## CLI 동작 조정

### 무음 임계값

Hermes가 너무 민감하게 시작하거나 멈추면 다음을 조정하세요:

```yaml
voice:
  silence_threshold: 250
```

값이 높을수록 민감도가 낮아집니다.

### 무음 지속 시간

문장 사이에 자주 멈춘다면 다음 값을 늘리세요:

```yaml
voice:
  silence_duration: 4.0
```

### 녹음 키

`Ctrl+B`가 터미널 또는 tmux 사용 습관과 충돌한다면:

```yaml
voice:
  record_key: "ctrl+space"
```

## 사용 사례 2: Telegram 또는 Discord의 음성 응답

이 모드는 전체 음성 채널보다 간단합니다.

Hermes는 일반 채팅 봇으로 유지되지만 응답을 음성으로 읽을 수 있습니다.

### 게이트웨이 시작

```bash
hermes gateway
```

### 음성 응답 켜기

Telegram 또는 Discord 내부에서:

```text
/voice on
```

또는

```text
/voice tts
```

### 모드

| 모드 | 의미 |
|---|---|
| `off` | 텍스트만 |
| `voice_only` | 사용자가 음성을 보낸 경우에만 음성으로 말함 |
| `all` | 모든 응답을 음성으로 말함 |

### 어떤 모드를 사용할지

- 음성으로 보낸 메시지에 대해서만 음성 응답을 원하면 `/voice on`
- 항상 음성으로 말하는 어시스턴트를 원하면 `/voice tts`

### 유용한 메시징 워크플로

#### 휴대폰의 Telegram 어시스턴트

다음과 같은 경우에 사용하세요:
- 컴퓨터에서 떨어져 있을 때
- 음성 메모를 보내고 빠른 음성 응답을 받고 싶을 때
- Hermes를 휴대용 조사 또는 운영 어시스턴트처럼 사용하고 싶을 때

#### 음성 출력이 있는 Discord DM

서버 채널의 멘션 동작 없이 개인적으로 상호작용하고 싶을 때 유용합니다.

## 사용 사례 3: Discord 음성 채널

가장 고급 모드입니다.

Hermes가 Discord VC에 참여하고, 사용자의 음성을 듣고, 전사하고, 일반 에이전트 파이프라인을 실행한 다음, 채널에 음성으로 답변합니다.

## 필요한 Discord 권한

일반 텍스트 봇 설정 외에도 봇에 다음 권한이 있는지 확인하세요:
- Connect
- Speak
- 가능하면 Use Voice Activity

또한 Developer Portal에서 권한이 필요한 인텐트를 활성화하세요:
- Presence Intent
- Server Members Intent
- Message Content Intent

## 참여 및 나가기

봇이 있는 Discord 텍스트 채널에서:

```text
/voice join
/voice leave
/voice status
```

### 참여하면 일어나는 일

- 사용자가 VC에서 말함
- Hermes가 음성 경계를 감지함
- 전사 내용이 연결된 텍스트 채널에 게시됨
- Hermes가 텍스트와 오디오로 응답함
- 텍스트 채널은 `/voice join`을 실행한 채널임

### Discord VC 사용 모범 사례

- `DISCORD_ALLOWED_USERS`를 엄격하게 설정하세요
- 처음에는 전용 봇/테스트 채널을 사용하세요
- VC 모드를 시도하기 전에 일반 텍스트 채팅 음성 모드에서 STT와 TTS가 작동하는지 확인하세요

## 음성 품질 권장 사항

### 최고의 품질 설정

- STT: 로컬 `large-v3` 또는 Groq `whisper-large-v3`
- TTS: ElevenLabs

### 최고의 속도 / 편의성 설정

- STT: 로컬 `base` 또는 Groq
- TTS: Edge

### 최고의 무료 설정

- STT: local
- TTS: Edge

## 일반적인 실패 유형

### "오디오 장치를 찾을 수 없음"

`portaudio`를 설치하세요.

### "봇은 참여하지만 아무것도 듣지 못함"

다음을 확인하세요:
- Discord 사용자 ID가 `DISCORD_ALLOWED_USERS`에 있음
- 음소거 상태가 아님
- 권한이 필요한 인텐트가 활성화됨
- 봇에 Connect/Speak 권한이 있음

### "전사는 되지만 말하지 않음"

다음을 확인하세요:
- TTS 프로바이더 구성
- ElevenLabs 또는 OpenAI의 API 키 / 할당량
- Edge 변환 경로에 `ffmpeg`가 설치되어 있음

### "Whisper 출력이 엉망임"

다음을 시도하세요:
- 더 조용한 환경
- 더 높은 `silence_threshold`
- 다른 STT 프로바이더/모델
- 더 짧고 명확한 발화

### "DM에서는 작동하지만 서버 채널에서는 작동하지 않음"

대개 멘션 정책 때문입니다.

기본적으로 Discord 서버 텍스트 채널에서는 별도로 설정하지 않는 한 봇에 `@mention`을 포함해야 합니다.

## 첫 주 권장 설정

가장 빠르게 성공하고 싶다면:

1. 텍스트 Hermes를 작동시킵니다
2. `hermes setup tts`를 실행해 음성 지원을 활성화합니다
3. 로컬 STT + Edge TTS로 CLI 음성 모드를 사용합니다
4. 그런 다음 Telegram 또는 Discord에서 `/voice on`을 활성화합니다
5. 그 다음에 Discord VC 모드를 시도합니다

이 순서를 따르면 디버깅 범위를 작게 유지할 수 있습니다.

## 다음 읽을 문서

- [음성 모드 기능 참조](/user-guide/features/voice-mode)
- [메시징 게이트웨이](/user-guide/messaging)
- [Discord 설정](/user-guide/messaging/discord)
- [Telegram 설정](/user-guide/messaging/telegram)
- [구성](/user-guide/configuration)
