---
sidebar_position: 9
title: "음성 및 TTS (Voice & TTS)"
description: "모든 플랫폼에 걸친 텍스트 음성 변환(Text-to-speech) 및 음성 메시지 전사(transcription)"
---

# 음성 및 TTS (Voice & TTS)

Hermes Agent는 모든 메시징 플랫폼에서 텍스트 음성 변환(TTS) 출력과 음성 메시지 전사(STT)를 모두 지원합니다.

:::tip Nous 구독자
유료 [Nous Portal](https://portal.nousresearch.com)을 구독하는 경우, 별도의 OpenAI API 키 없이 **[Tool Gateway](tool-gateway.md)**를 통해 OpenAI TTS를 사용할 수 있습니다. 새로 설치하는 경우 `hermes setup --portal`을 실행하여 로그인하고 모든 게이트웨이 도구를 한 번에 켤 수 있습니다. 기존 설치의 경우 `hermes model` 또는 `hermes tools`를 통해 TTS에 대해서만 **Nous Subscription**을 선택할 수 있습니다.
:::

## 텍스트 음성 변환 (Text-to-Speech)

10개의 공급자를 통해 텍스트를 음성으로 변환하세요:

| 공급자 | 품질 | 비용 | API 키 |
|----------|---------|------|---------|
| **Edge TTS** (기본값) | 좋음 | 무료 | 필요 없음 |
| **ElevenLabs** | 뛰어남 | 유료 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 좋음 | 유료 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax TTS** | 뛰어남 | 유료 | `MINIMAX_API_KEY` |
| **Mistral (Voxtral TTS)** | 뛰어남 | 유료 | `MISTRAL_API_KEY` |
| **Google Gemini TTS** | 뛰어남 | 무료 등급 | `GEMINI_API_KEY` |
| **xAI TTS** | 뛰어남 | 유료 | `XAI_API_KEY` |
| **NeuTTS** | 좋음 | 무료 (로컬) | 필요 없음 |
| **KittenTTS** | 좋음 | 무료 (로컬) | 필요 없음 |
| **Piper** | 좋음 | 무료 (로컬) | 필요 없음 |

### 플랫폼 전송

| 플랫폼 | 전송 방식 | 포맷 |
|----------|----------|--------|
| Telegram | 음성 버블 (인라인 재생) | Opus `.ogg` |
| Discord | 음성 버블 (Opus/OGG), 실패 시 파일 첨부로 폴백 | Opus/MP3 |
| WhatsApp | 오디오 파일 첨부 | MP3 |
| CLI | `~/.hermes/audio_cache/`에 저장됨 | MP3 |

### 구성 (Configuration)

```yaml
# ~/.hermes/config.yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts" | "kittentts" | "piper"
  speed: 1.0                    # 전역 속도 배수 (공급자별 설정이 이를 재정의함)
  edge:
    voice: "en-US-AriaNeural"   # 322개 음성, 74개 언어
    speed: 1.0                  # 비율 퍼센트로 변환됨 (+/-%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # OpenAI 호환 TTS 엔드포인트를 위한 재정의
    speed: 1.0                  # 0.25 - 4.0
  minimax:
    model: "speech-2.8-hd"     # speech-2.8-hd (기본값), speech-2.8-turbo
    voice_id: "English_Graceful_Lady"  # 참조: https://platform.minimax.io/faq/system-voice-id
    speed: 1                    # 0.5 - 2.0
    vol: 1                      # 0 - 10
    pitch: 0                    # -12 - 12
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (기본값)
  gemini:
    model: "gemini-2.5-flash-preview-tts"  # 또는 gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30개의 사전 제작된 음성: Zephyr, Puck, Kore, Enceladus, Gacrux 등.
  xai:
    voice_id: "eve"             # 또는 사용자 지정 음성 ID — 아래 문서 참조
    language: "en"              # ISO 639-1 코드
    sample_rate: 24000          # 22050 / 24000 (기본값) / 44100 / 48000
    bit_rate: 128000            # MP3 비트레이트; codec=mp3일 때만 적용됨
    # base_url: "https://api.x.ai/v1"   # XAI_BASE_URL 환경 변수를 통한 재정의
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
  kittentts:
    model: KittenML/kitten-tts-nano-0.8-int8   # 25MB int8; 기타: kitten-tts-micro-0.8 (41MB), kitten-tts-mini-0.8 (80MB)
    voice: Jasper                               # Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    speed: 1.0                                  # 0.5 - 2.0
    clean_text: true                            # 숫자, 통화, 단위를 확장(읽기 쉽게 변환)
  piper:
    voice: en_US-lessac-medium                  # 음성 이름 (자동 다운로드됨) 또는 .onnx 파일의 절대 경로
    # voices_dir: ''                            # 기본값: ~/.hermes/cache/piper-voices/
    # use_cuda: false                           # onnxruntime-gpu 필요
    # length_scale: 1.0                         # 2.0 = 두 배 느림
    # noise_scale: 0.667
    # noise_w_scale: 0.8
    # volume: 1.0                               # 0.5 = 절반 소리 크기
    # normalize_audio: true
```

**속도 제어**: 전역 `tts.speed` 값은 기본적으로 모든 공급자에게 적용됩니다. 각 공급자는 고유한 `speed` 설정으로 이를 재정의할 수 있습니다(예: `tts.openai.speed: 1.5`). 공급자별 속도 설정은 전역 값보다 우선합니다. 기본값은 `1.0`(보통 속도)입니다.

### 입력 길이 제한

각 공급자에게는 문서화된 요청당 입력 문자 수 제한(cap)이 있습니다. Hermes는 공급자를 호출하기 전에 텍스트를 자르므로 길이 오류로 인해 요청이 실패하는 일이 없습니다:

| 공급자 | 기본 제한 (문자) |
|----------|---------------------|
| Edge TTS | 5000 |
| OpenAI | 4096 |
| xAI | 15000 |
| MiniMax | 10000 |
| Mistral | 4000 |
| Google Gemini | 5000 |
| ElevenLabs | 모델에 따라 다름 (아래 참조) |
| NeuTTS | 2000 |
| KittenTTS | 2000 |
| Piper | 5000 |

**ElevenLabs**는 구성된 `model_id`에서 제한을 가져옵니다:

| `model_id` | 제한 (문자) |
|------------|-------------|
| `eleven_flash_v2_5` | 40000 |
| `eleven_flash_v2` | 30000 |
| `eleven_multilingual_v2` (기본값), `eleven_multilingual_v1`, `eleven_english_sts_v2`, `eleven_english_sts_v1` | 10000 |
| `eleven_v3`, `eleven_ttv_v3` | 5000 |
| 알 수 없는 모델 | 공급자 기본값(10000)으로 폴백 |

**공급자별 재정의**: TTS 구성의 공급자 섹션 아래에 `max_text_length:`를 설정하여 재정의할 수 있습니다.

```yaml
tts:
  openai:
    max_text_length: 8192   # 공급자 제한을 높이거나 낮춤
```

양의 정수만 적용됩니다. 0, 음수, 숫자가 아닌 값, 또는 불리언(boolean) 값은 공급자 기본값으로 폴백되므로, 잘못된 구성으로 인해 실수로 텍스트 자르기(truncation) 기능이 비활성화되는 일이 없습니다.

### Telegram 음성 버블 및 ffmpeg

Telegram 음성 버블에는 Opus/OGG 오디오 포맷이 필요합니다:

- **OpenAI, ElevenLabs, 및 Mistral**은 기본적으로 Opus를 생성합니다 — 추가 설정 필요 없음
- **Edge TTS** (기본값)는 MP3를 출력하며 변환을 위해 **ffmpeg**가 필요합니다:
- **MiniMax TTS**는 MP3를 출력하며 Telegram 음성 버블을 위한 변환에 **ffmpeg**가 필요합니다.
- **Google Gemini TTS**는 raw PCM을 출력하며 Telegram 음성 버블을 위해 **ffmpeg**를 사용하여 Opus로 직접 인코딩합니다.
- **xAI TTS**는 MP3를 출력하며 Telegram 음성 버블을 위한 변환에 **ffmpeg**가 필요합니다.
- **NeuTTS**는 WAV를 출력하며 Telegram 음성 버블을 위한 변환에 또한 **ffmpeg**가 필요합니다.
- **KittenTTS**는 WAV를 출력하며 Telegram 음성 버블을 위한 변환에 또한 **ffmpeg**가 필요합니다.
- **Piper**는 WAV를 출력하며 Telegram 음성 버블을 위한 변환에 또한 **ffmpeg**가 필요합니다.

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

ffmpeg가 없으면 Edge TTS, MiniMax TTS, NeuTTS, KittenTTS, 그리고 Piper 오디오는 일반 오디오 파일로 전송됩니다 (재생은 가능하지만, 음성 버블 대신 직사각형 플레이어로 표시됨).

:::tip
ffmpeg를 설치하지 않고 음성 버블을 원한다면 OpenAI, ElevenLabs, 또는 Mistral 공급자로 전환하세요.
:::

### xAI 사용자 지정 음성 (음성 복제)

xAI는 음성을 복제하여 TTS와 함께 사용하는 것을 지원합니다. [xAI Console](https://console.x.ai/team/default/voice/voice-library)에서 사용자 지정 음성을 만든 다음, 구성 파일에 생성된 `voice_id`를 설정하세요:

```yaml
tts:
  provider: xai
  xai:
    voice_id: "nlbqfwie"   # 사용자 지정 음성 ID
```

녹음, 지원되는 포맷 및 제한에 대한 자세한 내용은 [xAI 사용자 지정 음성 문서(Custom Voices docs)](https://docs.x.ai/developers/model-capabilities/audio/custom-voices)를 참조하세요.

### Piper (로컬, 44개 언어)

Piper는 Open Home Foundation(Home Assistant 유지 관리자)에서 만든 빠르고 로컬 환경에서 실행되는 신경망(neural) TTS 엔진입니다. 이는 전적으로 CPU에서 실행되며, 사전 훈련된 음성이 있는 **44개 언어**를 지원하고 API 키가 필요하지 않습니다.

**`hermes tools`를 통해 설치:** 음성 및 TTS (Voice & TTS) → Piper를 선택하세요 — Hermes가 사용자를 위해 `pip install piper-tts`를 실행합니다. 또는 수동으로 설치하세요: `pip install piper-tts`.

**Piper로 전환:**

```yaml
tts:
  provider: piper
  piper:
    voice: en_US-lessac-medium
```

로컬에 캐시되지 않은 음성에 대한 첫 번째 TTS 호출에서 Hermes는 `python -m piper.download_voices <name>`을 실행하고 모델(품질 등급에 따라 ~20-90MB)을 `~/.hermes/cache/piper-voices/`에 다운로드합니다. 이후의 호출은 캐시된 모델을 재사용합니다.

**음성 선택하기:** [전체 음성 카탈로그](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md)는 영어, 스페인어, 프랑스어, 독일어, 이탈리아어, 네덜란드어, 포르투갈어, 러시아어, 폴란드어, 튀르키예어, 중국어, 아랍어, 힌디어 등을 지원합니다 — 각각 `x_low` / `low` / `medium` / `high` 품질 등급이 있습니다. 샘플 음성은 [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/)에서 들을 수 있습니다.

**미리 다운로드된 음성 사용하기:** `tts.piper.voice`를 `.onnx`로 끝나는 절대 경로로 설정하세요:

```yaml
tts:
  piper:
    voice: /path/to/my-custom-voice.onnx
```

**고급 설정** (`tts.piper.length_scale` / `noise_scale` / `noise_w_scale` / `volume` / `normalize_audio`, `use_cuda`)은 Piper의 `SynthesisConfig`와 1:1로 대응됩니다. 구버전의 `piper-tts`에서는 무시됩니다.

### 사용자 지정 명령 공급자 (Custom command providers)

원하는 TTS 엔진이 기본적으로 지원되지 않는 경우 (VoxCPM, MLX-Kokoro, XTTS CLI, 음성 복제 스크립트 등 CLI를 노출하는 모든 것), Python 코드를 작성하지 않고도 **명령어 유형 공급자(command-type provider)**로 연결할 수 있습니다. Hermes는 입력 텍스트를 임시 UTF-8 파일에 쓰고, 쉘 명령어를 실행한 다음, 명령어가 생성한 오디오 파일을 읽습니다.

`tts.providers.<name>` 아래에 하나 이상의 공급자를 선언하고 `tts.provider: <name>`으로 전환하세요 — `edge`나 `openai`와 같은 내장(built-in) 제공자들 사이를 전환하는 것과 동일한 방식입니다.

```yaml
tts:
  provider: voxcpm                 # tts.providers 아래의 아무 이름이나 선택하세요
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      timeout: 180
      voice_compatible: true       # Telegram 음성 버블로 전송하려고 시도합니다.

    mlx-kokoro:
      type: command
      command: "python -m mlx_kokoro --in {input_path} --out {output_path} --voice {voice}"
      voice: af_sky
      output_format: wav

    piper-custom:                  # 네이티브 Piper는 tts.piper.voice를 통해 사용자 지정 .onnx도 지원합니다.
      type: command
      command: "piper -m /path/to/custom.onnx -f {output_path} < {input_path}"
      output_format: wav
```

#### 예시: Doubao (Chinese seed-tts-2.0)

ByteDance의 [seed-tts-2.0](https://www.volcengine.com/docs/6561/1257544) 양방향 스트리밍 API를 통한 고품질 중국어 TTS를 사용하려면, [`doubao-speech`](https://pypi.org/project/doubao-speech/) PyPI 패키지를 설치하고 명령 공급자로 연결하세요:

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
```

```yaml
tts:
  provider: doubao
  providers:
    doubao:
      type: command
      command: "doubao-speech say --text-file {input_path} --out {output_path}"
      output_format: mp3
      max_text_length: 1024
      timeout: 30
```

자격 증명은 쉘 환경(`VOLCENGINE_APP_ID` / `VOLCENGINE_ACCESS_TOKEN`) 또는 `~/.doubao-speech/config.yaml`에서 가져옵니다. 명령어에 `--voice zh-female-warm` (또는 `doubao-speech list-voices`에서 나온 다른 별칭)을 추가하여 음성을 선택하세요. `doubao-speech`는 스트리밍 ASR도 포함하고 있습니다 — Hermes 연동은 [아래의 STT 섹션](#example-doubao--volcengine-asr)을 참조하세요. 소스 및 전체 문서는 [github.com/Hypnus-Yuan/doubao-speech](https://github.com/Hypnus-Yuan/doubao-speech)에 있습니다.

#### 플레이스홀더 (Placeholders)

명령 템플릿은 이러한 플레이스홀더를 참조할 수 있습니다. Hermes는 렌더링 시점에 이들을 치환하며 주변 컨텍스트 (따옴표 없음 / 작은따옴표 / 큰따옴표)에 맞게 각 값을 쉘에서 안전하게 이스케이프(shell-quotes)하므로, 공백 및 쉘에 민감한 다른 문자가 포함된 경로도 안전합니다.

| 플레이스홀더      | 의미                                              |
|------------------|------------------------------------------------------|
| `{input_path}`   | Hermes가 작성한 임시 UTF-8 텍스트 파일 경로        |
| `{text_path}`    | `{input_path}`의 별칭                             |
| `{output_path}`  | 명령어가 오디오를 기록해야 하는 경로                 |
| `{format}`       | `mp3` / `wav` / `ogg` / `flac`                       |
| `{voice}`        | `tts.providers.<name>.voice`, 설정되지 않은 경우 비어 있음       |
| `{model}`        | `tts.providers.<name>.model`                         |
| `{speed}`        | 결정된 속도 배수 (공급자별 또는 전역 속도)       |

리터럴 중괄호(`{`, `}`)를 사용하려면 `{{` 및 `}}`를 사용하세요.

#### 선택적 키 (Optional keys)

| 키                | 기본값 | 의미                                                                                                    |
|--------------------|---------|------------------------------------------------------------------------------------------------------------|
| `timeout`          | `120`   | 초 단위; 만료 시 프로세스 트리가 종료됩니다 (Unix `killpg`, Windows `taskkill /T`).                       |
| `output_format`    | `mp3`   | `mp3` / `wav` / `ogg` / `flac` 중 하나. Hermes가 경로를 선택하는 경우 출력 확장자에서 자동으로 유추됩니다.      |
| `voice_compatible` | `false` | `true`인 경우, Telegram이 음성 버블을 렌더링할 수 있도록 Hermes가 ffmpeg를 통해 MP3/WAV 출력을 Opus/OGG로 변환합니다.      |
| `max_text_length`  | `5000`  | 명령을 렌더링하기 전에 입력 텍스트가 이 길이로 잘립니다.                                             |
| `voice` / `model`  | 비어 있음   | 오직 명령에 플레이스홀더 값으로 전달되기 위해서만 존재합니다.                                                           |

#### 동작에 관한 참고 사항

- **내장(Built-in) 이름이 항상 우선합니다.** `tts.providers.openai` 항목은 네이티브 OpenAI 공급자를 가리지(shadow) 못하므로 사용자 설정이 기본 내장 공급자를 몰래 대체할 수 없습니다.
- **기본 전송 방식은 문서 첨부입니다.** 명령 공급자는 모든 플랫폼에서 일반 오디오 파일 첨부로 전송합니다. 공급자별로 음성 버블 전송을 활성화하려면 `voice_compatible: true`로 설정하세요.
- **명령 실패는 에이전트에게 표출됩니다.** 0이 아닌 종료 코드, 빈 출력, 또는 시간 초과 등은 명령의 stderr/stdout을 포함한 오류를 반환하여 대화창에서 제공자의 오류를 디버깅할 수 있게 합니다.
- **`command:`가 설정되면 `type: command`가 기본값입니다.** `type: command`를 명시적으로 작성하는 것이 좋은 관행이지만 필수는 아닙니다. 비어 있지 않은 `command` 문자열이 있는 항목은 명령 공급자로 처리됩니다.
- **`{input_path}`와 `{text_path}`는 상호 교환이 가능합니다.** 명령에서 읽기 좋은 것을 사용하세요.

#### 보안

명령 유형 공급자는 귀하의 사용자 권한으로 귀하가 설정한 어떠한 쉘 명령어든 실행합니다. Hermes는 플레이스홀더 값을 따옴표로 묶어 이스케이프하고 구성된 시간 제한을 적용하지만, 명령 템플릿 자체는 신뢰할 수 있는 로컬 입력입니다 — PATH에 있는 쉘 스크립트와 동일하게 취급하세요.

### Python 플러그인 공급자 (Python plugin providers)

단일 쉘 명령으로 표현할 수 없는 TTS 엔진(CLI가 없는 Python SDK, 스트리밍 엔진, 음성 목록 API, OAuth 기반 인증 등)의 경우, `ctx.register_tts_provider()`를 통해 Python 플러그인을 등록할 수 있습니다. 플러그인은 [사용자 지정 명령 공급자](#custom-command-providers) 레지스트리를 대체하지 않고 **공존**합니다. 엔진에 맞는 방식을 선택하세요.

#### 어떤 것을 선택할 것인가

| 백엔드가 다음과 같을 때… | 사용할 것 |
|---|---|
| 파일/stdin에서 텍스트를 읽고 파일/stdout에 오디오를 쓰는 단일 CLI | **명령 공급자 (Command provider)** (Python 필요 없음) |
| 쉘 파이프로 연결된 2~3개의 CLI | **명령 공급자 (Command provider)** |
| CLI가 없는 Python SDK | **플러그인 (Plugin)** |
| 청크 단위로 전송하려는 스트리밍 바이트 (생성 중간 음성 버블) | **플러그인 (Plugin)** (`stream()` 재정의) |
| `hermes setup`에서 사용되는 음성 목록 API | **플러그인 (Plugin)** (`list_voices()` 재정의) |
| OAuth 토큰 갱신 플로우 (고정된 bearer 토큰이 아님) | **플러그인 (Plugin)** |

기본 내장 공급자는 항상 우선하며, 명령 공급자는 동일한 이름의 플러그인보다 우선합니다. 따라서 기존 구성을 덮어쓸 염려 없이 기본 내장 공급자가 아닌 어떤 이름으로든 플러그인을 등록하는 것은 안전합니다.

#### 최소 플러그인 예제

이것을 `~/.hermes/plugins/my-tts/`에 넣으세요:

`plugin.yaml`:
```yaml
name: my-tts
version: 0.1.0
description: "나의 사용자 지정 Python TTS 백엔드"
```

`__init__.py`:
```python
from agent.tts_provider import TTSProvider


class MyTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "my-tts"  # tts.provider가 매칭하는 이름

    @property
    def display_name(self) -> str:
        return "My Custom TTS"

    def is_available(self) -> bool:
        # 자격 증명/종속성이 누락된 경우 False를 반환 — 선택기(picker)는
        # 이 행을 건너뛰지만 디스패처는 명시적인 설정이 있으면 이리로 라우팅합니다.
        import os
        return bool(os.environ.get("MY_TTS_API_KEY"))

    def synthesize(self, text, output_path, *, voice=None, model=None,
                   speed=None, format="mp3", **extra) -> str:
        # 오디오 바이트를 output_path에 쓰고 경로를 반환합니다.
        # 실패 시 예외 발생 — 디스패처가 예외를 표준 오류 봉투(envelope)로 변환합니다.
        import my_tts_sdk
        client = my_tts_sdk.Client()
        audio_bytes = client.synthesize(text=text, voice=voice or "default")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return output_path


def register(ctx):
    ctx.register_tts_provider(MyTTSProvider())
```

이를 활성화하고 (`hermes plugins enable my-tts`), `tts.provider`가 이를 가리키도록 설정하면 (`config.yaml`에 `tts.provider: my-tts`), `text_to_speech` 도구가 플러그인을 통해 라우팅됩니다.

#### 선택적 훅 (Optional hooks)

더 풍부한 연동을 위해 공급자 클래스에서 다음을 재정의하세요:

- `list_voices()` → `hermes tools`에 표시될 `{id, display, language, gender, preview_url}` 딕셔너리 목록.
- `list_models()` → `{id, display, languages, max_text_length}` 딕셔너리 목록.
- `get_setup_schema()` → `hermes tools` / `hermes setup`의 선택기 행을 구동하기 위한 `{name, badge, tag, env_vars: [{key, prompt, url}]}` 반환. 이것이 없어도 플러그인은 작동하지만 선택기의 항목은 최소한으로 표시됩니다.
- `stream(text, *, voice, model, format, **extra)` → 스트리밍 전송을 위해 오디오 바이트를 생성하는 반복자(iterator) (기본값은 `NotImplementedError`를 발생시킵니다).
- `voice_compatible` 속성(property) → 출력이 Opus와 호환되고 게이트웨이가 이를 음성 버블로 전송해야 하는 경우 `True`로 설정합니다 (기본값 `False` = 일반 오디오 첨부).

독스트링(docstrings)을 포함한 전체 ABC(추상 기본 클래스)는 `agent/tts_provider.py`를 참조하세요.

## 음성 메시지 전사 (Voice Message Transcription, STT)

Telegram, Discord, WhatsApp, Slack, Signal에서 보낸 음성 메시지는 자동으로 전사되어 텍스트로 대화에 주입됩니다. 에이전트는 전사된 내용을 일반 텍스트로 봅니다.

| 공급자 | 품질 | 비용 | API 키 |
|----------|---------|------|---------| 
| **Local Whisper** (기본값) | 좋음 | 무료 | 필요 없음 |
| **Groq Whisper API** | 좋음~최고 | 무료 등급 | `GROQ_API_KEY` |
| **OpenAI Whisper API** | 좋음~최고 | 유료 | `VOICE_TOOLS_OPENAI_KEY` 또는 `OPENAI_API_KEY` |

:::info Zero Config (무설정)
로컬 전사는 `faster-whisper`가 설치되어 있으면 별도의 설정 없이 바로 작동합니다. 만약 사용할 수 없다면 Hermes는 일반적인 설치 경로(`/opt/homebrew/bin` 등)에 있는 로컬 `whisper` CLI나 `HERMES_LOCAL_STT_COMMAND`를 통한 사용자 지정 명령을 사용할 수도 있습니다.
:::

### 구성 (Configuration)

```yaml
# ~/.hermes/config.yaml
stt:
  provider: "local"           # "local" | "groq" | "openai" | "mistral" | "xai"
  local:
    model: "base"             # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"        # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe
  mistral:
    model: "voxtral-mini-latest"  # voxtral-mini-latest, voxtral-mini-2602
  xai:
    model: "grok-stt"         # xAI Grok STT
```

### 공급자 세부 정보

**로컬 (faster-whisper)** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)를 통해 로컬에서 Whisper를 실행합니다. 기본적으로 CPU를 사용하며 가능한 경우 GPU를 사용합니다. 모델 크기:

| 모델 | 크기 | 속도 | 품질 |
|-------|------|-------|---------|
| `tiny` | ~75 MB | 가장 빠름 | 기본 |
| `base` | ~150 MB | 빠름 | 좋음 (기본값) |
| `small` | ~500 MB | 보통 | 더 좋음 |
| `medium` | ~1.5 GB | 느림 | 훌륭함 |
| `large-v3` | ~3 GB | 가장 느림 | 최고 |

**Groq API** — `GROQ_API_KEY`가 필요합니다. 호스팅되는 무료 STT 옵션이 필요할 때 좋은 클라우드 폴백입니다.

**OpenAI API** — `VOICE_TOOLS_OPENAI_KEY`를 먼저 허용하고 `OPENAI_API_KEY`로 폴백합니다. `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`를 지원합니다.

**Mistral API (Voxtral Transcribe)** — `MISTRAL_API_KEY`가 필요합니다. Mistral의 [Voxtral Transcribe](https://docs.mistral.ai/capabilities/audio/speech_to_text/) 모델을 사용합니다. 13개 언어, 화자 분리(speaker diarization), 단어 수준 타임스탬프를 지원합니다. `pip install hermes-agent[mistral]`로 설치하세요.

**xAI Grok STT** — `XAI_API_KEY`가 필요합니다. `https://api.x.ai/v1/stt`에 multipart/form-data로 포스트합니다. 이미 대화나 TTS에 xAI를 사용 중이고 모든 것에 대해 하나의 API 키를 원할 때 좋은 선택입니다. 자동 감지 순서에서는 Groq 뒤에 위치하므로 강제로 사용하려면 명시적으로 `stt.provider: xai`를 설정하세요.

**사용자 지정 로컬 CLI 폴백** — Hermes가 로컬 전사 명령을 직접 호출하도록 하려면 `HERMES_LOCAL_STT_COMMAND`를 설정하세요. 명령 템플릿은 `{input_path}`, `{output_dir}`, `{language}`, `{model}` 플레이스홀더를 지원합니다. 당신의 명령어는 `{output_dir}` 아래 어딘가에 `.txt` 전사본을 생성해야 합니다.

#### 예시: Doubao / Volcengine ASR

만약 Doubao TTS를 위해 [`doubao-speech`](https://pypi.org/project/doubao-speech/)를 사용한다면 (위의 [중국어 seed-tts-2.0 섹션](#example-doubao-chinese-seed-tts-20) 참조), 동일한 패키지가 로컬-명령 STT 표면(surface)을 통해 음성-텍스트 변환(Speech-to-text)도 처리합니다:

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
export HERMES_LOCAL_STT_COMMAND='doubao-speech transcribe {input_path} --out {output_dir}/transcript.txt'
```

```yaml
stt:
  provider: local_command
```

Hermes는 들어오는 음성 메시지를 `{input_path}`에 쓰고, 명령어를 실행한 다음, `{output_dir}` 아래 생성된 `.txt` 파일을 읽습니다. 언어는 Volcengine 빅모델 엔드포인트에 의해 자동 감지됩니다.

### 폴백 동작 (Fallback Behavior)

설정된 공급자를 사용할 수 없는 경우, Hermes는 자동으로 폴백(fallback)합니다:
- **로컬 faster-whisper 사용 불가** → 클라우드 공급자 이전에 로컬 `whisper` CLI 또는 `HERMES_LOCAL_STT_COMMAND`를 시도합니다.
- **Groq 키 미설정** → 로컬 전사로 폴백한 다음, OpenAI로 이동합니다.
- **OpenAI 키 미설정** → 로컬 전사로 폴백한 다음, Groq로 이동합니다.
- **Mistral 키/SDK 미설정** → 자동 감지에서 건너뛰고, 사용 가능한 다음 공급자로 넘어갑니다.
- **사용 가능한 공급자 없음** → 사용자에게 정확한 알림과 함께 음성 메시지가 그대로 통과됩니다.

### STT 사용자 지정 명령 공급자

원하는 STT 엔진이 기본적으로 지원되지 않는 경우(Doubao ASR, NVIDIA Parakeet, whisper.cpp 빌드, 오픈소스 SenseVoice CLI 등 쉘 명령어를 노출하는 모든 것), Python 코드를 작성하지 않고도 **명령어 유형 공급자(command-type provider)**로 연결할 수 있습니다. Hermes는 오디오 파일에 쉘 명령을 실행하고 전사 결과를 읽어들입니다.

`stt.providers.<name>` 아래에 하나 이상의 제공자를 선언하고 `stt.provider: <name>`으로 전환하세요. 이는 TTS의 [명령어 공급자 레지스트리](#custom-command-providers)와 같은 형태이며 입력=오디오 → 출력=텍스트 방향에 맞게 조정되었습니다.

```yaml
stt:
  provider: parakeet                # stt.providers 아래의 어떤 이름이든 선택하세요
  providers:
    parakeet:
      type: command
      command: "parakeet-asr --model nvidia/parakeet-tdt-0.6b-v2 --in {input_path} --out {output_path}"
      format: txt
      language: en
      timeout: 300

    whispercpp:
      type: command
      command: "whisper-cli -m ~/models/ggml-large-v3.bin -f {input_path} -otxt -of {output_dir}/transcript"
      format: txt

    sensevoice:
      type: command
      command: "sensevoice-cli {input_path} --json | tee {output_path}"
      format: json
```

이것은 기존의 `HERMES_LOCAL_STT_COMMAND` 탈출구(escape hatch)를 보완합니다. 이 환경 변수는 내장된 `local_command` 경로를 통해 온전하게 계속 작동합니다. **여러 개의** 쉘 구동 STT 엔진이 필요하거나, `stt.provider`를 통해 이름을 지정하여 선택하고 싶거나, 제공자별로 `language` / `model` / `timeout` 설정이 필요한 경우에는 `stt.providers.<name>`을 사용하세요.

#### STT 플레이스홀더

명령 템플릿은 이러한 플레이스홀더를 참조할 수 있습니다. Hermes는 렌더링 시점에 이들을 치환하며 주변 컨텍스트 (따옴표 없음 / 작은따옴표 / 큰따옴표)에 맞게 각 값을 쉘에서 안전하게 이스케이프하므로 공백이 포함된 경로도 안전합니다.

| 플레이스홀더       | 의미                                                              |
|-------------------|----------------------------------------------------------------------|
| `{input_path}`    | 입력 오디오 파일의 절대 경로 (원본 위치, 읽기 전용) |
| `{output_path}`   | 명령어가 전사 내용을 써야 하는 절대 경로             |
| `{output_dir}`    | `{output_path}`의 상위 디렉터리 (whisper 스타일 도구에 유용함)  |
| `{format}`        | 설정된 출력 포맷: `txt` / `json` / `srt` / `vtt`             |
| `{language}`      | 구성된 언어 코드 (기본값은 `en`)                          |
| `{model}`         | `stt.providers.<name>.model`, 설정되지 않은 경우 비어 있음                       |

리터럴 중괄호(`{`, `}`)를 사용하려면 `{{` 및 `}}`를 사용하세요 (명령어에 JSON 조각을 포함할 때 유용합니다).

#### 전사 내용(transcript)을 읽어오는 방식

명령어가 성공적으로 종료된 후:

1. `{output_path}`가 존재하고 비어 있지 않은 경우 → Hermes가 이를 UTF-8 텍스트로 읽습니다.
2. 그렇지 않고, 명령어가 stdout에 출력을 기록한 경우 → Hermes는 그것을 사용합니다.
3. 그 외의 경우 → 에러 발생: "명령 STT 공급자가 어떤 출력 파일도 쓰지 않았고 stdout 출력도 생성하지 않았습니다".

이를 통해 파일 쓰기 기반의 CLI(`whisper-cli`, `parakeet-asr`)와 stdout으로 결과를 내보내는 curl 스타일의 단일 행 명령어(`curl … | jq -r .text`) 모두에 이 레지스트리를 사용할 수 있습니다.

`format: json` / `srt` / `vtt`의 경우, Hermes는 `transcript` 필드에 원시 파일 내용(raw file content)을 반환합니다. JSON에서 `.text`를 추출하는 것은 실행기(runner)의 범위를 벗어납니다 — `format: txt`로 설정하거나 다운스트림에서 JSON을 후처리하세요.

#### STT 명령 공급자 선택적 키

| 키             | 기본값 | 의미                                                                                              |
|-----------------|---------|------------------------------------------------------------------------------------------------------|
| `timeout`       | `300`   | 초 단위; 만료 시 프로세스 트리가 종료됩니다 (Unix `start_new_session`, Windows `taskkill /T`).     |
| `format`        | `txt`   | `txt` / `json` / `srt` / `vtt` 중 하나. `{output_path}`의 확장자를 결정합니다.                       |
| `language`      | `en`    | `{language}`에 전달됩니다. `stt.language`를 기본값으로 하며 없으면 `en`이 됩니다.                                     |
| `model`         | 비어 있음   | `{model}`에 전달됩니다. `transcribe_audio()`에 전달된 `model=` 인자가 이것을 덮어씁니다.                |

#### STT 명령 공급자 동작에 관한 참고 사항

- **내장(Built-in) 이름이 항상 우선합니다.** `stt.providers.openai: type: command`를 선언하더라도 네이티브 OpenAI Whisper 핸들러를 대체(override)하지 않습니다. 내장 이름은 명령 제공자 리졸버가 실행되기 전에 처리(short-circuit)됩니다.
- **프로세스 트리 정리.** `timeout`을 초과하여 실행되는 명령은 쉘 래퍼(wrapper)만이 아니라 그 전체 프로세스 트리가 종료됩니다. 백그라운드 프로세스에서 모델을 로딩하는 시간이 오래 걸리는 ASR 파이프라인도 안정적으로 정리됩니다.
- **쉘 이스케이프는 자동입니다.** `'…'` 내부의 플레이스홀더는 작은따옴표에서 안전한 이스케이프를 얻고, `"…"` 내부는 `$`/`` ` ``/`"` 이스케이프를 얻으며, 따옴표 외부는 `shlex.quote` 처리가 됩니다. 플레이스홀더 값을 미리 따옴표로 감싸지 마세요.

#### STT 명령 공급자 보안

쉘 명령어는 완전한 파일 시스템 접근 권한을 가지고 Hermes와 동일한 사용자로 실행됩니다 — `tts.providers.<name>: type: command` 및 `HERMES_LOCAL_STT_COMMAND`와 동일한 신뢰 모델을 가집니다. 신뢰할 수 있는 출처의 명령 공급자만 선언하세요.

### Python 플러그인 공급자 (STT)

내장되지 않았으면서 쉘 명령으로도 표현할 수 없는 STT 엔진(Python SDK 필요, OAuth 갱신 인증, 스트리밍 청크 등)의 경우 `ctx.register_transcription_provider()`를 통해 Python 플러그인을 등록할 수 있습니다. 플러그인은 6개의 내장 공급자(`local`, `local_command`, `groq`, `openai`, `mistral`, `xai`)와 `stt.providers.<name>: type: command` 레지스트리와 **공존**합니다 — 내장 공급자들은 네이티브 구현을 유지하며 이름 충돌 시 항상 승리합니다. 명령 공급자는 동일한 이름의 플러그인보다 승리합니다(구성이 플러그인 설치보다 더 로컬이기 때문입니다).

#### 언제 무엇을 선택할 것인가 (STT)

| 백엔드가 다음과 같을 때…                                                 | 사용할 것                                                              |
|--------------------------------------------------------------|------------------------------------------------------------------|
| 오디오 파일을 받아 텍스트를 내뱉는 단일 쉘 명령 | `stt.providers.<name>: type: command` (Python 불필요)        |
| 오직 레거시 단일 명령 탈출구(escape hatch)만 원할 때        | `HERMES_LOCAL_STT_COMMAND` 환경 변수 (하위 호환성을 위해 유지됨)  |
| CLI가 없는 Python SDK                                     | `register_transcription_provider()` 플러그인                      |
| OAuth 갱신 인증, 스트리밍 청크, 음성 목록 메타데이터 | `register_transcription_provider()` 플러그인                      |
| 내장 기능이 이미 커버하는 경우 (`local`, `groq`, `openai` 등)  | `stt.provider: <name>` 설정 — 내장 기능 사용               |

#### 해결(Resolution) 순서

1. **`stt.provider`가 내장(built-in) 이름인 경우** → 내장 기능 사용. **항상 우선.**
2. **`stt.provider`가 `command:`가 설정된 `stt.providers.<name>`과 일치하는 경우** → 명령 공급자 실행기(runner) (참조: [STT 사용자 지정 명령 공급자](#stt-custom-command-providers)). 동일한 이름의 플러그인보다 우선.
3. **`stt.provider`가 플러그인으로 등록된 `TranscriptionProvider`와 일치하는 경우** → 플러그인 호출:
   - 만약 플러그인의 `is_available()`이 `False`를 반환하면(자격 증명이나 SDK 누락), 호출은 일반적인 "STT 공급자를 사용할 수 없음" 메시지가 **아닌** 해당 플러그인을 식별하는 에러(unavailability error envelope)를 표출합니다.
   - 그렇지 않은 경우, 플러그인의 `transcribe()`가 `model` (공개된 `model=` 인자에서 오며 `stt.<provider>.model`로 폴백됨)과 `language` (`stt.<provider>.language`에서 옴) 인자와 함께 호출됩니다.
4. **일치하는 항목 없음** → "No STT provider available(STT 공급자를 사용할 수 없음)" 에러 발생.

#### 공급자별 구성 네임스페이스

플러그인은 `config.yaml`의 `stt.<provider>`에서 공급자별 구성을 읽습니다. 내장 공급자가 `stt.openai.model` / `stt.mistral.model`을 읽는 방식과 동일합니다:

```yaml
stt:
  provider: my-stt
  my-stt:
    model: whisper-large-v3
    language: ja          # transcribe()에 language=로 전달됨
    # 플러그인별 고유 키는 여기에 위치합니다; 플러그인의
    # __init__/is_available/transcribe에서 config.yaml을 통해 이들을 읽으세요
```

디스패처는 이 섹션에서 `model`과 `language`를 전달하며, 나머지 설정은 플러그인에서 직접 읽을 수 있습니다.

#### 최소 플러그인 예제

이것을 `~/.hermes/plugins/my-stt/`에 넣으세요:

`plugin.yaml`:
```yaml
name: my-stt
version: 0.1.0
description: "나의 사용자 지정 Python STT 백엔드"
```

`__init__.py`:
```python
from agent.transcription_provider import TranscriptionProvider


class MySTTProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "my-stt"  # stt.provider가 매칭하는 이름

    @property
    def display_name(self) -> str:
        return "My Custom STT"

    def is_available(self) -> bool:
        # 자격 증명/종속성이 누락된 경우 False를 반환 — 선택기(picker)는
        # 이 행을 건너뛰지만 디스패처는 명시적인 설정이 있으면 이리로 라우팅합니다.
        import os
        return bool(os.environ.get("MY_STT_API_KEY"))

    def transcribe(self, file_path, *, model=None, language=None, **extra):
        # 표준 전사 봉투(envelope) 반환:
        #   {"success": bool, "transcript": str, "provider": str, "error": str}
        # 예외(Exception)를 발생시키지 마세요 — 게이트웨이/CLI 호출자가
        # 실패 시에도 일관된 형태를 볼 수 있도록 예외를 에러 봉투로 변환하세요.
        try:
            import my_stt_sdk
            client = my_stt_sdk.Client()
            text = client.transcribe(open(file_path, "rb"))
            return {
                "success": True,
                "transcript": text,
                "provider": "my-stt",
            }
        except Exception as exc:
            return {
                "success": False,
                "transcript": "",
                "error": f"my-stt failed: {exc}",
                "provider": "my-stt",
            }


def register(ctx):
    ctx.register_transcription_provider(MySTTProvider())
```

이를 활성화하고 (`hermes plugins enable my-stt`), `config.yaml`에 `stt.provider: my-stt`를 설정하면 음성 메시지 전사가 당신의 플러그인을 거쳐갑니다.

#### 선택적 훅 (Optional hooks)

더 풍부한 연동을 위해 공급자 클래스에서 다음을 재정의하세요:

- `list_models()` → `{id, display, languages, max_audio_seconds}` 딕셔너리 목록 반환.
- `default_model()` → 사용자가 모델을 덮어쓰지 않을 때 반환되는 문자열.
- `get_setup_schema()` → `hermes tools` / `hermes setup`에 있는 선택기(picker)의 행을 구동하기 위한 `{name, badge, tag, env_vars: [{key, prompt, url}]}` 반환. (STT용 선택기 카테고리는 아직 제공되지 않지만, 플러그인들의 상위 호환성(forward compatibility)을 위해 이 메타데이터는 사용할 수 있게 되어 있습니다.)

독스트링(docstrings)을 포함한 전체 ABC(추상 기본 클래스)는 `agent/transcription_provider.py`를 참조하세요.
