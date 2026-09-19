---
sidebar_position: 9
title: "음성 및 TTS"
description: "모든 플랫폼에서의 텍스트 음성 변환 및 음성 메시지 전사"
---

# 음성 및 TTS

Hermes Agent는 모든 메시징 플랫폼에서 텍스트 음성 변환 출력과 음성 메시지 전사를 모두 지원합니다.

:::tip Nous 구독자
유료 [Nous Portal](https://portal.nousresearch.com) 구독이 있다면 별도의 OpenAI API 키 없이 **[Tool Gateway](tool-gateway.md)**를 통해 OpenAI TTS를 사용할 수 있습니다. 새로 설치한 경우 `hermes setup --portal`을 실행해 로그인하고 모든 게이트웨이 도구를 한 번에 활성화할 수 있으며, 기존 설치에서는 `hermes model` 또는 `hermes tools`에서 TTS에만 **Nous Subscription**을 선택할 수 있습니다.
:::

## 텍스트 음성 변환

11개 프로바이더를 사용해 텍스트를 음성으로 변환할 수 있습니다.

| 프로바이더 | 품질 | 비용 | API 키 |
|----------|---------|------|---------|
| **Edge TTS** (기본값) | 양호 | 무료 | 필요 없음 |
| **ElevenLabs** | 우수 | 유료 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 양호 | 유료 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax TTS** | 우수 | 유료 | `MINIMAX_API_KEY` 또는 `MINIMAX_CN_API_KEY` |
| **Mistral (Voxtral TTS)** | 우수 | 유료 | `MISTRAL_API_KEY` |
| **Google Gemini TTS** | 우수 | 무료 등급 | `GEMINI_API_KEY` |
| **xAI TTS** | 우수 | 유료 | `XAI_API_KEY` |
| **DeepInfra TTS** | 양호 | 유료 | `DEEPINFRA_API_KEY` |
| **NeuTTS** | 양호 | 무료(로컬) | 필요 없음 |
| **KittenTTS** | 양호 | 무료(로컬) | 필요 없음 |
| **Piper** | 양호 | 무료(로컬) | 필요 없음 |

### 플랫폼별 전송

| 플랫폼 | 전송 방식 | 형식 |
|----------|----------|--------|
| Telegram | 음성 말풍선(인라인 재생) | Opus `.ogg` |
| Discord | 음성 말풍선(Opus/OGG), 파일 첨부로 대체 가능 | Opus/MP3 |
| WhatsApp | 오디오 파일 첨부 | MP3 |
| CLI | `~/.hermes/audio_cache/`에 저장 | MP3 |

### 구성

```yaml
# In ~/.hermes/config.yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "deepinfra" | "neutts" | "kittentts" | "piper"
  speed: 1.0                    # Global speed multiplier (provider-specific settings override this)
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
    speed: 1.0                  # Converted to rate percentage (+/-%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # Override for OpenAI-compatible TTS endpoints
    speed: 1.0                  # 0.25 - 4.0
    # language: "es"            # Sent as lang_code — only for OpenAI-compatible endpoints that support it (e.g. Kokoro)
  minimax:
    region: "global"           # "global" or "cn"; see selection rules below
    model: "speech-02-hd"     # speech-02-hd (default), speech-02-turbo
    voice_id: "English_expressive_narrator"  # See https://platform.minimax.io/faq/system-voice-id
    speed: 1                    # 0.5 - 2.0
    vol: 1                      # 0 - 10
    pitch: 0                    # -12 - 12
    # base_url: "https://tts.example/v1/t2a_v2"  # Optional endpoint override for the selected region
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (default)
  gemini:
    model: "gemini-2.5-flash-preview-tts"  # or gemini-3.1-flash-tts-preview
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, Gacrux, etc.
    audio_tags: false           # Enable hidden Gemini 3.1 TTS audio-tag insertion
    persona_prompt_file: ""      # Optional Markdown/text file with Gemini voice direction
  xai:
    voice_id: "eve"             # or a custom voice ID — see docs below
    language: "en"              # BCP-47 code (e.g. "en", "pt-BR") or "auto" for detection
    speed: 1.0                  # 0.7–1.5, playback speed (default: 1.0)
    auto_speech_tags: false     # insert expressive audio tags via LLM rewrite
    text_normalization: false   # normalize numbers/abbreviations/symbols to spoken form
    optimize_streaming_latency: 0  # 0–2, trades quality for lower latency (default: 0)
    sample_rate: 24000          # 22050 / 24000 (default) / 44100 / 48000
    bit_rate: 128000            # MP3 bitrate; only applies when codec=mp3
    # base_url: "https://api.x.ai/v1"   # Override via XAI_BASE_URL env var
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
  kittentts:
    model: KittenML/kitten-tts-nano-0.8-int8   # 25MB int8; also: kitten-tts-micro-0.8 (41MB), kitten-tts-mini-0.8 (80MB)
    voice: Jasper                               # Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    speed: 1.0                                  # 0.5 - 2.0
    clean_text: true                            # Expand numbers, currencies, units
  piper:
    voice: en_US-lessac-medium                  # voice name (auto-downloaded) OR absolute path to .onnx
    # voices_dir: ''                            # default: ~/.hermes/cache/piper-voices/
    # use_cuda: false                           # requires onnxruntime-gpu
    # length_scale: 1.0                         # 2.0 = twice as slow
    # noise_scale: 0.667
    # noise_w_scale: 0.8
    # volume: 1.0                               # 0.5 = half as loud
    # normalize_audio: true
```

MiniMax TTS는 리전, 엔드포인트, 자격 증명을 함께 선택합니다.

- `region: "global"`은 `MINIMAX_API_KEY`와 함께 `https://api.minimax.io/v1/t2a_v2`를 사용합니다.
- `region: "cn"`은 `MINIMAX_CN_API_KEY`와 함께 `https://api.minimaxi.com/v1/t2a_v2`를 사용합니다.
- `region`을 생략하면 하위 호환성을 위해 `MINIMAX_API_KEY`가 우선합니다. `MINIMAX_CN_API_KEY`만 구성된 경우 Hermes는 `cn`을 선택합니다.
- 명시적으로 선택한 리전에는 일치하는 자격 증명이 있어야 합니다. Hermes는 다른 리전의 키를 대신 사용하지 않습니다. `base_url` 재정의는 선택된 자격 증명을 변경하지 않으며, 다른 리전의 공식 엔드포인트를 가리키는 재정의는 거부됩니다.

**속도 제어**: 전역 `tts.speed` 값은 기본적으로 모든 프로바이더에 적용됩니다. 각 프로바이더는 자체 `speed` 설정(예: `tts.openai.speed: 1.5`)으로 이를 재정의할 수 있습니다. 프로바이더별 속도 설정이 전역 값보다 우선합니다. 기본값은 `1.0`(보통 속도)입니다.

### Gemini 페르소나 프롬프트

Gemini TTS는 자연어로 작성된 연기 지시를 따를 수 있습니다. 음성 페르소나를 설명하는 로컬 Markdown 또는 텍스트 파일로 `tts.gemini.persona_prompt_file`을 설정하세요. 파일에는 `AUDIO PROFILE`, `SCENE`, `DIRECTOR'S NOTES`, `SAMPLE CONTEXT`, `TRANSCRIPT`와 같은 Gemini 스타일 섹션을 포함할 수 있습니다.

파일에 `{transcript}` 또는 `{{ transcript }}`가 있으면 Hermes가 해당 플레이스홀더를 실제 TTS 텍스트로 바꿉니다. 그렇지 않으면 Hermes가 레이블이 지정된 `TRANSCRIPT` 섹션을 자동으로 덧붙입니다. 페르소나 프롬프트는 로컬에 유지되며 채팅 답변에는 표시되지 않습니다.

```yaml
tts:
  provider: gemini
  gemini:
    voice: Algieba
    persona_prompt_file: ~/.hermes/tts/butler-voice.md
```

### 오디오 태그(Gemini, xAI)

Google의 Gemini 3.1 Flash TTS와 xAI의 Grok TTS는 `[whispers]`, `[excitedly]`, `[very slow]`, `[laughs]` 및 기타 표현 전달 지시와 같은 자유 형식의 대괄호 오디오 태그를 지원합니다. `tts.gemini.audio_tags` 또는 `tts.xai.auto_speech_tags`를 활성화하면 Hermes가 TTS 전에 숨겨진 재작성 단계를 실행합니다. 재작성 과정에서는 TTS 스크립트에만 인라인 태그를 삽입하며, 화면에 표시되는 채팅 답변은 변경되지 않습니다.

```yaml
tts:
  provider: gemini
  gemini:
    model: gemini-3.1-flash-tts-preview
    audio_tags: true
  xai: 
    auto_speech_tags: true
```

재작성에는 `auxiliary.tts_audio_tags`가 사용되며 기본값은 주 채팅 모델입니다. 태그 삽입을 더 저렴하거나 빠른 모델로 처리하고 싶다면 해당 보조 작업을 재정의하세요.

**언어(OpenAI 호환 엔드포인트)**: `tts.openai.language`는 `lang_code` 요청 매개변수로 엔드포인트에 전달됩니다. 이는 `lang_code`를 지원하는 OpenAI 호환 TTS 서버를 위한 설정입니다. 예를 들어 [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)에서는 `language: "es"`가 영어 기본값 대신 스페인어 음소 변환기를 선택합니다. 이 매개변수를 허용하지 않는 공식 OpenAI API를 사용할 때는 설정하지 않은 상태로 두세요. 설정하지 않으면 추가로 전송되는 값이 없습니다.


### 입력 길이 제한

각 프로바이더에는 요청당 입력 문자 수 제한이 문서화되어 있습니다. Hermes는 프로바이더를 호출하기 전에 더 긴 답변을 순서가 유지되고 문장을 고려한 청크로 나누므로, 조용히 잘리는 대신 정규화된 전체 텍스트가 보존됩니다.

| 프로바이더 | 기본 제한(문자) |
|----------|---------------------|
| Edge TTS | 5000 |
| OpenAI | 4096 |
| xAI | 15000 |
| MiniMax | 10000 |
| Mistral | 4000 |
| Google Gemini | 32000 |
| ElevenLabs | 모델에 따라 다름(아래 참조) |
| NeuTTS | 2000 |
| KittenTTS | 2000 |
| Piper | 5000 |

**ElevenLabs**는 구성된 `model_id`에서 제한을 선택합니다.

| `model_id` | 제한(문자) |
|------------|-------------|
| `eleven_flash_v2_5` | 40000 |
| `eleven_flash_v2` | 30000 |
| `eleven_multilingual_v2` (기본값), `eleven_multilingual_v1`, `eleven_english_sts_v2`, `eleven_english_sts_v1` | 10000 |
| `eleven_v3`, `eleven_ttv_v3` | 5000 |
| 알 수 없는 모델 | 프로바이더 기본값(10000)으로 대체 |

**프로바이더별 재정의**: TTS 구성의 프로바이더 섹션 아래에 `max_text_length:`를 지정하세요.

```yaml
tts:
  openai:
    max_text_length: 8192   # raise or lower the provider cap
```

양의 정수만 적용됩니다. 0, 음수, 숫자가 아닌 값 또는 불리언 값은 프로바이더 기본값으로 처리되므로, 잘못된 구성으로 인해 프로바이더 요청 제한을 실수로 우회할 수 없습니다.

### Telegram 음성 말풍선 및 ffmpeg

Telegram 음성 말풍선에는 Opus/OGG 오디오 형식이 필요합니다.

- **OpenAI, ElevenLabs, Mistral**은 기본적으로 Opus를 생성하므로 추가 설정이 필요하지 않습니다.
- **Edge TTS**(기본값)는 MP3를 출력하므로 변환하려면 **ffmpeg**가 필요합니다.
- **MiniMax TTS**는 MP3를 출력하므로 Telegram 음성 말풍선으로 변환하려면 **ffmpeg**가 필요합니다.
- **Google Gemini TTS**는 원시 PCM을 출력하며 Telegram 음성 말풍선을 위해 **ffmpeg**를 사용해 Opus로 직접 인코딩합니다.
- **xAI TTS**는 MP3를 출력하므로 Telegram 음성 말풍선으로 변환하려면 **ffmpeg**가 필요합니다.
- **NeuTTS**는 WAV를 출력하며 Telegram 음성 말풍선으로 변환하려면 역시 **ffmpeg**가 필요합니다.
- **KittenTTS**는 WAV를 출력하며 Telegram 음성 말풍선으로 변환하려면 역시 **ffmpeg**가 필요합니다.
- **Piper**는 WAV를 출력하며 Telegram 음성 말풍선으로 변환하려면 역시 **ffmpeg**가 필요합니다.

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

ffmpeg가 없으면 Edge TTS, MiniMax TTS, NeuTTS, KittenTTS 및 Piper 오디오는 일반 오디오 파일로 전송됩니다(재생할 수 있지만 음성 말풍선 대신 직사각형 플레이어로 표시됨).

:::tip
ffmpeg를 설치하지 않고 음성 말풍선을 사용하려면 OpenAI, ElevenLabs 또는 Mistral 프로바이더로 전환하세요.
:::

### xAI 사용자 지정 음성(음성 복제)

xAI는 음성을 복제해 TTS에 사용하는 기능을 지원합니다. [xAI Console](https://console.x.ai/team/default/voice/voice-library)에서 사용자 지정 음성을 만든 다음, 생성된 `voice_id`를 구성에 설정하세요.

```yaml
tts:
  provider: xai
  xai:
    voice_id: "nlbqfwie"   # your custom voice ID
```

녹음, 지원 형식 및 제한에 관한 자세한 내용은 [xAI Custom Voices docs](https://docs.x.ai/developers/model-capabilities/audio/custom-voices)를 참조하세요.

### Piper(로컬, 44개 언어)

Piper는 Open Home Foundation(Home Assistant 유지 관리자)이 만든 빠른 로컬 신경망 TTS 엔진입니다. CPU에서 완전히 실행되며, 사전 학습된 음성으로 **44개 언어**를 지원하고 API 키가 필요하지 않습니다.

**`hermes tools`로 설치** → Voice & TTS → Piper — Hermes가 `pip install piper-tts`를 대신 실행합니다. 또는 수동으로 설치할 수 있습니다: `pip install piper-tts`.

**Piper로 전환:**

```yaml
tts:
  provider: piper
  piper:
    voice: en_US-lessac-medium
```

아직 로컬에 캐시되지 않은 음성으로 TTS를 처음 호출하면 Hermes가 `python -m piper.download_voices <name>`을 실행하고 모델을 품질 등급에 따라 약 20~90MB 크기로 `~/.hermes/cache/piper-voices/`에 다운로드합니다. 이후 호출에서는 캐시된 모델을 재사용합니다.

**음성 선택.** [전체 음성 카탈로그](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md)에는 영어, 스페인어, 프랑스어, 독일어, 이탈리아어, 네덜란드어, 포르투갈어, 러시아어, 폴란드어, 터키어, 중국어, 아랍어, 힌디어 등의 음성이 포함되어 있으며, 각 언어에는 `x_low` / `low` / `medium` / `high` 품질 등급이 있습니다. [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/)에서 음성을 미리 들어볼 수 있습니다.

**미리 다운로드한 음성 사용.** `tts.piper.voice`를 `.onnx`로 끝나는 절대 경로로 설정하세요.

```yaml
tts:
  piper:
    voice: /path/to/my-custom-voice.onnx
```

**고급 옵션**(`tts.piper.length_scale` / `noise_scale` / `noise_w_scale` / `volume` / `normalize_audio`, `use_cuda`)은 Piper의 `SynthesisConfig`에 1:1로 대응합니다. 이전 버전의 `piper-tts`에서는 무시됩니다.
### 사용자 지정 명령 제공자

원하는 TTS 엔진이 기본 지원되지 않는 경우(VoxCPM, MLX-Kokoro, XTTS CLI, 음성 복제 스크립트 등 CLI를 제공하는 모든 것), Python을 작성하지 않고도 **command 유형 제공자**로 연결할 수 있습니다. Hermes는 입력 텍스트를 임시 UTF-8 파일에 쓰고, 셸 명령을 실행한 다음, 명령이 생성한 오디오 파일을 읽습니다.

`tts.providers.<name>` 아래에 하나 이상의 제공자를 선언하고 `tts.provider: <name>`으로 제공자를 전환하세요. `edge`, `openai` 같은 기본 제공 기능 간에 전환하는 것과 같은 방식입니다.

```yaml
tts:
  provider: voxcpm                 # pick any name under tts.providers
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      timeout: 180
      voice_compatible: true       # try to deliver as a Telegram voice bubble

    mlx-kokoro:
      type: command
      command: "python -m mlx_kokoro --in {input_path} --out {output_path} --voice {voice}"
      voice: af_sky
      output_format: wav

    piper-custom:                  # native Piper also supports custom .onnx via tts.piper.voice
      type: command
      command: "piper -m /path/to/custom.onnx -f {output_path} < {input_path}"
      output_format: wav
```

**지원되는 `output_format` 값:** `mp3`(기본값), `wav`, `ogg`, `flac`, `m4a`, `aac`, `amr`, `opus`. 명령은 실제로 해당 형식의 파일을 생성해야 합니다(예: `ffmpeg` 사용). Hermes는 선언된 값을 검증하고 그에 맞게 출력 파일 이름을 지정할 뿐입니다. 알 수 없는 값은 `mp3`로 대체됩니다. 선택한 형식은 `{format}` 플레이스홀더를 통해 명령에도 전달됩니다.

**서브프로세스 환경:** 명령 제공자(TTS 및 STT)는 Hermes의 시크릿을 자식 환경에서 제거한 상태로 실행됩니다. 게이트웨이 봇 토큰, LLM 제공자 API 키, 내부 릴레이 자격 증명은 제거되며 `PATH`, `HOME`, 로캘 및 기타 일반 변수는 유지됩니다. 명령 템플릿에 자체 API 키가 환경 변수로 필요하다면(예: `curl` 한 줄 명령), 제공자 설정의 `env_passthrough` 아래에 변수 이름을 나열하세요.

```yaml
tts:
  providers:
    mycloud:
      type: command
      command: 'curl -s -H "Authorization: Bearer $MYCLOUD_API_KEY" ... -o {output_path}'
      env_passthrough: [MYCLOUD_API_KEY]
```


#### 예시: Doubao (중국어 seed-tts-2.0)

ByteDance의 [seed-tts-2.0](https://www.volcengine.com/docs/6561/1257544) 양방향 스트리밍 API를 통해 고품질 중국어 TTS를 사용하려면 [`doubao-speech`](https://pypi.org/project/doubao-speech/) PyPI 패키지를 설치하고 명령 제공자로 연결하세요.

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

자격 증명은 셸 환경(`VOLCENGINE_APP_ID` / `VOLCENGINE_ACCESS_TOKEN`) 또는 `~/.doubao-speech/config.yaml`에서 가져옵니다. 명령에 `--voice zh-female-warm`(또는 `doubao-speech list-voices`에서 확인한 다른 별칭)을 추가하여 음성을 선택하세요. `doubao-speech`에는 스트리밍 ASR도 포함되어 있습니다. Hermes 통합 방법은 [아래 STT 섹션](#example-doubao--volcengine-asr)을 참조하세요. 소스 및 전체 문서는 [github.com/Hypnus-Yuan/doubao-speech](https://github.com/Hypnus-Yuan/doubao-speech)에서 확인할 수 있습니다.

#### 플레이스홀더

명령 템플릿에서 다음 플레이스홀더를 참조할 수 있습니다. Hermes는 렌더링 시 이를 치환하고 각 값을 주변 문맥(일반 / 작은따옴표 / 큰따옴표)에 맞게 셸 인용하므로, 공백이 있는 경로와 기타 셸에서 특수한 문자를 포함한 값도 안전하게 사용할 수 있습니다.

| 플레이스홀더      | 의미                                              |
|------------------|------------------------------------------------------|
| `{input_path}`   | Hermes가 작성한 임시 UTF-8 텍스트 파일의 경로        |
| `{text_path}`    | `{input_path}`의 별칭                             |
| `{output_path}`  | 명령이 오디오를 작성해야 하는 경로                 |
| `{format}`       | `mp3` / `wav` / `ogg` / `flac`                       |
| `{voice}`        | `tts.providers.<name>.voice`, 설정되지 않으면 빈 값       |
| `{model}`        | `tts.providers.<name>.model`                         |
| `{speed}`        | 확인된 속도 배율(제공자 또는 전역)       |

리터럴 중괄호에는 `{{` 및 `}}`를 사용하세요.

#### 선택적 키

| 키                | 기본값 | 의미                                                                                                    |
|--------------------|---------|------------------------------------------------------------------------------------------------------------|
| `timeout`          | `120`   | 유휴 시간(초). stdout 또는 stderr 출력이 제한 시간을 초기화합니다. 비활성 상태가 되면 프로세스 트리가 종료됩니다(Unix `killpg`, Windows `taskkill /T`). |
| `output_format`    | `mp3`   | `mp3` / `wav` / `ogg` / `flac` 중 하나. Hermes가 경로를 선택하면 출력 확장자에서 자동으로 추론됩니다.      |
| `voice_compatible` | `false` | `true`이면 Hermes가 ffmpeg를 통해 MP3/WAV 출력을 Opus/OGG로 변환하여 Telegram에서 음성 말풍선으로 렌더링합니다.      |
| `max_text_length`  | `5000`  | 명령 호출당 최대 입력 문자 수. 더 긴 텍스트는 순서가 유지되는 청크로 분할됩니다.                  |
| `voice` / `model`  | empty   | 플레이스홀더 값으로만 명령에 전달됩니다.                                                           |

#### 동작 참고 사항

- **기본 제공 이름이 항상 우선합니다.** `tts.providers.openai` 항목은 기본 OpenAI 제공자를 가리지 않으므로, 사용자 설정으로 기본 제공 기능을 조용히 대체할 수 없습니다.
- **기본 전달 방식은 문서입니다.** 명령 제공자는 모든 플랫폼에서 일반 오디오 첨부 파일로 전달합니다. 제공자별로 `voice_compatible: true`를 설정하면 음성 말풍선 전달을 사용할 수 있습니다.
- **명령 실패는 에이전트에 표시됩니다.** 0이 아닌 종료 코드, 빈 출력 또는 시간 초과가 발생하면 명령의 stderr/stdout이 포함된 오류가 반환되므로 대화에서 제공자를 디버깅할 수 있습니다.
- **`command:`가 설정되면 `type: command`가 기본값입니다.** `type: command`를 명시적으로 작성하는 것이 좋은 방법이지만 필수는 아닙니다. 비어 있지 않은 `command` 문자열이 있는 항목은 명령 제공자로 처리됩니다.
- **`{input_path}` / `{text_path}`는 서로 바꿔 쓸 수 있습니다.** 명령에서 더 읽기 좋은 쪽을 사용하세요.

#### 보안

명령 유형 제공자는 사용자 권한으로 설정한 셸 명령을 그대로 실행합니다. Hermes는 플레이스홀더 값을 인용하고 설정된 시간 제한을 적용하지만, 명령 템플릿 자체는 신뢰된 로컬 입력입니다. PATH에 있는 셸 스크립트와 동일하게 취급하세요.

### Python 플러그인 제공자

단일 셸 명령으로 표현할 수 없는 TTS 엔진(Python SDK만 제공되고 CLI가 없는 엔진, 스트리밍 엔진, 음성 목록 API, OAuth 갱신 인증 등)의 경우 `ctx.register_tts_provider()`를 통해 Python 플러그인을 등록하세요. 플러그인은 [사용자 지정 명령 제공자](#custom-command-providers) 레지스트리와 **공존하며** 이를 대체하지 않습니다. 엔진에 맞는 표면을 선택하세요.

#### 어떤 것을 선택할지

| 백엔드에 있는 기능… | 사용 |
|---|---|
| 파일/stdin에서 텍스트를 읽고 파일/stdout에 오디오를 쓰는 단일 CLI | **명령 제공자**(Python 불필요) |
| 셸 파이프로 연결된 두세 개의 CLI | **명령 제공자** |
| Python SDK만 제공되고 CLI는 없음 | **플러그인** |
| 청크 단위로 전달하고 싶은 스트리밍 바이트(생성 중 음성 말풍선) | **플러그인**(`stream()` 재정의) |
| `hermes setup`에서 사용하는 음성 목록 API | **플러그인**(`list_voices()` 재정의) |
| OAuth 갱신 흐름(정적 bearer 토큰 아님) | **플러그인** |

기본 제공 기능이 항상 우선하며, 명령 제공자는 같은 이름의 플러그인보다 우선합니다. 따라서 플러그인은 기존 설정을 가릴 걱정 없이 기본 제공 기능이 아닌 어떤 이름으로도 안전하게 등록할 수 있습니다.

#### 최소 플러그인

다음 파일을 `~/.hermes/plugins/my-tts/`에 추가하세요.

`plugin.yaml`:
```yaml
name: my-tts
version: 0.1.0
description: "My custom Python TTS backend"
```

`__init__.py`:
```python
from agent.tts_provider import TTSProvider


class MyTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "my-tts"  # what tts.provider matches against

    @property
    def display_name(self) -> str:
        return "My Custom TTS"

    def is_available(self) -> bool:
        # Return False when credentials/deps are missing — picker skips
        # this row but the dispatcher still routes here on explicit config.
        import os
        return bool(os.environ.get("MY_TTS_API_KEY"))

    def synthesize(self, text, output_path, *, voice=None, model=None,
                   speed=None, format="mp3", **extra) -> str:
        # Write audio bytes to output_path, return the path.
        # Raise on failure — the dispatcher converts exceptions to a
        # standard error envelope.
        import my_tts_sdk
        client = my_tts_sdk.Client()
        audio_bytes = client.synthesize(text=text, voice=voice or "default")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return output_path


def register(ctx):
    ctx.register_tts_provider(MyTTSProvider())
```

이를 활성화하고(`hermes plugins enable my-tts`), `tts.provider`가 플러그인을 가리키도록 설정하면(`config.yaml`의 `tts.provider: my-tts`), `text_to_speech` 도구가 플러그인을 통해 라우팅됩니다.

#### 선택적 훅

더 풍부한 통합을 위해 제공자 클래스에서 다음을 재정의할 수 있습니다.

- `list_voices()` → `hermes tools`에 표시되는 `{id, display, language, gender, preview_url}` 딕셔너리 목록.
- `list_models()` → `{id, display, languages, max_text_length}` 딕셔너리 목록.
- `get_setup_schema()` → `hermes tools` / `hermes setup`의 선택기 행을 구성하는 `{name, badge, tag, env_vars: [{key, prompt, url}]}` 반환. 이 값이 없어도 플러그인은 작동하지만 선택기의 해당 행은 최소 정보만 표시됩니다.
- `stream(text, *, voice, model, format, **extra)` → 스트리밍 전달을 위해 오디오 바이트를 생성하는 이터레이터(기본값은 `NotImplementedError` 발생).
- `voice_compatible` 속성 → 출력이 Opus와 호환되고 게이트웨이가 음성 말풍선으로 전달해야 하면 `True`로 설정합니다(기본값 `False` = 일반 오디오 첨부 파일).

docstring을 포함한 전체 ABC는 `agent/tts_provider.py`를 참조하세요.

## 음성 메시지 전사(STT)

Telegram, Discord, WhatsApp, Slack 또는 Signal로 전송된 음성 메시지는 자동으로 전사되어 텍스트로 대화에 삽입됩니다. 에이전트는 전사 결과를 일반 텍스트로 인식합니다.

| 제공자 | 품질 | 비용 | API 키 |
|----------|---------|------|---------| 
| **로컬 Whisper**(기본값) | 좋음 | 무료 | 필요 없음 |
| **Groq Whisper API** | 좋음–최고 | 무료 요금제 | `GROQ_API_KEY` |
| **OpenAI Whisper API** | 좋음–최고 | 유료 | `VOICE_TOOLS_OPENAI_KEY` 또는 `OPENAI_API_KEY` |

:::info 설정 불필요
`faster-whisper`가 설치되어 있으면 로컬 전사가 바로 작동합니다. 사용할 수 없는 경우 Hermes는 일반적인 설치 위치(예: `/opt/homebrew/bin`)의 로컬 `whisper` CLI 또는 `HERMES_LOCAL_STT_COMMAND`를 통한 사용자 지정 명령도 사용할 수 있습니다.
:::

### 설정

```yaml
# In ~/.hermes/config.yaml
stt:
  provider: "local"           # "local" | "groq" | "openai" | "mistral" | "xai" | "elevenlabs" | "deepinfra"
  language: "en"              # Global language hint applied to every provider unless a per-provider language overrides it; set "" to restore auto-detect
  local:
    model: "base"             # tiny, base, small, medium, large-v3
    language: ""              # optional ISO-639-1 hint; blank = use HERMES_LOCAL_STT_LANGUAGE if set, else auto-detect
  groq:
    language: ""              # optional ISO-639-1 hint; blank = use HERMES_LOCAL_STT_LANGUAGE if set, else auto-detect
  openai:
    model: "whisper-1"        # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe, gpt-transcribe
  mistral:
    model: "voxtral-mini-latest"  # voxtral-mini-latest, voxtral-mini-2602
  xai:
    model: "grok-stt"         # xAI Grok STT
    language: ""              # optional ISO-639-1 hint; blank = use HERMES_LOCAL_STT_LANGUAGE if set, else "en"
```
### 제공자 세부 정보

**로컬(faster-whisper)** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)를 사용해 Whisper를 로컬에서 실행합니다. 기본적으로 CPU를 사용하고, GPU를 사용할 수 있으면 GPU를 사용합니다. 모델 크기:

| 모델 | 크기 | 속도 | 품질 |
|-------|------|------|------|
| `tiny` | 약 75 MB | 가장 빠름 | 기본 |
| `base` | 약 150 MB | 빠름 | 좋음(기본값) |
| `small` | 약 500 MB | 보통 | 더 좋음 |
| `medium` | 약 1.5 GB | 느림 | 매우 좋음 |
| `large-v3` | 약 3 GB | 가장 느림 | 최고 |

**Groq API** — `GROQ_API_KEY`가 필요합니다. 무료 호스팅 STT 옵션을 원할 때 좋은 클라우드 대체 수단입니다. Whisper의 자동 감지를 건너뛰고 이미 알고 있는 언어의 오디오에서 지연 시간을 줄이려면 `stt.groq.language`(또는 전역 `HERMES_LOCAL_STT_LANGUAGE` 환경 변수)를 설정하세요.

**OpenAI API** — 먼저 `VOICE_TOOLS_OPENAI_KEY`를 사용하고, 없으면 `OPENAI_API_KEY`로 대체합니다. `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`, `gpt-transcribe`를 지원합니다.

**Mistral API(Voxtral Transcribe)** — `MISTRAL_API_KEY`가 필요합니다. Mistral의 [Voxtral Transcribe](https://docs.mistral.ai/capabilities/audio/speech_to_text/) 모델을 사용합니다. 13개 언어, 화자 분리, 단어 단위 타임스탬프를 지원합니다. `cd ~/.hermes/hermes-agent && uv pip install -e "[mistral]"`을 실행해 설치하세요.

**xAI Grok STT** — `XAI_API_KEY`가 필요합니다. `multipart/form-data` 형식으로 `https://api.x.ai/v1/stt`에 전송합니다. 채팅이나 TTS에도 이미 xAI를 사용하고 있어 하나의 API 키로 모든 기능을 사용하려는 경우 좋은 선택입니다. 자동 감지 순서에서는 Groq 다음에 배치됩니다. 강제로 사용하려면 `stt.provider: xai`를 명시적으로 설정하세요.

**사용자 지정 로컬 CLI 대체 수단** — Hermes가 로컬 전사 명령을 직접 호출하도록 하려면 `HERMES_LOCAL_STT_COMMAND`를 설정하세요. 명령 템플릿은 `{input_path}`, `{output_dir}`, `{language}`, `{model}` 자리 표시자를 지원합니다. Hermes는 렌더링된 템플릿을 인수 목록으로 토큰화하고 셸 없이 실행하므로 `|`, `>`, `&&`, `;` 같은 연산자는 리터럴 인수로 전달됩니다. 명령은 `{output_dir}` 아래 어딘가에 `.txt` 전사 파일을 작성해야 합니다.

#### 예시: Doubao / Volcengine ASR

Doubao TTS에 [`doubao-speech`](https://pypi.org/project/doubao-speech/)를 사용하는 경우([위의 예시](#example-doubao-chinese-seed-tts-20) 참조), 동일한 패키지가 로컬 명령 STT 인터페이스를 통해 음성-텍스트 변환도 처리합니다.

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
export HERMES_LOCAL_STT_COMMAND='doubao-speech transcribe {input_path} --out {output_dir}/transcript.txt'
```

신뢰할 수 있는 로컬 템플릿에 파이프, 리디렉션 또는 다른 셸 기능이 의도적으로 필요한 경우 셸을 명시적으로 호출하세요. 동적 경로는 셸 프로그램 밖에 두고 위치 인수로 전달하세요.

```bash
export HERMES_LOCAL_STT_COMMAND='sh -c '\''whisper "$1" --output_format txt --output_dir "$2" | tee "$2/whisper.log"'\'' _ {input_path} {output_dir}'
```

Windows에서는 대신 명시적인 `cmd /c` 또는 PowerShell 래퍼를 사용하세요. 명시적 래퍼를 사용하면 셸 해석이 모든 로컬 STT 템플릿의 암묵적 속성이 아니라 구성된 argv의 옵트인 요소가 됩니다.

```yaml
stt:
  provider: local_command
```

Hermes는 수신한 음성 메시지를 `{input_path}`에 기록하고, 명령을 실행한 다음 `{output_dir}` 아래에 생성된 `.txt` 파일을 읽습니다. 언어는 Volcengine 빅모델 엔드포인트가 자동으로 감지합니다.

### 대체 동작

구성한 제공자를 사용할 수 없으면 Hermes가 자동으로 대체합니다.
- **로컬 faster-whisper를 사용할 수 없음** → 클라우드 제공자보다 먼저 로컬 `whisper` CLI 또는 `HERMES_LOCAL_STT_COMMAND`를 시도합니다.
- **Groq 키가 설정되지 않음** → 로컬 전사로 대체한 다음 OpenAI를 시도합니다.
- **OpenAI 키가 설정되지 않음** → 로컬 전사로 대체한 다음 Groq를 시도합니다.
- **Mistral 키/SDK가 설정되지 않음** → 자동 감지에서 건너뛰고 다음으로 사용 가능한 제공자로 넘어갑니다.
- **사용 가능한 것이 없음** → 음성 메시지를 사용자에게 정확한 안내와 함께 그대로 전달합니다.

### STT 사용자 지정 명령 제공자

원하는 STT 엔진이 기본적으로 지원되지 않는 경우(Doubao ASR, NVIDIA Parakeet, whisper.cpp 빌드, 오픈 소스 SenseVoice CLI 또는 셸 명령을 제공하는 그 밖의 모든 것), Python을 작성하지 않고 **명령 유형 제공자**로 연결할 수 있습니다. Hermes는 오디오 파일에 대해 셸 명령을 실행하고 전사 결과를 다시 읽습니다.

`stt.providers.<name>` 아래에 하나 이상의 제공자를 선언하고 `stt.provider: <name>`으로 전환하세요. TTS [명령 제공자 레지스트리](#custom-command-providers)와 동일한 형태이며, 입력=오디오 → 출력=전사 방향에 맞게 조정되었습니다.

```yaml
stt:
  provider: parakeet                # pick any name under stt.providers
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

이는 내장 `local_command` 경로를 통한 기존 `HERMES_LOCAL_STT_COMMAND` 이스케이프 해치를 보완합니다. 셸 기반 명령 제공자 레지스트리와 달리 기존 템플릿은 argv로 토큰화되며 암묵적인 셸 해석 없이 실행됩니다. **여러** 셸 기반 STT 엔진을 사용하거나, `stt.provider`로 선택할 이름이 필요하거나, 제공자별 `language` / `model` / `timeout`이 필요한 경우 `stt.providers.<name>`을 사용하세요.

#### STT 자리 표시자

명령 템플릿에서 다음 자리 표시자를 참조할 수 있습니다. Hermes는 렌더링 시 이를 대체하고, 주변 컨텍스트(일반 / 작은따옴표 / 큰따옴표)에 맞게 각 값을 셸 인용하므로 공백이 포함된 경로도 안전합니다.

| 자리 표시자       | 의미                                                              |
|-------------------|----------------------------------------------------------------------|
| `{input_path}`    | 입력 오디오 파일의 절대 경로(원래 위치, 읽기 전용) |
| `{output_path}`   | 명령이 전사 결과를 기록해야 하는 절대 경로             |
| `{output_dir}`    | `{output_path}`의 상위 디렉터리(whisper 스타일 도구에 유용)  |
| `{format}`        | 구성된 출력 형식: `txt` / `json` / `srt` / `vtt`             |
| `{language}`      | 구성된 언어 코드(`en`이 기본값)                          |
| `{model}`         | `stt.providers.<name>.model`, 설정되지 않으면 빈 값                       |

명령에 JSON 조각을 포함할 때처럼 리터럴 중괄호가 필요하면 `{{` 및 `}}`를 사용하세요.

#### 전사 결과를 다시 읽는 방법

명령이 성공적으로 종료된 후:

1. `{output_path}`가 존재하고 비어 있지 않으면 → Hermes가 UTF-8 텍스트로 읽습니다.
2. 그렇지 않고 명령이 stdout에 기록했다면 → Hermes가 이를 사용합니다.
3. 그렇지 않으면 → 오류: "Command STT provider wrote no output file and produced no stdout"

이를 통해 파일을 작성하는 CLI(`whisper-cli`, `parakeet-asr`)와 전사 결과를 stdout으로 출력하는 curl 스타일 한 줄 명령(`curl … | jq -r .text`) 모두에 레지스트리를 사용할 수 있습니다.

`format: json` / `srt` / `vtt`의 경우 Hermes는 원시 파일 콘텐츠를 `transcript` 필드로 반환합니다. JSON에서 `.text`를 추출하는 것은 러너의 범위가 아닙니다. `format: txt`를 구성하거나, 이후 단계에서 JSON을 후처리하세요.

#### STT 명령 제공자의 선택적 키

| 키             | 기본값 | 의미                                                                                              |
|-----------------|---------|------------------------------------------------------------------------------------------------------|
| `timeout`       | `300`   | 초 단위; 만료 시 프로세스 트리를 종료합니다(Unix `start_new_session`, Windows `taskkill /T`).     |
| `format`        | `txt`   | `txt` / `json` / `srt` / `vtt` 중 하나. `{output_path}`의 확장자를 설정합니다.                       |
| `language`      | `en`    | `{language}`로 전달됩니다. `stt.language`를 따르고, 없으면 `en`을 사용합니다.                                     |
| `model`         | 빈 값   | `{model}`로 전달됩니다. `transcribe_audio()`의 `model=` 인수가 이를 재정의합니다.                |

#### STT 명령 제공자 동작 참고 사항

- **내장 제공자가 항상 우선합니다.** `stt.providers.openai: type: command`를 선언해도 실제 OpenAI Whisper 핸들러를 재정의하지 않습니다. 명령 제공자 리졸버가 실행되기 전에 내장 이름이 먼저 처리됩니다.
- **프로세스 트리 정리.** `timeout`을 초과해 실행 중인 명령은 셸 래퍼뿐 아니라 전체 프로세스 트리가 종료됩니다. 모델 로딩 하위 프로세스를 생성하는 장시간 ASR 파이프라인도 안정적으로 회수됩니다.
- **셸 인용은 자동입니다.** `'…'` 안의 자리 표시자는 작은따옴표에 안전하도록 이스케이프되고, `"…"` 안에서는 `$`/`` ` ``/`"`가 이스케이프되며, 따옴표 밖에서는 `shlex.quote`가 사용됩니다. 자리 표시자 값을 미리 인용하지 마세요.

#### STT 명령 제공자 보안

셸 명령은 Hermes와 동일한 사용자로, 전체 파일 시스템 접근 권한을 사용해 실행됩니다. 이는 `tts.providers.<name>: type: command` 및 `HERMES_LOCAL_STT_COMMAND`와 동일한 신뢰 모델입니다. 신뢰할 수 있는 출처의 명령 제공자만 선언하세요.

### Python 플러그인 제공자(STT)

기본 제공되지 않으며 셸 명령으로도 표현할 수 없는 STT 엔진(Python SDK, OAuth 갱신 인증, 스트리밍 청크 등이 필요한 경우)은 `ctx.register_transcription_provider()`를 통해 Python 플러그인으로 등록하세요. 플러그인은 8개의 기본 제공자(`local`, `local_command`, `groq`, `openai`, `mistral`, `xai`, `elevenlabs`, `deepinfra`) 및 `stt.providers.<name>: type: command` 레지스트리와 **함께 존재**합니다. 기본 제공자는 고유한 구현을 유지하며 이름 충돌 시 항상 우선합니다. 명령 제공자는 같은 이름의 플러그인보다 우선합니다(구성이 플러그인 설치보다 더 로컬에 있기 때문입니다).

#### 어떤 STT를 선택할지

| 백엔드에 있는 것…                                                 | 사용                                                              |
|--------------------------------------------------------------|------------------------------------------------------------------|
| 오디오 파일을 받아 텍스트를 출력하는 단일 셸 명령 | `stt.providers.<name>: type: command` (Python 불필요)        |
| 기존 단일 명령 이스케이프 해치만 필요함        | `HERMES_LOCAL_STT_COMMAND` 환경 변수(argv 토큰화, 암묵적 셸 없음) |
| CLI가 없는 Python SDK                                     | `register_transcription_provider()` 플러그인                      |
| OAuth 갱신 인증, 스트리밍 청크, 음성 목록 메타데이터 | `register_transcription_provider()` 플러그인                      |
| 이미 내장 제공자가 지원함(`local`, `groq`, `openai`, …)  | `stt.provider: <name>` 설정 — 내장 제공자는 인라인 처리됨               |

#### 확인 순서

1. **`stt.provider`가 내장 이름임** → 내장 디스패치. **항상 우선합니다.**
2. **`stt.provider`가 `command:`가 설정된 `stt.providers.<name>`과 일치함** → 명령 제공자 러너([STT 사용자 지정 명령 제공자](#stt-custom-command-providers) 참조). 같은 이름의 플러그인보다 우선합니다.
3. **`stt.provider`가 플러그인에 등록된 `TranscriptionProvider`와 일치함** → 플러그인 디스패치:
   - 플러그인의 `is_available()`가 `False`를 반환하면(자격 증명 또는 SDK 누락), 호출은 플러그인을 식별하는 사용 불가 오류 봉투를 표시합니다. 일반적인 "No STT provider available" 메시지는 표시하지 않습니다.
   - 그렇지 않으면 `model`(공개 `model=` 인수에서 가져오며, 없으면 `stt.<provider>.model`) 및 `language`(`stt.<provider>.language`에서 가져옴)와 함께 플러그인의 `transcribe()`를 호출합니다.
4. **일치하는 항목 없음** → "No STT provider available" 오류.

#### 제공자별 구성 네임스페이스

플러그인은 내장 제공자가 `stt.openai.model` / `stt.mistral.model`을 읽는 방식과 마찬가지로 `config.yaml`의 `stt.<provider>`에서 제공자별 구성을 읽습니다.

```yaml
stt:
  provider: my-stt
  my-stt:
    model: whisper-large-v3
    language: ja          # forwarded as language= to transcribe()
    # any other plugin-specific keys go here; read them via your
    # own config.yaml access in __init__/is_available/transcribe
```

디스패처는 이 섹션의 `model`과 `language`를 전달하며, 그 밖의 항목은 플러그인이 직접 읽을 수 있습니다.
#### 최소 플러그인

`~/.hermes/plugins/my-stt/`에 다음을 추가합니다.

`plugin.yaml`:
```yaml
name: my-stt
version: 0.1.0
description: "My custom Python STT backend"
```

`__init__.py`:
```python
from agent.transcription_provider import TranscriptionProvider


class MySTTProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "my-stt"  # what stt.provider matches against

    @property
    def display_name(self) -> str:
        return "My Custom STT"

    def is_available(self) -> bool:
        # Return False when credentials/deps are missing — picker skips
        # this row but the dispatcher still routes here on explicit config.
        import os
        return bool(os.environ.get("MY_STT_API_KEY"))

    def transcribe(self, file_path, *, model=None, language=None, **extra):
        # Return the standard transcribe envelope:
        #   {"success": bool, "transcript": str, "provider": str, "error": str}
        # Do NOT raise — convert exceptions to the error envelope so the
        # gateway/CLI caller sees a consistent shape on failure.
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

`hermes plugins enable my-stt`를 실행하고, `config.yaml`에 `stt.provider: my-stt`를 설정하면 음성 메시지의 전사가 플러그인을 통해 처리됩니다.

#### 선택적 훅

더 풍부한 통합을 위해 공급자 클래스에서 다음 메서드를 재정의할 수 있습니다.

- `list_models()` → `{id, display, languages, max_audio_seconds}` 딕셔너리 목록입니다.
- `default_model()` → 사용자가 모델을 재정의하지 않았을 때 반환되는 문자열입니다.
- `get_setup_schema()` → `hermes tools` / `hermes setup`의 선택기 행을 구성할 수 있도록 `{name, badge, tag, env_vars: [{key, prompt, url}]}`를 반환합니다(아직 STT용 선택기 카테고리는 제공되지 않지만, 향후 호환성을 위해 이 메타데이터를 플러그인에서 사용할 수 있습니다).

전체 ABC와 독스트링은 `agent/transcription_provider.py`를 참조하세요.
