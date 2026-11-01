---
sidebar_position: 11
title: "웨이크 워드"
description: "핸즈프리 'Hey Hermes' 웨이크 워드 — 말로 음성 세션을 시작하는 방식, 'Hey Siri'처럼"
---

# 웨이크 워드 ("Hey Hermes")

웨이크 워드는 CLI, TUI, 데스크톱 앱 전체에서 Hermes를 핸즈프리 어시스턴트로 바꿉니다. 설정 하나만 켜면 Hermes가 백그라운드에서 음성 트리거 문구를 듣습니다. 문구를 말하면 Hermes가 새 세션을 시작하고, 마이크를 열고, 일반 [음성 파이프라인](/user-guide/features/voice-mode)을 통해 명령을 받아 "Hey Siri"나 "Alexa"와 똑같이 답합니다. 어느 환경이 듣게 할지는 `surface`로 선택합니다.

감지는 **기기에서만** 완전히 수행됩니다. 상시 대기 리스너는 웨이크 문구만 감시하며, 실제로 에이전트에게 명령을 말하기 전까지 오디오가 컴퓨터 밖으로 나가지 않습니다.

## 작동 방식

1. `wake_word.enabled: true`(또는 `/wake on` 실행) 상태가 되면 가벼운 핫워드 감지기가 설정된 입력 장치, 또는 `wake_word.input_device`가 설정되지 않은 경우 프로세스 기본 마이크에서 듣습니다.
2. 웨이크 문구를 들으면 스스로 일시 중지해 마이크를 해제하고, 새 세션을 시작한 뒤 음성 모드의 무음 감지로 한 번의 발화를 녹음합니다.
3. 음성이 전사되어 에이전트로 전송됩니다. 에이전트가 답한 후 리스너가 자동으로 재개되어 다음 웨이크 워드를 기다립니다.

기본값은 **꺼짐**입니다. 직접 켤 때까지 아무것도 듣지 않습니다.

데스크톱 앱에서는 **"stop"**(또는 "never mind", "goodbye", "cancel", "that's all")이라고 말하기만 하면 핸즈프리 음성 대화를 종료할 수 있습니다. 이 음성 명령은 에이전트로 전송되지 않고 대화를 종료합니다. 전체 발화가 중지 명령과 일치할 때만 동작하므로, "stop the docker container"처럼 실제 요청은 정상적으로 처리됩니다.



## 원격 데스크톱 (클라이언트 캡처)

데스크톱 앱이 **원격** Hermes 백엔드(예: 헤드리스 Docker 호스트 또는 다른 방의 컴퓨터)에 연결되면 백엔드에 **마이크가 없는** 경우가 많습니다. 그러면 서버 측 PortAudio가 “Failed to open the wake-word microphone.” 오류를 냅니다.

Hermes는 이 경우를 위해 **클라이언트 캡처**를 지원합니다.

1. 데스크톱에서 `capture: client`로 웨이크를 활성화합니다(백엔드에 로컬 입력 장치가 없으면 GUI에서 자동으로 활성화되거나, 아래처럼 명시적으로 설정할 수 있습니다).
2. openWakeWord는 계속 **백엔드에서** 실행됩니다(동일한 엔진과 모델 사용).
3. 데스크톱이 **로컬 Mac/PC 마이크**를 열고, 16 kHz 모노 int16으로 리샘플링한 짧은 프레임을 `wake.feed` RPC를 통해 스트리밍합니다.
4. 감지되면 백엔드가 평소처럼 `wake.detected`를 내보내고, 데스크톱이 클라이언트 마이크에서 일반 음성 파이프라인을 시작합니다.

```yaml
wake_word:
  enabled: true
  capture: auto    # auto | local | client
  # auto   — local PortAudio unless the desktop arms with client_capture
  # local  — always open the backend mic (CLI/TUI default)
  # client — always expect wake.feed PCM from the desktop (remote-friendly)
```

데스크톱 GUI는 `wake.start`에 항상 `client_capture: true`를 전달하므로, 마이크가 없는 원격 백엔드는 자동으로 클라이언트 모드로 활성화됩니다. CLI와 TUI는 `capture: client`를 명시적으로 설정하지 않는 한 로컬 캡처를 유지합니다.

개인정보 보호 참고: 클라이언트 캡처를 사용하면 웨이크 PCM이 인증된 데스크톱↔백엔드 WebSocket을 통해 이동합니다(세션의 나머지 부분과 동일한 채널). 감지는 여전히 오디오를 서드파티 웨이크 API로 보내지 않으며, 엔진은 백엔드 프로세스에 로컬로 존재합니다.

## 엔진

| 엔진 | 비용 | API 키 | 참고 |
|--------|------|---------|-------|
| **openWakeWord** (기본값) | 무료 | 없음 | 로컬 ONNX 모델. 번들된 **"hey hermes"** 모델(기본값)을 제공하며, `hey_jarvis`, `alexa`, `hey_mycroft`, … 및 사용자 지정 모델도 지원 |
| **sherpa** | 무료 | 없음 | **오픈 보캐뷸러리** — 학습 없이 입력한 ANY 문구를 감지합니다. 작은 영어 모델을 처음 사용할 때 자동으로 다운로드합니다(~13 MB) |
| **Porcupine** | 무료 요금제 / 유료 | `PORCUPINE_ACCESS_KEY` | Picovoice 엔진. 기본 제공 키워드와 사용자 지정 `.ppn` 파일 |

기본 문구는 **"hey hermes"**입니다. 이 문구용 모델이 Hermes에 포함되어 있으므로 학습 없이 바로 사용할 수 있습니다. (처음 사용할 때 openWakeWord가 공유 특징 추출 모델을 다운로드하며, 한 번만 소량의 데이터를 가져옵니다.)

두 엔진 모두 웨이크 워드를 처음 활성화할 때 지연 설치됩니다(`--include-desktop`으로 설치한 데스크톱 버전은 미리 설치하므로 즉시 귀가 활성화됩니다). 미리 설치하려면:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[wake]"
```

## 빠른 시작

```bash
# In an interactive `hermes` session:
/wake on        # start listening (installs the engine on first use)
/wake status    # show phrase, provider, and state
/wake off       # stop listening
```

데스크톱 앱에서는 컴포저의 귀 아이콘을 클릭합니다.

토글 자체가 설정입니다. `/wake` 또는 데스크톱의 귀 버튼으로 웨이크 워드를 켜거나 끄면 `wake_word.enabled`가 `~/.hermes/config.yaml`에 함께 기록되므로 선택이 세션 간에 유지됩니다. 직접 변경할 수도 있습니다.

```yaml
wake_word:
  enabled: true
```

## 구성

```yaml
wake_word:
  enabled: false
  surface: auto               # eligible surface: "auto" | "cli" | "tui" | "gui"
  input_device: null           # PortAudio input index or device-name substring; null = process default
  capture: auto               # auto | local | client — where PCM is captured (see Remote desktop)
  provider: openwakeword      # "openwakeword" (free, local) | "sherpa" (free, any phrase) | "porcupine"
  phrase: "hey hermes"        # cosmetic label only — detection is keyed by the model/keyword below
  sensitivity: 0.6            # 0.0-1.0 — higher = stricter (fewer false triggers), consistent across all engines
  confirmation_frames: 3      # openWakeWord only — consecutive over-threshold frames required to fire
  start_new_session: true     # start a fresh session on wake vs. continue the current one
  openwakeword:
    model: hey_hermes         # bundled default; OR a built-in name OR a path to a custom .onnx/.tflite
    inference_framework: ""   # "" (auto) | "onnx" | "tflite"
  porcupine:
    keyword: jarvis           # built-in keyword OR path to a custom .ppn
```

두 엔진 모두 `sensitivity`, `phrase`, `start_new_session`을 적용합니다. `openwakeword`와 `porcupine` 블록이 실제 감지 모델을 선택합니다.

`input_device`는 웨이크 리스너의 PortAudio(`sounddevice`) 스트림에 직접 전달됩니다. 숫자 장치 인덱스 또는 모호하지 않은 장치 이름 일부를 사용하세요. 이 설정은 웨이크 워드 캡처만 변경하며, 데스크톱의 푸시 투 토크는 여전히 데스크톱 애플리케이션의 마이크 경로를 사용합니다.

### 주변 음성으로 인한 오감지 줄이기

openWakeWord는 한 번에 짧은(~80ms) 오디오 프레임 하나의 점수를 계산하므로, 배경 대화의 무관한 음소가 때때로 임계값을 넘는 단일 프레임으로 급증해 웨이크 워드를 의도치 않게 실행할 수 있습니다. 두 가지 설정으로 이를 제어합니다.

- **`confirmation_frames`** (기본값 `3`, openWakeWord 전용) — 웨이크가 실행되기 전에 임계값을 넘겨야 하는 *연속* 프레임 수입니다. 실제 "hey hermes"는 여러 프레임 동안 높은 점수를 유지하지만 주변의 짧은 소리는 보통 한 프레임만 급증합니다. 시끄러운 공간에서 오감지가 계속되면 `4`–`5`처럼 높이세요. 대신 지연 시간이 수십 밀리초 늘어납니다. `1`로 설정하면 기존의 첫 프레임 즉시 실행 동작으로 돌아갑니다.
- **`sensitivity`** (기본값 `0.6`) — 감지 임계값으로, 범위는 `0.0`–`1.0`입니다. 높을수록 엄격해져 오감지가 줄어듭니다. 이 방향은 **모든** 엔진에서 일관됩니다. openWakeWord에서는 프레임별 원시 점수 임계값이고, sherpa에서는 키워드 임계값으로 변환되며, Porcupine에서는 내부적으로 반전되어 여기서도 “높을수록 엄격”이 유지됩니다. 기본값 `0.6`은 openWakeWord의 관대한 기준선 `0.5`보다 높아서 "hey hor" 같은 아슬아슬한 오인식을 걸러냅니다. 그래도 오감지가 발생하면 `0.8` 쪽으로 높이고, 실제 "hey hermes" 발화를 놓치면 낮추세요.

`sherpa`와 `porcupine` 엔진은 내부적으로 전체 문구를 디코딩하므로 단일 프레임 급증 문제가 없으며 `confirmation_frames`를 무시합니다(단, `sensitivity`는 적용합니다).

`inference_framework`는 openWakeWord 백엔드를 선택합니다. 비워 두면(기본값) Hermes가 플랫폼별로 선택하도록 하세요. **Apple Silicon에서는 tflite**, 그 외에는 어디서나 onnx를 사용합니다. openWakeWord의 onnx 백엔드는 macOS ARM64에서 거의 0에 가까운 점수를 반환하므로([openWakeWord#336](https://github.com/dscripka/openWakeWord/issues/336)), 해당 환경에서 `onnx`로 고정한 리스너는 활성화되고 듣는 중으로 표시되지만 절대 실행되지 않습니다. tflite 백엔드는 macOS에서 `ai-edge-litert`가 필요하며 Hermes가 다른 웨이크 워드 의존성과 함께 필요할 때 설치합니다.

### 표면 (CLI, TUI, GUI)

웨이크 워드는 Hermes의 세 표면 모두에서 작동하며, `surface`는 감지되었을 때 리스너를 소유하고 새 세션을 여는 표면을 선택합니다.

| `surface` | 동작 |
|-----------|----------|
| `auto` (기본값) | 모든 로컬 표면이 대상이며, 먼저 활성화한 표면이 리스너를 소유합니다. |
| `cli` | 클래식 `hermes` CLI만 사용합니다. |
| `tui` | `hermes --tui`만 사용합니다. |
| `gui` | 데스크톱 앱만 사용합니다. |

감지기는 기기에서 실행되고 마이크 하나만 사용하므로 Hermes 표면이 별도 프로세스에서 실행될 때를 포함해 한 번에 하나의 표면만 듣습니다. 소유권은 고정됩니다. 처음 자격을 갖춘 요청자가 리스너를 중지하거나 연결을 끊거나 프로세스가 종료될 때까지 리스너를 유지합니다. Hermes는 열려 있는 다른 표면으로 조용히 자동 전환하지 않습니다. 먼저 요청한 표면이 소유하는 방식을 쓰지 않고 소유권을 고정하려면 `surface`를 설정하세요. TUI와 데스크톱 GUI는 동일한 Python 백엔드(`tui_gateway`)를 공유하며, 백엔드가 서버 측에서 감지기를 실행하고 명령을 녹음하는 동안 음성 캡처에 마이크를 넘깁니다.

## 다른 문구 사용

"Hey Hermes"는 기본으로 바로 작동합니다. 번들된 openWakeWord 모델(`model: hey_hermes`)이 기본값이기 때문입니다. 다른 문구로 깨우려면 가장 쉬운 방법은 오픈 보캐뷸러리 엔진입니다.

### 옵션 A — sherpa (어떤 문구든, 학습 불필요)

원하는 문구를 입력하면 런타임에 토큰화됩니다. "hey coder", "computer", "wake up neo" 등 무엇이든 가능합니다.

```yaml
wake_word:
  enabled: true
  provider: sherpa
  phrase: "hey coder"        # detection key — just type your phrase
```

작은 영어 KWS 모델(~13 MB)은 처음 사용할 때 한 번 다운로드됩니다. 각 프로필은 자체 문구를 설정할 수 있으며, 실행하는 각 프로필에 대해 "hey \<프로필\>"처럼 사용할 수 있습니다.

### 특정 프로필 깨우기 (데스크톱)

sherpa 엔진을 사용하면 하나의 리스너로 **어떤 프로필이든** 깨울 수 있습니다. `wake_word.enabled: true`로 설정된 모든 프로필이 자동으로 등록되며, 문구를 지정하지 않으면 기본값은 `hey <프로필 이름>`입니다. 프로필의 문구를 말하면 데스크톱 앱이 해당 프로필로 실시간 전환하고, 그곳에서 새 세션을 연 뒤 핸즈프리 음성을 시작합니다.

- "hey hermes" → 기본 프로필
- "hey coder" → `coder` 프로필
- "hey trader" → `trader` 프로필

리스너가 있는 프로필에서 `wake_word.profile_routing: false`를 설정하면 라우팅을 사용하지 않고 자체 문구만 듣습니다. CLI와 TUI는 단일 프로필 프로세스이므로 다른 프로필에 속한 웨이크 문구가 들리면 라우팅하는 대신 전환 명령(`hermes -p <profile>`)을 출력합니다.

이름은 영어 하위 단어의 음향으로 매칭됩니다. 구별되는 2음절 이상의 이름을 가진 두 단어 문구가 가장 잘 작동합니다. 매우 짧은 이름, 강한 비영어권 음운 체계, 또는 발음이 비슷한 두 프로필은 정확도를 떨어뜨리므로 필요하면 프로필별 `sensitivity`를 조정하세요.

### 옵션 B — openWakeWord (무료, 학습된 모델)

기본 제공 모델(`hey_jarvis`, `alexa`, `hey_mycroft`, …)의 이름을 지정하거나, 최대한의 견고성을 위해 사용자 지정 모델(무료/Colab GPU에서 약 75–90분)을 학습하고 `.onnx` 파일을 어딘가에 둔 다음 경로를 참조합니다.

```yaml
wake_word:
  enabled: true
  provider: openwakeword
  phrase: "computer"
  openwakeword:
    model: ~/.hermes/wakewords/computer.onnx   # or a built-in name like hey_jarvis
```

학습 참고 자료:

- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [2026 training Colab](https://github.com/alfiedennen/openwakeword-colab-2026)

:::tip 눈에 잘 띄는 문구 선택
일상적인 말과 겹치지 않는 웨이크 문구가 가장 잘 일반화됩니다. 흔한 단어인 "hello"나 "stop"보다 드문 단어를 포함한 두 음절 문구("hermes"가 여기에 해당)가 낫습니다.
:::

### 옵션 C — Porcupine (몇 초 만에 사용자 지정 키워드 만들기)

[Picovoice Console](https://console.picovoice.ai/)에서 "Hey Hermes" 키워드를 만들고, `.ppn`을 다운로드한 뒤 다음과 같이 설정합니다.

```yaml
wake_word:
  enabled: true
  provider: porcupine
  phrase: "hey hermes"
  porcupine:
    keyword: ~/.hermes/wakewords/hey_hermes.ppn
```

액세스 키를 `~/.hermes/.env`에 설정합니다.

```bash
PORCUPINE_ACCESS_KEY=your-key-here
```

## 요구 사항

- 작동하는 마이크와 `sounddevice` + `numpy` 오디오 스택(음성 모드와 공유).
- 음성 명령을 전사할 STT 제공자 — 로컬 `faster-whisper`는 기본으로 바로 작동합니다. 전체 제공자 목록은 [음성 모드](/user-guide/features/voice-mode)를 참고하세요.
- 답변을 말할 TTS 제공자 — 기본 `edge-tts`는 키 없이 작동합니다. 웨이크 흐름은 완전한 핸즈프리 방식이므로 STT와 TTS가 모두 준비될 때까지 토글로 활성화할 수 없습니다. `hermes tools`의 음성 섹션에서 설정할 수 있습니다.
- 웨이크 엔진 의존성(자동 설치되거나 `hermes-agent[wake]`로 설치).

리스너가 시작되지 않으면 `/wake status`에서 정확히 누락된 항목을 알려줍니다.

### "Listening"으로 표시되지만 macOS에서 절대 깨우지 못함

macOS는 프로세스별로 마이크 접근 권한을 부여합니다. 데스크톱 앱에서 STT가 작동한다고 해서 *렌더러*가 마이크에 접근할 수 있다는 뜻일 뿐이며, 웨이크 리스너는 Python *백엔드*에서 실행되므로 별도의 권한이 필요합니다. 권한이 없으면 CoreAudio가 백엔드에 "작동하는" 스트림을 넘기지만 실제로는 무음만 전달하므로, 귀 아이콘은 듣는 중으로 표시되어도 문구가 실행되지 않습니다. Hermes가 이를 감지합니다(`/wake status`에 "mic delivers only silence"가 표시되며 데스크톱 귀 툴팁에도 같은 힌트가 표시됩니다). 해결 방법: 시스템 설정 → 개인정보 보호 및 보안 → 마이크에서 Hermes 백엔드를 활성화하세요(터미널, `python` 또는 Hermes로 표시될 수 있습니다). 그런 다음 웨이크 워드를 껐다가 다시 켭니다.

### "Listening"으로 표시되지만 Windows에서 무음을 받음

데스크톱 푸시 투 토크와 웨이크 워드 캡처는 서로 다른 마이크 경로를 사용합니다. 푸시 투 토크는 데스크톱 애플리케이션의 브라우저 캡처를 사용하지만, 웨이크 워드 리스너는 Python 백엔드에서 PortAudio 스트림을 엽니다. 한쪽은 작동해도 다른 쪽이 무음이거나 사용할 수 없는 Windows 입력을 선택할 수 있습니다.

`/wake status`에서 선택된 입력 장치와 Windows 오디오 호스트 API를 확인할 수 있습니다. 무음으로 표시되면 `wake_word.input_device`를 작동하는 PortAudio 입력의 숫자 인덱스 또는 모호하지 않은 이름으로 설정한 다음 웨이크 워드를 다시 켭니다.

```bash
hermes config set wake_word.input_device "Microphone Array"
```

프로세스 기본값으로 돌아가려면 `null`을 사용합니다.

```bash
hermes config set wake_word.input_device null
```

## 참고 및 제한 사항

- **로컬 표면만 지원합니다.** 웨이크 워드는 로컬 마이크를 사용할 수 있는 CLI, TUI, 데스크톱 GUI에서 실행됩니다. 마이크가 없는 메시징 게이트웨이(Telegram, Discord, …)에서는 실행되지 않습니다.
- **한 번에 하나의 마이크만 사용합니다.** 감지기는 명령을 녹음하는 동안 마이크를 해제하고 턴이 끝나면 다시 확보하므로 음성 캡처와 충돌하지 않습니다.
- **개인정보 보호.** 핫워드 감지는 로컬에서 수행됩니다. 오감지가 발생하면 `sensitivity`를 높이고, 사용자의 말을 놓치면 낮추세요.
