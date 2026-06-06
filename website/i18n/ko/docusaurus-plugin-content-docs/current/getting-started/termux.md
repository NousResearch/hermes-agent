---
sidebar_position: 3
title: "Android / Termux"
description: "Termux를 사용하여 안드로이드 휴대폰에서 직접 Hermes Agent 실행하기"
---

# Android에서 Termux로 Hermes 실행하기

이 가이드는 [Termux](https://termux.dev/)를 통해 안드로이드 휴대폰에서 직접 Hermes Agent를 실행하는 테스트된 방법입니다.

이 방법을 통해 휴대폰에 작동하는 로컬 CLI와 현재 안드로이드에 정상적으로 설치되는 것으로 확인된 핵심 추가 기능을 제공받을 수 있습니다.

## 테스트된 방법에서 지원하는 기능은 무엇인가요?

테스트된 Termux 번들은 다음을 설치합니다:
- Hermes CLI
- cron 지원
- PTY/백그라운드 터미널 지원
- Telegram 게이트웨이 지원 (수동 / 베스트 에포트(best-effort) 백그라운드 실행)
- MCP 지원
- Honcho 메모리 지원
- ACP 지원

구체적으로 다음 명령어에 매핑됩니다:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## 아직 테스트되지 않았거나 지원되지 않는 기능은 무엇인가요?

일부 기능은 여전히 안드로이드용으로 제공되지 않거나 휴대폰에서 아직 검증되지 않은 데스크톱/서버 스타일의 종속성(dependencies)이 필요합니다:

- 현재 안드로이드에서는 `.[all]`이 지원되지 않습니다.
- `voice` 추가 기능은 `faster-whisper -> ctranslate2`로 인해 제한됩니다. `ctranslate2`는 안드로이드용 휠(wheels)을 제공하지 않습니다.
- Termux 설치 프로그램에서는 브라우저 자동화 / Playwright 부트스트랩이 생략됩니다.
- Termux 내부에서는 Docker 기반의 터미널 격리(isolation)를 사용할 수 없습니다.
- 안드로이드 시스템이 Termux 백그라운드 작업을 일시 중단할 수 있으므로, 게이트웨이 유지는 일반적인 관리형 서비스 형태가 아닌 베스트 에포트(best-effort) 방식으로 제공됩니다.

그렇다고 해서 Hermes가 휴대폰 네이티브 CLI 에이전트로서 훌륭하게 작동하는 데 지장을 주지는 않습니다. 단지 권장되는 모바일 설치 범위가 데스크톱/서버 설치보다 의도적으로 더 좁다는 것을 의미할 뿐입니다.

---

## 방법 1: 한 줄 설치 스크립트

이제 Hermes는 Termux를 감지하는 설치 경로를 제공합니다:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Termux에서 이 설치 스크립트는 자동으로 다음 작업을 수행합니다:
- 시스템 패키지 설치에 `pkg`를 사용합니다.
- `python -m venv`로 venv 가상 환경을 생성합니다.
- 먼저 더 넓은 범위의 `.[termux-all]` 추가 기능 설치를 시도하고, 실패 시 더 작은 `.[termux]` 추가 기능(그 다음 기본 설치)으로 폴백합니다. curl 설치 프로그램은 이 순서를 자동으로 맞춥니다.
- `hermes`를 `$PREFIX/bin`에 링크하여 Termux PATH에 유지되도록 합니다.
- 테스트되지 않은 브라우저 / WhatsApp 부트스트랩을 건너뜁니다.

명시적인 명령어를 확인하고 싶거나 설치 실패 문제를 디버깅해야 하는 경우, 아래의 수동 설치 방법을 사용하세요.

---

## 방법 2: 수동 설치 (상세 과정)

### 1. Termux 업데이트 및 시스템 패키지 설치

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

이 패키지들이 필요한 이유는 무엇인가요?
- `python` — 런타임 + venv 지원
- `git` — 리포지토리 클론 및 업데이트
- `clang`, `rust`, `make`, `pkg-config`, `libffi`, `openssl` — 안드로이드에서 일부 Python 종속성을 빌드하는 데 필요
- `nodejs` — 테스트된 핵심 경로 이외의 실험을 위한 선택적 Node 런타임
- `ripgrep` — 빠른 파일 검색
- `ffmpeg` — 미디어 / TTS 변환

### 2. Hermes 클론하기

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

### 3. 가상 환경 생성

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
```

`ANDROID_API_LEVEL`은 `jiter`와 같이 Rust / maturin 기반 패키지를 설치할 때 중요합니다.

### 4. 테스트된 Termux 번들 설치

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

최소한의 핵심 에이전트 기능만 원하는 경우, 다음 명령어도 작동합니다:

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. Termux PATH에 `hermes` 등록하기

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

`$PREFIX/bin`은 Termux의 PATH에 이미 포함되어 있으므로, 이 링크 작업을 해두면 매번 venv를 다시 활성화하지 않고도 새로운 쉘 세션에서 `hermes` 명령어를 계속 사용할 수 있습니다.

### 6. 설치 확인

```bash
hermes version
hermes doctor
```

### 7. Hermes 시작

```bash
hermes
```

---

## 권장 추가 설정

### 모델 구성하기

```bash
hermes model
```

또는 `~/.hermes/.env` 파일에 API 키를 직접 설정할 수도 있습니다.

### 나중에 전체 대화형 설정 마법사 다시 실행하기

```bash
hermes setup
```

### 선택적 Node 종속성 수동 설치

테스트된 Termux 경로에서는 Node/브라우저 부트스트랩을 의도적으로 건너뜁니다. 나중에 브라우저 도구를 실험해 보고 싶다면 다음과 같이 실행하세요:

```bash
pkg install nodejs-lts
npm install
```

브라우저 도구는 자동으로 Termux 디렉토리 (`/data/data/com.termux/files/usr/bin`)를 PATH 검색에 포함하므로, 추가적인 PATH 설정 없이도 `agent-browser` 및 `npx`를 찾아냅니다.

별도로 문서화되기 전까지는 안드로이드에서의 브라우저 / WhatsApp 도구를 실험적인 기능으로 간주해 주세요.

---

## 문제 해결

### `.[all]` 설치 시 `No solution found` 오류 발생

대신 테스트된 Termux 번들을 사용하세요:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

현재 원인은 `voice` 추가 기능의 제한 때문입니다:
- `voice`는 `faster-whisper`를 가져옵니다.
- `faster-whisper`는 `ctranslate2`에 의존합니다.
- `ctranslate2`는 안드로이드용 휠(wheels)을 제공하지 않습니다.

### 안드로이드에서 `uv pip install` 실패

대신 stdlib venv + `pip`를 사용하는 Termux 경로를 사용하세요:

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` 설치 중 `ANDROID_API_LEVEL` 관련 경고 또는 오류 발생

설치 전에 API 레벨을 명시적으로 설정하세요:

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor`에서 ripgrep 또는 Node가 누락되었다고 표시됨

Termux 패키지로 직접 설치해 주세요:

```bash
pkg install ripgrep nodejs
```

### Python 패키지 설치 중 빌드 실패 오류 발생

빌드 툴체인이 설치되어 있는지 확인하세요:

```bash
pkg install clang rust make pkg-config libffi openssl
```

그 다음 다시 시도하세요:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

---

## 모바일 환경에서의 알려진 제한 사항

- Docker 백엔드를 사용할 수 없습니다.
- 테스트된 경로에서는 `faster-whisper`를 통한 로컬 음성 전사(transcription)를 사용할 수 없습니다.
- 설치 프로그램이 브라우저 자동화 설정을 의도적으로 건너뜁니다.
- 다른 선택적 추가 기능들이 작동할 수도 있지만, 현재 문서화되어 테스트 완료된 안드로이드 번들은 `.[termux]` 및 `.[termux-all]`뿐입니다.

안드로이드 특유의 새로운 문제를 발견하셨다면 다음 정보를 포함하여 GitHub 이슈를 열어주세요:
- 사용 중인 안드로이드 버전
- `termux-info` 실행 결과
- `python --version` 실행 결과
- `hermes doctor` 실행 결과
- 실행한 정확한 설치 명령어 및 전체 오류 출력 내용
