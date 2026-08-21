---
sidebar_position: 3
title: "Android / Termux"
description: "Termux를 사용해 Android 휴대폰에서 Hermes Agent 직접 실행"
---

# Termux를 사용하는 Android의 Hermes

:::warning Tier 2 플랫폼
Termux(Android)는 [Tier 2 플랫폼](./platform-support.md#tier-2)입니다. 여기의 설치 스크립트와 문서는 최선의 노력에 한해 유지 관리됩니다. `main`에 커밋되면 언제든 이 패키지가 중단될 수 있습니다.
:::

[Termux](https://termux.dev/)를 통해 Android 휴대폰에서 Hermes Agent를 직접 실행할 수 있습니다.

휴대폰에서 작동하는 로컬 CLI와 현재 Android에 문제없이 설치되는 것으로 알려진 핵심 추가 기능을 제공합니다.

## 테스트된 경로에서 지원하는 기능

테스트된 Termux 번들은 다음을 설치합니다.

- Hermes CLI
- cron 지원
- PTY/백그라운드 터미널 지원
- Telegram 게이트웨이 지원(수동 / 최선의 노력에 의한 백그라운드 실행)
- MCP 지원
- Honcho 메모리 지원
- ACP 지원

구체적으로 다음에 해당합니다.

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## 아직 테스트된 경로에 포함되지 않는 기능

일부 기능은 Android용으로 배포되지 않은 데스크톱/서버 수준의 의존성이 필요하거나 아직 휴대폰에서 검증되지 않았습니다.

- 현재 Android에서는 `.[all]`을 지원하지 않습니다.
- `voice` 추가 기능은 `faster-whisper -> ctranslate2` 때문에 차단됩니다. `ctranslate2`는 Android wheel을 배포하지 않습니다.
- Termux 설치 프로그램은 자동 브라우저 / Playwright 부트스트랩을 건너뜁니다.
- Termux 내부에서는 Docker 기반 터미널 격리를 사용할 수 없습니다.
- Android가 Termux 백그라운드 작업을 중지할 수 있으므로 게이트웨이 지속성은 일반적인 관리 서비스가 아니라 최선의 노력 수준입니다.

이는 Hermes가 휴대폰 전용 CLI 에이전트로 잘 작동하는 것을 막지 않습니다. 다만 권장 모바일 설치가 데스크톱/서버 설치보다 의도적으로 좁다는 뜻입니다.

---

## 옵션 1: 한 줄 설치 프로그램

Hermes는 이제 Termux를 인식하는 설치 경로를 제공합니다.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Termux에서 설치 프로그램은 자동으로 다음을 수행합니다.

- 시스템 패키지에 `pkg` 사용
- `python -m venv`로 venv 생성
- 먼저 광범위한 `.[termux-all]` 추가 기능을 시도하고 더 작은 `.[termux]` 추가 기능(그 다음에는 기본 설치)으로 대체 — curl 설치 프로그램은 이 순서를 자동으로 따릅니다.
- `hermes`를 `$PREFIX/bin`에 연결하여 Termux PATH에 유지
- 테스트되지 않은 브라우저 / WhatsApp 부트스트랩 건너뛰기

명시적인 명령이 필요하거나 설치 실패를 디버깅하려면 아래 수동 경로를 사용하세요.

---

## 옵션 2: 수동 설치(모든 단계 명시)

### 1. Termux 업데이트 및 시스템 패키지 설치

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

이 패키지가 필요한 이유는 다음과 같습니다.

- `python` — 런타임 및 venv 지원
- `git` — 저장소 복제/업데이트
- `clang`, `rust`, `make`, `pkg-config`, `libffi`, `openssl` — Android에서 일부 Python 의존성을 빌드하는 데 필요
- `nodejs` — 테스트된 핵심 경로 외의 실험을 위한 선택적 Node 런타임
- `ripgrep` — 빠른 파일 검색
- `ffmpeg` — 미디어 / TTS 변환

### 2. Hermes 복제

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

`ANDROID_API_LEVEL`은 `jiter`와 같은 Rust / maturin 기반 패키지에 중요합니다.

### 4. 테스트된 Termux 번들 설치

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

최소한의 핵심 에이전트만 원한다면 다음도 작동합니다.

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. Termux PATH에 `hermes` 추가

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

Termux에서는 이미 `$PREFIX/bin`이 PATH에 있으므로, 매번 venv를 다시 활성화하지 않아도 새 셸에서 `hermes` 명령을 사용할 수 있습니다.

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

## 권장 후속 설정

### 모델 구성

```bash
hermes model
```

또는 `~/.hermes/.env`에 키를 직접 설정하세요.

### 나중에 전체 대화형 설정 마법사 다시 실행

```bash
hermes setup
```

### 선택적 Node 의존성 수동 설치

테스트된 Termux 경로는 의도적으로 Node/브라우저 부트스트랩을 건너뜁니다. 나중에 브라우저 도구를 실험하려면 사용하는 백엔드에 따라 필요한 항목이 다릅니다.

- **클라우드 브라우저 제공자**(Browserbase, Browser Use, Firecrawl)는 자체 Chromium을 호스팅하므로 Node.js만 있으면 됩니다. `agent-browser`는 처음 사용할 때 `npx agent-browser`를 통해 지연 해결됩니다.

  ```bash
  pkg install nodejs-lts
  ```

- **Termux의 로컬 브라우저 자동화**에는 실제 `agent-browser` 설치가 필요합니다. 기본 npx 대체 경로는 너무 불안정하여 준비된 기능으로 안내하지 않도록 로컬 모드에서 의도적으로 거부됩니다.

  ```bash
  pkg install nodejs-lts
  npm install -g agent-browser && agent-browser install
  ```

브라우저 도구는 PATH 검색에 Termux 디렉터리(`/data/data/com.termux/files/usr/bin`)를 자동으로 포함하므로 별도의 PATH 설정 없이 `agent-browser`와 `npx`를 찾습니다.

Android의 브라우저 / WhatsApp 도구는 달리 문서화될 때까지 실험적 기능으로 취급하세요.

---

## 문제 해결

### `.[all]` 설치 시 `No solution found`

대신 테스트된 Termux 번들을 사용하세요.

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

현재 차단 요인은 `voice` 추가 기능입니다.

- `voice`가 `faster-whisper`를 가져옵니다.
- `faster-whisper`가 `ctranslate2`에 의존합니다.
- `ctranslate2`는 Android wheel을 배포하지 않습니다.

### Android에서 `uv pip install` 실패

대신 표준 라이브러리 venv + `pip`를 사용하는 Termux 경로를 이용하세요.

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin`에서 `ANDROID_API_LEVEL` 관련 오류

설치 전에 API 수준을 명시적으로 설정하세요.

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor`에서 ripgrep 또는 Node가 없다고 표시

Termux 패키지로 설치하세요.

```bash
pkg install ripgrep nodejs
```

### Python 패키지 설치 중 빌드 실패

빌드 도구 모음이 설치되어 있는지 확인하세요.

```bash
pkg install clang rust make pkg-config libffi openssl
```

그런 다음 다시 시도하세요.

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

---

## 휴대폰의 알려진 제한 사항

- Docker 백엔드를 사용할 수 없습니다.
- 테스트된 경로에서는 `faster-whisper`를 통한 로컬 음성 전사가 불가능합니다.
- 설치 프로그램은 브라우저 자동화 설정을 의도적으로 건너뜁니다.
- 일부 선택적 추가 기능은 작동할 수 있지만, 현재 테스트된 Android 번들로 문서화된 것은 `.[termux]`와 `.[termux-all]`뿐입니다.

새로운 Android 관련 문제가 발생하면 다음 정보를 포함해 GitHub issue를 열어 주세요.

- Android 버전
- `termux-info`
- `python --version`
- `hermes doctor`
- 정확한 설치 명령과 전체 오류 출력
