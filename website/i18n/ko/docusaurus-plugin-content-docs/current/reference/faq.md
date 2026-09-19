---
sidebar_position: 3
title: "FAQ 및 문제 해결"
description: "Hermes Agent에 대한 자주 묻는 질문과 일반적인 문제의 해결 방법"
---

# FAQ 및 문제 해결

가장 자주 발생하는 질문과 문제에 대한 빠른 답변 및 해결 방법입니다.

---

## 자주 묻는 질문

### Hermes에서 사용할 수 있는 LLM 제공자는 무엇인가요?

Hermes Agent는 OpenAI 호환 API라면 무엇이든 사용할 수 있습니다. 지원되는 제공자는 다음과 같습니다.

- **[OpenRouter](https://openrouter.ai/)** — 하나의 API 키로 수백 개의 모델에 액세스 (유연성이 중요한 경우 권장)
- **[Nous Portal](/integrations/nous-portal)** — Nous Research의 구독 게이트웨이 — 하나의 OAuth 로그인으로 300개 이상의 모델과 웹/이미지/TTS/브라우저 이용 (초보자에게 권장)
- **OpenAI** — GPT-5.4, GPT-5-codex, GPT-4.1, GPT-4o 등
- **Anthropic** — Claude 모델 (직접 API, `hermes auth add anthropic`을 통한 OAuth, OpenRouter 또는 호환 프록시)
- **Google** — Gemini 모델 (`gemini` 제공자를 통한 직접 API, OpenRouter 또는 호환 프록시)
- **z.ai / ZhipuAI** — GLM 모델
- **Kimi / Moonshot AI** — Kimi 모델
- **MiniMax** — 글로벌 및 중국 엔드포인트
- **로컬 모델** — [Ollama](https://ollama.com/), [vLLM](https://docs.vllm.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [SGLang](https://github.com/sgl-project/sglang) 또는 OpenAI 호환 서버를 통해 사용

`hermes model`을 사용하거나 `~/.hermes/.env`를 편집하여 제공자를 설정하세요. 모든 제공자 키는 [환경 변수](./environment-variables.md) 참조 문서를 확인하세요.

### Windows/Android/Termux/내 플랫폼에서도 작동하나요??
전체 플랫폼 지원 현황은 **[플랫폼 지원](../getting-started/platform-support.md)**을 확인하세요.

### WSL2에서 Hermes를 실행하고 있습니다. 일반적인 Windows Chrome을 제어하는 가장 좋은 방법은 무엇인가요?
`/browser connect`보다 MCP 브리지를 사용하는 것이 좋습니다.

권장 패턴은 다음과 같습니다.

- WSL2 안에서 Hermes 실행
- Windows에서 로그인된 일반 Chrome 계속 사용
- `cmd.exe` 또는 `powershell.exe`를 통해 `chrome-devtools-mcp`를 MCP 서버로 추가
- 그 결과로 생성된 MCP 브라우저 도구를 Hermes가 사용하도록 함

이 방법이 WSL2/Windows 경계를 넘어 Hermes 핵심 브라우저 전송을 직접 연결하려는 것보다 안정적입니다.

자세한 내용은 다음을 참고하세요.

- [Hermes에서 MCP 사용](../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)
- [브라우저 자동화](../user-guide/features/browser.md#wsl2--windows-chrome-prefer-mcp-over-browser-connect)

### 제 데이터가 외부로 전송되나요?

API 호출은 **사용자가 설정한 LLM 제공자에게만** 전송됩니다 (예: OpenRouter, 로컬 Ollama 인스턴스). Hermes Agent는 텔레메트리, 사용 데이터 또는 분석 정보를 수집하지 않습니다. 대화, 메모리, 스킬은 `~/.hermes/`에 로컬로 저장됩니다.

### 오프라인이나 로컬 모델로 사용할 수 있나요?

네. `hermes model`을 실행하고 **Custom endpoint**를 선택한 다음 서버 URL을 입력하세요.

```bash
hermes model
# Select: Custom endpoint (enter URL manually)
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 64000   ← Hermes minimum; set this to match your server's actual context window
```

또는 `config.yaml`에서 직접 설정할 수 있습니다.

```yaml
model:
  default: qwen3.5:27b
  provider: custom
  base_url: http://localhost:11434/v1
```

Hermes는 엔드포인트, 제공자, 기본 URL을 `config.yaml`에 저장하므로 재시작 후에도 유지됩니다. 로컬 서버에 로드된 모델이 정확히 하나라면 `/model custom`이 해당 모델을 자동으로 감지합니다. `config.yaml`에서 `provider: custom`을 설정할 수도 있습니다. 이는 다른 어떤 것의 별칭이 아니라 정식 제공자입니다.

이는 Ollama, vLLM, llama.cpp server, SGLang, LocalAI 및 기타 도구에서 작동합니다. 자세한 내용은 [구성 가이드](../user-guide/configuration.md)를 확인하세요.

:::tip Ollama 사용자
Ollama에서 사용자 지정 `num_ctx`를 설정했다면 (예: `ollama run --num_ctx 64000`) Hermes에서도 일치하는 컨텍스트 길이를 설정하세요. Ollama의 `/api/show`는 설정한 유효 `num_ctx`가 아니라 모델의 *최대* 컨텍스트를 보고합니다.
:::

:::tip 로컬 모델의 타임아웃
Hermes는 로컬 엔드포인트를 자동으로 감지하고 스트리밍 타임아웃을 완화합니다 (읽기 타임아웃을 120초에서 1800초로 늘리고 오래된 스트림 감지를 비활성화). 매우 긴 컨텍스트에서 여전히 타임아웃이 발생하면 `.env`에 `HERMES_STREAM_READ_TIMEOUT=1800`을 설정하세요. 자세한 내용은 [로컬 LLM 가이드](../guides/local-llm-on-mac.md#timeouts)를 확인하세요.
:::

### 비용은 얼마나 드나요?

Hermes Agent 자체는 **무료 오픈 소스**입니다 (MIT 라이선스). 선택한 제공자의 LLM API 사용량에 대해서만 비용을 지불합니다. 로컬 모델은 실행 비용이 완전히 무료입니다.

### 한 인스턴스를 여러 사람이 사용할 수 있나요?

네. [메시징 게이트웨이](../user-guide/messaging/index.md)를 사용하면 Telegram, Discord, Slack, WhatsApp 또는 Home Assistant를 통해 여러 사용자가 동일한 Hermes Agent 인스턴스와 상호작용할 수 있습니다. 액세스는 허용 목록 (특정 사용자 ID)과 DM 페어링 (처음 메시지를 보낸 사용자가 액세스 권한을 가짐)으로 제어됩니다.

### 메모리와 스킬의 차이는 무엇인가요?

- **메모리**는 **사실**을 저장합니다 — 에이전트가 사용자, 프로젝트, 선호 사항에 대해 알고 있는 내용입니다. 관련성에 따라 메모리가 자동으로 검색됩니다.
- **스킬**은 **절차**를 저장합니다 — 작업을 수행하는 방법을 단계별로 설명하는 지침입니다. 에이전트가 비슷한 작업을 만나면 스킬이 검색됩니다.

둘 다 세션 간에 유지됩니다. 자세한 내용은 [메모리](../user-guide/features/memory.md) 및 [스킬](../user-guide/features/skills.md)을 확인하세요.

### 내 Python 프로젝트에서 사용할 수 있나요?

네. `AIAgent` 클래스를 가져와 Hermes를 프로그래밍 방식으로 사용할 수 있습니다.

```python
from run_agent import AIAgent

agent = AIAgent(model="anthropic/claude-opus-4.7")
response = agent.chat("Explain quantum computing briefly")
```

전체 API 사용법은 [Python 라이브러리 가이드](../user-guide/features/code-execution.md)를 확인하세요.

---

## 문제 해결

### 설치 문제

#### 설치 후 `hermes: command not found`

**원인:** 셸에서 업데이트된 PATH를 다시 불러오지 않았습니다.

**해결 방법:**
```bash
# Reload your shell profile
source ~/.bashrc    # bash
source ~/.zshrc     # zsh

# Or start a new terminal session
```

여전히 작동하지 않으면 설치 위치를 확인하세요.
```bash
which hermes
ls ~/.local/bin/hermes
```

:::tip
설치 프로그램은 `~/.local/bin`을 PATH에 추가합니다. 비표준 셸 설정을 사용한다면 `export PATH="$HOME/.local/bin:$PATH"`을 수동으로 추가하세요.
:::

#### Python 버전이 너무 낮음

**원인:** Hermes에는 Python 3.11 이상이 필요합니다.

**해결 방법:**
```bash
python3 --version   # Check current version

# Install a newer Python
sudo apt install python3.12   # Ubuntu/Debian
brew install python@3.12      # macOS
```

설치 프로그램이 이를 자동으로 처리합니다. 수동 설치 중 이 오류가 표시되면 먼저 Python을 업그레이드하세요.

#### 터미널 명령에서 `node: command not found` (또는 `nvm`, `pyenv`, `asdf`, …) 표시

**원인:** Hermes는 시작 시 `bash -l`을 한 번 실행하여 세션별 환경 스냅샷을 만듭니다. bash 로그인 셸은 `/etc/profile`, `~/.bash_profile`, `~/.profile`을 읽지만 **`~/.bashrc`는 소싱하지 않습니다**. 따라서 여기에 설치되는 도구 (`nvm`, `asdf`, `pyenv`, `cargo`, 사용자 지정 `PATH` 내보내기)는 스냅샷에서 보이지 않습니다. Hermes가 systemd 아래에서 실행되거나 대화형 셸 프로필을 미리 로드하지 않은 최소 셸에서 실행될 때 가장 흔히 발생합니다.

**해결 방법:** Hermes는 기본적으로 `~/.bashrc`를 자동 소싱합니다. 그래도 충분하지 않다면 (예: PATH가 `~/.zshrc`에 있는 zsh 사용자이거나 독립 파일에서 `nvm`을 초기화하는 경우) `~/.hermes/config.yaml`에서 소싱할 추가 파일을 나열하세요.

```yaml
terminal:
  shell_init_files:
    - ~/.zshrc                     # zsh users: pulls zsh-managed PATH into the bash snapshot
    - ~/.nvm/nvm.sh                # direct nvm init (works regardless of shell)
    - /etc/profile.d/cargo.sh      # system-wide rc files
  # When this list is set, the default ~/.bashrc auto-source is NOT added —
  # include it explicitly if you want both:
  #   - ~/.bashrc
  #   - ~/.zshrc
```

없는 파일은 자동으로 건너뜁니다. 소싱은 bash에서 수행되므로 zsh 전용 문법에 의존하는 파일에서는 오류가 발생할 수 있습니다. 문제가 우려된다면 전체 rc 파일 대신 PATH를 설정하는 부분 (예: nvm의 `nvm.sh` 직접 소싱)만 소싱하세요.

자동 소싱 동작을 비활성화하려면 (엄격한 로그인 셸 의미만 사용):

```yaml
terminal:
  auto_source_bashrc: false
```

#### `uv: command not found`

**원인:** `uv` 패키지 관리자가 설치되지 않았거나 PATH에 없습니다.

**해결 방법:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### 설치 중 권한 거부 오류

**원인:** 설치 디렉터리에 쓸 권한이 부족합니다.

**해결 방법:**
```bash
# Don't use sudo with the installer — it installs to ~/.local/bin
# If you previously installed with sudo, clean up:
sudo rm /usr/local/bin/hermes
# Then re-run the standard installer
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

---

### 제공자 및 모델 문제

#### 에이전트가 "Hermes policy" 또는 "Hermes guardrails"가 요청을 거부했다고 말함

모델은 요청을 거부한 이유를 안정적으로 식별할 수 없습니다. 거부가 어시스턴트의 말로만 나타난다면, 숨겨진 Hermes 런타임 정책이 원인이라는 주장은 환각으로 생성된 설명이거나 선택한 모델 또는 제공자가 적용한 제한일 수 있습니다.

Hermes의 적용은 명시적입니다. 차단된 도구 작업은 거부된 명령 또는 경로를 명시하는 도구 오류를 반환하며, 승인이 필요한 작업은 승인 프롬프트를 표시합니다. Hermes는 이러한 실행 제어를 일반적인 콘텐츠 거부 계층으로 조용히 변환하지 않습니다. Amazon Bedrock Guardrails처럼 구성된 제공자 수준의 제어는 여전히 적용될 수 있습니다.

출처를 분리하려면 다음을 수행하세요.

1. `/status`를 실행하여 활성 모델과 제공자를 확인합니다.
2. 거부에 실제 Hermes 도구 오류 또는 승인 프롬프트가 포함되어 있는지 확인합니다. 말로만 설명되어 있다면 모델의 원인 설명을 런타임 증거로 취급하지 마세요.
3. 다른 구성 모델 또는 제공자로 새 세션에서 다시 시도합니다. 모델에 따라 거부가 달라진다면 이는 Hermes 실행 제어가 아니라 모델/제공자 동작입니다.
4. 명시적 도구 오류가 표시되면 문제를 보고할 때 정확한 오류 문구를 사용합니다.

[보안](/user-guide/security)에서 Hermes의 문서화된 실행 제어를, [제공자](/integrations/providers)에서 제공자 구성을 확인하세요.

#### `/model`에 한 제공자만 표시됨 / 제공자를 전환할 수 없음

**원인:** `/model` (채팅 세션 내부)은 **이미 구성한** 제공자 사이에서만 전환할 수 있습니다. OpenRouter만 설정했다면 `/model`에 표시되는 것도 OpenRouter뿐입니다.

**해결 방법:** 세션을 종료하고 터미널에서 `hermes model`을 사용하여 새 제공자를 추가하세요.

```bash
# Exit the Hermes chat session first (Ctrl+C or /quit)

# Run the full provider setup wizard
hermes model

# This lets you: add providers, run OAuth, enter API keys, configure endpoints
```

`hermes model`을 통해 새 제공자를 추가한 후 새 채팅 세션을 시작하면 `/model`에 구성된 모든 제공자가 표시됩니다.

:::tip 빠른 참조
| 원하는 작업 | 사용할 명령 |
|-----------|-----|
| 새 제공자 추가 | `hermes model` (터미널에서) |
| API 키 입력/변경 | `hermes model` (터미널에서) |
| 세션 중 모델 전환 | `/model <name>` (세션 내부) |
| 다른 구성 제공자로 전환 | `/model provider:model` (세션 내부) |
:::

#### API 키가 작동하지 않음

**원인:** 키가 없거나, 만료되었거나, 잘못 설정되었거나, 잘못된 제공자용 키입니다.

**해결 방법:**
```bash
# Check your configuration
hermes config show

# Re-configure your provider
hermes model

# Or set directly
hermes config set OPENROUTER_API_KEY sk-or-v1-xxxxxxxxxxxx
```

:::warning
키가 제공자와 일치하는지 확인하세요. OpenAI 키는 OpenRouter에서 작동하지 않으며 그 반대도 마찬가지입니다. 충돌하는 항목이 있는지 `~/.hermes/.env`를 확인하세요.
:::

#### 모델을 사용할 수 없음 / 모델을 찾을 수 없음

**원인:** 모델 식별자가 잘못되었거나 제공자에서 사용할 수 없습니다.

**해결 방법:**
```bash
# List available models for your provider
hermes model

# Set a valid model
hermes config set HERMES_MODEL anthropic/claude-opus-4.7

# Or specify per-session
hermes chat --model openrouter/meta-llama/llama-3.1-70b-instruct
```

#### 속도 제한 (429 오류)

**원인:** 제공자의 속도 제한을 초과했습니다.

**해결 방법:** 잠시 기다린 후 다시 시도하세요. 지속적으로 사용한다면 다음을 고려하세요.
- 제공자 요금제 업그레이드
- 다른 모델 또는 제공자로 전환
- `hermes chat --provider <alternative>`를 사용하여 다른 백엔드로 라우팅

#### 컨텍스트 길이 초과

**원인:** 대화가 모델의 컨텍스트 창에 비해 너무 길어졌거나 Hermes가 모델의 컨텍스트 길이를 잘못 감지했습니다.

**해결 방법:**
```bash
# Compress the current session
/compress

# Or start a fresh session
hermes chat

# Use a model with a larger context window
hermes chat --model openrouter/google/gemini-3-flash-preview
```

첫 번째 긴 대화에서 이 문제가 발생한다면 Hermes가 모델의 컨텍스트 길이를 잘못 알고 있을 수 있습니다. 감지된 값을 확인하세요.

CLI 시작 줄을 확인하면 감지된 컨텍스트 길이가 표시됩니다 (예: `📊 Context limit: 128000 tokens`). 세션 중 `/usage`로도 확인할 수 있습니다.

컨텍스트 감지를 수정하려면 명시적으로 설정하세요.

```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  context_length: 131072  # your model's actual context window
```

사용자 지정 엔드포인트의 경우 제공자 항목에서 모델별로 추가하세요.

```yaml
providers:
  my-server:
    api: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
```

(이전 구성에서는 레거시 `custom_providers:` 목록을 사용하며, 여전히 지원되고 `providers:`로 자동 마이그레이션됩니다.)

자동 감지 방식과 모든 재정의 옵션은 [컨텍스트 길이 감지](../integrations/providers.md#context-length-detection)를 확인하세요.

---

### 터미널 문제

#### 위험한 명령으로 차단됨

**원인:** Hermes가 잠재적으로 파괴적인 명령 (예: `rm -rf`, `DROP TABLE`)을 감지했습니다. 이는 안전 기능입니다.

**해결 방법:** 프롬프트가 표시되면 명령을 검토하고 `y`를 입력하여 승인하세요. 다음 방법도 사용할 수 있습니다.
- 에이전트에게 더 안전한 대안을 사용하도록 요청
- [보안 문서](../user-guide/security.md)에서 위험 패턴의 전체 목록 확인

:::tip
이는 의도된 동작입니다. Hermes는 파괴적인 명령을 조용히 실행하지 않습니다. 승인 프롬프트에는 실제로 실행될 내용이 정확히 표시됩니다.
:::

#### 메시징 게이트웨이에서 `sudo`가 작동하지 않음

**원인:** 메시징 게이트웨이는 대화형 터미널 없이 실행되므로 `sudo`가 비밀번호를 요청할 수 없습니다.

**해결 방법:**
- 메시징에서 `sudo`를 피하고 에이전트에게 대안을 찾도록 요청
- `sudo`가 반드시 필요하다면 `/etc/sudoers`에서 특정 명령에 대해 비밀번호 없는 sudo 구성
- 또는 관리 작업에는 터미널 인터페이스인 `hermes chat` 사용

#### Docker 백엔드가 연결되지 않음

**원인:** Docker 데몬이 실행 중이 아니거나 사용자에게 권한이 없습니다.

**해결 방법:**
```bash
# Check Docker is running
docker info

# Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker run hello-world
```

---

### 메시징 문제

#### 봇이 메시지에 응답하지 않음

**원인:** 봇이 실행 중이 아니거나, 인증되지 않았거나, 사용자가 허용 목록에 없습니다.

**해결 방법:**
```bash
# Check if the gateway is running
hermes gateway status

# Start the gateway
hermes gateway start

# Check logs for errors
cat ~/.hermes/logs/gateway.log | tail -50
```

#### 메시지가 전달되지 않음

**원인:** 네트워크 문제, 만료된 봇 토큰 또는 플랫폼 웹훅 설정 오류입니다.

**해결 방법:**
- `hermes gateway setup`으로 봇 토큰이 유효한지 확인
- 게이트웨이 로그 확인: `cat ~/.hermes/logs/gateway.log | tail -50`
- 웹훅 기반 플랫폼 (Slack, WhatsApp)의 경우 서버가 공개적으로 액세스 가능한지 확인

#### 허용 목록이 혼란스러움 — 누가 봇과 대화할 수 있나요?

**원인:** 인증 모드가 액세스 권한을 결정합니다.

**해결 방법:**

| 모드 | 작동 방식 |
|------|-------------|
| **허용 목록** | 구성에 나열된 사용자 ID만 상호작용 가능 |
| **DM 페어링** | DM으로 메시지를 보내는 첫 사용자가 독점 액세스 권한을 가짐 |
| **공개** | 누구나 상호작용 가능 (프로덕션에서는 권장하지 않음) |

게이트웨이 설정 아래 `~/.hermes/config.yaml`에서 구성하세요. [메시징 문서](../user-guide/messaging/index.md)를 확인하세요.

#### 게이트웨이가 시작되지 않음

**원인:** 누락된 종속성, 포트 충돌 또는 잘못 구성된 토큰입니다.

**해결 방법:**
```bash
# Install core messaging gateway dependencies
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"  # Telegram, Discord, Slack, and shared gateway deps

# Check for port conflicts
lsof -i :8080

# Verify configuration
hermes config show
```

#### WSL: 게이트웨이가 계속 연결 해제되거나 `hermes gateway start`가 실패함

**원인:** WSL의 systemd 지원은 안정적이지 않습니다. 많은 WSL2 설치에서 systemd가 활성화되어 있지 않으며, 활성화되어 있더라도 WSL 재시작이나 Windows 유휴 종료 후 서비스가 유지되지 않을 수 있습니다.

**해결 방법:** systemd 서비스 대신 포그라운드 모드를 사용하세요.

```bash
# Option 1: Direct foreground (simplest)
hermes gateway run

# Option 2: Persistent via tmux (survives terminal close)
tmux new -s hermes 'hermes gateway run'
# Reattach later: tmux attach -t hermes

# Option 3: Background via nohup
nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &
```

그래도 systemd를 사용하려면 활성화되어 있는지 확인하세요.

1. `/etc/wsl.conf`를 엽니다 (없으면 생성).
2. 다음을 추가합니다.
   ```ini
   [boot]
   systemd=true
   ```
3. PowerShell에서 `wsl --shutdown`을 실행합니다.
4. WSL 터미널을 다시 엽니다.
5. `systemctl is-system-running`이 `running` 또는 `degraded`라고 표시되는지 확인합니다.

:::tip Windows 부팅 시 자동 시작
안정적인 자동 시작을 위해 Windows 작업 스케줄러를 사용하여 로그인 시 WSL과 게이트웨이를 실행하세요.
1. 로그인 시 `wsl -d Ubuntu -- bash -lc 'hermes gateway run'`을 실행하는 작업을 만듭니다.
2. 사용자 로그온 시 트리거되도록 설정합니다.
:::

#### macOS: 게이트웨이에서 Node.js / ffmpeg / 기타 도구를 찾지 못함

**원인:** launchd 서비스는 Homebrew, nvm, cargo 또는 기타 사용자 설치 도구 디렉터리를 포함하지 않는 최소 PATH (`/usr/bin:/bin:/usr/sbin:/sbin`)를 상속합니다. 이로 인해 WhatsApp 브리지 (`node not found`) 또는 음성 전사 (`ffmpeg not found`)가 자주 작동하지 않습니다.

**해결 방법:** 게이트웨이는 `hermes gateway install`을 실행할 때 셸 PATH를 캡처합니다. 게이트웨이 설정 후 도구를 설치했다면 설치를 다시 실행하여 업데이트된 PATH를 캡처하세요.

```bash
hermes gateway install    # Re-snapshots your current PATH
hermes gateway start      # Detects the updated plist and reloads
```

plist에 올바른 PATH가 있는지 확인할 수 있습니다.
```bash
/usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:PATH" \
  ~/Library/LaunchAgents/ai.hermes.gateway.plist
```

---

### 성능 문제

#### 응답이 느림

**원인:** 대형 모델, 원격 API 서버 또는 많은 도구가 포함된 무거운 시스템 프롬프트입니다.

**해결 방법:**
- 더 빠르고 작은 모델을 사용해 보세요: `hermes chat --model openrouter/meta-llama/llama-3.1-8b-instruct`
- 활성 도구 세트를 줄이세요: `hermes chat -t "terminal"`
- 제공자까지의 네트워크 지연 시간을 확인하세요.
- 로컬 모델의 경우 GPU VRAM이 충분한지 확인하세요.

#### 토큰 사용량이 많음

**원인:** 긴 대화, 장황한 시스템 프롬프트 또는 누적되는 많은 도구 호출입니다.

**해결 방법:**
```bash
# See exactly what the fixed prompt costs — breakdown by block
# (system prompt, skills index, memory, tool schemas). Runs offline.
hermes prompt-size

# Compress the conversation to reduce tokens
/compress

# Check session token usage
/usage
```

입력하기 전부터 기준 사용량이 높다면 이는 고정 프롬프트 예산입니다. 모든 호출에 전송되는 시스템 프롬프트와 도구 스키마가 포함됩니다. [`hermes prompt-size`](/reference/cli-commands#hermes-prompt-size)를 실행하여 측정한 다음, 사용하지 않는 도구 세트를 비활성화하고 (`hermes tools`) 필요하지 않은 스킬을 제거하거나 비활성화하세요 (`hermes skills`).

:::tip
긴 세션에서는 `/compress`를 정기적으로 사용하세요. 대화 기록을 요약하고 컨텍스트를 유지하면서 토큰 사용량을 크게 줄입니다.
:::

#### 세션이 너무 길어짐

**원인:** 장시간 대화로 메시지와 도구 출력이 누적되어 컨텍스트 제한에 가까워졌습니다.

**해결 방법:**
```bash
# Compress current session (preserves key context)
/compress

# Start a new session with a reference to the old one
hermes chat

# Resume a specific session later if needed
hermes chat --continue
```

---

### MCP 문제

#### MCP 서버가 연결되지 않음

**원인:** 서버 바이너리를 찾을 수 없거나, 명령 경로가 잘못되었거나, 런타임이 누락되었습니다.

**해결 방법:**
```bash
# Ensure MCP dependencies are installed (already included in standard install)
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

# For npm-based servers, ensure Node.js is available
node --version
npx --version

# Test the server manually
npx -y @modelcontextprotocol/server-filesystem /tmp
```

`~/.hermes/config.yaml`의 MCP 구성을 확인하세요.
```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
```

#### MCP 서버의 도구가 표시되지 않음

**원인:** 서버는 시작되었지만 도구 검색에 실패했거나, 구성에서 도구가 필터링되었거나, 서버가 예상한 MCP 기능을 지원하지 않습니다.

**해결 방법:**
- MCP 연결 오류가 있는지 게이트웨이/에이전트 로그 확인
- 서버가 `tools/list` RPC 메서드에 응답하는지 확인
- 해당 서버에서 `tools.include`, `tools.exclude`, `tools.resources`, `tools.prompts` 또는 `enabled` 설정 검토
- 리소스/프롬프트 유틸리티 도구는 세션이 실제로 해당 기능을 지원할 때만 등록된다는 점에 유의
- 구성을 변경한 후 `/reload-mcp` 사용

```bash
# Verify MCP servers are configured
hermes config show | grep -A 12 mcp_servers

# Restart Hermes or reload MCP after config changes
hermes chat
```

다음 문서도 참고하세요.
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [Hermes에서 MCP 사용](/guides/use-mcp-with-hermes)
- [MCP 구성 참조](/reference/mcp-config-reference)

#### MCP 타임아웃 오류

**원인:** MCP 서버가 응답하는 데 너무 오래 걸리거나 실행 중 충돌했습니다.

**해결 방법:**
- 지원되는 경우 MCP 서버 구성에서 타임아웃을 늘리세요.
- MCP 서버 프로세스가 아직 실행 중인지 확인하세요.
- 원격 HTTP MCP 서버의 경우 네트워크 연결을 확인하세요.

:::warning
MCP 서버가 요청 중간에 충돌하면 Hermes는 타임아웃을 보고합니다. 근본 원인을 진단하려면 Hermes 로그뿐 아니라 서버 자체의 로그도 확인하세요.
:::

## 프로필

### 프로필은 단순히 HERMES_HOME을 설정하는 것과 어떻게 다른가요?

프로필은 `HERMES_HOME` 위에 구축된 관리 계층입니다. 명령을 실행할 때마다 `HERMES_HOME=/some/path`를 수동으로 설정할 *수도* 있지만, 프로필은 디렉터리 구조 생성, 셸 별칭 (`hermes-work`) 생성, 활성 프로필 추적 (`~/.hermes/active_profile`), 모든 프로필 간 스킬 업데이트 동기화를 처리합니다. 또한 탭 완성 기능과 통합되므로 경로를 기억할 필요가 없습니다.

### 두 프로필이 같은 봇 토큰을 공유할 수 있나요?

아니요. 각 메시징 플랫폼 (Telegram, Discord 등)은 봇 토큰에 대한 독점 액세스를 요구합니다. 두 프로필이 동시에 같은 토큰을 사용하려 하면 두 번째 게이트웨이는 연결에 실패합니다. 프로필마다 별도의 봇을 만드세요. Telegram의 경우 [@BotFather](https://t.me/BotFather)에게 이야기하여 봇을 추가로 만들 수 있습니다.

### 프로필은 메모리나 세션을 공유하나요?

아니요. 각 프로필에는 고유한 메모리 저장소, 세션 데이터베이스, 스킬 디렉터리가 있습니다. 서로 완전히 격리됩니다. 기존 메모리와 세션으로 새 프로필을 시작하려면 `hermes profile create newname --clone-all`을 사용하여 현재 프로필의 모든 항목을 복사하거나, 특정 원본 프로필에서 복사하려면 `--clone-from <profile>`을 추가하세요.

이러한 격리가 필요한 이유는 동일한 프로필 또는 Hermes 홈에 두 에이전트를 절대 실행해서는 안 되기 때문이기도 합니다. 두 에이전트 모두 메모리를 자동으로 기록하고 세션 시작 시 서로의 기록을 로드하므로, 세션이 계속될수록 저장된 상태가 저하됩니다. 에이전트 하나당 프로필 하나를 사용하세요. 에이전트 간에 메모리를 실제로 공유하려면 [외부 메모리 제공자](/user-guide/features/memory-providers)를 사용하세요.

### `hermes update`를 실행하면 어떻게 되나요?

`hermes update`는 최신 코드를 가져오고 종속성을 **한 번** 다시 설치합니다 (프로필별로 설치하지 않음). 그런 다음 업데이트된 스킬을 모든 프로필에 자동으로 동기화합니다. 컴퓨터의 모든 프로필에 적용되므로 `hermes update`는 한 번만 실행하면 됩니다.

### 몇 개의 프로필을 실행할 수 있나요?

엄격한 제한은 없습니다. 각 프로필은 `~/.hermes/profiles/` 아래의 디렉터리일 뿐입니다. 실제 한도는 디스크 공간과 시스템이 처리할 수 있는 동시 게이트웨이 수에 따라 달라집니다 (각 게이트웨이는 가벼운 Python 프로세스입니다). 프로필 수십 개를 실행해도 괜찮으며, 유휴 프로필은 리소스를 사용하지 않습니다.

---

## 워크플로 및 패턴

### 작업마다 다른 모델 사용 (다중 모델 워크플로)

**상황:** 평소에는 GPT-5.4를 사용하지만 Gemini 또는 Grok이 소셜 미디어 콘텐츠를 더 잘 작성합니다. 매번 수동으로 모델을 전환하는 것은 번거롭습니다.

**해결 방법: 위임 구성.** Hermes는 서브에이전트를 다른 모델로 자동 라우팅할 수 있습니다. `~/.hermes/config.yaml`에서 다음을 설정하세요.

```yaml
delegation:
  model: "google/gemini-3-flash-preview"   # subagents use this model
  provider: "openrouter"                    # provider for subagents
```

이제 Hermes에 "X에 대한 Twitter 스레드를 작성해 줘"라고 말하고 `delegate_task` 서브에이전트를 생성하면, 해당 서브에이전트는 주 모델 대신 Gemini에서 실행됩니다. 기본 대화는 GPT-5.4로 유지됩니다.

프롬프트에서 명시적으로 요청할 수도 있습니다. *"제품 출시 소셜 미디어 게시물을 작성하는 작업을 위임해 줘. 실제 작성은 서브에이전트를 사용해."* 에이전트는 위임 구성을 자동으로 사용하는 `delegate_task`를 호출합니다.

위임 없이 일회성 모델 전환을 하려면 CLI에서 `/model`을 사용하세요.

```bash
/model google/gemini-3-flash-preview    # switch for this session
# ... write your content ...
/model openai/gpt-5.4                   # switch back
```

:::warning
각 `/model` 전환은 프롬프트 캐시를 초기화합니다. 캐시 키에 모델이 포함되므로 전환할 때마다 첫 메시지는 전체 대화를 전체 입력 비용으로 다시 읽습니다. 긴 세션에서는 반복적인 양방향 전환 대신 위임 (서브에이전트는 자체적으로 새로운 컨텍스트를 사용) 또는 새 세션을 사용하세요.
:::

위임 작동 방식에 대한 자세한 내용은 [서브에이전트 위임](../user-guide/features/delegation.md)을 확인하세요.

### 하나의 WhatsApp 번호에서 여러 에이전트 실행 (채팅별 바인딩)

**상황:** OpenClaw에서는 특정 WhatsApp 채팅에 여러 독립 에이전트를 연결했습니다. 하나는 가족 쇼핑 목록 그룹용이고 다른 하나는 개인 채팅용이었습니다. Hermes에서도 가능한가요?

**현재 제한:** Hermes 프로필마다 고유한 WhatsApp 번호/세션이 필요합니다. 같은 WhatsApp 번호에서 여러 프로필을 서로 다른 채팅에 연결할 수 없습니다. WhatsApp 브리지 (Baileys)는 번호당 인증된 세션 하나를 사용합니다.

**해결 방법:**

1. **성격 전환 기능이 있는 단일 프로필 사용.** 서로 다른 `AGENTS.md` 컨텍스트 파일을 만들거나 `/personality` 명령을 사용하여 채팅별 동작을 변경하세요. 에이전트는 현재 대화 중인 채팅을 보고 적응할 수 있습니다.

2. **특수 작업에 cron 작업 사용.** 쇼핑 목록 추적기라면 특정 채팅을 모니터링하고 목록을 관리하는 cron 작업을 설정하세요. 별도의 에이전트가 필요하지 않습니다.

3. **별도의 번호 사용.** 진정으로 독립적인 에이전트가 필요하다면 각 프로필을 자체 WhatsApp 번호와 페어링하세요. Google Voice와 같은 서비스의 가상 번호를 사용할 수 있습니다.

4. **대신 Telegram 또는 Discord 사용.** 이러한 플랫폼은 채팅별 바인딩을 더 자연스럽게 지원합니다. 각 Telegram 그룹 또는 Discord 채널은 자체 세션을 가지며, 같은 계정에서 여러 봇 토큰 (프로필당 하나)을 실행할 수 있습니다.

[프로필](../user-guide/profiles.md) 및 [WhatsApp 설정](../user-guide/messaging/whatsapp.md)에서 자세한 내용을 확인하세요.

### Telegram에 표시되는 항목 제어 (로그 및 추론 숨기기)

**상황:** 최종 출력만 보고 싶은데 Telegram에서 게이트웨이 exec 로그, Hermes 추론, 도구 호출 세부 정보를 보고 있습니다.

**해결 방법:** `config.yaml`의 `display.tool_progress` 설정으로 표시되는 도구 활동의 양을 제어할 수 있습니다.

```yaml
display:
  tool_progress: "off"   # options: off, new, all, verbose
```

- **`off`** — 최종 응답만 표시합니다. 도구 호출, 추론, 로그는 표시하지 않습니다.
- **`new`** — 새 도구 호출이 발생할 때 간단한 한 줄로 표시합니다.
- **`all`** — 결과를 포함한 모든 도구 활동을 표시합니다.
- **`verbose`** — 도구 인수와 출력을 포함한 전체 세부 정보를 표시합니다.

메시징 플랫폼에서는 보통 `off` 또는 `new`가 적합합니다. `config.yaml`을 편집한 후 게이트웨이를 재시작해야 변경 사항이 적용됩니다.

활성화된 경우 `/verbose` 명령으로 세션별 설정을 전환할 수도 있습니다.

```yaml
display:
  tool_progress_command: true   # enables /verbose in the gateway
```

### Telegram에서 스킬 관리 (슬래시 명령 제한)

**상황:** Telegram에는 슬래시 명령 100개 제한이 있는데 스킬 때문에 이 수를 초과하고 있습니다. Telegram에서 필요하지 않은 스킬을 비활성화하려 하지만 `hermes skills config` 설정이 적용되지 않는 것 같습니다.

**해결 방법:** `hermes skills config`를 사용하여 플랫폼별로 스킬을 비활성화하세요. 이 명령은 `config.yaml`에 다음을 기록합니다.

```yaml
skills:
  disabled: []                    # globally disabled skills
  platform_disabled:
    telegram: [skill-a, skill-b]  # disabled only on telegram
```

변경 후 **게이트웨이를 재시작**하세요 (`hermes gateway restart` 또는 프로세스를 종료하고 다시 실행). Telegram 봇 명령 메뉴는 시작 시 다시 생성됩니다.

:::tip
Telegram 메뉴의 페이로드 크기 제한을 지키기 위해 설명이 매우 긴 스킬은 40자로 잘립니다. 스킬이 표시되지 않는다면 100개 명령 제한이 아니라 전체 페이로드 크기 문제일 수 있습니다. 사용하지 않는 스킬을 비활성화하면 두 문제 모두 해결하는 데 도움이 됩니다.
:::

### 공유 스레드 세션 (여러 사용자, 하나의 대화)

**상황:** 여러 사람이 봇을 멘션하는 Telegram 또는 Discord 스레드가 있습니다. 각 사용자별로 별도 세션을 만드는 대신 해당 스레드의 모든 멘션을 하나의 공유 대화로 만들고 싶습니다.

**현재 동작:** 대부분의 플랫폼에서 Hermes는 사용자 ID를 기준으로 세션을 생성하므로 각자 고유한 대화 컨텍스트를 갖습니다. 이는 개인정보 보호와 컨텍스트 격리를 위한 의도된 설계입니다.

**해결 방법:**

1. **Slack 사용.** Slack 세션은 사용자가 아니라 스레드를 기준으로 합니다. 같은 스레드의 여러 사용자가 하나의 대화를 공유하므로 설명한 동작에 정확히 맞습니다. 가장 자연스러운 선택입니다.

2. **한 명의 사용자와 그룹 채팅 사용.** 한 사람이 질문을 전달하는 지정 운영자라면 세션이 통합된 상태로 유지됩니다. 다른 사람들은 대화를 지켜볼 수 있습니다.

3. **Discord 채널 사용.** Discord 세션은 채널을 기준으로 하므로 같은 채널의 모든 사용자가 컨텍스트를 공유합니다. 공유 대화 전용 채널을 사용하세요.

### Hermes를 다른 컴퓨터로 내보내기

**상황:** 한 컴퓨터에서 스킬, cron 작업, 메모리를 구축했으며 새 전용 Linux 장비로 옮기고 싶습니다.

**해결 방법:**

1. 새 컴퓨터에 Hermes Agent를 설치합니다.
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

2. **원본 컴퓨터에서** 전체 백업을 만듭니다.
   ```bash
   hermes backup
   ```
   이 명령은 전체 `~/.hermes/` 디렉터리 (구성, API 키, 메모리, 스킬, 세션, 프로필)를 zip으로 만들어 홈 디렉터리에 `~/hermes-backup-<timestamp>.zip`으로 저장합니다.

3. zip 파일을 새 컴퓨터로 복사하고 가져옵니다.
   ```bash
   # On the source machine
   scp ~/hermes-backup-<timestamp>.zip newmachine:~/

   # On the new machine
   hermes import ~/hermes-backup-<timestamp>.zip
   ```

4. 새 컴퓨터에서 `hermes setup`을 실행하여 API 키와 제공자 구성이 작동하는지 확인합니다.

### 단일 프로필을 다른 컴퓨터로 이동

**상황:** 전체 설치가 아니라 특정 프로필 하나만 이동하거나 공유하고 싶습니다.

```bash
# On the source machine
hermes profile export work ./work-backup.tar.gz

# Copy the file to the target machine, then:
hermes profile import ./work-backup.tar.gz work
```

가져온 프로필에는 내보낸 프로필의 모든 구성, 메모리, 세션, 스킬이 포함됩니다. 새 컴퓨터의 설정이 다르면 경로를 업데이트하거나 제공자에 다시 인증해야 할 수 있습니다.

### `hermes backup`과 `hermes profile export` 비교

| 기능 | `hermes backup` | `hermes profile export` |
| :--- | :--- | :--- |
| **사용 사례** | **전체 컴퓨터 마이그레이션** | **특정 프로필 포팅/공유** |
| **범위** | 전역 (전체 `~/.hermes` 디렉터리) | 로컬 (단일 프로필 디렉터리) |
| **포함 항목** | 모든 프로필, 전역 구성, API 키, 세션 | 단일 프로필: SOUL.md, 메모리, 세션, 스킬 |
| **자격 증명** | **포함** (`.env` 및 `auth.json`) | **제외** (안전한 공유를 위해 제거) |
| **형식** | `.zip` | `.tar.gz` |

**수동 대안 (rsync):** 파일을 직접 복사하려면 코드 저장소를 제외하세요.
```bash
rsync -av --exclude='hermes-agent' ~/.hermes/ newmachine:~/.hermes/
```

:::tip
`hermes backup`은 Hermes가 활발히 실행 중일 때도 일관된 스냅샷을 생성합니다. 복원된 아카이브에서는 `gateway.pid` 및 `cron.pid`와 같은 컴퓨터 로컬 런타임 파일이 제외됩니다.
:::

### 설치 후 셸을 다시 불러올 때 권한 거부

**상황:** Hermes 설치 프로그램을 실행한 후 `source ~/.zshrc`에서 권한 거부 오류가 발생합니다.

**원인:** 대개 `~/.zshrc` (또는 `~/.bashrc`)의 파일 권한이 잘못되었거나 설치 프로그램이 파일에 제대로 기록하지 못했기 때문입니다. Hermes만의 문제는 아니며 셸 구성 권한 문제입니다.

**해결 방법:**
```bash
# Check permissions
ls -la ~/.zshrc

# Fix if needed (should be -rw-r--r-- or 644)
chmod 644 ~/.zshrc

# Then reload
source ~/.zshrc

# Or just open a new terminal window — it picks up PATH changes automatically
```

설치 프로그램이 PATH 줄을 추가했지만 권한이 잘못된 경우 수동으로 추가할 수 있습니다.
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

### 첫 에이전트 실행에서 오류 400

**상황:** 설정은 정상적으로 완료되었지만 첫 채팅 시도가 HTTP 400으로 실패합니다.

**원인:** 대개 모델 이름이 일치하지 않기 때문입니다. 구성된 모델이 제공자에 없거나 API 키에 해당 모델을 사용할 권한이 없습니다.

**해결 방법:**
```bash
# Check what model and provider are configured
hermes config show | head -20

# Re-run model selection
hermes model

# Or test with a known-good model
hermes chat -q "hello" --model anthropic/claude-opus-4.7
```

OpenRouter를 사용한다면 API 키에 크레딧이 있는지 확인하세요. OpenRouter의 400 오류는 모델에 유료 요금제가 필요하거나 모델 ID에 오타가 있다는 의미인 경우가 많습니다.

---

## 여전히 해결되지 않나요?

문제가 여기에 설명되어 있지 않다면 다음을 이용하세요.

1. **기존 이슈 검색:** [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
2. **커뮤니티에 질문:** [Nous Research Discord](https://discord.gg/nousresearch)
3. **버그 보고:** 운영체제, Python 버전 (`python3 --version`), Hermes 버전 (`hermes --version`), 전체 오류 메시지를 포함하세요.
