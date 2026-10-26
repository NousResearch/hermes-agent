---
sidebar_position: 1
title: "빠른 시작"
description: "Hermes Agent와 나누는 첫 대화 — 설치부터 대화까지 5분 이내"
---

# 빠른 시작

이 가이드는 처음부터 실제 사용을 견딜 수 있는 작동하는 Hermes 설정까지 안내합니다. 설치하고, 제공자를 선택하고, 채팅이 작동하는지 확인하고, 문제가 발생했을 때 정확히 무엇을 해야 하는지 알아보세요.

## 영상으로 보고 싶나요?

**Onchain AI Garage**에서 설치, 설정, 기본 명령을 다루는 마스터클래스 영상을 준비했습니다. 영상으로 따라 하고 싶다면 이 페이지와 함께 보기 좋은 자료입니다. 더 많은 내용은 전체 [Hermes Agent 튜토리얼 및 사용 사례](https://www.youtube.com/playlist?list=PLmpUb_PWAkDxewld5ZYyKifuHxgIbiq2d) 재생목록을 참고하세요.

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', maxWidth: '100%', marginBottom: '1.5rem'}}>
  <iframe
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}}
    src="https://www.youtube-nocookie.com/embed/R3YOGfTBcQg"
    title="Hermes Agent 마스터클래스: 설치, 설정, 기본 명령"
    frameBorder="0"
    allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
  ></iframe>
</div>

## 이 문서의 대상

- 처음 시작하며 작동하는 설정까지 가장 빠른 경로를 원하는 분
- 설정 실수로 시간을 낭비하지 않고 제공자를 바꾸려는 분
- 팀, 봇 또는 상시 실행 워크플로를 위해 Hermes를 설정하려는 분
- "설치했는데 여전히 아무것도 하지 않는다"는 상황에 지친 분

## 가장 빠른 경로

목표에 맞는 행을 선택하세요.

| 목표 | 먼저 할 일 | 그다음 할 일 |
|---|---|---|
| 내 컴퓨터에서 Hermes를 작동시키고 싶다 | `hermes setup` | 실제 채팅을 실행하고 응답하는지 확인 |
| 사용할 제공자를 이미 알고 있다 | `hermes model` | 설정을 저장한 다음 채팅 시작 |
| 봇 또는 상시 실행 설정을 원한다 | CLI가 작동한 후 `hermes gateway setup` | Telegram, Discord, Slack 또는 다른 플랫폼 연결 |
| 로컬 또는 자체 호스팅 모델을 원한다 | `hermes model` → 사용자 지정 엔드포인트 | 엔드포인트, 모델 이름, 컨텍스트 길이 확인 |
| 여러 제공자 간 폴백을 원한다 | 먼저 `hermes model` | 기본 채팅이 작동한 후에만 라우팅 및 폴백 추가 |

**경험칙:** Hermes가 일반 채팅을 완료하지 못한다면 아직 기능을 더 추가하지 마세요. 먼저 깔끔한 대화 하나가 작동하게 한 다음 게이트웨이, cron, 스킬, 음성 또는 라우팅을 추가하세요.

---

## 1. Hermes Agent 설치
### macOS 또는 Windows에서 Hermes Desktop 설치 프로그램 사용(권장)
명령줄 애플리케이션과 데스크톱 애플리케이션을 쉽게 설치하려면 웹사이트에서 [Hermes Desktop 설치 프로그램을 다운로드](https://hermes-agent.nousresearch.com/)한 후 실행하세요.

### Hermes Desktop 없이 설치:
Hermes Desktop 없이 명령줄만 설치하려면 다음을 실행하세요.

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (네이티브)

PowerShell에서 실행하세요.
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

:::tip Android / Termux
휴대폰에 설치한다면, 테스트된 수동 설치 경로와 지원되는 추가 기능, 현재 Android 관련 제한 사항을 설명하는 전용 [Termux 가이드](./termux.md)를 참고하세요.
:::

설치가 끝나면 셸을 다시 불러오세요.

```bash
source ~/.bashrc   # or source ~/.zshrc
```

자세한 설치 옵션, 사전 요구 사항, 문제 해결 방법은 [설치 가이드](./installation.md)를 참고하세요.

## 2. 제공자 선택

설정에서 가장 중요한 단계입니다. `hermes model`을 사용하면 대화형으로 선택 과정을 진행할 수 있습니다.

```bash
hermes model
```

:::tip 가장 쉬운 경로: Nous Portal
하나의 구독으로 300개 이상의 모델과 [Tool Gateway](../user-guide/features/tool-gateway.md)(웹 검색, 이미지 생성, TTS, 클라우드 브라우저)를 이용할 수 있습니다. 새로 설치한 경우 다음을 실행하세요.

```bash
hermes setup --portal
```

이 명령 하나로 로그인하고, Nous를 제공자로 설정하며, Tool Gateway를 켭니다.
:::

:::info 설정 모드
새로 설치한 경우 `hermes setup`은 세 가지 모드를 제공합니다.

- **빠른 설정( Nous Portal )** — 무료 OAuth 로그인으로 API 키가 필요 없으며, 모델과 Tool Gateway 도구를 설정합니다. 가장 빠른 경로로 권장됩니다.
- **전체 설정** — 모든 제공자, 도구, 옵션을 직접 하나씩 설정합니다(자신의 키 사용).
- **빈 상태로 시작** — 에이전트를 실행하는 데 필요한 최소한의 항목인 **provider & model, File Operations toolset, Terminal toolset**을 제외한 모든 항목이 **꺼진** 상태로 시작합니다. 웹, 브라우저, 코드 실행, 비전, 메모리, 위임, cron, 스킬, 플러그인, MCP 서버가 없으며 압축, 체크포인트, 스마트 라우팅, 메모리 캡처도 모두 비활성화됩니다. 최소 기준 설정이 적용된 후 다음 두 경로 중 하나를 선택합니다. **모든 항목을 비활성화한 채 시작**(최소 에이전트로 지금 완료)하거나 **모든 설정을 단계별로 진행**(도구, 스킬, 플러그인, MCP, 메시징을 선택적으로 활성화)합니다. 최소한의 완전 제어 에이전트를 원하고 필요한 것만 활성화할 계획이라면 이 모드를 선택하세요.

빈 상태로 시작은 `platform_toolsets.cli` 목록과 `agent.disabled_toolsets`를 명시적으로 기록하므로, 선택하지 않은 항목은 `hermes update` 이후에도 절대 로드되지 않습니다. 나중에 `hermes tools`로 무엇이든 다시 활성화하고, `hermes skills opt-in --sync`로 스킬을 준비하거나, `hermes setup agent`로 설정을 조정할 수 있습니다.
:::

권장 기본값:

| 제공자 | 설명 | 설정 방법 |
|----------|-----------|---------------|
| **Nous Portal** | 구독 기반, 설정 불필요 | `hermes model`을 통한 OAuth 로그인 |
| **OpenAI Codex** | ChatGPT 또는 Codex 구독, Codex 모델 사용 | `hermes model` → **ChatGPT 또는 Codex Subscription**을 통한 디바이스 코드 인증 |
| **Anthropic** | Claude 모델 직접 사용 — Max 요금제 + 추가 사용 크레딧(OAuth) 또는 종량제 API 키 | `hermes model` → OAuth 로그인(Max + 추가 크레딧 필요) 또는 Anthropic API 키 |
| **OpenRouter** | 여러 모델에 걸친 멀티 제공자 라우팅 | API 키 입력 |
| **Fireworks AI** | 직접 사용하는 OpenAI 호환 모델 API | `FIREWORKS_API_KEY` 설정 |
| **Z.AI** | GLM / Zhipu 호스팅 모델 | `GLM_API_KEY` / `ZAI_API_KEY` 설정(`Z_AI_API_KEY`도 허용) |
| **Kimi / Moonshot** | Moonshot 호스팅 코딩 및 채팅 모델 | `KIMI_API_KEY` 설정(또는 Kimi Coding 전용 `KIMI_CODING_API_KEY`) |
| **Kimi / Moonshot China** | 중국 지역 Moonshot 엔드포인트 | `KIMI_CN_API_KEY` 설정 |
| **Arcee AI** | Trinity 모델 | `ARCEEAI_API_KEY` 설정 |
| **GMI Cloud** | 멀티 모델 직접 API | `GMI_API_KEY` 설정 |
| **Actual Computer** | 호스팅 릴레이 또는 로컬 데몬으로 제공되는 개인 추론 클러스터인 자체 하드웨어 | `ACTUAL_API_KEY`(릴레이) 또는 `ACTUAL_BASE_URL=http://127.0.0.1:8080`(로컬, 키 없음) 설정 |
| **MiniMax (OAuth)** | 브라우저 OAuth를 통한 MiniMax 프런티어 모델 — API 키 불필요(모델 이름은 릴리스마다 `hermes_cli/models.py`에서 변경될 수 있음) | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | 국제 MiniMax 엔드포인트 | `MINIMAX_API_KEY` 설정 |
| **MiniMax China** | 중국 지역 MiniMax 엔드포인트 | `MINIMAX_CN_API_KEY` 설정 |
| **Alibaba Cloud** | DashScope를 통한 Qwen 모델 | `DASHSCOPE_API_KEY` 설정(Qwen Coding Plan은 `ALIBABA_CODING_PLAN_API_KEY`도 허용) |
| **Hugging Face** | 통합 라우터를 통한 20개 이상의 오픈 모델(Qwen, DeepSeek, Kimi 등) | `HF_TOKEN` 설정 |
| **AWS Bedrock** | 네이티브 Converse API를 통한 Claude, Nova, Llama, DeepSeek | IAM 역할 또는 `aws configure`([가이드](../guides/aws-bedrock.md)) |
| **Azure Foundry** | Azure AI Foundry 호스팅 모델 | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` 설정 |
| **Google AI Studio** | 직접 API를 통한 Gemini 모델 | `GOOGLE_API_KEY` / `GEMINI_API_KEY` 설정 |
| **xAI** | 직접 API를 통한 Grok 모델 | `XAI_API_KEY` 설정 |
| **xAI Grok OAuth** | SuperGrok / Premium+ 구독, API 키 불필요 | `hermes model` → xAI Grok OAuth |
| **NovitaAI** | 멀티 모델 API 게이트웨이 | `NOVITA_API_KEY` 설정 |
| **StepFun** | Step Plan 모델 | `STEPFUN_API_KEY` 설정 |
| **Xiaomi MiMo** | Xiaomi 호스팅 모델 | `XIAOMI_API_KEY` 설정 |
| **Tencent TokenHub** | Tencent 호스팅 모델 | `TOKENHUB_API_KEY` 설정 |
| **Ollama Cloud** | 관리형 Ollama 호스팅 모델 | `OLLAMA_API_KEY` 설정 |
| **LM Studio** | OpenAI 호환 API를 제공하는 로컬 데스크톱 앱 | `LM_API_KEY` 설정(기본값이 아니면 `LM_BASE_URL`도 설정) |
| **Qwen OAuth** | Qwen Portal 브라우저 OAuth — API 키 불필요 | `hermes model` → Qwen OAuth |
| **Kilo Code** | KiloCode 호스팅 모델 | `KILOCODE_API_KEY` 설정 |
| **OpenCode Zen** | 엄선된 모델에 대한 종량제 이용 | `OPENCODE_ZEN_API_KEY` 설정 |
| **OpenCode Go** | 오픈 모델용 월 10달러 구독 | `OPENCODE_GO_API_KEY` 설정 |
| **DeepSeek** | DeepSeek API 직접 이용 | `DEEPSEEK_API_KEY` 설정 |
| **NVIDIA NIM** | build.nvidia.com 또는 로컬 NIM을 통한 Nemotron 모델 | `NVIDIA_API_KEY` 설정(선택 사항: `NVIDIA_BASE_URL`) |
| **GitHub Copilot** | GitHub Copilot 구독(GPT-5.x, Claude, Gemini 등) | `hermes model`을 통한 OAuth 또는 `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | Copilot ACP 에이전트 백엔드(로컬 `copilot` CLI 실행) | `hermes model`(`copilot` CLI + `copilot login` 필요) |
| **Vercel AI Gateway** | Vercel AI Gateway 라우팅 | `AI_GATEWAY_API_KEY` 설정 |
| **사용자 지정 엔드포인트** | VLLM, SGLang, Ollama 또는 OpenAI 호환 API | 기본 URL + API 키 설정 |

대부분의 처음 사용하는 분들은 제공자를 선택하고, 변경할 이유를 알고 있지 않다면 기본값을 그대로 사용하면 됩니다. 전체 제공자 목록과 환경 변수 및 설정 단계는 [Providers](../integrations/providers.md) 페이지에 있습니다.

:::caution 최소 컨텍스트: 64K 토큰
Hermes Agent는 컨텍스트가 최소 **64,000 토큰**인 모델을 필요로 합니다. 더 작은 윈도우의 모델은 여러 단계의 도구 호출 워크플로에 필요한 작업 메모리를 충분히 유지할 수 없으므로 시작 시 거부됩니다. 대부분의 호스팅 모델(Claude, GPT, Gemini, Qwen, DeepSeek)은 이 조건을 쉽게 충족합니다. 로컬 모델을 실행한다면 컨텍스트 크기를 최소 64K로 설정하세요(예: llama.cpp에서는 `--ctx-size 65536`, Ollama에서는 `-c 65536`).
:::

:::tip
`hermes model`을 사용하면 언제든 제공자를 바꿀 수 있으므로 특정 제공자에 종속되지 않습니다. 지원되는 모든 제공자와 설정 정보를 보려면 [AI Providers](../integrations/providers.md)를 참고하세요.
:::

### 설정 저장 방식

Hermes는 비밀 정보와 일반 설정을 분리합니다.

- **비밀 정보 및 토큰** → `~/.hermes/.env`
- **비밀이 아닌 설정** → `~/.hermes/config.yaml`

값을 올바르게 설정하는 가장 쉬운 방법은 CLI를 이용하는 것입니다.

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

올바른 값이 자동으로 올바른 파일에 저장됩니다.

## 3. 첫 대화 실행

```bash
hermes            # classic CLI
hermes --tui      # modern TUI (recommended)
```

모델, 사용 가능한 도구, 스킬이 표시된 환영 배너가 나타납니다. 구체적이고 확인하기 쉬운 프롬프트를 사용하세요.

:::tip 인터페이스 선택
Hermes에는 두 가지 터미널 인터페이스가 있습니다. 하나는 기존 `prompt_toolkit` CLI이고, 다른 하나는 모달 오버레이, 마우스 선택, 비차단 입력을 지원하는 새로운 [TUI](../user-guide/tui.md)입니다. 두 인터페이스는 세션, 슬래시 명령, 설정을 공유하므로 `hermes`와 `hermes --tui`를 각각 사용해 보세요.
:::

```
Summarize this repo in 5 bullets and tell me what the main entrypoint is.
```

```
Check my current directory and tell me what looks like the main project file.
```

```
Help me set up a clean GitHub PR workflow for this codebase.
```

**성공한 상태:**

- 배너에 선택한 모델/제공자가 표시됨
- Hermes가 오류 없이 응답함
- 필요할 때 도구(터미널, 파일 읽기, 웹 검색)를 사용할 수 있음
- 두 번 이상 대화를 이어가도 정상적으로 작동함

이 단계가 작동한다면 가장 어려운 부분을 통과한 것입니다.

## 4. 세션 작동 확인

다음 단계로 넘어가기 전에 세션 재개가 작동하는지 확인하세요.

```bash
hermes --continue    # Resume the most recent session
hermes -c            # Short form
```

방금 진행한 세션으로 돌아와야 합니다. 그렇지 않다면 같은 프로필을 사용하고 있는지, 세션이 실제로 저장되었는지 확인하세요. 나중에 여러 설정이나 컴퓨터를 오가며 작업할 때 중요한 부분입니다.

## 5. 주요 기능 사용

### 터미널 사용

```
❯ What's my disk usage? Show the top 5 largest directories.
```

에이전트가 대신 터미널 명령을 실행하고 결과를 보여줍니다.

### 슬래시 명령

`/`를 입력하면 모든 명령의 자동 완성 드롭다운이 표시됩니다.

| 명령 | 기능 |
|---------|-------------|
| `/help` | 사용 가능한 모든 명령 표시 |
| `/tools` | 사용 가능한 도구 목록 표시 |
| `/model` | 대화형으로 모델 전환 |
| `/personality pirate` | 재미있는 성격 사용 |
| `/save` | 대화 저장 |

### 여러 줄 입력

`Alt+Enter`, `Ctrl+J` 또는 `Shift+Enter`를 눌러 새 줄을 추가하세요. `Shift+Enter`를 사용하려면 이를 별도의 시퀀스로 보내는 터미널이 필요합니다(기본적으로 Kitty / foot / WezTerm / Ghostty, Kitty 키보드 프로토콜을 활성화한 경우 iTerm2 / Alacritty / VS Code 터미널). `Alt+Enter`와 `Ctrl+J`는 모든 터미널에서 작동합니다.

### 에이전트 중단

에이전트가 너무 오래 걸리면 새 메시지를 입력하고 Enter를 누르세요. 현재 작업을 중단하고 새 지시로 전환합니다. `Ctrl+C`도 사용할 수 있습니다.

## 6. 다음 계층 추가

기본 채팅이 작동한 후에만 진행하세요. 필요한 항목을 선택하세요.

### 봇 또는 공유 어시스턴트

```bash
hermes gateway setup    # Interactive platform configuration
```

[Telegram](/user-guide/messaging/telegram), [Discord](/user-guide/messaging/discord), [Slack](/user-guide/messaging/slack), [WhatsApp](/user-guide/messaging/whatsapp), [Signal](/user-guide/messaging/signal), [Email](/user-guide/messaging/email), [Home Assistant](/user-guide/messaging/homeassistant) 또는 [Microsoft Teams](/user-guide/messaging/teams)를 연결하세요.

### 자동화 및 도구

- `hermes tools` — 플랫폼별 도구 접근 권한 조정
- `hermes skills` — 재사용 가능한 워크플로 탐색 및 설치
- Cron — 봇 또는 CLI 설정이 안정된 후에만 사용

### 샌드박스 터미널

안전을 위해 에이전트를 Docker 컨테이너 또는 원격 서버에서 실행하세요.

```bash
hermes config set terminal.backend docker    # Docker isolation
hermes config set terminal.backend ssh       # Remote server
```

Docker 샌드박스에서는 **송신 자격 증명 주입 프록시**를 활성화할 수도 있습니다. 그러면 샌드박스에는 실제 API 키가 전달되지 않고, 로컬 TLS 가로채기 데몬 뒤에서만 작동하는 불투명한 프록시 토큰만 전달됩니다. [Egress proxy](../user-guide/egress/iron-proxy.md)를 참고하세요. 설정은 `hermes egress setup && hermes egress start`이며, `hermes setup terminal`도 Docker 사용자에게 이 기능을 안내합니다. Modal, SSH, Daytona, Singularity는 아직 연결되어 있지 않습니다.

### 음성 모드

```bash
# From the Hermes install directory (the curl installer placed it at
# ~/.hermes/hermes-agent on Linux/macOS or %LOCALAPPDATA%\hermes\hermes-agent on Windows):
cd ~/.hermes/hermes-agent
uv pip install --python ./venv/bin/python -e ".[voice]"
# Includes faster-whisper for free local speech-to-text
```

그런 다음 CLI에서 `/voice on`을 입력하세요. 녹음하려면 `Ctrl+B`를 누릅니다. [Voice Mode](../user-guide/features/voice-mode.md)를 참고하세요.

### 스킬

스킬은 특정 작업을 수행하는 방법을 Hermes에 알려주는 필요할 때 사용하는 지침 문서입니다. Kubernetes에 배포하기, GitHub PR 열기, 모델 미세 조정, GIF 검색과 같은 작업을 가르칩니다. 각 스킬은 이름, 설명, 단계별 절차가 담긴 `SKILL.md` 파일입니다. 에이전트는 짧은 설명을 무료로 읽고 실제 작업에서 해당 스킬이 필요할 때만 전체 내용을 불러오므로, 스킬을 추가해도 모든 요청이 불필요하게 커지지 않습니다.

Hermes에는 `~/.hermes/skills/`에 이미 설치된 번들 스킬 카탈로그가 함께 제공됩니다. Skills Hub에서 더 추가하거나 직접 작성할 수 있습니다.

**허브에서 탐색 및 설치:**

```bash
hermes skills browse                      # list everything available
hermes skills search kubernetes           # find skills by keyword
hermes skills install openai/skills/k8s   # install one (runs a security scan first)
```

설치 인수는 허브에서 사용하는 `source/path` 형식의 슬러그입니다. `openai/skills/k8s`는 OpenAI 카탈로그에 있는 `k8s` 스킬을 의미합니다. `hermes skills browse`에서 사용할 정확한 슬러그를 확인할 수 있습니다.

**스킬 사용** — 설치된 모든 스킬은 자동으로 슬래시 명령이 됩니다.

```bash
/k8s deploy the staging manifest          # run the skill with a request
/k8s                                       # load it and let Hermes ask what you need
```

CLI와 연결된 모든 메시징 플랫폼에서 작동합니다. 모든 것을 미리 설치할 필요는 없습니다. 일반 대화 중 작업과 일치하는 스킬이 있으면 에이전트가 적절한 번들 스킬을 스스로 선택합니다.

직접 스킬을 작성하는 방법, 외부 스킬 디렉터리, 전체 허브 소스 목록은 [Skills System](../user-guide/features/skills.md)을 참고하세요.

### MCP 서버

```yaml
# Add to ~/.hermes/config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

### 편집기 통합(ACP)

ACP 지원은 표준 `[all]` 추가 기능에 포함되어 있으므로 curl 설치 프로그램에 이미 들어 있습니다. 다음만 실행하세요.

```bash
hermes acp
```

(`[all]` 없이 설치했다면 먼저 `cd ~/.hermes/hermes-agent && uv pip install -e ".[acp]"`를 실행하세요.)

[ACP Editor Integration](../user-guide/features/acp.md)을 참고하세요.

---

## 일반적인 실패 상황

다음은 가장 많은 시간을 낭비하게 만드는 문제입니다.

| 증상 | 가능한 원인 | 해결 방법 |
|---|---|---|
| Hermes는 열리지만 빈 응답이나 깨진 응답을 보냄 | 제공자 인증 또는 모델 선택이 잘못됨 | `hermes model`을 다시 실행하고 제공자, 모델, 인증을 확인 |
| 사용자 지정 엔드포인트가 "작동"하지만 엉뚱한 결과를 반환함 | 기본 URL, 모델 이름이 잘못되었거나 실제로 OpenAI 호환이 아님 | 먼저 별도의 클라이언트에서 엔드포인트 확인 |
| 게이트웨이가 시작되지만 아무도 메시지를 보낼 수 없음 | 봇 토큰, 허용 목록 또는 플랫폼 설정이 완료되지 않음 | `hermes gateway setup`을 다시 실행하고 `hermes gateway status` 확인 |
| `hermes --continue`가 이전 세션을 찾지 못함 | 프로필을 바꿨거나 세션이 저장되지 않음 | `hermes sessions list`를 실행하고 올바른 프로필인지 확인 |
| 모델을 사용할 수 없거나 이상한 폴백 동작이 발생함 | 제공자 라우팅 또는 폴백 설정이 지나치게 공격적임 | 기본 제공자가 안정될 때까지 라우팅을 끈 상태로 유지 |
| `hermes doctor`가 설정 문제를 표시함 | 설정 값이 없거나 오래됨 | 설정을 수정하고 기능을 추가하기 전에 일반 채팅을 다시 테스트 |

## 복구 도구 모음

문제가 있는 것 같을 때는 다음 순서로 사용하세요.

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

이 순서를 따르면 "뭔가 이상한 상태"에서 빠르게 정상 상태로 돌아올 수 있습니다.

---

## 빠른 참조

| 명령 | 설명 |
|---------|-------------|
| `hermes` | 채팅 시작 |
| `hermes model` | LLM 제공자와 모델 선택 |
| `hermes tools` | 플랫폼별 활성화할 도구 설정 |
| `hermes setup` | 전체 설정 마법사(모든 항목을 한 번에 설정) |
| `hermes doctor` | 문제 진단 |
| `hermes update` | 최신 버전으로 업데이트 |
| `hermes gateway` | 메시징 게이트웨이 시작 |
| `hermes --continue` | 마지막 세션 재개 |

## 다음 단계

- **[CLI Guide](../user-guide/cli.md)** — 터미널 인터페이스 완전히 익히기
- **[Configuration](../user-guide/configuration.md)** — 설정 사용자 지정
- **[Messaging Gateway](../user-guide/messaging/index.md)** — Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant, Teams 등 연결
- **[Tools & Toolsets](../user-guide/features/tools.md)** — 사용 가능한 기능 살펴보기
- **[AI Providers](../integrations/providers.md)** — 전체 제공자 목록 및 설정 정보
- **[Skills System](../user-guide/features/skills.md)** — 재사용 가능한 워크플로 및 지식
- **[Tips & Best Practices](../guides/tips.md)** — 고급 사용자 팁
- **[다른 컴퓨터로 이동](/reference/faq#exporting-hermes-to-another-machine)** — `hermes backup`으로 전체 설정을 마이그레이션합니다(또는 [단일 프로필](/reference/faq#moving-a-single-profile-to-another-machine)); 처음부터 다시 구성할 필요가 없습니다.
