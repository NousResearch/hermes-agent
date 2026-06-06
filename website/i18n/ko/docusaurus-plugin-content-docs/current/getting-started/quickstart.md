---
sidebar_position: 1
title: "빠른 시작"
description: "Hermes Agent와의 첫 대화 — 설치부터 채팅까지 5분 이내 완료"
---

# 빠른 시작

이 가이드는 Hermes의 설치부터 실제 사용 가능한 설정까지 한 번에 완료할 수 있도록 도와줍니다. 설치, 프로바이더(Provider) 선택, 채팅 작동 여부 확인, 그리고 문제가 발생했을 때 대처하는 방법에 대해 알아봅니다.

## 동영상 가이드를 선호하시나요?

**Onchain AI Garage**에서 설치, 설정 및 기본 명령어를 다루는 마스터클래스 가이드를 준비했습니다. 동영상을 보며 따라 하고 싶으신 분들께 유용한 참고 자료가 될 것입니다. 자세한 내용은 전체 [Hermes Agent Tutorials & Use Cases](https://www.youtube.com/playlist?list=PLmpUb_PWAkDxewld5ZYyKifuHxgIbiq2d) 재생목록을 확인하세요.

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', maxWidth: '100%', marginBottom: '1.5rem'}}>
  <iframe
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}}
    src="https://www.youtube-nocookie.com/embed/R3YOGfTBcQg"
    title="Hermes Agent Masterclass: Installation, Setup, Basic Commands"
    frameBorder="0"
    allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
  ></iframe>
</div>

## 대상 독자

- 설치부터 작동까지 가장 빠른 경로를 원하는 입문자
- 프로바이더를 변경하려 하며 설정 실수로 시간을 낭비하고 싶지 않은 사용자
- 팀, 봇 또는 상시 실행(always-on) 워크플로우를 위해 Hermes를 설정하려는 사용자
- "설치는 되었는데 아무 동작도 하지 않는" 상황에 지친 분

## 가장 빠른 경로

목표에 맞는 항목을 선택하세요:

| 목표 | 가장 먼저 할 일 | 그 다음 할 일 |
|---|---|---|
| 내 PC에서 Hermes를 작동시키고 싶음 | `hermes setup` | 실제 채팅을 실행하여 응답하는지 확인 |
| 이미 사용할 프로바이더를 결정함 | `hermes model` | 설정을 저장한 후 채팅 시작 |
| 봇 또는 상시 실행 설정을 원함 | CLI 작동 확인 후 `hermes gateway setup` 실행 | Telegram, Discord, Slack 등의 플랫폼 연결 |
| 로컬 또는 셀프 호스팅 모델을 사용하고 싶음 | `hermes model` → custom endpoint | 엔드포인트, 모델명, 컨텍스트 길이 확인 |
| 여러 프로바이더의 대체(Fallback) 작동을 원함 | `hermes model` 먼저 설정 | 기본 채팅이 잘 작동하는 것을 확인한 후 라우팅 및 대체 설정 추가 |

**기본 규칙:** Hermes가 일반적인 채팅을 정상적으로 완료하지 못한다면, 아직 다른 기능을 추가하지 마세요. 먼저 하나의 완전한 대화가 정상 작동하는지 확인한 후 게이트웨이, 크론(Cron), 스킬(Skills), 음성(Voice), 라우팅 등을 차례대로 추가하세요.

---

## 1. Hermes Agent 설치
### macOS 또는 Windows용 Hermes Desktop 설치 프로그램 사용 (권장)
커맨드 라인 및 데스크톱 애플리케이션을 쉽게 설치하려면, 웹사이트에서 [Hermes Desktop 설치 프로그램 다운로드](https://hermes-agent.nousresearch.com/desktop)를 받아 실행하세요.

### Hermes Desktop 없이 설치:
Hermes Desktop 없이 커맨드 라인(CLI) 버전만 설치하려면 다음을 실행하세요:

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (네이티브)

PowerShell에서 실행:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

:::tip Android / Termux
휴대폰에 설치하는 경우, 검증된 수동 설치 경로, 지원되는 추가 기능 및 현재 Android 전용 제한 사항이 안내된 [Termux 가이드](./termux.md)를 참조하세요.
:::

설치가 완료되면 셸을 다시 로드하세요:

```bash
source ~/.bashrc   # 또는 source ~/.zshrc
```

상세한 설치 옵션, 필수 요구 사항 및 트러블슈팅에 대해서는 [설치 가이드](./installation.md)를 참조하세요.

## 2. 프로바이더 선택

설정에서 가장 중요한 단계입니다. `hermes model`을 사용하여 대화식으로 프로바이더를 선택하세요:

```bash
hermes model
```

:::tip 가장 쉬운 방법: Nous Portal
하나의 구독으로 300개 이상의 모델과 [Tool Gateway](../user-guide/features/tool-gateway.md)(웹 검색, 이미지 생성, TTS, 클라우드 브라우저)를 사용할 수 있습니다. 처음 설치하는 경우:

```bash
hermes setup --portal
```

이 명령어 하나로 로그인, Nous 프로바이더 설정, Tool Gateway 활성화를 한 번에 완료할 수 있습니다.
:::

권장하는 기본값:

| 프로바이더 | 설명 | 설정 방법 |
|----------|-----------|---------------|
| **Nous Portal** | 구독 기반, 별도 설정 불필요 | `hermes model`을 통해 OAuth 로그인 |
| **OpenAI Codex** | ChatGPT OAuth, Codex 모델 사용 | `hermes model`을 통해 디바이스 코드 인증 |
| **Anthropic** | Claude 모델 직접 사용 — Max 요금제 + 추가 크레딧(OAuth) 또는 토큰당 결제용 API 키 | `hermes model` → OAuth 로그인 (Max 요금제 및 추가 크레딧 필요) 또는 Anthropic API 키 입력 |
| **OpenRouter** | 다양한 모델에 대한 멀티 프로바이더 라우팅 | API 키 입력 |
| **Z.AI** | GLM / Zhipu 호스팅 모델 | `GLM_API_KEY` / `ZAI_API_KEY` 설정 (혹은 `Z_AI_API_KEY`도 허용) |
| **Kimi / Moonshot** | Moonshot 호스팅 코딩 및 채팅 모델 | `KIMI_API_KEY` 설정 (혹은 Kimi 코딩 전용 `KIMI_CODING_API_KEY`) |
| **Kimi / Moonshot China** | 중국 지역 Moonshot 엔드포인트 | `KIMI_CN_API_KEY` 설정 |
| **Arcee AI** | Trinity 모델 | `ARCEEAI_API_KEY` 설정 |
| **GMI Cloud** | 다중 모델 직접 API | `GMI_API_KEY` 설정 |
| **MiniMax (OAuth)** | 브라우저 OAuth를 통한 MiniMax 프론티어 모델 사용 — API 키 불필요 (`hermes_cli/models.py` 내의 모델 이름은 릴리스마다 변경될 수 있음) | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | 글로벌 MiniMax 엔드포인트 | `MINIMAX_API_KEY` 설정 |
| **MiniMax China** | 중국 지역 MiniMax 엔드포인트 | `MINIMAX_CN_API_KEY` 설정 |
| **Alibaba Cloud** | DashScope을 통한 Qwen 모델 | `DASHSCOPE_API_KEY` 설정 (Qwen Coding Plan의 경우 `ALIBABA_CODING_PLAN_API_KEY`도 허용) |
| **Hugging Face** | 통합 라우터를 통해 20개 이상의 오픈 모델(Qwen, DeepSeek, Kimi 등) 사용 | `HF_TOKEN` 설정 |
| **AWS Bedrock** | 네이티브 Converse API를 통해 Claude, Nova, Llama, DeepSeek 사용 | IAM 역할 또는 `aws configure` 실행 ([가이드](../guides/aws-bedrock.md)) |
| **Azure Foundry** | Azure AI Foundry 호스팅 모델 | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` 설정 |
| **Google AI Studio** | 직접 API를 통해 Gemini 모델 사용 | `GOOGLE_API_KEY` / `GEMINI_API_KEY` 설정 |
| **Google Gemini (OAuth)** | `google-gemini-cli` OAuth 흐름을 통한 Gemini 사용 — API 키 불필요 | `hermes model` → Google Gemini (OAuth) |
| **xAI** | 직접 API를 통해 Grok 모델 사용 | `XAI_API_KEY` 설정 |
| **xAI Grok OAuth** | SuperGrok / Premium+ 구독 필요, API 키 불필요 | `hermes model` → xAI Grok OAuth |
| **NovitaAI** | 다중 모델 API 게이트웨이 | `NOVITA_API_KEY` 설정 |
| **StepFun** | Step Plan 모델 | `STEPFUN_API_KEY` 설정 |
| **Xiaomi MiMo** | Xiaomi 호스팅 모델 | `XIAOMI_API_KEY` 설정 |
| **Tencent TokenHub** | Tencent 호스팅 모델 | `TOKENHUB_API_KEY` 설정 |
| **Ollama Cloud** | 매니지드 Ollama 호스팅 모델 | `OLLAMA_API_KEY` 설정 |
| **LM Studio** | OpenAI 호환 API를 제공하는 로컬 데스크톱 앱 | `LM_API_KEY` 설정 (기본값이 아닌 경우 `LM_BASE_URL`도 설정) |
| **Qwen OAuth** | Qwen Portal 브라우저 OAuth — API 키 불필요 | `hermes model` → Qwen OAuth |
| **Kilo Code** | KiloCode 호스팅 모델 | `KILOCODE_API_KEY` 설정 |
| **OpenCode Zen** | 선별된 모델에 대한 종량제 사용 | `OPENCODE_ZEN_API_KEY` 설정 |
| **OpenCode Go** | 오픈 모델을 위한 월 $10 구독 요금제 | `OPENCODE_GO_API_KEY` 설정 |
| **DeepSeek** | 직접 DeepSeek API 사용 | `DEEPSEEK_API_KEY` 설정 |
| **NVIDIA NIM** | build.nvidia.com 또는 로컬 NIM을 통한 Nemotron 모델 | `NVIDIA_API_KEY` 설정 (선택 사항: `NVIDIA_BASE_URL`) |
| **GitHub Copilot** | GitHub Copilot 구독 (GPT-5.x, Claude, Gemini 등) | `hermes model`을 통해 OAuth 로그인 또는 `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` 설정 |
| **GitHub Copilot ACP** | Copilot ACP 에이전트 백엔드 (로컬 `copilot` CLI 실행) | `hermes model` (`copilot` CLI 및 `copilot login` 필요) |
| **Custom Endpoint** | VLLM, SGLang, Ollama 또는 기타 OpenAI 호환 API | 베이스 URL + API 키 설정 |

대부분의 처음 사용자의 경우: 프로바이더를 선택하고 변경하려는 특별한 이유가 없다면 기본값을 그대로 수락하는 것이 좋습니다. 환경 변수와 설정 단계를 포함한 전체 프로바이더 카탈로그는 [AI Providers](../integrations/providers.md) 페이지를 참조하세요.

:::caution 최소 컨텍스트: 64K 토큰
Hermes Agent는 최소 **64,000 토큰**의 컨텍스트 길이를 제공하는 모델이 필요합니다. 컨텍스트 창이 이보다 작은 모델은 다단계 도구 호출(multi-step tool-calling) 워크플로우에 필요한 충분한 작동 메모리를 유지할 수 없어 시작 시 거부됩니다. 대부분의 호스팅 모델(Claude, GPT, Gemini, Qwen, DeepSeek)은 이 조건을 쉽게 만족합니다. 로컬 모델을 사용하는 경우 컨텍스트 크기를 최소 64K로 설정하세요 (예: llama.cpp의 경우 `--ctx-size 65536`, Ollama의 경우 `-c 65536`).
:::

:::tip
종속성 없이 언제든지 `hermes model`을 사용하여 프로바이더를 변경할 수 있습니다. 지원되는 모든 프로바이더의 목록과 세부 설정 방법은 [AI Providers](../integrations/providers.md)를 참조하세요.
:::

### 설정 저장 방식

Hermes는 민감한 비밀 정보(Secrets)와 일반 설정을 분리하여 관리합니다:

- **비밀 정보 및 토큰** → `~/.hermes/.env`
- **일반 설정** → `~/.hermes/config.yaml`

가장 손쉽게 값을 설정하는 방법은 CLI를 사용하는 것입니다:

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

각 값이 올바른 파일에 자동으로 저장됩니다.

## 3. 첫 번째 채팅 실행

```bash
hermes            # classic CLI
hermes --tui      # modern TUI (recommended)
```

선택한 모델, 사용 가능한 도구 및 스킬 정보가 담긴 웰컴 배너가 나타납니다. 구체적이고 확인하기 쉬운 프롬프트로 대화를 시작해 보세요:

:::tip 인터페이스 선택
Hermes는 두 가지 터미널 인터페이스를 제공합니다. 클래식한 `prompt_toolkit` CLI와 모달 오버레이, 마우스 선택, 비차단(non-blocking) 입력을 지원하는 새로운 [TUI](../user-guide/tui.md)입니다. 두 인터페이스는 세션, 슬래시 명령어, 설정을 공유하므로 `hermes`와 `hermes --tui`를 모두 실행하여 마음에 드는 것을 선택해 보세요.
:::

```
이 레포지토리를 5개의 글머리 기호로 요약하고, 메인 엔트리포인트가 무엇인지 알려줘.
```

```
현재 디렉토리를 확인하고 메인 프로젝트 파일로 보이는 것이 무엇인지 알려줘.
```

```
이 코드베이스를 위한 깔끔한 GitHub PR 워크플로우를 설정할 수 있도록 도와줘.
```

**성공적인 작동 기준:**

- 배너에 선택한 모델/프로바이더가 표시됨
- 에러 없이 Hermes가 응답함
- 필요한 경우 도구(터미널, 파일 읽기, 웹 검색 등)를 정상적으로 사용함
- 두 번 이상의 턴 동안 대화가 정상적으로 이어짐

여기까지 정상 작동한다면, 가장 어려운 단계를 무사히 마친 것입니다.

## 4. 세션 복구 기능 확인

다음 단계로 넘어가기 전에, 세션 이어하기가 제대로 작동하는지 확인하세요:

```bash
hermes --continue    # 가장 최근 세션 이어하기
hermes -c            # 단축 명령어
```

방금 전 대화하던 세션으로 다시 연결되어야 합니다. 작동하지 않는다면, 현재 동일한 프로필을 사용 중인지 그리고 실제로 세션이 저장되었는지 확인하세요. 이 기능은 나중에 여러 설정이나 기기를 오가며 작업할 때 매우 유용합니다.

## 5. 주요 기능 사용해보기

### 터미널 사용하기

```
❯ 내 디스크 사용량은 어때? 가장 용량이 큰 디렉토리 5개만 보여줘.
```

에이전트가 사용자를 대신하여 터미널 명령을 실행하고 결과를 보여줍니다.

### 슬래시 명령어

`/`를 입력하여 사용 가능한 모든 명령어가 포함된 자동 완성 드롭다운을 확인해 보세요:

| 명령어 | 기능 |
|---------|-------------|
| `/help` | 사용 가능한 모든 명령어 표시 |
| `/tools` | 사용 가능한 도구 목록 표시 |
| `/model` | 대화식으로 모델 전환 |
| `/personality pirate` | 재미있는 페르소나 적용해보기 |
| `/save` | 대화 저장하기 |

### 여러 줄 입력

줄 바꿈을 하려면 `Alt+Enter`, `Ctrl+J` 또는 `Shift+Enter`를 누르세요. `Shift+Enter`를 사용하려면 해당 키를 고유한 시퀀스로 전송하는 터미널(기본적으로 Kitty, foot, WezTerm, Ghostty, 혹은 Kitty 키보드 프로토콜이 활성화된 iTerm2, Alacritty, VS Code 터미널)이 필요합니다. `Alt+Enter`와 `Ctrl+J`는 모든 터미널에서 기본적으로 지원됩니다.

### 에이전트 동작 중단

에이전트가 응답하는 데 시간이 너무 오래 걸리는 경우, 새로운 메시지를 입력하고 Enter를 누르면 현재 작업을 중단하고 새 지시사항으로 전환됩니다. `Ctrl+C` 단축키를 사용할 수도 있습니다.

## 6. 추가 기능 설정

기본 채팅이 정상 작동할 때만 다음 단계를 추가하세요. 필요에 따라 선택할 수 있습니다:

### 봇 또는 공유 어시스턴트

```bash
hermes gateway setup    # 대화식 플랫폼 설정
```

[Telegram](/user-guide/messaging/telegram), [Discord](/user-guide/messaging/discord), [Slack](/user-guide/messaging/slack), [WhatsApp](/user-guide/messaging/whatsapp), [Signal](/user-guide/messaging/signal), [Email](/user-guide/messaging/email), [Home Assistant](/user-guide/messaging/homeassistant) 또는 [Microsoft Teams](/user-guide/messaging/teams)를 연결하세요.

### 자동화 및 도구

- `hermes tools` — 플랫폼별 도구 액세스 권한 조절
- `hermes skills` — 재사용 가능한 워크플로우 탐색 및 설치
- Cron — 봇 또는 CLI 설정이 안정화된 후에만 설정 권장

### 샌드박스 터미널

보안을 위해 에이전트를 Docker 컨테이너 또는 원격 서버에서 실행할 수 있습니다:

```bash
hermes config set terminal.backend docker    # Docker 격리 환경
hermes config set terminal.backend ssh       # 원격 서버
```

### 음성 모드

```bash
# Hermes 설치 디렉토리에서 실행합니다 (curl 설치 프로그램을 사용하는 경우
# Linux/macOS에서는 ~/.hermes/hermes-agent, Windows에서는 %LOCALAPPDATA%\hermes\hermes-agent에 위치합니다):
cd ~/.hermes/hermes-agent
uv pip install -e ".[voice]"
# 무료 로컬 음성 인식을 위한 faster-whisper가 포함되어 있습니다
```

이후 CLI에서 `/voice on`을 입력하세요. 녹음하려면 `Ctrl+B`를 누릅니다. 자세한 내용은 [Voice Mode](../user-guide/features/voice-mode.md)를 참조하세요.

### 스킬 (Skills)

스킬(Skills)은 Hermes에게 특정 작업(예: Kubernetes 배포, GitHub PR 생성, 모델 미세 조정, GIF 검색 등)을 수행하는 방법을 알려주는 온디맨드 지침 문서입니다. 각 스킬은 이름, 설명, 단계별 절차를 담은 `SKILL.md` 파일로 구성됩니다. 에이전트는 스킬의 간단한 설명만 미리 읽어두고, 해당 작업이 실제로 필요할 때만 스킬의 전체 내용을 로드하므로 스킬을 많이 추가하더라도 개별 요청의 처리 속도나 비용에 영향을 주지 않습니다.

Hermes는 `~/.hermes/skills/` 디렉토리에 기본 스킬 번들을 포함하고 있습니다. 스킬 허브(Skills Hub)에서 추가 스킬을 찾아 설치하거나 직접 작성할 수도 있습니다.

**허브에서 스킬 찾기 및 설치:**

```bash
hermes skills browse                      # 사용 가능한 전체 목록 표시
hermes skills search kubernetes           # 키워드로 스킬 검색
hermes skills install openai/skills/k8s   # 스킬 설치 (설치 전 보안 검사가 먼저 실행됩니다)
```

설치 명령어의 인자는 허브의 `source/path` 슬러그 형태입니다. 예컨대 `openai/skills/k8s`는 OpenAI 카탈로그의 `k8s` 스킬을 의미합니다. 사용할 정확한 슬러그는 `hermes skills browse`를 실행하여 확인할 수 있습니다.

**스킬 사용하기** — 설치된 스킬은 자동으로 슬래시 명령어로 등록됩니다:

```bash
/k8s deploy the staging manifest          # 요청 사항과 함께 스킬 실행
/k8s                                       # 스킬을 로드하여 Hermes가 요구 사항을 묻도록 함
```

이 명령어는 CLI뿐만 아니라 연동된 모든 메시징 플랫폼에서도 작동합니다. 모든 스킬을 미리 설치할 필요는 없습니다. 에이전트는 일반적인 대화 중에 관련 작업이 발생하면 자동으로 적절한 기본 제공 스킬을 찾아 실행합니다.

직접 스킬을 작성하는 방법, 외부 스킬 디렉토리 설정, 허브의 전체 소스 목록에 대해서는 [Skills System](../user-guide/features/skills.md)을 참조하세요.

### MCP 서버

```yaml
# ~/.hermes/config.yaml에 추가
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

### 에디터 연동 (ACP)

기본 `[all]` 추가 기능에 ACP 지원이 포함되어 있으므로, curl 설치 프로그램으로 설치한 경우 이미 포함되어 있습니다. 다음 명령어를 실행하세요:

```bash
hermes acp
```

(`[all]` 없이 설치한 경우에는 먼저 `cd ~/.hermes/hermes-agent && uv pip install -e ".[acp]"`를 실행하세요.)

자세한 내용은 [ACP Editor Integration](../user-guide/features/acp.md)을 참조하세요.

---

## 자주 발생하는 오류 해결

가장 많은 시간이 소요되는 문제 유형들입니다:

| 문제 증상 | 예상 원인 | 해결 방법 |
|---|---|---|
| Hermes는 열리지만 비어 있거나 손상된 응답을 제공함 | 프로바이더 인증 또는 모델 선택이 잘못됨 | `hermes model`을 다시 실행하여 프로바이더, 모델 및 인증 정보 확인 |
| 커스텀 엔드포인트가 '작동'하는 듯하나 의미 없는 응답을 반환함 | 베이스 URL 또는 모델 이름이 잘못되었거나, 실제로는 OpenAI 호환 API가 아님 | 다른 클라이언트를 통해 엔드포인트의 작동 여부를 먼저 검증 |
| 게이트웨이는 시작되었으나 메시지 전송이 안 됨 | 봇 토큰, 허용 목록(Allowlist) 또는 플랫폼 설정이 누락됨 | `hermes gateway setup`을 다시 실행하고 `hermes gateway status` 확인 |
| `hermes --continue` 실행 시 이전 세션을 찾을 수 없음 | 프로필이 전환되었거나 세션이 저장되지 않음 | `hermes sessions list`를 실행하여 올바른 프로필에 있는지 확인 |
| 모델을 사용할 수 없거나 비정상적인 대체(Fallback) 동작이 발생함 | 프로바이더 라우팅 또는 대체 설정이 너무 엄격하게 지정됨 | 기본 프로바이더가 안정화될 때까지 라우팅 설정을 비활성화 |
| `hermes doctor`가 설정 오류를 지적함 | 설정 값이 누락되었거나 만료됨 | 설정을 수정하고 추가 기능을 더하기 전에 일반 채팅을 먼저 테스트 |

## 복구 도구 키트

작동 상태가 평소와 다를 때 아래 순서대로 실행해 보세요:

1. `hermes doctor` (자가 진단)
2. `hermes model` (모델 설정)
3. `hermes setup` (통합 설정)
4. `hermes sessions list` (세션 목록)
5. `hermes --continue` (세션 이어하기)
6. `hermes gateway status` (게이트웨이 상태)

이 순서대로 실행하면 문제가 발생한 상태에서 작동이 검증된 안정적인 상태로 신속하게 복구할 수 있습니다.

---

## 빠른 참조

| 명령어 | 설명 |
|---------|-------------|
| `hermes` | 채팅 시작 |
| `hermes model` | LLM 프로바이더 및 모델 선택 |
| `hermes tools` | 플랫폼별로 활성화할 도구 설정 |
| `hermes setup` | 전체 설정 마법사 (모든 설정을 한 번에 진행) |
| `hermes doctor` | 문제 진단 및 자가 진단 |
| `hermes update` | 최신 버전으로 업데이트 |
| `hermes gateway` | 메시징 게이트웨이 시작 |
| `hermes --continue` | 최근 세션 이어하기 |

## 다음 단계

- **[CLI Guide](../user-guide/cli.md)** — 터미널 인터페이스 마스터하기
- **[Configuration](../user-guide/configuration.md)** — 맞춤 설정 구성하기
- **[Messaging Gateway](../user-guide/messaging/index.md)** — Telegram, Discord, Slack, WhatsApp, Signal, 이메일, Home Assistant, Teams 등 연결하기
- **[Tools & Toolsets](../user-guide/features/tools.md)** — 사용 가능한 기능 둘러보기
- **[AI Providers](../integrations/providers.md)** — 전체 프로바이더 목록 및 세부 설정 방법
- **[Skills System](../user-guide/features/skills.md)** — 재사용 가능한 워크플로우 및 지식 문서
- **[Tips & Best Practices](../guides/tips.md)** — 고급 사용자를 위한 팁과 우수 사례
