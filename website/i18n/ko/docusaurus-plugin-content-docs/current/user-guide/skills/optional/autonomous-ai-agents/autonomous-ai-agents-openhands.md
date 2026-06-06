---
title: "Openhands — 코딩 작업을 OpenHands CLI(모델에 구애받지 않음, LiteLLM)에 위임"
sidebar_label: "Openhands"
description: "코딩 작업을 OpenHands CLI(모델에 구애받지 않음, LiteLLM)에 위임"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Openhands

코딩 작업을 OpenHands CLI(모델에 구애받지 않음, LiteLLM)에 위임합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/openhands` 명령어로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/openhands` |
| 버전 | `0.1.0` |
| 작성자 | Tim Koepsel (xzessmedia), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `Coding-Agent`, `OpenHands`, `Model-Agnostic`, `LiteLLM` |
| 관련 스킬 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# OpenHands CLI

`terminal` 도구를 통해 코딩 작업을 [OpenHands CLI](https://github.com/All-Hands-AI/OpenHands)에 위임합니다. OpenHands는 모델에 구애받지 않습니다: LiteLLM이 지원하는 모든 제공자(OpenAI, Anthropic, OpenRouter, DeepSeek, Ollama, vLLM 등)를 사용할 수 있습니다.

이 스킬은 일괄 처리(batch) / 원샷(one-shot) 위임을 위한 헤드리스(headless) 모드 래퍼입니다. 대화형 텍스트 UI는 Hermes에서 사용되지 않습니다.

## 사용 시기

- 사용자가 코딩 작업을 구체적으로 OpenHands에 위임하고자 할 때.
- 사용자가 Anthropic이나 OpenAI 제공자가 아닌 모델(DeepSeek, Qwen, Ollama, vLLM, Nous 등)에서 실행할 수 있는 코딩 에이전트를 원할 때 — 형제 스킬인 `claude-code`와 `codex`는 특정 공급업체에 종속되어 있습니다.
- 작업 공간 내부의 여러 단계 파일 편집 + 셸 명령어 실행이 필요할 때.

Claude 네이티브의 경우 `claude-code`를 선호하고, OpenAI 네이티브의 경우 `codex`를 선호하세요. Hermes 네이티브 하위 에이전트의 경우 `delegate_task`를 사용하세요.

## 전제 조건

1. 업스트림 설치 (Python 3.12+ 및 `uv` 필요):

   ```
   terminal(command="uv tool install openhands --python 3.12")
   ```

   확인: `openhands --version` (작성 당시 기준 현재 `OpenHands CLI 1.16.0` / `SDK v1.21.0`).

2. 모델을 선택하고 `--override-with-envs`를 위한 환경 변수를 설정합니다:

   ```
   export LLM_MODEL=openrouter/openai/gpt-4o-mini       # 또는 아무 LiteLLM 슬러그(slug)
   export LLM_API_KEY=$OPENROUTER_API_KEY
   export LLM_BASE_URL=https://openrouter.ai/api/v1     # 네이티브 OpenAI의 경우 생략
   ```

   `LLM_MODEL`은 LiteLLM의 전체 슬러그를 사용합니다. 제공자가 OpenRouter인 경우 슬러그에는 `openrouter/<vendor>/<model>`과 같이 이중 접두사가 붙습니다 (예: `openrouter/anthropic/claude-sonnet-4.5`). 네이티브 Anthropic의 경우: `anthropic/claude-sonnet-4-5`. 네이티브 OpenAI의 경우: `openai/gpt-4o-mini`.

3. JSON 출력 앞에 ASCII 아트가 표시되지 않도록 시작 배너를 표시하지 않습니다:

   ```
   export OPENHANDS_SUPPRESS_BANNER=1
   ```

## 실행 방법

항상 `terminal` 도구를 통해 호출하세요. 자동화를 위해 항상 `--headless --json --override-with-envs --exit-without-confirmation` 플래그를 전달하세요.

### 원샷 작업

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=openrouter/openai/gpt-4o-mini LLM_API_KEY=$OPENROUTER_API_KEY LLM_BASE_URL=https://openrouter.ai/api/v1 openhands --headless --json --override-with-envs --exit-without-confirmation -t 'Add error handling to all API calls in src/'",
  workdir="/path/to/project",
  timeout=600
)
```

### 긴 작업을 위한 백그라운드 모드

```
terminal(command="<위와 동일>", workdir="/path/to/project", background=true, notify_on_complete=true)
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")
```

### 이전 대화 재개

OpenHands는 각 실행의 끝에 `Conversation ID: <32-hex>`와 `Hint: openhands --resume <dashed-uuid>` 줄을 출력합니다. 대시(-)가 포함된 형식을 사용하여 재개합니다:

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=... openhands --headless --json --override-with-envs --exit-without-confirmation --resume <dashed-uuid> -t 'Now fix the bug you found'",
  workdir="/path/to/project"
)
```

## 실제 플래그 목록

`openhands --help` (CLI 1.16.0)를 기준으로 확인됨. 이 표에 없는 것은 플래그가 아니며 환경 변수나 설정 파일을 통해 전달해야 합니다.

| 플래그 | 효과 |
|------|--------|
| `--headless` | UI 없음, `-t` 또는 `-f` 필요. 모든 작업을 자동 승인합니다(이 모드에는 `--llm-approve`가 없음). |
| `--json` | JSONL 이벤트 스트림 (필수: `--headless`). |
| `-t TEXT` | 작업 프롬프트. |
| `-f PATH` | 파일에서 작업 읽기. |
| `--resume [ID]` | 대화 재개. ID가 없으면 → 최근 항목 나열. |
| `--last` | 가장 최근 대화 재개 (`--resume`과 함께 사용). |
| `--override-with-envs` | `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 환경 변수 적용. 이것이 없으면 OpenHands는 `~/.openhands/settings.json`을 사용하고 env를 무시합니다. |
| `--exit-without-confirmation` | "확실합니까" 종료 대화 상자를 표시하지 않음. |
| `--always-approve` / `--yolo` | 모든 작업을 자동 승인 (`--headless`의 기본값). |
| `--llm-approve` | LLM 기반 보안 게이트 (대화형 전용 — 헤드리스 모드에서는 작동하지 않음). |
| `--version` / `-v` | 버전을 출력하고 종료. |

**`--model`, `--max-iterations`, `--workspace`, `--sandbox`, `--sandbox-type` 플래그는 없습니다.** 모델은 `LLM_MODEL`입니다. 작업 공간은 `terminal` 도구에 전달하는 `workdir`입니다. 샌드박스 / 런타임은 `RUNTIME` 및 `SANDBOX_VOLUMES` 환경 변수입니다.

## JSON 이벤트 스키마

`--json --headless`를 사용하면 OpenHands는 한 줄에 하나의 JSON 객체가 있는 JSONL과 몇 개의 비 JSON 상태 줄(`Initializing agent...`, `Agent is working`, `Agent finished`, 최종 요약 상자, `Goodbye!`, `Conversation ID:`, `Hint:`)을 내보냅니다. `{`로 시작하는 줄을 필터링하세요.

최상위 `kind` 필드는 이벤트를 구별합니다:

- `MessageEvent` — 사용자 / 에이전트 텍스트 턴. `source`는 `user` 또는 `agent`입니다.
- `ActionEvent` — 에이전트가 도구를 선택했습니다. `tool_name` (`file_editor`, `terminal`, `finish`)과 `action.kind` (`FileEditorAction`, `TerminalAction`, `FinishAction`)를 읽습니다.
- `ObservationEvent` — 도구 결과. `observation.is_error`는 성공 플래그입니다. `source`는 `environment`입니다.
- `ActionEvent` 내부의 `FinishAction`은 `action.message`에 에이전트의 최종 메시지를 전달합니다.

cli는 LiteLLM/Authlib의 모든 stderr를 먼저 인쇄합니다 — 주의 사항(Pitfalls) 참조. `{`로 시작하지 않는 줄은 무시하고 줄 단위로 stdout만 구문 분석하세요.

## 주의 사항 (Pitfalls)

- **호출할 때마다 LiteLLM 경고 발생.** `botocore`가 설치되어 있지 않기 때문에 CLI는 stderr에 `bedrock-runtime` 및 `sagemaker-runtime` 경고를 출력합니다. 거기에 Authlib의 더 이상 사용되지 않음(deprecation) 경고가 추가됩니다. 이는 실패가 아니라 노이즈입니다. stderr를 `/dev/null`로 파이프하거나 사용자에게 표시하기 전에 필터링하세요.
- **배너 스팸.** `OPENHANDS_SUPPRESS_BANNER=1`이 없으면 모든 실행이 SDK를 광고하는 다중 줄 `+--+` ASCII 상자로 시작됩니다. 항상 이를 내보내기(export) 하세요.
- **`--override-with-envs`는 자동화를 위해 필수입니다.** 이 플래그가 없으면 OpenHands는 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`을 무시하고 `~/.openhands/settings.json`으로 대체합니다. 새로 설치한 상태에서는 이 파일이 존재하지 않아 CLI가 첫 실행 설정을 기다리며 멈춥니다(hang).
- **모델 슬러그는 제공자의 것이 아니라 LiteLLM의 슬러그입니다.** `openrouter/openai/gpt-4o-mini`는 작동하지만; OpenRouter를 가리키면서 `openai/gpt-4o-mini`를 쓰면 작동하지 않습니다. `anthropic/claude-sonnet-4-5` (하이픈)는 네이티브 Anthropic이고; `openrouter/anthropic/claude-sonnet-4.5` (점)는 OpenRouter를 통한 것입니다. 잘못 설정하면 → 알 수 없는 LiteLLM 400 에러가 발생합니다.
- **`pip install openhands-ai`는 잘못된 패키지입니다.** 그것은 레거시 V0 SDK입니다. 새로운 CLI는 `uv tool install openhands --python 3.12`입니다. 유지 관리되는 conda 패키지는 없습니다.
- **재개(Resume) ID 형식은 까다롭습니다.** CLI는 끝에 `Conversation ID: f46573d9cfdb45e492ca189bde40019b` (대시 없음)를 표시한 다음 `Hint: openhands --resume f46573d9-cfdb-45e4-92ca-189bde40019b` (대시 있음)를 표시합니다. 대시가 있는 형식을 사용하세요.
- **헤드리스는 `--llm-approve`를 무시합니다.** 전달하면 argparse 에러가 발생합니다. 헤드리스 모드는 항상 승인(always-approve)하도록 하드코딩되어 있습니다.
- **Windows를 업스트림에서 지원하지 않습니다.** OpenHands 문서는 Windows 환경에 WSL이 필요하다고 명시하고 있습니다. 이에 따라 이 스킬은 `[linux, macos]`로 제한됩니다.
- **`~/.openhands/conversations/<id>/` 데이터가 누적됩니다.** 매 실행마다 궤적(trajectory)이 지속적으로 저장됩니다. 일괄 처리를 실행하는 경우 이를 정리하세요.
- **무거운 설치 (~200개 패키지).** 활성 프로젝트와의 종속성 충돌을 피하기 위해 `uv tool install` (격리된 venv)을 사용하세요.

## 검증 (Verification)

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=openrouter/openai/gpt-4o-mini LLM_API_KEY=$OPENROUTER_API_KEY LLM_BASE_URL=https://openrouter.ai/api/v1 openhands --headless --json --override-with-envs --exit-without-confirmation -t 'Print the string OPENHANDS_OK to stdout via the terminal tool.'",
  workdir="/tmp",
  timeout=120
)
```

JSONL 스트림이 `action.message`에 `OPENHANDS_OK`를 언급하는 `FinishAction`으로 끝나면, 설치가 작동하는 것입니다.

## 관련 항목

- [OpenHands GitHub](https://github.com/All-Hands-AI/OpenHands)
- [OpenHands CLI 명령어 참조](https://docs.openhands.dev/openhands/usage/cli/command-reference)
- 형제 스킬: `claude-code` (Anthropic 전용), `codex` (OpenAI 전용), `opencode` (OpenCode를 통한 멀티 제공자), `hermes-agent` (`delegate_task`를 통한 Hermes 하위 에이전트).
