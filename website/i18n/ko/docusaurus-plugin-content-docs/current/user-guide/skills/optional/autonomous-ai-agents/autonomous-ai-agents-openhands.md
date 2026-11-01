---
title: "Openhands — OpenHands CLI에 코딩 위임하기(LiteLLM을 사용하는 모델 독립형 방식)"
sidebar_label: "Openhands"
description: "OpenHands CLI에 코딩 위임하기(LiteLLM을 사용하는 모델 독립형 방식)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Openhands

OpenHands CLI에 코딩을 위임합니다(LiteLLM을 사용하는 모델 독립형 방식).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/openhands`로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/openhands` |
| 버전 | `0.1.0` |
| 작성자 | Tim Koepsel (xzessmedia), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `Coding-Agent`, `OpenHands`, `Model-Agnostic`, `LiteLLM` |
| 관련 스킬 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# OpenHands CLI

`terminal` 도구를 통해 [OpenHands CLI](https://github.com/All-Hands-AI/OpenHands)에 코딩 작업을 위임합니다. OpenHands는 모델 독립형입니다. LiteLLM이 지원하는 모든 provider(OpenAI, Anthropic, OpenRouter, DeepSeek, Ollama, vLLM, Nous 등)를 사용할 수 있습니다.

이 스킬은 일괄 처리 또는 일회성 위임을 위한 헤드리스 모드 래퍼입니다. 대화형 텍스트 UI는 Hermes에서 사용하지 않습니다.

## 사용 시점

- 사용자가 특히 OpenHands에 코딩 작업을 위임하고 싶어 하는 경우
- 사용자가 Anthropic이나 OpenAI가 아닌 provider(DeepSeek, Qwen, Ollama, vLLM, Nous 등)에서 실행할 수 있는 코딩 에이전트를 원하는 경우 — 형제 스킬인 `claude-code`와 `codex`는 각각 하나의 벤더에 연결되어 있습니다.
- 워크스페이스 안에서 여러 단계의 파일 편집과 셸 명령을 수행하는 경우

Claude 네이티브 작업에는 `claude-code`를, OpenAI 네이티브 작업에는 `codex`를 우선 사용하세요. Hermes 네이티브 서브에이전트에는 `delegate_task`를 사용하세요.

## 사전 요구 사항

1. 업스트림을 설치합니다(Python 3.12 이상 및 `uv` 필요).

   ```
   terminal(command="uv tool install openhands --python 3.12")
   ```

   확인: `openhands --version`(작성 시점 기준 현재 `OpenHands CLI 1.16.0` / `SDK v1.21.0`).

2. 모델을 선택하고 `--override-with-envs`에 사용할 환경 변수를 설정합니다.

   ```
   export LLM_MODEL=openrouter/openai/gpt-4o-mini       # or any LiteLLM slug
   export LLM_API_KEY=$OPENROUTER_API_KEY
   export LLM_BASE_URL=https://openrouter.ai/api/v1     # omit for native OpenAI
   ```

   `LLM_MODEL`은 LiteLLM의 전체 slug를 사용합니다. provider가 OpenRouter인 경우 slug에 접두사가 두 번 붙습니다. `openrouter/<vendor>/<model>` 형식입니다(예: `openrouter/anthropic/claude-sonnet-4.5`). 네이티브 Anthropic은 `anthropic/claude-sonnet-4-5`, 네이티브 OpenAI는 `openai/gpt-4o-mini`입니다.

3. JSON 출력 앞에 ASCII 아트가 표시되지 않도록 시작 배너를 억제합니다.

   ```
   export OPENHANDS_SUPPRESS_BANNER=1
   ```

## 실행 방법

항상 `terminal` 도구를 통해 호출하세요. 자동화를 위해 항상 `--headless --json --override-with-envs --exit-without-confirmation`을 전달하세요.

### 일회성 작업

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=openrouter/openai/gpt-4o-mini LLM_API_KEY=$OPENROUTER_API_KEY LLM_BASE_URL=https://openrouter.ai/api/v1 openhands --headless --json --override-with-envs --exit-without-confirmation -t 'Add error handling to all API calls in src/'",
  workdir="/path/to/project",
  timeout=600
)
```

### 장시간 작업을 백그라운드에서 실행

```
terminal(command="<same as above>", workdir="/path/to/project", background=true, notify_on_complete=true)
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")
```

### 이전 대화 재개

OpenHands는 각 실행이 끝날 때 `Conversation ID: <32-hex>`와 `Hint: openhands --resume <dashed-uuid>` 줄을 출력합니다. 재개하려면 대시가 포함된 형식을 사용하세요.

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=... openhands --headless --json --override-with-envs --exit-without-confirmation --resume <dashed-uuid> -t 'Now fix the bug you found'",
  workdir="/path/to/project"
)
```

## 실제 플래그 목록

`openhands --help`에 대해 확인했습니다(CLI 1.16.0). 이 표에 없는 것은 플래그가 아니므로 환경 변수나 설정 파일을 통해 전달하세요.

| 플래그 | 효과 |
|------|--------|
| `--headless` | UI 없음. `-t` 또는 `-f`가 필요합니다. 모든 작업을 자동 승인합니다(헤드리스 모드에는 `--llm-approve`가 없음). |
| `--json` | JSONL 이벤트 스트림(함께 `--headless` 필요). |
| `-t TEXT` | 작업 프롬프트. |
| `-f PATH` | 파일에서 작업을 읽습니다. |
| `--resume [ID]` | 대화를 재개합니다. ID가 없으면 최근 대화를 나열합니다. |
| `--last` | 가장 최근 대화를 재개합니다(`--resume`과 함께 사용). |
| `--override-with-envs` | `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 환경 변수를 적용합니다. 이 옵션이 없으면 OpenHands는 `~/.openhands/settings.json`을 사용하고 환경 변수를 무시합니다. |
| `--exit-without-confirmation` | "are you sure" 종료 대화를 표시하지 않습니다. |
| `--always-approve` / `--yolo` | 모든 작업을 자동 승인합니다(기본값은 `--headless`). |
| `--llm-approve` | LLM 기반 보안 게이트(대화형 모드 전용 — 헤드리스에서는 작동하지 않음). |
| `--version` / `-v` | 버전을 출력하고 종료합니다. |

**`--model`, `--max-iterations`, `--workspace`, `--sandbox`, `--sandbox-type` 플래그는 없습니다.** 모델은 `LLM_MODEL`입니다. 워크스페이스는 `terminal` 도구에 전달하는 `workdir`입니다. 샌드박스/런타임은 `RUNTIME` 및 `SANDBOX_VOLUMES` 환경 변수입니다.

## JSON 이벤트 스키마

`--json --headless`를 사용하면 OpenHands는 한 줄에 JSON 객체 하나씩 JSONL을 출력하며, JSON이 아닌 상태 줄(`Initializing agent...`, `Agent is working`, `Agent finished`, 마지막 요약 상자, `Goodbye!`, `Conversation ID:`, `Hint:`)도 일부 출력합니다. `{`로 시작하는 줄만 필터링하세요.

최상위 `kind` 필드가 이벤트를 구분합니다.

- `MessageEvent` — 사용자 또는 에이전트 텍스트 턴. `source`는 `user` 또는 `agent`입니다.
- `ActionEvent` — 에이전트가 도구를 선택했습니다. `tool_name`(`file_editor`, `terminal`, `finish`)과 `action.kind`(`FileEditorAction`, `TerminalAction`, `FinishAction`)를 읽으세요.
- `ObservationEvent` — 도구 결과. `observation.is_error`가 성공 여부를 나타냅니다. `source`는 `environment`입니다.

`ActionEvent` 안의 `FinishAction`에는 에이전트의 최종 메시지가 `action.message`로 들어 있습니다.

cli는 LiteLLM/Authlib의 모든 stderr를 먼저 출력합니다. 자세한 내용은 Pitfalls를 참조하세요. stdout을 한 줄씩 파싱하고 `{`로 시작하지 않는 줄은 무시하세요.

## 주의 사항

- **모든 호출에서 LiteLLM 경고 발생.** `botocore`가 설치되어 있지 않기 때문에 CLI가 stderr에 `bedrock-runtime` 및 `sagemaker-runtime` 경고를 출력합니다. Authlib 지원 중단 경고도 있습니다. 이는 실패가 아니라 노이즈입니다. 사용자에게 표시하기 전에 stderr를 `/dev/null`로 파이프하거나 필터링하세요.
- **배너 출력.** `OPENHANDS_SUPPRESS_BANNER=1`을 설정하지 않으면 각 실행이 SDK를 홍보하는 여러 줄의 `+--+` ASCII 상자로 시작합니다. 항상 이 변수를 export하세요.
- **자동화에는 `--override-with-envs`가 필수.** 이 옵션이 없으면 OpenHands는 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`을 무시하고 `~/.openhands/settings.json`으로 돌아갑니다. 새로 설치한 환경에서는 이 파일이 존재하지 않아 CLI가 최초 설정을 기다리며 멈춥니다.
- **모델 slug는 provider의 이름이 아니라 LiteLLM의 이름.** `openrouter/openai/gpt-4o-mini`는 작동하지만 OpenRouter를 가리키면서 `openai/gpt-4o-mini`를 사용하면 작동하지 않습니다. 네이티브 Anthropic은 `anthropic/claude-sonnet-4-5`(하이픈), OpenRouter를 통한 Anthropic은 `openrouter/anthropic/claude-sonnet-4.5`(점)입니다. 잘못 입력하면 이해하기 어려운 LiteLLM 400 오류가 발생합니다.
- **`pip install openhands-ai`는 잘못된 패키지.** 이는 레거시 V0 SDK입니다. 새로운 CLI는 `uv tool install openhands --python 3.12`입니다. 유지 관리되는 conda 패키지는 없습니다.
- **재개 ID 형식이 까다로움.** CLI는 마지막에 `Conversation ID: f46573d9cfdb45e492ca189bde40019b`(대시 없음)를 출력한 다음 `Hint: openhands --resume f46573d9-cfdb-45e4-92ca189bde40019b`(대시 있음)를 출력합니다. 대시가 있는 형식을 사용하세요.
- **헤드리스는 `--llm-approve`를 무시.** 전달하면 argparse 오류가 발생합니다. 헤드리스 모드는 항상 승인을 하도록 하드코딩되어 있습니다.
- **업스트림은 Windows를 지원하지 않음.** OpenHands 문서에서는 Windows에서 WSL을 요구합니다. 따라서 이 스킬은 `[linux, macos]`로 제한됩니다.
- **`~/.openhands/conversations/<id>/`가 누적됨.** 실행할 때마다 trajectory가 저장됩니다. 배치를 실행할 때는 정리하세요.
- **설치 규모가 큼(약 200개 패키지).** 격리된 가상 환경을 사용해 의존성 충돌을 피하려면 `uv tool install`을 사용하세요.

## 검증

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=openrouter/openai/gpt-4o-mini LLM_API_KEY=$OPENROUTER_API_KEY LLM_BASE_URL=https://openrouter.ai/api/v1 openhands --headless --json --override-with-envs --exit-without-confirmation -t 'Print the string OPENHANDS_OK to stdout via the terminal tool.'",
  workdir="/tmp",
  timeout=120
)
```

JSONL 스트림이 `FinishAction`으로 끝나고 `action.message`에 `OPENHANDS_OK`가 언급되어 있다면 설치가 작동하는 것입니다.

## 관련 문서

- [OpenHands GitHub](https://github.com/All-Hands-AI/OpenHands)
- [OpenHands CLI 명령어 참조](https://docs.openhands.dev/openhands/usage/cli/command-reference)
- 형제 스킬: `claude-code`(Anthropic 전용), `codex`(OpenAI 전용), `opencode`(OpenCode를 통한 다중 provider), `hermes-agent`(`delegate_task`를 통한 Hermes 서브에이전트).
