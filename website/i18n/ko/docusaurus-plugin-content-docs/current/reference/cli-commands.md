---
sidebar_position: 1
title: "CLI 명령어 참조"
description: "Hermes 터미널 명령어와 명령어 제품군에 대한 권위 있는 참조"
---

# CLI 명령어 참조

이 페이지에서는 셸에서 실행하는 **터미널 명령어**를 다룹니다.

채팅 내 슬래시 명령어는 [슬래시 명령어 참조](./slash-commands.md)를 참조하세요.

## 전역 진입점

```bash
hermes [global-options] <command> [subcommand/options]
```

### 전역 옵션

| 옵션 | 설명 |
|--------|-------------|
| `--version`, `-V` | 버전을 표시하고 종료합니다. |
| `--profile <name>`, `-p <name>` | 이 호출에 사용할 Hermes 프로필을 선택합니다. `hermes profile use`로 설정한 고정 기본값을 재정의합니다. |
| `--resume <session>`, `-r <session>` | ID 또는 제목으로 이전 세션을 재개합니다. 키워드 `latest`는 가장 최근 세션을 재개합니다(`-c`와 동일한 작업 공간 범위 조회). |
| `--continue [name]`, `-c [name]` | 가장 최근 세션 또는 제목이 일치하는 가장 최근 세션을 재개합니다. |
| `--in <dir>` | 시작 또는 재개하기 전에 `<dir>`로 이동합니다. `--resume latest` / `-c` 조회를 해당 디렉터리의 작업 공간으로 제한하고 세션을 그곳에 유지합니다(기록된 cwd 복원을 건너뜁니다). |
| `--worktree`, `-w` | 병렬 에이전트 워크플로를 위한 격리된 git worktree에서 시작합니다. |
| `--yolo` | 위험한 명령어 승인 프롬프트를 우회합니다. |
| `--pass-session-id` | 에이전트의 시스템 프롬프트에 세션 ID를 포함합니다. |
| `--ignore-user-config` | `~/.hermes/config.yaml`을 무시하고 내장 기본값으로 대체합니다. `.env`의 자격 증명은 계속 로드됩니다. |
| `--ignore-rules` | `AGENTS.md`, `SOUL.md`, `.cursorrules`, 메모리 및 미리 로드된 스킬의 자동 주입을 건너뜁니다. |
| `--tui` | 클래식 CLI 대신 [TUI](../user-guide/tui.md)를 실행합니다. `HERMES_TUI=1`과 동일합니다. `display.interface`보다 항상 우선합니다. |
| `--cli` | 클래식 prompt_toolkit REPL을 강제로 사용합니다. 한 번의 호출에서 `display.interface: tui`를 재정의할 때 사용합니다. |
| `--dev` | `--tui`와 함께 사용하면 미리 빌드된 번들 대신 `tsx`를 통해 TypeScript 소스를 직접 실행합니다(TUI 기여자용). |

## 최상위 명령어

| 명령어 | 용도 |
|---------|---------|
| `hermes chat` | 에이전트와 대화형 또는 단일 실행 채팅을 시작합니다. |
| `hermes model` | 기본 제공자와 모델을 대화형으로 선택합니다. |
| `hermes moa` | 모델 선택기에서 선택할 수 있는 이름 있는 Mixture of Agents 프리셋을 구성합니다. |
| `hermes fallback` | 기본 모델에 오류가 발생했을 때 시도할 대체 제공자를 관리합니다. |
| `hermes gateway` | 메시징 게이트웨이 서비스를 실행하거나 관리합니다. |
| `hermes proxy` | OAuth 제공자 자격 증명을 연결하는 로컬 OpenAI 호환 프록시입니다. [구독 프록시](../user-guide/features/subscription-proxy.md)를 참조하세요. |
| `hermes egress` | 원격 터미널 샌드박스를 위한 아웃바운드 자격 증명 주입 방화벽(iron-proxy)입니다. 기본적으로 비활성화되어 있습니다. [Egress 프록시](../user-guide/egress/iron-proxy.md)를 참조하세요. |
| `hermes lsp` | Language Server Protocol 통합(쓰기 작업의 의미론적 진단)을 관리합니다. |
| `hermes setup` | 구성 전체 또는 일부를 위한 대화형 설정 마법사입니다. |
| `hermes whatsapp` | WhatsApp 브리지를 구성하고 페어링합니다. |
| `hermes whatsapp-cloud` | 공식 Meta WhatsApp Business Cloud API 어댑터를 구성합니다(Business 계정 + 공개 웹훅 필요). `hermes whatsapp`(Baileys 개인 계정 브리지)과는 별개입니다. |
| `hermes slack` | Slack 도우미입니다(현재: 모든 명령어를 네이티브 슬래시 명령어로 포함하는 앱 매니페스트 생성). |
| `hermes auth` | 자격 증명을 관리합니다 — 추가, 목록, 제거, 초기화, 상태, 로그아웃. Codex/Nous/Anthropic의 OAuth 플로를 처리합니다. |
| `hermes login` / `logout` | **사용 중단됨** — 대신 `hermes auth`를 사용하세요. |
| `hermes send` | 구성된 메시징 플랫폼(Telegram, Discord, Slack, Signal, SMS, …)으로 단일 실행 메시지를 보냅니다. 셸 스크립트, cron 작업, CI 훅 및 모니터링 데몬에서 유용하며 에이전트 루프나 LLM이 필요하지 않습니다. |
| `hermes secrets` | 외부 시크릿 소스(현재 Bitwarden Secrets Manager)를 관리하여 `~/.hermes/.env` 대신 프로세스 시작 시 API 키를 가져옵니다. |
| `hermes migrate` | 사용 중단된 모델이나 더 이상 사용되지 않는 설정에 대한 참조를 대체하도록 `config.yaml`을 진단하고 (선택적으로) 다시 작성합니다(예: `migrate xai`). |
| `hermes status` | 에이전트, 인증 및 플랫폼 상태를 표시합니다. |
| `hermes cron` | cron 스케줄러를 검사하고 실행합니다. |
| `hermes kanban` | 다중 프로필 협업 보드(작업, 링크, 디스패처)입니다. |
| `hermes project` | 이름이 지정된 다중 폴더 작업 공간을 관리합니다. 데스크톱 세션 그룹화를 고정하고, kanban 보드에 연결하면 작업에 결정적인 worktree + 브랜치 규칙을 부여합니다. 상태는 프로필별입니다. |
| `hermes webhook` | 이벤트 기반 활성화를 위한 동적 웹훅 구독을 관리합니다. |
| `hermes hooks` | `config.yaml`에 선언된 셸 스크립트 훅을 검사, 승인 또는 제거합니다. |
| `hermes doctor` | 구성 및 종속성 문제를 진단합니다. |
| `hermes security audit` | venv, 플러그인 요구 사항 및 고정된 MCP 서버에 대한 주문형 공급망 감사(OSV.dev)를 수행합니다. |
| `hermes approvals` | 승인 프롬프트 도구 — 승인 기록을 분석하여 허용 목록 제안을 만듭니다. |
| `hermes dump` | 복사하여 붙여넣을 수 있는 설정 요약을 지원/디버깅용으로 출력합니다. |
| `hermes prompt-size` | 시스템 프롬프트 + 도구 스키마(스킬 색인, 메모리, 프로필)의 바이트별 분석을 표시합니다. 오프라인으로 실행됩니다. |
| `hermes debug` | 디버그 도구 — 지원을 위해 로그와 시스템 정보를 업로드합니다. |
| `hermes backup` | Hermes 홈 디렉터리를 zip 파일로 백업합니다. |
| `hermes checkpoints` | `~/.hermes/checkpoints/`(`/rollback`에서 사용하는 섀도 스토어)를 검사/정리/삭제합니다. 인수 없이 실행하면 상태 개요를 표시합니다. |
| `hermes import` | zip 파일에서 Hermes 백업을 복원합니다. |
| `hermes logs` | 에이전트/게이트웨이/오류 로그 파일을 보고, 추적하고, 필터링합니다. |
| `hermes config` | 구성 파일을 표시, 편집, 마이그레이션 및 조회합니다. |
| `hermes skin` | 디스플레이 스킨을 나열, 전환 및 조정합니다. |
| `hermes console` | 안전한 Hermes 명령어 콘솔을 엽니다. |
| `hermes pairing` | 메시징 페어링 코드를 승인하거나 취소합니다. |
| `hermes skills` | 스킬을 탐색, 설치, 게시, 감사 및 구성합니다. |
| `hermes bundles` | 여러 스킬을 하나의 `/<name>` 슬래시 명령어로 그룹화합니다. [스킬 번들](../user-guide/features/skills.md#skill-bundles)을 참조하세요. |
| `hermes curator` | 백그라운드 스킬 유지 관리 — 상태, 실행, 일시 중지, 고정. [Curator](../user-guide/features/curator.md)를 참조하세요. |
| `hermes journey` (별칭 `learning`, `memory-graph`) | 시간에 따른 학습된 스킬 + 메모리의 타임라인입니다. |
| `hermes memory` | 외부 메모리 제공자를 구성합니다. 제공자가 활성화되면 플러그인별 하위 명령어(예: `hermes honcho`)가 자동으로 등록됩니다. |
| `hermes acp` | 편집기 통합을 위한 ACP 서버로 Hermes를 실행합니다. |
| `hermes mcp` | MCP 서버 구성을 관리하고 Hermes를 MCP 서버로 실행합니다. |
| `hermes plugins` | Hermes Agent 플러그인을 관리합니다(설치, 활성화, 비활성화, 제거). |
| `hermes portal` | Nous Portal 상태, 구독 링크 및 Tool Gateway 라우팅입니다. [Tool Gateway](../user-guide/features/tool-gateway.md)를 참조하세요. |
| `hermes tools` | 플랫폼별 활성화된 도구를 구성합니다. |
| `hermes computer-use` | Computer Use(cua-driver) 백엔드를 설치하거나 확인합니다(macOS/Windows/Linux). |
| `hermes pets` | CLI, TUI 및 데스크톱 앱에 표시되는 [petdex](../user-guide/features/pets.md) 애니메이션 펫을 탐색, 설치 및 선택합니다. 하위 명령어: `list`, `install`, `select`, `show`, `off`, `scale`, `remove`, `doctor`. |
| `hermes sessions` | 세션을 탐색, 내보내기, 정리, 이름 변경 및 삭제합니다. |
| `hermes insights` | 토큰/비용/활동 분석을 표시합니다. |
| `hermes claw` | OpenClaw 마이그레이션 도우미입니다. |
| `hermes import-agent` | Claude Code(`~/.claude`) 또는 Codex CLI(`~/.codex`) 설정을 가져옵니다. |
| `hermes dashboard` | 구성, API 키 및 세션을 관리하기 위한 웹 대시보드를 실행합니다. |
| `hermes serve` | Hermes 백엔드 서버를 시작합니다(헤드리스; 데스크톱 앱과 원격 백엔드를 구동). |
| `hermes desktop` (별칭 `gui`) | 네이티브 Electron 데스크톱 앱을 빌드하고 실행합니다. |
| `hermes profile` | 프로필을 관리합니다 — 여러 개의 격리된 Hermes 인스턴스입니다. |
| `hermes completion` | 셸 자동 완성 스크립트(bash/zsh/fish)를 출력합니다. |
| `hermes version` | 버전 정보를 표시합니다. |
| `hermes update` | 최신 코드를 가져오고 종속성을 다시 설치합니다. `--check`는 설치 없이 미리 확인하며, `--backup`은 가져오기 전에 `HERMES_HOME` 스냅샷을 생성합니다. |
| `hermes uninstall` | 시스템에서 Hermes를 제거합니다. |

## `hermes chat`

```bash
hermes chat [options]
```

일반 옵션:

| 옵션 | 설명 |
|--------|-------------|
| `-q`, `--query "..."` | 단일 실행 비대화형 프롬프트입니다. |
| `-m`, `--model <model>` | 이 실행에 사용할 모델을 재정의합니다. |
| `-t`, `--toolsets <csv>` | 쉼표로 구분된 도구 모음을 활성화합니다. |
| `--provider <provider>` | 제공자를 강제합니다: `auto`, `openrouter`, `nous`, `openai-codex`, `copilot-acp`, `copilot`, `anthropic`, `gemini`, `huggingface`, `novita`(별칭 `novita-ai`, `novitaai`), `openai-api`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `kilocode`, `xiaomi`, `arcee`, `gmi`, `upstage`(별칭 `solar`), `alibaba`, `alibaba-coding-plan`(별칭 `alibaba_coding`), `deepseek`, `nvidia`, `ollama-cloud`, `xai`(별칭 `grok`), `xai-oauth`(별칭 `grok-oauth`), `qwen-oauth`, `bedrock`, `opencode-zen`, `opencode-go`, `ai-gateway`, `azure-foundry`, `lmstudio`, `stepfun`, `tencent-tokenhub`(별칭 `tencent`, `tokenhub`). |
| `-s`, `--skills <name>` | 세션에 하나 이상의 스킬을 미리 로드합니다(반복하거나 쉼표로 구분할 수 있음). |
| `-v`, `--verbose` | 자세한 출력을 표시합니다. |
| `-Q`, `--quiet` | 프로그래밍 모드: 배너/스피너/도구 미리 보기를 표시하지 않습니다. |
| `--image <path>` | 단일 쿼리에 로컬 이미지를 첨부합니다. |
| `--resume <session>` / `--continue [name]` | `chat`에서 직접 세션을 재개합니다. |
| `--worktree` | 이 실행을 위한 격리된 git worktree를 생성합니다. |
| `--checkpoints` | 파일을 파괴적으로 변경하기 전에 파일 시스템 체크포인트를 활성화합니다. |
| `--yolo` | 승인 프롬프트를 건너뜁니다. |
| `--pass-session-id` | 세션 ID를 시스템 프롬프트에 전달합니다. |
| `--ignore-user-config` | `~/.hermes/config.yaml`을 무시하고 내장 기본값을 사용합니다. `.env`의 자격 증명은 계속 로드됩니다. 격리된 CI 실행, 재현 가능한 버그 보고서 및 서드파티 통합에 유용합니다. |
| `--ignore-rules` | `AGENTS.md`, `SOUL.md`, `.cursorrules`, 영구 메모리 및 미리 로드된 스킬의 자동 주입을 건너뜁니다. 완전히 격리된 실행을 위해 `--ignore-user-config`와 함께 사용하세요. |
| `--safe-mode` | 문제 해결 모드: 모든 사용자 지정(사용자 구성, 규칙/메모리 주입, 플러그인, 셸 훅 및 MCP 서버)을 비활성화합니다(`--ignore-user-config` 및 `--ignore-rules` 포함). 문제가 설정에서 비롯되었는지 Hermes 자체에서 비롯되었는지 분리할 때 사용합니다. |
| `--source <tag>` | 필터링을 위한 세션 소스 태그(기본값: `cli`)입니다. 사용자의 세션 목록에 표시되지 않아야 하는 서드파티 통합에는 `tool`을 사용하세요. |
| `--max-turns <N>` | 대화 턴당 최대 도구 호출 반복 횟수(기본값: 500 또는 구성의 `agent.max_turns`)입니다. |

예시:

```bash
hermes
hermes chat -q "Summarize the latest PRs"
hermes chat --provider openrouter --model anthropic/claude-sonnet-4.6
hermes chat --toolsets web,terminal,skills
hermes chat --quiet -q "Return only JSON"
hermes chat --worktree -q "Review this repo and open a PR"
hermes chat --ignore-user-config --ignore-rules -q "Repro without my personal setup"
hermes chat --safe-mode -q "Is this bug mine or Hermes'?"
```

### `hermes -z <prompt>` — 스크립트용 단일 실행

프로그래밍 방식의 호출자(셸 스크립트, CI, cron, 프롬프트를 파이프로 전달하는 상위 프로세스)를 위해 `hermes -z`는 가장 순수한 단일 실행 진입점입니다. **표준 출력이나 표준 오류에는 아무것도 추가하지 않고, 입력으로 단일 프롬프트를 받아 최종 응답 텍스트를 출력합니다.** 배너, 스피너, 도구 미리 보기, `Session:` 줄 없이 에이전트의 최종 답변만 일반 텍스트로 출력합니다.

```bash
hermes -z "What's the capital of France?"
# → Paris.

# Parent scripts can cleanly capture the response:
answer=$(hermes -z "summarize this" < /path/to/file.txt)
```

실행별 재정의(`~/.hermes/config.yaml`을 변경하지 않음):

| 플래그 | 동등한 환경 변수 | 용도 |
|---|---|---|
| `-m` / `--model <model>` | `HERMES_INFERENCE_MODEL` | 이 실행에 사용할 모델을 재정의합니다. |
| `--provider <provider>` | _(없음)_ | 이 실행에 사용할 제공자를 재정의합니다. |
| `--usage-file <path>` | _(없음)_ | 실행 후 JSON 사용량 보고서를 작성합니다(아래 참조). |

```bash
hermes -z "…" --provider openrouter --model openai/gpt-5.5
# or:
HERMES_INFERENCE_MODEL=anthropic/claude-sonnet-4.6 hermes -z "…"
```

동일한 에이전트, 동일한 도구, 동일한 스킬을 사용하지만 모든 대화형/장식 레이어를 제거합니다. 트랜스크립트에 도구 출력도 필요하면 대신 `hermes chat -q`를 사용하세요. `-z`는 명시적으로 “최종 답변만 필요합니다”를 위한 옵션입니다.
#### `--usage-file` — 파이프라인용 JSON 사용량 보고서

`hermes -z "…" --usage-file /path/report.json`은 실행 후 기계가 읽을 수 있는 사용량 보고서인 `estimated_cost_usd`, `input_tokens` / `output_tokens` / `cache_read_tokens` / `cache_write_tokens` / `reasoning_tokens` / `total_tokens`, `api_calls`, `model`, `provider`, `session_id`, `service_tier`, 그리고 `completed` / `failed` 플래그를 기록합니다. 보고서는 실행이 실패하더라도 항상 기록되므로, 배치 파이프라인에서 비용을 빠짐없이 추적할 수 있습니다. `-z`/`--oneshot` 외부에서는 아무런 영향을 주지 않으며, 사용량 보고서 기록에 실패해도 실행 자체의 결과가 가려지지 않습니다.

```bash
hermes -z "summarize this repo" --usage-file /tmp/usage.json
jq .estimated_cost_usd /tmp/usage.json
```

## `hermes model`

```bash
hermes model
```

대화형 provider + model 선택기입니다. **새 provider 추가, API 키 설정, OAuth 흐름 실행을 위한 명령입니다.** 활성 Hermes 채팅 세션 안이 아니라 터미널에서 실행하세요.

다음과 같은 경우에 사용합니다.
- **새 provider 추가** (OpenRouter, Anthropic, Copilot, DeepSeek, 사용자 지정 등)
- OAuth 기반 provider 로그인 (Anthropic, Copilot, Codex, Nous Portal)
- API 키 입력 또는 업데이트
- provider별 모델 목록에서 선택
- 사용자 지정/셀프 호스팅 endpoint 구성
- 새 기본값을 config에 저장

:::warning `hermes model`과 `/model`의 차이 — 꼭 알아두세요
**`hermes model`** (활성 Hermes 세션 외부의 터미널에서 실행)는 **전체 provider 설정 마법사**입니다. provider를 추가하고, OAuth 흐름을 실행하고, API 키를 요청하고, endpoint를 구성할 수 있습니다.

**`/model`** (활성 Hermes 채팅 세션 안에서 입력)는 이미 설정한 provider와 model 사이만 전환할 수 있습니다. provider를 추가하거나 OAuth를 실행하거나 API 키를 요청할 수 없습니다.

**새 provider를 추가해야 한다면:** 먼저 Hermes 세션을 종료한 다음(`Ctrl+C` 또는 `/quit`), 터미널 프롬프트에서 `hermes model`을 실행하세요.
:::

### `/model` 슬래시 명령 (세션 중)

세션을 나가지 않고 이미 구성된 model 사이를 전환합니다.

```
/model                              # Show current model and available options
/model claude-sonnet-4              # Switch model (auto-detects provider)
/model zai:glm-5                    # Switch provider and model
/model custom:qwen-2.5              # Use model on your custom endpoint
/model custom                       # Auto-detect model from custom endpoint
/model custom:local:qwen-2.5        # Use a named custom provider
/model openrouter:anthropic/claude-sonnet-4  # Switch back to cloud
```

기본적으로 `/model` 변경 사항은 **현재 세션에만** 적용됩니다. `--global`을 추가하면 변경 사항이 `config.yaml`에 저장됩니다(또는 `model.persist_switch_by_default: true`를 설정해 모든 전환을 저장하도록 할 수 있습니다).

```
/model claude-sonnet-4 --global     # Switch and save as new default
```

:::info OpenRouter model만 보이는 경우는 어떻게 하나요?
OpenRouter만 구성했다면 `/model`에는 OpenRouter model만 표시됩니다. 다른 provider(Anthropic, DeepSeek, Copilot 등)를 추가하려면 세션을 종료하고 터미널에서 `hermes model`을 실행하세요.
:::

`--global` 전환 시 provider와 base URL 변경 사항은 model과 함께 `config.yaml`에 저장됩니다. 사용자 지정 endpoint에서 전환할 때는 오래된 base URL을 삭제하여 다른 provider로 유출되지 않도록 합니다.

## `hermes gateway`

```bash
hermes gateway <subcommand>
```

하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `run` | gateway를 포그라운드에서 실행합니다. WSL, Docker, Termux에 권장됩니다. |
| `start` | 설치된 systemd/launchd 백그라운드 서비스를 시작합니다. |
| `stop` | 서비스를 중지합니다(또는 포그라운드 프로세스를 중지합니다). |
| `restart` | 서비스를 다시 시작합니다. |
| `status` | 서비스 상태를 표시합니다. |
| `list` | **모든 profile**과 각 profile의 gateway 실행 여부를 표시합니다(가능한 경우 PID 포함). 여러 profile을 나란히 실행하면서 한눈에 전체 상태를 확인할 때 유용합니다. |
| `install` | systemd(Linux) 또는 launchd(macOS) 백그라운드 서비스로 설치합니다. |
| `uninstall` | 설치된 서비스를 제거합니다. |
| `setup` | 대화형 메시징 플랫폼 설정을 실행합니다. |
| `migrate-legacy` | 이름을 변경하기 전 설치에서 남은 레거시 `hermes.service` unit을 제거합니다. profile unit(`hermes-gateway-<profile>.service`)과 관련 없는 서비스는 절대 건드리지 않습니다. 플래그: `--dry-run`, `-y`/`--yes`. |
| `enroll` | 실험적 기능: 이 gateway를 relay connector에 등록하고 connector 기반 platform용 relay 자격 증명을 저장합니다. [Hermes Relay](/user-guide/messaging/relay)를 참고하세요. |

옵션:

| 옵션 | 설명 |
|--------|-------------|
| `--all` | `start` / `restart` / `stop`에서 활성 `HERMES_HOME`뿐 아니라 **모든 profile의** gateway에 적용합니다. 여러 profile을 나란히 실행하는 경우 `hermes update` 후 모두 다시 시작할 때 유용합니다. |
| `--no-supervise` | `run`에서 s6-overlay Docker 이미지 내부의 자동 supervision을 사용하지 않고, s6 이전의 포그라운드 동작을 사용합니다. gateway가 자동 재시작 없이 컨테이너의 주 프로세스로 실행됩니다. s6 이미지 외부에서는 아무 동작도 하지 않습니다. `HERMES_GATEWAY_NO_SUPERVISE=1`을 설정하는 것과 같습니다. |
| `--external-supervisor` | `run`에서 wrapper가 제공하는 프로세스 관리자가 포그라운드 gateway를 관리한다고 선언합니다. `sudo`, `env -i` 또는 다른 wrapper가 launchd/systemd의 기본 환경 marker를 제거하는 경우 사용하세요. 채팅 내 재시작 및 업데이트는 분리된 대체 프로세스를 생성하는 대신 해당 관리자에게 돌아갑니다. |

`--external-supervisor`는 재시작 정책 계약입니다. 채팅 내 재시작 또는 서비스 재시작 업데이트는 상태 `75`로 종료되므로, wrapper의 supervisor는 해당 비정상 종료 후 gateway를 다시 실행해야 합니다. systemd에서는 `Restart=on-failure` 또는 `Restart=always`를 사용하고 `RestartPreventExitStatus`에 `75`를 포함하지 마세요. launchd에서는 실패한 종료 후 다시 실행하도록 `KeepAlive`를 구성하세요. 이 정책이 없으면 요청한 재시작 후 gateway가 중지된 상태로 남습니다.

`hermes gateway enroll`은 `--token`, `--connector-url`, `--gateway-id`, `--wake-url`을 허용합니다. enrollment token을 connector와 교환하고, 그 결과로 얻은 `GATEWAY_RELAY_ID`, `GATEWAY_RELAY_SECRET`, `GATEWAY_RELAY_DELIVERY_KEY`, 선택적 `GATEWAY_RELAY_URL`, 그리고 `--wake-url`이 지정된 경우 `GATEWAY_RELAY_WAKE_URL` 값을 활성 profile의 `.env`에 기록합니다.

:::tip WSL 사용자
`hermes gateway start` 대신 `hermes gateway run`을 사용하세요. WSL의 systemd 지원은 불안정합니다. 지속 실행을 위해 tmux로 감싸세요: `tmux new -s hermes 'hermes gateway run'`. 자세한 내용은 [WSL FAQ](/reference/faq#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails)를 참고하세요.
:::

## `hermes lsp`

```bash
hermes lsp <subcommand>
```

Language Server Protocol 통합을 관리합니다. LSP는 실제 language server(pyright, gopls, rust-analyzer, …)를 백그라운드에서 실행하고, `write_file`과 `patch`에서 사용하는 작성 후 검사에 해당 server의 진단 결과를 전달합니다. git workspace가 감지된 경우에만 활성화됩니다. 즉 cwd 또는 편집한 파일이 git worktree 안에 있을 때만 LSP가 실행됩니다.

하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `status` | 서비스 상태, 구성된 server, 설치 상태를 표시합니다. |
| `list` | 지원되는 server의 registry를 출력합니다. 없는 server를 건너뛰려면 `--installed-only`를 전달하세요. |
| `install <id>` | server 하나의 binary를 미리 설치합니다. |
| `install-all` | 자동 설치 recipe가 알려진 모든 server를 설치합니다. |
| `restart` | 실행 중인 client를 종료하여 다음 편집 시 다시 생성되도록 합니다. |
| `which <id>` | server 하나의 확인된 binary 경로를 출력합니다. |

지원 언어와 전체 구성 옵션은 [LSP — Semantic Diagnostics](/user-guide/features/lsp)를 참고하세요.

## `hermes setup`

```bash
hermes setup [model|tts|terminal|gateway|tools|agent] [--non-interactive] [--reset] [--quick] [--reconfigure] [--portal]
```

**가장 쉬운 방법:** `hermes setup --portal` — Nous Portal에 OAuth로 로그인하고 [Tool Gateway](../user-guide/features/tool-gateway.md)를 한 번에 활성화합니다.

**첫 실행:** 최초 실행 마법사를 시작합니다.

**재방문 사용자(이미 구성됨):** 전체 재구성 마법사로 바로 들어갑니다. 모든 프롬프트는 현재 값을 기본값으로 표시하며, Enter를 누르면 유지되고 새 값을 입력하면 변경됩니다. 메뉴는 표시되지 않습니다.

전체 마법사 대신 특정 섹션으로 바로 이동할 수 있습니다.

| 섹션 | 설명 |
|---------|-------------|
| `model` | provider 및 model 설정입니다. |
| `terminal` | terminal backend 및 sandbox 설정입니다. |
| `gateway` | 메시징 platform 설정입니다. |
| `tools` | platform별 tool을 활성화/비활성화합니다. |
| `agent` | agent 동작 설정입니다. |

옵션:

| 옵션 | 설명 |
|--------|-------------|
| `--quick` | 재방문 사용자 실행에서 누락되었거나 설정되지 않은 항목만 요청합니다. 이미 구성한 항목은 건너뜁니다. |
| `--non-interactive` | 프롬프트 없이 기본값/환경 값을 사용합니다. |
| `--reset` | 설정을 setup 전에 기본값으로 재설정합니다. |
| `--reconfigure` | 이전 버전과의 호환을 위한 별칭입니다. 기존 설치에서 인수 없이 `hermes setup`을 실행하면 이제 기본적으로 이 동작을 수행합니다. |
| `--portal` | 한 번에 Nous Portal을 설정합니다. OAuth로 로그인하고 Nous를 inference provider로 설정한 뒤 [Tool Gateway](../user-guide/features/tool-gateway.md)를 활성화합니다. 나머지 마법사는 건너뜁니다. |

## `hermes portal`

```bash
hermes portal [status|open|tools]
```

Nous Portal 인증, Tool Gateway routing을 확인하고 subscription 페이지로 이동합니다. 하위 명령 없이 실행하면 `status`를 실행합니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `status` (기본값) | Portal 인증 상태와 tool별 Tool Gateway routing 요약을 표시합니다. 하위 명령을 지정하지 않은 경우에도 표시됩니다. |
| `open` | 기본 browser에서 `portal.nousresearch.com/manage-subscription`을 엽니다. |
| `tools` | 모든 Tool Gateway partner(Firecrawl, FAL, OpenAI TTS, Browser Use, Modal)와 Nous를 통해 routing되는 partner를 나열합니다. |

gateway 자체의 구성은 [Tool Gateway](../user-guide/features/tool-gateway.md)를 참고하세요. 한 번에 설정하는 방법은 위의 `hermes setup --portal`을 참고하세요.

## `hermes whatsapp`

```bash
hermes whatsapp
```

모드 선택과 QR 코드 pairing을 포함한 WhatsApp pairing/setup 흐름을 실행합니다.

## `hermes slack`

```bash
hermes slack manifest              # print manifest to stdout
hermes slack manifest --write      # write to ~/.hermes/slack-manifest.json
hermes slack manifest --long-description-file AGENTS.md --write
hermes slack manifest --slashes-only  # just the features.slash_commands array
```

`COMMAND_REGISTRY`의 모든 gateway 명령(`/btw`, `/stop`, `/model`, …)을 일반 Slack 슬래시 명령으로 등록하는 Slack app manifest를 생성합니다. Discord 및 Telegram과 동일한 기능 수준을 제공합니다. 출력을 Slack app 구성의 [https://api.slack.com/apps](https://api.slack.com/apps) → 해당 app → **Features → App Manifest → Edit**에 붙여 넣은 다음 **Save**를 선택하세요. scope 또는 슬래시 명령이 변경되면 Slack에서 재설치를 요청합니다.

| 플래그 | 기본값 | 용도 |
|------|---------|---------|
| `--write [PATH]` | stdout | stdout 대신 파일에 기록합니다. 인수 없는 `--write`는 `$HERMES_HOME/slack-manifest.json`에 기록합니다. |
| `--name NAME` | `Hermes` | Slack에서 표시되는 bot 이름입니다. |
| `--description DESC` | 기본 설명 | Slack app directory에 표시되는 bot 설명입니다. |
| `--long-description TEXT` | 설정 안 됨 | `display_information.long_description`을 inline으로 설정합니다(175~4,000자). `--slashes-only`와 함께 사용할 수 없습니다. |
| `--long-description-file PATH` | 설정 안 됨 | UTF-8 텍스트 파일에서 긴 설명을 읽고 내용을 정확히 보존합니다. `--long-description`과 상호 배타적이며 `--slashes-only`와 함께 사용할 수 없습니다. |
| `--slashes-only` | 꺼짐 | 직접 관리하는 manifest에 병합할 수 있도록 `features.slash_commands`만 출력합니다. |

새 명령을 반영하려면 `hermes update` 후 `hermes slack manifest --write`를 다시 실행하세요.


## `hermes send`

```bash
hermes send --to <target> "message text"
hermes send --to <target> --file <path>
echo "message" | hermes send --to <target>
hermes send --list [platform]
```

agent 또는 gateway loop를 시작하지 않고 구성된 messaging platform으로 일회성 메시지를 보냅니다. gateway에 이미 구성된 자격 증명(`~/.hermes/.env` + `~/.hermes/config.yaml`)을 재사용하므로, 운영 script, cron job, CI hook, monitoring daemon이 각 platform의 REST client를 다시 구현하지 않고도 상태 업데이트를 게시할 수 있습니다.

bot token platform(Telegram, Discord, Slack, Signal, SMS, WhatsApp-CloudAPI)은 실행 중인 gateway가 필요하지 않습니다. `hermes send`가 platform의 REST endpoint와 직접 통신합니다. 지속적인 adapter가 필요한 plugin platform은 여전히 실행 중인 gateway가 필요합니다.

| 옵션 | 설명 |
|------------|-------------|
| `-t`, `--to <TARGET>` | 전송 대상입니다. 형식: `platform`(home channel 사용), `platform:chat_id`, `platform:chat_id:thread_id`, 또는 `platform:#channel-name`. 예: `telegram`, `telegram:-1001234567890`, `discord:#ops`, `slack:C0123ABCD`, `signal:+15551234567`. |
| `-f`, `--file <PATH>` | `PATH`에서 메시지 본문을 읽습니다(텍스트 파일만 — log, report, markdown). stdin에서 강제로 읽으려면 `-`을 전달하세요. image 또는 기타 binary 파일을 보내려면 `MEDIA:<path>`를 사용하세요(아래 참고). |
| `-s`, `--subject <LINE>` | 메시지 본문 앞에 subject/header 행을 추가합니다. |
| `-l`, `--list [platform]` | 모든 platform에 구성된 대상(또는 지정한 platform의 대상만)을 나열합니다. |
| `-q`, `--quiet` | 성공 시 stdout을 억제합니다 — exit code만 확인하는 script에 유용합니다. |
| `--json` | 사람이 읽을 수 있는 출력 대신 원시 JSON 결과를 출력합니다. |

위치 인수인 `message`와 `--file`을 모두 지정하지 않으면, `hermes send`는 stdin이 TTY가 아닐 때 stdin에서 읽습니다. 종료 코드: 성공 시 `0`, 전송/backend 실패 시 `1`, 사용법 오류 시 `2`입니다.
### 이미지 및 기타 미디어 보내기

`--file`은 *텍스트* 본문에만 사용합니다. 이미지, 문서, 동영상 또는 오디오를 네이티브 플랫폼 첨부 파일로 전달하려면 메시지 본문 안에서 `MEDIA:<local_path>` 지시어를 참조하세요.

```bash
hermes send --to telegram "MEDIA:/tmp/screenshot.png"
hermes send --to telegram "Build chart for today MEDIA:/tmp/chart.png"   # with caption
hermes send --to discord:#ops "MEDIA:/tmp/report.pdf"
```

기본적으로 이미지 파일은 사진으로 전송됩니다(Telegram과 같은 플랫폼에서는 이미지가 재압축됨). 압축되지 않은 문서 첨부 파일로 전송하려면 `[[as_document]]`를 추가하세요.

```bash
hermes send --to telegram "[[as_document]] MEDIA:/tmp/screenshot.png"
```

예시:

```bash
hermes send --to telegram "deploy finished"
echo "RAM 92%" | hermes send --to telegram:-1001234567890
hermes send --to discord:#ops --file /tmp/report.md
hermes send --to slack:#eng --subject "[CI]" --file build.log
hermes send --list                  # all platforms
hermes send --list telegram         # filter by platform
```


## `hermes secrets`

```bash
hermes secrets bitwarden <subcommand>
hermes secrets bw <subcommand>          # short alias
```

`~/.hermes/.env`에 저장하는 대신 프로세스 시작 시 외부 시크릿 관리자에서 API 키를 가져옵니다. 현재는 **Bitwarden Secrets Manager**를 지원합니다. 전체 가이드는 [Bitwarden 통합](../user-guide/secrets/bitwarden.md)을 참조하세요.

`bitwarden`(`bw` 별칭) 하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `setup` | 대화형 마법사: 고정된 `bws` 바이너리를 설치하고 액세스 토큰을 저장한 다음 프로젝트를 선택합니다. 비대화형 사용 시 `--project-id`, `--access-token`, `--server-url`을 사용할 수 있습니다. |
| `status` | 현재 설정, 바이너리 경로/버전, 토큰 검증 상태를 표시합니다. |
| `token` | 액세스 토큰을 교체합니다. 저장하기 전에 Bitwarden에서 새 토큰을 검증하며(거부된 토큰은 아무것도 변경하지 않음), 비대화형 사용 시 `--access-token`을 사용하거나 `--no-verify`로 프로브를 건너뛸 수 있습니다. |
| `sync` | 지금 시크릿을 가져오고 변경 사항을 보고합니다. 실제로 시크릿을 현재 셸 환경으로 내보내려면 `--apply`를 추가합니다(기본값은 드라이 런). |
| `install` | 고정된 `bws` 바이너리를 다운로드하고 검증합니다. 관리되는 사본이 이미 있어도 다시 다운로드하려면 `--force`를 사용합니다. |
| `disable` | Bitwarden 통합을 끕니다. |


## `hermes migrate`

```bash
hermes migrate <type>
```

활성 `config.yaml`에서 사용 중단 예정 모델이나 더 이상 사용되지 않는 설정에 대한 참조를 진단하고 (선택적으로) 다시 작성합니다. 다시 작성하기 전에 원본 `config.yaml`의 타임스탬프가 포함된 백업을 생성합니다(`--no-backup`으로 건너뛸 수 있음).

| 하위 명령 | 설명 |
|------------|-------------|
| `xai` | `config.yaml`에서 2026년 5월 15일 폐기 예정인 xAI 모델에 대한 참조를 검색하고, `--apply`를 사용하면 xAI 마이그레이션 가이드에 따른 공식 대체 모델로 파일을 제자리에서 다시 작성합니다. 기본값은 드라이 런입니다. |

마이그레이션 하위 명령의 공통 플래그:

| 플래그 | 설명 |
|------|-------------|
| `--apply` | `config.yaml`을 제자리에서 다시 작성합니다(기본값: 드라이 런, 쓰기 작업 없음). |
| `--no-backup` | 적용할 때 `config.yaml`의 타임스탬프 백업을 건너뜁니다. |

> `hermes claw migrate`(OpenClaw 설정을 Hermes로 일회성 가져오기)와 혼동하지 마세요. `hermes migrate`는 최상위 설정 재작성 명령입니다.


## `hermes proxy`

```bash
hermes proxy <subcommand>
```

OAuth로 인증된 업스트림 제공자(예: Nous Portal, xAI)로 요청을 전달하는 로컬 OpenAI 호환 HTTP 서버를 실행합니다. 외부 앱은 어떤 bearer 토큰으로든 프록시에 연결할 수 있으며, 프록시가 외부로 나가는 요청에 실제 OAuth 자격 증명을 추가합니다. 전체 가이드는 [구독 프록시](../user-guide/features/subscription-proxy.md)를 참조하세요.

| 하위 명령 | 설명 |
|------------|-------------|
| `start` | 포그라운드에서 프록시를 실행합니다. 플래그: `--provider <nous\|xai>`(기본값 `nous`), `--host <addr>`(기본값 `127.0.0.1`; LAN에 노출하려면 `0.0.0.0` 사용), `--port <int>`(기본값 `8645`). |
| `status` | 준비된 프록시 업스트림(자격 증명 존재, OAuth 유효)을 표시합니다. |
| `providers` | 사용 가능한 프록시 업스트림 제공자를 나열합니다. |


## `hermes security`

```bash
hermes security <subcommand>
```

[OSV.dev](https://osv.dev)에 대한 요청 시 취약점 검사를 실행합니다. Hermes venv(설치된 PyPI 배포판), `~/.hermes/plugins/` 아래 플러그인이 선언한 Python 종속성, `config.yaml`에 고정된 `npx`/`uvx` MCP 서버를 검사합니다. 전역 설치 패키지나 에디터/브라우저 확장 프로그램은 검사하지 않습니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `audit` | 일회성 공급망 감사를 실행합니다. |

`audit` 플래그:

| 플래그 | 기본값 | 설명 |
|------|---------|-------------|
| `--json` | off | 사람이 읽을 수 있는 텍스트 대신 기계 판독용 JSON을 출력합니다. |
| `--fail-on <level>` | `critical` | 발견 항목이 이 심각도(`low`, `moderate`, `high`, `critical`) 이상이면 0이 아닌 상태 코드로 종료합니다. |
| `--skip-venv` | off | Hermes Python venv 검사를 건너뜁니다. |
| `--skip-plugins` | off | 플러그인 요구 사항 파일 검사를 건너뜁니다. |
| `--skip-mcp` | off | `config.yaml`에 고정된 MCP 서버 검사를 건너뜁니다. |


## `hermes login` / `hermes logout` *(사용 중단됨)*

:::caution
`hermes login`은 제거되었습니다. OAuth 자격 증명을 관리하려면 `hermes auth`, 제공자를 선택하려면 `hermes model`, 전체 대화형 설정을 사용하려면 `hermes setup`을 사용하세요.
:::

## `hermes auth`

동일 제공자의 키를 교체하기 위한 자격 증명 풀을 관리합니다. 전체 문서는 [자격 증명 풀](/user-guide/features/credential-pools)을 참조하세요.

```bash
hermes auth                                              # Interactive wizard
hermes auth list                                         # Show all pools
hermes auth list openrouter                              # Show specific provider
hermes auth add openrouter --api-key sk-or-v1-xxx        # Add API key
hermes auth add anthropic --type oauth                   # Add OAuth credential
hermes auth remove openrouter 2                          # Remove by index
hermes auth reset openrouter                             # Clear cooldowns
hermes auth status anthropic                             # Show auth status for a provider
hermes auth logout anthropic                             # Log out and clear stored auth state
hermes auth spotify                                      # Authenticate Hermes with Spotify via PKCE
```

하위 명령은 `add`, `list`, `remove`, `reset`, `status`, `logout`, `spotify`입니다. 하위 명령 없이 호출하면 대화형 관리 마법사를 시작합니다.

## `hermes status`

```bash
hermes status [--all] [--deep]
```

| 옵션 | 설명 |
|--------|-------------|
| `--all` | 공유 가능한 비식별화 형식으로 모든 세부 정보를 표시합니다. |
| `--deep` | 시간이 더 걸릴 수 있는 심층 검사를 실행합니다. |

## `hermes cron`

```bash
hermes cron <list|create|edit|pause|resume|run|remove|status|tick>
```

| 하위 명령 | 설명 |
|------------|-------------|
| `list` | 예약된 작업을 표시합니다. |
| `create` / `add` | 프롬프트로 예약 작업을 생성하며, 반복되는 `--skill`을 통해 하나 이상의 스킬을 선택적으로 연결할 수 있습니다. |
| `edit` | 작업의 일정, 프롬프트, 이름, 전달 방식, 반복 횟수 또는 연결된 스킬을 업데이트합니다. `--clear-skills`, `--add-skill`, `--remove-skill`을 지원합니다. |
| `pause` | 삭제하지 않고 작업을 일시 중지합니다. |
| `resume` | 일시 중지된 작업을 재개하고 다음 미래 실행 시각을 계산합니다. |
| `run` | 다음 스케줄러 틱에서 작업을 트리거합니다. |
| `remove` | 예약 작업을 삭제합니다. |
| `status` | cron 스케줄러가 실행 중인지 확인합니다. |
| `tick` | 실행 기한이 된 작업을 한 번 실행한 후 종료합니다. |

cron **트리거**는 `cron.provider` 설정 키를 통해 플러그인 방식으로 구성할 수 있습니다. 비어 있으면(기본값) 내장 프로세스 내 틱커를 사용합니다. `chronos`로 설정하면(`portal_url`, `callback_url`, `expected_audience`, `nas_jwks_url` 키로 구성되는 스케일 투 제로 호스팅 게이트웨이용 NAS 관리 제공자) 사용하거나, `plugins/cron/<name>/` 또는 `$HERMES_HOME/plugins/<name>/` 아래의 사용자 지정 제공자 이름을 지정할 수 있습니다. 알 수 없거나 사용할 수 없는 제공자는 내장 제공자로 폴백되므로 cron에 트리거가 없는 상태가 되지 않습니다. [cron 내부 구조](../developer-guide/cron-internals.md#gateway-integration) 문서를 참조하세요.

## `hermes kanban`

```bash
hermes kanban [--board <slug>] <action> [options]
```

다중 프로필·다중 프로젝트 협업 보드입니다. 각 설치에서 여러 보드(프로젝트, 저장소 또는 도메인마다 하나)를 호스팅할 수 있으며, 각 보드는 자체 SQLite DB와 디스패처 범위를 가진 독립 큐입니다. 새 설치는 `default`라는 보드 하나로 시작하며, 이 보드의 DB는 이전 버전과의 호환성을 위해 `~/.hermes/kanban.db`에 있습니다. 추가 보드는 `~/.hermes/kanban/boards/<slug>/kanban.db`에 저장됩니다. 게이트웨이에 내장된 디스패처는 매 틱마다 모든 보드를 순회합니다.

**전역 플래그(아래의 모든 작업에 적용):**

| 플래그 | 목적 |
|------|---------|
| `--board <slug>` | 특정 보드에서 작업합니다. 현재 보드(`hermes kanban boards switch`, `HERMES_KANBAN_BOARD` 환경 변수 또는 `default`로 설정)로 기본 설정됩니다. |

**사람 및 스크립트를 위한 인터페이스입니다.** 디스패처가 생성한 에이전트 워커는 셸에서 `hermes kanban`을 실행하는 대신 전용 `kanban_*` [툴셋](/user-guide/features/kanban#how-workers-interact-with-the-board)(`kanban_show`, `kanban_complete`, `kanban_request_review`, `kanban_request_changes`, `kanban_block`, `kanban_create`, `kanban_link`, `kanban_comment`, `kanban_heartbeat`; 오케스트레이터 프로필에는 `kanban_list`와 `kanban_unblock`도 제공)을 통해 보드를 관리합니다. 워커의 환경에는 `HERMES_KANBAN_BOARD`가 고정되어 있으므로 물리적으로 다른 보드를 볼 수 없습니다.

| 작업 | 목적 |
|--------|---------|
| `init` | 없으면 `kanban.db`를 생성합니다. 멱등적입니다. |
| `boards list` / `boards ls` | 작업 수와 함께 모든 보드를 나열합니다. `--json`, `--all`(보관된 보드 포함)을 사용할 수 있습니다. |
| `boards create <slug>` | 새 보드를 생성합니다. 플래그: `--name`, `--description`, `--icon`, `--color`, `--switch`(활성화). slug는 kebab-case이며 자동으로 소문자화됩니다. |
| `boards switch <slug>` / `boards use` | `<slug>`를 활성 보드로 저장합니다(`~/.hermes/kanban/current`에 기록). |
| `boards show` / `boards current` | 현재 활성 보드의 이름, DB 경로 및 작업 수를 출력합니다. |
| `boards rename <slug> "<name>"` | 보드의 표시 이름을 변경합니다. slug는 변경할 수 없습니다. |
| `boards rm <slug>` | 보드를 보관(기본값)하거나 완전히 삭제합니다. `--delete`는 보관 단계를 건너뜁니다. 보관된 보드는 `boards/_archived/<slug>-<ts>/`로 이동합니다. `default`에는 사용할 수 없습니다. |
| `create "<title>"` | 활성 보드에 새 작업을 생성합니다. 플래그: `--body`, `--assignee`, `--parent`(반복 가능), `--workspace scratch\|worktree\|dir:<path>`, `--tenant`, `--priority`, `--triage`, `--idempotency-key`, `--max-runtime`, `--max-retries`, `--skill`(반복 가능). |
| `list` / `ls` | 활성 보드의 작업을 나열합니다. `--mine`, `--assignee`, `--status`, `--tenant`, `--archived`, `--json`으로 필터링합니다. |
| `show <id>` | 댓글과 이벤트를 포함해 작업을 표시합니다. 기계 출력에는 `--json`을 사용합니다. |
| `assign <id> <profile>` | 할당하거나 재할당합니다. 할당을 취소하려면 `none`을 사용합니다. 작업 실행 중에는 사용할 수 없습니다. |
| `link <parent> <child>` | 종속성을 추가합니다. 순환을 감지합니다. 두 작업은 같은 보드에 있어야 합니다. |
| `unlink <parent> <child>` | 종속성을 제거합니다. |
| `claim <id>` | 준비된 작업을 원자적으로 할당받습니다. 확인된 작업 공간 경로를 출력합니다. |
| `comment <id> "<text>"` | 댓글을 추가합니다. 다음에 작업을 할당받는 워커는 `kanban_show()` 응답의 일부로 이 내용을 읽습니다. |
| `complete <id>` | 작업을 완료로 표시합니다. 플래그: `--result`, `--summary`, `--metadata`. |
| `block <id> "<reason>"` | 사람의 입력이 필요하도록 작업을 차단됨으로 표시합니다. 이유도 댓글로 추가합니다. |
| `request-review <id>` | 검토자에게 인계하며 작업을 `review`로 이동합니다. 차단이 아닙니다. 플래그: `--summary`, `--metadata`, `--reviewer`(검토 디스패치 전에 재할당). |
| `request-changes <id> <reason>` | 활성 검토 실행에 대한 검토자 판정입니다. 검토 시도를 종료하고 작업을 원래 구현자에게 돌려보냅니다. |
| `reopen-review <id>...` | 검토 작업을 변경을 위해 되돌립니다(`review` → ready/todo). 플래그: `--reason`(댓글로 추가). |
| `schedule <id> "<reason>"` | 시간 지연/후속 작업을 `scheduled`에 보관하여 사람의 차단 항목으로 표시되지 않게 합니다. |
| `unblock <id>` | 차단된 작업을 원래 단계(`review` 또는 `ready`)로 복원하거나, 종속성이 아직 열려 있으면 `todo`로 복원합니다. |
| `archive <id>` | 기본 목록에서 숨깁니다. `gc`가 스크래치 작업 공간을 제거합니다. |
| `tail <id>` | 작업의 이벤트 스트림을 추적합니다. |
| `dispatch` | 활성 보드에서 디스패처를 한 번 실행합니다. 플래그: `--dry-run`, `--max N`, `--failure-limit N`, `--json`. |
| `context <id>` | 워커가 보게 될 전체 컨텍스트(제목 + 본문 + 상위 작업 결과 + 댓글)를 출력합니다. |
| `specify <id>` / `specify --all` | 보조 LLM을 통해 triage 열의 작업을 구체적인 사양(목표, 접근 방식, 인수 기준이 포함된 제목 + 본문)으로 구체화한 다음 `todo`로 승격합니다. 플래그: `--tenant`(`--all`을 한 테넌트로 제한), `--author`, `--json`. 모델은 `config.yaml`의 `auxiliary.triage_specifier`에서 구성합니다. |
| `decompose <id>` / `decompose --all` | triage 열의 작업을 설명에 따라 전문 프로필로 라우팅되는 하위 작업 그래프로 분할합니다. LLM이 팬아웃의 이점이 없다고 판단하면 specify 방식의 단일 작업 승격으로 폴백합니다. `specify`와 동일한 플래그를 사용합니다. 분해기 모델은 `config.yaml`의 `auxiliary.kanban_decomposer`에서 구성합니다. `kanban.orchestrator_profile`은 팬아웃 후 루트/오케스트레이션 작업의 담당자만 제어합니다. `kanban.auto_decompose: true`(기본값)인 경우 모든 디스패처 틱에서 자동으로 실행됩니다. [자동 및 수동 오케스트레이션](/user-guide/features/kanban#auto-vs-manual-orchestration)을 참조하세요. |
| `gc` | 보관된 작업의 스크래치 작업 공간을 제거합니다. |

예시:

```bash
# Create a second board and put a task on it without switching away.
hermes kanban boards create atm10-server --name "ATM10 Server" --icon 🎮
hermes kanban --board atm10-server create "Restart server" --assignee ops

# Switch the active board for subsequent calls.
hermes kanban boards switch atm10-server
hermes kanban list                  # shows atm10-server tasks

# Archive a board (recoverable) or hard-delete it.
hermes kanban boards rm atm10-server
hermes kanban boards rm atm10-server --delete
```

보드 확인 순서(우선순위가 높은 순): `--board <slug>` 플래그 → `HERMES_KANBAN_BOARD` 환경 변수 → `~/.hermes/kanban/current` 파일 → `default`.

모든 작업은 게이트웨이의 슬래시 명령(`/kanban …`)으로도 사용할 수 있으며, `boards` 하위 명령과 `--board` 플래그를 포함해 동일한 인수 인터페이스를 제공합니다.

전체 설계( Cline Kanban / Paperclip / NanoClaw / Gemini Enterprise와의 비교, 8가지 협업 패턴, 4가지 사용자 스토리, 동시성 정확성 증명)는 저장소의 `docs/hermes-kanban-v1-spec.pdf` 또는 [Kanban 사용자 가이드](/user-guide/features/kanban)를 참조하세요.
## `hermes egress`

원격 터미널 샌드박스를 위한 외부 자격 증명 주입 방화벽입니다. [iron-proxy](https://github.com/ironsh/iron-proxy) 데몬을 감싸며, 샌드박스가 실제 키를 보유하지 않도록 네트워크 경계에서 불투명한 프록시 토큰을 실제 업스트림 API 자격 증명으로 교체합니다. 기본적으로 비활성화되어 있습니다. 설정 및 아키텍처는 전체 [Egress 프록시](../user-guide/egress/iron-proxy.md) 페이지를 참조하세요.

```bash
hermes egress install                  # download the pinned iron-proxy binary
hermes egress install --force          # re-download even if already installed

hermes egress setup                    # interactive wizard: CA, mappings, config
hermes egress setup --tunnel-port N    # override the tunnel listener port (default 9090)
hermes egress setup --from-bitwarden   # use Bitwarden Secrets Manager as credential source
hermes egress setup --no-bitwarden     # explicitly switch back to env-based credentials
hermes egress setup --rotate-tokens    # mint fresh proxy tokens (default preserves existing)

hermes egress start                    # spawn the managed proxy daemon
hermes egress stop                     # SIGTERM (then SIGKILL after 5s grace)
hermes egress restart                  # stop (if running) then start — needed for secret changes
hermes egress reload                   # hot-reload the ruleset in-place (no restart, no dropped
                                       #   connections) via the loopback management API

hermes egress status                   # binary + config + pid + listening + mappings
hermes egress status --show-tokens     # print proxy tokens in full (default: redacted)

hermes egress disable                  # flip proxy.enabled = false (does not stop a running proxy)
hermes egress config                   # print the path to proxy.yaml for inspection
```

### 일반적인 흐름

```bash
# First-time setup
export OPENROUTER_API_KEY=…
hermes egress setup && hermes egress start
hermes config set terminal.backend docker   # if not already

# Switching credential source after the fact
hermes egress setup --from-bitwarden       # env → bitwarden
hermes egress setup --no-bitwarden         # bitwarden → env
# (just `setup` without either flag preserves the existing mode)

# Rotating all tokens (e.g. after a suspected token leak)
hermes egress setup --rotate-tokens    # setup offers to restart the running daemon for you
# (running sandboxes still hold old tokens; restart them too)

# Adding a new upstream
# Edit ~/.hermes/config.yaml proxy.extra_allowed_hosts: [api.example.com]
hermes egress setup
hermes egress restart                  # one-command apply (stop + start)
```

### 진단 바로 가기

```bash
hermes egress status                     # current state in one view
cat ~/.hermes/proxy/proxy.yaml           # the rendered iron-proxy config
tail -20 ~/.hermes/proxy/iron-proxy.log  # daemon-level diagnostics
tail -f ~/.hermes/proxy/iron-proxy.log | jq  # daemon + per-request log (line-delimited JSON; v0.39 combines both streams)
```

일반적인 오류와 복구 방법은 [Egress 프록시 → 문제 해결](../user-guide/egress/iron-proxy.md#troubleshooting)에서 다룹니다.

## `hermes project`

```bash
hermes project <create|list|show|add-folder|remove-folder|rename|set-primary|use|archive|restore|bind-board>
```

프로젝트는 여러 폴더/저장소에 걸칠 수 있는 사람이 지정한 작업 공간입니다. 데스크톱 세션 그룹화를 위한 기준이 되며, 칸반 보드에 연결하면 작업에 결정적인 worktree 및 브랜치 규칙을 부여합니다. 상태는 프로필별로 관리됩니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `create` | 새 프로젝트를 만듭니다. |
| `list` (별칭 `ls`) | 프로젝트를 나열합니다. |
| `show` | 프로젝트의 세부 정보를 표시합니다. |
| `add-folder` | 프로젝트에 폴더/저장소를 추가합니다. |
| `remove-folder` | 프로젝트에서 폴더를 제거합니다. |
| `rename` | 프로젝트 이름을 변경합니다. |
| `set-primary` | 기본 폴더를 설정합니다. |
| `use` | 활성 프로젝트를 설정합니다. |
| `archive` | 프로젝트를 보관합니다(복구 가능). |
| `restore` | 보관된 프로젝트를 복원합니다. |
| `bind-board` | 칸반 보드를 이 프로젝트에 연결합니다. |

## `hermes webhook`

```bash
hermes webhook <subscribe|list|remove|test>
```

이벤트 기반 에이전트 활성화를 위한 동적 웹훅 구독을 관리합니다. 설정에서 웹훅 플랫폼을 활성화해야 합니다. 구성되지 않은 경우 설정 안내를 출력합니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `subscribe` / `add` | 웹훅 경로를 생성합니다. 서비스에 설정할 URL과 HMAC 시크릿을 반환합니다. |
| `list` / `ls` | 에이전트가 생성한 모든 구독을 표시합니다. |
| `remove` / `rm` | 동적 구독을 삭제합니다. config.yaml의 정적 경로에는 영향을 주지 않습니다. |
| `test` | 테스트 POST를 보내 구독이 작동하는지 확인합니다. |

### `hermes webhook subscribe`

```bash
hermes webhook subscribe <name> [options]
```

| 옵션 | 설명 |
|--------|-------------|
| `--prompt` | `{dot.notation}` 페이로드 참조를 포함한 프롬프트 템플릿입니다. |
| `--events` | 수락할 쉼표로 구분된 이벤트 유형입니다(예: `issues,pull_request`). 비워 두면 모두 수락합니다. |
| `--description` | 사람이 읽을 수 있는 설명입니다. |
| `--skills` | 에이전트 실행에 로드할 쉼표로 구분된 스킬 이름입니다. |
| `--deliver` | 전달 대상: `log` (기본값), `telegram`, `discord`, `slack`, `github_comment`. |
| `--deliver-chat-id` | 플랫폼 간 전달을 위한 대상 채팅/채널 ID입니다. |
| `--secret` | 사용자 지정 HMAC 시크릿입니다. 생략하면 자동 생성됩니다. |
| `--deliver-only` | 에이전트를 건너뛰고 렌더링된 `--prompt`를 리터럴 메시지로 전달합니다. LLM 비용이 없으며 1초 이내에 전달됩니다. `--deliver`를 실제 대상(`log`가 아님)으로 설정해야 합니다. |
| `--script` | `~/.hermes/scripts/` 아래의 필터/변환 스크립트입니다. 웹훅 페이로드는 stdin으로 JSON 형식으로 전달되며, JSON stdout이 페이로드를 대체합니다. stdout이 비어 있거나 `[SILENT]`이거나 종료 코드가 0이 아니면 웹훅을 무시합니다. [스크립트 필터 및 변환](../user-guide/messaging/webhooks.md#script-filters-and-transforms)을 참조하세요. |

구독은 `~/.hermes/webhook_subscriptions.json`에 저장되며, 게이트웨이를 재시작하지 않아도 웹훅 어댑터가 핫 리로드합니다.

## `hermes doctor`

```bash
hermes doctor [--fix]
```

| 옵션 | 설명 |
|--------|-------------|
| `--fix` | 가능한 경우 자동 복구를 시도합니다. |

## `hermes dump`

```bash
hermes dump [--show-keys]
```

Hermes 전체 설정의 간결한 일반 텍스트 요약을 출력합니다. 지원을 요청할 때 Discord, GitHub 이슈 또는 Telegram에 복사해 붙여 넣을 수 있도록 설계되었습니다. ANSI 색상이나 특수 서식 없이 데이터만 출력합니다.

| 옵션 | 설명 |
|--------|-------------|
| `--show-keys` | `set`/`not set`만 표시하는 대신 마스킹된 API 키 접두사(처음과 마지막 4자)를 표시합니다. |

### 포함 내용

| 섹션 | 세부 정보 |
|---------|---------|
| **헤더** | Hermes 버전, 릴리스 날짜, git 커밋 해시 |
| **환경** | OS, Python 버전, OpenAI SDK 버전 |
| **식별 정보** | 활성 프로필 이름, HERMES_HOME 경로 |
| **모델** | 구성된 기본 모델 및 제공자 |
| **터미널** | 백엔드 유형(local, docker, ssh 등) |
| **API 키** | 22개 제공자/도구 API 키 모두의 존재 여부 |
| **기능** | 활성화된 도구 세트, MCP 서버 수, 메모리 제공자 |
| **서비스** | 게이트웨이 상태, 구성된 메시징 플랫폼 |
| **작업량** | Cron 작업 수, 설치된 스킬 수 |
| **설정 재정의** | 기본값과 다른 모든 설정 값 |

### 출력 예시

```
--- hermes dump ---
version:          0.8.0 (2026.4.8) [af4abd2f]
os:               Linux 6.14.0-37-generic x86_64
python:           3.11.14
openai_sdk:       2.24.0
profile:          default
hermes_home:      ~/.hermes
model:            anthropic/claude-opus-4.6
provider:         openrouter
terminal:         local

api_keys:
  openrouter           set
  openai               not set
  anthropic            set
  nous                 not set
  firecrawl            set
  ...

features:
  toolsets:           all
  mcp_servers:        0
  memory_provider:    built-in
  gateway:            running (systemd)
  platforms:          telegram, discord
  cron_jobs:          3 active / 5 total
  skills:             42

config_overrides:
  agent.max_turns: 250
  compression.threshold: 0.85
  display.streaming: True
--- end dump ---
```

### 사용 시점

- GitHub에서 버그를 보고할 때 — 덤프를 이슈에 붙여 넣습니다.
- Discord에서 도움을 요청할 때 — 코드 블록으로 공유합니다.
- 자신의 설정을 다른 사람의 설정과 비교할 때
- 무언가 작동하지 않을 때 빠르게 상태를 점검할 때

:::tip
`hermes dump`는 공유를 위해 특별히 설계되었습니다. 대화형 진단에는 `hermes doctor`를 사용하세요. 시각적 개요가 필요하면 `hermes status`를 사용하세요.
:::

## `hermes debug`

```bash
hermes debug share [options]
```

디버그 보고서(시스템 정보 및 최근 로그)를 붙여넣기 서비스에 업로드하고 공유 가능한 URL을 가져옵니다. 빠르게 지원을 요청할 때 유용하며, 문제를 진단하는 데 필요한 모든 정보가 포함됩니다.

| 옵션 | 설명 |
|--------|-------------|
| `--lines <N>` | 로그 파일마다 포함할 로그 줄 수입니다(기본값: 200). |
| `--expire <days>` | 붙여넣기 만료 기간(일)입니다(기본값: 7). |
| `--nous` | 공개 붙여넣기 서비스 대신 Nous 내부 진단 저장소에 업로드합니다. Nous 지원팀에서 비공개 진단 번들을 요청한 경우 사용하세요. |
| `--local` | 업로드하지 않고 보고서를 로컬에 출력합니다. |
| `--no-redact` | 업로드 시 시크릿 마스킹을 비활성화합니다. 기본적으로 업로드 내용은 마스킹됩니다. |

보고서에는 시스템 정보(OS, Python 버전, Hermes 버전), 최근 에이전트·게이트웨이·GUI/대시보드·데스크톱 로그(파일당 512KB 제한), 마스킹된 API 키 상태가 포함됩니다. 기본적으로 업로드 내용은 마스킹되므로 시크릿이 포함되지 않습니다.

기본 업로드는 다음 공개 붙여넣기 서비스를 순서대로 시도합니다: paste.rs, dpaste.com. `--nous`를 사용하면 동일한 디버그 번들을 대신 비공개 Nous 진단 저장소에 업로드합니다. 반환되는 뷰어 링크는 Nous 팀 전용이며 14일 후 자동 삭제됩니다.

### 예시

```bash
hermes debug share              # Upload debug report, print URL
hermes debug share --lines 500  # Include more log lines
hermes debug share --expire 30  # Keep paste for 30 days
hermes debug share --nous       # Upload a private diagnostics bundle for Nous support
hermes debug share --local      # Print report to terminal (no upload)
```

## `hermes backup`

```bash
hermes backup [options]
```

Hermes 설정, 스킬, 세션 및 데이터를 zip 아카이브로 만듭니다. hermes-agent 코드베이스 자체는 백업에서 제외됩니다.

| 옵션 | 설명 |
|--------|-------------|
| `-o`, `--output <path>` | zip 파일의 출력 경로입니다(기본값: `~/hermes-backup-<timestamp>.zip`). |
| `-q`, `--quick` | 빠른 스냅샷입니다. 중요한 상태 파일(config.yaml, state.db, .env, auth, cron 작업)만 포함하며 전체 백업보다 훨씬 빠릅니다. |
| `-l`, `--label <name>` | 스냅샷의 레이블입니다(`--quick`과 함께 사용할 때만 사용). |

백업은 안전한 복사를 위해 SQLite의 `backup()` API를 사용하므로 Hermes가 실행 중이어도 올바르게 작동합니다.

**zip에서 제외되는 항목:**

- `*.db-wal`, `*.db-shm`, `*.db-journal` — SQLite의 WAL/공유 메모리/저널 사이드카입니다. `*.db` 파일은 `sqlite3.backup()`을 통해 이미 일관된 스냅샷을 얻었으므로, 라이브 사이드카를 함께 제공하면 복원 시 커밋이 절반만 적용된 상태를 보게 됩니다.
- `checkpoints/` — 세션별 trajectory 캐시입니다. 해시 키로 관리되며 세션마다 다시 생성되므로 다른 설치 환경으로 깔끔하게 이식되지 않습니다.
- `hermes-agent` 코드 자체(이것은 저장소 스냅샷이 아니라 사용자 데이터 백업입니다).

### 예시

```bash
hermes backup                           # Full backup to ~/hermes-backup-*.zip
hermes backup -o /tmp/hermes.zip        # Full backup to specific path
hermes backup --quick                   # Quick state-only snapshot
hermes backup --quick --label "pre-upgrade"  # Quick snapshot with label
```

## `hermes checkpoints`

```bash
hermes checkpoints [COMMAND]
```

`~/.hermes/checkpoints/`의 shadow git 저장소를 검사하고 관리합니다. 이 저장소는 세션 내 `/rollback` 명령을 뒷받침하는 저장소 계층입니다. 언제든 안전하게 실행할 수 있으며 에이전트가 실행 중일 필요가 없습니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `status` (기본값) | 전체 크기, 프로젝트 수 및 프로젝트별 세부 내역을 표시합니다. 인수 없이 실행한 `hermes checkpoints`와 같습니다. |
| `list` | `status`의 별칭입니다. |
| `prune` | 정리 작업을 강제로 실행합니다. 고아 및 오래된 프로젝트를 삭제하고 저장소를 GC하며 크기 제한을 적용합니다. 24시간 멱등성 마커를 무시합니다. |
| `clear` | 전체 체크포인트 기반을 삭제합니다. 되돌릴 수 없으며 `-f`를 사용하지 않으면 확인을 요청합니다. |
| `clear-legacy` | v1→v2 마이그레이션에서 생성된 `legacy-<timestamp>/` 아카이브만 삭제합니다. |
### 옵션

| 옵션 | 하위 명령 | 설명 |
|--------|------------|-------------|
| `--limit N` | `status`, `list` | 나열할 프로젝트의 최대 개수(기본값 20). |
| `--retention-days N` | `prune` | `last_touch`가 N일보다 오래된 프로젝트를 삭제합니다(기본값 7). |
| `--max-size-mb N` | `prune` | 고아/오래된 항목을 정리한 후 전체 저장소 크기가 N MB 이하가 될 때까지 프로젝트별로 가장 오래된 커밋을 삭제합니다(기본값 500). |
| `--keep-orphans` | `prune` | 작업 디렉터리가 더 이상 존재하지 않는 프로젝트를 삭제하지 않습니다. |
| `-f`, `--force` | `clear`, `clear-legacy` | 확인 프롬프트를 건너뜁니다. |

### 예시

```bash
hermes checkpoints                                  # status overview
hermes checkpoints prune --retention-days 3         # aggressive cleanup
hermes checkpoints prune --max-size-mb 200          # tighten size cap once
hermes checkpoints clear-legacy -f                  # drop v1 archive dirs
hermes checkpoints clear -f                         # wipe everything
```

전체 아키텍처와 세션 내 명령은 [체크포인트 및 `/rollback`](../user-guide/checkpoints-and-rollback.md)을 참조하세요.

## `hermes import`

```bash
hermes import <zipfile> [options]
```

이전에 생성한 Hermes 백업을 Hermes 홈 디렉터리로 복원합니다. 아카이브의 모든 파일이 Hermes 홈에 있는 기존 파일을 덮어쓰며, `--force`는 대상에 이미 Hermes 설치가 있을 때 표시되는 확인 프롬프트만 건너뜁니다.

| 옵션 | 설명 |
|------------|-------------|
| `-f`, `--force` | 기존 설치 확인 프롬프트를 건너뜁니다. |

:::warning
실행 중인 프로세스와의 충돌을 방지하려면 가져오기 전에 게이트웨이를 중지하세요.
:::

### 예시
```bash
hermes import ~/hermes-backup-20260423.zip           # Prompts before overwriting existing config
hermes import ~/hermes-backup-20260423.zip --force   # Overwrite without prompting
```

## `hermes logs`

```bash
hermes logs [log_name] [options]
```

Hermes 로그 파일을 확인하고, 실시간으로 추적하며, 필터링합니다. 모든 로그는 `~/.hermes/logs/`에 저장됩니다(기본 프로필이 아닌 경우 `<profile>/logs/`).

### 로그 파일

| 이름 | 파일 | 기록 내용 |
|------|------|-----------------|
| `agent` (기본값) | `agent.log` | 모든 에이전트 활동 — API 호출, 도구 디스패치, 세션 수명 주기(INFO 이상) |
| `errors` | `errors.log` | 경고와 오류만 — agent.log에서 필터링된 일부 내용 |
| `gateway` | `gateway.log` | 메시징 게이트웨이 활동 — 플랫폼 연결, 메시지 디스패치, 웹훅 이벤트 |
| `gui` | `gui.log` | 대시보드 / TUI 게이트웨이 / PTY 브리지 / 웹소켓 이벤트 |
| `desktop` | `desktop.log` | Electron 데스크톱 앱 — 부팅, 백엔드 생성 출력, 최근 Python 트레이스백 |

### 옵션

| 옵션 | 설명 |
|------------|-------------|
| `log_name` | 확인할 로그: `agent`(기본값), `errors`, `gateway`, 또는 크기와 함께 사용 가능한 파일을 표시하는 `list`. |
| `-n`, `--lines <N>` | 표시할 줄 수(기본값: 50). |
| `-f`, `--follow` | `tail -f`처럼 로그를 실시간으로 추적합니다. 중지하려면 Ctrl+C를 누르세요. |
| `--level <LEVEL>` | 표시할 최소 로그 수준: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `--session <ID>` | 세션 ID 부분 문자열이 포함된 줄만 필터링합니다. |
| `--since <TIME>` | 상대적인 과거 시점부터 로그를 표시합니다: `30m`, `1h`, `2d` 등. `s`(초), `m`(분), `h`(시간), `d`(일)를 지원합니다. |
| `--component <NAME>` | 컴포넌트별로 필터링합니다: `gateway`, `agent`, `tools`, `cli`, `cron`. |

### 예시

```bash
# View the last 50 lines of agent.log (default)
hermes logs

# Follow agent.log in real time
hermes logs -f

# View the last 100 lines of gateway.log
hermes logs gateway -n 100

# Show only warnings and errors from the last hour
hermes logs --level WARNING --since 1h

# Filter by a specific session
hermes logs --session abc123

# Follow errors.log, starting from 30 minutes ago
hermes logs errors --since 30m -f

# List all log files with their sizes
hermes logs list
```

### 필터링

필터는 조합해서 사용할 수 있습니다. 여러 필터가 활성화된 경우 로그 줄이 표시되려면 **모든** 필터를 통과해야 합니다.

```bash
# WARNING+ lines from the last 2 hours containing session "tg-12345"
hermes logs --level WARNING --since 2h --session tg-12345
```

구문 분석 가능한 타임스탬프가 없는 줄은 `--since`가 활성화된 경우 포함됩니다(여러 줄로 구성된 로그 항목의 연속 줄일 수 있습니다). 감지 가능한 수준이 없는 줄은 `--level`이 활성화된 경우 포함됩니다.

### 로그 순환

Hermes는 Python의 `RotatingFileHandler`를 사용합니다. 오래된 로그는 자동으로 순환되므로 `agent.log.1`, `agent.log.2` 등을 확인하세요. `hermes logs list` 하위 명령은 순환된 파일을 포함한 모든 로그 파일을 표시합니다.


## `hermes prompt-size`

```bash
hermes prompt-size [--platform <name>] [--json]
```

새 세션의 고정 프롬프트 예산, 즉 대화 내용이 포함되기 *전에* 모든 API 호출에 전송되는 내용을 보고합니다. 다운스트림 어댑터나 프록시의 프롬프트 예산이 모델의 컨텍스트 창보다 작을 때, 또는 어떤 블록(스킬 인덱스, 메모리, 프로필)이 가장 큰 비중을 차지하는지 확인할 때 유용합니다.

에이전트가 사용할 것과 동일한 시스템 프롬프트를 구성한 다음 다음과 같이 세부적으로 나눕니다.

- **시스템 프롬프트 전체** — 조합된 전체 프롬프트(정체성, 지침, 스킬 인덱스, 컨텍스트 파일, 메모리, 프로필, 타임스탬프).
- **스킬 인덱스** — `<available_skills>` 블록. 설치된 스킬이 많을 때 단일 블록으로는 가장 큰 경우가 많습니다.
- **메모리** 및 **사용자 프로필** — `MEMORY.md` / `USER.md` 스냅샷.
- **프롬프트 계층** — Hermes가 캐시 친화성을 위해 프롬프트를 계층화하는 방식에 맞춘 stable / context / volatile.
- **도구 스키마** — 활성화된 모든 도구의 JSON(호출당 고정 페이로드의 나머지 절반).

완전히 오프라인으로 실행되므로 API 호출이 없고, 자격 증명을 설정하지 않아도 작동합니다.

```bash
# Human-readable breakdown for the CLI platform (default)
hermes prompt-size

# Simulate a messaging platform's prompt (different platform hint)
hermes prompt-size --platform telegram

# Machine-readable output for scripts
hermes prompt-size --json
```

:::tip
스킬 인덱스와 도구 스키마의 크기는 활성화한 스킬과 도구의 수에 따라 증가합니다. 프롬프트를 줄이려면 사용하지 않는 도구 세트를 비활성화하거나(`hermes tools`), 필요하지 않은 스킬을 제거하세요(`hermes skills`). 현재 디렉터리의 컨텍스트 파일(AGENTS.md, .cursorrules)도 전체 크기에 포함됩니다.
:::

## `hermes config`

```bash
hermes config <subcommand>
```

하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `show` | 현재 구성 값을 표시합니다. |
| `edit` | 편집기에서 `config.yaml`을 엽니다. |
| `get <key> [--json]` | 점으로 구분된 키(예: `hermes config get model.default`)로 단일 구성 값을 출력합니다. `--json`은 기계 판독 가능한 출력을 생성합니다. |
| `set <key> <value>` | 구성 값을 설정합니다. |
| `unset <key>` | 구성 키를 제거하여 내장 기본값으로 되돌립니다. |
| `path` | 구성 파일 경로를 출력합니다. |
| `env-path` | `.env` 파일 경로를 출력합니다. |
| `check` | 누락되었거나 오래된 구성을 확인합니다. |
| `migrate` | 새로 도입된 옵션을 대화형으로 추가합니다. |

## `hermes pairing`

```bash
hermes pairing <list|approve|revoke|clear-pending>
```

| 하위 명령 | 설명 |
|------------|-------------|
| `list` | 대기 중인 사용자와 승인된 사용자를 표시합니다. |
| `approve <platform> <code>` | 페어링 코드를 승인합니다. |
| `revoke <platform> <user-id>` | 사용자의 액세스 권한을 취소합니다. |
| `clear-pending` | 대기 중인 페어링 코드를 삭제합니다. |

## `hermes skills`

```bash
hermes skills <subcommand>
```

하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `browse` | 스킬 레지스트리를 페이지 단위로 탐색합니다. |
| `search` | 스킬 레지스트리를 검색합니다. |
| `install` | 스킬을 설치합니다. |
| `inspect` | 설치하지 않고 스킬을 미리 봅니다. |
| `list` | 설치된 스킬을 나열합니다. |
| `check` | 설치된 허브 스킬에 업스트림 업데이트가 있는지 확인합니다. |
| `update` | 사용 가능한 경우 업스트림 변경 사항과 함께 허브 스킬을 다시 설치합니다. |
| `audit` | 설치된 허브 스킬을 다시 검색합니다. |
| `uninstall` | 허브에서 설치한 스킬을 제거합니다. |
| `reset` | 매니페스트 항목을 삭제하여 `user_modified`로 표시된 번들 스킬의 고정을 해제합니다. `--restore`를 사용하면 사용자 사본도 번들 버전으로 교체합니다. |
| `opt-out` | 활성 프로필에 번들 스킬이 시드되지 않도록 합니다. `.no-bundled-skills` 마커를 기록하여 설치 프로그램, `hermes update`, 모든 동기화 과정에서 번들 스킬 시드를 건너뛰게 합니다. 기본적으로 안전하며 디스크의 어떤 것도 건드리지 않습니다. `--remove`를 사용하면 이미 존재하는 번들 스킬 중 **수정되지 않은** 것도 삭제합니다(사용자가 수정했거나 허브에서 설치했거나 직접 작성한 스킬은 절대 제거하지 않으며, 미리보기와 확인을 먼저 표시하고 `--yes`로 건너뛸 수 있습니다). |
| `opt-in` | `.no-bundled-skills` 마커를 제거하여 `opt-out`을 취소하고 다음 `hermes update`에서 번들 스킬이 다시 시드되도록 합니다. `--sync`를 사용하면 즉시 다시 시드합니다. |
| `publish` | 레지스트리에 스킬을 게시합니다. |
| `snapshot` | 스킬 구성을 내보내거나 가져옵니다. |
| `tap` | 사용자 지정 스킬 소스를 관리합니다. |
| `config` | 플랫폼별 스킬 활성화/비활성화를 대화형으로 구성합니다. |

일반적인 예시:

```bash
hermes skills browse
hermes skills browse --source official
hermes skills search react --source skills-sh
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect official/security/1password
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install official/migration/openclaw-migration
hermes skills install skills-sh/anthropics/skills/pdf --force
hermes skills install https://sharethis.chat/SKILL.md                     # Direct URL (+ referenced support files)
hermes skills install https://example.com/SKILL.md --name my-skill        # Override name when frontmatter has none
hermes skills check
hermes skills update
hermes skills config
hermes skills reset google-workspace
hermes skills reset google-workspace --restore --yes
hermes skills opt-out                  # stop future bundled-skill seeding (nothing deleted)
hermes skills opt-out --remove --yes   # also delete UNMODIFIED bundled skills
hermes skills opt-in --sync            # undo: remove marker and re-seed now
```

참고:
- `--force`는 서드파티/커뮤니티 스킬에 대한 위험하지 않은 정책 차단을 재정의할 수 있습니다.
- `--force`는 `dangerous` 스캔 판정을 재정의하지 않습니다.
- `--source skills-sh`는 공개 `skills.sh` 디렉터리를 검색합니다.
- `--source well-known`을 사용하면 `/.well-known/skills/index.json`을 제공하는 사이트를 지정할 수 있습니다.
- `--source browse-sh`는 [browse.sh](https://browse.sh)의 200개가 넘는 사이트별 브라우저 자동화 스킬 카탈로그를 검색합니다. 식별자는 `browse-sh/airbnb.com/search-listings-ddgioa`와 같은 형식입니다.
- `http(s)://…/*.md` URL을 전달하면 `SKILL.md`와 `references/`, `templates/`, `scripts/`, `assets/`, `examples/` 아래에서 명시적으로 참조된 파일을 설치합니다. 프론트매터에 `name:`이 없고 URL 슬러그가 올바른 식별자가 아니면 대화형 터미널에서 이름을 입력하라는 메시지가 표시됩니다. 비대화형 인터페이스(TUI 내부의 `/skills install`, 게이트웨이 플랫폼)에서는 대신 `--name <x>`가 필요합니다.

## `hermes bundles`

```bash
hermes bundles <subcommand>
```

스킬 번들은 여러 스킬을 하나의 `/<bundle-name>` 슬래시 명령으로 묶습니다. 번들을 호출하면 참조된 모든 스킬이 하나의 결합된 사용자 메시지로 로드됩니다. 저장 위치: `~/.hermes/skill-bundles/<slug>.yaml`. YAML 스키마와 동작은 [스킬 번들](../user-guide/features/skills.md#skill-bundles)을 참조하세요.

하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `list` | 설치된 번들을 나열합니다(하위 명령을 지정하지 않을 때의 기본값). |
| `show <name>` | 하나의 번들 이름, 설명, 스킬 및 파일 경로를 표시합니다. |
| `create <name>` | 새 번들을 생성합니다. `--skill <id>`를 반복해서 전달하거나 생략하여 대화형으로 입력할 수 있습니다. `--description`, `--instruction`, `--force`를 사용할 수 있습니다. |
| `delete <name>` | 번들 파일을 제거합니다. |
| `reload` | `~/.hermes/skill-bundles/`를 다시 검색하고 추가/제거된 번들을 보고합니다. |

예시:

```bash
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work"

hermes bundles list
hermes bundles show backend-dev
hermes bundles delete backend-dev
```

채팅 세션에서 `/bundles`는 설치된 번들을 나열하고 `/<bundle-name>`은 번들 하나를 로드합니다.
## `hermes curator`

curator는 에이전트가 만든 스킬을 주기적으로 검토하고, 오래된 스킬을 정리하며, 중복되는 스킬을 통합하고, 더 이상 사용되지 않는 스킬을 보관하는 보조 모델 백그라운드 작업입니다. 번들로 제공되거나 허브에서 설치된 스킬은 절대 건드리지 않습니다. 보관된 항목은 복구할 수 있으며 자동 삭제는 수행되지 않습니다.

```bash
hermes curator <subcommand>
```

| 하위 명령 | 설명 |
|------------|-------------|
| `status` | curator 상태와 스킬 통계를 표시합니다. |
| `run` | 지금 curator 검토를 트리거합니다(LLM 단계가 완료될 때까지 차단됨). |
| `run --background` | LLM 단계를 백그라운드 스레드에서 시작하고 즉시 반환합니다. |
| `run --dry-run` | 미리 보기만 수행합니다 — 변경 없이 검토 보고서를 생성합니다. |
| `backup` | `~/.hermes/skills/`의 수동 tar.gz 스냅샷을 생성합니다(curator는 실제 실행 전에 자동으로도 스냅샷을 생성함). |
| `rollback` | 스냅샷에서 `~/.hermes/skills/`를 복원합니다(기본값은 최신 스냅샷). |
| `rollback --list` | 사용 가능한 스냅샷을 나열합니다. |
| `rollback --id <ts>` | 지정한 ID의 스냅샷을 복원합니다. |
| `rollback -y` | 확인 프롬프트를 건너뜁니다. |
| `pause` | 재개할 때까지 curator를 일시 중지합니다. |
| `resume` | 일시 중지된 curator를 재개합니다. |
| `pin <skill>` | 스킬을 고정하여 curator가 자동으로 상태를 변경하지 못하게 합니다. |
| `unpin <skill>` | 스킬 고정을 해제합니다. |
| `restore <skill>` | 보관된 스킬을 복원합니다. |
| `archive <skill>` | 스킬을 수동으로 보관합니다. |
| `prune` | curator가 일반적으로 정리할 스킬을 수동으로 정리합니다. |
| `list-archived` | 보관된 스킬을 나열합니다(`restore`로 복구 가능). |

자세한 동작과 설정은 [Curator](../user-guide/features/curator.md)를 참조하세요.

## `hermes moa`

이름이 지정된 Mixture of Agents 프리셋을 설정합니다. 프리셋은 모든 모델 선택기에서 `Mixture of Agents` 제공자의 선택 가능한 모델로 표시되며, `/moa <prompt>`는 기본 프리셋을 사용해 하나의 프롬프트를 실행합니다.

```bash
hermes moa list
hermes moa configure [name]
hermes moa delete <name>
```

`hermes moa configure`는 각 참조 모델과 집계기에 Hermes의 제공자 → 모델 선택기를 재사용합니다. 프리셋은 기본 모델이나 제공자가 아니라 실행 모드 설정입니다.

## `hermes fallback`

```bash
hermes fallback <subcommand>
```

폴백 제공자 체인을 관리합니다. 기본 모델이 속도 제한, 과부하 또는 연결 오류로 실패하면 폴백 제공자를 순서대로 시도합니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `list` (별칭: `ls`) | 현재 폴백 체인을 표시합니다(하위 명령이 없을 때 기본값). |
| `add` | `hermes model`과 동일한 선택기에서 제공자 + 모델을 선택하여 체인에 추가합니다. |
| `remove` (별칭: `rm`) | 삭제할 항목을 선택하여 체인에서 제거합니다. |
| `clear` | 모든 폴백 항목을 제거합니다. |

[폴백 제공자](../user-guide/features/fallback-providers.md)를 참조하세요.

## `hermes hooks`

```bash
hermes hooks <subcommand>
```

`~/.hermes/config.yaml`에 선언된 셸 스크립트 훅을 검사하고, 합성 페이로드를 대상으로 테스트하며, `~/.hermes/shell-hooks-allowlist.json`의 최초 사용 동의 허용 목록을 관리합니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `list` (별칭: `ls`) | 매처, 타임아웃, 동의 상태와 함께 설정된 훅을 나열합니다. |
| `test <event>` | `<event>`와 일치하는 모든 훅을 합성 페이로드로 실행합니다. |
| `revoke` (별칭: `remove`, `rm`) | 명령의 허용 목록 항목을 제거합니다(다음 재시작부터 적용). |
| `doctor` | 설정된 각 훅의 실행 비트, 허용 목록, mtime 변경, JSON 유효성, 합성 실행 시간을 확인합니다. |

이벤트 시그니처와 페이로드 형태는 [훅](../user-guide/features/hooks.md)을 참조하세요.

## `hermes memory`

```bash
hermes memory <subcommand>
```

외부 메모리 제공자 플러그인을 설정하고 관리합니다. 사용 가능한 제공자는 honcho, openviking, mem0, hindsight, holographic, retaindb, byterover, supermemory입니다. 외부 제공자는 한 번에 하나만 활성화할 수 있습니다. 내장 메모리(MEMORY.md/USER.md)는 항상 활성화됩니다.

하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `setup` | 대화형 제공자 선택 및 설정입니다. |
| `status` | 현재 메모리 제공자 설정을 표시합니다. |
| `off` | 외부 제공자를 비활성화합니다(내장 기능만 사용). |

:::info 제공자별 하위 명령
외부 메모리 제공자가 활성화되면 제공자별 관리를 위한 자체 최상위 `hermes <provider>` 명령을 등록할 수 있습니다(예: Honcho가 활성화된 경우 `hermes honcho`). 비활성 제공자는 하위 명령을 노출하지 않습니다. 현재 연결된 항목을 확인하려면 `hermes --help`를 실행하세요.
:::

## `hermes acp`

```bash
hermes acp
```

편집기 통합을 위한 ACP(Agent Client Protocol) stdio 서버로 Hermes를 시작합니다.

관련 진입점:

```bash
hermes-acp
python -m acp_adapter
```

먼저 지원 기능을 설치하세요.

```bash
cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'
```

[ACP 편집기 통합](../user-guide/features/acp.md) 및 [ACP 내부 구조](../developer-guide/acp-internals.md)를 참조하세요.

## `hermes mcp`

```bash
hermes mcp <subcommand>
```

MCP(Model Context Protocol) 서버 설정을 관리하고 Hermes를 MCP 서버로 실행합니다.

| 하위 명령 | 설명 |
|------------|-------------|
| *(없음)* 또는 `picker` | 대화형 카탈로그 선택기 — Nous 승인 MCP를 찾아 설치/활성화/비활성화합니다. |
| `catalog` | Nous 승인 MCP를 나열합니다(일반 텍스트, 스크립트에서 사용 가능). |
| `install <name>` | 카탈로그 항목을 설치합니다(예: `hermes mcp install n8n`). |
| `serve [-v\|--verbose]` | Hermes를 MCP 서버로 실행하여 다른 에이전트에 대화를 노출합니다. |
| `add <name> [--url URL] [--command CMD] [--auth oauth\|header] [--args ...]` | 자동 도구 검색으로 사용자 지정 MCP 서버를 추가합니다. `--args`는 나머지 argv를 stdio 명령에 전달하므로 마지막에 배치하세요. |
| `remove <name>` (별칭: `rm`) | 설정에서 MCP 서버를 제거합니다. |
| `list` (별칭: `ls`) | 설정된 MCP 서버를 나열합니다. |
| `test <name>` | MCP 서버 연결을 테스트합니다. |
| `configure <name>` (별칭: `config`) | 서버의 도구 선택을 전환합니다. |
| `login <name>` | OAuth 기반 MCP 서버의 재인증을 강제로 수행합니다. |

[MCP 설정 참조](./mcp-config-reference.md), [Hermes에서 MCP 사용](../guides/use-mcp-with-hermes.md), [MCP 서버 모드](../user-guide/features/mcp.md#running-hermes-as-an-mcp-server)를 참조하세요.

## `hermes plugins`

```bash
hermes plugins [subcommand]
```

일반 플러그인, 메모리 제공자, 컨텍스트 엔진을 한곳에서 관리합니다. 하위 명령 없이 `hermes plugins`를 실행하면 두 섹션으로 구성된 대화형 화면이 열립니다.

- **일반 플러그인** — 설치된 플러그인을 활성화/비활성화하는 다중 선택 체크박스
- **제공자 플러그인** — 메모리 제공자와 컨텍스트 엔진을 설정하는 단일 선택. 카테고리에서 ENTER를 누르면 라디오 선택기가 열립니다.

| 하위 명령 | 설명 |
|------------|-------------|
| *(없음)* | 일반 플러그인 전환 및 제공자 플러그인 설정을 위한 통합 대화형 UI입니다. |
| `install <identifier> [--force] [--ref COMMIT_SHA]` | Git URL, `owner/repo` 또는 이름만 있는 인덱스 이름으로 플러그인을 설치합니다. 슬래시가 없는 이름은 커뮤니티 플러그인 인덱스를 통해 `owner/repo`와 인덱스에 고정된 커밋으로 확인됩니다. 모호한 이름은 후보를 나열하고 종료합니다. `--ref`는 완전한 40자 커밋 SHA만 허용하며, 해당 변경 불가능한 리비전을 정확히 설치하고 모든 인덱스 고정을 덮어씁니다. |
| `search [term] [--json] [--capability CAP] [--refresh]` | 커뮤니티 플러그인 인덱스를 검색합니다(이름/설명/태그의 퍼지 매칭; `term`을 생략하면 탐색). `plugins.index_url`에서 가져오며(기본값: NousResearch 플러그인 인덱스), `~/.hermes/cache/`에 24시간 동안 캐시하고, 오프라인이면 오래된 캐시와 번들 시드 순으로 대체합니다. 인덱싱됨 ≠ 감사됨 — 포함 여부는 메타데이터 검토만을 의미합니다. |
| `update <name>` | 고정되지 않은 설치 플러그인의 최신 변경 사항을 가져옵니다. 고정된 플러그인은 이동하려면 `--force --ref <new-commit>`으로 다시 설치해야 합니다. |
| `remove <name>` (별칭: `rm`, `uninstall`) | 설치된 플러그인을 제거합니다. |
| `enable <name>` | 비활성화된 플러그인을 활성화합니다. |
| `disable <name>` | 제거하지 않고 플러그인을 비활성화합니다. |
| `list` (별칭: `ls`) | 활성화/비활성화 상태와 함께 설치된 플러그인을 나열합니다. |
| `doctor [path-or-id] [--ci]` | 실제 매니페스트 파서, 로더, 등록 경로를 통해 네이티브 플러그인을 검증합니다. `--ci`는 오류 발생 시 1로 종료합니다. |
| `pack install <path-or-url> [--force]` | 플러그인 팩(`hermes-pack.yaml`)을 설치합니다 — 정확한 40자 커밋 SHA로 고정된 플러그인 선언 집합입니다. 필수 검토 화면(각 플러그인, 소스, 고정된 ref, 선언된 기능)을 표시하고 팩 내용에 대한 확인을 한 번 요청한 뒤 일반적인 고정 설치를 실행합니다. 각 플러그인의 선언된 기능은 여전히 표준 플러그인별 동의를 거칩니다 — 팩이 일괄 승인을 부여하지는 않습니다. 부분 실패는 플러그인별로 보고되며, 하나라도 실패하면 0이 아닌 상태로 종료합니다. 대화형 전용입니다(`--yes` 없음). |
| `pack export [--enabled-only] [--name NAME]` | 현재 설치에서 팩 YAML을 stdout으로 출력합니다: Git으로 설치된 각 플러그인의 저장소 + 정확한 SHA와 비밀이 아닌 정제된 `plugins.entries` 설정입니다. 로컬 전용 플러그인(Git 출처 없음)은 경고 주석으로만 나열되며 설치 가능한 항목으로 표시되지 않습니다. 비밀, 기능 승인, `allow_*` 게이트는 항상 제거됩니다. |
| `pack show <path-or-url>` | 드라이 런: 설치하지 않고 팩을 파싱하고 검증하여 표시합니다. |

제공자 플러그인 선택은 `config.yaml`에 저장됩니다.
- `memory.provider` — 활성 메모리 제공자(비어 있으면 내장 기능만 사용)
- `context.engine` — 활성 컨텍스트 엔진(`"compressor"` = 내장 기본값)

일반 플러그인 비활성화 목록은 `config.yaml`의 `plugins.disabled` 아래에 저장됩니다.
Git 설치는 프로필별 `plugins/.install-metadata.json` 사이드카에 정식 소스, 정확한 설치 리비전, 고정 상태만 기록합니다. 여기에는 플러그인 설정, 환경 값, 비밀 또는 기능 승인이 포함되지 않습니다.

[플러그인](../user-guide/features/plugins.md) 및 [Hermes 플러그인 빌드](../developer-guide/plugins/index.md)를 참조하세요.

## `hermes tools`

```bash
hermes tools [--summary]
```

| 옵션 | 설명 |
|--------|-------------|
| `--summary` | 현재 활성화된 도구 요약을 출력하고 종료합니다. |

`--summary` 없이 실행하면 플랫폼별 대화형 도구 설정 UI가 시작됩니다.

## `hermes computer-use`

```bash
hermes computer-use <subcommand>
```

하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `install` | 업스트림 cua-driver 설치 프로그램을 실행합니다(macOS, Windows, Linux). |
| `install --upgrade` | cua-driver가 이미 PATH에 있어도 설치 프로그램을 다시 실행합니다. 업스트림 스크립트는 항상 최신 릴리스를 가져오므로 현재 위치에서 업그레이드가 수행됩니다. |
| `status` | `cua-driver`가 `$PATH`에 있는지와 설치된 버전을 출력합니다. |

`hermes computer-use install`은 `computer_use` 도구 세트에서 사용하는 [cua-driver](https://github.com/trycua/cua) 바이너리를 설치하는 안정적인 진입점입니다. 처음 Computer Use를 활성화할 때 `hermes tools`가 호출하는 것과 동일한 업스트림 설치 프로그램을 실행하므로, 도구 세트 전환이 설치를 시작하지 않은 경우(예: 기존 사용자 설정)에도 안전하게 다시 실행할 수 있습니다.

`hermes update`는 cua-driver가 PATH에 있으면 업데이트 마지막에 업스트림 설치 프로그램을 자동으로 다시 실행하므로, 대부분의 사용자는 수동으로 `--upgrade`를 호출할 필요가 없습니다. 다음 Hermes 업데이트를 기다리지 않고 업스트림에서 제공한 수정 사항을 바로 적용하려면 사용하세요.

## `hermes pets`

```bash
hermes pets <list|install|select|show|off|scale|remove|doctor>
```

[Petdex](https://github.com/crafter-station/petdex)는 코딩 에이전트를 위한 애니메이션 스프라이트 펫 공개 갤러리입니다. 하나를 설치하면 Hermes가 CLI, TUI, 데스크톱 앱 전반에서 에이전트 활동에 반응하는 모습을 보여줍니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `list` | petdex 갤러리를 탐색합니다. |
| `install` | 갤러리에서 펫을 설치합니다. |
| `select` | 활성 펫을 설정합니다(`display.pet.*`에 기록). |
| `show` | 터미널에서 활성 펫을 애니메이션으로 표시합니다. |
| `off` | 펫 표시를 비활성화합니다. |
| `scale` | 모든 곳에서 펫 크기를 조정합니다(`display.pet.scale`). |
| `remove` | 설치된 펫을 삭제합니다. |
| `doctor` | 펫 설정과 터미널 그래픽 지원을 확인합니다. |

`/hatch` 슬래시 명령으로 텍스트 설명을 바탕으로 완전히 새로운 펫을 생성할 수도 있습니다. [펫](../user-guide/features/pets.md)을 참조하세요.
## `hermes sessions`

```bash
hermes sessions <subcommand>
```

하위 명령:

| 하위 명령 | 설명 |
|------------|-------------|
| `list` | 최근 세션을 나열합니다. |
| `browse` | 검색 및 재개 기능이 있는 대화형 세션 선택기입니다. |
| `export <output> [--session-id ID]` | 세션을 JSONL로 내보냅니다. |
| `delete <session-id>` | 필터와 일치하는 세션을 삭제합니다: 시간 범위 `--older-than`/`--newer-than`/`--before`/`--after`(예: `5h`/`2d`, 일수만 입력하거나 ISO 타임스탬프); 속성 `--source`, `--title`, `--model`, `--provider`, `--branch`, `--end-reason`, `--user`, `--chat-id`, `--chat-type`, `--cwd`; 숫자 범위 `--min/--max-messages`, `--min/--max-tokens`, `--min/--max-cost`, `--min/--max-tool-calls`; 그리고 `--include-archived`, `--dry-run`, `--yes`. 기본값은 90일보다 오래된 세션입니다. |
| `archive` | 동일한 필터로 일치하는 세션을 일괄 보관 처리(소프트 숨김)합니다. 필터를 하나 이상 지정해야 합니다. |
| `stats` | 세션 저장소 통계를 표시합니다. |
| `rename <session-id> <title>` | 세션 제목을 설정하거나 변경합니다. |
| `optimize` | 디스크 공간을 회수합니다: FTS5 인덱스 세그먼트를 병합하고 VACUUM을 수행합니다. 세션 데이터는 변경되지 않는 비파괴 작업입니다. |
| `optimize-storage` | 전체 텍스트 검색 인덱스를 압축된 v23 외부 콘텐츠 레이아웃으로 마이그레이션합니다. 대규모 데이터베이스에서는 `state.db`의 상당 부분을 회수할 수 있습니다. |
| `repair` | 잘못된 `state.db` 스키마(예: `table messages_fts already exists`)를 복구하여 숨겨진 세션이 다시 표시되도록 합니다. 먼저 백업이 생성됩니다. |
| `repair-routing` | 라우팅 ID를 잃은 세션 행에 고립된 게이트웨이 대화를 다시 연결합니다(재시작 후 채팅이 과거 시점으로 되돌아가는 현상). 기본값은 미리 보기이며, `--apply`를 사용하면 연결을 적용합니다(먼저 게이트웨이를 중지하세요). `--max-gap-seconds N`으로 연속성 시간 범위를 조정할 수 있습니다. 모호하지 않은 사례만 복구합니다. [세션 → 고립된 게이트웨이 세션 복구](../user-guide/sessions.md#repair-stranded-gateway-sessions)를 참고하세요. |
| `recover` | 손상된 `state.db`를 별도의 깨끗한 데이터베이스로 오프라인에서 비파괴 복구합니다. |
| `retitle-skills` | 사용자가 실제로 입력한 내용을 바탕으로 `/skill`로 연 세션의 제목을 다시 생성합니다. `--apply`를 전달하지 않으면 변경 사항을 나열합니다. |

## `hermes insights`

```bash
hermes insights [--days N] [--source platform]
```

| 옵션 | 설명 |
|--------|-------------|
| `--days <n>` | 최근 `n`일을 분석합니다(기본값: 30). |
| `--source <platform>` | `cli`, `telegram`, `discord`와 같은 소스로 필터링합니다. |

## `hermes claw`

```bash
hermes claw migrate [options]
```

OpenClaw 설정을 Hermes로 마이그레이션합니다. `~/.openclaw`(또는 사용자 지정 경로)에서 읽고 `~/.hermes`에 씁니다. 레거시 디렉터리 이름(`~/.clawdbot`, `~/.moltbot`)과 구성 파일 이름(`clawdbot.json`, `moltbot.json`)을 자동으로 감지합니다.

| 옵션 | 설명 |
|--------|-------------|
| `--dry-run` | 아무것도 쓰지 않고 마이그레이션될 내용을 미리 봅니다. |
| `--preset <name>` | 마이그레이션 프리셋: `full`(호환되는 모든 설정) 또는 `user-data`(인프라 구성 제외)입니다. 어느 프리셋도 시크릿을 가져오지 않으며, 명시적으로 `--migrate-secrets`를 전달해야 합니다. |
| `--overwrite` | 충돌 시 기존 Hermes 파일을 덮어씁니다(기본값: 계획에 충돌이 있으면 적용하지 않음). |
| `--migrate-secrets` | API 키를 마이그레이션에 포함합니다. `--preset full`에서도 필요합니다. |
| `--no-backup` | 마이그레이션 전 `~/.hermes/`의 zip 스냅샷을 건너뜁니다(기본적으로 적용 전에 단일 복원 지점 아카이브가 `~/.hermes/backups/pre-migration-*.zip`에 기록되며, `hermes import`로 복원할 수 있습니다). |
| `--source <path>` | 사용자 지정 OpenClaw 디렉터리(기본값: `~/.openclaw`)입니다. |
| `--workspace-target <path>` | 워크스페이스 지침(AGENTS.md)의 대상 디렉터리입니다. |
| `--skill-conflict <mode>` | 스킬 이름 충돌 처리 방식: `skip`(기본값), `overwrite` 또는 `rename`입니다. |
| `--yes` | 확인 프롬프트를 건너뜁니다. |

### 마이그레이션되는 항목

마이그레이션은 페르소나, 메모리, 스킬, 모델 제공자, 메시징 플랫폼, 에이전트 동작, 세션 정책, MCP 서버, TTS 등을 포함한 30개 이상의 범주를 다룹니다. 항목은 Hermes에 대응하는 항목으로 **직접 가져오거나**, 수동 검토를 위해 **보관**됩니다.

**직접 가져오기:** SOUL.md, MEMORY.md, USER.md, AGENTS.md, 스킬(소스 디렉터리 4개), 기본 모델, 사용자 지정 제공자, MCP 서버, 메시징 플랫폼 토큰 및 허용 목록(Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost), 에이전트 기본값(추론 강도, 압축, 사람 지연, 시간대, 샌드박스), 세션 재설정 정책, 승인 규칙, TTS 구성, 브라우저 설정, 도구 설정, 실행 시간 제한, 명령 허용 목록, 게이트웨이 구성, 3개 소스의 API 키입니다.

**수동 검토를 위해 보관:** Cron 작업, 플러그인, 훅/웹훅, 메모리 백엔드(QMD), 스킬 레지스트리 구성, UI/ID, 로깅, 다중 에이전트 설정, 채널 바인딩, IDENTITY.md, TOOLS.md, HEARTBEAT.md, BOOTSTRAP.md입니다.

**API 키 확인**은 우선순위에 따라 세 가지 소스를 검사합니다: 구성 값 → `~/.openclaw/.env` → `auth-profiles.json`. 모든 토큰 필드는 일반 문자열, 환경 변수 템플릿(`${VAR}`), SecretRef 객체를 처리합니다.

전체 구성 키 매핑, SecretRef 처리 세부 정보 및 마이그레이션 후 체크리스트는 **[전체 마이그레이션 가이드](../guides/migrate-from-openclaw.md)**를 참조하세요.

### 예시

```bash
# Preview what would be migrated
hermes claw migrate --dry-run

# Full migration (all compatible settings, no secrets)
hermes claw migrate --preset full

# Full migration including API keys
hermes claw migrate --preset full --migrate-secrets

# Migrate user data only (no secrets), overwrite conflicts
hermes claw migrate --preset user-data --overwrite

# Migrate from a custom OpenClaw path
hermes claw migrate --source /home/user/old-openclaw
```

## `hermes import-agent`

```bash
hermes import-agent [claude-code|codex] [options]
```

**Claude Code**(`~/.claude`) 또는 **OpenAI Codex CLI**(`~/.codex`) 설정을 Hermes로 가져옵니다. `CLAUDE.md`/`AGENTS.md` 지침을 메모리 항목으로, `Bash(...)` 권한 허용/거부 규칙을 `command_allowlist`/`approvals.deny`로, MCP 서버를 `config.yaml`의 `mcp_servers`로 매핑하고, 스킬 디렉터리를 `~/.hermes/skills/`로 가져옵니다. 적용 전에 항상 미리 보며, API 키와 자격 증명은 절대 가져오지 않습니다.

| 옵션 | 설명 |
| --- | --- |
| `agent` | `claude-code` 또는 `codex`입니다(기본값: 자동 감지). |
| `--source <path>` | 사용자 지정 소스 디렉터리(기본값: `~/.claude` 또는 `~/.codex`)입니다. |
| `--dry-run` | 미리 보기만 수행하며 아무것도 쓰지 않습니다. |
| `--overwrite` | 충돌하는 MCP 서버/스킬을 교체합니다(기본값: 건너뜀). |
| `--yes`, `-y` | 확인 프롬프트를 건너뜁니다. |

전체 매핑 표는 **[가져오기 가이드](../user-guide/import-from-other-agents.md)**를 참조하세요.

## `hermes serve`

```bash
hermes serve [options]
```

Hermes **백엔드 서버**를 시작합니다. [데스크톱 앱](/user-guide/desktop)과 원격 클라이언트가 연결하는 JSON-RPC/WebSocket 게이트웨이입니다. `hermes dashboard`가 실행하는 것과 동일한 서버이지만 **헤드리스**이므로 브라우저 UI를 절대 열지 않습니다. 데스크톱 앱은 자체 `hermes serve` 백엔드를 시작합니다. 원격 호스트에서 헤드리스 백엔드를 사용하려면 이 명령을 직접 실행하세요. 아래의 `hermes dashboard`와 동일한 `--host` / `--port` / `--insecure` / `--skip-build` / `--stop` / `--status` 옵션을 허용합니다(루프백이 아닌 바인드 주소를 사용하면 동일한 인증 게이트가 활성화됩니다). `[web]` 추가 기능이 필요하며, 내장 Chat 소켓에는 POSIX 호스트에서 `[pty]`도 필요합니다.

## `hermes dashboard`

```bash
hermes dashboard [options]
```

구성, API 키를 관리하고 세션을 모니터링하기 위한 브라우저 기반 UI인 웹 대시보드를 실행합니다. (브라우저 UI가 없는 헤드리스 백엔드, 예를 들어 데스크톱 앱이 시작하는 백엔드가 필요하면 위의 [`hermes serve`](#hermes-serve)를 사용하세요.) `cd ~/.hermes/hermes-agent && uv pip install -e ".[web]"`(FastAPI + Uvicorn)가 필요합니다. 내장 브라우저 Chat 탭은 항상 사용할 수 있으며, 추가로 `pty` 추가 기능(`cd ~/.hermes/hermes-agent && uv pip install -e ".[web,pty]"`)과 Linux, macOS 또는 WSL2와 같은 POSIX PTY 환경이 필요합니다. 전체 문서는 [웹 대시보드](/user-guide/features/web-dashboard)를 참조하세요.

| 옵션 | 기본값 | 설명 |
|--------|---------|-------------|
| `--port` | `9119` | 웹 서버를 실행할 포트입니다. |
| `--host` | `127.0.0.1` | 바인드 주소입니다. |
| `--no-open` | — | 브라우저를 자동으로 열지 않습니다. |
| `--insecure` | 꺼짐 | **지원 중단됨 / 아무 작업도 하지 않음.** 이전에는 루프백이 아닌 바인드에서 인증을 우회했습니다. 2026년 6월 보안 강화 이후 공개 바인드는 항상 인증 제공자(비밀번호 또는 OAuth)를 요구합니다. `127.0.0.1`에 바인드하고 터널링하여 로컬로 유지하세요. |
| `--skip-build` | 꺼짐 | 웹 UI 빌드 단계를 건너뛰고 기존 `dist`를 직접 제공합니다. npm을 사용할 수 없는 비대화형 컨텍스트(Windows 예약 작업, CI)에서 유용합니다. `cd web && npm run build`로 미리 빌드하세요. |
| `--isolated` | 꺼짐 | 이름이 지정된 프로필(`worker dashboard`)에서 시작할 때 머신 대시보드로 라우팅하는 대신 프로필별 전용 서버를 실행합니다. |
| `--stop` | — | 실행 중인 `hermes dashboard` 프로세스를 중지하고 종료합니다. |
| `--status` | — | 실행 중인 `hermes dashboard` 프로세스를 나열하고 종료합니다. |

### `hermes dashboard register`

이 설치를 Nous Portal 계정의 셀프 호스팅 대시보드로 등록합니다. OAuth 클라이언트를 생성하고 `HERMES_DASHBOARD_OAUTH_CLIENT_ID`를 `~/.hermes/.env`에 기록하며 로그인 게이트를 활성화하는 방법을 출력합니다. 로그인 상태여야 합니다(`hermes setup`).

| 옵션 | 설명 |
|--------|-------------|
| `--name` | 사람이 읽을 수 있는 대시보드 레이블입니다(기본값: 자동 생성). |
| `--redirect-uri` | 공개 HTTPS OAuth 리디렉션 URI입니다(예: `https://hermes.example.com/auth/callback`). localhost 전용으로 사용할 때는 생략하세요. |
| `--portal-url` | 등록에 사용할 Nous Portal 기본 URL을 재정의합니다(기본값: 로그인한 포털). `HERMES_DASHBOARD_PORTAL_URL`로도 설정할 수 있습니다. |

```bash
# Default — opens browser to http://127.0.0.1:9119
hermes dashboard

# Custom port, no browser
hermes dashboard --port 8080 --no-open

# From a profile alias — routes to the machine dashboard with the
# profile preselected in the sidebar switcher (attach if running)
worker dashboard
```

## `hermes profile`

```bash
hermes profile <subcommand>
```

프로필을 관리합니다. 각 프로필은 자체 구성, 세션, 스킬 및 홈 디렉터리를 가진 여러 개의 격리된 Hermes 인스턴스입니다.

| 하위 명령 | 설명 |
|------------|-------------|
| `list` | 모든 프로필을 나열합니다. |
| `use <name>` | 고정 기본 프로필을 설정합니다. |
| `create <name> [--clone] [--clone-all] [--clone-from <source>] [--no-alias]` | 새 프로필을 생성합니다. `--clone`은 활성 프로필의 구성, `.env`, `SOUL.md` 및 스킬을 복사합니다. `--clone-all`은 모든 상태를 복사합니다. `--clone-from`은 소스 프로필을 지정하며 `--clone-all`과 함께 사용하지 않는 한 구성 복제를 의미합니다. |
| `delete <name> [-y]` | 프로필을 삭제합니다. |
| `show <name>` | 프로필 세부 정보(홈 디렉터리, 구성 등)를 표시합니다. |
| `alias <name> [--remove] [--name NAME]` | 빠른 프로필 접근을 위한 래퍼 스크립트를 관리합니다. |
| `rename <old> <new>` | 프로필 이름을 변경합니다. |
| `export <name> [-o FILE]` | 프로필을 `.tar.gz` 아카이브로 내보냅니다(로컬 백업). |
| `import <archive> [--name NAME]` | `.tar.gz` 아카이브에서 프로필을 가져옵니다(로컬 복원). |
| `install <source> [--name N] [--alias] [--force] [-y]` | git URL 또는 로컬 디렉터리에서 프로필 배포판을 설치합니다. |
| `update <name> [--force-config] [-y]` | 배포판을 다시 가져옵니다. 사용자 데이터(메모리, 세션, 인증)는 보존됩니다. |
| `info <name>` | 프로필 배포 매니페스트(버전, 요구 사항, 소스)를 표시합니다. |

예시:

```bash
hermes profile list
hermes profile create work --clone
hermes profile use work
hermes profile alias work --name h-work
hermes profile export work -o work-backup.tar.gz
hermes profile import work-backup.tar.gz --name restored
hermes profile install github.com/user/my-distro --alias
hermes profile update work
hermes -p work chat -q "Hello from work profile"
```

## `hermes completion`

```bash
hermes completion [bash|zsh|fish]
```

셸 완료 스크립트를 표준 출력으로 출력합니다. Hermes 명령, 하위 명령 및 프로필 이름의 탭 완성을 사용하려면 셸 프로필에서 출력 내용을 소스로 불러오세요.

예시:

```bash
# Bash
hermes completion bash >> ~/.bashrc

# Zsh
hermes completion zsh >> ~/.zshrc

# Fish
hermes completion fish > ~/.config/fish/completions/hermes.fish
```
## `hermes update`

```bash
hermes update [--gateway] [--check] [--no-backup] [--backup] [--yes]
```

최신 `hermes-agent` 코드를 가져오고 관리되는 venv에 종속성을 다시 설치한 다음, 설치 후 훅(MCP 서버, 스킬 동기화, 자동 완성 설치)을 다시 실행합니다. 실행 중인 설치에서도 안전하게 사용할 수 있습니다. 설치하지 않고 체크아웃이 `origin/main`보다 뒤처져 있는지 확인하려면 `--check`를 사용합니다.

`hermes update`는 구성된 업데이트 브랜치(기본값: `main`)를 가져옵니다. 체크아웃이 다른 브랜치에 있다면 Hermes가 가져오기 전에 업데이트 브랜치로 체크아웃할 수 있습니다. 업데이트 자동 스태시 흐름에 포함하지 않고 브랜치 작업을 보존하려면 업데이트 전에 커밋하세요.

| 옵션 | 설명 |
|--------|-------------|
| `--gateway` | 메시징 `/update` 명령에서 사용하는 내부 모드입니다. 터미널 stdin을 읽는 대신 파일 기반 IPC를 사용해 프롬프트와 진행 상황을 스트리밍합니다. 게이트웨이 재시작 플래그가 아닙니다. |
| `--check` | 가져오기, 종속성 설치 또는 재시작 없이 업데이트가 가능한지 확인합니다. |
| `--no-backup` | 이번 실행에서 업데이트 전 백업을 모두 건너뜁니다(`updates.pre_update_backup`의 빠른 상태 스냅샷과 전체 zip 모두 해당). |
| `--backup` | 이번 실행에서 업데이트 전 **전체** 백업을 강제합니다. 빠른 상태 스냅샷과 `HERMES_HOME`의 전체 zip(구성, 인증, 세션, 스킬, 페어링 데이터)을 모두 포함합니다. 기본 모드는 `quick`이며, 영구 모드는 `config.yaml`의 `updates.pre_update_backup: quick | full | off`로 설정합니다. |
| `--yes`, `-y` | 구성 마이그레이션 및 스태시 복원과 같은 대화형 프롬프트에 자동으로 yes를 선택합니다. API 키 입력은 건너뛰므로 해당 키는 별도로 `hermes config migrate`를 실행해 입력하세요. |

추가 동작:

- **게이트웨이 재시작.** 업데이트가 성공하면 Hermes는 실행 중인 모든 게이트웨이 프로필을 자동으로 재시작해 새 코드를 적용하려고 시도합니다. 업데이트를 적용하지 않고 게이트웨이만 재시작하려면 `hermes gateway restart`를 사용하세요.
- **로컬 소스 변경 사항.** Git 설치의 경우 브랜치 체크아웃 또는 pull 전에 추적 중인 변경 파일과 추적되지 않는 파일을 자동으로 스태시합니다(`git stash push --include-untracked`). 대화형 터미널 업데이트에서는 스태시 복원 여부를 묻습니다. 비대화형 업데이트에서는 기본적으로 복원합니다. 성공적인 pull 후 로컬 소스 편집 내용을 버려도 되는 관리형 설치에서는 `updates.non_interactive_local_changes: discard`를 설정하세요. 스태시 복원 과정에서 충돌이 발생하거나 pull이 실패하면 스태시는 수동 복구를 위해 남겨 둡니다.
- **npm lockfile 변경 발생.** 스태시하거나 브랜치를 전환하기 전에 Hermes는 npm 설치/빌드 단계에서 생성된 추적 중인 `package-lock.json` 차이를 최선을 다해 정리합니다. 의도적으로 수정한 lockfile은 업데이트를 실행하기 전에 커밋하거나 수동으로 스태시하세요.
- **페어링 데이터 스냅샷.** `--backup`이 꺼져 있어도 `hermes update`는 `git pull` 전에 `~/.hermes/pairing/`과 Feishu 댓글 규칙의 경량 스냅샷을 생성합니다. pull로 편집 중인 파일이 덮어써졌다면 `hermes backup restore --state pre-update`로 복원할 수 있습니다.
- **레거시 `hermes.service` 경고.** Hermes가 현재의 `hermes-gateway.service` 대신 이름 변경 전의 `hermes.service` systemd 유닛을 감지하면, 플랩 루프 문제를 피할 수 있도록 일회성 마이그레이션 안내를 출력합니다.
- **종료 코드.** 성공 시 `0`, pull/설치/설치 후 오류 시 `1`, `git pull`을 막는 예기치 않은 작업 트리 변경 시 `2`를 반환합니다.

## 유지 관리 명령

| 명령 | 설명 |
|---------|-------------|
| `hermes version` | 버전 정보를 출력합니다. |
| `hermes update` | 최신 변경 사항을 가져오고 종속성을 다시 설치합니다. |

| `hermes uninstall [--full] [--gui] [--dry-run] [--yes]` | Hermes를 제거하며, 선택적으로 모든 구성/데이터를 삭제합니다. `--gui`는 에이전트는 유지한 채 데스크톱 Chat GUI만 제거하고, `--full`은 구성/데이터도 삭제하며, `--dry-run`은 변경 없이 제거 대상만 출력하고, `--yes`는 프롬프트를 건너뜁니다. |

## 참고

- [슬래시 명령어 참조](./slash-commands.md)
- [CLI 인터페이스](../user-guide/cli.md)
- [세션](../user-guide/sessions.md)
- [스킬 시스템](../user-guide/features/skills.md)
- [스킨 및 테마](../user-guide/features/skins.md)
