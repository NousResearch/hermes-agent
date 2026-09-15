---
sidebar_position: 1
title: "아키텍처"
description: "Hermes Agent 내부 구조 — 주요 하위 시스템, 실행 경로, 데이터 흐름 및 다음에 읽을 내용"
---

# 아키텍처

이 페이지는 Hermes Agent 내부 구조를 한눈에 보여 주는 지도입니다. 먼저 코드베이스의 전체 방향을 파악한 다음, 구현 세부 사항은 하위 시스템별 문서에서 확인하세요.

## 시스템 개요

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Entry Points                                  │
│                                                                      │
│  CLI (cli.py)    Gateway (gateway/run.py)    ACP (acp_adapter/)     │
│  Batch Runner    API Server                  Python Library          │
└──────────┬──────────────┬───────────────────────┬───────────────────┘
           │              │                       │
           ▼              ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AIAgent (run_agent.py)                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Prompt       │  │ Provider     │  │ Tool         │               │
│  │ Builder      │  │ Resolution   │  │ Dispatch     │               │
│  │ (prompt_     │  │ (runtime_    │  │ (model_      │               │
│  │  builder.py) │  │  provider.py)│  │  tools.py)   │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                       │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐               │
│  │ Compression  │  │ 3 API Modes  │  │ Tool Registry│               │
│  │ & Caching    │  │ chat_compl.  │  │ (registry.py)│               │
│  │              │  │ codex_resp.  │  │ 70+ tools    │               │
│  │              │  │ anthropic    │  │ 28 toolsets  │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────┴─────────────────┴─────────────────┴───────────────────────┘
           │                                    │
           ▼                                    ▼
┌───────────────────┐              ┌──────────────────────┐
│ Session Storage   │              │ Tool Backends         │
│ (SQLite + FTS5)   │              │ Terminal (6 backends) │
│ hermes_state.py   │              │ Browser (5 backends)  │
│ gateway/session.py│              │ Web (4 backends)      │
└───────────────────┘              │ MCP (dynamic)         │
                                   │ File, Vision, etc.    │
                                   └──────────────────────┘
```

## 디렉터리 구조

```text
hermes-agent/
├── run_agent.py              # AIAgent — core conversation loop (large file)
├── cli.py                    # HermesCLI — interactive terminal UI (large file)
├── model_tools.py            # Tool discovery, schema collection, dispatch
├── toolsets.py               # Tool groupings and platform presets
├── hermes_state.py           # SQLite session/state database with FTS5
├── hermes_constants.py       # HERMES_HOME, profile-aware paths
├── batch_runner.py           # Batch trajectory generation
│
├── agent/                    # Agent internals
│   ├── prompt_builder.py     # System prompt assembly
│   ├── context_engine.py     # ContextEngine ABC (pluggable)
│   ├── context_compressor.py # Default engine — lossy summarization
│   ├── prompt_caching.py     # Anthropic prompt caching
│   ├── auxiliary_client.py   # Auxiliary LLM for side tasks (vision, summarization)
│   ├── model_metadata.py     # Model context lengths, token estimation
│   ├── models_dev.py         # models.dev registry integration
│   ├── anthropic_adapter.py  # Anthropic Messages API format conversion
│   ├── display.py            # KawaiiSpinner, tool preview formatting
│   ├── skill_commands.py     # Skill slash commands
│   ├── memory_manager.py    # Memory manager orchestration
│   ├── memory_provider.py   # Memory provider ABC
│   └── trajectory.py         # Trajectory saving helpers
│
├── hermes_cli/               # CLI subcommands and setup
│   ├── main.py               # Entry point — all `hermes` subcommands (large file)
│   ├── config.py             # DEFAULT_CONFIG, OPTIONAL_ENV_VARS, migration
│   ├── commands.py           # COMMAND_REGISTRY — central slash command definitions
│   ├── auth.py               # PROVIDER_REGISTRY, credential resolution
│   ├── runtime_provider.py   # Provider → api_mode + credentials
│   ├── models.py             # Model catalog, provider model lists
│   ├── model_switch.py       # /model command logic (CLI + gateway shared)
│   ├── setup.py              # Interactive setup wizard (large file)
│   ├── skin_engine.py        # CLI theming engine
│   ├── skills_config.py      # hermes skills — enable/disable per platform
│   ├── skills_hub.py         # /skills slash command
│   ├── tools_config.py       # hermes tools — enable/disable per platform
│   ├── plugins.py            # PluginManager — discovery, loading, hooks
│   ├── callbacks.py          # Terminal callbacks (clarify, sudo, approval)
│   └── gateway.py            # hermes gateway start/stop
│
├── tools/                    # Tool implementations (one file per tool)
│   ├── registry.py           # Central tool registry
│   ├── approval.py           # Dangerous command detection
│   ├── terminal_tool.py      # Terminal orchestration
│   ├── process_registry.py   # Background process management
│   ├── file_tools.py         # read_file, write_file, patch, search_files
│   ├── web_tools.py          # web_search, web_extract
│   ├── browser_tool.py       # 10 browser automation tools
│   ├── code_execution_tool.py # execute_code sandbox
│   ├── delegate_tool.py      # Subagent delegation
│   ├── mcp_tool.py           # MCP client (large file)
│   ├── credential_files.py   # File-based credential passthrough
│   ├── env_passthrough.py    # Env var passthrough for sandboxes
│   ├── ansi_strip.py         # ANSI escape stripping
│   └── environments/         # Terminal backends (local, docker, ssh, modal, daytona, singularity)
│
├── gateway/                  # Messaging platform gateway
│   ├── run.py                # GatewayRunner — message dispatch (large file)
│   ├── session.py            # SessionStore — conversation persistence
│   ├── delivery.py           # Outbound message delivery
│   ├── pairing.py            # DM pairing authorization
│   ├── hooks.py              # Hook discovery and lifecycle events
│   ├── mirror.py             # Cross-session message mirroring
│   ├── status.py             # Token locks, profile-scoped process tracking
│   ├── builtin_hooks/        # Extension point for always-registered hooks (none shipped)
│   └── platforms/            # Built-in adapters: signal, weixin, bluebubbles,
│                             #   qqbot, whatsapp_cloud, yuanbao, webhook, api_server
│
├── plugins/platforms/        # Bundled platform plugins: telegram, discord, slack,
│                             #   whatsapp, matrix, mattermost, email, sms, dingtalk,
│                             #   feishu, wecom, homeassistant, irc, line, teams,
│                             #   google_chat, buzz, ntfy, photon, raft, simplex
│
├── acp_adapter/              # ACP server (VS Code / Zed / JetBrains)
├── cron/                     # Scheduler (jobs.py, scheduler.py)
├── plugins/memory/           # Memory provider plugins
├── plugins/context_engine/   # Context engine plugins
├── skills/                   # Bundled skills (always available)
├── optional-skills/          # Official optional skills (install explicitly)
├── website/                  # Docusaurus documentation site
└── tests/                    # Pytest suite (~25,000 tests across ~1,250 files)
```

## 데이터 흐름

### CLI 세션

```text
User input → HermesCLI.process_input()
  → AIAgent.run_conversation()
    → prompt_builder.build_system_prompt()
    → runtime_provider.resolve_runtime_provider()
    → API call (chat_completions / codex_responses / anthropic_messages)
    → tool_calls? → model_tools.handle_function_call() → loop
    → final response → display → save to SessionDB
```

### 게이트웨이 메시지

```text
Platform event → Adapter.on_message() → MessageEvent
  → GatewayRunner._handle_message()
    → authorize user
    → resolve session key
    → create AIAgent with session history
    → AIAgent.run_conversation()
    → deliver response back through adapter
```

### Cron 작업

```text
Scheduler tick → load due jobs from jobs.json
  → create fresh AIAgent (no history)
  → inject attached skills as context
  → run job prompt
  → deliver response to target platform
  → update job state and next_run
```

## 권장 읽기 순서

코드베이스를 처음 접한다면 다음 순서로 읽으세요.

1. **이 페이지** — 전체 방향 파악
2. **[에이전트 루프 내부 구조](./agent-loop.md)** — AIAgent의 작동 방식
3. **[프롬프트 조립](./prompt-assembly.md)** — 시스템 프롬프트 구성
4. **[프로바이더 런타임 해석](./provider-runtime.md)** — 프로바이더 선택 방식
5. **[프로바이더 추가](./adding-providers.md)** — 새 프로바이더 추가 실전 가이드
6. **[도구 런타임](./tools-runtime.md)** — 도구 레지스트리, 디스패치, 환경
7. **[세션 저장소](./session-storage.md)** — SQLite 스키마, FTS5, 세션 계보
8. **[게이트웨이 내부 구조](./gateway-internals.md)** — 메시징 플랫폼 게이트웨이
9. **[컨텍스트 압축 및 프롬프트 캐싱](./context-compression-and-caching.md)** — 압축 및 캐싱
10. **[ACP 내부 구조](./acp-internals.md)** — IDE 통합

## 주요 하위 시스템

### 에이전트 루프

동기식 오케스트레이션 엔진입니다(`run_agent.py`의 `AIAgent`). 프로바이더 선택, 프롬프트 구성, 도구 실행, 재시도, 대체 처리, 콜백, 압축 및 영속성을 담당합니다. 서로 다른 프로바이더 백엔드를 위해 세 가지 API 모드를 지원합니다.

→ [에이전트 루프 내부 구조](./agent-loop.md)

### 프롬프트 시스템

대화 생명주기 전반에 걸쳐 프롬프트를 구성하고 유지합니다.

- **`system_prompt.py` + `prompt_builder.py`** — 시스템 프롬프트 계층(`stable` → `context` → `volatile`)을 순서대로 조립합니다. 정체성/도구 안내/스킬, 컨텍스트 파일, 그리고 메모리/프로필/타임스탬프 블록 순서입니다.
- **`prompt_caching.py`** — 접두사 캐싱을 위해 Anthropic 캐시 중단점을 적용합니다.
- **`context_compressor.py`** — 컨텍스트가 임계값을 초과하면 대화 중간의 턴을 요약합니다.

→ [프롬프트 조립](./prompt-assembly.md), [컨텍스트 압축 및 프롬프트 캐싱](./context-compression-and-caching.md)

### 프로바이더 해석

CLI, 게이트웨이, cron, ACP 및 보조 호출에서 공유하는 런타임 해석기입니다. `(provider, model)` 튜플을 `(api_mode, api_key, base_url)`로 매핑합니다. 18개 이상의 프로바이더, OAuth 흐름, 자격 증명 풀 및 별칭 해석을 처리합니다.

→ [프로바이더 런타임 해석](./provider-runtime.md)

### 도구 시스템

약 28개 도구 세트에 걸쳐 70개 이상의 등록 도구를 제공하는 중앙 도구 레지스트리(`tools/registry.py`)입니다. 각 도구 파일은 가져올 때 스스로 등록합니다. 레지스트리는 스키마 수집, 디스패치, 사용 가능 여부 확인 및 오류 래핑을 처리합니다. 터미널 도구는 7개 백엔드(local, Docker, SSH, Daytona, Modal, Singularity, Vercel Sandbox)를 지원합니다.

→ [도구 런타임](./tools-runtime.md)

### 세션 영속성

FTS5 전문 검색을 지원하는 SQLite 기반 세션 저장소입니다. 세션은 계보 추적(압축 전후의 부모/자식), 플랫폼별 격리, 경합을 처리하는 원자적 쓰기를 지원합니다.

→ [세션 저장소](./session-storage.md)

### 메시징 게이트웨이

25개 이상의 플랫폼 어댑터(내장 및 번들 플러그인), 통합 세션 라우팅, 사용자 인증(허용 목록 및 DM 페어링), 슬래시 명령 디스패치, 훅 시스템, cron 틱 및 백그라운드 유지 관리를 제공하는 장기 실행 프로세스입니다.

→ [게이트웨이 내부 구조](./gateway-internals.md)

### 플러그인 시스템

세 가지 검색 소스가 있습니다: `~/.hermes/plugins/`(사용자), `.hermes/plugins/`(프로젝트), pip 엔트리 포인트입니다. 플러그인은 컨텍스트 API를 통해 도구, 훅 및 CLI 명령을 등록합니다. 두 가지 특수 플러그인 유형인 메모리 프로바이더(`plugins/memory/`)와 컨텍스트 엔진(`plugins/context_engine/`)이 있습니다. 둘 다 단일 선택 방식이므로 한 번에 하나씩만 활성화할 수 있으며, `hermes plugins` 또는 `config.yaml`로 구성합니다.

→ [플러그인 가이드](/developer-guide/plugins), [메모리 프로바이더 플러그인](./memory-provider-plugin.md)

### Cron

셸 작업이 아닌 일급 에이전트 작업입니다. 작업은 JSON에 저장되고 여러 일정 형식을 지원하며, 스킬과 스크립트를 연결하고 모든 플랫폼으로 전달할 수 있습니다.

→ [Cron 내부 구조](./cron-internals.md)

### ACP 통합

stdio/JSON-RPC를 통해 Hermes를 VS Code, Zed 및 JetBrains용 에디터 네이티브 에이전트로 노출합니다.

→ [ACP 내부 구조](./acp-internals.md)

### 트래젝터리

학습 데이터 생성을 위해 에이전트 세션에서 ShareGPT 형식의 트래젝터리를 생성합니다.

→ [트래젝터리 및 학습 형식](./trajectory-format.md)

## 설계 원칙

| 원칙 | 실제 의미 |
|-----------|--------------------------|
| **프롬프트 안정성** | 대화 중 시스템 프롬프트가 변경되지 않습니다. 명시적인 사용자 동작(`/model`)을 제외하면 캐시를 깨는 변경이 없습니다. |
| **관찰 가능한 실행** | 모든 도구 호출은 콜백을 통해 사용자에게 표시됩니다. CLI에서는 진행 상황이 스피너로, 게이트웨이에서는 채팅 메시지로 업데이트됩니다. |
| **중단 가능성** | 사용자 입력이나 시그널을 통해 API 호출과 도구 실행을 진행 중에 취소할 수 있습니다. |
| **플랫폼에 구애받지 않는 코어** | 하나의 AIAgent 클래스가 CLI, 게이트웨이, ACP, 배치 및 API 서버를 처리합니다. 플랫폼별 차이는 에이전트가 아니라 진입점에 존재합니다. |
| **느슨한 결합** | 선택적 하위 시스템(MCP, 플러그인, 메모리 프로바이더, RL 환경)은 하드 의존성이 아니라 레지스트리 패턴과 check_fn 게이팅을 사용합니다. |
| **프로필 격리** | 각 프로필(`hermes -p <name>`)은 자체 HERMES_HOME, 설정, 메모리, 세션 및 게이트웨이 PID를 가집니다. 여러 프로필을 동시에 실행할 수 있습니다. |

## 파일 의존성 체인

```text
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports tools/registry + triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

이 체인에서는 에이전트 인스턴스가 생성되기 전에 가져오기 시점에 도구 등록이 이루어집니다. 최상위 수준에서 `registry.register()`를 호출하는 `tools/*.py` 파일은 자동으로 검색되므로 수동 import 목록이 필요하지 않습니다.
