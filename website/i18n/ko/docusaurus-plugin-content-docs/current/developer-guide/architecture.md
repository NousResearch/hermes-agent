---
sidebar_position: 1
title: "아키텍처 (Architecture)"
description: "Hermes Agent 내부 구조 — 주요 하위 시스템, 실행 경로, 데이터 흐름 및 다음에 읽을 문서"
---

# 아키텍처 (Architecture)

이 페이지는 Hermes Agent 내부 구조의 최상위 지도입니다. 코드베이스의 전체적인 윤곽을 잡은 다음, 구현 세부 사항을 위한 하위 시스템별 문서로 깊이 들어가 보세요.

## 시스템 개요 (System Overview)

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Entry Points (진입점)                         │
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

## 디렉토리 구조 (Directory Structure)

```text
hermes-agent/
├── run_agent.py              # AIAgent — 핵심 대화 루프 (큰 파일)
├── cli.py                    # HermesCLI — 인터랙티브 터미널 UI (큰 파일)
├── model_tools.py            # 도구 검색, 스키마 수집, 디스패치
├── toolsets.py               # 도구 그룹화 및 플랫폼 프리셋
├── hermes_state.py           # FTS5를 사용하는 SQLite 세션/상태 데이터베이스
├── hermes_constants.py       # HERMES_HOME, 프로필 인식 경로
├── batch_runner.py           # 일괄 궤적(trajectory) 생성
│
├── agent/                    # 에이전트 내부
│   ├── prompt_builder.py     # 시스템 프롬프트 조립
│   ├── context_engine.py     # ContextEngine ABC (플러그인 가능)
│   ├── context_compressor.py # 기본 엔진 — 손실 요약(lossy summarization)
│   ├── prompt_caching.py     # Anthropic 프롬프트 캐싱
│   ├── auxiliary_client.py   # 부수적인 작업을 위한 보조 LLM (비전, 요약)
│   ├── model_metadata.py     # 모델 컨텍스트 길이, 토큰 추정
│   ├── models_dev.py         # models.dev 레지스트리 통합
│   ├── anthropic_adapter.py  # Anthropic Messages API 형식 변환
│   ├── display.py            # KawaiiSpinner, 도구 미리보기 포맷팅
│   ├── skill_commands.py     # 스킬 슬래시 명령어
│   ├── memory_manager.py    # 메모리 관리자 오케스트레이션
│   ├── memory_provider.py   # 메모리 제공자 ABC
│   └── trajectory.py         # 궤적 저장 도우미
│
├── hermes_cli/               # CLI 하위 명령어 및 설정
│   ├── main.py               # 진입점 — 모든 `hermes` 하위 명령어 (큰 파일)
│   ├── config.py             # DEFAULT_CONFIG, OPTIONAL_ENV_VARS, 마이그레이션
│   ├── commands.py           # COMMAND_REGISTRY — 중앙 슬래시 명령어 정의
│   ├── auth.py               # PROVIDER_REGISTRY, 자격 증명 확인
│   ├── runtime_provider.py   # Provider → api_mode + 자격 증명
│   ├── models.py             # 모델 카탈로그, 제공자 모델 목록
│   ├── model_switch.py       # /model 명령어 로직 (CLI + 게이트웨이 공유)
│   ├── setup.py              # 인터랙티브 설정 마법사 (큰 파일)
│   ├── skin_engine.py        # CLI 테마 엔진
│   ├── skills_config.py      # hermes skills — 플랫폼별 활성화/비활성화
│   ├── skills_hub.py         # /skills 슬래시 명령어
│   ├── tools_config.py       # hermes tools — 플랫폼별 활성화/비활성화
│   ├── plugins.py            # PluginManager — 검색, 로드, 훅
│   ├── callbacks.py          # 터미널 콜백 (명확화, sudo, 승인)
│   └── gateway.py            # hermes gateway 시작/중지
│
├── tools/                    # 도구 구현 (도구당 하나의 파일)
│   ├── registry.py           # 중앙 도구 레지스트리
│   ├── approval.py           # 위험한 명령어 감지
│   ├── terminal_tool.py      # 터미널 오케스트레이션
│   ├── process_registry.py   # 백그라운드 프로세스 관리
│   ├── file_tools.py         # read_file, write_file, patch, search_files
│   ├── web_tools.py          # web_search, web_extract
│   ├── browser_tool.py       # 10개의 브라우저 자동화 도구
│   ├── code_execution_tool.py # execute_code 샌드박스
│   ├── delegate_tool.py      # 하위 에이전트 위임
│   ├── mcp_tool.py           # MCP 클라이언트 (큰 파일)
│   ├── credential_files.py   # 파일 기반 자격 증명 패스스루
│   ├── env_passthrough.py    # 샌드박스를 위한 환경 변수 패스스루
│   ├── ansi_strip.py         # ANSI 이스케이프 스트리핑
│   └── environments/         # 터미널 백엔드 (local, docker, ssh, modal, daytona, singularity)
│
├── gateway/                  # 메시징 플랫폼 게이트웨이
│   ├── run.py                # GatewayRunner — 메시지 디스패치 (큰 파일)
│   ├── session.py            # SessionStore — 대화 지속성
│   ├── delivery.py           # 아웃바운드 메시지 전달
│   ├── pairing.py            # DM 페어링 권한 부여
│   ├── hooks.py              # 훅 검색 및 수명 주기 이벤트
│   ├── mirror.py             # 세션 간 메시지 미러링
│   ├── status.py             # 토큰 잠금, 프로필 범위 프로세스 추적
│   ├── builtin_hooks/        # 항상 등록되는 훅을 위한 확장 지점 (기본 제공 없음)
│   └── platforms/            # 20개의 어댑터: telegram, discord, slack, whatsapp,
│                             #   signal, matrix, mattermost, email, sms,
│                             #   dingtalk, feishu, wecom, wecom_callback, weixin,
│                             #   bluebubbles, qqbot, homeassistant, webhook, api_server,
│                             #   yuanbao
│
├── acp_adapter/              # ACP 서버 (VS Code / Zed / JetBrains)
├── cron/                     # 스케줄러 (jobs.py, scheduler.py)
├── plugins/memory/           # 메모리 제공자 플러그인
├── plugins/context_engine/   # 컨텍스트 엔진 플러그인
├── skills/                   # 번들 스킬 (항상 사용 가능)
├── optional-skills/          # 공식 선택적 스킬 (명시적 설치 필요)
├── website/                  # Docusaurus 문서 사이트
└── tests/                    # Pytest 스위트 (~1,250개 파일에 걸친 ~25,000개 테스트)
```

## 데이터 흐름 (Data Flow)

### CLI 세션 (CLI Session)

```text
User input → HermesCLI.process_input()
  → AIAgent.run_conversation()
    → prompt_builder.build_system_prompt()
    → runtime_provider.resolve_runtime_provider()
    → API call (chat_completions / codex_responses / anthropic_messages)
    → tool_calls? → model_tools.handle_function_call() → loop
    → final response → display → save to SessionDB
```

### 게이트웨이 메시지 (Gateway Message)

```text
Platform event → Adapter.on_message() → MessageEvent
  → GatewayRunner._handle_message()
    → authorize user
    → resolve session key
    → create AIAgent with session history
    → AIAgent.run_conversation()
    → deliver response back through adapter
```

### 크론 작업 (Cron Job)

```text
Scheduler tick → load due jobs from jobs.json
  → create fresh AIAgent (no history)
  → inject attached skills as context
  → run job prompt
  → deliver response to target platform
  → update job state and next_run
```

## 권장 읽기 순서 (Recommended Reading Order)

코드베이스가 처음이라면 다음 순서로 읽어보세요:

1. **이 페이지** — 전반적인 구조 파악
2. **[에이전트 루프 내부 (Agent Loop Internals)](./agent-loop.md)** — AIAgent의 작동 방식
3. **[프롬프트 조립 (Prompt Assembly)](./prompt-assembly.md)** — 시스템 프롬프트 구성
4. **[제공자 런타임 확인 (Provider Runtime Resolution)](./provider-runtime.md)** — 제공자가 선택되는 방식
5. **[제공자 추가하기 (Adding Providers)](./adding-providers.md)** — 새 제공자를 추가하는 실용적인 가이드
6. **[도구 런타임 (Tools Runtime)](./tools-runtime.md)** — 도구 레지스트리, 디스패치, 환경
7. **[세션 저장소 (Session Storage)](./session-storage.md)** — SQLite 스키마, FTS5, 세션 계보
8. **[게이트웨이 내부 (Gateway Internals)](./gateway-internals.md)** — 메시징 플랫폼 게이트웨이
9. **[컨텍스트 압축 및 프롬프트 캐싱 (Context Compression & Prompt Caching)](./context-compression-and-caching.md)** — 압축 및 캐싱
10. **[ACP 내부 (ACP Internals)](./acp-internals.md)** — IDE 통합

## 주요 하위 시스템 (Major Subsystems)

### 에이전트 루프 (Agent Loop)

동기식 오케스트레이션 엔진(`run_agent.py`의 `AIAgent`)입니다. 제공자 선택, 프롬프트 구성, 도구 실행, 재시도, 폴백, 콜백, 압축 및 지속성을 처리합니다. 서로 다른 제공자 백엔드를 위해 3가지 API 모드를 지원합니다.

→ [에이전트 루프 내부](./agent-loop.md)

### 프롬프트 시스템 (Prompt System)

대화 수명 주기 전반에 걸친 프롬프트 구성 및 유지 관리:

- **`system_prompt.py` + `prompt_builder.py`** — 정렬된 시스템 프롬프트 계층(`stable` → `context` → `volatile`)을 조립합니다: 정체성/도구 지침/스킬, 컨텍스트 파일, 그 다음 메모리/프로필/타임스탬프 블록.
- **`prompt_caching.py`** — 접두사(prefix) 캐싱을 위해 Anthropic 캐시 중단점(breakpoint)을 적용합니다.
- **`context_compressor.py`** — 컨텍스트가 임계값을 초과할 때 중간 대화 턴을 요약합니다.

→ [프롬프트 조립](./prompt-assembly.md), [컨텍스트 압축 및 프롬프트 캐싱](./context-compression-and-caching.md)

### 제공자 확인 (Provider Resolution)

CLI, 게이트웨이, 크론, ACP 및 보조 호출에서 사용하는 공유 런타임 리졸버입니다. `(provider, model)` 튜플을 `(api_mode, api_key, base_url)`에 매핑합니다. 18개 이상의 제공자, OAuth 흐름, 자격 증명 풀 및 별칭 확인을 처리합니다.

→ [제공자 런타임 확인](./provider-runtime.md)

### 도구 시스템 (Tool System)

약 28개의 도구 세트(toolsets)에 걸쳐 70개 이상의 등록된 도구가 있는 중앙 도구 레지스트리(`tools/registry.py`)입니다. 각 도구 파일은 임포트 시 자체 등록됩니다. 레지스트리는 스키마 수집, 디스패치, 가용성 검사 및 오류 래핑을 처리합니다. 터미널 도구는 6개의 백엔드(local, Docker, SSH, Daytona, Modal, Singularity)를 지원합니다.

→ [도구 런타임](./tools-runtime.md)

### 세션 지속성 (Session Persistence)

FTS5 전문 검색(full-text search)을 지원하는 SQLite 기반 세션 스토리지입니다. 세션은 플랫폼별 격리 및 경합 처리를 통한 원자적 쓰기(atomic writes)와 함께 계보 추적(압축에 따른 부모/자식 관계) 기능을 갖추고 있습니다.

→ [세션 저장소](./session-storage.md)

### 메시징 게이트웨이 (Messaging Gateway)

20개의 플랫폼 어댑터, 통합 세션 라우팅, 사용자 권한 부여(허용 목록 + DM 페어링), 슬래시 명령어 디스패치, 훅 시스템, 크론 틱킹(cron ticking) 및 백그라운드 유지 관리가 포함된 장기 실행 프로세스입니다.

→ [게이트웨이 내부](./gateway-internals.md)

### 플러그인 시스템 (Plugin System)

세 가지 검색 소스: `~/.hermes/plugins/` (사용자), `.hermes/plugins/` (프로젝트), 그리고 pip entry points. 플러그인은 컨텍스트 API를 통해 도구, 훅 및 CLI 명령어를 등록합니다. 메모리 제공자(`plugins/memory/`)와 컨텍스트 엔진(`plugins/context_engine/`)이라는 두 가지 특수 플러그인 유형이 존재합니다. 둘 다 단일 선택 방식이며, 한 번에 각각 하나씩만 활성화할 수 있고 `hermes plugins` 또는 `config.yaml`을 통해 설정됩니다.

→ [플러그인 가이드](/guides/build-a-hermes-plugin), [메모리 제공자 플러그인](./memory-provider-plugin.md)

### 크론 (Cron)

일급 에이전트 작업(쉘 작업이 아님). 작업은 JSON으로 저장되고, 여러 일정 형식을 지원하며, 스킬과 스크립트를 연결할 수 있고, 모든 플랫폼으로 전달할 수 있습니다.

→ [크론 내부](./cron-internals.md)

### ACP 통합 (ACP Integration)

VS Code, Zed, JetBrains를 위해 stdio/JSON-RPC를 통해 Hermes를 편집기 기본 에이전트로 노출합니다.

→ [ACP 내부](./acp-internals.md)

### 궤적 (Trajectories)

학습 데이터 생성을 위해 에이전트 세션에서 ShareGPT 형식의 궤적을 생성합니다.

→ [궤적 및 학습 형식](./trajectory-format.md)

## 설계 원칙 (Design Principles)

| 원칙 | 실제 의미 |
|-----------|--------------------------|
| **프롬프트 안정성 (Prompt stability)** | 시스템 프롬프트는 대화 중에 변경되지 않습니다. 명시적인 사용자 작업(`/model`)을 제외하고는 캐시를 깨는 변형이 없습니다. |
| **관찰 가능한 실행 (Observable execution)** | 콜백을 통해 사용자가 모든 도구 호출을 볼 수 있습니다. CLI(스피너) 및 게이트웨이(채팅 메시지)에서 진행 상황 업데이트를 제공합니다. |
| **중단 가능성 (Interruptible)** | 진행 중인 API 호출 및 도구 실행을 사용자 입력이나 신호를 통해 중단할 수 있습니다. |
| **플랫폼 독립적 코어 (Platform-agnostic core)** | 하나의 AIAgent 클래스가 CLI, 게이트웨이, ACP, 배치 및 API 서버를 모두 처리합니다. 플랫폼 차이는 에이전트가 아닌 진입점(entry point)에 위치합니다. |
| **느슨한 결합 (Loose coupling)** | 선택적 하위 시스템(MCP, 플러그인, 메모리 제공자, RL 환경)은 강한 의존성이 아니라 레지스트리 패턴과 check_fn 게이팅을 사용합니다. |
| **프로필 격리 (Profile isolation)** | 각 프로필(`hermes -p <name>`)은 자체 HERMES_HOME, 설정, 메모리, 세션 및 게이트웨이 PID를 갖습니다. 여러 프로필이 동시에 실행될 수 있습니다. |

## 파일 종속성 체인 (File Dependency Chain)

```text
tools/registry.py  (의존성 없음 — 모든 도구 파일에서 가져옴)
       ↑
tools/*.py  (각각 임포트 시 registry.register() 호출)
       ↑
model_tools.py  (tools/registry를 가져오고 + 도구 검색 트리거)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

이 체인은 도구 등록이 에이전트 인스턴스가 생성되기 전에 임포트 시점에 발생함을 의미합니다. 최상위 `registry.register()` 호출이 있는 모든 `tools/*.py` 파일은 자동으로 검색되며, 수동으로 임포트 목록을 작성할 필요가 없습니다.
