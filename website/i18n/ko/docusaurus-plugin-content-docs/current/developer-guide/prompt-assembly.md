---
sidebar_position: 5
title: "프롬프트 조립 (Prompt Assembly)"
description: "Hermes가 시스템 프롬프트를 구성하고, 캐시 안정성을 유지하며, 임시(ephemeral) 레이어를 주입하는 방법"
---

# 프롬프트 조립 (Prompt Assembly)

Hermes는 다음과 같은 두 가지를 의도적으로 분리합니다:

- **캐시된 시스템 프롬프트 상태**
- **API 호출 시 일시적으로 추가되는 항목 (ephemeral)**

이는 프로젝트에서 가장 중요한 설계 선택 중 하나이며, 다음에 영향을 미칩니다:

- 토큰 사용량
- 프롬프트 캐싱의 효율성
- 세션 연속성
- 메모리 정확성

주요 파일:

- `run_agent.py`
- `agent/prompt_builder.py`
- `tools/memory_tool.py`

## 캐시된 시스템 프롬프트 레이어

캐시된 시스템 프롬프트는 3개의 순서가 지정된 티어(tier)로 조립됩니다 (`agent/system_prompt.py` 참고):

1. **안정(stable)** — 정체성(`SOUL.md` 또는 대체값), 도구/모델 가이드라인, 기술(skills) 프롬프트, 환경 힌트, 플랫폼 힌트
2. **컨텍스트(context)** — 호출자가 제공한 `system_message` 및 프로젝트 컨텍스트 파일들(`.hermes.md` / `AGENTS.md` / `CLAUDE.md` / `.cursorrules`)
3. **가변(volatile)** — 내장된 메모리 스냅샷(`MEMORY.md`), 사용자 프로필 스냅샷(`USER.md`), 외부 메모리 제공자 블록, 타임스탬프/세션/모델/제공자 줄

최종 시스템 프롬프트는 `stable` → `context` → `volatile` 순서로 결합됩니다.

우선순위 논의에서 이 순서는 매우 중요합니다:
- 기술(skills)은 **stable** 티어에 속합니다.
- 메모리/프로필 스냅샷은 **volatile** 티어에 속합니다.
- 두 가지 모두 캐시된 시스템 프롬프트에 속합니다 (턴 도중의 임시 오버레이로 주입되지 않음).

`skip_context_files`가 설정된 경우(예: 하위 에이전트 위임), `SOUL.md`는 로드되지 않고 하드코딩된 `DEFAULT_AGENT_IDENTITY`가 대신 사용됩니다.

### 구체적인 예: 조립된 시스템 프롬프트

모든 레이어가 존재할 때 최종 시스템 프롬프트가 어떤 모습인지 보여주는 단순화된 예입니다 (주석은 각 섹션의 출처를 나타냄):

```text
# Layer 1: 에이전트 정체성 (Agent Identity, ~/.hermes/SOUL.md 에서 가져옴)
You are Hermes, an AI assistant created by Nous Research.
You are an expert software engineer and researcher.
You value correctness, clarity, and efficiency.
...

# Layer 2: 도구 인식 동작 가이드라인 (Tool-aware behavior guidance)
You have persistent memory across sessions. Save durable facts using
the memory tool: user preferences, environment details, tool quirks,
and stable conventions. Memory is injected into every turn, so keep
it compact and focused on facts that will still matter later.
...
When the user references something from a past conversation or you
suspect relevant cross-session context exists, use session_search
to recall it before asking them to repeat themselves.

# 도구 사용 강제 (GPT/Codex 모델 전용)
You MUST use your tools to take action — do not describe what you
would do or plan to do without actually doing it.
...

# Layer 3: Honcho 정적 블록 (활성화된 경우)
[Honcho personality/context data]

# Layer 4: 선택적 시스템 메시지 (설정 또는 API에서 가져온 재정의)
[User-configured system message override]

# Layer 5: 고정된 MEMORY 스냅샷
## Persistent Memory
- User prefers Python 3.12, uses pyproject.toml
- Default editor is nvim
- Working on project "atlas" in ~/code/atlas
- Timezone: US/Pacific

# Layer 6: 고정된 USER 프로필 스냅샷
## User Profile
- Name: Alice
- GitHub: alice-dev

# Layer 7: 기술 인덱스 (Skills index)
## Skills (mandatory)
Before replying, scan the skills below. If one clearly matches
your task, load it with skill_view(name) and follow its instructions.
...
<available_skills>
  software-development:
    - code-review: Structured code review workflow
    - test-driven-development: TDD methodology
  research:
    - arxiv: Search and summarize arXiv papers
</available_skills>

# Layer 8: 컨텍스트 파일 (프로젝트 디렉터리에서 가져옴)
# Project Context
The following project context files have been loaded and should be followed:

## AGENTS.md
This is the atlas project. Use pytest for testing. The main
entry point is src/atlas/main.py. Always run `make lint` before
committing.

# Layer 9: 타임스탬프 + 세션
Current time: 2026-03-30T14:30:00-07:00
Session: abc123

# Layer 10: 플랫폼 힌트
You are a CLI AI Agent. Try not to use markdown but simple text
renderable inside a terminal.
```

## SOUL.md가 프롬프트에 나타나는 방식

`SOUL.md`는 `~/.hermes/SOUL.md`에 위치하며 시스템 프롬프트의 첫 번째 섹션인 에이전트의 정체성 역할을 합니다. `prompt_builder.py`의 로딩 로직은 다음과 같이 작동합니다:

```python
# agent/prompt_builder.py의 단순화된 코드
def load_soul_md() -> Optional[str]:
    soul_path = get_hermes_home() / "SOUL.md"
    if not soul_path.exists():
        return None
    content = soul_path.read_text(encoding="utf-8").strip()
    content = _scan_context_content(content, "SOUL.md")  # 보안 스캔
    content = _truncate_content(content, "SOUL.md")       # 20k 자로 제한
    return content
```

`load_soul_md()`가 내용을 반환하면 하드코딩된 `DEFAULT_AGENT_IDENTITY`를 대체합니다. 그런 다음 `SOUL.md`가 프롬프트에 두 번 (한 번은 정체성으로, 한 번은 컨텍스트 파일로) 나타나는 것을 방지하기 위해 `skip_soul=True` 인자와 함께 `build_context_files_prompt()`가 호출됩니다.

`SOUL.md`가 존재하지 않으면 시스템은 다음과 같은 대체 메시지를 사용합니다:

```text
You are Hermes Agent, an intelligent AI assistant created by Nous Research.
You are helpful, knowledgeable, and direct. You assist users with a wide
range of tasks including answering questions, writing and editing code,
analyzing information, creative work, and executing actions via your tools.
You communicate clearly, admit uncertainty when appropriate, and prioritize
being genuinely useful over being verbose unless otherwise directed below.
Be targeted and efficient in your exploration and investigations.
```

## 컨텍스트 파일이 주입되는 방식

`build_context_files_prompt()`는 **우선순위 시스템(priority system)**을 사용합니다. 오직 하나의 프로젝트 컨텍스트 유형만 로드됩니다 (가장 먼저 일치하는 항목 우선):

```python
# agent/prompt_builder.py의 단순화된 코드
def build_context_files_prompt(cwd=None, skip_soul=False):
    cwd_path = Path(cwd).resolve()

    # 우선순위: 먼저 일치하는 것이 우선 — 단 하나의 프로젝트 컨텍스트만 로드됨
    project_context = (
        _load_hermes_md(cwd_path)       # 1. .hermes.md / HERMES.md (git 루트까지 위로 검색)
        or _load_agents_md(cwd_path)    # 2. AGENTS.md (cwd에서만 검색)
        or _load_claude_md(cwd_path)    # 3. CLAUDE.md (cwd에서만 검색)
        or _load_cursorrules(cwd_path)  # 4. .cursorrules / .cursor/rules/*.mdc
    )

    sections = []
    if project_context:
        sections.append(project_context)

    # HERMES_HOME에 있는 SOUL.md (프로젝트 컨텍스트와 독립적)
    if not skip_soul:
        soul_content = load_soul_md()
        if soul_content:
            sections.append(soul_content)

    if not sections:
        return ""

    return (
        "# Project Context\n\n"
        "The following project context files have been loaded "
        "and should be followed:\n\n"
        + "\n".join(sections)
    )
```

### 컨텍스트 파일 발견에 대한 상세 내용

| 우선순위 | 파일 | 검색 범위 | 참고 |
|----------|-------|-------------|-------|
| 1 | `.hermes.md`, `HERMES.md` | CWD부터 git 루트까지 | Hermes 네이티브 프로젝트 구성 |
| 2 | `AGENTS.md` | CWD에서만 | 일반적인 에이전트 명령어 파일 |
| 3 | `CLAUDE.md` | CWD에서만 | Claude Code 호환성 |
| 4 | `.cursorrules`, `.cursor/rules/*.mdc` | CWD에서만 | Cursor 호환성 |

모든 컨텍스트 파일은 다음을 거칩니다:
- **보안 검사** — 보이지 않는 유니코드 문자, "이전 지시 무시(ignore previous instructions)" 등 프롬프트 인젝션 패턴, 자격 증명 유출 시도가 있는지 확인합니다.
- **길이 제한 (Truncated)** — 70/20 비율(처음 70%, 마지막 20%)과 생략 마커를 사용하여 20,000자로 제한됩니다.
- **YAML Frontmatter 제거** — `.hermes.md`의 frontmatter가 제거됩니다 (추후 구성 덮어쓰기를 위해 예약됨).

## API 호출 시점에만 주입되는 임시 레이어

다음 항목들은 의도적으로 캐시된 시스템 프롬프트의 일부로 **저장되지 않습니다**:

- `ephemeral_system_prompt`
- 사전 채우기 메시지 (prefill messages)
- 게이트웨이 파생 세션 컨텍스트 오버레이
- 이전 턴의 Honcho/외부 리콜이 현재 턴의 사용자 메시지에 주입된 것

`pre_llm_call` 플러그인 컨텍스트 또한 이 API 호출 시점 경로에 해당합니다: 이 내용은 캐시된 시스템 프롬프트에 기록되는 것이 아니라 현재 턴의 **사용자 메시지** 끝에 덧붙여집니다. 여러 플러그인이 컨텍스트를 반환할 때 Hermes는 이러한 컨텍스트 블록들을 이어 붙입니다 ([훅(Hooks) → `pre_llm_call`](../user-guide/features/hooks.md#pre_llm_call) 참고).

이러한 분리를 통해 안정적인 접두사(stable prefix)를 캐싱에 유리하도록 유지합니다.

## 메모리 스냅샷

로컬 메모리 및 사용자 프로필 데이터는 시스템 프롬프트의 **가변(volatile) 티어**에 캡처됩니다. 세션 진행 중에 발생하는 메모리 기록은 디스크 상태를 업데이트하지만, 재구축(rebuild) 경로(새 세션 생성 또는 압축에 의한 재구축 흐름과 같은 명시적인 무효화/재구축 프로세스)가 실행되기 전까지는 이미 빌드된 캐시 시스템 프롬프트를 변형하지 않습니다.

## 컨텍스트 파일

`agent/prompt_builder.py`는 **우선순위 시스템**을 사용하여 프로젝트 컨텍스트 파일을 스캔하고 살균(sanitize)합니다. 단 하나의 유형만 로드됩니다(가장 먼저 일치하는 항목이 선택됨):

1. `.hermes.md` / `HERMES.md` (git 루트까지 위로 검색)
2. `AGENTS.md` (시작 시 CWD; 하위 디렉터리는 `agent/subdirectory_hints.py`를 통해 세션 중에 점진적으로 발견됨)
3. `CLAUDE.md` (CWD에서만 검색)
4. `.cursorrules` / `.cursor/rules/*.mdc` (CWD에서만 검색)

`SOUL.md`는 정체성 자리(identity slot)를 위해 `load_soul_md()`를 통해 별도로 로드됩니다. 성공적으로 로드된 경우 `build_context_files_prompt(skip_soul=True)`를 통해 중복으로 나타나는 것을 방지합니다.

긴 파일은 주입되기 전에 잘립니다.

## 기술 인덱스 (Skills index)

기술 시스템(skills system)은 기술 도구를 사용할 수 있을 때 컴팩트한 기술 인덱스를 프롬프트에 제공합니다.

## 지원되는 프롬프트 커스터마이징 방법

대부분의 사용자는 `agent/prompt_builder.py`를 설정 표면(configuration surface)이 아닌 구현 코드로 취급해야 합니다. 지원되는 커스터마이징 방식은 파이썬 템플릿을 직접 수정하는 것이 아니라, Hermes가 이미 로드하고 있는 프롬프트 입력값 자체를 변경하는 것입니다.

### 다음 방법들을 먼저 사용하세요

- `~/.hermes/SOUL.md` — 내장된 기본 정체성 블록을 여러분만의 에이전트 페르소나 및 기본 동작으로 교체하세요.
- `~/.hermes/MEMORY.md` 및 `~/.hermes/USER.md` — 새 세션에 스냅샷으로 저장되어야 하는 세션 간 지속적인 사실 및 사용자 프로필 데이터를 제공하세요.
- 프로젝트 컨텍스트 파일(`.hermes.md`, `HERMES.md`, `AGENTS.md`, `CLAUDE.md`, 또는 `.cursorrules` 등) — 저장소(repo) 고유의 작업 규칙을 주입하세요.
- 기술 (Skills) — 핵심 프롬프트 코드를 수정하지 않고 재사용 가능한 워크플로 및 참조 문서를 패키징하세요.
- 선택적 시스템 프롬프트 설정 / API 재정의 — Hermes를 포크(fork)하지 않고 배포 환경 전용 지시문 텍스트를 추가하세요.
- 임시 오버레이(`HERMES_EPHEMERAL_SYSTEM_PROMPT` 또는 사전 채우기 메시지 등) — 캐시된 프롬프트 접두사의 일부가 되지 않아야 하는 특정 턴 전용(turn-scoped) 지침을 추가하세요.

### 코드 편집이 필요한 경우

Hermes를 포크하여 유지 관리하거나 상위(upstream) 동작 변경 사항에 기여하려는 의도가 있는 경우에만 `agent/prompt_builder.py`를 직접 편집하세요. 이 파일은 모든 세션에 대해 프롬프트 연결 구조(plumbing), 캐시 경계 및 주입 순서를 조립합니다. 이곳을 직접 수정하는 것은 전역적인 제품 변경이지 사용자별 프롬프트 커스터마이징이 아닙니다.

다시 말해:

- 에이전트 정체성을 바꾸고 싶다면 `SOUL.md`를 편집하세요.
- 저장소(repo) 규칙을 바꾸고 싶다면 프로젝트 컨텍스트 파일을 편집하세요.
- 재사용 가능한 표준 운영 절차를 원한다면 기술(skills)을 추가하거나 수정하세요.
- Hermes가 모든 사람을 위해 프롬프트를 조립하는 방식 자체를 바꾸고 싶다면 파이썬 코드를 수정하고 이를 코드 기여(code contribution)로 취급하세요.

## 프롬프트 조립을 이렇게 분리한 이유

이 아키텍처는 다음과 같은 목적을 위해 의도적으로 최적화되었습니다:

- 제공자 측의 프롬프트 캐싱 보존
- 불필요한 기록 변형 방지
- 메모리 작동 방식을 이해하기 쉽게 유지
- 게이트웨이/ACP/CLI가 영구적인 프롬프트 상태를 오염시키지 않고 컨텍스트를 추가할 수 있게 함

## 관련 문서

- [Context Compression & Prompt Caching](./context-compression-and-caching.md)
- [Session Storage](./session-storage.md)
- [Gateway Internals](./gateway-internals.md)
