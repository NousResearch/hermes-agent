---
sidebar_position: 5
title: "프롬프트 조립"
description: "Hermes가 시스템 프롬프트를 구성하고, 캐시 안정성을 유지하며, 임시 계층을 주입하는 방법"
---

# 프롬프트 조립

Hermes는 의도적으로 다음을 분리합니다.

- **캐시된 시스템 프롬프트 상태**
- **일시적인 API 호출 시점 추가 항목**

이는 다음 항목에 영향을 주기 때문에 프로젝트에서 가장 중요한 설계 선택 중 하나입니다.

- 토큰 사용량
- 프롬프트 캐싱 효과
- 세션 연속성
- 메모리 정확성

주요 파일:

- `run_agent.py`
- `agent/prompt_builder.py`
- `tools/memory_tool.py`

## 캐시된 시스템 프롬프트 계층

캐시된 시스템 프롬프트는 순서가 있는 세 개의 계층으로 구성됩니다(`agent/system_prompt.py` 참조).

1. **stable** — 정체성(`SOUL.md` 또는 대체값), 도구/모델 지침, 스킬 프롬프트, 환경 힌트, 플랫폼 힌트
2. **context** — 호출자가 제공한 `system_message`와 프로젝트 컨텍스트 파일(`.hermes.md` / `AGENTS.md` / `CLAUDE.md` / `.cursorrules`)
3. **volatile** — 내장 메모리 스냅샷(`MEMORY.md`), 사용자 프로필 스냅샷(`USER.md`), 외부 메모리 제공자 블록, 타임스탬프/세션/모델/제공자 행

최종 시스템 프롬프트는 `stable` → `context` → `volatile` 순서로 결합됩니다.

이 순서는 우선순위 논의에서 중요합니다.
- 스킬은 **stable** 계층의 일부입니다.
- 메모리/프로필 스냅샷은 **volatile** 계층의 일부입니다.
- 둘 다 여전히 캐시된 시스템 프롬프트에 포함되며(턴 중간에 임시 오버레이로 주입되지 않으며),

`skip_context_files`가 설정되면(예: 서브에이전트 위임 시) SOUL.md가 로드되지 않고 하드코딩된 `DEFAULT_AGENT_IDENTITY`가 대신 사용됩니다.

### 구체적인 예: 조립된 시스템 프롬프트

모든 계층이 존재할 때 최종 시스템 프롬프트가 어떤 모습인지 단순화해 나타내면 다음과 같습니다(주석은 각 섹션의 출처를 보여 줍니다).

```
# Layer 1: Agent Identity (from ~/.hermes/SOUL.md)
You are Hermes, an AI assistant created by Nous Research.
You are an expert software engineer and researcher.
You value correctness, clarity, and efficiency.
...

# Layer 2: Tool-aware behavior guidance
You have persistent memory across sessions. Save durable facts using
the memory tool: user preferences, environment details, tool quirks,
and stable conventions. Memory is injected into every turn, so keep
it compact and focused on facts that will still matter later.
...
When the user references something from a past conversation or you
suspect relevant cross-session context exists, use session_search
to recall it before asking them to repeat themselves.

# Tool-use enforcement (for GPT/Codex models only)
You MUST use your tools to take action — do not describe what you
would do or plan to do without actually doing it.
...

# Layer 3: Honcho static block (when active)
[Honcho personality/context data]

# Layer 4: Optional system message (from config or API)
[User-configured system message override]

# Layer 5: Frozen MEMORY snapshot
## Persistent Memory
- User prefers Python 3.12, uses pyproject.toml
- Default editor is nvim
- Working on project "atlas" in ~/code/atlas
- Timezone: US/Pacific

# Layer 6: Frozen USER profile snapshot
## User Profile
- Name: Alice
- GitHub: alice-dev

# Layer 7: Skills index
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

# Layer 8: Context files (from project directory)
# Project Context
The following project context files have been loaded and should be followed:

## AGENTS.md
This is the atlas project. Use pytest for testing. The main
entry point is src/atlas/main.py. Always run `make lint` before
committing.

# Layer 9: Timestamp + session
Current time: 2026-03-30T14:30:00-07:00
Session: abc123

# Layer 10: Platform hint
You are a CLI AI Agent. Try not to use markdown but simple text
renderable inside a terminal.
```

## 플랫폼 힌트 사용자 지정

플랫폼 힌트(위의 Layer 10)는 Hermes가 Telegram, WhatsApp, Slack, CLI 및 기타 플랫폼을 위해 주입하는 표면별 지침입니다. 예를 들어 "터미널에 있으므로 Markdown을 피하세요"와 같습니다. 내장 기본값은 `PLATFORM_HINTS`(`agent/system_prompt.py`)에 있으며, 플러그인이 제공하는 플랫폼은 플랫폼 레지스트리를 통해 자체 힌트를 제공합니다.

관리자는 다른 플랫폼을 건드리지 않고 최상위 `platform_hints` 키를 통해 `config.yaml`에서 특정 플랫폼의 힌트를 추가하거나 대체할 수 있습니다.

```yaml
platform_hints:
  whatsapp:
    append: >
      When tabular output would be useful, invoke the table_formatting
      skill instead of emitting a Markdown table.
  slack:
    replace: "You are on Slack. Keep responses tight and avoid wide tables."
  telegram: "Prefer short messages; split long answers."   # shorthand = append
```

- `append` — 내장 힌트를 유지하고 그 뒤에 추가 텍스트를 덧붙입니다.
- `replace` — 내장 힌트를 완전히 대체합니다.
- 일반 문자열 — `append`의 축약형입니다.
- `replace`와 `append`가 모두 있으면 `replace`가 우선합니다.
- 잘못된 항목은 방어적으로 무시되고 수정되지 않은 기본값으로 대체되므로, 잘못된 구성 값이 프롬프트 조립을 중단시키거나 플랫폼 간에 유출될 수 없습니다.

오버라이드는 시스템 프롬프트가 빌드될 때(세션 시작 시, 그리고 프롬프트를 다시 빌드하는 압축 시) 해석됩니다. 고정된 구성에 대해 바이트 단위로 안정적인 힌트를 생성하므로 내장 힌트와 함께 **stable** 계층에 포함되며 프롬프트 캐싱을 깨뜨리지 않습니다. 이는 고정된 프롬프트를 세션 중에 실시간으로 변경하는 방식이 아닙니다.

## 프롬프트에 SOUL.md가 표시되는 방식

`SOUL.md`는 `~/.hermes/SOUL.md`에 있으며 시스템 프롬프트의 첫 번째 섹션인 에이전트의 정체성 역할을 합니다. `prompt_builder.py`의 로딩 로직은 다음과 같이 동작합니다.

```python
# From agent/prompt_builder.py (simplified)
def load_soul_md() -> Optional[str]:
    soul_path = get_hermes_home() / "SOUL.md"
    if not soul_path.exists():
        return None
    content = soul_path.read_text(encoding="utf-8").strip()
    content = _scan_context_content(content, "SOUL.md")  # Security scan
    content = _truncate_content(content, "SOUL.md")       # Cap scales with model context window (20k floor); config override wins
    return content
```

`load_soul_md()`가 콘텐츠를 반환하면 하드코딩된 `DEFAULT_AGENT_IDENTITY`를 대체합니다. 그런 다음 `build_context_files_prompt()` 함수가 `skip_soul=True`로 호출되어 SOUL.md가 정체성으로 한 번, 컨텍스트 파일로 한 번 총 두 번 표시되지 않도록 합니다.

`SOUL.md`가 존재하지 않으면 시스템은 다음 대체값으로 돌아갑니다.

```
You are Hermes Agent, an intelligent AI assistant created by Nous Research.
You are helpful, knowledgeable, and direct. You assist users with a wide
range of tasks including answering questions, writing and editing code,
analyzing information, creative work, and executing actions via your tools.
You communicate clearly, admit uncertainty when appropriate, and prioritize
being genuinely useful over being verbose unless otherwise directed below.
Be targeted and efficient in your exploration and investigations.
```

## 컨텍스트 파일을 주입하는 방법

`build_context_files_prompt()`는 우선순위 시스템을 사용하며, 프로젝트 컨텍스트 유형 중 하나만 로드합니다(먼저 일치하는 항목이 우선).

```python
# From agent/prompt_builder.py (simplified)
def build_context_files_prompt(cwd=None, skip_soul=False):
    cwd_path = Path(cwd).resolve()

    # Priority: first match wins — only ONE project context loaded
    project_context = (
        _load_hermes_md(cwd_path)       # 1. .hermes.md / HERMES.md (walks to git root)
        or _load_agents_md(cwd_path)    # 2. AGENTS.md (cwd only)
        or _load_claude_md(cwd_path)    # 3. CLAUDE.md (cwd only)
        or _load_cursorrules(cwd_path)  # 4. .cursorrules / .cursor/rules/*.mdc
    )

    sections = []
    if project_context:
        sections.append(project_context)

    # SOUL.md from HERMES_HOME (independent of project context)
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

### 컨텍스트 파일 탐색 세부 사항

| 우선순위 | 파일 | 검색 범위 | 참고 |
|----------|-------|-----------|------|
| 1 | `.hermes.md`, `HERMES.md` | CWD에서 git root까지 | Hermes 네이티브 프로젝트 구성 |
| 2 | `AGENTS.md` | CWD만 | 일반적인 에이전트 지침 파일 |
| 3 | `CLAUDE.md` | CWD만 | Claude Code 호환성 |
| 4 | `.cursorrules`, `.cursor/rules/*.mdc` | CWD만 | Cursor 호환성 |

모든 컨텍스트 파일은 다음 처리를 거칩니다.
- **보안 스캔** — 프롬프트 인젝션 패턴(보이지 않는 유니코드, "이전 지침 무시", 자격 증명 탈취 시도)을 검사합니다.
- **잘림** — 70/20 헤드/테일 분할과 잘림 표시를 사용해 `context_file_max_chars` 문자로 제한됩니다. 제한값은 모델의 컨텍스트 창에 맞춰 조정되며(최솟값 20,000자, 최댓값 500K), `config.yaml`의 명시적인 `context_file_max_chars`가 항상 우선합니다.
- **YAML frontmatter 제거** — `.hermes.md`의 frontmatter는 제거됩니다(향후 구성 오버라이드를 위해 예약됨).

## API 호출 시점에만 적용되는 계층

다음 항목은 캐시된 시스템 프롬프트의 일부로 저장되지 않도록 의도적으로 제외됩니다.

- `ephemeral_system_prompt`
- prefill 메시지
- gateway에서 파생된 세션 컨텍스트 오버레이
- 이후 턴에서 Honcho/외부 리콜이 현재 턴의 사용자 메시지에 주입하는 내용

`pre_llm_call` 플러그인 컨텍스트도 이 API 호출 시점 경로에 포함됩니다. 즉, 캐시된 시스템 프롬프트에 기록되지 않고 현재 턴의 **user message**에 추가됩니다. 여러 플러그인이 컨텍스트를 반환하면 Hermes는 해당 컨텍스트 블록을 연결합니다([Hooks → `pre_llm_call`](../user-guide/features/hooks.md#pre_llm_call) 참조).

이러한 분리를 통해 캐싱을 위한 안정적인 접두사가 안정적으로 유지됩니다.

## 메모리 스냅샷

로컬 메모리와 사용자 프로필 데이터는 시스템 프롬프트의 **volatile** 계층에 캡처됩니다. 세션 중간에 기록된 내용은 디스크 상태를 업데이트하지만, 이미 빌드된 캐시 시스템 프롬프트를 새 세션이나 압축으로 인한 재빌드처럼 재빌드 경로가 실행될 때까지 변경하지 않습니다.

## 컨텍스트 파일

`agent/prompt_builder.py`는 **우선순위 시스템**을 사용해 프로젝트 컨텍스트 파일을 스캔하고 정제합니다. 한 유형만 로드됩니다(먼저 일치하는 항목이 우선).

1. `.hermes.md` / `HERMES.md` (git root까지 탐색)
2. `AGENTS.md` (시작 시 CWD, 세션 중 `agent/subdirectory_hints.py`를 통해 하위 디렉터리를 점진적으로 탐색)
3. `CLAUDE.md` (CWD만)
4. `.cursorrules` / `.cursor/rules/*.mdc` (CWD만)

`SOUL.md`는 정체성 슬롯을 위해 `load_soul_md()`를 통해 별도로 로드됩니다. 성공적으로 로드되면 `build_context_files_prompt(skip_soul=True)`가 SOUL.md가 두 번 표시되는 것을 방지합니다.

긴 파일은 주입 전에 잘립니다.

## 스킬 인덱스

스킬 도구를 사용할 수 있을 때 스킬 시스템은 간결한 스킬 인덱스를 프롬프트에 추가합니다.

## 지원되는 프롬프트 사용자 지정 표면

대부분의 사용자는 `agent/prompt_builder.py`를 구성 표면이 아닌 구현 코드로 다뤄야 합니다. 지원되는 사용자 지정 경로는 Python 템플릿을 직접 수정하는 대신 Hermes가 이미 로드하는 프롬프트 입력을 변경하는 것입니다.

### 다음 표면을 먼저 사용하세요

- `~/.hermes/SOUL.md` — 내장 기본 정체성 블록을 자신만의 에이전트 페르소나와 상시 동작으로 대체합니다.
- `~/.hermes/MEMORY.md` 및 `~/.hermes/USER.md` — 새 세션에 스냅샷으로 포함해야 하는 세션 간 영속 사실과 사용자 프로필 데이터를 제공합니다.
- `.hermes.md`, `HERMES.md`, `AGENTS.md`, `CLAUDE.md` 또는 `.cursorrules` 같은 프로젝트 컨텍스트 파일 — 저장소별 작업 규칙을 주입합니다.
- 스킬 — 핵심 프롬프트 코드를 수정하지 않고 재사용 가능한 워크플로와 참고 자료를 패키징합니다.
- 선택적 시스템 프롬프트 구성 / API 오버라이드 — Hermes를 포크하지 않고 배포별 지침 텍스트를 추가합니다.
- `HERMES_EPHEMERAL_SYSTEM_PROMPT` 또는 prefill 메시지 같은 임시 오버레이 — 캐시된 프롬프트 접두사의 일부가 되어서는 안 되는 턴 범위의 지침을 추가합니다.
### 대신 코드를 수정해야 하는 경우

의도적으로 포크를 유지 관리하거나 업스트림 동작 변경에 기여하는 경우에만 `agent/prompt_builder.py`를 수정하세요. 이 파일은 모든 세션의 프롬프트 연결, 캐시 경계 및 주입 순서를 조합합니다. 직접 수정하는 것은 사용자별 프롬프트 사용자 지정이 아니라 전역 제품 변경입니다.

다시 말해:

- 어시스턴트의 정체성을 다르게 지정하려면 `SOUL.md`를 수정하세요.
- 저장소 규칙을 다르게 지정하려면 프로젝트 컨텍스트 파일을 수정하세요.
- 재사용 가능한 운영 절차를 추가하거나 수정하려면 스킬을 추가하거나 수정하세요.
- 모든 사용자를 대상으로 Hermes가 프롬프트를 조합하는 방식을 변경하려면 Python을 변경하고 코드 기여로 다루세요.

## 프롬프트 조합이 이렇게 분리된 이유

이 아키텍처는 다음을 최적화하도록 의도적으로 설계되었습니다.

- 제공업체 측 프롬프트 캐싱을 보존합니다.
- 기록을 불필요하게 변경하지 않습니다.
- 메모리 의미를 이해하기 쉽게 유지합니다.
- gateway/ACP/CLI가 영구 프롬프트 상태를 오염시키지 않고 컨텍스트를 추가할 수 있게 합니다.

## 관련 문서

- [컨텍스트 압축 및 프롬프트 캐싱](./context-compression-and-caching.md)
- [세션 저장소](./session-storage.md)
- [Gateway 내부 구조](./gateway-internals.md)
