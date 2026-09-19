---
sidebar_position: 8
title: "컨텍스트 파일"
description: "모든 대화에 자동으로 주입되는 프로젝트 컨텍스트 파일 — .hermes.md, AGENTS.md, CLAUDE.md, 전역 SOUL.md, .cursorrules"
---

# 컨텍스트 파일

Hermes Agent는 동작 방식을 결정하는 컨텍스트 파일을 자동으로 찾아서 불러옵니다. 일부 파일은 작업 디렉터리에서 찾아내는 프로젝트 로컬 파일입니다. 이제 `SOUL.md`는 Hermes 인스턴스 전역에서 사용되며 `HERMES_HOME`에서만 불러옵니다.

## 지원되는 컨텍스트 파일

| 파일 | 용도 | 검색 위치 |
|------|---------|-----------| 
| **.hermes.md** / **HERMES.md** | 프로젝트 지침(최우선) | git 루트까지 상위 디렉터리 탐색 |
| **AGENTS.md** | 프로젝트 지침, 규칙, 아키텍처 | 시작 시 CWD + 하위 디렉터리를 점진적으로 탐색 |
| **CLAUDE.md** | Claude Code 컨텍스트 파일(함께 감지됨) | 시작 시 CWD + 하위 디렉터리를 점진적으로 탐색 |
| **SOUL.md** | 이 Hermes 인스턴스의 전역 성격 및 말투 사용자 지정 | `HERMES_HOME/SOUL.md`만 해당 |
| **.cursorrules** | Cursor IDE 코딩 규칙 | CWD만 해당 |
| **.cursor/rules/*.mdc** | Cursor IDE 규칙 모듈 | CWD만 해당 |

:::info 우선순위 시스템
세션마다 프로젝트 컨텍스트 유형은 **하나만** 불러옵니다(처음 일치하는 항목 우선): `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. **SOUL.md**는 에이전트 정체성(슬롯 #1)으로 항상 독립적으로 불러옵니다.
:::

## AGENTS.md

`AGENTS.md`는 기본 프로젝트 컨텍스트 파일입니다. 프로젝트 구조와 따라야 할 규칙, 특별 지침을 에이전트에 알려줍니다.

### 디렉터리 체인(git 루트 → 작업 디렉터리)

작업 디렉터리가 git 저장소 안에 있으면 Hermes는 세션 시작 시 `AGENTS.md` 파일의 병합된 체인을 불러옵니다. git 루트의 `AGENTS.md`를 먼저 불러온 다음, 작업 디렉터리까지 이어지는 모든 중간 디렉터리의 `AGENTS.md`를 불러옵니다. 더 깊은 파일이 프롬프트 뒤쪽에 나타나므로 더 구체적인 지침이 우선합니다. 각 파일에는 자체 출처 헤더(예: `## ../../AGENTS.md`)가 붙고, 체인에 있는 동일한 사본은 중복 제거됩니다.

```
monorepo/                   (git root, cwd = packages/webapp/)
├── AGENTS.md              ← Loaded first (repo-wide conventions)
└── packages/
    ├── AGENTS.md          ← Loaded second
    └── webapp/
        └── AGENTS.md      ← Loaded last (most specific, takes precedence)
```

### 하위 디렉터리 점진적 탐색

세션 시작 시 Hermes는 작업 디렉터리의 `AGENTS.md`를 시스템 프롬프트에 불러옵니다. 에이전트가 세션 중 도구를 통해 하위 디렉터리로 이동하면(예: `read_file`, `terminal`, `search_files`), 해당 디렉터리의 컨텍스트 파일을 점진적으로 발견하고 관련 시점에 대화에 주입합니다.

```
my-project/
├── AGENTS.md              ← Loaded at startup (system prompt)
├── frontend/
│   └── AGENTS.md          ← Discovered when agent reads frontend/ files
├── backend/
│   └── AGENTS.md          ← Discovered when agent reads backend/ files
└── shared/
    └── AGENTS.md          ← Discovered when agent reads shared/ files
```

이 방식에는 두 가지 장점이 있습니다.
- 시스템 프롬프트가 비대해지지 않습니다 — 하위 디렉터리 지침은 필요할 때만 나타납니다
- 프롬프트 캐시가 유지됩니다 — 시스템 프롬프트가 턴 사이에 안정적으로 유지됩니다

각 하위 디렉터리는 세션당 최대 한 번만 확인합니다. 또한 탐색은 상위 디렉터리로 올라가므로, 하위 디렉터리에 자체 파일이 없더라도 `backend/src/main.py`를 읽으면 `backend/AGENTS.md`가 발견됩니다.

:::info
하위 디렉터리 컨텍스트 파일에도 시작 시 컨텍스트 파일과 동일한 [보안 검사](#security-prompt-injection-protection)가 적용됩니다. 악성 파일은 차단됩니다.
:::

### AGENTS.md 예시

```markdown
# Project Context

This is a Next.js 14 web application with a Python FastAPI backend.

## Architecture
- Frontend: Next.js 14 with App Router in `/frontend`
- Backend: FastAPI in `/backend`, uses SQLAlchemy ORM
- Database: PostgreSQL 16
- Deployment: Docker Compose on a Hetzner VPS

## Conventions
- Use TypeScript strict mode for all frontend code
- Python code follows PEP 8, use type hints everywhere
- All API endpoints return JSON with `{data, error, meta}` shape
- Tests go in `__tests__/` directories (frontend) or `tests/` (backend)

## Important Notes
- Never modify migration files directly — use Alembic commands
- The `.env.local` file has real API keys, don't commit it
- Frontend port is 3000, backend is 8000, DB is 5432
```

## SOUL.md

`SOUL.md`는 에이전트의 성격과 말투를 제어합니다. 자세한 내용은 [성격](/user-guide/features/personality) 페이지를 참조하세요.

**위치:**

- `~/.hermes/SOUL.md`
- 또는 사용자 지정 홈 디렉터리로 Hermes를 실행하는 경우 `$HERMES_HOME/SOUL.md`

중요한 세부 사항:

- Hermes는 아직 `SOUL.md`가 없으면 기본 `SOUL.md`를 자동으로 생성합니다
- Hermes는 `SOUL.md`를 `HERMES_HOME`에서만 불러옵니다
- Hermes는 작업 디렉터리에서 `SOUL.md`를 검색하지 않습니다
- 파일이 비어 있으면 `SOUL.md`의 어떤 내용도 프롬프트에 추가되지 않습니다
- 파일에 내용이 있으면 스캔 및 잘림 처리 후 그 내용을 그대로 주입합니다

## .cursorrules

Hermes는 Cursor IDE의 `.cursorrules` 파일과 `.cursor/rules/*.mdc` 규칙 모듈을 지원합니다. 상위 우선순위 컨텍스트 파일(`.hermes.md`, `AGENTS.md`, `CLAUDE.md`)을 찾지 못하고 프로젝트 루트에 해당 파일이 있는 경우 불러옵니다.

즉, 기존 Cursor 규칙이 Hermes 사용 시 자동으로 적용됩니다.

## 컨텍스트 파일을 불러오는 방법

### 시작 시(시스템 프롬프트)

컨텍스트 파일은 `build_context_files_prompt()` in `agent/prompt_builder.py`에서 불러옵니다:

1. **작업 디렉터리 검색** — `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`를 확인합니다(처음 일치하는 항목 우선)
2. **콘텐츠 읽기** — 각 파일을 UTF-8 텍스트로 읽습니다
3. **보안 검사** — 프롬프트 인젝션 패턴이 있는지 콘텐츠를 확인합니다
4. **잘림 처리** — 문자 제한을 초과하는 파일은 앞/뒤를 잘라냅니다(앞 70%, 뒤 20%, 중간에 표시 추가). 제한은 설정된 경우 config.yaml의 명시적인 `context_file_max_chars`이며, 그렇지 않으면 모델의 컨텍스트 창에 따라 동적으로 조정됩니다(최소 20,000자, 최대 500,000자)
5. **조합** — 모든 섹션을 `# Project Context` 헤더 아래에 합칩니다
6. **주입** — 조합된 콘텐츠를 시스템 프롬프트에 추가합니다

### 세션 중(점진적 탐색)

`agent/subdirectory_hints.py`의 `SubdirectoryHintTracker`가 도구 호출 인수에서 파일 경로를 감시합니다.

1. **경로 추출** — 각 도구 호출 후 인수(`path`, `workdir`, 셸 명령)에서 파일 경로를 추출합니다
2. **상위 디렉터리 탐색** — 디렉터리와 최대 5개의 상위 디렉터리를 확인합니다(이미 방문한 디렉터리에서 중단)
3. **힌트 불러오기** — `AGENTS.md`, `CLAUDE.md` 또는 `.cursorrules`가 있으면 불러옵니다(디렉터리별 첫 일치 항목)
4. **보안 검사** — 시작 시와 동일한 프롬프트 인젝션 검사를 수행합니다
5. **잘림 처리** — 파일당 8,000자로 제한합니다
6. **주입** — 도구 결과에 추가하여 에이전트가 자연스럽게 확인하도록 합니다

최종 프롬프트 섹션은 대략 다음과 같습니다:

```text
# Project Context

The following project context files have been loaded and should be followed:

## AGENTS.md

[Your AGENTS.md content here]

## .cursorrules

[Your .cursorrules content here]

[Your SOUL.md content here]
```

SOUL 콘텐츠는 추가 래퍼 텍스트 없이 직접 삽입된다는 점에 유의하세요.

## 보안: 프롬프트 인젝션 방지

모든 컨텍스트 파일은 포함되기 전에 잠재적인 프롬프트 인젝션 검사를 거칩니다. 스캐너는 다음을 확인합니다.

- **지침 재정의 시도**: "ignore previous instructions", "disregard your rules"
- **기만 패턴**: "do not tell the user"
- **시스템 프롬프트 재정의**: "system prompt override"
- **숨겨진 HTML 주석**: `<!-- ignore instructions -->`
- **숨겨진 div 요소**: `<div style="display:none">`
- **자격 증명 유출**: `curl ... $API_KEY`
- **비밀 파일 접근**: `cat .env`, `cat credentials`
- **보이지 않는 문자**: 0 너비 공백, 양방향 재정의 문자, 단어 결합자

위협 패턴이 감지되면 파일이 차단됩니다.

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

:::warning
이 스캐너는 일반적인 인젝션 패턴을 방지하지만, 공유 저장소의 컨텍스트 파일을 검토하는 것을 대신할 수는 없습니다. 직접 작성하지 않은 프로젝트에서는 항상 `AGENTS.md` 콘텐츠를 검증하세요.
:::

## 크기 제한

| 제한 | 값 |
|-------|-------|
| 파일당 최대 문자 수 | 설정된 경우 `context_file_max_chars`; 그렇지 않으면 동적(모델 컨텍스트 창에 따라 조정되며 최소 20,000자, 최대 500,000자) |
| 앞부분 잘림 비율 | 70% |
| 뒷부분 잘림 비율 | 20% |
| 잘림 표시 | 10%(문자 수를 표시하고 파일 도구 사용을 제안) |

파일이 설정된 제한을 초과하면 잘림 메시지는 다음과 같습니다.

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## 효과적인 컨텍스트 파일을 위한 팁

:::tip AGENTS.md 모범 사례
1. **간결하게 유지하세요** — 에이전트는 매 턴 파일을 읽으므로 설정된 `context_file_max_chars`보다 짧게 유지합니다
2. **헤더로 구조화하세요** — 아키텍처, 규칙, 중요 참고 사항에 `##` 섹션을 사용합니다
3. **구체적인 예를 포함하세요** — 선호하는 코드 패턴, API 형태, 명명 규칙을 보여줍니다
4. **하지 말아야 할 일을 언급하세요** — "마이그레이션 파일을 직접 수정하지 마세요"
5. **주요 경로와 포트를 나열하세요** — 에이전트가 터미널 명령에 사용합니다
6. **프로젝트가 발전하면 업데이트하세요** — 오래된 컨텍스트는 컨텍스트가 없는 것보다 좋지 않습니다
:::

### 하위 디렉터리별 컨텍스트

모노레포에서는 중첩된 `AGENTS.md` 파일에 하위 디렉터리별 지침을 배치합니다.

```markdown
<!-- frontend/AGENTS.md -->
# Frontend Context

- Use `pnpm` not `npm` for package management
- Components go in `src/components/`, pages in `src/app/`
- Use Tailwind CSS, never inline styles
- Run tests with `pnpm test`
```

```markdown
<!-- backend/AGENTS.md -->
# Backend Context

- Use `poetry` for dependency management
- Run the dev server with `poetry run uvicorn main:app --reload`
- All endpoints need OpenAPI docstrings
- Database models are in `models/`, schemas in `schemas/`
```
