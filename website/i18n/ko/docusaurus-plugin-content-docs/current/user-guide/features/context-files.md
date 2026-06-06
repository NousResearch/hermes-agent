---
sidebar_position: 8
title: "컨텍스트 파일 (Context Files)"
description: "프로젝트 컨텍스트 파일 — .hermes.md, AGENTS.md, CLAUDE.md, 전역 SOUL.md, 그리고 .cursorrules — 모든 대화에 자동으로 주입됩니다."
---

# 컨텍스트 파일 (Context Files)

Hermes Agent는 에이전트의 동작 방식을 형성하는 컨텍스트 파일을 자동으로 검색하고 로드합니다. 일부는 프로젝트 로컬 파일이며 작업 디렉토리에서 발견됩니다. `SOUL.md`는 이제 Hermes 인스턴스에 전역적이며 `HERMES_HOME`에서만 로드됩니다.

## 지원되는 컨텍스트 파일

| 파일 | 목적 | 검색 범위 |
|------|---------|-----------| 
| **.hermes.md** / **HERMES.md** | 프로젝트 지침 (최우선 순위) | git 루트까지 탐색 |
| **AGENTS.md** | 프로젝트 지침, 규칙, 아키텍처 | 시작 시 CWD + 점진적으로 하위 디렉토리 |
| **CLAUDE.md** | Claude Code 컨텍스트 파일 (감지됨) | 시작 시 CWD + 점진적으로 하위 디렉토리 |
| **SOUL.md** | 이 Hermes 인스턴스를 위한 전역 성격과 어조(tone) 커스터마이징 | `HERMES_HOME/SOUL.md` 전용 |
| **.cursorrules** | Cursor IDE 코딩 규칙 | CWD 전용 |
| **.cursor/rules/*.mdc** | Cursor IDE 규칙 모듈 | CWD 전용 |

:::info 우선순위 시스템
세션당 오직 **하나**의 프로젝트 컨텍스트 유형만 로드됩니다 (첫 번째로 일치하는 파일이 적용됨): `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. **SOUL.md**는 항상 에이전트 식별자(슬롯 #1)로서 독립적으로 로드됩니다.
:::

## AGENTS.md

`AGENTS.md`는 주요 프로젝트 컨텍스트 파일입니다. 이것은 에이전트에게 프로젝트가 어떻게 구성되어 있는지, 따라야 할 규칙은 무엇인지, 특별한 지침은 무엇인지 알려줍니다.

### 점진적 하위 디렉토리 검색

세션 시작 시, Hermes는 작업 디렉토리에 있는 `AGENTS.md`를 시스템 프롬프트에 로드합니다. 에이전트가 세션 중에 (`read_file`, `terminal`, `search_files` 등을 통해) 하위 디렉토리로 이동함에 따라, 해당 디렉토리의 컨텍스트 파일을 **점진적으로 검색**하여 필요해지는 시점에 대화에 주입합니다.

```
my-project/
├── AGENTS.md              ← 시작 시 로드됨 (시스템 프롬프트)
├── frontend/
│   └── AGENTS.md          ← 에이전트가 frontend/ 파일을 읽을 때 검색됨
├── backend/
│   └── AGENTS.md          ← 에이전트가 backend/ 파일을 읽을 때 검색됨
└── shared/
    └── AGENTS.md          ← 에이전트가 shared/ 파일을 읽을 때 검색됨
```

이 접근 방식은 시작할 때 모든 것을 로드하는 것보다 두 가지 장점이 있습니다:
- **시스템 프롬프트가 부풀려지지 않음** — 하위 디렉토리 힌트는 필요할 때만 나타납니다.
- **프롬프트 캐시 보존** — 시스템 프롬프트는 대화 턴(turn) 동안 안정적으로 유지됩니다.

각 하위 디렉토리는 세션당 최대 한 번만 확인됩니다. 검색은 상위 디렉토리로도 이동하므로, `backend/src/main.py`를 읽을 때 `backend/src/`에 자체 컨텍스트 파일이 없더라도 `backend/AGENTS.md`를 발견하게 됩니다.

:::info
하위 디렉토리 컨텍스트 파일은 시작 컨텍스트 파일과 동일한 [보안 스캔](#보안-프롬프트-주입-보호-security-prompt-injection-protection)을 거칩니다. 악성 파일은 차단됩니다.
:::

### AGENTS.md 예시

```markdown
# 프로젝트 컨텍스트

이것은 Python FastAPI 백엔드를 사용하는 Next.js 14 웹 애플리케이션입니다.

## 아키텍처
- 프론트엔드: `/frontend`에 App Router가 있는 Next.js 14
- 백엔드: `/backend`에 위치한 FastAPI, SQLAlchemy ORM 사용
- 데이터베이스: PostgreSQL 16
- 배포: Hetzner VPS의 Docker Compose

## 규칙
- 모든 프론트엔드 코드에는 TypeScript 엄격 모드(strict mode)를 사용하세요
- Python 코드는 PEP 8을 따르며, 모든 곳에 타입 힌트를 사용하세요
- 모든 API 엔드포인트는 `{data, error, meta}` 형태의 JSON을 반환합니다
- 테스트는 프론트엔드의 경우 `__tests__/` 디렉토리에, 백엔드의 경우 `tests/`에 작성합니다

## 중요 참고 사항
- 마이그레이션 파일을 직접 수정하지 마세요 — Alembic 명령어를 사용하세요
- `.env.local` 파일에는 실제 API 키가 있으므로 커밋하지 마세요
- 프론트엔드 포트는 3000, 백엔드는 8000, 데이터베이스는 5432입니다
```

## SOUL.md

`SOUL.md`는 에이전트의 성격, 어조, 의사소통 스타일을 제어합니다. 전체 세부 정보는 [성격 (Personality)](/user-guide/features/personality) 페이지를 참조하세요.

**위치:**

- `~/.hermes/SOUL.md`
- 또는 사용자 지정 홈 디렉토리로 Hermes를 실행하는 경우 `$HERMES_HOME/SOUL.md`

중요 세부 사항:

- 아직 `SOUL.md`가 없는 경우 Hermes가 자동으로 기본값을 생성(seed)합니다.
- Hermes는 `HERMES_HOME`에서만 `SOUL.md`를 로드합니다.
- Hermes는 `SOUL.md`를 찾기 위해 작업 디렉토리를 탐색하지 않습니다.
- 파일이 비어 있으면, `SOUL.md`의 어떤 내용도 프롬프트에 추가되지 않습니다.
- 파일에 내용이 있으면, 내용을 스캔하고 잘라낸 후 그대로(verbatim) 주입됩니다.

## .cursorrules

Hermes는 Cursor IDE의 `.cursorrules` 파일 및 `.cursor/rules/*.mdc` 규칙 모듈과 호환됩니다. 이러한 파일이 프로젝트 루트에 있고 더 높은 우선순위의 컨텍스트 파일(`.hermes.md`, `AGENTS.md` 또는 `CLAUDE.md`)이 발견되지 않으면 프로젝트 컨텍스트로 로드됩니다.

이는 Hermes를 사용할 때 기존 Cursor 규칙이 자동으로 적용됨을 의미합니다.

## 컨텍스트 파일이 로드되는 방식

### 시작 시 (시스템 프롬프트)

컨텍스트 파일은 `agent/prompt_builder.py`의 `build_context_files_prompt()`에 의해 로드됩니다:

1. **작업 디렉토리 스캔** — `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules` 순으로 확인합니다 (첫 번째 일치 항목 적용).
2. **내용 읽기** — 각 파일은 UTF-8 텍스트로 읽힙니다.
3. **보안 스캔** — 내용에서 프롬프트 주입(prompt injection) 패턴을 확인합니다.
4. **자르기(Truncation)** — 20,000자를 초과하는 파일은 앞/뒤 부분이 잘립니다 (앞 70%, 뒤 20%를 남기고 중간에 마커 삽입).
5. **조립(Assembly)** — 모든 섹션이 `# Project Context` 헤더 아래 결합됩니다.
6. **주입(Injection)** — 조립된 내용이 시스템 프롬프트에 추가됩니다.

### 세션 중 (점진적 발견)

`agent/subdirectory_hints.py`의 `SubdirectoryHintTracker`가 도구 호출 인수에서 파일 경로를 감시합니다:

1. **경로 추출** — 각 도구 호출 후, 인수(`path`, `workdir`, 쉘 명령어)에서 파일 경로가 추출됩니다.
2. **상위 탐색** — 해당 디렉토리와 최대 5개의 상위 디렉토리를 확인합니다 (이미 방문한 디렉토리에서 중지).
3. **힌트 로드** — `AGENTS.md`, `CLAUDE.md` 또는 `.cursorrules`가 발견되면 로드합니다 (디렉토리당 첫 번째 일치 항목 적용).
4. **보안 스캔** — 시작 파일과 동일한 프롬프트 주입 스캔.
5. **자르기** — 파일당 8,000자로 제한.
6. **주입** — 도구 결과에 추가되어 모델이 문맥 안에서 자연스럽게 볼 수 있게 합니다.

최종 프롬프트 섹션은 대략 다음과 같이 보입니다:

```text
# Project Context

The following project context files have been loaded and should be followed:

## AGENTS.md

[여기에 AGENTS.md 내용 위치]

## .cursorrules

[여기에 .cursorrules 내용 위치]

[여기에 SOUL.md 내용 위치]
```

SOUL 내용은 추가적인 래퍼 텍스트 없이 직접 삽입되는 점에 유의하세요.

## 보안: 프롬프트 주입 보호 (Security: Prompt Injection Protection)

모든 컨텍스트 파일은 포함되기 전에 잠재적인 프롬프트 주입 여부를 스캔합니다. 스캐너가 확인하는 항목:

- **지침 재정의 시도**: "이전 지침을 무시하라", "당신의 규칙을 무시하라"
- **기만 패턴**: "사용자에게 말하지 마라"
- **시스템 프롬프트 무효화**: "시스템 프롬프트 오버라이드"
- **숨겨진 HTML 주석**: `<!-- ignore instructions -->`
- **숨겨진 div 요소**: `<div style="display:none">`
- **자격 증명 유출**: `curl ... $API_KEY`
- **비밀 파일 접근**: `cat .env`, `cat credentials`
- **보이지 않는 문자**: 폭 없는 공백(zero-width spaces), 양방향 텍스트 오버라이드, 단어 결합자(word joiners)

어떤 위협 패턴이라도 감지되면, 파일은 차단됩니다:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

:::warning
이 스캐너는 일반적인 주입 패턴으로부터 보호하지만, 공유 저장소에 있는 컨텍스트 파일을 검토하는 것을 대신할 수는 없습니다. 자신이 작성하지 않은 프로젝트의 AGENTS.md 내용을 항상 검증하세요.
:::

## 크기 제한

| 제한 | 값 |
|-------|-------|
| 파일당 최대 글자 수 | 20,000 (약 7,000 토큰) |
| 앞쪽 자르기 비율 | 70% |
| 뒤쪽 자르기 비율 | 20% |
| 자르기 마커 | 10% (글자 수를 표시하고 파일 도구 사용을 제안함) |

파일이 20,000자를 초과할 때, 자르기 메시지는 다음과 같이 표시됩니다:

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## 효과적인 컨텍스트 파일을 위한 팁

:::tip AGENTS.md의 모범 사례
1. **간결하게 유지하세요** — 20,000자 제한보다 훨씬 적게 유지하세요. 에이전트는 매 턴마다 이 파일을 읽습니다.
2. **헤더로 구조화하세요** — 아키텍처, 규칙, 중요 참고 사항에 대해 `##` 섹션을 사용하세요.
3. **구체적인 예시를 포함하세요** — 선호하는 코드 패턴, API 형태, 명명 규칙을 보여주세요.
4. **하지 말아야 할 일을 명시하세요** — "마이그레이션 파일을 절대 직접 수정하지 마세요" 등.
5. **주요 경로와 포트를 나열하세요** — 에이전트가 터미널 명령어에 이를 활용합니다.
6. **프로젝트 발전에 따라 업데이트하세요** — 오래된 컨텍스트는 아예 없는 것보다 못합니다.
:::

### 하위 디렉토리별 컨텍스트

모노레포의 경우 하위 디렉토리별 지침을 중첩된 AGENTS.md 파일에 넣으세요:

```markdown
<!-- frontend/AGENTS.md -->
# 프론트엔드 컨텍스트

- 패키지 관리를 위해 `npm`이 아닌 `pnpm`을 사용하세요.
- 컴포넌트는 `src/components/`에, 페이지는 `src/app/`에 배치합니다.
- 인라인 스타일을 사용하지 말고 Tailwind CSS를 사용하세요.
- `pnpm test`로 테스트를 실행하세요.
```

```markdown
<!-- backend/AGENTS.md -->
# 백엔드 컨텍스트

- 의존성 관리를 위해 `poetry`를 사용하세요.
- `poetry run uvicorn main:app --reload`로 개발 서버를 실행합니다.
- 모든 엔드포인트에는 OpenAPI 독스트링이 필요합니다.
- 데이터베이스 모델은 `models/`에, 스키마는 `schemas/`에 있습니다.
```
