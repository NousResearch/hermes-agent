---
title: "Spike — 빌드 전에 아이디어를 검증하기 위한 일회성 실험"
sidebar_label: "Spike"
description: "빌드 전에 아이디어를 검증하기 위한 일회성 실험"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Spike

빌드 전에 아이디어를 검증하기 위한 일회성 실험입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본으로 설치됨) |
| 경로 | `skills/software-development/spike` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent (gsd-build/get-shit-done에서 각색) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `spike`, `prototype`, `experiment`, `feasibility`, `throwaway`, `exploration`, `research`, `planning`, `mvp`, `proof-of-concept` |
| 관련 스킬 | [`sketch`](/docs/user-guide/skills/bundled/creative/creative-sketch), [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development), [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Spike

사용자가 실제 빌드에 착수하기 전에 **아이디어를 가늠해 보고 싶을** 때 이 스킬을 사용하세요. 실현 가능성을 검증하거나, 접근 방식을 비교하거나, 아무리 조사해도 답을 얻을 수 없는 미지의 요소를 드러내는 경우입니다. Spike는 원래 폐기하도록 설계되었습니다. 목적을 다하면 버리세요.

사용자가 다음과 같이 말할 때 로드하세요: "이걸 한번 해 보자", "X가 되는지 보고 싶어", "이걸 spike해 봐", "Y를 결정하기 전에", "Z의 빠른 프로토타입", "이게 가능하기는 한가?" 또는 "A와 B를 비교해 줘".

## 이 스킬을 사용하지 않을 때

- 답을 문서나 코드 읽기로 알 수 있는 경우 — 빌드하지 말고 조사만 하세요
- 작업이 프로덕션 경로인 경우 — 대신 `plan` 스킬을 사용하세요
- 아이디어가 이미 검증된 경우 — 바로 구현으로 넘어가세요

## 사용자가 전체 GSD 시스템을 설치한 경우

`gsd-spike`가 형제 스킬로 표시된다면(npx get-shit-done-cc --hermes로 설치), 사용자가 전체 GSD 워크플로를 원할 때는 **`gsd-spike`**를 우선 사용하세요. 여기에는 지속되는 `.planning/spikes/` 상태, 세션 간 MANIFEST 추적, Given/When/Then 판정 형식, 나머지 GSD와 통합되는 커밋 패턴이 포함됩니다. 이 스킬은 GSD가 없는(또는 원하지 않는) 사용자를 위한 가벼운 독립형 버전입니다.

## 핵심 방법

규모와 관계없이 모든 spike는 다음 루프를 따릅니다:

```
decompose  →  research  →  build  →  verdict
   ↑__________________________________________↓
                  iterate on findings
```

### 1. 분해

사용자의 아이디어를 **2~5개의 독립적인 실현 가능성 질문**으로 나누세요. 각 질문이 하나의 spike입니다. Given/When/Then 형식으로 표에 제시하세요:

| # | Spike | 검증 내용 (Given/When/Then) | 위험도 |
|---|-------|----------------------------|------|
| 001 | websocket-streaming | Given a WS connection, when LLM streams tokens, then client receives chunks &lt; 100ms | 높음 |
| 002a | pdf-parse-pdfjs | Given a multi-page PDF, when parsed with pdfjs, then structured text is extractable | 중간 |
| 002b | pdf-parse-camelot | Given a multi-page PDF, when parsed with camelot, then structured text is extractable | 중간 |

**Spike 유형:**
- **standard** — 하나의 질문에 답하는 하나의 접근 방식
- **comparison** — 같은 질문에 대한 서로 다른 접근 방식(번호를 공유하고 `a`/`b`/`c` 문자 접미사 사용)

**좋은 spike 질문:** 관찰 가능한 결과가 있는 구체적인 실현 가능성 질문입니다.
**나쁜 spike 질문:** 지나치게 광범위하거나, 관찰 가능한 결과가 없거나, 단순히 "X의 문서를 읽어 보자"인 질문입니다.

**위험도순으로 정렬하세요.** 아이디어를 무산시킬 가능성이 가장 높은 spike를 먼저 실행합니다. 어려운 부분이 되지 않는다면 쉬운 부분을 먼저 프로토타이핑할 이유가 없습니다.

사용자가 무엇을 spike할지 이미 정확히 알고 있고 그렇게 말한 경우에만 **분해를 건너뛰세요**. 그때는 사용자의 아이디어를 하나의 spike로 취급하세요.

### 2. 조율(여러 spike 아이디어인 경우)

Spike 표를 제시하세요. "이 순서대로 모두 빌드할까요, 아니면 조정할까요?"라고 물으세요. 코드를 작성하기 전에 사용자가 항목을 삭제하거나, 순서를 바꾸거나, 틀을 다시 잡게 하세요.

### 3. 조사(각 spike별, 빌드 전에)

Spike가 조사 없이 진행된다는 뜻은 아닙니다 — 적절한 접근 방식을 선택할 만큼 조사한 다음 빌드하세요. 각 spike마다 다음을 수행하세요:

1. **개요를 작성하세요.** 이 spike가 무엇인지, 왜 중요한지, 핵심 위험이 무엇인지 2~3문장으로 설명합니다.
2. 실제 선택지가 있다면 **경쟁하는 접근 방식을 드러내세요**:

   | 접근 방식 | 도구/라이브러리 | 장점 | 단점 | 상태 |
   |----------|-------------|------|------|--------|
   | ... | ... | ... | ... | 유지 관리됨 / 중단됨 / 베타 |

3. **하나를 선택하세요.** 이유를 밝힙니다. 신뢰할 수 있는 방법이 2개 이상이면 spike 안에서 빠른 변형을 여러 개 빌드하세요.
4. 외부 의존성이 없는 순수 로직이라면 **조사를 건너뛰세요**.

조사 단계에는 Hermes 도구를 사용하세요:

- `web_search("python websocket streaming libraries 2025")` — 후보 찾기
- `web_extract(urls=["https://websockets.readthedocs.io/..."])` — 실제 문서 읽기(markdown 반환)
- `terminal("pip show websockets | grep Version")` — 프로젝트의 venv에 설치된 버전 확인

문서 페이지가 없는 라이브러리라면 `read_file`로 해당 라이브러리의 `README.md` / `examples/`를 복제하고 읽으세요. Context7 MCP(사용자가 구성한 경우)도 좋은 출처입니다 — `mcp_*_resolve-library-id` 다음에 `mcp_*_query-docs`를 호출하세요.

### 4. 빌드

Spike마다 디렉터리 하나를 사용하세요. 독립적으로 유지하세요.

<!-- ascii-guard-ignore -->
```
spikes/
├── 001-websocket-streaming/
│   ├── README.md
│   └── main.py
├── 002a-pdf-parse-pdfjs/
│   ├── README.md
│   └── parse.js
└── 002b-pdf-parse-camelot/
    ├── README.md
    └── parse.py
```
<!-- ascii-guard-ignore-end -->

**사용자가 직접 상호작용할 수 있는 것을 우선하세요.** 유일한 출력이 "작동합니다"라고 적힌 로그 한 줄뿐이면 spike는 실패합니다. 사용자는 spike가 작동하는 것을 *체감하고* 싶어 합니다. 기본 선택지는 다음 순서로 지정하세요:

1. 입력을 받아 관찰 가능한 출력을 출력하는 실행 가능한 CLI
2. 동작을 보여 주는 최소한의 HTML 페이지
3. 엔드포인트 하나가 있는 작은 웹 서버
4. 알아보기 쉬운 assertion으로 질문을 실행하는 단위 테스트

**속도보다 깊이를 우선하세요.** 한 번의 정상 경로 실행 후에 "작동한다"고 선언하지 마세요. 엣지 케이스를 테스트하세요. 놀라운 결과를 끝까지 추적하세요. 정직한 조사가 이루어진 경우에만 판정 결과를 신뢰할 수 있습니다.

Spike에 특별히 필요하지 않다면 다음을 **피하세요**: 복잡한 패키지 관리, 빌드 도구/번들러, Docker, env 파일, 구성 시스템. 모든 것을 하드코딩하세요 — spike입니다.

**Spike 하나를 빌드하는** 일반적인 도구 순서:

```
terminal("mkdir -p spikes/001-websocket-streaming")
write_file("spikes/001-websocket-streaming/README.md", "# 001: websocket-streaming\n\n...")
write_file("spikes/001-websocket-streaming/main.py", "...")
terminal("cd spikes/001-websocket-streaming && python3 main.py")
# Observe output, iterate.
```

**병렬 비교 spike(002a / 002b) — 위임하세요.** 두 접근 방식을 병렬로 실행할 수 있고 둘 다 실제 엔지니어링이 필요한 경우(10줄짜리 프로토타입이 아닌 경우), `delegate_task`로 분기하세요:

```
delegate_task(tasks=[
    {"goal": "Build 002a-pdf-parse-pdfjs: ...", "toolsets": ["terminal", "file", "web"]},
    {"goal": "Build 002b-pdf-parse-camelot: ...", "toolsets": ["terminal", "file", "web"]},
])
```

각 하위 에이전트가 자체 판정 결과를 반환하면, 정면 비교를 작성하세요.

### 5. 판정

각 spike의 `README.md`는 다음으로 끝나야 합니다:

```markdown
## Verdict: VALIDATED | PARTIAL | INVALIDATED

### What worked
- ...

### What didn't
- ...

### Surprises
- ...

### Recommendation for the real build
- ...
```

**VALIDATED** = 핵심 질문에 증거를 바탕으로 긍정적인 답을 얻었습니다.
**PARTIAL** = X, Y, Z라는 제약 아래에서 작동합니다 — 이를 문서화하세요.
**INVALIDATED** = 이 이유로 작동하지 않습니다. 이것도 성공적인 spike입니다.

## 비교 spike

두 접근 방식이 같은 질문에 답하는 경우(002a / 002b), 연속해서 빌드한 다음 마지막에 정면 비교를 수행하세요:

```markdown
## Head-to-head: pdfjs vs camelot

| Dimension | pdfjs (002a) | camelot (002b) |
|-----------|--------------|----------------|
| Extraction quality | 9/10 structured | 7/10 table-only |
| Setup complexity | npm install, 1 line | pip + ghostscript |
| Perf on 100-page PDF | 3s | 18s |
| Handles rotated text | no | yes |

**Winner:** pdfjs for our use case. Camelot if we need table-first extraction later.
```

## 프런티어 모드(다음에 무엇을 spike할지 선택하기)

Spike가 이미 존재하고 사용자가 "다음에는 무엇을 spike해야 하지?"라고 말하면 기존 디렉터리를 살펴보고 다음을 찾으세요:

- **통합 위험** — 같은 리소스에 접근하지만 독립적으로 테스트된 검증 완료 spike 두 개
- **데이터 전달** — spike A의 출력이 spike B의 입력과 호환된다고 가정했지만 한 번도 입증되지 않은 경우
- **비전의 공백** — 가능하다고 가정했지만 입증되지 않은 기능
- **대안적 접근 방식** — PARTIAL 또는 INVALIDATED spike에 대한 다른 관점

Given/When/Then 형식으로 2~4개의 후보를 제안하세요. 사용자가 선택하게 하세요.

## 출력

- 저장소 루트에 `spikes/`(또는 사용자가 GSD 규칙을 따르는 경우 `.planning/spikes/`)를 생성하세요
- spike마다 디렉터리 하나: `NNN-descriptive-name/`
- 각 spike의 `README.md`에 질문, 접근 방식, 결과, 판정 결과를 기록하세요
- 코드는 폐기 가능한 상태로 유지하세요 — "프로덕션용으로 정리"하는 데 이틀이 걸리는 spike는 나쁜 spike입니다

## 저작자 표시

GSD(Get Shit Done) 프로젝트의 `/gsd-spike` 워크플로를 바탕으로 각색 — MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)). 전체 GSD 시스템은 지속적인 spike 상태, MANIFEST 추적, 더 광범위한 명세 중심 개발 파이프라인과의 통합을 제공합니다. `npx get-shit-done-cc --hermes --global`로 설치하세요.
