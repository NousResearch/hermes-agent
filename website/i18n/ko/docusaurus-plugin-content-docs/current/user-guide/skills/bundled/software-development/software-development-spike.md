---
title: "Spike — 코드를 만들기 전 아이디어를 검증하기 위한 버려지는 실험들"
sidebar_label: "Spike"
description: "코드를 만들기 전 아이디어를 검증하기 위한 버려지는 실험들"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Spike

코드를 만들기 전 아이디어를 검증하기 위한 버려지는(throwaway) 실험들.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/software-development/spike` |
| Version | `1.0.0` |
| Author | Hermes Agent (gsd-build/get-shit-done에서 채택) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `spike`, `prototype`, `experiment`, `feasibility`, `throwaway`, `exploration`, `research`, `planning`, `mvp`, `proof-of-concept` |
| Related skills | [`sketch`](/docs/user-guide/skills/bundled/creative/creative-sketch), [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development), [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# 스파이크 (Spike)

사용자가 실제 구현에 들어가기 전에 실현 가능성을 검증하거나, 접근 방식을 비교하거나, 리서치만으로는 답을 얻을 수 없는 미지의 요소들을 파악하여 **아이디어를 미리 체감해보고 싶어 할 때** 이 스킬을 사용하세요. 스파이크는 설계상 한 번 쓰고 버려집니다. 빚을 청산하고 나면 폐기하세요.

사용자가 "한번 시도해 보자", "X가 작동하는지 보고 싶다", "이거 스파이크 한 번 돌려봐", "Y에 투자하기 전에", "Z의 빠른 프로토타입", "이게 아예 가능한가?", "A와 B를 비교해봐"와 같은 말을 할 때 이 스킬을 로드하세요.

## 언제 이 스킬을 쓰지 말아야 하는가

- 문서나 코드를 읽어서 답을 알 수 있을 때 — 구현하지 말고 그냥 리서치를 수행하세요
- 작업이 프로덕션 경로인 경우 — 대신 `plan` 스킬을 사용하세요
- 아이디어가 이미 검증된 경우 — 바로 구현 단계로 넘어가세요

## 사용자가 전체 GSD 시스템을 설치한 경우

만약 형제 스킬로서 `gsd-spike`가 표시된다면 (`npx get-shit-done-cc --hermes`를 통해 설치됨), 사용자가 완전한 GSD 워크플로를 원할 때 **`gsd-spike`**를 우선 사용하세요: `.planning/spikes/`에 지속되는 상태 보존, 세션 간 MANIFEST 추적, Given/When/Then 판정 형식, 그리고 다른 GSD 시스템과 통합되는 커밋 패턴이 지원됩니다. 이 스킬은 완전한 시스템을 원하지 않거나 없는 사용자를 위한 가벼운 독립형 버전입니다.

## 핵심 방법론

규모에 상관없이 모든 스파이크는 다음 루프를 따릅니다:

```
분해(decompose)  →  조사(research)  →  빌드(build)  →  판정(verdict)
    ↑__________________________________________↓
                   결과에 따라 반복
```

### 1. 분해 (Decompose)

사용자의 아이디어를 **2~5개의 독립적인 타당성 관련 질문**으로 나눕니다. 각 질문이 하나의 스파이크입니다. 이들을 Given/When/Then 프레임을 사용하여 표로 제시하세요:

| # | 스파이크 | 검증 목표 (Given/When/Then) | 위험도 |
|---|-------|----------------------------|------|
| 001 | websocket-streaming | Given 웹소켓 연결이 주어졌을 때, When LLM이 토큰을 스트리밍하면, Then 클라이언트는 < 100ms 지연으로 청크를 받는다 | 높음 |
| 002a | pdf-parse-pdfjs | Given 여러 장의 PDF 파일이 주어지고, When pdfjs로 파싱하면, Then 구조화된 텍스트를 추출할 수 있다 | 중간 |
| 002b | pdf-parse-camelot | Given 여러 장의 PDF 파일이 주어지고, When camelot으로 파싱하면, Then 구조화된 텍스트를 추출할 수 있다 | 중간 |

**스파이크 유형:**
- **표준 (standard)** — 하나의 질문에 답하는 하나의 접근법
- **비교 (comparison)** — 같은 질문에 대한 다른 접근법들 (같은 숫자 번호 공유, 접미사 `a`/`b`/`c` 사용)

**좋은 스파이크 질문:** 관찰 가능한 결과물이 있는 구체적인 타당성 확인.
**나쁜 스파이크 질문:** 너무 광범위하거나, 관찰 가능한 결과물이 없거나, 단순히 "X에 대한 문서 읽기".

**위험도 순으로 정렬하세요.** 이 아이디어를 망칠 가능성이 가장 높은 스파이크를 먼저 실행해야 합니다. 어려운 부분이 작동하지 않는데 쉬운 부분을 프로토타이핑하는 것은 무의미합니다.

사용자가 자신이 테스트하고 싶은 것이 무엇인지 이미 정확히 알고 있고 그것을 명시한 경우에만 **분해 단계를 건너뛰세요.** 이 경우에는 사용자의 아이디어를 단일 스파이크로 취급합니다.

### 2. 정렬 (다중 스파이크 아이디어의 경우)

스파이크 표를 제시하세요. "이 순서대로 전부 빌드할까요, 아니면 수정할까요?"라고 물어보세요. 코드를 작성하기 전에 사용자가 버리거나, 순서를 바꾸거나, 프레임을 다시 잡도록 합니다.

### 3. 리서치 (각 스파이크별, 빌드하기 전)

스파이크는 리서치가 전혀 없는 작업이 아닙니다 — 올바른 접근 방식을 고를 수 있을 만큼 조사한 후 빌드에 들어갑니다. 스파이크마다 다음을 수행하세요:

1. **요약하기.** 2~3문장: 이 스파이크가 무엇인지, 왜 중요한지, 핵심 위험 요소가 무엇인지 설명합니다.
2. **경쟁 접근 방식 비교하기** (실제 선택지가 있는 경우):

   | 접근법 | 도구/라이브러리 | 장점 | 단점 | 상태 |
   |----------|-------------|------|------|--------|
   | ... | ... | ... | ... | 유지됨 / 버려짐 / 베타 |

3. **하나 고르기.** 그 이유를 명시하세요. 만약 2개 이상의 방법이 모두 신뢰할 만하다면 스파이크 내에서 빠른 변형 버전을 빌드하세요.
4. **리서치 건너뛰기** 외부 종속성이 없는 순수 로직의 경우.

리서치 단계에서는 Hermes 도구를 활용하세요:

- `web_search("python websocket streaming libraries 2025")` — 후보 찾기
- `web_extract(urls=["https://websockets.readthedocs.io/..."])` — 실제 문서 읽기 (마크다운 반환)
- `terminal("pip show websockets | grep Version")` — 프로젝트 venv에 무엇이 설치되어 있는지 확인

문서 페이지가 없는 라이브러리의 경우, 저장소를 복제한 뒤 `read_file`로 `README.md` / `examples/`를 읽어보세요. 사용자가 설정해둔 Context7 MCP도 좋은 정보원이 될 수 있습니다 — `mcp_*_resolve-library-id` 사용 후 `mcp_*_query-docs`.

### 4. 빌드

스파이크별로 하나의 디렉토리를 만듭니다. 독립성을 유지하세요.

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

**사용자가 상호작용할 수 있는 것을 만드는 데 중점을 두세요.** 스파이크는 결과가 그저 "잘 됨"이라는 로그 한 줄 뿐이면 실패한 것입니다. 사용자는 스파이크가 동작하는 것을 *느끼고* 싶어 합니다. 권장되는 기본 형태 (선호도 순):

1. 입력을 받고 관찰 가능한 결과를 출력하는 실행 가능한 CLI 프로그램
2. 동작을 시연하는 최소한의 HTML 페이지
3. 하나의 엔드포인트를 가지는 소규모 웹 서버
4. 인식할 수 있는 assertion으로 해당 질문을 실행하는 단위 테스트

**속도보다는 깊이.** 단일 해피-패스 실행 한 번으로 "잘 된다"고 선언하지 마세요. 예외 사례(edge cases)를 테스트하세요. 의외의 결과를 발견하면 따라가 보세요. 판정 결과는 정직한 조사가 선행되었을 때만 신뢰할 수 있습니다.

**스파이크에서 명확히 요구하지 않는 한 피해야 할 것:** 복잡한 패키지 관리, 빌드 도구/번들러, Docker, env 파일, 환경 설정 시스템. 이것들은 스파이크입니다 — 모든 것을 하드코딩하세요.

**단일 스파이크 빌드** — 일반적인 도구 사용 순서:

```
terminal("mkdir -p spikes/001-websocket-streaming")
write_file("spikes/001-websocket-streaming/README.md", "# 001: websocket-streaming\n\n...")
write_file("spikes/001-websocket-streaming/main.py", "...")
terminal("cd spikes/001-websocket-streaming && python3 main.py")
# 출력을 관찰하고, 필요하면 코드를 수정한 뒤 다시 반복합니다.
```

**병렬 비교 스파이크 (002a / 002b) — 위임 활용.** 두 접근 방식이 병렬로 실행될 수 있고, 두 방식 모두 10줄짜리 프로토타입이 아닌 실제 엔지니어링이 필요할 때, `delegate_task`를 활용해 작업을 나누세요:

```
delegate_task(tasks=[
    {"goal": "Build 002a-pdf-parse-pdfjs: ...", "toolsets": ["terminal", "file", "web"]},
    {"goal": "Build 002b-pdf-parse-camelot: ...", "toolsets": ["terminal", "file", "web"]},
])
```

각 서브에이전트가 각자의 결과를 반환하면 당신이 직접 둘의 정면 승부 비교 평가를 작성하세요.

### 5. 판정 (Verdict)

각 스파이크의 `README.md`는 다음 항목으로 마무리합니다:

```markdown
## 판정: 검증됨(VALIDATED) | 부분적(PARTIAL) | 무효화됨(INVALIDATED)

### 작동한 부분 (What worked)
- ...

### 작동하지 않은 부분 (What didn't)
- ...

### 놀라운 점 (Surprises)
- ...

### 실제 구현을 위한 권장 사항 (Recommendation for the real build)
- ...
```

**VALIDATED (검증됨)** = 핵심 질문에 대해 '예'라고 답했으며 증거가 있음.
**PARTIAL (부분적 검증됨)** = X, Y, Z 같은 특정한 제약조건 하에서만 작동함 — 제약 조건을 문서화할 것.
**INVALIDATED (무효화됨)** = 이런저런 이유로 작동하지 않음. 작동하지 않는다는 것을 아는 것만으로도 이 스파이크는 성공적입니다.

## 비교 스파이크 (Comparison spikes)

002a / 002b처럼 두 접근 방식이 같은 질문에 답하는 경우, 이들을 **연달아 빌드**한 다음, 마지막에 1:1 비교를 작성하세요:

```markdown
## 1:1 비교: pdfjs vs camelot

| 기준 | pdfjs (002a) | camelot (002b) |
|-----------|--------------|----------------|
| 추출 품질 | 9/10 (구조화됨) | 7/10 (테이블만 잘됨) |
| 설정 복잡성 | npm install, 1줄이면 끝 | pip + ghostscript |
| 100페이지 PDF 처리 속도 | 3s | 18s |
| 회전된 텍스트 처리 | 불가능 | 가능 |

**승자:** 현재 우리의 유스케이스에서는 pdfjs. 만약 향후 테이블 중심의 추출이 필요해지면 Camelot으로 전환.
```

## 탐구 모드 (다음에 스파이크할 내용 고르기)

기존 스파이크들이 있고 사용자가 "다음엔 뭘 검증해볼까?"라고 묻는다면 기존의 디렉토리들을 살펴본 뒤 다음을 찾아보세요:

- **통합의 위험성** — 개별적으로는 검증되었지만 동일한 리소스에 닿는 두 스파이크가 아직 함께 작동하는지 확인되지 않은 경우
- **데이터 핸드오프(Handoffs)** — 스파이크 A의 출력이 스파이크 B의 입력과 호환된다고 가정했지만 검증된 적이 없는 경우
- **비전의 공백** — 당연히 가능하다고 가정했지만 전혀 입증되지 않은 역량
- **대안 접근법** — PARTIAL이나 INVALIDATED를 받은 스파이크에 대한 다른 접근 시각

이들 중 2~4개의 후보를 Given/When/Then 형태로 제안하세요. 사용자가 고르게 하세요.

## 결과물

- 레포지토리 루트에 `spikes/` 디렉토리를 만드세요 (사용자가 GSD 규칙을 따른다면 `.planning/spikes/`).
- 스파이크 당 1개의 디렉토리: `NNN-descriptive-name/`
- 각 스파이크마다 `README.md` 작성 (질문, 접근 방식, 결과, 판정 등을 기록)
- 코드는 버려도 되게 작성하세요 — "프로덕션을 위해 코드를 깔끔하게 다듬는 데" 이틀이 걸린다면 그건 나쁜 스파이크였습니다.

## 출처

GSD(Get Shit Done) 프로젝트의 `/gsd-spike` 워크플로에서 채택 — MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)). 전체 GSD 시스템은 지속적인 스파이크 상태 보존, MANIFEST 추적, 그리고 더 넓은 명세 주도 개발(spec-driven development) 파이프라인과의 통합을 제공합니다; `npx get-shit-done-cc --hermes --global`을 통해 설치할 수 있습니다.
