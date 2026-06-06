---
title: "Sketch — 일회용 HTML 목업: 비교할 수 있는 2-3가지 디자인 변형"
sidebar_label: "Sketch"
description: "일회용 HTML 목업: 비교할 수 있는 2-3가지 디자인 변형"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sketch

일회용 HTML 목업: 비교할 수 있는 2-3가지 디자인 변형을 제공합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/sketch` |
| 버전 | `1.0.0` |
| 저자 | Hermes Agent (gsd-build/get-shit-done에서 각색) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `sketch`, `mockup`, `design`, `ui`, `prototype`, `html`, `variants`, `exploration`, `wireframe`, `comparison` |
| 관련 스킬 | [`spike`](/docs/user-guide/skills/bundled/software-development/software-development-spike), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# Sketch (스케치)

사용자가 하나의 방향으로 **결정하기 전에 디자인 방향을 보고 싶어 할 때** — UI/UX 아이디어를 일회용 HTML 목업으로 탐색할 때 이 스킬을 사용합니다. 목적은 출시 가능한 코드를 생성하는 것이 아니라, 사용자가 시각적 방향을 나란히 비교할 수 있도록 2-3개의 인터랙티브한 변형을 만드는 것입니다.

사용자가 "이 화면을 스케치해줘", "X가 어떤 모습일지 보여줘", "레이아웃 A와 B를 비교해줘", "이 UI에 대한 2-3가지 버전을 보여줘", "변형을 몇 가지 볼 수 있게 해줘", "내가 만들기 전에 이걸 목업해줘"와 같이 말할 때 로드하세요.

## 이 스킬을 사용하면 안 되는 경우

- 사용자가 프로덕션 컴포넌트를 원할 때 — `claude-design`을 사용하거나 제대로 구축하세요.
- 사용자가 세련된 단일 HTML 아티팩트(랜딩 페이지, 덱)를 원할 때 — `claude-design`을 사용하세요.
- 사용자가 다이어그램을 원할 때 — `excalidraw`, `architecture-diagram`을 사용하세요.
- 디자인이 이미 확정되었을 때 — 그냥 구축하세요.

## 사용자가 전체 GSD 시스템을 설치한 경우

`gsd-sketch`가 형제 스킬(`npx get-shit-done-cc --hermes`를 통해 설치됨)로 나타나면, MANIFEST를 포함한 지속적인 `.planning/sketches/`, 프론티어 모드 분석, 과거 스케치 전반에 걸친 일관성 감사 및 나머지 GSD 시스템과의 통합을 제공하는 전체 워크플로우를 위해 **`gsd-sketch`**를 우선적으로 사용하세요. 이 스킬은 가벼운 독립형 버전 — 상태 머신이 없는 일회성 스케치 작업입니다.

## 핵심 방법

```
의견 수렴(intake)  →  변형(variants)  →  맞대결(head-to-head)  →  승자 선택(또는 반복)
```

### 1. 의견 수렴 (사용자가 이미 충분한 정보를 제공했다면 건너뜀)

변형을 생성하기 전에, 한 번에 하나씩 질문하여 다음 세 가지를 파악하세요:

1. **느낌 (Feel).** "어떤 느낌이어야 하나요? 형용사, 감정, 분위기를 알려주세요." — *"차분한, 편집 디자인 같은, Linear 같은"*은 *"미니멀한"*보다 더 많은 것을 알려줍니다.
2. **참조 (References).** "상상하고 있는 느낌을 잘 담아낸 앱, 사이트 또는 제품은 무엇인가요?" — 실제 참조 자료가 추상적인 설명보다 낫습니다.
3. **핵심 행동 (Core action).** "이 화면에서 사용자가 하는 가장 중요한 한 가지 행동은 무엇인가요?" — 모든 변형은 이 행동을 잘 지원해야 합니다. 그렇지 않다면 그저 장식일 뿐입니다.

다음 질문을 하기 전에 각 답변을 간략하게 반영하세요. 사용자가 이미 세 가지 정보를 모두 제공했다면 바로 변형 생성으로 넘어갑니다.

### 2. 변형 (Variants) (2-3개, 절대 1개가 아니며 4개 이상은 드묾)

한 번에 **2-3가지 변형**을 생성하세요. 각 변형은 완전한 독립형 HTML 파일입니다. 변형을 설명하지 마세요 — 구축하세요. 핵심은 비교입니다.

각 변형은 단순히 픽셀 값이 다른 것이 아니라 **다른 디자인 스탠스(stance)**를 취해야 합니다. 세 가지 좋은 변형 축은 다음과 같습니다:

- **밀도 (Density):** 콤팩트 / 여유로움 / 초고밀도 (대비되는 두 극단을 선택)
- **강조 (Emphasis):** 콘텐츠 우선 / 액션 우선 / 도구 우선
- **미학 (Aesthetic):** 편집 디자인풍 / 실용적 / 장난스러움
- **레이아웃 (Layout):** 단일 열 / 사이드바 / 분할 창
- **기반 (Grounding):** 카드 기반 / 베어 콘텐츠 / 문서 스타일

하나의 축을 선택하고 그를 기준으로 나눕니다. 강조 색상만 다른 두 변형은 노력의 낭비입니다 — 사용자는 그 차이를 구별하지 못합니다.

**변형 명명:** 번호가 아닌 스탠스를 설명하세요.

<!-- ascii-guard-ignore -->
```
sketches/
├── 001-calm-editorial/
│   ├── index.html
│   └── README.md
├── 001-utilitarian-dense/
│   ├── index.html
│   └── README.md
└── 001-playful-split/
    ├── index.html
    └── README.md
```
<!-- ascii-guard-ignore-end -->

### 3. 실제 HTML로 만들기

각 변형은 **단일 독립형 HTML 파일**입니다:

- 인라인 `<style>` — 빌드 단계나 외부 CSS 없음
- 시스템 폰트 또는 `<link>`를 통한 하나의 Google Font
- CDN을 통한 Tailwind(`<script src="https://cdn.tailwindcss.com"></script>`) 사용 가능
- 현실적인 가짜 콘텐츠 — "Lorem ipsum"이 아닌 실제 문장, 실제 이름 사용
- **인터랙티브**: 링크는 클릭 가능해야 하고, 호버 효과가 실제적이어야 하며, 적어도 하나의 상태 전환(열기/닫기, 필터, 토글)이 있어야 합니다. 얼어붙은 정적 이미지는 조잡하게 애니메이션된 이미지보다 나쁜 스파이크(spike)입니다.

브라우저에서 열어보세요. 깨져 보인다면 사용자에게 보여주기 전에 고치세요.

**변형을 시각적으로 검증하세요 — Hermes의 브라우저 도구를 사용하세요.** HTML을 작성하고 제대로 렌더링되기를 바라기만 하지 마세요. 각 변형을 로드하고 살펴보세요:

```
browser_navigate(url="file:///absolute/path/to/sketches/001-calm-editorial/index.html")
browser_vision(question="이 레이아웃이 깔끔하고 읽기 쉽게 보이나요? 눈에 띄는 버그(겹치는 텍스트, 스타일이 적용되지 않은 요소, 깨진 이미지)가 있나요?")
```

`browser_vision`은 페이지에 실제로 있는 내용에 대한 AI 설명과 스크린샷 경로를 반환합니다 — 순수한 소스 코드 검사로는 놓칠 수 있는 레이아웃 버그(예: 조용히 실패한 폰트 임포트, 무너진 flex 컨테이너)를 포착합니다. 각 변형이 제대로 보일 때까지 수정하고 다시 탐색하세요.

빠른 시작을 위한 **기본 CSS 리셋 + 시스템 폰트 스택**:

```html
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    color: #1a1a1a;
    background: #fafafa;
    line-height: 1.5;
  }
</style>
```

### 4. 변형 README

각 변형의 `README.md`는 다음 질문에 답합니다:

```markdown
## Variant: {스탠스 이름}

### 디자인 스탠스
이 변형을 주도하는 원칙에 대한 한 문장.

### 주요 선택 사항
- 레이아웃: ...
- 타이포그래피: ...
- 색상: ...
- 상호작용: ...

### 트레이드오프
- 강점: ...
- 약점: ...

### 적합한 대상
- 이 변형이 실제로 유용한 사용자 유형이나 사용 사례
```

### 5. 맞대결 (Head-to-head)

모든 변형이 구축된 후, 비교로 제시하세요. 단순히 나열만 하지 말고 **의견을 내세요**:

```markdown
## 홈 화면에 대한 3가지 시각

| 차원 | Calm editorial (차분한 편집형) | Utilitarian dense (실용적 고밀도형) | Playful split (장난스러운 분할형) |
|-----------|----------------|-------------------|---------------|
| 밀도 | 낮음 | 높음 | 중간 |
| 주요 액션 가시성 | 낮음 | 높음 | 중간 |
| 훑어보기 편의성 | 높음 | 중간 | 낮음 |
| 느낌 | 차분함, 신뢰감 | 날카로움, 도구 같음 | 매력적, 활기참 |

**제 의견:** 파워 유저를 위해서는 실용적 고밀도형, 콘텐츠 중심의 청중을 위해서는 차분한 편집형이 좋습니다. 장난스러운 분할형은 가장 약합니다 — 두 가지를 모두 하려다 이도 저도 안 되었습니다.
```

사용자가 승자를 선택하게 하거나, 두 가지를 혼합하게 하거나, 다음 라운드를 요청하게 하세요.

## 테마 적용 (프로젝트에 시각적 정체성이 있는 경우)

사용자에게 기존 테마(색상, 폰트, 토큰)가 있는 경우, 공유 토큰을 `sketches/themes/tokens.css`에 넣고 각 변형에서 `@import` 하세요. 토큰은 최소한으로 유지하세요:

```css
/* sketches/themes/tokens.css */
:root {
  --color-bg: #fafafa;
  --color-fg: #1a1a1a;
  --color-accent: #0066ff;
  --color-muted: #666;
  --radius: 8px;
  --font-display: "Inter", sans-serif;
  --font-body: -apple-system, BlinkMacSystemFont, sans-serif;
}
```

일회용 스케치를 너무 세분화하지 마세요 — 보통 세 가지 색상과 한 가지 폰트면 충분합니다.

## 상호작용 기준

스케치는 사용자가 다음을 할 수 있을 때 충분히 인터랙티브합니다:

1. **기본 액션을 클릭**하면 눈에 띄는 현상(상태 변경, 모달, 토스트, 탐색 흉내)이 일어납니다.
2. **하나의 의미 있는 상태 전환**을 봅니다 (목록 필터링, 모드 전환, 패널 열기/닫기).
3. **인지 가능한 어포던스(affordances)**에 호버 효과가 있습니다 (버튼, 행, 탭).

이보다 많으면 일회용 목업에 과도한 엔지니어링을 하는 것이고, 이보다 적으면 스크린샷에 불과합니다.

## 프론티어 모드 (다음에 스케치할 것 선택)

스케치가 이미 존재하고 사용자가 "다음에는 뭘 스케치해야 할까?"라고 묻는 경우:

- **일관성 격차** — 서로 다른 스케치에서 승리한 두 변형이 독립적인 선택을 했지만 아직 함께 구성되지 않은 경우
- **스케치되지 않은 화면** — 언급은 되었으나 탐색되지 않은 화면
- **상태 커버리지** — 정상 경로(happy path)는 스케치되었지만, 비어있음 / 로딩 / 오류 / 항목 1000개일 때는 스케치되지 않은 경우
- **반응형 격차** — 하나의 뷰포트에서는 검증되었지만 모바일 / 울트라와이드에서도 유지되는지
- **상호작용 패턴** — 정적 레이아웃은 존재하지만 트랜지션, 드래그, 스크롤 동작이 없는 경우

이름이 지정된 2-4개의 후보를 제안하고 사용자가 선택하게 하세요.

## 출력

- 저장소 루트에 `sketches/` (또는 사용자가 GSD 규칙을 사용하는 경우 `.planning/sketches/`) 생성
- 변형당 하나의 하위 디렉토리: `NNN-stance-name/index.html` + `README.md`
- 사용자에게 여는 방법을 알려주세요: macOS의 경우 `open sketches/001-calm-editorial/index.html`, Linux의 경우 `xdg-open`, Windows의 경우 `start`
- 변형은 일회성으로 유지하세요 — 보존할 필요가 느껴지는 스케치는 자산으로 큐레이팅할 것이 아니라 실제 프로젝트 코드로 승격시켜야 합니다.

**하나의 변형에 대한 일반적인 도구 시퀀스:**

```
terminal("mkdir -p sketches/001-calm-editorial")
write_file("sketches/001-calm-editorial/index.html", "<!doctype html>...")
write_file("sketches/001-calm-editorial/README.md", "## Variant: Calm editorial\n...")
browser_navigate(url="file://$(pwd)/sketches/001-calm-editorial/index.html")
browser_vision(question="어때 보이나요? 눈에 띄는 레이아웃 문제가 있나요?")
```

각 변형에 대해 반복한 다음 비교 테이블을 제시합니다.

## 출처

GSD(Get Shit Done) 프로젝트의 `/gsd-sketch` 워크플로우에서 각색함 — MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)). 전체 GSD 시스템은 지속적인 스케치 상태, 테마/변형 패턴 참조 및 일관성 감사 워크플로우를 제공합니다; `npx get-shit-done-cc --hermes --global`로 설치하세요.
