---
title: "스케치 — 버리는 HTML 목업: 비교할 2~3가지 디자인 변형"
sidebar_label: "스케치"
description: "비교할 2~3가지 디자인 변형을 만드는 버리는 HTML 목업"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 스케치

버리는 HTML 목업으로 비교할 2~3가지 디자인 변형을 만듭니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본 설치) |
| 경로 | `skills/creative/sketch` |
| 버전 | `1.0.1` |
| 작성자 | Hermes Agent (gsd-build/get-shit-done에서 각색) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `sketch`, `mockup`, `design`, `ui`, `prototype`, `html`, `variants`, `exploration`, `wireframe`, `comparison` |
| 관련 스킬 | [`spike`](/docs/user-guide/skills/bundled/software-development/software-development-spike), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 불러오는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# 스케치

사용자가 **하나를 선택하기 전에 디자인 방향을 보고 싶어 할 때** — 일회용 HTML 목업으로 UI/UX 아이디어를 탐색할 때 이 스킬을 사용합니다. 목적은 사용자가 시각적 방향을 나란히 비교할 수 있도록 상호작용 가능한 변형 2~3개를 만드는 것이며, 출시 가능한 코드를 만드는 것이 아닙니다.

사용자가 다음과 같이 말할 때 로드합니다: "이 화면을 스케치해 줘", "X가 어떻게 보일 수 있는지 보여 줘", "레이아웃 A와 B를 비교해 줘", "이 UI를 2~3가지로 만들어 줘", "몇 가지 변형을 보여 줘", "만들기 전에 목업을 만들어 줘".

## 사용하지 말아야 할 때

- 사용자가 프로덕션 컴포넌트를 원함 — `claude-design`을 사용하거나 제대로 구현합니다
- 사용자가 다듬어진 단일 HTML 결과물(랜딩 페이지, 데크)을 원함 — `claude-design`
- 사용자가 다이어그램을 원함 — `excalidraw`, `architecture-diagram`
- 디자인이 이미 확정됨 — 바로 구현합니다

## 사용자가 전체 GSD 시스템을 설치한 경우

`gsd-sketch`가 형제 스킬로 표시된다면(`npx get-shit-done-cc --hermes`로 설치), 더 완전한 워크플로를 위해 **`gsd-sketch`**를 사용할 수 있습니다. 여기에는 MANIFEST가 포함된 영속적인 `.planning/sketches/`, 프론티어 모드 분석, 과거 스케치 전반의 일관성 감사, 나머지 GSD와의 통합이 있습니다. 이 스킬은 가벼운 독립형 버전으로, 상태 관리 없이 일회성 스케치를 만들 수 있습니다.

> **참고:** 업스트림 GSD 프로젝트([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)는 GitHub에서 보관 처리되었으며 더 이상 유지 관리되지 않습니다. npm 패키지(`get-shit-done-cc`)는 여전히 설치되지만, 보관된 커뮤니티 프로젝트로 취급하세요 — 이 독립형 `sketch` 스킬이 유지 관리되는 경로이며 추가로 필요한 것은 없습니다.

## 핵심 방법

```
intake  →  variants  →  head-to-head  →  pick winner (or iterate)
```

### 1. 인테이크 (사용자가 이미 충분히 제공했다면 건너뜁니다)

변형을 만들기 전에 세 가지를 확인합니다 — 한 번에 모두가 아니라 한 번에 하나씩 질문합니다:

1. **느낌.** "어떤 느낌이어야 하나요? 형용사, 감정, 분위기를 말해 주세요." — *"차분하고, 편집 디자인 같고, Linear 같은 느낌"*이 *"미니멀하게"*보다 더 많은 정보를 줍니다.
2. **참고 대상.** "상상하는 느낌을 담은 앱, 사이트 또는 제품은 무엇인가요?" — 추상적인 설명보다 실제 참고 대상이 낫습니다.
3. **핵심 행동.** "사용자가 이 화면에서 하는 가장 중요한 단일 행동은 무엇인가요?" — 모든 변형은 이를 잘 지원해야 합니다. 그렇지 않으면 그저 장식일 뿐입니다.

다음 질문을 하기 전에 각 답변을 간단히 되짚습니다. 사용자가 처음부터 세 가지를 모두 제공했다면 바로 변형으로 넘어갑니다.

### 2. 변형 (2~3개, 절대 1개가 아니며, 4개 이상은 드뭅니다)

**2~3개의 변형**을 한 번에 만듭니다. 각 변형은 완전하고 독립적인 HTML 파일입니다. 변형을 설명하지 말고 만드세요. 목적은 비교입니다.

각 변형은 픽셀 값이 아니라 **서로 다른 디자인 입장**을 취해야 합니다. 좋은 변형 축은 다음과 같습니다:

- **밀도:** 조밀함 / 여유로움 / 매우 조밀함(서로 대비되는 두 극을 선택)
- **강조:** 콘텐츠 우선 / 행동 우선 / 도구 우선
- **미학:** 편집 디자인 / 실용적 / 유쾌함
- **레이아웃:** 단일 열 / 사이드바 / 분할 창
- **기반:** 카드 기반 / 콘텐츠만 / 문서 스타일

축 하나를 고르고 양 끝으로 벌립니다. 강조 색상만 다른 두 변형은 노력 낭비입니다 — 사용자가 구분할 수 없습니다.

**변형 이름 지정:** 숫자가 아니라 입장을 설명합니다.

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

각 변형은 **하나의 자체 완결형 HTML 파일**입니다:

- 인라인 `<style>` — 빌드 단계 없음, 외부 CSS 없음
- 시스템 글꼴 또는 `<link>`를 통한 Google 글꼴 하나
- Tailwind는 CDN을 통해 사용할 수 있습니다(`<script src="https://cdn.tailwindcss.com"></script>`)
- 현실적인 가짜 콘텐츠 — 실제 문장과 실제 이름을 사용하며 "Lorem ipsum"은 사용하지 않음
- **상호작용 가능:** 링크는 클릭 가능하고, 호버는 실제로 작동하며, 최소 하나의 상태 전환(열기/닫기, 필터, 토글)이 있어야 합니다. 정지된 정적 이미지는 서툴게 애니메이션한 것보다 나쁜 스파이크입니다.

브라우저에서 엽니다. 망가져 보이면 사용자에게 보여 주기 전에 수정합니다.

**변형을 시각적으로 검증 — Hermes의 브라우저 도구를 사용합니다.** HTML을 작성하고 렌더링되기를 바라기만 하지 말고, 각 변형을 로드해 직접 확인합니다:

```
browser_navigate(url="file:///absolute/path/to/sketches/001-calm-editorial/index.html")
browser_vision(question="Does this layout look clean and readable? Any visible bugs (overlapping text, unstyled elements, broken images)?")
```

`browser_vision`은 페이지에 실제로 표시되는 내용을 AI가 설명하고 스크린샷 경로도 반환합니다 — 순수한 소스 검사로는 발견하기 어려운 레이아웃 버그(예: 글꼴 가져오기가 조용히 실패하거나 flex 컨테이너가 축소되는 문제)를 찾아냅니다. 각 변형이 제대로 보일 때까지 수정하고 다시 탐색합니다.

**빠른 시작을 위한 기본 CSS 리셋 + 시스템 글꼴 스택:**

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
## Variant: {stance name}

### Design stance
One sentence on the principle driving this variant.

### Key choices
- Layout: ...
- Typography: ...
- Color: ...
- Interaction: ...

### Trade-offs
- Strong at: ...
- Weak at: ...

### Best for
- The kind of user or use case this variant actually serves
```

### 5. 정면 비교

모든 변형을 만든 후 비교 대상으로 제시합니다. 단순히 나열하지 말고 **관점을 제시합니다**:

```markdown
## Three takes on the home screen

| Dimension | Calm editorial | Utilitarian dense | Playful split |
|-----------|----------------|-------------------|---------------|
| Density   | Low            | High              | Medium        |
| Primary action visibility | Low | High | Medium |
| Scan-ability | High | Medium | Low |
| Feel | Calm, trusted | Sharp, tool-like | Inviting, energetic |

**My take:** Utilitarian dense for power users, calm editorial for content-forward audiences. Playful split is weakest — tries to do both and commits to neither.
```

사용자가 하나를 선택하게 하거나, 두 가지를 결합해 하이브리드로 만들게 하거나, 다음 라운드를 요청하게 합니다.

## 테마 설정 (프로젝트에 시각적 아이덴티티가 있는 경우)

사용자에게 기존 테마(색상, 글꼴, 토큰)가 있다면 공유 토큰을 `sketches/themes/tokens.css`에 넣고 각 변형에서 `@import`합니다. 토큰은 최소한으로 유지합니다:

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

버리는 스케치를 과도하게 토큰화하지 마세요 — 보통 색상 세 가지와 글꼴 하나면 충분합니다.

## 상호작용 기준

사용자가 다음을 할 수 있으면 스케치는 충분히 상호작용 가능합니다:

1. **주요 행동을 클릭**하면 눈에 보이는 일이 발생함(상태 변경, 모달, 토스트, 탐색하는 듯한 동작)
2. **의미 있는 상태 전환 하나를 확인**할 수 있음(목록 필터링, 모드 토글, 패널 열기/닫기)
3. **호버로 인식 가능한 어포던스를 확인**할 수 있음(버튼, 행, 탭)

그 이상은 버리는 스케치를 과도하게 설계하는 것입니다. 그 이하는 스크린샷입니다.

## 프론티어 모드 (다음에 무엇을 스케치할지 선택)

이미 스케치가 있고 사용자가 "다음에는 무엇을 스케치해야 해?"라고 말한다면:

- **일관성의 빈틈** — 서로 다른 스케치에서 선택된 두 변형이 아직 함께 구성되지 않은 독립적인 선택을 했는지
- **스케치하지 않은 화면** — 언급되었지만 아직 탐색하지 않은 화면
- **상태 범위** — 정상 경로는 스케치했지만 비어 있음 / 로딩 / 오류 / 항목 1000개 상태는 다루지 않았는지
- **반응형의 빈틈** — 한 뷰포트에서 검증했지만 모바일 / 초광폭에서도 유지되는지
- **상호작용 패턴** — 정적 레이아웃은 있지만 전환, 드래그, 스크롤 동작은 없는지

이름을 붙인 후보 2~4개를 제안합니다. 사용자가 선택하게 합니다.

## 출력

- 저장소 루트에 `sketches/`(또는 사용자가 GSD 규칙을 따른다면 `.planning/sketches/`)를 만듭니다
- 변형마다 하나의 하위 디렉터리를 만듭니다: `NNN-stance-name/index.html` + `README.md`
- 여는 방법을 사용자에게 알려 줍니다: macOS에서는 `open sketches/001-calm-editorial/index.html`, Linux에서는 `xdg-open`, Windows에서는 `start`
- 변형은 버릴 수 있는 상태로 유지합니다 — 보존하고 싶다는 생각이 든 스케치는 자산으로 관리하지 말고 실제 프로젝트 코드로 승격합니다

**변형 하나에 대한 일반적인 도구 순서:**

```
terminal("mkdir -p sketches/001-calm-editorial")
write_file("sketches/001-calm-editorial/index.html", "<!doctype html>...")
write_file("sketches/001-calm-editorial/README.md", "## Variant: Calm editorial\n...")
browser_navigate(url="file://$(pwd)/sketches/001-calm-editorial/index.html")
browser_vision(question="How does this look? Any obvious layout issues?")
```

각 변형에 대해 반복한 다음 비교 표를 제시합니다.

## 저작자 표시

GSD(Get Shit Done) 프로젝트의 `/gsd-sketch` 워크플로에서 각색 — MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)). 업스트림 GSD 저장소는 현재 **보관 처리되어 유지 관리되지 않으며**, `get-shit-done-cc` npm 패키지는 여전히 설치되고(`npx get-shit-done-cc --hermes --global`) 영속적인 스케치 상태, 테마/변형 패턴 참고 자료, 일관성 감사 워크플로를 제공하지만, 보관된 프로젝트로 취급하세요.
