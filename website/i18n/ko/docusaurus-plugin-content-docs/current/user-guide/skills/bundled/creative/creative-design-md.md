---
title: "Design Md — Google의 DESIGN.md 토큰 사양 파일 작성/검증/내보내기"
sidebar_label: "Design Md"
description: "Google의 DESIGN.md 토큰 사양 파일 작성/검증/내보내기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Design Md

Google의 DESIGN.md 토큰 사양 파일을 작성하고, 검증하고, 내보냅니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들 포함(기본 설치) |
| 경로 | `skills/creative/design-md` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `design`, `design-system`, `tokens`, `ui`, `accessibility`, `wcag`, `tailwind`, `dtcg`, `google` |
| 관련 스킬 | [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보게 되는 내용입니다.
:::

# DESIGN.md 스킬

DESIGN.md는 코딩 에이전트에게 시각적 정체성을 설명하기 위한 Google의 공개 사양(`google-labs-code/design.md`, Apache-2.0)입니다. 하나의 파일에 다음 내용이 함께 들어갑니다.

- **YAML 프런트 매터** — 기계가 읽을 수 있는 디자인 토큰(규범적 값)
- **Markdown 본문** — 표준 섹션으로 구성된 사람이 읽을 수 있는 근거 설명

토큰은 정확한 값을 제공합니다. 설명문은 이러한 값이 존재하는 이유와 적용 방법을 에이전트에게 알려 줍니다. CLI(`npx @google/design.md`)는 구조와 WCAG 대비를 린트하고, 회귀를 확인하기 위해 버전을 비교하며, Tailwind 또는 W3C DTCG JSON으로 내보냅니다.

## 이 스킬을 사용하는 경우

- 사용자가 DESIGN.md 파일, 디자인 토큰 또는 디자인 시스템 사양을 요청할 때
- 사용자가 여러 프로젝트나 도구에서 일관된 UI/브랜드를 원할 때
- 사용자가 기존 DESIGN.md를 붙여 넣고 린트, 비교, 내보내기 또는 확장을 요청할 때
- 사용자가 스타일 가이드를 에이전트가 사용할 수 있는 형식으로 옮기려 할 때
- 사용자가 색상 팔레트의 대비 또는 WCAG 접근성 검증을 원할 때

순수한 시각적 영감이나 레이아웃 예시에는 `popular-web-designs`를 사용하세요. 일회성 HTML 아티팩트(프로토타입, 덱, 랜딩 페이지, 컴포넌트 랩)를 처음부터 디자인할 때의 *프로세스와 취향*에는 `claude-design`을 사용하세요. 이 스킬은 *형식화된 사양 파일 자체*를 위한 것입니다.

## 파일 구조

```md
---
version: alpha
name: Heritage
description: Architectural minimalism meets journalistic gravitas.
colors:
  primary: "#1A1C1E"
  secondary: "#6C7278"
  tertiary: "#B8422E"
  neutral: "#F7F5F2"
typography:
  h1:
    fontFamily: Public Sans
    fontSize: 3rem
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  body-md:
    fontFamily: Public Sans
    fontSize: 1rem
rounded:
  sm: 4px
  md: 8px
  lg: 16px
spacing:
  sm: 8px
  md: 16px
  lg: 24px
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: 12px
  button-primary-hover:
    backgroundColor: "{colors.primary}"
---

## Overview

Architectural Minimalism meets Journalistic Gravitas...

## Colors

- **Primary (#1A1C1E):** Deep ink for headlines and core text.
- **Tertiary (#B8422E):** "Boston Clay" — the sole driver for interaction.

## Typography

Public Sans for everything except small all-caps labels...

## Components

`button-primary` is the only high-emphasis action on a page...
```

## 토큰 유형

| 유형 | 형식 | 예시 |
|------|--------|---------|
| 색상 | 모든 CSS 색상(hex, `rgb()`, `oklch()`, 이름 지정) | `"#1A1C1E"`, `"oklch(62% 0.18 250)"` |
| 치수 | 숫자 + 단위(`px`, `em`, `rem`) | `48px`, `-0.02em` |
| 토큰 참조 | `{path.to.token}` | `{colors.primary}` |
| 타이포그래피 | `fontFamily`, `fontSize`, `fontWeight`, `lineHeight`, `letterSpacing`, `fontFeature`, `fontVariation`이 포함된 객체 | 위 예시 참조 |

컴포넌트 속성 허용 목록: `backgroundColor`, `textColor`, `typography`, `rounded`, `padding`, `size`, `height`, `width`. 변형(hover, active, pressed)은 중첩하지 말고 관련 키 이름(`button-primary-hover`)을 사용하는 **별도의 컴포넌트 항목**으로 작성합니다.

## 표준 섹션 순서

섹션은 선택 사항이지만, 작성하는 섹션은 다음 순서로 배치해야 합니다. 린터는 순서가 잘못된 섹션(`section-order`, 경고)과 중복 제목을 표시합니다. 사양에 따라 사용하는 소비자는 중복을 거부하므로 파일을 반환하기 전에 두 문제를 모두 수정하세요.

1. 개요(별칭: 브랜드 및 스타일)
2. 색상
3. 타이포그래피
4. 레이아웃(별칭: 레이아웃 및 간격)
5. 고도 및 깊이(별칭: 고도)
6. 형태
7. 컴포넌트
8. 해야 할 일과 하지 말아야 할 일

알 수 없는 섹션은 오류로 처리하지 않고 유지합니다. 값 유형이 유효하면 알 수 없는 토큰 이름도 허용됩니다. 알 수 없는 컴포넌트 속성은 경고를 생성합니다.

## 워크플로: 새 DESIGN.md 작성

1. 브랜드의 분위기, 강조 색상, 타이포그래피 방향을 사용자에게 묻거나 추론합니다. 사이트, 이미지 또는 분위기를 제공했다면 위의 토큰 형태로 옮깁니다.
2. `write_file`을 사용해 프로젝트 루트에 `DESIGN.md`를 작성합니다. 항상 `name:`과 `colors:`를 포함하고, 다른 섹션도 선택 사항이지만 포함하는 것이 좋습니다.
3. `components:` 섹션에서는 16진수 값을 다시 입력하지 말고 토큰 참조(`{colors.primary}`)를 사용합니다. 이렇게 하면 팔레트를 단일 소스로 유지할 수 있습니다.
4. 린트합니다(아래 참조). 파일을 반환하기 전에 깨진 참조나 WCAG 실패를 모두 수정합니다.
5. 기존 프로젝트가 있다면 Tailwind 또는 DTCG 내보내기도 파일 옆에 작성합니다(`tailwind.theme.json`, `tokens.json`).

## 워크플로: 린트 / 비교 / 내보내기

CLI는 `@google/design.md`(Node)입니다. 전역 설치는 필요하지 않으므로 `npx`를 사용하세요.

```bash
# Validate structure + token references + WCAG contrast
npx -y @google/design.md lint DESIGN.md

# Compare two versions, fail on regression (exit 1 = regression)
npx -y @google/design.md diff DESIGN.md DESIGN-v2.md

# Export to Tailwind v3 theme JSON (`tailwind` is a back-compat alias)
npx -y @google/design.md export --format json-tailwind DESIGN.md > tailwind.theme.json

# Export to a Tailwind v4 CSS @theme block (--color-*, --text-*, --radius-*, ...)
npx -y @google/design.md export --format css-tailwind DESIGN.md > theme.css

# Export to W3C DTCG (Design Tokens Format Module) JSON
npx -y @google/design.md export --format dtcg DESIGN.md > tokens.json

# Print the spec itself — useful when injecting into an agent prompt
npx -y @google/design.md spec --rules-only --format json
```

모든 명령은 표준 입력에 `-`을 사용할 수 있습니다. `lint`는 오류가 있으면 종료 코드 1을 반환합니다(경고만 있는 경우 종료 코드 0). `export`는 소스에 린트 결과가 있더라도 내보내기에 성공하면 종료 코드 0으로 끝납니다 — 해당 결과를 기준으로 삼으려면 별도로 `lint`를 실행하세요. 기본 출력은 JSON입니다. 결과를 구조적으로 보고해야 한다면 파싱하세요.

Windows에서는 `design.md` 바이너리 이름이 `.md` 파일 연결과 충돌할 수 있습니다(아무 동작도 하지 않거나 파일이 편집기에서 열림). 점이 없는 별칭을 사용하세요: `npx -y -p @google/design.md designmd lint DESIGN.md`.

### 린트 규칙 참고(2026년 7월 기준 CLI 0.3.0의 9개 규칙)

- `broken-ref`(오류) — `{colors.missing}`이 존재하지 않는 토큰을 가리킴
- `contrast-ratio`(경고) — WCAG AA(4.5:1) 미만인 컴포넌트 `textColor`와 `backgroundColor`의 대비
- `missing-primary`(경고) — 색상이 정의되었지만 `primary` 토큰이 없음
- `missing-typography`(경고) — 색상은 정의되었지만 타이포그래피 토큰이 없음
- `orphaned-tokens`(경고) — 어떤 컴포넌트에서도 참조되지 않는 색상 토큰
- `section-order`(경고) — 표준 순서와 다른 섹션
- `unknown-key`(경고) — 스키마 키의 오타처럼 보이는 최상위 YAML 키(`colours:` → `colors:`); 사용자 지정 확장 키는 조용히 유지됨
- `token-summary`, `missing-sections`(정보) — 개수와 누락된 선택적 섹션

사용자가 접근성을 중요하게 생각한다면 요약에서 이를 명시적으로 언급하세요 — WCAG 결과는 이 CLI를 사용하는 가장 중요한 이유입니다.

## 주의 사항

- **컴포넌트 변형을 중첩하지 마세요.** `button-primary.hover`는 잘못된 형식이고, 형제 키인 `button-primary-hover`가 올바른 형식입니다.
- **16진수 색상은 따옴표로 묶은 문자열이어야 합니다.** 그렇지 않으면 YAML이 오류를 일으키거나 `#1A1C1E` 같은 값을 이상하게 잘라낼 수 있습니다.
- **음수 치수도 따옴표로 묶어야 합니다.** `letterSpacing: -0.02em`은 YAML 흐름으로 파싱되므로 `letterSpacing: "-0.02em"`으로 작성하세요.
- **섹션 순서는 중요합니다.** 린터는 경고만 표시하지만, 사용자가 무작위 순서로 설명문을 제공하면 저장하기 전에 표준 목록에 맞게 순서를 바꾸세요 — 사양을 준수하는 소비자는 예상되는 순서를 따릅니다.
- **타이포그래피 하위 속성의 오타는 조용히 삭제됩니다.** CLI 0.3.0부터 `fontwight:` 같은 오타는 어떤 결과도 생성하지 않고 내보내기에서 값이 사라집니다 — 스키마의 하위 속성 이름(`fontFamily`, `fontSize`, `fontWeight`, `lineHeight`, `letterSpacing`, `fontFeature`, `fontVariation`)과 대조해 다시 확인하세요.
- **`version: alpha`는 현재 사양 버전입니다**(2026년 7월 기준, CLI 0.3.0). 사양은 alpha로 표시되어 있으므로 호환성이 깨지는 변경 사항을 주의하세요.
- **토큰 참조는 점으로 구분된 경로로 확인됩니다.** `{colors.primary}`는 작동하지만 `{primary}`는 작동하지 않습니다.

## 사양의 기준 출처

- 저장소: https://github.com/google-labs-code/design.md (Apache-2.0)
- CLI: npm의 `@google/design.md`
- 생성된 DESIGN.md 파일의 라이선스: 사용자의 프로젝트가 사용하는 라이선스. 사양 자체는 Apache-2.0입니다.
