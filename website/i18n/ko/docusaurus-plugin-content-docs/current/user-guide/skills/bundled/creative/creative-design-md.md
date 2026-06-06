---
title: "Design Md — Google의 DESIGN.md 작성/유효성 검사/내보내기"
sidebar_label: "Design Md"
description: "Google의 DESIGN.md 작성/유효성 검사/내보내기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Design Md

Google의 DESIGN.md 토큰 사양 파일 작성/유효성 검사/내보내기.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/creative/design-md` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `design`, `design-system`, `tokens`, `ui`, `accessibility`, `wcag`, `tailwind`, `dtcg`, `google` |
| Related skills | [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 것입니다.
:::

# DESIGN.md 스킬

DESIGN.md는 코딩 에이전트에게 시각적 정체성(visual identity)을 설명하기 위한 Google의 오픈 스펙(Apache-2.0, `google-labs-code/design.md`)입니다. 이 하나의 파일은 다음을 결합합니다:

- **YAML 프런트매터 (front matter)** — 기계가 읽을 수 있는 디자인 토큰 (표준 값)
- **Markdown 본문 (body)** — 표준적인 섹션으로 구성된 사람이 읽을 수 있는 근거(rationale)

토큰은 정확한 값을 제공합니다. 산문(Prose)은 에이전트에게 *왜* 그 값들이 존재하고 어떻게 적용해야 하는지 알려줍니다. CLI (`npx @google/design.md`)는 구조 + WCAG 대비율 린트 검사(lint), 회귀 방지를 위한 버전 비교(diff), Tailwind 또는 W3C DTCG JSON으로의 내보내기를 수행합니다.

## 이 스킬을 사용할 때

- 사용자가 DESIGN.md 파일, 디자인 토큰 또는 디자인 시스템 스펙을 요청할 때
- 사용자가 여러 프로젝트나 도구에서 일관된 UI/브랜드를 원할 때
- 사용자가 기존 DESIGN.md를 붙여넣고 이를 린트(lint), diff, 내보내기 또는 확장하도록 요청할 때
- 사용자가 에이전트가 소비할 수 있는 형식으로 스타일 가이드를 변환하도록 요청할 때
- 사용자가 색상 팔레트에 대한 대비율 / WCAG 접근성 검증을 원할 때

순수한 시각적 영감이나 레이아웃 예시가 필요한 경우에는 대신 `popular-web-designs`를 사용하세요. 처음부터 일회성 HTML 아티팩트(프로토타입, 프레젠테이션, 랜딩 페이지, 컴포넌트 랩)를 설계할 때 *프로세스와 취향*을 원한다면 `claude-design`을 사용하세요. 이 스킬은 *공식 스펙 파일* 그 자체를 위한 것입니다.

## 파일 구조 (File anatomy)

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

## 토큰 유형 (Token types)

| 유형 (Type) | 형식 (Format) | 예시 (Example) |
|------|--------|---------|
| 색상 (Color) | `#` + hex (sRGB) | `"#1A1C1E"` |
| 치수 (Dimension) | 숫자 + 단위 (`px`, `em`, `rem`) | `48px`, `-0.02em` |
| 토큰 참조 (Token ref) | `{path.to.token}` | `{colors.primary}` |
| 타이포그래피 | `fontFamily`, `fontSize`, `fontWeight`, `lineHeight`, `letterSpacing`, `fontFeature`, `fontVariation`이 포함된 객체 | 위 참조 |

컴포넌트 속성 허용 목록(whitelist): `backgroundColor`, `textColor`, `typography`, `rounded`, `padding`, `size`, `height`, `width`. 상태 변형(hover, active, pressed 등)은 중첩되지 않고 관련된 키 이름(`button-primary-hover`)을 가진 **별도의 컴포넌트 항목**입니다.

## 표준 섹션 순서 (Canonical section order)

섹션은 선택 사항이지만, 존재하는 경우 반드시 이 순서대로 나타나야 합니다. 중복된 제목은 파일을 거부합니다.

1. Overview (별칭: Brand & Style)
2. Colors
3. Typography
4. Layout (별칭: Layout & Spacing)
5. Elevation & Depth (별칭: Elevation)
6. Shapes
7. Components
8. Do's and Don'ts

알 수 없는 섹션은 오류가 발생하지 않고 보존됩니다. 알 수 없는 토큰 이름은 값의 유형이 유효하다면 허용됩니다. 알 수 없는 컴포넌트 속성은 경고(warning)를 생성합니다.

## 워크플로우: 새로운 DESIGN.md 작성

1. **사용자에게 질문**하거나(또는 추론하여) 브랜드 톤, 포인트 색상(accent color) 및 타이포그래피 방향을 파악합니다. 사용자가 사이트, 이미지 또는 분위기를 제공한 경우 위의 토큰 형태로 변환합니다.
2. `write_file`을 사용하여 사용자의 프로젝트 루트에 **`DESIGN.md`를 작성**합니다. 항상 `name:`과 `colors:`를 포함하고; 다른 섹션은 선택 사항이지만 권장됩니다.
3. `components:` 섹션에서는 16진수 값을 다시 입력하는 대신 **토큰 참조** (`{colors.primary}`)를 사용하세요. 이것은 팔레트를 단일 출처(single-source)로 유지합니다.
4. **린트 검사**를 수행합니다(아래 참조). 반환하기 전에 끊어진 참조나 WCAG 실패를 수정합니다.
5. **사용자에게 기존 프로젝트가 있는 경우**, 파일 옆에 Tailwind 또는 DTCG 내보내기도 함께 작성합니다 (`tailwind.theme.json`, `tokens.json`).

## 워크플로우: 린트 / diff / 내보내기

CLI는 `@google/design.md` (Node)입니다. `npx`를 사용하세요 — 전역 설치(global install)가 필요하지 않습니다.

```bash
# 구조 + 토큰 참조 + WCAG 대비율 검증
npx -y @google/design.md lint DESIGN.md

# 두 버전을 비교하고, 퇴행(regression)이 있으면 실패(exit 1) 처리
npx -y @google/design.md diff DESIGN.md DESIGN-v2.md

# Tailwind theme JSON으로 내보내기
npx -y @google/design.md export --format tailwind DESIGN.md > tailwind.theme.json

# W3C DTCG (Design Tokens Format Module) JSON으로 내보내기
npx -y @google/design.md export --format dtcg DESIGN.md > tokens.json

# 스펙 자체 출력 — 에이전트 프롬프트에 주입할 때 유용함
npx -y @google/design.md spec --rules-only --format json
```

모든 명령어는 표준 입력(stdin)에 대해 `-`를 허용합니다. `lint`는 오류 시 종료 코드 1을 반환합니다. 결과를 구조적으로 보고해야 하는 경우 `--format json` 플래그를 사용하고 출력을 구문 분석하세요.

### 린트 규칙 참조 (7가지 규칙이 잡아내는 내용)

- `broken-ref` (오류) — `{colors.missing}`이 존재하지 않는 토큰을 가리킴
- `duplicate-section` (오류) — 동일한 `## Heading`이 두 번 나타남
- `invalid-color`, `invalid-dimension`, `invalid-typography` (오류)
- `wcag-contrast` (경고/정보) — 컴포넌트의 `textColor`와 `backgroundColor` 비율을 WCAG AA (4.5:1) 및 AAA (7:1) 기준과 비교
- `unknown-component-property` (경고) — 위의 허용 목록을 벗어남

사용자가 접근성에 신경을 쓸 때 요약에서 이것을 명시적으로 언급하세요 — WCAG 발견 사항은 이 CLI를 사용하는 가장 중요한 이유입니다.

## 주의 사항 (Pitfalls)

- **컴포넌트 변형(variants)을 중첩하지 마세요.** `button-primary.hover`는 잘못된 것입니다. `button-primary-hover`와 같이 형제(sibling) 키로 지정하는 것이 올바릅니다.
- **Hex 색상은 반드시 따옴표로 묶인 문자열이어야 합니다.** 그렇지 않으면 YAML이 `#`에서 막히거나 `#1A1C1E`와 같은 값을 이상하게 잘라냅니다.
- **음수 치수도 따옴표가 필요합니다.** `letterSpacing: -0.02em`은 YAML 플로우(flow)로 파싱됩니다 — `letterSpacing: "-0.02em"`으로 작성하세요.
- **섹션 순서는 강제됩니다.** 사용자가 임의의 순서로 산문(prose)을 제공하면 저장하기 전에 정해진 목록과 일치하도록 재정렬하세요.
- **`version: alpha`가 현재 스펙 버전입니다** (2026년 4월 기준). 스펙은 알파 단계로 표시되어 있습니다 — 파괴적인 변경(breaking changes)을 주의하세요.
- **토큰 참조는 점 표기법(dotted path)으로 해석됩니다.** `{colors.primary}`는 작동하지만; `{primary}`는 작동하지 않습니다.

## 스펙 출처 (Spec source of truth)

- 저장소(Repo): https://github.com/google-labs-code/design.md (Apache-2.0)
- CLI: npm의 `@google/design.md`
- 생성된 DESIGN.md 파일의 라이선스: 사용자 프로젝트의 라이선스를 따릅니다. 스펙 자체는 Apache-2.0입니다.
