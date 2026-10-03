---
title: "인기 웹 디자인 — HTML/CSS로 구현한 54가지 실제 디자인 시스템 (Stripe, Linear, Vercel)"
sidebar_label: "인기 웹 디자인"
description: "HTML/CSS로 구현한 54가지 실제 디자인 시스템 (Stripe, Linear, Vercel)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 인기 웹 디자인

HTML/CSS로 바로 사용할 수 있는 54가지 실제 디자인 시스템 (Stripe, Linear, Vercel).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공 번들 |
| 경로 | `skills/creative/popular-web-designs` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent + Teknium (VoltAgent/awesome-design-md에서 디자인 시스템 제공) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지침으로 보게 되는 내용입니다.
:::

# 인기 웹 디자인

HTML/CSS를 생성할 때 바로 사용할 수 있는 54가지 실제 웹사이트 디자인 시스템입니다. 각 템플릿은
사이트의 전체 시각 언어인 색상 팔레트, 타이포그래피 계층, 컴포넌트 스타일, 간격
시스템, 그림자, 반응형 동작, 정확한 CSS 값이 포함된 실용적인 에이전트 프롬프트를 담고 있습니다.

## 관련 디자인 스킬

- **`claude-design`** — 디자인 *프로세스와 감각*에 사용합니다 (브리프 범위 설정,
  변형안 제작, 로컬 HTML 산출물 검증, AI 디자인의 상투성 방지).
  잘 설계된 페이지를 알려진 브랜드 스타일로 만들고 싶을 때 이 스킬과 함께 사용하세요:
  `claude-design`이 워크플로를 주도하고, 이 스킬이 시각적 어휘를 제공합니다.
- **`design-md`** — 결과물이 렌더링된 산출물이 아니라 정식 DESIGN.md 토큰 사양
  파일일 때 사용합니다.

## 사용 방법

1. 아래 카탈로그에서 디자인을 선택합니다.
2. 로드합니다: `skill_view(name="popular-web-designs", file_path="templates/<site>.md")`
3. HTML을 생성할 때 디자인 토큰과 컴포넌트 사양을 사용합니다.
4. 결과를 cloudflared 터널로 제공하려면 `generative-widgets` 스킬과 함께 사용합니다.

각 템플릿의 상단에는 다음 내용이 담긴 **Hermes 구현 참고 사항** 블록이 있습니다:
- CDN 글꼴 대체안과 Google Fonts `<link>` 태그 (바로 붙여 넣어 사용 가능)
- 기본 및 모노스페이스 글꼴을 위한 CSS `font-family` 스택
- HTML 생성에는 `write_file`을, 검증에는 `browser_vision`을 사용하라는 알림

## HTML 생성 패턴

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
  <!-- Paste the Google Fonts <link> from the template's Hermes notes -->
  <link href="https://fonts.googleapis.com/css2?family=..." rel="stylesheet">
  <style>
    /* Apply the template's color palette as CSS custom properties */
    :root {
      --color-bg: #ffffff;
      --color-text: #171717;
      --color-accent: #533afd;
      /* ... more from template Section 2 */
    }
    /* Apply typography from template Section 3 */
    body {
      font-family: 'Inter', system-ui, sans-serif;
      color: var(--color-text);
      background: var(--color-bg);
    }
    /* Apply component styles from template Section 4 */
    /* Apply layout from template Section 5 */
    /* Apply shadows from template Section 6 */
  </style>
</head>
<body>
  <!-- Build using component specs from the template -->
</body>
</html>
```

`write_file`로 파일을 작성하고, `generative-widgets` 워크플로 (cloudflared 터널)로 제공한 뒤,
`browser_vision`으로 결과를 검증해 시각적 정확성을 확인하세요.

## 글꼴 대체 참고표

대부분의 사이트는 CDN에서 사용할 수 없는 독점 글꼴을 사용합니다. 각 템플릿은 디자인의 특성을
유지하는 Google Fonts 대체 글꼴을 지정합니다. 자주 사용하는 매핑은 다음과 같습니다:

| 독점 글꼴 | CDN 대체 글꼴 | 특징 |
|---|---|---|
| Geist / Geist Sans | Geist (Google Fonts) | 기하학적, 압축된 자간 |
| Geist Mono | Geist Mono (Google Fonts) | 깔끔한 고정폭, 합자 |
| sohne-var (Stripe) | Source Sans 3 | 가벼운 우아함 |
| Berkeley Mono | JetBrains Mono | 기술적인 고정폭 |
| Airbnb Cereal VF | DM Sans | 둥글고 친근한 기하학적 형태 |
| Circular (Spotify) | DM Sans | 기하학적이고 따뜻함 |
| figmaSans | Inter | 깔끔한 휴머니스트 스타일 |
| Pin Sans (Pinterest) | DM Sans | 친근하고 둥근 형태 |
| NVIDIA-EMEA | Inter (또는 시스템 Arial) | 산업적이고 깔끔함 |
| CoinbaseDisplay/Sans | DM Sans | 기하학적이고 신뢰감 있음 |
| UberMove | DM Sans | 굵고 촘촘함 |
| HashiCorp Sans | Inter | 기업용, 중립적 |
| waldenburgNormal (Sanity) | Space Grotesk | 기하학적이고 약간 좁은 형태 |
| IBM Plex Sans/Mono | IBM Plex Sans/Mono | Google Fonts에서 사용 가능 |
| Rubik (Sentry) | Rubik | Google Fonts에서 사용 가능 |

템플릿의 CDN 글꼴이 원본과 일치하는 경우(Inter, IBM Plex, Rubik, Geist) 대체로 인한 손실은 없습니다. 대체 글꼴을 사용하는 경우(Circular 대신 DM Sans, sohne-var 대신 Source Sans 3)에는 해당 글꼴 자체보다 디자인의 시각적 정체성에 더 큰 영향을 미치는 템플릿의 굵기, 크기 및 자간 값을 세심하게 따르세요.

## 디자인 카탈로그

### AI 및 머신러닝

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `claude.md` | Anthropic Claude | 따뜻한 테라코타 강조색, 깔끔한 에디토리얼 레이아웃 |
| `cohere.md` | Cohere | 생생한 그라데이션, 데이터가 풍부한 대시보드 미학 |
| `elevenlabs.md` | ElevenLabs | 어두운 시네마틱 UI, 오디오 파형 미학 |
| `minimax.md` | Minimax | 네온 강조색을 사용한 대담한 어두운 인터페이스 |
| `mistral.ai.md` | Mistral AI | 프랑스풍 엔지니어링 미니멀리즘, 보라색 계열 |
| `ollama.md` | Ollama | 터미널 우선, 단색의 간결함 |
| `opencode.ai.md` | OpenCode AI | 개발자 중심의 어두운 테마, 전체 고정폭 글꼴 |
| `replicate.md` | Replicate | 깔끔한 흰색 캔버스, 코드 중심 |
| `runwayml.md` | RunwayML | 시네마틱 다크 UI, 미디어 중심 레이아웃 |
| `together.ai.md` | Together AI | 기술적이고 청사진 같은 디자인 |
| `voltagent.md` | VoltAgent | 칠흑 같은 캔버스, 에메랄드 강조색, 터미널 기반 |
| `x.ai.md` | xAI | 강렬한 단색, 미래적인 미니멀리즘, 전체 고정폭 글꼴 |

### 개발자 도구 및 플랫폼

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `cursor.md` | Cursor | 세련된 어두운 인터페이스, 그라데이션 강조색 |
| `expo.md` | Expo | 어두운 테마, 촘촘한 자간, 코드 중심 |
| `linear.app.md` | Linear | 초미니멀 다크 모드, 정밀함, 보라색 강조색 |
| `lovable.md` | Lovable | 유쾌한 그라데이션, 친근한 개발자 미학 |
| `mintlify.md` | Mintlify | 깔끔하고 녹색 강조색을 사용하며 읽기에 최적화됨 |
| `posthog.md` | PostHog | 유쾌한 브랜딩, 개발자 친화적인 어두운 UI |
| `raycast.md` | Raycast | 세련된 어두운 크롬, 생생한 그라데이션 강조색 |
| `resend.md` | Resend | 미니멀 다크 테마, 고정폭 강조 |
| `sentry.md` | Sentry | 어두운 대시보드, 데이터가 빽빽한 분홍-보라색 강조색 |
| `supabase.md` | Supabase | 어두운 에메랄드 테마, 코드 우선 개발자 도구 |
| `superhuman.md` | Superhuman | 고급스러운 어두운 UI, 키보드 우선, 보라색 광택 |
| `vercel.md` | Vercel | 흑백의 정밀함, Geist 글꼴 시스템 |
| `warp.md` | Warp | 어두운 IDE 스타일 인터페이스, 블록 기반 명령 UI |
| `zapier.md` | Zapier | 따뜻한 오렌지색, 친근한 일러스트 중심 |

### 인프라 및 클라우드

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `clickhouse.md` | ClickHouse | 노란색 강조색, 기술 문서 스타일 |
| `composio.md` | Composio | 다채로운 통합 아이콘을 사용한 현대적인 어두운 디자인 |
| `hashicorp.md` | HashiCorp | 기업용으로 깔끔한 흑백 디자인 |
| `mongodb.md` | MongoDB | 녹색 잎 브랜딩, 개발자 문서 중심 |
| `sanity.md` | Sanity | 빨간색 강조색, 콘텐츠 중심 에디토리얼 레이아웃 |
| `stripe.md` | Stripe | 시그니처 보라색 그라데이션, 300 굵기의 우아함 |

### 디자인 및 생산성

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `airtable.md` | Airtable | 다채롭고 친근한 구조화 데이터 미학 |
| `cal.md` | Cal.com | 깔끔하고 중립적인 UI, 개발자 지향의 간결함 |
| `clay.md` | Clay | 유기적인 형태, 부드러운 그라데이션, 아트 디렉션 레이아웃 |
| `figma.md` | Figma | 생생한 다색 구성, 유쾌하면서도 전문적 |
| `framer.md` | Framer | 대담한 검정과 파랑, 모션 우선, 디자인 중심 |
| `intercom.md` | Intercom | 친근한 파란색 팔레트, 대화형 UI 패턴 |
| `miro.md` | Miro | 밝은 노란색 강조색, 무한 캔버스 미학 |
| `notion.md` | Notion | 따뜻한 미니멀리즘, 세리프 제목, 부드러운 표면 |
| `pinterest.md` | Pinterest | 빨간색 강조색, 메이슨리 그리드, 이미지 중심 |
| `webflow.md` | Webflow | 파란색 강조색, 세련된 마케팅 사이트 미학 |

### 핀테크 및 암호화폐

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `coinbase.md` | Coinbase | 깔끔한 파란색 정체성, 신뢰 중심의 기관 느낌 |
| `kraken.md` | Kraken | 보라색 강조색의 어두운 UI, 데이터가 빽빽한 대시보드 |
| `revolut.md` | Revolut | 세련된 어두운 인터페이스, 그라데이션 카드, 핀테크의 정밀함 |
| `wise.md` | Wise | 밝은 녹색 강조색, 친근하고 명확함 |

### 기업 및 소비자

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `airbnb.md` | Airbnb | 따뜻한 코럴 강조색, 사진 중심, 둥근 UI |
| `apple.md` | Apple | 고급스러운 여백, SF Pro, 시네마틱 이미지 |
| `bmw.md` | BMW | 어두운 고급 표면, 정밀한 엔지니어링 미학 |
| `ibm.md` | IBM | Carbon 디자인 시스템, 구조화된 파란색 팔레트 |
| `nvidia.md` | NVIDIA | 녹색과 검정의 에너지, 기술적이고 강력한 미학 |
| `spacex.md` | SpaceX | 강렬한 흑백, 전체 화면 이미지, 미래적 |
| `spotify.md` | Spotify | 어두운 배경의 생생한 녹색, 대담한 글꼴, 앨범 아트 중심 |
| `uber.md` | Uber | 대담한 흑백, 촘촘한 글꼴, 도시적인 에너지 |

## 디자인 선택

콘텐츠에 맞춰 디자인을 선택하세요:

- **개발자 도구 / 대시보드:** Linear, Vercel, Supabase, Raycast, Sentry
- **문서 / 콘텐츠 사이트:** Mintlify, Notion, Sanity, MongoDB
- **마케팅 / 랜딩 페이지:** Stripe, Framer, Apple, SpaceX
- **다크 모드 UI:** Linear, Cursor, ElevenLabs, Warp, Superhuman
- **밝고 깔끔한 UI:** Vercel, Stripe, Notion, Cal.com, Replicate
- **유쾌하고 친근한 스타일:** PostHog, Figma, Lovable, Zapier, Miro
- **고급 / 럭셔리:** Apple, BMW, Stripe, Superhuman, Revolut
- **데이터가 빽빽한 대시보드:** Sentry, Kraken, Cohere, ClickHouse
- **고정폭 글꼴 / 터미널 미학:** Ollama, OpenCode, x.ai, VoltAgent
