---
title: "Popular Web Designs — HTML/CSS로 구현된 54개의 실제 디자인 시스템 (Stripe, Linear, Vercel 등)"
sidebar_label: "Popular Web Designs"
description: "HTML/CSS로 구현된 54개의 실제 디자인 시스템 (Stripe, Linear, Vercel 등)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Popular Web Designs

HTML/CSS로 구현된 54개의 실제 디자인 시스템 (Stripe, Linear, Vercel 등).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/popular-web-designs` |
| 버전 | `1.0.0` |
| 저자 | Hermes Agent + Teknium (VoltAgent/awesome-design-md에서 제공된 디자인 시스템) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# Popular Web Designs

HTML/CSS를 생성할 때 즉시 사용할 수 있는 54개의 실제 디자인 시스템입니다. 각 템플릿은 색상 팔레트, 타이포그래피 계층, 컴포넌트 스타일, 간격(spacing) 시스템, 그림자, 반응형 동작 및 정확한 CSS 값이 포함된 실용적인 에이전트 프롬프트 등 사이트의 전체 시각적 언어를 포착합니다.

## 관련 디자인 스킬

- **`claude-design`** — 디자인 *프로세스와 미적 감각*에 사용하세요 (요구 사항 파악, 변형 생성, 로컬 HTML 결과물 검증, AI 특유의 디자인 피하기 등). 사용자가 잘 알려진 브랜드를 본뜬 신중하게 디자인된 페이지를 원할 때 이 스킬과 함께 사용하세요: `claude-design`이 워크플로우를 주도하고, 이 스킬이 시각적 어휘를 제공합니다.
- **`design-md`** — 결과물이 렌더링된 화면이 아니라 공식적인 DESIGN.md 토큰 사양 파일일 때 사용하세요.

## 사용 방법

1. 아래 카탈로그에서 디자인을 선택합니다.
2. 불러옵니다: `skill_view(name="popular-web-designs", file_path="templates/<site>.md")`
3. HTML을 생성할 때 디자인 토큰과 컴포넌트 사양을 사용합니다.
4. `generative-widgets` 스킬과 결합하여 cloudflared 터널을 통해 결과를 제공(serve)합니다.

각 템플릿의 상단에는 다음 항목이 포함된 **Hermes Implementation Notes(Hermes 구현 참고 사항)** 블록이 있습니다:
- CDN 폰트 대체재 및 Google Fonts `<link>` 태그 (복사/붙여넣기 가능)
- 기본 및 모노스페이스를 위한 CSS font-family 스택
- HTML 생성을 위한 `write_file` 사용 및 검증을 위한 `browser_vision` 사용 안내

## HTML 생성 패턴

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
  <!-- 템플릿의 Hermes 노트에 있는 Google Fonts <link>를 붙여넣으세요 -->
  <link href="https://fonts.googleapis.com/css2?family=..." rel="stylesheet">
  <style>
    /* 템플릿의 색상 팔레트를 CSS 사용자 지정 속성(Custom Properties)으로 적용하세요 */
    :root {
      --color-bg: #ffffff;
      --color-text: #171717;
      --color-accent: #533afd;
      /* ... 템플릿 섹션 2의 추가 내용 ... */
    }
    /* 템플릿 섹션 3의 타이포그래피 적용 */
    body {
      font-family: 'Inter', system-ui, sans-serif;
      color: var(--color-text);
      background: var(--color-bg);
    }
    /* 템플릿 섹션 4의 컴포넌트 스타일 적용 */
    /* 템플릿 섹션 5의 레이아웃 적용 */
    /* 템플릿 섹션 6의 그림자 적용 */
  </style>
</head>
<body>
  <!-- 템플릿의 컴포넌트 사양을 사용하여 구축하세요 -->
</body>
</html>
```

`write_file`로 파일을 작성하고, `generative-widgets` 워크플로우(cloudflared 터널)로 제공(serve)하며, `browser_vision`으로 결과를 확인하여 시각적 정확성을 검증하세요.

## 글꼴 대체 참조

대부분의 사이트는 CDN을 통해 사용할 수 없는 독점 글꼴을 사용합니다. 각 템플릿은 디자인의 성격을 보존하는 Google Fonts 대체재와 매핑됩니다. 일반적인 매핑:

| 독점 글꼴 | CDN 대체재 | 특징 |
|---|---|---|
| Geist / Geist Sans | Geist (Google Fonts) | 기하학적, 압축된 자간(tracking) |
| Geist Mono | Geist Mono (Google Fonts) | 깔끔한 고정폭, 합자(ligatures) |
| sohne-var (Stripe) | Source Sans 3 | 가벼운 두께의 우아함 |
| Berkeley Mono | JetBrains Mono | 기술적인 고정폭 |
| Airbnb Cereal VF | DM Sans | 둥글고 친근한 기하학적 형태 |
| Circular (Spotify) | DM Sans | 기하학적이고 따뜻함 |
| figmaSans | Inter | 깔끔한 휴머니스트 스타일 |
| Pin Sans (Pinterest) | DM Sans | 친근하고 둥근 형태 |
| NVIDIA-EMEA | Inter (또는 Arial 시스템 폰트) | 산업적이고 깔끔함 |
| CoinbaseDisplay/Sans | DM Sans | 기하학적, 신뢰감 |
| UberMove | DM Sans | 굵고 촘촘함 |
| HashiCorp Sans | Inter | 엔터프라이즈 느낌, 중립적 |
| waldenburgNormal (Sanity) | Space Grotesk | 기하학적, 약간 응축됨 |
| IBM Plex Sans/Mono | IBM Plex Sans/Mono | Google Fonts에서 사용 가능 |
| Rubik (Sentry) | Rubik | Google Fonts에서 사용 가능 |

템플릿의 CDN 폰트가 원본(Inter, IBM Plex, Rubik, Geist)과 일치할 때는 대체로 인한 손실이 발생하지 않습니다. 대체재가 사용된 경우(Circular 대신 DM Sans, sohne-var 대신 Source Sans 3), 템플릿의 두께, 크기, 자간(letter-spacing) 값을 밀접하게 따르세요 — 이러한 요소들이 특정 글꼴 자체보다 더 많은 시각적 아이덴티티를 담고 있습니다.

## 디자인 카탈로그

### AI 및 머신 러닝 (AI & Machine Learning)

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `claude.md` | Anthropic Claude | 따뜻한 테라코타 악센트, 깔끔한 에디토리얼 레이아웃 |
| `cohere.md` | Cohere | 생생한 그라디언트, 데이터가 풍부한 대시보드 미학 |
| `elevenlabs.md` | ElevenLabs | 어둡고 시네마틱한 UI, 오디오 파형 미학 |
| `minimax.md` | Minimax | 네온 악센트가 있는 대담한 어두운 인터페이스 |
| `mistral.ai.md` | Mistral AI | 프랑스 엔지니어링의 미니멀리즘, 보라색 톤 |
| `ollama.md` | Ollama | 터미널 중심의 흑백 단순성 |
| `opencode.ai.md` | OpenCode AI | 개발자 중심의 어두운 테마, 완전한 고정폭 |
| `replicate.md` | Replicate | 깔끔한 흰색 캔버스, 코드 중심 |
| `runwayml.md` | RunwayML | 시네마틱한 어두운 UI, 미디어가 풍부한 레이아웃 |
| `together.ai.md` | Together AI | 기술적, 청사진(blueprint) 스타일의 디자인 |
| `voltagent.md` | VoltAgent | 완전한 블랙 캔버스, 에메랄드 악센트, 터미널 네이티브 |
| `x.ai.md` | xAI | 극명한 흑백, 미래지향적 미니멀리즘, 완전한 고정폭 |

### 개발자 도구 및 플랫폼 (Developer Tools & Platforms)

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `cursor.md` | Cursor | 매끄러운 어두운 인터페이스, 그라디언트 악센트 |
| `expo.md` | Expo | 어두운 테마, 좁은 자간, 코드 중심 |
| `linear.app.md` | Linear | 초미니멀리즘 다크 모드, 정밀함, 보라색 악센트 |
| `lovable.md` | Lovable | 장난스러운 그라디언트, 친숙한 개발자 미학 |
| `mintlify.md` | Mintlify | 깔끔한 녹색 악센트, 읽기 최적화 |
| `posthog.md` | PostHog | 장난기 있는 브랜딩, 개발자 친화적인 어두운 UI |
| `raycast.md` | Raycast | 매끄러운 어두운 크롬 느낌, 생생한 그라디언트 악센트 |
| `resend.md` | Resend | 미니멀한 어두운 테마, 고정폭 악센트 |
| `sentry.md` | Sentry | 어두운 대시보드, 높은 데이터 밀도, 핑크-보라 악센트 |
| `supabase.md` | Supabase | 짙은 에메랄드 테마, 코드 중심의 개발자 도구 |
| `superhuman.md` | Superhuman | 프리미엄 어두운 UI, 키보드 우선, 보라색 빛 |
| `vercel.md` | Vercel | 흑백의 정밀함, Geist 폰트 시스템 |
| `warp.md` | Warp | 어두운 IDE 스타일 인터페이스, 블록 기반 명령 UI |
| `zapier.md` | Zapier | 따뜻한 주황색, 친근한 일러스트레이션 위주 |

### 인프라 및 클라우드 (Infrastructure & Cloud)

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `clickhouse.md` | ClickHouse | 노란색 악센트, 기술 문서 스타일 |
| `composio.md` | Composio | 컬러풀한 통합 아이콘이 있는 모던 다크 |
| `hashicorp.md` | HashiCorp | 엔터프라이즈의 깔끔함, 흑백 |
| `mongodb.md` | MongoDB | 녹색 잎 브랜딩, 개발자 문서 중심 |
| `sanity.md` | Sanity | 빨간색 악센트, 콘텐츠 중심의 에디토리얼 레이아웃 |
| `stripe.md` | Stripe | 시그니처 보라색 그라디언트, 300 굵기의 우아함 |

### 디자인 및 생산성 (Design & Productivity)

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `airtable.md` | Airtable | 다채롭고 친근함, 구조화된 데이터 미학 |
| `cal.md` | Cal.com | 깔끔한 중립적 UI, 개발자 지향적인 단순성 |
| `clay.md` | Clay | 유기적인 형태, 부드러운 그라디언트, 아트 디렉팅 레이아웃 |
| `figma.md` | Figma | 생생한 다색, 장난스러우면서도 전문적임 |
| `framer.md` | Framer | 강렬한 검정과 파랑, 모션 우선, 디자인 중심 |
| `intercom.md` | Intercom | 친근한 파란색 팔레트, 대화형 UI 패턴 |
| `miro.md` | Miro | 밝은 노란색 악센트, 무한 캔버스 미학 |
| `notion.md` | Notion | 따뜻한 미니멀리즘, 세리프 제목, 부드러운 표면 |
| `pinterest.md` | Pinterest | 빨간색 악센트, masonry 그리드, 이미지 우선 레이아웃 |
| `webflow.md` | Webflow | 파란색 악센트, 세련된 마케팅 사이트 미학 |

### 핀테크 및 암호화폐 (Fintech & Crypto)

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `coinbase.md` | Coinbase | 깔끔한 파란색 아이덴티티, 신뢰 중심, 기관의 느낌 |
| `kraken.md` | Kraken | 보라색 악센트의 어두운 UI, 데이터 밀도가 높은 대시보드 |
| `revolut.md` | Revolut | 매끄러운 어두운 인터페이스, 그라디언트 카드, 핀테크의 정밀함 |
| `wise.md` | Wise | 밝은 녹색 악센트, 친근하고 명확함 |

### 엔터프라이즈 및 소비자 (Enterprise & Consumer)

| 템플릿 | 사이트 | 스타일 |
|---|---|---|
| `airbnb.md` | Airbnb | 따뜻한 코랄 악센트, 사진 위주, 둥근 UI |
| `apple.md` | Apple | 프리미엄 여백, SF Pro, 시네마틱 이미지 |
| `bmw.md` | BMW | 어두운 프리미엄 표면, 정밀한 엔지니어링 미학 |
| `ibm.md` | IBM | Carbon 디자인 시스템, 구조화된 파란색 팔레트 |
| `nvidia.md` | NVIDIA | 녹색-검은색의 에너지, 기술적인 파워 미학 |
| `spacex.md` | SpaceX | 극명한 흑백, 풀 블리드 이미지, 미래지향적 |
| `spotify.md` | Spotify | 어두운 배경의 선명한 녹색, 굵은 서체, 앨범 아트 위주 |
| `uber.md` | Uber | 대담한 흑백, 촘촘한 서체, 도시의 에너지 |

## 디자인 선택하기

콘텐츠에 맞는 디자인을 선택하세요:

- **개발자 도구 / 대시보드:** Linear, Vercel, Supabase, Raycast, Sentry
- **문서 / 콘텐츠 사이트:** Mintlify, Notion, Sanity, MongoDB
- **마케팅 / 랜딩 페이지:** Stripe, Framer, Apple, SpaceX
- **다크 모드 UI:** Linear, Cursor, ElevenLabs, Warp, Superhuman
- **밝은 / 깔끔한 UI:** Vercel, Stripe, Notion, Cal.com, Replicate
- **장난기 있는 / 친숙한:** PostHog, Figma, Lovable, Zapier, Miro
- **프리미엄 / 럭셔리:** Apple, BMW, Stripe, Superhuman, Revolut
- **높은 데이터 밀도 / 대시보드:** Sentry, Kraken, Cohere, ClickHouse
- **고정폭 / 터미널 미학:** Ollama, OpenCode, x.ai, VoltAgent
