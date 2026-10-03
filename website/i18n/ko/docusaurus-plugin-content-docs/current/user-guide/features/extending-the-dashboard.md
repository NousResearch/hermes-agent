---
sidebar_position: 17
title: "대시보드 확장"
description: "Hermes 웹 대시보드용 테마와 플러그인 구축 — 팔레트, 타이포그래피, 레이아웃, 사용자 지정 탭, 셸 슬롯, 페이지 범위 슬롯 및 백엔드 API 라우트"
---

# 대시보드 확장

Hermes 웹 대시보드(`hermes dashboard`)는 코드베이스를 포크하지 않고도 테마를 바꾸고 확장할 수 있도록 설계되었습니다. 세 가지 계층이 공개됩니다.

1. **테마** — 대시보드의 팔레트, 타이포그래피, 레이아웃 및 컴포넌트별 크롬을 다시 그리는 YAML 파일입니다. `~/.hermes/dashboard-themes/`에 파일을 넣으면 테마 선택기에 표시됩니다.
2. **UI 플러그인** — `manifest.json`과 JavaScript 번들로 이루어진 디렉터리입니다. 탭을 등록하거나, 기본 페이지를 교체하거나, 페이지 범위 슬롯으로 페이지를 보강하거나, 이름 있는 셸 슬롯에 컴포넌트를 주입합니다.
3. **백엔드 플러그인** — 해당 플러그인 디렉터리 안의 Python 파일로 FastAPI `router`를 노출합니다. 라우트는 `/api/plugins/<name>/` 아래에 마운트되며 플러그인의 UI에서 호출합니다.

세 가지 모두 **실행 시 바로 추가할 수 있습니다**. 저장소를 복제하거나, `npm run build`를 실행하거나, 대시보드 소스를 패치할 필요가 없습니다. 이 페이지는 세 가지 모두에 대한 표준 참고 자료입니다.

대시보드를 사용하기만 하려면 [웹 대시보드](./web-dashboard)를 참고하세요. 웹 대시보드가 아닌 터미널 CLI의 테마를 바꾸려면 [스킨 및 테마](./skins)를 참고하세요. CLI 스킨 시스템과 대시보드 테마는 서로 관련이 없습니다.

:::note 데스크톱 앱이 아닙니다
이 페이지에서 다루는 것은 **웹 대시보드**(`hermes dashboard`) 플러그인 시스템입니다 — `window.__HERMES_PLUGIN_SDK__`, `manifest.json`, 사전 빌드된 JS 번들을 사용합니다. **네이티브 데스크톱 앱**(`hermes desktop`)에는 서로 무관한 자체 SDK가 있습니다 — 단일 ESM 파일이며 빌드 단계가 없습니다 — 자세한 내용은 [데스크톱 플러그인 SDK](/developer-guide/desktop-plugin-sdk)를 참고하세요. 두 시스템에서 공유되는 것은 백엔드 `plugin_api.py` 네임스페이스(`/api/plugins/<name>`)뿐입니다.
:::

:::note 구성 방식
테마와 플러그인은 독립적이지만 함께 사용할 때 더 큰 효과를 냅니다. 테마는 단독으로 사용할 수 있습니다(YAML 파일 하나만 필요). 플러그인도 단독으로 사용할 수 있습니다(탭 하나만 필요). 둘을 함께 사용하면 사용자 지정 HUD를 포함한 완전한 시각적 리스킨을 만들 수 있습니다 — 예제 `strike-freedom-cockpit` 데모( `hermes-example-plugins` 동반 저장소에 있으며 설치 방법은 [테마 + 플러그인 결합 데모](#combined-theme--plugin-demo) 참고)가 바로 그 예입니다.
:::

---

## 목차

- [테마](#themes)
  - [빠른 시작 — 첫 테마](#quick-start--your-first-theme)
  - [팔레트, 타이포그래피, 레이아웃](#palette-typography-layout)
  - [레이아웃 변형](#layout-variants)
  - [테마 에셋(이미지를 CSS 변수로 사용)](#theme-assets-images-as-css-vars)
  - [컴포넌트 크롬 재정의](#component-chrome-overrides)
  - [색상 재정의](#color-overrides)
  - [Raw `customCSS`](#raw-customcss)
  - [내장 테마](#built-in-themes)
  - [전체 테마 YAML 참고](#full-theme-yaml-reference)
- [플러그인](#plugins)
  - [빠른 시작 — 첫 플러그인](#quick-start--your-first-plugin)
  - [디렉터리 구조](#directory-layout)
  - [매니페스트 참고](#manifest-reference)
  - [플러그인 SDK](#the-plugin-sdk)
  - [셸 슬롯](#shell-slots)
  - [내장 페이지 교체(`tab.override`)](#replacing-built-in-pages-taboverride)
  - [내장 페이지 보강(페이지 범위 슬롯)](#augmenting-built-in-pages-page-scoped-slots)
  - [슬롯 전용 플러그인(`tab.hidden`)](#slot-only-plugins-tabhidden)
  - [백엔드 API 라우트](#backend-api-routes)
  - [플러그인별 사용자 지정 CSS](#custom-css-per-plugin)
  - [플러그인 검색 및 다시 로드](#plugin-discovery--reload)
- [테마 + 플러그인 결합 데모](#combined-theme--plugin-demo)
- [API 참고](#api-reference)
- [문제 해결](#troubleshooting)

---

## 테마

테마는 `~/.hermes/dashboard-themes/`에 저장되는 YAML 파일입니다. 파일 이름은 중요하지 않습니다(시스템이 사용하는 것은 테마의 `name:` 필드). 다만 일반적으로 `<name>.yaml` 형식을 사용합니다. 모든 필드는 선택 사항입니다 — 누락된 키는 내장 `default` 테마로 대체되므로 테마는 색상 하나만 정의해도 됩니다.

### 빠른 시작 — 첫 테마

```bash
mkdir -p ~/.hermes/dashboard-themes
```

```yaml
# ~/.hermes/dashboard-themes/neon.yaml
name: neon
label: Neon
description: Pure magenta on black

palette:
  background: "#000000"
  midground: "#ff00ff"
```

대시보드를 새로 고칩니다. 헤더의 팔레트 아이콘을 클릭하고 **Neon**을 선택합니다. 배경은 검은색이 되고 텍스트와 강조 색상은 자홍색이 되며, 카드·테두리·약한 색상·링 등의 모든 파생 색상은 CSS의 `color-mix()`를 통해 이 2색 조합에서 다시 계산됩니다.

온보딩은 이것으로 끝입니다. 파일 하나와 색상 두 개면 됩니다. 아래 내용은 모두 선택적인 세부 조정입니다.

### 팔레트, 타이포그래피, 레이아웃

이 세 블록이 테마의 핵심입니다. 각각 독립적이므로 하나만 재정의하고 나머지는 그대로 둘 수 있습니다.

#### 팔레트(3계층)

팔레트는 세 가지 색상 계층에 따뜻한 빛의 비네트 색상과 노이즈 그레인 배율을 더한 조합입니다. 대시보드의 디자인 시스템 계단식 구조는 CSS `color-mix()`를 통해 이 조합에서 shadcn 호환 토큰(카드, 팝오버, 약한 색상, 테두리, 기본 색상, 파괴적 색상, 링 등)을 모두 파생합니다. 세 색상을 재정의하면 전체 UI로 계단식 적용됩니다.

| 키 | 설명 |
|-----|-------------|
| `palette.background` | 가장 깊은 캔버스 색상 — 일반적으로 거의 검은색입니다. 페이지 배경과 카드 채우기를 결정합니다. |
| `palette.midground` | 기본 텍스트 및 강조 색상입니다. 대부분의 UI 크롬이 이 색상을 사용합니다(전경 텍스트, 버튼 외곽선, 포커스 링). |
| `palette.foreground` | 최상위 하이라이트입니다. 기본 테마에서는 알파 0인 흰색(보이지 않음)으로 설정하며, 위에 밝은 강조 색상을 표시하려는 테마는 알파를 높일 수 있습니다. |
| `palette.warmGlow` | `<Backdrop />`가 비네트 색상으로 사용하는 `rgba(...)` 문자열입니다. |
| `palette.noiseOpacity` | 그레인 오버레이의 0–1.2 배율입니다. 낮을수록 부드럽고 높을수록 거칠어집니다. |

각 계층에는 `{hex: "#RRGGBB", alpha: 0.0–1.0}` 또는 일반 hex 문자열을 사용할 수 있습니다(알파의 기본값은 1.0).

```yaml
palette:
  background:
    hex: "#05091a"
    alpha: 1.0
  midground: "#d8f0ff"          # bare hex, alpha = 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0                    # invisible top layer
  warmGlow: "rgba(255, 199, 55, 0.24)"
  noiseOpacity: 0.7
```

#### 타이포그래피

| 키 | 유형 | 설명 |
|-----|------|-------------|
| `fontSans` | string | 본문용 CSS font-family 스택(`html`, `body`에 적용)입니다. |
| `fontMono` | string | 코드 블록, `<code>`, `.font-mono` 유틸리티용 CSS font-family 스택입니다. |
| `fontDisplay` | string | 선택적인 제목/디스플레이 스택입니다. 지정하지 않으면 `fontSans`를 사용합니다. |
| `fontUrl` | string | 선택적인 외부 스타일시트 URL입니다. 테마 전환 시 `<head>`에 `<link rel="stylesheet">`로 삽입됩니다. 동일한 URL은 두 번 삽입하지 않습니다. Google Fonts, Bunny Fonts, 자체 호스팅 `@font-face` 시트 등 링크할 수 있는 모든 것을 사용할 수 있습니다. |
| `baseSize` | string | 루트 글꼴 크기 — rem 배율을 제어합니다. 예: `"14px"`, `"16px"`. |
| `lineHeight` | string | 기본 줄 높이입니다. 예: `"1.5"`, `"1.65"`. |
| `letterSpacing` | string | 기본 자간입니다. 예: `"0"`, `"0.01em"`, `"-0.01em"`. |

```yaml
typography:
  fontSans: '"Orbitron", "Eurostile", "Impact", sans-serif'
  fontMono: '"Share Tech Mono", ui-monospace, monospace'
  fontDisplay: '"Orbitron", "Eurostile", sans-serif'
  fontUrl: "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Share+Tech+Mono&display=swap"
  baseSize: "14px"
  lineHeight: "1.5"
  letterSpacing: "0.04em"
```

##### UI에서 글꼴 변경(YAML 불필요)

대시보드 헤더의 테마 선택기에는 **테마 목록** 아래에 **글꼴** 섹션이 있습니다. 여기서 글꼴을 선택하면 현재 활성화된 테마의 본문 글꼴이 재정의됩니다 — 이 선택은 테마와 독립적이며 테마를 바꿔도 유지됩니다(`config.yaml`의 `dashboard.font`에 저장). **테마 기본값**을 선택하면 재정의를 지우고 활성 테마 자체의 `fontSans`로 돌아갑니다.

선택기에는 엄선된 카탈로그가 제공됩니다(시스템 스택과 sans/serif/mono 전반의 Google Fonts 계열). 글꼴 스타일시트가 `<link>`로 삽입되므로 주입 출처를 고정하기 위해 자유 입력 글꼴 URL은 의도적으로 지원하지 않습니다. 완전히 사용자 지정한 글꼴은 위 예시처럼 테마 YAML에 `fontSans`와 `fontUrl`을 설정하세요. 테마의 `fontMono`(코드 블록, 터미널)는 UI 재정의의 영향을 받지 않습니다.

#### 레이아웃

| 키 | 값 | 설명 |
|-----|--------|-------------|
| `radius` | 임의의 CSS 길이(`"0"`, `"0.25rem"`, `"0.5rem"`, `"1rem"`, ...) | 모서리 반지름 토큰입니다. `--radius`에 매핑되고 `--radius-sm/md/lg/xl`로 계단식 적용되어 모든 둥근 요소가 함께 바뀝니다. |
| `density` | `compact` \| `comfortable` \| `spacious` | `--spacing-mul` CSS 변수로 적용되는 간격 배율입니다. `compact = 0.85×`, `comfortable = 1.0×`(기본값), `spacious = 1.2×`입니다. Tailwind의 기본 간격을 배율 조정하므로 padding, gap, space-between 유틸리티가 모두 비례해 바뀝니다. |

```yaml
layout:
  radius: "0"
  density: compact
```

### 레이아웃 변형

`layoutVariant`는 전체 셸 레이아웃을 선택합니다. 지정하지 않으면 기본값은 `"standard"`입니다.

| 변형 | 동작 |
|---------|-----------|
| `standard` | 단일 열, 최대 너비 1600px(기본값)입니다. |
| `cockpit` | 왼쪽 사이드바 레일(260px)과 주 콘텐츠입니다. 플러그인이 `sidebar` 슬롯을 통해 내용을 채웁니다 — [셸 슬롯](#shell-slots)을 참고하세요. 플러그인이 없으면 레일에 자리 표시자가 표시됩니다. |
| `tiled` | 최대 너비 제한을 없애 페이지가 뷰포트 전체 너비를 사용할 수 있습니다. |

```yaml
layoutVariant: cockpit
```

현재 변형은 `document.documentElement.dataset.layoutVariant`로 공개되므로, `customCSS`의 raw CSS에서 `:root[data-layout-variant="cockpit"] ...`로 대상을 지정할 수 있습니다.

### 테마 에셋(이미지를 CSS 변수로 사용)

테마에 아트워크 URL을 포함할 수 있습니다. 이름 있는 각 슬롯은 CSS 변수(`--theme-asset-<name>`)가 되며, 내장 셸과 모든 플러그인이 읽을 수 있습니다. `bg` 슬롯은 백드롭에 자동 연결되고 나머지는 플러그인에서 사용합니다.

```yaml
assets:
  bg: "https://example.com/hero-bg.jpg"           # auto-wired into <Backdrop />
  hero: "/my-images/strike-freedom.png"           # for plugin sidebars
  crest: "/my-images/crest.svg"                   # for header-left plugins
  logo: "/my-images/logo.png"
  sidebar: "/my-images/rail.png"
  header: "/my-images/header-art.png"
  custom:
    scanLines: "/my-images/scanlines.png"         # → --theme-asset-custom-scanLines
```

값에는 다음을 사용할 수 있습니다.

- 일반 URL — 자동으로 `url(...)`로 감쌉니다.
- 미리 감싼 `url(...)`, `linear-gradient(...)`, `radial-gradient(...)` 표현식 — 있는 그대로 사용합니다.
- `"none"` — 명시적으로 사용하지 않습니다.

모든 에셋은 `--theme-asset-<name>-raw`(래퍼가 없는 URL)로도 출력되므로, 플러그인이 `background-image` 대신 `<img src>`에 전달해야 할 때 사용할 수 있습니다.

플러그인은 일반 CSS 또는 JS로 이를 읽습니다.

```javascript
// In a plugin slot
const hero = getComputedStyle(document.documentElement)
  .getPropertyValue("--theme-asset-hero").trim();
```

### 컴포넌트 크롬 재정의

`componentStyles`는 CSS 선택자를 작성하지 않고도 개별 셸 컴포넌트의 스타일을 다시 지정합니다. 각 버킷의 항목은 CSS 변수(`--component-<bucket>-<kebab-property>`)가 되어 셸의 공유 컴포넌트가 읽습니다. 따라서 `card:` 재정의는 모든 `<Card>`에, `header:` 재정의는 앱 바에 적용됩니다.

```yaml
componentStyles:
  card:
    clipPath: "polygon(12px 0, 100% 0, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0 100%, 0 12px)"
    background: "linear-gradient(180deg, rgba(10, 22, 52, 0.85), rgba(5, 9, 26, 0.92))"
    boxShadow: "inset 0 0 0 1px rgba(64, 200, 255, 0.28)"
  header:
    background: "linear-gradient(180deg, rgba(16, 32, 72, 0.95), rgba(5, 9, 26, 0.9))"
  tab:
    clipPath: "polygon(6px 0, 100% 0, calc(100% - 6px) 100%, 0 100%)"
  sidebar: {}
  backdrop: {}
  footer: {}
  progress: {}
  badge: {}
  page: {}
```

지원되는 버킷은 `card`, `header`, `footer`, `sidebar`, `tab`, `progress`, `badge`, `backdrop`, `page`입니다.

속성 이름은 camelCase(`clipPath`)를 사용하며 kebab(`clip-path`)으로 출력됩니다. 값은 일반 CSS 문자열이므로 CSS가 허용하는 모든 것(`clip-path`, `border-image`, `background`, `box-shadow`, `animation`, ...)을 사용할 수 있습니다.

### 색상 재정의

대부분의 테마에는 필요하지 않습니다 — 3계층 팔레트가 모든 shadcn 토큰을 파생하기 때문입니다. 파생 결과로 만들 수 없는 특정 강조 색상이 필요할 때 `colorOverrides`를 사용하세요(파스텔 테마의 더 부드러운 파괴적 빨간색이나 브랜드의 특정 성공 녹색 등).

```yaml
colorOverrides:
  primary: "#ffce3a"
  primaryForeground: "#05091a"
  accent: "#3fd3ff"
  ring: "#3fd3ff"
  destructive: "#ff3a5e"
  border: "rgba(64, 200, 255, 0.28)"
```

지원되는 키는 `card`, `cardForeground`, `popover`, `popoverForeground`, `primary`, `primaryForeground`, `secondary`, `secondaryForeground`, `muted`, `mutedForeground`, `accent`, `accentForeground`, `destructive`, `destructiveForeground`, `success`, `warning`, `border`, `input`, `ring`입니다.

각 키는 `--color-<kebab>` CSS 변수에 1:1로 매핑됩니다(예: `primaryForeground` → `--color-primary-foreground`). 여기에 설정한 키는 활성 테마에서 팔레트 계단식 결과보다 우선합니다 — 다른 테마로 전환하면 재정의가 지워집니다.

### Raw `customCSS`

`componentStyles`로 표현할 수 없는 선택자 수준의 크롬(의사 요소, 애니메이션, 미디어 쿼리, 테마 범위 재정의)에는 `customCSS`에 raw CSS를 넣습니다.

```yaml
customCSS: |
  /* Scanline overlay — only visible when cockpit variant is active. */
  :root[data-layout-variant="cockpit"] body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 100;
    background: repeating-linear-gradient(to bottom,
      transparent 0px, transparent 2px,
      rgba(64, 200, 255, 0.035) 3px, rgba(64, 200, 255, 0.035) 4px);
    mix-blend-mode: screen;
  }
```

CSS는 테마 적용 시 단일 범위 지정 `<style data-hermes-theme-css>` 태그로 삽입되고 테마 전환 시 정리됩니다. **테마당 32KiB로 제한됩니다.**

### 내장 테마

각 내장 테마는 자체 팔레트, 타이포그래피 및 레이아웃을 제공하므로 전환하면 색상 외에도 눈에 띄는 변화가 생깁니다.

| 테마 | 팔레트 | 타이포그래피 | 레이아웃 |
|-------|---------|------------|--------|
| **Hermes Teal** (`default`) | 어두운 청록색 + 크림색 | 시스템 스택, 15px | 반지름 0.5rem, 편안함 |
| **Hermes Teal (Large)** (`default-large`) | 기본값과 동일 | 시스템 스택, 18px, 줄 높이 1.65 | 반지름 0.5rem, 넉넉함 |
| **Midnight** (`midnight`) | 짙은 청색-보라색 | Inter + JetBrains Mono, 14px | 반지름 0.75rem, 편안함 |
| **Ember** (`ember`) | 따뜻한 진홍색 + 청동색 | Spectral(세리프) + IBM Plex Mono, 15px | 반지름 0.25rem, 편안함 |
| **Mono** (`mono`) | 회색조 | IBM Plex Sans + IBM Plex Mono, 13px | 반지름 0, 조밀함 |
| **Cyberpunk** (`cyberpunk`) | 검은색 위 네온 녹색 | 어디서나 Share Tech Mono, 14px | 반지름 0, 조밀함 |
| **Rosé** (`rose`) | 분홍색 + 상아색 | Fraunces(세리프) + DM Mono, 16px | 반지름 1rem, 넉넉함 |

Hermes Teal을 제외하고 Google Fonts를 참조하는 테마는 필요할 때 스타일시트를 로드합니다 — 해당 테마로 처음 전환할 때 `<link>` 태그가 `<head>`에 삽입됩니다.

### 전체 테마 YAML 참고

필요한 것만 복사하고 삭제할 수 있도록 모든 설정을 한 파일에 넣은 예시입니다.

```yaml
# ~/.hermes/dashboard-themes/ocean.yaml
name: ocean
label: Ocean Deep
description: Deep sea blues with coral accents

# 3-layer palette (accepts {hex, alpha} or bare hex)
palette:
  background:
    hex: "#0a1628"
    alpha: 1.0
  midground:
    hex: "#a8d0ff"
    alpha: 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0.0
  warmGlow: "rgba(255, 107, 107, 0.35)"
  noiseOpacity: 0.7

typography:
  fontSans: "Poppins, system-ui, sans-serif"
  fontMono: "Fira Code, ui-monospace, monospace"
  fontDisplay: "Poppins, system-ui, sans-serif"   # optional
  fontUrl: "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap"
  baseSize: "15px"
  lineHeight: "1.6"
  letterSpacing: "-0.003em"

layout:
  radius: "0.75rem"
  density: comfortable

layoutVariant: standard        # standard | cockpit | tiled

assets:
  bg: "https://example.com/ocean-bg.jpg"
  hero: "/my-images/kraken.png"
  crest: "/my-images/anchor.svg"
  logo: "/my-images/logo.png"
  custom:
    pattern: "/my-images/waves.svg"

componentStyles:
  card:
    boxShadow: "inset 0 0 0 1px rgba(168, 208, 255, 0.18)"
  header:
    background: "linear-gradient(180deg, rgba(10, 22, 40, 0.95), rgba(5, 9, 26, 0.9))"

colorOverrides:
  destructive: "#ff6b6b"
  ring: "#ff6b6b"

customCSS: |
  /* Any additional selector-level tweaks */
```

파일을 만든 후 대시보드를 새로 고칩니다. 헤더 바에서 팔레트 아이콘을 클릭해 실시간으로 테마를 전환할 수 있습니다. 선택은 `config.yaml`의 `dashboard.theme`에 저장되며 다시 로드할 때 복원됩니다.

---

## 플러그인

대시보드 플러그인은 `manifest.json`, 사전 빌드된 JS 번들, 그리고 선택적인 CSS 파일 및 FastAPI 라우트용 Python 파일로 이루어진 디렉터리입니다. 플러그인은 `~/.hermes/plugins/<name>/`의 다른 Hermes 플러그인 옆에 위치합니다 — 대시보드 확장은 해당 플러그인 디렉터리 안의 `dashboard/` 하위 디렉터리이므로 하나의 플러그인으로 CLI/게이트웨이와 대시보드를 모두 확장할 수 있습니다.

플러그인은 React나 UI 컴포넌트를 번들로 포함하지 않습니다. `window.__HERMES_PLUGIN_SDK__`에 노출된 **플러그인 SDK**를 사용합니다. 따라서 플러그인 번들은 매우 작고(일반적으로 수 KB) 버전 충돌을 피할 수 있습니다.

### 빠른 시작 — 첫 플러그인

디렉터리 구조를 만듭니다.

```bash
mkdir -p ~/.hermes/plugins/my-plugin/dashboard/dist
```

매니페스트를 작성합니다.

```json
// ~/.hermes/plugins/my-plugin/dashboard/manifest.json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills"
  },
  "entry": "dist/index.js"
}
```

JS 번들을 작성합니다(일반 IIFE이므로 빌드 단계가 필요하지 않습니다).

```javascript
// ~/.hermes/plugins/my-plugin/dashboard/dist/index.js
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardHeader, CardTitle, CardContent } = SDK.components;

  function MyPage() {
    return React.createElement(Card, null,
      React.createElement(CardHeader, null,
        React.createElement(CardTitle, null, "My Plugin"),
      ),
      React.createElement(CardContent, null,
        React.createElement("p", { className: "text-sm text-muted-foreground" },
          "Hello from my custom dashboard tab.",
        ),
      ),
    );
  }

  window.__HERMES_PLUGINS__.register("my-plugin", MyPage);
})();
```

대시보드를 새로 고치면 탐색 바의 **Skills** 뒤에 탭이 표시됩니다.

:::tip React.createElement 건너뛰기
JSX를 선호한다면 React를 external로, IIFE 출력을 사용해 esbuild, Vite, rollup 등의 번들러를 사용하세요. 유일한 필수 조건은 최종 파일이 `<script>`로 로드할 수 있는 단일 JS 파일이어야 한다는 것입니다. React는 번들에 포함하지 않습니다. `SDK.React`에서 제공됩니다.
:::

### 디렉터리 구조

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml              # optional — existing CLI/gateway plugin manifest
├── __init__.py              # optional — existing CLI/gateway hooks
└── dashboard/               # dashboard extension
    ├── manifest.json        # required — tab config, icon, entry point
    ├── dist/
    │   ├── index.js         # required — pre-built JS bundle (IIFE)
    │   └── style.css        # optional — custom CSS
    └── plugin_api.py        # optional — backend API routes (FastAPI)
```

하나의 플러그인 디렉터리는 서로 독립적인 세 가지 확장을 포함할 수 있습니다.

- `plugin.yaml` + `__init__.py` — CLI/게이트웨이 플러그인([플러그인 페이지](./plugins) 참고).
- `dashboard/manifest.json` + `dashboard/dist/index.js` — 대시보드 UI 플러그인.
- `dashboard/plugin_api.py` — 대시보드 백엔드 라우트.

어느 것도 필수는 아니므로 필요한 계층만 포함하세요.

### 매니페스트 참고

```json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "description": "What this plugin does",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills",
    "override": "/",
    "hidden": false
  },
  "slots": ["sidebar", "header-left"],
  "entry": "dist/index.js",
  "css": "dist/style.css",
  "api": "plugin_api.py"
}
```

| 필드 | 필수 | 설명 |
|-------|----------|-------------|
| `name` | 예 | 고유한 플러그인 식별자입니다. 소문자이며 하이픈을 사용할 수 있습니다. URL과 등록에 사용됩니다. |
| `label` | 예 | 탐색 탭에 표시되는 이름입니다. |
| `description` | 아니요 | 짧은 설명입니다(대시보드 관리 화면에 표시). |
| `icon` | 아니요 | Lucide 아이콘 이름입니다. 기본값은 `Puzzle`이며 알 수 없는 이름도 `Puzzle`로 대체됩니다. |
| `version` | 아니요 | Semver 문자열입니다. 기본값은 `0.0.0`입니다. |
| `tab.path` | 예 | 탭의 URL 경로입니다(예: `/my-plugin`). |
| `tab.position` | 아니요 | 탭을 삽입할 위치입니다. `"end"`(기본값), `"after:<path>"`, `"before:<path>"` 중 하나입니다 — 콜론 뒤 값은 대상 탭의 **경로 세그먼트**입니다(앞에 슬래시를 넣지 않음). 예: `"after:skills"`, `"before:config"`. |
| `tab.override` | 아니요 | 새 탭을 추가하는 대신 **교체**할 내장 라우트 경로(`"/"`, `"/sessions"`, `"/config"`, ...)입니다. [내장 페이지 교체](#replacing-built-in-pages-taboverride)를 참고하세요. |
| `tab.hidden` | 아니요 | true이면 컴포넌트와 슬롯을 등록하지만 탐색에 탭을 추가하지 않습니다. 슬롯 전용 플러그인에 사용합니다. [슬롯 전용 플러그인](#slot-only-plugins-tabhidden)을 참고하세요. |
| `slots` | 아니요 | 플러그인이 채우는 이름 있는 셸 슬롯입니다. **문서 보조 정보일 뿐**이며 실제 등록은 JS 번들의 `registerSlot()`에서 이루어집니다. 여기에 슬롯을 나열하면 검색 화면이 더 많은 정보를 제공합니다. |
| `entry` | 예 | `dashboard/`를 기준으로 한 JS 번들의 경로입니다. 기본값은 `dist/index.js`입니다. |
| `css` | 아니요 | `<link>` 태그로 삽입할 CSS 파일의 경로입니다. |
| `api` | 아니요 | FastAPI 라우트가 있는 Python 파일의 경로입니다. `/api/plugins/<name>/`에 마운트됩니다. |

#### 사용 가능한 아이콘

플러그인은 Lucide 아이콘 이름을 사용합니다. 대시보드는 이름으로 매핑하며, 알 수 없는 이름은 조용히 `Puzzle`로 대체됩니다.

현재 매핑된 이름은 `Activity`, `BarChart3`, `Clock`, `Code`, `Database`, `Eye`, `FileText`, `Globe`, `Heart`, `KeyRound`, `MessageSquare`, `Package`, `Puzzle`, `Settings`, `Shield`, `Sparkles`, `Star`, `Terminal`, `Wrench`, `Zap`입니다.

다른 아이콘이 필요하면 `web/src/App.tsx`의 `ICON_MAP`에 순수 추가 변경을 하는 PR을 제출하세요.

### 플러그인 SDK

플러그인에 필요한 모든 것은 `window.__HERMES_PLUGIN_SDK__`에 있습니다. 플러그인은 React를 직접 import해서는 안 됩니다.

```javascript
const SDK = window.__HERMES_PLUGIN_SDK__;

// React + hooks
SDK.React                    // the React instance
SDK.hooks.useState
SDK.hooks.useEffect
SDK.hooks.useCallback
SDK.hooks.useMemo
SDK.hooks.useRef
SDK.hooks.useContext
SDK.hooks.createContext

// UI components (shadcn/ui primitives)
SDK.components.Card
SDK.components.CardHeader
SDK.components.CardTitle
SDK.components.CardContent
SDK.components.Badge
SDK.components.Button
SDK.components.Input
SDK.components.Label
SDK.components.Select
SDK.components.SelectOption
SDK.components.Separator
SDK.components.Tabs
SDK.components.TabsList
SDK.components.TabsTrigger
SDK.components.PluginSlot    // render a named slot (useful for nested plugin UIs)

// Hermes API client + raw fetcher
SDK.api                      // typed client — getStatus, getSessions, getConfig, ...
SDK.fetchJSON                // raw fetch for custom endpoints (plugin-registered routes)

// Utilities
SDK.utils.cn                 // Tailwind class merger (clsx + twMerge)
SDK.utils.timeAgo            // "5m ago" from unix timestamp
SDK.utils.isoTimeAgo         // "5m ago" from ISO string

// Hooks
SDK.useI18n                  // i18n hook for multi-language plugins
```

#### 플러그인의 백엔드 호출

```javascript
SDK.fetchJSON("/api/plugins/my-plugin/data")
  .then((data) => console.log(data))
  .catch((err) => console.error("API call failed:", err));
```

`fetchJSON`는 세션 인증 토큰을 주입하고, 오류를 throw된 예외로 전달하며, JSON을 자동으로 파싱합니다.

#### 내장 Hermes 엔드포인트 호출

```javascript
// Agent status
SDK.api.getStatus().then((s) => console.log("Version:", s.version));

// Recent sessions
SDK.api.getSessions(10).then((resp) => console.log(resp.sessions.length));
```

전체 목록은 [웹 대시보드 → REST API](./web-dashboard#rest-api)를 참고하세요.

### 셸 슬롯

슬롯을 사용하면 플러그인이 앱 셸의 이름 있는 위치(콕핏 사이드바, 헤더, 푸터, 오버레이 계층)에 컴포넌트를 주입할 수 있습니다. 여러 플러그인이 같은 슬롯을 채울 수 있으며 등록 순서대로 쌓아 렌더링합니다.

플러그인 번들 내부에서 등록합니다.

```javascript
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sidebar", MySidebar);
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "header-left", MyCrest);
```

#### 슬롯 목록

**셸 전체 슬롯**(앱 크롬 어디에서나 렌더링):

| 슬롯 | 위치 |
|------|----------|
| `backdrop` | 노이즈 계층 위의 `<Backdrop />` 계층 내부입니다. |
| `header-left` | 상단 바의 Hermes 브랜드 앞입니다. |
| `header-right` | 상단 바의 테마/언어 전환기 앞입니다. |
| `header-banner` | 탐색 아래의 전체 너비 스트립입니다. |
| `sidebar` | 콕핏 사이드바 레일 — **`layoutVariant === "cockpit"`일 때만 렌더링됩니다**. |
| `pre-main` | 라우트 아웃렛 위(`<main>` 내부)입니다. |
| `post-main` | 라우트 아웃렛 아래(`<main>` 내부)입니다. |
| `footer-left` | 푸터 셀 콘텐츠(기본값을 대체)입니다. |
| `footer-right` | 푸터 셀 콘텐츠(기본값을 대체)입니다. |
| `overlay` | 다른 모든 요소 위에 고정 배치되는 계층입니다. `customCSS`만으로는 구현하기 어려운 크롬(스캔라인, 비네트)에 유용합니다. |

**페이지 범위 슬롯**(이름 있는 내장 페이지에서만 렌더링 — 전체 라우트를 재정의하지 않고 기존 페이지에 위젯, 카드 또는 툴바를 주입할 때 사용):

| 슬롯 | 렌더링 위치 |
|------|------------------|
| `sessions:top` / `sessions:bottom` | `/sessions` 페이지의 위 / 아래입니다. |
| `analytics:top` / `analytics:bottom` | `/analytics` 페이지의 위 / 아래입니다. |
| `logs:top` / `logs:bottom` | `/logs`의 위(필터 툴바 위) / 아래(로그 뷰어 아래)입니다. |
| `cron:top` / `cron:bottom` | `/cron` 페이지의 위 / 아래입니다. |
| `skills:top` / `skills:bottom` | `/skills` 페이지의 위 / 아래입니다. |
| `config:top` / `config:bottom` | `/config` 페이지의 위 / 아래입니다. |
| `env:top` / `env:bottom` | `/env`(키) 페이지의 위 / 아래입니다. |
| `docs:top` / `docs:bottom` | `/docs`의 위(iframe 위) / 아래입니다. |
| `chat:top` / `chat:bottom` | `/chat`의 위 / 아래입니다(내장 채팅이 활성화된 경우에만). |

예 — Sessions 페이지 상단에 배너 카드를 추가합니다.

```javascript
function PinnedSessionsBanner() {
  return React.createElement(Card, null,
    React.createElement(CardContent, { className: "py-2 text-xs" },
      "Pinned note injected by my-plugin"),
  );
}

window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sessions:top", PinnedSessionsBanner);
```

셸은 위 슬롯에 대해서만 `<PluginSlot name="..." />`을 렌더링합니다. 중첩 플러그인 UI를 위해 플러그인이 자체 슬롯을 노출할 수 있도록 레지스트리에서는 추가 이름도 허용합니다 — 플러그인은 `SDK.components.PluginSlot`을 사용할 수 있습니다.

#### 재등록 및 HMR

동일한 `(plugin, slot)` 쌍이 두 번 등록되면 나중 호출이 이전 호출을 대체합니다 — 이는 React HMR이 플러그인 다시 마운트를 처리하는 방식과 일치합니다.

### 내장 페이지 교체(`tab.override`)

`tab.override`를 내장 라우트 경로로 설정하면 플러그인 컴포넌트가 새 탭을 추가하는 대신 해당 페이지를 교체합니다. 다른 대시보드 부분은 유지하면서 테마에 맞춘 사용자 지정 홈 페이지(`/`)를 만들 때 유용합니다.

```json
{
  "name": "my-home",
  "label": "Home",
  "tab": {
    "path": "/my-home",
    "override": "/",
    "position": "end"
  },
  "entry": "dist/index.js"
}
```

`override`가 설정되면 다음과 같이 동작합니다.

- `/`의 원래 페이지 컴포넌트가 라우터에서 제거됩니다.
- 플러그인이 `/`에서 렌더링됩니다.
- `tab.path`에 대한 탐색 탭은 추가되지 않습니다(override가 목적이기 때문).

하나의 경로는 하나의 플러그인만 재정의할 수 있습니다. 두 플러그인이 같은 override를 선언하면 첫 번째가 승리하고 두 번째는 개발 모드 경고와 함께 무시됩니다.

기존 페이지를 인수하지 않고 카드나 툴바만 추가하려면 대신 [페이지 범위 슬롯](#augmenting-built-in-pages-page-scoped-slots)을 사용하세요.

### 내장 페이지 보강(페이지 범위 슬롯)

`tab.override`를 통한 전체 교체는 무겁습니다 — 플러그인이 해당 페이지 전체와 앞으로 제공될 업데이트까지 소유하게 됩니다. 대부분의 경우 기존 페이지에 배너, 카드 또는 툴바를 추가하려는 것뿐입니다. 이를 위한 것이 **페이지 범위 슬롯**입니다.

모든 내장 페이지는 콘텐츠 영역의 위와 아래에 렌더링되는 `<page>:top` 및 `<page>:bottom` 슬롯을 노출합니다. 플러그인은 `registerSlot()`을 호출해 하나를 채웁니다 — 내장 페이지는 정상적으로 작동하고 플러그인 컴포넌트는 그 옆에 렌더링됩니다.

사용 가능한 슬롯은 `sessions:*`, `analytics:*`, `logs:*`, `cron:*`, `skills:*`, `config:*`, `env:*`, `docs:*`, `chat:*`이며 각각 `:top`과 `:bottom`을 가집니다. 전체 목록은 [셸 슬롯 → 슬롯 목록](#slot-catalogue)을 참고하세요.

최소 예시 — Sessions 페이지 상단에 배너를 고정합니다.

```json
// ~/.hermes/plugins/session-notes/dashboard/manifest.json
{
  "name": "session-notes",
  "label": "Session Notes",
  "tab": { "path": "/session-notes", "hidden": true },
  "slots": ["sessions:top"],
  "entry": "dist/index.js"
}
```

```javascript
// ~/.hermes/plugins/session-notes/dashboard/dist/index.js
(function () {
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardContent } = SDK.components;

  function Banner() {
    return React.createElement(Card, null,
      React.createElement(CardContent, { className: "py-2 text-xs" },
        "Remember to label important sessions before archiving."),
    );
  }

  // Placeholder for the hidden tab.
  window.__HERMES_PLUGINS__.register("session-notes", function () { return null; });

  // The real work.
  window.__HERMES_PLUGINS__.registerSlot("session-notes", "sessions:top", Banner);
})();
```

핵심 사항:

- `tab.hidden: true`는 플러그인을 사이드바에서 숨깁니다 — 독립적인 페이지가 없습니다.
- `slots` 매니페스트 필드는 문서용일 뿐입니다. 실제 연결은 JS 번들의 `registerSlot()`에서 이루어집니다.
- 여러 플러그인이 동일한 페이지 범위 슬롯을 선언할 수 있습니다. 등록 순서대로 쌓아 렌더링합니다.
- 등록하는 플러그인이 없으면 영향이 전혀 없습니다. 내장 페이지는 이전과 정확히 동일하게 렌더링됩니다.

참고 플러그인([`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/example-dashboard)의 `example-dashboard`)에는 `sessions:top`에 배너를 주입하는 실제 데모가 포함되어 있습니다 — 설치하면 처음부터 끝까지 패턴을 확인할 수 있습니다.

### 슬롯 전용 플러그인(`tab.hidden`)

`tab.hidden: true`이면 플러그인은 컴포넌트(직접 URL 방문용)와 슬롯을 등록하지만 탐색에 탭을 추가하지 않습니다. 슬롯에만 주입하는 플러그인(헤더 문장, 사이드바 HUD, 오버레이)에 사용합니다.

```json
{
  "name": "header-crest",
  "label": "Header Crest",
  "tab": {
    "path": "/header-crest",
    "position": "end",
    "hidden": true
  },
  "slots": ["header-left"],
  "entry": "dist/index.js"
}
```

번들은 여전히 자리 표시자 컴포넌트로 `register()`를 호출하고(누군가 직접 URL을 방문하는 경우를 대비한 좋은 관행), 실제 작업을 수행하기 위해 `registerSlot()`을 호출합니다.

### 백엔드 API 라우트

매니페스트에 `api`를 설정하면 플러그인이 FastAPI 라우트를 등록할 수 있습니다. 파일을 만들고 `router`를 export합니다.

```python
# ~/.hermes/plugins/my-plugin/dashboard/plugin_api.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/data")
async def get_data():
    return {"items": ["one", "two", "three"]}

@router.post("/action")
async def do_action(body: dict):
    return {"ok": True, "received": body}
```

라우트는 `/api/plugins/<name>/` 아래에 마운트되므로 위 예시는 다음과 같이 됩니다.

- `GET  /api/plugins/my-plugin/data`
- `POST  /api/plugins/my-plugin/action`

플러그인 API 라우트는 대시보드의 일반 인증 게이트 뒤에 있습니다 — 인증되지 않은 요청은 플러그인 라우트가 실행되기 전에 `401`을 받고, 비활성화된 플러그인의 라우트 요청도 요청 시 거부됩니다. 그래도 **신뢰할 수 없는 플러그인을 실행한다면 `--host 0.0.0.0`으로 대시보드를 공개 인터페이스에 노출하지 마세요** — 인증된 세션은 해당 라우트에도 접근할 수 있습니다.

#### Hermes 내부 접근

백엔드 라우트는 대시보드 프로세스 안에서 실행되므로 hermes-agent 코드베이스를 직접 import할 수 있습니다.

```python
from fastapi import APIRouter
from hermes_state import SessionDB
from hermes_cli.config import load_config

router = APIRouter()

@router.get("/session-count")
async def session_count():
    db = SessionDB()
    try:
        count = len(db.list_sessions(limit=9999))
        return {"count": count}
    finally:
        db.close()

@router.get("/config-snapshot")
async def config_snapshot():
    cfg = load_config()
    return {"model": cfg.get("model", {})}
```

### 플러그인별 사용자 지정 CSS

플러그인에 Tailwind 클래스와 인라인 `style=`을 넘어서는 스타일이 필요하면 CSS 파일을 추가하고 매니페스트에서 참조합니다.

```json
{
  "css": "dist/style.css"
}
```

파일은 플러그인 로드 시 `<link>` 태그로 삽입됩니다. 대시보드 스타일과의 충돌을 피하도록 구체적인 클래스 이름을 사용하고, 테마에 맞게 유지되도록 대시보드의 CSS 변수를 참조하세요.

```css
/* dist/style.css */
.my-plugin-chart {
  border: 1px solid var(--color-border);
  background: var(--color-card);
  color: var(--color-card-foreground);
  padding: 1rem;
}
.my-plugin-chart:hover {
  border-color: var(--color-ring);
}
```

대시보드는 모든 shadcn 토큰을 `--color-*`로, 테마 추가 토큰을(`--theme-asset-*`, `--component-<bucket>-*`, `--radius`, `--spacing-mul`) 노출합니다. 이들을 참조하면 활성 테마에 따라 플러그인도 자동으로 다시 스킨됩니다.

### 플러그인 검색 및 다시 로드

대시보드는 세 디렉터리에서 `dashboard/manifest.json`을 검색합니다.

| 우선순위 | 디렉터리 | 소스 레이블 |
|----------|-----------|--------------|
| 1(충돌 시 우선) | `~/.hermes/plugins/<name>/dashboard/` | `user` |
| 2 | `<repo>/plugins/memory/<name>/dashboard/` | `bundled` |
| 2 | `<repo>/plugins/<name>/dashboard/` | `bundled` |
| 3 | `./.hermes/plugins/<name>/dashboard/` | `project` — `HERMES_ENABLE_PROJECT_PLUGINS`가 설정된 경우에만 |

검색 결과는 대시보드 프로세스별로 캐시됩니다. 새 플러그인을 추가한 후 다음 중 하나를 실행합니다.

```bash
# Force a rescan without restart
curl http://127.0.0.1:9119/api/dashboard/plugins/rescan
```

…또는 `hermes dashboard`를 다시 시작합니다.

#### 플러그인 로드 수명 주기

1. 대시보드가 로드됩니다. `main.tsx`가 SDK를 `window.__HERMES_PLUGIN_SDK__`에, 레지스트리를 `window.__HERMES_PLUGINS__`에 노출합니다.
2. `App.tsx`가 `usePlugins()`를 호출하고 `GET /api/dashboard/plugins`를 가져옵니다.
3. 각 매니페스트에 대해 CSS `<link>`를 삽입하고(선언된 경우), JS 번들을 로드하는 `<script>` 태그를 삽입합니다.
4. 플러그인의 IIFE가 실행되어 `window.__HERMES_PLUGINS__.register(name, Component)`를 호출합니다 — 선택적으로 각 슬롯에 대해 `.registerSlot(name, slot, Component)`도 호출합니다.
5. 대시보드는 매니페스트에 등록된 컴포넌트를 연결하고, `hidden`이 아니면 탐색에 탭을 추가하며, 컴포넌트를 라우트로 마운트합니다.

플러그인은 스크립트가 로드된 후 **최대 2초** 동안 `register()`를 호출할 수 있습니다. 그 후 대시보드는 기다림을 중단하고 초기 렌더링을 완료합니다. 나중에 플러그인이 등록되어도 탐색은 반응형이므로 표시됩니다.

플러그인의 스크립트를 로드하지 못하면(404, 구문 오류, IIFE 실행 중 예외) 대시보드는 브라우저 콘솔에 경고를 기록하고 해당 플러그인 없이 계속합니다.

---

## 테마 + 플러그인 결합 데모

[`strike-freedom-cockpit`](https://github.com/NousResearch/hermes-example-plugins/tree/main/strike-freedom-cockpit) 플러그인(`hermes-example-plugins` 동반 저장소)은 완전한 리스킨 데모입니다. 테마 YAML과 슬롯 전용 플러그인을 결합해 대시보드를 포크하지 않고 콕핏 스타일 HUD를 만듭니다.

**시연 내용:**

- 팔레트, 타이포그래피, `fontUrl`, `layoutVariant: cockpit`, `assets`, `componentStyles`(노치가 있는 카드 모서리, 그라데이션 배경), `colorOverrides`, `customCSS`(스캔라인 오버레이)를 사용하는 완전한 테마.
- 세 슬롯에 등록하는 슬롯 전용 플러그인(`tab.hidden: true`):
  - `sidebar` — `SDK.api.getStatus()`로 구동되는 실시간 텔레메트리 막대가 있는 MS-STATUS 패널.
  - `header-left` — 활성 테마에서 `--theme-asset-crest`를 읽는 세력 문장.
  - `footer-right` — 기본 조직 문구를 대체하는 사용자 지정 태그라인.
- 플러그인은 CSS 변수를 통해 테마가 제공하는 아트워크를 읽으므로 테마를 바꾸면 플러그인 코드 변경 없이 hero/crest가 바뀝니다.

**설치:**

```bash
git clone https://github.com/NousResearch/hermes-example-plugins.git

# Theme
cp hermes-example-plugins/strike-freedom-cockpit/theme/strike-freedom.yaml \
   ~/.hermes/dashboard-themes/

# Plugin
cp -r hermes-example-plugins/strike-freedom-cockpit ~/.hermes/plugins/
```

대시보드를 열고 테마 선택기에서 **Strike Freedom**을 선택합니다. 콕핏 사이드바가 나타나고, 헤더에 문장이 표시되며, 태그라인이 푸터를 대체합니다. **Hermes Teal**로 돌아가면 플러그인은 설치된 상태로 남지만 보이지 않습니다(`sidebar` 슬롯은 `cockpit` 레이아웃 변형에서만 렌더링됩니다).

동반 저장소의 플러그인 소스(`strike-freedom-cockpit/dashboard/dist/index.js`)를 보면 CSS 변수를 읽고, 슬롯을 지원하지 않는 이전 대시보드를 처리하며, 하나의 번들에서 세 슬롯을 등록하는 방법을 확인할 수 있습니다.

---

## API 참고

### 테마 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|----------|--------|-------------|
| `/api/dashboard/themes` | GET | 사용 가능한 테마와 활성 이름을 나열합니다. 내장 테마는 `{name, label, description}`을 반환하며, 사용자 테마에는 전체 정규화 테마 객체가 있는 `definition` 필드도 포함됩니다. |
| `/api/dashboard/theme` | PUT | 활성 테마를 설정합니다. 본문: `{"name": "midnight"}`. `dashboard.theme` 아래의 `config.yaml`에 저장됩니다. |

### 플러그인 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|----------|--------|-------------|
| `/api/dashboard/plugins` | GET | 검색된 플러그인을 나열합니다(매니페스트 포함, 내부 필드 제외). |
| `/api/dashboard/plugins/rescan` | GET | 재시작하지 않고 플러그인 디렉터리를 강제로 다시 검색합니다. |
| `/dashboard-plugins/<name>/<path>` | GET | 플러그인의 `dashboard/` 디렉터리에서 정적 에셋을 제공합니다. 경로 탐색은 차단됩니다. |
| `/api/plugins/<name>/*` | * | 플러그인이 등록한 백엔드 라우트입니다. |

### `window`의 SDK

| 전역 | 유형 | 제공자 |
|--------|------|----------|
| `window.__HERMES_PLUGIN_SDK__` | object | `registry.ts` — React, 훅, UI 컴포넌트, API 클라이언트, 유틸리티. |
| `window.__HERMES_PLUGINS__.register(name, Component)` | function | 플러그인의 주 컴포넌트를 등록합니다. |
| `window.__HERMES_PLUGINS__.registerSlot(name, slot, Component)` | function | 이름 있는 셸 슬롯에 등록합니다. |

---

## 문제 해결

**테마가 선택기에 나타나지 않습니다.**
파일이 `~/.hermes/dashboard-themes/`에 있고 `.yaml` 또는 `.yml`로 끝나는지 확인하세요. 페이지를 새로 고칩니다. `curl http://127.0.0.1:9119/api/dashboard/themes`를 실행하면 응답에 테마가 있어야 합니다. YAML 파싱 오류가 있으면 대시보드는 `~/.hermes/logs/` 아래의 `errors.log`에 기록합니다.

**플러그인 탭이 표시되지 않습니다.**
1. 매니페스트가 `~/.hermes/plugins/<name>/dashboard/manifest.json`에 있는지 확인합니다(`dashboard/` 하위 디렉터리에 유의).
2. `curl http://127.0.0.1:9119/api/dashboard/plugins/rescan`으로 검색을 강제합니다.
3. 브라우저 개발자 도구 → Network를 열고 `manifest.json`, `index.js`, CSS가 404 없이 로드되는지 확인합니다.
4. 브라우저 개발자 도구 → Console을 열고 IIFE 실행 중 오류나 `window.__HERMES_PLUGINS__ is undefined`를 확인합니다(대개 이전 React 렌더링 충돌로 SDK가 초기화되지 않았다는 뜻).
5. 번들이 `manifest.json:name`과 **동일한 이름**으로 `window.__HERMES_PLUGINS__.register(...)`를 호출하는지 확인합니다.

**슬롯에 등록한 컴포넌트가 렌더링되지 않습니다.**
`sidebar` 슬롯은 활성 테마의 `layoutVariant: cockpit`에서만 렌더링됩니다. 다른 슬롯은 항상 렌더링됩니다. 적중되지 않는 슬롯에 등록하는 경우 `registerSlot` 내부에 `console.log`를 추가해 플러그인 번들이 실제로 실행되었는지 확인하세요.

**플러그인 백엔드 라우트가 404를 반환합니다.**
1. 매니페스트에 `dashboard/` 안의 기존 파일을 가리키는 `"api": "plugin_api.py"`가 있는지 확인합니다.
2. `hermes dashboard`를 다시 시작합니다 — 플러그인 API 라우트는 다시 검색할 때가 아니라 시작 시 한 번 마운트됩니다.
3. `plugin_api.py`가 모듈 수준의 `router = APIRouter()`를 export하는지 확인합니다. 다른 export 이름은 검색되지 않습니다.
4. `~/.hermes/logs/errors.log`에서 `Failed to load plugin <name> API routes`를 확인합니다 — import 오류가 여기에 기록됩니다.

**테마 변경으로 색상 재정의가 사라집니다.**
`colorOverrides`는 활성 테마의 범위에 속하며 테마를 전환하면 지워집니다 — 의도된 동작입니다. 계속 유지되는 재정의가 필요하다면 실시간 전환기가 아니라 테마의 YAML에 넣으세요.

**테마의 customCSS가 잘립니다.**
`customCSS` 블록은 테마당 32KiB로 제한됩니다. 큰 스타일시트를 여러 테마로 나누거나, `css` 필드로 전체 스타일시트를 주입하는 플러그인으로 전환하세요(크기 제한 없음).

**PyPI에 플러그인을 배포하고 싶습니다.**
대시보드 플러그인은 pip 엔트리 포인트가 아니라 디렉터리 구조로 설치됩니다. 현재 가장 깔끔한 배포 방법은 사용자가 git 저장소를 `~/.hermes/plugins/`에 복제하는 것입니다. 대시보드 플러그인용 pip 기반 설치기는 현재 연결되어 있지 않습니다.
