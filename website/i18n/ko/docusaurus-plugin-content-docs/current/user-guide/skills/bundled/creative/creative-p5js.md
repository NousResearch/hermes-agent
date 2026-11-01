---
title: "P5Js — p5.js 스케치: 생성 예술, 셰이더, 인터랙티브, 3D"
sidebar_label: "P5Js"
description: "p5.js 스케치: 생성 예술, 셰이더, 인터랙티브, 3D"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 소스 SKILL.md를 편집하세요. */}

# P5Js

p5.js 스케치: 생성 예술, 셰이더, 인터랙티브, 3D.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 기본 포함(설치 시 기본값) |
| 경로 | `skills/creative/p5js` |
| 버전 | `1.0.0` |
| 작성자 | SHL0MS, Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `creative-coding`, `generative-art`, `p5js`, `canvas`, `interactive`, `visualization`, `webgl`, `shaders`, `animation` |
| 관련 스킬 | [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# p5.js 제작 파이프라인

## 사용 시점

사용자가 p5.js 스케치, 크리에이티브 코딩, 생성 예술, 인터랙티브 시각화, 캔버스 애니메이션, 브라우저 기반 시각 예술, 데이터 시각화, 셰이더 효과 또는 p5.js 프로젝트를 요청할 때 사용합니다.

## 구성 내용

p5.js를 사용한 인터랙티브 및 생성형 시각 예술을 위한 제작 파이프라인입니다. 브라우저 기반 스케치, 생성 예술, 데이터 시각화, 인터랙티브 경험, 3D 장면, 오디오 반응형 시각 효과, 모션 그래픽을 만들고 HTML, PNG, GIF, MP4 또는 SVG로 내보냅니다. 다루는 내용: 2D/3D 렌더링, 노이즈 및 파티클 시스템, 플로 필드, 셰이더(GLSL), 픽셀 조작, 키네틱 타이포그래피, WebGL 장면, 오디오 분석, 마우스/키보드 상호작용, 헤드리스 고해상도 내보내기.

## 크리에이티브 기준

이것은 브라우저에서 렌더링되는 시각 예술입니다. 캔버스는 매체이고, 알고리즘은 붓입니다.

**코드를 한 줄이라도 작성하기 전에** 창작 콘셉트를 명확히 설명합니다. 이 작품은 무엇을 전달하나요? 무엇이 보는 사람의 스크롤을 멈추게 하나요? 코드 튜토리얼 예제와 무엇이 다른가요? 사용자의 프롬프트는 출발점일 뿐입니다. 창작적 야심을 가지고 해석하세요.

**첫 렌더의 완성도는 타협할 수 없습니다.** 첫 로드부터 결과물이 시각적으로 강렬해야 합니다. p5.js 튜토리얼 연습처럼 보이거나, 기본 설정처럼 보이거나, "AI가 생성한 크리에이티브 코딩"처럼 보인다면 잘못된 것입니다. 배포하기 전에 다시 생각하세요.

**참고 자료의 어휘를 넘어가세요.** 참고 자료의 노이즈 함수, 파티클 시스템, 색상 팔레트, 셰이더 효과는 시작 어휘일 뿐입니다. 모든 프로젝트에서 결합하고, 겹치고, 새롭게 발명하세요. 카탈로그는 물감 팔레트이고, 그림은 여러분이 그립니다.

**적극적으로 창의성을 발휘하세요.** 사용자가 "파티클 시스템"을 요청했다면, 창발적인 군집 행동, 잔상을 남기는 고스트 에코, 팔레트가 이동하는 깊이 안개, 숨 쉬는 배경 노이즈 필드를 갖춘 파티클 시스템을 제공하세요. 사용자가 요청하지 않았지만 좋아할 만한 시각적 디테일을 하나 이상 포함하세요.

**밀도 있고, 층이 있으며, 숙고된 결과를 만드세요.** 모든 프레임은 오래 볼수록 새로운 것을 보여줘야 합니다. 평평한 흰색 배경은 절대 사용하지 마세요. 항상 구도상의 위계를 두세요. 항상 의도적인 색을 사용하세요. 가까이서 볼 때만 드러나는 미세한 디테일을 항상 포함하세요.

**기능 수보다 일관된 미학을 우선하세요.** 모든 요소는 하나의 통일된 시각 언어를 따라야 합니다. 색온도를 공유하고, 선 두께의 어휘를 일관되게 사용하며, 움직임의 속도를 조화롭게 맞추세요. 서로 관련 없는 효과 열 가지를 넣은 스케치는 서로 어울리는 효과 세 가지를 넣은 스케치보다 좋지 않습니다.

## 모드

| 모드 | 입력 | 출력 | 참고 |
|------|-------|--------|-----------|
| **생성 예술** | 시드 / 매개변수 | 절차적 시각 구성(정지 또는 애니메이션) | `references/visual-effects.md` |
| **데이터 시각화** | 데이터셋 / API | 인터랙티브 차트, 그래프, 사용자 정의 데이터 표시 | `references/interaction.md` |
| **인터랙티브 경험** | 없음(사용자가 조작) | 마우스/키보드/터치로 조작하는 스케치 | `references/interaction.md` |
| **애니메이션 / 모션 그래픽** | 타임라인 / 스토리보드 | 시간에 따른 시퀀스, 키네틱 타이포그래피, 트랜지션 | `references/animation.md` |
| **3D 장면** | 콘셉트 설명 | WebGL 지오메트리, 조명, 카메라, 머티리얼 | `references/webgl-and-3d.md` |
| **이미지 처리** | 이미지 파일 | 픽셀 조작, 필터, 모자이크, 점묘화 | `references/visual-effects.md` § 픽셀 조작 |
| **오디오 반응형** | 오디오 파일 / 마이크 | 사운드 기반 생성 시각 효과 | `references/interaction.md` § 오디오 입력 |

## 스택

프로젝트마다 빌드 단계가 필요 없는 독립형 단일 HTML 파일을 사용합니다.

| 계층 | 도구 | 용도 |
|-------|------|---------|
| 코어 | p5.js 1.11.3 (CDN) | 캔버스 렌더링, 수학, 변환, 이벤트 처리 |
| 3D | p5.js WebGL 모드 | 3D 지오메트리, 카메라, 조명, GLSL 셰이더 |
| 오디오 | p5.sound.js (CDN) | FFT 분석, 진폭, 마이크 입력, 오실레이터 |
| 내보내기 | 내장 `saveCanvas()` / `saveGif()` / `saveFrames()` | PNG, GIF, 프레임 시퀀스 출력 |
| 캡처 | CCapture.js (선택 사항) | 결정론적 프레임 레이트의 동영상 캡처(WebM, GIF) |
| 헤드리스 | Puppeteer + Node.js (선택 사항) | 자동화된 고해상도 렌더링, ffmpeg를 통한 MP4 |
| SVG | p5.js-svg 1.6.0 (선택 사항) | 인쇄용 벡터 출력 — p5.js 1.x 필요 |
| 내추럴 미디어 | p5.brush (선택 사항) | 수채화, 목탄, 펜 — p5.js 2.x + WEBGL 필요 |
| 텍스처 | p5.grain (선택 사항) | 필름 그레인, 텍스처 오버레이 |
| 폰트 | Google Fonts / `loadFont()` | OTF/TTF/WOFF2를 통한 사용자 정의 타이포그래피 |

### 버전 참고

**p5.js 1.x** (1.11.3)가 기본값입니다. 안정적이고 문서가 잘 갖춰져 있으며 라이브러리 호환성이 가장 넓습니다. 프로젝트에 2.x 기능이 필요한 경우가 아니라면 이를 사용하세요.

**p5.js 2.x** (2.2+)에는 `preload()`를 대체하는 `async setup()`, OKLCH/OKLAB 색상 모드, `splineVertex()`, 셰이더 `.modify()` API, 가변 폰트, `textToContours()`, 포인터 이벤트가 추가됩니다. p5.brush에 필요합니다. `references/core-api.md` § p5.js 2.0을 참고하세요.

## 파이프라인

모든 프로젝트는 동일한 6단계 경로를 따릅니다:

```
CONCEPT → DESIGN → CODE → PREVIEW → EXPORT → VERIFY
```

1. **콘셉트** — 창작 비전 설명: 분위기, 색의 세계, 움직임의 어휘, 무엇이 독특한지
2. **디자인** — 모드, 캔버스 크기, 상호작용 모델, 색상 시스템, 내보내기 형식을 선택합니다. 콘셉트를 기술적 결정으로 매핑합니다.
3. **코드** — 인라인 p5.js를 포함한 단일 HTML 파일 작성. 구조: 전역 변수 → `preload()` → `setup()` → `draw()` → 헬퍼 → 클래스 → 이벤트 핸들러
4. **미리보기** — 브라우저에서 열고 시각적 품질을 확인합니다. 목표 해상도에서 테스트합니다. 성능을 확인합니다.
5. **내보내기** — 출력 캡처: PNG에는 `saveCanvas()`, GIF에는 `saveGif()`, MP4에는 `saveFrames()` + ffmpeg, 헤드리스 일괄 처리에는 Puppeteer
6. **검증** — 출력이 콘셉트와 일치하나요? 의도한 표시 크기에서 시각적으로 강렬한가요? 액자에 넣고 싶을 정도인가요?

## 크리에이티브 디렉션

### 미학적 차원

| 차원 | 선택지 | 참고 |
|-----------|---------|-----------|
| **색상 시스템** | HSB/HSL, RGB, 명명된 팔레트, 절차적 조화, 그라디언트 보간 | `references/color-systems.md` |
| **노이즈 어휘** | Perlin 노이즈, simplex, 프랙탈(옥타브), 도메인 워핑, curl 노이즈 | `references/visual-effects.md` § 노이즈 |
| **파티클 시스템** | 물리 기반, 군집, 트레일 그리기, 어트랙터 기반, 플로 필드 추종 | `references/visual-effects.md` § 파티클 |
| **형태 언어** | 기하학적 기본 도형, 사용자 정의 정점, 베지어 곡선, SVG 경로 | `references/shapes-and-geometry.md` |
| **움직임 스타일** | 이징, 스프링 기반, 노이즈 기반, 물리 시뮬레이션, lerp, 스텝 | `references/animation.md` |
| **타이포그래피** | 시스템 폰트, 로드한 OTF, `textToPoints()` 파티클 텍스트, 키네틱 | `references/typography.md` |
| **셰이더 효과** | GLSL 프래그먼트/버텍스, 필터 셰이더, 후처리, 피드백 루프 | `references/webgl-and-3d.md` § 셰이더 |
| **구성** | 그리드, 방사형, 황금비, 삼분할, 유기적 산포, 타일 | `references/core-api.md` § 구성 |
| **상호작용 모델** | 마우스 추종, 클릭 생성, 드래그, 키보드 상태, 스크롤 기반, 마이크 입력 | `references/interaction.md` |
| **블렌드 모드** | `BLEND`, `ADD`, `MULTIPLY`, `SCREEN`, `DIFFERENCE`, `EXCLUSION`, `OVERLAY` | `references/color-systems.md` § 블렌드 모드 |
| **레이어링** | `createGraphics()` 오프스크린 버퍼, 알파 합성, 마스킹 | `references/core-api.md` § 오프스크린 버퍼 |
| **텍스처** | Perlin 표면, 점묘, 해칭, 하프톤, 픽셀 정렬 | `references/visual-effects.md` § 텍스처 생성 |

### 프로젝트별 변형 규칙

기본 설정은 절대 사용하지 마세요. 모든 프로젝트에서 다음을 지킵니다:
- **사용자 정의 색상 팔레트** — 원시 `fill(255, 0, 0)`는 절대 사용하지 마세요. 3~7색으로 설계한 팔레트를 항상 사용하세요.
- **사용자 정의 선 두께 어휘** — 가는 강조(0.5), 중간 구조(1-2), 굵은 강조(3-5)
- **배경 처리** — 평범한 `background(0)` 또는 `background(255)`는 절대 사용하지 마세요. 항상 텍스처, 그라디언트 또는 레이어를 사용하세요.
- **움직임의 다양성** — 요소마다 속도를 다르게 하세요. 주 요소는 1배, 보조 요소는 0.3배, 주변 요소는 0.1배로 설정합니다.
- **최소 하나의 발명 요소** — 사용자 정의 파티클 행동, 새로운 노이즈 적용, 독특한 상호작용 반응 중 하나를 포함합니다.

### 프로젝트별 발명

모든 프로젝트에서 다음 중 하나 이상을 발명합니다:
- 분위기에 맞는 사용자 정의 색상 팔레트(프리셋이 아님)
- 새로운 노이즈 필드 조합(예: curl 노이즈 + 도메인 워프 + 피드백)
- 독특한 파티클 행동(사용자 정의 힘, 사용자 정의 트레일, 사용자 정의 생성)
- 사용자가 요청하지 않았지만 작품을 끌어올리는 상호작용 메커니즘
- 시각적 위계를 만드는 구성 기법

### 매개변수 설계 철학

매개변수는 일반적인 메뉴가 아니라 알고리즘에서 도출되어야 합니다. "이 시스템의 어떤 속성을 조정할 수 있어야 하는가?"라고 질문하세요.

**좋은 매개변수**는 알고리즘의 성격을 드러냅니다:
- **수량** — 파티클, 가지, 셀의 개수(밀도 조절)
- **스케일** — 노이즈 주파수, 요소 크기, 간격(텍스처 조절)
- **속도** — 속도, 성장률, 감쇠(에너지 조절)
- **임계값** — 행동은 언제 바뀌는가?(극적 효과 조절)
- **비율** — 비례, 힘 사이의 균형(조화 조절)

**나쁜 매개변수**는 알고리즘과 관계없는 일반적인 제어입니다:
- "color1", "color2", "size" — 맥락이 없으면 의미가 없습니다.
- 서로 관련 없는 효과의 토글 스위치
- 행동이 아니라 외형만 바꾸는 매개변수

모든 매개변수는 알고리즘이 *생각하는 방식*을 바꿔야 하며, 단지 *보이는 방식*만 바꿔서는 안 됩니다. 노이즈 옥타브를 바꾸는 "turbulence" 매개변수는 좋습니다. `ellipse()` 반지름만 바꾸는 "particle size" 슬라이더는 피상적입니다.

## 워크플로

### 1단계: 창작 비전

코드를 작성하기 전에 다음을 설명합니다:

- **분위기 / 기운**: 보는 사람이 무엇을 느껴야 하나요? 사색적? 활기찬? 불안한? 장난스러운?
- **시각적 이야기**: 시간의 흐름(또는 상호작용)에 따라 무엇이 일어나나요? 쌓이나요? 쇠퇴하나요? 변형되나요? 진동하나요?
- **색의 세계**: 따뜻한가요/차가운가요? 단색인가요? 보색인가요? 주조색은 무엇인가요? 강조색은 무엇인가요?
- **형태 언어**: 유기적 곡선인가요? 날카로운 기하학인가요? 점인가요? 선인가요? 혼합인가요?
- **움직임의 어휘**: 느린 흐름인가요? 폭발적인 분출인가요? 숨 쉬는 맥동인가요? 기계적인 정밀함인가요?
- **이 스케치를 차별화하는 것**: 이 스케치를 독특하게 만드는 한 가지는 무엇인가요?

사용자의 프롬프트를 미학적 선택으로 매핑합니다. "편안한 생성 배경"은 "글리치 데이터 시각화"와 모든 것이 달라야 합니다.

### 2단계: 기술 설계

- **모드** — 위 표의 7개 모드 중 하나
- **캔버스 크기** — 가로 1920x1080, 세로 1080x1920, 정사각형 1080x1080 또는 반응형 `windowWidth/windowHeight`
- **렌더러** — `P2D`(기본값) 또는 `WEBGL`(3D, 셰이더, 고급 블렌드 모드용)
- **프레임 레이트** — 60fps(인터랙티브), 30fps(앰비언트 애니메이션) 또는 `noLoop()`(정적 생성)
- **내보내기 대상** — 브라우저 표시, PNG 정지 이미지, GIF 루프, MP4 동영상, SVG 벡터
- **상호작용 모델** — 수동(입력 없음), 마우스 기반, 키보드 기반, 오디오 반응형, 스크롤 기반
- **뷰어 UI** — 인터랙티브 생성 예술의 경우 시드 탐색, 매개변수 슬라이더, 다운로드를 제공하는 `templates/viewer.html`에서 시작하세요. 단순한 스케치나 동영상 내보내기에는 기본 HTML을 사용하세요.

### 3단계: 스케치 코드 작성

**인터랙티브 생성 예술**(시드 탐색, 매개변수 조정)의 경우 `templates/viewer.html`에서 시작하세요. 먼저 템플릿을 읽고 고정된 섹션(시드 탐색, 작업)을 유지한 채 알고리즘과 매개변수 제어를 교체하세요. 이렇게 하면 시드 이전/다음/무작위/이동, 실시간 업데이트가 되는 매개변수 슬라이더, PNG 다운로드가 모두 연결된 상태로 사용자에게 제공됩니다.

**애니메이션, 동영상 내보내기 또는 단순한 스케치**에는 기본 HTML을 사용하세요:

단일 HTML 파일. 구조:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Name</title>
  <script>p5.disableFriendlyErrors = true;</script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js"></script>
  <!-- <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/addons/p5.sound.min.js"></script> -->
  <!-- <script src="https://unpkg.com/p5.js-svg@1.6.0"></script> -->  <!-- SVG export -->
  <!-- <script src="https://cdn.jsdelivr.net/npm/ccapture.js-npmfixed/build/CCapture.all.min.js"></script> -->  <!-- video capture -->
  <style>
    html, body { margin: 0; padding: 0; overflow: hidden; }
    canvas { display: block; }
  </style>
</head>
<body>
<script>
// === Configuration ===
const CONFIG = {
  seed: 42,
  // ... project-specific params
};

// === Color Palette ===
const PALETTE = {
  bg: '#0a0a0f',
  primary: '#e8d5b7',
  // ...
};

// === Global State ===
let particles = [];

// === Preload (fonts, images, data) ===
function preload() {
  // font = loadFont('...');
}

// === Setup ===
function setup() {
  createCanvas(1920, 1080);
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  colorMode(HSB, 360, 100, 100, 100);
  // Initialize state...
}

// === Draw Loop ===
function draw() {
  // Render frame...
}

// === Helper Functions ===
// ...

// === Classes ===
class Particle {
  // ...
}

// === Event Handlers ===
function mousePressed() { /* ... */ }
function keyPressed() { /* ... */ }
function windowResized() { resizeCanvas(windowWidth, windowHeight); }
</script>
</body>
</html>
```

주요 구현 패턴:
- **시드 기반 무작위성**: 재현성을 위해 항상 `randomSeed()` + `noiseSeed()`를 사용합니다.
- **색상 모드**: 직관적인 색상 제어를 위해 `colorMode(HSB, 360, 100, 100, 100)`을 사용합니다.
- **상태 분리**: 매개변수는 CONFIG에, 색상은 PALETTE에, 변경 가능한 상태는 전역 변수에 둡니다.
- **클래스 기반 엔티티**: `update()` + `display()` 메서드를 갖는 Particle, agent, shape 클래스를 사용합니다.
- **오프스크린 버퍼**: 계층적 구성, 트레일, 마스크에는 `createGraphics()`를 사용합니다.

### 4단계: 미리보기 및 반복

- HTML 파일을 브라우저에서 직접 엽니다 — 기본 스케치에는 서버가 필요하지 않습니다.
- 로컬 파일에서 `loadImage()`/`loadFont()`를 사용하는 경우 `scripts/serve.sh` 또는 `python3 -m http.server`를 사용합니다.
- Chrome DevTools Performance 탭에서 60fps를 확인합니다.
- 창 크기뿐 아니라 목표 내보내기 해상도에서 테스트합니다.
- 1단계의 콘셉트와 시각적 결과가 일치할 때까지 매개변수를 조정합니다.

### 5단계: 내보내기

| 형식 | 방법 | 명령 |
|--------|--------|---------|
| **PNG** | `keyPressed()`에서 `saveCanvas('output', 'png')` | 's'를 눌러 저장 |
| **고해상도 PNG** | Puppeteer 헤드리스 캡처 | `node scripts/export-frames.js sketch.html --width 3840 --height 2160 --frames 1` |
| **GIF** | `saveGif('output', 5)` — N초 캡처 | 'g'를 눌러 저장 |
| **프레임 시퀀스** | `saveFrames('frame', 'png', 10, 30)` — 30fps로 10초 | 그런 다음 `ffmpeg -i frame-%04d.png -c:v libx264 output.mp4` |
| **MP4** | Puppeteer 프레임 캡처 + ffmpeg | `bash scripts/render.sh sketch.html output.mp4 --duration 30 --fps 30` |
| **SVG** | p5.js-svg를 사용한 `createCanvas(w, h, SVG)` | `save('output.svg')` |

### 6단계: 품질 검증

- **비전과 일치하나요?** 출력을 창작 콘셉트와 비교합니다. 평범해 보이면 1단계로 돌아갑니다.
- **해상도 확인**: 목표 표시 크기에서 선명한가요? 앨리어싱 아티팩트가 없나요?
- **성능 확인**: 브라우저에서 60fps를 유지하나요? (애니메이션은 최소 30fps)
- **색상 확인**: 색상이 서로 어울리나요? 밝은 모니터와 어두운 모니터 모두에서 테스트합니다.
- **경계 사례**: 캔버스 가장자리에서는 어떻게 되나요? 크기를 조정하면? 10분 동안 실행한 뒤에는?

## 핵심 구현 참고 사항

### 성능 — 먼저 FES 비활성화

친화적 오류 시스템(FES)은 최대 10배의 오버헤드를 추가합니다. 모든 프로덕션 스케치에서 비활성화하세요:

```javascript
p5.disableFriendlyErrors = true;  // BEFORE setup()

function setup() {
  pixelDensity(1);  // prevent 2x-4x overdraw on retina
  createCanvas(1920, 1080);
}
```

핫 루프(파티클, 픽셀 연산)에서는 p5 래퍼보다 `Math.*`를 사용하세요 — 측정 가능한 수준으로 더 빠릅니다:

```javascript
// In draw() or update() hot paths:
let a = Math.sin(t);          // not sin(t)
let r = Math.sqrt(dx*dx+dy*dy); // not dist() — or better: skip sqrt, compare magSq
let v = Math.random();        // not random() — when seed not needed
let m = Math.min(a, b);       // not min(a, b)
```

`draw()` 안에서는 절대 `console.log()`를 호출하지 마세요. `draw()` 안에서 DOM을 조작하지 마세요. `references/troubleshooting.md` § 성능을 참고하세요.

### 시드 기반 무작위성 — 항상 사용

모든 생성 스케치는 재현 가능해야 합니다. 같은 시드, 같은 출력입니다.

```javascript
function setup() {
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  // All random() and noise() calls now deterministic
}
```

생성 콘텐츠에 `Math.random()`을 절대 사용하지 마세요 — 성능이 중요한 비시각적 코드에만 사용합니다. 시각적 요소에는 항상 `random()`을 사용하세요. 무작위 시드가 필요하다면 `CONFIG.seed = floor(random(99999))`로 설정하세요.

### 생성 예술 플랫폼 지원(fxhash / Art Blocks)

생성 예술 플랫폼에서는 p5의 PRNG를 플랫폼의 결정론적 무작위 함수로 교체하세요:

```javascript
// fxhash convention
const SEED = $fx.hash;              // unique per mint
const rng = $fx.rand;               // deterministic PRNG
$fx.features({ palette: 'warm', complexity: 'high' });

// In setup():
randomSeed(SEED);   // for p5's noise()
noiseSeed(SEED);

// Replace random() with rng() for platform determinism
let x = rng() * width;  // instead of random(width)
```

`references/export-pipeline.md` § 플랫폼 내보내기를 참고하세요.

### 색상 모드 — HSB 사용

생성 예술에서는 RGB보다 HSB(색상, 채도, 명도)를 다루기가 훨씬 쉽습니다:

```javascript
colorMode(HSB, 360, 100, 100, 100);
// Now: fill(hue, sat, bri, alpha)
// Rotate hue: fill((baseHue + offset) % 360, 80, 90)
// Desaturate: fill(hue, sat * 0.3, bri)
// Darken: fill(hue, sat, bri * 0.5)
```

원시 RGB 값을 하드코딩하지 마세요. 팔레트 객체를 정의하고 변형을 절차적으로 도출하세요. `references/color-systems.md`를 참고하세요.

### 노이즈 — 원시 노이즈가 아닌 다중 옥타브

원시 `noise(x, y)`는 부드러운 얼룩처럼 보입니다. 자연스러운 텍스처를 위해 옥타브를 겹치세요:

```javascript
function fbm(x, y, octaves = 4) {
  let val = 0, amp = 1, freq = 1, sum = 0;
  for (let i = 0; i < octaves; i++) {
    val += noise(x * freq, y * freq) * amp;
    sum += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return val / sum;
}
```

흐르는 유기적 형태에는 **도메인 워핑**을 사용하세요. 노이즈 출력을 노이즈 입력 좌표로 다시 전달하는 방식입니다. `references/visual-effects.md`를 참고하세요.

### 레이어에는 createGraphics() 사용 — 선택 사항이 아님

평평한 단일 패스 렌더링은 평면적으로 보입니다. 구성을 위해 오프스크린 버퍼를 사용하세요:

```javascript
let bgLayer, fgLayer, trailLayer;
function setup() {
  createCanvas(1920, 1080);
  bgLayer = createGraphics(width, height);
  fgLayer = createGraphics(width, height);
  trailLayer = createGraphics(width, height);
}
function draw() {
  renderBackground(bgLayer);
  renderTrails(trailLayer);   // persistent, fading
  renderForeground(fgLayer);  // cleared each frame
  image(bgLayer, 0, 0);
  image(trailLayer, 0, 0);
  image(fgLayer, 0, 0);
}
```

### 성능 — 가능한 곳에서 벡터화

p5.js 그리기 호출은 비용이 큽니다. 수천 개의 파티클에는 다음을 사용하세요:

```javascript
// SLOW: individual shapes
for (let p of particles) {
  ellipse(p.x, p.y, p.size);
}

// FAST: single shape with beginShape()
beginShape(POINTS);
for (let p of particles) {
  vertex(p.x, p.y);
}
endShape();

// FASTEST: pixel buffer for massive counts
loadPixels();
for (let p of particles) {
  let idx = 4 * (floor(p.y) * width + floor(p.x));
  pixels[idx] = r; pixels[idx+1] = g; pixels[idx+2] = b; pixels[idx+3] = 255;
}
updatePixels();
```

`references/troubleshooting.md` § 성능을 참고하세요.

### 여러 스케치에는 인스턴스 모드 사용

전역 모드는 `window`를 오염시킵니다. 프로덕션에서는 인스턴스 모드를 사용하세요:

```javascript
const sketch = (p) => {
  p.setup = function() {
    p.createCanvas(800, 800);
  };
  p.draw = function() {
    p.background(0);
    p.ellipse(p.mouseX, p.mouseY, 50);
  };
};
new p5(sketch, 'canvas-container');
```

한 페이지에 여러 스케치를 삽입하거나 프레임워크와 통합할 때 필요합니다.

### WebGL 모드 주의 사항

- `createCanvas(w, h, WEBGL)` — 원점은 왼쪽 위가 아니라 중앙입니다.
- Y축은 반전됩니다(WebGL에서는 양의 Y가 위로, P2D에서는 아래로 향함).
- P2D와 같은 좌표를 얻으려면 `translate(-width/2, -height/2)`를 사용합니다.
- 모든 변환 주위에 `push()`/`pop()`을 사용하세요 — 행렬 스택 오버플로가 조용히 발생합니다.
- `rect()`/`plane()` 전에 `texture()`를 호출하세요 — 뒤가 아닙니다.
- 사용자 정의 셰이더: `createShader(vert, frag)` — 여러 브라우저에서 테스트하세요.

### 내보내기 — 키 바인딩 규칙

모든 스케치의 `keyPressed()`에 다음을 포함해야 합니다:

```javascript
function keyPressed() {
  if (key === 's' || key === 'S') saveCanvas('output', 'png');
  if (key === 'g' || key === 'G') saveGif('output', 5);
  if (key === 'r' || key === 'R') { randomSeed(millis()); noiseSeed(millis()); }
  if (key === ' ') CONFIG.paused = !CONFIG.paused;
}
```

### 헤드리스 동영상 내보내기 — noLoop() 사용

Puppeteer를 통한 헤드리스 렌더링에서는 스케치가 반드시 `setup`에서 `noLoop()`를 사용해야 합니다. 그렇지 않으면 스크린샷이 느린 동안 p5의 draw 루프가 자유롭게 실행되어 스케치가 앞서 나가고, 프레임이 건너뛰거나 중복됩니다.

```javascript
function setup() {
  createCanvas(1920, 1080);
  pixelDensity(1);
  noLoop();                    // capture script controls frame advance
  window._p5Ready = true;      // signal readiness to capture script
}
```

번들된 `scripts/export-frames.js`는 `_p5Ready`를 감지하고 캡처마다 한 번씩 `redraw()`를 호출하여 정확히 1:1로 프레임을 대응시킵니다. 결정론적 캡처에 대해서는 `references/export-pipeline.md` § 결정론적 캡처를 참고하세요.

여러 장면의 동영상에는 클립별 아키텍처를 사용하세요. 장면마다 HTML 하나를 독립적으로 렌더링한 뒤 `ffmpeg -f concat`으로 연결합니다. `references/export-pipeline.md` § 클립별 아키텍처를 참고하세요.

### 에이전트 워크플로

p5.js 스케치를 만들 때:

1. **HTML 파일 작성** — 단일 독립형 파일, 모든 코드를 인라인으로 작성
2. **브라우저에서 열기** — `open sketch.html`(macOS) 또는 `xdg-open sketch.html`(Linux)
3. **로컬 자산**(폰트, 이미지)은 서버가 필요합니다: 프로젝트 디렉터리에서 `python3 -m http.server 8080`을 실행한 다음 `http://localhost:8080/sketch.html`을 엽니다.
4. **PNG/GIF 내보내기** — 위에 나온 `keyPressed()` 단축키를 추가하고 어떤 키를 누르면 되는지 사용자에게 알려줍니다.
5. **헤드리스 내보내기** — 자동 프레임 캡처에는 `node scripts/export-frames.js sketch.html --frames 300`을 사용합니다(스케치는 `noLoop()` + `_p5Ready`를 사용해야 함).
6. **MP4 렌더링** — `bash scripts/render.sh sketch.html output.mp4 --duration 30`을 사용합니다.
7. **반복 개선** — HTML 파일을 편집하고 사용자가 브라우저를 새로 고쳐 변경 사항을 확인합니다.
8. **필요할 때 참고 자료 로드** — 구현 중 필요에 따라 `skill_view(name="p5js", file_path="references/...")`를 사용하여 특정 참고 파일을 로드합니다.

## 성능 목표

| 지표 | 목표 |
|--------|--------|
| 프레임 레이트(인터랙티브) | 60fps 지속 |
| 프레임 레이트(애니메이션 내보내기) | 최소 30fps |
| 파티클 수(P2D 도형) | 60fps에서 5,000-10,000 |
| 파티클 수(픽셀 버퍼) | 60fps에서 50,000-100,000 |
| 캔버스 해상도 | 최대 3840x2160(내보내기), 1920x1080(인터랙티브) |
| 파일 크기(HTML) | &lt; 100KB(CDN 라이브러리 제외) |
| 로드 시간 | 첫 프레임까지 &lt; 2초 |

## 참고 자료

| 파일 | 내용 |
|------|----------|
| `references/core-api.md` | 캔버스 설정, 좌표계, draw 루프, `push()`/`pop()`, 오프스크린 버퍼, 구성 패턴, `pixelDensity()`, 반응형 디자인 |
| `references/shapes-and-geometry.md` | 2D 기본 도형, `beginShape()`/`endShape()`, 베지어/Catmull-Rom 곡선, `vertex()` 시스템, 사용자 정의 도형, `p5.Vector`, 부호 있는 거리장, SVG 경로 변환 |
| `references/visual-effects.md` | 노이즈(Perlin, 프랙탈, 도메인 워프, curl), 플로 필드, 파티클 시스템(물리, 군집, 트레일), 픽셀 조작, 텍스처 생성(점묘, 해칭, 하프톤), 피드백 루프, 반응-확산 |
| `references/animation.md` | 프레임 기반 애니메이션, 이징 함수, `lerp()`/`map()`, 스프링 물리, 상태 머신, 타임라인 시퀀싱, `millis()` 기반 타이밍, 트랜지션 패턴 |
| `references/typography.md` | `text()`, `loadFont()`, `textToPoints()`, 키네틱 타이포그래피, 텍스트 마스크, 폰트 메트릭, 반응형 텍스트 크기 조정 |
| `references/color-systems.md` | `colorMode()`, HSB/HSL/RGB, `lerpColor()`, `paletteLerp()`, 절차적 팔레트, 색상 조화, `blendMode()`, 그라디언트 렌더링, 엄선된 팔레트 라이브러리 |
| `references/webgl-and-3d.md` | WEBGL 렌더러, 3D 기본 도형, 카메라, 조명, 머티리얼, 사용자 정의 지오메트리, GLSL 셰이더(`createShader()`, `createFilterShader()`), 프레임버퍼, 후처리 |
| `references/interaction.md` | 마우스 이벤트, 키보드 상태, 터치 입력, DOM 요소, `createSlider()`/`createButton()`, 오디오 입력(p5.sound FFT/진폭), 스크롤 기반 애니메이션, 반응형 이벤트 |
| `references/export-pipeline.md` | `saveCanvas()`, `saveGif()`, `saveFrames()`, 결정론적 헤드리스 캡처, ffmpeg 프레임-동영상 변환, CCapture.js, SVG 내보내기, 클립별 아키텍처, 플랫폼 내보내기(fxhash), 동영상 주의 사항 |
| `references/troubleshooting.md` | 성능 프로파일링, 픽셀별 예산, 일반적인 실수, 브라우저 호환성, WebGL 디버깅, 폰트 로딩 문제, 픽셀 밀도 함정, 메모리 누수, CORS |
| `templates/viewer.html` | 인터랙티브 뷰어 템플릿: 시드 탐색(이전/다음/무작위/이동), 매개변수 슬라이더, PNG 다운로드, 반응형 캔버스. 탐색 가능한 생성 예술은 여기서 시작하세요. |

---

## 크리에이티브 발산(사용자가 실험적/창의적/독특한 출력을 요청한 경우에만 사용)

사용자가 창의적이거나, 실험적이거나, 놀랍거나, 색다른 출력을 요청했다면 가장 적합한 전략을 선택하고 코드를 생성하기 **전에** 그 단계들을 검토합니다.

- **개념적 혼합** — 사용자가 두 가지를 결합하라고 하거나 혼합 미학을 원할 때
- **SCAMPER** — 알려진 생성 예술 패턴에 변주를 원할 때
- **거리 연상** — 사용자가 하나의 콘셉트를 주고 탐색을 원할 때("시간에 관한 무언가를 만들어줘")

### 개념적 혼합
1. 서로 다른 두 시각 시스템을 이름 붙입니다(예: 파티클 물리 + 손글씨)
2. 대응 관계를 매핑합니다(파티클 = 잉크 방울, 힘 = 펜 압력, 필드 = 글자 형태)
3. 선택적으로 혼합합니다 — 흥미로운 창발을 만드는 매핑을 유지합니다.
4. 혼합물을 나란히 놓인 두 시스템이 아니라 하나의 통합된 시스템으로 코딩합니다.

### SCAMPER 변환
알려진 생성 패턴(플로 필드, 파티클 시스템, 흐름도, 셀룰러 오토마타)을 취해 체계적으로 변환합니다:
- **대체**: 원을 텍스트 문자로, 선을 그라디언트로 교체
- **결합**: 두 패턴을 합치기(플로 필드 + 보로노이)
- **적용**: 2D 패턴을 3D 투영에 적용
- **변형**: 스케일을 과장하고 좌표 공간을 왜곡
- **용도 변경**: 타이포그래피에 물리 시뮬레이션을, 색상에 정렬 알고리즘을 사용
- **제거**: 그리드를 제거하고, 색을 제거하고, 대칭을 제거
- **반전**: 시뮬레이션을 거꾸로 실행하고, 매개변수 공간을 반전

### 거리 연상
1. 사용자의 콘셉트(예: "외로움")를 기준점으로 삼습니다.
2. 세 가지 거리에서 연상을 생성합니다:
   - 가까움(명백함): 빈 방, 한 사람, 침묵
   - 중간(흥미로움): 무리와 반대 방향으로 헤엄치는 물고기 한 마리, 알림이 하나도 없는 휴대전화, 지하철 차량 사이의 틈
   - 멂(추상적): 소수, 점근 곡선, 새벽 3시의 색
3. 중간 거리의 연상을 발전시킵니다 — 시각화할 만큼 구체적이면서도 흥미로울 만큼 예상 밖인 것들입니다.
