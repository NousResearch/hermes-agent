---
title: "P5Js — p5"
sidebar_label: "P5Js"
description: "p5"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# P5Js

p5.js 스케치: 생성 예술, 셰이더, 인터랙티브, 3D.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/p5js` |
| 버전 | `1.0.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `creative-coding`, `generative-art`, `p5js`, `canvas`, `interactive`, `visualization`, `webgl`, `shaders`, `animation` |
| 관련 스킬 | [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# p5.js 프로덕션 파이프라인

## 사용 시기

사용자가 다음과 같이 요청할 때 사용합니다: p5.js 스케치, 크리에이티브 코딩, 생성 예술, 인터랙티브 시각화, 캔버스 애니메이션, 브라우저 기반 시각 예술, 데이터 시각화, 셰이더 효과 또는 모든 p5.js 프로젝트.

## 포함된 내용

p5.js를 사용한 인터랙티브 및 생성 시각 예술을 위한 프로덕션 파이프라인. 브라우저 기반 스케치, 생성 예술, 데이터 시각화, 인터랙티브 경험, 3D 씬, 오디오 반응형 시각 자료 및 모션 그래픽을 생성하며 — HTML, PNG, GIF, MP4 또는 SVG로 내보냅니다. 2D/3D 렌더링, 노이즈 및 파티클 시스템, 흐름장(flow fields), 셰이더(GLSL), 픽셀 조작, 키네틱 타이포그래피, WebGL 씬, 오디오 분석, 마우스/키보드 상호작용 및 헤드리스 고해상도 내보내기를 다룹니다.

## 크리에이티브 기준

이것은 브라우저에서 렌더링되는 시각 예술입니다. 캔버스는 매체이며, 알고리즘은 붓입니다.

**코드 한 줄을 작성하기 전에** 크리에이티브 컨셉을 명확히 하세요. 이 작품은 무엇을 소통하나요? 시청자가 스크롤을 멈추게 만드는 것은 무엇인가요? 이 작품을 단순한 코드 튜토리얼 예제와 차별화하는 점은 무엇인가요? 사용자의 프롬프트는 출발점일 뿐입니다 — 창의적인 야망을 가지고 해석하세요.

**첫 번째 렌더링의 탁월함은 타협할 수 없습니다.** 출력물은 처음 로드될 때 시각적으로 인상적이어야 합니다. p5.js 튜토리얼 연습문제나, 기본 설정, 또는 "AI가 생성한 크리에이티브 코딩"처럼 보인다면 잘못된 것입니다. 결과물을 내놓기 전에 다시 생각하세요.

**레퍼런스 어휘를 넘어서세요.** 참조 자료에 있는 노이즈 함수, 파티클 시스템, 색상 팔레트, 셰이더 효과는 시작하기 위한 어휘입니다. 모든 프로젝트에서 이들을 결합하고, 겹치고, 발명하세요. 카탈로그는 물감의 팔레트일 뿐이며, 그림은 당신이 그리는 것입니다.

**주도적으로 창의성을 발휘하세요.** 사용자가 "파티클 시스템"을 요청한다면, 창발적인 무리 행동(flocking behavior), 흔적을 남기는 유령 같은 에코, 팔레트가 변화하는 깊이 안개, 그리고 숨쉬는 배경 노이즈 필드가 있는 파티클 시스템을 제공하세요. 사용자가 요청하지 않았지만 전체 작품의 수준을 높일 수 있는 시각적 디테일을 최소한 하나 이상 포함하세요.

**밀도 있고, 층이 있으며, 심사숙고해야 합니다.** 모든 프레임은 볼 가치가 있어야 합니다. 절대 평면적인 흰색 배경을 사용하지 마세요. 항상 구성적 계층 구조를 가지세요. 항상 의도적인 색상을 사용하세요. 항상 자세히 보아야만 나타나는 미세한 디테일을 포함하세요.

**기능 개수보다 일관된 미학이 중요합니다.** 모든 요소는 공유된 색온도, 일관된 선 굵기 어휘, 조화로운 움직임 속도 등 통일된 시각적 언어를 제공해야 합니다. 10개의 연관 없는 효과가 있는 스케치는 잘 어울리는 3개의 효과가 있는 스케치보다 나쁩니다.

## 모드

| 모드 | 입력 | 출력 | 참조 |
|------|-------|--------|-----------|
| **Generative art** | 시드 / 파라미터 | 절차적 시각적 구성 (정지 또는 애니메이션) | `references/visual-effects.md` |
| **Data visualization** | 데이터셋 / API | 인터랙티브 차트, 그래프, 맞춤형 데이터 표시 | `references/interaction.md` |
| **Interactive experience** | 없음 (사용자 구동) | 마우스/키보드/터치 구동 스케치 | `references/interaction.md` |
| **Animation / motion graphics** | 타임라인 / 스토리보드 | 타이밍 시퀀스, 키네틱 타이포그래피, 트랜지션 | `references/animation.md` |
| **3D scene** | 컨셉 설명 | WebGL 기하학, 조명, 카메라, 재질 | `references/webgl-and-3d.md` |
| **Image processing** | 이미지 파일 | 픽셀 조작, 필터, 모자이크, 점묘법 | `references/visual-effects.md` § Pixel Manipulation |
| **Audio-reactive** | 오디오 파일 / 마이크 | 소리에 구동되는 생성형 비주얼 | `references/interaction.md` § Audio Input |

## 스택

프로젝트당 단일 독립형 HTML 파일. 빌드 단계가 필요하지 않습니다.

| 계층 | 도구 | 목적 |
|-------|------|---------|
| Core | p5.js 1.11.3 (CDN) | 캔버스 렌더링, 수학, 변환, 이벤트 처리 |
| 3D | p5.js WebGL mode | 3D 기하학, 카메라, 조명, GLSL 셰이더 |
| Audio | p5.sound.js (CDN) | FFT 분석, 진폭, 마이크 입력, 오실레이터 |
| Export | 내장 `saveCanvas()` / `saveGif()` / `saveFrames()` | PNG, GIF, 프레임 시퀀스 출력 |
| Capture | CCapture.js (선택 사항) | 확정적(Deterministic) 프레임레이트 비디오 캡처 (WebM, GIF) |
| Headless | Puppeteer + Node.js (선택 사항) | 자동화된 고해상도 렌더링, ffmpeg를 통한 MP4 |
| SVG | p5.js-svg 1.6.0 (선택 사항) | 인쇄용 벡터 출력 — p5.js 1.x 필요 |
| Natural media | p5.brush (선택 사항) | 수채화, 목탄, 펜 — p5.js 2.x + WEBGL 필요 |
| Texture | p5.grain (선택 사항) | 필름 그레인, 텍스처 오버레이 |
| Fonts | Google Fonts / `loadFont()` | OTF/TTF/WOFF2를 통한 맞춤형 타이포그래피 |

### 버전 참고

**p5.js 1.x** (1.11.3)이 기본값입니다 — 안정적이고 문서화가 잘 되어 있으며 라이브러리 호환성이 가장 넓습니다. 프로젝트에서 2.x 기능이 특별히 필요하지 않은 한 이것을 사용하세요.

**p5.js 2.x** (2.2+) 추가 기능: `preload()`를 대체하는 `async setup()`, OKLCH/OKLAB 색상 모드, `splineVertex()`, 셰이더 `.modify()` API, 가변 폰트, `textToContours()`, 포인터 이벤트. p5.brush에 필요합니다. `references/core-api.md` § p5.js 2.0 참조.

## 파이프라인

모든 프로젝트는 동일한 6단계 경로를 따릅니다:

```
CONCEPT → DESIGN → CODE → PREVIEW → EXPORT → VERIFY
```

1. **CONCEPT** — 크리에이티브 비전을 명확히 합니다: 무드, 색상의 세계, 모션 어휘, 무엇이 독특하게 만드는지.
2. **DESIGN** — 모드, 캔버스 크기, 상호작용 모델, 색상 시스템, 내보내기 형식을 선택합니다. 컨셉을 기술적 결정에 매핑합니다.
3. **CODE** — 인라인 p5.js로 단일 HTML 파일을 작성합니다. 구조: 전역 변수 → `preload()` → `setup()` → `draw()` → 도우미 함수 → 클래스 → 이벤트 핸들러.
4. **PREVIEW** — 브라우저에서 열고 시각적 품질을 확인합니다. 대상 해상도에서 테스트합니다. 성능을 확인합니다.
5. **EXPORT** — 출력을 캡처합니다: PNG의 경우 `saveCanvas()`, GIF의 경우 `saveGif()`, MP4의 경우 `saveFrames()` + ffmpeg, 헤드리스 일괄 처리의 경우 Puppeteer.
6. **VERIFY** — 출력이 컨셉과 일치하나요? 의도한 디스플레이 크기에서 시각적으로 인상적인가요? 액자에 넣을 만한가요?

## 크리에이티브 디렉션

### 미학적 차원

| 차원 | 옵션 | 참조 |
|-----------|---------|-----------|
| **색상 시스템** | HSB/HSL, RGB, 명명된 팔레트, 절차적 조화, 그라데이션 보간 | `references/color-systems.md` |
| **노이즈 어휘** | 펄린(Perlin) 노이즈, 심플렉스(Simplex), 프랙탈(옥타브), 도메인 워핑, 컬(Curl) 노이즈 | `references/visual-effects.md` § Noise |
| **파티클 시스템** | 물리 기반, 무리 지어 날기(flocking), 궤적 그리기, 끌개 기반, 흐름장(flow-field) 추종 | `references/visual-effects.md` § Particles |
| **도형 언어** | 기하학적 기본형, 맞춤형 정점, 베지어 곡선, SVG 패스 | `references/shapes-and-geometry.md` |
| **모션 스타일** | 이징(eased), 스프링 기반, 노이즈 구동, 물리 시뮬레이션, 보간(lerped), 단계별(stepped) | `references/animation.md` |
| **타이포그래피** | 시스템 폰트, 로드된 OTF, `textToPoints()` 파티클 텍스트, 키네틱 | `references/typography.md` |
| **셰이더 효과** | GLSL 프래그먼트/버텍스, 필터 셰이더, 포스트 프로세싱, 피드백 루프 | `references/webgl-and-3d.md` § Shaders |
| **구성 (Composition)** | 그리드, 방사형, 황금비, 3분할, 유기적 산포, 타일링 | `references/core-api.md` § Composition |
| **상호작용 모델** | 마우스 따라가기, 클릭 생성, 드래그, 키보드 상태, 스크롤 구동, 마이크 입력 | `references/interaction.md` |
| **블렌드 모드** | `BLEND`, `ADD`, `MULTIPLY`, `SCREEN`, `DIFFERENCE`, `EXCLUSION`, `OVERLAY` | `references/color-systems.md` § Blend Modes |
| **레이어링** | `createGraphics()` 오프스크린 버퍼, 알파 합성, 마스킹 | `references/core-api.md` § Offscreen Buffers |
| **텍스처** | 펄린 표면, 점묘(stippling), 해칭(hatching), 하프톤, 픽셀 소팅 | `references/visual-effects.md` § Texture Generation |

### 프로젝트별 변주 규칙

기본 구성을 절대 사용하지 마세요. 모든 프로젝트에 대해:
- **맞춤형 색상 팔레트** — 절대 원시 `fill(255, 0, 0)`을 사용하지 마세요. 항상 3-7개의 색상으로 설계된 팔레트를 사용하세요.
- **맞춤형 선 굵기 어휘** — 얇은 강조(0.5), 중간 구조(1-2), 굵은 강조(3-5).
- **배경 처리** — 절대 단순한 `background(0)` 또는 `background(255)`를 사용하지 마세요. 항상 텍스처, 그라데이션 또는 레이어를 사용하세요.
- **다양한 움직임** — 요소에 따라 다른 속도를 적용하세요. 기본은 1x, 부차적인 것은 0.3x, 배경은 0.1x.
- **최소한 하나 이상의 발명 요소** — 맞춤형 파티클 동작, 새로운 노이즈 적용, 독특한 상호작용 반응 등.

### 프로젝트 특화 발명

모든 프로젝트에 대해 다음 중 최소 하나를 발명하세요:
- 분위기에 맞는 맞춤형 색상 팔레트 (프리셋 아님)
- 참신한 노이즈 필드 조합 (예: 컬 노이즈 + 도메인 워프 + 피드백)
- 독특한 파티클 동작 (맞춤형 힘, 맞춤형 궤적, 맞춤형 생성)
- 사용자가 요청하지 않았지만 작품을 향상시키는 상호작용 메커니즘
- 시각적 계층 구조를 만드는 구성 기법

### 파라미터 설계 철학

파라미터는 일반적인 메뉴가 아니라 알고리즘에서 나와야 합니다. "이 시스템의 어떤 속성이 조정 가능해야 하는가?"라고 질문하세요.

**좋은 파라미터**는 알고리즘의 특성을 드러냅니다:
- **수량(Quantities)** — 얼마나 많은 파티클, 가지, 셀 (밀도 제어)
- **규모(Scales)** — 노이즈 빈도, 요소 크기, 간격 (텍스처 제어)
- **비율(Rates)** — 속도, 성장률, 감쇠 (에너지 제어)
- **임계값(Thresholds)** — 언제 행동이 바뀌는가? (드라마 제어)
- **비(Ratios)** — 비율, 힘 사이의 균형 (조화 제어)

**나쁜 파라미터**는 알고리즘과 무관한 일반적인 컨트롤입니다:
- "color1", "color2", "size" — 컨텍스트가 없으면 의미가 없음
- 관련 없는 효과에 대한 토글 스위치
- 행동이 아닌 외관만 바꾸는 파라미터

모든 파라미터는 단순히 어떻게 *보이는지*가 아니라 알고리즘이 어떻게 *생각하는지*를 바꿔야 합니다. 노이즈 옥타브를 바꾸는 "난기류(turbulence)" 파라미터는 좋습니다. `ellipse()` 반경만 바꾸는 "파티클 크기" 슬라이더는 얕은(shallow) 파라미터입니다.

## 워크플로우

### 1단계: 크리에이티브 비전

코드를 작성하기 전에 명확히 하세요:

- **무드 / 분위기**: 시청자가 무엇을 느껴야 하나요? 명상적? 에너제틱? 불안함? 장난스러움?
- **시각적 스토리**: 시간(또는 상호작용)이 지남에 따라 어떤 일이 일어나나요? 형성? 붕괴? 변형? 진동?
- **색상의 세계**: 따뜻함/차가움? 흑백? 보색? 주된 색조는 무엇인가요? 포인트 색상은?
- **도형 언어**: 유기적인 곡선? 날카로운 기하학? 점? 선? 혼합?
- **모션 어휘**: 느린 표류? 폭발적인 버스트? 숨쉬는 펄스? 기계적인 정밀함?
- **이것을 다르게 만드는 것**: 이 스케치를 독특하게 만드는 한 가지는 무엇인가요?

사용자의 프롬프트를 미학적 선택과 매핑하세요. "편안한 생성 배경"은 "글리치 데이터 시각화"와 모든 면에서 달라야 합니다.

### 2단계: 기술적 설계

- **모드** — 위의 7가지 모드 중 어느 것인지
- **캔버스 크기** — 가로 1920x1080, 세로 1080x1920, 정사각형 1080x1080, 또는 반응형 `windowWidth/windowHeight`
- **렌더러** — `P2D` (기본값) 또는 `WEBGL` (3D, 셰이더, 고급 블렌드 모드용)
- **프레임레이트** — 60fps (인터랙티브), 30fps (앰비언트 애니메이션), 또는 `noLoop()` (정적 생성형)
- **내보내기 타겟** — 브라우저 표시, PNG 정지, GIF 루프, MP4 비디오, SVG 벡터
- **상호작용 모델** — 수동적 (입력 없음), 마우스 구동, 키보드 구동, 오디오 반응형, 스크롤 구동
- **뷰어 UI** — 인터랙티브 생성 예술의 경우 시드 내비게이션, 파라미터 슬라이더, 다운로드를 제공하는 `templates/viewer.html`에서 시작합니다. 단순한 스케치나 비디오 수출의 경우 일반 HTML을 사용합니다.

### 3단계: 스케치 코드 작성

**인터랙티브 생성 예술** (시드 탐색, 파라미터 튜닝)의 경우: `templates/viewer.html`에서 시작하세요. 먼저 템플릿을 읽고 고정된 섹션(시드 탐색, 액션)을 유지한 다음, 알고리즘과 파라미터 컨트롤을 교체하세요. 이를 통해 사용자는 시드 이전/다음/랜덤/점프, 실시간 업데이트가 가능한 파라미터 슬라이더, PNG 다운로드 기능을 모두 연결된 상태로 사용할 수 있습니다.

**애니메이션, 비디오 내보내기, 또는 단순 스케치**의 경우: 일반 HTML을 사용합니다.

단일 HTML 파일. 구조:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>프로젝트 이름</title>
  <script>p5.disableFriendlyErrors = true;</script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js"></script>
  <!-- <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/addons/p5.sound.min.js"></script> -->
  <!-- <script src="https://unpkg.com/p5.js-svg@1.6.0"></script> -->  <!-- SVG 내보내기 -->
  <!-- <script src="https://cdn.jsdelivr.net/npm/ccapture.js-npmfixed/build/CCapture.all.min.js"></script> -->  <!-- 비디오 캡처 -->
  <style>
    html, body { margin: 0; padding: 0; overflow: hidden; }
    canvas { display: block; }
  </style>
</head>
<body>
<script>
// === Configuration (설정) ===
const CONFIG = {
  seed: 42,
  // ... 프로젝트 특화 파라미터
};

// === Color Palette (색상 팔레트) ===
const PALETTE = {
  bg: '#0a0a0f',
  primary: '#e8d5b7',
  // ...
};

// === Global State (전역 상태) ===
let particles = [];

// === Preload (폰트, 이미지, 데이터 등) ===
function preload() {
  // font = loadFont('...');
}

// === Setup (초기 설정) ===
function setup() {
  createCanvas(1920, 1080);
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  colorMode(HSB, 360, 100, 100, 100);
  // 상태 초기화...
}

// === Draw Loop (그리기 루프) ===
function draw() {
  // 프레임 렌더링...
}

// === Helper Functions (도우미 함수) ===
// ...

// === Classes (클래스) ===
class Particle {
  // ...
}

// === Event Handlers (이벤트 핸들러) ===
function mousePressed() { /* ... */ }
function keyPressed() { /* ... */ }
function windowResized() { resizeCanvas(windowWidth, windowHeight); }
</script>
</body>
</html>
```

주요 구현 패턴:
- **시드 기반 무작위성(Seeded randomness)**: 재현 가능성을 위해 항상 `randomSeed()` + `noiseSeed()`를 사용하세요.
- **색상 모드**: 직관적인 색상 제어를 위해 `colorMode(HSB, 360, 100, 100, 100)`을 사용하세요.
- **상태 분리**: 매개변수를 위한 CONFIG, 색상을 위한 PALETTE, 가변 상태를 위한 전역 변수를 분리하세요.
- **클래스 기반 엔티티**: `update()` + `display()` 메서드를 가지는 클래스로 파티클, 에이전트, 모양을 정의하세요.
- **오프스크린 버퍼**: 층이 있는 구성, 궤적, 마스크를 위해 `createGraphics()`를 사용하세요.

### 4단계: 미리보기 및 반복

- HTML 파일을 브라우저에서 직접 엽니다 — 단순 스케치에는 서버가 필요하지 않습니다.
- 로컬 파일에서 `loadImage()`/`loadFont()`를 사용할 경우: `scripts/serve.sh` 또는 `python3 -m http.server`를 사용합니다.
- Chrome DevTools 성능 탭에서 60fps가 유지되는지 확인합니다.
- 창 크기뿐만 아니라 타겟 내보내기 해상도에서 테스트합니다.
- 1단계의 컨셉과 시각이 일치할 때까지 매개변수를 조정합니다.

### 5단계: 내보내기

| 형식 | 메서드 | 명령 |
|--------|--------|---------|
| **PNG** | `keyPressed()` 안의 `saveCanvas('output', 'png')` | 저장하려면 's' 키를 누르세요 |
| **High-res PNG** | Puppeteer 헤드리스 캡처 | `node scripts/export-frames.js sketch.html --width 3840 --height 2160 --frames 1` |
| **GIF** | `saveGif('output', 5)` — N초 캡처 | 저장하려면 'g' 키를 누르세요 |
| **Frame sequence** | `saveFrames('frame', 'png', 10, 30)` — 30fps로 10초 | 그런 다음 `ffmpeg -i frame-%04d.png -c:v libx264 output.mp4` |
| **MP4** | Puppeteer 프레임 캡처 + ffmpeg | `bash scripts/render.sh sketch.html output.mp4 --duration 30 --fps 30` |
| **SVG** | p5.js-svg를 사용하는 `createCanvas(w, h, SVG)` | `save('output.svg')` |

### 6단계: 품질 검증

- **비전과 일치하는가?** 출력을 크리에이티브 컨셉과 비교하세요. 평범해 보인다면 1단계로 돌아가세요.
- **해상도 확인**: 목표 디스플레이 크기에서 선명한가요? 앨리어싱(계단 현상)이 없는가요?
- **성능 확인**: 브라우저에서 60fps를 유지하나요? (애니메이션의 경우 최소 30fps)
- **색상 확인**: 색상이 서로 잘 어울리나요? 밝은 모니터와 어두운 모니터 모두에서 테스트하세요.
- **예외 상황**: 캔버스 가장자리에서는 어떻게 되나요? 크기를 조절할 때는? 10분 동안 실행된 후에는?

## 주요 구현 노트

### 성능 — FES 먼저 비활성화

Friendly Error System(FES)은 오버헤드를 최대 10배까지 늘립니다. 프로덕션 스케치에서는 비활성화하세요:

```javascript
p5.disableFriendlyErrors = true;  // setup() 전에 작성

function setup() {
  pixelDensity(1);  // 레티나 디스플레이에서 2x-4x 오버드로우 방지
  createCanvas(1920, 1080);
}
```

핫 루프(파티클, 픽셀 작업)에서는 p5 래퍼 대신 `Math.*`를 사용하세요 — 눈에 띄게 빠릅니다:

```javascript
// draw() 또는 update()의 핫 패스 내에서:
let a = Math.sin(t);          // sin(t) 아님
let r = Math.sqrt(dx*dx+dy*dy); // dist() 아님 — 더 좋게: sqrt를 건너뛰고 magSq를 비교
let v = Math.random();        // random() 아님 — 시드가 필요하지 않을 때
let m = Math.min(a, b);       // min(a, b) 아님
```

`draw()` 안에서 절대 `console.log()`를 사용하지 마세요. `draw()` 안에서 절대 DOM을 조작하지 마세요. `references/troubleshooting.md` § Performance를 참조하세요.

### 시드 기반 무작위성 — 항상

모든 생성 스케치는 재현 가능해야 합니다. 같은 시드, 같은 출력.

```javascript
function setup() {
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  // 이제 모든 random() 및 noise() 호출이 결정적(deterministic)입니다.
}
```

생성 콘텐츠에 절대 `Math.random()`을 사용하지 마세요 — 성능에 중요한 비시각적 코드에만 사용하세요. 시각적 요소에는 항상 `random()`을 사용하세요. 임의의 시드가 필요한 경우: `CONFIG.seed = floor(random(99999))`.

### 생성 예술 플랫폼 지원 (fxhash / Art Blocks)

생성 예술 플랫폼의 경우, p5의 PRNG를 플랫폼의 확정적 무작위성으로 대체하세요:

```javascript
// fxhash 규칙
const SEED = $fx.hash;              // 발매(mint)마다 고유함
const rng = $fx.rand;               // 확정적 PRNG
$fx.features({ palette: 'warm', complexity: 'high' });

// setup()에서:
randomSeed(SEED);   // p5의 noise()를 위해
noiseSeed(SEED);

// 플랫폼 확정성을 위해 random()을 rng()로 교체
let x = rng() * width;  // random(width) 대신
```

`references/export-pipeline.md` § Platform Export 참조.

### 색상 모드 — HSB 사용

생성 예술에서는 RGB보다 HSB(Hue, Saturation, Brightness)가 작업하기 훨씬 쉽습니다.

```javascript
colorMode(HSB, 360, 100, 100, 100);
// 이제: fill(hue, sat, bri, alpha)
// Hue 회전: fill((baseHue + offset) % 360, 80, 90)
// 채도 낮추기: fill(hue, sat * 0.3, bri)
// 어둡게 하기: fill(hue, sat, bri * 0.5)
```

원시 RGB 값을 하드코딩하지 마세요. 팔레트 객체를 정의하고 절차적으로 변형을 도출하세요. `references/color-systems.md` 참조.

### 노이즈 — 원시 노이즈가 아닌 다중 옥타브(Multi-Octave)

원시 `noise(x, y)`는 매끄러운 덩어리처럼 보입니다. 자연스러운 텍스처를 위해 옥타브를 겹치세요:

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

흐르는 유기적 형태의 경우, **도메인 워핑(domain warping)**을 사용하세요: 노이즈 출력 값을 다시 노이즈 입력 좌표로 전달합니다. `references/visual-effects.md` 참조.

### 레이어를 위한 createGraphics() — 선택이 아님

평면적인 단일 패스 렌더링은 평면적으로 보입니다. 구성을 위해 오프스크린 버퍼를 사용하세요:

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
  renderTrails(trailLayer);   // 지속적이고 페이드되는 레이어
  renderForeground(fgLayer);  // 매 프레임마다 지워지는 레이어
  image(bgLayer, 0, 0);
  image(trailLayer, 0, 0);
  image(fgLayer, 0, 0);
}
```

### 성능 — 가능한 한 벡터화

p5.js의 draw 호출은 비용이 많이 듭니다. 수천 개의 파티클의 경우:

```javascript
// 느림: 개별 도형
for (let p of particles) {
  ellipse(p.x, p.y, p.size);
}

// 빠름: beginShape()을 사용한 단일 도형
beginShape(POINTS);
for (let p of particles) {
  vertex(p.x, p.y);
}
endShape();

// 가장 빠름: 대규모 개수를 위한 픽셀 버퍼
loadPixels();
for (let p of particles) {
  let idx = 4 * (floor(p.y) * width + floor(p.x));
  pixels[idx] = r; pixels[idx+1] = g; pixels[idx+2] = b; pixels[idx+3] = 255;
}
updatePixels();
```

`references/troubleshooting.md` § Performance 참조.

### 다중 스케치를 위한 인스턴스 모드(Instance Mode)

글로벌 모드는 `window`를 오염시킵니다. 프로덕션에서는 인스턴스 모드를 사용하세요:

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

한 페이지에 여러 스케치를 포함하거나 프레임워크와 통합할 때 필요합니다.

### WebGL 모드 주의사항

- `createCanvas(w, h, WEBGL)` — 원점이 좌측 상단이 아니라 중앙입니다.
- Y축이 반전되어 있습니다 (WEBGL에서는 양의 Y가 위쪽이고, P2D에서는 아래쪽입니다).
- P2D와 같은 좌표를 얻으려면 `translate(-width/2, -height/2)`를 사용합니다.
- 행렬 스택 오버플로우가 조용히 발생하므로, 모든 변환에 `push()`/`pop()`을 사용하세요.
- `texture()`는 `rect()`/`plane()` 앞에 옵니다 — 뒤가 아닙니다.
- 커스텀 셰이더: `createShader(vert, frag)` — 여러 브라우저에서 테스트하세요.

### 내보내기 — 단축키 규칙

모든 스케치는 `keyPressed()`에 다음을 포함해야 합니다:

```javascript
function keyPressed() {
  if (key === 's' || key === 'S') saveCanvas('output', 'png');
  if (key === 'g' || key === 'G') saveGif('output', 5);
  if (key === 'r' || key === 'R') { randomSeed(millis()); noiseSeed(millis()); }
  if (key === ' ') CONFIG.paused = !CONFIG.paused;
}
```

### 헤드리스 비디오 내보내기 — noLoop() 사용

Puppeteer를 통한 헤드리스 렌더링의 경우, 스케치의 설정에 반드시 `noLoop()`를 사용해야 합니다. 이것이 없으면 스크린샷이 찍히는 동안 p5의 그리기 루프가 자유롭게 실행되어 스케치가 앞서 나가고 건너뛰어지거나 중복된 프레임이 발생합니다.

```javascript
function setup() {
  createCanvas(1920, 1080);
  pixelDensity(1);
  noLoop();                    // 캡처 스크립트가 프레임 진행을 제어
  window._p5Ready = true;      // 캡처 스크립트에 준비 상태 알림
}
```

번들된 `scripts/export-frames.js`는 `_p5Ready`를 감지하고 정확히 1:1 프레임 일치를 위해 캡처당 한 번 `redraw()`를 호출합니다. `references/export-pipeline.md` § Deterministic Capture 참조.

다중 씬 비디오의 경우, 클립별 아키텍처를 사용하세요: 씬당 하나의 HTML, 독립적으로 렌더링, `ffmpeg -f concat`으로 결합. `references/export-pipeline.md` § Per-Clip Architecture 참조.

### 에이전트 워크플로우

p5.js 스케치를 빌드할 때:

1. **HTML 파일 작성** — 인라인 코드가 모두 포함된 단일 독립 파일.
2. **브라우저에서 열기** — `open sketch.html` (macOS) 또는 `xdg-open sketch.html` (Linux).
3. **로컬 애셋** (폰트, 이미지)은 서버가 필요합니다: 프로젝트 디렉토리에서 `python3 -m http.server 8080`, 그 다음 `http://localhost:8080/sketch.html` 열기.
4. **PNG/GIF 내보내기** — 위에 표시된 대로 `keyPressed()` 단축키를 추가하고, 사용자에게 어떤 키를 누를지 알려주세요.
5. **헤드리스 내보내기** — 자동화된 프레임 캡처를 위해 `node scripts/export-frames.js sketch.html --frames 300` 실행 (스케치는 반드시 `noLoop()` + `_p5Ready`를 사용해야 함).
6. **MP4 렌더링** — `bash scripts/render.sh sketch.html output.mp4 --duration 30`
7. **반복적 개선** — HTML 파일을 편집하고, 사용자가 브라우저를 새로 고쳐 변경 사항을 확인합니다.
8. **요청 시 참조 로드** — 구현 중에 특정 참조 파일을 로드하려면 `skill_view(name="p5js", file_path="references/...")`를 사용하세요.

## 성능 목표

| 지표 | 목표 |
|--------|--------|
| 프레임레이트 (인터랙티브) | 60fps 유지 |
| 프레임레이트 (애니메이션 내보내기) | 최소 30fps |
| 파티클 개수 (P2D 도형) | 60fps에서 5,000-10,000 |
| 파티클 개수 (픽셀 버퍼) | 60fps에서 50,000-100,000 |
| 캔버스 해상도 | 최대 3840x2160 (내보내기), 1920x1080 (인터랙티브) |
| 파일 크기 (HTML) | < 100KB (CDN 라이브러리 제외) |
| 로딩 시간 | 첫 프레임까지 < 2초 |

## 참조 자료

| 파일 | 내용 |
|------|----------|
| `references/core-api.md` | 캔버스 설정, 좌표계, 그리기 루프, `push()`/`pop()`, 오프스크린 버퍼, 합성 패턴, `pixelDensity()`, 반응형 디자인 |
| `references/shapes-and-geometry.md` | 2D 원시도형, `beginShape()`/`endShape()`, 베지어/캣멀-롬(Catmull-Rom) 곡선, `vertex()` 시스템, 커스텀 도형, `p5.Vector`, 부호화된 거리 장(SDF), SVG 패스 변환 |
| `references/visual-effects.md` | 노이즈 (펄린, 프랙탈, 도메인 워프, 컬), 흐름장(flow fields), 파티클 시스템 (물리, 무리 지기, 궤적), 픽셀 조작, 텍스처 생성 (점묘, 해칭, 하프톤), 피드백 루프, 반응 확산 |
| `references/animation.md` | 프레임 기반 애니메이션, 이징(easing) 함수, `lerp()`/`map()`, 스프링 물리, 상태 머신, 타임라인 시퀀싱, `millis()` 기반 타이밍, 트랜지션 패턴 |
| `references/typography.md` | `text()`, `loadFont()`, `textToPoints()`, 키네틱 타이포그래피, 텍스트 마스크, 폰트 메트릭, 반응형 텍스트 크기 조정 |
| `references/color-systems.md` | `colorMode()`, HSB/HSL/RGB, `lerpColor()`, `paletteLerp()`, 절차적 팔레트, 색상 조화, `blendMode()`, 그라데이션 렌더링, 선별된 팔레트 라이브러리 |
| `references/webgl-and-3d.md` | WEBGL 렌더러, 3D 원시도형, 카메라, 조명, 재질, 커스텀 기하학, GLSL 셰이더 (`createShader()`, `createFilterShader()`), 프레임버퍼, 포스트 프로세싱 |
| `references/interaction.md` | 마우스 이벤트, 키보드 상태, 터치 입력, DOM 요소, `createSlider()`/`createButton()`, 오디오 입력 (p5.sound FFT/진폭), 스크롤 구동 애니메이션, 반응형 이벤트 |
| `references/export-pipeline.md` | `saveCanvas()`, `saveGif()`, `saveFrames()`, 확정적 헤드리스 캡처, ffmpeg 프레임에서 비디오로, CCapture.js, SVG 수출, 클립별 아키텍처, 플랫폼 수출(fxhash), 비디오 주의사항 |
| `references/troubleshooting.md` | 성능 프로파일링, 픽셀당 예산, 흔한 실수, 브라우저 호환성, WebGL 디버깅, 폰트 로드 문제, 픽셀 밀도 함정, 메모리 누수, CORS |
| `templates/viewer.html` | 인터랙티브 뷰어 템플릿: 시드 내비게이션(이전/다음/랜덤/점프), 파라미터 슬라이더, PNG 다운로드, 반응형 캔버스. 탐색 가능한 생성 예술을 위해 이것부터 시작하세요 |

---

## 크리에이티브 발산 (사용자가 실험적/창의적/독특한 출력을 요청할 때만 사용)

사용자가 창의적이거나, 실험적이거나, 놀랍거나, 틀에 얽매이지 않은 출력을 요청하는 경우, 가장 잘 맞는 전략을 선택하고 코드를 생성하기 전에 단계를 추론해 보세요.

- **개념적 혼합 (Conceptual Blending)** — 사용자가 결합할 두 가지를 지명하거나 하이브리드 미학을 원할 때
- **SCAMPER** — 사용자가 알려진 생성 예술 패턴의 변형을 원할 때
- **거리 연상 (Distance Association)** — 사용자가 단일 컨셉을 주고 탐구를 원할 때 ("시간에 대한 것을 만들어줘")

### 개념적 혼합
1. 서로 다른 두 가지 시각 시스템 (예: 파티클 물리 + 손글씨)을 지명하세요.
2. 대응 관계를 매핑하세요 (파티클 = 잉크 방울, 힘 = 펜의 압력, 필드 = 글자 형태).
3. 선택적으로 혼합하세요 — 흥미롭고 창발적인 시각적 결과를 만들어내는 매핑을 유지하세요.
4. 나란히 있는 두 시스템이 아닌 통일된 시스템으로서 혼합을 코딩하세요.

### SCAMPER 변형
알려진 생성 패턴(흐름장, 파티클 시스템, L-시스템, 셀룰러 오토마타)을 체계적으로 변환합니다:
- **Substitute(대체)**: 원을 텍스트 문자로, 선을 그라데이션으로 대체
- **Combine(결합)**: 두 개의 패턴을 합치기 (흐름장 + 보로노이)
- **Adapt(적용)**: 2D 패턴을 3D 투영에 적용
- **Modify(수정)**: 크기를 과장하고, 좌표 공간을 왜곡
- **Purpose(목적 변경)**: 물리 시뮬레이션을 타이포그래피에, 정렬 알고리즘을 색상에 사용
- **Eliminate(제거)**: 그리드를 제거, 색상을 제거, 대칭을 제거
- **Reverse(반전)**: 시뮬레이션을 역방향으로 실행, 매개변수 공간을 반전

### 거리 연상
1. 사용자의 컨셉(예: "외로움")에 닻을 내립니다.
2. 세 가지 거리에서 연상을 생성합니다:
   - 가까움 (뻔함): 텅 빈 방, 외로운 사람, 침묵
   - 중간 (흥미로움): 잘못된 방향으로 헤엄치는 물고기 한 마리, 알림이 없는 전화기, 지하철 칸 사이의 틈
   - 멀음 (추상적): 소수(prime numbers), 점근선, 새벽 3시의 색
3. 중간 거리의 연상을 발전시키세요 — 시각화할 수 있을 만큼 구체적이면서도 흥미로울 만큼 예상치 못한 것입니다.
