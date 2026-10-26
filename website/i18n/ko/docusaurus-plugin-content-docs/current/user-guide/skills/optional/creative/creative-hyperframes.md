---
title: "Hyperframes — HTML 컴포지션에서 MP4/WebM 비디오 렌더링"
sidebar_label: "Hyperframes"
description: "HTML 컴포지션에서 MP4/WebM 비디오 렌더링"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Hyperframes

HTML 컴포지션에서 MP4/WebM 비디오를 렌더링합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/creative/hyperframes`로 설치 |
| 경로 | `optional-skills/creative/hyperframes` |
| 버전 | `1.0.0` |
| 작성자 | heygen-com |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `creative`, `video`, `animation`, `html`, `gsap`, `motion-graphics` |
| 관련 스킬 | [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), [`meme-generation`](/docs/user-guide/skills/optional/creative/creative-meme-generation) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# HyperFrames

비디오의 진실의 원천은 HTML입니다. 컴포지션은 타이밍을 위한 `data-*` 속성, 애니메이션을 위한 GSAP 타임라인, 화면 표시를 위한 CSS가 포함된 HTML 파일입니다. HyperFrames 엔진은 페이지를 프레임 단위로 캡처하고 FFmpeg로 MP4/WebM을 인코딩합니다.

**`manim-video`의 보완재:** 수학적/기하학적 설명 영상(방정식, 3B1B 스타일)에는 `manim-video`를 사용하세요. 모션 그래픽, 캡션이 포함된 토킹헤드 영상, 제품 투어, 소셜 오버레이, 셰이더 전환, 실제 비디오/오디오 미디어를 사용하는 모든 작업에는 `hyperframes`를 사용하세요.

## 사용 시점

- 사용자가 텍스트, 스크립트 또는 웹사이트에서 렌더링된 비디오를 요청하는 경우
- 애니메이션 타이틀 카드, 로워 서드 또는 타이포그래피 인트로
- 캡션이 포함된 내레이션 비디오(TTS + 파형에 동기화된 캡션)
- 오디오 반응형 비주얼(비트 동기화, 스펙트럼 바, 맥동하는 글로우)
- 장면 간 전환(크로스페이드, 와이프, 셰이더 워프, 백색 플래시 통과)
- 소셜 오버레이(Instagram/TikTok/YouTube 스타일)
- 웹사이트-비디오 파이프라인(URL을 캡처하여 프로모션 영상 제작)
- 비디오 파일로 결정론적으로 렌더링해야 하는 모든 HTML/CSS/JS 애니메이션

다음 작업에는 이 스킬을 사용하지 마세요.
- 순수한 수학/방정식 애니메이션(→ `manim-video`)
- 이미지 생성 또는 밈(→ `meme-generation`, 이미지 모델)
- 실시간 화상 회의 또는 스트리밍

## 빠른 참고

```bash
npx hyperframes init my-video               # scaffold a project
cd my-video
npx hyperframes lint                        # validate before preview/render
npx hyperframes preview                     # live-reload browser preview (port 3002)
npx hyperframes render --output final.mp4   # render to MP4
npx hyperframes doctor                      # diagnose environment issues
```

렌더링 플래그: `--quality draft|standard|high` · `--fps 24|30|60` · `--format mp4|webm` · `--docker`(재현 가능) · `--strict`.

전체 CLI 참고: [references/cli.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/cli.md).

## 설정(최초 한 번)

```bash
bash "$(dirname "$(find ~/.hermes/skills -path '*/hyperframes/SKILL.md' 2>/dev/null | head -1)")/scripts/setup.sh"
```

스크립트가 수행하는 작업:
1. Node.js >= 22 및 FFmpeg가 설치되어 있는지 확인합니다(설치되어 있지 않으면 해결 방법을 출력합니다).
2. `hyperframes` CLI를 전역으로 설치합니다(`npm install -g hyperframes@>=0.4.2`).
3. Puppeteer를 통해 `chrome-headless-shell`을 미리 캐시합니다 — Chrome의 `HeadlessExperimental.beginFrame` 캡처 경로로 최고 품질의 렌더링을 하려면 **필수**입니다.
4. `npx hyperframes doctor`를 실행하고 결과를 보고합니다.

설정에 실패하면 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md)를 참고하세요.

## 절차

### 1. HTML 작성 전에 계획 수립

코드를 다루기 전에 다음 항목을 상위 수준에서 명확히 설명하세요.
- **무엇을** — 내러티브 아크, 핵심 순간, 감정적 박자
- **구조** — 컴포지션, 트랙(비디오/오디오/오버레이), 재생 시간
- **시각적 정체성** — 색상, 글꼴, 모션의 성격(폭발적 / 영화적 / 유려한 / 기술적)
- **히어로 프레임** — 각 장면에서 가장 많은 요소가 동시에 보이는 순간. 먼저 구축할 정적 레이아웃입니다.

**시각적 정체성 게이트(HARD-GATE).** 어떤 컴포지션 HTML이든 작성하기 전에 시각적 정체성을 정의해야 합니다. 기본 또는 일반적인 색상(`#333`, `#3b82f6`, `Roboto`는 이 단계를 건너뛴 신호입니다)으로 컴포지션을 작성하지 마세요. 다음 순서로 확인하세요.

1. **프로젝트 루트에 `DESIGN.md`가 있나요?** → 정확한 색상, 글꼴, 모션 규칙 및 "하지 말아야 할 것" 제약 조건을 사용하세요.
2. **사용자가 스타일을 지정했나요**(예: "Swiss Pulse", "어둡고 기술적인", "럭셔리 브랜드")? → `## Style Prompt`, `## Colors`(역할이 포함된 3~5개의 hex), `## Typography`(1~2개의 패밀리), `## What NOT to Do`(3~5개의 안티패턴)이 포함된 최소한의 `DESIGN.md`를 생성하세요.
3. **위 항목이 모두 없나요?** → HTML을 작성하기 전에 다음 3가지 질문을 하세요.
   - 분위기는? (폭발적 / 영화적 / 유려한 / 기술적 / 혼돈스러운 / 따뜻한)
   - 밝은 캔버스인가요, 어두운 캔버스인가요?
   - 브랜드 색상, 글꼴 또는 시각적 레퍼런스가 있나요?

그런 다음 답변으로 `DESIGN.md`를 생성하세요. 모든 컴포지션은 팔레트와 타이포그래피를 `DESIGN.md` 또는 사용자의 명시적 지시로 추적할 수 있어야 합니다.

### 2. 스캐폴딩

```bash
npx hyperframes init my-video --non-interactive
```

템플릿: `blank`, `warm-grain`, `play-mode`, `swiss-grid`, `vignelli`, `decision-tree`, `kinetic-type`, `product-promo`, `nyt-graph`. `--example <name>`을 전달해 선택하고, `--video clip.mp4` 또는 `--audio track.mp3`를 사용해 미디어를 시드하세요.

### 3. 애니메이션 전 레이아웃

먼저 히어로 프레임의 정적 HTML+CSS를 작성하세요 — 아직 GSAP은 사용하지 않습니다. `.scene-content` 컨테이너는 `display:flex` + `gap`과 함께 장면을 채워야 합니다(`width:100%; height:100%; padding:Npx`). 콘텐츠를 안쪽으로 밀어 넣을 때는 패딩을 사용하세요 — 콘텐츠 컨테이너에 `position: absolute; top: Npx`를 사용하지 마세요(남은 공간보다 콘텐츠가 더 클 때 콘텐츠가 넘칩니다).

히어로 프레임이 올바르게 보인 후에야 `gsap.from()` 진입 애니메이션(CSS 위치 **로** 애니메이션)과 `gsap.to()` 종료 애니메이션(해당 위치 **에서** 애니메이션)을 추가하세요.

전체 data-attribute 스키마와 컴포지션 규칙은 [references/composition.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/composition.md)를 참고하세요.

### 4. GSAP으로 애니메이션 적용

모든 컴포지션은 다음을 충족해야 합니다.
- 타임라인 등록: `window.__timelines["<composition-id>"] = tl`
- 일시 정지 상태로 시작: `gsap.timeline({ paused: true })` — 재생은 플레이어가 제어합니다.
- 유한한 `repeat` 값 사용(`repeat: -1`은 사용하지 마세요 — 캡처 엔진이 중단됩니다). 다음과 같이 계산합니다: `repeat: Math.ceil(duration / cycleDuration) - 1`.
- 결정론적이어야 함 — `Math.random()`, `Date.now()` 또는 벽시계 로직을 사용하지 마세요. 의사 난수가 필요하면 시드가 있는 PRNG를 사용하세요.
- 동기적으로 빌드 — 타임라인 구성에 `async`/`await`, `setTimeout` 또는 Promise를 사용하지 마세요.

핵심 GSAP API(트윈, 이징, stagger, 타임라인)는 [references/gsap.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/gsap.md)를 참고하세요.

### 5. 장면 간 전환

여러 장면으로 구성된 컴포지션에는 전환이 필요합니다. 규칙:
1. **항상 장면 사이에 전환을 사용하세요** — 점프 컷을 사용하지 마세요.
2. **모든 장면 요소에 항상 진입 애니메이션을 사용하세요**(`gsap.from(...)`).
3. **최종 장면을 제외하고 종료 애니메이션을 사용하지 마세요** — 전환이 곧 종료입니다.
4. 최종 장면은 페이드 아웃할 수 있습니다.

`npx hyperframes add <transition-name>`을 사용해 셰이더 전환(`flash-through-white`, `liquid-wipe` 등)을 설치하세요. 전체 목록: `npx hyperframes add --list`.

### 6. 오디오, 캡션, TTS, 오디오 반응형, 하이라이트

- **오디오:** 항상 별도의 `<audio>` 요소를 사용하세요(비디오는 `muted playsinline`).
- **TTS:** `npx hyperframes tts "Script text" --voice af_nova --output narration.wav`. `--list`로 음성을 나열하세요. 음성 ID의 첫 글자는 언어를 인코딩합니다(`a`/`b`=영어, `e`=스페인어, `f`=프랑스어, `j`=일본어, `z`=중국어 등) — CLI가 음소화 로케일을 자동으로 추론합니다. 재정의할 때만 `--lang`을 전달하세요. 영어가 아닌 음소화에는 시스템 전체에 `espeak-ng`가 설치되어 있어야 합니다.
- **캡션:** `npx hyperframes transcribe narration.wav` → 단어 단위 트랜스크립트. 트랜스크립트의 어조에 맞춰 스타일을 선택하세요(하이프 / 기업 / 튜토리얼 / 스토리텔링 / 소셜 — 표는 `references/features.md` 참조). **언어 규칙:** 오디오가 영어임이 확인되지 않았다면 `.en` whisper 모델을 사용하지 마세요 — `.en`은 영어가 아닌 오디오를 전사하는 대신 번역합니다. 모든 캡션 그룹은 종료 트윈 이후에 `tl.set(el, { opacity: 0, visibility: "hidden" }, group.end)`를 사용한 하드 kill을 반드시 가져야 합니다 — 그렇지 않으면 이후 그룹에 캡션이 보이는 상태로 누수됩니다.
- **오디오 반응형 비주얼:** 오디오 대역(bass / mid / treble)을 미리 추출하고 `for` 루프와 `tl.call(draw, [], f / fps)`를 사용해 타임라인 안에서 프레임마다 샘플링하세요 — 하나의 긴 트윈만으로는 오디오에 반응하지 않습니다. bass → `scale`(펄스), treble → `textShadow`/`boxShadow`(글로우), 전체 진폭 → `opacity`/`y`/`backgroundColor`로 매핑하세요. 이퀄라이저 바 같은 상투적인 표현은 피하세요 — 콘텐츠가 비주얼을 이끌고 오디오는 동작을 이끌도록 하세요.
- **마커 스타일 하이라이트:** 텍스트 강조를 위한 하이라이트, 원, 버스트, 낙서, 스케치아웃 효과는 결정론적인 CSS+GSAP입니다 — `references/features.md#marker-highlighting`을 참조하세요. 완전히 탐색 가능하며 애니메이션 SVG 필터를 사용하지 않습니다.
- **장면 전환:** 여러 장면으로 구성된 모든 컴포지션은 반드시 전환을 사용해야 합니다(점프 컷 금지). CSS 프리미티브(푸시 슬라이드, 블러 크로스페이드, 줌 스루, 스태거 블록) 또는 `npx hyperframes add`를 통한 셰이더 전환(`flash-through-white`, `liquid-wipe`, `cross-warp-morph`, `chromatic-split` 등) 중에서 선택하세요. 분위기와 에너지 표는 `references/features.md#transitions`에 있습니다. 동일한 컴포지션에서 CSS 전환과 셰이더 전환을 섞지 마세요.

### 7. 린트, 검증, 검사, 미리 보기, 렌더링

```bash
npx hyperframes lint              # catches missing data-composition-id, overlapping tracks, unregistered timelines
npx hyperframes validate          # WCAG contrast audit at 5 timestamps
npx hyperframes inspect           # visual layout audit — overflow, off-frame elements, occluded text
npx hyperframes preview           # live browser preview
npx hyperframes render --quality draft --output draft.mp4    # fast iteration
npx hyperframes render --quality high --output final.mp4     # final delivery
```

`hyperframes validate`는 모든 텍스트 요소 뒤의 배경 픽셀을 샘플링하고 대비율이 4.5:1 미만(큰 텍스트는 3:1 미만)이면 경고합니다. `hyperframes inspect`는 레이아웃 측면의 동반 도구입니다 — 여러 타임스탬프에서 페이지를 실행하고 정적 린트로는 볼 수 없는 문제(4.5초 시점에만 캡션이 안전 영역을 넘어 줄바꿈되는 경우, 제목이 가장 긴 변형일 때 카드가 넘치는 경우, 전환 셰이더 뒤에 요소가 배치되는 경우)를 표시합니다. 말풍선, 카드, 캡션 또는 빽빽한 타이포그래피가 있는 컴포지션에서는 특히 `inspect`를 실행하세요.

### 8. 웹사이트-비디오(사용자가 URL을 제공한 경우)

[references/website-to-video.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/website-to-video.md)의 7단계 캡처-비디오 워크플로를 사용하세요: 캡처 → DESIGN.md → SCRIPT.md → 스토리보드 → 컴포지션 → 렌더링 → 전달.

## 주의 사항

- **`HeadlessExperimental.beginFrame' wasn't found`** — Chromium 147+에서 이 프로토콜을 제거했습니다. `hyperframes@>=0.4.2`를 사용 중인지 확인하세요(자동으로 감지하고 스크린샷 모드로 대체합니다). 비상 탈출구: `export PRODUCER_FORCE_SCREENSHOT=true`. [hyperframes#294](https://github.com/heygen-com/hyperframes/issues/294) 및 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md)를 참고하세요.
- **시스템 Chrome(`chrome-headless-shell` 아님)** — 렌더링이 120초 동안 멈춘 후 타임아웃됩니다. `npx puppeteer browsers install chrome-headless-shell`을 실행하세요(`setup.sh`가 이를 수행합니다). `hyperframes doctor`가 사용할 바이너리를 보고합니다.
- **어디서든 `repeat: -1` 사용** — 캡처 엔진이 중단됩니다. 항상 유한한 repeat 횟수를 계산하세요.
- **나중에 진입하는 클립 요소에 `gsap.set()` 사용** — 페이지 로드 시 요소가 존재하지 않습니다. 대신 타임라인 내부의 클립 `data-start` 시점 또는 그 이후에 `tl.set(selector, vars, timePosition)`을 사용하세요.
- **콘텐츠 텍스트 내부의 `<br>`** — 강제 줄바꿈은 렌더링된 글꼴 너비를 알지 못하므로 자연스러운 줄바꿈과 `<br>`이 이중 줄바꿈을 만듭니다. `max-width`를 사용해 텍스트가 줄바꿈되도록 하세요. 예외: 각 단어를 의도적으로 한 줄에 하나씩 배치한 짧은 디스플레이 제목.
- **`visibility` 또는 `display` 애니메이션** — GSAP은 이를 트윈할 수 없습니다. 둘 다 visibility와 opacity를 처리하는 `autoAlpha`를 사용하세요.
- **`video.play()` 또는 `audio.play()` 호출** — 재생은 프레임워크가 담당합니다. 직접 호출하지 마세요.
- **타임라인을 비동기적으로 빌드** — 캡처 엔진은 페이지 로드 직후 `window.__timelines`를 동기적으로 읽습니다. 타임라인 구성을 `async`, `setTimeout` 또는 Promise로 감싸지 마세요.
- **`<template>`로 감싼 독립적인 `index.html`** — 브라우저에서 모든 콘텐츠가 숨겨집니다. `data-composition-src`로 로드되는 **하위 컴포지션**만 `<template>`을 사용합니다.
- **오디오에 비디오 사용** — 항상 음소거된 `<video>`와 별도의 `<audio>`를 사용하세요.

## 검증

렌더링 전후에 다음을 수행하세요.

1. **Lint + validate + inspect 통과:** `npx hyperframes lint --strict && npx hyperframes validate && npx hyperframes inspect`(lint는 구조적 문제, validate는 대비, inspect는 시각적 레이아웃/오버플로 문제를 감지합니다 — 경고가 표시되면 troubleshooting.md를 참고하세요).
2. **애니메이션 안무** — 새 컴포지션 또는 중대한 애니메이션 변경의 경우 애니메이션 맵을 실행하세요. `npx hyperframes init`이 스킬 스크립트를 프로젝트에 복사하므로 경로는 프로젝트 로컬입니다.
   ```bash
   node skills/hyperframes/scripts/animation-map.mjs <composition-dir> \
     --out <composition-dir>/.hyperframes/anim-map
   ```
   트윈별 요약, ASCII 간트 타임라인, stagger 감지, 데드존(애니메이션이 없는 1초 초과 구간), 요소 수명 주기 및 플래그(`offscreen`, `collision`, `invisible`, `paced-fast` &lt;0.2s, `paced-slow` >2s)가 포함된 단일 `animation-map.json`을 출력합니다. 요약과 플래그를 검사하고 각각을 수정하거나 정당화하세요. 작은 편집에서는 건너뛰어도 됩니다.
3. **파일 존재 및 0이 아닌 크기:** `ls -lh final.mp4`.
4. **`data-duration`과 재생 시간 일치:** `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 final.mp4`.
5. **시각적 확인:** 중간 컴포지션 프레임 추출: `ffmpeg -i final.mp4 -ss 00:00:05 -vframes 1 preview.png`.
6. **예상한 경우 오디오 존재:** `ffprobe -v error -show_streams -select_streams a -of default=nw=1:nk=1 final.mp4 | head -1`.

`hyperframes render`가 실패하면 `npx hyperframes doctor`를 실행하고 보고할 때 그 출력을 첨부하세요.

## 참고 자료

- [composition.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/composition.md) — data attribute, 타임라인 계약, 협상할 수 없는 규칙, 타이포그래피/에셋 규칙
- [cli.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/cli.md) — 모든 CLI 명령(init, capture, lint, validate, inspect, preview, render, transcribe, tts, doctor, browser, info, upgrade, benchmark)
- [gsap.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/gsap.md) — HyperFrames용 GSAP 핵심 API(tween, ease, stagger, timeline, matchMedia)
- [features.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/features.md) — 캡션, TTS, 오디오 반응형, 마커 하이라이트, 전환(필요할 때 로드)
- [website-to-video.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/website-to-video.md) — 7단계 캡처-비디오 워크플로
- [troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md) — OpenClaw 수정, env vars, 일반적인 렌더링 오류
