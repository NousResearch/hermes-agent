---
title: "Hyperframes"
sidebar_label: "Hyperframes"
description: "HyperFrames를 사용하여 HTML 기반 비디오 구성 요소, 애니메이션 타이틀 카드, 소셜 오버레이, 자막이 있는 토킹 헤드 비디오, 오디오 반응형 시각 효과 및 셰이더 트랜지션을 만듭니다."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hyperframes

HyperFrames를 사용하여 HTML 기반 비디오 컴포지션, 애니메이션 타이틀 카드, 소셜 오버레이, 자막이 있는 토킹 헤드 비디오, 오디오 반응형 시각 효과 및 셰이더 전환(transition)을 만듭니다. HTML은 비디오의 단일 진실 공급원(source of truth)입니다. 사용자가 HTML 컴포지션에서 렌더링된 MP4/WebM을 원하거나, 미디어 위에 텍스트/로고/차트를 애니메이션화하고 싶거나, 오디오에 동기화된 자막이 필요하거나, TTS 내레이션을 원하거나, 웹사이트를 비디오로 변환하고 싶을 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/creative/hyperframes`로 설치 |
| Path | `optional-skills/creative/hyperframes` |
| Version | `1.0.0` |
| Author | heygen-com |
| License | Apache-2.0 |
| Platforms | linux, macos, windows |
| Tags | `creative`, `video`, `animation`, `html`, `gsap`, `motion-graphics` |
| Related skills | [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), [`meme-generation`](/docs/user-guide/skills/optional/creative/creative-meme-generation) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# HyperFrames

HTML은 비디오의 단일 진실 공급원입니다. 컴포지션은 타이밍을 위한 `data-*` 속성, 애니메이션을 위한 GSAP 타임라인, 그리고 외형을 위한 CSS가 포함된 HTML 파일입니다. HyperFrames 엔진은 페이지를 프레임 단위로 캡처하여 FFmpeg를 통해 MP4/WebM으로 인코딩합니다.

**`manim-video`에 대한 보완:** 수학적/기하학적 설명 영상(방정식, 3B1B 스타일)에는 `manim-video`를 사용하세요. 모션 그래픽, 자막이 있는 토킹 헤드, 제품 투어, 소셜 오버레이, 셰이더 전환, 그리고 실제 비디오/오디오 미디어에 의해 구동되는 모든 것에는 `hyperframes`를 사용하세요.

## 사용 시기

- 사용자가 텍스트, 대본 또는 웹사이트에서 렌더링된 비디오를 요청할 때
- 애니메이션 타이틀 카드, 로어 서드(lower thirds), 또는 타이포그래피 인트로
- 자막이 있는 내레이션 비디오 (파형에 동기화된 TTS + 자막)
- 오디오 반응형 시각 효과 (비트 동기화, 스펙트럼 바, 맥동하는 광원)
- 장면 간 전환 (크로스페이드, 와이프, 셰이더 워프, 화이트 아웃)
- 소셜 오버레이 (Instagram/TikTok/YouTube 스타일)
- 웹사이트-to-비디오 파이프라인 (URL을 캡처하여 홍보 영상 제작)
- 비디오 파일로 결정론적으로(deterministically) 렌더링되어야 하는 모든 HTML/CSS/JS 애니메이션

다음과 같은 경우에는 이 스킬을 **사용하지 마세요**:
- 순수 수학/방정식 애니메이션 (→ `manim-video`)
- 이미지 생성 또는 밈 (→ `meme-generation`, 이미지 모델)
- 실시간 화상 회의 또는 스트리밍

## 빠른 참조

```bash
npx hyperframes init my-video               # 프로젝트 생성(scaffold)
cd my-video
npx hyperframes lint                        # 미리보기/렌더링 전 검증
npx hyperframes preview                     # 라이브 리로드 브라우저 미리보기 (포트 3002)
npx hyperframes render --output final.mp4   # MP4로 렌더링
npx hyperframes doctor                      # 환경 문제 진단
```

렌더링 플래그: `--quality draft|standard|high` · `--fps 24|30|60` · `--format mp4|webm` · `--docker` (재현 가능) · `--strict`.

전체 CLI 참조: [references/cli.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/cli.md).

## 설정 (최초 1회)

```bash
bash "$(dirname "$(find ~/.hermes/skills -path '*/hyperframes/SKILL.md' 2>/dev/null | head -1)")/scripts/setup.sh"
```

스크립트 수행 내용:
1. Node.js >= 22 및 FFmpeg 설치 확인 (설치되지 않은 경우 해결 지침 출력).
2. `hyperframes` CLI를 전역으로 설치 (`npm install -g hyperframes@>=0.4.2`).
3. Puppeteer를 통해 `chrome-headless-shell` 사전 캐싱 — Chrome의 `HeadlessExperimental.beginFrame` 캡처 경로를 통한 최고 품질 렌더링에 **필수**.
4. `npx hyperframes doctor`를 실행하고 결과를 보고합니다.

설치에 실패하면 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md)를 참조하세요.

## 절차

### 1. HTML을 작성하기 전 계획 수립

코드를 작성하기 전에 높은 수준에서 다음을 명확히 하세요:
- **내용 (What)** — 서사 구조(narrative arc), 주요 순간, 감정적 박자
- **구조** — 컴포지션, 트랙(비디오/오디오/오버레이), 지속 시간
- **시각적 정체성** — 색상, 글꼴, 모션 특성 (폭발적인 / 영화 같은 / 부드러운 / 기술적인)
- **히어로 프레임 (Hero frame)** — 각 장면에서 가장 많은 요소가 동시에 보이는 순간. 이것이 여러분이 가장 먼저 구축할 정적 레이아웃입니다.

**시각적 정체성 확인 게이트 (HARD-GATE).** HTML 컴포지션을 작성하기 전에 반드시 시각적 정체성이 정의되어야 합니다. 기본 또는 일반 색상(`#333`, `#3b82f6`, `Roboto` 등)으로 컴포지션을 작성하지 마세요(이 단계를 건너뛰었다는 징후입니다). 순서대로 확인하세요:

1. **프로젝트 루트에 `DESIGN.md`가 있습니까?** → 그곳의 정확한 색상, 글꼴, 모션 규칙, "하지 말아야 할 것(What NOT to Do)" 제약을 사용하세요.
2. **사용자가 스타일을 지정했습니까?** (예: "Swiss Pulse", "어둡고 기술적인", "럭셔리 브랜드") → `## Style Prompt`, `## Colors` (역할이 부여된 3-5개의 16진수 색상), `## Typography` (1-2개의 글꼴), `## What NOT to Do` (3-5개의 안티 패턴)가 포함된 최소한의 `DESIGN.md`를 생성하세요.
3. **위 항목에 해당하지 않습니까?** → HTML을 작성하기 전에 다음 3가지 질문을 하세요:
   - 분위기(Mood)? (폭발적인 / 영화 같은 / 부드러운 / 기술적인 / 혼란스러운 / 따뜻한)
   - 밝은 캔버스 아니면 어두운 캔버스?
   - 브랜드 색상, 글꼴 또는 시각적 레퍼런스가 있나요?

   답변을 바탕으로 `DESIGN.md`를 생성하세요. 모든 컴포지션은 팔레트와 타이포그래피를 `DESIGN.md` 또는 사용자의 명시적 지시에 근거해야 합니다.

### 2. 프로젝트 생성 (Scaffold)

```bash
npx hyperframes init my-video --non-interactive
```

템플릿: `blank`, `warm-grain`, `play-mode`, `swiss-grid`, `vignelli`, `decision-tree`, `kinetic-type`, `product-promo`, `nyt-graph`. 템플릿을 선택하려면 `--example <name>`을 전달하고, 미디어를 추가하려면 `--video clip.mp4` 또는 `--audio track.mp3`를 전달하세요.

### 3. 애니메이션 전 레이아웃

**히어로 프레임**을 위한 정적 HTML+CSS를 먼저 작성하세요 — 아직 GSAP을 사용하지 마세요. `.scene-content` 컨테이너는 `display:flex` + `gap`과 함께 장면을 가득 채워야 합니다 (`width:100%; height:100%; padding:Npx`). 콘텐츠를 안쪽으로 밀어 넣으려면 패딩을 사용하세요 — 콘텐츠 컨테이너에 절대 `position: absolute; top: Npx`를 사용하지 마세요 (콘텐츠가 남은 공간보다 길어지면 넘칩니다).

히어로 프레임이 제대로 보인 후에야 비로소 `gsap.from()` 등장 (해당 CSS 위치**로** 애니메이션)과 `gsap.to()` 퇴장 (해당 위치**에서** 애니메이션)을 추가합니다.

전체 데이터 속성 스키마 및 구성 규칙은 [references/composition.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/composition.md)를 참조하세요.

### 4. GSAP 애니메이션

모든 컴포지션의 필수 조건:
- 타임라인 등록: `window.__timelines["<composition-id>"] = tl`
- 일시 중지 상태로 시작: `gsap.timeline({ paused: true })` — 플레이어가 재생을 제어합니다.
- 유한한 `repeat` 값 사용 (`repeat: -1` 사용 금지 — 캡처 엔진을 고장냅니다). 계산 방법: `repeat: Math.ceil(duration / cycleDuration) - 1`.
- 결정론적(deterministic)일 것 — `Math.random()`, `Date.now()`, 또는 실시간 논리(wall-clock logic)를 사용하지 마세요. 의사 난수가 필요한 경우 시드(seed)가 있는 PRNG를 사용하세요.
- 동기적으로 빌드할 것 — 타임라인 구성 주변에 `async`/`await`, `setTimeout`, 또는 Promise를 사용하지 마세요.

HyperFrames를 위한 핵심 GSAP API (tweens, eases, stagger, timelines)는 [references/gsap.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/gsap.md)를 참조하세요.

### 5. 장면 간 전환 (Transitions)

다중 장면(Multi-scene) 컴포지션에는 전환이 필요합니다. 규칙:
1. **항상 장면 간 전환 효과를 사용하세요** — 점프 컷(jump cut)을 하지 마세요.
2. **항상 모든 장면 요소에 등장 애니메이션**(`gsap.from(...)`)을 사용하세요.
3. **마지막 장면을 제외하고는 절대로 퇴장 애니메이션을 사용하지 마세요** — 전환 자체가 퇴장입니다.
4. 마지막 장면은 페이드아웃될 수 있습니다.

`npx hyperframes add <transition-name>`을 사용하여 셰이더 전환(`flash-through-white`, `liquid-wipe` 등)을 설치하세요. 전체 목록: `npx hyperframes add --list`.

### 6. 오디오, 자막, TTS, 오디오 반응, 하이라이팅

- **오디오:** 항상 별도의 `<audio>` 요소를 사용합니다 (비디오는 `muted playsinline`입니다).
- **TTS:** `npx hyperframes tts "Script text" --voice af_nova --output narration.wav`. `--list`로 목소리 목록을 확인하세요. 음성 ID의 첫 글자는 언어를 나타냅니다 (`a`/`b`=영어, `e`=스페인어, `f`=프랑스어, `j`=일본어, `z`=중국어 등) — CLI는 포니마이저(phonemizer) 로케일을 자동 추론합니다; 덮어쓰려면 `--lang`만 전달하세요. 영어가 아닌 음소 변환에는 시스템 전역에 `espeak-ng`가 설치되어 있어야 합니다.
- **자막:** `npx hyperframes transcribe narration.wav` → 단어 수준 트랜스크립트. 트랜스크립트 톤에서 스타일을 선택하세요 (hype / corporate / tutorial / storytelling / social — `references/features.md`의 표 참조). **언어 규칙:** 오디오가 영어임이 확실하지 않으면 절대 `.en` whisper 모델을 사용하지 마세요 — `.en`은 영어가 아닌 오디오를 필사(transcribe)하는 대신 번역(translate)해 버립니다. 모든 캡션 그룹은 반드시 퇴장 트윈(tween) 이후에 강제 `tl.set(el, { opacity: 0, visibility: "hidden" }, group.end)` 종료 명령이 있어야 합니다 — 그렇지 않으면 이전 그룹이 나중 그룹까지 계속 보이게 됩니다.
- **오디오 반응형 시각 효과:** 사전에 오디오 대역(bass / mid / treble)을 추출하고 타임라인 내에서 `tl.call(draw, [], f / fps)`의 `for` 루프를 사용하여 프레임 단위로 샘플링하세요 — 하나의 긴 트윈은 오디오에 반응하지 **않습니다**. 베이스(bass) → `scale` (펄스), 트레블(treble) → `textShadow`/`boxShadow` (발광), 전체 진폭 → `opacity`/`y`/`backgroundColor`로 매핑하세요. 진부한 이퀄라이저 바를 피하세요 — 콘텐츠가 비주얼을 이끌게 하고, 오디오가 그 동작을 주도하게 하세요.
- **마커 스타일 하이라이팅:** 텍스트 강조를 위한 하이라이트, 원, 파열(burst), 낙서(scribble), 스케치아웃 효과는 결정론적 CSS+GSAP입니다 — `references/features.md#marker-highlighting` 참조. 완벽하게 탐색 가능(seekable)하며, 애니메이션 SVG 필터가 없습니다.
- **장면 전환:** 다중 장면 컴포지션은 항상 전환을 사용해야 합니다(점프 컷 금지). CSS 원형(primitives)(밀어내기, 흐림 크로스페이드, 줌 아웃, 시차 블록) 또는 셰이더 전환(`flash-through-white`, `liquid-wipe`, `cross-warp-morph`, `chromatic-split` 등) 중 하나를 `npx hyperframes add`로 선택하세요. 분위기와 에너지 표는 `references/features.md#transitions`에 있습니다. 동일한 컴포지션 내에서 CSS와 셰이더 전환을 혼합하지 마세요.

### 7. Lint, validate, inspect, preview, render

```bash
npx hyperframes lint              # 누락된 data-composition-id, 겹치는 트랙, 등록되지 않은 타임라인 포착
npx hyperframes validate          # 5개의 타임스탬프에서 WCAG 명암비 검사
npx hyperframes inspect           # 시각적 레이아웃 검사 — 오버플로우, 프레임 이탈 요소, 가려진 텍스트
npx hyperframes preview           # 라이브 브라우저 미리보기
npx hyperframes render --quality draft --output draft.mp4    # 빠른 반복 작업용
npx hyperframes render --quality high --output final.mp4     # 최종 결과물용
```

`hyperframes validate`는 모든 텍스트 요소 뒤에 있는 배경 픽셀을 샘플링하여 4.5:1 (또는 큰 텍스트의 경우 3:1) 미만의 명암비에 대해 경고합니다. `hyperframes inspect`는 레이아웃 검증 도우미입니다 — 다양한 타임스탬프에서 페이지를 실행하고 정적 lint가 볼 수 없는 문제(예: 4.5초에서만 안전 영역을 넘어서 줄바꿈되는 캡션, 가장 긴 변형 제목일 때 오버플로우되는 카드, 전환 셰이더 뒤에 가려진 요소)를 표시합니다. 말풍선, 카드, 캡션 또는 빡빡한 타이포그래피가 있는 컴포지션에서는 특히 `inspect`를 실행하세요.

### 8. 웹사이트를 비디오로 (사용자가 URL을 제공한 경우)

[references/website-to-video.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/website-to-video.md)의 7단계 캡처-to-비디오 워크플로우를 사용하세요: 캡처 → DESIGN.md → SCRIPT.md → 스토리보드 → 컴포지션 → 렌더링 → 배포.

## 주의 사항

- **`HeadlessExperimental.beginFrame' wasn't found`** — Chromium 147 이상에서 이 프로토콜을 제거했습니다. `hyperframes@>=0.4.2` 버전인지 확인하세요(자동 감지하고 스크린샷 모드로 대체됨). 탈출구: `export PRODUCER_FORCE_SCREENSHOT=true`. [hyperframes#294](https://github.com/heygen-com/hyperframes/issues/294) 및 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md)를 참조하세요.
- **시스템 Chrome (`chrome-headless-shell` 아님)** — 렌더링이 120초 동안 멈춘 후 시간 초과됩니다. `npx puppeteer browsers install chrome-headless-shell`을 실행하세요 (setup.sh가 이 작업을 수행합니다). `hyperframes doctor`가 사용될 바이너리를 보고합니다.
- **어디서든 `repeat: -1` 사용** — 캡처 엔진을 망가뜨립니다. 항상 유한한 반복 횟수를 계산하세요.
- **나중에 들어오는 클립(clip) 요소에 `gsap.set()` 사용** — 페이지 로드 시에는 요소가 존재하지 않습니다. 대신 타임라인 내에서, 클립의 `data-start` 시점 이후에 `tl.set(selector, vars, timePosition)`을 사용하세요.
- **콘텐츠 텍스트 내의 `<br>`** — 강제 줄바꿈은 렌더링된 글꼴 너비를 모르므로, 자연스러운 줄바꿈 + `<br>`의 이중 줄바꿈이 발생합니다. 텍스트가 자연스럽게 줄바꿈되도록 `max-width`를 사용하세요. 예외: 각 단어가 의도적으로 개별 줄에 있는 짧은 표시 제목(display titles).
- **`visibility` 또는 `display` 애니메이션화** — GSAP은 이들을 트윈(tween)할 수 없습니다. `autoAlpha`를 사용하세요 (가시성과 투명도를 모두 처리함).
- **`video.play()` 또는 `audio.play()` 호출** — 프레임워크가 재생을 제어합니다. 절대로 직접 호출하지 마세요.
- **비동기 타임라인 구성** — 캡처 엔진은 페이지 로드 후 `window.__timelines`를 동기적으로 읽습니다. 타임라인 구성을 `async`, `setTimeout`, 또는 Promise로 감싸지 마세요.
- **`<template>`으로 감싸진 독립형(Standalone) `index.html`** — 브라우저에서 모든 콘텐츠를 숨깁니다. `data-composition-src`를 통해 로드되는 **서브 컴포지션(sub-compositions)**만 `<template>`을 사용합니다.
- **오디오에 비디오 사용** — 항상 음소거된 `<video>` + 별도의 `<audio>`를 사용하세요.

## 검증

렌더링 전후:

1. **Lint + validate + inspect 통과:** `npx hyperframes lint --strict && npx hyperframes validate && npx hyperframes inspect` (lint는 구조적 문제를 파악하고, validate는 대비를, inspect는 시각적 레이아웃/오버플로우 문제를 파악합니다 — 경고가 나타나면 troubleshooting.md 참조).
2. **애니메이션 안무 (Animation choreography)** — 새로운 컴포지션이나 큰 애니메이션 변경 사항이 있는 경우 애니메이션 맵을 실행하세요. `npx hyperframes init`이 스킬 스크립트를 프로젝트에 복사하므로 경로는 프로젝트 로컬입니다:
   ```bash
   node skills/hyperframes/scripts/animation-map.mjs <composition-dir> \
     --out <composition-dir>/.hyperframes/anim-map
   ```
   각 트윈 요약, ASCII Gantt 타임라인, 스태거 감지, 데드 존 (애니메이션이 없는 >1초), 요소 수명 주기, 그리고 플래그(`offscreen`, `collision`, `invisible`, `paced-fast` &lt;0.2초, `paced-slow` >2초)가 포함된 단일 `animation-map.json`을 출력합니다. 요약과 플래그를 검사하여 각각을 수정하거나 정당화하세요. 작은 편집에는 건너뜁니다.
3. **파일 존재 + 0바이트 아님:** `ls -lh final.mp4`.
4. **기간이 `data-duration`과 일치:** `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 final.mp4`.
5. **시각적 검사:** 컴포지션 중간 프레임을 추출합니다: `ffmpeg -i final.mp4 -ss 00:00:05 -vframes 1 preview.png`.
6. **오디오가 예상대로 존재하는지 확인:** `ffprobe -v error -show_streams -select_streams a -of default=nw=1:nk=1 final.mp4 | head -1`.

`hyperframes render`가 실패하면 `npx hyperframes doctor`를 실행하고 보고할 때 그 출력을 첨부하세요.

## 참조 자료

- [composition.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/composition.md) — 데이터 속성, 타임라인 규칙, 타협할 수 없는 규칙, 타이포그래피/에셋 규칙
- [cli.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/cli.md) — 모든 CLI 명령어 (init, capture, lint, validate, inspect, preview, render, transcribe, tts, doctor, browser, info, upgrade, benchmark)
- [gsap.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/gsap.md) — HyperFrames를 위한 GSAP 핵심 API (tweens, eases, stagger, timelines, matchMedia)
- [features.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/features.md) — 캡션, TTS, 오디오 반응, 마커 하이라이팅, 전환 (필요에 따라 로드)
- [website-to-video.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/website-to-video.md) — 7단계 캡처-to-비디오 워크플로우
- [troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md) — OpenClaw 수정, 환경 변수, 일반적인 렌더링 오류
