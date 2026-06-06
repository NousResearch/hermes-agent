---
title: "Pretext"
sidebar_label: "Pretext"
description: "@chenglou/pretext를 활용한 창의적인 브라우저 데모 빌드 시 사용 — ASCII 아트, 장애물을 우회하는 타이포그래피 흐름, 텍스트를 기하학적 요소로 활용하는 게임..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pretext

@chenglou/pretext를 활용하여 창의적인 브라우저 데모를 빌드할 때 사용합니다 — ASCII 아트, 장애물을 우회하는 타이포그래피 흐름, 텍스트를 기하학적 요소로 활용하는 게임, 키네틱 타이포그래피, 텍스트 기반 생성형 아트를 위한 DOM 없는 텍스트 레이아웃 도구입니다. 기본적으로 단일 파일 HTML 데모를 생성합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/pretext` |
| 버전 | `1.0.0` |
| 저자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `creative-coding`, `typography`, `pretext`, `ascii-art`, `canvas`, `generative`, `text-layout`, `kinetic-typography` |
| 관련 스킬 | [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# Pretext Creative Demos

## 개요

[`@chenglou/pretext`](https://github.com/chenglou/pretext)는 Cheng Lou (React core, ReasonML, Midjourney 출신)가 만든 15KB의 무설치(zero-dependency) TypeScript 라이브러리로, **DOM을 사용하지 않는 다중 행 텍스트 측정 및 레이아웃** 기능을 제공합니다. 이 도구는 단 하나의 기능을 수행합니다: `(텍스트, 폰트, 너비)`가 주어지면 캔버스 측정을 통해(리플로우 없이) 줄 바꿈, 줄별 너비, 문자소별 위치 및 전체 높이를 반환합니다.

단순한 도구처럼 들릴 수 있지만, 이는 **창의적인 기본 요소(creative primitive)**입니다. 빠르고 기하학적이기 때문에 움직이는 스프라이트 주위로 단락을 60fps로 재정렬하거나, 실제 단어로 구성된 지형으로 게임을 만들거나, 산문을 통해 ASCII 로고를 주도하거나, 텍스트를 정확한 문자소별 시작 위치와 함께 입자(particle)로 부수거나, `getBoundingClientRect`로 인한 성능 저하 없이 다중 행 UI를 래핑할 수 있습니다.

이 스킬은 Hermes가 사람들이 X(트위터)에 올릴 만한 **멋진 데모**를 만들기 위해 존재합니다. 커뮤니티의 데모 코퍼스를 보려면 `pretext.cool` 및 `chenglou.me/pretext`를 참조하세요.

## 이 스킬을 사용하는 시기

사용자가 다음을 요청할 때 사용하세요:
- "pretext 데모" / "멋진 pretext 기능" / "text-as-X (텍스트를 활용한 X)"
- 움직이는 도형 주위로 흐르는 텍스트 (히어로 섹션, 에디토리얼 레이아웃, 애니메이션 긴 형식 페이지)
- 단색 래스터 이미지가 아닌 **실제 단어나 산문**을 사용하는 ASCII 아트 효과
- 텍스트로 만들어진 경기장 / 장애물 / 벽돌이 있는 게임 (글자로 만든 테트리스, 산문으로 만든 벽돌깨기)
- 문자별 물리 엔진이 적용된 키네틱 타이포그래피 (산산조각, 흩어짐, 무리 지음, 흐름)
- 타이포그래피 생성형 아트, 특히 비라틴계 문자나 혼합 스크립트 사용 시
- 다중 행 "수축 포장(shrink-wrap)" UI (텍스트에 딱 맞는 가장 작은 컨테이너 너비)
- 렌더링하기 *전*에 줄 바꿈을 알아야 하는 모든 작업

다음에 대해서는 사용하지 마세요:
- CSS가 이미 레이아웃을 해결하는 정적 SVG/HTML 페이지 — 그냥 CSS를 사용하세요.
- 서식 있는 텍스트 편집기(Rich text editors), 일반적인 인라인 서식 지정 엔진 (pretext는 의도적으로 제한적입니다).
- 이미지 → 텍스트 변환 (`ascii-art` / `ascii-video` 스킬 사용).
- 텍스트 역할이 전혀 없는 순수 캔버스 생성형 아트 — `p5js` 사용.

## 창의적 기준 (Creative Standard)

이것은 브라우저에서 렌더링되는 시각 예술입니다. Pretext는 숫자를 반환하며, 렌더링은 **당신**이 직접 해야 합니다.

- **"hello world" 수준의 데모를 제공하지 마세요.** `hello-orb-flow.html` 템플릿은 *시작*점일 뿐입니다. 제공되는 모든 데모에는 의도적인 색상, 모션, 구도, 그리고 사용자가 요구하지 않았지만 좋아할 만한 시각적 디테일이 하나 이상 추가되어야 합니다.
- **어두운 배경, 따뜻한 중심, 신중한 팔레트.** 고전적인 검은 바탕에 호박색(CRT / 터미널)도 좋고, 차콜 바탕에 차가운 흰색(에디토리얼)이나 채도가 낮은 파스텔 톤(리소그래프)도 좋습니다. 하나를 선택하고 집중하세요.
- **가변 폭 폰트(Proportional fonts)가 핵심입니다.** Pretext의 분위기는 "고정 폭이 아님"에 있습니다 — 이를 적극 활용하세요. Iowan Old Style, Inter, JetBrains Mono, Helvetica Neue 또는 가변 폰트를 사용하세요. 절대로 기본 산세리프체를 쓰지 마세요.
- **lorem ipsum 대신 실제 텍스트 사용.** 코퍼스는 의미가 있어야 합니다. 짧은 선언문, 시, 실제 소스 코드, 발견된 텍스트, 라이브러리 자체의 README 등을 사용하세요 — 절대 `lorem ipsum`은 안 됩니다.
- **탁월한 첫 화면(First-paint).** 로딩 상태나 빈 프레임이 없어야 합니다. 데모는 열리는 즉시 배포할 수 있는 상태로 보여야 합니다.

## 기술 스택 (Stack)

데모당 단일의 독립적인 HTML 파일을 생성합니다. 빌드 단계가 없습니다.

| 계층 | 도구 | 목적 |
|-------|------|---------|
| 코어 (Core) | `esm.sh` CDN을 통한 `@chenglou/pretext` | 텍스트 측정 + 줄 레이아웃 |
| 렌더 (Render) | HTML5 Canvas 2D | 글리프 렌더링, 프레임별 컴포지션 |
| 세분화 (Segmentation) | `Intl.Segmenter` (내장) | 이모지 / CJK / 결합 부호를 위한 문자소 분할 |
| 상호작용 (Interaction) | 순수 DOM 이벤트 | 마우스 / 터치 / 휠 — 프레임워크 없음 |

```html
<script type="module">
import {
  prepare, layout,                   // 사용 사례 1: 단순한 높이
  prepareWithSegments, layoutWithLines,  // 사용 사례 2a: 고정 너비 줄
  layoutNextLineRange, materializeLineRange, // 사용 사례 2b: 스트리밍 / 가변 너비
  measureLineStats, walkLineRanges,  // 문자열 할당 없는 통계
} from "https://esm.sh/@chenglou/pretext@0.0.6";
</script>
```

버전을 고정하세요. 작성 당시 `@0.0.6`이었습니다 — 데모 동작이 이상하다면 [npm](https://www.npmjs.com/package/@chenglou/pretext)에서 최신 버전을 확인하세요.

## 두 가지 사용 사례 (The Two Use Cases)

거의 모든 것은 이 두 가지 형태 중 하나로 귀결됩니다. 둘 다 배우세요.

### 사용 사례 1 — 측정 후 CSS/DOM으로 렌더링

```js
const prepared = prepare(text, "16px Inter");
const { height, lineCount } = layout(prepared, 320, 20);
```

브라우저가 계속 텍스트를 그리게 합니다. Pretext는 DOM 읽기 **없이** 주어진 너비에서 상자의 높이가 얼마나 될지 알려줍니다. 다음 경우에 사용하세요:
- 텍스트 래핑이 포함된 행이 있는 가상화된 목록
- 정확한 카드 높이를 가진 Masonry 레이아웃
- 개발 단계에서 "이 라벨이 맞을까?" 확인
- 원격 텍스트가 로드될 때 레이아웃 이동(layout shift) 방지

**`font`와 `letterSpacing`이 CSS와 정확히 동기화되게 하세요.** 캔버스 `ctx.font` 형식 (예: `"16px Inter"`, `"500 17px 'JetBrains Mono'"`)은 렌더링된 CSS와 일치해야 하며, 그렇지 않으면 측정값이 어긋납니다.

### 사용 사례 2 — 직접 측정 *및* 렌더링

```js
const prepared = prepareWithSegments(text, FONT);
const { lines } = layoutWithLines(prepared, 320, 26);
for (let i = 0; i < lines.length; i++) {
  ctx.fillText(lines[i].text, 0, i * 26);
}
```

이곳이 창의적인 작업이 이루어지는 곳입니다. 드로잉을 완전히 제어할 수 있으므로 다음이 가능합니다:
- 캔버스, SVG, WebGL 또는 모든 좌표계로 렌더링
- 문자(글리프)별 변형(회전, 지터, 크기, 불투명도) 적용
- 줄 메타데이터(너비, 문자소 위치)를 기하학 요소로 사용

**줄마다 너비가 가변적인 경우**(도형 주위의 텍스트, 도넛 밴드의 텍스트, 직사각형이 아닌 열의 텍스트):

```js
let cursor = { segmentIndex: 0, graphemeIndex: 0 };
let y = 0;
while (true) {
  const lineWidth = widthAtY(y);  // 당신의 함수: 이 y 좌표에서 공간 너비가 얼마나 되는가?
  const range = layoutNextLineRange(prepared, cursor, lineWidth);
  if (!range) break;
  const line = materializeLineRange(prepared, range);
  ctx.fillText(line.text, leftEdgeAtY(y), y);
  cursor = range.end;
  y += lineHeight;
}
```

이것이 전체 라이브러리에서 가장 중요한 패턴입니다. 이 패턴이 X에서 화제가 된 "드래그된 스프라이트 주위로 흐르는 텍스트" 데모를 가능하게 합니다.

### 알아두면 유용한 헬퍼 함수들

- `measureLineStats(prepared, maxWidth)` → `{ lineCount, maxLineWidth }` — 가장 넓은 줄, 즉 다중 행 수축 포장(shrink-wrap) 너비.
- `walkLineRanges(prepared, maxWidth, callback)` — 문자열을 할당하지 않고 줄을 반복합니다. 문자가 필요하지 않고 문자소에 대한 통계나 물리가 필요할 때 사용하세요.
- `@chenglou/pretext/rich-inline` — 폰트 / 칩 / 멘션을 혼합하는 단락에 대해 동일한 시스템. 서브 경로에서 임포트하세요.

## 데모 레시피 패턴

커뮤니티 코퍼스(`references/patterns.md` 참조)는 몇 가지 강력한 패턴으로 분류됩니다. 하나를 선택하고 변형을 가하세요 — 요청받지 않는 한 새로운 범주를 발명하지 마세요.

| 패턴 | 핵심 API | 아이디어 예시 |
|---|---|---|
| **장애물 우회 (Reflow around obstacle)** | `layoutNextLineRange` + 행별 너비 함수 | 드래그 가능한 커서 스프라이트를 피해서 흐르는 에디토리얼 단락 |
| **텍스트 지형 게임 (Text-as-geometry game)** | `layoutWithLines` + 줄별 충돌 상자 | 각 벽돌이 측정된 단어인 벽돌깨기 |
| **산산조각 / 입자 (Shatter / particles)** | `walkLineRanges` → 문자소별 (x,y) → 물리 엔진 | 클릭 시 글자들로 폭발하는 문장 |
| **ASCII 장애물 타이포그래피** | `layoutNextLineRange` + 측정된 행별 장애물 범위 | 텍스트가 실제 지형 주위로 열리게 하는 비트맵 ASCII 로고, 도형 모핑, 드래그 가능한 철사 물체 |
| **에디토리얼 다단 레이아웃** | 열 단위 `layoutNextLineRange` + 공유 커서 | 인용문(pull quotes)이 포함된 애니메이션 매거진 펼침면 |
| **키네틱 타이포그래피 (Kinetic type)** | `layoutWithLines` + 시간에 따른 줄별 변형 | 스타워즈 오프닝 크롤, 파도, 바운스, 글리치 |
| **다중 행 수축 포장 (Multiline shrink-wrap)** | `measureLineStats` | 텍스트에 가장 타이트한 컨테이너 크기로 자동 조절되는 인용구 카드 |

작동하는 단일 파일 시작점을 보려면 `templates/donut-orbit.html` 및 `templates/hello-orb-flow.html`을 참조하세요.

## 워크플로우

1. 사용자의 요청(brief)을 바탕으로 위 표에서 **패턴을 선택**합니다.
2. **템플릿에서 시작**합니다:
   - `templates/hello-orb-flow.html` — 움직이는 구 주위로 텍스트가 리플로우되는 기능 (장애물 우회 패턴)
   - `templates/donut-orbit.html` — 고급 예제: 측정된 ASCII 로고 장애물, 드래그 가능한 구/정육면체 철사 모델, 모핑 도형 영역, 선택 가능한 DOM 텍스트, 개발자 전용 컨트롤
   - `/tmp/`나 사용자의 작업 공간에 새로운 `.html` 파일로 `write_file` 합니다.
3. 요구 사항에 맞는 의도적인 것으로 **코퍼스를 교체**합니다. 실제 산문 10-100 문장을 사용하고 lorem ipsum은 사용하지 마세요.
4. **미학(aesthetic) 조정** — 글꼴, 팔레트, 구도, 상호작용. 이것이 핵심 작업이므로 건너뛰지 마세요.
5. **로컬에서 검증**:
   ```sh
   cd <html이 있는 폴더> && python3 -m http.server 8765
   # 그다음 http://localhost:8765/<파일>.html 접속
   ```
6. **콘솔 확인** — 잘못된 글꼴 문자열로 `prepareWithSegments`가 호출되면 pretext에서 오류를 발생시킵니다; `Intl.Segmenter`는 모든 최신 브라우저에서 사용 가능합니다.
7. 사용자가 직접 열 수 있도록 코드만이 아니라 **파일 경로를 사용자에게 보여주세요.**

## 성능 노트

- `prepare()` / `prepareWithSegments()`는 무거운 호출입니다. 텍스트+폰트 쌍당 **한 번만** 실행하세요. 핸들을 캐시해두세요.
- 크기 조절(resize) 시에는 `layout()` / `layoutWithLines()`만 다시 실행하세요 — 절대 `prepare`를 다시 호출하지 마세요.
- 텍스트는 변하지 않지만 지형이 변하는 프레임 단위 애니메이션의 경우, 짧은 루프 내의 `layoutNextLineRange`는 일반적인 길이의 단락에 대해 60fps로 매 프레임 실행할 수 있을 만큼 가볍습니다.
- 매 프레임 ASCII 마스크를 렌더링할 때는, 셀 버퍼(`Uint8Array`/타입 배열)를 유지하고, 셀이나 투영된 지형에서 측정된 행별 장애물 범위를 도출하여 합친 다음, 텍스트를 그리기 전에 이 범위를 `layoutNextLineRange`에 공급하세요.
- 시각적 애니메이션과 레이아웃 애니메이션을 결합된 상태로 유지하세요. 구(sphere)가 정육면체(cube)로 모핑되면, 렌더링된 셀 버퍼와 장애물 범위를 모두 같은 값으로 트윈(tween)하세요. 그렇지 않으면 데모가 물리적으로 리플로우되는 대신 그 위에 덧그려진 것처럼 보입니다.
- 페이드 인/아웃 효과를 줄 때는 글리프 강도나 장애물 크기를 변경하는 것보다 레이어 불투명도를 사용하는 것이 좋습니다. 임시 ASCII 스프라이트를 별도의 캔버스에 놓고 CSS/GSAP 불투명도로 캔버스를 페이드시켜 도형이 작아지는 것처럼 보이지 않게 하세요.
- 캔버스의 `ctx.font` 설정은 의외로 느립니다. 폰트가 변하지 않는 경우 `fillText` 호출 때마다 설정하지 말고 프레임당 **한 번만** 설정하세요.

## 흔히 범하는 실수 (Common Pitfalls)

1. **CSS/캔버스 폰트 문자열이 달라짐.** 캔버스는 `ctx.font = "16px Inter"`로 측정했지만, CSS에는 `font-family: Inter, sans-serif; font-size: 16px`로 설정된 경우. Inter가 정상적으로 로드되면 괜찮습니다. 하지만 Inter 로드에 실패하여 CSS가 sans-serif로 대체되면 측정값이 5-20% 어긋납니다. 항상 글꼴을 `preload`하거나 웹 안전 글꼴 제품군을 사용하세요.

2. **애니메이션 루프 내에서 재준비(Re-preparing).** `layout*` 함수들만 가볍습니다. 매 프레임 `prepare`를 다시 호출하면 성능이 크게 저하됩니다. 준비된 핸들을 모듈 스코프에 보관하세요.

3. **문자소 분할을 위해 `Intl.Segmenter` 사용을 잊음.** 이모지, 결합 부호, CJK 문자에서 `"é".split("")`은 두 개의 문자를 반환합니다. 개별적으로 표시되는 글리프를 샘플링할 때는 `new Intl.Segmenter(undefined, { granularity: "grapheme" })`를 사용하세요.

4. **`rich-inline`에서 `extraWidth` 없이 `break: 'never'` 칩 사용.** `rich-inline`에서 분할할 수 없는 원자성 칩/멘션에 대해 `break: 'never'`를 사용하는 경우 알약(pill) 모양 패딩을 위한 `extraWidth`도 제공해야 합니다 — 그렇지 않으면 칩 디자인이 컨테이너 밖으로 넘칩니다.

5. **TypeScript 전용 진입점이 있는 `unpkg`에서 `@chenglou/pretext` 사용.** `esm.sh`를 사용하세요 — TS 익스포트를 브라우저에 적합한 ESM으로 자동 컴파일합니다. `unpkg`은 404를 반환하거나 원시 TS를 제공합니다.

6. **모노스페이스 대체를 사용해 핵심 아이디어를 훼손함.** 고정폭(monospace) 스타일의 출력을 보는 사용자는 보통 CSS의 `font-family` 설정이 결국 `monospace`로 떨어졌을 때입니다. DevTools를 통해 실제로 렌더링된 폰트를 확인하세요.

7. **도형 주위로 텍스트가 흐를 때 너비를 조절하는 대신 행 건너뛰기.** 해당 행의 공간이 좁아 텍스트가 들어갈 수 없다면 `layoutNextLineRange`에 아주 작은 maxWidth를 전달하기보다 해당 행을 아예 건너뛰세요(`y += lineHeight; continue;`). 억지로 맞추려 하면 pretext가 한 글자짜리 엉망인 줄을 반환합니다.

8. **미완성(Cold) 데모 배포.** 기본 첫 화면은 너무 튜토리얼 수준처럼 보입니다. 비네트 효과(vignette), 미세한 스캔라인, 유휴 자동 모션, 신중하게 선택된 상호작용 피드백(드래그, 호버, 스크롤, 클릭 등)을 추가하세요. 이런 요소들이 없다면 "멋진 pretext 데모"는 그저 "인턴이 다시 만든 README 수준"으로 전락합니다.

## 검증 체크리스트

- [ ] 데모가 독립적인 단일 `.html` 파일인지 여부 — 더블 클릭하거나 `python3 -m http.server`로 열려야 함
- [ ] `@chenglou/pretext`가 고정된 버전의 `esm.sh`를 통해 임포트되었는지 여부
- [ ] 코퍼스가 lorem ipsum이 아닌 실제 산문이며 데모의 콘셉트와 일치하는지 여부
- [ ] `prepare`에 전달된 글꼴 문자열이 CSS 글꼴과 정확히 일치하는지 여부
- [ ] `prepare()` / `prepareWithSegments()`가 프레임별이 아닌 한 번만 호출되었는지 여부
- [ ] 기본 흰색 캔버스가 아닌, 어두운 배경과 신중한 팔레트 사용 여부
- [ ] 상호작용 피드백(드래그 / 호버 / 스크롤 / 클릭) 또는 유휴 자동 모션이 하나 이상 있는지 여부
- [ ] 로컬에서 `python3 -m http.server`로 테스트하고 콘솔 오류가 없음을 확인했는지 여부
- [ ] 중간 사양 노트북에서 60fps 유지 (또는 우아한 성능 저하가 문서화되었는지) 여부
- [ ] 사용자가 요구하지 않았지만 특별히 신경 쓴 추가 디테일(extra mile detail) 하나 이상 추가 여부

## 참조: 커뮤니티 데모

영감 / 패턴을 위해 다음을 클론하세요 (모두 MIT 유사 라이선스, [pretext.cool](https://www.pretext.cool/)에서 링크됨):

- **Pretext Breaker** — 단어로 된 벽돌을 부수는 브레이크아웃 — `github.com/rinesh/pretext-breaker`
- **Tetris × Pretext** — `github.com/shinichimochizuki/tetris-pretext`
- **Dragon animation** — `github.com/qtakmalay/PreTextExperiments`
- **Somnai editorial engine** — `github.com/somnai-dreams/pretext-demos`
- **Bad Apple!! ASCII** — `github.com/frmlinn/bad-apple-pretext`
- **Drag-sprite reflow** — `github.com/dokobot/pretext-demo`
- **Alarmy editorial clock** — `github.com/SmisLee/alarmy-pretext-demo`

공식 플레이그라운드: [chenglou.me/pretext](https://chenglou.me/pretext/) — 아코디언, 버블, 동적 레이아웃, 에디토리얼 엔진, 양쪽 정렬 비교, masonry, 마크다운 채팅, 리치 노트.
