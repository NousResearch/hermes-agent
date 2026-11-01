---
title: "Pretext — DOM 없는 텍스트 레이아웃으로 창의적인 브라우저 데모 만들기"
sidebar_label: "Pretext"
description: "DOM 없는 텍스트 레이아웃으로 창의적인 브라우저 데모 만들기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 skill의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Pretext

DOM 없는 텍스트 레이아웃으로 창의적인 브라우저 데모를 만듭니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 번들(기본 설치) |
| 경로 | `skills/creative/pretext` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `creative-coding`, `typography`, `pretext`, `ascii-art`, `canvas`, `generative`, `text-layout`, `kinetic-typography` |
| 관련 스킬 | [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Pretext 창의적 데모

## 개요

[`@chenglou/pretext`](https://github.com/chenglou/pretext)는 Cheng Lou(React 코어, ReasonML, Midjourney)가 만든 15KB 크기의 무의존성 TypeScript 라이브러리로, **DOM 없이 여러 줄 텍스트를 측정하고 레이아웃**합니다. 이 라이브러리가 하는 일은 하나입니다. `(text, font, width)`를 받아 줄바꿈, 줄별 너비, 자소별 위치, 전체 높이를 모두 캔버스 측정으로 반환합니다. 리플로우는 사용하지 않습니다.

겉보기에는 배관 작업처럼 들립니다. 하지만 그렇지 않습니다. 빠르고 기하학적이기 때문에 **창의적인 기본 요소**가 됩니다. 60fps로 움직이는 스프라이트 주변에 문단을 재배치하거나, 실제 단어로 레벨 지오메트리를 구성한 게임을 만들거나, 산문을 통해 ASCII 로고를 구동하거나, 자소별 정확한 시작 위치로 텍스트를 입자로 산산이 흩뜨리거나, `getBoundingClientRect`를 반복 호출하지 않고 여러 줄 UI를 내용에 꼭 맞게 배치할 수 있습니다.

이 스킬은 Hermes가 이를 활용해 사람들이 X에 올리고 싶어 할 **멋진 데모**를 만들 수 있도록 존재합니다. 커뮤니티 데모 모음은 `pretext.cool`과 `chenglou.me/pretext`를 참고하세요.

## 사용 시점

다음과 같은 요청을 받았을 때 사용하세요.
- "pretext 데모" / "멋진 pretext 작업" / "텍스트를 X로" 요청
- 움직이는 도형 주변으로 흐르는 텍스트(히어로 섹션, 편집 레이아웃, 애니메이션이 적용된 긴 형식의 페이지)
- 고정폭 래스터가 아닌 **실제 단어나 산문**을 사용하는 ASCII 아트 효과
- 텍스트로 플레이필드 / 장애물 / 벽돌을 구성하는 게임(글자로 만든 테트리스, 산문으로 만든 브레이크아웃)
- 자소별 물리 효과가 있는 키네틱 타이포그래피(산산이 흩뜨리기, 분산, 군집, 흐름)
- 특히 비라틴 문자나 여러 문자가 섞인 타이포그래픽 생성 예술
- 여러 줄 "내용 맞춤" UI(텍스트가 들어가는 가장 작은 컨테이너 너비)
- 렌더링 전에 줄바꿈을 알아야 하는 모든 작업

다음에는 사용하지 마세요.
- CSS로 레이아웃을 해결할 수 있는 정적 SVG/HTML 페이지 — CSS를 사용하세요.
- 서식 있는 텍스트 편집기, 일반 인라인 서식 엔진(pretext는 의도적으로 범위가 좁습니다)
- 이미지 → 텍스트(`ascii-art` / `ascii-video` 스킬 사용)
- 텍스트 역할이 없는 순수 캔버스 생성 예술 — `p5js` 사용

## 창작 기준

이것은 브라우저에서 렌더링되는 시각 예술입니다. Pretext는 숫자를 반환하고, **그림은 여러분이 그립니다.**

- **"hello world" 데모를 배포하지 마세요.** `hello-orb-flow.html` 템플릿은 *출발점*입니다. 제공하는 모든 데모에는 의도적인 색상, 움직임, 구도, 그리고 사용자가 요청하지 않았지만 마음에 들어 할 시각적 디테일 하나를 추가해야 합니다.
- **어두운 배경, 따뜻한 중심, 신중하게 고른 팔레트.** 고전적인 황색-검정 조합(CRT / 터미널)도 좋고, 차가운 흰색-짙은 회색 조합(편집 디자인)이나 채도를 낮춘 파스텔(리소그래프)도 좋습니다. 하나를 골라 끝까지 밀고 가세요.
- **비례 글꼴이 핵심입니다.** Pretext의 본질은 "고정폭이 아님"에 있습니다. Iowan Old Style, Inter, JetBrains Mono, Helvetica Neue 또는 가변 글꼴을 활용하세요. 기본 산세리프로 처리하지 마세요.
- **실제 소스/텍스트를 사용하고, lorem ipsum은 사용하지 마세요.** 모음에는 의미가 있어야 합니다. 짧은 선언문, 시, 실제 소스 코드, 발견한 텍스트, 라이브러리 자체의 README 등을 사용하고, `lorem ipsum`은 절대 사용하지 마세요.
- **첫 화면부터 완성도 있게 만드세요.** 로딩 상태나 빈 프레임을 두지 마세요. 데모는 열리는 즉시 출시 가능한 모습이어야 합니다.

## 스택

데모 하나당 빌드 단계가 없는 독립적인 HTML 파일 하나를 사용합니다.

| 계층 | 도구 | 용도 |
|-------|------|---------|
| 코어 | `@chenglou/pretext` via `esm.sh` CDN | 텍스트 측정 + 줄 레이아웃 |
| 렌더링 | HTML5 Canvas 2D | 글리프 렌더링, 자소별 프레임 구성 |
| 세분화 | `Intl.Segmenter` (내장) | 이모지 / CJK / 결합 문자의 자소 분할 |
| 상호작용 | Raw DOM events | 마우스 / 터치 / 휠 — 프레임워크 없음 |

```html
<script type="module">
import {
  prepare, layout,                   // use-case 1: simple height
  prepareWithSegments, layoutWithLines,  // use-case 2a: fixed-width lines
  layoutNextLineRange, materializeLineRange, // use-case 2b: streaming / variable width
  measureLineStats, walkLineRanges,  // stats without string allocation
} from "https://esm.sh/@chenglou/pretext@0.0.6";
</script>
```

버전을 고정하세요. 작성 시점에는 `@0.0.6`이 사용됩니다. 데모 동작이 이상하면 [npm](https://www.npmjs.com/package/@chenglou/pretext)에서 최신 버전을 확인하세요.

## 두 가지 사용 사례

거의 모든 작업은 다음 두 형태 중 하나로 환원됩니다. 둘 다 익혀 두세요.

### 사용 사례 1 — 측정한 뒤 CSS/DOM으로 렌더링

```js
const prepared = prepare(text, "16px Inter");
const { height, lineCount } = layout(prepared, 320, 20);
```

여전히 브라우저가 텍스트를 그리게 합니다. Pretext는 DOM을 읽지 않고도 주어진 너비에서 상자의 높이를 알려줄 뿐입니다. 다음에 사용하세요.
- 줄바꿈 텍스트가 들어 있는 행의 가상화 목록
- 정확한 카드 높이를 사용하는 메이슨리 레이아웃
- "이 라벨이 들어갈까?"와 같은 개발 시점 확인
- 원격 텍스트가 로드될 때 레이아웃 이동 방지

**`font`와 `letterSpacing`을 CSS와 정확히 동기화하세요.** 캔버스 `ctx.font` 형식(예: `"16px Inter"`, `"500 17px 'JetBrains Mono'"`)은 렌더링되는 CSS와 일치해야 합니다. 그렇지 않으면 측정값이 어긋납니다.

### 사용 사례 2 — 직접 측정하고 *직접* 렌더링

```js
const prepared = prepareWithSegments(text, FONT);
const { lines } = layoutWithLines(prepared, 320, 26);
for (let i = 0; i < lines.length; i++) {
  ctx.fillText(lines[i].text, 0, i * 26);
}
```

창의적인 작업은 여기서 이루어집니다. 그리는 작업을 직접 소유하므로 다음을 할 수 있습니다.
- 캔버스, SVG, WebGL 또는 어떤 좌표계로든 렌더링
- 자소별 변환(회전, 지터, 크기 조절, 불투명도) 적용
- 줄 메타데이터(너비, 자소 위치)를 기하 정보로 사용

**줄마다 너비가 달라지는** 흐름(도형 주변의 텍스트, 도넛 띠 안의 텍스트, 직사각형이 아닌 열의 텍스트)은 다음과 같이 처리합니다.

```js
let cursor = { segmentIndex: 0, graphemeIndex: 0 };
let y = 0;
while (true) {
  const lineWidth = widthAtY(y);  // your function: how wide is the corridor at this y?
  const range = layoutNextLineRange(prepared, cursor, lineWidth);
  if (!range) break;
  const line = materializeLineRange(prepared, range);
  ctx.fillText(line.text, leftEdgeAtY(y), y);
  cursor = range.end;
  y += lineHeight;
}
```

이것이 라이브러리 전체에서 가장 중요한 패턴입니다. "드래그한 스프라이트 주변으로 흐르는 텍스트"를 가능하게 하는 패턴이며, X에서 바이럴이 된 데모도 바로 이것을 사용합니다.

### 알아둘 만한 헬퍼

- `measureLineStats(prepared, maxWidth)` → `{ lineCount, maxLineWidth }` — 가장 넓은 줄, 즉 여러 줄 내용 맞춤 너비입니다.
- `walkLineRanges(prepared, maxWidth, callback)` — 문자열을 할당하지 않고 줄을 순회합니다. 문자가 필요하지 않은 자소 단위 통계/물리에 사용하세요.
- `@chenglou/pretext/rich-inline` — 글꼴 / 칩 / 멘션이 섞인 문단을 위한 동일한 시스템입니다. 서브패스에서 import하세요.

## 데모 레시피 패턴

커뮤니티 모음(아래 `references/patterns.md` 참고)은 몇 가지 강력한 패턴으로 분류됩니다. 하나를 골라 변주하세요. 요청받지 않았다면 새로운 범주를 만들지 마세요.

| 패턴 | 핵심 API | 예시 아이디어 |
|---|---|---|
| **장애물 주변 재배치** | `layoutNextLineRange` + 행별 너비 함수 | 드래그한 커서 스프라이트 주변으로 갈라지는 편집 문단 |
| **텍스트를 기하로 사용하는 게임** | `layoutWithLines` + 줄별 충돌 사각형 | 측정된 단어가 각 벽돌인 브레이크아웃 |
| **산산이 흩뜨리기 / 입자** | `walkLineRanges` → 자소별 (x,y) → 물리 | 클릭하면 글자로 폭발하는 문장 |
| **ASCII 장애물 타이포그래피** | `layoutNextLineRange` + 측정된 행별 장애물 구간 | 비트맵 ASCII 로고, 형태가 변하고 드래그할 수 있는 와이어 오브젝트가 실제 지오메트리 주변으로 텍스트를 열어 줌 |
| **편집용 다단** | 열마다 `layoutNextLineRange` + 공유 커서 | 인용문이 끌어당기는 애니메이션 잡지 펼침면 |
| **키네틱 타입** | `layoutWithLines` + 시간에 따른 줄별 변환 | 스타워즈 오프닝 크롤, 웨이브, 바운스, 글리치 |
| **여러 줄 내용 맞춤** | `measureLineStats` | 가장 조밀한 컨테이너에 맞춰 자동 크기 조절되는 인용 카드 |

작동하는 단일 파일 시작점은 `templates/donut-orbit.html`과 `templates/hello-orb-flow.html`을 참고하세요.

## 워크플로

1. 위 표에서 사용자의 요구사항에 맞는 **패턴을 고릅니다.**
2. **템플릿에서 시작합니다.**
   - `templates/hello-orb-flow.html` — 움직이는 구 주위로 텍스트가 재배치됨(장애물 주변 재배치 패턴)
   - `templates/donut-orbit.html` — 고급 예시: 측정된 ASCII 로고 장애물, 드래그 가능한 와이어 구/큐브, 형태가 변하는 필드, 선택 가능한 DOM 텍스트, 개발 전용 컨트롤
   - `write_file`로 `/tmp/` 또는 사용자의 작업 공간에 새 `.html` 파일을 만듭니다.
3. **모음(corpus)을 요구사항에 맞는 의도적인 내용으로 교체합니다.** 실제 산문 10~100문장을 사용하고, lorem은 사용하지 마세요.
4. **미적 요소를 조정합니다.** 글꼴, 팔레트, 구도, 상호작용을 조정하세요. 이것이 작업의 핵심이므로 건너뛰지 마세요.
5. **로컬에서 확인합니다.**
   ```sh
   cd <dir-with-html> && python3 -m http.server 8765
   # then open http://localhost:8765/<file>.html
   ```
6. **콘솔을 확인합니다.** 잘못된 글꼴 문자열로 `prepareWithSegments`를 호출하면 pretext가 오류를 발생시킵니다. `Intl.Segmenter`는 모든 최신 브라우저에서 사용할 수 있습니다.
7. **사용자에게 코드만이 아니라 파일 경로를 보여 줍니다.** 사용자는 파일을 열고 싶어 합니다.

## 성능 참고 사항

- `prepare()` / `prepareWithSegments()` 호출이 비용이 큽니다. 텍스트+글꼴 쌍마다 **한 번만** 실행하세요. 핸들을 캐시하세요.
- 크기를 조정할 때는 `layout()` / `layoutWithLines()`만 다시 실행하고, 다시 준비하지 마세요.
- 텍스트는 바뀌지 않고 지오메트리만 바뀌는 프레임별 애니메이션에서는 일반적인 길이의 문단이라면 `layoutNextLineRange`를 빠른 반복문에서 매 프레임 60fps로 실행해도 충분히 빠릅니다.
- 매 프레임 ASCII 마스크를 렌더링할 때는 셀 버퍼(`Uint8Array`/typed arrays)를 유지하고, 셀 또는 투영된 지오메트리에서 측정된 행별 장애물 구간을 도출한 다음, 구간을 병합해 텍스트를 그리기 전에 `layoutNextLineRange`에 전달하세요.
- 시각 애니메이션과 레이아웃 애니메이션을 결합하세요. 구가 큐브로 변한다면 동일한 값으로 렌더링된 셀 버퍼와 장애물 구간을 모두 트윈하세요. 그렇지 않으면 데모가 실제로 재배치되는 대신 덧칠된 것처럼 보입니다.
- 페이드에는 글리프 강도나 장애물 크기를 바꾸기보다 레이어 불투명도를 사용하세요. 일시적인 ASCII 스프라이트를 별도의 캔버스에 두고 CSS/GSAP 불투명도로 캔버스를 페이드하면 지오메트리가 줄어드는 것처럼 보이지 않습니다.
- 캔버스 `ctx.font` 설정은 놀랄 만큼 느립니다. 글꼴이 변하지 않는다면 `fillText`를 호출할 때마다가 아니라 프레임마다 **한 번** 설정하세요.

## 흔한 함정

1. **CSS/캔버스 글꼴 문자열이 어긋나는 경우.** `ctx.font = "16px Inter"`로 측정했는데 CSS가 `font-family: Inter, sans-serif; font-size: 16px`라고 합시다. Inter가 로드되면 괜찮습니다. Inter가 404를 반환하면 CSS는 sans-serif로 대체되고 측정값이 5~20% 어긋납니다. 항상 글꼴을 `preload`하거나 웹 안전 글꼴을 사용하세요.

2. **애니메이션 루프 안에서 다시 준비하는 경우.** 저렴한 것은 `layout*`뿐입니다. 매 프레임 `prepare`를 다시 호출하면 성능이 크게 떨어집니다. 준비된 핸들을 모듈 스코프에 보관하세요.

3. **자소 분할에 `Intl.Segmenter`를 잊는 경우.** 이모지, 결합 문자, CJK에서 `"é".split("")`은 두 문자를 반환합니다. 개별적으로 보이는 글리프를 샘플링할 때는 `new Intl.Segmenter(undefined, { granularity: "grapheme" })`을 사용하세요.

4. **`extraWidth` 없이 `break: 'never'` 칩을 사용하는 경우.** `rich-inline`에서 원자적 칩/멘션에 `break: 'never'`를 사용한다면 필 패딩을 위한 `extraWidth`도 반드시 제공해야 합니다. 그렇지 않으면 칩 장식이 컨테이너 밖으로 넘칩니다.

5. **TypeScript 전용 진입점이 있는 `unpkg`에서 `@chenglou/pretext`를 사용하는 경우.** `esm.sh`를 사용하세요. 이 서비스는 TS export를 브라우저에서 바로 사용할 수 있는 ESM으로 자동 컴파일합니다. `unpkg`는 404를 반환하거나 원시 TS를 제공합니다.

6. **고정폭 대체 글꼴이 핵심을 조용히 지워 버리는 경우.** 고정폭처럼 보이는 결과가 나오는 사용자는 CSS `font-family`가 `monospace`로 넘어갔을 수 있습니다. DevTools에서 실제 렌더링된 글꼴을 확인하세요.

7. **도형 주변으로 흐르게 할 때 행을 건너뛰지 않고 너비를 조정하는 경우.** 이 행의 통로가 줄 하나를 넣기에 너무 좁다면 작은 `maxWidth`를 `layoutNextLineRange`에 넘기지 말고 *행을 건너뛰세요* (`y += lineHeight; continue;`). pretext는 한 자소짜리 줄을 반환해 깨져 보이게 됩니다.

8. **차가운 데모를 배포하는 경우.** 기본 첫 화면은 튜토리얼 수준으로 보입니다. 비네트, 은은한 스캔라인, 유휴 자동 움직임, 신중하게 고른 상호작용 하나(드래그, 호버, 스크롤, 클릭)를 추가하세요. 이것들이 없으면 "멋진 pretext 데모"가 "README를 보고 만든 인턴의 재현작"처럼 느껴집니다.

## 확인 목록

- [ ] 데모는 독립적인 `.html` 파일 하나이며, 더블클릭 또는 `python3 -m http.server`로 열립니다.
- [ ] `@chenglou/pretext`를 고정된 버전으로 `esm.sh`를 통해 import합니다.
- [ ] 모음은 lorem ipsum이 아닌 실제 산문이며 데모의 콘셉트와 맞습니다.
- [ ] `prepare`에 전달하는 글꼴 문자열이 CSS 글꼴과 정확히 일치합니다.
- [ ] `prepare()` / `prepareWithSegments()`를 프레임마다가 아니라 한 번 호출합니다.
- [ ] 어두운 배경과 신중하게 고른 팔레트를 사용합니다 — 기본 흰색 캔버스가 아닙니다.
- [ ] 상호작용 반응(드래그 / 호버 / 스크롤 / 클릭)이 하나 이상 있거나 유휴 자동 움직임이 있습니다.
- [ ] `python3 -m http.server`로 로컬 테스트를 완료했고 콘솔 오류가 없음을 확인했습니다.
- [ ] 중급형 노트북에서 60fps로 동작하거나, 우아한 성능 저하가 문서화되어 있습니다.
- [ ] 사용자가 요청하지 않은 "한 단계 더 나아간" 디테일이 하나 있습니다.

## 참고: 커뮤니티 데모

영감과 패턴을 위해 다음을 복제해 보세요(모두 MIT 계열이며 [pretext.cool](https://www.pretext.cool/)에서 연결됩니다).

- **Pretext Breaker** — 단어 벽돌을 사용하는 브레이크아웃 — `github.com/rinesh/pretext-breaker`
- **Tetris × Pretext** — `github.com/shinichimochizuki/tetris-pretext`
- **Dragon animation** — `github.com/qtakmalay/PreTextExperiments`
- **Somnai editorial engine** — `github.com/somnai-dreams/pretext-demos`
- **Bad Apple!! ASCII** — `github.com/frmlinn/bad-apple-pretext`
- **Drag-sprite reflow** — `github.com/dokobot/pretext-demo`
- **Alarmy editorial clock** — `github.com/SmisLee/alarmy-pretext-demo`

공식 플레이그라운드: [chenglou.me/pretext](https://chenglou.me/pretext/) — accordion, bubbles, dynamic-layout, editorial-engine, justification-comparison, masonry, markdown-chat, rich-note.
