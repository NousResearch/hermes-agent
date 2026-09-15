---
title: "Excalidraw — 손으로 그린 Excalidraw JSON 다이어그램 (아키텍처, 플로우, 시퀀스)"
sidebar_label: "Excalidraw"
description: "손으로 그린 Excalidraw JSON 다이어그램 (아키텍처, 플로우, 시퀀스)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Excalidraw

손으로 그린 Excalidraw JSON 다이어그램 (아키텍처, 플로우, 시퀀스).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함 (기본 설치됨) |
| 경로 | `skills/creative/excalidraw` |
| 버전 | `1.0.1` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Excalidraw`, `Diagrams`, `Flowcharts`, `Architecture`, `Visualization`, `JSON` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Excalidraw 다이어그램 스킬

표준 Excalidraw 요소 JSON을 작성하고 `.excalidraw` 파일로 저장해 다이어그램을 만듭니다. 이 파일은 [excalidraw.com](https://excalidraw.com)에 끌어다 놓아 보고 편집할 수 있습니다. 계정, API 키, 렌더링 라이브러리가 필요하지 않고 JSON만 있으면 됩니다.

## 사용 시점

아키텍처 다이어그램, 플로우차트, 시퀀스 다이어그램, 개념 맵 등을 위한 `.excalidraw` 파일을 생성합니다. 파일은 excalidraw.com에서 열거나 업로드해 공유 가능한 링크를 만들 수 있습니다.

## 워크플로

1. **이 스킬 로드** (이미 완료함)
2. **요소 JSON 작성** -- Excalidraw 요소 객체 배열
3. **파일 저장** -- `write_file`을 사용해 `.excalidraw` 파일 생성
4. **선택적 업로드** -- 공유 가능한 링크를 만들려면 `terminal`을 통해 `scripts/upload.py` 사용

### 다이어그램 저장

요소 배열을 표준 `.excalidraw` 봉투로 감싸고 `write_file`로 저장합니다.

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "hermes-agent",
  "elements": [ ...your elements array here... ],
  "appState": {
    "viewBackgroundColor": "#ffffff"
  }
}
```

예: `~/diagrams/my_diagram.excalidraw`와 같은 경로에 저장합니다.

### 공유 가능한 링크를 위한 업로드

다음 터미널 명령으로 업로드 스크립트(이 스킬의 `scripts/` 디렉터리에 있음)를 실행합니다.

```bash
python skills/creative/excalidraw/scripts/upload.py ~/diagrams/my_diagram.excalidraw
```

이 명령은 excalidraw.com에 업로드하며(계정 불필요) 공유 가능한 URL을 출력합니다. `cryptography` pip 패키지가 필요합니다(`pip install cryptography`).

---

## 요소 형식 참고

### 필수 필드(모든 요소)
`type`, `id` (고유 문자열), `x`, `y`, `width`, `height`

### 기본값(생략 가능 -- 자동으로 적용됨)
- `strokeColor`: `"#1e1e1e"`
- `backgroundColor`: `"transparent"`
- `fillStyle`: `"solid"`
- `strokeWidth`: `2`
- `roughness`: `1` (손으로 그린 모양)
- `opacity`: `100`

캔버스 배경은 흰색입니다.

### 요소 유형

**사각형**:
```json
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 100 }
```
- 둥근 모서리는 `roundness: { "type": 3 }`
- 채우려면 `backgroundColor: "#a5d8ff"`, `fillStyle: "solid"`

**타원**:
```json
{ "type": "ellipse", "id": "e1", "x": 100, "y": 100, "width": 150, "height": 150 }
```

**마름모**:
```json
{ "type": "diamond", "id": "d1", "x": 100, "y": 100, "width": 150, "height": 150 }
```

**레이블이 있는 도형(컨테이너 바인딩)** -- 도형에 바인딩된 텍스트 요소를 만듭니다.

> **경고:** 도형에 `"label": { "text": "..." }`를 사용하지 마세요. 이는 유효한 Excalidraw 속성이 아니며 조용히 무시되어 빈 도형이 생성됩니다. 아래의 컨테이너 바인딩 방식을 반드시 사용해야 합니다.

도형에는 텍스트를 나열하는 `boundElements`가 필요하고, 텍스트에는 도형을 가리키는 `containerId`가 필요합니다.
```json
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 80,
  "roundness": { "type": 3 }, "backgroundColor": "#a5d8ff", "fillStyle": "solid",
  "boundElements": [{ "id": "t_r1", "type": "text" }] },
{ "type": "text", "id": "t_r1", "x": 105, "y": 110, "width": 190, "height": 25,
  "text": "Hello", "fontSize": 20, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "r1", "originalText": "Hello", "autoResize": true }
```
- 사각형, 타원, 마름모에서 작동
- `containerId`가 설정되면 Excalidraw가 텍스트를 자동으로 가운데 정렬
- 텍스트의 `x`/`y`/`width`/`height`는 대략적인 값이며 Excalidraw가 로드할 때 다시 계산
- `originalText`는 `text`와 일치해야 함
- 항상 `fontFamily: 1`(Virgil/손글씨 글꼴)을 포함

**레이블이 있는 화살표** -- 동일한 컨테이너 바인딩 방식을 사용합니다.
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow",
  "boundElements": [{ "id": "t_a1", "type": "text" }] },
{ "type": "text", "id": "t_a1", "x": 370, "y": 130, "width": 60, "height": 20,
  "text": "connects", "fontSize": 16, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "a1", "originalText": "connects", "autoResize": true }
```

**독립 텍스트** (제목과 주석만 -- 컨테이너 없음):
```json
{ "type": "text", "id": "t1", "x": 150, "y": 138, "text": "Hello", "fontSize": 20,
  "fontFamily": 1, "strokeColor": "#1e1e1e", "originalText": "Hello", "autoResize": true }
```
- `x`는 왼쪽 가장자리입니다. 위치를 `cx`를 중심으로 맞추려면 `x = cx - (text.length * fontSize * 0.5) / 2`
- 위치 지정에 `textAlign` 또는 `width`를 의존하지 마세요.

**화살표**:
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow" }
```
- `points`: 요소의 `x`, `y`를 기준으로 한 `[dx, dy]` 오프셋
- `endArrowhead`: `null` | `"arrow"` | `"bar"` | `"dot"` | `"triangle"`
- `strokeStyle`: `"solid"` (기본값) | `"dashed"` | `"dotted"`

### 화살표 바인딩(화살표를 도형에 연결)

```json
{
  "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 150, "height": 0,
  "points": [[0,0],[150,0]], "endArrowhead": "arrow",
  "startBinding": { "elementId": "r1", "fixedPoint": [1, 0.5] },
  "endBinding": { "elementId": "r2", "fixedPoint": [0, 0.5] }
}
```

`fixedPoint` 좌표: `top=[0.5,0]`, `bottom=[0.5,1]`, `left=[0,0.5]`, `right=[1,0.5]`

### 그리기 순서(z-order)
- 배열 순서 = z-order (처음 요소가 뒤)
- 순차적으로 출력: 배경 영역 → 도형 → 바인딩된 텍스트 → 화살표 → 화살표 레이블 텍스트 → 다음 도형
- 나쁜 예: 모든 사각형, 그다음 모든 텍스트, 그다음 모든 화살표
- 좋은 예: bg_zone → shape1 → text_for_shape1 → arrow1 → arrow_label_text → shape2 → text_for_shape2 → ...
- 바인딩된 텍스트 요소는 항상 컨테이너 도형 바로 뒤에 배치

### 크기 지정 지침

**글꼴 크기:**
- 본문 및 레이블, 설명의 `fontSize`는 **최소 16**
- 제목과 헤딩의 `fontSize`는 **최소 20**
- 보조 주석만 `fontSize` **14** 사용(꼭 필요한 경우에 한함)
- `fontSize`를 14보다 작게 사용하지 마세요.

**요소 크기:**
- 레이블이 있는 사각형/타원의 최소 도형 크기: 120x60
- 요소 사이에 최소 20~30px의 간격 유지
- 작은 요소를 많이 만들기보다 더 적고 큰 요소를 선호

### 색상 팔레트

전체 색상 표는 `references/colors.md`를 참고하세요. 빠른 참고:

| 용도 | 채우기 색상 | Hex |
|-----|-----------|-----|
| 기본 / 입력 | 연한 파랑 | `#a5d8ff` |
| 성공 / 출력 | 연한 초록 | `#b2f2bb` |
| 경고 / 외부 | 연한 주황 | `#ffd8a8` |
| 처리 / 특수 | 연한 보라 | `#d0bfff` |
| 오류 / 중요 | 연한 빨강 | `#ffc9c9` |
| 메모 / 결정 | 연한 노랑 | `#fff3bf` |
| 저장소 / 데이터 | 연한 청록 | `#c3fae8` |

### 팁
- 다이어그램 전체에서 색상 팔레트를 일관되게 사용하세요.
- **텍스트 대비는 매우 중요합니다** -- 흰색 배경에 연한 회색을 사용하지 마세요. 흰색에서 텍스트 색상의 최소값: `#757575`
- 텍스트에 이모지를 사용하지 마세요 -- Excalidraw 글꼴에서 렌더링되지 않습니다.
- 어두운 모드 다이어그램은 `references/dark-mode.md`를 참고하세요.
- 더 큰 예시는 `references/examples.md`를 참고하세요.
