---
title: "Excalidraw — 손으로 그린 듯한 Excalidraw JSON 다이어그램 (아키텍처, 플로우차트, 시퀀스)"
sidebar_label: "Excalidraw"
description: "손으로 그린 듯한 Excalidraw JSON 다이어그램 (아키텍처, 플로우차트, 시퀀스)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Excalidraw

손으로 그린 듯한 Excalidraw JSON 다이어그램 (아키텍처, 플로우차트, 시퀀스).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/excalidraw` |
| 버전 | `1.0.0` |
| 저자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Excalidraw`, `Diagrams`, `Flowcharts`, `Architecture`, `Visualization`, `JSON` |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# Excalidraw Diagram Skill

표준 Excalidraw 요소 JSON을 작성하고 `.excalidraw` 파일로 저장하여 다이어그램을 만듭니다. 이 파일들은 [excalidraw.com](https://excalidraw.com)에 드래그 앤 드롭하여 확인하고 편집할 수 있습니다. 계정, API 키, 렌더링 라이브러리가 전혀 필요 없이 오직 JSON만 사용합니다.

## 이 스킬을 사용하는 시기

아키텍처 다이어그램, 플로우차트, 시퀀스 다이어그램, 개념도 등을 위한 `.excalidraw` 파일을 생성할 때 사용하세요. 파일은 excalidraw.com에서 열거나 업로드하여 공유 가능한 링크를 만들 수 있습니다.

## 워크플로우

1. **이 스킬 로드** (이미 완료함)
2. **요소 JSON 작성** -- Excalidraw 요소 객체들의 배열
3. **파일 저장** -- `write_file`을 사용하여 `.excalidraw` 파일 생성
4. **선택적으로 업로드** -- `terminal`을 통해 `scripts/upload.py`를 실행하여 공유 가능한 링크 생성

### 다이어그램 저장하기

요소 배열을 표준 `.excalidraw` 껍데기(envelope) 안에 넣고 `write_file`로 저장하세요:

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "hermes-agent",
  "elements": [ ...여기에 요소 배열 삽입... ],
  "appState": {
    "viewBackgroundColor": "#ffffff"
  }
}
```

예를 들어 `~/diagrams/my_diagram.excalidraw`와 같은 원하는 경로에 저장하세요.

### 공유 가능한 링크를 위해 업로드하기

터미널을 통해 이 스킬의 `scripts/` 디렉토리에 있는 업로드 스크립트를 실행하세요:

```bash
python skills/diagramming/excalidraw/scripts/upload.py ~/diagrams/my_diagram.excalidraw
```

이 명령은 excalidraw.com(계정 불필요)에 업로드하고 공유 가능한 URL을 출력합니다. `cryptography` pip 패키지(`pip install cryptography`)가 필요합니다.

---

## 요소 형식 참조

### 필수 필드 (모든 요소)
`type`, `id` (고유 문자열), `x`, `y`, `width`, `height`

### 기본값 (자동으로 적용되므로 생략 가능)
- `strokeColor`: `"#1e1e1e"`
- `backgroundColor`: `"transparent"`
- `fillStyle`: `"solid"`
- `strokeWidth`: `2`
- `roughness`: `1` (손으로 그린 듯한 느낌)
- `opacity`: `100`

캔버스 배경은 흰색입니다.

### 요소 유형 (Element Types)

**직사각형 (Rectangle)**:
```json
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 100 }
```
- 둥근 모서리의 경우: `roundness: { "type": 3 }`
- 채우기의 경우: `backgroundColor: "#a5d8ff"`, `fillStyle: "solid"`

**타원 (Ellipse)**:
```json
{ "type": "ellipse", "id": "e1", "x": 100, "y": 100, "width": 150, "height": 150 }
```

**마름모 (Diamond)**:
```json
{ "type": "diamond", "id": "d1", "x": 100, "y": 100, "width": 150, "height": 150 }
```

**라벨이 있는 도형 (컨테이너 바인딩)** -- 도형에 바인딩된 텍스트 요소를 만듭니다:

> **경고:** 도형 요소 내부에 `"label": { "text": "..." }`를 사용하지 **마십시오**. 이는 유효한 Excalidraw 속성이 아니며 조용히 무시되어 빈 도형이 생성됩니다. 반드시 아래의 컨테이너 바인딩 방식을 사용해야 합니다.

도형에는 텍스트를 나열하는 `boundElements`가 필요하고, 텍스트에는 다시 도형을 가리키는 `containerId`가 필요합니다:
```json
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 80,
  "roundness": { "type": 3 }, "backgroundColor": "#a5d8ff", "fillStyle": "solid",
  "boundElements": [{ "id": "t_r1", "type": "text" }] },
{ "type": "text", "id": "t_r1", "x": 105, "y": 110, "width": 190, "height": 25,
  "text": "Hello", "fontSize": 20, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "r1", "originalText": "Hello", "autoResize": true }
```
- 직사각형, 타원, 마름모에 적용됩니다.
- `containerId`가 설정되면 Excalidraw에 의해 텍스트가 자동으로 중앙 정렬됩니다.
- 텍스트의 `x`/`y`/`width`/`height`는 대략적인 값이며, Excalidraw가 로드 시 다시 계산합니다.
- `originalText`는 `text`와 일치해야 합니다.
- 항상 `fontFamily: 1` (Virgil/손글씨 폰트)을 포함하세요.

**라벨이 있는 화살표 (Labeled arrow)** -- 동일한 컨테이너 바인딩 방식을 사용합니다:
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow",
  "boundElements": [{ "id": "t_a1", "type": "text" }] },
{ "type": "text", "id": "t_a1", "x": 370, "y": 130, "width": 60, "height": 20,
  "text": "connects", "fontSize": 16, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "a1", "originalText": "connects", "autoResize": true }
```

**독립 텍스트** (제목 및 주석용으로만 사용 -- 컨테이너 없음):
```json
{ "type": "text", "id": "t1", "x": 150, "y": 138, "text": "Hello", "fontSize": 20,
  "fontFamily": 1, "strokeColor": "#1e1e1e", "originalText": "Hello", "autoResize": true }
```
- `x`는 왼쪽 가장자리입니다. 위치 `cx`에 중앙 정렬하려면: `x = cx - (text.length * fontSize * 0.5) / 2`
- 위치를 지정할 때 `textAlign`이나 `width`에 의존하지 **마십시오**.

**화살표 (Arrow)**:
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow" }
```
- `points`: 기준 `x`, `y`로부터의 오프셋 `[dx, dy]`
- `endArrowhead`: `null` | `"arrow"` | `"bar"` | `"dot"` | `"triangle"`
- `strokeStyle`: `"solid"` (기본값) | `"dashed"` | `"dotted"`

### 화살표 바인딩 (도형에 화살표 연결하기)

```json
{
  "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 150, "height": 0,
  "points": [[0,0],[150,0]], "endArrowhead": "arrow",
  "startBinding": { "elementId": "r1", "fixedPoint": [1, 0.5] },
  "endBinding": { "elementId": "r2", "fixedPoint": [0, 0.5] }
}
```

`fixedPoint` 좌표: 위쪽(`top`)=`[0.5,0]`, 아래쪽(`bottom`)=`[0.5,1]`, 왼쪽(`left`)=`[0,0.5]`, 오른쪽(`right`)=`[1,0.5]`

### 그리기 순서 (Z-순서)
- 배열 순서 = Z-순서 (첫 번째 = 뒤, 마지막 = 앞)
- 점진적으로 출력: 배경 영역 → 도형 → 도형에 바인딩된 텍스트 → 도형의 화살표 → 다음 도형
- 나쁜 예(BAD): 모든 직사각형, 그다음 모든 텍스트, 그다음 모든 화살표
- 좋은 예(GOOD): bg_zone → shape1 → text_for_shape1 → arrow1 → arrow_label_text → shape2 → text_for_shape2 → ...
- 바인딩된 텍스트 요소는 항상 그 컨테이너 도형 바로 뒤에 배치하세요.

### 크기 지정 지침

**폰트 크기:**
- 최소 `fontSize`: 본문 텍스트, 라벨, 설명의 경우 **16**
- 최소 `fontSize`: 제목 및 머리글의 경우 **20**
- 최소 `fontSize`: 보조 주석의 경우에만 보수적으로 **14** 사용
- 14 미만의 `fontSize`는 절대 사용하지 마세요.

**요소 크기:**
- 최소 도형 크기: 라벨이 있는 직사각형/타원의 경우 120x60
- 요소 간 최소 20-30px의 간격을 두세요.
- 작고 많은 요소보다는 크고 적은 요소를 선호하세요.

### 색상 팔레트

전체 색상표는 `references/colors.md`를 참조하세요. 빠른 참조:

| 용도 | 채우기 색상 | Hex |
|-----|-----------|-----|
| Primary / Input | Light Blue | `#a5d8ff` |
| Success / Output | Light Green | `#b2f2bb` |
| Warning / External | Light Orange | `#ffd8a8` |
| Processing / Special | Light Purple | `#d0bfff` |
| Error / Critical | Light Red | `#ffc9c9` |
| Notes / Decisions | Light Yellow | `#fff3bf` |
| Storage / Data | Light Teal | `#c3fae8` |

### 팁
- 다이어그램 전체에 걸쳐 색상 팔레트를 일관되게 사용하세요.
- **텍스트 대비는 매우 중요합니다** -- 흰색 배경에 밝은 회색을 사용하지 마세요. 흰색 배경의 최소 텍스트 색상: `#757575`
- 텍스트에 이모지를 사용하지 마세요 -- Excalidraw의 폰트에서는 렌더링되지 않습니다.
- 다크 모드 다이어그램의 경우 `references/dark-mode.md`를 참조하세요.
- 더 큰 예제는 `references/examples.md`를 참조하세요.
