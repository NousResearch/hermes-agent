---
title: "Concept Diagrams"
sidebar_label: "Concept Diagrams"
description: "9개의 의미론적 색상 램프, 문장 첫 글자만 대문자로 쓰는 타이포그래피, 자동 다크 모드를 갖춘 통합 교육용 시각적 언어를 사용하여 단일 HTML 파일로 평면적이고 미니멀한 라이트/다크 인식 SVG 다이어그램을 생성합니다."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Concept Diagrams

9개의 의미론적 색상 램프, 문장 첫 글자만 대문자로 쓰는 타이포그래피, 자동 다크 모드를 갖춘 통합 교육용 시각적 언어를 사용하여 평면적이고 미니멀한 라이트/다크 인식 SVG 다이어그램을 단일 HTML 파일로 생성합니다. 교육용 및 비소프트웨어 시각 자료에 가장 적합합니다 — 물리 실험 설정, 화학 메커니즘, 수학 곡선, 물리적 물체(항공기, 터빈, 스마트폰, 기계식 시계), 해부도, 평면도, 단면도, 서사적 여정(X의 라이프사이클, Y의 과정), 허브 앤 스포크(hub-spoke) 시스템 통합(스마트 시티, IoT), 전개도(exploded layer views). 주제에 대해 더 특화된 스킬이 존재한다면(전용 소프트웨어/클라우드 아키텍처, 손그림 스케치, 애니메이션 설명 영상 등), 그것을 우선적으로 사용하세요 — 그렇지 않다면 이 스킬은 깔끔한 교육용 외관을 갖춘 범용 SVG 다이어그램 폴백(fallback)으로도 사용될 수 있습니다. 15개의 예제 다이어그램이 함께 제공됩니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/creative/concept-diagrams`로 설치 |
| Path | `optional-skills/creative/concept-diagrams` |
| Version | `0.1.0` |
| Author | v1k22 (original PR), ported into hermes-agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `diagrams`, `svg`, `visualization`, `education`, `physics`, `chemistry`, `engineering` |
| Related skills | [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), `generative-widgets` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Concept Diagrams

통합된 평면적이고 미니멀한 디자인 시스템으로 프로덕션 품질의 SVG 다이어그램을 생성합니다. 출력물은 모든 최신 브라우저에서 동일하게 렌더링되는 단일 독립형 HTML 파일이며, 라이트/다크 모드가 자동으로 적용됩니다.

## 범위 (Scope)

**가장 적합한 용도:**
- 물리 실험 설정, 화학 메커니즘, 수학 곡선, 생물학
- 물리적 물체 (항공기, 터빈, 스마트폰, 기계식 시계, 세포)
- 해부도, 단면도, 전개도 (exploded layer views)
- 평면도, 건축 구조 변경
- 서사적 여정 (X의 라이프사이클, Y의 과정)
- 허브 앤 스포크(hub-spoke) 시스템 통합 (스마트 시티, IoT 네트워크, 전력망)
- 모든 분야의 교육용 / 교과서 스타일 시각 자료
- 정량적 차트 (그룹화된 막대, 에너지 프로파일)

**다음과 같은 경우에는 다른 스킬을 먼저 찾으세요:**
- 어두운 기술 미학을 갖춘 전용 소프트웨어 / 클라우드 인프라 아키텍처 (사용 가능한 경우 `architecture-diagram` 고려)
- 손그림 화이트보드 스케치 (사용 가능한 경우 `excalidraw` 고려)
- 애니메이션 설명 영상 또는 비디오 출력 (애니메이션 스킬 고려)

주제에 대해 더 특화된 스킬이 있다면 그것을 선호하세요. 적합한 것이 없다면 이 스킬이 범용 SVG 다이어그램 폴백 역할을 할 수 있습니다 — 출력물은 아래에 설명된 깔끔한 교육용 미학을 따르며, 이는 거의 모든 주제에 대한 합리적인 기본값이 됩니다.

## 워크플로우

1. 다이어그램 유형을 결정합니다 (아래 다이어그램 유형 참조).
2. 디자인 시스템 규칙을 사용하여 구성 요소를 배치합니다.
3. `templates/template.html`을 래퍼(wrapper)로 사용하여 전체 HTML 페이지를 작성합니다 — 템플릿의 `<!-- PASTE SVG HERE -->` 부분에 SVG를 붙여넣으세요.
4. 독립형 `.html` 파일로 저장합니다 (예: `~/my-diagram.html` 또는 `./my-diagram.html`).
5. 사용자가 서버나 종속성 없이 브라우저에서 직접 엽니다.

선택 사항: 사용자가 여러 다이어그램의 탐색 가능한 갤러리를 원하는 경우 맨 아래의 "로컬 미리보기 서버"를 참조하세요.

HTML 템플릿 로드:
```
skill_view(name="concept-diagrams", file_path="templates/template.html")
```

템플릿에는 전체 CSS 디자인 시스템(`c-*` 색상 클래스, 텍스트 클래스, 라이트/다크 변수, 화살표 마커 스타일)이 내장되어 있습니다. 생성하는 SVG는 이 클래스들이 호스팅 페이지에 존재한다는 사실에 의존합니다.

---

## 디자인 시스템

### 철학

- **평면적 (Flat)**: 그라데이션, 그림자, 흐림 효과, 발광 또는 네온 효과가 없습니다.
- **미니멀 (Minimal)**: 필수적인 것만 보여줍니다. 박스 안에 장식용 아이콘을 넣지 않습니다.
- **일관성 (Consistent)**: 모든 다이어그램에 동일한 색상, 간격, 타이포그래피 및 선 두께를 사용합니다.
- **다크 모드 지원 (Dark-mode ready)**: 모든 색상은 CSS 클래스를 통해 자동 적응합니다 — 모드별 SVG를 만들 필요가 없습니다.

### 색상 팔레트

각각 7개의 단계가 있는 9개의 색상 램프. `<g>` 또는 도형 요소에 클래스 이름을 넣으세요. 템플릿 CSS가 두 모드를 모두 처리합니다.

| 클래스      | 50 (가장 밝음) | 100     | 200     | 400     | 600     | 800     | 900 (가장 어두움) |
|------------|---------------|---------|---------|---------|---------|---------|---------------|
| `c-purple` | #EEEDFE | #CECBF6 | #AFA9EC | #7F77DD | #534AB7 | #3C3489 | #26215C |
| `c-teal`   | #E1F5EE | #9FE1CB | #5DCAA5 | #1D9E75 | #0F6E56 | #085041 | #04342C |
| `c-coral`  | #FAECE7 | #F5C4B3 | #F0997B | #D85A30 | #993C1D | #712B13 | #4A1B0C |
| `c-pink`   | #FBEAF0 | #F4C0D1 | #ED93B1 | #D4537E | #993556 | #72243E | #4B1528 |
| `c-gray`   | #F1EFE8 | #D3D1C7 | #B4B2A9 | #888780 | #5F5E5A | #444441 | #2C2C2A |
| `c-blue`   | #E6F1FB | #B5D4F4 | #85B7EB | #378ADD | #185FA5 | #0C447C | #042C53 |
| `c-green`  | #EAF3DE | #C0DD97 | #97C459 | #639922 | #3B6D11 | #27500A | #173404 |
| `c-amber`  | #FAEEDA | #FAC775 | #EF9F27 | #BA7517 | #854F0B | #633806 | #412402 |
| `c-red`    | #FCEBEB | #F7C1C1 | #F09595 | #E24B4A | #A32D2D | #791F1F | #501313 |

#### 색상 지정 규칙

색상은 순서가 아니라 **의미**를 인코딩합니다. 무지개처럼 색상을 순환하지 마세요.

- 노드를 **카테고리**별로 그룹화합니다 — 동일한 유형의 모든 노드는 하나의 색상을 공유합니다.
- 중립/구조적 노드(시작, 끝, 일반 단계, 사용자)에는 `c-gray`를 사용합니다.
- **다이어그램당 2-3개의 색상**을 사용하고 6개 이상은 사용하지 마세요.
- 일반적인 카테고리에는 `c-purple`, `c-teal`, `c-coral`, `c-pink`를 선호합니다.
- 의미론적 의미(정보, 성공, 경고, 오류)를 위해 `c-blue`, `c-green`, `c-amber`, `c-red`를 예약하세요.

라이트/다크 단계 매핑 (템플릿 CSS에 의해 처리됨 — 클래스만 사용하세요):
- 라이트 모드: 50 채우기 + 600 선 + 800 제목 / 600 부제목
- 다크 모드: 800 채우기 + 200 선 + 100 제목 / 200 부제목

### 타이포그래피

오직 두 가지 글꼴 크기만 사용합니다. 예외는 없습니다.

| 클래스 | 크기 | 굵기 | 용도 |
|-------|------|--------|-----|
| `th`  | 14px | 500    | 노드 제목, 영역 레이블 |
| `ts`  | 12px | 400    | 부제목, 설명, 화살표 레이블 |
| `t`   | 14px | 400    | 일반 텍스트 |

- **항상 문장의 첫 글자만 대문자로 (Sentence case).** 제목 대문자(Title Case)나 모두 대문자(ALL CAPS)를 절대 사용하지 마세요.
- 모든 `<text>`는 반드시 클래스(`t`, `ts`, 또는 `th`)를 가져야 합니다. 클래스 없는 텍스트는 허용되지 않습니다.
- 박스 안의 모든 텍스트에 `dominant-baseline="central"`을 사용하세요.
- 박스 안에서 텍스트를 중앙에 정렬하려면 `text-anchor="middle"`을 사용하세요.

**너비 예측 (대략):**
- 14px 굵기 500: 문자당 약 8px
- 12px 굵기 400: 문자당 약 6.5px
- 항상 확인하세요: `box_width >= (char_count × px_per_char) + 48` (양쪽에 24px 패딩)

### 간격 및 레이아웃

- **ViewBox**: `viewBox="0 0 680 H"` (H = 콘텐츠 높이 + 40px 버퍼).
- **안전 영역 (Safe area)**: x=40에서 x=640, y=40에서 y=(H-40).
- **박스 사이**: 최소 60px 간격.
- **박스 내부**: 24px 가로 패딩, 12px 세로 패딩.
- **화살촉 간격**: 화살촉과 박스 가장자리 사이 10px.
- **단일 줄 박스**: 44px 높이.
- **두 줄 박스**: 56px 높이, 제목과 부제목 기준선 사이 18px.
- **컨테이너 패딩**: 모든 컨테이너 내부에 최소 20px.
- **최대 중첩**: 2-3단계 깊이. 더 깊어지면 680px 너비에서 읽을 수 없습니다.

### 선 & 모양 (Stroke & Shape)

- **선 두께**: 모든 노드 테두리에 0.5px. 1px도, 2px도 안 됩니다.
- **사각형 둥글기**: 노드에 `rx="8"`, 내부 컨테이너에 `rx="12"`, 외부 컨테이너에 `rx="16"`에서 `rx="20"`.
- **연결선 경로**: 반드시 `fill="none"`이어야 합니다. 그렇지 않으면 SVG는 기본값인 `fill: black`을 적용합니다.

### 화살표 마커 (Arrow Marker)

**모든** SVG의 시작 부분에 이 `<defs>` 블록을 포함하세요:

```xml
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
```

선에 `marker-end="url(#arrow)"`를 사용하세요. 화살촉은 `context-stroke`를 통해 선 색상을 상속합니다.

### CSS 클래스 (템플릿에서 제공)

템플릿 페이지는 다음을 제공합니다:

- 텍스트: `.t`, `.ts`, `.th`
- 중립: `.box`, `.arr`, `.leader`, `.node`
- 색상 램프: `.c-purple`, `.c-teal`, `.c-coral`, `.c-pink`, `.c-gray`, `.c-blue`, `.c-green`, `.c-amber`, `.c-red` (모두 라이트/다크 모드 자동 적용)

이것들을 재정의할 필요는 **없습니다** — SVG에 적용하기만 하면 됩니다. 템플릿 파일에는 전체 CSS 정의가 포함되어 있습니다.

---

## SVG 보일러플레이트 (Boilerplate)

템플릿 페이지 내부의 모든 SVG는 정확히 이 구조로 시작합니다:

```xml
<svg width="100%" viewBox="0 0 680 {HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- 다이어그램 내용이 여기에 들어갑니다 -->

</svg>
```

`{HEIGHT}`를 실제 계산된 높이(마지막 요소 바닥 + 40px)로 바꿉니다.

### 노드 패턴

**단일 줄 노드 (44px):**
```xml
<g class="node c-blue">
  <rect x="100" y="20" width="180" height="44" rx="8" stroke-width="0.5"/>
  <text class="th" x="190" y="42" text-anchor="middle" dominant-baseline="central">Service name</text>
</g>
```

**두 줄 노드 (56px):**
```xml
<g class="node c-teal">
  <rect x="100" y="20" width="200" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="200" y="38" text-anchor="middle" dominant-baseline="central">Service name</text>
  <text class="ts" x="200" y="56" text-anchor="middle" dominant-baseline="central">Short description</text>
</g>
```

**커넥터 (레이블 없음):**
```xml
<line x1="200" y1="76" x2="200" y2="120" class="arr" marker-end="url(#arrow)"/>
```

**컨테이너 (점선 또는 실선):**
```xml
<g class="c-purple">
  <rect x="40" y="92" width="600" height="300" rx="16" stroke-width="0.5"/>
  <text class="th" x="66" y="116">Container label</text>
  <text class="ts" x="66" y="134">Subtitle info</text>
</g>
```

---

## 다이어그램 유형

주제에 맞는 레이아웃을 선택하세요:

1. **순서도 (Flowchart)** — CI/CD 파이프라인, 요청 라이프사이클, 승인 워크플로우, 데이터 처리. 단방향 흐름(하향식 또는 좌우). 한 행에 최대 4-5개 노드.
2. **구조 / 포함 관계 (Structural / Containment)** — 클라우드 인프라 중첩, 계층이 있는 시스템 아키텍처. 내부에 영역이 있는 큰 외부 컨테이너. 논리적 그룹화를 위한 점선 사각형.
3. **API / 엔드포인트 맵 (API / Endpoint Map)** — REST 라우트, GraphQL 스키마. 루트에서 트리 구조로 뻗어나가며, 리소스 그룹으로 분기되고 각 그룹에 엔드포인트 노드가 포함됨.
4. **마이크로서비스 토폴로지 (Microservice Topology)** — 서비스 메시, 이벤트 기반 시스템. 노드로서의 서비스, 통신 패턴을 위한 화살표, 서비스 사이의 메시지 큐.
5. **데이터 흐름 (Data Flow)** — ETL 파이프라인, 스트리밍 아키텍처. 소스에서 처리를 거쳐 싱크(sink)로 향하는 왼쪽에서 오른쪽 흐름.
6. **물리적 / 구조적 (Physical / Structural)** — 차량, 건물, 하드웨어, 해부도. 물리적 형태와 일치하는 모양을 사용 — 구부러진 몸체에는 `<path>`, 테이퍼 모양에는 `<polygon>`, 원통형 부품에는 `<ellipse>`/`<circle>`, 구획에는 중첩된 `<rect>`. `references/physical-shape-cookbook.md` 참조.
7. **인프라 / 시스템 통합 (Infrastructure / Systems Integration)** — 스마트 시티, IoT 네트워크, 다중 도메인 시스템. 서브시스템을 연결하는 중앙 플랫폼이 있는 허브 앤 스포크(Hub-spoke) 레이아웃. 시스템별 의미론적 선 스타일(`.data-line`, `.power-line`, `.water-pipe`, `.road`). `references/infrastructure-patterns.md` 참조.
8. **UI / 대시보드 목업 (UI / Dashboard Mockups)** — 관리자 패널, 모니터링 대시보드. 중첩된 차트/게이지/인디케이터 요소가 있는 화면 프레임. `references/dashboard-patterns.md` 참조.

물리적, 인프라 및 대시보드 다이어그램의 경우 생성하기 전에 일치하는 참조 파일을 로드하세요 — 각 파일은 준비된 CSS 클래스와 도형 기본 요소(primitives)를 제공합니다.

---

## 검증 체크리스트

SVG를 마무리하기 전에 다음을 모두 확인하세요:

1. 모든 `<text>`에 `t`, `ts`, 또는 `th` 클래스가 있습니다.
2. 박스 안의 모든 `<text>`에 `dominant-baseline="central"`이 있습니다.
3. 화살표로 사용되는 모든 커넥터 `<path>` 또는 `<line>`에 `fill="none"`이 있습니다.
4. 어떤 화살표 선도 관련 없는 박스를 가로지르지 않습니다.
5. 14px 텍스트의 경우 `box_width >= (longest_label_chars × 8) + 48`입니다.
6. 12px 텍스트의 경우 `box_width >= (longest_label_chars × 6.5) + 48`입니다.
7. ViewBox 높이 = 가장 아래에 있는 요소 + 40px입니다.
8. 모든 내용이 x=40에서 x=640 이내에 머뭅니다.
9. 색상 클래스(`c-*`)는 `<g>` 또는 도형 요소에 있으며, `<path>` 커넥터에는 절대 적용하지 않습니다.
10. 화살표 `<defs>` 블록이 존재합니다.
11. 그라데이션, 그림자, 흐림 효과 또는 발광 효과가 없습니다.
12. 선 두께는 모든 노드 테두리에서 0.5px입니다.

---

## 출력 및 미리보기

### 기본: 독립형 HTML 파일

사용자가 직접 열 수 있는 단일 `.html` 파일을 작성합니다. 서버나 종속성이 없으며 오프라인에서 작동합니다. 패턴:

```python
# 1. 템플릿 로드
template = skill_view("concept-diagrams", "templates/template.html")

# 2. 제목, 부제목을 채우고 SVG를 붙여넣기
html = template.replace(
    "<!-- DIAGRAM TITLE HERE -->", "SN2 reaction mechanism"
).replace(
    "<!-- OPTIONAL SUBTITLE HERE -->", "Bimolecular nucleophilic substitution"
).replace(
    "<!-- PASTE SVG HERE -->", svg_content
)

# 3. 사용자가 선택한 경로(또는 기본적으로 ./)에 작성
write_file("./sn2-mechanism.html", html)
```

사용자에게 여는 방법을 알려주세요:

```
# macOS
open ./sn2-mechanism.html
# Linux
xdg-open ./sn2-mechanism.html
```

### 선택 사항: 로컬 미리보기 서버 (다중 다이어그램 갤러리)

사용자가 여러 다이어그램의 탐색 가능한 갤러리를 명시적으로 원할 때만 이것을 사용하세요.

**규칙:**
- `127.0.0.1`에만 바인딩하세요. 절대 `0.0.0.0`에 바인딩하지 마세요. 모든 네트워크 인터페이스에 다이어그램을 노출하는 것은 공유 네트워크에서 보안 위험 요소입니다.
- 빈 포트를 선택하고(하드코딩하지 마세요) 선택한 URL을 사용자에게 알려주세요.
- 서버는 선택 사항이며 옵트인 방식입니다 — 독립형 HTML 파일을 먼저 선호하세요.

권장 패턴 (OS가 임시 빈 포트를 선택하도록 함):

```bash
# 각 다이어그램을 .diagrams/ 아래의 자체 폴더에 넣습니다
mkdir -p .diagrams/sn2-mechanism
# ... .diagrams/sn2-mechanism/index.html 작성 ...

# 루프백에서만 빈 포트로 제공
cd .diagrams && python3 -c "
import http.server, socketserver
with socketserver.TCPServer(('127.0.0.1', 0), http.server.SimpleHTTPRequestHandler) as s:
    print(f'Serving at http://127.0.0.1:{s.server_address[1]}/')
    s.serve_forever()
" &
```

사용자가 고정 포트를 고집하는 경우 `127.0.0.1:<port>`를 사용하세요 — 여전히 `0.0.0.0`은 안 됩니다. 서버를 중지하는 방법을 문서화하세요 (`kill %1` 또는 `pkill -f "http.server"`).

---

## 예제 참조

`examples/` 디렉토리는 테스트를 거친 15개의 완전한 다이어그램을 제공합니다. 비슷한 유형의 새 다이어그램을 작성하기 전에 작동 패턴을 찾아보세요:

| 파일 | 유형 | 설명 |
|------|------|--------------|
| `hospital-emergency-department-flow.md` | 순서도 | 의미론적 색상을 통한 우선순위 라우팅 |
| `feature-film-production-pipeline.md` | 순서도 | 단계별 워크플로우, 수평적 하위 흐름 |
| `automated-password-reset-flow.md` | 순서도 | 오류 분기가 있는 인증 흐름 |
| `autonomous-llm-research-agent-flow.md` | 순서도 | 루프백 화살표, 의사결정 분기 |
| `place-order-uml-sequence.md` | 시퀀스 | UML 시퀀스 다이어그램 스타일 |
| `commercial-aircraft-structure.md` | 물리적 | 현실적인 모양을 위한 Paths, polygons, ellipses |
| `wind-turbine-structure.md` | 물리적 단면도 | 지하/지상 분리, 색상 코딩 |
| `smartphone-layer-anatomy.md` | 전개도(Exploded view) | 번갈아 나타나는 좌/우 레이블, 레이어된 구성요소 |
| `apartment-floor-plan-conversion.md` | 평면도 | 벽, 문, 제안된 변경 사항을 점선 빨간색으로 표시 |
| `banana-journey-tree-to-smoothie.md` | 서사적 여정 | 구불구불한 길, 점진적 상태 변화 |
| `cpu-ooo-microarchitecture.md` | 하드웨어 파이프라인 | 팽창(Fan-out), 메모리 계층 사이드바 |
| `sn2-reaction-mechanism.md` | 화학 | 분자, 구부러진 화살표, 에너지 프로파일 |
| `smart-city-infrastructure.md` | 허브 앤 스포크 | 시스템별 의미론적 선 스타일 |
| `electricity-grid-flow.md` | 다단계 흐름 | 전압 계층구조, 흐름 마커 |
| `ml-benchmark-grouped-bar-chart.md` | 차트 | 그룹화된 막대, 이중 축 |

다음을 사용하여 예제를 로드하세요:
```
skill_view(name="concept-diagrams", file_path="examples/<filename>")
```

---

## 빠른 참조: 언제 무엇을 사용할 것인가

| 사용자 요청 | 다이어그램 유형 | 권장 색상 |
|-----------|--------------|------------------|
| "파이프라인을 보여줘" | 순서도 (Flowchart) | gray 시작/끝, purple 단계, red 오류, teal 배포 |
| "데이터 흐름을 그려줘" | 데이터 파이프라인 (좌-우) | gray 소스, purple 처리, teal 싱크(sinks) |
| "시스템을 시각화해줘" | 구조적 (포함 관계) | purple 컨테이너, teal 서비스, coral 데이터 |
| "엔드포인트를 매핑해줘" | API 트리 | purple 루트, 리소스 그룹당 하나의 램프 |
| "서비스들을 보여줘" | 마이크로서비스 토폴로지 | gray 수신, teal 서비스, purple 버스, coral 워커 |
| "항공기/차량을 그려줘" | 물리적 (Physical) | 현실적 형태를 위한 paths, polygons, ellipses |
| "스마트 시티 / IoT" | 허브 앤 스포크 통합 | 서브시스템당 의미론적 선 스타일 |
| "대시보드를 보여줘" | UI 목업 | 어두운 화면, 차트 색상: 알림에 teal, purple, coral |
| "전력망 / 전기" | 다단계 흐름 | 전압 계층구조 (HV/MV/LV 선 굵기) |
| "풍력 터빈 / 터빈" | 물리적 단면도 | 기초 + 타워 단면 + 색상 코딩된 나셀 |
| "X의 여정 / 라이프사이클" | 서사적 여정 | 구불구불한 길, 점진적 상태 변화 |
| "X의 층 / 전개도" | 전개도(Exploded layer view) | 수직 스택, 번갈아 나타나는 레이블 |
| "CPU / 파이프라인" | 하드웨어 파이프라인 | 수직 단계, 실행 포트로 팽창(fan-out) |
| "평면도 / 아파트" | 평면도 | 벽, 문, 점선 빨간색의 제안된 변경 사항 |
| "반응 메커니즘" | 화학 | 원자, 결합, 굽은 화살표, 전이 상태, 에너지 프로파일 |
