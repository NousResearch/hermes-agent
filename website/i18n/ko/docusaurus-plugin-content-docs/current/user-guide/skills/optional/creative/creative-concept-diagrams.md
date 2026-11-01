---
title: "개념 다이어그램 — 평면적이고 미니멀한 교육용 SVG 시각 자료를 HTML로 생성"
sidebar_label: "개념 다이어그램"
description: "평면적이고 미니멀한 교육용 SVG 시각 자료를 HTML로 생성"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 개념 다이어그램

평면적이고 미니멀한 교육용 SVG 시각 자료를 HTML로 생성합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/concept-diagrams`로 설치 |
| 경로 | `optional-skills/creative/concept-diagrams` |
| 버전 | `0.1.0` |
| 작성자 | v1k22 (원본 PR), hermes-agent로 포팅 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `diagrams`, `svg`, `visualization`, `education`, `physics`, `chemistry`, `engineering` |
| 관련 스킬 | [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 불러오는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보게 되는 지침입니다.
:::

# 개념 다이어그램

통일된 평면적·미니멀 디자인 시스템으로 프로덕션 품질의 SVG 다이어그램을 생성합니다. 결과물은 최신 브라우저라면 어디서나 동일하게 렌더링되는 단일 독립형 HTML 파일이며, 라이트/다크 모드를 자동으로 지원합니다.

## 범위

**다음에 가장 적합합니다:**
- 물리 설정, 화학 반응 메커니즘, 수학 곡선, 생물학
- 물리적 객체(항공기, 터빈, 스마트폰, 기계식 시계, 세포)
- 해부학, 단면도, 분해 레이어 뷰
- 평면도, 건축 변환
- 서사적 여정(X의 생애 주기, Y의 프로세스)
- 허브-스포크 시스템 통합(스마트 시티, IoT 네트워크, 전력망)
- 모든 분야의 교육용/교과서 스타일 시각 자료
- 정량 차트(그룹 막대, 에너지 프로파일)

**먼저 다른 곳을 살펴보세요:**
- 어두운 테크 미학의 전용 소프트웨어/클라우드 인프라 아키텍처(가능하다면 `architecture-diagram` 고려)
- 손으로 그린 화이트보드 스케치(가능하다면 `excalidraw` 고려)
- 애니메이션 설명 자료 또는 동영상 출력(애니메이션 스킬 고려)

주제에 더 특화된 스킬을 사용할 수 있다면 해당 스킬을 우선하세요. 적합한 스킬이 없다면 이 스킬을 범용 SVG 다이어그램 대안으로 사용할 수 있습니다. 출력물에는 아래에 설명된 깔끔한 교육용 미학이 적용되므로 거의 모든 주제에 합리적인 기본값이 됩니다.

## 워크플로

1. 다이어그램 유형을 결정합니다(아래 다이어그램 유형 참조).
2. 디자인 시스템 규칙에 따라 구성 요소를 배치합니다.
3. `templates/template.html`을 래퍼로 사용해 전체 HTML 페이지를 작성합니다 — 템플릿에서 `<!-- PASTE SVG HERE -->`라고 표시된 곳에 SVG를 붙여 넣습니다.
4. 독립 실행형 `.html` 파일로 저장합니다(예: `~/my-diagram.html` 또는 `./my-diagram.html`).
5. 사용자가 브라우저에서 직접 엽니다 — 서버나 종속성이 필요하지 않습니다.

선택 사항: 사용자가 여러 다이어그램을 둘러볼 수 있는 갤러리를 원하면 아래쪽의 "로컬 미리보기 서버"를 참조하세요.

HTML 템플릿을 불러옵니다:
```
skill_view(name="concept-diagrams", file_path="templates/template.html")
```

템플릿에는 전체 CSS 디자인 시스템(`c-*` 색상 클래스, 텍스트 클래스, 라이트/다크 변수, 화살표 마커 스타일)이 포함되어 있습니다. 생성하는 SVG는 호스팅 페이지에 이러한 클래스가 존재한다고 가정합니다.

---

## 디자인 시스템

### 철학

- **평면적**: 그레이디언트, 드롭 섀도, 블러, 글로우 또는 네온 효과를 사용하지 않습니다.
- **미니멀**: 핵심만 보여 줍니다. 상자 안에 장식용 아이콘을 넣지 않습니다.
- **일관성**: 모든 다이어그램에서 색상, 간격, 서체, 선 두께를 동일하게 유지합니다.
- **다크 모드 지원**: 모든 색상은 CSS 클래스를 통해 자동으로 조정됩니다 — 모드별 SVG가 필요하지 않습니다.

### 색상 팔레트

각각 7개의 스톱으로 구성된 9개의 색상 램프가 있습니다. `<g>` 또는 도형 요소에 클래스 이름을 넣으면 템플릿 CSS가 두 모드를 모두 처리합니다.

| Class      | 50 (가장 밝음) | 100     | 200     | 400     | 600     | 800     | 900 (가장 어두움) |
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

#### 색상 할당 규칙

색상은 **순서가 아니라 의미**를 인코딩합니다. 무지개처럼 색상을 순환해서는 안 됩니다.

- **범주별로** 노드를 그룹화합니다 — 같은 유형의 모든 노드는 하나의 색상을 공유합니다.
- 중립/구조적 노드(시작, 종료, 일반 단계, 사용자)에는 `c-gray`를 사용합니다.
- 다이어그램당 **2~3개의 색상**을 사용하며, 6개 이상은 사용하지 않습니다.
- 일반 범주에는 `c-purple`, `c-teal`, `c-coral`, `c-pink`를 우선 사용합니다.
- `c-blue`, `c-green`, `c-amber`, `c-red`는 의미(정보, 성공, 경고, 오류)에 따라 사용하도록 남겨 둡니다.

라이트/다크 스톱 매핑(템플릿 CSS가 처리하므로 클래스를 사용하기만 하면 됩니다):
- 라이트 모드: 50 채우기 + 600 선 + 800 제목 / 600 부제목
- 다크 모드: 800 채우기 + 200 선 + 100 제목 / 200 부제목

### 타이포그래피

글꼴 크기는 두 가지만 사용합니다. 예외는 없습니다.

| 클래스 | 크기 | 두께 | 용도 |
|-------|------|--------|-----|
| `th`  | 14px | 500    | 노드 제목, 영역 레이블 |
| `ts`  | 12px | 400    | 부제목, 설명, 화살표 레이블 |
| `t`   | 14px | 400    | 일반 텍스트 |

- **문장형 대소문자**를 항상 사용합니다. 제목식 대문자나 ALL CAPS를 사용하지 않습니다.
- 모든 `<text>`에는 반드시 클래스(`t`, `ts`, 또는 `th`)가 있어야 합니다. 클래스가 없는 text는 허용하지 않습니다.
- 상자 안의 모든 텍스트에 `dominant-baseline="central"`을 사용합니다.
- 상자 안에서 가운데 정렬하는 텍스트에는 `text-anchor="middle"`을 사용합니다.

**너비 추정(대략):**
- 14px 두께 500: 문자당 약 8px
- 12px 두께 400: 문자당 약 6.5px
- 항상 확인합니다: `box_width >= (char_count × px_per_char) + 48`(양쪽에 24px 패딩)

### 간격 및 레이아웃

- **ViewBox**: `viewBox="0 0 680 H"`에서 H = 콘텐츠 높이 + 40px 버퍼.
- **안전 영역**: x=40~640, y=40~(H-40).
- **상자 사이**: 최소 간격 60px.
- **상자 내부**: 가로 패딩 24px, 세로 패딩 12px.
- **화살촉 간격**: 화살촉과 상자 가장자리 사이 10px.
- **한 줄 상자**: 높이 44px.
- **두 줄 상자**: 높이 56px, 제목과 부제목 기준선 사이 18px.
- **컨테이너 패딩**: 모든 컨테이너 내부 최소 20px.
- **최대 중첩**: 2~3단계. 680px 너비에서는 더 깊으면 읽기 어려워집니다.

### 선 및 도형

- **선 두께**: 모든 노드 테두리에 0.5px. 1px도 2px도 사용하지 않습니다.
- **사각형 모서리**: 노드는 `rx="8"`, 내부 컨테이너는 `rx="12"`, 외부 컨테이너는 `rx="16"`~`rx="20"`을 사용합니다.
- **연결 경로**: 반드시 `fill="none"`을 지정합니다. 그렇지 않으면 SVG 기본값이 `fill: black`이 됩니다.

### 화살표 마커

**모든** SVG의 시작 부분에 다음 `<defs>` 블록을 포함합니다:

```xml
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
```

선에는 `marker-end="url(#arrow)"`를 사용합니다. 화살촉은 `context-stroke`를 통해 선 색상을 상속합니다.

### CSS 클래스(템플릿 제공)

템플릿 페이지에서 다음을 제공합니다:

- 텍스트: `.t`, `.ts`, `.th`
- 중립: `.box`, `.arr`, `.leader`, `.node`
- 색상 램프: `.c-purple`, `.c-teal`, `.c-coral`, `.c-pink`, `.c-gray`, `.c-blue`, `.c-green`, `.c-amber`, `.c-red`(모두 자동 라이트/다크 모드 지원)

이들을 다시 정의할 필요는 없습니다 — SVG에 적용하기만 하면 됩니다. 템플릿 파일에 전체 CSS 정의가 들어 있습니다.

---

## SVG 보일러플레이트

템플릿 페이지 내부의 모든 SVG는 다음과 같은 정확한 구조로 시작합니다:

```xml
<svg width="100%" viewBox="0 0 680 {HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Diagram content here -->

</svg>
```

`{HEIGHT}`를 실제 계산된 높이(마지막 요소의 하단 + 40px)로 바꿉니다.

### 노드 패턴

**한 줄 노드(44px):**
```xml
<g class="node c-blue">
  <rect x="100" y="20" width="180" height="44" rx="8" stroke-width="0.5"/>
  <text class="th" x="190" y="42" text-anchor="middle" dominant-baseline="central">Service name</text>
</g>
```

**두 줄 노드(56px):**
```xml
<g class="node c-teal">
  <rect x="100" y="20" width="200" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="200" y="38" text-anchor="middle" dominant-baseline="central">Service name</text>
  <text class="ts" x="200" y="56" text-anchor="middle" dominant-baseline="central">Short description</text>
</g>
```

**연결선(레이블 없음):**
```xml
<line x1="200" y1="76" x2="200" y2="120" class="arr" marker-end="url(#arrow)"/>
```

**컨테이너(점선 또는 실선):**
```xml
<g class="c-purple">
  <rect x="40" y="92" width="600" height="300" rx="16" stroke-width="0.5"/>
  <text class="th" x="66" y="116">Container label</text>
  <text class="ts" x="66" y="134">Subtitle info</text>
</g>
```

---

## 다이어그램 유형

주제에 맞는 레이아웃을 선택합니다:

1. **플로차트** — CI/CD 파이프라인, 요청 수명 주기, 승인 워크플로, 데이터 처리. 단일 방향 흐름(위에서 아래 또는 왼쪽에서 오른쪽). 한 행에 최대 4~5개 노드.
2. **구조/포함 관계** — 클라우드 인프라 중첩, 레이어가 있는 시스템 아키텍처. 내부 영역이 있는 큰 외부 컨테이너. 논리적 그룹화에는 점선 사각형.
3. **API/엔드포인트 맵** — REST 경로, GraphQL 스키마. 루트에서 시작하는 트리 구조로 리소스 그룹으로 분기하고, 각 그룹에 엔드포인트 노드를 포함.
4. **마이크로서비스 토폴로지** — 서비스 메시, 이벤트 기반 시스템. 서비스를 노드로 표시하고 통신 패턴을 화살표로 표시하며, 서비스 사이에 메시지 큐를 배치.
5. **데이터 흐름** — ETL 파이프라인, 스트리밍 아키텍처. 소스에서 처리 단계를 거쳐 싱크로 이어지는 왼쪽에서 오른쪽 흐름.
6. **물리/구조** — 차량, 건물, 하드웨어, 해부학. 물리적 형태에 맞는 도형을 사용 — 곡선 본체에는 `<path>`, 끝이 가늘어지는 형태에는 `<polygon>`, 원통형 부품에는 `<ellipse>`/`<circle>`, 구획에는 중첩된 `<rect>`. `references/physical-shape-cookbook.md`를 참조하세요.
7. **인프라/시스템 통합** — 스마트 시티, IoT 네트워크, 다중 도메인 시스템. 중앙 플랫폼과 하위 시스템을 연결하는 허브-스포크 레이아웃. 의미를 담은 선 스타일(`.data-line`, `.power-line`, `.water-pipe`, `.road`)을 사용합니다. `references/infrastructure-patterns.md`를 참조하세요.
8. **UI/대시보드 목업** — 관리자 패널, 모니터링 대시보드. 중첩된 차트/게이지/표시기 요소가 있는 화면 프레임. `references/dashboard-patterns.md`를 참조하세요.

물리, 인프라, 대시보드 다이어그램의 경우 생성하기 전에 해당 참조 파일을 불러오세요 — 각 파일에 바로 사용할 수 있는 CSS 클래스와 도형 원형이 제공됩니다.

---

## 검증 체크리스트

SVG를 마무리하기 전에 다음 항목을 **모두** 확인합니다:

1. 모든 `<text>`에 `t`, `ts`, 또는 `th` 클래스가 있습니다.
2. 상자 안의 모든 `<text>`에 `dominant-baseline="central"`이 있습니다.
3. 화살표로 사용되는 모든 연결 `<path>` 또는 `<line>`에 `fill="none"`이 있습니다.
4. 어떤 화살표 선도 관련 없는 상자를 통과하지 않습니다.
5. 14px 텍스트에 대해 `box_width >= (longest_label_chars × 8) + 48`입니다.
6. 12px 텍스트에 대해 `box_width >= (longest_label_chars × 6.5) + 48`입니다.
7. ViewBox 높이 = 가장 아래에 있는 요소 + 40px입니다.
8. 모든 콘텐츠가 x=40~640 안에 있습니다.
9. 색상 클래스(`c-*`)는 연결 `<path>`가 아니라 `<g>` 또는 도형 요소에 있습니다.
10. 화살표 `<defs>` 블록이 있습니다.
11. 그레이디언트, 그림자, 블러 또는 글로우 효과가 없습니다.
12. 모든 노드 테두리의 선 두께가 0.5px입니다.

---

## 출력 및 미리보기

### 기본값: 독립형 HTML 파일

사용자가 직접 열 수 있는 단일 `.html` 파일을 작성합니다. 서버나 종속성이 필요하지 않으며 오프라인에서도 작동합니다. 패턴:

```python
# 1. Load the template
template = skill_view("concept-diagrams", "templates/template.html")

# 2. Fill in title, subtitle, and paste your SVG
html = template.replace(
    "<!-- DIAGRAM TITLE HERE -->", "SN2 reaction mechanism"
).replace(
    "<!-- OPTIONAL SUBTITLE HERE -->", "Bimolecular nucleophilic substitution"
).replace(
    "<!-- PASTE SVG HERE -->", svg_content
)

# 3. Write to a user-chosen path (or ./ by default)
write_file("./sn2-mechanism.html", html)
```

사용자에게 여는 방법을 알려 줍니다:

```
# macOS
open ./sn2-mechanism.html
# Linux
xdg-open ./sn2-mechanism.html
```

### 선택 사항: 로컬 미리보기 서버(다중 다이어그램 갤러리)

사용자가 여러 다이어그램을 둘러볼 수 있는 갤러리를 명시적으로 원할 때만 사용합니다.

**규칙:**
- `127.0.0.1`에만 바인딩합니다. 절대로 `0.0.0.0`을 사용하지 않습니다. 모든 네트워크 인터페이스에 다이어그램을 노출하는 것은 공유 네트워크에서 보안 위험입니다.
- 사용하지 않는 포트를 선택하고(하나를 하드코딩하지 마세요) 선택한 URL을 사용자에게 알려 줍니다.
- 서버는 선택 사항이며 옵트인 방식입니다 — 먼저 독립형 HTML 파일을 제공하는 것을 우선합니다.

권장 패턴(운영 체제가 사용 가능한 임시 포트를 선택하도록 함):

```bash
# Put each diagram in its own folder under .diagrams/
mkdir -p .diagrams/sn2-mechanism
# ...write .diagrams/sn2-mechanism/index.html...

# Serve on loopback only, free port
cd .diagrams && python3 -c "
import http.server, socketserver
with socketserver.TCPServer(('127.0.0.1', 0), http.server.SimpleHTTPRequestHandler) as s:
    print(f'Serving at http://127.0.0.1:{s.server_address[1]}/')
    s.serve_forever()
" &
```

사용자가 고정 포트를 고집하면 `127.0.0.1:<port>`를 사용합니다 — 그래도 `0.0.0.0`은 절대 사용하지 않습니다. 서버를 중지하는 방법(`kill %1` 또는 `pkill -f "http.server"`)을 문서화합니다.

---

## 예시 참고 자료

`examples/` 디렉터리에는 완전하고 테스트된 다이어그램 15개가 포함되어 있습니다. 유사한 유형의 새 다이어그램을 작성하기 전에 작동하는 패턴을 살펴보세요:

| 파일 | 유형 | 보여 주는 내용 |
|------|------|--------------|
| `hospital-emergency-department-flow.md` | 플로차트 | 의미 기반 색상을 사용한 우선순위 라우팅 |
| `feature-film-production-pipeline.md` | 플로차트 | 단계별 워크플로, 가로 하위 흐름 |
| `automated-password-reset-flow.md` | 플로차트 | 오류 분기가 있는 인증 흐름 |
| `autonomous-llm-research-agent-flow.md` | 플로차트 | 되돌아가는 화살표, 의사 결정 분기 |
| `place-order-uml-sequence.md` | 시퀀스 | UML 시퀀스 다이어그램 스타일 |
| `commercial-aircraft-structure.md` | 물리 | 사실적인 형태를 위한 경로, 다각형, 타원 |
| `wind-turbine-structure.md` | 물리 단면 | 지하/지상 분리, 색상 코딩 |
| `smartphone-layer-anatomy.md` | 분해도 | 좌우 교대 레이블, 레이어 구성 요소 |
| `apartment-floor-plan-conversion.md` | 평면도 | 벽, 문, 점선 빨간색으로 표시한 변경 제안 |
| `banana-journey-tree-to-smoothie.md` | 서사적 여정 | 굽이치는 경로, 점진적인 상태 변화 |
| `cpu-ooo-microarchitecture.md` | 하드웨어 파이프라인 | 팬아웃, 메모리 계층 사이드바 |
| `sn2-reaction-mechanism.md` | 화학 | 분자, 곡선 화살표, 에너지 프로파일 |
| `smart-city-infrastructure.md` | 허브-스포크 | 시스템별 의미 기반 선 스타일 |
| `electricity-grid-flow.md` | 다단계 흐름 | 전압 계층, 흐름 마커 |
| `ml-benchmark-grouped-bar-chart.md` | 차트 | 그룹 막대, 이중 축 |

다음과 같이 예시를 불러옵니다:
```
skill_view(name="concept-diagrams", file_path="examples/<filename>")
```

---

## 빠른 참조: 상황별 사용법

| 사용자가 말하는 내용 | 다이어그램 유형 | 권장 색상 |
|-----------|--------------|------------------|
| "show the pipeline" | 플로차트 | 회색 시작/종료, 보라색 단계, 빨간색 오류, 청록색 배포 |
| "draw the data flow" | 데이터 파이프라인(왼쪽에서 오른쪽) | 회색 소스, 보라색 처리, 청록색 싱크 |
| "visualize the system" | 구조(포함 관계) | 보라색 컨테이너, 청록색 서비스, 코랄색 데이터 |
| "map the endpoints" | API 트리 | 보라색 루트, 리소스 그룹마다 하나의 램프 |
| "show the services" | 마이크로서비스 토폴로지 | 회색 인그레스, 청록색 서비스, 보라색 버스, 코랄색 워커 |
| "draw the aircraft/vehicle" | 물리 | 사실적인 형태를 위한 경로, 다각형, 타원 |
| "smart city / IoT" | 허브-스포크 통합 | 하위 시스템별 의미 기반 선 스타일 |
| "show the dashboard" | UI 목업 | 어두운 화면, 차트 색상: 알림에는 청록색, 보라색, 코랄색 |
| "power grid / electricity" | 다단계 흐름 | 전압 계층(HV/MV/LV 선 두께) |
| "wind turbine / turbine" | 물리 단면 | 기초 + 타워 절개도 + 색상으로 구분한 나셀 |
| "journey of X / lifecycle" | 서사적 여정 | 굽이치는 경로, 점진적인 상태 변화 |
| "layers of X / exploded" | 분해 레이어 뷰 | 수직 스택, 교대 레이블 |
| "CPU / pipeline" | 하드웨어 파이프라인 | 수직 단계, 실행 포트로의 팬아웃 |
| "floor plan / apartment" | 평면도 | 벽, 문, 점선 빨간색으로 표시한 변경 제안 |
| "reaction mechanism" | 화학 | 원자, 결합, 곡선 화살표, 전이 상태, 에너지 프로파일 |
