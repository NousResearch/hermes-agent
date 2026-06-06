---
title: "Architecture Diagram — 다크 테마의 SVG 아키텍처/클라우드/인프라 다이어그램 HTML 파일"
sidebar_label: "Architecture Diagram"
description: "다크 테마의 SVG 아키텍처/클라우드/인프라 다이어그램 HTML 파일"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Architecture Diagram

다크 테마의 SVG 아키텍처/클라우드/인프라 다이어그램이 포함된 독립형 HTML 파일 생성.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/architecture-diagram` |
| 버전 | `1.0.0` |
| 저자 | Cocoon AI (hello@cocoon-ai.com), Hermes Agent에 의해 포팅됨 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `architecture`, `diagrams`, `SVG`, `HTML`, `visualization`, `infrastructure`, `cloud` |
| 관련 스킬 | [`concept-diagrams`](/docs/user-guide/skills/optional/creative/creative-concept-diagrams), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# Architecture Diagram Skill

외부 도구, API 키, 렌더링 라이브러리 없이 인라인 SVG 그래픽이 포함된 독립 실행형 HTML 파일로 전문적이고 어두운 테마의 기술 아키텍처 다이어그램을 생성합니다. 단순히 HTML 파일을 작성하고 브라우저에서 열면 됩니다.

## 범위

**가장 적합한 용도:**
- 소프트웨어 시스템 아키텍처 (프론트엔드 / 백엔드 / 데이터베이스 계층)
- 클라우드 인프라 (VPC, 리전, 서브넷, 매니지드 서비스)
- 마이크로서비스 / 서비스 메시 토폴로지
- 데이터베이스 + API 맵, 배포 다이어그램
- 어둡고 그리드 배경의 미학에 어울리는 기술/인프라 주제의 모든 것

**다음과 같은 경우에는 다른 스킬을 우선적으로 고려하세요:**
- 물리, 화학, 수학, 생물학 등 기타 과학적 주제
- 물리적 물체 (차량, 하드웨어, 해부도, 단면도)
- 평면도, 서사적 여정(narrative journeys), 교육용 / 교과서 스타일 시각 자료
- 손으로 그린 칠판 스케치 (`excalidraw` 고려)
- 애니메이션 설명 (`animation` 스킬 고려)

해당 주제에 대해 더 전문화된 스킬이 있다면 그것을 선호하세요. 적합한 스킬이 없는 경우 이 스킬이 일반적인 SVG 다이어그램 대안으로 사용될 수도 있습니다 — 이 경우 출력물은 아래 설명된 어두운 기술적 미학을 따릅니다.

[Cocoon AI's architecture-diagram-generator](https://github.com/Cocoon-AI/architecture-diagram-generator) (MIT) 기반.

## 워크플로우

1. 사용자가 시스템 아키텍처(구성요소, 연결, 기술)를 설명합니다.
2. 아래의 디자인 시스템에 따라 HTML 파일을 생성합니다.
3. `write_file`을 사용하여 `.html` 파일(예: `~/architecture-diagram.html`)로 저장합니다.
4. 사용자가 어떤 브라우저에서든 엽니다 — 오프라인에서 작동하며 종속성이 없습니다.

### 출력 위치

사용자가 지정한 경로에 다이어그램을 저장하거나 기본적으로 현재 작업 디렉터리에 저장합니다:
```
./[project-name]-architecture.html
```

### 미리보기

저장한 후 사용자에게 열어보도록 제안합니다:
```bash
# macOS
open ./my-architecture.html
# Linux
xdg-open ./my-architecture.html
```

## 디자인 시스템 및 시각적 언어

### 색상 팔레트 (시맨틱 매핑)

컴포넌트를 분류하기 위해 특정 `rgba` 채우기(fill)와 `hex` 테두리선(stroke)을 사용하세요:

| 컴포넌트 유형 | 채우기 (rgba) | 테두리 (Hex) |
| :--- | :--- | :--- |
| **Frontend** | `rgba(8, 51, 68, 0.4)` | `#22d3ee` (cyan-400) |
| **Backend** | `rgba(6, 78, 59, 0.4)` | `#34d399` (emerald-400) |
| **Database** | `rgba(76, 29, 149, 0.4)` | `#a78bfa` (violet-400) |
| **AWS/Cloud** | `rgba(120, 53, 15, 0.3)` | `#fbbf24` (amber-400) |
| **Security** | `rgba(136, 19, 55, 0.4)` | `#fb7185` (rose-400) |
| **Message Bus** | `rgba(251, 146, 60, 0.3)` | `#fb923c` (orange-400) |
| **External** | `rgba(30, 41, 59, 0.5)` | `#94a3b8` (slate-400) |

### 타이포그래피 및 배경
- **폰트:** JetBrains Mono (Monospace), Google Fonts에서 불러옴
- **크기:** 12px (이름), 9px (하위 라벨), 8px (주석), 7px (작은 라벨)
- **배경:** 미세한 40px 그리드 패턴이 있는 Slate-950 (`#020617`)

```svg
<!-- Background Grid Pattern -->
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.5"/>
</pattern>
```

## 기술적 구현 세부 사항

### 컴포넌트 렌더링
컴포넌트는 1.5px 두께의 테두리를 가진 모서리가 둥근 사각형(`rx="6"`)입니다. 화살표가 반투명한 채우기 영역을 통해 비치는 것을 방지하려면 **이중 사각형 마스킹 기법(double-rect masking technique)**을 사용하세요:
1. 불투명한 배경 사각형 그리기 (`#0f172a`)
2. 그 위에 반투명하게 스타일이 적용된 사각형 그리기

### 연결 규칙
- **Z-순서(Z-Order):** SVG에서 화살표를 *초기*에 (그리드 바로 뒤에) 그려서 컴포넌트 박스들 뒤에 렌더링되게 합니다.
- **화살촉(Arrowheads):** SVG 마커를 통해 정의됨
- **보안 흐름(Security Flows):** 장미색(`#fb7185`)의 점선 사용
- **경계(Boundaries):**
  - *보안 그룹(Security Groups):* 점선(`4,4`), 장미색
  - *리전(Regions):* 큰 점선(`8,4`), 호박색(amber), `rx="12"`

### 간격 및 레이아웃 논리
- **표준 높이:** 60px (서비스); 80-120px (대형 컴포넌트)
- **수직 간격:** 컴포넌트 간 최소 40px
- **메시지 버스:** 서비스와 겹쳐서는 안 되며, 서비스 *사이 간격*에 위치해야 합니다.
- **범례 배치:** **매우 중요.** 모든 경계 상자 바깥에 배치되어야 합니다. 모든 경계의 가장 낮은 Y 좌표를 계산하고 범례를 그보다 최소 20px 아래에 배치하세요.

## 문서 구조

생성된 HTML 파일은 4부분 구조를 따릅니다:
1. **헤더:** 깜박이는 점 표시기와 부제목이 있는 제목
2. **메인 SVG:** 둥근 테두리 카드 내에 포함된 다이어그램
3. **요약 카드:** 다이어그램 아래, 핵심 정보를 담은 3개의 카드 그리드
4. **푸터:** 최소한의 메타데이터

### 정보 카드 패턴
```html
<div class="card">
  <div class="card-header">
    <div class="card-dot cyan"></div>
    <h3>Title</h3>
  </div>
  <ul>
    <li>• Item one</li>
    <li>• Item two</li>
  </ul>
</div>
```

## 출력 요구 사항
- **단일 파일:** 하나의 독립적인 `.html` 파일
- **외부 종속성 없음:** (Google Fonts를 제외한) 모든 CSS와 SVG는 인라인이어야 합니다.
- **JavaScript 없음:** (깜박이는 점과 같은) 모든 애니메이션에는 순수 CSS를 사용하세요.
- **호환성:** 모든 최신 웹 브라우저에서 올바르게 렌더링되어야 합니다.

## 템플릿 참조

전체 HTML 템플릿을 불러와 정확한 구조, CSS, SVG 구성 요소 예제를 확인하세요:

```
skill_view(name="architecture-diagram", file_path="templates/template.html")
```

이 템플릿에는 모든 컴포넌트 유형(프론트엔드, 백엔드, 데이터베이스, 클라우드, 보안), 화살표 스타일(표준, 점선, 곡선), 보안 그룹, 지역 경계, 범례의 작동 예제가 포함되어 있습니다. 다이어그램을 생성할 때 이를 구조적 레퍼런스로 활용하세요.
