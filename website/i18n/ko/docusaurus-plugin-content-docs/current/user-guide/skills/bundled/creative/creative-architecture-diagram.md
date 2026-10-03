---
title: "아키텍처 다이어그램 — HTML로 만드는 어두운 테마의 SVG 아키텍처/클라우드/인프라 다이어그램"
sidebar_label: "아키텍처 다이어그램"
description: "HTML로 만드는 어두운 테마의 SVG 아키텍처/클라우드/인프라 다이어그램"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 아키텍처 다이어그램

HTML로 만드는 어두운 테마의 SVG 아키텍처/클라우드/인프라 다이어그램입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본 설치됨) |
| 경로 | `skills/creative/architecture-diagram` |
| 버전 | `1.0.0` |
| 작성자 | Cocoon AI (hello@cocoon-ai.com), Hermes Agent 포팅 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `architecture`, `diagrams`, `SVG`, `HTML`, `visualization`, `infrastructure`, `cloud` |
| 관련 스킬 | [`concept-diagrams`](/docs/user-guide/skills/optional/creative/creative-concept-diagrams), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보는 내용입니다.
:::

# 아키텍처 다이어그램 스킬

인라인 SVG 그래픽을 포함한 독립 실행형 HTML 파일로 전문적인 어두운 테마의 기술 아키텍처 다이어그램을 생성합니다. 외부 도구, API 키, 렌더링 라이브러리는 필요하지 않습니다. HTML 파일을 작성하고 브라우저에서 열기만 하면 됩니다.

## 범위

**다음에 가장 적합합니다:**
- 소프트웨어 시스템 아키텍처(프런트엔드 / 백엔드 / 데이터베이스 계층)
- 클라우드 인프라(VPC, 리전, 서브넷, 관리형 서비스)
- 마이크로서비스 / 서비스 메시 토폴로지
- 데이터베이스 + API 맵, 배포 다이어그램
- 어두운 격자 배경 미학에 어울리는 기술·인프라 주제

**다음 주제에는 먼저 다른 방법을 살펴보세요:**
- 물리학, 화학, 수학, 생물학 또는 기타 과학 주제
- 물리적 사물(차량, 하드웨어, 해부도, 단면도)
- 평면도, 서사적 여정, 교육 자료 / 교과서 스타일의 시각 자료
- 손으로 그린 화이트보드 스케치(`excalidraw` 고려)
- 애니메이션 설명 자료(애니메이션 스킬 고려)

주제에 더 특화된 스킬을 사용할 수 있다면 그것을 우선하세요. 적합한 스킬이 없다면 이 스킬을 일반적인 SVG 다이어그램 대체 수단으로 사용할 수도 있습니다. 이 경우에도 아래 설명된 어두운 기술 테마의 미학이 적용됩니다.

[Cocoon AI의 architecture-diagram-generator](https://github.com/Cocoon-AI/architecture-diagram-generator)를 기반으로 합니다(MIT).

## 워크플로

1. 사용자가 시스템 아키텍처(구성 요소, 연결, 기술)를 설명합니다.
2. 아래 디자인 시스템에 따라 HTML 파일을 생성합니다.
3. `write_file`로 HTML 파일을 `.html` 파일에 저장합니다(예: `~/architecture-diagram.html`)
4. 사용자가 아무 브라우저에서나 엽니다 — 오프라인에서도 작동하며 종속성이 없습니다.

### 출력 위치

사용자가 지정한 경로에 다이어그램을 저장하거나, 기본값으로 현재 작업 디렉터리에 저장합니다:
```
./[project-name]-architecture.html
```

### 미리 보기

저장한 뒤 사용자에게 다음과 같이 열어 보라고 안내하세요:
```bash
# macOS
open ./my-architecture.html
# Linux
xdg-open ./my-architecture.html
```

## 디자인 시스템 및 시각 언어

### 색상 팔레트(의미 매핑)

구성 요소를 분류할 때 다음의 특정 `rgba` 채우기와 hex 선 색상을 사용하세요:

| 구성 요소 유형 | 채우기(rgba) | 선(Hex) |
| :--- | :--- | :--- |
| **프런트엔드** | `rgba(8, 51, 68, 0.4)` | `#22d3ee` (cyan-400) |
| **백엔드** | `rgba(6, 78, 59, 0.4)` | `#34d399` (emerald-400) |
| **데이터베이스** | `rgba(76, 29, 149, 0.4)` | `#a78bfa` (violet-400) |
| **AWS/클라우드** | `rgba(120, 53, 15, 0.3)` | `#fbbf24` (amber-400) |
| **보안** | `rgba(136, 19, 55, 0.4)` | `#fb7185` (rose-400) |
| **메시지 버스** | `rgba(251, 146, 60, 0.3)` | `#fb923c` (orange-400) |
| **외부** | `rgba(30, 41, 59, 0.5)` | `#94a3b8` (slate-400) |

### 타이포그래피 및 배경
- **글꼴:** Google Fonts에서 불러오는 JetBrains Mono(고정폭)
- **크기:** 12px(이름), 9px(하위 레이블), 8px(주석), 7px(작은 레이블)
- **배경:** 미묘한 40px 격자 패턴이 있는 Slate-950(`#020617`)

```svg
<!-- Background Grid Pattern -->
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.5"/>
</pattern>
```

## 기술 구현 세부 사항

### 구성 요소 렌더링
구성 요소는 1.5px 선이 있는 둥근 사각형(`rx="6"`)입니다. 반투명한 채우기를 화살표가 관통해 보이지 않게 하려면 **이중 사각형 마스킹 기법**을 사용하세요:
1. 불투명한 배경 사각형(`#0f172a`)을 그립니다.
2. 그 위에 반투명한 스타일 사각형을 그립니다.

### 연결 규칙
- **Z 순서:** SVG에서 화살표가 구성 요소 상자 뒤에 표시되도록 격자 다음에 화살표를 먼저 그립니다.
- **화살촉:** SVG 마커로 정의합니다.
- **보안 흐름:** 장밋빛 색상(`#fb7185`)의 점선을 사용합니다.
- **경계:**
  - *보안 그룹:* 점선(`4,4`), 장밋빛 색상
  - *리전:* 큰 점선(`8,4`), 호박색, `rx="12"`

### 간격 및 레이아웃 로직
- **표준 높이:** 60px(서비스), 80-120px(큰 구성 요소)
- **세로 간격:** 구성 요소 사이 최소 40px
- **메시지 버스:** 서비스와 겹치지 않게 서비스 사이의 간격에 배치해야 합니다.
- **범례 배치:** **중요.** 모든 경계 상자 바깥에 배치해야 합니다. 모든 경계의 가장 낮은 Y 좌표를 계산하고 범례를 그보다 최소 20px 아래에 배치하세요.

## 문서 구조

생성되는 HTML 파일은 네 부분으로 구성됩니다:
1. **헤더:** 펄스 점 표시기와 부제목이 있는 제목
2. **메인 SVG:** 둥근 테두리 카드 안에 포함된 다이어그램
3. **요약 카드:** 다이어그램 아래에 배치하는 상위 세부 정보용 카드 세 개의 격자
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
- **단일 파일:** 자체적으로 완결된 `.html` 파일 하나
- **외부 종속성 없음:** 모든 CSS와 SVG는 인라인이어야 합니다(Google Fonts 제외).
- **JavaScript 없음:** 펄스 점과 같은 애니메이션은 순수 CSS로 구현합니다.
- **호환성:** 모든 최신 웹 브라우저에서 올바르게 렌더링되어야 합니다.

## 템플릿 참고

정확한 구조, CSS, SVG 구성 요소 예시가 담긴 전체 HTML 템플릿을 불러오세요:

```
skill_view(name="architecture-diagram", file_path="templates/template.html")
```

템플릿에는 모든 구성 요소 유형(frontend, backend, database, cloud, security), 화살표 스타일(표준, 점선, 곡선), 보안 그룹, 리전 경계, 범례의 작동 예시가 들어 있습니다. 다이어그램을 생성할 때 구조 참고 자료로 사용하세요.

