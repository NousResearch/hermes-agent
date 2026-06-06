---
title: "Pptx Author"
sidebar_label: "Pptx Author"
description: "python-pptx를 사용하여 투자 은행용 프레젠테이션(피치 덱, 회사 프로필, 벤치마킹) 구축"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Pptx Author

`python-pptx` 라이브러리를 사용하여 전문적인 투자 은행 스타일 프레젠테이션(피치 덱, 회사 프로필, 벤치마킹 보고서)을 프로그래밍 방식으로 구축합니다. 이 스킬은 깔끔하고 브랜드화되지 않은 템플릿(여백 여백, 바닥글, 마스터 슬라이드)을 처리하며 표, 차트, 팀 프로필 및 데이터 통합을 제공합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/finance/pptx-author`를 사용하여 설치 |
| 경로 | `optional-skills/finance/pptx-author` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (adapted by Nous Research) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `finance`, `powerpoint`, `presentation`, `pitch-deck`, `python-pptx`, `investment-banking` |
| 관련 스킬 | [`excel-author`](/docs/user-guide/skills/optional/finance/finance-excel-author), [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`comps-analysis`](/docs/user-guide/skills/optional/finance/finance-comps-analysis) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

# 헤드리스(Headless) 파워포인트 에이전트

이 스킬을 사용하면 `python-pptx`를 통해 헤드리스 프로그래밍 방식으로 PowerPoint(.pptx) 파일을 만들고 편집할 수 있습니다. Office.js, COM 상호 운용성 또는 기본 PowerPoint 앱이 필요하지 않습니다. macOS, Linux, Windows 환경 전반에서 에이전트 워크플로우 내에서 완전히 실행됩니다.

템플릿, 스크립트, 예제 및 전체 API를 제공하여 회사 프로필, 피치 덱, 운영 벤치마킹 보고서 등 투자 은행 수준의 프레젠테이션을 생성합니다.

## 개요
이 스킬은 기업용 스타일 슬라이드를 프로그래밍 방식으로 조립하는 방법을 가르칩니다. 깨끗하고 브랜드가 없는 프레젠테이션을 생성하기 위해 `templates/`에 있는 강력한 빈 템플릿(`.pptx`) 집합과 `scripts/`에 있는 Python 래퍼 스크립트를 사용합니다.

**Python 스크립트를 사용하여 다음을 수행합니다:**
- 슬라이드(제목, 목차, 내용, 팀 프로필, 표, 차트) 추가
- 제목, 글머리 기호 텍스트(계층 구조 포함), 표 셀의 텍스트 삽입/포맷팅
- 데이터프레임 구조에서 기업 재무/벤치마킹 데이터 입력
- `python-pptx`를 사용하여 기본 차트(막대, 선, 파이) 렌더링
- 여러 데이터 소스(DCF 분석, Comps 테이블 등)의 출력을 단일 피치 덱으로 모으기

## 필수 도구:
이 스킬의 핵심은 `python-pptx`입니다. 아직 설치되지 않았다면 워크플로우의 첫 번째 단계에서 Python 환경에 이를 설치해야 합니다:
`pip install python-pptx pandas`

## 핵심 디렉토리
- `scripts/`: 슬라이드를 구축하기 위한 Python 헬퍼/래퍼 스크립트(예: 표나 차트가 있는 슬라이드 추가, 텍스트 설정 등)가 포함되어 있습니다. 일반적인 피치 덱 생성 과정을 안내하기 위해 `build_deck.py`와 같은 마스터 스크립트를 작성하거나 사용하게 됩니다.
- `templates/`: 여백, 폰트 비율(제목 24pt~32pt, 바디 14~18pt), 마스터 레이아웃이 설정된 빈 `.pptx` 템플릿(예: `IB_Pitchbook_Template.pptx`)이 포함되어 있습니다.
- `examples/`: 생성된 .pptx 및 스크립트가 어떻게 보이는지 보여주는 샘플 출력이 포함되어 있습니다.

## 주요 사용 사례

1. **회사 프로필 생성 (Tearsheets):** 회사 개요, 경영진 프로필, 주가 차트(정적인 이미지 또는 네이티브 차트로), 주요 성과 지표(KPI) 및 재무 요약이 있는 1-2페이지 슬라이드를 자동으로 생성합니다.
2. **벤치마킹 / Comps 프레젠테이션:** 비교 분석 도구(`comps-analysis`)의 출력을 가져와 Excel의 데이터 프레임/CSV 출력을 PowerPoint의 표 슬라이드 또는 막대 차트로 변환하여 렌더링합니다.
3. **M&A 피치 덱:** (1) 제목 슬라이드, (2) 경영 요약, (3) 산업 환경, (4) 대상 회사 개요, (5) 재무 가치 평가(DCF 요약 출력), (6) 연락처와 같은 표준 투자 은행 흐름으로 완전한 뼈대 덱을 구성합니다.

## 표준화된 프레젠테이션 구조 (IB 기준)
프레젠테이션을 생성할 때, 다음 구조를 따르세요:
- **글꼴:** 모양을 일관되게 유지하기 위해 Arial 또는 Calibri (템플릿에서 적용됨).
- **제목:** 한 줄(최대 두 줄), 행동 중심(Action-oriented) (예: "2023년 지속적인 수익 성장 달성" 대신 "재무 개요").
- **여백/공백:** 항상 슬라이드 가장자리에서 숨쉴 공간을 남겨두세요; `python-pptx`에서 위치를 지정할 때 절대 인치/cm 픽셀 오프셋을 존중하세요(템플릿에 미리 매핑되어 있음).
- **데이터 출처:** 재무 데이터가 있는 모든 슬라이드의 왼쪽 하단(바닥글 레이아웃 위)에 "Source: Company Filings, FactSet" 등을 나타내는 10pt/8pt 텍스트 상자를 포함하세요.

## 에이전트 단계별 지침

### 1. 콘텐츠 계획
코드를 작성하기 전에 만들고자 하는 프레젠테이션의 개요를 잡으세요. 예:
- 슬라이드 1: 제목 슬라이드 (프로젝트 이름, 대상 회사, 날짜)
- 슬라이드 2: 경영 요약 (3-4개의 핵심 사항)
- 슬라이드 3: 재무 요약 표
- 슬라이드 4: 가치 평가 차트

### 2. 템플릿 활용
Python 스크립트의 시작점으로 항상 `templates/IB_Pitchbook_Template.pptx`를 로드하세요. 백지 상태에서 시작하지 마세요.
```python
from pptx import Presentation
prs = Presentation('~/.hermes/skills/finance/pptx-author/templates/IB_Pitchbook_Template.pptx')
```

### 3. 데이터 준비
사용자 또는 시스템에 표나 차트에 들어갈 데이터가 있는 경우, Python 내에서 깔끔한 구조(딕셔너리 리스트 또는 Pandas DataFrame)로 정리하세요. 이는 `python-pptx` 도우미 함수에 매개변수로 직접 전달하는 데 매우 중요합니다.

### 4. 스크립트 구성
덱을 조립하는 단일 스크립트를 작성합니다. 다음과 같은 구문을 사용합니다:
```python
# Create title slide
title_slide_layout = prs.slide_layouts[0] # Assuming 0 is Title
slide = prs.slides.add_slide(title_slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "Project Phoenix"
subtitle.text = "Acquisition Discussion Materials\nOctober 2024"
```

### 5. 표 삽입 (가치 평가 / Comps용)
`python-pptx`의 표 기능을 활용합니다:
```python
# Slide layout for content
content_layout = prs.slide_layouts[5] # e.g., Title Only
slide = prs.slides.add_slide(content_layout)
slide.shapes.title.text = "Comparable Company Analysis"

# Add table (rows, cols, left, top, width, height)
from pptx.util import Inches
x, y, cx, cy = Inches(1), Inches(2), Inches(8), Inches(4)
table_shape = slide.shapes.add_table(4, 3, x, y, cx, cy)
table = table_shape.table

# Fill table
table.cell(0,0).text = "Company"
table.cell(0,1).text = "EV/Revenue"
table.cell(0,2).text = "EV/EBITDA"
# ... loop through data ...
```

### 6. 출력 저장 및 반환
스크립트를 실행하여 사용자의 현재 작업 디렉토리 또는 지정된 출력 경로에 결과를 저장합니다:
```python
prs.save('./out/Project_Phoenix_Pitch_Deck.pptx')
```
그런 다음, 작성된 파일의 위치를 사용자에게 알립니다.

## 통합: 모델을 슬라이드로 가져오기
사용자가 DCF 모델 및 Comps 분석을 실행한 후, 해당 스크립트의 출력(`recalc.py`에서 JSON으로, 또는 pandas 스크립트를 통해 생성된 요약 CSV/DataFrames)을 요약 슬라이드로 캡처할 수 있습니다. 

1. **가치 평가 축구장 차트(Football Field Chart)**: 스크립트 도우미가 값 범위(예: DCF 범위 $10-15, Comps 범위 $12-18, 52주 고/저)를 생성할 수 있도록 모델의 범위를 읽고 막대/선 요약 슬라이드를 작성합니다.
2. **이그제큐티브 대시보드 요약**: 대상 회사의 5년 수익 CAGR을 손익계산서 탭에서 가져와 슬라이드 총알(Bullet point)에 요약으로 삽입합니다.

## 주의 사항
- **이미지 배치**: 정적 이미지(예: 회사 로고, 웹에서 가져온 가격 차트)를 넣을 경우, 해상도 크기 조정 규칙을 준수하세요. 너무 크게 만들지 마세요.
- **오버플로우 처리**: `python-pptx`는 텍스트를 자동으로 줄여주지 않습니다. 단일 텍스트 상자 내에 너무 많은 텍스트를 생성하지 않도록 주의하세요(일반적으로 슬라이드당 5~6개 글머리 기호 이하 권장).
- **차트**: 네이티브 PPT 차트는 가능하지만, Python 스크립트 내에서 카테고리와 시리즈 설정이 필요합니다. 복잡할 경우, 단순한 비교 표를 삽입하는 것이 가치 평가 출력에는 더 나을 수 있습니다.

## 데이터 소스

프레젠테이션에는 종종 데이터가 필요합니다. 우선적으로 구조화된 데이터 제공자(사용 가능한 경우)를 통해 데이터를 조회하거나, 명시된 경우 회사 재무 문서를 사용하세요. 

재무 분석 및 프레젠테이션 작성을 완료하는 행위가 주요 가치 제안입니다. 데이터를 지어내지 마세요. 알 수 없는 경우 `[COMPANY XYZ]` 또는 `[XX.X]`를 사용하여 사용자가 수동으로 검증하고 삽입할 수 있도록 자리 표시자를 남겨두세요.

## 기여

이 스킬은 Anthropic의 Financial Services용 Claude 플러그인 제품군 (Apache-2.0)에서 채택되었습니다. 원본: https://github.com/anthropics/financial-services
