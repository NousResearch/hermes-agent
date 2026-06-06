---
title: "Excel Author"
sidebar_label: "Excel Author"
description: "openpyxl 라이브러리를 사용하여 0부터 전문적인, 기관 수준의 Excel 재무 모델(수식 포함) 구축"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Excel Author

`openpyxl` 라이브러리를 사용하여 0부터 전문적인, 기관 수준의 Excel 재무 모델(.xlsx)을 만듭니다. 수식 처리, 조건부 서식, 오류 검사, 재계산을 관리합니다. DCF 모델, 3재무제표 모델(3-Statement Model), 비교 기업 분석(Comps Analysis)을 포함한 모든 재무 모델링 스킬의 기초로 사용됩니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/finance/excel-author`를 사용하여 설치 |
| 경로 | `optional-skills/finance/excel-author` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (adapted by Nous Research) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `finance`, `excel`, `openpyxl`, `modeling`, `investment-banking`, `data-analysis` |
| 관련 스킬 | [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`comps-analysis`](/docs/user-guide/skills/optional/finance/finance-comps-analysis), [`lbo-model`](/docs/user-guide/skills/optional/finance/finance-lbo-model), [`3-statement-model`](/docs/user-guide/skills/optional/finance/finance-3-statement-model) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

# 헤드리스(Headless) Excel 에이전트

이 스킬을 사용하면 `openpyxl`을 통해 헤드리스 프로그래밍 방식으로 Excel(.xlsx) 모델을 만들고 편집할 수 있습니다. COM 상호 운용성이나 백그라운드에서 Excel 애플리케이션을 구동할 필요가 없습니다. macOS, Linux, Windows 환경 전반에서 에이전트 샌드박스 내에서 100% 실행됩니다.

수식 작성, 서식 지정 및 재계산을 위한 도구와 엄격한 규칙이 포함되어 있어 텍스트 덤프가 아닌 **작동하는 모델**을 생성합니다.

## 전제 조건

1. **Python 라이브러리**: `uv pip install openpyxl`
2. **재계산 엔진**: LibreOffice (`recalc.py`가 백그라운드 수식 평가에 이를 사용함)
   - Mac: `brew install --cask libreoffice`
   - Linux: `sudo apt install libreoffice`
   - Windows: `choco install libreoffice-fresh`

## 개요
이 스킬의 핵심 목적은 계산과 관련된 금융 작업을 수행할 때 CSV 텍스트 덤프를 피하고 **실시간 계산, 참조 및 시나리오 분석이 포함된 올바른 `.xlsx` 파일을 작성**하도록 강제하는 것입니다. 

비교 기업 분석(Comps), DCF, LBO, 3재무제표 모델 등 모든 것을 지원합니다.

## 핵심 규칙: 수식 > 하드코딩

모델을 구축할 때 모든 **파생된** 값, 비율, 합계 및 시나리오 출력은 반드시 Excel 수식으로 작성해야 합니다. 
**파이썬에서 값을 계산한 뒤 최종 숫자를 셀에 기록해서는 절대 안 됩니다.**

* **나쁨**: `ws["C10"] = 125000` (이것이 수익 성장률 %에 의존하는 경우)
* **좋음**: `ws["C10"] = "=C9*(1+Assumptions!$B$5)"`

이것이 투자 은행 수준 모델과 일회성 계산 스크립트의 차이입니다.

## 서식 표준 (IB 규칙)

코드에서 `openpyxl`의 `Font`, `PatternFill`, `Border`, `Alignment`를 사용하여 모든 재무 모델에서 기대하는 시각적 계층 구조를 만듭니다.

1. **글꼴 색상 구분 (필수!)**:
   - 파란색 (`#0000FF`): 하드코드된/수동 입력 (예: 과거 수익, 가정 드라이버).
   - 검은색 (`#000000`): 공식 및 계산 (예: 총 마진, EBITDA).
   - 녹색 (`#008000`): 다른 시트로의 링크 (예: BS 시트에서 IS 순이익 연결).

2. **숫자 형식**:
   - `openpyxl`의 `.number_format` 속성을 사용합니다.
   - 통화: 수백만 달러의 경우 `#,##0` 단위, 소수점 없는 단위로 포맷팅.
   - 비율: `0.0%` 포맷팅 사용.
   - 배수: `0.0"x"` 포맷팅 사용.

3. **열 너비 및 행 높이**:
   - 기본적으로 라벨이 들어가는 첫 번째 열(`A`)의 너비를 `35` 정도로 조정합니다.
   - 연도별 데이터가 들어가는 열들의 너비는 `12` 정도로 설정합니다.

## `recalc.py`를 사용한 수식 재계산 및 오류 검사

`openpyxl`은 수식의 계산 결과를 즉각 평가하지 않습니다. 그것은 단순히 텍스트 문자열(예: `"=SUM(A1:A5)"`)을 파일에 씁니다.

따라서 모델을 구축한 후 **반드시** `scripts/recalc.py` 유틸리티를 실행해야 합니다. 이 스크립트는 백그라운드에서 LibreOffice를 실행하여 모든 종속성을 다시 계산하고, 계산된 값을 저장하며, 셀에서 수식 오류(#REF!, #DIV/0! 등)가 발생하지 않았는지 검사합니다.

### 워크플로우:

1. `openpyxl` 스크립트를 작성하고 실행하여 `model_v1.xlsx`를 생성합니다.
2. `recalc.py` 스크립트를 백그라운드 작업으로 실행하거나 OS 명령어로 실행합니다:
   ```bash
   python ~/.hermes/skills/finance/excel-author/scripts/recalc.py path/to/model_v1.xlsx 30
   ```
   (30은 LibreOffice 처리의 타임아웃 초 단위입니다.)
3. `recalc.py`는 JSON 요약을 반환합니다:
   ```json
   {
     "status": "errors_found",
     "total_errors": 2,
     "total_formulas": 145,
     "error_summary": {
       "#REF!": {
         "count": 2,
         "locations": ["DCF!B25", "DCF!C25"]
       }
     }
   }
   ```
4. 만약 상태가 `errors_found`라면, 파이썬 스크립트를 수정하여 잘못된 수식이나 깨진 참조를 고치고, `model_v2.xlsx`를 저장한 다음, 오류 개수가 0이 될 때까지 다시 검사하세요.
5. 오직 `status: success`인 파일만 완성된 것으로 간주하고 사용자에게 전달해야 합니다.

## 시나리오 분석: 표(Data Tables) 피하기

`openpyxl`은 기본적으로 시나리오 테이블(What-If 분석) 생성 도구를 완전히 지원하지 않습니다. 모델에 민감도 분석이 필요하다면(예를 들어 WACC 대비 성장률에 따른 가치 평가), Excel의 데이터 테이블 도구 사용 지침을 남겨두지 마세요. 
대신 2D 그리드를 설정하고 **해당 특정 입력 쌍을 사용하여 결과를 계산하는 긴 형식의 하드코딩된 수식 문자열을 작성**하세요.

예제 모델을 생성할 때, 항상 단일 결과만 표시하지 말고, 시나리오 매개변수를 전환할 수 있는 셀(예: 셀 $B$1에 "1" 입력 시 Bear, "2" 입력 시 Base, "3" 입력 시 Bull)을 포함시켜 전체 시트의 수식이 연동되도록 하세요.

## 셀 주석 요구 사항 (강력 권장)

주요 하드코딩된 입력(예: 10-K에서 가져온 과거 매출액, FactSet에서 조회한 WACC) 옆에는 `openpyxl.comments.Comment`를 활용해 출처를 밝히세요:
```python
from openpyxl.comments import Comment
cell = ws["B5"]
cell.comment = Comment("Source: Apple FY23 10-K, page 32", "Hermes Agent")
```
이를 통해 사용자는 에이전트가 만든 모델을 쉽게 신뢰하고 감사할 수 있습니다.

## 요약 프로세스

1. `openpyxl`을 사용하여 데이터 딕셔너리를 포함하는 Python 스크립트 뼈대 생성
2. `Workbook()` 초기화 및 탭/구조 구축
3. 하드코드된(과거 데이터/가정) 값 삽입 (파란색 글꼴)
4. 수식 적용(계산/참조) (검은색/녹색 글꼴)
5. 워크북을 `.xlsx` 형식으로 저장
6. **MANDATORY**: `recalc.py` 실행하여 #REF! / #DIV/0! 점검
7. 오류 발생 시 3~6단계 반복 (디버깅)
8. 클린 모델 사용자에게 전달

## 기여

이 스킬은 Anthropic의 Financial Services용 Claude 플러그인 제품군 (Apache-2.0)에서 채택되었습니다. Office-JS / Cowork 라이브 Excel 경로는 제거되었습니다. 이 버전은 헤드리스 openpyxl을 대상으로 합니다. 원본: https://github.com/anthropics/financial-services
