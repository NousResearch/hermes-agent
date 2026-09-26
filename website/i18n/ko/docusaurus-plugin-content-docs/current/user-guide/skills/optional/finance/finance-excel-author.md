---
title: "Excel Author — openpyxl을 사용해 감사 가능한 금융 워크북을 헤드리스로 구축"
sidebar_label: "Excel Author"
description: "openpyxl을 사용해 감사 가능한 금융 워크북을 헤드리스로 구축"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Excel Author

openpyxl을 사용해 감사 가능한 금융 워크북을 헤드리스로 구축합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/finance/excel-author`로 설치 |
| 경로 | `optional-skills/finance/excel-author` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (Nous Research가 적용) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `excel`, `openpyxl`, `금융`, `스프레드시트`, `모델링` |
| 관련 스킬 | [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx), [`pptx-author`](/docs/user-guide/skills/optional/finance/finance-pptx-author), [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`comps-analysis`](/docs/user-guide/skills/optional/finance/finance-comps-analysis), [`lbo-model`](/docs/user-guide/skills/optional/finance/finance-lbo-model), [`3-statement-model`](/docs/user-guide/skills/optional/finance/finance-3-statement-model) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침입니다.
:::

# excel-author

`openpyxl`을 사용해 디스크에 .xlsx 파일을 생성합니다. 아래의 뱅커급 규칙을 따라 모델을 감사 가능하고, 유연하며, 작성자가 아닌 다른 사람도 검토할 수 있도록 만드세요.

Anthropic의 `xlsx-author` 및 `audit-xls` 스킬을 [anthropics/financial-services](https://github.com/anthropics/financial-services) 저장소에서 바탕으로 적용했습니다. 원본의 MCP / Office-JS / Cowork 관련 분기는 제거했으며, 이 스킬은 헤드리스 Python을 전제로 합니다.

## 출력 계약

- `./out/<name>.xlsx`에 기록합니다. `./out/`이 없으면 생성합니다.
- 최종 메시지에 후속 도구가 사용할 수 있도록 상대 경로를 반환합니다.
- 파일 하나당 논리적 모델 하나만 둡니다. 명시적으로 요청받지 않는 한 기존 워크북에 추가하지 않습니다.

## 설정

```bash
pip install "openpyxl>=3.0"
```

## 핵심 규칙 (협상 불가)

### 파란색 / 검은색 / 녹색 셀 색상
- **파란색** (`Font(color="0000FF")`) — 사람이 입력한 하드코딩 입력값입니다. 매출 동인, WACC 입력값, 터미널 성장률, 시장 데이터가 이에 해당합니다.
- **검은색** (기본값) — 수식입니다. 모든 파생 셀은 실제 Excel 수식이어야 합니다.
- **녹색** (`Font(color="006100")`) — 다른 시트나 외부 파일에 대한 링크입니다.

그러면 검토자가 시트를 훑어보면서 가정과 계산값을 즉시 구분할 수 있습니다.

### 하드코딩보다 수식
모든 계산 셀은 Python에서 계산해 값으로 붙여넣은 숫자가 아니라 반드시 수식 문자열이어야 합니다.

```python
# WRONG — silent bug waiting to happen
ws["D20"] = revenue_prior_year * (1 + growth)

# CORRECT — flexes when the user changes the assumption
ws["D20"] = "=D19*(1+$B$8)"
```

허용되는 하드코딩 숫자는 다음뿐입니다.
1. 원시 과거 입력값 (실제 매출, 공시 EBITDA 등)
2. 사용자가 조정하도록 의도된 가정 동인 (성장률, WACC 입력값, 터미널 g)
3. 현재 시장 데이터 (주가, 부채 잔액) — 출처와 날짜를 기록한 셀 주석이 있어야 합니다.

Python에서 값을 계산해 그 결과를 기록하고 있다면 멈추세요.

### 시트 간 참조를 위한 명명된 범위
다른 시트, 프레젠테이션 자료 또는 메모에서 참조하는 수치는 명명된 범위를 사용합니다.

```python
from openpyxl.workbook.defined_name import DefinedName
wb.defined_names["WACC"] = DefinedName("WACC", attr_text="Inputs!$C$8")
# then elsewhere:
calc["D30"] = "=D29/WACC"
```

### 검증 탭
모든 것을 연결하고 TRUE/FALSE를 표시하는 `Checks` 탭을 포함합니다.
- 대차대조표가 일치하는지 (자산 = 부채 + 자본)
- 현금흐름이 대차대조표의 기간 간 현금 변동과 일치하는지
- 합산부문이 연결 총계와 일치하는지
- 계산 범위 안에 잘못된 하드코딩이 없는지

예시:
```python
checks = wb.create_sheet("Checks")
checks["A2"] = "BS balances"
checks["B2"] = "=IS!D20-IS!D21-IS!D22"
checks["C2"] = "=ABS(B2)<0.01"  # TRUE/FALSE
```

### 모든 하드코딩 입력값에 셀 주석 추가
나중에 추가하지 말고 셀을 생성할 때 주석을 추가합니다.

```python
from openpyxl.comments import Comment
ws["C2"] = 1_250_000_000
ws["C2"].font = Font(color="0000FF")
ws["C2"].comment = Comment("Source: 10-K FY2024, p.47, revenue line", "analyst")
```

형식: `Source: [System/Document], [Date], [Reference], [URL if applicable]`.

출처 기록을 미루지 마세요. `TODO: add source`를 절대 작성하지 마세요.

## 기본 골격: 일반적인 금융 모델

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter
from pathlib import Path

BLUE = Font(color="0000FF")
BLACK = Font(color="000000")
GREEN = Font(color="006100")
BOLD = Font(bold=True)
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True)

wb = Workbook()

# --- Inputs tab ---
inp = wb.active
inp.title = "Inputs"
inp["A1"] = "MARKET DATA & KEY INPUTS"
inp["A1"].font = HEADER_FONT
inp["A1"].fill = HEADER_FILL
inp.merge_cells("A1:C1")

inp["B3"] = "Revenue FY2024"
inp["C3"] = 1_250_000_000
inp["C3"].font = BLUE
inp["C3"].comment = Comment("Source: 10-K FY2024 p.47", "model")

inp["B4"] = "Growth Rate"
inp["C4"] = 0.12
inp["C4"].font = BLUE

# --- Calc tab ---
calc = wb.create_sheet("DCF")
calc["B2"] = "Projected Revenue"
calc["C2"] = "=Inputs!C3*(1+Inputs!C4)"   # formula, black

# --- Checks tab ---
chk = wb.create_sheet("Checks")
chk["A2"] = "BS balances"
chk["B2"] = "=ABS(BS!D20-BS!D21-BS!D22)<0.01"

Path("./out").mkdir(exist_ok=True)
wb.save("./out/model.xlsx")
```

## 병합 셀을 사용한 섹션 헤더

openpyxl의 특이점: 병합할 때는 왼쪽 위 셀에 값을 설정하고 전체 범위에 스타일을 별도로 적용합니다.

```python
ws["A7"] = "CASH FLOW PROJECTION"
ws["A7"].font = HEADER_FONT
ws.merge_cells("A7:H7")
for col in range(1, 9):  # A..H
    ws.cell(row=7, column=col).fill = HEADER_FILL
```

## 민감도 표

하드코딩된 셀별 수식이 아니라 반복문으로 작성합니다. 규칙:

- **행/열 개수는 홀수** (5×5 또는 7×7) — 진정한 중앙 셀을 보장합니다.
- **중앙 셀 = 기준 사례.** 가운데 행/열 헤더는 모델의 실제 WACC 및 터미널 g와 같아야 하므로 중앙 결과가 기준 사례의 내재 주가와 같아야 합니다. 이것이 타당성 점검입니다.
- **중앙 셀을 중간 파란색 채우기** (`"BDD7EE"`)와 굵은 글꼴로 강조합니다.
- 모든 셀을 완전한 재계산 수식으로 채웁니다 — 절대 근사값을 사용하지 않습니다.

```python
# 5x5 WACC (rows) x terminal growth (cols) sensitivity
wacc_axis = [0.08, 0.085, 0.09, 0.095, 0.10]        # center row = base 9.0%
term_axis = [0.02, 0.025, 0.03, 0.035, 0.04]        # center col = base 3.0%

start_row = 40
ws.cell(row=start_row, column=1).value = "Implied Share Price ($)"
ws.cell(row=start_row, column=1).font = BOLD

for j, g in enumerate(term_axis):
    ws.cell(row=start_row+1, column=2+j).value = g
    ws.cell(row=start_row+1, column=2+j).font = BLUE

for i, w in enumerate(wacc_axis):
    r = start_row + 2 + i
    ws.cell(row=r, column=1).value = w
    ws.cell(row=r, column=1).font = BLUE
    for j, g in enumerate(term_axis):
        c = 2 + j
        # Full DCF recalc formula (simplified for illustration).
        # In a real model this references the full projection block.
        ws.cell(row=r, column=c).value = (
            f"=SUMPRODUCT(FCF_range,1/(1+{w})^year_offset) + "
            f"FCF_terminal*(1+{g})/({w}-{g})/(1+{w})^terminal_year"
        )

# Highlight center cell (base case)
center = ws.cell(row=start_row+2+len(wacc_axis)//2,
                 column=2+len(term_axis)//2)
center.fill = PatternFill("solid", fgColor="BDD7EE")
center.font = BOLD
```

## 제공 전 재계산

openpyxl은 수식 문자열을 기록하지만 계산하지는 않습니다. Excel은 열 때 다시 계산하지만, 후속 소비자 (자동 검사 스크립트, CI)에게는 계산된 값이 필요합니다.

전달 전에 LibreOffice 또는 전용 재계산 단계를 실행합니다.

```bash
# LibreOffice headless recalc
libreoffice --headless --calc --convert-to xlsx ./out/model.xlsx --outdir ./out/
```

또는 Python 재계산 도우미를 사용합니다 (이 스킬의 `scripts/recalc.py` 참고).

## 모델 레이아웃 계획

수식을 작성하기 전에:
1. 모든 섹션의 행 위치를 정의합니다.
2. 모든 헤더와 레이블을 작성합니다.
3. 모든 섹션 구분선과 빈 행을 작성합니다.
4. **그 다음에** 고정된 행 위치를 사용해 수식을 작성합니다.

이렇게 하면 수식을 작성한 뒤 헤더 행을 삽입해 모든 하위 참조가 연쇄적으로 깨지는 패턴을 방지할 수 있습니다.

## 사용자와 단계별로 검증

대형 모델 (DCF, 3-statement, LBO)의 경우 계속 진행하기 전에 멈추고 사용자에게 중간 산출물을 보여줍니다. 잘못된 마진 가정을 하위 민감도 표까지 만든 뒤 발견하면 한 시간을 절약할 수 없습니다.

체크포인트 패턴:
- Inputs 블록 이후 → 원시 입력값을 보여주고 투영 전에 확인
- Revenue projections 이후 → 톱라인과 성장률 확인
- FCF build 이후 → 전체 스케줄 확인
- WACC 이후 → 입력값 확인
- valuation 이후 → 자기자본 브리지 확인
- **그 다음에** 민감도 표 작성

## 이 스킬을 사용하지 않는 경우

- Office MCP를 사용할 수 있는 실시간 Excel 세션의 사용자 — 대신 실시간 워크북을 조작합니다.
- 수식이 없는 순수 표 형식 데이터 내보내기 — `csv` 또는 `pandas.to_excel`이 더 간단합니다.
- 상호작용이 많은 대시보드 / 차트 — 실제 BI 도구를 사용합니다.

## 출처
