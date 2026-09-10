---
title: "Xlsx — Excel .xlsx 통합 문서와 CSV 생성, 읽기, 편집"
sidebar_label: "Xlsx"
description: "Excel .xlsx 통합 문서와 CSV를 생성하고 읽고 편집합니다"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Xlsx

Excel .xlsx 통합 문서와 CSV를 생성하고 읽고 편집합니다.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 기본 포함 (기본으로 설치됨) |
| 경로 | `skills/productivity/xlsx` |
| 버전 | `1.0.0` |
| 작성자 | Nous Research |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `excel`, `spreadsheet`, `xlsx`, `csv`, `openpyxl`, `productivity` |
| 관련 skill | [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx), [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf), [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 활성화될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성 상태일 때 에이전트가 보는 지침이기도 합니다.
:::

# Xlsx Skill

Python과 openpyxl을 사용해 Excel .xlsx 통합 문서를 다룹니다. 수식과 차트가 포함된 스타일 지정 다중 시트 통합 문서를 만들고, 기존 파일을 검사하거나 덤프하고, 셀과 구조를 편집하며, CSV로 변환하거나 CSV에서 변환할 수 있습니다. 모든 도우미 스크립트는 argparse CLI이며 JSON을 출력하고 명시적인 UTF-8 I/O를 사용합니다.

## 사용 시점

- .xlsx 보고서 생성: 여러 시트, 숫자 형식, 스타일 지정, 셀 병합, 틀 고정, 자동 필터, 조건부 서식, 차트, 데이터 유효성 검사 드롭다운.
- 통합 문서 읽기: 시트 목록, 데이터를 JSON 또는 CSV로 덤프, 수식과 캐시된 값 구분해 나열.
- 기존 파일 편집: 셀 설정, 행 추가, 행/열 삽입 및 삭제, 시트 복사/이름 변경.
- 형식 추론과 비 UTF-8 인코딩을 지원하는 CSV 상호 운용.
- 레거시 `.xls` 바이너리 형식에는 사용하지 않습니다 (먼저 LibreOffice로 변환: `soffice --headless --convert-to xlsx old.xls`).

## 사전 요구 사항

- Python 3.10+ 및 `openpyxl` (`pip install openpyxl`). 그 외에는 서드파티 패키지가 필요하지 않으며 나머지는 모두 표준 라이브러리입니다.
- 선택 사항: 헤드리스 재계산 또는 형식 변환을 위한 LibreOffice (`soffice`).

## 실행 방법

이 skill의 `scripts/` 디렉터리에서 `terminal` 도구로 도우미 스크립트를 실행합니다 (모든 스크립트가 `--help`를 지원합니다).

```bash
python scripts/xlsx_create.py spec.json report.xlsx   # build from JSON spec
python scripts/xlsx_read.py report.xlsx --sheets      # inventory
python scripts/xlsx_read.py report.xlsx --json --sheet Data
python scripts/xlsx_read.py report.xlsx --formulas
python scripts/xlsx_edit.py report.xlsx --sheet Data --set B2=42 --recalc
python scripts/csv_to_xlsx.py data.csv out.xlsx --encoding utf-8
python scripts/xlsx_to_csv.py report.xlsx out.csv --sheet Data
```

JSON 사양은 `write_file`로 작성하고, 스크립트의 JSON 출력은 `read_file` 또는 표준 출력에서 직접 확인합니다.

## 빠른 참조

| 작업 | 명령 |
|---|---|
| 사양에서 통합 문서 생성 | `xlsx_create.py spec.json out.xlsx` |
| 시트 이름 + 크기 | `xlsx_read.py f.xlsx --sheets` |
| 시트를 JSON으로 덤프 | `xlsx_read.py f.xlsx --json --sheet S` |
| 시트를 CSV로 덤프 | `xlsx_read.py f.xlsx --csv --out d.csv` |
| 수식 + 캐시된 값 나열 | `xlsx_read.py f.xlsx --formulas` |
| 셀 / 수식 설정 | `xlsx_edit.py f.xlsx --set "A1==SUM(B:B)"` |
| 행 추가 | `xlsx_edit.py f.xlsx --append '[1,"x",true]'` |
| 3행 앞에 2개 행 삽입 | `xlsx_edit.py f.xlsx --insert-rows 3:2` |
| 시트 복사 / 이름 변경 | `--copy-sheet Src:New --rename-sheet Old:New` |
| 열 때 강제 재계산 | `xlsx_edit.py f.xlsx --recalc` |
| CSV -> 스타일 지정 xlsx | `csv_to_xlsx.py in.csv out.xlsx` |
| xlsx -> CSV | `xlsx_to_csv.py f.xlsx out.csv --encoding utf-8` |

## 절차

1. **생성**: JSON 사양을 작성합니다 (스키마는 `xlsx_create.py --help`와 docstring에 설명되어 있습니다). 각 시트는 `rows` (스칼라 또는 스타일 지정 셀 객체), 희소 `cells` 재정의, `column_widths`, `row_heights`, `merges`, `freeze_panes`, `autofilter`, `conditional_formats` (cell_is 규칙 및 색상 스케일), `charts` (셀 범위에서 막대/선/원형 차트), `validations` (목록 드롭다운)를 지원합니다. 형식이 지정된 값: JSON 숫자/불리언은 그대로 전달하고, 날짜는 `{"value": "2026-01-31", "type": "date"}`를 사용합니다. 숫자 형식은 Excel 형식 문자열입니다: 통화 `"$#,##0.00"`, 백분율 `"0.0%"`, 날짜 `"yyyy-mm-dd"`.
2. **수식**: 사양에서 `"formula": "SUM(B2:B9)"`로 설정하거나 편집기에서 `--set "C1==SUM(A:A)"`를 사용합니다. 수식을 작성할 때는 `"full_calc_on_load": true` (사양) 또는 `--recalc` (편집기)를 추가합니다. 이렇게 하면 통합 문서의 `fullCalcOnLoad` 플래그가 설정되어 Excel/LibreOffice가 열 때 모든 항목을 다시 계산합니다. openpyxl 자체는 수식을 절대 평가하지 않습니다.
3. **읽기**: 목록 확인에는 `--sheets` (이름, 크기, 병합 범위, 차트 수), 데이터에는 `--json`/`--csv`, 각 수식 문자열과 캐시된 결과를 짝지을 때는 `--formulas`를 사용합니다. 캐시된 결과는 실제 스프레드시트 앱에서 마지막으로 저장한 경우에만 존재하며, openpyxl에서 새로 저장한 파일은 여기서 `null`을 반환합니다. 헤드리스로 결과를 구체화하려면 `soffice --headless --convert-to xlsx file.xlsx`를 실행한 다음 `--data-only`로 다시 로드합니다.
4. **편집**: `xlsx_edit.py`는 먼저 이름 변경/복사를 적용하고, 다음으로 구조적인 행/열 변경을 적용한 뒤 `--set`/`--append`를 적용합니다. `--out`을 지정하지 않으면 현재 위치에서 편집하므로 원본이 필요하다면 먼저 파일을 복사합니다.
5. **CSV 상호 운용**: `csv_to_xlsx.py`는 셀마다 int/float/bool/ISO 날짜를 추론하고 헤더 행에 스타일을 적용합니다. `xlsx_to_csv.py`는 ISO 날짜와 빈 셀에 빈 문자열을 기록합니다. 두 스크립트 모두 기본값은 UTF-8이며 `--encoding`을 허용합니다 (예: Excel에서 사용하기 좋은 BOM에는 `utf-8-sig`, 레거시 Windows 내보내기에는 `cp1252`).

## 주의 사항

- **openpyxl은 계산하지 않습니다.** 수식 결과는 `load_workbook(path, data_only=True)`를 통해서만, 그리고 파일이 이전에 Excel/LibreOffice로 저장된 경우에만 사용할 수 있습니다. 그렇지 않으면 `None`을 받습니다.
- **삽입/삭제는 참조를 이동하지 않습니다.** `insert_rows`, `delete_cols` 등은 셀 값을 이동하지만 병합 셀 범위, 수식 참조, 차트 앵커 또는 조건부 서식 범위를 업데이트하지 않습니다. 병합이나 수식이 있는 시트에서 구조를 편집한 후에는 `--sheets` 및 `--formulas`로 다시 확인하고 수동으로 수정합니다.
- **`data_only=True`로 로드한 뒤 저장하면** 모든 수식이 조용히 삭제됩니다 (캐시된 값이 수식을 대체합니다). 이것이 목적이 아니라면 해당 방식으로 로드한 통합 문서를 절대 저장하지 마세요.
- **로드하면 차트/이미지가 제거됩니다**: openpyxl은 차트를 왕복 처리하지 않으므로 차트가 있는 통합 문서를 편집하고 저장하면 차트가 사라집니다. 편집 후 차트를 다시 추가하거나 차트 파일을 다시 저장하지 마세요.
- **CSV 로캘 주의**: 항상 명시적인 인코딩을 전달하고 (스크립트는 이미 그렇게 합니다), 유럽 CSV는 `;` 구분자와 소수점 쉼표를 자주 사용한다는 점을 기억하세요 — `--delimiter ';'`를 사용하고 `"12,5"` 같은 값이 문자열로 유지될 것을 예상합니다.
- **날짜는 datetime입니다**: Excel은 날짜를 일련번호로 저장하고, openpyxl은 `datetime`/`date` 객체를 반환합니다. 이 덤프에서는 ISO 문자열을 출력합니다.
- 시트 이름은 31자로 제한되며 `[ ] : * ? / \`를 사용할 수 없습니다.

## 검증

- 생성 후: `xlsx_read.py out.xlsx --sheets`를 실행하고 시트 이름, 크기, 병합 범위, 차트 수가 의도와 일치하는지 확인합니다.
- `--json`으로 데이터를 덤프하고 원본 값과 비교합니다.
- 편집 후: 변경한 범위를 다시 덤프하고, 수식을 작성했다면 `--formulas`에 수식이 나열되는지와 `--recalc`가 적용되었는지 확인합니다.
- 전체적인 시각 검사를 하려면 LibreOffice에서 엽니다:
  `soffice --headless --convert-to pdf out.xlsx`를 실행하고 PDF를 검사합니다.
