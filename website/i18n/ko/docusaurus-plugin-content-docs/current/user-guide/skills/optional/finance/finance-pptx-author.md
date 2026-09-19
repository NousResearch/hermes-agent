---
title: "Pptx 작성 — python-pptx로 PowerPoint 데크를 헤드리스 방식으로 만들기"
sidebar_label: "Pptx 작성"
description: "python-pptx로 PowerPoint 데크를 헤드리스 방식으로 만들기"
---

{/* 이 페이지는 스킬의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Pptx 작성

python-pptx로 PowerPoint 데크를 헤드리스 방식으로 만듭니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/finance/pptx-author`로 설치 |
| 경로 | `optional-skills/finance/pptx-author` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (Nous Research가 수정) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `powerpoint`, `pptx`, `python-pptx`, `presentation`, `finance` |
| 관련 스킬 | [`excel-author`](/docs/user-guide/skills/optional/finance/finance-excel-author), [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되어 있을 때 에이전트가 보는 지침이 바로 이것입니다.
:::

# pptx-author

`python-pptx`를 사용해 디스크에 .pptx 파일을 생성합니다. 실시간 PowerPoint 세션을 조작하는 것이 아니라 파일 아티팩트로 데크를 전달해야 할 때 사용하세요.

Anthropic의 `pptx-author` 및 `pitch-deck` 스킬을 바탕으로 하며, [anthropics/financial-services](https://github.com/anthropics/financial-services)에서 가져왔습니다. 원본의 MCP / Office-JS 분기는 제거했습니다. 이 스킬은 헤드리스 Python을 전제로 합니다.

더 폭넓고 이미 제공되는 PowerPoint 작성 스킬 (슬라이드, 발표자 노트, 임베드, 미디어)은 내장 `powerpoint` 스킬을 참고하세요. 이 스킬은 모델이 생성하는 데크 (피치 데크, IC 메모, 실적 노트)에 맞춘 더 가벼운 패턴으로, 모든 수치가 원본 워크북까지 추적되도록 합니다.

## 출력 계약

- `./out/<name>.pptx`에 기록하세요. 존재하지 않으면 `./out/`을 생성하세요.
- 최종 메시지에 상대 경로를 반환하세요.

## 설정

```bash
pip install "python-pptx>=0.6"
```

## 핵심 규칙

### 슬라이드 하나에 아이디어 하나

제목은 핵심 메시지를 말하고 본문은 이를 뒷받침해야 합니다. "Q3 Revenue"라는 제목의 슬라이드는 약합니다. "Revenue growth accelerated to 14% Y/Y in Q3"가 강한 제목입니다.

### 모든 수치는 모델까지 추적 가능해야 함

슬라이드의 수치가 `./out/model.xlsx`에서 나온 것이라면 시트와 셀을 각주로 표시하세요.

```
Revenue: $1,250M  (Source: model.xlsx, Inputs!C3)
```

수치를 기억이나 요약에서 옮겨 적지 마세요. 워크북을 열고 명명된 범위를 읽은 다음, 가능하면 프로그래밍 방식으로 데크의 값에 연결하세요.

### 탑재된 회사 템플릿 사용

`./templates/firm-template.pptx`가 있으면 이를 로드하여 데크가 회사의 브랜드 색상, 글꼴, 마스터 레이아웃을 상속하도록 하세요.

```python
from pptx import Presentation
from pathlib import Path

template = Path("./templates/firm-template.pptx")
prs = Presentation(str(template)) if template.exists() else Presentation()
```

### 차트: 기본 pptx 차트보다 모델에서 PNG로 생성

충실도가 중요할 때 (모델의 차트 스타일이 데크와 정확히 일치해야 할 때)는 원본 워크북에서 차트를 PNG로 렌더링하고 이미지를 임베드하세요. 기본 `pptx.chart` 차트는 불안정하며 회사 규칙과 일치하지 않는 경우가 많습니다.

```python
from pptx.util import Inches
slide.shapes.add_picture("./out/charts/football_field.png",
                         Inches(1), Inches(2),
                         width=Inches(8))
```

### 외부 전송 금지

이 스킬은 파일을 기록합니다. 이메일을 보내거나 업로드하거나 게시하지 않습니다. 전달은 오케스트레이션 계층이 처리합니다.

## 뼈대

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pathlib import Path

template = Path("./templates/firm-template.pptx")
prs = Presentation(str(template)) if template.exists() else Presentation()

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Project Aurora — Strategic Alternatives"
slide.placeholders[1].text = "Preliminary Discussion Materials"

# Valuation summary slide (title-only layout)
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Valuation implies $38–$52 per share across methodologies"

# Add a table bound to model outputs
rows, cols = 5, 4
tbl_shape = slide.shapes.add_table(rows, cols,
                                   Inches(0.5), Inches(1.5),
                                   Inches(9), Inches(3))
tbl = tbl_shape.table
headers = ["Methodology", "Low ($)", "Mid ($)", "High ($)"]
for c, h in enumerate(headers):
    tbl.cell(0, c).text = h

# In a real deck, read these from the model workbook with openpyxl
data = [
    ("Trading comps",     "35", "41", "48"),
    ("Precedent M&A",     "39", "45", "52"),
    ("DCF (base)",        "36", "43", "51"),
    ("LBO (10% IRR)",     "33", "38", "44"),
]
for r, row in enumerate(data, start=1):
    for c, val in enumerate(row):
        tbl.cell(r, c).text = val

# Embed a chart rendered from the model
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Football field — current price $42"
slide.shapes.add_picture("./out/charts/football_field.png",
                         Inches(1), Inches(1.8), width=Inches(8))

Path("./out").mkdir(exist_ok=True)
prs.save("./out/pitch-aurora.pptx")
```

## 데크 수치를 원본 워크북에 연결하기

Excel 모델에서 명명된 범위 또는 특정 셀을 읽어 데크의 수치가 절대 어긋나지 않도록 하세요.

```python
from openpyxl import load_workbook

wb = load_workbook("./out/model.xlsx", data_only=True)
def nr(name):
    """Resolve a named range to its current computed value."""
    rng = wb.defined_names[name]
    sheet, coord = next(rng.destinations)
    return wb[sheet][coord].value

revenue_fy24 = nr("RevenueFY24")
implied_mid  = nr("ImpliedSharePriceBase")
```

그런 다음 이 값들을 사용해 데크 콘텐츠를 만드세요:
```python
slide.shapes.title.text = f"Implied share price of ${implied_mid:.2f} (base case)"
```

워크북에서 읽기 전에 다시 계산해야 한다는 점을 기억하세요. openpyxl은 시트가 이미 계산된 경우에만 계산된 값을 볼 수 있습니다. 먼저 `excel-author` 스킬의 재계산 도우미를 실행하거나 실제 Excel 세션을 통해 열고 저장하세요.

## 피치 데크용 슬라이드 유형 체크리스트

일반적인 투자은행 피치 데크는 다음 구조를 따릅니다. 반드시 따라야 하는 것은 아니지만, 시작용 뼈대로 유용합니다:

1. 표지 / 제목
2. 면책 조항
3. 목차
4. 상황 개요
5. 회사 개요 (대상 회사)
6. 시장 / 섹터 맥락
7. 가치평가 요약 (풋볼 필드) — 핵심 슬라이드
8. 거래 비교 기업 상세
9. 선행 거래 상세
10. DCF 요약
11. 예시 LBO / 스폰서 사례
12. 프로세스 고려 사항
13. 부록

## 이 스킬을 사용하지 않는 경우

- Office MCP를 사용할 수 있는 실시간 PowerPoint 세션의 사용자 — 대신 실시간 문서를 조작하세요.
- 비재무 슬라이드 자료 (분기 전체 회의, 마케팅 데크) — 더 폭넓은 `powerpoint` 스킬을 사용하세요.
- 애니메이션, 전환 효과 또는 발표자 노트가 많은 데크 — 더 폭넓은 `powerpoint` 스킬을 사용하세요.

## 출처 표시

Anthropic의 Claude for Financial Services 플러그인 제품군의 규칙을 바탕으로 수정했으며 Apache-2.0 라이선스를 따릅니다. 원본: https://github.com/anthropics/financial-services/tree/main/plugins/agent-plugins/pitch-agent/skills/pptx-author
