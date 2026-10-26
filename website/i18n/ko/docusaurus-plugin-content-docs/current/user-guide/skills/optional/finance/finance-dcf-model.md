---
title: "Dcf Model — Excel에서 할인현금흐름 가치평가 워크북 구축"
sidebar_label: "Dcf Model"
description: "Excel에서 할인현금흐름 가치평가 워크북 구축"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dcf Model

Excel에서 할인현금흐름 가치평가 워크북을 구축합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/finance/dcf-model`로 설치 |
| 경로 | `optional-skills/finance/dcf-model` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (Nous Research에서 각색) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `finance`, `valuation`, `dcf`, `excel`, `openpyxl`, `modeling`, `investment-banking` |
| 관련 스킬 | [`excel-author`](/docs/user-guide/skills/optional/finance/finance-excel-author), [`pptx-author`](/docs/user-guide/skills/optional/finance/finance-pptx-author), [`comps-analysis`](/docs/user-guide/skills/optional/finance/finance-comps-analysis), [`lbo-model`](/docs/user-guide/skills/optional/finance/finance-lbo-model), [`3-statement-model`](/docs/user-guide/skills/optional/finance/finance-3-statement-model) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 불러오는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 확인하는 내용입니다.
:::

## 환경

이 스킬은 **헤드리스 openpyxl**을 전제로 합니다 — 디스크에 .xlsx 파일을 생성합니다.
셀 색상, 수식, 명명된 범위, 민감도 표에 대해서는 `excel-author` 스킬의 규칙을 따릅니다.
전달 전에 다시 계산합니다: `python /path/to/excel-author/scripts/recalc.py ./out/model.xlsx`.

# DCF 모델 빌더

## 개요

이 스킬은 투자은행 기준에 따라 주식 가치평가를 위한 기관급 DCF 모델을 생성합니다. 각 분석은 상세한 Excel 모델을 생성하며(DCF 시트 하단에 민감도 분석 포함), 다음과 같은 구성으로 이루어집니다.

## 도구

- 데이터 소싱을 위해 사용자가 제공한 모든 정보와 사용 가능한 MCP 서버를 기본적으로 모두 활용합니다.

## 핵심 제약 조건 — 먼저 읽을 것

이 제약 조건은 모든 DCF 모델 구축 과정에 적용됩니다. 시작하기 전에 검토하세요.

**하드코딩보다 수식 우선(타협 불가):**
- 모든 전망치, 마진, 할인 계수, 현재가치, 민감도 셀은 반드시 실제 Excel 수식이어야 합니다 — Python에서 계산한 값을 숫자로 작성해서는 안 됩니다.
- openpyxl 사용 시: `ws["D20"] = "=D19*(1+$B$8)"`은 올바릅니다. `ws["D20"] = calculated_revenue`는 잘못된 방식입니다.
- 하드코딩이 허용되는 숫자는 다음뿐입니다: (1) 과거 원시 입력값, (2) 가정 드라이버(성장률, WACC 입력값, 터미널 g), (3) 현재 시장 데이터(주가, 부채 잔액).
- Python에서 무언가를 계산해 그 결과를 작성하고 있다면 — 멈추세요. 사용자가 가정을 변경할 때 모델이 동적으로 변해야 합니다.

**사용자와 단계별로 검증(엔드투엔드로 구축하지 말 것):**
- 데이터 조회 후 → 과거 입력 블록(매출, 마진, 주식 수, 순부채)을 사용자에게 보여주고 전망을 시작하기 전에 확인받습니다.
- 매출 전망 후 → 전망된 매출과 성장률을 보여주고 마진 구축 전에 확인받습니다.
- FCF 구축 후 → 전체 FCF 일정을 보여주고 WACC를 계산하기 전에 로직을 확인받습니다.
- WACC 산출 후 → 할인하기 전에 계산 결과와 입력값을 보여줍니다.
- 터미널 가치 + 현재가치 산출 후 → 민감도 표를 만들기 전에 자기자본 브리지를 보여줍니다(EV → 자기자본 가치 → 주당 가치).
- 각 단계에서 오류를 포착합니다 — 민감도 표를 만든 후 잘못된 마진 가정을 발견하면 이후 모든 작업을 다시 구축해야 합니다.

**민감도 표:**
- **행과 열은 홀수 개를 사용합니다**(표준: 5×5, 경우에 따라 7×7) — 이렇게 해야 진정한 중앙 셀이 보장됩니다.
- **중앙 셀 = 기준 사례.** 축 값을 구성할 때 가운데 행 머리글과 가운데 열 머리글이 모델의 실제 가정과 정확히 일치하도록 합니다(예: 기준 WACC = 9.0%라면 가운데 행은 9.0%, 터미널 g = 3.0%라면 가운데 열은 3.0%). 따라서 중앙 셀의 출력값은 모델의 실제 내재 주당 가치와 같아야 합니다 — 이것이 표가 올바르게 구축되었는지 확인하는 검증입니다.
- **중앙 셀을** 중간 파란색 채우기(`#BDD7EE`)와 굵은 글꼴로 강조해 기준 사례가 어느 셀인지 즉시 보이게 합니다.
- 모든 셀(일반적으로 표 3개 × 셀 25개 = 75개)에 완전한 DCF 재계산 수식을 입력합니다.
- openpyxl 반복문을 사용해 수식을 프로그래밍 방식으로 작성합니다.
- 자리 표시자 텍스트 금지, 선형 근사 금지, 수동 단계 불필요.
- 각 셀은 해당 가정 조합에 대해 전체 DCF를 재계산해야 합니다.

**셀 주석:**
- 하드코딩된 값을 생성할 때마다 셀 주석을 추가합니다.
- 형식: "출처: [시스템/문서], [날짜], [참조], [해당하는 경우 URL]"
- 다음 섹션으로 넘어가기 전에 모든 파란색 입력값에 주석이 있어야 합니다.
- 마지막까지 미루거나 "TODO: 출처 추가"라고 작성하지 않습니다.

**모델 레이아웃 계획:**
- 수식을 작성하기 전에 모든 섹션의 행 위치를 정의합니다.
- 모든 머리글과 레이블을 먼저 작성합니다.
- 모든 섹션 구분선과 빈 행을 두 번째로 작성합니다.
- 그런 다음 고정된 행 위치를 사용해 수식을 작성합니다.
- 생성 직후 수식을 테스트합니다.

**수식 재계산:**
- 전달 전에 `python recalc.py model.xlsx 30`을 실행합니다.
- 상태가 "success"가 될 때까지 모든 오류를 수정합니다.
- 수식 오류(`#REF!`, `#DIV/0!`, `#VALUE!` 등)는 하나도 없어야 합니다.

**시나리오 블록:**
- Bear/Base/Bull 사례를 별도의 블록으로 만듭니다.
- 각 블록에서 전망 연도에 걸쳐 가정을 가로 방향으로 표시합니다.
- IF 수식을 사용합니다: `=IF($B$6=1,[Bear cell],IF($B$6=2,[Base cell],[Bull cell]))`
- 수식이 올바른 시나리오 블록 셀을 참조하는지 확인합니다.

## DCF 프로세스 워크플로

### 1단계: 데이터 조회 및 검증

MCP 서버, 사용자가 제공한 데이터, 웹에서 데이터를 가져옵니다.

**데이터 출처 우선순위:**
1. **MCP 서버**(구성된 경우) — Daloopa와 같은 제공업체의 구조화된 금융 데이터
2. **사용자 제공 데이터** — 자체 조사에서 얻은 과거 재무 데이터
3. **웹 검색/가져오기** — 필요한 경우 현재 주가, 베타, 부채, 현금

**검증 체크리스트:**
- 순부채와 순현금을 확인합니다(가치평가에 중요).
- 희석 주식 수를 확인합니다(최근 자사주 매입/신주 발행 여부 점검).
- 과거 마진이 비즈니스 모델과 일관되는지 검증합니다.
- 매출 성장률을 업계 벤치마크와 교차 검증합니다.
- 세율이 합리적인지 확인합니다(일반적으로 21~28%).

### 2단계: 과거 분석(3~5년)

다음을 분석하고 문서화합니다.
- **매출 성장 추세**: CAGR을 계산하고 성장 동인을 식별합니다.
- **마진 변화**: 매출총이익률, EBIT 마진, FCF 마진을 추적합니다.
- **자본 집약도**: 매출 대비 D&A 및 CapEx 비율
- **운전자본 효율성**: 매출 성장 대비 NWC 변동
- **수익률 지표**: ROIC, ROE 추세

다음을 보여주는 요약 표를 만듭니다.
```
Historical Metrics (LTM):
Revenue: $X million
Revenue growth: X% CAGR
Gross margin: X%
EBIT margin: X%
D&A % of revenue: X%
CapEx % of revenue: X%
FCF margin: X%
```

### 3단계: 매출 전망 구축

**방법론:**
1. 최신 실제 매출(LTM 또는 가장 최근 회계연도)에서 시작합니다.
2. 각 전망 연도에 성장률을 적용합니다.
3. 달러 금액과 계산된 성장률을 모두 표시합니다.

**성장률 프레임워크:**
- 1~2년 차: 단기 가시성을 반영해 더 높은 성장률
- 3~4년 차: 업계 평균을 향해 점진적으로 둔화
- 5년 차 이후: 터미널 성장률에 근접

**수식 구조:**
- Revenue(Year N) = Revenue(Year N-1) × (1 + Growth Rate)
- Growth %(Year N) = Revenue(Year N) / Revenue(Year N-1) - 1

**3가지 시나리오 접근법:**
```
Bear Case: Conservative growth (e.g., 8-12%)
Base Case: Most likely scenario (e.g., 12-16%)
Bull Case: Optimistic growth (e.g., 16-20%)
```

### 4단계: 영업비용 모델링

**고정비/변동비 분석:**

영업비용은 현실적인 영업 레버리지를 반영해 모델링해야 합니다.
- **영업 및 마케팅**: 비즈니스 모델에 따라 일반적으로 매출의 15~40%
- **연구개발**: 기술 기업의 경우 일반적으로 10~30%
- **일반관리비**: 일반적으로 매출의 8~15%이며, 기업 규모 확대에 따른 레버리지를 보여줍니다.

**핵심 원칙:**
- 모든 비율은 매출총이익이 아니라 매출을 기준으로 합니다.
- 영업 레버리지를 모델링합니다: 매출이 증가함에 따라 비율이 낮아져야 합니다.
- S&M, R&D, G&A를 별도 항목으로 유지합니다.
- EBIT = 매출총이익 - 총 영업비용으로 계산합니다.

**마진 확대 프레임워크:**
```
Current State → Target State (Year 5)
Gross Margin: X% → Y% (justify based on scale, efficiency)
EBIT Margin: X% → Y% (result of revenue growth + opex leverage)
```

### 5단계: 잉여현금흐름 계산

**올바른 순서로 FCF 구축:**

```
EBIT
(-) Taxes (EBIT × Tax Rate)
= NOPAT (Net Operating Profit After Tax)
(+) D&A (non-cash expense, % of revenue)
(-) CapEx (% of revenue, typically 4-8%)
(-) Δ NWC (change in working capital)
= Unlevered Free Cash Flow
```

**운전자본 모델링:**
- 매출 변동(매출 증감액) 대비 비율로 계산합니다.
- 일반적인 범위: 매출 증감액의 -2%~+2%
- 음수 = 현금 원천(운전자본 회수)
- 양수 = 현금 사용(운전자본 구축)

**유지보수 CapEx와 성장 CapEx:**
- 유지보수 CapEx: 현재 운영을 유지(~매출의 2~3%)
- 성장 CapEx: 확장을 지원(매출의 추가 2~5%)
- 총 CapEx는 회사의 성장 전략과 일치해야 합니다.

### 6단계: 자본비용(WACC) 조사

**자기자본비용의 CAPM 방법론:**

```
Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium

Where:
- Risk-Free Rate = Current 10-Year Treasury Yield
- Beta = 5-year monthly stock beta vs market index
- Equity Risk Premium = 5.0-6.0% (market standard)
```

**부채비용 계산:**

```
After-Tax Cost of Debt = Pre-Tax Cost of Debt × (1 - Tax Rate)

Determine Pre-Tax Cost of Debt from:
- Credit rating (if available)
- Current yield on company bonds
- Interest expense / Total Debt from financials
```

**자본구조 가중치:**

```
Market Value Equity = Current Stock Price × Shares Outstanding
Net Debt = Total Debt - Cash & Equivalents
Enterprise Value = Market Cap + Net Debt

Equity Weight = Market Cap / Enterprise Value
Debt Weight = Net Debt / Enterprise Value

WACC = (Cost of Equity × Equity Weight) + (After-Tax Cost of Debt × Debt Weight)
```

**특수 사례:**
- **순현금 포지션**: 현금 > 부채인 경우 순부채는 음수입니다.
  - 부채 가중치는 음수가 될 수 있습니다.
  - 그에 따라 WACC 계산이 조정됩니다.
- **부채 없음**: WACC = 자기자본비용

**일반적인 WACC 범위:**
- 대형주, 안정적 기업: 7~9%
- 성장 기업: 9~12%
- 고성장/고위험: 12~15%

### 7단계: 할인율 적용(5~10년 전망)

**중간 연도 관례:**
- 현금흐름은 연도 중간에 발생한다고 가정합니다.
- 할인 기간: 0.5, 1.5, 2.5, 3.5, 4.5 등
- 할인 계수 = 1 / (1 + WACC)^기간

**현재가치 계산:**
```
For each projection year:
PV of FCF = Unlevered FCF × Discount Factor

Example (Year 1):
FCF = $1,000
WACC = 10%
Period = 0.5
Discount Factor = 1 / (1.10)^0.5 = 0.9535
PV = $1,000 × 0.9535 = $954
```

**전망 기간 선택:**
- **5년**: 대부분의 분석에서 표준
- **7~10년**: 더 긴 성장 여력을 가진 고성장 기업
- **3년**: 성숙하고 안정적인 기업

### 8단계: 터미널 가치 계산

**영구성장법(권장):**

```
Terminal FCF = Final Year FCF × (1 + Terminal Growth Rate)
Terminal Value = Terminal FCF / (WACC - Terminal Growth Rate)

Critical Constraint: Terminal Growth < WACC (otherwise infinite value)
```

**터미널 성장률 선택:**
- 보수적: 2.0~2.5%(GDP 성장률)
- 중간: 2.5~3.5%
- 공격적: 3.5~5.0%(시장 선도 기업에만 적용)

**초과하지 말아야 할 기준**: 무위험 수익률 또는 장기 GDP 성장률

**Exit Multiple 방식(대안):**
```
Terminal Value = Final Year EBITDA × Exit Multiple

Where Exit Multiple comes from:
- Industry comparable trading multiples
- Precedent transaction multiples
- Typical range: 8-15x EBITDA
```

**터미널 가치의 현재가치:**
```
PV of Terminal Value = Terminal Value / (1 + WACC)^Final Period

Where Final Period accounts for timing:
5-year model with mid-year convention: Period = 4.5
```

**터미널 가치 타당성 점검:**
- 기업가치의 50~70%를 차지해야 합니다.
- 75%를 초과하면 모델이 터미널 가정에 지나치게 의존할 수 있습니다.
- 40% 미만이면 터미널 가정이 지나치게 보수적인지 확인합니다.
### 9단계: 기업가치에서 주주가치로의 브리지

**가치평가 요약 구조:**

```
(+) Sum of PV of Projected FCFs = $X million
(+) PV of Terminal Value = $Y million
= Enterprise Value = $Z million

(-) Net Debt [or + Net Cash if negative] = $A million
= Equity Value = $B million

÷ Diluted Shares Outstanding = C million shares
= Implied Price per Share = $XX.XX

Current Stock Price = $YY.YY
Implied Return = (Implied Price / Current Price) - 1 = XX%
```

**주요 조정 항목:**
- **순부채 = 총부채 - 현금 및 현금성 자산**
  - 양수인 경우: EV에서 차감 (주주가치 감소)
  - 음수인 경우 (순현금): EV에 가산 (주주가치 증가)
- **희석 후 발행주식수 사용**: 옵션, RSU, 전환증권 포함
- **기타 조정 항목** (해당하는 경우):
  - 비지배지분
  - 연금 부채
  - 운용리스 의무

**가치평가 출력 형식:**
```csv
Valuation Component,Amount ($M)
PV Explicit FCFs,X.X
PV Terminal Value,Y.Y
Enterprise Value,Z.Z
(-) Net Debt,A.A
Equity Value,B.B
,,
Shares Outstanding (M),C.C
Implied Price per Share,$XX.XX
Current Share Price,$YY.YY
Implied Upside/(Downside),+XX%
```

### 10단계: 민감도 분석

DCF 시트 하단에 서로 다른 가정에 따라 가치평가가 어떻게 변하는지 보여주는 **민감도 표 3개**를 작성합니다.

1. **WACC 대 터미널 성장률** - 할인율과 영구 성장률에 따른 기업가치 민감도 표시
2. **매출 성장률 대 EBIT 마진** - 외형 성장과 영업 레버리지의 영향 표시
3. **베타 대 무위험이자율** - 자기자본비용 구성요소에 대한 민감도 표시

**구현**: 각 셀에 수식을 넣은 단순한 2차원 그리드이며, Excel의 "데이터 표(Data Table)" 기능은 사용하지 않습니다. 각 셀에는 해당 가정 조합에 대한 전체 DCF 재계산식이 들어 있어야 합니다. openpyxl을 사용해 75개 셀 전체를 프로그래밍 방식으로 채우는 상세 요구사항은 주요 제약사항 섹션을 참조하세요.

&lt;correct_patterns>

이 섹션에는 DCF 모델을 작성할 때 따라야 할 모든 올바른 패턴이 담겨 있습니다.

### 시나리오 블록 선택 패턴 - 다음 방식을 따르세요

**가정은 각 시나리오별로 별도의 블록에 구성합니다:**

**중요 구조 - 각 섹션 제목마다 세 개의 행:**

```csv
BEAR CASE ASSUMPTIONS (section header, merge cells across)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),12%,10%,9%,8%,7%
EBIT Margin (%),45%,44%,43%,42%,41%

BASE CASE ASSUMPTIONS (section header, merge cells across)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),16%,14%,12%,10%,9%
EBIT Margin (%),48%,49%,50%,51%,52%

BULL CASE ASSUMPTIONS (section header, merge cells across)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),20%,18%,15%,13%,11%
EBIT Margin (%),50%,51%,52%,53%,54%
```

각 시나리오 블록에는 예상 연도(FY2025E, FY2026E 등)를 보여주는 열 제목 행이 섹션 제목 바로 아래에 **반드시** 있어야 합니다. 그렇지 않으면 사용자는 어느 가정값이 어느 연도에 해당하는지 알 수 없습니다.

**가정 참조 방법 - 통합 열을 만드세요:**
1. 케이스 선택 셀(예: B6)에 1=Bear, 2=Base, 3=Bull을 입력합니다.
2. INDEX 또는 OFFSET 수식을 사용해 올바른 시나리오 블록에서 값을 가져오는 통합 열을 만듭니다.
3. 예상 수식은 통합 열(깔끔한 셀 참조)을 참조합니다.
4. 각 시나리오 블록에는 예상 연도 전체에 대한 DCF 가정의 전체 세트가 들어 있어야 합니다.

**권장 통합 열 패턴(INDEX 사용):**
`=INDEX(B10:D10, 1, $B$6)`

**다음 방식은 사용하지 마세요 - 곳곳에 흩어진 IF 문:**
`=IF($B$6=1,[Bear block cell],IF($B$6=2,[Base block cell],[Bull block cell]))`

통합 열 방식은 로직을 중앙화하므로 모델을 더 쉽게 감사할 수 있습니다.

### 올바른 매출 예상 패턴

**INDEX 수식을 사용해 통합 열을 만든 다음 예상치에서 참조합니다:**

**1단계 - FY1 성장률용 통합 열:**
`=INDEX([Bear FY1 growth]:[Bull FY1 growth], 1, $B$6)`

**2단계 - 매출 예상치에서 통합 열 참조:**
`Revenue Year 1: =D29*(1+$E$10)`

여기서:
- D29 = 전년도 매출
- $E$10 = FY1 성장률용 통합 열 셀(INDEX 수식 포함)
- $B$6 = 케이스 선택기(1=Bear, 2=Base, 3=Bull)

**이 방식은 모든 예상 수식에 IF 문을 직접 넣는 것보다 깔끔하며**, 어떤 시나리오 가정이 사용되는지 훨씬 쉽게 감사할 수 있습니다.

### 올바른 FCF 수식 패턴

**INDEX 수식이 있는 통합 열을 사용한 다음 FCF 계산에서 참조합니다:**

**통합 열 방식:**
```csv
Item,Formula,Reference
D&A,=E29*$E$21,$E$21 = consolidation column for D&A %
CapEx,=E29*$E$22,$E$22 = consolidation column for CapEx %
Δ NWC,=(E29-D29)*$E$23,$E$23 = consolidation column for NWC %
Unlevered FCF,=E57+E58-E60-E62,E57=NOPAT E58=D&A E60=CapEx E62=Δ NWC
```

**각 통합 열 셀에는 케이스 선택기에 따라 적절한 시나리오 블록에서 값을 가져오는 INDEX 수식이 들어 있습니다.** 이를 통해 예상 수식이 깔끔하고 감사 가능한 상태로 유지됩니다.

수식을 작성하기 전에 시나리오 블록의 행 위치를 확인하고 통합 열을 설정하세요.

### 올바른 셀 주석 형식

**모든 하드코딩된 값은 다음 형식을 사용해야 합니다:**

"Source: [System/Document], [Date], [Reference], [URL if applicable]"

**예시:**
```csv
Item,Source Comment
Stock price,Source: Market data script 2025-10-12 Close price
Shares outstanding,Source: 10-K FY2024 Page 45 Note 12
Historical revenue,Source: 10-K FY2024 Page 32 Consolidated Statements
Beta,Source: Market data script 2025-10-12 5-year monthly beta
Consensus estimates,Source: Management guidance Q3 2024 earnings call
```

### 올바른 가정 테이블 구조

**중요: 각 시나리오 블록에는 세 가지 구조 요소가 필요합니다:**

1. **섹션 제목 행**(병합 셀): 예: "BEAR CASE ASSUMPTIONS"
2. 연도를 보여주는 **열 제목 행** - **필수이며 생략하지 마세요**
3. 가정값이 있는 **데이터 행**

**구조:**
```csv
BEAR CASE ASSUMPTIONS (section header - merge across columns A:G)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),X%,X%,X%,X%,X%
EBIT Margin (%),X%,X%,X%,X%,X%
Terminal Growth,X%,,,,
WACC,X%,,,,

BASE CASE ASSUMPTIONS (section header - merge across columns A:G)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),X%,X%,X%,X%,X%
EBIT Margin (%),X%,X%,X%,X%,X%
Terminal Growth,X%,,,,
WACC,X%,,,,

BULL CASE ASSUMPTIONS (section header - merge across columns A:G)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),X%,X%,X%,X%,X%
EBIT Margin (%),X%,X%,X%,X%,X%
Terminal Growth,X%,,,,
WACC,X%,,,,
```

**예상 연도(FY2025E, FY2026E 등)를 보여주는 열 제목 행이 없으면 사용자는 어느 가정값이 어느 연도에 해당하는지 알 수 없습니다. 이 행은 **필수**입니다.**

**그런 다음 통합 열을 만드세요.** 일반적으로 오른쪽의 다음 열에 만들며, 케이스 선택기에 따라 선택한 시나리오 블록에서 INDEX 수식으로 값을 가져옵니다. 예상 수식은 이 통합 열을 참조합니다.

### 올바른 행 계획 프로세스

**1. 모든 제목과 라벨을 먼저 작성합니다:**
```csv
Row,Content
1,[Company Name] DCF Model
2,Ticker | Date | Year End
4,Case Selector
7,KEY ASSUMPTIONS
26,Assumption headers
27-31,Growth assumptions
...,...
```

**2. 모든 섹션 구분선과 빈 행을 작성합니다.**

**3. 그다음 고정된 행 위치를 사용해 수식을 작성합니다.**

**4. 생성 직후 수식을 테스트합니다.**

**건설 과정이라고 생각하세요:**
- 좋음: 기초를 붓고 벽을 세움(안정적인 구조)
- 나쁨: 벽을 세운 뒤 기초를 부음(벽이 무너짐)

**Excel 버전:**
- 좋음: 제목을 추가한 다음 수식을 작성함(수식이 안정적)
- 나쁨: 수식을 작성한 다음 제목을 추가함(수식이 깨짐)

### 올바른 민감도 표 구현

**중요**: Excel의 "데이터 표(Data Table)" 기능이 아닙니다. openpyxl을 사용해 일반 수식을 작성하는 단순한 그리드입니다. 즉, 총 약 75개의 수식(표 3개 × 셀 25개)을 작성해야 하지만, 이는 간단하며 필수입니다.

**수식을 사용한 프로그래밍 방식 채우기:**

각 민감도 표는 가정 조합별 내재 주당 가격을 재계산하는 수식으로 완전히 채워져야 합니다. **Excel의 데이터 표 기능을 사용하지 마세요**(수동 개입이 필요하고 openpyxl로 자동화할 수 없습니다).

**구현 방식 - 구체적인 예시:**

**표 구조 — 5×5 그리드(홀수 차원, 중심에 기본 케이스):**

모델의 기본 WACC가 9.0%이고 기본 터미널 성장률이 3.0%라면, 다음과 같이 해당 값들을 중심으로 축을 대칭적으로 구성합니다:

```csv
WACC vs Terminal Growth,  2.0%,  2.5%,  3.0%,  3.5%,  4.0%
              8.0%,       [fml], [fml], [fml], [fml], [fml]
              8.5%,       [fml], [fml], [fml], [fml], [fml]
              9.0%,       [fml], [fml], [★  ], [fml], [fml]   ← middle row = base WACC
              9.5%,       [fml], [fml], [fml], [fml], [fml]
             10.0%,       [fml], [fml], [fml], [fml], [fml]
                                   ↑
                          middle col = base terminal g
```

**★ = 중심 셀.** 이 셀의 수식 출력값은 가치평가 요약의 실제 내재 주당 가격과 **반드시** 같아야 합니다. 이 셀에 중간 파란색 채우기(`#BDD7EE`)와 굵은 글꼴을 적용해 기본 케이스가 시각적으로 고정되도록 하세요.

**축 값 규칙:** `axis_values = [base - 2*step, base - step, base, base + step, base + 2*step]` — 기본값을 중심으로 대칭이며, 홀수 개수이므로 중심이 보장됩니다.

**수식 패턴 - 셀 B88 (WACC=8.0%, 터미널 성장률=2.0%):**

B88의 수식은 다음을 사용해 내재 가격을 재계산해야 합니다:
- 행 제목의 WACC: `$A88` (8.0%)
- 열 제목의 터미널 성장률: `B$87` (2.0%)

**권장 방식:** 주 DCF 계산을 참조하되 이 값들로 대체합니다.

**수식 구조 예시:**
`=([SUM of PV FCFs using $A88 as discount rate] + [Terminal Value using B$87 as growth rate and $A88 as WACC] - [Net Debt]) / [Shares]`

**중요 - 5x5 그리드의 모든 셀(표당 25개, 총 75개)에 수식을 작성하세요.** openpyxl을 사용해 루프에서 프로그래밍 방식으로 이 수식을 작성합니다. 이 단계를 건너뛰거나 자리 표시자 텍스트를 남겨두지 마세요.

**Python 구현 패턴:**
```python
# Pseudocode for populating sensitivity table
for row_idx, wacc_value in enumerate(wacc_range):
    for col_idx, term_growth_value in enumerate(term_growth_range):
        # Build formula that uses wacc_value and term_growth_value
        formula = f"=<DCF recalc using {wacc_value} and {term_growth_value}>"
        ws.cell(row=start_row+row_idx, column=start_col+col_idx).value = formula
```

&lt;/correct_patterns>

&lt;common_mistakes>

이 섹션에는 DCF 모델을 작성할 때 피해야 할 모든 잘못된 패턴이 담겨 있습니다.

### 잘못된 방식: 단순화된 민감도 표 근사값 또는 자리 표시자 텍스트

**선형 근사값을 사용하지 마세요:**

```
// WRONG - Linear approximation
B97: =B88*(1+(0.096-0.116))    // Assumes linear relationship

// WRONG - Division shortcut
B105: =B88/(1+(E48-0.07))      // Doesn't recalculate full DCF
```

**자리 표시자 텍스트를 남겨두지 마세요:**
```
// WRONG - Placeholder note
"Note: Use Excel Data Table feature (Data → What-If Analysis → Data Table) to populate sensitivity tables."

// WRONG - Empty cells
[leaving cells blank because "this is complex"]
```

**용어를 혼동하지 마세요:**
- ❌ "민감도 표에는 Excel의 데이터 표 기능이 필요하다" (아니요 - 사용할 수 없는 특정 Excel 도구입니다)
- ✅ "민감도 표는 각 셀에 수식이 들어 있는 단순한 그리드다" (예 - 우리가 만드는 방식입니다)

**이러한 지름길이 잘못된 이유:**
- 선형 근사 수식은 DCF를 실제로 재계산하지 않고 단순한 수학적 조정만 적용합니다.
- 관계가 선형이 아니므로 결과가 부정확합니다.
- 자리 표시자 텍스트는 사용자의 수동 개입을 요구합니다.
- 제공된 모델을 즉시 사용할 수 없습니다.
- 전문적이지 않고 고객에게 제공할 수준이 아닙니다.
- 빈 셀 = 불완전한 결과물

**거부해야 할 일반적인 합리화:**
"75개 이상의 수식을 작성하는 것은 복잡해 보이므로 사용자가 수동으로 완료하도록 메모를 남기겠습니다."

**현실:** Python에서 openpyxl과 루프를 사용하면 75개의 수식을 작성하는 것은 간단합니다. 각 수식은 같은 패턴을 따르며 행/열 값만 대체하면 됩니다. 이는 결과물에 반드시 포함해야 하는 부분입니다.

**대신:** 특정 조합에 대해 전체 DCF를 재계산하는 수식으로 모든 민감도 셀을 채우세요.
### 잘못된 예: 셀 주석 누락

**다음과 같이 하지 마세요:**
- 모든 하드코딩 입력값을 주석 없이 생성
- "나중에 추가하자"고 생각하기
- "TODO: 출처 추가"라고 작성
- 파란색 입력 셀에 문서화 누락

**잘못된 이유:**
- 데이터의 출처를 확인할 수 없음
- xlsx 스킬 요구사항을 충족하지 못함
- 감사 대응이 불가능함
- 나중에 수정하느라 시간을 낭비함

**대신:** 각 하드코딩 값을 생성하는 즉시 셀 주석을 추가하세요

### 잘못된 예: 수식의 행 참조 오류

**증상:**
FCF 섹션이 잘못된 가정 행을 참조합니다:
`D&A:  =E29*$E$34    // Should be $E$21, but referencing wrong row`
`CapEx: =E29*$E$41   // Should be $E$22, but row shifted`

**발생 원인:**
1. 수식을 먼저 작성함
2. 이후 헤더를 삽입함
3. 모든 행 참조가 이동함
4. 그 결과 수식이 잘못된 셀을 가리켜 #REF! 오류가 발생함

**대신:** 먼저 행 레이아웃을 확정한 후 수식을 작성하세요

### 잘못된 예: 시나리오별 가정을 한 행에 하나씩 배치

**다음과 같이 가정을 구성하지 마세요:**
```csv
Assumption,Bear,Base,Bull
Revenue Growth FY1,10%,13%,16%
Revenue Growth FY2,9%,12%,15%
```
이 세로 레이아웃은 각 시나리오에서 연도별 변화 과정을 확인하기 어렵게 만듭니다.

**잘못된 이유:**
- 각 시나리오에서 가정이 연도별로 어떻게 변화하는지 파악하기 어려움
- 전체 전망 기간에 걸쳐 시나리오 가정을 비교하기 어려움
- 시나리오 논리를 하나의 일관된 묶음으로 검토하기 어려움

**대신:**
- 각 시나리오(Bear, Base, Bull)에 대해 별도의 블록을 만드세요
- 각 블록 안에서 전망 연도별 가정을 가로 방향으로 표시하세요
- 이렇게 하면 각 시나리오의 가정을 하나의 일관된 집합으로 더 쉽게 검토할 수 있습니다

### 잘못된 예: 테두리 없음

**테두리 없는 모델을 제공하지 마세요:**
- 섹션 구분이 없음
- 모든 셀이 서로 섞여 보임
- 읽기 어렵고 전문성이 떨어짐

**잘못된 이유:**
- 고객에게 제공할 수 있는 수준이 아님
- 탐색하기 어려움
- 아마추어처럼 보임

**대신:** 모든 주요 섹션에 테두리를 추가하세요

### 잘못된 예: 잘못된 글꼴 색상 또는 글꼴 색상 구분 없음

**다음과 같이 하지 마세요:**
- 모든 텍스트를 검은색으로 설정
- 채우기 색상만 사용하고 글꼴 색상은 변경하지 않음
- 어떤 셀이 파란색이어야 하고 어떤 셀이 검은색이어야 하는지 혼동

**잘못된 이유:**
- 입력값과 수식을 구분할 수 없음
- 감사가 불가능해짐
- xlsx 스킬 요구사항을 위반함

**대신:** 모든 하드코딩 입력값은 파란색, 모든 수식은 검은색, 시트 간 참조는 녹색으로 표시하세요

### 잘못된 예: 매출총이익을 기준으로 영업비용 계산

**다음과 같이 하지 마세요:**
`S&M: =E33*0.15    // E33 = Gross Profit (WRONG)`

**잘못된 이유:**
- 영업비용은 매출총이익이 아니라 매출에 따라 증가함
- 마진 변화가 비현실적으로 나타남
- 실제 기업 운영 방식과 맞지 않음

**대신:**
`S&M: =E29*0.15    // E29 = Revenue (CORRECT)`

### 상위 5가지 오류 요약

1. **수식의 행 참조 오류** → 모든 행 위치를 수식 작성 전에 정의하세요
2. **셀 주석 누락** → 마지막이 아니라 셀을 생성하는 즉시 주석을 추가하세요
3. **단순화된 민감도 표** → 근사값이 아니라 전체 DCF 재계산 수식으로 모든 셀을 채우세요
4. **시나리오 블록 참조 오류** → IF 수식이 올바른 Bear/Base/Bull 블록을 가져오는지 확인하세요
5. **테두리 없음** → 고객에게 제공할 수 있는 전문적인 모델을 위해 섹션 테두리를 추가하세요

또한 다음 오류에도 주의하세요:

### WACC 계산 오류
- 자본 구조에서 장부가와 시장가를 혼용함
- 자기자본 베타를 사용하거나 자산/무차입 베타를 잘못 사용함
- 부채비용에 세율을 잘못 적용함
- 무위험 수익률이 잘못됨(현재 10년물 국채 수익률을 사용해야 함)
- 순부채와 순현금 포지션을 조정하지 않음

### 성장 가정의 결함
- 말기 성장률이 WACC보다 큼(무한 가치가 발생함)
- 전망 성장률이 과거 실적과 일치하지 않음
- 산업 성장 제약을 무시함
- 매출 성장이 단위 경제성과 맞지 않음
- 운영상 근거 없이 마진이 확대됨

### 말기 가치 오류
- 잘못된 성장 방식을 사용함(영구성장법과 출구 배수법)
- 말기 가치가 기업가치의 80%를 초과함(과도한 의존을 시사함)
- 말기 마진이 정상 상태 가정과 일치하지 않음
- 말기 가치의 할인 기간이 잘못됨

### 현금흐름 전망 오류
- 매출이익이 아니라 매출총이익을 기준으로 영업비용을 계산함
- D&A/CapEx 비율이 비즈니스 모델과 맞지 않음
- 운전자본 변동을 올바르게 계산하지 않음
- 연도별 세율이 일관되지 않음
- NOPAT 계산 오류

**이러한 오류가 가장 흔하게 발생합니다. DCF 작성을 시작하기 전에 이 섹션을 다시 읽으세요.**

&lt;/common_mistakes>

## Excel 파일 생성

**이 스킬은 모든 스프레드시트 작업에 `xlsx` 스킬을 사용합니다.** xlsx 스킬은 다음을 제공합니다:
- 표준화된 수식 작성 규칙
- 숫자 서식 규칙
- `excel-author` 스킬의 `recalc.py` 스크립트를 통한 자동 수식 재계산
- 포괄적인 오류 검사 및 검증

이 스킬로 생성하는 모든 Excel 파일은 수식 오류가 0건이고 올바르게 재계산되는 등 xlsx 스킬 요구사항을 따라야 합니다.

## 품질 기준

모든 DCF 모델은 다음 항목을 최대화해야 합니다:
1. **과거 실적에 근거한 현실적인 매출 및 마진 가정**
2. **적절한 CAPM 방법론에 따른 자본비용 계산**
3. **밸류에이션 범위를 보여주는 포괄적인 민감도 분석**
4. **근거를 뒷받침하는 명확한 말기 가치 계산**
5. **시나리오 분석이 가능한 전문적인 모델 구조**
6. **모든 주요 가정에 대한 투명한 문서화**

## 입력 요구사항

### 최소 필수 입력값
1. **기업 식별자**: 티커 심볼 또는 기업명
2. **성장 가정**: 전망 기간의 매출 성장률(또는 "컨센서스 사용")
3. **선택적 매개변수**:
   - 전망 기간(기본값: 5년)
   - 시나리오 케이스(Bear/Base/Bull 성장 및 마진 가정)
   - 말기 성장률(기본값: 2.5-3.0%)
   - CAPM을 사용하지 않는 경우의 구체적인 WACC 입력값

## Excel 모델 구조

### 시트 구성

**두 개의 시트를 생성하세요:**

1. **DCF** - 하단에 민감도 분석이 포함된 주요 가치평가 모델
2. **WACC** - 자본비용 계산

**중요**: 민감도 표는 DCF 시트의 **하단**에 배치하세요(별도 시트에 배치하지 않음). 이렇게 하면 모든 가치평가 결과를 한곳에 모을 수 있습니다.

### 수식 재계산 (필수)

Excel 모델을 생성하거나 수정한 후 `excel-author` 스킬의 `recalc.py` 스크립트를 사용하여 **모든 수식을 재계산하세요**:

```bash
python recalc.py [path_to_excel_file] [timeout_seconds]
```

예시:
```bash
python recalc.py AAPL_DCF_Model_2025-10-12.xlsx 30
```

스크립트는 다음 작업을 수행합니다:
- LibreOffice를 사용하여 모든 시트의 모든 수식을 재계산
- 모든 셀에서 Excel 오류(#REF!, #DIV/0!, #VALUE!, #NAME?, #NULL!, #NUM!, #N/A)를 검사
- 오류 위치와 개수가 포함된 상세 JSON 반환

**예상 출력 형식:**
```json
{
  "status": "success",           // or "errors_found"
  "total_errors": 0,              // Total error count
  "total_formulas": 42,           // Number of formulas in file
  "error_summary": {}             // Only present if errors found
}
```

**오류가 발견되면** 출력에 다음과 같은 세부 정보가 포함됩니다:
```json
{
  "status": "errors_found",
  "total_errors": 2,
  "total_formulas": 42,
  "error_summary": {
    "#REF!": {
      "count": 2,
      "locations": ["DCF!B25", "DCF!C25"]
    }
  }
}
```

**모든 오류를 수정하고** 상태가 "success"가 될 때까지 `recalc.py`를 다시 실행한 후 모델을 제공하세요.

### 서식 기준

**중요**: 수식 작성 규칙과 숫자 서식 규칙은 xlsx 스킬을 따르세요. DCF 스킬은 여기에 구체적인 시각적 표현 기준을 추가합니다.

**색상 체계 - 두 개의 레이어**:

**레이어 1: 글꼴 색상(xlsx 스킬에서 필수)**
- **파란색 텍스트(RGB: 0,0,255)**: 모든 하드코딩 입력값(주가, 주식 수, 과거 데이터, 가정)
- **검은색 텍스트(RGB: 0,0,0)**: 모든 수식 및 계산
- **녹색 텍스트(RGB: 0,128,0)**: 다른 시트로 연결되는 참조(WACC 시트 참조)

**레이어 2: 채우기 색상 - 전문적인 파랑/회색 팔레트(사용자가 별도로 지정하지 않은 경우의 기본값)**
- **최소한으로 사용하세요** — 채우기에는 파란색과 회색만 사용합니다. 녹색, 노란색, 주황색 또는 여러 강조 색상을 추가하지 마세요. 색상이 너무 많으면 모델이 아마추어처럼 보입니다.
- **기본 채우기 팔레트:**
  - **섹션 헤더**: 진한 파란색(RGB: 31,78,121 / `#1F4E79`) 배경과 흰색 굵은 텍스트
  - **하위 헤더/열 헤더**: 연한 파란색(RGB: 217,225,242 / `#D9E1F2`) 배경과 검은색 굵은 텍스트
  - **입력 셀**: 연한 회색(RGB: 242,242,242 / `#F2F2F2`) 배경과 파란색 글꼴 — 또는 최소한의 표현을 원한다면 파란색 글꼴의 흰색 배경
  - **계산 셀**: 흰색 배경과 검은색 글꼴
  - **출력/요약 행**(주당 가치, EV 등): 중간 파란색(RGB: 189,215,238 / `#BDD7EE`) 배경과 검은색 굵은 글꼴
- **이것이 전부입니다 — 파란색 3가지 + 회색 1가지 + 흰색입니다.** 색상을 더 추가하고 싶은 충동을 억제하세요.
- 사용자가 제공한 템플릿이나 명시적인 색상 선호도는 언제나 이 기본값보다 우선합니다.

**레이어가 함께 작동하는 방식:**
- 입력 셀: 파란색 글꼴 + 연한 회색 채우기 = "하드코딩 입력값"
- 수식 셀: 검은색 글꼴 + 흰색 배경 = "계산된 값"
- 시트 참조: 녹색 글꼴 + 흰색 배경 = "다른 시트의 참조값"
- 핵심 출력: 검은색 굵은 글꼴 + 중간 파란색 채우기 = "답변"

**글꼴 색상은 셀의 성격(입력값/수식/참조)을 알려주고, 채우기 색상은 셀의 위치(헤더/데이터/출력)를 알려줍니다.**

### 테두리 기준 (전문적인 표현을 위해 필수)

**주요 섹션 주위의 굵은 테두리**(1.5pt):
- KEY INPUTS 섹션
- PROJECTION ASSUMPTIONS 섹션
- 5-YEAR CASH FLOW PROJECTION 섹션
- TERMINAL VALUE 섹션
- VALUATION SUMMARY 섹션
- 각 SENSITIVITY ANALYSIS 표

**하위 섹션 사이의 중간 테두리**(1pt):
- Company Details와 Historical Performance 사이
- Growth Assumptions와 EBIT Margin, FCF Parameters 사이

**데이터 표 주위의 얇은 테두리**(0.5pt):
- 시나리오 가정 표(Bear | Base | Bull | Selected)
- 과거 재무와 전망 재무 매트릭스

**테두리 없음:** 표 안의 개별 셀(깔끔하고 한눈에 파악할 수 있도록 유지)

**테두리는 필수입니다** — 전문적인 테두리가 없는 모델은 고객에게 제공할 수 있는 수준이 아닙니다.

**숫자 서식**(xlsx 스킬 기준 준수):
- **연도**: 텍스트 문자열로 서식 지정(예: "2,024"가 아니라 "2024")
- **백분율**: `0.0%`(소수점 한 자리)
- **통화**: 백만 단위는 `$#,##0`, 주당 금액은 `$#,##0.00` - 헤더에 단위를 항상 명시("Revenue ($mm)")
- **0**: 숫자 서식을 사용하여 모든 0을 "-"로 표시(예: `$#,##0;($#,##0);-`)
- **큰 숫자**: 천 단위 구분 기호가 있는 `#,##0`
- **음수**: 괄호 안에 표시(`(#,##0)`), 마이너스 기호는 사용하지 않음

**셀 주석(모든 하드코딩 입력값에 필수):**

xlsx 스킬에 따라 **모든 하드코딩 값에는 출처를 기록한 셀 주석이 있어야 합니다.** 형식: "Source: [System/Document], [Date], [Reference], [URL if applicable]"

**중요**: 셀을 생성하는 즉시 주석을 추가하세요. 마지막까지 미루지 마세요.

### DCF 시트 상세 구조

**섹션 1: 헤더**
```csv
Row,Content
1,[Company Name] DCF Model
2,Ticker: [XXX] | Date: [Date] | Year End: [FYE]
3,Blank
4,Case Selector Cell (1=Bear 2=Base 3=Bull)
5,Case Name Display (formula: =IF([Selector]=1"Bear"IF([Selector]=2"Base""Bull")))
```

**섹션 2: 시장 데이터(케이스에 종속되지 않음)**
```csv
Item,Value
Current Stock Price,$XX.XX
Shares Outstanding (M),XX.X
Market Cap ($M),[Formula]
Net Debt ($M),XXX [or Net Cash if negative]
```

**섹션 3: DCF 시나리오 가정**

각 시나리오(Bear, Base, Bull)에 대해 DCF 전용 가정(매출 성장률 %, EBIT 마진 %, 세율 %, 매출 대비 D&A %, 매출 대비 CapEx %, 매출 변동 대비 NWC 변동 %, 말기 성장률, WACC)을 별도의 가정 블록으로 만들고 전망 연도별로 가로 방향으로 배치하세요. 각 블록에는 섹션 헤더, 전망 연도(FY1, FY2 등)를 보여주는 열 헤더 행, 데이터 행이 포함되어야 합니다. 정확한 레이아웃은 `<correct_patterns>` 섹션의 "Correct Assumption Table Structure"를 참조하세요.

**섹션 4: 과거 및 전망 재무**

**시나리오 블록에서 값을 가져오는 통합 열(예: "Selected Case")을 참조하세요.** 모든 전망 행에 IF 수식을 흩어져 배치하지 마세요.

```csv
Income Statement ($M),2020A,2021A,2022A,2023A,2024E,2025E,2026E
Revenue,XXX,XXX,XXX,XXX,[=E29*(1+$E$10)],[=F29*(1+$E$11)],[=G29*(1+$E$12)]
  % growth,XX%,XX%,XX%,XX%,[=E29/D29-1],[=F29/E29-1],[=G29/F29-1]
,,,,,,
Gross Profit,XXX,XXX,XXX,XXX,[=E29*E33],[=F29*F33],[=G29*G33]
  % margin,XX%,XX%,XX%,XX%,[=E33/E29],[=F33/F29],[=G33/G29]
,,,,,,
Operating Expenses:,,,,,,,
  S&M,XXX,XXX,XXX,XXX,[=E29*0.15],[=F29*0.14],[=G29*0.13]
  R&D,XXX,XXX,XXX,XXX,[=E29*0.12],[=F29*0.11],[=G29*0.10]
  G&A,XXX,XXX,XXX,XXX,[=E29*0.08],[=F29*0.07],[=G29*0.07]
  Total OpEx,XXX,XXX,XXX,XXX,[=E36+E37+E38],[=F36+F37+F38],[=G36+G37+G38]
,,,,,,
EBIT,XXX,XXX,XXX,XXX,[=E33-E39],[=F33-F39],[=G33-G39]
  % margin,XX%,XX%,XX%,XX%,[=E41/E29],[=F41/F29],[=G41/G29]
,,,,,,
Taxes,(XX),(XX),(XX),(XX),[=E41*$E$24],[=F41*$E$24],[=G41*$E$24]
  Tax rate,XX%,XX%,XX%,XX%,[=E43/E41],[=F43/F41],[=G43/G41]
,,,,,,
NOPAT,XXX,XXX,XXX,XXX,[=E41-E43],[=F41-F43],[=G41-G43]
```

**핵심 수식 패턴:**
- 매출 성장률: `$E$10`이 1년 차 성장률의 통합 열인 `=E29*(1+$E$10)`
- 사용하지 말아야 할 방식: `=E29*(1+IF($B$6=1,$B$10,IF($B$6=2,$C$10,$D$10)))`

이 방식은 더 깔끔하고 감사하기 쉬우며, 시나리오 로직을 중앙화하여 수식 오류를 방지합니다.

**섹션 5: 잉여현금흐름 작성**

**중요**: 행 참조가 **올바른** 가정 행을 가리키는지 확인하세요. 수식을 생성한 즉시 테스트하세요.

```csv
Cash Flow ($M),2020A,2021A,2022A,2023A,2024E,2025E,2026E
NOPAT,XXX,XXX,XXX,XXX,[=E45],[=F45],[=G45]
(+) D&A,XXX,XXX,XXX,XXX,[=E29*$E$21],[=F29*$E$21],[=G29*$E$21]
    % of Rev,XX%,XX%,XX%,XX%,[=E58/E29],[=F58/F29],[=G58/G29]
(-) CapEx,(XX),(XX),(XX),(XX),[=E29*$E$22],[=F29*$E$22],[=G29*$E$22]
    % of Rev,XX%,XX%,XX%,XX%,[=E60/E29],[=F60/F29],[=G60/G29]
(-) Δ NWC,(XX),(XX),(XX),(XX),[=(E29-D29)*$E$23],[=(F29-E29)*$E$23],[=(G29-F29)*$E$23]
    % of Δ Rev,XX%,XX%,XX%,XX%,[=E62/(E29-D29)],[=F62/(F29-E29)],[=G62/(G29-F29)]
,,,,,,
Unlevered FCF,XXX,XXX,XXX,XXX,[=E57+E58-E60-E62],[=F57+F58-F60-F62],[=G57+G58-G60-G62]
```

**행 참조 예시**(레이아웃 계획에 기반):
- `$E$21` = D&A % 가정(통합 열, 21행)
- `$E$22` = CapEx % 가정(통합 열, 22행)
- `$E$23` = NWC % 가정(통합 열, 23행)
- `E29` = 해당 연도의 매출(29행)
- `E45` = 해당 연도의 NOPAT(45행)

**수식 작성 전**: 이 행 번호가 실제 레이아웃과 일치하는지 확인하세요. 한 열을 먼저 테스트한 후 가로 방향으로 복사하세요.

**섹션 6: 할인 및 가치평가**
```csv
DCF Valuation,2024E,2025E,2026E,2027E,2028E,Terminal
Unlevered FCF ($M),XXX,XXX,XXX,XXX,XXX,
Period,0.5,1.5,2.5,3.5,4.5,
Discount Factor,0.XX,0.XX,0.XX,0.XX,0.XX,
PV of FCF ($M),XXX,XXX,XXX,XXX,XXX,
,,,,,,
Terminal FCF ($M),,,,,,,XXX
Terminal Value ($M),,,,,,,XXX
PV Terminal Value ($M),,,,,,,XXX
,,,,,,
Valuation Summary ($M),,,,,,
Sum of PV FCFs,XXX,,,,,
PV Terminal Value,XXX,,,,,
Enterprise Value,XXX,,,,,
(-) Net Debt,(XX),,,,,
Equity Value,XXX,,,,,
,,,,,,
Shares Outstanding (M),XX.X,,,,,
IMPLIED PRICE PER SHARE,$XX.XX,,,,,
Current Stock Price,$XX.XX,,,,,
Implied Upside/(Downside),XX%,,,,,
```
### WACC 시트 구조

```csv
COST OF EQUITY CALCULATION,,
Risk-Free Rate (10Y Treasury),X.XX%,[Yellow input]
Beta (5Y monthly),X.XX,[Yellow input]
Equity Risk Premium,X.XX%,[Yellow input]
Cost of Equity,X.XX%,[Calculated blue]
,,
COST OF DEBT CALCULATION,,
Credit Rating,AA-,[Yellow input]
Pre-Tax Cost of Debt,X.XX%,[Yellow input]
Tax Rate,XX.X%,[Link to DCF sheet]
After-Tax Cost of Debt,X.XX%,[Calculated blue]
,,
CAPITAL STRUCTURE,,
Current Stock Price,$XX.XX,[Link to DCF]
Shares Outstanding (M),XX.X,[Link to DCF]
Market Capitalization ($M),"X,XXX",[Calculated]
,,
Total Debt ($M),XXX,[Yellow input]
Cash & Equivalents ($M),XXX,[Yellow input]
Net Debt ($M),XXX,[Calculated]
,,
Enterprise Value ($M),"X,XXX",[Calculated]
,,
WACC CALCULATION,Weight,Cost,Contribution
Equity,XX.X%,X.X%,X.XX%
Debt,XX.X%,X.X%,X.XX%
,,
WEIGHTED AVERAGE COST OF CAPITAL,X.XX%,[Green output]
```

**주요 WACC 공식:**
```
Market Cap = Price × Shares
Net Debt = Total Debt - Cash
Enterprise Value = Market Cap + Net Debt
Equity Weight = Market Cap / EV
Debt Weight = Net Debt / EV
WACC = (Cost of Equity × Equity Weight) + (After-tax Cost of Debt × Debt Weight)
```

### 민감도 분석(DCF 시트 하단)

**용어 안내**: "민감도 표"는 행 머리글, 열 머리글, 각 데이터 셀의 수식으로 구성된 단순한 2D 그리드입니다. Excel의 "데이터 표" 기능(Data → 가상 분석 → 데이터 표)이 아닙니다. openpyxl을 사용해 각 셀에 일반 Excel 수식을 작성합니다.

**위치**: DCF 시트의 87행 이후(별도 시트가 아님)

**세 개의 민감도 표를 세로로 배치:**

1. **WACC 대 터미널 성장률**(87~100행) - 5x5 그리드 = 수식이 있는 25개 셀
2. **매출 성장률 대 EBIT 마진**(102~115행) - 5x5 그리드 = 수식이 있는 25개 셀
3. **베타 대 무위험 수익률**(117~130행) - 5x5 그리드 = 수식이 있는 25개 셀

**작성할 총 수식 수: 75개**(필수이며 선택 사항이 아님)

**중요**: 모든 민감도 표 셀은 openpyxl을 사용해 프로그래밍 방식으로 수식을 채워야 합니다. 선형 근사 방식의 지름길을 사용하지 마세요. 수동 작업 단계에 대한 자리 표시자 텍스트나 메모를 남기지 마세요. "복잡하다"는 이유로 셀을 비워 두는 것을 정당화하지 마세요. Python 루프를 사용해 수식을 생성하세요.

**표 설정:**
1. 행/열 머리글(테스트할 가정값)과 함께 표 구조를 만듭니다.
2. 모든 데이터 셀에 다음 조건을 충족하는 수식을 채웁니다.
   - 행 머리글 값을 사용합니다(예: WACC = 9.0%).
   - 열 머리글 값을 사용합니다(예: 터미널 성장률 = 3.0%).
   - 지정된 가정으로 전체 DCF를 다시 계산합니다.
   - 해당 시나리오의 내재 주가를 반환합니다.
3. 전달 시 모든 셀에 작동하는 수식이 들어 있어야 합니다.
4. 조건부 서식으로 셀을 지정합니다. 높은 값에는 녹색 계열, 낮은 값에는 빨간색 계열을 사용합니다.
5. 기본 사례 셀을 굵게 표시합니다.
6. 표 사이에 빈 행을 1~2개 둡니다.

**수동 개입은 필요하지 않습니다** - 사용자가 파일을 열었을 때 민감도 표가 완전히 작동해야 합니다.

## 사례 선택기 구현

**세 가지 사례 프레임워크:**

### 약세 사례
- 보수적인 매출 성장률(과거 범위의 하단)
- 마진 축소 또는 마진 확대 없음
- 더 높은 WACC(위험 프리미엄 증가)
- 더 낮은 터미널 성장률
- 더 높은 CapEx 가정

### 기본 사례
- 컨센서스 또는 경영진 가이던스에 따른 매출 성장률
- 영업 레버리지에 기반한 완만한 마진 확대
- 현재 시장이 암시하는 WACC
- GDP에 부합하는 터미널 성장률(2.5~3.0%)
- 표준 CapEx 가정

### 강세 사례
- 낙관적인 매출 성장률(전망치 상단)
- 상당한 마진 확대
- 더 낮은 WACC(위험 프리미엄 감소)
- 더 높은 터미널 성장률(3.5~5.0%)
- CapEx 집약도 감소

**수식 구현:**

**중첩 IF 수식을 여러 곳에 분산해 사용하지 마세요.** 대신 INDEX 또는 OFFSET 수식을 사용해 적절한 시나리오 블록에서 값을 가져오는 통합 열을 만드세요.

**권장 패턴(INDEX 사용):**
`=INDEX(B10:D10, 1, $B$6)` 여기서 `B10:D10` = 약세/기본/강세 값, `1` = 행 오프셋, `$B$6` = 사례 선택기 셀(1, 2 또는 3)

**그런 다음** 모든 전망에서 통합 열을 참조합니다.
`Revenue Year 1: =D29*(1+$E$10)` 여기서 $E$10은 1년 차 성장률에 대한 통합 열 값입니다.

이 접근법은 시나리오 로직을 중앙화하므로 모델을 더 쉽게 감사하고 유지 관리할 수 있습니다.

## 산출물 구조

**파일 이름**: `[Ticker]_DCF_Model_[Date].xlsx`

**두 개의 시트:**
1. **DCF** - 약세/기본/강세 사례와 하단의 세 민감도 표(WACC 대 터미널 성장률, 매출 성장률 대 EBIT 마진, 베타 대 무위험 수익률)를 포함한 완전한 모델
2. **WACC** - 자본 비용 계산

**주요 기능**: 사례 선택기(1/2/3), INDEX/OFFSET 수식이 있는 통합 열, 색상으로 구분된 셀, 모든 입력값의 셀 주석, 전문적인 테두리

## 모범 사례

### 모델 구성
1. **점진적으로 구축**: 다음 단계로 넘어가기 전에 각 섹션을 완료합니다.
2. **구축하면서 테스트**: 수식이 올바른지 확인하기 위해 샘플 숫자를 입력합니다.
3. **일관된 구조 사용**: 유사한 계산에는 유사한 패턴을 따릅니다.
4. **복잡한 수식에 주석 추가**: 특이한 계산에 대한 메모를 추가합니다.
5. **검증 기능 구축**: 해당되는 경우 합계 검증과 잔액 검증을 추가합니다.

### 문서화
1. **모든 가정 문서화**: 주요 입력값의 근거를 설명합니다.
2. **데이터 출처 인용**: 각 데이터 포인트의 출처를 기록합니다.
3. **방법론 설명**: 표준적이지 않은 접근법을 설명합니다.
4. **불확실성 표시**: 가시성이 제한된 영역을 강조 표시합니다.

### 품질 관리
1. **계산 교차 검증**: 여러 방식으로 수학적 계산을 확인합니다.
2. **가정 스트레스 테스트**: 모델의 견고성을 확인하기 위해 민감도 분석을 실행합니다.
3. **동료 검토**: 다른 사람이 수식을 확인하도록 합니다.
4. **버전 관리**: 작업이 진행되는 동안 버전을 저장합니다.

## 일반적인 변형

### 고성장 기술 기업
- 더 긴 전망 기간(7~10년)
- 더 높은 초기 성장률(20~30%)
- 시간에 따른 상당한 마진 확대
- 더 높은 WACC(12~15%)
- 단위 경제성(사용자 수, ARPU 등) 모델링

### 성숙/안정 기업
- 더 짧은 전망 기간(3~5년)
- 완만한 성장률(GDP +1~3%)
- 안정적인 마진
- 더 낮은 WACC(7~9%)
- 현금 창출과 자본 배분에 집중

### 경기순환 기업
- 경기 사이클 전체를 모델링
- 중간 사이클 수준으로 마진 정상화
- 저점 및 고점 시나리오 고려
- 경기순환성에 맞춰 베타 조정

### 다중 부문 기업
- 각 사업 부문별 DCF 분리
- 부문별로 다른 성장률과 마진 적용
- SOTP(sum-of-parts) 가치평가
- 시너지 고려

## 문제 해결

**오류나 비합리적인 결과가 발생하면** 자세한 디버깅 지침은 [TROUBLESHOOTING.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/finance/dcf-model/TROUBLESHOOTING.md)를 읽으세요.

## 워크플로 통합

### DCF 구축 시작 시

1. **시장 데이터 수집**:
   - 현재 시장 데이터에 사용할 수 있는 MCP 서버가 있는지 확인합니다.
   - 주가, 베타 및 기타 시장 지표에 웹 검색/가져오기를 사용합니다.
   - 특정 데이터가 필요한 경우 사용자에게 요청합니다.

2. **과거 재무 데이터 수집**:
   - 사용 가능한 MCP 서버(Daloopa 등)가 있는지 확인합니다.
   - MCP를 통해 사용할 수 없는 경우 사용자에게 요청합니다.
   - 필요한 경우 10-K에서 수동으로 추출합니다.

3. 이 스킬에 자세히 설명된 DCF 방법론을 사용해 **모델 구축을 시작합니다.**

### 모델 구축 중

1. 수식(하드코딩된 값이 아님)을 포함한 openpyxl을 사용해 **Excel 모델을 구축합니다.**
2. 수식 구성에는 **xlsx 스킬 규칙을 따릅니다.**
3. 사용자가 요청했거나 특정 브랜드 가이드라인이 제공된 경우에만 **채우기 색상을 적용합니다.**

### 모델 전달 전(필수)

1. **구조를 검증합니다**:
   - 전망 연도별 가정이 포함된 약세/기본/강세 시나리오 블록
   - 올바른 시나리오 블록을 참조하는 수식으로 사례 선택기가 작동하는지 확인
   - DCF 시트 하단의 민감도 표(별도 시트가 아님)
   - 글꼴 색상: 입력값은 파란색, 수식은 검은색, 시트 링크는 녹색
   - 모든 하드코딩 입력값에 셀 주석 추가
   - 주요 섹션 주위에 전문적인 테두리 적용

2. **수식을 다시 계산합니다**: `python recalc.py model.xlsx 30` 실행

3. **출력을 확인합니다**:
   - `status`가 `"success"`이면 4단계로 계속 진행합니다.
   - `status`가 `"errors_found"`이면 `error_summary`를 확인하고 디버깅 지침은 [TROUBLESHOOTING.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/finance/dcf-model/TROUBLESHOOTING.md)를 읽습니다.

4. `status`가 `"success"`가 될 때까지 오류를 수정하고 `recalc.py`를 다시 실행합니다.

5. **수식을 표본 검사합니다**:
   - FCF 수식 하나를 테스트합니다. 올바른 가정 행을 참조합니까?
   - 사례 선택기를 변경합니다. 통합 열이 올바르게 업데이트됩니까?
   - 매출 수식이 중첩 IF 수식이 아닌 통합 열을 참조하는지 확인합니다.

6. **모델을 전달합니다.**

### 사용 가능한 데이터 출처

- **MCP 서버**: 구성된 경우(과거 재무 데이터에는 Daloopa 사용)
- **웹 검색/가져오기**: 현재 주가, 베타 및 시장 데이터
- **사용자 제공 데이터**: 과거 재무 데이터, 컨센서스 추정치
- **수동 추출**: 대안으로 SEC EDGAR 제출 자료 사용

## 최종 출력 체크리스트

DCF 모델을 전달하기 전에:

**필수:**
- `python recalc.py model.xlsx 30`을 `status`가 `"success"`가 될 때까지 실행(수식 오류 0개)
- 두 개의 시트: DCF(하단에 민감도 분석 포함), WACC
- 글꼴 색상: 파란색=입력값, 검은색=수식, 녹색=시트 링크
- 모든 하드코딩 입력값에 셀 주석 추가
- 수식으로 민감도 표를 완전히 채움
- 주요 섹션 주위에 전문적인 테두리 적용

**검증:**
- OpEx가 매출을 기반으로 하는지(매출총이익이 아님)
- 터미널 가치가 EV의 50~70%인지
- 터미널 성장률 &lt; WACC인지
- 세율이 21~28%인지
- 파일 이름: `[Ticker]_DCF_Model_[Date].xlsx`

## 데이터 출처 — MCP 우선, 웹 대체

아래의 많은 문단에서는 "S&P Kensho MCP / Daloopa MCP / FactSet MCP를 사용"하라고 설명합니다. 이러한 상용 금융 데이터 MCP는 원래 Cowork 플러그인 컨텍스트에서 가져온 것입니다. Hermes에서는 다음을 따릅니다.

- **구조화된 금융 데이터 MCP가 구성되어 있다면**(Hermes는 MCP를 지원합니다. `native-mcp` 스킬 참조), 시점 기준 비교 기업, 선례 거래 및 공시 자료에 이를 우선 사용합니다.
- **그렇지 않다면** 다음으로 대체합니다.
  - 미국 공시는 SEC EDGAR(`https://www.sec.gov/cgi-bin/browse-edgar`)에 대한 `web_search` / `web_extract`
  - 보도 자료와 실적 발표 자료는 기업 IR 페이지
  - 대화형 데이터 포털은 `browser_navigate`
  - 사용자 제공 데이터(컨텍스트에 데이터가 없으면 명시적으로 요청)
- **절대 지어내지 마세요.** 배수, 선례 또는 공시 수치를 출처와 함께 제시할 수 없다면 해당 셀을 `[UNSOURCED]`로 표시하고 사용자에게 알립니다.

## 저작자 표시

이 스킬은 Anthropic의 Financial Services 플러그인 제품군(Apache-2.0)을 바탕으로 수정되었습니다. Office-JS / Cowork 실시간 Excel 경로는 제거되었으며, 이 버전은 `excel-author` 스킬 규칙에 따른 헤드리스 openpyxl을 대상으로 합니다. 원본: https://github.com/anthropics/financial-services
