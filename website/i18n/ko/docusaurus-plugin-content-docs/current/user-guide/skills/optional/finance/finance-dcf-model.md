---
title: "Dcf Model"
sidebar_label: "Dcf Model"
description: "Excel에서 기관 수준의 DCF 평가 모델 구축 — 수익 예측, FCF 빌드, WACC, 영구 가치, Bear/Base/Bull 시나리오, 5x5 민감도..."
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Dcf Model

Excel에서 기관 수준의 DCF 평가 모델을 구축합니다 — 수익 예측, FCF(잉여현금흐름) 빌드, WACC(가중평균자본비용), 영구 가치, Bear/Base/Bull 시나리오, 5x5 민감도 분석표. excel-author와 함께 사용합니다. 내재 가치 주식 분석에 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/finance/dcf-model`을 사용하여 설치 |
| 경로 | `optional-skills/finance/dcf-model` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (adapted by Nous Research) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `finance`, `valuation`, `dcf`, `excel`, `openpyxl`, `modeling`, `investment-banking` |
| 관련 스킬 | [`excel-author`](/docs/user-guide/skills/optional/finance/finance-excel-author), [`pptx-author`](/docs/user-guide/skills/optional/finance/finance-pptx-author), [`comps-analysis`](/docs/user-guide/skills/optional/finance/finance-comps-analysis), [`lbo-model`](/docs/user-guide/skills/optional/finance/finance-lbo-model), [`3-statement-model`](/docs/user-guide/skills/optional/finance/finance-3-statement-model) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

## 환경

이 스킬은 **헤드리스(headless) openpyxl**을 가정합니다 — 디스크에 .xlsx 파일을 생성합니다.
셀 색상, 수식, 이름이 지정된 범위 및 민감도 분석표에 대한 `excel-author` 스킬의 규칙을 따르세요.
전달하기 전에 다시 계산하세요: `python /path/to/excel-author/scripts/recalc.py ./out/model.xlsx`.

# DCF 모델 빌더

## 개요

이 스킬은 투자 은행 표준을 따르는 주식 가치 평가를 위한 기관 수준의 DCF 모델을 생성합니다. 각 분석은 상세한 Excel 모델(DCF 시트 하단에 민감도 분석 포함)을 생성합니다.

## 도구

- 기본적으로 사용자가 제공한 모든 정보와 데이터 소싱에 사용할 수 있는 MCP 서버를 사용합니다.

## 핵심 제약 조건 - 이 내용을 먼저 읽으세요

이 제약 조건은 모든 DCF 모델 구축 전체에 적용됩니다. 시작하기 전에 검토하세요:

**하드코드보다 수식 우선 (협상 불가):**
- 모든 예측, 마진, 할인 요소, PV(현재 가치) 및 민감도 셀은 반드시 라이브 Excel 수식이어야 합니다 — Python에서 계산하여 숫자로 기록한 값이면 절대 안 됩니다.
- openpyxl을 사용할 때: `ws["D20"] = "=D19*(1+$B$8)"`는 올바르며, `ws["D20"] = calculated_revenue`는 **잘못된 것**입니다.
- 허용되는 유일한 하드코딩 숫자는 (1) 원시 과거 입력 데이터, (2) 가정 동인(성장률, WACC 입력, 영구 성장률 g), (3) 현재 시장 데이터(주가, 부채 잔액)뿐입니다.
- Python에서 무언가를 계산하고 그 결과를 기록하고 있는 자신을 발견한다면 — **멈추세요**. 사용자가 가정을 변경할 때 모델이 유연하게 반응해야 합니다.

**사용자와 함께 단계별 확인 (처음부터 끝까지 한 번에 빌드하지 마세요):**
- 데이터 검색 후 → 사용자에게 원시 입력 블록(수익, 마진, 주식 수, 순부채)을 보여주고 예측하기 전에 확인합니다.
- 수익 예측 후 → 예상되는 최상위 라인과 성장률을 보여주고, 마진 빌드를 구축하기 전에 확인합니다.
- FCF 빌드 후 → 전체 FCF 일정을 보여주고, WACC를 계산하기 전에 논리를 확인합니다.
- WACC 후 → 계산 및 입력을 보여주고, 할인하기 전에 확인합니다.
- 영구 가치 + PV 후 → 자본 브릿지(기업 가치(EV) → 자본 가치 → 주당 가치)를 보여주고, 민감도 분석표 전에 확인합니다.
- 각 단계에서 오류 포착 — 민감도 분석표가 구축된 후에 발견된 잘못된 마진 가정은 하위의 모든 것을 다시 구축해야 함을 의미합니다.

**민감도 분석표:**
- **홀수 개의 행과 열을 사용하세요** (표준: 5×5, 때로는 7×7) — 이는 진정한 중심 셀을 보장합니다.
- **중심 셀 = 기본(base) 케이스.** 중간 행 헤더와 중간 열 헤더가 모델의 실제 가정과 정확히 일치하도록 축 값을 구성하세요 (예: 기본 WACC = 9.0%이면 중간 행은 9.0%이고, 영구 g = 3.0%이면 중간 열은 3.0%입니다). 따라서 중심 셀의 출력은 모델의 실제 내재 주가와 일치해야 합니다 — 이것이 표가 올바르게 구축되었는지 확인하는 온전성 검사입니다.
- 기본 케이스가 어떤 셀인지 즉시 알 수 있도록 중간 파란색 채우기(`#BDD7EE`) + 굵은 글꼴로 **중심 셀을 강조**하세요.
- 모든 셀(일반적으로 3개 표 × 25개 셀 = 75개)을 전체 DCF 재계산 수식으로 채우세요.
- openpyxl 루프를 사용하여 프로그래밍 방식으로 수식을 작성하세요.
- 자리 표시자 텍스트 없음, 선형 근사값 없음, 수동 단계 필요 없음.
- 각 셀은 해당 가정 조합에 대해 전체 DCF를 다시 계산해야 합니다.

**셀 주석:**
- 각 하드코드된 값이 생성될 **때** 셀 주석을 추가하세요.
- 형식: "Source: [System/Document], [Date], [Reference], [URL if applicable]"
- 다음 섹션으로 이동하기 전에 모든 파란색 입력에 주석이 있어야 합니다.
- 끝으로 미루거나 "TODO: add source"라고 쓰지 마세요.

**모델 레이아웃 계획:**
- 수식을 작성하기 **전에** 모든 섹션 행 위치를 정의하세요.
- 모든 헤더와 레이블을 먼저 작성하세요.
- 모든 섹션 구분선과 빈 행을 두 번째로 작성하세요.
- **그런 다음** 고정된 행 위치를 사용하여 수식을 작성하세요.
- 생성 직후 수식을 테스트하세요.

**수식 재계산:**
- 전달하기 전에 `python recalc.py model.xlsx 30`을 실행하세요.
- 상태가 "success"가 될 때까지 모든 오류를 수정하세요.
- 수식 오류(#REF!, #DIV/0!, #VALUE! 등)가 0개여야 합니다.

**시나리오 블록:**
- Bear/Base/Bull 케이스에 대해 별도의 블록을 만드세요.
- 각 블록 내에서 예측 연도에 걸쳐 가정을 가로로 표시하세요.
- IF 수식을 사용하세요: `=IF($B$6=1,[Bear cell],IF($B$6=2,[Base cell],[Bull cell]))`
- 수식이 올바른 시나리오 블록 셀을 참조하는지 확인하세요.

## DCF 프로세스 워크플로우

### 1단계: 데이터 검색 및 검증

MCP 서버, 사용자 제공 데이터 및 웹에서 데이터를 가져옵니다.

**데이터 소스 우선순위:**
1. **MCP 서버** (구성된 경우) - Daloopa와 같은 공급자의 구조화된 재무 데이터
2. **사용자 제공 데이터** - 조사를 통한 과거 재무 데이터
3. **웹 검색/가져오기** - 필요할 때 현재 가격, 베타, 부채 및 현금

**검증 체크리스트:**
- 순부채 vs 순현금 확인 (가치 평가에 중요)
- 희석 발행 주식 수 확인 (최근 자사주 매입/발행 확인)
- 과거 마진이 비즈니스 모델과 일치하는지 검증
- 업계 벤치마크와 수익 성장률 교차 확인
- 세율이 합리적인지 확인 (일반적으로 21-28%)

### 2단계: 과거 분석 (3-5년)

분석 및 문서화:
- **수익 성장 추세**: 연평균성장률(CAGR) 계산, 동인 식별
- **마진 진행 상황**: 매출총이익률(Gross margin), 영업이익률(EBIT margin), FCF 마진 추적
- **자본 집약도**: 수익 대비 감가상각비(D&A) 및 자본적 지출(CapEx) 비율
- **운전자본 효율성**: 수익 성장률 대비 순운전자본(NWC) 변동 비율
- **수익성 지표**: 투하자본수익률(ROIC), 자기자본이익률(ROE) 추세

다음을 보여주는 요약 표 작성:
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

### 3단계: 수익 예측 구축

**방법론:**
1. 최신 실제 수익(최근 12개월(LTM) 또는 가장 최근 회계 연도)에서 시작합니다.
2. 각 예측 연도에 성장률을 적용합니다.
3. 달러 금액과 계산된 성장률(%)을 모두 표시합니다.

**성장률 프레임워크:**
- 1-2년차: 단기 가시성을 반영한 더 높은 성장
- 3-4년차: 업계 평균을 향한 점진적인 둔화
- 5년차 이상: 영구 성장률에 접근

**수식 구조:**
- 수익(N년차) = 수익(N-1년차) × (1 + 성장률)
- 성장률 %(N년차) = 수익(N년차) / 수익(N-1년차) - 1

**3가지 시나리오 접근법:**
```
Bear Case: 보수적 성장 (예: 8-12%)
Base Case: 가장 가능성 높은 시나리오 (예: 12-16%)
Bull Case: 낙관적 성장 (예: 16-20%)
```

### 4단계: 영업 비용 모델링

**고정/변동 비용 분석:**

영업 비용은 현실적인 영업 레버리지를 모델링해야 합니다:
- **영업 및 마케팅(S&M)**: 비즈니스 모델에 따라 일반적으로 수익의 15-40%
- **연구 및 개발(R&D)**: 기술 회사의 경우 일반적으로 10-30%
- **일반 및 관리(G&A)**: 수익의 일반적으로 8-15%, 회사가 확장됨에 따라 레버리지 표시

**핵심 원칙:**
- 매출총이익이 아닌 **수익(REVENUE)**에 기반한 모든 비율
- 영업 레버리지 모델링: 수익이 증가함에 따라 비율이 감소해야 함
- S&M, R&D, G&A에 대한 별도의 라인 항목 유지
- EBIT(영업이익) = 매출총이익 - 총 영업비용(OpEx) 계산

**마진 확장 프레임워크:**
```
현재 상태 → 목표 상태 (5년차)
매출총이익률(Gross Margin): X% → Y% (규모, 효율성 기반으로 정당화)
영업이익률(EBIT Margin): X% → Y% (수익 성장 + 영업비용 레버리지의 결과)
```

### 5단계: 잉여현금흐름(FCF) 계산

**올바른 순서로 FCF 구축:**

```
EBIT
(-) 세금 (EBIT × 세율)
= NOPAT (세후 순영업이익)
(+) D&A (비현금성 비용, 수익의 %)
(-) CapEx (수익의 %, 일반적으로 4-8%)
(-) Δ NWC (운전자본 변동)
= Unlevered Free Cash Flow (무차입 잉여현금흐름)
```

**운전자본 모델링:**
- 수익 변동(delta revenue)의 %로 계산
- 일반적인 범위: 수익 변동의 -2% ~ +2%
- 음수 = 현금의 원천 (운전자본 방출)
- 양수 = 현금의 사용 (운전자본 축적)

**유지보수 vs 성장 CapEx:**
- 유지보수 CapEx: 현재 운영 유지 (수익의 ~2-3%)
- 성장 CapEx: 확장 지원 (수익의 추가 2-5%)
- 총 CapEx는 회사의 성장 전략과 일치해야 함

### 6단계: 자본 비용(WACC) 연구

**자기자본비용(Cost of Equity)을 위한 CAPM 방법론:**

```
자기자본비용 = 무위험 수익률 + 베타 × 주식 위험 프리미엄

여기서:
- 무위험 수익률(Risk-Free Rate) = 현재 10년 만기 국채 수익률
- 베타 = 시장 지수 대비 5년 월별 주식 베타
- 주식 위험 프리미엄 = 5.0-6.0% (시장 표준)
```

**타인자본비용(Cost of Debt) 계산:**

```
세후 타인자본비용 = 세전 타인자본비용 × (1 - 세율)

세전 타인자본비용 결정 기준:
- 신용 등급 (가능한 경우)
- 회사 채권의 현재 수익률
- 재무제표의 이자 비용 / 총 부채
```

**자본 구조 가중치:**

```
자기자본 시장 가치(Market Value Equity) = 현재 주가 × 발행 주식 수
순부채(Net Debt) = 총 부채 - 현금 및 현금성 자산
기업 가치(Enterprise Value) = 시가총액 + 순부채

자기자본 가중치 = 시가총액 / 기업 가치
부채 가중치 = 순부채 / 기업 가치

WACC = (자기자본비용 × 자기자본 가중치) + (세후 타인자본비용 × 부채 가중치)
```

**특수 케이스:**
- **순현금(Net Cash) 위치**: 현금 > 부채인 경우, 순부채는 음수(NEGATIVE)입니다.
  - 부채 가중치가 음수일 수 있습니다.
  - WACC 계산이 그에 따라 조정됩니다.
- **부채 없음**: WACC = 자기자본비용

**일반적인 WACC 범위:**
- 대형주, 안정적: 7-9%
- 성장 기업: 9-12%
- 고성장/고위험: 12-15%

### 7단계: 할인율 적용 (5-10년 예측)

**기중(Mid-Year) 관행:**
- 현금 흐름이 연중(mid-year)에 발생한다고 가정
- 할인 기간: 0.5, 1.5, 2.5, 3.5, 4.5 등
- 할인 요소 = 1 / (1 + WACC)^기간

**현재 가치(PV) 계산:**
```
각 예측 연도에 대해:
FCF의 PV = 무차입 FCF × 할인 요소

예시 (1년차):
FCF = $1,000
WACC = 10%
기간 = 0.5
할인 요소 = 1 / (1.10)^0.5 = 0.9535
PV = $1,000 × 0.9535 = $954
```

**예측 기간 선택:**
- **5년**: 대부분의 분석에 대한 표준
- **7-10년**: 활주로(runway)가 긴 고성장 기업
- **3년**: 성숙하고 안정적인 비즈니스

### 8단계: 영구 가치(Terminal Value) 계산

**영구 성장법 (권장):**

```
영구 FCF = 마지막 연도 FCF × (1 + 영구 성장률)
영구 가치 = 영구 FCF / (WACC - 영구 성장률)

핵심 제약 조건: 영구 성장률 < WACC (그렇지 않으면 가치가 무한대가 됨)
```

**영구 성장률 선택:**
- 보수적: 2.0-2.5% (GDP 성장률)
- 중간: 2.5-3.5%
- 공격적: 3.5-5.0% (시장 선도 기업에 한함)

**초과 금지**: 무위험 수익률 또는 장기 GDP 성장률

**종료 배수법(Exit Multiple Method) (대안):**
```
영구 가치 = 마지막 연도 EBITDA × 종료 배수

종료 배수의 출처:
- 업계 비교 가능한 거래 배수
- 선행 거래 배수
- 일반적인 범위: 8-15x EBITDA
```

**영구 가치의 현재 가치:**
```
영구 가치의 PV = 영구 가치 / (1 + WACC)^마지막 기간

마지막 기간이 타이밍을 설명하는 곳:
기중 관행이 있는 5년 모델: 기간 = 4.5
```

**영구 가치 온전성 검사:**
- 기업 가치의 50-70%를 차지해야 합니다.
- 75%를 초과하는 경우, 모델이 영구 가정에 지나치게 의존할 수 있습니다.
- 40% 미만인 경우, 영구 가정이 너무 보수적인지 확인하세요.

### 9단계: 기업 가치에서 자기자본 가치로의 브릿지

**가치 평가 요약 구조:**

```
(+) 예상 FCF의 PV 합계 = $X million
(+) 영구 가치의 PV = $Y million
= 기업 가치(Enterprise Value) = $Z million

(-) 순부채 [또는 음수인 경우 + 순현금] = $A million
= 자기자본 가치(Equity Value) = $B million

÷ 희석 발행 주식 수 = C million shares
= 내재 주당 가격(Implied Price per Share) = $XX.XX

현재 주가 = $YY.YY
내재 수익률(Implied Return) = (내재 가격 / 현재 가격) - 1 = XX%
```

**핵심 조정:**
- **순부채 = 총 부채 - 현금 및 현금성 자산**
  - 양수인 경우: EV에서 차감 (자기자본 가치 감소)
  - 음수인 경우 (순현금): EV에 가산 (자기자본 가치 증가)
- **희석 주식 수 사용**: 옵션, RSU, 전환 증권 포함
- **기타 조정** (해당하는 경우):
  - 소수 주주 지분 (Minority interests)
  - 연금 부채
  - 운용 리스 의무

**가치 평가 출력 형식:**
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

DCF 시트 하단에 가정이 변경됨에 따라 가치 평가가 어떻게 변하는지 보여주는 **세 가지 민감도 분석표**를 구축합니다:

1. **WACC vs 영구 성장률(Terminal Growth)** - 할인율 및 영구 성장에 대한 기업 가치 민감도 표시
2. **수익 성장률 vs EBIT 마진** - 최상위 라인 성장 및 영업 레버리지의 영향 표시
3. **베타 vs 무위험 수익률** - 자기자본비용 구성요소에 대한 민감도 표시

**구현**: 이는 Excel의 "데이터 표(Data Table)" 기능이 **아닙니다**. 각 셀에 수식이 있는 간단한 2D 그리드입니다. 각 셀에는 해당 특정 가정 조합에 대한 전체 DCF 재계산이 포함되어야 합니다. openpyxl을 사용하여 75개 셀 모두를 프로그래밍 방식으로 채우는 것에 대한 자세한 요구 사항은 '핵심 제약 조건' 섹션을 참조하세요.

<correct_patterns>

이 섹션에는 DCF 모델을 구축할 때 따라야 할 모든 **올바른** 패턴이 포함되어 있습니다.

### 시나리오 블록 선택 패턴 - 이 접근 방식을 따르세요

**가정은 각 시나리오에 대해 별도의 블록으로 구성됩니다:**

**중요 구조 - 섹션 헤더당 3개 행:**

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

**각 시나리오 블록에는** 섹션 제목 바로 아래에 예측 연도(FY2025E, FY2026E 등)를 보여주는 **열 헤더 행이 있어야 합니다.** 이것이 없으면 사용자는 어떤 가정 값이 어느 연도에 해당하는지 알 수 없습니다.

**가정을 참조하는 방법 - 통합(consolidation) 열 만들기:**
1. 케이스 선택기 셀 (예: B6)에는 1=Bear, 2=Base, 또는 3=Bull이 포함됩니다.
2. 올바른 시나리오 블록에서 가져오기 위해 INDEX 또는 OFFSET 수식을 사용하여 통합 열을 만듭니다.
3. 예측 수식은 통합 열을 참조합니다 (깔끔한 셀 참조).
4. 각 시나리오 블록에는 예측 연도에 걸친 전체 DCF 가정 세트가 포함됩니다.

**권장 통합 열 패턴 (INDEX 사용):**
`=INDEX(B10:D10, 1, $B$6)`

**이렇게 하지 마세요 - 전체에 분산된 IF 문:**
`=IF($B$6=1,[Bear block cell],IF($B$6=2,[Base block cell],[Bull block cell]))`

통합 열 접근 방식은 논리를 중앙 집중화하고 모델을 감사하기 쉽게 만듭니다.

### 올바른 수익 예측 패턴

**INDEX 수식으로 통합 열을 만든 다음, 예측에서 이를 참조하세요:**

**1단계 - 1년차(FY1) 성장을 위한 통합 열:**
`=INDEX([Bear FY1 growth]:[Bull FY1 growth], 1, $B$6)`

**2단계 - 수익 예측이 통합 열을 참조함:**
`Revenue Year 1: =D29*(1+$E$10)`

여기서:
- D29 = 전년도 수익
- $E$10 = FY1 성장을 위한 통합 열 셀 (INDEX 수식 포함)
- $B$6 = 케이스 선택기 (1=Bear, 2=Base, 3=Bull)

**이 접근 방식은 모든 예측 수식에 IF 문을 포함하는 것보다 깔끔하며** 어떤 시나리오 가정이 사용되고 있는지 감사하기가 훨씬 쉽습니다.

### 올바른 FCF 수식 패턴

**INDEX 수식으로 통합 열을 사용한 다음, FCF 계산에서 이를 참조하세요:**

**통합 열 접근 방식:**
```csv
Item,Formula,Reference
D&A,=E29*$E$21,$E$21 = consolidation column for D&A %
CapEx,=E29*$E$22,$E$22 = consolidation column for CapEx %
Δ NWC,=(E29-D29)*$E$23,$E$23 = consolidation column for NWC %
Unlevered FCF,=E57+E58-E60-E62,E57=NOPAT E58=D&A E60=CapEx E62=Δ NWC
```

**각 통합 열 셀에는** 케이스 선택기에 기반하여 적절한 시나리오 블록에서 가져오는 **INDEX 수식이 포함되어 있습니다.** 이렇게 하면 예측 수식이 깔끔하고 감사 가능하게 유지됩니다.

수식을 작성하기 전에 시나리오 블록 행 위치를 확인하고 통합 열을 설정하세요.

### 올바른 셀 주석 형식

**모든 하드코드된 값에는 다음 형식이 필요합니다:**

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

### 올바른 가정 표 구조

**중요: 각 시나리오 블록에는 3가지 구조적 요소가 필요합니다:**

1. **섹션 헤더 행** (병합된 셀): 예: "BEAR CASE ASSUMPTIONS"
2. **연도를 보여주는 열 헤더 행** - 이것은 필수입니다, 건너뛰지 마세요
3. 가정 값이 있는 **데이터 행**

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

**예측 연도(FY2025E, FY2026E 등)를 보여주는 열 헤더 행이 없으면 사용자는 어떤 가정 값이 어느 연도에 해당하는지 알 수 없습니다. 이 행은 필수(MANDATORY)입니다.**

**그런 다음 통합 열을 만드세요** (일반적으로 오른쪽 다음 열). 케이스 선택기에 따라 선택한 시나리오 블록에서 가져오는 INDEX 수식을 사용합니다. 이 통합 열이 예측 수식에서 참조하는 것입니다.

### 올바른 행 계획 프로세스

**1. 모든 헤더와 레이블을 먼저 작성하세요:**
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

**2. 모든 섹션 구분선과 빈 행을 작성하세요**

**3. 그런 다음 고정된 행 위치를 사용하여 수식을 작성하세요**

**4. 생성 직후 수식을 테스트하세요**

**건축과 같다고 생각하세요:**
- 좋음: 기초를 붓고 벽을 쌓습니다 (안정적인 구조)
- 나쁨: 벽을 짓고 기초를 붓습니다 (벽이 무너짐)

**Excel 버전:**
- 좋음: 헤더를 추가한 다음 수식을 작성합니다 (수식이 안정적임)
- 나쁨: 수식을 작성한 다음 헤더를 추가합니다 (수식이 깨짐)

### 올바른 민감도 분석표 구현

**중요**: 이것은 Excel의 "데이터 표(Data Table)" 기능이 아닙니다. 이것은 openpyxl을 사용하여 일반 수식을 작성하는 간단한 그리드입니다. 예, 총 75개의 수식(3개 표 × 각 25개 셀)을 의미하지만, 이는 직관적이고 필수적입니다.

**수식을 사용한 프로그래밍 방식 채우기:**

각 민감도 분석표는 각 가정 조합에 대한 내재 주가를 다시 계산하는 수식으로 완전히 채워져야 합니다. **Excel의 데이터 표(Data Table) 기능을 사용하지 마세요** (수동 개입이 필요하며 openpyxl을 통해 자동화할 수 없습니다).

**구현 접근 방식 - 구체적인 예:**

**표 구조 — 5×5 그리드 (홀수 크기, 기본 케이스 중앙 배치):**

모델의 기본 WACC = 9.0%이고 기본 영구 성장률 = 3.0%인 경우, 해당 값을 중심으로 축을 대칭으로 구축합니다:

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

**★ = 중심 셀.** 이 수식의 출력은 반드시 모델의 실제 내재 주가(가치 평가 요약에서 가져옴)와 같아야 합니다. 이 셀에 중간 파란색 채우기(`#BDD7EE`)와 굵은 글꼴을 적용하여 기본 케이스가 시각적으로 고정되도록 하세요.

**축 값에 대한 규칙:** `axis_values = [base - 2*step, base - step, base, base + step, base + 2*step]` — 기본값을 중심으로 대칭이며, 홀수 개수이므로 중심이 보장됩니다.

**수식 패턴 - B88 셀 (WACC=8.0%, 영구 성장률=2.0%):**

B88의 수식은 다음을 사용하여 내재 가격을 다시 계산해야 합니다:
- 행 헤더의 WACC: `$A88` (8.0%)
- 열 헤더의 영구 성장률: `B$87` (2.0%)

**권장 접근 방식:** 메인 DCF 계산을 참조하되 이러한 값을 대체합니다.

**예시 수식 구조:**
`=([SUM of PV FCFs using $A88 as discount rate] + [Terminal Value using B$87 as growth rate and $A88 as WACC] - [Net Debt]) / [Shares]`

**중요 - 5x5 그리드의 모든 셀에 대해 수식을 작성하세요 (표당 25개 셀, 총 75개 셀).** openpyxl을 사용하여 루프에서 프로그래밍 방식으로 이러한 수식을 작성하세요. 이 단계를 건너뛰거나 자리 표시자 텍스트를 남기지 마세요.

**Python 구현 패턴:**
```python
# Pseudocode for populating sensitivity table
for row_idx, wacc_value in enumerate(wacc_range):
    for col_idx, term_growth_value in enumerate(term_growth_range):
        # Build formula that uses wacc_value and term_growth_value
        formula = f"=<DCF recalc using {wacc_value} and {term_growth_value}>"
        ws.cell(row=start_row+row_idx, column=start_col+col_idx).value = formula
```

**민감도 분석표는 사용자의 수동 개입 없이 모델을 열었을 때 즉시 작동해야 합니다.**

</correct_patterns>

<common_mistakes>

이 섹션에는 DCF 모델을 구축할 때 피해야 할 모든 **잘못된** 패턴이 포함되어 있습니다.

### 잘못됨: 단순화된 민감도 표 근사치 또는 자리 표시자 텍스트

**선형 근사값을 사용하지 마세요:**

```
// WRONG - Linear approximation
B97: =B88*(1+(0.096-0.116))    // Assumes linear relationship

// WRONG - Division shortcut
B105: =B88/(1+(E48-0.07))      // Doesn't recalculate full DCF
```

**자리 표시자 텍스트를 남기지 마세요:**
```
// WRONG - Placeholder note
"Note: Use Excel Data Table feature (Data → What-If Analysis → Data Table) to populate sensitivity tables."

// WRONG - Empty cells
[leaving cells blank because "this is complex"]
```

**용어를 혼동하지 마세요:**
- ❌ "민감도 표에는 Excel의 데이터 표 기능이 필요합니다" (아니요 - 그것은 우리가 사용할 수 없는 특정한 Excel 도구입니다)
- ✅ "민감도 표는 각 셀에 수식이 있는 간단한 그리드입니다" (예 - 이것이 우리가 구축하는 것입니다)

**이러한 지름길이 잘못된 이유:**
- 선형 근사값 수식은 실제로 DCF를 다시 계산하지 않습니다 - 단순한 수학적 조정만 적용합니다
- 관계는 선형이 아니므로 결과가 부정확할 것입니다
- 자리 표시자 텍스트는 수동 사용자 개입이 필요합니다
- 제공될 때 모델을 즉시 사용할 수 없습니다
- 전문적이거나 클라이언트가 사용할 수 있는 수준이 아닙니다
- 빈 셀 = 불완전한 결과물

**거부해야 할 일반적인 합리화:**
"75개 이상의 수식을 작성하는 것은 복잡하게 느껴지므로 사용자가 수동으로 완료하도록 메모를 남기겠습니다."

**현실:** openpyxl과 함께 Python에서 루프를 사용하면 75개의 수식을 작성하는 것은 간단합니다. 각 수식은 동일한 패턴을 따릅니다 - 행/열 값만 대체하면 됩니다. 이것은 산출물의 필수 부분입니다.

**대신:** 모든 민감도 셀에 해당 특정 가정 조합에 대한 전체 DCF를 다시 계산하는 수식을 채우세요.

### 잘못됨: 셀 주석 누락

**이렇게 하지 마세요:**
- 주석 없이 모든 하드코드된 입력 생성
- "나중에 추가할게"라고 생각하기
- "TODO: add source" 쓰기
- 파란색 입력을 문서화 없이 두기

**잘못된 이유:**
- 데이터가 어디서 왔는지 확인할 수 없음
- xlsx 스킬 요구 사항 실패
- 감사 준비 안 됨
- 나중에 고치느라 시간 낭비

**대신:** 각 하드코드된 값이 생성될 때 바로 셀 주석을 추가하세요.

### 잘못됨: 수식 행 참조 어긋남

**증상:**
FCF 섹션이 잘못된 가정 행을 참조함:
`D&A:  =E29*$E$34    // Should be $E$21, but referencing wrong row`
`CapEx: =E29*$E$41   // Should be $E$22, but row shifted`

**발생 원인:**
1. 수식을 먼저 작성함
2. 그런 다음 헤더를 삽입함
3. 모든 행 참조가 이동됨
4. 이제 수식이 잘못된 셀을 가리킴 → #REF! 오류

**대신:** 행 레이아웃을 먼저 잠그고 나서 수식을 작성하세요.

### 잘못됨: 여러 시나리오에 걸쳐 하나의 가정에 대한 단일 행

**가정을 다음과 같이 구성하지 마세요:**
```csv
Assumption,Bear,Base,Bull
Revenue Growth FY1,10%,13%,16%
Revenue Growth FY2,9%,12%,15%
```
이러한 수직적 레이아웃은 각 시나리오 내에서 연도별 진행 상황을 보기 어렵게 만듭니다.

**잘못된 이유:**
- 각 시나리오 내에서 연도별로 발전하는 가정을 보기 어렵습니다
- 전체 예측 기간에 걸쳐 시나리오 가정을 비교하기 어렵습니다
- 시나리오 논리를 검토하기에 직관적이지 않습니다

**대신:**
- 각 시나리오(Bear, Base, Bull)에 대해 별도의 블록을 만드세요
- 각 블록 내에서 예측 연도에 걸쳐 가정을 가로로 표시하세요
- 이는 각 시나리오의 가정을 일관된 세트로 검토하기 쉽게 만듭니다

### 잘못됨: 테두리 없음

**테두리 없는 모델을 제공하지 마세요:**
- 섹션 구분이 없음
- 모든 셀이 섞여 보임
- 읽기 어렵고 비전문적임

**잘못된 이유:**
- 클라이언트용으로 준비되지 않음
- 탐색하기 어려움
- 아마추어처럼 보임

**대신:** 모든 주요 섹션 주위에 테두리를 추가하세요.

### 잘못됨: 잘못된 글꼴 색상 또는 글꼴 색상 구분 없음

**이렇게 하지 마세요:**
- 모든 텍스트가 검은색임
- 채우기 색상만 사용함 (글꼴 색상 변경 없음)
- 어떤 셀이 파란색인지 검은색인지 혼합함

**잘못된 이유:**
- 입력과 수식을 구별할 수 없음
- 감사가 불가능해짐
- xlsx 스킬 요구 사항 위반

**대신:** 모든 하드코드된 입력은 파란색 텍스트, 모든 수식은 검은색 텍스트, 시트 링크는 녹색 텍스트 사용

### 잘못됨: 매출총이익에 기반한 영업 비용

**이렇게 하지 마세요:**
`S&M: =E33*0.15    // E33 = Gross Profit (WRONG)`

**잘못된 이유:**
- 영업 비용은 매출총이익이 아닌 수익(revenue)에 비례하여 조정됨
- 비현실적인 마진 진행을 생성함
- 실제 비즈니스 운영 방식이 아님

**대신:**
`S&M: =E29*0.15    // E29 = Revenue (CORRECT)`

### 상위 5가지 오류 요약

1. **수식 행 참조 어긋남** → 수식을 작성하기 전에 모든 행 위치를 정의하세요
2. **셀 주석 누락** → 나중이 아니라 셀이 생성될 때 주석을 추가하세요
3. **단순화된 민감도 표** → 근사값이 아닌 전체 DCF 재계산 수식으로 모든 셀을 채우세요
4. **시나리오 블록 참조 오류** → IF 수식이 올바른 Bear/Base/Bull 블록에서 가져오는지 확인하세요
5. **테두리 없음** → 클라이언트 준비된 외관을 위해 전문적인 섹션 테두리를 추가하세요

추가로 다음 오류를 주의하세요:

### WACC 계산 오류
- 자본 구조에서 장부 가치와 시장 가치 혼합
- 자산/무차입 베타 대신 주식 베타를 잘못 사용
- 타인자본비용에 잘못된 세율 적용
- 잘못된 무위험 수익률 (현재 10년물 국채를 사용해야 함)
- 순현금 상태 대 순부채 상태에 대한 조정 실패

### 성장 가정 결함
- 영구 성장률 > WACC (무한한 가치 생성)
- 과거 성과와 일치하지 않는 예측 성장률
- 산업 성장 제약 무시
- 단위 경제학과 일치하지 않는 수익 성장
- 운영상 정당성 없는 마진 확장

### 영구 가치(Terminal Value) 실수
- 잘못된 성장법 사용 (영구 vs 종료 배수)
- 영구 가치가 기업 가치의 >80% (지나친 의존성 시사)
- 정상 상태 가정과 일치하지 않는 영구 마진
- 영구 가치에 대한 잘못된 할인 기간

### 현금 흐름 예측 오류
- 수익이 아닌 매출총이익에 기반한 영업 비용
- 비즈니스 모델과 일치하지 않는 D&A/CapEx 백분율
- 운전자본 변동이 제대로 계산되지 않음
- 연도 간 세율 불일치
- NOPAT 계산 오류

**이러한 오류가 가장 흔합니다. DCF 빌드를 시작하기 전에 이 섹션을 다시 읽어보세요.**

</common_mistakes>

## Excel 파일 생성

**이 스킬은 모든 스프레드시트 작업에 `xlsx` 스킬을 사용합니다.** xlsx 스킬은 다음을 제공합니다:
- 표준화된 수식 구성 규칙
- 숫자 형식 지정 규칙
- `recalc.py` 스크립트를 통한 자동 수식 재계산
- 포괄적인 오류 검사 및 검증

이 스킬로 생성된 모든 Excel 파일은 제로 수식 오류 및 올바른 재계산을 포함하여 xlsx 스킬 요구 사항을 따라야 합니다.

## 품질 루브릭

모든 DCF 모델은 다음을 극대화해야 합니다:
1. 과거 성과에 기반한 **현실적인 수익 및 마진 가정**
2. 적절한 CAPM 방법론을 사용한 **적절한 자본 비용 계산**
3. 가치 평가 범위를 보여주는 **포괄적인 민감도 분석**
4. 지원 근거가 있는 **명확한 영구 가치 계산**
5. 시나리오 분석을 가능하게 하는 **전문적인 모델 구조**
6. 모든 주요 가정의 **투명한 문서화**

## 입력 요구 사항

### 최소 필수 입력
1. **회사 식별자**: 티커 기호 또는 회사 이름
2. **성장 가정**: 예측 기간의 수익 성장률 (또는 "컨센서스 사용")
3. **선택적 매개변수**:
   - 예측 기간 (기본값: 5년)
   - 시나리오 케이스 (Bear/Base/Bull 성장 및 마진 가정)
   - 영구 성장률 (기본값: 2.5-3.0%)
   - CAPM을 사용하지 않는 경우 특정 WACC 입력

## Excel 모델 구조

### 시트 아키텍처

**두 개의 시트** 생성:

1. **DCF** - 하단에 민감도 분석이 있는 메인 가치 평가 모델
2. **WACC** - 자본 비용 계산

**중요**: 민감도 표는 (별도의 시트가 아닌) DCF 시트의 맨 아래에 위치합니다. 이렇게 하면 모든 가치 평가 출력이 함께 유지됩니다.

### 수식 재계산 (필수)

Excel 모델을 생성하거나 수정한 후에는 `excel-author` 스킬의 `recalc.py` 스크립트를 사용하여 **모든 수식을 다시 계산**하세요:

```bash
python recalc.py [path_to_excel_file] [timeout_seconds]
```

예시:
```bash
python recalc.py AAPL_DCF_Model_2025-10-12.xlsx 30
```

이 스크립트는 다음을 수행합니다:
- LibreOffice를 사용하여 모든 시트의 모든 수식을 다시 계산합니다.
- Excel 오류(#REF!, #DIV/0!, #VALUE!, #NAME?, #NULL!, #NUM!, #N/A)에 대해 모든 셀을 검사합니다.
- 오류 위치 및 개수가 포함된 자세한 JSON을 반환합니다.

**예상 출력 형식:**
```json
{
  "status": "success",           // or "errors_found"
  "total_errors": 0,              // Total error count
  "total_formulas": 42,           // Number of formulas in file
  "error_summary": {}             // Only present if errors found
}
```

**오류가 발견되면** 출력에 세부 정보가 포함됩니다:
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

모델을 전달하기 전에 상태가 "success"가 될 때까지 **모든 오류를 수정**하고 recalc.py를 다시 실행하세요.

### 서식 표준

**중요**: 수식 구성 규칙 및 숫자 형식 규칙은 xlsx 스킬을 따르세요. DCF 스킬은 특정한 시각적 프레젠테이션 표준을 추가합니다.

**색상 구성표 - 2개 레이어**:

**레이어 1: 글꼴 색상 (xlsx 스킬에서 필수)**
- **파란색 텍스트 (RGB: 0,0,255)**: 모든 하드코드된 입력 (주가, 주식, 과거 데이터, 가정)
- **검은색 텍스트 (RGB: 0,0,0)**: 모든 수식 및 계산
- **녹색 텍스트 (RGB: 0,128,0)**: 다른 시트로의 링크 (WACC 시트 참조)

**레이어 2: 채우기 색상 — 전문적인 파란색/회색 팔레트 (사용자가 달리 지정하지 않는 한 기본값)**
- **최소화 유지** — 채우기에는 파란색과 회색만 사용하세요. 녹색, 노란색, 주황색 또는 여러 강조 색상을 도입하지 마세요. 너무 많은 색상이 있는 모델은 아마추어처럼 보입니다.
- **기본 채우기 팔레트:**
  - **섹션 헤더**: 짙은 파란색 (RGB: 31,78,121 / `#1F4E79`) 배경에 흰색 굵은 텍스트
  - **하위 헤더/열 헤더**: 밝은 파란색 (RGB: 217,225,242 / `#D9E1F2`) 배경에 검은색 굵은 텍스트
  - **입력 셀**: 밝은 회색 (RGB: 242,242,242 / `#F2F2F2`) 배경에 파란색 글꼴 — 또는 완전한 미니멀리즘을 원하면 그냥 흰색 배경에 파란색 글꼴
  - **계산된 셀**: 흰색 배경에 검은색 글꼴
  - **출력/요약 행** (주당 가치, EV 등): 중간 파란색 (RGB: 189,215,238 / `#BDD7EE`) 배경에 검은색 굵은 글꼴
- **이게 전부입니다 — 파란색 3개 + 회색 1개 + 흰색.** 더 추가하려는 충동을 억제하세요.
- 사용자가 제공한 템플릿이나 명시적인 색상 기본 설정은 항상 이러한 기본값보다 우선합니다.

**레이어가 함께 작동하는 방식:**
- 입력 셀: 파란색 글꼴 + 밝은 회색 채우기 = "하드코드된 입력"
- 수식 셀: 검은색 글꼴 + 흰색 배경 = "계산된 값"
- 시트 링크: 녹색 글꼴 + 흰색 배경 = "다른 시트에서의 참조"
- 주요 출력: 검은색 굵은 글꼴 + 중간 파란색 채우기 = "이것이 답입니다"

**글꼴 색상은 그것이 무엇인지(입력/수식/링크)를 알려줍니다. 채우기 색상은 현재 위치(헤더/데이터/출력)를 알려줍니다.**

### 테두리 표준 (전문적인 외관을 위해 필수)

주요 섹션 주변의 **굵은 테두리** (1.5pt):
- KEY INPUTS 섹션
- PROJECTION ASSUMPTIONS 섹션
- 5-YEAR CASH FLOW PROJECTION 섹션
- TERMINAL VALUE 섹션
- VALUATION SUMMARY 섹션
- 각 SENSITIVITY ANALYSIS 표

하위 섹션 간의 **중간 테두리** (1pt):
- Company Details vs Historical Performance
- Growth Assumptions vs EBIT Margin vs FCF Parameters

데이터 표 주변의 **얇은 테두리** (0.5pt):
- 시나리오 가정 표 (Bear | Base | Bull | Selected)
- Historical vs projected financials 행렬

**테두리 없음:** 테이블 내의 개별 셀 (깔끔하고 스캔하기 쉽게 유지)

**테두리는 필수입니다** - 전문적인 테두리가 없는 모델은 클라이언트용으로 준비되지 않은 것입니다.

**숫자 형식** (xlsx 스킬 표준을 따름):
- **연도**: 텍스트 문자열로 서식 지정 (예: "2,024"가 아닌 "2024")
- **백분율**: `0.0%` (소수점 한 자리)
- **통화**: 수백만 단위의 경우 `$#,##0`; 주당 가격의 경우 `$#,##0.00` - 항상 헤더에 단위를 지정하세요 ("Revenue ($mm)")
- **0(Zeros)**: 숫자 서식을 사용하여 모든 0을 "-"로 만드세요 (예: `$#,##0;($#,##0);-`)
- **큰 숫자**: 천 단위 구분 기호가 있는 `#,##0`
- **음수**: 괄호 안의 `(#,##0)` (마이너스 기호가 아님)

**셀 주석 (모든 하드코드된 입력에 필수)**:

xlsx 스킬에 따라 모든 하드코드된 값에는 소스를 문서화하는 셀 주석이 있어야 합니다. 형식: "Source: [System/Document], [Date], [Reference], [URL if applicable]"

**중요**: 셀이 생성될 때 주석을 추가하세요. 끝으로 미루지 마세요.

### DCF 시트 세부 구조

**섹션 1: 헤더**
```csv
Row,Content
1,[Company Name] DCF Model
2,Ticker: [XXX] | Date: [Date] | Year End: [FYE]
3,Blank
4,Case Selector Cell (1=Bear 2=Base 3=Bull)
5,Case Name Display (formula: =IF([Selector]=1"Bear"IF([Selector]=2"Base""Bull")))
```

**섹션 2: 시장 데이터 (케이스에 종속되지 않음)**
```csv
Item,Value
Current Stock Price,$XX.XX
Shares Outstanding (M),XX.X
Market Cap ($M),[Formula]
Net Debt ($M),XXX [or Net Cash if negative]
```

**섹션 3: DCF 시나리오 가정**

각 시나리오(Bear, Base, Bull)에 대해 별도의 가정 블록을 생성하고, 예측 연도에 걸쳐 가로로 배치된 DCF 관련 가정(Revenue Growth %, EBIT Margin %, Tax Rate %, D&A % of Revenue, CapEx % of Revenue, NWC Change % of ΔRev, Terminal Growth Rate, WACC)을 만듭니다. 각 블록에는 섹션 헤더, 예측 연도(FY1, FY2 등)를 보여주는 열 헤더 행 및 데이터 행이 포함되어야 합니다. 정확한 레이아웃은 `<correct_patterns>` 섹션의 "올바른 가정 표 구조"를 참조하세요.

**섹션 4: 과거 및 예상 재무제표**

모든 예측 행에 분산된 IF 수식이 아니라, **시나리오 블록에서 가져오는 통합 열(예: "Selected Case")을 참조하세요.**

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

**핵심 수식 패턴**:
- 수익 성장률: `=E29*(1+$E$10)` (여기서 $E$10은 1년차 성장을 위한 통합 열입니다)
- **아님**: `=E29*(1+IF($B$6=1,$B$10,IF($B$6=2,$C$10,$D$10)))`

이 접근 방식은 더 깔끔하고, 감사하기 쉬우며, 시나리오 논리를 중앙 집중화하여 수식 오류를 방지합니다.

**섹션 5: 잉여현금흐름(FCF) 빌드**

**중요**: 행 참조가 **올바른** 가정 행을 가리키는지 확인하세요. 생성 직후 수식을 테스트하세요.

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

**행 참조 예시** (레이아웃 계획 기준):
- $E$21 = D&A % 가정 (통합 열, 행 21)
- $E$22 = CapEx % 가정 (통합 열, 행 22)
- $E$23 = NWC % 가정 (통합 열, 행 23)
- E29 = 해당 연도의 수익 (행 29)
- E45 = 해당 연도의 NOPAT (행 45)

**수식을 작성하기 전에**: 이 행 번호가 실제 레이아웃과 일치하는지 확인하세요. 한 열을 테스트한 다음 복사하여 가로로 붙여넣으세요.

**섹션 6: 할인 및 가치 평가**
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

**주요 WACC 수식:**
```
Market Cap = Price × Shares
Net Debt = Total Debt - Cash
Enterprise Value = Market Cap + Net Debt
Equity Weight = Market Cap / EV
Debt Weight = Net Debt / EV
WACC = (Cost of Equity × Equity Weight) + (After-tax Cost of Debt × Debt Weight)
```

### 민감도 분석 (DCF 시트 하단)

**용어 알림**: "민감도 표(Sensitivity tables)" = 행 헤더, 열 헤더 및 각 데이터 셀의 수식이 있는 간단한 2D 그리드입니다. Excel의 "데이터 표(Data Table)" 기능(데이터 → 가상 분석 → 데이터 표)이 **아닙니다**. openpyxl을 사용하여 일반 Excel 수식을 각 셀에 작성해야 합니다.

**위치**: DCF 시트의 행 87+ (별도의 시트 아님)

**수직으로 쌓인 세 개의 민감도 분석표:**

1. **WACC vs Terminal Growth** (행 87-100) - 5x5 그리드 = 25개 수식 셀
2. **Revenue Growth vs EBIT Margin** (행 102-115) - 5x5 그리드 = 25개 수식 셀
3. **Beta vs Risk-Free Rate** (행 117-130) - 5x5 그리드 = 25개 수식 셀

**작성해야 할 총 수식 수: 75** (이것은 필수이며 선택 사항이 아닙니다)

**중요**: 모든 민감도 표 셀은 openpyxl을 사용하여 프로그래밍 방식으로 수식으로 채워져야 합니다. 선형 근사값 지름길을 사용하지 **마세요**. 수동 단계에 대한 자리 표시자 텍스트나 메모를 남기지 **마세요**. "복잡하다"는 이유로 셀을 비워두는 것을 합리화하지 마세요 - Python 루프를 사용하여 수식을 생성하세요.

**표 설정:**
1. 행/열 헤더(테스트할 가정 값)가 있는 표 구조 생성
2. 다음을 수행하는 수식으로 **모든** 데이터 셀 채우기:
   - 행 헤더 값 사용 (예: WACC = 9.0%)
   - 열 헤더 값 사용 (예: 영구 성장률 = 3.0%)
   - 특정 가정을 사용하여 전체 DCF 재계산
   - 해당 시나리오에 대한 내재 주가 반환
3. 전달될 때 모든 셀에 작동하는 수식이 포함되어야 함
4. 조건부 서식으로 셀 형식 지정: 더 높은 값은 녹색 스케일, 더 낮은 값은 빨간색 스케일
5. 기본 케이스 셀 굵게 표시
6. 표 사이에 1-2개의 빈 행 남기기

**수동 개입 불필요** - 사용자가 파일을 열 때 민감도 표가 완전히 작동해야 합니다.

## 케이스 선택기 구현

**3개 케이스 프레임워크:**

### Bear Case
- 보수적인 수익 성장률 (과거 범위의 하단)
- 마진 압박 또는 확장 없음
- 더 높은 WACC (위험 프리미엄 증가)
- 더 낮은 영구 성장률
- 더 높은 CapEx 가정

### Base Case
- 컨센서스 또는 경영진 가이던스 수익 성장률
- 영업 레버리지에 기반한 적절한 마진 확장
- 현재 시장 내재 WACC
- GDP와 일치하는 영구 성장률 (2.5-3.0%)
- 표준 CapEx 가정

### Bull Case
- 낙관적인 수익 성장률 (예측치 상단)
- 상당한 마진 확장
- 더 낮은 WACC (위험 프리미엄 감소)
- 더 높은 영구 성장률 (3.5-5.0%)
- 감소된 CapEx 집약도

**수식 구현:**

전체에 분산된 중첩 IF 수식을 **사용하지 마세요**. 대신, 적절한 시나리오 블록에서 가져오는 INDEX 또는 OFFSET 수식을 사용하는 통합 열을 만드세요.

**권장 패턴 (INDEX 사용):**
`=INDEX(B10:D10, 1, $B$6)` 여기서 `B10:D10` = Bear/Base/Bull 값, `1` = 행 오프셋, `$B$6` = 케이스 선택기 셀 (1, 2, 또는 3)

**그런 다음 모든 예측에서 통합 열을 참조하세요:**
`Revenue Year 1: =D29*(1+$E$10)` 여기서 $E$10은 1년차 성장에 대한 통합 열 값입니다.

이 접근 방식은 시나리오 논리를 중앙 집중화하여 모델을 더 쉽게 감사하고 유지 관리할 수 있게 해줍니다.

## 결과물 구조

**파일 이름 지정**: `[Ticker]_DCF_Model_[Date].xlsx`

**두 개의 시트**:
1. **DCF** - Bear/Base/Bull 케이스 + 하단에 세 개의 민감도 분석표(WACC vs Terminal Growth, Revenue Growth vs EBIT Margin, Beta vs Risk-Free Rate)가 있는 완전한 모델
2. **WACC** - 자본 비용 계산

**주요 기능**: 케이스 선택기 (1/2/3), INDEX/OFFSET 수식이 있는 통합 열, 색상으로 구분된 셀, 모든 입력에 대한 셀 주석, 전문적인 테두리

## 모범 사례

### 모델 구축
1. **점진적으로 구축**: 다음으로 넘어가기 전에 각 섹션 완료
2. **구축하면서 테스트**: 수식을 검증하기 위해 샘플 숫자 입력
3. **일관된 구조 사용**: 유사한 계산은 유사한 패턴을 따름
4. **복잡한 수식 주석 달기**: 특이한 계산에 대한 메모 추가
5. **검사 기능 구축**: 해당되는 경우 합계 검사 및 잔액 검사

### 문서화
1. **모든 가정 문서화**: 주요 입력의 근거 설명
2. **데이터 소스 인용**: 각 데이터 포인트가 어디서 왔는지 메모
3. **방법론 설명**: 비표준 접근 방식 설명
4. **불확실성 표시**: 가시성이 제한된 영역 강조

### 품질 관리
1. **계산 교차 검사**: 여러 방식으로 수학 검증
2. **가정 스트레스 테스트**: 모델이 견고한지 확인하기 위해 민감도 실행
3. **피어 리뷰**: 다른 사람이 수식을 확인하도록 함
4. **버전 관리**: 작업이 진행됨에 따라 버전을 저장

## 일반적인 변형

### 고성장 기술 기업
- 더 긴 예측 기간 (7-10년)
- 더 높은 초기 성장률 (20-30%)
- 시간에 따른 상당한 마진 확장
- 더 높은 WACC (12-15%)
- 단위 경제 모델 (사용자, ARPU 등)

### 성숙/안정 기업
- 더 짧은 예측 기간 (3-5년)
- 완만한 성장률 (GDP +1-3%)
- 안정적인 마진
- 더 낮은 WACC (7-9%)
- 현금 창출 및 자본 배분에 중점

### 경기 순환 기업
- 경제 주기를 통한 모델링
- 주기 중반에서 마진 정규화
- 저점 및 고점 시나리오 고려
- 주기성을 위해 베타 조정

### 다중 세그먼트 기업
- 각 비즈니스 유닛에 대해 별도의 DCF
- 세그먼트별 상이한 성장률 및 마진
- 부문별 가치 합산(Sum-of-parts) 평가
- 시너지 고려

## 문제 해결

**오류나 불합리한 결과가 발생하면, 상세한 디버깅 지침은 [TROUBLESHOOTING.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/finance/dcf-model/TROUBLESHOOTING.md)를 읽어보세요.**

## 워크플로우 통합

### DCF 빌드 시작 시

1. **시장 데이터 수집**:
   - 현재 시장 데이터에 사용할 수 있는 MCP 서버 확인
   - 주가, 베타 및 기타 시장 지표에 대한 웹 검색/가져오기 사용
   - 특정 데이터가 필요한 경우 사용자에게 요청

2. **과거 재무 데이터 수집**:
   - 사용 가능한 MCP 서버 (Daloopa 등) 확인
   - MCP를 통해 사용할 수 없는 경우 사용자에게 요청
   - 필요한 경우 10-K에서 수동 추출

3. 이 스킬에 자세히 설명된 DCF 방법론을 사용하여 **모델 구축 시작**

### 모델 구축 중

1. (하드코드된 값이 아닌) 수식과 함께 openpyxl을 사용하여 **Excel 모델 구축**
2. 수식 구성 및 서식 지정에 대한 **xlsx 스킬 규칙 준수**
3. 사용자가 요청하거나 특정 브랜드 가이드라인이 제공된 경우에만 **채우기 색상 적용**

### 모델 제공 전 (필수)

1. **구조 확인**:
   - 예측 연도 전체에 걸친 가정이 있는 Bear/Base/Bull 시나리오 블록
   - 올바른 시나리오 블록을 참조하는 수식으로 작동하는 케이스 선택기
   - (별도의 시트가 아닌) DCF 시트 하단의 민감도 분석표
   - 글꼴 색상: 파란색 입력, 검은색 수식, 녹색 시트 링크
   - 모든 하드코드된 입력의 셀 주석
   - 주요 섹션 주변의 전문적인 테두리

2. **수식 재계산**: `python recalc.py model.xlsx 30` 실행

3. **출력 확인**:
   - `status`가 `"success"`인 경우 → 4단계로 계속
   - `status`가 `"errors_found"`인 경우 → `error_summary`를 확인하고 디버깅 지침은 [TROUBLESHOOTING.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/finance/dcf-model/TROUBLESHOOTING.md)를 읽으세요.

4. 상태가 "success"가 될 때까지 **오류를 수정하고 recalc.py를 다시 실행**

5. **수식 스팟 체크**:
   - FCF 수식 하나를 테스트합니다 - 올바른 가정 행을 참조합니까?
   - 케이스 선택기를 변경합니다 - 통합 열이 올바르게 업데이트됩니까?
   - 수익 수식이 통합 열을 참조하는지(중첩된 IF 수식이 아닌지) 확인합니다.

6. **모델 제공**

### 사용 가능한 데이터 소스

- **MCP 서버**: 구성된 경우 (과거 재무 데이터용 Daloopa)
- **웹 검색/가져오기**: 현재 주가, 베타 및 시장 데이터용
- **사용자 제공 데이터**: 과거 재무, 컨센서스 추정치
- **수동 추출**: 폴백으로서의 SEC EDGAR 문서

## 최종 출력 체크리스트

DCF 모델을 제공하기 전에:

**필수:**
- 상태가 "success"가 될 때까지 (수식 오류 0) `python recalc.py model.xlsx 30` 실행
- 두 개의 시트: DCF (하단에 민감도 포함), WACC
- 글꼴 색상: 파란색=입력, 검은색=수식, 녹색=시트 링크
- 모든 하드코드된 입력에 셀 주석
- 수식으로 완전히 채워진 민감도 분석표
- 주요 섹션 주위의 전문적인 테두리

**검증:**
- (매출총이익이 아닌) 수익에 기반한 OpEx
- EV의 50-70%인 영구 가치
- 영구 성장률 < WACC
- 세율 21-28%
- 파일 이름 지정: `[Ticker]_DCF_Model_[Date].xlsx`

## 데이터 소스 — MCP 우선, 웹 폴백

아래의 많은 구절에서 "S&P Kensho MCP / Daloopa MCP / FactSet MCP 사용"이라고 말합니다. 이는 원래 Cowork 플러그인 컨텍스트의 상용 재무 데이터 MCP입니다. Hermes에서는 다음을 수행합니다:

- **구조화된 재무 데이터 MCP가 구성되어 있는 경우** (Hermes는 MCP를 지원합니다 — `native-mcp` 스킬 참조), 특정 시점의 비교, 선행 거래 및 서류 제출을 위해 이를 우선 사용하세요.
- **그렇지 않은 경우**, 다음으로 폴백합니다:
  - 미국 서류의 경우 SEC EDGAR(`https://www.sec.gov/cgi-bin/browse-edgar`)에 대한 `web_search` / `web_extract`
  - 보도 자료, 실적 발표 자료에 대한 회사 IR 페이지
  - 대화형 데이터 포털에 대한 `browser_navigate`
  - 사용자 제공 데이터 (컨텍스트에 없을 때 명시적으로 요청)
- **절대 위조하지 마세요**. 배수, 선례 또는 서류 제출 번호를 찾을 수 없는 경우 셀을 `[UNSOURCED]`로 플래그 지정하고 사용자에게 표시하세요.

## 기여

이 스킬은 Anthropic의 Financial Services용 Claude 플러그인 제품군 (Apache-2.0)에서 채택되었습니다. Office-JS / Cowork 라이브 Excel 경로는 제거되었습니다. 이 버전은 `excel-author` 스킬 규칙을 통해 헤드리스 openpyxl을 대상으로 합니다. 원본: https://github.com/anthropics/financial-services
