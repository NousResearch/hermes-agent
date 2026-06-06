---
title: "Comps Analysis"
sidebar_label: "Comps Analysis"
description: "Excel에서 비교 기업 분석 구축 — 운영 지표, 가치 평가 배수, 피어 그룹 대비 통계적 벤치마킹"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Comps Analysis

Excel에서 비교 기업 분석(Comparable company analysis)을 구축합니다 — 운영 지표, 가치 평가 배수, 피어 그룹 대비 통계적 벤치마킹. excel-author와 함께 사용합니다. 상장 기업 가치 평가, IPO 가격 책정, 부문 벤치마킹 또는 이상치(outlier) 감지에 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/finance/comps-analysis`를 사용하여 설치 |
| 경로 | `optional-skills/finance/comps-analysis` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (adapted by Nous Research) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `finance`, `valuation`, `comps`, `excel`, `openpyxl`, `modeling`, `investment-banking` |
| 관련 스킬 | [`excel-author`](/docs/user-guide/skills/optional/finance/finance-excel-author), [`pptx-author`](/docs/user-guide/skills/optional/finance/finance-pptx-author), [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`lbo-model`](/docs/user-guide/skills/optional/finance/finance-lbo-model) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

## 환경

이 스킬은 **헤드리스(headless) openpyxl**을 가정합니다 — 디스크에 .xlsx 파일을 생성합니다.
셀 색상, 수식, 이름이 지정된 범위 및 민감도 분석표에 대한 `excel-author` 스킬의 규칙을 따르세요.
전달하기 전에 다시 계산하세요: `python /path/to/excel-author/scripts/recalc.py ./out/model.xlsx`.

# 비교 기업 분석 (Comparable Company Analysis)

## ⚠️ 중요: 데이터 소스 우선순위 (먼저 읽으세요)

**항상 다음 데이터 소스 계층 구조를 따르세요:**

1. **첫째: MCP 데이터 소스 확인** - S&P Kensho MCP, FactSet MCP 또는 Daloopa MCP를 사용할 수 있는 경우, 이를 재무 및 거래 정보에 전적으로 사용하세요.
2. 위의 MCP 데이터 소스를 사용할 수 있는 경우 **웹 검색을 사용하지 마세요.**
3. **MCP를 사용할 수 없는 경우에만:** 그런 다음 Bloomberg Terminal, SEC EDGAR filings 또는 기타 기관 소스를 사용하세요.
4. **웹 검색을 주요 데이터 소스로 절대 사용하지 마세요** - 기관 수준 분석에 필요한 정확성, 감사 추적 및 신뢰성이 부족합니다.

**이것이 중요한 이유:** MCP 소스는 적절한 인용과 함께 검증된 기관 수준의 데이터를 제공합니다. 웹 검색 결과는 재무 분석을 위해 구식이거나 부정확하거나 신뢰할 수 없을 수 있습니다.

---

## 개요
이 스킬은 운영 지표, 가치 평가 배수 및 통계적 벤치마킹을 결합하는 기관 수준의 비교 기업 분석을 구축하도록 에이전트를 교육합니다. 결과물은 동종 그룹 비교를 통해 정보에 입각한 투자 결정을 가능하게 하는 구조화된 Excel/스프레드시트입니다.

**참고자료 및 맥락화:**

비교 기업 분석 예제가 `examples/comps_example.xlsx`에 제공됩니다. 이 파일이나 이 스킬 디렉토리의 다른 예제 파일을 사용할 때 지능적으로 사용하세요:

**예제를 다음과 같은 용도로 사용하세요:**
- 구조적 계층 구조 이해 (섹션의 흐름)
- 예상되는 엄격함 수준 파악 (통계적 깊이, 문서화 표준)
- 원칙 학습 (명확한 헤더, 투명한 수식, 감사 추적)

**예제를 다음과 같은 용도로 사용하지 마세요:**
- 서식이나 지표를 그대로 복제
- 맥락을 고려하지 않고 레이아웃 복사
- 청중에 관계없이 동일한 시각적 스타일 적용

**항상 먼저 자신에게 물어보세요:**
1. **"선호하는 형식이 있습니까, 아니면 템플릿 스타일을 적용해야 합니까?"**
2. **"청중은 누구입니까?"** (투자 위원회, 이사회 프레젠테이션, 빠른 참조, 상세 메모)
3. **"핵심 질문은 무엇입니까?"** (가치 평가, 성장 분석, 경쟁 포지셔닝, 효율성)
4. **"맥락은 무엇입니까?"** (M&A 평가, 투자 결정, 부문 벤치마킹, 성과 검토)

**세부 사항에 따른 조정:**
- **산업 컨텍스트**: 빅테크 대형주는 신흥 SaaS 스타트업과 다른 지표를 필요로 합니다.
- **부문별 요구 사항**: 관련 지표를 조기에 추가하세요 (예: 기술의 경우 클라우드 ARR, 엔터프라이즈 고객, 개발자 생태계).
- **회사 친숙도**: 잘 알려진 회사는 배경 설명이 덜 필요하고, 델타 분석에 더 집중할 수 있습니다.
- **의사 결정 유형**: M&A는 지속적인 포트폴리오 모니터링과 다른 강조점을 요구합니다.

**핵심 원칙:** 템플릿 원칙(명확한 구조, 통계적 엄격함, 투명한 수식)을 사용하되 맥락에 따라 실행을 달리하세요. 목표는 기관처럼 보이는 템플릿이 아니라 기관 수준의 분석입니다.

사용자가 제공한 예제와 명시적 기본 설정이 항상 기본값보다 우선합니다.

## 핵심 철학
**"올바른 구조를 먼저 구축한 다음, 데이터가 이야기를 하도록 하세요."**

중요한 것이 무엇인지 전략적 사고를 강제하는 헤더로 시작하여 깔끔한 데이터를 입력하고, 투명한 수식을 작성하여 통계가 자동으로 나타나게 하세요. 좋은 비교 분석은 그것을 구축하지 않은 사람이 즉시 읽을 수 있어야 합니다.

---

## ⚠️ 중요: 하드코드보다 수식 우선 + 단계별 검증

**하드코드가 아닌 수식:**
- 파생된 모든 값(마진, 배수, 통계)은 반드시 입력 셀을 참조하는 Excel 수식이어야 합니다 — 미리 계산된 숫자를 붙여넣어서는 절대 안 됩니다.
- Python/openpyxl을 사용하여 시트를 작성할 때: `cell.value = "=E7/C7"`(수식 문자열)로 작성하고, `cell.value = 0.687`(계산된 결과)로 작성하지 **마세요**.
- 유일하게 하드코딩된 값은 원시 입력 데이터(수익, EBITDA, 주가 등)여야 하며, 모든 데이터에는 출처를 명시한 셀 주석이 있어야 합니다.
- 이유: 사용자가 입력을 변경할 때 모델이 자동으로 업데이트되어야 합니다. 하드코딩된 마진은 언제 터질지 모르는 조용한 버그입니다.

**사용자와 함께 단계별 확인:**
- 구조를 설정한 후 → 데이터를 채우기 전에 사용자에게 헤더 레이아웃을 보여줍니다.
- 원시 입력을 입력한 후 → 입력 블록을 보여주고 수식을 작성하기 전에 출처/기간을 확인합니다.
- 운영 지표 수식을 구축한 후 → 계산된 마진을 보여주고 가치 평가로 넘어가기 전에 사용자와 함께 온전성을 점검합니다.
- 가치 평가 배수를 구축한 후 → 배수를 보여주고 통계를 추가하기 전에 합리적으로 보이는지 확인합니다.
- 시트 전체를 처음부터 끝까지 한 번에 구축한 다음 제시하지 **마세요** — 각 섹션을 확인하여 오류를 조기에 포착하세요.

---

## 섹션 1: 문서 구조 및 설정

### 헤더 블록 (행 1-3)
```
Row 1: [ANALYSIS TITLE] - COMPARABLE COMPANY ANALYSIS
Row 2: [List of Companies with Tickers] • [Company 1 (TICK1)] • [Company 2 (TICK2)] • [Company 3 (TICK3)]
Row 3: As of [Period] | All figures in [USD Millions/Billions] except per-share amounts and ratios
```

**이것이 중요한 이유:** 맥락을 즉시 설정합니다. 이 파일을 여는 사람은 누구나 자신이 무엇을 보고 있는지, 언제 만들어졌는지, 숫자를 어떻게 해석해야 하는지 알 수 있습니다.

### 시각적 규칙 표준 (선택 사항 - 사용자 환경설정 및 업로드된 템플릿이 항상 우선함)

**중요: 이들은 제안된 기본값일 뿐입니다. 항상 다음을 우선시하세요:**
1. 사용자의 명시적인 서식 기본 설정
2. 업로드된 모든 템플릿 파일의 서식
3. 회사/팀 스타일 가이드
4. 이 기본값 (다른 지침이 제공되지 않은 경우에만)

**제안하는 글꼴 및 타이포그래피:**
- **글꼴 종류**: Times New Roman (전문적이고 읽기 쉬우며 업계 표준)
- **글꼴 크기**: 데이터 셀은 11pt, 헤더는 12pt
- **굵은 텍스트**: 섹션 헤더, 회사 이름, 통계 레이블

**기본 색상 및 음영 — 전문적인 파란색/회색 팔레트 (최소화할수록 좋음):**
- **절제 유지** — 파란색과 회색만 사용합니다. 녹색, 주황색, 빨간색 또는 여러 강조 색상을 도입하지 마세요. 깔끔한 비교 시트는 총 3-4가지 색상만 사용합니다.
- **섹션 헤더** (예: "OPERATING STATISTICS & FINANCIAL METRICS"):
  - 짙은 파란색 배경 (`#1F4E79` 또는 `#17365D` 네이비)
  - 흰색 굵은 텍스트
  - 모든 열에 걸친 전체 행 음영
- **열 헤더** (예: "Company", "Revenue", "Margin"):
  - 밝은 파란색 배경 (`#D9E1F2` 또는 유사한 연한 파란색)
  - 검은색 굵은 텍스트
  - 가운데 정렬
- **데이터 행**:
  - 회사 데이터의 경우 흰색 배경
  - 수식의 경우 검은색 텍스트; 하드코드된 입력의 경우 파란색 텍스트
- **통계 행** (Maximum, 75th Percentile 등):
  - 밝은 회색 배경 (`#F2F2F2`)
  - 검은색 텍스트, 왼쪽 정렬 레이블
- **이것이 전체 팔레트입니다**: 짙은 파란색 + 밝은 파란색 + 밝은 회색 + 흰색. 사용자의 템플릿에 달리 명시되어 있지 않는 한 다른 것은 없습니다.

**제안하는 서식 규칙:**
- **소수점 정밀도**:
  - 백분율: 소수점 첫째 자리 (12.3%)
  - 배수: 소수점 첫째 자리 (13.5x)
  - 달러 금액: 소수점 없음, 천 단위 구분 기호 (69,632)
  - 백분율로 표시된 마진: 소수점 첫째 자리 (68.7%)
- **테두리**: 테두리 없음 (깔끔하고 미니멀한 외관)
- **정렬**: 깔끔하고 균일한 외관을 위해 모든 지표 가운데 정렬
- **셀 치수**: 모든 열 너비는 균일해야 하며, 모든 행 높이는 일관되어야 함 (깔끔하고 전문적인 그리드 생성)

**참고:** 사용자가 템플릿 파일을 제공하거나 다른 서식을 지정하는 경우 이를 사용하세요.

---

## 섹션 2: 운영 통계 및 재무 지표

### 핵심 열 (이것으로 시작)
1. **Company** - 일관된 서식의 회사 이름
2. **Revenue** - 규모 지표 (컨텍스트에 따라 LTM, 분기별 또는 연간일 수 있음)
3. **Revenue Growth** - 전년 동기 대비 백분율 변화
4. **Gross Profit** - 매출액에서 매출원가를 뺀 값
5. **Gross Margin** - 매출총이익/매출액 (기본 수익성)
6. **EBITDA** - 이자, 세금, 감가상각비, 무형자산상각비 차감 전 이익
7. **EBITDA Margin** - EBITDA/매출액 (운영 효율성)

### 선택적 추가 (산업/목적에 따라 선택)
- **Quarterly vs LTM** - 계절성이 중요한 경우 둘 다 포함
- **Free Cash Flow** - 자본 집약적이거나 SaaS 비즈니스의 경우
- **FCF Margin** - FCF/매출액 (현금 창출 효율성)
- **Net Income** - 성숙하고 수익성 있는 회사의 경우
- **Operating Income** - 감가상각비(D&A)가 다양한 비즈니스의 경우
- **CapEx metrics** - 자산 집약적 산업의 경우
- **Rule of 40** - 특히 SaaS용 (성장률 % + 마진 %)
- **FCF Conversion** - 이익의 질 분석용 (고급)

### 수식 예시 (행 7을 예로 사용)
```excel
// Core ratios - these are always calculated
Gross Margin (F7): =E7/C7
EBITDA Margin (H7): =G7/C7

// Optional ratios - include if relevant
FCF Margin: =[FCF]/[Revenue]
Net Margin: =[Net Income]/[Revenue]
Rule of 40: =[Growth %]+[FCF Margin %]
```

**황금률:** 모든 비율은 [무언가] / [Revenue] 또는 [무언가] / [이 시트의 무언가] 여야 합니다. 단순하게 유지하세요.

### 통계 블록 (회사 데이터 다음)

**중요: 모든 비교 가능한 지표(비율, 마진, 성장률, 배수)에 통계 수식을 추가하세요.**

```
[Leave one blank row for visual separation]
- Maximum: =MAX(B7:B9)
- 75th Percentile: =QUARTILE(B7:B9,3)
- Median: =MEDIAN(B7:B9)
- 25th Percentile: =QUARTILE(B7:B9,1)
- Minimum: =MIN(B7:B9)
```

**통계가 필요한 열 (비교 가능한 지표):**
- Revenue Growth %, Gross Margin %, EBITDA Margin %, EPS
- EV/Revenue, EV/EBITDA, P/E, Dividend Yield %, Beta

**통계가 필요 없는 열 (규모 지표):**
- Revenue, EBITDA, Net Income (절대 규모는 회사 규모에 따라 다름)
- Market Cap, Enterprise Value (크기가 다른 회사 간에 비교할 수 없음)

**참고:** 시각적 분리를 위해 회사 데이터와 통계 행 사이에 빈 행을 하나 추가하세요. "SECTOR STATISTICS" 또는 "VALUATION STATISTICS" 헤더 행을 추가하지 **마세요**.

**사분위수가 중요한 이유:** 이들은 단순히 평균이 아니라 분포를 보여줍니다. 75백분위수 배수는 어떤 "프리미엄" 회사가 거래되는지 알려줍니다.

---

## 섹션 3: 가치 평가 배수 및 투자 지표

### 핵심 가치 평가 열 (이것으로 시작)
1. **Company** - 운영 섹션과 동일한 순서
2. **Market Cap** - 현재 시장 가치 평가
3. **Enterprise Value** - 시가총액 ± 순부채/현금
4. **EV/Revenue** - 시장이 매출 1달러당 얼마를 지불하는지
5. **EV/EBITDA** - 시장이 수익 1달러당 얼마를 지불하는지
6. **P/E Ratio** - 순이익 대비 주가

### 선택적 가치 평가 지표 (맥락에 따라 선택)
- **FCF Yield** - FCF/시가총액 (현금 중심 분석용)
- **PEG Ratio** - P/E/성장률 (성장 기업용)
- **Price/Book** - 시장 가치 vs 장부 가치 (자산 집약적 비즈니스용)
- **ROE/ROA** - 수익률 지표 (수익성 비교용)
- **Revenue/EBITDA CAGR** - 과거 성장률 (추세 분석용)
- **Asset Turnover** - 매출액/자산 (운영 효율성용)
- **Debt/Equity** - 레버리지 (자본 구조 분석용)

**핵심 원칙:** 해당 산업에 중요한 3-5개의 핵심 배수를 포함하세요. 할 수 있다고 해서 가능한 모든 지표를 포함하지 마세요.

### 수식 예시
```excel
// Core multiples - always include these
EV/Revenue: =[Enterprise Value]/[LTM Revenue]
EV/EBITDA: =[Enterprise Value]/[LTM EBITDA]
P/E Ratio: =[Market Cap]/[Net Income]

// Optional multiples - include if data available
FCF Yield: =[LTM FCF]/[Market Cap]
PEG Ratio: =[P/E]/[Growth Rate %]
```

### 상호 참조 규칙
**중요:** 가치 평가 배수는 MUST 운영 지표 섹션을 참조해야 합니다. 동일한 원시 데이터를 두 번 입력하지 마세요. 수익이 C7에 있으면 EV/Revenue 수식은 C7을 참조해야 합니다.

### 통계 블록
운영 섹션과 동일한 구조: 모든 지표에 대한 Max, 75th, Median, 25th, Min. 시각적 분리를 위해 회사 데이터와 통계 사이에 빈 행을 하나 추가하세요. "VALUATION STATISTICS" 헤더 행을 추가하지 마세요.

---

## 섹션 4: 노트 및 방법론 문서화

### 필수 구성요소

**데이터 소스 및 품질:**
- 데이터의 출처는 어디입니까? (S&P Kensho MCP, FactSet MCP, Daloopa MCP, Bloomberg, SEC filings)
- 어떤 기간을 포괄합니까? (Q4 2024, 감사된 수치)
- 어떻게 검증되었습니까? (10-K/10-Q와 교차 확인)
- 참고: 더 나은 정확성과 추적 가능성을 위해 가능한 경우 MCP 데이터 소스(S&P Kensho, FactSet, Daloopa)를 우선하세요.

**주요 정의:**
- EBITDA 계산 방법 (매출총이익 + D&A 또는 영업이익 + D&A)
- 잉여현금흐름 수식 (영업현금흐름 - CapEx)
- 특별한 지표 설명 (Rule of 40, FCF Conversion)
- 기간 정의 (LTM, CAGR 계산 기간)

**가치 평가 방법론:**
- 기업 가치는 어떻게 계산되었습니까? (시가총액 + 순부채)
- 어떤 성장률이 사용되었습니까? (과거 CAGR, 선도 추정치)
- 어떤 조정이 이루어졌습니까? (일회성 항목 제외, 정규화된 마진)

**분석 프레임워크:**
- 투자 논제는 무엇입니까? (클라우드/SaaS 효율성)
- 어떤 지표가 가장 중요합니까? (현금 창출, 자본 효율성)
- 독자들은 통계를 어떻게 해석해야 합니까? (사분위수가 맥락 제공)

---

## 섹션 5: 올바른 지표 선택 (의사 결정 프레임워크)

### "내가 어떤 질문에 대답하고 있는가?"로 시작하기

**"어떤 회사가 저평가되어 있는가?"**
→ 집중: EV/Revenue, EV/EBITDA, P/E, Market Cap
→ 건너뛰기: 운영 세부 정보, 성장 지표

**"어떤 회사가 가장 효율적인가?"**
→ 집중: Gross Margin, EBITDA Margin, FCF Margin, Asset Turnover
→ 건너뛰기: 규모 지표, 절대 달러 금액

**"어떤 회사가 가장 빠르게 성장하고 있는가?"**
→ 집중: Revenue Growth %, EBITDA CAGR, User/Customer Growth
→ 건너뛰기: 마진 지표, 레버리지 비율

**"최고의 현금 창출 기업은 어디인가?"**
→ 집중: FCF, FCF Margin, FCF Conversion, CapEx intensity
→ 건너뛰기: EBITDA, P/E ratios

### 산업별 지표 선택

**Software/SaaS:**
필수: Revenue Growth, Gross Margin, Rule of 40
선택: ARR, Net Dollar Retention, CAC Payback
건너뛰기: Asset Turnover, Inventory metrics

**Manufacturing/Industrials:**
필수: EBITDA Margin, Asset Turnover, CapEx/Revenue
선택: ROA, Inventory Turns, Backlog
건너뛰기: Rule of 40, SaaS metrics

**Financial Services:**
필수: ROE, ROA, Efficiency Ratio, P/E
선택: Net Interest Margin, Loan Loss Reserves
건너뛰기: Gross Margin, EBITDA (은행에는 의미 없음)

**Retail/E-commerce:**
필수: Revenue Growth, Gross Margin, Inventory Turnover
선택: Same-Store Sales, Customer Acquisition Cost
건너뛰기: Heavy R&D or CapEx metrics

### "5-10 원칙"

**5개 운영 지표** - 수익, 성장률, 2-3개의 마진/효율성 지표
**5개 가치 평가 지표** - 시가총액, 기업 가치, 3개 배수
**= 총 10개 열** - 이야기를 전달하기에 충분하며, 너무 많아서 흐름을 잃지 않습니다.

지표가 15개를 넘는다면, 아마도 노이즈를 포함하고 있을 것입니다. 무자비하게 편집하세요.

---

## 섹션 6: 모범 사례 및 품질 검사

### 시작하기 전에
1. **피어 그룹 정의** - 회사는 진정으로 비교 가능해야 합니다 (유사한 비즈니스 모델, 규모, 지리적 위치)
2. **올바른 기간 선택** - LTM은 계절성을 완화하고 분기별은 추세를 보여줍니다
3. **사전에 단위 표준화** - 수백만 대 수십억의 결정은 모든 것에 영향을 미칩니다
4. **데이터 소스 매핑** - 각 숫자가 어디서 오는지 파악하세요

### 구축하면서
1. **모든 원시 데이터를 먼저 입력** - 수식을 작성하기 전에 파란색 텍스트를 완료하세요
2. **모든 하드코드된 입력에 셀 주석 추가** - 셀을 마우스 오른쪽 버튼으로 클릭 → 주석 삽입 → 소스 OR 가정 문서화

   **출처가 있는 데이터의 경우 정확히 어디서 가져왔는지 인용:**
   - 예: "Bloomberg Terminal - MSFT Equity DES, accessed 2024-10-02"
   - 예: "Q4 2024 10-K filing, page 42, line item 'Total Revenue'"
   - 예: "FactSet consensus estimate as of 2024-10-02"
   - **가능한 경우 하이퍼링크 포함**: 셀을 마우스 오른쪽 버튼으로 클릭 → 링크 → SEC 파일링, 데이터 소스 또는 보고서에 URL 붙여넣기

   **가정의 경우 이유를 설명:**
   - 예: "회사가 공개하지 않아 동종 그룹 중앙값에 기반하여 15% EBITDA 마진 가정"
   - 예: "기업 가치를 시가총액 + 5,000만 달러의 순부채로 추정 (Q3 대차대조표에서 가져옴, Q4는 아직 제공되지 않음)"
   - 예: "스트리트 컨센서스 EPS $3.45에 기반한 포워드 P/E (12명의 애널리스트 추정치 평균)"

   **이것이 중요한 이유**: 감사 추적, 데이터 검증, 가정 투명성 및 향후 업데이트를 가능하게 합니다.
3. **한 행씩 수식 구축** - 다음으로 넘어가기 전에 각 계산을 테스트하세요
4. **헤더에 절대 참조 사용** - $C$6는 헤더 행을 잠급니다
5. **일관성 있는 서식** - 백분율은 소수가 아닌 백분율로 표시합니다
6. **조건부 서식 추가** - 이상치를 자동으로 강조합니다

### 온전성 검사 (Sanity Checks)
- **마진 테스트**: 매출총이익률 > EBITDA 마진 > 순이익률 (항상 정의상 참)
- **배수 합리성**: 
  - EV/Revenue: 일반적으로 0.5-20x (산업마다 크게 다름)
  - EV/EBITDA: 일반적으로 8-25x (산업 전반에 걸쳐 상당히 일관됨)
  - P/E: 일반적으로 10-50x (성장률에 따라 다름)
- **성장-배수 상관관계**: 성장이 높을수록 일반적으로 배수도 높습니다
- **규모-효율성 트레이드오프**: 대기업은 종종 더 나은 마진을 가집니다 (규모의 이점)

### 피해야 할 일반적인 실수
❌ 수식에서 시가총액과 기업 가치를 혼합함
❌ 분자와 분모에 다른 기간을 사용함 (LTM vs 분기별)
❌ 셀 참조 대신 수식에 숫자를 하드코딩함
❌ **출처를 인용하거나 가정을 설명하는 셀 주석이 없는 하드코드된 입력**
❌ 가능한 경우 SEC 제출 파일이나 데이터 소스에 대한 하이퍼링크 누락
❌ 명확한 목적 없이 너무 많은 지표를 포함함
❌ 비교할 수 없는 회사(다른 비즈니스 모델)를 포함함
❌ 출처 공개 없이 오래된 데이터를 사용함
❌ 백분율의 평균을 잘못 계산함 (중앙값을 사용해야 함)

---

## 섹션 6: 고급 기능

### 동적 헤더
계산을 보여주는 열의 경우 명확한 단위 레이블을 사용하세요:
```
Revenue Growth (YoY) % | EBITDA Margin | FCF Margin | Rule of 40
```

### 사분위수 분석의 이점
단순한 평균/중앙값 대신 사분위수는 다음을 보여줍니다:
- **75백분위수** = "프리미엄" 기업들이 여기서 거래됨
- **중앙값** = 일반적인 시장 가치 평가
- **25백분위수** = "할인" 영역

이것은 다음 질문에 답하는 데 도움이 됩니다: "대상 회사가 피어 그룹에 비해 비싸게 거래되는가, 싸게 거래되는가?"

### 산업별 변형

**Software/SaaS:**
- 추가: ARR, Net Dollar Retention, CAC Payback Period
- 강조: Rule of 40, FCF margins, gross margins >70%

**Healthcare:**
- 추가: R&D/Revenue, Pipeline value, Regulatory status
- 강조: EBITDA margins, growth rates, reimbursement risk

**Industrials:**
- 추가: Backlog, Order book trends, Geographic mix
- 강조: ROIC, asset turnover, cyclical adjustments

**Consumer:**
- 추가: Same-store sales, Customer acquisition cost, Brand value
- 강조: Revenue growth, gross margins, inventory turns

---

## 섹션 7: 워크플로우 및 실용적인 팁

### 단계별 프로세스
1. **구조 설정** (30분)
   - 모든 헤더 생성
   - 셀 형식 지정 (입력은 파란색, 수식은 검은색)
   - 단위 및 날짜 참조 고정

2. **데이터 수집** (60-90분)
   - 1차 소스에서 가져오기 (사용 가능한 경우 S&P Kensho MCP, FactSet MCP, Daloopa MCP; 그렇지 않으면 Bloomberg, SEC)
   - 파란색으로 모든 원시 숫자 입력
   - 노트 섹션에 소스 문서화

3. **수식 작성** (30분)
   - 간단한 비율(마진)부터 시작
   - 배수(EV/Revenue)로 진행
   - 교차 확인 추가 (마진이 이치에 맞습니까?)

4. **통계 추가** (15분)
   - 모든 열에 수식 구조 복사
   - 범위가 올바른지 확인 (B7:B10이 아닌 B7:B9)
   - 사분위수 논리 확인

5. **품질 관리** (30분)
   - 온전성 검사 실행
   - 수식 참조 확인
   - #DIV/0! 또는 #REF! 오류 확인
   - 알려진 벤치마크와 비교

6. **문서화** (15분)
   - 노트 섹션 완료
   - 데이터 소스 추가
   - 방법론 정의
   - 분석 날짜 스탬프

### 프로 팁
- **템플릿 저장**: 한 번 만들고 영원히 재사용하세요.
- **이상치 색상 코딩**: 표준 편차가 2를 초과하는 값에 대한 조건부 서식을 적용하세요.
- **소스 파일에 연결**: Bloomberg 스크린샷이나 SEC 제출 파일에 하이퍼링크를 연결하세요.
- **버전 관리**: 명확한 날짜가 있는 "Comps_v1_2024-12-15"로 저장하세요.
- **공동 검토**: 다른 사람이 수식을 확인하게 하세요.

### Excel 서식 체크리스트 (선택 사항 - 사용자의 선호에 맞게 조정)
- [ ] 글꼴이 사용자가 선호하는 스타일로 설정됨 (기본값: Times New Roman, 11pt 데이터, 12pt 헤더)
- [ ] 섹션 헤더가 사용자의 템플릿에 맞게 포맷됨 (기본값: 흰색 굵은 텍스트가 있는 짙은 파란색 #17365D)
- [ ] 열 헤더가 사용자의 템플릿에 맞게 포맷됨 (기본값: 검은색 굵은 텍스트가 있는 밝은 파란색/회색 #D9E2F3)
- [ ] 통계 행이 사용자의 템플릿에 맞게 포맷됨 (기본값: 밝은 회색 #F2F2F2)
- [ ] 테두리가 적용되지 않음 (깔끔하고 미니멀한 외관)
- [ ] **열 너비가 균일/동일한 너비로 설정됨** (깔끔하고 전문적인 외관 생성)
- [ ] **행 높이가 일관된 높이로 설정됨** (데이터 행의 경우 일반적으로 20-25pt)
- [ ] 숫자가 적절한 소수점 정밀도 및 천 단위 구분 기호로 포맷됨
- [ ] 깔끔하고 균일한 모양을 위해 **모든 지표 가운데 정렬됨**
- [ ] **회사 데이터와 통계 행 사이의 분리를 위한 하나의 빈 행**
- [ ] **별도의 "SECTOR STATISTICS" 또는 "VALUATION STATISTICS" 헤더 행 없음**
- [ ] **모든 하드코드된 입력 셀에는 (1) 정확한 데이터 소스 또는 (2) 가정 설명을 포함하는 주석이 있음**
- [ ] **해당하는 경우 하이퍼링크가 추가됨** (SEC 서류 제출, 데이터 제공자 페이지, 보고서)

---

## 섹션 8: 예시 템플릿 레이아웃

**단순한 버전 (여기서 시작):**
<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────────────┐
│ TECHNOLOGY - COMPARABLE COMPANY ANALYSIS                    │
│ Microsoft • Alphabet • Amazon                               │
│ As of Q4 2024 | All figures in USD Millions                │
├─────────────────────────────────────────────────────────────┤
│ OPERATING METRICS                                           │
├──────────┬─────────┬─────────┬──────────┬──────────────────┤
│ Company  │ Revenue │ Growth  │ Gross    │ EBITDA  │ EBITDA │
│          │ (LTM)   │ (YoY)   │ Margin   │ (LTM)   │ Margin │
├──────────┼─────────┼─────────┼──────────┼─────────┼────────┤
│ MSFT     │ 261,400 │ 12.3%   │ 68.7%    │ 205,100 │ 78.4%  │
│ GOOGL    │ 349,800 │ 11.8%   │ 57.9%    │ 239,300 │ 68.4%  │
│ AMZN     │ 638,100 │ 10.5%   │ 47.3%    │ 152,600 │ 23.9%  │
│          │         │         │          │         │        │ [blank row]
│ Median   │ =MEDIAN │ =MEDIAN │ =MEDIAN  │ =MEDIAN │=MEDIAN │
│ 75th %   │ =QUART  │ =QUART  │ =QUART   │ =QUART  │=QUART  │
│ 25th %   │ =QUART  │ =QUART  │ =QUART   │ =QUART  │=QUART  │
├─────────────────────────────────────────────────────────────┤
│ VALUATION MULTIPLES                                         │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ Company  │ Mkt Cap  │ EV       │ EV/Rev   │ EV/EBITDA │ P/E│
├──────────┼──────────┼──────────┼──────────┼───────────┼────┤
│ MSFT     │3,550,000 │3,530,000 │ 13.5x    │ 17.2x     │36.0│
│ GOOGL    │2,030,000 │1,960,000 │  5.6x    │  8.2x     │24.5│
│ AMZN     │2,226,000 │2,320,000 │  3.6x    │ 15.2x     │58.3│
│          │          │          │          │           │    │ [blank row]
│ Median   │ =MEDIAN  │ =MEDIAN  │ =MEDIAN  │ =MEDIAN   │=MED│
│ 75th %   │ =QUART   │ =QUART   │ =QUART   │ =QUART    │=QRT│
│ 25th %   │ =QUART   │ =QUART   │ =QUART   │ =QUART    │=QRT│
└──────────┴──────────┴──────────┴──────────┴───────────┴────┘
```
<!-- ascii-guard-ignore-end -->

**필요할 때만 복잡성 추가:**
- 계절성이 중요하다면 분기별 및 LTM 모두 포함
- 현금 창출이 핵심 이야기라면 FCF 지표 추가
- 산업별 특정 지표 포함 (SaaS의 경우 Rule of 40 등)
- 5개 이상의 회사가 있는 경우 통계 행을 더 추가

---

## 섹션 9: 산업별 추가 항목 (선택 사항)

분석에 필수적인 경우에만 이를 추가하세요. 대부분의 비교 기업 분석은 핵심 지표만으로도 잘 작동합니다.

**Software/SaaS:**
관련이 있는 경우 추가: ARR, Net Dollar Retention, Rule of 40

**Financial Services:**
관련이 있는 경우 추가: ROE, Net Interest Margin, Efficiency Ratio

**E-commerce:**
관련이 있는 경우 추가: GMV, Take Rate, Active Buyers

**Healthcare:**
관련이 있는 경우 추가: R&D/Revenue, Pipeline Value, Patent Timeline

**Manufacturing:**
관련이 있는 경우 추가: Asset Turnover, Inventory Turns, Backlog

---

## 섹션 10: 레드 플래그 및 경고 신호

### 데이터 품질 문제
🚩 일관성 없는 기간 (분기별 및 연간 혼합)  
🚩 설명 없는 누락된 데이터  
🚩 데이터 소스 간의 상당한 차이 (>10% 분산)

### 가치 평가 레드 플래그
🚩 음수 EBITDA 기업이 EBITDA 배수로 가치 평가됨 (대신 수익 배수 사용)  
🚩 초고도 성장 스토리 없이 P/E 비율이 100배 초과  
🚩 산업에 맞지 않는 마진율

### 비교 가능성 문제
🚩 회계 연도 종료일이 다름 (타이밍 문제 발생)  
🚩 순수 플레이(pure-play)와 복합 기업(conglomerates) 혼합  
🚩 "비교 기업(comps)"으로 지정되었으나 실제로는 크게 다른 비즈니스 모델

**의심스러운 경우, 해당 회사를 제외하세요.** 6개의 의심스러운 비교보다 3개의 완벽한 비교가 더 낫습니다.

---

## 섹션 11: 수식 참조 가이드

### 필수 Excel 수식
```excel
// Statistical Functions
=AVERAGE(range)          // Simple mean
=MEDIAN(range)           // Middle value
=QUARTILE(range, 1)      // 25th percentile
=QUARTILE(range, 3)      // 75th percentile
=MAX(range)              // Maximum value
=MIN(range)              // Minimum value
=STDEV.P(range)          // Standard deviation

// Financial Calculations
=B7/C7                   // Simple ratio (Margin)
=SUM(B7:B9)/3            // Average of multiple companies
=IF(B7>0, C7/B7, "N/A")  // Conditional calculation
=IFERROR(C7/D7, 0)       // Handle divide by zero

// Cross-Sheet References
='Sheet1'!B7             // Reference another sheet
=VLOOKUP(A7, Table1, 2)  // Lookup from data table
=INDEX(MATCH())          // Advanced lookup

// Formatting
=TEXT(B7, "0.0%")        // Format as percentage
=TEXT(C7, "#,##0")       // Thousands separator
```

### 일반적인 비율 수식
```excel
Gross Margin = Gross Profit / Revenue
EBITDA Margin = EBITDA / Revenue
FCF Margin = Free Cash Flow / Revenue
FCF Conversion = FCF / Operating Cash Flow
ROE = Net Income / Shareholders' Equity
ROA = Net Income / Total Assets
Asset Turnover = Revenue / Total Assets
Debt/Equity = Total Debt / Shareholders' Equity
```

---

## 핵심 원칙 요약

1. **구조가 통찰력을 주도합니다** - 올바른 헤더는 올바른 사고를 강제합니다.
2. **적을수록 좋습니다** - 의미 없는 20개 지표보다 의미 있는 5-10개 지표가 낫습니다.
3. **질문에 맞는 지표를 선택하세요** - 가치 평가 분석은 효율성 분석과 다릅니다.
4. **통계가 패턴을 보여줍니다** - 중앙값/사분위수는 평균보다 더 많은 것을 드러냅니다.
5. **투명성이 복잡성보다 낫습니다** - 모두가 이해할 수 있는 간단한 수식.
6. **비교 가능성이 핵심입니다** - 나쁜 비교 기업을 억지로 끼워넣는 것보다 제외하는 것이 낫습니다.
7. **선택을 문서화하세요** - 왜 이 지표를 선택했는지 노트 섹션에 설명하세요.

---

## 출력 체크리스트

비교 분석을 전달하기 전에 다음을 확인하세요:
- [ ] 모든 회사가 진정으로 비교 가능한가
- [ ] 데이터가 일관된 기간의 것인가
- [ ] 단위가 명확하게 라벨링되어 있는가 (수백만/수십억)
- [ ] 수식이 하드코딩된 값이 아닌 셀을 참조하는가
- [ ] **모든 하드코딩된 입력 셀에 (1) 인용과 함께 정확한 데이터 소스, 또는 (2) 설명이 포함된 명확한 가정에 대한 주석이 있는가**
- [ ] **관련 있는 곳에 하이퍼링크가 추가되었는가** (SEC EDGAR 파일링, Bloomberg 페이지, 연구 보고서)
- [ ] 통계에 최소 5개의 지표가 포함되어 있는가 (Max, 75th, Med, 25th, Min)
- [ ] 노트 섹션에 소스와 방법론이 문서화되어 있는가
- [ ] 시각적 서식이 규칙을 따르는가 (파란색 = 입력, 검은색 = 수식)
- [ ] 온전성 검사를 통과하는가 (논리적인 마진, 합리적인 배수)
- [ ] 날짜 스탬프가 최신인가 ("As of [Date]")
- [ ] 수식 감사를 통해 오류가 없음이 확인되었는가 (#DIV/0!, #REF!, #N/A)

---

## 지속적인 개선

비교 분석을 완료한 후 질문해 보세요:
1. 통계가 예상치 못한 통찰력을 드러냈는가?
2. 분석을 제한하는 데이터 간극이 있었는가?
3. 이해관계자가 포함하지 않은 지표를 요구했는가?
4. 예상 시간 대비 실제 소요 시간은 얼마인가?
5. 다음번에 이것을 더 유용하게 만들려면 어떻게 해야 하는가?

가장 훌륭한 비교 분석은 반복될 때마다 진화합니다. 템플릿을 저장하고, 피드백에서 배우며, 의사 결정자들이 실제로 사용하는 것에 기반하여 구조를 다듬어 나가세요.


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
