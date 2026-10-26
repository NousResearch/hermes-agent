---
title: "비교기업 분석 — Excel에서 비교기업 가치평가 워크북 구축"
sidebar_label: "비교기업 분석"
description: "Excel에서 비교기업 가치평가 워크북 구축"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# 비교기업 분석

Excel에서 비교기업 가치평가 워크북을 구축합니다.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/finance/comps-analysis`로 설치 |
| 경로 | `optional-skills/finance/comps-analysis` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (Nous Research가 적용) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `finance`, `valuation`, `comps`, `excel`, `openpyxl`, `modeling`, `investment-banking` |
| 관련 skill | [`excel-author`](/docs/user-guide/skills/optional/finance/finance-excel-author), [`pptx-author`](/docs/user-guide/skills/optional/finance/finance-pptx-author), [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`lbo-model`](/docs/user-guide/skills/optional/finance/finance-lbo-model) |

## 참조: 전체 SKILL.md

:::info
다음은 이 skill이 트리거될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

## 환경

이 skill은 **헤드리스 openpyxl**을 전제로 합니다 — 디스크에 .xlsx 파일을 생성합니다.
셀 색상, 수식, 명명된 범위, 민감도 표에는 `excel-author` skill의 규칙을 따르세요.
전달 전에 다시 계산하세요: `python /path/to/excel-author/scripts/recalc.py ./out/model.xlsx`.

# 비교기업 분석

## ⚠️ 중요: 데이터 출처 우선순위 (먼저 읽으세요)

**항상 다음 데이터 출처 계층을 따르세요:**

1. **먼저: MCP 데이터 출처 확인** - S&P Kensho MCP, FactSet MCP 또는 Daloopa MCP를 사용할 수 있다면 금융 및 거래 정보에 해당 출처만 사용
2. **위 MCP 데이터 출처를 사용할 수 있으면 웹 검색을 사용하지 말 것**
3. **MCP를 사용할 수 없는 경우에만:** Bloomberg Terminal, SEC EDGAR 제출 자료 또는 기타 기관 출처를 사용
4. **웹 검색을 기본 데이터 출처로 절대 사용하지 말 것** - 기관 수준 분석에 필요한 정확성, 감사 추적, 신뢰성이 부족함

**이것이 중요한 이유:** MCP 출처는 적절한 인용과 함께 검증된 기관 수준 데이터를 제공합니다. 웹 검색 결과는 금융 분석에 사용하기에는 오래되었거나 부정확하거나 신뢰할 수 없을 수 있습니다.

---

## 개요
이 skill은 운영 지표, 가치평가 배수, 통계적 벤치마킹을 결합한 기관 수준의 비교기업 분석을 구축하는 방법을 에이전트에게 가르칩니다. 결과물은 동종 기업 비교를 통해 정보에 기반한 투자 의사결정을 가능하게 하는 구조화된 Excel/스프레드시트입니다.

**참조 자료 및 맥락화:**

비교기업 분석 예시는 `examples/comps_example.xlsx`에 제공됩니다. 이 skill 디렉터리의 이 파일 또는 다른 예시 파일을 사용할 때는 지능적으로 활용하세요.

**예시를 사용할 때 해야 할 일:**
- 구조적 계층 이해 (섹션이 어떻게 이어지는지)
- 요구되는 엄격성 수준 파악 (통계적 깊이, 문서화 기준)
- 원칙 학습 (명확한 헤더, 투명한 수식, 감사 추적)

**예시를 사용할 때 하지 말아야 할 일:**
- 형식이나 지표를 그대로 재현
- 맥락을 고려하지 않고 레이아웃 복사
- 청중과 관계없이 동일한 시각적 스타일 적용

**항상 먼저 자문할 질문:**
1. **"선호하는 형식이 있나요, 아니면 템플릿 스타일에 맞출까요?"**
2. **"청중은 누구인가요?"** (투자위원회, 이사회 프레젠테이션, 빠른 참조, 상세 메모)
3. **"핵심 질문은 무엇인가요?"** (가치평가, 성장 분석, 경쟁적 포지셔닝, 효율성)
4. **"맥락은 무엇인가요?"** (M&A 평가, 투자 의사결정, 섹터 벤치마킹, 성과 검토)

**구체적인 상황에 맞게 조정하세요:**
- **산업 맥락**: 대형 기술주 메가캡은 신흥 SaaS 스타트업과 다른 지표가 필요합니다.
- **섹터별 요구사항**: 관련 지표를 초기에 추가하세요 (예: 기술 기업에는 클라우드 ARR, 엔터프라이즈 고객, 개발자 생태계)
- **기업 친숙도**: 잘 알려진 기업은 배경 설명을 줄이고 차이 분석에 더 집중할 수 있습니다.
- **의사결정 유형**: M&A는 지속적인 포트폴리오 모니터링과 다른 강조점이 필요합니다.

**핵심 원칙:** 템플릿의 원칙(명확한 구조, 통계적 엄격성, 투명한 수식)은 사용하되 맥락에 따라 실행을 달리하세요. 목표는 기관처럼 보이는 템플릿이 아니라 기관 수준의 분석입니다.

사용자가 제공한 예시와 명시적인 선호사항이 항상 기본값보다 우선합니다.

## 핵심 철학
**"먼저 올바른 구조를 구축하고, 그다음 데이터가 이야기를 말하게 하세요."**

중요한 사항을 전략적으로 생각하게 만드는 헤더에서 시작하고, 깨끗한 입력 데이터를 넣고, 투명한 수식을 구축하면 통계가 자동으로 드러납니다. 좋은 비교기업 분석은 작성하지 않은 사람이 보더라도 즉시 읽고 이해할 수 있어야 합니다.

---

## ⚠️ 중요: 하드코딩보다 수식 + 단계별 검증

**하드코딩이 아닌 수식:**
- 모든 파생 값(마진, 배수, 통계)은 입력 셀을 참조하는 Excel 수식이어야 합니다 — 미리 계산한 숫자를 붙여넣지 마세요.
- Python/openpyxl로 시트를 구축할 때: `cell.value = "=E7/C7"` (수식 문자열)을 쓰고 `cell.value = 0.687` (계산 결과)은 쓰지 마세요.
- 하드코딩할 수 있는 값은 원시 입력 데이터(매출, EBITDA, 주가 등)뿐이며 — 이 값들은 모두 출처가 적힌 셀 주석을 가져야 합니다.
- 이유: 입력값이 바뀌면 모델이 자동으로 업데이트되어야 합니다. 하드코딩된 마진은 언제든 발생할 수 있는 조용한 버그입니다.

**사용자와 단계별로 검증하세요:**
- 구조 설정 후 → 데이터를 채우기 전에 사용자에게 헤더 레이아웃을 보여주세요.
- 원시 입력값 입력 후 → 수식을 구축하기 전에 입력 블록을 보여주고 출처/기간을 확인받으세요.
- 운영 지표 수식 구축 후 → 가치평가로 넘어가기 전에 계산된 마진을 보여주고 사용자와 상식 검사를 수행하세요.
- 가치평가 배수 구축 후 → 통계를 추가하기 전에 배수를 보여주고 합리적으로 보이는지 확인받으세요.
- 시트 전체를 처음부터 끝까지 구축한 뒤 제시하지 마세요 — 각 섹션을 확인하여 오류를 조기에 발견하세요.

---

## 섹션 1: 문서 구조 및 설정

### 헤더 블록 (1~3행)
```
Row 1: [ANALYSIS TITLE] - COMPARABLE COMPANY ANALYSIS
Row 2: [List of Companies with Tickers] • [Company 1 (TICK1)] • [Company 2 (TICK2)] • [Company 3 (TICK3)]
Row 3: As of [Period] | All figures in [USD Millions/Billions] except per-share amounts and ratios
```

**이것이 중요한 이유:** 맥락을 즉시 확립합니다. 이 파일을 여는 누구나 무엇을 보고 있는지, 언제 작성되었는지, 숫자를 어떻게 해석해야 하는지 알 수 있습니다.

### 시각적 규칙 표준 (선택 사항 - 사용자 선호와 업로드된 템플릿이 항상 우선)

**중요: 이는 제안된 기본값일 뿐입니다. 항상 다음 순서로 우선순위를 두세요:**
1. 사용자의 명시적인 서식 선호
2. 업로드된 템플릿 파일의 서식
3. 회사/팀 스타일 가이드
4. 이 기본값 (다른 지침이 없을 때만)

**권장 글꼴 및 타이포그래피:**
- **글꼴**: Times New Roman (전문적이고 가독성이 높으며 업계 표준)
- **글꼴 크기**: 데이터 셀 11pt, 헤더 12pt
- **굵은 텍스트**: 섹션 헤더, 회사명, 통계 레이블

**기본 색상 및 음영 — 전문적인 파란색/회색 팔레트 (최소화가 더 좋음):**
- **절제해서 사용하세요** — 파란색과 회색만 사용하세요. 녹색, 주황색, 빨간색 또는 여러 강조 색상을 도입하지 마세요. 깔끔한 비교기업 시트는 총 3~4가지 색상을 사용합니다.
- **섹션 헤더** (예: "OPERATING STATISTICS & FINANCIAL METRICS"):
  - 진한 파란색 배경 (`#1F4E79` 또는 `#17365D` 네이비)
  - 흰색 굵은 텍스트
  - 모든 열에 걸친 전체 행 음영
- **열 헤더** (예: "Company", "Revenue", "Margin"):
  - 연한 파란색 배경 (`#D9E1F2` 또는 이와 유사한 옅은 파란색)
  - 검은색 굵은 텍스트
  - 가운데 정렬
- **데이터 행**:
  - 회사 데이터는 흰색 배경
  - 수식은 검은색 텍스트, 하드코딩 입력값은 파란색 텍스트
- **통계 행** (Maximum, 75th Percentile 등):
  - 연한 회색 배경 (`#F2F2F2`)
  - 검은색 텍스트, 레이블은 왼쪽 정렬
- **팔레트는 이것이 전부입니다**: 진한 파란색 + 연한 파란색 + 연한 회색 + 흰색. 사용자의 템플릿이 달리 지정하는 경우를 제외하고 다른 색상은 사용하지 마세요.

**권장 서식 규칙:**
- **소수점 정밀도**:
  - 백분율: 소수점 1자리 (12.3%)
  - 배수: 소수점 1자리 (13.5x)
  - 금액: 소수점 없음, 천 단위 구분 기호 (69,632)
  - 백분율로 표시되는 마진: 소수점 1자리 (68.7%)
- **테두리**: 테두리 없음 (깔끔하고 최소한의 표현)
- **정렬**: 깔끔하고 균일한 표현을 위해 모든 지표 가운데 정렬
- **셀 크기**: 모든 열 너비는 균일/동일하게, 모든 행 높이는 일관되게 설정 (깔끔하고 전문적인 격자 생성)

**참고:** 사용자가 템플릿을 제공하거나 다른 서식을 지정하면 그 형식을 사용하세요.

## 섹션 2: 운영 통계 및 재무 지표

### 핵심 열 (다음 열부터 시작)
1. **Company** - 일관된 형식의 회사명
2. **Revenue** - 규모 지표 (맥락에 따라 LTM, 분기 또는 연간일 수 있음)
3. **Revenue Growth** - 전년 대비 백분율 변화
4. **Gross Profit** - 매출에서 매출원가를 차감한 값
5. **Gross Margin** - GP/Revenue (기본 수익성)
6. **EBITDA** - 이자, 세금, 감가상각비 및 무형자산상각비 차감 전 이익
7. **EBITDA Margin** - EBITDA/Revenue (운영 효율성)

### 선택적 추가 항목 (산업/목적에 따라 선택)
- **Quarterly vs LTM** - 계절성이 중요하면 둘 다 포함
- **Free Cash Flow** - 자본집약적 기업 또는 SaaS 기업
- **FCF Margin** - FCF/Revenue (현금 창출 효율성)
- **Net Income** - 성숙하고 수익성이 있는 기업
- **Operating Income** - D&A가 기업마다 다른 경우
- **CapEx metrics** - 자산집약적 산업
- **Rule of 40** - 특히 SaaS용 (성장률 % + 마진 %)
- **FCF Conversion** - 이익의 질 분석용 (고급)

### 수식 예시 (7행 사용)
```excel
// Core ratios - these are always calculated
Gross Margin (F7): =E7/C7
EBITDA Margin (H7): =G7/C7

// Optional ratios - include if relevant
FCF Margin: =[FCF]/[Revenue]
Net Margin: =[Net Income]/[Revenue]
Rule of 40: =[Growth %]+[FCF Margin %]
```

**황금률:** 모든 비율은 [Something] / [Revenue] 또는 [이 시트의 Something] / [Something]이어야 합니다. 단순하게 유지하세요.

### 통계 블록 (회사 데이터 뒤)

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

**통계가 필요하지 않은 열 (규모 지표):**
- Revenue, EBITDA, Net Income (절대 규모는 기업 규모에 따라 달라짐)
- Market Cap, Enterprise Value (서로 다른 규모의 기업 간에는 비교할 수 없음)

**참고:** 시각적 구분을 위해 회사 데이터와 통계 행 사이에 빈 행 하나를 추가하세요. "SECTOR STATISTICS" 또는 "VALUATION STATISTICS" 헤더 행은 추가하지 마세요.

**사분위수가 중요한 이유:** 평균뿐 아니라 분포를 보여줍니다. 75번째 백분위 배수는 "프리미엄" 기업이 거래되는 수준을 알려줍니다.

---

## 섹션 3: 가치평가 배수 및 투자 지표

### 핵심 가치평가 열 (다음 열부터 시작)
1. **Company** - 운영 섹션과 동일한 순서
2. **Market Cap** - 현재 시장 가치평가
3. **Enterprise Value** - 시가총액 ± 순부채/현금
4. **EV/Revenue** - 시장이 매출 1달러당 지불하는 금액
5. **EV/EBITDA** - 시장이 이익 1달러당 지불하는 금액
6. **P/E Ratio** - 순이익 대비 가격

### 선택적 가치평가 지표 (맥락에 따라 선택)
- **FCF Yield** - FCF/Market Cap (현금 중심 분석용)
- **PEG Ratio** - P/E/Growth Rate (성장 기업용)
- **Price/Book** - 시장가치와 장부가치 비교 (자산집약적 기업용)
- **ROE/ROA** - 수익성 비교를 위한 수익률 지표
- **Revenue/EBITDA CAGR** - 과거 성장률 (추세 분석용)
- **Asset Turnover** - Revenue/Assets (운영 효율성)
- **Debt/Equity** - 레버리지 (자본 구조 분석용)

**핵심 원칙:** 산업에 중요한 핵심 배수 3~5개를 포함하세요. 가능한 모든 지표를 넣을 수 있다는 이유만으로 전부 포함하지 마세요.

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
**중요:** 가치평가 배수는 운영 지표 섹션을 반드시 참조해야 합니다. 동일한 원시 데이터를 두 번 입력하지 마세요. 매출이 C7에 있다면 EV/Revenue 수식은 C7을 참조해야 합니다.

### 통계 블록
운영 섹션과 동일한 구조: 모든 지표에 대해 최대값, 75번째 백분위, 중앙값, 25번째 백분위, 최소값을 사용하세요. 회사 데이터와 통계 사이를 시각적으로 구분할 수 있도록 빈 행 하나를 추가하세요. "VALUATION STATISTICS" 헤더 행은 추가하지 마세요.

---

## 섹션 4: 주석 및 방법론 문서화

### 필수 구성요소

**데이터 출처 및 품질:**
- 데이터는 어디에서 가져왔나요? (S&P Kensho MCP, FactSet MCP, Daloopa MCP, Bloomberg, SEC 제출 자료)
- 어떤 기간을 다루나요? (2024년 4분기, 감사 완료 수치)
- 어떻게 검증했나요? (10-K/10-Q와 교차 확인)
- 참고: 더 나은 정확성과 추적성을 위해 사용 가능한 경우 MCP 데이터 출처(S&P Kensho, FactSet, Daloopa)를 우선하세요.

**핵심 정의:**
- EBITDA 계산 방법 (Gross Profit + D&A 또는 Operating Income + D&A)
- Free Cash Flow 수식 (Operating CF - CapEx)
- 특수 지표 설명 (Rule of 40, FCF Conversion)
- 기간 정의 (LTM, CAGR 계산 기간)

**가치평가 방법론:**
- Enterprise Value는 어떻게 계산했나요? (Market Cap + Net Debt)
- 어떤 성장률을 사용했나요? (과거 CAGR, 선행 추정치)
- 조정 사항이 있나요? (일회성 항목 제외, 정상화된 마진)

**분석 프레임워크:**
- 투자 논지는 무엇인가요? (클라우드/SaaS 효율성)
- 어떤 지표가 가장 중요한가요? (현금 창출, 자본 효율성)
- 독자는 통계를 어떻게 해석해야 하나요? (사분위수가 맥락을 제공)

---

## 섹션 5: 적절한 지표 선택 (의사결정 프레임워크)

### 먼저 "어떤 질문에 답하고 있는가?"를 시작점으로 삼으세요

**"어느 기업이 저평가되어 있는가?"**
→ 집중할 항목: EV/Revenue, EV/EBITDA, P/E, Market Cap
→ 제외할 항목: 운영 세부사항, 성장 지표

**"어느 기업이 가장 효율적인가?"**
→ 집중할 항목: Gross Margin, EBITDA Margin, FCF Margin, Asset Turnover
→ 제외할 항목: 규모 지표, 절대 금액

**"어느 기업이 가장 빠르게 성장하고 있는가?"**
→ 집중할 항목: Revenue Growth %, EBITDA CAGR, User/Customer Growth
→ 제외할 항목: 마진 지표, 레버리지 비율

**"어느 기업이 최고의 현금 창출 기업인가?"**
→ 집중할 항목: FCF, FCF Margin, FCF Conversion, CapEx intensity
→ 제외할 항목: EBITDA, P/E ratios

### 산업별 지표 선택

**소프트웨어/SaaS:**
필수: Revenue Growth, Gross Margin, Rule of 40
선택: ARR, Net Dollar Retention, CAC Payback
제외: Asset Turnover, Inventory metrics

**제조/산업재:**
필수: EBITDA Margin, Asset Turnover, CapEx/Revenue
선택: ROA, Inventory Turns, Backlog
제외: Rule of 40, SaaS metrics

**금융 서비스:**
필수: ROE, ROA, Efficiency Ratio, P/E
선택: Net Interest Margin, Loan Loss Reserves
제외: Gross Margin, EBITDA (은행에는 의미가 없음)

**소매/전자상거래:**
필수: Revenue Growth, Gross Margin, Inventory Turnover
선택: Same-Store Sales, Customer Acquisition Cost
제외: Heavy R&D or CapEx metrics

### "5-10 규칙"

**운영 지표 5개** - Revenue, Growth, 2-3 margins/efficiency metrics
**가치평가 지표 5개** - Market Cap, EV, 3 multiples
**= 총 10개 열** - 흐름을 잃지 않으면서 이야기를 전달하기에 충분한 수

지표가 15개를 넘는다면 아마 잡음을 포함하고 있는 것입니다. 가차 없이 편집하세요.

---

## 섹션 6: 모범 사례 및 품질 검사

### 시작하기 전에
1. **동종 그룹 정의** - 기업은 진정으로 비교 가능해야 합니다 (유사한 비즈니스 모델, 규모, 지역)
2. **적절한 기간 선택** - LTM은 계절성을 완화하고, 분기는 추세를 보여줍니다.
3. **단위를 미리 표준화** - 백만 단위와 십억 단위 중 어떤 것을 선택하는지가 모든 것에 영향을 줍니다.
4. **데이터 출처 매핑** - 각 숫자가 어디에서 오는지 파악하세요.

### 구축하면서
1. **먼저 모든 원시 데이터 입력** - 수식을 작성하기 전에 파란색 텍스트를 완성하세요.
2. **모든 하드코딩 입력값에 셀 주석 추가** - 셀을 마우스 오른쪽 버튼으로 클릭 → 주석 삽입 → 출처 또는 가정 문서화

   **출처 데이터의 경우 출처를 정확히 인용하세요:**
   - 예: "Bloomberg Terminal - MSFT Equity DES, accessed 2024-10-02"
   - 예: "Q4 2024 10-K filing, page 42, line item 'Total Revenue'"
   - 예: "FactSet consensus estimate as of 2024-10-02"
   - **가능하면 하이퍼링크 포함**: 셀을 마우스 오른쪽 버튼으로 클릭 → 링크 → SEC 제출 자료, 데이터 출처 또는 보고서 URL 붙여넣기

   **가정의 경우 추론을 설명하세요:**
   - 예: "Assumed 15% EBITDA margin based on peer median, company does not disclose"
   - 예: "Estimated Enterprise Value as Market Cap + $50M net debt (from Q3 balance sheet, Q4 not yet available)"
   - 예: "Forward P/E based on street consensus EPS of $3.45 (average of 12 analyst estimates)"

   **이것이 중요한 이유**: 감사 추적, 데이터 검증, 가정의 투명성 및 향후 업데이트를 가능하게 합니다.
3. **행별로 수식 구축** - 다음 단계로 넘어가기 전에 각 계산을 테스트하세요.
4. **헤더에는 절대 참조 사용** - $C$6은 헤더 행을 고정합니다.
5. **일관된 형식 적용** - 백분율을 소수가 아닌 백분율로 표시하세요.
6. **조건부 서식 추가** - 이상치를 자동으로 강조 표시하세요.

### 상식 검사
- **마진 테스트**: 총마진 > EBITDA 마진 > 순마진 (정의상 항상 참)
- **배수의 합리성**: 
  - EV/Revenue: 일반적으로 0.5-20x (산업에 따라 크게 다름)
  - EV/EBITDA: 일반적으로 8-25x (산업 전반에서 비교적 일관됨)
  - P/E: 일반적으로 10-50x (성장률에 따라 달라짐)
- **성장률-배수 상관관계**: 성장률이 높을수록 일반적으로 배수도 높습니다.
- **규모-효율성 상충관계**: 규모가 큰 기업은 종종 더 나은 마진을 보입니다 (규모의 이점).

### 피해야 할 일반적인 실수
❌ 수식에서 시가총액과 기업가치를 혼동
❌ 분자와 분모에 서로 다른 기간 사용 (LTM 대 분기)
❌ 셀 참조 대신 숫자를 수식에 하드코딩
❌ **출처를 인용하거나 가정을 설명하는 셀 주석이 없는 하드코딩 입력값**
❌ 가능한 경우 SEC 제출 자료 또는 데이터 출처에 대한 하이퍼링크 누락
❌ 명확한 목적 없이 너무 많은 지표 포함
❌ 비교할 수 없는 기업 포함 (서로 다른 비즈니스 모델)
❌ 공개 없이 오래된 데이터 사용
❌ 백분율의 평균을 잘못 계산 (중앙값이어야 함)

---

## 섹션 6: 고급 기능

### 동적 헤더
계산을 표시하는 열에는 명확한 단위 레이블을 사용하세요:
```
Revenue Growth (YoY) % | EBITDA Margin | FCF Margin | Rule of 40
```

### 사분위수 분석의 이점
단순한 평균/중앙값 대신 사분위수는 다음을 보여줍니다:
- **75번째 백분위** = "프리미엄" 기업이 이 수준에서 거래됨
- **중앙값** = 일반적인 시장 가치평가
- **25번째 백분위** = "할인" 영역

이를 통해 다음 질문에 답할 수 있습니다: "우리의 대상 기업은 동종 기업 대비 비싸게 거래되고 있는가, 싸게 거래되고 있는가?"

### 산업별 수정

**소프트웨어/SaaS:**
- 추가: ARR, Net Dollar Retention, CAC Payback Period
- 강조: Rule of 40, FCF margins, gross margins >70%

**헬스케어:**
- 추가: R&D/Revenue, Pipeline value, Regulatory status
- 강조: EBITDA margins, growth rates, reimbursement risk

**산업재:**
- 추가: Backlog, Order book trends, Geographic mix
- 강조: ROIC, asset turnover, cyclical adjustments

**소비재:**
- 추가: Same-store sales, Customer acquisition cost, Brand value
- 강조: Revenue growth, gross margins, inventory turns

---

## 섹션 7: 워크플로 및 실무 팁

### 단계별 프로세스
1. **구조 설정** (30분)
   - 모든 헤더 생성
   - 셀 서식 지정 (입력값은 파란색, 수식은 검은색)
   - 단위와 날짜 참조 확정

2. **데이터 수집** (60~90분)
   - 1차 출처에서 가져오기 (S&P Kensho MCP, FactSet MCP, 사용 가능한 경우 Daloopa MCP; 그렇지 않으면 Bloomberg, SEC)
   - 모든 원시 숫자를 파란색으로 입력
   - 주석 섹션에 출처 문서화

3. **수식 구축** (30분)
   - 간단한 비율(마진)부터 시작
   - 배수(EV/Revenue)로 진행
   - 교차 확인 추가 (마진이 합리적인가?)

4. **통계 추가** (15분)
   - 모든 열에 수식 구조 복사
   - 범위가 올바른지 확인 (B7:B9이지 B7:B10이 아님)
   - 사분위수 로직 확인

5. **품질 관리** (30분)
   - 상식 검사 실행
   - 수식 참조 검증
   - #DIV/0! 또는 #REF! 오류 확인
   - 알려진 벤치마크와 비교

6. **문서화** (15분)
   - 주석 섹션 완성
   - 데이터 출처 추가
   - 방법론 정의
   - 분석 날짜 기록

### 실무 팁
- **템플릿 저장**: 한 번 구축하고 계속 재사용하세요.
- **이상치에 색상 지정**: 표준편차 2배를 초과하는 값에 조건부 서식 적용
- **출처 파일 연결**: Bloomberg 스크린샷 또는 SEC 제출 자료에 하이퍼링크
- **버전 관리**: "Comps_v1_2024-12-15"처럼 명확한 날짜와 함께 저장
- **협업 검토**: 다른 사람이 수식을 확인하게 하세요.

### Excel 서식 체크리스트 (선택 사항 - 사용자 선호에 맞게 조정)
- [ ] 사용자가 선호하는 스타일로 글꼴 설정 (기본값: Times New Roman, 데이터 11pt, 헤더 12pt)
- [ ] 사용자 템플릿에 따라 섹션 헤더 서식 지정 (기본값: 흰색 굵은 텍스트와 진한 파란색 #17365D)
- [ ] 사용자 템플릿에 따라 열 헤더 서식 지정 (기본값: 검은색 굵은 텍스트와 연한 파란색/회색 #D9E2F3)
- [ ] 사용자 템플릿에 따라 통계 행 서식 지정 (기본값: 연한 회색 #F2F2F2)
- [ ] 테두리 적용 안 함 (깔끔하고 최소한의 표현)
- [ ] **열 너비를 균일/동일하게 설정** (깔끔하고 전문적인 표현 생성)
- [ ] **행 높이를 일정하게 설정** (데이터 행은 일반적으로 20~25pt)
- [ ] 적절한 소수점 정밀도와 천 단위 구분 기호로 숫자 서식 지정
- [ ] **깔끔하고 균일한 표현을 위해 모든 지표를 가운데 정렬**
- [ ] **회사 데이터와 통계 행 사이를 구분하는 빈 행 하나**
- [ ] **별도의 "SECTOR STATISTICS" 또는 "VALUATION STATISTICS" 헤더 행 없음**
- [ ] **모든 하드코딩 입력 셀에 다음 중 하나가 포함된 주석: (1) 정확한 데이터 출처 또는 (2) 가정 설명**
- [ ] **해당하는 셀에 하이퍼링크 추가** (SEC 제출 자료, 데이터 제공업체 페이지, 보고서)

---

## 섹션 8: 예시 템플릿 레이아웃

**간단한 버전 (여기서 시작):**
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

**필요할 때만 복잡성을 추가하세요:**
- 계절성이 중요하면 분기 AND LTM 포함
- 현금 창출이 핵심 이야기라면 FCF 지표 추가
- 산업별 지표 포함 (SaaS의 경우 Rule of 40 등)
- 기업이 5개를 초과하면 통계 행 추가

---

## 섹션 9: 산업별 추가 항목 (선택 사항)

분석에 핵심적인 경우에만 추가하세요. 대부분의 비교기업 분석은 핵심 지표만으로도 충분합니다.

**소프트웨어/SaaS:**
관련성이 있으면 추가: ARR, Net Dollar Retention, Rule of 40

**금융 서비스:**
관련성이 있으면 추가: ROE, Net Interest Margin, Efficiency Ratio

**전자상거래:**
관련성이 있으면 추가: GMV, Take Rate, Active Buyers

**헬스케어:**
관련성이 있으면 추가: R&D/Revenue, Pipeline Value, Patent Timeline

**제조:**
관련성이 있으면 추가: Asset Turnover, Inventory Turns, Backlog

---

## 섹션 10: 위험 신호 및 경고 징후

### 데이터 품질 문제
🚩 일관되지 않은 기간 (분기와 연간 혼용)  
🚩 설명 없는 데이터 누락  
🚩 데이터 출처 간 큰 차이 (>10% 차이)

### 가치평가 위험 신호
🚩 EBITDA가 음수인 기업을 EBITDA 배수로 가치평가 (대신 매출 배수 사용)  
🚩 초고성장 스토리 없이 P/E 비율이 100배 초과  
🚩 산업에 맞지 않는 마진

### 비교 가능성 문제
🚩 서로 다른 회계연도 말 (시점 문제 발생)  
🚩순수 사업 기업과 복합기업 혼합  
🚩 실질적으로 다른 비즈니스 모델을 "comps"로 표시

**확신이 없으면 기업을 제외하세요.** 의심스러운 기업 6개보다 완벽한 비교기업 3개가 낫습니다.

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

1. **구조가 통찰을 이끈다** - 올바른 헤더가 올바른 사고를 강제합니다.
2. **적을수록 좋다** - 중요하지 않은 지표 20개보다 중요한 지표 5~10개가 낫습니다.
3. **질문에 맞는 지표 선택** - 가치평가 분석 ≠ 효율성 분석
4. **통계가 패턴을 보여준다** - 중앙값/사분위수가 평균보다 더 많은 것을 드러냅니다.
5. **복잡성보다 투명성** - 누구나 이해할 수 있는 단순한 수식
6. **비교 가능성이 최우선** - 나쁜 비교를 억지로 맞추기보다 제외하는 편이 낫습니다.
7. **선택을 문서화** - 어떤 지표를 선택했는지, 왜 선택했는지 주석 섹션에 설명합니다.

---

## 결과물 체크리스트

비교기업 분석을 전달하기 전에 다음을 확인하세요:
- [ ] 모든 기업이 실제로 비교 가능한가
- [ ] 데이터가 일관된 기간에서 나온 것인가
- [ ] 단위가 명확하게 표시되어 있는가 (백만/십억)
- [ ] 수식이 하드코딩된 값이 아니라 셀을 참조하는가
- [ ] **모든 하드코딩 입력 셀에 다음 중 하나가 포함된 주석이 있는가: (1) 인용이 포함된 정확한 데이터 출처 또는 (2) 설명이 포함된 명확한 가정**
- [ ] **해당하는 곳에 하이퍼링크가 추가되어 있는가** (SEC EDGAR 제출 자료, Bloomberg 페이지, 리서치 보고서)
- [ ] 통계에 최소 5개 지표가 포함되는가 (최대값, 75번째, 중앙값, 25번째, 최소값)
- [ ] 주석 섹션에 출처와 방법론이 문서화되어 있는가
- [ ] 시각적 서식이 규칙을 따르는가 (파란색 = 입력값, 검은색 = 수식)
- [ ] 상식 검사를 통과하는가 (마진이 논리적이고 배수가 합리적인가)
- [ ] 날짜가 현재 날짜로 기록되어 있는가 ("As of [Date]")
- [ ] 수식 감사에서 오류가 없는가 (#DIV/0!, #REF!, #N/A)

---

## 지속적인 개선

비교기업 분석을 완료한 뒤 다음을 질문하세요:
1. 통계가 예상하지 못한 통찰을 보여주었는가?
2. 분석을 제한한 데이터 공백이 있었는가?
3. 이해관계자가 포함하지 않은 지표를 요청했는가?
4. 실제 소요 시간과 적정 소요 시간은 각각 얼마였는가?
5. 다음에 이 결과물을 더 유용하게 만들려면 무엇이 필요한가?

최고의 비교기업 분석은 매번의 반복을 통해 발전합니다. 템플릿을 저장하고, 피드백에서 배우고, 의사결정자가 실제로 사용하는 내용을 바탕으로 구조를 개선하세요.


## 데이터 출처 — MCP 우선, 웹 대체

아래의 많은 문단은 S&P Kensho MCP / Daloopa MCP / FactSet MCP를 사용하라고 안내합니다. 이는 원래 Cowork 맥락의 상용 금융 데이터 MCP입니다. Hermes에서는 다음을 따르세요:

- **구조화된 금융 데이터 MCP가 구성되어 있다면** (Hermes는 MCP를 지원합니다 — `native-mcp` skill 참조), 이를 시점별 비교기업 분석, 선례 거래 및 제출 자료에 우선 사용하세요.
- **그렇지 않으면** 다음으로 대체하세요:
  - 미국 제출 자료는 SEC EDGAR (`https://www.sec.gov/cgi-bin/browse-edgar`)에 대한 `web_search` / `web_extract`
  - 기업 IR 페이지의 보도자료 및 실적 발표 자료
  - 대화형 데이터 포털에는 `browser_navigate`
  - 사용자 제공 데이터 (맥락에 데이터가 없으면 명시적으로 요청)
- **절대 조작하지 마세요.** 배수, 선례 또는 제출 자료 수치를 출처로 확인할 수 없다면 셀에 `[UNSOURCED]`라고 표시하고 사용자에게 알리세요.

## 출처 표시

이 skill은 Anthropic의 Claude for Financial Services 플러그인 모음에서 적용되었습니다 (Apache-2.0). Office-JS / Cowork 실시간 Excel 경로는 제거되었으며, 이 버전은 `excel-author` skill의 규칙에 따른 헤드리스 openpyxl을 대상으로 합니다. 원본: https://github.com/anthropics/financial-services

