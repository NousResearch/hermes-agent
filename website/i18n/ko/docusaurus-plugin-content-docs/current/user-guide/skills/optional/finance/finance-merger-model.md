---
title: "Merger Model — Excel에서 희석/증가 (합병) 모델 구축 — 추정 손익계산서, 시너지, 자금 조달 비율, EPS 영향"
sidebar_label: "Merger Model"
description: "Excel에서 희석/증가 (합병) 모델 구축 — 추정 손익계산서, 시너지, 자금 조달 비율, EPS 영향"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Merger Model

Excel에서 희석/증가 (합병) 모델을 구축합니다 — 추정 손익계산서(pro-forma P&L), 시너지, 자금 조달 비율, EPS 영향을 포함합니다. `excel-author`와 함께 사용하세요. M&A 피치, 이사회 자료 또는 거래 평가를 위해 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/finance/merger-model`로 설치 |
| Path | `optional-skills/finance/merger-model` |
| Version | `1.0.0` |
| Author | Anthropic (Nous Research에 의해 수정됨) |
| License | Apache-2.0 |
| Platforms | linux, macos, windows |
| Tags | `finance`, `m-and-a`, `merger`, `accretion-dilution`, `excel`, `openpyxl`, `modeling`, `investment-banking` |
| Related skills | [`excel-author`](/docs/user-guide/skills/optional/finance/finance-excel-author), [`pptx-author`](/docs/user-guide/skills/optional/finance/finance-pptx-author), [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`3-statement-model`](/docs/user-guide/skills/optional/finance/finance-3-statement-model) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

## 환경

이 스킬은 **헤드리스 openpyxl**을 가정합니다 — 즉, 디스크에 .xlsx 파일을 생성합니다.
셀 색상, 수식, 명명된 범위 및 민감도 분석표에 대해서는 `excel-author` 스킬의 규칙을 따르세요.
전달하기 전에 다시 계산하십시오: `python /path/to/excel-author/scripts/recalc.py ./out/model.xlsx`.

# Merger Model

M&A 거래에 대한 희석/증가(accretion/dilution) 분석을 구축합니다. 추정 EPS 영향, 시너지 민감도 및 매수가격 배분을 모델링합니다. 잠재적 인수 평가, 피치 북을 위한 합병 결과 분석 준비 또는 거래 조건 자문에 사용하세요.

## 워크플로우

### 1단계: 입력값 수집

**인수기업 (Acquirer):**
- 회사명, 현재 주가, 발행 주식수
- LTM 및 NTM EPS (GAAP 및 조정)
- P/E 배수
- 세전 타인자본비용, 세율
- 대차대조표상의 현금, 기존 부채

**피인수기업 (Target):**
- 회사명, 현재 주가, 발행 주식수 (상장사인 경우)
- LTM 및 NTM EPS 또는 순이익
- 기업 가치 (Enterprise value) 또는 지분 가치 (Equity value)

**거래 조건 (Deal Terms):**
- 주당 제안 가격 (또는 현재 대비 프리미엄)
- 대가 비율: 현금 % vs 주식 %
- 현금 부분을 조달하기 위해 차입한 신규 부채
- 예상 시너지 (수익 및 비용) 및 단계적 적용 일정
- 거래 수수료 및 자금 조달 비용
- 예상 마감일

### 2단계: 매수가격 분석 (Purchase Price Analysis)

| 항목 | 금액 |
|------|-------|
| 주당 제안 가격 (Offer price per share) | |
| 현재 대비 프리미엄 (Premium to current) | |
| 지분 가치 (Equity value) | |
| 더하기: 인수된 순부채 (net debt assumed) | |
| 기업 가치 (Enterprise value) | |
| 내재 EV / EBITDA (EV / EBITDA implied) | |
| 내재 P/E (P/E implied) | |

### 3단계: 자금의 조달 및 운용 (Sources & Uses)

| 조달 (Sources) | $ | 운용 (Uses) | $ |
|---------|---|------|---|
| 신규 부채 (New debt) | | 지분 매수가격 (Equity purchase price) | |
| 보유 현금 (Cash on hand) | | 피인수기업 부채 차환 (Refinance target debt) | |
| 신규 주식 발행 (New equity issued) | | 거래 수수료 (Transaction fees) | |
| | | 자금 조달 수수료 (Financing fees) | |
| **총계 (Total)** | | **총계 (Total)** | |

### 4단계: 추정 EPS (증가 / 희석) (Pro Forma EPS)

연도별 계산 (Year 1-3):

| | 독립적 (Standalone) | 추정 (Pro Forma) | 증가/(희석) (Accretion/(Dilution)) |
|---|-----------|-----------|---------------------|
| 인수기업 순이익 | | | |
| 피인수기업 순이익 | | | |
| 시너지 (세후) | | | |
| 사용된 현금에 대한 이자 수익 포기 (세후) | | | |
| 신규 부채 이자 (세후) | | | |
| 무형자산 상각비 (세후) | | | |
| 추정 순이익 | | | |
| 추정 발행 주식수 | | | |
| **추정 EPS (Pro forma EPS)** | | | |
| **증가 / (희석) %** | | | |

### 5단계: 민감도 분석 (Sensitivity Analysis)

**시너지 및 제안 프리미엄에 따른 증가/희석:**

| | $0M syn | $25M syn | $50M syn | $75M syn | $100M syn |
|---|---------|----------|----------|----------|-----------|
| 15% premium | | | | | |
| 20% premium | | | | | |
| 25% premium | | | | | |
| 30% premium | | | | | |

**현금/주식 비율에 따른 증가/희석:**

| | 100% cash | 75/25 | 50/50 | 25/75 | 100% stock |
|---|-----------|-------|-------|-------|------------|
| Year 1 | | | | | |
| Year 2 | | | | | |

### 6단계: 손익분기 시너지 (Breakeven Synergies)

Year 1에서 거래가 EPS에 중립이 되기 위해 필요한 최소 시너지를 계산합니다.

### 7단계: 출력 (Output)

- 다음이 포함된 Excel 통합 문서:
  - 가정 탭 (Assumptions tab)
  - 자금 조달 및 운용 (Sources & uses)
  - 추정 손익계산서 (Pro forma income statement)
  - 증가/희석 요약 (Accretion/dilution summary)
  - 민감도 분석표 (Sensitivity tables)
  - 손익분기 분석 (Breakeven analysis)
- 피치 북을 위한 1페이지 합병 결과 요약

## 중요 참고 사항

- 관련된 경우 항상 GAAP 및 조정된 (현금) EPS를 모두 표시하세요.
- 주식 거래: 교환 비율에 인수기업의 현재 가격을 사용하고, 신규 주식 발행으로 인한 희석을 명시하세요.
- 매수가격 배분 포함 — 영업권 및 무형자산 상각은 GAAP EPS에 중요합니다.
- 시너지의 단계적 적용 일정은 매우 중요합니다 — Year 1은 종종 런레이트 시너지의 25-50%에 불과합니다.
- 사용된 현금에 대해 포기한 이자 수익과 새로 차입한 부채에 대한 신규 이자 비용을 잊지 마세요.
- 시너지 및 이자 조정에 대한 세율은 인수기업의 한계 세율과 일치해야 합니다.

## 데이터 출처 — MCP 우선, 웹 폴백

아래의 여러 구절에는 "S&P Kensho MCP / Daloopa MCP / FactSet MCP를 사용하라"고 되어 있습니다. 이들은 원본 Cowork 플러그인 컨텍스트의 상업용 금융 데이터 MCP입니다. Hermes에서는 다음을 따릅니다:

- **구조화된 금융 데이터 MCP가 구성되어 있는 경우** (Hermes는 MCP를 지원합니다 — `native-mcp` 스킬 참조), 특정 시점의 비교 기업 분석, 이전 거래 및 공시를 위해 우선적으로 사용하세요.
- **그렇지 않은 경우**, 다음으로 폴백하세요:
  - 미국 공시의 경우 SEC EDGAR(`https://www.sec.gov/cgi-bin/browse-edgar`)에 대한 `web_search` / `web_extract`
  - 보도자료, 실적 발표 자료를 위한 회사 IR 페이지
  - 인터랙티브 데이터 포털을 위한 `browser_navigate`
  - 사용자 제공 데이터 (컨텍스트에 데이터가 없을 때는 명시적으로 요청하세요)
- **절대 위조하지 마세요**. 배수, 이전 거래 또는 공시 수치를 찾을 수 없는 경우 셀을 `[UNSOURCED]`로 표시하고 사용자에게 알리세요.

## 저작자 표시 (Attribution)

이 스킬은 Anthropic의 Financial Services용 Claude 플러그인 제품군(Apache-2.0)에서 수정되었습니다. Office-JS / Cowork 실시간 Excel 경로는 제거되었으며, 이 버전은 `excel-author` 스킬의 규칙을 통해 헤드리스 openpyxl을 대상으로 합니다. 원본: https://github.com/anthropics/financial-services
