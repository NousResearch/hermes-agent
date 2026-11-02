---
title: "합병 모델 — Excel에서 M&A 증가/희석 워크북 구축"
sidebar_label: "합병 모델"
description: "Excel에서 M&A 증가/희석 워크북 구축"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 소스 SKILL.md를 편집하세요. */}

# 합병 모델

Excel에서 M&A 거래의 증가/희석 워크북을 구축합니다.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/finance/merger-model`로 설치 |
| 경로 | `optional-skills/finance/merger-model` |
| 버전 | `1.0.0` |
| 작성자 | Anthropic (Nous Research가 조정) |
| 라이선스 | Apache-2.0 |
| 플랫폼 | linux, macos, windows |
| 태그 | `finance`, `m-and-a`, `merger`, `accretion-dilution`, `excel`, `openpyxl`, `modeling`, `investment-banking` |
| 관련 skills | [`excel-author`](/docs/user-guide/skills/optional/finance/finance-excel-author), [`pptx-author`](/docs/user-guide/skills/optional/finance/finance-pptx-author), [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`3-statement-model`](/docs/user-guide/skills/optional/finance/finance-3-statement-model) |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 트리거될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

## 환경

이 skill은 **headless openpyxl**을 전제로 합니다 — 디스크에 `.xlsx` 파일을 생성합니다.
셀 색상, 수식, 명명된 범위, 민감도 표에는 `excel-author` skill의 규칙을 따르세요.
전달 전에 다시 계산하세요: `python /path/to/excel-author/scripts/recalc.py ./out/model.xlsx`.

# 합병 모델

M&A 거래의 증가/희석 분석을 구축합니다. 프로 forma EPS 영향, 시너지 민감도, 인수가격 배분을 모델링합니다. 잠재적 인수를 평가하거나, 피치 자료용 합병 결과 분석을 준비하거나, 거래 조건에 조언할 때 사용합니다.

## 워크플로

### 1단계: 입력값 수집

**인수자:**
- 회사명, 현재 주가, 발행 주식 수
- LTM 및 NTM EPS (GAAP 및 조정)
- P/E 배수
- 세전 부채비용, 세율
- 대차대조표상 현금, 기존 부채

**피인수자:**
- 회사명, 현재 주가, 발행 주식 수 (상장사인 경우)
- LTM 및 NTM EPS 또는 순이익
- 기업가치 또는 자기자본가치

**거래 조건:**
- 주당 제안 가격 (또는 현재가 대비 프리미엄)
- 대가 구성: 현금 비율 대 주식 비율
- 현금 부분 조달을 위해 새로 조달하는 부채
- 예상 시너지 (매출 및 비용)와 단계적 반영 일정
- 거래 수수료 및 금융 비용
- 예상 거래 종결일

### 2단계: 인수가격 분석

| 항목 | 값 |
|------|-------|
| 주당 제안 가격 | |
| 현재가 대비 프리미엄 | |
| 자기자본가치 | |
| 더하기: 인수한 순부채 | |
| 기업가치 | |
| 내재 EV / EBITDA | |
| 내재 P/E | |

### 3단계: 자금 조달원 및 사용처

| 조달원 | $ | 사용처 | $ |
|---------|---|------|---|
| 신규 부채 | | 자기자본 매입 가격 | |
| 보유 현금 | | 피인수자 부채 차환 | |
| 신규 발행 주식 | | 거래 수수료 | |
| | | 금융 수수료 | |
| **합계** | | **합계** | |

### 4단계: 프로 forma EPS (증가 / 희석)

연도별로 계산합니다 (1~3년 차):

| | 독립 기준 | 프로 forma | 증가/(희석) |
|---|-----------|-----------|---------------------|
| 인수자 순이익 | | | |
| 피인수자 순이익 | | | |
| 시너지 (세후) | | | |
| 현금의 포기 이자수익 (세후) | | | |
| 신규 부채 이자비용 (세후) | | | |
| 무형자산 상각 (세후) | | | |
| 프로 forma 순이익 | | | |
| 프로 forma 주식 수 | | | |
| **프로 forma EPS** | | | |
| **증가 / (희석) %** | | | |

### 5단계: 민감도 분석

**시너지 및 제안 프리미엄에 따른 증가/희석:**

| | $0M 시너지 | $25M 시너지 | $50M 시너지 | $75M 시너지 | $100M 시너지 |
|---|---------|----------|----------|----------|-----------|
| 프리미엄 15% | | | | | |
| 프리미엄 20% | | | | | |
| 프리미엄 25% | | | | | |
| 프리미엄 30% | | | | | |

**현금/주식 구성에 따른 증가/희석:**

| | 현금 100% | 75/25 | 50/50 | 25/75 | 주식 100% |
|---|-----------|-------|-------|-------|------------|
| 1년 차 | | | | | |
| 2년 차 | | | | | |

### 6단계: 손익분기 시너지

1년 차에 거래가 EPS 중립이 되기 위해 필요한 최소 시너지를 계산합니다.

### 7단계: 출력물

- 다음을 포함한 Excel 워크북:
  - 가정 탭
  - 자금 조달원 및 사용처
  - 프로 forma 손익계산서
  - 증가/희석 요약
  - 민감도 표
  - 손익분기 분석
- 피치북용 1페이지 합병 결과 요약

## 중요 참고 사항

- 관련되는 경우 GAAP EPS와 조정 EPS(현금 기준)를 항상 모두 표시하세요.
- 주식 거래에서는 교환 비율에 인수자의 현재 주가를 사용하고, 신규 주식으로 인한 희석을 기록하세요.
- 인수가격 배분을 포함하세요 — 영업권과 무형자산 상각은 GAAP EPS에 영향을 줍니다.
- 시너지의 단계적 반영은 중요합니다 — 1년 차에는 정상 수준 시너지의 25~50%만 반영되는 경우가 많습니다.
- 사용한 현금의 포기 이자수익과 새로 조달한 부채의 신규 이자비용을 잊지 마세요.
- 시너지 및 이자 조정에 적용하는 세율은 인수자의 한계세율과 일치해야 합니다.


## 데이터 출처 — MCP 우선, 웹 대체

아래의 많은 구절은 “S&P Kensho MCP / Daloopa MCP / FactSet MCP를 사용하라”고 말합니다. 이는 원래 Cowork 플러그인 컨텍스트의 상용 금융 데이터 MCP입니다. Hermes에서는 다음을 따르세요.

- **구조화된 금융 데이터 MCP가 구성되어 있다면** (Hermes는 MCP를 지원합니다 — `native-mcp` skill 참조), 시점 기준 비교기업, 선례 거래, 공시에는 이를 우선 사용하세요.
- **그렇지 않다면**, 다음으로 대체하세요.
  - 미국 공시에는 SEC EDGAR(`https://www.sec.gov/cgi-bin/browse-edgar`)를 대상으로 하는 `web_search` / `web_extract`
  - 회사 IR 페이지의 보도자료 및 실적 자료
  - 대화형 데이터 포털에는 `browser_navigate`
  - 사용자가 제공한 데이터 (컨텍스트에 데이터가 없으면 명시적으로 요청)
- **절대 지어내지 마세요.** 배수, 선례 또는 공시 수치를 출처로 확인할 수 없다면 해당 셀을 `[UNSOURCED]`로 표시하고 사용자에게 알려야 합니다.

## 저작자 표시

이 skill은 Anthropic의 Claude for Financial Services 플러그인 제품군(Apache-2.0)을 바탕으로 조정되었습니다. Office-JS / Cowork 실시간 Excel 경로는 제거되었으며, 이 버전은 `excel-author` skill의 규칙에 따른 headless openpyxl을 대상으로 합니다. 원본: https://github.com/anthropics/financial-services
