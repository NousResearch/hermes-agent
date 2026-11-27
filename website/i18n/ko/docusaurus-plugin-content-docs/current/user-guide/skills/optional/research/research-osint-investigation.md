---
title: "OSINT 조사 — 공공 기록과 제재 데이터로 자금 흐름 추적하기"
sidebar_label: "OSINT 조사"
description: "공공 기록과 제재 데이터로 자금 흐름 추적하기"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# OSINT 조사

공공 기록과 제재 데이터로 자금 흐름을 추적합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/research/osint-investigation`으로 설치 |
| 경로 | `optional-skills/research/osint-investigation` |
| 버전 | `0.1.0` |
| 작성자 | Hermes Agent (ShinMegamiBoson/OpenPlanter에서 수정, MIT) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `osint`, `investigation`, `public-records`, `sec`, `sanctions`, `corporate-registry`, `property`, `courts`, `due-diligence`, `journalism` |
| 관련 스킬 | [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# OSINT 조사 — 공공 기록 상호 참조

정부 계약, 기업 공시, 로비, 제재, 역외 유출 자료, 부동산 기록, 법원
기록, 웹 아카이브, 지식 베이스 및 국제 뉴스를 대상으로 하는 공공 기록 OSINT
조사 프레임워크입니다. 서로 다른 출처의 엔터티를 식별하고, 명시적인
신뢰도와 함께 연결 관계를 구축하며, 통계적 시점 검정을 수행하고, 구조화된
증거 사슬을 작성합니다.

**Python 표준 라이브러리만 사용합니다.** 설치가 필요 없습니다. Linux, macOS,
Windows에서 작동합니다. 대부분의 출처는 API 키 없이 사용할 수 있습니다
(OpenCorporates는 선택적 무료 토큰을 사용하면 요청 한도가 높아집니다).

MIT 라이선스인 ShinMegamiBoson/OpenPlanter 프로젝트를 바탕으로 수정했으며,
원본에서 다루지 않았던 신원 / 부동산 / 소송 / 아카이브 / 뉴스 출처를
확장했습니다.

## 이 스킬을 사용할 때

다음과 같은 요청에 사용합니다.

- “자금 흐름 추적” — 정부 계약, 로비 → 법안, 제재
- 기업 실사 — 회사 X를 지배하는 사람, 설립 관할지, 이사회 구성원, 제출한 공시
- 제재 심사 — 엔터티 X가 OFAC SDN, ICIJ 역외 유출 자료에 포함되는지 확인
- 대가성 거래 조사 — 역외 연계가 있는 계약업체, 수주에 성공한 로비 고객
- 부동산 소유권 — 이름 또는 주소로 등기된 증서/담보대출 찾기
  (뉴욕시; 다른 카운티는 해당 기록관을 안내)
- 소송 이력 — 연방 및 주 법원 판결과 PACER 도켓 찾기
- 명칭이 서로 다른 다중 출처 엔터티 식별(LLC 접미사, 약어)
- 명시적인 신뢰도 수준을 포함한 증거 사슬 구축
- “X에 대해 어떤 말이 나왔나” — 국제 뉴스(GDELT) + Wikipedia
  서술 + Wayback Machine으로 사라진 URL 복구

다음에는 이 스킬을 사용하지 마세요.

- 일반 웹 조사 → `web_search` / `web_extract`
- 도메인/인프라 OSINT → `domain-intel` 스킬
- 학술 문헌 → `arxiv` 스킬
- 소셜 미디어 프로필 찾기 → `sherlock` 스킬(선택 사항)
- 미국 **연방** 선거 자금 — FEC는 의도적으로 다루지 않습니다
  (무료 `DEMO_KEY` 등급에서 임의 기여자 이름 조회 시 API가 불안정함).
  연방 기부금은 https://www.fec.gov/data/ 를 직접 안내하세요.

## 워크플로

에이전트는 `terminal` 도구를 통해 스크립트를 실행합니다. `SKILL_DIR`은
이 SKILL.md를 담고 있는 디렉터리입니다.

### 1. 적용할 출처 식별

조사 계획을 세우려면 데이터 출처 위키 항목을 읽으세요.

```
ls SKILL_DIR/references/sources/

# Federal financial / regulatory
cat SKILL_DIR/references/sources/sec-edgar.md       # corporate filings
cat SKILL_DIR/references/sources/usaspending.md     # federal contracts
cat SKILL_DIR/references/sources/senate-ld.md       # lobbying
cat SKILL_DIR/references/sources/ofac-sdn.md        # sanctions
cat SKILL_DIR/references/sources/icij-offshore.md   # offshore leaks

# Identity / property / litigation / archives / news
cat SKILL_DIR/references/sources/nyc-acris.md       # NYC property records
cat SKILL_DIR/references/sources/opencorporates.md  # global corporate registry
cat SKILL_DIR/references/sources/courtlistener.md   # court records (federal + state)
cat SKILL_DIR/references/sources/wayback.md         # Wayback Machine archives
cat SKILL_DIR/references/sources/wikipedia.md       # Wikipedia + Wikidata
cat SKILL_DIR/references/sources/gdelt.md           # global news monitoring
```

각 항목은 9개 섹션 템플릿을 따릅니다. 요약, 접근, 스키마, 범위,
상호 참조 키, 데이터 품질, 수집, 법률, 참고 문헌입니다.

**상호 참조 가능성** 섹션은 출처 간 결합 키를 보여 줍니다. 적절한 조합을
고르려면 먼저 이 섹션을 읽으세요.

### 2. 데이터 수집

각 출처에는 `SKILL_DIR/scripts/`에 표준 라이브러리만 사용하는 가져오기
스크립트가 있습니다.

**Federal financial / regulatory**

```bash
# SEC EDGAR filings (corporate disclosures)
python3 SKILL_DIR/scripts/fetch_sec_edgar.py --cik 0000320193 \
    --types 10-K,10-Q --out data/edgar_filings.csv

# USAspending federal contracts
python3 SKILL_DIR/scripts/fetch_usaspending.py --recipient "EXAMPLE CORP" \
    --fy 2024 --out data/contracts.csv

# Senate LD-1 / LD-2 lobbying disclosures
python3 SKILL_DIR/scripts/fetch_senate_ld.py --client "EXAMPLE CORP" \
    --year 2024 --out data/lobbying.csv

# OFAC SDN sanctions list (full snapshot)
python3 SKILL_DIR/scripts/fetch_ofac_sdn.py --out data/ofac_sdn.csv

# ICIJ Offshore Leaks — downloads ~70 MB bulk CSV on first use,
# then searches it locally. Cached for 30 days under
# $HERMES_OSINT_CACHE/icij/ (default: ~/.cache/hermes-osint/icij/).
python3 SKILL_DIR/scripts/fetch_icij_offshore.py --entity "EXAMPLE CORP" \
    --out data/icij.csv
```

**Identity / property / litigation / archives / news**

```bash
# NYC property records (deeds, mortgages, liens) — ACRIS via Socrata
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --name "SMITH, JOHN" \
    --out data/acris.csv
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --address "571 HUDSON" \
    --out data/acris_addr.csv

# OpenCorporates — 130+ jurisdiction corporate registry
# (free token required; set OPENCORPORATES_API_TOKEN or pass --token)
python3 SKILL_DIR/scripts/fetch_opencorporates.py --query "Example Corp" \
    --jurisdiction us_ny --out data/opencorporates.csv

# CourtListener — federal + state court opinions, PACER dockets
python3 SKILL_DIR/scripts/fetch_courtlistener.py --query "Smith v. Example Corp" \
    --type opinions --out data/courts.csv

# Wayback Machine — historical web captures
python3 SKILL_DIR/scripts/fetch_wayback.py --url "example.com" \
    --match host --collapse digest --out data/wayback.csv

# Wikipedia + Wikidata — narrative bio + structured facts
# Set HERMES_OSINT_UA=your-app/1.0 (your@email) to identify yourself
python3 SKILL_DIR/scripts/fetch_wikipedia.py --query "Bill Gates" \
    --out data/wp.csv

# GDELT — global news in 100+ languages, ~2015→present
python3 SKILL_DIR/scripts/fetch_gdelt.py --query '"Example Corp"' \
    --timespan 1y --out data/gdelt.csv
```

모든 출력은 헤더 행이 있는 정규화된 CSV입니다. 스크립트를 다시 실행해도
멱등적으로 동작합니다.

개인이 출처에 포함되지 않는 경우(예: 비상장 회사 관계자라서 SEC EDGAR에
없거나, 연방 계약자가 아니라서 USAspending에 없거나, 로비 고객이 아니라서
Senate LDA에 없는 경우) 스크립트는 아무 경고 없이 빈 CSV를 작성하는 대신
명확한 경고와 함께 0행을 반환합니다. 특히 EDGAR는 회사명 확인 결과 기업
등록자가 아니라 개인 Form 3/4/5 신고자와 일치했는지도 표시합니다.

요청 속도 제한 관련 내용은 각 출처의 위키 항목에 있습니다. 기본 가져오기
도구는 페이지가 나뉜 요청 사이에 예의 바르게 대기합니다. 이를 지원하는
출처에서는 **API 키가 요청 한도를 높입니다**(`SEC_USER_AGENT`,
`SENATE_LDA_TOKEN`, `OPENCORPORATES_API_TOKEN`, `COURTLISTENER_TOKEN`). 모든
스크립트는 429 응답이 발생하면 업스트림의 할당량 메시지와 함께 즉시
표시하므로 사용자는 속도를 늦추거나 키를 제공해야 한다는 것을 알 수 있습니다.

### 3. 출처 간 엔터티 식별

두 CSV 파일 사이에서 이름을 정규화하고 일치 항목을 찾습니다.

```bash
# Match lobbying clients (Senate LDA) against contract recipients (USAspending)
python3 SKILL_DIR/scripts/entity_resolution.py \
    --left  data/lobbying.csv   --left-name-col  client_name \
    --right data/contracts.csv  --right-name-col recipient_name \
    --out data/cross_links.csv
```

명시적인 신뢰도를 포함한 세 가지 일치 등급입니다.

| 등급 | 방법 | 신뢰도 |
|------|------|--------|
| `exact` | 접미사/구두점 제거 후 정규화된 문자열이 동일 | 높음 |
| `fuzzy` | 정렬된 토큰이 동일(단어 묶음 일치) | 중간 |
| `token_overlap` | 토큰 중복률 ≥60%, 공유 토큰 ≥2개, 토큰 길이 ≥4자 | 낮음 |

`cross_links.csv` 열: `match_type, confidence, left_name,
right_name, left_normalized, right_normalized, left_row, right_row`.

### 4. 통계적 시점 상관(선택 사항)

두 시계열이 의심스러울 정도로 가까운 시점에 군집을 이루는지, 예를 들어
로비 신고와 계약 수주가 가까운 시점에 발생하는지를 순열 검정으로 확인합니다.

```bash
python3 SKILL_DIR/scripts/timing_analysis.py \
    --donations data/lobbying.csv --donation-date-col filing_date \
        --donation-amount-col income --donation-donor-col client_name \
        --donation-recipient-col registrant_name \
    --contracts data/contracts.csv --contract-date-col award_date \
        --contract-vendor-col recipient_name \
    --cross-links data/cross_links.csv \
    --permutations 1000 \
    --out data/timing.json
```

스크립트의 열 플래그는 의도적으로 일반화되어 있습니다. 원래 도구는 기부금과
수주를 대상으로 작성되었지만, `cross_links`를 통해 결합된 어떤
(사건, 수령인) 시계열에도 사용할 수 있습니다. 귀무가설은 사건 시점이 수주
날짜와 독립적이라는 것입니다. 단측 p값 = 평균 최근 수주까지의 거리가
관측값 이하인 순열의 비율입니다. 검정을 실행하려면 (지급자, 공급업체) 쌍마다
최소 3개의 사건이 필요합니다.

### 5. findings JSON 작성(증거 사슬)

```bash
python3 SKILL_DIR/scripts/build_findings.py \
    --cross-links data/cross_links.csv \
    --timing data/timing.json \
    --out data/findings.json
```

모든 finding에는 `id, title, severity, confidence, summary, evidence[], sources[]`가
있습니다. 각 증거 항목은 출처 CSV의 특정 행을 가리킵니다. 사용자(또는 후속
에이전트)는 출처를 기준으로 모든 주장을 검증할 수 있습니다.

## 신뢰도와 증거 규율

이 스킬의 핵심 규칙입니다. 사용자에게 다음을 알리세요.

- 모든 주장은 기록으로 추적할 수 있어야 합니다. 근거 없는 주장을 하지 마세요.
- 신뢰도 등급은 주장과 함께 전달됩니다. `match_type=fuzzy`는 “확인됨”이
  아니라 “가능성이 높음”입니다.
- 엔터티 식별은 결론이 아니라 후보를 생성합니다. “ACME LLC”와 “Acme
  Holdings Group”의 `fuzzy` 일치는 단서이지 사실이 아닙니다.
- 통계적 유의성 ≠ 위법 행위입니다. p &lt; 0.05는 시점 패턴이 귀무가설 아래에서
  나타날 가능성이 낮다는 뜻일 뿐, 부패를 입증하지는 않습니다.
- 여기의 모든 데이터 출처는 공공 기록입니다. 그래도 부정확하거나 오래된 정보,
  또는 삭제된 정보(GDPR, 봉인 기록)가 포함될 수 있습니다.

## 새 데이터 출처 추가

다음 템플릿을 사용합니다.

```bash
cp SKILL_DIR/templates/source-template.md \
    SKILL_DIR/references/sources/<your-source>.md
```

9개 섹션을 모두 작성합니다. 표준 라이브러리만 사용하고 정규화된 CSV를
작성하는 `scripts/`의 `fetch_<source>.py` 스크립트를 작성합니다. 위의
“이 스킬을 사용할 때” 섹션에 출처 목록을 업데이트합니다.

## 도구와 한계

- `entity_resolution.py`는 외부 퍼지 라이브러리를 사용하지 않습니다(rapidfuzz도,
  jellyfish도 사용하지 않음). 토큰 묶음 일치가 여기서 가능한 상한입니다.
  Levenshtein 거리, 음역 또는 음성학적 일치가 필요하면 별도로 pip 설치하세요.
- `timing_analysis.py`는 순열에 Python의 `random`을 사용합니다. 재현성을
  확보하려면 `--seed N`을 전달하세요.
- `fetch_*.py` 스크립트는 `urllib.request`를 사용하고 `Retry-After`를
  준수합니다. 대량 사용은 여전히 서비스 약관을 위반할 수 있으므로 먼저 각
  출처의 법률 섹션을 읽으세요.

## 법률 고지

1단계의 모든 출처는 공공 기록입니다. 각 출처의 접근 조건(FOIA, 공공 기록법,
ICIJ의 명시적 공개, OFAC 공개 데이터)에 따라 대량 수집이 허용됩니다. 그러나:

- 일부 출처는 요청 속도를 엄격하게 제한합니다. 헤더를 준수하세요.
- 일부 출처는 등록자 정보를 삭제합니다(WHOIS의 GDPR, 봉인된 신고서).
- 공공 기록을 교차 참조하여 개인을 식별하는 일에는 윤리적 함의가 있을 수
  있습니다. 이 스킬은 고발이 아니라 증거 사슬을 생성합니다.
