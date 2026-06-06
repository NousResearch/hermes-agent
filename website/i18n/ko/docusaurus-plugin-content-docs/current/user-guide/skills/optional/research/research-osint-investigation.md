---
title: "Osint Investigation"
sidebar_label: "Osint Investigation"
description: "공개 기록 OSINT 조사 프레임워크 — SEC EDGAR 문서, USAspending 정부 계약, 미국 상원 로비, OFAC 제재, ICIJ 조세회피처 유출 문건, 뉴욕시 부동산 기록..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Osint Investigation

공개 기록 OSINT 조사 프레임워크 — SEC EDGAR 문서, USAspending 정부 계약, 미국 상원 로비, OFAC 제재, ICIJ 조세회피처 유출 문건, 뉴욕시 부동산 기록 (ACRIS), OpenCorporates 법인 등기부, CourtListener 법원 기록, Wayback Machine 아카이브, Wikipedia + Wikidata, GDELT 뉴스 모니터링. 다양한 소스에 걸친 엔티티(Entity) 식별(resolution), 상호 교차 링크(cross-link) 분석, 타이밍 상관관계(timing correlation), 증거 사슬(evidence chains). Python 표준 라이브러리만 사용합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Optional — `hermes skills install official/research/osint-investigation` 명령으로 설치 |
| Path | `optional-skills/research/osint-investigation` |
| Version | `0.1.0` |
| Author | Hermes Agent (ShinMegamiBoson/OpenPlanter, MIT 프로젝트 기반으로 개조됨) |
| Platforms | linux, macos, windows |
| Tags | `osint`, `investigation`, `public-records`, `sec`, `sanctions`, `corporate-registry`, `property`, `courts`, `due-diligence`, `journalism` |
| Related skills | [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# OSINT Investigation — Public Records Cross-Reference

공개 기록 OSINT를 위한 조사 프레임워크: 정부 계약, 기업 공시, 로비, 제재, 조세회피처 유출 문건, 부동산 기록, 법원 기록, 웹 아카이브, 지식 베이스(knowledge bases) 및 글로벌 뉴스. 다양한 출처의 엔티티를 통합 식별하고 명시적인 신뢰도를 갖춘 교차 링크를 생성하며, 통계적 타이밍 테스트를 실행하고 구조화된 증거 사슬을 생성합니다.

**오직 Python 표준 라이브러만 사용합니다.** 설치가 필요 없습니다. Linux, macOS, Windows에서 작동합니다. 대부분의 데이터 소스는 API 키 없이 작동합니다 (OpenCorporates의 경우 선택적인 무료 토큰을 통해 속도 제한(rate limits)을 늘릴 수 있습니다).

MIT 라이선스인 ShinMegamiBoson/OpenPlanter 프로젝트를 개조하였으며; 원본 프로젝트에서 다루지 않았던 신원 / 부동산 / 소송 / 아카이브 / 뉴스 소스를 포함하도록 확장되었습니다.

## 사용 시기 (When to use this skill)

사용자가 다음과 같은 내용을 요청할 때 사용하세요:

- "자금 추적(follow the money)" — 정부 계약, 로비 → 입법, 제재
- 기업 실사(due diligence) — 회사 X의 실소유주는 누구인가, 어디에 설립되었나, 이사회는 누구인가, 어떤 공시를 제출했나
- 제재 스크리닝(sanctions screening) — 엔티티 X가 OFAC SDN, ICIJ 조세회피처 명단에 있는가
- 이해관계 개입(pay-to-play) 조사 — 조세회피처 연관 계약업체, 로비 고객의 정부 지원금 수상 여부
- 부동산 소유권 — 이름이나 주소로 등기(deeds)/모기지 찾기 (뉴욕시 전용; 다른 카운티의 경우 사용자에게 해당 등기소를 안내할 것)
- 소송 기록 — 연방 및 주 법원 판결문(opinions) 및 PACER 법원 기록부(dockets) 찾기
- 다양한 명칭(LLC 접미사, 약어 등)이 혼용되는 다중 소스 간의 엔티티 식별(entity resolution)
- 명시적인 신뢰 수준을 갖춘 증거 사슬 구성
- "X에 대해 어떤 말이 오갔는가" — 글로벌 뉴스(GDELT) + 위키백과 내러티브 + 연결이 끊긴(dead) URL 복구를 위한 Wayback Machine

이 스킬을 다음과 같은 목적에 사용하지 **마십시오**:

- 일반적인 웹 검색 → `web_search` / `web_extract` 사용
- 도메인/인프라 OSINT → `domain-intel` 스킬 사용
- 학술 문헌 → `arxiv` 스킬 사용
- 소셜 미디어 프로필 검색 → `sherlock` 스킬 사용 (옵션)
- 미국 **연방** 선거 자금(campaign finance) — FEC는 의도적으로 여기에 포함되지 않았습니다 (무료 DEMO_KEY 티어에서 일회성 기부자 이름 검색을 위한 API가 불안정함). 연방 기부금의 경우 사용자를 직접 https://www.fec.gov/data/ 로 안내하세요.

## 워크플로우 (Workflow)

에이전트는 `terminal` 도구를 통해 스크립트를 실행합니다. `SKILL_DIR`은 이 SKILL.md를 포함하는 디렉토리입니다.

### 1. 적용할 소스 파악하기

어떤 소스가 해당하는지 데이터 소스 위키 항목을 읽고 조사를 계획하세요:

```
ls SKILL_DIR/references/sources/

# 연방 금융 / 규제
cat SKILL_DIR/references/sources/sec-edgar.md       # 기업 공시
cat SKILL_DIR/references/sources/usaspending.md     # 연방 계약
cat SKILL_DIR/references/sources/senate-ld.md       # 로비
cat SKILL_DIR/references/sources/ofac-sdn.md        # 제재 목록
cat SKILL_DIR/references/sources/icij-offshore.md   # 조세회피처 문건

# 신원 / 부동산 / 소송 / 아카이브 / 뉴스
cat SKILL_DIR/references/sources/nyc-acris.md       # 뉴욕시 부동산 기록
cat SKILL_DIR/references/sources/opencorporates.md  # 글로벌 기업 레지스트리
cat SKILL_DIR/references/sources/courtlistener.md   # 법원 기록 (연방 + 주)
cat SKILL_DIR/references/sources/wayback.md         # Wayback Machine 아카이브
cat SKILL_DIR/references/sources/wikipedia.md       # Wikipedia + Wikidata
cat SKILL_DIR/references/sources/gdelt.md           # 글로벌 뉴스 모니터링
```

각 항목은 9개 섹션(요약, 액세스, 스키마, 커버리지, 상호 참조 키, 데이터 품질, 획득 방법, 법적 정보, 참조) 템플릿을 따릅니다.

**상호 참조 가능성(cross-reference potential)** 섹션은 소스 간의 조인(join) 키를 매핑합니다 — 올바른 쌍을 선택하기 위해 먼저 이 내용을 읽어보세요.

### 2. 데이터 획득 (Acquire data)

각 소스는 `SKILL_DIR/scripts/`에 파이썬 표준 라이브러리만을 사용하는 수집 스크립트가 있습니다:

**연방 금융 / 규제 (Federal financial / regulatory)**

```bash
# SEC EDGAR filings (기업 공시)
python3 SKILL_DIR/scripts/fetch_sec_edgar.py --cik 0000320193 \
    --types 10-K,10-Q --out data/edgar_filings.csv

# USAspending 연방 계약
python3 SKILL_DIR/scripts/fetch_usaspending.py --recipient "EXAMPLE CORP" \
    --fy 2024 --out data/contracts.csv

# 미국 상원 LD-1 / LD-2 로비 공시
python3 SKILL_DIR/scripts/fetch_senate_ld.py --client "EXAMPLE CORP" \
    --year 2024 --out data/lobbying.csv

# OFAC SDN 제재 리스트 (전체 스냅샷)
python3 SKILL_DIR/scripts/fetch_ofac_sdn.py --out data/ofac_sdn.csv

# ICIJ Offshore Leaks — 처음 사용할 때 약 70MB의 벌크 CSV를 다운로드하고,
# 이후 로컬에서 검색합니다. 다음 위치에 30일 동안 캐시됩니다.
# $HERMES_OSINT_CACHE/icij/ (기본값: ~/.cache/hermes-osint/icij/).
python3 SKILL_DIR/scripts/fetch_icij_offshore.py --entity "EXAMPLE CORP" \
    --out data/icij.csv
```

**신원 / 부동산 / 소송 / 아카이브 / 뉴스 (Identity / property / litigation / archives / news)**

```bash
# 뉴욕시 부동산 기록 (deeds, mortgages, liens) — ACRIS via Socrata
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --name "SMITH, JOHN" \
    --out data/acris.csv
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --address "571 HUDSON" \
    --out data/acris_addr.csv

# OpenCorporates — 130개 이상 관할 구역의 기업 레지스트리
# (무료 토큰 필요; OPENCORPORATES_API_TOKEN 설정 또는 --token 전달)
python3 SKILL_DIR/scripts/fetch_opencorporates.py --query "Example Corp" \
    --jurisdiction us_ny --out data/opencorporates.csv

# CourtListener — 연방 + 주 법원 판결문, PACER dockets
python3 SKILL_DIR/scripts/fetch_courtlistener.py --query "Smith v. Example Corp" \
    --type opinions --out data/courts.csv

# Wayback Machine — 웹 기록 아카이브
python3 SKILL_DIR/scripts/fetch_wayback.py --url "example.com" \
    --match host --collapse digest --out data/wayback.csv

# Wikipedia + Wikidata — 서술형(narrative) 약력 + 구조화된 팩트 데이터
# 자신을 식별하기 위해 HERMES_OSINT_UA=your-app/1.0 (your@email)를 설정하세요
python3 SKILL_DIR/scripts/fetch_wikipedia.py --query "Bill Gates" \
    --out data/wp.csv

# GDELT — 100개 이상의 언어로 된 글로벌 뉴스, ~2015→현재
python3 SKILL_DIR/scripts/fetch_gdelt.py --query '"Example Corp"' \
    --timespan 1y --out data/gdelt.csv
```

모든 출력 결과는 헤더 행(header row)이 포함된 정규화된(normalized) CSV입니다. 스크립트 실행은 멱등성(idempotent)을 가집니다.

개인이 해당 소스에 없는 경우(예: 비상장 회사 소속인 경우 SEC EDGAR, 정부 계약자가 아닌 경우 USAspending, 로비 고객이 아닌 경우 상원 LDA), 스크립트는 조용히 빈 CSV를 작성하는 대신 명확한 경고와 함께 0개 행을 반환합니다. EDGAR는 회사 이름 리졸버(resolver)가 법인 등록자가 아닌 개인의 Form 3/4/5 제출자와 일치할 때 특별히 그 사실을 알립니다.

속도 제한(Rate-limit)에 대한 참고사항은 각 데이터 소스의 위키 항목에 있습니다. 기본 페처(fetcher)들은 페이지 간 요청 사이에 정중하게 슬립(sleep) 대기를 수행합니다. **API 키는 API 속도 제한을 높여줍니다** (`SEC_USER_AGENT`, `SENATE_LDA_TOKEN`, `OPENCORPORATES_API_TOKEN`, `COURTLISTENER_TOKEN`). 모든 스크립트는 429 응답 발생 시 업스트림 측의 할당량 한도 메시지와 함께 즉시 표출하므로, 사용자는 속도를 늦추거나 키를 입력해야 한다는 것을 알 수 있습니다.

### 3. 소스 간 엔티티 식별(resolution)

두 CSV 파일 간의 이름을 정규화하고 매칭을 찾습니다:

```bash
# 로비 고객(Senate LDA)과 계약 수주자(USAspending) 일치 확인
python3 SKILL_DIR/scripts/entity_resolution.py \
    --left  data/lobbying.csv   --left-name-col  client_name \
    --right data/contracts.csv  --right-name-col recipient_name \
    --out data/cross_links.csv
```

명시적 신뢰도에 기반한 3개의 매칭 티어(tiers):

| Tier | Method | Confidence |
|------|--------|------------|
| `exact` | 접미사/구두점 제거 후 정규화된 문자열이 일치함 | 높음(high) |
| `fuzzy` | 정렬된 토큰 일치(sorted-token equality, word-bag match) | 중간(medium) |
| `token_overlap` | 60% 이상 토큰 겹침, 공유 토큰 2개 이상, 토큰 길이 4자 이상 | 낮음(low) |

출력 `cross_links.csv` 컬럼 정보: `match_type, confidence, left_name, right_name, left_normalized, right_normalized, left_row, right_row`.

### 4. 통계적 타이밍 상관관계(Statistical timing correlation) (선택 사항)

두 시계열이 의심스러울 정도로 서로 밀접하게 몰려 있는지 (예: 계약 체결일 부근에 몰린 로비 지출) 치환 검정(permutation test)을 사용하여 테스트합니다:

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

스크립트의 컬럼 플래그는 의도적으로 일반적(generic)입니다 — 기존 도구는 기부금 대 계약 시상용으로 작성되었으나, 크로스 링크를 통해 결합된 모든 (이벤트, 수취인) 시계열 데이터에서 작동합니다. 귀무 가설: 이벤트 타이밍은 계약 체결 날짜와 독립적입니다. 일측 검정 p-value = 평균 최단 계약 체결 거리가 관찰된 값보다 작거나 같은 치환 세트의 비율입니다. 검정을 실행하려면 (지불자, 수취자) 쌍마다 최소 3개의 이벤트가 필요합니다.

### 5. 조사 결과(Findings) JSON 빌드 (증거 사슬)

```bash
python3 SKILL_DIR/scripts/build_findings.py \
    --cross-links data/cross_links.csv \
    --timing data/timing.json \
    --out data/findings.json
```

모든 조사 결과는 `id, title, severity, confidence, summary, evidence[], sources[]`를 갖습니다. 각 증거 항목은 원본 소스 CSV의 특정 행을 가리킵니다. 사용자는 (또는 후속 에이전트는) 모든 주장을 원본 소스에 대조하여 검증할 수 있습니다.

## 신뢰도 및 증거 원칙 (Confidence and evidence discipline)

이것은 이 스킬의 핵심적인 하중 지지(load-bearing) 규칙입니다. 사용자에게 다음을 알려주세요:

- 모든 주장은 반드시 기록으로 추적 가능해야 합니다. 아무 근거 없는 주장은 금물입니다.
- 신뢰도 등급(Confidence tier)은 해당 주장과 항상 함께 제공되어야 합니다. `match_type=fuzzy`는 "개연성 있는(probable)" 매칭이지, 확정적인 "확인(confirmed)"이 아닙니다.
- 엔티티 식별(Entity resolution) 과정은 후보군을 생성할 뿐, 결론을 내리지 않습니다. "ACME LLC"와 "Acme Holdings Group" 사이의 `fuzzy` 매칭은 단서(lead)일 뿐 사실(fact)이 아닙니다.
- 통계적 유의성이 곧 부정행위를 의미하지는 않습니다. p &lt; 0.05는 해당 타이밍 패턴이 귀무 가설 하에서 드문 현상이라는 뜻입니다. 부패를 입증하는 것이 아닙니다.
- 여기 있는 모든 데이터 소스는 공개된 공공 기록입니다. 하지만 여전히 부정확한 정보, 오래된 정보 또는 편집(GDPR, 비공개 기록 등)된 내용이 포함되어 있을 수 있습니다.

## 새로운 데이터 소스 추가 (Adding a new data source)

템플릿을 사용하세요:

```bash
cp SKILL_DIR/templates/source-template.md \
    SKILL_DIR/references/sources/<your-source>.md
```

9개 섹션을 모두 채우세요. 표준 라이브러리만 사용하고 정규화된 CSV를 작성하는 `fetch_<source>.py` 스크립트를 `scripts/` 디렉토리에 작성하세요. 위의 "사용 시기" 섹션의 소스 목록을 업데이트하세요.

## 도구 및 한계 (Tools and their limits)

- `entity_resolution.py`는 외부의 퍼지(fuzzy) 라이브러리(rapidfuzz, jellyfish 등)를 사용하지 않습니다. 토큰-백(Token-bag) 매칭이 여기서의 상한선(upper bound)입니다. Levenshtein, 음역(transliteration) 또는 음성학적(phonetic) 매칭이 필요하다면 별도로 `pip-install`하세요.
- `timing_analysis.py`는 치환(permutation)을 위해 파이썬의 `random` 라이브러리를 사용합니다. 재현 가능성(reproducibility)이 필요하다면 `--seed N` 인자를 전달하세요.
- `fetch_*.py` 스크립트는 `urllib.request`를 사용하며 `Retry-After` 응답 헤더를 준수합니다. 과도한 대량 사용은 여전히 ToS(서비스 약관)를 위반할 수 있습니다 — 각 데이터 소스의 '법적 정보' 섹션을 먼저 읽으세요.

## 법적 고지 (Legal note)

모든 1단계 소스는 공개 기록입니다. 이들의 대량 수집은 각각의 접근 약관(정보공개법, FOIA, 공공기록법, ICIJ 명시적 공개 약관, OFAC 퍼블릭 데이터 정책 등) 하에 허용됩니다. 단:

- 일부 데이터 소스들은 매우 엄격하게 속도 제한을 가합니다. 그들의 HTTP 헤더를 존중하세요.
- 일부는 등록자 정보를 편집(redact)합니다 (WHOIS의 GDPR 대응, 비공개된 판결문 등).
- 사적인 개인의 신원을 확인하기 위해 공공 기록을 상호 참조하는 것은 윤리적 파장을 초래할 수 있습니다. 이 스킬은 증거 사슬(evidence chains)을 생성하는 것이며, 누군가를 고발(accusation)하는 것이 아닙니다.
