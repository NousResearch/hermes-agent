---
title: "Parallel Cli"
sidebar_label: "Parallel Cli"
description: "Parallel CLI를 위한 선택적 벤더 스킬 — 에이전트 네이티브 웹 검색, 추출, 심층 연구, 데이터 보강, FindAll 및 모니터링"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Parallel Cli

Parallel CLI를 위한 선택적 벤더 스킬 — 에이전트 네이티브 웹 검색, 추출, 심층 연구(deep research), 데이터 보강(enrichment), FindAll 및 모니터링. JSON 출력 및 비대화형 흐름을 선호합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Optional — `hermes skills install official/research/parallel-cli` 명령으로 설치 |
| Path | `optional-skills/research/parallel-cli` |
| Version | `1.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Research`, `Web`, `Search`, `Deep-Research`, `Enrichment`, `CLI` |
| Related skills | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`mcporter`](/docs/user-guide/skills/optional/mcp/mcp-mcporter) |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Parallel CLI

사용자가 명시적으로 Parallel을 원하거나, 터미널 네이티브 워크플로우에서 웹 검색, 추출, 심층 연구, 데이터 보강, 엔티티 발견, 또는 모니터링을 위한 Parallel의 벤더별 스택이 유리할 때 `parallel-cli`를 사용하세요.

이것은 Hermes 코어 기능이 아닌, 선택적인 타사 워크플로우입니다.

중요한 기대사항:
- Parallel은 완전 무료 로컬 도구가 아니며, 무료 티어가 있는 유료 서비스입니다.
- Hermes의 기본 `web_search` / `web_extract`와 기능이 겹치므로, 일반적인 검색을 위해 기본값으로 이것을 선호하지 마세요.
- 사용자가 구체적으로 Parallel을 언급하거나 Parallel의 enrichment, FindAll 또는 monitor 워크플로우 같은 기능이 필요할 때 이 스킬을 우선시하세요.

`parallel-cli`는 에이전트를 위해 설계되었습니다:
- `--json`을 통한 JSON 출력
- 비대화형(Non-interactive) 명령 실행
- `--no-wait`, `status`, `poll`을 이용한 비동기 장기 실행 작업(long-running jobs)
- `--previous-interaction-id`를 이용한 컨텍스트 체이닝
- 검색, 추출, 연구, 데이터 보강, 엔티티 발견 및 모니터링을 하나의 CLI에서 지원

## 사용 시기 (When to use it)

다음과 같은 경우 이 스킬을 우선적으로 사용하세요:
- 사용자가 Parallel 또는 `parallel-cli`를 명시적으로 언급할 때
- 단순한 일회성 검색/추출 단계보다 더 풍부한 워크플로우가 필요할 때
- 비동기식 심층 연구 작업을 시작하고 나중에 폴링(poll)해야 할 때
- 구조화된 데이터 보강, FindAll 엔티티 발견 또는 모니터링이 필요할 때

Parallel을 특별히 요청하지 않은 경우, 빠른 일회성 검색에는 Hermes 기본 `web_search` / `web_extract`를 우선 사용하세요.

## 설치 (Installation)

해당 환경에 가장 덜 침투적인(least invasive) 설치 경로를 시도하세요.

### Homebrew

```bash
brew install parallel-web/tap/parallel-cli
```

### npm

```bash
npm install -g parallel-web-cli
```

### Python package

```bash
pip install "parallel-web-tools[cli]"
```

### 독립 실행형 설치 프로그램 (Standalone installer)

```bash
curl -fsSL https://parallel.ai/install.sh | bash
```

격리된 Python 설치를 원할 경우, `pipx`도 작동할 수 있습니다:

```bash
pipx install "parallel-web-tools[cli]"
pipx ensurepath
```

## 인증 (Authentication)

대화형 로그인:

```bash
parallel-cli login
```

헤드리스(Headless) / SSH / CI:

```bash
parallel-cli login --device
```

API 키 환경 변수:

```bash
export PARALLEL_API_KEY="***"
```

현재 인증 상태 확인:

```bash
parallel-cli auth
```

인증에 브라우저 상호작용이 필요한 경우, `pty=true`와 함께 실행하세요.

## 핵심 규칙 (Core rule set)

1. 기계가 읽을 수 있는 출력이 필요할 때는 항상 `--json`을 선호하세요.
2. 명시적인 인수(arguments)와 비대화형(non-interactive) 흐름을 선호하세요.
3. 오래 걸리는 작업의 경우 `--no-wait`를 사용하고 이후에 `status` / `poll`을 사용하세요.
4. CLI 출력에서 반환된 URL만 인용하세요.
5. 후속 질문이 예상될 때는 대용량 JSON 출력을 임시 파일로 저장하세요.
6. 진짜로 오래 걸리는 워크플로우에만 백그라운드 프로세스를 사용하세요; 그렇지 않으면 포그라운드에서 실행하세요.
7. 사용자가 특별히 Parallel을 원하거나 Parallel 전용 워크플로우가 필요하지 않은 한 Hermes 기본 도구를 선호하세요.

## 빠른 참조 (Quick reference)

<!-- ascii-guard-ignore -->
```text
parallel-cli
├── auth
├── login
├── logout
├── search
├── extract / fetch
├── research run|status|poll|processors
├── enrich run|status|poll|plan|suggest|deploy
├── findall run|ingest|status|poll|result|enrich|extend|schema|cancel
└── monitor create|list|get|update|delete|events|event-group|simulate
```
<!-- ascii-guard-ignore-end -->

## 공통 플래그 및 패턴 (Common flags and patterns)

자주 유용한 플래그:
- `--json`: 구조화된 출력
- `--no-wait`: 비동기 작업
- `--previous-interaction-id <id>`: 이전 컨텍스트를 재사용하는 후속 작업용
- `--max-results <n>`: 검색 결과 수 지정
- `--mode one-shot|agentic`: 검색 동작 모드
- `--include-domains domain1.com,domain2.com`
- `--exclude-domains domain1.com,domain2.com`
- `--after-date YYYY-MM-DD`

편리할 때는 표준 입력(stdin)에서 읽기:

```bash
echo "What is the latest funding for Anthropic?" | parallel-cli search - --json
echo "Research question" | parallel-cli research run - --json
```

## 검색 (Search)

구조화된 결과를 포함하는 최신 웹 검색에 사용합니다.

```bash
parallel-cli search "What is Anthropic's latest AI model?" --json
parallel-cli search "SEC filings for Apple" --include-domains sec.gov --json
parallel-cli search "bitcoin price" --after-date 2026-01-01 --max-results 10 --json
parallel-cli search "latest browser benchmarks" --mode one-shot --json
parallel-cli search "AI coding agent enterprise reviews" --mode agentic --json
```

유용한 제약 조건:
- `--include-domains`: 신뢰할 수 있는 소스로 좁히기
- `--exclude-domains`: 노이즈가 많은 도메인 제거
- `--after-date`: 최신 정보 필터링
- `--max-results`: 더 넓은 커버리지가 필요할 때

후속 질문이 예상되면 출력을 저장하세요:

```bash
parallel-cli search "latest React 19 changes" --json -o /tmp/react-19-search.json
```

결과 요약 시:
- 답변을 먼저 제시하세요
- 날짜, 이름 및 구체적인 사실을 포함하세요
- 반환된 소스만 인용하세요
- URL이나 소스 제목을 꾸며내지(invent) 마세요

## 추출 (Extraction)

URL에서 깨끗한 콘텐츠나 마크다운을 가져올 때 사용합니다.

```bash
parallel-cli extract https://example.com --json
parallel-cli extract https://company.com --objective "Find pricing info" --json
parallel-cli extract https://example.com --full-content --json
parallel-cli fetch https://example.com --json
```

페이지가 방대하고 정보의 특정 부분만 필요할 때 `--objective`를 사용하세요.

## 심층 연구 (Deep research)

시간이 걸릴 수 있는 깊이 있는 다단계 연구 작업에 사용합니다.

일반적인 프로세서 티어(tiers):
- `lite` / `base`: 더 빠르고 저렴한 단계
- `core` / `pro`: 더 철저한 합성(synthesis)
- `ultra`: 가장 무거운 연구 작업

### 동기식 (Synchronous)

```bash
parallel-cli research run \
  "Compare the leading AI coding agents by pricing, model support, and enterprise controls" \
  --processor core \
  --json
```

### 비동기 실행 + 폴링 (Async launch + poll)

```bash
parallel-cli research run \
  "Compare the leading AI coding agents by pricing, model support, and enterprise controls" \
  --processor ultra \
  --no-wait \
  --json

parallel-cli research status trun_xxx --json
parallel-cli research poll trun_xxx --json
parallel-cli research processors --json
```

### 컨텍스트 체이닝 / 후속 작업 (Context chaining / follow-up)

```bash
parallel-cli research run "What are the top AI coding agents?" --json
parallel-cli research run \
  "What enterprise controls does the top-ranked one offer?" \
  --previous-interaction-id trun_xxx \
  --json
```

권장되는 Hermes 워크플로우:
1. `--no-wait --json`과 함께 실행(launch)합니다
2. 반환된 run/task ID를 캡처합니다
3. 사용자가 다른 작업을 계속 원하면 그대로 진행합니다
4. 나중에 `status` 또는 `poll`을 호출합니다
5. 반환된 소스의 인용과 함께 최종 보고서를 요약합니다

## 데이터 보강 (Enrichment)

사용자가 CSV/JSON/표 형식의 입력을 가지고 있고 웹 연구에서 추론된 추가 열(columns)을 원할 때 사용합니다.

### 열 제안 (Suggest columns)

```bash
parallel-cli enrich suggest "Find the CEO and annual revenue" --json
```

### 구성 계획 (Plan a config)

```bash
parallel-cli enrich plan -o config.yaml
```

### 인라인 데이터 (Inline data)

```bash
parallel-cli enrich run \
  --data '[{"company": "Anthropic"}, {"company": "Mistral"}]' \
  --intent "Find headquarters and employee count" \
  --json
```

### 비대화형 파일 실행 (Non-interactive file run)

```bash
parallel-cli enrich run \
  --source-type csv \
  --source companies.csv \
  --target enriched.csv \
  --source-columns '[{"name": "company", "description": "Company name"}]' \
  --intent "Find the CEO and annual revenue"
```

### YAML 구성 실행 (YAML config run)

```bash
parallel-cli enrich run config.yaml
```

### 상태 / 폴링 (Status / polling)

```bash
parallel-cli enrich status <task_group_id> --json
parallel-cli enrich poll <task_group_id> --json
```

비대화형으로 작업할 때는 열 정의에 명시적인 JSON 배열을 사용하세요.
성공을 보고하기 전에 출력 파일을 검증하세요.

## FindAll

사용자가 짧은 답변보다는 발견된 데이터셋을 원할 때 웹 규모의 엔티티 발견(entity discovery)을 위해 사용합니다.

```bash
parallel-cli findall run "Find AI coding agent startups with enterprise offerings" --json
parallel-cli findall run "AI startups in healthcare" -n 25 --json
parallel-cli findall status <run_id> --json
parallel-cli findall poll <run_id> --json
parallel-cli findall result <run_id> --json
parallel-cli findall schema <run_id> --json
```

이것은 사용자가 나중에 검토, 필터링 또는 데이터 보강을 할 수 있는 발견된 엔티티 세트를 원할 때 일반 검색보다 더 적합합니다.

## 모니터 (Monitor)

시간 경과에 따른 지속적인 변화 감지를 위해 사용합니다.

```bash
parallel-cli monitor list --json
parallel-cli monitor get <monitor_id> --json
parallel-cli monitor events <monitor_id> --json
parallel-cli monitor delete <monitor_id> --json
```

리듬(cadence)과 전달(delivery)이 중요하기 때문에 생성이 가장 민감한 부분입니다:

```bash
parallel-cli monitor create --help
```

사용자가 일회성 가져오기가 아니라 페이지나 소스의 반복적인 추적을 원할 때 이것을 사용하세요.

## 권장되는 Hermes 사용 패턴 (Recommended Hermes usage patterns)

### 인용과 함께 빠른 답변 (Fast answer with citations)
1. `parallel-cli search ... --json` 실행
2. 제목, URL, 날짜, 발췌문 파싱
3. 반환된 URL에서만 나온 인라인 인용과 함께 요약

### URL 조사 (URL investigation)
1. `parallel-cli extract URL --json` 실행
2. 필요한 경우 `--objective` 또는 `--full-content`와 함께 재실행
3. 추출된 마크다운을 인용하거나 요약

### 긴 연구 워크플로우 (Long research workflow)
1. `parallel-cli research run ... --no-wait --json` 실행
2. 반환된 ID 저장
3. 다른 작업을 계속하거나 주기적으로 폴링(poll)
4. 인용과 함께 최종 보고서 요약

### 구조화된 보강 워크플로우 (Structured enrichment workflow)
1. 입력 파일과 열 검사
2. `enrich suggest`를 사용하거나 명시적인 보강(enriched) 열 제공
3. `enrich run` 실행
4. 필요한 경우 완료될 때까지 폴링
5. 성공을 보고하기 전에 출력 파일 검증

## 오류 처리 및 종료 코드 (Error handling and exit codes)

CLI는 다음 종료 코드를 문서화합니다:
- `0`: 성공
- `2`: 잘못된 입력
- `3`: 인증 오류
- `4`: API 오류
- `5`: 시간 초과

인증 오류가 발생할 경우:
1. `parallel-cli auth` 확인
2. `PARALLEL_API_KEY`를 확인하거나 `parallel-cli login` / `parallel-cli login --device` 실행
3. `parallel-cli`가 `PATH`에 있는지 확인

## 유지보수 (Maintenance)

현재 인증 / 설치 상태 확인:

```bash
parallel-cli auth
parallel-cli --help
```

명령어 업데이트:

```bash
parallel-cli update
pip install --upgrade parallel-web-tools
parallel-cli config auto-update-check off
```

## 주의사항 (Pitfalls)

- 사용자가 명시적으로 사람이 읽을 수 있는 형식의 출력을 원하지 않는 한 `--json`을 생략하지 마세요.
- CLI 출력에 없는 소스를 인용하지 마세요.
- `login`에는 PTY/브라우저 상호작용이 필요할 수 있습니다.
- 짧은 작업의 경우 포그라운드 실행을 선호하세요; 백그라운드 프로세스를 남용하지 마세요.
- 결과 세트가 클 경우, 모든 것을 컨텍스트에 쑤셔 넣는 대신 `/tmp/*.json`에 JSON을 저장하세요.
- Hermes 기본 도구로 충분할 때는 조용히 Parallel을 선택하지 마세요.
- 이것은 일반적으로 무료 티어를 넘어 계정 인증 및 유료 사용이 필요한 벤더 워크플로우임을 기억하세요.
