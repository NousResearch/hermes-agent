---
title: "Parallel Cli — 에이전트 네이티브 웹 검색, 심층 리서치 및 보강"
sidebar_label: "Parallel Cli"
description: "에이전트 네이티브 웹 검색, 심층 리서치 및 보강"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Parallel Cli

에이전트 네이티브 웹 검색, 심층 리서치 및 보강.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/research/parallel-cli`로 설치 |
| 경로 | `optional-skills/research/parallel-cli` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Research`, `Web`, `Search`, `Deep-Research`, `Enrichment`, `CLI` |
| 관련 스킬 | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`mcporter`](/docs/user-guide/skills/optional/mcp/mcp-mcporter) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Parallel CLI

사용자가 명시적으로 Parallel을 원하거나, 웹 검색·추출·심층 리서치·보강·엔터티 탐색·모니터링을 위한 터미널 네이티브 워크플로가 Parallel의 공급업체별 스택을 활용하면 좋은 경우 `parallel-cli`를 사용합니다.

이는 Hermes의 핵심 기능이 아닌 선택적 서드파티 워크플로입니다.

중요한 기대 사항:
- Parallel은 완전 무료 로컬 도구가 아니라 무료 티어가 있는 유료 서비스입니다.
- Hermes 네이티브 `web_search` / `web_extract`와 기능이 겹치므로 일반적인 조회에 기본적으로 Parallel을 우선하지 마세요.
- 사용자가 Parallel을 구체적으로 언급했거나 Parallel의 보강, FindAll 또는 모니터 워크플로와 같은 기능이 필요한 경우 이 스킬을 우선하세요.

`parallel-cli`는 에이전트를 위해 설계되었습니다.
- `--json`을 통한 JSON 출력
- 비대화형 명령 실행
- `--no-wait`, `status`, `poll`을 통한 비동기 장기 실행 작업
- `--previous-interaction-id`를 통한 컨텍스트 연결
- 하나의 CLI에서 검색, 추출, 리서치, 보강, 엔터티 탐색 및 모니터링

## 사용 시점

다음과 같은 경우 이 스킬을 우선하세요.
- 사용자가 Parallel 또는 `parallel-cli`를 명시적으로 언급한 경우
- 단순한 일회성 검색/추출보다 풍부한 워크플로가 필요한 경우
- 나중에 시작하고 폴링할 수 있는 비동기 심층 리서치 작업이 필요한 경우
- 구조화된 보강, FindAll 엔터티 탐색 또는 모니터링이 필요한 경우

Parallel을 특별히 요청하지 않았고 빠른 일회성 조회가 목적이라면 Hermes 네이티브 `web_search` / `web_extract`를 우선하세요.

## 설치

환경에서 가능한 가장 간단한 설치 경로를 시도하세요.

### Homebrew

```bash
brew install parallel-web/tap/parallel-cli
```

### npm

```bash
npm install -g parallel-web-cli
```

### Python 패키지

```bash
pip install "parallel-web-tools[cli]"
```

### 독립 실행형 설치 프로그램

```bash
curl -fsSL https://parallel.ai/install.sh | bash
```

격리된 Python 설치를 원한다면 `pipx`도 사용할 수 있습니다.

```bash
pipx install "parallel-web-tools[cli]"
pipx ensurepath
```

## 인증

대화형 로그인:

```bash
parallel-cli login
```

헤드리스 / SSH / CI:

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

인증에 브라우저 상호작용이 필요하면 `pty=true`로 실행하세요.

## 핵심 규칙

1. 기계가 읽을 수 있는 출력이 필요하면 항상 `--json`을 우선하세요.
2. 명시적 인수와 비대화형 흐름을 우선하세요.
3. 장기 실행 작업에는 `--no-wait`를 사용한 다음 `status` / `poll`을 사용하세요.
4. CLI 출력이 반환한 URL만 인용하세요.
5. 후속 질문이 예상되면 큰 JSON 출력을 임시 파일에 저장하세요.
6. 진정으로 오래 실행되는 워크플로에만 백그라운드 프로세스를 사용하고, 그 외에는 포그라운드에서 실행하세요.
7. 사용자가 Parallel을 구체적으로 원하거나 Parallel 전용 워크플로가 필요한 경우가 아니라면 Hermes 네이티브 도구를 우선하세요.

## 빠른 참조

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

## 일반적인 플래그와 패턴

일반적으로 유용한 플래그:
- 구조화된 출력을 위한 `--json`
- 비동기 작업을 위한 `--no-wait`
- 이전 컨텍스트를 재사용하는 후속 작업을 위한 `--previous-interaction-id <id>`
- 검색 결과 수를 위한 `--max-results <n>`
- 검색 동작을 위한 `--mode one-shot|agentic`
- `--include-domains domain1.com,domain2.com`
- `--exclude-domains domain1.com,domain2.com`
- `--after-date YYYY-MM-DD`

편리할 때 표준 입력에서 읽으세요.

```bash
echo "What is the latest funding for Anthropic?" | parallel-cli search - --json
echo "Research question" | parallel-cli research run - --json
```

결과를 요약할 때:
- 답변을 먼저 제시하세요.
- 날짜, 이름, 구체적인 사실을 포함하세요.
- 반환된 출처만 인용하세요.
- URL이나 출처 제목을 지어내지 마세요.

## 검색

구조화된 결과가 필요한 최신 웹 조회에 사용합니다.

```bash
parallel-cli search "What is Anthropic's latest AI model?" --json
parallel-cli search "SEC filings for Apple" --include-domains sec.gov --json
parallel-cli search "bitcoin price" --after-date 2026-01-01 --max-results 10 --json
parallel-cli search "latest browser benchmarks" --mode one-shot --json
parallel-cli search "AI coding agent enterprise reviews" --mode agentic --json
```

유용한 제약 조건:
- 신뢰할 수 있는 출처로 범위를 좁히려면 `--include-domains`
- 잡음이 많은 도메인을 제외하려면 `--exclude-domains`
- 최신 결과를 위해 `--after-date`
- 더 폭넓은 결과가 필요할 때 `--max-results`

후속 질문이 예상되면 출력을 저장하세요.

```bash
parallel-cli search "latest React 19 changes" --json -o /tmp/react-19-search.json
```

## 추출

URL에서 정제된 콘텐츠나 마크다운을 가져올 때 사용합니다.

```bash
parallel-cli extract https://example.com --json
parallel-cli extract https://company.com --objective "Find pricing info" --json
parallel-cli extract https://example.com --full-content --json
parallel-cli fetch https://example.com --json
```

페이지가 광범위하고 그중 한 부분만 필요한 경우 `--objective`를 사용하세요.

## 심층 리서치

시간이 걸릴 수 있는 더 깊은 다단계 리서치 작업에 사용합니다.

일반적인 프로세서 등급:
- 더 빠르고 저렴한 작업에는 `lite` / `base`
- 더 철저한 종합에는 `core` / `pro`
- 가장 무거운 리서치 작업에는 `ultra`

### 동기 실행

```bash
parallel-cli research run \
  "Compare the leading AI coding agents by pricing, model support, and enterprise controls" \
  --processor core \
  --json
```

### 비동기 시작 + 폴링

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

### 컨텍스트 연결 / 후속 작업

```bash
parallel-cli research run "What are the top AI coding agents?" --json
parallel-cli research run \
  "What enterprise controls does the top-ranked one offer?" \
  --previous-interaction-id trun_xxx \
  --json
```

권장 Hermes 워크플로:
1. `--no-wait --json`으로 시작합니다.
2. 반환된 실행/작업 ID를 기록합니다.
3. 사용자가 다른 작업을 계속하고 싶어 하면 계속 진행합니다.
4. 나중에 `status` 또는 `poll`을 호출합니다.
5. 반환된 출처의 인용과 함께 최종 보고서를 요약합니다.

## 보강

사용자가 CSV/JSON/표 형식 입력을 가지고 웹 리서치로 추론한 추가 열을 원할 때 사용합니다.

### 열 제안

```bash
parallel-cli enrich suggest "Find the CEO and annual revenue" --json
```

### 설정 계획

```bash
parallel-cli enrich plan -o config.yaml
```

### 인라인 데이터

```bash
parallel-cli enrich run \
  --data '[{"company": "Anthropic"}, {"company": "Mistral"}]' \
  --intent "Find headquarters and employee count" \
  --json
```

### 비대화형 파일 실행

```bash
parallel-cli enrich run \
  --source-type csv \
  --source companies.csv \
  --target enriched.csv \
  --source-columns '[{"name": "company", "description": "Company name"}]' \
  --intent "Find the CEO and annual revenue"
```

### YAML 설정 실행

```bash
parallel-cli enrich run config.yaml
```

### 상태 / 폴링

```bash
parallel-cli enrich status <task_group_id> --json
parallel-cli enrich poll <task_group_id> --json
```

비대화형으로 실행할 때는 열 정의에 명시적인 JSON 배열을 사용하세요.
성공을 보고하기 전에 출력 파일을 검증하세요.

## FindAll

사용자가 짧은 답변이 아니라 탐색된 데이터 세트를 원할 때 웹 규모의 엔터티 탐색에 사용합니다.

```bash
parallel-cli findall run "Find AI coding agent startups with enterprise offerings" --json
parallel-cli findall run "AI startups in healthcare" -n 25 --json
parallel-cli findall status <run_id> --json
parallel-cli findall poll <run_id> --json
parallel-cli findall result <run_id> --json
parallel-cli findall schema <run_id> --json
```

사용자가 나중에 검토·필터링·보강할 수 있는 탐색된 엔터티 집합을 원한다면 일반 검색보다 적합합니다.

## 모니터

시간에 따른 변경 사항을 지속적으로 감지할 때 사용합니다.

```bash
parallel-cli monitor list --json
parallel-cli monitor get <monitor_id> --json
parallel-cli monitor events <monitor_id> --json
parallel-cli monitor delete <monitor_id> --json
```

주기와 전달 방식이 중요하므로 일반적으로 생성이 민감한 부분입니다.

```bash
parallel-cli monitor create --help
```

일회성 가져오기가 아니라 페이지나 출처를 반복적으로 추적하려는 경우 사용하세요.

## 권장 Hermes 사용 패턴

### 인용이 포함된 빠른 답변
1. `parallel-cli search ... --json`을 실행합니다.
2. 제목, URL, 날짜, 발췌문을 파싱합니다.
3. 반환된 URL만 사용해 인라인 인용과 함께 요약합니다.

### URL 조사
1. `parallel-cli extract URL --json`을 실행합니다.
2. 필요하면 `--objective` 또는 `--full-content`로 다시 실행합니다.
3. 추출된 마크다운을 인용하거나 요약합니다.

### 장기 리서치 워크플로
1. `parallel-cli research run ... --no-wait --json`을 실행합니다.
2. 반환된 ID를 저장합니다.
3. 다른 작업을 계속하거나 주기적으로 폴링합니다.
4. 인용과 함께 최종 보고서를 요약합니다.

### 구조화된 보강 워크플로
1. 입력 파일과 열을 확인합니다.
2. `enrich suggest`를 사용하거나 보강할 열을 명시적으로 제공합니다.
3. `enrich run`을 실행합니다.
4. 필요하면 완료될 때까지 폴링합니다.
5. 성공을 보고하기 전에 출력 파일을 검증합니다.

## 오류 처리 및 종료 코드

CLI에 문서화된 종료 코드:
- `0` 성공
- `2` 잘못된 입력
- `3` 인증 오류
- `4` API 오류
- `5` 시간 초과

인증 오류가 발생하면:
1. `parallel-cli auth`를 확인합니다.
2. `PARALLEL_API_KEY`를 확인하거나 `parallel-cli login` / `parallel-cli login --device`를 실행합니다.
3. `parallel-cli`가 `PATH`에 있는지 확인합니다.

## 유지 관리

현재 인증 / 설치 상태를 확인합니다.

```bash
parallel-cli auth
parallel-cli --help
```

명령을 업데이트합니다.

```bash
parallel-cli update
pip install --upgrade parallel-web-tools
parallel-cli config auto-update-check off
```

## 주의 사항

- 사용자가 사람이 읽는 형식의 출력을 명시적으로 원하지 않는 한 `--json`을 생략하지 마세요.
- CLI 출력에 없는 출처를 인용하지 마세요.
- `login`에는 PTY/브라우저 상호작용이 필요할 수 있습니다.
- 짧은 작업에는 포그라운드 실행을 우선하고, 백그라운드 프로세스를 과도하게 사용하지 마세요.
- 결과 집합이 크면 모든 내용을 컨텍스트에 넣는 대신 JSON을 `/tmp/*.json`에 저장하세요.
- Hermes 네이티브 도구로 이미 충분한데도 Parallel을 조용히 선택하지 마세요.
- 이는 일반적으로 계정 인증이 필요하고 무료 티어를 넘어서는 사용에는 유료 사용량이 필요한 공급업체 워크플로임을 기억하세요.
