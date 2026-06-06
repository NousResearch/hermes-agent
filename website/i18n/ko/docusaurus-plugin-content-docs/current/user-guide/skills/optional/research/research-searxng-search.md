---
title: "Searxng Search — SearXNG를 통한 무료 메타 검색 — 70개 이상의 검색 엔진 결과 집계"
sidebar_label: "Searxng Search"
description: "SearXNG를 통한 무료 메타 검색 — 70개 이상의 검색 엔진 결과 집계"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Searxng Search

SearXNG를 통한 무료 메타 검색 — 70개 이상의 검색 엔진 결과를 집계합니다. 자체 호스팅하거나 공개 인스턴스를 사용할 수 있습니다. API 키가 필요하지 않습니다. 웹 검색 툴셋을 사용할 수 없을 때 자동으로 폴백(fallback)으로 작동합니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/searxng-search` |
| Path | `optional-skills/research/searxng-search` |
| Version | `1.0.0` |
| Author | hermes-agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `search`, `searxng`, `meta-search`, `self-hosted`, `free`, `fallback` |
| Related skills | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel) |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# SearXNG Search

[SearXNG](https://searxng.org/)를 사용한 무료 메타 검색 — 70개 이상의 검색 엔진에 동시에 쿼리를 수행하는 프라이버시 존중형, 자체 호스팅 검색 집계기입니다.

공개 인스턴스를 사용할 경우 **API 키가 필요하지 않습니다**. 완벽한 제어를 위해 자체 호스팅할 수도 있습니다. 메인 웹 검색 툴셋(`FIRECRAWL_API_KEY`)이 구성되지 않은 경우 자동으로 폴백으로 나타납니다.

## Configuration

SearXNG에는 SearXNG 인스턴스를 가리키는 `SEARXNG_URL` 환경 변수가 필요합니다.

```bash
# 공개 인스턴스 (설정 불필요)
SEARXNG_URL=https://searxng.example.com

# 자체 호스팅 SearXNG
SEARXNG_URL=http://localhost:8888
```

인스턴스가 구성되지 않은 경우 이 스킬은 사용할 수 없으며 에이전트는 다른 검색 옵션으로 폴백합니다.

## Detection Flow

접근 방식을 선택하기 전에 실제로 사용할 수 있는 것이 무엇인지 확인하세요:

```bash
# SEARXNG_URL이 설정되어 있고 인스턴스에 도달 가능한지 확인
curl -s --max-time 5 "${SEARXNG_URL}/search?q=test&format=json" | head -c 200
```

결정 트리:
1. `SEARXNG_URL`이 설정되어 있고 인스턴스가 응답하면 SearXNG를 사용합니다.
2. `SEARXNG_URL`이 설정되어 있지 않거나 도달할 수 없으면 다른 사용 가능한 검색 도구로 폴백합니다.
3. 사용자가 특별히 SearXNG를 원할 경우, 인스턴스를 설정하거나 공개 인스턴스를 찾도록 도와줍니다.

## Method 1: CLI via curl (Preferred)

`terminal`을 통해 `curl`을 사용하여 SearXNG JSON API를 호출하세요. 이렇게 하면 특정 Python 패키지가 설치되어 있다고 가정하지 않게 됩니다.

```bash
# 텍스트 검색 (JSON 출력)
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=python+async+programming&format=json&engines=google,bing&limit=10"

# 세이프서치 끄기
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=example&format=json&safesearch=0"

# 특정 카테고리 (general, news, science 등)
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=AI+news&format=json&categories=news"
```

### Common CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `q` | 쿼리 문자열 (URL 인코딩됨) | `q=python+async` |
| `format` | 출력 형식: `json`, `csv`, `rss` | `format=json` |
| `engines` | 쉼표로 구분된 엔진 이름 | `engines=google,bing,ddg` |
| `limit` | 엔진당 최대 결과 수 (기본값 10) | `limit=5` |
| `categories` | 카테고리별 필터링 | `categories=news,science` |
| `safesearch` | 0=없음, 1=보통, 2=엄격 | `safesearch=0` |
| `time_range` | 필터: `day`, `week`, `month`, `year` | `time_range=week` |

### Parsing JSON Results

```bash
# JSON에서 제목과 URL 추출
curl -s --max-time 10 "${SEARXNG_URL}/search?q=fastapi&format=json&limit=5" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('results', []):
    print(r.get('title',''))
    print(r.get('url',''))
    print(r.get('content','')[:200])
    print()
"
```

결과당 다음을 반환합니다: `title`, `url`, `content` (스니펫), `engine`, `parsed_url`, `img_src`, `thumbnail`, `author`, `published_date`

## Method 2: Python API via `requests`

`requests` 라이브러리를 사용하여 Python에서 직접 SearXNG REST API를 사용하세요:

```python
import os, requests, urllib.parse

base_url = os.environ.get("SEARXNG_URL", "")
if not base_url:
    raise RuntimeError("SEARXNG_URL is not set")

query = "fastapi deployment guide"
params = {
    "q": query,
    "format": "json",
    "limit": 5,
    "engines": "google,bing",
}

resp = requests.get(f"{base_url}/search", params=params, timeout=10)
resp.raise_for_status()
data = resp.json()

for r in data.get("results", []):
    print(r["title"])
    print(r["url"])
    print(r.get("content", "")[:200])
    print()
```

## Method 3: searxng-data Python Package

더 구조화된 접근을 위해 `searxng-data` 패키지를 설치하세요:

```bash
pip install searxng-data
```

```python
from searxng_data import engines

# 사용 가능한 엔진 목록 확인
print(engines.list_engines())
```

참고: 이 패키지는 엔진 메타데이터만 제공하며 검색 API 자체는 제공하지 않습니다.

## Self-Hosting SearXNG

고유한 SearXNG 인스턴스를 실행하려면:

```bash
# Docker 사용
docker run -d -p 8888:8080 \
  -v $(pwd)/searxng:/etc/searxng \
  searxng/searxng:latest

# 그런 다음 설정합니다
SEARXNG_URL=http://localhost:8888
```

또는 pip를 통해 설치합니다:
```bash
pip install searxng
# /etc/searxng/settings.yml 편집
searxng-run
```

공개 SearXNG 인스턴스는 다음에서 사용할 수 있습니다:
- `https://searxng.example.com` (원하는 공개 인스턴스로 교체하세요)

## Workflow: Search then Extract

SearXNG는 전체 페이지 내용이 아닌 제목, URL 및 스니펫을 반환합니다. 전체 페이지 내용을 얻으려면 먼저 검색한 다음 `web_extract`, 브라우저 도구 또는 `curl`을 사용하여 가장 관련성 높은 URL을 추출하세요.

```bash
# 관련 페이지 검색
curl -s "${SEARXNG_URL}/search?q=fastapi+deployment&format=json&limit=3"
# 출력: 제목과 URL이 있는 결과 목록

# 그런 다음 web_extract를 사용하여 최고의 URL 추출
```

## Limitations

- **인스턴스 가용성**: SearXNG 인스턴스가 다운되었거나 도달할 수 없는 경우 검색이 실패합니다. 항상 `SEARXNG_URL`이 설정되어 있고 인스턴스에 도달 가능한지 확인하세요.
- **콘텐츠 추출 불가**: SearXNG는 전체 페이지 내용이 아닌 스니펫을 반환합니다. 전체 기사는 `web_extract`, 브라우저 도구 또는 `curl`을 사용하세요.
- **속도 제한**: 일부 공개 인스턴스는 요청을 제한합니다. 자체 호스팅은 이 문제를 피할 수 있습니다.
- **엔진 커버리지**: 사용 가능한 엔진은 SearXNG 인스턴스 구성에 따라 다릅니다. 일부 엔진은 비활성화될 수 있습니다.
- **결과 최신성**: 메타 검색은 외부 엔진을 집계합니다 — 결과의 최신성은 해당 엔진에 의존합니다.

## Troubleshooting

| Problem | Likely Cause | What To Do |
|---------|--------------|------------|
| `SEARXNG_URL` not set | 인스턴스가 구성되지 않음 | 공개 SearXNG 인스턴스를 사용하거나 자체 설정하세요 |
| Connection refused | 인스턴스가 실행되지 않거나 잘못된 URL임 | URL이 올바르고 인스턴스가 실행 중인지 확인하세요 |
| Empty results | 인스턴스가 쿼리를 차단함 | 다른 인스턴스를 시도하거나 자체 호스팅하세요 |
| Slow responses | 공개 인스턴스에 부하가 걸림 | 자체 호스팅하거나 부하가 적은 공개 인스턴스를 사용하세요 |
| `json` format not supported | 오래된 SearXNG 버전 | `format=rss`를 시도하거나 SearXNG를 업그레이드하세요 |

## Pitfalls

- **항상 `SEARXNG_URL` 설정**: 이 설정 없이는 스킬이 작동할 수 없습니다.
- **URL 인코딩 쿼리**: 공백 및 특수 문자는 curl에서 URL 인코딩해야 하며, Python에서는 `urllib.parse.quote()`를 사용하세요.
- **`format=json` 사용**: 기본 형식은 기계가 읽을 수 없을 수 있습니다. 항상 JSON을 명시적으로 요청하세요.
- **시간 초과 설정**: 도달할 수 없는 인스턴스에서 중단되지 않도록 항상 `--max-time` 또는 `timeout=`을 사용하세요.
- **자체 호스팅 권장**: 공개 인스턴스는 다운되거나, 속도 제한이 걸리거나, 차단될 수 있습니다. 자체 호스팅 인스턴스는 안정적입니다.

## Instance Discovery

`SEARXNG_URL`이 설정되어 있지 않고 사용자가 SearXNG에 대해 묻는 경우, 다음 중 하나를 도와주세요:
1. 공개 SearXNG 인스턴스 찾기 ("public searxng instance" 검색)
2. Docker 또는 pip를 사용하여 자체 인스턴스 설정하기

공개 인스턴스 목록은 다음 사이트에서 확인할 수 있습니다: https://searxng.org/
