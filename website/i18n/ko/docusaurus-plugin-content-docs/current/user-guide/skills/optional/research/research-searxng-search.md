---
title: "Searxng 검색 — 70개 이상의 엔진을 집계하는 무료 무키 메타 검색"
sidebar_label: "Searxng 검색"
description: "70개 이상의 엔진을 집계하는 무료 무키 메타 검색"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Searxng 검색

70개 이상의 엔진을 집계하는 무료 무키 메타 검색입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/research/searxng-search`로 설치 |
| 경로 | `optional-skills/research/searxng-search` |
| 버전 | `1.0.1` |
| 작성자 | hermes-agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `search`, `searxng`, `meta-search`, `self-hosted`, `free`, `fallback` |
| 관련 스킬 | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침이기도 합니다.
:::

# SearXNG 검색

[SearXNG](https://searxng.org/)를 사용하는 무료 메타 검색입니다. SearXNG는 70개 이상의 검색 엔진을 동시에 조회하는 개인정보 보호 중심의 자체 호스팅 검색 집계기입니다.

공개 인스턴스를 사용할 때는 **API 키가 필요하지 않습니다**. 완전한 제어를 위해 직접 호스팅할 수도 있습니다. 주 웹 검색 도구 세트(`FIRECRAWL_API_KEY`)가 설정되지 않은 경우 자동으로 대체 수단으로 표시됩니다.

## 설정

SearXNG는 SearXNG 인스턴스를 가리키는 `SEARXNG_URL` 환경 변수가 필요합니다.

```bash
# Public instances (no setup required)
SEARXNG_URL=https://searxng.example.com

# Self-hosted SearXNG
SEARXNG_URL=http://localhost:8888
```

인스턴스가 설정되지 않으면 이 스킬을 사용할 수 없으며 에이전트는 다른 검색 옵션으로 대체합니다.

## 감지 흐름

접근 방식을 선택하기 전에 실제로 무엇을 사용할 수 있는지 확인하세요.

```bash
# Check if SEARXNG_URL is set and the instance is reachable
curl -s --max-time 5 "${SEARXNG_URL}/search?q=test&format=json" | head -c 200
```

결정 트리:
1. `SEARXNG_URL`이 설정되어 있고 인스턴스가 응답하면 SearXNG를 사용합니다.
2. `SEARXNG_URL`이 설정되지 않았거나 연결할 수 없으면 사용 가능한 다른 검색 도구로 대체합니다.
3. 사용자가 SearXNG를 특별히 원하면 인스턴스 설정이나 공개 인스턴스 찾기를 도와줍니다.

## 방법 1: curl을 통한 CLI(권장)

`terminal`을 통해 `curl`을 사용하여 SearXNG JSON API를 호출합니다. 이렇게 하면 특정 Python 패키지가 설치되어 있다고 가정하지 않아도 됩니다.

```bash
# Text search (JSON output)
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=python+async+programming&format=json&engines=google,bing&limit=10"

# With Safesearch off
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=example&format=json&safesearch=0"

# Specific categories (general, news, science, etc.)
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=AI+news&format=json&categories=news"
```

### 일반적인 CLI 플래그

| 플래그 | 설명 | 예시 |
|------|-------------|---------|
| `q` | 검색어(URL 인코딩) | `q=python+async` |
| `format` | 출력 형식: `json`, `csv`, `rss` | `format=json` |
| `engines` | 쉼표로 구분한 엔진 이름 | `engines=google,bing,ddg` |
| `limit` | 엔진별 최대 결과 수(기본값 10) | `limit=5` |
| `categories` | 카테고리로 필터링 | `categories=news,science` |
| `safesearch` | 0=없음, 1=보통, 2=엄격 | `safesearch=0` |
| `time_range` | 필터: `day`, `week`, `month`, `year` | `time_range=week` |

### JSON 결과 파싱

```bash
# Extract titles and URLs from JSON
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

각 결과에는 `title`, `url`, `content`(발췌), `engine`, `parsed_url`, `img_src`, `thumbnail`, `author`, `published_date`가 반환됩니다.

## 방법 2: `requests`를 통한 Python API

`requests` 라이브러리를 사용하여 SearXNG REST API를 Python에서 직접 사용합니다.

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

## SearXNG 자체 호스팅

직접 SearXNG 인스턴스를 실행하려면 다음과 같이 하세요.

```bash
# Using Docker
docker run -d -p 8888:8080 \
  -v $(pwd)/searxng:/etc/searxng \
  searxng/searxng:latest

# Then set
SEARXNG_URL=http://localhost:8888
```

또는 pip로 설치합니다.
```bash
pip install searxng
# Edit /etc/searxng/settings.yml
searxng-run
```

공개 SearXNG 인스턴스는 다음에서 이용할 수 있습니다.
- `https://searxng.example.com`(공개 인스턴스로 대체)

## 워크플로: 검색 후 추출

SearXNG는 제목, URL, 발췌문을 반환하며 전체 페이지 콘텐츠는 반환하지 않습니다. 전체 페이지 콘텐츠를 얻으려면 먼저 검색한 다음 `web_extract`, 브라우저 도구 또는 `curl`로 가장 관련성 높은 URL을 추출하세요.

```bash
# Search for relevant pages
curl -s "${SEARXNG_URL}/search?q=fastapi+deployment&format=json&limit=3"
# Output: list of results with titles and URLs

# Then extract the best URL with web_extract
```

## 제한 사항

- **인스턴스 가용성**: SearXNG 인스턴스가 중단되었거나 연결할 수 없으면 검색이 실패합니다. 항상 `SEARXNG_URL`이 설정되어 있고 인스턴스에 연결할 수 있는지 확인하세요.
- **콘텐츠 추출 불가**: SearXNG는 전체 글이 아니라 발췌문을 반환합니다. 전체 글에는 `web_extract`, 브라우저 도구 또는 `curl`을 사용하세요.
- **속도 제한**: 일부 공개 인스턴스는 요청을 제한합니다. 직접 호스팅하면 이를 피할 수 있습니다.
- **엔진 범위**: 사용 가능한 엔진은 SearXNG 인스턴스의 설정에 따라 달라집니다. 일부 엔진은 비활성화되어 있을 수 있습니다.
- **결과 최신성**: 메타 검색은 외부 엔진을 집계하므로 결과의 최신성은 해당 엔진에 따라 달라집니다.

## 문제 해결

| 문제 | 가능한 원인 | 조치 |
|------------|--------------|------------|
| `SEARXNG_URL` not set | 인스턴스가 설정되지 않음 | 공개 SearXNG 인스턴스를 사용하거나 직접 설정 |
| Connection refused | 인스턴스가 실행 중이 아니거나 URL이 잘못됨 | URL이 올바르고 인스턴스가 실행 중인지 확인 |
| Empty results | 인스턴스가 쿼리를 차단함 | 다른 인스턴스를 시도하거나 직접 호스팅 |
| Slow responses | 공개 인스턴스에 부하가 걸림 | 직접 호스팅하거나 부하가 적은 공개 인스턴스 사용 |
| `json` format not supported | 오래된 SearXNG 버전 | `format=rss`를 시도하거나 SearXNG 업그레이드 |

## 함정

- **항상 `SEARXNG_URL`을 설정하세요**: 설정하지 않으면 스킬이 작동하지 않습니다.
- **쿼리를 URL 인코딩하세요**: curl에서 공백과 특수 문자를 URL 인코딩하거나 Python에서 `urllib.parse.quote()`를 사용하세요.
- **`format=json`을 사용하세요**: 기본 형식은 기계가 읽을 수 없을 수 있습니다. 항상 JSON을 명시적으로 요청하세요.
- **시간 제한을 설정하세요**: 연결할 수 없는 인스턴스에서 멈추지 않도록 항상 `--max-time` 또는 `timeout=`을 사용하세요.
- **직접 호스팅이 가장 좋습니다**: 공개 인스턴스는 중단되거나 속도를 제한하거나 차단할 수 있습니다. 직접 호스팅한 인스턴스는 안정적입니다.

## 인스턴스 찾기

`SEARXNG_URL`이 설정되지 않았고 사용자가 SearXNG에 관해 물으면 다음 중 하나를 돕습니다.
1. 공개 SearXNG 인스턴스를 찾습니다("public searxng instance"로 검색).
2. Docker 또는 pip로 직접 설정합니다.

공개 인스턴스 목록은 https://searxng.org/에서 확인할 수 있습니다.
