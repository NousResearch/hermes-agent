---
title: 웹 검색 및 추출
description: 여러 백엔드 제공업체를 통해 웹을 검색하고 페이지 콘텐츠를 추출합니다. 무료 셀프 호스팅 SearXNG를 포함합니다.
sidebar_label: 웹 검색
sidebar_position: 6
---

# 웹 검색 및 추출

Hermes Agent에는 여러 제공업체를 기반으로 하는 모델 호출 가능 웹 도구가 두 가지 포함되어 있습니다.

- **`web_search`** — 웹을 검색하고 순위가 매겨진 결과를 반환합니다
- **`web_extract`** — 하나 이상의 URL에서 읽기 쉬운 콘텐츠를 가져와 추출합니다

두 도구는 하나의 백엔드 선택을 통해 구성됩니다. 제공업체는 `hermes tools`를 통해 선택하거나 `config.yaml`에 직접 설정할 수 있습니다.

## 백엔드

| 제공업체 | 환경 변수 | 검색 | 추출 | 무료 요금제 |
|----------|---------|------|------|-----------|
| **Firecrawl** (기본값) | `FIRECRAWL_API_KEY` | ✔ | ✔ | 월 500크레딧 |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | ✔ 무료 (셀프 호스팅) |
| **Brave Search (무료 요금제)** | `BRAVE_SEARCH_API_KEY` | ✔ | — | 월 2,000회 검색 |
| **DDGS (DuckDuckGo)** | — (키 없음) | ✔ | — | ✔ 무료 |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | 월 1,000회 검색 |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | 월 1,000회 검색 |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | 유료 |
| **xAI (Grok)** | `XAI_API_KEY` 또는 `hermes auth add xai-oauth` | ✔ | — | 유료 (SuperGrok 또는 토큰별 과금) |

Brave Search, DDGS, xAI는 **검색 전용**이므로, `web_extract`도 필요한 경우 Firecrawl/Tavily/Exa/Parallel 중 하나와 함께 사용하세요. DDGS는 내부적으로 [`ddgs` Python 패키지](https://pypi.org/project/ddgs/)를 사용합니다. 아직 설치되지 않았다면 `pip install ddgs`를 실행하거나 Hermes가 처음 사용할 때 지연 설치하도록 하세요. xAI는 Responses API에서 Grok의 서버 측 `web_search` 도구를 실행합니다. 결과는 인덱스 기반이 아니라 LLM이 생성하므로 제목, 설명, URL 선택이 모두 모델 출력입니다 (아래의 [신뢰 모델 주의사항](#xai-grok)을 참조하세요).

**기능별 분리:** 검색과 추출에 서로 다른 제공업체를 독립적으로 사용할 수 있습니다. 예를 들어 검색에는 SearXNG(무료), 추출에는 Firecrawl을 사용할 수 있습니다. 아래의 [기능별 구성](#per-capability-configuration)을 참조하세요.

:::tip Nous 구독자
유료 [Nous Portal](https://portal.nousresearch.com) 구독이 있다면, 관리형 Firecrawl을 사용하는 **[Tool Gateway](tool-gateway.md)**를 통해 웹 검색과 추출을 이용할 수 있으며 API 키가 필요하지 않습니다. 새로 설치한 경우 `hermes setup --portal`을 실행해 로그인하면 모든 게이트웨이 도구를 한 번에 켤 수 있고, 기존 설치에서는 `hermes tools`를 통해 웹 기능만 켤 수 있습니다.
:::

---

## `web_extract`의 긴 페이지 처리 방식

백엔드는 원시 페이지 마크다운을 반환하며, 이는 매우 클 수 있습니다(포럼 스레드, 문서 사이트, 댓글이 삽입된 뉴스 기사 등). 컨텍스트 창을 사용하기 편하게 유지하기 위해 `web_extract`는 **결정론적 문자 예산**을 적용하며, LLM 요약은 수행하지 않습니다.

| 페이지 크기 (문자 수) | 처리 방식 |
|------------------------|------------|
| 예산 이하 또는 예산과 같음 (기본값 15,000) | 전체 반환 — 전체 마크다운이 에이전트에 전달됩니다 |
| 예산 초과 | 앞부분+뒷부분 창(약 75% 앞부분 / 25% 뒷부분, 마크다운 줄 경계에서 잘림)과 명시적인 `[TRUNCATED]` 푸터를 반환합니다. 전체 정리된 텍스트는 디스크에 저장되며, 푸터에는 에이전트가 생략된 중간 부분을 페이지 단위로 읽을 수 있도록 파일 경로와 정확한 `read_file` 호출이 안내됩니다 |
| 2,000,000 초과 | 저장되는 텍스트가 2MB로 제한됩니다 |

페이지별 예산은 `config.yaml`의 `web.extract_char_limit`로 구성할 수 있습니다(기본값 `15000`, 2,000~500,000으로 제한). 또한 에이전트는 도구의 `char_limit` 인수로 호출별 한도를 높일 수 있습니다.

### 잘림이 방해가 되는 경우

추출된 마크다운이 아니라 라이브 DOM이 꼭 필요한 경우(예: JS가 많은 페이지에서 추출 결과가 거의 없는 경우)에는 `browser_navigate` + `browser_snapshot`을 사용하세요. 브라우저 도구는 대형 페이지에서 자체 스냅샷 제한이 적용되는 라이브 접근성 트리를 반환합니다.

---

## 설정

### `hermes tools`를 통한 빠른 설정

`hermes tools`를 실행하고 **웹 검색 및 추출**로 이동한 다음 제공업체를 선택하세요. 마법사가 필요한 URL 또는 API 키를 요청하고 이를 구성에 기록합니다.

```bash
hermes tools
```

---

### Firecrawl (기본값)

검색과 추출을 모두 지원하는 완전한 기능의 제공업체입니다. 대부분의 사용자에게 권장됩니다.

```bash
# ~/.hermes/.env
FIRECRAWL_API_KEY=fc-your-key-here
```

[firecrawl.dev](https://firecrawl.dev)에서 키를 발급받으세요. 무료 요금제에는 월 500크레딧이 포함됩니다.

**셀프 호스팅 Firecrawl:** 클라우드 API 대신 자체 인스턴스를 사용하도록 설정할 수 있습니다.

```bash
# ~/.hermes/.env
FIRECRAWL_API_URL=http://localhost:3002
```

`FIRECRAWL_API_URL`이 설정되면 API 키는 선택 사항입니다(`USE_DB_AUTHENTICATION=false`로 서버 인증을 비활성화하세요).

---

### SearXNG (무료, 셀프 호스팅)

SearXNG는 70개 이상의 검색 엔진 결과를 집계하는 개인정보 보호 중심의 오픈 소스 메타검색 엔진입니다. **API 키가 필요하지 않으며**, 실행 중인 SearXNG 인스턴스를 Hermes에 연결하기만 하면 됩니다.

SearXNG는 **검색 전용**이므로 `web_extract`에는 별도의 추출 제공업체가 필요합니다.

#### 옵션 A — Docker로 셀프 호스팅 (권장)

이 방법을 사용하면 속도 제한이 없는 비공개 인스턴스를 만들 수 있습니다.

**1. 작업 디렉터리를 만듭니다:**

```bash
mkdir -p ~/searxng/searxng
cd ~/searxng
```

**2. `docker-compose.yml`을 작성합니다:**

```yaml
# ~/searxng/docker-compose.yml
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888/
    restart: unless-stopped
```

**3. 컨테이너를 시작합니다:**

```bash
docker compose up -d
```

**4. JSON API 형식을 활성화합니다:**

SearXNG는 기본적으로 JSON 출력을 비활성화한 상태로 제공됩니다. 생성된 구성을 복사하고 활성화하세요.

```bash
# Copy the auto-generated config out of the container
docker cp searxng:/etc/searxng/settings.yml ~/searxng/searxng/settings.yml
```

`~/searxng/searxng/settings.yml`을 엽니다.
`use_default_settings: true`가 있으면 파일에는 재정의 내용만 들어 있습니다. 나머지 설정은 내장 기본값에서 상속됩니다.
Hermes의 JSON 응답을 활성화하려면 다음 재정의를 추가하세요.

```yaml
search:
  formats:
    - html
    - json
```

`settings.yml`은 다음과 비슷해야 합니다.

```yaml
# Read the documentation before extending the defaults:
# https://docs.searxng.org/admin/settings/

use_default_settings: true

server:
  secret_key: "abcdef12345678"
  image_proxy: true

search:
  formats:
    - html
    - json
```

**5. 적용을 위해 다시 시작합니다:**

```bash
docker cp ~/searxng/searxng/settings.yml searxng:/etc/searxng/settings.yml
docker restart searxng
```

**6. 작동 여부를 확인합니다:**

```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"results\"])} results')"
```

`10 results`와 비슷한 결과가 표시되어야 합니다. `403 Forbidden`이 표시되면 JSON 형식이 아직 비활성화된 것입니다. 4단계를 다시 확인하세요.

**7. Hermes를 구성합니다:**

```bash
# ~/.hermes/.env
SEARXNG_URL=http://localhost:8888
```

그런 다음 `~/.hermes/config.yaml`에서 검색 백엔드로 SearXNG를 선택합니다.

```yaml
web:
  search_backend: "searxng"
```

또는 `hermes tools` → 웹 검색 및 추출 → SearXNG를 통해 설정하세요.

---

#### 옵션 B — 공개 인스턴스 사용

공개 SearXNG 인스턴스는 [searx.space](https://searx.space/)에 나열되어 있습니다. **JSON 형식이 활성화된** 인스턴스(표에 표시됨)를 기준으로 필터링하세요.

```bash
# ~/.hermes/.env
SEARXNG_URL=https://searx.example.com
```

:::caution 공개 인스턴스
공개 인스턴스에는 속도 제한과 가변적인 가동 시간이 있으며, 언제든 JSON 형식을 비활성화할 수 있습니다. 프로덕션 사용에는 셀프 호스팅을 강력히 권장합니다.
:::

---

#### SearXNG를 추출 제공업체와 함께 사용

SearXNG는 검색을 담당하므로 `web_extract`를 사용하려면 별도의 제공업체가 필요합니다. 기능별 키를 사용하세요.

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # or tavily, exa, parallel
```

이 구성에서 Hermes는 모든 검색 쿼리에 SearXNG를 사용하고 URL 추출에는 Firecrawl을 사용하므로, 무료 검색과 고품질 추출을 결합할 수 있습니다.

---

### Tavily

넉넉한 무료 요금제를 제공하는 AI 최적화 검색 및 추출 도구입니다.

```bash
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

[app.tavily.com](https://app.tavily.com/home)에서 키를 발급받으세요. 무료 요금제에는 월 1,000회 검색이 포함됩니다.

---

### Exa

의미를 이해하는 뉴럴 검색 도구입니다. 연구와 개념적으로 관련된 콘텐츠를 찾는 데 적합합니다.

```bash
# ~/.hermes/.env
EXA_API_KEY=your-exa-key-here
```

[exa.ai](https://exa.ai)에서 키를 발급받으세요. 무료 요금제에는 월 1,000회 검색이 포함됩니다.

---

### Parallel

AI 네이티브 검색 및 추출 도구로, 심층적인 연구 기능을 제공합니다.

```bash
# ~/.hermes/.env
PARALLEL_API_KEY=your-parallel-key-here
```

[parallel.ai](https://parallel.ai)에서 액세스 권한을 받으세요.

---

### xAI (Grok) {#xai-grok}

Responses API에서 Grok의 서버 측 [web_search 도구](https://docs.x.ai/developers/tools/web-search)를 통해 `web_search`를 라우팅합니다. Grok이 실제 검색을 수행하고 상위 결과를 구조화된 JSON으로 반환합니다.

두 가지 자격 증명 경로 중 하나로 사용할 수 있으며, 새 환경 변수나 새 설정 마법사는 필요하지 않습니다.

```bash
# ~/.hermes/.env (env-var path)
XAI_API_KEY=sk-xai-your-key-here
```

또는 SuperGrok 구독자의 경우:

```bash
hermes auth add xai-oauth
```

그런 다음 검색 백엔드로 xAI를 선택합니다.

```yaml
# ~/.hermes/config.yaml
web:
  backend: "xai"
```

**선택적 설정:**

```yaml
web:
  backend: "xai"
  xai:
    model: grok-build-0.1        # reasoning model required by web_search (default)
    allowed_domains:             # optional, max 5 — mutex with excluded_domains
      - arxiv.org
    excluded_domains:            # optional, max 5
      - example-spam.com
    timeout: 90                  # seconds (default)
```

**검색 전용**이므로 `web_extract`도 필요한 경우 Firecrawl / Tavily / Exa / Parallel과 함께 사용하세요. 401이 발생하면 제공업체가 OAuth 토큰을 한 번 강제로 새로 고친 뒤 재시도합니다(유효 기간 중간의 폐기와 사전 만료 검사가 디코딩할 수 없는 불투명 토큰을 처리). 환경 변수 자격 증명은 재시도를 건너뜁니다.

:::caution 신뢰 모델
검색 엔진 결과를 그대로 반환하는 인덱스 기반 제공업체(Brave, Tavily, Exa)와 달리, xAI는 어떤 URL을 표시할지 LLM이 선택하고 제목과 설명도 직접 작성합니다. 쿼리의 *내용*이 출력에 영향을 주므로, 악의적으로 작성된 쿼리(예: 에이전트가 신뢰할 수 없는 상위 입력에서 가져온 쿼리)가 이론적으로 Grok을 조종해 공격자가 선택한 URL을 내보내도록 만들 수 있습니다. 반환된 URL은 모델이 생성한 링크와 동일하게 취급하고, 특히 쿼리가 신뢰할 수 없는 입력에서 비롯된 경우 가져오기 전에 검증하세요.
:::

---

## 구성

### 단일 백엔드

모든 웹 기능에 하나의 제공업체를 설정합니다.

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | xai
```

### 기능별 구성

검색과 추출에 서로 다른 제공업체를 사용합니다. 이를 통해 무료 검색(SearXNG)과 유료 추출 제공업체를 조합하거나 그 반대로 구성할 수 있습니다.

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # used by web_search
  extract_backend: "firecrawl"  # used by web_extract
```

기능별 키가 비어 있으면 두 기능 모두 `web.backend`로 대체됩니다. `web.backend`도 비어 있으면 사용 가능한 API 키 또는 URL에서 백엔드를 자동으로 감지합니다.

**우선순위(기능별):**
1. `web.search_backend` / `web.extract_backend` (명시적인 기능별 설정)
2. `web.backend` (공유 대체값)
3. 환경 변수에서 자동 감지

### 자동 감지

백엔드를 명시적으로 구성하지 않으면 Hermes는 설정된 자격 증명에 따라 다음 순서에서 처음 사용 가능한 백엔드를 선택합니다.

| 존재하는 자격 증명 | 자동 선택 백엔드 |
|--------------------|--------------------|
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `PARALLEL_API_KEY` | parallel |
| `FIRECRAWL_API_KEY` 또는 `FIRECRAWL_API_URL` (또는 Nous Tool Gateway가 준비됨) | firecrawl |
| `SEARXNG_URL` | searxng |
| `BRAVE_SEARCH_API_KEY` | brave-free |
| `ddgs` 패키지를 가져올 수 있음 | ddgs |

xAI Web Search는 자동 감지 체인에 **포함되지 않습니다**. `XAI_API_KEY`가 설정되어 있거나 xAI Grok OAuth로 로그인되어 있어도 웹 트래픽이 xAI를 통해 자동으로 라우팅되지 않습니다. 해당 자격 증명은 추론 / TTS / 이미지 생성에도 사용되므로 사용자가 웹에 다른 백엔드를 원할 수 있기 때문입니다. `web.backend: "xai"`로 명시적으로 선택하세요.

---

## 설정 확인

`hermes setup`을 실행하면 감지된 웹 백엔드를 확인할 수 있습니다.

```
✅ Web Search & Extract (searxng)
```

또는 CLI를 통해 확인하세요.

```bash
# Activate the venv and run the web tools module directly
source ~/.hermes/hermes-agent/.venv/bin/activate
python -m tools.web_tools
```

이 명령은 활성 백엔드와 상태를 출력합니다.

```
✅ Web backend: searxng
   Using SearXNG (search only): http://localhost:8888
```

---

## 문제 해결

### `web_search`가 `{"success": false}`를 반환합니다

- `SEARXNG_URL`에 연결할 수 있는지 확인하세요: `curl -s "http://localhost:8888/search?q=test&format=json"`
- HTTP 403이 표시되면 JSON 형식이 비활성화된 것입니다. `settings.yml`의 `formats` 목록에 `json`을 추가하고 다시 시작하세요
- 연결 오류가 표시되면 컨테이너가 실행 중이 아닐 수 있습니다: `docker ps | grep searxng`

### `web_extract`에 "검색 전용 백엔드"라고 표시됩니다

SearXNG는 URL 콘텐츠를 추출할 수 없습니다. 추출을 지원하는 제공업체로 `web.extract_backend`를 설정하세요.

```yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"  # or tavily / exa / parallel
```

### SearXNG가 결과를 0개 반환합니다

일부 공개 인스턴스는 특정 검색 엔진이나 카테고리를 비활성화합니다. 다음을 시도하세요.
- 다른 쿼리
- [searx.space](https://searx.space/)에서 다른 공개 인스턴스
- 안정적인 결과를 위해 자체 인스턴스 셀프 호스팅

### 공개 인스턴스에서 속도 제한이 적용됩니다

셀프 호스팅 인스턴스로 전환하세요(위의 [옵션 A](#option-a--self-host-with-docker-recommended)를 참조). Docker를 사용하면 자체 인스턴스에는 속도 제한이 없습니다.

### `web_extract`가 `[TRUNCATED]` 푸터와 함께 잘린 콘텐츠를 반환합니다

문자 예산을 초과하는 페이지에서는 정상적인 동작입니다. 푸터에는 전체 정리된 텍스트가 저장된 디스크상의 파일과 생략된 중간 부분을 페이지 단위로 읽는 정확한 `read_file` 호출이 표시됩니다. 인라인으로 더 많이 보려면 `config.yaml`의 `web.extract_char_limit`을 높이거나 호출 시 더 큰 `char_limit`을 전달하세요.

---

## 선택적 스킬: `searxng-search`

`curl`을 통해 SearXNG를 직접 사용해야 하는 에이전트(예: 웹 도구 모음을 사용할 수 없을 때의 대체 수단)는 다음 선택적 스킬을 설치하세요.

```bash
hermes skills install official/research/searxng-search
```

이 스킬은 에이전트에게 다음 방법을 알려줍니다.
- `curl` 또는 Python을 통해 SearXNG JSON API 호출
- 카테고리(`general`, `news`, `science` 등)별 필터링
- 페이지 매김 및 오류 사례 처리
- SearXNG에 연결할 수 없을 때의 원활한 대체 처리
