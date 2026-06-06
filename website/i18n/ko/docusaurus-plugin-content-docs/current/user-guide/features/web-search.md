---
title: 웹 검색 & 추출 (Web Search & Extract)
description: 웹을 검색하고 무료 자체 호스팅 SearXNG를 포함한 여러 백엔드 제공업체를 사용하여 페이지 콘텐츠를 추출합니다.
sidebar_label: 웹 검색 (Web Search)
sidebar_position: 6
---

# 웹 검색 & 추출 (Web Search & Extract)

Hermes Agent는 여러 제공업체에서 지원하는 두 가지 모델 호출 가능한 웹 도구를 포함합니다:

- **`web_search`** — 웹을 검색하고 순위가 매겨진 결과를 반환합니다.
- **`web_extract`** — 하나 이상의 URL에서 읽을 수 있는 콘텐츠를 가져오고 추출합니다.

두 도구 모두 단일 백엔드 선택을 통해 구성됩니다. 제공업체는 `hermes tools`를 통해 선택하거나 `config.yaml`에 직접 설정합니다.

## 백엔드 (Backends)

| 제공업체 (Provider) | 환경 변수 (Env Var) | 검색 (Search) | 추출 (Extract) | 무료 티어 (Free tier) |
|----------|---------|--------|---------|-----------|
| **Firecrawl** (기본값) | `FIRECRAWL_API_KEY` | ✔ | ✔ | 월 500 크레딧 |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | ✔ 무료 (자체 호스팅) |
| **Brave Search (무료 티어)** | `BRAVE_SEARCH_API_KEY` | ✔ | — | 월 2,000 쿼리 |
| **DDGS (DuckDuckGo)** | — (키 없음) | ✔ | — | ✔ 무료 |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | 월 1,000 검색 |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | 월 1,000 검색 |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | 유료 |
| **xAI (Grok)** | `XAI_API_KEY` 또는 `hermes auth login xai-oauth` | ✔ | — | 유료 (SuperGrok 또는 토큰당) |

Brave Search, DDGS 및 xAI는 **검색 전용**입니다 — `web_extract`도 필요할 때 Firecrawl/Tavily/Exa/Parallel 중 하나와 결합하세요. DDGS는 기본적으로 [`ddgs` Python 패키지](https://pypi.org/project/ddgs/)를 사용합니다. 이미 설치되어 있지 않으면 `pip install ddgs`를 실행하거나 Hermes가 처음 사용할 때 지연 설치(lazy-install)하도록 하세요. xAI는 Responses API에서 Grok의 서버 측 `web_search` 도구를 실행합니다 — 결과는 인덱스 기반이 아니라 LLM 생성되므로 제목, 설명, URL 선택 모두가 모델의 출력입니다 (아래의 [신뢰 모델 주의사항](#xai-grok) 참고).

**기능별 분리:** 검색과 추출에 대해 백엔드를 독립적으로 설정할 수 있습니다 — 예를 들어, 검색에는 SearXNG(무료)를, 추출에는 Firecrawl을 사용할 수 있습니다. 아래의 [기능별 구성](#per-capability-configuration)을 참고하세요.

:::tip Nous 구독자
유료 [Nous Portal](https://portal.nousresearch.com) 구독이 있는 경우, API 키 없이도 관리형 Firecrawl을 통해 **[Tool Gateway](tool-gateway.md)**에서 웹 검색 및 추출을 사용할 수 있습니다. 새로 설치하는 경우 `hermes setup --portal`을 실행하여 로그인하고 모든 게이트웨이 도구를 한 번에 켤 수 있습니다. 기존 설치에서는 `hermes tools`를 통해 웹 기능만 켤 수 있습니다.
:::

---

## `web_extract`가 긴 페이지를 처리하는 방법

백엔드는 매우 클 수 있는 원시 페이지 마크다운(포럼 스레드, 문서 사이트, 임베디드 댓글이 있는 뉴스 기사 등)을 반환합니다. 컨텍스트 창을 사용 가능하게 유지하고 비용을 낮추기 위해 `web_extract`는 에이전트에게 전달하기 전에 반환된 콘텐츠를 **`web_extract` 보조 모델(auxiliary model)**을 통해 실행합니다. 작동 방식은 전적으로 크기에 따라 결정됩니다:

| 페이지 크기 (문자 수) | 발생하는 일 (What happens) |
|------------------------|--------------|
| 5,000 미만 | 있는 그대로 반환됨 — LLM 호출 없음, 전체 마크다운이 에이전트에 도달함 |
| 5,000 – 500,000 | `web_extract` 보조 모델을 통한 단일 패스 요약, 약 5,000자의 출력으로 제한됨 |
| 500,000 – 2,000,000 | 청크로 분할: 10만 자 청크로 나누고, 병렬로 각각 요약한 다음 최종 요약(약 5,000자)으로 통합함 |
| 2,000,000 이상 | 거부되고 더 구체적인 소스 URL을 사용하라는 힌트가 제공됨 |

요약은 원본 형식에 있는 인용문, 코드 블록, 핵심 사실을 유지합니다 — 즉, 패러프레이징(paraphraser) 도구가 아니라 콘텐츠 압축기(content compressor)입니다. 요약에 실패하거나 시간 초과가 발생하면 Hermes는 쓸모없는 오류를 내는 대신 원본 콘텐츠의 처음 약 5,000자를 반환하는 폴백(fallback)을 사용합니다.

### 어떤 모델이 요약을 수행하나요?

`web_extract` 보조 작업이 수행합니다. 기본적으로 (`auxiliary.web_extract.provider: "auto"`) 이것은 `hermes model`과 동일한 제공자, 동일한 모델인 **주요 채팅 모델**입니다. 대부분의 설정에는 적합하지만, 비싼 추론 모델(Opus, MiniMax M2.7 등)에서는 긴 페이지 추출마다 상당한 비용이 추가될 수 있습니다.

주요 모델에 상관없이 추출 요약을 저렴하고 빠른 모델로 라우팅하려면:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  web_extract:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 360       # 초 단위; 요약 시간 초과가 발생하면 늘리세요
```

또는 대화형으로 선택할 수도 있습니다: `hermes model` → **Configure auxiliary models** → `web_extract`.

전체 참조 및 작업별 재정의 패턴은 [보조 모델(Auxiliary Models)](/user-guide/configuration#auxiliary-models)을 참고하세요.

### 요약이 방해가 될 때

구조화된 페이지를 스크래핑할 때 LLM 요약이 중요한 필드를 삭제할 수 있는 경우처럼 원시(raw), 요약되지 않은 페이지 콘텐츠가 특별히 필요한 경우, 대신 `browser_navigate` + `browser_snapshot`을 사용하세요. 브라우저 도구는 보조 모델의 재작성 없이 (거대한 페이지에 대한 자체 8,000자 스냅샷 제한의 적용을 받음) 라이브 접근성 트리를 반환합니다.

---

## 설정 (Setup)

### `hermes tools`를 통한 빠른 설정

`hermes tools`를 실행하고 **Web Search & Extract**로 이동하여 제공업체를 선택합니다. 마법사가 필요한 URL이나 API 키를 입력하라는 프롬프트를 표시하고 구성에 작성합니다.

```bash
hermes tools
```

---

### Firecrawl (기본값)

풀 피처 검색 및 추출. 대부분의 사용자에게 권장됩니다.

```bash
# ~/.hermes/.env
FIRECRAWL_API_KEY=fc-your-key-here
```

[firecrawl.dev](https://firecrawl.dev)에서 키를 얻으세요. 무료 티어에는 월 500 크레딧이 포함됩니다.

**자체 호스팅 Firecrawl:** 클라우드 API 대신 자체 인스턴스를 지정합니다:

```bash
# ~/.hermes/.env
FIRECRAWL_API_URL=http://localhost:3002
```

`FIRECRAWL_API_URL`이 설정되면 API 키는 선택 사항입니다 (`USE_DB_AUTHENTICATION=false`로 서버 인증을 비활성화).

---

### SearXNG (무료, 자체 호스팅)

SearXNG는 70개 이상의 검색 엔진에서 결과를 집계하는 개인 정보 보호 중심의 오픈소스 메타검색 엔진입니다. **API 키가 필요하지 않습니다** — 실행 중인 SearXNG 인스턴스를 Hermes에 지정하기만 하면 됩니다.

SearXNG는 **검색 전용**입니다 — `web_extract`에는 별도의 추출 제공업체가 필요합니다.

#### 옵션 A — Docker로 자체 호스팅 (권장)

이 방법은 속도 제한이 없는 개인 인스턴스를 제공합니다.

**1. 작업 디렉토리 생성:**

```bash
mkdir -p ~/searxng/searxng
cd ~/searxng
```

**2. `docker-compose.yml` 작성:**

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

**3. 컨테이너 시작:**

```bash
docker compose up -d
```

**4. JSON API 형식 활성화:**

SearXNG는 기본적으로 JSON 출력이 비활성화된 채로 제공됩니다. 생성된 구성을 복사하여 활성화하세요:

```bash
# 생성된 구성을 컨테이너에서 복사
docker cp searxng:/etc/searxng/settings.yml ~/searxng/searxng/settings.yml
```

`~/searxng/searxng/settings.yml`을 열고 `formats` 블록을 찾습니다(약 84번째 줄):

```yaml
# 변경 전 (기본값 — JSON 비활성화됨):
formats:
  - html

# 변경 후 (Hermes를 위해 JSON 활성화):
formats:
  - html
  - json
```

**5. 적용을 위한 재시작:**

```bash
docker cp ~/searxng/searxng/settings.yml searxng:/etc/searxng/settings.yml
docker restart searxng
```

**6. 작동 확인:**

```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"results\"])} results')"
```

`10 results`와 비슷한 내용을 확인해야 합니다. `403 Forbidden`이 나오면 JSON 형식이 여전히 비활성화된 것입니다 — 4단계를 다시 확인하세요.

**7. Hermes 구성:**

```bash
# ~/.hermes/.env
SEARXNG_URL=http://localhost:8888
```

그런 다음 `~/.hermes/config.yaml`에서 검색 백엔드로 SearXNG를 선택합니다:

```yaml
web:
  search_backend: "searxng"
```

또는 `hermes tools` → Web Search & Extract → SearXNG를 통해 설정합니다.

---

#### 옵션 B — 공개 인스턴스 사용

공개 SearXNG 인스턴스는 [searx.space](https://searx.space/)에 나열되어 있습니다. **JSON 형식이 활성화된** 인스턴스로 필터링하세요(표에 표시됨).

```bash
# ~/.hermes/.env
SEARXNG_URL=https://searx.example.com
```

:::caution 공개 인스턴스
공개 인스턴스에는 속도 제한이 있고 가동 시간이 가변적이며 언제든지 JSON 형식을 비활성화할 수 있습니다. 프로덕션 용도로는 자체 호스팅을 강력히 권장합니다.
:::

---

#### SearXNG와 추출 제공업체 결합

SearXNG는 검색을 처리하므로 `web_extract`에는 별도의 제공업체가 필요합니다. 기능별 키를 사용하세요:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # 또는 tavily, exa, parallel
```

이 구성을 사용하면 Hermes는 모든 검색 쿼리에 SearXNG를 사용하고 URL 추출에 Firecrawl을 사용하여 무료 검색과 고품질 추출을 결합합니다.

---

### Tavily

관대한 무료 티어가 포함된 AI 최적화 검색 및 추출.

```bash
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

[app.tavily.com](https://app.tavily.com/home)에서 키를 얻으세요. 무료 티어에는 월 1,000 검색이 포함됩니다.

---

### Exa

의미론적 이해가 포함된 신경망 검색(Neural search). 개념적으로 관련된 콘텐츠를 연구하고 찾는 데 적합합니다.

```bash
# ~/.hermes/.env
EXA_API_KEY=your-exa-key-here
```

[exa.ai](https://exa.ai)에서 키를 얻으세요. 무료 티어에는 월 1,000 검색이 포함됩니다.

---

### Parallel

깊이 있는 리서치 기능을 갖춘 AI 네이티브 검색 및 추출.

```bash
# ~/.hermes/.env
PARALLEL_API_KEY=your-parallel-key-here
```

[parallel.ai](https://parallel.ai)에서 엑세스 권한을 얻으세요.

---

### xAI (Grok) {#xai-grok}

Responses API의 Grok 서버 측 [web_search 도구](https://docs.x.ai/developers/tools/web-search)를 통해 `web_search`를 라우팅합니다. Grok은 실제 검색을 실행하고 상위 결과를 구조화된 JSON으로 반환합니다.

새로운 환경 변수나 새로운 설정 마법사 없이 두 가지 자격 증명 경로 중 하나와 작동합니다:

```bash
# ~/.hermes/.env (환경 변수 경로)
XAI_API_KEY=sk-xai-your-key-here
```

또는 SuperGrok 구독자의 경우:

```bash
hermes auth login xai-oauth
```

그런 다음 xAI를 검색 백엔드로 선택합니다:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "xai"
```

**선택 사항 (Optional knobs):**

```yaml
web:
  backend: "xai"
  xai:
    model: grok-4.3              # web_search에 필요한 추론 모델 (기본값)
    allowed_domains:             # 선택 사항, 최대 5개 — excluded_domains와 상호 배타적
      - arxiv.org
    excluded_domains:            # 선택 사항, 최대 5개
      - example-spam.com
    timeout: 90                  # 초 (기본값)
```

**검색 전용** — `web_extract`도 필요한 경우 Firecrawl / Tavily / Exa / Parallel과 페어링하세요. 401 오류가 발생하면 공급자는 한 번의 강제 OAuth 토큰 새로 고침 및 재시도를 수행합니다 (사전 대응적 만료 검사에서 디코딩할 수 없는 창 내 취소 및 불투명 토큰 포함); 환경 변수 자격 증명은 재시도를 건너뜁니다.

:::caution 신뢰 모델 (Trust model)
그대로 검색 엔진 결과를 반환하는 인덱스 기반 제공업체(Brave, Tavily, Exa)와 달리 xAI는 노출할 URL을 선택하고 제목과 설명을 직접 작성하는 LLM입니다. 쿼리의 *내용*이 출력에 영향을 미치므로 악의적으로 조작된 쿼리(예: 에이전트가 선택한 신뢰할 수 없는 업스트림 입력을 통해 주입된 경우)는 이론적으로 Grok이 공격자가 선택한 URL을 내보내도록 조종할 수 있습니다. 반환된 URL은 일반적인 모델 생성 링크를 취급할 때와 같은 방식으로 취급하세요 — 특히 쿼리가 신뢰할 수 없는 입력에서 온 경우 가져오기 전에 유효성을 검사하세요.
:::

---

## 구성 (Configuration)

### 단일 백엔드

모든 웹 기능에 하나의 제공업체를 설정합니다:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | xai
```

### 기능별 구성 (Per-capability configuration)

검색과 추출에 대해 서로 다른 제공업체를 사용하세요. 이렇게 하면 무료 검색(SearXNG)을 유료 추출 제공업체와 결합하거나 그 반대로 결합할 수 있습니다:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # web_search에서 사용
  extract_backend: "firecrawl"  # web_extract에서 사용
```

기능별 키가 비어 있으면 두 항목 모두 `web.backend`를 사용합니다. `web.backend`도 비어 있으면 제공된 API 키/URL에 따라 백엔드가 자동 감지됩니다.

**우선순위 순서 (기능별):**
1. `web.search_backend` / `web.extract_backend` (명시적 기능별 설정)
2. `web.backend` (공유 폴백)
3. 환경 변수에서 자동 감지

### 자동 감지 (Auto-detection)

백엔드가 명시적으로 구성되지 않은 경우 Hermes는 설정된 자격 증명에 따라 사용 가능한 첫 번째 백엔드를 선택합니다:

| 존재하는 자격 증명 (Credential present) | 자동 선택된 백엔드 (Auto-selected backend) |
|--------------------|-----------------------|
| `FIRECRAWL_API_KEY` 또는 `FIRECRAWL_API_URL` | firecrawl |
| `PARALLEL_API_KEY` | parallel |
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `SEARXNG_URL` | searxng |

xAI 웹 검색은 자동 감지 체인에 포함되지 **않습니다** — `XAI_API_KEY`가 설정되어 있거나 (또는 xAI Grok OAuth를 통해 로그인되어 있거나) 자동으로 웹 트래픽을 xAI를 통해 라우팅하지 않습니다. 이러한 자격 증명은 추론 / TTS / 이미지 생성에도 사용되며 사용자가 웹용 다른 백엔드를 원할 수 있기 때문입니다. `web.backend: "xai"`로 명시적으로 선택하세요.

---

## 설정 확인 (Verify your setup)

`hermes setup`을 실행하여 어떤 웹 백엔드가 감지되었는지 확인하세요:

```
✅ Web Search & Extract (searxng)
```

또는 CLI를 통해 확인하세요:

```bash
# venv를 활성화하고 웹 도구 모듈을 직접 실행합니다
source ~/.hermes/hermes-agent/.venv/bin/activate
python -m tools.web_tools
```

활성 백엔드 및 해당 상태가 인쇄됩니다:

```
✅ Web backend: searxng
   Using SearXNG (search only): http://localhost:8888
```

---

## 문제 해결 (Troubleshooting)

### `web_search`가 `{"success": false}`를 반환합니다

- `SEARXNG_URL`에 접근할 수 있는지 확인하세요: `curl -s "http://localhost:8888/search?q=test&format=json"`
- HTTP 403 오류가 발생하면 JSON 형식이 비활성화된 것입니다 — `settings.yml`의 `formats` 목록에 `json`을 추가하고 다시 시작하세요.
- 연결 오류가 발생하면 컨테이너가 실행되지 않았을 수 있습니다: `docker ps | grep searxng`

### `web_extract`가 "검색 전용 백엔드(search-only backend)"라고 표시합니다

SearXNG는 URL 콘텐츠를 추출할 수 없습니다. `web.extract_backend`를 추출을 지원하는 제공업체로 설정하세요:

```yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"  # 또는 tavily / exa / parallel
```

### SearXNG가 결과를 0개 반환합니다

일부 공개 인스턴스는 특정 검색 엔진이나 범주를 비활성화합니다. 시도해 볼 사항:
- 다른 쿼리 시도
- [searx.space](https://searx.space/)의 다른 공개 인스턴스 시도
- 신뢰할 수 있는 결과를 위해 자신의 인스턴스를 자체 호스팅

### 공개 인스턴스에서 속도 제한에 걸렸습니다

자체 호스팅 인스턴스로 전환하세요 (위의 [옵션 A](#옵션-a--docker로-자체-호스팅-권장) 참고). Docker를 사용하면 자체 인스턴스에 속도 제한이 없습니다.

### `web_extract`가 "요약 시간 초과(summarization timed out)" 메모와 함께 잘린 콘텐츠를 반환합니다

보조 모델이 구성된 시간 내에 요약을 완료하지 못했습니다. 해결 방법:

- `config.yaml`의 `auxiliary.web_extract.timeout`을 늘립니다 (새로 설치 시 기본값 360초, 키가 누락된 경우 30초)
- `web_extract` 보조 작업을 더 빠른 모델(예: `google/gemini-3-flash-preview`)로 전환합니다 — [`web_extract`가 긴 페이지를 처리하는 방법](#web_extract가-긴-페이지를-처리하는-방법) 참고
- 요약이 적절하지 않은 페이지의 경우 대신 `browser_navigate`를 사용합니다

---

## 선택적 기술: `searxng-search`

웹 도구 세트를 사용할 수 없을 때 폴백(fallback)으로 에이전트가 `curl`을 통해 SearXNG를 직접 사용해야 하는 경우, `searxng-search` 선택적 기술을 설치하세요:

```bash
hermes skills install official/research/searxng-search
```

이렇게 하면 에이전트에게 다음을 알려주는 기술이 추가됩니다:
- `curl` 또는 Python을 통해 SearXNG JSON API 호출
- 카테고리별 필터링 (`general`, `news`, `science` 등)
- 페이지 매기기(pagination) 및 오류 사례 처리
- SearXNG에 연결할 수 없을 때 우아하게 폴백하기
