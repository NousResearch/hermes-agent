---
title: "Scrapling — 스텔스 브라우징 및 Cloudflare 우회로 사이트 스크래핑"
sidebar_label: "Scrapling"
description: "스텔스 브라우징 및 Cloudflare 우회로 사이트를 스크래핑합니다"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Scrapling

스텔스 브라우징 및 Cloudflare 우회를 지원하는 사이트 스크래핑.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/research/scrapling`으로 설치 |
| 경로 | `optional-skills/research/scrapling` |
| 버전 | `1.0.0` |
| 작성자 | FEUAZUR |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Web Scraping`, `Browser`, `Cloudflare`, `Stealth`, `Crawling`, `Spider` |
| 관련 스킬 | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 불러오는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보게 되는 내용입니다.
:::

# Scrapling

[Scrapling](https://github.com/D4Vinci/Scrapling)은 안티봇 우회, 스텔스 브라우저 자동화 및 스파이더 프레임워크를 제공하는 웹 스크래핑 프레임워크입니다. HTTP, 동적 JS, 스텔스/Cloudflare의 세 가지 가져오기 전략과 전체 CLI를 제공합니다.

**이 스킬은 교육 및 연구 목적으로만 사용해야 합니다.** 사용자는 현지 및 국제 데이터 스크래핑 법률을 준수하고 웹사이트 서비스 약관을 존중해야 합니다.

## 사용 시점

- 정적 HTML 페이지 스크래핑(브라우저 도구보다 빠름)
- 실제 브라우저가 필요한 JS 렌더링 페이지 스크래핑
- Cloudflare Turnstile 또는 봇 감지 우회
- 스파이더로 여러 페이지 크롤링
- 기본 제공 `web_extract` 도구가 필요한 데이터를 반환하지 않을 때

## 설치

```bash
pip install "scrapling[all]"
scrapling install
```

최소 설치(브라우저 없이 HTTP만 사용):
```bash
pip install scrapling
```

브라우저 자동화만 사용:
```bash
pip install "scrapling[fetchers]"
scrapling install
```

## 빠른 참고

| 접근 방식 | 클래스 | 사용 시점 |
|----------|-------|----------|
| HTTP | `Fetcher` / `FetcherSession` | 정적 페이지, API, 빠른 대량 요청 |
| Dynamic | `DynamicFetcher` / `DynamicSession` | JS 렌더링 콘텐츠, SPA |
| Stealth | `StealthyFetcher` / `StealthySession` | Cloudflare, 안티봇 보호 사이트 |
| Spider | `Spider` | 링크를 따라가는 여러 페이지 크롤링 |

## CLI 사용법

### 정적 페이지 추출

```bash
scrapling extract get 'https://example.com' output.md
```

CSS 선택자와 브라우저 임퍼서네이션 사용:

```bash
scrapling extract get 'https://example.com' output.md \
  --css-selector '.content' \
  --impersonate 'chrome'
```

### JS 렌더링 페이지 추출

```bash
scrapling extract fetch 'https://example.com' output.md \
  --css-selector '.dynamic-content' \
  --disable-resources \
  --network-idle
```

### Cloudflare 보호 페이지 추출

```bash
scrapling extract stealthy-fetch 'https://protected-site.com' output.html \
  --solve-cloudflare \
  --block-webrtc \
  --hide-canvas
```

### POST 요청

```bash
scrapling extract post 'https://example.com/api' output.json \
  --json '{"query": "search term"}'
```

### 출력 형식

출력 형식은 파일 확장자로 결정됩니다.
- `.html` -- 원시 HTML
- `.md` -- Markdown으로 변환
- `.txt` -- 일반 텍스트
- `.json` / `.jsonl` -- JSON

## Python: HTTP 스크래핑

### 단일 요청

```python
from scrapling.fetchers import Fetcher

page = Fetcher.get('https://quotes.toscrape.com/')
quotes = page.css('.quote .text::text').getall()
for q in quotes:
    print(q)
```

### 세션(지속적인 쿠키)

```python
from scrapling.fetchers import FetcherSession

with FetcherSession(impersonate='chrome') as session:
    page = session.get('https://example.com/', stealthy_headers=True)
    links = page.css('a::attr(href)').getall()
    for link in links[:5]:
        sub = session.get(link)
        print(sub.css('h1::text').get())
```

### POST / PUT / DELETE

```python
page = Fetcher.post('https://api.example.com/data', json={"key": "value"})
page = Fetcher.put('https://api.example.com/item/1', data={"name": "updated"})
page = Fetcher.delete('https://api.example.com/item/1')
```

### 프록시 사용

```python
page = Fetcher.get('https://example.com', proxy='http://user:pass@proxy:8080')
```

## Python: 동적 페이지(JS 렌더링)

JavaScript 실행이 필요한 페이지(SPA, 지연 로드 콘텐츠)의 경우:

```python
from scrapling.fetchers import DynamicFetcher

page = DynamicFetcher.fetch('https://example.com', headless=True)
data = page.css('.js-loaded-content::text').getall()
```

### 특정 요소가 나타날 때까지 대기

```python
page = DynamicFetcher.fetch(
    'https://example.com',
    wait_selector=('.results', 'visible'),
    network_idle=True,
)
```

### 속도 향상을 위해 리소스 비활성화

글꼴, 이미지, 미디어, 스타일시트를 차단합니다(약 25% 더 빠름).

```python
from scrapling.fetchers import DynamicSession

with DynamicSession(headless=True, disable_resources=True, network_idle=True) as session:
    page = session.fetch('https://example.com')
    items = page.css('.item::text').getall()
```

### 사용자 지정 페이지 자동화

```python
from playwright.sync_api import Page
from scrapling.fetchers import DynamicFetcher

def scroll_and_click(page: Page):
    page.mouse.wheel(0, 3000)
    page.wait_for_timeout(1000)
    page.click('button.load-more')
    page.wait_for_selector('.extra-results')

page = DynamicFetcher.fetch('https://example.com', page_action=scroll_and_click)
results = page.css('.extra-results .item::text').getall()
```

## Python: 스텔스 모드(안티봇 우회)

Cloudflare로 보호되거나 지문이 강하게 감지되는 사이트의 경우:

```python
from scrapling.fetchers import StealthyFetcher

page = StealthyFetcher.fetch(
    'https://protected-site.com',
    headless=True,
    solve_cloudflare=True,
    block_webrtc=True,
    hide_canvas=True,
)
content = page.css('.protected-content::text').getall()
```

### 스텔스 세션

```python
from scrapling.fetchers import StealthySession

with StealthySession(headless=True, solve_cloudflare=True) as session:
    page1 = session.fetch('https://protected-site.com/page1')
    page2 = session.fetch('https://protected-site.com/page2')
```

## 요소 선택

모든 fetcher는 다음 메서드를 가진 `Selector` 객체를 반환합니다.

### CSS 선택자

```python
page.css('h1::text').get()              # First h1 text
page.css('a::attr(href)').getall()      # All link hrefs
page.css('.quote .text::text').getall() # Nested selection
```

### XPath

```python
page.xpath('//div[@class="content"]/text()').getall()
page.xpath('//a/@href').getall()
```

### 찾기 메서드

```python
page.find_all('div', class_='quote')       # By tag + attribute
page.find_by_text('Read more', tag='a')    # By text content
page.find_by_regex(r'\$\d+\.\d{2}')       # By regex pattern
```

### 유사 요소

구조가 유사한 요소를 찾습니다(제품 목록 등에 유용).

```python
first_product = page.css('.product')[0]
all_similar = first_product.find_similar()
```

### 탐색

```python
el = page.css('.target')[0]
el.parent                # Parent element
el.children              # Child elements
el.next_sibling          # Next sibling
el.prev_sibling          # Previous sibling
```

## Python: 스파이더 프레임워크

링크를 따라 여러 페이지를 크롤링하는 경우:

```python
from scrapling.spiders import Spider, Request, Response

class QuotesSpider(Spider):
    name = "quotes"
    start_urls = ["https://quotes.toscrape.com/"]
    concurrent_requests = 10
    download_delay = 1

    async def parse(self, response: Response):
        for quote in response.css('.quote'):
            yield {
                "text": quote.css('.text::text').get(),
                "author": quote.css('.author::text').get(),
                "tags": quote.css('.tag::text').getall(),
            }

        next_page = response.css('.next a::attr(href)').get()
        if next_page:
            yield response.follow(next_page)

result = QuotesSpider().start()
print(f"Scraped {len(result.items)} quotes")
result.items.to_json("quotes.json")
```

### 다중 세션 스파이더

요청을 서로 다른 fetcher 유형으로 라우팅합니다.

```python
from scrapling.fetchers import FetcherSession, AsyncStealthySession

class SmartSpider(Spider):
    name = "smart"
    start_urls = ["https://example.com/"]

    def configure_sessions(self, manager):
        manager.add("fast", FetcherSession(impersonate="chrome"))
        manager.add("stealth", AsyncStealthySession(headless=True), lazy=True)

    async def parse(self, response: Response):
        for link in response.css('a::attr(href)').getall():
            if "protected" in link:
                yield Request(link, sid="stealth")
            else:
                yield Request(link, sid="fast", callback=self.parse)
```

### 크롤링 일시 중지/재개

```python
spider = QuotesSpider(crawldir="./crawl_checkpoint")
spider.start()  # Ctrl+C to pause, re-run to resume from checkpoint
```

## 주의 사항

- **브라우저 설치 필요**: pip install 후 `scrapling install`을 실행해야 합니다. 실행하지 않으면 `DynamicFetcher` 및 `StealthyFetcher`가 실패합니다
- **시간 초과**: DynamicFetcher/StealthyFetcher의 시간 초과 단위는 **밀리초**(기본값 30000)이고, Fetcher의 단위는 **초**입니다
- **Cloudflare 우회**: `solve_cloudflare=True`를 사용하면 가져오기 시간에 5~15초가 추가됩니다. 필요한 경우에만 활성화하세요
- **리소스 사용량**: StealthyFetcher는 실제 브라우저를 실행하므로 동시 사용량을 제한하세요
- **법적 사항**: 스크래핑 전에 항상 robots.txt와 웹사이트 ToS를 확인하세요. 이 라이브러리는 교육 및 연구 목적으로 사용됩니다
- **Python 버전**: Python 3.10 이상이 필요합니다
