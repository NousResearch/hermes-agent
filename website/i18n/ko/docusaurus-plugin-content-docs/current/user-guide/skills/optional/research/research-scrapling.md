---
title: "Scrapling"
sidebar_label: "Scrapling"
description: "Scrapling을 사용한 웹 스크래핑 - CLI와 Python을 통한 HTTP 가져오기, 스텔스 브라우저 자동화, Cloudflare 우회 및 스파이더 크롤링"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Scrapling

Scrapling을 사용한 웹 스크래핑 - CLI와 Python을 통한 HTTP 가져오기, 스텔스 브라우저 자동화, Cloudflare 우회 및 스파이더 크롤링을 지원합니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/scrapling` |
| Path | `optional-skills/research/scrapling` |
| Version | `1.0.0` |
| Author | FEUAZUR |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Web Scraping`, `Browser`, `Cloudflare`, `Stealth`, `Crawling`, `Spider` |
| Related skills | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel) |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Scrapling

[Scrapling](https://github.com/D4Vinci/Scrapling)은 안티 봇(anti-bot) 우회, 스텔스 브라우저 자동화 및 스파이더 프레임워크를 갖춘 웹 스크래핑 프레임워크입니다. 세 가지 가져오기 전략(HTTP, 동적 JS, 스텔스/Cloudflare)과 전체 CLI를 제공합니다.

**이 스킬은 교육 및 연구 목적으로만 제공됩니다.** 사용자는 데이터 스크래핑에 관한 현지/국제 법률을 준수해야 하며 웹사이트 서비스 약관(Terms of Service)을 존중해야 합니다.

## When to Use

- 정적 HTML 페이지 스크래핑 (브라우저 도구보다 빠름)
- 실제 브라우저가 필요한 JS 렌더링 페이지 스크래핑
- Cloudflare Turnstile 또는 봇 탐지 우회
- 스파이더를 사용하여 여러 페이지 크롤링
- 내장된 `web_extract` 도구가 필요한 데이터를 반환하지 않을 때

## Installation

```bash
pip install "scrapling[all]"
scrapling install
```

최소 설치 (HTTP만 가능, 브라우저 없음):
```bash
pip install scrapling
```

브라우저 자동화만 필요한 경우:
```bash
pip install "scrapling[fetchers]"
scrapling install
```

## Quick Reference

| Approach | Class | Use When |
|----------|-------|----------|
| HTTP | `Fetcher` / `FetcherSession` | 정적 페이지, API, 빠른 대량 요청 |
| Dynamic | `DynamicFetcher` / `DynamicSession` | JS 렌더링 콘텐츠, SPA (단일 페이지 애플리케이션) |
| Stealth | `StealthyFetcher` / `StealthySession` | Cloudflare, 안티 봇 보호 사이트 |
| Spider | `Spider` | 링크를 따라가는 다중 페이지 크롤링 |

## CLI Usage

### Extract Static Page

```bash
scrapling extract get 'https://example.com' output.md
```

CSS 선택자 및 브라우저 가장 사용:

```bash
scrapling extract get 'https://example.com' output.md \
  --css-selector '.content' \
  --impersonate 'chrome'
```

### Extract JS-Rendered Page

```bash
scrapling extract fetch 'https://example.com' output.md \
  --css-selector '.dynamic-content' \
  --disable-resources \
  --network-idle
```

### Extract Cloudflare-Protected Page

```bash
scrapling extract stealthy-fetch 'https://protected-site.com' output.html \
  --solve-cloudflare \
  --block-webrtc \
  --hide-canvas
```

### POST Request

```bash
scrapling extract post 'https://example.com/api' output.json \
  --json '{"query": "search term"}'
```

### Output Formats

출력 형식은 파일 확장자에 의해 결정됩니다:
- `.html` -- 원시 HTML
- `.md` -- Markdown으로 변환
- `.txt` -- 일반 텍스트
- `.json` / `.jsonl` -- JSON

## Python: HTTP Scraping

### Single Request

```python
from scrapling.fetchers import Fetcher

page = Fetcher.get('https://quotes.toscrape.com/')
quotes = page.css('.quote .text::text').getall()
for q in quotes:
    print(q)
```

### Session (Persistent Cookies)

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

### With Proxy

```python
page = Fetcher.get('https://example.com', proxy='http://user:pass@proxy:8080')
```

## Python: Dynamic Pages (JS-Rendered)

JavaScript 실행이 필요한 페이지 (SPA, 지연 로드되는 콘텐츠)의 경우:

```python
from scrapling.fetchers import DynamicFetcher

page = DynamicFetcher.fetch('https://example.com', headless=True)
data = page.css('.js-loaded-content::text').getall()
```

### Wait for Specific Element

```python
page = DynamicFetcher.fetch(
    'https://example.com',
    wait_selector=('.results', 'visible'),
    network_idle=True,
)
```

### Disable Resources for Speed

글꼴, 이미지, 미디어, 스타일시트를 차단합니다 (약 25% 빠름):

```python
from scrapling.fetchers import DynamicSession

with DynamicSession(headless=True, disable_resources=True, network_idle=True) as session:
    page = session.fetch('https://example.com')
    items = page.css('.item::text').getall()
```

### Custom Page Automation

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

## Python: Stealth Mode (Anti-Bot Bypass)

Cloudflare로 보호되거나 지문 추적이 심한 사이트의 경우:

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

### Stealth Session

```python
from scrapling.fetchers import StealthySession

with StealthySession(headless=True, solve_cloudflare=True) as session:
    page1 = session.fetch('https://protected-site.com/page1')
    page2 = session.fetch('https://protected-site.com/page2')
```

## Element Selection

모든 가져오기(Fetcher)는 다음 메서드를 가진 `Selector` 객체를 반환합니다:

### CSS Selectors

```python
page.css('h1::text').get()              # 첫 번째 h1 텍스트
page.css('a::attr(href)').getall()      # 모든 링크 href
page.css('.quote .text::text').getall() # 중첩된 선택
```

### XPath

```python
page.xpath('//div[@class="content"]/text()').getall()
page.xpath('//a/@href').getall()
```

### Find Methods

```python
page.find_all('div', class_='quote')       # 태그 + 속성으로 검색
page.find_by_text('Read more', tag='a')    # 텍스트 내용으로 검색
page.find_by_regex(r'\$\d+\.\d{2}')       # 정규식 패턴으로 검색
```

### Similar Elements

구조가 비슷한 요소를 찾습니다 (제품 목록 등에 유용함):

```python
first_product = page.css('.product')[0]
all_similar = first_product.find_similar()
```

### Navigation

```python
el = page.css('.target')[0]
el.parent                # 부모 요소
el.children              # 자식 요소
el.next_sibling          # 다음 형제
el.prev_sibling          # 이전 형제
```

## Python: Spider Framework

링크를 따라가는 다중 페이지 크롤링의 경우:

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

### Multi-Session Spider

서로 다른 fetcher 유형으로 요청 라우팅:

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

### Pause/Resume Crawling

```python
spider = QuotesSpider(crawldir="./crawl_checkpoint")
spider.start()  # Ctrl+C로 일시 중지, 다시 실행하면 체크포인트부터 다시 시작
```

## Pitfalls

- **브라우저 설치 필수**: pip install 후 `scrapling install`을 실행하세요 -- 그렇지 않으면 `DynamicFetcher` 및 `StealthyFetcher`가 실패합니다.
- **타임아웃(Timeouts)**: DynamicFetcher/StealthyFetcher 타임아웃은 **밀리초(ms)** 단위 (기본값 30000)이며, Fetcher 타임아웃은 **초(s)** 단위입니다.
- **Cloudflare 우회**: `solve_cloudflare=True`는 가져오는 시간에 5-15초를 추가합니다 -- 필요한 경우에만 활성화하세요.
- **리소스 사용량**: StealthyFetcher는 실제 브라우저를 실행하므로 동시 사용량을 제한하세요.
- **법적 고지**: 스크래핑 전 항상 robots.txt 및 웹사이트 ToS를 확인하세요. 이 라이브러리는 교육 및 연구 목적으로 제공됩니다.
- **Python 버전**: Python 3.10 이상이 필요합니다.
