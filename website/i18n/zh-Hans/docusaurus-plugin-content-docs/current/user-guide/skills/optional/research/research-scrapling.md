---
title: "Scrapling"
sidebar_label: "Scrapling"
description: "使用 Scrapling 进行网页抓取 — HTTP 获取、隐身浏览器自动化、Cloudflare 绕过和蜘蛛爬行，通过 CLI 和 Python"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Scrapling

使用 Scrapling 进行网页抓取 — HTTP 获取、隐身浏览器自动化、Cloudflare 绕过和蜘蛛爬行，通过 CLI 和 Python。

## 技能元数据

| | |
|---|---|
| 来源 | 可选技能 — 使用 `hermes skills install official/research/scrapling` 安装 |
| 路径 | `optional-skills/research/scrapling` |
| 版本 | `1.0.0` |
| 作者 | FEUAZUR |
| 许可证 | MIT |
| 标签 | `Web Scraping`, `Browser`, `Cloudflare`, `Stealth`, `Crawling`, `Spider` |
| 相关技能 | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel) |

## 参考：完整的 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时智能体看到的指令。
:::

# Scrapling

[Scrapling](https://github.com/D4Vinci/Scrapling) 是一个网页抓取框架，具有反机器人绕过、隐身浏览器自动化和蜘蛛框架。它提供三种获取策略（HTTP、动态 JS、隐身/Cloudflare）和完整的 CLI。

**此技能仅用于教育和研究目的。** 用户必须遵守当地/国际数据抓取法律并尊重网站服务条款。

## 何时使用

- 抓取静态 HTML 页面（比浏览器工具更快）
- 抓取需要真正浏览器的 JS 渲染页面
- 绕过 Cloudflare Turnstile 或机器人检测
- 使用蜘蛛爬行多个页面
- 当内置 `web_extract` 工具未返回您需要的数据时

## 安装

```bash
pip install "scrapling[all]"
scrapling install
```

最小安装（仅 HTTP，无浏览器）：
```bash
pip install scrapling
```

仅带浏览器自动化：
```bash
pip install "scrapling[fetchers]"
scrapling install
```

## 快速参考

| 方法 | 类 | 使用时机 |
|----------|-------|----------|
| HTTP | `Fetcher` / `FetcherSession` | 静态页面、API、快速批量请求 |
| 动态 | `DynamicFetcher` / `DynamicSession` | JS 渲染内容、SPA |
| 隐身 | `StealthyFetcher` / `StealthySession` | Cloudflare、受反机器人保护的站点 |
| 蜘蛛 | `Spider` | 带链接跟随的多页面爬行 |

## CLI 使用

### 提取静态页面

```bash
scrapling extract get 'https://example.com' output.md
```

使用 CSS 选择器和浏览器模拟：

```bash
scrapling extract get 'https://example.com' output.md \
  --css-selector '.content' \
  --impersonate 'chrome'
```

### 提取 JS 渲染页面

```bash
scrapling extract fetch 'https://example.com' output.md \
  --css-selector '.dynamic-content' \
  --disable-resources \
  --network-idle
```

### 提取 Cloudflare 保护页面

```bash
scrapling extract stealthy-fetch 'https://protected-site.com' output.html \
  --solve-cloudflare \
  --block-webrtc \
  --hide-canvas
```

### POST 请求

```bash
scrapling extract post 'https://example.com/api' output.json \
  --json '{"query": "search term"}'
```

### 输出格式

输出格式由文件扩展名决定：
- `.html` -- 原始 HTML
- `.md` -- 转换为 Markdown
- `.txt` -- 纯文本
- `.json` / `.jsonl` -- JSON

## Python：HTTP 抓取

### 单个请求

```python
from scrapling.fetchers import Fetcher

page = Fetcher.get('https://quotes.toscrape.com/')
quotes = page.css('.quote .text::text').getall()
for q in quotes:
    print(q)
```

### 会话（持久 Cookie）

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

### 带代理

```python
page = Fetcher.get('https://example.com', proxy='http://user:pass@proxy:8080')
```

## Python：动态页面（JS 渲染）

对于需要 JavaScript 执行的页面（SPA、延迟加载内容）：

```python
from scrapling.fetchers import DynamicFetcher

page = DynamicFetcher.fetch('https://example.com', headless=True)
data = page.css('.js-loaded-content::text').getall()
```

### 等待特定元素

```python
page = DynamicFetcher.fetch(
    'https://example.com',
    wait_selector=('.results', 'visible'),
    network_idle=True,
)
```

### 为速度禁用资源

阻止字体、图片、媒体、样式表（约 25% 加速）：

```python
from scrapling.fetchers import DynamicSession

with DynamicSession(headless=True, disable_resources=True, network_idle=True) as session:
    page = session.fetch('https://example.com')
    items = page.css('.item::text').getall()
```

### 自定义页面自动化

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

## Python：隐身模式（反机器人绕过）

对于 Cloudflare 保护或重度指纹识别的站点：

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

### 隐身会话

```python
from scrapling.fetchers import StealthySession

with StealthySession(headless=True, solve_cloudflare=True) as session:
    page1 = session.fetch('https://protected-site.com/page1')
    page2 = session.fetch('https://protected-site.com/page2')
```

## 元素选择

所有 fetcher 返回带有这些方法的 `Selector` 对象：

### CSS 选择器

```python
page.css('h1::text').get()              # 第一个 h1 文本
page.css('a::attr(href)').getall()      # 所有链接 href
page.css('.quote .text::text').getall() # 嵌套选择
```

### XPath

```python
page.xpath('//div[@class="content"]/text()').getall()
page.xpath('//a/@href').getall()
```

### 查找方法

```python
page.find_all('div', class_='quote')       # 按标签 + 属性
page.find_by_text('Read more', tag='a')    # 按文本内容
page.find_by_regex(r'\$\d+\.\d{2}')       # 按正则模式
```

### 相似元素

查找结构相似的元素（适用于产品列表等）：

```python
first_product = page.css('.product')[0]
all_similar = first_product.find_similar()
```

### 导航

```python
el = page.css('.target')[0]
el.parent                # 父元素
el.children              # 子元素
el.next_sibling          # 下一个兄弟
el.prev_sibling          # 上一个兄弟
```

## Python：蜘蛛框架

用于带链接跟随的多页面爬行：

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

### 多会话蜘蛛

将请求路由到不同的 fetcher 类型：

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

### 暂停/恢复爬行

```python
spider = QuotesSpider(crawldir="./crawl_checkpoint")
spider.start()  # Ctrl+C 暂停，重新运行从检查点恢复
```

## 陷阱

- **需要安装浏览器**：pip install 后运行 `scrapling install` — 没有它，`DynamicFetcher` 和 `StealthyFetcher` 会失败
- **超时**：DynamicFetcher/StealthyFetcher 超时是**毫秒**（默认 30000），Fetcher 超时是**秒**
- **Cloudflare 绕过**：`solve_cloudflare=True` 使获取时间增加 5-15 秒 — 仅在需要时启用
- **资源使用**：StealthyFetcher 运行真正的浏览器 — 限制并发使用
- **法律**：抓取前始终检查 robots.txt 和网站服务条款。此库用于教育和研究目的
- **Python 版本**：需要 Python 3.10+
