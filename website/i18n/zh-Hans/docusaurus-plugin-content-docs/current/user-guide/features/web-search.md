---
title: 网页搜索和提取
description: "使用多个后端提供商搜索网页、提取页面内容并抓取网站——包括免费自托管 SearXNG。"
sidebar_label: 网页搜索
sidebar_position: 6
---

# 网页搜索和提取

Hermes Agent 包含三个由多个提供商支持的网页工具：

- **`web_search`** — 搜索网络并返回排名结果
- **`web_extract`** — 从一个或多个 URL 获取并提取可读内容
- **`web_crawl`** — 递归抓取网站并返回结构化内容

所有三个都通过单个后端选择进行配置。通过 `hermes tools` 或直接在 `config.yaml` 中设置来选择提供商。

## 后端

| 提供商 | 环境变量 | 搜索 | 提取 | 抓取 | 免费层级 |
|----------|---------|--------|---------|-------|-----------|
| **Firecrawl**（默认） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ | 500 积分/月 |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — | ✔ 免费（自托管） |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ | 1,000 次搜索/月 |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — | 1,000 次搜索/月 |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — | 付费 |

**每个能力拆分：** 你可以独立使用不同的提供商进行搜索和提取——例如，使用 SearXNG（免费）进行搜索，使用 Firecrawl 进行提取。请参阅下面的[每个能力配置](#per-capability-configuration)。

:::tip Nous 订阅者
如果你有付费的 [Nous Portal](https://portal.nousresearch.com) 订阅，网页搜索和提取可通过托管 Firecrawl 的 **[Tool Gateway](tool-gateway.md)** 获得——无需 API 密钥。运行 `hermes tools` 启用它。
:::

---

## 设置

### 通过 `hermes tools` 快速设置

运行 `hermes tools`，导航到 **Web Search & Extract**，然后选择一个提供商。向导提示输入所需的 URL 或 API 密钥并将其写入你的配置。

```bash
hermes tools
```

---

### Firecrawl（默认）

全功能搜索、提取和抓取。推荐给大多数用户。

```bash
# ~/.hermes/.env
FIRECRAWL_API_KEY=fc-your-key-here
```

在 [firecrawl.dev](https://firecrawl.dev) 获取密钥。免费层级包括每月 500 积分。

**自托管 Firecrawl：** 改为指向你自己的实例而不是云 API：

```bash
# ~/.hermes/.env
FIRECRAWL_API_URL=http://localhost:3002
```

当设置了 `FIRECRAWL_API_URL` 时，API 密钥是可选的（使用 `USE_DB_AUTHENTICATION=false` 禁用服务器身份验证）。

---

### SearXNG（免费、自托管）

SearXNG 是一个尊重隐私的开源元搜索引擎，聚合了 70+ 搜索引擎的结果。**无需 API 密钥**——只需将 Hermes 指向运行的 SearXNG 实例。

SearXNG **仅支持搜索**——`web_extract` 和 `web_crawl` 需要单独的提取提供商。

#### 选项 A — 使用 Docker 自托管（推荐）

这为你提供了一个没有速率限制的私有实例。

**1. 创建工作目录：**

```bash
mkdir -p ~/searxng/searxng
cd ~/searxng
```

**2. 编写 `docker-compose.yml`：**

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

**3. 启动容器：**

```bash
docker compose up -d
```

**4. 启用 JSON API 格式：**

SearXNG 默认禁用 JSON 输出。复制生成的配置并启用它：

```bash
# 从容器中复制自动生成的配置
docker cp searxng:/etc/searxng/settings.yml ~/searxng/searxng/settings.yml
```

打开 `~/searxng/searxng/settings.yml` 并找到 `formats` 块（约第 84 行）：

```yaml
# 之前（默认——JSON 禁用）：
formats:
  - html

# 之后（为 Hermes 启用 JSON）：
formats:
  - html
  - json
```

**5. 重启以应用：**

```bash
docker cp ~/searxng/searxng/settings.yml searxng:/etc/searxng/settings.yml
docker restart searxng
```

**6. 验证它是否正常工作：**

```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"results\"])} results')"
```

你应该看到类似 `10 results` 的内容。如果你得到 `403 Forbidden`，JSON 格式仍然被禁用——重新检查步骤 4。

**7. 配置 Hermes：**

```bash
# ~/.hermes/config.yaml
SEARXNG_URL: http://localhost:8888
```

或通过 `hermes tools` → Web Search & Extract → SearXNG 设置。

---

#### 选项 B — 使用公共实例

公共 SearXNG 实例列在 [searx.space](https://searx.space/)。过滤显示**已启用 JSON 格式**的实例（在表格中显示）。

```bash
# ~/.hermes/config.yaml
SEARXNG_URL: https://searx.example.com
```

:::caution 公共实例
公共实例有速率限制、可变正常运行时间，可能会随时禁用 JSON 格式。对于生产使用，强烈建议自托管。
:::

---

#### 将 SearXNG 与提取提供商配对

SearXNG 处理搜索；你需要单独的提供商用于 `web_extract` 和 `web_crawl`。使用每个能力键：

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # 或 tavily、exa、parallel
```

使用此配置，Hermes 对所有搜索查询使用 SearXNG，对 URL 提取使用 Firecrawl——将免费搜索与高质量提取相结合。

---

### Tavily

AI 优化的搜索、提取和抓取，提供慷慨的免费层级。

```bash
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

在 [app.tavily.com](https://app.tavily.com/home) 获取密钥。免费层级包括每月 1,000 次搜索。

---

### Exa

具有语义理解的神经搜索。适合研究和查找概念相关的内容。

```bash
# ~/.hermes/.env
EXA_API_KEY=your-exa-key-here
```

在 [exa.ai](https://exa.ai) 获取密钥。免费层级包括每月 1,000 次搜索。

---

### Parallel

具有深度研究能力的 AI 原生搜索和提取。

```bash
# ~/.hermes/.env
PARALLEL_API_KEY=your-parallel-key-here
```

在 [parallel.ai](https://parallel.ai) 获取访问权限。

---

## 配置

### 单个后端

为所有网页能力设置一个提供商：

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | tavily | exa | parallel
```

### 每个能力的配置

对搜索和提取使用不同的提供商。这让你可以结合免费搜索（SearXNG）与付费提取提供商，反之亦然：

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # web_search 使用
  extract_backend: "firecrawl"  # web_extract 和 web_crawl 使用
```

当每个能力键为空时，两者都回退到 `web.backend`。当 `web.backend` 也为空时，后端从存在的任何 API 密钥/URL 自动检测。

**优先级顺序（按能力）：**
1. `web.search_backend` / `web.extract_backend`（显式每个能力）
2. `web.backend`（共享回退）
3. 从环境变量自动检测

### 自动检测

如果没有明确配置后端，Hermes 根据设置的凭据选择第一个可用的：

| 存在的凭据 | 自动选择的后端 |
|--------------------|-----------------------|
| `FIRECRAWL_API_KEY` 或 `FIRECRAWL_API_URL` | firecrawl |
| `PARALLEL_API_KEY` | parallel |
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `SEARXNG_URL` | searxng |

---

## 验证你的设置

运行 `hermes setup` 查看检测到的网页后端：

```
✅ Web Search & Extract (searxng)
```

或通过 CLI 检查：

```bash
# 激活 venv 并直接运行网页工具模块
source ~/.hermes/hermes-agent/.venv/bin/activate
python -m tools.web_tools
```

这打印活动后端及其状态：

```
✅ Web backend: searxng
   Using SearXNG (search only): http://localhost:8888
```

---

## 故障排除

### `web_search` 返回 `{"success": false}`

- 检查 `SEARXNG_URL` 是否可访问：`curl -s "http://localhost:8888/search?q=test&format=json"`
- 如果得到 HTTP 403，JSON 格式被禁用——在 `settings.yml` 的 `formats` 列表中添加 `json` 并重启
- 如果得到连接错误，容器可能未运行：`docker ps | grep searxng`

### `web_extract` 说"search-only backend"

SearXNG 无法提取 URL 内容。将 `web.extract_backend` 设置为支持提取的提供商：

```yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"  # 或 tavily / exa / parallel
```

### SearXNG 返回 0 结果

一些公共实例禁用某些搜索引擎或类别。尝试：
- 不同的查询
- [searx.space](https://searx.space/) 上的不同公共实例
- 自托管你自己的实例以获得可靠结果

### 公共实例上被速率限制

切换到自托管实例（请参阅上面的[选项 A](#option-a--self-host-with-docker-recommended)）。使用 Docker，你自己的实例没有速率限制。

---

## 可选技能：`searxng-search`

对于需要通过 `curl` 直接使用 SearXNG 的 agent（例如，当网页工具集不可用时作为回退），安装 `searxng-search` 可选技能：

```bash
hermes skills install official/research/searxng-search
```

这添加了一个教 agent 如何的技能：
- 通过 `curl` 或 Python 调用 SearXNG JSON API
- 按类别过滤（`general`、`news`、`science` 等）
- 处理分页和错误情况
- 在 SearXNG不可访问时优雅回退
