---
title: 网页搜索与提取
description: 搜索网页、提取页面内容，并使用多个后端提供商抓取网站——包括免费的自托管 SearXNG。
sidebar_label: 网页搜索
sidebar_position: 6
---

# 网页搜索与提取

<a id="option-a--self-host-with-docker-recommended"></a>

## 选项 A — 推荐：使用 Docker 自托管

如果你希望获得更好的可控性与成本表现，建议优先采用 Docker 自托管搜索后端，并按需与托管提取服务混合使用。

Hermes Agent 包含两个模型可调用的网页工具，由多个提供商支持：

- **`web_search`** —— 搜索网页并返回排名结果
- **`web_extract`** —— 从一个或多个 URL 获取并提取可读内容（当后端提供时内置深度抓取支持）

两者都通过单一后端选择进行配置。提供商通过 `hermes tools` 选择，或直接在 `config.yaml` 中设置。递归抓取功能（Firecrawl/Tavily）通过 `web_extract` 暴露，而非作为单独的 `web_crawl` 工具。

## 后端

| 提供商 | 环境变量 | 搜索 | 提取 | 抓取 | 免费套餐 |
|----------|---------|--------|---------|-------|-----------|
| **Firecrawl** (默认) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ | 500 积分/月 |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — | ✔ 免费 (自托管) |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ | 1 000 次搜索/月 |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — | 1 000 次搜索/月 |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — | 付费 |

**按能力拆分：** 您可以为搜索和提取使用不同的提供商——例如 SearXNG（免费）用于搜索，Firecrawl 用于提取。请参阅下面的[按能力配置](#按能力配置)。

:::tip Nous 订阅者
如果您有付费的 [Nous Portal](https://portal.nousresearch.com) 订阅，网页搜索和提取可通过 **[工具网关](tool-gateway.md)** 使用托管的 Firecrawl——无需 API 密钥。运行 `hermes tools` 启用它。
:::

---

## `web_extract` 如何处理长页面

后端返回原始页面 markdown，这可能非常庞大（论坛线程、文档站点、带嵌入式评论的新闻文章）。为了让您的上下文窗口可用并降低成本，`web_extract` 在交给智能体之前通过 **`web_extract` 辅助模型**运行返回的内容。行为纯粹由大小驱动：

| 页面大小（字符） | 会发生什么 |
|------------------------|--------------|
| 低于 5 000 | 原样返回——无 LLM 调用，完整 markdown 到达智能体 |
| 5 000 – 500 000 | 通过 `web_extract` 辅助模型单遍摘要，输出上限约 5 000 字符 |
| 500 000 – 2 000 000 | 分块：拆分为 100k 字符块，并行摘要每个块，然后合成最终摘要（约 5 000 字符） |
| 超过 2 000 000 | 拒绝并提示使用带聚焦提取指令的 `web_crawl` 或更具体的来源 |

摘要保留原始格式的引用、代码块和关键事实——它是内容压缩器，而非转述器。如果摘要失败或超时，Hermes 回退到原始内容的前 ~5 000 字符，而非无用的错误。

### 哪个模型进行摘要？

`web_extract` 辅助任务。默认情况下（`auxiliary.web_extract.provider: "auto"`），这是您的**主聊天模型**——与 `hermes model` 相同的提供商和模型。这对大多数设置没问题，但在昂贵的推理模型（Opus、MiniMax M2.7 等）上，每个长页面提取都会增加有意义的成本。

要将提取摘要路由到便宜、快速的模型，无论您的主模型如何：

```yaml
# ~/.hermes/config.yaml
auxiliary:
  web_extract:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 360       # 秒；如果遇到摘要超时则提高
```

或交互式选择：`hermes model` → **配置辅助模型** → `web_extract`。

请参阅 [辅助模型](/user-guide/configuration#辅助模型) 获取完整参考和每任务覆盖模式。

### 当摘要碍事时

如果您特别需要原始、未摘要的页面内容——例如，您正在抓取结构化页面，其中 LLM 摘要会丢弃重要字段——请改用 `browser_navigate` + `browser_snapshot`。浏览器工具返回实时可访问性树，无需辅助模型重写（受自身 8 000 字符快照上限限制）。

---

## 设置

### 通过 `hermes tools` 快速设置

运行 `hermes tools`，导航到 **网页搜索与提取**，然后选择一个提供商。向导会提示所需的 URL 或 API 密钥并将其写入您的配置。

```bash
hermes tools
```

---

### Firecrawl (默认)

功能齐全的搜索、提取和抓取。推荐给大多数用户。

```bash
# ~/.hermes/.env
FIRECRAWL_API_KEY=fc-your-key-here
```

在 [firecrawl.dev](https://firecrawl.dev) 获取密钥。免费套餐包括 500 积分/月。

**自托管 Firecrawl：** 指向您自己的实例而非云 API：

```bash
# ~/.hermes/.env
FIRECRAWL_API_URL=http://localhost:3002
```

当设置 `FIRECRAWL_API_URL` 时，API 密钥是可选的（使用 `USE_DB_AUTHENTICATION=false` 禁用服务器身份验证）。

---

### SearXNG (免费, 自托管)

SearXNG 是一个尊重隐私、开源的元搜索引擎，聚合来自 70+ 搜索引擎的结果。**无需 API 密钥**——只需将 Hermes 指向一个运行的 SearXNG 实例。

SearXNG **仅搜索**——`web_extract`（包括其抓取模式）需要单独的提取提供商。

#### 选项 A —— 使用 Docker 自托管（推荐）

这为您提供一个无速率限制的私有实例。

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
# 从容器复制自动生成的配置
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

**6. 验证它是否工作：**

```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"results\"])} results')"
```

您应该看到类似 `10 results` 的内容。如果您得到 `403 Forbidden`，JSON 格式仍被禁用——重新检查步骤 4。

**7. 配置 Hermes：**

```bash
# ~/.hermes/.env
SEARXNG_URL=http://localhost:8888
```

然后在 `~/.hermes/config.yaml` 中选择 SearXNG 作为搜索后端：

```yaml
web:
  search_backend: "searxng"
```

或通过 `hermes tools` → 网页搜索与提取 → SearXNG 设置。

---

#### 选项 B —— 使用公共实例

公共 SearXNG 实例列在 [searx.space](https://searx.space/)。按具有 **JSON 格式已启用** 的实例过滤（显示在表格中）。

```bash
# ~/.hermes/.env
SEARXNG_URL=https://searx.example.com
```

:::caution 公共实例
公共实例有速率限制、可变正常运行时间，并且可能随时禁用 JSON 格式。对于生产使用，强烈建议自托管。
:::

---

#### 将 SearXNG 与提取提供商配对

SearXNG 处理搜索；您需要单独的提供商进行 `web_extract`（包括任何深度抓取模式）。使用按能力键：

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # 或 tavily、exa、parallel
```

使用此配置，Hermes 对所有搜索查询使用 SearXNG，对 URL 提取使用 Firecrawl——结合免费搜索与高质量提取。

---

### Tavily

AI 优化的搜索、提取和抓取，提供慷慨的免费套餐。

```bash
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

在 [app.tavily.com](https://app.tavily.com/home) 获取密钥。免费套餐包括 1 000 次搜索/月。

---

### Exa

具有语义理解的神经搜索。适合研究和查找概念相关的内容。

```bash
# ~/.hermes/.env
EXA_API_KEY=your-exa-key-here
```

在 [exa.ai](https://exa.ai) 获取密钥。免费套餐包括 1 000 次搜索/月。

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

### 单一后端

为所有网页能力设置一个提供商：

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | tavily | exa | parallel
```

### 按能力配置

为搜索与提取使用不同的提供商。这让您可以结合免费搜索（SearXNG）与付费提取提供商，或反之：

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # 由 web_search 使用
  extract_backend: "firecrawl"  # 由 web_extract（及其深度抓取模式）使用
```

当按能力键为空时，两者都回退到 `web.backend`。当 `web.backend` 也为空时，后端从存在的 API 密钥/URL 自动检测。

**优先级顺序（按能力）：**
1. `web.search_backend` / `web.extract_backend`（显式按能力）
2. `web.backend`（共享回退）
3. 从环境变量自动检测

### 自动检测

如果未显式配置后端，Hermes 根据设置哪些凭证选择第一个可用的：

| 凭证存在 | 自动选择的后端 |
|--------------------|-----------------------|
| `FIRECRAWL_API_KEY` 或 `FIRECRAWL_API_URL` | firecrawl |
| `PARALLEL_API_KEY` | parallel |
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `SEARXNG_URL` | searxng |

---

## 验证您的设置

运行 `hermes setup` 查看检测到哪个网页后端：

```
✅ 网页搜索与提取 (searxng)
```

或通过 CLI 检查：

```bash
# 激活 venv 并直接运行网页工具模块
source ~/.hermes/hermes-agent/.venv/bin/activate
python -m tools.web_tools
```

这会打印活动后端及其状态：

```
✅ 网页后端: searxng
   使用 SearXNG (仅搜索): http://localhost:8888
```

---

## 故障排除

### `web_search` 返回 `{"success": false}`

- 检查 `SEARXNG_URL` 是否可达：`curl -s "http://localhost:8888/search?q=test&format=json"`
- 如果您得到 HTTP 403，JSON 格式被禁用——将 `json` 添加到 `settings.yml` 中的 `formats` 列表并重启
- 如果您得到连接错误，容器可能未运行：`docker ps | grep searxng`

### `web_extract` 显示 "仅搜索后端"

SearXNG 无法提取 URL 内容。将 `web.extract_backend` 设置为支持提取的提供商：

```yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"  # 或 tavily / exa / parallel
```

### SearXNG 返回 0 结果

某些公共实例禁用某些搜索引擎或类别。尝试：
- 不同的查询
- [searx.space](https://searx.space/) 上的不同公共实例
- 自托管您自己的实例以获得可靠结果

### 在公共实例上被速率限制

切换到自托管实例（请参阅上面的[选项 A](#选项-a--使用-docker-自托管推荐)）。使用 Docker，您自己的实例没有速率限制。

### `web_extract` 返回截断内容并带有 "摘要超时" 注释

辅助模型未在配置的超时内完成摘要。要么：

- 在 `config.yaml` 中提高 `auxiliary.web_extract.timeout`（全新安装默认 360s，如果键缺失则为 30s）
- 将 `web_extract` 辅助任务切换到更快的模型（例如 `google/gemini-3-flash-preview`）——请参阅 [`web_extract` 如何处理长页面](#web_extract-如何处理长页面)
- 对于摘要是错误工具的页面，请改用 `browser_navigate`

---

## 可选技能：`searxng-search`

对于需要直接通过 `curl` 使用 SearXNG 的智能体（例如当网页工具集不可用时作为回退），安装 `searxng-search` 可选技能：

```bash
hermes skills install official/research/searxng-search
```

这添加了一个技能，教智能体如何：
- 通过 `curl` 或 Python 调用 SearXNG JSON API
- 按类别过滤（`general`、`news`、`science` 等）
- 处理分页和错误情况
- 当 SearXNG 不可达时优雅地回退

