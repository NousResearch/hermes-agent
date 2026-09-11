---
title: 网页搜索与提取
description: 通过多个后端提供商搜索网页并提取页面内容——包括免费的自托管 SearXNG。
sidebar_label: Web Search
sidebar_position: 6
---

# 网页搜索与提取

Hermes Agent 内置两个可供模型调用的网页工具，由多个提供商支持：

- **`web_search`** — 搜索网页并返回排序结果
- **`web_extract`** — 从一个或多个 URL 获取并提取可读内容

两者均通过单一后端选择进行配置。提供商可通过 `hermes tools` 选择，或直接在 `config.yaml` 中设置。

## 后端

| 提供商 | 环境变量 | 搜索 | 提取 | 免费层级 |
|----------|---------|--------|---------|-----------|
| **Firecrawl**（默认） | `FIRECRAWL_API_KEY` | ✔ | ✔ | 500 积分/月 |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | ✔ 免费（自托管） |
| **Brave Search（免费层级）** | `BRAVE_SEARCH_API_KEY` | ✔ | — | 2 000 次查询/月 |
| **DDGS (DuckDuckGo)** | —（无需密钥） | ✔ | — | ✔ 免费 |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | 1 000 次搜索/月 |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | 1 000 次搜索/月 |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | 付费 |
| **MrScraper** | `MRSCRAPER_API_TOKEN` | ✔ | ✔ | 付费 |
| **xAI (Grok)** | `XAI_API_KEY` 或 `hermes auth login xai-oauth` | ✔ | — | 付费（SuperGrok 或按 token 计费） |

Brave Search、DDGS 和 xAI 均为**仅搜索**——如果同时需要 `web_extract`，可将其中任意一个与 Firecrawl/Tavily/Exa/Parallel 配合使用。DDGS 底层使用 [`ddgs` Python 包](https://pypi.org/project/ddgs/)；若尚未安装，请运行 `pip install ddgs`（或让 Hermes 在首次使用时懒加载安装）。xAI 通过 Responses API 运行 Grok 服务端的 `web_search` 工具——结果由 LLM 生成而非基于索引，因此标题、描述和 URL 选择均为模型输出（参见下方[信任模型说明](#xai-grok)）。

**按能力拆分：** 搜索和提取可分别使用不同的提供商——例如搜索使用 SearXNG（免费），提取使用 Firecrawl。详见下方[按能力配置](#per-capability-configuration)。

:::tip Nous 订阅用户
如果您拥有付费 [Nous Portal](https://portal.nousresearch.com) 订阅，网页搜索和提取可通过 **[Tool Gateway](tool-gateway.md)** 使用托管的 Firecrawl——无需 API 密钥。新安装可运行 `hermes setup --portal` 登录并一次性开启所有 gateway 工具；现有安装可通过 `hermes tools` 单独开启网页功能。
:::

---

## `web_extract` 如何处理长页面

后端返回的原始页面 markdown 可能非常庞大（论坛帖子、文档站点、带嵌入评论的新闻文章）。为保持上下文窗口可用，`web_extract` 采用**确定性字符预算** —— 不涉及任何 LLM 摘要：

| 页面大小（字符数） | 处理方式 |
|------------------------|--------------|
| 预算以内（默认 15 000） | 原样返回 —— 完整 markdown 直达 agent |
| 超出预算 | 头+尾窗口（约 75% 头部 / 25% 尾部，按 markdown 行边界切分），并附带明确的 `[TRUNCATED]` 尾注。完整的干净文本存储到磁盘，尾注告知 agent 文件路径以及分页读取被省略中间部分的确切 `read_file` 调用 |
| 超过 2 000 000 | 存储的文本上限为 2 MB |

每页预算可通过 `config.yaml` 中的 `web.extract_char_limit` 配置（默认 `15000`，范围限制在 2 000–500 000），agent 也可以通过工具的 `char_limit` 参数按调用提高。

### 当截断带来不便时

如果您明确需要实时 DOM 而非提取的 markdown —— 例如提取内容很少的 JS 密集页面 —— 请改用 `browser_navigate` + `browser_snapshot`。浏览器工具返回实时无障碍树（超大页面受其自身快照上限约束）。

---

## 设置

### 通过 `hermes tools` 快速设置

运行 `hermes tools`，导航至 **Web Search & Extract**，选择一个提供商。向导会提示输入所需的 URL 或 API 密钥，并写入您的配置。

```bash
hermes tools
```

---

### Firecrawl（默认）

功能完整的搜索和提取。推荐大多数用户使用。

```bash
# ~/.hermes/.env
FIRECRAWL_API_KEY=fc-your-key-here
```

在 [firecrawl.dev](https://firecrawl.dev) 获取密钥。免费层级包含每月 500 积分。

**自托管 Firecrawl：** 指向您自己的实例而非云端 API：

```bash
# ~/.hermes/.env
FIRECRAWL_API_URL=http://localhost:3002
```

设置 `FIRECRAWL_API_URL` 后，API 密钥为可选项（使用 `USE_DB_AUTHENTICATION=false` 禁用服务器认证）。

---

### SearXNG（免费，自托管）

SearXNG 是一个注重隐私的开源元搜索引擎，聚合来自 70 多个搜索引擎的结果。**无需 API 密钥**——只需将 Hermes 指向一个运行中的 SearXNG 实例。

SearXNG 为**仅搜索**——`web_extract` 需要单独的提取提供商。

#### 方案 A — 使用 Docker 自托管（推荐）

这为您提供无速率限制的私有实例。

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

打开 `~/searxng/searxng/settings.yml`，找到 `formats` 块（约第 84 行）：

```yaml
# 修改前（默认——JSON 已禁用）：
formats:
  - html

# 修改后（为 Hermes 启用 JSON）：
formats:
  - html
  - json
```

**5. 重启以应用更改：**

```bash
docker cp ~/searxng/searxng/settings.yml searxng:/etc/searxng/settings.yml
docker restart searxng
```

**6. 验证是否正常工作：**

```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"results\"])} results')"
```

您应该看到类似 `10 results` 的输出。如果收到 `403 Forbidden`，说明 JSON 格式仍未启用——请重新检查第 4 步。

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

或通过 `hermes tools` → Web Search & Extract → SearXNG 设置。

---

#### 方案 B — 使用公共实例

公共 SearXNG 实例列表见 [searx.space](https://searx.space/)。筛选**已启用 JSON 格式**的实例（表格中有显示）。

```bash
# ~/.hermes/.env
SEARXNG_URL=https://searx.example.com
```

:::caution 公共实例
公共实例有速率限制、可用性不稳定，且可能随时禁用 JSON 格式。生产环境强烈建议自托管。
:::

---

#### 将 SearXNG 与提取提供商配合使用

SearXNG 负责搜索；`web_extract` 需要单独的提供商。使用按能力配置的键：

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # 或 tavily、exa、parallel
```

使用此配置，Hermes 对所有搜索查询使用 SearXNG，对 URL 提取使用 Firecrawl——将免费搜索与高质量提取相结合。

---

### Tavily

针对 AI 优化的搜索和提取，免费层级慷慨。

```bash
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

在 [app.tavily.com](https://app.tavily.com/home) 获取密钥。免费层级包含每月 1 000 次搜索。

---

### Exa

具有语义理解的神经搜索。适合研究和查找概念相关内容。

```bash
# ~/.hermes/.env
EXA_API_KEY=your-exa-key-here
```

在 [exa.ai](https://exa.ai) 获取密钥。免费层级包含每月 1 000 次搜索。

---

### Parallel

具备深度研究能力的 AI 原生搜索和提取。

```bash
# ~/.hermes/.env
PARALLEL_API_KEY=your-parallel-key-here
```

在 [parallel.ai](https://parallel.ai) 申请访问权限。

---

### MrScraper {#mrscraper}

[MrScraper](https://mrscraper.com) 提供 Google SERP 搜索、JavaScript 渲染页面提取、结构化提取、网站抓取、scraper 创建与重新运行，以及结果检索。

```bash
# ~/.hermes/.env
MRSCRAPER_API_TOKEN=your-token-here
```

从 [MrScraper 控制台](https://app.mrscraper.com) 获取 token，然后在 `hermes tools` → **Web Search & Extract** 中选择 **MrScraper**，或直接配置：

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "mrscraper"
  extract_backend: "mrscraper"
```

除了标准的 `web_search` 和 `web_extract`，内置集成还提供一组专用 `mrscraper_*` 工具，用于抓取、按 prompt/listing/结构化规则提取、管理 scraper、批量运行和获取结果。`mrscraper_fetch_rendered_html` 可返回渲染后的 HTML 或 Markdown，并可请求截图、代理国家/地区、Cookie 或等待选择器。

:::note 渲染页面，而非交互式浏览器
MrScraper 可以渲染单个 URL，但不提供 `browser_click`、`browser_type` 等交互式 `browser_*` 工具所需的持久 CDP/WebSocket 会话。需要多步骤页面交互时，请单独配置浏览器提供商。参见[浏览器自动化](browser.md#mrscraper-rendered-page-tool)。
:::

Hermes MCP 目录中也提供官方托管的 MrScraper MCP。这是一个独立集成，使用独立密钥 `MCP_MRSCRAPER_API_KEY`；如希望使用 MCP 工具而非内置原生工具，可运行 `hermes mcp install mrscraper`。

---

### xAI (Grok) {#xai-grok}

通过 Responses API 将 `web_search` 路由至 Grok 服务端的 [web_search 工具](https://docs.x.ai/developers/tools/web-search)。Grok 执行实际搜索并以结构化 JSON 返回最佳结果。

支持两种凭证路径——无需新的环境变量，无需新的设置向导：

```bash
# ~/.hermes/.env（环境变量路径）
XAI_API_KEY=sk-xai-your-key-here
```

或对于 SuperGrok 订阅用户：

```bash
hermes auth login xai-oauth
```

然后选择 xAI 作为搜索后端：

```yaml
# ~/.hermes/config.yaml
web:
  backend: "xai"
```

**可选配置项：**

```yaml
web:
  backend: "xai"
  xai:
    model: grok-build-0.1        # web_search 所需的推理模型（默认）
    allowed_domains:             # 可选，最多 5 个——与 excluded_domains 互斥
      - arxiv.org
    excluded_domains:            # 可选，最多 5 个
      - example-spam.com
    timeout: 90                  # 秒（默认）
```

**仅搜索**——如果同时需要 `web_extract`，请与 Firecrawl / Tavily / Exa / Parallel 配合使用。遇到 401 时，提供商会执行一次强制 OAuth token 刷新并重试（覆盖窗口中途吊销和主动过期检查无法解码的不透明 token）；环境变量凭证跳过重试。

:::caution 信任模型
与基于索引的提供商（Brave、Tavily、Exa）返回逐字搜索引擎结果不同，xAI 是由 LLM 选择要呈现的 URL 并自行撰写标题和描述。查询的*内容*会影响输出，因此恶意构造的查询（例如通过 agent 获取的不可信上游输入注入）原则上可以引导 Grok 输出攻击者指定的 URL。对返回的 URL 应与对待任何模型生成链接一样——在获取前进行验证，尤其是当查询来自不可信输入时。
:::

---

## 配置

### 单一后端

为所有网页功能设置一个提供商：

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | mrscraper | xai
```

### 按能力配置 {#per-capability-configuration}

搜索和提取使用不同的提供商。这允许您将免费搜索（SearXNG）与付费提取提供商组合使用，反之亦然：

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # 由 web_search 使用
  extract_backend: "firecrawl"  # 由 web_extract 使用
```

当按能力键为空时，两者均回退到 `web.backend`。当 `web.backend` 也为空时，后端根据存在的 API 密钥/URL 自动检测。

**优先级顺序（按能力）：**
1. `web.search_backend` / `web.extract_backend`（显式按能力配置）
2. `web.backend`（共享回退）
3. 从环境变量自动检测

### 自动检测

如果未显式配置后端，Hermes 根据已设置的凭证选择第一个可用的后端：

| 存在的凭证 | 自动选择的后端 |
|--------------------|-----------------------|
| `FIRECRAWL_API_KEY` 或 `FIRECRAWL_API_URL` | firecrawl |
| `PARALLEL_API_KEY` | parallel |
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `SEARXNG_URL` | searxng |

xAI Web Search **不在**自动检测链中——设置了 `XAI_API_KEY`（或通过 xAI Grok OAuth 登录）不会自动将网页流量路由至 xAI，因为这些凭证同时用于推理/TTS/图像生成，用户可能希望为网页使用不同的后端。请通过 `web.backend: "xai"` 显式启用。

MrScraper 同样不在旧版优先级自动检测链中。请显式设置 `web.backend: "mrscraper"`（或两个按能力配置键）；`hermes tools` 会自动写入这些配置。

---

## 验证设置

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

这将打印活动后端及其状态：

```
✅ Web backend: searxng
   Using SearXNG (search only): http://localhost:8888
```

---

## 故障排查

### `web_search` 返回 `{"success": false}`

- 检查 `SEARXNG_URL` 是否可达：`curl -s "http://localhost:8888/search?q=test&format=json"`
- 如果收到 HTTP 403，说明 JSON 格式已禁用——在 `settings.yml` 的 `formats` 列表中添加 `json` 并重启
- 如果收到连接错误，容器可能未运行：`docker ps | grep searxng`

### `web_extract` 提示"search-only backend"

SearXNG 无法提取 URL 内容。将 `web.extract_backend` 设置为支持提取的提供商：

```yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"  # 或 tavily / exa / parallel
```

### SearXNG 返回 0 条结果

部分公共实例禁用了某些搜索引擎或分类。请尝试：
- 换一个查询词
- 从 [searx.space](https://searx.space/) 换一个公共实例
- 自托管实例以获得稳定结果

### 公共实例遭遇速率限制

切换到自托管实例（参见上方[方案 A](#option-a--self-host-with-docker-recommended)）。使用 Docker，您自己的实例没有速率限制。

### `web_extract` 返回截断内容并附有 `[TRUNCATED]` 尾注

对于超出字符预算的页面这是预期行为。尾注会给出保存完整干净文本的磁盘文件，以及分页读取被省略中间部分的确切 `read_file` 调用。若要内联查看更多内容，请在 `config.yaml` 中调大 `web.extract_char_limit`，或在调用时传入更大的 `char_limit`。

---

## 可选技能：`searxng-search`

对于需要直接通过 `curl` 使用 SearXNG 的 agent（例如作为网页工具集不可用时的回退），请安装 `searxng-search` 可选技能：

```bash
hermes skills install official/research/searxng-search
```

这将添加一个技能，教 agent 如何：
- 通过 `curl` 或 Python 调用 SearXNG JSON API
- 按分类筛选（`general`、`news`、`science` 等）
- 处理分页和错误情况
- 在 SearXNG 不可达时优雅降级
