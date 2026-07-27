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
| **xAI (Grok)** | `XAI_API_KEY` 或 `hermes auth login xai-oauth` | ✔ | — | 付费（SuperGrok 或按 token 计费） |

Brave Search、DDGS 和 xAI 均为**仅搜索**——如果同时需要 `web_extract`，可将其中任意一个与 Firecrawl/Tavily/Exa/Parallel 配合使用。DDGS 底层使用 [`ddgs` Python 包](https://pypi.org/project/ddgs/)；若尚未安装，请运行 `pip install ddgs`（或让 Hermes 在首次使用时懒加载安装）。xAI 通过 Responses API 运行 Grok 服务端的 `web_search` 工具——结果由 LLM 生成而非基于索引，因此标题、描述和 URL 选择均为模型输出（参见下方[信任模型说明](#xai-grok)）。

**按能力拆分：** 搜索和提取可分别使用不同的提供商——例如搜索使用 SearXNG（免费），提取使用 Firecrawl。详见下方[按能力配置](#per-capability-configuration)。

:::tip Nous 订阅用户
如果您拥有付费 [Nous Portal](https://portal.nousresearch.com) 订阅，网页搜索和提取可通过 **[Tool Gateway](tool-gateway.md)** 使用托管的 Firecrawl——无需 API 密钥。新安装可运行 `hermes setup --portal` 登录并一次性开启所有 gateway 工具；现有安装可通过 `hermes tools` 单独开启网页功能。
:::

---

## `web_extract` 如何处理长页面

后端返回的原始页面 markdown 可能非常庞大（论坛帖子、文档站点、带嵌入评论的新闻文章）。为保持上下文窗口可用并降低成本，`web_extract` 在将内容交给 agent 之前，会通过 **`web_extract` 辅助模型**对返回内容进行处理。行为完全由大小决定：

| 页面大小（字符数） | 处理方式 |
|------------------------|--------------|
| 5 000 以下 | 原样返回——不调用 LLM，完整 markdown 直达 agent |
| 5 000 – 500 000 | 通过 `web_extract` 辅助模型单次摘要，输出上限约 5 000 字符 |
| 500 000 – 2 000 000 | 分块处理：拆分为 10 万字符的块，并行摘要每块，再合成最终摘要（约 5 000 字符） |
| 超过 2 000 000 | 拒绝处理，并提示使用更具体的来源 URL |

摘要保留引用、代码块和关键事实的原始格式——它是内容压缩器，而非改写器。如果摘要失败或超时，Hermes 会回退到原始内容的前约 5 000 字符，而非返回无用的错误信息。

### 哪个模型负责摘要？

`web_extract` 辅助任务。默认情况下（`auxiliary.web_extract.provider: "auto"`），使用您的**主聊天模型**——与 `hermes model` 相同的提供商和模型。对大多数配置而言这没问题，但在昂贵的推理模型（Opus、MiniMax M2.7 等）上，每次长页面提取都会产生可观的成本。

若要将提取摘要路由到廉价快速的模型，无论主模型是什么：

```yaml
# ~/.hermes/config.yaml
auxiliary:
  web_extract:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 360       # 秒；如果遇到摘要超时，请调大此值
```

或交互式选择：`hermes model` → **Configure auxiliary models** → `web_extract`。

完整参考和按任务覆盖模式，请参阅[辅助模型](/user-guide/configuration#auxiliary-models)。

### 摘要处理不适用的情况

如果您明确需要原始、未经摘要的页面内容——例如正在抓取结构化页面，LLM 摘要会丢失重要字段——请改用 `browser_navigate` + `browser_snapshot`。浏览器工具返回实时无障碍树，不经辅助模型改写（在超大页面上受其自身 8 000 字符快照上限约束）。

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
  backend: "searxng"   # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | xai
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

### `web_extract` 返回截断内容并附有"summarization timed out"提示

辅助模型未能在配置的超时时间内完成摘要。可以：

- 在 `config.yaml` 中调大 `auxiliary.web_extract.timeout`（新安装默认 360 秒，若键缺失则为 30 秒）
- 将 `web_extract` 辅助任务切换到更快的模型（例如 `google/gemini-3-flash-preview`）——参见 [`web_extract` 如何处理长页面](#how-web_extract-handles-long-pages)
- 对于摘要处理不适用的页面，改用 `browser_navigate`

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