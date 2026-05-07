---
title: "Nous Tool Gateway（工具网关）"
description: "通过 Nous 订阅统一使用网页搜索、文生图、语音合成与浏览器自动化，无需单独申请 Firecrawl、FAL、OpenAI、Browser Use 等 API Key"
sidebar_label: "Tool Gateway"
sidebar_position: 2
---

# Nous Tool Gateway（工具网关）

:::tip 快速开始
Tool Gateway 包含在付费 Nous Portal 订阅中。**[管理订阅 →](https://portal.nousresearch.com/manage-subscription)**
:::

**Tool Gateway** 让已付费的 [Nous Portal](https://portal.nousresearch.com) 用户通过同一份订阅，直接使用网页搜索、文生图、语音合成（TTS）与浏览器自动化，而**不必**再分别注册 Firecrawl、FAL、OpenAI、Browser Use 等服务的 API Key。

## 包含能力

| 工具 | 作用 | 若不用网关，可改用 |
|------|------|---------------------|
| **网页搜索与抓取** | 通过 Firecrawl 搜索并抽取页面内容 | `FIRECRAWL_API_KEY`、`EXA_API_KEY`、`PARALLEL_API_KEY`、`TAVILY_API_KEY` |
| **文生图** | 通过 FAL 生成图像（8 个模型：FLUX 2 Klein/Pro、GPT-Image、Nano Banana Pro、Ideogram、Recraft V4 Pro、Qwen、Z-Image） | `FAL_KEY` |
| **语音合成** | 通过 OpenAI TTS 将文字转为语音 | `VOICE_TOOLS_OPENAI_KEY`、`ELEVENLABS_API_KEY` |
| **浏览器自动化** | 通过 Browser Use 控制云端浏览器 | `BROWSER_USE_API_KEY`、`BROWSERBASE_API_KEY` |

上述四类能力均计入 Nous 订阅计费。你可以按需组合——例如网页与文生图走网关，TTS 仍使用自己的 ElevenLabs Key。

## 资格与账号

Tool Gateway 仅对 **[付费](https://portal.nousresearch.com/manage-subscription)** Nous Portal 订阅开放；免费档不可用——请 [升级订阅](https://portal.nousresearch.com/manage-subscription) 后解锁。

检查当前状态：

```bash
hermes status
```

在输出中找到 **Nous Tool Gateway** 小节：会标明哪些工具经订阅网关启用、哪些使用直连 Key、哪些尚未配置。

## 如何启用 Tool Gateway

### 在模型配置流程中

运行 `hermes model` 并选择 Nous Portal 作为提供商时，Hermes 会主动询问是否启用 Tool Gateway：

```
Your Nous subscription includes the Tool Gateway.

  The Tool Gateway gives you access to web search, image generation,
  text-to-speech, and browser automation through your Nous subscription.
  No need to sign up for separate API keys — just pick the tools you want.

  ○ Web search & extract (Firecrawl) — not configured
  ○ Image generation (FAL) — not configured
  ○ Text-to-speech (OpenAI TTS) — not configured
  ○ Browser automation (Browser Use) — not configured

  ● Enable Tool Gateway
  ○ Skip
```

选择 **Enable Tool Gateway** 即可。

若 `.env` 中已有部分直连 API Key，提示会相应变化：可为全部工具启用网关（直连 Key 仍保留在 `.env` 但运行时不用）、仅为未配置项启用，或完全跳过。

### 通过 `hermes tools`

也可在交互式工具配置中逐项启用：

```bash
hermes tools
```

选择工具类别（Web、Browser、Image Generation、TTS），再将提供商选为 **Nous Subscription**。这会在配置里把对应工具的 `use_gateway` 设为 `true`。

### 手动编辑配置

在 `~/.hermes/config.yaml` 中直接设置 `use_gateway`：

```yaml
web:
  backend: firecrawl
  use_gateway: true

image_gen:
  use_gateway: true

tts:
  provider: openai
  use_gateway: true

browser:
  cloud_provider: browser-use
  use_gateway: true
```

## 工作原理

当某工具的 `use_gateway: true` 时，运行时会把 API 调用路由到 Nous Tool Gateway，而不是使用直连 Key：

1. **网页工具** — `web_search` / `web_extract` 走网关的 Firecrawl 端点  
2. **文生图** — `image_generate` 走网关的 FAL 端点  
3. **TTS** — `text_to_speech` 走网关的 OpenAI Audio 端点  
4. **浏览器** — `browser_navigate` 等走网关的 Browser Use 端点  

网关使用 Nous Portal 凭据认证（在 `hermes model` 完成后写入 `~/.hermes/auth.json`）。

### 优先级

每个工具都会先看 `use_gateway`：

- **`use_gateway: true`** → 强制走网关，即使 `.env` 里仍有直连 Key  
- **`use_gateway: false`**（或未设置）→ 若有直连 Key 则优先直连；仅在没有直连凭据时才回退到网关  

因此你可以在网关与直连之间切换，而无需删除 `.env` 中的旧 Key。

## 切回直连 Key

对单个工具停用网关：

```bash
hermes tools    # 选择该工具 → 选直连提供商
```

或在配置中设 `use_gateway: false`：

```yaml
web:
  backend: firecrawl
  use_gateway: false  # 此时使用 .env 中的 FIRECRAWL_API_KEY
```

在 `hermes tools` 中选择非网关提供商时，`use_gateway` 会自动设为 `false`，避免配置自相矛盾。

## 查看状态

```bash
hermes status
```

**Nous Tool Gateway** 小节示例：

```
◆ Nous Tool Gateway
  Nous Portal   ✓ managed tools available
  Web tools       ✓ active via Nous subscription
  Image gen       ✓ active via Nous subscription
  TTS             ✓ active via Nous subscription
  Browser         ○ active via Browser Use key
  Modal           ○ available via subscription (optional)
```

标记为 “active via Nous subscription” 的即经网关路由；带自有 Key 的会显示当前激活的提供商。

## 进阶：自建网关

若使用自建或自定义网关，可在 `~/.hermes/.env` 中用环境变量覆盖端点：

```bash
TOOL_GATEWAY_DOMAIN=nousresearch.com     # 网关路由基础域名
TOOL_GATEWAY_SCHEME=https                 # http 或 https（默认 https）
TOOL_GATEWAY_USER_TOKEN=your-token        # 鉴权 Token（通常由程序自动填充）
FIRECRAWL_GATEWAY_URL=https://...         # 单独覆盖 Firecrawl 端点
```

这些变量与订阅状态无关，始终可在配置中看到，便于自建基础设施。

## 常见问题

### 需要删掉已有的 API Key 吗？

不需要。`use_gateway: true` 时运行时会跳过直连 Key 并走网关；Key 仍保留在 `.env`。之后若关闭网关，会自动恢复使用直连 Key。

### 能否部分工具走网关、部分走直连？

可以。`use_gateway` 按工具独立配置。例如：网页与文生图走网关，TTS 用 ElevenLabs，浏览器用 Browserbase。

### 订阅到期会怎样？

经网关路由的工具会停止工作，直到你 [续订](https://portal.nousresearch.com/manage-subscription) 或通过 `hermes tools` 改回直连 Key。

### 与「消息网关」（各聊天平台）是否冲突？

不冲突。Tool Gateway 作用于**工具运行时**的 API 路由，与 CLI、Telegram、Discord 等入口无关。

### Modal 算在 Tool Gateway 里吗？

Modal（无服务器终端后端）可作为 Nous 订阅的可选附加能力，但**不会**由 Tool Gateway 安装向导一并打开——请单独通过 `hermes setup terminal` 或在 `config.yaml` 中配置。
