---
title: "Page Agent"
sidebar_label: "Page Agent"
description: "将 alibaba/page-agent 嵌入你自己的 Web 应用——一个纯 JavaScript 页内 GUI agent，以单个 <script> 标签或 npm 包形式发布，让你网站的终端用户能用自然语言驱动 UI（如'点击登录，将用户名填为 John'）。"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Page Agent

将 alibaba/page-agent 嵌入你自己的 Web 应用——一个纯 JavaScript 页内 GUI agent，以单个 &lt;script> 标签或 npm 包形式发布，让你网站的终端用户能用自然语言驱动 UI（"点击登录，将用户名填为 John"）。无需 Python，无需无头浏览器，无需扩展程序。当用户是 Web 开发者，希望为其 SaaS / 管理面板 / B2B 工具添加 AI copilot、通过自然语言让遗留 Web 应用可访问，或针对本地（Ollama）或云端（Qwen / OpenAI / OpenRouter）LLM 评估 page-agent 时，使用此 skill。不适用于服务端浏览器自动化——此类需求请将用户引导至 Hermes 内置的浏览器工具。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选——通过 `hermes skills install official/web-development/page-agent` 安装 |
| 路径 | `optional-skills/web-development/page-agent` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `web`, `javascript`, `agent`, `browser`, `gui`, `alibaba`, `embed`, `copilot`, `saas` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# page-agent

alibaba/page-agent（https://github.com/alibaba/page-agent，17k+ stars，MIT）是一个用 TypeScript 编写的页内 GUI agent。它运行在网页内部，以文本形式读取 DOM（无需截图，无需多模态 LLM），并对当前页面执行自然语言指令，如"点击登录按钮，然后将用户名填为 John"。纯客户端——宿主网站只需引入一个 script 并传入兼容 OpenAI 的 LLM 端点即可。

## 何时使用此 skill

当用户希望实现以下目标时，加载此 skill：

- **在自己的 Web 应用中集成 AI copilot**（SaaS、管理面板、B2B 工具、ERP、CRM）——"我仪表盘上的用户应该能输入'为 Acme Corp 创建发票并发送邮件'，而不是点击五个页面"
- **在不重写前端的情况下现代化遗留 Web 应用**——page-agent 可直接叠加在现有 DOM 之上
- **通过自然语言提升无障碍访问能力**——语音 / 屏幕阅读器用户通过描述需求来驱动 UI
- **演示或评估 page-agent**，对接本地（Ollama）或托管（Qwen、OpenAI、OpenRouter）LLM
- **构建交互式培训 / 产品演示**——让 AI 在真实 UI 中引导用户完成"如何提交报销单"

## 何时不应使用此 skill

- 用户希望 **Hermes 本身驱动浏览器** → 使用 Hermes 内置的浏览器工具（Browserbase / Camofox）。page-agent 是*相反*的方向。
- 用户希望**在不嵌入的情况下实现跨标签页自动化** → 使用 Playwright、browser-use 或 page-agent Chrome 扩展
- 用户需要**视觉定位 / 截图** → page-agent 仅支持文本 DOM；请改用多模态浏览器 agent

## 前置条件

- Node 22.13+ 或 24+，npm 10+（文档声称需要 11+，但 10.9 实际可用）
- 兼容 OpenAI 的 LLM 端点：Qwen（DashScope）、OpenAI、Ollama、OpenRouter，或任何支持 `/v1/chat/completions` 的服务
- 带开发者工具的浏览器（用于调试）

## 路径 1——通过 CDN 30 秒快速体验（无需安装）

最快的上手方式。使用阿里巴巴的免费测试 LLM 代理——**仅供评估使用**，须遵守其服务条款。

添加到任意 HTML 页面（或粘贴到开发者工具控制台作为书签脚本）：

```html
<script src="https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js" crossorigin="true"></script>
```

面板随即出现。输入指令。完成。

书签脚本形式（拖入书签栏，在任意页面点击）：

```javascript
javascript:(function(){var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js';document.head.appendChild(s);})();
```

## 路径 2——npm 安装到你自己的 Web 应用（生产使用）

在现有 Web 项目中（React / Vue / Svelte / 纯 HTML）：

```bash
npm install page-agent
```

使用你自己的 LLM 端点进行配置——**切勿将演示 CDN 用于真实用户**：

```javascript
import { PageAgent } from 'page-agent'

const agent = new PageAgent({
    model: 'qwen3.5-plus',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    apiKey: process.env.LLM_API_KEY,   // never hardcode
    language: 'en-US',
})

// 为终端用户显示面板：
agent.panel.show()

// 或以编程方式驱动：
await agent.execute('Click submit button, then fill username as John')
```

Provider 示例（任何兼容 OpenAI 的端点均可使用）：

| Provider | `baseURL` | `model` |
|----------|-----------|---------|
| Qwen / DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.5-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama（本地） | `http://localhost:11434/v1` | `qwen3:14b` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.6` |

**关键配置字段**（传入 `new PageAgent({...})`）：

- `model`、`baseURL`、`apiKey` — LLM 连接配置
- `language` — UI 语言（`en-US`、`zh-CN` 等）
- 存在白名单和数据脱敏 hook，用于限制 agent 可操作的范围——完整选项列表见 https://alibaba.github.io/page-agent/

**安全性。** 在真实部署中，不要将 `apiKey` 放在客户端代码中——通过你的后端代理 LLM 调用，并将 `baseURL` 指向你的代理。演示 CDN 之所以存在，是因为阿里巴巴为评估目的运行了该代理。

## 路径 3——克隆源码仓库（贡献代码，或深度定制）

当用户希望修改 page-agent 本身、通过本地 IIFE bundle 在任意网站上测试，或开发浏览器扩展时使用此路径。

```bash
git clone https://github.com/alibaba/page-agent.git
cd page-agent
npm ci              # exact lockfile install (or `npm i` to allow updates)
```

在仓库根目录创建 `.env` 文件，配置 LLM 端点。示例：

```
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

Ollama 配置：

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=NA
LLM_MODEL_NAME=qwen3:14b
```

常用命令：

```bash
npm start           # docs/website dev server
npm run build       # build every package
npm run dev:demo    # serve IIFE bundle at http://localhost:5174/page-agent.demo.js
npm run dev:ext     # develop the browser extension (WXT + React)
npm run build:ext   # build the extension
```

**在任意网站上测试**，使用本地 IIFE bundle。添加此书签脚本：

```javascript
javascript:(function(){var s=document.createElement('script');s.src=`http://localhost:5174/page-agent.demo.js?t=${Math.random()}`;s.onload=()=>console.log('PageAgent ready!');document.head.appendChild(s);})();
```

然后：运行 `npm run dev:demo`，在任意页面点击书签脚本，本地构建即注入页面。保存后自动重新构建。

**警告：** 在开发构建期间，`.env` 中的 `LLM_API_KEY` 会被内联到 IIFE bundle 中。不要分享该 bundle，不要提交它，不要将 URL 粘贴到 Slack。（已验证：对公开开发 bundle 执行 grep 会返回 `.env` 中的字面值。）

## 仓库结构（路径 3）

使用 npm workspaces 的 monorepo。核心包：

| 包 | 路径 | 用途 |
|---------|------|---------|
| `page-agent` | `packages/page-agent/` | 带 UI 面板的主入口 |
| `@page-agent/core` | `packages/core/` | 核心 agent 逻辑，无 UI |
| `@page-agent/mcp` | `packages/mcp/` | MCP server（beta） |
| — | `packages/llms/` | LLM 客户端 |
| — | `packages/page-controller/` | DOM 操作 + 视觉反馈 |
| — | `packages/ui/` | 面板 + 国际化 |
| — | `packages/extension/` | Chrome/Firefox 扩展 |
| — | `packages/website/` | 文档 + 落地页 |

## 验证是否正常工作

路径 1 或路径 2 完成后：
1. 在浏览器中打开页面并开启开发者工具
2. 应看到一个浮动面板。若未出现，检查控制台报错（最常见原因：LLM 端点 CORS 问题、错误的 `baseURL`，或无效的 API key）
3. 输入一条与页面可见内容匹配的简单指令（"click the Login link"）
4. 观察 Network 标签页——应看到发往你的 `baseURL` 的请求

路径 3 完成后：
1. `npm run dev:demo` 输出 `Accepting connections at http://localhost:5174`
2. `curl -I http://localhost:5174/page-agent.demo.js` 返回 `HTTP/1.1 200 OK`，`Content-Type: application/javascript`
3. 在任意网站点击书签脚本，面板出现

## 常见问题

- **在生产环境使用演示 CDN** — 不要这样做。它有速率限制，使用阿里巴巴的免费代理，且其服务条款禁止生产使用。
- **API key 泄露** — 传入 `new PageAgent({apiKey: ...})` 的任何 key 都会打包进你的 JS bundle。真实部署时务必通过自己的后端代理。
- **不兼容 OpenAI 格式的端点**会静默失败或报出难以理解的错误。如果你的 provider 需要原生 Anthropic/Gemini 格式，请在前面加一层 OpenAI 兼容代理（LiteLLM、OpenRouter）。
- **CSP 拦截** — 启用严格 Content-Security-Policy 的网站可能拒绝加载 CDN script 或禁止内联 eval。此时请从你自己的域名自托管。
- **编辑路径 3 中的 `.env` 后需重启开发服务器** — Vite 仅在启动时读取环境变量。
- **Node 版本** — 仓库声明支持 `^22.13.0 || >=24`。Node 20 在 `npm ci` 时会因引擎检查报错失败。
- **npm 10 vs 11** — 文档要求 npm 11+；npm 10.9 实际可正常使用。

## 参考资料

- 仓库：https://github.com/alibaba/page-agent
- 文档：https://alibaba.github.io/page-agent/
- 许可证：MIT（基于 browser-use 的 DOM 处理内部实现，Copyright 2024 Gregor Zunic）