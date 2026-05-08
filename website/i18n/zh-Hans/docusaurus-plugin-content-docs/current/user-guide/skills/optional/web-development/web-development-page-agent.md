---
title: "Page Agent"
sidebar_label: "Page Agent"
description: "将 alibaba/page-agent 嵌入你自己的 Web 应用——一个纯 JavaScript 的页面内 GUI 代理，以单个 <script> 标签或 npm 包形式交付，让最终用户..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Page Agent

将 alibaba/page-agent 嵌入你自己的 Web 应用——一个纯 JavaScript 的页面内 GUI 代理，以单个 &lt;script> 标签或 npm 包形式交付，让最终用户通过自然语言驱动 UI（"点击登录，用户名填 John"）。无需 Python，无需无头浏览器，无需扩展。当用户是希望为 SaaS / 管理后台 / B2B 工具添加 AI 副驾驶的 Web 开发者，希望通过自然语言访问旧版 Web 应用，或想用本地 (Ollama) 或云端 (Qwen / OpenAI / OpenRouter) 大语言模型评估 page-agent 时，请使用此技能。不适用于服务端浏览器自动化——请将此类用户引导至 Hermes 内置的浏览器工具。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/web-development/page-agent` 安装 |
| 路径 | `optional-skills/web-development/page-agent` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `web`、`javascript`、`agent`、`browser`、`gui`、`alibaba`、`embed`、`copilot`、`saas` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# page-agent

alibaba/page-agent（https://github.com/alibaba/page-agent，17k+ stars，MIT）是一个用 TypeScript 编写的页面内 GUI 代理。它运行在网页内部，将 DOM 作为文本读取（不使用截图，不使用多模态 LLM），并针对当前页面执行自然语言指令，如"点击登录按钮，然后用户名填 John"。纯客户端——宿主站点只需引入脚本并传入 OpenAI 兼容的 LLM 端点。

## 何时使用此技能

当用户想要以下操作时加载此技能：

- **在自己的 Web 应用中部署 AI 副驾驶**（SaaS、管理后台、B2B 工具、ERP、CRM）——"我的仪表板用户应该能够输入'为 Acme Corp 创建发票并邮件发送'而不是点击五个页面"
- **现代化旧版 Web 应用**而无需重写前端——page-agent 直接覆盖在现有 DOM 上
- **通过自然语言添加无障碍功能**——语音/屏幕阅读器用户通过描述他们想要什么来驱动 UI
- **演示或评估 page-agent**——使用本地 (Ollama) 或托管 (Qwen、OpenAI、OpenRouter) LLM
- **构建交互式培训/产品演示**——让 AI 在真实 UI 中引导用户完成"如何提交费用报告"

## 何时不使用此技能

- 用户希望 **Hermes 自身驱动浏览器** → 使用 Hermes 内置的浏览器工具 (Browserbase / Camofox)。page-agent 是*相反的*方向。
- 用户希望 **跨标签页自动化且不嵌入** → 使用 Playwright、browser-use 或 page-agent Chrome 扩展
- 用户需要 **视觉定位/截图** → page-agent 仅支持文本 DOM；请使用多模态浏览器代理

## 前置条件

- Node 22.13+ 或 24+，npm 10+（文档声称需要 11+，但 10.9 也可以）
- OpenAI 兼容的 LLM 端点：Qwen (DashScope)、OpenAI、Ollama、OpenRouter 或任何支持 `/v1/chat/completions` 的服务
- 带有开发者工具的浏览器（用于调试）

## 路径 1 — 通过 CDN 进行 30 秒演示（无需安装）

最快的体验方式。使用 alibaba 的免费测试 LLM 代理——**仅供评估使用**，受其条款约束。

添加到任何 HTML 页面（或作为书签粘贴到开发者工具控制台）：

```html
<script src="https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js" crossorigin="true"></script>
```

出现一个面板。输入指令即可。

书签形式（拖到书签栏，在任何页面上点击）：

```javascript
javascript:(function(){var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js';document.head.appendChild(s);})();
```

## 路径 2 — 通过 npm 安装到你的 Web 应用（生产使用）

在现有的 Web 项目中（React / Vue / Svelte / 纯 HTML）：

```bash
npm install page-agent
```

连接你自己的 LLM 端点——**切勿将演示 CDN 用于真实用户**：

```javascript
import { PageAgent } from 'page-agent'

const agent = new PageAgent({
    model: 'qwen3.5-plus',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    apiKey: process.env.LLM_API_KEY,   // 永远不要硬编码
    language: 'en-US',
})

// 为最终用户显示面板：
agent.panel.show()

// 或以编程方式驱动：
await agent.execute('Click submit button, then fill username as John')
```

提供商示例（任何 OpenAI 兼容端点都可以）：

| 提供商 | `baseURL` | `model` |
|--------|-----------|---------|
| Qwen / DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.5-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama（本地） | `http://localhost:11434/v1` | `qwen3:14b` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.6` |

**关键配置字段**（传递给 `new PageAgent({...})`）：

- `model`、`baseURL`、`apiKey` — LLM 连接
- `language` — UI 语言（`en-US`、`zh-CN` 等）
- 存在允许列表和数据脱敏钩子用于限制代理可以触摸的内容——请参阅 https://alibaba.github.io/page-agent/ 获取完整选项列表

**安全。** 在真实部署中，不要将 `apiKey` 放在客户端代码中——通过后端代理 LLM 调用并将 `baseURL` 指向你的代理。演示 CDN 存在是因为 alibaba 运行了该代理用于评估。

## 路径 3 — 克隆源代码仓库（贡献或修改）

当用户想要修改 page-agent 本身、通过本地 IIFE 包在任意网站上测试，或开发浏览器扩展时使用。

```bash
git clone https://github.com/alibaba/page-agent.git
cd page-agent
npm ci              # 精确锁定文件安装（或使用 `npm i` 允许更新）
```

在仓库根目录创建 `.env` 文件并配置 LLM 端点。示例：

```
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

Ollama 版本：

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=NA
LLM_MODEL_NAME=qwen3:14b
```

常用命令：

```bash
npm start           # 文档/网站开发服务器
npm run build       # 构建所有包
npm run dev:demo    # 在 http://localhost:5174/page-agent.demo.js 提供 IIFE 包
npm run dev:ext     # 开发浏览器扩展 (WXT + React)
npm run build:ext   # 构建扩展
```

**在任何网站上测试**，使用本地 IIFE 包。添加此书签：

```javascript
javascript:(function(){var s=document.createElement('script');s.src=`http://localhost:5174/page-agent.demo.js?t=${Math.random()}`;s.onload=()=>console.log('PageAgent ready!');document.head.appendChild(s);})();
```

然后：`npm run dev:demo`，在任何页面上点击书签，本地构建就会被注入。保存时自动重新构建。

**警告：** 你的 `.env` 中的 `LLM_API_KEY` 会在开发构建期间被内联到 IIFE 包中。不要分享该包。不要提交它。不要将 URL 粘贴到 Slack。（已验证：在公共开发包中 grep 可以返回 `.env` 中的字面值。）

## 仓库布局（路径 3）

使用 npm workspaces 的 monorepo。关键包：

| 包 | 路径 | 用途 |
|----|------|------|
| `page-agent` | `packages/page-agent/` | 带 UI 面板的主入口 |
| `@page-agent/core` | `packages/core/` | 核心代理逻辑，无 UI |
| `@page-agent/mcp` | `packages/mcp/` | MCP 服务器 (beta) |
| — | `packages/llms/` | LLM 客户端 |
| — | `packages/page-controller/` | DOM 操作 + 视觉反馈 |
| — | `packages/ui/` | 面板 + 国际化 |
| — | `packages/extension/` | Chrome/Firefox 扩展 |
| — | `packages/website/` | 文档 + 着陆页 |

## 验证是否工作

路径 1 或路径 2 之后：
1. 在浏览器中打开页面，开启开发者工具
2. 你应该看到一个浮动面板。如果没有，检查控制台错误（最常见：LLM 端点的 CORS 问题、错误的 `baseURL` 或无效的 API 密钥）
3. 输入一个与页面上可见内容匹配的简单指令（"点击登录链接"）
4. 查看网络标签——你应该看到对 `baseURL` 的请求

路径 3 之后：
1. `npm run dev:demo` 输出 `Accepting connections at http://localhost:5174`
2. `curl -I http://localhost:5174/page-agent.demo.js` 返回 `HTTP/1.1 200 OK`，`Content-Type: application/javascript`
3. 在任何站点点击书签；面板出现

## 注意事项

- **生产环境使用演示 CDN** — 不要这样做。它有速率限制，使用 alibaba 的免费代理，且其条款禁止生产使用。
- **API 密钥泄露** — 传递给 `new PageAgent({apiKey: ...})` 的任何密钥都会包含在你的 JS 包中。真实部署时始终通过你自己的后端代理。
- **非 OpenAI 兼容端点**会静默失败或产生晦涩的错误。如果你的提供商需要原生 Anthropic/Gemini 格式，请在前端使用 OpenAI 兼容代理（LiteLLM、OpenRouter）。
- **CSP 阻止** — 具有严格 Content-Security-Policy 的网站可能拒绝加载 CDN 脚本或禁止内联 eval。在这种情况下，从你自己的域名自托管。
- **编辑 `.env` 后重启开发服务器** — Vite 只在启动时读取环境变量。
- **Node 版本** — 仓库声明 `^22.13.0 || >=24`。Node 20 会导致 `npm ci` 出现引擎错误。
- **npm 10 vs 11** — 文档说需要 npm 11+；npm 10.9 实际上可以正常工作。

## 参考

- 仓库：https://github.com/alibaba/page-agent
- 文档：https://alibaba.github.io/page-agent/
- 许可证：MIT（基于 browser-use 的 DOM 处理内部实现，Copyright 2024 Gregor Zunic）
