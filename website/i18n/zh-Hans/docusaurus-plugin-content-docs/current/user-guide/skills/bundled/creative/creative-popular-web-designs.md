---
title: "流行网页设计 — 54 个真实设计系统（Stripe、Linear、Vercel）的 HTML/CSS"
sidebar_label: "流行网页设计"
description: "54 个真实设计系统（Stripe、Linear、Vercel）的 HTML/CSS"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 流行网页设计

54 个真实设计系统（Stripe、Linear、Vercel）的 HTML/CSS。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/popular-web-designs` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent + Teknium（设计系统来源：VoltAgent/awesome-design-md） |
| 许可证 | MIT |
| 平台 | linux, macos, windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 流行网页设计

54 个可直接用于生成 HTML/CSS 的真实设计系统。每个模板都完整呈现了某个网站的视觉语言：色彩调色板、排版层级、组件样式、间距系统、阴影、响应式行为，以及包含精确 CSS 值的实用 agent prompt（提示词）。

## 相关设计 skill

- **`claude-design`** — 用于设计*流程与品味*（梳理需求、生成变体、验证本地 HTML 产物、避免 AI 设计陷阱）。当用户希望按照某个已知品牌风格设计页面时，可与本 skill 配合使用：`claude-design` 驱动工作流，本 skill 提供视觉词汇。
- **`design-md`** — 当交付物是正式的 DESIGN.md token（设计令牌）规范文件而非渲染产物时使用。

## 使用方法

1. 从下方目录中选择一个设计
2. 加载它：`skill_view(name="popular-web-designs", file_path="templates/<site>.md")`
3. 生成 HTML 时使用设计 token 和组件规范
4. 配合 `generative-widgets` skill，通过 cloudflared tunnel 提供服务

每个模板顶部都包含一个 **Hermes 实现说明** 块，内容包括：
- CDN 字体替代方案及 Google Fonts `<link>` 标签（可直接粘贴）
- 主字体和等宽字体的 CSS font-family 栈
- 提醒使用 `write_file` 创建 HTML 文件，使用 `browser_vision` 进行验证

## HTML 生成模式

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
  <!-- Paste the Google Fonts <link> from the template's Hermes notes -->
  <link href="https://fonts.googleapis.com/css2?family=..." rel="stylesheet">
  <style>
    /* Apply the template's color palette as CSS custom properties */
    :root {
      --color-bg: #ffffff;
      --color-text: #171717;
      --color-accent: #533afd;
      /* ... more from template Section 2 */
    }
    /* Apply typography from template Section 3 */
    body {
      font-family: 'Inter', system-ui, sans-serif;
      color: var(--color-text);
      background: var(--color-bg);
    }
    /* Apply component styles from template Section 4 */
    /* Apply layout from template Section 5 */
    /* Apply shadows from template Section 6 */
  </style>
</head>
<body>
  <!-- Build using component specs from the template -->
</body>
</html>
```

使用 `write_file` 写入文件，通过 `generative-widgets` 工作流（cloudflared tunnel）提供服务，并使用 `browser_vision` 验证结果以确认视觉准确性。

## 字体替代参考

大多数网站使用无法通过 CDN 获取的专有字体。每个模板都映射到一个 Google Fonts 替代字体，以保留设计的整体风格。常见映射关系：

| 专有字体 | CDN 替代字体 | 风格特征 |
|---|---|---|
| Geist / Geist Sans | Geist（Google Fonts 上可用） | 几何感，字距紧凑 |
| Geist Mono | Geist Mono（Google Fonts 上可用） | 简洁等宽，支持连字 |
| sohne-var (Stripe) | Source Sans 3 | 轻字重优雅感 |
| Berkeley Mono | JetBrains Mono | 技术感等宽字体 |
| Airbnb Cereal VF | DM Sans | 圆润、友好的几何风格 |
| Circular (Spotify) | DM Sans | 几何感，温暖 |
| figmaSans | Inter | 简洁人文主义风格 |
| Pin Sans (Pinterest) | DM Sans | 友好，圆润 |
| NVIDIA-EMEA | Inter（或 Arial 系统字体） | 工业感，简洁 |
| CoinbaseDisplay/Sans | DM Sans | 几何感，值得信赖 |
| UberMove | DM Sans | 粗犷，紧凑 |
| HashiCorp Sans | Inter | 企业级，中性 |
| waldenburgNormal (Sanity) | Space Grotesk | 几何感，略微压缩 |
| IBM Plex Sans/Mono | IBM Plex Sans/Mono | Google Fonts 上可用 |
| Rubik (Sentry) | Rubik | Google Fonts 上可用 |

当模板的 CDN 字体与原始字体一致时（Inter、IBM Plex、Rubik、Geist），不存在替代损失。当使用替代字体时（如用 DM Sans 替代 Circular，用 Source Sans 3 替代 sohne-var），请严格遵循模板中的字重、字号和字距值——这些参数承载的视觉识别度往往高于字体本身。

## 设计目录

### AI 与机器学习

| 模板 | 网站 | 风格 |
|---|---|---|
| `claude.md` | Anthropic Claude | 暖赤陶色强调色，简洁编辑排版 |
| `cohere.md` | Cohere | 鲜艳渐变，数据丰富的仪表盘美学 |
| `elevenlabs.md` | ElevenLabs | 暗色电影感 UI，音频波形美学 |
| `minimax.md` | Minimax | 带霓虹强调色的粗犷暗色界面 |
| `mistral.ai.md` | Mistral AI | 法式工程极简主义，紫色调 |
| `ollama.md` | Ollama | 终端优先，单色简约 |
| `opencode.ai.md` | OpenCode AI | 开发者向暗色主题，全等宽字体 |
| `replicate.md` | Replicate | 干净白色画布，代码优先 |
| `runwayml.md` | RunwayML | 电影感暗色 UI，媒体丰富布局 |
| `together.ai.md` | Together AI | 技术感，蓝图风格设计 |
| `voltagent.md` | VoltAgent | 纯黑画布，翠绿强调色，终端原生 |
| `x.ai.md` | xAI | 极简单色，未来主义，全等宽字体 |

### 开发者工具与平台

| 模板 | 网站 | 风格 |
|---|---|---|
| `cursor.md` | Cursor | 流畅暗色界面，渐变强调色 |
| `expo.md` | Expo | 暗色主题，紧凑字距，代码中心 |
| `linear.app.md` | Linear | 极简暗色模式，精准，紫色强调色 |
| `lovable.md` | Lovable | 活泼渐变，友好开发者美学 |
| `mintlify.md` | Mintlify | 简洁，绿色强调，阅读优化 |
| `posthog.md` | PostHog | 活泼品牌，开发者友好暗色 UI |
| `raycast.md` | Raycast | 流畅暗色外壳，鲜艳渐变强调色 |
| `resend.md` | Resend | 极简暗色主题，等宽字体强调 |
| `sentry.md` | Sentry | 暗色仪表盘，数据密集，粉紫强调色 |
| `supabase.md` | Supabase | 暗色翠绿主题，代码优先开发工具 |
| `superhuman.md` | Superhuman | 高端暗色 UI，键盘优先，紫色光晕 |
| `vercel.md` | Vercel | 黑白精准，Geist 字体系统 |
| `warp.md` | Warp | 暗色 IDE 风界面，块式命令 UI |
| `zapier.md` | Zapier | 暖橙色，友好插图驱动 |

### 基础设施与云

| 模板 | 网站 | 风格 |
|---|---|---|
| `clickhouse.md` | ClickHouse | 黄色强调，技术文档风格 |
| `composio.md` | Composio | 现代暗色，彩色集成图标 |
| `hashicorp.md` | HashiCorp | 企业级简洁，黑白配色 |
| `mongodb.md` | MongoDB | 绿叶品牌，开发者文档焦点 |
| `sanity.md` | Sanity | 红色强调，内容优先编辑布局 |
| `stripe.md` | Stripe | 标志性紫色渐变，300 字重优雅感 |

### 设计与生产力

| 模板 | 网站 | 风格 |
|---|---|---|
| `airtable.md` | Airtable | 多彩，友好，结构化数据美学 |
| `cal.md` | Cal.com | 简洁中性 UI，开发者向简约 |
| `clay.md` | Clay | 有机形状，柔和渐变，艺术指导布局 |
| `figma.md` | Figma | 鲜艳多色，活泼而专业 |
| `framer.md` | Framer | 粗犷黑蓝，动效优先，设计前沿 |
| `intercom.md` | Intercom | 友好蓝色调，对话式 UI 模式 |
| `miro.md` | Miro | 亮黄强调色，无限画布美学 |
| `notion.md` | Notion | 温暖极简，衬线标题，柔和表面 |
| `pinterest.md` | Pinterest | 红色强调，瀑布流网格，图片优先布局 |
| `webflow.md` | Webflow | 蓝色强调，精致营销站美学 |

### 金融科技与加密货币

| 模板 | 网站 | 风格 |
|---|---|---|
| `coinbase.md` | Coinbase | 简洁蓝色标识，信任导向，机构感 |
| `kraken.md` | Kraken | 紫色强调暗色 UI，数据密集仪表盘 |
| `revolut.md` | Revolut | 流畅暗色界面，渐变卡片，金融科技精准感 |
| `wise.md` | Wise | 亮绿强调色，友好清晰 |

### 企业与消费者

| 模板 | 网站 | 风格 |
|---|---|---|
| `airbnb.md` | Airbnb | 暖珊瑚强调色，摄影驱动，圆润 UI |
| `apple.md` | Apple | 高端留白，SF Pro，电影感图像 |
| `bmw.md` | BMW | 暗色高端表面，精准工程美学 |
| `ibm.md` | IBM | Carbon 设计系统，结构化蓝色调色板 |
| `nvidia.md` | NVIDIA | 绿黑能量感，技术力量美学 |
| `spacex.md` | SpaceX | 极简黑白，全出血图像，未来主义 |
| `spotify.md` | Spotify | 暗底鲜绿，粗犷字体，专辑封面驱动 |
| `uber.md` | Uber | 粗犷黑白，紧凑字体，都市能量 |

## 选择设计

根据内容匹配设计：

- **开发者工具 / 仪表盘：** Linear、Vercel、Supabase、Raycast、Sentry
- **文档 / 内容站点：** Mintlify、Notion、Sanity、MongoDB
- **营销 / 落地页：** Stripe、Framer、Apple、SpaceX
- **暗色模式 UI：** Linear、Cursor、ElevenLabs、Warp、Superhuman
- **浅色 / 简洁 UI：** Vercel、Stripe、Notion、Cal.com、Replicate
- **活泼 / 友好：** PostHog、Figma、Lovable、Zapier、Miro
- **高端 / 奢华：** Apple、BMW、Stripe、Superhuman、Revolut
- **数据密集 / 仪表盘：** Sentry、Kraken、Cohere、ClickHouse
- **等宽 / 终端美学：** Ollama、OpenCode、x.ai、VoltAgent