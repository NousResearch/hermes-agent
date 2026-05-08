---
title: "Popular Web Designs — 54个真实设计系统（Stripe, Linear, Vercel）作为HTML/CSS"
sidebar_label: "Popular Web Designs"
description: "54个真实设计系统（Stripe, Linear, Vercel）作为HTML/CSS"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Popular Web Designs

54个可直接使用的真实世界设计系统。每个模板捕捉一个
网站的完整视觉语言：调色板、排版层级、组件样式、间距
系统、阴影、响应式行为，以及带确切CSS值的实用代理提示。

## 相关设计技能

- **`claude-design`** — 用于设计*过程和品味*（界定简报、
  产生变体、验证本地HTML产物、避免AI设计slop）。
  当想要深思熟虑的设计页面以已知品牌样式为风格时，
  与此技能配对使用：`claude-design`驱动工作流，此技能提供
  视觉词汇。
- **`design-md`** — 当交付物是正式的DESIGN.md令牌规范
  文件，而不是渲染的产物时使用。

## 如何使用

1. 从下面的目录中选取一个设计
2. 加载它：`skill_view(name="popular-web-designs", file_path="templates/<site>.md")`
3. 生成HTML时使用设计令牌和组件规范
4. 与`generative-widgets`技能配对以通过cloudflared隧道提供结果

每个模板在顶部包含一个**Hermes实施说明**块，其中：
- CDN字体替代和Google Fonts `<link>`标签（可直接粘贴）
- 主要和等宽字体的CSS字体堆栈
- 使用`write_file`创建HTML和`browser_vision`进行验证的提醒

## HTML生成模式

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
  <!-- 从模板的Hermes说明中粘贴Google Fonts <link> -->
  <link href="https://fonts.googleapis.com/css2?family=..." rel="stylesheet">
  <style>
    /* 将模板的调色板应用为CSS自定义属性 */
    :root {
      --color-bg: #ffffff;
      --color-text: #171717;
      --color-accent: #533afd;
      /* ... 更多来自模板第2节的内容 */
    }
    /* 从模板第3节应用排版 */
    body {
      font-family: 'Inter', system-ui, sans-serif;
      color: var(--color-text);
      background: var(--color-bg);
    }
    /* 从模板第4节应用组件样式 */
    /* 从模板第5节应用布局 */
    /* 从模板第6节应用阴影 */
  </style>
</head>
<body>
  <!-- 使用模板中的组件规范进行构建 -->
</body>
</html>
```

使用`write_file`写入文件，使用`generative-widgets`工作流（cloudflared隧道）提供，
并使用`browser_vision`验证结果以确认视觉保真度。

## 字体替代参考

大多数网站使用无法通过CDN获得的专有字体。每个模板映射到
保留设计特征的Google Fonts替代字体。常见映射：

| 专有字体 | CDN替代字体 | 特征 |
|---|---|---|
| Geist / Geist Sans | Geist（在Google Fonts上） | 几何、压缩字距 |
| Geist Mono | Geist Mono（在Google Fonts上） | 简洁等宽、连字 |
| sohne-var (Stripe) | Source Sans 3 | 光重优雅 |
| Berkeley Mono | JetBrains Mono | 技术等宽 |
| Airbnb Cereal VF | DM Sans | 圆润、友好的几何 |
| Circular (Spotify) | DM Sans | 几何、温暖 |
| figmaSans | Inter | 简洁人文 |
| Pin Sans (Pinterest) | DM Sans | 友好、圆润 |
| NVIDIA-EMEA | Inter（或Arial系统） | 工业、简洁 |
| CoinbaseDisplay/Sans | DM Sans | 几何、可信 |
| UberMove | DM Sans | 粗体、紧实 |
| HashiCorp Sans | Inter | 企业、中性 |
| waldenburgNormal (Sanity) | Space Grotesk | 几何、略压缩 |
| IBM Plex Sans/Mono | IBM Plex Sans/Mono | 在Google Fonts上可用 |
| Rubik (Sentry) | Rubik | 在Google Fonts上可用 |

当模板的CDN字体与原始字体匹配时（Inter、IBM Plex、Rubik、Geist），
不会发生替代损失。当使用替代字体时（DM Sans用于Circular、Source Sans 3
用于sohne-var），请密切遵循模板的字重、大小和字母间距值——
这些承载的视觉识别度比特定字体外观更多。

## 设计目录

### AI与机器学习

| 模板 | 网站 | 样式 |
|---|---|---|
| `claude.md` | Anthropic Claude | 温暖赤土强调色、简洁编辑布局 |
| `cohere.md` | Cohere | 活力渐变、数据丰富仪表板审美 |
| `elevenlabs.md` | ElevenLabs | 暗色电影UI、音频波形审美 |
| `minimax.md` | Minimax | 粗体暗色界面与霓虹强调色 |
| `mistral.ai.md` | Mistral AI | 法式工程极简主义、紫色调 |
| `ollama.md` | Ollama | 终端优先、单色简洁 |
| `opencode.ai.md` | OpenCode AI | 开发者中心暗色主题、全等宽 |
| `replicate.md` | Replicate | 简洁白色画布、代码优先 |
| `runwayml.md` | RunwayML | 电影感暗色UI、媒体丰富布局 |
| `together.ai.md` | Together AI | 技术、蓝图风格设计 |
| `voltagent.md` | VoltAgent | 虚空黑画布、翡翠强调色、终端原生 |
| `x.ai.md` | xAI | 纯粹单色、未来主义极简主义、全等宽 |

### 开发者工具与平台

| 模板 | 网站 | 样式 |
|---|---|---|
| `cursor.md` | Cursor | 圆滑暗色界面、渐变强调色 |
| `expo.md` | Expo | 暗色主题、紧实字母间距、以代码为中心 |
| `linear.app.md` | Linear | 超极简暗模式、精确、紫色强调色 |
| `lovable.md` | Lovable | 俏皮渐变、友好的开发者审美 |
| `mintlify.md` | Mintlify | 简洁、绿色强调、阅读优化 |
| `posthog.md` | PostHog | 俏皮品牌、开发者友好的暗色UI |
| `raycast.md` | Raycast | 圆滑暗色铬、活力渐变强调色 |
| `resend.md` | Resend | 极简暗色主题、等宽强调 |
| `sentry.md` | Sentry | 暗色仪表板、数据密集、粉紫强调色 |
| `supabase.md` | Supabase | 暗色翡翠主题、代码优先开发者工具 |
| `superhuman.md` | Superhuman | 高级暗色UI、键盘优先、紫色光晕 |
| `vercel.md` | Vercel | 黑色和白色精确、Geist字体系统 |
| `warp.md` | Warp | 暗色IDE风格界面、基于块的命令UI |
| `zapier.md` | Zapier | 温暖橙色、友好的插图驱动 |

### 基础设施与云

| 模板 | 网站 | 样式 |
|---|---|---|
| `clickhouse.md` | ClickHouse | 黄色强调、技术文档风格 |
| `composio.md` | Composio | 现代暗色与多彩集成图标 |
| `hashicorp.md` | HashiCorp | 企业简洁、黑色和白色 |
| `mongodb.md` | MongoDB | 绿色叶品牌、开发者文档重点 |
| `sanity.md` | Sanity | 红色强调、以内容为中心的编辑布局 |
| `stripe.md` | Stripe | 签名紫色渐变、字重300优雅 |

### 设计与生产力

| 模板 | 网站 | 样式 |
|---|---|---|
| `airtable.md` | Airtable | 多彩、友好、结构化数据审美 |
| `cal.md` | Cal.com | 简洁中性UI、开发者导向简洁 |
| `clay.md` | Clay | 有机形状、柔和渐变、艺术导向布局 |
| `figma.md` | Figma | 活力多色、俏皮但专业 |
| `framer.md` | Framer | 粗体黑色和蓝色、运动优先、设计优先 |
| `intercom.md` | Intercom | 友好蓝色调色板、对话式UI模式 |
| `miro.md` | Miro | 明亮黄色强调、无限画布审美 |
| `notion.md` | Notion | 温暖极简主义、衬线标题、柔和表面 |
| `pinterest.md` | Pinterest | 红色强调、砌体网格、以图像为先的布局 |
| `webflow.md` | Webflow | 蓝色强调、打磨过的营销网站审美 |

### 金融科技与加密货币

| 模板 | 网站 | 样式 |
|---|---|---|
| `coinbase.md` | Coinbase | 简洁蓝色标识、信任聚焦、机构感 |
| `kraken.md` | Kraken | 紫色强调暗色UI、数据密集仪表板 |
| `revolut.md` | Revolut | 圆滑暗色界面、渐变卡片、金融科技精度 |
| `wise.md` | Wise | 明亮绿色强调、友好而清晰 |

### 企业与消费者

| 模板 | 网站 | 样式 |
|---|---|---|
| `airbnb.md` | Airbnb | 温暖珊瑚强调色、摄影驱动、圆角UI |
| `apple.md` | Apple | 高级白色空间、SF Pro、电影感图像 |
| `bmw.md` | BMW | 暗色高级表面、精确工程审美 |
| `ibm.md` | IBM | Carbon设计系统、结构化蓝色调色板 |
| `nvidia.md` | NVIDIA | 绿黑能量、技术力量审美 |
| `spacex.md` | SpaceX | 纯粹黑白、全出血图像、未来主义 |
| `spotify.md` | Spotify | 活力绿色在暗色上、粗体类型、专辑艺术驱动 |
| `uber.md` | Uber | 粗体黑白、紧实字体、城市能量 |

## 选择设计

将设计与内容匹配：

- **开发者工具/仪表板：** Linear、Vercel、Supabase、Raycast、Sentry
- **文档/内容网站：** Mintlify、Notion、Sanity、MongoDB
- **营销/着陆页：** Stripe、Framer、Apple、SpaceX
- **暗色模式UIs：** Linear、Cursor、ElevenLabs、Warp、Superhuman
- **明亮/简洁UIs：** Vercel、Stripe、Notion、Cal.com、Replicate
- **俏皮/友好：** PostHog、Figma、Lovable、Zapier、Miro
- **高级/奢侈品：** Apple、BMW、Stripe、Superhuman、Revolut
- **数据密集/仪表板：** Sentry、Kraken、Cohere、ClickHouse
- **等宽/终端审美：** Ollama、OpenCode、x.ai、VoltAgent |
