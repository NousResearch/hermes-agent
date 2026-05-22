---
sidebar_position: 17
title: "扩展仪表板"
description: "为 Hermes Web 仪表板构建主题和插件 —— 调色板、排版、布局、自定义标签页、Shell 插槽、页面作用域插槽和后端 API 路由"
---

# 扩展仪表板

Hermes Web 仪表板（`hermes dashboard`）旨在无需分叉代码库即可重新换肤和扩展。暴露三个层次：

1. **主题** —— YAML 文件，重新绘制仪表板的调色板、排版、布局和每个组件的 chrome。将文件放入 `~/.hermes/dashboard-themes/`；它会出现在主题切换器中。
2. **UI 插件** —— 一个包含 `manifest.json` + JavaScript 包的目录，注册标签页、替换内置页面、通过页面作用域插槽增强页面，或将组件注入命名 Shell 插槽。
3. **后端插件** —— 该插件目录内的 Python 文件，暴露 FastAPI `router`；路由挂载在 `/api/plugins/<name>/` 下，并从插件的 UI 调用。

这三个都是**运行时即插即用**：无需仓库克隆、无需 `npm run build`、无需修补仪表板源码。本页面是这三者的规范参考。

如果您只想使用仪表板，请参阅 [Web 仪表板](./web-dashboard)。如果您想重新换肤终端 CLI（而非 Web 仪表板），请参阅 [皮肤与主题](./skins) —— CLI 皮肤系统与仪表板主题无关。

:::note 各部分如何组合
主题和插件是独立但协同的。主题可以独立存在（仅一个 YAML 文件）。插件可以独立存在（仅一个标签页）。两者结合可让您构建完整的视觉换肤和自定义 HUD —— 捆绑的 `strike-freedom-cockpit` 演示正是这样做的。请参阅[组合主题 + 插件演示](#组合主题--插件演示)。
:::

---

## 目录

- [主题](#主题)
  - [快速开始 —— 您的第一个主题](#快速开始--您的第一个主题)
  - [调色板、排版、布局](#调色板排版布局)
  - [布局变体](#布局变体)
  - [主题资源（作为 CSS 变量的图片）](#主题资源作为-css-变量的图片)
  - [组件 chrome 覆盖](#组件-chrome-覆盖)
  - [颜色覆盖](#颜色覆盖)
  - [原始 `customCSS`](#原始-customcss)
  - [内置主题](#内置主题)
  - [完整主题 YAML 参考](#完整主题-yaml-参考)
- [插件](#插件)
  - [快速开始 —— 您的第一个插件](#快速开始--您的第一个插件)
  - [目录布局](#目录布局)
  - [清单参考](#清单参考)
  - [插件 SDK](#插件-sdk)
  - [Shell 插槽](#shell-插槽)
  - [替换内置页面（`tab.override`）](#替换内置页面-taboverride)
  - [增强内置页面（页面作用域插槽）](#增强内置页面页面作用域插槽)
  - [仅插槽插件（`tab.hidden`）](#仅插槽插件-tabhidden)
  - [后端 API 路由](#后端-api-路由)
  - [每个插件的自定义 CSS](#每个插件的自定义-css)
  - [插件发现与重载](#插件发现与重载)
- [组合主题 + 插件演示](#组合主题--插件演示)
- [API 参考](#api-参考)
- [故障排除](#故障排除)

---

## 主题

主题是存储在 `~/.hermes/dashboard-themes/` 中的 YAML 文件。文件名不重要（系统使用主题的 `name:` 字段），但惯例是 `<name>.yaml`。每个字段都是可选的 —— 缺失的键回退到内置的 `default` 主题，因此主题可以小到一个颜色。

### 快速开始 —— 您的第一个主题

```bash
mkdir -p ~/.hermes/dashboard-themes
```

```yaml
# ~/.hermes/dashboard-themes/neon.yaml
name: neon
label: Neon
description: 纯洋红色配黑色

palette:
  background: "#000000"
  midground: "#ff00ff"
```

刷新仪表板。点击标题栏中的调色板图标并选择 **Neon**。背景变为黑色，文本和强调色变为洋红色，每个派生颜色（卡片、边框、静音、ring 等）都通过 CSS 中的 `color-mix()` 从这个双色三元组重新计算。

这就是整个入门：一个文件，两种颜色。下面所有内容都是可选的细化。

### 调色板、排版、布局

这三个块是主题的核心。每个都是独立的 —— 覆盖一个，保留其他。

#### 调色板（3 层）

调色板是一个三层颜色三元组加上暖光晕影颜色和噪点颗粒乘数。仪表板的设计系统级联通过 CSS `color-mix()` 从这个三元组派生每个 shadcn 兼容的 token（卡片、弹出框、静音、边框、主色、破坏色、ring 等）。覆盖三种颜色会级联到整个 UI。

| 键 | 描述 |
|-----|-------------|
| `palette.background` | 最深的画布颜色 —— 通常接近黑色。驱动页面背景和卡片填充。 |
| `palette.midground` | 主文本和强调色。大多数 UI chrome 读取此值（前景文本、按钮轮廓、焦点 ring）。 |
| `palette.foreground` | 顶层高亮。默认主题将其设置为 alpha 0（不可见）；想要顶部明亮强调色的主题可以提高其 alpha。 |
| `palette.warmGlow` | `<Backdrop />` 使用的晕影颜色的 `rgba(...)` 字符串。 |
| `palette.noiseOpacity` | 0–1.2 的颗粒覆盖层乘数。越低越柔和，越高越粗糙。 |

每一层接受 `{hex: "#RRGGBB", alpha: 0.0–1.0}` 或裸十六进制字符串（alpha 默认为 1.0）。

```yaml
palette:
  background:
    hex: "#05091a"
    alpha: 1.0
  midground: "#d8f0ff"          # 裸十六进制，alpha = 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0                    # 不可见顶层
  warmGlow: "rgba(255, 199, 55, 0.24)"
  noiseOpacity: 0.7
```

#### 排版

| 键 | 类型 | 描述 |
|-----|------|-------------|
| `fontSans` | string | 正文复制的 CSS font-family 堆栈（应用于 `html`、`body`）。 |
| `fontMono` | string | 代码块、`<code>`、`.font-mono` 工具的 CSS font-family 堆栈。 |
| `fontDisplay` | string | 可选的标题/展示堆栈。回退到 `fontSans`。 |
| `fontUrl` | string | 可选的外部样式表 URL。在主题切换时作为 `<link rel="stylesheet">` 注入 `<head>`。同一 URL 永远不会注入两次。适用于 Google Fonts、Bunny Fonts、自托管 `@font-face` 表 —— 任何可链接的内容。 |
| `baseSize` | string | 根字体大小 —— 控制 rem 比例。例如 `"14px"`、`"16px"`。 |
| `lineHeight` | string | 默认行高。例如 `"1.5"`、`"1.65"`。 |
| `letterSpacing` | string | 默认字间距。例如 `"0"`、`"0.01em"`、`"-0.01em"`。 |

```yaml
typography:
  fontSans: '"Orbitron", "Eurostile", "Impact", sans-serif'
  fontMono: '"Share Tech Mono", ui-monospace, monospace'
  fontDisplay: '"Orbitron", "Eurostile", sans-serif'
  fontUrl: "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Share+Tech+Mono&display=swap"
  baseSize: "14px"
  lineHeight: "1.5"
  letterSpacing: "0.04em"
```

#### 布局

| 键 | 值 | 描述 |
|-----|--------|-------------|
| `radius` | 任何 CSS 长度（`"0"`、`"0.25rem"`、`"0.5rem"`、`"1rem"`、...） | 圆角 token。映射到 `--radius` 并级联到 `--radius-sm/md/lg/xl` —— 每个圆角元素一起移动。 |
| `density` | `compact` \| `comfortable` \| `spacious` | 作为 `--spacing-mul` CSS 变量应用的比例乘数。`compact = 0.85×`、`comfortable = 1.0×`（默认）、`spacious = 1.2×`。缩放 Tailwind 的基础间距，因此 padding、gap 和 space-between 工具都按比例移动。 |

```yaml
layout:
  radius: "0"
  density: compact
```

### 布局变体

`layoutVariant` 选择整体 Shell 布局。缺失时默认为 `"standard"。

| 变体 | 行为 |
|---------|-----------|
| `standard` | 单列，最大宽度 1600px（默认）。 |
| `cockpit` | 左侧边栏导轨（260px）+ 主内容。由插件通过 `sidebar` 插槽填充 —— 请参阅 [Shell 插槽](#shell-插槽)。没有插件时导轨显示占位符。 |
| `tiled` | 移除最大宽度限制，因此页面可以使用完整视口宽度。 |

```yaml
layoutVariant: cockpit
```

当前变体作为 `document.documentElement.dataset.layoutVariant` 暴露，因此 `customCSS` 中的原始 CSS 可以通过 `:root[data-layout-variant="cockpit"] ...` 定位它。

### 主题资源（作为 CSS 变量的图片）

与主题一起提供 artwork URL。每个命名插槽变为 CSS 变量（`--theme-asset-<name>`），内置 Shell 和任何插件都可以读取。`bg` 插槽自动连接到背景；其他插槽面向插件。

```yaml
assets:
  bg: "https://example.com/hero-bg.jpg"           # 自动接入 <Backdrop />
  hero: "/my-images/strike-freedom.png"           # 用于插件侧边栏
  crest: "/my-images/crest.svg"                   # 用于 header-left 插件
  logo: "/my-images/logo.png"
  sidebar: "/my-images/rail.png"
  header: "/my-images/header-art.png"
  custom:
    scanLines: "/my-images/scanlines.png"         # → --theme-asset-custom-scanLines
```

值接受：

- 裸 URL —— 自动包装在 `url(...)` 中。
- 预包装的 `url(...)`、`linear-gradient(...)`、`radial-gradient(...)` 表达式 —— 按原样使用。
- `"none"` —— 显式选择退出。

每个资源也会作为 `--theme-asset-<name>-raw`（未包装的 URL）发出，以防插件需要将其传递给 `<img src>` 而不是 `background-image`。

插件用普通 CSS 或 JS 读取它们：

```javascript
// 在插件插槽中
const hero = getComputedStyle(document.documentElement)
  .getPropertyValue("--theme-asset-hero").trim();
```

### 组件 chrome 覆盖

`componentStyles` 重新设计单个 Shell 组件的样式，无需编写 CSS 选择器。每个桶的条目变为 CSS 变量（`--component-<bucket>-<kebab-property>`），Shell 的共享组件读取。因此 `card:` 覆盖应用于每个 `<Card>`，`header:` 应用于应用栏等。

```yaml
componentStyles:
  card:
    clipPath: "polygon(12px 0, 100% 0, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0 100%, 0 12px)"
    background: "linear-gradient(180deg, rgba(10, 22, 52, 0.85), rgba(5, 9, 26, 0.92))"
    boxShadow: "inset 0 0 0 1px rgba(64, 200, 255, 0.28)"
  header:
    background: "linear-gradient(180deg, rgba(16, 32, 72, 0.95), rgba(5, 9, 26, 0.9))"
  tab:
    clipPath: "polygon(6px 0, 100% 0, calc(100% - 6px) 100%, 0 100%)"
  sidebar: {}
  backdrop: {}
  footer: {}
  progress: {}
  badge: {}
  page: {}
```

支持的桶：`card`、`header`、`footer`、`sidebar`、`tab`、`progress`、`badge`、`backdrop`、`page`。

属性名使用 camelCase（`clipPath`）并作为 kebab（`clip-path`）发出。值是普通 CSS 字符串 —— CSS 接受的任何内容（`clip-path`、`border-image`、`background`、`box-shadow`、`animation`、...）。

### 颜色覆盖

大多数主题不需要这个 —— 三层调色板派生每个 shadcn token。当您想要派生无法产生的特定强调色（柔和主题的柔和破坏红色、品牌的特定成功绿色）时使用 `colorOverrides`。

```yaml
colorOverrides:
  primary: "#ffce3a"
  primaryForeground: "#05091a"
  accent: "#3fd3ff"
  ring: "#3fd3ff"
  destructive: "#ff3a5e"
  border: "rgba(64, 200, 255, 0.28)"
```

支持的键：`card`、`cardForeground`、`popover`、`popoverForeground`、`primary`、`primaryForeground`、`secondary`、`secondaryForeground`、`muted`、`mutedForeground`、`accent`、`accentForeground`、`destructive`、`destructiveForeground`、`success`、`warning`、`border`、`input`、`ring`。

每个键 1:1 映射到 `--color-<kebab>` CSS 变量（例如 `primaryForeground` → `--color-primary-foreground`）。此处设置的任何键仅对活动主题覆盖调色板级联 —— 切换到另一个主题会清除覆盖。

### 原始 `customCSS`

对于 `componentStyles` 无法表达的选择器级 chrome —— 伪元素、动画、媒体查询、主题作用域覆盖 —— 将原始 CSS 放入 `customCSS`：

```yaml
customCSS: |
  /* 扫描线覆盖层 —— 仅在 cockpit 变体激活时可见。 */
  :root[data-layout-variant="cockpit"] body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 100;
    background: repeating-linear-gradient(to bottom,
      transparent 0px, transparent 2px,
      rgba(64, 200, 255, 0.035) 3px, rgba(64, 200, 255, 0.035) 4px);
    mix-blend-mode: screen;
  }
```

CSS 在主题应用时作为单个作用域 `<style data-hermes-theme-css>` 标签注入，并在主题切换时清理。**每个主题上限 32 KiB。**

### 内置主题

每个内置主题都有自己的调色板、排版和布局 —— 切换会产生超出颜色的可见变化。

| 主题 | 调色板 | 排版 | 布局 |
|-------|---------|------------|--------|
| **Hermes Teal** (`default`) | 深青色 + 奶油色 | 系统堆栈，15px | 0.5rem 圆角，comfortable |
| **Hermes Teal (Large)** (`default-large`) | 与默认相同 | 系统堆栈，18px，行高 1.65 | 0.5rem 圆角，spacious |
| **Midnight** (`midnight`) | 深蓝紫色 | Inter + JetBrains Mono，14px | 0.75rem 圆角，comfortable |
| **Ember** (`ember`) | 暖深红 + 青铜色 | Spectral（衬线）+ IBM Plex Mono，15px | 0.25rem 圆角，comfortable |
| **Mono** (`mono`) | 灰度 | IBM Plex Sans + IBM Plex Mono，13px | 0 圆角，compact |
| **Cyberpunk** (`cyberpunk`) | 霓虹绿配黑色 | 到处 Share Tech Mono，14px | 0 圆角，compact |
| **Rosé** (`rose`) | 粉色 + 象牙色 | Fraunces（衬线）+ DM Mono，16px | 1rem 圆角，spacious |

引用 Google Fonts 的主题（除 Hermes Teal 外）按需加载样式表 —— 第一次切换到它们时，`<link>` 标签会注入 `<head>`。

### 完整主题 YAML 参考

一个文件中的所有旋钮 —— 复制并修剪您不需要的：

```yaml
# ~/.hermes/dashboard-themes/ocean.yaml
name: ocean
label: Ocean Deep
description: 深海蓝色配珊瑚色强调

# 3 层调色板（接受 {hex, alpha} 或裸十六进制）
palette:
  background:
    hex: "#0a1628"
    alpha: 1.0
  midground:
    hex: "#a8d0ff"
    alpha: 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0.0
  warmGlow: "rgba(255, 107, 107, 0.35)"
  noiseOpacity: 0.7

typography:
  fontSans: "Poppins, system-ui, sans-serif"
  fontMono: "Fira Code, ui-monospace, monospace"
  fontDisplay: "Poppins, system-ui, sans-serif"   # 可选
  fontUrl: "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap"
  baseSize: "15px"
  lineHeight: "1.6"
  letterSpacing: "-0.003em"

layout:
  radius: "0.75rem"
  density: comfortable

layoutVariant: standard        # standard | cockpit | tiled

assets:
  bg: "https://example.com/ocean-bg.jpg"
  hero: "/my-images/kraken.png"
  crest: "/my-images/anchor.svg"
  logo: "/my-images/logo.png"
  custom:
    pattern: "/my-images/waves.svg"

componentStyles:
  card:
    boxShadow: "inset 0 0 0 1px rgba(168, 208, 255, 0.18)"
  header:
    background: "linear-gradient(180deg, rgba(10, 22, 40, 0.95), rgba(5, 9, 26, 0.9))"

colorOverrides:
  destructive: "#ff6b6b"
  ring: "#ff6b6b"

customCSS: |
  /* 任何额外的选择器级调整 */
```

创建文件后刷新仪表板。从标题栏实时切换主题 —— 点击调色板图标。选择持久保存到 `config.yaml` 的 `dashboard.theme` 下，并在重新加载时恢复。

---

## 插件

仪表板插件是一个包含 `manifest.json`、预构建 JS 包以及可选 CSS 文件和带 FastAPI 路由的 Python 文件的目录。插件与其他 Hermes 插件一起位于 `~/.hermes/plugins/<name>/` —— 仪表板扩展是该插件目录内的 `dashboard/` 子文件夹，因此一个插件可以从单次安装同时扩展 CLI/网关和仪表板。

插件不打包 React 或 UI 组件。它们使用暴露在 `window.__HERMES_PLUGIN_SDK__` 上的**插件 SDK**。这使插件包保持小巧（通常几 KB）并避免版本冲突。

### 快速开始 —— 您的第一个插件

创建目录结构：

```bash
mkdir -p ~/.hermes/plugins/my-plugin/dashboard/dist
```

编写清单：

```json
// ~/.hermes/plugins/my-plugin/dashboard/manifest.json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills"
  },
  "entry": "dist/index.js"
}
```

编写 JS 包（普通 IIFE —— 无需构建步骤）：

```javascript
// ~/.hermes/plugins/my-plugin/dashboard/dist/index.js
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardHeader, CardTitle, CardContent } = SDK.components;

  function MyPage() {
    return React.createElement(Card, null,
      React.createElement(CardHeader, null,
        React.createElement(CardTitle, null, "My Plugin"),
      ),
      React.createElement(CardContent, null,
        React.createElement("p", { className: "text-sm text-muted-foreground" },
          "Hello from my custom dashboard tab.",
        ),
      ),
    );
  }

  window.__HERMES_PLUGINS__.register("my-plugin", MyPage);
})();
```

刷新仪表板 —— 您的标签页出现在导航栏中，在 **Skills** 之后。

:::tip 跳过 React.createElement
如果您更喜欢 JSX，请使用任何打包器（esbuild、Vite、rollup），将 React 作为外部依赖并以 IIFE 输出。唯一硬性要求是最终文件是可作为 `<script>` 加载的单个 JS 文件。React 从不打包；它来自 `SDK.React`。
:::

### 目录布局

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml              # 可选 —— 现有 CLI/网关插件清单
├── __init__.py              # 可选 —— 现有 CLI/网关钩子
└── dashboard/               # 仪表板扩展
    ├── manifest.json        # 必需 —— 标签页配置、图标、入口点
    ├── dist/
    │   ├── index.js         # 必需 —— 预构建 JS 包（IIFE）
    │   └── style.css        # 可选 —— 自定义 CSS
    └── plugin_api.py        # 可选 —— 后端 API 路由（FastAPI）
```

单个插件目录可以承载三个正交扩展：

- `plugin.yaml` + `__init__.py` —— CLI/网关插件（[请参阅插件页面](./plugins)）。
- `dashboard/manifest.json` + `dashboard/dist/index.js` —— 仪表板 UI 插件。
- `dashboard/plugin_api.py` —— 仪表板后端路由。

这些都不是必需的；只包含您需要的层。

### 清单参考

```json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "description": "What this plugin does",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills",
    "override": "/",
    "hidden": false
  },
  "slots": ["sidebar", "header-left"],
  "entry": "dist/index.js",
  "css": "dist/style.css",
  "api": "plugin_api.py"
}
```

| 字段 | 必需 | 描述 |
|-------|----------|-------------|
| `name` | 是 | 唯一插件标识符。小写，允许连字符。用于 URL 和注册。 |
| `label` | 是 | 导航标签页中显示的显示名称。 |
| `description` | 否 | 简短描述（在仪表板管理界面中显示）。 |
| `icon` | 否 | Lucide 图标名称。默认为 `Puzzle`。未知名称静默回退到 `Puzzle`。 |
| `version` | 否 | Semver 字符串。默认为 `0.0.0`。 |
| `tab.path` | 是 | 标签页的 URL 路径（例如 `/my-plugin`）。 |
| `tab.position` | 否 | 插入标签页的位置。`"end"`（默认）、`"after:<path>"` 或 `"before:<path>"` —— 冒号后的值是目标标签页的**路径段**（无前导斜杠）。示例：`"after:skills"`、`"before:config"`。 |
| `tab.override` | 否 | 设置为内置路由路径（`"/"`、`"/sessions"`、`"/config"`、...）以**替换**该页面而不是添加新标签页。请参阅[替换内置页面](#替换内置页面-taboverride)。 |
| `tab.hidden` | 否 | 为 true 时，注册组件和任何插槽，但不向导航添加标签页。由仅插槽插件使用。请参阅[仅插槽插件](#仅插槽插件-tabhidden)。 |
| `slots` | 否 | 此插件填充的命名 Shell 插槽。**仅文档辅助** —— 实际注册从 JS 包通过 `registerSlot()` 发生。在此处列出插槽使发现界面更具信息性。 |
| `entry` | 是 | 相对于 `dashboard/` 的 JS 包路径。默认为 `dist/index.js`。 |
| `css` | 否 | 要作为 `<link>` 标签注入的 CSS 文件路径。 |
| `api` | 否 | 带 FastAPI 路由的 Python 文件路径。挂载在 `/api/plugins/<name>/`。 |

#### 可用图标

插件使用 Lucide 图标名称。仪表板按名称映射它们 —— 未知名称静默回退到 `Puzzle`。

当前映射：`Activity`、`BarChart3`、`Clock`、`Code`、`Database`、`Eye`、`FileText`、`Globe`、`Heart`、`KeyRound`、`MessageSquare`、`Package`、`Puzzle`、`Settings`、`Shield`、`Sparkles`、`Star`、`Terminal`、`Wrench`、`Zap`。

需要不同的图标？向 `web/src/App.tsx` 的 `ICON_MAP` 提交 PR —— 纯增量更改。

### 插件 SDK

插件所需的一切都在 `window.__HERMES_PLUGIN_SDK__` 上。插件不应直接导入 React。

```javascript
const SDK = window.__HERMES_PLUGIN_SDK__;

// React + hooks
SDK.React                    // React 实例
SDK.hooks.useState
SDK.hooks.useEffect
SDK.hooks.useCallback
SDK.hooks.useMemo
SDK.hooks.useRef
SDK.hooks.useContext
SDK.hooks.createContext

// UI 组件（shadcn/ui 原语）
SDK.components.Card
SDK.components.CardHeader
SDK.components.CardTitle
SDK.components.CardContent
SDK.components.Badge
SDK.components.Button
SDK.components.Input
SDK.components.Label
SDK.components.Select
SDK.components.SelectOption
SDK.components.Separator
SDK.components.Tabs
SDK.components.TabsList
SDK.components.TabsTrigger
SDK.components.PluginSlot    // 渲染命名插槽（对嵌套插件 UI 有用）

// Hermes API 客户端 + 原始 fetcher
SDK.api                      // 类型化客户端 —— getStatus、getSessions、getConfig、...
SDK.fetchJSON                // 自定义端点的原始 fetch（插件注册的路由）

// 工具
SDK.utils.cn                 // Tailwind 类合并器（clsx + twMerge）
SDK.utils.timeAgo            // 从 unix 时间戳显示 "5m ago"
SDK.utils.isoTimeAgo         // 从 ISO 字符串显示 "5m ago"

// Hooks
SDK.useI18n                  // 多语言插件的 i18n hook
```

#### 调用插件的后端

```javascript
SDK.fetchJSON("/api/plugins/my-plugin/data")
  .then((data) => console.log(data))
  .catch((err) => console.error("API call failed:", err));
```

`fetchJSON` 注入会话认证令牌，将错误作为抛出异常显示，并自动解析 JSON。

#### 调用内置 Hermes 端点

```javascript
// 智能体状态
SDK.api.getStatus().then((s) => console.log("Version:", s.version));

// 最近会话
SDK.api.getSessions(10).then((resp) => console.log(resp.sessions.length));
```

完整列表请参阅 [Web 仪表板 → REST API](./web-dashboard#rest-api)。

### Shell 插槽

插槽让插件将组件注入应用 Shell 的命名位置 —— cockpit 侧边栏、标题栏、页脚、覆盖层 —— 无需声明整个标签页。多个插件可以填充同一插槽；它们按注册顺序堆叠渲染。

从插件包内部注册：

```javascript
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sidebar", MySidebar);
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "header-left", MyCrest);
```

#### 插槽目录

**Shell 范围插槽**（在应用 chrome 的任何位置渲染）：

| 插槽 | 位置 |
|------|----------|
| `backdrop` | 在 `<Backdrop />` 层堆栈内部，噪点层上方。 |
| `header-left` | 顶部栏中 Hermes 品牌之前。 |
| `header-right` | 顶部栏中主题/语言切换器之前。 |
| `header-banner` | 导航下方的全宽条带。 |
| `sidebar` | Cockpit 侧边栏导轨 —— **仅在 `layoutVariant === "cockpit"` 时渲染**。 |
| `pre-main` | 路由出口上方（在 `<main>` 内部）。 |
| `post-main` | 路由出口下方（在 `<main>` 内部）。 |
| `footer-left` | 页脚单元格内容（替换默认）。 |
| `footer-right` | 页脚单元格内容（替换默认）。 |
| `overlay` | 所有内容上方的固定定位层。适用于 chrome（扫描线、晕影），`customCSS` 单独无法实现。 |

**页面作用域插槽**（仅在命名内置页面上渲染 —— 使用这些来将 widget、卡片或工具栏注入现有页面，无需覆盖整个路由）：

| 插槽 | 渲染位置 |
|------|------------------|
| `sessions:top` / `sessions:bottom` | `/sessions` 页面的顶部 / 底部。 |
| `analytics:top` / `analytics:bottom` | `/analytics` 页面的顶部 / 底部。 |
| `logs:top` / `logs:bottom` | `/logs` 的顶部（筛选工具栏上方）/ 底部（日志查看器下方）。 |
| `cron:top` / `cron:bottom` | `/cron` 页面的顶部 / 底部。 |
| `skills:top` / `skills:bottom` | `/skills` 页面的顶部 / 底部。 |
| `config:top` / `config:bottom` | `/config` 页面的顶部 / 底部。 |
| `env:top` / `env:bottom` | `/env`（Keys）页面的顶部 / 底部。 |
| `docs:top` / `docs:bottom` | `/docs` 的顶部（iframe 上方）/ 底部。 |
| `chat:top` / `chat:bottom` | `/chat` 的顶部 / 底部（仅在嵌入聊天启用时激活）。 |

示例 —— 在 Sessions 页面顶部添加横幅卡片：

```javascript
function PinnedSessionsBanner() {
  return React.createElement(Card, null,
    React.createElement(CardContent, { className: "py-2 text-xs" },
      "Pinned note injected by my-plugin"),
  );
}

window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sessions:top", PinnedSessionsBanner);
```

将页面作用域插槽与 `tab.hidden: true` 结合，如果您的插件仅增强现有页面且不需要自己的侧边栏标签页。

Shell 仅对上述插槽渲染 `<PluginSlot name="..." />`。其他名称被注册表接受用于嵌套插件 UI —— 插件可以通过 `SDK.components.PluginSlot` 暴露自己的插槽。

#### 重新注册和 HMR

如果相同的 `(plugin, slot)` 对被注册两次，后面的调用替换前面的 —— 这与 React HMR 期望的插件重新挂载行为匹配。

### 替换内置页面（`tab.override`） {#替换内置页面-taboverride}

将 `tab.override` 设置为内置路由路径会使插件的组件替换该页面，而不是添加新标签页。当主题想要自定义主页（`/`）但想保留仪表板其余部分时很有用。

```json
{
  "name": "my-home",
  "label": "Home",
  "tab": {
    "path": "/my-home",
    "override": "/",
    "position": "end"
  },
  "entry": "dist/index.js"
}
```

设置 `override` 时：

- `/` 处的原始页面组件从路由器中移除。
- 您的插件在 `/` 处渲染。
- 不为 `tab.path` 添加导航标签页（override 就是目的）。

只有一个插件可以覆盖给定路径。如果两个插件声明相同的覆盖，第一个获胜，第二个被忽略并附带开发模式警告。

如果您只需要向现有页面添加卡片或工具栏而不接管它，请改用[页面作用域插槽](#增强内置页面页面作用域插槽)。

### 增强内置页面（页面作用域插槽）

通过 `tab.override` 的完整替换很重 —— 您的插件现在拥有整个页面，包括我们将来发布的任何更新。大多数时候您只想向现有页面添加横幅、卡片或工具栏。这就是**页面作用域插槽**的用途。

每个内置页面在其内容区域的顶部和底部暴露 `<page>:top` 和 `<page>:bottom` 插槽。您的插件通过调用 `registerSlot()` 填充一个 —— 内置页面正常工作，您的组件在其旁边渲染。

可用插槽：`sessions:*`、`analytics:*`、`logs:*`、`cron:*`、`skills:*`、`config:*`、`env:*`、`docs:*`、`chat:*`（每个都有 `:top` 和 `:bottom`）。完整目录请参阅 [Shell 插槽 → 插槽目录](#插槽目录)。

最小示例 —— 将横幅固定到 Sessions 页面顶部：

```json
// ~/.hermes/plugins/session-notes/dashboard/manifest.json
{
  "name": "session-notes",
  "label": "Session Notes",
  "tab": { "path": "/session-notes", "hidden": true },
  "slots": ["sessions:top"],
  "entry": "dist/index.js"
}
```

```javascript
// ~/.hermes/plugins/session-notes/dashboard/dist/index.js
(function () {
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardContent } = SDK.components;

  function Banner() {
    return React.createElement(Card, null,
      React.createElement(CardContent, { className: "py-2 text-xs" },
        "Remember to label important sessions before archiving."),
    );
  }

  // 隐藏标签页的占位符。
  window.__HERMES_PLUGINS__.register("session-notes", function () { return null; });

  // 真正的工作。
  window.__HERMES_PLUGINS__.registerSlot("session-notes", "sessions:top", Banner);
})();
```

关键点：

- `tab.hidden: true` 将插件保留在侧边栏外 —— 它没有独立页面。
- `slots` 清单字段仅用于文档。实际绑定在 JS 包中通过 `registerSlot()` 发生。
- 多个插件可以声明相同的页面作用域插槽。它们按注册顺序堆叠渲染。
- 没有插件注册时零占用：内置页面完全按原样渲染。

参考插件（[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/example-dashboard) 中的 `example-dashboard`）提供了一个实时演示，将横幅注入 `sessions:top` —— 安装它以端到端查看该模式。

### 仅插槽插件（`tab.hidden`） {#仅插槽插件-tabhidden}

当 `tab.hidden: true` 时，插件注册其组件（用于直接 URL 访问）和任何插槽，但从不向导航添加标签页。由仅用于注入插槽的插件使用 —— 标题纹章、侧边栏 HUD、覆盖层。

```json
{
  "name": "header-crest",
  "label": "Header Crest",
  "tab": {
    "path": "/header-crest",
    "position": "end",
    "hidden": true
  },
  "slots": ["header-left"],
  "entry": "dist/index.js"
}
```

包仍然用占位符组件调用 `register()`（以防有人直接访问 URL），然后 `registerSlot()` 做真正的工作。

### 后端 API 路由

插件可以通过在清单中设置 `api` 来注册 FastAPI 路由。创建文件并导出 `router`：

```python
# ~/.hermes/plugins/my-plugin/dashboard/plugin_api.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/data")
async def get_data():
    return {"items": ["one", "two", "three"]}

@router.post("/action")
async def do_action(body: dict):
    return {"ok": True, "received": body}
```

路由挂载在 `/api/plugins/<name>/` 下，因此上述变为：

- `GET  /api/plugins/my-plugin/data`
- `POST /api/plugins/my-plugin/action`

插件 API 路由绕过会话令牌认证，因为仪表板服务器默认绑定到 localhost。**如果您运行不受信任的插件，不要将仪表板暴露在公共接口上（`--host 0.0.0.0`）** —— 它们的路由也会变得可访问。

#### 访问 Hermes 内部

后端路由在仪表板进程内部运行，因此它们可以直接从 hermes-agent 代码库导入：

```python
from fastapi import APIRouter
from hermes_state import SessionDB
from hermes_cli.config import load_config

router = APIRouter()

@router.get("/session-count")
async def session_count():
    db = SessionDB()
    try:
        count = len(db.list_sessions(limit=9999))
        return {"count": count}
    finally:
        db.close()

@router.get("/config-snapshot")
async def config_snapshot():
    cfg = load_config()
    return {"model": cfg.get("model", {})}
```

### 每个插件的自定义 CSS

如果您的插件需要 Tailwind 类和内联 `style=` 之外的样式，请添加 CSS 文件并在清单中引用它：

```json
{
  "css": "dist/style.css"
}
```

文件作为 `<link>` 标签在插件加载时注入。使用特定的类名以避免与仪表板样式冲突，并引用仪表板的 CSS 变量以保持主题感知：

```css
/* dist/style.css */
.my-plugin-chart {
  border: 1px solid var(--color-border);
  background: var(--color-card);
  color: var(--color-card-foreground);
  padding: 1rem;
}
.my-plugin-chart:hover {
  border-color: var(--color-ring);
}
```

仪表板将每个 shadcn token 暴露为 `--color-*` 加上主题额外项（`--theme-asset-*`、`--component-<bucket>-*`、`--radius`、`--spacing-mul`）。引用这些，您的插件会随着活动主题自动换肤。

### 插件发现与重载

仪表板扫描三个目录以查找 `dashboard/manifest.json`：

| 优先级 | 目录 | 来源标签 |
|----------|-----------|--------------|
| 1（冲突时获胜） | `~/.hermes/plugins/<name>/dashboard/` | `user` |
| 2 | `<repo>/plugins/memory/<name>/dashboard/` | `bundled` |
| 2 | `<repo>/plugins/<name>/dashboard/` | `bundled` |
| 3 | `./.hermes/plugins/<name>/dashboard/` | `project` —— 仅在设置 `HERMES_ENABLE_PROJECT_PLUGINS` 时 |

发现结果按仪表板进程缓存。添加新插件后，要么：

```bash
# 强制重新扫描而不重启
curl http://127.0.0.1:9119/api/dashboard/plugins/rescan
```

…要么重启 `hermes dashboard`。

#### 插件加载生命周期

1. 仪表板加载。`main.tsx` 在 `window.__HERMES_PLUGIN_SDK__` 上暴露 SDK，在 `window.__HERMES_PLUGINS__` 上暴露注册表。
2. `App.tsx` 调用 `usePlugins()` → 获取 `GET /api/dashboard/plugins`。
3. 对于每个清单：注入 CSS `<link>`（如果声明），然后 `<script>` 标签加载 JS 包。
4. 插件的 IIFE 运行并调用 `window.__HERMES_PLUGINS__.register(name, Component)` —— 并可选地对每个插槽调用 `.registerSlot(name, slot, Component)`。
5. 仪表板将注册的组件解析到清单，将标签页添加到导航（除非 `hidden`），并将组件挂载为路由。

插件在脚本加载后有最多 **2 秒** 调用 `register()`。之后仪表板停止等待并完成初始渲染。如果插件稍后注册，它仍然会出现 —— 导航是反应式的。

如果插件的脚本加载失败（404、语法错误、IIFE 期间异常），仪表板会向浏览器控制台记录警告并继续运行而不加载它。

---

## 组合主题 + 插件演示

[`strike-freedom-cockpit`](https://github.com/NousResearch/hermes-example-plugins/tree/main/strike-freedom-cockpit) 插件（配套仓库 `hermes-example-plugins`）是一个完整的换肤演示。它将主题 YAML 与仅插槽插件配对，无需分叉仪表板即可生成 cockpit 风格的 HUD。

**它演示了什么：**

- 使用调色板、排版、`fontUrl`、`layoutVariant: cockpit`、`assets`、`componentStyles`（凹口卡片角、渐变背景）、`colorOverrides` 和 `customCSS`（扫描线覆盖层）的完整主题。
- 注册到三个插槽的仅插槽插件（`tab.hidden: true`）：
  - `sidebar` —— 由 `SDK.api.getStatus()` 驱动的实时遥测条的 MS-STATUS 面板。
  - `header-left` —— 从活动主题读取 `--theme-asset-crest` 的派系纹章。
  - `footer-right` —— 替换默认组织行的自定义标语。
- 插件通过 CSS 变量读取主题提供的 artwork，因此切换主题会改变 hero/纹章而无需插件代码更改。

**安装：**

```bash
git clone https://github.com/NousResearch/hermes-example-plugins.git

# 主题
cp hermes-example-plugins/strike-freedom-cockpit/theme/strike-freedom.yaml \
   ~/.hermes/dashboard-themes/

# 插件
cp -r hermes-example-plugins/strike-freedom-cockpit ~/.hermes/plugins/
```

打开仪表板，从主题切换器中选择 **Strike Freedom**。Cockpit 侧边栏出现，纹章显示在标题栏中，标语替换页脚。切换回 **Hermes Teal**，插件保持安装但不可见（`sidebar` 插槽仅在 `cockpit` 布局变体下渲染）。

阅读插件源码（配套仓库中的 `strike-freedom-cockpit/dashboard/dist/index.js`）以了解它如何读取 CSS 变量、防御没有插槽支持的旧仪表板，以及从一个包注册三个插槽。

---

## API 参考

### 主题端点

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/api/dashboard/themes` | GET | 列出可用主题 + 活动名称。内置返回 `{name, label, description}`；用户主题还包括带有完整规范化主题对象的 `definition` 字段。 |
| `/api/dashboard/theme` | PUT | 设置活动主题。正文：`{"name": "midnight"}`。持久保存到 `config.yaml` 的 `dashboard.theme` 下。 |

### 插件端点

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/api/dashboard/plugins` | GET | 列出发现的插件（带清单，减去内部字段）。 |
| `/api/dashboard/plugins/rescan` | GET | 强制重新扫描插件目录而不重启。 |
| `/dashboard-plugins/<name>/<path>` | GET | 从插件的 `dashboard/` 目录提供静态资源。路径遍历被阻止。 |
| `/api/plugins/<name>/*` | * | 插件注册的后端路由。 |

### `window` 上的 SDK

| 全局 | 类型 | 提供者 |
|--------|------|----------|
| `window.__HERMES_PLUGIN_SDK__` | object | `registry.ts` —— React、hooks、UI 组件、API 客户端、工具。 |
| `window.__HERMES_PLUGINS__.register(name, Component)` | function | 注册插件的主组件。 |
| `window.__HERMES_PLUGINS__.registerSlot(name, slot, Component)` | function | 注册到命名 Shell 插槽。 |

---

## 故障排除

**我的主题没有出现在选择器中。**
检查文件是否在 `~/.hermes/dashboard-themes/` 中并以 `.yaml` 或 `.yml` 结尾。刷新页面。运行 `curl http://127.0.0.1:9119/api/dashboard/themes` —— 您的主题应在响应中。如果 YAML 有解析错误，仪表板会记录到 `~/.hermes/logs/` 下的 `errors.log`。

**我的插件标签页没有显示。**
1. 检查清单是否在 `~/.hermes/plugins/<name>/dashboard/manifest.json`（注意 `dashboard/` 子目录）。
2. `curl http://127.0.0.1:9119/api/dashboard/plugins/rescan` 强制重新发现。
3. 打开浏览器开发者工具 → Network —— 确认 `manifest.json`、`index.js` 和任何 CSS 加载时没有 404。
4. 打开浏览器开发者工具 → Console —— 查找 IIFE 期间的错误或 `window.__HERMES_PLUGINS__ is undefined`（表示 SDK 未初始化，通常是之前的 React 渲染崩溃）。
5. 验证您的包使用与 `manifest.json:name` **相同**的名称调用 `window.__HERMES_PLUGINS__.register(...)`。

**插槽注册的组件没有渲染。**
`sidebar` 插槽仅在活动主题具有 `layoutVariant: cockpit` 时渲染。其他插槽始终渲染。如果您注册的插槽没有命中，在 `registerSlot` 内部添加 `console.log` 以确认插件包确实运行了。

**插件后端路由返回 404。**
1. 确认清单有 `"api": "plugin_api.py"` 指向 `dashboard/` 内的现有文件。
2. 重启 `hermes dashboard` —— 插件 API 路由在启动时挂载一次，**不在**重新扫描时挂载。
3. 检查 `plugin_api.py` 是否导出模块级 `router = APIRouter()`。其他导出名称不会被拾取。
4. 跟踪 `~/.hermes/logs/errors.log` 中的 `Failed to load plugin <name> API routes` —— 导入错误记录在那里。

**主题切换丢失了我的颜色覆盖。**
`colorOverrides` 限定在活动主题，并在主题切换时清除 —— 这是设计使然。如果您想要持久覆盖，请将它们放在主题的 YAML 中，而不是实时切换器中。

**主题 customCSS 被截断。**
`customCSS` 块每个主题上限为 32 KiB。将大样式表拆分到多个主题，或切换到通过其 `css` 字段注入完整样式表的插件（无大小限制）。

**我想在 PyPI 上发布插件。**
仪表板插件通过目录布局安装，而非 pip 入口点。今天最干净的发布路径是用户克隆到 `~/.hermes/plugins/` 的 git 仓库。基于 pip 的仪表板插件安装程序当前尚未接入。
