---
sidebar_position: 17
title: "扩展 Dashboard"
description: "为 Hermes Web Dashboard 构建主题和插件——调色板、字体排版、布局、自定义标签页、shell 插槽、页面级插槽以及后端 API 路由"
---

# 扩展 Dashboard

Hermes Web Dashboard（`hermes dashboard`）在设计上支持换肤和扩展，无需 fork 代码库。对外暴露三个层次：

1. **主题（Themes）** — YAML 文件，用于重绘 dashboard 的调色板、字体排版、布局以及各组件的外观。将文件放入 `~/.hermes/dashboard-themes/`，即可在主题切换器中看到它。
2. **UI 插件（UI plugins）** — 一个包含 `manifest.json` 和 JavaScript bundle 的目录，可注册标签页、替换内置页面、通过页面级插槽增强内置页面，或向命名 shell 插槽注入组件。
3. **后端插件（Backend plugins）** — 插件目录内的 Python 文件，暴露一个 FastAPI `router`；路由挂载在 `/api/plugins/<name>/` 下，由插件的 UI 调用。

三者均为**运行时即插即用**：无需克隆仓库、无需 `npm run build`、无需修改 dashboard 源码。本页是三者的权威参考文档。

如果只是想使用 dashboard，请参阅 [Web Dashboard](./web-dashboard)。如果想为终端 CLI（而非 Web Dashboard）换肤，请参阅 [Skins & Themes](./skins) —— CLI 皮肤系统与 dashboard 主题无关。

:::note 各部分如何组合
主题和插件相互独立，但可协同工作。主题可以单独使用（仅一个 YAML 文件）。插件也可以单独使用（仅一个标签页）。两者结合可构建带有自定义 HUD 的完整视觉换肤方案——内置的 `strike-freedom-cockpit` 演示正是如此。参见[主题 + 插件组合演示](#combined-theme--plugin-demo)。
:::

---

## 目录

- [主题](#themes)
  - [快速上手——你的第一个主题](#quick-start--your-first-theme)
  - [调色板、字体排版、布局](#palette-typography-layout)
  - [布局变体](#layout-variants)
  - [主题资源（图片作为 CSS 变量）](#theme-assets-images-as-css-vars)
  - [组件外观覆盖](#component-chrome-overrides)
  - [颜色覆盖](#color-overrides)
  - [原始 `customCSS`](#raw-customcss)
  - [内置主题](#built-in-themes)
  - [完整主题 YAML 参考](#full-theme-yaml-reference)
- [插件](#plugins)
  - [快速上手——你的第一个插件](#quick-start--your-first-plugin)
  - [目录结构](#directory-layout)
  - [Manifest 参考](#manifest-reference)
  - [Plugin SDK](#the-plugin-sdk)
  - [Shell 插槽](#shell-slots)
  - [替换内置页面（`tab.override`）](#replacing-built-in-pages-taboverride)
  - [增强内置页面（页面级插槽）](#augmenting-built-in-pages-page-scoped-slots)
  - [仅插槽插件（`tab.hidden`）](#slot-only-plugins-tabhidden)
  - [后端 API 路由](#backend-api-routes)
  - [插件自定义 CSS](#custom-css-per-plugin)
  - [插件发现与重载](#plugin-discovery--reload)
- [主题 + 插件组合演示](#combined-theme--plugin-demo)
- [API 参考](#api-reference)
- [故障排查](#troubleshooting)

---

## 主题

主题是存储在 `~/.hermes/dashboard-themes/` 中的 YAML 文件。文件名无关紧要（系统使用主题的 `name:` 字段），但惯例是 `<name>.yaml`。所有字段均为可选——缺失的键会回退到内置的 `default` 主题，因此一个主题可以只包含一个颜色。

### 快速上手——你的第一个主题

```bash
mkdir -p ~/.hermes/dashboard-themes
```

```yaml
# ~/.hermes/dashboard-themes/neon.yaml
name: neon
label: Neon
description: Pure magenta on black

palette:
  background: "#000000"
  midground: "#ff00ff"
```

刷新 dashboard。点击顶栏的调色板图标，选择 **Neon**。背景变为黑色，文字和强调色变为洋红色，所有派生颜色（card、border、muted、ring 等）均通过 CSS 的 `color-mix()` 从这两个颜色自动计算得出。

这就是全部入门流程：一个文件，两个颜色。以下内容均为可选的进阶配置。

### 调色板、字体排版、布局

这三个块是主题的核心。每个块相互独立——覆盖其中一个，其余保持不变。

#### 调色板（3 层）

调色板由三层颜色加一个暖光晕（warm-glow）颜色和一个噪点颗粒倍增器组成。Dashboard 的设计系统级联通过 CSS `color-mix()` 从这三层颜色派生出所有兼容 shadcn 的 token（card、popover、muted、border、primary、destructive、ring 等）。覆盖三个颜色即可级联影响整个 UI。

| 键 | 描述 |
|-----|-------------|
| `palette.background` | 最深的画布颜色——通常接近黑色。驱动页面背景和卡片填充。 |
| `palette.midground` | 主要文字和强调色。大多数 UI 外观读取此值（前景文字、按钮轮廓、焦点环）。 |
| `palette.foreground` | 顶层高亮色。默认主题将其设为 alpha 为 0 的白色（不可见）；需要顶层亮色强调的主题可提高其 alpha 值。 |
| `palette.warmGlow` | `rgba(...)` 字符串，用作 `<Backdrop />` 的晕光颜色。 |
| `palette.noiseOpacity` | 0–1.2 的颗粒叠加层倍增器。越低越柔和，越高越粗粝。 |

每层接受 `{hex: "#RRGGBB", alpha: 0.0–1.0}` 或裸十六进制字符串（alpha 默认为 1.0）。

```yaml
palette:
  background:
    hex: "#05091a"
    alpha: 1.0
  midground: "#d8f0ff"          # bare hex, alpha = 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0                    # invisible top layer
  warmGlow: "rgba(255, 199, 55, 0.24)"
  noiseOpacity: 0.7
```

#### 字体排版

| 键 | 类型 | 描述 |
|-----|------|-------------|
| `fontSans` | string | 正文的 CSS font-family 栈（应用于 `html`、`body`）。 |
| `fontMono` | string | 代码块、`<code>`、`.font-mono` 工具类的 CSS font-family 栈。 |
| `fontDisplay` | string | 可选的标题/展示字体栈。回退到 `fontSans`。 |
| `fontUrl` | string | 可选的外部样式表 URL。在主题切换时以 `<link rel="stylesheet">` 注入 `<head>`。相同 URL 不会重复注入。支持 Google Fonts、Bunny Fonts、自托管 `@font-face` 样式表——任何可链接的资源均可。 |
| `baseSize` | string | 根字体大小——控制 rem 比例。例如 `"14px"`、`"16px"`。 |
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
| `radius` | 任意 CSS 长度（`"0"`、`"0.25rem"`、`"0.5rem"`、`"1rem"` 等） | 圆角 token。映射到 `--radius` 并级联到 `--radius-sm/md/lg/xl`——所有圆角元素同步变化。 |
| `density` | `compact` \| `comfortable` \| `spacious` | 间距倍增器，以 `--spacing-mul` CSS 变量形式应用。`compact = 0.85×`，`comfortable = 1.0×`（默认），`spacious = 1.2×`。缩放 Tailwind 的基础间距，因此 padding、gap 和 space-between 工具类均按比例调整。 |

```yaml
layout:
  radius: "0"
  density: compact
```

### 布局变体

`layoutVariant` 选择整体 shell 布局。缺省时默认为 `"standard"`。

| 变体 | 行为 |
|---------|-----------|
| `standard` | 单列，最大宽度 1600px（默认）。 |
| `cockpit` | 左侧边栏轨道（260px）+ 主内容区。由插件通过 `sidebar` 插槽填充——参见 [Shell 插槽](#shell-slots)。没有插件时轨道显示占位符。 |
| `tiled` | 取消最大宽度限制，页面可使用完整视口宽度。 |

```yaml
layoutVariant: cockpit
```

当前变体通过 `document.documentElement.dataset.layoutVariant` 暴露，因此 `customCSS` 中的原始 CSS 可通过 `:root[data-layout-variant="cockpit"] ...` 定向匹配。

### 主题资源（图片作为 CSS 变量）

随主题附带图片 URL。每个命名插槽会成为一个 CSS 变量（`--theme-asset-<name>`），内置 shell 和任何插件均可读取。`bg` 插槽自动接入 backdrop；其他插槽面向插件开放。

```yaml
assets:
  bg: "https://example.com/hero-bg.jpg"           # auto-wired into <Backdrop />
  hero: "/my-images/strike-freedom.png"           # for plugin sidebars
  crest: "/my-images/crest.svg"                   # for header-left plugins
  logo: "/my-images/logo.png"
  sidebar: "/my-images/rail.png"
  header: "/my-images/header-art.png"
  custom:
    scanLines: "/my-images/scanlines.png"         # → --theme-asset-custom-scanLines
```

值接受：

- 裸 URL——自动包装为 `url(...)`。
- 已包装的 `url(...)`、`linear-gradient(...)`、`radial-gradient(...)` 表达式——直接使用。
- `"none"` ——明确禁用。

每个资源还会以 `--theme-asset-<name>-raw`（未包装的 URL）形式输出，以便插件需要将其传给 `<img src>` 而非 `background-image` 时使用。

插件通过普通 CSS 或 JS 读取这些变量：

```javascript
// In a plugin slot
const hero = getComputedStyle(document.documentElement)
  .getPropertyValue("--theme-asset-hero").trim();
```

### 组件外观覆盖

`componentStyles` 可在不编写 CSS 选择器的情况下重新设置各 shell 组件的样式。每个桶（bucket）的条目会成为 CSS 变量（`--component-<bucket>-<kebab-property>`），shell 的共享组件会读取这些变量。因此 `card:` 的覆盖应用于所有 `<Card>`，`header:` 应用于应用栏，以此类推。

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

属性名使用 camelCase（`clipPath`），输出为 kebab-case（`clip-path`）。值为纯 CSS 字符串——CSS 接受的任何内容均可（`clip-path`、`border-image`、`background`、`box-shadow`、`animation` 等）。

### 颜色覆盖

大多数主题不需要此功能——3 层调色板已派生出所有 shadcn token。当你需要派生无法产生的特定强调色时（例如柔和主题的更柔和的破坏性红色，或品牌专属的成功绿色），才使用 `colorOverrides`。

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

每个键与 `--color-<kebab>` CSS 变量一一对应（例如 `primaryForeground` → `--color-primary-foreground`）。此处设置的任何键仅对当前激活主题生效，切换到其他主题时覆盖会被清除。

### 原始 `customCSS`

对于 `componentStyles` 无法表达的选择器级外观——伪元素、动画、媒体查询、主题范围内的覆盖——可将原始 CSS 写入 `customCSS`：

```yaml
customCSS: |
  /* Scanline overlay — only visible when cockpit variant is active. */
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

CSS 在主题应用时以单个带作用域的 `<style data-hermes-theme-css>` 标签注入，主题切换时清除。**每个主题上限为 32 KiB。**

### 内置主题

每个内置主题都有自己的调色板、字体排版和布局——切换时产生的变化不仅限于颜色。

| 主题 | 调色板 | 字体排版 | 布局 |
|-------|---------|------------|--------|
| **Hermes Teal**（`default`） | 深青色 + 奶油色 | 系统字体栈，15px | 0.5rem 圆角，comfortable |
| **Hermes Teal (Large)**（`default-large`） | 同 default | 系统字体栈，18px，行高 1.65 | 0.5rem 圆角，spacious |
| **Midnight**（`midnight`） | 深蓝紫色 | Inter + JetBrains Mono，14px | 0.75rem 圆角，comfortable |
| **Ember**（`ember`） | 暖深红 + 古铜色 | Spectral（衬线）+ IBM Plex Mono，15px | 0.25rem 圆角，comfortable |
| **Mono**（`mono`） | 灰度 | IBM Plex Sans + IBM Plex Mono，13px | 0 圆角，compact |
| **Cyberpunk**（`cyberpunk`） | 黑底霓虹绿 | Share Tech Mono 全局，14px | 0 圆角，compact |
| **Rosé**（`rose`） | 粉色 + 象牙色 | Fraunces（衬线）+ DM Mono，16px | 1rem 圆角，spacious |

引用 Google Fonts 的主题（除 Hermes Teal 外均如此）会按需加载样式表——首次切换时会向 `<head>` 注入一个 `<link>` 标签。

### 完整主题 YAML 参考

所有配置项汇总在一个文件中——复制后删除不需要的部分：

```yaml
# ~/.hermes/dashboard-themes/ocean.yaml
name: ocean
label: Ocean Deep
description: Deep sea blues with coral accents

# 3-layer palette (accepts {hex, alpha} or bare hex)
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
  fontDisplay: "Poppins, system-ui, sans-serif"   # optional
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
  /* Any additional selector-level tweaks */
```

创建文件后刷新 dashboard。通过顶栏的调色板图标实时切换主题。选择结果会持久化到 `config.yaml` 的 `dashboard.theme` 下，并在重载时恢复。

---

## 插件

Dashboard 插件是一个包含 `manifest.json`、预构建 JS bundle，以及可选的 CSS 文件和带 FastAPI 路由的 Python 文件的目录。插件与其他 Hermes 插件一起存放在 `~/.hermes/plugins/<name>/`——dashboard 扩展是该插件目录内的 `dashboard/` 子文件夹，因此一个插件可以从单次安装中同时扩展 CLI/gateway 和 dashboard。

插件不打包 React 或 UI 组件，而是使用暴露在 `window.__HERMES_PLUGIN_SDK__` 上的 **Plugin SDK**。这使插件 bundle 保持极小体积（通常只有几 KB），并避免版本冲突。

### 快速上手——你的第一个插件

创建目录结构：

```bash
mkdir -p ~/.hermes/plugins/my-plugin/dashboard/dist
```

编写 manifest：

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

编写 JS bundle（普通 IIFE——无需构建步骤）：

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

刷新 dashboard——你的标签页出现在导航栏中，位于 **Skills** 之后。

:::tip 跳过 React.createElement
如果你偏好 JSX，可使用任意打包工具（esbuild、Vite、rollup），将 React 设为外部依赖并输出 IIFE 格式。唯一的硬性要求是最终文件是可通过 `<script>` 加载的单个 JS 文件。React 永远不会被打包进去；它来自 `SDK.React`。
:::

### 目录结构

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml              # optional — existing CLI/gateway plugin manifest
├── __init__.py              # optional — existing CLI/gateway hooks
└── dashboard/               # dashboard extension
    ├── manifest.json        # required — tab config, icon, entry point
    ├── dist/
    │   ├── index.js         # required — pre-built JS bundle (IIFE)
    │   └── style.css        # optional — custom CSS
    └── plugin_api.py        # optional — backend API routes (FastAPI)
```

单个插件目录可承载三个正交扩展：

- `plugin.yaml` + `__init__.py` — CLI/gateway 插件（[参见插件页面](./plugins)）。
- `dashboard/manifest.json` + `dashboard/dist/index.js` — dashboard UI 插件。
- `dashboard/plugin_api.py` — dashboard 后端路由。

三者均非必须；按需包含所需层次即可。

### Manifest 参考

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

| 字段 | 必填 | 描述 |
|-------|----------|-------------|
| `name` | 是 | 唯一插件标识符。小写，可用连字符。用于 URL 和注册。 |
| `label` | 是 | 导航标签页中显示的名称。 |
| `description` | 否 | 简短描述（显示在 dashboard 管理界面）。 |
| `icon` | 否 | Lucide 图标名称。默认为 `Puzzle`。未知名称回退到 `Puzzle`。 |
| `version` | 否 | Semver 字符串。默认为 `0.0.0`。 |
| `tab.path` | 是 | 标签页的 URL 路径（例如 `/my-plugin`）。 |
| `tab.position` | 否 | 标签页插入位置。`"end"`（默认）、`"after:<path>"` 或 `"before:<path>"`——冒号后的值是目标标签页的**路径段**（无前导斜杠）。例如：`"after:skills"`、`"before:config"`。 |
| `tab.override` | 否 | 设置为内置路由路径（`"/"`、`"/sessions"`、`"/config"` 等）以**替换**该页面，而非添加新标签页。参见[替换内置页面](#replacing-built-in-pages-taboverride)。 |
| `tab.hidden` | 否 | 为 true 时，注册组件和所有插槽，但不向导航添加标签页。用于仅插槽插件。参见[仅插槽插件](#slot-only-plugins-tabhidden)。 |
| `slots` | 否 | 此插件填充的命名 shell 插槽。**仅作文档说明**——实际注册通过 JS bundle 中的 `registerSlot()` 完成。在此列出插槽可使发现界面更具信息量。 |
| `entry` | 是 | 相对于 `dashboard/` 的 JS bundle 路径。默认为 `dist/index.js`。 |
| `css` | 否 | 以 `<link>` 标签注入的 CSS 文件路径。 |
| `api` | 否 | 包含 FastAPI 路由的 Python 文件路径。挂载在 `/api/plugins/<name>/`。 |

#### 可用图标

插件使用 Lucide 图标名称。Dashboard 按名称映射——未知名称静默回退到 `Puzzle`。

当前已映射：`Activity`、`BarChart3`、`Clock`、`Code`、`Database`、`Eye`、`FileText`、`Globe`、`Heart`、`KeyRound`、`MessageSquare`、`Package`、`Puzzle`、`Settings`、`Shield`、`Sparkles`、`Star`、`Terminal`、`Wrench`、`Zap`。

需要其他图标？向 `web/src/App.tsx` 的 `ICON_MAP` 提交 PR——纯增量修改。

### Plugin SDK

插件所需的一切均在 `window.__HERMES_PLUGIN_SDK__` 上。插件不应直接导入 React。

```javascript
const SDK = window.__HERMES_PLUGIN_SDK__;

// React + hooks
SDK.React                    // the React instance
SDK.hooks.useState
SDK.hooks.useEffect
SDK.hooks.useCallback
SDK.hooks.useMemo
SDK.hooks.useRef
SDK.hooks.useContext
SDK.hooks.createContext

// UI components (shadcn/ui primitives)
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
SDK.components.PluginSlot    // render a named slot (useful for nested plugin UIs)

// Hermes API client + raw fetcher
SDK.api                      // typed client — getStatus, getSessions, getConfig, ...
SDK.fetchJSON                // raw fetch for custom endpoints (plugin-registered routes)

// Utilities
SDK.utils.cn                 // Tailwind class merger (clsx + twMerge)
SDK.utils.timeAgo            // "5m ago" from unix timestamp
SDK.utils.isoTimeAgo         // "5m ago" from ISO string

// Hooks
SDK.useI18n                  // i18n hook for multi-language plugins
```

#### 调用插件的后端

```javascript
SDK.fetchJSON("/api/plugins/my-plugin/data")
  .then((data) => console.log(data))
  .catch((err) => console.error("API call failed:", err));
```

`fetchJSON` 会自动注入会话认证 token，将错误作为异常抛出，并自动解析 JSON。

#### 调用内置 Hermes 端点

```javascript
// Agent status
SDK.api.getStatus().then((s) => console.log("Version:", s.version));

// Recent sessions
SDK.api.getSessions(10).then((resp) => console.log(resp.sessions.length));
```

完整列表参见 [Web Dashboard → REST API](./web-dashboard#rest-api)。

### Shell 插槽

插槽（slot）允许插件向应用 shell 的命名位置注入组件——cockpit 侧边栏、顶栏、底栏、覆盖层——而无需占用整个标签页。多个插件可以填充同一个插槽；它们按注册顺序堆叠渲染。

在插件 bundle 内部注册：

```javascript
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sidebar", MySidebar);
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "header-left", MyCrest);
```

#### 插槽目录

**Shell 全局插槽**（在应用外壳的任意位置渲染）：

| 插槽 | 位置 |
|------|----------|
| `backdrop` | `<Backdrop />` 层叠栈内，噪点层之上。 |
| `header-left` | 顶栏 Hermes 品牌之前。 |
| `header-right` | 顶栏主题/语言切换器之前。 |
| `header-banner` | 导航栏下方的全宽条带。 |
| `sidebar` | Cockpit 侧边栏轨道——**仅在 `layoutVariant === "cockpit"` 时渲染**。 |
| `pre-main` | 路由出口之上（`<main>` 内部）。 |
| `post-main` | 路由出口之下（`<main>` 内部）。 |
| `footer-left` | 底栏单元格内容（替换默认内容）。 |
| `footer-right` | 底栏单元格内容（替换默认内容）。 |
| `overlay` | 位于所有内容之上的固定定位层。适用于 `customCSS` 无法单独实现的外观效果（扫描线、晕影等）。 |

**页面级插槽**（仅在指定内置页面上渲染——用于向现有页面注入小部件、卡片或工具栏，而无需覆盖整个路由）：

| 插槽 | 渲染位置 |
|------|------------------|
| `sessions:top` / `sessions:bottom` | `/sessions` 页面顶部 / 底部。 |
| `analytics:top` / `analytics:bottom` | `/analytics` 页面顶部 / 底部。 |
| `logs:top` / `logs:bottom` | `/logs` 顶部（过滤工具栏之上）/ 底部（日志查看器之下）。 |
| `cron:top` / `cron:bottom` | `/cron` 页面顶部 / 底部。 |
| `skills:top` / `skills:bottom` | `/skills` 页面顶部 / 底部。 |
| `config:top` / `config:bottom` | `/config` 页面顶部 / 底部。 |
| `env:top` / `env:bottom` | `/env`（Keys）页面顶部 / 底部。 |
| `docs:top` / `docs:bottom` | `/docs` 顶部（iframe 之上）/ 底部。 |
| `chat:top` / `chat:bottom` | `/chat` 顶部 / 底部（仅在启用嵌入式聊天时有效）。 |

示例——向 Sessions 页面顶部添加横幅卡片：

```javascript
function PinnedSessionsBanner() {
  return React.createElement(Card, null,
    React.createElement(CardContent, { className: "py-2 text-xs" },
      "Pinned note injected by my-plugin"),
  );
}

window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sessions:top", PinnedSessionsBanner);
```

如果插件只增强现有页面而不需要独立的侧边栏标签页，可将页面级插槽与 `tab.hidden: true` 结合使用。

Shell 只为上述插槽渲染 `<PluginSlot name="..." />`。注册表接受额外的名称用于嵌套插件 UI——插件可通过 `SDK.components.PluginSlot` 暴露自己的插槽。

#### 重复注册与 HMR

如果同一个 `(plugin, slot)` 对被注册两次，后一次调用会替换前一次——这与 React HMR 期望插件重新挂载时的行为一致。

### 替换内置页面（`tab.override`）

将 `tab.override` 设置为内置路由路径，可使插件组件替换该页面，而非添加新标签页。适用于主题希望自定义首页（`/`）但保留 dashboard 其余部分的场景。

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

设置 `override` 后：

- 路由器中 `/` 处的原始页面组件被移除。
- 你的插件改为在 `/` 处渲染。
- 不会为 `tab.path` 添加导航标签页（覆盖本身才是目的）。

每个路径只能有一个插件进行覆盖。如果两个插件声明相同的覆盖路径，第一个生效，第二个被忽略并在开发模式下输出警告。

如果只需要向现有页面添加卡片或工具栏而不完全接管它，请改用[页面级插槽](#augmenting-built-in-pages-page-scoped-slots)。

### 增强内置页面（页面级插槽）

通过 `tab.override` 完全替换页面代价较重——你的插件现在拥有整个页面，包括我们未来对其的所有更新。大多数情况下，你只是想向现有页面添加横幅、卡片或工具栏。这正是**页面级插槽**的用途。

每个内置页面都在其内容区域的顶部和底部暴露 `<page>:top` 和 `<page>:bottom` 插槽。你的插件通过调用 `registerSlot()` 填充其中一个——内置页面正常工作，你的组件在其旁边渲染。

可用插槽：`sessions:*`、`analytics:*`、`logs:*`、`cron:*`、`skills:*`、`config:*`、`env:*`、`docs:*`、`chat:*`（每个均有 `:top` 和 `:bottom`）。完整目录参见 [Shell 插槽 → 插槽目录](#slot-catalogue)。

最简示例——在 Sessions 页面顶部固定一个横幅：

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

  // Placeholder for the hidden tab.
  window.__HERMES_PLUGINS__.register("session-notes", function () { return null; });

  // The real work.
  window.__HERMES_PLUGINS__.registerSlot("session-notes", "sessions:top", Banner);
})();
```

要点：

- `tab.hidden: true` 使插件不出现在侧边栏——它没有独立页面。
- manifest 中的 `slots` 字段仅作文档说明。实际绑定通过 JS bundle 中的 `registerSlot()` 完成。
- 多个插件可以声明同一个页面级插槽。它们按注册顺序堆叠渲染。
- 无插件注册时零开销：内置页面与之前完全相同地渲染。

参考插件（[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/example-dashboard) 中的 `example-dashboard`）提供了一个向 `sessions:top` 注入横幅的实时演示——安装它可端到端了解该模式。

### 仅插槽插件（`tab.hidden`）

当 `tab.hidden: true` 时，插件注册其组件（用于直接 URL 访问）和所有插槽，但不向导航添加标签页。适用于仅用于注入插槽的插件——顶栏徽标、侧边栏 HUD、覆盖层。

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

Bundle 仍需调用带占位符组件的 `register()`（以防有人直接访问该 URL），然后调用 `registerSlot()` 完成实际工作。

### 后端 API 路由

插件可通过在 manifest 中设置 `api` 来注册 FastAPI 路由。创建文件并导出 `router`：

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

路由挂载在 `/api/plugins/<name>/` 下，因此上述路由变为：

- `GET  /api/plugins/my-plugin/data`
- `POST /api/plugins/my-plugin/action`

插件 API 路由绕过会话 token 认证，因为 dashboard 服务器默认绑定到 localhost。**如果运行不受信任的插件，请勿使用 `--host 0.0.0.0` 将 dashboard 暴露在公共接口上**——其路由也会变得可访问。

#### 访问 Hermes 内部模块

后端路由在 dashboard 进程内运行，因此可以直接从 hermes-agent 代码库导入：

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

### 插件自定义 CSS

如果插件需要超出 Tailwind 类和内联 `style=` 的样式，可添加 CSS 文件并在 manifest 中引用：

```json
{
  "css": "dist/style.css"
}
```

文件在插件加载时以 `<link>` 标签注入。使用特定类名以避免与 dashboard 样式冲突，并引用 dashboard 的 CSS 变量以保持主题感知：

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

Dashboard 将每个 shadcn token 暴露为 `--color-*`，以及主题额外变量（`--theme-asset-*`、`--component-<bucket>-*`、`--radius`、`--spacing-mul`）。引用这些变量后，你的插件会随激活主题自动换肤。

### 插件发现与重载

Dashboard 扫描三个目录中的 `dashboard/manifest.json`：

| 优先级 | 目录 | 来源标签 |
|----------|-----------|--------------|
| 1（冲突时优先） | `~/.hermes/plugins/<name>/dashboard/` | `user` |
| 2 | `<repo>/plugins/memory/<name>/dashboard/` | `bundled` |
| 2 | `<repo>/plugins/<name>/dashboard/` | `bundled` |
| 3 | `./.hermes/plugins/<name>/dashboard/` | `project`——仅在设置 `HERMES_ENABLE_PROJECT_PLUGINS` 时生效 |

发现结果在每个 dashboard 进程中缓存。添加新插件后，可以：

```bash
# Force a rescan without restart
curl http://127.0.0.1:9119/api/dashboard/plugins/rescan
```

……或重启 `hermes dashboard`。

#### 插件加载生命周期

1. Dashboard 加载。`main.tsx` 在 `window.__HERMES_PLUGIN_SDK__` 上暴露 SDK，在 `window.__HERMES_PLUGINS__` 上暴露注册表。
2. `App.tsx` 调用 `usePlugins()` → 获取 `GET /api/dashboard/plugins`。
3. 对于每个 manifest：注入 CSS `<link>`（如已声明），然后通过 `<script>` 标签加载 JS bundle。
4. 插件的 IIFE 运行并调用 `window.__HERMES_PLUGINS__.register(name, Component)`——以及可选的 `.registerSlot(name, slot, Component)` 用于每个插槽。
5. Dashboard 将注册的组件与 manifest 对应，将标签页添加到导航（除非 `hidden`），并将组件挂载为路由。

插件在脚本加载后最多有 **2 秒**时间调用 `register()`。超时后 dashboard 停止等待并完成初始渲染。如果插件之后才注册，它仍会出现——导航是响应式的。

如果插件脚本加载失败（404、语法错误、IIFE 执行期间抛出异常），dashboard 会向浏览器控制台输出警告并继续运行。

---

## 主题 + 插件组合演示

[`strike-freedom-cockpit`](https://github.com/NousResearch/hermes-example-plugins/tree/main/strike-freedom-cockpit) 插件（伴随仓库 `hermes-example-plugins`）是一个完整的换肤演示。它将主题 YAML 与仅插槽插件配对，在不 fork dashboard 的情况下生成驾驶舱风格的 HUD。

**演示内容：**

- 完整主题，使用调色板、字体排版、`fontUrl`、`layoutVariant: cockpit`、`assets`、`componentStyles`（切角卡片、渐变背景）、`colorOverrides` 和 `customCSS`（扫描线叠加）。
- 仅插槽插件（`tab.hidden: true`），注册到三个插槽：
  - `sidebar` — 带有由 `SDK.api.getStatus()` 驱动的实时遥测条的 MS-STATUS 面板。
  - `header-left` — 从激活主题读取 `--theme-asset-crest` 的派系徽标。
  - `footer-right` — 替换默认组织行的自定义标语。
- 插件通过 CSS 变量读取主题提供的图片，因此切换主题可在不修改插件代码的情况下更换英雄图/徽标。

**安装：**

```bash
git clone https://github.com/NousResearch/hermes-example-plugins.git

# Theme
cp hermes-example-plugins/strike-freedom-cockpit/theme/strike-freedom.yaml \
   ~/.hermes/dashboard-themes/

# Plugin
cp -r hermes-example-plugins/strike-freedom-cockpit ~/.hermes/plugins/
```

打开 dashboard，从主题切换器中选择 **Strike Freedom**。驾驶舱侧边栏出现，徽标显示在顶栏，标语替换底栏。切换回 **Hermes Teal**，插件仍然安装但不可见（`sidebar` 插槽仅在 `cockpit` 布局变体下渲染）。

阅读插件源码（伴随仓库中的 `strike-freedom-cockpit/dashboard/dist/index.js`），了解它如何读取 CSS 变量、防范不支持插槽的旧版 dashboard，以及如何从单个 bundle 注册三个插槽。

---

## API 参考

### 主题端点

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/api/dashboard/themes` | GET | 列出可用主题及当前激活名称。内置主题返回 `{name, label, description}`；用户主题还包含带有完整规范化主题对象的 `definition` 字段。 |
| `/api/dashboard/theme` | PUT | 设置激活主题。请求体：`{"name": "midnight"}`。持久化到 `config.yaml` 的 `dashboard.theme` 下。 |

### 插件端点

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/api/dashboard/plugins` | GET | 列出已发现的插件（含 manifest，去除内部字段）。 |
| `/api/dashboard/plugins/rescan` | GET | 强制重新扫描插件目录，无需重启。 |
| `/dashboard-plugins/<name>/<path>` | GET | 从插件的 `dashboard/` 目录提供静态资源。路径遍历已被阻止。 |
| `/api/plugins/<name>/*` | * | 插件注册的后端路由。 |

### `window` 上的 SDK

| 全局变量 | 类型 | 提供方 |
|--------|------|----------|
| `window.__HERMES_PLUGIN_SDK__` | object | `registry.ts` — React、hooks、UI 组件、API 客户端、工具函数。 |
| `window.__HERMES_PLUGINS__.register(name, Component)` | function | 注册插件的主组件。 |
| `window.__HERMES_PLUGINS__.registerSlot(name, slot, Component)` | function | 注册到命名 shell 插槽。 |

---

## 故障排查

**我的主题没有出现在选择器中。**
检查文件是否在 `~/.hermes/dashboard-themes/` 中且以 `.yaml` 或 `.yml` 结尾。刷新页面。运行 `curl http://127.0.0.1:9119/api/dashboard/themes`——你的主题应出现在响应中。如果 YAML 有解析错误，dashboard 会记录到 `~/.hermes/logs/` 下的 `errors.log`。

**我的插件标签页没有显示。**
1. 检查 manifest 是否在 `~/.hermes/plugins/<name>/dashboard/manifest.json`（注意 `dashboard/` 子目录）。
2. 运行 `curl http://127.0.0.1:9119/api/dashboard/plugins/rescan` 强制重新发现。
3. 打开浏览器开发工具 → Network——确认 `manifest.json`、`index.js` 和任何 CSS 均无 404 加载成功。
4. 打开浏览器开发工具 → Console——查找 IIFE 执行期间的错误或 `window.__HERMES_PLUGINS__ is undefined`（表示 SDK 未初始化，通常是更早的 React 渲染崩溃导致）。
5. 验证你的 bundle 以与 `manifest.json:name` **相同的名称**调用 `window.__HERMES_PLUGINS__.register(...)`。

**插槽注册的组件没有渲染。**
`sidebar` 插槽仅在激活主题设置了 `layoutVariant: cockpit` 时渲染。其他插槽始终渲染。如果你注册到某个插槽但没有命中，在 `registerSlot` 内添加 `console.log` 以确认插件 bundle 是否已运行。

**插件后端路由返回 404。**
1. 确认 manifest 中有 `"api": "plugin_api.py"` 且指向 `dashboard/` 内的现有文件。
2. 重启 `hermes dashboard`——插件 API 路由在启动时挂载一次，**不会**在重新扫描时挂载。
3. 检查 `plugin_api.py` 是否导出了模块级的 `router = APIRouter()`。其他导出名称不会被识别。
4. 查看 `~/.hermes/logs/errors.log` 中的 `Failed to load plugin <name> API routes`——导入错误会记录在那里。

**切换主题后我的颜色覆盖丢失了。**
`colorOverrides` 的作用域限于激活主题，切换主题时会被清除——这是设计行为。如果你希望覆盖持久化，请将其写入主题的 YAML，而非实时切换器。

**主题 customCSS 被截断了。**
`customCSS` 块每个主题上限为 32 KiB。可将大型样式表拆分到多个主题中，或改用通过 `css` 字段注入完整样式表的插件（无大小限制）。

**我想在 PyPI 上发布插件。**
Dashboard 插件通过目录结构安装，而非 pip 入口点。目前最简洁的分发方式是用户克隆到 `~/.hermes/plugins/` 的 git 仓库。基于 pip 的 dashboard 插件安装器目前尚未实现。