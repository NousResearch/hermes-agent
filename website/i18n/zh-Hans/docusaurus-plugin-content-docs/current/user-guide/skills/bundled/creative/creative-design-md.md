---
title: "Design Md — 创作/验证/导出Google的DESIGN.md"
sidebar_label: "Design Md"
description: "创作/验证/导出Google的DESIGN.md"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Design Md

创作/验证/导出Google的DESIGN.md令牌规范文件。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/design-md` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `design`, `design-system`, `tokens`, `ui`, `accessibility`, `wcag`, `tailwind`, `dtcg`, `google` |
| 相关技能 | [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) |

## 参考：完整的 SKILL.md

:::info
以下是Hermes加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# DESIGN.md 技能

DESIGN.md是Google的开放规范（Apache-2.0，`google-labs-code/design.md`），用于向编码代理描述视觉身份。一个文件组合：

- **YAML front matter** — 机器可读的设计令牌（规范性值）
- **Markdown body** — 人类可读的理由，组织成规范部分

令牌给出确切值。文章告诉代理*为什么*这些值存在以及如何应用它们。CLI（`npx @google/design.md`）检查结构+WCAG对比度，差异版本以查找回归，并导出到Tailwind或W3C DTCG JSON。

## 何时使用此技能

- 用户要求DESIGN.md文件、设计令牌或设计系统规范
- 用户想要跨多个项目或工具一致的UI/品牌
- 用户粘贴现有DESIGN.md并要求检查、差异、导出或扩展
- 用户要求将风格指南移植到代理可消费的格式
- 用户想要对比度/ WCAG无障碍验证其调色板

对于纯视觉灵感或布局示例，使用`popular-web-designs`。对于*流程和品味*，从头设计一次性HTML产物（原型、演示文稿、着陆页、组件实验室），使用`claude-design`。此技能用于*正式规范文件*本身。

## 文件结构

```md
---
version: alpha
name: Heritage
description: Architectural minimalism meets journalistic gravitas.
colors:
  primary: "#1A1C1E"
  secondary: "#6C7278"
  tertiary: "#B8422E"
  neutral: "#F7F5F2"
typography:
  h1:
    fontFamily: Public Sans
    fontSize: 3rem
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  body-md:
    fontFamily: Public Sans
    fontSize: 1rem
rounded:
  sm: 4px
  md: 8px
  lg: 16px
spacing:
  sm: 8px
  md: 16px
  lg: 24px
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: 12px
  button-primary-hover:
    backgroundColor: "{colors.primary}"
---

## Overview

Architectural Minimalism meets Journalistic Gravitas...

## Colors

- **Primary (#1A1C1E):** Deep ink for headlines and core text.
- **Tertiary (#B8422E):** "Boston Clay" — the sole driver for interaction.

## Typography

Public Sans for everything except small all-caps labels...

## Components

`button-primary` is the only high-emphasis action on a page...
```

## 令牌类型

| 类型 | 格式 | 示例 |
|------|------|------|
| 颜色 | `#` + 十六进制（sRGB） | `"#1A1C1E"` |
| 尺寸 | 数字 + 单位（`px`、`em`、`rem`） | `48px`、`-0.02em` |
| 令牌引用 | `{path.to.token}` | `{colors.primary}` |
| 排版 | 带`fontFamily`、`fontSize`、`fontWeight`、`lineHeight`、`letterSpacing`、`fontFeature`、`fontVariation`的对象 | 见上 |

组件属性白名单：`backgroundColor`、`textColor`、`typography`、
`rounded`、`padding`、`size`、`height`、`width`。变体（hover、active、
pressed）是**单独组件条目**，具有相关键名
（`button-primary-hover`），而不是嵌套。

## 规范部分顺序

部分是可选的，但存在的必须以此顺序出现。重复
标题拒绝文件。

1. Overview（别名：Brand & Style）
2. Colors
3. Typography
4. Layout（别名：Layout & Spacing）
5. Elevation & Depth（别名：Elevation）
6. Shapes
7. Components
8. Do's and Don'ts

未知部分被保留而不是报错。如果值类型有效，则接受未知令牌名称。未知组件属性产生警告。

## 工作流：创作新DESIGN.md

1. **询问用户**（或推断）品牌调性、强调色和排版方向。如果他们提供了站点、图像或氛围，将其转换为上面的令牌形状。
2. **使用`write_file`在他们的项目根目录写`DESIGN.md`**。始终包含`name:`和`colors:`；其他部分可选但鼓励。
3. **使用令牌引用**（`{colors.primary}`）在`components:`部分，而不是重新输入十六进制值。保持调色板单一来源。
4. **检查它**（见下）。在返回之前修复任何损坏的引用或WCAG失败。
5. **如果用户有现有项目**，也在文件旁边写Tailwind或DTCG导出（`tailwind.theme.json`、`tokens.json`）。

## 工作流：检查/差异/导出

CLI是`@google/design.md`（Node）。使用`npx` — 无需全局安装。

```bash
# 验证结构 + 令牌引用 + WCAG对比度
npx -y @google/design.md lint DESIGN.md

# 比较两个版本，回归时失败（exit 1 = 回归）
npx -y @google/design.md diff DESIGN.md DESIGN-v2.md

# 导出到Tailwind主题JSON
npx -y @google/design.md export --format tailwind DESIGN.md > tailwind.theme.json

# 导出到W3C DTCG（设计令牌格式模块）JSON
npx -y @google/design.md export --format dtcg DESIGN.md > tokens.json

# 打印规范本身 — 在注入代理提示时有用
npx -y @google/design.md spec --rules-only --format json
```

所有命令接受`-`表示stdin。`lint`在错误时返回exit 1。如果需要结构化报告，使用`--format json`标志并解析输出。

### 检查规则参考（7条规则捕获的内容）

- `broken-ref`（错误） — `{colors.missing}`指向不存在的令牌
- `duplicate-section`（错误） — 相同`## Heading`出现两次
- `invalid-color`、`invalid-dimension`、`invalid-typography`（错误）
- `wcag-contrast`（警告/信息） — 组件`textColor` vs `backgroundColor`与WCAG AA（4.5:1）和AAA（7:1）的比率
- `unknown-component-property`（警告） — 在上面的白名单之外

当用户关心无障碍时，在摘要中明确指出 — WCAG发现是最有负载的原因使用CLI。

## 陷阱

- **不要嵌套组件变体。** `button-primary.hover`是错误的；
  `button-primary-hover`作为同级键是正确的。
- **十六进制颜色必须是带引号的字符串。** 否则YAML会因`#`而失败，或
  奇怪地截断`#1A1C1E`等值。
- **负尺寸也需要引号。** `letterSpacing: -0.02em`被解析为
  YAML流 — 写`letterSpacing: "-0.02em"`。
- **强制执行部分顺序。** 如果用户以随机顺序给您文章，
  在保存前重新排序以匹配规范列表。
- **`version: alpha`是当前规范版本**（截至2026年4月）。规范
  标记为alpha — 关注突破性更改。
- **令牌引用通过点路径解析。** `{colors.primary}`有效；
  `{primary}`无效。

## 规范真实来源

- 仓库：https://github.com/google-labs-code/design.md（Apache-2.0）
- CLI：npm上的`@google/design.md`
- 生成DESIGN.md文件的许可证：用户项目使用的任何许可证；
  规范本身是Apache-2.0。
