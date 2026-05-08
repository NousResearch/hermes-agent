---
title: "Spike — 在构建前验证想法的抛弃式实验"
sidebar_label: "Spike"
description: "在构建前验证想法的抛弃式实验"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Spike

在构建前验证想法的抛弃式实验。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/spike` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent（改编自 gsd-build/get-shit-done） |
| 许可证 | MIT |
| 标签 | `spike`, `prototype`, `experiment`, `feasibility`, `throwaway`, `exploration`, `research`, `planning`, `mvp`, `proof-of-concept` |
| 相关技能 | [`sketch`](/docs/user-guide/skills/bundled/creative/creative-sketch), [`writing-plans`](/docs/user-guide/skills/bundled/software-development/software-development-writing-plans), [`subagent-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-subagent-driven-development), [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Spike

当用户想在承诺真正构建之前**试探一个想法**——验证可行性、比较方案或揭示任何研究都无法回答的未知——时使用此技能。Spike 本质上是一次性的。一旦完成了它们的使命就丢掉它们。

当用户说"让我试试"、"我想看看 X 是否可行"、"spike 一下"、"在承诺 Y 之前"、"快速做一个 Z 的原型"、"这甚至可能吗？"或"比较 A 和 B"时加载此技能。

## 何时不使用此技能

- 答案可以从文档或阅读代码中获知——直接做研究，不要构建
- 工作在生产路径上——使用 `writing-plans` / `plan` 代替
- 想法已经验证——直接跳到实现

## 如果用户安装了完整的 GSD 系统

如果 `gsd-spike` 作为同级技能出现（通过 `npx get-shit-done-cc --hermes` 安装），当用户想要完整的 GSD 工作流时优先使用 **`gsd-spike`**：持久的 `.planning/spikes/` 状态、跨会话的 MANIFEST 跟踪、Given/When/Then 结论格式，以及与 GSD 其余部分集成的提交模式。此技能是不需要（或不想用）完整系统的用户的轻量级独立版本。

## 核心方法

无论规模如何，每个 Spike 遵循此循环：

```
分解  →  研究  →  构建  →  结论
   ↑__________________________________________↓
                 在发现上迭代
```

### 1. 分解

将用户的想法分解为 **2-5 个独立的可行性问题**。每个问题是一个 Spike。以 Given/When/Then 框架将它们呈现为表格：

| # | Spike | 验证（Given/When/Then） | 风险 |
|---|-------|------------------------|------|
| 001 | websocket-streaming | Given WS 连接，当 LLM 流式传输 token 时，客户端在 100ms 内收到块 | 高 |
| 002a | pdf-parse-pdfjs | Given 多页 PDF，当用 pdfjs 解析时，可提取结构化文本 | 中 |
| 002b | pdf-parse-camelot | Given 多页 PDF，当用 camelot 解析时，可提取结构化文本 | 中 |

**Spike 类型：**
- **标准** ——一个方法回答一个问题
- **对比** ——同一问题，不同方法（共享编号，字母后缀 `a`/`b`/`c`）

**好的 Spike 问题：** 有具体可行性和可观察输出。
**坏的 Spike 问题：** 太宽泛、无可观察输出，或只是"阅读 X 的文档"。

**按风险排序。** 最可能扼杀想法的 Spike 先运行。如果困难的部分不行，没有必要原型化容易的部分。

**跳过分解** 仅当用户已经确切知道要 Spike 什么并且明确说了。然后将他们的想法作为一个 Spike。

### 2. 对齐（对于多 Spike 想法）

呈现 Spike 表格。问："按此顺序全部构建，还是调整？"让用户在编写任何代码之前删除、重新排序或重新表述。

### 3. 研究（每个 Spike，构建前）

Spike 不是无研究的——你研究到足以选择正确方法，然后构建。每个 Spike：

1. **简述。** 2-3 句话：这个 Spike 是什么，为什么重要，关键风险。
2. **如果真的有选择，展示竞争方案：**

   | 方案 | 工具/库 | 优点 | 缺点 | 状态 |
   |----------|-------------|------|------|--------|
   | ... | ... | ... | ... | 维护中 / 已弃用 / beta |

3. **选择一个。** 说明原因。如果有 2+ 可信，在 Spike 中构建快速变体。
4. 对于没有外部依赖的纯逻辑，**跳过研究**。

使用 Hermes 工具进行研究步骤：

- `web_search("python websocket streaming libraries 2025")` ——查找候选
- `web_extract(urls=["https://websockets.readthedocs.io/..."])` ——阅读实际文档（返回 Markdown）
- `terminal("pip show websockets | grep Version")` ——检查项目 venv 中安装了什么

对于没有文档页面的库，通过 `read_file` 克隆并阅读其 `README.md` / `examples/`。如果用户配置了 Context7 MCP，也是好的来源。

### 4. 构建

每个 Spike 一个目录。保持独立。

<!-- ascii-guard-ignore -->
```
spikes/
├── 001-websocket-streaming/
│   ├── README.md
│   └── main.py
├── 002a-pdf-parse-pdfjs/
│   ├── README.md
│   └── parse.js
└── 002b-pdf-parse-camelot/
    ├── README.md
    └── parse.py
```
<!-- ascii-guard-ignore-end -->

**偏向用户可以交互的东西。** Spike 在唯一输出是一个说"it works"的日志行时失败。用户想要*感受* Spike 工作。默认选择，按优先顺序：

1. 一个可运行的 CLI，接受输入并打印可观察输出
2. 一个最小 HTML 页面演示行为
3. 一个有一个端点的小型 Web 服务器
4. 一个用可识别的断言测试问题的单元测试

**深度优于速度。** 永远不要在一条顺利路径运行后就宣布"it works"。测试边缘情况。追踪意外发现。只有当调查是诚实的，结论才值得信赖。

**避免** 除非 Spike 特别需要：复杂的包管理、构建工具/bundler、Docker、env 文件、配置系统。全部硬编码——这是一个 Spike。

**构建一个 Spike** ——典型工具序列：

```
terminal("mkdir -p spikes/001-websocket-streaming")
write_file("spikes/001-websocket-streaming/README.md", "# 001: websocket-streaming\n\n...")
write_file("spikes/001-websocket-streaming/main.py", "...")
terminal("cd spikes/001-websocket-streaming && python3 main.py")
# 观察输出，迭代。
```

**并行对比 Spike（002a / 002b）——委派。** 当两个方案可以并行运行且都需要真正的工程时，用 `delegate_task` 展开：

```
delegate_task(tasks=[
    {"goal": "Build 002a-pdf-parse-pdfjs: ...", "toolsets": ["terminal", "file", "web"]},
    {"goal": "Build 002b-pdf-parse-camelot: ...", "toolsets": ["terminal", "file", "web"]},
])
```

每个子代理返回自己的结论；你来写对比。

### 5. 结论

每个 Spike 的 `README.md` 以以下结尾：

```markdown
## 结论: VALIDATED | PARTIAL | INVALIDATED

### 有效的部分
- ...

### 无效的部分
- ...

### 意外发现
- ...

### 对正式构建的建议
- ...
```

**VALIDATED** = 核心问题得到肯定回答，有证据支持。
**PARTIAL** = 在约束 X、Y、Z 下可行——记录它们。
**INVALIDATED** = 不可行，原因如下。这是一个成功的 Spike。

## 对比 Spike

当两个方案回答同一问题（002a / 002b）时，**背靠背**构建它们，最后做一个正面比较：

```markdown
## 正面对比: pdfjs vs camelot

| 维度 | pdfjs (002a) | camelot (002b) |
|------|--------------|----------------|
| 提取质量 | 9/10 结构化 | 7/10 仅表格 |
| 设置复杂度 | npm install, 1 行 | pip + ghostscript |
| 100 页 PDF 性能 | 3s | 18s |
| 处理旋转文本 | 否 | 是 |

**胜者：** 在我们的用例中是 pdfjs。如果以后需要表格优先提取则是 camelot。
```

## 探索模式（选择接下来要 Spike 什么）

如果 Spike 已经存在且用户说"我接下来应该 Spike 什么？"，遍历现有目录并查找：

- **集成风险** ——两个已验证的 Spike 触摸同一资源但独立测试
- **数据交接** ——Spike A 的输出被假设与 Spike B 的输入兼容；从未证明
- **愿景中的空白** ——被假设但未证明的能力
- **替代方案** ——针对 PARTIAL 或 INVALIDATED Spike 的不同角度

以 Given/When/Then 提出 2-4 个候选。让用户选择。

## 输出

- 在仓库根目录创建 `spikes/`（或如果用户使用 GSD 约定则创建 `.planning/spikes/`）
- 每个 Spike 一个目录：`NNN-descriptive-name/`
- 每个 Spike 一个 `README.md` 捕获问题、方案、结果、结论
- 保持代码抛弃式——一个需要 2 天"清理以投入生产"的 Spike 是一个糟糕的 Spike

## 致谢

改编自 GSD (Get Shit Done) 项目的 `/gsd-spike` 工作流——MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done))。完整的 GSD 系统提供持久的 Spike 状态、MANIFEST 跟踪以及与更广泛的规范驱动开发流水线的集成；使用 `npx get-shit-done-cc --hermes --global` 安装。
