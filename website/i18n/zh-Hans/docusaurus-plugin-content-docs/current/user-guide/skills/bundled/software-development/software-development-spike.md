---
title: "Spike — 在构建前验证想法的一次性实验"
sidebar_label: "Spike"
description: "在构建前验证想法的一次性实验"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Spike

在构建前验证想法的一次性实验。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/spike` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent（改编自 gsd-build/get-shit-done） |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `spike`, `prototype`, `experiment`, `feasibility`, `throwaway`, `exploration`, `research`, `planning`, `mvp`, `proof-of-concept` |
| 相关 skill | [`sketch`](/user-guide/skills/bundled/creative/creative-sketch)、[`writing-plans`](/user-guide/skills/bundled/software-development/software-development-writing-plans)、[`subagent-driven-development`](/user-guide/skills/bundled/software-development/software-development-subagent-driven-development)、[`plan`](/user-guide/skills/bundled/software-development/software-development-plan) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Spike

当用户想在正式构建前**摸清一个想法**时使用此 skill——验证可行性、比较方案，或暴露单靠调研无法回答的未知问题。Spike 本质上是可丢弃的。一旦完成使命，就扔掉它。

当用户说出以下内容时加载此 skill："让我试试这个"、"我想看看 X 是否可行"、"spike 一下"、"在我决定用 Y 之前"、"Z 的快速原型"、"这到底可不可能？"或"比较 A 和 B"。

## 何时不使用此 skill

- 答案可以从文档或阅读代码中直接获得——做调研即可，不必构建
- 工作属于生产路径——改用 `writing-plans` / `plan`
- 想法已经验证——直接跳到实现

## 如果用户安装了完整的 GSD 系统

如果 `gsd-spike` 作为同级 skill 出现（通过 `npx get-shit-done-cc --hermes` 安装），当用户需要完整 GSD 工作流时，优先使用 **`gsd-spike`**：持久化的 `.planning/spikes/` 状态、跨会话的 MANIFEST 追踪、Given/When/Then 结论格式，以及与 GSD 其余部分集成的提交模式。本 skill 是面向未安装（或不需要）完整系统的用户的轻量独立版本。

## 核心方法

无论规模大小，每个 spike 都遵循以下循环：

```
decompose  →  research  →  build  →  verdict
   ↑__________________________________________↓
                  iterate on findings
```

### 1. 分解（Decompose）

将用户的想法拆解为 **2-5 个独立的可行性问题**。每个问题对应一个 spike。以表格形式呈现，采用 Given/When/Then 框架：

| # | Spike | 验证内容（Given/When/Then） | 风险 |
|---|-------|----------------------------|------|
| 001 | websocket-streaming | Given 一个 WS 连接，when LLM 流式输出 token，then 客户端接收到的数据块延迟 &lt; 100ms | 高 |
| 002a | pdf-parse-pdfjs | Given 一个多页 PDF，when 用 pdfjs 解析，then 可提取结构化文本 | 中 |
| 002b | pdf-parse-camelot | Given 一个多页 PDF，when 用 camelot 解析，then 可提取结构化文本 | 中 |

**Spike 类型：**
- **standard（标准型）** — 一种方案回答一个问题
- **comparison（对比型）** — 同一问题，不同方案（共享编号，字母后缀 `a`/`b`/`c`）

**好的 spike 问题：** 具体的可行性问题，有可观测的输出。
**差的 spike 问题：** 过于宽泛、无可观测输出，或仅仅是"阅读 X 的文档"。

**按风险排序。** 最可能否定整个想法的 spike 优先执行。如果难点行不通，就没必要先做简单的部分。

**跳过分解**的唯一情形：用户已明确知道要 spike 什么并明确说明。此时将其想法作为单个 spike 处理。

### 2. 对齐（Align，适用于多 spike 想法）

展示 spike 表格。询问："按此顺序全部构建，还是需要调整？"在写任何代码之前，让用户删减、重排或重新定义。

### 3. 调研（Research，每个 spike 构建前）

Spike 并非不需要调研——你需要调研到足以选定正确方案，然后再构建。每个 spike 的步骤：

1. **简述。** 2-3 句话：这个 spike 是什么、为何重要、关键风险。
2. **列出竞争方案**（如果存在真实选择）：

   | 方案 | 工具/库 | 优点 | 缺点 | 状态 |
   |------|---------|------|------|------|
   | ... | ... | ... | ... | 维护中 / 已废弃 / beta |

3. **选定一个。** 说明原因。如果有 2 个以上可信方案，在 spike 内构建快速变体。
4. **跳过调研**的情形：纯逻辑，无外部依赖。

调研步骤使用 Hermes 工具：

- `web_search("python websocket streaming libraries 2025")` — 查找候选库
- `web_extract(urls=["https://websockets.readthedocs.io/..."])` — 阅读实际文档（返回 markdown）
- `terminal("pip show websockets | grep Version")` — 检查项目 venv 中已安装的版本

对于没有文档页面的库，克隆并通过 `read_file` 阅读其 `README.md` / `examples/`。Context7 MCP（如果用户已配置）也是好的来源——`mcp_*_resolve-library-id` 然后 `mcp_*_query-docs`。

### 4. 构建（Build）

每个 spike 一个目录，保持独立。

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

**偏向构建用户可以交互的东西。** Spike 失败的常见原因是唯一输出只是一行写着"it works"的日志。用户想要*感受*到 spike 在运行。默认选择，按优先级排序：

1. 可运行的 CLI，接受输入并打印可观测的输出
2. 演示该行为的最小化 HTML 页面
3. 带有一个端点的小型 web 服务器
4. 用可识别断言验证问题的单元测试

**深度优于速度。** 绝不在一次 happy-path 运行后就宣称"它可以用"。测试边界情况，追踪意外发现。只有调查足够诚实，结论才值得信赖。

**避免**以下内容（除非 spike 明确需要）：复杂的包管理、构建工具/打包器、Docker、env 文件、配置系统。全部硬编码——这是 spike。

**构建单个 spike** — 典型工具调用序列：

```
terminal("mkdir -p spikes/001-websocket-streaming")
write_file("spikes/001-websocket-streaming/README.md", "# 001: websocket-streaming\n\n...")
write_file("spikes/001-websocket-streaming/main.py", "...")
terminal("cd spikes/001-websocket-streaming && python3 main.py")
# 观察输出，迭代。
```

**并行对比 spike（002a / 002b）— 委托执行。** 当两种方案可以并行运行且都需要真正的工程实现（而非 10 行原型）时，使用 `delegate_task` 分发：

```
delegate_task(tasks=[
    {"goal": "Build 002a-pdf-parse-pdfjs: ...", "toolsets": ["terminal", "file", "web"]},
    {"goal": "Build 002b-pdf-parse-camelot: ...", "toolsets": ["terminal", "file", "web"]},
])
```

每个子 agent 返回自己的结论；由你撰写对比总结。

### 5. 结论（Verdict）

每个 spike 的 `README.md` 以如下内容结尾：

```markdown
## Verdict: VALIDATED | PARTIAL | INVALIDATED

### What worked
- ...

### What didn't
- ...

### Surprises
- ...

### Recommendation for the real build
- ...
```

**VALIDATED** = 核心问题得到肯定回答，有证据支撑。
**PARTIAL** = 在约束条件 X、Y、Z 下可行——记录这些约束。
**INVALIDATED** = 不可行，原因如下。这也是一次成功的 spike。

## 对比 spike

当两种方案回答同一个问题（002a / 002b）时，**依次构建**，然后在最后做正面对比：

```markdown
## Head-to-head: pdfjs vs camelot

| 维度 | pdfjs (002a) | camelot (002b) |
|------|--------------|----------------|
| 提取质量 | 9/10 结构化 | 7/10 仅表格 |
| 配置复杂度 | npm install，1 行代码 | pip + ghostscript |
| 100 页 PDF 性能 | 3s | 18s |
| 处理旋转文本 | 否 | 是 |

**胜者：** pdfjs 适合我们的用例。如果后续需要以表格为主的提取，再考虑 camelot。
```

## 前沿模式（决定下一步 spike 什么）

如果已有 spike 存在，且用户问"下一步应该 spike 什么？"，遍历现有目录，寻找：

- **集成风险** — 两个已验证的 spike 独立测试时都访问同一资源
- **数据交接** — spike A 的输出被假设与 spike B 的输入兼容，但从未验证
- **愿景中的空白** — 被假设但未经验证的能力
- **替代方案** — 针对 PARTIAL 或 INVALIDATED spike 的不同角度

以 Given/When/Then 形式提出 2-4 个候选，让用户选择。

## 输出

- 在仓库根目录创建 `spikes/`（如果用户使用 GSD 约定，则为 `.planning/spikes/`）
- 每个 spike 一个目录：`NNN-descriptive-name/`
- 每个 spike 的 `README.md` 记录问题、方案、结果和结论
- 保持代码可丢弃——一个需要花 2 天"清理以投入生产"的 spike 本身就是一个失败的 spike

## 致谢

改编自 GSD（Get Shit Done）项目的 `/gsd-spike` 工作流——MIT © 2025 Lex Christopherson（[gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)）。完整 GSD 系统提供持久化 spike 状态、MANIFEST 追踪，以及与更广泛的规格驱动开发流水线的集成；通过 `npx get-shit-done-cc --hermes --global` 安装。