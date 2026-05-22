---
sidebar_position: 8
title: "上下文文件"
description: "项目上下文文件 —— .hermes.md、AGENTS.md、CLAUDE.md、全局 SOUL.md 和 .cursorrules —— 自动注入每次对话"
---

# 上下文文件

Hermes Agent 自动发现并加载塑造其行为的上下文文件。有些是项目本地的，从您的工作目录发现。`SOUL.md` 现在是 Hermes 实例全局的，仅从 `HERMES_HOME` 加载。

## 支持的上下文文件

| 文件 | 用途 | 发现方式 |
|------|---------|-----------| 
| **.hermes.md** / **HERMES.md** | 项目指令（最高优先级） | 走到 git 根目录 |
| **AGENTS.md** | 项目指令、约定、架构 | 启动时 CWD + 子目录渐进式 |
| **CLAUDE.md** | Claude Code 上下文文件（也被检测） | 启动时 CWD + 子目录渐进式 |
| **SOUL.md** | 此 Hermes 实例的全局个性和语气定制 | 仅 `HERMES_HOME/SOUL.md` |
| **.cursorrules** | Cursor IDE 编码约定 | 仅 CWD |
| **.cursor/rules/*.mdc** | Cursor IDE 规则模块 | 仅 CWD |

:::info 优先级系统
每个会话仅加载**一个**项目上下文类型（首个匹配胜出）：`.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。**SOUL.md** 始终作为智能体身份独立加载（槽位 #1）。
:::

## AGENTS.md

`AGENTS.md` 是主要项目上下文文件。它告诉智能体您的项目结构、要遵循的约定以及任何特殊指令。

### 渐进式子目录发现

在会话开始时，Hermes 将工作目录中的 `AGENTS.md` 加载到系统提示中。当智能体会话期间导航到子目录时（通过 `read_file`、`terminal`、`search_files` 等），它会**渐进式发现**这些目录中的上下文文件，并在它们变得相关时注入到对话中。

```
my-project/
├── AGENTS.md              ← 启动时加载（系统提示）
├── frontend/
│   └── AGENTS.md          ← 智能体读取 frontend/ 文件时发现
├── backend/
│   └── AGENTS.md          ← 智能体读取 backend/ 文件时发现
└── shared/
    └── AGENTS.md          ← 智能体读取 shared/ 文件时发现
```

与在启动时加载所有内容相比，这种方法有两个优势：
- **无系统提示膨胀** —— 子目录提示仅在需要时出现
- **提示缓存保留** —— 系统提示在轮次之间保持稳定

每个子目录每个会话最多检查一次。发现还会向上遍历父目录，因此读取 `backend/src/main.py` 会发现 `backend/AGENTS.md`，即使 `backend/src/` 没有自己的上下文文件。

:::info
子目录上下文文件与启动上下文文件经过相同的[安全扫描](#安全提示注入保护)。恶意文件会被阻止。
:::

### AGENTS.md 示例

```markdown
# 项目上下文

这是一个带有 Python FastAPI 后端的 Next.js 14 Web 应用程序。

## 架构
- 前端：Next.js 14，使用 App Router，位于 `/frontend`
- 后端：FastAPI，位于 `/backend`，使用 SQLAlchemy ORM
- 数据库：PostgreSQL 16
- 部署：Hetzner VPS 上的 Docker Compose

## 约定
- 所有前端代码使用 TypeScript 严格模式
- Python 代码遵循 PEP 8，到处使用类型提示
- 所有 API 端点返回 JSON，格式为 `{data, error, meta}`
- 测试放在 `__tests__/` 目录（前端）或 `tests/`（后端）

## 重要说明
- 永远不要直接修改迁移文件 —— 使用 Alembic 命令
- `.env.local` 文件包含真实 API 密钥，不要提交它
- 前端端口是 3000，后端是 8000，数据库是 5432
```

## SOUL.md

`SOUL.md` 控制智能体的个性、语气和沟通风格。有关完整详情，请参阅 [个性](/user-guide/features/personality) 页面。

**位置：**

- `~/.hermes/SOUL.md`
- 或如果您使用自定义主目录运行 Hermes，则为 `$HERMES_HOME/SOUL.md`

重要详情：

- 如果尚不存在，Hermes 会自动播种默认 `SOUL.md`
- Hermes 仅从 `HERMES_HOME` 加载 `SOUL.md`
- Hermes 不会探测工作目录中的 `SOUL.md`
- 如果文件为空，不会将任何 `SOUL.md` 内容添加到提示中
- 如果文件有内容，内容在扫描和截断后逐字注入

## .cursorrules

Hermes 兼容 Cursor IDE 的 `.cursorrules` 文件和 `.cursor/rules/*.mdc` 规则模块。如果这些文件存在于您的项目根目录中，并且没有找到更高优先级的上下文文件（`.hermes.md`、`AGENTS.md` 或 `CLAUDE.md`），它们将作为项目上下文加载。

这意味着您现有的 Cursor 约定在使用 Hermes 时自动适用。

## 上下文文件如何加载

### 启动时（系统提示）

上下文文件由 `agent/prompt_builder.py` 中的 `build_context_files_prompt()` 加载：

1. **扫描工作目录** —— 检查 `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`（首个匹配胜出）
2. **读取内容** —— 每个文件作为 UTF-8 文本读取
3. **安全扫描** —— 内容被检查提示注入模式
4. **截断** —— 超过 20,000 字符的文件被头/尾截断（70% 头部，20% 尾部，中间有标记）
5. **组装** —— 所有部分在 `# Project Context` 标题下组合
6. **注入** —— 组装的内容添加到系统提示中

### 会话期间（渐进式发现）

`agent/subdirectory_hints.py` 中的 `SubdirectoryHintTracker` 监视工具调用参数中的文件路径：

1. **路径提取** —— 每次工具调用后，从参数中提取文件路径（`path`、`workdir`、shell 命令）
2. **祖先遍历** —— 检查目录和最多 5 个父目录（在已访问的目录处停止）
3. **提示加载** —— 如果找到 `AGENTS.md`、`CLAUDE.md` 或 `.cursorrules`，则加载（每个目录首个匹配）
4. **安全扫描** —— 与启动文件相同的提示注入扫描
5. **截断** —— 每个文件上限为 8,000 字符
6. **注入** —— 附加到工具结果，以便模型在上下文中自然地看到它

最终的提示部分大致如下：

```text
# Project Context

The following project context files have been loaded and should be followed:

## AGENTS.md

[Your AGENTS.md content here]

## .cursorrules

[Your .cursorrules content here]

[Your SOUL.md content here]
```

注意 SOUL 内容直接插入，没有额外的包装文本。

## 安全：提示注入保护

所有上下文文件在包含之前都会扫描潜在的提示注入。扫描器检查：

- **指令覆盖尝试**："ignore previous instructions"、"disregard your rules"
- **欺骗模式**："do not tell the user"
- **系统提示覆盖**："system prompt override"
- **隐藏 HTML 注释**：`<!-- ignore instructions -->`
- **隐藏 div 元素**：`<div style="display:none">`
- **凭证外泄**：`curl ... $API_KEY`
- **秘密文件访问**：`cat .env`、`cat credentials`
- **不可见字符**：零宽空格、双向覆盖、词连接符

如果检测到任何威胁模式，文件会被阻止：

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

:::warning
此扫描器可防止常见注入模式，但不能替代审查共享仓库中的上下文文件。始终验证您未创作的项目的 AGENTS.md 内容。
:::

## 大小限制

| 限制 | 值 |
|-------|-------|
| 每文件最大字符数 | 20,000（~7,000 tokens） |
| 头部截断比例 | 70% |
| 尾部截断比例 | 20% |
| 截断标记 | 10%（显示字符计数并建议使用文件工具） |

当文件超过 20,000 字符时，截断消息显示为：

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## 有效上下文文件的技巧

:::tip AGENTS.md 最佳实践
1. **保持简洁** —— 保持在 20K 字符以下；智能体每轮都读取它
2. **使用标题结构化** —— 使用 `##` 章节表示架构、约定、重要说明
3. **包含具体示例** —— 展示首选代码模式、API 形状、命名约定
4. **提及不要做什么** —— "永远不要直接修改迁移文件"
5. **列出关键路径和端口** —— 智能体将这些用于终端命令
6. **随项目演进更新** —— 过时的上下文比没有上下文更糟
:::

### 每子目录上下文

对于 monorepos，将子目录特定指令放在嵌套的 AGENTS.md 文件中：

```markdown
<!-- frontend/AGENTS.md -->
# 前端上下文

- 使用 `pnpm` 而非 `npm` 进行包管理
- 组件放在 `src/components/`，页面放在 `src/app/`
- 使用 Tailwind CSS，永远不要内联样式
- 使用 `pnpm test` 运行测试
```

```markdown
<!-- backend/AGENTS.md -->
# 后端上下文

- 使用 `poetry` 进行依赖管理
- 使用 `poetry run uvicorn main:app --reload` 运行开发服务器
- 所有端点都需要 OpenAPI 文档字符串
- 数据库模型在 `models/`，schema 在 `schemas/`
```
