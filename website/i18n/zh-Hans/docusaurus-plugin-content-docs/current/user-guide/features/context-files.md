---
sidebar_position: 8
title: "上下文文件"
description: "项目上下文文件——.hermes.md、AGENTS.md、CLAUDE.md、全局 SOUL.md 和 .cursorrules——自动注入每个对话"
---

# 上下文文件

Hermes 代理自动发现并加载塑造其行为的上下文文件。有些是项目本地的，从你的工作目录中发现。`SOUL.md` 现在是全局的，仅从 `HERMES_HOME` 加载。

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
每个会话只加载**一个**项目上下文类型（先到先得）：`.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。**SOUL.md** 始终作为代理身份独立加载（槽位 #1）。
:::

## AGENTS.md

`AGENTS.md` 是主要项目上下文文件。它告诉代理项目如何构建、要遵循的约定以及任何特殊指令。

### 渐进式子目录发现

在会话开始时，Hermes 将工作目录的 `AGENTS.md` 加载到系统提示中。当代理在会话期间导航到子目录时（通过 `read_file`、`terminal`、`search_files` 等），它**渐进地发现**这些目录中的上下文文件，并在它们变得相关时将它们注入对话。

```
my-project/
├── AGENTS.md              ← 在启动时加载（系统提示）
├── frontend/
│   └── AGENTS.md          ← 当代理读取 frontend/ 文件时发现
├── backend/
│   └── AGENTS.md          ← 当代理读取 backend/ 文件时发现
└── shared/
    └── AGENTS.md          ← 当代理读取 shared/ 文件时发现
```

这种方法相对于启动时加载所有内容有两个优点：
- **无系统提示膨胀** — 子目录提示仅在需要时出现
- **提示缓存保留** — 系统提示在跨轮次保持稳定

每个子目录在每个会话最多检查一次。发现也会向上走父目录，所以读取 `backend/src/main.py` 会发现 `backend/AGENTS.md`，即使 `backend/src/` 没有自己的上下文文件。

:::info
子目录上下文文件与启动上下文文件通过相同的[安全扫描](#security-prompt-injection-protection)。恶意文件被阻止。
:::

### AGENTS.md 示例

```markdown
# Project Context

This is a Next.js 14 web application with a Python FastAPI backend.

## Architecture
- Frontend: Next.js 14 with App Router in `/frontend`
- Backend: FastAPI in `/backend`, uses SQLAlchemy ORM
- Database: PostgreSQL 16
- Deployment: Docker Compose on a Hetzner VPS

## Conventions
- Use TypeScript strict mode for all frontend code
- Python code follows PEP 8, use type hints everywhere
- All API endpoints return JSON with `{data, error, meta}` shape
- Tests go in `__tests__/` directories (frontend) or `tests/` (backend)

## Important Notes
- Never modify migration files directly — use Alembic commands
- The `.env.local` file has real API keys, don't commit it
- Frontend port is 3000, backend is 8000, DB is 5432
```

## SOUL.md

`SOUL.md` 控制代理的个性、语气和沟通风格。请参阅[个性](/docs/user-guide/features/personality)页面获取完整详情。

**位置：**

- `~/.hermes/SOUL.md`
- 或如果你用自定义主目录运行 Hermes，则为 `$HERMES_HOME/SOUL.md`

重要细节：

- 如果 `SOUL.md` 尚不存在，Hermes 自动生成一个默认的
- Hermes 仅从 `HERMES_HOME` 加载 `SOUL.md`
- Hermes 不会在工作目录中探测 `SOUL.md`
- 如果文件为空，不会从 `SOUL.md` 向提示添加任何内容
- 如果文件有内容，内容在扫描和截断后逐字注入

## .cursorrules

Hermes 与 Cursor IDE 的 `.cursorrules` 文件和 `.cursor/rules/*.mdc` 规则模块兼容。如果这些文件存在于你的项目根目录且未找到更高优先级的上下文文件（`.hermes.md`、`AGENTS.md` 或 `CLAUDE.md`），它们作为项目上下文加载。

这意味着当你使用 Hermes 时，你现有的 Cursor 约定自动适用。

## 上下文文件如何加载

### 在启动时（系统提示）

上下文文件通过 `agent/prompt_builder.py` 中的 `build_context_files_prompt()` 加载：

1. **扫描工作目录** — 检查 `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`（先到先得）
2. **读取内容** — 每个文件作为 UTF-8 文本读取
3. **安全扫描** — 内容被检查提示注入模式
4. **截断** — 超过 20,000 字符的文件被头尾截断（头部 70%，尾部 20%，中间有标记）
5. **组装** — 所有部分在 `# Project Context` 标题下组合
6. **注入** — 组装的内容被添加到系统提示

### 会话期间（渐进发现）

`agent/subdirectory_hints.py` 中的 `SubdirectoryHintTracker` 监视工具调用参数中的文件路径：

1. **路径提取** — 每个工具调用后，从参数（`path`、`workdir`、shell 命令）中提取文件路径
2. **祖先遍历** — 检查目录及向上最多 5 个父目录（停在已访问目录）
3. **提示加载** — 如果找到 `AGENTS.md`、`CLAUDE.md` 或 `.cursorrules`，则加载（每个目录先到先得）
4. **安全扫描** — 与启动文件相同的提示注入扫描
5. **截断** — 每个文件上限 8,000 字符
6. **注入** — 追加到工具结果，这样模型自然地在上下文中看到它

最终提示部分大致如下：

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

所有上下文文件在被包含之前都被扫描潜在的提示注入。扫描器检查：

- **指令覆盖尝试**："忽略之前的指令"、"无视你的规则"
- **欺骗模式**："不要告诉用户"
- **系统提示覆盖**："系统提示覆盖"
- **隐藏 HTML 注释**：`<!-- ignore instructions -->`
- **隐藏 div 元素**：`<div style="display:none">`
- **凭据泄露**：`curl ... $API_KEY`
- **密钥文件访问**：`cat .env`、`cat credentials`
- **不可见字符**：零宽空格、双向覆盖、词连接符

如果检测到任何威胁模式，文件被阻止：

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

:::warning
此扫描器保护免受常见注入模式，但它不能替代审查共享仓库中的上下文文件。始终验证你未创作的项目的 AGENTS.md 内容。
:::

## 大小限制

| 限制 | 值 |
|-------|-------|
| 每个文件最大字符数 | 20,000（约 7,000 令牌） |
| 头部截断比例 | 70% |
| 尾部截断比例 | 20% |
| 截断标记 | 10%（显示字符计数并建议使用文件工具） |

当文件超过 20,000 字符时，截断消息读取：

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## 有效上下文文件的提示

:::tip AGENTS.md 最佳实践
1. **保持简洁** — 保持在 20K 字符以下；代理每轮都读取它
2. **用标题结构** — 使用 `##` 部分表示架构、约定、重要笔记
3. **包含具体示例** — 显示首选代码模式、API 形状、命名约定
4. **提及不要做什么** — "不要直接修改迁移文件"
5. **列出关键路径和端口** — 代理将这些用于终端命令
6. **随着项目发展更新** — 过时的上下文比没有上下文更糟糕
:::

### 每个子目录上下文

对于单体仓库，将子目录特定指令放在嵌套的 AGENTS.md 文件中：

```markdown
<!-- frontend/AGENTS.md -->
# Frontend Context

- Use `pnpm` not `npm` for package management
- Components go in `src/components/`, pages in `src/app/`
- Use Tailwind CSS, never inline styles
- Run tests with `pnpm test`
```

```markdown
<!-- backend/AGENTS.md -->
# Backend Context

- Use `poetry` for dependency management
- Run the dev server with `poetry run uvicorn main:app --reload`
- All endpoints need OpenAPI docstrings
- Database models are in `models/`, schemas in `schemas/`
```
