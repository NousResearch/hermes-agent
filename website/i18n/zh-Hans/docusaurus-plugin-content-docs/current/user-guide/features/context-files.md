---
sidebar_position: 8
title: "上下文文件"
description: "项目上下文文件 — .hermes.md、AGENTS.md、CLAUDE.md、全局 SOUL.md 以及 .cursorrules — 自动注入每次对话"
---

# 上下文文件

Hermes Agent 会自动发现并加载上下文文件，以塑造其行为方式。部分文件属于项目本地文件，从工作目录中发现。`SOUL.md` 现在对整个 Hermes 实例全局生效，仅从 `HERMES_HOME` 加载。

## 支持的上下文文件

| 文件 | 用途 | 发现方式 |
|------|---------|-----------| 
| **.hermes.md** / **HERMES.md** | 项目指令（最高优先级） | 向上遍历至 git 根目录 |
| **AGENTS.md** | 项目指令、规范、架构说明 | 启动时的 CWD 及子目录（渐进式） |
| **CLAUDE.md** | Claude Code 上下文文件（同样支持检测） | 启动时的 CWD 及子目录（渐进式） |
| **SOUL.md** | 当前 Hermes 实例的全局个性与语气定制 | 仅 `HERMES_HOME/SOUL.md` |
| **.cursorrules** | Cursor IDE 编码规范 | 仅 CWD |
| **.cursor/rules/*.mdc** | Cursor IDE 规则模块 | 仅 CWD |

:::info 优先级系统
每次会话仅加载**一种**项目上下文类型（先匹配先生效）：`.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。**SOUL.md** 始终作为 agent 身份独立加载（插槽 #1）。
:::

## AGENTS.md

`AGENTS.md` 是主要的项目上下文文件。它告知 agent 项目的结构、需要遵循的规范以及任何特殊指令。

### 渐进式子目录发现

会话启动时，Hermes 将工作目录中的 `AGENTS.md` 加载到系统 prompt（提示词）中。在会话期间，当 agent 通过 `read_file`、`terminal`、`search_files` 等工具导航进入子目录时，它会**渐进式发现**这些目录中的上下文文件，并在其变得相关的时刻将其注入对话。

```
my-project/
├── AGENTS.md              ← 启动时加载（系统 prompt）
├── frontend/
│   └── AGENTS.md          ← agent 读取 frontend/ 文件时发现
├── backend/
│   └── AGENTS.md          ← agent 读取 backend/ 文件时发现
└── shared/
    └── AGENTS.md          ← agent 读取 shared/ 文件时发现
```

与启动时加载所有内容相比，此方式有两个优势：
- **避免系统 prompt 膨胀** — 子目录提示仅在需要时出现
- **保留 prompt 缓存** — 系统 prompt 在各轮次间保持稳定

每个子目录在每次会话中最多检查一次。发现机制同样会向上遍历父目录，因此读取 `backend/src/main.py` 时，即使 `backend/src/` 没有自己的上下文文件，也会发现 `backend/AGENTS.md`。

:::info
子目录上下文文件与启动时的上下文文件经过相同的[安全扫描](#security-prompt-injection-protection)。恶意文件会被拦截。
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

`SOUL.md` 控制 agent 的个性、语气和沟通风格。完整详情请参阅[个性](/user-guide/features/personality)页面。

**位置：**

- `~/.hermes/SOUL.md`
- 或 `$HERMES_HOME/SOUL.md`（若使用自定义主目录运行 Hermes）

重要说明：

- 若 `SOUL.md` 尚不存在，Hermes 会自动生成一个默认文件
- Hermes 仅从 `HERMES_HOME` 加载 `SOUL.md`
- Hermes 不会在工作目录中探测 `SOUL.md`
- 若文件为空，`SOUL.md` 中的内容不会添加到 prompt
- 若文件有内容，内容在扫描和截断后原样注入

## .cursorrules

Hermes 兼容 Cursor IDE 的 `.cursorrules` 文件和 `.cursor/rules/*.mdc` 规则模块。若这些文件存在于项目根目录，且未找到更高优先级的上下文文件（`.hermes.md`、`AGENTS.md` 或 `CLAUDE.md`），则将其作为项目上下文加载。

这意味着使用 Hermes 时，现有的 Cursor 规范会自动生效。

## 上下文文件的加载方式

### 启动时（系统 prompt）

上下文文件由 `agent/prompt_builder.py` 中的 `build_context_files_prompt()` 加载：

1. **扫描工作目录** — 依次检查 `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`（先匹配先生效）
2. **读取内容** — 以 UTF-8 文本读取每个文件
3. **安全扫描** — 检查内容是否存在 prompt 注入模式
4. **截断** — 超过 20,000 个字符的文件进行首尾截断（70% 头部，20% 尾部，中间插入标记）
5. **组装** — 所有部分合并在 `# Project Context` 标题下
6. **注入** — 组装后的内容添加到系统 prompt

### 会话期间（渐进式发现）

`agent/subdirectory_hints.py` 中的 `SubdirectoryHintTracker` 监视工具调用参数中的文件路径：

1. **路径提取** — 每次工具调用后，从参数（`path`、`workdir`、shell 命令）中提取文件路径
2. **祖先目录遍历** — 检查该目录及最多 5 个父目录（跳过已访问的目录）
3. **提示加载** — 若发现 `AGENTS.md`、`CLAUDE.md` 或 `.cursorrules`，则加载（每个目录先匹配先生效）
4. **安全扫描** — 与启动文件相同的 prompt 注入扫描
5. **截断** — 每个文件最多 8,000 个字符
6. **注入** — 追加到工具结果中，使模型在上下文中自然看到

最终 prompt 部分大致如下：

```text
# Project Context

The following project context files have been loaded and should be followed:

## AGENTS.md

[Your AGENTS.md content here]

## .cursorrules

[Your .cursorrules content here]

[Your SOUL.md content here]
```

注意，SOUL 内容直接插入，不带额外的包装文本。

## 安全性：Prompt 注入防护

所有上下文文件在被纳入之前都会扫描潜在的 prompt 注入。扫描器检查以下内容：

- **指令覆盖尝试**：「ignore previous instructions」、「disregard your rules」
- **欺骗模式**：「do not tell the user」
- **系统 prompt 覆盖**：「system prompt override」
- **隐藏 HTML 注释**：`<!-- ignore instructions -->`
- **隐藏 div 元素**：`<div style="display:none">`
- **凭据窃取**：`curl ... $API_KEY`
- **密钥文件访问**：`cat .env`、`cat credentials`
- **不可见字符**：零宽空格、双向覆盖字符、词连接符

若检测到任何威胁模式，该文件将被拦截：

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

:::warning
此扫描器可防范常见注入模式，但不能替代对上下文文件的人工审查。对于非本人编写的共享仓库，请务必验证 AGENTS.md 的内容。
:::

## 大小限制

| 限制 | 值 |
|-------|-------|
| 每个文件最大字符数 | 20,000（约 7,000 个 token） |
| 头部截断比例 | 70% |
| 尾部截断比例 | 20% |
| 截断标记 | 10%（显示字符数并建议使用文件工具） |

当文件超过 20,000 个字符时，截断提示如下：

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## 有效使用上下文文件的技巧

:::tip AGENTS.md 最佳实践
1. **保持简洁** — 远低于 20K 字符；agent 每轮都会读取
2. **使用标题结构** — 用 `##` 分节描述架构、规范、重要说明
3. **包含具体示例** — 展示首选代码模式、API 结构、命名规范
4. **说明禁止事项** — 例如「不得直接修改迁移文件」
5. **列出关键路径和端口** — agent 在执行终端命令时会用到
6. **随项目演进更新** — 过时的上下文比没有上下文更糟
:::

### 子目录上下文

对于 monorepo，在嵌套的 AGENTS.md 文件中放置子目录专属指令：

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