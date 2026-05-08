---
sidebar_position: 5
title: "提示词组装"
description: "Hermes 如何构建系统提示词、保持缓存稳定性以及注入临时层"
---

# 提示词组装

Hermes 故意分离：

- **缓存的系统提示词状态**
- **临时 API 调用时添加**

这是项目中最重要设计选择之一，因为它影响：

- token 使用量
- 提示词缓存效果
- 会话连续性
- 内存正确性

主要文件：

- `run_agent.py`
- `agent/prompt_builder.py`
- `tools/memory_tool.py`

## 缓存的系统提示词层

缓存的系统提示词大致按此顺序组装：

1. 代理身份 — 可用时从 `HERMES_HOME` 的 `SOUL.md`，否则回退到 `prompt_builder.py` 中的 `DEFAULT_AGENT_IDENTITY`
2. 工具感知行为指导
3. Honcho 静态块（激活时）
4. 可选系统消息
5. 冻结的 MEMORY 快照
6. 冻结的 USER 配置快照
7. 技能索引
8. 上下文文件（`AGENTS.md`、`.cursorrules`、`.cursor/rules/*.mdc`）— SOUL.md **不**在此处包含，因为它已经在第 1 步作为身份加载
9. 时间戳 / 可选会话 ID
10. 平台提示

当设置 `skip_context_files` 时（例如子代理委托），不加载 SOUL.md 并使用硬编码的 `DEFAULT_AGENT_IDENTITY` 代替。

### 具体示例：组装系统提示词

这里是一个简化视图，展示当所有层都存在时最终系统提示词的样子（注释显示每个部分的来源）：

```
# Layer 1: Agent Identity (from ~/.hermes/SOUL.md)
You are Hermes, an AI assistant created by Nous Research.
You are an expert software engineer and researcher.
You value correctness, clarity, and efficiency.
...

# Layer 2: Tool-aware behavior guidance
You have persistent memory across sessions. Save durable facts using
the memory tool: user preferences, environment details, tool quirks,
and stable conventions. Memory is injected into every turn, so keep
it compact and focused on facts that will still matter later.
...
When the user references something from a past conversation or you
suspect relevant cross-session context exists, use session_search
to recall it before asking them to repeat themselves.

# Tool-use enforcement (for GPT/Codex models only)
You MUST use your tools to take action — do not describe what you
would do or plan to do without actually doing it.
...

# Layer 3: Honcho static block (when active)
[Honcho personality/context data]

# Layer 4: Optional system message (from config or API)
[User-configured system message override]

# Layer 5: Frozen MEMORY snapshot
## Persistent Memory
- User prefers Python 3.12, uses pyproject.toml
- Default editor is nvim
- Working on project "atlas" in ~/code/atlas
- Timezone: US/Pacific

# Layer 6: Frozen USER profile snapshot
## User Profile
- Name: Alice
- GitHub: alice-dev

# Layer 7: Skills index
## Skills (mandatory)
Before replying, scan the skills below. If one clearly matches
your task, load it with skill_view(name) and follow its instructions.
...
<available_skills>
  software-development:
    - code-review: Structured code review workflow
    - test-driven-development: TDD methodology
  research:
    - arxiv: Search and summarize arXiv papers
</available_skills>

# Layer 8: Context files (from project directory)
# Project Context
The following project context files have been loaded and should be followed:

## AGENTS.md
This is the atlas project. Use pytest for testing. The main
entry point is src/atlas/main.py. Always run `make lint` before
committing.

# Layer 9: Timestamp + session
Current time: 2026-03-30T14:30:00-07:00
Session: abc123

# Layer 10: Platform hint
You are a CLI AI Agent. Try not to use markdown but simple text
renderable inside a terminal.
```

## SOUL.md 如何出现在提示词中

`SOUL.md` 位于 `~/.hermes/SOUL.md`，作为代理的身份 — 系统提示词的第一个部分。`prompt_builder.py` 中的加载逻辑如下：

```python
# From agent/prompt_builder.py (simplified)
def load_soul_md() -> Optional[str]:
    soul_path = get_hermes_home() / "SOUL.md"
    if not soul_path.exists():
        return None
    content = soul_path.read_text(encoding="utf-8").strip()
    content = _scan_context_content(content, "SOUL.md")  # Security scan
    content = _truncate_content(content, "SOUL.md")       # Cap at 20k chars
    return content
```

当 `load_soul_md()` 返回内容时，它替换硬编码的 `DEFAULT_AGENT_IDENTITY`。然后用 `skip_soul=True` 调用 `build_context_files_prompt()` 函数以防止 SOUL.md 出现两次（一次作为身份，一次作为上下文文件）。

如果 `SOUL.md` 不存在，系统回退到：

```
You are Hermes Agent, an intelligent AI assistant created by Nous Research.
You are helpful, knowledgeable, and direct. You assist users with a wide
range of tasks including answering questions, writing and editing code,
analyzing information, creative work, and executing actions via your tools.
You communicate clearly, admit uncertainty when appropriate, and prioritize
being genuinely useful over being verbose unless otherwise directed below.
Be targeted and efficient in your exploration and investigations.
```

## 上下文文件如何注入

`build_context_files_prompt()` 使用**优先级系统** — 只加载一种项目上下文类型（先到先得）：

```python
# From agent/prompt_builder.py (simplified)
def build_context_files_prompt(cwd=None, skip_soul=False):
    cwd_path = Path(cwd).resolve()

    # Priority: first match wins — only ONE project context loaded
    project_context = (
        _load_hermes_md(cwd_path)       # 1. .hermes.md / HERMES.md (walks to git root)
        or _load_agents_md(cwd_path)    # 2. AGENTS.md (cwd only)
        or _load_claude_md(cwd_path)    # 3. CLAUDE.md (cwd only)
        or _load_cursorrules(cwd_path)  # 4. .cursorrules / .cursor/rules/*.mdc
    )

    sections = []
    if project_context:
        sections.append(project_context)

    # SOUL.md from HERMES_HOME (independent of project context)
    if not skip_soul:
        soul_content = load_soul_md()
        if soul_content:
            sections.append(soul_content)

    if not sections:
        return ""

    return (
        "# Project Context\n\n"
        "The following project context files have been loaded "
        "and should be followed:\n\n"
        + "\n".join(sections)
    )
```

### 上下文文件发现详情

| 优先级 | 文件 | 搜索范围 | 备注 |
|----------|-------|-------------|-------|
| 1 | `.hermes.md`、`HERMES.md` | CWD 到 git 根目录 | Hermes 原生项目配置 |
| 2 | `AGENTS.md` | 仅 CWD | 常见代理指令文件 |
| 3 | `CLAUDE.md` | 仅 CWD | Claude Code 兼容性 |
| 4 | `.cursorrules`、`.cursor/rules/*.mdc` | 仅 CWD | Cursor 兼容性 |

所有上下文文件都：
- **安全扫描** — 检查提示词注入模式（不可见 unicode、"忽略先前指令"、凭据泄露尝试）
- **截断** — 使用 70/20 头尾比和截断标记上限为 20,000 字符
- **剥离 YAML frontmatter** — `.hermes.md` frontmatter 被移除（保留用于未来配置覆盖）

## 仅 API 调用时的层

这些故意**不**作为缓存系统提示词的一部分持久化：

- `ephemeral_system_prompt`
- 预填充消息
- 网关派生的会话上下文覆盖
- 后续轮次中注入到当前轮次用户消息的 Honcho 回忆

这种分离保持稳定前缀稳定以便缓存。

## 内存快照

本地内存和用户配置数据在会话开始时作为冻结快照注入。中途写入更新磁盘状态但不改变已经构建的系统提示词，直到新会话或强制重建发生。

## 上下文文件

`agent/prompt_builder.py` 使用**优先级系统**扫描和清理项目上下文文件 — 只加载一种类型（先到先得）：

1. `.hermes.md` / `HERMES.md`（遍历到 git 根目录）
2. `AGENTS.md`（启动时的 CWD；通过 `agent/subdirectory_hints.py` 在会话期间逐步发现子目录）
3. `CLAUDE.md`（仅 CWD）
4. `.cursorrules` / `.cursor/rules/*.mdc`（仅 CWD）

`SOUL.md` 通过 `load_soul_md()` 单独加载用于身份槽。成功加载时，`build_context_files_prompt(skip_soul=True)` 防止它出现两次。

长文件在注入前被截断。

## 技能索引

当技能工具可用时，技能系统向提示词贡献紧凑的技能索引。

## 支持的提示词自定义表面

大多数用户应将 `agent/prompt_builder.py` 视为实现代码，而不是配置表面。支持的自定义路径是更改 Hermes 已经加载的提示词输入，而不是就地编辑 Python 模板。

### 首先使用这些表面

- `~/.hermes/SOUL.md` — 用您自己的代理角色和常设行为替换内置默认身份块。
- `~/.hermes/MEMORY.md` 和 `~/.hermes/USER.md` — 提供应快照到新会话的持久跨会话事实和用户配置数据。
- 项目上下文文件如 `.hermes.md`、`HERMES.md`、`AGENTS.md`、`CLAUDE.md` 或 `.cursorrules` — 注入仓库特定的工作规则。
- 技能 — 打包可重用工作流和引用，而不编辑核心提示词代码。
- 可选系统提示词配置 / API 覆盖 — 添加部署特定指令文本而不分叉 Hermes。
- 临时覆盖如 `HERMES_EPHEMERAL_SYSTEM_PROMPT` 或预填充消息 — 添加不应成为缓存提示词前缀一部分的轮次作用域指导。

### 何时编辑代码

仅当您有意维护分叉或贡献上游行为更改时编辑 `agent/prompt_builder.py`。该文件组装每个会话的提示词管道、缓存边界和注入顺序。直接编辑那里是全局产品更改，而不是每个用户的提示词自定义。

换句话说：

- 如果您想要不同的助手身份，编辑 `SOUL.md`
- 如果您想要不同的仓库规则，编辑项目上下文文件
- 如果您想要可重用的操作程序，添加或修改技能
- 如果您想为每个人更改 Hermes 如何组装提示词，更改 Python 并将其视为代码贡献

## 提示词组装为什么这样拆分

架构故意优化以：

- 保留 provider 端提示词缓存
- 避免不必要地改变历史
- 保持内存语义可理解
- 让网关/ACP/CLI 添加上下文而不污染持久提示词状态

## 相关文档

- [上下文压缩和提示缓存](./context-compression-and-caching.md)
- [会话存储](./session-storage.md)
- [网关内部原理](./gateway-internals.md)
