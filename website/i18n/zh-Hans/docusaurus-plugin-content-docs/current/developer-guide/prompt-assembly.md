---
sidebar_position: 5
title: "Prompt 组装"
description: "Hermes 如何构建系统 prompt、保持缓存稳定性并注入临时层"
---

# Prompt 组装

Hermes 刻意将以下内容分离：

- **已缓存的系统 prompt 状态**
- **API 调用时临时添加的内容**

这是项目中最重要的设计决策之一，因为它影响：

- token 用量
- prompt 缓存效果
- 会话连续性
- 记忆正确性

主要文件：

- `run_agent.py`
- `agent/prompt_builder.py`
- `tools/memory_tool.py`

## 已缓存的系统 prompt 层

已缓存的系统 prompt 大致按以下顺序组装：

1. agent 身份 — 优先使用 `HERMES_HOME` 中的 `SOUL.md`，否则回退到 `prompt_builder.py` 中的 `DEFAULT_AGENT_IDENTITY`
2. 工具感知行为指导
3. Honcho 静态块（激活时）
4. 可选系统消息
5. 冻结的 MEMORY 快照
6. 冻结的 USER 配置文件快照
7. skills 索引
8. 上下文文件（`AGENTS.md`、`.cursorrules`、`.cursor/rules/*.mdc`）— 若 SOUL.md 已在第 1 步作为身份加载，则此处**不**再包含它
9. 时间戳 / 可选会话 ID
10. 平台提示

当设置了 `skip_context_files`（例如子 agent 委托）时，不会加载 SOUL.md，而是使用硬编码的 `DEFAULT_AGENT_IDENTITY`。

### 具体示例：组装后的系统 prompt

以下是所有层都存在时最终系统 prompt 的简化视图（注释说明每个部分的来源）：

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

## SOUL.md 在 prompt 中的位置

`SOUL.md` 位于 `~/.hermes/SOUL.md`，作为 agent 的身份标识——系统 prompt 的第一个部分。`prompt_builder.py` 中的加载逻辑如下：

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

当 `load_soul_md()` 返回内容时，它会替换硬编码的 `DEFAULT_AGENT_IDENTITY`。随后调用 `build_context_files_prompt()` 时传入 `skip_soul=True`，以防止 SOUL.md 出现两次（一次作为身份，一次作为上下文文件）。

若 `SOUL.md` 不存在，系统将回退到：

```
You are Hermes Agent, an intelligent AI assistant created by Nous Research.
You are helpful, knowledgeable, and direct. You assist users with a wide
range of tasks including answering questions, writing and editing code,
analyzing information, creative work, and executing actions via your tools.
You communicate clearly, admit uncertainty when appropriate, and prioritize
being genuinely useful over being verbose unless otherwise directed below.
Be targeted and efficient in your exploration and investigations.
```

## 上下文文件的注入方式

`build_context_files_prompt()` 使用**优先级系统**——只加载一种项目上下文类型（先匹配先赢）：

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

| 优先级 | 文件 | 搜索范围 | 说明 |
|--------|------|----------|------|
| 1 | `.hermes.md`、`HERMES.md` | 从 CWD 向上至 git 根目录 | Hermes 原生项目配置 |
| 2 | `AGENTS.md` | 仅 CWD | 常见 agent 指令文件 |
| 3 | `CLAUDE.md` | 仅 CWD | Claude Code 兼容性 |
| 4 | `.cursorrules`、`.cursor/rules/*.mdc` | 仅 CWD | Cursor 兼容性 |

所有上下文文件均会：
- **安全扫描** — 检查 prompt 注入模式（不可见 unicode、"ignore previous instructions"、凭据窃取尝试）
- **截断处理** — 使用 70/20 头尾比例上限为 20,000 字符，并附截断标记
- **剥离 YAML frontmatter** — `.hermes.md` 的 frontmatter 会被移除（保留供未来配置覆盖使用）

## 仅在 API 调用时生效的层

以下内容刻意*不*作为已缓存系统 prompt 的一部分持久化：

- `ephemeral_system_prompt`
- prefill 消息
- gateway 派生的会话上下文覆盖层
- 注入当前轮次用户消息的后续轮次 Honcho 召回内容

这种分离使稳定前缀保持稳定，从而有效缓存。

## 记忆快照

本地记忆和用户配置文件数据在会话开始时作为冻结快照注入。会话中途的写入操作会更新磁盘状态，但不会修改已构建的系统 prompt，直到新会话开始或强制重建时才生效。

## 上下文文件

`agent/prompt_builder.py` 使用**优先级系统**扫描并清理项目上下文文件——只加载一种类型（先匹配先赢）：

1. `.hermes.md` / `HERMES.md`（向上遍历至 git 根目录）
2. `AGENTS.md`（启动时的 CWD；子目录在会话期间通过 `agent/subdirectory_hints.py` 逐步发现）
3. `CLAUDE.md`（仅 CWD）
4. `.cursorrules` / `.cursor/rules/*.mdc`（仅 CWD）

`SOUL.md` 通过 `load_soul_md()` 单独加载用于身份槽位。加载成功后，`build_context_files_prompt(skip_soul=True)` 会防止其出现两次。

长文件在注入前会被截断。

## Skills 索引

当 skills 工具可用时，skills 系统会向 prompt 贡献一个紧凑的 skills 索引。

## 支持的 prompt 自定义入口

大多数用户应将 `agent/prompt_builder.py` 视为实现代码，而非配置入口。推荐的自定义路径是修改 Hermes 已加载的 prompt 输入，而非直接编辑 Python 模板。

### 优先使用这些入口

- `~/.hermes/SOUL.md` — 用自定义 agent 角色和固定行为替换内置默认身份块。
- `~/.hermes/MEMORY.md` 和 `~/.hermes/USER.md` — 提供应在新会话中快照的持久跨会话事实和用户配置文件数据。
- 项目上下文文件，如 `.hermes.md`、`HERMES.md`、`AGENTS.md`、`CLAUDE.md` 或 `.cursorrules` — 注入仓库特定的工作规则。
- Skills — 打包可复用的工作流和参考资料，无需编辑核心 prompt 代码。
- 可选系统 prompt 配置 / API 覆盖 — 添加部署特定的指令文本，无需 fork Hermes。
- 临时覆盖层，如 `HERMES_EPHEMERAL_SYSTEM_PROMPT` 或 prefill 消息 — 添加不应成为已缓存 prompt 前缀一部分的轮次级指导。

### 何时应编辑代码

仅当你刻意维护一个 fork 或向上游贡献行为变更时，才编辑 `agent/prompt_builder.py`。该文件为每个会话组装 prompt 管道、缓存边界和注入顺序。直接编辑该文件是全局产品变更，而非针对单个用户的 prompt 自定义。

换言之：

- 若想要不同的助手身份，编辑 `SOUL.md`
- 若想要不同的仓库规则，编辑项目上下文文件
- 若想要可复用的操作流程，添加或修改 skills
- 若想改变 Hermes 为所有人组装 prompt 的方式，修改 Python 代码并将其视为代码贡献

## Prompt 组装为何如此拆分

该架构刻意优化以：

- 保留提供商侧的 prompt 缓存
- 避免不必要地修改历史记录
- 保持记忆语义清晰可理解
- 允许 gateway/ACP/CLI 添加上下文而不污染持久 prompt 状态

## 相关文档

- [上下文压缩与 Prompt 缓存](./context-compression-and-caching.md)
- [会话存储](./session-storage.md)
- [Gateway 内部机制](./gateway-internals.md)