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

已缓存的系统 prompt 按三个有序层级组装（见 `agent/system_prompt.py`）：

1. **stable（稳定）** — 身份（`SOUL.md` 或回退值）、工具/模型指引、编码工作简报
2. **context（上下文）** — 调用方提供的 `system_message`、项目上下文文件（`.hermes.md` / `AGENTS.md` / `CLAUDE.md` / `.cursorrules`），然后是依赖工作树（worktree）的 git 工作区快照、操作者指令和平台提示
3. **volatile（易变）** — skills 索引、内置记忆快照（`MEMORY.md`）、用户档案快照（`USER.md`）、外部记忆 provider 块、时间戳/会话/模型/提供商，然后是运行时环境提示（host / home / **当前工作目录**）

最终系统 prompt 按如下顺序拼接：`stable` → `context` → `volatile`。

这个顺序对「优先级」的讨论很重要：
- skills（索引）属于 **volatile** 层，且位居 volatile 层首位
- 记忆/档案快照也属于 **volatile** 层
- 两者都仍在已缓存的系统 prompt 中（它们不是作为轮次中的临时覆盖层注入的）

在 context 层内部，共享的项目文件排在**任何提到当前工作树的内容之前**。同一项目的会话运行在不同的 git 工作树中时，就能共享一段覆盖整个 context 块的 prompt 前缀，而不是在第一个依赖 cwd 的行就中断——这段前缀正是「最长前缀」provider 缓存会复用的部分。一个没有工作区快照的会话会把其尾随的指引留在 stable 层；运行时环境块则永远结束 volatile 层。

对已存储 prompt 的后果：`_stored_prompt_matches_runtime()`（`agent/conversation_loop.py`）会在渲染出的 `# Hermes runtime environment` 边界之后读取第一个 host-info 段落，并在绝对末尾用一个闭合标记，把这种布局与引用该标题的旧式散文区分开。运行时边界跟随在所有项目、操作者、记忆和插件文本之后，因此这些块中的示例不会伪装成运行时 cwd。模型/提供商在运行时边界**之前**读取，从而排除 embedder 描述。`Platform:` 刻意不是一个身份字段：一次界面切换（desktop ↔ TUI）会保留已存储的字节，并把当前界面的指引作为一次性说明放到每轮用户消息通道（`agent/surface_switch.py`），因此缓存前缀得以保留（#104414）。`<available_skills>` 索引也以同样方式保留：每轮都会把它与「会话现在会构建出的索引」进行比较（`agent/skills_index_delta.py`，从 `build_turn_context` 调用，因此长驻的 CLI/TUI/Desktop agent 也被覆盖，而不只是 prompt 恢复时），一旦出现差异（新增索引行、移除的名字），就通过同一条一次性说明通道传递，且只在差异发生变化时才发送。进程内的索引缓存以 skill 文件的签名为 key，因此另一个进程写入的变化也能被看见。索引本身在下一个重建边界（压缩）收敛。旧式 prompt 保留其原来的「host-before-context」锚点，因此在重排之前就持久化的 prompt 仍然能通过校验。

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
Task-learned procedures, pitfalls, and task-specific preferences belong
in skills. Memory is the narrow exception for facts that apply to EVERY
session regardless of task. Skill-writing instructions appear here only
when skill_manage is available; its absence does not widen memory's scope.
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

## 自定义平台提示

平台提示（上面的 Layer 10）是 Hermes 为 Telegram、WhatsApp、Slack、CLI 及其他平台注入的「每种界面（surface）各自的指引」——例如「你在终端上，避免 Markdown」。内置默认值位于 `PLATFORM_HINTS`（`agent/system_prompt.py`）；插件提供的平台通过平台注册表提供它们自己的提示。

管理员可以通过 `config.yaml` 顶层的 `platform_hints` 键，在不影响其他平台的情况下，为单个平台**追加**或**替换**提示：

```yaml
platform_hints:
  whatsapp:
    append: >
      When tabular output would be useful, invoke the table_formatting
      skill instead of emitting a Markdown table.
  slack:
    replace: "You are on Slack. Keep responses tight and avoid wide tables."
  telegram: "Prefer short messages; split long answers."   # shorthand = append
```

- `append` — 保留内置提示，并在其后追加额外文本。
- `replace` — 完全替换内置提示。
- 一个裸字符串 — `append` 的简写。
- 两者同时存在时，`replace` 优先于 `append`。
- 格式错误的条目会被防御性地忽略，并回退到未修改的默认值，因此一个错误的配置值绝不会破坏 prompt 组装或跨平台泄漏。

Cron 任务以 `cron` 平台运行，但其最终响应落在任务的 `deliver` 通道上，因此 cron agent 的 prompt 也会在 `Delivery destination (<channel>):` 行下携带该通道的提示（内置文本加上其 `platform_hints.<channel>` 覆盖）。因此 `platform_hints.slack.append` 同时作用于 Slack 投递的定时任务和实时的 Slack 聊天；`platform_hints.cron` 仍然管理 cron 段落本身。

覆盖在系统 prompt 构建时解析（会话开始时，以及压缩时——因为压缩会重建 prompt）。对于一个固定的配置，它产生一个字节稳定的提示，因此它不会破坏 prompt 缓存——它不是对冻结 prompt 的会话中途实时修改。其最终落点取决于是否有 workspace 快照：有工作区快照的会话，平台提示与内置提示一起进入 context 层；没有工作区快照的会话则留在 stable 层（源码 `agent/system_prompt.py` 的 `_coding_parts` / `_post_workspace_parts` 逻辑）。

## SOUL.md 在 prompt 中的位置

`SOUL.md` 位于 `~/.hermes/SOUL.md`，作为 agent 的身份标识——系统 prompt 的第一个部分。`prompt_builder.py` 中的加载逻辑如下：

```python
# From agent/prompt_builder.py (simplified)
def load_soul_md() -> Optional[str]:
    soul_path = get_hermes_home() / "SOUL.md"
    if not soul_path.exists():
        return None
    content = soul_path.read_text(encoding="utf-8").strip()
    content = _scan_context_content(content, "SOUL.md", user_authored=True)  # Security scan: warn + load, never block
    content = _truncate_content(content, "SOUL.md")       # Cap scales with model context window (20k floor); config override wins
    return content
```

当 `load_soul_md()` 返回内容时，它会替换硬编码的 `DEFAULT_AGENT_IDENTITY`。随后调用 `build_context_files_prompt()` 时传入 `skip_soul=True`，以防止 SOUL.md 出现两次（一次作为身份，一次作为上下文文件）。

若 `SOUL.md` 不存在，系统将回退到：

```
You are Hermes Agent, built by Nous Research. Be direct: match the length
of your reply to the weight of the ask — a one-line question gets a
one-line answer, and finished work gets a short report of what changed,
what's verified, and what's left, never a replay of the process. No
filler ("Great question," "I'd be happy to"), no restating the request
back, no re-summarizing what you already said, no narrating tool calls
the user can see. Plain claims over adjectives; when unsure, say so
plainly. Agree because it's right, not because the user said it. Depth
is earned — give it when the user asks for detail, teaches, or the
stakes demand it, not by default.
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
- **安全扫描** — 检查 prompt 注入模式（不可见 unicode、"ignore previous instructions"、凭据窃取尝试）。命中后，项目文件会被替换为 `[BLOCKED: …]` 标记；用户自己的 `SOUL.md`（在 `HERMES_HOME` 中）则被警告但仍照常加载（它在写入时经人工批准，因此与 `config.yaml` 属于同一信任级别）
- **截断处理** — 上限为 `context_file_max_chars` 字符，使用 70/20 头尾比例并附截断标记。该上限随模型的上下文窗口伸缩（20,000 字符下限，500K 上限）；`config.yaml` 中显式的 `context_file_max_chars` 始终优先。
- **剥离 YAML frontmatter** — `.hermes.md` 的 frontmatter 会被移除（保留供未来配置覆盖使用）

## 仅在 API 调用时生效的层

以下内容刻意*不*作为已缓存系统 prompt 的一部分持久化：

- `ephemeral_system_prompt`
- prefill 消息
- gateway 派生的会话上下文覆盖层
- 注入当前轮次用户消息的后续轮次 Honcho/外部召回内容

`pre_llm_call` 插件上下文也走这条「API 调用时」路径：它被追加到当前轮的 **用户消息**，而非写入已缓存的系统 prompt。当多个插件返回上下文时，Hermes 会拼接这些上下文块（见 [Hooks → `pre_llm_call`](../user-guide/features/hooks.md#pre_llm_call)）。

这种分离使稳定前缀保持稳定，从而有效缓存。

## 记忆快照

本地记忆和用户档案数据在系统 prompt 的 **volatile 层**中被捕获。会话中途的写入操作会更新磁盘状态，但不会修改已构建的缓存系统 prompt，直到某个重建路径运行时才生效（新会话，或显式的失效/重建流程，例如压缩触发的重建）。

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

## 评审对 prompt 界面（surface）的改动

**prompt surface** 是指一个原样安装（stock install）在第一条用户消息之前发送的内容：系统 prompt 和 `tools[]` 数组。一行代码改动就可以重写其中任意一个，而 diff 未必能体现出来。一个指引常量可能跨层移动、一个 toolset 可能新增工具，或者一个 schema 重写器可能修改描述。因此渲染出的 surface 被提交并像代码一样评审：

- `scripts/ci/prompt_surface.py render` 通过生产组装代码（`build_system_prompt_parts` 和 agent 自己的 `tools[]`）构建 surface，并写入 `tests/fixtures/prompt_surface/`。它渲染九种 surface：`cli`（四种模型族——模型名控制指引和补丁方言）、代码工作区内的 `cli`、`desktop`、`telegram`、`discord` 和 `cron`。渲染使用一个全新的 home。时钟、宿主 OS、网络、工具可用性和机器路径都被钉死，因此笔记本和 CI 会产出相同的字节。每个已注册工具都被视为可用，因此受凭据门控的 schema 也会被评审。
- `tests/ci/test_prompt_surface.py` 在提交的快照过期时失败。一个改动 surface 的 PR 会携带重新生成的快照，评审者在 Files 标签页读取真实的 prompt 文本和 schema。
- **Prompt surface diff** CI 任务向 PR 评审评论发布一份逐 surface 的摘要：token 增量、prompt hunk，以及新增或移除的工具和参数。
- 对 `tests/fixtures/prompt_surface/` 下的改动，在合并前需要**来自 hermes-agent-core 的批准**。

在移动 surface 的任何改动之后重新生成：

```bash
scripts/run-in-hermes-env python scripts/ci/prompt_surface.py render
```

用户侧自定义（`SOUL.md`、上下文文件、你安装的 skills、`platform_hints`、`agent.*` 指引开关）永远不会触碰快照。渲染使用原样默认值和仅随包提供的 skills。

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