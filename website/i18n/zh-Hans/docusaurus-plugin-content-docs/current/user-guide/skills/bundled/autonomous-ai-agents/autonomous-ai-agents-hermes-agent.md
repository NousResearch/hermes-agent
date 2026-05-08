---
title: "Hermes Agent — 配置、扩展或贡献 Hermes Agent"
sidebar_label: "Hermes Agent"
description: "配置、扩展或贡献 Hermes Agent"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Agent

配置、扩展或贡献 Hermes Agent。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/autonomous-ai-agents/hermes-agent` |
| 版本 | `2.0.0` |
| 作者 | Hermes Agent + Teknium |
| 许可证 | MIT |
| 标签 | `hermes`、`setup`、`configuration`、`multi-agent`、`spawning`、`cli`、`gateway`、`development` |
| 相关技能 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code)、[`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex)、[`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# Hermes Agent

Hermes Agent 是 Nous Research 的开源 AI 代理框架，在终端、消息平台和 IDE 中运行。它与 Claude Code（Anthropic）、Codex（OpenAI）和 OpenClaw 属于同类 — 自主编码和任务执行代理，使用工具调用与系统交互。Hermes 可与任何 LLM 提供商（OpenRouter、Anthropic、OpenAI、DeepSeek、本地模型及 15+ 其他）协作，运行在 Linux、macOS 和 WSL 上。

Hermes 的与众不同之处：

- **通过技能自我提升** — Hermes 通过将可复用程序保存为技能来从经验中学习。当它解决复杂问题、发现工作流或被纠正时，可以将该知识持久化为加载到未来会话的技能文档。技能随时间积累，使代理在你的特定任务和环境中表现更好。
- **跨会话持久记忆** — 记住你是谁、你的偏好、环境细节和所学经验。可插拔的记忆后端（内置、Honcho、Mem0 等）让你选择记忆的工作方式。
- **多平台网关** — 同一代理运行在 Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Email 等 10+ 平台上，具有完整工具访问，不仅仅是聊天。
- **提供商无关** — 无需改变任何其他设置即可中途切换模型和提供商。凭据池跨多个 API 密钥自动轮换。
- **配置文件** — 运行多个具有隔离配置、会话、技能和记忆的独立 Hermes 实例。
- **可扩展** — 插件、MCP 服务器、自定义工具、Webhook 触发器、cron 调度和完整的 Python 生态系统。

人们用 Hermes 进行软件开发、研究、系统管理、数据分析、内容创作、家庭自动化，以及任何受益于具有持久上下文和完整系统访问的 AI 代理的工作。

**此技能帮助你有效地使用 Hermes Agent** — 设置、配置功能、生成额外代理实例、故障排除、找到正确的命令和设置，以及在需要扩展或贡献时理解系统工作方式。

**文档：** https://hermes-agent.nousresearch.com/docs/

## 快速开始

```bash
# 安装
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# 交互式聊天（默认）
hermes

# 单次查询
hermes chat -q "What is the capital of France?"

# 设置向导
hermes setup

# 更改模型/提供商
hermes model

# 健康检查
hermes doctor
```

---

## CLI 参考

### 全局标志

```
hermes [flags] [command]

  --version, -V             显示版本
  --resume, -r SESSION      通过 ID 或标题恢复会话
  --continue, -c [NAME]     通过名称恢复，或最近的会话
  --worktree, -w            隔离的 git 工作树模式（并行代理）
  --skills, -s SKILL        预加载技能（逗号分隔或重复）
  --profile, -p NAME        使用命名配置文件
  --yolo                    跳过危险命令批准
  --pass-session-id         在系统提示中包含会话 ID
```

无子命令默认为 `chat`。

### 聊天

```
hermes chat [flags]
  -q, --query TEXT          单次查询，非交互式
  -m, --model MODEL         模型（如 anthropic/claude-sonnet-4）
  -t, --toolsets LIST       逗号分隔的工具集
  --provider PROVIDER       强制提供商（openrouter、anthropic、nous 等）
  -v, --verbose             详细输出
  -Q, --quiet               抑制横幅、旋转器、工具预览
  --checkpoints             启用文件系统检查点（/rollback）
  --source TAG              会话来源标签（默认：cli）
```

### 配置

```
hermes setup [section]      交互式向导（model|terminal|gateway|tools|agent）
hermes model                交互式模型/提供商选择器
hermes config               查看当前配置
hermes config edit          在 $EDITOR 中打开 config.yaml
hermes config set KEY VAL   设置配置值
hermes config path          打印 config.yaml 路径
hermes config env-path      打印 .env 路径
hermes config check         检查缺失/过时的配置
hermes config migrate       用新选项更新配置
hermes login [--provider P] OAuth 登录（nous、openai-codex）
hermes logout               清除已存储的认证
hermes doctor [--fix]       检查依赖和配置
hermes status [--all]       显示组件状态
```

### 工具和技能

```
hermes tools                交互式工具启用/禁用（curses UI）
hermes tools list           显示所有工具和状态
hermes tools enable NAME    启用工具集
hermes tools disable NAME   禁用工具集

hermes skills list          列出已安装的技能
hermes skills search QUERY  搜索技能中心
hermes skills install ID    安装技能（ID 可以是中心标识符或直接 https://…/SKILL.md URL；当 frontmatter 没有名称时使用 --name 覆盖）
hermes skills inspect ID    预览而不安装
hermes skills config        每平台启用/禁用技能
hermes skills check         检查更新
hermes skills update        更新过时的技能
hermes skills uninstall N   移除中心技能
hermes skills publish PATH  发布到注册表
hermes skills browse        浏览所有可用技能
hermes skills tap add REPO  将 GitHub 仓库添加为技能来源
```

### MCP 服务器

```
hermes mcp serve            将 Hermes 作为 MCP 服务器运行
hermes mcp add NAME         添加 MCP 服务器（--url 或 --command）
hermes mcp remove NAME      移除 MCP 服务器
hermes mcp list             列出已配置的服务器
hermes mcp test NAME        测试连接
hermes mcp configure NAME   切换工具选择
```

### 网关（消息平台）

```
hermes gateway run          在前台启动网关
hermes gateway install      安装为后台服务
hermes gateway start/stop   控制服务
hermes gateway restart      重启服务
hermes gateway status       检查状态
hermes gateway setup        配置平台
```

支持的平台：Telegram、Discord、Slack、WhatsApp、Signal、Email、SMS、Matrix、Mattermost、Home Assistant、DingTalk、Feishu、WeCom、BlueBubbles（iMessage）、Weixin（WeChat）、Microsoft Teams、API Server、Webhooks。Open WebUI 通过 API Server 适配器连接。

平台文档：https://hermes-agent.nousresearch.com/docs/user-guide/messaging/

### 会话

```
hermes sessions list        列出最近会话
hermes sessions browse      交互式选择器
hermes sessions export OUT  导出为 JSONL
hermes sessions rename ID T 重命名会话
hermes sessions delete ID   删除会话
hermes sessions prune       清理旧会话（--older-than N days）
hermes sessions stats       会话存储统计
```

### Cron 任务

```
hermes cron list            列出任务（--all 包含已禁用的）
hermes cron create SCHED    创建：'30m'、'every 2h'、'0 9 * * *'
hermes cron edit ID         编辑计划、提示、传递
hermes cron pause/resume ID 控制任务状态
hermes cron run ID          在下次触发时运行
hermes cron remove ID       删除任务
hermes cron status          调度器状态
```

### Webhooks

```
hermes webhook subscribe N  在 /webhooks/<name> 创建路由
hermes webhook list         列出订阅
hermes webhook remove NAME  移除订阅
hermes webhook test NAME    发送测试 POST
```

### 配置文件

```
hermes profile list         列出所有配置文件
hermes profile create NAME  创建（--clone、--clone-all、--clone-from）
hermes profile use NAME     设置粘性默认值
hermes profile delete NAME  删除配置文件
hermes profile show NAME    显示详情
hermes profile alias NAME   管理包装脚本
hermes profile rename A B   重命名配置文件
hermes profile export NAME  导出为 tar.gz
hermes profile import FILE  从归档导入
```

### 凭据池

```
hermes auth add             交互式凭据向导
hermes auth list [PROVIDER] 列出已池化的凭据
hermes auth remove P INDEX  通过提供商 + 索引移除
hermes auth reset PROVIDER  清除耗尽状态
```

### 其他

```
hermes insights [--days N]  使用分析
hermes update               更新到最新版本
hermes pairing list/approve/revoke  DM 授权
hermes plugins list/install/remove  插件管理
hermes honcho setup/status  Honcho 记忆集成（需要 honcho 插件）
hermes memory setup/status/off  记忆提供商配置
hermes completion bash|zsh  Shell 补全
hermes acp                  ACP 服务器（IDE 集成）
hermes claw migrate         从 OpenClaw 迁移
hermes uninstall            卸载 Hermes
```

---

## 斜杠命令（会话内）

在交互式聊天会话中输入这些命令。

### 会话控制
```
/new (/reset)        新会话
/clear               清除屏幕 + 新会话（CLI）
/retry               重新发送上条消息
/undo                移除最后一次交流
/title [name]        命名会话
/compress            手动压缩上下文
/stop                终止后台进程
/rollback [N]        恢复文件系统检查点
/background <prompt> 在后台运行提示
/queue <prompt>      排队等待下一轮
/resume [name]       恢复命名会话
```

### 配置
```
/config              显示配置（CLI）
/model [name]        显示或更改模型
/personality [name]  设置个性
/reasoning [level]   设置推理（none|minimal|low|medium|high|xhigh|show|hide）
/verbose             循环：off → new → all → verbose
/voice [on|off|tts]  语音模式
/yolo                切换审批绕过
/skin [name]         更改主题（CLI）
/statusbar           切换状态栏（CLI）
```

### 工具和技能
```
/tools               管理工具（CLI）
/toolsets            列出工具集（CLI）
/skills              搜索/安装技能（CLI）
/skill <name>        将技能加载到会话
/cron                管理 cron 任务（CLI）
/reload-mcp          重新加载 MCP 服务器
/plugins             列出插件（CLI）
```

### 网关
```
/approve             批准待处理命令（网关）
/deny                拒绝待处理命令（网关）
/restart             重启网关（网关）
/sethome             将当前聊天设为主频道（网关）
/update              将 Hermes 更新到最新（网关）
/platforms (/gateway) 显示平台连接状态（网关）
```

### 实用工具
```
/branch (/fork)      分叉当前会话
/fast                切换优先/快速处理
/browser             打开 CDP 浏览器连接
/history             显示对话历史（CLI）
/save                将对话保存到文件（CLI）
/paste               附加剪贴板图片（CLI）
/image               附加本地图片文件（CLI）
```

### 信息
```
/help                显示命令
/commands [page]     浏览所有命令（网关）
/usage               令牌使用量
/insights [days]     使用分析
/status              会话信息（网关）
/profile             活动配置文件信息
```

### 退出
```
/quit (/exit, /q)    退出 CLI
```

---

## 关键路径和配置

```
~/.hermes/config.yaml       主配置
~/.hermes/.env              API 密钥和密钥
$HERMES_HOME/skills/        已安装的技能
~/.hermes/sessions/         会话记录
~/.hermes/logs/             网关和错误日志
~/.hermes/auth.json         OAuth 令牌和凭据池
~/.hermes/hermes-agent/     源代码（如果通过 git 安装）
```

配置文件使用 `~/.hermes/profiles/<name>/` 相同布局。

### 配置部分

使用 `hermes config edit` 或 `hermes config set section.key value` 编辑。

| 部分 | 关键选项 |
|---------|-------------|
| `model` | `default`、`provider`、`base_url`、`api_key`、`context_length` |
| `agent` | `max_turns`（90）、`tool_use_enforcement` |
| `terminal` | `backend`（local/docker/ssh/modal）、`cwd`、`timeout`（180） |
| `compression` | `enabled`、`threshold`（0.50）、`target_ratio`（0.20） |
| `display` | `skin`、`tool_progress`、`show_reasoning`、`show_cost` |
| `stt` | `enabled`、`provider`（local/groq/openai/mistral） |
| `tts` | `provider`（edge/elevenlabs/openai/minimax/mistral/neutts） |
| `memory` | `memory_enabled`、`user_profile_enabled`、`provider` |
| `security` | `tirith_enabled`、`website_blocklist` |
| `delegation` | `model`、`provider`、`base_url`、`api_key`、`max_iterations`（50）、`reasoning_effort` |
| `checkpoints` | `enabled`、`max_snapshots`（50） |

完整配置参考：https://hermes-agent.nousresearch.com/docs/user-guide/configuration

### 提供商

支持 20+ 提供商。通过 `hermes model` 或 `hermes setup` 设置。

| 提供商 | 认证 | 密钥环境变量 |
|----------|------|-------------|
| OpenRouter | API 密钥 | `OPENROUTER_API_KEY` |
| Anthropic | API 密钥 | `ANTHROPIC_API_KEY` |
| Nous Portal | OAuth | `hermes auth` |
| OpenAI Codex | OAuth | `hermes auth` |
| GitHub Copilot | Token | `COPILOT_GITHUB_TOKEN` |
| Google Gemini | API 密钥 | `GOOGLE_API_KEY` 或 `GEMINI_API_KEY` |
| DeepSeek | API 密钥 | `DEEPSEEK_API_KEY` |
| xAI / Grok | API 密钥 | `XAI_API_KEY` |
| Hugging Face | Token | `HF_TOKEN` |
| Z.AI / GLM | API 密钥 | `GLM_API_KEY` |
| MiniMax | API 密钥 | `MINIMAX_API_KEY` |
| MiniMax CN | API 密钥 | `MINIMAX_CN_API_KEY` |
| Kimi / Moonshot | API 密钥 | `KIMI_API_KEY` |
| Alibaba / DashScope | API 密钥 | `DASHSCOPE_API_KEY` |
| Xiaomi MiMo | API 密钥 | `XIAOMI_API_KEY` |
| Kilo Code | API 密钥 | `KILOCODE_API_KEY` |
| AI Gateway (Vercel) | API 密钥 | `AI_GATEWAY_API_KEY` |
| OpenCode Zen | API 密钥 | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | API 密钥 | `OPENCODE_GO_API_KEY` |
| Qwen OAuth | OAuth | `hermes login --provider qwen-oauth` |
| 自定义端点 | 配置 | config.yaml 中的 `model.base_url` + `model.api_key` |
| GitHub Copilot ACP | 外部 | `COPILOT_CLI_PATH` 或 Copilot CLI |

完整提供商文档：https://hermes-agent.nousresearch.com/docs/integrations/providers

### 工具集

通过 `hermes tools`（交互式）或 `hermes tools enable/disable NAME` 启用/禁用。

| 工具集 | 提供的功能 |
|---------|-----------------|
| `web` | Web 搜索和内容提取 |
| `browser` | 浏览器自动化（Browserbase、Camofox 或本地 Chromium） |
| `terminal` | Shell 命令和进程管理 |
| `file` | 文件读/写/搜索/补丁 |
| `code_execution` | 沙箱 Python 执行 |
| `vision` | 图像分析 |
| `image_gen` | AI 图像生成 |
| `tts` | 文字转语音 |
| `skills` | 技能浏览和管理 |
| `memory` | 持久跨会话记忆 |
| `session_search` | 搜索过去的对话 |
| `delegation` | 子代理任务委托 |
| `cronjob` | 计划任务管理 |
| `clarify` | 向用户提问澄清问题 |
| `messaging` | 跨平台消息发送 |
| `search` | 仅 Web 搜索（`web` 的子集） |
| `todo` | 会话内任务规划和追踪 |
| `rl` | 强化学习工具（默认关闭） |
| `moa` | Mixture of Agents（默认关闭） |
| `homeassistant` | 智能家居控制（默认关闭） |

工具更改在 `/reset`（新会话）后生效。它们**不适用**于会话中以保留提示缓存。

---

## 安全和隐私开关

常见的"为什么 Hermes 对我的输出/工具调用/命令做 X？"开关 — 以及更改它们的确切命令。其中大多数需要新会话（聊天中的 `/reset` 或启动新的 `hermes` 调用），因为它们在启动时读取一次。

### 工具输出中的密钥编辑

密钥编辑**默认关闭** — 工具输出（终端 stdout、`read_file`、web 内容、子代理摘要等）未经修改地传递。如果用户希望 Hermes 在 API 密钥、令牌和密钥进入对话上下文和日志之前自动掩码：

```bash
hermes config set security.redact_secrets true       # 全局启用
```

**需要重启。** `security.redact_secrets` 在导入时快照 — 在会话中（例如通过工具调用 `export HERMES_REDACT_SECRETS=true`）切换它**不会**对正在运行的进程生效。告诉用户在终端运行 `hermes config set security.redact_secrets true`，然后开始新会话。这是故意的 — 防止 LLM 在任务中途自行切换开关。

再次禁用：
```bash
hermes config set security.redact_secrets false
```

### 网关消息中的 PII 编辑

与密钥编辑分开。启用后，网关在将用户 ID 哈希处理并从模型可见的会话上下文中去除电话号码之前：

```bash
hermes config set privacy.redact_pii true    # 启用
hermes config set privacy.redact_pii false   # 禁用（默认）
```

### 命令审批提示

默认情况下（`approvals.mode: manual`），Hermes 在运行被标记为破坏性的 shell 命令（`rm -rf`、`git reset --hard` 等）之前提示用户。模式为：

- `manual` — 始终提示（默认）
- `smart` — 使用辅助 LLM 自动批准低风险命令，对高风险提示
- `off` — 跳过所有审批提示（等同于 `--yolo`）

```bash
hermes config set approvals.mode smart       # 推荐的中间选项
hermes config set approvals.mode off         # 绕过所有（不推荐）
```

每次调用绕过而不更改配置：
- `hermes --yolo …`
- `export HERMES_YOLO_MODE=1`

注意：YOLO / `approvals.mode: off` 不会关闭密钥编辑。它们是独立的。

### Shell 钩子允许列表

某些 shell 钩子集成在运行之前需要明确允许列表。通过 `~/.hermes/shell-hooks-allowlist.json` 管理 — 钩子第一次想要运行时交互式提示。

### 禁用 web/browser/image-gen 工具

要完全阻止模型访问网络或媒体工具，打开 `hermes tools` 并按平台切换。在下次会话时生效（`/reset`）。参见上面的工具和技能部分。

---

## 语音和转录

### STT（语音 → 文本）

来自消息平台的语音消息自动转录。

提供商优先级（自动检测）：
1. **本地 faster-whisper** — 免费，无需 API 密钥：`pip install faster-whisper`
2. **Groq Whisper** — 免费层：设置 `GROQ_API_KEY`
3. **OpenAI Whisper** — 付费：设置 `VOICE_TOOLS_OPENAI_KEY`
4. **Mistral Voxtral** — 设置 `MISTRAL_API_KEY`

配置：
```yaml
stt:
  enabled: true
  provider: local        # local, groq, openai, mistral
  local:
    model: base          # tiny, base, small, medium, large-v3
```

### TTS（文本 → 语音）

| 提供商 | 环境变量 | 免费？ |
|----------|---------|-------|
| Edge TTS | 无 | 是（默认） |
| ElevenLabs | `ELEVENLABS_API_KEY` | 免费层 |
| OpenAI | `VOICE_TOOLS_OPENAI_KEY` | 付费 |
| MiniMax | `MINIMAX_API_KEY` | 付费 |
| Mistral (Voxtral) | `MISTRAL_API_KEY` | 付费 |
| NeuTTS（本地） | 无（`pip install neutts[all]` + `espeak-ng`） | 免费 |

语音命令：`/voice on`（语音到语音）、`/voice tts`（始终语音）、`/voice off`。

---

## 生成额外 Hermes 实例

将额外的 Hermes 进程作为完全独立的子进程运行 — 独立的会话、工具和环境。

### 何时用这个 vs delegate_task

| | `delegate_task` | 生成 `hermes` 进程 |
|-|-----------------|--------------------------|
| 隔离 | 独立对话，共享进程 | 完全独立进程 |
| 持续时间 | 分钟（受父循环限制） | 小时/天 |
| 工具访问 | 父工具的子集 | 完整工具访问 |
| 交互式 | 否 | 是（PTY 模式） |
| 用例 | 快速并行子任务 | 长期自主任务 |

### 一次性模式

```
terminal(command="hermes chat -q 'Research GRPO papers and write summary to ~/research/grpo.md'", timeout=300)

# 长任务后台运行：
terminal(command="hermes chat -q 'Set up CI/CD for ~/myapp'", background=true)
```

### 通过 tmux 的交互式 PTY 模式

Hermes 使用 prompt_toolkit，需要真实终端。使用 tmux 进行交互式生成：

```
# 启动
terminal(command="tmux new-session -d -s agent1 -x 120 -y 40 'hermes'", timeout=10)

# 等待启动，然后发送消息
terminal(command="sleep 8 && tmux send-keys -t agent1 'Build a FastAPI auth service' Enter", timeout=15)

# 读取输出
terminal(command="sleep 20 && tmux capture-pane -t agent1 -p", timeout=5)

# 发送后续
terminal(command="tmux send-keys -t agent1 'Add rate limiting middleware' Enter", timeout=5)

# 退出
terminal(command="tmux send-keys -t agent1 '/exit' Enter && sleep 2 && tmux kill-session -t agent1", timeout=10)
```

### 多代理协调

```
# 代理 A：后端
terminal(command="tmux new-session -d -s backend -x 120 -y 40 'hermes -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t backend 'Build REST API for user management' Enter", timeout=15)

# 代理 B：前端
terminal(command="tmux new-session -d -s frontend -x 120 -y 40 'hermes -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t frontend 'Build React dashboard for user management' Enter", timeout=15)

# 检查进度，在它们之间中继上下文
terminal(command="tmux capture-pane -t backend -p | tail -30", timeout=5)
terminal(command="tmux send-keys -t frontend 'Here is the API schema from the backend agent: ...' Enter", timeout=5)
```

### 会话恢复

```
# 恢复最近会话
terminal(command="tmux new-session -d -s resumed 'hermes --continue'", timeout=10)

# 恢复特定会话
terminal(command="tmux new-session -d -s resumed 'hermes --resume 20260225_143052_a1b2c3'", timeout=10)
```

### 提示

- **快速子任务优先使用 `delegate_task`** — 比生成完整进程开销少
- **生成编辑代码的代理时使用 `-w`（工作树模式）** — 防止 git 冲突
- **为一次性模式设置超时** — 复杂任务可能需要 5-10 分钟
- **使用 `hermes chat -q` 进行即发即忘** — 不需要 PTY
- **交互式会话使用 tmux** — 原始 PTY 模式在 prompt_toolkit 中有 `\r` vs `\n` 问题
- **对于计划任务**，使用 `cronjob` 工具而不是生成 — 处理传递和重试

---

## 故障排除

### 语音不工作
1. 检查 config.yaml 中的 `stt.enabled: true`
2. 验证提供商：`pip install faster-whisper` 或设置 API 密钥
3. 在网关中：`/restart`。在 CLI 中：退出并重新启动。

### 工具不可用
1. `hermes tools` — 检查工具集是否为你的平台启用
2. 某些工具需要环境变量（检查 `.env`）
3. 启用工具后 `/reset`

### 模型/提供商问题
1. `hermes doctor` — 检查配置和依赖
2. `hermes login` — 重新认证 OAuth 提供商
3. 检查 `.env` 是否有正确的 API 密钥
4. **Copilot 403**：`gh auth login` 令牌**不适用**于 Copilot API。必须通过 `hermes model` → GitHub Copilot 使用 Copilot 特定的 OAuth 设备代码流。

### 更改未生效
- **工具/技能：** `/reset` 以更新的工具集开始新会话
- **配置更改：** 在网关中：`/restart`。在 CLI 中：退出并重新启动。
- **代码更改：** 重启 CLI 或网关进程

### 技能不显示
1. `hermes skills list` — 验证已安装
2. `hermes skills config` — 检查平台启用情况
3. 显式加载：`/skill name` 或 `hermes -s name`

### 网关问题
先检查日志：
```bash
grep -i "failed to send\|error" ~/.hermes/logs/gateway.log | tail -20
```

常见网关问题：
- **SSH 退出后网关挂掉**：启用 linger：`sudo loginctl enable-linger $USER`
- **WSL2 关闭后网关挂掉**：WSL2 需要 `/etc/wsl.conf` 中的 `systemd=true` 才能让 systemd 服务工作。没有它，网关回退到 `nohup`（会话关闭时挂掉）。
- **网关崩溃循环**：重置失败状态：`systemctl --user reset-failed hermes-gateway`

### 平台特定问题
- **Discord 机器人静默**：必须在 Bot → Privileged Gateway Intents 中启用 **Message Content Intent**。
- **Slack 机器人仅在 DM 中工作**：必须订阅 `message.channels` 事件。没有它，机器人忽略公共频道。
- **Windows HTTP 400 "No models provided"**：配置文件编码问题（BOM）。确保 `config.yaml` 保存为 UTF-8 无 BOM。

### 辅助模型不工作
如果 `auxiliary` 任务（vision、压缩、session_search）静默失败，`auto` 提供商找不到后端。设置 `OPENROUTER_API_KEY` 或 `GOOGLE_API_KEY`，或明确配置每个辅助任务的提供商：
```bash
hermes config set auxiliary.vision.provider <your_provider>
hermes config set auxiliary.vision.model <model_name>
```

---

## 在哪里找到东西

| 寻找... | 位置 |
|----------------|----------|
| 配置选项 | `hermes config edit` 或[配置文档](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) |
| 可用工具 | `hermes tools list` 或[工具参考](https://hermes-agent.nousresearch.com/docs/reference/tools-reference) |
| 斜杠命令 | 会话中的 `/help` 或[斜杠命令参考](https://hermes-agent.nousresearch.com/docs/reference/slash-commands) |
| 技能目录 | `hermes skills browse` 或[技能目录](https://hermes-agent.nousresearch.com/docs/reference/skills-catalog) |
| 提供商设置 | `hermes model` 或[提供商指南](https://hermes-agent.nousresearch.com/docs/integrations/providers) |
| 平台设置 | `hermes gateway setup` 或[消息文档](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/) |
| MCP 服务器 | `hermes mcp list` 或[MCP 指南](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) |
| 配置文件 | `hermes profile list` 或[配置文件文档](https://hermes-agent.nousresearch.com/docs/user-guide/profiles) |
| Cron 任务 | `hermes cron list` 或[Cron 文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) |
| 记忆 | `hermes memory status` 或[记忆文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) |
| 环境变量 | `hermes config env-path` 或[环境变量参考](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) |
| CLI 命令 | `hermes --help` 或[CLI 参考](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) |
| 网关日志 | `~/.hermes/logs/gateway.log` |
| 会话文件 | `~/.hermes/sessions/` 或 `hermes sessions browse` |
| 源代码 | `~/.hermes/hermes-agent/` |

---

## 贡献者快速参考

对于偶尔的贡献者和 PR 作者。完整开发者文档：https://hermes-agent.nousresearch.com/docs/developer-guide/

### 项目结构

<!-- ascii-guard-ignore -->
```
hermes-agent/
├── run_agent.py          # AIAgent — 核心对话循环
├── model_tools.py        # 工具发现和调度
├── toolsets.py           # 工具集定义
├── cli.py                # 交互式 CLI（HermesCLI）
├── hermes_state.py       # SQLite 会话存储
├── agent/                # 提示构建器、上下文压缩、记忆、模型路由、凭据池、技能调度
├── hermes_cli/           # CLI 子命令、配置、设置、命令
│   ├── commands.py       # 斜杠命令注册表（CommandDef）
│   ├── config.py         # DEFAULT_CONFIG、环境变量定义
│   └── main.py           # CLI 入口点和 argparse
├── tools/                # 每个工具一个文件
│   └── registry.py       # 中央工具注册表
├── gateway/              # 消息网关
│   └── platforms/        # 平台适配器（telegram、discord 等）
├── cron/                 # 任务调度器
├── tests/                # ~3000 个 pytest 测试
└── website/              # Docusaurus 文档站点
```
<!-- ascii-guard-ignore-end -->

配置：`~/.hermes/config.yaml`（设置）、`~/.hermes/.env`（API 密钥）。

### 添加工具（3 个文件）

**1. 创建 `tools/your_tool.py`：**
```python
import json, os
from tools.registry import registry

def check_requirements() -> bool:
    return bool(os.getenv("EXAMPLE_API_KEY"))

def example_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: example_tool(
        param=args.get("param", ""), task_id=kw.get("task_id")),
    check_fn=check_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

**2. 添加到 `toolsets.py`** → `_HERMES_CORE_TOOLS` 列表。

自动发现：任何具有顶级 `registry.register()` 调用的 `tools/*.py` 文件都会自动导入 — 无需手动列表。

所有处理器必须返回 JSON 字符串。使用 `get_hermes_home()` 获取路径，不要硬编码 `~/.hermes`。

### 添加斜杠命令

1. 在 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY` 中添加 `CommandDef`
2. 在 `cli.py` → `process_command()` 中添加处理器
3. （可选）在 `gateway/run.py` 中添加网关处理器

所有消费者（帮助文本、自动完成、Telegram 菜单、Slack 映射）自动从中央注册表派生。

### 代理循环（高级）

```
run_conversation():
  1. 构建系统提示
  2. 循环 while iterations < max:
     a. 调用 LLM（OpenAI 格式消息 + 工具 schema）
     b. 如果 tool_calls → 通过 handle_function_call() 调度各个 → 追加结果 → 继续
     c. 如果文本响应 → 返回
  3. 接近令牌限制时自动触发上下文压缩
```

### 测试

```bash
python -m pytest tests/ -o 'addopts=' -q   # 完整套件
python -m pytest tests/tools/ -q            # 特定区域
```

- 测试自动将 `HERMES_HOME` 重定向到临时目录 — 永不接触真实的 `~/.hermes/`
- 推送任何更改前运行完整套件
- 使用 `-o 'addopts='` 清除任何内置的 pytest 标志

### 提交约定

```
type: concise subject line

Optional body.
```

类型：`fix:`、`feat:`、`refactor:`、`docs:`、`chore:`

### 关键规则

- **不要破坏提示缓存** — 不要在会话中途更改上下文、工具或系统提示
- **消息角色交替** — 不要连续两条助手或用户消息
- 对所有路径使用 `hermes_constants` 中的 `get_hermes_home()`（配置文件安全）
- 配置值放入 `config.yaml`，密钥放入 `.env`
- 新工具需要 `check_fn`，仅在满足要求时出现
