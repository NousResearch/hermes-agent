---
sidebar_position: 11
title: "ACP 宿主集成"
description: "在兼容 ACP 的编辑器和协作平台中使用 Hermes Agent"
---

# ACP 宿主集成

Hermes Agent 可作为 ACP 服务器运行，让兼容 ACP 的编辑器通过 stdio 与 Hermes 通信并渲染：

- 聊天消息
- 工具活动
- 文件差异
- 终端命令
- 审批 prompt（提示词）
- 流式思考 / 响应块

当你希望 Hermes 表现得像编辑器原生的编码 agent，而非独立 CLI 或消息机器人时，ACP 是合适的选择。

## Hermes 在 ACP 模式下暴露的内容

Hermes 使用专为编辑器工作流设计的精选 `hermes-acp` 工具集运行，包括：

- 文件工具：`read_file`、`write_file`、`patch`、`search_files`
- 终端工具：`terminal`、`process`
- 网页/浏览器工具
- 记忆、待办事项、会话搜索
- skills
- `execute_code` 和 `delegate_task`
- 视觉

它有意排除了不适合典型编辑器 UX 的功能，例如消息投递和 cronjob 管理。

## 安装

正常安装 Hermes 后，从安装检出目录添加 ACP 扩展：

```bash
cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'
```

这将安装 `agent-client-protocol` 依赖并启用：

- `hermes acp`
- `hermes-acp`
- `python -m acp_adapter`

## 启动 ACP 服务器

以下任意命令均可以 ACP 模式启动 Hermes：

```bash
hermes acp
```

```bash
hermes-acp
```

```bash
python -m acp_adapter
```

Hermes 将日志输出到 stderr，以保留 stdout 用于 ACP JSON-RPC 流量。

非交互式检查：

```bash
hermes acp --version
hermes acp --check
```

### 浏览器工具（可选）

浏览器工具（`browser_navigate`、`browser_click` 等）依赖 `agent-browser` npm 包和 Chromium，这些不包含在 Python wheel 中。通过以下命令安装：

```bash
hermes acp --setup-browser           # 交互式（下载约 400 MB 前会提示确认）
hermes acp --setup-browser --yes     # 非交互式接受下载
```

这是独立命令。终端认证流程（`hermes acp --setup`）在模型选择后也会将浏览器引导作为后续问题提供，因此大多数用户无需直接运行 `--setup-browser`。

具体操作：

- 若缺少 Node.js 22 LTS，将其安装到 `~/.hermes/node/`
- 将 `npm install -g agent-browser @askjo/camofox-browser` 安装到该前缀（无需 sudo — `npm` 的 `--prefix` 指向用户可写的 Hermes 管理 Node）
- 安装 Playwright Chromium，或在检测到系统 Chrome/Chromium 时使用已有版本

该引导过程是幂等的——重复运行速度很快，已完成的步骤会被跳过。

## 宿主设置

### Buzz 频道（中继桥接）

[Buzz](https://github.com/block/buzz) 是一个基于 Nostr 的人机协作平台。
其 `buzz-acp` harness 通过 stdio 将 Buzz 频道连接到任意 ACP agent：

```text
Buzz relay <-- WebSocket --> buzz-acp <-- ACP over stdio --> Hermes Agent
```

这是一种传输层集成，不是第二个 Hermes 安装。由 `buzz-acp` 启动的子进程使用该主机上
与 `hermes` 相同的配置、凭据、记忆、技能和状态。

（这与 [Buzz Desktop 的托管运行时](#buzz-desktop)不同——后者在本地将 Hermes 作为
预设 harness 启动。中继桥接用于以 agent 身份加入 Buzz *频道*，通常部署在服务器上。）

前置条件：

- 完成上文的 ACP 安装并通过 `hermes acp --check`。
- 从 [Buzz 仓库](https://github.com/block/buzz)构建 `buzz-acp` 和 `buzz` CLI
  （`cargo build --release -p buzz-acp`）。
- 为 Hermes 铸造专用的 Nostr 密钥对（`buzz-admin generate-key`）并将其注册为
  中继成员（`buzz-admin add-member`）。每个 agent 都需要自己的身份——不要复用
  人类的密钥对。
- 将该身份加入目标 Buzz 频道。

启动桥接：

```bash
export BUZZ_RELAY_URL="wss://community.example.com"
export BUZZ_PRIVATE_KEY="..."
export BUZZ_API_TOKEN="..."
export BUZZ_ACP_AGENT_COMMAND="hermes"
export BUZZ_ACP_AGENT_ARGS="acp"

buzz-acp
```

仅当中继强制 token 认证时才需要 `BUZZ_API_TOKEN`。切勿提交或粘贴私钥和 API token。

若要持久化部署到服务器，请以拥有目标 Hermes home 的同一操作系统用户身份，
在服务管理器下运行 `buzz-acp`。安装、密钥生成、频道发现和各项 agent 选项见
[buzz-acp README](https://github.com/block/buzz/tree/main/crates/buzz-acp)。

桥接会发现 Hermes 身份所属的每个 Buzz 频道，并在其被加入新频道时自动订阅。
因此 Buzz 频道成员资格就是访问边界；Hermes 自身配置中无需单独的频道列表。

若要在所有者的 Buzz Desktop 中展示 Hermes 的 ACP 活动，添加：

```bash
export BUZZ_ACP_RELAY_OBSERVER="true"
```

这会发布加密的 kind `24200` 观察者帧（Buzz 的 NIP-AO），仅所有者可解密。
Desktop 会在该 agent 的 **Activity log** 中实时渲染生命周期、工具、响应和用量流。
中继将这些帧视为临时数据，因此 Desktop 必须在回合开始前在线；其本地观察者归档
才是所有者侧的持久历史。

无头桥接会自行回应 ACP 权限请求，因为没有编辑器来展示审批对话框——参见
[将 Buzz agent 保持为 owner-only](#将-buzz-agent-保持为-owner-only)。请将桥接视为
特权自动化：使用专用操作系统账户，限制哪些 Buzz 用户可以触发 agent
（`buzz-acp` 通过 `BUZZ_ACP_AGENT_OWNER` 支持仅所有者响应门控），
并仅在预期 Hermes 工作的频道中授予成员资格。

## 编辑器设置

### VS Code

安装 [ACP Client](https://marketplace.visualstudio.com/items?itemName=formulahendry.acp-client) 扩展。

连接步骤：

1. 从活动栏打开 ACP Client 面板。
2. 从内置 agent 列表中选择 **Hermes Agent**。
3. 连接并开始聊天。

如需手动定义 Hermes，通过 VS Code 设置在 `acp.agents` 下添加：

```json
{
  "acp.agents": {
    "Hermes Agent": {
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

### Zed

在 Zed 设置中将 Hermes 配置为自定义 agent 服务器：

1. 打开 Agent 面板。
2. 使用以下配置添加自定义 agent 服务器：

```json
{
  "agent_servers": {
    "hermes-agent": {
      "type": "custom",
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

3. 启动新的 Hermes 外部 agent 线程。

前提条件：

- 先通过 `hermes model` 配置 Hermes provider 凭据，或在 `~/.hermes/.env` / `~/.hermes/config.yaml` 中设置。

### JetBrains

使用兼容 ACP 的插件并将其指向 `hermes acp` 或 `hermes-acp`。

### Buzz Desktop

[Buzz](https://github.com/block/buzz) 将 Hermes Agent 作为预设运行时提供。
按常规方式安装 Hermes 后，Buzz 会自动发现它 —— 打开 **Settings → Runtimes**，
Hermes 就会出现在你的运行时列表中。

如果发现失败（较旧的安装），请确认 ACP 启动器可以在登录 shell 的 PATH 上解析：

```bash
command -v hermes-acp || command -v hermes
```

较新的安装会将 `hermes` 和 `hermes-acp` 两个启动器写入 `~/.local/bin`；
运行 `hermes update` 会为较旧的安装补上 `hermes-acp` 启动器。作为手动兜底方案，
可以将 Buzz 的 agent 命令配置为 `hermes`，参数为 `["acp"]`。

#### 将 Buzz agent 保持为 owner-only

Buzz 创建的每个 agent 默认都将 **Who can talk to this agent** 设为 `Owner only`。
当运行时为 Hermes 时，请保持该设置。

这条路径上有两种行为叠加。`hermes-acp` 工具集包含 `terminal` 和 `execute_code`，
而 Buzz 的 ACP 桥接层会自行以 `allow_once` 回应 Hermes 的权限请求，不会转交给你确认。
因此 Buzz 中的 Hermes agent 会在不提示的情况下在宿主机上执行 shell 命令。
让它对一个临时目录执行 `rm -rf`，该目录会被直接删除，全程没有任何提示。

将该设置改为 `Anyone`，等于把同样的 shell 访问权限交给频道中的每一位发言者。
Buzz 在你选择该选项时不会给出任何警告。

目前两种看起来可行的缓解手段都无效：

- `approvals.mode: manual` 确实会让 Hermes 发出权限请求，但 Buzz 仍会自动批准，
  命令照样执行。
- `platform_toolsets.acp` 不会收窄 ACP 工具集，因此无法用它去掉 `terminal`。

来自 owner 的 `!shutdown` 在任何模式下都能停止 agent，而 Buzz 会忽略其他人发出的同一命令。

## 配置与凭据

ACP 模式使用与 CLI 相同的 Hermes 配置：

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

Provider 解析使用 Hermes 的正常运行时解析器，因此 ACP 继承当前配置的 provider 和凭据。Hermes 还为首次运行的 ACP 客户端提供终端认证方法（`--setup`）；这将打开 Hermes 的交互式模型/provider 设置。

## 会话行为

ACP 会话在服务器运行期间由 ACP 适配器的内存会话管理器跟踪。

每个会话存储：

- 会话 ID
- 工作目录
- 已选模型
- 当前对话历史
- 取消事件

底层 `AIAgent` 仍使用 Hermes 的正常持久化/日志路径，但 ACP 的 `list/load/resume/fork` 仅限于当前运行的 ACP 服务器进程。

## 工作目录行为

ACP 会话将编辑器的 cwd 绑定到 Hermes 任务 ID，使文件和终端工具相对于编辑器工作区运行，而非服务器进程的 cwd。

## 审批

危险的终端命令可作为审批 prompt 路由回编辑器。ACP 审批选项比 CLI 流程更简单：

- 允许一次
- 始终允许
- 拒绝

你是否真的会看到提示取决于宿主端。宿主可以用程序方式直接回应该请求而不展示给你，
此时这些选项只存在于协议层面，永远不会到达人类手中。Buzz Desktop 就是这样做的，
因此无论你的 `approvals` 如何设置，都应把该路径视为无人值守执行。

超时或出错时，审批桥接会拒绝请求。

### 会话范围的编辑自动审批

ACP 在*允许一次*和*始终允许*之间提供第三层：**允许本次会话**。在编辑器的权限提示中选择此选项，会将审批记录在当前 ACP 会话内——该会话中所有后续匹配命令无需提示即可通过，但新的 ACP 会话（或重启编辑器）会重置状态，并在第一次时重新提示。

| 选项 | 编辑器标签 | 范围 | 重启后是否持久化 |
|---|---|---|---|
| `allow_once` | 允许一次 | 本次工具调用 | 否 |
| `allow_session` | 允许本次会话 | 本 ACP 会话中所有匹配调用 | 否——会话结束时清除 |
| `allow_always` | 始终允许 | 所有未来会话 | 是（写入 Hermes 永久允许列表） |
| `deny` | 拒绝 | 本次工具调用 | 否 |

`allow_session` 是编辑器工作流的正确默认选项——你在任务期间信任 agent，但不想授予长期允许列表条目。安全权衡很直接：范围越广，编辑器打断你的次数越少，行为异常的 agent（或 prompt 注入）在被发现前能造成的损害也越大。对不熟悉的命令从 `allow_once` 开始；在看到 agent 多次正确运行相同模式后升级为 `allow_session`；将 `allow_always` 保留给你永远信任的真正幂等命令（例如 `git status`）。

ACP 桥接将这些选项映射到 Hermes 的内部审批语义——`allow_always` 与 CLI 相同地写入永久允许列表条目，而 `allow_session` 仅影响当前 ACP 会话的进程内审批缓存。

## 故障排查

### ACP agent 未出现在编辑器中

检查：

- 对于手动/本地开发，验证自定义 `agent_servers` 命令是否指向 `hermes acp`。
- Hermes 已安装且在 PATH 中。
- ACP 扩展已安装（`cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'`）。

### ACP 启动后立即报错

尝试以下检查：

```bash
hermes acp --version
hermes acp --check
hermes doctor
hermes status
```

### 缺少凭据

ACP 模式使用 Hermes 现有的 provider 设置。通过以下方式配置凭据：

```bash
hermes model
```

或编辑 `~/.hermes/.env`。终端认证流程（`hermes acp --setup`）也可以触发交互式 provider/模型设置。

## 另请参阅

- [ACP 内部机制](../../developer-guide/acp-internals.md)
- [Provider 运行时解析](../../developer-guide/provider-runtime.md)
- [工具运行时](../../developer-guide/tools-runtime.md)