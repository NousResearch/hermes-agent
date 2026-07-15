---
sidebar_position: 8
title: "安全"
description: "安全模型、危险命令审批、用户授权、容器隔离及生产部署最佳实践"
---

# 安全

Hermes Agent 采用纵深防御安全模型。本页涵盖所有安全边界——从命令审批到容器隔离，再到消息平台上的用户授权。

## 概述

安全模型共有七层：

1. **用户授权** — 谁可以与 Agent 通信（允许列表、DM 配对）
2. **危险命令审批** — 针对破坏性操作的人工审核环节
3. **容器隔离** — Docker/Singularity/Modal 沙箱及加固配置
4. **MCP 凭据过滤** — MCP 子进程的环境变量隔离
5. **上下文文件扫描** — 检测项目文件中的 prompt（提示词）注入
6. **跨会话隔离** — 会话之间无法访问彼此的数据或状态；cron 任务存储路径已针对路径遍历攻击进行加固
7. **输入清理** — 终端工具后端中的工作目录参数会经过允许列表验证，以防止 shell 注入

## 危险命令审批

在执行任何命令之前，Hermes 会将其与一份精心维护的危险模式列表进行比对。若匹配，用户必须明确批准。

### 审批模式

审批系统支持三种模式，通过 `~/.hermes/config.yaml` 中的 `approvals.mode` 配置：

```yaml
approvals:
  mode: smart     # smart | manual | off
  timeout: 60     # 等待用户响应的秒数（默认：60）
```

| 模式 | 行为 |
|------|----------|
| **smart**（默认） | 使用辅助 LLM 评估风险。低风险命令（如 `python -c "print('hello')"`）仅对当前命令自动批准，真正危险的命令自动拒绝，不确定的情况升级为手动提示。 |
| **manual** | 始终提示用户审批危险命令。 |
| **off** | 禁用所有审批检查——等同于使用 `--yolo` 运行。所有命令无需提示即可执行。 |

:::warning
设置 `approvals.mode: off` 将禁用所有安全提示。仅在受信任的环境（CI/CD、容器等）中使用。
:::

### YOLO 模式

YOLO 模式会绕过当前会话中**所有**危险命令审批提示。可通过以下三种方式激活：

1. **CLI 标志**：使用 `hermes --yolo` 或 `hermes chat --yolo` 启动会话
2. **斜杠命令**：在会话中输入 `/yolo` 以切换开/关
3. **环境变量**：设置 `HERMES_YOLO_MODE=1`

`/yolo` 命令是一个**切换开关**——每次使用都会翻转模式的开/关状态：

```
> /yolo
  ⚡ YOLO mode ON — all commands auto-approved. Use with caution.

> /yolo
  ⚠ YOLO mode OFF — dangerous commands will require approval.
```

YOLO 模式在 CLI 和 gateway 会话中均可使用。在内部，它会设置 `HERMES_YOLO_MODE` 环境变量，该变量在每次命令执行前都会被检查。

当 YOLO 激活时，Hermes 会显示两个持久的视觉提醒，以确保用户不会忘记审批提示已被绕过：

- 当 YOLO 已激活时，会话开始时显示一条红色横幅：`⚠ YOLO mode — all approval prompts bypassed`。YOLO 关闭时隐藏，以保持默认横幅整洁。
- 状态栏中所有宽度层级均显示 `⚠ YOLO` 片段，随着 YOLO 的切换实时更新（富文本渲染器和纯文本回退均支持）。

:::danger
YOLO 模式会禁用会话中**所有**危险命令安全检查——**但硬性黑名单除外**（见下文）。仅在完全信任所生成命令的情况下使用（例如，在一次性环境中经过充分测试的自动化脚本）。
:::

对于破坏性会话斜杠命令（`/clear`、`/new` / `/reset`、`/undo`、`/exit --delete`），CLI 在执行前也会提示确认。参见[斜杠命令——破坏性命令的确认提示](../reference/slash-commands.md#confirmation-prompts-for-destructive-commands)。

### 硬性黑名单（始终生效的底线）

某些命令极具破坏性——不可逆的文件系统清除、fork 炸弹、直接写入块设备——无论以下任何情况，Hermes 都**拒绝**执行：

- `--yolo` / `/yolo` 已开启
- `approvals.mode: off`
- Cron 任务以无头 `approve` 模式运行
- 用户明确点击"始终允许"

黑名单是 `--yolo` 之下的底线。它在审批层看到命令**之前**就会触发，且没有任何覆盖标志。当前涵盖的模式（非详尽列表；与 `tools/approval.py::UNRECOVERABLE_BLOCKLIST` 保持同步）：

| 模式 | 为何列为硬性规则 |
|---|---|
| `rm -rf /` 及明显变体 | 清除文件系统根目录 |
| `rm -rf --no-preserve-root /` | 明确表示"我就是要删根目录"的变体 |
| `:(){ :\|:& };:` （bash fork 炸弹） | 使主机挂起直至重启 |
| `mkfs.*` 作用于已挂载的根设备 | 格式化运行中的系统 |
| `dd if=/dev/zero of=/dev/sd*` | 清零物理磁盘 |
| 将不受信任的 URL 通过管道传给 `sh`（作用于根文件系统顶层） | 远程代码执行攻击面过大，无法批准 |

若触发黑名单，工具调用会向 Agent 返回一条说明性错误，且不执行任何操作。如果某个合法工作流确实需要这些命令（例如，你是一个清除并重装流水线的操作者），请在 Agent 外部运行。

### 审批超时

当危险命令提示出现时，用户有一段可配置的时间来响应。若在超时内未响应，命令将**默认被拒绝**（故障关闭）。

在 `~/.hermes/config.yaml` 中配置超时：

```yaml
approvals:
  timeout: 60  # 秒（默认：60）
```

### 触发审批的条件

以下模式会触发审批提示（定义于 `tools/approval.py`）：

| 模式 | 描述 |
|---------|-------------|
| `rm -r` / `rm --recursive` | 递归删除 |
| `rm ... /` | 在根路径下删除 |
| `chmod 777/666` / `o+w` / `a+w` | 全局/其他用户可写权限 |
| `chmod --recursive` 配合不安全权限 | 递归全局/其他用户可写（长标志） |
| `chown -R root` / `chown --recursive root` | 递归 chown 为 root |
| `mkfs` | 格式化文件系统 |
| `dd if=` | 磁盘复制 |
| `> /dev/sd` | 写入块设备 |
| `DROP TABLE/DATABASE` | SQL DROP |
| `DELETE FROM`（不含 WHERE） | 不含 WHERE 的 SQL DELETE |
| `TRUNCATE TABLE` | SQL TRUNCATE |
| `> /etc/` | 覆盖系统配置 |
| `systemctl stop/restart/disable/mask` | 停止/重启/禁用系统服务 |
| `kill -9 -1` | 杀死所有进程 |
| `pkill -9` | 强制杀死进程 |
| Fork 炸弹模式 | Fork 炸弹 |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | 通过 `-c` 标志执行 shell 命令（包括组合标志如 `-lc`） |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | 通过 `-e`/`-c` 标志执行脚本 |
| `curl ... \| sh` / `wget ... \| sh` | 将远程内容通过管道传给 shell |
| `bash <(curl ...)` / `sh <(wget ...)` | 通过进程替换执行远程脚本 |
| `tee` 写入 `/etc/`、`~/.ssh/`、`~/.hermes/.env` | 通过 tee 覆盖敏感文件 |
| `>` / `>>` 写入 `/etc/`、`~/.ssh/`、`~/.hermes/.env` | 通过重定向覆盖敏感文件 |
| `xargs rm` | xargs 配合 rm |
| `find -exec rm` / `find -delete` | find 配合破坏性操作 |
| `cp`/`mv`/`install` 写入 `/etc/` | 复制/移动文件到系统配置目录 |
| `sed -i` / `sed --in-place` 作用于 `/etc/` | 就地编辑系统配置 |
| `pkill`/`killall` hermes/gateway | 防止自我终止 |
| `gateway run` 配合 `&`/`disown`/`nohup`/`setsid` | 防止在服务管理器外启动 gateway |

:::info
**容器绕过**：在 `docker`、`singularity`、`modal` 或 `daytona` 后端运行时，危险命令检查会被**跳过**，因为容器本身就是安全边界。容器内的破坏性命令不会危害宿主机。
:::

### 审批流程（CLI）

在交互式 CLI 中，危险命令会显示内联审批提示：

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

四个选项：

- **once** — 仅允许本次执行
- **session** — 在本次会话剩余时间内允许此模式
- **always** — 添加到永久允许列表（保存至 `config.yaml`）
- **deny**（默认） — 阻止该命令

### 审批流程（Gateway/消息平台）

在消息平台上，Agent 会将危险命令详情发送到聊天中，并等待用户回复：

- 回复 **yes**、**y**、**approve**、**ok** 或 **go** 以批准
- 回复 **no**、**n**、**deny** 或 **cancel** 以拒绝

运行 gateway 时，`HERMES_EXEC_ASK=1` 环境变量会自动设置。

### 永久允许列表

通过"always"批准的命令会保存到 `~/.hermes/config.yaml`：

```yaml
# 永久允许的危险命令模式
command_allowlist:
  - rm
  - systemctl
```

这些模式在启动时加载，并在所有后续会话中静默批准。

:::tip
使用 `hermes config edit` 查看或删除永久允许列表中的模式。
:::

## 用户授权（Gateway）

运行消息 gateway 时，Hermes 通过分层授权系统控制谁可以与机器人交互。

### 授权检查顺序

`_is_user_authorized()` 方法按以下顺序检查：

1. **每平台允许所有用户标志**（如 `DISCORD_ALLOW_ALL_USERS=true`）
2. **DM 配对已批准列表**（通过配对码批准的用户）
3. **平台专属允许列表**（如 `TELEGRAM_ALLOWED_USERS=12345,67890`）
4. **全局允许列表**（`GATEWAY_ALLOWED_USERS=12345,67890`）
5. **全局允许所有用户**（`GATEWAY_ALLOW_ALL_USERS=true`）
6. **默认：拒绝**

### 平台允许列表

在 `~/.hermes/.env` 中以逗号分隔的值设置允许的用户 ID：

```bash
# 平台专属允许列表
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# 跨平台允许列表（对所有平台均检查）
GATEWAY_ALLOWED_USERS=123456789

# 每平台允许所有用户（谨慎使用）
DISCORD_ALLOW_ALL_USERS=true

# 全局允许所有用户（极度谨慎使用）
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
若**未配置任何允许列表**且未设置 `GATEWAY_ALLOW_ALL_USERS`，则**所有用户均被拒绝**。Gateway 在启动时会记录警告：

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DM 配对系统

为实现更灵活的授权，Hermes 提供了基于验证码的配对系统。无需预先提供用户 ID，未知用户会收到一次性配对码，由机器人所有者通过 CLI 批准。

**工作原理：**

1. 未知用户向机器人发送 DM
2. 机器人回复一个 8 位配对码
3. 机器人所有者在 CLI 上运行 `hermes pairing approve <platform> <code>`
4. 该用户在该平台上获得永久批准

在 `~/.hermes/config.yaml` 中控制未授权私信的处理方式：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` 为默认值。未授权的 DM 会收到配对码回复。
- `ignore` 静默丢弃未授权的 DM。
- 平台部分会覆盖全局默认值，因此可以在 Telegram 上保持配对，同时让 WhatsApp 保持静默。

**安全特性**（基于 OWASP + NIST SP 800-63-4 指南）：

| 特性 | 详情 |
|---------|---------|
| 验证码格式 | 8 位字符，来自 32 位无歧义字母表（不含 0/O/1/I） |
| 随机性 | 密码学安全（`secrets.choice()`） |
| 验证码有效期 | 1 小时过期 |
| 速率限制 | 每用户每 10 分钟 1 次请求 |
| 待处理上限 | 每平台最多 3 个待处理验证码 |
| 锁定 | 5 次失败的批准尝试 → 1 小时锁定 |
| 文件安全 | 所有配对数据文件执行 `chmod 0600` |
| 日志 | 验证码永不记录到 stdout |

**配对 CLI 命令：**

```bash
# 列出待处理和已批准的用户
hermes pairing list

# 批准配对码
hermes pairing approve telegram ABC12DEF

# 撤销用户访问权限
hermes pairing revoke telegram 123456789

# 清除所有待处理验证码
hermes pairing clear-pending
```

**存储：** 配对数据存储于 `~/.hermes/pairing/`，按平台分为独立的 JSON 文件：
- `{platform}-pending.json` — 待处理的配对请求
- `{platform}-approved.json` — 已批准的用户
- `_rate_limits.json` — 速率限制和锁定追踪

## 容器隔离

使用 `docker` 终端后端时，Hermes 对每个容器应用严格的安全加固。

### Docker 安全标志

每个容器均使用以下标志运行（定义于 `tools/environments/docker.py`）：

```python
_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # 丢弃所有 Linux capabilities
    "--cap-add", "DAC_OVERRIDE",                  # root 可写入绑定挂载目录
    "--cap-add", "CHOWN",                         # 包管理器需要文件所有权
    "--cap-add", "FOWNER",                        # 包管理器需要文件所有权
    "--security-opt", "no-new-privileges",         # 阻止权限提升
    "--pids-limit", "256",                         # 限制进程数量
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # 有大小限制的 /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # 禁止执行的 /var/tmp
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",   # 禁止执行的 /run
]
```

### 资源限制

容器资源可在 `~/.hermes/config.yaml` 中配置：

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # 仅显式允许列表；空值可防止密钥进入容器
  container_cpu: 1        # CPU 核心数
  container_memory: 5120  # MB（默认 5GB）
  container_disk: 51200   # MB（默认 50GB，需要 XFS 上的 overlay2）
  container_persistent: true  # 跨会话持久化文件系统
```

### 文件系统持久化

- **持久模式**（`container_persistent: true`）：从 `~/.hermes/sandboxes/docker/<task_id>/` 绑定挂载 `/workspace` 和 `/root`
- **临时模式**（`container_persistent: false`）：工作区使用 tmpfs——清理后所有内容丢失

:::tip
对于生产 gateway 部署，使用 `docker`、`modal` 或 `daytona` 后端，将 Agent 命令与宿主机系统隔离。这样可以完全消除危险命令审批的需要。
:::

:::warning
若向 `terminal.docker_forward_env` 添加名称，这些变量会被有意注入容器供终端命令使用。这对于任务专属凭据（如 `GITHUB_TOKEN`）很有用，但也意味着容器内运行的代码可以读取并泄露这些变量。
:::

## 终端后端安全对比

| 后端 | 隔离 | 危险命令检查 | 适用场景 |
|---------|-----------|-------------------|----------|
| **local** | 无——在宿主机上运行 | ✅ 是 | 开发、受信任用户 |
| **ssh** | 远程机器 | ✅ 是 | 在独立服务器上运行 |
| **docker** | 容器 | ❌ 跳过（容器即边界） | 生产 gateway |
| **singularity** | 容器 | ❌ 跳过 | HPC 环境 |
| **modal** | 云沙箱 | ❌ 跳过 | 可扩展的云隔离 |
| **daytona** | 云沙箱 | ❌ 跳过 | 持久化云工作区 |

## 环境变量透传 {#environment-variable-passthrough}

`execute_code` 和 `terminal` 都会从子进程中剥离敏感环境变量，以防止 LLM 生成的代码泄露凭据。但是，声明了 `required_environment_variables` 的技能（skill）确实需要访问这些变量。

### 工作原理

两种机制允许特定变量通过沙箱过滤器：

**1. 技能作用域透传（自动）**

当技能通过 `skill_view` 或 `/skill` 命令加载，且声明了 `required_environment_variables` 时，环境中实际已设置的这些变量会自动注册为透传变量。尚未设置（仍处于待配置状态）的变量**不会**被注册。

```yaml
# 在技能的 SKILL.md frontmatter 中
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
```

加载此技能后，`TENOR_API_KEY` 会透传到 `execute_code`、`terminal`（本地）**以及远程后端（Docker、Modal）**——无需手动配置。

:::info Docker & Modal
在 v0.5.1 之前，Docker 的 `forward_env` 与技能透传是独立的系统。现在它们已合并——技能声明的环境变量会自动转发到 Docker 容器和 Modal 沙箱，无需手动添加到 `docker_forward_env`。
:::

**2. 基于配置的透传（手动）**

对于未被任何技能声明的环境变量，将其添加到 `config.yaml` 中的 `terminal.env_passthrough`：

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

### 凭据文件透传（OAuth token 等） {#credential-file-passthrough}

某些技能需要在沙箱中访问**文件**（而非仅环境变量）——例如，Google Workspace 将 OAuth token 存储为活跃 profile 的 `HERMES_HOME` 下的 `google_token.json`。技能在 frontmatter 中声明这些文件：

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

加载后，Hermes 会检查这些文件是否存在于活跃 profile 的 `HERMES_HOME` 中，并将其注册为挂载：

- **Docker**：只读绑定挂载（`-v host:container:ro`）
- **Modal**：在沙箱创建时挂载，并在每次命令前同步（处理会话中途的 OAuth 配置）
- **本地**：无需操作（文件已可访问）

也可以在 `config.yaml` 中手动列出凭据文件：

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

路径相对于 `~/.hermes/`。文件在容器内挂载到 `/root/.hermes/`。

### 各沙箱的过滤规则

| 沙箱 | 默认过滤 | 透传覆盖 |
|---------|---------------|---------------------|
| **execute_code** | 阻止名称中包含 `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL`、`PASSWD`、`AUTH` 的变量；仅允许安全前缀变量通过 | ✅ 透传变量绕过两项检查 |
| **terminal**（本地） | 阻止明确的 Hermes 基础设施变量（提供商密钥、gateway token、工具 API 密钥） | ✅ 透传变量绕过黑名单 |
| **terminal**（Docker） | 默认不传入宿主机环境变量 | ✅ 透传变量 + `docker_forward_env` 通过 `-e` 转发 |
| **terminal**（Modal） | 默认不传入宿主机环境/文件 | ✅ 凭据文件挂载；环境变量通过同步透传 |
| **MCP** | 阻止所有变量，仅允许安全系统变量 + 显式配置的 `env` | ❌ 不受透传影响（改用 MCP `env` 配置） |

### 安全注意事项

- 透传仅影响你或你的技能明确声明的变量——任意 LLM 生成代码的默认安全态势不变
- 凭据文件以**只读**方式挂载到 Docker 容器中
- Skills Guard 在安装前会扫描技能内容中的可疑环境变量访问模式
- 缺失/未设置的变量永远不会被注册（不存在的内容无法泄露）
- Hermes 基础设施密钥（提供商 API 密钥、gateway token）不应添加到 `env_passthrough`——它们有专用机制

## MCP 凭据处理

MCP（Model Context Protocol）服务器子进程接收**经过过滤的环境**，以防止意外泄露凭据。

### 安全环境变量

从宿主机传递到 MCP stdio 子进程的变量仅限以下几项：

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

以及所有 `XDG_*` 变量。所有其他环境变量（API 密钥、token、密钥）均被**剥离**。

在 MCP 服务器的 `env` 配置中显式定义的变量会被透传：

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # 仅此变量被传递
```

### 凭据脱敏

MCP 工具的错误消息在返回给 LLM 之前会经过清理。以下模式会被替换为 `[REDACTED]`：

- GitHub PAT（`ghp_...`）
- OpenAI 风格密钥（`sk-...`）
- Bearer token
- `token=`、`key=`、`API_KEY=`、`password=`、`secret=` 参数

### 网站访问策略

你可以限制 Agent 通过其 Web 和浏览器工具可访问的网站。这对于防止 Agent 访问内部服务、管理面板或其他敏感 URL 非常有用。

```yaml
# 在 ~/.hermes/config.yaml 中
security:
  website_blocklist:
    enabled: true
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

当请求被阻止的 URL 时，工具会返回一条错误，说明该域名已被策略阻止。黑名单在 `web_search`、`web_extract`、`browser_navigate` 及所有支持 URL 的工具中均强制执行。

完整详情请参见配置指南中的[网站黑名单](/user-guide/configuration#website-blocklist)。

### SSRF 防护

所有支持 URL 的工具（网页搜索、网页提取、视觉、浏览器）在获取 URL 之前都会进行验证，以防止服务器端请求伪造（SSRF）攻击。被阻止的地址包括：

- **私有网络**（RFC 1918）：`10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`
- **回环地址**：`127.0.0.0/8`、`::1`
- **链路本地地址**：`169.254.0.0/16`（包括 `169.254.169.254` 处的云元数据）
- **CGNAT / 共享地址空间**（RFC 6598）：`100.64.0.0/10`（Tailscale、WireGuard VPN）
- **云元数据主机名**：`metadata.google.internal`、`metadata.goog`
- **保留地址、多播地址和未指定地址**

SSRF 防护对面向互联网的使用始终有效，DNS 失败被视为阻止（故障关闭）。重定向链在每一跳都会重新验证，以防止基于重定向的绕过。

#### 有意允许私有 URL

某些场景确实需要访问私有/内部 URL——将 `home.arpa` 解析到 RFC 1918 空间的家庭网络、仅限局域网的 Ollama/llama.cpp 端点、内部 wiki、云元数据调试等。对于这些情况，提供了一个全局选项：

```yaml
security:
  allow_private_urls: true   # 默认：false
```

开启后，Web 工具、浏览器、视觉 URL 获取和 gateway 媒体下载不再拒绝 RFC 1918 / 回环 / 链路本地 / CGNAT / 云元数据目标。**这是一个有意为之的信任边界**——仅在 Agent 针对本地网络执行任意 prompt 注入 URL 属于可接受风险的机器上启用。面向公众的 gateway 应保持关闭。

主机子字符串防护（即使底层 IP 是公共的，也能阻止 Unicode 同形字域名欺骗）无论此设置如何均保持开启。

### Tirith 预执行安全扫描

Hermes 集成了 [tirith](https://github.com/sheeki03/tirith) 用于在执行前进行内容级命令扫描。Tirith 能检测单纯模式匹配所遗漏的威胁：

- 同形字 URL 欺骗（国际化域名攻击）
- 管道传解释器模式（`curl | bash`、`wget | sh`）
- 终端注入攻击

Tirith 在首次使用时从 GitHub Releases 自动安装，并进行 SHA-256 校验和验证（若 cosign 可用，还会进行 cosign 来源验证）。

```yaml
# 在 ~/.hermes/config.yaml 中
security:
  tirith_enabled: true       # 启用/禁用 tirith 扫描（默认：true）
  tirith_path: "tirith"      # tirith 二进制路径（默认：PATH 查找）
  tirith_timeout: 5          # 子进程超时（秒）
  tirith_fail_open: true     # tirith 不可用时允许执行（默认：true）
```

当 `tirith_fail_open` 为 `true`（默认）时，若 tirith 未安装或超时，命令照常执行。在高安全性环境中，将其设置为 `false` 可在 tirith 不可用时阻止命令执行。

Tirith 为 Linux（x86_64 / aarch64）和 macOS（x86_64 / arm64）提供预构建二进制文件。在没有预构建二进制文件的平台（Windows 等）上，tirith 会被静默跳过——模式匹配防护仍然运行，CLI 不会显示"不可用"横幅。若要在 Windows 上使用 tirith，请在 WSL 下运行 Hermes。

Tirith 的判定与审批流程集成：安全命令直接通过，可疑和被阻止的命令会触发用户审批，并附上完整的 tirith 发现（严重性、标题、描述、更安全的替代方案）。用户可以批准或拒绝——默认选择为拒绝，以确保无人值守场景的安全。

### 上下文文件注入防护

上下文文件（AGENTS.md、.cursorrules、SOUL.md）在被纳入系统 prompt 之前会扫描 prompt 注入。扫描器检查以下内容：

- 指示忽略/无视先前指令的内容
- 含有可疑关键词的隐藏 HTML 注释
- 尝试读取密钥（`.env`、`credentials`、`.netrc`）
- 通过 `curl` 泄露凭据
- 不可见 Unicode 字符（零宽空格、双向覆盖）

被阻止的文件会显示警告：

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## 生产部署最佳实践

### Gateway 部署检查清单

1. **设置明确的允许列表** — 生产环境中切勿使用 `GATEWAY_ALLOW_ALL_USERS=true`
2. **使用容器后端** — 在 config.yaml 中设置 `terminal.backend: docker`
3. **限制资源上限** — 设置合适的 CPU、内存和磁盘限制
4. **安全存储密钥** — 将 API 密钥保存在具有适当文件权限的 `~/.hermes/.env` 中
5. **启用 DM 配对** — 尽可能使用配对码，而非硬编码用户 ID
6. **审查命令允许列表** — 定期审计 config.yaml 中的 `command_allowlist`
7. **设置 `MESSAGING_CWD`** — 不要让 Agent 在敏感目录中操作
8. **以非 root 用户运行** — 切勿以 root 身份运行 gateway
9. **监控日志** — 检查 `~/.hermes/logs/` 中的未授权访问尝试
10. **保持更新** — 定期运行 `hermes update` 以获取安全补丁

### 保护 API 密钥

```bash
# 为 .env 文件设置适当权限
chmod 600 ~/.hermes/.env

# 为不同服务使用独立密钥
# 切勿将 .env 文件提交到版本控制
```

### 网络隔离

为获得最高安全性，请在独立的机器或虚拟机上运行 gateway。在 `config.yaml` 中设置 `terminal.backend: ssh`，然后通过 `~/.hermes/.env` 中的环境变量提供主机详情：

```yaml
# ~/.hermes/config.yaml
terminal:
  backend: ssh
```

```bash
# ~/.hermes/.env
TERMINAL_SSH_HOST=agent-worker.local
TERMINAL_SSH_USER=hermes
TERMINAL_SSH_KEY=~/.ssh/hermes_agent_key
```

SSH 连接详情保存在 `.env`（而非 `config.yaml`）中，以避免随 profile 导出时被检入或共享。这样可以将 gateway 的消息连接与 Agent 的命令执行分离。

## 供应链安全公告检查

Hermes 内置了一个公告扫描器，用于标记活跃 venv 中与已知受损版本目录匹配的 Python 包（例如 2026 年 5 月的 `mistralai 2.4.6` 供应链投毒事件）。实现位于 `hermes_cli/security_advisories.py`。

运行方式：

- **CLI 启动横幅。** 若有任何公告匹配，会打印一行警告，并指向 `hermes doctor` 获取完整修复方案。
- **`hermes doctor`。** 显示所有活跃公告的版本详情和 2-4 步修复说明。
- **Gateway 启动。** 记录到 `gateway.log`；第一条交互消息会附带简短的操作者横幅。

每条公告都有一个稳定 ID。阅读并处理后，可以永久忽略它：

```bash
hermes doctor --ack <advisory-id>
```

确认信息持久化到 `config.security.acked_advisories`，重启后仍有效。旧公告**不会**从目录中删除——保留它们可以确保新安装的用户收到关于历史受损版本的警告，这些版本可能仍缓存在私有镜像中。

检查本身仅使用标准库，每条公告执行一次 `importlib.metadata.version()` 查找，因此在每次启动时运行是安全的。

### 可选依赖的懒加载安装

许多功能（Mistral TTS、ElevenLabs、Honcho 记忆、Bedrock、Slack、Matrix 等）依赖并非每个用户都需要的 Python 包。Hermes 在首次使用时**懒加载**安装这些包，而非在 `hermes-agent[all]` 下急切安装。实现位于 `tools/lazy_deps.py`。

此方案解决的权衡问题：

- **脆弱性。** 当某个额外依赖的传递依赖在 PyPI 上不可用时（因恶意软件被隔离、被撤回、上传损坏），整个 `[all]` 解析会失败，新安装会静默回退到精简版本——同时丢失 10 个以上不相关的额外功能。懒加载安装将每个后端隔离，使一个受损依赖不会破坏不相关的功能。
- **臃肿。** 只使用一个提供商的用户不再需要拉取数百个永远不会导入的包。

工作原理：

1. 后端模块在其首次导入路径的顶部调用 `ensure("feature.name")`。
2. 若依赖缺失，`ensure` 检查 `config.yaml` 中的 `security.allow_lazy_installs`（默认 `true`），并为允许列表中的规格运行 venv 作用域的 `pip install`。
3. 若安装失败或用户已禁用懒加载安装，调用会抛出 `FeatureUnavailable`，附带实际的 pip stderr 和指向 `hermes tools` 的提示。

`tools/lazy_deps.py` 强制执行的安全保证：

| 保证 | 含义 |
|---|---|
| 仅限 venv 作用域 | 安装目标为活跃 venv 中的 `sys.executable`——绝不安装到系统 Python |
| 仅按名称从 PyPI 安装 | 规格接受 `"package>=1.0,<2"` 语法。不允许 `--index-url`、`git+https://` 或 `file:` 路径——恶意的 `config.yaml` 无法重定向安装 |
| 允许列表 | 只有出现在内置 `LAZY_DEPS` 映射中的规格才能通过此路径安装。功能名称中的拼写错误**不会**获得任意安装语义 |
| 可选退出 | 设置 `security.allow_lazy_installs: false` 可完全禁用运行时安装。适用于受限网络或严格安全态势 |
| 无静默重试 | 失败以 `FeatureUnavailable` 形式呈现——不缓存错误状态，不发生重试风暴 |

禁用运行时安装：

```yaml
# ~/.hermes/config.yaml
security:
  allow_lazy_installs: false
```

禁用后，需要可选依赖的后端会提示用户手动运行安装（`pip install …`）或通过 `hermes tools` 选择其他后端。