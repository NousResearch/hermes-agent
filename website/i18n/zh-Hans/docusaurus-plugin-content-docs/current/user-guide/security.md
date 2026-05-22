---
sidebar_position: 8
title: "安全"
description: "安全模型、危险命令审批、用户授权、容器隔离以及生产部署最佳实践"
---

# 安全

Hermes Agent 采用纵深防御安全模型设计。本页涵盖所有安全边界——从命令审批到容器隔离，再到消息平台上的用户授权。

## 概述

安全模型包含七个层次：

1. **用户授权** — 谁可以与智能体对话（白名单、DM 配对）
2. **危险命令审批** — 破坏性操作的人工介入
3. **容器隔离** — Docker/Singularity/Modal 沙箱及加固设置
4. **MCP 凭证过滤** — MCP 子进程的环境变量隔离
5. **上下文文件扫描** — 项目文件中的提示注入检测
6. **跨会话隔离** — 会话之间无法访问彼此的数据或状态；定时任务存储路径经过加固，防止路径遍历攻击
7. **输入清理** — 终端工具后端中的工作目录参数经过白名单验证，防止 shell 注入

## 危险命令审批

在执行任何命令之前，Hermes 会将其与精心整理的危险模式列表进行比对。如果匹配成功，用户必须明确批准。

### 审批模式

审批系统支持三种模式，通过 `~/.hermes/config.yaml` 中的 `approvals.mode` 配置：

```yaml
approvals:
  mode: manual    # manual | smart | off
  timeout: 60     # 等待用户响应的秒数（默认：60）
```

| 模式 | 行为 |
|------|------|
| **manual**（默认） | 危险命令始终提示用户审批 |
| **smart** | 使用辅助 LLM 评估风险。低风险命令（例如 `python -c "print('hello')"`）自动批准。真正危险的命令自动拒绝。不确定的情况升级为手动提示。 |
| **off** | 禁用所有审批检查——等同于使用 `--yolo` 运行。所有命令无需提示直接执行。 |

:::warning
设置 `approvals.mode: off` 会禁用所有安全提示。仅在受信任的环境中使用（CI/CD、容器等）。
:::

### YOLO 模式

YOLO 模式绕过当前会话的**所有**危险命令审批提示。有三种激活方式：

1. **CLI 标志**：使用 `hermes --yolo` 或 `hermes chat --yolo` 启动会话
2. **斜杠命令**：在会话中输入 `/yolo` 切换开关
3. **环境变量**：设置 `HERMES_YOLO_MODE=1`

`/yolo` 命令是一个**切换**——每次使用都会翻转模式：

```
> /yolo
  ⚡ YOLO 模式已开启 — 所有命令自动批准。请谨慎使用。

> /yolo
  ⚠ YOLO 模式已关闭 — 危险命令将需要审批。
```

YOLO 模式在 CLI 和网关会话中均可用。在内部，它会设置 `HERMES_YOLO_MODE` 环境变量，该变量在每次命令执行前都会被检查。

:::danger
YOLO 模式禁用会话的**所有**危险命令安全检查——**除了**硬线黑名单（见下文）。仅在您完全信任所生成命令时使用（例如，在可丢弃环境中运行经过充分测试的自动化脚本）。
:::

### 硬线黑名单（始终生效的底线）

某些命令极具破坏性——不可逆的文件系统擦除、fork 炸弹、直接块设备写入——以至于 Hermes **无论**以下情况如何都拒绝运行：

- `--yolo` / `/yolo` 已开启
- `approvals.mode: off`
- 定时任务在无人值守的 `approve` 模式下运行
- 用户明确点击"始终允许"

黑名单是 `--yolo` 之下的底线。它在审批层看到命令之前就已触发，并且没有覆盖标志。当前覆盖的模式（非穷尽；与 `tools/approval.py::UNRECOVERABLE_BLOCKLIST` 保持同步）：

| 模式 | 为何是硬线 |
|---|---|
| `rm -rf /` 及其明显变体 | 擦除文件系统根目录 |
| `rm -rf --no-preserve-root /` | 显式的"我就是要删根目录"变体 |
| `:(){ :\|:& };:`（bash fork 炸弹） | 让主机卡死直到重启 |
| `mkfs.*` 在已挂载的根设备上 | 格式化正在运行的系统 |
| `dd if=/dev/zero of=/dev/sd*` | 清零物理磁盘 |
| 将不受信任的 URL 通过管道传给根文件系统顶层的 `sh` | 远程代码执行攻击面太广，无法审批 |

如果触发了黑名单，工具调用会向智能体返回解释性错误，不会执行任何操作。如果合法工作流需要这些命令之一（例如，您是擦除重装流水线的运维人员），请在智能体外运行。

### 审批超时

当出现危险命令提示时，用户有 configurable 的时间来响应。如果在超时时间内没有响应，命令默认被拒绝（故障关闭）。

在 `~/.hermes/config.yaml` 中配置超时：

```yaml
approvals:
  timeout: 60  # 秒（默认：60）
```

### 触发审批的内容

以下模式会触发审批提示（定义在 `tools/approval.py` 中）：

| 模式 | 描述 |
|---------|-------------|
| `rm -r` / `rm --recursive` | 递归删除 |
| `rm ... /` | 在根路径删除 |
| `chmod 777/666` / `o+w` / `a+w` | 全局/其他用户可写权限 |
| `chmod --recursive` 配合不安全权限 | 递归全局/其他用户可写（长标志） |
| `chown -R root` / `chown --recursive root` | 递归 chown 到 root |
| `mkfs` | 格式化文件系统 |
| `dd if=` | 磁盘复制 |
| `> /dev/sd` | 写入块设备 |
| `DROP TABLE/DATABASE` | SQL DROP |
| `DELETE FROM`（不带 WHERE） | 不带 WHERE 的 SQL DELETE |
| `TRUNCATE TABLE` | SQL TRUNCATE |
| `> /etc/` | 覆盖系统配置 |
| `systemctl stop/restart/disable/mask` | 停止/重启/禁用系统服务 |
| `kill -9 -1` | 杀死所有进程 |
| `pkill -9` | 强制杀死进程 |
| Fork 炸弹模式 | Fork 炸弹 |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | 通过 `-c` 标志执行 shell 命令（包括组合标志如 `-lc`） |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | 通过 `-e`/`-c` 标志执行脚本 |
| `curl ... \| sh` / `wget ... \| sh` | 将远程内容管道到 shell |
| `bash <(curl ...)` / `sh <(wget ...)` | 通过进程替换执行远程脚本 |
| `tee` 到 `/etc/`、`~/.ssh/`、`~/.hermes/.env` | 通过 tee 覆盖敏感文件 |
| `>` / `>>` 到 `/etc/`、`~/.ssh/`、`~/.hermes/.env` | 通过重定向覆盖敏感文件 |
| `xargs rm` | 配合 rm 的 xargs |
| `find -exec rm` / `find -delete` | 配合破坏性操作的 find |
| `cp`/`mv`/`install` 到 `/etc/` | 复制/移动文件到系统配置目录 |
| `sed -i` / `sed --in-place` 在 `/etc/` 上 | 系统配置的就地编辑 |
| `pkill`/`killall` hermes/gateway | 防止自终止 |
| `gateway run` 配合 `&`/`disown`/`nohup`/`setsid` | 防止在服务管理器外启动网关 |

:::info
**容器绕过**：当使用 `docker`、`singularity`、`modal`、`daytona` 或 `vercel_sandbox` 后端运行时，危险命令检查会被**跳过**，因为容器本身就是安全边界。容器内的破坏性命令无法损害宿主机。
:::

### 审批流程（CLI）

在交互式 CLI 中，危险命令会显示内联审批提示：

```
  ⚠️  危险命令：递归删除
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

四个选项：

- **once** — 允许本次执行
- **session** — 允许该模式在本次会话剩余时间内执行
- **always** — 添加到永久白名单（保存到 `config.yaml`）
- **deny**（默认）— 阻止命令

### 审批流程（网关/消息平台）

在消息平台上，智能体会将危险命令详情发送到聊天并等待用户回复：

- 回复 **yes**、**y**、**approve**、**ok** 或 **go** 以批准
- 回复 **no**、**n**、**deny** 或 **cancel** 以拒绝

运行网关时会自动设置 `HERMES_EXEC_ASK=1` 环境变量。

### 永久白名单

使用 "always" 批准的命令会保存到 `~/.hermes/config.yaml`：

```yaml
# 永久允许的危险命令模式
command_allowlist:
  - rm
  - systemctl
```

这些模式在启动时加载，并在所有未来会话中静默批准。

:::tip
使用 `hermes config edit` 查看或移除永久白名单中的模式。
:::

## 用户授权（网关）

运行消息网关时，Hermes 通过分层授权系统控制谁可以与机器人交互。

### 授权检查顺序

`_is_user_authorized()` 方法按以下顺序检查：

1. **按平台允许所有标志**（例如，`DISCORD_ALLOW_ALL_USERS=true`）
2. **DM 配对批准列表**（通过配对码批准的用户）
3. **平台特定白名单**（例如，`TELEGRAM_ALLOWED_USERS=12345,67890`）
4. **全局白名单**（`GATEWAY_ALLOWED_USERS=12345,67890`）
5. **全局允许所有**（`GATEWAY_ALLOW_ALL_USERS=true`）
6. **默认：拒绝**

### 平台白名单

在 `~/.hermes/.env` 中将允许的用户 ID 设置为逗号分隔值：

```bash
# 平台特定白名单
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# 跨平台白名单（对所有平台生效）
GATEWAY_ALLOWED_USERS=123456789

# 按平台允许所有（谨慎使用）
DISCORD_ALLOW_ALL_USERS=true

# 全局允许所有（极度谨慎使用）
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
如果**未配置任何白名单**且未设置 `GATEWAY_ALLOW_ALL_USERS`，**所有用户都会被拒绝**。网关启动时会记录警告：

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g. TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DM 配对系统

为了更灵活的授权，Hermes 包含基于验证码的配对系统。无需预先提供用户 ID，未知用户会收到一次性配对码，由机器人所有者在 CLI 上批准。

**工作原理：**

1. 未知用户向机器人发送 DM
2. 机器人回复一个 8 位字符的配对码
3. 机器人所有者在 CLI 上运行 `hermes pairing approve <platform> <code>`
4. 该用户被永久批准用于该平台

在 `~/.hermes/config.yaml` 中控制未授权私信的处理方式：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` 是默认设置。未授权 DM 会收到配对码回复。
- `ignore` 静默丢弃未授权 DM。
- 平台配置覆盖全局默认值，因此您可以在 Telegram 上保持配对，同时让 WhatsApp 保持静默。

**安全特性**（基于 OWASP + NIST SP 800-63-4 指南）：

| 特性 | 详情 |
|---------|---------|
| 验证码格式 | 8 位字符，来自 32 位无歧义字母表（不含 0/O/1/I） |
| 随机性 | 加密安全（`secrets.choice()`） |
| 验证码有效期 | 1 小时过期 |
| 速率限制 | 每用户每 10 分钟 1 次请求 |
| 待处理上限 | 每平台最多 3 个待处理验证码 |
| 锁定 | 5 次批准失败尝试 → 1 小时锁定 |
| 文件安全 | 所有配对数据文件设置 `chmod 0600` |
| 日志 | 验证码永远不会记录到 stdout |

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

**存储：** 配对数据存储在 `~/.hermes/pairing/` 中，包含按平台划分的 JSON 文件：
- `{platform}-pending.json` — 待处理配对请求
- `{platform}-approved.json` — 已批准用户
- `_rate_limits.json` — 速率限制和锁定跟踪

## 容器隔离

使用 `docker` 终端后端时，Hermes 对每个容器应用严格的安全加固。

### Docker 安全标志

每个容器都使用以下标志运行（定义在 `tools/environments/docker.py` 中）：

```python
_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # 丢弃所有 Linux 能力
    "--cap-add", "DAC_OVERRIDE",                  # root 可以写入绑定挂载的目录
    "--cap-add", "CHOWN",                         # 包管理器需要文件所有权
    "--cap-add", "FOWNER",                        # 包管理器需要文件所有权
    "--security-opt", "no-new-privileges",         # 阻止权限提升
    "--pids-limit", "256",                         # 限制进程数量
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # 大小受限的 /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # 不可执行的 /var/tmp
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",   # 不可执行的 /run
]
```

### 资源限制

容器资源可在 `~/.hermes/config.yaml` 中配置：

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # 仅显式白名单；空值可防止机密进入容器
  container_cpu: 1        # CPU 核心数
  container_memory: 5120  # MB（默认 5GB）
  container_disk: 51200   # MB（默认 50GB，需要 overlay2 on XFS）
  container_persistent: true  # 跨会话持久化文件系统
```

### 文件系统持久化

- **持久模式**（`container_persistent: true`）：从 `~/.hermes/sandboxes/docker/<task_id>/` 绑定挂载 `/workspace` 和 `/root`
- **临时模式**（`container_persistent: false`）：使用 tmpfs 作为工作区——清理后所有内容丢失

:::tip
对于生产网关部署，使用 `docker`、`modal`、`daytona` 或 `vercel_sandbox` 后端来隔离智能体命令与宿主机系统。这完全消除了危险命令审批的需要。
:::

:::warning
如果您在 `terminal.docker_forward_env` 中添加名称，这些变量会被有意注入容器以用于终端命令。这对任务特定凭证（如 `GITHUB_TOKEN`）很有用，但也意味着在容器中运行的代码可以读取并外泄它们。
:::

## 终端后端安全对比

| 后端 | 隔离 | 危险命令检查 | 最适合 |
|---------|-----------|-------------------|----------|
| **local** | 无——在宿主机上运行 | ✅ 是 | 开发、受信任用户 |
| **ssh** | 远程机器 | ✅ 是 | 在独立服务器上运行 |
| **docker** | 容器 | ❌ 跳过（容器即边界） | 生产网关 |
| **singularity** | 容器 | ❌ 跳过 | HPC 环境 |
| **modal** | 云沙箱 | ❌ 跳过 | 可扩展的云隔离 |
| **daytona** | 云沙箱 | ❌ 跳过 | 持久化云工作区 |
| **vercel_sandbox** | 云 microVM | ❌ 跳过 | 带快照持久化的云执行 |

## 环境变量透传 {#environment-variable-passthrough}

`execute_code` 和 `terminal` 都会从子进程中剥离敏感环境变量，以防止 LLM 生成的代码外泄凭证。然而，声明了 `required_environment_variables` 的技能确实需要访问这些变量。

### 工作原理

两种机制允许特定变量通过沙箱过滤器：

**1. 技能作用域透传（自动）**

当技能通过 `skill_view` 或 `/skill` 命令加载并声明 `required_environment_variables` 时，环境中实际设置的任何这些变量都会自动注册为透传。缺失的变量（仍处于待设置状态）**不会**被注册。

```yaml
# 在技能的 SKILL.md frontmatter 中
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
```

加载此技能后，`TENOR_API_KEY` 会透传到 `execute_code`、`terminal`（本地）以及**远程后端（Docker、Modal）**——无需手动配置。

:::info Docker 与 Modal
在 v0.5.1 之前，Docker 的 `forward_env` 是与技能透传分开的系统。现在它们已合并——技能声明的环境变量会自动转发到 Docker 容器和 Modal 沙箱，无需手动添加到 `docker_forward_env`。
:::

**2. 基于配置的透传（手动）**

对于未由任何技能声明的环境变量，将它们添加到 `config.yaml` 中的 `terminal.env_passthrough`：

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

### 凭证文件透传（OAuth token 等）{#credential-file-passthrough}

某些技能需要在沙箱中使用**文件**（而不仅仅是环境变量）——例如，Google Workspace 将 OAuth token 存储为活动配置文件的 `HERMES_HOME` 下的 `google_token.json`。技能在 frontmatter 中声明这些：

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token（由安装脚本创建）
  - path: google_client_secret.json
    description: Google OAuth2 客户端凭证
```

加载时，Hermes 会检查这些文件是否存在于活动配置文件的 `HERMES_HOME` 中，并将其注册为挂载：

- **Docker**：只读绑定挂载（`-v host:container:ro`）
- **Modal**：在沙箱创建时挂载 + 每次命令前同步（处理会话中期的 OAuth 设置）
- **本地**：无需操作（文件已可访问）

您也可以在 `config.yaml` 中手动列出凭证文件：

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

路径相对于 `~/.hermes/`。文件在容器内挂载到 `/root/.hermes/`。

### 每个沙箱过滤的内容

| 沙箱 | 默认过滤器 | 透传覆盖 |
|---------|---------------|---------------------|
| **execute_code** | 阻止名称包含 `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL`、`PASSWD`、`AUTH` 的变量；仅允许安全前缀变量通过 | ✅ 透传变量绕过两项检查 |
| **terminal**（本地） | 阻止显式 Hermes 基础设施变量（提供商密钥、网关 token、工具 API 密钥） | ✅ 透传变量绕过黑名单 |
| **terminal**（Docker） | 默认不传递宿主机环境变量 | ✅ 透传变量 + `docker_forward_env` 通过 `-e` 转发 |
| **terminal**（Modal） | 默认不传递宿主机环境变量/文件 | ✅ 凭证文件已挂载；环境透传通过同步 |
| **MCP** | 阻止除安全系统变量 + 显式配置的 `env` 之外的所有内容 | ❌ 不受透传影响（改用 MCP `env` 配置） |

### 安全注意事项

- 透传仅影响您或您的技能显式声明的变量——任意 LLM 生成代码的默认安全态势不变
- 凭证文件以**只读**方式挂载到 Docker 容器
- Skills Guard 在安装前扫描技能内容中的可疑环境访问模式
- 缺失/未设置的变量永远不会被注册（您无法泄漏不存在的东西）
- Hermes 基础设施机密（提供商 API 密钥、网关 token）不应添加到 `env_passthrough`——它们有专用机制

## MCP 凭证处理

MCP（Model Context Protocol）服务器子进程接收**过滤后的环境**，以防止意外凭证泄漏。

### 安全环境变量

只有这些变量从宿主机传递到 MCP stdio 子进程：

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

以及任何 `XDG_*` 变量。所有其他环境变量（API 密钥、token、机密）都被**剥离**。

MCP 服务器的 `env` 配置中显式定义的变量会被传递：

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # 仅传递此变量
```

### 凭证脱敏

MCP 工具的错误消息在返回给 LLM 之前会被清理。以下模式会被替换为 `[REDACTED]`：

- GitHub PAT（`ghp_...`）
- OpenAI 风格密钥（`sk-...`）
- Bearer token
- `token=`、`key=`、`API_KEY=`、`password=`、`secret=` 参数

### 网站访问策略

您可以限制智能体可以通过其 web 和 browser 工具访问哪些网站。这对于防止智能体访问内部服务、管理面板或其他敏感 URL 很有用。

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

当请求被阻止的 URL 时，工具会返回错误，说明该域名已被策略阻止。黑名单在 `web_search`、`web_extract`、`browser_navigate` 和所有支持 URL 的工具上强制执行。

有关完整详情，请参阅配置指南中的[网站黑名单](/user-guide/configuration#website-blocklist)。

### SSRF 防护

所有支持 URL 的工具（web search、web extract、vision、browser）在获取前会验证 URL，以防止服务器端请求伪造（SSRF）攻击。阻止的地址包括：

- **私有网络**（RFC 1918）：`10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`
- **回环地址**：`127.0.0.0/8`、`::1`
- **链路本地**：`169.254.0.0/16`（包括云元数据 `169.254.169.254`）
- **CGNAT / 共享地址空间**（RFC 6598）：`100.64.0.0/10`（Tailscale、WireGuard VPN）
- **云元数据主机名**：`metadata.google.internal`、`metadata.goog`
- **保留、组播和未指定地址**

SSRF 防护在互联网面向使用中始终处于活动状态，DNS 失败被视为已阻止（故障关闭）。重定向链在每一跳都会重新验证，以防止基于重定向的绕过。

#### 有意允许私有 URL

某些设置确实需要私有/内部 URL 访问——解析到 RFC 1918 空间的 `home.arpa` 家庭网络、仅限局域网的 Ollama/llama.cpp 端点、内部 wiki、云元数据调试等。对于这些情况，有一个全局退出选项：

```yaml
security:
  allow_private_urls: true   # 默认：false
```

开启后，web 工具、browser、vision URL 获取和网关媒体下载不再拒绝 RFC 1918 / 回环 / 链路本地 / CGNAT / 云元数据目标。**这是一个刻意的信任边界**——仅在智能体对本地网络运行任意提示注入 URL 是可接受风险的机器上启用。面向公众的网关应保持关闭。

主机子字符串守卫（即使底层 IP 是公共的也会阻止相似的 Unicode 域名技巧）无论此设置如何都保持开启。

### Tirith 执行前安全扫描

Hermes 集成 [tirith](https://github.com/sheeki03/tirith) 用于执行前的内容级命令扫描。Tirith 检测仅靠模式匹配无法发现的威胁：

- 同形异义 URL 欺骗（国际化域名攻击）
- 管道到解释器模式（`curl | bash`、`wget | sh`）
- 终端注入攻击

Tirith 在首次使用时从 GitHub releases 自动安装，并进行 SHA-256 校验和验证（如果 cosign 可用，还会进行 cosign 来源验证）。

```yaml
# 在 ~/.hermes/config.yaml 中
security:
  tirith_enabled: true       # 启用/禁用 tirith 扫描（默认：true）
  tirith_path: "tirith"      # tirith 二进制文件路径（默认：PATH 查找）
  tirith_timeout: 5          # 子进程超时秒数
  tirith_fail_open: true     # tirith 不可用时允许执行（默认：true）
```

当 `tirith_fail_open` 为 `true`（默认）时，如果 tirith 未安装或超时，命令会继续执行。在高安全环境中设置为 `false`，以在 tirith 不可用时阻止命令。

Tirith 的裁决与审批流程集成：安全命令直接通过，而可疑和被阻止的命令都会触发用户审批，并显示完整的 tirith 发现结果（严重性、标题、描述、更安全的替代方案）。用户可以批准或拒绝——默认选择是拒绝，以保持无人值守场景的安全。

### 上下文文件注入防护

上下文文件（AGENTS.md、.cursorrules、SOUL.md）在包含到系统提示之前会扫描提示注入。扫描器检查：

- 忽略/无视先前指令的指令
- 带有可疑关键字的隐藏 HTML 注释
- 尝试读取机密（`.env`、`credentials`、`.netrc`）
- 通过 `curl` 外泄凭证
- 不可见的 Unicode 字符（零宽空格、双向覆盖）

被阻止的文件会显示警告：

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## 生产部署最佳实践

### 网关部署检查清单

1. **设置显式白名单** — 切勿在生产环境中使用 `GATEWAY_ALLOW_ALL_USERS=true`
2. **使用容器后端** — 在 config.yaml 中设置 `terminal.backend: docker`
3. **限制资源** — 设置适当的 CPU、内存和磁盘限制
4. **安全存储机密** — 将 API 密钥保存在 `~/.hermes/.env` 中，并设置适当的文件权限
5. **启用 DM 配对** — 尽可能使用配对码而非硬编码用户 ID
6. **审查命令白名单** — 定期审计 config.yaml 中的 `command_allowlist`
7. **设置 `MESSAGING_CWD`** — 不要让智能体在敏感目录中操作
8. **以非 root 运行** — 切勿以 root 运行网关
9. **监控日志** — 检查 `~/.hermes/logs/` 中的未授权访问尝试
10. **保持更新** — 定期运行 `hermes update` 获取安全补丁

### 保护 API 密钥

```bash
# 为 .env 文件设置适当权限
chmod 600 ~/.hermes/.env

# 为不同服务使用单独的密钥
# 切勿将 .env 文件提交到版本控制
```

### 网络隔离

为了最大安全性，在单独的机器或 VM 上运行网关。在 `config.yaml` 中设置 `terminal.backend: ssh`，然后通过 `~/.hermes/.env` 中的环境变量提供主机详情：

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

SSH 连接详情保存在 `.env` 中（而非 `config.yaml`），因此它们不会被检入或随配置文件导出共享。这会将网关的消息连接与智能体的命令执行分开。
