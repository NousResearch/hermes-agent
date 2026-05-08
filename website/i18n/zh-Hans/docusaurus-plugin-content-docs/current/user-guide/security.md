---
sidebar_position: 8
title: "安全"
description: "安全模型、危险命令审批、用户授权、容器隔离和生产部署最佳实践"
---

# 安全

Hermes Agent 采用纵深防御安全模型设计。本页面涵盖每个安全边界——从命令审批到容器隔离再到消息平台上的用户授权。

## 概述

安全模型有七层：

1. **用户授权** — 谁可以与 agent 交谈（允许列表、DM 配对）
2. **危险命令审批** — 破坏性操作的人类参与循环
3. **容器隔离** — Docker/Singularity/Modal 沙箱与强化设置
4. **MCP 凭证过滤** — MCP 子进程的的环境变量隔离
5. **上下文文件扫描** — 项目文件中的提示注入检测
6. **跨会话隔离** — 会话无法访问彼此的数据或状态；cron 作业存储路径经过强化以防止路径遍历攻击
7. **输入清理** — 终端工具后端中的工作目录参数根据允许列表进行验证，以防止 shell 注入

## 危险命令审批

在执行任何命令之前，Hermes 会根据危险模式列表进行检查。如果找到匹配项，用户必须明确批准。

### 审批模式

审批系统支持三种模式，通过 `~/.hermes/config.yaml` 中的 `approvals.mode` 配置：

```yaml
approvals:
  mode: manual    # manual | smart | off
  timeout: 60     # 等待用户响应的秒数（默认：60）
```

| 模式 | 行为 |
|------|----------|
| **manual**（默认） | 始终提示用户批准危险命令 |
| **smart** | 使用辅助 LLM 评估风险。低风险命令（例如 `python -c "print('hello')"`）自动批准。真正危险的命令自动拒绝。不确定的情况升级到手动提示。 |
| **off** | 禁用所有审批检查——等同于使用 `--yolo`。所有命令执行时不会提示。 |

:::warning
设置 `approvals.mode: off` 会禁用所有安全提示。仅在受信任的环境中使用（CI/CD、容器等）。
:::

### YOLO 模式

YOLO 模式绕过**所有**危险命令审批提示为当前会话。可以通过三种方式激活：

1. **CLI 标志**：使用 `hermes --yolo` 或 `hermes chat --yolo` 启动会话
2. **斜杠命令**：在会话期间输入 `/yolo` 以切换开/关
3. **环境变量**：设置 `HERMES_YOLO_MODE=1`

`/yolo` 命令是一个**切换**——每次使用都会翻转模式：

```
> /yolo
  ⚡ YOLO mode ON — all commands auto-approved. Use with caution.

> /yolo
  ⚠ YOLO mode OFF — dangerous commands will require approval.
```

YOLO 模式在 CLI 和 gateway 会话中都可用。在内部，它设置 `HERMES_YOLO_MODE` 环境变量，在每次命令执行之前检查。

:::danger
YOLO 模式为会话禁用**所有**危险命令安全检查——**除了**强硬黑名单（见下文）。仅在您完全信任生成的命令时使用（例如，在可丢弃环境中的经过良好测试的自动化脚本）。
:::

### 强硬黑名单（始终开启的底线）

有些命令如此灾难性——不可逆的文件系统擦除、fork 炸弹、直接块设备写入——以至于 Hermes **无论**以下情况都会拒绝运行它们：

- `--yolo` / `/yolo` 切换开启
- `approvals.mode: off`
- 在 headless `approve` 模式下运行的 cron 作业
- 用户明确点击"始终允许"

黑名单是 `--yolo` 之下的底线。它在审批层甚至看到命令之前触发，并且没有覆盖标志。当前覆盖的模式（并非详尽无遗；与 `tools/approval.py::UNRECOVERABLE_BLOCKLIST` 保持同步）：

| 模式 | 原因 |
|---|---|
| `rm -rf /` 及其明显变体 | 擦除文件系统根目录 |
| `rm -rf --no-preserve-root /` | 明确的"是的，我的意思是 root"变体 |
| `:(){ :\|:& };:` (bash fork bomb) | 将主机钉住直到重启 |
| `mkfs.*` 在挂载的根设备上 | 格式化实时系统 |
| `dd if=/dev/zero of=/dev/sd*` | 归零物理磁盘 |
| 在 rootfs 顶层将不受信任的 URL 管道传输到 `sh` | 远程代码执行攻击向量太宽泛无法批准 |

如果您触发了黑名单，工具调用会向 agent 返回解释性错误，什么都不会运行。如果合法工作流程需要其中一个命令（例如，您是 wipe-and-reinstall 管道的操作员），请在 agent 外部运行它。

### 审批超时

当危险命令提示出现时，用户有可配置的时间来响应。如果在超时内没有响应，默认**拒绝**命令（fail-closed）。

在 `~/.hermes/config.yaml` 中配置超时：

```yaml
approvals:
  timeout: 60  # 秒（默认：60）
```

### 什么会触发审批

以下模式触发审批提示（定义在 `tools/approval.py`）：

| 模式 | 描述 |
|---------|-------------|
| `rm -r` / `rm --recursive` | 递归删除 |
| `rm ... /` | 在根路径中删除 |
| `chmod 777/666` / `o+w` / `a+w` | 世界/其他可写权限 |
| `chmod --recursive` 使用不安全的权限 | 递归世界/其他可写（长标志） |
| `chown -R root` / `chown --recursive root` | 递归 chown 到 root |
| `mkfs` | 格式化文件系统 |
| `dd if=` | 磁盘复制 |
| `> /dev/sd` | 写入块设备 |
| `DROP TABLE/DATABASE` | SQL DROP |
| `DELETE FROM`（无 WHERE） | 无 WHERE 的 SQL DELETE |
| `TRUNCATE TABLE` | SQL TRUNCATE |
| `> /etc/` | 覆盖系统配置 |
| `systemctl stop/restart/disable/mask` | 停止/重启/禁用系统服务 |
| `kill -9 -1` | 终止所有进程 |
| `pkill -9` | 强制终止进程 |
| Fork bomb 模式 | Fork 炸弹 |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | 通过 `-c` 标志执行 shell 命令（包括组合标志如 `-lc`） |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | 通过 `-e`/`-c` 标志执行脚本 |
| `curl ... \| sh` / `wget ... \| sh` | 将远程内容管道传输到 shell |
| `bash <(curl ...)` / `sh <(wget ...)` | 通过进程替换执行远程脚本 |
| `tee` 到 `/etc/`、`~/.ssh/`、`~/.hermes/.env` | 通过 tee 覆盖敏感文件 |
| `>` / `>>` 到 `/etc/`、`~/.ssh/`、`~/.hermes/.env` | 通过重定向覆盖敏感文件 |
| `xargs rm` | 带 rm 的 xargs |
| `find -exec rm` / `find -delete` | 带破坏性操作的 find |
| `cp`/`mv`/`install` 到 `/etc/` | 复制/移动文件到系统配置 |
| `sed -i` / `sed --in-place` 到 `/etc/` | 系统配置的原地编辑 |
| `pkill`/`killall` hermes/gateway | 自我终止防护 |
| `gateway run` 带 `&`/`disown`/`nohup`/`setsid` | 防止在服务管理器外部启动 gateway |

:::info
**容器绕过**：当在 `docker`、`singularity`、`modal`、`daytona` 或 `vercel_sandbox` 后端中运行时，危险命令检查**被跳过**，因为容器本身就是安全边界。容器内的破坏性命令无法伤害主机。
:::

### 审批流程（CLI）

在交互式 CLI 中，危险命令显示内联审批提示：

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

四个选项：

- **once** — 允许这次执行
- **session** — 在会话剩余时间内允许此模式
- **always** — 添加到永久允许列表（保存到 `config.yaml`）
- **deny**（默认）— 阻止命令

### 审批流程（Gateway/消息传递）

在消息平台上，agent 会将危险命令详情发送到聊天并等待用户回复：

- 回复 **yes**、**y**、**approve**、**ok** 或 **go** 以批准
- 回复 **no**、**n**、**deny** 或 **cancel** 以拒绝

`HERMES_EXEC_ASK=1` 环境变量在运行 gateway 时自动设置。

### 永久允许列表

使用"always"批准的命令保存到 `~/.hermes/config.yaml`：

```yaml
# 永久允许的危险命令模式
command_allowlist:
  - rm
  - systemctl
```

这些模式在启动时加载，并在所有未来会话中静默批准。

:::tip
使用 `hermes config edit` 查看或从永久允许列表中删除模式。
:::

## 用户授权（Gateway）

运行消息传递 gateway 时，Hermes 通过分层授权系统控制谁可以与 bot 交互。

### 授权检查顺序

`_is_user_authorized()` 方法按此顺序检查：

1. **每个平台的允许所有人标志**（例如 `DISCORD_ALLOW_ALL_USERS=true`）
2. **DM 配对批准列表**（通过配对代码批准的用户）
3. **平台特定的允许列表**（例如 `TELEGRAM_ALLOWED_USERS=12345,67890`）
4. **全局允许列表**（`GATEWAY_ALLOWED_USERS=12345,67890`）
5. **全局允许所有人**（`GATEWAY_ALLOW_ALL_USERS=true`）
6. **默认：拒绝**

### 平台允许列表

在 `~/.hermes/.env` 中将允许的用户 ID 设置为逗号分隔的值：

```bash
# 平台特定的允许列表
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# 跨平台允许列表（为所有平台检查）
GATEWAY_ALLOWED_USERS=123456789

# 每个平台的允许所有人（谨慎使用）
DISCORD_ALLOW_ALL_USERS=true

# 全局允许所有人（极其谨慎使用）
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
如果**没有配置允许列表**且未设置 `GATEWAY_ALLOW_ALL_USERS`，**所有用户都被拒绝**。Gateway 在启动时记录警告：

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DM 配对系统

为了更灵活的授权，Hermes 包含基于代码的配对系统。无需预先要求用户 ID，未知用户会收到一次性配对代码，bot 所有者通过 CLI 批准。

**工作原理：**

1. 未知用户向 bot 发送 DM
2. Bot 回复一个 8 字符配对代码
3. Bot 所有者运行 `hermes pairing approve <platform> <code>`
4. 用户永久批准该平台

在 `~/.hermes/config.yaml` 中控制未授权 DM 的处理方式：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` 是默认值。未授权的 DM 收到配对代码回复。
- `ignore` 静默丢弃未授权的 DM。
- 平台部分覆盖全局默认值，因此您可以在 Telegram 上保持配对而在 WhatsApp 上保持沉默。

**安全功能**（基于 OWASP + NIST SP 800-63-4 指导）：

| 功能 | 详情 |
|---------|---------|
| 代码格式 | 来自 32 字符明确字母表的 8 个字符（无 0/O/1/I） |
| 随机性 | 加密的（`secrets.choice()`） |
| 代码 TTL | 1 小时过期 |
| 速率限制 | 每个用户每 10 分钟 1 次请求 |
| 待处理限制 | 每个平台最多 3 个待处理代码 |
| 锁定 | 5 次失败的批准尝试 → 1 小时锁定 |
| 文件安全 | 所有配对数据文件 `chmod 0600` |
| 日志记录 | 代码从不记录到 stdout |

**配对 CLI 命令：**

```bash
# 列出待处理和批准的用户
hermes pairing list

# 批准配对代码
hermes pairing approve telegram ABC12DEF

# 撤销用户的访问权限
hermes pairing revoke telegram 123456789

# 清除所有待处理代码
hermes pairing clear-pending
```

**存储：** 配对数据存储在 `~/.hermes/pairing/` 中，带有每个平台的 JSON 文件：
- `{platform}-pending.json` — 待处理配对请求
- `{platform}-approved.json` — 批准的用户
- `_rate_limits.json` — 速率限制和锁定跟踪

## 容器隔离

当使用 `docker` 终端后端时，Hermes 对每个容器应用严格的安全强化。

### Docker 安全标志

每个容器都使用这些标志运行（定义在 `tools/environments/docker.py`）：

```python
_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # 丢弃所有 Linux 功能
    "--cap-add", "DAC_OVERRIDE",                  # Root 可以写入绑定挂载的目录
    "--cap-add", "CHOWN",                         # 包管理器需要文件所有权
    "--cap-add", "FOWNER",                        # 包管理器需要文件所有权
    "--security-opt", "no-new-privileges",         # 阻止权限提升
    "--pids-limit", "256",                         # 限制进程数
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # 大小限制的 /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # No-exec /var/tmp
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",   # No-exec /run
]
```

### 资源限制

容器资源可在 `~/.hermes/config.yaml` 中配置：

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # 仅显式允许列表；空值将密钥排除在容器外
  container_cpu: 1        # CPU 核
  container_memory: 5120  # MB（默认 5GB）
  container_disk: 51200   # MB（默认 50GB，需要 XFS 上的 overlay2）
  container_persistent: true  # 跨会话保持文件系统
```

### 文件系统持久性

- **持久模式**（`container_persistent: true`）：从 `~/.hermes/sandboxes/docker/<task_id>/` 绑定挂载 `/workspace` 和 `/root`
- **临时模式**（`container_persistent: false`）：为工作区使用 tmpfs——清理时一切都会丢失

:::tip
对于生产 gateway 部署，使用 `docker`、`modal`、`daytona` 或 `vercel_sandbox` 后端将 agent 命令与主机系统隔离。这完全消除了对危险命令审批的需要。
:::

:::warning
如果您将名称添加到 `terminal.docker_forward_env`，那些变量会被故意注入容器以供终端命令使用。这对于特定于任务的凭证（如 `GITHUB_TOKEN`）很有用，但也意味着在容器中运行的代码可以读取和窃取它们。
:::

## 终端后端安全比较

| 后端 | 隔离 | 危险命令检查 | 适用于 |
|---------|-----------|-------------------|----------|
| **local** | 无——在主机上运行 | ✅ 是 | 开发、受信任用户 |
| **ssh** | 远程机器 | ✅ 是 | 在单独的服务器上运行 |
| **docker** | 容器 | ❌ 跳过（容器是边界） | 生产 gateway |
| **singularity** | 容器 | ❌ 跳过 | HPC 环境 |
| **modal** | 云沙箱 | ❌ 跳过 | 可扩展云隔离 |
| **daytona** | 云沙箱 | ❌ 跳过 | 持久云工作区 |
| **vercel_sandbox** | 云 microVM | ❌ 跳过 | 带快照持久性的云执行 |

## 环境变量穿透 {#environment-variable-passthrough}

`execute_code` 和 `terminal` 都会从子进程中剥离敏感环境变量，以防止 LLM 生成代码的凭证泄露。但是，声明 `required_environment_variables` 的 skills 合法需要访问这些变量。

### 工作原理

两种机制允许特定变量通过沙箱过滤器：

**1. Skill 作用域穿透（自动）**

当通过 `skill_view` 或 `/skill` 命令加载 skill 并声明 `required_environment_variables` 时，这些变量中任何实际在环境中设置的变量都会自动注册为穿透。未设置的变量（仍在 setup-needed 状态）**不会**被注册。

```yaml
# 在 skill 的 SKILL.md frontmatter 中
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
```

加载此 skill 后，`TENOR_API_KEY` 穿透到 `execute_code`、`terminal`（本地）**以及远程后端（Docker、Modal）**——无需手动配置。

:::info Docker 和 Modal
在 v0.5.1 之前，Docker 的 `forward_env` 是与 skill 穿透分离的系统。它们现在已合并——skill 声明的 env 变量会自动转发到 Docker 容器和 Modal 沙箱，无需手动将它们添加到 `docker_forward_env`。
:::

**2. 基于配置的穿透（手动）**

对于任何 skill 未声明的 env 变量，将它们添加到 `config.yaml` 中的 `terminal.env_passthrough`：

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

### 凭证文件穿透（OAuth 令牌等）{#credential-file-passthrough}

某些 skills 需要沙箱中的**文件**（而不仅仅是 env vars）——例如，Google Workspace 将 OAuth 令牌存储为活动 profile `HERMES_HOME` 下的 `google_token.json`。Skills 在 frontmatter 中声明这些：

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

加载时，Hermes 检查这些文件是否存在于活动 profile 的 `HERMES_HOME` 中，并注册它们以进行挂载：

- **Docker**：只读绑定挂载（`-v host:container:ro`）
- **Modal**：在沙箱创建时挂载 + 在每个命令之前同步（处理会话中 OAuth 设置）
- **Local**：无需操作（文件已经可访问）

您也可以在 `config.yaml` 中手动列出凭证文件：

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

路径相对于 `~/.hermes/`。文件挂载到容器内的 `/root/.hermes/`。

### 每个沙箱过滤的内容

| 沙箱 | 默认过滤器 | 穿透覆盖 |
|---------|---------------|---------------------|
| **execute_code** | 阻止名称中包含 `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL`、`PASSWD`、`AUTH` 的变量；仅允许安全前缀变量通过 | ✅ 穿透变量绕过两个检查 |
| **terminal**（本地） | 阻止显式 Hermes 基础设施变量（provider 密钥、gateway 令牌、工具 API 密钥） | ✅ 穿透变量绕过阻止列表 |
| **terminal**（Docker） | 默认无主机 env vars | ✅ 穿透变量 + `docker_forward_env` 通过 `-e` 转发 |
| **terminal**（Modal） | 默认无主机 env/files | ✅ 凭证文件挂载；env 穿透通过同步 |
| **MCP** | 阻止除安全系统变量 + 显式配置的 `env` 之外的所有内容 | ❌ 不受穿透影响（改用 MCP `env` 配置） |

### 安全注意事项

- 穿透仅影响您或您的 skills 显式声明的变量——默认安全态势对任意 LLM 生成代码不变
- 凭证文件以**只读**方式挂载到 Docker 容器
- Skills Guard 在安装前扫描 skill 内容以获取可疑 env 访问模式
- 未设置/未设置的变量永远不会被注册（您无法泄露不存在的内容）
- Hermes 基础设施密钥（provider API 密钥、gateway 令牌）永远不应添加到 `env_passthrough`——它们有专用机制

## MCP 凭证处理

MCP（Model Context Protocol）服务器子进程接收**过滤的环境**以防止凭证意外泄露。

### 安全环境变量

只有这些变量从主机传递到 MCP stdio 子进程：

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

加上任何 `XDG_*` 变量。所有其他环境变量（API 密钥、令牌、密钥）都**被剥离**。

MCP 服务器 `env` 配置中显式定义的变量会通过：

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # 只有这个会传递
```

### 凭证编辑

MCP 工具的错误消息在返回 LLM 之前被清理。以下模式替换为 `[REDACTED]`：

- GitHub PATs（`ghp_...`）
- OpenAI 风格的密钥（`sk-...`）
- Bearer 令牌
- `token=`、`key=`、`API_KEY=`、`password=`、`secret=` 参数

### 网站访问策略

您可以限制 agent 通过其 Web 和浏览器工具访问的网站。这对于防止 agent 访问内部服务、管理面板或其他敏感 URL 很有用。

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

当请求被阻止的 URL 时，工具返回解释该域被策略阻止的错误。阻止列表在 `web_search`、`web_extract`、`browser_navigate` 和所有支持 URL 的工具中强制执行。

请参阅配置指南中的 [网站阻止列表](/docs/user-guide/configuration#website-blocklist) 获取完整详情。

### SSRF 防护

所有支持 URL 的工具（网页搜索、网页提取、视觉、浏览器）在获取之前验证 URL，以防止服务器端请求伪造（SSRF）攻击。阻止的地址包括：

- **私有网络**（RFC 1918）：`10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`
- **环回**：`127.0.0.0/8`、`::1`
- **链路本地**：`169.254.0.0/16`（包括云元数据 `169.254.169.254`）
- **CGNAT/共享地址空间**（RFC 6598）：`100.64.0.0/10`（Tailscale、WireGuard VPN）
- **云元数据主机名**：`metadata.google.internal`、`metadata.goog`
- **保留、多播和未指定地址**

SSRF 防护始终对面向互联网的使用处于活动状态，DNS 失败被视为阻止（fail-closed）。重定向链在每个跃点重新验证，以防止基于重定向的绕过。

#### 故意允许私有 URL

某些设置合法需要私有/内部 URL 访问——将 `home.arpa` 解析为 RFC 1918 空间的家庭网络、LAN-only Ollama/llama.cpp 端点、内部 wiki、云元数据调试等。对于这些情况，有一个全局选择退出：

```yaml
security:
  allow_private_urls: true   # 默认：false
```

开启时，Web 工具、浏览器、视觉 URL 获取和 gateway 媒体下载不再拒绝 RFC 1918 / 环回 / 链路本地 / CGNAT / 云元数据目标。**这是一个有意的信任边界**——仅在将 agent 针对本地网络运行任意提示注入 URL 视为可接受风险的机器上启用。面向公众的 gateway 应保持关闭。

主机子字符串保护（即使底层 IP 是公开的，也会阻止看起来相似的 Unicode 域欺骗技巧）无论此设置如何都保持开启。

### Tirith 预执行安全扫描

Hermes 集成 [tirith](https://github.com/sheeki03/tirith) 用于执行前的内容级命令扫描。Tirith 检测模式匹配单独遗漏的威胁：

- 同形 URL 欺骗（国际化域攻击）
- 管道到解释器模式（`curl | bash`、`wget | sh`）
- 终端注入攻击

Tirith 在首次使用时从 GitHub releases 自动安装，并进行 SHA-256 校验和验证（如果 cosign 可用，还会有 cosign 来源验证）。

```yaml
# 在 ~/.hermes/config.yaml 中
security:
  tirith_enabled: true       # 启用/禁用 tirith 扫描（默认：true）
  tirith_path: "tirith"      # tirith 二进制文件路径（默认：PATH 查找）
  tirith_timeout: 5          # 子进程超时秒数
  tirith_fail_open: true     # 当 tirith 不可用时允许执行（默认：true）
```

当 `tirith_fail_open` 为 `true`（默认）时，如果 tirith 未安装或超时，命令继续。在高安全环境中设置为 `false` 以在 tirith 不可用时阻止命令。

Tirith 的裁决与审批流程集成：安全命令通过，而可疑和阻止的命令都会触发用户审批，并显示完整的 tirith 发现结果（严重性、标题、描述、更安全的替代方案）。用户可以批准或拒绝——默认选择是拒绝，以保持无人值守场景的安全。

### 上下文文件注入保护

上下文文件（AGENTS.md、.cursorrules、SOUL.md）在包含在系统提示之前被扫描提示注入。扫描器检查：

- 忽略/无视先前指示的指示
- 带可疑关键字的隐藏 HTML 注释
- 尝试读取密钥（`.env`、`credentials`、`.netrc`）
- 通过 `curl` 进行凭证泄露
- 不可见 Unicode 字符（零宽空格、双向覆盖）

阻止的文件显示警告：

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## 生产部署最佳实践

### Gateway 部署清单

1. **设置显式允许列表** — 在生产中永远不要使用 `GATEWAY_ALLOW_ALL_USERS=true`
2. **使用容器后端** — 在 config.yaml 中设置 `terminal.backend: docker`
3. **限制资源限制** — 设置适当的 CPU、内存和磁盘限制
4. **安全存储密钥** — 将 API 密钥保存在 `~/.hermes/.env` 中，并设置正确的文件权限
5. **启用 DM 配对** — 尽可能使用配对代码而不是硬编码用户 ID
6. **审查命令允许列表** — 定期审计 config.yaml 中的 `command_allowlist`
7. **设置 `MESSAGING_CWD`** — 不要让 agent 从敏感目录操作
8. **以非 root 运行** — 永远不要以 root 身份运行 gateway
9. **监控日志** — 检查 `~/.hermes/logs/` 中的未授权访问尝试
10. **保持更新** — 定期运行 `hermes update` 以获取安全补丁

### 保护 API 密钥

```bash
# 设置 .env 文件的正确权限
chmod 600 ~/.hermes/.env

# 为不同服务保留单独的密钥
# 永远不要将 .env 文件提交到版本控制
```

### 网络隔离

为获得最大安全性，在单独的机器或 VM 上运行 gateway：

```yaml
terminal:
  backend: ssh
  ssh_host: "agent-worker.local"
  ssh_user: "hermes"
  ssh_key: "~/.ssh/hermes_agent_key"
```

这将 gateway 的消息传递连接与 agent 的命令执行分开。
