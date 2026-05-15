---
sidebar_position: 8
title: "安全"
description: "安全模型、危险命令批准、用户授权、容器隔离以及生产部署最佳实践"
---

# 安全

Hermes Agent 采用深度防御（defense-in-depth）安全模型进行设计。本页覆盖所有安全边界——从命令批准到容器隔离再到消息平台上的用户授权。

## 概览

安全模型共有七层：

1. **用户授权** —— 谁可以与代理对话（白名单、私聊配对）
2. **危险命令批准** —— 对破坏性操作进行人工审查
3. **容器隔离** —— 使用 Docker/Singularity/Modal 沙箱并使用强化设置
4. **MCP 凭证过滤** —— 为 MCP 子进程提供环境变量隔离
5. **上下文文件扫描** —— 检测项目文件中的提示注入
6. **跨会话隔离** —— 会话之间无法访问对方的数据或状态；cron 作业的存储路径针对路径遍历攻击进行硬化
7. **输入消毒** —— 在终端工具后端对工作目录参数进行白名单校验，以防止 Shell 注入

## 危险命令批准

在执行任何命令之前，Hermes 会将其与精心挑选的危险模式列表进行匹配。如果匹配成功，则必须由用户显式批准。

### 批准模式

批准系统支持三种模式，可在 `~/.hermes/config.yaml` 的 `approvals.mode` 中配置：

```yaml
approvals:
  mode: manual    # manual | smart | off
  timeout: 60     # 等待用户响应的秒数（默认 60）
```

| 模式 | 行为 |
|------|------|
| **manual**（默认） | 对所有危险命令始终弹出提示让用户批准 |
| **smart** | 使用辅助 LLM 评估风险。低风险命令（例如 `python -c "print('hello')"`）自动批准。真正危险的命令自动拒绝。无法确定的情况升级为手动提示 |
| **off** | 关闭所有批准检查 —— 等同于使用 `--yolo` 运行。所有命令均不提示 |

:::warning
将 `approvals.mode: off` 关闭所有安全提示。仅在受信任的环境（CI/CD、容器等）中使用。
:::

### YOLO 模式

YOLO 模式会为当前会话绕过 **所有** 危险命令批准提示。可通过三种方式激活：

1. **CLI 标志**：使用 `hermes --yolo` 或 `hermes chat --yolo` 启动会话
2. **斜杠命令**：在会话中输入 `/yolo` 切换开关
3. **环境变量**：设置 `HERMES_YOLO_MODE=1`

`/yolo` 命令是 **切换**——每次使用都会翻转状态：

```
> /yolo
  ⚡ YOLO mode ON — all commands auto‑approved. Use with caution.

> /yolo
  ⚠ YOLO mode OFF — dangerous commands will require approval.
```

YOLO 模式在 CLI 与网关会话中均可使用。内部实现上，它会设置 `HERMES_YOLO_MODE` 环境变量，并在每次命令执行前进行检查。

:::danger
YOLO 模式会禁用 **所有** 危险命令安全检查 **除了** 硬性阻断列表（见下文）。仅在您完全信任生成的命令时使用（例如在一次性环境中的经过充分测试的自动化脚本）。
:::

### 硬性阻断列表（始终开启的底线）

以下命令极具破坏性——不可恢复的文件系统擦除、fork 炸弹、直接写块设备——即使开启 `--yolo`、`approvals.mode: off`、cron 作业的 `approve` 模式或用户手动点击 “always allow”，Hermes 仍会拒绝执行。这些模式在 **审批层之前** 即被拦截，且没有覆盖标志。

当前阻断的模式（非穷尽，随 `tools/approval.py::UNRECOVERABLE_BLOCKLIST` 同步更新）：

| 模式 | 原因 |
|---|---|
| `rm -rf /` 及明显变体 | 删除根文件系统 |
| `rm -rf --no-preserve-root /` | 明确表示要删除根目录 |
| `:(){ :|:& };:`（bash fork 炸弹） | 使主机卡死直至重启 |
| `mkfs.*` 在已挂载的根设备上 | 格式化运行系统 |
| `dd if=/dev/zero of=/dev/sd*` | 将物理磁盘清零 |
| 在根文件系统顶部将不可信 URL 管道至 `sh` | 过宽的远程代码执行向量 |

若触发阻断列表，工具调用会返回解释性错误，且不会实际运行。如果您确实需要执行此类命令（例如在擦除‑重新安装流水线中），请在代理外部手动运行。

### 批准超时

当出现危险命令提示时，用户有可配置的响应时间。若在超时内未作出响应，默认 **拒绝**（fail‑closed）。

在 `~/.hermes/config.yaml` 中配置超时：

```yaml
approvals:
  timeout: 60  # 秒（默认 60）
```

### 触发批准的模式

以下模式会触发批准提示（定义于 `tools/approval.py`）：

| 模式 | 描述 |
|------|------|
| `rm -r` / `rm --recursive` | 递归删除 |
| `rm ... /` | 在根路径下删除 |
| `chmod 777/666` / `o+w` / `a+w` | 设为全局可写权限 |
| `chmod --recursive` 与不安全权限组合 |
| `chown -R root` / `chown --recursive root` | 递归 chown 为 root |
| `mkfs` | 格式化文件系统 |
| `dd if=` | 磁盘拷贝 |
| `> /dev/sd` | 写块设备 |
| `DROP TABLE/DATABASE` | SQL DROP |
| `DELETE FROM`（无 WHERE） | SQL DELETE 无条件 |
| `TRUNCATE TABLE` | SQL TRUNCATE |
| `> /etc/` | 覆写系统配置文件 |
| `systemctl stop/restart/disable/mask` | 停止/重启/禁用系统服务 |
| `kill -9 -1` | 杀死所有进程 |
| `pkill -9` | 强制杀进程 |
| fork 炸弹模式 |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | 通过 `-c` 执行脚本 |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | 通过 `-e`/`-c` 执行单行脚本 |
| `curl ... \| sh` / `wget ... \| sh` | 将远程内容管道至 shell |
| `bash <(curl ...)` / `sh <(wget ...)` | 通过进程替换执行远程脚本 |
| `tee` 到 `/etc/`、`~/.ssh/`、`~/.hermes/.env` | 通过 tee 覆写敏感文件 |
| `>` / `>>` 到 `/etc/`、`~/.ssh/`、`~/.hermes/.env` | 通过重定向覆盖敏感文件 |
| `xargs rm` | xargs 与 rm 组合 |
| `find -exec rm` / `find -delete` | find 与破坏性动作结合 |
| `cp`/`mv`/`install` 到 `/etc/` | 将文件复制/移动到系统配置目录 |
| `sed -i` / `sed --in-place` 在 `/etc/` 上 | 原地编辑系统配置 |
| `pkill`/`killall` hermes/gateway | 防止自我终止 |
| `gateway run` 与 `&`/`disown`/`nohup`/`setsid` 组合 | 防止在服务管理器之外启动网关 |

:::info
**容器绕过**：在 `docker`、`singularity`、`modal`、`daytona` 或 `vercel_sandbox` 后端运行时，危险命令检查会 **跳过**，因为容器本身已经是安全边界。容器内的破坏性命令不会危及宿主机。
:::

### 审批流程（CLI）

在交互式 CLI 中，危险命令会显示内联批准提示：

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

四个选项含义：

- **once** —— 仅本次执行允许
- **session** —— 本会话期间均允许此模式
- **always** —— 写入永久白名单（保存至 `config.yaml`）
- **deny**（默认） —— 阻止命令执行

### 审批流程（网关/消息平台）

在消息平台中，代理会将危险命令细节发送至聊天，并等待用户回复：

- 回复 **yes**、**y**、**approve**、**ok**、**go** 进行批准
- 回复 **no**、**n**、**deny**、**cancel** 拒绝

`HERMES_EXEC_ASK=1` 环境变量会在网关运行时自动设置。

### 永久白名单

使用 “always” 批准的命令会写入 `~/.hermes/config.yaml`：

```yaml
# 永久允许的危险命令模式
command_allowlist:
  - rm
  - systemctl
```

这些模式在启动时加载，并在所有后续会话中静默批准。

:::tip
使用 `hermes config edit` 查看或删除永久白名单中的模式。
:::

## 用户授权（网关）

在运行消息网关时，Hermes 通过分层授权系统控制谁可以与机器人交互。

### 授权检查顺序

`_is_user_authorized()` 方法按以下顺序检查：

1. **平台特定的全局允许标志**（如 `DISCORD_ALLOW_ALL_USERS=true`）
2. **已配对的私聊代码列表**（通过配对码批准的用户）
3. **平台特定白名单**（例如 `TELEGRAM_ALLOWED_USERS=12345,67890`）
4. **全局白名单**（`GATEWAY_ALLOWED_USERS=12345,67890`）
5. **全局全开放**（`GATEWAY_ALLOW_ALL_USERS=true`）
6. **默认：拒绝**

### 平台白名单

在 `~/.hermes/.env` 中以逗号分隔方式设置允许的用户 ID：

```bash
# 平台特定白名单
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# 跨平台白名单（所有平台统一检查）
GATEWAY_ALLOWED_USERS=123456789

# 平台全开放（谨慎使用）
DISCORD_ALLOW_ALL_USERS=true

# 全局全开放（极度谨慎）
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
如果 **未配置任何白名单** 且 `GATEWAY_ALLOW_ALL_USERS` 未设置，**所有用户均被拒绝**。网关启动时会记录警告：

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### 私聊配对系统

为提供更灵活的授权，Hermes 包含基于一次性配对码的系统。未知用户会收到配对码，机器人所有者在 CLI 上批准后即可永久授权该用户。

**工作流程**：

1. 未知用户给机器人发送私聊
2. 机器人回复 8 位配对码
3. 机器人所有者运行 `hermes pairing approve <platform> <code>`
4. 该用户在该平台上永久获准

在 `~/.hermes/config.yaml` 中配置未授权私聊的处理方式：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` 为默认：未授权私聊会收到配对码回复
- `ignore` 静默丢弃未授权私聊
- 平台段可以覆盖全局默认，以实现如 Telegram 采用配对而 WhatsApp 直接忽略的需求

**安全特性**（基于 OWASP + NIST SP 800‑63‑4 指南）：

| 特性 | 说明 |
|------|------|
| 代码格式 | 8 位，使用 32 位无歧义字符（不含 0/O/1/I） |
| 随机性 | 加密 (`secrets.choice()`) |
| 代码有效期 | 1 小时 |
| 限流 | 每用户每 10 分钟最多 1 次请求 |
| 待处理上限 | 每平台最多 3 个待处理代码 |
| 锁定 | 5 次失败批准 → 1 小时锁定 |
| 文件安全 | 对所有配对数据文件 `chmod 0600` |
| 日志 | 代码从不记录到 stdout |

**配对 CLI 命令**：

```bash
# 列出待处理和已批准的用户
hermes pairing list

# 批准配对码
hermes pairing approve telegram ABC12DEF

# 撤销用户访问
hermes pairing revoke telegram 123456789

# 清除所有待处理代码
hermes pairing clear-pending
```

**存储**：配对数据保存在 `~/.hermes/pairing/`，每个平台对应 JSON 文件：
- `{platform}-pending.json` — 待处理配对请求
- `{platform}-approved.json` — 已批准用户
- `_rate_limits.json` — 限流与锁定追踪

## 容器隔离

使用 `docker` 终端后端时，Hermes 对每个容器施加严格的安全加固。

### Docker 安全标记

每个容器都使用以下标记（定义于 `tools/environments/docker.py`）：

```python
_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # 丢弃所有 Linux 能力
    "--cap-add", "DAC_OVERRIDE",                  # 让 root 能写入挂载目录
    "--cap-add", "CHOWN",                         # 包管理器需要文件所有权
    "--cap-add", "FOWNER",                        # 同上
    "--security-opt", "no-new-privileges",         # 阻止特权升级
    "--pids-limit", "256",                         # 限制进程数
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # 限制大小的 /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # /var/tmp 禁止 exec
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",   # /run 禁止 exec
]
```

### 资源限制

容器资源可在 `~/.hermes/config.yaml` 中配置：

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # 仅显式白名单；空列表保持凭证不进入容器
  container_cpu: 1        # CPU 核心数
  container_memory: 5120  # MB（默认 5GB）
  container_disk: 51200   # MB（默认 50GB，需要 XFS 上的 overlay2）
  container_persistent: true  # 跨会话持久化文件系统
```

### 文件系统持久化

- **持久模式** (`container_persistent: true`)：将 `~/.hermes/sandboxes/docker/<task_id>/` 中的 `/workspace` 与 `/root` 绑定挂载
- **临时模式** (`container_persistent: false`)：使用 tmpfs，所有工作区在清理时即被删除

:::tip
在生产网关部署时，请使用 `docker`、`modal`、`daytona` 或 `vercel_sandbox` 后端，以将代理命令与宿主系统隔离。这可以完全省去危险命令批准机制。
:::

:::warning
如果在 `terminal.docker_forward_env` 中添加变量名，这些变量会被显式注入容器供终端命令使用。适用于任务特定凭证（如 `GITHUB_TOKEN`），但也意味着容器内代码能够读取并外泄这些凭证。
:::

## 终端后端安全对比

| 后端 | 隔离程度 | 危险命令检查 | 适用场景 |
|------|----------|---------------|----------|
| **local** | 无 —— 直接在宿主机运行 | ✅ 有 | 开发、可信用户 |
| **ssh** | 远程机器 | ✅ 有 | 在独立服务器上运行 |
| **docker** | 容器 | ❌ 跳过（容器即边界） | 生产网关 |
| **singularity** | 容器 | ❌ 跳过 | HPC 环境 |
| **modal** | 云沙箱 | ❌ 跳过 | 可扩展云隔离 |
| **daytona** | 云沙箱 | ❌ 跳过 | 持久化云工作区 |
| **vercel_sandbox** | 云微虚拟机 | ❌ 跳过 | 云执行并支持快照持久化 |

## 环境变量透传 {#environment-variable-passthrough}

`execute_code` 与 `terminal` 会从子进程中剥离敏感环境变量，以防止 LLM 生成的代码泄露凭证。但声明了 `required_environment_variables` 的技能会自动获得所需变量的透传权限。

### 工作原理

有两种机制可让特定变量通过沙箱过滤：

1. **技能级透传（自动）**
   当加载带有 `required_environment_variables` 声明的技能时，凡在宿主环境中已设置的变量会自动注册为透传。缺失的变量不会注册。

   ```yaml
   # 在技能的 SKILL.md frontmatter 中
   required_environment_variables:
     - name: TENOR_API_KEY
       prompt: Tenor API key
       help: Get a key from https://developers.google.com/tenor
   ```

   加载该技能后，`TENOR_API_KEY` 会在 `execute_code`、`terminal`（本地）以及远程后端（Docker、Modal）中自动可用，无需手动配置。

   :::info Docker & Modal
   在 v0.5.1 之前，Docker 的 `forward_env` 与技能透传是两套系统。现已合并——技能声明的环境变量会自动转发至 Docker 容器和 Modal 沙箱，无需手动添加至 `docker_forward_env`。
   :::

2. **配置式透传（手动）**
   对于未被任何技能声明的变量，可在 `config.yaml` 中手动添加至 `terminal.env_passthrough`：

   ```yaml
   terminal:
     env_passthrough:
       - MY_CUSTOM_KEY
       - ANOTHER_TOKEN
   ```

### 凭证文件透传（OAuth 令牌等） {#credential-file-passthrough}

部分技能需要 **文件**（而非仅环境变量）在沙箱中可用，例如 Google Workspace 将 OAuth 令牌存放为 `google_token.json`。技能在 frontmatter 中声明：

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

加载后，Hermes 会检查这些文件是否存在于活动配置文件夹的 `HERMES_HOME`，并在不同后端中进行挂载：
- **Docker**：只读绑定挂载 (`-v host:container:ro`)
- **Modal**：在沙箱创建时挂载，并在每次命令前同步（支持中途 OAuth 设置）
- **本地**：无需操作，文件本身已可访问

也可以在 `config.yaml` 中手动列出凭证文件：

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

路径相对于 `~/.hermes/`，在容器内部会挂载至 `/root/.hermes/`。

### 各沙箱过滤规则概览

| 沙箱 | 默认过滤 | 透传覆盖 |
|------|----------|----------|
| **execute_code** | 阻止包含 `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD`, `AUTH` 的变量；仅允许安全前缀变量 | ✅ 透传变量可绕过两层检查 |
| **terminal**（本地） | 阻止 Hermes 基础设施变量（provider keys、gateway token、tool API keys） | ✅ 透传变量可绕过阻断列表 |
| **terminal**（Docker） | 默认不转发宿主环境变量 | ✅ 透传变量 + `docker_forward_env` 通过 `-e` 注入 |
| **terminal**（Modal） | 默认不转发宿主环境变量/文件 | ✅ 凭证文件挂载；环境变量通过同步透传 |
| **MCP** | 仅保留系统变量 + 明确配置的 `env` 项 | ❌ 不受透传影响（需在 MCP `env` 中配置） |

### 安全考量

- 透传仅限于您或技能明确声明的变量，默认安全姿态不受影响
- 凭证文件在 Docker 中为 **只读** 挂载
- 技能 Guard 在安装前会扫描技能内容，检测可疑的环境变量访问模式
- 未设置或不存在的变量永远不会被注册（不存在的东西不可能泄露）
- Hermes 基础设施的密钥（provider API key、gateway token）绝不应加入 `env_passthrough`，它们有专用的机制进行管理

## MCP 凭证处理

MCP（Model Context Protocol）服务器子进程会获取 **过滤后的环境变量**，以防止意外泄露凭证。

### 安全环境变量

仅以下变量会从宿主传递至 MCP 子进程：

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```
以及任何 `XDG_*` 变量。其它所有环境变量（API 密钥、令牌、秘密）均会被 **剥离**。

可以在 MCP 服务器的 `env` 配置中显式声明要透传的变量：

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # 仅此变量会被传递
```

### 凭证脱敏

MCP 工具返回的错误信息在返回给 LLM 前会被清理，以下模式会被替换为 `[REDACTED]`：

- GitHub PAT (`ghp_…`)
- OpenAI‑style keys (`sk‑…`)
- Bearer tokens
- `token=`, `key=`, `API_KEY=`, `password=`, `secret=` 参数

## 网站访问策略

您可以通过配置限制代理能够访问的站点，以防止访问内部服务、管理面板或其他敏感 URL。

```yaml
# 位于 ~/.hermes/config.yaml
security:
  website_blocklist:
    enabled: true
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

当请求被阻止的 URL 时，工具会返回错误说明该域名被策略阻止。该阻名单在 `web_search`、`web_extract`、`browser_navigate` 以及所有具备 URL 能力的工具中统一生效。详细信息请参考配置指南中的 [Website Blocklist](/docs/user-guide/configuration#website-blocklist)。

## SSRF 保护

所有具备 URL 能力的工具（网页搜索、网页提取、视觉、浏览器）在获取前都会验证 URL，以防止服务器端请求伪造（SSRF）攻击。被阻断的地址包括：

- **私有网络**（RFC 1918）：`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
- **回环**：`127.0.0.0/8`, `::1`
- **链路本地**：`169.254.0.0/16`（包括云元数据 `169.254.169.254`）
- **CGNAT / 共享地址空间**（RFC 6598）：`100.64.0.0/10`（Tailscale、WireGuard VPN 等）
- **云元数据主机名**：`metadata.google.internal`, `metadata.goog`
- **保留、组播和未指定地址**

SSRF 保护在面向互联网的使用场景下始终开启，DNS 解析失败视作被阻断（fail‑closed）。重定向链会在每一次跳转时重新校验，以防止通过重定向绕过。

### 有意允许私有 URL

某些部署确实需要访问私有/内部 URL——例如在本地网络中解析 `home.arpa` 指向 RFC 1918 地址、LAN‑only 的 Ollama/llama.cpp 端点、内部 wiki、云元数据调试等。对此类需求提供全局关闭选项：

```yaml
security:
  allow_private_urls: true   # 默认: false
```

开启后，网页工具、浏览器、视觉 URL 获取以及网关媒体下载将不再拒绝 RFC 1918 / 回环 / 链路本地 / CGNAT / 云元数据目标。**这是一条明确的信任边界**——仅在您能够接受代理对本地网络进行任意 URL 请求的机器上启用。面向公共的网关应保持关闭。

即使开启该选项，域名同形异体（Unicode 伪造）防护仍然保持开启。

## Tirith 预执行安全扫描

Hermes 集成了 [tirith](https://github.com/sheeki03/tirith) 进行内容层面的命令扫描。Tirith 能检测模式匹配无法捕获的威胁：

- 同形异体 URL 欺骗（国际化域名攻击）
- 管道至解释器模式（`curl | bash`, `wget | sh`）
- 终端注入攻击

Tirith 会在首次使用时自动从 GitHub Release 下载，并进行 SHA‑256 校验（若系统中有 cosign 则进行可供性验证）。

```yaml
# 位于 ~/.hermes/config.yaml
security:
  tirith_enabled: true       # 启用/禁用 tirith 扫描（默认 true）
  tirith_path: "tirith"      # tirith 二进制路径（默认在 PATH 中查找）
  tirith_timeout: 5          # 子进程超时秒数
  tirith_fail_open: true     # tirith 不可用或超时时仍允许执行（默认 true）
```

当 `tirith_fail_open` 为 `true`（默认），若 tirith 未安装或超时，命令仍会继续执行。若在高安全环境中希望在 tirith 不可用时阻止命令，请将其设为 `false`。

Tirith 的判定会整合到批准流程中：安全命令直接通过，疑似或被阻断的命令会向用户展示 tirith 检测结果（严重程度、标题、描述、可行的更安全替代方案），用户可批准或拒绝——默认选择为拒绝，以保障无人值守场景的安全。

## 上下文文件注入防护

上下文文件（AGENTS.md、.cursorrules、SOUL.md）在被加入系统提示前会进行提示注入扫描。扫描检测以下风险：

- 指示忽略先前指令的指令
- 含有可疑关键字的隐藏 HTML 注释
- 读取敏感文件（`.env`、`credentials`、`.netrc`）的尝试
- 通过 `curl` 的凭证外泄
- 隐形 Unicode 字符（零宽空格、双向覆盖符）

被阻断的文件会显示警告：

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## 生产部署最佳实践

### 网关部署检查清单

1. **设置明确的白名单** —— 生产环境切勿使用 `GATEWAY_ALLOW_ALL_USERS=true`
2. **使用容器后端** —— 在 `config.yaml` 中设 `terminal.backend: docker`
3. **限制资源配额** —— 合理设置 CPU、内存、磁盘限制
4. **安全存储密钥** —— 将 API 密钥放入 `~/.hermes/.env`，并确保文件权限得当
5. **启用 DM 配对** —— 如可能，使用配对码代替硬编码用户 ID
6. **审计命令白名单** —— 定期检查 `command_allowlist` 中的条目
7. **设置 `MESSAGING_CWD`** —— 不让代理在敏感目录下运行
8. **以非 root 身份运行** —— 永不以 root 身份运行网关
9. **监控日志** —— 检查 `~/.hermes/logs/` 中的未授权访问尝试
10. **保持更新** —— 定期运行 `hermes update` 以获取安全补丁

### 保护 API 密钥

```bash
# 为 .env 文件设置合适的权限
chmod 600 ~/.hermes/.env

# 为不同服务使用独立密钥
# 永不将 .env 提交至版本控制
```

### 网络隔离

若要获得最高安全性，请在独立机器或虚拟机上运行网关。将 `terminal.backend: ssh` 写入 `config.yaml`，并在 `~/.hermes/.env` 中提供主机信息：

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

SSH 连接细节存放在 `.env`（而非 `config.yaml`），以避免在导出或共享配置时泄露。这样即可将网关的消息连接与代理的命令执行彻底分离。
