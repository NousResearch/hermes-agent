---
sidebar_position: 9
title: "Matrix"
description: "将 Hermes Agent 设置为 Matrix 机器人"
---

# Matrix 设置

Hermes Agent 与 Matrix 集成，Matrix 是一个开放的、联邦化的消息传递协议。Matrix 让你运行自己的 homeserver 或使用像 matrix.org 这样的公共服务器 —— 无论哪种方式，你都保持对通信的控制。机器人通过 `mautrix` Python SDK 连接，通过 Hermes Agent 管道（包括工具使用、记忆和推理）处理消息，并实时回复。它支持文本、文件附件、图片、音频、视频和可选的端到端加密（E2EE）。

Hermes 可与任何 Matrix homeserver 配合使用 —— Synapse、Conduit、Dendrite 或 matrix.org。

在设置之前，这里是大多数人想了解的内容：Hermes 连接后是如何行为的。

## Hermes 的行为方式

| 上下文 | 行为 |
|---------|----------|
| **私信** | Hermes 回复每条消息。不需要 `@mention`。每个私信都有自己的会话。将 `MATRIX_DM_MENTION_THREADS=true` 设置为在机器人被 @mention 时启动一个主题。 |
| **房间** | 默认情况下，Hermes 需要 `@mention` 才能回复。设置 `MATRIX_REQUIRE_MENTION=false` 或将房间 ID 添加到 `MATRIX_FREE_RESPONSE_ROOMS` 以获得自由回复房间。房间邀请会自动接受。 |
| **主题** | Hermes 支持 Matrix 主题（MSC3440）。如果你在主题中回复，Hermes 会将主题上下文与主房间时间线隔离。机器人已参与的主题不需要提及。 |
| **自动主题** | 默认情况下，Hermes 为它在房间中回复的每条消息自动创建一个主题。这使对话保持隔离。设置 `MATRIX_AUTO_THREAD=false` 以禁用。 |
| **多用户共享房间** | 默认情况下，Hermes 在房间内为每个用户隔离会话历史。同一房间中的两个人不会共享一个对话记录，除非你明确禁用该选项。 |

:::tip
机器人在被邀请时自动加入房间。只需将机器人的 Matrix 用户邀请到任何房间，它就会加入并开始回复。
:::

### Matrix 中的会话模型

默认情况下：

- 每个私信都有自己的会话
- 每个主题都有自己的会话命名空间
- 共享房间中的每个用户在该房间内都有自己的会话

这可以通过 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望整个房间共享一个对话时，才将其设置为 `false`：

```yaml
group_sessions_per_user: false
```

共享会话对于协作房间可能很有用，但它们也意味着：

- 用户共享上下文增长和 token 成本
- 一个人的长时间工具密集型任务可能会使其他人的上下文膨胀
- 一个人的运行中任务可能会中断同一房间中另一个人的后续操作

### 提及和主题配置

你可以通过环境变量或 `config.yaml` 配置提及和自动主题行为：

```yaml
matrix:
  require_mention: true           # 在房间中需要 @mention（默认：true）
  free_response_rooms:            # 免除提及要求的房间
    - "!abc123:matrix.org"
  auto_thread: true               # 为回复自动创建主题（默认：true）
  dm_mention_threads: false       # 在 DM 中被 @mention 时创建主题（默认：false）
```

或通过环境变量：

```bash
MATRIX_REQUIRE_MENTION=true
MATRIX_FREE_RESPONSE_ROOMS=!abc123:matrix.org,!def456:matrix.org
MATRIX_AUTO_THREAD=true
MATRIX_DM_MENTION_THREADS=false
MATRIX_REACTIONS=true          # 默认：true — 处理期间的 emoji 反应
```

:::tip 禁用反应
`MATRIX_REACTIONS=false` 关闭机器人在入站消息上发布处理生命周期 emoji 反应（👀/✅/❌）。对于反应事件嘈杂或并非所有参与的客户端都支持的房间很有用。
:::

:::note
如果你从没有 `MATRIX_REQUIRE_MENTION` 的版本升级，机器人以前会回复房间中的所有消息。要保留该行为，请设置 `MATRIX_REQUIRE_MENTION=false`。
:::

本指南将引导你完成完整的设置过程 —— 从创建你的机器人账号到发送你的第一条消息。

## 步骤 1：创建机器人账号

你需要为机器人提供一个 Matrix 用户账号。有几种方法可以做到这一点：

### 选项 A：在你的 Homeserver 上注册（推荐）

如果你运行自己的 homeserver（Synapse、Conduit、Dendrite）：

1. 使用管理 API 或注册工具创建新用户：

```bash
# Synapse 示例
register_new_matrix_user -c /etc/synapse/homeserver.yaml http://localhost:8008
```

2. 选择一个用户名如 `hermes` —— 完整的用户 ID 将是 `@hermes:your-server.org`。

### 选项 B：使用 matrix.org 或其他公共 Homeserver

1. 前往 [Element Web](https://app.element.io) 并创建一个新账号。
2. 为你的机器人选择一个用户名（例如 `hermes-bot`）。

### 选项 C：使用你自己的账号

你也可以将 Hermes 作为你自己的用户运行。这意味着机器人以你发帖 —— 对于个人助手很有用。

## 步骤 2：获取访问令牌

Hermes 需要访问令牌来向 homeserver 认证。你有两个选择：

### 选项 A：访问令牌（推荐）

获取令牌的最可靠方法：

**通过 Element：**

1. 使用机器人账号登录 [Element](https://app.element.io)。
2. 转到 **Settings** → **Help & About**。
3. 向下滚动并展开 **Advanced** —— 访问令牌显示在那里。
4. **立即复制它。**

**通过 API：**

```bash
curl -X POST https://your-server/_matrix/client/v3/login \
  -H "Content-Type: application/json" \
  -d '{
    "type": "m.login.password",
    "user": "@hermes:your-server.org",
    "password": "your-password"
  }'
```

响应包含一个 `access_token` 字段 —— 复制它。

:::warning[保护好你的访问令牌]
访问令牌授予对机器人 Matrix 账号的完全访问权限。永远不要公开分享它或将其提交到 Git。如果泄露，通过注销该用户的所有会话来撤销它。
:::

### 选项 B：密码登录

你可以提供机器人的用户 ID 和密码，而不是提供访问令牌。Hermes 会在启动时自动登录。这更简单，但意味着密码存储在你的 `.env` 文件中。

```bash
MATRIX_USER_ID=@hermes:your-server.org
MATRIX_PASSWORD=your-password
```

## 步骤 3：找到你的 Matrix 用户 ID

Hermes Agent 使用你的 Matrix 用户 ID 来控制谁可以与机器人交互。Matrix 用户 ID 遵循格式 `@username:server`。

要找到你的：

1. 打开 [Element](https://app.element.io)（或你首选的 Matrix 客户端）。
2. 点击你的头像 → **Settings**。
3. 你的用户 ID 显示在个人资料顶部（例如 `@alice:matrix.org`）。

:::tip
Matrix 用户 ID 始终以 `@` 开头，包含 `:` 后跟服务器名称。例如：`@alice:matrix.org`、`@bob:your-server.com`。
:::

## 步骤 4：配置 Hermes Agent

### 选项 A：交互式设置（推荐）

运行引导式设置命令：

```bash
hermes gateway setup
```

当提示时选择 **Matrix**，然后在询问时提供你的 homeserver URL、访问令牌（或用户 ID + 密码）和允许的用户 ID。

### 选项 B：手动配置

将以下内容添加到你的 `~/.hermes/.env` 文件：

**使用访问令牌：**

```bash
# 必需
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# 可选：用户 ID（如果省略则从令牌自动检测）
# MATRIX_USER_ID=@hermes:matrix.example.org

# 安全：限制谁可以与机器人交互
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# 多个允许的用户（逗号分隔）
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

**使用密码登录：**

```bash
# 必需
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_USER_ID=@hermes:matrix.example.org
MATRIX_PASSWORD=***

# 安全
MATRIX_ALLOWED_USERS=@alice:matrix.example.org
```

`~/.hermes/config.yaml` 中的可选行为设置：

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true` 保持每个参与者的上下文在共享房间中隔离

### 启动 Gateway

配置完成后，启动 Matrix gateway：

```bash
hermes gateway
```

机器人应在几秒钟内连接到你的 homeserver 并开始同步。向它发送一条消息 —— 私信或它已加入的房间 —— 进行测试。

:::tip
你可以在后台运行 `hermes gateway` 或作为 systemd 服务以实现持久运行。详见部署文档。
:::

## 端到端加密（E2EE）

Hermes 支持 Matrix 端到端加密，因此你可以在加密房间中与你的机器人聊天。

### 要求

E2EE 需要带有加密额外组件的 `mautrix` 库和 `libolm` C 库：

```bash
# 安装带有 E2EE 支持的 mautrix
pip install 'mautrix[encryption]'

# 或使用 hermes 额外组件安装
pip install 'hermes-agent[matrix]'
```

你还需要在系统上安装 `libolm`：

```bash
# Debian/Ubuntu
sudo apt install libolm-dev

# macOS
brew install libolm

# Fedora
sudo dnf install libolm-devel
```

### 启用 E2EE

将以下内容添加到你的 `~/.hermes/.env`：

```bash
MATRIX_ENCRYPTION=true
```

当 E2EE 启用时，Hermes：

- 将加密密钥存储在 `~/.hermes/platforms/matrix/store/`（旧安装：`~/.hermes/matrix/store/`）
- 在首次连接时上传设备密钥
- 自动解密入站消息和加密出站消息
- 被邀请时自动加入加密房间

### 交叉签名验证（推荐）

如果你的 Matrix 账号启用了交叉签名（Element 中的默认设置），请设置恢复密钥，以便机器人在启动时可以对自身进行签名。没有这个，其他 Matrix 客户端可能会在设备密钥轮换后拒绝与机器人共享加密会话。

```bash
MATRIX_RECOVERY_KEY=EsT... 你的恢复密钥
```

**在哪里找到它：** 在 Element 中，转到 **Settings** → **Security & Privacy** → **Encryption** → 你的恢复密钥（也称为"安全密钥"）。这是在你首次设置交叉签名时被要求保存的密钥。

每次启动时，如果设置了 `MATRIX_RECOVERY_KEY`，Hermes 会从 homeserver 的安全密钥存储中导入交叉签名密钥并签署当前设备。这是幂等的，可以安全地永久启用。

:::warning[删除加密存储]
如果你删除 `~/.hermes/platforms/matrix/store/crypto.db`，机器人会丢失其加密身份。简单地使用相同的设备 ID 重启**不会**完全恢复 —— homeserver 仍然持有用旧身份密钥签名的一次性密钥，对等方无法建立新的 Olm 会话。

Hermes 在启动时检测到此情况并拒绝启用 E2EE，记录：`device XXXX has stale one-time keys on the server signed with a previous identity key`。

**最简单的恢复：生成一个新的访问令牌**（这会获得一个新的设备 ID，没有陈旧密钥历史）。请参阅下面的"从带有 E2EE 的先前版本升级"部分。这是最可靠的路径，避免接触 homeserver 数据库。

**手动恢复**（高级 —— 保持相同的设备 ID）：

1. 停止 Synapse 并从其数据库中删除旧设备：
   ```bash
   sudo systemctl stop matrix-synapse
   sudo sqlite3 /var/lib/matrix-synapse/homeserver.db "
     DELETE FROM e2e_device_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_one_time_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_fallback_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM devices WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
   "
   sudo systemctl start matrix-synapse
   ```
   或者通过 Synapse 管理 API（注意 URL 编码的用户 ID）：
   ```bash
   curl -X DELETE -H "Authorization: Bearer ADMIN_TOKEN" \
     'https://your-server/_synapse/admin/v2/users/%40hermes%3Ayour-server/devices/DEVICE_ID'
   ```
   注意：通过管理 API 删除设备也可能使关联的访问令牌失效。之后你可能需要生成一个新令牌。

2. 删除本地加密存储并重启 Hermes：
   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db*
   # 重启 hermes
   ```

其他 Matrix 客户端（Element、matrix-commander）可能会缓存旧的设备密钥。恢复后，在 Element 中输入 `/discardsession` 以强制与机器人建立新的加密会话。
:::

:::info
如果未安装 `mautrix[encryption]` 或缺少 `libolm`，机器人会自动回退到普通（未加密）客户端。你会在日志中看到警告。
:::

## 主页房间

你可以指定一个"主页房间"，机器人可以在那里发送主动消息（如 cron 作业输出、提醒和通知）。有两种设置方法：

### 使用斜杠命令

在你所在的任何 Matrix 房间中输入 `/sethome`。该房间成为主页房间。

### 手动配置

将以下内容添加到你的 `~/.hermes/.env`：

```bash
MATRIX_HOME_ROOM=!abc123def456:matrix.example.org
```

:::tip
要找到房间 ID：在 Element 中，转到房间 → **Settings** → **Advanced** → **Internal room ID** 显示在那里（以 `!` 开头）。
:::

## 故障排除

### 机器人未响应消息

**原因**：机器人未加入房间，或 `MATRIX_ALLOWED_USERS` 不包含你的用户 ID。

**修复**：邀请机器人到房间 —— 它在邀请时自动加入。验证你的用户 ID 在 `MATRIX_ALLOWED_USERS` 中（使用完整的 `@user:server` 格式）。重启 gateway。

### 启动时出现"Failed to authenticate" / "whoami failed"

**原因**：访问令牌或 homeserver URL 不正确。

**修复**：验证 `MATRIX_HOMESERVER` 指向你的 homeserver（包含 `https://`，没有尾部斜杠）。检查 `MATRIX_ACCESS_TOKEN` 是否有效 —— 用 curl 尝试它：

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/_matrix/client/v3/account/whoami
```

如果这返回你的用户信息，令牌是有效的。如果返回错误，请生成新令牌。

### "mautrix not installed" 错误

**原因**：未安装 `mautrix` Python 包。

**修复**：安装它：

```bash
pip install 'mautrix[encryption]'
```

或使用 Hermes 额外组件：

```bash
pip install 'hermes-agent[matrix]'
```

### 加密错误 / "could not decrypt event"

**原因**：缺少加密密钥、未安装 `libolm` 或机器人的设备不受信任。

**修复**：

1. 验证 `libolm` 已安装在系统上（请参阅上面的 E2EE 部分）。
2. 确保你的 `.env` 中设置了 `MATRIX_ENCRYPTION=true`。
3. 在你的 Matrix 客户端（Element）中，转到机器人的个人资料 -> Sessions -> 验证/信任机器人的设备。
4. 如果机器人刚加入一个加密房间，它只能解密加入*后*发送的消息。旧消息不可访问。

### 从带有 E2EE 的先前版本升级

:::tip
如果你也手动删除了 `crypto.db`，请参阅上面的 E2EE 部分中的"删除加密存储"警告 —— 还有其他步骤要从 homeserver 清除陈旧的一次性密钥。
:::

如果你之前使用 `MATRIX_ENCRYPTION=true` 使用 Hermes，并升级到使用新的基于 SQLite 的加密存储的版本，机器人的加密身份已更改。你的 Matrix 客户端（Element）可能会缓存旧设备密钥并拒绝与机器人共享加密会话。

**症状**：机器人连接并在日志中显示"E2EE enabled"，但所有消息显示"could not decrypt event"，机器人从不回复。

**正在发生的事情**：旧加密状态（来自先前的 `matrix-nio` 或基于序列化的 `mautrix` 后端）与新的 SQLite 加密存储不兼容。机器人创建了一个新的加密身份，但你的 Matrix 客户端仍然缓存旧密钥，不会与密钥已更改的设备共享房间的加密会话。这是 Matrix 安全功能 —— 客户端将同一设备的更改身份密钥视为可疑。

**修复**（一次性迁移）：

1. **生成新的访问令牌** 以获得新的设备 ID。最简单的方法：

   ```bash
   curl -X POST https://your-server/_matrix/client/v3/login \
     -H "Content-Type: application/json" \
     -d '{
       "type": "m.login.password",
       "identifier": {"type": "m.id.user", "user": "@hermes:your-server.org"},
       "password": "***",
       "initial_device_display_name": "Hermes Agent"
     }'
   ```

   复制新的 `access_token` 并在 `~/.hermes/.env` 中更新 `MATRIX_ACCESS_TOKEN`。

2. **删除旧加密状态**：

   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db
   rm -f ~/.hermes/platforms/matrix/store/crypto_store.*
   ```

3. **设置你的恢复密钥**（如果你使用交叉签名 —— 大多数 Element 用户使用）。添加到 `~/.hermes/.env`：

   ```bash
   MATRIX_RECOVERY_KEY=EsT... 你的恢复密钥
   ```

   这让机器人在启动时用交叉签名密钥对自身进行签名，以便 Element 立即信任新设备。没有这个，Element 可能会将新设备视为未验证并拒绝共享加密会话。在 Element 中的 **Settings** → **Security & Privacy** → **Encryption** 下找到你的恢复密钥。

4. **强制你的 Matrix 客户端轮换加密会话**。在 Element 中，打开与机器人的 DM 房间并输入 `/discardsession`。这会强制 Element 创建一个新的加密会话并与机器人的新设备共享它。

5. **重启 gateway**：

   ```bash
   hermes gateway run
   ```

   如果设置了 `MATRIX_RECOVERY_KEY`，你应该在日志中看到 `Matrix: cross-signing verified via recovery key`。

6. **发送新消息**。机器人应该正常解密和回复。

:::note
迁移后，*升级之前*发送的消息无法解密 —— 旧的加密密钥已消失。这仅影响过渡；新消息正常工作。
:::

:::tip
**新安装不受影响。** 只有在你之前有带有先前版本 Hermes 的工作 E2EE 设置并正在升级时，才需要此迁移。

**为什么需要新的访问令牌？** 每个 Matrix 访问令牌绑定到特定的设备 ID。使用新加密密钥重用相同的设备 ID 会导致其他 Matrix 客户端不信任该设备（它们将更改的身份密钥视为潜在安全漏洞）。新的访问令牌会获得一个新的设备 ID，没有陈旧密钥历史，因此其他客户端会立即信任它。
:::

## 代理模式（macOS 上的 E2EE）

Matrix E2EE 需要 `libolm`，它不能在 macOS ARM64（Apple Silicon）上编译。`hermes-agent[matrix]` 额外组件仅限于 Linux。如果你使用的是 macOS，代理模式让你可以在 Linux VM 上的 Docker 容器中运行 E2EE，而实际的 agent 在 macOS 上本地运行，可以完全访问你的本地文件、记忆和技能。

### 工作原理

```
macOS（主机）：
  └─ hermes gateway
       ├─ api_server 适配器 ← 在 0.0.0.0:8642 监听
       ├─ AIAgent ← 单一真相来源
       ├─ 会话、记忆、技能
       └─ 本地文件访问（Obsidian、项目等）

Linux VM（Docker）：
  └─ hermes gateway（代理模式）
       ├─ Matrix 适配器 ← E2EE 解密/加密
       └─ HTTP 转发 → macOS:8642/v1/chat/completions
           （无 LLM API 密钥、无 agent、无推理）
```

Docker 容器仅处理 Matrix 协议 + E2EE。当消息到达时，它解密并通过标准 HTTP 请求将文本转发到主机。主机运行 agent、调用工具、生成响应并流式传输回它。容器加密并发送响应到 Matrix。所有会话都是统一的 —— CLI、Matrix、Telegram 和任何其他平台共享相同的记忆和对话历史。

### 步骤 1：配置主机（macOS）

启用 API 服务器，以便主机接受来自 Docker 容器的传入请求。

将以下内容添加到 `~/.hermes/.env`：

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=your-secret-key-here
API_SERVER_HOST=0.0.0.0
```

- `API_SERVER_HOST=0.0.0.0` 绑定到所有接口，以便 Docker 容器可以访问它。
- `API_SERVER_KEY` 是非回环绑定的必需项。选择一个强随机字符串。
- API 服务器默认在端口 8642 上运行（如果需要，用 `API_SERVER_PORT` 更改）。

启动 gateway：

```bash
hermes gateway
```

你应该看到 API 服务器与你配置的任何其他平台一起启动。从 VM 验证它可达：

```bash
# 从 Linux VM
curl http://<mac-ip>:8642/health
```

### 步骤 2：配置 Docker 容器（Linux VM）

容器需要 Matrix 凭证和代理 URL。它**不需要** LLM API 密钥。

**`docker-compose.yml`：**

```yaml
services:
  hermes-matrix:
    build: .
    environment:
      # Matrix 凭证
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # 代理模式 — 转发到主机 agent
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

**`Dockerfile`：**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libolm-dev && rm -rf /var/lib/apt/lists/*
RUN pip install 'hermes-agent[matrix]'

CMD ["hermes", "gateway"]
```

这就是整个容器。没有 OpenRouter、Anthropic 或任何推理提供商的 API 密钥。

### 步骤 3：启动两者

1. 首先启动主机 gateway：
   ```bash
   hermes gateway
   ```

2. 启动 Docker 容器：
   ```bash
   docker compose up -d
   ```

3. 在加密 Matrix 房间中发送消息。容器解密它，转发到主机，并将响应流回。

### 配置参考

代理模式在**容器端**（薄 gateway）配置：

| 设置 | 描述 |
|---------|-------------|
| `GATEWAY_PROXY_URL` | 远程 Hermes API 服务器的 URL（例如 `http://192.168.1.100:8642`） |
| `GATEWAY_PROXY_KEY` | 认证的 Bearer 令牌（必须与主机上的 `API_SERVER_KEY` 匹配） |
| `gateway.proxy_url` | 与 `GATEWAY_PROXY_URL` 相同，但在 `config.yaml` 中 |

主机端需要：

| 设置 | 描述 |
|---------|-------------|
| `API_SERVER_ENABLED` | 设置为 `true` |
| `API_SERVER_KEY` | Bearer 令牌（与容器共享） |
| `API_SERVER_HOST` | 设置为 `0.0.0.0` 以进行网络访问 |
| `API_SERVER_PORT` | 端口号（默认：`8642`） |

### 适用于任何平台

代理模式不限于 Matrix。任何平台适配器都可以使用它 —— 在任何 gateway 实例上设置 `GATEWAY_PROXY_URL`，它会转发到远程 agent，而不是在本地运行一个。这对于平台适配器需要在与 agent 不同的环境中运行的任何部署都很有用（网络隔离、E2EE 要求、资源限制）。

:::tip
会话连续性通过 `X-Hermes-Session-Id` 标头维护。主机的 API 服务器按此 ID 跟踪会话，因此对话跨消息持久化，就像使用本地 agent 一样。
:::

:::note
**限制（v1）：** 不中继来自远程 agent 的工具进度消息 —— 用户只看到流式最终响应，而不是单个工具调用。危险命令批准提示在主机端处理，不中继到 Matrix 用户。这些可以在未来更新中解决。
:::

### 同步问题 / 机器人落后

**原因**：长时间运行的工具执行可能会延迟同步循环，或者 homeserver 很慢。

**修复**：同步循环在错误时自动每 5 秒重试。检查 Hermes 日志中与同步相关的警告。如果机器人持续落后，确保你的 homeserver 有足够的资源。

### 机器人离线

**原因**：Hermes gateway 未运行，或连接失败。

**修复**：检查 `hermes gateway` 是否正在运行。查看终端输出中的错误消息。常见问题：错误的 homeserver URL、过期的访问令牌、homeserver 不可达。

### "User not allowed" / 机器人忽略你

**原因**：你的用户 ID 不在 `MATRIX_ALLOWED_USERS` 中。

**修复**：将你的用户 ID 添加到 `~/.hermes/.env` 中的 `MATRIX_ALLOWED_USERS` 并重启 gateway。使用完整的 `@user:server` 格式。

## 安全

:::warning
始终设置 `MATRIX_ALLOWED_USERS` 以限制谁可以与机器人交互。如果没有它，gateway 默认拒绝所有用户，作为一种安全措施。只添加你信任的人的用户 ID —— 授权用户拥有对 agent 功能的完全访问权限，包括工具使用和系统访问。
:::

有关保护你的 Hermes Agent 部署的更多信息，请参阅 [安全指南](../security.md)。

## 注意事项

- **任何 homeserver**：可与 Synapse、Conduit、Dendrite、matrix.org 或任何符合规范的 Matrix homeserver 配合使用。不需要特定的 homeserver 软件。
- **联邦**：如果你在联邦 homeserver 上，机器人可以与来自其他服务器的用户通信 —— 只需将他们的完整 `@user:server` ID 添加到 `MATRIX_ALLOWED_USERS`。
- **自动加入**：机器人自动接受房间邀请并加入。它在加入后立即开始回复。
- **媒体支持**：Hermes 可以发送和接收图片、音频、视频和文件附件。媒体使用 Matrix 内容存储库 API 上传到你的 homeserver。
- **原生语音消息（MSC3245）**：Matrix 适配器自动标记带有 `org.matrix.msc3245.voice` 标志的外发语音消息。这意味着 TTS 响应和语音音频在 Element 和其他支持 MSC3245 的客户端中呈现为**原生语音气泡**，而不是作为通用音频文件附件。带有 MSC3245 标志的入站语音消息也会被正确识别并路由到语音转文本转录。无需配置 —— 这会自动工作。
