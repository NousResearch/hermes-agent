---
sidebar_position: 11
title: "飞书 / Lark"
description: "将 Hermes Agent 配置为飞书或 Lark 机器人"
---

# 飞书 / Lark 配置

Hermes Agent 可以作为全功能机器人与飞书和 Lark 集成。连接后，你可以在私聊或群聊中与 Agent 对话，将定时任务结果投递到主页聊天，通过网关正常收发文本、图片、音频和文件。

集成支持两种连接模式：

- `websocket` — **推荐**；Hermes 主动发起出站连接，无需公网 Webhook 地址
- `webhook` — 适用于需要通过 HTTP 将飞书/Lark 事件推送至网关的场景

## 使用行为

| 场景 | 行为 |
|------|------|
| 私聊 | Hermes 回复每一条消息 |
| 群聊 | 仅当机器人在聊天中被 @提及 时才回复 |
| 共享群聊 | 默认情况下，共享群聊中每个用户的会话历史是隔离的 |

共享群聊行为由 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望每个聊天共享一个对话时才设为 `false`。

## 步骤 1：创建飞书 / Lark 应用

### 推荐方式：扫码创建（一条命令）

```bash
hermes gateway setup
```

选择 **Feishu / Lark**，用飞书或 Lark 手机端扫描二维码。Hermes 会自动创建具有正确权限的机器人应用并保存凭据。

### 备选方式：手动配置

如果扫码创建不可用，向导会回退到手动输入：

1. 打开飞书或 Lark 开发者控制台：
   - 飞书：[https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark：[https://open.larksuite.com/](https://open.larksuite.com/)
2. 创建一个新应用。
3. 在 **凭证与基础信息** 中，复制 **App ID** 和 **App Secret**。
4. 为应用启用 **机器人** 能力。
5. 运行 `hermes gateway setup`，选择 **Feishu / Lark**，按提示输入凭据。

:::warning
请妥善保管 App Secret。任何持有它的人都可以冒充你的应用。
:::

### 配置权限

在飞书开发者控制台中，进入 **权限管理**，添加以下权限范围。你可以在权限页面批量导入。

**必需权限：**

| 权限范围 | 用途 |
|---------|------|
| `im:message` | 接收和读取消息 |
| `im:message:send_as_bot` | 以机器人身份发送消息 |
| `im:resource` | 访问用户发送的图片、文件和音频 |
| `im:chat` | 访问聊天/群组元数据 |
| `im:chat:readonly` | 读取聊天列表和成员信息 |

**推荐权限（完整功能）：**

| 权限范围 | 用途 |
|---------|------|
| `im:message.reactions:readonly` | 接收表情回应事件 |
| `admin:app.info:readonly` | 自动检测机器人身份用于 @提及 过滤 |
| `contact:user.id:readonly` | 解析用户 ID 用于许可名单匹配 |

### 配置事件

在 **事件与回调** 中：

1. 将连接模式设为 **长连接（WebSocket）**（推荐），或配置 Webhook URL
2. 在 **事件配置** 中，订阅：
   - `im.message.receive_v1` — 接收消息所必需

### 发布应用

配置完权限和事件后，前往 **版本管理与发布**，发布应用的新版本。权限在企业应用中需要管理员审批后才能生效。

## 步骤 2：选择连接模式

### 推荐：WebSocket 模式

当 Hermes 运行在你的笔记本电脑、工作站或私有服务器上时，使用 WebSocket 模式。无需公网 URL。官方 Lark SDK 会打开并维护一个持久的出站 WebSocket 连接，支持自动重连。

```bash
FEISHU_CONNECTION_MODE=websocket
```

**要求：** 必须安装 `websockets` Python 包。SDK 在内部处理连接生命周期、心跳和自动重连。

**工作原理：** 适配器在后台执行器线程中运行 Lark SDK 的 WebSocket 客户端。入站事件（消息、表情回应、卡片动作）被分发到主 asyncio 事件循环。断连时 SDK 会自动尝试重连。

:::tip 实盘经验

在生产服务器上运行时，需注意以下几点：

1. **systemd linger** — 服务用户账号会在 SSH 登出后停止所有用户级服务。必须启用 lingering：
   ```bash
   sudo loginctl enable-linger $USER
   ```

2. **防火墙** — WebSocket 是出站连接，不需要开放入站端口。但确保服务器能访问 `open.feishu.cn`。

3. **网关重启风险** — 重启网关会断开飞书 WebSocket，导致用户在当前对话中掉线。生产环境中应避免在对话活跃时段重启。

4. **静默掉线检测** — WebSocket 可能处于"连接但无响应"的状态。建议通过 cron 定期向机器人发送测试消息来验证连接健康。飞书 WebSocket `connected` 状态和实际消息可达性是两回事。
:::

### 可选：Webhook 模式

仅在 Hermes 已经部署在可公网访问的 HTTP 端点后才使用 Webhook 模式。

```bash
FEISHU_CONNECTION_MODE=webhook
```

Webhook 模式下，Hermes 通过 `aiohttp` 启动 HTTP 服务器，并在以下端点提供服务：

```text
/feishu/webhook
```

**要求：** 必须安装 `aiohttp` Python 包。

可自定义 Webhook 服务器的绑定地址和路径：

```bash
FEISHU_WEBHOOK_HOST=127.0.0.1  # 默认：127.0.0.1
FEISHU_WEBHOOK_PORT=8765        # 默认：8765
FEISHU_WEBHOOK_PATH=/feishu/webhook # 默认：/feishu/webhook
```

## 步骤 3：配置 Hermes

### 方案 A：交互式配置

```bash
hermes gateway setup
```

选择 **Feishu / Lark**，按提示填写。

### 方案 B：手动配置

在 `~/.hermes/.env` 中添加以下内容：

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=secret_xxx
FEISHU_DOMAIN=feishu
FEISHU_CONNECTION_MODE=websocket

# 可选但强烈推荐
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
FEISHU_HOME_CHANNEL=oc_xxx
```

`FEISHU_DOMAIN` 可选值：

- `feishu` 飞书中国版
- `lark` Lark 国际版

## 步骤 4：启动网关

```bash
hermes gateway
```

然后从飞书/Lark 向机器人发送消息，确认连接正常。

## 主页聊天

在飞书/Lark 聊天中使用 `/set-home` 将其标记为定时任务结果和跨平台通知的主页频道。

也可以预先配置：

```bash
FEISHU_HOME_CHANNEL=oc_xxx
```

## 安全

### 用户许可名单

生产环境建议设置飞书 Open ID 许可名单：

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

如果许可名单为空，任何能联系到机器人的人都可能使用它。群聊中，许可名单会在消息处理前检查发送者的 open_id。

### Webhook 加密密钥

Webhook 模式下，设置加密密钥以启用入站 Webhook 载荷的签名验证：

```bash
FEISHU_ENCRYPT_KEY=your-encrypt-key
```

此密钥可在飞书应用的 **事件订阅** 部分找到。设置后，适配器会使用签名算法验证每个 Webhook 请求：

```
SHA256(timestamp + nonce + encrypt_key + body)
```

:::tip
WebSocket 模式下，签名验证由 SDK 内部处理，`FEISHU_ENCRYPT_KEY` 是可选的。Webhook 模式下强烈建议在生产环境中启用。
:::

## 媒体支持

### 入站（接收）

适配器接收并缓存用户发送的以下媒体类型：

| 类型 | 扩展名 | 处理方式 |
|------|--------|---------|
| **图片** | .jpg, .jpeg, .png, .gif, .webp, .bmp | 通过飞书 API 下载并本地缓存 |
| **音频** | .ogg, .mp3, .wav, .m4a, .aac, .flac, .opus, .webm | 下载并缓存；小文本文件自动提取内容 |
| **视频** | .mp4, .mov, .avi, .mkv, .webm, .m4v, .3gp | 作为文档下载并缓存 |
| **文件** | .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx 等 | 作为文档下载并缓存 |

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| `lark-oapi not installed` | 安装 SDK：`pip install lark-oapi` |
| `websockets not installed` | 安装 websockets：`pip install websockets` |
| `FEISHU_APP_ID 或 FEISHU_APP_SECRET 未设置` | 设置环境变量或通过 `hermes gateway setup` 配置 |
| 另一个本地 Hermes 网关正在使用相同的 app_id | 同一 app_id 只能被一个 Hermes 实例使用。先停止另一个网关 |
| 机器人在群聊中不响应 | 确保机器人被 @提及，检查 `FEISHU_GROUP_POLICY`，验证发送者在许可名单中 |
| 机器人收到消息但不回复（网关状态正常） | 检查 LLM 提供商是否正常——`grep -i "402\|429\|Insufficient" ~/.hermes/logs/gateway.log`。常见原因：HTTP 402（余额不足）、429（速率限制）、5xx（提供商故障）。网关运行正常 + 飞书 WS 已连接 ≠ 模型可用 |
| 飞书手机端长消息渲染异常 | 飞书手机端无法可靠渲染超长 Markdown——内容会折叠或截断。解决方案：将聊天消息控制在 1500 字以内作为摘要，完整内容写入 `.md` 文件通过文件附件发送 |
| 点击审批按钮返回错误 200340 | 在飞书开发者控制台中启用 **交互式卡片** 能力并配置 **卡片请求 URL** |
| 图片/文件未被机器人收到 | 为飞书应用授予 `im:message` 和 `im:resource` 权限 |

## 生产环境多节点部署陷阱

基于 CTGC 团队 6 节点 A2A 拓扑的实际部署经验：

| 陷阱 | 表现 | 修复 |
|------|------|------|
| **gateway.platforms 为空** | 飞书 WebSocket 已连接但机器人不响应任何消息 | `config.yaml` 中 `gateway.platforms.feishu.enabled` 必须为 `true`。WebSocket 连接独立于平台适配器——连接成功不等于适配器已加载 |
| **`.env` 和 systemd EnvironmentFile 不一致** | 凭据存在于 `.env` 但网关找不到 | systemd 服务不会自动加载 `~/.hermes/.env`。在 override.conf 中显式设置 `Environment=` 或 `EnvironmentFile=` |
| **服务用户 home 目录不一致** | 网关启动但找不到配置文件 | 确保 systemd 单元的 `HOME` 环境变量与服务用户的 home 目录一致（例如 `/home/admin`，而非 `/root`） |
| **config set 重复键覆盖** | `hermes config set` 将列表值存为字符串 | 手动编辑 `config.yaml` 确保列表格式正确（尤其是 `platform_toolsets`） |
| **网关崩溃循环** | 网关不断重启 | 重置失败状态：`systemctl --user reset-failed hermes-gateway`，然后检查日志 |
| **国内网络 DNS 污染** | 安装依赖超时 | 使用代理：`export https_proxy=http://127.0.0.1:7890`；避免使用清华/阿里云等国内镜像 |
