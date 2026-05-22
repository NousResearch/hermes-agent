---
sidebar_position: 1
title: "Telegram"
description: "将 Hermes Agent 设置为 Telegram 机器人"
---

# Telegram 设置

Hermes Agent 可作为功能齐全的 Telegram 机器人集成。一旦连接，你可以在任何设备上与代理聊天、发送会被自动转写的语音备忘、接收定时任务结果，并在群组中使用该代理。该集成基于 [python-telegram-bot](https://python-telegram-bot.org/)，支持文本、语音、图片和文件附件。

## 第 1 步：通过 BotFather 创建机器人

每个 Telegram 机器人都需要由 [@BotFather](https://t.me/BotFather) 发放的 API token。

1. 打开 Telegram 并搜索 **@BotFather**，或访问 [t.me/BotFather](https://t.me/BotFather)
2. 发送 `/newbot`
3. 选择一个**显示名称**（例如 "Hermes Agent"）— 任意可读名称
4. 选择一个**用户名**— 必须唯一并以 `bot` 结尾（例如 `my_hermes_bot`）
5. BotFather 会回复你的 **API token**，形式类似：

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
请妥善保管你的 bot token。任何持有该 token 的人都可以控制你的机器人。如有泄露，请立即通过 BotFather 撤销。 
:::

## 第 2 步：自定义机器人（可选）

通过 BotFather 的若干命令可以改善用户体验：

| 命令 | 作用 |
|------|------|
| `/setdescription` | 在用户开始聊天前显示的“这个机器人可以做什么”文本 |
| `/setabouttext` | 机器人资料页上的简短说明 |
| `/setuserpic` | 上传机器人头像 |
| `/setcommands` | 定义命令菜单（聊天中的 `/` 按钮） |
| `/setprivacy` | 控制机器人在群组中的隐私模式（见第 3 步） |

:::tip
对于 `/setcommands`，一个实用的初始命令集：

```
help - 显示帮助信息
new - 开始新会话
sethome - 将此聊天设为主频道
```
:::

## 第 3 步：隐私模式（群组场景关键）

Telegram 机器人的**隐私模式**默认是 **开启** 的。这通常是群组中机器人行为出现困惑的最常见原因。

**开启隐私模式** 时，机器人只能看到：
- 以 `/` 开头的命令消息
- 直接回复机器人的消息
- 服务消息（成员加入/离开、置顶消息等）
- 机器人作为管理员时的频道消息

**关闭隐私模式** 时，机器人能接收群组中的所有消息。

### 如何关闭隐私模式

1. 私信 **@BotFather**
2. 发送 `/mybots`
3. 选择你的机器人
4. 进入 **Bot Settings → Group Privacy → Turn off**

:::warning
在更改隐私设置后，你必须**将机器人从群组中移除并重新加入**。Telegram 在机器人加入群组时缓存隐私状态，只有重新加入后设置才会生效。
:::

:::tip
另一种替代方案是将机器人提升为**群组管理员**。管理员状态能让机器人无视隐私模式，始终收到所有消息，从而避免切换隐私设置的需要。
:::

## 第 4 步：查找你的用户 ID

Hermes Agent 使用数值 Telegram 用户 ID 来控制访问权限。用户 ID 不是用户名，而是形如 `123456789` 的数字。

**方法 1（推荐）**：私信 [@userinfobot](https://t.me/userinfobot) — 它会即时回复你的用户 ID。

**方法 2**：私信 [@get_id_bot](https://t.me/get_id_bot)。

保存该数字，在下一步配置时使用。

## 第 5 步：配置 Hermes

### 选项 A：交互式设置（推荐）

```bash
hermes gateway setup
```

选择 **Telegram**，按向导填写 bot token 和允许的用户 ID，向导会为你写好配置。

### 选项 B：手动配置

将以下内容加入 `~/.hermes/.env`：

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789    # 多个用户逗号分隔
```

### 启动网关

```bash
hermes gateway
```

机器人应在数秒内上线。向其发送消息以验证连接。

## 从 Docker 后端发送生成文件

如果终端后端是 `docker`，请注意 Telegram 附件由**网关进程**发送，而不是容器内部。因此最终的 `MEDIA:/...` 路径必须是宿主可读的。

常见问题：

- agent 在 Docker 内部写入 `/workspace/report.txt`
- 模型输出 `MEDIA:/workspace/report.txt`
- Telegram 发送失败，因为该路径仅在容器内存在，而宿主不可见

推荐做法：

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/.hermes/cache/documents:/output"
```

然后：在容器内将文件写到 `/output/...`，并在回复中使用宿主可见路径，例如：`MEDIA:/home/user/.hermes/cache/documents/report.txt`。

### 支持的 `MEDIA:` 文件扩展名

表格省略（与英文一致）

## Webhook 模式

Webhook 模式适合把 Telegram 作为事件入口，而不是只作为聊天终端。启用后，Hermes 会通过 webhook 接收更新，减少轮询开销，并让机器人在托管环境里更容易稳定运行。实际部署时，通常需要同时配置公开可达的回调地址、bot token、允许的用户列表，以及反向代理或 TLS 终止层。

## 代理支持与 Home Channel、语音消息、群组使用等说明

这一部分主要说明三类常见场景：

- **代理支持**：当你需要经过企业代理或受限网络访问 Telegram 时，确保网关进程的出站连接能走代理，并验证 webhook 回调仍可被外部访问。
- **Home Channel**：适合作为单一主对话入口，把某个群组或私聊固定为默认落点，减少会话切换成本。
- **语音消息和群组**：语音消息通常会自动转写后再进入 Hermes 会话；群组中则要特别注意隐私模式、管理员权限和消息可见性。

## 斜杠命令访问控制 {#slash-command-access-control}

Telegram 的斜杠命令可见性与执行权限都受网关授权配置影响。建议在部署后先用受限账号验证命令边界，再逐步开放。
