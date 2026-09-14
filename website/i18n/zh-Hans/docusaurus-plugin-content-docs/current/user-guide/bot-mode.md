---
title: "Bot 模式"
description: "把你的 Hermes profile 变成一支具名的 Bot 团队——每个 Bot 都有自己的对话、角色、模型、记忆、技能和头像。Bot 可以运行例行任务、共享群聊，并互相发消息。"
---

# Bot 模式

**Bot 模式**把你的 [Hermes profile](./profiles.md) 变成一支具名 **Bot** 团队。每个 Bot 都有自己的角色、模型、记忆、技能和头像；Bot 之间可以运行周期性的例行任务、在群聊中共同商议，并直接互相发消息。花一次功夫搭建一个专精 Bot，它就永远留在那里，一键可达。

Bot 模式**内置于[桌面应用](./desktop.md)**中，**默认开启**——无需安装。它在左侧边栏中以 **Bots** 标签页的形式出现，紧挨着 Sessions；当 Bots 标签页处于激活状态时，一个 **Routines** 面板会停靠在对话旁边。

:::tip Bot 就是 profile
这里没有新概念需要学习：Bot **就是** 一个 Hermes profile——位于 `~/.hermes/profiles/<name>/` 下的独立配置、记忆、技能、凭据和聊天记录。Bot 模式只是这个基本单元之上的一层 UI，所以你在其中做的一切在 CLI 里同样可见：`hermes -p <bot> chat` 打开的是同一个 agent，Bot 的例行任务也会出现在 `hermes cron list` 中。没有核心补丁，没有后台守护进程，也不需要额外的存储。
:::

## Bots 面板

花名册中每个 agent profile 各占一行：头像、最新消息预览和时间戳。

- **点击一个 Bot** 进入它的对话——每个 Bot 都有一个唯一的、持久的**规范 Bot Chat** 对话，在该 Bot 诞生的那一刻就被创建（并置顶）。
- **Active now（当前活跃）**——花名册上方的一条状态栏显示当前正在工作的每个 Bot：占用 gateway 的 profile，以及过去 90 秒内有过写入的任何 Bot。每个小卡片都可打开对应 Bot 的对话。这条状态栏不会打乱花名册顺序，当整个团队空闲时会自动消失。
- **搜索**会随输入实时过滤花名册。
- **隐藏一个 Bot**——右键点击一行 → **Hide Bot**，把不常用的 Bot 从花名册和 Active-now 状态栏中移出。隐藏只影响显示：@提及依然能解析，群聊成员关系不受影响，例行任务照常运行。一旦至少有一个 Bot 被隐藏，面板标题栏会出现一个**眼睛图标**切换开关——点击它可就地以变暗的样式显示隐藏的 Bot，再右键 → **Unhide Bot** 即可恢复。隐藏的 Bot 不会弹出提示，但仍会静默累积未读消息，眼睛图标上会出现一个小红点提示有新动态。隐藏状态保存在该 Bot 的 profile 元数据中，因此它会跟随这个 Bot 出现在连接到同一后端的每台桌面设备上。

:::note 规范 Bot Chat 是一个永不重置的对话
在 Bot 的规范对话中输入 `/new`（或 `/reset`）本应把这段关系分叉成一个临时会话——而这正是 Bot 模式承诺永远不会发生的事。输入框会把它重新路由为 `/compact`：获得全新的工作上下文，但保留同一段对话。同一 profile 下的普通会话仍然拥有完整的 `/new` 自由。
:::

## 创建一个 Bot

在花名册中点击 **New Agent**。最快的路径只需三个字段——**Name**、**Title**、**Description**——几秒钟内 Bot 就会诞生，并在它新建的 Bot Chat 中发出第一条自我介绍消息。

一个 **Advanced** 折叠面板会展开完整的能力配置界面：

- **从现有 profile 克隆**——从另一个 Bot 的配置、技能、SOUL 和记忆起步，或者选择 **Fresh profile** 从零开始。
- **Create empty**——完全跳过内置技能，得到一个最小化的 profile。
- **模型与 provider 锁定**——为 Bot 指定专属模型。Hermes 支持的任意 provider/model 组合都可以使用，不同的 Bot 可以并排运行在不同的模型上。留空则继承自启动 profile。
- **自定义 SOUL.md**——Bot 的人格与常驻指令。
- **按技能、按工具集、按 MCP 服务器逐项启用**——精确勾选这个专精 Bot 需要的能力。
- **共享密钥**——默认情况下，新 Bot 与主 profile 共用一个 OAuth/token 池，这样凭据刷新不会互相失效。（较旧的 gateway 会改为复制凭据——依然可用，只是分叉了。）

### 选择它运行在哪台机器上（"Create on"）

当[设置 → Connections](./multi-connection-desktop.md) 中注册了不止一个连接时，New Agent 对话框会新增一个 **Create on** 选择器。选定一台设备后，profile 就会在**那台**机器的后端上创建——你的窗口不会切换 gateway。这个新 Bot 随后会以 Connections Bot 的身份出现在花名册中（当同名 Bot 存在于多台机器上时，带有 `@name-device` 形式的 handle），与它对话会路由到它自己所在的机器。

在只有一个连接的常见情况下，这个选择器会被隐藏，Bot 会在你当前连接的机器上创建——和以前的行为完全一致。

远程创建的注意事项：

- **克隆源**取自*目标*机器上的 profile（它的 `default`）——远程主机没有你本地的 profile 可供克隆。
- 实时的 Capabilities 标签页会锁定到目标机器的后端，因此你在创建过程中配置的技能、工具和 MCP 服务器都会落在这个 Bot 最终运行的机器上。（较旧的桌面版本会为远程目标回退到分阶段的 Skills/Tools/MCP 勾选列表；两者读取的都是目标机器的目录。）
- 取消对话框会丢弃草稿 profile，无论它是在哪台机器上创建的。

**Edit Profile**（右键点击一个 Bot）随时可以在这个实时 profile 上重新打开同一个界面：头像、标题、描述、模型锁定、技能、工具集、MCP 服务器，以及完整的 SOUL.md。

**Duplicate**（右键）会完整克隆一个 Bot——配置、技能、SOUL.md、记忆及外观。**Delete Profile** 会永久移除一个 Bot，需要经过与桌面 profile 菜单相同的破坏性操作确认；默认 profile 无法被删除。

## 头像

每个 Bot 都有一张脸：

- **Blob 脸**（默认）——从 Bot 名字派生出的确定性软体脸：同名同脸，永远不变。在 New Agent 中输入名字时，这张脸会实时跟随变化；点击 **Randomize** 重新生成，点击 **Lock face** 锁定你喜欢的那张脸（即使名字之后改变），或者固定六种轮廓之一（圆形、有机形、方形、圆突形、云朵形、太阳形），其余部分仍由名字派生。
- **几何脸**——经典的 7 种形状 × 10 种颜色组合，Bot 工作时眼睛会眨动扫视。
- **上传的图片**——任何你喜欢的图片。
- **AI 生成的肖像**——配置了图像后端时，在原地生成（这走的是标准的 `image.generate` RPC，本地和远程 gateway 均可使用）。
- **像素宠物**——来自 [petdex 图鉴](./features/pets.md) 的伙伴，在 Bot 工作时会在头像旁跳动。在终端中运行 `hermes pets` 即可浏览图鉴。

一个 Bot 的外观、标题和描述都保存在该 profile 的后端元数据中，因此同一个 Bot 在连接到该后端的每台桌面上看起来都一样。

## Routines（例行任务）

**Routines** 面板把周期性任务挂载到负责它的 Bot 上——"每天早上帮我总结收件箱"就紧挨着负责这件事的 Bot。这个面板只在 Bots 标签页激活时才停靠在对话旁边，切回 Sessions 时会自动让开（较旧的桌面版本会始终显示它）。一个结构化的调度选择器会构建调度规则（先选频率，再填入真正重要的细节），Advanced 字段则暴露原始的 Hermes 调度字符串。

Routines 本质上就是命名空间为 `[bot:<name>] <routine>` 的普通 [Hermes cron 任务](./features/cron.md)——它们同样会出现在 `hermes cron list` 和核心 Cron 页面中。运行结果会写入该 Bot 自己的对话历史，所以结果正好出现在你本来就会找这个 Bot 交流的地方。

## 群组与群聊

右键点击一个本地 Bot → **Manage groups**，即可把它加入或移出任意数量的群聊。可以单独挑选已有的群，也可以直接内联创建一个新群。本地成员关系保存在该 Bot 的后端同步 profile 元数据中，因此它会跟随这个 profile 出现在各个桌面上；带有一个旧版群组的旧 profile 仍能正常工作。Connections Bot 通过 New Group Chat 选择器加入群聊，并在房间的共享状态中保留来源标识。

**房间跟随的是你的 gateway，而不是某一台 Desktop。** 每个房间的近期记录、成员、图片和名称都会被镜像到你的 Desktop 所连接的**每一个** gateway 的共享 profile 元数据中，并带有按 gateway 划分的版本号，因此两台 Desktop 同时写入时会合并而不是互相覆盖。在另一台机器上打开 Hermes Desktop（局域网、Tailscale，任何地方都可以），连接到同一个 gateway，这个房间及其历史就会出现；仅有 gateway 的客户端也能看到它。房间携带一个持久的内部身份，因此重命名一个房间只会在各处改变它的显示名称，解散一个房间会在每个客户端上永久移除它——即便是当时离线的客户端也不例外——而重新创建一个同名群组会开启一个真正全新的房间。如果某个 gateway 挂掉或被移除，不会丢失任何数据：每台连接的 Desktop 都在本地保留完整的房间，并在重新连接时向任何 gateway 重新播种。（完整的编排日志留在每台 Desktop 的本地存储中；共享镜像只是一个有界的近期历史投影。）

群组是与 Bot 私信同属一个按活跃度排序的花名册中的独立行。一个 Bot 即便属于多个群组，也只占花名册中的一行私信；而每个群组都拥有自己独立的一行，显示成员数、最新消息预览、时间戳和"需要你"状态。

在任意群组行上点击 **Open chat**（2–6 个 Bot）会打开一个共享房间，整个群组在其中协作：

- 你的消息会触发最多**三轮串行**的成员发言。被 @提及 的 Bot 会回应（如果没有人被提及，则所有人都会回应）；每个 Bot 简短回复或选择跳过，当完整一轮都保持沉默时，房间就会安静下来。
- Bot 之间通过 `@name` 互相拉入对话，遇到真正需要判断的问题会用 `@user` 上报给你——这种情况下群组行会显示一个**需要你**的徽标。
- 硬性上限（每次发送 10 条消息，3 轮）防止房间失控。
- 每个成员都保留自己独立、持久的 `Group: <name>` 会话，因此房间上下文会像其他对话一样持续保存。
- **不是每个 Bot 都会回复每一条消息。** 是否发言由每个成员自己决定——一个 Bot 只有在有新内容可补充时才会回复，否则就跳过；@提及特定成员会把这一轮范围限定到他们身上。你可以预期被 @提及 的成员（或任何有话要说的成员）会发言，其余的保持安静。
- **房间可以跨越多台机器。** New Group Chat 选择器可以从任意已注册的连接中挑选 Bot；每个成员的发言都运行在它自己的机器上，在它自己那台机器的 `Group: <name>` 会话里。跨机器的成员在房间和其他成员的对话记录中都带有设备徽标（`dixie · Mac Mini`），消除歧义的 `@name-device` handle 在房间提及中同样有效——因此两台机器上同名的 agent 永远不会混淆。

## Bot 之间的消息

Bot 之间发消息会带有署名，你也可以从任意对话中把工作转交出去：

- **@提及**——在任意对话中输入 `@researcher have a look at this`，输入框的 `@` 自动补全会帮你选中正确的 Bot；发送时，这个提及会与当前花名册进行解析，当前活跃的 Bot 会被准确告知你指的是谁（profile、友好名称，以及跨连接 Bot 的设备信息）。随后该 Bot 会自己撰写消息，并通过 `message_agent` 发送出去——你的原文永远不会被原样转发，回复会带着那个 agent 的署名返回。一个邮箱地址或未知的 `@` 会原样透传。运行在其他已连接机器上的 Bot 同样可以这样触达（见下方*跨机器的 Bot*）：Desktop 会通过那个连接自己的 socket 转发消息。
- **重命名的 Bot 会同步保留标签**——给一个 Bot 起一个友好名字（它对话头部的铅笔图标，或 `hermes profile rename`），它就可以用这个名字被 @标记：一个标题为 *Research Buddy* 的 Bot 会响应 `@research-buddy`（以及 `@researchbuddy`），无论是在普通对话还是群聊中都是如此。输入框的 `@` 自动补全会提供重命名后的标签，同时输入旧的 profile 名称依然能匹配并继续生效。
- **私信**——每个 Bot Chat 都携带 `message_agent` 工具：一个 Bot 通过调用 `message_agent(target="researcher", message="…")` 给队友发消息。这个工具会根据当前花名册校验目标，自动加上发送方的 `Message from 🤖 <sender> (@<sender>):` 署名前缀，并投递到队友的规范 Bot Chat 中。投递是**发后即忘**的：发送方会收到一个确认，完成自己的这一轮，而回复会作为一条后台完成通知稍后到达。这条消息作为真正的参数传递（不经过 shell 解释——引号、`$(...)` 和反引号都会原样到达），而且 Bot 会自己撰写消息，而不是转发你的原话。队友花名册——每个 profile 的名称**和角色**（来自标题/描述）——是每个 Bot Chat 系统提示的一部分，因此 Bot 在选择接收方之前就知道谁负责什么。这个工具**只**存在于 Bot 模式管理的安装上的规范 Bot Chat 会话中；普通对话、群聊成员会话和 CLI 会话都看不到它。

后端会在构建提示词时自动把消息协议教给每个 Bot 的规范 Bot Chat 会话——包括队友从 CLI 无界面打开它的情况。只有规范 Bot Chat 会获得协议部分；你的普通会话和 SOUL.md 不受影响。这由 `config.yaml` 中的 `agent.bot_mode_protocol` 控制（默认：开启）：

```yaml
agent:
  bot_mode_protocol: true   # 向规范 Bot Chat 注入 Bot 间消息协议
```

:::note
Bot 间投递是按次调用的：接收方 Bot 会在它下一次运行时取走这条消息。中途打断一个正在对话的 Bot 目前还做不到，是未来的工作方向。
:::

### 失败的轮次会安全重试

一次失败的投递轮次最多重试一次，并且只在重试确实可能有帮助时才会重试。瞬时性失败（目标运行时离线、投递超时、provider 限流或服务端错误）会不加改动地重新运行同一个 Bot Chat 会话。上下文溢出失败同样会重新运行同一个会话——重试的这一轮会先通过标准的上下文压缩流程压缩超出阈值的记录，再调用模型，这样重试就能装进原本装不下的空间。认证、配额和配置类的失败永远不会自动重试：第二次尝试无法解决这些问题，只会白白消耗配额，因此这类失败会被立即上报。一次重试永远不会开启新会话——你的 Bot Chat 历史和上下文会保持完整。

### 投递失败时：带类型的原因码

一次失败的 Bot 轮次或中继投递会全程携带一个机器可读的 `reason` 代码，与人类可读的错误文本并列：目标 gateway 会对失败进行分类（`provider_auth_or_access`、`provider_quota_limit`、`provider_rate_limit`、`provider_server_error`、`context_overflow`、`missing_config`、`model_unavailable`、`runtime_offline`、`queued_expired`、`delivery_timeout`、`target_busy`、`unknown`），Desktop 负责转发，发送方 agent 的完成通知会在错误文本前带上 `[reason: <code>]` 标签。调用方 agent 可以据此分支处理——"需要重新登录"还是"稍后重试"——而不必解析 provider 的自然语言描述。Desktop 的"需要关注"徽标使用的是同一套代码。

### 跨已连接机器的消息（Desktop 中继）

你在**设置 → Connections** 中注册的每一个 gateway——本地、远程 URL、SSH、Hermes Cloud、docker——都是 Desktop 持续保持打开的一条常驻连接，Bot 模式会自动利用这些连接来发消息。无需额外配置：

- **花名册会自行同步。** 只要 Desktop 在运行，它就会定期告诉每个已连接的 gateway，*其他*连接上都有哪些 agent。每个 Bot Chat 的队友花名册随即会列出它们（"在其他已连接机器上的队友"），包含名称、角色以及所在的机器——当 agent 出现、消失或被重命名时，这份花名册也会刷新（能力版本）。
- **`message_agent` 可以直接触达它们。** 你笔记本上的 Bot 可以用 `message_agent(target="moxie", …)` 给云端 agent 发消息，和给本地队友发消息完全一样。如果同一个 handle 在多台机器上都存在，用 `target="moxie@<connection>"` 消除歧义（这个工具的报错信息会告诉 Bot 确切的写法）。投递走的是 Desktop：发送方的 gateway 把消息入队，Desktop 把它中继到目标连接自己的 gateway，目标 Bot 在它自己的规范 Bot Chat 中运行一轮，回复以本地私信同样使用的那种后台完成通知的形式返回给发送方。
- **Desktop 是信使。** 只要一台同时认识两个连接的 Desktop 在运行，跨连接投递就能工作（它持有 socket 和凭据——gateway 之间彼此看不到对方的认证信息）。如果 Desktop 在投递途中被关闭，发送方的 Bot 会被告知回复没有送达，而不是被无限期挂起。若需要完全不经过 Desktop 的、永远在线的机器对机器消息，请注册一个 peer（见下方 `hermes peer`）——这两条路径可以共存。

### Bot 发起的跨机器私信（`hermes peer`）

一台机器上的 Bot 可以在没有任何 Desktop 参与的情况下，给**另一台机器的 gateway** 上的 Bot 发消息。把对方的 gateway 注册为一个 *peer*（它的 API server URL + `API_SERVER_KEY`）：

```bash
hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>
hermes peer list
hermes peer dm spark < /tmp/dm.txt        # 消息内容来自一个文件(不经过 shell 解释)
hermes peer dm spark/researcher < /tmp/dm.txt   # 多路复用 peer 上的指定 profile
```

`hermes peer dm` 会通过该 peer 已有的 API server，把消息投递进远程 agent 的规范 Bot Chat，在那里运行一轮 agent，并把回复打印到 stdout——这正是本地 `hermes -p <bot> chat` 命令的跨机器对应版本。

一旦注册了 peer，教给每个 Bot Chat 的消息协议（`agent.bot_mode_protocol`）就会自动包含 peer 花名册，`message_agent` 也可以直接接受 peer 目标——`message_agent(target="spark/researcher", …)`，或用 `target="spark"` 指向该 peer 的主 agent——这样**你的 Bot 会自己了解到**其他机器上存在队友，以及如何触达它们。注册或移除一个 peer 会在下一条消息时刷新每个 Bot Chat 的协议（能力版本）。

前提条件：peer 所在机器运行 `api_server` gateway 平台，并配置了强密码的 `API_SERVER_KEY`；能否触达取决于你的网络（局域网、Tailscale、VPN）。这个 key 是一项凭据，保存在 `~/.hermes/.env` 中的 `HERMES_PEER_<NAME>_KEY` 下；peer 的名称/URL 保存在 `config.yaml` 的 `bot_peers` 下。

## 跨机器的 Bot

当你在**设置 → Connections** 中注册了多个后端——本地运行时、远程 gateway、SSH 主机、Hermes Cloud 实例——花名册会持续显示来自**每一个**已连接来源的 Bot：SSH 来源会在不在远程主机上启动任何进程的前提下被盘点，暂时无法触达的机器会保留它们最后已知的行，而不是直接消失。当同一个 profile 名称存在于多个来源时，handle 会以 `@name-device` 的形式消除歧义（例如 `@research-homelab`）。一个 Bot 的对话、会话、记忆和例行任务都存放在拥有该 profile 的那台机器上。

点击一个 Connections Bot **不会**把你的窗口切换到那台机器上——留在你当前的对话里 @提及 它、把它安排进某个群聊，或者用 **Create on** 选择器直接在它所在的机器上创建新的 agent。云端和本地 agent 就这样共用同一个花名册：注册你的 Hermes Cloud 实例和你的桌面（比如通过 Tailscale 或 SSH），它们的 Bot 就能互相发消息、共处同一个房间，每个 agent 的工作都运行在自己的机器上。跨这些机器的 Bot 间私信会自动走 Desktop 中继（见上文*跨已连接机器的消息*）。

完整的多连接指南请参阅[将 Desktop 连接到多个 Hermes 实例](./multi-connection-desktop.md)。

## 关闭它

Bot 模式是一个内置的桌面插件。在**设置 → Plugins → Bots** 中把它关闭——花名册、Routines 面板和输入框中间件会实时注销，无需重启。无论开关状态如何，你的 profile、会话和 cron 任务都不受影响；Bot 模式从不拥有你的数据，它只是负责渲染。

还有一个偏好设置可以把规范 Bot Chat 从常规侧边栏会话列表中隐藏，让它们只出现在 Bots 面板里。（这依赖核心的隐藏会话标记；在较旧的 gateway 上这些对话会照常保持可见。）

## 与 CLI 的对应关系

因为 Bot 本质上就是 profile，所以每个操作都有对应的终端命令：

| 在 Bot 模式中 | 在终端中 |
| --- | --- |
| 与一个 Bot 对话 | `hermes -p <bot> chat` |
| 一个 Bot 的文件、技能、记忆 | `~/.hermes/profiles/<bot>/` |
| 例行任务 | `hermes cron list`（任务名为 `[bot:<name>] …`） |
| 创建 / 查看 profile | `hermes profile create`、`hermes profile list` |

底层原理请参阅 [Profiles](./profiles.md)，完整 CLI 参考请参阅 [Profile Commands](../reference/profile-commands.md)。
