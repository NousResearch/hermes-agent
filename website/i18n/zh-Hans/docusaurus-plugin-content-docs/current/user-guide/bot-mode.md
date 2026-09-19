---
title: "Bot 模式"
description: "把你的 Hermes profile 变成一支具名的 Bot 团队——每个 Bot 都有自己的对话、角色、模型、记忆、技能和头像。Bot 可以运行例行任务、共享群聊，并互相发消息。"
---

# Bot 模式

**Bot 模式**把你的 [Hermes profile](./profiles.md) 变成一支具名 **Bot** 团队。每个 Bot 都有自己的角色、模型、记忆、技能和头像；Bot 之间可以运行周期性的例行任务、在群聊中共同商议，并直接互相发消息。花一次功夫搭建一个专精 Bot，它就永远留在那里，一键可达。

Bot 模式**内置于[桌面应用](./desktop)**中，**默认开启**——无需安装。它在左侧边栏中以 **Bots** 标签页的形式出现，紧挨着 Sessions；当 Bots 标签页处于激活状态时，一个 **Routines** 面板会停靠在对话旁边。

:::tip Bot 就是 profile
这里没有新概念需要学习：Bot **就是** 一个 Hermes profile——位于 `~/.hermes/profiles/<name>/` 下的独立配置、记忆、技能、凭据和聊天记录。Bot 模式只是这个基本单元之上的一层 UI，所以你在其中做的一切在 CLI 里同样可见：`hermes -p <bot> chat` 打开的是同一个 agent，Bot 的例行任务也会出现在 `hermes cron list` 中。没有核心补丁，没有后台守护进程，也不需要额外的存储。
:::

## Bots 面板

花名册中每个 agent profile 各占一行：头像、最新消息预览和时间戳。

- **点击一个 Bot** 进入它的对话——每个 Bot 都有一个唯一的、持久的**规范 Bot Chat** 对话，在该 Bot 诞生的那一刻就被创建（并置顶）。点击一行总是打开那个 Bot Chat（也就是该行预览的那段对话），即便你为这个 Bot 还开着其他标签页；那些标签页会保留在它旁边。在标签栏中，Bot Chat 以 Bot 的名字作为标题，因此两个同时打开的 Bot 一眼就能分清。
- **Active now（当前活跃）**——花名册的活跃过滤器包含当前聚焦的实时轮次的所有者、过去 90 秒内有过写入的 Bot，以及最近有 worker 心跳的 Bot。仅仅连接着某个 gateway 并不代表某个 Bot 正在工作。
- **搜索**会随输入实时过滤花名册。
- **隐藏一个 Bot**——右键点击一行 → **Hide Bot**，把不常用的 Bot 从花名册和 Active-now 状态栏中移出。隐藏只影响显示：@提及依然能解析，群聊成员关系不受影响，例行任务照常运行。一旦至少有一个 Bot 被隐藏，面板标题栏会出现一个**眼睛图标**切换开关——点击它可就地以变暗的样式显示隐藏的 Bot，再右键 → **Unhide Bot** 即可恢复。隐藏的 Bot 不会弹出提示，但仍会静默累积未读消息，眼睛图标上会出现一个小红点提示有新动态。隐藏状态保存在该 Bot 的 profile 元数据中，因此它会跟随这个 Bot 出现在连接到同一后端的每台桌面设备上。

:::note 规范 Bot Chat 是一个永不重置的对话
在 Bot 的规范对话中输入 `/new`（或 `/reset`）本应把这段关系分叉成一个临时会话——而这正是 Bot 模式承诺永远不会发生的事。输入框会把它重新路由为 `/compact`：获得全新的工作上下文，但保留同一段对话。同一 profile 下的普通会话仍然拥有完整的 `/new` 自由。

从侧边栏归档一个 Bot Chat 会让它退役：下一次点击这个 Bot 会开启一段全新的对话，它将成为新的规范 Bot Chat。退役的对话保持归档并隐藏——它的历史仍保存在数据库中，但不再能从这个 Bot 或归档视图中打开。自动的空闲归档清理（`sessions.auto_archive`）永远不会让 Bot Chat 退役；只有显式归档才会。
:::

### 把 Bot 整理进分区

分区（section）是你自己创建的文件夹——**Clients**、**Team**，怎么合适怎么来——作为自动按 gateway 分组之外的第二个维度。没有创建任何分区时，花名册仍是原来那份普通列表。

- **创建分区**：从面板的 **+** 菜单 → **New section**，或右键点击一个 Bot → **Move to section** → **New section…**（创建的同时就把这个 Bot 归入其中）。
- **归档一个 Bot**：把它的行拖到某个分区上——悬停时目标会高亮，按 **Esc** 取消拖动——或者右键 → **Move to section** 再选择一个。**Remove from section** 会把它放回 **Unassigned**。
- **重命名、重新排序或删除**分区：通过分区标题的右键菜单（或悬停时出现的 **⋯**）；双击标题即可重命名。标题可以像 gateway 标题那样折叠。
- **删除分区永远不会删除 Bot**——它们会回到 **Unassigned**，弹出的提示会提供 **Undo**。不会要求确认。

成员关系保存在每个 Bot 的 profile 元数据（`ui_meta`）中，因此一个 Bot 的分区会跟随它出现在连接到同一后端的每台桌面上。当花名册显示多个 gateway 时，分区会嵌套在各个 gateway 的分组之内。

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

当[设置 → Connections](./multi-connection-desktop) 中注册了不止一个连接时，New Agent 对话框会新增一个 **Create on** 选择器。选定一台设备后，profile 就会在**那台**机器的后端上创建——你的窗口不会切换 gateway。这个新 Bot 随后会以 Connections Bot 的身份出现在花名册中（当同名 Bot 存在于多台机器上时，带有 `@name-device` 形式的 handle），与它对话会路由到它自己所在的机器。

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
- **几何脸**——经典的 7 种形状 × 10 种颜色组合。在聚焦的实时轮次进行期间，拥有该轮次的 Bot 会侧身向上看，并显示三个动态的小点，轮次结束后再缓缓回到空闲姿态。所有权包含连接信息，因此不同 gateway 上的同名 Bot 不会借用彼此的姿态。后台 worker 保留原有的工作动画；照片、Blob 脸和印记（sigil）保留各自的渲染方式。
- **上传的图片**——任何你喜欢的图片。
- **AI 生成的肖像**——配置了图像后端时，在原地生成（这走的是标准的 `image.generate` RPC，本地和远程 gateway 均可使用）。
- **像素宠物**——来自 [petdex 图鉴](./features/pets) 的伙伴，在 Bot 工作时会在头像旁跳动。在终端中运行 `hermes pets` 即可浏览图鉴。

一个 Bot 的外观、标题和描述都保存在该 profile 的后端元数据中，因此同一个 Bot 在连接到该后端的每台桌面上看起来都一样。

## Routines（例行任务）

**Routines** 面板把周期性任务挂载到负责它的 Bot 上——"每天早上帮我总结收件箱"就紧挨着负责这件事的 Bot。这个面板只在 Bots 标签页激活时才停靠在对话旁边，切回 Sessions 时会自动让开（较旧的桌面版本会始终显示它）。一个结构化的调度选择器会构建调度规则（先选频率，再填入真正重要的细节），Advanced 字段则暴露原始的 Hermes 调度字符串。

Routines 本质上就是命名空间为 `[bot:<name>] <routine>` 的普通 [Hermes cron 任务](./features/cron.md)——它们同样会出现在 `hermes cron list` 和核心 Cron 页面中。运行结果会写入该 Bot 自己的对话历史，所以结果正好出现在你本来就会找这个 Bot 交流的地方。

## 群组与群聊

在某个成员的轮次进行期间发送的消息（包括同一话题线程中的回复）会排在当前房间驱动之后等待。它们不会打断那个成员，也不会把未读消息标记为已读。Stop 会取消排队中的后续轮次，并让成员保持暂停直到恢复。房间安静下来后，会在前台等待结束后继续观察超时的成员 21 分钟；这个观察窗口不会延长轮次本身。房间平息后，未解决的成员失败仍会显示在折叠的 Activity 摘要中。展开 Activity 可查看轮次顺序；重新 @ 该成员即可重试。结果不明确的提交失败不会被自动重新提交。

右键点击一个本地 Bot → **Manage groups**，即可把它加入或移出任意数量的群聊。可以单独挑选已有的群，也可以直接内联创建一个新群。本地成员关系保存在该 Bot 的后端同步 profile 元数据中，因此它会跟随这个 profile 出现在各个桌面上；带有一个旧版群组的旧 profile 仍能正常工作。Connections Bot 通过 New Group Chat 选择器加入群聊，并在房间的共享状态中保留来源标识。

**房间跟随的是你的 gateway，而不是某一台 Desktop。** 每个房间的近期记录、成员、图片和名称都会被镜像到你的 Desktop 所连接的**每一个** gateway 的共享 profile 元数据中，并带有按 gateway 划分的版本号，因此两台 Desktop 同时写入时会合并而不是互相覆盖。在另一台机器上打开 Hermes Desktop（局域网、Tailscale，任何地方都可以），连接到同一个 gateway，这个房间及其历史就会出现；仅有 gateway 的客户端也能看到它。房间携带一个持久的内部身份，因此重命名一个房间只会在各处改变它的显示名称，解散一个房间会在每个客户端上永久移除它——即便是当时离线的客户端也不例外——而重新创建一个同名群组会开启一个真正全新的房间。如果某个 gateway 挂掉或被移除，不会丢失任何数据：每台连接的 Desktop 都在本地保留完整的房间，并在重新连接时向任何 gateway 重新播种。（完整的编排日志留在每台 Desktop 的本地存储中；共享镜像只是一个有界的近期历史投影。）

群组的身份在创建时和创建后都可以编辑：

- **New Group Chat** 允许你在名称之外设置一张可选的**房间图片**——从设备上传一张，或（配置了图像后端时）生成一张，使用的是与 Bot 头像相同的流程。这张图片会出现在花名册行中（取代默认的群组图标），并在房间头部置于标题之前。
- 房间头部的**齿轮**按钮会打开 **Group settings**，你可以在这里随时**重命名**群组，或设置、替换、移除它的图片。重命名会把一切一并带走——房间日志、每个成员的房间会话、成员关系和图片——不会丢失任何历史。重命名为已被占用的名称会被拒绝，而不是悄悄加上后缀。

群组是与 Bot 私信同属一个按活跃度排序的花名册中的独立行。一个 Bot 即便属于多个群组，也只占花名册中的一行私信；而每个群组都拥有自己独立的一行，显示成员数、最新消息预览、时间戳和"需要你"状态。

使用房间旁边的 **Move up** 和 **Move down** 箭头可以选择它在各个房间中的位置。在第一次移动之前，原有的"置顶优先、按近期活跃度"排序保持不变。移动之后，房间顺序会保存在这台 Desktop 上并在重新加载后保留；新房间会排在各自置顶或未置顶分段内已显式排序的房间之后。移动不能跨越置顶边界，过滤也不会把隐藏的房间从已保存的顺序中丢弃。这些控件重新排序的是真正的 Group Chat 房间，而不是用户创建的 Bot 文件夹，也不会改变成员关系或 gateway 归属。

在任意群组行上点击 **Open chat**（2–6 个 Bot）会打开一个共享房间，整个群组在其中协作：

- **一条可见的对话。** 公开消息和每个成员的回复按到达顺序保持可读，并带有发言者的名字和时间戳。开启另一个话题不会折叠之前的回复。**Reply in thread** 会在不打乱房间顺序的前提下延续那个话题；**Activity** 是一个辅助的状态视图，而不是消息的替代品。私有的 Bot Chat 保持独立。
- 你的消息会触发最多**三轮串行**的成员发言。被 @提及 的 Bot 会回应（如果没有人被提及，则所有人都会回应）；每个 Bot 简短回复或选择跳过，当完整一轮都保持沉默时，房间就会安静下来。
- 队友可以用 `@hermes` 把工作转交给主 Bot，在较早保存的房间中同样有效；其他 gateway 上的 Bot 会保留带设备限定的标签（例如 `@default-vera`）。
- Bot 之间通过 `@name` 互相拉入对话，遇到真正需要判断的问题会用 `@user` 上报给你——这种情况下群组行会显示一个**需要你**的徽标。待处理的提问和命令审批同样会点亮这个徽标；解决最后一个提示只会清除提示类的关注，而不会清除独立的提及。提示会跟随重命名后的房间，而解散房间会让它们退役，即使某个成员正在进行中的轮询稍后才到达。
- 硬性上限（每次发送 10 条消息，3 轮）防止房间失控。
- 每个成员都保留自己独立、持久的房间会话，因此房间上下文会像其他对话一样持续保存。
- **不是每个 Bot 都会回复每一条消息。** 是否发言由每个成员自己决定——一个 Bot 只有在有新内容可补充时才会回复，否则就跳过；@提及特定成员会把这一轮范围限定到他们身上。你可以预期被 @提及 的成员（或任何有话要说的成员）会发言，其余的保持安静。
- **关闭 Desktop 后房间仍会继续运行。** 当一个房间的所有成员都位于同一个 gateway 上时，该 gateway 会通过一个持久的驱动器来负责轮次调度：关闭 Hermes Desktop（或失去它的连接）不会让讨论中途停止，Desktop 重新连接时只需从房间日志中补上进度即可。适用这种情况时，gateway 上的 `groups.capabilities` 会报告 `driver: true`。成员跨多台机器的房间则不同：每个成员的轮次运行在它自己的 gateway 上，*Bot 之间的消息*一节中描述的跨连接信使机制仍然适用于它们。
- **房间可以跨越多台机器。** New Group Chat 选择器可以从任意已注册的连接中挑选 Bot；每个成员的发言都运行在它自己的机器上，在它自己那台机器的房间会话里。跨机器的成员在房间和其他成员的对话记录中都带有设备徽标（`dixie · Mac Mini`），消除歧义的 `@name-device` handle 在房间提及中同样有效——因此两台机器上同名的 agent 永远不会混淆。
- **插件可以观察成员的工作。** 持久的房间日志会记录 `turn.started` 和 `turn.settled`；成员在这两者之间做的事情（工具、审批、流式文本）会通过 [`on_room_member_activity`](/user-guide/features/hooks#on_room_member_activity) 钩子投射给插件，并附带房间、成员和轮次坐标，因此社区客户端无需读取 Hermes 内部实现，就能在 Group Chat 之上构建工具卡片和实时成员状态。

## Bot 之间的消息

Bot 之间发消息会带有署名，你也可以从任意对话中把工作转交出去：

- **@提及**——在任意对话中输入 `@researcher have a look at this`，输入框的 `@` 自动补全会帮你选中正确的 Bot；发送时，这个提及会与当前花名册进行解析，当前活跃的 Bot 会被准确告知你指的是谁（profile、友好名称，以及跨连接 Bot 的设备信息）。随后该 Bot 会自己撰写消息，并通过 `message_agent` 发送出去——你的原文永远不会被原样转发，回复会带着那个 agent 的署名返回。一个邮箱地址或未知的 `@` 会原样透传。运行在其他已连接机器上的 Bot 同样可以这样触达（见下方*跨机器的 Bot*）：Desktop 会通过那个连接自己的 socket 转发消息。
- **重命名的 Bot 会同步保留标签**——给一个 Bot 起一个友好名字（它对话头部的铅笔图标，或 `hermes profile rename`），它就可以用这个名字被 @标记：一个标题为 *Research Buddy* 的 Bot 会响应 `@research-buddy`（以及 `@researchbuddy`），无论是在普通对话还是群聊中都是如此。输入框的 `@` 自动补全会提供重命名后的标签，同时输入旧的 profile 名称依然能匹配并继续生效。
- **私信**——每个 Bot Chat 都携带 `message_agent` 工具：一个 Bot 通过调用 `message_agent(target="researcher", message="…")` 给队友发消息。这个工具会根据当前花名册校验目标，自动加上发送方的 `Message from 🤖 <sender> (@<sender>):` 署名前缀，并投递到队友的规范 Bot Chat 中。投递是**发后即忘**的：发送方会收到一个确认，完成自己的这一轮，而回复会作为一条后台完成通知稍后到达。这条消息作为真正的参数传递（不经过 shell 解释——引号、`$(...)` 和反引号都会原样到达），而且 Bot 会自己撰写消息，而不是转发你的原话。队友花名册——每个 profile 的名称**和角色**（来自标题/描述）——是每个 Bot Chat 系统提示的一部分，因此 Bot 在选择接收方之前就知道谁负责什么。这个工具**只**存在于 Bot 模式管理的安装上的规范 Bot Chat 会话中；普通对话、群聊成员会话和 CLI 会话都看不到它。

本地消息同样能送达一个在 Desktop 或 TUI 中保持打开的 Bot Chat。接收方后端保留所有权：它在现有的通知轮询器上读取持久化的入站消息，空闲时立即接纳，否则等到正在运行的轮次和已排队的人类提示完成后再处理。`queued` 确认表示消息已被持久接纳，**而不是**回复已完成。目标 profile 会在 `runtime/bot_live_delivery/` 下保留投递 ID 和回执；`settled` 表示已完成。崩溃或被取消的导入轮次不会被自动重放，绑定到已离开的所有者的待处理工作会保留下来供检查，而不是被悄悄重新运行。不要重发结果未知的投递。不具备实时投递能力的较旧后端仍会返回原有的所有权拒绝；升级后请重启那个后端。

- **保持沉默**——没有新内容可补充的 Bot 可以用某个[有意沉默标记](./messaging/index.md#intentional-silence-tokens)（`[SILENT]`、`NO_REPLY`、……）来结束一轮。Bot Chat 会把这一轮保留在记录中但不渲染任何内容，给它发消息的队友会收到一个空回复而不是这个标记。失败的轮次以及仅仅提到某个标记的文字会原样显示。

后端会在构建提示词时自动把消息协议教给每个 Bot 的规范 Bot Chat 会话——包括队友从 CLI 无界面打开它的情况。只有规范 Bot Chat 会获得协议部分；你的普通会话和 SOUL.md 不受影响。这由 `config.yaml` 中的 `agent.bot_mode_protocol` 控制（默认：开启）：

```yaml
agent:
  bot_mode_protocol: true   # 向规范 Bot Chat 注入 Bot 间消息协议
```

:::note
Bot 间投递是按次调用的：接收方 Bot 会在它下一次运行时取走这条消息。中途打断一个正在对话的 Bot 目前还做不到，是未来的工作方向。
:::

### 失败的轮次会安全重试

本地一次性投递会把活跃会话的拒绝代码与人类可读的消息分开保留。`SESSION_NOT_OWNED` 会产生 `target_busy`；无法读取的协调注册表不会被误标为另一个所有者。没有代码标记的较旧本地 CLI 仍使用原有的拒绝措辞。

一次失败的投递轮次最多重试一次，并且只在重试确实可能有帮助时才会重试。瞬时性失败（目标运行时离线、投递超时、provider 限流或服务端错误）会不加改动地重新运行同一个 Bot Chat 会话。上下文溢出失败同样会重新运行同一个会话——重试的这一轮会先通过标准的上下文压缩流程压缩超出阈值的记录，再调用模型，这样重试就能装进原本装不下的空间。认证、配额和配置类的失败永远不会自动重试：第二次尝试无法解决这些问题，只会白白消耗配额，因此这类失败会被立即上报。一次重试永远不会开启新会话——你的 Bot Chat 历史和上下文会保持完整。

当目标没有活跃的 Desktop 或 TUI 所有者时，本地投递会通过一次安静的 CLI 轮次打开该 profile 的规范 Bot Chat。这条传输路径优先使用与发送方运行时的 Python 解释器并列的 Hermes 入口，因此只要这个同级入口存在，某个服务 `PATH` 上不相关或较旧的 `hermes` 就不会抢先。`--in ~` 选择工作目录；显式的 Bot Chat 标题会在目标 profile 的会话数据库中解析。

### 投递失败时：带类型的原因码

一次失败的 Bot 轮次或中继投递会全程携带一个机器可读的 `reason` 代码，与人类可读的错误文本并列：目标 gateway 会对失败进行分类（`provider_auth_or_access`、`provider_quota_limit`、`provider_rate_limit`、`provider_server_error`、`context_overflow`、`missing_config`、`model_unavailable`、`runtime_offline`、`queued_expired`、`delivery_timeout`、`target_busy`、`unknown`），Desktop 负责转发，发送方 agent 的完成通知会在错误文本前带上 `[reason: <code>]` 标签。调用方 agent 可以据此分支处理——"需要重新登录"还是"稍后重试"——而不必解析 provider 的自然语言描述。Desktop 的"需要关注"徽标使用的是同一套代码。

### 跨已连接机器的消息（Desktop 中继）

你在**设置 → Connections** 中注册的每一个 gateway——本地、远程 URL、SSH、Hermes Cloud、docker——都是 Desktop 持续保持打开的一条常驻连接，Bot 模式会自动利用这些连接来发消息。无需额外配置：

- **花名册会自行同步。** 只要 Desktop 在运行，它就会定期告诉每个已连接的 gateway，*其他*连接上都有哪些 agent。每个 Bot Chat 的队友花名册随即会列出它们（"在其他已连接机器上的队友"），包含名称、角色以及所在的机器——当 agent 出现、消失或被重命名时，这份花名册也会刷新（能力版本）。
- **`message_agent` 可以直接触达它们。** 你笔记本上的 Bot 可以用 `message_agent(target="moxie", …)` 给云端 agent 发消息，和给本地队友发消息完全一样。如果同一个 handle 在多台机器上都存在，用 `target="moxie@<connection>"` 消除歧义（这个工具的报错信息会告诉 Bot 确切的写法）。投递走的是 Desktop：发送方的 gateway 把消息入队，Desktop 把它中继到目标连接自己的 gateway，目标 Bot 在它自己的规范 Bot Chat 中运行一轮，回复以本地私信同样使用的那种后台完成通知的形式返回给发送方。发给不同 Bot 的消息会并行投递，因此一个 Bot 的长轮次永远不会拖延另一个 Bot 的邮件（也不会让它超过 `bot_mode.envelope_ttl_seconds` 而过期）；发给*同一个* Bot 的消息则按顺序、一次一轮地投递。
- **Desktop 是信使。** 只要一台同时认识两个连接的 Desktop 在运行，跨连接投递就能工作（它持有 socket 和凭据——gateway 之间彼此看不到对方的认证信息）。如果 Desktop 在投递途中被关闭，发送方的 Bot 会被告知回复没有送达，而不是被无限期挂起。若需要完全不经过 Desktop 的、永远在线的机器对机器消息，请注册一个 peer（见下方 `hermes peer`）——这两条路径可以共存。

### Bot 发起的跨机器私信（`hermes peer`）

一台机器上的 Bot 可以在没有任何 Desktop 参与的情况下，给**另一台机器的 gateway** 上的 Bot 发消息。把对方的 gateway 注册为一个 *peer*（它的 API server URL + `API_SERVER_KEY`）：

```bash
hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>
hermes peer list
hermes peer dm spark < /tmp/dm.txt        # 消息内容来自一个文件(不经过 shell 解释)
hermes peer dm spark/researcher < /tmp/dm.txt   # 多路复用 peer 上的指定 profile
hermes peer run spark --idempotency-key ticket-123 < /tmp/long-task.txt
hermes peer status spark run_abc123
hermes peer stop spark run_abc123
```

`hermes peer dm` 会通过该 peer 已有的 API server，把消息投递进远程 agent 的规范 Bot Chat，在那里运行一轮 agent，并把回复打印到 stdout——这正是本地 `hermes -p <bot> chat` 命令的跨机器对应版本。

`peer dm` 只适合简短的查询和回执，因为它会一直占用一条 HTTP 连接直到轮次结束。对于较长的轮次，`peer run` 会立即返回一个 `run_id`；用 `peer status` 轮询它。这次运行会继承规范 Bot Chat 的记录，而一个稳定的 `--idempotency-key` 会让重试返回原来的那次运行，而不是开始重复的工作。用 `peer stop` 加上这个确切的运行 ID 来中断它，而不会误伤其他轮次。

一旦注册了 peer，教给每个 Bot Chat 的消息协议（`agent.bot_mode_protocol`）就会自动包含 peer 花名册，`message_agent` 也可以直接接受 peer 目标——`message_agent(target="spark/researcher", …)`，或用 `target="spark"` 指向该 peer 的主 agent——这样**你的 Bot 会自己了解到**其他机器上存在队友，以及如何触达它们。注册或移除一个 peer 会在下一条消息时刷新每个 Bot Chat 的协议（能力版本）。

前提条件：peer 所在机器运行 `api_server` gateway 平台，并配置了强密码的 `API_SERVER_KEY`；能否触达取决于你的网络（局域网、Tailscale、VPN）。这个 key 是一项凭据，保存在 `~/.hermes/.env` 中的 `HERMES_PEER_<NAME>_KEY` 下；peer 的名称/URL 保存在 `config.yaml` 的 `bot_peers` 下。

:::note 单向可达（NAT）
跨 gateway 的链接是 gateway 到 gateway 的直接连接——Desktop 只是观察者，不是中继。位于家庭 NAT 之后的 gateway 可以向公网 peer 主动拨出（笔记本 → VPS 可行），但反方向没有入站路由（VPS → 家庭会失败），除非你的网络提供了一条。如果你的 Group Chat 跨越了 NAT 边界，请把房间的权威放在每个参与者都能触达的主机上（通常是公网 VPS），或者用 Tailscale/VPN 打通网络。
:::

### 转移托管房间的权威

权威接管是一项**运维恢复流程**，而不是原子化的交接。请在相应的 gateway 上使用现有的 JSON-RPC 方法 `groups.promote` 和 `groups.demote`。不存在 `groups.peer.promote` 或 `groups.peer.demote` 方法；`groups.capabilities` 会列出你的 gateway 支持的方法。

:::warning 提升之前先隔离旧的写入方
在发送 `confirm: true` 之前，先确认之前的权威**无法再提交**，并在它被降级之前一直保持这道隔离。停止它写入房间的进程并阻止自动重启，或者使用等效的基础设施隔离手段。网络超时、Desktop 断开连接或 `groups.stop` 都不算证明：旧 gateway 可能仍在运行，而停止一个轮次并不会撤销房间权威。如果无法建立隔离，就不要提升。
:::

1. **检查副本覆盖度。** 在替代 gateway 上，用 `{"room_id":"ROOM_ID"}` 检查 `groups.replica_state`，比较 `last_seq` 与 `latest_seq`。计划内接管前要求副本完整；提升本身并不会检查这一覆盖度。`groups.replicate` 在摄取完 `groups.log` 返回的页面后会报告 `caught_up`；仅仅注册了 peer 并不能证明替代方已拥有房间历史。追平状态描述的是最后一个已复制的页面，并不能证明旧写入方已经停止，或不存在更新的事件。对于计划内迁移，先让写入方静默，复制到最终游标，然后保持隔离。对于灾难恢复，要考虑到从未到达副本的历史。
2. **只在旧写入方已被隔离时提升。** 在替代方上：

   ```json
   {"jsonrpc":"2.0","id":1,"method":"groups.promote","params":{"room_id":"ROOM_ID","confirm":true,"reason":"planned-handover"}}
   ```

   `room_id` 和 `confirm: true` 是必需的；`reason` 可选，默认为 `authority-unreachable`。确认是你对"之前的权威无法再提交"这一事实的断言，**而不是**请求自动隔离它。缺少确认时该调用会返回错误 `4118`。成功的结果会报告 `authority_gateway_id` 和 `authority_epoch`（已复制的 epoch 加一）。
3. **在旧权威恢复服务之前先将其降级。** 在旧 gateway 上通过受控的恢复连接应用这个 RPC，同时保持其常规房间写入方处于隔离状态。把示例中的 gateway ID 和 epoch 替换为成功提升时返回的确切值：

   ```json
   {"jsonrpc":"2.0","id":2,"method":"groups.demote","params":{"room_id":"ROOM_ID","observed_gateway_id":"NEW_GATEWAY_ID","observed_epoch":2}}
   ```

   三个参数都是必需的。不要猜测未来的 epoch：降级需要的是存在更新权威的证据，而不是编造的值。它会记录 `authority.lost` 并采用观察到的谱系；重复相同的谱系是幂等的。如果旧主机不可用，请保持隔离，并在恢复其常规写入方之前完成这一步。
4. **验证并重新连接。** 在两个 gateway 上读取 `groups.state`，将 `room.authority_gateway_id` 和 `room.authority_epoch` 与提升结果进行比较。旧权威的发送必须被拒绝；把客户端指向替代方。降级只隔离写入；它不会合并历史，也不会自动把旧的权威存储变成同步副本。

在旧 gateway 仍可写入时进行提升，会让两个独立的 `state.db` 存储都接受消息并产生分叉的历史。替代方上更高的 epoch 并不会远程禁用旧的写入方；出现脑裂并不要求 epoch 相等。如果历史已经分叉，请隔离写入方并保留两份历史以供恢复，而不要假设提升、降级或重放会把它们合并。

## 跨机器的 Bot

当你在**设置 → Connections** 中注册了多个后端——本地运行时、远程 gateway、SSH 主机、Hermes Cloud 实例——花名册会持续显示来自**每一个**已连接来源的 Bot：SSH 来源会在不在远程主机上启动任何进程的前提下被盘点，暂时无法触达的机器会保留它们最后已知的行，而不是直接消失。当同一个 profile 名称存在于多个来源时，handle 会以 `@name-device` 的形式消除歧义（例如 `@research-homelab`）。一个 Bot 的对话、会话、记忆和例行任务都存放在拥有该 profile 的那台机器上。

点击一个 Connections Bot **不会**把你的窗口切换到那台机器上——留在你当前的对话里 @提及 它、把它安排进某个群聊，或者用 **Create on** 选择器直接在它所在的机器上创建新的 agent。云端和本地 agent 就这样共用同一个花名册：注册你的 Hermes Cloud 实例和你的桌面（比如通过 Tailscale 或 SSH），它们的 Bot 就能互相发消息、共处同一个房间，每个 agent 的工作都运行在自己的机器上。跨这些机器的 Bot 间私信会自动走 Desktop 中继（见上文*跨已连接机器的消息*）。

完整的多连接指南请参阅[将 Desktop 连接到多个 Hermes 实例](./multi-connection-desktop)。

## Warm Bot Backends（同时运行多少个 Bot）

每个本地 Bot 都运行在自己的后端进程中，Desktop 最多同时保留 **设置 → Advanced → Warm Bot Backends** 个后端存活（默认 3 个，每个约 60 MB）。空闲的后端会在该设置旁边的空闲超时（默认 10 分钟）后被回收；`desktop.log` 中紧跟空闲回收消息之后出现的 `Hermes backend for profile "<name>" exited (1)` 一行就是这次清理，而不是崩溃。当所有槽位都被占用时，你打开的 Bot 会最多等待 30 秒以获得一个槽位，然后以 *timed out waiting for a free local slot* 失败。

读取另一个 Bot 的聊天历史以及后台的记录刷新**不会**占用槽位——只有交互式打开或正在运行的轮次才会。如果你在驾驭一支庞大的队伍（成员众多的群聊，或跨多个 profile 的 Kanban 派发），请把 Warm Bot Backends 调高到你预期同时活跃的 Bot 数量，并为机器配备相应的内存。把它设得比你实际使用的 profile 数量还高，只会增加启动开销。

## 关闭它

Bot 模式是一个内置的桌面插件。在 **Capabilities → Plugins → Bots** 中关闭它的 **Desktop** 开关——花名册、Routines 面板和输入框中间件会实时注销，无需重启。无论开关状态如何，你的 profile、会话和 cron 任务都不受影响；Bot 模式从不拥有你的数据，它只是负责渲染。

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
