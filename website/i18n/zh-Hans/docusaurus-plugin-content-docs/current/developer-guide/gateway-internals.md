---
sidebar_position: 7
title: "Gateway 内部机制"
description: "消息 gateway 如何启动、授权用户、路由会话以及投递消息"
---

# Gateway 内部机制

gateway 是一个长期运行的进程，通过统一架构将 Hermes 连接到 20 余个外部消息平台。

## 规范群组元数据与 profile 发现

已认证的规范 WebSocket 暴露 `groups.capabilities`、`groups.list`、`groups.state` 和 `groups.log`（需要 `session:read`）；`groups.create`、`groups.rename` 和 `groups.disband` 需要 `session:control`。它们复用拥有该权威（authority）的数据库中的 hosted-room 元数据协议。一个显式的 `profile` 选择器必须解析到该权威的 home。`disband` 会拒绝仍有未解决 driver 工作的房间，而不是绕过 Stop。

这是**元数据管理，不是 hosted-room 推断**。现有的本地 hosted-room driver 仍使用旧式 TUI 会话处理器；因此规范 surface 报告 `driver: false`，且不宣传 send、retry、approval 或 peer-grant 操作。它绝不导入那个 driver 去执行群组回合。

`profiles.list` 需要 `session:read`，并首先返回已认证的 profile（连同其显示元数据和头像存在情况），然后是作为发现元数据的被服务的兄弟 profile 和 hosted-room 目标。`include_sessions` 默认为 true：`canonical_session` 解析精确的 `Bot Chat` 标题（而非最近的对话），`last_session` 是最新的可见本地对话（隐藏行绝不获胜）。两者都只计算该 actor 作为 operator 拥有或可读的本地 chat：铸造的 `local-*` chat，以及在持久绑定下被领养的 pre-upgrade chat（它们保留旧 id）。只有当此连接已经把 session 动词路由到某个兄弟 profile（一个 host 范围的授权：Desktop 的共享 socket、dashboard）时，该兄弟的行才携带相同字段，读取自该兄弟自己的权威、并以本连接的 subject 和 capabilities 进行。一个 profile 绑定的 ticket 只得到元数据，一个正在退役、列在路由到别处名字下的兄弟也一样，其存储不响应也一样；仍在启动的兄弟尚未列出。当一个 profile 的存储尚未移动时，行会被复用。发现不会创建、恢复或取消归档 session，不会打开自己的数据库句柄或重新打开一个 unserve 关闭的句柄，也不会运行推理。

## 规范辅助读取

Desktop 和 Ink 使用已认证的 `session.control.read`、`process.list`、`subagent.list` 和 `subagent.tail`（带 `session_id`）。它们需要 `session:read` 以及拥有 profile/session 身份。这些轮询绝不恢复一个冷会话：先使用 `session.resume`。一个退役的路由绑定返回 `stale_generation`，而非暴露其替代对话。

控制读取通过现有前端字段 allowlist 投射物理 transcript 的已持久化 goal、loop 和 heartbeat，而不构造 manager 或清除等待屏障。修订对可见快照做哈希；带时限的等待屏障在其截止时间过了之后可以消失，而不改变持久化状态。进程读取使用已拥有的注册表对象，并包含一个有边界的输出尾部；它们不会恢复进程、消费通知或扫描保留的收据。子 agent 读取需要当前父级的活跃对象祖先关系，而非仅仅相同的 session ID；尾部被限制在 16 KiB。缺失或退役的子级返回 `available: false`。这些是读取，不是子级 steering 或 stop 控制。

托管 worker 策略对进程/子 agent 读取返回 `unsupported_projection`：owner 的进程本地注册表无法如实描述另一个解释器。控制变更（`session.control`）、进程变更和子 agent steering 不会被这些读取投射所启用。

## 共享权威的 pending-input 快照

`session.resume` 返回每个待处理 admission，携带其原始 `input_id` 和已提交的公有 `text`，以及规范的 `admission_id`、sequence 和 status。这包括原生消息输入和本地客户端输入。快照不暴露私有的原生路由、source 或授权信封。提交与接收 ACK 保留其既有形态。客户端必须使用 `admission_id` 做取消和收据查询，而不能假设它等于原始输入 ID。

队列 admission、claim、cancellation 和 settlement 也向附加的 viewer 发布 `session.info`。其 `pending` 行使用与 resume 相同的投射。settlement 在 `message.complete` 之前发布该状态，把终止事件保留为该回合的最后一个事件。Desktop 把两条路径都翻译到它现有的 `pending_submissions` 队列存储。

一个在 owning gateway 死亡时处于 `started` 的回合，会被下一个 owner 恢复为 `unknown`：Hermes 无法证明哪些 side effect 发生过，因此它既不重放输入，也不让其后的排队回合运行。`prompt.resolve_unknown`（`session:control`，参数 `session_id`、`admission_id`、`execution_generation`）确认这次丢失：该行 settle 为 `interrupted`，session FIFO 恢复，follower 恰好运行一次。generation 必须是 stamp 在那个 unknown 行上的那个（在 resume/`session.info` 投射中可见），因此一个过期或已解决的行会被以 `stale_generation` 拒绝。Desktop 在队列面板中把这种行显示为「Turn lost during restart」并带一个 **Discard** 按钮；gateway CLI 把它暴露为 `/discard <admission_id>`。丢失的输入留在 transcript 中供用户重发；discard 用一条 Hermes 通知（像一个 failed turn 那样）关闭它，因此它绝不会被合并进 follower 的模型请求。

一个被这样暂停的消息用户（Telegram、Discord、...）会在每次暂停事件期间通过适配器的普通回复路径被告知一次：对话已暂停，`/reset` 会开启一个全新对话；之后发到同一暂停上的消息会被静默接纳并保持排队。它们的投递等待者会带着暂停原因被释放，而不是让适配器循环停住。`/reset` 把路由轮换到一个新 session（丢失的回合及其 follower 留在旧 session 上）；operator 也可以从 Desktop 或 CLI 解决那个 unknown 行。

## 共享权威的 prompt 附件

`prompt.submit` 接受一个可选的 `attachments: [{path, mime}]` 列表，与 `text` 并列。客户端把图片字节 staging 到 profile 图片缓存（served `HERMES_HOME` 下的 `cache/images/`，即消息适配器 staging 下载的同一个地方）；path 必须是该目录中的普通文件，`mime` 必须是图片类型（`image/png`、`image/jpeg`、`image/gif`、`image/webp`），每次提交至多 10 个，每个都在 `gateway.max_inbound_media_bytes` 之内。权威在 admission 时把这些字节复制进它不可变的 `native-inputs` 存储，并把经 sha256 校验的引用存进私有 payload，因此之后对 staging 文件的变更或缓存清理都无法改变要执行的内容，包括在 owner 重启之后。执行时把它们恢复到 `MessageEvent.media_urls`/`media_types` 上，并应用普通的图片路由（`agent.image_input_mode`，原生 `image_url` 部分 vs. 文本预分析）。纯文本提交保留 `{text}` payload 和指纹。ACP transport agent 用它处理 image、image resource-link 和 embedded-image prompt 块；Ink 和 Desktop 仍通过旧式 `image.attach`/`file.attach` RPC 附加。

## 原生 admission 的启动恢复

普通 gateway 启动会在适配器连接、路由发布和启动恢复门之后，恢复已接受的原生输入。只有从未启动的排队工作才有资格。恢复使用当前路由索引和已连接的适配器，重新检查发送者授权和保留的 connector/profile 绑定，并在路由、transport 或授权不可用时把行留在排队状态。恢复该配置即可让它们在稍后的启动中被恢复。

一个被中断的 started admission 变为 `unknown` 并暂停其 session 的 FIFO；其 follower 不会自动运行。旧式合成重启回合被排除在带有规范 admission 历史的 session 之外，因此它们无法与持久队列竞争或绕过 unknown 暂停。已完成的输入不会在重启时被重新推断。在 `gateway.multiplex_profiles` 下，bootstrap 为每个 served profile home（默认加上 `profiles/` 下的每个目录）构建一个 session authority，每个都绑定到该 home 的 `state.db` 并在该 profile 的运行时 scope 下录入。每条 admission 路径（bot 消息、`/p/<profile>/` API 和 webhook、cron fire、kanban 分发、原生 CLI/Desktop/ACP attach）都从被路由 profile 的 home 解析其 authority，因此恢复、FIFO 顺序和 exactly-once settlement 按 profile 成立，secondary 的行绝不会落到 default 的账本里。一个被服务 secondary 的客户端通过 default home 的描述符（`served_profiles`）发现 multiplexer，并用自己的 `profile_id` 附加到其控制 socket。

## 共享权威的本地斜杠命令

已认证的本地会话接受 `slash.exec({session_id, command})` 和 `command.dispatch({session_id, name, arg})`。两者都使用现有的命令注册表和 gateway 处理器，无需旧式 slash worker 或第二个执行运行时。一个可选的 `profile` 选择器必须匹配 owning daemon 的 profile。调用方 source 和路由覆盖会被拒绝；另一个已认证 actor 无法操作该会话。

支持的读取命令是 `/help`、`/commands`、`/status`、`/context`、`/version` 和 `/whoami`（`session:read`）。`/title` 需要 `session:control`，即使只是查询，并保留现有的忙拒绝。注册表别名可用。处理结果使用 `{type: "exec", output}`。

Skill 名称和已配置的 quick-command 别名（指向受支持的命令或 skill）在 owner 的 profile 中解析。Skill 解析需要 `session:submit` 并返回 `{type: "skill", name, message, display}`。Desktop/Ink 通过它们正常的带身份 `prompt.submit` 发送扩展后的消息；解析不是一次 admission ACK。权威的持久 FIFO 拥有重试和执行，skill 内容保持为用户消息而不改变缓存系统前缀。

`type: exec` quick command 和插件斜杠命令需要 `session:control`，并通过消息 gateway 自己的 quick/plugin 分发执行（slash 访问策略、带 30 秒上限和脱敏输出的清理环境 exec、插件 session 环境）。

其他命令返回 `unsupported_command`：bundles、配置/运行时变更、生命周期命令，以及 approval/secret slash 快捷方式都不暴露。既有的按 generation 绑定的 approval 和 interrupt RPC 保持独立。目录发现比这个经过评审的执行集更广。

## 共享权威的 busy input 与纠正

`config.get({session_id, key: "busy"})` 投射 session 冻结的 busy-input 偏好。`config.set({session_id, key: "busy", value})` 接受 `interrupt`、`steer` 或 `queue`，只改变那个活动会话的偏好，由它的 viewer 共享直到 owner 重启。两个操作都不改变 profile 设置或缓存的 agent/system prompt。此适配器不处理其他 config 键。

`session.steer` 和 `session.redirect` 需要 `session_id`、`text` 和当前的 `execution_generation`，加上已认证的 session-control 权限。可选的 `profile` 必须匹配权威。控制针对现有运行中的 agent：steer 使用普通纠正 drain；redirect 取消一个活跃的模型请求并继续同一回合（或在工具边界处 steer）。它们绝不创建一个排队 admission 或静默回退到一个新回合。`queued`（steer）和 `redirected` 确认运行时已接受，而不是崩溃持久化的投递；不要自动重试一个模棱两可的回复。`rejected` 意味着运行时错过了活跃纠正窗口，而 `stale_generation` 拒绝已 settle/被替换的工作。托管 worker 的纠正当前返回 `unsupported_control`；正常的持久队列提交和 Stop 仍保持独立控制。

## 共享权威的 setup readiness

已认证的本地客户端可以在创建 session 之前调用 `setup.status` 和 `setup.runtime_check`。两者都需要 `session:create` 并针对 daemon 拥有的 profile 运行；指向另一个 profile 的选择器会被拒绝。运行时 readiness 在事件循环之外复用现有的 provider 解析器，且不创建旧式 session 运行时。缺失的 provider 凭据仍是 failed readiness 结果，而非绕过 onboarding。

## 共享权威的 clarify 协议

当 HTTP/WebSocket surface 绑定到一个 `SessionAuthority` 时，附加的授权 viewer 会收到 `clarify.request` 事件，以及在 `session.resume` 的 `prompts` 快照中的待处理问题。这是 authority API 契约；它并不意味着每个原生前端都已渲染共享表单，或普通 bootstrap 已完成。

一个响应使用 `clarify.respond`，带 `session_id`、`execution_generation`、`prompt_id` 和一个 `answer` 字符串。连接的 viewer 必须已订阅并具有 `session:respond`。权威检查 running generation 并按精确 prompt ID 解析现存的原生 clarify waiter，而不向 admission 队列添加用户 prompt。第一个响应获胜；detach 不会取消它。原生回答和超时会使同一个 prompt 退役并发出 `clarify.settled`；响应文本不会被放到控制重放流中，尽管普通的 clarify 工具结果会成为对话的一部分。问题和选项标签使用现有显示脱敏器。

发布跟随成功的原生卡片投递。一次失败的原生投递不会创建一个可操作的共享表单。普通 approval 使用独立的 `approval.respond` 权限路径；sudo/secret prompt 和 daemon 重启后的持久控制恢复不由这个 clarify 路径实现。

## 修订检查的 session 元数据

绑定权威的 WebSocket 接受针对已注册 session 的 `session.mutate`。参数是 `session_id`、`request_id`、`expected_revision`、`operation` 和 `payload`。`rename` 只接受 `{"title": "New title"}`；`archive` 只接受 `{"archived": true}` 或 `{"archived": false}`。调用方需要 `session:control`。`session.resume` 中的 session 句柄提供当前 revision。

编辑及其重试收据一起提交。重复相同请求会返回原始结果而不再次递增 revision；用不同的内容复用其 ID 会返回 `admission_conflict`。带过期 revision 的竞争编辑返回 `revision_conflict`。draining 会在路由或 transcript 创建之前拒绝新的变更和新的 session 注册。这些编辑不中断运行中的回合。规范 reset、delete、rewind、model、branch 和 compression 操作使用相同的收据边界。`branch` 接受一个可选 `title` 和一个 `through_message_id`（要保留的最后一条消息的 row id；应答 assistant 边界的工具结果随附）。没有它则复制整个活跃 transcript。一个请求断开不会取消其 owner 持有的变更：commit、live 投射和队列唤醒一起完成，profile 退役会在释放 store 之前 join 该工作。一个精确的旧收据无法驱逐一个更新的运行中 agent。

规范线协议模型位于 `tui_gateway/contracts/canonical*.py`，并生成 `CanonicalRpcMethods` 加上 OpenRPC `x-canonical-methods` 目录，独立于 standalone 协议。参数和成功结果都被检查。

当一个已接受的输入被阻塞在一个 unknown 执行之后时，ACP 和有界 CLI viewer 会带着一个显式恢复提示 detach。已接受的输入保持排队：先解决丢失的回合再继续，而不是重发它。

## 私有原生 HTTP 认证

权威 daemon 接受来自其同用户私有控制 socket 的**每次请求一次性的新鲜 HTTP grant**。发送 `session-ticket`，精确携带 `profile_id`（规范 served primary home）、`instance_id`（就绪 daemon 的 boot ID）和 `purpose: "native-http"`。返回的 ticket 在 30 秒后过期，且只携带 `http:owner`；从原生主进程以 `X-Hermes-Gateway-Ticket` 发送它。绝不把它放进 URL、cookie、公开描述符、renderer 配置或日志，也绝不通过抓取公开页面来获取 dashboard token。

现有 HTTP 认证 seam 只对匹配的就绪 daemon 和 primary profile 兑现该 grant。它需要一个真实的 loopback socket 对端（在 trusted-proxy 重写之前捕获），以及**完全没有 Origin 头**（即使是空的 Origin 也会被拒绝）。一个无效、重复、过期、重放或 wrong-purpose 的头返回 401，而不回退到提供的静态 token 或 OAuth cookie。draining/未就绪的 listener 拒绝 admission；restart 和 shutdown 会撤销 grant。HTTP grant 不认证 WebSocket，interactive/exposure/worker grant 不认证 HTTP。没有此头的请求保留现有的静态 session-token、OAuth bearer/cookie、Host 和 route-specific token-provider 门。

这个本地 bootstrap 是一个显式的同 OS 用户管理边界，而非针对远程客户端的 OAuth 绕过。一个 exposure 代理绝不能用自己的原生 grant 替换远程调用方的凭据。它也不是针对已以同一 OS 用户身份运行代码的文件系统沙箱。

一个原生 grant 不会通过 URL 或 JSON `profile` 选择器重定向 daemon：每个提供的选择器都必须解析到 served primary home。现有 config 的 `current`/empty/default-name 契约被保留，而不是把 `default` 当作任意启动 home 的别名。跨 profile 目标返回 403 `profile_scope_mismatch`，包括冲突的重复查询选择器。`GET /api/profiles` 和 `GET /api/profiles/active` 保留其发现元数据契约；发现并不授予另一个 profile 的数据权威。聚合 session/sidebar 路由需要显式的 own-profile 选择器，cron job 路由需要显式的 own-profile 查询。全 profile 的 project/transcript 聚合、profile 创建/导入/激活，以及 profile 重命名/删除不由此 primary-profile 凭据授予。原生客户端绝不能把 `all` 重写成 `current`，或借用 primary grant 隐藏跨 profile 失败。

从原生 Desktop 创建 profile 使用一个独立的、同用户的 `purpose: "native-profile-create"` grant，只携带 `profiles:create`。它只被 `POST /api/profiles` 接受；它不授权 profile 读取、导入、激活、重命名、删除或 WebSocket 访问。现有 create 处理器校验名称和 clone 选项。可选的 `soul` 内容在 staging profile 发布之前写入，因此创建不需要后续写或一个运行中的新 profile gateway；`soul_set` 确认它已被应用。较旧的远程后端保留独立的、显式 profile-scoped 的 SOUL 更新。Desktop 对每个 create 请求铸造此 grant，绝不发送到 renderer，后续请求仍使用普通的 profile-bound grant。同样的过期、一次性、daemon、socket-peer 和 Origin 检查都适用。普通 `native-http` grant 在创建时仍会收到 403。

原生 HTTP 集成由 `tests/gateway/test_native_http_auth.py` 中真实的 ordinary-daemon/control-socket/TCP 探针覆盖，还有门控的 OAuth/cookie 和真实的非 loopback-peer 控制。这并不建立原生 Desktop UI 对等或平台特定的 Windows/macOS bootstrap 校验。

## 冻结的本地启动凭据

本地 `session.create` 收据保留 launch policy，而非明文 API key。profile 配置中的内联凭据在重启后从那个现有私有源借用：规范 profile home、凭据路径、值指纹和冻结 policy 身份都必须匹配。provider 端点和请求设置仍来自冻结 policy，而非当前 config。缺失或变化的凭据会 fail closed，而非选择另一个 key。不创建第二个秘密存储。带凭据的头、环境映射和 terminal 投射从持久 policy JSON 中脱敏，并通过相同的源检查私有回填。

重试一个创建请求会返回其原始 session 和 policy，而不绑定当前的替换凭据。一个临时的 `--api-key` 仍只在 authority 生命周期内有效；重启会撤销该 key，而非基于 config 的 session。只包含实例绑定凭据引用的旧收据无法安全恢复丢失的值，保持 fail-closed。它们的历史仍可读；创建新 session 以领用当前凭据。凭据指纹是私有元数据，绝不出现在公开 API 响应中。

CLI 的 cost/data-training 同意在 gateway 发现和 session 创建之前运行，包括顶层的一次性启动。它不会变成进程范围选项。

## 关键文件

| 文件 | 用途 |
|------|------|
| `gateway/run.py` | `GatewayRunner` facade——组合 `gateway/run_*.py` 兄弟 mixin（startup、adapters、inbound、turn、busy、goals、notifications、shutdown、…）与 `gateway/slash_commands_*.py` 处理器 |
| `gateway/session.py` | `SessionStore`——会话持久化与会话键构造 |
| `gateway/delivery.py` | 向目标平台/频道投递出站消息 |
| `gateway/pairing.py` | 用于用户授权的 DM 配对流程 |
| `gateway/channel_directory.py` | 将聊天 ID 映射为可读名称，用于 cron 投递 |
| `gateway/hooks.py` | Hook 发现、加载与生命周期事件分发 |
| `gateway/mirror.py` | 为 `send_message` 提供跨会话消息镜像 |
| `gateway/status.py` | 面向 profile 范围的 gateway 实例的 token 锁管理 |
| `gateway/builtin_hooks/` | 始终注册的 hook 扩展点（当前未内置任何 hook） |
| `gateway/platform_registry.py` | 适配器注册表、工厂，以及对捆绑平台插件的延迟（惰性）加载 |
| `plugins/platforms/<name>/` | 捆绑的消息适配器（多数平台：`adapter.py` + `plugin.yaml`） |
| `gateway/platforms/` | 共享的 `base.py` 与遗留/直接适配器（Signal、API server、webhooks 等） |

## 可选服务安装

运行中的 gateway 和已安装的 OS 服务是两回事。`hermes gateway run` 在没有服务的情况下也能工作，包括没有配置任何消息平台的情况。

`hermes setup`、quick setup 和 `hermes gateway setup` 会向交互用户询问一次：

> Install the gateway service so Hermes starts automatically at login and keeps scheduled jobs and messaging available?

该 profile 的 `gateway.service_install_choice` 存储 `null`、`install` 或 `decline`。未设置的值不是安装许可。拒绝会记录 `decline` 而不创建、启用或移除服务。在 gateway 向导中，选择 **Start now** 但拒绝服务安装会启动一个不受管（unmanaged）的 gateway；它不会创建一个禁用服务。没有服务持久化时，消息和计划工作会在登出/重启时停止。主机停机时任务无法运行。

非交互式 setup 和导入不会安装缺失的服务，即使导入的 config 说 `install`。已安装的服务保持权威，无论保存的偏好如何都能启动。要稍后显式选择加入，运行 `hermes gateway install`；该选择只在成功安装后被记录。要在 setup 期间再次被询问，用 `hermes config set gateway.service_install_choice null` 重置该选择。

Setup 编排位于 `hermes_cli/gateway_setup_service.py`。其 `ensure_gateway_service` 辅助函数不是通用的运行时发现或自动启动 API：普通调用绝不会仅仅基于存储的偏好就重新安装一个缺失的服务。

## `hermes gateway ensure`

客户端（Desktop、`hermes chat`、TUI）用 `hermes gateway ensure --json --timeout 60` 发现或启动该 profile 的 session 运行时；摘要见 [CLI reference](../reference/cli-commands.md#local-runtime-discovery-and-startup)。

timeout 是一个有限的、正的以秒计的总截止时间（默认 `60`，足够一个繁忙主机上的冷 daemon 启动）。已存在的 reservation 或正在启动的 owner 会被等待，而非替换。当确认缺席时，已有服务优先；只有当找不到服务时才请求一个 unmanaged 进程。此命令绝不安装或重写服务定义、启用 linger、提升权限或清除更新栅栏。模棱两可或不可访问的 ownership 会 fail closed。

在请求服务启动之前，`ensure` 检查其规范 profile 目录和执行账户，而不仅是其 unit/task 名称。Linux 使用 manager 加载的命令和环境（包括有效 drop-in），macOS 检查已加载的 launchd job 而非假设磁盘上的 plist 是最新的，Windows 查询实际 task XML、principal 和已安装 launcher。

- `profile_mismatch`：已安装服务选择了不同的 home 或 profile。检查所选服务的 `HERMES_HOME` 和命令行 profile 选择器；使用匹配的 profile 或显式修复服务配置。
- `service_account_mismatch`：服务属于另一个账户。以该账户运行客户端，或为预期用户显式配置一个服务。
- `service_identity_unverified`：manager 没有暴露足够的身份数据，或定义使用了不支持动态配置。用 `systemctl [--user] show <unit> --all`、`launchctl print <domain>/<label>` 或 `schtasks /Query /TN <task> /XML` 检查它。Linux 环境文件、PAM/动态用户、启动时 hook 和备用根文件系统需要 operator 审查；任意 shell launcher 和修改过的 Windows launcher 脚本不被 `ensure` 解释。在重试前通过一个有意为之的服务管理操作恢复一个显式受支持的定义。

这些拒绝不会启动服务、重写配置或启动 unmanaged 替代品。标准显式绑定的 default 和 custom-root 安装仍符合资格；一次成功的启动请求仍须通过 live readiness 检查。

在 POSIX 上，本地 bootstrap 在信任里面的控制 socket 之前检查 profile 目录。新 reserve 的 home 以 `0700` 模式创建；`ensure` 从不更改现有权限。home 保留 operator 的模式（一个符号链接的 `~/.hermes`、`HERMES_HOME_MODE` `0701`/`0750`，或一个 `0755` home 都能工作）：read 和 search 位对 owner-only（`0600`）的 socket 毫无意义。只有另一个用户可能用来 swap socket 的写访问才会被以 `inaccessible / unsafe_control_permissions` 拒绝：world-write 始终拒绝，group-write 除非该组是 owner 的私有组（umask-`002` user-private-group 默认）且没有其他成员、没有访问 ACL。socket、其指针文件和 fallback socket 目录保持严格 owner-only。在重试本地附加前移除额外的写位（`chmod o-w,g-w`）。

| Exit code | 含义 |
|-----------|------|
| `0` | 兼容的 session 运行时就绪。 |
| `2` | 无效调用，包括无效 timeout 或未知参数。 |
| `3` | 不兼容的运行时协议或能力。 |
| `4` | 授权或 profile 不匹配。 |
| `5` | 截止时间已到；启动可能仍在进行。 |
| `6` | 运行时 draining 或更新进行中。 |
| `7` | 运行时不可访问、监督者冲突或启动失败。 |

服务启动命令或进程创建不是 readiness 确认。一个尚未暴露 session-authority 能力的 gateway 可以一直保持 `starting` 直到截止时间，即使其消息适配器正在工作。不要把 exit `5` 当作替换该 owner 的许可。在 Windows 上，只有 Startup-folder 的安装需要登录而非 unmanaged fallback；难以从父 job detach 会报告为失败，而非用更弱的进程隔离重试。

服务持久化在 setup 期间单独选择加入。导入和非交互式 setup 不会基于导入的偏好安装缺失的服务。见 [可选服务安装](#可选服务安装)。

## 架构概览

```text
┌─────────────────────────────────────────────────┐
│                  GatewayRunner                  │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Telegram │  │ Discord  │  │  Slack   │       │
│  │ Adapter  │  │ Adapter  │  │ Adapter  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │
│       └─────────────┼─────────────┘             │
│                     ▼                           │
│              _handle_message()                  │
│                     │                           │
│         ┌───────────┼───────────┐               │
│         ▼           ▼           ▼               │
│  Slash command   AIAgent    Queue/BG            │
│    dispatch      creation   sessions            │
│                     │                           │
│                     ▼                           │
│                 SessionStore                    │
│              (SQLite persistence)               │
└───────┴─────────────┴─────────────┴─────────────┘
```

## 消息流程

当消息从任意平台到达时：

1. **平台适配器**接收原始事件，将其规范化为 `MessageEvent`
2. **基础适配器**检查活跃会话守卫：
   - 若该会话的 agent 正在运行 → 将消息加入队列，设置中断事件
   - 若为 `/approve`、`/deny`、`/stop` → 绕过守卫（内联分发）
3. **GatewayRunner._handle_message()** 接收事件：
   - 通过 `_session_key_for_source()` 解析会话键（格式：`agent:{namespace}:{platform}:{chat_type}:{chat_id}`；namespace 对默认 profile 为 `main`，在多路复用下为 `<profile>` —— 见 [多路复用 profile](#多路复用-profile)）
   - 检查授权（见下方授权章节）
   - 检查是否为斜杠命令 → 分发至命令处理器
   - 检查 agent 是否已在运行 → 拦截 `/stop`、`/status` 等命令
   - 否则 → 创建 `AIAgent` 实例并运行对话
4. **响应**通过平台适配器回传

### 会话键格式

会话键编码了完整的路由上下文：

```
agent:{namespace}:{platform}:{chat_type}:{chat_id}
```

例如：默认 profile 为 `agent:main:telegram:private:123456789`，或当 multiplexer 把该 chat 路由到 profile `work` 时为 `agent:work:telegram:private:123456789`（`gateway/session.py::_session_key_namespace`；一个字面名叫 `main` 的 profile 会被标记为 `main~`）。

支持线程的平台（Telegram 论坛话题、Discord 线程、Slack 线程）可能在 chat_id 部分包含线程 ID。**切勿手动构造会话键** —— 请始终使用 `gateway/session.py` 中的 `build_session_key()`。

### 两级消息守卫

当 agent 正在运行时，传入消息会依次经过两级守卫：

1. **第一级 — 基础适配器**（`gateway/platforms/base.py`）：检查 `_active_sessions`。若会话处于活跃状态，将消息加入 `_pending_messages` 队列并设置中断事件。此级在消息到达 gateway runner *之前*进行拦截。

2. **第二级 — Gateway runner**（`gateway/run_inbound.py`）：检查 `_running_agents`。拦截特定命令（`/stop`、`/new`、`/queue`、`/status`、`/approve`、`/deny`）并进行相应路由。其余所有消息触发 `running_agent.interrupt()`。

必须在 agent 被阻塞时到达 runner 的命令（如 `/approve`）通过 `await self._message_handler(event)` **内联**分发 —— 绕过后台任务系统以避免竞态条件。

## 授权

Gateway 使用多层授权检查，按顺序评估：

1. **平台级全量放行标志**（如 `TELEGRAM_ALLOW_ALL_USERS`）—— 若设置，该平台所有用户均被授权
2. **平台白名单**（如 `TELEGRAM_ALLOWED_USERS`）—— 逗号分隔的用户 ID
3. **DM 配对** —— 已认证用户可通过配对码为新用户授权
4. **全局放行标志**（`GATEWAY_ALLOW_ALL_USERS`，或 `config.yaml` 中的 `gateway.allow_all_users`，由 `gateway/config_loader.py::bridge_core_env_settings` 桥接为环境变量）—— 若设置，所有平台的所有用户均被授权
5. **默认：拒绝** —— 未授权用户被拒绝

### DM 配对流程

```text
Admin: /pair
Gateway: "Pairing code: ABC123. Share with the user."
New user: ABC123
Gateway: "Paired! You're now authorized."
```

配对状态持久化于 `gateway/pairing.py`，重启后仍然有效。

## 斜杠命令分发

Gateway 中所有斜杠命令均经过相同的解析流程：

1. `hermes_cli/commands.py` 中的 `resolve_command()` 将输入映射为规范名称（处理别名、前缀匹配）
2. 规范名称与 `GATEWAY_KNOWN_COMMANDS` 进行比对
3. `_handle_message()`（`gateway/run_inbound.py`）按名称查找处理器 —— `gateway/slash_commands_*.py` mixin 上的 `_handle_<name>_command` —— 通过 `gateway/run_busy.py` 中 `_IDLE_COMMANDS` / `_PLAIN_COMMANDS` 之上的 `_command_handler_table`；没有 `if canonical == ...` 链
4. 部分命令受配置门控（`CommandDef` 上的 `gateway_config_gate`）

### 运行中 Agent 守卫

在 agent 处理消息期间不得执行的命令会被提前拒绝：

当 `_quick_key in self._running_agents` 时，`gateway/run_busy.py` 中的 `_dispatch_busy_slash_command()` 按每个已识别命令的 `CommandDef.busy_policy` / `busy_handler` 路由：若存在运行中变体（`_busy_<key>_command`）则用之，否则当 `busy_policy` 允许时用普通处理器，否则返回拒绝消息（"⏳ Agent is running — `/model` can't run mid-turn…"）。

绕过命令（`/stop`、`/new`、`/approve`、`/deny`、`/queue`、`/status`）具有运行中处理器并被内联分发。

## 配置来源

Gateway 从多个来源读取配置：

| 来源 | 提供内容 |
|------|---------|
| `~/.hermes/.env` | API 密钥、bot token、平台凭据 |
| `~/.hermes/config.yaml` | 模型设置、工具配置、显示选项 |
| 环境变量 | 覆盖上述任意配置 |

与 CLI（使用带硬编码默认值的 `load_cli_config()`）不同，gateway 通过 YAML 加载器直接读取 `config.yaml`。这意味着存在于 CLI 默认值字典但不在用户配置文件中的配置键，在 CLI 和 gateway 之间可能表现不同。

## 平台适配器

所有消息平台都以插件适配器形式位于 `plugins/platforms/<name>/`；`gateway/platforms/` 只保留基类和 HTTP 入站适配器（`api_server`、`webhook`、`msgraph_webhook`）。它们都继承 `gateway/platforms/base.py` 中的 `BasePlatformAdapter`：

```text
plugins/platforms/                  # 插件打包的适配器（每个一个目录）
├── telegram/adapter.py     # Telegram Bot API（长轮询或 webhook）
├── discord/adapter.py      # Discord bot（通过 discord.py）
├── slack/adapter.py        # Slack Socket Mode
├── whatsapp/adapter.py     # WhatsApp（通过本地 Baileys 桥接）
├── whatsapp_cloud/adapter.py # WhatsApp Business Cloud API
├── matrix/adapter.py       # Matrix（通过 mautrix，可选 E2EE）
├── mattermost/adapter.py   # Mattermost WebSocket API
├── email/adapter.py        # 电子邮件（通过 IMAP/SMTP）
├── sms/adapter.py          # 短信（通过 Twilio）
├── dingtalk/adapter.py     # 钉钉 WebSocket
├── feishu/adapter.py       # 飞书/Lark WebSocket 或 webhook
├── wecom/adapter.py        # 企业微信（WeCom）回调
├── line/adapter.py         # LINE Messaging API
├── teams/adapter.py        # Microsoft Teams
├── signal/adapter.py       # Signal（通过 signal-cli HTTP 守护进程，SSE + JSON-RPC）
├── irc/adapter.py          # IRC（作用域锁的标准示例）
├── bluebubbles/adapter.py  # Apple iMessage（通过 BlueBubbles macOS 服务端）
├── weixin/adapter.py       # 微信（个人版，通过 iLink Bot API）
├── qqbot/                  # QQ Bot（腾讯 QQ，通过官方 API v2）
├── yuanbao/adapter.py      # 元宝（腾讯）私信/群组 WebSocket
└── …                       # google_chat、ntfy、photon、raft、simplex 等

gateway/platforms/                  # 核心 base 与 HTTP 入站适配器
├── base.py              # BasePlatformAdapter —— 所有平台的共享逻辑
├── msgraph_webhook.py   # Microsoft Graph 变更通知 webhook（Teams、Outlook 等）
├── webhook.py           # 入站/出站 webhook 适配器
└── api_server.py        # REST API 服务器适配器
```

**延迟加载：** 捆绑的 `kind: platform` 插件通过 `hermes_cli/plugins.py` 在 `gateway/platform_registry.py` 注册廉价的 `register_deferred` 加载器，这样平台 SDK 只在 gateway 启动、投递或运行 setup/status 时才导入——不在普通的 `hermes chat` 时导入。解析时按需加载一个适配器；完整枚举只在需要每个平台的路径上运行待处理加载器。

实验性的 connector 后台平台使用 `gateway/relay/` 中的通用 relay 适配器，而非直接平台模块。当配置了 `GATEWAY_RELAY_URL` 或 `gateway.relay_url` 时，gateway 注册 `relay` 平台，通过一个出站 WebSocket 拨号 connector，并在同一 socket 上接收 `descriptor`、`inbound` 和 `interrupt_inbound` 帧。connector 广告一个 `CapabilityDescriptor`；Hermes 能通过 relay 发回普通出站回复、无 token 的 `follow_up` 操作和 interrupt 帧。source-grounded 线协议契约见 [Relay ↔ Connector 契约](relay-connector-contract.md)。

适配器实现统一接口：
- `connect()` / `disconnect()` —— 生命周期管理
- `send()` —— 出站消息投递
- 入站事件被规范化为 `MessageEvent` 并通过 `handle_message()` 转发

内部 push 唤醒使用 `gateway.wake.admit_internal_event`：公开的 `handle_message()` 仍返回 `None`，但事件的进程本地 `_gateway_accepted` 收据只在调度或排队插入之后才被设置。缺失的处理器、不匹配的显式会话键或 queue-cap 丢失都不算 acceptance。覆盖 ingress 的自定义适配器应把内部事件委托给 `BasePlatformAdapter.handle_message()`（或显式记录实际 admission），而不是把一个被消费/丢弃的回调等同于 acceptance。该收据独立于 heartbeat 执行记账，且不绕过授权、紧急停止或之后的 turn-preparation 门。

### Token 锁

使用唯一凭据连接的适配器在 `connect()` 中调用 `acquire_scoped_lock()`，在 `disconnect()` 中调用 `release_scoped_lock()`。这可防止两个 profile 同时使用同一 bot token。

锁冲突会作为 `{scope}_lock` 发出，带 `retryable=True`，这样一次**运行中**的重连就能在另一个持有者退出后恢复。但在**启动**时，一个活的 foreign 持有者是一个配置冲突：`gateway/restart.py::is_global_startup_conflict()` 识别 `*_lock` / `lock_conflict` 代码族，启动路由器会把该平台 `fatal` 停车而非重试排队。在没有其他连接时 gateway 以 `78` 退出（`EX_CONFIG`，`gateway_state=startup_failed`），这样监督者停止重启它：systemd 通过 `RestartPreventExitStatus=78`，s6 通过 finish→125，launchd 在 stderr wrapper 把 78→0 映射后通过 `KeepAlive.SuccessfulExit=false`。当旁边存在一个真正瞬态的对端失败时，gateway 保持存活，只有对端重试。

## 投递路径

出站投递（`gateway/delivery.py`）处理以下场景：

- **直接回复** —— 将响应发回原始聊天
- **主频道投递** —— 将 cron 任务输出和后台结果路由至已配置的主频道
- **显式目标投递** —— 发送引擎指定 `telegram:-1001234567890`，通过 [`hermes send` CLI](../guides/pipe-script-output.md) 供 shell 脚本使用，以及 cron 的 `deliver:` 目标
- **跨平台投递** —— 投递至与原始消息不同的平台

Cron 任务投递**不会**镜像到 gateway 会话历史中 —— 它们仅存在于各自的 cron 会话中。这是有意为之的设计选择，以避免消息交替违规。

## Hooks

Gateway hook 是响应生命周期事件的 Python 模块。

### Gateway Hook 事件

| 事件 | 触发时机 |
|------|---------|
| `gateway:startup` | Gateway 进程启动时 |
| `session:start` | 新对话会话开始时 |
| `session:end` | 会话完成或超时时 |
| `session:reset` | 用户通过 `/new` 重置会话时 |
| `agent:start` | Agent 开始处理消息时 |
| `agent:step` | Agent 完成一次工具调用迭代时 |
| `agent:end` | Agent 完成并返回响应时 |
| `command:*` | 任意斜杠命令被执行时 |

Hook 从 `gateway/builtin_hooks/`（扩展点 —— 当前发行版中为空；`_register_builtin_hooks()` 是一个空操作存根）和 `<profile home>/hooks/`（用户安装；默认 profile 为 `~/.hermes/hooks/`，多路复用下每个 served profile 一个目录——路径在调用时解析，绝不在导入时）中发现。每个 hook 是一个包含 `HOOK.yaml` 清单和 `handler.py` 的目录。

## Memory Provider 集成

当 Memory Provider 插件（如 Honcho）启用时：

1. Gateway 为每条消息创建一个带会话 ID 的 `AIAgent`
2. `MemoryManager` 使用会话上下文初始化提供者
3. 提供者工具（如 `honcho_profile`、`viking_search`）通过以下路径路由：

```text
AIAgent._invoke_tool()
  → self._memory_manager.handle_tool_call(name, args)
    → provider.handle_tool_call(name, args)
```

4. 会话结束/重置时，`on_session_end()` 触发以进行清理和最终数据刷写

### Memory 刷写生命周期

显式对话边界（例如 `/new`、`/reset` 或 `/resume`）会刷新并结束原会话。空闲时间和每日时间边界不会结束会话。

TTL、LRU 和内存压力触发的资源缓存淘汰会先在释放 agent client 之前，把缓存的对话记录提交给已配置的 Memory Provider。它不会关闭持久化对话：下一轮会重新加载相同的对话记录和会话标识。

## 后台维护

Gateway 在处理消息的同时运行周期性维护任务：

- **Cron 计时** —— 检查任务计划并触发到期任务
- **会话维护** —— 回收缓存资源，但不结束会话记录
- **Memory 刷写** —— 在软释放缓存前提交记忆
- **缓存刷新** —— 刷新模型列表和提供者状态

## 进程管理

Gateway 作为长期运行进程运行，管理方式如下：

- `hermes gateway start` / `hermes gateway stop` —— 手动控制
- `systemctl`（Linux）或 `launchctl`（macOS）—— 服务管理
- PID 文件位于 `~/.hermes/gateway.pid` —— 面向 profile 的进程追踪

**Profile 范围 vs 全局**：`start_gateway()` 使用 profile 范围的 PID 文件。Standalone（每个 profile 一个 gateway）下，`hermes -p x gateway stop` 只停止该 profile 的 gateway。多路复用下每个 host 只有一个 gateway 进程，由启动它的那个 profile 拥有（`gateway/host_rendezvous.py` 发布其 PID、home 和 served set；`gateway/host_attach.py` 是每个生命周期动词都要经过的 attach/rescan/refuse 决策）：owner 上的 `hermes gateway stop` 会把所有 served profile 停下来，而 served secondary 的 `hermes -p x gateway stop` 会以 exit 78 拒绝（它没有自己的 gateway）。served profile 的第二个 `gateway run` 会 attach 并以 0 退出——在服务监督者下改为以 75（EX_TEMPFAIL）退出，这样冗余 unit 会被 RETRIED 而非停车：「现在有人在服务我」是一个会随该进程结束而结束的运行时观察，而 78（systemd、s6 和 launchd 都把它当作永久）会让该 profile 滞留。ATTACH 需要 owner 的一个活跃 `identify` 回答；一个背后没人应答的 rendezvous 记录证明了存在一个 owner，却永远无法证明它服务你，因此它产生一个瞬时拒绝（exit 75），而非 attach。一个对 rescan 回答 `multiplex: False` 的 owner 是另一个 profile 的 *standalone* gateway，而非把你排除在外的 multiplexer：该动词在该 gateway 旁边启动本 profile 自己的 gateway（一进程一 profile 拓扑），它并不拒绝——在那里拒绝会 exit 78 并把除抢到 host 锁的第一个之外的所有 launchd unit 都停住。`hermes gateway stop --all` 使用全局 `ps aux` 扫描来终止所有 gateway 进程（用于更新时）。存活性由 `gateway.status.live_gateway_pid_for_home`（PID + 启动时间指纹）决定，绝不只是 PID 存在。

## 后台进程通知

`terminal(background=true, notify_on_complete=true)` 会启动一个 gateway watcher，检测完成并触发一个新的 agent 回合。详细程度：`display.background_process_notifications`（或 `HERMES_BACKGROUND_NOTIFICATIONS`）：`concise`（默认；一行，失败时附一段输出尾部）、`all`（运行中更新 + 最终原始输出）、`result`（仅最终原始输出）、`error`（仅非零退出时的最终原始输出）、`off`。

watcher 在注册时被挂到 gateway loop 上（`terminal_tool_background.py::_register_completion_watcher` → `run_notifications.py::arm_process_watcher`）；`pending_watchers` 只是 gateway 服务之前注册的进程（checkpoint 恢复）或停止期间的 fallback，在启动和 post-turn 被排空。Agent-notify watcher 不发用户可见收据（agent 的下一回合就是报告），除非启动回合在退出时仍在运行——那样注入只会排一个 follow-up，因此简洁收据会立即发出。

空闲完成 watcher 也会排空 `watch_match` / `watch_disabled`；不需要用户 follow-up。Notify-off 排空这些而不唤醒。Transport 失败会重试；不可用的持久完成 owner/transport 不消耗投递尝试。Profile 命名空间的进程事件保留其 owning 适配器，包括原始进度/最终通知。适配器 acceptance 是至少一次 admission，而非一个模型回合或最终出站回复已完成的证明。API-server 异步完成仍是持久投递行，而不是自主的新模型回合。完成/watch 注入要求 `admit_internal_event` 收据：返回 None 的 handler 不是 acceptance。被拒 admission 会退回每个已认领的批量兄弟而不消耗一次尝试；实际投递错误保留其有界重试策略。已识别的原始 API 路由在持久化消息来源之后解析，并在不可用时静默延迟；格式错误的路由仍会警告。

Cron 执行有自己的 session。符合条件的可续投递可能镜像或播种 reply-facing 对话：来源、无来源的 home fallback、用户写的 bare-platform home，或选择加入的显式目标。`all` 展开不获得 home mirror 资格。镜像 brief 被标记为用户回合，在回合边界追加，保留角色交替（`cron/AGENTS.md`）。

## `/login`（off-turn，仅配对 DM）

`/login` 在 `hermes_cli/commands.py` 中注册为 `busy_policy="dispatch"`、`desktop="settings"`，列在 `run_busy.py::_PLAIN_COMMANDS`，由 `GatewayLoginCommandsMixin`（`gateway/slash_commands_login.py`）处理。它在配对 DM 之外拒绝：`chat_type in {"dm","private"}`、一个 truthy 的 `chat_id`，以及一个「dm」真的是配对对话的平台——ntfy、raft 和 a2a 都对广播话题、channel 和 agent 对端报告 `chat_type="dm"`，所以在那发布同意链接会把它公开出去。

**它绑定整个安装。** 登录写入单例 `providers.nous`，所以批准该码的人拥有此 gateway 所服务的每个 chat 的推理和 connector。slash 门控是选择加入的（`gateway/slash_access.py`）：没有设置 `allow_admin_from` 时，每个被允许 DM 该 bot 的用户都能运行它。共享 gateway 的 operator 必须设置它。

handler 立刻返回其 ack，并在一个**私有单 worker executor**（`_login_executor`）上 drain `anon_auth.run_sign_in`，绝不使用共享的 10 线程 gateway 池——一次 promotion 等待可能持续该码的整个有效期，而共享池中的一个活跃 worker 会让 shutdown 跳过 SessionDB close/checkpoint。每个进程一次尝试，stamp 上启动它的身份：同一身份的第二次 `/login` 取代前者（失败者以被取代的副本推入它自己的 chat 结束）；不同身份被拒绝。`task.cancel()` 无法中断 worker 线程内的阻塞轮询，因此 shutdown 和 supersede 都通过 `attempt.cancelled` 工作，而 `wait_for_promotion` 现在以 ≤1 s 的 tick 轮询它。一个被 shutdown 取消的尝试不推入任何东西——任务正在死亡，其适配器可能已经消失——但如果服务器已经完成转移，它仍会持久化，因为那次转移是不可逆的。

完成后 handler **驱逐** 每个仍位于 `nous/welcome` 的缓存 agent，并清除任何钉在其上的 session model override。它不会原地切换 agent，不写 model override：`settle_after_upgrade` 已经移动了 config 里的 `model.default`/`model.base_url`，每个回合都会重新解析 config 和凭据，而且 `_agent_config_signature` 已经在路由变化时强制重建——所以原地切换买不到缓存热度，一个 override 会钉住一个没有任何东西去刷新的过期访问 token。

## 多路复用 profile

在 `gateway.multiplex_profiles: true` 下，一个进程服务默认 profile 加上 `profiles/` 下（`hermes_cli/profiles.py::profiles_to_serve(multiplex=True)`）的每个活动目录。`os.environ` 和模块全局存着**launch** profile 的值，所以 secondary 的每个活动都显式绑定其 scope——一个 profile 是 home + secret scope + terminal scope 的组合：

| 活动 | 绑定 |
|---|---|
| 被路由回合 | `gateway/run.py::_profile_runtime_scope(home)`，经 `run_turn.py::_profile_scope_for_source` |
| Agent 释放 / 驱逐（TTL、LRU、内存压力） | `gateway/run_agent_cache.py::_run_release_in_profile_scope` |
| 关闭 | `gateway/run_shutdown.py::_finalize_session` |
| Post-turn 媒体投递 | `gateway/platforms/base.py::_media_delivery_scope` |
| Cron tick | `cron/scheduler_provider.py::_profile_cron_scope(home)`（一个 ticker，顺序遍历 profile） |
| 子进程（`hermes -p X` worker、relay turns、browser drivers） | `tools/environments/local.py::served_profile_child_env` |
| 后台线程 | `agent/memory_provider.py::spawn_context_thread` |

秘密读取只在 `set_multiplex_active(True)` 之后才 fail closed（`agent.secret_scope.get_secret` 抛 `UnscopedSecretError`），而 gateway、cron、`gateway migrate` 和 Desktop/dashboard `serve` 后端会设置它。多路复用下适配器 YAML 绝不写入 `os.environ`：`gateway/platforms/_shared.py::apply_yaml_bridge` 播种 `PlatformConfig.extra` 并在 secondary 的 scope 下跳过 environ 写入；门读取经 `platform_gate_env`。共享 ingress 平台（WhatsApp bridge、Relay）只在默认 profile 上运行；启用其中一个的 secondary 会被记录一次并 stamp 进运行时状态（`run_adapters.py::_note_unserved_secondary_platform`）。用户看到的逐 profile 隔离：见 [多 profile 网关 § 每个 profile 隔离了什么](../user-guide/multi-profile-gateways.md#what-is-isolated-per-profile)。

## 运行中插件加载

在适配器连接之后才加载的插件（从 CLI、Desktop、dashboard 或 `plugins.manage` install/enable；工具触发的强制重新发现）会无重启地重新接线其平台处理器（#87770）。所有部件都在 `gateway/run_plugin_rewire.py`：

- **发现监听器** —— `_start_recover_previous_run` 为 launch profile 订阅 `PluginManager.on_plugin_loaded`，`_load_secondary_profile_config` 对每个 served profile 做同样的事。该事件从 `discover_and_load` 内部触发（绝不从 RPC），针对新加载的插件；回调用 `call_soon_threadsafe` 跳到 gateway loop 上。
- **幂等重接线** —— `BasePlatformAdapter.rewire_plugin_handlers()` 重读 `get_platform_handler_factories(platform)`，并只运行尚未在 live 原生客户端上接线的工厂（以 `(plugin, qualname)` 为 key，因为强制重载会返回新的函数对象）。Telegram 把新增处理器提升到 core 的 catch-all 之前；Slack 也会对每个 `AsyncApp` 注册一次缺失的 `register_slack_action_handler` 回调。
- **`reload-plugins` 控制动词** —— 其他进程（`hermes plugins install`、`hermes serve`）让运行中的 gateway 强制重扫所请求的（已服务）home；回答携带 `plugins`、逐个插件的 `activations` 和 `adapters_rewired`，让调用方能如实说「现在已激活」。
- **作用域限制** —— 仅处理器。晚到插件的 tools 和 system-prompt 段等待下一个 session（prompt-cache 不变式）；portable MCP server 等待 `mcp.reload`。禁用时不解除任何接线。

## 相关文档

- [会话存储](./session-storage.md)
- [Cron 内部机制](./cron-internals.md)
- [ACP 内部机制](./acp-internals.md)
- [Agent 循环内部机制](./agent-loop.md)
- [消息 Gateway（用户指南）](../user-guide/messaging/index.md)