# 会话存储

Hermes Agent 使用 SQLite 数据库（`~/.hermes/state.db`）跨 CLI 和 gateway 会话持久化会话元数据、完整消息历史及模型配置。这替代了早期的逐会话 JSONL 文件方案。

该 profile 的 gateway 是**规范会话的唯一写入者**：`hermes chat`、TUI、Desktop（本地）、API server、ACP、cron、Bot Chat 和 Kanban worker 都是把输入提交给它那个会话权威（session authority）的客户端（通过 `/api/ws`，见 [Gateway 本地会话](./gateway-local-sessions.md)）。进程外的托管 worker 通过那个 owner（`RuntimeSessionStore` RPC）写入，而绝不自行打开 `state.db` 来写 transcript。

源文件：`hermes_state.py`（facade）加上 `hermes_state_*.py` 兄弟模块（schema、fts、search、compression、portability、gateway 等）。

## Hermes home 与 profile 隔离

`get_hermes_home()` 是状态与配置的权威文件系统解析器。它依次使用上下文本地覆盖、`HERMES_HOME` 环境变量，最后才是平台默认值（macOS 和 Linux 上为 `~/.hermes`；Windows 上为 `%LOCALAPPDATA%/hermes`）。因此默认数据库始终是 `get_hermes_home() / "state.db"`，调用方不应硬编码为 `~/.hermes/state.db`。

命名 profile 是相互隔离的目录：例如名为 `coder` 的 profile 使用 `<默认 Hermes 根目录>/profiles/coder/`，拥有自己的 `state.db`、配置、日志和其他 profile 级状态。为某个 profile 创建数据库、读取配置或启动子进程的进程必须保留或传递该 profile 的 `HERMES_HOME`；回退到默认根目录会把错误 profile 的状态混入操作中。

CLI 启动流程在导入 Hermes 其余部分之前调用 `_apply_profile_override()`。显式的 `--profile`/`-p` 会解析该 profile 并把解析后的目录写入 `HERMES_HOME`。没有显式选择器时，指向 profile 的 `HERMES_HOME` 会被保留；否则启动流程可以使用默认根目录中的 active-profile 选择。`HOME` 只决定在没有上下文覆盖和 `HERMES_HOME` 时使用的平台默认值。修改 `HOME` 不是选择命名 profile 的安全方式。特别地，丢弃了 `HERMES_HOME` 的子进程即使在另一个 profile 处于激活状态时也可能回退到默认 profile，因此子进程的启动方应显式传递 `HERMES_HOME`。

`display_hermes_home()` 仅用于面向用户的文本。它在可能时把解析出的 home 格式化为相对用户主目录的形式（例如 `~/.hermes/profiles/coder`），并不提供另一套解析规则。

### 测试隔离守卫

测试必须使用临时 `HERMES_HOME` 或显式的临时数据库路径。实时系统守卫会在测试上下文进程打开真实默认 Hermes 根目录或真实命名 profile 下的生产 `state.db` 之前抛出异常，防止 fixture 数据或 SQLite side effect 触及正在使用的安装。

`HERMES_STATE_DB_GUARD_BYPASS=1` 是仅供测试使用的逃生口，用于确实必须访问实时数据库的派生子进程。进程内的等价逃生口是 `@pytest.mark.live_system_guard_bypass`。不要在普通 Hermes 命令、开发 shell 或应用配置中设置任一绕过：它会禁用保护实时会话历史的守卫（一个硬性的 `RuntimeError`），而导出了它的 shell 会把该绕过传给之后的每一次 pytest 运行。

### Desktop profile 隔离与压缩世代（generation）

每个命名 profile 都把其 transcript 存在自己的 `$HERMES_HOME/state.db` 中，包括一个进程（多路复用 gateway，或 `hermes serve`）服务多个 profile 的情况。会话内的 agent 重建（Bot Chat 能力刷新和 `tools.configure`）必须在构造期间保留该会话的数据库句柄并绑定其 profile home。释放被替换掉的 agent 时，不得关闭由后继者继承的句柄。`tools.configure` 从活动会话的 `profile_home` 解析配置，即使客户端只提供了 `session_id`。重建会先在分配后继者之前准备好模型配置，然后一起安装 agent 并转移所有权；准备失败时，由现有 agent 负责清理。无法解析或目录已消失的显式 profile，会在访问启动配置或历史之前就失败。过期的 `tools.configure` 会话 ID 同样返回 `session not found` 而不改配置；省略会话 ID 仍支持全局设置操作。

原地压缩（in-place compaction）把旧行归档为 `active=0`，并把保留的上下文插入为 `active=1` 行。因此一条受保护的消息可以合法地同时出现在两个世代中，内容与时间戳完全相同。不要把那些归档行当作重复行删除。诊断重复的*活跃*写入时使用 `active=1`，并在排查看似「回退」的历史时既检查数据库的 profile 也检查会话 ID。

## Codex app-server 输入归属

agent 在开始 Codex 回合之前就持久化一条已接受的用户输入。Codex 随后把该输入投射为一个前导的 `userMessage` 通知。在运行时拼接边界，Hermes 只排除那个前导项——当它恰好与序列化进 `turn/start` 的文本（包括富输入强制转换）完全匹配时。更晚的或不匹配的用户事件保持不变，单独接受的相同回合也保持不变。这同样适用于合成的/无 key 的输入；它不依赖平台消息 ID。既有历史中的重复不会被改写。当 agent 报告它自己拥有持久化权时，gateway 跳过其 transcript 写入。

## Gateway 异常路径的输入归属

gateway 异常可能发生在 agent 构造之前，或在其输入到达 SQLite 之后。gateway 在既有的 `display_metadata` 侧车（sidecar）里给已接受的输入一个 owner 标记，并让它走 agent 正常的持久化路径。provider 消息从不包含这个元数据。平台标记按平台、profile、scope、chat 和 thread 给入站消息 ID 命名空间化；原始的 `platform_message_id` 保持不变，用于 quote/reply 解析。无 key 的回合会获得一个新的标记，即便文本和时间戳相同。

异常写入器只探测那个标记，顺着已发布的 reroute 与规范的活跃压缩后继者，然后是压缩祖先。活跃行和压缩归档都计数；undone 行、被观察到的输入以及无关写入器则不计。一个正在写同一会话的无关进程无法压制这个回合。不需要全历史基线或归档消息体分配。failed 的归属读取不会授权一次投机性的追加；普通的历史读取失败保留既有的「历史不可用」响应。

正常的 agent 自有持久化保持不变。这是失败写入器的仲裁，而非通用的 exactly-once 投递、内容去重或 schema 迁移。历史行不会被改写；未标记的历史输入无法为一次重新投递的事件确立归属。

## 架构概览

```
~/.hermes/state.db (SQLite, WAL mode)
├── sessions              — 会话元数据、token 计数、计费信息
├── messages              — 每个会话的完整消息历史
├── session_model_usage   — 按模型/按任务的用量归属行
├── messages_fts          — FTS5 虚拟表（content + tool_name + tool_calls）
├── messages_fts_trigram  — 使用 trigram tokenizer 的 FTS5 虚拟表（CJK / 子串搜索）
├── messages_fts_cjk      — 使用 cjk_unicode61 tokenizer 的 FTS5 虚拟表
├── state_meta            — 键值元数据表
├── gateway_routing       — Gateway 路由元数据
├── compression_locks     — 跨进程压缩加锁
├── async_delegations     — 异步委派记账
├── delivery_obligations  — Gateway 发件箱（欠下的回复）；由 gateway/delivery_ledger.py 惰性创建
├── runtime_epoch         — 会话权威的 owner epoch + 实例 id（单行）
├── session_admissions    — 每个会话的持久化输入 FIFO（queued → started → terminal；owner 丢失时为 unknown）
├── worker_executions     — 绑定到会话和 epoch 的托管 worker/child/cron/kanban 执行
├── worker_receipts       — 托管 worker 报告的、有序且经 digest 校验的结果
└── schema_version        — 单行表，跟踪迁移状态
```

`session_admissions`、`worker_executions` 和 `worker_receipts` 通过 `ON DELETE RESTRICT` 引用 `sessions`：删除或修剪一个会话会在同一事务里 retire 其 terminal 账本行，并在存在进行中的工作或 `unknown` 时拒绝。一个在 owner 死亡时处于 `started` 的输入会在下一个 owner 启动时变为 `unknown` 且绝不重放；它后面的输入会等待，直到它被解决（`prompt.resolve_unknown`）。

`hermes sessions recover` 把上面承载行的表复制进恢复后的数据库（FTS 索引和 `schema_version` 会被重新生成），包括源那里存在的、惰性创建的 `delivery_obligations` 账本——其行数会像 `sessions`/`messages` 那样被校验。四个运行时账本表（`runtime_epoch`、`session_admissions`、`worker_executions`、`worker_receipts`）会被盘点并报告为「排除在外」而非复制：一个部分的账本可能重放已启动的 side effect，或领养另一个 owner 的 worker。完整的 `hermes backup` 恢复则为启动对账保留完整账本。

关键设计决策：
- **WAL 模式**：支持并发读取 + 单写入者（gateway 多平台）
- **FTS5 虚拟表**：跨所有会话消息的快速全文搜索
- **会话血缘**：通过 `parent_session_id` 链实现（压缩触发的会话分割）
- **来源标记**（`cli`、`telegram`、`discord` 等）：用于平台过滤
- 批量运行器和 RL 轨迹不存储于此（独立系统）

## SQLite Schema

### Sessions 表

节选 —— 完整的当前列清单见 `hermes_state_common.py` 中的 `SCHEMA_SQL`（由 `hermes_state_schema.py` 应用），其中还包括 gateway 路由元数据，例如 `session_key`、`chat_id`、`chat_type`、`thread_id`、`display_name`、`origin_json`、`expiry_finalized`，工作区字段 `cwd` / `git_branch` / `git_repo_root`，handoff 与压缩失败字段，`profile_name`、`transport_profile`（接收该通道的多路复用 bot，可为空）、`rewind_count`、`archived`、`auto_archived`（仅由空闲清扫设置；恢复或压缩续接会清除一个仅由清扫产生的归档，而绝不清除一个刻意归档）以及 `pinned`）：

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    -- ... 其他 gateway/workspace/handoff/compression 列 ...
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique
    ON sessions(title) WHERE title IS NOT NULL;
```

`user_id` 是会话另一端的主体：消息适配器存平台发送者 id，而通过已认证的 `hermes serve`（OAuth 或 basic username/password provider）打开的 `desktop` / dashboard 会话存登录名 `<provider>:<user id>`（例如 `basic:alice`）。背后没有人的会话——匿名 loopback 使用、`subagent`、`cron`、`kanban`——保持它为空。该值在行创建时设置，之后不再推断。

### Messages 表

节选 —— 完整 schema 还包括 `effect_disposition`、`platform_message_id`、`observed`、`active`、`compacted`、`api_content`、`display_kind`、`display_metadata`、`message_uid`、`absorbed_message_uids`、`tool_call_uids` 和 `tool_call_uid`：

```sql
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT
    -- ... 其他 display/compaction 列 ...
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
```

说明：
- `tool_calls` 以 JSON 字符串存储（序列化的 tool call 对象列表）
- `reasoning_details`、`codex_reasoning_items` 和 `codex_message_items` 以 JSON 字符串存储
- `reasoning`、`reasoning_content` 和 `reasoning_details` 始终保留在历史中。在 chat-completions 线协议上，每个 assistant 回合都会在路由读取的载体上重放其已存储的 reasoning（`agent/message_sanitization.py::route_reasoning_carriers`）：默认在所有地方重放 `reasoning_content`；在 OpenRouter 格式的 gateway（OpenRouter、Kilo、Vercel AI Gateway、Nous Portal）上额外重放 `reasoning` 和 `reasoning_details`；在 Novita、MiniMax `/v1` 和 TokenHub 上重放 `reasoning_details`；在本地服务器、Ollama 和 Upstage 上重放 `reasoning`。严格 schema（Mistral、Groq、Cerebras、SambaNova）不携带 reasoning 键，Fireworks 只携带 `reasoning_content`。只有 DeepSeek、Kimi 和 MiMo 会用 `" "` 填充一个不带 reasoning 的工具调用回合。当某个路由按名称拒绝一个字段（"Extra inputs are not permitted, field: messages[N].reasoning_details"）时，Hermes 会为该 provider/host/model 记录它，剥离该字段并重试一次。该记录在会话期间有效，且绝不触碰历史。
- Desktop 历史回填在 REST 和 JSON-RPC 投影（`session.resume`、`session.activate`、`session.history`）中都保留 assistant 侧车，包括带 reasoning 和工具调用的行。REST 可能返回 SQLite 的 JSON 字符串，而 RPC 返回解码后的项；Desktop 两者都能接受。一个最终的 Responses 回复可能只存在于 `codex_message_items` 中而 `content` 为空。规范内容（canonical content）仍优先，analysis/commentary 项不会被提升为回复文本。
- `reasoning` 存储提供商暴露的原始推理文本
- 一个仅 reasoning 的干净停止（`content` 为空、`finish_reason=stop`、reasoning 存在）会用 reasoning 文本来回答，但 assistant 行绝不会把该文本写成 `content`：`content` 保持为空，文本留在 `reasoning`/`reasoning_content`，而 `api_content` 承载它，使下一次请求能字节一致地重放该回答。因此历史界面会把它显示为 reasoning，而非回复。
- `api_content` 是一个字节保真的侧车：本条消息实际发给 API 的精确内容字符串，当其与 `content` 不同时（临时记忆/插件注入、持久化覆盖）使用。它保存线协议字节以供 prompt-cache 稳定的重放——按发送原样存储，唯一的例外是孤立代理项（lone surrogate），sqlite3 无法绑定它们，而会话循环反正会从每个出站 payload 中清理它们。`NULL` 意味着 `content` 是逐字发送的。
- 时间戳为 Unix epoch 浮点数（`time.time()`）
- `message_uid` 是持久的逐消息 id（32 位十六进制，`uuid4().hex`）：在一行首次插入时铸造一次，并被每个克隆复制（原地压缩世代、轮换子会话、并发尾部克隆、`replace_messages` 重发、导出/导入），因此一条逻辑消息在物理 `id` 变化时仍保持一个 uid。按行寻址的重写不会动它。它在每个投影上被恢复（`get_messages_as_conversation` 无条件设置它；`_row_id` 保持可选），并从 provider 请求中被剥离。上下文引擎用它来为每条消息的状态做 key——见 [Context Engine 插件](./context-engine-plugin.md#stable-message-identity-message_uid)。
- `absorbed_message_uids` 是合并见证：一个 JSON 列表，记录宿主折叠进本行的 `message_uid`（交替修复中 user 和 assistant 的合并、压缩器的在途重述与锚点折叠、微压缩的相邻 user 合并；合成行保留第一个组成部分的 uid）。在幸存行刷写时写入，随它一起重写，并恢复为活跃的 `_absorbed_message_uids`；未吸收任何内容的行为 `NULL`。
- `tool_call_uids`（assistant 行）是一个 JSON `{tool_call_id: uid}` 映射，给 `tool_calls` 的每个条目一个逐出现次数的 id，因为 provider 的 tool-call id 会重复（一个响应内重复同一 id 的调用共享其 uid；两个 assistant 回合被折叠后，一个同时命中的 id 映射为一个 uid 列表，按调用顺序每次出现一个）；`tool_call_uid`（tool 行）是结果的匹配值。`tool_calls` JSON 本身从不被修改。在 assistant 行首次插入时铸造，在刷写时配对到结果上，或在列为 `NULL` 时从恢复时所依据的前一条 assistant 行派生；恢复为活跃的 `_tool_call_uids` / `_tool_call_uid`。

### FTS5 全文搜索

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);
```

FTS5 表通过三个触发器与 `messages` 表保持同步，分别在 INSERT、UPDATE 和 DELETE 时触发。当前触发器被 `state_meta` 中的 `fts_rebuild_high_water` / `fts_rebuild_progress` 标记门控（这样后台 FTS 重建就能继续而不重复索引），并覆盖全部三个被索引列——精确 SQL 见 `hermes_state_common.py` 中的 `SCHEMA_SQL`。

## Dashboard 浏览与 owner 维护

会话浏览以只读方式打开 `state.db`。GET 列取、搜索、详情、消息、统计和导出绝不创建缺失的数据库、隔离一个零字节文件、对账过期 schema 或触发自动归档。未初始化或不兼容的存储返回 HTTP 503，并附带 owner-start/recovery 指引；profile 聚合会在其逐 profile 的 `errors` 列表中报告 schema 失败。

该 profile 的 gateway 在启动时初始化并对账其存储。独立的 `serve` 后端保留其现有的启动对账 worker。已配置的 `sessions.auto_archive` 在 gateway 启动和例行维护期间（或在独立后端的维护 ticker 中）保持活跃，独立于浏览。显式变更端点保留其既有写入行为。

SQLite WAL 读取者仍参与共享内存协调：`mode=ro` 可能创建空的 WAL/SHM 侧车文件，或更新 SHM 读取标记。只读契约禁止规范的 schema/数据写入、修复和 checkpoint；它并不使活跃的 SQLite 协调文件变得不可变。

## Schema 版本与迁移

当前 schema 版本：**32**

`schema_version` 表存储单个整数。简单的列添加由 `_reconcile_columns()` 声明式处理（对比实时列与 `SCHEMA_SQL` 并 ADD 缺失列）。版本门控链保留用于无法声明式表达的数据迁移及索引/FTS 变更：

| 版本 | 变更 |
|------|------|
| 1 | 初始 schema（sessions、messages、FTS5） |
| 2 | 向 messages 添加 `finish_reason` 列 |
| 3 | 向 sessions 添加 `title` 列 |
| 4 | 在 `title` 上添加唯一索引（允许 NULL，非 NULL 必须唯一） |
| 5 | 添加计费列：`cache_read_tokens`、`cache_write_tokens`、`reasoning_tokens`、`billing_provider`、`billing_base_url`、`billing_mode`、`estimated_cost_usd`、`actual_cost_usd`、`cost_status`、`cost_source`、`pricing_version` |
| 6 | 向 messages 添加推理列：`reasoning`、`reasoning_details`、`codex_reasoning_items` |
| 7 | 向 messages 添加 `reasoning_content` 列 |
| 8 | 向 sessions 添加 `api_call_count` 列 |
| 9 | 向 messages 添加 `codex_message_items` 列，用于 Codex Responses 消息 id/phase 重放 |
| 10 | 添加 `messages_fts_trigram` 虚拟表（trigram tokenizer，用于 CJK / 子串搜索）并回填现有行 |
| 11 | 重新索引 `messages_fts` 和 `messages_fts_trigram` 以覆盖 `tool_name` + `tool_calls`，从外部内容模式切换为内联模式；删除旧触发器并回填所有消息行 |
| 16 | 在 `model_config` 中标记 delegate 子 agent 行（`$._delegate_from`），使会话选择器在父级删除而使其成为孤儿后保持干净 |
| 18 | Gateway 元数据整合——从 `sessions.json` 回填 `display_name` / `origin_json` / `expiry_finalized` |
| 20 | 按模型用量归属——从历史的逐会话聚合总量播种 `session_model_usage` 行 |
| 22 | 任务维度用量归属——重建 `session_model_usage` 使 `task` 列参与 PRIMARY KEY |
| 23 | FTS 存储重设计——外部内容 FTS 表替换 v11 内联模式副本（存量 DB 可选迁移） |
| 29 | Cron 会话离开 trigram（子串/CJK）索引；`messages_fts_trigram_src` 视图 + 触发器按 `sessions.source` 过滤，一次性重建清除历史行 |
| 30 | Delegate-child（子 agent）会话也离开 trigram 索引——`source='subagent'` 或 `$._delegate_from` 标记（`FTS_TRIGRAM_SESSION_SQL`）。行仍留在 `messages` 和标准 `messages_fts` 词索引中，`session_search` 仍能找到它们；只有约 2.6× 的 trigram 影子表缩小。与 v29 相同的一次性重建 |
| 31 | `messages.message_uid`、`absorbed_message_uids`、`tool_call_uids`、`tool_call_uid`（声明式列添加），为早于该列的每一行回填一个新的 `message_uid`（2,000-id 分块，每块各自一次短写入；一次打开在其首个分块后约花费一秒，并在下次打开时从 `message_uid_backfill_id` 元游标续接，直到 `message_uid_backfill` 标记结束它，这样大存储就绝不会为整张表持有写锁），以及 `messages_message_uid_insert` 触发器，为任何不带 id 插入的行（旧版本写入存储时）铸造一个，从而使 v31 存储能把每行都带着 id 交给消费者（tool-call uid 在行下一次被写时铸造；结果在恢复时配对） |

未在表中列出的版本都是 `_reconcile_columns()` 处理的声明式列添加（仅版本号递增，无数据迁移）。

声明式列添加使用 `ALTER TABLE ADD COLUMN`，包裹在 try/except 中以处理列已存在的情况（幂等）。每个成功的迁移块完成后版本号递增。

## 写入竞争处理

仍会有多个进程打开同一个 `state.db`：gateway（规范会话写入者）、只读浏览器（dashboard、`hermes sessions list`）、维护命令，以及非会话表的写入者。`SessionDB` 类通过以下方式处理写入竞争：

- **短 SQLite 超时**（1 秒），而非默认的 30 秒
- **带时间预算的应用层重试**，加随机抖动（前 2 秒为 20-150ms，之后为 250ms-1s）：常规写入 20s，transcript 写入 60s（其失败会中止回合），仅观察性质的 activity 写入 0.5s
- **BEGIN IMMEDIATE** 事务，在事务开始时暴露锁竞争
- **定期 WAL checkpoint**，每 50 次成功写入执行一次（PASSIVE 模式）

这避免了「护卫效应」（convoy effect）——SQLite 确定性内部退避会导致所有竞争写入者在相同间隔重试。

```
_WRITE_PATIENCE_S, _TRANSCRIPT_WRITE_PATIENCE_S, _ACTIVITY_WRITE_PATIENCE_S = 20.0, 60.0, 0.5
_WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S = 0.020, 0.150
_WRITE_RETRY_SLOW_MIN_S, _WRITE_RETRY_SLOW_MAX_S = 0.250, 1.000
_CHECKPOINT_EVERY_N_WRITES = 50
```

当一个写入者耗尽预算时，该回合以 `session_persistence_failed:locked` 结束，并且在 Linux 上，`hermes_state_lockowners` 会记录一条 WARNING，指明那一刻持有锁的进程（`PID 594094 (hermes --worktree --yolo) holds WAL write lock on state.db-shm`），从 `/proc/locks` 读取——SQLite 的字节范围 `fcntl` 锁把锁类型编码进其 offset（`state.db-shm` 字节 120 = WAL write、121 = checkpoint、123-127 = read slots；`state.db` 上 1 GiB 的 pending-byte 页 = rollback-journal PENDING/RESERVED/SHARED）。打开描述符的扫描无法做出这种区分，因为每个 Hermes 进程都打开了 DB。在 `database is locked` 失败旁边的 `~/.hermes/logs/errors.log` 里找那一行。

锁竞争由 SQLite 结果码（`SQLITE_BUSY` / `SQLITE_LOCKED`，`hermes_state_errors.is_sqlite_lock_error`）识别，而非按消息文本。在 rollback-journal（`delete`）模式下，在 FTS5 的表构造器内部丢失的锁会以 `SQLITE_BUSY` 呈现，文本为 `vtable constructor failed: messages_fts`；它被当作 `database is locked` 处理。打开一个可写的 `SessionDB` 最多等待 `_WRITE_PATIENCE_S`；只读打开只等待一次其 `_READ_BUSY_TIMEOUT_S`（5 s）读取预算。如果锁持续超过该预算，dashboard 返回 503（busy），而非 500。

## 常用操作

### 初始化

```python
from hermes_state import SessionDB

db = SessionDB()                           # 默认：~/.hermes/state.db
db = SessionDB(db_path=Path("~/.hermes/cache/scratch/test.db").expanduser())  # 自定义路径
```

### 创建和管理会话

```python
# 创建新会话
db.create_session(
    session_id="sess_abc123",
    source="cli",
    model="anthropic/claude-sonnet-4.6",
    user_id="user_1",
    parent_session_id=None,  # 或用于血缘追踪的上一个会话 ID
)

# 结束会话
db.end_session("sess_abc123", end_reason="user_exit")

# 重新打开会话（清除 ended_at/end_reason）
db.reopen_session("sess_abc123")
```

### 存储消息

```python
msg_id = db.append_message(
    session_id="sess_abc123",
    role="assistant",
    content="Here's the answer...",
    tool_calls=[{"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}],
    token_count=150,
    finish_reason="stop",
    reasoning="Let me think about this...",
)
```

### 检索消息

```python
# 包含所有元数据的原始消息
messages = db.get_messages("sess_abc123")

# OpenAI 对话格式（用于 API 重放）
conversation = db.get_messages_as_conversation("sess_abc123")
# 返回：[{"role": "user", "content": "..."}, {"role": "assistant", ...}]
```

### 会话标题

```python
# 设置标题（非 NULL 标题中必须唯一）
db.set_session_title("sess_abc123", "Fix Docker Build")

# 按标题解析（返回血缘中最新的）
session_id = db.resolve_session_by_title("Fix Docker Build")

# 自动生成血缘中的下一个标题
next_title = db.get_next_title_in_lineage("Fix Docker Build")
# 返回："Fix Docker Build #2"
```

## 全文搜索

`search_messages()` 方法支持 FTS5 查询语法，并自动对用户输入进行清理。

### 基本搜索

```python
results = db.search_messages("docker deployment")
```

### FTS5 查询语法

| 语法 | 示例 | 含义 |
|------|------|------|
| 关键词 | `docker deployment` | 两个词均包含（隐式 AND） |
| 引号短语 | `"exact phrase"` | 精确短语匹配 |
| 布尔 OR | `docker OR kubernetes` | 任一词 |
| 布尔 NOT | `python NOT java` | 排除词 |
| 前缀 | `deploy*` | 前缀匹配 |

### 过滤搜索

```python
# 仅搜索 CLI 会话
results = db.search_messages("error", source_filter=["cli"])

# 排除 gateway 会话
results = db.search_messages("bug", exclude_sources=["telegram", "discord"])

# 仅搜索用户消息
results = db.search_messages("help", role_filter=["user"])
```

### 搜索结果格式

每条结果包含：
- `id`、`session_id`、`role`、`timestamp`
- `snippet` — FTS5 生成的片段，带 `>>>match<<<` 标记
- `context` — 匹配前后各 1 条消息（内容截断至 200 字符）
- `source`、`model`、`session_started` — 来自父会话

`_sanitize_fts5_query()` 方法处理边缘情况：
- 去除不匹配的引号和特殊字符
- 将含连字符的词包裹在引号中（`chat-send` → `"chat-send"`）
- 移除悬空的布尔运算符（`hello AND` → `hello`）

## 会话血缘

会话可通过 `parent_session_id` 形成链。这发生在 gateway 中上下文压缩触发会话分割时。

### 查询：查找会话血缘

```sql
-- 查找会话的所有祖先
WITH RECURSIVE lineage AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN lineage l ON s.id = l.parent_session_id
)
SELECT id, title, started_at, parent_session_id FROM lineage;

-- 查找会话的所有后代
WITH RECURSIVE descendants AS (
    SELECT * FROM sessions WHERE id = ?
    UNION ALL
    SELECT s.* FROM sessions s
    JOIN descendants d ON s.parent_session_id = d.id
)
SELECT id, title, started_at FROM descendants;
```

### 查询：带预览的最近会话

```sql
SELECT s.*,
    COALESCE(
        (SELECT SUBSTR(m.content, 1, 63)
         FROM messages m
         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
         ORDER BY m.timestamp, m.id LIMIT 1),
        ''
    ) AS preview,
    COALESCE(
        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
        s.started_at
    ) AS last_active
FROM sessions s
ORDER BY s.started_at DESC
LIMIT 20;
```

### 查询：Token 使用统计

```sql
-- 按模型统计总 token 数
SELECT model,
       COUNT(*) as session_count,
       SUM(input_tokens) as total_input,
       SUM(output_tokens) as total_output,
       SUM(estimated_cost_usd) as total_cost
FROM sessions
WHERE model IS NOT NULL
GROUP BY model
ORDER BY total_cost DESC;

-- token 使用量最高的会话
SELECT id, title, model, input_tokens + output_tokens AS total_tokens,
       estimated_cost_usd
FROM sessions
ORDER BY total_tokens DESC
LIMIT 10;
```

## 导出与清理

```python
# 导出单个会话及其消息
data = db.export_session("sess_abc123")

# 导出所有会话（含消息）为字典列表
all_data = db.export_all(source="cli")

# 删除旧会话（仅删除已结束的会话）
deleted_count = db.prune_sessions(older_than_days=90)
deleted_count = db.prune_sessions(older_than_days=30, source="telegram")

# 清除消息但保留会话记录
db.clear_messages("sess_abc123")

# 删除会话及所有消息
db.delete_session("sess_abc123")
```

## 数据库位置

默认路径：`get_hermes_home() / "state.db"` —— 默认 profile 为 `~/.hermes/state.db`，命名 profile 为 `~/.hermes/profiles/<name>/state.db`，或 `HERMES_HOME` 指向的任何地方（见 [Hermes home 与 profile 隔离](#hermes-home-与-profile-隔离)）。

数据库文件、WAL 文件（`state.db-wal`）和共享内存文件（`state.db-shm`）均创建于同一目录。