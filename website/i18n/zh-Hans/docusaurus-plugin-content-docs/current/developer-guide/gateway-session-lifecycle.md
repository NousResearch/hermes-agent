---
title: "Gateway Session 生命周期"
description: "Gateway 中的 SessionSource、SessionEntry、SessionStore、session-key 规则以及多用户隔离"
---

# Session 生命周期

> **面向读者：** Gateway 开发者和维护者
> **源文件：** `gateway/session.py`（约 1200 行 + `session_*.py` 兄弟模块）、`gateway/run.py`（约 5500 行 facade + `run_*.py` 阶段）、`gateway/config.py`
> **最后更新：** 2026-06-16

## 概述

一个 **session** 表示 agent 与某个消息平台上一个或多个用户之间的持续对话。session 生命周期决定了对话何时持久化、何时重置、如何在 gateway 重启后存活，以及并发操作期间消息如何排队。

session 系统主要分布在两个模块中：

- `gateway/session.py` — 数据模型（`SessionSource`、`SessionEntry`、`SessionContext`）、key 生成（`build_session_key`）以及主存储（`SessionStore`）。
- `gateway/run.py` — Gateway 运行器（`GatewayRunner`）facade，将 session 接入消息处理流水线；各阶段位于 `run_*.py` 兄弟模块中：session 例行维护（`run_watchers.py`）、agent 缓存（`run_agent_cache.py`）、重启恢复（`session_recovery.py`）以及消息排队（`run_busy.py`）。

---

## 1. SessionSource — 消息来源描述符

`SessionSource` 是一个冻结的记录，用于描述**一条消息来自哪里**。它被附加到每条传入的 `MessageEvent` 上，用于路由、隔离和上下文注入。

### 字段

| 字段 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `platform` | `Platform` | *（必需）* | 标识消息平台的枚举（telegram、discord、slack、signal、whatsapp、matrix、local 等）。 |
| `chat_id` | `str` | *（必需）* | 平台级别的 chat/group/channel 标识符。经由适配器的 `chat_id_key` 转换进行路由。 |
| `chat_name` | `Optional[str]` | `None` | chat 或 group 的人类可读名称。 |
| `chat_type` | `str` | `"dm"` | `"dm"`、`"group"`、`"channel"`、`"thread"` 之一。控制 session key 的生成和隔离。 |
| `user_id` | `Optional[str]` | `None` | 平台特定的用户标识符。用于授权和按用户隔离 session。 |
| `user_name` | `Optional[str]` | `None` | 消息作者的显示名称。被注入到 system prompt 中。 |
| `thread_id` | `Optional[str]` | `None` | 论坛话题 / Discord thread / Slack thread 的标识符。用于区分线程化对话。 |
| `chat_topic` | `Optional[str]` | `None` | channel 话题或描述（Discord channel topic、Slack channel purpose）。 |
| `user_id_alt` | `Optional[str]` | `None` | 平台特定的稳定备用 ID（Signal UUID、飞书 union_id）。在 `user_id` 是临时性的情况下使用。 |
| `chat_id_alt` | `Optional[str]` | `None` | Signal group 内部 ID——将 Signal group V2 标识符映射到其规范形式。 |
| `is_bot` | `bool` | `False` | 当消息作者是 bot 或 webhook（Discord bots）时为 True。 |
| `guild_id` | `Optional[str]` | `None` | Discord guild / Slack workspace / Matrix server 作用域标识符。 |
| `parent_chat_id` | `Optional[str]` | `None` | 当 `chat_id` 指向一个 thread 时的父 channel。 |
| `message_id` | `Optional[str]` | `None` | 触发消息的 ID。用于 pin/reply/react 操作以及 Discord ID 注入（注入的 `[Triggering message id: …]` 提示仅随发送给 API 的那条消息传递；持久化的用户行保留原始文本）。 |
| `role_authorized` | `bool` | `False` | 当适配器通过平台角色（而非单个用户 ID）授予访问权限时为 True。 |

### 关键方法

- **`description`**（property：`str`）— 人类可读摘要，例如 `"DM with Alice"`、`"group: My Group, thread: 12345"`。
- **`to_dict()` / `from_dict()`** — 序列化往返，用于持久化到 `sessions.json`。

---

## 2. SessionEntry — 活动 session 记录

`SessionEntry` 是每个 session 的元数据记录，存储在内存中并持久化到 `{sessions_dir}/sessions.json`。每条记录将一个 `session_key` 映射到其当前的 `session_id`。

### 字段

| 字段 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `session_key` | `str` | *（必需）* | 标识对话通道的确定性 key（见 §4）。 |
| `session_id` | `str` | *（必需）* | 该特定对话实例的唯一标识符。格式：`YYYYMMDD_HHMMSS_<8hex>`。 |
| `created_at` | `datetime` | *（必需）* | 该 session 实例何时被创建。 |
| `updated_at` | `datetime` | *（必需）* | 最后一次活动时间戳，用于资源例行维护。 |
| `origin` | `Optional[SessionSource]` | `None` | 创建此 session 的来源，用于投递路由。 |
| `display_name` | `Optional[str]` | `None` | chat 显示名称（取自 `SessionSource.chat_name`）。 |
| `platform` | `Optional[Platform]` | `None` | 平台枚举，在重启之间持久化以用于路由。 |
| `chat_type` | `str` | `"dm"` | chat 类型，同样持久化以用于策略查询。 |
| `input_tokens` | `int` | `0` | 累计消耗的 LLM 输入（prompt）token。 |
| `output_tokens` | `int` | `0` | 累计消耗的 LLM 输出（completion）token。 |
| `cache_read_tokens` | `int` | `0` | 累计的 prompt cache 读 token。 |
| `cache_write_tokens` | `int` | `0` | 累计的 prompt cache 写 token。 |
| `total_tokens` | `int` | `0` | 所有回合的 token 总数。 |
| `estimated_cost_usd` | `float` | `0.0` | 估计的累计 USD 成本。 |
| `cost_status` | `str` | `"unknown"` | 成本跟踪状态标签。 |
| `last_prompt_tokens` | `int` | `0` | API 最后一次报告的 prompt token 数。用于精确的压缩预检查。 |

### 布尔标志（状态机）

`SessionEntry` 有若干布尔标志，构成一个简单的状态机，支配下一次访问时的 session 行为。

| 标志 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `was_auto_reset` | `bool` | `False` | 当显式暂停导致替换 session 时设置。也会为历史记录保留。 |
| `auto_reset_reason` | `Optional[str]` | `None` | 显式暂停时为 `"suspended"`；较旧的行可能保留历史的重置原因。 |
| `reset_had_activity` | `bool` | `False` | 被替换的 session 是否曾有过活动。 |
| `is_fresh_reset` | `bool` | `False` | 由显式的 `/new` 或 `/reset` 设置。在第一条消息时触发 topic/channel skill 重新注入。与 `was_auto_reset` 区分，以避免误导性的「session 已过期」提示。 |
| `expiry_finalized` | `bool` | `False` | 为恢复保留的历史终结栅栏；没有定时器写入它。 |
| `suspended` | `bool` | `False` | 硬性的强制清空信号。由 `/stop` 或卡死循环升级（3 次以上连续重启失败）设置。在下一次 `get_or_create_session()` 时，无论 `resume_pending` 如何都强制生成新的 `session_id`。 |
| `resume_pending` | `bool` | `False` | 软恢复标记。由 `recover_interrupted_turns()`（对已标记但未回复的回合的崩溃恢复）或 drain 超时设置。在下一次访问时保留现有 `session_id`——用户在同一份 transcript 上继续。在下一次成功的回合完成后清除。 |
| `resume_reason` | `Optional[str]` | `None` | 为何标记 resume：`"restart_timeout"`、`"shutdown_timeout"`、`"restart_interrupted"`。 |
| `last_resume_marked_at` | `Optional[datetime]` | `None` | 最后一次标记 resume-pending 的时间戳。 |

### 状态转换逻辑（get_or_create_session）

```
                    ┌──────────┐
                    │  传入     │
                    │  消息     │
                    └────┬─────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  session_key 存在    │──── 否 ──► 创建新的 SessionEntry
              │  且 !force_new       │
              └──────────┬───────────┘
                         │ 是
                         ▼
              ┌──────────────────────┐
              │  entry.suspended?    │──── 是 ──► 自动重置：新 session_id
              └──────────┬───────────┘           （reason="suspended"）
                         │ 否
                         ▼
              ┌──────────────────────┐
              │ entry.resume_pending?│──── 是 ──► 返回现有 entry
              └──────────┬───────────┘           （保留 session_id）
                         │ 否                    在下一次成功回合时清除标志
                         ▼
              ┌──────────────────────┐
              │  策略要求重置？      │──── 是 ──► 自动重置：新 session_id
              └──────────┬───────────┘           （reason="idle"/"daily"）
                         │ 否
                         ▼
              ┌──────────────────────┐
              │  返回现有 entry，    │
              │  更新 updated_at     │
              └──────────────────────┘
```

**`get_or_create_session()` 中的优先级顺序：**
1. `suspended=True` → 总是强制重置（硬清空）
2. `resume_pending=True` → 保留 session_id（软恢复）
3. 无触发 → 返回现有 entry（更新 `updated_at`）

---

## 3. SessionStore — 存储与操作

`SessionStore` 是主存储层。它维护一个内存 dict（`_entries`），持久化到 `sessions.json`，同时 SQLite（`SessionDB`）作为 session 元数据和消息 transcript 的规范存储。

### 构造函数

```python
SessionStore(sessions_dir: Path, config: GatewayConfig, has_active_processes_fn=None)
```

- `sessions_dir` — `sessions.json` 所在的目录。
- `config` — 用于路由和例行维护设置的 `GatewayConfig` 实例。
- `has_active_processes_fn` — 可选的、按 `session_key` 索引的回调，用于检查正在运行的后台进程。有活动进程的 session 会被保护，免于路由 entry 修剪。

### 操作（方法）

| 方法 | 描述 |
|---|---|
| `get_or_create_session(source, force_new=False)` | 核心入口。返回现有或创建新的 `SessionEntry`。评估显式暂停和重启恢复状态。创建/结束 SQLite 记录。 |
| `update_session(session_key, last_prompt_tokens=None)` | 交互后的轻量元数据更新。更新 `updated_at`，可选记录 `last_prompt_tokens`。 |
| `reset_session(session_key, display_name=None)` | 显式重置（来自 `/new` 或 `/reset`）。创建新 `session_id`，设置 `is_fresh_reset=True`。结束旧 SQLite session，创建新 session。 |
| `switch_session(session_key, target_session_id, *, expected_session_id=None)` | 切换到另一个已存在的 session ID（来自 `/resume`）。结束当前 SQLite session，重新打开目标 session。带上 `expected_session_id=` 后，重新指向是一次 compare-and-swap：当该 key 不再指向那个 session 时返回 `None` 而不执行切换，这样调用方即使基于一次 `await` 之前的快照进行解析（异步委派的重新绑定、Telegram topic 绑定修复）也不会覆盖并发的 `/new` 或 `/resume`。 |
| `suspend_session(session_key)` | 将 session 标记为 `suspended=True`（来自 `/stop`）。在下一次访问时强制自动重置。 |
| `mark_resume_pending(session_key, reason)` | 将 session 标记为 `resume_pending=True`（来自 drain 超时）。在下一次访问时保留 session_id。不会覆盖 `suspended=True`。 |
| `clear_resume_pending(session_key)` | 在成功恢复的回合后清除 `resume_pending`。在 `run_conversation()` 返回后由 gateway 调用。 |
| `recover_interrupted_turns(max_age_seconds)` | 崩溃恢复：将已死进程遗留的持久化活动回合标记提升为 `resume_pending=True`（`restart_interrupted`）。没有标记的 session 已完成其回合，保持不动。在非正常关闭后的启动时调用。 |
| `prune_old_entries(max_age_days)` | 删除超过 `max_age_days` 的 entry（基于 `updated_at`）。跳过 `suspended` entry 和有活动进程的 session。 |
| `list_sessions(active_minutes=None)` | 返回所有 session，可选按近期活动过滤。按 `updated_at` 降序排序。 |
| `lookup_by_session_id(session_id)` | 为某个持久化 session ID 找到活动的 `SessionEntry`。 |
| `has_any_sessions()` | 检查是否创建过任何 session（使用 SQLite 历史，而非仅内存 dict）。 |
| `append_to_transcript(session_id, message, skip_db=False)` | 向 SQLite transcript 追加一条消息。当 agent 已持久化时，`skip_db=True` 可避免重复写入。 |
| `rewrite_transcript(session_id, messages)` | 完整替换 session transcript（由 `/retry`、`/undo`、`/compress` 使用）。 |
| `load_transcript(session_id)` | 从 session 的 SQLite transcript 加载所有消息。 |
| `rewind_session(session_id, n=1)` | 通过软删除回退 `n` 个用户回合（保留审计轨迹）；是对 `SessionDB.rewind_user_turn`（`hermes_state_rewind.py`）的轻量封装，该回退逻辑与 CLI 的 `/undo`/`/retry` 以及 TUI 共享。返回 `{rewound_count, turns_undone, target_text}`。 |

### 内部辅助函数

- `_ensure_loaded()` / `_ensure_loaded_locked()` — 将 `sessions.json` 加载进 `_entries` dict。
- `_save()` — 通过临时文件 + `atomic_replace` 原子写入 `sessions.json`。
- `_generate_session_key(source)` — 委托给 `build_session_key()`，并使用 config 参数。

### 存储布局

```
{sessions_dir}/
  sessions.json          # 内存中的 _entries dict，以 JSON 持久化
                          将 session_key → SessionEntry（仅元数据）
  {session_id}.jsonl     # （旧式，已在 spec 002 中移除）
```

规范的 transcript 存储是通过 `SessionDB`（来自 `hermes_state`）的 SQLite。`sessions.json` 文件持久化 `session_key → session_id` 的映射以及 entry 元数据（标志、时间戳、token 计数）。如果 SQLite 不可用，存储会回退到 JSONL，但这是一条降级路径。

---

## 4. SessionKey 生成规则

Session key 是标识对话通道的确定性字符串。它们由 `build_session_key(source, group_sessions_per_user, thread_sessions_per_user)` 生成。

### Key 格式

```
agent:main:{platform}:{chat_type}[:{slack scope_id}][:{chat_id}][:{thread_id}][:{participant_id}]
```

实际布局中，Slack 会多一个 `scope_id` 槽位（位于 chat 标识之前；Discord 的 guild 作用域被刻意不加入，以保持 key 兼容性）。当一条消息属于线程时，`chat_type` 槽位会被规范化为 `"thread"`（源码 `gateway/session.py::build_session_key` 主持这一定义）。本节示例是简化视图；**切勿手动构造会话键，始终以 `build_session_key()` 为准。**

### DM 规则

| 场景 | Key |
|---|---|
| 带 chat_id 的 DM | `agent:main:telegram:dm:12345` |
| 带 chat_id + thread 的 DM | `agent:main:telegram:dm:12345:thread_678` |
| 无 chat_id、带 participant_id 的 DM | `agent:main:signal:dm:user_abc` |
| 无 chat_id 或 participant_id 的 DM | `agent:main:telegram:dm` |
| WhatsApp DM（规范化后） | `agent:main:whatsapp:dm:{canonical_number}` |

- 存在 `chat_id` 时 DM 总是包含它，从而隔离每个私密对话。
- `thread_id` 进一步区分同一 DM chat 内的线程化 DM。
- 没有 `chat_id` 时，回退到 `user_id_alt` 或 `user_id` 作为 participant_id。
- 没有任何标识符时，该平台上所有 DM 都合并为一个共享 session。

### Group/Channel 规则

| 场景 | Key |
|---|---|
| Group chat | `agent:main:telegram:group:-10012345` |
| Group chat，按用户隔离 | `agent:main:telegram:group:-10012345:user_abc` |
| Group 中的 thread，共享 | `agent:main:discord:group:12345:thread_678` |
| Group 中的 thread，按用户 | `agent:main:discord:group:12345:thread_678:user_abc` |
| Channel | `agent:main:slack:channel:C12345` |
| WhatsApp group（规范化后） | `agent:main:whatsapp:group:{canonical_id}:{participant}` |

- `chat_id` 标识父 group/channel。
- `thread_id` 区分该父级内的 thread。
- **按用户隔离**（追加 `participant_id`）由以下配置控制：
  - `group_sessions_per_user`（默认：`True`）— group/channel session 被隔离。
  - `thread_sessions_per_user`（默认：`False`）— thread 默认是**共享**的（Telegram forum topic、Discord thread、Slack thread 都每个 thread 共享一个 session）。
- `participant_id` = `user_id_alt` 或 `user_id`（按此优先级）。
- WhatsApp 标识符会被规范化，以处理 JID/LID 别名翻转。

### 特殊情况：WhatsApp

WhatsApp 电话号码经过 `canonical_whatsapp_identifier()` 处理，它会去除 `@s.whatsapp.net` 后缀并规范化为 E.164 格式。这可以防止当桥接返回同一电话号码的不同别名形式时造成 session 碎片化。

---

## 5. 多用户隔离策略

多用户隔离决定同一 chat 中的多个用户是共享一个对话，还是各自拥有自己的私有 session。

### 判定逻辑（`is_shared_multi_user_session`）

```python
def is_shared_multi_user_session(source, *, group_sessions_per_user, thread_sessions_per_user):
    if source.chat_type == "dm":
        return False  # DM 始终是私密的
    if source.thread_id:
        return not thread_sessions_per_user  # Thread：除非按用户，否则共享
    return not group_sessions_per_user       # Group：除非共享，否则隔离
```

### 总结

| Chat 类型 | 默认 | 配置控制 |
|---|---|---|
| DM | 私密（从不共享） | 不适用 |
| Group/Channel | 按用户隔离 | `group_sessions_per_user`（默认：True） |
| Thread（forum、discord） | 共享（所有参与者看到相同上下文） | `thread_sessions_per_user`（默认：False） |

### 对 System Prompt 的影响

当 `shared_multi_user_session=True` 时，system prompt 会省略固定的用户名，而改为声明：*「多用户 \{thread|session\}——消息会带上前缀 [sender name]。可能有多个用户参与。」* 各个发送者名称由 gateway 在运行时加到每条用户消息前面，从而保持 prompt caching（system prompt 不会逐回合改变）。

---

## 6. 显式对话边界

非活动状态和墙钟时间永远不会轮换对话。`/new` 和 `/reset` 创建显式边界；上下文压缩继续管理长历史。旧式定时器配置会被忽略。

显式暂停仍会在下一次传入回合时创建边界。恢复会尊重显式以及历史已完结的边界，而不是重新打开它们。仅资源层面的驱逐和 WebSocket 孤儿回收会让对话保持可恢复。

---

## 7. 重启恢复流程

重启恢复系统确保在途 session 在 gateway 重启、崩溃和 drain 超时之间得以保留。它是 issue #7536 的解决方案。

### 启动恢复序列

```
Gateway 启动
       │
       ▼
┌───────────────────────────────┐
│ 检查 .clean_shutdown         │── 存在？ ──► 跳过暂停（干净退出）
│ 标记                          │
└───────────────────────────────┘
       │ 缺失
       ▼
┌───────────────────────────────┐
│ _recover_unclean_sessions()   │── 带持久化回复的已标记回合
│                               │   → 投递账本（sent、marked）；
│                               │   无回复的已标记回合 →
│                               │   resume_pending（一次）
└───────────────────────────────┘
       │
       ▼
┌───────────────────────────────┐
│ _suspend_stuck_loop_sessions()│── 暂停在 3 次以上重启之间
│                               │   一直处于活动状态的 session
└───────────────────────────────┘
       │
       ▼
┌───────────────────────────────┐
│ 在启动恢复运行期间排队       │
│ 传入消息                      │
│ （_startup_restore_in_progress）│
└───────────────────────────────┘
       │
       ▼
┌───────────────────────────────┐
│ 对每个适配器，找到           │
│ resume_pending session →     │
│ 合成 MessageEvent 并         │
│ 运行 _handle_message 以让    │
│ agent 自动继续               │
└───────────────────────────────┘
```

### 崩溃恢复（`_recover_unclean_sessions`）

当没有 `.clean_shutdown` 标记存在（崩溃或非预期退出）时，在 gateway 启动时调用。它只作用于持久化的活动回合标记，从不依据「近期性」：一个仅在崩溃前不久处于活动状态的 chat 已完成其回合，不会被再次应答。

该标记在回合开始时设置，并一直保持到最终回复进入投递账本（适配器在 `record_delivery_obligation` 之后立即释放它），或者直到不再有任何欠账（流式回复、被抑制或空响应）。因此，启动时仍留存的标记意味着以下两种情况之一：

- **回复已持久化但从未记入账本。** 存储的 transcript 回复被记录为一行无归属的账本条目并清除标记；启动扫描用「Recovered reply」提示投递它一次。该回合不会被重新生成。这条回复会以实时投递本来会采用的相同标准来判定：内部回合上的裸静默标记（`[SILENT]`、`NO_REPLY`、...），或对诊断唤醒的回复被该 chat 的策略静音，则不欠任何东西（标记被清除，不发送也不恢复）。人类回合的裸静默标记会变成与实时路径发送的相同的「仅返回了静默标记」提示。
- **没有持久化任何回复。** `recover_interrupted_turns()` 设置 `resume_pending=True`、`resume_reason="restart_interrupted"`，该回合自动恢复一次。

标记的开始时间以带时区的 UTC 存储，并按 epoch 秒进行比较，因此发生在不同本地时区的重启（DST 变化、容器与单元的 `TZ`）既不会把新标记误判为过期，也不会把上一回合的回复当作本回合的回复。由较旧版本写入的标记（无时区的本地时间）会被当作主机本地时间读取。

已在账本中的回合由账本扫描重新投递，该扫描也会清除该 session 的任何 `resume_pending`，因此它绝不会既被投递又被重新应答。

### 卡死循环检测（`_suspend_stuck_loop_sessions`）

通过一个 JSON 文件（`{HERMES_HOME}/restart_counts.json`）统计连续重启次数。如果某个 session 在 3 次以上连续重启期间一直处于活动状态，它会被自动暂停，让用户获得一个干净的新起点。

### Drain 超时标记

在优雅关闭/重启时，drain 系统会对任何在 drain 超时触发时处于回合中途的 session 调用 `mark_resume_pending()`。原因包括：

- `"restart_timeout"` — 在重启 drain 期间被终止
- `"shutdown_timeout"` — 在关闭 drain 期间被终止
- `"restart_interrupted"` — 对已标记但未回复回合的崩溃恢复（来自 `recover_interrupted_turns`）

以上三种原因都在 `_AUTO_RESUME_REASONS` 中，有资格在启动时自动恢复。

### 下一次访问时自动恢复

当 `get_or_create_session()` 遇到 `resume_pending=True` 时：

1. 它返回现有 entry，**不**创建新的 `session_id`。
2. 现有 transcript 完整加载。
3. 此处的标记不会被清除——它会一直保留到下一次成功回合完成（`clear_resume_pending()` 在 `run_conversation()` 返回真实响应后由 gateway 调用）。
4. 如果恢复的回合再次被中断，`resume_pending` 标志保持设置，下一次重启将重试。卡死循环计数器处理最终的升级（3 次重试 → 暂停）。

### 干净关闭标记（`.clean_shutdown`）

在优雅关闭结束时写入。在下一次启动时：

- 如果存在：完全跳过崩溃恢复并丢弃孤儿回合标记。活动 agent 已被 drain，因此没有 session 卡住。
- 然后删除该标记。

这可以防止在 `hermes update`、`hermes gateway restart` 或 `/restart` 之后出现不必要的自动重置。

---

## 8. 消息排队流程

消息排队系统处理两种场景：

1. **中断后的后续消息** — 当用户在 agent 处理期间发送多条消息时，后续消息会作为单槽待处理消息排队。
2. **`/queue` FIFO** — 显式的 `/queue` 命令，每条都必须按顺序各自产生一次完整的 agent 回合，且不合并。

### 数据结构

```
adapter._pending_messages: Dict[session_key, MessageEvent]
    └── 每个 session 一个「下一个」单槽。重复发送时被覆盖
        （突发折叠）。与照片突发后续消息共享。

self._queued_events: Dict[session_key, List[MessageEvent]]
    └── 溢出缓冲。当槽被占用时，每次 /queue 调用追加到这里。
        每次 drain 后一次提升一个。
```

### 入队（`_enqueue_fifo`）

```
_enqueue_fifo(session_key, event, adapter)
       │
       ▼
┌───────────────────────────────────────┐
│ 槽是否空闲？                         │
│ （session_key 不在 _pending_messages 中）│── 是 ──► 将 event 放入槽
└───────────────────────────────────────┘
       │ 否
       ▼
追加到 _queued_events[session_key]（溢出尾部）
```

### 出队 / 提升（`_promote_queued_event`）

在槽被消费后的 drain 位置调用。如果存在溢出项：

- 当 `pending_event is None`（槽为空）时，返回溢出头部作为新 event。
- 当 `pending_event` 存在时，将溢出头部暂存进槽，供下一次递归使用。
- 如果没有可用的适配器，推回 `_queued_events`（不要静默丢弃）。

### 队列深度

`_queue_depth(session_key, adapter)` 返回 `len(overflow) + (1 if slot occupied else 0)`。

### 清理

session 的排队 event 会在 `/new` 和 `/reset` 时被清除（通过 `_handle_reset_command`）。`/stop` 丢弃用户在被中断回合期间发送的单槽后续消息。停在任一个存储中的**内部**唤醒（异步委派完成通知、kanban/cron 的 `notify+wake`）在这三条命令下都会存活：`_interrupt_and_clear_session` 将其保留在槽中（当被丢弃的人类后续消息占用槽时，把它从溢出中提升出来），这样命令后的 drain 就能立即启动它，而不是让 session 一直空闲到下一个用户消息。一个被 `/new` 刚关闭的 session 上固定的唤醒是否仍可运行，由处理时决定（`_resolve_async_delegation_session`，失败即关闭）。

这两条命令也会终止该 session 的**后台委派**（`tools.async_delegation.interrupt_for_session`，按路由 key 以及发起者的持久 session id 选定）：`_interrupt_and_clear_session` 在忙碌路径下扇出 stop，而 `_handle_stop_command` 对派发回合已结束的空闲 session 做同样的事（回复「Stopped」而非「No active task to stop」）。回合自身的硬中断永远不会到达这些单元——它们在派发时已从 `_active_children` 分离——因此如果没有扇出，它们会一直运行到完成并在几分钟后唤醒 chat。每个停止的单元仍会正常终结，并以带 `status="interrupted"` 和子进程部分输出的完成通知重新进入。`/new` 和 `/reset` 已经在 `_handle_reset_command` 中做了这件事；共享辅助函数更早的调用在那里是幂等的（两次请求的硬中断只是一次 stop）。

### FIFO 不变式

每次 `/queue` 调用都会按 FIFO 顺序恰好产生一次完整的 agent 回合，不合并。单槽 `_pending_messages` + 溢出 `_queued_events` 的设计确保活动回合期间的重复发送不会导致乱序处理。

---

## 9. Session 上下文注入

`SessionContext` 由 `SessionSource` 和 `GatewayConfig` 构建，并注入到 agent 的 system prompt 中。它告诉 agent：

- 当前消息来自哪里
- 连接了哪些平台
- 它可以把计划任务输出投递到哪里
- 这是否是一个共享的多用户 session

### 构造（`build_session_context`）

```python
def build_session_context(source, config, session_entry=None) -> SessionContext
```

1. 从 config 收集已连接的平台。
2. 为每个平台收集主 channel。
3. 通过 `is_shared_multi_user_session()` 判定 `shared_multi_user_session`。
4. 如果提供了 `session_entry`，附加 session 元数据（key、id、时间戳）。

### PII 脱敏（`build_session_context_prompt`）

动态 system prompt 段（`## Current Session Context`）可以可选地在发送给 LLM 之前脱敏个人身份信息：

- User ID → `user_<12hex>`（SHA-256 前缀）
- Chat ID → `<platform>:<12hex>` 或仅 `<12hex>`
- 不做脱敏的平台：Discord（需要原始 ID 用于 `@mentions`），以及任何未标记为 `pii_safe` 的插件注册平台。

脱敏仅适用于 system prompt 文本。路由、session key 和适配器操作始终使用原始值。

---

## 10. 后台例行维护

`_session_housekeeping_watcher` 定期清扫闲置的已缓存 agent，在内存压力下释放缓存 entry，并每小时修剪旧的 routing entry。它绝不会因为非活动或一天中的时间而结束一份 transcript。

TTL、LRU 和压力驱逐会在软释放 client 之前，把实时 transcript 提交到内存 provider。活动回合保持受保护；terminal、browser 和后台进程资源在软释放后仍然存活。routing entry 修剪会保留规范 SQLite transcript，而实时进程会保护其 routing entry 不被修剪。历史 `expiry_finalized` 标志仍是恢复栅栏，但不再由定时器 watcher 写入。

---

## 11. Agent Cache

gateway 维护一个以 `session_key` 为键的 `AIAgent` 实例 LRU 缓存，以在回合之间保持 prompt caching。

### 缓存属性

- **最大大小：** 128 个 entry（`agent.agent_cache.max_size`，默认 `_AGENT_CACHE_MAX_SIZE`）。
- **驱逐策略：** 最近最少使用（通过 `OrderedDict` 的 LRU）。
- **空闲 TTL：** 3600 秒（1 小时）— `agent.agent_cache.idle_ttl_secs`，由 `_session_housekeeping_watcher` 强制执行。
- **内存预算：** `agent.agent_cache.memory_high_mb`（默认 `auto`）— 见下文。
- **锁：** `_agent_cache_lock`（threading），用于线程安全。
### 内存压力驱逐

一个被缓存的 agent 会固定 `_session_messages`，即完整的实时 transcript，包括工具输出——在 100 次以上工具调用的 session 上可达数十 MB。entry 上限和空闲 TTL 对此都是盲目的：一个服务多个 chat 的 gateway 会让每个热 transcript 常驻（在 TTL 内转过回合的 agent 永远不会被空闲清扫），因此 RSS 不断攀升，直到 cgroup 限流，SIGTERM 再也无法在 systemd 的停止超时内完成 flush（#80764）。

`_sweep_agent_cache_under_pressure()` 是那个阀门。每个 watcher tick 会把匿名内存与 `memory_high_mb` 比较——当 gateway 在 cgroup 限制下运行时，用 cgroup 自身的 `memory.stat` 中的 `anon`（这是预算计费的作用域，因此同单元的子进程如 `execute_code` 内核也计入；#110549），否则用进程自身的匿名 RSS；超出预算时，它通过 cap 执行器使用的同一软路径（`_commit_then_release_soft`）驱逐 LRU agent，然后运行 `malloc_trim` 让释放出来的 arena 真正归还给 OS。被驱逐的 session 在下一回合会从持久化的 session 重建其 transcript。

有三类 session 永远不会被释放：

- 当前正处于回合中途的 agent（其 client 和 sandbox 正在使用中）；
- `protect_recent` 个最近使用的 session（它们的 prompt cache 最有价值）；
- 任何实时 transcript 尚未完全写入磁盘的 session —— `transcript_persistence_caught_up()` 将 `_last_flushed_db_idx` 与 `len(_session_messages)` 比较，这与 FTS 写入损坏防护在保留实时历史而非滞后 transcript 时所响应的同一差异有关。

`memory_high_mb: auto` 从 gateway 所在的 cgroup 限制推导预算（`memory.high`，然后是 `memory.max`，再是 cgroup v1），在没有上限时回退到总 RAM。设置一个数字可将其固定，`0`/`off` 则完全禁用这一轮处理。辅助函数位于 `gateway/agent_cache_pressure.py` 中。

### 缓存生命周期

```
消息到达
    │
    ▼
get_or_create_session()  →  获得 session_key
    │
    ▼
查找 _agent_cache[session_key]
    │
    ├── 命中 → move_to_end()，复用 AIAgent（保持 prompt cache）
    │
    └── 未命中 → 创建新的 AIAgent，存入缓存
                （若已满，popitem(last=False) 驱逐 LRU entry）
    │
    ▼
run_conversation()  →  agent 处理消息
    │
    ▼
例行维护软释放闲置 agent，而不结束 transcript
```

### 清理流程

资源驱逐会移除缓存的 agent，并在软释放 client 之前提交内存。完整的 `_cleanup_agent_resources(agent)` 拆解保留给实际对话边界和关闭。

---

## 附录：关键配置

| 配置键 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `group_sessions_per_user` | `bool` | `true` | 按用户隔离 group/channel session |
| `thread_sessions_per_user` | `bool` | `false` | 按用户隔离 thread session |
| `session_store_max_age_days` | `int` | `0` | 修剪超过 N 天的 session（0 = 禁用） |
| `agent.gateway_auto_continue_freshness` | `int` | `3600` | 恢复新鲜度窗口的秒数 |
| `agent.gateway_timeout` | `int` | `1800` | agent 回合超时（默认 30 分钟） |
| `agent.agent_cache.max_size` | `int` | `128` | 缓存 AIAgent 的 LRU entry 上限 |
| `agent.agent_cache.idle_ttl_secs` | `int` | `3600` | 闲置这么久的 agent 会被驱逐 |
| `agent.agent_cache.memory_high_mb` | `int`/`str` | `auto` | 超过该匿名 RSS 预算就释放 LRU transcript |
| `agent.agent_cache.max_evictions_per_pass` | `int` | `16` | 每轮压力处理最多释放的 session 数 |
| `agent.agent_cache.protect_recent` | `int` | `8` | 压力处理绝不触碰的最近使用（MRU）session 数 |

## 状态数据库和 FTS 恢复

规范 transcript 保存在 `sessions` 和 `messages` 表中。FTS5 表及其同步触发器是被派生的索引，可以在不删除规范消息的情况下分离并重建。请参见 [State DB 恢复](state-db-recovery.md)，了解有边界的实时故障模式以及显式修复流程。

### 对话生命周期

不支持任何空闲或每日重置设置。显式的 `/new` 和 `/reset`、压缩、暂停和崩溃恢复各自保留着独立的生命周期职责。