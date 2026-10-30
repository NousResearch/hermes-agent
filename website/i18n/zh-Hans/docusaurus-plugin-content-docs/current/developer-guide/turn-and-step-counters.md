---
sidebar_position: 19
title: "Turn 与 Step 计数"
description: "每轮对话与每次 API 调用的计数器在哪里、如何暴露给 hook，以及会话持久化了什么"
---

# Turn 与 Step 计数

Hermes 统计两个容易混淆的量：**turn**（一条用户消息及 agent 的全部响应）和
**step**（一个 turn 的工具循环中的一次 API 调用）。二者归属不同、生命周期不同，
且只有其中一个有专门的持久化列。本页把每个计数器映射到它的更新点、消费方和暴露面。

## 总览

| 计数器 | 作用域 | 归属 | 写入点 | 是否持久化 |
|---|---|---|---|---|
| `_user_turn_count` | 会话（单个 agent 进程） | `AIAgent` | `agent/turn_context.py:1132` | 否——从历史重建 |
| `api_call_count`（循环局部） | Turn | `TurnState` / `IterationBudget` | `agent/turn_iteration_prep.py:406-407` | 否 |
| `agent._api_call_count` | Agent，供状态展示镜像 | `AIAgent` | 同上自增（含退款） | 否 |
| `agent.session_api_calls` | 会话 | `AIAgent` | `agent/turn_usage.py:85` | 仅通过下面的 DB delta |
| `sessions.api_call_count` | 会话，持久 | SessionDB | 每次调用排队 delta（`agent/turn_usage.py:264-281`） | 是 |
| `session_model_usage.api_call_count` | 会话 × 模型路由 | SessionDB | Upsert（`hermes_state_usage.py:55-62`） | 是 |

## 用户轮次：`agent._user_turn_count`

`_user_turn_count` 统计该对话中 agent 已处理的用户消息数。构造时归零
（`run_agent.py:406`，也在 `agent/agent_init.py:2120` 的默认状态表里），每个 turn
只自增一次——在用户行写入 transcript 之后、压缩与工具循环之前
（`agent/turn_context.py:1132`）。

它是进程内状态。恢复会话时，第一个 turn 会按持久化 transcript 中
`role == "user"` 的行数重建该值（`agent/turn_context.py:729-762`），并同步
memory nudge 的节奏（`_turns_since_memory`）。

消费方：

- memory nudge 计数 `_turns_since_memory`（`agent/turn_context.py:770`）。
- 外部记忆 provider，通过 `on_turn_start(turn_number, message, ...)`
  （`agent/turn_context.py:928` → `agent/memory_manager.py:613`）。这是数字形式的
  turn 序号离开进程的唯一出口。

会话数据库没有 `turn_count` 列，hook 总线上也没有 turn 序号。需要 turn 总数时只能
从 transcript 推导；`hermes_cli/session_recap.py:66-70`（`_count_visible_turns`）
是标准写法。

## 步数：turn 内的 `api_call_count`

`api_call_count` 是当前 turn 已发起的 provider 调用数。它存在于循环状态上
（`agent/conversation_loop.py:1484`），在循环条件中与 `max_iterations` 比较
（`agent/conversation_loop.py:1729`），并在 `begin_iteration` 中、请求组装之前自增
（`agent/turn_iteration_prep.py:406-407`）。同一条语句把该值镜像到
`agent._api_call_count`，供状态展示使用（`run_agent.py:863-864`）。

上限由两部分构成：局部调用数与每 turn 新建的 `IterationBudget`
（`agent/turn_context.py:664`）。`begin_iteration` 消耗一个预算名额
（`agent/turn_iteration_prep.py:416`）。未真正到达 provider 的迭代——preflight 压缩、
provider 切换跳转——会被退款，局部计数与预算同步减一
（`agent/turn_context_compaction.py:103-118`）。

触顶时 turn 以 `max_iterations_reached(N/M)` 结束；若没有待保留的回答，会请求模型
做一次总结（`agent/turn_finalizer.py:130-163`）。

计数从 1 开始：`step_callback` 触发时 `begin_iteration` 已执行，所以第一个 step 是 1
（阶段顺序见 `agent/conversation_loop.py:1731-1733`）。

## Hook 能看到什么

**step 有暴露，turn 计数没有。**

| 暴露面 | 事件 | 字段 | 语义 |
|---|---|---|---|
| Gateway shell/outbound hook | `agent:step` | `iteration` | 当前 turn 的 step，从 1 开始（`gateway/run_turn_progress.py:883-899`） |
| 插件 lifecycle | `pre_api_request`、`post_api_request`、`api_request_error` | `api_call_count` | 当前 turn 的 step |
| 插件 streaming 观察者 | `on_stream_start` / `on_stream_delta` / `on_stream_end` / `on_interim_message` | `iteration` | `agent._api_call_count`（`agent/stream_delivery.py:316-324`） |
| 插件 lifecycle | `pre_auxiliary_call`、`post_auxiliary_call` | `api_call_count` | **该辅助调用自身的尝试序号**，不是父 turn 的 step（`agent/auxiliary_hooks.py:106-107`） |
| 插件 lifecycle | `pre_llm_call`、`post_llm_call`、`on_session_end` | `turn_id`、`is_first_turn` | 不透明身份与布尔值；没有数字 turn 计数 |
| Gateway shell/outbound hook | `agent:start`、`agent:end` | — | 无计数器 |

`agent.step_callback(api_call_count, prev_tools)` 是 `agent:step` 背后的内部回调
（`agent/turn_iteration_prep.py:158-161`）；gateway 在 `gateway/run_turn_runner.py:472`
绑定它，ACP 适配器在 `acp_adapter/events.py:259` 绑定。精确的载荷约定见 hooks 参考
（`user-guide/features/hooks`）。

## 会话持久化了什么

持久化的 step 总量只有一列：`sessions.api_call_count`
（`hermes_state_common.py:507`）。它通过每次调用的 delta 累加——`queue_token_counts`
排队 `api_call_count=1`（`agent/turn_usage.py:264-281`），再由
`_update_token_counts_in_transaction` 用 delta SQL 执行
`api_call_count = COALESCE(api_call_count, 0) + ?`
（`hermes_state_usage.py:36-48`）。按路由的归因写入 `session_model_usage`（`hermes_state_schema.py:85`）。主循环行
`task = ''`，最近一次路由由 `get_recent_session_model_route`
（`hermes_state_sessions.py:855-877`）读取。辅助调用是另一本账：它们只写
`session_model_usage` 中 `task` 非空的行，从不更新 `sessions` 行
（`hermes_state_usage.py:392-424`），随后由 `auxiliary_usage_by_task`
（`hermes_state_usage.py:427-449`，`WHERE task != ''`）按任务汇总。

内存侧的对应物是 `agent.session_api_calls`
（`agent/agent_init.py:2131`，在 `agent/turn_usage.py:85` 与
`agent/codex_runtime.py:155` 自增）。`/usage` 里显示的 "API calls"
（`gateway/slash_commands_status.py:215`）、状态栏、以及本地前端收到的
`session.usage.api_calls`（`gateway/session_usage.py:20`）都来自它。

### 两个 step 总数可能不一致

chat-completions 路径只在响应确实带 usage 时才排队 DB delta：无 usage 的响应先
自增 `session_api_calls`，随后在排队之前提前返回
（`agent/turn_usage.py:85-104` 对比 `:264-281`）。codex app-server 路径相反——无 usage
的 turn 仍会排队 `api_call_count=1`（`agent/codex_runtime.py:166-168`）。因此主链路上
持久化的 `sessions.api_call_count` 可能比内存计数少掉"无 usage 的尝试"个数；codex
路径则一致。失败与中断的调用两边都不计。

## 实践建议

- 做进度展示请用 `iteration` / `api_call_count`；不要把
  `sessions.api_call_count` 当作"工具迭代数"——它统计的是返回了响应的 provider
  尝试，而不是循环预算消耗。
- 没有持久化的 turn 总数。数 `role == "user"` 行，或自己维护计数；`message_count`
  包含 assistant 与 tool 行，不是 turn 数。
- `turn_id` 当作不透明值。它是身份（以及 Relay/租约键），不是序号。
