---
sidebar_position: 11
title: "Plugin LLM 访问"
description: "通过 ctx.llm 在 plugin 内部运行任意 LLM 调用——支持对话或结构化输出、同步或异步。宿主持有认证凭据，失败关闭信任门控，可选 JSON Schema 验证。"
---

# Plugin LLM 访问

`ctx.llm` 是 plugin 发起 LLM 调用的官方方式。
对话补全、结构化提取、同步、异步、带或不带图像——
同一接口，同一信任门控，同一宿主持有的凭据。

Plugin 在需要涉及模型但又不属于 agent 对话的场景时使用它。
例如：将工具报错改写成非工程师也能理解的语言的 hook；
在消息入队前进行翻译的 gateway 适配器；
对长段粘贴内容进行摘要的斜杠命令；
对前一天活动评分并向状态看板写一行记录的定时任务；
以及决定某条消息是否值得唤醒 agent 的预过滤器。

这些任务不应让 agent 介入。它们只需要一次 LLM 调用、一个有类型的答案，然后结束。

## 最简调用

```python
result = ctx.llm.complete(messages=[{"role": "user", "content": "ping"}])
return result.text
```

这就是整个 API 的一行示例。无需密钥、无需 provider 配置、无需 SDK 初始化。Plugin 运行在用户当前使用的任意 provider 和模型上——用户切换 provider 时，plugin 自动跟随。

## 更完整的对话示例

```python
result = ctx.llm.complete(
    messages=[
        {"role": "system", "content": "Rewrite errors as one short sentence a non-engineer can act on."},
        {"role": "user",   "content": traceback_text},
    ],
    max_tokens=64,
    purpose="hooks.error-rewrite",
)
return result.text
```

`purpose` 是一个自由格式的审计字符串——它会出现在 `agent.log` 和 `result.audit` 中，方便运营人员查看哪个 plugin 发起了哪次调用。可选，但对于频繁触发的场景建议填写。

## 结构化输出

当 plugin 需要有类型的答案时，切换到结构化模式：

```python
result = ctx.llm.complete_structured(
    instructions="Score this support reply for urgency (0–1) and pick a category.",
    input=[{"type": "text", "text": message_body}],
    json_schema=TRIAGE_SCHEMA,
    purpose="support.triage",
    temperature=0.0,
    max_tokens=128,
)

if result.parsed["urgency"] > 0.8:
    await dispatch_to_oncall(result.parsed["category"], message_body)
```

宿主向 provider 请求 JSON 输出，在本地作为兜底进行解析，若安装了 `jsonschema` 则对你的 schema 进行验证，最终在 `result.parsed` 上返回一个 Python 对象。如果模型无法生成有效 JSON，`result.parsed` 为 `None`，`result.text` 携带原始响应。

## 此模式的优势

* **一次调用，四种形态。** `complete()` 用于对话，`complete_structured()` 用于有类型的 JSON，`acomplete()` 和 `acomplete_structured()` 用于 asyncio。参数相同，结果对象相同。
* **宿主持有凭据。** OAuth token、刷新流程、凭据池、每任务辅助覆盖——Hermes 已有的所有凭据概念均适用。Plugin 永远看不到 token；宿主通过 `result.audit` 将调用归因回溯。
* **有界。** 单次同步或异步调用。无流式输出，无工具循环，无需管理对话状态。给定输入，获取结果，返回。
* **失败关闭信任。** 从未配置过的 plugin 无法自行选择 provider、模型、agent 或存储的凭据。默认行为是"使用用户正在使用的"。运营人员在 `config.yaml` 中按 plugin 逐一选择开启特定覆盖。

## 快速开始

以下是两个完整的 plugin 示例——一个对话，一个结构化。两者均在单个 `register(ctx)` 函数中实现，无需任何外部配置即可针对用户当前激活的模型运行。

### 对话补全——`/tldr`

```python
def register(ctx):
    ctx.register_command(
        name="tldr",
        handler=lambda raw: _tldr(ctx, raw),
        description="Summarise the supplied text in one paragraph.",
        args_hint="<text>",
    )


def _tldr(ctx, raw_args: str) -> str:
    text = raw_args.strip()
    if not text:
        return "Usage: /tldr <text to summarise>"
    result = ctx.llm.complete(
        messages=[
            {"role": "system",
             "content": "Summarise the user's text in one tight paragraph. No preamble."},
            {"role": "user", "content": text},
        ],
        max_tokens=256,
        temperature=0.3,
        purpose="tldr",
    )
    return result.text
```

`result.text` 是模型的响应；`result.usage` 携带 token 计数；`result.provider` 和 `result.model` 携带归因信息。

### 结构化提取——`/paste-to-tasks`

```python
def register(ctx):
    ctx.register_command(
        name="paste-to-tasks",
        handler=lambda raw: _paste_to_tasks(ctx, raw),
        description="Turn freeform meeting notes into structured tasks.",
        args_hint="<text>",
    )


_TASKS_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner":  {"type": "string"},
                    "action": {"type": "string"},
                    "due":    {"type": "string", "description": "ISO date or empty"},
                },
                "required": ["action"],
            },
        },
    },
    "required": ["tasks"],
}


def _paste_to_tasks(ctx, raw_args: str) -> str:
    if not raw_args.strip():
        return "Usage: /paste-to-tasks <meeting notes>"
    result = ctx.llm.complete_structured(
        instructions=(
            "Extract concrete action items from these meeting notes. "
            "One task per actionable line. If no owner is named, leave 'owner' blank."
        ),
        input=[{"type": "text", "text": raw_args}],
        json_schema=_TASKS_SCHEMA,
        schema_name="meeting.tasks",
        purpose="paste-to-tasks",
        temperature=0.0,
        max_tokens=512,
    )
    if result.parsed is None:
        return f"Couldn't parse a response. Raw output:\n{result.text}"
    lines = [f"- [{t.get('owner') or '?'}] {t['action']}" for t in result.parsed["tasks"]]
    return "\n".join(lines) or "(no tasks found)"
```

第三个完整示例（包含图像输入）位于
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-example)
仓库（参考 plugin 的配套仓库——不随 hermes-agent 本体打包）。关于异步接口（`acomplete()` / `acomplete_structured()` 与 `asyncio.gather()` 配合使用），请参见同一仓库中的
[`plugin-llm-async-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-async-example)。

## 何时使用哪种方式

| 你需要…… | 使用 |
|---|---|
| 自由格式文本响应（翻译、摘要、改写、生成） | `complete()` |
| 多轮 prompt（system + few-shot 示例 + user） | `complete()` |
| 经 schema 验证的有类型 dict | `complete_structured()` |
| 图像或文本输入并返回有类型 dict | `complete_structured()` |
| 在异步代码中发起相同调用（gateway 适配器、异步 hook） | `acomplete()` / `acomplete_structured()` |

其他所有内容——provider 选择、模型解析、认证、回退、超时、视觉路由——在四种形态中完全一致。

## API 接口

`ctx.llm` 是 `agent.plugin_llm.PluginLlm` 的实例。

### `complete()`

```python
result = ctx.llm.complete(
    messages=[{"role": "user", "content": "Hi"}],
    provider=None,         # 可选，受门控——Hermes provider id（如 "openrouter"）
    model=None,            # 可选，受门控——该 provider 期望的任意字符串
    temperature=None,
    max_tokens=None,
    timeout=None,          # 秒
    agent_id=None,         # 可选，受门控
    profile=None,          # 可选，受门控——显式指定认证 profile 名称
    purpose="optional-audit-string",
)
# → PluginLlmCompleteResult(text, provider, model, agent_id, usage, audit)
```

普通对话补全。`messages` 采用标准 OpenAI 格式——`{"role": "...", "content": "..."}` 字典列表。多轮 prompt（system + few-shot user/assistant 对 + 最终 user）的用法与 OpenAI SDK 完全一致。

`provider=` 和 `model=` 相互独立，格式与宿主主配置（`model.provider` + `model.model`）相同。仅设置 `model=` 可在用户当前激活的 provider 上使用不同模型。同时设置两者则完全切换 provider。任一参数在未获运营人员授权时均会抛出 `PluginLlmTrustError`。

### `complete_structured()`

```python
result = ctx.llm.complete_structured(
    instructions="What you want extracted.",
    input=[
        {"type": "text",  "text": "..."},
        {"type": "image", "data": b"...", "mime_type": "image/png"},
        {"type": "image", "url":  "https://..."},
    ],
    json_schema={...},     # 可选——触发解析结果及验证
    json_mode=False,       # 设为 True 可在不提供 schema 的情况下请求 JSON
    schema_name=None,      # 可选的人类可读 schema 名称
    system_prompt=None,
    provider=None,         # 可选，受门控
    model=None,            # 可选，受门控
    temperature=None,
    max_tokens=None,
    timeout=None,
    agent_id=None,
    profile=None,
    purpose=None,
)
# → PluginLlmStructuredResult(text, provider, model, agent_id,
#                             usage, parsed, content_type, audit)
```

输入为有类型的文本或图像块（原始字节会自动 base64 编码为 `data:` URL）。当提供 `json_schema` 或设置 `json_mode=True` 时，宿主通过 `response_format` 向 provider 请求 JSON 输出，在本地作为兜底进行解析，若安装了 `jsonschema` 则对你的 schema 进行验证。

* `result.content_type == "json"` — `result.parsed` 是符合你 schema 的 Python 对象。
* `result.content_type == "text"` — 解析或验证失败；检查 `result.text` 获取原始模型响应。

### 异步

```python
result = await ctx.llm.acomplete(messages=...)
result = await ctx.llm.acomplete_structured(instructions=..., input=...)
```

参数和结果类型与对应的同步版本相同。在 gateway 适配器、异步 hook 或任何已运行在 asyncio 事件循环上的 plugin 代码中使用。

### 结果属性

```python
@dataclass
class PluginLlmCompleteResult:
    text: str                    # 助手的响应
    provider: str                # 如 "openrouter"、"anthropic"
    model: str                   # provider 为本次调用返回的模型标识
    agent_id: str                # 使用了哪个 agent 的模型/认证
    usage: PluginLlmUsage        # token 数 + 缓存 + 费用估算
    audit: Dict[str, Any]        # plugin_id、purpose、profile

@dataclass
class PluginLlmStructuredResult(PluginLlmCompleteResult):
    parsed: Optional[Any]        # content_type == "json" 时的 JSON 对象
    content_type: str            # "json" 或 "text"
    # 提供 schema_name 时 audit 中也会携带该字段
```

当 provider 返回相应字段时，`usage` 携带 `input_tokens`、`output_tokens`、`total_tokens`、`cache_read_tokens`、`cache_write_tokens` 和 `cost_usd`。

## 信任门控

默认行为是失败关闭。在没有 `plugins.entries` 配置块的情况下，plugin 可以：

* 针对用户当前激活的 provider 和模型运行四种方法中的任意一种，
* 设置请求塑形参数（`temperature`、`max_tokens`、`timeout`、`system_prompt`、`purpose`、`messages`、`instructions`、`input`、`json_schema`），

……仅此而已。`provider=`、`model=`、`agent_id=` 和 `profile=` 参数在运营人员授权前均会抛出 `PluginLlmTrustError`。

**大多数 plugin 永远不需要此部分。** 仅调用 `ctx.llm.complete(messages=...)` 且不带任何覆盖的 plugin，会针对用户当前激活的内容运行，零配置即可工作。以下配置块仅在 plugin 明确需要固定到与用户不同的模型或 provider 时才有意义。

```yaml
plugins:
  entries:
    my-plugin:
      llm:
        # 允许此 plugin 选择不同的 Hermes provider
        # （必须是 Hermes 已知的 provider——与
        # `hermes model` 和 config.yaml model.provider 中的名称相同）
        allow_provider_override: true

        # 可选：限制允许的 provider。使用 ["*"] 表示任意。
        allowed_providers:
          - openrouter
          - anthropic

        # 允许此 plugin 请求特定模型。
        allow_model_override: true

        # 可选：限制允许的模型。使用 ["*"] 表示任意。
        # 模型与 plugin 发送的字符串进行字面匹配——
        # Hermes 不做任何查找。
        allowed_models:
          - openai/gpt-4o-mini
          - anthropic/claude-3-5-haiku

        # 允许跨 agent 调用（罕见）。
        allow_agent_id_override: false

        # 允许 plugin 请求特定的存储认证 profile
        # （如同一 provider 上的不同 OAuth 账户）。
        allow_profile_override: false
```

Plugin id 对于扁平 plugin 是 manifest 中的 `name:` 字段，对于嵌套 plugin 是路径派生的键（`image_gen/openai`、`memory/honcho` 等）。

### 门控执行内容

| 覆盖项          | 默认  | 配置键                           |
| --------------- | ----- | -------------------------------- |
| `provider=`     | 拒绝  | `allow_provider_override: true`  |
| ↳ 允许列表      | —     | `allowed_providers: [...]`       |
| `model=`        | 拒绝  | `allow_model_override: true`     |
| ↳ 允许列表      | —     | `allowed_models: [...]`          |
| `agent_id=`     | 拒绝  | `allow_agent_id_override: true`  |
| `profile=`      | 拒绝  | `allow_profile_override: true`   |

每项覆盖独立门控。授予 `allow_model_override` **不会**同时授予 `allow_provider_override`——被信任可选择模型的 plugin，在未获得 provider 门控授权前仍固定在用户当前激活的 provider 上。

### 门控无需执行的内容

* 请求塑形参数——`temperature`、`max_tokens`、`timeout`、`system_prompt`、`purpose`、`messages`、`instructions`、`input`、`json_schema`、`schema_name`、`json_mode`——始终允许；它们不涉及凭据或路由选择。
* 默认拒绝策略意味着未配置的 plugin 仍可完成有用的工作——只是针对当前激活的 provider 和模型运行。运营人员只需在 plugin 明确需要更精细路由时才考虑 `plugins.entries`。

## 宿主负责的内容

以下是 `ctx.llm` 为 plugin 代劳的完整列表，你无需自行处理：

* **Provider 解析。** 从用户配置中读取 `model.provider` + `model.model`（或在受信任时读取显式覆盖值）。
* **认证。** 从 `~/.hermes/auth.json` / 环境变量中提取 API 密钥、OAuth token 或刷新 token，包括配置了凭据池时的处理。Plugin 永远看不到这些内容。
* **视觉路由。** 当提供图像输入而用户当前激活的文本模型仅支持文本时，宿主自动回退到已配置的视觉模型。
* **回退链。** 若用户主 provider 返回 5xx 或 429，请求在向 plugin 返回错误前会经过 Hermes 常规的聚合器感知回退流程。
* **超时。** 遵循你的 `timeout=` 参数，回退到 `auxiliary.<task>.timeout` 配置或全局辅助默认值。
* **JSON 塑形。** 在你请求 JSON 时向 provider 发送 `response_format`，若 provider 返回了代码围栏格式的响应则在本地重新解析。
* **Schema 验证。** 安装了 `jsonschema` 时对你的 `json_schema` 进行验证；否则记录一行 debug 日志并跳过严格验证。
* **审计日志。** 每次调用向 `agent.log` 写入一条 INFO 日志，包含 plugin id、provider/模型、purpose 和 token 总量。

## Plugin 负责的内容

* **请求结构。** 对话用 `messages`，结构化用 `instructions` + `input`。Plugin 构建 prompt（提示词）；宿主执行它。
* **Schema。** 你期望返回的任意结构。宿主不会为你推断。
* **错误处理。** `complete_structured()` 在输入为空或 schema 验证失败时抛出 `ValueError`。信任门控拒绝覆盖时抛出 `PluginLlmTrustError`。其他情况（provider 5xx、未配置凭据、超时）抛出 `auxiliary_client.call_llm()` 本身抛出的异常。
* **费用。** 每次调用都针对用户的付费 provider 运行。不要在不考虑 token 消耗的情况下对每条 gateway 消息循环调用 `complete()`。

## 在 plugin 接口中的定位

现有 `ctx.*` 方法各自扩展一个已有的 Hermes 子系统：

| `ctx.register_tool` | 添加 agent 可调用的工具 |
| `ctx.register_platform` | 接入新的 gateway 适配器 |
| `ctx.register_image_gen_provider` | 替换图像生成后端 |
| `ctx.register_memory_provider` | 替换记忆后端 |
| `ctx.register_context_engine` | 替换上下文压缩器 |
| `ctx.register_hook` | 监听生命周期事件 |

`ctx.llm` 是第一个允许 plugin 在*带外*运行用户正在对话的同一模型的接口，无需上述任何注册。这是它唯一的职责。如果你的 plugin 需要注册一个由 agent 调用的工具，使用 `register_tool`。如果需要响应生命周期事件，使用 `register_hook`。如果需要发起自己的模型调用——无论出于何种原因，结构化与否——使用 `ctx.llm`。

## 参考资料

* 实现：[`agent/plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/agent/plugin_llm.py)
* 测试：[`tests/agent/test_plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/tests/agent/test_plugin_llm.py)
* 参考 plugin（配套仓库）：
  * [`plugin-llm-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-example) — 带图像输入的同步结构化提取
  * [`plugin-llm-async-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-async-example) — 使用 `asyncio.gather()` 的异步示例
* 辅助客户端（底层引擎）：参见
  [Provider 运行时](/developer-guide/provider-runtime)。