---
sidebar_position: 6
title: "イベントフック"
description: "ライフサイクルの重要なポイントでカスタムコードを実行 — アクティビティのログ記録、アラート送信、Webhookへのポスト"
---

# イベントフック

Hermesには、ライフサイクルの重要なポイントでカスタムコードを実行する3つのフックシステムがあります:

| システム | 登録方法 | 実行場所 | ユースケース |
|--------|---------------|---------|----------|
| **[Gatewayフック](#gateway-event-hooks)** | `~/.hermes/hooks/` 内の `HOOK.yaml` + `handler.py` | Gatewayのみ | ログ記録、アラート、Webhook |
| **[プラグインフック](#plugin-hooks)** | [プラグイン](/docs/user-guide/features/plugins)内の `ctx.register_hook()` | CLI + Gateway | ツールのインターセプト、メトリクス、ガードレール |
| **[シェルフック](#shell-hooks)** | シェルスクリプトを指す `~/.hermes/config.yaml` 内の `hooks:` ブロック | CLI + Gateway | ブロック、自動フォーマット、コンテキスト注入のためのドロップインスクリプト |

これら3つのシステムはすべて非ブロッキングです — いずれのフックでもエラーはキャッチされてログに記録され、エージェントがクラッシュすることはありません。

## Gatewayイベントフック {#gateway-event-hooks}

Gatewayフックは、メインのエージェントパイプラインをブロックすることなく、Gateway動作中（Telegram、Discord、Slack、WhatsApp、Teams）に自動的に発火します。

### フックの作成

各フックは、2つのファイルを含む `~/.hermes/hooks/` 配下のディレクトリです:

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # どのイベントをリッスンするかを宣言
    └── handler.py     # Pythonのハンドラー関数
```

#### HOOK.yaml

```yaml
name: my-hook
description: Log all agent activity to a file
events:
  - agent:start
  - agent:end
  - agent:step
```

`events` リストは、どのイベントがハンドラーをトリガーするかを決定します。`command:*` のようなワイルドカードを含め、任意のイベントの組み合わせをサブスクライブできます。

#### handler.py

```python
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".hermes" / "hooks" / "my-hook" / "activity.log"

async def handle(event_type: str, context: dict):
    """サブスクライブした各イベントに対して呼び出される。名前は必ず 'handle' にする。"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **context,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**ハンドラーのルール:**
- 名前は必ず `handle` にする
- `event_type`（文字列）と `context`（dict）を受け取る
- `async def` でも通常の `def` でも動作する
- エラーはキャッチされてログに記録され、エージェントがクラッシュすることはない

### 利用可能なイベント

| イベント | 発火タイミング | コンテキストキー |
|-------|---------------|--------------|
| `gateway:startup` | Gatewayプロセスの起動時 | `platforms`（アクティブなプラットフォーム名のリスト） |
| `session:start` | 新しいメッセージングセッションの作成時 | `platform`、`user_id`、`session_id`、`session_key` |
| `session:end` | セッション終了時（リセット前） | `platform`、`user_id`、`session_key` |
| `session:reset` | ユーザーが `/new` または `/reset` を実行したとき | `platform`、`user_id`、`session_key` |
| `agent:start` | エージェントがメッセージの処理を開始したとき | `platform`、`user_id`、`session_id`、`message` |
| `agent:step` | ツール呼び出しループの各イテレーション | `platform`、`user_id`、`session_id`、`iteration`、`tool_names` |
| `agent:end` | エージェントが処理を完了したとき | `platform`、`user_id`、`session_id`、`message`、`response` |
| `command:*` | 任意のスラッシュコマンドの実行時 | `platform`、`user_id`、`command`、`args` |

#### ワイルドカードマッチング

`command:*` に登録されたハンドラーは、任意の `command:` イベント（`command:model`、`command:reset` など）に対して発火します。1つのサブスクリプションですべてのスラッシュコマンドを監視できます。

### 例

#### 長時間タスクのTelegramアラート

エージェントが10ステップを超えたときに自分自身にメッセージを送信します:

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Alert when agent is taking many steps
events:
  - agent:step
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
import os
import httpx

THRESHOLD = 10
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_HOME_CHANNEL")

async def handle(event_type: str, context: dict):
    iteration = context.get("iteration", 0)
    if iteration == THRESHOLD and BOT_TOKEN and CHAT_ID:
        tools = ", ".join(context.get("tool_names", []))
        text = f"⚠️ Agent has been running for {iteration} steps. Last tools: {tools}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
            )
```

#### コマンド使用状況ロガー

どのスラッシュコマンドが使われているかを追跡します:

```yaml
# ~/.hermes/hooks/command-logger/HOOK.yaml
name: command-logger
description: Log slash command usage
events:
  - command:*
```

```python
# ~/.hermes/hooks/command-logger/handler.py
import json
from datetime import datetime
from pathlib import Path

LOG = Path.home() / ".hermes" / "logs" / "command_usage.jsonl"

def handle(event_type: str, context: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(),
        "command": context.get("command"),
        "args": context.get("args"),
        "platform": context.get("platform"),
        "user": context.get("user_id"),
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

#### セッション開始Webhook

新しいセッションで外部サービスにPOSTします:

```yaml
# ~/.hermes/hooks/session-webhook/HOOK.yaml
name: session-webhook
description: Notify external service on new sessions
events:
  - session:start
  - session:reset
```

```python
# ~/.hermes/hooks/session-webhook/handler.py
import httpx

WEBHOOK_URL = "https://your-service.example.com/hermes-events"

async def handle(event_type: str, context: dict):
    async with httpx.AsyncClient() as client:
        await client.post(WEBHOOK_URL, json={
            "event": event_type,
            **context,
        }, timeout=5)
```

### チュートリアル: BOOT.md — Gateway起動のたびにスタートアップチェックリストを実行する

コミュニティでよく使われるパターン: `~/.hermes/BOOT.md` にMarkdownのチェックリストを置き、Gatewayが起動するたびにエージェントに1回それを実行させます。「起動のたびに夜間のcron失敗をチェックし、何か失敗していたらDiscordで知らせて」や、「過去24時間の deploy.log を要約してSlackの #ops にポストして」といった用途に便利です。

このチュートリアルでは、これをユーザー定義フックとして自分で構築する方法を示します。HermesはBOOT.mdの組み込みフックを同梱していません — 望む動作を正確に自分で組み立てます。

#### 構築するもの

1. 自然言語のスタートアップ指示を記述した `~/.hermes/BOOT.md` ファイル。
2. `gateway:startup` で発火し、Gatewayの解決済みモデル/認証情報でワンショットエージェントを起動して、BOOT.mdの指示を実行するGatewayフック。
3. 報告すべきことがないときにエージェントがメッセージ送信を見送れるようにする `[SILENT]` 規約。

#### ステップ1: チェックリストを書く

`~/.hermes/BOOT.md` を作成します。人間のアシスタントに指示を与えるかのように書きます:

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, send a summary to Discord #ops using the `send_message` tool.
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same Discord message.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

エージェントはこれをプロンプトの一部として認識するため、平易な言葉で記述できることなら何でも機能します — ツール呼び出し、シェルコマンド、メッセージ送信、ファイルの要約などです。

#### ステップ2: フックを作成する

```text
~/.hermes/hooks/boot-md/
├── HOOK.yaml
└── handler.py
```

**`~/.hermes/hooks/boot-md/HOOK.yaml`**

```yaml
name: boot-md
description: Run ~/.hermes/BOOT.md on gateway startup
events:
  - gateway:startup
```

**`~/.hermes/hooks/boot-md/handler.py`**

```python
"""Gateway起動のたびに ~/.hermes/BOOT.md を実行する。"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger("hooks.boot-md")

BOOT_FILE = Path.home() / ".hermes" / "BOOT.md"


def _build_prompt(content: str) -> str:
    return (
        "You are running a startup boot checklist. Follow the instructions "
        "below exactly.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Execute each instruction. Use the send_message tool to deliver any "
        "messages to platforms like Discord or Slack.\n"
        "If nothing needs attention and there is nothing to report, reply "
        "with ONLY: [SILENT]"
    )


def _run_boot_agent(content: str) -> None:
    """ワンショットエージェントを起動してチェックリストを実行する。

    Gatewayの解決済みモデルとランタイム認証情報を使用するため、カスタム
    エンドポイント、アグリゲーター、OAuthベースのプロバイダーいずれに対しても
    同様に機能する。
    """
    try:
        from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs
        from run_agent import AIAgent

        agent = AIAgent(
            model=_resolve_gateway_model(),
            **_resolve_runtime_agent_kwargs(),
            platform="gateway",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
        )
        result = agent.run_conversation(_build_prompt(content))
        response = result.get("final_response", "")
        if response and "[SILENT]" not in response:
            logger.info("boot-md completed: %s", response[:200])
        else:
            logger.info("boot-md completed (nothing to report)")
    except Exception as e:
        logger.error("boot-md agent failed: %s", e)


async def handle(event_type: str, context: dict) -> None:
    if not BOOT_FILE.exists():
        return
    content = BOOT_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return

    logger.info("Running BOOT.md (%d chars)", len(content))

    # Gatewayの起動がエージェントの1ターン全体でブロックされないようバックグラウンドスレッドにする。
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
```

2つの重要な行:

- `_resolve_gateway_model()` は、Gatewayの現在設定されているモデルを読み取ります。
- `_resolve_runtime_agent_kwargs()` は、通常のGatewayターンと同じ方法でプロバイダーの認証情報を解決します — APIキー、ベースURL、OAuthトークン、認証情報プールを含みます。

これらがないと、素の `AIAgent()` は組み込みのデフォルトにフォールバックし、デフォルト以外のエンドポイントに対しては401になります。

#### ステップ3: テストする

Gatewayを再起動します:

```bash
hermes gateway restart
```

ログを監視します:

```bash
hermes logs --follow --level INFO | grep boot-md
```

`Running BOOT.md (N chars)` に続いて、`boot-md completed: ...`（エージェントが行ったことの要約）か、エージェントが `[SILENT]` で応答した場合は `boot-md completed (nothing to report)` のいずれかが表示されるはずです。

`~/.hermes/BOOT.md` を削除するとチェックリストを無効化できます — フックはロードされたままですが、ファイルが存在しないときは静かにスキップされます。

#### パターンの拡張

- **スケジュール対応のチェックリスト:** BOOT.mdの指示内で `datetime.now().weekday()` を起点にします（「月曜日なら週次デプロイログもチェックして」）。指示は自由形式のテキストなので、エージェントが推論できることなら何でも対象になります。
- **複数のチェックリスト:** フックを別のファイル（`STARTUP.md`、`MORNING.md` など）に向け、それぞれに対して個別のフックディレクトリを登録します。
- **エージェントを使わないバリアント:** 完全なエージェントループが不要なら、`AIAgent` を完全に省略し、ハンドラーに `httpx` 経由で固定の通知を直接ポストさせます。より安く、より速く、プロバイダー依存もありません。

#### なぜこれが組み込みでないのか

以前のバージョンのHermesは、これを組み込みフックとして同梱し、Gateway起動のたびに素のデフォルトでエージェントを静かに起動していました。これはカスタムエンドポイントを持つユーザーを驚かせ、また機能が動いていることを知らないユーザーにとっては見えないものになっていました。ドキュメント化されたパターンとして — あなた自身が、あなたのフックディレクトリで構築するものとして — 残すことで、それが何をするのかを正確に把握でき、ファイルを書くことで自分の意思でオプトインできます。

### 仕組み

1. Gateway起動時に、`HookRegistry.discover_and_load()` が `~/.hermes/hooks/` をスキャンします
2. `HOOK.yaml` + `handler.py` を含む各サブディレクトリが動的にロードされます
3. ハンドラーは、宣言されたイベントに対して登録されます
4. 各ライフサイクルポイントで、`hooks.emit()` がマッチするすべてのハンドラーを発火させます
5. いずれのハンドラーでもエラーはキャッチされてログに記録されます — 壊れたフックがエージェントをクラッシュさせることはありません

:::info
Gatewayフックは**Gateway**（Telegram、Discord、Slack、WhatsApp、Teams）でのみ発火します。CLIはGatewayフックをロードしません。どこでも動作するフックには、[プラグインフック](#plugin-hooks)を使用してください。
:::

## プラグインフック {#plugin-hooks}

[プラグイン](/docs/user-guide/features/plugins)は、**CLIとGatewayの両方**のセッションで発火するフックを登録できます。これらは、プラグインの `register()` 関数内で `ctx.register_hook()` を介してプログラム的に登録されます。

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
```

**すべてのフックに共通の一般ルール:**

- コールバックは**キーワード引数**を受け取ります。前方互換性のため、常に `**kwargs` を受け取るようにしてください — 将来のバージョンでプラグインを壊すことなく新しいパラメーターが追加される可能性があります。
- コールバックが**クラッシュ**した場合、それはログに記録されてスキップされます。他のフックとエージェントは通常どおり続行します。不正な動作をするプラグインがエージェントを壊すことは決してありません。
- 2つのフックの戻り値が動作に影響します: [`pre_tool_call`](#pre_tool_call) はツールを**ブロック**でき、[`pre_llm_call`](#pre_llm_call) はLLM呼び出しに**コンテキストを注入**できます。それ以外のフックはすべて、結果を待たないオブザーバーです。

### クイックリファレンス

| フック | 発火タイミング | 戻り値 |
|------|-----------|---------|
| [`pre_tool_call`](#pre_tool_call) | 任意のツールの実行前 | 呼び出しを拒否する `{"action": "block", "message": str}` |
| [`post_tool_call`](#post_tool_call) | 任意のツールの返却後 | 無視される |
| [`pre_llm_call`](#pre_llm_call) | ターンごとに1回、ツール呼び出しループの前 | ユーザーメッセージにコンテキストを前置する `{"context": str}` |
| [`post_llm_call`](#post_llm_call) | ターンごとに1回、ツール呼び出しループの後 | 無視される |
| [`on_session_start`](#on_session_start) | 新しいセッションの作成時（最初のターンのみ） | 無視される |
| [`on_session_end`](#on_session_end) | セッション終了時 | 無視される |
| [`on_session_finalize`](#on_session_finalize) | CLI/Gatewayがアクティブなセッションを破棄するとき（フラッシュ、保存、統計） | 無視される |
| [`on_session_reset`](#on_session_reset) | Gatewayが新しいセッションキーに切り替えるとき（例: `/new`、`/reset`） | 無視される |
| [`subagent_stop`](#subagent_stop) | `delegate_task` の子が終了したとき | 無視される |
| [`pre_gateway_dispatch`](#pre_gateway_dispatch) | Gatewayがユーザーメッセージを受信し、認証 + ディスパッチの前 | フローに影響を与える `{"action": "skip" \| "rewrite" \| "allow", ...}` |
| [`pre_approval_request`](#pre_approval_request) | 危険なコマンドがユーザー承認を必要とし、プロンプト/通知が送信される前 | 無視される |
| [`post_approval_response`](#post_approval_response) | ユーザーが承認プロンプトに応答した（またはタイムアウトした） | 無視される |
| [`transform_tool_result`](#transform_tool_result) | 任意のツールの返却後、結果がモデルに渡される前 | 結果を置き換える `str`、変更しない場合は `None` |
| [`transform_terminal_output`](#transform_terminal_output) | `terminal` ツール内、切り詰め/ANSI除去/秘匿化の前 | 生の出力を置き換える `str`、変更しない場合は `None` |
| [`transform_llm_output`](#transform_llm_output) | ツール呼び出しループの完了後、最終応答が配信される前 | 応答テキストを置き換える `str`、変更しない場合は `None`/空 |

---

### `pre_tool_call`

すべてのツール実行の**直前**に発火します — 組み込みツールとプラグインツールの両方です。

**コールバックのシグネチャ:**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `tool_name` | `str` | これから実行されるツールの名前（例: `"terminal"`、`"web_search"`、`"read_file"`） |
| `args` | `dict` | モデルがツールに渡した引数 |
| `task_id` | `str` | セッション/タスクの識別子。設定されていない場合は空文字列。 |

**発火:** `model_tools.py` 内、`handle_function_call()` の中で、ツールのハンドラーが実行される前に発火します。ツール呼び出しごとに1回発火します — モデルが3つのツールを並列に呼び出すと、これは3回発火します。

**戻り値 — 呼び出しを拒否する:**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
```

エージェントはツールを短絡し、`message` をモデルに返すエラーとします。最初にマッチしたブロックディレクティブが優先されます（最初にPythonプラグインが登録され、その後にシェルフック）。それ以外の戻り値は無視されるため、既存のオブザーバー専用コールバックは変更なしで動作し続けます。

**ユースケース:** ログ記録、監査証跡、ツール呼び出しカウンター、危険な操作のブロック、レート制限、ユーザーごとのポリシー強制。

**例 — ツール呼び出しの監査ログ:**

```python
import json, logging
from datetime import datetime

logger = logging.getLogger(__name__)

def audit_tool_call(tool_name, args, task_id, **kwargs):
    logger.info("TOOL_CALL session=%s tool=%s args=%s",
                task_id, tool_name, json.dumps(args)[:200])

def register(ctx):
    ctx.register_hook("pre_tool_call", audit_tool_call)
```

**例 — 危険なツールに対する警告:**

```python
DANGEROUS = {"terminal", "write_file", "patch"}

def warn_dangerous(tool_name, **kwargs):
    if tool_name in DANGEROUS:
        print(f"⚠ Executing potentially dangerous tool: {tool_name}")

def register(ctx):
    ctx.register_hook("pre_tool_call", warn_dangerous)
```

---

### `post_tool_call`

すべてのツール実行の返却**直後**に発火します。

**コールバックのシグネチャ:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `tool_name` | `str` | たった今実行されたツールの名前 |
| `args` | `dict` | モデルがツールに渡した引数 |
| `result` | `str` | ツールの戻り値（常にJSON文字列） |
| `task_id` | `str` | セッション/タスクの識別子。設定されていない場合は空文字列。 |
| `duration_ms` | `int` | ツールのディスパッチにかかった時間（ミリ秒単位、`registry.dispatch()` の周囲を `time.monotonic()` で計測）。 |

**発火:** `model_tools.py` 内、`handle_function_call()` の中で、ツールのハンドラーが返却した後に発火します。ツール呼び出しごとに1回発火します。ツールがハンドルされない例外を発生させた場合は発火**しません**（代わりにエラーがキャッチされてエラーJSON文字列として返され、`post_tool_call` はそのエラー文字列を `result` として発火します）。

**戻り値:** 無視されます。

**ユースケース:** ツール結果のログ記録、メトリクス収集、ツールの成功/失敗率の追跡、レイテンシーダッシュボード、ツールごとの予算アラート、特定のツールが完了したときの通知送信。

**例 — ツール使用状況のメトリクスを追跡する:**

```python
from collections import Counter, defaultdict
import json

_tool_counts = Counter()
_error_counts = Counter()
_latency_ms = defaultdict(list)

def track_metrics(tool_name, result, duration_ms=0, **kwargs):
    _tool_counts[tool_name] += 1
    _latency_ms[tool_name].append(duration_ms)
    try:
        parsed = json.loads(result)
        if "error" in parsed:
            _error_counts[tool_name] += 1
    except (json.JSONDecodeError, TypeError):
        pass

def register(ctx):
    ctx.register_hook("post_tool_call", track_metrics)
```

---

### `pre_llm_call`

ツール呼び出しループが始まる前に、**ターンごとに1回**発火します。これは**戻り値が使用される唯一のフック**です — 現在のターンのユーザーメッセージにコンテキストを注入できます。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 現在のセッションの一意の識別子 |
| `user_message` | `str` | このターンのユーザーの元のメッセージ（スキル注入前） |
| `conversation_history` | `list` | 完全なメッセージリストのコピー（OpenAI形式: `[{"role": "user", "content": "..."}]`） |
| `is_first_turn` | `bool` | 新しいセッションの最初のターンの場合は `True`、以降のターンでは `False` |
| `model` | `str` | モデル識別子（例: `"anthropic/claude-sonnet-4.6"`） |
| `platform` | `str` | セッションが実行されている場所: `"cli"`、`"telegram"`、`"discord"` など |

**発火:** `run_agent.py` 内、`run_conversation()` の中で、コンテキスト圧縮の後、メインの `while` ループの前に発火します。`run_conversation()` の呼び出しごと（つまりユーザーのターンごとに1回）に発火し、ツールループ内のAPI呼び出しごとには発火しません。

**戻り値:** コールバックが `"context"` キーを持つdict、または空でないプレーン文字列を返した場合、そのテキストが現在のターンのユーザーメッセージに追加されます。注入しない場合は `None` を返します。

```python
# コンテキストを注入する
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# プレーン文字列（同等）
return "Recalled memories:\n- User likes Python"

# 注入しない
return None
```

**コンテキストが注入される場所:** 常に**ユーザーメッセージ**であり、システムプロンプトには注入されません。これによりプロンプトキャッシュが保たれます — システムプロンプトはターンをまたいで同一のままなので、キャッシュされたトークンが再利用されます。システムプロンプトはHermesの領域です（モデルガイダンス、ツールの強制、パーソナリティ、スキル）。プラグインは、ユーザーの入力と並んでコンテキストを提供します。

注入されたすべてのコンテキストは**一時的**です — API呼び出し時にのみ追加されます。会話履歴内の元のユーザーメッセージが変更されることはなく、セッションデータベースに永続化されるものもありません。

**複数のプラグイン**がコンテキストを返した場合、それらの出力はプラグイン検出順（ディレクトリ名のアルファベット順）に2つの改行で結合されます。

**ユースケース:** メモリの想起、RAGコンテキストの注入、ガードレール、ターンごとの分析。

**例 — メモリの想起:**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall(session_id, user_message, is_first_turn, **kwargs):
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None
        text = "Recalled context:\n" + "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None

def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
```

**例 — ガードレール:**

```python
POLICY = "Never execute commands that delete files without explicit user confirmation."

def guardrails(**kwargs):
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", guardrails)
```

---

### `post_llm_call`

ツール呼び出しループが完了し、エージェントが最終応答を生成した後に、**ターンごとに1回**発火します。**成功した**ターンでのみ発火します — ターンが中断された場合は発火しません。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 現在のセッションの一意の識別子 |
| `user_message` | `str` | このターンのユーザーの元のメッセージ |
| `assistant_response` | `str` | このターンに対するエージェントの最終テキスト応答 |
| `conversation_history` | `list` | ターン完了後の完全なメッセージリストのコピー |
| `model` | `str` | モデル識別子 |
| `platform` | `str` | セッションが実行されている場所 |

**発火:** `run_agent.py` 内、`run_conversation()` の中で、ツールループが最終応答とともに終了した後に発火します。`if final_response and not interrupted` でガードされています — そのため、ユーザーがターン途中で中断した場合や、エージェントが応答を生成せずにイテレーション制限に達した場合は発火**しません**。

**戻り値:** 無視されます。

**ユースケース:** 会話データを外部メモリシステムに同期する、応答品質メトリクスを計算する、ターンの要約をログに記録する、フォローアップアクションをトリガーする。

**例 — 外部メモリへの同期:**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def sync_memory(session_id, user_message, assistant_response, **kwargs):
    try:
        httpx.post(f"{MEMORY_API}/store", json={
            "session_id": session_id,
            "user": user_message,
            "assistant": assistant_response,
        }, timeout=5)
    except Exception:
        pass  # ベストエフォート

def register(ctx):
    ctx.register_hook("post_llm_call", sync_memory)
```

**例 — 応答の長さを追跡する:**

```python
import logging
logger = logging.getLogger(__name__)

def log_response_length(session_id, assistant_response, model, **kwargs):
    logger.info("RESPONSE session=%s model=%s chars=%d",
                session_id, model, len(assistant_response or ""))

def register(ctx):
    ctx.register_hook("post_llm_call", log_response_length)
```

---

### `on_session_start`

まったく新しいセッションが作成されたときに**1回**発火します。セッションの継続時（ユーザーが既存のセッションで2番目のメッセージを送信したとき）には発火**しません**。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 新しいセッションの一意の識別子 |
| `model` | `str` | モデル識別子 |
| `platform` | `str` | セッションが実行されている場所 |

**発火:** `run_agent.py` 内、`run_conversation()` の中で、新しいセッションの最初のターン中に発火します — 具体的には、システムプロンプトが構築された後、ツールループが始まる前です。チェックは `if not conversation_history`（先行メッセージがない = 新しいセッション）です。

**戻り値:** 無視されます。

**ユースケース:** セッションスコープの状態の初期化、キャッシュのウォーミング、外部サービスへのセッション登録、セッション開始のログ記録。

**例 — セッションキャッシュを初期化する:**

```python
_session_caches = {}

def init_session(session_id, model, platform, **kwargs):
    _session_caches[session_id] = {
        "model": model,
        "platform": platform,
        "tool_calls": 0,
        "started": __import__("datetime").datetime.now().isoformat(),
    }

def register(ctx):
    ctx.register_hook("on_session_start", init_session)
```

---

### `on_session_end`

すべての `run_conversation()` 呼び出しの**最後**に、結果にかかわらず発火します。また、ユーザーが終了したときにエージェントがターン途中だった場合、CLIの終了ハンドラーからも発火します。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | セッションの一意の識別子 |
| `completed` | `bool` | エージェントが最終応答を生成した場合は `True`、そうでない場合は `False` |
| `interrupted` | `bool` | ターンが中断された場合は `True`（ユーザーが新しいメッセージを送信、`/stop`、または終了） |
| `model` | `str` | モデル識別子 |
| `platform` | `str` | セッションが実行されている場所 |

**発火:** 2か所で発火します:
1. **`run_agent.py`** — すべての `run_conversation()` 呼び出しの最後、すべてのクリーンアップの後。ターンがエラーになった場合でも常に発火します。
2. **`cli.py`** — CLIのatexitハンドラー内。ただし、終了が発生したときにエージェントがターン途中だった場合（`_agent_running=True`）に**のみ**発火します。これは処理中のCtrl+Cや `/exit` をキャッチします。この場合、`completed=False` かつ `interrupted=True` です。

**戻り値:** 無視されます。

**ユースケース:** バッファのフラッシュ、接続のクローズ、セッション状態の永続化、セッション継続時間のログ記録、`on_session_start` で初期化したリソースのクリーンアップ。

**例 — フラッシュとクリーンアップ:**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # 蓄積したデータをディスクまたは外部サービスにフラッシュする
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"Session {session_id} ended: {status}, {cache['tool_calls']} tool calls")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**例 — セッション継続時間の追跡:**

```python
import time, logging
logger = logging.getLogger(__name__)

_start_times = {}

def on_start(session_id, **kwargs):
    _start_times[session_id] = time.time()

def on_end(session_id, completed, interrupted, **kwargs):
    start = _start_times.pop(session_id, None)
    if start:
        duration = time.time() - start
        logger.info("SESSION_DURATION session=%s seconds=%.1f completed=%s interrupted=%s",
                     session_id, duration, completed, interrupted)

def register(ctx):
    ctx.register_hook("on_session_start", on_start)
    ctx.register_hook("on_session_end", on_end)
```

---

### `on_session_finalize`

CLIまたはGatewayがアクティブなセッションを**破棄**するときに発火します — 例えば、ユーザーが `/new` を実行したとき、GatewayがアイドルセッションをGCしたとき、またはCLIがアクティブなエージェントとともに終了したときです。これは、退出するセッションのアイデンティティが失われる前に、それに紐づく状態をフラッシュする最後のチャンスです。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` または `None` | 退出するセッションID。アクティブなセッションが存在しなかった場合は `None` になることがある。 |
| `platform` | `str` | `"cli"` またはメッセージングプラットフォーム名（`"telegram"`、`"discord"` など）。 |

**発火:** `cli.py`（`/new` / CLI終了時）と `gateway/run.py`（セッションがリセットまたはGCされたとき）で発火します。Gateway側では常に `on_session_reset` とペアになります。

**戻り値:** 無視されます。

**ユースケース:** セッションIDが破棄される前に最終的なセッションメトリクスを永続化する、セッションごとのリソースをクローズする、最終的なテレメトリイベントを発行する、キューに入った書き込みをドレインする。

---

### `on_session_reset`

Gatewayがアクティブなチャットに対して**新しいセッションキーに切り替える**ときに発火します — ユーザーが `/new`、`/reset`、`/clear` を呼び出したとき、またはアダプターがアイドルウィンドウ後に新しいセッションを選んだときです。これにより、次の `on_session_start` を待つことなく、会話状態がワイプされたという事実にプラグインが反応できます。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 新しいセッションのID（すでに新しい値にローテーション済み）。 |
| `platform` | `str` | メッセージングプラットフォーム名。 |

**発火:** `gateway/run.py` 内、新しいセッションキーが割り当てられた直後、次の受信メッセージが処理される前に発火します。Gateway上での順序は: `on_session_finalize(old_id)` → 切り替え → `on_session_reset(new_id)` → 最初の受信ターンで `on_session_start(new_id)` です。

**戻り値:** 無視されます。

**ユースケース:** `session_id` をキーとするセッションごとのキャッシュのリセット、「セッションローテーション」分析の発行、新しい状態バケットの準備。

---

完全なウォークスルー（ツールスキーマ、ハンドラー、高度なフックパターンを含む）については、**[プラグイン構築ガイド](/docs/guides/build-a-hermes-plugin)** を参照してください。

---

### `subagent_stop`

`delegate_task` が完了した後、**子エージェントごとに1回**発火します。単一のタスクを委譲したか3つのバッチを委譲したかにかかわらず、このフックは各子に対して1回ずつ、親スレッド上で直列に発火します。

**コールバックのシグネチャ:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                duration_ms: int, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `parent_session_id` | `str` | 委譲する親エージェントのセッションID |
| `child_role` | `str \| None` | 子に設定されたオーケストレーターのロールタグ（機能が有効でない場合は `None`） |
| `child_summary` | `str \| None` | 子が親に返した最終応答 |
| `child_status` | `str` | `"completed"`、`"failed"`、`"interrupted"`、または `"error"` |
| `duration_ms` | `int` | 子の実行に費やされた実時間（ミリ秒単位） |

**発火:** `tools/delegate_tool.py` 内、`ThreadPoolExecutor.as_completed()` がすべての子のfutureをドレインした後に発火します。発火は親スレッドにマーシャリングされるため、フック作者は並行するコールバック実行について考える必要はありません。

**戻り値:** 無視されます。

**ユースケース:** オーケストレーションアクティビティのログ記録、課金のための子の継続時間の累積、委譲後の監査レコードの書き込み。

**例 — オーケストレーターのアクティビティをログに記録する:**

```python
import logging
logger = logging.getLogger(__name__)

def log_subagent(parent_session_id, child_role, child_status, duration_ms, **kwargs):
    logger.info(
        "SUBAGENT parent=%s role=%s status=%s duration_ms=%d",
        parent_session_id, child_role, child_status, duration_ms,
    )

def register(ctx):
    ctx.register_hook("subagent_stop", log_subagent)
```

:::info
大量の委譲（例: オーケストレーターのロール × 5つのリーフ × ネストされた深さ）では、`subagent_stop` はターンごとに何度も発火します。コールバックを高速に保ち、重い処理はバックグラウンドキューに回してください。
:::

---

### `pre_gateway_dispatch`

Gatewayで受信した`MessageEvent`ごとに、内部イベントガードの後、認証/ペアリングとエージェントディスパッチの**前**に**1回**発火します。これは、どの単一プラットフォームアダプターにもきれいに収まらない、Gatewayレベルのメッセージフローポリシー（リッスン専用ウィンドウ、人間へのハンドオーバー、チャットごとのルーティングなど）のインターセプトポイントです。

**コールバックのシグネチャ:**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `event` | `MessageEvent` | 正規化された受信メッセージ（`.text`、`.source`、`.message_id`、`.internal` などを持つ）。 |
| `gateway` | `GatewayRunner` | アクティブなGatewayランナー。プラグインがサイドチャネルの返信（オーナー通知など）のために `gateway.adapters[platform].send(...)` を呼び出せる。 |
| `session_store` | `SessionStore` | `session_store.append_to_transcript(...)` を介したサイレントなトランスクリプト取り込み用。 |

**発火:** `gateway/run.py` 内、`GatewayRunner._handle_message()` の中で、`is_internal` が計算された直後に発火します。**内部イベントはフックを完全にスキップします**（それらはシステム生成 — バックグラウンドプロセスの完了など — であり、ユーザー向けのポリシーでゲートキープされてはなりません）。

**戻り値:** `None` またはdict。最初に認識されたアクションdictが優先され、残りのプラグイン結果は無視されます。プラグインコールバック内の例外はキャッチされてログに記録されます。エラー時、Gatewayは常に通常のディスパッチにフォールスルーします。

| 戻り値 | 効果 |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | メッセージをドロップする — エージェントの返信なし、ペアリングフローなし、認証なし。プラグインが処理したとみなされる（例: トランスクリプトにサイレントに取り込まれた）。 |
| `{"action": "rewrite", "text": "new text"}` | `event.text` を置き換え、変更されたイベントで通常のディスパッチを続行する。バッファされたアンビエントメッセージを単一のプロンプトに折りたたむのに便利。 |
| `{"action": "allow"}` / `None` | 通常のディスパッチ — 完全な認証 / ペアリング / エージェントループのチェーンを実行する。 |

**ユースケース:** リッスン専用のグループチャット（タグ付けされたときのみ応答し、アンビエントメッセージをコンテキストにバッファする）、人間へのハンドオーバー（オーナーが手動でチャットを処理する間、顧客メッセージをサイレントに取り込む）、プロファイルごとのレート制限、ポリシー駆動のルーティング。

**例 — ペアリングコードをトリガーせずに未承認のDMをサイレントにドロップする:**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**例 — メンション時にアンビエントメッセージのバッファを単一のプロンプトに書き換える:**

```python
_buffers = {}

def buffer_or_rewrite(event, **kwargs):
    key = (event.source.platform, event.source.chat_id)
    buf = _buffers.setdefault(key, [])
    if _bot_mentioned(event.text):
        combined = "\n".join(buf + [event.text])
        buf.clear()
        return {"action": "rewrite", "text": combined}
    buf.append(event.text)
    return {"action": "skip", "reason": "ambient-buffered"}

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", buffer_or_rewrite)
```

---

### `pre_approval_request`

承認リクエストがユーザーに表示される**直前**に発火します — あらゆるサーフェスをカバーします: インタラクティブCLI、Ink TUI、Gatewayプラットフォーム（Telegram、Discord、Slack、WhatsApp、Matrix など）、ACPクライアント（VS Code、Zed、JetBrains）。

ここは、カスタム通知機構を組み込むのに適した場所です — 例えば、許可/拒否の通知をポップアップするmacOSのメニューバーアプリや、すべての承認リクエストをコンテキストとともに記録する監査ログなどです。

**コールバックのシグネチャ:**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    **kwargs,
):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `command` | `str` | 承認待ちのシェルコマンド |
| `description` | `str` | コマンドがフラグ付けされた理由（人間が読める形式、複数のパターンがマッチした場合は結合される） |
| `pattern_key` | `str` | 承認をトリガーしたプライマリパターンキー（例: `"rm_rf"`、`"sudo"`） |
| `pattern_keys` | `list[str]` | マッチしたすべてのパターンキー |
| `session_key` | `str` | セッション識別子。チャットごとに通知をスコープするのに便利。 |
| `surface` | `str` | インタラクティブCLI/TUIプロンプトの場合は `"cli"`、非同期のプラットフォーム承認の場合は `"gateway"` |

**戻り値:** 無視されます。ここでのフックはオブザーバー専用です。承認を拒否したり事前に応答したりすることはできません。ツールが承認システムに到達する前にブロックするには、[`pre_tool_call`](#pre_tool_call) を使用してください。

**ユースケース:** デスクトップ通知、プッシュアラート、監査ログ、Slack Webhook、エスカレーションルーティング、メトリクス。

**例 — macOSでのデスクトップ通知:**

```python
import subprocess

def notify_approval(command, description, session_key, **kwargs):
    title = "Hermes needs approval"
    body = f"{description}: {command[:80]}"
    subprocess.Popen([
        "osascript", "-e",
        f'display notification "{body}" with title "{title}"',
    ])

def register(ctx):
    ctx.register_hook("pre_approval_request", notify_approval)
```

---

### `post_approval_response`

ユーザーが承認プロンプトに応答した（またはプロンプトがタイムアウトした）**後**に発火します。

**コールバックのシグネチャ:**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    choice: str,
    **kwargs,
):
```

`pre_approval_request` と同じkwargsに加えて:

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `choice` | `str` | `"once"`、`"session"`、`"always"`、`"deny"`、`"timeout"` のいずれか |

**戻り値:** 無視されます。

**ユースケース:** マッチするデスクトップ通知を閉じる、最終的な決定を監査ログに記録する、メトリクスを更新する、レートリミッターを進める。

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `transform_tool_result`

ツールが返却した**後**、結果が会話に追加される**前**に発火します。プラグインが、ターミナル出力だけでなく、任意のツールの結果文字列を、モデルが見る前に書き換えられるようにします。

**コールバックのシグネチャ:**

```python
def my_callback(
    tool_name: str,
    arguments: dict,
    result: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `tool_name` | `str` | 結果を生成したツール（`read_file`、`web_extract`、`delegate_task` など）。 |
| `arguments` | `dict` | モデルがツールを呼び出した際の引数。 |
| `result` | `str` | ツールの生の結果文字列（切り詰めおよびANSI除去後）。 |
| `task_id` | `str \| None` | RL/ベンチマーク環境内で実行する際のタスク/セッションID。 |

**戻り値:** 結果を置き換える `str`（返された文字列がモデルが見るもの）、変更しない場合は `None`。

**ユースケース:** `web_extract` 出力から組織固有のPIIを秘匿化する、長いJSONツール応答を要約ヘッダーでラップする、検索拡張ヒントを `read_file` 結果に注入する、`delegate_task` のサブエージェントレポートをプロジェクト固有のスキーマに書き換える。

```python
import re
SECRET = re.compile(r"sk-[A-Za-z0-9]{32,}")

def redact_secrets(tool_name, result, **kwargs):
    if SECRET.search(result):
        return SECRET.sub("[REDACTED]", result)
    return None

def register(ctx):
    ctx.register_hook("transform_tool_result", redact_secrets)
```

すべてのツールに適用されます。ターミナル限定の書き換えについては、以下の `transform_terminal_output` を参照してください — そちらはより範囲が狭く、パイプラインの早い段階（切り詰め前、秘匿化前）で実行されます。

---

### `transform_terminal_output`

`terminal` ツールのフォアグラウンド出力パイプライン内で、デフォルトの50 KB切り詰め、ANSI除去、秘匿化の**前**に発火します。プラグインが、下流の処理が触れる前にシェルコマンドの生のstdout/stderrを書き換えられるようにします。

**コールバックのシグネチャ:**

```python
def my_callback(
    command: str,
    output: str,
    exit_code: int,
    cwd: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `command` | `str` | 出力を生成したシェルコマンド。 |
| `output` | `str` | 生の結合済みstdout/stderr（非常に大きい場合がある — 切り詰めはフックの後に発生）。 |
| `exit_code` | `int` | プロセスの終了コード。 |
| `cwd` | `str` | コマンドが実行された作業ディレクトリ。 |

**戻り値:** 出力を置き換える `str`、変更しない場合は `None`。

**ユースケース:** 大量の出力を生成するコマンド（`du -ah`、`find`、`tree`）の要約を注入する、下流のフックが処理方法を判断できるようプロジェクト固有のマーカーで出力にタグ付けする、実行間で変動してプロンプトキャッシュを無効化するタイミングノイズを除去する。

```python
def summarize_find(command, output, **kwargs):
    if command.startswith("find ") and len(output) > 50_000:
        lines = output.count("\n")
        head = "\n".join(output.splitlines()[:40])
        return f"{head}\n\n[summary: {lines} paths total, showing first 40]"
    return None

def register(ctx):
    ctx.register_hook("transform_terminal_output", summarize_find)
```

`transform_tool_result`（他のすべてのツールをカバーする）とうまく組み合わせられます。

---

### `transform_llm_output`

ツール呼び出しループが完了し、モデルが最終応答を生成した後、その応答がユーザー（CLI、Gateway、またはプログラム的な呼び出し元）に配信される**前**に、**ターンごとに1回**発火します。プラグインが、古典的なプログラミング手法を使ってアシスタントの最終テキストを書き換えられるようにします — SOULのフレーバーテキストやスキル駆動の変換に余分な推論トークンを消費しません。

**コールバックのシグネチャ:**

```python
def my_callback(
    response_text: str,
    session_id: str,
    model: str,
    platform: str,
    **kwargs,
) -> str | None:
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `response_text` | `str` | このターンのアシスタントの最終応答テキスト。 |
| `session_id` | `str` | この会話のセッションID（ワンショット実行では空の場合がある）。 |
| `model` | `str` | 応答を生成したモデル名（例: `anthropic/claude-sonnet-4.6`）。 |
| `platform` | `str` | 配信プラットフォーム（`cli`、`telegram`、`discord` など。未設定の場合は空）。 |

**戻り値:** 応答テキストを置き換える空でない `str`、変更しない場合は `None` または空文字列。複数のプラグインが登録された場合、**最初の空でない文字列が優先**されます — `transform_tool_result` と同様です。

**ユースケース:** パーソナリティ/語彙の変換を適用する（海賊言葉、スポンジ・ボブ）、最終テキストからユーザー固有の識別子を秘匿化する、プロジェクト固有の署名フッターを追加する、SOULの指示にトークンを費やすことなくハウススタイルガイドを強制する。

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # 変更せずそのまま通す
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

このフックは、空でない、中断されていない応答に対してガードされています — ストップボタンによる中断や空のターンでは発火しません。例外は警告としてログに記録され、エージェントの実行を壊すことはありません。

---

## シェルフック {#shell-hooks}

`cli-config.yaml` でシェルスクリプトのフックを宣言すると、対応するプラグインフックイベントが発火するたびに、Hermesがそれらをサブプロセスとして実行します — CLIとGatewayの両方のセッションで。Pythonプラグインの作成は不要です。

ドロップインの単一ファイルスクリプト（Bash、Python、シェバンを持つもの何でも）で次のことをしたいときに、シェルフックを使用してください:

- **ツール呼び出しをブロックする** — 危険な `terminal` コマンドを拒否する、ディレクトリごとのポリシーを強制する、破壊的な `write_file` / `patch` 操作に承認を必要とする。
- **ツール呼び出しの後に実行する** — エージェントがたった今書いたPythonやTypeScriptのファイルを自動フォーマットする、API呼び出しをログに記録する、CIワークフローをトリガーする。
- **次のLLMターンにコンテキストを注入する** — `git status` の出力、現在の曜日、または取得したドキュメントをユーザーメッセージに前置する（[`pre_llm_call`](#pre_llm_call) を参照）。
- **ライフサイクルイベントを観測する** — サブエージェントが完了したとき（`subagent_stop`）やセッションが開始したとき（`on_session_start`）にログ行を書く。

シェルフックは、CLI起動時（`hermes_cli/main.py`）とGateway起動時（`gateway/run.py`）の両方で `agent.shell_hooks.register_from_config(cfg)` を呼び出すことで登録されます。Pythonプラグインフックと自然に組み合わさります — どちらも同じディスパッチャーを通ります。

### 一覧での比較

| 観点 | シェルフック | [プラグインフック](#plugin-hooks) | [Gatewayフック](#gateway-event-hooks) |
|-----------|-------------|-------------------------------|---------------------------------------|
| 宣言場所 | `~/.hermes/config.yaml` 内の `hooks:` ブロック | `plugin.yaml` プラグイン内の `register()` | `HOOK.yaml` + `handler.py` のディレクトリ |
| 配置場所 | `~/.hermes/agent-hooks/`（慣例として） | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| 言語 | 任意（Bash、Python、Goバイナリ など） | Pythonのみ | Pythonのみ |
| 実行場所 | CLI + Gateway | CLI + Gateway | Gatewayのみ |
| イベント | `VALID_HOOKS`（`subagent_stop` を含む） | `VALID_HOOKS` | Gatewayライフサイクル（`gateway:startup`、`agent:*`、`command:*`） |
| ツール呼び出しのブロック | 可能（`pre_tool_call`） | 可能（`pre_tool_call`） | 不可 |
| LLMコンテキストの注入 | 可能（`pre_llm_call`） | 可能（`pre_llm_call`） | 不可 |
| 同意 | `(event, command)` のペアごとに初回利用時のプロンプト | 暗黙（Pythonプラグインの信頼） | 暗黙（ディレクトリの信頼） |
| プロセス間の分離 | あり（サブプロセス） | なし（インプロセス） | なし（インプロセス） |

### 設定スキーマ

```yaml
hooks:
  <event_name>:                  # VALID_HOOKS に含まれている必要がある
    - matcher: "<regex>"         # 任意。pre/post_tool_call でのみ使用
      command: "<shell command>" # 必須。shlex.split, shell=False で実行される
      timeout: <seconds>         # 任意。デフォルト60、最大300でクランプ
```

イベント名は[プラグインフックイベント](#plugin-hooks)のいずれかである必要があります。タイプミスは「Did you mean X?」という警告を生成してスキップされます。単一エントリ内の不明なキーは無視されます。`command` の欠落は警告付きでスキップされます。`timeout > 300` は警告とともにクランプされます。

### JSONワイヤープロトコル

イベントが発火するたびに、Hermesはマッチするフックごと（matcherが許可する範囲で）にサブプロセスを起動し、JSONペイロードを**stdin**にパイプして、**stdout**をJSONとして読み戻します。

**stdin — スクリプトが受け取るペイロード:**

```json
{
  "hook_event_name": "pre_tool_call",
  "tool_name":       "terminal",
  "tool_input":      {"command": "rm -rf /"},
  "session_id":      "sess_abc123",
  "cwd":             "/home/user/project",
  "extra":           {"task_id": "...", "tool_call_id": "..."}
}
```

`tool_name` と `tool_input` は、ツール以外のイベント（`pre_llm_call`、`subagent_stop`、セッションライフサイクル）では `null` になります。`extra` dictは、イベント固有のすべてのkwargs（`user_message`、`conversation_history`、`child_role`、`duration_ms` など）を運びます。シリアライズできない値は、省略されるのではなく文字列化されます。

**stdout — 任意の応答:**

```jsonc
// pre_tool_call をブロックする（両方の形式を受け付け、内部で正規化される）:
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Codeスタイル
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes標準

// pre_llm_call にコンテキストを注入する:
{"context": "Today is Friday, 2026-04-17"}

// サイレントなno-op — 空 / マッチしない出力はいずれも問題ない:
```

不正なJSON、ゼロ以外の終了コード、タイムアウトは警告をログに記録しますが、エージェントループを中断することは決してありません。

### 実践例

#### 1. 書き込みのたびにPythonファイルを自動フォーマットする

```yaml
# ~/.hermes/config.yaml
hooks:
  post_tool_call:
    - matcher: "write_file|patch"
      command: "~/.hermes/agent-hooks/auto-format.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/auto-format.sh
payload="$(cat -)"
path=$(echo "$payload" | jq -r '.tool_input.path // empty')
[[ "$path" == *.py ]] && command -v black >/dev/null && black "$path" 2>/dev/null
printf '{}\n'
```

エージェントがコンテキスト内で持っているファイルのビューは、自動的に再読み込みされ**ません** — 再フォーマットはディスク上のファイルにのみ影響します。以降の `read_file` 呼び出しでフォーマット済みバージョンが取得されます。

#### 2. 破壊的な `terminal` コマンドをブロックする

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal"
      command: "~/.hermes/agent-hooks/block-rm-rf.sh"
      timeout: 5
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/block-rm-rf.sh
payload="$(cat -)"
cmd=$(echo "$payload" | jq -r '.tool_input.command // empty')
if echo "$cmd" | grep -qE 'rm[[:space:]]+-rf?[[:space:]]+/'; then
  printf '{"decision": "block", "reason": "blocked: rm -rf / is not permitted"}\n'
else
  printf '{}\n'
fi
```

#### 3. 各ターンに `git status` を注入する（Claude-Codeの `UserPromptSubmit` 相当）

```yaml
hooks:
  pre_llm_call:
    - command: "~/.hermes/agent-hooks/inject-cwd-context.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/inject-cwd-context.sh
cat - >/dev/null   # stdinペイロードを破棄
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("Uncommitted changes in cwd:\n" + $s)}'
else
  printf '{}\n'
fi
```

Claude Codeの `UserPromptSubmit` イベントは、意図的にHermesの別個のイベントになっていません — `pre_llm_call` が同じ場所で発火し、すでにコンテキスト注入をサポートしています。ここではそれを使用してください。

#### 4. すべてのサブエージェント完了をログに記録する

```yaml
hooks:
  subagent_stop:
    - command: "~/.hermes/agent-hooks/log-orchestration.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/log-orchestration.sh
log=~/.hermes/logs/orchestration.log
jq -c '{ts: now, parent: .session_id, extra: .extra}' < /dev/stdin >> "$log"
printf '{}\n'
```

### 同意モデル

一意な `(event, command)` のペアそれぞれについて、Hermesが初めて見たときにユーザーに承認を求め、その後その決定を `~/.hermes/shell-hooks-allowlist.json` に永続化します。以降の実行（CLIまたはGateway）ではプロンプトをスキップします。

3つの抜け道がインタラクティブなプロンプトをバイパスします — どれか1つで十分です:

1. CLIの `--accept-hooks` フラグ（例: `hermes --accept-hooks chat`）
2. `HERMES_ACCEPT_HOOKS=1` 環境変数
3. `cli-config.yaml` 内の `hooks_auto_accept: true`

非TTYの実行（Gateway、cron、CI）には、これら3つのうち1つが必要です — そうでないと、新しく追加されたフックは静かに未登録のままになり、警告をログに記録します。

**スクリプトの編集は黙示的に信頼されます。** 許可リストはスクリプトのハッシュではなく正確なコマンド文字列をキーとするため、ディスク上のスクリプトを編集しても同意は無効になりません。`hermes hooks doctor` はmtimeのドリフトをフラグ付けするので、編集を見つけて再承認するかどうかを判断できます。

### `hermes hooks` CLI

| コマンド | 何をするか |
|---------|--------------|
| `hermes hooks list` | 設定されたフックを、matcher、timeout、同意ステータスとともにダンプする |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | 合成ペイロードに対してマッチするすべてのフックを発火させ、パース済みの応答を表示する |
| `hermes hooks revoke <command>` | `<command>` にマッチするすべての許可リストエントリを削除する（次回の再起動時に有効になる） |
| `hermes hooks doctor` | 設定された各フックについて、実行ビット、許可リストステータス、mtimeドリフト、JSON出力の妥当性、おおよその実行時間をチェックする |

### セキュリティ

シェルフックは**あなたのフルユーザー権限**で実行されます — cronエントリやシェルエイリアスと同じ信頼境界です。`config.yaml` の `hooks:` ブロックを特権的な設定として扱ってください:

- 自分で書いたか、完全にレビューしたスクリプトのみを参照する。
- パスを監査しやすいよう、スクリプトを `~/.hermes/agent-hooks/` 内に置く。
- 共有設定をプルした後は、フックが登録される前に新しく追加されたフックを見つけるため `hermes hooks doctor` を再実行する。
- config.yamlがチームをまたいでバージョン管理されている場合、`hooks:` セクションを変更するPRを、CI設定をレビューするのと同じようにレビューする。

### 順序と優先順位

Pythonプラグインフックとシェルフックは、どちらも同じ `invoke_hook()` ディスパッチャーを通ります。Pythonプラグインが先に登録され（`discover_and_load()`）、シェルフックが2番目（`register_from_config()`）なので、同点の場合はPythonの `pre_tool_call` ブロック決定が優先されます。最初の有効なブロックが優先されます — アグリゲーターは、いずれかのコールバックが空でないメッセージを持つ `{"action": "block", "message": str}` を生成した時点で即座にreturnします。
