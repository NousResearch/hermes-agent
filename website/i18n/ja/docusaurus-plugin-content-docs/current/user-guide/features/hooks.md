---
sidebar_position: 6
title: "イベントフック"
description: "ライフサイクルの重要なポイントでカスタムコードを実行 — アクティビティのログ記録、アラート送信、Webhookへのポスト"
---

# イベントフック

Hermesには、ライフサイクルの重要なポイントでカスタムコードを実行する4つのフックシステムがあります:

| システム | 登録方法 | 実行場所 | ユースケース |
|--------|---------------|---------|----------|
| **[Gatewayフック](#gateway-event-hooks)** | `~/.hermes/hooks/` 内の `HOOK.yaml` + `handler.py` | Gatewayのみ | ログ記録、アラート、Webhook |
| **[プラグインフック](#plugin-hooks)** | [プラグイン](./plugins.md)内の `ctx.register_hook()` | CLI + Gateway | ツールのインターセプト、メトリクス、ガードレール |
| **[シェルフック](#shell-hooks)** | シェルスクリプトを指すプロファイルの `config.yaml` 内の `hooks:` ブロック | CLI + Gateway + Desktop/TUI/ダッシュボードチャット | ブロック、自動フォーマット、コンテキスト注入のためのドロップインスクリプト |
| **[アウトバウンドWebhook](#outbound-webhooks)** | `~/.hermes/config.yaml` 内の `hooks.outbound:` リスト | CLI + Gateway | 署名付きライフサイクルイベントを外部HTTPエンドポイント（CI、ダッシュボード、他のエージェント）にプッシュ |

フックのコールバックで発生したエラーは隔離されてログに記録され、エージェントがクラッシュすることはありません。ただし、フックはすべてが受動的なわけではありません。ディレクティブ/制御系のフックはフローを変更でき、変換系のフックはコンテンツを置き換えることができ、シェルの `pre_tool_call` フックはブロックしたりフェイルクローズしたりできます。

## Gatewayイベントフック {#gateway-event-hooks}

Gatewayフックは、メインのエージェントパイプラインをブロックすることなく、Gateway動作中（Telegram、Discord、Slack、WhatsApp、Teams）に自動的に発火します。

### 信頼モデル: ファイルを配置することがオプトイン {#gateway-hook-trust}

hooksディレクトリは **配置による信頼（trusted-by-placement）** の拡張ポイントです。これは `3988c3c245f`（2026年4月）以降に文書化された契約であり、このとき下の比較表で同意モデルが初めて「Implicit (dir trust)」と記録されました。有効化リストはなく、`plugins.enabled` / `plugins.disabled` も適用されません — Gatewayフックはプラグインではありません。読み込まれる内容は正確に次のとおりです:

- **いつ:** Gateway起動時に1回（`gateway/run_startup.py` から呼ばれる `HookRegistry.discover_and_load()`）。[マルチプロファイルGateway](/docs/user-guide/multi-profile-gateways) では、提供される各プロファイル自身の `hooks/` が、そのプロファイル内で最初にイベントが発火したときに読み込まれます。CLI、TUI、Desktop、cronはGatewayフックを読み込みません。
- **何を:** `<profile home>/hooks/`（デフォルトプロファイルでは `~/.hermes/hooks/`）のサブディレクトリのうち、空でない `events` リストを持つマッピングとしてパースされる `HOOK.yaml` **と** `handler.py` の両方を含むものすべて。どちらかのファイルが欠けているディレクトリは黙ってスキップされます。無効なマニフェストや空の `events` リストは、`[hooks] Skipping …` というログ行とともにスキップされます。
- **どのように:** `handler.py` はプロセス内でインポートされます — モジュール本体はインポート時に実行され、その `handle` 関数が宣言されたイベントに登録されます。Gatewayプロセスとして、Gateway自体と同じアクセス権（読み込まれた認証情報、ツール、プラグインの状態）で実行されます。サンドボックスも初回使用時の確認もなく、`HERMES_SAFE_MODE` でもこのローダーはスキップされません。

2つのファイルをディレクトリに置くこと**が**オプトインであり、`HOOK.yaml` またはディレクトリを削除（またはリネーム）することがオプトアウトです。プロファイルのホームに書き込める人は、`config.yaml` のシェルフックや `plugins.enabled` を通じてすでにあなたとしてコードを実行できるため、このディレクトリは `~/.hermes/` の他の部分と同じ信頼範囲内にあります — セキュリティページの[配置による信頼の拡張ポイント](../security.md#trusted-by-placement)を参照してください。プラグインを有効にする前に確認するのとまったく同じように、配置する前にフックの `handler.py` を確認してください。

### フックの作成

各フックは、2つのファイルを含む `~/.hermes/hooks/` 配下のディレクトリです:

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # Declares which events to listen for
    └── handler.py     # Python handler function
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
    """Called for each subscribed event. Must be named 'handle'."""
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

### 利用可能なイベント {#available-events}

| イベント | 発火タイミング | コンテキストキー |
|-------|---------------|--------------|
| `gateway:startup` | Gatewayプロセスの起動時 | `platforms`（アクティブなプラットフォーム名のリスト） |
| `session:start` | 新しいメッセージングセッションの作成時 | `platform`、`user_id`、`session_id`、`session_key` |
| `session:end` | セッション終了時（リセット前） | `platform`、`user_id`、`session_key` |
| `session:reset` | ユーザーが `/new` または `/reset` を実行したとき | `platform`、`user_id`、`session_key` |
| `session:compress` | セッションのコンテキスト圧縮が完了したとき | `platform`、`session_id`、`old_session_id`（その場で圧縮された場合は空）、`in_place`（bool — `true` = 同じIDのままトランスクリプトを圧縮、`false` = `old_session_id` からローテーション）、`compression_count` |
| `agent:start` | エージェントがメッセージの処理を開始したとき | `platform`、`user_id`、`chat_id`、`thread_id`（フォーラムトピック / スレッドのルートID。スレッド外では空）、`chat_type`（`"dm"` \| `"group"` \| `"forum"`。不明な場合は空）、`session_id`、`message`（500文字に切り詰め） |
| `agent:step` | ツール呼び出しループの各イテレーション | `platform`、`user_id`、`session_id`、`iteration`、`tool_names` |
| `agent:end` | エージェントが処理を完了したとき | `agent:start` と同じキーに加えて `response`（500文字に切り詰め） |
| `reaction:added` | ボットが見えるメッセージに絵文字リアクションが追加されたとき（現在はSlackアダプターのみ）。`reactions:read` スコープと `reaction_added` ボットイベントのサブスクリプションが必要で、ボットがチャンネルのメンバーである必要があります。 | `platform`、`reaction`、`user_id`、`item_user_id`、`item_type`、`channel_id`、`message_ts`、`team_id`、`event_ts`、`raw_event` |
| `reaction:removed` | ボットが見えるメッセージから絵文字リアクションが削除されたとき。`reaction_removed` ボットイベントのサブスクリプションが必要です。 | `reaction:added` と同じ形式 |
| `command:*` | 任意のスラッシュコマンドの実行時 | `platform`、`user_id`、`command`、`args` |

#### ワイルドカードマッチング

`command:*` に登録されたハンドラーは、任意の `command:` イベント（`command:model`、`command:reset` など）に対して発火します。1つのサブスクリプションですべてのスラッシュコマンドを監視できます。

:::tip スレッドへの返信
同じTelegramフォーラムトピックにフォローアップメッセージを投稿するハンドラーは、`chat_type == "forum"` かつ `thread_id` が空でない場合に `message_thread_id=int(thread_id)` を含める必要があります。
:::

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

新しいセッションが作られたときに外部サービスへPOSTします:

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

コミュニティで人気のパターンです: `~/.hermes/BOOT.md` にMarkdownのチェックリストを置き、Gatewayが起動するたびにエージェントにそれを1回実行させます。「起動のたびに夜間のcronの失敗を確認し、失敗があればDiscordで知らせる」や「deploy.logの直近24時間を要約してSlackの #ops に投稿する」といった用途に便利です。

このチュートリアルでは、ユーザー定義のフックとして自分で構築する方法を示します。Hermesには組み込みのBOOT.mdフックは同梱されていません — 望む動作を自分で正確に組み上げます。

#### 構築するもの

1. 自然言語の起動時指示を書いた `~/.hermes/BOOT.md` ファイル。
2. `gateway:startup` で発火し、Gatewayが解決したモデル/認証情報でワンショットのエージェントを起動して、BOOT.mdの指示を実行するGatewayフック。
3. 報告することがないときにエージェントがメッセージ送信をオプトアウトできる `[SILENT]` の規約。

#### ステップ1: チェックリストを書く

`~/.hermes/BOOT.md` を作成します。人間のアシスタントに指示を出すつもりで書いてください:

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, summarize them for Discord #ops (the hook delivers your final response to its configured target).
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same report.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

エージェントはこれをプロンプトの一部として受け取るので、平易な言葉で説明できることなら何でも機能します — ツール呼び出し、シェルコマンド、メッセージ送信、ファイルの要約など。

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
"""Run ~/.hermes/BOOT.md on every gateway startup."""

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
        "Execute each instruction. Put any user-facing summary in your "
        "final response — the hook delivers it to the configured channel "
        "(e.g. Discord or Slack); you do not send messages yourself.\n"
        "If nothing needs attention and there is nothing to report, reply "
        "with ONLY: [SILENT]"
    )


def _run_boot_agent(content: str) -> None:
    """Spawn a one-shot agent and execute the checklist.

    Uses the gateway's resolved model and runtime credentials so this works
    against custom endpoints, aggregators, and OAuth-based providers alike.
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
        response = (result.get("final_response", "") or "").strip()
        if response.upper() not in {"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"}:
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

    # Background thread so gateway startup isn't blocked on a full agent turn.
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
```

重要な2行:

- `_resolve_gateway_model()` は、Gatewayで現在設定されているモデルを読み取ります。
- `_resolve_runtime_agent_kwargs()` は、通常のGatewayのターンと同じ方法でプロバイダーの認証情報を解決します — APIキー、ベースURL、OAuthトークン、認証情報プールを含みます。

これらがないと、素の `AIAgent()` は組み込みのデフォルトにフォールバックし、デフォルト以外のエンドポイントに対しては401になります。

#### ステップ3: テストする

Gatewayを再起動します:

```bash
hermes gateway restart
```

ログを確認します:

```bash
hermes logs --follow --level INFO | grep boot-md
```

`Running BOOT.md (N chars)` に続いて、`boot-md completed: ...`（エージェントが行ったことの要約）か、エージェントが `[SILENT]` のような正確な沈黙トークンで返答した場合は `boot-md completed (nothing to report)` が表示されるはずです。

チェックリストを無効にするには `~/.hermes/BOOT.md` を削除します — フックは読み込まれたままですが、ファイルがない場合は黙ってスキップします。

#### パターンの拡張

- **スケジュールを考慮したチェックリスト:** BOOT.mdの指示の中で `datetime.now().weekday()` に応じて分岐させます（「月曜日なら週次のデプロイログも確認する」）。指示は自由形式のテキストなので、エージェントが推論できることなら何でも使えます。
- **複数のチェックリスト:** フックを別のファイル（`STARTUP.md`、`MORNING.md` など）に向け、それぞれに別のフックディレクトリを登録します。
- **エージェントを使わない版:** 完全なエージェントループが不要なら、`AIAgent` を完全に省略し、ハンドラーから `httpx` で固定の通知を直接ポストさせます。より安価で高速で、プロバイダーへの依存もありません。

#### なぜこれが組み込みでないのか

以前のバージョンのHermesはこれを組み込みフックとして同梱しており、Gatewayの起動のたびに素のデフォルトでエージェントを黙って起動していました。これはカスタムエンドポイントを使うユーザーを驚かせ、実行されていることを知らないユーザーにはこの機能が見えないものになっていました。文書化されたパターンとして — あなた自身がhooksディレクトリ内で構築するものとして — 維持することで、何をするのかを正確に把握でき、ファイルを書くことでオプトインすることになります。

### 仕組み

1. Gateway起動時に、`HookRegistry.discover_and_load()` が `~/.hermes/hooks/` をスキャンする
2. `HOOK.yaml` + `handler.py` を持つ各サブディレクトリがプロセス内でインポートされる — 有効化リストは参照されない（[信頼モデル](#gateway-hook-trust)を参照）
3. ハンドラーは宣言されたイベントに登録される
4. 各ライフサイクルポイントで、`hooks.emit()` が一致するすべてのハンドラーを発火する
5. いずれのハンドラーのエラーもキャッチされてログに記録される — 壊れたフックがエージェントをクラッシュさせることはない

:::info
Gatewayフックは **Gateway**（Telegram、Discord、Slack、WhatsApp、Teams）でのみ発火します。CLIはGatewayフックを読み込みません。どこでも動作するフックには、[プラグインフック](#plugin-hooks)を使ってください。
:::

## プラグインフック {#plugin-hooks}

[プラグイン](./plugins.md)は、**CLIとGatewayの両方**のセッションで発火するフックを登録できます。これらはプラグインの `register()` 関数内で `ctx.register_hook()` を使ってプログラム的に登録します。

プラグインのパッケージングと登録の詳細については、
[プラグインガイド](./plugins.md)を参照してください。

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
    # Kanban board lifecycle (each fires after its board write commits):
    ctx.register_hook("kanban_task_claimed", my_claim_callback)     # dispatcher process
    ctx.register_hook("kanban_task_completed", my_done_callback)    # worker process
    ctx.register_hook("kanban_task_blocked", my_blocked_callback)   # worker process
```

**すべてのフックに共通するルール:**

- コールバックは**キーワード引数**を受け取ります。前方互換性のため、常に `**kwargs` を受け付けてください。
- コールバックの例外はログに記録されてスキップされ、後続のコールバックは引き続き実行されます。毎回同じ形で失敗するコールバック（典型的には、フックが送らないフィールド名をシグネチャに書いたもの。例: `tool_name`/`args` ではなく `tool_data`）は WARNING で**1回だけ**報告され — メッセージにはフックが提供するフィールドが列挙されます — 同一の繰り返しは DEBUG に回されるため、宣言を誤ったプラグインがログをあふれさせることはありません。
- **タイムアウト制限付き**のフック（`post_tool_call` / `pre_llm_call` のようなホットパスのオブザーバーと、ポリシーフックの `pre_tool_call`）上のPythonプラグインのコールバックが `plugins.hook_callback_timeout`（デフォルト30秒、`0` で無効化、最大600）より長く**ブロック**した場合、ワーカーをjoinせずに放棄され、エージェントループは続行します。タイムアウトした、またはまだ実行中の `pre_tool_call` コールバックは**フェイルクローズ**（ツールをブロック）し、その他の制限付きフックはフェイルオープン（スキップ）します。呼び出し元スレッドの契約が文書化されているフック（`subagent_stop`）は、タイムアウト用ワーカーに移されることはありません。シェルフックはエントリごとの独自の `timeout` を保持します。
- 以下のカタログは記述的なものです: **オブザーバー**は戻り値を無視し、**変換**は最初の有効な文字列置換を受け入れ、**ディレクティブ/制御**フックは文書化された形式の戻り値を消費します。プラグインのミドルウェアは別のレジストリおよびサーフェスであり、別のフックカテゴリではありません。
- `turn_id`、`api_request_id`、`task_id`、`session_id`、`api_call_count` のような相関フィールドはフック固有であり、存在しない場合があります。IDは不透明な値として扱ってください。
- ツールフックは `conversation_id` も持ちます: 会話の最初のセッションIDで、ローテーションする圧縮が `session_id` を置き換えても変わらず、委譲されたサブエージェントは委譲元と同じ値を共有します。会話ごとの状態はこれをキーにしてください。
- 実行時に有効なイベント名は `hermes_cli.plugins.VALID_HOOKS` から決まります。`hermes hooks list` は設定済みのシェル/アウトバウンドフックを一覧表示するもので、利用可能なすべてのイベントではありません。`hermes hooks test <event>` は無効なイベントが指定された場合にのみ有効なイベントの一覧を報告します。

### キャッシュセーフなシステムプロンプトセクション

永続的で常に有効なガイダンスが必要なプラグインは、毎ターン `pre_llm_call` で
同じテキストを注入する代わりに、サイズ制限付きのシステムプロンプトセクションを
登録できます:

```python
def board_rules(session_info):
    return f"Apply the worker rules for profile {session_info['profile_name']}."

def register(ctx):
    ctx.register_system_prompt_section(
        "kanban-advanced.worker-rules",
        board_rules,                       # a string is also accepted
        position="after_memory",
        max_chars=4000,
    )
```

この契約は意図的に狭くなっています:

- IDはグローバルで安定した1〜128文字の小文字の識別子で、
  英字、数字、`.`、`_`、`-` のみを使用します。重複するIDは拒否されます。
- 配置アンカーは `after_memory` のみです。セクションはIDでソートされ、
  メモリ/プロファイルのコンテキストの後、セッションメタデータの前に描画されます。
  プラグインはコアのプロンプト内容を並べ替えたり置き換えたりできません。
- 呼び出し可能オブジェクトは、`session_id`、`conversation_id`、
  `model`、`provider`、`platform`、`profile_name`、`cwd`、`client`、`client_version`
  を含む読み取り専用のマッピングを受け取ります
  （`client`/`client_version`: 作成元クライアントが `session.create` で宣言した名前とバージョンで、
  何も送られなかった場合は `""`。自己申告のラベルなので、これに基づいて権限を与えてはいけません）。
  これは**新しいセッションに対して1回**実行されます。描画されたバイト列は圧縮時に凍結され、
  プロセスの再起動/再開後は、すでに永続化された完全なシステムプロンプトから復元されます。
  既存のセッションに対してプラグインの状態が再読み込みされることはありません。
- `max_chars` の上限は4,000文字です。すべてのプラグインセクションを合わせた量は、
  監査用の見出しを含めて8,000文字かつ32セクションが上限です。空、文字列以外、
  サイズ超過、合計予算超過、または例外を送出するセクションは警告とともにスキップされ、
  プロンプトの構築は続行されます。
- 受け入れられた各セクションは、プロンプト内で名前が示され、セッション開始時に
  プラグイン、位置、文字数とともにログに記録されます。

本当に動的なターンごとのコンテキストには `pre_llm_call` を使ってください。この契約には
意図的にプラグイン用の環境ヒントフックがありません: cwd、ブランチ、その他の
環境データの変化が、セッションのキャッシュされたプロンプトを黙って変えてはならないからです。
そのようなフックを追加するには、具体的な利用者と、同じく凍結/再開に安全なセマンティクスが
必要です。

### 同梱のプラグインフックカタログ {#shipped-plugin-hook-catalog}

以下のペイロードフィールドは、各呼び出し箇所が提供するイベント固有のフィールドそのものです。後方互換性のため、`PluginManager` はすべてのプラグインフックのコールバックに `telemetry_schema_version="hermes.observer.v1"` も追加します。このレガシーなエンベロープのマーカーは、すべてのフックのペイロードが1つの意味的スキーマを共有することを意味するものではありません。新しいバージョン付きの契約は、それぞれ具体的なイベントまたはケイパビリティのファミリーに属します。

| フック | カテゴリ | 正確なタイミングと戻り値の挙動 | 明示的なペイロードフィールド | プライバシー / 機密性 |
|---|---|---|---|---|
| [`pre_tool_call`](#pre_tool_call) | ディレクティブ/制御 | 実行前に1回。有効な `block` はどの `approve` よりも優先され（次に最初の有効な `approve`）、`modify` の戻り値はツール引数に浅くマージされます。 | `tool_name`、`args`、`task_id`、`session_id`、`conversation_id`、`tool_call_id`、`turn_id`、`api_request_id`、`middleware_trace` | 生の引数にはユーザーコンテンツ、パス、コマンド、シークレットが含まれる可能性があります。 |
| `post_tool_call` | オブザーバー | ブロック、エラー、成功のいずれの結果の後にも発火。戻り値は無視。 | `tool_name`、`args`、`result`、`task_id`、`session_id`、`conversation_id`、`tool_call_id`、`turn_id`、`api_request_id`、`duration_ms`、`status`、`error_type`、`error_message`、`middleware_trace` | 結果/エラーテキストには任意のツールやユーザーのコンテンツ、シークレットが含まれる可能性があります。 |
| `transform_tool_result` | 変換 | `post_tool_call` の後、会話への追加の前。最初の文字列が結果を置き換えます。 | `tool_name`、`args`、`result`、`task_id`、`session_id`、`conversation_id`、`tool_call_id`、`turn_id`、`api_request_id`、`duration_ms`、`status`、`error_type`、`error_message` | モデルに渡される完全な結果と引数が公開されます。 |
| `transform_terminal_output` | 変換 | サイズ制限付きのフォアグラウンドプロセスのキャプチャ後、最終的な出力制限の前。最初の文字列が出力を置き換えます。 | `command`、`output`、`returncode`、`task_id`、`env_type` | コマンド/出力に認証情報が含まれる可能性があります。 |
| `pre_transcription` | 変換 | STTディスパッチャーがプロバイダーを解決した後、いずれかのバックエンド（組み込み、コマンド型、プラグイン登録）を呼び出す前に発火。dictの結果は登録順に適用され、フィールドごとに最後に書いたものが勝ちます（`prompt`、`language`、`model`。`file_path` は読み取り専用）。 | `file_path`、`provider`、`model`、`language`、`prompt`、`source` | 最終的なプロンプトは音声とともに設定されたSTTプロバイダーにアップロードされます — フックの戻り値にシークレットを含めないでください。 |
| `pre_llm_call` | ディレクティブ/制御 | ループ前にターンごとに1回。有効な文字列/`{"context": ...}` の戻り値はすべて結合されてユーザーメッセージに注入されます。最初の有効な `{"action": "block", "message": ...}` は推論リクエストの前にターンを終了させます。 | `session_id`、`task_id`、`turn_id`、`user_message`、`conversation_history`、`is_first_turn`、`model`、`platform`、`parent_session_id`、`sender_id` | 完全なユーザーメッセージと会話履歴。 |
| `post_llm_call` | オブザーバー | 成功し、中断されなかったターンの確定時。戻り値は無視。 | `session_id`、`task_id`、`turn_id`、`user_message`、`assistant_response`、`conversation_history`、`model`、`platform`、`side_question_token`、`review_signals` | 完全なプロンプト、応答、履歴。 |
| `transform_llm_output` | 変換 | `post_llm_call` と最終配信の前。最初の空でない文字列が応答を置き換えます。 | `response_text`、`session_id`、`model`、`platform` | アシスタントの最終テキスト全体。 |
| `pre_verify` | ディレクティブ/制御 | 回数制限付きの編集コード検証ゲートで発火。最初の有効な continue/block-stop ディレクティブがターンを継続させます。 | `session_id`、`platform`、`model`、`coding`、`attempt`、`final_response`、`changed_paths` | 応答の下書きと変更されたパス。 |
| `pre_api_request` | オブザーバー | プロバイダーへの試行ごとに、リクエストの直前。戻り値は無視。 | `task_id`、`turn_id`、`api_request_id`、`session_id`、`user_message`、`conversation_history`、`platform`、`model`、`provider`、`base_url`、`api_mode`、`api_call_count`、`retry_count`、`request_messages`、`message_count`、`tool_count`、`approx_input_tokens`、`request_char_count`、`max_tokens`、`started_at`、`middleware_trace`、`request` | 機密性が高い: レガシーの `user_message`、`conversation_history`、`request_messages` は意図的に生のままです。サニタイズ済みの `request` を優先してください。 |
| `post_api_request` | オブザーバー | 正規化されたプロバイダーの成功後。戻り値は無視。 | `task_id`、`turn_id`、`api_request_id`、`session_id`、`platform`、`model`、`provider`、`base_url`、`api_mode`、`api_call_count`、`api_duration`、`started_at`、`ended_at`、`finish_reason`、`message_count`、`response_model`、`response`、`usage`、`assistant_message`、`assistant_content_chars`、`assistant_tool_call_count` | サニタイズ済みの `response` が利用できますが、生の正規化済み `assistant_message` にはモデル/ユーザーのコンテンツが含まれる可能性があります。`usage` は課金計測データです。 |
| `api_request_error` | オブザーバー | プロバイダーへの試行が失敗するたび。戻り値は無視。 | `task_id`、`turn_id`、`api_request_id`、`session_id`、`platform`、`model`、`provider`、`base_url`、`api_mode`、`api_call_count`、`api_duration`、`started_at`、`ended_at`、`status_code`、`retry_count`、`max_retries`、`retryable`、`reason`、`error`、`request` | エラーテキストにプロバイダー/ユーザーのデータが含まれる可能性があります。`request` はサニタイズ済みであることを意図しています。 |
| `pre_auxiliary_call` | オブザーバー | 補助LLM呼び出し（タイトル付け、圧縮、MoA、ビジョン、承認など）のプロバイダーへの試行ごとに、リクエストの直前。戻り値は無視。 | `aux_task`、`task_id`、`turn_id`、`session_id`、`platform`（親ターンのもの。ターン外では空）、`api_request_id`、`api_call_count`、`retry_count`、`streaming`、`model`、`provider`、`base_url`、`api_mode`、`request_messages`、`system_prompt`、`message_count`、`tool_count`、`approx_input_tokens`、`request_char_count`、`max_tokens`、`started_at`、`request` | `request_messages` は生のままです（圧縮ではトランスクリプト全体が見えます）。サニタイズ済みの `request` を優先してください。 |
| `post_auxiliary_call` | オブザーバー | 同じ試行が戻るか例外を送出した後。戻り値は無視。 | `pre_auxiliary_call` の識別フィールドに加えて `api_duration`、`ended_at`、`finish_reason`、`response_model`、`usage`、`response`、`assistant_content_chars`、`assistant_tool_call_count`、`error`、`error_type`（成功時は `None`。エラー時と `streaming=True` のときは `usage`/`response` が `None`） | サニタイズ済みの `response`。`usage` は課金計測データ。`error` にはプロバイダーのテキストが含まれる可能性があります。 |
| `on_stream_start` | オブザーバー | ストリーミングのLLM応答が始まったときにディスパッチ。ホストが所有するサイズ制限付きキューを介し、コールバックごとに1つのワーカーでトークンパスの外で配信されます。戻り値は無視。 | `turn_id`、`iteration`、`session_id`、`model`、`provider`、`surface` | 識別子とルーティングのメタデータのみ。 |
| `on_stream_delta` | オブザーバー | 正規化されたストリーミングのテキスト差分ごとに、サイズ制限付きのオブザーバーキューを介してディスパッチ。停止したコールバックは自分自身の最も古いイベントのみを破棄します。戻り値は無視。 | `delta`、`kind`（`text` または `reasoning`）、`turn_id`、`iteration`、`session_id`、`model`、`provider`、`surface` | 差分テキストはモデルの生の出力です。推論の差分には `plugins.stream_reasoning_deltas` のオプトインが必要です。 |
| `on_stream_end` | オブザーバー | ストリーミング応答が終了またはエラーになったとき、ストリームが閉じた後にディスパッチ。戻り値は無視。 | `final_text`、`finished`、`error`、`turn_id`、`iteration`、`session_id`、`model`、`provider`、`surface` | 組み立てられた応答テキスト全体。エラーテキストにはプロバイダーのデータが含まれる可能性があります。 |
| `on_interim_message` | オブザーバー | 最終回答の前にループ途中のアシスタントメッセージが表出したときにディスパッチ（ストリーミング/非ストリーミングの両方）。戻り値は無視。 | `text`、`already_streamed`、`turn_id`、`iteration`、`session_id`、`model`、`provider`、`surface` | 途中のアシスタントテキスト全体。 |
| `transform_api_error_classification` | 変換 | プロバイダーへの試行が失敗するたび、組み込み分類器の先頭で発火。すべてのコールバックが実行された後、有効な `reason` を持つ最初のdictが勝ち（全実行後に最初を選択）、スキップされた有効な結果は実行時警告としてログに記録されます。Pythonプラグインのみ。 | `provider`、`model`、`status_code`、`error_type`、`error_code`、`error_message`、`error_body`、`error`、`approx_tokens`、`context_length`、`num_messages` | `error_message` と `error_body` には生のプロバイダー/ユーザーのデータが含まれる可能性があります。 |
| `on_session_start` | オブザーバー | 新しいセッションの最初のターン。戻り値は無視。 | `session_id`、`model`、`platform` | 識別子とルーティングのメタデータのみ。 |
| `on_session_open` | オブザーバー | クライアントがローカルセッション（CLI、TUI、Desktop、ACP、kanban/A2Aワーカー）を作成または開くたびに1回、Gateway内でそのセッションのプロファイルのもとで、最初のターンの前に発火。メッセージングのチャットでは発火しません。タイムアウト制限付き。戻り値は無視。 | `session_id`、`platform`、`reason`（`create` または `resume`） | セッション識別子のみ。 |
| `on_session_end` | オブザーバー | 正規には各ターンの確定時。CLI/TUIの終了時には、縮小されたレガシーの形式が追加であります。戻り値は無視。 | 正規: `session_id`、`task_id`、`turn_id`、`completed`、`failed`、`interrupted`、`turn_exit_reason`、`model`、`platform`。終了パスでは `reason`/`api_request_id` が追加されたり、フィールドが省略されたりすることがあります。 | ID、モデル/プラットフォーム、結果。正規のペイロードにメッセージ本文はありません。 |
| `on_session_finalize` | オブザーバー | `finalize_session` を通じたCLI/TUI/Gatewayの終了処理。Gatewayのシャットダウンはリセットなしで確定することがあります。戻り値は無視。 | サーフェスに依存する `session_id`、`platform`、任意で `reason`、`old_session_id`、`new_session_id` | セッションとルーティングの識別子。 |
| `on_session_reset` | オブザーバー | CLI/TUIのセッション境界、およびGatewayで置き換えのセッションが存在した後。戻り値は無視。 | CLI: `session_id`、`platform`、`reason`。TUI: `session_id`、`platform`。Gateway: それらに加えて `reason`、`old_session_id`、`new_session_id` | セッションとルーティングの識別子。 |
| `on_session_delete` | オブザーバー | セッションの削除または意図的なアーカイブがコミットされた後（CLI、TUI、Desktop/ダッシュボード、Gateway API、スイープ、prune）。戻り値は無視。 | `session_id`、`reason`、`profile`、任意の `deleted_ids`（複数行が削除された場合はその全集合）、`sessions_dir` | セッション識別子のみ。メッセージ内容はありません。 |
| `post_compaction` | オブザーバー | コミットされた圧縮ごとに1回、すべてのパスで発火: エージェントループ内の自動および手動の圧縮（CLI、TUI、Desktop、Gatewayの衛生処理、オーバーフローからの回復）、Gatewayの `/compress`（`session.mutate compress`）、Codex app-serverの圧縮、マイクロ圧縮、プロアクティブなツール結果の刈り込み。何も変更しなかった試行（中止、ブロック、拒否）では発火しません。タイムアウト制限付き。戻り値は無視。 | `session_id`、`old_session_id`、`kind`、`reason`、`before_tokens`、`after_tokens`、`messages_removed` | 識別子と件数のみ。 |
| `post_curator_run` | オブザーバー | キュレーターのパスが実行ごとのレポート（`logs/curator/<run>/run.json` + `REPORT.md`）を書き出した後、そのパスを実行したプロセス内で発火。戻り値は無視。 | `run_dir`、`run_json_path`、`report_md_path`、`dry_run` | ローカルのファイルシステムパス。レポートにはスキル名が含まれます。 |
| `agent_loop_stopped` | オブザーバー | 実際に実行中のエージェントが中断された直後 — Gatewayの `_interrupt_and_clear_session` またはTUI/デスクトップの `session.interrupt`。戻り値は無視。 | `session_key`、`platform`、`reason`、`invalidation_reason` | セッション/ルーティングの識別子と中断理由。メッセージ本文はありません。 |
| `on_skill_lifecycle` | オブザーバー | 権威あるスキル使用状態の変更の後。戻り値は無視。 | `action`、`skill_name`、`provenance`、`task_id`、`session_id`、`use_count`、`reused`、`reuse_after_patch` | ローカルのスキル名と出所が公開されます。 |
| `subagent_start` | オブザーバー | 子が構築され、実行される直前。戻り値は無視。 | `parent_session_id`、`parent_turn_id`、`parent_subagent_id`、`child_session_id`、`child_subagent_id`、`child_role`、`child_goal` | 子のゴールにはユーザー/プロジェクトのコンテンツが含まれる可能性があります。 |
| `subagent_stop` | オブザーバー | 子の終了時。戻り値は無視。 | `parent_session_id`、`parent_turn_id`、`child_session_id`、`child_role`、`child_summary`、`child_status`、`tool_call_history`、`duration_ms` | 要約とリダクト済みのツール履歴メタデータからプロジェクト構造がわかる可能性があります。 |
| `pre_gateway_dispatch` | ディレクティブ/制御 | 内部以外の受信メッセージごとに1回、入口で発火: プラットフォームアダプターのメッセージは認証/ペアリング/ディスパッチの前、Webhookとリレーの配信はそれぞれの認可後・受け入れのコミット前。ローカルのCLI/TUI/DesktopやAPIのターン、コミット済み配信の再試行、キューに入った受け入れの実行時には発火しません。最初の有効な `skip`、`rewrite`、`allow` がフローを制御します。 | `event`、`gateway`、`session_store` | 極めて特権的なプロセス内オブジェクトで、受信したユーザー/ルーティングのデータとホストのハンドルが公開されます。 |
| `post_gateway_admission` | ディレクティブ/制御（フェイルオープン） | 認証、一時停止/ドレイン、保留中の返信、実行中セッション、スラッシュコマンドの各レーンを通過して受け入れられた内部以外のメッセージに対し、確保されたセッションスロット内、ルーティング先プロファイルのスコープ内で発火。最初の `handled` 結果がエージェントのターンをスキップし、それ以外（例外送出やタイムアウトを含む）はターンを実行します。 | `session_key`、`platform`、`source`（dictのスナップショット）、`message_id`、`text` | 受信テキストは信頼できないユーザーデータです。ランナーやセッションストアのハンドルは渡されません。 |
| `gateway_platform_event` | オブザーバー | Gatewayのプロファイル単位の認可が成功した後、サポートされるプラットフォームネイティブのイベントがGatewayの境界で正規化されたとき（Telegram: リアクション、メッセージ編集。Discord: メッセージの編集/削除、スレッドの作成/名前変更）。戻り値は無視。 | `platform`、`event_type`、`payload`（イベント種別ごとのdict — 下記のイベントごとの契約を参照） | 正規化されたプレーンなdictのエンベロープのみ。生のSDKオブジェクト、アダプターのハンドル、ボットクライアントが公開されることはありません。 |
| `pre_command` | オブザーバー | 認識されたスラッシュコマンドがディスパッチされる直前、ハンドラーの実行前に、CLIとGatewayのコールドパスのディスパッチで発火。v1では戻り値は無視されます（ディレクティブ形式のdictはdebugでログに記録）。クラシックCLI、TUI、Desktopが `slash.exec` または `command.dispatch` を通じてGatewayに送るコマンドは、Gateway内で1回発火し、`surface` と `platform` はこれまでどおりともに `"cli"`（InkとDesktopはCLIのスラッシュワーカー経由で発火していた）、`session_key` はセッションIDになります。クラシックCLIが自分で実行する認識済みのコマンドは、実行後にGatewayへ報告され（`command.observe`）、次のTUIコマンドも同様です: `/help`、`/history`、`/clear`、`/new`、`/tools`（`enable`/`disable` を除く）、`/usage`、`/reasoning`（Gatewayが受け入れた場合のレベル指定）。Gatewayは返信後にそこで発火するため、どちらのクライアントもプラグインコードを読み込みません。どちらかのクライアントが拒否したコマンドは何も発火しません。これは統合Gateway以前のクラシックCLIからの意図的な変更です。以前はディスパッチ直前に認識済みのすべてのコマンドで発火していましたが、現在はコマンドが実行された場合にのみ発火するため、拒否されたコマンドやクライアントにハンドラーがないコマンド（Gatewayセッションでの `/handoff`、`/start`）は何も発火しません。`/reasoning high` のようにGatewayが適用するTUIの変更は、Gatewayが変更を受け入れた時点、つまり変更が反映された後に発火します。クライアントより古いGatewayでは、これらのコマンドに対してフックは発火しません（クラシックCLIは1回限りの通知を表示します）。既知の欠落: Desktop独自のピッカー、アクション、専用RPC（`slash.exec` を使わずに応答するコマンド）はまだ何も発火せず、TUIが自分で、または専用のGateway呼び出しを通じて応答するその他のコマンドも同様です。例: `/title`、`/undo`、`/retry`、`/model`、`/compress`、`/branch`、`/save`、`/copy`、`/queue`、`/steer`、`/bg`、`/btw`、`/skin`、`/yolo`、`/busy`、`/fast`、`/verbose`、`/rollback`、`/reload-mcp`、`/quit`、`/reasoning --global`。ローカルセッションのターン中は、`/stop`、`/approve`、`/deny` を除き、実行されるすべてのコマンドで発火します。拒否されたコマンド（ビジー中に拒否された、またはGatewayで利用不可と示された）は何も発火しません。各コールバックはタイムアウト制限付き（`plugins.hook_callback_timeout`）で、各配信は専用のスレッドで待機し、他のセッションやイベントループのexecutorと共有されるワーカーは使いません。1つのセッションのコマンドは順序どおりに届き、そのセッションでまだ実行中の（ハングした）コールバックは、順序を乱して実行するのではなく、後続のコマンドではスキップされ、他のセッションを止めることはありません。Gatewayの実行中エージェントへの割り込みコマンド（`/stop`、アクティブな実行中の `/approve`）は意図的に除外されています — コントロールプレーンの脱出口はプラグインの手の届かないところに置かなければなりません。 | `surface`（`"cli"` \| `"gateway"`）、`command`（正規名）、`alias_used`、`args_raw`、`session_key`、`platform` | `args_raw` にはコマンドの後に入力されたユーザーコンテンツやシークレットが含まれる可能性があります。 |
| `pre_approval_request` | オブザーバー | プロンプトによる承認またはスマート承認の前。戻り値は無視。 | `command`、`description`、`pattern_key`、`pattern_keys`、`session_key`、`surface`、`turn_id`、`tool_call_id` | コマンドにシークレットが含まれる可能性があります。スマートオブザーバーの準備では強制的にリダクトされますが、すべてのサーフェスで同一のリダクトが行われるわけではありません。 |
| `post_approval_response` | オブザーバー | 決定、タイムアウト、またはGatewayの通知失敗の後。戻り値は無視。 | `command`、`description`、`pattern_key`、`pattern_keys`、`session_key`、`surface`、`turn_id`、`tool_call_id`、`choice`。スマートパスでは `decided_by` が追加される場合があります | コマンドと同じ機密性に加えて決定のメタデータ。 |
| `on_human_input_request` | オブザーバー | エージェントが人の入力待ちでブロックしようとするとき — sudoのパスワード入力、`clarify` の質問、危険なコマンドの承認 — CLI、Ink TUI/Desktop、ACP、Gatewayプラットフォームで発火。戻り値は無視。 | `kind`（`"sudo"` \| `"clarify"` \| `"approval"`）、`request_id`、`session_id`、`session_key`、`platform`、`prompt` | `prompt` は常に強制的にリダクトされます。入力されたパスワードや回答が渡されることはありません。 |
| `on_human_input_resolved` | オブザーバー | `on_human_input_request` ごとにちょうど1回、待機が終わったとき（回答、スキップ、タイムアウト、取り下げ、失敗）。戻り値は無視。 | リクエストと同じものに加えて `outcome` | 結果のみ — 回答そのものは渡されません。 |
| `on_room_member_activity` | オブザーバー | Bot ModeのGateway上でホストされたGroup Chatのメンバーのターンが実行されている間、メンバーセッションが発行するランタイムイベント（ツールの開始/完了、承認リクエスト、メッセージ/推論の差分、エラー）ごとに1回発火。利用者ごとにトークンパスの外でキューイングされます。戻り値は無視。 | `room_id`、`thread_id`、`member_id`、`turn_id`、`task_id`、`execution_generation`、`kind`、`seq`、`payload` | `payload` はクライアントに安全なセッションイベント本文です: ツールの引数と結果、リダクト済みの承認コマンド、ストリーミングされたメンバーのテキスト。 |
| `kanban_task_claimed` | オブザーバー | クレームのコミット後、ディスパッチャープロセス内でワーカー起動前。戻り値は無視。 | `task_id`、`profile_name`、`board`、`assignee`、`run_id` | ボード/タスク/プロファイル/担当者の識別子。 |
| `kanban_task_completed` | オブザーバー | 完了とクリーンアップの後、通常はワーカープロセス内。戻り値は無視。 | `task_id`、`profile_name`、`board`、`assignee`、`run_id`、`summary` | 要約にはプロジェクト/ユーザーのコンテンツが含まれる可能性があります。 |
| `kanban_task_blocked` | オブザーバー | blockedへの遷移がコミットされた後（依存待ちを含む）。戻り値は無視。 | `task_id`、`profile_name`、`board`、`assignee`、`run_id`、`reason` | 理由にはプロジェクト/ユーザーのコンテンツが含まれる可能性があります。 |
| `on_kanban_worker_spawned` | オブザーバー | `spawn_fn` が戻り、ワーカーのPIDが永続化された後。ディスパッチロック内で実行されるので、コールバックは高速に保ってください。戻り値は無視。 | `task_id`、`profile_name`、`board`、`assignee`、`run_id`、`worker_pid`、`workspace_path` | `workspace_path` はファイルシステムのパスで、プロジェクトの構成やユーザー名がわかる可能性があります。 |
| `on_kanban_worker_exited` | オブザーバー | ティックから導出: `detect_crashed_workers` がPIDの死んだタスクを回収し、その回収がコミットされた後。戻り値は無視。 | `task_id`、`profile_name`、`board`、`assignee`、`run_id`、`worker_pid`、`exit_kind`、`exit_code`、`outcome`、`retry_status` | 識別子と終了のメタデータのみ。 |
| `on_kanban_worker_stale_claim` | オブザーバー | TTL切れのクレームが回収された後。PIDが生きている場合の延長では発火しません。戻り値は無視。 | `task_id`、`profile_name`、`board`、`assignee`、`run_id`、`worker_pid`、`heartbeat_stale`、`retry_status` | 識別子とクレームのメタデータのみ。 |
| `on_kanban_task_updated` | オブザーバー | クレーム/完了/ブロックのライフサイクル外で、タスクのフィールドへの書き込みがコミットされた後（割り当て、オーバーライド、ダッシュボードのエディター）。戻り値は無視。 | `task_id`、`profile_name`、`board`、`assignee`、`run_id`、`changed_fields` | `changed_fields` はフィールド名のみを持ち、値は持ちません。ボードDB内の該当するタイトル/本文の値にはユーザー/プロジェクトのコンテンツが含まれる可能性があります。 |
| `on_kanban_dispatch_tick` | オブザーバー | ディスパッチャーのティックごとに1回、ディスパッチロックが解放された厳密に後。アイドルや競合したティックでも発火します。戻り値は無視。 | `board`、`profile_name`、`dry_run`、`outcome`、`result` | `result` はそのティックの `DispatchResult` で、タスクID、担当者、ワークスペースのパスを含みます。 |

---

### ストリーミング出力フック

これらのオブザーバー専用フックを使うと、プラグインは応答を変更することなく、テレメトリ、ライブダッシュボード、TTSパイプラインのためにストリーミングのLLM出力を消費できます。これらはホストが所有するサイズ制限付きのキューを通じて、登録されたコールバックごとに1つのバックグラウンドワーカーで配信されるため、プラグインのコールバックがトークンパス上でインライン実行されることはありません。あるコールバックが停止した場合、満杯になって最も古い保留中のオブザーバーイベントを破棄するのはそのコールバックのキューだけで、他のオブザーバーは独立してイベントを受け取り続けます。

他のプラグインフックと同様に登録します:

```python
def on_delta(delta, kind, model, provider, **kwargs):
    if kind == "text":
        print(delta, end="", flush=True)

def register(ctx):
    ctx.register_hook("on_stream_delta", on_delta)
```

4つのフックすべてに共通するフィールド:

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `turn_id` | `str` | 不透明なターン識別子（利用可能な場合） |
| `iteration` | `int` | 現在のAPI呼び出し/ツールループのイテレーション |
| `session_id` | `str` | 現在のHermesセッションID |
| `model` | `str` | アクティブなモデル識別子 |
| `provider` | `str` | アクティブなプロバイダー名 |
| `surface` | `str` | 呼び出し元のサーフェス。例: `cli`、`discord`、`telegram` |

追加フィールド:

| フック | 追加フィールド |
|------|--------------|
| `on_stream_start` | なし |
| `on_stream_delta` | `delta: str`、`kind: "text" | "reasoning"` |
| `on_stream_end` | `final_text: str`、`finished: bool`、`error: str | None` |
| `on_interim_message` | `text: str`、`already_streamed: bool` |

`on_interim_message` は非ストリーミングの応答の後にも発火することがあるため、このフックだけを登録してもプロバイダー呼び出しがストリーミング転送に強制されることはありません。

推論の差分はデフォルトではプラグインに公開されません。明示的にオプトインしてください:

```yaml
plugins:
  stream_reasoning_deltas: true
```

戻り値は無視されます。ストリームを高速に保つため、コールバックは自分の処理をキューに入れてすぐに戻るべきです。例外はログに記録され、ストリームを止めることはありません。

---

### `pre_tool_call`

すべてのツール実行の**直前**に発火します — 組み込みツールもプラグインツールも同様です。

**コールバックのシグネチャ:**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `tool_name` | `str` | 実行されようとしているツールの名前（例: `"terminal"`、`"web_search"`、`"read_file"`） |
| `args` | `dict` | モデルがツールに渡した引数 |
| `task_id` | `str` | セッション/タスクの識別子。設定されていない場合は空文字列。 |

**発火場所:** `model_tools.py` の `handle_function_call()` 内で、ツールのハンドラーが実行される前。ツール呼び出しごとに1回発火します — モデルが3つのツールを並列に呼び出すと、3回発火します。

**戻り値 — ブロックまたは承認を要求:**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
# or
return {"action": "approve", "message": "Why approval is required", "rule_key": "optional:scope"}
```

優先順位は登録順に関係なく `block` > `approve` > ディレクティブなし です: どのプラグインの有効な `block` も、それより前のプラグインの `approve` に勝ち、`approve` ディレクティブの中では最初の有効なものが勝ちます（Pythonプラグインが先に登録され、次にシェルフック）。`block` には空でない `message` が必要で、そのテキストをモデルに返すエラーとしてツールを短絡させます。`approve` はその呼び出しを既存の人間による承認ゲートにエスカレートします。`message` と `rule_key` は任意で、拒否、タイムアウト、ゲートのエラーはフェイルクローズします。その他の戻り値は無視されるため、既存のオブザーバー専用のコールバックは変更なしで動作し続けます。

**戻り値 — ツールの引数を書き換える:**

```python
return {"action": "modify", "args": {"new_string": "fixed content"}}
```

返された `args` 辞書は、ツールが実行される前に元のツール引数に浅くマージされます。複数の `modify` フックは累積します — 各フックのキーは元の引数から構築された1つの累積dictにマージされるため、フックAが `path` を変更し、フックBが `content` を変更した場合、両方が残ります。2つのフックが同じキーを変更した場合は、後のフックが勝ちます。

シェルフックはClaude Code互換の形式も受け付けます:

```json
{"decision": "modify", "tool_input": {"new_string": "fixed content"}}
```

どちらの形式も内部的には `{"action": "modify", "args": {...}}` に正規化されます。

`pre_tool_call` コールバックが `plugins.hook_callback_timeout` を超えた場合（または以前にタイムアウトした発火からまだ実行中の場合）、Hermesは**フェイルクローズ**します: ポリシーの決定なしに進むのではなく、タイムアウトメッセージとともにツールがブロックされます。例外を送出するコールバックにも同じことが当てはまり、ブロックメッセージにはコールバックとエラーが示されます。ハングしたコールバックは60秒の抑制期間スキップされ、その後は新しいツール呼び出しで再び実行されます（コールバックごとに最大3つの放棄されたワーカーまで。そのため、永続的にハングしたプラグインは、再起動までエージェントを黙って行き詰まらせるのではなく、その名前を示す警告とともにツール呼び出しをブロックします）。

**ユースケース:** ログ記録、監査証跡、ツール呼び出しカウンター、危険な操作のブロック、レート制限、ユーザーごとのポリシー適用、引数のサニタイズ、パスの書き換え、デフォルトパラメーターの注入。

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

**例 — 危険なツールで警告する:**

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

すべてのツール実行が戻った**直後**に発火します。

**コールバックのシグネチャ:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `tool_name` | `str` | 実行されたばかりのツールの名前 |
| `args` | `dict` | モデルがツールに渡した引数 |
| `result` | `str` | ツールの戻り値（常にJSON文字列） |
| `task_id` | `str` | セッション/タスクの識別子。設定されていない場合は空文字列。 |
| `duration_ms` | `int` | ツールのディスパッチにかかった時間（ミリ秒。`registry.dispatch()` の前後で `time.monotonic()` を使って計測）。 |

**発火場所:** `model_tools.py` の `handle_function_call()` 内で、ツールのハンドラーが戻った後。ツール呼び出しごとに1回発火します。ツールが未処理の例外を送出した場合は発火**しません**（エラーはキャッチされて代わりにエラーのJSON文字列として返され、`post_tool_call` はそのエラー文字列を `result` として発火します）。

**戻り値:** 無視されます。

**ユースケース:** ツール結果のログ記録、メトリクス収集、ツールの成功/失敗率の追跡、レイテンシーダッシュボード、ツールごとの予算アラート、特定のツールの完了時の通知送信。

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

ツール呼び出しループが始まる前に、**ターンごとに1回**発火します。有効なコールバックの戻り値はすべてプラグインの順に集約され、現在のターンのユーザーメッセージに注入されます。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 現在のセッションの一意な識別子 |
| `user_message` | `str \| list` | このターンのユーザーの元のメッセージ（スキル注入前）。マルチモーダルのターン（画像やその他の添付）では、送信されたとおりのコンテンツパーツのリスト |
| `conversation_history` | `list` | 完全なメッセージリストのコピー（OpenAI形式: `[{"role": "user", "content": "..."}]`） |
| `is_first_turn` | `bool` | 新しいセッションの最初のターンなら `True`、以降のターンでは `False` |
| `model` | `str` | モデル識別子（例: `"anthropic/claude-sonnet-4.6"`） |
| `platform` | `str` | セッションが実行されている場所: `"cli"`、`"telegram"`、`"discord"` など |

**発火場所:** `agent/turn_context.py`（`agent/conversation_loop.py` の `run_conversation()` のためのターン準備）内で、コンテキスト圧縮の後、メインの `while` ループの前。`run_conversation()` の呼び出しごと（つまりユーザーのターンごと）に1回発火し、ツールループ内のAPI呼び出しごとではありません。

**戻り値:** コールバックが `"context"` キーを持つdict、または空でないプレーンな文字列を返した場合、そのテキストが現在のターンのユーザーメッセージに追加されます。注入しない場合は `None` を返します。

`{"action": "block", "message": "..."}` を返すと、**推論リクエストの前に**ターンを拒否します: メッセージはモデルの返信の代わりにユーザーに表示され、アシスタントの返信として保存されます（そのためトランスクリプトは通常の `user → assistant` のペアを保ち、`--resume` も機能します）。ターンはモデル呼び出しを一切行わず、結果には `turn_exit_reason: "blocked_by_plugin_pre_llm_call"` が含まれます。最初のブロックが勝ちます。メッセージのないブロックは無視されます。

ブロックされたプロンプトは `[Message withheld: a pre_llm_call policy hook blocked it.]` として保存されます。このプレースホルダーが、以降のターンで再生され、圧縮の要約、`session_search`、外部メモリプロバイダー、バックグラウンドのメモリレビューが読み取るものになるため、プロンプトがこのターンでも以降のターンでもモデルに届くことはありません。ユーザーが入力した内容は表示用メタデータとしてのみ保持されるため、CLIの `--resume` の振り返り、TUI、デスクトップアプリ、ダッシュボードでは引き続き表示されます。フックはターン開始時の圧縮の前に実行され、ブロックされたターンでは圧縮、タイトル付け、外部メモリのプリフェッチ/同期がスキップされます。

ブロックはコールバックがそれを返した場合にのみ適用されます。例外を送出したり、`plugins.hook_callback_timeout`（デフォルトで30秒）を超えて実行されたりしたコールバックはログに記録されてスキップされ、ターンは続行されます。したがってポリシーフックは自分でエラーをキャッチし、判断できない場合はブロックを返すべきです。モデルのメタデータの参照（新しいセッションでの `/models` のコンテキスト長のプローブなど）はフックの前に実行されることがありますが、メッセージ内容は含みません。

```python
# Inject context
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# Plain string (equivalent)
return "Recalled memories:\n- User likes Python"

# No injection
return None
```

**コンテキストが注入される場所:** 常に**ユーザーメッセージ**であり、システムプロンプトではありません。これによりプロンプトキャッシュが保たれます — システムプロンプトはターン間で同一のままなので、キャッシュされたトークンが再利用されます。システムプロンプトはHermesの領域です（モデルのガイダンス、ツールの強制、パーソナリティ、スキル）。プラグインはユーザーの入力とともにコンテキストを提供します。

クリーンなユーザーメッセージの `content` は変更されません。再生とプロンプトキャッシュの安定性のために、Hermesはプラグインが注入したコンテキストを含む、APIに送られた正確なメッセージを行の `api_content` サイドカーに永続化することがあります。

**マルチモーダルのターン**（ユーザーメッセージがコンテンツパーツのリスト — 画像の添付、またはパーツとして送られたテキスト）では文字列のサイドカーはありません: 結合されたコンテキストは、最初のリクエストの前に、そのターンのコンテンツに1つの追加の `{"type": "text"}` パーツとして追加され、そのパーツはターンとともに永続化されるため、再開されたセッション、圧縮、再生のすべてがモデルが見たのと同じメッセージを見ます。それ以前のメッセージとシステムプロンプトが変更されることはありません。

**複数のプラグイン**がコンテキストを返した場合、それらの出力はプラグインの検出順（ディレクトリ名のアルファベット順）に二重改行で結合されます。

**ユースケース:** メモリの想起、RAGのコンテキスト注入、ガードレール、ターンごとの分析。

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
| `session_id` | `str` | 現在のセッションの一意な識別子 |
| `user_message` | `str` | このターンのユーザーの元のメッセージ |
| `assistant_response` | `str` | このターンのエージェントの最終テキスト応答 |
| `conversation_history` | `list` | ターン完了後の完全なメッセージリストのコピー |
| `model` | `str` | モデル識別子 |
| `platform` | `str` | セッションが実行されている場所 |
| `side_question_token` | `str \| None` | [`ctx.side_question`](/docs/developer-guide/plugin-llm-access#asking-the-sessions-own-agent-ctxside_question) または [`ctx.fork_session`](/docs/developer-guide/plugin-llm-access#forking-the-sessions-own-agent-ctxfork_session) 用の使い捨てトークン（ターンごとに1リクエスト）で、セッションのプロファイルのすべてのプラグインで共有されます。読み込まれたプラグインが `llm.side_question` または `llm.session_fork` のケイパビリティを持たない限り `None` で、cronの実行とサブエージェントのターンでは常に `None` |
| `review_signals` | `dict`（読み取り専用） | このターンが終わった時点の自己学習のケイデンス。[下記](#review_signals)を参照 |

**発火場所:** `agent/turn_finalizer.py`（`agent/conversation_loop.py` の `run_conversation()` から呼ばれる `finalize_turn()`）内で、ツールループが最終応答とともに終了した後。`if final_response and not interrupted` でガードされているため、ユーザーがターンの途中で中断した場合や、エージェントが応答を生成せずにイテレーション上限に達した場合は発火**しません**。

**戻り値:** 無視されます。

**ユースケース:** 会話データの外部メモリシステムへの同期、応答品質メトリクスの計算、ターン要約のログ記録、フォローアップアクションのトリガー。

<a id="review_signals"></a>
**`review_signals`** は、Hermes自身のバックグラウンドのメモリ/スキルレビューの背後にあるカウンターを表します。これにより、ターン後の処理を行うプラグインは、自分でツール呼び出しを数える代わりに、ユーザーの自己学習のケイデンスに従うことができます。フックはレビューの判断の前に発火し、期限フラグはその判断が使う値そのものです。

| キー | 型 | 意味 |
|-----|------|---------|
| `tool_iterations_since_skill_review` | `int` | 最後のスキルレビューまたは `skill_manage` 呼び出し以降のエージェントループのイテレーション数（トリガーが比較した値。`skill_review_due` のときはフックの後にリセットされる） |
| `user_turns_since_memory_review` | `int` | 最後のメモリレビューまたは `memory` 呼び出し以降のユーザーのターン数（`memory_review_due` のターンではすでに `0` にリセットされている） |
| `skill_review_due` | `bool` | このターンでスキルのトリガーが発火した（`skills.creation_nudge_interval` に到達し、`skill_manage` が利用可能） |
| `memory_review_due` | `bool` | このターンでメモリのトリガーが発火した（`memory.nudge_interval` に到達し、組み込みメモリが利用可能） |
| `background_review_enabled` | `bool` | 期限に達したトリガーが実際にレビューをフォークする: `auxiliary.background_review.enabled` で、cronの実行でもサブエージェントでもない |
| `skill_nudge_interval` | `int` | `skills.creation_nudge_interval`（0 = オフ） |
| `memory_nudge_interval` | `int` | `memory.nudge_interval`（0 = オフ） |

Hermesは `background_review_enabled and (skill_review_due or memory_review_due)` のときにレビューをフォークします。このマッピングはすべてのコールバックで共有され、変更できません。

```python
def on_post_llm_call(review_signals=None, side_question_token=None, **kwargs):
    if review_signals and review_signals["skill_review_due"]:
        ...  # the user's skill cadence is due: do the plugin's own skill follow-up now
```

**例 — 外部メモリに同期する:**

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
        pass  # best-effort

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

### `pre_verify`

**エージェントがコードを編集したターンごとに1回**、終了する直前（組み込みのverify-on-stopガードの後）に発火します。これはユーザー/プラグインのポリシーゲートです: コールバックはエージェントを停止させる代わりに継続させることができます — チェックを実行する、後回しにする、差分を整理する、など。

Hermesに同梱の検証ガイダンスは、デフォルトの `pre_verify` フックではありません。編集されたコードに新しい検証の証拠がない場合に、証拠ベースのverify-on-stopのナッジに追加されるものなので、2つ目のデフォルトの継続パスを作ることはありません。組み込みの証拠ナッジを簡潔に保つには `agent.verify_guidance: false` を設定してください。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, platform: str, model: str, coding: bool,
                attempt: int, final_response: str, changed_paths: list, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 現在のセッションの一意な識別子 |
| `platform` | `str` | セッションが実行されている場所（`"cli"`、`"telegram"` など） |
| `model` | `str` | モデル識別子 |
| `coding` | `bool` | ターンがコーディングの姿勢（コードのワークスペース内）にあるかどうか — これでフックの対象範囲を絞る |
| `attempt` | `int` | このターンがすでにナッジされた回数（最初は0） — これで自己抑制する |
| `final_response` | `str` | エージェントがこれから届けようとしている回答 |
| `changed_paths` | `list` | このターンでエージェントが編集したファイル（ソート済み、ここでは常に空でない） |

`pre_tool_call` フックが `tool_name` で対象範囲を絞るのと同じように、`coding` を確認してフックをコーディングのコンテキストに限定し、`attempt` で1回限りにします（シェルフックはどちらも `.extra` から読み取ります） — これにより複数の `pre_verify` フックを登録し、それぞれが発火すべき場所でのみ発火するようにできます。

**発火場所:** `agent/conversation_loop.py` 内で、エージェントが最終回答を受け入れようとする時点、verify-on-stopのチェックの直後 — ただし、このターンでエージェントがコードを編集し、少なくとも1つの `pre_verify` フックが登録されている場合のみ。

**戻り値 — エージェントを継続させる:**

```python
return {"action": "continue", "message": "Run the formatter on your changes, then finish."}
```

`message` は合成されたユーザーのターンとして追加され、ループが再度実行されます。Claude-CodeのStop形式（`{"decision": "block", "reason": "..."}`。停止をブロックすることは*継続する*ことを意味する）も受け付けます。メッセージのないディレクティブ — またはその他の戻り値 — はターンを終了させます。

**回数制限:** 1つのターン内で連続する継続ディレクティブは `agent.max_verify_nudges`（デフォルト3）で上限が設けられているため、常に継続を指示するフックがループを閉じ込めることはありません。試みられた回答は履歴に保持されますが、エージェントがナッジされている間はユーザーに表示されません。

**冪等にする:** フックは各ナッジの後に再発火するので、`attempt` でゲートしてください（`if attempt: return None`） — そうしないと上限に達するまでナッジし続けるだけになります。

**ユースケース:** 創造的な反復作業中はテスト/リントを後回しにする、特定のパスにはグリーンなチェックを必須にする、changelogのエントリが存在するまで「完了」をブロックする、プロジェクト固有の検証チェックリストを実行する。

**例 — 創造的なUI作業ではチェックを後回しにする（範囲限定 + 1回限り）:**

```python
UI = (".tsx", ".jsx", ".css", ".scss")

def defer_ui_checks(coding, attempt, changed_paths, **kwargs):
    if attempt or not coding:
        return None  # one-shot, coding only
    if not all(p.endswith(UI) for p in changed_paths):
        return None  # only pure-UI edits
    return {
        "action": "continue",
        "message": "This is UI work — don't run tests/lints yet; ask the user to "
                   "eyeball it first, and clean the diff before any commit.",
    }

def register(ctx):
    ctx.register_hook("pre_verify", defer_ui_checks)
```

組み込みの証拠欠落ナッジに反映させるべき常設のガイダンスには、`agent.verify_guidance` を使ってください。検証を*ゲート*する必要のない、より広範なコーディング姿勢のルールには、`config.yaml` の `agent.coding_instructions` を優先してください — これはコーディングブリーフに乗るので、追加のターンのコストがかかりません。

---

### `transform_api_error_classification`

失敗したAPI呼び出しごとに1回、`agent/error_classifier.classify_api_error()` の先頭で、組み込みのパイプラインの前に発火します。プロバイダープラグインはこれを使って、コアにパッチを当てることなく自分のプロバイダーのエラーの癖を扱います。これは挙動を変えるもの（変換ファミリー）です: 返された分類は、リトライ、圧縮、認証情報のローテーション、フォールバックのルーティングを左右します。

コールバックは解析済みのエラーコンテキストをkwargsとして受け取ります — `provider`（これで自分の範囲に限定する）、`model`、`status_code`、`error_type`、`error_code`、`error_message`、`error_body`、`error`、`approx_tokens`、`context_length`、`num_messages`。辞退するには `None` を、エラーを引き受けるにはdictを返します:

```python
return {"reason": "model_not_found",   # required: a FailoverReason name
        "retryable": False, "should_fallback": True}  # optional recovery-hint overrides
```

ディスパッチは全実行後に最初を選択する方式です: すべてのコールバックが実行され、失敗は隔離され、登録順で最初の有効な結果が勝ちます（有効だが負けた結果は実行時警告としてログに記録されます）。無効なdictや未知の理由はスキップされるため、壊れたプラグインが分類を壊すことはありません。

**プライバシー:** `error_message` と `error_body` にはリダクトされていないプロバイダーのデータが含まれる可能性があります。**Pythonプラグインのみ** — シェルからの登録は設定の解析時に警告とともに拒否されます。

---

### `on_session_start`

まったく新しいセッションが作成されたときに**1回**発火します。セッションの継続時（既存のセッションでユーザーが2つ目のメッセージを送ったとき）には発火**しません**。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 新しいセッションの一意な識別子 |
| `model` | `str` | モデル識別子 |
| `platform` | `str` | セッションが実行されている場所 |

**発火場所:** `agent/conversation_loop.py` の `run_conversation()` 内で、新しいセッションの最初のターン中 — 具体的にはシステムプロンプトが構築された後、ツールループが始まる前。判定は `if not conversation_history`（以前のメッセージがない = 新しいセッション）です。

**戻り値:** 無視されます。

**ユースケース:** セッションスコープの状態の初期化、キャッシュのウォームアップ、外部サービスへのセッションの登録、セッション開始のログ記録。

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

### `on_session_open`

クライアントがローカルセッションを**作成または開いた**ときに発火します: `hermes`、TUI、Desktopでの新しいチャット、または `hermes --resume` / セッションピッカーによる保存済みセッションの再オープン。モデルが会話を見る前に準備するために使います。例えば、メモリプラグインが最近のコンテキストを事前に読み込む、ステータスバーがセッションを開いた状態として表示する、などです。[`on_session_start`](#on_session_start) とは異なります。そちらは最初のエージェントのターン中に発火し、再オープン時には発火しません。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, platform: str, reason: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 開かれたセッション |
| `platform` | `str` | それを作成したサーフェス: `cli`、`tui`、`desktop`、`acp` など |
| `reason` | `str` | セッションにまだメッセージがない間（新しいチャット）は `create`、会話がある場合（保存済みのチャットの再オープン）は `resume` |

**発火場所:** Gateway（`gateway/session_open.py`）内で、クライアント接続がまだアタッチしていなかったローカルセッションにアタッチしたとき。オープンごとに1回: すでに開いているセッションをクライアントが再読み込みしたり、ソケットが切れた後に再接続したりしても、再度は発火しません。2つのクライアントが同じセッションを開くと2回発火します。セッションを所有するプロファイル内で実行されるため、複数のプロファイルを提供するGatewayでは、各プロファイルのプラグインは自分のセッションのみを見ます。コールバックは専用のスレッドで実行されます: セッションのオープンはそれを待ちませんが、セッションの次のターンは最大5秒（`plugins.hook_callback_timeout` の方が長ければその値）まで待ちます。それを過ぎるとターンは進行し、コールバックは実行を続けます。あるセッションのコールバックがまだ実行中の間、そのセッションのさらなるオープンは1つの呼び出しとしてその後ろで待機し、終了したときに最新のオープンの詳細で1回発火します。`--safe-mode` / `--ignore-user-config` のセッションは何も発火しません。メッセージングのチャット（Telegram、Discordなど）では発火しません: 最初のメッセージがセッションを開始するので、そこでは `on_session_start` を使ってください。

**戻り値:** 無視されます。

**例 — 最初のターンの前にメモリを事前に読み込む:**

```python
def warm(session_id, platform, reason, **kwargs):
    _cache[session_id] = load_recent_notes(limit=20)

def register(ctx):
    ctx.register_hook("on_session_open", warm)
```

---

### `on_session_end`

結果にかかわらず、すべての `run_conversation()` 呼び出しの**最後の最後**に発火します。ユーザーが終了したときにエージェントがターンの途中だった場合は、CLIの終了ハンドラーからも発火します。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | セッションの一意な識別子 |
| `completed` | `bool` | エージェントが最終応答を生成した場合は `True`、それ以外は `False` |
| `interrupted` | `bool` | ターンが中断された場合（ユーザーが新しいメッセージを送った、`/stop`、または終了）は `True` |
| `model` | `str` | モデル識別子 |
| `platform` | `str` | セッションが実行されている場所 |

**発火場所:** 2か所:
1. **`agent/turn_finalizer.py`** — すべての `run_conversation()` 呼び出し（`agent/conversation_loop.py`）の最後、すべてのクリーンアップの後。ターンがエラーになった場合でも常に発火します。
2. **`cli.py`** — CLIのatexitハンドラー内で、ただし終了が起きたときにエージェントがターンの途中（`_agent_running=True`）だった場合**のみ**。処理中のCtrl+Cや `/exit` を捕捉します。この場合、`completed=False` かつ `interrupted=True` です。

**戻り値:** 無視されます。名前に反して、このフックは**毎ターン**の後に発火するため、ユーザーに何かを表示することはできません。セッション終了時に何かを表示するには、[`on_session_finalize`](#on_session_finalize) からメッセージを返してください。

**ユースケース:** バッファのフラッシュ、接続のクローズ、セッション状態の永続化、セッション時間のログ記録、`on_session_start` で初期化したリソースのクリーンアップ。

**例 — フラッシュとクリーンアップ:**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # Flush accumulated data to disk or external service
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"Session {session_id} ended: {status}, {cache['tool_calls']} tool calls")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**例 — セッション時間の追跡:**

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

CLIまたはGatewayがアクティブなセッションを**終了処理**するときに発火します — 例えば、ユーザーが `/new` を実行したときや、アクティブなエージェントがある状態でCLIが終了したとき。リソースのみのアイドルキャッシュ退避では、永続的な会話は確定されません。出ていくセッションIDに紐づく状態をフラッシュするために使ってください。Gatewayのリセット時には、このコールバックが実行される前に置き換えのセッションがすでに存在しています。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` または `None` | 出ていくセッションID。アクティブなセッションが存在しなかった場合は `None` のことがあります。 |
| `platform` | `str` | `"cli"` またはメッセージングプラットフォーム名（`"telegram"`、`"discord"` など）。 |

**発火場所:** CLI/TUIの終了処理（`hermes -z` のワンショット実行の終了時を含む。成功・失敗を問わない）と、Gatewayのリセットまたはシャットダウンのパス。Gatewayのシャットダウンは、対応する `on_session_reset` なしで確定することがあります。

**戻り値 — ユーザーにメッセージを表示する:** 空でない `str`、または `"message"` 文字列を持つdictを返すと、Hermesはそれを通知としてユーザーに表示します。モデルのターンになることはなく、アシスタントの返信を編集することもありません。何も表示しない場合は `None` を返します。表示される場所:

| サーフェス | 表示のされ方 |
|---------|---------|
| CLIの `/new` | 新しいセッションが始まる前にターミナルに出力されます。 |
| CLIの終了 | 終了時のサマリー（`hermes --resume …`）の上に出力されます。 |
| `hermes chat -q` / `-z` のワンショット | **stderr** に出力されるため、stdoutには引き続き回答のみが含まれます。 |
| TUIの `/new` | 新しいセッションのトランスクリプト内のシステム行。 |
| メッセージングGatewayの `/new` または `/reset` | セッションを所有していたチャットに、独立したメッセージとして送信されます。 |
| Gatewayのシャットダウン / 再起動 | アダプターが切断される前に、アクティブな各チャットに送信されます。 |

誰も見ていない場合は配信されません: Gatewayでのアイドル期限切れとキャッシュ退避、TUIの終了、リーパーや切断によって閉じられたTUI/Desktopのセッション、Desktopの新規チャットやチャットの削除。セッションを確定しないcronジョブからも配信されません。これらの場合、メッセージは破棄されますが、フックは引き続き実行されます。

```python
def digest(session_id, **kwargs):
    count = _edits.pop(session_id, 0)
    if count:
        return {"message": f"📝 This session edited {count} file(s)."}

def register(ctx):
    ctx.register_hook("on_session_finalize", digest)
```

**ユースケース:** セッションIDが破棄される前に最終的なセッションメトリクスを永続化する、セッションごとのリソースをクローズする、最終的なテレメトリイベントを送出する、キューに入った書き込みを排出する。

---

### `on_session_reset`

CLIまたはTUIのセッション境界、あるいはGatewayがアクティブなチャットに**新しいセッションキーを差し替えた**ときに発火します。これによりプラグインは、次の `on_session_start` を待つことなく、クリアされた会話状態に反応できます。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 新しいセッションのID（すでに新しい値にローテーション済み）。 |
| `platform` | `str` | `"cli"`、`"tui"`、またはメッセージングプラットフォーム名。 |
| `reason` | `str`、任意 | CLIとGatewayのリセットパスで存在します。 |
| `old_session_id` | `str`、任意 | Gatewayのみ。出ていくセッションID。 |
| `new_session_id` | `str`、任意 | Gatewayのみ。置き換えのセッションID。 |

**発火場所:** CLIは `session_id`、`platform`、`reason` を提供し、TUIは `session_id` と `platform` を提供し、Gatewayは置き換えのキーを割り当てた後に `reason`、`old_session_id`、`new_session_id` を追加します。Gatewayのリセット時の順序は次のとおりです: 置き換えを作成して永続化 → `on_session_finalize(old_id)` → `on_session_reset(new_id)` → 最初の受信ターンで `on_session_start(new_id)`。

**戻り値:** 無視されます。

**ユースケース:** `session_id` をキーとするセッションごとのキャッシュのリセット、「セッションがローテーションした」という分析イベントの送出、新しい状態バケットの準備。

---

### `on_session_delete`

セッションが実際になくなったとき — 削除（単一、一括、空セッションのスイープ、保持期間によるprune、ゴーストのpruneメンテナンス、プロファイルの修復）または意図的なアーカイブ — に、**操作ごとに1回、トランザクションのコミット後に**発火します。すべてのサーフェスは同じ `SessionDB` のメソッドに行き着くため、CLI（`hermes sessions delete|prune|archive`、`/exit --delete`）、TUI、Desktop/ダッシュボード（単一および一括の削除、アーカイブ）、GatewayのAPIサーバーのすべてが対象です。ディスク上にセッションごとの状態を保持するプラグインは、セッションストアをポーリングせずにクリーンアップできます。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, reason: str, profile: str | None, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 操作の対象となったセッション。常に存在します。 |
| `reason` | `str` | `"delete"`、`"archive"`、`"empty_sweep"`、`"prune"`、`"ghost_prune"`、または `"profile_repair"`。 |
| `profile` | `str \| None` | セッションストアを所有するプロファイル（ルートのホームでは `"default"`）。どのプロファイルのホームにも属さないストアでは `None`。 |
| `deleted_ids` | `list[str]`、任意 | 削除されたすべての行。カスケード削除された委譲の子、圧縮チェーンのリンク、一括削除の各行を含みます。複数行が削除された場合に存在します。 |
| `sessions_dir` | `str`、任意 | 呼び出し元が渡した場合、プロファイルの `sessions/` ディレクトリ。 |

**発火場所:** コミット後のみ、かつ何かが変わった場合のみ: 未知のID、繰り返しのアーカイブ、アーカイブ解除では何も発火しません。`reason="archive"` はユーザーがセッションを非表示にしたことを意味します（トランスクリプトは保持され、アーカイブ解除できます）。アイドル時の自動アーカイブのスイープとプロファイルの取り込みでは発火しません。失敗したコールバックは他のすべてのオブザーバーと同様にログに記録されてスキップされ、削除を失敗させることはありません。ペイロードにメッセージ内容は含まれません。

**配信範囲:** プロセスローカル — コールバックは削除をコミットしたプロセス内でのみ実行されます。永続的なプロセス間の配信チャネルではなく、ちょうど1回の保証もありません: コールバックから駆動する外部の副作用は、呼び出しの欠落や重複に耐えられるようにしてください。

**戻り値:** 無視されます。

**ユースケース:** セッションごとのサイドカーデータベースやキャッシュディレクトリの削除、ユーザーがチャットをアーカイブしたときの関連タスクのクローズ、「セッションが削除された」という分析イベントの送出。

---

ツールスキーマ、ハンドラー、高度なフックパターンを含む完全な手順については、**[プラグイン作成ガイド](../../developer-guide/plugins/index.md)** を参照してください。

---

### `post_compaction`

圧縮が**コミットされた**後に1回発火します。これによりプラグイン（メモリプロバイダー、コンテキストエンジン、外部の知識ストア）は、以前に注入した詳細がちょうど要約されて消えたことを知り、通常は次の `pre_llm_call` から再注入できます。

**コールバックのシグネチャ:**

```python
def my_callback(session_id: str, kind: str, reason: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_id` | `str` | 現在、圧縮されたトランスクリプトを保持しているセッション。 |
| `old_session_id` | `str` | 新しいセッションにローテーションした場合に、圧縮が開始されたセッション。その場で圧縮された場合は `""`。 |
| `kind` | `str` | `"summarize"`（完全な圧縮）、`"micro_summarize"`、`"prune_tool_results"`、または `"provider_native"`（Codex app-server）。 |
| `reason` | `str` | 何がトリガーしたか: `"manual"`（`/compress`）、`"pre_api"`、`"post_tool"`、`"overflow"`、`"gateway_hygiene"`、`"between_turns"`、`"proactive_prune"`、`"provider"`、または `"unknown"`。 |
| `before_tokens` / `after_tokens` | `int \| None` | 前後のトランスクリプトのおおよそのトークン推定値。不明な場合（Codexがスレッドを所有する）は `None`。 |
| `messages_removed` | `int \| None` | 前のメッセージ数から後のメッセージ数を引いたもの。不明な場合は `None`。 |

**発火場所:** エージェントループ内で、圧縮したプロセス（CLI、TUI、Desktopのバックエンド、Gatewayのエージェント、cron）で発火します。Gatewayの `/compress` では、コミット後にGateway内で発火します。何も変更しなかった試行（要約の中止、クールダウンによるブロック、増加してしまうための拒否）とバックグラウンドレビューのフォークは何も発火しません。

**戻り値:** 無視されます。例外を送出したコールバックはログに記録されてスキップされます。圧縮はすでにコミットされており、影響を受けません。コールバックは `plugins.hook_callback_timeout` で制限されます。

**ユースケース:** 圧縮後のセッションコンテキストの再注入、外部インデックスの更新、圧縮回数のカウント。

### `post_curator_run`

キュレーターのパスが実行ごとのレポートディレクトリを書き出した後に1回発火します。これによりプラグインやシェルフックは、`logs/curator/` をポーリングする代わりに、キュレーターが変更した内容に反応できます（新しいスキルを別のプロファイルに振り分ける、アーカイブを上書きする、監査ログを書く）。

**コールバックのシグネチャ:**

```python
def my_callback(run_dir: str, run_json_path: str, report_md_path: str, dry_run: bool, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `run_dir` | `str` | プロファイルの `logs/curator/` 配下にある実行のレポートディレクトリ。 |
| `run_json_path` | `str` | `run.json`。機械可読な記録（追加、アーカイブ、統合、刈り込み、状態遷移、cronの書き換え）。 |
| `report_md_path` | `str` | `REPORT.md`。人間が読めるレポート。 |
| `dry_run` | `bool` | プレビュー（`hermes curator run --dry-run`）では `True`: レポートには変更される予定の内容が列挙され、実際には何もアーカイブも書き換えもされていません。レポートに基づいて行動する前に確認してください。 |

**発火場所:** パスを実行したプロセス（Gatewayのスケジュールされたキュレーターのティック、または `hermes curator run`）内で、そのプロファイルのもとで、両方のファイルが書き込まれた後。レポートディレクトリを作成できなかったパスは何も発火しません。

**戻り値:** 無視されます。例外を送出したコールバックはログに記録されてスキップされます。キュレーターのパスは影響を受けません。

### `agent_loop_stopped`

Gatewayが**実行中のエージェントのターンを中断した**ときに発火します — ループの動作中にユーザーが `/stop` を実行した場合、または `/new` 内の実行中エージェント用ファストパスがセッションを差し替える前に進行中の実行をクリアした場合です。`on_session_finalize` とは異なり、ターンが進行中のより早い段階で発火するため、プラグインはエージェントループが決して消費しないターンごとの外部リソース（例: ツール結果を待っていたアウトバウンドのRPC）を破棄できます。

両方の中断サーフェスで発火します: メッセージング**Gateway**（`/stop`、`/new` のファストパス）と、**TUI/デスクトップ**の `session.interrupt` パス（プラットフォームは `"tui"` として報告されます）。素のCLIでは発火しません。そこには同等の中断サーフェスがありません。

**コールバックのシグネチャ:**

```python
def my_callback(session_key: str, platform: str, reason: str, invalidation_reason: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `session_key` | `str` | 実行が中断されたセッション。 |
| `platform` | `str` | メッセージングプラットフォーム名（`"telegram"`、`"discord"` など）。不明な場合は空文字列。 |
| `reason` | `str` | エージェントが中断された理由（例: `"user_stop"`、リセット/newの理由）。 |
| `invalidation_reason` | `str` | キューに入ったセッション状態が無効化された理由（例: `"stop_command"`、`"stop_command_thread_sibling"`、`"stop_command_chat_scope"`、`"reset_command"`）。 |

**発火場所:** `gateway/run_agent_cache.py::_interrupt_and_clear_session` 内で、`request_hard_interrupt()` が実行中のエージェントを中断した直後。実際のエージェントが実行中だった場合のみ — 保留中のセンチネルの `/stop` パス（まだエージェントループが開始されていない）では、破棄すべき進行中の作業がないため、このフックは発火**しません**。遅い `/new` のリセットパスでは、代わりに後で `_handle_reset_command` 内で `on_session_finalize` が発火します。

**戻り値:** 無視されます。

**ユースケース:** ループが決して消費しないツール結果を待ってブロックしている外部リクエストのキャンセル、接続された音声/リアルタイムクライアントへのツール呼び出し放棄の通知、アクティブなターンの間だけ保持していたターンごとの認証情報やロックの解放。

---

### `subagent_start`

`delegate_task` が子の `AIAgent` を構築した後、その子が実行される前に、**子エージェントごとに1回**発火します。単一のタスクを委譲しても3つのバッチを委譲しても、このフックは各子に対して1回ずつ発火します。

このフックは委譲/サブエージェントのライフサイクルに固有のものです。Gateway、CLI、cron、バッチ、MoA、その他のランナーから発生するエージェント実行に対する汎用的な「あらゆるエージェント呼び出しの前」のゲートではありません。

**コールバックのシグネチャ:**

```python
def my_callback(parent_session_id: str | None,
                parent_turn_id: str,
                parent_subagent_id: str | None,
                child_session_id: str | None,
                child_subagent_id: str,
                child_role: str,
                child_goal: str,
                **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `parent_session_id` | `str \| None` | 委譲元の親エージェントのセッションID。 |
| `parent_turn_id` | `str` | 委譲を要求した親エージェントのターンのID（利用可能な場合）。 |
| `parent_subagent_id` | `str \| None` | この子が別のサブエージェントによって起動された場合は親のサブエージェントID。トップレベルの親エージェントでは `None`。 |
| `child_session_id` | `str \| None` | 子エージェントに割り当てられたセッションID。 |
| `child_subagent_id` | `str` | 委譲の可観測性と制御で使われる安定したサブエージェントID。 |
| `child_role` | `str` | 委譲ポリシー適用後の実効的な子のロール。例: `"leaf"` または `"orchestrator"`。 |
| `child_goal` | `str` | 子エージェントが実行する委譲されたゴール/プロンプト。 |

**発火場所:** `tools/delegate_tool.py` の `_build_child_agent()` 内で、子の `AIAgent` が構築されてサブエージェントの識別メタデータが付与された後、`_run_single_child()` が子を実行する前。

**戻り値:** 無視されます。これはオブザーバーフックのみです。値を返しても子エージェントの実行をブロックしたり変更したりすることはありません。

**ユースケース:** サブエージェントの作成のログ記録、親/子セッションの関係のマッピング、ネストした委譲ツリーの追跡、実行前の監査記録の送出、子ごとの可観測性リソースの事前割り当て。

**例 — サブエージェントの作成をログに記録する:**

```python
import logging

logger = logging.getLogger(__name__)

def log_subagent_start(
    parent_session_id,
    parent_turn_id,
    child_session_id,
    child_subagent_id,
    child_role,
    child_goal,
    **kwargs,
):
    logger.info(
        "SUBAGENT_START parent=%s turn=%s child_session=%s child=%s role=%s goal=%r",
        parent_session_id,
        parent_turn_id,
        child_session_id,
        child_subagent_id,
        child_role,
        child_goal[:200],
    )

def register(ctx):
    ctx.register_hook("subagent_start", log_subagent_start)
```

:::info
`subagent_start` は委譲の可観測性に便利ですが、ブロッキングのポリシーフックではありません。子が構築される前に委譲をブロックするには、[`pre_tool_call`](#pre_tool_call) を使って `delegate_task` のツール呼び出しをブロックしてください。
:::

---

### `subagent_stop`

`delegate_task` が終了した後、**子エージェントごとに1回**発火します。単一のタスクを委譲しても3つのバッチを委譲しても、このフックは各子に対して1回ずつ発火します。ディスパッチは子のfutureがすべて完了した後に親スレッド上で直列化され、各Pythonコールバックの本体は同じ呼び出し元スレッド上で実行されます（タイムアウト用ワーカー上ではありません）。

**コールバックのシグネチャ:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                tool_call_history: list[dict], duration_ms: int, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `parent_session_id` | `str` | 委譲元の親エージェントのセッションID |
| `child_role` | `str \| None` | 子に設定されたオーケストレーターのロールタグ（機能が有効でない場合は `None`） |
| `child_summary` | `str \| None` | 子が親に返した最終応答 |
| `child_status` | `str` | `"completed"`、`"failed"`、`"interrupted"`、または `"error"` |
| `tool_call_history` | `list[dict]` | 順序付きのメタデータのみのツール呼び出し: `tool_name`、サイズ制限付きの `tool_input`、`input_bytes`、`output_bytes`、`status`。生の入力と出力は除外されます |
| `duration_ms` | `int` | 子の実行に費やした実時間（ミリ秒） |

**発火場所:** `tools/delegate_tool.py` 内で、`ThreadPoolExecutor.as_completed()` がすべての子のfutureを完了させた後。`invoke_hook("subagent_stop", ...)` は親スレッドにマーシャリングされるため、作成者が子のプールの再入を目にすることはなく、コールバックはその呼び出し元スレッド上にとどまります。

**戻り値:** 無視されます。

**ユースケース:** オーケストレーションのアクティビティのログ記録、課金のための子の実行時間の累積、委譲後の監査記録の書き込み。

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
大量の委譲（例: オーケストレーターのロール × 5つのリーフ × ネストの深さ）では、`subagent_stop` はターンごとに何度も発火します。コールバックは高速に保ち、重い処理はバックグラウンドのキューに回してください。
:::

---

### `pre_gateway_dispatch`

Gateway内で**受信した `MessageEvent` ごとに1回**、入口で、内部イベントのガードの後、メッセージが受け入れられたりディスパッチされたりする前に発火します。認可に対してどの位置で実行されるかは、メッセージがどのように届いたかによります（下記の**発火場所**を参照）。これは、単一のプラットフォームアダプターにはきれいに収まらないGatewayレベルのメッセージフローのポリシー（聞くだけの時間帯、人間への引き継ぎ、チャットごとのルーティングなど）のためのインターセプトポイントです。

**コールバックのシグネチャ:**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `event` | `MessageEvent` | 正規化された受信メッセージ（`.text`、`.source`、`.message_id`、`.internal` などを持つ）。 |
| `gateway` | `GatewayRunner` | アクティブなGatewayランナー。プラグインはサイドチャネルの返信（オーナーへの通知など）のために `gateway.adapters[platform].send(...)` を呼び出せます。 |
| `session_store` | `SessionStore` | `session_store.append_to_transcript(...)` によるトランスクリプトへのサイレントな取り込みのため。 |

**発火場所:** **内部イベントはフックを完全にスキップします**（これらはシステムが生成するもの — バックグラウンドプロセスの完了など — であり、ユーザー向けのポリシーで門番されてはなりません）。それ以外の場合:

| パス | フックが実行されるタイミング |
|------|--------------------|
| プラットフォームアダプター（Telegram、Discord、Slackなど）とリレーのスラッシュコマンド | `GatewayRunner._handle_message()`（`gateway/run_inbound.py` の `_hm_admit_event`）内で、認証/ペアリングの**前**。そのためプラグインは未認可の送信者を処理できます。 |
| Webhookの配信とリレーのメッセージ | `admit_producer`（`gateway/platforms/webhook_ingress.py`）内で、配信の認可の**後**、受け入れのコミットの**前**: 拒否された送信者はフックに到達せず、スキップされた配信は受け入れられず、書き換えたものが受け入れられます。すでに受け入れられた配信のプロバイダーによる再試行やリレーの再送は、フックを再度呼び出すことなくその受け入れに解決されます。リレーでは、登録されたフックがある場合の受け入れはリレーソケットのリーダーの外で実行されるため、コールバックは `gateway.adapters[...].send(...)` をawaitできます。1つのチャットのメッセージは引き続き到着順に互いを待ちます。 |
| ローカルのCLI、TUI、Desktop、APIサーバーのターン | 発火しません。 |
| キューに入った受け入れの実行時（初回実行または再起動時の再生） | 発火しません: メッセージはすでに入口でフックを通過しているため、`skip` で受け入れ済みの入力を破棄することはできません。 |

**戻り値:** `None` またはdict。最初に認識されたアクションのdictが勝ち、残りのプラグインの結果は無視されます。プラグインのコールバックの例外はキャッチされてログに記録され、エラー時はGatewayは常に通常のディスパッチにフォールスルーします。

コールバックは `async def` でも構いません: Gateway自身のイベントループ上でawaitされるため、ループに紐づく処理（`asyncio.Event`、aiohttpセッション、`asyncio.to_thread`）をawaitしても進行し、コールバックの実行中も他の受信メッセージは流れ続けます。このフックは意図的に `plugins.hook_callback_timeout` で制限されていません — タイムアウト時にメッセージを破棄するのも通すのも、ポリシーゲートとしては誤りだからです — そのため、決して戻らないコールバックはそのメッセージのディスパッチを止めます。

| 戻り値 | 効果 |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | メッセージを破棄します — エージェントの返信なし、ペアリングフローなし、認証なし（Webhookやリレーの配信は受領を確認されますが、受け入れられません）。プラグインがそれを処理した（例: トランスクリプトにサイレントに取り込んだ）ものとみなされます。 |
| `{"action": "rewrite", "text": "new text"}` | `event.text` を置き換え、変更されたイベントで通常のディスパッチを続行します。バッファされた周囲のメッセージを1つのプロンプトにまとめるのに便利です。 |
| `{"action": "allow"}` / `None` | 通常のディスパッチ — 残りの認証 / ペアリング / エージェントループのチェーンを実行します。 |

**ユースケース:** 聞くだけのグループチャット（タグ付けされたときだけ応答し、周囲のメッセージはコンテキストとしてバッファする）、人間への引き継ぎ（オーナーが手動でチャットを対応している間、顧客のメッセージをサイレントに取り込む）、プロファイルごとのレート制限、ポリシー駆動のルーティング。

**例 — ペアリングコードをトリガーせずに未認可のDMを黙って破棄する**（プラットフォームアダプターのメッセージ。拒否されたWebhookやリレーの送信者はフックに到達しません）:

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**例 — メンションされたときに周囲のメッセージのバッファを1つのプロンプトに書き換える:**

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

### `post_gateway_admission`

Gateway内で**受け入れられた内部以外のメッセージごとに1回**、認可、ボットの受け入れ、一時停止/ドレイン、保留中の返信のインターセプト（clarify、確認）、実行中セッションのレーン（ステアリング、ビジー時のコマンド）、アイドル時のスラッシュコマンドのディスパッチの後に発火します。拒否された、無視された、内部の、および制御のトラフィックがここに到達することはありません。確保されたセッションスロット内で実行されるため、同じチャットへの同時のメッセージはその後ろでキューに入ります。また、メッセージの**ルーティング先プロファイル**のスコープのもとで実行されるため、そのプロファイルのプラグインのみが実行されます。

メッセージを消費するには `{"action": "handled", "reply": "..."}` を返します: エージェントのターンはスキップされ、`reply`（空でない文字列の場合）が通常のルートで配信されます。黙って消費するには `reply` を省略します。その他の戻り値では通常のエージェントのターンが実行されます。

```python
def consume(session_key, platform, source, message_id, text, **kwargs):
    if not text.startswith("follow up:"):
        return None                       # ordinary agent turn
    enqueue_follow_up(source, text)       # commit your own durable state first
    return {"action": "handled", "reply": "Got it - queued."}

def register(ctx):
    ctx.register_hook("post_gateway_admission", consume)
```

**フェイルオープン。** 例外を送出した、タイムアウトした（`plugins.hook_callback_timeout`）、またはそれ以外のものを返したコールバックはログに記録され、メッセージはエージェントへ進むため、1つのバグのあるプラグインがプロファイルのトラフィックをブロックすることはありません。フェイルクローズの挙動が必要なプラグインは、自分でエラーをキャッチし、独自の失敗時の返信とともに `handled` を返してください。ペイロードはスナップショットです（`source` は `SessionSource.to_dict()`）。アダプター経由で直接送信せず、代わりに `reply` を返してください。重複排除と永続的な受け入れはプラグインの責任です。

---

### `gateway_platform_event`

サポートされるプラットフォームネイティブのイベントについて、Gatewayの通常のプロファイル単位の認可チェックが成功した**後**にのみ発火します。コールバックはプレーンな辞書を受け取ります。生のSDKオブジェクト、アダプターのハンドル、ボットクライアント、コールバックのコンテキストがこの安定した契約に含まれることはありません。

Telegramのメッセージリアクションが最初にサポートされたイベントで、その後メッセージの編集、削除、スレッドのライフサイクルのイベントが続きました:

```python
def on_platform_event(platform, event_type, payload, **kwargs):
    if platform == "telegram" and event_type == "reaction":
        print(payload["chat_id"], payload["message_id"], payload["emojis"])
    elif event_type == "message_edited":
        print(platform, payload["chat_id"], payload["message_id"], payload["text"])

def register(ctx):
    ctx.register_hook("gateway_platform_event", on_platform_event)
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `platform` | `str` | 安定したプラットフォームID（`"telegram"`、`"discord"`）。 |
| `event_type` | `str` | イベントローカルの契約ID（下の表を参照）。 |
| `payload` | `dict` | イベント種別固有のフィールド。下記のイベント種別ごとに文書化されています。 |

すべてのペイロードは追加的かつイベント固有であり、一枚岩のGatewayペイロードのバージョンはありません。すべてのIDは文字列で、欠落している/利用できないフィールドは `None` になり、推測されることはありません。不正な形式のイベントと、送信元を認可できないイベントは破棄されます（フェイルクローズ）。一時的なTelegram Applicationの再構築では、オブザーバーはコアのハンドラーとともに再登録されます。

**イベントごとのペイロード契約（v1、追加的）:**

| `event_type` | プラットフォーム | ペイロードフィールド |
|--------------|-----------|----------------|
| `reaction` | telegram | `emojis: list[str]`、`custom_emoji_ids: list[str]`、`chat_id: str`、`message_id: str`、`thread_id: str \| None`（Telegramのリアクション更新にはトピックIDが含まれないため、現在は常に `None`）。 |
| `message_edited` | telegram, discord | `chat_id: str`、`message_id: str`、`thread_id: str \| None`、`text: str \| None`（編集後のテキストまたはキャプション、サイズ制限付き。メディアのみの編集やキャッシュされていない場合は `None`）、`edited_at: str \| None`（ISO 8601）。 |
| `message_deleted` | discord | `chat_id: str`、`message_id: str`、`thread_id: str \| None`、`author_id: str \| None`。Discordの削除イベントは削除した人を特定しないため、認可される送信元は削除されたメッセージの作成者です。キャッシュされていない削除は発火しません。 |
| `thread_created` | discord | `thread_id: str`、`parent_chat_id: str \| None`、`name: str \| None`、`owner_id: str \| None`。 |
| `thread_renamed` | discord | `thread_id: str`、`parent_chat_id: str \| None`、`old_name: str \| None`、`new_name: str`。名前が実際に変わった場合にのみ発火し、その他のスレッドの更新（アーカイブ、スローモード、タグ）は破棄されます。Discordのスレッド更新イベントには実行者が含まれないため、スレッドのオーナーが認可される送信元です。 |

ボット自身の段階的なメッセージ編集（ストリーミング）は、Discordで `message_edited` を発火させることはありません — ボットが作成したイベントは発火箇所で破棄されます。

このフックはオブザーバー専用です: 生のイベントへのアクセスやアダプターへのアクセスは追加**しません**。**生のSDKペイロードへのアクセスは意図的に提供されていません** — アダプターのSDKオブジェクトは予告なく形が変わり、進化させられないAPIサーフェスになってしまうためです。本当に必要な場合は、「安定性の保証なし」のラベルが付いた独自の明示的なケイパビリティ（`gateway.raw_events`）と独自の設計が必要です（#64228 で追跡中）。プラットフォーム上で*行動する*（リアクションの追加、スレッドの名前変更）には、[プラグインガイド](plugins.md#platform-actions)で説明されているケイパビリティでゲートされた `ctx.platform_actions` ファサードを使ってください — これはデフォルトでは `gateway.platform_actions` ケイパビリティの背後で無効になっています。`PluginContext.dispatch_tool()` はツールレジストリに登録されたツールしか呼び出せません。`send_message` は意図的にそこには登録されていません（その転送は、明示的なCLI、cron、kanban、MCPの配信パスのために予約されています）。将来のアウトバウンド配信の契約は、まずすべてのアダプターにわたって安定した配信内容/ハンドルを提供しなければなりません。この範囲では、機能しない `gateway_message_delivered` フックを事前登録することはしません。

---

### `pre_approval_request`

承認の決定が要求される前に発火します。プロンプトを出すサーフェス — 対話型CLI、Ink TUI、Gatewayプラットフォーム、ACPクライアント — を対象とし、保護されたエージェント指示ファイルへの書き込みのプロンプトやMCP/vaultの同意プロンプトも含みます。さらに、人間へのプロンプトなしに行われる `approvals.mode=smart` の決定（`surface="smart"`）も対象です。スマートモードでは、補助LLMが呼ばれる前にフックが実行されます。

カスタムの通知機能を組み込むのに最適な場所です — 例えば、許可/拒否の通知をポップアップするmacOSのメニューバーアプリや、すべての承認リクエストをコンテキストとともに記録する監査ログなどです。

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
| `command` | `str` | 評価されているターミナルコマンドまたは `execute_code` のスクリプト。スマートとGatewayのペイロードはオブザーバーへのディスパッチ前にリダクトされます。スマートオブザーバーのリダクトは `security.redact_secrets` が無効でも必須で、リダクトに失敗した場合はスマートフックはスキップされます。 |
| `description` | `str` | コマンドがフラグ付けされた人間が読める理由（複数のパターンに一致した場合は結合） |
| `pattern_key` | `str` | 承認をトリガーした主たるパターンキー（例: `"rm_rf"`、`"sudo"`） |
| `pattern_keys` | `list[str]` | 一致したすべてのパターンキー |
| `session_key` | `str` | セッション識別子。チャットごとに通知の範囲を絞るのに便利 |
| `surface` | `str` | クラシックCLIのセッション（危険なコマンド、保護されたエージェント指示ファイルへの書き込み、同意プロンプト。Gatewayがそのセッションのプロンプトを描画する場合も含む）では `"cli"`、非同期のプラットフォームの承認では `"gateway"`（Ink TUIとDesktopアプリも現在は `"gateway"` を報告します）、補助LLMによる自動の承認/拒否の決定では `"smart"`、MCP/vaultのプロンプトでは呼び出し元の同意サーフェス（`"mcp-elicitation/<server>"`、`"mcp-trust/<server>"`、`"vault-payment"`） |

**戻り値:** 無視されます。ここでのフックはオブザーバー専用で、承認を拒否したり事前に回答したりすることはできません。承認システムに到達する前にツールをブロックするには [`pre_tool_call`](#pre_tool_call) を使ってください。

**ユースケース:** デスクトップ通知、プッシュアラート、監査ログ、Slack Webhook、エスカレーションのルーティング、メトリクス。

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

プロンプトによる承認またはスマート承認の決定の後、プロンプトがタイムアウトしたか取り下げられた（回答前にターンが中断または終了した）後、またはGatewayが承認の通知を配信できなかったときに発火します。通知の失敗では、承認の決定が存在する前に `choice="notify_failed"` が送出されます。

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
| `choice` | `str` | プロンプトを出すサーフェスでは `"once"`、`"session"`、`"always"`、`"deny"`、`"timeout"`、`"cancelled"`（誰も回答しなかった — ターンが中断または終了したためにプロンプトが取り下げられた、あるいはCLIで承認コールバックが失敗した、prompt_toolkitのもとでコールバックが登録されていなかった、読み取りが中断されたためにユーザーに届かなかった。コマンドは実行されていない）、または `"notify_failed"`。スマートの決定では `"smart_approve"` または `"smart_deny"` |
| `decided_by` | `str` | スマートの決定では `"aux_llm"`。プロンプトを出すサーフェスでは存在しません |

**戻り値:** 無視されます。

**ユースケース:** 対応するデスクトップ通知を閉じる、最終的な決定を監査ログに記録する、メトリクスを更新する、レートリミッターを進める。

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `on_human_input_request`

プロンプトの内容や表示場所にかかわらず、人が何かを入力またはクリックするのを待ってHermesがブロックする直前に発火します。1組のフックですべてのブロッキングプロンプトをカバーするため、通知機能（「Hermesがあなたを待っています」）はプロンプトの種類ごとに別のフックをサブスクライブしたり、ツールの引数をパターンマッチしたりする必要がありません:

| `kind` | プロンプト | サーフェス |
|--------|--------|----------|
| `"sudo"` | `sudo` コマンドの実行前のマスクされたsudoパスワードのプロンプト | 対話型CLI、Ink TUI / Desktop（`sudo.request` カード）、`/dev/tty` のフォールバック |
| `"clarify"` | `clarify` ツールの質問 | CLI、Ink TUI / Desktop、Gatewayプラットフォーム |
| `"approval"` | 危険なコマンド / 書き込みの承認、MCP elicitationの同意、プラグインの承認トランスポート | CLI、Ink TUI / Desktop、ACP、Gatewayプラットフォーム、`ctx.register_approval_transport` プラグイン |

補助LLMによって行われる `approvals.mode=smart` の決定では、誰にも尋ねないため発火しません。すでに保留中の同一のプロンプトに合流するGatewayの承認でも発火しません（人が見るプロンプトは1つで、フックの組はそれに対して1回発火します）。新しい種類が後で追加される可能性があるため、扱わない種類は無視してください。

**コールバックのシグネチャ:**

```python
def my_callback(kind: str, request_id: str, session_id: str, session_key: str,
                platform: str, prompt: str, **kwargs):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `kind` | `str` | `"sudo"`、`"clarify"`、または `"approval"` |
| `request_id` | `str` | プロンプトごとに一意。対応する `on_human_input_resolved` も同じ値を持ちます |
| `session_id` | `str` | わかっている場合はHermesのセッションID、それ以外は `""` |
| `session_key` | `str` | セッション/チャットのルーティングキー。チャットごとに通知の範囲を絞るのに便利 |
| `platform` | `str` | Gatewayプラットフォーム（`"telegram"`、`"discord"` など）またはローカルのソース（`"cli"`、`"tui"`、`"desktop"` など） |
| `prompt` | `str` | 人が尋ねられている内容: `sudo`/`approval` ではコマンド、`clarify` では質問のテキスト。`security.redact_secrets` がオフでも、常にシークレットのリダクターでリダクトされます |

sudoのパスワードとclarifyの回答は、どちらのフックにも**決して**渡されません。**戻り値:** 無視されます。例外を送出したコールバックはログに記録され、プロンプトは続行されます。コールバックはこれからブロックしようとするスレッド上で実行されるため、高速に保ってください — ネットワーク呼び出し（プッシュサービス、Webhook）はバックグラウンドスレッドに渡してください。

### `on_human_input_resolved`

すべての `on_human_input_request` に対して、待機が終わった後にちょうど1回発火します。同じkwargsに加えて `outcome` を受け取ります:

| `kind` | `outcome` の値 |
|--------|------------------|
| `"sudo"` | `"provided"`、`"skipped"`（空の回答）、`"timeout"`、`"cancelled"`、`"error"` |
| `"clarify"` | サーフェスの結果: `"submitted"`、`"timed_out"`、`"cancelled"`、`"undelivered"`、`"terminated"`、または `"error"` |
| `"approval"` | `"once"`、`"session"`、`"always"`、`"deny"`、`"timeout"`、`"cancelled"`、`"notify_failed"`、またはプラグインのトランスポートが失敗した場合は `"transport_<failure>"` |

```python
import threading

def push_notification(text):
    ...  # call your push service here

def register(ctx):
    pending = {}

    def on_request(kind, request_id, prompt, session_key, **kwargs):
        pending[request_id] = threading.Timer(30, push_notification, (f"Hermes needs you ({kind}): {prompt[:80]}",))
        pending[request_id].start()  # only page if nobody answers within 30s

    def on_resolved(request_id, **kwargs):
        timer = pending.pop(request_id, None)
        if timer:
            timer.cancel()

    ctx.register_hook("on_human_input_request", on_request)
    ctx.register_hook("on_human_input_resolved", on_resolved)
```

---

### `on_room_member_activity`

ホストされた[Group Chat](/docs/user-guide/bot-mode#groups-and-group-chats)のメンバーのターンが実行されている間に発火します。メンバーはどのクライアントもアタッチしていない非表示の `Group: <room>` セッション上で実行されるため、ルームログの `turn.started` と `turn.settled` の間、ターンはブラックボックスです。このフックは、そのセッションがすでに生成しているランタイムイベント — ツールの開始/完了、承認リクエスト、ストリーミングされたテキストと推論、エラー — をルームの座標を付けて投影するため、クライアント（Hermes Crew、ダッシュボード、監査ログ）はテキストから何かを推測することなく、ツールカード、承認プロンプト、ライブのメンバーステータスを描画できます。Group Chatのランタイムは実行、スケジューリング、永続ログの所有権を保持し、プラグインは観察するだけです。

**コールバックのシグネチャ:**

```python
def my_callback(
    room_id: str,
    thread_id: str,
    member_id: str,
    turn_id: str,
    task_id: str,
    execution_generation: int,
    kind: str,
    seq: int | None,
    payload: dict,
    **kwargs,
):
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `room_id`、`thread_id`、`turn_id`、`task_id` | `str` | ルームログの `turn.*` と `message.member` イベントが持つのと同じ座標。これらで結合してください。 |
| `member_id` | `str` | 着席しているメンバー（`groups.state` の `members[].member_id`）。 |
| `execution_generation` | `int` | 同じタスクの再試行のたびに増加します。置き換えられた試行からのイベントは古い値を持ちます。 |
| `kind` | `str` | `tool.started`、`tool.completed`、`tool.output_risk`、`request.opened`（承認）、`message.delta`、`message.interim`、`reasoning.delta`、`turn.error`。新しい種類は追加的に増えます。 |
| `seq` | `int \| None` | メンバーセッションのプロセスごとのイベントシーケンス（`session.events.since` と同じ番号付け）。1つのGatewayプロセス内で単調増加し、再起動でリセットされます。 |
| `payload` | `dict` | 元となるセッションイベントのクライアントに安全な本文（`tool_id`、`name`、`args`、`result`、`request_id`、`choices`、`text` など）。承認のコマンドはすでに認証情報がリダクトされています。 |

**配信:** 登録された各コールバックは、独自のサイズ制限付きキューとワーカースレッドを持ちます（`on_stream_*` の仕組み）。遅いコールバックは最も古い保留中のイベントを破棄し、メンバーのターンを遅らせることはありません。ルームログには何も書き込まれません — 差分は永続的ではなく、再生もされません。永続性が必要なクライアントは、受け取ったものを自分で永続化してください。ローカルのメンバーのみ: 別のマシンから着席したメンバーはそのマシンのGateway上で実行され、そちらのプラグインがそれを見ます。

**戻り値:** 無視されます。

```python
def on_member_activity(room_id, member_id, turn_id, kind, payload, **kwargs):
    if kind == "request.opened":
        notify(f"{member_id} in {room_id} needs approval: {payload['command']}")

def register(ctx):
    ctx.register_hook("on_room_member_activity", on_member_activity)
```

---

### `pre_transcription`

STTディスパッチャー（`tools.transcription_tools.transcribe_audio`）内で、プロバイダーが解決された**後**、いずれかのバックエンドが呼び出される**前**に発火します。そのバックエンドが組み込みか、`type: command` のプロバイダーか、プラグインが登録したプロバイダーかは問いません。プラグインは、後からトランスクリプトを観察するだけでなく、文字起こしのリクエスト自体を操作できます。

**コールバックのシグネチャ:**

```python
def my_callback(
    file_path: str,
    provider: str,
    model: str | None,
    language: str | None,
    prompt: str | None,
    source: str | None,
    **kwargs,
) -> dict | None:
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `file_path` | `str` | 文字起こしされようとしている音声ファイルの絶対パス。読み取り専用。 |
| `provider` | `str` | 解決されたSTTプロバイダー（`local`、`groq`、`openai`、`mistral`、`xai`、`elevenlabs`、`deepinfra`、`local_command`、コマンドプロバイダー名、またはプラグインプロバイダー名）。 |
| `model` | `str \| None` | それまでに解決されたモデル。バックエンドのデフォルトが適用される場合は `None`。 |
| `language` | `str \| None` | プロバイダーの設定セクションの言語、または `None`。 |
| `prompt` | `str \| None` | 静的な [`stt.prompt`](../configuration.md#transcription-prompt-vocabulary-hints) の値、または `None`。 |
| `source` | `str \| None` | 呼び出し元のサーフェスのラベル（`gateway`、`voice_mode` など）。可観測性のためだけのもので、ディスパッチには使われません。 |

**戻り値:** `"prompt"`、`"language"`、`"model"` のいずれかを文字列にマッピングした `dict`、またはリクエストを変更しない場合は `None`。文字列以外の値、未知のキー、`file_path` は無視されます（`file_path` を変更しようとすると警告としてログに記録されます）。結果は `stt.prompt` の設定値の上に、**登録順で、フィールドごとに最後に書いたものが勝つ**形で適用されます。`prompt` に `""` を返すと、そのリクエストでは設定済みのプロンプトがクリアされます。

**ユースケース:** 音声がアップロードされる前にユーザーごとまたはチャットごとの語彙リストを注入する、呼び出し元のロケールから `language` を強制する、長い録音では `model` をダウングレードする、ノイズの多いソースを別のモデルにルーティングする。

```python
VOCAB = "Hermes, Teknium, Nous Research, kanban"

def add_vocab(provider, prompt, source, **kwargs):
    if source != "gateway":
        return None
    return {"prompt": f"{prompt}. {VOCAB}" if prompt else VOCAB}

def register(ctx):
    ctx.register_hook("pre_transcription", add_vocab)
```

すべてのバックエンドがプロンプトを受け付けるわけではありません。`local` はfaster-whisperの `initial_prompt` にマッピングし、`openai`、`groq`、`mistral`、`deepinfra` は `prompt` として送信し、`xai`、`elevenlabs`、`local_command`、`type: command` のプロバイダーはDEBUGでログに記録してプロンプトなしで文字起こしします。完全な対応表とプライバシーの境界については、[プロバイダーの対応表](../configuration.md#transcription-prompt-vocabulary-hints)を参照してください。フックの配管のエラーはフェイルオープンです: ディスパッチは変更されていないリクエストで続行します。

---

### `transform_tool_result`

ツールが戻った**後**、結果が会話に追加される**前**に発火します。プラグインは、ターミナル出力だけでなく、あらゆるツールの結果文字列を、モデルが見る前に書き換えることができます。

**コールバックのシグネチャ:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str, **kwargs) -> str | None:
```

完全なペイロードには `session_id`、`conversation_id`、`tool_call_id`、`turn_id`、`api_request_id`、`duration_ms`、`status`、`error_type`、`error_message` も含まれます。`result` はツールのディスパッチが返した最終的な結果です。これと `args` には任意のユーザー/ツールのコンテンツやシークレットが含まれる可能性があります。

**戻り値:** 最初の `str` が結果を置き換えます（空文字列を含む）。`None` は変更しません。

**ユースケース:** `web_extract` の出力から組織固有の個人情報をリダクトする、長いJSONのツール応答を要約ヘッダーで包む、`read_file` の結果に検索拡張のヒントを注入する、`delegate_task` のサブエージェントのレポートをプロジェクト固有のスキーマに書き換える。

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

すべてのツールに適用されます。ターミナルのみの書き換えについては下記の `transform_terminal_output` を参照してください — そちらはより範囲が狭く、`transform_tool_result` の前に実行され、その置換結果は引き続きターミナルツールの最終的な出力制限の対象となります。

---

### `transform_terminal_output`

`terminal` ツール内で、フォアグラウンドプロセスのキャプチャが環境によってすでにサイズ制限された後、最終的な出力制限の前に発火します。プラグインはキャプチャされたstdout/stderrを置き換えることができます。置換結果は引き続き最終的な出力制限の対象です。

バックグラウンドプロセスの出力がモデルやチャットに向かう途中でも発火します: `process_manage` の `poll` / `wait` / `log` / `kill` の結果（および `list` のプレビュー）と、完了、ハートビート、ウォッチパターンの通知です。そこでは、プロセスがまだ実行中の間は `returncode` が `None` で、`env_type` は空文字列です（環境はプロセスごとには記録されません）。どちらの場合も、フックはシークレットのリダクトの*前*に実行されるため、まだ認証情報を含む置換結果はマスクされます。

**コールバックのシグネチャ:**

```python
def my_callback(
    command: str,
    output: str,
    returncode: int,
    task_id: str,
    env_type: str,
    **kwargs,
) -> str | None:
```

| パラメーター | 型 | 説明 |
|-----------|------|-------------|
| `command` | `str` | 出力を生成したシェルコマンド。 |
| `output` | `str` | サイズ制限付きのプロセスキャプチャ後の結合されたstdout/stderr。 |
| `returncode` | `int` | プロセスの終了コード。 |
| `task_id` | `str` | 実効的なタスク識別子、または空文字列。 |
| `env_type` | `str` | 実行環境の種類。 |

**戻り値:** 最初の `str` が出力を置き換えます。`None` は変更しません。コマンドと出力には認証情報やその他の機密データが含まれる可能性があります。

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

`terminal` を含むすべてのツールに対してその後に実行される `transform_tool_result` と組み合わせて使えます。

---

### `transform_llm_output`

ツール呼び出しループが完了し、モデルが最終応答を生成した後、その応答がユーザー（CLI、Gateway、またはプログラム的な呼び出し元）に届けられる**前**、かつアシスタントの行が永続化される**前**に、**ターンごとに1回**発火します — 置換結果がセッションに保存され、`/resume` で表示され、次のターンで再生されるものになるため、トランスクリプトがユーザーの見たものと食い違うことはありません。Hermes自身の付記（ファイル変更の警告、異常終了の注記）はその後に追加され、`response_text` には含まれません。プラグインは古典的なプログラミング手法でアシスタントの最終テキストを書き換えることができます — SOULのフレーバーテキストやスキル駆動の変換に追加の推論トークンを消費しません。

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
| `session_id` | `str` | この会話のセッションID（ワンショット実行では空のことがあります）。 |
| `model` | `str` | 応答を生成したモデル名（例: `anthropic/claude-sonnet-4.6`）。 |
| `platform` | `str` | 配信プラットフォーム（`cli`、`telegram`、`discord` など。未設定の場合は空）。 |

**戻り値:** 応答テキストを置き換えるには空でない `str` を、変更しない場合は `None` または空文字列を返します。複数のプラグインが登録した場合は**最初の空でない文字列が勝ちます**。ツールやターミナルの変換とは異なり、空文字列は置換として受け付けられません。

**ユースケース:** パーソナリティ/語彙の変換（海賊口調、スポンジ・ボブ）を適用する、最終テキストからユーザー固有の識別子をリダクトする、プロジェクト固有の署名フッターを追加する、SOULの指示にトークンを消費せずに社内のスタイルガイドを強制する。

CLIのストリーミングが有効な場合、追記のみの変換はストリーミングされた本文の後に
出力されます。応答を置き換える変換は、ストリーミング後の変換であることを示すラベル付きで、
ストリーミングされた本文の後に全文が出力されるため、置換内容が黙って失われることは
ありません。TUI、デスクトップアプリ、およびストリーミングされたメッセージをその場で編集する
メッセージングのチャットでは、ターンが完了したときにストリーミングされた返信が書き換えで
置き換えられます。フックは完成したテキストに対して実行されるため、ストリーミング中は
書き換え前の返信が見えます。

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # pass through unchanged
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

このフックは空でなく中断されていない応答でガードされています — 停止ボタンによる中断や空のターンでは発火しません。例外は警告としてログに記録され、エージェントの実行を壊すことはありません。

#### 書き換えられるまで返信を保留する（`hold_stream=True`） {#hold-stream}

フックは完成した返信に対して実行されるため、ストリーミングが有効だと、ユーザーは書き換え前のテキストが届くのを見てから、それが置き換えられるのを見ることになります。ユーザーより先に返信を見なければならないプラグイン（リダクト、ポリシーフィルター）は、登録時にそれを宣言します:

```python
def register(ctx):
    ctx.register_hook("transform_llm_output", redact, hold_stream=True)
```

あるプロファイルで読み込まれたプラグインがストリームを保留している間、そのプロファイルのセッションでは返信は書き換えられた最終版としてのみ表示されます:

| サーフェス | 返信が保留されている間 | ターンが終わったとき |
|---|---|---|
| メッセージング（Telegram、Discord、Slack、Matrixなど） | プラットフォームの入力中インジケーター。下書き、編集、ネイティブのストリームメッセージはなし | 書き換えが通常のメッセージとして1回送信されます（長いものは通常の返信と同様に分割されます） |
| Relay経由のSlack | 入力中インジケーター。`chat.*Stream` メッセージはなし | 書き換えが1回送信されます |
| TUI、デスクトップアプリ、クラシックCLI、ACPエディター | 通常のビジー/思考中インジケーター。返信テキストはなし | 書き換えが1回表示されます |
| APIサーバー（`/v1/chat/completions`、`/v1/responses`、ストリーミング） | 応答ストリームはテキストなしで開いたまま | 書き換えが応答の唯一のテキストになります |

その他に保留されるもの:

- **コメンタリー**（ツール呼び出しの間に送られるモデルのテキスト、`display.interim_assistant_messages`）は `transform_llm_output` を通らないため、保留されたターンではそれは破棄されます。モデルのコンテキストには残りますが非表示で保存されるため、セッションを再開したり開き直したりしても（TUI、デスクトップアプリ、ダッシュボード、クラシックCLI）表示されません。ユーザーに届かなかったその他のすべてのモデルのテキストも同様です: Hermesがやり直しのために差し戻した回答（遅れて届いた言語サーバーのエラー、verify-on-stop）、継続の前に出力制限で切られた返信、未知のツールの呼び出し。
- **ターンが終わるすべての経路がフックを通ります。** 毎回の継続で出力制限により切られた返信、イテレーション予算の要約、プロバイダーや圧縮のエラー、中断: ターンがどんなテキストで終わっても、それは `transform_llm_output` が見た後にのみユーザーに届き、モデルのテキストは書き換えとして保存されます。
- **失敗したフックは返信を差し止めます。** どのフックも返信を書き換えず、いずれかが例外を送出またはタイムアウトした場合、ユーザーには元の返信ではなく短い「返信が差し止められました」という通知が届きます。
- **フックの集合はターンの間は固定です。** 保留されたターンは、途中でプラグインが再検出されても、開始時に読み込まれていた `transform_llm_output` フックを実行します。
- **ツールの進捗と推論**は返信ではなく、独自の設定（`display.tool_progress`、`display.show_reasoning`）を保持します。プラグインが見ていないモデルの出力がユーザーに届いてはならない場合は、これらもオフにしてください。

`hold_stream` は `transform_llm_output` にのみ適用されます。その他のフックでは警告とともに無視されます。`hold_stream` なしで `transform_llm_output` を登録したプラグインは、通常のストリーミングを維持します。

### APIリクエストのオブザーバーフック

#### `pre_api_request`

プロバイダーへの各試行について、送信の直前に発火します。これはオブザーバー専用です。レガシーの `user_message`、`conversation_history`、`request_messages` フィールドは生のままで、互換性のために意図的にサニタイズされていません。新しい利用者はサニタイズ済みの `request` エンベロープを優先してください。

#### `post_api_request`

プロバイダーの応答が正常に正規化された後に発火します。これはオブザーバー専用です。サニタイズ済みの `response` を優先してください。`assistant_message` は生の正規化済みメッセージで、`usage` には課金計測データが含まれます。

#### `api_request_error`

失敗したプロバイダーへの試行について、ステータス/リトライのタイミング、`error` オブジェクト、サニタイズ済みの `request` とともに発火します。これはオブザーバー専用です。エラーメッセージにはプロバイダーやユーザーのデータが含まれている可能性があります。

### 補助呼び出しのオブザーバーフック

#### `pre_auxiliary_call` / `post_auxiliary_call`

補助LLM呼び出し — セッションのタイトル付け、コンテキスト圧縮、MoAのアドバイザーとアグリゲーター、ビジョン、承認の分類、メモリ、その他のサイドタスク — はメインのツール呼び出しループの外で実行され、`pre_api_request` / `post_api_request` を発火**しません**（これらはターンのスコープにとどまるため、ターンごとにトレースするプラグインがサイドのトラフィックを誤って目にすることはありません）。代わりに `pre_auxiliary_call` / `post_auxiliary_call` をサブスクライブしてください: これらは物理的なプロバイダーへの試行ごと（リトライとフォールバックを含む）に1回、同じ形式のペイロードに `aux_task` を加えて発火します（タスク名。例: `title_generation`、`compression`、`moa_aggregator`、`vision`、プラグインの [`ctx.side_question`](/docs/developer-guide/plugin-llm-access#asking-the-sessions-own-agent-ctxside_question) では `plugin_side_question`、プラグインの [`ctx.fork_session`](/docs/developer-guide/plugin-llm-access#forking-the-sessions-own-agent-ctxfork_session) の各リクエストでは `plugin_fork`。後者2つはプラグインの `plugin_id` も持ちます）。`session_id` / `task_id` / `turn_id` は、呼び出しが親ターンのもとで実行される場合はそのターンのもの、それ以外は空です。`api_request_id`（`aux-…`）は1つの論理的な呼び出しのすべての試行で共有され、`retry_count` でそれらを区別します。どちらもオブザーバー専用でフェイルオープンです: 例外を送出したりタイムアウトしたりしたコールバックはログに記録され、補助タスクは続行します。`post_auxiliary_call` は、試行が例外を送出した場合は `error` / `error_type` を持ち、応答がストリームとして返される場合は `streaming: True`（`usage`/`response` は `None`）を持ちます。

### `on_skill_lifecycle`

権威あるスキル使用状態の変更の後に発火します。これはオブザーバー専用で、ローカルの `skill_name`、出所、相関ID、使用回数、再利用フラグを公開します。

### Kanbanライフサイクルのオブザーバー

#### `kanban_task_claimed`

ディスパッチャープロセスでクレームがコミットされた後、ワーカーの起動の直前に発火します。

#### `kanban_task_completed`

完了とクリーンアップの後、通常はワーカープロセス内で発火します。その `summary` にはプロジェクトやユーザーのコンテンツが含まれる可能性があります。

#### `kanban_task_blocked`

blockedへの遷移がコミットされた後（依存待ちを含む）に発火するため、コールバックはボードを読み書きできます。その `reason` にはプロジェクトやユーザーのコンテンツが含まれる可能性があります。

3つのKanbanフックはすべてオブザーバー専用で、`task_id`、`profile_name`、`board`、`assignee`、`run_id` を持ちます。completedは `summary` を、blockedは `reason` を追加します。

### Kanbanのワーカーライフサイクル、タスク変更、ディスパッチのオブザーバー

5つの追加のオブザーバー（RFC #58548）がKanbanファミリーを拡張します。すべてオブザーバー専用で、該当するトランザクションのコミット後に発火し、`has_hook` で短絡します — サブスクライバーがいない場合、ディスパッチの挙動は変わりません。タスクスコープのフックは、上記のフックと同じ共通フィールドを持ちます。

- **`on_kanban_worker_spawned`** — `spawn_fn` が戻り、ワーカーのPIDが永続化された後。`worker_pid`（`None` のことがあります）と `workspace_path` を追加します。ディスパッチロック内で実行されるので、コールバックは高速に保ってください。
- **`on_kanban_worker_exited`** — ティックから導出され、`detect_crashed_workers` がPIDの死んだタスクを回収したとき。`worker_pid`、`exit_kind`、`exit_code`、`outcome`、`retry_status` を追加します。
- **`on_kanban_worker_stale_claim`** — TTL切れのクレームが回収されたとき。PIDが生きている場合の延長では発火しません。`worker_pid`、`heartbeat_stale`、`retry_status` を追加します。
- **`on_kanban_task_updated`** — クレーム/完了/ブロックのライフサイクル外で、タスクのフィールドへの書き込みがコミットされた後（`assign_task`、モデル/推論のオーバーライド、ダッシュボードのエディター）。`changed_fields` を追加します — フィールド名のみで、値は含みません。
- **`on_kanban_dispatch_tick`** — ディスパッチャーのティックごとに1回、ディスパッチロックが解放された厳密に後。アイドルやロック競合のティックも含みます。ペイロード: `board`、`profile_name`、`dry_run`、`outcome`、`result`。

---

## シェルフック {#shell-hooks}

プロファイルの `config.yaml` でシェルスクリプトのフックを宣言すると、対応するプラグインフックのイベントが発火するたびに、Hermesがそれらをサブプロセスとして実行します — CLI、Gateway、Desktop、TUI、ダッシュボードのチャットセッションで動作します。Pythonプラグインを書く必要はありません。

Desktop、TUI、ダッシュボードのチャットは、エージェントを構築するときに、そのセッションのプロファイルの設定と同意の許可リストを使ってフックを登録します。プロファイルを切り替えても、別のプロファイルのフックが再利用されることはありません。既存のフックの同意要件とセーフモードの挙動は引き続き適用され、未承認のフックは黙って承認されるのではなくスキップされます。

次のようなことを、ドロップインの単一ファイルのスクリプト（Bash、Python、shebangを持つものなら何でも）で行いたい場合にシェルフックを使ってください:

- **ツール呼び出しをブロックまたは変更する** — 危険な `terminal` コマンドを拒否する、ディレクトリごとのポリシーを強制する、破壊的な `write_file` / `patch` 操作に承認を必須にする、またはツールの実行前に引数を書き換える（パスのサニタイズ、デフォルトの注入）。
- **ツール呼び出しの後に実行する** — エージェントが書いたばかりのPythonやTypeScriptのファイルを自動フォーマットする、API呼び出しをログに記録する、CIワークフローをトリガーする。
- **次のLLMのターンにコンテキストを注入する** — `git status` の出力、現在の曜日、検索したドキュメントをユーザーメッセージの先頭に付ける（[`pre_llm_call`](#pre_llm_call) を参照）。
- **ライフサイクルイベントを観察する** — サブエージェントが完了したとき（`subagent_stop`）やセッションが開始したとき（`on_session_start`）にログ行を書く。

シェルフックは、CLIの起動時（`hermes_cli/main.py`）とGatewayの起動時（`gateway/run.py`）の両方で `agent.shell_hooks.register_from_config(cfg)` を呼び出すことで登録されます。Pythonのプラグインフックと自然に組み合わせられます — どちらも同じディスパッチャーを通ります。

### 一覧での比較

| 観点 | シェルフック | [プラグインフック](#plugin-hooks) | [Gatewayフック](#gateway-event-hooks) |
|-----------|-------------|-------------------------------|---------------------------------------|
| 宣言場所 | `~/.hermes/config.yaml` 内の `hooks:` ブロック | `plugin.yaml` プラグイン内の `register()` | `HOOK.yaml` + `handler.py` のディレクトリ |
| 配置場所 | `~/.hermes/agent-hooks/`（慣例） | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| 言語 | 任意（Bash、Python、Goバイナリなど） | Pythonのみ | Pythonのみ |
| 実行場所 | CLI + Gateway | CLI + Gateway | Gatewayのみ |
| イベント | `VALID_HOOKS`（`subagent_stop` を含む） | `VALID_HOOKS` | Gatewayのライフサイクル（`gateway:startup`、`agent:*`、`command:*`） |
| ツール呼び出しをブロックできるか | はい（`pre_tool_call`） | はい（`pre_tool_call`） | いいえ |
| LLMのコンテキストを注入できるか | はい（`pre_llm_call`） | はい（`pre_llm_call`） | いいえ |
| 同意 | `(event, command)` のペアごとに初回使用時のプロンプト | 明示的（`plugins.enabled`）、その後はプロセス内で信頼 | 暗黙的（[ディレクトリの信頼](#gateway-hook-trust)） |
| プロセス間の分離 | あり（サブプロセス） | なし（プロセス内） | なし（プロセス内） |

### 設定スキーマ

```yaml
hooks:
  <event_name>:                  # Must be in VALID_HOOKS
    - matcher: "<regex>"         # Optional; used for pre/post_tool_call only
      command: "<shell command>" # Required; runs via shlex.split, shell=False
      timeout: <seconds>         # Optional; default 60, capped at 300
      fail_closed: <bool>        # Optional; default false. pre_tool_call only.
                                 # `failClosed` also accepted (Cursor/Claude Code compat)

hooks_auto_accept: false         # See "Consent model" below
```

イベント名は[プラグインフックのイベント](#plugin-hooks)のいずれかでなければなりません。タイプミスは「Did you mean X?」という警告を出してスキップされます。単一のエントリ内の未知のキーは無視され、`command` が欠けている場合は警告付きでスキップされます。`timeout > 300` は警告とともに上限に丸められます。`pre_tool_call` 以外のイベントでの `fail_closed: true` は警告が出て無視されます（ブロック可能なイベントだけがフェイルクローズできます）。

Windowsでは、既存のスクリプトファイルで始まる `command` — 以下の例で使っている `~/.hermes/agent-hooks/x.sh` の形 — は、そのファイル自身のインタープリター（`.sh`/`.bash` にはGit Bash、`.py` には実行中のHermesのPython）を通じて起動されます。これは `CreateProcess` がshebangをサポートせず、素のスクリプトを `WinError 193` で拒否するためです。それ以外のすべてのコマンド、およびすべてのPOSIXプラットフォームでは、`argv` がそのまま `Popen` に渡され、カーネルがすでにshebangを尊重します。

### JSONワイヤープロトコル

イベントが発火するたびに、Hermesは一致する各フック（マッチャーが許す場合）に対してサブプロセスを起動し、JSONのペイロードを **stdin** にパイプし、**stdout** をJSONとして読み戻します。

**stdin — スクリプトが受け取るペイロード:**

```json
{
  "hook_event_name": "pre_tool_call",
  "tool_name":       "terminal",
  "tool_input":      {"command": "rm -rf /"},
  "session_id":      "sess_abc123",
  "cwd":             "/home/user/project",
  "profile":         "default",
  "extra":           {"task_id": "...", "tool_call_id": "..."}
}
```

`profile` はフックを発火させたHermesプロファイルの名前で（プロファイル外では `"default"`）、
1つのスクリプトで多重化されたGatewayの背後にあるすべてのプロファイルに対応できます。サブプロセスも
そのプロファイルの `HERMES_HOME` で実行されます。ツール以外のイベント（`pre_llm_call`、`subagent_stop`、セッションのライフサイクル）では、`tool_name` と `tool_input` は `null` です。`extra` dictはイベント固有のすべてのkwargs（`user_message`、`conversation_history`、`child_role`、`duration_ms` など）を持ちます。シリアライズできない値は省略されるのではなく文字列化されます。

`extra.kanban_task_id` は、フックを発火させた実行が所有するボードのタスクを
識別します。通常のセッション、委譲された子、プロセス内のcron実行では、
そのプロセスがワーカーの環境を継承していても空です。`extra.task_id` は
従来どおりターン/セッションの意味を保ちます。ワークスペースのパスから推測する代わりに、
ポリシーフックではボードの識別子を使ってください。フックのサブプロセスでは
引き続きワーカーの識別環境が取り除かれます。ペイロードのメタデータがワーカーの権限を与えることはありません。
同じフィールドは、共有のアウトバウンドWebhookのペイロードビルダーにも含まれます。

**stdout — 任意の応答:**

```jsonc
// Block a pre_tool_call (both shapes accepted; normalised internally):
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code style
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes-canonical

// Modify a pre_tool_call — rewrite tool args before dispatch:
{"action": "modify", "args": {"new_string": "fixed content"}}         // Hermes-canonical
{"decision": "modify", "tool_input": {"new_string": "fixed content"}} // Claude-Code style

// Escalate a pre_tool_call to the human-approval gate (Hermes-only; `message` and `rule_key`
// are optional). Claude-Code's `{"decision": "approve"}` means auto-allow and is NOT mapped here:
{"action": "approve", "message": "Why approval is required", "rule_key": "optional:scope"}

// Inject context for pre_llm_call:
{"context": "Today is Friday, 2026-04-17"}

// Block the turn before any inference request (pre_llm_call); both shapes accepted:
{"action": "block", "message": "Blocked by policy: prompt contains a customer ID"}
{"decision": "block", "reason": "Blocked by policy: prompt contains a customer ID"}

// Keep the agent going at the verify gate (pre_verify); both shapes accepted:
{"action": "continue", "message": "Run the formatter, then finish."}
{"decision": "block",  "reason":  "Run the formatter, then finish."}

// Silent no-op — any empty / non-matching output is fine:
```

下記の `pre_tool_call` の終了コード2を除き、不正な形式のJSON、通常のゼロ以外の終了コード、タイムアウトはデフォルトでフェイルオープンです: 警告をログに記録しますが、エージェントループを中止しません。`fail_closed` のフックは、下記のように挙動を変えます。

### 終了コード2 = ブロック（Claude Code / Cursor互換）

終了コード **2** で終了する `pre_tool_call` フックは、stdoutにブロックのJSONがなくてもツール呼び出しをブロックします。ブロックメッセージは次の優先順位で決まります:

1. stdoutのブロックJSON（`reason` / `message`）がある場合はそれ
2. stderrの先頭400文字
3. 汎用のデフォルト `"Blocked by shell hook."`

したがって、最もシンプルなブロッキングフックは次のとおりです:

```bash
#!/usr/bin/env bash
echo "policy violation: rm -rf is not permitted" >&2
exit 2
```

ブロックのディレクティブが尊重されないイベント（`pre_tool_call` 以外のすべて）では、終了コード2は他のゼロ以外の終了と同様に扱われます: 警告がログに記録され、stdoutは引き続き解析されます。

### フェイルオープン vs フェイルクローズ

デフォルトでは、シェルフックは**フェイルオープン**します: 起動エラー、タイムアウト、解析できないstdoutは警告をログに記録し、アクションは続行します。これは可観測性のフックには正しいデフォルトですが、セキュリティゲートには誤りです。クラッシュしたシークレットスキャナーが、審査するはずだったツール呼び出しを黙って許可してはなりません。

これを反転させるには、`pre_tool_call` のエントリに `fail_closed: true`（またはCursor/Claude Codeの表記である `failClosed: true`）を設定します:

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal|write_file|patch"
      command: "~/.hermes/agent-hooks/secret-scan.sh"
      timeout: 10
      fail_closed: true
```

`fail_closed: true` では、以下のそれぞれが `hook <command> failed closed: <reason>` とともにツール呼び出しを**ブロック**するようになります:

| 失敗 | フェイルオープン（デフォルト） | `fail_closed: true` |
|---------|--------------------|--------------------|
| コマンドが見つからない / 実行できない | 警告して続行 | **ブロック** |
| タイムアウト | 警告して続行 | **ブロック** |
| 認識されるディレクティブなしのゼロ以外の終了 | 警告して続行 | **ブロック** |
| JSONでないstdout（例: スタックトレース） | 警告して続行 | **ブロック** |
| 正常終了、有効な何もしないJSON（`{}`） | 続行 | 続行 |

`fail_closed` はブロック可能なイベント（現在は `pre_tool_call`）にのみ適用されます。その他のイベントに設定すると、設定の解析時に警告がログに記録されて無視されます。`hermes hooks test` はこれらのセマンティクスを反映します — `parsed` の行には、ディスパッチャーが受け取るブロックの形がそのまま表示されます。

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

エージェントのコンテキスト内でのファイルの見え方は自動的には再読み込み**されません** — 再フォーマットはディスク上のファイルにのみ影響します。以降の `read_file` 呼び出しでフォーマット後のバージョンが取得されます。

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
cat - >/dev/null   # discard stdin payload
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("Uncommitted changes in cwd:\n" + $s)}'
else
  printf '{}\n'
fi
```

Claude Codeの `UserPromptSubmit` イベントは、意図的にHermesの独立したイベントにはしていません — `pre_llm_call` が同じ場所で発火し、すでにコンテキスト注入をサポートしています。ここではそれを使ってください。

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

一意の `(event, command)` のペアごとに、Hermesが初めてそれを見たときにユーザーに承認を求め、その決定を `~/.hermes/shell-hooks-allowlist.json` に永続化します。以降の実行（CLIまたはGateway）ではプロンプトがスキップされます。クラシックCLIはGatewayセッションを開く前にターミナルで確認します。Gatewayはローカルセッション（CLI、TUI、Desktop）が作成または再開されるたびに `hooks:` と許可リストを再読み込みするため、Gatewayの起動後に追加、承認、削除、取り消されたフックはその時点で反映されます。プロファイルの許可リストや `hooks:` ブロックへの変更（承認、`hermes hooks revoke`、編集）は、セッションが開かれなくても、Gatewayが提供するすべてのプロファイルで約30秒以内に反映されます。

対話的なプロンプトを回避する手段が3つあります — どれか1つで十分です:

1. CLIの `--accept-hooks` フラグ（例: `hermes --accept-hooks chat`）
2. `HERMES_ACCEPT_HOOKS=1` 環境変数
3. `~/.hermes/config.yaml` の `hooks_auto_accept: true`

確認を求めるのはフォアグラウンドの `hermes chat`（または `hermes rl`）だけです。Gateway、cron、`hermes mcp-serve --tools`（mcp-serveプラグイン）、`hermes acp`、CIは、stdinがターミナルのように見える場合でも（Windowsのログイン項目ランチャーからの非表示コンソールはそう見えます）決してプロンプトを出さないため、これら3つのいずれかか許可リストのエントリが必要です。そうでなければ、新しく追加されたフックは未登録のままとなり、そのフックが欠けているプロファイルの許可リストファイル（デフォルト以外のプロファイルでは `profiles/<name>/shell-hooks-allowlist.json`）を示す警告が出ます。`hermes gateway run --accept-hooks`（Gatewayでの `HERMES_ACCEPT_HOOKS=1` と同様）は、そのGatewayが提供するすべてのプロファイルのフックを、起動後に追加されたフックも含めて承認します。

**スクリプトの編集は黙って信頼されます。** 許可リストはスクリプトのハッシュではなく正確なコマンド文字列をキーにしているため、ディスク上のスクリプトを編集しても同意は無効になりません。`hermes hooks doctor` はmtimeのずれを検出するので、編集に気づいて再承認するかどうかを判断できます。

#### 手動での許可リスト登録

手動での許可リスト登録は、オペレーターが初回使用時のプロンプトに対話的に回答できない非TTYやサービスアカウントのデプロイで便利です。許可リストのファイルは `~/.hermes/shell-hooks-allowlist.json` で、期待される形式は `approvals` 配列です。各承認はフックの `event` と正確な `command` 文字列を記録します:

```json
{
  "approvals": [
    {
      "event": "post_llm_call",
      "command": "/home/hermes/.hermes/hooks/my-hook.py"
    }
  ]
}
```

コマンド文字列は設定されたフックのコマンドと完全に一致しなければなりません。`sha256` フィールドを持つパスをキーとしたオブジェクトは期待される形式ではなく、フックを承認しません。手動のエントリは `hermes hooks list` で確認してください。

### `hermes hooks` CLI

| コマンド | 機能 |
|---------|--------------|
| `hermes hooks list` | 設定済みのフックを、マッチャー、タイムアウト、同意の状態とともに出力する |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | 一致するすべてのフックを合成ペイロードに対して発火させ、解析された応答を表示する |
| `hermes hooks revoke <command>` | `<command>` に一致するすべての許可リストのエントリを削除する（実行中のGatewayは約30秒以内にその実行を停止する） |
| `hermes hooks doctor` | 設定済みの各フックについて、実行ビット、許可リストの状態、mtimeのずれ、JSON出力の妥当性、おおよその実行時間を確認する |

### セキュリティ

シェルフックは**あなたのユーザーの完全な認証情報**で実行されます — cronのエントリやシェルのエイリアスと同じ信頼境界です。`config.yaml` の `hooks:` ブロックは特権的な設定として扱ってください:

- 自分で書いたか、完全にレビューしたスクリプトのみを参照する。
- パスを監査しやすいように、スクリプトは `~/.hermes/agent-hooks/` 内に置く。
- 共有の設定をpullした後は `hermes hooks doctor` を再実行し、新しく追加されたフックが登録される前に見つける。
- config.yamlをチームでバージョン管理している場合は、`hooks:` セクションを変更するPRを、CIの設定をレビューするのと同じようにレビューする。

### 順序と優先順位

Pythonのプラグインフックとシェルフックは、どちらも同じ `invoke_hook()` ディスパッチャーを通ります。Pythonプラグインが先に登録され（`discover_and_load()`）、シェルフックが2番目に登録される（`register_from_config()`）ため、同順位の場合はPythonの `pre_tool_call` の決定が優先されます。最初の有効なブロックが勝ちます — アグリゲーターは、いずれかのコールバックが空でないメッセージを持つ `{"action": "block", "message": str}` を生成するとすぐに戻ります — そして、リスト内のどこにあるブロックも、それより前に返された `approve` より優先されます。

## アウトバウンドWebhook {#outbound-webhooks}

アウトバウンドWebhookは、[インバウンドWebhookプラットフォーム](../messaging/webhooks.md)のプッシュ側の鏡像です: インバウンドWebhookは世界が変化したときにHermesを起こし、アウトバウンドWebhookはHermesが何かをしたときに世界に伝えます。HTTPエンドポイントのリストと、それぞれが関心を持つライフサイクルイベントを設定すると、一致するイベントが発火するたびにHermesが署名付きのJSONペイロードを各エンドポイントにPOSTします — 受信側でのポーリングは不要です。

典型的な用途:

- エージェントのターンが終了したときにCIシステムやダッシュボードに通知する（`on_session_end`）
- フリート全体でサブエージェントの完了を追跡する（`subagent_stop`）
- ツールのアクティビティを外部の監視に流す（`matcher` 付きの `post_tool_call`）
- *別の* Hermesインスタンスを起こす: URLをそのインスタンスのインバウンドWebhookに向ける

### 設定

`~/.hermes/config.yaml` に `hooks.outbound:` リストを追加します:

```yaml
hooks:
  outbound:
    - name: ci-notify                       # optional label for logs
      url: https://ci.example.com/hermes-events
      events: [on_session_end, subagent_stop]
      secret_env: HERMES_OUTBOUND_WEBHOOK_SECRET   # env var holding the HMAC secret
      timeout: 10                           # per-attempt seconds (1–60)

    - name: tool-monitor
      url: https://metrics.example.com/hooks/hermes
      events: [post_tool_call]
      matcher: "terminal|delegate_task"     # regex, tool-scoped events only
```

プラグインフックの集合に含まれるイベントであれば何でも有効です（`pre_tool_call`、`post_tool_call`、`pre_llm_call`、`post_llm_call`、`on_session_start`、`on_session_end`、`subagent_start`、`subagent_stop` など）。不正な形式のエントリは警告が出てスキップされます — 壊れたWebhookがエージェントをクラッシュさせることはありません。変更は次のCLIセッション / Gatewayの再起動で反映されます。

シークレット: 設定ファイルに認証情報を含めないよう、インラインの `secret:` リテラルよりも `secret_env`（環境変数の名前。通常は `~/.hermes/.env` で設定）を優先してください。シークレットのないエントリは署名なしで配信されます（`hermes hooks list` で `UNSIGNED` としてフラグ付けされます）。

### ワイヤーフォーマット

各発火では、シェルフックのstdinと同じトップレベルの形に配信メタデータを加えたJSON本文がPOSTされます。`profile` はイベントを送出したHermesプロファイルの名前で（プロファイル外では `"default"`）、多重化されたGatewayの背後にある受信側がプロファイルを区別できます:

```json
{
  "hook_event_name": "on_session_end",
  "profile": "default",
  "tool_name": null,
  "tool_input": null,
  "session_id": "sess_abc123",
  "cwd": "/home/user/project",
  "extra": {"completed": true, "interrupted": false, "model": "...", "platform": "cli"},
  "delivery_id": "3f2c9a...",
  "timestamp": "2026-07-22T14:00:00Z"
}
```

ヘッダー:

| ヘッダー | 値 |
|--------|-------|
| `Content-Type` | `application/json` |
| `X-Hermes-Event` | フックのイベント名 |
| `X-Hermes-Delivery` | 配信ごとの一意なID — 本文の `delivery_id` と同じ値 |
| `X-Hermes-Signature-256` | `sha256=<hex>` — 生の本文のHMAC-SHA256（GitHub方式）。シークレットが設定されている場合のみ存在 |

署名はGitHubのWebhookとまったく同じように検証します:

```python
import hashlib, hmac

def verify(body: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
```

`delivery_id` と `timestamp` は**署名された本文の中**にあるため、検証を行う受信側はリプレイ保護も無料で得られます:

- `delivery_id`（または対応する `X-Hermes-Delivery` ヘッダー）で**重複排除**する — 最近見たIDを記憶し、重複をスキップします。Hermesは失敗した配信を1回再試行するため、同じIDが正当に2回届くことがあります。
- 許容範囲（5分が一般的なデフォルト）を設けて `timestamp` を自分の時計と比較し、**古いイベントを拒否**する。キャプチャしたリクエストをリプレイする攻撃者は、シークレットなしに新しいタイムスタンプを偽造できません。

### 配信のセマンティクス

- **ファイア・アンド・フォーゲット、ホットパスの外。** イベントは即座にシリアライズされてキューに入れられ、単一のバックグラウンドスレッドがHTTPのPOSTを実行します。遅いまたは停止したエンドポイントが、ツール呼び出しやエージェントのターンを止めることは決してありません。
- **通知のみ。** シェルフックとは異なり、アウトバウンドWebhookはツール呼び出しをブロックしたりコンテキストを注入したりできません — 応答本文は無視されます。観察するだけで、操作はしません。
- **回数制限付きのリトライ。** 接続エラーと5xx応答はバックオフ付きで1回リトライされます。4xx応答はリトライされません（受信側がリクエスト自体が誤っていると言っているため）。失敗はログに記録されて破棄されます — 配信はベストエフォートであり、保証されません。
- **リダイレクトは決して追従されません。** 3xx応答は設定ミスとして扱われ、ログに記録されます — リダイレクトされたPOSTに追従すると、署名付きのペイロードが黙って失われてしまいます。`url` は最終的なエンドポイントに向けてください。
- **サイズ制限付きのキュー。** キューが詰まった場合（停止したエンドポイント、イベントの嵐）、無制限にメモリを消費するのではなく、新しいイベントが警告とともに破棄されます。
- **同意のプロンプトなし。** アウトバウンドのターゲットはあなたのマシン上でコードを実行しません — あなたが設定したURLでデータを受け取るだけです。`HERMES_SAFE_MODE=1` は、プラグインやシェルフックと同様に、引き続き登録をスキップします。ペイロードにはツールの入力とイベントのメタデータが含まれるため、ターゲットは信頼できるエンドポイントにのみ向け、`https://` を優先してください。

`hermes hooks list` は、設定済みのアウトバウンドのターゲットをシェルフックと並べて、各ターゲットが署名されているかどうかを含めて表示します。
