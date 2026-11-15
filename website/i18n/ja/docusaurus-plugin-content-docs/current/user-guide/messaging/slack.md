---
sidebar_position: 4
title: "Slack"
description: "Socket Modeを使ってHermes AgentをSlackボットとしてセットアップする"
---

# Slackのセットアップ

Socket Modeを使って、Hermes AgentをSlackにボットとして接続します。Socket Modeは公開HTTPエンドポイントの代わりにWebSocketを使用するため、Hermesインスタンスを外部からアクセス可能にする必要はありません — ファイアウォールの背後、ノートPC上、プライベートサーバー上でも動作します。

:::warning クラシックSlackアプリは非推奨
クラシックSlackアプリ（RTM APIを使用）は**2025年3月に完全に非推奨**となりました。Hermesは、Socket Modeを使ったモダンなBolt SDKを使用します。古いクラシックアプリをお持ちの場合は、以下の手順に従って新しいものを作成する必要があります。
:::

## 概要

| Component | Value |
|-----------|-------|
| **ライブラリ** | Python用 `slack-bolt` / `slack_sdk`（Socket Mode） |
| **接続** | WebSocket — 公開URL不要 |
| **必要な認証トークン** | Bot Token（`xoxb-`）+ App-Level Token（`xapp-`） |
| **ユーザー識別** | SlackメンバーID（例: `U01ABC2DEF3`） |

---

## ステップ1: Slackアプリを作成する

最も速い方法は、Hermesが生成するマニフェストを貼り付けることです。これは、すべての組み込みスラッシュコマンド（`/btw`、`/stop`、`/model` など）、必要なすべてのOAuthスコープ、すべてのイベントサブスクリプションを宣言し、Socket Modeを有効化します — すべて一度に行えます。

### オプションA: Hermesが生成したマニフェストから（推奨）

1. マニフェストを生成します。新しいSlackアプリはAgentビューを使用する必要があります:
   ```bash
   hermes slack manifest --agent-view --write
   ```
   これは `~/.hermes/slack-manifest.json` を書き出し、貼り付け用の手順を表示します。Slackのレガシーな
   Assistantビューをまだ使っている既存のアプリは、移行の準備ができるまで `--agent-view` を省略できます。

   既存のUTF-8テキストまたはMarkdownファイルからSlackのアプリの長い説明を設定するには、
   `--long-description-file` を追加します:

   ```bash
   hermes slack manifest --agent-view \
     --long-description-file AGENTS.md --write
   ```

   ファイルの内容は、Slackの175〜4,000文字の範囲内でそのまま保持されます。代わりにインラインのテキストを
   使うには `--long-description "..."` を使用してください。インラインとファイルのオプションは相互に排他的で、
   `--slashes-only` と組み合わせることはできません。
2. [https://api.slack.com/apps](https://api.slack.com/apps) →
   **Create New App** → **From an app manifest** にアクセス
3. ワークスペースを選び、JSONの内容を貼り付け、確認して **Next**
   → **Create** をクリック
4. **ステップ6: アプリをワークスペースにインストール** まで進んでください。マニフェストがスコープ、イベント、スラッシュコマンドを処理済みです。

### オプションB: 一から（手動）

1. [https://api.slack.com/apps](https://api.slack.com/apps) にアクセス
2. **Create New App** をクリック
3. **From scratch** を選択
4. アプリ名（例: "Hermes Agent"）を入力し、ワークスペースを選択
5. **Create App** をクリック

アプリの **Basic Information** ページに移動します。以下のステップ2〜6に進んでください。

---

## ステップ2: Bot Tokenのスコープを設定する

サイドバーで **Features → OAuth & Permissions** に移動します。**Scopes → Bot Token Scopes** までスクロールし、次を追加します:

| Scope | Purpose |
|-------|---------|
| `chat:write` | ボットとしてメッセージを送信 |
| `app_mentions:read` | チャンネルで@メンションされたことを検出 |
| `channels:history` | ボットが参加しているパブリックチャンネルのメッセージを読む |
| `channels:read` | パブリックチャンネルの一覧と情報を取得 |
| `groups:history` | ボットが招待されているプライベートチャンネルのメッセージを読む |
| `im:history` | ダイレクトメッセージの履歴を読む |
| `im:read` | DMの基本情報を表示 |
| `im:write` | DMを開いて管理 |
| `mpim:history` | グループダイレクトメッセージ（複数人DM）の履歴を読む |
| `mpim:read` | グループDMの基本情報を表示 |
| `users:read` | ユーザー情報を照会 |
| `files:read` | ボイスノート/音声を含む添付ファイルを読み取り・ダウンロード |
| `files:write` | ファイル（画像、音声、ドキュメント）をアップロード |

:::caution スコープがない = 機能がない
`channels:history` と `groups:history` がないと、ボットは**チャンネルでメッセージを受信しません** — DMでしか動作しません。`files:read` がないと、Hermesはチャットはできても、**ユーザーがアップロードした添付ファイルを確実に読み取れません**。これらは最もよく見落とされるスコープです。
:::

**任意のスコープ:**

| Scope | Purpose |
|-------|---------|
| `groups:read` | プライベートチャンネルの一覧と情報を取得 |
| `assistant:write` | メッセージの処理中に、ボット名の横に作業中のステータス行（"is thinking…"）を表示します。このスコープがないとステータスの呼び出し（slack-sdk 3.44以降では `agents.sessions.setStatus`、それより古いSDKでは `assistant.threads.setStatus`）は黙って失敗し、Slackは代わりに独自の汎用プレースホルダー（"Finding answers…"、"Reviewing findings…" など）を順に表示します — そのテキストをHermesが制御することはありません。`typing_status_text` が目に見える効果を持つために必要です。 |

---

## ステップ3: Socket Modeを有効にする

Socket Modeにより、ボットは公開URLを必要とせずにWebSocket経由で接続できます。

1. サイドバーで **Settings → Socket Mode** に移動
2. **Enable Socket Mode** をONに切り替え
3. **App-Level Token** の作成を求められます:
   - `hermes-socket` のような名前を付けます（名前は何でもかまいません）
   - **`connections:write`** スコープを追加
   - **Generate** をクリック
4. **トークンをコピー** します — `xapp-` で始まります。これがあなたの `SLACK_APP_TOKEN` です

:::tip
app-levelトークンは、いつでも **Settings → Basic Information → App-Level Tokens** で確認または再生成できます。
:::

---

## ステップ4: イベントをサブスクライブする

このステップは極めて重要です — ボットがどのメッセージを見られるかを制御します。


1. サイドバーで **Features → Event Subscriptions** に移動
2. **Enable Events** をONに切り替え
3. **Subscribe to bot events** を展開して、次を追加:

| Event | Required? | Purpose |
|-------|-----------|---------|
| `message.im` | **はい** | ボットがダイレクトメッセージを受信 |
| `message.mpim` | **はい** | ボットが追加された**グループDM**（複数人DM）のメッセージを受信 |
| `message.channels` | **はい** | ボットが追加された**パブリック**チャンネルのメッセージを受信 |
| `message.groups` | **推奨** | ボットが招待された**プライベート**チャンネルのメッセージを受信 |
| `app_mention` | **はい** | ボットが@メンションされたときのBolt SDKエラーを防止 |

4. ページ下部の **Save Changes** をクリック

:::danger イベントサブスクリプションの欠落はセットアップ問題の第1位
ボットがDMでは動作するのに**チャンネルでは動作しない**場合、ほぼ確実に `message.channels`（パブリックチャンネル用）や `message.groups`（プライベートチャンネル用）の追加を忘れています。これらのイベントがないと、Slackはチャンネルメッセージをボットに一切配信しません。
:::


---

## ステップ5: Messagesタブを有効にする

このステップはボットへのダイレクトメッセージを有効にします。これがないと、ユーザーがボットにDMしようとすると **"Sending messages to this app has been turned off"** と表示されます。

1. サイドバーで **Features → App Home** に移動
2. **Show Tabs** までスクロール
3. **Messages Tab** をONに切り替え
4. **"Allow users to send Slash commands and messages from the messages tab"** をチェック

:::danger このステップがないとDMは完全にブロックされる
すべてのスコープとイベントサブスクリプションが正しくても、Messagesタブが有効でない限り、Slackはユーザーがボットにダイレクトメッセージを送信することを許可しません。これはHermesの設定の問題ではなく、Slackプラットフォームの要件です。
:::

---

## ステップ6: アプリをワークスペースにインストールする

1. サイドバーで **Settings → Install App** に移動
2. **Install to Workspace** をクリック
3. 権限を確認し、**Allow** をクリック
4. 認可後、`xoxb-` で始まる **Bot User OAuth Token** が表示されます
5. **このトークンをコピー** します — これがあなたの `SLACK_BOT_TOKEN` です

:::tip
後でスコープやイベントサブスクリプションを変更した場合、変更を反映するには**アプリを再インストールする必要があります**。Install Appページには、そうするよう促すバナーが表示されます。
:::

---

## ステップ7: 許可リスト用のユーザーIDを見つける

Hermesは許可リストにSlackの**メンバーID**（ユーザー名や表示名ではない）を使用します。

メンバーIDを見つけるには:

1. Slackで、ユーザーの名前またはアバターをクリック
2. **View full profile** をクリック
3. **⋮**（その他）ボタンをクリック
4. **Copy member ID** を選択

メンバーIDは `U01ABC2DEF3` のような形式です。最低でも自分自身のメンバーIDが必要です。

---

## ステップ8: Hermesを設定する

`~/.hermes/.env` ファイルに次を追加します:

```bash
# Required
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_ALLOWED_USERS=U01ABC2DEF3              # Comma-separated Member IDs

# Optional
SLACK_HOME_CHANNEL=C01234567890              # Default channel for cron/scheduled messages
SLACK_HOME_CHANNEL_NAME=general              # Human-readable name for the home channel (optional)
```

または、対話型セットアップを実行します:

```bash
hermes gateway setup    # Select Slack when prompted
```

その後、ゲートウェイを起動します:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

:::tip Codexの推論エフォートの安全性
CodexをバックエンドとするSlackのピアエージェントチャンネルでは、`agent.reasoning_effort: high` 以下を推奨します。`xhigh`
はターン全体を隠れた推論に費やし、目に見えるアシスタントのテキストを一切生成しないことがあります。Hermesは現在、
そうした未完了ターンの警告をスレッドに表示せず、診断情報はゲートウェイのログに残します。
:::

---

## ステップ9: ボットをチャンネルに招待する

ゲートウェイを起動した後、ボットに応答してほしいチャンネルにボットを**招待する**必要があります:

```
/invite @Hermes Agent
```

ボットは自動的にチャンネルに参加**しません**。各チャンネルに個別に招待する必要があります。

---

## スラッシュコマンド

すべてのHermesコマンド（`/btw`、`/stop`、`/new`、`/model`、`/help` など）は、ネイティブのSlackスラッシュコマンドです — TelegramやDiscordでの動作とまったく同じです。Slackで `/` と入力すると、オートコンプリートのピッカーがすべてのHermesコマンドとその説明を一覧表示します。

内部の仕組み: Hermesには、[`COMMAND_REGISTRY`](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/commands.py) のすべてのコマンドをスラッシュコマンドとして宣言する、生成済みのSlackアプリマニフェスト（ステップ1、オプションAを参照）が同梱されています。Socket Modeでは、Slackはマニフェストの `url` フィールドに関係なく、コマンドイベントをWebSocket経由でルーティングします。

### Agentメッセージング体験

新しいSlackアプリは、Slackの **Agent** メッセージング体験を使用します。既存のHermes
Assistantアプリは、`--agent-view` を付けてマニフェストを再生成することで移行できます:

```bash
hermes slack manifest --agent-view --write
```

**Features → App Manifest** でマニフェストを更新し、Slackに求められたらアプリを再インストールします。
AgentビューをAssistantビューに戻すことはできず、切り替え後はユーザーがSlackをハードリフレッシュする必要がある場合があります。
生成されたAgentマニフェストは `message.im`、`app_home_opened`、`app_context_changed` をサブスクライブするため、
HermesはMessagesタブのDMを識別し、ユーザーのアクティブなSlackコンテキストをターンとともに受け取れます。
Hermesはそのコンテキストをラベルとして提供するだけで、表示中のチャンネルの履歴を読み取ることはありません。

### 更新後にスラッシュコマンドを更新する

Hermesが新しいコマンドを追加したとき（例: `hermes update` の後）、マニフェストを再生成してSlackアプリを更新します:

```bash
hermes slack manifest --write
```

その後、Slackで:
1. [https://api.slack.com/apps](https://api.slack.com/apps) →
   あなたのHermesアプリを開く
2. **Features → App Manifest → Edit**
3. `~/.hermes/slack-manifest.json` の新しい内容を貼り付け
4. **Save**。スコープやスラッシュコマンドが変更された場合、Slackはアプリの再インストールを促します。

### レガシーの `/hermes <subcommand>` も引き続き動作する

古いマニフェストとの後方互換性のため、`/hermes bg run the tests` と入力することもできます — Hermesは `/bg run the tests` と同じ方法でルーティングします。自由形式の質問も機能します: `/hermes what's the weather?` は通常のメッセージとして扱われます。

### スレッド内でコマンドを使う（`!cmd` プレフィックス） {#using-commands-inside-threads-the-cmd-prefix}

Slack自体が、スレッド返信内でのネイティブのスラッシュコマンドをブロックしています — スレッドで
`/queue` を試すと、Slackは *"/queue is not supported in threads. Sorry!"* と応答します。
これを再度有効にするアプリ側の設定はなく、SlackがそれらをHermesに配信することはありません。

回避策として、Hermesは先頭の `!` を、スレッド内（およびその他の場所）で動作する代替のコマンドプレフィックスとして認識します。
通常のスレッド返信として `!queue`、`!stop`、`!model gpt-5.4` などと入力してください —
Hermesはスラッシュ形式とまったく同じように扱い、同じスレッドで返信します。

既知のコマンドリストと照合されるのは最初のトークンだけなので、`!nice work` のような
カジュアルなメッセージはそのままエージェントに渡されます。
感嘆符の形式はメンションの後（`@Hermes !stop`）や先頭に空白がある場合にも動作し、
どちらもスレッド内でコマンドとしてディスパッチされます。

承認プロンプト（危険なコマンド / `execute_code` の承認）は通常、インタラクティブなボタンとして表示されます。
ボタンを配信できずHermesがテキストプロンプトにフォールバックした場合、プロンプトは
`!approve` / `!deny` で返信するよう指示します — これはスレッド内で動作する形式です。

### スラッシュコマンドの返信はエフェメラル

ネイティブのスラッシュコマンド（例: `/status`、`/help`）への返信は **エフェメラル** に配信されます
— "Only visible to you" — そのため、コマンドの出力がチャンネルを荒らすことはありません。"Running /cmd…" の
プレースホルダーは実際の返信に置き換えられ、長い返信は後続のエフェメラルメッセージに分割されます。
Slackは返信フローを5件の投稿に制限しているため、極端に長い出力は黙って破棄されるのではなく、
明示的な切り詰めの通知で閉じられます。主要なエフェメラルのパスが失敗した場合、Hermesは2つ目の
エフェメラルAPIパスでリトライします — フォールバックとしてスラッシュコマンドの返信がチャンネルに公開投稿されることはありません。
（通常のメッセージとして入力したコマンド — スレッド内の `!cmd`、`@Hermes /cmd` — は、
代わりに通常の目に見えるメッセージとして返信します。）

### 確認プロンプト（ワンタップのボタン）

エージェントが多肢選択の質問をする必要がある場合（`clarify` ツール）、Slackはそれを
**Block Kitのボタン** として表示します — 選択肢ごとにワンタップで、さらに自由入力モードに切り替える
"✏️ Other…" ボタンがあります（次に入力したメッセージが回答になります）。タップすると、メッセージは
その場で更新され、誰が回答し何が選ばれたかが表示されます。同じプロンプトへのそれ以降のクリックは無視されます。
ボタンのクリックには、メッセージと同じユーザー認可が適用されます。プロンプトがタイムアウトした場合
（`agent.clarify_timeout`）、セッションがリセットされた場合、またはボタンをタップする代わりに自由入力で返信した場合、
カードはボタンなしでその場で書き換えられます（"⏳ This prompt expired…" または "↩️ Clarification cancelled…"）。
ゲートウェイの再起動によって取り残されたカードをクリックしても、クリックが黙って無視されるのではなく、
再度質問するよう伝えられます。自由回答形式の確認質問は通常の質問として表示され、
次に入力した返信を受け付けます。設定は不要です — これは `rich_blocks` の設定に関係なく動作します。

### 高度な使い方: スラッシュコマンド配列のみを出力する

Slackマニフェストを手動で管理していて、スラッシュコマンドのリストだけが欲しい場合:

```bash
hermes slack manifest --slashes-only > ~/.hermes/cache/scratch/slashes.json
```

その配列を、既存のマニフェストの `features.slash_commands` キーに貼り付けます。

---

## ボットの応答の仕組み

Hermesがさまざまなコンテキストでどのように振る舞うかを理解しましょう:

| Context | Behavior |
|---------|----------|
| **DM** | ボットはすべてのメッセージに応答します — @メンション不要 |
| **チャンネル** | ボットは**@メンションされたときのみ応答します**（例: `@Hermes Agent what time is it?`）。チャンネルでは、Hermesはそのメッセージに紐づくスレッドで返信します。 |
| **スレッド** | 既存のスレッド内でHermesを@メンションすると、その同じスレッドで返信します。ボットがスレッド内でアクティブなセッションを持つと、**そのスレッドでの以降の返信には@メンションが不要になります** — ボットは会話を自然に追跡します。 |

:::tip
チャンネルでは、会話を始めるには常にボットを@メンションしてください。ボットがスレッド内でアクティブになると、メンションせずにそのスレッドで返信できます。スレッドの外では、@メンションのないメッセージは、混雑したチャンネルでのノイズを防ぐために無視されます。
:::

---

## 設定オプション

ステップ8の必須環境変数に加えて、`~/.hermes/config.yaml` を通じてSlackボットの動作をカスタマイズできます。

### スレッドと返信の動作

```yaml
platforms:
  slack:
    # Controls how multi-part responses are threaded
    # "off"   — never thread replies to the original message
    # "first" — first chunk threads to user's message (default)
    # "all"   — all chunks thread to user's message
    reply_to_mode: "first"

    extra:
      # Whether to reply in a thread (default: true).
      # When false, channel messages get direct channel replies instead
      # of threads. Messages inside existing threads still reply in-thread.
      reply_in_thread: true

      # Also post thread replies to the main channel
      # (Slack's "Also send to channel" feature).
      # Only the first chunk of the first reply is broadcast.
      reply_broadcast: false

      # Control Slack's automatic link-preview cards without changing or
      # removing clickable links from message text. Omit either key to keep
      # Slack's default behavior for that preview type.
      unfurl_links: false
      unfurl_media: false

      # Render agent messages as Slack Block Kit blocks (default: false).
      # When true, the final agent message is sent with structured blocks —
      # section headers, dividers, true nested lists (via rich_text), and
      # native Block Kit tables — instead of flat mrkdwn text. A plain-text
      # fallback is always sent alongside for notifications/accessibility.
      # Tables exceeding Slack's limits (100 rows / 20 cols / 10k chars)
      # gracefully fall back to aligned monospace.
      rich_blocks: false

      # Append Slack-native feedback controls to final Block Kit replies.
      # Requires rich_blocks: true. Default: false.
      feedback_buttons: false

      # Render live tool calls as Slack-native plan/task cards. Works with
      # Slack's built-in tool_progress: off default; a tool_progress: off you
      # write yourself disables cards too. Cards need a thread: an un-threaded
      # chat shows no tool progress (text progress if you wrote new/all).
      # Recoverable native API failures keep one
      # editable text fallback current for the rest of the turn.
      native_task_cards: false

      # Suggested prompts pinned at the top of Agent view's Messages tab.
      # Either a list of {title, message} rows, or a titled object:
      # {title: "Start here", prompts: [{title: "Plan", message: "..."}]}
      suggested_prompts: []

      # Title Agent/Assistant DM threads from the first user message.
      # Default: true. Set false to leave Slack's default thread titles.
      assistant_thread_titles: true

      # Accept messages posted by other Slack bots (default: "none").
      # "none" ignores bots, "mentions" accepts a bot message only when
      # that message itself @mentions Hermes, and "all" accepts every
      # other bot. Hermes always ignores its own bot user to prevent
      # self-echoes.
      allow_bots: "none"

      # Continuable-cron delivery surface (default: "thread").
      # "in_channel" delivers a continuable cron job FLAT into the channel
      # (no dedicated thread); pair with reply_in_thread: false (and
      # require_mention: false) so a plain reply continues the job.
      # See the cron guide → "Flat, in-channel continuation".
      cron_continuable_surface: thread
```

| Key | Default | Description |
|-----|---------|-------------|
| `platforms.slack.reply_to_mode` | `"first"` | 複数パートメッセージのスレッド化モード: `"off"`、`"first"`、または `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | `false` の場合、チャンネルメッセージはスレッドではなく直接返信を受け取ります。既存のスレッド内のメッセージは引き続きスレッド内で返信します。 |
| `platforms.slack.extra.reply_broadcast` | `false` | `true` の場合、スレッド返信はメインチャンネルにも投稿されます。最初のチャンクのみがブロードキャストされます。 |
| `platforms.slack.extra.unfurl_links` | Slackのデフォルト | `false` に設定すると、クリック可能なリンクを保持したまま、リンクされたWebページの自動プレビューを抑制します。いずれかのunfurlキーが設定されている場合、メディアのキャプションはファイルの *前に* 別のメッセージとして投稿され（SlackのアップロードAPIはunfurlの制御を運べないため）、ネイティブのドラフトストリーミングは編集ベースの配信にフォールバックします。 |
| `platforms.slack.extra.unfurl_media` | Slackのデフォルト | `false` に設定すると、クリック可能なリンクを保持したまま、メディアの自動プレビューを抑制します。キャプションの順序とストリーミングに関する注意点は `unfurl_links` と同じです。 |
| `platforms.slack.extra.rich_blocks` | `false` | `true` の場合、エージェントのメッセージは [Block Kit](https://docs.slack.dev/block-kit/) のブロック（ヘッダー、区切り線、真のネストされたリスト、ネイティブのテーブル）として表示されます。Markdownの `[label](url)` リンクとSlackの `<url\|label>` 自動リンクは、どちらもリスト、引用、テーブルのセル内でクリック可能なリンクになります。メンション（`<@U…>`、`<#C…>`、`<!here>`）はそのまま残されます。プレーンテキストのフォールバックが常に送信されます。Slackの制限を超えるテーブルは、整列された等幅テキストにフォールバックします。アプリの再インストールは不要です — 送信側だけの変更です。 |
| `platforms.slack.extra.feedback_buttons` | `false` | `rich_blocks` とともに `true` にすると、最終返信にSlackネイティブのフィードバックコントロールを追加します。 |
| `platforms.slack.extra.native_task_cards` | `false` | `true` の場合、ライブのツール呼び出しをSlackネイティブのplan/taskカードとして表示します。カードはSlackの組み込みのデフォルト `tool_progress: off` で動作します。明示的に設定された `display.tool_progress: off`（グローバルまたは `display.platforms.slack`。`/verbose` も同じキーに書き込みます）は、カードも無効にします。カードにはスレッドが必要です: カードのレーンがアクティブで、チャットに固定先のスレッドがない場合（`reply_in_thread: false` のトップレベルのDM）、Hermesはテキストバブルの代わりにツールの進捗を表示しません。ただし、明示的に `tool_progress: new`/`all` を設定した場合は、そこで編集可能なテキストの進捗にフォールバックします。回復可能なネイティブAPIの失敗は、継続的に編集される1つのテキスト更新にフォールバックします。 |
| `platforms.slack.extra.suggested_prompts` | `[]` | Agent/AssistantのDMの入り口用の、最大4つの `{title, message}` プロンプト。リストまたは `{title, prompts}` のどちらも受け付けます。 |
| `platforms.slack.extra.assistant_thread_titles` | `true` | `true` の場合、Agent/AssistantのDMスレッドに最初のユーザーメッセージから名前を付けます。 |
| `platforms.slack.extra.allow_bots` | `"none"` | 他のSlackボットからのメッセージを制御します: `"none"` は無視し、`"mentions"` は **そのメッセージ自体** がHermesを@メンションしている場合にのみボットのメッセージを受け付け、`"all"` はすべてを受け付けます。最も安全なボット間連携モードには `"mentions"` を使ってください。[他のボットからのメッセージを受け付ける](#accepting-messages-from-other-bots-allow_bots) を参照してください。 |
| `platforms.slack.extra.api_human_users` | `[]` | **Web API（ユーザートークン）での投稿を人間として扱う** SlackユーザーID。そのような投稿は投稿した `app_id` を持ち、`client_msg_id` を持たないため、デフォルトではアプリのトラフィックとして破棄されます。`allow_bots: all` の代わりに、自分のフロントエンドのユーザーをここで許可リストに追加してください。[自分のアプリのユーザートークンでの投稿を人間として扱う](#treating-your-own-apps-user-token-posts-as-human-api_human_users) を参照してください。 |
| `platforms.slack.extra.cron_continuable_surface` | `"thread"` | [継続可能なcronジョブ](../features/cron.md#flat-in-channel-continuation-slack) の配信先。`"thread"` は配信ごとに専用のスレッドを開きます（デフォルト）。`"in_channel"` はチャンネルのタイムラインにフラットに配信します。`in_channel` は `reply_in_thread: false`（および `require_mention: false`）と組み合わせて、チャンネルでの通常の返信でジョブが継続するようにしてください。 |

同等の環境変数は `SLACK_ALLOW_BOTS=none|mentions|all` です。
両方が設定されている場合は、明示的な環境変数が優先されます（他のすべての設定と同じ、環境変数がYAMLより優先されるルール）。
ピアボット同士が明示的なメンションなしに互いに応答できる場合は `all` を避けてください。
それぞれの返信ポリシーによってループが発生する可能性があります。

### 作業中のステータス行

エージェントがメッセージを処理している間、Slackはスレッド内のボット名の横にステータス行を表示します。
デフォルトではHermesはこれを `is thinking...` に設定します。`typing_status_text` でカスタマイズできます
— 例えば、Adaという名前の子猫のアシスタントの場合:

```yaml
platforms:
  slack:
    # Custom working-state status line (default: "is thinking...").
    typing_status_text: "is pouncing… 🐾"
```

| Key | Default | Description |
|-----|---------|-------------|
| `platforms.slack.typing_status_text` | `"is thinking..."` | エージェントがメッセージを処理している間に表示される、作業中のステータス行のテキスト。`assistant:write` スコープが必要です — これがないとステータスの呼び出しは黙って失敗し、これが何に設定されていても、Slackは独自の汎用プレースホルダーを表示します。ステータス行を完全に無効にするには `typing_indicator: false` を設定します。 |

:::note ステータスが表示される場所
カスタムのステータスは、メッセージリスト内のインラインではなく、**返信入力欄の下のフッター** に表示されます（"*BotName* is thinking…"）。AIアプリの動作中にSlackがメッセージエリアに表示するインラインの "Generating response…" / "Finding answers…" の行は **Slack独自の順に切り替わるインジケーター** です — ステータスAPI（`agents.sessions.setStatus` / `assistant.threads.setStatus`）はそれらを制御せず、両方が同時に表示されることもあります。
:::

同じキーで、Google Chatの目に見える作業中マーカーのメッセージ
（`platforms.google_chat.typing_status_text`、デフォルト `"Hermes is thinking…"`）もカスタマイズできます
— Google Chatでは、それはエフェメラルなステータスではなく、返信に差し替えられる実際に投稿されたメッセージである点に注意してください。

### ライブステータス（ツールごと）

デフォルトでは、ステータス行は **エージェントの作業に合わせてライブに更新** されます: 静的な `is thinking...` の代わりに、
エージェントが今何をしているかを表示します — `is running pytest tests/…`、`is reading docs/api.md…`、
`is searching the web for slack api limits…` など。ツール呼び出しの合間には静的なテキストに戻ります。
これは既存のステータス更新の周期に乗るため、追加のSlack API呼び出しは発生せず、`tool_progress: off`（Slackのデフォルト）でも動作します
— 進捗バブルとは異なり、ステータス行はエフェメラルで、チャンネルに何も残しません。

`display.live_status`（グローバルまたはプラットフォームごと）で制御します:

```yaml
display:
  platforms:
    slack:
      # full = verb + argument ("is running pytest…")   [default]
      # verb = verb only ("is running…") — hides commands/paths,
      #        useful in shared or customer-facing channels
      # off  = static text (typing_status_text or "is thinking...")
      live_status: full
```

| Key | Default | Description |
|-----|---------|-------------|
| `display.live_status` | `"full"` | ツールごとのライブステータス行。`full` は動詞 + 引数のプレビューを表示し、`verb` は動詞のみを表示し（ファイルパスやコマンドを共有チャンネルに出さない）、`off` は静的なテキストに戻します。静的なステータス行と同じく `assistant:write` スコープが必要です。 |

### ネイティブストリーミング（ライブタイピングの返信）

Slackの [Agents & AI Apps](https://docs.slack.dev/ai/) 機能には、返信をライブタイピングのメッセージとして表示する
ネイティブのストリーミング機能（`chat.startStream` / `chat.appendStream` / `chat.stopStream`）が含まれています
— それ以外の場合に使われる編集ベースの段階的な更新よりもはるかに滑らかです。`streaming.enabled` がオン
（トランスポートが `auto` または `draft`）の場合、Hermesは利用可能な場所で自動的にネイティブストリーミングを使用します:

- ストリームは最初のフレームで開始され、差分のみを追加します（APIは追加専用です）。ストリーミングされたメッセージ
  **が** 最終メッセージです — Hermesは重複した最終返信を投稿する代わりに、`chat.stopStream` でそれを確定させます。
- SlackアプリでAI機能が有効になっていない（または `assistant:write` スコープがない）場合、最初の失敗がキャッシュされ、
  Hermesは修正方法を示す1回のログ警告とともに編集ベースのストリーミングにフォールバックします。
- オプトインのBlock Kit（`rich_blocks: true`）は、編集ベースの確定パスと同様に、確定したメッセージに適用されます。

ストリーミングを有効にする以外に、追加の設定は不要です:

```yaml
streaming:
  enabled: true       # transport auto/draft lights up Slack native streaming
```

### ネイティブタスクカード（ライブのツール進捗）

`platforms.slack.extra.native_task_cards: true` を設定すると、ライブのツール呼び出しは、テキストの進捗バブルの代わりに
Slackネイティブの **plan/taskカード**（Slack自身のAI機能が使うのと同じUI）として表示されます: ターンごとに1枚のカード、
ツール呼び出しごとに1行で、タスクごとの実行中/完了/エラーの状態がその場で更新されます。

```yaml
platforms:
  slack:
    extra:
      native_task_cards: true
```

- カードはツール進捗のSlackでの表示形式です。Slackの組み込みのデフォルト `tool_progress: off` で動作します。
  自分で `tool_progress: off` を書く（グローバル、`display.platforms.slack` の下、または `/verbose` をoffまで切り替える）と、
  カードもオフになります。`new` または `all` ではカードは維持されます。`null` の値は継承を意味し、「off」ではありません。
  また、どのYAMLレイヤーもnull以外の値を設定していない場合、nullは既存の環境変数ブリッジがモードを提供することを許可します。
  変更は、次のターンが表示設定を解決するときに適用されます。
- カードにはスレッドが必要です。カードのレーンがアクティブな場合、固定先のスレッドがないチャット
  （`reply_in_thread: false` でのトップレベルのDM）では、Slackのデフォルト `tool_progress: off` のもとで
  テキストバブルではなく、ツールの進捗が表示されません。`new` または `all` を書いた場合、そのチャットでは要求した
  編集可能なテキストの進捗が表示されます。既存のスレッド内の返信では引き続きカードが表示されます。
- 同じツールへの同時呼び出しは実際のツール呼び出しIDで対応付けられるため、並列の `web_search` 呼び出しは
  それぞれ正しいステータスを持つ独自の行を得ます。
- Hermesは公開を試みる前にスレッドの適格性を確認するため、切断やタイムアウトによって、スレッドのない配信先が
  テキストのフォールバックになることはありません。
- サポートされるスレッド付きの配信先で、ネイティブのストリームが回復可能な理由（APIエラー、レート制限）で失敗した場合、
  Hermesは継続的に編集される1つのテキストメッセージにフォールバックするため、そのターンの間は進捗がライブのまま保たれます。
  リレーのエグレスによる配信先の拒否は回復可能ではなく、そのターンの進捗は抑制されます。
- 長いターンの途中でSlackがストリームを閉じた場合、Hermesは同じスレッドに現在のタスクリストを持つ新しいカードを開き、
  そちらで更新を続けます。以前のカードは表示されたまま残ります。
- カードのストリームは、割り込み/切断時を含め、ターンの確定時にちょうど1回だけ停止されるため、
  ライブのインジケーターが残されたままになることはありません。

### セッションの分離

```yaml
# Global setting — applies to Slack and all other platforms
group_sessions_per_user: true
```

`true`（デフォルト）の場合、共有チャンネル内の各ユーザーは、それぞれ分離された会話セッションを持ちます。`#general` でHermesと話す2人は、別々の履歴とコンテキストを持ちます。

チャンネル全体が1つの会話セッションを共有する協調モードが必要な場合は `false` に設定します。これはユーザーがコンテキストの増大とトークンコストを共有し、あるユーザーの `/reset` が全員のセッションをクリアすることを意味する点に注意してください。

### メンションとトリガーの動作

```yaml
slack:
  # Require @mention in channels (this is the default behavior;
  # the Slack adapter enforces @mention gating in channels regardless,
  # but you can set this explicitly for consistency with other platforms)
  require_mention: true

  # Prevent thread auto-engagement: only reply to channel messages that
  # contain an explicit @mention. With this OFF (default), Slack can
  # "auto-engage" — remembering past mentions in a thread and following
  # up on bot-message replies, and resuming active sessions without a
  # fresh mention. With strict_mention ON, every new channel message
  # must @mention the bot before Hermes will respond.
  strict_mention: false

  # Ignore messages addressed to another user: when a channel or thread
  # message *opens* by @mentioning someone other than the bot (e.g.
  # "@rasha can you take this?"), stay silent unless the bot is also
  # mentioned. Only a *leading* mention counts as "addressed to" — a
  # message that references someone mid-sentence ("loop in @rasha")
  # still reaches the bot. Overrides free_response_channels and thread
  # auto-engagement. Opt-in; default off. Env: SLACK_IGNORE_OTHER_USER_MENTIONS.
  ignore_other_user_mentions: false

  # Require an explicit @mention for THREAD replies, while leaving
  # top-level channel messages governed by require_mention /
  # free_response_channels. Narrower than strict_mention: use it when a
  # free-response bot should not join every follow-up in busy threads.
  # Opt-in; default off. Env: SLACK_THREAD_REQUIRE_MENTION.
  thread_require_mention: false

  # Let a bare silence marker stand on unaddressed replies in a channel
  # thread the bot follows. Without it, when a human in that thread talks
  # to someone else without @mentioning them and the model answers with
  # only [SILENT], Hermes posts the "model returned only a silence marker"
  # notice instead. DMs (1:1 and group), @mentions of the bot, and flat
  # (reply_in_thread: false) channels are unaffected.
  # Opt-in; default off. Env: SLACK_THREAD_FOLLOWUP_SILENCE.
  thread_followup_silence: false

  # Per-channel force-mention override — the opposite direction of
  # free_response_channels. Channels listed here ALWAYS require an
  # explicit @mention, even when require_mention is false globally.
  # Ongoing conversations still auto-follow (mentioned threads, active
  # sessions, bot-authored threads). Comma-separated IDs or a list.
  # Env: SLACK_REQUIRE_MENTION_CHANNELS.
  require_mention_channels: ""

  # Custom mention patterns that trigger the bot
  # (in addition to the default @mention detection)
  mention_patterns:
    - "hey hermes"
    - "hermes,"
```

:::tip `strict_mention` を使うべきとき
Slackのデフォルトの「ボットがこのスレッドを記憶する」動作がユーザーを驚かせる、混雑したワークスペースでは `true` に設定してください — 例えば、ボットが最初に手助けした長い技術サポートスレッドで、明示的に再びpingされない限り静かにしていてほしい場合です。DMとアクティブな対話セッションは影響を受けません。
:::

:::tip `ignore_other_user_mentions` を使うべきとき
ボットが（スレッドの自動エンゲージメントや `free_response_channels` によって）混雑したスレッドを追跡していて、人間同士が宛てたメッセージに割り込んでしまう場合は `true` に設定してください。これは `strict_mention` よりも対象の狭いツールです: エンゲージ済みのスレッドでの通常のフォローアップには引き続き応答し、別の人を@メンションして始まるメッセージだけがスキップされます。**1:1のDMは影響を受けません**。グループDM（MPIM）とチャンネルはどちらもこれを適用し、以下の共有サーフェスのポリシーと一致します。ブロードキャストのトークン（`@here`、`@channel`）やチャンネルの参照は、個人ではなくその場に宛てたものなので、スキップされることはありません。
:::

:::note ボット宛てでないメッセージでのサイレンスマーカー
ボットが人間のメッセージに [サイレンストークン](index.md#intentional-silence-tokens) だけで応答した場合、質問が未回答のままにならないよう、Hermesは通常、代わりに短い通知を投稿します。Slackでは、メッセージが別の人を@メンションして始まっていた場合、または独自のスレッドを開始する（デフォルトの `reply_in_thread: true`）`free_response_channels` チャンネルでのメンションなしのトップレベルメッセージだった場合に、トークンがそのまま残ることが許されます。1:1のDM、ボットへのメンション、コマンド、リアクショントリガー、またはボットが参加している会話（スレッド、または `reply_in_thread: false` のチャンネル）での通常のフォローアップでは、引き続き通知が投稿されます。人々が@メンションせずに互いに話すことが多いスレッド向けに、チャンネルのスレッドでの宛先のない返信でもトークンをそのまま残したい場合は `thread_followup_silence: true` を設定してください。DM（1:1とグループ）、ボットへのメンション、フラットなチャンネルでは通知が維持されます。すべてのSlackチャット（リレー経由のSlackやDMを含む）で単独のトークンをそのまま残したい場合は、`platforms.slack` の下で `allow_silence: true` を設定してください。[プラットフォームごと・ルームごとの意図的なサイレンス](index.md#per-platform-and-per-room-intentional-silence) を参照してください。
:::

:::info
Slackは両方のパターンをサポートします: デフォルトでは会話を始めるのに `@mention` が必要ですが、`SLACK_FREE_RESPONSE_CHANNELS`（カンマ区切りのチャンネルID）または `config.yaml` の `slack.free_response_channels` で特定のチャンネルを除外できます。ボットがスレッド内でアクティブなセッションを持つと、以降のスレッド返信にメンションは不要です。**1:1のDM** では、ボットはメンションを必要とせずに常に応答します。これらのゲーティングのキーは、トップレベルの `slack:` ブロックまたは `platforms.slack.extra` の下で設定できます。ユーザーのYAML内で同じキーが両方に設定されている場合は `platforms.slack.extra` が優先され、ゲートウェイは警告をログに記録します。管理者が管理するピン留めは引き続き権威を持ち、明示的な環境変数の設定は既存の優先順位を維持します。
:::

:::caution グループDM（MPIM）は1:1のDMではなく共有サーフェス
**1:1のダイレクトメッセージ** は1人とのプライベートな会話なので、メンションが免除されます。**グループDM（MPIM / 複数人DM）** は *共有サーフェス* です — 複数の人がボットを見てトリガーできます — そのため、チャンネルと同じオペレーターの制御に従います: `require_mention`、`strict_mention`、`free_response_channels`、`allowed_channels` がすべて適用され、ボットは実際に `@mentioned` されたときにのみ `:eyes:`/`:white_check_mark:` のリアクションを追加します。特定のグループDMでボットに自由に応答させるには、そのチャンネルID（`G` で始まる）を `free_response_channels` に追加してください。
:::

#### どのメンションオプションを使うべきか

ゲーティングのオプションは組み合わせられます — それぞれが異なる問いに答えます:

| Option | 答える問い | Default | Scope |
|--------|--------------------|---------|-------|
| `require_mention` | **トップレベルのチャンネルメッセージ** に@メンションが必要か？ | `true` | すべてのチャンネル |
| `free_response_channels` | どのチャンネルを `require_mention` から除外するか？ | なし | 列挙したチャンネル |
| `require_mention_channels` | `require_mention` が `false` の場合やチャンネルがフリーレスポンスの場合でも、どのチャンネルで常に@メンションを必要とするか？ 両方より優先されます。 | なし | 列挙したチャンネル |
| `thread_require_mention` | トップレベルのメッセージで不要な場合でも、**スレッド返信** に@メンションが必要か？ メンションされたスレッドは記憶されません。 | `false` | スレッドのみ |
| `strict_mention` | **すべての** チャンネルメッセージ（トップレベルとスレッド）に新たな@メンションが必要か？ すべての自動追跡（メンションされたスレッドの記憶、ボットの返信へのフォローアップ、アクティブなセッションの再開）を無効にします。 | `false` | すべてのチャンネル + スレッド |
| `ignore_other_user_mentions` | **別の人を@メンションして始まる** メッセージ（`@rasha can you take this?`）をスキップすべきか？ フリーレスポンスとスレッドの自動追跡より優先されます。文中での参照は引き続きボットに届きます。 | `false` | チャンネル + グループDM |

経験則: `strict_mention` は最も広範なハンマーです。`thread_require_mention` はトップレベルのゲーティングに触れずに混雑したスレッドを静かにします。`require_mention_channels` は、それ以外はフリーレスポンスのボットで個々のチャンネルを再び厳しくします。`ignore_other_user_mentions` は、明示的に別の人に宛てたメッセージだけをスキップします。1:1のDMは常に応答し、これらのいずれの影響も受けません。

### 他のボットからのメッセージを受け付ける（`allow_bots`） {#accepting-messages-from-other-bots-allow_bots}

デフォルトでは、Hermesは他のSlackボットやアプリが作成したすべてのメッセージ（Workflow Builderの投稿を含む）を無視します。マルチエージェントのワークスペース — 複数のHermesインスタンスやピアボットが1つのチャンネルで連携する場合 — では、`allow_bots` でオプトインします:

```yaml
platforms:
  slack:
    extra:
      # "none" (default) — ignore all bot/app-authored messages
      # "mentions"       — accept a bot message only when THAT message
      #                    @mentions this bot
      # "all"            — accept every bot message (except the bot's own)
      allow_bots: mentions
```

同等の環境変数: `SLACK_ALLOW_BOTS=none|mentions|all`（両方が設定されている場合は設定キーが優先されます）。不明な値は `none` として扱われます。

`mentions` モードのゲーティングの仕組み:

- ピアボットのメッセージは、**メッセージ自体がこのボットへの現在の `@mention` を含む場合にのみ** 受け付けられます — テキストまたはBlock Kitのブロック内で。スレッドの履歴はカウントされません: スレッド内で以前にボットがメンションされたこと、ボット自身のメッセージへの返信、アクティブなスレッドセッションは、その後のメンションなしのピアボットのメッセージを受け入れる理由には **なりません**。これは意図的なもので — エージェント間のack/statusのループを断ち切るのはこの仕組みです。
- 人間のメッセージは影響を受けません。通常のメンションのゲーティングが適用されます。
- Hermesは、自己エコーのループを防ぐため、すべてのモードで常に自分自身のメッセージを無視します。

`mentions` は、ボット間連携に推奨されるモードです: 各エージェントは、ターンごとに相手を明示的に呼び出す必要があります。すべてのピアボットの返信ポリシーがループに対して安全でない限り、`all` は避けてください — すべてに応答する2つのボットは、永遠に互いに応答し続けます。検出は、ラベル付きのボットメッセージ（`bot_id`、`subtype: bot_message`）、アプリ発のイベント、ラベルのないボット *ユーザー*（`users.info` で調査）をカバーするため、ピアのHermesエージェントはワークスペース全体で一貫してフィルタリングされます。

厳格なマルチボットのデプロイでは、`require_mention: true` と `strict_mention: true` を組み合わせてください — 以下のスモークチェックのプロファイルを参照してください。

### 自分のアプリのユーザートークンでの投稿を人間として扱う（`api_human_users`） {#treating-your-own-apps-user-token-posts-as-human-api_human_users}

**ユーザートークン**（`xoxp-`）を使ってWeb API経由で投稿されたメッセージは実際の人物によって作成されたものですが、
投稿した `app_id` を持ち、`client_msg_id` を持たない状態で届きます — これはHermesがアプリの投稿を認識するのに
使うのと同じシグネチャです — そのため、ボットのトラフィックとして破棄されます。これは一般的なパターンを妨げます:
ログインしたユーザー *として* Hermesにメッセージを送るカスタムのフロントエンド（社内ダッシュボード、モバイルシェル、キオスク）です。

`allow_bots: all` にすればそれらの投稿は通りますが、チャンネル内のすべてのボットに扉を開き、ループの保護を弱めます。
代わりに、フロントエンドを使う人だけを許可リストに追加してください:

```yaml
platforms:
  slack:
    extra:
      api_human_users: ["U0AAAAAAA", "U0BBBBBBB"]
```

同等の環境変数は `SLACK_API_HUMAN_USERS`（カンマ区切り）です。

範囲と安全性:

- 許可リストは **ユーザーのみ** です。意図的にアプリIDの版はありません: モダンなボットトークン（`xoxb-`）は
  同じ `user` + `app_id` の形で投稿するため、アプリを信頼するとそのボット自身の投稿も受け入れてしまい、ループガードが無効になります。
- `bot_id` や `subtype: bot_message` を持つイベント、または `user` をまったく持たないイベントは、
  許可リストに関係なく常にボットの投稿として扱われます。
- パイプラインの残りの部分は変わりません: メンションのゲーティング、`allowed_channels`、
  `SLACK_ALLOWED_USERS` は、（今は人間となった）送信者に引き続き適用されます。

### リアクショントリガー（`reaction_triggers`）

デフォルトでは、絵文字のリアクションは確認応答されて破棄されます — ボットのメッセージへの👍は何も起こしません。
`slack.reaction_triggers` を設定すると、リアクションをエージェントのループにルーティングします（`reactions:read` スコープと、
Slackアプリのマニフェストでの `reaction_added`/`reaction_removed` ボットイベントのサブスクリプションが必要です
— `hermes slack manifest` で再生成してください）:

```yaml
slack:
  # Opt-in. false/absent (default) = reactions are acked and dropped.
  # true = any reaction ON THE BOT'S OWN MESSAGES routes to the agent.
  reaction_triggers: true
  # Or an explicit emoji allowlist — only these names route, and they may
  # target ANY message (emoji-handoff workflows, e.g. :task: to capture):
  # reaction_triggers: [white_check_mark, thumbsup, task]
  # Optional handoff target: respond in this channel (top-level) or thread
  # (C123:<thread_ts>) instead of the reacted-to message's thread.
  # reaction_trigger_target: C0123456789
```

同等の環境変数: `SLACK_REACTION_TRIGGERS`（`true`/`all` またはカンマ区切りのリスト）と `SLACK_REACTION_TRIGGER_TARGET`。

動作:

- リアクションは、テキスト `reaction:added:👍` / `reaction:removed:👍` を持つ通常のエージェントのターンとして届きます
  （一般的なSlackの名前はUnicodeに変換されます。不明な名前はそのまま渡されます。例: `reaction:added:custom-emoji`）。
  リアクションされたメッセージの下にスレッド化されるため、エージェントは何にリアクションされたかを確認でき、
  ターンは返信と同じセッションに入ります。
- リアクションした人がメッセージのユーザーになるため、**ユーザーの認可と `allowed_channels` のゲーティングは、
  入力されたメッセージとまったく同じように適用されます** — 任意のユーザーのリアクションが、その人のメッセージでは
  エージェントをトリガーできない場所でトリガーすることはできません。
- `reaction_triggers: true` の場合、ルーティングされるのはボット **自身の** メッセージへのリアクションだけです
  （承認/確認応答のフロー）。明示的な絵文字の許可リストを使う場合、列挙した絵文字は任意のメッセージからルーティングされます。
- ボット自身のライフサイクルのリアクション（`:eyes:` など）がフィードバックされることはありません。
- このオプトインとは無関係に、すべての人間のリアクションは、エージェントのターンを必要としないオブザーバーのために
  `reaction:added`/`reaction:removed` の [ゲートウェイフック](../features/hooks.md#available-events) を発火します。

### ピアエージェントのスモークチェック

ターンごとの厳格なメンションに依存するマルチボットのSlackデプロイでは、次のプロファイルを維持してください:

```yaml
slack:
  require_mention: true
  strict_mention: true
  allow_bots: mentions
  allowed_channels: ""
```

ゲートウェイの設定変更、デプロイ、再起動の後に、この合成スモークターゲットを実行します:

```bash
scripts/run_tests.sh tests/gateway/test_slack_peer_agent_smoke.py -q
```

このターゲットは、プロセス内の合成Slackイベントのみを使用します。ライブのSlackメッセージは送信せず、デフォルトでは実際のボットトークンも必要ありません。

失敗の分類:

- `config:` `test_peer_agent_smoke_preflight_contract` がプロファイルの不一致（`require_mention`、`strict_mention`、`allow_bots`、または `allowed_channels`）を検出しました。
- `platform_connectivity:` アダプター/クライアントが初期化されていないため、ルーティングのスモークはまだ信頼できるシグナルではありません。
- `bot_identity:` アダプターがボットのユーザーIDを解決できなかったため、現在のメッセージのメンションチェックが機能しません。
- `routing_logic:` Slackアダプターが、ピアエージェントの不変条件のいずれか（人間のメンションのルーティング、ピアボットの無視、明示的なピアのメンションの受け入れ、受動的なack/status/errorの抑制）で退行しました。

このターゲットはパスするのに、ライブのワークスペースでまだメッセージが誤ってルーティングされる場合は、ルーティングロジック自体の外側で、Slackのトークン/ワークスペースの接続性とランタイムのデプロイ状態を調査してください。

### チャンネルの許可リスト（`allowed_channels`）

ボットを固定のSlackチャンネルのセットに制限します — ボットが多くのチャンネルに招待されているものの、一部でだけ応答すべき場合に便利です。設定すると、このリストにないチャンネルからのメッセージは、ボットが `@mentioned` されていても **黙って無視されます**。

**1:1のDMはこのフィルターから除外される** ため、認可されたユーザーはダイレクトメッセージで常にボットに到達できます。**グループDM（MPIM）は除外されません** — チャンネルと同様に、MPIMも許可リストに含まれている（そのIDは `G` で始まる）必要があり、そうでなければメッセージは破棄されます。

```yaml
slack:
  allowed_channels:
    - "C0123456789"   # #ops
    - "C0987654321"   # #incident-response
```

または環境変数で（カンマ区切り）:

```bash
SLACK_ALLOWED_CHANNELS="C0123456789,C0987654321"
```

動作:

- 空 / 未設定 → 制限なし（完全に後方互換）。
- 空でない → チャンネルIDがリストに含まれている必要があり、そうでなければ他のゲーティング（メンションの要件、`free_response_channels` など）が実行される前にメッセージは破棄されます。
- SlackのチャンネルIDは `C`（パブリック）、`G`（プライベート）、または `D`（DM）で始まります。SlackのUIの "Open channel details" → "About" パネル、またはAPIで確認できます。

関連項目: [管理者/ユーザーのスラッシュコマンドの分割](../../reference/slash-commands.md#permissions-and-adminuser-split)。

### 認可されていないユーザーの取り扱い

```yaml
slack:
  # What happens when an unauthorized user (not in SLACK_ALLOWED_USERS) DMs the bot
  # "pair"   — prompt them for a pairing code (default)
  # "ignore" — silently drop the message
  unauthorized_dm_behavior: "pair"
```

これをすべてのプラットフォームに対してグローバルに設定することもできます:

```yaml
unauthorized_dm_behavior: "pair"
```

`slack:` の下のプラットフォーム固有の設定が、グローバル設定より優先されます。

### 音声の文字起こし

```yaml
# Global setting — enable/disable automatic transcription of incoming voice messages
stt_enabled: true
```

`true`（デフォルト）の場合、受信した音声メッセージは、エージェントによって処理される前に、設定されたSTTプロバイダーを使って自動的に文字起こしされます。

### 完全な例

```yaml
# Global gateway settings
group_sessions_per_user: true
unauthorized_dm_behavior: "pair"
stt_enabled: true

# Slack-specific settings
slack:
  require_mention: true
  unauthorized_dm_behavior: "pair"

# Platform config
platforms:
  slack:
    reply_to_mode: "first"
    extra:
      reply_in_thread: true
      reply_broadcast: false
```

---


## ホームチャンネル

`SLACK_HOME_CHANNEL` に、Hermesがスケジュールされたメッセージ、cronジョブの結果、その他のプロアクティブな通知を配信するチャンネルIDを設定します。チャンネルIDを見つけるには:

1. Slackでチャンネル名を右クリック
2. **View channel details** をクリック
3. 一番下までスクロール — そこにChannel IDが表示されます

```bash
SLACK_HOME_CHANNEL=C01234567890
```

ボットがその**チャンネルに招待されている**（`/invite @Hermes Agent`）ことを確認してください。

### cronの配信先の指定

cronジョブ（[cronガイド](../features/cron.md#delivery-options) を参照）は、3つの方法でSlackを配信先に指定できます:

| `deliver:` value | 配信先 |
|------------------|----------------|
| `slack` | ホームチャンネル（`SLACK_HOME_CHANNEL`） |
| `slack:C0123456789` | IDで指定した特定のチャンネル |
| `slack:U0123456789` | そのユーザーの **DM** — 単独のユーザーIDは自動的にDMの会話に解決されます（`im:write` スコープが必要） |

配信は、cronプロセスがゲートウェイと同じ場所にない場合でも機能します — Hermesは `SLACK_BOT_TOKEN` を使うスタンドアロンのWeb API送信にフォールバックします。cronの出力内の `MEDIA:` 添付ファイルは、ネイティブのSlackファイル共有として同じ配信先にアップロードされます。

### メッセージとメディアの送信（`send_message`）

エージェントの `send_message` ツールは同じ形式の配信先を受け付けます: チャンネルID（`C…`/`G…`）、DMの会話（`D…`）、または単独のユーザーID（`U…`/`W…`）で、ユーザーIDはテキスト、メディア、インタラクティブなプロンプトのいずれの送信パスでも、そのユーザーのDMに解決されます。`MEDIA:<path>` の添付ファイル（画像、PDF、ドキュメント）はネイティブのファイル共有としてアップロードされます。短いメッセージが単一の添付ファイルに付随する場合、それは別のメッセージではなくファイルのキャプションとして添えられます。見つからないファイルは、送信全体を失敗させるのではなく、ファイルごとに警告として報告されます。

---

## マルチワークスペースのサポート

Hermesは、単一のゲートウェイインスタンスを使って**複数のSlackワークスペース**に同時に接続できます。各ワークスペースは、それぞれのボットユーザーIDで独立して認証されます。

### 設定

`SLACK_BOT_TOKEN` に**カンマ区切りのリスト**として複数のボットトークンを指定します:

```bash
# Multiple bot tokens — one per workspace
SLACK_BOT_TOKEN=xoxb-workspace1-token,xoxb-workspace2-token,xoxb-workspace3-token

# A single app-level token is still used for Socket Mode
SLACK_APP_TOKEN=xapp-your-app-token
```

または `~/.hermes/config.yaml` で:

```yaml
platforms:
  slack:
    token: "xoxb-workspace1-token,xoxb-workspace2-token"
```

### OAuthトークンファイル

環境や設定のトークンに加えて、Hermesは次の場所にある**OAuthトークンファイル**からもトークンを読み込みます:

```
~/.hermes/slack_tokens.json
```

このファイルは、チームIDをトークンエントリにマッピングするJSONオブジェクトです:

```json
{
  "T01ABC2DEF3": {
    "token": "xoxb-workspace-token-here",
    "team_name": "My Workspace"
  }
}
```

このファイルのトークンは、`SLACK_BOT_TOKEN` で指定されたトークンとマージされます。重複したトークンは自動的に重複排除されます。

### 仕組み

- リスト内の**最初のトークン**がプライマリトークンで、Socket Mode接続（AsyncApp）に使用されます。
- 各トークンは起動時に `auth.test` で認証されます。ゲートウェイは各 `team_id` をそれぞれの `WebClient` と `bot_user_id` にマッピングします。
- メッセージが到着すると、Hermesは正しいワークスペース固有のクライアントを使って応答します。
- プライマリの `bot_user_id`（最初のトークン由来）は、単一のボットアイデンティティを前提とする機能との後方互換性のために使用されます。

---

## 音声メッセージ

HermesはSlackで音声をサポートします:

- **受信:** 音声/オーディオメッセージは、設定されたSTTプロバイダー（ローカルの `faster-whisper`、Groq Whisper（`GROQ_API_KEY`）、またはOpenAI Whisper（`VOICE_TOOLS_OPENAI_KEY`））を使って自動的に文字起こしされます
- **送信:** TTS応答はオーディオファイルの添付として送信されます

---

## チャンネルごとのプロンプト

特定のSlackチャンネルにエフェメラルなシステムプロンプトを割り当てます。プロンプトはすべてのターンで実行時に注入され、トランスクリプト履歴には決して永続化されないため、変更は即座に有効になります。

```yaml
slack:
  channel_prompts:
    "C01RESEARCH": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "C02ENGINEERING": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

キーはSlackのチャンネルID（チャンネル詳細 → "About" → 一番下までスクロールで見つけられます）です。一致するチャンネル内のすべてのメッセージ（スラッシュコマンドのターン — `/hermes <question>` とネイティブの `/<command>` — を含む）には、エフェメラルなシステム指示としてプロンプトが注入されます。

## チャンネルごとのスキルバインディング

特定のチャンネルまたはDMで新しいセッションが始まるたびに、スキルを自動的に読み込みます。チャンネルごとのプロンプト（すべてのターンで注入される）とは異なり、スキルバインディングは**セッション開始時**にスキルのコンテンツをユーザーメッセージとして注入します — それは会話履歴の一部となり、以降のターンで再読み込みする必要はありません。

これは、モデル自身のスキルセレクターに、短い返信のたびに読み込むかどうかを判断させたくない、専用の目的（フラッシュカード、ドメイン固有のQ&Aボット、サポートのトリアージチャンネルなど）を持つDMやチャンネルに最適です。

```yaml
slack:
  channel_skill_bindings:
    # DM channel — always runs in "german-flashcards" mode
    - id: "D0ATH9TQ0G6"
      skills:
        - german-flashcards
    # Research channel — preload multiple skills in order
    - id: "C01RESEARCH"
      skills:
        - arxiv
        - writing-plans
    # Short form: single skill as a string
    - id: "C02SUPPORT"
      skill: hubspot-on-demand
```

メモ:
- バインディングはチャンネルIDで一致します。バインドされたチャンネル内のスレッドメッセージについては、スレッドは親チャンネルのバインディングを継承します。スラッシュコマンド（`/hermes <question>`）で開始されたセッションもバインディングを読み込みます。
- スキルはセッション開始時（新しいセッション）にのみ読み込まれます。バインディングを変更した場合は、`/new` を実行すると有効になります。
- スキルの指示の上にチャンネルごとのトーン/制約を加えるには、`channel_prompts` と組み合わせてください。

## トラブルシューティング

| Problem | Solution |
|---------|----------|
| ボットがDMに応答しない | `message.im` がイベントサブスクリプションに含まれていること、アプリが再インストールされていることを確認 |
| ボットはDMでは動くがチャンネルでは動かない | **最もよくある問題。** `message.channels` と `message.groups` をイベントサブスクリプションに追加し、アプリを再インストールし、`/invite @Hermes Agent` でボットをチャンネルに招待 |
| ボットがチャンネルで@メンションに応答しない | 1) `message.channels` イベントがサブスクライブされているか確認。2) ボットがチャンネルに招待されている必要がある。3) `channels:history` スコープが追加されていることを確認。4) スコープ/イベント変更後にアプリを再インストール |
| ボットがプライベートチャンネルのメッセージを無視する | `message.groups` イベントサブスクリプションと `groups:history` スコープの両方を追加し、アプリを再インストールしてボットを `/invite` |
| ボットがグループDM（複数人DM）で応答しない | `message.mpim` イベントサブスクリプションと `mpim:history` スコープ（および `mpim:read`）を追加し、アプリを **再インストール**。`message.mpim` がないと、1:1のDMは動作していても、SlackはグループDMのメッセージをボットに一切配信しません。 |
| DMで "Sending messages to this app has been turned off" | App Home設定で**Messagesタブ**を有効化（ステップ5を参照） |
| "not_authed" または "invalid_auth" エラー | Bot TokenとApp Tokenを再生成し、`.env` を更新 |
| ボットは応答するがチャンネルに投稿できない | `/invite @Hermes Agent` でボットをチャンネルに招待 |
| ボットはチャットできるがアップロードされた画像/ファイルを読めない | `files:read` を追加し、アプリを**再インストール**。Slackがスコープ/認証/権限の失敗を返したとき、Hermesは添付ファイルアクセスの診断をチャット内に表示するようになりました。 |
| `missing_scope` エラー | OAuth & Permissionsで必要なスコープを追加し、アプリを**再インストール** |
| ソケットが頻繁に切断される | ネットワークを確認。Boltは自動再接続しますが、不安定な接続は遅延を引き起こします |
| スコープ/イベントを変更したが何も変わらない | スコープまたはイベントサブスクリプションを変更した後は、アプリをワークスペースに**再インストールする必要があります** |

### クイックチェックリスト

ボットがチャンネルで動作しない場合は、次の**すべて**を確認してください:

1. ✅ `message.channels` イベントがサブスクライブされている（パブリックチャンネル用）
2. ✅ `message.groups` イベントがサブスクライブされている（プライベートチャンネル用）
3. ✅ `app_mention` イベントがサブスクライブされている
4. ✅ `channels:history` スコープが追加されている（パブリックチャンネル用）
5. ✅ `groups:history` スコープが追加されている（プライベートチャンネル用）
6. ✅ スコープ/イベントを追加した後にアプリが**再インストール**された
7. ✅ ボットがチャンネルに**招待**された（`/invite @Hermes Agent`）
8. ✅ メッセージでボットを**@メンション**している

---

## セキュリティ

:::warning
**必ず `SLACK_ALLOWED_USERS` を設定**し、認可されたユーザーのメンバーIDを指定してください。この設定がないと、ゲートウェイは安全策としてデフォルトで**すべてのメッセージを拒否**します。ボットトークンを決して共有しないでください — パスワードと同じように扱ってください。
:::

- トークンは `~/.hermes/.env` に保存すべきです（ファイル権限 `600`）
- Slackアプリ設定を通じて定期的にトークンをローテーションしてください
- 誰がHermes設定ディレクトリにアクセスできるかを監査してください
- Socket Modeは公開エンドポイントが公開されないことを意味します — 攻撃対象領域が1つ減ります
