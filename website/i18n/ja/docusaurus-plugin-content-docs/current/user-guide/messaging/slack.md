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

1. マニフェストを生成します:
   ```bash
   hermes slack manifest --write
   ```
   これは `~/.hermes/slack-manifest.json` を書き出し、貼り付け用の手順を表示します。
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
# 必須
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_ALLOWED_USERS=U01ABC2DEF3              # カンマ区切りのメンバーID

# 任意
SLACK_HOME_CHANNEL=C01234567890              # cron/スケジュールされたメッセージのデフォルトチャンネル
SLACK_HOME_CHANNEL_NAME=general              # ホームチャンネルの人間が読める名前（任意）
```

または、対話型セットアップを実行します:

```bash
hermes gateway setup    # プロンプトが表示されたらSlackを選択
```

その後、ゲートウェイを起動します:

```bash
hermes gateway              # フォアグラウンド
hermes gateway install      # ユーザーサービスとしてインストール
sudo hermes gateway install --system   # Linuxのみ: 起動時のシステムサービス
```

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

古いマニフェストとの後方互換性のため、`/hermes btw run the tests` と入力することもできます — Hermesは `/btw run the tests` と同じ方法でルーティングします。自由形式の質問も機能します: `/hermes what's the weather?` は通常のメッセージとして扱われます。

### 高度な使い方: スラッシュコマンド配列のみを出力する

Slackマニフェストを手動で管理していて、スラッシュコマンドのリストだけが欲しい場合:

```bash
hermes slack manifest --slashes-only > /tmp/slashes.json
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
    # 複数パートの応答をどうスレッド化するかを制御
    # "off"   — 元のメッセージに返信をスレッド化しない
    # "first" — 最初のチャンクをユーザーのメッセージにスレッド化（デフォルト）
    # "all"   — すべてのチャンクをユーザーのメッセージにスレッド化
    reply_to_mode: "first"

    extra:
      # スレッドで返信するかどうか（デフォルト: true）。
      # false の場合、チャンネルメッセージはスレッドではなく
      # チャンネルへの直接返信を受け取る。既存のスレッド内の
      # メッセージは引き続きスレッド内で返信する。
      reply_in_thread: true

      # スレッド返信をメインチャンネルにも投稿
      # （Slackの "Also send to channel" 機能）。
      # 最初の返信の最初のチャンクのみがブロードキャストされる。
      reply_broadcast: false
```

| Key | Default | Description |
|-----|---------|-------------|
| `platforms.slack.reply_to_mode` | `"first"` | 複数パートメッセージのスレッド化モード: `"off"`、`"first"`、または `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | `false` の場合、チャンネルメッセージはスレッドではなく直接返信を受け取ります。既存のスレッド内のメッセージは引き続きスレッド内で返信します。 |
| `platforms.slack.extra.reply_broadcast` | `false` | `true` の場合、スレッド返信はメインチャンネルにも投稿されます。最初のチャンクのみがブロードキャストされます。 |

### セッションの分離

```yaml
# グローバル設定 — Slackと他のすべてのプラットフォームに適用
group_sessions_per_user: true
```

`true`（デフォルト）の場合、共有チャンネル内の各ユーザーは、それぞれ分離された会話セッションを持ちます。`#general` でHermesと話す2人は、別々の履歴とコンテキストを持ちます。

チャンネル全体が1つの会話セッションを共有する協調モードが必要な場合は `false` に設定します。これはユーザーがコンテキストの増大とトークンコストを共有し、あるユーザーの `/reset` が全員のセッションをクリアすることを意味する点に注意してください。

### メンションとトリガーの動作

```yaml
slack:
  # チャンネルで@メンションを必須にする（これがデフォルトの動作。
  # Slackアダプターはいずれにせよチャンネルで@メンションゲーティングを
  # 強制するが、他のプラットフォームとの一貫性のため明示的に設定できる）
  require_mention: true

  # スレッドの自動エンゲージメントを防止: 明示的な@メンションを含む
  # チャンネルメッセージにのみ返信する。これがOFF（デフォルト）の場合、
  # Slackは「自動エンゲージ」できる — スレッド内の過去のメンションを記憶し、
  # ボットメッセージへの返信をフォローアップし、新しいメンションなしで
  # アクティブなセッションを再開する。strict_mention をONにすると、
  # すべての新しいチャンネルメッセージは、Hermesが応答する前にボットを
  # @メンションする必要がある。
  strict_mention: false

  # ボットをトリガーするカスタムメンションパターン
  # （デフォルトの@メンション検出に加えて）
  mention_patterns:
    - "hey hermes"
    - "hermes,"

  # すべての送信メッセージの先頭に付加されるテキスト
  reply_prefix: ""
```

:::tip `strict_mention` を使うべきとき
Slackのデフォルトの「ボットがこのスレッドを記憶する」動作がユーザーを驚かせる、混雑したワークスペースでは `true` に設定してください — 例えば、ボットが最初に手助けした長い技術サポートスレッドで、明示的に再びpingされない限り静かにしていてほしい場合です。DMとアクティブな対話セッションは影響を受けません。
:::

:::info
Slackは両方のパターンをサポートします: デフォルトでは会話を始めるのに `@mention` が必要ですが、`SLACK_FREE_RESPONSE_CHANNELS`（カンマ区切りのチャンネルID）または `config.yaml` の `slack.free_response_channels` で特定のチャンネルを除外できます。ボットがスレッド内でアクティブなセッションを持つと、以降のスレッド返信にメンションは不要です。DMでは、ボットはメンションを必要とせずに常に応答します。
:::

### 認可されていないユーザーの取り扱い

```yaml
slack:
  # 認可されていないユーザー（SLACK_ALLOWED_USERSにいない）がボットにDMしたときの動作
  # "pair"   — ペアリングコードを促す（デフォルト）
  # "ignore" — メッセージを黙って破棄
  unauthorized_dm_behavior: "pair"
```

これをすべてのプラットフォームに対してグローバルに設定することもできます:

```yaml
unauthorized_dm_behavior: "pair"
```

`slack:` の下のプラットフォーム固有の設定が、グローバル設定より優先されます。

### 音声の文字起こし

```yaml
# グローバル設定 — 受信した音声メッセージの自動文字起こしの有効/無効
stt_enabled: true
```

`true`（デフォルト）の場合、受信した音声メッセージは、エージェントによって処理される前に、設定されたSTTプロバイダーを使って自動的に文字起こしされます。

### 完全な例

```yaml
# グローバルゲートウェイ設定
group_sessions_per_user: true
unauthorized_dm_behavior: "pair"
stt_enabled: true

# Slack固有の設定
slack:
  require_mention: true
  unauthorized_dm_behavior: "pair"

# プラットフォーム設定
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

---

## マルチワークスペースのサポート

Hermesは、単一のゲートウェイインスタンスを使って**複数のSlackワークスペース**に同時に接続できます。各ワークスペースは、それぞれのボットユーザーIDで独立して認証されます。

### 設定

`SLACK_BOT_TOKEN` に**カンマ区切りのリスト**として複数のボットトークンを指定します:

```bash
# 複数のボットトークン — ワークスペースごとに1つ
SLACK_BOT_TOKEN=xoxb-workspace1-token,xoxb-workspace2-token,xoxb-workspace3-token

# Socket Modeには引き続き単一のapp-levelトークンが使われる
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

キーはSlackのチャンネルID（チャンネル詳細 → "About" → 一番下までスクロールで見つけられます）です。一致するチャンネル内のすべてのメッセージは、エフェメラルなシステム指示としてプロンプトが注入されます。

## チャンネルごとのスキルバインディング

特定のチャンネルまたはDMで新しいセッションが始まるたびに、スキルを自動的に読み込みます。チャンネルごとのプロンプト（すべてのターンで注入される）とは異なり、スキルバインディングは**セッション開始時**にスキルのコンテンツをユーザーメッセージとして注入します — それは会話履歴の一部となり、以降のターンで再読み込みする必要はありません。

これは、モデル自身のスキルセレクターに、短い返信のたびに読み込むかどうかを判断させたくない、専用の目的（フラッシュカード、ドメイン固有のQ&Aボット、サポートのトリアージチャンネルなど）を持つDMやチャンネルに最適です。

```yaml
slack:
  channel_skill_bindings:
    # DMチャンネル — 常に "german-flashcards" モードで実行
    - id: "D0ATH9TQ0G6"
      skills:
        - german-flashcards
    # Researchチャンネル — 複数のスキルを順番にプリロード
    - id: "C01RESEARCH"
      skills:
        - arxiv
        - writing-plans
    # 短い形式: 文字列としての単一スキル
    - id: "C02SUPPORT"
      skill: hubspot-on-demand
```

メモ:
- バインディングはチャンネルIDで一致します。バインドされたチャンネル内のスレッドメッセージについては、スレッドは親チャンネルのバインディングを継承します。
- スキルはセッション開始時（新しいセッションまたは自動リセット後）にのみ読み込まれます。バインディングを変更した場合は、`/new` を実行するか、セッションが自動リセットされるのを待つと有効になります。
- スキルの指示の上にチャンネルごとのトーン/制約を加えるには、`channel_prompts` と組み合わせてください。

## トラブルシューティング

| Problem | Solution |
|---------|----------|
| ボットがDMに応答しない | `message.im` がイベントサブスクリプションに含まれていること、アプリが再インストールされていることを確認 |
| ボットはDMでは動くがチャンネルでは動かない | **最もよくある問題。** `message.channels` と `message.groups` をイベントサブスクリプションに追加し、アプリを再インストールし、`/invite @Hermes Agent` でボットをチャンネルに招待 |
| ボットがチャンネルで@メンションに応答しない | 1) `message.channels` イベントがサブスクライブされているか確認。2) ボットがチャンネルに招待されている必要がある。3) `channels:history` スコープが追加されていることを確認。4) スコープ/イベント変更後にアプリを再インストール |
| ボットがプライベートチャンネルのメッセージを無視する | `message.groups` イベントサブスクリプションと `groups:history` スコープの両方を追加し、アプリを再インストールしてボットを `/invite` |
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
