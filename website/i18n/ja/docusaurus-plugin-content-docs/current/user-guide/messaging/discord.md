---
sidebar_position: 3
title: "Discord"
description: "Hermes AgentをDiscordボットとしてセットアップする"
---

# Discordセットアップ

Hermes AgentはDiscordとボットとして統合され、ダイレクトメッセージやサーバーチャンネルを通じてAIアシスタントとチャットできます。ボットはあなたのメッセージを受信し、Hermes Agentのパイプライン（ツールの使用、メモリ、推論を含む）で処理し、リアルタイムで応答します。テキスト、ボイスメッセージ、ファイル添付、スラッシュコマンドをサポートします。

セットアップの前に、多くの人が知りたい部分を説明します。Hermesがサーバーに参加した後、どう振る舞うかです。

## Hermesの振る舞い

| コンテキスト | 振る舞い |
|---------|----------|
| **DM** | Hermesはすべてのメッセージに応答します。`@mention`は不要です。各DMは独自のセッションを持ちます。 |
| **サーバーチャンネル** | デフォルトでは、Hermesは`@mention`されたときだけ応答します。メンションせずにチャンネルに投稿した場合、Hermesはそのメッセージを無視します。 |
| **フリーレスポンスチャンネル** | `DISCORD_FREE_RESPONSE_CHANNELS`で特定のチャンネルをメンション不要にしたり、`DISCORD_REQUIRE_MENTION=false`でメンションをグローバルに無効化したりできます。これらのチャンネルのメッセージはデフォルトでインラインで応答されます — 自動スレッド化はスキップされるため、チャンネルは軽量なチャットのままです。`discord.free_response_auto_thread: true`を設定すると、メンション不要の返信とトップレベルメッセージごとのスレッドの両方が得られます。 |
| **スレッド** | Hermesは同じスレッド内で応答します。そのスレッドまたはその親チャンネルがフリーレスポンスとして設定されていない限り、メンションのルールは引き続き適用されます。スレッドは、セッション履歴については親チャンネルから分離されたままです。 |
| **複数ユーザーがいる共有チャンネル** | デフォルトでは、Hermesは安全性と明確さのため、チャンネル内のユーザーごとにセッション履歴を分離します。同じチャンネルで会話する2人は、明示的に無効化しない限り、1つのトランスクリプトを共有しません。 |
| **他のユーザーをメンションするメッセージ** | `DISCORD_IGNORE_NO_MENTION`が`true`（デフォルト）のとき、メッセージが他のユーザーを@mentionするがボットをメンション**しない**場合、Hermesは沈黙します。これにより、他人に向けられた会話にボットが割り込むのを防ぎます。誰がメンションされているかに関係なくすべてのメッセージに応答させたい場合は`false`に設定します。これはサーバーチャンネルにのみ適用され、DMには適用されません。 |

:::tip
タグ付けせずに毎回Hermesと話せる通常のボットヘルプチャンネルが欲しい場合は、そのチャンネルを`DISCORD_FREE_RESPONSE_CHANNELS`に追加してください。
:::

### Discordゲートウェイモデル

Discord上のHermesは、ステートレスに応答するWebhookではありません。完全なメッセージングゲートウェイを通じて動作し、つまり受信した各メッセージは次を通過します。

1. 認可（`DISCORD_ALLOWED_USERS`）
2. メンション / フリーレスポンスのチェック
3. セッションの検索
4. セッショントランスクリプトのロード
5. ツール、メモリ、スラッシュコマンドを含む通常のHermesエージェント実行
6. Discordへの応答配信

これが重要なのは、混雑したサーバーでの振る舞いが、Discordのルーティングとセッションポリシーの両方に依存するからです。

### Discordでのセッションモデル {#session-model-in-discord}

デフォルトでは:

- 各DMは独自のセッションを持つ
- 各サーバースレッドは独自のセッション名前空間を持つ
- 共有チャンネル内の各ユーザーは、そのチャンネル内で独自のセッションを持つ

つまり、AliceとBobが両方とも`#research`でHermesと話す場合、同じ見えているDiscordチャンネルを使っていても、Hermesはデフォルトでそれらを別々の会話として扱います。

これは`config.yaml`で制御されます。

```yaml
group_sessions_per_user: true
```

ルーム全体で1つの共有会話を明示的に望む場合にのみ`false`に設定します。

```yaml
group_sessions_per_user: false
```

共有セッションは協調的なルームに役立つことがありますが、次のことも意味します。

- ユーザーはコンテキストの増大とトークンコストを共有する
- 1人のツールを多用する長いタスクが、他の全員のコンテキストを膨らませる
- 1人の進行中の実行が、同じルームの別の人のフォローアップを中断する可能性がある

### 割り込みと並行性

Hermesは実行中のエージェントをセッションキーで追跡します。

デフォルトの`group_sessions_per_user: true`では:

- Aliceが自分の進行中のリクエストを割り込んでも、そのチャンネルでのAliceのセッションにのみ影響する
- Bobは、Aliceの履歴を引き継いだりAliceの実行を中断したりせず、同じチャンネルで話し続けられる

`group_sessions_per_user: false`では:

- ルーム全体が、そのチャンネル/スレッドの1つの実行中エージェントスロットを共有する
- 異なる人からのフォローアップメッセージが、互いを中断したり待ち行列に並んだりする可能性がある

このガイドでは、Discordの開発者ポータルでボットを作成することから最初のメッセージを送信することまで、完全なセットアッププロセスを順を追って説明します。

### Gateway WebSocketのヘルス

Discord RESTとGateway WebSocketは別々のトランスポートです。RESTの応答が成功しても（`fetch_user()`がHTTP 200を返す場合を含む）、ボットがまだGatewayイベントを受信できることの証明にはなりません。そのためHermesは、ready状態、クライアント/ソケットのクローズ状態、ソケットが開いているか、ハートビートACKの経過時間、有限のハートビートレイテンシー、そして — ディスパッチ側の指標が加わって以降は — 最後にパースされたGatewayイベントからの経過時間を組み合わせて判断します。

設定された回数だけ連続して不健全なサンプルが続くと、アダプターはリトライ可能な致命的イベントを1つ発行します。閉じたトランスポート（`socket_closed` / `client_closed`）は確定した死とみなされ、最初の不健全なサンプルで再接続を強制します。しきい値はソフトなシグナル（古いハートビートACK、レイテンシー、イベントの沈黙）にのみ適用されます — [#118487](https://github.com/NousResearch/hermes-agent/issues/118487)を参照してください。既存のゲートウェイ再接続ウォッチャーが新しいアダプターを作成します。Discordアダプターが2つ目の無制限な再接続ループを開始することはありません。

シークレットではないしきい値は`config.yaml`で設定します。

```yaml
discord:
  websocket_liveness_interval_seconds: 15
  websocket_liveness_failure_threshold: 2
  websocket_heartbeat_ack_max_age_seconds: 60
  websocket_max_latency_seconds: 30
  websocket_event_max_silence_seconds: 14400
```

古い`liveness_interval_seconds`と`liveness_failure_threshold`という名前は互換性のためのエイリアスとしてのみ残っています。もはやRESTによるプローブを意味しません。

いずれかの設定値が`0`の場合、WebSocketのライブネスプローブ全体が無効になります。正の数としてパースできない値（例: `15s`、`nan`、`true`、`-1`）もプローブを無効にし、アダプターが起動するたびに警告をログに出します — プローブが動作していないように見える場合は`gateway.log`を確認してください。

`websocket_event_max_silence_seconds`は例外です。これは単一の指標（イベントのディスパッチ）だけを監視するため、`0`は**そのチェックだけ**をオプトアウトします — ready/ACK/レイテンシーの監視は続きます。ソケットはESTABLISHEDのまま、ハートビートにACKを返し続けながら、Gatewayイベントを1つも配信しないことがあります。ハートビートACKはイベント型を持たないフレームなので、トランスポート側のチェックではこの状態を検知できません。デフォルト（4時間）は、運用者が実際に観測した障害時間に合わせています。静かなギルドでは正当に何時間もGatewayイベントが1つも来ないことがあるため、トラフィックを把握していない限り、この上限は余裕を持たせておいてください。

:::tip[近道: ステップ5〜7はウィザードに任せる]
アプリケーションを作成してボットトークンをコピーしたら（ステップ1〜4）、`hermes gateway setup`を実行して**Discord**を選びます。HermesはDiscordでトークンを検証し、**Message Content Intent**がオフならそれを知らせ（トグルへの直接リンク付き）、サーバー用のすぐ使える招待リンクを表示し、あなたをボットのオーナーとして許可リストに追加します — 開発者モードは不要です。
:::

## ステップ1: Discordアプリケーションを作成する

1. [Discord開発者ポータル](https://discord.com/developers/applications)にアクセスし、Discordアカウントでサインインします。
2. 右上の**New Application**をクリックします。
3. アプリケーションの名前（例: "Hermes Agent"）を入力し、Developer Terms of Serviceに同意します。
4. **Create**をクリックします。

**General Information**ページに移動します。**Application ID**をメモしてください — 後で招待URLを構築するために必要になります。

## ステップ2: ボットを作成する

1. 左サイドバーで**Bot**をクリックします。
2. Discordがアプリケーションのボットユーザーを自動的に作成します。ボットのユーザー名が表示され、カスタマイズできます。
3. **Authorization Flow**の下で:
   - **Public Bot**を**ON**に設定します — Discordが提供する招待リンクを使うために必要です（推奨）。これにより、Installationタブがデフォルトの認可URLを生成できます。
   - **Require OAuth2 Code Grant**は**OFF**のままにします。

:::tip
このページで、ボットのカスタムアバターとバナーを設定できます。これはユーザーがDiscordで目にするものです。
:::

:::info[プライベートボットの代替案]
ボットをプライベートに保ちたい場合（Public Bot = OFF）、Installationタブの代わりにステップ5の**手動URL**方式を**必ず**使ってください。Discordが提供するリンクは、Public Botが有効になっている必要があります。
:::

## ステップ3: 特権ゲートウェイインテントを有効化する

これはセットアップ全体で最も重要なステップです。Hermesは常にDiscordにメッセージ内容を要求するため、**Message Content Intent**がオフだと、Discordは**ボットの接続を拒否**し、ボットは一切オンラインになりません。

**Bot**ページで、**Privileged Gateway Intents**までスクロールします。3つのトグルが表示されます。

| インテント | 目的 | 必須か？ |
|--------|---------|-----------| 
| **Presence Intent** | ユーザーのオンライン/オフライン状態を確認する | 任意 |
| **Server Members Intent** | メンバーリストにアクセスし、ユーザー名を解決する | **必須** |
| **Message Content Intent** | メッセージのテキスト内容を読み取る | **必須** |

両方を**ON**に切り替えて、**Server Members IntentとMessage Content Intentを有効化**します。

- **Message Content Intent**がないと、Discordは接続を拒否します。`gateway.log`には「Discord rejected the connection because privileged Gateway Intents are not enabled」と表示されます。
- **Server Members Intent**がないと、ボットは許可ユーザーリストのためにユーザー名を解決できず、誰がメッセージを送っているかを識別できないことがあります。

:::warning[これはDiscordボットが動かない理由の第1位です]
ボットがオフラインのままで、`gateway.log`が特権Gateway Intentsに言及している場合、**Message Content Intent**が無効になっています。[開発者ポータル](https://discord.com/developers/applications)に戻り、アプリケーションを選択 → Bot → Privileged Gateway Intentsを開き、**Message Content Intent**がONになっていることを確認してください。**Save Changes**をクリックします。
:::

**サーバー数について:**
- ボットが**100未満のサーバー**にいる場合、インテントを自由にオン/オフできます。
- ボットが**100以上のサーバー**にいる場合、特権インテントを使うには、Discordへの検証申請の提出が必要です。個人利用では、これは問題になりません。

ページ下部の**Save Changes**をクリックします。

## ステップ4: ボットトークンを取得する

ボットトークンは、Hermes Agentがボットとしてログインするために使う認証情報です。**Bot**ページで:

1. **Token**セクションの下で、**Reset Token**をクリックします。
2. Discordアカウントで二要素認証を有効にしている場合、2FAコードを入力します。
3. Discordが新しいトークンを表示します。**すぐにコピーしてください。**

:::warning[トークンは一度しか表示されません]
トークンは一度しか表示されません。失った場合は、リセットして新しいものを生成する必要があります。トークンを公開で共有したりGitにコミットしたりしないでください — このトークンを持つ者は誰でも、ボットを完全に制御できます。
:::

トークンを安全な場所（たとえばパスワードマネージャー）に保存してください。ステップ8で必要になります。

## ステップ5: 招待URLを生成する

ボットをサーバーに招待するには、OAuth2 URLが必要です。これには2つの方法があります。

### オプションA: Installationタブを使う（推奨）

:::note[Public Botが必要]
この方式は、ステップ2で**Public Bot**を**ON**に設定する必要があります。Public BotをOFFに設定した場合は、代わりに下記の手動URL方式を使ってください。
:::

1. 左サイドバーで**Installation**をクリックします。
2. **Installation Contexts**の下で、**Guild Install**を有効化します。
3. **Install Link**で、**Discord Provided Link**を選択します。
4. Guild Installの**Default Install Settings**の下で:
   - **Scopes**: `bot`と`applications.commands`を選択します
   - **Permissions**: 下記に列挙された権限を選択します。

### オプションB: 手動URL

次の形式を使って、招待URLを直接構築できます。

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=309237763136
```

`YOUR_APP_ID`を、ステップ1のApplication IDに置き換えてください。

### 必要な権限

これらは、ボットが必要とする最小限の権限です。

- **View Channels** — アクセスできるチャンネルを見る
- **Send Messages** — あなたのメッセージに応答する
- **Embed Links** — リッチな応答をフォーマットする
- **Attach Files** — 画像、音声、ファイル出力を送る
- **Read Message History** — 会話のコンテキストを維持する

### 推奨される追加の権限

- **Create Public Threads** — `/thread`と自動スレッド化で分離された会話を作成する
- **Send Messages in Threads** — スレッドの会話で応答する
- **Add Reactions** — 確認のためにメッセージにリアクションする

### 権限の整数値

| レベル | 権限の整数値 | 含まれるもの |
|-------|-------------------|-----------------|
| 最小限 | `117760` | View Channels、Send Messages、Embed Links、Attach Files、Read Message History |
| 推奨 | `309237763136` | 上記すべてに加えて、Create Public Threads、Send Messages in Threads、Add Reactions |
| フル（`hermes gateway setup`が表示するもの） | `309240908864` | 推奨に加えて、ConnectとSpeak（ボイスチャンネル） |

既存のインストールは、新たに要求された権限を自動的には得ません。
古い推奨URLを使った場合は、上記のURLでボットを再招待して
**Create Public Threads**を付与してください。

## ステップ6: サーバーに招待する

1. 招待URL（Installationタブのもの、または構築した手動URL）をブラウザで開きます。
2. **Add to Server**ドロップダウンで、サーバーを選択します。
3. **Continue**、続いて**Authorize**をクリックします。
4. プロンプトが表示されたらCAPTCHAを完了します。

:::info
ボットを招待するには、Discordサーバーで**Manage Server**権限が必要です。ドロップダウンにサーバーが表示されない場合は、サーバー管理者に招待リンクを使ってもらってください。
:::

認可後、ボットはサーバーのメンバーリストに表示されます（Hermesゲートウェイを起動するまではオフラインと表示されます）。

## ステップ7: あなたのDiscordユーザーIDを見つける

Hermes Agentは、誰がボットと対話できるかを制御するために、あなたのDiscordユーザーIDを使います。見つけるには:

1. Discord（デスクトップまたはWebアプリ）を開きます。
2. **設定** → **詳細設定** → **開発者モード**を**ON**に切り替えます。
3. 設定を閉じます。
4. 自分のユーザー名（メッセージ、メンバーリスト、またはプロフィール上）を右クリック → **IDをコピー**します。

あなたのユーザーIDは、`284102345871466496`のような長い数字です。

:::tip
開発者モードでは、同じ方法で**チャンネルID**と**サーバーID**もコピーできます — チャンネル名やサーバー名を右クリックして「IDをコピー」を選択します。ホームチャンネルを手動で設定したい場合は、チャンネルIDが必要になります。
:::

## ステップ8: Hermes Agentを設定する

### オプションA: インタラクティブセットアップ（推奨）

ガイド付きのセットアップコマンドを実行します。

```bash
hermes gateway setup
```

プロンプトが表示されたら**Discord**を選択し、ボットトークンを貼り付けます。ウィザードはDiscordでトークンを検証し（誤ったトークンや古いトークンは保存前に拒否されます）、特権インテントを確認し、招待リンクを表示し、ボットのオーナー — あなた — を許可リストに追加するよう提案します。追加のユーザーはユーザーIDまたはユーザー名で追加できます。

### オプションB: 手動設定

`~/.hermes/.env`ファイルに次を追加します。

```bash
# Required
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=284102345871466496

# Multiple allowed users (comma-separated)
# DISCORD_ALLOWED_USERS=284102345871466496,198765432109876543
```

次にゲートウェイを起動します。

```bash
hermes gateway
```

ボットは数秒以内にDiscordでオンラインになるはずです。テストのため、メッセージを送ってください — DMでも、ボットが見られるチャンネルでも構いません。

:::tip
`hermes gateway`をバックグラウンドで、またはsystemdサービスとして実行すると、永続的に動作させられます。詳細はデプロイのドキュメントを参照してください。
:::

## 設定リファレンス

Discordの振る舞いは2つのファイルで制御されます: 認証情報と環境レベルのトグル用の**`~/.hermes/.env`**と、構造化された設定用の**`~/.hermes/config.yaml`**です。両方が設定されている場合、環境変数が常にconfig.yamlの値より優先されます。

### 環境変数（`.env`）

| 変数 | 必須 | デフォルト | 説明 |
|----------|----------|---------|-------------|
| `DISCORD_BOT_TOKEN` | **はい** | — | [Discord開発者ポータル](https://discord.com/developers/applications)からのボットトークン。 |
| `DISCORD_ALLOWED_USERS` | 条件付き | — | ボットと対話することを許可されたDiscordユーザーIDのカンマ区切り。これ**または**`DISCORD_ALLOWED_ROLES`がないと、`DISCORD_ALLOW_ALL_USERS=true`、`GATEWAY_ALLOW_ALL_USERS=true`、または`DISCORD_ALLOWED_CHANNELS`でギルドへのアクセスを明示的にスコープしていない限り、ゲートウェイはすべてのユーザーを拒否します。ユーザー名も使えます: ボットの接続時にIDへ解決されます（**Server Members Intent**が必要）。表示名やサーバーニックネームは、どのメンバーでも設定できるため、決して照合されません。 |
| `DISCORD_ALLOWED_ROLES` | いいえ | — | DiscordロールIDのカンマ区切り。これらのロールのいずれかを持つメンバーは認可されます — `DISCORD_ALLOWED_USERS`とのOR論理です。接続時に**Server Members Intent**を自動有効化します。モデレーションチームが入れ替わる場合に便利です: 新しいモデレーターは、ロールが付与されるとすぐにアクセスでき、設定のプッシュは不要です。 |
| `DISCORD_ALLOW_ALL_USERS` | いいえ | `false` | ボットに到達できるすべてのDiscordユーザーを許可する明示的なオプトイン。Discordについてのみ、0.18より前のオープンな動作を復元します。信頼できる/プライベートなギルドや開発用途でのみ使ってください。 |
| `GATEWAY_ALLOW_ALL_USERS` | いいえ | `false` | すべてのゲートウェイプラットフォームに対するグローバルな全許可のオプトイン。接続されたすべてのプラットフォームを意図的にオープンにしたい場合を除き、プラットフォーム固有の`DISCORD_ALLOW_ALL_USERS`を優先してください。 |
| `DISCORD_HOME_CHANNEL` | いいえ | — | ボットがプロアクティブなメッセージ（cron出力、リマインダー、通知）を送るチャンネルID。 |
| `DISCORD_HOME_CHANNEL_NAME` | いいえ | `"Home"` | ログとステータス出力でのホームチャンネルの表示名。 |
| `DISCORD_COMMAND_SYNC_POLICY` | いいえ | `"safe"` | ネイティブのスラッシュコマンドの起動時同期を制御します。`"safe"`は既存のグローバルコマンドを差分比較し、変更されたものだけを更新し、Discordのメタデータの変更がパッチで適用できない場合はコマンドを再作成します。`"bulk"`は古い`tree.sync()`の動作を維持します。`"off"`は起動時同期を完全にスキップします。 |
| `DISCORD_REQUIRE_MENTION` | いいえ | `true` | `true`のとき、ボットはサーバーチャンネルで`@mention`されたときだけ応答します。すべてのチャンネルのすべてのメッセージに応答させるには`false`に設定します。 |
| `DISCORD_THREAD_REQUIRE_MENTION` | いいえ | `false` | `true`のとき、スレッド内のメンション省略が無効になります — スレッドはチャンネルと同じようにゲートされ、ボットがすでに参加した後でも`@mention`が必要になります。複数のボットが1つのスレッドを共有し、それぞれが明示的な`@mention`のときだけ反応するようにしたい場合に使います。 |
| `DISCORD_FREE_RESPONSE_CHANNELS` | いいえ | — | `DISCORD_REQUIRE_MENTION`が`true`であっても、`@mention`を要求せずにボットが応答するチャンネルIDのカンマ区切り。 |
| `DISCORD_IGNORE_NO_MENTION` | いいえ | `true` | `true`のとき、メッセージが他のユーザーを`@mention`するがボットをメンション**しない**場合、ボットは沈黙します。他人に向けられた会話にボットが割り込むのを防ぎます。サーバーチャンネルにのみ適用され、DMには適用されません。 |
| `DISCORD_AUTO_THREAD` | いいえ | `true` | `true`のとき、テキストチャンネルでの`@mention`ごとに新しいスレッドを自動作成し、各会話が分離されるようにします（Slackの動作に似ています）。すでにスレッド内またはDMにあるメッセージは影響を受けません。 |
| `DISCORD_FREE_RESPONSE_AUTO_THREAD` | いいえ | `false` | `true`のとき、フリーレスポンスチャンネル（`DISCORD_FREE_RESPONSE_CHANNELS`に列挙されたもの）でも、メンション不要のまま、トップレベルメッセージごとにスレッドを自動作成します。デフォルトの`false`は軽量なインラインチャットの動作を維持します。`DISCORD_AUTO_THREAD=true`が必要です。`DISCORD_NO_THREAD_CHANNELS`は引き続き優先され、ボイスに紐づいたチャンネルは常にこれを無視します。 |
| `DISCORD_ALLOW_BOTS` | いいえ | `"none"` | 他のDiscordボットやWebhookからのメッセージの扱いを制御します（CIやデプロイ通知など、`webhook_id`を持つ投稿は、Discordがボットフラグを省略していてもボットとして扱われます）。`"none"` — 他のすべてのボットを無視。`"mentions"` — Hermesを`@mention`するボットメッセージのみ受け付ける。`"all"` — すべてのボットメッセージを受け付ける。デフォルトでは、どちらの有効モードでもリテラルのインラインメンションが必要です。次の設定を参照してください。 |
| `DISCORD_BOTS_REQUIRE_INLINE_MENTION` | いいえ | `true` | ボットのハンドオフを開始するには、リテラルの`<@BOT_ID>` / `<@!BOT_ID>`トークンを必須にします。返信メタデータだけではハンドオフは開始しません。同じ送信者/チャンネルからの短い続きは、下記のとおり受け付けられます。従来の受け付け方が必要な信頼できるリレーの場合にのみ`false`に設定してください。人間のメッセージには影響しません。 |
| `DISCORD_REACTIONS` | いいえ | `true` | `true`のとき、ボットは処理中にメッセージへ絵文字リアクションを追加します（開始時に👀、成功時に✅、エラー時に❌）。リアクションを完全に無効化するには`false`に設定します。 |
| `DISCORD_IGNORED_CHANNELS` | いいえ | — | `@mention`されてもボットが**決して**応答しないチャンネルIDのカンマ区切り。他のすべてのチャンネル設定より優先されます。 |
| `DISCORD_ALLOWED_CHANNELS` | いいえ | — | チャンネルIDのカンマ区切り。設定すると、ボットはこれらのチャンネル（および許可されていればDM）で**のみ**応答します。`config.yaml`の`discord.allowed_channels`をオーバーライドします。`DISCORD_IGNORED_CHANNELS`と組み合わせて、許可/拒否のルールを表現できます。 |
| `DISCORD_NO_THREAD_CHANNELS` | いいえ | — | スレッドを作成する代わりに、チャンネル内で直接ボットが応答するチャンネルIDのカンマ区切り。`DISCORD_AUTO_THREAD`が`true`のときだけ関係します。 |
| `DISCORD_HISTORY_BACKFILL` | いいえ | `true` | `true`のとき、ボットがメンションされると、最近のチャンネルのスクロールバック（ボットの最後の応答以降）をユーザーメッセージの前に付加します。`require_mention`によってボットが見逃すはずのコンテキストを回復します。DMとフリーレスポンスチャンネルではスキップされます。無効化するには`false`に設定します。 |
| `DISCORD_HISTORY_BACKFILL_LIMIT` | いいえ | `50` | バックフィルブロックを組み立てる際に遡ってスキャンするメッセージの最大数。実際には、スキャンは通常それより早く — チャンネル内のボット自身の最後のメッセージで — 止まります。 |
| `DISCORD_REPLY_TO_MODE` | いいえ | `"first"` | 返信参照の振る舞いを制御します: `"off"` — 元のメッセージに決して返信しない、`"first"` — 最初のメッセージチャンクだけに返信参照（デフォルト）、`"all"` — すべてのチャンクに返信参照。 |
| `DISCORD_ALLOW_MENTION_EVERYONE` | いいえ | `false` | `false`（デフォルト）のとき、応答にそれらのトークンが含まれていても、ボットは`@everyone`や`@here`をpingできません。オプトインし直すには`true`に設定します。下記の[メンション制御](#mention-control)を参照してください。 |
| `DISCORD_ALLOW_MENTION_ROLES` | いいえ | `false` | `false`（デフォルト）のとき、ボットは`@role`メンションをpingできません。許可するには`true`に設定します。 |
| `DISCORD_ALLOW_MENTION_USERS` | いいえ | `true` | `true`（デフォルト）のとき、ボットは個々のユーザーをIDでpingできます。 |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | いいえ | `true` | `true`（デフォルト）のとき、メッセージへの返信は元の作成者をpingします。 |
| `DISCORD_PROXY` | いいえ | — | Discord接続（HTTP、WebSocket、REST）用のプロキシURL。`HTTPS_PROXY`/`ALL_PROXY`をオーバーライドします。`http://`、`https://`、`socks5://`のスキームをサポートします。 |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | いいえ | `false` | `true`のとき、ボットは（組み込みのPDF/テキスト/zip/Officeの許可リストに限らず）あらゆるファイル型の添付を受け付けます。未知の型はディスクにキャッシュされ、`application/octet-stream`のMIMEを持つローカルパスとしてエージェントに渡されるため、エージェントは`terminal` / `read_file` / `ffprobe`などで中身を調べられます。 |
| `DISCORD_MAX_ATTACHMENT_BYTES` | いいえ | `33554432` | ゲートウェイがダウンロードしてキャッシュする添付1つあたりの最大バイト数。デフォルトは32 MiB。上限なしにするには`0`に設定します（添付は書き込み中メモリに保持されるため、無制限には実際のメモリコストがかかります）。 |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | いいえ | `0.6` | キューに入ったテキストチャンクをフラッシュする前にアダプターが待つ猶予ウィンドウ。ストリーム出力をスムーズにするのに便利です。 |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | いいえ | `2.0` | 上限に近いDiscordの分割や、同じチャンネルで最近タグ付けされたボットからの続きのチャンクに対する、より長い静止期間。 |

### ボット間のハンドオフ: 一度タグ付けし、連続投稿をまとめて受け取る

ボットからの入力はオプトインのままです（デフォルトは`DISCORD_ALLOW_BOTS=none`）。`mentions`または`all`で有効にした場合、**ボットのハンドオフを開始するには、デフォルトでメッセージ内容にリテラルの`<@BOT_ID>` / `<@!BOT_ID>`が必要です**。Discordの自動的な返信pingだけではハンドオフは開始しません。人間が書いたメッセージは既存のメンションの振る舞いを維持します。

ボットのメンションが受け付けられた後、Hermesは**同じチャンネルまたはスレッド内の同じボット**からのメンションなしの続きを短時間受け付けます。テキストの続きは既存のテキストバッチャーに入るため、素早く分割された応答は、各パートでタグを繰り返さなくてもまとめてエージェントに届きます。特別な送信者プロトコルやパートマーカーは不要です。

たとえば、ボットAが`<@BOT_B_ID> Here is the review …`を送り、すぐに同じスレッドでタグなしのテキストチャンクを2つ続けたとします。ボットBは続きのウィンドウ内でそれらのチャンクを受け付け、タグ付きのテキストとまとめてバッチ処理します。ウィンドウが切れた後は、デフォルトのポリシーでは、タグなしのメッセージや返信pingで新たなハンドオフを開始できません。明示的な相互タグ付けは引き続きサポートされます。これは意図的に続けられる会話ではなく、偶発的な返信メタデータのループを防ぐためのものです。

#### タイミングと制限

受け付けウィンドウは、受け付けられたメンションの後`max(HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS, HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS)`だけ続き（デフォルトでは2秒）、受け付けられた続きのたびに再設定されるため、Discordの送信レートで送られる長いハンドオフも丸ごと届きます。これとは別に、キューに入った各テキストチャンクはバッチの静止タイマーを再スタートします。タグ付けされたボットのバッチは、最初のチャンクが短くても分割用の静止期間を使います。これらの設定は受信側のバッチ処理を制御するもので、送信側のペースを制御するものではありません。テキストのバッチ処理を無効にすると（`HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS=0`）、続きの受け付けも無効になります。

これは短い連続投稿のためのヒューリスティックであり、マルチパートの配信を保証するものではありません。ウィンドウ外で遅れて届いたチャンクは、デフォルトのポリシーではそれ自体にメンションが必要です。ウィンドウ内の同じボット・同じチャンネルからの無関係なメッセージも受け付けられることがあります。この例外はテキストに限定されません: 添付やコマンドも受け付けを通過することがありますが、このバッチャーに入るのはテキストだけで、他のメッセージ型は通常の処理に従います。既存のチャンネル制限と`DISCORD_ALLOW_BOTS=none`は引き続き適用されます。履歴/バックフィルの振る舞いは変わりません。ゲートウェイのボットループガードは最後の防御として残ります: 1つのチャンネルで5分以内にボットが書いたメッセージが20件に達すると、そのチャンネルでの以降のボットメッセージは10分間破棄されます（`config.yaml`の`gateway.bot_loop_guard`で調整可能。人間のメッセージは決してカウントされません）。

#### 信頼できるリレーとの互換性

インラインメンションの要件は、`DISCORD_ALLOW_BOTS=all`の場合も含め、デフォルトで**true**になりました（以前はfalse）。既存のリレーが意図的に返信pingやメンションなしのボットメッセージに依存している場合は、古い受け付け方を明示的に維持してください。

```yaml
discord:
  bots_require_inline_mention: false
```

YAMLのオプションが未設定の場合は、既存の`DISCORD_BOTS_REQUIRE_INLINE_MENTION=false`環境変数によるオーバーライドもサポートされます。このオプトアウトでは、`mentions`は返信pingを含むDiscordの解決済みメンションを受け付け、`all`はボット固有のメンション要件を取り除きます。その他のチャンネルのメンションルールは引き続き適用されます。この互換モードでは、受け付けられた返信pingも続きのウィンドウを開くことがあります。偶発的な返信ループのリスクが戻るため、信頼できるリレーにのみ使ってください。

### 設定ファイル（`config.yaml`）

`~/.hermes/config.yaml`の`discord`セクションは、上記の環境変数をミラーします。config.yamlの設定はデフォルトとして適用されます — 等価な環境変数がすでに設定されている場合は、環境変数が勝ちます。

```yaml
# Discord-specific settings
discord:
  require_mention: true           # Require @mention in server channels
  thread_require_mention: false   # If true, require @mention in threads too (multi-bot threads)
  bots_require_inline_mention: true  # Bot authors must type a literal @mention (default: true)
  free_response_channels: ""      # Comma-separated channel IDs (or YAML list)
  auto_thread: true               # Auto-create threads on @mention
  free_response_auto_thread: false # If true, free_response_channels also auto-thread (default: inline)
  reactions: true                 # Add emoji reactions during processing
  ignored_channels: []            # Channel IDs where bot never responds
  no_thread_channels: []          # Channel IDs where bot responds without threading
  history_backfill: true          # Prepend recent channel scrollback on mention (default: true)
  history_backfill_limit: 50      # Max messages to scan backwards (default: 50)
  missed_message_backfill:        # Replay messages missed while disconnected (opt-in)
    enabled: false
    channels: []                  # Empty uses free_response_channels
    window_seconds: 21600         # Look back at most 6 hours
    limit: 100                    # Global scan cap per reconnect
    max_dispatches: 10            # Recovery dispatch cap per reconnect
    max_attempts: 3               # Lifetime re-dispatch cap per message
  channel_prompts: {}             # Per-channel ephemeral system prompts
  voice_channel_inactivity_timeout_seconds: 300  # Set 0 to stay in VC until explicit /voice leave
  voice_playback_timeout_seconds: 120             # Minimum playback watchdog; long clips get duration+padding
  allow_mentions:                 # What the bot is allowed to ping (safe defaults)
    everyone: false               # @everyone / @here pings (default: false)
    roles: false                  # @role pings (default: false)
    users: true                   # @user pings (default: true)
    replied_user: true            # reply-reference pings the author (default: true)

# Session isolation (applies to all gateway platforms, not just Discord)
group_sessions_per_user: true     # Isolate sessions per user in shared channels
```

#### `discord.require_mention`

**型:** boolean — **デフォルト:** `true`

有効にすると、ボットはサーバーチャンネルで直接`@mention`されたときだけ応答します。DMはこの設定に関係なく常に応答を受け取ります。

#### `discord.thread_require_mention`

**型:** boolean — **デフォルト:** `false`

デフォルトでは、ボットがスレッドに一度参加すると（`@mention`で自動作成された、または一度返信した）、再び`@mention`されなくても、そのスレッド内の以降のすべてのメッセージに応答し続けます。1対1の会話にはこれが適切なデフォルトです。

ユーザーがターンごとに1つのボットに話しかける**複数ボットのスレッド**では、このデフォルトは落とし穴になります — スレッド内の他のすべてのボットもすべてのメッセージに反応し、クレジットを消費してチャンネルをスパムします。`thread_require_mention: true`を設定すると、スレッド内のメンション省略が無効になり、スレッドもチャンネルと同じようにゲートされます。明示的な`@mention`はこれまでどおり機能します。

```yaml
discord:
  require_mention: true
  thread_require_mention: true    # multi-bot setup
```

#### `discord.free_response_channels`

**型:** string またはリスト — **デフォルト:** `""`

`@mention`を必要とせずに、ボットがすべてのメッセージに応答するチャンネルID。カンマ区切りの文字列、またはYAMLリストのいずれかを受け付けます。

```yaml
# String format
discord:
  free_response_channels: "1234567890,9876543210"

# List format
discord:
  free_response_channels:
    - 1234567890
    - 9876543210
```

スレッドの親チャンネルがこのリストにある場合、そのスレッドもメンション不要になります。

フリーレスポンスチャンネルは、デフォルトで**自動スレッド化もスキップ**します — ボットはメッセージごとに新しいスレッドを派生させるのではなく、インラインで応答します。これにより、チャンネルを軽量なチャット面として使い続けられます。

フリーレスポンスチャンネルでスレッド化をオプトインするには、`discord.free_response_auto_thread: true`（または`DISCORD_FREE_RESPONSE_AUTO_THREAD=true`）を設定します。このモードでは、フリーチャンネルの新しいトップレベルメッセージごとに独自のスレッドが作られますが、チャンネルは@mention不要のままです。`discord.auto_thread: true`が必要です。

#### `discord.free_response_auto_thread`

**型:** boolean — **デフォルト:** `false`

`true`のとき、`discord.free_response_channels`に列挙されたチャンネルでも、インラインで応答する代わりに、トップレベルメッセージごとにスレッドを自動作成します。チャンネルはメンション不要のままで、変わるのは会話が置かれる場所だけです。

```yaml
discord:
  free_response_channels:
    - 1234567890
  auto_thread: true                # required — this flag refines it
  free_response_auto_thread: true  # thread every top-level message there
```

`discord.auto_thread: true`が必要です（オフの場合、どこでもスレッド化されません）。[`discord.no_thread_channels`](#discordno_thread_channels)は引き続き優先され、ボイスに紐づいたテキストチャンネルは常にインラインで返信し、返信型のメッセージが自動スレッド化されることはありません。

両方が設定されている場合、`DISCORD_FREE_RESPONSE_AUTO_THREAD`が`config.yaml`のキーより優先されます — 他のすべての`discord.*`ブリッジと同様に、YAMLの値は環境変数がまだ設定されていない場合にのみ環境変数の初期値になります。

#### `discord.auto_thread`

**型:** boolean — **デフォルト:** `true`

有効にすると、通常のテキストチャンネルでの`@mention`ごとに、会話用の新しいスレッドを自動作成します。これによりメインチャンネルがきれいに保たれ、各会話が独自の分離されたセッション履歴を持ちます。スレッドが作成されると、そのスレッド内の以降のメッセージには`@mention`が不要になります — ボットはすでに参加していることを知っています。複数ボットのセットアップでこのスレッド内のメンション省略を無効にするには、[`thread_require_mention`](#discordthread_require_mention)を`true`に設定します。

既存のスレッドやDMで送られたメッセージは、この設定の影響を受けません。`discord.no_thread_channels`に列挙されたチャンネル、および[`discord.free_response_auto_thread`](#discordfree_response_auto_thread)が`true`でない限り`discord.free_response_channels`に列挙されたチャンネルも、自動スレッド化をバイパスし、代わりにインラインの応答を受け取ります。

#### `discord.reactions`

**型:** boolean — **デフォルト:** `true`

ボットがビジュアルなフィードバックとしてメッセージに絵文字リアクションを追加するかを制御します。
- ボットがメッセージの処理を開始したとき👀が追加される
- 応答が正常に配信されたとき✅が追加される
- 処理中にエラーが発生した場合❌が追加される

リアクションが気が散ると感じる場合や、ボットのロールに**Add Reactions**権限がない場合は、これを無効化してください。

#### `discord.ignored_channels`

**型:** string またはリスト — **デフォルト:** `[]`

直接`@mention`されても、ボットが**決して**応答しないチャンネルID。これは最高の優先度を持ちます — チャンネルがこのリストにある場合、ボットは`require_mention`、`free_response_channels`、その他の設定に関係なく、そこのすべてのメッセージを静かに無視します。

```yaml
# String format
discord:
  ignored_channels: "1234567890,9876543210"

# List format
discord:
  ignored_channels:
    - 1234567890
    - 9876543210
```

スレッドの親チャンネルがこのリストにある場合、そのスレッド内のメッセージも無視されます。

#### `discord.no_thread_channels`

**型:** string またはリスト — **デフォルト:** `[]`

スレッドを自動作成する代わりに、チャンネル内で直接ボットが応答するチャンネルID。これは`auto_thread`が`true`（デフォルト）のときだけ効果があります。これらのチャンネルでは、ボットは新しいスレッドを生成せず、通常のメッセージのようにインラインで応答します。

```yaml
discord:
  no_thread_channels:
    - 1234567890  # Bot responds inline here
```

スレッドが不要なノイズになるような、ボット対話専用のチャンネルに便利です。

#### `discord.channel_prompts`

**型:** マッピング — **デフォルト:** `{}`

該当するDiscordチャンネルまたはスレッドでのすべてのターンで注入される、チャンネルごとの一時的なシステムプロンプト。トランスクリプト履歴には永続化されません。

```yaml
discord:
  channel_prompts:
    "1234567890": |
      This channel is for research tasks. Prefer deep comparisons,
      citations, and concise synthesis.
    "9876543210": |
      This forum is for therapy-style support. Be warm, grounded,
      and non-judgmental.
```

振る舞い:
- 厳密なスレッド/チャンネルIDの一致が勝ちます。
- メッセージがスレッドまたはフォーラム投稿の内部に到着し、そのスレッドに明示的なエントリがない場合、Hermesは親チャンネル/フォーラムIDにフォールバックします。
- プロンプトは実行時に一時的に適用されるため、それらを変更すると、過去のセッション履歴を書き換えることなく、即座に今後のターンに影響します。

#### `discord.history_backfill`

**型:** boolean — **デフォルト:** `true`

有効にすると、ボットは`@mention`のたびに見逃したチャンネルメッセージを回復します。`require_mention: true`の場合、ボットは自分を直接タグ付けしたメッセージだけを処理します — チャンネル内のそれ以外のものはすべて、セッショントランスクリプトからは見えません。履歴のバックフィルは、トリガーされると最近のチャンネル履歴を遡ってスキャンし、ボットの最後の応答と現在のメンションの間のメッセージを集めて、コンテキストとして含めます。

場所ごとの振る舞い:

- **サーバーチャンネル**（`require_mention: true`の場合）: バックフィルはボットの最後の応答以降のチャンネルをスキャンします。ボットが話しかけられていない間に他の参加者が投稿した場合に便利です。
- **スレッド**: バックフィルはスレッドだけをスキャンします — スレッドに対するDiscordの`channel.history()`は、親チャンネルではなくそのスレッドのメッセージだけを返します。スレッドは通常それ自体で完結した会話なので、これが適切な範囲です。
- **DM**: スキップされます。DMのすべてのメッセージがボットをトリガーするため、セッショントランスクリプトはすでに完全です — 埋めるべきメンションの空白はありません。
- **フリーレスポンスチャンネル**と**ボット自身が自動作成したスレッド**: 同じ理由でスキップされます — メンションによるゲートがないので空白もありません。

ユーザーごとのセッション（`group_sessions_per_user: true`、デフォルト）にも利点があります: ユーザーのセッションには、他のチャンネル参加者が投稿したコンテキストと、ボットをタグ付けする前のユーザー自身のメッセージが欠けています。バックフィルはその両方の空白を埋めます。

```yaml
discord:
  history_backfill: true   # default
```

オフにするには:

```yaml
discord:
  history_backfill: false
```

> **注:** ボットが処理している*間*（トリガーから応答までの間）に届いたメッセージは取り込まれません。これは許容された単純化です — ユーザーは再送信するか、もう一度タグ付けできます。

#### `discord.history_backfill_limit`

**型:** integer — **デフォルト:** `50`

チャンネルのコンテキストを回復する際に遡ってスキャンするメッセージの最大数。実際には、スキャンは通常それよりずっと早く — ターン間の自然な境界であるチャンネル内のボット自身の最後のメッセージで — 止まります。この上限は、コールドスタートや、最近の履歴に以前のボットメッセージが存在しない長い空白に対する安全上限です。

```yaml
discord:
  history_backfill: true
  history_backfill_limit: 50
```

#### `discord.missed_message_backfill`

**型:** object — **デフォルト:** 無効

DiscordのWebSocketのresumeウィンドウは、再起動やネットワーク障害の間に期限切れになることがあります。その空白の間に送られたメッセージは、ライブのゲートウェイイベントとしては配信されません。このオプションを有効にすると、HermesはDiscordへの再接続後に、設定されたチャンネルとスレッドの履歴の限られた範囲をスキャンし、まだ処理されていないメッセージを、ライブイベントと同じ認可、メンション、チャンネル、重複排除、ディスパッチの経路で送ります。

```yaml
discord:
  missed_message_backfill:
    enabled: true
    channels: ["123456789012345678"]
    window_seconds: 3600
    limit: 100
    max_dispatches: 10
    max_attempts: 3
```

`channels`が空の場合、Hermesは`discord.free_response_channels`を使います。到達可能なすべてのサーバーテキストチャンネルをボットに調べさせたい場合にのみ`"*"`に設定してください。回復の台帳はプロファイルごとに`gateway/discord_message_recovery.db`に保存され、正常に応答されたメッセージが後の再起動で再び再生されるのを防ぎます。メッセージは、そのターンが最終的な返信を配信した時点で応答済みとみなされます。その返信がDiscordの返信参照を伴っていたかどうかは問いません（`reply_to_mode: "off"`、ストリーミングされた返信、メディアのみの返信も含みます）。

`max_dispatches`は1回のスキャンの上限です。`max_attempts`（デフォルト3）は1つのメッセージが再ディスパッチされうる回数の上限で、ターンが失敗し続けるメッセージが再接続のたびに再実行されないようにします。`window_seconds`は常に尊重されます: チャンネルごとのスキャンカーソルはスキャンを狭めることはあっても、ウィンドウより前に遡ることはありません。

#### `group_sessions_per_user`

**型:** boolean — **デフォルト:** `true`

これは、同じチャンネルのユーザーが分離されたセッション履歴を持つかどうかを制御する、グローバルなゲートウェイ設定です（Discord固有ではありません）。

`true`のとき: `#research`で話すAliceとBobは、それぞれHermesとの独自の別々の会話を持ちます。`false`のとき: チャンネル全体が1つの会話トランスクリプトと1つの実行中エージェントスロットを共有します。

```yaml
group_sessions_per_user: true
```

各モードの完全な意味については、上記の[セッションモデル](#session-model-in-discord)のセクションを参照してください。

#### `display.tool_progress`

**型:** string — **デフォルト:** `"all"` — **値:** `off`, `new`, `all`, `verbose`

ボットが処理中にチャットへ進捗メッセージ（例: "Reading file..."、"Running terminal command..."）を送るかどうかを制御します。これは、すべてのプラットフォームに適用されるグローバルなゲートウェイ設定です。

```yaml
display:
  tool_progress: "all"    # off | new | all | verbose
```

- `off` — 進捗メッセージなし
- `new` — ターンごとに最初のツール呼び出しだけを表示
- `all` — すべてのツール呼び出しを表示（ゲートウェイメッセージでは40文字に切り詰め）
- `verbose` — 完全なツール呼び出しの詳細を表示（長いメッセージになることがある）

#### `display.tool_progress_command`

**型:** boolean — **デフォルト:** `false`

有効にすると、ゲートウェイで`/verbose`スラッシュコマンドが利用可能になり、config.yamlを編集せずにツール進捗モード（`off → new → all → verbose → off`）を循環できます。

```yaml
display:
  tool_progress_command: true
```

#### `display.reasoning_style`

**型:** string — **デフォルト（Discord）:** `"subtext"` — **値:** `code`, `blockquote`, `subtext`

推論の表示が有効なとき、モデルの推論ブロックをどのようにレンダリングするかを制御します。Discordのデフォルトは`subtext`で、Discordネイティブの`-# `による小さな灰色のメタデータテキストを使うため、推論は回答より視覚的に控えめに表示されます。`blockquote`は`>`の引用としてレンダリングし、`code`（他のプラットフォームでのデフォルト）はフェンス付きコードブロックを使います。長い推論は最初の15行に折りたたまれます。

```yaml
display:
  platforms:
    discord:
      reasoning_style: subtext   # code | blockquote | subtext
```

## スラッシュコマンドのアクセス制御 {#slash-command-access-control}

デフォルトでは、すべての許可ユーザーがすべてのスラッシュコマンドを実行できます。許可リストを、**管理者**（すべてのスラッシュコマンドにアクセス）と**通常ユーザー**（明示的に有効化したコマンドのみ）に分けるには、Discordプラットフォームの`extra`ブロックに`allow_admin_from`と`user_allowed_commands`を追加します。

```yaml
gateway:
  platforms:
    discord:
      extra:
        # Existing user allowlist (unchanged)
        allow_from:
          - "123456789012345678"  # admin user ID
          - "999888777666555444"  # regular user ID

        # NEW — admins get all slash commands (built-in + plugin)
        allow_admin_from:
          - "123456789012345678"

        # NEW — non-admin allowed users can only run these slash commands.
        # /help and /whoami are always allowed so users can see their access.
        user_allowed_commands:
          - status
          - model
          - history

        # Optional: separate admin / command lists for server channels
        group_allow_admin_from:
          - "123456789012345678"
        group_user_allowed_commands:
          - status
```

**振る舞い:**

- あるスコープ（DMまたはサーバーチャンネル）で`allow_admin_from`にいるユーザーは、ライブのコマンドレジストリを通じて、登録されている**すべての**スラッシュコマンド — 組み込みおよびプラグイン登録のもの — を実行できます。
- `allow_admin_from`にいないユーザーは、`user_allowed_commands`に列挙されたコマンドと、常に許可される下限である`/help`および`/whoami`のみ実行できます。
- 通常のチャット（スラッシュではないメッセージ）は影響を受けません。管理者でないユーザーも通常どおりエージェントと話せます。ただ任意のコマンドをトリガーできないだけです。
- **後方互換:** あるスコープで`allow_admin_from`が設定されていない場合、そのスコープではスラッシュコマンドのゲーティングが無効になります。既存のインストールは変更なしで動作し続けます。
- DMの管理者ステータスは、サーバーチャンネルの管理者ステータスを意味しません。各スコープは独自の管理者リストを持ちます。

`/whoami`を使うと、アクティブなスコープ、あなたのティア（admin / user / unrestricted）、実行できるスラッシュコマンドを確認できます。

## インタラクティブなモデルピッカー

Discordチャンネルで引数なしの`/model`を送ると、ドロップダウンベースのモデルピッカーが開きます。

1. **プロバイダーの選択** — 利用可能なプロバイダーを表示するSelectドロップダウン（最大25個）。
2. **モデルの選択** — 選択したプロバイダーのモデルを表示する2つ目のドロップダウン（最大25個）。

ピッカーは120秒後にタイムアウトします。認可されたユーザー（`DISCORD_ALLOWED_USERS`にいる者）だけがそれと対話できます。モデル名が分かっている場合は、`/model <name>`を直接入力してください。

## スキル用のネイティブスラッシュコマンド

Hermesは、インストールされたスキルを**ネイティブのDiscord Application Command**として自動的に登録します。つまり、スキルは組み込みコマンドと並んでDiscordのオートコンプリート`/`メニューに表示されます。

- 各スキルはDiscordのスラッシュコマンドになります（例: `/code-review`、`/ascii-art`）
- スキルは任意の`args`文字列パラメータを受け付けます
- Discordにはボットあたり100個のアプリケーションコマンドの上限があります — 利用可能なスロットより多くのスキルがある場合、余分なスキルはログに警告を出してスキップされます
- スキルは、`/model`、`/reset`、`/bg`などの組み込みコマンドと並んで、ボット起動時に登録されます

追加の設定は不要です — `hermes skills install`でインストールされたスキルは、次回のゲートウェイ再起動時に、Discordのスラッシュコマンドとして自動的に登録されます。

### スラッシュコマンドの登録を無効化する

同じDiscordアプリケーションに対して複数のHermesゲートウェイを実行する場合（例: ステージング + 本番）、グローバルなスラッシュコマンドの登録を所有すべきはそのうち1つだけです — そうでないと、最後に起動したものが勝ち、登録がばたつきます。「フォロワー」のゲートウェイではスラッシュ登録をオフにしてください。

```yaml
gateway:
  platforms:
    discord:
      extra:
        slash_commands: false   # default: true
```

これを「プライマリ」のゲートウェイで`true`のままにすると、通常の動作が保たれます — 組み込みとインストール済みスキルのグローバルな`/`メニューコマンド。

## メディアの送信（インラインの`MEDIA:`タグ）

Discordアダプターは、エージェントの応答に含まれるインラインの`MEDIA:/path/to/file`タグを通じて、あらゆる一般的なメディア型のネイティブなファイルアップロードをサポートします — アダプターはタグを取り除き、ファイルを自動的にアップロードします。

| 型 | 配信方法 |
|---|---|
| 画像（PNG/JPG/WebP） | インラインプレビュー付きのネイティブDiscord画像添付 |
| アニメーションGIF | `send_animation`が`animation.gif`としてアップロードし、Discordがインラインで再生する（静的なサムネイルとしてではなく） |
| 動画（MP4/MOV） | `send_video` — ネイティブの動画プレイヤー |
| 音声 / ボイス | `send_voice` — 可能な場合はネイティブのボイスメッセージ、それ以外はファイル添付 |
| ドキュメント（PDF/ZIP/docxなど） | `send_document` — ダウンロードボタン付きのネイティブ添付 |

Discordのアップロードあたりのサイズ上限は、サーバーのブーストティアに依存します（無料で25 MB、最大500 MB）。HermesがHTTP 413を受け取ると、アダプターは静かに失敗するのではなく、ローカルキャッシュパスを指すリンクにフォールバックします。

## 長い返信

8件より多くのDiscordメッセージ（約16,000文字）が必要になる返信は、2,000文字のメッセージを長々と連ねる代わりに、全文を含む1つの`reply-<id>.md`添付として短いキャプション付きで送られます。長い返信をストリーミングしている間、Hermesは上限に達すると新しいメッセージを開くのをやめ、完成した返信をアップロードしてから、それまでに投稿していた部分的なメッセージを削除します。アップロードが失敗した場合（たとえば**Attach Files**権限がない場合）は、何も削除されず、返信全体が途中で切られることなくメッセージとして配信されます。メッセージはDiscordのレート制限が許すペース（チャンネルごとに5秒あたり約5件）で送られ、レート制限されたメッセージはDiscordが指定する待ち時間の後に再試行されます。添付の上限を超える返信は、それぞれ最大7 MBの番号付きファイルに分割されます。

## 任意のファイル型の受信

ユーザーがアップロードしたファイルは、どのファイル型でも受け付けられます。関門となるのはファイル拡張子ではなく、エージェントにメッセージを送る認可です。すべてのアップロードはダウンロードされ、`~/.hermes/cache/documents/`の下にキャッシュされ、`DOCUMENT`型のメッセージイベントとしてエージェントに渡されるため、エージェントは`terminal`（`ffprobe`、`unzip`、`file`、`strings`など）や`read_file`でファイルを調べられます。

- 既知の型（PDF、docx/xlsx/pptx、zip、画像/音声/動画など）は正確なMIMEを保持します。
- 未知の型は、アップロード時に報告されたコンテンツタイプにフォールバックし、それがない場合は`application/octet-stream`になります。
- UTF-8でデコード可能な小さなファイル（テキスト、コード、設定、HTML、CSS、JSON、YAMLなど）は、100 KiBまで内容がプロンプトに自動挿入されます。デコードできないバイナリファイルは、パスを指すコンテキストノートとしてのみ渡されるため（Docker/Modalのサンドボックス化されたターミナル向けに`to_agent_visible_cache_path`で自動変換されます）、コンテキストウィンドウを圧迫しません。

受信側の唯一の制限は、ファイルごとのサイズ上限（デフォルト32 MiB）です。

```yaml
discord:
  # Optional — raise/disable the per-file size cap. Default is 32 MiB.
  # The whole file is held in memory while being cached, so unlimited
  # uploads carry a real memory cost.
  max_attachment_bytes: 33554432   # bytes; 0 = unlimited
```

同等の環境変数: `DISCORD_MAX_ATTACHMENT_BYTES=33554432`（上限なしの場合は`0`）。

従来の`discord.allow_any_attachment`フラグは現在は何もしません — どのファイル型も常に受け付けられます — 既存の設定がエラーにならないよう残されているだけです。

:::warning 無制限のメモリコスト
サイズ上限を無効化する（`max_attachment_bytes: 0`）と、ユーザーが数GBのファイルをボットに投下でき、ゲートウェイはディスクにキャッシュする間、律儀にそれをメモリにバッファリングします。信頼できる単一ユーザーのインストールでのみ設定してください。共有ボットでは、デフォルトの32 MiBのままにするか、控えめに引き上げてください。
:::

## インタラクティブなプロンプト（clarify）

エージェントが`clarify`ツールを呼び出すと — どのアプローチが好みかを尋ねる、タスク後のフィードバックを得る、重要な決定の前に確認するなど — Discordは質問を**選択肢ごとに1つのボタン**でレンダリングします。

> Which framework should I use for the dashboard?
>
> [1. Next.js] [2. Remix] [3. Astro] [Other (type answer)]

番号付きのボタンをクリックして回答するか、**Other**をクリックして自由形式の回答を入力します（そのチャンネルで次に送ったメッセージが回答になります）。自由回答の`clarify`呼び出し（プリセットの選択肢がないもの）はボタンを省略し、単に次のメッセージを取り込みます。

選択が行われるとボタンは自動的に無効になるため、重複クリックでプロンプトが二重に解決されることはありません。応答のタイムアウトは`~/.hermes/config.yaml`の`agent.clarify_timeout`で設定します（デフォルトは`90`秒。`0`以下 = 無制限）。タイムアウト内に応答しない場合、エージェントはハングするのではなく、`"outcome": "timed_out"`でブロックを解除して対応を変えます。質問をスキップするには`skip`と返信します。

### プロンプトのレイアウト

インタラクティブなプロンプト（コマンドの承認、`clarify`の質問、スラッシュコマンドの確認）は1つのレイアウトを共有します: **プレーンなメッセージ**が完全な内容 — コマンドとそれがフラグ付けされた理由および承認期限、または質問と返信のヒント — を運び、その下の**埋め込みカード**はヘッダーだけで、ボタンはカードの下に置かれます。判断に必要なものはすべてプレーンテキストにあるため、埋め込みを非表示にしたり切り離したりするクライアントでもプロンプトは正しく読め、埋め込みをレンダリングするクライアントでも何も二重に表示されません。

`approvals.discord_prompt_timeout`は、コマンドの承認、確認の質問、更新プロンプト、
スラッシュコマンドの確認のための、Discordのインタラクティブなボタンビューを制御します。
デフォルトは300秒で、最小は30秒、上限は`approvals.timeout`と同じ
プラットフォームで安全な値です。

`approvals.timeout`は別途、コマンドが承認をどれだけ待つかを制御します。
コマンドとそのDiscordボタンの両方に1時間を与えるには、所有する
プロファイルの`config.yaml`で両方を設定します。

```yaml
approvals:
  timeout: 3600
  discord_prompt_timeout: 3600
```

各コマンド承認カードは、それ自身のリクエストに応答します。Discordのビューのタイムアウトが
より長くても、そのリクエストが応答される、キャンセルされる、または期限切れになると、
そのコントロールは退役します。Discordのタイムアウトが短い場合、ボタンは退役しますが、
基になる承認の待機を延長したり終了したりはしません。

## ホームチャンネル

ボットがプロアクティブなメッセージ（cronジョブの出力、リマインダー、通知など）を送る「ホームチャンネル」を指定できます。設定する方法は2つあります。

### スラッシュコマンドを使う

ボットが存在する任意のDiscordチャンネルで`/sethome`を入力します。そのチャンネルがホームチャンネルになります。

### 手動設定

`~/.hermes/.env`に次を追加します。

```bash
DISCORD_HOME_CHANNEL=123456789012345678
DISCORD_HOME_CHANNEL_NAME="#bot-updates"
```

IDを実際のチャンネルID（開発者モードをオンにして右クリック → チャンネルIDをコピー）に置き換えてください。

## ボイスメッセージ

Hermes AgentはDiscordのボイスメッセージをサポートします。

- **受信したボイスメッセージ**は、設定されたSTTプロバイダーを使って自動的に文字起こしされます: ローカルの`faster-whisper`（キー不要）、Groq Whisper（`GROQ_API_KEY`）、またはOpenAI Whisper（`VOICE_TOOLS_OPENAI_KEY`）。
- **テキスト読み上げ**: `/voice tts`を使うと、テキストの返信と並んでボットが音声で話す応答を送ります。
- **Discordのボイスチャンネル**: Hermesはボイスチャンネルに参加し、ユーザーが話すのを聞き、チャンネルで話し返すこともできます。

完全なセットアップと運用のガイドについては、次を参照してください。
- [ボイスモード](../features/voice-mode.md)
- [Hermesでボイスモードを使う](../../guides/use-voice-mode-with-hermes.md)

### ボイスチャンネルのオーディオエフェクト（アンビエント + 口頭の相づち）

ボットがボイスチャンネルにいるとき、より会話的な雰囲気を与えられます: 作業を始める前の短い口頭の相づち（「let me look into that」）と、ツールの実行中に下で流れる控えめなアンビエントの「考え中」のBGMです — 発話中はアンビエントの音量が下がり、終わると元に戻ります。Grokのボイスモードに似ています。

discord.pyは接続ごとに1つのオーディオストリームしか再生しないため、Hermesは送信ストリームにソフトウェアミキサーを組み込み、アンビエントのループ、相づち、TTSの返信をその単一のストリームに合成します — これらは互いを途切れさせるのではなく、重なって再生されます。

これは**デフォルトでオフ**です。`config.yaml`で有効にします。

```yaml
discord:
  voice_fx:
    enabled: true          # master switch
    ambient_enabled: true  # idle "thinking" bed while tools run
    ambient_path: ""       # custom loop file (any audio format); "" = built-in synthesised pad
    ambient_gain: 0.18     # idle bed loudness (0.0–1.0)
    duck_gain: 0.06        # ambient loudness while the bot is speaking
    speech_gain: 1.0       # TTS / acknowledgement loudness
    ack_enabled: true      # speak a short phrase before the first tool call of a turn
    ack_phrases:           # picked at random; set to [] to disable the spoken ack
      - "Let me look into that."
      - "One moment."
      - "Checking on that now."
```

注:
- 明示的な`/voice leave`または手動の切断まで、ボットをボイスチャンネルに留めたい場合は、`voice_channel_inactivity_timeout_seconds: 0`を設定します。デフォルトは、従来の300秒のアイドル自動退出を維持します。
- `voice_playback_timeout_seconds`は下限であり、長いTTSに対する厳密な上限ではありません。Hermesは生成された音声の再生時間を調べ、それが設定された下限より長い場合は`duration + 30s`だけ待ちます。
- 相づちはターンごとに最大1回、ボットがボイスチャンネルにいてミキサーが有効なときだけ発せられます。設定されたTTSプロバイダーを使います。
- `ambient_path`は`ffmpeg`がデコードできる任意のファイルを受け付け、シームレスにループ再生されます。空のままにすると、組み込みの合成パッドを使います（アセット不要）。
- すべての設定は（`.env`ではなく）`config.yaml`に置きます — これらは振る舞いの設定であり、シークレットではありません。
- `voice_fx.enabled`が`false`のとき、ボイス再生は元の単発の経路を使い、何も変わりません。


## フォーラムチャンネル

Discordのフォーラムチャンネル（型15）はダイレクトメッセージを受け付けません — フォーラム内のすべての投稿はスレッドでなければなりません。Hermesはフォーラムチャンネルを自動検出し、そこに送る必要があるときは常に新しいスレッド投稿を作成するため、テキストの返信、TTS、画像、ボイスメッセージ、ファイル添付がすべて、エージェント側の特別な処理なしで動作します。

- **スレッド名**は、メッセージの最初の行から導出されます（markdownの見出しプレフィックスを除去し、100文字に上限）。メッセージが添付のみの場合、ファイル名がフォールバックのスレッド名として使われます。
- **添付**は、新しいスレッドの開始メッセージに同行します — 別のアップロード手順も、部分的な送信もありません。
- **1呼び出し、1スレッド**: 各フォーラム送信は新しいスレッドを作成します。したがって、同じフォーラムへの連続した送信は、別々のスレッドを生みます。
- **検出は3層構成**: まずチャンネルディレクトリのキャッシュ、次にプロセスローカルなプローブキャッシュ、最後の手段としてライブの`GET /channels/{id}`プローブ（その結果はプロセスの生存期間中メモ化されます）。

ディレクトリを更新する（公開しているプラットフォームでは`/channels refresh`、またはゲートウェイの再起動）と、ボット起動後に作成されたフォーラムチャンネルがキャッシュに取り込まれます。

## トラブルシューティング

### ボットはオンラインだがメッセージに応答しない

**原因**: 通常は、アクセスポリシーが設定されていないため、Discordの認可がフェイルクローズしています。（Message Content Intentが無効な場合は、ボットはオフラインのままになります。ステップ3を参照してください。）

**修正**:

1. [開発者ポータル](https://discord.com/developers/applications) → あなたのアプリ → Bot → Privileged Gateway Intents → **Message Content Intent**を有効化 → Save Changes。
2. 少なくとも1つのDiscordアクセスポリシーが設定されていることを確認します。

   ```bash
   # recommended: allow specific users
   DISCORD_ALLOWED_USERS=284102345871466496

   # or allow a trusted guild/dev bot to behave like pre-0.18 Discord
   DISCORD_ALLOW_ALL_USERS=true
   ```

3. ゲートウェイを再起動します。

   ```bash
   hermes gateway restart
   ```

ゲートウェイのログでDiscordが接続済みでREST APIのチェックも機能しているのに、受信メッセージがすべて無反応な場合は、`~/.hermes/logs/gateway.log`で次の警告を探してください。

```text
No Discord access policy configured; inbound Discord messages will be denied by default.
```

Hermes 0.18は、外部から到達可能なアダプターで意図的にフェイルクローズします。`DISCORD_ALLOWED_USERS`、`DISCORD_ALLOWED_ROLES`、`DISCORD_ALLOWED_CHANNELS`、明示的な全許可フラグのいずれもないDiscordボットは、正常に接続しますが、通常のメッセージ処理の前に受信ユーザーを拒否します。

### 起動時の「Privileged intents」 / `PrivilegedIntentsRequired`エラー

**原因**: Hermesが、開発者ポータルでボットに対して有効になっていない特権Gateway Intentsを要求しています。そのためDiscordはWebSocket接続を拒否します。Hermesは常に**Message Content Intent**を要求します。また、許可リストに（数値IDではなく）ユーザー名を使っている場合や、`DISCORD_ALLOWED_ROLES`が設定されている場合は、**Server Members Intent**も要求します。Presence Intentは不要です。

**修正**:

1. [開発者ポータル](https://discord.com/developers/applications) → あなたのアプリ → Bot → Privileged Gateway Intentsを開きます。
2. **Message Content Intent**を有効化します（必須）。ユーザー名やロールの許可リストを使う場合は、**Server Members Intent**も有効化します。
3. **Save Changes**をクリックし、ゲートウェイを再起動します（`hermes gateway restart`）。

ゲートウェイのログには、Hermesが要求した正確なインテントが記載されているはずです。それらが有効になるまで、Discordは接続を拒否し続けます — これはポータルの設定エラーであり、不安定なネットワークの問題ではありません。

### ボットが特定のチャンネルでメッセージを見られない

**原因**: ボットのロールに、そのチャンネルを表示する権限がありません。

**修正**: Discordで、チャンネルの設定 → 権限 → ボットのロールを**View Channel**と**Read Message History**を有効にして追加します。

### 403 Forbiddenエラー

**原因**: ボットに必要な権限が不足しています。

**修正**: ステップ5のURLを使って正しい権限でボットを再招待するか、サーバー設定 → ロールでボットのロールの権限を手動で調整します。

`/thread`がDiscordのエラー`50001`（`Missing Access`）を報告する場合は、親チャンネルでの
ボットの実効権限を確認してください。**View Channel**、**Send
Messages**、**Create Public Threads**、**Send Messages in Threads**が必要です。親カテゴリと
チャンネル固有の権限の上書きも確認してください: そこでの明示的な拒否は、
サーバーロールで付与された権限より優先されることがあります。

### ボットがオフライン

**原因**: Hermesゲートウェイが動作していない、またはトークンが正しくありません。

**修正**: `hermes gateway`が動作していることを確認します。`.env`ファイルの`DISCORD_BOT_TOKEN`を検証します。最近トークンをリセットした場合は、それを更新します。

### 「User not allowed」 / ボットがあなたを無視する

**原因**: あなたのユーザーIDが`DISCORD_ALLOWED_USERS`にありません。

**修正**: `~/.hermes/.env`の`DISCORD_ALLOWED_USERS`にあなたのユーザーIDを追加し、ゲートウェイを再起動します。

### 同じチャンネルの人々が予期せずコンテキストを共有している

**原因**: `group_sessions_per_user`が無効になっている、またはプラットフォームがそのコンテキストのメッセージにユーザーIDを提供できません。

**修正**: `~/.hermes/config.yaml`にこれを設定し、ゲートウェイを再起動します。

```yaml
group_sessions_per_user: true
```

意図的に共有のルーム会話を望む場合は、オフのままにしてください — ただし、共有のトランスクリプト履歴と共有の割り込み動作を想定してください。

## セキュリティ

:::warning
ボットと対話できる人を制限するため、必ず`DISCORD_ALLOWED_USERS`（または`DISCORD_ALLOWED_ROLES`）を設定してください。どちらもないと、ゲートウェイは安全策としてデフォルトですべてのユーザーを拒否します。信頼する人だけを認可してください — 認可されたユーザーは、ツールの使用やシステムアクセスを含む、エージェントのあらゆる機能への完全なアクセス権を持ちます。
:::

### ロールベースのアクセス制御

個々のユーザーリストの代わりにロールでアクセスを管理するサーバー（モデレーターチーム、サポートスタッフ、社内ツール）では、`DISCORD_ALLOWED_ROLES` — ロールIDのカンマ区切りリスト — を使ってください。これらのロールのいずれかを持つメンバーは認可されます。

```bash
# ~/.hermes/.env — works alongside or instead of DISCORD_ALLOWED_USERS
DISCORD_ALLOWED_ROLES=987654321098765432,876543210987654321
```

意味:

- **ユーザー許可リストとのOR。** ユーザーは、IDが`DISCORD_ALLOWED_USERS`にある**または**`DISCORD_ALLOWED_ROLES`のいずれかのロールを持つ場合に認可されます。
- **Server Members Intentの自動有効化。** `DISCORD_ALLOWED_ROLES`が設定されると、ボットは接続時にMembersインテントを有効化します — Discordがメンバーレコードとともにロール情報を送るために必要です。
- **ロール名ではなくロールID。** Discordから取得します: **ユーザー設定 → 詳細設定 → 開発者モードをON**、その後、任意のロールを右クリック → **ロールIDをコピー**。
- **DMフォールバック。** DMでは、ロールチェックは相互のギルドをスキャンします。共有するいずれかのサーバーで許可されたロールを持つユーザーは、DMでも認可されます。

これは、モデレーションチームが入れ替わるときに好ましいパターンです — 新しいモデレーターは、`.env`の編集やゲートウェイの再起動なしで、ロールが付与された瞬間にアクセスを得ます。

**`config.yaml`でリストを編集すると、次のメッセージから反映されます。** `discord.allowed_roles`、`discord.allow_from`、およびチャンネルのリストは`config.yaml`が変更されると再読み込みされ、ゲートウェイの再起動は不要です: 削除したロールやユーザーは、次のメッセージ、スラッシュコマンド、ボタンクリックから拒否されます。これは、マルチプレックスされたゲートウェイ（`gateway.multiplex_profiles`）が提供するすべてのプロファイルに当てはまり、起動時のホストプロファイルも含みます。`.env`、外部のシークレットソース、またはゲートウェイ自身の環境で設定された値は、引き続き`config.yaml`より優先されます。Server Members Intentなしで接続したボットに*最初の*ロールやユーザー名を追加する場合は、インテントが接続時に選ばれるため、引き続き再起動が必要です。後から追加したユーザー名は、次の再接続でIDに解決された後にのみ照合されます。

### メンション制御 {#mention-control}

デフォルトでは、Hermesは、返信にそれらのトークンが含まれていても、ボットが`@everyone`、`@here`、ロールメンションをpingするのをブロックします。これにより、不適切に書かれたプロンプトやエコーされたユーザーコンテンツが、サーバー全体をスパムするのを防ぎます。個々の`@user`のpingと返信参照のping（小さな「返信先…」のチップ）は、通常の会話が機能するよう有効のままです。

これらのデフォルトは、環境変数または`config.yaml`のいずれかで緩められます。

```yaml
# ~/.hermes/config.yaml
discord:
  allow_mentions:
    everyone: false      # allow the bot to ping @everyone / @here
    roles: false         # allow the bot to ping @role mentions
    users: true          # allow the bot to ping individual @users
    replied_user: true   # ping the author when replying to their message
```

```bash
# ~/.hermes/.env — env vars win over config.yaml
DISCORD_ALLOW_MENTION_EVERYONE=false
DISCORD_ALLOW_MENTION_ROLES=false
DISCORD_ALLOW_MENTION_USERS=true
DISCORD_ALLOW_MENTION_REPLIED_USER=true
```

:::tip
なぜ必要なのかを正確に分かっている場合を除き、`everyone`と`roles`は`false`のままにしてください。LLMが通常の見た目の応答の中に文字列`@everyone`を生成するのは非常に簡単であり、この保護がなければ、それはサーバーの全メンバーに通知してしまいます。
:::

Hermes Agentのデプロイを保護する方法の詳細については、[セキュリティガイド](../security.md)を参照してください。

