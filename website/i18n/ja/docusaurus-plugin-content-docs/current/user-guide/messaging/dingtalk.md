---
sidebar_position: 10
title: "DingTalk"
description: "Hermes AgentをDingTalkチャットボットとしてセットアップする"
---

# DingTalkのセットアップ

Hermes AgentはDingTalk（钉钉）とチャットボットとして連携し、ダイレクトメッセージやグループチャットを通じてAIアシスタントとチャットできるようにします。ボットはDingTalkのStream Mode（公開URLやWebhookサーバーを必要としない長寿命のWebSocket接続）経由で接続し、DingTalkのセッションWebhook APIを通じてmarkdown形式のメッセージで返信します。

セットアップの前に、多くの人が知りたい部分から説明します。Hermesがあなたのworkspace（ワークスペース）に入った後、DingTalk上でどう振る舞うかです。

## Hermesの振る舞い

| コンテキスト | 振る舞い |
|---------|----------|
| **DM（1対1チャット）** | Hermesはすべてのメッセージに応答します。`@mention` は不要です。各DMには独自のセッションがあります。 |
| **グループチャット** | Hermesは `@mention` されたときに応答します。メンションがない場合、Hermesはメッセージを無視します。 |
| **複数ユーザーがいる共有グループ** | デフォルトでは、Hermesはグループ内でユーザーごとにセッション履歴を分離します。同じグループで会話する2人が、明示的に無効化しない限り1つのトランスクリプトを共有することはありません。 |

### DingTalkにおけるセッションモデル

デフォルトでは:

- 各DMは独自のセッションを取得します
- 共有グループチャット内の各ユーザーは、そのグループ内で独自のセッションを取得します

これは `config.yaml` で制御されます。

```yaml
group_sessions_per_user: true
```

グループ全体で1つの共有会話を明示的に望む場合にのみ、`false` に設定します。

```yaml
group_sessions_per_user: false
```

このガイドでは、DingTalkボットの作成から最初のメッセージ送信まで、セットアッププロセス全体を順を追って説明します。

## 前提条件

必要なPythonパッケージをインストールします。

```bash
pip install "hermes-agent[dingtalk]"
```

または個別に:

```bash
pip install dingtalk-stream httpx alibabacloud-dingtalk
```

- `dingtalk-stream` — Stream Mode（WebSocketベースのリアルタイムメッセージング）用のDingTalk公式SDK
- `httpx` — セッションWebhook経由で返信を送信するために使用される非同期HTTPクライアント
- `alibabacloud-dingtalk` — AIカード、絵文字リアクション、メディアダウンロード用のDingTalk OpenAPI SDK

## ステップ1: DingTalkアプリを作成する

1. [DingTalk Developer Console](https://open-dev.dingtalk.com/)にアクセスします。
2. DingTalk管理者アカウントでログインします。
3. **Application Development** → **Custom Apps** → **Create App via H5 Micro-App**（コンソールのバージョンによっては **Robot**）をクリックします。
4. 次を入力します。
   - **App Name**: 例 `Hermes Agent`
   - **Description**: 任意
5. 作成後、**Credentials & Basic Info** に移動して **Client ID**（AppKey）と **Client Secret**（AppSecret）を確認します。両方をコピーしてください。

:::warning[認証情報は一度だけ表示されます]
Client Secretは、アプリを作成したときに一度だけ表示されます。紛失した場合は再生成する必要があります。これらの認証情報を公開したり、Gitにコミットしたりしないでください。
:::

## ステップ2: ロボット機能を有効化する

1. アプリの設定ページで、**Add Capability** → **Robot** に移動します。
2. ロボット機能を有効化します。
3. **Message Reception Mode** で、**Stream Mode** を選択します（推奨 — 公開URLは不要）。

:::tip
Stream Modeが推奨のセットアップです。あなたのマシンから開始される長寿命のWebSocket接続を使用するため、公開IP、ドメイン名、Webhookエンドポイントは不要です。これはNAT、ファイアウォールの内側、ローカルマシンでも機能します。
:::

## ステップ3: DingTalkユーザーIDを確認する

Hermes Agentは、DingTalkユーザーIDを使ってボットと対話できる人を制御します。DingTalkユーザーIDは、組織の管理者が設定する英数字の文字列です。

自分のIDを確認するには:

1. DingTalk組織の管理者に尋ねてください。ユーザーIDはDingTalk管理コンソールの **Contacts** → **Members** で設定されています。
2. あるいは、ボットは受信メッセージごとに `sender_id` をログに記録します。ゲートウェイを起動し、ボットにメッセージを送信してから、ログであなたのIDを確認してください。

## ステップ4: Hermes Agentを設定する

### オプションA: 対話的セットアップ（推奨）

ガイド付きのセットアップコマンドを実行します。

```bash
hermes gateway setup
```

プロンプトが表示されたら **DingTalk** を選択します。セットアップウィザードは、次の2つのいずれかの方法で認可できます。

- **QRコードデバイスフロー（推奨）。** ターミナルに表示されるQRをDingTalkモバイルアプリでスキャンします。Client IDとClient Secretが自動的に返され、`~/.hermes/.env` に書き込まれます。デベロッパーコンソールへのアクセスは不要です。
- **手動貼り付け。** すでに認証情報がある場合（またはQRスキャンが不便な場合）、プロンプトが表示されたらClient ID、Client Secret、許可するユーザーIDを貼り付けます。

:::note openClawブランディングの開示
DingTalkの `verification_uri_complete` はAPIレイヤーでopenClawのアイデンティティにハードコードされているため、Alibaba / DingTalk-Real-AIがサーバー側でHermes固有のテンプレートを登録するまで、QRは現在 `openClaw` のソース文字列の下で認可されます。これは純粋にDingTalkが同意画面をどう提示するかの問題です。作成するボットは完全にあなたのものであり、あなたのテナント内で非公開です。
:::

### オプションB: 手動設定

`~/.hermes/.env` ファイルに次を追加します。

```bash
# 必須
DINGTALK_CLIENT_ID=your-app-key
DINGTALK_CLIENT_SECRET=your-app-secret

# セキュリティ: ボットと対話できる人を制限する
DINGTALK_ALLOWED_USERS=user-id-1

# 複数の許可ユーザー（カンマ区切り）
# DINGTALK_ALLOWED_USERS=user-id-1,user-id-2

# 任意: グループチャットのゲーティング（Slack/Telegram/Discord/WhatsAppと同様）
# DINGTALK_REQUIRE_MENTION=true
# DINGTALK_FREE_RESPONSE_CHATS=cidABC==,cidDEF==
# DINGTALK_MENTION_PATTERNS=^小马
# DINGTALK_HOME_CHANNEL=cidXXXX==
# DINGTALK_ALLOW_ALL_USERS=true
```

`~/.hermes/config.yaml` の任意の動作設定:

```yaml
group_sessions_per_user: true

gateway:
  platforms:
    dingtalk:
      extra:
        # ボットが返信する前にグループで @mention を要求する（Slack/Telegram/Discordと同等）。
        # DMはこれを無視する — ボットは1対1チャットでは常に返信する。
        require_mention: true

        # プラットフォーム単位の許可リスト。設定すると、これらのDingTalkユーザーIDのみがボットと対話できる
        # （DINGTALK_ALLOWED_USERS と同じ意味だが、.env ではなくここにスコープされる）。
        allowed_users:
          - user-id-1
          - user-id-2
```

- `group_sessions_per_user: true` は、共有グループチャット内で各参加者のコンテキストを分離して保ちます
- `require_mention: true` は、ボットがすべてのグループメッセージに応答するのを防ぎます。誰かが @-mention したときだけ応答します
- `dingtalk.extra` の下の `allowed_users` は `DINGTALK_ALLOWED_USERS` の代替です。両方が設定されている場合はマージされます

### ゲートウェイを起動する

設定が完了したら、DingTalkゲートウェイを起動します。

```bash
hermes gateway
```

ボットは数秒以内にDingTalkのStream Modeに接続するはずです。テストするには、DMで、またはボットが追加されているグループでメッセージを送信してください。

:::tip
`hermes gateway` をバックグラウンドで、またはsystemdサービスとして実行して永続的に動作させることができます。詳細はデプロイドキュメントを参照してください。
:::

## 機能

### AIカード

Hermesは、プレーンなmarkdownメッセージの代わりにDingTalk AIカードを使って返信できます。カードはよりリッチで構造化された表示を提供し、エージェントが応答を生成する際のストリーミング更新をサポートします。

AIカードを有効化するには、`config.yaml` でカードテンプレートIDを設定します。

```yaml
platforms:
  dingtalk:
    enabled: true
    extra:
      card_template_id: "your-card-template-id"
```

カードテンプレートIDは、DingTalk Developer ConsoleのアプリのAIカード設定で確認できます。AIカードを有効化すると、すべての返信がストリーミングテキスト更新付きのカードとして送信されます。

### 絵文字リアクション

Hermesは、処理状況を示すためにメッセージに自動的に絵文字リアクションを追加します。

- 🤔Thinking — ボットがメッセージの処理を開始したときに追加されます
- 🥳Done — 応答が完了したときに追加されます（Thinkingリアクションを置き換えます）

これらのリアクションは、DMとグループチャットの両方で機能します。

### 表示設定

DingTalkの表示動作を、他のプラットフォームとは独立してカスタマイズできます。

```yaml
display:
  platforms:
    dingtalk:
      show_reasoning: false   # 返信にモデルの推論/思考を表示する
      streaming: true         # ストリーミング応答を有効化する（AIカードで機能）
      tool_progress: all      # ツール実行の進捗を表示する（all/new/off）
      interim_assistant_messages: true  # 中間のコメントメッセージを表示する
```

よりすっきりした体験のためにツールの進捗と中間メッセージを無効化するには:

```yaml
display:
  platforms:
    dingtalk:
      tool_progress: off
      interim_assistant_messages: false
```

## トラブルシューティング

### ボットがメッセージに応答しない

**原因**: ロボット機能が有効になっていないか、`DINGTALK_ALLOWED_USERS` にあなたのユーザーIDが含まれていません。

**対処法**: アプリ設定でロボット機能が有効になっており、Stream Modeが選択されていることを確認してください。あなたのユーザーIDが `DINGTALK_ALLOWED_USERS` に含まれていることを確認してください。ゲートウェイを再起動します。

### 「dingtalk-stream not installed」エラー

**原因**: `dingtalk-stream` Pythonパッケージがインストールされていません。

**対処法**: インストールします。

```bash
pip install dingtalk-stream httpx
```

### 「DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required」

**原因**: 認証情報が環境または `.env` ファイルに設定されていません。

**対処法**: `DINGTALK_CLIENT_ID` と `DINGTALK_CLIENT_SECRET` が `~/.hermes/.env` に正しく設定されていることを確認してください。Client IDはAppKey、Client SecretはDingTalk Developer ConsoleのAppSecretです。

### ストリームが切断する / 再接続ループ

**原因**: ネットワークの不安定さ、DingTalkプラットフォームのメンテナンス、または認証情報の問題。

**対処法**: アダプターは指数バックオフ（2秒 → 5秒 → 10秒 → 30秒 → 60秒）で自動的に再接続します。認証情報が有効であり、アプリが無効化されていないことを確認してください。ネットワークが送信WebSocket接続を許可していることを確認してください。

### ボットがオフライン

**原因**: Hermesゲートウェイが稼働していないか、接続に失敗しました。

**対処法**: `hermes gateway` が稼働していることを確認してください。ターミナル出力でエラーメッセージを確認してください。よくある問題: 認証情報の誤り、アプリの無効化、`dingtalk-stream` または `httpx` がインストールされていない。

### 「No session_webhook available」

**原因**: ボットは返信しようとしましたが、セッションWebhook URLを持っていません。これは通常、Webhookが期限切れになったか、メッセージの受信と返信の送信の間にボットが再起動された場合に発生します。

**対処法**: ボットに新しいメッセージを送信してください。各受信メッセージは返信用の新しいセッションWebhookを提供します。これはDingTalkの通常の制限です。ボットは最近受信したメッセージにのみ返信できます。

## セキュリティ

:::warning
ボットと対話できる人を制限するために、必ず `DINGTALK_ALLOWED_USERS` を設定してください。これがないと、ゲートウェイは安全策としてデフォルトですべてのユーザーを拒否します。信頼する人のユーザーIDのみを追加してください。認可されたユーザーは、ツールの使用やシステムアクセスを含む、エージェントの全機能にフルアクセスできます。
:::

Hermes Agentのデプロイをセキュアにする方法の詳細については、[セキュリティガイド](../security.md)を参照してください。

## 補足

- **Stream Mode**: 公開URL、ドメイン名、Webhookサーバーは不要です。接続はあなたのマシンからWebSocket経由で開始されるため、NATやファイアウォールの内側でも機能します。
- **AIカード**: 任意で、プレーンなmarkdownの代わりにリッチなAIカードで返信します。`card_template_id` で設定します。
- **絵文字リアクション**: 処理状況を示す自動的な 🤔Thinking/🥳Done リアクション。
- **Markdown応答**: 返信はリッチテキスト表示のためにDingTalkのmarkdown形式でフォーマットされます。
- **メディアサポート**: 受信メッセージ内の画像とファイルは自動的に解決され、ビジョンツールで処理できます。
- **メッセージの重複排除**: アダプターは、同じメッセージを2回処理しないように5分間のウィンドウでメッセージを重複排除します。
- **自動再接続**: ストリーム接続が切断された場合、アダプターは指数バックオフで自動的に再接続します。
- **メッセージ長の制限**: 応答は1メッセージあたり20,000文字に制限されます。より長い応答は切り詰められます。
