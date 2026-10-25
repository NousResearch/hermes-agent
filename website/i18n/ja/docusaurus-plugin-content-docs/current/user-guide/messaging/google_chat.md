---
sidebar_position: 12
title: "Google Chat"
description: "Cloud Pub/Sub を使って Hermes Agent を Google Chat ボットとしてセットアップする"
---

# Google Chat のセットアップ

Hermes Agent を Google Chat にボットとして接続します。この統合では、受信イベントに
Cloud Pub/Sub のプルサブスクリプションを、送信メッセージに Chat REST API を使用します。
Slack の Socket Mode や Telegram のロングポーリングと同等の使い勝手です。Hermes
プロセスは公開 URL、トンネル、TLS 証明書を必要としません。Telegram ボットがトークンで
リッスンするのと同じように、接続し、認証し、サブスクリプションでリッスンします。

:::note Workspace エディション
Google Chat は Google Workspace の一部です。この統合は、個人の Workspace
（Google を通じて登録した `@yourdomain.com`）でも、アプリを公開する管理者権限を持つ
業務用の Workspace でも利用できます。Gmail のみのアカウントでは Chat アプリをホストできません。
:::

## 概要

| コンポーネント | 値 |
|-----------|-------|
| **ライブラリ** | `google-cloud-pubsub`、`google-api-python-client`、`google-auth` |
| **受信トランスポート** | Cloud Pub/Sub プルサブスクリプション（公開エンドポイント不要） |
| **送信トランスポート** | Chat REST API（`chat.googleapis.com`） |
| **認証** | サブスクリプションに `roles/pubsub.subscriber` を付与したサービスアカウント JSON |
| **ユーザー識別** | Chat リソース名（`users/{id}`）＋メールアドレス |

---

## ステップ 1: GCP プロジェクトの作成または選択

Pub/Sub トピックをホストするための Google Cloud プロジェクトが必要です。お持ちでない場合は、
[console.cloud.google.com](https://console.cloud.google.com) で作成してください。
個人アカウントでも、ボットのトラフィックを十分にカバーできる無料枠が利用できます。

プロジェクト ID（例: `my-chat-bot-123`）を控えておいてください。以降のすべての
ステップで使用します。

---

## ステップ 2: 2 つの API を有効化する

コンソールで **APIs & Services → Library** に移動し、次を有効化します。

- **Google Chat API**
- **Cloud Pub/Sub API**

どちらも個人ボットが生成する程度のボリュームであれば無料です。

---

## ステップ 3: サービスアカウントの作成

**IAM & Admin → Service Accounts → Create Service Account**

- 名前: `hermes-chat-bot`
- 「このサービスアカウントにプロジェクトへのアクセスを許可」のステップはスキップします。
  特定のサブスクリプションに対する IAM だけで十分です。プロジェクトレベルの Pub/Sub
  ロールは**付与しないでください**。

作成後、そのサービスアカウントを開き、**Keys → Add Key → Create new key → JSON** に進んで
ファイルをダウンロードします。Hermes だけが読み取れる場所に保存してください（例:
`~/.hermes/google-chat-sa.json`、`chmod 600`）。

:::caution 「Chat Bot Caller」ロールは存在しません
よくある間違いは、Chat 専用の IAM ロールを探してプロジェクトレベルで付与しようとすることです。
そのようなロールは存在しません。Chat ボットの権限は IAM からではなく、スペースに
インストールされていることに由来します。サービスアカウントに必要なのは、次のステップで
作成するサブスクリプションに対する Pub/Sub subscriber だけです。
:::

---

## ステップ 4: Pub/Sub トピックとサブスクリプションの作成

**Pub/Sub → Topics → Create topic**

- トピック ID: `hermes-chat-events`
- その他はすべてデフォルトのままにします。

作成後、トピックの詳細ページに **Subscriptions** タブがあります。次の内容で 1 つ作成します。

- サブスクリプション ID: `hermes-chat-events-sub`
- 配信タイプ: **Pull**
- メッセージ保持期間: **7 日**（hermes の再起動時にバックログが残るように）
- 残りはデフォルトのままにします。

---

## ステップ 5: トピックへの IAM バインディング（重要）

**トピック**（サブスクリプションではありません）に IAM プリンシパルを追加します。

- プリンシパル: `chat-api-push@system.gserviceaccount.com`
- ロール: `Pub/Sub Publisher`

これがないと、Google Chat はトピックにイベントを公開できず、ボットは何も受信できなくなります。

---

## ステップ 6: サブスクリプションへの IAM バインディング

**サブスクリプション**に、自分のサービスアカウントをプリンシパルとして追加します。

- プリンシパル: `hermes-chat-bot@<your-project>.iam.gserviceaccount.com`
- ロール: `Pub/Sub Subscriber`

同じサブスクリプションに `Pub/Sub Viewer` も付与してください。Hermes は起動時に
到達性チェックとして `subscription.get()` を呼び出します。

---

## ステップ 7: Chat アプリの設定

**APIs & Services → Google Chat API → Configuration** に移動します。

- **App name**: ユーザーに表示したい任意の名前（「Hermes」が妥当です）。
- **Avatar URL**: 任意の公開 PNG（Google にもいくつかデフォルトがあります）。
- **Description**: アプリディレクトリに表示される短い説明文。
- **Functionality**: **Receive 1:1 messages** と **Join spaces and group
  conversations** を有効にします。
- **Connection settings**: **Cloud Pub/Sub** を選択し、トピック名
  `projects/<your-project>/topics/hermes-chat-events` を入力します。
- **Visibility**: 自分の workspace（または特定のユーザー）に制限します。テスト中は
  全員に公開しないでください。

保存します。

---

## ステップ 8: テスト用スペースへのボットのインストール

ブラウザで Google Chat を開きます。**+ New Chat** メニューでアプリ名を検索して、DM を
開始します。初めてメッセージを送ると、Google は `ADDED_TO_SPACE` イベントを送信します。
Hermes はこれを使って、自己メッセージのフィルタリングのためにボット自身の `users/{id}` を
キャッシュします。

---

## ステップ 9: Hermes の設定

`~/.hermes/.env` に Google Chat のセクションを追加します。

```bash
# Required
GOOGLE_CHAT_PROJECT_ID=my-chat-bot-123
GOOGLE_CHAT_SUBSCRIPTION_NAME=projects/my-chat-bot-123/subscriptions/hermes-chat-events-sub
GOOGLE_CHAT_SERVICE_ACCOUNT_JSON=/home/you/.hermes/google-chat-sa.json

# Authorization — ボットと会話を許可する人のメールアドレスを貼り付ける
GOOGLE_CHAT_ALLOWED_USERS=you@yourdomain.com,coworker@yourdomain.com

# Optional
GOOGLE_CHAT_HOME_CHANNEL=spaces/AAAA...         # cron ジョブのデフォルト配信先
GOOGLE_CHAT_MAX_MESSAGES=1                      # Pub/Sub FlowControl; 1 はセッションごとにコマンドを直列化する
GOOGLE_CHAT_MAX_BYTES=16777216                  # 16 MiB — 処理中のメッセージバイト数の上限
```

プロジェクト ID は `GOOGLE_CLOUD_PROJECT` にもフォールバックし、サービスアカウントのパスは
`GOOGLE_APPLICATION_CREDENTIALS` にフォールバックします。好みの慣習を使ってください。

Google Chat アダプターが必要とする依存関係をインストールします（現在 Hermes の extra は
公開されていないため、直接インストールします）。

```bash
pip install google-cloud-pubsub google-api-python-client google-auth google-auth-oauthlib
```

ゲートウェイを起動します。

```bash
hermes gateway
```

次のようなログ行が表示されるはずです。

```
[GoogleChat] Connected; project=my-chat-bot-123, subscription=<redacted>,
             bot_user_id=users/XXXX, flow_control(msgs=1, bytes=16777216)
```

テスト DM で「hola」と送信します。ボットは「Hermes is thinking…」というマーカーを投稿し、
その同じメッセージをその場で編集して実際の応答に置き換えます。「message deleted」の
墓標は残りません。

---

## フォーマットと機能

Google Chat は限定的な markdown のサブセットをレンダリングします。

| サポート対象 | 非サポート |
|-----------|---------------|
| `*bold*`、`_italic_`、`~strike~`、`` `code` `` | 見出し、リスト |
| URL によるインライン画像 | インタラクティブな Card v2 ボタン（このゲートウェイの v1） |
| ネイティブのファイル添付（`/setup-files` 後 — ステップ 10 参照） | ネイティブのボイスノート／円形ビデオノート |

エージェントのシステムプロンプトには Google Chat 固有のヒントが含まれており、これらの
制限を把握して、レンダリングされないフォーマットを避けるようになっています。

メッセージサイズの上限: 1 メッセージあたり 4000 文字。これより長いエージェントの応答は、
自動的に複数のメッセージに分割されます。

スレッドのサポート: ユーザーがスレッド内で返信すると、Hermes は `thread.name` を検出して
同じスレッド内に返信を投稿します。これにより、各スレッドが個別の Hermes セッションになります。

---

## ステップ 10: ネイティブ添付ファイルの配信（オプション）

標準では、ボットはテキスト、URL によるインライン画像、音声／動画／ドキュメントの
ダウンロードカードを投稿できます。**ネイティブ**の Chat 添付ファイル（人間がファイルを
ドラッグ＆ドロップしたときと同じファイルウィジェット）を配信するには、各ユーザーが
ユーザーごとの OAuth フローを通じて一度だけボットを承認します。

### 別フローが必要な理由

Google Chat の `media.upload` エンドポイントは、サービスアカウント認証を完全に拒否します。

> This method doesn't support app authentication with a service account.
> Authenticate with a user account.

これを解決する IAM ロールやスコープはありません。このエンドポイントはユーザー認証情報のみを
受け付けます。そのため、ボットはファイルをアップロードするたびに*ユーザーとして*、具体的には
そのファイルを要求したユーザーとして振る舞う必要があります。

### ホストの 1 回限りのセットアップ

1. 同じ GCP プロジェクトで **APIs & Services → Credentials** に移動します。
2. **Create credentials → OAuth client ID → Desktop app** を選択します。
3. JSON をダウンロードします。Hermes を実行するホストに移動します。
4. ホスト上で、クライアントを Hermes に登録します。

```bash
python -m gateway.platforms.google_chat_user_oauth \
    --client-secret /path/to/client_secret.json
```

これにより `~/.hermes/google_chat_user_client_secret.json` が書き込まれます。これは
共有インフラであり、個々のユーザーではなく OAuth *アプリ*を識別します。後で何人の
ユーザーが承認しても、ホストごとに 1 ファイルあれば十分です。

### ユーザーごとの承認（チャット内）

各ユーザーは、自分のボットとの DM 内で一度だけフローを実行します。

1. ボットに `/setup-files` を送信します。ボットはステータスと次のステップを返信します。
2. `/setup-files start` を送信します。ボットは OAuth URL を返信します。
3. その URL を開いて **Allow** をクリックすると、ブラウザが
   `http://localhost:1/?...&code=...` の読み込みに失敗します。この失敗は想定どおりです。
   認証コードは URL バーにあります。
4. 失敗した URL（または `code=...` の値だけ）をコピーし、`/setup-files <PASTED_URL>`
   としてチャットに貼り付け直します。ボットはこれをリフレッシュトークンと交換します。

トークンは `~/.hermes/google_chat_user_tokens/<sanitized_email>.json` に保存されます。
そのユーザーの DM での以降のファイルリクエストは*そのユーザーの*トークンを使うため、ボットは
そのユーザーとしてアップロードし、メッセージはそのユーザーのスペースに届きます。

後で取り消すには、`/setup-files revoke` でそのユーザーのトークンだけを削除します。他の
ユーザーのトークンには影響しません。

### スコープ

このフローが要求するスコープはちょうど 1 つ、`chat.messages.create` だけです。これは
`media.upload` と、アップロードした `attachmentDataRef` を参照する `messages.create` の
両方をカバーします。Drive も、より広い Chat スコープもありません。意図的に最小権限になっています。

### マルチユーザーの挙動

要求者にユーザーごとのトークンがまだない場合、ボットは
`~/.hermes/google_chat_user_token.json` にあるレガシーの単一ユーザートークンに
フォールバックします（マルチユーザー対応前のインストールから残っている場合）。どちらも
利用できない場合、ボットは要求者に `/setup-files` の実行を促す明確なテキスト通知を投稿します。

ユーザーが取り消しを行っても、自分のスロットだけがクリアされます。あるユーザーのトークンから
401/403 が返ると、そのユーザーのキャッシュだけが破棄されます。ユーザー同士が互いに干渉することは
ありません。

---

## トラブルシューティング

**「hola」を送ってもボットが沈黙したままです。**

1. コンソールで Pub/Sub サブスクリプションに未配信メッセージがあるか確認します。
   ある場合は Hermes が認証されていません。`GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` と、
   サービスアカウントがサブスクリプションの `Pub/Sub Subscriber` に登録されているか確認します。
2. サブスクリプションにメッセージがゼロの場合は、Google Chat が公開していません。
   **トピック**への IAM バインディングを再確認します。
   `chat-api-push@system.gserviceaccount.com` に `Pub/Sub Publisher` が必要です。
3. `hermes gateway` のログで `[GoogleChat] Connected` を確認します。
   `[GoogleChat] Config validation failed` が表示される場合は、エラーメッセージが
   どの環境変数を修正すべきか教えてくれます。

**ボットは返信するが、エージェントの回答ではなくエラーメッセージが表示されます。**

ログで `[GoogleChat] Pub/Sub stream died` を確認します。これが繰り返される場合、
サービスアカウントの認証情報がローテーションされたか、サブスクリプションが削除された
可能性があります。10 回試行した後、アダプターは自身を致命的としてマークします。

**送信メッセージのたびに「403 Forbidden」が出ます。**

ボットがスペースから削除されたか、Chat API コンソールで取り消されています。スペースに
再インストールしてください（次の `ADDED_TO_SPACE` イベントで自動的にメッセージングが
再有効化されます）。

**「Rate limit hit」の警告が多すぎます。**

Chat API のデフォルトのクォータでは、スペースごとに 1 分あたり 60 メッセージまで許可されます。
エージェントがこれを超える長いストリーミング応答を生成する場合、アダプターは指数バックオフで
リトライしますが、それでもユーザーに見えるレイテンシは発生します。簡潔な応答にするか、GCP
コンソールでクォータを引き上げることを検討してください。

**ボットがファイルの代わりに「/setup-files」の通知を投稿し続けます。**

要求者にユーザーごとの OAuth トークンがなく、レガシーのフォールバックもありません。その
ユーザーの DM で `/setup-files` を実行し、ステップ 10 に従ってください。交換が完了すれば、
次のファイルリクエストはゲートウェイの再起動なしにネイティブでアップロードされます。

**`/setup-files start` が「No client credentials stored on the host.」と表示します。**

1 回限りのホストセットアップが行われていません。Hermes を実行するホストのターミナルから
次を実行します。

```bash
python -m gateway.platforms.google_chat_user_oauth \
    --client-secret /path/to/client_secret.json
```

その後、再度 `/setup-files start` を送信します。

**`/setup-files <PASTED_URL>` が「Token exchange failed.」と表示します。**

認証コードは 1 回限りで有効期間が短い（通常は数分）です。`/setup-files start` を送信して
新しい URL を取得し、再試行してください。

---

## セキュリティに関する注意

- **サービスアカウントのスコープ**: アダプターは `chat.bot` と `pubsub` のスコープを
  要求します。実際の強制は IAM で行うべきです。サービスアカウントには最小限
  （サブスクリプションに対する `roles/pubsub.subscriber` ＋ `roles/pubsub.viewer`）を
  付与し、プロジェクトレベルや組織レベルの Pub/Sub ロールは付与しないでください。
- **添付ファイルのダウンロード保護**: Hermes は、ホストが Google 所有ドメインの短い
  許可リスト（`googleapis.com`、`drive.google.com`、`lh[3-6].googleusercontent.com`、
  その他いくつか）に一致する URL にのみ、サービスアカウントのベアラートークンを付与します。
  それ以外のホストは HTTP リクエスト前に拒否されます。これは、細工されたイベントがベアラー
  トークンを GCE メタデータサービスにリダイレクトする SSRF シナリオから保護するためです。
- **マスキング**: サービスアカウントのメールアドレス、サブスクリプションパス、トピックパスは、
  `agent/redact.py` によってログ出力から取り除かれます。デバッグ用エンベロープのダンプ
  （`GOOGLE_CHAT_DEBUG_RAW=1`）も同じマスキングフィルタを経由し、DEBUG レベルで記録されます。
- **コンプライアンス**: このボットを規制対象のワークスペース（データレジデンシーや
  AI ガバナンスのポリシーがあるもの）に接続する予定がある場合は、最初のインストール前に
  承認を得てください。
- **ユーザー OAuth のスコープ**: ユーザーごとの添付ファイルフローは*唯一*
  `chat.messages.create` のみを要求します。これは `media.upload` と後続の
  `messages.create` をカバーする最小限です。トークンは
  `~/.hermes/google_chat_user_tokens/<sanitized_email>.json` にプレーン JSON として
  永続化されます（保護はファイルシステムの権限であり、サービスアカウントのキーファイルと
  同じモデルです）。各トークンはちょうど 1 人のユーザーが所有し、取り消しはそのユーザーに
  スコープされます。
