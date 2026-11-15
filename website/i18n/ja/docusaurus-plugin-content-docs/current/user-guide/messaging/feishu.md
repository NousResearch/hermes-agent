---
sidebar_position: 11
title: "Feishu / Lark"
description: "Hermes AgentをFeishuまたはLarkのボットとしてセットアップする"
---

# Feishu / Lark のセットアップ

このページのPython依存関係のコマンドは、
[PMで準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を使用します。
依存関係を変更した後は、チェックアウトを再アクティブ化してHermesを再起動してください。

Hermes Agentは、フル機能のボットとしてFeishuおよびLarkと統合します。接続すると、ダイレクトメッセージやグループチャットでエージェントとチャットでき、ホームチャットでcronジョブの結果を受け取り、通常のゲートウェイフローを通じてテキスト、画像、音声、ファイル添付を送信できます。

この統合は両方の接続モードをサポートします。

- `websocket` — 推奨。Hermesがアウトバウンド接続を開くため、公開のwebhookエンドポイントは不要です
- `webhook` — Feishu/LarkにHTTP経由でイベントをゲートウェイにプッシュさせたい場合に便利です

## Hermesの振る舞い

| コンテキスト | 振る舞い |
|---------|----------|
| ダイレクトメッセージ | Hermesはすべてのメッセージに応答します。 |
| グループチャット | Hermesはチャット内でボットが@メンションされた場合にのみ応答します。 |
| 共有グループチャット | デフォルトでは、セッション履歴は共有チャット内でユーザーごとに分離されます。 |

この共有チャットの振る舞いは`config.yaml`で制御されます。

```yaml
group_sessions_per_user: true
```

チャットごとに1つの共有会話を明示的に望む場合にのみ`false`に設定してください。

## ステップ1: Feishu / Lark アプリの作成

### 推奨: スキャンして作成（1コマンド）

```bash
hermes gateway setup
```

**Feishu / Lark** を選択し、FeishuまたはLarkのモバイルアプリでQRコードをスキャンします。Hermesは正しい権限を持つボットアプリケーションを自動的に作成し、認証情報を保存します。

### 代替: 手動セットアップ

スキャンして作成が利用できない場合、ウィザードは手動入力にフォールバックします。

1. FeishuまたはLarkの開発者コンソールを開きます。
   - Feishu: [https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark: [https://open.larksuite.com/](https://open.larksuite.com/)
2. 新しいアプリを作成します。
3. **Credentials & Basic Info** で、**App ID** と **App Secret** をコピーします。
4. アプリの **Bot** 機能を有効化します。
5. `hermes gateway setup` を実行し、**Feishu / Lark** を選択して、プロンプトが表示されたら認証情報を入力します。

:::warning
App Secretは非公開に保ってください。これを持つ者は誰でもあなたのアプリになりすませます。
:::

### 権限の設定

Feishu開発者コンソールで **Permission Management** に移動し、次のスコープを追加します。権限ページで一括インポートできます。

**必須の権限:**

| スコープ | 目的 |
|-------|---------|
| `im:message` | メッセージの受信と読み取り |
| `im:message:send_as_bot` | ボットとしてメッセージを送信 |
| `im:resource` | ユーザーが送信した画像、ファイル、音声へのアクセス |
| `im:chat` | チャット/グループのメタデータへのアクセス |
| `im:chat:readonly` | チャット一覧とメンバーシップの読み取り |

**推奨の権限（全機能を使う場合）:**

| スコープ | 目的 |
|-------|---------|
| `im:message.reactions:readonly` | 絵文字リアクションイベントの受信 |
| `admin:app.info:readonly` | @メンションのゲーティングのためにボットのアイデンティティを自動検出 |
| `contact:user.id:readonly` | 許可リストの照合のためにユーザーIDを解決 |

### イベントの設定

**Events and Callbacks** で次を行います。

1. 接続モードを **Long Connection (WebSocket)**（推奨）に設定するか、webhook URLを設定します
2. **Event Configuration** タブで、次をサブスクライブします。
   - `im.message.receive_v1` — メッセージの受信に必須
3. **Callback Configuration** タブ（イベントとは別のタブ）で、同じ接続モードを設定し、`card.action.trigger`コールバックを追加します — 承認 / 更新プロンプトのボタンに必須です。[必要なFeishuアプリの設定](#required-feishu-app-configuration)を参照してください。

### アプリの公開

権限とイベントを設定した後、**Version Management** に移動してアプリの新しいバージョンを公開します。バージョンが公開され承認されるまで、権限は有効になりません（エンタープライズアプリの場合、管理者の承認が必要になることがあります）。

## ステップ2: 接続モードの選択

### 推奨: WebSocketモード

Hermesをラップトップ、ワークステーション、またはプライベートサーバーで実行する場合はWebSocketモードを使用します。公開URLは不要です。公式のLark SDKが、自動再接続を伴う持続的なアウトバウンドWebSocket接続を開いて維持します。

```bash
FEISHU_CONNECTION_MODE=websocket
```

**要件:** `websockets` Pythonパッケージがインストールされている必要があります。SDKが接続のライフサイクル、ハートビート、自動再接続を内部で処理します。

**仕組み:** アダプターは、Lark SDKのWebSocketクライアントをバックグラウンドのexecutorスレッドで実行します。インバウンドイベント（メッセージ、リアクション、カードアクション）はメインのasyncioループにディスパッチされます。切断時には、SDKが自動的に再接続を試みます。リンクが完全に切れた場合（SDKのリトライの段階が尽きた場合や、クライアントスレッドが終了した場合）、Hermesのスーパーバイザーが上限付きのバックオフでクライアントを再構築します。リンクがダウンしている間、`hermes gateway status`は接続が再確立されるまでプラットフォームを`retrying`として表示します。

### 任意: Webhookモード

到達可能なHTTPエンドポイントの背後で既にHermesを実行している場合にのみ、webhookモードを使用します。

```bash
FEISHU_CONNECTION_MODE=webhook
```

webhookモードでは、Hermesは（`aiohttp`経由で）HTTPサーバーを起動し、次の場所でFeishuエンドポイントを提供します。

```text
/feishu/webhook
```

**要件:** `aiohttp` Pythonパッケージがインストールされている必要があります。

webhookサーバーのバインドアドレスとパスはカスタマイズできます。

```bash
FEISHU_WEBHOOK_HOST=127.0.0.1   # default: 127.0.0.1
FEISHU_WEBHOOK_PORT=8765         # default: 8765
FEISHU_WEBHOOK_PATH=/feishu/webhook  # default: /feishu/webhook
```

FeishuがURL検証チャレンジ（`type: url_verification`）を送信すると、webhookは自動的に応答するため、Feishu開発者コンソールでサブスクリプションのセットアップを完了できます。`FEISHU_VERIFICATION_TOKEN`が設定されている場合、チャレンジ応答はそれによってゲートされます — トークンが欠落または不一致のチャレンジリクエストは拒否されるため、認証されていないリモートが攻撃者の制御するチャレンジデータをエコーしてエンドポイントの制御を証明することはできません。

## ステップ3: Hermesの設定

### オプションA: 対話型セットアップ

```bash
hermes gateway setup
```

**Feishu / Lark** を選択し、プロンプトに入力します。

### オプションB: 手動設定

`~/.hermes/.env`に次を追加します。

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=secret_xxx
FEISHU_DOMAIN=feishu
FEISHU_CONNECTION_MODE=websocket

# Optional but strongly recommended
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
FEISHU_HOME_CHANNEL=oc_xxx
```

`FEISHU_DOMAIN`は次を受け付けます。

- Feishu China用の`feishu`
- Lark international用の`lark`

## ステップ4: ゲートウェイの起動

```bash
hermes gateway
```

その後、Feishu/Larkからボットにメッセージを送り、接続が有効であることを確認します。

## ホームチャット

Feishu/Larkのチャットで`/set-home`を使い、それをcronジョブの結果とクロスプラットフォーム通知のためのホームチャンネルとしてマークします。

事前に設定しておくこともできます。

```bash
FEISHU_HOME_CHANNEL=oc_xxx
```

## セキュリティ

### ユーザー許可リスト

本番利用では、Feishu Open IDの許可リストを設定します。

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

許可リストを空のままにすると、ボットに到達できる者は誰でもそれを使える可能性があります。グループチャットでは、メッセージが処理される前に、送信者のopen_idに対して許可リストがチェックされます。

### Webhook暗号化キー

webhookモードで実行する場合、インバウンドのwebhookペイロードの署名検証を有効にするために暗号化キーを設定します。

```bash
FEISHU_ENCRYPT_KEY=your-encrypt-key
```

このキーは、Feishuアプリ設定の **Event Subscriptions** セクションにあります。設定すると、アダプターは次の署名アルゴリズムを使ってすべてのwebhookリクエストを検証します。

```
SHA256(timestamp + nonce + encrypt_key + body)
```

計算されたハッシュは、タイミングセーフな比較を使って`x-lark-signature`ヘッダーと照合されます。無効または欠落した署名を持つリクエストはHTTP 401で拒否されます。

:::tip
WebSocketモードでは、署名検証はSDK自体が処理するため、`FEISHU_ENCRYPT_KEY`は任意です。webhookモードでは、本番利用には強く推奨されます。
:::

### 検証トークン

webhookペイロード内の`token`フィールドをチェックする追加の認証レイヤーです。

```bash
FEISHU_VERIFICATION_TOKEN=your-verification-token
```

このトークンも、Feishuアプリの **Event Subscriptions** セクションにあります。設定すると、すべてのインバウンドwebhookペイロードは、その`header`オブジェクト内に一致する`token`を含む必要があります。一致しないトークンはHTTP 401で拒否されます。

`FEISHU_ENCRYPT_KEY`と`FEISHU_VERIFICATION_TOKEN`は、多層防御のために併用できます。

## グループメッセージのポリシー

`FEISHU_GROUP_POLICY`環境変数は、Hermesがグループチャットで応答するかどうか、またどのように応答するかを制御します。

```bash
FEISHU_GROUP_POLICY=allowlist   # default
```

| 値 | 振る舞い |
|-------|----------|
| `open` | Hermesは任意のグループ内の任意のユーザーからの@メンションに応答します。 |
| `allowlist` | Hermesは`FEISHU_ALLOWED_USERS`に列挙されたユーザーからの@メンションにのみ応答します。 |
| `disabled` | Hermesはすべてのグループメッセージを完全に無視します。 |

いずれのモードでも、メッセージが処理される前に、ボットがグループ内で明示的に@メンション（または@all）される必要があります。ダイレクトメッセージは常にこのゲートをバイパスします。

デフォルトの`allowlist`ポリシーで`FEISHU_ALLOWED_USERS`が空の場合、DMは引き続き動作しますが、人間からのグループメッセージはすべて拒否されます。最初の破棄は、設定すべきキーとともに`WARNING`で一度だけログに記録され、それ以降の破棄は`DEBUG`になります。[多重化ゲートウェイ](/docs/user-guide/multi-profile-gateways)では、各プロファイルは**自身の**`.env`のみを読み込みます — デフォルトプロファイルの`.env`にある`FEISHU_GROUP_POLICY=open`は、セカンダリプロファイルのボットには適用されません。`FEISHU_GROUP_POLICY` / `FEISHU_ALLOWED_USERS`を`profiles/<name>/.env`に置くか、そのプロファイルの`config.yaml`で`group_rules`を使ってください。

`FEISHU_REQUIRE_MENTION=false`を設定すると、@メンションを要求せずにHermesがすべてのグループのやり取りを読めるようになります。

```bash
FEISHU_REQUIRE_MENTION=false
```

チャットごとの制御については、`group_rules`エントリに`require_mention`を設定します — 後述の[グループごとのアクセス制御](#per-group-access-control)を参照してください。

### ボットのアイデンティティ

Hermesは起動時にボットの`open_id`と表示名を自動検出します。これらを手動で設定する必要があるのは、自動検出がFeishu APIに到達できない場合や、アプリがテナントスコープのユーザーIDを使用する場合のみです。

```bash
FEISHU_BOT_OPEN_ID=ou_xxx     # only when auto-detection fails
FEISHU_BOT_USER_ID=xxx        # required if your app uses sender_id_type=user_id
FEISHU_BOT_NAME=MyBot         # only when auto-detection fails
```

## ボット間メッセージング {#bot-to-bot-messaging}

デフォルトでは、Hermesは他のボットが送信したメッセージを無視します。HermesにA2Aオーケストレーションに参加させたい場合や、同じグループ内の他のボットから通知を受け取りたい場合に、ボット間メッセージングを有効にします。

```bash
FEISHU_ALLOW_BOTS=mentions   # default: none
```

| 値 | 振る舞い |
|-------|----------|
| `none` | 他のボットからのすべてのメッセージを無視（デフォルト）。 |
| `mentions` | ピアボットがHermesを@メンションした場合にのみ受け入れる。 |
| `all` | すべてのピアボットのメッセージを受け入れる。 |

`config.yaml`の`feishu.allow_bots`としても設定できます（両方が設定されている場合は環境変数が優先されます）。

ピアボットは`FEISHU_ALLOWED_USERS`に追加する必要はありません — その許可リストは人間の送信者にのみ適用されます。

ピアボット名を表示するには`application:bot.basic_info:read`スコープを付与します。これがない場合でも、ピアボットは正しくルーティングされますが、`open_id`として表示されます。

## インタラクティブカードアクション

ユーザーがボットから送信されたインタラクティブカードのボタンをクリックしたり操作したりすると、アダプターはこれらを合成的な`/card`コマンドイベントとしてルーティングします。

- ボタンのクリックは次のようになります: `/card button {"key": "value", ...}`
- カード定義からのアクションの`value`ペイロードがJSONとして含まれます。
- カードアクションは、二重処理を防ぐために15分のウィンドウで重複排除されます。

ゲートウェイ駆動の更新プロンプトは、プレーンテキストの返信にフォールバックする代わりに、ネイティブのFeishuの`Yes` / `No`カードを使用します。`hermes update --gateway`が確認を必要とする場合、アダプターは選択された回答をHermesの`.update_response`ファイルに記録し、カードをインラインで解決済みの状態に置き換えます。

カードアクションイベントは`MessageType.COMMAND`でディスパッチされるため、通常のコマンド処理パイプラインを通って流れます。

**確認の質問（clarify）** もカードを使います: エージェントが`clarify`ツールで多肢選択の質問をすると、カードには選択肢ごとに番号付きのボタンが1つずつと、「✏️ Other (type answer)」が表示されます。タップすると即座に回答され、カードはチャット内の全員に対してその場で更新されます。「Other」はテキスト入力の受け付けに切り替わり、次のメッセージが回答になります。タップする代わりに番号や選択肢のテキストを入力しても機能します。自由回答の質問はプレーンテキストで届きます。承認ボタンと同じ認可が適用されます: 許可されたユーザーのみが回答できます。

**有限の選択肢を持つスラッシュコマンド**（`/reasoning`、`/fast`）はカードピッカーを送信します: 1回のタップで設定が適用され、カードは選択内容で確定し、コマンドの確認がチャットに続きます。

これは**コマンド承認**の仕組みでもあります — エージェントが危険なコマンドを実行する必要があるとき、Allow Once / Session / Always / Denyのボタンを持つインタラクティブカードを送信します。ユーザーがボタンをクリックすると、カードアクションのコールバックが承認の決定をエージェントに返します。

### 必要なFeishuアプリの設定 {#required-feishu-app-configuration}

インタラクティブカードには、Feishu開発者コンソールで次の設定が必要です。ここに不足があると、通常はユーザーがカードボタンをクリックしたときにエラー**200340**が発生します。

1. **カードアクションのコールバックをサブスクライブする（イベントではありません）:**
   **Development Configuration > Events and Callbacks** で **Callback Configuration** タブを開き — これは`im.message.receive_v1`がある **Event Configuration** タブとは別です — *Subscribed Callbacks* の下に`card.action.trigger`を追加します。イベントとして追加しても、ボタンのクリックは配信されません。

2. **コールバックの配信モードを設定する:**
   同じタブで、Hermesを`websocket`モードで実行している場合は **Long Connection** を選択するか（Lark SDKが既存の接続でコールバックを受信します）、webhookモードではリクエストURL（イベントwebhookと同じエンドポイント、例: `https://your-server:8765/feishu/webhook`）を入力します。FeishuがそのURLに到達して名前解決できる必要があります。そうでないと、クリックは200342/200343で失敗します。

3. **Interactive Card機能を有効化する:**
   **App Features > Bot** で、**Interactive Card** トグルが有効になっていることを確認します。

4. **新しいアプリバージョンを公開する:**
   コールバックの変更は、**Version Management > Create version** が公開されて（エンタープライズアプリの場合は承認されて）初めて有効になります。Feishu自身による200340の説明は「the application has not configured the card callback address or the configured address is invalid … ensure that you have created and published the latest version of the app」です。

:::warning
公開されたカードコールバックがない場合でも、Feishuはインタラクティブカードの*送信*には成功しますが（送信には`im:message:send`権限だけが必要です）、いずれかのボタンをクリックするとエラー200340が返されます。カードは動作しているように見えます — エラーはユーザーがそれを操作したときにのみ表面化し、Feishuがコールバックを配信する前に拒否するため、クリックはHermesに到達しません（何もログに記録されません）。
:::

エラーコード200672 / 200673は、コールバックがHermesに*到達した*うえでFeishuが応答を拒否したことを示します。これらが表示された場合は、該当する`gateway.log`の行を添えてissueを登録してください。

## ドキュメントコメントのインテリジェント返信

チャットを超えて、アダプターは**Feishu/Larkのドキュメント**に残された`@`メンションにも応答できます。ユーザーがドキュメントにコメント（ローカルなテキスト選択またはドキュメント全体のコメント）して、ボットを@メンションすると、Hermesはドキュメントと周辺のコメントスレッドを読み、LLMの返信をスレッドにインラインで投稿します。

`drive.notice.comment_add_v1`イベントを使い、ハンドラは次を行います。

- ドキュメントの内容とコメントのタイムラインを並行して取得します（ドキュメント全体のスレッドには20メッセージ、ローカル選択のスレッドには12メッセージ）。
- その単一のコメントセッションにスコープされた`feishu_doc` + `feishu_drive`ツールセットでエージェントを実行します。
- 返信を4000文字でチャンク化し、スレッド返信として投稿し返します。
- 同じドキュメントへのフォローアップコメントがコンテキストを保てるよう、ドキュメントごとのセッションを50メッセージの上限付きで1時間キャッシュします。

### 3層のアクセス制御

ドキュメントコメントの返信は**明示的な付与のみ**です — 暗黙的なすべて許可モードはありません。権限は次の順序で解決されます（フィールドごとに最初の一致が優先されます）。

1. **完全一致のドキュメント** — 特定のドキュメントトークンにスコープされたルール。
2. **ワイルドカード** — ドキュメントのパターンに一致するルール。
3. **トップレベル** — ワークスペースのデフォルトルール。

ルールごとに2つのポリシーが利用できます。

- **`allowlist`** — ユーザー / テナントの静的なリスト。
- **`pairing`** — 静的リスト ∪ 実行時に承認されたストア。モデレーターがアクセスをライブで付与できるロールアウトに便利です。

ルールは`~/.hermes/feishu_comment_rules.json`（ペアリングの付与は`~/.hermes/feishu_comment_pairing.json`）に存在し、mtimeキャッシュのホットリロードが付いています — 編集はゲートウェイを再起動することなく次のコメントイベントで有効になります。

CLI:

```bash
# Inspect current rules and pairing state
python -m gateway.platforms.feishu_comment_rules status

# Simulate an access check for a specific doc + user
python -m gateway.platforms.feishu_comment_rules check <fileType:fileToken> <user_open_id>

# Manage pairing grants at runtime
python -m gateway.platforms.feishu_comment_rules pairing list
python -m gateway.platforms.feishu_comment_rules pairing add <user_open_id>
python -m gateway.platforms.feishu_comment_rules pairing remove <user_open_id>
```

### 必要なFeishuアプリの設定

既に付与されているチャット/カードの権限に加えて、ドライブコメントイベントを追加します。

- **Event Subscriptions** で`drive.notice.comment_add_v1`をサブスクライブします。
- ハンドラがドキュメントの内容を読めるよう、`docs:doc:readonly`と`drive:drive:readonly`のスコープを付与します。

## 会議招待イベント

人間の参加者を招待するのと同じ方法で、HermesのFeishu/Larkボットをビデオ会議に招待できます。ボットが会議招待イベントを受信すると、Hermesは会議への参加を試みるエージェントのターンを自動的に開始できます。

`vc.bot.meeting_invited_v1`イベントを使い、フローは次のとおりです。

- ユーザーがFeishu/Larkのビデオ会議にボットを招待します。
- Feishu/LarkがHermesに会議招待イベントを送信します。
- Hermesは招待者、会議のトピック、会議番号を抽出します。
- 招待者が通常のゲートウェイの許可リストまたはペアリングポリシーで認可されている場合、エージェントは会議番号を受け取り、自動的に参加を試みます。
- 招待の形式が不正な場合、またはエージェントが参加できない場合、Hermesはイベントを破棄するか、簡潔な説明とともに招待者に返信します。

招待者と`meeting_no`の両方を含まない不正な形式の招待は無視されます。

### 必要なFeishuアプリの設定

既に付与されているチャット/カードの権限に加えて、ビデオ会議の招待イベントを追加します。

- **Event Subscriptions** で`vc.bot.meeting_invited_v1`をサブスクライブします。
- そのイベントに対してFeishu/Lark開発者コンソールが求めるVideo Conferencingの権限スコープを有効にします。
- Hermesが招待者に返信できるよう、`im:message`と`im:message:send_as_bot`を有効のままにします。
- ゲートウェイのユーザー許可リストまたはペアリングポリシーで招待者が認可されていることを確認します。会議招待は通常のゲートウェイのアクセスチェックをバイパスしません。

## メディアのサポート

### インバウンド（受信）

アダプターは、ユーザーから次のメディアタイプを受信してキャッシュします。

| タイプ | 拡張子 | 処理方法 |
|------|-----------|-------------------|
| **画像** | .jpg, .jpeg, .png, .gif, .webp, .bmp | Feishu API経由でダウンロードしてローカルにキャッシュ |
| **音声** | .ogg, .mp3, .wav, .m4a, .aac, .flac, .opus, .webm | ダウンロードしてキャッシュ。小さなテキストファイルは自動抽出 |
| **動画** | .mp4, .mov, .avi, .mkv, .webm, .m4v, .3gp | ダウンロードしてドキュメントとしてキャッシュ |
| **ファイル** | .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx など | ダウンロードしてドキュメントとしてキャッシュ |

リッチテキスト（post）メッセージのメディアも抽出されてキャッシュされます — postの本文内のインライン画像/ファイルと、作成画面がトップレベルの`files`リストで送信する添付（1つのバブル内のキャプションとファイル）の両方です。すべての添付が収集されます。フォルダーのエントリはスキップされ、各添付はテキストに`[Attachment: <name>]`マーカーを残します。

小さなテキストベースのドキュメント（.txt、.md）については、ファイルの内容がメッセージテキストの後に自動的に追加されるため、エージェントはツールを必要とせずに直接読めます — ファイルと一緒に入力したキャプションはそのまま残ります。

### アウトバウンド（送信）

| メソッド | 送信するもの |
|--------|--------------|
| `send` | テキストまたはリッチpostメッセージ（markdownの内容に基づいて自動検出） |
| `send_image` / `send_image_file` | 画像をFeishuにアップロードし、ネイティブの画像バブルとして送信（任意のキャプション付き） |
| `send_document` | ファイルをFeishu APIにアップロードし、ファイル添付として送信 |
| `send_voice` | 音声ファイルをFeishuのファイル添付としてアップロード |
| `send_video` | 動画をアップロードし、ネイティブのメディアメッセージとして送信 |
| `send_animation` | GIFはファイル添付に格下げされる（FeishuにはネイティブのGIFバブルがない） |

ファイルのアップロードルーティングは拡張子に基づいて自動です。

- `.ogg`, `.opus` → `opus`音声としてアップロード
- `.mp4`, `.mov`, `.avi`, `.m4v` → `mp4`メディアとしてアップロード
- `.pdf`, `.doc(x)`, `.xls(x)`, `.ppt(x)` → そのドキュメントタイプでアップロード
- その他すべて → 汎用のストリームファイルとしてアップロード

## Markdownレンダリングとpostフォールバック

アウトバウンドのテキストにmarkdownの書式（見出し、太字、リスト、コードブロック、リンクなど）が含まれる場合、アダプターはそれをプレーンテキストとしてではなく、埋め込まれた`md`タグを持つFeishuの**post**メッセージとして自動的に送信します。これにより、Feishuクライアントでのリッチなレンダリングが可能になります。

Feishu APIがpostペイロードを拒否した場合（例えば、サポートされていないmarkdown構文のため）、アダプターは自動的にmarkdownを取り除いたプレーンテキストとしての送信にフォールバックします。この2段階のフォールバックにより、メッセージは常に配信されます。

プレーンテキストのメッセージ（markdownが検出されない場合）は、シンプルな`text`メッセージタイプとして送信されます。

## 処理ステータスのリアクション

エージェントが作業している間、ボットはあなたのメッセージに`Typing`リアクションを表示します。返信が届くとクリアされ、処理が失敗した場合は`CrossMark`に置き換えられます。

`FEISHU_REACTIONS=false`を設定するとオフにできます。

## バースト保護とバッチング

アダプターには、エージェントを圧倒しないように、急速なメッセージのバーストに対するデバウンスが含まれます。

### テキストバッチング

ユーザーが立て続けに複数のテキストメッセージを送信すると、それらはディスパッチされる前に単一のイベントにマージされます。

| 設定 | 環境変数 | デフォルト |
|---------|---------|---------|
| 静止期間 | `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | 0.6秒 |
| バッチごとの最大メッセージ数 | `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | 8 |
| バッチごとの最大文字数 | `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | 4000 |

### メディアバッチング

立て続けに送信された複数のメディア添付（例えば、複数の画像をドラッグした場合）は、単一のイベントにマージされます。

| 設定 | 環境変数 | デフォルト |
|---------|---------|---------|
| 静止期間 | `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | 0.8秒 |

### チャットごとの逐次処理

同じチャット内のメッセージは、会話の一貫性を保つために逐次（一度に1つずつ）処理されます。各チャットは独自のロックを持つため、異なるチャットのメッセージは並行して処理されます。

## レート制限（Webhookモード）

webhookモードでは、アダプターは悪用から保護するためにIPごとのレート制限を強制します。

- **ウィンドウ:** 60秒のスライディングウィンドウ
- **上限:** (app_id, path, IP) の組ごとにウィンドウあたり120リクエスト
- **追跡の上限:** 最大4096個のユニークキーを追跡（メモリの無制限な増大を防止）

上限を超えるリクエストはHTTP 429（Too Many Requests）を受け取ります。

検証トークンと署名のチェックを通過したリクエストのみが上限にカウントされるため、偽造されたトラフィックが本物のFeishuコールバックのためのウィンドウを使い切ることはできません。署名のない`url_verification`チャレンジは、検証トークンによって認証された場合にのみカウントされます。

### Webhookの異常検出

アダプターはIPアドレスごとの連続したエラー応答を追跡します。6時間のウィンドウ内に同じIPから25回連続でエラーが発生すると、警告がログに記録されます。これは、誤設定されたクライアントや探索の試みの検出に役立ちます。

追加のwebhook保護:
- **ボディサイズの上限:** 最大1 MB
- **ボディ読み取りのタイムアウト:** 30秒
- **Content-Typeの強制:** `application/json`のみ受け付ける

## WebSocketのチューニング {#websocket-tuning}

`websocket`モードを使う場合、再接続とpingの動作をカスタマイズできます。

```yaml
platforms:
  feishu:
    extra:
      ws_reconnect_interval: 120   # Seconds between reconnect attempts (default: 120)
      ws_ping_interval: 30         # Seconds between WebSocket pings (optional; SDK default if unset)
```

| 設定 | 設定キー | デフォルト | 説明 |
|---------|-----------|---------|-------------|
| 再接続間隔 | `ws_reconnect_interval` | 120秒 | 再接続試行間に待つ時間 |
| ping間隔 | `ws_ping_interval` | _(SDKのデフォルト)_ | WebSocketキープアライブpingの頻度 |

## グループごとのアクセス制御 {#per-group-access-control}

グローバルな`FEISHU_GROUP_POLICY`を超えて、config.yamlの`group_rules`を使ってグループチャットごとにきめ細かなルールを設定できます。

```yaml
platforms:
  feishu:
    extra:
      default_group_policy: "open"     # Default for groups not in group_rules
      admins:                          # Users who can manage bot settings
        - "ou_admin_open_id"
      group_rules:
        "oc_group_chat_id_1":
          policy: "allowlist"          # open | allowlist | blacklist | admin_only | disabled
          allowlist:
            - "ou_user_open_id_1"
            - "ou_user_open_id_2"
        "oc_group_chat_id_2":
          policy: "admin_only"
        "oc_group_chat_id_3":
          policy: "blacklist"
          blacklist:
            - "ou_blocked_user"
        "oc_free_chat":
          policy: "open"
          require_mention: false       # overrides FEISHU_REQUIRE_MENTION for this chat
```

| ポリシー | 説明 |
|--------|-------------|
| `open` | グループ内の誰でもボットを使える |
| `allowlist` | グループの`allowlist`にいるユーザーのみがボットを使える |
| `blacklist` | グループの`blacklist`にいるユーザーを除く全員がボットを使える |
| `admin_only` | グローバルな`admins`リストにいるユーザーのみがこのグループでボットを使える |
| `disabled` | ボットはこのグループのすべてのメッセージを無視する |

`group_rules`エントリに`require_mention: false`を設定すると、その特定のチャットについて@メンションの要件をスキップします。省略すると、チャットはグローバルな`FEISHU_REQUIRE_MENTION`の値を継承します。

`group_rules`に列挙されていないグループは、`default_group_policy`（デフォルトは`FEISHU_GROUP_POLICY`の値）にフォールバックします。

## 重複排除

インバウンドのメッセージは、24時間のTTLを持つメッセージIDを使って重複排除されます。重複排除の状態は再起動をまたいで`~/.hermes/feishu_seen_message_ids.json`に永続化されます。

| 設定 | 環境変数 | デフォルト |
|---------|---------|---------|
| キャッシュサイズ | `HERMES_FEISHU_DEDUP_CACHE_SIZE` | 2048エントリ |

## すべての環境変数

| 変数 | 必須 | デフォルト | 説明 |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | ✅ | — | Feishu/LarkのApp ID |
| `FEISHU_APP_SECRET` | ✅ | — | Feishu/LarkのApp Secret |
| `FEISHU_DOMAIN` | — | `feishu` | `feishu`（China）または`lark`（international） |
| `FEISHU_CONNECTION_MODE` | — | `websocket` | `websocket`または`webhook` |
| `FEISHU_ALLOWED_USERS` | — | _(空)_ | ユーザー許可リスト用のカンマ区切りopen_idリスト |
| `FEISHU_ALLOW_BOTS` | — | `none` | 他のボットからのメッセージを受け入れる: `none`、`mentions`、`all` |
| `FEISHU_REQUIRE_MENTION` | — | `true` | グループメッセージがボットを@メンションする必要があるかどうか |
| `FEISHU_HOME_CHANNEL` | — | — | cron/通知出力用のチャットID |
| `FEISHU_ENCRYPT_KEY` | — | _(空)_ | webhook署名検証用の暗号化キー |
| `FEISHU_VERIFICATION_TOKEN` | — | _(空)_ | webhookペイロード認証用の検証トークン |
| `FEISHU_GROUP_POLICY` | — | `allowlist` | グループメッセージポリシー: `open`、`allowlist`、`disabled` |
| `FEISHU_BOT_OPEN_ID` | — | _(空)_ | ボットのopen_id（@メンション検出用） |
| `FEISHU_BOT_USER_ID` | — | _(空)_ | ボットのuser_id（@メンション検出用） |
| `FEISHU_BOT_NAME` | — | _(空)_ | ボットの表示名（@メンション検出用） |
| `FEISHU_WEBHOOK_HOST` | — | `127.0.0.1` | webhookサーバーのバインドアドレス |
| `FEISHU_WEBHOOK_PORT` | — | `8765` | webhookサーバーのポート |
| `FEISHU_WEBHOOK_PATH` | — | `/feishu/webhook` | webhookエンドポイントのパス |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | — | `2048` | 追跡する重複排除済みメッセージIDの最大数 |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | — | `0.6` | テキストバーストのデバウンス静止期間 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | — | `8` | テキストバッチごとにマージする最大メッセージ数 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | — | `4000` | テキストバッチごとにマージする最大文字数 |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | — | `0.8` | メディアバーストのデバウンス静止期間 |

WebSocketおよびグループごとのACL設定は、`config.yaml`の`platforms.feishu.extra`の下で設定します（上記の[WebSocketのチューニング](#websocket-tuning)と[グループごとのアクセス制御](#per-group-access-control)を参照）。

## トラブルシューティング

| 問題 | 対処 |
|---------|-----|
| `lark-oapi not installed` | SDKをインストール: `python -c "import pm; pm.sync_venv(['feishu'], explicit=True)"` |
| `websockets not installed; websocket mode unavailable` | websocketsをインストール: `hermes pm repair` |
| `aiohttp not installed; webhook mode unavailable` | aiohttpをインストール: `python -c "import pm; pm.sync_venv(['messaging'], explicit=True)"` |
| `FEISHU_APP_ID or FEISHU_APP_SECRET not set` | 両方の環境変数を設定するか、`hermes gateway setup`経由で設定する |
| `Another local Hermes gateway is already using this Feishu app_id` | 同じapp_idを一度に使えるHermesインスタンスは1つだけです。先に他のゲートウェイを停止してください。 |
| ボットがグループで応答しない | ボットが@メンションされていることを確認し、`FEISHU_GROUP_POLICY`をチェックし、ポリシーが`allowlist`の場合は送信者が`FEISHU_ALLOWED_USERS`にいることを確認する |
| `Webhook rejected: invalid verification token` | `FEISHU_VERIFICATION_TOKEN`が、FeishuアプリのEvent Subscriptions設定のトークンと一致することを確認する |
| `Webhook rejected: invalid signature` | `FEISHU_ENCRYPT_KEY`が、Feishuアプリ設定の暗号化キーと一致することを確認する |
| postメッセージがプレーンテキストとして表示される | Feishu APIがpostペイロードを拒否しました。これは通常のフォールバック動作です。詳細はログを確認してください。 |
| 画像/ファイルがボットに受信されない | Feishuアプリに`im:message`と`im:resource`の権限スコープを付与する |
| ボットのアイデンティティが自動検出されない | 通常、Feishuのボット情報エンドポイントへの到達における一時的なネットワークの問題です。回避策として`FEISHU_BOT_OPEN_ID`と`FEISHU_BOT_NAME`を手動で設定してください。 |
| `FEISHU_ALLOW_BOTS`を有効にした後もピアボットのメッセージが無視される | Hermesがまだ自身を識別できていません — `FEISHU_BOT_OPEN_ID`（アプリが`sender_id_type=user_id`を使う場合は`FEISHU_BOT_USER_ID`も）を設定してください。 |
| ピアボットが名前ではなく`ou_xxxxxx`として表示される | `application:bot.basic_info:read`スコープを付与してください。 |
| 承認ボタンをクリックするとエラー200340（220340として表示されることもある） | 公開されたアプリバージョンに有効なカードコールバックがありません: **Callback Configuration** タブ（Event Configurationではありません）で`card.action.trigger`を追加し、Long Connection / リクエストURLを選択し、**Interactive Card**を有効にしてから、新しいバージョンを公開してください。[必要なFeishuアプリの設定](#required-feishu-app-configuration)を参照してください。クリックはHermesに到達しないため、`gateway.log`には何も表示されません。 |
| 承認ボタンをクリックするとエラー200342 / 200343 | webhookモード: FeishuがコールバックのリクエストURLに接続できない / 名前解決できません。URLを修正するか、Long Connectionに切り替えてください。 |
| `Webhook rate limit exceeded` | 同じIPから1分あたり120件の認証済みリクエストを超えました。これは通常、誤設定またはループです。 |

## ツールセット

Feishu / Larkは`hermes-feishu`プラットフォームプリセットを使用します。これは、Telegramや他のゲートウェイベースのメッセージングプラットフォームと同じコアツールを含みます。
