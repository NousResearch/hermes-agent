# BlueBubbles（iMessage）

[BlueBubbles](https://bluebubbles.app/) 経由で Hermes を Apple iMessage に接続します。BlueBubbles は、iMessage を任意のデバイスにブリッジする無料のオープンソース macOS サーバーです。

## 前提条件

- [BlueBubbles Server](https://bluebubbles.app/) を実行している **Mac**（常時起動）
- その Mac の Messages.app にサインインした Apple ID
- BlueBubbles Server v1.0.0 以上（Webhook にはこのバージョンが必要です）
- Hermes と BlueBubbles サーバー間のネットワーク接続

## セットアップ

### 1. BlueBubbles Server をインストールする

[bluebubbles.app](https://bluebubbles.app/) からダウンロードしてインストールします。セットアップウィザードを完了します — Apple ID でサインインし、接続方法（ローカルネットワーク、Ngrok、Cloudflare、または Dynamic DNS）を設定します。

### 2. サーバー URL とパスワードを取得する

BlueBubbles Server → **Settings → API** で、以下をメモします:
- **Server URL**（例: `http://192.168.1.10:1234`）
- **Server Password**

### 3. Hermes を設定する

セットアップウィザードを実行します:

```bash
hermes gateway setup
```

**BlueBubbles (iMessage)** を選択し、サーバー URL とパスワードを入力します。

または、`~/.hermes/.env` で環境変数を直接設定します:

```bash
BLUEBUBBLES_SERVER_URL=http://192.168.1.10:1234
BLUEBUBBLES_PASSWORD=your-server-password
```

#### オプション: グループチャットでメンションを必須にする

デフォルトでは、Hermes は認可されたすべての BlueBubbles/iMessage の DM またはグループメッセージに応答します。グループチャットをオプトイン方式にするには、メンションゲーティングを有効にします:

```yaml
platforms:
  bluebubbles:
    enabled: true
    extra:
      require_mention: true
```

`require_mention: true` の場合、DM は通常どおり動作しますが、グループチャットのメッセージはメンションパターンに一致しない限り無視されます。カスタムパターンを設定しない場合、Hermes は `Hermes` と `@Hermes agent` のバリエーションに対する控えめなデフォルトを使用します。

カスタムのエージェント名を使う場合は、正規表現パターンを設定します:

```yaml
platforms:
  bluebubbles:
    extra:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

### 4. ユーザーを認可する

いずれかのアプローチを選択します:

**DM ペアリング（推奨）:**
誰かがあなたの iMessage にメッセージを送ると、Hermes は自動的にペアリングコードを送信します。以下で承認します:
```bash
hermes pairing approve bluebubbles <CODE>
```
`hermes pairing list` を使用して、保留中のコードと承認済みユーザーを確認できます。

**特定のユーザーを事前認可する**（`~/.hermes/.env` 内）:
```bash
BLUEBUBBLES_ALLOWED_USERS=user@icloud.com,+15551234567
```

**オープンアクセス**（`~/.hermes/.env` 内）:
```bash
BLUEBUBBLES_ALLOW_ALL_USERS=true
```

### 5. ゲートウェイを起動する

```bash
hermes gateway run
```

Hermes は BlueBubbles サーバーに接続し、Webhook を登録し、iMessage メッセージのリッスンを開始します。

### 6. セットアップを確認する

セットアップウィザードは認証情報を `~/.hermes/.env` に保存します — `~/.hermes/config.yaml` に `platforms.bluebubbles.enabled` は書き込み**ません**。明示的な設定がない場合は、認証情報が存在すればアダプターは起動します。しかし、明示的な `enabled: false` は常に認証情報より優先されるため、以前にアダプターを無効にしていた場合（たとえば別の iMessage ブリッジを使っていたとき）、ウィザードは成功を報告しても、アダプターは決して起動しません。

保存されている設定を確認します:

```bash
hermes config get platforms.bluebubbles.enabled
```

- `true` — 明示的に有効
- `Config key not set` — 明示的な設定なし。`.env` の認証情報によって有効化が決まります
- `false` — 明示的に無効。セットアップを再実行してもこれは切り替わりません — 自分で `true` に設定してください

次に、ゲートウェイのログで、接続と Webhook 登録を示す 2 行を確認します:

```bash
hermes logs gateway
```

```text
[bluebubbles] connected to http://192.168.1.10:1234 (private_api=False, helper=False)
[bluebubbles] webhook registered with server: http://localhost:8645/bluebubbles-webhook?password=***
```

最後に、別のデバイスから自分宛てにテストメッセージを送信します。返信（新しい DM の場合はペアリングコード）が届けば、インバウンド配信がエンドツーエンドで動作していることの証明になります。

## 仕組み

```
iMessage → Messages.app → BlueBubbles Server → Webhook → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **インバウンド:** 新しいメッセージが届くと、BlueBubbles は Webhook イベントをローカルリスナーに送信します。ポーリングなし — 即時配信です。
- **アウトバウンド:** Hermes は BlueBubbles REST API 経由でメッセージを送信します。
- **メディア:** 画像、ボイスメッセージ、動画、ドキュメントは双方向でサポートされています。インバウンドの添付ファイルはダウンロードされ、エージェントが処理できるようにローカルにキャッシュされます。

### 2 つの URL、逆方向

このセットアップでは、逆方向を指す 2 つの URL を使います — 混同しないでください:

- `BLUEBUBBLES_SERVER_URL`（例: `http://192.168.1.10:1234`）— Hermes が BlueBubbles サーバーの API を**呼び出し**ます。これは BlueBubbles Server → Settings → API に表示される Server URL です。
- Webhook（デフォルト `http://localhost:8645/bluebubbles-webhook`）— BlueBubbles が新着メッセージのイベントを Hermes に **POST** します。ホスト/ポート/パスは `BLUEBUBBLES_WEBHOOK_HOST` / `BLUEBUBBLES_WEBHOOK_PORT` / `BLUEBUBBLES_WEBHOOK_PATH` で決まります。

### Webhook の登録方法

BlueBubbles の UI で Webhook を作成する必要は**ありません**。ゲートウェイが接続すると、Hermes が BlueBubbles REST API（`/api/v1/webhook`）経由で `new-message` と `updated-message` イベント用の Webhook を自ら登録し、正常なシャットダウン時に登録を再び削除します。

知っておくべき 2 つの詳細:

- BlueBubbles の Webhook API はカスタムヘッダーをサポートしていないため、登録される URL にはサーバーパスワードがクエリパラメータ（`?password=…`）として含まれます — これによってインバウンドイベントが認証されます。
- Webhook リスナーはデフォルトで `127.0.0.1` にバインドします。Hermes と BlueBubbles が同じマシンで動作している場合は問題ありません。別々のマシンで動作している場合は、`BLUEBUBBLES_WEBHOOK_HOST` を BlueBubbles を実行している Mac から到達可能なアドレスに設定してください。

## 環境変数

| 変数 | 必須 | デフォルト | 説明 |
|----------|----------|---------|-------------|
| `BLUEBUBBLES_SERVER_URL` | はい | — | BlueBubbles サーバー URL |
| `BLUEBUBBLES_PASSWORD` | はい | — | サーバーパスワード |
| `BLUEBUBBLES_WEBHOOK_HOST` | いいえ | `127.0.0.1` | Webhook リスナーのバインドアドレス |
| `BLUEBUBBLES_WEBHOOK_PORT` | いいえ | `8645` | Webhook リスナーのポート |
| `BLUEBUBBLES_WEBHOOK_PATH` | いいえ | `/bluebubbles-webhook` | Webhook URL パス |
| `BLUEBUBBLES_HOME_CHANNEL` | いいえ | — | cron 配信用の電話番号/メールアドレス |
| `BLUEBUBBLES_ALLOWED_USERS` | いいえ | — | カンマ区切りの認可済みユーザー |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | いいえ | `false` | すべてのユーザーを許可 |
| `BLUEBUBBLES_REQUIRE_MENTION` | いいえ | `false` | グループチャットで応答する前にメンションパターンを必須にする |
| `BLUEBUBBLES_MENTION_PATTERNS` | いいえ | Hermes のウェイクワード | グループのメンション照合用の正規表現パターン（JSON 配列、改行区切り、またはカンマ区切り） |

メッセージを自動的に既読としてマークする機能は、`~/.hermes/config.yaml` の `platforms.bluebubbles.extra` 配下にある `send_read_receipts` キーで制御されます（デフォルト: `true`）。対応する環境変数はありません。

## 機能

### テキストメッセージング
iMessage を送受信します。Markdown は自動的に除去され、クリーンなプレーンテキストで配信されます。

### リッチメディア
- **画像:** 写真は iMessage の会話内にネイティブに表示されます
- **ボイスメッセージ:** 音声ファイルは iMessage のボイスメッセージとして送信されます
- **動画:** 動画の添付ファイル
- **ドキュメント:** ファイルは iMessage の添付ファイルとして送信されます

### Tapback リアクション
Love、like、dislike、laugh、emphasize、question のリアクション。BlueBubbles の [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation) が必要です。

### タイピングインジケーター
エージェントが処理している間、iMessage の会話に「入力中...」を表示します。Private API が必要です。

### 開封確認
処理後にメッセージを自動的に既読としてマークします。Private API が必要です。

### チャットのアドレス指定
メールアドレスまたは電話番号でチャットを指定できます — Hermes はそれらを BlueBubbles のチャット GUID に自動的に解決します。生の GUID 形式を使用する必要はありません。

## Private API

一部の機能には BlueBubbles の [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation) が必要です:
- Tapback リアクション
- タイピングインジケーター
- 開封確認
- アドレスによる新規チャットの作成

Private API がなくても、基本的なテキストメッセージングとメディアは引き続き動作します。

注意点が 1 つあります: 「Private API なしでも基本的なメッセージングは動作する」というのは、BlueBubbles がログイン中の macOS ユーザーのフォアグラウンドで Messages.app を操作できることが前提です。BlueBubbles を実行している Mac がログイン画面のままであったり、サーバーユーザーから切り替えられていたり（ファストユーザスイッチ）すると、AppleScript ベースの送信は失敗します — [BlueBubbles: multiple users on the same Mac](https://docs.bluebubbles.app/server/basic-guides/multiple-users-on-the-same-mac) を参照してください。そのようなセットアップでは、上記の追加機能のためだけでなく、確実な送信のためにも Private API が必要です。

## トラブルシューティング

### 「Cannot reach server」
- サーバー URL が正しいこと、Mac が起動していることを確認します
- BlueBubbles Server が実行されていることを確認します
- ネットワーク接続を確認します（ファイアウォール、ポート転送）

### メッセージが届かない
- Webhook エラーについて `hermes logs gateway` を確認します（リアルタイムで追跡するには `hermes logs -f`）
- Hermes は接続時に Webhook を自ら登録します — ログに登録失敗が表示された場合にのみ、BlueBubbles Server → Settings → API → Webhooks を確認してください
- BlueBubbles の UI に Webhook の行があっても配信の証明にはなりません。エンドツーエンドの証明は、Hermes がメッセージをログに記録して返信することです
- Hermes と BlueBubbles が別々のマシンで動作している場合、デフォルトの Webhook バインドアドレス `127.0.0.1` には Mac から到達できません — `BLUEBUBBLES_WEBHOOK_HOST` を到達可能なアドレスに設定し、ゲートウェイを再起動してください

### セットアップは成功したが、アダプターが起動しない
- `hermes gateway setup` は認証情報を `~/.hermes/.env` に保存します。`platforms.bluebubbles.enabled: true` は設定しません
- `~/.hermes/config.yaml` の明示的な `enabled: false` は、認証情報の存在より優先されます — `hermes config get platforms.bluebubbles.enabled` で確認してください
- これは iMessage ブリッジを切り替えた後によく起こります: 別の iMessage ブリッジを使っていてその際に BlueBubbles を無効にしていた場合、セットアップを再実行しても再び有効にはなりません。`enabled: true` を設定してください（そして使わなくなったブリッジは無効にしてください — 2 つの iMessage ブリッジがあるとメッセージが二重に処理されます）

### 1 台の Mac に 2 つの BlueBubbles サーバー（間違った Apple ID）
- Hermes は、BlueBubbles の UI に表示される Server URL（DHCP の変更後に古くなっている可能性があります）ではなく、`~/.hermes/.env` の `BLUEBUBBLES_SERVER_URL` を使います
- 1 台の Mac 上の 2 人の macOS ユーザーは、それぞれ独自の API ポートと Apple ID を持つ BlueBubbles サーバーを実行します — Hermes がどちらに到達しているかを確認してください: `curl "http://<server-url>/api/v1/server/info?password=<password>"` を実行して `computer_id` を比較します
- マルチユーザーのセットアップ自体については、[BlueBubbles: multiple users on the same Mac](https://docs.bluebubbles.app/server/basic-guides/multiple-users-on-the-same-mac) に従ってください — ユーザーごとに 1 つのポートを使い、サーバーを実行しているユーザーをログアウトさせないでください

### 返信が重複する
- 既知の問題: セッション処理により、1 人の相手が 2 つのセッション（生の GUID 形式と電話番号/メールアドレス形式）に分かれることがあります — [#30708](https://github.com/NousResearch/hermes-agent/issues/30708) と [#34372](https://github.com/NousResearch/hermes-agent/issues/34372) で追跡されています
- ドキュメントや設定の問題ではありません — 修正についてはこれらの Issue を追ってください

### 「♻️ Recovered reply」が繰り返される、または送信が数分間停止する
- `BLUEBUBBLES_HOME_CHANNEL`（または `platforms.bluebubbles.home_channel`）が設定されている場合、ゲートウェイの再起動やプラットフォームの再接続時にホームチャンネルに通知され、保留中の返信が「♻️ Recovered reply」として再配信されることがあります
- BlueBubbles 側の送信が黙って失敗している場合 — 典型的には、バックグラウンドの macOS ユーザーで長いタイムアウトを伴う `Not authorized to send Apple events to Messages. (-1743)` のような AppleScript エラー — 再試行が積み重なり、Webhook 配信を圧迫することがあります
- 送信が確実に動作することが確認できるまでホームチャンネルは空のままにし、バックグラウンドユーザーのセットアップでは Private API helper を有効にしてください（[Private API](#private-api) を参照）

### 「Private API helper not connected」
- Private API helper をインストールします: [docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- 基本的なメッセージングはこれなしで動作します — リアクション、タイピング、開封確認のみがこれを必要とします
