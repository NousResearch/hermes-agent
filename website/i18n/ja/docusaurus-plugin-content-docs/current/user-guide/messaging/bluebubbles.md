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

## 仕組み

```
iMessage → Messages.app → BlueBubbles Server → Webhook → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **インバウンド:** 新しいメッセージが届くと、BlueBubbles は Webhook イベントをローカルリスナーに送信します。ポーリングなし — 即時配信です。
- **アウトバウンド:** Hermes は BlueBubbles REST API 経由でメッセージを送信します。
- **メディア:** 画像、ボイスメッセージ、動画、ドキュメントは双方向でサポートされています。インバウンドの添付ファイルはダウンロードされ、エージェントが処理できるようにローカルにキャッシュされます。

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

## トラブルシューティング

### 「Cannot reach server」
- サーバー URL が正しいこと、Mac が起動していることを確認します
- BlueBubbles Server が実行されていることを確認します
- ネットワーク接続を確認します（ファイアウォール、ポート転送）

### メッセージが届かない
- BlueBubbles Server → Settings → API → Webhooks で Webhook が登録されていることを確認します
- Webhook URL が Mac から到達可能であることを確認します
- Webhook エラーについて `hermes logs gateway` を確認します（リアルタイムで追跡するには `hermes logs -f`）

### 「Private API helper not connected」
- Private API helper をインストールします: [docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- 基本的なメッセージングはこれなしで動作します — リアクション、タイピング、開封確認のみがこれを必要とします
