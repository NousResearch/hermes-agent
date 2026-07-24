---
sidebar_position: 6
title: "Signal"
description: "signal-cli デーモンを介して Hermes Agent を Signal メッセンジャーボットとしてセットアップします"
---

# Signal のセットアップ

Hermes は、HTTPモードで動作する [signal-cli](https://github.com/AsamK/signal-cli) デーモンを通じて Signal に接続します。アダプターは SSE（Server-Sent Events）を介してリアルタイムでメッセージをストリーミングし、JSON-RPC を介して応答を送信します。

Signal は、主流のメッセンジャーの中で最もプライバシーに重点を置いています。デフォルトでエンドツーエンド暗号化され、オープンソースのプロトコルで、メタデータの収集が最小限です。これにより、セキュリティが重要なエージェントワークフローに最適です。

:::info 新しい Python 依存関係は不要
Signal アダプターは、すべての通信に `httpx`（すでに Hermes のコア依存関係）を使用します。追加の Python パッケージは不要です。signal-cli を外部にインストールするだけで済みます。
:::

---

## 前提条件

- **signal-cli** — Java ベースの Signal クライアント（[GitHub](https://github.com/AsamK/signal-cli)）
- **Java 17+** ランタイム — signal-cli が必要とします
- **電話番号** — Signal がインストールされたもの（セカンダリデバイスとしてリンクするため）

### signal-cli のインストール

```bash
# macOS
brew install signal-cli

# Linux（最新リリースをダウンロード）
VERSION=$(curl -Ls -o /dev/null -w %{url_effective} \
  https://github.com/AsamK/signal-cli/releases/latest | sed 's/^.*\/v//')
curl -L -O "https://github.com/AsamK/signal-cli/releases/download/v${VERSION}/signal-cli-${VERSION}.tar.gz"
sudo tar xf "signal-cli-${VERSION}.tar.gz" -C /opt
sudo ln -sf "/opt/signal-cli-${VERSION}/bin/signal-cli" /usr/local/bin/
```

:::caution
signal-cli は apt や snap のリポジトリには **ありません**。上記の Linux のインストールは、[GitHub リリース](https://github.com/AsamK/signal-cli/releases) から直接ダウンロードします。
:::

---

## ステップ1: Signal アカウントをリンクする

signal-cli は **リンクされたデバイス** として動作します。WhatsApp Web に似ていますが、Signal 向けです。あなたの電話がプライマリデバイスのままです。

```bash
# リンク用 URI を生成（QRコードまたはリンクを表示）
signal-cli link -n "HermesAgent"
```

1. 電話で **Signal** を開きます
2. **設定 → リンクされたデバイス** に移動します
3. **新しいデバイスをリンク** をタップします
4. QRコードをスキャンするか、URI を入力します

---

## ステップ2: signal-cli デーモンを起動する

```bash
# +1234567890 を自分の Signal 電話番号（E.164 形式）に置き換えてください
signal-cli --account +1234567890 daemon --http 127.0.0.1:8080
```

:::tip
これをバックグラウンドで実行し続けてください。`systemd`、`tmux`、`screen` を使うか、サービスとして実行できます。
:::

実行されていることを確認します。

```bash
curl http://127.0.0.1:8080/api/v1/check
# 次のような出力になるはずです: {"versions":{"signal-cli":...}}
```

---

## ステップ3: Hermes を設定する

最も簡単な方法:

```bash
hermes gateway setup
```

プラットフォームメニューから **Signal** を選択します。ウィザードは次のことを行います。

1. signal-cli がインストールされているかを確認
2. HTTP URL を入力（デフォルト: `http://127.0.0.1:8080`）
3. デーモンへの接続をテスト
4. アカウントの電話番号を入力
5. 許可するユーザーとアクセスポリシーを設定

### 手動設定

`~/.hermes/.env` に追加します。

```bash
# 必須
SIGNAL_HTTP_URL=http://127.0.0.1:8080
SIGNAL_ACCOUNT=+1234567890

# セキュリティ（推奨）
SIGNAL_ALLOWED_USERS=+1234567890,+0987654321    # カンマ区切りの E.164 番号または UUID

# 任意
SIGNAL_GROUP_ALLOWED_USERS=groupId1,groupId2     # グループを有効化（省略で無効、* ですべて）
SIGNAL_HOME_CHANNEL=+1234567890                  # cronジョブのデフォルト配信先
```

その後、ゲートウェイを起動します。

```bash
hermes gateway              # フォアグラウンド
hermes gateway install      # ユーザーサービスとしてインストール
sudo hermes gateway install --system   # Linux のみ: 起動時のシステムサービス
```

---

## アクセス制御

### DM アクセス

DM アクセスは、他のすべての Hermes プラットフォームと同じパターンに従います。

1. **`SIGNAL_ALLOWED_USERS` を設定** → これらのユーザーのみがメッセージを送信できます
2. **許可リストを未設定** → 不明なユーザーには DM ペアリングコードが送られます（`hermes pairing approve signal CODE` で承認）
3. **`SIGNAL_ALLOW_ALL_USERS=true`** → 誰でもメッセージを送信できます（注意して使用してください）

### グループアクセス

グループアクセスは `SIGNAL_GROUP_ALLOWED_USERS` 環境変数で制御します。

| 設定 | 動作 |
|---------------|----------|
| 未設定（デフォルト） | すべてのグループメッセージは無視されます。ボットは DM にのみ応答します。 |
| グループIDで設定 | リストされたグループのみが監視されます（例: `groupId1,groupId2`）。 |
| `*` に設定 | ボットは、メンバーであるすべてのグループで応答します。 |

---

## 機能

### 添付ファイル

アダプターは、双方向でのメディアの送受信をサポートします。

**受信**（ユーザー → エージェント）:

- **画像** — PNG、JPEG、GIF、WebP（マジックバイトで自動検出）
- **音声** — MP3、OGG、WAV、M4A（Whisper が設定されていれば音声メッセージを文字起こし）
- **ドキュメント** — PDF、ZIP、その他のファイルタイプ

**送信**（エージェント → ユーザー）:

エージェントは、応答内の `MEDIA:` タグを介してメディアファイルを送信できます。以下の配信方法がサポートされています。

- **画像** — `send_multiple_images` と `send_image_file` は、PNG、JPEG、GIF、WebP をネイティブな Signal 添付ファイルとして送信します
- **音声** — `send_voice` は、音声ファイル（OGG、MP3、WAV、M4A、AAC）を添付ファイルとして送信します
- **動画** — `send_video` は、MP4 動画ファイルを送信します
- **ドキュメント** — `send_document` は、任意のファイルタイプ（PDF、ZIP など）を送信します

すべての送信メディアは、Signal の標準的な添付ファイル API を経由します。一部のプラットフォームとは異なり、Signal はプロトコルレベルで音声メッセージとファイル添付を区別しません。

添付ファイルのサイズ制限: **100 MB**（双方向）。
:::warning
**Signal サーバーは添付ファイルのアップロードをレート制限します**。アダプターは、複数画像の送信にスケジューラーを使用し、画像を32個ずつのグループにまとめ、Signal サーバーのポリシーに合わせてアップロードをスロットリングします。
:::

### ネイティブな整形、引用返信、リアクション

Signal メッセージは、リテラルな markdown 文字ではなく **ネイティブな整形** でレンダリングされます。アダプターは markdown（`**bold**`、`*italic*`、`` `code` ``、`~~strike~~`、`||spoiler||`、見出し）を Signal の `bodyRanges` に変換するため、テキストは受信者のクライアント上で目に見える `**` / `` ` `` 文字としてではなく、実際のスタイルで表示されます。

**引用返信。** Hermes が特定のメッセージに返信すると、元のメッセージを引用するネイティブな返信を投稿するようになりました。Signal ユーザーが自分で「返信」を使うときに見るのと同じ UI の振る舞いです。これは受信メッセージへの応答として生成される返信に対して自動的に行われます。

**リアクション。** エージェントは標準のリアクション API を介してメッセージにリアクションできます。リアクションは、余分なテキストとしてではなく、参照されたメッセージへの絵文字リアクションとして Signal に表示されます。

これらに追加の設定は不要です。最近の signal-cli ビルドではデフォルトで有効になっています。`signal-cli` のバージョンが古すぎる場合、Hermes はプレーンテキスト配信にフォールバックし、一度だけ警告をログに記録します。

### 入力中インジケーター

ボットはメッセージの処理中に入力中インジケーターを送信し、8秒ごとに更新します。

### 電話番号の秘匿化

すべての電話番号は、ログ内で自動的に秘匿化されます。
- `+15551234567` → `+155****4567`
- これは Hermes ゲートウェイのログとグローバルな秘匿化システムの両方に適用されます

### Note to Self（単一番号のセットアップ）

別個のボット番号ではなく、自分の電話番号で signal-cli を **リンクされたセカンダリデバイス** として実行している場合、Signal の「Note to Self」機能を通じて Hermes とやり取りできます。

電話から自分自身にメッセージを送るだけで、signal-cli がそれを拾い、Hermes が同じ会話で応答します。

**仕組み:**
- 「Note to Self」メッセージは `syncMessage.sentMessage` エンベロープとして到着します
- アダプターは、これらがボット自身のアカウント宛であることを検出し、通常の受信メッセージとして処理します
- エコーバック保護（送信タイムスタンプの追跡）が無限ループを防ぎます。ボット自身の返信は自動的に除外されます

**追加の設定は不要です。** `SIGNAL_ACCOUNT` が自分の電話番号と一致している限り、自動的に機能します。

### ヘルスモニタリング

アダプターは SSE 接続を監視し、次の場合に自動的に再接続します。
- 接続が切断された場合（指数バックオフ付き: 2s → 60s）
- 120秒間アクティビティが検出されなかった場合（signal-cli に ping を送って確認）

---

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| セットアップ中の **"Cannot reach signal-cli"** | signal-cli デーモンが実行されていることを確認: `signal-cli --account +YOUR_NUMBER daemon --http 127.0.0.1:8080` |
| **メッセージが受信されない** | `SIGNAL_ALLOWED_USERS` に送信者の番号が E.164 形式（`+` プレフィックス付き）で含まれているか確認 |
| **"signal-cli not found on PATH"** | signal-cli をインストールし、PATH に含まれていることを確認するか、Docker を使用 |
| **接続が切れ続ける** | signal-cli のログでエラーを確認。Java 17+ がインストールされているか確認。 |
| **グループメッセージが無視される** | `SIGNAL_GROUP_ALLOWED_USERS` に特定のグループIDを設定するか、`*` ですべてのグループを許可。 |
| **ボットが誰にも応答しない** | `SIGNAL_ALLOWED_USERS` を設定するか、DM ペアリングを使用するか、より広いアクセスを望むならゲートウェイポリシーで明示的にすべてのユーザーを許可。 |
| **メッセージの重複** | 自分の電話番号をリッスンする signal-cli インスタンスが1つだけであることを確認 |

---

## セキュリティ

:::warning
**常にアクセス制御を設定してください。** ボットはデフォルトでターミナルアクセスを持ちます。`SIGNAL_ALLOWED_USERS` または DM ペアリングがない場合、ゲートウェイは安全策としてすべての受信メッセージを拒否します。
:::

- 電話番号はすべてのログ出力で秘匿化されます
- 新しいユーザーを安全にオンボーディングするには、DM ペアリングまたは明示的な許可リストを使用してください
- グループサポートが特に必要な場合を除きグループを無効のままにするか、信頼するグループのみを許可リストに入れてください
- Signal のエンドツーエンド暗号化は、転送中のメッセージ内容を保護します
- `~/.local/share/signal-cli/` 内の signal-cli セッションデータには、アカウントの認証情報が含まれています。パスワードのように保護してください

---

## 環境変数リファレンス

| 変数 | 必須 | デフォルト | 説明 |
|----------|----------|---------|-------------|
| `SIGNAL_HTTP_URL` | はい | — | signal-cli HTTP エンドポイント |
| `SIGNAL_ACCOUNT` | はい | — | ボットの電話番号（E.164） |
| `SIGNAL_ALLOWED_USERS` | いいえ | — | カンマ区切りの電話番号／UUID |
| `SIGNAL_GROUP_ALLOWED_USERS` | いいえ | — | 監視するグループID、またはすべてに `*`（省略でグループ無効） |
| `SIGNAL_ALLOW_ALL_USERS` | いいえ | `false` | 任意のユーザーの操作を許可（許可リストをスキップ） |
| `SIGNAL_HOME_CHANNEL` | いいえ | — | cronジョブのデフォルト配信先 |
