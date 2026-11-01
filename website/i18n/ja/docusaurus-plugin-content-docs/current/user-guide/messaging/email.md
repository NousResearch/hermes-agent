---
sidebar_position: 7
title: "メール"
description: "Hermes Agent を IMAP/SMTP 経由のメールアシスタントとしてセットアップ"
---

# メールセットアップ

Hermes は標準的な IMAP と SMTP プロトコルを使ってメールを受信・返信できます。エージェントのアドレスにメールを送ると、スレッド内で返信します — 特別なクライアントやボット API は不要です。Gmail、Outlook、Yahoo、Fastmail、または IMAP/SMTP をサポートする任意のプロバイダーで動作します。

:::info 外部依存なし
メールアダプターは Python 組み込みの `imaplib`、`smtplib`、`email` モジュールを使用します。追加のパッケージや外部サービスは不要です。
:::

---

## 前提条件

- Hermes エージェント用の**専用メールアカウント**（個人用メールは使わないでください）
- メールアカウントで **IMAP が有効化**されていること
- Gmail や 2FA を使う他のプロバイダーの場合は**アプリパスワード**

### Gmail のセットアップ

1. Google アカウントで 2 段階認証を有効化する
2. [アプリパスワード](https://myaccount.google.com/apppasswords)にアクセスする
3. 新しいアプリパスワードを作成する（「メール」または「その他」を選択）
4. 16 文字のパスワードをコピーする — 通常のパスワードの代わりにこれを使います

### Outlook / Microsoft 365

1. [セキュリティ設定](https://account.microsoft.com/security)にアクセスする
2. 2FA がまだ有効でなければ有効化する
3. 「追加のセキュリティオプション」でアプリパスワードを作成する
4. IMAP ホスト: `outlook.office365.com`、SMTP ホスト: `smtp.office365.com`

### その他のプロバイダー

ほとんどのメールプロバイダーは IMAP/SMTP をサポートしています。次の点についてプロバイダーのドキュメントを確認してください:
- IMAP ホストとポート（通常は SSL でポート 993）
- SMTP ホストとポート（通常は STARTTLS でポート 587）
- アプリパスワードが必要かどうか

---

## ステップ 1: Hermes を設定する

最も簡単な方法:

```bash
hermes gateway setup
```

プラットフォームメニューから **Email** を選択します。ウィザードがメールアドレス、パスワード、IMAP/SMTP ホスト、許可する送信者を尋ねます。

### 手動設定

`~/.hermes/.env` に追加します:

```bash
# 必須
EMAIL_ADDRESS=hermes@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop    # アプリパスワード（通常のパスワードではない）
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com

# セキュリティ（推奨）
EMAIL_ALLOWED_USERS=your@email.com,colleague@work.com

# 任意
EMAIL_IMAP_PORT=993                    # デフォルト: 993（IMAP SSL）
EMAIL_SMTP_PORT=587                    # デフォルト: 587（SMTP STARTTLS）
EMAIL_POLL_INTERVAL=15                 # 受信箱チェックの間隔（秒）（デフォルト: 15）
EMAIL_HOME_ADDRESS=your@email.com      # Cron ジョブのデフォルト配信先
```

---

## ステップ 2: ゲートウェイを起動する

```bash
hermes gateway              # フォアグラウンドで実行
hermes gateway install      # ユーザーサービスとしてインストール
sudo hermes gateway install --system   # Linux のみ: 起動時のシステムサービス
```

起動時に、アダプターは:
1. IMAP と SMTP の接続をテストします
2. 既存の受信箱メッセージをすべて「既読」としてマークします（新しいメールのみ処理）
3. 新しいメッセージのポーリングを開始します

---

## 仕組み

### メッセージの受信

アダプターは設定可能な間隔（デフォルト: 15 秒）で IMAP 受信箱の UNSEEN メッセージをポーリングします。新しいメールごとに:

- **件名**がコンテキストとして含まれます（例: `[Subject: Deploy to production]`）
- **返信メール**（件名が `Re:` で始まる）は件名のプレフィックスをスキップします — スレッドのコンテキストは既に確立されています
- **添付ファイル**はローカルにキャッシュされます:
  - 画像（JPEG、PNG、GIF、WebP）→ vision ツールで利用可能
  - ドキュメント（PDF、ZIP など）→ ファイルアクセス用に利用可能
- **HTML のみのメール**はプレーンテキスト抽出のためにタグが除去されます
- **自己メッセージ**は返信ループを防ぐためにフィルタリングされます
- **自動/noreply の送信者**は静かに無視されます — `noreply@`、`mailer-daemon@`、`bounce@`、`no-reply@`、および `Auto-Submitted`、`Precedence: bulk`、`List-Unsubscribe` ヘッダーを持つメール

### 返信の送信

返信は適切なメールスレッディングを伴って SMTP で送信されます:

- **In-Reply-To** と **References** ヘッダーがスレッドを維持します
- **件名**は `Re:` プレフィックス付きで保持されます（二重の `Re: Re:` にはなりません）
- **Message-ID** がエージェントのドメインで生成されます
- 応答はプレーンテキスト（UTF-8）で送信されます

### ファイル添付

エージェントは返信にファイル添付を含められます。応答に `MEDIA:/path/to/file` を含めると、そのファイルが送信メールに添付されます。

### 添付ファイルのスキップ

すべての受信添付ファイルを無視する（マルウェア対策や帯域節約のため）には、`config.yaml` に追加します:

```yaml
platforms:
  email:
    skip_attachments: true
```

有効にすると、ペイロードのデコード前に添付パートとインラインパートがスキップされます。メール本文のテキストは通常どおり処理されます。

---

## アクセス制御

メールアクセスは他のすべての Hermes プラットフォームと同じパターンに従います:

1. **`EMAIL_ALLOWED_USERS` が設定済み** → それらのアドレスからのメールのみ処理されます
2. **許可リスト未設定** → 未知の送信者にはペアリングコードが付与されます
3. **`EMAIL_ALLOW_ALL_USERS=true`** → 任意の送信者が受け入れられます（慎重に使用）

:::warning
**常に `EMAIL_ALLOWED_USERS` を設定してください。** これがないと、エージェントのメールアドレスを知る誰もがコマンドを送れてしまいます。エージェントはデフォルトでターミナルアクセスを持っています。
:::

---

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| 起動時に **「IMAP connection failed」** | `EMAIL_IMAP_HOST` と `EMAIL_IMAP_PORT` を確認してください。アカウントで IMAP が有効か確認してください。Gmail の場合、設定 → 転送と POP/IMAP で有効化します。 |
| 起動時に **「SMTP connection failed」** | `EMAIL_SMTP_HOST` と `EMAIL_SMTP_PORT` を確認してください。パスワードが正しいか確認してください（Gmail はアプリパスワードを使用）。 |
| **メッセージが受信されない** | `EMAIL_ALLOWED_USERS` に送信者のメールが含まれているか確認してください。迷惑メールフォルダを確認してください — 一部のプロバイダーは自動返信をフラグします。 |
| **「Authentication failed」** | Gmail では通常のパスワードではなくアプリパスワードを使う必要があります。先に 2FA を有効にしてください。 |
| **返信が重複する** | ゲートウェイインスタンスが 1 つだけ実行されているか確認してください。`hermes gateway status` を確認してください。 |
| **応答が遅い** | デフォルトのポーリング間隔は 15 秒です。より速い応答のために `EMAIL_POLL_INTERVAL=5` で短縮できます（ただし IMAP 接続は増えます）。 |
| **返信がスレッド化されない** | アダプターは In-Reply-To ヘッダーを使用します。一部のメールクライアント（特に Web ベース）は自動メッセージで正しくスレッド化しないことがあります。 |

---

## セキュリティ

:::warning
**専用のメールアカウントを使ってください。** 個人用メールは使わないでください — エージェントはパスワードを `.env` に保存し、IMAP 経由で受信箱への完全なアクセスを持ちます。
:::

- メインのパスワードではなく**アプリパスワード**を使う（Gmail で 2FA の場合は必須）
- エージェントと対話できる相手を制限するため `EMAIL_ALLOWED_USERS` を設定する
- パスワードは `~/.hermes/.env` に保存されます — このファイルを保護してください（`chmod 600`）
- IMAP はデフォルトで SSL（ポート 993）、SMTP は STARTTLS（ポート 587）を使用します — 接続は暗号化されています

---

## 環境変数リファレンス

| 変数 | 必須 | デフォルト | 説明 |
|----------|----------|---------|-------------|
| `EMAIL_ADDRESS` | はい | — | エージェントのメールアドレス |
| `EMAIL_PASSWORD` | はい | — | メールパスワードまたはアプリパスワード |
| `EMAIL_IMAP_HOST` | はい | — | IMAP サーバーホスト（例: `imap.gmail.com`） |
| `EMAIL_SMTP_HOST` | はい | — | SMTP サーバーホスト（例: `smtp.gmail.com`） |
| `EMAIL_IMAP_PORT` | いいえ | `993` | IMAP サーバーポート |
| `EMAIL_SMTP_PORT` | いいえ | `587` | SMTP サーバーポート |
| `EMAIL_POLL_INTERVAL` | いいえ | `15` | 受信箱チェックの間隔（秒） |
| `EMAIL_ALLOWED_USERS` | いいえ | — | カンマ区切りの許可送信者アドレス |
| `EMAIL_HOME_ADDRESS` | いいえ | — | Cron ジョブのデフォルト配信先 |
| `EMAIL_ALLOW_ALL_USERS` | いいえ | `false` | すべての送信者を許可（非推奨） |
