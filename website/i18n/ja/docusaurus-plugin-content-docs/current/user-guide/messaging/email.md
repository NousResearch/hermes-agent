---
sidebar_position: 7
title: "メール"
description: "Hermes Agent を IMAP/SMTP 経由のメールアシスタントとしてセットアップ"
---

# メールセットアップ

Hermes は標準的な IMAP と SMTP プロトコルを使ってメールを受信・返信できます。エージェントのアドレスにメールを送ると、スレッド内で返信します — 特別なクライアントやボット API は不要です。Gmail、Outlook、Yahoo、Fastmail、または IMAP/SMTP をサポートする任意のプロバイダーで動作します。

:::info ゲートウェイアダプターのみ: 外部依存なし
このページでは、Python 組み込みの `imaplib`、`smtplib`、`email` モジュールを使用するメールゲートウェイアダプターについて説明します。このゲートウェイ経路では、追加のパッケージや外部サービスは不要です。
:::

これは、同梱の [Himalaya メールスキル](/docs/user-guide/skills/bundled/email/email-himalaya)とは別物です。Himalaya メールスキルは、エージェントがターミナルコマンドを通じてメールを管理できるようにするもので、外部の `himalaya` CLI と Himalaya の設定ファイルが必要です。

| ユースケース | 設定するもの | 外部依存 |
|---|---|---|
| 人が Hermes エージェントにメールを送り、返信を受け取れるようにする | このページのメールゲートウェイアダプター | IMAP/SMTP のメールアカウント以外は不要 |
| エージェントがターミナルツールからメールボックスのメッセージを確認、作成、移動、管理できるようにする | Himalaya メールスキル | `himalaya` CLI と `~/.config/himalaya/config.toml` |

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

### Proton Mail Bridge / ローカルリレー

Proton Mail Bridge（およびセルフホストの MTA などの同様のローカルリレー）は、
ループバック上で **STARTTLS** と自己署名証明書を使ってリッスンするため、デフォルト設定
（IMAP 993 での暗黙的 TLS、検証済み証明書）では接続できません。
`~/.hermes/config.yaml` でトランスポートを上書きしてください:

```yaml
platforms:
  email:
    enabled: true
    extra:
      imap_host: 127.0.0.1
      imap_security: starttls     # tls (default) | starttls | plain
      imap_tls_verify: false      # Bridge uses a self-signed cert
      smtp_host: 127.0.0.1
      smtp_security: starttls     # default: tls on port 465, starttls otherwise
      smtp_tls_verify: false
```

そして `~/.hermes/.env` に、Bridge の認証情報と一緒に `EMAIL_IMAP_PORT=1143` / `EMAIL_SMTP_PORT=1025` を
設定します。不明な `*_security` の値は警告をログに記録し、安全なデフォルトにフォールバックします。
`*_tls_verify` を無効にするのはループバックホストに対してのみにしてください —
それ以外のホストで検証がオフになっている場合、Hermes は警告をログに記録します。

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
# Required
EMAIL_ADDRESS=hermes@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop    # App password (not your regular password)
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com

# Security (recommended)
EMAIL_ALLOWED_USERS=your@email.com,colleague@work.com
# Required: exact authserv-id on your receiving server's topmost
# Authentication-Results header (Gmail: mx.google.com).
# That server must strip inbound headers claiming this id before adding its own.
EMAIL_AUTHSERV_ID=mx.google.com

# Optional
EMAIL_IMAP_PORT=993                    # Default: 993 (IMAP SSL)
EMAIL_SMTP_PORT=587                    # Default: 587 (SMTP STARTTLS)
EMAIL_POLL_INTERVAL=15                 # Seconds between inbox checks (default: 15)
EMAIL_HOME_ADDRESS=your@email.com      # Default delivery target for cron jobs
```

---

## ステップ 2: ゲートウェイを起動する

```bash
hermes gateway              # Run in foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
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

メールのアクセスは、チャット型のプラットフォームよりもデフォルトで厳格です:

1. **`EMAIL_ALLOWED_USERS` が設定済み** → それらのアドレス（および `GATEWAY_ALLOWED_USERS` に含まれるアドレスや承認済みのペアリング）からのメールのみ処理されます
2. **許可リスト未設定** → 未知の送信者は静かに無視されます
3. **`EMAIL_ALLOW_ALL_USERS=true`** → 任意の送信者が受け入れられます（慎重に使用）
4. **`platforms.email.unauthorized_dm_behavior: pair`** → 未知の送信者はペアリングコードを受け取ります
5. **`platforms.email.unauthorized_dm_behavior: decline`** → 未知の送信者は丁寧な拒否を 1 回受け取り、その後 24 時間は何も受け取りません

許可リストのエントリはアドレス全体と照合されます。`alice` のような裸のエントリ（たとえば `GATEWAY_ALLOWED_USERS` 内のチャットのユーザー名）は、どのドメインの `alice@` も許可しません。そのようなアドレスからのメールは、ペアリングや拒否の対象にはならず破棄されます。

Hermes がメッセージに対応するのは、受信サーバーが付与した `Authentication-Results` ヘッダーがその `From:` ドメインを認証している場合（DMARC、またはアラインメントの取れた SPF/DKIM）のみです。これはオープンアクセス（`EMAIL_ALLOW_ALL_USERS` / `GATEWAY_ALLOW_ALL_USERS`）でも適用されます。オープンアクセスはどの送信者も許可しますが、各会話とすべての返信は `From:` をキーにしているため、偽造された `From:` があると、攻撃者のテキストがそのアドレスのセッションに入り込み、Hermes がそのアドレスにメールを送ることになります。ペアリングコードと拒否も同様に認証済みの `From:` を必要とするため、偽造されたアドレスにはどちらも送信されません。メールサーバーがそのヘッダーを付与しない場合は、`platforms.email.require_authenticated_sender: false`（または `EMAIL_TRUST_FROM_HEADER=true`）を設定してリスクを受け入れてください。そうしないと、許可された送信者からのメールでもそのようなヘッダーがないものは、ゲートウェイのログに `Dropping sender with unauthenticated From` という警告を出して破棄されます。最上位の `Authentication-Results` ヘッダー（サーバーが先頭に付加するもの。それより下のヘッダーは参照されません）が、固定した `authserv_id` と完全に一致する名前を持たない場合にも、`platforms.email.authserv_id`（`EMAIL_AUTHSERV_ID`）を指し示す同じ警告が出ます。`authserv_id` は、サーバーがそのヘッダーの最初の `;` の前に書き込む authserv-id と完全に一致しなければなりません。`EMAIL_ALLOWED_USERS` / `GATEWAY_ALLOWED_USERS` に記載されたアドレスからのメールがそれ以外の理由で認証に失敗した場合（たとえば DKIM が壊れている送信者や、アラインメントを壊す転送者）は、同じ警告とその判定結果とともに破棄され、オプトアウトのヒントは表示されません。その他の送信者からの失敗したメール（オープンアクセス下での偽造された `From:` など）は、デバッグレベルでのみログに記録され、静かに破棄されます。

**送信者認証がオンの間は、`authserv_id` の固定が必須です。** `platforms.email.authserv_id`（`EMAIL_AUTHSERV_ID`）がないと、Hermes はサーバーの `Authentication-Results` ヘッダーと送信者が書いたヘッダーを区別できない（自前のヘッダーを付加しないセルフホストの MTA では、送信者のヘッダーが最上位に残ります）ため、どれも信頼しません。認証済みの `From:` を必要とするすべてのメッセージが破棄され、ゲートウェイのログに修正方法とともに一度だけその旨が記録されます。値を調べるには、サーバーが配信したメッセージを開き（「メッセージのソースを表示」/ 生のソース）、**最上位**の `Authentication-Results` ヘッダーの最初の `;` の前にあるトークンをコピーします（たとえば Gmail では `mx.google.com`）。そのトークンがメッセージごとに変わるように見える場合、それは authserv-id ではなくリレーごとのホスト名です。RFC 8601 は authserv-id を管理ドメインの安定した名前として定義しているため、MTA が安定した ID を付与するよう設定し（たとえば OpenDKIM / OpenDMARC の `AuthservID`）、それを固定してください。ID が一致することは固定であって、出所の証明ではありません。MTA が自身のヘッダーを先頭に付加する前に、その ID を名乗る受信 `Authentication-Results` ヘッダーを取り除くことを確認してください。安定した ID を固定できない場合は、`platforms.email.require_authenticated_sender: false`（または `EMAIL_TRUST_FROM_HEADER=true`）で明示的にオプトアウトし、その場合 `From:` は書かれたとおりに信頼されることを受け入れてください。

:::warning
**通常の運用では、専用の受信箱を使い、`EMAIL_ALLOWED_USERS` を設定してください。** 共有の受信箱には無関係な未読メッセージが含まれていることが多く、Hermes がデフォルトでそれらの連絡先に返信すべきではないため、メールのペアリングはオプトインになっています。
:::

---

## トラブルシューティング

| 問題 | 解決策 |
|---------|----------|
| 起動時に **「IMAP connection failed」** | `EMAIL_IMAP_HOST` と `EMAIL_IMAP_PORT` を確認してください。アカウントで IMAP が有効か確認してください。Gmail の場合、設定 → 転送と POP/IMAP で有効化します。 |
| 起動時に **「SMTP connection failed」** | `EMAIL_SMTP_HOST` と `EMAIL_SMTP_PORT` を確認してください。パスワードが正しいか確認してください（Gmail はアプリパスワードを使用）。 |
| **メッセージが受信されない** | `EMAIL_ALLOWED_USERS` に送信者のメールが含まれているか確認してください。ゲートウェイのログに `authserv-id is not configured` と表示される場合は、`EMAIL_AUTHSERV_ID` を、サーバーが書き込む最上位の `Authentication-Results` ヘッダーの ID と完全に一致するよう設定してください。迷惑メールフォルダを確認してください — 一部のプロバイダーは自動返信をフラグします。 |
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
- `EMAIL_AUTHSERV_ID` を受信 MTA の最上位の `Authentication-Results` スタンプの ID と完全に一致するよう設定し、その MTA がその ID を名乗る受信ヘッダーを取り除くことを確認する
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
| `EMAIL_AUTHSERV_ID` | はい（送信者認証がオフの場合を除く） | — | 受信サーバーの最上位の `Authentication-Results` ヘッダーにある authserv-id（完全一致）。これがないと、認証済みの `From:` を必要とするすべてのメッセージが破棄されます |
| `EMAIL_HOME_ADDRESS` | いいえ | — | Cron ジョブのデフォルト配信先 |
| `EMAIL_ALLOW_ALL_USERS` | いいえ | `false` | すべての送信者を許可（非推奨） |
