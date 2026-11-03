---
sidebar_position: 9
title: "Matrix"
description: "Hermes AgentをMatrixボットとしてセットアップする"
---

# Matrixセットアップ

Hermes Agentは、オープンで連合型のメッセージングプロトコルであるMatrixと統合されます。Matrixでは、自分のホームサーバーを運用することも、matrix.orgのような公開ホームサーバーを使うこともできます — いずれにしても、通信のコントロールはあなたが保持します。ボットは`mautrix` Python SDK経由で接続し、メッセージをHermes Agentのパイプライン（ツールの使用、メモリ、推論を含む）で処理し、リアルタイムで応答します。テキスト、ファイル添付、画像、音声、動画、そして任意のエンドツーエンド暗号化（E2EE）をサポートします。

Hermesは、Synapse、Conduit、Dendrite、matrix.orgなど、あらゆるMatrixホームサーバーで動作します。

セットアップの前に、多くの人が知りたい部分を説明します。Hermesが接続された後、どう振る舞うかです。

## Hermesの振る舞い

| コンテキスト | 振る舞い |
|---------|----------|
| **DM** | Hermesはすべてのメッセージに応答します。`@mention`は不要です。各DMは独自のセッションを持ちます。DMでボットが`@mention`されたときにスレッドを開始するには、`MATRIX_DM_MENTION_THREADS=true`を設定します。 |
| **ルーム** | デフォルトでは、Hermesは応答に`@mention`を要求します。`MATRIX_REQUIRE_MENTION=false`を設定するか、フリーレスポンスルームとして`MATRIX_FREE_RESPONSE_ROOMS`にルームIDを追加します。ルームへの招待は自動承諾されます。 |
| **スレッド** | HermesはMatrixのスレッド（MSC3440）をサポートします。スレッド内で返信すると、Hermesはスレッドのコンテキストをメインルームのタイムラインから分離して保ちます。ボットがすでに参加したスレッドでは、メンションは不要です。 |
| **自動スレッド化** | デフォルトでは、Hermesはルームで応答する各メッセージについてスレッドを自動作成します。これにより会話が分離されます。無効化するには`MATRIX_AUTO_THREAD=false`を設定します。 |
| **複数ユーザーがいる共有ルーム** | デフォルトでは、Hermesはルーム内のユーザーごとにセッション履歴を分離します。同じルームで会話する2人は、明示的に無効化しない限り、1つのトランスクリプトを共有しません。 |

:::tip
ボットは招待されると自動的にルームに参加します。ボットのMatrixユーザーを任意のルームに招待するだけで、参加して応答を始めます。
:::

### Matrixでのセッションモデル

デフォルトでは:

- 各DMは独自のセッションを持つ
- 各スレッドは独自のセッション名前空間を持つ
- 共有ルーム内の各ユーザーは、そのルーム内で独自のセッションを持つ

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

### メンションとスレッド化の設定

メンションと自動スレッド化の振る舞いは、環境変数または`config.yaml`で設定できます。

```yaml
matrix:
  require_mention: true           # ルームで @mention を要求する（デフォルト: true）
  free_response_rooms:            # メンション要求から免除されるルーム
    - "!abc123:matrix.org"
  auto_thread: true               # 応答用にスレッドを自動作成（デフォルト: true）
  dm_mention_threads: false       # DMで @mention されたときスレッドを作成（デフォルト: false）
```

または環境変数で:

```bash
MATRIX_REQUIRE_MENTION=true
MATRIX_FREE_RESPONSE_ROOMS=!abc123:matrix.org,!def456:matrix.org
MATRIX_AUTO_THREAD=true
MATRIX_DM_MENTION_THREADS=false
MATRIX_REACTIONS=true          # デフォルト: true — 処理中の絵文字リアクション
```

:::tip リアクションの無効化
`MATRIX_REACTIONS=false`は、ボットが受信メッセージに付ける処理ライフサイクルの絵文字リアクション（👀/✅/❌）をオフにします。リアクションイベントがノイズになる、または参加するすべてのクライアントがサポートしていないルームに便利です。
:::

:::note
`MATRIX_REQUIRE_MENTION`がなかったバージョンからアップグレードする場合、以前はボットがルーム内のすべてのメッセージに応答していました。その動作を維持するには、`MATRIX_REQUIRE_MENTION=false`を設定してください。
:::

このガイドでは、ボットアカウントの作成から最初のメッセージの送信まで、完全なセットアッププロセスを順を追って説明します。

## ステップ1: ボットアカウントを作成する

ボット用のMatrixユーザーアカウントが必要です。これにはいくつかの方法があります。

### オプションA: 自分のホームサーバーで登録する（推奨）

自分のホームサーバー（Synapse、Conduit、Dendrite）を運用している場合:

1. admin APIまたは登録ツールを使って、新しいユーザーを作成します。

```bash
# Synapse の例
register_new_matrix_user -c /etc/synapse/homeserver.yaml http://localhost:8008
```

2. `hermes`のようなユーザー名を選びます — 完全なユーザーIDは`@hermes:your-server.org`になります。

### オプションB: matrix.orgまたは別の公開ホームサーバーを使う

1. [Element Web](https://app.element.io)にアクセスし、新しいアカウントを作成します。
2. ボットのユーザー名（例: `hermes-bot`）を選びます。

### オプションC: 自分のアカウントを使う

Hermesを自分のユーザーとして実行することもできます。これはボットがあなたとして投稿することを意味します — 個人アシスタントに便利です。

## ステップ2: アクセストークンを取得する

Hermesは、ホームサーバーで認証するためにアクセストークンが必要です。2つのオプションがあります。

### オプションA: アクセストークン（推奨）

トークンを取得する最も信頼できる方法:

**Element経由:**
1. ボットアカウントで[Element](https://app.element.io)にログインします。
2. **Settings** → **Help & About**に移動します。
3. 下にスクロールして**Advanced**を展開します — アクセストークンがそこに表示されます。
4. **すぐにコピーしてください。**

**API経由:**

```bash
curl -X POST https://your-server/_matrix/client/v3/login \
  -H "Content-Type: application/json" \
  -d '{
    "type": "m.login.password",
    "user": "@hermes:your-server.org",
    "password": "your-password"
  }'
```

レスポンスには`access_token`フィールドが含まれます — それをコピーします。

:::warning[アクセストークンを安全に保ってください]
アクセストークンは、ボットのMatrixアカウントへの完全なアクセスを与えます。決して公開で共有したりGitにコミットしたりしないでください。漏洩した場合は、そのユーザーのすべてのセッションからログアウトして失効させてください。
:::

### オプションB: パスワードログイン

アクセストークンを提供する代わりに、ボットのユーザーIDとパスワードをHermesに与えられます。Hermesは起動時に自動的にログインします。これはよりシンプルですが、パスワードが`.env`ファイルに保存されることを意味します。

```bash
MATRIX_USER_ID=@hermes:your-server.org
MATRIX_PASSWORD=your-password
```

## ステップ3: あなたのMatrixユーザーIDを見つける

Hermes Agentは、誰がボットと対話できるかを制御するために、あなたのMatrixユーザーIDを使います。MatrixのユーザーIDは`@username:server`の形式に従います。

自分のものを見つけるには:

1. [Element](https://app.element.io)（またはお好みのMatrixクライアント）を開きます。
2. アバターをクリック → **Settings**。
3. ユーザーIDがプロフィールの上部に表示されます（例: `@alice:matrix.org`）。

:::tip
MatrixのユーザーIDは常に`@`で始まり、`:`の後にサーバー名が続きます。例: `@alice:matrix.org`、`@bob:your-server.com`。
:::

## ステップ4: Hermes Agentを設定する

### オプションA: インタラクティブセットアップ（推奨）

ガイド付きのセットアップコマンドを実行します。

```bash
hermes gateway setup
```

プロンプトが表示されたら**Matrix**を選択し、尋ねられたらホームサーバーのURL、アクセストークン（またはユーザーID＋パスワード）、許可するユーザーIDを提供します。

### オプションB: 手動設定

`~/.hermes/.env`ファイルに次を追加します。

**アクセストークンを使う場合:**

```bash
# 必須
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# 任意: ユーザーID（省略するとトークンから自動検出）
# MATRIX_USER_ID=@hermes:matrix.example.org

# セキュリティ: ボットと対話できる人を制限する
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# 複数の許可ユーザー（カンマ区切り）
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

**パスワードログインを使う場合:**

```bash
# 必須
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_USER_ID=@hermes:matrix.example.org
MATRIX_PASSWORD=***

# セキュリティ
MATRIX_ALLOWED_USERS=@alice:matrix.example.org
```

`~/.hermes/config.yaml`での任意の動作設定:

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true`は、共有ルーム内で各参加者のコンテキストを分離して保ちます

### ゲートウェイを起動する

設定したら、Matrixゲートウェイを起動します。

```bash
hermes gateway
```

ボットは数秒以内にホームサーバーに接続し、同期を始めるはずです。テストのため、メッセージを送ってください — DMでも、参加しているルームでも構いません。

:::tip
`hermes gateway`をバックグラウンドで、またはsystemdサービスとして実行すると、永続的に動作させられます。詳細はデプロイのドキュメントを参照してください。
:::

## エンドツーエンド暗号化（E2EE）

HermesはMatrixのエンドツーエンド暗号化をサポートするため、暗号化されたルームでボットとチャットできます。

### 要件

E2EEには、暗号化エクストラ付きの`mautrix`ライブラリと、`libolm` Cライブラリが必要です。

```bash
# E2EEサポート付きで mautrix をインストール
pip install 'mautrix[encryption]'

# または hermes エクストラ付きでインストール
pip install 'hermes-agent[matrix]'
```

システムに`libolm`もインストールする必要があります。

```bash
# Debian/Ubuntu
sudo apt install libolm-dev

# macOS
brew install libolm

# Fedora
sudo dnf install libolm-devel
```

### E2EEを有効化する

`~/.hermes/.env`に追加します。

```bash
MATRIX_ENCRYPTION=true
```

E2EEが有効になると、Hermesは:

- 暗号化キーを`~/.hermes/platforms/matrix/store/`に保存します（レガシーインストール: `~/.hermes/matrix/store/`）
- 初回接続時にデバイスキーをアップロードします
- 受信メッセージを復号し、送信メッセージを自動的に暗号化します
- 招待されると暗号化されたルームに自動参加します

### クロスサイニング検証（推奨）

Matrixアカウントでクロスサイニングが有効になっている場合（Elementのデフォルト）、起動時にボットが自身のデバイスを自己署名できるよう、リカバリーキーを設定してください。これがないと、デバイスキーのローテーション後に、他のMatrixクライアントがボットと暗号化セッションを共有することを拒否することがあります。

```bash
MATRIX_RECOVERY_KEY=EsT... your recovery key here
```

**見つける場所:** Elementで、**Settings** → **Security & Privacy** → **Encryption** → あなたのリカバリーキー（「セキュリティキー」とも呼ばれる）に移動します。これは、クロスサイニングを最初にセットアップしたときに保存するよう求められたキーです。

各起動時、`MATRIX_RECOVERY_KEY`が設定されていれば、Hermesはホームサーバーのセキュアなシークレットストレージからクロスサイニングキーをインポートし、現在のデバイスに署名します。これは冪等であり、永続的に有効にしておいて安全です。

:::warning[暗号ストアの削除]
`~/.hermes/platforms/matrix/store/crypto.db`を削除すると、ボットは暗号化アイデンティティを失います。同じデバイスIDで単に再起動しても、**完全には**回復しません — ホームサーバーは、古いアイデンティティキーで署名されたワンタイムキーをまだ保持しており、ピアは新しいOlmセッションを確立できません。

Hermesは起動時にこの状況を検出し、E2EEの有効化を拒否して、次のようにログ出力します: `device XXXX has stale one-time keys on the server signed with a previous identity key`。

**最も簡単な回復: 新しいアクセストークンを生成する**（古いキー履歴のない新鮮なデバイスIDを得られます）。下記の「E2EE付きの以前のバージョンからのアップグレード」セクションを参照してください。これは最も信頼できるパスであり、ホームサーバーのデータベースに触れずに済みます。

**手動回復**（上級者向け — 同じデバイスIDを保つ）:

1. Synapseを停止し、データベースから古いデバイスを削除します。
   ```bash
   sudo systemctl stop matrix-synapse
   sudo sqlite3 /var/lib/matrix-synapse/homeserver.db "
     DELETE FROM e2e_device_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_one_time_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_fallback_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM devices WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
   "
   sudo systemctl start matrix-synapse
   ```
   または Synapse admin API 経由で（URLエンコードされたユーザーIDに注意）:
   ```bash
   curl -X DELETE -H "Authorization: Bearer ADMIN_TOKEN" \
     'https://your-server/_synapse/admin/v2/users/%40hermes%3Ayour-server/devices/DEVICE_ID'
   ```
   注意: admin API経由でデバイスを削除すると、関連するアクセストークンも無効化されることがあります。その後、新しいトークンを生成する必要があるかもしれません。

2. ローカルの暗号ストアを削除し、Hermesを再起動します。
   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db*
   # hermes を再起動
   ```

他のMatrixクライアント（Element、matrix-commander）は、古いデバイスキーをキャッシュしていることがあります。回復後、Elementで`/discardsession`と入力して、ボットとの新しい暗号化セッションを強制してください。
:::

:::info
`mautrix[encryption]`がインストールされていない、または`libolm`が見つからない場合、ボットは自動的にプレーン（非暗号化）クライアントにフォールバックします。ログに警告が表示されます。
:::

## ホームルーム

ボットがプロアクティブなメッセージ（cronジョブの出力、リマインダー、通知など）を送る「ホームルーム」を指定できます。設定する方法は2つあります。

### スラッシュコマンドを使う

ボットが存在する任意のMatrixルームで`/sethome`を入力します。そのルームがホームルームになります。

### 手動設定

`~/.hermes/.env`に次を追加します。

```bash
MATRIX_HOME_ROOM=!abc123def456:matrix.example.org
```

:::tip
ルームIDを見つけるには: Elementで、ルーム → **Settings** → **Advanced** → **Internal room ID**がそこに表示されます（`!`で始まります）。
:::

## トラブルシューティング

### ボットがメッセージに応答しない

**原因**: ボットがルームに参加していない、または`MATRIX_ALLOWED_USERS`にあなたのユーザーIDが含まれていません。

**修正**: ボットをルームに招待します — 招待で自動参加します。あなたのユーザーIDが`MATRIX_ALLOWED_USERS`にあることを検証します（完全な`@user:server`形式を使ってください）。ゲートウェイを再起動します。

### 起動時の「Failed to authenticate」 / 「whoami failed」

**原因**: アクセストークンまたはホームサーバーのURLが正しくありません。

**修正**: `MATRIX_HOMESERVER`がホームサーバーを指していることを検証します（`https://`を含め、末尾のスラッシュなし）。`MATRIX_ACCESS_TOKEN`が有効か確認します — curlで試します。

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/_matrix/client/v3/account/whoami
```

これがあなたのユーザー情報を返せば、トークンは有効です。エラーを返す場合は、新しいトークンを生成します。

### 「mautrix not installed」エラー

**原因**: `mautrix` Pythonパッケージがインストールされていません。

**修正**: インストールします。

```bash
pip install 'mautrix[encryption]'
```

または Hermes エクストラ付きで:

```bash
pip install 'hermes-agent[matrix]'
```

### 暗号化エラー / 「could not decrypt event」

**原因**: 暗号化キーの欠如、`libolm`未インストール、またはボットのデバイスが信頼されていません。

**修正**:
1. `libolm`がシステムにインストールされていることを検証します（上記のE2EEセクションを参照）。
2. `.env`に`MATRIX_ENCRYPTION=true`が設定されていることを確認します。
3. Matrixクライアント（Element）で、ボットのプロフィール -> Sessions -> ボットのデバイスを検証/信頼します。
4. ボットが暗号化されたルームに参加したばかりの場合、参加した*後*に送られたメッセージだけを復号できます。それより古いメッセージにはアクセスできません。

### E2EE付きの以前のバージョンからのアップグレード

:::tip
`crypto.db`も手動で削除した場合は、上記のE2EEセクションの「暗号ストアの削除」の警告を参照してください — ホームサーバーから古いワンタイムキーをクリアする追加の手順があります。
:::

以前に`MATRIX_ENCRYPTION=true`でHermesを使っていて、新しいSQLiteベースの暗号ストアを
使うバージョンにアップグレードする場合、ボットの暗号化
アイデンティティが変わっています。あなたのMatrixクライアント（Element）は古いデバイスキーを
キャッシュしており、ボットと暗号化セッションを共有することを拒否することがあります。

**症状**: ボットは接続し、ログに「E2EE enabled」と表示されますが、すべての
メッセージが「could not decrypt event」と表示され、ボットが決して応答しません。

**何が起きているか**: 古い暗号化状態（以前の`matrix-nio`または
シリアライズベースの`mautrix`バックエンドからのもの）は、新しいSQLite暗号
ストアと互換性がありません。ボットは新鮮な暗号化アイデンティティを作成しますが、あなたのMatrixクライアントは依然として
古いキーをキャッシュしており、キーが変わったデバイスとルームの暗号化セッションを
共有しません。これはMatrixのセキュリティ機能です -- クライアントは、
同じデバイスのアイデンティティキーが変わったことを疑わしいものとして扱います。

**修正**（一度きりの移行）:

1. **新しいアクセストークンを生成して**新鮮なデバイスIDを得ます。最も簡単な方法:

   ```bash
   curl -X POST https://your-server/_matrix/client/v3/login \
     -H "Content-Type: application/json" \
     -d '{
       "type": "m.login.password",
       "identifier": {"type": "m.id.user", "user": "@hermes:your-server.org"},
       "password": "***",
       "initial_device_display_name": "Hermes Agent"
     }'
   ```

   新しい`access_token`をコピーし、`~/.hermes/.env`の`MATRIX_ACCESS_TOKEN`を更新します。

2. **古い暗号化状態を削除します**:

   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db
   rm -f ~/.hermes/platforms/matrix/store/crypto_store.*
   ```

3. **リカバリーキーを設定します**（クロスサイニングを使う場合 — ほとんどのElementユーザーは使います）。`~/.hermes/.env`に追加します。

   ```bash
   MATRIX_RECOVERY_KEY=EsT... your recovery key here
   ```

   これにより、起動時にボットがクロスサイニングキーで自己署名できるため、Elementは新しいデバイスを即座に信頼します。これがないと、Elementは新しいデバイスを未検証と見なし、暗号化セッションの共有を拒否することがあります。Elementで**Settings** → **Security & Privacy** → **Encryption**の下にリカバリーキーがあります。

4. **Matrixクライアントに暗号化セッションをローテーションさせます**。Elementで、
   ボットとのDMルームを開いて`/discardsession`と入力します。これにより、Elementは
   新しい暗号化セッションを作成し、ボットの新しいデバイスと共有することを強制されます。

5. **ゲートウェイを再起動します**:

   ```bash
   hermes gateway run
   ```

   `MATRIX_RECOVERY_KEY`が設定されていれば、ログに`Matrix: cross-signing verified via recovery key`が表示されるはずです。

6. **新しいメッセージを送ります**。ボットは正常に復号して応答するはずです。

:::note
移行後、アップグレード*前*に送られたメッセージは復号できません -- 古い
暗号化キーはなくなっています。これは移行のみに影響します。新しいメッセージは
正常に動作します。
:::

:::tip
**新規インストールは影響を受けません。** この移行は、Hermesの以前のバージョンで
動作するE2EEセットアップがあり、アップグレードする場合にのみ必要です。

**なぜ新しいアクセストークンなのか？** 各Matrixアクセストークンは、特定のデバイス
IDに紐付けられています。同じデバイスIDを新しい暗号化キーで再利用すると、他のMatrix
クライアントがそのデバイスを不信にします（変わったアイデンティティキーを潜在的な
セキュリティ侵害と見なします）。新しいアクセストークンは、古いキー
履歴のない新しいデバイスIDを得るため、他のクライアントは即座にそれを信頼します。
:::

## プロキシモード（macOSでのE2EE）

Matrix E2EEには`libolm`が必要ですが、これはmacOS ARM64（Apple Silicon）ではコンパイルできません。`hermes-agent[matrix]`エクストラはLinux限定にゲートされています。macOSの場合、プロキシモードを使うと、E2EEをLinux VM上のDockerコンテナで実行しつつ、実際のエージェントはmacOS上でネイティブに、あなたのローカルファイル、メモリ、スキルへの完全なアクセス付きで実行できます。

### 仕組み

```
macOS (Host):
  └─ hermes gateway
       ├─ api_server adapter ← listens on 0.0.0.0:8642
       ├─ AIAgent ← single source of truth
       ├─ Sessions, memory, skills
       └─ Local file access (Obsidian, projects, etc.)

Linux VM (Docker):
  └─ hermes gateway (proxy mode)
       ├─ Matrix adapter ← E2EE decryption/encryption
       └─ HTTP forward → macOS:8642/v1/chat/completions
           (no LLM API keys, no agent, no inference)
```

Dockerコンテナは、Matrixプロトコル＋E2EEのみを処理します。メッセージが到着すると、それを復号し、テキストを標準的なHTTPリクエストでホストに転送します。ホストはエージェントを実行し、ツールを呼び出し、応答を生成し、それをストリームバックします。コンテナは応答を暗号化してMatrixに送ります。すべてのセッションは統一されています — CLI、Matrix、Telegram、その他のプラットフォームは、同じメモリと会話履歴を共有します。

### ステップ1: ホストを設定する（macOS）

Dockerコンテナからの受信リクエストをホストが受け付けるよう、APIサーバーを有効化します。

`~/.hermes/.env`に追加します。

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=your-secret-key-here
API_SERVER_HOST=0.0.0.0
```

- `API_SERVER_HOST=0.0.0.0`は、Dockerコンテナが到達できるよう、すべてのインターフェースにバインドします。
- `API_SERVER_KEY`は、非ループバックのバインドに必須です。強力なランダム文字列を選んでください。
- APIサーバーはデフォルトでポート8642で動作します（必要に応じて`API_SERVER_PORT`で変更）。

ゲートウェイを起動します。

```bash
hermes gateway
```

設定済みの他のプラットフォームと並んで、APIサーバーが起動するのが見えるはずです。VMから到達できることを検証します。

```bash
# Linux VM から
curl http://<mac-ip>:8642/health
```

### ステップ2: Dockerコンテナを設定する（Linux VM）

コンテナには、Matrixの認証情報とプロキシURLが必要です。LLM APIキーは不要です。

**`docker-compose.yml`:**

```yaml
services:
  hermes-matrix:
    build: .
    environment:
      # Matrix の認証情報
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # プロキシモード — ホストのエージェントに転送
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

**`Dockerfile`:**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libolm-dev && rm -rf /var/lib/apt/lists/*
RUN pip install 'hermes-agent[matrix]'

CMD ["hermes", "gateway"]
```

これがコンテナのすべてです。OpenRouter、Anthropic、その他の推論プロバイダーのAPIキーは不要です。

### ステップ3: 両方を起動する

1. まずホストのゲートウェイを起動します。
   ```bash
   hermes gateway
   ```

2. Dockerコンテナを起動します。
   ```bash
   docker compose up -d
   ```

3. 暗号化されたMatrixルームでメッセージを送ります。コンテナがそれを復号し、ホストに転送し、応答をストリームバックします。

### 設定リファレンス

プロキシモードは、**コンテナ側**（薄いゲートウェイ）で設定します。

| 設定 | 説明 |
|---------|-------------|
| `GATEWAY_PROXY_URL` | リモートHermes APIサーバーのURL（例: `http://192.168.1.100:8642`） |
| `GATEWAY_PROXY_KEY` | 認証用のBearerトークン（ホストの`API_SERVER_KEY`と一致する必要がある） |
| `gateway.proxy_url` | `GATEWAY_PROXY_URL`と同じだが`config.yaml`での指定 |

ホスト側には次が必要です。

| 設定 | 説明 |
|---------|-------------|
| `API_SERVER_ENABLED` | `true`に設定 |
| `API_SERVER_KEY` | Bearerトークン（コンテナと共有） |
| `API_SERVER_HOST` | ネットワークアクセスのため`0.0.0.0`に設定 |
| `API_SERVER_PORT` | ポート番号（デフォルト: `8642`） |

### あらゆるプラットフォームで動作する

プロキシモードはMatrixに限定されません。あらゆるプラットフォームアダプターがそれを使えます — 任意のゲートウェイインスタンスに`GATEWAY_PROXY_URL`を設定すると、ローカルでエージェントを実行する代わりにリモートのエージェントに転送します。これは、プラットフォームアダプターがエージェントとは異なる環境で実行される必要がある任意のデプロイ（ネットワーク分離、E2EE要件、リソース制約）に便利です。

:::tip
セッションの継続性は、`X-Hermes-Session-Id`ヘッダー経由で維持されます。ホストのAPIサーバーはこのIDでセッションを追跡するため、ローカルエージェントの場合と同じように、会話がメッセージをまたいで持続します。
:::

:::note
**制限（v1）:** リモートエージェントからのツール進捗メッセージは中継されません — ユーザーはストリームされた最終的な応答のみを見て、個々のツール呼び出しは見ません。危険なコマンドの承認プロンプトはホスト側で処理され、Matrixユーザーには中継されません。これらは将来のアップデートで対応される可能性があります。
:::

### 同期の問題 / ボットが遅れる

**原因**: 長時間動作するツールの実行が同期ループを遅らせることがある、またはホームサーバーが遅いです。

**修正**: 同期ループはエラー時に5秒ごとに自動的にリトライします。同期関連の警告がないか、Hermesのログを確認してください。ボットが一貫して遅れる場合は、ホームサーバーに十分なリソースがあることを確認してください。

### ボットがオフライン

**原因**: Hermesゲートウェイが動作していない、または接続に失敗しました。

**修正**: `hermes gateway`が動作していることを確認します。ターミナルの出力にエラーメッセージがないか確認します。よくある問題: 誤ったホームサーバーURL、期限切れのアクセストークン、到達できないホームサーバー。

### 「User not allowed」 / ボットがあなたを無視する

**原因**: あなたのユーザーIDが`MATRIX_ALLOWED_USERS`にありません。

**修正**: `~/.hermes/.env`の`MATRIX_ALLOWED_USERS`にあなたのユーザーIDを追加し、ゲートウェイを再起動します。完全な`@user:server`形式を使ってください。

## セキュリティ

:::warning
ボットと対話できる人を制限するため、必ず`MATRIX_ALLOWED_USERS`を設定してください。それがないと、ゲートウェイは安全策としてデフォルトですべてのユーザーを拒否します。信頼する人のユーザーIDだけを追加してください — 認可されたユーザーは、ツールの使用やシステムアクセスを含む、エージェントのあらゆる機能への完全なアクセス権を持ちます。
:::

Hermes Agentのデプロイを保護する方法の詳細については、[セキュリティガイド](../security.md)を参照してください。

## 補足

- **あらゆるホームサーバー**: Synapse、Conduit、Dendrite、matrix.org、その他あらゆる仕様準拠のMatrixホームサーバーで動作します。特定のホームサーバーソフトウェアは不要です。
- **連合（フェデレーション）**: 連合型のホームサーバー上にいる場合、ボットは他のサーバーのユーザーと通信できます — 彼らの完全な`@user:server` IDを`MATRIX_ALLOWED_USERS`に追加するだけです。
- **自動参加**: ボットは自動的にルームへの招待を承諾して参加します。参加後すぐに応答を始めます。
- **メディアサポート**: Hermesは画像、音声、動画、ファイル添付を送受信できます。メディアは、MatrixのコンテンツリポジトリAPIを使ってあなたのホームサーバーにアップロードされます。
- **ネイティブのボイスメッセージ（MSC3245）**: Matrixアダプターは、送信するボイスメッセージに`org.matrix.msc3245.voice`フラグを自動的にタグ付けします。つまり、TTSの応答やボイス音声は、汎用の音声ファイル添付としてではなく、ElementやMSC3245をサポートする他のクライアントで**ネイティブのボイスバブル**としてレンダリングされます。MSC3245フラグ付きの受信ボイスメッセージも正しく識別され、音声テキスト変換にルーティングされます。設定は不要です — これは自動的に動作します。
