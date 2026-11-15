---
sidebar_position: 9
title: "Matrix"
description: "Hermes AgentをMatrixボットとしてセットアップする"
---

# Matrixセットアップ

このページのPython依存関係コマンドは、
[PMで準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を使います。
依存関係を変更した後は、チェックアウトを再アクティベートしてHermesを再起動してください。

Hermes Agentは、オープンで連合型のメッセージングプロトコルであるMatrixと統合されます。Matrixでは、自分のホームサーバーを運用することも、matrix.orgのような公開ホームサーバーを使うこともできます — いずれにしても、通信のコントロールはあなたが保持します。ボットは`mautrix` Python SDK経由で接続し、メッセージをHermes Agentのパイプライン（ツールの使用、メモリ、推論を含む）で処理し、リアルタイムで応答します。テキスト、ファイル添付、画像、音声、動画、そして任意のエンドツーエンド暗号化（E2EE）をサポートします。

Hermesは、Synapse、Conduit、Dendrite、matrix.orgなど、あらゆるMatrixホームサーバーで動作します。

セットアップの前に、多くの人が知りたい部分を説明します。Hermesが接続された後、どう振る舞うかです。

## Hermesの振る舞い

| コンテキスト | 振る舞い |
|---------|----------|
| **DM** | Hermesはすべてのメッセージに応答します。`@mention`は不要です。各DMは独自のセッションを持ちます。DMでボットが`@mention`されたときにスレッドを開始するには、`MATRIX_DM_MENTION_THREADS=true`を設定します。参加メンバーが2人以下のルームは、明示的な名前があってもこの方法でDMとして扱われます — Matrixクライアントは1対1チャットに自動で名前を付けるため、名前だけでは信頼できる判断材料にならないからです。 |
| **ルーム** | デフォルトでは、Hermesは応答に`@mention`を要求します。`MATRIX_REQUIRE_MENTION=false`を設定するか、フリーレスポンスルームとして`MATRIX_FREE_RESPONSE_ROOMS`にルームIDを追加します。ルームへの招待は自動承諾されます。意図的に作成した2人のルームも引き続きDMとして分類され（上記参照）、`MATRIX_ALLOWED_ROOMS`、`MATRIX_REQUIRE_MENTION`、`MATRIX_FREE_RESPONSE_ROOMS`を暗黙的にバイパスします — 通常のルームとして振る舞わせる必要がある場合は、3人目のメンバーを追加してください。 |
| **スレッド** | HermesはMatrixのスレッド（MSC3440）をサポートします。スレッド内で返信すると、Hermesはスレッドのコンテキストをメインルームのタイムラインから分離して保ちます。ボットがすでに参加したスレッドでは、メンションは不要です。 |
| **自動スレッド化** | デフォルトでは、Hermesはルームで応答する各メッセージについてスレッドを自動作成します。これにより会話が分離されます。無効化するには`MATRIX_AUTO_THREAD=false`を設定します。DMメッセージでもスレッドを自動作成するには`MATRIX_DM_AUTO_THREAD=true`（デフォルトはfalse）を設定します — これは、DMでボットが`@mention`されたときにのみスレッドを開始する`MATRIX_DM_MENTION_THREADS`とは別物です。参加メンバーが2人以下のルームはDMとして分類され（上記参照）、`MATRIX_AUTO_THREAD`ではなく`MATRIX_DM_AUTO_THREAD`に従います。 |
| **コマンド** | Matrixクライアントが送信すれば、Hermesは通常の`/commands`を受け付けます。クライアントが`/`をローカルコマンド用に予約している場合は、代わりに`!commands`を使ってください。Hermesは既知の`!command`エイリアスを`/command`に正規化します。 |
| **インタラクティブなコントロール** | 危険なコマンドの承認と`/model`の選択には、Matrixのリアクションを使えます。承認リアクションは、そのアクションを要求したユーザーに限定できます。 |
| **思考とツールのアクティビティ** | ゲートウェイの進捗表示が有効な場合、Matrixはスレッド化された編集可能な思考/ツールアクティビティのペインを使うため、更新がメインルームのタイムラインを埋め尽くすことはありません。 |
| **複数ユーザーがいる共有ルーム** | デフォルトでは、Hermesはルーム内のユーザーごとにセッション履歴を分離します。同じルームで会話する2人は、明示的に無効化しない限り、1つのトランスクリプトを共有しません。 |
| **LaTeX数式** | 返信内の`$...$`（インライン）と`$$...$$`（ディスプレイ）は、Elementの`data-mx-maths`マークアップとして送信されるため、**Settings → Labs → Render LaTeX maths in messages**を有効にしたクライアントではKaTeXで組版されます。対になっていないドル記号（`$5 or $10`）はそのまま残り、プレーンテキストの`body`には他のクライアント向けに生のTeXが保持されます。 |

:::tip
ボットは招待されると自動的にルームに参加します。ボットのMatrixユーザーを任意のルームに招待するだけで、参加して応答を始めます。
:::

## 機能マトリクス

この表は、Matrixアダプターの機能宣言とMatrixのテストカバレッジに基づいています。
E2EEはモードベースです。デプロイごとに、暗号化ルームを無効にするか、
可能なら使うか、必須にするかを選ぶためです。

| 機能 | Matrix |
|------------|--------|
| テキスト | はい |
| スレッド | はい |
| リアクション | はい |
| 承認 | はい |
| モデルピッカー | はい |
| 思考ペイン | はい |
| 画像 | はい |
| 複数画像 | はい |
| ファイル | はい |
| ボイス/音声 | はい |
| 動画 | はい |
| E2EE | off / optional / required |
| 診断 | はい |

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
  require_mention: true           # Require @mention in rooms (default: true)
  allowed_users:                  # Matrix users allowed to trigger agent turns
    - "@alice:matrix.org"
  allowed_rooms:                  # Matrix rooms allowed to trigger agent turns
    - "!abc123:matrix.org"
  free_response_rooms:            # Rooms exempt from mention requirement
    - "!abc123:matrix.org"
  ignore_user_patterns:           # Bridge/appservice ghost users to ignore
    - "^@telegram_"
    - "^@whatsapp_"
  process_notices: false          # Ignore m.notice by default
  session_scope: room             # auto|room|thread; room is recommended for project rooms
  auto_thread: true               # Auto-create threads for responses (default: true)
  dm_mention_threads: false       # Create thread when @mentioned in DM (default: false)
  max_message_length: 15000       # Outbound chunk size in UTF-8 bytes (default and max: 15000)
```

または環境変数で:

```bash
MATRIX_REQUIRE_MENTION=true
MATRIX_ALLOWED_USERS=@alice:matrix.org
MATRIX_ALLOWED_ROOMS=!abc123:matrix.org
MATRIX_FREE_RESPONSE_ROOMS=!abc123:matrix.org,!def456:matrix.org
MATRIX_IGNORE_USER_PATTERNS='^@telegram_,^@whatsapp_'
MATRIX_PROCESS_NOTICES=false
MATRIX_SESSION_SCOPE=room       # recommended for stable project-room context
MATRIX_AUTO_THREAD=true
MATRIX_DM_MENTION_THREADS=false
MATRIX_REACTIONS=true          # default: true — emoji reactions during processing
MATRIX_ALLOW_ROOM_MENTIONS=false
```

:::tip リアクションの無効化
`MATRIX_REACTIONS=false`は、ボットが受信メッセージに付ける処理ライフサイクルの絵文字リアクション（👀/✅/❌）をオフにします。リアクションイベントがノイズになる、または参加するすべてのクライアントがサポートしていないルームに便利です。
:::

:::tip ルーム全体へのメンション
Hermesは、`@alice:example.org`のような明示的なMatrix IDに対して、構造化されたMatrixユーザーメンションを送信します。ルーム全体への`@room`通知はデフォルトで無効です。ボットが全員に通知することを許可されたルームでのみ、`MATRIX_ALLOW_ROOM_MENTIONS=true`を設定してください。
:::

:::note
`MATRIX_REQUIRE_MENTION`がなかったバージョンからアップグレードする場合、以前はボットがルーム内のすべてのメッセージに応答していました。その動作を維持するには、`MATRIX_REQUIRE_MENTION=false`を設定してください。
:::

### プロジェクトルームの分離

同じMatrixボットを複数のプロジェクトルームで使う場合は、安定した
ルームスコープのセッションを設定してください。

```bash
MATRIX_SESSION_SCOPE=room
MATRIX_AUTO_THREAD=false
```

`MATRIX_SESSION_SCOPE`には次の値を指定できます。

| スコープ | 振る舞い |
|-------|----------|
| `auto` | 後方互換のデフォルトです。既存の`MATRIX_AUTO_THREAD`の振る舞いが合成スレッドを制御します。 |
| `room` | スレッド化されていないルームメッセージは、1つの安定したルームセッションに留まります。実際のMatrixスレッドは引き続きスレッドルートを使います。 |
| `thread` | スレッド化されていないルームメッセージについて、トリガーとなったイベントIDからスレッド/セッションを合成します。 |

Hermesは現在、エージェントのプロンプトに、現在のMatrixルーム名、ルームID、トピック、メッセージID、
およびMatrixのルーム境界に関する注記を含めます。`/status`も現在の
Matrixルーム/セッションスコープを表示し、`/resume`は、明示的に
`/resume --cross-room <session name>`を使わない限り、別のMatrixルームの名前付きセッションを
暗黙的に再開しません。

`MATRIX_SESSION_SCOPE=room`はルーム/スレッドのレーンを制御します。既存の
`group_sessions_per_user`設定は、引き続きそのルーム内のユーザーがレーンを
共有するかどうかを制御します。`group_sessions_per_user: true`（デフォルト）では、AliceとBobは
別々のProject Bセッションを持ちます。`group_sessions_per_user: false`では、そのルームには
1つの共有されたProject Bトランスクリプトがあります。

このガイドでは、ボットアカウントの作成から最初のメッセージの送信まで、完全なセットアッププロセスを順を追って説明します。

## ステップ1: ボットアカウントを作成する

ボット用のMatrixユーザーアカウントが必要です。これにはいくつかの方法があります。

### オプションA: 自分のホームサーバーで登録する（推奨）

自分のホームサーバー（Synapse、Conduit、Dendrite）を運用している場合:

1. admin APIまたは登録ツールを使って、新しいユーザーを作成します。

```bash
# Synapse example
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
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# Optional: user ID (auto-detected from token if omitted)
# MATRIX_USER_ID=@hermes:matrix.example.org

# Security: restrict who can interact with the bot
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# Optional: restrict which rooms can trigger the bot
MATRIX_ALLOWED_ROOMS=!abc123:matrix.example.org

# Multiple allowed users (comma-separated)
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

**パスワードログインを使う場合:**

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_USER_ID=@hermes:matrix.example.org
MATRIX_PASSWORD=***

# Security
MATRIX_ALLOWED_USERS=@alice:matrix.example.org
```

## プライベートデプロイの堅牢化

プライベートなMatrixデプロイでは、ユーザーとルームの両方の許可リストを設定してください。
`MATRIX_ALLOWED_USERS`が未設定の場合、参加済みのルームでボットに到達できる送信者は誰でも
エージェントのターンをトリガーできます。`MATRIX_ALLOWED_ROOMS`が未設定の場合、ボットが
参加したどのルームからでもエージェントのターンをトリガーできます。厳重に制限したデプロイでは両方を設定すべきです。

```bash
MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
MATRIX_ALLOWED_ROOMS=!ops:matrix.example.org,!dmroom:matrix.example.org
```

ブリッジやappserviceのデプロイには、追加のループ保護が必要です。Hermesは常に、
自身のイベント、ローカルパートが`_`で始まるMatrix appservice形式のユーザー、
重複したイベントID、起動前の古いイベント、編集の置換イベント、そしてデフォルトでは
`m.notice`イベントを無視します。ブリッジが異なる命名規則を使う場合は、
デプロイ固有のブリッジゴーストのパターンを追加してください。

```bash
MATRIX_IGNORE_USER_PATTERNS='^@telegram_,^@slack_,^@whatsapp_'
```

信頼できる人間のワークフローが実際に`m.notice`を送る場合にのみ、noticeを有効にしてください。

```bash
MATRIX_PROCESS_NOTICES=true
```

処理されたnoticeはボットメッセージとして扱われます。ゲートウェイのボットループガードが適用され、
ペアリングの返信は決して送られません。`m.notice`はどのMatrixユーザーでも送れるため、それ自体は
アクセス権を与えません。noticeの送信者にも`MATRIX_ALLOWED_USERS`（または別の人間向けの段階）が必要です。
ループガードはやり取りも終わらせます。デフォルトでは、Hermesがあるnotice送信者に応答した後、
誰かがルームで通常のメッセージを送るか、ルームでnoticeのない状態が10分続くまで、その送信者の
次のnoticeには返信しません。やり取りの往復が必要なワークフローでは、`config.yaml`で
`gateway.bot_loop_guard.bot_replies_expect_reply: true`を設定してください（それでも`max_hops`、デフォルト4が上限です）。

送信側のルーム全体への通知はデフォルトで無効です。ボットが`@room`でルーム全体を
起こすことを明示的に許可されていない限り、`MATRIX_ALLOW_ROOM_MENTIONS=false`のままにしてください。

診断およびデバッグのペイロードでは、Matrixのアクセストークン、リカバリーキー、
デバイス識別子、メッセージ本文がマスクされます。メディアのダウンロードはMatrixの
`mxc://`コンテンツURIに限定され、`MATRIX_MAX_MEDIA_BYTES`を超えると拒否されます。
連合ルームや信頼できないホームサーバーは信頼できない入力として扱ってください。ルームの
許可リストは厳しく保ち、ツールを多用する作業にはDMやプライベートルームを優先し、
ブリッジゴーストやappserviceのパペットを許可ユーザーとして認可するのは避けてください。

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
# Request the declared Matrix dependencies
python -c "import pm; pm.sync_venv(['matrix'], explicit=True)"
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
MATRIX_E2EE_MODE=required
```

`MATRIX_E2EE_MODE`には次の値を指定できます。

| モード | 振る舞い |
|------|----------|
| `off` | Matrix E2EEを初期化しません。 |
| `optional` | 依存関係が利用可能ならE2EEを試みますが、暗号処理を初期化できない場合でも非暗号化ルームは動作し続けます。 |
| `required` | E2EEの依存関係または暗号のセットアップが利用できない場合、フェイルクローズします。 |

optionalモードでは、暗号のセットアップが利用できない場合に非E2EEの動作にフォールバックすることがあります。requiredモードでは、暗黙的にダウングレードせずにフェイルクローズします。

後方互換性のため、`MATRIX_ENCRYPTION=true`でも引き続きrequiredのE2EE動作が有効になります。

E2EEが有効になると、Hermesは:

- 暗号化キーを`~/.hermes/platforms/matrix/store/`に保存します（レガシーインストール: `~/.hermes/matrix/store/`）
- 初回接続時にデバイスキーをアップロードします
- 受信メッセージを復号し、送信メッセージを自動的に暗号化します
- 招待されると暗号化されたルームに自動参加します

### Matrixのツールとコントロール

Hermesは、Matrix固有のエージェントツール（ルームの作成、招待、redactionなど）を公開しません — エージェントは通常のメッセージ配信を通じてMatrixとやり取りします。アダプターは、承認プロンプトやピッカーを実現するために、内部でリアクションとredactionを使います。

`MATRIX_ALLOWED_ROOMS`が設定されている場合、Hermesはそれらのルームでのみ応答します（DMは除外されます）。

リアクションによるコントロール:

- ✅ 1回だけ承認
- ♾️ 常に承認
- ❌ 拒否
- `/model`の選択肢には数字のリアクション

ルーム内の認可された任意のMatrixユーザーに承認/モデルピッカーのプロンプトを操作させたい場合は、意図的に`MATRIX_APPROVAL_REQUIRE_SENDER=false`を設定してください。Hermesがアクションの要求者を把握している場合、デフォルトでは要求者に限定されます。

### メディアの制限

Hermesは、MatrixのメディアAPIを通じてMatrixの画像、ファイル、音声、動画をアップロード/ダウンロードします。生成された複数の画像は、順序付けされた1つの論理バッチとして送信され、バッチ全体でキャプションとスレッドのコンテキストが保持されます。

デフォルトでは、100 MBを超えるMatrixメディアはアップロード/ダウンロード前に拒否されます。次で上書きできます。

```bash
MATRIX_MAX_MEDIA_BYTES=104857600
```

受信メディアはMatrixの`mxc://`コンテンツURIを使う必要があります。Hermesは、連合ルームが
無制限のダウンローダーになるのを避けるため、Matrixイベント内の任意のHTTP(S)メディアURLを
拒否します。

### クロスサイニング検証（推奨）

Matrixアカウントでクロスサイニングが有効になっている場合（Elementのデフォルト）、起動時にボットが自身のデバイスを自己署名できるよう、リカバリーキーを設定してください。これがないと、デバイスキーのローテーション後に、他のMatrixクライアントがボットと暗号化セッションを共有することを拒否することがあります。

```bash
MATRIX_RECOVERY_KEY=EsT... your recovery key here
```

**見つける場所:** Elementで、**Settings** → **Security & Privacy** → **Encryption** → あなたのリカバリーキー（「セキュリティキー」とも呼ばれる）に移動します。これは、クロスサイニングを最初にセットアップしたときに保存するよう求められたキーです。

各起動時、`MATRIX_RECOVERY_KEY`が設定されていれば、Hermesはホームサーバーのセキュアなシークレットストレージからクロスサイニングキーをインポートし、現在のデバイスに署名します。これは冪等であり、永続的に有効にしておいて安全です。

Hermesが新しいMatrixリカバリーキーをブートストラップする場合、生のキーは決してログに出力しません。
起動前に`MATRIX_RECOVERY_KEY_OUTPUT_FILE=/secure/path/matrix-recovery-key.txt`を設定すると、
生成されたキーがファイルモード`0600`で一度だけ書き出されます。ファイルがすでに存在する場合は
上書きされません。

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
   # restart hermes
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
Matrixクライアントがスラッシュコマンドを横取りする場合は、代わりに`!sethome`と入力してください。

### 手動設定

`~/.hermes/.env`に次を追加します。

```bash
MATRIX_HOME_ROOM=!abc123def456:matrix.example.org
```

## ルームの許可リスト（`allowed_rooms`）

ボットを固定のMatrixルームの集合に制限します。設定すると、ボットはIDがリストに含まれるルームで**のみ**応答します — それ以外のルームからのメッセージは、ボットがメンションされていても暗黙的に無視されます。

**DM（ダイレクトチャットルーム）はこのフィルターから除外される**ため、認可されたユーザーはいつでも1対1でボットにアクセスできます。

```yaml
matrix:
  allowed_rooms:
    - "!abc123def456:matrix.example.org"
    - "!opsroom789:matrix.example.org"
```

または環境変数で（カンマ区切り）:

```bash
MATRIX_ALLOWED_ROOMS="!abc123def456:matrix.example.org,!opsroom789:matrix.example.org"
```

振る舞い:

- 空 / 未設定 → 制限なし（デフォルト）。
- 空でない → ルームIDがリストに含まれている必要があります。このチェックは、他のどのゲーティング（メンション要求、送信者の許可リストなど）よりも**前に**実行されます。
- エイリアス（`#room:server`）ではなく、ルームの**内部ID**（`!abc...:server`）を使ってください。ルームの内部IDは、ElementのRoom → Settings → Advancedで確認できます。

関連項目: [admin/userのスラッシュコマンドの分割](../../reference/slash-commands.md#permissions-and-adminuser-split)。


:::tip
ルームIDを見つけるには: Elementで、ルーム → **Settings** → **Advanced** → **Internal room ID**がそこに表示されます（`!`で始まります）。
:::

## Matrixでのコマンド

Hermesは、他のメッセージングプラットフォームでサポートしているのと同じゲートウェイコマンドを
Matrixでもサポートします。これには`/commands`、`/model`、`/stop`、`/queue`、
`/steer`、`/goal`、`/subgoal`、`/bg`、`/btw`、`/tasks`、
`/yolo`が含まれます。

一部のMatrixクライアントは、先頭の`/`をローカルのクライアントコマンド用に予約しており、
未知のスラッシュコマンドをルームに送信しないことがあります。その場合は、Matrixで安全な
エイリアスとして`!`を使ってください。

```text
!commands
!model
!model gpt-5.5 --provider openrouter
!queue continue with the next task
!stop
```

Hermesが`!command`を正規化するのは、そのコマンドがゲートウェイの既知のコマンド、
登録済みのプラグインコマンド、またはインストール済みのスキルコマンドである場合のみです。
`!important`のような通常の感嘆表現は、普通のチャットメッセージのままです。

## トラブルシューティング

### ボットがメッセージに応答しない

**原因**: ボットがルームに参加していない、`MATRIX_ALLOWED_USERS`にあなたのユーザーIDが含まれていない、`MATRIX_ALLOWED_ROOMS`にそのルームが含まれていない、またはルームのメッセージがボットをメンションしていません。

**修正**: ボットをルームに招待します — 招待で自動参加します。あなたのユーザーIDが`MATRIX_ALLOWED_USERS`にあること（完全な`@user:server`形式を使ってください）、そしてルームの許可リストを設定している場合はルームIDが`MATRIX_ALLOWED_ROOMS`にあることを検証します。ルームでは、ボットをメンションするか、ルームを`MATRIX_FREE_RESPONSE_ROOMS`に追加します。ゲートウェイを再起動します。

### ボットはルームに参加するが、すべてのメッセージを暗黙的に破棄する（クロックのずれ）

**原因**: ホストのシステムクロックが実際の時刻より進んでいます。Matrixアダプターは、初期同期で再生されるイベントを無視するため、5秒の起動猶予フィルター（`event_ts < startup_ts - 5`）を適用します。壁時計が進んでいると、すべての受信イベントが「起動より古い」ように見え、メッセージハンドラーに届く前に破棄されます — ボットは接続しているように見えますが、決して返信しません。[#12614](https://github.com/NousResearch/hermes-agent/issues/12614)を参照してください。

**症状**: ゲートウェイのログに`Matrix: dropped N live events as 'too old' more than 30s after startup`と表示されます。

**修正**: ホストのクロックをNTPで同期し、ボットを再起動します。

```bash
# Debian/Ubuntu
sudo timedatectl set-ntp true
timedatectl status   # confirm "System clock synchronized: yes"

# macOS
sudo sntp -sS time.apple.com
```

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
python -c "import pm; pm.sync_venv(['matrix'], explicit=True)"
```

または Hermes エクストラ付きで:

```bash
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['matrix'], explicit=True)"
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

## プロキシモード（macOSでのE2EE） {#proxy-mode-e2ee-on-macos}

`matrix`エクストラはLinux限定にゲートされています。macOSやWindowsでは、Matrix
アダプターと暗号化の依存関係をLinuxコンテナで実行し、リクエストをネイティブの
エージェントに転送します。以下の例はmacOSホストを使っていますが、同じ分離構成は、
対応するホストアドレスと認証を使えばWindowsにも当てはまります。

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
# From the Linux VM
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
      # Matrix credentials
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # Proxy mode — forward to host agent
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

リポジトリの[Dockerビルド](../docker.md)を使ってください。これには、互換性のあるLinuxターゲット向けの
Matrixエクストラと、必要なネイティブライブラリが含まれています。
実行時に、封印されたイメージへ依存関係をインストールしないでください。コンテナに
必要なのはMatrixの認証情報とプロキシへのアクセスであり、推論プロバイダーのAPIキーは不要です。

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

### ボットは接続して送信もできるが、受信メッセージを無視する

**原因**: Matrixのイベントハンドラーは、同期ペイロードがmautrixの`handle_sync()`の仕組みを通じて
ディスパッチされた場合にのみ発火します。`handle_sync()`を一度も呼ばない生の`client.sync()`ポーリングでは、
アダプターは接続されたまま（送信は動作する）でも、受信メッセージが
`_on_room_message`に届かないことがあります。

**修正**: Hermesは、初期同期とすべての増分同期レスポンスの両方で`client.handle_sync()`を呼ぶ
明示的な同期ループを使います。これはupstreamのissue #7914とクローズされたPR #37807での
診断に一致しますが、ライフサイクル全体を`client.start()`に委ねるのではなく、Hermes独自の
バックグラウンドのメンテナンスタスク（参加ルームの追跡、招待の処理、E2EEキーの
共有）を維持します。ゲートウェイの再起動後も受信メッセージが失敗する場合は、最初の同期より前に
ハンドラーが登録されていることを検証し、ログに`sync event dispatch error`がないか確認してください。

### 同期の問題 / ボットが遅れる

**原因**: 長時間動作するツールの実行が同期ループを遅らせることがある、またはホームサーバーが遅いです。

**修正**: 同期ループはエラー時に5秒ごとに自動的にリトライします。同期関連の警告がないか、Hermesのログを確認してください。ボットが一貫して遅れる場合は、ホームサーバーに十分なリソースがあることを確認してください。

### ボットがオフライン

**原因**: Hermesゲートウェイが動作していない、または接続に失敗しました。

**修正**: `hermes gateway`が動作していることを確認します。ターミナルの出力にエラーメッセージがないか確認します。よくある問題: 誤ったホームサーバーURL、期限切れのアクセストークン、到達できないホームサーバー。

### 「User not allowed」 / ボットがあなたを無視する

**原因**: あなたのユーザーIDが`MATRIX_ALLOWED_USERS`にありません。

**修正**: `~/.hermes/.env`の`MATRIX_ALLOWED_USERS`にあなたのユーザーIDを追加し、ゲートウェイを再起動します。完全な`@user:server`形式を使ってください。

### ボットがルーム全体を無視する

**原因**: `MATRIX_ALLOWED_ROOMS`が設定されていて現在のルームIDが含まれていない、またはルームがメンションを要求していてメッセージがボットをメンションしていません。

**修正**: ルームIDを`MATRIX_ALLOWED_ROOMS`に追加するか、個人用のデプロイであればルームの許可リストを削除します。ElementでルームIDを見つけるには、ルームの設定を開いて**Advanced**を確認してください。

### ブリッジのメッセージがループまたはエコーする

**原因**: ブリッジ/appserviceのパペットがボットの出力を新しいユーザーメッセージとして中継し返している、またはブリッジが非標準のゴーストユーザーIDを使っています。

**修正**: ブリッジゴーストを`MATRIX_ALLOWED_USERS`に含めず、一致する`MATRIX_IGNORE_USER_PATTERNS`のエントリーを追加し、noticeが信頼できるワークフローの一部でない限り`MATRIX_PROCESS_NOTICES=false`のままにしてください。

## セキュリティ

:::warning
必ず`MATRIX_ALLOWED_USERS`を設定し、共有/プライベートのデプロイでは`MATRIX_ALLOWED_ROOMS`も設定してください。それらがないと、参加済みのルームでボットにメッセージを送れる人は誰でもエージェントをトリガーできる可能性があります。信頼する人とルームだけを認可してください — 認可されたユーザーは、ツールの使用やシステムアクセスを含む、エージェントのあらゆる機能への完全なアクセス権を持ちます。
:::

Hermes Agentのデプロイを保護する方法の詳細については、[セキュリティガイド](../security.md)を参照してください。

## 補足

- **あらゆるホームサーバー**: Synapse、Conduit、Dendrite、matrix.org、その他あらゆる仕様準拠のMatrixホームサーバーで動作します。特定のホームサーバーソフトウェアは不要です。
- **連合（フェデレーション）**: 連合型のホームサーバー上にいる場合、ボットは他のサーバーのユーザーと通信できます — 彼らの完全な`@user:server` IDを`MATRIX_ALLOWED_USERS`に追加するだけです。
- **自動参加**: ボットは自動的にルームへの招待を承諾して参加します。参加後すぐに応答を始めます。
- **メディアサポート**: Hermesは画像、音声、動画、ファイル添付を送受信できます。メディアは、MatrixのコンテンツリポジトリAPIを使ってあなたのホームサーバーにアップロードされます。
- **ネイティブのボイスメッセージ（MSC3245）**: Matrixアダプターは、送信するボイスメッセージに`org.matrix.msc3245.voice`フラグを自動的にタグ付けします。つまり、TTSの応答やボイス音声は、汎用の音声ファイル添付としてではなく、ElementやMSC3245をサポートする他のクライアントで**ネイティブのボイスバブル**としてレンダリングされます。MSC3245フラグ付きの受信ボイスメッセージも正しく識別され、音声テキスト変換にルーティングされます。設定は不要です — これは自動的に動作します。`MEDIA:`タグで添付された音声ファイル（エージェントが生成した`.mp3`/`.wav`で、`[[audio_as_voice]]`が付いていないもの）は、ボイスバブルではなく、元の形式のまま通常の`m.audio`添付として送信されます。
