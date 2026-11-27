---
sidebar_position: 15
title: "Hermes Web ダッシュボード"
description: "設定、API キー、MCP サーバー、メッセージングのペアリング、webhook、ゲートウェイ、メモリ、認証情報、セッション、ログ、分析、cron ジョブ、スキルを管理するためのブラウザベースの管理パネル"
---

# Hermes Web ダッシュボード

Web ダッシュボードは、Hermes Agent のインストールを管理するためのブラウザベースの UI です。YAML ファイルを編集したり CLI コマンドを実行したりする代わりに、整理された Web インターフェースから設定の構成、API キーの管理、セッションの監視ができます。

:::tip
ホステッドモードの認証には Nous Portal OAuth を使用します。ダッシュボードを実際のバックエンドにも接続したい場合は、`hermes setup --portal` でモデルとツールゲートウェイもあわせて設定できます。[Nous Portal](/docs/integrations/nous-portal) を参照してください。
:::

## クイックスタート

```bash
hermes dashboard
```

これはローカルの Web サーバーを起動し、ブラウザで `http://127.0.0.1:9119` を開きます。ダッシュボードは完全にお使いのマシン上で動作し、データが localhost の外に出ることはありません。

### オプション

| フラグ | デフォルト | 説明 |
|------|---------|-------------|
| `--port` | `9119` | Web サーバーを実行するポート |
| `--host` | `127.0.0.1` | バインドアドレス |
| `--no-open` | — | ブラウザを自動で開かない |
| `--insecure` | off | **非推奨 / 何もしません。** 以前は loopback 以外へのバインド時に認証をバイパスしていましたが、現在は認証を無効にしません — パブリックなバインドには常に認証プロバイダー（パスワードまたは OAuth）が必要です |
| `--isolated` | off | 名前付きプロファイルから起動した場合（`worker dashboard`）、マシンのダッシュボードにルーティングする代わりに、プロファイル専用のサーバーを実行します。そのプロファイルのみが一覧表示されます。これはセキュリティ境界ではありません（[複数プロファイルの管理](#managing-multiple-profiles)を参照） |

```bash
# Custom port
hermes dashboard --port 8080

# Bind to all interfaces (use with caution on shared networks)
hermes dashboard --host 0.0.0.0

# Start without opening browser
hermes dashboard --no-open
```

## 複数プロファイルの管理 {#managing-multiple-profiles}

ダッシュボードは**マシンレベル**の管理画面です。1 つのサーバーがマシン上のすべての[プロファイル](../profiles.md)を管理します。サイドバーのプロファイルスイッチャー（プロファイルが 2 つ以上ある場合に表示）で、管理ページがどのプロファイルを読み書きするかを決めます — Config、API Keys、Skills、MCP、Models、Chat タブはすべてこれに従います。ダッシュボード自身のもの以外のプロファイルが選択されている間は、管理対象のプロファイル名を示すアンバー色のバナーが表示されるので、書き込み先が曖昧になることはありません。

選択は URL（`?profile=<name>`）に保持されるため、`http://127.0.0.1:9119/skills?profile=worker` のようなディープリンクはスイッチャーが事前選択された状態で開き、リロードしても維持されます。

プロファイルのエイリアスからダッシュボードを起動すると、2 つ目のサーバーを起動する代わりにマシンのダッシュボードにルーティングされます。

```bash
worker dashboard
# → already running: opens the browser at ?profile=worker
# → not running:     starts the machine dashboard with "worker" preselected
```

この方法で起動したダッシュボードは、サーバーの稼働中ずっと `worker` をフォールバックのスコープとして保持します。`?profile=` を省略したディープリンク（例: `/chat?resume=<id>` リンク）も `worker` で開くため、埋め込みチャットはそのプロファイルの MCP サーバー、モデル、スキルを参照します。URL に明示的な `?profile=` がある場合は常にそちらが優先されます。

`--isolated` を渡すとこれをオプトアウトし、そのプロファイル専用のサーバーを実行します（統合前の動作）。そのプロファイル一覧、`/api/status` のプロファイル名、統合セッションサイドバー、Desktop のプロファイルレール（したがってその SOUL エディタ）には、起動したプロファイルのみが表示されます。

:::warning `--isolated` はセキュリティ境界ではありません
`--isolated` はルーティングと運用のための便宜的な機能であり、テナント分離ではありません。1 つの OS ユーザー配下のすべてのプロファイルは、他のすべてのプロファイルのファイルを読み取れます。専用サーバー上でも、プロファイルセレクターを受け取るルート、`/api/fs/*`、エージェント自身のターミナルおよびファイルツールは、兄弟プロファイル（`~/.hermes/profiles/*`）に到達できます。プロファイルのサーバーごとに異なる認証を使っても、これは変わりません。人々のデータ（家族、顧客、チーム）を分離するには、テナントごとの Hermes を**別々の OS ユーザー**または**別々のコンテナ**で実行してください。[SECURITY.md](https://github.com/NousResearch/hermes-agent/blob/main/SECURITY.md) §2.2 を参照してください。
:::

**Chat** タブもスイッチャーに従います。スコープされたチャットは、選択されたプロファイルの `HERMES_HOME` で PTY 子プロセスを起動するため、会話はそのプロファイルのモデル、スキル、メモリ、セッション履歴で実行されます。プロファイルを切り替えると、新しいターミナルセッションが開始されます。

Hub アクション（スキルのインストール／更新／アンインストール、MCP のインストール、ツールセットのセットアップ）は、ダッシュボードプロセスの環境ではなく、対象プロファイル自身のシークレットスコープ — その `.env` と設定されたシークレットソース — で実行されます。これには、マシンのダッシュボードから `default` プロファイルを対象とするアクションも含まれます。

プロファイルごとに保たれ、スイッチャーに*吸収されない*もの: ゲートウェイプロセス（`hermes -p <name> gateway …` で管理します）、各プロファイルのセッションデータベース、cron スケジューラ（Cron ページは独自のフィルターでプロファイル横断の集約をすでに行っています）。

## 前提条件

FastAPI、Uvicorn、プラットフォームの PTY ヘルパーは Hermes のコア依存関係です。`web` extra は HTTP スタックに対する厳密な制約を追加します。`pty` extra は、その依存関係がすでにコアに含まれているため空です。標準の PM セットアップでは、`all` を通じて `web` が含まれます。

これらの依存関係が壊れている場合は、`hermes pm repair` を実行して Hermes を再起動してください。ソースからのセットアップについては、[PM 開発者ワークフロー](/docs/reference/package-management#developer-workflow)を使用してください。メッセージングと音声の extra は別途指定するもので、`all` には含まれません。

依存関係なしで `hermes dashboard` を実行すると、何をインストールすればよいか教えてくれます。フロントエンドがまだビルドされていない場合は、Hermes がプロビジョニングする Node と pnpm を使って、初回起動時に自動的にビルドされます。

Chat タブはすべての `hermes dashboard` 起動に含まれます — 埋め込みのブラウザチャットペイン（PTY/WebSocket 上で TUI を実行）は常に利用可能で、追加のフラグは不要です。

## ページ

### Status

ランディングページには、インストールのライブな概要が表示されます。

- **エージェントバージョン**とリリース日
- **ゲートウェイの状態** — 実行中／停止中、PID、接続されているプラットフォームとその状態
- **アクティブセッション** — 直近 5 分間にアクティブだったセッション数
- **最近のセッション** — 直近 20 件のセッションの一覧（モデル、メッセージ数、トークン使用量、会話のプレビュー付き）

Status ページは 5 秒ごとに自動更新されます。

#### リソース逼迫バナー

ホストのメモリまたはディスクが不足してくると、ダッシュボードの上部にバナーが表示されます（同じステータスポーリングから供給されるため、追加のリクエストはありません）。

- **"Your agent is almost out of memory and may restart"** — ゲートウェイの 30 秒ごとのハートビートでサンプリングされたシステムの利用可能メモリが、*elevated*（128 MiB 未満または 15% 未満）または *critical*（64 MiB 未満または 5% 未満）のレベルまで低下しています。
- **"Your agent restarted unexpectedly, most likely because it ran out of memory"** — ライフサイクル台帳に、前回のブートでメモリ逼迫下での不正な終了が記録されています（OOM kill の疑い）。
- **ディスク警告** — `~/.hermes` を格納するボリュームがほぼ満杯です（空き 512 MB 未満で *elevated*、空き 256 MB 未満で *critical*）。

一度に表示されるのは、アクティブな警告のうち最も深刻なもの 1 つだけです（disk critical > memory critical > OOM restart > disk elevated > memory elevated）。非表示にした状態は現在のゲートウェイのブートにスコープされます。警告を非表示にすると次にアクティブな警告が表示され、ゲートウェイの再起動またはエスカレーション（elevated → critical）で再び表示されます。古いハートビートの場合は、誤ったアラートではなく何も表示しません。

### Chat

**Chat** タブは、完全な Hermes TUI（`hermes --tui` で得られるのと同じインターフェース）をブラウザ内に直接埋め込みます。スラッシュコマンド、モデルピッカー、ツール呼び出しカード、markdown ストリーミング、clarify/sudo/approval のプロンプト、スキンのテーマ設定など、ターミナルの TUI でできることはすべてここでも同じように動作します。ダッシュボードが実際の TUI バイナリを実行し、その ANSI 出力を [xterm.js](https://xtermjs.org/) の WebGL レンダラーでピクセルパーフェクトなセルレイアウトとしてレンダリングしているからです。

**仕組み:**

- `/api/pty` は、ダッシュボードのセッショントークンで認証された WebSocket を開きます
- サーバーは POSIX 疑似ターミナルの背後で `hermes --tui` を起動します
- キー入力は PTY に送られ、ANSI 出力がブラウザにストリーミングされます
- xterm.js の WebGL レンダラーが各セルを整数ピクセルグリッドに描画します。マウストラッキング（SGR 1006）、全角文字（Unicode 11）、罫線素片グリフはすべてネイティブにレンダリングされます
- ブラウザウィンドウのリサイズは、`@xterm/addon-fit` アドオンを介して TUI をリサイズします
- 静かな PTY ソケットは 20 秒ごとに小さなリサイズ keepalive を送信するため、アイドルタイムアウトを持つリバースプロキシ（nginx、Caddy）が単に待機中のチャットを切断することはありません。接続の合計存続時間に上限を設けるプロキシ（一部のトンネルはトラフィックに関係なく約 30 秒後にソケットを閉じます）は依然として接続を閉じますが、チャットは同じセッションに自動的に再接続します。ブラウザのタブが非表示のときや別のダッシュボードページにいるときは自動再接続が一時停止し、戻ってくると再開します

**既存のセッションを再開する:** **Sessions** タブから、任意のセッションの横にある再生アイコン（▶）をクリックします。これは `/chat?resume=<id>` にジャンプし、`--resume` で TUI を起動して完全な履歴を読み込みます。

**セッションスイッチャー（右レール）:** Chat タブには、ターミナルの横の細い右レールに ChatGPT 風の独自の会話リストがあり、ページを離れずに会話を切り替えられます。レールの上部にモデルピッカー、そのすぐ下にセッションリストが積み重なり、ターミナルが画面の大部分を占めます。リストには、アクティブなプロファイルの最近のセッションが表示されます — タイトル（ない場合はメッセージのプレビュー）、相対的な最終アクティブ時刻、メッセージ数、CLI 以外のセッションではソースチャネル。任意の行をクリックすると、その場で再開します（ターミナルがその会話の履歴で再起動します）。アクティブなセッションはハイライトされます。**New chat** は新しいセッションを開始し、更新コントロールはリストを再取得します。レールは切り替え専用で読み取り専用です — 削除、名前変更、エクスポート、一括クリーンアップは引き続き **Sessions** タブにあります。狭い画面では、スライドオーバーパネルに折りたたまれます。

**ワークスペースピッカー:** 新しいチャットは、ダッシュボードプロセスが起動された場所で開始されます — スマートフォンから Hermes を操作していて `~/code/foo` で動かしたい場合には役に立ちません。レールの **workspace** セレクターには、Desktop サイドバーが認識しているのと同じディレクトリが一覧表示されます。明示的なプロジェクト（`hermes projects`）と、発見されたすべての git リポジトリ（セッション由来のものと `desktop.repo_scan_roots` のスキャンによるもの）が、最近アクティブな順に並び、それ以外のものには **Other path…** エントリがあります。選択はプロファイルごとに記憶され、次の **New chat** に適用されます（再開したセッションは自身の作業ディレクトリを保持します）。再スキャンボタンはホスト上の検出ルートを再走査するため、SSH 経由でクローンしたばかりのリポジトリも再起動なしで表示されます。存在しなくなったパスは、起動ディレクトリで黙って開始するのではなく、エラーで拒否されます。これは `GET /api/chat/workspaces` と `/api/pty` WebSocket の `cwd` パラメータによって実現されています。

**前提条件:**

- Node.js（`hermes --tui` と同じ要件。TUI バンドルは初回起動時にビルドされます）
- `ptyprocess` — POSIX ではコア依存関係です
- POSIX カーネル（Linux、macOS、または WSL2）。`/chat` のターミナルペインは特に POSIX PTY を必要とします。ネイティブ Windows の Python には同等のものがないため、ネイティブ Windows インストールでは、ダッシュボードの残り（セッション、ジョブ、メトリクス、設定エディタ）は動作しますが、`/chat` タブにはその機能のために WSL2 を使うよう促すバナーが表示されます。

ブラウザのタブを閉じると、PTY はサーバー上でクリーンに回収されます。再度開くと新しいセッションが起動します。

[Hermes Desktop](#connecting-hermes-desktop-to-a-remote-backend) を、同梱のバックエンドではなく別のマシンで動作しているダッシュボードに向けるには、以下のリモートバックエンドのセクションを参照してください。

### Hermes Desktop をリモートバックエンドに接続する {#connecting-hermes-desktop-to-a-remote-backend}

Hermes Desktop は通常、自身のローカルバックエンドを起動しますが、**Settings → Gateways → Remote gateway** から、リモートマシン（VM、ホームラボのマシンなど）で動作しているダッシュボードに接続することもできます。これは「Desktop はバックエンドの準備ができたと言うのに、チャットがまったく動かない」という報告の最も一般的な原因です。Desktop の準備完了チェックが検証する内容は、ライブチャット接続が実際に必要とするものよりも少ないためです。

:::info 前提条件: リモートホストで `hermes dashboard` が実行されている必要があります
Desktop が接続する「リモートバックエンド」**とは**、リモートマシンで実行されている `hermes dashboard` プロセス — このページで説明しているのと同じサーバー — のことです。以下の手順に意味を持たせるには、まずそれが起動していて到達可能である必要があります。Desktop はそれに接続するだけで、起動はしてくれません。ログアウトや再起動後も存続するよう、`systemd`/`tmux` などの下で実行し続けてください。**ゲートウェイ**（Telegram/Discord/Slack など）は*別の*長時間実行プロセスです — メッセージングチャネルに依存する場合は独立して起動してください。デスクトップアプリが接続する対象はゲートウェイではありません。
:::

Desktop の「リモートバックエンドの準備完了」プローブは `GET /api/status` にアクセスするだけですが、これはパブリックなエンドポイントで、ホスト上で*何らかの*ダッシュボードが動いていればすぐに応答します。ライブチャット接続は `/api/ws`（および `/api/pty`）への**別の** WebSocket であり、そのソケットはステータスプローブが一切触れない 2 つの追加チェックで保護されています。

1. **認証されている必要があります。** ダッシュボードが loopback 以外のアドレスにバインドされていると、認証ゲートが有効になります。ユーザー名とパスワード（同梱の[ユーザー名／パスワードプロバイダー](#usernamepassword-provider-no-oauth-idp)）で保護してください。Desktop は一度サインインし、得られたセッションを使い捨てのチケット経由で WebSocket に再利用します。プロバイダーが設定されていない場合、loopback 以外のダッシュボードは**起動時にフェイルクローズ**します。
2. **バインドホストがクライアントを許可し、Host ヘッダーと一致している必要があります。** loopback バインド（`127.0.0.1`）は loopback クライアントのみを受け付けるため、リモートマシンは認証情報に関係なくソケット層で拒否されます。ピア IP ガードがリモートクライアントを通すよう、loopback 以外のアドレス（`--host 0.0.0.0`）にバインドしてください。Desktop に入力するリモート URL は、ダッシュボードがバインドしたのと同じホストでダッシュボードに到達する必要があります — DNS リバインディングガードが Host ヘッダーの一致を要求します。

#### リモートダッシュボードのセットアップ

ユーザー名とパスワードを設定し、到達可能なアドレスにバインドしてダッシュボードを実行します。`systemd` サービスの場合:

```ini
[Service]
EnvironmentFile=%h/.hermes/.env
ExecStart=/path/to/venv/bin/python -m hermes_cli.main dashboard \
    --host 0.0.0.0 --port 9119 --no-open
Restart=always
RestartSec=10
# Exit 78 (EX_CONFIG) is a deliberate refusal ("this host is already served by PID … on
# another port"); parking on it beats an infinite restart loop with nothing listening.
RestartPreventExitStatus=78
```

1 つのバックエンドがホスト全体を担当するため、`hermes dashboard` が、指定された `--host`/`--port` で提供できないライブなバックエンドを見つけた場合、終了コード 78 で拒否し、その所有者を示します。そのバックエンドを停止するか、`--isolated` でサービス専用のサーバーを与えてください。Desktop アプリ自身の loopback バックエンドはホストの所有権を主張しないため、どちらのフラグもなしに両者は共存できます。

`~/.hermes/.env` には次の内容を含めます。

```bash
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
HERMES_DASHBOARD_BASIC_AUTH_SECRET=<32+ random bytes; openssl rand -base64 32>
```

次に Desktop で **Remote URL**（例: `http://VM_IP:9119`）を入力し、そのユーザー名とパスワードで **Sign in** します。設定項目の全体については、[ユーザー名／パスワードプロバイダー](#usernamepassword-provider-no-oauth-idp)のセクションを参照してください。

:::tip Desktop で再試行する前にゲートが有効か確認する
任意のマシンから、ダッシュボードがユーザー名／パスワードプロバイダーを公開していることを確認します。

```bash
curl -s http://VM_IP:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["basic"]
```

- `auth_required: true` かつプロバイダー一覧に `"basic"` がある → Desktop の **Sign in** フローは動作します。
- `auth_required: false` → バインドが loopback であるか、ゲートが有効になっていません。loopback 以外のアドレスにバインドしてください。
- `auth_required: true` だが `"basic"` プロバイダーがない → ユーザー名／パスワードの環境変数が読み込まれていません。まずそれを修正してください。
:::

`/api/status` で `"basic"` プロバイダー付きでゲートが有効と表示されているのに、サインイン後も Desktop が接続に*まだ*失敗する場合、問題は基本的なセットアップの先にあります。新しい `desktop.log`（Settings → Gateways → Open logs）と、同じ再試行の時間帯のダッシュボードのログを取得し、`/api/ws` のクローズコードを確認してください（4403 = チャット WS がリクエストガードによって拒否された。例: Host/ピアの不一致。4401 = WS チケットが認証されなかった）。

### Config

`config.yaml` のためのフォームベースのエディタです。150 以上のすべての設定フィールドが `DEFAULT_CONFIG` から自動検出され、タブ付きのカテゴリに整理されます。

![Config 管理ページ — 左側にセクションフィルター、右側に自動検出されたフィールド](/img/dashboard/admin-config.png)


- **model** — デフォルトモデル、プロバイダー、ベース URL、reasoning 設定
- **terminal** — バックエンド（local/docker/ssh/modal）、タイムアウト、シェル設定
- **display** — スキン、ツール進捗、resume 表示、スピナー設定
- **agent** — 最大イテレーション数、ゲートウェイタイムアウト、サービスティア
- **delegation** — サブエージェントの上限、reasoning effort
- **memory** — プロバイダーの選択、コンテキスト注入の設定
- **approvals** — 危険なコマンドの承認モード（smart/manual/off）
- その他 — config.yaml の各セクションには対応するフォームフィールドがあります

既知の有効な値を持つフィールド（terminal backend、skin、approval mode など）はドロップダウンとしてレンダリングされます。真偽値はトグルとしてレンダリングされます。それ以外はすべてテキスト入力です。

**アクション:**

- **Save** — 変更を即座に `config.yaml` に書き込みます
- **Reset to defaults** — すべてのフィールドをデフォルト値に戻します（Save をクリックするまで保存されません）
- **Export** — 現在の設定を JSON としてダウンロードします
- **Import** — JSON 設定ファイルをアップロードして現在の値を置き換えます

:::tip
設定変更は、次のエージェントセッションまたはゲートウェイの再起動で有効になります。Web ダッシュボードは、`hermes config set` やゲートウェイが読み取るのと同じ `config.yaml` ファイルを編集します。
:::

### API Keys

API キーや認証情報が保存される `.env` ファイルを管理します。キーはカテゴリ別にグループ化されます。

- **LLM Providers** — OpenRouter、Anthropic、OpenAI、DeepSeek など
- **Tool API Keys** — Browserbase、Firecrawl、Tavily、Keenable、ElevenLabs など
- **Messaging Platforms** — Telegram、Discord、Slack のボットトークンなど
- **Agent Settings** — `API_SERVER_ENABLED` のような秘密でない環境変数

各キーには次が表示されます。
- 現在設定されているかどうか（値のマスキングされたプレビュー付き）
- 用途の説明
- プロバイダーのサインアップ／キー取得ページへのリンク
- 値を設定または更新するための入力フィールド
- 削除するための削除ボタン

高度な、またはめったに使われないキーは、デフォルトではトグルの背後に隠されています。

### Sessions

すべてのエージェントセッションを閲覧して調べます。各行には、セッションのタイトル、ソースプラットフォームのアイコン（CLI、Telegram、Discord、Slack、cron）、モデル名、メッセージ数、ツール呼び出し数、最後にアクティブだったのがどれくらい前かが表示されます。ライブセッションは脈打つバッジでマークされます。

- **Filter** — **Chats / Automation / All** タブでリストの範囲を決めます。*Chats*（デフォルト）は人間との会話を表示し、自動化のノイズ（cron、tool、API、ACP のセッション）を隠します。*Automation* はそれらのみを表示し、*All* はすべてを表示します。ソース指定のドロップダウンで、さらに単一のチャネル（例: Telegram のみ）に絞り込めます。検索はアクティブなフィルターに従います。
- **Search** — FTS5 を使ったすべてのメッセージ内容に対する全文検索。結果にはハイライトされたスニペットが表示され、展開すると最初に一致するメッセージへ自動スクロールします。
- **Stats** — サマリーバーに、合計セッション数、ストア内でアクティブなセッション数、アーカイブ数、合計メッセージ数、ソース別の内訳が表示されます。
- **Expand** — セッションをクリックして完全なメッセージ履歴を読み込みます。メッセージはロール（user、assistant、system、tool）別に色分けされ、シンタックスハイライト付きの Markdown としてレンダリングされます。
- **Tool calls** — ツール呼び出しを含む assistant メッセージは、関数名と JSON 引数を含む折りたたみ可能なブロックを表示します。
- **Rename** — セッションのタイトルをインラインで設定またはクリアします（鉛筆アイコン）。
- **Export** — セッション（メタデータ＋完全なメッセージ履歴）を JSON としてダウンロードします（ダウンロードアイコン）。
- **Prune** — ヘッダーの "Prune old sessions" ボタンで、N 日より古い終了済みセッションを削除します。
- **Delete** — ゴミ箱アイコンでセッションとそのメッセージ履歴を削除します。

![Sessions 管理ページ — 統計バー、prune、行ごとの名前変更／エクスポート／削除](/img/dashboard/admin-sessions.png)

### Logs

エージェント、ゲートウェイ、エラーのログファイルを、フィルタリングとライブ tailing 付きで表示します。

- **File** — `agent`、`errors`、`gateway` のログファイルを切り替えます
- **Level** — ログレベルでフィルタリングします: ALL、DEBUG、INFO、WARNING、ERROR
- **Component** — ソースコンポーネントでフィルタリングします: all、gateway、agent、tools、cli、cron
- **Lines** — 表示する行数を選択します（50、100、200、500）
- **Auto-refresh** — 5 秒ごとに新しいログ行をポーリングするライブ tailing を切り替えます
- **Color-coded** — ログ行は重大度別に色分けされます（エラーは赤、警告は黄、デバッグは淡色）

### Analytics

セッション履歴から計算される使用量とコストの分析です。期間（7、30、90 日）を選択すると、次が表示されます。

- **サマリーカード** — 合計トークン（入力／出力）、キャッシュヒット率、見積もりまたは実際の合計コスト、日次平均付きの合計セッション数
- **日次トークンチャート** — 1 日あたりの入力トークンと出力トークンの使用量を示す積み上げ棒グラフ。ホバー時のツールチップで内訳とコストを表示します
- **日次内訳テーブル** — 各日の日付、セッション数、入力トークン、出力トークン、キャッシュヒット率、コスト
- **モデルごとの内訳** — 使用された各モデル、そのセッション数、トークン使用量、見積もりコストを示すテーブル

### Cron

エージェントのプロンプトを定期的なスケジュールで実行する cron ジョブを作成・管理します。

- **Create** — 名前（オプション）、プロンプト、cron 式（例: `0 9 * * *`）、配信先（local、Telegram、Discord、Slack、email）を入力します
- **Job list** — 各ジョブには、名前、プロンプトのプレビュー、スケジュール式、状態バッジ（enabled/paused/error）、配信先、前回の実行時刻、次回の実行時刻が表示されます
- **Pause / Resume** — ジョブをアクティブ状態と一時停止状態の間で切り替えます
- **Edit** — 事前入力されたモーダルを開き、ジョブのプロンプト、スケジュール、名前、配信先を変更します
- **Trigger now** — 通常のスケジュール外でジョブを即座に実行します
- **Delete** — cron ジョブを完全に削除します

### Profiles

[プロファイル](../profiles.md) — 独自の設定、スキル、セッションを持つ分離された Hermes インスタンス — を作成・管理します。

- **プロファイルカード** — 各カードには、モデル／プロバイダー、スキル数、ゲートウェイの状態、説明、バッジ（active、default、alias）が表示されます
- **Create** — 名前と、オプションの clone-from-default / clone-everything / no-bundled-skills、説明、モデル。専用の Profile Builder ページ（`/profiles/new`）では完全なフロー（モデル、MCP、スキル）を利用できます
- **Manage skills & tools** — そのプロファイルにスコープされた Skills ページにジャンプします（サイドバーのプロファイルスイッチャーを設定します）
- **Set as active** — **今後の CLI／ゲートウェイの実行**が使用するスティッキーなデフォルトを切り替えます（`hermes profile use` と同じ）。これはダッシュボードが管理する対象を*変更しません* — それはプロファイルスイッチャーの役割です
- **Edit model / description / SOUL** — そのプロファイルに書き込むインラインエディタ
- **Rename / Delete** — 名前付きプロファイルのみ

### Skills

インストール済みのスキルとツールセットを閲覧、検索、切り替えし、ハブから新しいものをインストールします。スキルは `~/.hermes/skills/` から読み込まれ、カテゴリ別にグループ化されます。

- **Search** — 名前、説明、カテゴリでインストール済みのスキルとツールセットをフィルタリングします
- **Category filter** — カテゴリのピルをクリックして一覧を絞り込みます（例: MLOps、MCP、Red Teaming、AI）
- **Toggle** — スイッチで個々のスキルを有効／無効にします。変更は次のセッションで有効になります。
- **Toolsets** — 別のビューに、組み込みツールセット（ファイル操作、Web ブラウジングなど）が、そのアクティブ／非アクティブの状態、セットアップ要件、含まれるツールの一覧とともに表示されます
- **Browse hub** — 3 つ目のビューでは、すべてのソースにわたってスキルハブを検索し（`hermes skills search` と同じ）、任意の結果を識別子でライブのインストールログ付きでインストールでき、インストール済みスキルを更新する "Update all" ボタンも用意されています。

![Skills 管理ページ — Browse hub ビュー: 検索、インストール、更新](/img/dashboard/admin-skills-hub.png)

### MCP

CLI を使わずに [MCP](./mcp) サーバーを管理します。`hermes mcp` が読み取るのと同じ、`config.yaml` の `mcp_servers` ブロックを扱います。

**Your MCP servers:**

- **Add** — HTTP/SSE サーバー（URL）または stdio サーバー（コマンド＋引数）を登録します。stdio サーバーにはオプションで `KEY=VALUE` の環境変数を指定できます
- **Enable / disable** — サーバーを削除せずにオン／オフを切り替えます。無効にしたサーバーは設定に残るため、後で再度有効にできます。次のゲートウェイ再起動で有効になります。
- **Test** — サーバーに接続し、そのツールを一覧表示して切断します — エージェントが依存する前に接続を検証します
- **Remove** — 設定からサーバーを削除します
- シークレットらしい形の環境変数の値は、一覧ビューでマスキングされます

**Catalog:** Nous が承認した MCP サーバー（同梱の `optional-mcps/` カタログ）を閲覧し、ワンクリックでインストールできます。API キーが必要なエントリはインラインで入力を求め、値は `.env` に保存されます。これは `hermes mcp catalog` / `hermes mcp install` が使うのと同じカタログです。

![MCP 管理ページ — 有効／無効トグル付きのサーバー一覧とインストールカタログ](/img/dashboard/admin-mcp.png)

### Webhooks

動的な [webhook サブスクリプション](../messaging/webhooks.md)を管理します。まずメッセージング設定で webhook プラットフォームを有効にしておく必要があります。有効でない場合、ページにヒントが表示されます。

- **Create** — 名前、説明、イベントフィルター、配信先、オプションのダイレクト配信モード、エージェントプロンプト。作成時に、ページにルート URL と、コピー用の一度だけ表示される HMAC シークレットが表示されます。
- **Enable / disable** — サブスクリプションのオン／オフを切り替えます。無効にしたルートはサブスクリプションファイルに残りますが、ゲートウェイはその受信イベントを拒否します（403）。ゲートウェイはファイルをホットリロードするため、変更は次のイベントで有効になります — 再起動は不要です。
- **List** — 各サブスクリプションには、URL、イベント、配信先が表示されます
- **Delete** — サブスクリプションを削除します

![Webhooks 管理ページ — 有効／無効トグル付きのサブスクリプション](/img/dashboard/admin-webhooks.png)

### Pairing

CLI を使わずにメッセージングユーザーを承認・取り消します — リモートの管理者が Telegram/Discord などのユーザーをペアリング済みゲートウェイに迎え入れる方法です。`hermes pairing` と完全に同等の機能を持ちます。

- **Pending requests** — 各リクエストには、プラットフォーム、コード、ユーザー、経過時間が Approve ボタンとともに表示されます
- **Approved users** — 各ユーザーには、プラットフォームとユーザーが Revoke ボタンとともに表示されます
- **Clear pending** — 未処理のペアリングコードをすべて破棄します

![Pairing 管理ページ](/img/dashboard/admin-pairing.png)

### Channels

ブラウザから Hermes を任意のメッセージングプラットフォームに接続します — `hermes setup gateway` と完全に同等の機能を持ちます。ページには、サポートされているすべてのチャネル（Telegram、Discord、Slack、Matrix、Mattermost、WhatsApp、Signal、BlueBubbles/iMessage、Email、SMS/Twilio、DingTalk、Feishu/Lark、WeCom、WeChat、QQ Bot、および API サーバーと webhook エンドポイント）が、ライブの接続状態とともに一覧表示されます。インストール済みプラグインのプラットフォーム（Yuanbao、Home Assistant など）も表示されます。

- **Configure** — そのチャネルに必要なフィールド（ボットトークン、アプリトークン、サーバー URL、許可リストなど）だけを持つプラットフォームごとのフォームを開きます。シークレットはパスワード入力としてレンダリングされ、マスキングされて保存されます。フィールドを空のままにすると既存の値が保持されます。必須フィールドはマークされ、検証されます。"Setup guide" リンクは、そのプラットフォームの認証情報のドキュメントを指します。
- **Enable / disable** — チャネルのオン／オフを切り替えます。認証情報はディスクに残り、アクティブ状態のみが変わります。
- **Test** — チャネルが設定済みか、有効か、ゲートウェイからライブ接続を報告しているかを確認します。
- **Restart gateway** — 認証情報は `~/.hermes/.env` に、有効フラグは `config.yaml` に書き込まれます。ゲートウェイは次回の再起動時に有効な各チャネルに接続し、その再起動はページから直接トリガーできます。システムスコープのインストール（`hermes gateway install --system`）では、ダッシュボードは再起動を `sudo -n` で実行するため、ダッシュボードのユーザーにはパスワードなしの sudo が必要です。それがない場合、CLI が後で拒否する再起動を報告するのではなく、リクエストは即座に失敗します。

![Channels 管理ページ — 状態、有効トグル、プラットフォームごとのセットアップフォーム付きのすべてのメッセージングプラットフォーム](/img/dashboard/admin-channels.png)

### System

インストール全体の操作のための統合管理パネルです。

- **Host** — ライブなシステム統計: OS／カーネル、アーキテクチャ、ホスト名、Python と Hermes のバージョン、CPU コア数と使用率、メモリ、Hermes ホームのディスク使用量、稼働時間、ロードアベレージ。（CPU／メモリ／ディスクは `psutil` がインストールされている場合に取得されます。識別系のフィールドは常に表示されます。）Hermes のバージョンには**更新状態バッジ**（最新 / N コミット遅れ）と **Check for updates** ボタンが表示されます。git インストールで更新が利用可能な場合、**Update now** ボタンを押すと、取り込むコミット数を示す確認ダイアログが開き、その後バックグラウンドで `hermes update` を実行します。Docker/Nix インストールでは、ダッシュボードはその場で更新を適用できないため、代わりに正しい帯域外のコマンドを表示します。
- **Nous Portal** — ログイン状態、アクティブな推論プロバイダー、Tool Gateway のルーティングテーブル（どのツールが Portal 経由で、どれがローカルで実行されるか）と、サブスクリプションを管理するためのリンク。`hermes portal` の読み取り専用のミラーです。
- **Skill curator** — バックグラウンドのスキルメンテナンスの状態（active / paused、間隔、前回の実行）と、一時停止／再開および今すぐ実行のボタン。`hermes curator` のミラーです。
- **Gateway** — メッセージングゲートウェイの起動、停止、再起動と、ライブな状態（実行中／停止中、PID、状態）
- **Memory** — 外部メモリプロバイダー（または組み込みのみ）を選択し、組み込みの `MEMORY.md` / `USER.md` ストアをリセットします
- **Credential pool** — エージェントがラウンドロビンで使うローテーション用 API キー（プロバイダーごと）を追加・削除します。キーは一覧ではマスキングされ、生の値はエージェントにのみ渡されます。
- **Operations** — `doctor`、セキュリティ監査、バックアップの作成、バックアップアーカイブからの復元、スキルの更新、システムプロンプトのサイズ内訳の表示、サポートダンプの生成、廃止された設定のための設定マイグレーションを実行します。それぞれがバックグラウンドアクションを起動し、そのライブログがページにストリーミングされます。
- **Checkpoints** — `/rollback` のシャドウストアのサイズを確認し、prune します
- **Shell hooks** — 設定済みのフックを、その同意状態と実行可能状態とともに一覧表示し、フックを**作成**（イベント、コマンド、matcher、タイムアウト、オプトインの同意付与付き）・削除します。フックは任意のコマンドを実行するため、作成フォームにはセキュリティ警告があり、フックは同意が付与された後にのみ発火します。

![System 管理ページ — ホスト統計と Nous Portal の状態](/img/dashboard/admin-system-top.png)

![System 管理ページ — スキルキュレーター、ゲートウェイ、メモリ、認証情報プール](/img/dashboard/admin-system-curator.png)

![System 管理ページ — 操作、チェックポイント、シェルフック](/img/dashboard/admin-system-ops.png)

シェルフックの作成（同意チェックボックスと、任意のコマンドを実行する旨の警告に注意してください）:

![新しいシェルフックのモーダル](/img/dashboard/admin-hook-create.png)

:::warning セキュリティ
Web ダッシュボードは、API キーやシークレットを含む `.env` ファイルを読み書きします。デフォルトでは `127.0.0.1` にバインドされ、ローカルマシンからのみアクセスでき、ログインは不要です。loopback 以外の任意のアドレス（`0.0.0.0` を含む）にバインドすると[認証ゲート](#authentication-gated-mode)が有効になり、認証プロバイダー（ユーザー名／パスワードまたは OAuth）が設定されるまでサーバーは起動を拒否します。
:::

## ダッシュボードのバックグラウンドサービスのレポート {#reporting-dashboard-background-services}

Hermes は `hermes dashboard` や `hermes serve` のためのサービスを一切インストールしないため、バックグラウンドで動作しているものはすべて手動でセットアップされたものです: このページからコピーした systemd ユニット、launchd plist、Nix の `hermes-backend` ユニット、Docker イメージの `dashboard` スロットや Compose のサイドカー、または Desktop の SSH バックエンドです。それらと現在の所有権を一覧表示するには次のようにします。

```bash
hermes gateway migrate --backends --dry-run          # human-readable report
hermes gateway migrate --backends --dry-run --json   # the same report as JSON
```

レポートには、各バックエンドについて、そのスーパーバイザー（systemd のユニット名、スコープ、ユニットファイル、launchd のラベルと plist、Nix 管理、s6 スロット、コンテナ、Desktop、デタッチされたプロセス、または unknown）、それが担当する `HERMES_HOME`、およびその所有権が表示されます。バックエンドのマイグレーションはまだ利用できません。レポートは何も停止・再起動・kill せず、サービスファイル、ユニット、plist、台帳を編集・書き込みすることもなく、ネットワーク接続も開きません。スキャンが失敗した場合でも終了コード 0 で終了します。通常の Hermes の起動処理は、他の `hermes` コマンドと同様に最初に実行されます: `HERMES_HOME` のレイアウトと、存在しない場合はデフォルトの `SOUL.md` を作成し、インストールを `HERMES_HOME/installs/` に記録し、`logs/agent.log` と `logs/errors.log` を開き、`cache/scratch` を prune します。実行中のバックエンドは、ユニットの名前に関係なく、何を実行しているかによって検出されます。有効だが実行されていないユニットは、`hermes-serve*`、`hermes-dashboard*`、`hermes-backend*` という名前の場合にのみ一覧表示されます。

## `/reload` スラッシュコマンド

このダッシュボードの PR は、インタラクティブ CLI に `/reload` スラッシュコマンドも追加します。Web ダッシュボードから（または `.env` を直接編集して）API キーを変更した後、アクティブな CLI セッションで `/reload` を使うと、再起動せずに変更を反映できます。

```
You → /reload
  Reloaded .env (3 var(s) updated)
```

これは `~/.hermes/.env` を実行中のプロセスの環境に読み直します。ダッシュボードから新しいプロバイダーキーを追加して、すぐに使いたい場合に便利です。

## REST API

Web ダッシュボードは、フロントエンドが消費する REST API を公開します。自動化のために、これらのエンドポイントを直接呼び出すこともできます。

:::tip プロファイルスコープのエンドポイント
管理用エンドポイント群 — `/api/config`、`/api/env`、`/api/skills`、
`/api/tools/toolsets`、`/api/mcp`、
`/api/model/{info,options,auxiliary,set,recommended-default}`、
`/api/cron/{delivery-targets,blueprints}`、`/api/audio/voice-config`、
`/api/ops/debug-share`、`/api/learning/{graph,node}`、`/api/dashboard/plugins/hub` —
は、オプションの `?profile=<name>` クエリパラメータ（書き込みの場合は JSON ボディ内の `"profile"`）を受け付け、読み書きをそのプロファイルの `HERMES_HOME` にスコープします。省略した場合はダッシュボード自身のプロファイルになります。不明なプロファイル名には `404` が返ります。`/api/pty` WebSocket も同じパラメータを受け付け、選択されたプロファイルでチャットを起動します。
:::

### GET /api/status

エージェントバージョン、ゲートウェイの状態、プラットフォームの状態、アクティブセッション数を返します。

レスポンスには、2 つの参考用リソースブロックも含まれます（これらが `components`/`overall` のヘルス判定に影響することはありません）。

- **`memory`** — ゲートウェイの 30 秒ごとのハートビートとライフサイクル台帳から抽出されます。フィールド: `pressure`（`ok` / `elevated` / `critical` / `unknown`）、`gateway_rss_mb`、`system_total_mb`、`system_available_mb`、`swap_used_mb`、`sampled_at`、`boot_id`、`last_boot_unclean`、`last_boot_suspected_oom`。pressure は、システムの利用可能メモリが 128 MiB（または 15%）未満で `elevated`、64 MiB（または 5%）未満で `critical` になります — その後の不正な終了が OOM kill の疑いとしてフラグ付けされるのと同じレベルです。150 秒より古い（または未来の日付の）ハートビートは数値を保持しますが `pressure` を `unknown` に格下げするため、停止したゲートウェイの最後のサンプルがライブな値を装うことはできません。
- **`disk`** — `~/.hermes` を格納するボリュームのライブな `shutil.disk_usage()` サンプルです。フィールド: `pressure`、`free_mb`、`total_mb`、`used_percent`、`sampled_at`。pressure は、空きが 512 MB 未満（または使用率 85% 以上かつ余裕が 4 GB 未満）で `elevated`、空きが 256 MB 未満（または使用率 95% 以上かつ余裕が 1 GB 未満）で `critical` になります。

どちらの収集処理もフェイルセーフです: サンプリングのエラーが発生しても、ステータスエンドポイントを失敗させるのではなく、ブロックを `{"pressure": "unknown"}` に格下げします。`/api/status` はパブリックなため、数値は粗い精度（MB 単位、パーセント単位の整数）です。

### GET /api/chat/workspaces

新しい Chat タブのセッションが開始できるディレクトリ: プロファイルのプロジェクト（フォルダ付き）と発見された git リポジトリ（`root`、`label`、`sessions`、`last_active`）、および `default_cwd`（何も選択されていない場合にチャットが開始される場所）と `home` を返します。`?scan=1` を付けると、最初にホスト上の `desktop.repo_scan_roots` を再スキャンします。存在しないディレクトリではフェイルクローズする `/api/pty?cwd=<path>` と組み合わせて使います。

### GET /api/sessions

直近 20 件のセッションをメタデータ付き（モデル、トークン数、タイムスタンプ、プレビュー）で返します。

### GET /api/config

現在の `config.yaml` の内容を JSON として返します。

### GET /api/config/defaults

デフォルトの設定値を返します。

### GET /api/config/schema

すべての設定フィールドを記述するスキーマ（型、説明、カテゴリ、該当する場合は選択肢）を返します。フロントエンドはこれを使って、各フィールドに対して正しい入力ウィジェットをレンダリングします。

### PUT /api/config

新しい設定を保存します。ボディ: `{"config": {...}}`。

### GET /api/env

既知のすべての環境変数を、その設定済み／未設定の状態、マスキングされた値、説明、カテゴリとともに返します。

### PUT /api/env

環境変数を設定します。ボディ: `{"key": "VAR_NAME", "value": "secret"}`。

### DELETE /api/env

環境変数を削除します。ボディ: `{"key": "VAR_NAME"}`。

### GET /api/sessions/\{session_id\}

単一のセッションのメタデータを返します。

### GET /api/sessions/\{session_id\}/messages

メッセージ履歴の上限付きのページを、ツール呼び出しとタイムスタンプを含めて返します。デフォルトでは、最新の 500 件のメッセージを時系列順で返します。明示的なページネーションには `limit`（最大 500）、`offset`、`order=oldest|latest` を使用します。

### GET /api/sessions/search

メッセージ内容に対する全文検索。クエリパラメータ: `q`。一致するセッション ID をハイライトされたスニペットとともに返します。

### DELETE /api/sessions/\{session_id\}

セッションとそのメッセージ履歴を削除します。リクエストには、クライアントが `GET /api/sessions/{id}/mutation-snapshot` から読み取った mutation identity（`request_id`、`expected_revision`、`expected_generation`）を含める必要があります。一部のみの identity には `409 mutation_identity_required` が、まったく含まないリクエスト（Desktop 0.22.2 以前）には `409 client_upgrade_required` が返されます — バックエンドがリビジョンフェンスなしで編集を適用することはありません。削除はプロファイルのゲートウェイによって実行されるため、実行中またはキューに入ったターンを持つセッションは `409 session_busy` で、最後のターンがゲートウェイの再起動をまたいで失われたセッションは `409 unknown_execution` で拒否されます — どちらの場合も何も削除されません。実行済みのターンを持つセッションを削除すると、`state_meta` テーブルに小さな終端レシートの tombstone（ユーザーのテキストと履歴を取り除いた admission と worker の identity）が残ります。これにより、同じ削除 — または削除されたセッションを対象とした API 実行 — の厳密な再試行が、セッションを再作成するのではなく同じ応答を返せるようになり、削除された ID が再インポートされるのを防ぎます。これらは無期限に保持されます（実行済みのターン 1 つあたり数百バイト）。

### ゲートウェイがストアを所有している間の一括メンテナンス

`POST /api/sessions/bulk-delete`、`DELETE /api/sessions/empty`、`POST /api/sessions/prune`（dry-run 以外）、`POST /api/sessions/owner-backfill` は、多数の行を一度に書き換え、セッションごとのリビジョンフェンスをバイパスするため、**オフラインメンテナンス**です: これらはプロファイルの排他的な `gateway.lock` を取得します。そのプロファイルのゲートウェイが実行中の間は、`409` と `Exclusive maintenance refused: Gateway runtime already owns profile <home>.
Drain and stop the gateway, then retry.` を返し、何も変更しません。ゲートウェイを停止し（`hermes gateway stop`）、メンテナンスの呼び出しを実行してから、再度起動してください。`dry_run` の prune とすべての読み取りエンドポイントは、ライブなゲートウェイに対しても引き続き動作します。Desktop の一回限りのレガシー owner backfill は、この `409` を接続されたゲートウェイの定常状態として扱い（そのルートを持たないバックエンドと同様）、サイドバーを更新するたびに再試行することはありません。

### GET /api/logs

ログ行を返します。クエリパラメータ: `file`（agent/errors/gateway）、`lines`（行数）、`level`、`component`。

### GET /api/analytics/usage

トークン使用量、コスト、セッションの分析を返します。クエリパラメータ: `days`（デフォルト 30）。レスポンスには日次の内訳とモデルごとの集計が含まれます。

### GET /api/cron/jobs

設定されたすべての cron ジョブを、その状態、スケジュール、実行履歴とともに返します。

### POST /api/cron/jobs

新しい cron ジョブを作成します。ボディ: `{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`。

### POST /api/cron/jobs/\{job_id\}/pause

cron ジョブを一時停止します。

### POST /api/cron/jobs/\{job_id\}/resume

一時停止した cron ジョブを再開します。

### POST /api/cron/jobs/\{job_id\}/trigger

スケジュール外で cron ジョブを即座にトリガーします。

### DELETE /api/cron/jobs/\{job_id\}

cron ジョブを削除します。

### GET /api/skills

すべてのスキルを、その名前、説明、カテゴリ、有効状態とともに返します。

### PUT /api/skills/toggle

スキルを有効または無効にします。ボディ: `{"name": "skill-name", "enabled": true}`。

### GET /api/tools/toolsets

すべてのツールセットを、そのラベル、説明、ツール一覧、アクティブ／設定済みの状態とともに返します。

### 管理用エンドポイント

これらは MCP、Channels、Webhooks、Pairing、System の各ページを支えています。いずれも `/api/` の他の部分と同じ認証ゲートの背後にあります。

| メソッドとパス | 用途 |
|---------------|---------|
| `GET /api/mcp/servers` | 設定済みの MCP サーバーを一覧表示（環境変数の値はマスキング） |
| `POST /api/mcp/servers` | サーバーを追加。ボディ: `{name, url?, command?, args?, env?, auth?}` |
| `POST /api/mcp/servers/{name}/test` | 接続し、ツールを一覧表示して切断 |
| `PUT /api/mcp/servers/{name}/enabled` | サーバーを有効化／無効化 |
| `DELETE /api/mcp/servers/{name}` | サーバーを削除 |
| `GET /api/mcp/catalog` | Nous が承認した MCP カタログを閲覧 |
| `POST /api/mcp/catalog/install` | カタログのエントリをインストール（必要な環境変数付き） |
| `GET /api/messaging/platforms` | すべてのメッセージングチャネルを状態とプラットフォームごとのセットアップフィールド付きで一覧表示 |
| `PUT /api/messaging/platforms/{id}` | チャネルを設定。ボディ: `{enabled?, env?, clear_env?}`（env は `.env` に、enabled は `config.yaml` に書き込まれる） |
| `POST /api/messaging/platforms/{id}/test` | チャネルが設定済みか、有効か、接続されているかを報告 |
| `GET /api/pairing` | 保留中および承認済みのメッセージングユーザーを一覧表示 |
| `POST /api/pairing/approve` | コードを承認。ボディ: `{platform, code}` |
| `POST /api/pairing/revoke` | ユーザーを取り消し。ボディ: `{platform, user_id}` |
| `POST /api/pairing/clear-pending` | 保留中のコードをすべて破棄 |
| `GET /api/webhooks` | サブスクリプションとプラットフォームの有効状態を一覧表示 |
| `POST /api/webhooks` | サブスクリプションを作成（一度だけのシークレットを返す） |
| `DELETE /api/webhooks/{name}` | サブスクリプションを削除 |
| `GET /api/credentials/pool` | プールされたローテーション用キーを一覧表示（マスキング） |
| `POST /api/credentials/pool` | キーを追加。ボディ: `{provider, api_key, label?}` |
| `DELETE /api/credentials/pool/{provider}/{index}` | キーを削除（1 始まりのインデックス） |
| `GET /api/memory` | アクティブなプロバイダー、利用可能なプロバイダー、組み込みファイルのサイズ |
| `PUT /api/memory/provider` | プロバイダーを選択（空 = 組み込みのみ） |
| `POST /api/memory/reset` | 組み込みメモリをリセット。ボディ: `{target: all\|memory\|user}` |
| `POST /api/gateway/start` · `/stop` · `/restart` | ゲートウェイのライフサイクル（バックグラウンド実行） |
| `POST /api/ops/doctor` · `/security-audit` · `/backup` · `/import` | 診断とメンテナンス（バックグラウンド実行。`/api/actions/{name}/status` で tail） |
| `GET /api/ops/hooks` | 設定済みのシェルフックと許可リストの状態 |
| `GET /api/ops/checkpoints` · `POST .../prune` | `/rollback` ストアの確認／prune |
| `POST /api/ops/hooks` · `DELETE /api/ops/hooks` | シェルフックの作成／削除（同意が必要） |
| `GET /api/system/stats` | ホスト統計 — OS、CPU、メモリ、ディスク、稼働時間 |
| `GET /api/hermes/update/check` | 更新を適用せずに、更新の有無（遅れているコミット数、インストール方法）を報告。遅れている git インストールの場合は、変更内容の `commits` リスト（`sha`、`summary`、`author`、`at`）も返す。`?force=1` で 24 時間のキャッシュを無効化（チェックは GitHub API を経由し、`git fetch` は決して使わない） |
| `GET /api/curator` · `PUT .../paused` · `POST .../run` | スキルキュレーターの状態、一時停止／再開、実行 |
| `GET /api/portal` | Nous Portal の認証と Tool Gateway のルーティング（読み取り専用） |
| `POST /api/ops/prompt-size` · `/dump` · `/config-migrate` | 診断（バックグラウンド実行） |
| `POST /api/ops/debug-share` | 許可リストに含まれるデバッグサマリー（ログ行やメッセージ本文は含まない）を公開ペーストにアップロードし、URL を返す（同期）。ボディ: `{consent: true, redact?, lines?}`。`"consent": true` がない場合は `400` を返し、何もアップロードしない |
| `PUT /api/webhooks/{name}/enabled` | webhook ルートを有効化／無効化 |
| `POST /api/skills/hub/install` · `/uninstall` · `/update` | スキルハブのアクション（バックグラウンド実行） |
| `GET /api/skills/hub/search` | すべてのソースにわたってスキルハブを検索 |
| `GET /api/sessions/stats` | セッションストアの統計 |
| `PATCH /api/sessions/{id}` | セッションの名前変更／アーカイブ |
| `GET /api/sessions/{id}/export` | セッション（メタデータ＋メッセージ）を JSON としてエクスポート |
| `POST /api/sessions/prune` | N 日より古い終了済みセッションを削除 |
| `PUT /api/cron/jobs/{id}` | cron ジョブのプロンプト／スケジュール／名前／配信先を編集 |

## 認証（ゲートモード） {#authentication-gated-mode}

ダッシュボードがパブリックまたは loopback 以外のアドレス — `127.0.0.1` / `localhost` 以外のすべて — にバインドされると、Hermes Agent は認証ゲートを有効にします。すべてのリクエストは検証済みのセッション Cookie を持っている必要があり、そうでなければログインページに戻されます。3 つのプロバイダーが同梱されています。

- **[ユーザー名／パスワード](#usernamepassword-provider-no-oauth-idp)** — セルフホスト／オンプレミス／ホームラボのダッシュボードに認証を付ける最も簡単な方法です。外部の ID プロバイダーは不要です。**信頼できるネットワーク上か VPN の背後でのみ使用してください — パブリックなインターネットへの公開には使わないでください。**
- **[OAuth (Nous Portal)](#default-provider-nous-research)** — ホステッドデプロイと、パブリックなインターネット経由で到達可能なあらゆるダッシュボード向けで、[リモートの Hermes Desktop 接続](#connecting-hermes-desktop-to-a-remote-backend)にも推奨される方法です。すべてのログインがあなたの Nous アカウントに対して検証されるため、インターネットに面した用途に適したプロバイダーです。
- **[セルフホスト OIDC](#self-hosted-oidc-provider)** — 標準の OpenID Connect を介して独自の ID プロバイダー（Keycloak、Auth0、Okta、Google、OIDC ブリッジ経由の GitHub など）を持ち込むためのものです。Nous Portal は関与しません。準拠した OIDC サーバーを前段に置けば、パブリックなインターネットへの公開に適しています。

loopback にバインドされた、オペレーター所有のダッシュボードは影響を受けません — 認証もログインページもありません。

### ゲートが有効になる条件

| フラグ | 認証ゲート | ユースケース |
|-------|-----------|----------|
| `hermes dashboard`（デフォルト — `127.0.0.1` にバインド） | OFF | ローカル開発 |
| `hermes dashboard --host 0.0.0.0` | **ON** | リモート／本番 — ユーザー名／パスワードプロバイダーまたは OAuth で保護 |

ゲートは、バインドホストが `127.0.0.1`、`::1`、`localhost` のいずれでもない場合に限り有効になります。`0.0.0.0`（または任意の RFC1918 / LAN アドレス）にバインドするとゲートが有効になります。レガシーな `--insecure` フラグは**もはやゲートを無効にしません** — 後方互換性のために受け付けられますが、警告とともに無視されます。

:::danger `--insecure` は何もしません — 認証を無効にしません
2026 年 6 月の強化以降、`--insecure` はダッシュボードの認証をバイパスしなくなりました: loopback 以外へのバインドには常に認証プロバイダー（ユーザー名／パスワードプロバイダーまたは OAuth）が必要です。認証なしのダッシュボードが必要な場合は、`127.0.0.1` にバインドし、SSH トンネルや Tailscale 経由でアクセスしてください。
:::

### フェイルクローズのセマンティクス

ゲートが有効になる状況で `DashboardAuthProvider` が**1 つも**登録されていない場合（Nous プラグインもカスタムプラグインもない場合）、`hermes dashboard` は明示的なエラーメッセージとともにバインドを拒否します。「デフォルト拒否だがすべてを受け入れる」ようなフォールバックはありません — 誤って設定されたゲート付きダッシュボードが起動することはありません。

プロバイダーがまだ設定されていない状態で `hermes dashboard --host 0.0.0.0` を**対話的に**（実際のターミナルで）実行すると、Hermes は単に失敗するのではなく、その場でセットアップすることを提案します: **username & password**（`config.yaml` に `dashboard.basic_auth` を書き込み、数秒で実行できます）または **OAuth**（`hermes dashboard register` を案内します）を選択します。非対話的な呼び出し元 — Docker/s6、CI、パイプ経由の実行 — はプロンプトをスキップして上記のフェイルクローズのエラーになるため、無人のデプロイが認証なしで起動することはありません。

### デフォルトプロバイダー: Nous Research {#default-provider-nous-research}

同梱の `plugins/dashboard_auth/nous` プラグインは**常にインストールされて**おり、自動的に読み込まれます。クライアント ID が設定されていると、`nous` という名前の `DashboardAuthProvider` を自動登録します。

すべてのログインが Nous Portal に対して検証され、あなたの Nous アカウントで保護されるため、**Nous プロバイダーは、ダッシュボードをパブリックなインターネットに公開するのに適したプロバイダーです。**

#### ダッシュボードの登録 {#registering-a-dashboard}

Nous プロバイダーを使うには、OAuth クライアント ID（形式 `agent:{id}`）が必要です。入手方法は 2 つあります。

- **CLI — `hermes dashboard register`。** ダッシュボードがあるホストで実行します。既存の Nous ログインを解決し（ログインしていない場合は先に `hermes setup` を実行してください）、Portal にセルフホストの OAuth クライアントを登録し、`HERMES_DASHBOARD_OAUTH_CLIENT_ID` を `~/.hermes/.env` に書き込みます。オプションのフラグ: `--name`（人間が読めるラベル。指定しない場合は自動生成）と `--redirect-uri`（インターネットに面したホスト用のパブリックな HTTPS コールバック URL）。

  ```bash
  hermes dashboard register
  # ✓ Registered dashboard "swift_falcon"
  # …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
  ```

- **GUI — Local Dashboards ページ。** Nous Portal で [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) を開くと、ブラウザからセルフホストのダッシュボードを登録、命名、管理、取り消しできます。得られた `agent:{id}` クライアント ID を `HERMES_DASHBOARD_OAUTH_CLIENT_ID`（環境変数）または `dashboard.oauth.client_id`（config.yaml）にコピーしてください。CLI で登録したダッシュボードを取り消すのもここです。

#### 設定

プラグインは 2 つの場所から読み取り、環境変数が空でない値に設定されている場合はそちらが優先されます。

**`config.yaml`** — 正規の設定場所:

```yaml
dashboard:
  oauth:
    client_id: agent:01HXYZ…             # required to engage the gate
    token_leeway: 60                     # optional; seconds of clock-skew tolerance
```

**環境変数** — オペレーターによる上書き:

| 環境変数 | 上書き対象 | 形式 | プロビジョニング元 |
|---------|-----------|--------|----------------|
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | `dashboard.oauth.client_id` | `agent:{instance_id}` | `hermes dashboard register` |

Hermes Agent の慣例（`~/.hermes/.env` は API キー／シークレット専用）に従い、ローカル開発、オンプレミス、および直接管理するあらゆるデプロイでは、**これらの値は `config.yaml` に設定することを推奨します**。環境変数のパスは、ホスティングプラットフォームのシークレット注入が、誰もイメージ内の `config.yaml` を編集することなくデプロイごとの `client_id` を渡せるようにするために存在します — それが主な目的です。

空の環境変数の値は未設定として扱われるため、プロビジョニングされたが値が入っていないプラットフォームのシークレットが、有効な `config.yaml` のエントリを誤って覆い隠すことはありません。

`dashboard.oauth.token_leeway`（config.yaml のみ）は、JWT 検証時に Portal のアクセストークンの `exp`/`nbf`/`iat` クレームに適用されるクロックスキューの許容範囲（秒）です。デフォルトは `60`（RFC 7519 §4.1.4-4.1.6）なので、時計が Portal より遅れているホストでもログインに失敗しません。`0` で厳密な検証に戻り、無効な値（解析不能／負／非有限）は `0` にフェイルクローズします。セルフホスト OIDC プロバイダーには、同等の `dashboard.oauth.self_hosted.id_token_leeway` ノブがあります。

どちらのソースからも client_id が得られない場合、プラグインは具体的な理由を報告し、ダッシュボードのフェイルクローズのバインドエラーが何を修正すべきかを正確に教えてくれます。

```
Refusing to bind dashboard to 0.0.0.0 — the auth gate engages on
non-loopback binds, but no auth providers are registered.

Bundled providers reported these issues:
  • nous: HERMES_DASHBOARD_OAUTH_CLIENT_ID is not set (and
    dashboard.oauth.client_id in config.yaml is empty). …

Configure an auth provider before exposing the dashboard:
  • Password: set dashboard.basic_auth.username + password_hash in config.yaml
  • OAuth: run `hermes dashboard register` (Nous Portal) or install a
    DashboardAuthProvider plugin.
There is no unauthenticated public-bind option — to keep it local, bind
127.0.0.1 and tunnel in (SSH / Tailscale).
```

#### 実例: Nous Research

ログイン済みの Hermes インストールから、Nous でゲートされたダッシュボードまで 3 ステップです。

**1. ログインしてダッシュボードを登録する。** `hermes dashboard register` は既存の Nous ログインを使って OAuth クライアントをプロビジョニングし、`HERMES_DASHBOARD_OAUTH_CLIENT_ID` を `~/.hermes/.env` に書き込みます。

```bash
hermes setup            # if you're not already logged into Nous Portal
hermes dashboard register
# ✓ Registered dashboard "swift_falcon"
# …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
```

**2. 到達可能なアドレスでダッシュボードを実行する。** loopback 以外へのバインドで OAuth ゲートが有効になり、書き込んだばかりの `client_id` が `nous` プロバイダーを有効化します。

```bash
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

**3. ログインする。** `http://<host>:9119/` を開くと、`/login` に戻されます。**Sign in with Nous Research** をクリック → Portal で認証 → 認証済みのダッシュボードに戻ります。任意のマシンからゲートを確認します。

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

その後、`GET /api/auth/me` は検証済みのセッション（`provider: nous`）を返します。インターネットに面したホストでは、`--redirect-uri https://hermes.example.com/auth/callback` で登録し、OAuth コールバックがパブリック URL に解決されるよう `HERMES_DASHBOARD_PUBLIC_URL` を設定してください（[パブリック URL の上書き](#public-url-override)を参照）。

### ユーザー名／パスワードプロバイダー（OAuth IDP なし） {#usernamepassword-provider-no-oauth-idp}

OAuth の ID プロバイダーを接続したくない場合 — セルフホストで「ダッシュボードにパスワードを付けるだけ」のデプロイ — 同梱の `plugins/dashboard_auth/basic` プラグインが `basic` という名前の `DashboardAuthProvider` を登録し、OAuth リダイレクトの代わりに**ユーザー名とパスワード**で認証します。

これは OAuth プロバイダーと同じゲートに組み込まれます: ゲートは loopback 以外へのバインドで有効になり、ログインページはこのプロバイダー用に（"Log in with X" ボタンの代わりに）認証情報フォームを表示し、ログイン以降のすべて — セッション Cookie、透過的なリフレッシュ、WS チケット、ログアウト、監査ログ — は OAuth のパスと同一です。セッションはプロバイダー自身が発行する HMAC 署名付きトークンなので、**データベースも外部 IDP も不要**です。署名鍵は設定されたユーザー名とパスワードに結び付けられているため、どちらかを変更すると既存のすべてのセッションが終了します。ログアウトは、Hermes ホーム内の小さな `dashboard_auth/basic-revoked-sessions.json` ファイルを介して、そのログイン（とその更新）を取り消します。パスワードのハッシュ化には標準ライブラリの `scrypt` を使用します（サードパーティの依存関係なし）。

:::warning 信頼できるネットワーク上でのみ使用してください — パブリックなインターネットでは使わないでください
ユーザー名／パスワードプロバイダーは、**信頼できるネットワーク**上の、または **VPN** 経由でのみ到達可能な、セルフホスト／オンプレミス／ホームラボのダッシュボード向けです。外部の ID プロバイダー、MFA、ユーザーごとのアカウントを背後に持たない単一の共有認証情報で保護するため、**ダッシュボードをパブリックなインターネットに直接公開するのには適していません**。インターネットに面したダッシュボードには、代わりに [Nous Research プロバイダー](#default-provider-nous-research)（または独自の[セルフホスト OIDC](#self-hosted-oidc-provider) / [カスタム OAuth](#custom-providers) プロバイダー）を使用してください。
:::

#### 設定

Nous プロバイダーと同様に、`config.yaml`（正規）から読み取り、環境変数が空でない値に設定されている場合はそちらが優先されます。`username` と、`password_hash`（推奨）または `password` のいずれかが設定されている場合にのみ有効になります — それ以外の場合は何もしないため、OAuth ユーザーや loopback のオペレーターには影響しません。

**`config.yaml`:**

```yaml
dashboard:
  basic_auth:
    username: admin
    # Preferred — no plaintext at rest. Compute with:
    #   python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"
    password_hash: "scrypt$16384$8$1$…$…"
    # ...or a plaintext password (hashed in-memory at load; less safe at rest):
    # password: "s3cret"
    secret: "<32+ random bytes, base64 or hex>"  # token-signing key
    session_ttl_seconds: 43200                    # optional; access-token lifetime (default 12h)
```

**環境変数による上書き:**

| 環境変数 | 上書き対象 | 備考 |
|---------|-----------|-------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | `dashboard.basic_auth.username` | 有効化に必須 |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | `dashboard.basic_auth.password_hash` | 推奨（保存時に平文なし） |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | `dashboard.basic_auth.password` | 平文。環境変数でローテーションできるよう、**config の `password_hash` より優先されます** |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | `dashboard.basic_auth.secret` | トークン署名鍵 |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | `dashboard.basic_auth.session_ttl_seconds` | アクセストークンの有効期間 |

:::caution 安定したセッションのために明示的な `secret` を設定してください
`secret` が空の場合、プロセスごとにランダムな署名鍵が生成されます。単一のプロセスなら問題ありませんが、**再起動のたびにすべてのセッションが無効になり**、セッションは**複数のワーカーにまたがりません**。再起動後も維持される／マルチワーカーのデプロイでは、明示的な `secret` を設定してください。
:::

`/auth/password-login` エンドポイントはクライアント IP ごとにレート制限され（デフォルトは 1 分あたり 10 回 → HTTP 429）、不明なユーザーと誤ったパスワードの両方に対して単一の汎用的な `401 Invalid credentials` を返すため、ユーザー名列挙のオラクルとして使うことはできません。

#### 実例: ユーザー名／パスワード

何もない状態から、信頼できるネットワーク上のパスワードでゲートされたダッシュボードまで 3 ステップです。

**1. `~/.hermes/.env` に認証情報を設定する。** 平文が保存されないようパスワードをハッシュ化し、セッションが再起動後も維持されるよう安定した署名シークレットを設定します。

```bash
# Compute a scrypt hash of your chosen password:
HASH=$(python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('choose-a-strong-password'))")

cat >> ~/.hermes/.env <<EOF
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH=$HASH
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env
```

**2. 到達可能なアドレスでダッシュボードを実行する。** loopback 以外へのバインドでゲートが有効になり、ユーザー名とハッシュが `basic` プロバイダーを有効化します。

```bash
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

**3. ログインする。** `http://<host>:9119/` を開くと、`/login` — （"Sign in with X" ボタンではなく）**認証情報フォーム** — に戻されます。`admin` とパスワードを入力 → 認証済みのダッシュボードに移動します。任意のマシンからゲートを確認します。

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["basic"]
```

その後、`GET /api/auth/me` は検証済みのセッション（`provider: basic`）を返します。これは VPN の背後に置いてください — 上記の警告を参照してください。パブリックなホストには、代わりに [Nous Research](#default-provider-nous-research) または[セルフホスト OIDC](#self-hosted-oidc-provider) プロバイダーを使用してください。

#### 独自のパスワードプロバイダーを書く

`basic` は拡張ポイントの 1 つの実装にすぎません。どのプラグインでもパスワードプロバイダーを登録できます: `DashboardAuthProvider` のサブクラスで `supports_password = True` を設定し、`complete_password_login(*, username, password) -> Session` を実装します（拒否時は `InvalidCredentialsError` を、バックエンドのストアがダウンしている場合は `ProviderError` を送出します）。純粋なパスワードプロバイダーでは、OAuth の `start_login` / `complete_login` メソッドは `NotImplementedError` のスタブのままでかまいません。これは LDAP バインド、認証情報データベース、その他リダイレクトを伴わない認証方式のための方法です — フォーム、ルート、Cookie、リフレッシュはフレームワークが処理します。

### デプロイメントセッショントークンプロバイダー（トークンのみのデプロイ） {#deployment-session-token-provider-token-only-deployments}

同梱の `plugins/dashboard_auth/session_token` プラグインは、**デプロイメントセッショントークン**（`HERMES_DASHBOARD_SESSION_TOKEN`）を、ゲートモードの第一級の認証情報にします。これは、シークレット管理されたデプロイがすでにプロビジョニングしている形です — 単一の共有トークン（nix/agenix は復号されたファイルとして渡し、Nix モジュールは `backend.sessionTokenFile` から `HERMES_DASHBOARD_SESSION_TOKEN=file:<path>` を設定します）で、Hermes Desktop のリモートゲートウェイの **"Session token"** フィールドが提示するものです。このプロバイダーがない場合、プロビジョニングされたトークンだけを持つ loopback 以外へのバインドはフェイルクローズのゲートに引っかかり、オペレーターはデプロイが求めてもいない basic 認証や OAuth の認証情報を強いられていました。

環境変数がプロビジョニングされていると、プロバイダーは自動的に登録され、`verify_session` が提示された認証情報をそれと定数時間で比較します。トークンモードのクライアントが使う両方の経路がこの継ぎ目を通ります — REST（Desktop の "Session token" 接続は `X-Hermes-Session-Token` ヘッダーでトークンを送信します。`Authorization` ヘッダーの bearer トークンも機能し、両方が送られた場合はそちらが優先されます。専用ヘッダーで機能するのはこのプロバイダーのトークンのみです）と、`/api/ws` の `?token=` アップグレード認証情報です — そのため、プロビジョニングされた 1 つのトークンで両方が認可されます。**ブラウザのサインインフローはありません**: `/login` はサインインボタンの代わりに認証情報の経路を説明する注記を表示し、このプロバイダーが自動 SSO リダイレクトで提示されることはありません。

:::warning トークンはパスワードのように扱ってください
プロビジョニングされたトークンは、それを持つ人にダッシュボードへの完全なアクセスを与えます。シークレットマネージャーからプロビジョニングし（Nix ストアのパスは決して使わないでください）、シークレットを再デプロイしてローテーションしてください — ローテーションは保存されたすべてのクライアント側のコピーを無効にし、新しいトークンが貼り付けられるまで Desktop では「HTTP には到達できるが WebSocket が失敗する」状態として現れます。背後にユーザーごとの ID はありません: 共有またはインターネットに面したデプロイでは、[Nous Research](#default-provider-nous-research) または[セルフホスト OIDC](#self-hosted-oidc-provider) プロバイダーを推奨します。
:::

**有効化とオプトアウト。** 登録は認証情報そのものによって行われます: `HERMES_DASHBOARD_SESSION_TOKEN` が設定されるまで、プロバイダーは何もしません（環境変数が優先されます。`file:<path>` は起動時に読み込まれ、存在しないか空の場合は明確に失敗します）。トークンを内部専用（loopback の SPA 利用、デスクトップが起動したバックエンド）に留めつつ、ゲートの認証は basic 認証／OAuth で行う場合は、プロバイダーを無効にします。

```yaml
dashboard:
  session_token_auth:
    enabled: false
```

任意のマシンから、トークンでゲートされたデプロイを確認します。

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["session-token"]

curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $TOKEN" http://<host>:9119/api/auth/me
# 200
```

### セルフホスト OIDC プロバイダー {#self-hosted-oidc-provider}

独自の ID プロバイダーを運用している場合、同梱の `plugins/dashboard_auth/self_hosted` プラグインが、**標準の OpenID Connect** を使ってダッシュボードをそれに対して認証します — IDP ごとのコードも Nous Portal も不要です。準拠したあらゆる OIDC サーバーに対して検証済みで、動作します。

> **Authentik · Keycloak · Zitadel · Authelia · Auth0 · Okta · Google · …**

Nous プロバイダーと同様に、自動的に読み込まれ、設定されて初めて自身を登録するため、loopback のダッシュボードでは何もしません。

#### 設定

**issuer** と **client_id**（パブリックな PKCE クライアント — クライアントシークレットなし）を設定します。プラグインは IDP の `authorization_endpoint`、`token_endpoint`、`jwks_uri` を `{issuer}/.well-known/openid-configuration` から取得するため、エンドポイント URL をハードコードする必要はありません。

**`config.yaml`** — 正規の設定場所:

```yaml
dashboard:
  oauth:
    provider: self-hosted
    self_hosted:
      issuer: https://auth.example.com/application/o/hermes/   # required
      client_id: hermes-dashboard                              # required
      scopes: "openid profile email"                           # optional (this is the default)
      id_token_leeway: 60                                      # optional; seconds of clock-skew tolerance
```

**環境変数** — オペレーターによる上書き（空でない値に設定されている場合は `config.yaml` より環境変数が優先されます。空の値は未設定として扱われます）:

| 環境変数 | 上書き対象 | 備考 |
|---------|-----------|-------|
| `HERMES_DASHBOARD_OIDC_ISSUER` | `dashboard.oauth.self_hosted.issuer` | OIDC の issuer URL — 必須 |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | `dashboard.oauth.self_hosted.client_id` | パブリッククライアント ID — 必須 |
| `HERMES_DASHBOARD_OIDC_SCOPES` | `dashboard.oauth.self_hosted.scopes` | デフォルトは `openid profile email` |

IDP では、authorization-code + PKCE（S256）グラントを持つ**パブリック**なアプリケーション／クライアントを登録し、ダッシュボードのコールバックを許可されたリダイレクト URI として追加してください。コールバックは `<dashboard public URL>/auth/callback` です（プロキシの背後でダッシュボードがパブリック URL をどのように導出するかについては[パブリック URL の上書き](#public-url-override)を参照してください）。

#### 検証する内容

サインインできるすべてのアカウントは、マシンの所有者であるあなたとして動作します。セルフホストのログインプロバイダーを使う場合は、プロバイダー側でアプリを自分のアカウントのみに制限してください（例: Okta のアプリ割り当て、Microsoft Entra の "Assignment required"、自分だけに制限した Google Workspace アプリ）。そうしないと、以前と同様に、プロバイダーが受け入れる誰もが完全なアクセス権を得ます。

プロバイダーは、OpenID Connect の **ID トークン**（RS256/ES256）を、発見された `jwks_uri` に対して検証し、`iss` と `aud` クレームを設定された `issuer` と `client_id` に固定します。時刻クレーム（`exp`/`nbf`/`iat`）は 60 秒のクロックスキュー許容範囲（RFC 7519 §4.1.4-4.1.6）で検証されるため、時計が IDP より遅れているダッシュボードホストでもログインに失敗しません。`config.yaml` の `dashboard.oauth.self_hosted.id_token_leeway` で調整できます（`60` 未満の値で厳しくなり、`0` で厳密な検証に戻り、無効な値は `0` にフェイルクローズします）。標準の OIDC クレームは、ダッシュボードのセッションに次のようにマッピングされます。

| セッションフィールド | クレーム |
|---------------|----------|
| `user_id` | `sub`（必須） |
| `email` | `email` |
| `display_name` | `name` → `preferred_username` → `nickname` → `email` |
| `org_id` | `org_id` / `organization`、なければ結合した `groups` |

ID を確立するのは ID トークンです — アクセストークンは不透明なものとして扱われます（OIDC の仕様は、それが JWT であることを要求していません）。エンドポイント URL は HTTPS である必要があり（ローカル開発用の IDP 向けに loopback の `http://` は許可されます）、ディスカバリドキュメントが公開する `issuer` は設定されたものと一致する必要があります（末尾のスラッシュの違いは許容されます）。IDP がリフレッシュトークンを発行する場合は、標準の `refresh_token` グラントを介したサイレントな再認証に使われます。ログアウトは、公開されていれば IDP の RFC 7009 `revocation_endpoint` を呼び出します。

> **コンフィデンシャルクライアント**（`client_secret` を持つもの）はまだサポートされていません — ブラウザに面したダッシュボードで一般的な選択である、パブリック + PKCE クライアントを設定してください。

#### 実例: Keycloak

[Keycloak](https://www.keycloak.org/) は、ローカルテスト用に立ち上げるのが最も簡単なセルフホスト OIDC サーバーの 1 つです — 開発モード（インメモリ DB）で単一のコンテナとして動作し、教科書どおりの OIDC ディスカバリを公開します。このウォークスルーでは、何もない状態から数分で動作するダッシュボードのログインまで進みます。

**1. 事前設定済みのレルムで Keycloak を実行する。** 次のレルムエクスポートを `realm-hermes.json` として保存します — `hermes` レルム、**パブリックな PKCE クライアント**（`hermes-dashboard`）、テストユーザーを定義しており、すべて起動時にインポートされるため、管理 UI でクリックする必要はありません。

```json
{
  "realm": "hermes",
  "enabled": true,
  "clients": [
    {
      "clientId": "hermes-dashboard",
      "name": "Hermes Agent Dashboard",
      "enabled": true,
      "publicClient": true,
      "standardFlowEnabled": true,
      "protocol": "openid-connect",
      "redirectUris": ["http://localhost:9119/auth/callback"],
      "webOrigins": ["http://localhost:9119"],
      "attributes": { "pkce.code.challenge.method": "S256" }
    }
  ],
  "users": [
    {
      "username": "testuser",
      "enabled": true,
      "emailVerified": true,
      "email": "testuser@example.com",
      "firstName": "Test",
      "lastName": "User",
      "credentials": [
        { "type": "password", "value": "testpassword", "temporary": false }
      ]
    }
  ]
}
```

そのファイルをインポートディレクトリにマウントして起動します（Keycloak 26 以降）。

```bash
docker run --rm -p 8080:8080 \
  -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
  -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin \
  -v "$PWD/realm-hermes.json:/opt/keycloak/data/import/realm-hermes.json:ro" \
  quay.io/keycloak/keycloak:26.0 \
  start-dev --import-realm
```

起動すると、レルムは標準の OIDC ディスカバリを `http://localhost:8080/realms/hermes/.well-known/openid-configuration`（issuer は `http://localhost:8080/realms/hermes`）で公開します。管理コンソールは `http://localhost:8080/`（`admin` / `admin`）にあります。

**2. ダッシュボードをそれに向ける。** セルフホストプラグインは loopback の `http://` issuer を許可するため（loopback 以外の issuer には HTTPS が必要です）、ローカルの Keycloak はそのまま動作します。

```bash
export HERMES_DASHBOARD_OIDC_ISSUER="http://localhost:8080/realms/hermes"
export HERMES_DASHBOARD_OIDC_CLIENT_ID="hermes-dashboard"
export HERMES_DASHBOARD_PUBLIC_URL="http://localhost:9119"
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

`HERMES_DASHBOARD_PUBLIC_URL` は、ダッシュボードの OAuth コールバックが `http://localhost:9119/auth/callback` — 上でレルムが登録したリダイレクト URI — であることをダッシュボードに伝えます。`0.0.0.0` へのバインド（loopback 以外へのバインド）が OAuth ゲートを有効にします。

**3. ログインする。** `http://localhost:9119/` を開くと、`/login` に戻されます。**Sign in with Self-Hosted OIDC** をクリック → Keycloak で `testuser` / `testpassword` として認証 → 認証済みのダッシュボードに戻ります。サイドバーには `Logged in as Test User via self-hosted` と表示され、`GET /api/auth/me` は検証済みのセッション（`provider: self-hosted`、`email: testuser@example.com`）を返します。

> 別のホスト／ポートでバインドまたはブラウズする場合は、そのオリジンの
> `…/auth/callback` を、Keycloak 管理コンソールのクライアントの **Valid redirect URIs**
> （Clients → hermes-dashboard → Settings）に追加してください。同じパターンは
> Authentik、Zitadel、Authelia、その他の OIDC サーバーでも機能します — 異なるのは
> issuer URL とクライアント登録の UI だけです。

### パブリック URL の上書き {#public-url-override}

デフォルトでは、ダッシュボードは OAuth コールバック URL をリクエストから再構築します — `X-Forwarded-Host` + `X-Forwarded-Proto` + `X-Forwarded-Prefix`（uvicorn が `proxy_headers=True` で設定されている場合。`start_server` はゲート下でこれを有効にします）。3 つのヘッダーすべてを正しく設定するリバースプロキシの背後では、そのまま動作します。

それらのヘッダーを確実に転送しないリバースプロキシの背後にあるデプロイ（手動の nginx 設定、オンプレミスのイングレス、部分的なプロキシチェーンを持つカスタムドメインのデプロイ）では、`dashboard.public_url`（または `HERMES_DASHBOARD_PUBLIC_URL`）に、ダッシュボードにアクセスする**完全なパブリック URL** を設定してください。

```yaml
dashboard:
  public_url: "https://dashboard.example.com/hermes"
  trusted_proxies:
    - "172.20.0.5"
```

設定すると、OAuth コールバック URL はそのまま `<public_url>/auth/callback` になります — オペレーターがパブリック URL を明示的に宣言しているため、そのコードパスでは `X-Forwarded-Prefix` は無視されます。これは意図的なものです: プレフィックスを上に重ねると、プレフィックスがすでに `public_url` に含まれている一般的なケースで、プレフィックスが二重になってしまいます。

`public_url` のホスト名は、HTTP の `Host` および WebSocket の `Origin` の値としても**完全一致**で受け付けられます。これにより、ブラウザに面したホスト名を保持したまま `127.0.0.1` にバインドされたダッシュボードに転送するリバースプロキシをサポートします。ワイルドカードやサフィックス一致は許可されないため、`dashboard.example.com.evil.test` のような攻撃者のホストは引き続き DNS リバインディングガードによって拒否されます。

loopback 以外の `public_url` を宣言すると、バックエンドが loopback にバインドしている場合でも、常にダッシュボードの認証ゲートが有効になります。先にパスワードまたは OAuth プロバイダーを設定してください。設定がない場合、Hermes は起動時にフェイルクローズします。これにより、ローカルの SPA セッショントークンがプロキシを通じてリモートの認証メカニズムになるのを防ぎます。このモードでは、Uvicorn もプロキシヘッダーの処理を有効にします。loopback のプロキシは自動的に信頼されます。TLS ターミネーターが別のコンテナやホストから接続する場合は、その正確な IP アドレスを `dashboard.trusted_proxies` に追加するか、アドレスが動的な場合は専用のプロキシネットワーク用に範囲を限定した CIDR を追加してください。

```yaml
dashboard:
  public_url: "https://dashboard.example.com/hermes"
  trusted_proxies:
    - "172.20.0.0/24"
```

`X-Forwarded-Proto` と `X-Forwarded-For` を供給できるのは、一覧に含まれるピアのみです。Hermes は常に loopback の信頼を維持し、`*`、`0.0.0.0/0`、`::/0` を拒否します。ネットワークを信頼するということは、そのネットワーク上のすべてのコンテナやマシンが転送メタデータを供給できるということなので、正確なプロキシ IP か、専用のプロキシ専用ネットワークを優先してください。信頼済みプロキシのエントリがない場合、そのプロキシの背後にあるクライアントは、パスワードログインのレート制限とネイティブサインインの上限をプロキシと共有し、認証の監査イベントにはプロキシのアドレスが記録されます。

```bash
# Backend remains reachable only on this machine.
hermes dashboard --host 127.0.0.1 --port 9119 --no-open
```

TLS リバースプロキシを `http://127.0.0.1:9119` に向け、`dashboard.public_url` には同じ外部オリジンを使用してください。

Tailscale Serve は、このデプロイ形態の一例です: `https://<machine>.<tailnet>.ts.net` のホスト名で tailnet 限定の HTTPS を終端しつつ、loopback のダッシュボードにプロキシできます。その正確な HTTPS オリジンを `dashboard.public_url` として使用してください。これも loopback 以外のブラウザに面したオリジンとして扱われるため、ダッシュボードの認証プロバイダーが必要です。ただし、サービスをパブリックなインターネットから到達可能にする必要はありません。

他のダッシュボード設定と同じ優先順位です — `config.yaml` より環境変数が優先されます。

| 設定場所 | 上書きパス | 使用する場面 |
|---------|---------------|-------------|
| `config.yaml` の `dashboard.public_url` | `HERMES_DASHBOARD_PUBLIC_URL` | ローカル開発／オンプレミス（正規） |
| `HERMES_DASHBOARD_PUBLIC_URL` 環境変数 | — | ホスティングプラットフォームのシークレット／CI |
| （未設定） | — | デフォルト — `X-Forwarded-*` ヘッダーから再構築 |

検証では、`http://` / `https://` スキームがない値、ホストがない値、引用符／山括弧／空白／制御文字を含む値は拒否されます。不正な値は黙ってヘッダーからの再構築にフォールスルーするため、ユーザーを悪意のある URL に送るのではなく、ログインフローが動作し続けます。

> **注:** `public_url` が上書きするのは OAuth コールバック URL のみです。`Secure` Cookie フラグは引き続き `request.url.scheme` によって制御され、接続元のピアが loopback であるか `trusted_proxies` に含まれている場合にのみ `X-Forwarded-Proto` を使用します。プロキシが loopback 上にない場合は、HTTPS の `public_url` を TLS 終端と範囲を限定した信頼済みプロキシのエントリと組み合わせてください。

### OAuth フロー

プロバイダーは [Nous Portal OAuth contract v1](https://github.com/NousResearch/nous-account-service/blob/main/docs/agent-dashboard-oauth-contract.md) — PKCE（S256）付きの authorization-code グラント — を実装しています。

1. ユーザーがセッション Cookie なしで `/` にアクセス → ゲートが `/login` にリダイレクトします。
2. ログインページに "Continue with Nous Research" ボタンが表示される → `/auth/login?provider=nous`。
3. サーバーが PKCE の状態を短命の Cookie に保存し、ユーザーを `https://portal.nousresearch.com/oauth/authorize?…` にリダイレクトします。
4. ユーザーが Portal で認証し、`/auth/callback?code=…&state=…` に到達します。
5. サーバーが `POST /api/oauth/token` でコードをアクセストークンと交換し、Portal の JWKS（`/.well-known/jwks.json`）に対して JWT の署名を検証し、`hermes_session_at` Cookie を設定します。
6. ユーザーは `/`（または `next=` クエリパラメータを介して元のディープリンクのパス）にリダイレクトされます。

アクセストークンの TTL は 15 分です。**contract v1 にはリフレッシュトークンはありません** — トークンが期限切れになると、SPA の fetch ラッパーが 401 のエンベロープを検出し、ページ全体を `/login` に遷移させてフローを再実行します。

### 設定される Cookie

| 名前 | 有効期間 | 備考 |
|------|----------|-------|
| `hermes_session_at` | トークン TTL（15 分） | HttpOnly、SameSite=Lax、HTTPS の場合は Secure |
| `hermes_session_pkce` | 10 分 | HttpOnly。往復の間、PKCE verifier とプロバイダーのヒントを保持します。HTTPS では SameSite=None + Secure（クロスサイトの IDP リダイレクトチェーンを生き延びる必要があります — Chromium は、クロスサイトのチェーン内の 302 で設定された SameSite=Lax Cookie を破棄します）。loopback の HTTP では SameSite=Lax |
| `hermes_session_rt` | v1 では未使用 | 前方互換性のために予約されています。`refresh_token` が空の場合は書き込まれません |

3 つとも `Path=/` です。セッション Cookie は `SameSite=Lax` で、PKCE Cookie は HTTPS で設定される場合 `SameSite=None` です（表を参照）。`Secure` フラグは、ダッシュボードに HTTPS でアクセスした場合に設定されます（リクエスト URL のスキームで検出されます — `proxy_headers=True` の下で、上流の TLS ターミネーターからの `X-Forwarded-Proto` を尊重します）。

### ログアウト

サイドバーのウィジェットには、ログアウトアイコンとともに `Logged in as <user_id…> via nous` と表示されます。クリックすると `/auth/logout` に POST し、すべてのダッシュボード認証の Cookie をクリアして `/login` にリダイレクトします。

### 監査ログ

すべてのログイン開始、成功、失敗、セッション検証の失敗は、JSON 行として `$HERMES_HOME/logs/dashboard-auth.log` に書き込まれます。機密フィールド（`access_token`、`refresh_token`、`code`、`code_verifier`、`state`、`Authorization` ヘッダー）は、ログに記録される前にマスキングされます。

### カスタムプロバイダー {#custom-providers}

Nous 以外の OAuth プロバイダー（例: Google、GitHub、カスタム OIDC）を組み込むには、`DashboardAuthProvider` を登録するプラグインを作成します。

```python
# ~/.hermes/plugins/dashboard-auth-myidp/__init__.py
from hermes_cli.dashboard_auth import DashboardAuthProvider, Session, LoginStart

class MyIdPProvider(DashboardAuthProvider):
    name = "myidp"
    display_name = "My Identity Provider"

    def start_login(self, *, redirect_uri): ...
    def complete_login(self, *, code, state, code_verifier, redirect_uri): ...
    def verify_session(self, *, access_token): ...
    def refresh_session(self, *, refresh_token): ...
    def revoke_session(self, *, refresh_token): ...

def register(ctx):
    ctx.register_dashboard_auth_provider(MyIdPProvider())
```

ログインページには登録されたすべてのプロバイダーが一覧表示されます。複数のプロバイダーを重ねることができ、ユーザーは `/login` でそのうちの 1 つを選択します。

IdP でトークンが取り消されないプロバイダーのログアウトには、2 つのオプションのフックがあります: アクセストークンがリフレッシュトークンより長く存続しうる場合は `revoke_access_session(access_token=...)` をオーバーライドします（ログアウトはアクセストークンの Cookie も渡します）。使用済みのリフレッシュトークンが有効なまま残る場合は `refresh_reuse_detection = False` を設定し、ゲートの 30 秒のリフレッシュリプレイキャッシュが、ログアウトによって取り消された後のローテーションを提供しないようにします。

### 非対話型（bearer トークン）認証

対話的な人間のログイン（セッション Cookie ＋リフレッシュ）と並んで、`DashboardAuthProvider` ABC は、`supports_token = True` + `verify_token(token=...)` による**非対話的なサービス間**の機能をサポートします。プロバイダーがオプトインすると、受信した `Authorization: Bearer <token>` が検証され、成功した場合は、そのプロバイダーがトークン認証可能とマークしたエンドポイントに対して、`TokenPrincipal` がリクエストにアタッチされます（`request.state.token_principal`）— Cookie もリダイレクトもリフレッシュもありません。

同梱されている最初の利用者は **drain** プロバイダー（`plugins/dashboard_auth/drain`）です: `nous-account-service` が `HERMES_DASHBOARD_DRAIN_SECRET` を介してエージェントごとのシークレットをプロビジョニングし、プロバイダーは受信した bearer トークンをそれと定数時間の比較で検証し、`/api/gateway/drain` をトークン認証可能として登録します。これは**フェイルクローズ**します — 弱い／短いシークレット（256 ビット未満）は登録時に拒否され、エンドポイントは無効のままになります。環境変数が未設定の場合は何もしません。動作に関するノブ（`scope`、`min_secret_chars`）は `config.yaml` の `dashboard.drain_auth` の下にあります。

カスタムプロバイダーも、同じ方法で `supports_token`/`verify_token` を実装し、独自のマシン認証可能なエンドポイントを公開できます。

### ピアパス（`/api/peer`）

プロバイダーは**ピアパス**（`supports_peer_pass = True` + `verify_peer_pass(token=...)`）も受け付けることができます: あるエージェント向けに発行される短命のトークンで、エージェントの `API_SERVER_KEY` なしに、別の Hermes エージェントがダッシュボードを通じてこのエージェントの API サーバーを呼び出せるようにします。

- `/api/peer/<path>` は loopback 上のゲートウェイ API サーバーに転送し、その途中で API サーバーのキーを追加します。`/api/peer/p/<profile>/<path>` は 1 つのプロファイルに到達します。プロファイル名は API サーバーの `/p/<profile>/` プレフィックスでの名前と同じです（プロファイルがその名前を持たない限り、`hermes` がデフォルトプロファイルです）。
- 各パスには、パスが持つ権限が必要です。
  `agents:message` は、`hermes peer` が使うものを、エージェントの Bot Chat（すべてのピアメッセージが届く 1 つのセッション）内でのみカバーします: ヘルス、ケイパビリティ、Bot Chat の検索と作成、その読み取り、チャットターン（通常およびストリーミング）、その中での実行（開始、状態、イベント、停止）。
  各ターンはターミナルを含むエージェントのツールで実行されるため、メッセージングのパスを持つ人はエージェントにタスクを実行させることができます。
  メッセージングのターンが運ぶのはそのメッセージのみです（実行の場合は Bot Chat のセッション ID も）。モデル、システムプロンプト、履歴を選択するには `agents:control` が必要です。
  `agents:control` は任意のセッションでのメッセージングを含み、ユーザー向け API の残りを追加します: 実行の承認、clarify、steer、resolve-unknown。セッションの一覧表示、fork、メッセージ、モデル、名前変更、削除。パスの期限切れ後も実行され続けるスケジュール済みジョブ。モデル、スキル、ツールセット。chat completions と responses。
  API サーバーのメモリスコープを選択する `X-Hermes-Session-Key` を設定または受信できるのは `agents:control` のみです。
- すべてのチャットターンと実行は、パスの保持者に帰属します: リクエストボディの内容にかかわらず、ターンの作成者 ID は検証済みの保持者に設定されます。
- その他のパスは決して転送されません。これには `/api/cron/fire`、プラットフォームのイベント受信、RoomLink の付与、`/health/detailed` が含まれます。
- スリープ中または再起動中のゲートウェイは、`Retry-After` 付きの 503 を返します。ゲートウェイが受信後にドロップしたリクエストは、実行された可能性があるため、再試行のヒントなしの 502 `peer_gateway_dropped` を返します。
- `/api/peer` を開けるのはピアパスのみです。これは `verify_token` とは別にチェックされるため、パスが他のプロバイダーのトークンルートを認可することはなく、ダッシュボードのスコープを持たないパスがダッシュボードのセッションを開くこともありません。
- `/api/status` は `peer_access` を報告します: このエージェントが受け付けるピアパスのコントラクト（現在は `1`）、または登録されたプロバイダーがピアパスを受け付けない場合は `0` です。`/api/peer` を持たないエージェントはパスをダッシュボードのサインインとして読み取るため、パスの発行者は、これを報告するエージェントに対してのみパスを発行する必要があります。
- コントロールアクションは、パスの保持者とともに `peer_control` イベントとして `logs/dashboard-auth.log` に書き込まれ、このルートを通じて作成されたジョブは、その origin に保持者を `peer_principal` として記録します。

同梱の Nous プロバイダーは、ダッシュボードのサインインと同じ Portal の鍵とオーディエンスでピアパスを検証します。ダッシュボードのサインインは `agent_dashboard:access` を持ち、ピアパスはピアの権限のみを持ちます。
`agent_dashboard:access` を持たない Portal トークン（`scope` クレームをまったく持たないものを含む）が、ダッシュボードのセッションを開くことはありません。

### ゲートが有効か確認する

```bash
# Quick env-var path.
HERMES_DASHBOARD_OAUTH_CLIENT_ID=agent:test \
  hermes dashboard --host 0.0.0.0

# Or the equivalent via config.yaml (recommended for local dev / on-prem):
#
#   dashboard:
#     oauth:
#       client_id: agent:test
#
# then just:
hermes dashboard --host 0.0.0.0

# Hit /api/status to see the gate state:
curl -s http://127.0.0.1:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

ダッシュボードの React の StatusPage は、同じフィールドを "Web server" の下に表示します。サインインすると、サイドバーの AuthWidget に現在の ID が表示されます。

## Hermes Desktop をリモートバックエンドに接続する {#connecting-hermes-desktop-to-a-remote-backend-1}

Hermes Desktop は、別のマシン（VPS、ホームサーバー、Tailscale の背後にある Mini）で動作している Hermes バックエンドを操作できます。アプリでは **Settings → Gateways → Remote gateway** の下にあり、**Remote URL** と **Sign in** の方法を尋ねられます。（デスクトップアプリ自体 — インストール、設定、チャット — については [Hermes Desktop](/docs/user-guide/desktop) のページを参照してください。）

リモートのダッシュボードは同梱の認証プロバイダーのいずれかで保護し、デスクトップアプリはバックエンドが公開しているプロバイダーに対してサインインします。自分のマシンの外から到達可能なバックエンド — VPS、パブリックなホスト、インターネットに面したあらゆるもの — には、**OAuth (Nous Portal)** プロバイダーを推奨します（[`hermes dashboard register`](#registering-a-dashboard) で登録し、*Sign in with Nous Research* でサインインします）。同梱の[ユーザー名／パスワードプロバイダー](#usernamepassword-provider-no-oauth-idp)は、バックエンドが信頼できる LAN 上にあるか VPN 経由でのみ到達可能な場合に最も手早い選択肢ですが、**パブリックなインターネットへの直接の公開には適していません**。ダッシュボードを loopback 以外のアドレスにバインドすると認証ゲートが有効になります。サインイン後、Desktop はセッションをチャットの WebSocket に自動的に再利用します — コピーしたり貼り付けたりするトークンはありません。

以下の手順では、信頼できるネットワーク上で最も手早く立ち上げられるユーザー名／パスワードのパスを使用します。OAuth のパスについては、[デフォルトプロバイダー: Nous Research](#default-provider-nous-research) を参照してください。

### バックエンド側（リモートマシン）

```bash
# 1. Set the dashboard login credentials in ~/.hermes/.env (secrets file, 0600).
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
EOF
# Recommended: a stable signing secret so sessions survive restarts.
# This runs openssl, so the secret is unique to you.
echo "HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)" >> ~/.hermes/.env
chmod 600 ~/.hermes/.env

# 2. Run the dashboard bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes dashboard --no-open --host 0.0.0.0 --port 9119
```

この手順の以前のバージョンでは、シークレットの行を引用符付きの heredoc の中に書いていたため、ランダムな値ではなく `$(openssl rand -base64 32)` というテキストが保存されていました。`grep BASIC_AUTH_SECRET ~/.hermes/.env` でそのテキストが表示される場合は、その行を新しく生成したシークレットに置き換え、ダッシュボードを再起動してください。

平文を保存したくない場合は、代わりに scrypt ハッシュを指定した `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` を使用してください — 設定項目の全体については[ユーザー名／パスワードプロバイダー](#usernamepassword-provider-no-oauth-idp)を参照してください。

ダッシュボードを systemd サービスとして実行する場合、ユニットに `EnvironmentFile=%h/.hermes/.env` があれば `~/.hermes/.env` が自動的に読み込まれるため、起動時に認証情報が環境に含まれます。

:::warning
ダッシュボードは `.env`（API キー、シークレット）を読み書きし、エージェントのコマンドを実行できます。ここで示した**ユーザー名／パスワード**のセットアップは信頼できるネットワーク向けです — パスワードで保護したダッシュボードを、オープンなインターネットに直接公開しないでください。VPN の背後に置いてください。[Tailscale](https://tailscale.com/) がすっきりした選択肢です: マシンの tailscale IP にバインドし（`--host <tailscale-ip>`）、Remote URL として `http://<tailscale-ip>:9119` を使用します。到達できるのは tailnet 上のデバイスのみです。パブリックなインターネット経由でバックエンドに到達するには、代わりに **OAuth (Nous Portal)** プロバイダーを使用してください。
:::

### Hermes Desktop 側

**Settings → Gateways → Remote gateway:**

- **Remote URL** — `http://<backend-host>:9119`（リバースプロキシを前段に置く場合は、`/hermes` のようなパスプレフィックスもサポートされます）
- **Sign in** — アプリがユーザー名／パスワードのゲートウェイを検出して **Sign in** ボタンを表示します。クリックして、ステップ 1 の認証情報を入力します
- **Save and reconnect** — デスクトップシェルをリモートバックエンドに切り替えます

バックエンドで `HERMES_DASHBOARD_BASIC_AUTH_SECRET` が設定されていれば、セッションは自動的にリフレッシュされ、再起動後も維持されます。

### 環境変数による上書き

アプリ内の設定の代わりに、起動前に環境変数でデスクトップをバックエンドに向けることもできます。`HERMES_DESKTOP_REMOTE_URL` が設定されていると、保存されたアプリ内の URL を上書きします（Gateway 設定パネルには "env override" バッジが表示され、編集が無効になります）。それでも、パネルからユーザー名とパスワードで **Sign in** する必要があります。

| 環境変数 | 値 |
|---------|-------|
| `HERMES_DESKTOP_REMOTE_URL` | `http://<backend-host>:9119` |

### トラブルシューティング

- **"Remote gateway incomplete"** — リモート URL を入力していません。
- **サインインが 401 / "Invalid credentials" で失敗する** — ユーザー名またはパスワードが、バックエンドの `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` と一致していません。バックエンドは不明なユーザーと誤ったパスワードに対して同じ汎用的なエラーを返すため、両方を確認してください。`curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'` でゲートを確認してください — `true` と報告し、`"basic"` を含んでいるはずです。
- **"Sign in" ボタンがなく、代わりにセッショントークンを求められる** — ユーザー名／パスワードプロバイダーが有効ではありません（`/api/status` に `"basic"` が含まれません）。ユーザー名とパスワード（またはパスワードハッシュ）が設定され、ダッシュボードのプロセスがそれらを読み込んでいることを確認してください。
- **再起動のたびにサインアウトされる** — `HERMES_DASHBOARD_BASIC_AUTH_SECRET` を安定した値に設定してください。設定しないと、署名鍵が起動ごとに再生成されます。
- **接続が拒否される／タイムアウトする** — バックエンドが到達可能なアドレスではなく `127.0.0.1`（デフォルト）にバインドされているか、ファイアウォール／VPN がポートをブロックしています。`0.0.0.0` または tailscale IP にバインドし、信頼できるネットワークに対してポートを開いてください。

## CORS

Web サーバーが CORS を許可し、ブラウザの WebSocket アップグレードを受け付けるのは、自身のオリジン — サーバーがバインドされているポート上の loopback オリジン（`localhost`、`127.0.0.1`、`[::1]`）— に対してのみです。これは、デフォルトの `hermes dashboard` では `http://127.0.0.1:9119`、または選択した `--port`、あるいは Desktop のバックエンドがバインドするエフェメラルポートです。パッケージ版のデスクトップアプリは `file://` から読み込まれるため、影響を受けません。

デスクトップの開発ビルドが起動したバックエンドは、そのビルド自身のレンダラー（継承した loopback の `HERMES_DESKTOP_DEV_SERVER`。例えば `5174+N` 上の [`hgui` スロット](/docs/developer-guide/worktree-ui-dev#hgui--desktop-app-from-the-worktree)）も信頼します。固定の開発サーバーポートは信頼されません: `5173` や `5174` 上の Vite アプリを含め、他のローカルポートから配信されるページには CORS ヘッダーが付与されず、その WebSocket アップグレードは拒否されるため、セッショントークンを読み取ったりターミナルを開いたりすることはできません。

ポートフォワード（`ssh -L 8000:localhost:9119`、`docker -p 8080:9119`）経由でアクセスした場合でも、ダッシュボード自身のページは WebSocket アップグレードを開けます: アップグレードの送信先であるホストとポートを正確に示す Origin は、そのページだからです。

別の loopback ポート上で `Host` をバックエンドのアドレスに書き換えるリバースプロキシ（`127.0.0.1:9119` の前段にある `https://localhost:8443` 上の nginx）は、そのアドレスを `dashboard.public_url`（Webapp の場合は `webapp.public_url`）に設定する必要があります。すると、その正確なオリジン（スキーム、ホスト、ポート）も信頼されます。他の loopback ポートは引き続き信頼されません。`0.0.0.0` や `::` へのバインドでも同じルールが適用されます: サインイン Cookie は `SameSite=Lax` であり、ブラウザは同じホストの別のポート上のページからもそれらを送信します。

デスクトップの開発ビルドは、SSH 経由で起動するバックエンドにこの変数を渡し、別のレンダラー URL 用に以前起動したバックエンドを置き換えます。Desktop が同じマシン上のゲートウェイに開くソケットは、Origin を持たない、一回限りのチケット付きネイティブ接続です。そのため、開発ビルドはすでに実行中のローカルゲートウェイを再起動せずに使用でき、起動するゲートウェイにはこの変数を渡しません（そのゲートウェイは開発セッションより長く存続します）。開発レンダラーの Origin を、Desktop が起動していないバックエンド（例えば、自分で起動して直接アクセスするリモートバックエンド）に提示する接続の場合にのみ、そのバックエンドを `HERMES_DESKTOP_DEV_SERVER` にビルドのレンダラー URL を設定して起動する必要があります。

## 開発

Web ダッシュボードのフロントエンドに貢献する場合は次のようにします。

```bash
# Terminal 1: start the backend API
hermes dashboard --no-open

# Terminal 2: start the Vite dev server with HMR
cd web/
pnpm install
pnpm run dev
```

`http://localhost:5173` の Vite 開発サーバーは、WebSocket を含む `/api` リクエストを `http://127.0.0.1:9119` の FastAPI バックエンドにプロキシします（別のバックエンドを使うには `HERMES_DASHBOARD_URL` を設定します）。セッショントークンはサーバー側でバックエンドから読み取ります。WebSocket アップグレードは `Host` と `Origin` を変更せずに転送するため、ページの Origin は接続先のホストとポートを示し、バックエンドはポート `5173` を信頼することなく、それを同一オリジンとして受け入れます。これは、`HERMES_DASHBOARD_URL` がバックエンドへのポートフォワード（例えば `ssh -L 8000:localhost:9119` 経由の `http://127.0.0.1:8000`）を指している場合にも成り立ちます。Vite 自身の CORS は無効になっているため、他のローカルページが開発サーバーからトークンを読み取ることもできません。

フロントエンドは React 19、TypeScript、Tailwind CSS v4、shadcn/ui スタイルのコンポーネントで構築されています。本番ビルドは `hermes_cli/web_dist/` に出力され、FastAPI サーバーが静的な SPA として配信します。

ビルドは CPU とメモリを大量に消費するため（Vite 8 の Rust ネイティブな Rolldown バンドラーはコア間で並列化します）、**デフォルトでリソースが制限されています**: V8 ヒープは上限が設けられ（`--max-old-space-size`。コンテナのメモリ制限から算出）、ネイティブバンドラーのスレッドプールは利用可能なコアの半分に制限されます（`RAYON_NUM_THREADS`）。これにより、小規模な VPS ホストでビルド中に発生していた 200% 超の CPU スパイクや OOM を防ぎます（#63338）。`HERMES_WEB_BUILD_MAX_OLD_SPACE_SIZE` / `HERMES_WEB_BUILD_THREADS` で上書きできます。ホストが CPU を十分に割けない場合は、`HERMES_WEB_BUILD_LIGHT=1` を設定して上限を厳しくします（1 スレッド、1 GB ヒープ）。

## 更新時の自動ビルド

`hermes update` を実行すると、Hermes がプロビジョニングする Node と pnpm を使って Web フロントエンドが自動的に再ビルドされます。これにより、ダッシュボードがコードの更新と同期した状態に保たれます。更新処理がそれらをプロビジョニングできない場合（例えば遅延インストールが無効になっている場合）、フロントエンドのビルドはスキップされ、`hermes dashboard` が初回起動時にビルドします。

## テーマとプラグイン

ダッシュボードには 8 つの組み込みテーマが同梱されており、ユーザー定義のテーマ、プラグインタブ、バックエンド API ルートで拡張できます。いずれもドロップインで、リポジトリのクローンは不要です。

ヘッダーバーから**テーマをライブで切り替え**られます。言語切り替えの横にあるパレットアイコンをクリックします。選択は `config.yaml` の `dashboard.theme` に永続化され、ページ読み込み時に復元されます。

同じピッカーから**フォントを独立して変更**できます — テーマ一覧の下にある **Font** セクションで、アクティブなテーマの UI フォントを上書きします。選択はテーマを切り替えても維持されます（`config.yaml` → `dashboard.font`）。**Theme default** を選ぶとクリアされ、アクティブなテーマ自身のフォントに戻ります。

組み込みテーマ:

| テーマ | 特徴 |
|-------|-----------|
| **Hermes Teal**（`default`） | ダークティール＋クリーム、システムフォント、ゆったりした余白 |
| **Hermes Teal (Large)**（`default-large`） | default と同じで 18px のテキストとより広い余白 |
| **Nous Blue**（`nous-blue`） | Nous ブランドのブルーのアクセント、ゆとりのある余白 |
| **Midnight**（`midnight`） | 深いブルーバイオレット、Inter ＋ JetBrains Mono |
| **Ember**（`ember`） | 暖かいクリムゾン＋ブロンズ、Spectral セリフ ＋ IBM Plex Mono |
| **Mono**（`mono`） | グレースケール、IBM Plex、コンパクト |
| **Cyberpunk**（`cyberpunk`） | 黒地にネオングリーン、Share Tech Mono |
| **Rosé**（`rose`） | ピンク＋アイボリー、Fraunces セリフ、ゆったり |

独自のテーマを作成したり、プラグインタブを追加したり、シェルスロットに注入したり、プラグイン固有の REST エンドポイントを公開したりするには、**[ダッシュボードの拡張](./extending-the-dashboard)** を参照してください。完全なガイドでは次を扱います。

- テーマ YAML スキーマ — palette、typography、layout、assets、componentStyles、colorOverrides、customCSS
- レイアウトのバリアント — `standard`、`cockpit`、`tiled`
- プラグインのマニフェスト、SDK、シェルスロット、ページスコープのスロット（組み込みページを上書きせずにウィジェットを注入）、バックエンドの FastAPI ルート
- テーマとプラグインを組み合わせた完全なウォークスルー（Strike Freedom cockpit デモ）
- 検出、リロード、トラブルシューティング
