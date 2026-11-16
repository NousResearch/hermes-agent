---
sidebar_position: 2
title: "Hermes Agent の設定"
description: "Hermes Agent を設定する — config.yaml、プロバイダー、モデル、API キーなど"
---

# Hermes Agent の設定

このページの Python 依存関係のコマンドは、
[PM で準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を使用します。
依存関係を変更した後は、チェックアウトを再アクティベートして Hermes を再起動してください。

すべての設定は、簡単にアクセスできるよう `~/.hermes/` ディレクトリに保存されます。

:::tip 動作する `config.yaml` への最短ルート
`hermes setup --portal` を実行してください — 1 回の OAuth で、YAML を手で編集することなくモデルプロバイダーと 4 つすべての Tool Gateway ツールが使えるようになります。Portal の購読者は、トークン課金のプロバイダーが 10% 割引にもなります。[Nous Portal](/docs/integrations/nous-portal) を参照してください。
:::

## ディレクトリ構成

```text
~/.hermes/
├── config.yaml     # Settings (model, terminal, TTS, compression, etc.)
├── .env            # API keys and secrets
├── auth.json       # OAuth provider credentials (Nous Portal, etc.)
├── SOUL.md         # Primary agent identity (slot #1 in system prompt)
├── memories/       # Persistent memory (MEMORY.md, USER.md)
├── skills/         # Agent-created skills (managed via skill_manage tool)
├── cron/           # Scheduled jobs
├── sessions/       # Gateway sessions
└── logs/           # Logs (errors.log, gateway.log — secrets auto-redacted)
```

## 設定の管理

```bash
hermes config              # View current configuration
hermes config edit         # Open config.yaml in your editor
hermes config get KEY      # Print a resolved value
hermes config set KEY VAL  # Set a specific value
hermes config unset KEY    # Remove a user-set value
hermes config check        # Check for missing options and stale saved selections
hermes config migrate      # Interactively add missing options

# Examples:
hermes config get model
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config unset terminal.backend
hermes config set OPENROUTER_API_KEY sk-or-...  # Saves to .env
```

:::tip
`hermes config set` コマンドは、値を自動的に適切なファイルへ振り分けます — `UPPER_SNAKE` 形式の名前（`OPENROUTER_API_KEY`、`DISCORD_HOME_CHANNEL`、`TELEGRAM_GROUP_ALLOWED_USERS`、`HERMES_TIMEZONE` など）はすべて環境変数であり、`config.yaml` ではなく `.env` に保存されます。ドット区切りの設定は `config.yaml` に保存されます。その他の `UPPER_SNAKE` 名もそのまま `.env` に保存されます（プラグインやスキル向けにプロセス環境へエクスポートされます）。env 書き込みの拒否リストに含まれる名前（`HERMES_YOLO_MODE`、`PATH` など）は拒否されます。既知のキーが誤ったプレフィックスで書かれた場合（`discord.foo` 自体が既知のキーであるときの `gateway.discord.foo` など）は、何も書き込まれる前に did-you-mean 付きで拒否されます。それでも書き込むには `--force` を渡してください。既知のセクション配下のその他の未知のパス（`agent.max_turnz` のようなタイプミスや、デフォルト値が用意されていない実行時に読み込まれるキー）は、スキーマだけでは両者を区別できないため、did-you-mean の通知とともに書き込まれます。このようなパスに対する `hermes config get` は、ファイル内の値とともに「Hermes がそれを読み込まない可能性がある」という通知を stderr に出力するため、残骸のキーが有効な設定として気づかれずに紛れ込むことはありません。
:::

## 設定の優先順位

設定は次の順序で解決されます（優先度の高い順）。

1. **CLI 引数** — 例: `hermes chat --model anthropic/claude-sonnet-4`（呼び出しごとのオーバーライド）
2. **`~/.hermes/config.yaml`** — シークレット以外のすべての設定を扱う主要な設定ファイル
3. **`~/.hermes/.env`** — 環境変数のフォールバック。シークレット（API キー、トークン、パスワード）には**必須**
4. **組み込みのデフォルト値** — 他に何も設定されていない場合に使われる、ハードコードされた安全なデフォルト値

:::info 目安
シークレット（API キー、ボットトークン、パスワード）は `.env` に入れます。それ以外（モデル、ターミナルバックエンド、圧縮設定、メモリ上限、ツールセット）はすべて `config.yaml` に入れます。両方に設定されている場合、シークレット以外の設定では `config.yaml` が優先されます。
:::

### Hermes が書き込む認証情報

Hermes 自身が設定フィールドの認証情報を書き込む場合、値はプロファイルの `.env` に保存され、`config.yaml` にはそれへの参照だけが保存されます。これは、これらのフィールドに対する `hermes config set`、ダッシュボード / デスクトップのモデルおよび補助モデルのピッカー、そして `hermes model` のカスタムエンドポイントおよび補助モデルのフローが対象です。

| フィールド | `.env` での保存名 | `config.yaml` に保存される内容 |
|---|---|---|
| `model.api_key`（`hermes config set`） | `HERMES_MODEL_API_KEY` | `${HERMES_MODEL_API_KEY}` |
| ダッシュボードで選択したカスタムエンドポイントの `model.api_key` | `HERMES_CUSTOM_<HOST>[_<PORT>][_<PATH>]_<HASH>_API_KEY` | `${...}` |
| `auxiliary.<task>.api_key` | `AUXILIARY_<TASK>_API_KEY` | `${AUXILIARY_<TASK>_API_KEY}` |
| `delegation.api_key` | `HERMES_DELEGATION_API_KEY` | `${HERMES_DELEGATION_API_KEY}` |
| 新しい `custom_providers` エントリ | `HERMES_CUSTOM_<HOST>[_<PORT>][_<PATH>]_<HASH>_API_KEY` | `key_env: <その名前>` |
| `custom_providers.<N>.api_key`（`hermes config set`） | エントリの `HERMES_CUSTOM_..._API_KEY`（`base_url` のないエントリはその名前を使用: `HERMES_PROVIDER_<NAME>_<HASH>_API_KEY`） | `${...}` |
| `providers.<name>.api_key`（`hermes config set`）、ダッシュボードの Custom Endpoints エントリ | `HERMES_PROVIDER_<NAME>_<HASH>_API_KEY` | `${...}` |

パスのセグメントは、`/v1` 以外の場合にのみ追加されます。すべて小文字ではないプラグインのタスクキー（`My_Task`）にも `_<HASH>` が付くため、`my_task` と変数を共有することはありません。`<HASH>` は正確なエンドポイント URL（またはプロバイダー名）から導出される 16 進 8 文字で、`my-host.com` と `my.host.com` のように名前が似ているエンドポイント同士が変数を共有することはありません。Hermes は、`config.yaml` 内の別のエンドポイントがすでに別のキーで読み込んでいる変数を上書きすることを拒否します。参照として入力した値（`hermes config set model.api_key '${MY_KEY}'`）は、入力したとおりに保存されます。これらの変数は、`env_passthrough` に列挙されていても `inherit_credentials` が有効でも、ターミナルやサブプロセスの環境から常に取り除かれます。ホスト間で `.env` を共有しないリモート設定バックエンドでは、値は指定されたとおりに保存されます。

稼働中のゲートウェイは毎ターンの開始時に `.env` を再読み込みするため、次のメッセージでは再起動なしで新しく保存されたキーが使われます。起動時にのみ `.env` を読み込み、ゲートウェイのターンで駆動されないプロセスは、`/reload` または再起動後にキーを取り込みます。それまでは、キーが保存されたばかりの補助タスクは、展開されていない `${VAR}` のテキストをキーとして送信します。

`config.yaml` にすでに存在するインラインの認証情報は引き続き機能し、移行もされませんが、推奨されません。`hermes config set <field> <value>` で手動で移動すると、参照として書き換えられます。

:::tip 組織でのデプロイ
管理者は、システムレベルの管理ディレクトリを通じて、標準ユーザーが
オーバーライドできない特定の設定値やシークレット値を固定できます。
[Managed Scope](/docs/user-guide/managed-scope) を参照してください。
:::

## ランタイムの上限

長時間稼働する Hermes のサーバー機能（ゲートウェイや
`hermes serve --isolated` を含む）は、オペレーティングシステムがサポートしている場合、
起動時に設定された `RLIMIT_NOFILE` のソフトリミットを適用します。

```yaml
runtime:
  nofile_soft_limit: 4096
```

デフォルトは `4096` です。Hermes は目標値をオペレーティングシステムのハードリミットに
制限し、すでにそれより高いソフトリミットを持つプロセスの値を下げることはありません。調整を無効にするには
値を `0`、`false`、または `null` に設定します。Windows や、上限を変更できない
サンドボックスでは、上限を変更せずに起動が続行されます。

glibc を使用する Linux では、ゲートウェイ、`hermes serve`、`hermes dashboard` は、プロセスが作成できる
malloc アリーナの数にも上限を設けます。

```yaml
runtime:
  malloc_arena_max: 4
```

そうしない場合、glibc は新しいスレッドごとに独自のアリーナを割り当て（CPU コアあたり最大 8 個）、
各アリーナの解放済みページを保持し続けます。短命のワーカースレッドが多い長時間稼働のゲートウェイでは、
解放済みだが OS に返却されていないメモリを数百 MB 抱え込むことになります。
この上限はサーバーがスレッドを開始する前の起動時に設定され、Hermes のサーバープロセス自体にのみ
適用されます — それが実行するコマンド（ターミナルツール、MCP サーバー、ビルド）は glibc のデフォルトのままです。
明示的な `MALLOC_ARENA_MAX`（または `GLIBC_TUNABLES=glibc.malloc.arena_max=…`）は、
起動前にエクスポートされたものでもプロファイルの `.env` で設定されたものでも優先されます。glibc の
デフォルトを維持するには、値を `0`、`false`、または `null` に設定します。他のプラットフォームではこの設定は無視されます。

## データベースの設定

`database:` セクションは、Hermes が SQLite の状態データベース（`state.db`）を開く方法を
制御します。このデータベースにはセッション、メッセージ、ゲートウェイのルーティングが保存されます。

```yaml
database:
  # Journal mode for state.db: wal (default) or delete.
  # Use delete on filesystems where WAL is unsafe (network mounts). On
  # virtiofs/9p bind mounts (Docker Desktop, Podman on macOS, OrbStack)
  # Hermes detects the mount and creates fresh databases in delete mode
  # automatically. On Linux NFS/CIFS/SMB/sshfs mounts, a fresh database
  # also starts in delete mode unless this key is set (set `wal` to force
  # WAL there). Note: an existing on-disk WAL database is never
  # live-downgraded — Hermes keeps WAL and logs a warning telling you the
  # configured delete did not apply (or that the WAL database sits on a
  # cross-VM or network mount). To convert an existing database, stop
  # every process using it and run
  # `hermes sessions set-journal-mode delete` (see Sessions).
  journal_mode: wal

  # Durability level for every state.db connection: OFF, NORMAL, FULL,
  # EXTRA (or 0-3). Unset leaves SQLite's compile-time default, which
  # differs between interpreter builds. On macOS this is a floor, not a
  # pin: values below FULL are refused to protect against Darwin fsync
  # reordering; EXTRA is honored.
  # synchronous: FULL

  # Optional WAL sizing pragmas (integers). Unset = SQLite defaults.
  # wal_autocheckpoint: 1000     # pages between automatic checkpoints
  # journal_size_limit: 67108864 # cap the WAL/journal size in bytes
```

また Hermes は、既存のデータベースのディスク上のジャーナルモードが、開く際に暗黙のうちに WAL に
切り替えられた場合（たとえばオペレーターが手動で `delete` に変換していたデータベース）に、
（プロセスごと・データベースごとに 1 回）警告を出し、その選択を維持させる設定として
`database.journal_mode` を示します。
逆方向が自動的に行われることはありません: すでに WAL モードになっているデータベースは、
`journal_mode: delete` を設定してもライブでダウングレードされません（接続が開いている状態での
ダウングレードはデータベースを破損させる可能性があります）。`hermes doctor` は、プロファイルの
すべての Hermes プロセスを停止して `hermes sessions set-journal-mode delete` を実行するまで
`<db> is in WAL mode despite database.journal_mode=delete` と警告します（このコマンドは、
何かがまだファイルを保持している間は実行を拒否し、変換後のヘッダーを検証します）。その警告の下には、
現在データベースを保持しているプロセス（`<db> is held by PID <n> (<command>)`）が表示されるため、
何を停止すべきかがわかります。保持プロセスのスキャンが部分的または利用できない場合は、問題なしとはせず
`cannot prove the database is quiet` と表示します。

## 環境変数の置換

`config.yaml` 内では `${VAR_NAME}` 構文を使って環境変数を参照できます。

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

1 つの値の中で複数の参照を使うこともできます: `url: "${HOST}:${PORT}"`。参照先の変数が設定されていない場合、プレースホルダーはそのまま保持され（`${UNDEFINED_VAR}` はそのまま残ります）、警告がログに記録されます。裸の `$VAR` は展開されません。

[多重化されたマルチプロファイルゲートウェイ](/docs/user-guide/multi-profile-gateways)の下では、プロファイルの `config.yaml` 内の参照は、共有のプロセス環境ではなく**そのプロファイルの** `.env`（そのシークレットスコープ）に対して解決されます — プロファイル B の `${MATRIX_ACCESS_TOKEN}` は、B 自身がその変数を定義していない限り未解決のまま（そのまま保持され、警告がログに記録されます）です。これは、多重化の中で B の設定が読み込まれるあらゆる場面に当てはまります: ルーティングされたゲートウェイのターン、B のアダプターの起動、B の cron ジョブです。単一プロファイルでの実行は変わりません。完全な一覧は[プロファイルごとに分離されるもの](/docs/user-guide/multi-profile-gateways#what-is-isolated-per-profile)を参照してください。

ゲートウェイが提供するローカルの `hermes chat` / TUI セッションは、これらの参照、`provider_routing`、`prefill_messages_file` を、ゲートウェイがセッションを作成する時点で、そのプロファイルの現在の `config.yaml` と `.env` から解決します。これらへの編集はゲートウェイを再起動しなくても次の新しいセッションに適用され、開いているセッションは作成時の値を保持します。ゲートウェイを再起動してもこれらは再読み込みされません: 再起動後に復元されたセッションは、自身とともに保存されたルーティングと prefill パスを保持し、prefill ファイルの内容は、初めて必要になったときにその保存済みパスから改めて読み込みます。ゲートウェイの起動に使われたプロファイルでは、`.env` から削除した変数も、ゲートウェイ起動時に読み込まれた値に解決され続けます。見つからない場合はプロセス環境にフォールバックするためです。それを取り除くにはゲートウェイを再起動してください。

Cursor スタイルの SecretRef 構文も使用できます: `${env:VAR_NAME}` は `${VAR_NAME}` とまったく同じように解決されます（`env:` プレフィックスは取り除かれます）。そのため、Cursor / Claude の設定からコピーした MCP やプロバイダーのスニペットは、`config.yaml` と `mcp_servers` ブロックの両方でそのまま機能します。その他の SecretRef ソース（`${file:...}`、`${vault:...}`、`${bitwarden:...}`）はインラインでは**解決されません** — 外部のシークレットバックエンドは `secrets:` ブロックを通じて起動時にその値を環境に注入するため、代わりに `${env:NAME}` として参照してください。未知のプレフィックスは 1 回だけ警告され、そのまま保持されます。

AI プロバイダーのセットアップ（OpenRouter、Anthropic、カスタムエンドポイント、セルフホスト LLM、フォールバックモデルなど）については、[AI プロバイダー](../integrations/providers.md)を参照してください。

### プロバイダーのタイムアウト

プロバイダー全体のリクエストタイムアウトとして `providers.<id>.request_timeout_seconds` を、加えてモデル固有のオーバーライドとして `providers.<id>.models.<model>.timeout_seconds` を設定できます。これはすべてのトランスポート（OpenAI ワイヤー、ネイティブ Anthropic、Anthropic 互換）におけるプライマリのターンクライアント、フォールバックチェーン、認証情報ローテーション後の再構築、そして（OpenAI ワイヤーの場合）リクエストごとのタイムアウト kwarg に適用されます。そのため、設定した値はレガシーの `HERMES_API_TIMEOUT` 環境変数より優先されます。

非ストリーミングのスタールコール検出器として `providers.<id>.stale_timeout_seconds` を、加えてモデル固有のオーバーライドとして `providers.<id>.models.<model>.stale_timeout_seconds` を設定することもできます。これはレガシーの `HERMES_API_CALL_STALE_TIMEOUT` 環境変数より優先されます。同じキーはストリーミングのスタールストリームの期限でもあります: 明示的な値はそのまま使われます — 暗黙のコンテキストサイズ段階（50k トークン超で 240 秒、100k 超で 300 秒）と推論モデルの下限は 180 秒のデフォルトにのみ適用されるため、明示的な値を設定すると、ハングしたストリームを許容する時間を短くできます。

これらを未設定のままにすると、レガシーのデフォルト値（`HERMES_API_TIMEOUT=1800` 秒、`HERMES_API_CALL_STALE_TIMEOUT=90` 秒、ネイティブ Anthropic は 900 秒）が維持されます。非ストリーミングのスタール検出器は、暗黙のままの場合ローカルエンドポイントでは自動的に無効化され、非常に大きなコンテキストでは上方向にスケールすることがあります。AWS Bedrock（`providers.bedrock`）では、解決されたリクエストタイムアウトは boto3 の Converse クライアント（`bedrock_converse`）と AnthropicBedrock SDK クライアントのソケット読み取りタイムアウトになり、アクティブなモデルに応じてリクエストごとに選ばれるため、`/model` による切り替えやフォールバックではそのモデルの値が使われます。接続タイムアウトは固定のままで（Converse では botocore の 60 秒、AnthropicBedrock では 10 秒）、読み取りタイムアウトは呼び出し全体ではなく各ソケット読み取りを制限します。未設定の場合、Converse では botocore の 60 秒、AnthropicBedrock では 900 秒の読み取りタイムアウトが維持されます。[`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example) のコメント付きの例を参照してください。

## アップデートの動作

### バックグラウンドチェック

受動的なアップデートチェック（CLI のバナー、TUI のバッジ、ダッシュボード、デスクトップアプリ）は、
GitHub REST API に `main` の先端を問い合わせ、それがあなたのチェックアウトと異なる場合は、
正確な件数と変更履歴を compare エンドポイントに問い合わせます。これらが
`git fetch` を実行することはありません — 部分的な（`--filter=blob:none`）クローンでも、
promisor リモートから欠けているオブジェクトをダウンロードすることはありません（Git 2.44 以降）— また
各インストールが問い合わせるのは最大で **24 時間に 1 回**です（チェックが失敗した場合は
1 時間後に再試行します）。アップデートの適用（`hermes update`、またはデスクトップの
Update ボタン）は常に新たにフェッチし、キャッシュされた回答を無効化します。明示的な
チェック — `hermes update --check`、デスクトップの「Check for Updates…」メニュー項目、
Settings → About → 「Check now」— はキャッシュをバイパスします。

### SSH 認証

起動時のアップデートチェックは、ネットワーク呼び出しに使うのと同じ分離された Git 設定で
origin URL を読み取ります。そのため、グローバルな `url.*.insteadOf` の書き換えによって、
公式の SSH リモートが公開 HTTPS パスから隠されることはありません。

Hermes の分離された内部 Git コマンドは、デフォルトで `ssh -o BatchMode=yes` を使用します:
未知のホストキー、パスワード、パスフレーズが必要な暗号化キーは、ターミナルのプロンプトを開く
代わりに失敗します。使用可能なキーや SSH エージェントを持つ信頼済みホストは、引き続き認証されます。
これはディスク上の Git や SSH の設定、あるいはターミナルツールで実行するコマンドを変更するものではありません。

この内部のデフォルトは、リポジトリの `core.sshCommand` 設定をオーバーライドします。明示的な
`GIT_SSH_COMMAND` 環境変数は引き続き優先されるため、カスタムの ID やトランスポートのコマンドは
そこで維持できます。非対話のままにする必要がある場合は、そのようなオーバーライドに
`-o BatchMode=yes` を含めてください。プロンプトを許可するオーバーライドは、依然として
バックグラウンドチェックを中断させる可能性があります。

`hermes update` の設定は、`config.yaml` の `updates` の下にあります。

```yaml
updates:
  pre_update_backup: quick       # quick (state snapshot, default) | full (snapshot + HERMES_HOME zip) | off
  backup_keep: 5                 # Keep this many full pre-update backup zips
  non_interactive_local_changes: stash  # stash | discard
  auto_switch_parked_branch: true       # auto-switch a clean, fully merged parked branch back to main
```

`pre_update_backup` は、アップデート前の安全性に関する唯一のつまみです: `quick`（デフォルト）は重要な状態ファイル（ペアリングデータ、cron ジョブ、設定、認証。1 GiB を超えるファイルはスキップ）を `state-snapshots/` にスナップショットします。`full` はさらに `HERMES_HOME` 全体を `backups/` に zip 化し、大きなホームでは数分かかることがあります。`off` は両方を無効にします。レガシーの真偽値も尊重されます（`true` → `full`、`false` → `off`）。

`config.yaml` 自体の時点コピー（`hermes setup` が書き換える前、`hermes migrate` が編集する前、ファイルの解析が成功するたび、そして解析に失敗したとき）は、`backups/config/config.yaml.<reason>.<timestamp>` に保存されます。同一の繰り返しはスキップされ、理由ごとに最新の 5 つだけが保持されるため、`config.yaml` の横に溜まっていくことはありません。`config.yaml` が壊れている場合、Hermes は組み込みのデフォルト値ではなく最新の `good` コピーを使用し、YAML が修正されるまで起動のたびに警告します。壊れたファイルが変更されることはありません。

git インストールでは、Hermes はアップデートブランチのチェックアウトやプルを行う前に、変更のある追跡済みファイルと、アップデートが追加するパスにある未追跡ファイルを自動的に stash します。アップデートが書き込まない未追跡ファイルはそのまま残ります。対話的なターミナルでのアップデートは、何かを stash する前に変更を保持するかどうかを尋ねます。いいえと答えると、stash エントリを残す代わりに変更を削除します。非対話のアップデート（デスクトップ/チャットアプリ、ゲートウェイ、または `--yes`）は `updates.non_interactive_local_changes` を使用します: `stash` はプルの成功後にローカルのソース編集を復元し、`discard` はプルの成功後にアップデートが作成した stash を破棄します。`discard` は、ローカルのソース編集を永続させるつもりがまったくない管理されたインストールでのみ使用してください。


その stash の手順の前に、Hermes は、npm のインストールによって WhatsApp ブリッジと Photon サイドカーに残される、追跡済みの `package-lock.json` の差分も復元します。意図的なロックファイルの編集は、アップデートの前にコミットするか手動で stash してください。

## ターミナルバックエンドの設定

Hermes には 6 種類のターミナルバックエンドが付属しています。それぞれが、エージェントのシェルコマンドが実際にどこで実行されるかを決定します — ローカルマシン、Docker コンテナ、SSH 経由のリモートサーバー、Modal クラウドサンドボックス（直接接続、または Nous 管理のゲートウェイ経由）、Daytona ワークスペース、または Vercel Sandbox です。プラグインによって、公式の [Singularity/Apptainer](#singularityapptainer-backend) や [Sprites](#sprites-backend) バックエンドなど、さらに追加できます。

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | vercel_sandbox | <plugin backend>
  cwd: "."          # Gateway/cron working directory (CLI always uses launch dir)
  temp_dir: ""      # Session temp root; empty = TMPDIR, else ~/.hermes/cache/terminal
  font_family: ""   # Desktop terminal font; e.g. "MesloLGS NF"
  timeout: 180      # Per-command timeout in seconds
  home_mode: auto   # auto | real | profile — subprocess HOME policy
  env_passthrough: []  # Env var names to forward to sandboxed execution (terminal + execute_code)
  client_env_passthrough: []  # Extra non-secret toolchain names (e.g. GEM_HOME) a local CLI/TUI session's tools take from its launching shell
  sync_back_max_bytes: 2147483648  # Remote backends: refuse to extract a state archive larger than this (bytes)
  modal_image: "nousresearch/hermes-sandbox:desktop"                 # Container image for Modal backend
  daytona_image: "nousresearch/hermes-sandbox:desktop"               # Container image for Daytona backend
```

`terminal.temp_dir` は、local バックエンドで Hermes がセッションの一時成果物 —
バックグラウンドプロセスのログ/pid/終了ファイル、コード実行の
サンドボックス、退避されたツール結果 — を置く場所を制御します。空の場合（デフォルト）、Hermes は
環境の明示的な `TMPDIR`/`TMP`/`TEMP` を尊重し、それ以外の場合は
`/tmp` の代わりに実ストレージ上の管理ディレクトリ `~/.hermes/cache/terminal` を使用します — 多くのディストリビューション（特に Arch ベースの環境）では `/tmp` <!-- no-tmp: ok — explains why /tmp is avoided -->
は小さな RAM ベースの tmpfs であり、負荷がかかると Hermes のセッション成果物で
いっぱいになることがあります。管理ディレクトリは自動的に整理されます: 24 時間アイドル状態の成果物（内部の
どこにも書き込みがないもの）は、ゲートウェイのハウスキーピングによって 1 時間ごとに、CLI のみのインストールでは
プロセスごとに 1 回掃除されます。セッションの一時領域を別の場所にリダイレクトするには、`temp_dir` を既存の絶対パスに
設定します。ユーザーが設定したパスが自動的に整理されることはありません。

`terminal.temp_dir` とは独立して、すべての Hermes プロセス（CLI、TUI、ゲートウェイ、Desktop
バックエンド、cron）とそれが起動するすべての子プロセスは、起動時に `TMPDIR`、`TMP`、`TEMP` が
**`~/.hermes/cache/scratch`**（プロファイルごと）を指すように設定されます。そのため `tempfile.mkdtemp()`、
`mktemp`、ブラウザプロファイル、プローブスクリプトはすべて、RAM ベースのシステム一時ディレクトリではなく
実ストレージ上に置かれます。システムプロンプトはこのディレクトリをスクラッチディレクトリとして示します。
Hermes がこれらを設定するのは、まだ設定されていない場合のみです — あなたや OS がエクスポートした `TMPDIR`
（macOS の `/var/folders`、Windows の `%TEMP%`）はそのまま残されます。エントリは、起動直後に開始される
バックグラウンドプロセスによって整理されます（最大で 1 時間に 1 回。起動がそれを待つことはなく、コマンドが
すぐに終了してもこの処理は最後まで完了します）。対象は **24 時間アイドル状態**になったエントリです: エントリは、
その内部のどこかが直近 1 日以内に書き込まれている限り残り、最後の書き込みから 1 日後に削除されます。
ハードリンクされたファイルは書き込みとして数えません（`git clone --local` はソースリポジトリとパックファイルを
共有するため、そうしないとソース側のアクティビティによってクローンが永遠に残ってしまいます）。アイドル状態の
エントリを削除する前に、Hermes は、作業ディレクトリがそのエントリ内（またはテスト実行が残したヘッドレスブラウザの
ように、もう存在しないスクラッチパス内）にあるまま実行中のプロセスも停止し、そこを指していた `git worktree`
の登録も削除します。Hermes 自身が起動してまだ追跡しているプロセス（バックグラウンドの `terminal` プロセス、
ゲートウェイ、CLI セッション、またはそれらが起動してまだ子プロセスのままのもの）が停止されることはなく、
Hermes コマンドを起動したシェル（またはその親）も、そのコマンドが終了した後であっても停止されません:
それらのいずれかが内部で作業しているアイドルエントリは、代わりに保持されます。プロセスを検査できない処理は
何も停止せず、プロセスが作業している可能性のあるエントリを削除しません。各エントリは削除の直前に
再確認されるため、整理処理がプロセスを停止している間に書き込まれたものは保持されます。すべての削除、
停止したプロセス、保持したエントリは `~/.hermes/logs/scratch-prune.log` に記録されます。あなたのユーザーが
読み取りや削除をできないファイルを含むエントリ（典型的には、bind マウントに書き込んだ Docker コンテナの
root 所有の出力）が、中身を把握できないまま部分的に削除されることはありません: それを削除するコマンド
（`sudo rm -rf '<path>'`、または Docker がインストールされている場合は `docker run … rm -rf` の代替）とともに
1 回だけログに記録され、その後はスキップされます。それがなくなるか、最上位ディレクトリが変わるか、
アクセスを拒否したディレクトリにあなたがアクセスできるようになる（たとえば `sudo chown -R` の後）まで、
1 時間ごとに再スキャンされることはありません。それとは関係なく週に 1 回は再確認されます。Hermes が
そのコマンドを自分で実行することはありません。`hermes doctor` は、
このディレクトリ、そのサイズ、そのようなスキップされたエントリを報告し、
どの整理処理の対象にもなっていない `cache/` 配下の他の場所にある 1 GB 超のディレクトリについて警告します。

`desktop.font_family` は、チャットおよび Hermes Desktop インターフェイスのその他の部分のフォントを設定します（ターミナルペインには上記の独自のキーがあります）。インストール済みのファミリー名を 1 つ（たとえば `OpenDyslexic` や `Atkinson Hyperlegible`）、または CSS のフォントスタックを指定します。Hermes はアクティブなテーマ自身のスタックをその後ろに保持するため、CJK や絵文字のグリフも引き続き解決されます。空の値の場合はテーマのフォントが使われます。**Settings → Appearance → Chat Font** で編集できます。

`terminal.font_family` は、Hermes Desktop の組み込みターミナルを制御します。ローカルにインストールされたファミリー名を 1 つ（たとえば `MesloLGS NF`）、または CSS のフォントスタックを受け付けます。Hermes はフォールバックとして同梱の JetBrains Mono スタックを追加し、空の値ではデフォルトが維持されます。同じプロファイル単位の設定は **Settings → Appearance → Terminal Font** でも編集できます。Google Fonts のダウンロードやシステムフォントの権限は不要です。

Modal、Daytona、Vercel Sandbox、Sprites などのクラウドサンドボックスでは、`container_persistent: true` を指定すると、Hermes はサンドボックスの再作成をまたいでファイルシステムの状態を保持しようとします。ただし、同じ稼働中のサンドボックス、PID 空間、またはバックグラウンドプロセスが後で実行され続けていることを保証するものではありません。

### バックエンドの概要

| バックエンド | コマンドの実行場所 | 分離 | 適した用途 |
|---------|-------------------|-----------|----------|
| **local** | あなたのマシン上で直接 | なし | 開発、個人利用 |
| **docker** | 単一の永続的な Docker コンテナ（セッション、`/new`、サブエージェントで共有） | 完全（namespace、cap-drop） | 安全なサンドボックス化、CI/CD |
| **ssh** | SSH 経由のリモートサーバー | ネットワーク境界 | リモート開発、高性能ハードウェア |
| **modal** | Modal クラウドサンドボックス | 完全（クラウド VM） | 一時的なクラウド計算、評価 |
| **daytona** | Daytona ワークスペース | 完全（クラウドコンテナ） | マネージドなクラウド開発環境 |
| **vercel_sandbox** | Vercel Sandbox | 完全（クラウド microVM） | スナップショットによるファイルシステム永続化を備えたクラウド実行 |
| **singularity**（プラグイン） | Singularity/Apptainer コンテナ | namespace（--containall） | HPC クラスタ、共有マシン |
| **sprites**（プラグイン） | Fly.io 上の Sprite | 完全（クラウド microVM） | セッション間で永続するステートフルなクラウドサンドボックス |

### Local バックエンド

デフォルトです。コマンドは分離なしであなたのマシン上で直接実行されます。特別なセットアップは不要です。

```yaml
terminal:
  backend: local
```

デフォルトでは、ローカルのツールサブプロセスはあなたの実際の OS ユーザーの `HOME` を保持します。これにより、
`git`、`ssh`、`gh`、`az`、`npm`、Claude Code、Codex などの外部 CLI は、通常のシェルですでに使っている
認証情報や設定を見つけられます。Hermes の状態は引き続き `HERMES_HOME` によってプロファイル単位に
分けられます。プロファイルが設定、メモリ、セッション、スキルを選択する手段は `HOME` ではありません。

Hermes は、システム全体の `HOME`、シェルの起動ファイル、オペレーティングシステムのアカウントのホームを
変更**しません**。この設定は、Hermes が `terminal`、バックグラウンドのターミナルプロセス、`execute_code`、
ACP ヘルパープロセスなどのツールを通じて起動するサブプロセスに渡される環境のみを制御します。

#### `terminal.home_mode`

| モード | ホストへのインストール | コンテナ | トレードオフ |
|---|---|---|---|
| `auto` | 実際の OS ユーザーの `HOME` を保持 | `{HERMES_HOME}/home` を使用 | 推奨のデフォルト。ホストの CLI は引き続き機能し、コンテナの状態は永続化されます。 |
| `real` | 実際の OS ユーザーの `HOME` を強制 | 見える場合は実際の OS ユーザーの `HOME` を強制 | 親プロセスが誤って `HOME` をプロファイルのホームに向けた状態で起動した場合に便利です。 |
| `profile` | 存在する場合は `{HERMES_HOME}/home` を使用 | 存在する場合は `{HERMES_HOME}/home` を使用 | プロファイルごとの CLI 設定を厳密に分離しますが、通常の `~/.ssh`、`~/.gitconfig`、`~/.azure`、`~/.config/gh`、Claude/Codex の認証、npm の状態などは、プロファイルのホーム内で初期化するかリンクしない限り見えません。 |

デフォルトの欠点は、ホスト上のプロファイルが `~` 配下の同じ通常のユーザーレベルの CLI 認証情報/設定を
共有することです。別の git ID、SSH キー、GitHub CLI のログイン、npm 設定、クラウド CLI のログインを
持つプロファイルが必要な場合は、`home_mode: profile` を使用し、そのプロファイルのホーム内でそれらの
ツールを意図的に初期化してください。

意図的にプロファイルごとのツール設定を厳密に分離したい場合は、次のように設定します。

```yaml
terminal:
  home_mode: profile
```

このモードでは、ツールのサブプロセスは `{HERMES_HOME}/home` を `HOME` として使用します。Hermes は
`HERMES_REAL_HOME` も設定するため、スクリプトは必要なときに実際のユーザーホームを引き続き特定できます。
コンテナバックエンドは、`auto` モードでも `{HERMES_HOME}/home` を使い続けます。このディレクトリは
永続的な Hermes データボリューム上にあるためです。

プロファイルの状態と実際のユーザーホームを区別する必要があるスクリプトは、Hermes のデータには
`HERMES_HOME` を、アカウントのホームには `HERMES_REAL_HOME` を使うことを推奨します。

```python
from pathlib import Path
import os

hermes_home = Path(os.environ["HERMES_HOME"])
real_home = Path(os.environ.get("HERMES_REAL_HOME", os.environ["HOME"]))
```

:::warning
エージェントはあなたのユーザーアカウントと同じファイルシステムアクセス権を持ちます。使いたくないツールは `hermes tools` で無効化するか、サンドボックス化のために Docker へ切り替えてください。
:::

### Docker バックエンド {#docker-backend}

セキュリティハードニング（すべてのケーパビリティを削除、特権昇格なし、PID 制限）を施した Docker コンテナ内でコマンドを実行します。

**Hermes プロセス間で共有される永続コンテナ。** `container_persistent: true` かつ `docker_mount_cwd_to_workspace: false`（デフォルト）の場合、同じプロファイル内のセッションはそのプロファイル本来のサンドボックスを共有します。このモードでは、Desktop で作業ディレクトリを選択しても別のワークスペースサンドボックスは作成されないため、CLI と Desktop は同じファイルにアクセスし続けます。ホストのワークスペースのマウントを有効にすると、永続コンテナは代わりにプロファイルと選択したワークスペースの組み合わせで識別されます: 同じワークスペースのセッションはコンテナを共有し、異なるワークスペースは分離されたままです。`delegate_task` のサブエージェントは親のコンテナを使用します。

以前のビルドでは、マウントがオフでも Desktop のフォルダごとに独自のサンドボックスが与えられていました。それらのディレクトリ（`TERMINAL_SANDBOX_DIR/docker` 配下の `*_ws_*`。デフォルトはプロファイルの Hermes ホームの `sandboxes/docker`）は保持されますが、ターミナルやファイルパネルではもう使われません: まだ必要なものは、プロファイルのサンドボックスの `workspace` または `home` にコピーしてください。

名前付きプロファイルでは、そのゲートウェイのチャット、Desktop セッション、cron ジョブ、Kanban ワーカーはすべて、プロファイルの 1 つのサンドボックス（`TERMINAL_SANDBOX_DIR/docker` 配下の `profile_<name>-*`）を使用します。以前のビルドでは、そのプロファイルとして起動したゲートウェイ（`hermes -p <name> gateway`）とその cron および Kanban の作業を、プロファイルのホーム配下の別の `default` サンドボックスに保持していました。そのディレクトリは保持されますが、もう使われません: まだ必要なものは同じ方法でコピーしてください。

ターミナル、ファイル、`execute_code` の呼び出しは、選択されたコンテナ内で `docker exec` を使用します。インストールしたパッケージ、`/workspace` 内のファイル、**バックグラウンドプロセス**は、そのコンテナを使用する呼び出しや Hermes プロセスの間で引き継がれます。TUI セッションを閉じる、`/quit` を実行する、新しい `hermes` を起動する、といった操作自体でコンテナが削除されることはありません。次のプロセスはラベルを通じてそれを再利用できます。正確なティアダウンのルールは、後述の **コンテナのライフサイクル** を参照してください。

**セッションごとの分離モード（`container_persistent: false`）。** Docker バックエンドで `container_persistent: false` を設定すると、**セッションごと**に 1 つのコンテナを使うモードに切り替わります: すべてのチャット（デスクトップアプリのセッション、ゲートウェイの会話、TUI セッション）は独自の新しいサンドボックスを取得します。これは最初のターミナル/ファイル呼び出しで作成され、セッションが閉じられるか `lifetime_seconds` を超えてアイドル状態になると削除されます。セッション間では何も引き継がれません — ファイルシステムの状態も、マウントも、バックグラウンドプロセスもです。`docker_mount_cwd_to_workspace: true` の場合、`/workspace` にマウントされるのは**そのセッションに紐付いた**ワークスペースだけです。ディレクトリが紐付いていない新しいセッションは、前のセッションのマウントを引き継ぐのではなく空のワークスペースを取得します。`delegate_task` のサブエージェントは引き続き親セッションのコンテナを共有します。サンドボックスを会話間のセキュリティ境界にしたい場合はこのモードを使用し、上述の長寿命の共有コンテナが必要な場合はデフォルトの `true` のままにしてください。

```yaml
terminal:
  backend: docker
  # Default: nikolaik/python-nodejs (Python 3.13 / Node 26) plus a display stack, so Bot Screen,
  # computer_use and the browser run INSIDE this sandbox (Bot Screen → "Where the screen runs").
  # Any other image works for shell work; the screen then needs bot_desktop.placement: gateway.
  # Writing this key is a decision: a persisted container on another image is recreated on the next
  # terminal call. Left unset, an existing container is kept and the CLI / Screen pane ask first.
  docker_image: "nousresearch/hermes-sandbox:desktop"
  docker_mount_cwd_to_workspace: false  # Mount launch dir into /workspace
  docker_run_as_host_user: false   # See "Running container as host user" below
  docker_snap_compat: false        # See "Snap-packaged Docker (AppArmor)" below
  docker_forward_env:              # Host env vars to forward into container
    - "GITHUB_TOKEN"
  docker_env:                      # Literal env vars to inject (KEY=value)
    DEBUG: "1"
    PYTHONUNBUFFERED: "1"
  docker_volumes:                  # Host directory mounts
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro for read-only
  docker_extra_args:               # Extra flags appended verbatim to `docker run`
    - "--gpus=all"
    - "--network=host"
  docker_network: true             # false = air-gap the container (--network=none)

  # Resource limits
  container_cpu: 1                 # CPU cores (0 = unlimited)
  container_memory: 5120           # MB (0 = unlimited)
  container_disk: 51200            # MB (requires overlay2 on XFS+pquota)
  container_persistent: true       # true = persist /workspace + /root, shared container; false = fresh container per session (see below)

  # Cross-process container reuse (defaults match the "one long-lived
  # container shared across sessions" contract — see Container lifecycle).
  docker_persist_across_processes: true   # Reuse container across Hermes restarts
  docker_shared_container_key: ""         # Opt in trusted profiles to one identity
  docker_orphan_reaper: true              # Sweep abandoned Exited containers at startup

  # Cross-backend lifecycle settings (apply to docker as well)
  timeout: 180                     # Per-command timeout in seconds
  lifetime_seconds: 300            # Idle-reaper window; also feeds 2× orphan-reaper threshold
```

**`docker_env`** と **`docker_forward_env`** の違い: 前者は設定で指定したリテラルの `KEY=value` ペアを注入します（値は `config.yaml` に置くか、`TERMINAL_DOCKER_ENV='{"DEBUG":"1"}'` のように JSON の dict として渡します）。後者はシェルまたは `~/.hermes/.env` から値を転送するため、実際のシークレットが設定ファイルに現れることはありません。トークンには `docker_forward_env` を、コンテナが必要とする静的なつまみには `docker_env` を使用してください。

**`terminal.docker_extra_args`**（`TERMINAL_DOCKER_EXTRA_ARGS='["--gpus=all"]'` でもオーバーライド可能）を使うと、Hermes がファーストクラスのキーとして提供していない任意の `docker run` フラグ — `--gpus`、`--network`、`--add-host`、別の `--security-opt` オーバーライドなど — を渡せます。各エントリは文字列である必要があります。このリストは組み立てられた `docker run` 呼び出しの最後に追加されるため、必要に応じて Hermes のデフォルトをオーバーライドできます。使用は控えめにしてください — サンドボックスのハードニング（ケーパビリティの削除、`--user`、ワークスペースの bind マウント）と衝突するフラグは、暗黙のうちに分離を弱めます。

**`terminal.docker_network`**（デフォルト `true`、環境変数: `TERMINAL_DOCKER_NETWORK`）— `false` に設定すると、サンドボックスコンテナを `--network=none` で実行し、エージェントのコマンドからのすべてのネットワーク送信を遮断します。これは `terminal`、`execute_code`、ファイルツールが使う実行コンテナに適用されます。コンテナは Hermes プロセスをまたいで永続化されるため、ネットワーク接続された古いコンテナが存在する状態でこれを `false` に切り替えると、そのコンテナは削除され、新しいエアギャップされたコンテナが起動します（警告がログに記録されます）。その中で実行中のバックグラウンドプロセスは失われます。`docker_extra_args` で `--network=none` を渡すよりも、このキーを使うことを推奨します。

**要件:** Docker Desktop または Docker Engine がインストールされ、稼働していること。Hermes は `$PATH` に加えて、一般的な macOS のインストール場所（`/usr/local/bin/docker`、`/opt/homebrew/bin/docker`、Docker Desktop のアプリバンドル）を探します。Podman は標準でサポートされています。両方がインストールされている場合に Podman を強制するには、`HERMES_DOCKER_BINARY=podman`（またはフルパス）を設定してください。

#### コンテナのライフサイクル

Hermes が管理するコンテナにはラベルが付いており、後続のプロセス（およびオーファンリーパー）がそれらを識別できます。

- `hermes-agent=1` — Hermes 管理であることを示す
- `hermes-task-id=<sanitized task_id>` — タスクごとの再利用プローブのキー
- `hermes-profile=<sanitized profile name>` — デフォルトでは、再利用と回収をアクティブな Hermes プロファイルに限定する。`docker_shared_container_key` が設定されている場合は、代わりにそのサニタイズ済みの値が使われる
- `hermes-environment=<digest>` — 明示的な共有キーのないコンテナについて、再利用を要求されたイメージ、マウント引数、アクティブな `HERMES_HOME` に限定する。ホストのパスとボリュームのソースは、このラベルで露出されるのではなくハッシュ化される。プロセスごとの一時ディレクトリを元にしたマウント（シンボリックリンクに安全なスキルのコピー）は、ランダムなホスト一時ディレクトリではなく安定したコンテナパスでハッシュ化されるため、再利用を妨げない

起動時、Hermes は `docker ps --filter label=hermes-task-id=<id> --filter label=hermes-profile=<identity>` を実行し、見つかった場合は**既存のコンテナにアタッチ**します。identity は、`docker_shared_container_key` によって信頼済みプロファイルを明示的に共通の値にオプトインしていない限り、アクティブなプロファイルです。コンテナが `exited` の場合（例: Docker デーモンの再起動後）は、`docker start` されて再利用されます — ファイルシステムの状態とインストール済みのパッケージは残りますが、コンテナ内のバックグラウンドプロセスは残りません。

明示的な共有キーがない場合、再利用プローブは一致する `hermes-environment` ラベルも要求します。イメージ、マウント引数、`HERMES_HOME` を変更すると、別の設定を暗黙のうちに使うのではなく、新しいコンテナが起動します。このラベルが導入される前に作成されたコンテナは一致しないため、アップグレード後の最初のセッションでは新しいコンテナが起動します。既存の実行中のコンテナは削除されません。

Hermes プロセスが終了したとき — `/quit`、TUI セッションのクローズ、ゲートウェイのシャットダウン、さらには SIGKILL であっても — クリーンアップの経路は**デフォルトモードではコンテナに対して何もしません**。コンテナは実行され続けます。次の Hermes プロセスは、ラベルのプローブを通じてミリ秒単位でそれにアタッチします。これは「セッション間で共有される 1 つの長寿命コンテナ」という契約が要求する動作です: バックグラウンドプロセス（npm のウォッチャー、開発サーバー、長時間実行の pytest）がセッションをまたいで生き残る唯一の方法です。

**コンテナがティアダウンされる（停止して `docker rm -f` される）のは、次の場合のみです:**

| トリガー | 発火するタイミング |
|---|---|
| `docker_persist_across_processes: false` | 明示的なプロセスごとの分離。すべての `cleanup()` が `stop` + `rm -f` を行います。issue #20561 以前の動作と一致します。 |
| アイドルリーパー（`lifetime_seconds`、デフォルト 300 秒） | 環境が `persist_across_processes=false` の場合のみ。永続モードの環境では何もせず、コンテナはアイドルスイープを生き延びます。 |
| 次回起動時のオーファンリーパー | `2 × lifetime_seconds`（デフォルト 600 秒 = 10 分）より古い、hermes ラベル付きの **Exited** コンテナを、現在のプロファイルに限定して掃除します。**実行中のコンテナには決して触れません** — 兄弟プロセスの安全のためです。無効にするには `docker_orphan_reaper: false` を設定します。 |
| ユーザーによる直接の操作 | `docker rm -f`、`docker system prune`、Docker Desktop の再起動。`--restart=always` は設定していないため、ホストを再起動するとコンテナは `Exited` のままになります（その CoW レイヤーは残り、次回起動時に再利用されますが、バックグラウンドプロセスは失われます）。 |

知っておく価値のあるエッジケース:

- **コンテナ内の PID 1 の OOM キル**によって、コンテナは `Exited` に遷移します。次の再利用時に `docker start` されます。ファイルシステムの状態は残りますが、バックグラウンドプロセスは残りません。
- **プロファイルの切り替え**によって、コンテナは互いに分離されます — `hermes-profile=work` とラベル付けされたコンテナは、`hermes-profile=research` で実行されている Hermes プロセスからは見えません。オーファンリーパーもプロファイル単位であるため、他のプロファイルのコンテナが誤って回収されることはありませんが、元のプロファイルで Hermes を再び起動するまで自動的にクリーンアップされることもありません。
- **明示的なプロファイル間の共有** — 1 つの信頼されたワークスペースで意図的に協働するプロファイルには、`terminal:` の下に同じ空でない `docker_shared_container_key` を設定します。これはコンテナの identity ラベルだけを置き換えます。タスク、送信、ネットワークの互換性チェックは引き続き適用されます。キーのないプロファイルは分離されたままです。identity ラベルはキーから短いダイジェストのサフィックス付きで導出されるため、似た見た目のキー（`team/workspace` と `team_workspace`）が 1 つのコンテナに衝突することはありません。**重要: 共有コンテナは、最初にそれを起動したプロファイルによって 1 回だけ作成されます** — そのプロファイルの `docker_image`、ボリューム、shm サイズ、その他の変更不可能な Docker 設定が採用され、後続のプロファイルはそのままアタッチします。それらの設定ファイルで異なる設定は、コンテナが削除されて再作成されるまで無視されます。キーを共有するプロファイルは、イメージとマウントについて合意しておくべきです。

`delegate_task(tasks=[...])` で生成された並列サブエージェントは、この 1 つのコンテナを共有します — 同時に行われる `cd`、環境変数の変更、同じパスへの書き込みは衝突します。サブエージェントが分離されたサンドボックスを必要とする場合は、`register_task_env_overrides()` を介してタスクごとのイメージオーバーライドを登録する必要があります。RL やベンチマーク環境（TerminalBench2、HermesSweEnv など）は、タスクごとの Docker イメージに対してこれを自動的に行います。

**セキュリティハードニング:**
- `--cap-drop ALL` に加え、`DAC_OVERRIDE`、`CHOWN`、`FOWNER` のみを戻す
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- `/tmp`（512MB）、`/var/tmp`（256MB）、`/run`（64MB）に対するサイズ制限付き tmpfs <!-- no-tmp: ok — documents the sandbox's own tmpfs -->

**認証情報の転送:** `docker_forward_env` に列挙された環境変数は、まずシェル環境から、次に `~/.hermes/.env` から解決されます。スキルは `required_environment_variables` を宣言することもでき、それらは自動的にマージされます。

#### 環境変数によるオーバーライド

`terminal:` 配下のすべてのキーには、`TERMINAL_<KEY_UPPERCASE>` という形式の環境変数オーバーライドがあります。Docker バックエンドで特に便利なものは次のとおりです。

| 環境変数 | 対応するキー | 備考 |
|---|---|---|
| `TERMINAL_DOCKER_IMAGE` | `docker_image` | ベースイメージ |
| `TERMINAL_DOCKER_FORWARD_ENV` | `docker_forward_env` | JSON 配列: `'["GITHUB_TOKEN","OPENAI_API_KEY"]'` |
| `TERMINAL_DOCKER_ENV` | `docker_env` | JSON の dict: `'{"DEBUG":"1"}'` |
| `TERMINAL_DOCKER_VOLUMES` | `docker_volumes` | `"host:container[:ro]"` 文字列の JSON 配列 |
| `TERMINAL_DOCKER_EXTRA_ARGS` | `docker_extra_args` | JSON 配列 |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | `docker_mount_cwd_to_workspace` | `true` / `false` |
| `TERMINAL_DOCKER_RUN_AS_HOST_USER` | `docker_run_as_host_user` | `true` / `false` |
| `TERMINAL_DOCKER_SNAP_COMPAT` | `docker_snap_compat` | `true` / `false` — デフォルト `false` |
| `TERMINAL_DOCKER_NETWORK` | `docker_network` | `true` / `false` — デフォルト `true`。`false` = `--network=none` |
| `TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES` | `docker_persist_across_processes` | `true` / `false` — デフォルト `true` |
| `TERMINAL_DOCKER_SHARED_CONTAINER_KEY` | `docker_shared_container_key` | 信頼済みプロファイル向けの明示的な共有 identity。デフォルトは空 |
| `TERMINAL_DOCKER_ORPHAN_REAPER` | `docker_orphan_reaper` | `true` / `false` — デフォルト `true` |
| `TERMINAL_CONTAINER_CPU` | `container_cpu` | CPU コア数 |
| `TERMINAL_CONTAINER_MEMORY` | `container_memory` | MB |
| `TERMINAL_CONTAINER_DISK` | `container_disk` | MB |
| `TERMINAL_CONTAINER_PERSISTENT` | `container_persistent` | `true` / `false` — bind マウントのワークスペースディレクトリを制御。`docker_persist_across_processes` とは別物 |
| `TERMINAL_LIFETIME_SECONDS` | `lifetime_seconds` | アイドルリーパーの時間枠 |
| `TERMINAL_TEMP_DIR` | `temp_dir` | セッションの一時ルート（local バックエンド） |
| `TERMINAL_TIMEOUT` | `timeout` | コマンドごとのタイムアウト |
| `HERMES_DOCKER_BINARY` | _なし_ | 特定の docker/podman バイナリのパスを強制 |

### SSH バックエンド {#ssh-backend}

SSH 経由でリモートサーバー上でコマンドを実行します。接続の再利用のために ControlMaster を使用します（5 分間のアイドルキープアライブ）。各コマンドは新しい `ssh … bash -c` です。作業ディレクトリは常に引き継がれ、エクスポートされた環境変数は `persistent_shell` が有効な間（デフォルト）引き継がれます。[永続シェル](#persistent-shell)を参照してください。

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # Carry exported env vars between commands (default: true)
```

**必須の環境変数:**

```bash
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=ubuntu
```

**オプション:**

| 変数 | デフォルト | 説明 |
|----------|---------|-------------|
| `TERMINAL_SSH_PORT` | `22` | SSH ポート |
| `TERMINAL_SSH_KEY` | （システムのデフォルト） | SSH 秘密鍵へのパス |
| `TERMINAL_SSH_PERSISTENT` | `terminal.persistent_shell` に従う（`true`） | エクスポートされた環境変数をコマンド間で引き継ぐ |

**仕組み:** 初期化時に `BatchMode=yes` と `StrictHostKeyChecking=accept-new` で接続し、リモートのログインシェルの環境を一度だけスナップショットファイルに取り込みます。各コマンドは、そのスナップショットを source し、追跡している作業ディレクトリに `cd` してからコマンドを実行する、新しい `bash -c` として実行されます。

**スキル / 設定の環境変数パススルー:** スキルが `required_environment_variables` で宣言した変数や、`terminal.env_passthrough` に列挙した変数は、OpenSSH の `SendEnv` で転送されます — 名前は `ssh` のコマンドラインに載り、値はクライアントの環境で運ばれ、リモートコマンドのテキストに含まれることはありません。リモートの `sshd` がそれらを受け入れる必要があります。サーバーの `/etc/ssh/sshd_config` に次を追加して sshd をリロードしてください。

```
AcceptEnv NEXTCLOUD_URL NEXTCLOUD_*      # or the names your skills need
```

一致する `AcceptEnv` がないと、サーバーは変数を暗黙のうちに破棄し、リモートシェルからは未設定に見えます。Hermes のプロバイダー認証情報（`OPENAI_API_KEY` など）は、列挙されていても決して転送されません。[環境変数のパススルー](security.md#environment-variable-passthrough)を参照してください。

### Modal バックエンド

[Modal](https://modal.com) クラウドサンドボックス内でコマンドを実行します。各タスクは、設定可能な CPU、メモリ、ディスクを備えた分離された VM を取得します。ファイルシステムはセッションをまたいでスナップショット/復元できます。

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB (5GB)
  container_disk: 51200            # MB (50GB)
  container_persistent: true       # Snapshot/restore filesystem
```

**必須:** `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` 環境変数、または `~/.modal.toml` 設定ファイルのいずれか。

**永続化:** 有効にすると、クリーンアップ時にサンドボックスのファイルシステムがスナップショットされ、次のセッションで復元されます。スナップショットは `~/.hermes/modal_snapshots.json` で追跡され、削除するまで保持されます（Hermes は Modal SDK の 30 日のスナップショット有効期限をオプトアウトしています）。これはファイルシステムの状態を保持するもので、稼働中のプロセス、PID 空間、バックグラウンドジョブは保持しません。

**認証情報ファイル:** `~/.hermes/`（OAuth トークンなど）から自動的にマウントされ、各コマンドの前に同期されます。

### Daytona バックエンド

[Daytona](https://daytona.io) のマネージドワークスペース内でコマンドを実行します。永続化のための stop/resume をサポートします。

```yaml
terminal:
  backend: daytona
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB → converted to GiB
  container_disk: 10240            # MB → converted to GiB (max 10 GiB)
  container_persistent: true       # Stop/resume instead of delete
```

**必須:** `DAYTONA_API_KEY` 環境変数。

**永続化:** 有効にすると、サンドボックスはクリーンアップ時に（削除されず）停止され、次のセッションで再開されます。サンドボックス名は `hermes-{task_id}` というパターンに従います。

**ディスク制限:** Daytona は最大 10 GiB を強制します。これを超えるリクエストは警告とともに上限に制限されます。

### Vercel Sandbox バックエンド

[Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) のクラウド microVM 内でコマンドを実行します。Hermes は通常のターミナルおよびファイルツールのインターフェイスを使用します。Vercel 固有のモデル向けツールはありません。

```yaml
terminal:
  backend: vercel_sandbox
  vercel_image: vercel/sandbox/universal:latest   # Vercel managed image or a VCR repository[:tag]
  cwd: /vercel/sandbox            # default workspace root
  container_persistent: true      # Snapshot/restore filesystem
  container_disk: 51200           # Shared default only; custom disk is unsupported
```

**必須のインストール:** オプションの SDK エクストラをインストールします。

```bash
python -c "import pm; pm.sync_venv(['vercel'], explicit=True)"
```

**必須の認証:** `VERCEL_TOKEN`、`VERCEL_PROJECT_ID`、`VERCEL_TEAM_ID` の 3 つすべてでアクセストークン認証を設定します。これは、Render、Railway、Docker などのホスト上でのデプロイや、通常の長時間稼働する Hermes プロセスでサポートされるセットアップです。

一度きりのローカル開発では、Hermes は短命の Vercel OIDC トークンも受け付けます。

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

リンクされた Vercel プロジェクトディレクトリからは、プロジェクト名を省略できます。

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

OIDC トークンは短命であり、ドキュメント化されたデプロイ経路としては使用すべきではありません。

**イメージ:** `terminal.vercel_image` は新しいサンドボックスのコンテナイメージを選択します: `vercel/sandbox/universal:latest`（デフォルト: Ubuntu、Node.js 24、Python 3.14）、`vercel/sandbox/node:26`、`vercel/sandbox/python:3.14` などの [Vercel マネージドイメージ](https://vercel.com/docs/sandbox/concepts/images)、またはプロジェクトの Vercel Container Registry のリポジトリ（名前だけの場合は `latest` に解決され、タグやダイジェストで固定できます）です。以前の `terminal.vercel_runtime` のプリセット（`node24`、`node22`、`python3.13`）は [Vercel によって非推奨になっています](https://vercel.com/docs/sandbox/concepts/runtimes)。固定されたランタイムは引き続き機能し、イメージをオーバーライドしますが、両者を組み合わせることはできません。スナップショットからの復元は独自のファイルシステムを持つため、どちらも送信しません。

**永続化:** `container_persistent: true` のとき、Hermes はクリーンアップ中にサンドボックスのファイルシステムをスナップショットし、同じタスクの後続のサンドボックスをそのスナップショットから復元します。スナップショットの内容には、サンドボックスにコピーされた Hermes 同期の認証情報、スキル、キャッシュファイルが含まれることがあります。これはファイルシステムの状態のみを保持するもので、稼働中のサンドボックスのアイデンティティ、PID 空間、シェルの状態、実行中のバックグラウンドプロセスは保持しません。

**バックグラウンドコマンド:** `terminal(background=true)` は Hermes の汎用的な非ローカルバックグラウンドプロセスフローを使用します。サンドボックスが稼働している間は、通常のプロセスツールを通じてプロセスの生成、ポーリング、待機、ログ表示、終了が行えます。Hermes は、クリーンアップや再起動の後に Vercel のネイティブなデタッチプロセス復旧を提供しません。

**ディスクサイズ設定:** 現時点で Vercel Sandbox は Hermes の `container_disk` リソースつまみをサポートしていません。`container_disk` は未設定のままにするか、共有のデフォルト `51200` のままにしてください。デフォルト以外の値は、暗黙のうちに無視されるのではなく、診断とバックエンド作成で失敗します。

### Singularity/Apptainer バックエンド {#singularityapptainer-backend}

[Singularity/Apptainer](https://apptainer.org) コンテナ内でコマンドを実行します。Docker が利用できない HPC クラスタや共有マシン向けに設計されています。これは公式の [`singularity` プラグイン](https://github.com/NousResearch/hermes-official-plugins/tree/main/singularity)です（以前はコアに組み込まれていました。すでにこれを使っているプロファイルには `hermes update` がインストールします）。

```bash
hermes plugins install singularity
```

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nousresearch/hermes-sandbox:desktop"
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB
  container_persistent: true       # Writable overlay persists across sessions
```

**要件:** `singularity` プラグインと、`$PATH` 上の `apptainer` または `singularity` バイナリ。

**イメージの扱い:** Docker URL（`docker://...`）は自動的に SIF ファイルへ変換され、キャッシュされます。既存の `.sif` ファイルはそのまま使用されます。

**スクラッチディレクトリ:** 次の順序で解決されます: `TERMINAL_SCRATCH_DIR` → `/scratch` が書き込み可能な場合は `/scratch/$USER/hermes-agent`（HPC の慣例） → `TERMINAL_SANDBOX_DIR/singularity`（デフォルト `~/.hermes/sandboxes/singularity`）。

**分離:** `--containall --no-home` を使用し、ホストのホームディレクトリをマウントせずに完全な namespace 分離を行います。

### Sprites バックエンド {#sprites-backend}

[Sprite](https://fly.io/sprites) 内でコマンドを実行します。Sprite は、チェックポイントと復元を備えた Fly.io 上のステートフルなクラウドサンドボックスです。これは [`sprites` プラグイン](https://github.com/NousResearch/hermes-plugin-sprites)です。インストールしてから選択してください。

```bash
hermes plugins install NousResearch/hermes-plugin-sprites --enable
hermes config set terminal.backend sprites
```

インストールでは、プラグインの Python 依存関係（`sprites-py`）を追加する前に確認されます。その後 Hermes を再起動してください。

```yaml
terminal:
  backend: sprites
  container_persistent: true       # Keep the Sprite between sessions (default)
```

**必須:** `~/.hermes/.env` の `SPRITES_TOKEN`。トークンは Fly.io ダッシュボードの組織の Sprites 設定で作成します。`sprite login` だけでは Hermes にトークンは渡されません。

**永続化:** 永続的な Sprite はセッション後も稼働し続け、次回は名前で再開されます。名前はプロファイル単位です: デフォルトプロファイルでは `hermes-{task_id}`、名前付きプロファイルでは `hermes-{profile}-{task_id}-{digest}`（収まるように短縮）です。Sprite 自身のファイルシステムが正式なコピーであるため、ホストへは何も同期されません。`container_persistent: false` の場合、各実行は一意な名前の Sprite を取得し、クリーンアップ時に削除されます。

**共有:** ゲートウェイと WebUI のセッションはそれぞれ独自の Sprite を取得し、`delegate_task` のサブエージェントは親のものを共有し、セッションのないフロー（CLI、cron）はプロファイルのデフォルトの Sprite を共有します。


**オプション設定**は `plugins.entries.sprites.settings` の下にあります（Plugins ハブでも編集できます）: `name_prefix`（すべての Sprite 名の `hermes-` を置き換える）、`scope`（`session` または `instance`。プロファイルごとに 1 つの Sprite）、そして新しく作成される Sprite 向けの `image`、`max_cpus`、`max_ram_gb`、`max_disk_gb` です。無効な値があるとバックエンドは起動せず、`hermes doctor` がそれを示します。[プラグインの README](https://github.com/NousResearch/hermes-plugin-sprites#settings) を参照してください。

### ターミナルバックエンドのよくある問題

ターミナルコマンドが即座に失敗する、またはターミナルツールが無効と報告される場合:

- **Local** — 特別な要件はありません。始める際の最も安全なデフォルトです。
- **Docker** — `docker version` を実行して Docker が動作していることを確認してください。失敗する場合は、Docker を修正するか `hermes config set terminal.backend local` を実行してください。
- **SSH** — `TERMINAL_SSH_HOST` と `TERMINAL_SSH_USER` の両方を設定する必要があります。いずれかが欠けている場合、Hermes は明確なエラーを記録します。
- **Modal** — `MODAL_TOKEN_ID` 環境変数または `~/.modal.toml` が必要です。`hermes doctor` を実行して確認してください。
- **Daytona** — `DAYTONA_API_KEY` が必要です。Daytona SDK がサーバー URL の設定を処理します。
- **Singularity** — `singularity` プラグイン（`hermes plugins install singularity`）と、`$PATH` 上の `apptainer` または `singularity` が必要です。HPC クラスタでは一般的です。
- **Sprites** — `sprites` プラグインと `SPRITES_TOKEN` が必要です。`hermes doctor` がトークン、SDK、命名のチェック結果を表示します。

迷ったときは、`terminal.backend` を `local` に戻し、まずそこでコマンドが実行されることを確認してください。

### ティアダウン時のリモート → ホスト間の状態同期

**SSH**、**Modal**、**Daytona** の各バックエンドでは、Hermes はセッション中にあなたの `~/.hermes/` の状態（認証情報ファイル、スキル、キャッシュ）をリモートサンドボックスへプッシュし、ティアダウン時に**変更された状態ファイルを元のホスト上の場所へ同期して戻します**。最初にプッシュした内容と異なるファイル（コンテンツハッシュで比較）はその場に適用され、同期対象のディレクトリ配下の新しいリモートファイル（例: エージェントがリモートで作成したスキル）は、対応するホストのパスへマッピングされます。アップロード専用の認証情報ファイルがホスト上で上書きされることはありません。

- 同期の書き戻しはバックオフ付きで最大 3 回再試行され、2 GiB を超えるリモートアーカイブの展開は拒否されます。より大きな状態ツリーのために上限を引き上げるには、`config.yaml` で `terminal.sync_back_max_bytes`（バイト）を設定してください。リモートの `~/.hermes/` 配下の稼働中のソケット（例: `gateway.sock`）は、転送を失敗させるのではなくスキップされます。
- ダウンロードされたアーカイブは、システムの一時ディレクトリ配下（`hermes-sync-back-<pid>-*`）にステージングされます。強制終了されたプロセスの残骸は、次の同期の書き戻し時に回収されます。
- Docker と Singularity は bind マウント（ホストのファイルシステムのライブビュー）を使うため、これは不要です。Sprites は独自の永続ファイルシステムを持つため、同様に書き戻しを行いません。
- これは Hermes の状態（`~/.hermes/`）を対象とするもので、サンドボックス内の任意の作業ツリーのファイルは対象**外**です — 重要な成果物は、サンドボックスが破棄される前にエージェントに明示的にコピーさせてください（例: `scp`、`modal volume put`）。

### Docker のボリュームマウント

Docker バックエンドを使用する場合、`docker_volumes` でホストディレクトリをコンテナと共有できます。各エントリは標準的な Docker `-v` 構文を使用します: `host_path:container_path[:options]`。

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # Read-write (default)
    - "/home/user/datasets:/data:ro"              # Read-only
    - "/home/user/.hermes/cache/documents:/output" # Gateway-visible exports
```

これは次の用途に役立ちます。
- エージェントに **ファイルを提供する**（データセット、設定、参照コード）
- エージェントから **ファイルを受け取る**（生成されたコード、レポート、エクスポート）
- あなたとエージェントの両方が同じファイルにアクセスする **共有ワークスペース**

メッセージングゲートウェイを使用し、エージェントに `MEDIA:/...` 経由で生成ファイルを送信させたい場合は、
`/home/user/.hermes/cache/documents:/output` のような専用のホストから見えるエクスポートマウントを使うことをおすすめします。

- Docker 内では `/output/...` にファイルを書き込む
- `MEDIA:` には **ホストパス** を出力する。例:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`
- その正確なパスがホスト上のゲートウェイプロセスにも存在する場合を除き、`/workspace/...` や `/output/...` を出力**しない**こと

:::warning
YAML の重複キーは、先のものを暗黙のうちに上書きします。すでに `docker_volumes:` ブロックがある場合は、
ファイル内の後ろに別の `docker_volumes:` キーを追加するのではなく、新しいマウントを同じリストにマージしてください。
:::

環境変数経由でも設定できます: `TERMINAL_DOCKER_VOLUMES='["/host:/container"]'`（JSON 配列）。

### Docker の認証情報転送

デフォルトでは、Docker のターミナルセッションは任意のホスト認証情報を継承しません。コンテナ内で特定のトークンが必要な場合は、`terminal.docker_forward_env` に追加してください。

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

Hermes は、列挙された各変数をまず現在のシェルから解決し、`hermes config set` で保存されていた場合は `~/.hermes/.env` にフォールバックします。

:::warning
`docker_forward_env` に列挙したものはすべて、コンテナ内で実行されるコマンドから見えるようになります。ターミナルセッションに公開しても問題のない認証情報のみを転送してください。
:::

### コンテナをホストユーザーとして実行する

デフォルトでは、Docker コンテナは `root`（UID 0）として実行されます。`/workspace` やその他のバインドマウント内で作成されたファイルはホスト上で root 所有となるため、セッション後にホストのエディタで編集するには、先に `sudo chown` する必要があります。Linux では、Hermes が root のコンテナを停止・削除する際、まずそのコンテナ自身のサンドボックスのバインドマウント（`TERMINAL_SANDBOX_DIR` 配下の `/root` と `/workspace`）内のファイルをあなたのユーザーに返します（`docker_volumes` やマウントしたホストディレクトリは対象外。rootless Docker と userns-remap ではスキップされます）。これらのサンドボックスファイルは、所有者にかかわらず setuid/setgid ビットとファイルケーパビリティも取り除かれるため、コンテナがそこに残したものをホストで実行しても昇格した権限で動くことはありません。削除されない長寿命のコンテナでは、セッションがデタッチまたは再アタッチするたび、および古いコンテナのスイープがそれを削除する前に、（所有者の変更なしで）この除去が行われます。エージェントが `MEDIA:` で送信したファイルをゲートウェイが読めない場合（稼働中のコンテナ内の root 所有で `chmod 600` の成果物）は、配信の直前にそのコンテナを通じてそのファイルだけがあなたのユーザーに返されるため、ティアダウンを待たずに届きます。そのモードはエージェントが設定したままです。`terminal.docker_run_as_host_user` フラグは、すべてのマウントと、削除されない長寿命のコンテナについてこれを解決します。

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # default: false
```

有効にすると、Hermes は `docker run` コマンドに `--user $(id -u):$(id -g)` を追加するため、バインドマウントされたディレクトリ（`/workspace`、`/root`、`docker_volumes` 内のすべて）に書き込まれるファイルは root ではなくホストユーザーの所有となります。トレードオフとして、コンテナは `apt install` や `/root/.npm` のような root 所有のパスへの書き込みができなくなります — 両方が必要な場合は、`HOME` が非 root ユーザーの所有であるベースイメージを使う（またはイメージのビルド時に必要なツールを追加する）ようにしてください。

後方互換の動作のためには、これを `false`（デフォルト）のままにします。ワークフローがほとんど「マウントされたホストファイルを編集する」もので、`sudo chown -R` にうんざりしているときに有効にしてください。

### Snap パッケージの Docker（AppArmor）

Docker が snap としてインストールされているホスト（Ubuntu のクラウドイメージ、たとえば Azure VM で一般的）では、snap の AppArmor による制限がサンドボックスのハードニングフラグのうち 2 つを拒否し、コンテナが起動時に終了します。

```
exec /sbin/docker-init: operation not permitted     # --init
exec /usr/bin/sleep: operation not permitted        # --security-opt no-new-privileges
```

これは snapd の制限（[LP#1908448](https://bugs.launchpad.net/snapd/+bug/1908448)）であり、Hermes がプローブで回避できるものではありません。snap ではなく Docker の apt リポジトリから Docker をインストールする（推奨 — すべてのハードニングが有効なまま）か、オプトインしてください。

```yaml
terminal:
  docker_snap_compat: true   # drops --init and no-new-privileges; cap-drop, tmpfs, PID limits stay
```

有効にすると、サンドボックス内のゾンビプロセスは init によって回収されなくなり、コンテナ内の setuid バイナリが権限を取り戻せるようになります。コンテナの起動時に警告がログに記録されます。

### オプション: 起動ディレクトリを `/workspace` にマウントする

Docker サンドボックスはデフォルトで分離されたままです。Hermes は、明示的にオプトインしない限り、現在のホスト作業ディレクトリをコンテナに渡し**ません**。

`config.yaml` で有効にします。

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

有効にすると:
- `~/projects/my-app` から Hermes を起動した場合、そのホストディレクトリが `/workspace` にバインドマウントされる
- Docker バックエンドは `/workspace` で開始する
- ファイルツールとターミナルコマンドの両方が、同じマウントされたプロジェクトを参照する

無効にすると、`docker_volumes` で明示的に何かをマウントしない限り、`/workspace` はサンドボックス所有のままになります。

セキュリティのトレードオフ:
- `false` はサンドボックスの境界を保持する
- `true` はサンドボックスに、Hermes を起動したディレクトリへの直接アクセスを与える

コンテナに稼働中のホストファイルで意図的に作業させたい場合にのみ、このオプトインを使用してください。

`terminal.cwd` にあるホストのパス（たとえば Windows の `C:\Users\me\project` や、デスクトップ/TUI セッションのワークスペース）が、コンテナの作業ディレクトリになることはありません: それが `/workspace` にマウントされたディレクトリである場合、ファイルツールとターミナルコマンドは `/workspace` を使用します。そうでない場合、コンテナは独自の作業ディレクトリを維持します。それでもファイルツールが作業ディレクトリに入れない場合、エラーはシェルの生の `cd:` 行ではなく、アクティブなバックエンドに対して無効な `terminal.cwd` を示します。

### 永続シェル {#persistent-shell}

**local** および **SSH** バックエンドでは、すべてのターミナルコマンドは新しい `bash` プロセスで実行されます。Hermes はログインシェルの環境（PATH、プロファイルからのエクスポート、関数、エイリアス）を一度だけ取り込み、最初のコマンドをそのスナップショットから開始します。設定がオフの場合、すべてのコマンドがそこから開始します。設定がオンの場合、各コマンドはエクスポートした変数を次のコマンドのために保存します。その保存されたコピーはエクスポートされた変数だけを保持するため、プロファイルの関数とエイリアスは最初のコマンドの後には失われます。

| 項目 | 設定オン（デフォルト） | 設定オフ |
|------|----------------------|-------------|
| 作業ディレクトリ（`cd ~/project`） | 引き継がれる | 引き継がれる |
| エクスポートした環境変数（`export FOO=bar`） | 引き継がれる | ログイン環境にリセットされる |
| 通常のシェル変数（`MY_VAR=hello`） | 引き継がれ**ない** | 引き継がれない |
| セッション中に定義した関数 / エイリアス | 引き継がれ**ない** | 引き継がれない |
| ログイン環境（PATH、プロファイルのエクスポート） | 常に存在する | 常に存在する |
| プロファイルの関数 / エイリアス | 最初のコマンドのみ | 常に存在する |

ファイルツールは相対パスを作業ディレクトリに対して解決するため、作業ディレクトリは常に引き継がれます。どちらのモードでも変数を維持するには、`export` する（設定オン）か、シェルプロファイルまたは `terminal.env_passthrough` に入れてください。

SSH バックエンドは `terminal.persistent_shell` を読み込みます。local バックエンドには独自のスイッチがあります。どちらもデフォルトはオンです。各コマンドをログイン環境から開始させるには:

```bash
hermes config set terminal.persistent_shell false   # SSH
export TERMINAL_LOCAL_PERSISTENT=false              # local
```

**優先順位:**

| バックエンド | 変数 | デフォルト |
|---------|----------|---------|
| SSH | `TERMINAL_SSH_PERSISTENT`、なければ `terminal.persistent_shell`（`TERMINAL_PERSISTENT_SHELL`） | `true` |
| Local | `TERMINAL_LOCAL_PERSISTENT` | `true` |

Docker、Modal、Daytona、Vercel Sandbox、およびプラグインのバックエンド（Singularity を含む）は、これらの設定を無視します。

各バックエンドの詳細は、[コード実行](features/code-execution.md)と [README のターミナルのセクション](features/tools.md)を参照してください。

## スキル設定 {#skill-settings}

スキルは、SKILL.md のフロントマターを通じて独自の設定項目を宣言できます。これらはシークレットではない値（パス、設定、ドメイン固有の設定）で、`config.yaml` の `skills.config` 名前空間の下に保存されます。

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # Example — each skill defines its own keys
```

**スキル設定の仕組み:**

- `hermes config migrate` は有効なすべてのスキルをスキャンし、未設定の項目を見つけて入力を促します
- `hermes config show` は、所属するスキルとともに「Skill Settings」の下にすべてのスキル設定を表示します
- スキルがロードされると、解決された設定値が自動的にスキルコンテキストへ注入されます

**値を手動で設定する:**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

独自のスキルで設定項目を宣言する方法の詳細は、[スキルの作成 — 設定項目](/docs/developer-guide/creating-skills#config-settings-configyaml)を参照してください。


### 毎セッションのスキルの自動ロード

スキルをピン留めすると、すべてのサーフェスで、新しいセッションの開始時に毎回完全にロードされます。

```yaml
skills:
  auto_load:
    - my-workflow
    - github-pr-workflow
```

システムプロンプトが最初に構築されるときにセッションごとに 1 回解決されます（そのためプロンプトはキャッシュが安定したまま保たれ、編集は次のセッションに適用されます）。存在しないスキルや無効なスキルは警告されてスキップされます。`--ignore-rules` / `HERMES_IGNORE_RULES=1` はこのリストを抑制します。プロファイル単位です。[CLI — 永続的な自動ロード](./cli.md#persistent-auto-load-via-config)を参照してください。

### エージェントが作成したスキル書き込みに対するガード {#guard-on-agent-created-skill-writes}

エージェントが `skill_manage` を使ってスキルを作成、編集、パッチ、削除する際、Hermes は新規/更新されたコンテンツについて、危険なキーワードパターン（認証情報の収集、明白なプロンプトインジェクション、漏えい指示）をオプションでスキャンできます。スキャナーは **デフォルトで無効** です — `~/.ssh/` に正当に触れたり `$OPENAI_API_KEY` に言及したりする実際のエージェントのワークフローが、このヒューリスティックに頻繁に引っかかっていたためです。エージェントのスキル書き込みが反映される前に確認を促してほしい場合は、これを有効に戻してください。

```yaml
skills:
  guard_agent_created: true   # default: false
```

有効にすると、フラグの立った `skill_manage` の書き込みは、スキャナーの根拠とともに承認プロンプトとして表示されます。承認された書き込みは反映され、拒否された書き込みはエージェントに説明的なエラーを返します。

バックグラウンドの自己改善レビューからの書き込みは、この設定にかかわらず常にスキャンされます: そのフォークには `terminal()` がないため、`skill_manage` がスキルを書き込む唯一の手段だからです。コンテンツにフラグを立てたスキャン、または完了できなかったスキャンは、書き込みをブロックします。ステージングされたレビューの書き込みを `/skills approve` で承認する場合も、同じスキャンが実行されます。

### スキル書き込みの承認

上記のコンテンツスキャナーとは独立して、`skills.write_approval` はエージェントの**すべての**スキル書き込み（作成 / 編集 / パッチ / 削除 / サポートファイル）を、あなたの明示的な承認の後ろに置きます — 危険なコマンドと同じ承認/拒否の仕組みです。

```yaml
skills:
  write_approval: false   # false = write freely (default) | true = stage every write for review
```

有効にすると、スキルの書き込みは `~/.hermes/pending/skills/` の下にステージングされ、CLI または任意のメッセージングプラットフォームから `/skills pending`、`/skills diff <id>`、`/skills approve <id>`、`/skills reject <id>` でレビューされます。実行時に `/skills approval on|off` で切り替えられます。メモリにも同じゲート（後述の `memory.write_approval`）があります。完全な手順: [エージェントのスキル書き込みのゲート](./features/skills.md#gating-agent-skill-writes-skillswrite_approval)。

## メモリ設定

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # true = require approval before any memory write
```

`memory.write_approval: true` の場合、メモリへの書き込みは反映される前にあなたの承認が必要になります: 対話的な CLI のターンではインラインで確認されます。メッセージングのセッションとバックグラウンドの自己改善レビューは、書き込みをステージングし、`/memory pending` → `/memory approve <id>` / `/memory reject <id>` でレビューできるようにします。実行時に `/memory approval on|off` で切り替えられます。[メモリへの書き込みの制御](./features/memory.md#controlling-memory-writes-write_approval)を参照してください。

## コンテキストファイルの切り詰め

先頭/末尾の切り詰めを適用する前に、Hermes が各自動コンテキストファイルからどれだけのコンテンツを読み込むかを制御します。これは `SOUL.md`、`.hermes.md`、`AGENTS.md`、`CLAUDE.md`、`.cursorrules` など、システムプロンプトに注入されるファイルに適用されます。`read_file` ツールには影響**しません**。

```yaml
context_file_max_chars: null  # default — dynamic cap scaled to the model's context window (floor 20K, ceiling 500K chars)
```

動的な動作の代わりに固定の上限にするには、正の整数を設定します。

```yaml
context_file_max_chars: 25000
```

各コンテキストファイルの読み取りは、`context_file_read_timeout`（秒、デフォルト `5.0`）によっても制限されます。読み取りにそれより長くかかるファイル — 典型的には iCloud Drive、OneDrive、NFS などのネットワークベースのファイルシステム上のもの — は警告とともにスキップされ、システムプロンプトの残りは引き続き読み込まれます。

```yaml
context_file_read_timeout: 5.0
```

## ファイル読み取りの安全性

単一の `read_file` 呼び出しが返せるコンテンツの量を制御します。上限を超える読み取りは、より小さな範囲のために `offset` と `limit` を使うよう促すエラーとともに拒否されます。これにより、ミニファイされた JS バンドルや大きなデータファイルを一度に読み取ってコンテキストウィンドウを溢れさせることを防ぎます。

```yaml
file_read_max_chars: 100000  # default — ~25-35K tokens
```

大きなコンテキストウィンドウを持つモデルを使用していて、大きなファイルを頻繁に読む場合は引き上げてください。小さなコンテキストのモデルでは、読み取りを効率的に保つために下げてください。

```yaml
# Large context model (200K+)
file_read_max_chars: 200000

# Small local model (16K context)
file_read_max_chars: 30000
```

エージェントはファイル読み取りも自動的に重複排除します — 同じファイル領域が二度読まれ、ファイルが変更されていない場合、コンテンツを再送する代わりに軽量なスタブが返されます。これはコンテキスト圧縮時にリセットされるため、コンテンツが要約されて失われた後でもエージェントはファイルを再読み取りできます。

## ツール出力の切り詰め上限

ツールが返せる生の出力量を Hermes が切り詰める前に制御する、3 つの関連する上限があります。

```yaml
tool_output:
  max_bytes: 50000        # terminal output cap (chars)
  max_lines: 2000         # read_file pagination cap
  max_line_length: 2000   # per-line cap in read_file's line-numbered view
```

- **`max_bytes`** — `terminal` コマンドが標準出力/標準エラー出力の合計でこの文字数を超える出力を生成した場合、Hermes は最初の 40% と最後の 60% を保持し、その間に `[OUTPUT TRUNCATED]` の通知を挿入します。デフォルト `50000`（一般的なトークナイザで約 12〜15K トークン）。
- **`max_lines`** — 単一の `read_file` 呼び出しの `limit` パラメータの上限。これを超えるリクエストは制限され、単一の読み取りがコンテキストウィンドウを溢れさせないようにします。デフォルト `2000`。
- **`max_line_length`** — `read_file` が行番号付きビューを出力する際に適用される 1 行あたりの上限。これより長い行は、この文字数に切り詰められ、その後に `... [truncated]` が付きます。デフォルト `2000`。

呼び出しごとにより多くの生の出力を許容できる大きなコンテキストウィンドウのモデルでは、上限を引き上げてください。小さなコンテキストのモデルでは、ツール結果をコンパクトに保つために下げてください。

```yaml
# Large context model (200K+)
tool_output:
  max_bytes: 150000
  max_lines: 5000

# Small local model (16K context)
tool_output:
  max_bytes: 20000
  max_lines: 500
```

### ツール結果の退避（スピルオーバー）予算

切り詰めとは別に、大きすぎるツールの*結果*は切り捨てられるのではなくディスクに退避されます: 完全な出力は `$HERMES_HOME/cache/spillover/` の下に保存され、コンテキスト内のコンテンツはプレビューと保存されたファイルのパスに置き換えられます（`offset`/`limit` を指定した `read_file` で読むか、`execute_code` で処理できます）。結果ごとの汎用的な退避の閾値は 100,000 文字で、小さなコンテキストのモデルでは自動的に縮小されます。退避ファイルは、いずれかのセッションのトランスクリプトがまだそれを指している限り保持され、そのセッションが削除または整理されるとともに削除されます。ゲートウェイの 1 時間ごとのハウスキーピングと各プロセスの最初の退避は、セッションの保持期間（`sessions.retention_days`、デフォルト 90 日、24 時間未満にはならない）より古い、参照されていないファイルだけを掃除するため、後で再開されたセッションでもトランスクリプトが指すものが見つかります。

MCP ツールの結果（`mcp_*` という名前のツール）は、より厳しい **50,000 文字**のデフォルトで退避されます: MCP サーバーは、ページネーションされていない大きなペイロード（ツール探索のカタログ、バッチ実行）を日常的に返すため、そうしないと汎用の閾値を下回ったまま、後続のターンのたびにコンテキストを肥大化させてしまいます。何も失われることはありません — 完全な結果はディスクに保存されています。閾値は次のようにオーバーライドします。

```yaml
tool_budget:
  mcp_result_size_chars: 50000   # per-result spillover threshold for mcp_* tools
```

MCP の閾値は常に（コンテキストに応じてスケールされる可能性のある）汎用の結果ごとの閾値で上限が設けられるため、引き上げてもアクティブなモデルのウィンドウが許す範囲を超えることはありません。

Hermes は**プロバイダー側の省略**にもフラグを立てます: MCP や Web ツールの結果が独自の切り詰めマーカー（`...N more items`、`"has_more": true`、「saved to sandbox」の注記）を含んでいる場合、表示されているデータは不完全であり、列挙を完全なものとして扱う前にページングや取得を行うべきだと警告する 1 行の通知が結果に追加されます。

## ツールセットのグローバル無効化

特定のツールセットを CLI とすべてのゲートウェイプラットフォームにわたって 1 か所で
抑制するには、その名前を `agent.disabled_toolsets` の下に列挙します。

```yaml
agent:
  disabled_toolsets:
    - memory       # hide memory tools + MEMORY_GUIDANCE injection
    - web          # no web_search / web_extract anywhere
```

これは（`hermes tools` によって書き込まれる）プラットフォームごとのツール設定（`platform_toolsets`）の
**後** に適用されるため、ここに列挙されたツールセットは常に削除されます — プラットフォームの
保存された設定にまだ列挙されていても同様です。`hermes tools` の UI で 15 以上のプラットフォーム行を
編集するのではなく、「どこでも X をオフにする」ための単一のスイッチが欲しい場合に使ってください。

削除はツール単位で行われるため、列挙されたツールセットは、それらのツールを共有する有効な
ツールセットからもそのツールを取り除きます: `search`（`web_search` のみ）は、有効な `web` から
`web_search` を取り除きます。`hermes tools list` と `hermes doctor` は、このようにツールを失った
有効なツールセットにフラグを立てます。また、`hermes tools`（チェックリスト、`hermes tools enable`、
Desktop またはダッシュボード）でツールセットを有効にすると、有効なツールセットが完全にカバーする
ここのエントリはすべてクリアされます。

リストを空のままにする、またはキーを省略すると、何も起こりません。

## Git ワークツリーの分離

同じリポジトリで複数のエージェントを並列実行するために、分離された git ワークツリーを有効にします。

```yaml
worktree: true    # Always create a worktree (same as hermes -w)
# worktree: false # Default — only when -w flag is passed
```

有効にすると、各 CLI セッションは独自のブランチを持つ新しいワークツリーを `.worktrees/` の下に作成します。エージェントは互いに干渉することなく、ファイルの編集、コミット、プッシュ、PR の作成ができます。クリーンなワークツリーは終了時に削除され、変更のあるものは手動での復旧のために保持されます。

デフォルトでは、新しいワークツリーは**新たにフェッチしたリモートの先端**（現在のブランチの upstream、なければリモートのデフォルトブランチ）から分岐するため、ローカルクローンの古いかもしれない `HEAD` からではなく、プロジェクトの最新の状態から開始します。これにより、PR の差分が、ローカルクローンが遅れていた分を引き継ぐことなく、実際の変更に限定されます。代わりにローカルの `HEAD` から分岐させるには `worktree_sync: false` を設定します — オフライン時や、クローンの現在の状態を意図的にベースにしたい場合に便利です。リモートに到達できない場合は、自動的にローカルの `HEAD` にフォールバックします。

```yaml
worktree_sync: true    # Default — branch from the fetched remote tip
# worktree_sync: false # Branch from local HEAD (offline / pinned base)
```

`.worktreeinclude` をリポジトリのルートに置くことで、ワークツリーにコピーする gitignore 対象のファイルを列挙することもできます。

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## コンテキスト圧縮 {#context-compression}

Hermes は、モデルのコンテキストウィンドウ内に収まるように、長い会話を自動的に圧縮します。圧縮の要約器は別の LLM 呼び出しです — 任意のプロバイダーやエンドポイントに向けることができます。

すべての圧縮設定は `config.yaml` にあります（環境変数はありません）。

### 完全なリファレンス

```yaml
compression:
  enabled: true                                     # Toggle compression on/off
  progress_notices: false                           # Opt-in: deliver routine compression progress notices to chat platforms — see below
  threshold: 0.50                                   # Compress at this % of context limit
  threshold_tokens: null                            # Absolute token cap (optional) — takes lower of ratio vs absolute
  target_ratio: 0.20                                # Fraction of threshold to preserve as recent tail
  tail_mode: lean                                   # Tail retention: "lean" (default — clamped 2.5% tail, 10K-25K, never above 20% of the window, with a detailed session log + anchor index + session_search recovery pointers in the summary, all from ONE auxiliary summarizer call; ~3x fewer retained tokens after compaction) or "legacy" (0.20×threshold verbatim tail)
  protect_last_n: 20                                # Min recent messages to keep uncompressed
  protect_first_n: 3                                # Non-system head messages pinned across compactions (0 = pin nothing)
  in_place: true                                    # Compact on the same session id (no rotation) — see below
  idle_compact_after_seconds: 0                     # Opt-in idle compaction (0 = disabled) — see below
  prepare_ahead: false                              # Opt-in: prepare the next summary in the background — see below
  hygiene_hard_message_limit: 5000                  # Gateway safety valve — see below
  hygiene_timeout_seconds: 30                       # Max seconds of NO summary-model output before hygiene compression is cut off
  hygiene_total_ceiling_seconds: 600                # Absolute cap on the hygiene wait even while tokens are still streaming
  hygiene_max_turn_hold_seconds: 10                 # Max wall-clock the incoming turn waits on hygiene compression before proceeding uncompressed — see below
  hygiene_failure_cooldown_seconds: 300             # First rung of the per-session hygiene-failure backoff (x1/x3/x9, capped at 1h)
  context_timeout_seconds: 120                      # Inactivity budget for in-agent compress_context (loop /compress / preflight) — see below
  context_total_ceiling_seconds: 600                # Absolute cap on the *pre-commit* in-agent compress_context wait even while tokens are still streaming (an already-started SessionDB commit is never abandoned; overruns are logged + surfaced)
  proactive_prune_tokens: 0                         # Opt-in tokens trigger for the no-LLM tool-result prune (0 = off; see below)
  proactive_prune_min_result_chars: 8000            # Prune's summarize pass only touches tool results larger than this (clamped >= 200)
  proactive_prune_min_reclaim_tokens: 4096          # Prune only commits when it reclaims at least this many tokens (0 = commit any)

# The summarization model/provider is configured under auxiliary:
auxiliary:
  compression:
    model: ""                                       # Empty = use main chat model. Override with e.g. "google/gemini-3-flash-preview" for cheaper/faster compression.
    provider: "auto"                                # Provider: "auto", "openrouter", "nous", "codex", "main", etc.
    base_url: null                                  # Custom OpenAI-compatible endpoint (overrides provider)
```

:::info レガシー設定の移行
`compression.summary_model`、`compression.summary_provider`、`compression.summary_base_url` を持つ古い設定は、初回ロード時に自動的に `auxiliary.compression.*` へ移行されます（設定バージョン 17）。手動での操作は不要です。
:::

`progress_notices`（デフォルト `false`）は、**定常的な**圧縮の進捗ステータスをチャットプラットフォーム（Telegram、Discord、Slack など）に届けるかどうかを制御します。設計上、自動圧縮はチャットのサーフェスでは無言です — バックグラウンドで実行され、サーバー側でのみログが記録されます。チャットプラットフォームで定常的なライフサイクルを見るようにオプトインするには、`progress_notices: true` を設定します: 「Compacting context…」の開始通知、プリフライト/API 前の圧縮トリガー、アイドル時の圧縮、再試行の進捗（「Compressed 30 → 12 messages, retrying…」）、そして「Context compaction complete」の通知です。このゲートは圧縮のステータスだけに限定されます — 無関係な運用上のノイズ（補助モデルの失敗、プロバイダーのレート制限/再試行のやり取り）はどちらの場合も抑制されたままです。圧縮の**失敗**の通知と手動の `/compress` のフィードバックは、この設定にかかわらず常に表示されます。稼働中のゲートウェイでこの値を編集すると、次のメッセージで反映されます。

`hygiene_hard_message_limit` はゲートウェイ専用の **圧縮前の安全弁** です。これは死のスパイラルを断ち切るために存在します: 大きすぎるセッションで API 呼び出しが切断され続けると、ゲートウェイはトークン使用量のデータを受け取れないため、トークンベースの閾値が発火できず、トランスクリプトは増え続け、切断はさらに悪化します。このメッセージ数ベースの下限は、メッセージ数だけで発火し（API の失敗にかかわらず常にわかります）、圧縮を強制してセッションを回復させます。デフォルト `5000` — 通常のどのセッションよりもはるかに大きな値であり、数千の短いターンを行う大きなコンテキスト（1M 以上）のモデルであっても、その前にトークンの閾値で圧縮されます。特殊なプラットフォームではさらに引き上げ、より積極的な圧縮を強制するには下げてください。稼働中のゲートウェイでこの値を編集すると、次のメッセージで反映されます（後述）。

同じ上限は、ターンの開始時点までにハイジーンが完了しない場合の**モデルに送信する内容に対するフェイルクローズドの上限**でもあります — ターン保持の予算が切れた、要約がタイムアウトまたは失敗した、失敗のクールダウンが有効、別の圧縮がまだ進行中、あるいは圧縮が無効になっている場合です。その場合、ゲートウェイは先頭のシステム/セットアップの行と最新のメッセージを、合計で最大 `hygiene_hard_message_limit` 件まで保持し、保持する末尾を孤立したツール結果から始めることはありません。切り詰められるのはその 1 ターンのペイロードだけです: ディスク上のトランスクリプトには手を付けず、何も削除されず、後で完了した要約はそのまま採用されます。これにより、圧縮の処理が失敗し続けるときに、1 週間続く DM が未圧縮の履歴全体をモデルに送り込むことを防ぎます。

`hygiene_timeout_seconds` は、このエージェント前の圧縮処理に対するゲートウェイの**非アクティブ予算**です — 合計の実時間の上限ではありません。圧縮の要約呼び出しはモデルからストリーミングされ、到着する各トークンは前進としてカウントされます: まだ生成中の遅い推論モデルは自身の期限を延長し続けるため、遅いが健全な要約モデルが生成の途中で打ち切られることはありません。要約モデルがこの秒数の間**何も出力しない**場合（バックエンドの停止、ハングした接続、応答のないプロバイダー）にのみ、ゲートウェイはユーザーに警告し、受信したメッセージを圧縮なしで続行し、固まって見える代わりにセッションごとの一時的な失敗クールダウンを記録します。

`hygiene_total_ceiling_seconds`（デフォルト `600`）は、トークンがまだ流れている間でも合計の待ち時間を制限するため、少しずつしか流れない異常なストリームがターンを無期限に人質に取ることはありません。これは少なくとも `hygiene_timeout_seconds` に制限されます。llama.cpp サーバーがプロンプトの読み込みに費やす時間はこれに含まれません（`context_total_ceiling_seconds` を参照）。

`hygiene_max_turn_hold_seconds`（デフォルト `10`）はゲートウェイの**ターン保持予算**です — ゲートウェイが待つのをやめて未圧縮のトランスクリプトで続行するまでに、受信したメッセージがハイジーン圧縮を待って保持される最大の実時間です。これが存在するのは、`hygiene_total_ceiling_seconds` だけでは、チャットのトランスポートのアイドルタイムアウトよりはるかに長く通信が無言のままになり得るためです: トークンをストリーミングし続ける要約モデルは非アクティブの区間をリセットし続けるため、ターン保持予算がないと、ユーザーに 1 バイトも届かないまま待ち時間が上限近くまで延びることがあります — すると Telegram（および同様のトランスポート）は接続を切断し、ターンは固まったように見えます。ターンの待ち時間をこの予算（一般的な約 30 秒のトランスポートのアイドルタイムアウトを十分に下回る）で制限することで、メッセージに速やかに応答することが保証されます。**予算が切れても圧縮は失われません**: ワーカーはデタッチされたまま実行を続け、そのコミットがウォーターマークで保護されている場合（セッション DB がある通常の場合）はコミットの許可を保持するため、完成した要約は次の安全な境界で採用され、待つのをやめた後に追加されたターンは並行する末尾としてそのまま残ります。これは特に、推論フェーズだけで予算を超えることがある**思考/推論型の要約モデル**（DeepSeek、QwQ など）で重要です: それらの要約は決して反映されないのではなく、1 ターン遅れて反映されます。コミットを安全に保護できない場合、遅れた結果は破棄され（`CompressionCommitFence`）、新しいターンを上書きすることはできません。同じターン内で圧縮を適用したく、トランスポートが待ち時間を許容する場合は予算を引き上げてください。非常に遅いバックエンドでより素早く回復させたい場合は下げてください。

`hygiene_failure_cooldown_seconds` は、ハイジーン圧縮のタイムアウトまたは中断後の、そのセッションごとのクールダウンを制御します。クールダウン中、ゲートウェイは同じ大きすぎるセッションに対するハイジーンの再試行をスキップするため、受信するすべてのメッセージが同じ壊れた補助バックエンドでブロックされることはありません。`/compress`、`/reset`、または後の健全なターンによって、引き続きセッションを回復できます。

この値は固定の間隔ではなく、段階的に増える階段の**最初の段**です: 同じセッションで連続して失敗すると、この値の `1x`、`3x`、そして `9x` だけ待ち、上限は 1 時間です。そのため、要約モデルが恒久的に壊れているセッションは、固定の間隔で永遠に再試行するのではなく後退します。実際にトランスクリプトを縮小した実行は、最初の段にリセットします。段階の上昇はセッションごとかつプロセスローカルです — ゲートウェイを再起動すると最初の段にリセットされますが、クールダウンの期限自体は維持されます。

`context_timeout_seconds`（デフォルト `120`）は、エージェント内の `compress_context` — 会話ループ、プリフライトの圧縮、手動の `/compress` — に対する同じ**非アクティブ予算**であり、ハングした要約モデルがセッションを無期限に止めることはありません。ストリーミングされた要約トークンは待ち時間を延長し、打ち切られるのは無言のワーカーだけです。この予算の下限は補助圧縮リクエスト自体のタイムアウト（`auxiliary.compression.timeout`、最小 300 秒）であるため、ホストがリクエスト自体よりも先に無言の要約器を見限ることはありません — 最初のトークンの前に思考する推論型の要約器や、ストリーミングできない経路も、プロバイダー呼び出しと同じ予算を得ます。タイムアウトすると、Hermes は `auxiliary.compression.fallback_chain` の最初のエントリに対して（そのエントリが独自の `timeout` を宣言していればそれを使って）要約を 1 回再試行します — 停滞した経路は例外を発生させないため、補助クライアント自身のフォールバック処理はそれを検知できません。その試行も失敗した場合、またはフォールバックチェーンが設定されていない場合、次に起こることはリクエストがまだモデルのコンテキストウィンドウに収まるかどうかによります: 収まるリクエストはこのターンでは未圧縮で送信されます（要約失敗のクールダウンにより、再試行が毎ターン繰り返されることはありません）。ウィンドウを超えるリクエストはまったく送信できないため、Hermes はターンを終了する代わりに、決定論的なフォールバック要約（古いツール結果を整理し、要約された中間部分の代わりに静的な引き継ぎを置いたもの）をコミットします — 「compression timed out」の復旧結果でターンを終了すること（メッセージングゲートウェイでは自動的なセッションのリセットも）は最後の手段であり、決定論的な処理でさえトランスクリプトを縮小できない場合にのみ到達します。無効にするには `0` を設定します。ウィンドウを超えたままのリクエストで何も回収できなかったプリフライトの処理は、モデルが受け付けられないリクエストを送信する代わりに、新しいセッション（`/new`）を開始するよう案内してターンを直ちに終了します。ゲートウェイのセッションハイジーンは独自の `hygiene_timeout_seconds` の経路を保持し、二重にラップされることはありません。

`context_total_ceiling_seconds`（デフォルト `600`）は、トークンがまだ流れている間でも、エージェント内の**コミット前**の待ち時間（要約 / ストリームのフェーズ）を制限します。これは少なくとも `context_timeout_seconds` に制限されます。llama.cpp サーバーがプロンプトの読み込みに費やす時間はこれに含まれません: ノート PC の GPU では、150K トークンの会話を読み込むだけで上限より長くかかることがあります。サーバーは読み込みを進捗として報告し、新たに読み込んだトークンの各報告は、読み込みにかかった時間だけ期限を後ろにずらします。何も読み込まずに報告するサーバーには追加の時間は与えられず、報告の間にはアイドルタイムアウトが引き続き適用されます。すでにモデルのコンテキストウィンドウを超えているリクエストでは、コミット前の待ち時間はこの上限ではなく 1 回分の `context_timeout_seconds` の予算で制限されます: そのようなリクエストはいずれにせよ未圧縮では送信できず、何も回収しないままストリーミングを続ける要約は、そうしないと毎ターン上限いっぱいまでセッション（と Desktop の UI）を拘束してしまうためです — その場合は決定論的なフォールバック要約が圧縮を担います。要約器が正当に長い時間を必要とする場合は、`context_timeout_seconds` を引き上げてください。正確な保証は次のとおりです: **要約のフェーズはこの上限で制限され、コミットのフェーズはそれを超えた場合にログに記録されて表示されます。** ワーカーが圧縮のコミットフェンスに入り、SessionDB の変更が進行中になると、コミットが途中で放棄されることは決してありません — それはトランスクリプトの不整合を招くおそれがあるためです — が、待機はもはや無言ではありません: コミットが上限を超えて実行されると、Hermes は超過をログに記録し（WARNING、繰り返すと ERROR に格上げ）、ユーザーに見える警告チャンネルを通じて 1 回限りの警告を送信し、コミットが完了するまで制限された単位で待ち続けます。要約のフェーズ中に上限が切れると、要約モデルのストリームは、すべての補助ワイヤー（chat.completions、Codex Responses、Anthropic Messages）でその瞬間に閉じられます — 放棄された要約が、誰も待っていない接続で完了まで課金されることはなく、そのセッションのリースは次の試行のために解放されます。

`protect_first_n` は、すべての圧縮をまたいでピン留めされる**システム以外の**先頭メッセージの数を制御します。デフォルト `3` — 冒頭のユーザー/アシスタントのやり取りはすべての要約処理を生き延びるため、元の目標が見えたままになります。冒頭のターンがもはや関係ない長時間のローリング圧縮セッションでは、`protect_first_n: 0` を設定すると、システムプロンプト + 要約 + 末尾以外は何もピン留めしません。システムプロンプト自体は、この設定にかかわらず常に保持されます。

`in_place`（デフォルト `true`）は、圧縮が発火したときにセッションのアイデンティティがどうなるかを制御します。`true` の場合、圧縮はメッセージリストを書き換え、**セッション ID をローテーションせずに**システムプロンプトを再構築します — 会話はその生涯を通じて 1 つの永続的な ID を保持します（`parent_session_id` のチェーンはなく、セッション一覧での `name #2` / `#3` の番号の付け直しもありません）。圧縮は非破壊的です: ライブのコンテキストは圧縮されますが、圧縮前のターンは同じ ID の下でソフトアーカイブされ（非アクティブ/圧縮済みとしてマーク）、削除されずに `session_search` で検索・復元できます。フックは `session:compress` イベントの `in_place` フィールドでモードを確認できます。各圧縮で古い ID にリンクされた新しいセッション ID にローテーションするレガシーの動作に戻すには、`in_place: false` を設定します。

`threshold_tokens` は、圧縮トリガーのオプションの**絶対トークン上限**を設定します。設定すると、圧縮は比率ベースの `threshold` とこの絶対数のうち低い方で発火するため、どのモデルがアクティブであっても、そのトークン数より後に圧縮が発火することはありません。呼び出しごとのコストの上限を固定したい場合に使います。たとえば `threshold_tokens: 256000` とすると、1M ウィンドウのモデルを 500K ではなく 256K で圧縮します。上限はモデルのコンテキスト長に制限されるため、ウィンドウを超える値は何もしません。デフォルト `null`（無効 — 比率ベースの閾値のみ）。この上限は、モデルの切り替えやフォールバックの有効化をまたいで維持されます。

`idle_compact_after_seconds` は、サイズベースの `threshold` を補完する**オプトインの時間ベース**のトリガーです。デフォルト `0`（無効）。0 より大きく設定すると、少なくともその秒数の非アクティブの後に再開したセッションは、最初の返信の前に蓄積された履歴を先に圧縮します — そのため、長寿命のスレッド（たとえば数時間後に戻ってくる Telegram の会話）が、後続のターンのたびに古いコンテキスト全体を読み直すことはありません。コンテキストがすでに圧縮後の目標（`threshold × target_ratio`）以下の場合は発火せず、すべての自動圧縮と同じ失敗クールダウン、スラッシング防止、セッションごとのロックのガードを尊重します。例: `idle_compact_after_seconds: 1800` は 30 分のアイドルの後に圧縮します。

`prepare_ahead`（デフォルト `false`）は、セッションがトリガーに近づいたら、ツールの実行中やターンの合間に、次の圧縮の要約をバックグラウンドで書いておきます。要約は圧縮ごとに 1 つ準備されます。自動圧縮が発火したとき、要約対象のメッセージがまだ同じメッセージであれば（プラグインによるテキストの編集は問題ありませんが、メッセージの削除や巻き戻しは不可）、その時点で要約を呼び出す代わりに準備された要約が使われ、それより新しいメッセージはすべてその外側の未圧縮の末尾に残ります。圧縮が発火したときに準備された要約がまだ書かれている途中であれば、圧縮はそれを書き直すのではなく（通常の圧縮の時間制限内で）それを待ちます。準備された要約がない、またはもう一致しない場合、圧縮はこのオプションがない場合とまったく同じように実行されます。使われない可能性のある補助の要約呼び出しを消費し、手動の `/compress`、プラグインのコンテキストエンジン、ネイティブ/アプリサーバーの圧縮には適用されません。圧縮の要約にコンテキストを追加するメモリプロバイダーがある場合は、準備した要約がすべて破棄されることになるため、最初の圧縮の後に自動的にオフになります。

`proactive_prune_tokens` は、`threshold` とは独立して実行される、古いツール結果のペイロードの決定論的で LLM を使わない整理を有効にします。大きなウィンドウのモデルでは `threshold` による圧縮（ウィンドウの約 50%）はめったに発火しないため、かさばるツール出力（ターミナルのダンプ、ファイルの読み取り、Web の抽出）が履歴に乗ったまま、後続のターンのたびに再送されます。再送される履歴が `proactive_prune_tokens`（デフォルト `0` = オフ。有効にするには `48000` を試してください）を超えると、整理処理は同一の結果を重複排除し、古い大きすぎるツール結果を要約します。直近の `protect_last_n` 件のメッセージは保護され、モデルが呼び出されることはありません。ツール呼び出しの引数は実行記録であり、整理によって書き換えられることはありません。要約モデルには、そのプロンプトに含めるコピーのための別の制限付きシリアライザーがあります。プレッシャー処理も同様にツール結果の本文だけを降格させます。完全なセマンティック圧縮の間、引き継がれる先頭/末尾の行はそれらのツール結果の降格を保持し（そのため大きすぎる末尾も引き続き圧縮できます）、ツール呼び出しの引数はバイト単位で正確なままです。オプトインの先行的な整理は、対象となるツール結果本文の降格を引き続きコミットするため、`proactive_prune_min_reclaim_tokens`（デフォルト `4096`）によってキャッシュを壊すそれらのコミットを散発的なものに保ちます。`proactive_prune_min_result_chars`（デフォルト `8000`、200 以上に制限）は、それ未満のツール結果には手を付けないサイズを設定します。これは組み込みの `compressor` エンジンの下でのみ実行され、他のコンテキストエンジンは何もしない動作を継承します。

:::tip 圧縮とコンテキスト長のゲートウェイのホットリロード
最近のリリースから、稼働中のゲートウェイで `config.yaml` の `model.context_length` または任意の `compression.*` キーを編集すると、次のメッセージで反映されます — ゲートウェイの再起動、`/reset`、セッションのローテーションは不要です。キャッシュされたエージェントのシグネチャにこれらのキーが含まれるため、ゲートウェイは変更を検知すると透過的にエージェントを再構築します。API キーやツール/スキル設定は、依然として通常のリロード経路が必要です。
:::

### 一般的なセットアップ

**デフォルト（自動検出）— 設定不要:**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
メインのプロバイダーとメインのモデルを使用します。メインのチャットモデルより安価なモデルで圧縮したい場合は、タスクごとにオーバーライドしてください（例: `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`）。

**特定のプロバイダーを強制**（OAuth または API キーベース）:
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```
任意のプロバイダーで動作します: `nous`、`openrouter`、`codex`、`anthropic`、`main` など。

**カスタムエンドポイント**（セルフホスト、Ollama、zai、DeepSeek など）:
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
    api_key: ${GLM_API_KEY}
```
カスタムの OpenAI 互換エンドポイントを指し、`api_key` で認証します。それがない場合、Hermes はメインのモデルのキーをメインのエンドポイントとまったく同じオリジンでのみ再利用し、`OPENAI_API_KEY` は OpenAI のホスト、またはそれと組になっている `OPENAI_BASE_URL` とまったく同じ場合にのみ使用します。

### 3 つのつまみの相互作用

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | 結果 |
|---------------------|---------------------|--------|
| `auto`（デフォルト） | 未設定 | 利用可能な最良のプロバイダーを自動検出 |
| `nous` / `openrouter` など | 未設定 | そのプロバイダーを強制し、その認証を使用 |
| いずれか | 設定済み | カスタムエンドポイントを直接使用（プロバイダーは無視） |

### ストリームの進捗タイムアウト

要約がストリーミングされる場合（圧縮は、`openai-codex` やカスタムの OpenAI 互換エンドポイントなど、chat-completions と Responses の経路では常にストリーミングされます）、2 つのタイムアウトが適用され、それらは独立しています。

- `auxiliary.compression.timeout` — リクエスト全体の予算（デフォルト 120 秒）。
- `auxiliary.compression.no_progress_timeout` — 試行が中断され（`... stream stalled: no new output for Ns`）、通常の再試行 / フォールバックチェーンが引き継ぐまでに、ストリームが**実質的な**イベント（テキスト/推論の差分、または完了した出力項目）なしで続けられる時間。未設定の場合のデフォルトは **60 秒**です。chat-completions のストリームでは、この時間枠は最初のトークンから始まります。それより前は、リモートのエンドポイントにはメインループのスタールに対する猶予が与えられます（プロバイダーの `providers.<id>.stale_timeout_seconds` が設定されていればその値、なければ 180 秒で、大きなプロンプトでは引き上げられ、o3 や DeepSeek R1 などの推論モデルでは 600 秒に引き上げられます。上限は `timeout`）。そのため、無言で思考しているモデルが打ち切られることはありません。ローカルのエンドポイント（localhost、LAN、Tailscale）では、無言のプリフィルは `timeout` によってのみ制限されます。同じルールは、MoA の参照呼び出しを含む、ストリーミングされるすべての補助タスクに適用されます。キープアライブやライフサイクルのフレーム（`response.in_progress`、ping）は進捗としてカウントされません。実質的なイベントのたびに時間枠が再設定されるため、遅いが進んでいる要約がこれによって打ち切られることはありません。

`timeout` だけを引き上げても、進捗の時間枠は広がり**ません** — 600 秒に設定したリクエストでも、60 秒の空白の後には中断されます。その空白を変更するには `no_progress_timeout` を設定してください。実効的な時間枠は `timeout` で上限が設けられ、ホストのハードな期限 / キャンセルが引き続き優先されます。ここではホスト自身の非アクティブ予算が外側の上限になります: エージェント内の圧縮は `compression.context_timeout_seconds`（デフォルト 120 秒、下限は実効的な `auxiliary.compression.timeout` で、それ自体が少なくとも 300 秒）の後に、ゲートウェイのハイジーンは `compression.hygiene_timeout_seconds`（デフォルト 30 秒）の後に、無言の要約器を見限ります。そのため、該当するホストの予算より大きな `no_progress_timeout` は、それによって暗黙のうちに短縮されます。このキーはタスクごと（`auxiliary.<task>.no_progress_timeout`）であるため、圧縮のために広げても他の補助タスクは変わりません。正の数でない値は、ログに警告が出されて無視され、60 秒のデフォルトが適用されます。

```yaml
auxiliary:
  compression:
    provider: openai-codex
    timeout: 600
    no_progress_timeout: 180   # tolerate a 3-minute silent gap on a long reasoning summary
```

:::warning 要約モデルのコンテキスト長の要件
要約モデルは、メインのエージェントモデルと **少なくとも同じ大きさ** のコンテキストウィンドウを持つ **必要があります**。圧縮器は会話の中間部分全体を要約モデルに送信します — そのモデルのコンテキストウィンドウがメインのモデルより小さいと、要約呼び出しはコンテキスト長エラーで失敗します。これが起きると、中間のターンは **要約なしで破棄され**、会話のコンテキストが暗黙のうちに失われます。モデルをオーバーライドする場合は、そのコンテキスト長がメインのモデル以上であることを確認してください。
:::

## ゲートウェイのターンリースのタイムアウト

ゲートウェイは、2 つのルーティングキーが同じトランスクリプトを同時に読み込んで書き込むことが
ないよう、解決されたセッション ID ごとにターンを直列化します。リースの最大待ち時間は、
通常のエージェントの非アクティブタイムアウトとは独立して設定します。

```yaml
agent:
  gateway_turn_lease_timeout: 5
```

この予算が切れたときに別のターンがまだセッションのリースを保持している場合、Hermes は
フェイルクローズドで動作します: 待機中のメッセージについて、トランスクリプトを読み込まず、モデルも実行しません。
ユーザーは拒否の通知を受け取り、再送する必要があります。永続的な順序付けと冪等性なしに
メッセージを自動的に再キューすると二重に処理される可能性があるため、Hermes はそれを行いません。
正でない値の場合は 5 秒のデフォルトが使われます。


## ゲートウェイの SessionDB ループガード

ゲートウェイの停止をデバッグするための補助機能です: ゲートウェイのイベントループ上での同期的な `state.db` の
読み書きは、ストレージにかかる時間だけすべてのプラットフォームをブロックします。ガードを有効にすると、
ゲートウェイはそのような呼び出しにフラグを立てます。

```yaml
gateway:
  sessiondb_loop_guard: warn   # off (default) | warn | raise
```

`warn` は問題のある呼び出し箇所ごとにスタックとともに 1 回ログに記録します。`raise` はその呼び出しを
失敗させます（開発や CI で便利です）。オフの場合はコストがかかりません: どのストレージメソッドも
ラップされません。次回のゲートウェイ起動時に反映されます。ガードはプロセス全体に適用されるため、
多重化の下では起動したプロファイルの `config.yaml` だけがこれを設定します。セカンダリプロファイルの値は無視されます。

## ゲートウェイの再起動時のドレイン {#gateway-restart-drain}

ゲートウェイが停止または再起動するとき（SIGTERM、`hermes update`、`hermes gateway stop`、または
`hermes gateway restart` / `/restart` の最終フェーズ）、`agent.restart_drain_timeout` は、進行中のチャットの
ターンを中断するまでに、それらを何秒間実行させ続けるかを表します。

```yaml
agent:
  restart_drain_timeout: 0          # default: interrupt in-flight chat turns at once
  cron_drain_timeout: 30            # floor for cron jobs and api_server (/v1) runs
  restart_after_turn_timeout: 1800  # /restart and `gateway restart`: wait for turns BEFORE stopping
```

デフォルトの `0` では、再起動は実行中のターンを直ちに中断します。そのターンは中断として確定し、
二度と実行されません。TUI と Desktop はそのターンに「Gateway restarting」（または「Gateway
shutting down」）の通知を表示するため、Hermes が再接続したらメッセージを再送してください。
その後ろにキューされたメッセージは保持され、新しいゲートウェイで順番に実行されます。

ターンを最後まで終わらせたい場合は、正の値を設定します。

```yaml
agent:
  restart_drain_timeout: 60
```

ドレインを長くすると、すべての再起動とアップデートが最大でその時間だけ待つことになります。`HERMES_RESTART_DRAIN_TIMEOUT`
は設定値をオーバーライドします。

:::caution systemd の `TimeoutStopSec`
生成される systemd ユニットは、この値から `TimeoutStopSec` の大きさを決めます: `restart_drain_timeout` と
`cron_drain_timeout` + 10 秒のうち大きい方に、30 秒の余裕を加えたもので、最低 60 秒です
（`gateway/restart.py`、`resolve_systemd_timeout_stop_sec`）。`restart_drain_timeout` を引き上げた場合は、
ユニットが新しい予算を取り込むように `hermes gateway install` を再実行してください。そうしないと、systemd は
ドレインの途中でゲートウェイを SIGKILL します。launchd は停止タイムアウトの上限を 60 秒としているため、macOS では
ドレインはその範囲に収まる長さに制限されます。
:::

## セッションの停止ウォッチドッグ

ゲートウェイは、通知のみの停止ウォッチドッグ（`agent.session_stall_timeout`、デフォルト `300` 秒、`0` = 無効）を実行します。ビジーなセッションに**保留中の受信フォローアップ**があり、エージェントの共有アクティビティクロックが少なくともこの時間アイドル状態になっている場合、ゲートウェイは WARNING をログに記録し、ユーザーに 1 回限りの通知を送信します。

```
⚠️ Agent session appears stalled (last activity N min ago). Try /new to reset.
```

セマンティクス:

- **通知のみ。** ウォッチドッグがターンを強制終了することはありません — 長時間の非アクティブの後に実行をキャンセルする `agent.gateway_timeout` とは対照的です。停止の通知は、エージェントが動かなくなっているように見えることを伝えるだけなので、どうするか（`/new`、`/stop`、または待ち続ける）を決められます。
- **停止のエピソードごとに 1 回の通知。** 保留中の受信がなくなるかアクティビティが再開するとラッチがクリアされるため、回復した後に再び停止したセッションは再び通知します。
- 進捗は共有アクティビティのスナップショット（ツール呼び出し、API ストリームの進捗、圧縮のハートビート）からのみ得られます。保留中の受信は通知のゲートであって、進捗のクロックではありません。
- **エージェントの開始前。** まだセットアップ中のターン（セッションの読み込み、トランスクリプト、`agent:start` フック）は、このタイムアウトを過ぎたフォローアップに「Still starting this turn … Use /stop if it should be cancelled」と応答します。セットアップ自体は `agent.gateway_timeout` で制限されます: それを過ぎるとターンは「Turn setup stalled … Please resend your message」で終了し、次のメッセージが実行されます。`/stop` と `/new` は、まだセットアップ中のターンを終了させます。

```yaml
agent:
  session_stall_timeout: 300   # seconds; 0 disables the watchdog
```

## 再接続の要対応エスカレーション

プラットフォームアダプターが接続に失敗した場合（ネットワークの障害、取り消されたボットトークン、壊れたサイドカー）、ゲートウェイは上限付きの指数バックオフで無期限に再試行します — 再試行は決して止まらないため、一時的な障害はオペレーターの操作なしに常に自己回復します。欠点は、*恒久的な*失敗（取り消された Telegram トークン、Discord の特権インテントの欠如）が一時的な不調と同じに見えることです: 「retrying」が永遠に続きます。

恒久的な失敗を可視化する仕組みが 2 つあります。

- **終端的な分類。** 例外の*型*によって決して自己回復しないことが証明される失敗 — 拒否/取り消されたトークン（`telegram_auth_error`、`discord_auth_error`、`email_auth_error`）、特権インテントの欠如（`discord_intents_required`）、依存関係をインストールできない Photon サイドカー（`SIDECAR_DEPS_MISSING`）や node バイナリが欠けている Photon サイドカー（`SIDECAR_NODE_MISSING`）— は、再試行キューに入る代わりに致命的とマークされます。分類は厳密に型ベースであり、曖昧なエラーは常に再試行を続けます。
- **要対応のエスカレーション。** `agent.reconnect_attention_after`（デフォルト `7200` 秒 = 2 時間、`0` で無効）を超えて再試行キューに入り続けているプラットフォームは、ゲートウェイの実行時ステータス（`hermes status`）で `needs_attention: true` と `retrying_since` のタイムスタンプを取得し、WARNING ログも出力されます。再試行は変わらず続きます — これはシグナルであって、サーキットブレーカーではありません。このフラグは再接続に成功するとクリアされます。

```yaml
agent:
  reconnect_attention_after: 7200   # seconds; 0 disables the escalation flag
```

## ゲートウェイのエージェントキャッシュ

ゲートウェイはセッションごとに 1 つのエージェントを保持するため、会話は毎ターンシステムプロンプトを再構築するのではなく、キャッシュされたプロンプトのプレフィックスを再利用します。そのキャッシュされたエージェントはセッションの完全なトランスクリプトも保持します — ツールの出力も含まれ、ツール呼び出しが 100 回あるセッションでは数十 MB になります。そのため、ビジーなマルチプラットフォームのゲートウェイでは、キャッシュがプロセス内で単独で最大のメモリ消費元になります。

```yaml
agent:
  agent_cache:
    max_size: 128            # LRU entry cap
    idle_ttl_secs: 3600      # evict an agent idle this long
    memory_high_mb: auto     # anon-RSS budget; number, "auto", or 0/off
    max_evictions_per_pass: 16
    protect_recent: 8
```

`max_size` と `idle_ttl_secs` は、キャッシュを件数と時間で制限します。どちらも保持しているバイト数はわからないため、`memory_high_mb` が 3 つ目の制限を追加します: 匿名メモリが予算を超えると、最も長く使われていないトランスクリプトを手放します。それらは次のターンで保存済みのセッションから再読み込みされます。ゲートウェイが他のサービスとメモリを奪い合っている場合は下げてください。すべてのプレフィックスを温かいまま保ちたい場合は引き上げる（または `0` を設定してこの処理をオフにする）ことができます。

`auto` は、ゲートウェイが実際に動作しているメモリ上限 — コンテナや systemd ユニットでは cgroup の上限、それ以外では総 RAM — から予算を導出するため、ユニットの `MemoryMax`/`MemoryHigh` は、同期を保つための 2 つ目の数値なしに尊重されます。そのような上限の下では、測定も同じ範囲で行われます: cgroup 自身の匿名メモリの課金（`memory.stat` の `anon`）であり、ユニットの上限に対してカウントされる `execute_code` のカーネルやターミナルコマンドなどの子プロセスも含まれます。上限がない場合は、ゲートウェイ自身の匿名 RSS が測定されます。

ターンの途中にあるセッション、最も最近使われた `protect_recent` 件のセッション、そしてトランスクリプトのディスクへの書き込みが完了していないセッションが手放されることはありません。退避は、測定された RSS と破棄されたセッションとともに WARNING でログに記録されます。

```
Agent cache pressure: anon RSS 6802MB over budget 6656MB — evicting 5 LRU session(s): ...
```

## コンテキストエンジン

コンテキストエンジンは、モデルのトークン上限に近づいたときに会話をどのように管理するかを制御します。組み込みの `compressor` エンジンは、損失のある要約を使用します（[コンテキスト圧縮](/docs/developer-guide/context-compression-and-caching)を参照）。プラグインエンジンは、これを代替戦略で置き換えられます。

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
```

プラグインエンジン（例: ロスレスなコンテキスト管理のための LCM）を使用するには:

```yaml
context:
  engine: "lcm"          # must match the plugin's name
```

プラグインエンジンは **決して自動起動されません** — `context.engine` をプラグイン名に明示的に設定する必要があります。利用可能なエンジンは `hermes plugins` → Provider Plugins → Context Engine から閲覧・選択できます。

メモリプラグインの類似の単一選択システムについては、[メモリプロバイダー](./features/memory-providers.md)を参照してください。

## イテレーション予算

エージェントが多くのツール呼び出しを伴う複雑なタスクに取り組んでいるとき、イテレーション予算（デフォルト: 500 ターン）を使い切ってしまうことがあります。Hermes はタスクの途中でプレッシャーの警告を注入**しません** — 以前のビルドは予算の 70%/90% でモデルに警告していましたが、それによってモデルが複雑なタスクを早々に放棄してしまうため、2026 年 4 月に削除されました。

代わりに、予算を実際に使い切ったとき（500/500）、Hermes はモデルに作業のまとめを求めるメッセージを 1 つ注入し、最終的なレスポンスを返せるように 1 回の**猶予呼び出し**を許可します。その猶予呼び出しでもテキストが生成されない場合、エージェントは達成した内容を要約するよう求められます。

```yaml
agent:
  max_turns: none              # Iterations per conversation turn (default: none = unlimited)
                               # Set a positive integer to cap; "none"/"null"/
                               # "unlimited"/"inf"/"infinity"/"infinite"/0/-1 = no limit
  budget_warning_ratio: null   # Optional one-time checkpoint warning, e.g. 0.75
  api_max_retries: 3           # Retries per provider before fallback engages (default: 3)
  auto_recovery_cycles: 5      # Wait-and-retry cycles after retries + fallback are spent on an outage (0 = off)
```

`agent.max_turns` は**デフォルトで無制限**です — ターン数の上限は、解決するよりも多くの問題（タスク途中の暗黙の打ち切り）を引き起こしていたため、そのままの状態では Hermes は会話のターンを完了まで実行します。上限を課すには、正の整数を設定します。「制限なし」を明示するには、大文字小文字を区別しない次のいずれかの表記が使えます: `"none"`、`"null"`、`"unlimited"`、`"infinite"`、`"infinity"`、`"inf"`、`0`、`-1`（これらは `sys.maxsize` の番兵値に解決されるため、ループがターン数で終了することはありません）。

`agent.budget_warning_ratio` は、通常の会話と委譲された会話ではデフォルトでオフです。有限の `max_turns` とともに `0` より大きく `1` より小さい値に設定すると、Hermes は閾値に達した後、最新のツール結果に、モデルから見えるチェックポイントの通知を 1 つ追加します。この通知は会話のターンごとに再設定され、各エージェント自身のイテレーション予算を使用します。追加されるのは現在のツール結果の末尾だけで、古いターンには追加されず、合成のユーザー/システムメッセージを追加したり、既存の使い切り時の猶予呼び出しを変更したりすることもありません。ディスパッチャーが所有する Kanban ワーカーは、ツールがまだ使える間に、デフォルトで 90% の時点で完了のチェックポイントを受け取ります（明示的な比率を指定するとその閾値が変わります）。このチェックポイントは、早まった成功ではなく、検証済みの完了または永続的な進捗コメントを求めます。

`agent.api_max_retries` は、フォールバックプロバイダーへの切り替えが作動する **前** に、一時的なエラー（レート制限、接続切断、5xx）でプロバイダーの API 呼び出しを Hermes が何回リトライするかを制御します。デフォルトは `3` — 合計 4 回の試行です。[フォールバックプロバイダー](./features/fallback-providers.md)を設定していて、より早くフェイルオーバーしたい場合は、これを `0` に下げると、プライマリでの最初の一時的なエラーが、不安定なエンドポイントに対してリトライを繰り返すのではなく、すぐにフォールバックへ引き渡されます。

リセット時刻を示すレート制限（`Retry-After`、またはボディの `retry_after` フィールド）は、リトライごとに最大 600 秒まで待機されます。ライブのステータス行には、待ち時間と、プロバイダーが示している場合は制限がリセットされる時刻が表示されます。例外が 1 つあります: 無料の Nous ティア（サインインなし）では、誰かが見ているセッション（Desktop アプリ、TUI、`hermes` ターミナルチャット、ACP エディター）は 60 秒より長いクールダウンを待ちません。無料ティアの一時停止はサービス全体に及ぶため、リセット前のリトライは同じ拒否に遭うだけだからです。ターンは直ちに終了し、制限がいつ解除されるかを伝えます。フォールバックが設定されていれば、先にそれが実行されます。無人の実行（`hermes chat -q`、cron、メッセージングプラットフォーム、委譲されたサブエージェント）は待ち続けます。

`agent.auto_recovery_cycles` は、リトライとフォールバックチェーンの両方を使い切った*後*のセーフティネットです。失敗が一時的な障害（HTTP 5xx、`overloaded`/529 のレスポンス、接続または読み取りのタイムアウト）で、まだ回答のテキストがあなたに届いていない場合、Hermes は「API failed after N retries」でターンを終了せず、待ってから再試行します。最大でこのサイクル数（デフォルト `5`）まで、ジッター付きの 15/30/60/60/60 秒のスケジュールで行います。プロバイダーの `Retry-After` ヘッダーはスケジュールより優先されます（最大 120 秒まで尊重）。待機中はすべてのサーフェスで同じ行が表示されます — CLI/TUI/Desktop では `⏳ Provider temporarily unavailable — retrying automatically in 30s (cycle 2/5); press Esc to stop`、メッセージングプラットフォームではステータスの吹き出し（`send /stop to cancel`）、API サーバーでは `hermes.status` の SSE イベント、cron ジョブではログ行です。Esc（または `/stop`）を押すと、待機は直ちにキャンセルされます。フォールバックが引き続き先に来ます: フォールバックチェーンが設定されている場合、使い切ると以前と同様に次のプロバイダーへ移り、この段階的な再試行はチェーンに何も残っていなくなって初めて作動します。認証、課金、リクエスト形式、利用資格、コンテンツポリシー、アカウントポリシーのエラーがこの段階的な再試行に入ることはありません。無効にするには `0` を設定します。

## 実時間の実行予算

イテレーション予算とは別に、各会話の実行にオプションの**実時間**予算を与えることができます。これは、厳しい外部の上限（例: タスクごとに 900 秒の制限）の下で動作する、ワンショットや評価ハーネスでの呼び出し向けに設計されています: これがないと、作業はほぼ終わっているのに実行がタイムアウトすることがあります — 最終回答を出力するまであと 1 回の生成というところで、あるいはハングした 1 回のプロバイダー呼び出しで止まったまま。

```yaml
agent:
  run_budget_seconds: null     # Optional; unset/null = feature fully off (default)
```

または CLI で呼び出しごとに指定します。

```bash
hermes chat --run-budget 850 -q "..."
```

予算が設定されると、2 つのことが起こります。

1. **80% での締めくくりの通知。** 予算の 80% が経過すると、Hermes は**1 回限り**の通知を注入し（`/steer` メッセージのように最新のツール結果に追加され、キャッシュに安全な方法で届けられます）、新しい探索/検証の作業をやめて、すでに持っている状態から最終的な成果物を作るようモデルに伝えます。これは実行ごとに最大 1 回発火し、既存のイテレーション予算の締めくくりの仕組みを踏襲しています — 繰り返されるプレッシャーの警告はありません。
2. **期限に応じてスケールするスタールタイムアウト。** 暗黙の非ストリーミングのスタールタイムアウト（90 秒のデフォルトと、DeepSeek の推論モデルの 600 秒などの推論モデルの下限）は、`max(60, remaining_budget × 0.5)` で上限が設けられるため、無言でハングした 1 回のプロバイダー呼び出しが実行の残りをすべて消費することはありません。この上限はタイムアウトを*厳しくする*だけで、引き上げることはありません。また、明示的に設定された `stale_timeout_seconds`（プロバイダー/モデルの設定または `HERMES_API_CALL_STALE_TIMEOUT`）は常に手を加えられずに優先されます。

予算は `run_conversation` のターンごと（ユーザーメッセージごとにリセット）であり、この機能は未設定の場合まったく休止しています — クロックの読み取りも、注入も、タイムアウトの変更もありません。


## 停止時の検証（Verify-on-Stop、コーディングの検証）

有効にすると、エージェントがワークスペース内のコードを編集したにもかかわらず新しい検証の証拠（成功したテスト実行、ビルド、lint など）を生成しなかったターンでは、Hermes は最終回答の受け入れを拒否します — 検証するか、できない理由を説明するようエージェントに求める合成のフォローアップを注入します。ドキュメント/Markdown/スキルだけの編集でこれが発火することはなく、ループには上限があるため、エージェントを閉じ込めることはありません。

```yaml
agent:
  verify_on_stop: false        # true | false | "auto" (surface-aware: on for CLI/TUI/desktop, off for messaging)
  verify_guidance: true        # Append creative-UI / clean-diff guidance to the missing-evidence nudge
  max_verify_nudges: 3         # Cap on consecutive continue nudges per turn (built-in + pre_verify hooks)
  coding_instructions: ""      # Standing project-wide coding rules appended to the coding brief
```

`verify_on_stop` は `true`（どこでもオン）、`false`（オフ — デフォルト）、または `"auto"`（レガシーのサーフェス対応の動作: 対話的なコーディングのサーフェス — CLI、TUI、デスクトップ — とプログラムからの呼び出し元ではオン、検証の説明がチャットのノイズとして読まれてしまう Telegram/Discord などのメッセージングのサーフェスではオフ）を受け付けます。どこでもデフォルトはオフです: 新規インストールは `false` で出荷され、設定の移行によって既存のインストールでもオフにされたため、有効にするには明示的なオプトインが必要です。`HERMES_VERIFY_ON_STOP` 環境変数は、設定されている場合に設定値をオーバーライドします。

このガードに供給される証拠（どのテスト/lint/ビルドのコマンドが実行されたか、その後にどのファイルが編集されたか）は `~/.hermes/verification_evidence.db` にあります。この台帳は、ガードが有効な間にのみ書き込まれたり作成されたりします。`verify_on_stop: false` では何も記録されず、既存のファイルは自由に削除できます。

同じ地点でのユーザー/プラグインのポリシーゲート — 独自のチェックでエージェントに作業を続けさせる — については、[`pre_verify` フック](./features/hooks.md#pre_verify)を参照してください。

## 継続的な目標（`/goal`）

継続的な目標がアクティブな場合、Hermes はアシスタントの各レスポンスがそれを満たしているかどうかを判定します。満たしていない場合は、継続のプロンプトを同じセッションに戻し、目標が達成されるか、ターンの予算を使い切るか、ユーザーが一時停止/クリアするまで作業を続けます。ターンの予算こそが本当の歯止めです — 判定の失敗は**オープン**（続行）側に倒れるため、不安定な判定が進捗を止めてしまうことはありません。

```yaml
goals:
  max_turns: 20   # Max continuation turns before Hermes auto-pauses the goal (default: 20)
```

`max_turns` は、Hermes が目標を自動的に一時停止してユーザーに `/goal resume` を求めるまでに、目標が駆動できる継続ターンの数を制限します。これは判定の偽陰性（目標は実際には達成されているのに判定が続行と言う）や、曖昧または達成不可能な目標に対するモデルの際限のない支出から保護します。機能の全体については[目標](./features/goals.md)を参照してください。

### API タイムアウト

Hermes には、ストリーミング用に別々のタイムアウト層があり、加えて非ストリーミング呼び出し用のスタール検出器があります。スタール検出器は、暗黙のデフォルトのままにした場合に限り、ローカルプロバイダー向けに自動調整されます。

外部プロセスのプロバイダー（Claude サブスクリプションの DirectSDK プラグインや ACP エージェントなど、CLI の子プロセスを通じて到達するモデル）は、すべてのモデルについて下表の「ローカルプロバイダー」列と同じ扱いになります。子プロセスはモデルの最初のトークンまで何も送らないためです。例外が 1 つあります: 外部プロセスへの非ストリーミング呼び出しは、無制限のままにされるのではなくリクエストタイムアウト（デフォルト 1800 秒）で制限されるため、入力の読み取りをやめた子プロセスが Hermes をハングさせることはありません。それ以外の場面でローカルサーバーとして扱われることはありません: コンテキストの拒否は圧縮して再試行し、Hermes がそれらに対して Ollama や llama.cpp かどうかをプローブすることはありません。

| タイムアウト | デフォルト | ローカルプロバイダー | Config / 環境変数 |
|---------|---------|----------------|--------------|
| ソケット読み取りタイムアウト | 120 秒 | 1800 秒に自動引き上げ | `HERMES_STREAM_READ_TIMEOUT` |
| スタールストリーム検出 | 180 秒 | 900 秒の上限に引き上げ（`agent.local_stream_stale_timeout`） | `HERMES_STREAM_STALE_TIMEOUT` |
| ストリームの最大存続時間（オプトイン） | オフ（`0`） | 同じ | `agent.stream_max_lifetime` |
| スタール非ストリーム検出 | 90 秒 | 暗黙のままなら自動無効化 | `providers.<id>.stale_timeout_seconds` または `HERMES_API_CALL_STALE_TIMEOUT` |
| Responses の最初のイベントのウォッチドッグ | 120 秒 | 900 秒の上限に引き上げ（`agent.local_stream_stale_timeout`） | `HERMES_CODEX_TTFB_TIMEOUT_SECONDS` |
| API 呼び出し（非ストリーミング） | 1800 秒 | 変更なし | `providers.<id>.request_timeout_seconds` / `timeout_seconds` または `HERMES_API_TIMEOUT` |
| 終端後のストリームのドレイン（Codex/Responses） | 2 秒 | 変更なし | `agent.stream_drain_timeout` |

**ソケット読み取りタイムアウト** は、httpx がプロバイダーからの次のデータチャンクをどれだけ待つかを制御します。ローカル LLM は、最初のトークンを生成する前に、大きなコンテキストのプリフィルに数分かかることがあるため、Hermes はローカルエンドポイントを検出すると、これを 30 分に引き上げます。`HERMES_STREAM_READ_TIMEOUT` を明示的に設定すると、エンドポイント検出にかかわらずその値が常に使用されます。

**スタールストリーム検出** は、SSE のキープアライブ ping を受信するが実際のコンテンツがない接続を切断します。ローカルプロバイダー（プリフィル中にキープアライブ ping を送信しない）では、デフォルトは 180 秒の基本値ではなく有限の 900 秒の上限に引き上げられます — `agent.local_stream_stale_timeout` または `HERMES_LOCAL_STREAM_STALE_TIMEOUT` 環境変数で設定できます。

**Responses の最初のイベントのウォッチドッグ**（Codex / `codex_responses` トランスポート。Responses トランスポートで宣言されたカスタムプロバイダーを含む）は、接続を受け入れたものの 120 秒以内にストリームイベントを出さないリクエストを中断して再接続します。大きなコンテキストをプリフィルしているローカルサーバーは正当にそれより長く無言のままになるため、ローカルエンドポイントでは暗黙のデフォルトがスタールストリーム検出器と同じ上限（`agent.local_stream_stale_timeout` / `HERMES_LOCAL_STREAM_STALE_TIMEOUT`、900 秒）に引き上げられます。明示的な `HERMES_CODEX_TTFB_TIMEOUT_SECONDS` は常にそのまま使われます（`0` でウォッチドッグを無効化）。

**ストリームの最大存続時間** は、1 つのストリーミングレスポンスの合計の実時間に対するオプトインの上限です。スタール検出器はチャンクのたびにリセットされるため、データを送り続けるが決して終わらないストリーム（チャンクを少しずつ流すゲートウェイ、暴走した生成）が打ち切られることはありません。そのようなストリームを打ち切るには `agent.stream_max_lifetime`（秒）を設定します。接続はスタールによる切断と同じ方法で切断され、再試行または部分的なレスポンスの継続が引き継ぎます。実効的なスタールストリームのタイムアウトより前に発火することはなく、プロバイダーは応答していたため、打ち切られた試行はスタールストリームのサーキットブレーカー（`HERMES_STREAM_STALE_GIVEUP`）にはカウントされません。この上限は Chat Completions と Anthropic Messages のストリームのみが対象です。Codex/Responses のストリーム（`codex_responses`。Responses トランスポートで宣言されたカスタムプロバイダーを含む）と Bedrock のストリームには上限が設けられません。正当に長時間ストリーミングする長い推論ターンを実行する場合は、`0`（デフォルト）のままにしてください。

**スタール非ストリーム検出** は、長すぎる時間レスポンスを生成しない非ストリーミング呼び出しを切断します。デフォルトでは、Hermes は長いプリフィル中の誤検出を避けるために、ローカルエンドポイントでこれを無効化します。`providers.<id>.stale_timeout_seconds`、`providers.<id>.models.<model>.stale_timeout_seconds`、または `HERMES_API_CALL_STALE_TIMEOUT` を明示的に設定すると、その明示的な値は、ローカルエンドポイントでも 50k トークン超でもそのまま使われます: コンテキストサイズによる引き上げ（50k トークン超で 150 秒、100k 超で 240 秒）は暗黙のデフォルトにのみ適用されます。

**終端後のストリームのドレイン** は、Codex/Responses のストリームが終端の `response.completed` フレームの後にどれだけ読み取りを続けるかを制限します（リレーのファイナライザーを実行させるための配慮です）。一部のリレーは終端フレームの後も SSE ソケットを閉じません。制限がないと、ターンはスタールストリームのウォッチドッグが発火するまで止まり、すでに課金されたレスポンスを破棄してから再試行していました。`agent.stream_drain_timeout` 秒が経過するとストリームは閉じられ、完了したレスポンスが返されます。接続を通常どおり閉じるエンドポイントはドレインを即座に終えるため、これほど長く待つことはありません。ドレインを完全にスキップするには `0` を設定します。

この予算はすべての非ストリーミング呼び出しを制限します。リクエストを受け入れた後に無言になるプロバイダー — 接続を開いたまま、バイトもエラーもない — は、はるかに長いソケット読み取りタイムアウトまで（あるいは無人の cron 実行では外部の何かがプロセスを終了させるまで）ハングするのではなく、スタールタイムアウトで中断されて再試行されます。

定期的なプロバイダー待ちの通知は、少なくとも **60 秒間の無言**の後にのみ表示されます。Codex Responses の**待機ステータス**は、生成にかかった合計時間ではなく無言の状態を表します: アクティブなストリームイベント（推論を含む）がある間は表示されません。イベントが止まると、レスポンスがまだ届いていないと主張するのではなく、ストリームイベントのない時間を報告します。イベントが再開すると通知はクリアされます。再接続によって新しい最初のイベントのウォッチドッグのフェーズが始まると、待機ステータスはそのフェーズに従います。この表示の動作は、別の実時間のスタールコールの予算を延長したり、ウォッチドッグのタイムアウトを変更したりするものではありません。ステータスは無言の期間ごとに 1 回（60 秒後）、待機のフェーズ（`waiting for the first provider event` と `provider stream active; Ns without stream events`）と、再接続を行うウォッチドッグ（`TTFB`、`stream idle`、または `wall-clock stale`）、それが発火するまでの残り秒数を示す中立的な文言で表示されます。書き換えられるのはフェーズが変わったときかその期限が近づいたときだけで、30 秒ごとの生存確認のハートビートのたびではありません。Chat Completions のストリームも同じルールに従い（`waiting for the first stream chunk` / `stream open; Ns without stream output`、`stream stale` ウォッチドッグ）、同様にチャンクが再開すると無言の通知を速やかにクリアします。ローカルのモデル読み込みのステータスを置き換えることはありません。

cron ジョブと委譲されたサブエージェントもストリーミングします。これらはリクエストを自身のスレッドでインラインに実行します（他のセッションが使う割り込みワーカーは、ゲートウェイの入れ子のスレッドプール内で止まってしまうためです）が、ワイヤー上のリクエストは引き続き `stream: true` であるため、上記の**スタールストリーム検出**の予算がそれらを制御します — すべてのトークンが生存の証としてカウントされるため、数分間思考する推論モデルがハングしたプロバイダーと誤認されることはなく、無言の接続を切断するエッジプロキシにもバイトが届き続けます。

### API ストリーミングの無効化

`model.streaming: false` は、セッション全体 — 親とサブエージェントの両方 — で非ストリーミングのリクエストを強制します。これは、*ストリーミング*のツール呼び出し経路が壊れているセルフホストの OpenAI 互換サーバーのための避難口です（たとえば、`--tool-call-parser qwen3_xml` と推論パーサーを併用した vLLM では、ツール呼び出しのマークアップがプレーンテキストに漏れ出し、`tool_calls` が 0 件で返されるため、委譲されたタスクが暗黙のうちに何もしなくなります）。デフォルトは `true` です。非ストリーミングの呼び出しでは上述の生存確認の特性が失われるため、その種のバグに遭遇しない限りそのままにしてください。これは、ターミナルでのトークンの描画だけを制御する `display.streaming` とは別物です。

Hermes は、ストリーミングで進捗が得られない場合にも、自らセッションを非ストリーミングに切り替えます: プロバイダーがストリーミングはサポートされていないと報告した場合や、OpenAI 互換のゲートウェイがストリーミングのリクエストに中身のない SSE フレーム（劣化したリレーに典型的な、ペイロードのない素の `data:` / `event: ping` のキープアライブ）で応答した場合です。ターンはストリーミングなしで再試行され、警告が表示され、そのセッションの残りの間ストリーミングはオフのままになります。

```yaml
model:
  streaming: false
```

## コンテキストプレッシャーの警告

イテレーション予算のプレッシャーとは別に、コンテキストプレッシャーは会話が **圧縮閾値** にどれだけ近いか — 古いメッセージを要約するためにコンテキスト圧縮が発火する地点 — を追跡します。これは、会話が長くなってきたタイミングを、あなたとエージェントの両方が理解するのに役立ちます。

| 進捗 | レベル | 何が起きるか |
|----------|-------|-------------|
| 閾値まで **60% 以上** | 情報 | CLI はシアン色の進捗バーを表示し、ゲートウェイは情報通知を送信 |
| 閾値まで **85% 以上** | 警告 | CLI は太字の黄色いバーを表示し、ゲートウェイは圧縮が間近であることを警告 |

CLI では、コンテキストプレッシャーはツール出力フィードに進捗バーとして表示されます。

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

メッセージングプラットフォームでは、プレーンテキストの通知が送信されます。

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

自動圧縮が無効な場合、警告は代わりにコンテキストが切り詰められる可能性があることを伝えます。

コンテキストプレッシャーは自動です — 設定は不要です。これは純粋にユーザー向けの通知として発火し、メッセージストリームを変更したり、モデルのコンテキストに何かを注入したりはしません。

## 認証情報プールの戦略 {#credential-pool-strategies}

同じプロバイダーに対して複数の API キーや OAuth トークンを持っている場合、ローテーション戦略を設定します。

```yaml
credential_pool_strategies:
  openrouter: round_robin    # cycle through keys evenly
  anthropic: least_used      # always pick the least-used key
```

オプション: `fill_first`（デフォルト）、`round_robin`、`least_used`、`random`。完全なドキュメントは[認証情報プール](./features/credential-pools.md)を参照してください。

## プロンプトキャッシュ {#prompt-caching}

Hermes は、アクティブなプロバイダーがサポートしている場合、セッションをまたぐプロンプトキャッシュを自動的に有効にします — ユーザーによる設定は不要です。

**ネイティブ Anthropic**、**OpenRouter**、**Nous Portal** 上の Claude では、Hermes はシステムプロンプトとスキルのブロックに 1 時間の TTL（`ttl: "1h"`）を持つ `cache_control` のブレークポイントを付けます。新しい 1 時間の中での最初の送信は通常の入力料金を支払いますが、同じ 1 時間以内のその後の送信は、どのセッションであっても、割引されたキャッシュ読み取り料金でキャッシュから取得されます。つまり、システムプロンプト、読み込まれたスキルの内容、そして長いコンテキストのインクルードの前半部分が、最初の 1 時間は `hermes` のセッション間やフォークされたサブエージェント間で再利用されます。

Qwen Cloud（Alibaba DashScope）の上流はキャッシュの TTL を 5 分に制限しているため、Hermes はそこでは代わりに 5 分のブレークポイント TTL を使います。その他のサードパーティ経由の Claude の経路（AWS Bedrock、Azure Foundry）は、プロバイダー自身のキャッシュのデフォルトにフォールバックします。xAI Grok は、別の、セッションに固定された会話 ID の仕組みを使います — [xAI のプロンプトキャッシュ](../integrations/providers.md#xai-grok--responses-api--prompt-caching)を参照してください。

これを無効にするつまみはありません — キャッシュは常に有効で、システムプロンプトだけでも入力トークン数のかなりの割合を占めるため、単一ターンの会話でもコストを節約できます。

唯一の明示的なつまみは、Hermes が Anthropic 形式のブレークポイントで要求するキャッシュ TTL の段階です。

```yaml
prompt_caching:
  cache_ttl: "5m"   # "5m", "1h" (Anthropic-supported tiers) or "auto"; other values are ignored
```

`cache_ttl` は、ネイティブの Anthropic API、OpenRouter、Nous Portal 経由の Claude に Hermes が付けるブレークポイントの TTL を選択します。2 つの Anthropic の段階（`"5m"`、`"1h"`）はそのまま送信され、それ以外の値は無視されます。独自の上限を持つプロバイダー（例: 最大 5 分の Qwen Cloud）は、引き続き上流が許す範囲に制限されます。

1h の段階は基本の入力料金の 2 倍で書き込まれ（5m の書き込みは 1.25 倍）、ターンの間隔が 5 分より長い場合にのみ元が取れます — そうでない場合、すべてのツール結果が、誰も使わない保持期間のために割高な料金で書き込まれます。`"auto"` は、セッションのペースを誰が決めるかによってセッションごとに段階を選びます: 人が入力するセッション（CLI、TUI、Desktop、Telegram/Discord/Slack などのメッセージングプラットフォーム）には `1h`、機械がペースを決めるセッション（サブエージェント、cron、`hermes -q` のワンショット、Webhook、Kanban ワーカー、API サーバー、ツールから呼び出される実行やバッチ実行）には `5m` です。対話的なセッションを 1 日を通じて中断・再開するインストールでは、`auto` によって対話的なキャッシュ書き込みの費用が約 40% 削減され、ファンアウトするサブエージェントの支出は変わりませんでした。委譲されたサブエージェントは、設定にかかわらず常に `5m` に制限されます。

## 補助モデル {#auxiliary-models}

Hermes は、画像分析、ブラウザのスクリーンショット分析、セッションタイトルの生成、コンテキスト圧縮などのサイドタスクに「補助」モデルを使用します。デフォルト（`auxiliary.*.provider: "auto"`）では、Hermes はすべての補助タスクを **メインのチャットモデル** — `hermes model` で選んだのと同じプロバイダー/モデル — にルーティングします。始めるにあたって何も設定する必要はありませんが、高価な推論モデル（Opus、MiniMax M2.7 など）では補助タスクが相応のコストを追加することに注意してください。メインのモデルにかかわらず安価で高速なサイドタスクが欲しい場合は、`auxiliary.<task>.provider` と `auxiliary.<task>.model` を明示的に設定してください（例: ビジョンに OpenRouter 上の Gemini Flash）。（Web の抽出は補助タスクではありません: `web_extract` とブラウザのスナップショットは長いコンテンツを決定論的に切り詰め、`read_file` でページングできるように全文を保存します — LLM は関与しません。）

:::note なぜ "auto" がメインモデルを使うのか
以前のビルドは、アグリゲーターのユーザー（OpenRouter、Nous Portal）を、安価なプロバイダー側のデフォルトに振り分けていました。これは意外なものでした — アグリゲーターのサブスクリプションに料金を払ったユーザーが、補助トラフィックを処理する別のモデルを目にすることになっていたためです。`auto` は今ではすべての人にとってメインのモデルを使用し、`config.yaml` のタスクごとのオーバーライドは依然として優先されます（後述の[補助設定の完全なリファレンス](#full-auxiliary-config-reference)を参照）。
:::


### 補助モデルを対話的に設定する

YAML を手で編集する代わりに、`hermes model` を実行してメニューから **「Configure auxiliary models」** を選んでください。タスクごとの対話的なピッカーが表示されます。

```
$ hermes model
→ Configure auxiliary models

[ ] vision               currently: auto / main model
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] tts_audio_tags       currently: auto / main model
[ ] voice_chat           currently: auto / main model
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
[ ] triage_specifier     currently: auto / main model
[ ] kanban_decomposer    currently: auto / main model
[ ] profile_describer    currently: auto / main model
[ ] delegation           currently: auto / inherit main agent
```

タスクを選び、プロバイダーを選び（OAuth フローはブラウザを開きます。API キーのプロバイダーは入力を促します）、モデルを選びます。変更は `config.yaml` の `auxiliary.<task>.*` に永続化されます。メインモデルのピッカーと同じ仕組みで — 覚えるべき追加の構文はありません。

**Delegation** のエントリは特別です: これは `delegate_task` のサブエージェントが使うモデルをルーティングし、`auxiliary.*` ではなくトップレベルの `delegation.*` セクション（`delegation.provider` / `delegation.model`）に永続化されます。サブエージェントはサイドの LLM 呼び出しではなく、完全な子エージェントだからです。その `auto` は「親エージェントのプロバイダー、モデル、認証情報を継承する」という意味です。

**Voice chat** のエントリは、音声モードの発話によるターン（ツールを含む）に応答するモデルを選びます。入力したターンはメインのモデルのままです。[音声チャットのモデル](features/voice-mode.md#voice-chat-model)を参照してください。

最初のやり取りの後に Hermes にタイトルを自動生成させたくない場合は、
`auxiliary.title_generation.enabled: false` を設定します。手動のタイトルは引き続き
`/title` と `hermes sessions rename` で機能します。

即座に導出されるタイトル（最初のメッセージの 1 行目）は維持しつつ、それを改善するために
モデル呼び出しを決して使わないようにするには、`auxiliary.title_generation.model_upgrade_enabled: false` を設定します。
バックグラウンドの `auto-title` スレッドは開始されず、自動のタイトルモデルへのリクエストも送信されません。
明示的な修復コマンド `hermes sessions retitle-skills` は引き続きモデルを呼び出します。`enabled: false`
は引き続き両方の段階を無効にします。

`custom` のメインプロバイダー（llama.cpp、Ollama、vLLM、LM Studio などのセルフホストの
OpenAI 互換サーバー）では、`auxiliary.title_generation` が別のプロバイダーや `base_url` に
固定されていない限り、タイトルモデルの呼び出しはターンの返信と並行してではなく、返信が
届いた**後**に送信されます。そうしないと、返信のデコード中に `json_schema` のタイトル
リクエストを受け取った単一スロットのローカルサーバーが、返信に `{"title": ...}` で応答してしまい、
それがアシスタントのターンとして保存・再生されることがあるためです。

Hermes Desktop では、3,000 文字を超えるプレーンテキストの貼り付けは、生成された `.txt` の
添付ファイルになります。その貼り付けの最初の約 1,000 文字は、タイトル専用のヒントとしてタイトルの
段階に渡されます（エージェントのターンには引き続き添付ファイルへの参照だけが見えます）。そのため、
「これを要約して」と大きな貼り付けを組み合わせた場合、貼り付けた話題にちなんだ名前が付きます。自分で
添付したファイルが、タイトル付けのために読まれることはありません。

ローカルのメッセージングゲートウェイでは、テキストメッセージは、チャンネルに紐付いたスキルと
プラットフォームのコンテキストが追加される前の元のリクエストを、セッションのタイトル付けに提供します。
メインのモデルと会話履歴には、引き続きスキルの内容全体が保持されます。
添付ファイルのみのターンでは、既存の拡充済みメッセージによるタイトルのフォールバックが維持されます。
これは新しいタイトル生成に影響するもので、以前に名前が付けられたセッションを修復するものではありません。

### ストリーミング専用のエンドポイント

一部の OpenAI 互換エンドポイントは、非ストリーミングのチャットリクエストを完全に拒否します（例: Tencent Copilot は HTTP 400 `"Non-stream chat request is currently not supported"` を返します）。対話的なチャットはすでにストリーミングしていますが、補助タスク（タイトル生成、圧縮、ビジョン）は非ストリーミングの呼び出しを使うため、毎回失敗してしまいます。Hermes は `copilot.tencent.com` を常にストリーミング専用として扱います。その他のそのようなエンドポイントについては、`auxiliary.stream_only_base_urls` の下に URL の部分文字列を列挙してください。

```yaml
auxiliary:
  stream_only_base_urls:
    - "my-stream-only-proxy.example.com"
```

一致する補助呼び出しは `stream=True` で送信され、チャンク（ツール呼び出しの差分を含む）はクライアント側で集約されます — 他のエンドポイントの動作は変わりません。

### 動画チュートリアル

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/NoF-YajElIM"
    title="Hermes Agent — Auxiliary Models Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

### 汎用的な設定パターン

Hermes のすべてのモデルスロット — 補助タスク、圧縮、フォールバック — は、同じ 3 つのつまみを使用します。

| キー | 役割 | デフォルト |
|-----|-------------|---------|
| `provider` | 認証とルーティングに使用するプロバイダー | `"auto"` |
| `model` | リクエストするモデル | プロバイダーのデフォルト |
| `base_url` | カスタムの OpenAI 互換エンドポイント（プロバイダーをオーバーライド） | 未設定 |

補助タスクのブロックは、さらに `reasoning_effort` のつまみも受け付けます。

| キー | 役割 | デフォルト |
|-----|-------------|---------|
| `reasoning_effort` | そのタスクの LLM 呼び出しの思考レベル: `none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`、`ultra` | 未設定（プロバイダーのデフォルト） |

これはグローバルな `agent.reasoning_effort` のタスクごとの対応物です: メインのモデルが高価な推論モデルである場合に、メインのチャットの動作には触れずに、圧縮を `low` で、ビジョンを `none` で実行して、サイドタスクのレイテンシとコストを削減できます。これは `vision`、`compression`、`title_generation`、`curator` などの補助クライアントのタスクに、3 つすべての補助ワイヤー形式（chat completions、Codex Responses、Anthropic Messages）で適用されます。同じタスクに明示的な `extra_body.reasoning` がある場合は、この省略形より優先されます。自身の呼び出しで思考をオフにする呼び出し元（タイトル生成がそうです — 64 トークンのタイトルには推論の余地がありません）は、その両方より優先されます: そのリクエストについては、プロバイダーの思考オフのフィールドと並べて送信されるのではなく、タスクレベルの労力が取り除かれます。

エンドポイントが推論のフィールドを完全に拒否する場合（OpenAI 互換のリレーの背後にあるチャット専用モデルが `400 Unrecognized request argument supplied: reasoning_effort`、または逆の文言の `400 reasoning_effort 'none' unsupported; use minimal|low|medium|high|xhigh` で応答する場合）、補助呼び出しはすべての推論フィールドを省略して 1 回再試行されるため、タスク（たとえばセッションのタイトル）はエンドポイントのデフォルトの動作で引き続き完了します。メインの会話にも同じ回復が適用されます: 思考のみで切り詰められた続きのために Hermes が送る推論オフのリクエストを経路が拒否した場合、そのセッションの残りでは無効化が取り除かれ、経路のデフォルトでリクエストが再試行されます。

一部のモデルは、思考をまったくオフにできません（`400 Reasoning is mandatory for this endpoint and cannot be disabled`）。それらについては、思考オフの補助呼び出し（タイトル生成、または `reasoning_effort: none` に設定された任意のタスク）は、無効化の代わりに最低の労力（`low`）で送信されます。Hermes は、経路のモデルカタログがモデルを必須とマークしている場合（OpenRouter と Nous Portal の `/v1/models`。`cache/reasoning_caps.json` にキャッシュ）や、同じプロセス内で経路が以前の無効化にすでにそのように応答している場合に、それを事前に把握します。そのため、拒否されるリクエストは送信されません。キャッシュされたカタログのない新規インストールでは、その 400 が 1 回だけ見えることがあります: 検索はバックグラウンドでカタログを取得し、以降の呼び出しはそれを使います。

**バックグラウンドレビューは異なります:** 同じモデルのレビューのフォークは、常に親の推論の労力を継承します。その経路では、親のプロバイダー/モデルが明示的に選択されている場合も含め、`auxiliary.background_review.reasoning_effort` は無視されます。これにより、プロンプトキャッシュの一致のために、推論の設定、システムプロンプト、会話の完全なスナップショット、ツール定義がバイト単位で同一に保たれます。同じモデルのレビューに独立した労力を設定するスイッチはありません。[バックグラウンドレビューの推論](./features/memory.md#same-model-review-reasoning)を参照してください。レビューが別のプロバイダー/モデルにルーティングされる場合、`reasoning_effort` はそのルーティングされたフォークに適用されます（未設定 = ルーティング先のプロバイダーのデフォルト）。キーが設定されているのにレビューがメインのモデルで実行される場合、Hermes は 1 回限りの警告を表示します。

**MoA も異なる設定を使います:** Mixture-of-Agents の推論の深さは、`moa_reference`/`moa_aggregator` の補助ブロックではなく、MoA のプリセットで**スロットごとに**設定します（`moa.presets.<name>.reference_models[].reasoning_effort` / `aggregator.reasoning_effort`）— [Mixture of Agents](/docs/user-guide/features/mixture-of-agents) を参照してください。

```yaml
auxiliary:
  compression:
    reasoning_effort: "low"    # summaries don't need deep thinking
  vision:
    reasoning_effort: "none"   # disable thinking for image description
```

`base_url` が設定されている場合、Hermes はプロバイダーを無視し、そのエンドポイントを直接呼び出します（認証には `api_key` を使用。フォールバックについては後述）。`provider` のみが設定されている場合、Hermes はそのプロバイダーの組み込みの認証とベース URL を使用します。

補助タスクで利用可能なプロバイダー: `auto`、`main`、加えて[プロバイダーレジストリ](../reference/environment-variables.md)の任意のプロバイダー — `openrouter`、`nous`、`openai-codex`、`anthropic`、`gemini`、`qwen-oauth`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`xai-oauth`、`ollama-cloud`、`alibaba`、`bedrock`、`huggingface`、`arcee`、`xiaomi`、`kilocode`、`opencode-zen`、`opencode-go`、`commandcode`、`commandcode-anthropic`、`ai-gateway`、`azure-foundry` — または `providers:` の dict にある任意の名前付きカスタムプロバイダー（例: `provider: "beans"`）。

ローカルの OpenAI 互換サーバーも独自の名前で動作します: `provider: ollama`（`vllm`、`llamacpp`、`llama.cpp` も同様）に `http://127.0.0.1:11434` のような `base_url` と空の `api_key` を指定すると、プレースホルダーのキーでカスタムエンドポイントを通じてルーティングされます。`host:port` だけの base_url には自動的に `/v1` のサフィックスが付きます。

`provider: openai` は直接 API のエイリアスです: ブロックの `base_url`、なければ `OPENAI_BASE_URL`、なければ `https://api.openai.com/v1` にあるカスタムエンドポイントを通じてルーティングし、`api_key`、なければ（エンドポイントが OpenAI のホスト、またはそれと組になっている `OPENAI_BASE_URL` とまったく同じ場合に）`OPENAI_API_KEY` で認証します。すべての補助タスクがこれを同じ方法で解決します — `compression`/`vision`/`title_generation` だけでなく、`background_review`、`curator`、MoA のスロットも同様です — そのため、`provider: openai` を残したまま `base_url` を削除すると、そのタスクは公開の OpenAI エンドポイントに移ります。`providers:` の dict にある `providers.openai` エントリは優先され、独自のエンドポイントとキーを保持します。

ルーティングされた `auxiliary.<task>` ブロックを解決できない場合（未知のプロバイダー、エンドポイントや認証情報の欠落）、タスクはメインのモデルで実行され、Hermes はその旨を伝えます: `background_review` はプロバイダーと理由を示す 1 回限りのユーザーに見える警告を出し（加えてレビューごとに `agent.log` に `WARNING` 行を出力）、`hermes doctor` はルーティングされたすべての `auxiliary.<task>` ブロックを同じリゾルバーで解決し、失敗したものを報告します。

:::tip MiniMax OAuth
`minimax-oauth` はブラウザ OAuth でログインします（API キー不要）。`hermes model` を実行し、**MiniMax (OAuth)** を選んで認証してください。補助タスクは自動的に `MiniMax-M2.7-highspeed` を使用します。[MiniMax OAuth ガイド](../guides/minimax-oauth.md)を参照してください。
:::

:::tip xAI Grok OAuth
`xai-oauth` は、SuperGrok および X Premium+ の購読者向けにブラウザ OAuth でログインします（API キー不要）。`hermes model` を実行し、**xAI Grok OAuth (SuperGrok / Premium+)** を選んで認証してください。同じ OAuth トークンは、xAI に直接接続するすべてのサーフェス（チャット、補助タスク、TTS、画像生成、動画生成、文字起こし）で再利用されます。[xAI Grok OAuth ガイド](/docs/guides/xai-grok-oauth)を参照してください。Hermes がリモートホスト上にある場合は、[SSH / リモートホスト経由の OAuth](/docs/guides/oauth-over-ssh) を参照してください。
:::

:::warning `"main"` は補助タスク専用
`"main"` プロバイダーオプションは「メインのエージェントが使うプロバイダーを何でも使う」という意味です — これは `auxiliary:`、`compression:`、およびプライマリのフォールバックのエントリ（`fallback_providers:` またはレガシーの `fallback_model:`）の内部でのみ有効です。トップレベルの `model.provider` 設定の値としては **有効ではありません**。カスタムの OpenAI 互換エンドポイントを使う場合は、`model:` セクションで `provider: custom` を設定してください。すべてのメインモデルのプロバイダーオプションについては [AI プロバイダー](../integrations/providers.md)を参照してください。
:::

### 補助設定の完全なリファレンス {#full-auxiliary-config-reference}

```yaml
auxiliary:
  # Image analysis (vision_analyze tool + browser screenshots)
  vision:
    provider: "auto"           # "auto", "openrouter", "nous", "codex", "main", etc.
    model: ""                  # e.g. "openai/gpt-4o", "google/gemini-2.5-flash"
    base_url: ""               # Custom OpenAI-compatible endpoint (overrides provider)
    api_key: ""                # API key for base_url (OPENAI_API_KEY fallback: OpenAI hosts only)
    timeout: 120               # seconds — LLM API call timeout; vision payloads need generous timeout
    download_timeout: 30       # seconds — image HTTP download; increase for slow connections
    max_concurrency: 8         # max concurrent image encode/resize bursts across the process
                               # (default: host CPU core count, no ceiling) — bounds only the
                               # CPU-bound encode step so a video-frame fan-out can't saturate
                               # every core and starve the event loop; LLM calls stay fully
                               # concurrent. Minimum 1; values < 1 are ignored.

  # Dangerous command approval classifier
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30                # seconds

  # Gemini 3.1 TTS hidden audio-tag insertion
  tts_audio_tags:
    provider: "auto"
    model: ""                  # empty = main chat model
    base_url: ""
    api_key: ""
    timeout: 30

  # Context compression timeout (separate from compression.* config)
  compression:
    timeout: 120               # seconds — compression summarizes long conversations, needs more time
    # no_progress_timeout: 60   # seconds a streamed summary may go without a substantive chunk
    #                           # before the attempt fails fast into retry/fallback
    # fallback_chain:           # Optional — providers to try on rate-limit / connectivity failure
    #   - provider: nous
    #     model: deepseek/deepseek-chat
    #   - provider: openrouter
    #     model: google/gemini-2.5-flash
    #     base_url: ""
    #     api_key: ""
    # max_concurrency: 2       # Optional: cap simultaneous compression LLM calls so
                               # multiple sessions don't pile retries on a degraded provider

  # Auto-generated session titles. Empty language follows the conversation;
  # set e.g. "English" or "Japanese" to pin titles to one language.
  title_generation:
    enabled: true              # set false to disable auto-title generation
    model_upgrade_enabled: true  # set false to keep the instant derived title, never call a model
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    language: ""

  # MCP tool dispatch
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # Auto-generated short session titles after the first exchange
  title_generation:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    # max_concurrency: 2       # Optional: cap simultaneous title-generation calls

  # Kanban triage specifier — `hermes kanban specify <id>` (or the
  # dashboard's ✨ Specify button on Triage-column cards) uses this
  # slot to expand a one-liner into a concrete spec and promote the
  # task to `todo`. Cheap fast models work well here; spec expansion
  # is short and doesn't need reasoning depth.
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
各補助タスクには設定可能な `timeout`（秒）があります。デフォルト: ビジョン 120 秒、承認 30 秒、圧縮 120 秒、タイトル生成 30 秒、その他のすべてのタスク 30 秒。補助タスクに低速なローカルモデルを使う場合は、これらを増やしてください — 回答の前に思考ブロックを出力する推論モデルは、タイトルのために日常的に 30 秒以上を必要とし、期限に達したリクエストは、Hermes がフォールバックチェーンを試す前に `Auxiliary <task>: request to <base_url> timed out after <N>s (raise auxiliary.<task>.timeout …)` としてログに記録されます。タイトル生成、圧縮、ビジョンは、1 回分のタイムアウトの時間枠が過ぎるとプライマリの経路を諦める（同じプロバイダーでの再試行はしない）ため、遅いモデルが待ち時間を何倍にも増やすことはありません。ビジョンには、HTTP での画像ダウンロード用の別の `download_timeout`（デフォルト 30 秒）もあります — 低速な接続やセルフホストの画像サーバーでは増やしてください。
:::

:::info
コンテキスト圧縮には、閾値のための独自の `compression:` ブロックと、モデル/プロバイダー設定のための `auxiliary.compression:` ブロックがあります — 前述の[コンテキスト圧縮](#context-compression)を参照してください。プライマリのフォールバックチェーンはトップレベルの `fallback_providers:` リストを使用します — [フォールバックプロバイダー](../integrations/providers.md#fallback-providers)を参照してください。3 つとも同じ provider/model/base_url のパターンに従います。
:::

### 補助タスクのタスクごとのフォールバックチェーン

各補助タスクは、オプションで `fallback_chain` を定義できます — プライマリの補助プロバイダーがレート制限、接続の問題、支払いの制限によって失敗したときに Hermes が試す、プロバイダー/モデルのエントリのリストです。

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openai/gpt-4o-mini
    fallback_chain:
      - provider: nous
        model: deepseek/deepseek-chat
      - provider: openrouter
        model: google/gemini-2.5-flash
```

プライマリの補助プロバイダー（`openrouter` / `openai/gpt-4o-mini`）がレート制限、接続タイムアウト、または支払いが必要というエラーを返すと、Hermes は `fallback_chain` を順に辿ります。すでに失敗したプロバイダーと一致するプロバイダーのエントリはスキップし、いずれかが成功するかチェーンを使い切るまで、残りの各エントリを試します。すべてのフォールバックが失敗した場合、Hermes は最終的なセーフティネットとしてメインのエージェントのモデルにフォールバックします。

各エントリは、任意の補助タスクの設定と同じ 3 つのつまみをサポートします。

| キー | 説明 |
|-----|-------------|
| `provider` | プロバイダー名（`nous`、`openrouter`、`anthropic`、`gemini`、`main` など） |
| `model` | そのプロバイダーのモデル名 |
| `base_url` | （オプション）カスタムの OpenAI 互換エンドポイント |

`fallback_chain` は任意の補助タスク — `compression`、`vision`、`approval`、`mcp` など — で利用できます。

### ネイティブのビジョン埋め込みの予算（トップレベルの `vision:`）

（記述用のモデルを選ぶ）`auxiliary.vision` とは別に: *メイン*のモデルがビジョン対応の場合、`vision_analyze` とブラウザのスクリーンショットは、実際のピクセルをツール結果に埋め込み、それは後続のすべてのターンで再送されます。`vision.embed_target_bytes`（デフォルト `262144`、64 KiB..4 MiB に制限）は 1 つの埋め込みの大きさを決めます。`vision.max_calls_per_image` は、セッションごとに同じ画像を埋め込める回数を制限します（未設定 = 委譲されたサブエージェント内では 3、メインのエージェントでは無制限。`0` = 無制限）。`vision.replay_recent_images`（デフォルト `3`、`0` = オフ）は、セッションデータベースから再構築されたターン（API サーバー、`--resume`）で、最新のユーザー画像を再送します。[ビジョン → ネイティブの埋め込みはセッションに乗る](./features/vision.md#native-embeds-ride-the-session-visionembed_target_bytes-and-visionmax_calls_per_image)を参照してください。

### 補助の並行性の制限

`max_concurrency` は、`compression` や `title_generation` などの補助タスクについて、プロセス全体での進行中の LLM 呼び出しを制限します。`auxiliary.vision.max_concurrency` は対象外です: これはすでに、LLM のリクエストではなく、ビジョンの CPU バウンドな画像のエンコード/リサイズのワーカーだけを制御しています。これは次の場合に最も役立ちます。

- 多くのセッションが同時にバックグラウンドの作業を生成しうる（Discord/Telegram のチャンネル、複数のターミナル）
- プロバイダーがレート制限されているか障害の最中で、再試行がバーストを増幅してしまう

デフォルトは無制限です。典型的な安全上の上限は `2` です。

```yaml
auxiliary:
  title_generation:
    max_concurrency: 2
  compression:
    max_concurrency: 2
```

セマフォは再試行とフォールバックを含む呼び出し全体をラップするため、1 回の遅い呼び出しが上限に対してカウントされるのは 1 回だけです。

### 補助タスク向けの OpenRouter ルーティングと Pareto Code {#openrouter-routing--pareto-code-for-auxiliary-tasks}

補助タスクが OpenRouter に解決される場合（明示的に、またはメインのエージェントが OpenRouter のときに `provider: "main"` を介して）、メインのエージェントの `provider_routing` と `openrouter.min_coding_score` の設定は **伝播しません** — 設計上、各補助タスクは独立しています。特定の補助タスクで OpenRouter のプロバイダー設定を行う、または [Pareto Code ルーター](../integrations/providers.md#openrouter-pareto-code-router)を使うには、`extra_body` を介してタスクごとに設定します。

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code         # use the Pareto Code router for this task
    extra_body:
      provider:                            # OpenRouter provider routing prefs
        order: [anthropic, google]         # try these providers in order
        sort: throughput                   # or "price" | "latency"
        # only: [anthropic]                # restrict to a specific provider
        # ignore: [deepinfra]              # exclude specific providers
      plugins:                             # OpenRouter Pareto Code router knob
        - id: pareto-router
          min_coding_score: 0.5            # 0.0–1.0; higher = stronger coders
```

この形は、OpenRouter がチャット補完リクエストボディで受け付けるものを反映しています。Hermes は `extra_body` 全体をそのまま転送するため、[openrouter.ai/docs](https://openrouter.ai/docs) に文書化されている他の OpenRouter リクエストボディフィールドも同じように機能します。

### ビジョンモデルの変更

画像分析に Gemini Flash の代わりに GPT-4o を使うには:

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

または環境変数経由（`~/.hermes/.env` 内）:

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### プロバイダーオプション

これらのオプションは **補助タスクの設定**（`auxiliary:`、`compression:`）とプライマリのフォールバックのエントリ（`fallback_providers:` またはレガシーの `fallback_model:`）に適用され、メインの `model.provider` 設定には適用されません。

| プロバイダー | 説明 | 要件 |
|----------|-------------|-------------|
| `"auto"` | 利用可能な最良（デフォルト）。ビジョンは OpenRouter → Nous → Codex を試す。 | — |
| `"openrouter"` | OpenRouter を強制 — 任意のモデル（Gemini、GPT-4o、Claude など）にルーティング | `OPENROUTER_API_KEY` |
| `"nous"` | Nous Portal を強制 | `hermes auth` |
| `"codex"` | Codex OAuth（ChatGPT アカウント）を強制。`model` を明示的に設定する（例: `gpt-5.4`）。 | `hermes model` → ChatGPT or Codex Subscription |
| `"minimax-oauth"` | MiniMax OAuth（ブラウザログイン、API キー不要）を強制。補助タスクに MiniMax-M2.7-highspeed を使用。 | `hermes model` → MiniMax (OAuth) |
| `"xai-oauth"` | xAI Grok OAuth（SuperGrok または X Premium+ の購読者向けのブラウザログイン、API キー不要）を強制。同じ OAuth トークンでチャット、TTS、画像、動画、文字起こしをカバー。 | `hermes model` → xAI Grok OAuth (SuperGrok / Premium+) |
| `"main"` | アクティブなカスタム/メインのエンドポイントを使用。これは `OPENAI_BASE_URL` + `OPENAI_API_KEY` から、または `hermes model` / `config.yaml` で保存されたカスタムエンドポイントから来ます。OpenAI、ローカルモデル、任意の OpenAI 互換 API で動作します。**補助タスク専用 — `model.provider` には無効。** | カスタムエンドポイントの認証情報 + ベース URL |

サイドタスクにデフォルトのルーターをバイパスさせたい場合、メインのプロバイダーカタログの直接 API キープロバイダーもここで動作します。たとえば、`GMI_API_KEY` を設定すれば `gmi` が、`FIREWORKS_API_KEY` を設定すれば `fireworks` が有効になります。

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

GMI の補助ルーティングでは、GMI の `/v1/models` エンドポイントが返す正確なモデル ID を使ってください。Fireworks のモデル ID は、プロバイダーのネイティブなスラッシュ形式を使います。たとえば `accounts/fireworks/models/glm-5p2` です。

### 一般的なセットアップ

**直接のカスタムエンドポイントを使う**（ローカル/セルフホストの API では `provider: "main"` より明確）:
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` は `provider` より優先されるため、これは補助タスクを特定のエンドポイントにルーティングする最も明示的な方法です。直接のエンドポイントオーバーライドでは、Hermes は設定された `api_key` を使用します。そのカスタムエンドポイントに `OPENROUTER_API_KEY` を再利用することはありません。それがない場合、メインのモデルのキーは、`base_url` がメインのエンドポイントとまったく同じオリジン（スキーム、ホスト、ポート）を持つ場合にのみ再利用されます。メインのエンドポイントとは、セッションが実行されているもの（`/model` で切り替えた後はそれ）であり、独自のキーと組になっていて、別のエンドポイントのキーと組になることはありません。`OPENAI_API_KEY` は最後のフォールバックで、OpenAI のホスト（`*.openai.com`、`*.openai.azure.com`）、またはそれと組になっている `OPENAI_BASE_URL` とまったく同じものにのみ送信されます。その他のエンドポイントには独自の `api_key` が必要です。

**ビジョンに OpenAI API キーを使う:**
```yaml
# In ~/.hermes/.env:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # or "gpt-4o-mini" for cheaper
```

**ビジョンに OpenRouter を使う**（任意のモデルにルーティング）:
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # or "google/gemini-2.5-flash", etc.
```

**Codex OAuth を使う**（ChatGPT Pro/Plus アカウント — API キー不要）:
```yaml
auxiliary:
  vision:
    provider: "codex"     # uses your ChatGPT OAuth token
    model: "gpt-5.4"      # no implicit default on the Codex route
```

**MiniMax OAuth を使う**（ブラウザログイン、API キー不要）:
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
`hermes model` を実行して **MiniMax (OAuth)** を選ぶと、ログインしてこれが自動的に設定されます。中国リージョンでは、ベース URL は `https://api.minimaxi.com/anthropic` になります。完全な手順は [MiniMax OAuth ガイド](../guides/minimax-oauth.md)を参照してください。


**ローカル/セルフホストのモデルを使う:**
```yaml
auxiliary:
  vision:
    provider: "main"      # uses your active custom endpoint
    model: "my-local-model"
```

`provider: "main"` は、Hermes が通常のチャットに使うプロバイダーを何でも使用します — それが名前付きカスタムプロバイダー（例: `beans`）であれ、`openrouter` のような組み込みプロバイダーであれ、レガシーの `OPENAI_BASE_URL` エンドポイントであれ。

:::tip
メインのモデルプロバイダーとして Codex OAuth を使う場合、ビジョンは自動的に動作します — 追加の設定は不要です。Codex はビジョンの自動検出チェーンに含まれています。
:::

:::warning
**ビジョンにはマルチモーダルモデルが必要です。** `provider: "main"` を設定する場合、エンドポイントがマルチモーダル/ビジョンをサポートしていることを確認してください — そうでないと画像分析は失敗します。
:::

### 環境変数（レガシー）

補助モデルは環境変数経由でも設定できます。ただし、`config.yaml` が推奨される方法です — 管理が容易で、`base_url` と `api_key` を含むすべてのオプションをサポートします。

| 設定 | 環境変数 |
|---------|---------------------|
| ビジョンのプロバイダー | `AUXILIARY_VISION_PROVIDER` |
| ビジョンのモデル | `AUXILIARY_VISION_MODEL` |
| ビジョンのエンドポイント | `AUXILIARY_VISION_BASE_URL` |
| ビジョンの API キー | `AUXILIARY_VISION_API_KEY` |

圧縮とフォールバックモデルの設定は config.yaml のみです。（`AUXILIARY_WEB_EXTRACT_*` 変数は廃止されています — Web の抽出はもう補助 LLM を使いません。）

:::tip
現在の補助モデル設定を見るには `hermes config` を実行してください。オーバーライドは、デフォルトと異なる場合にのみ表示されます。
:::

## 推論の労力

レスポンスの前にモデルがどれだけ「考える」かを制御します。

```yaml
agent:
  reasoning_effort: ""   # empty = medium. Options: none, minimal, low, medium, high, xhigh, max, ultra
```

未設定（デフォルト）の場合、推論の労力はデフォルトで「medium」になります — ほとんどのタスクでうまく機能するバランスの取れたレベルです。値を設定するとそれがオーバーライドされます — 推論の労力が高いほど、より多くのトークンとレイテンシと引き換えに、複雑なタスクでより良い結果が得られます。

### 回答の長さ（`text_verbosity`）

Responses API のモデル（OpenAI の GPT-5 ファミリー以降、直接の OpenAI、ChatGPT Codex、Azure の経路）は、推論の深さとは独立に、最終的な自然言語の回答の長さを決める別のつまみも受け付けます。

```yaml
agent:
  text_verbosity: ""   # empty = not sent (provider default). Options: low, medium, high
```

Hermes はこれを Responses ファミリーの経路でのみ、トップレベルの Responses の `text: {verbosity: ...}` フィールドとして送信します。`chat_completions`、Anthropic、xAI のリクエストで送信されることはなく、空または未知の値では何も送信されません。`request_overrides` で設定された構造化出力（`text.format`）は変更されずにそのまま渡されます。

### OpenAI Daybreak（`daybreak`）

[Daybreak](https://help.openai.com/en/articles/20001258-openai-daybreak-trusted-access-for-cyber-overview) は OpenAI の Trusted Access for Cyber プログラムです: 承認されたアカウントは、サポートされているモデルで、認可されたセキュリティ作業に対する拒否が少なくなります。ChatGPT/Codex のサブスクリプション（`openai-codex`）では、Hermes がこれを要求できます。

```yaml
agent:
  daybreak: false   # true = request Daybreak Blue on models your account is approved for
```

Desktop では、モデルメニューでモデルのオプションを開きます: **Daybreak** スイッチは **Fast** の隣に表示されますが、サインインしているアカウントの Codex カタログが Daybreak 向けに掲載しているモデルでのみ表示されます。Daybreak のないアカウントでは決して表示されません。このスイッチは Codex 自身のトグルと同様にプロファイル全体に適用され、次のターンから反映されます。

すべてのリクエストでカタログが改めて確認されるため、`access_programs.cyber: daybreak_blue` が送信されるのは、スイッチがオン**かつ**そのリクエストのアカウントとモデルが掲載されている場合だけです。掲載されていないモデルへの切り替え、アクセス権のないアカウントへのローテーション、読み取れないカタログの場合は、標準のリクエストが送信されます。API キーの経路や、オプトインの `codex_app_server` ランタイムでは送信されず、これを使うのはあなた自身のチャットのターン（Desktop、TUI、CLI）だけです。メッセージングプラットフォームのチャット、ホストされたルーム、A2A のピア、サードパーティの `--source tool` 連携、エディター（ACP）のクライアント、cron ジョブ、サブエージェントは標準の安全策を使います: OpenAI は Daybreak にサードパーティや下流からのアクセスを認めておらず、それらのチャットの相手は他の人である可能性があるためです。この設定がアクセス権を付与することはありません: 承認は OpenAI から与えられます。

:::note OpenRouter 経由の適応的思考モデル（Claude 4.6 以降、Fable/Mythos クラス）
これらのモデルは*適応的*思考を使い、通常の `reasoning.effort`
フィールドを受け付けません — OpenRouter はそれらに対してこのフィールドを無視します。Hermes は代わりに、
`reasoning_effort` を OpenRouter の `verbosity` パラメータ（Anthropic の
`output_config.effort` に対応）へ透過的にルーティングするため、同じ労力のつまみが、
選択したモデルがサポートするレベルで引き続き機能します。`none`（または未設定）の場合、モデルは
独自の適応的なデフォルトのままになります。
ネイティブの Anthropic プロバイダーはすでに労力を直接制御しており、影響を受けません。
:::

:::note OpenRouter のモデルとサポートされる労力レベル
OpenRouter を通じてルーティングされるその他のモデルについては、Hermes はライブのモデル
カタログの推論メタデータ（`supported_parameters` + モデルごとの
`reasoning.supported_efforts`）を読み取り、そもそも推論の制御を送信するかどうかを決め、
要求された労力を経路が実際にサポートする最も近いレベルに制限します
（常に下方向 — たとえば `high` までの経路では `ultra` は `high` になり、暗黙のうちに
引き上げられることはありません）。新しい推論対応のベンダーは、Hermes のアップデートを待たずに
自動的に機能します。カタログに到達できない場合やモデルが掲載されていない場合、Hermes は組み込みの
モデルファミリーのリストにフォールバックし、労力をそのまま渡します。
:::

:::note `ultra` は経路が受け付ける最も強いレベルに制限される
`ultra` は Hermes 内部の段階です: どのプロバイダーのワイヤーもこれを受け付けないため、すべての経路は
これを最も強いレベルに制限します（GPT-5.6 Codex と OpenAI 互換の経路では `max`、古い
Codex のモデルでは `xhigh`）。労力のピッカーと `/reasoning` のステータスは、これを
`ultra (sends max on this route)` と表示するため、表示されるレベルが送信されるレベルになります。
:::

`/reasoning` コマンドで実行時に推論の労力を変更することもできます。

```
/reasoning                # Show current effort level and display state
/reasoning high           # Set reasoning effort to high (this session only)
/reasoning high --global  # Set effort and persist to config.yaml
/reasoning none           # Disable reasoning (this session only)
/reasoning show           # Show model thinking above each response
/reasoning hide           # Hide model thinking
```

労力の変更はデフォルトでセッション単位です。新しいレベルを `agent.reasoning_effort` の
デフォルトとして保存するには、`--global` を追加してください。

#### モデルごとの推論のオーバーライド

モデルごとに異なる推論の労力レベルを設定できます。これは、複雑なモデルには高い推論を、より高速なモデルには medium を使いたい場合に便利です。

```yaml
agent:
  reasoning_effort: "medium"       # global default
  reasoning_overrides:
    "openrouter/anthropic/claude-opus-4.5": "xhigh"
    "openai/gpt-5": "low"
    "claude-sonnet-4.6": "high"    # bare model name also works
```

キーの照合は**表記の揺れに寛容**です — 妥当な表記であればどれでも一致します。
- `claude-opus-4.5`、`claude-opus-4-5`、`claude-opus.4.5`（ドットとダッシュは相互に置き換え可能）
- `anthropic/claude-opus-4.5`、`openrouter/anthropic/claude-opus-4.5`（プロバイダーのプレフィックスは任意）
- 名前付きカスタムプロバイダーのプレフィックスが付いたキー（`ollama-local/qwen3.6:27b-q4_k_m`）は、リクエストが素のモデル ID（`qwen3.6:27b-q4_k_m`）だけを持つ場合にも適用されます。これはフォールバックのエントリや `providers:` の経路が送るものです
- 完全一致は変形より優先されます

#### カスタムの推論段階の名前

一部の OpenAI 互換エンドポイントは、標準の段階の外にある思考の段階を公開しています（`low`…`max` の代わりに `fast`/`thinking` を提供するリレー）。段階の外にある素の文字列は `Unknown reasoning_effort '<value>', using default (medium)` として拒否されるため、タイプミスがワイヤーに届くことはありません。プロバイダー独自の段階の名前を要求するには、明示的な dict 形式を使います — `effort` の値は、トップレベルの `reasoning_effort` フィールドとしてそのまま送信されます。

```yaml
agent:
  reasoning_effort:
    enabled: true
    effort: thinking            # sent as-is
  reasoning_overrides:
    "my-relay/lumo-max":        # dict form works per model too
      enabled: true
      effort: fast
```

dict 形式の `enabled: false` は、`reasoning_effort: none` と同様に思考をオフにします。

dict 形式は `config.yaml` を直接編集して設定します: `/reasoning` のメニュー、`hermes model`、ダッシュボードの補助モデルのピッカーは標準の段階しか提供しません（TUI のステータスとセットアップウィザードは、設定済みのカスタムの段階の名前を引き続き表示します）。

:::note
モデル ID にはドットが含まれます（`claude-opus-4.5`、`qwen3.6:27b`）が、`hermes config set` はドットを入れ子の区切りとして扱います。リテラルのキーを書き込むにはバックスラッシュでエスケープする — `hermes config set 'agent.reasoning_overrides.ollama-local/qwen3\.6:27b-q4_k_m' low` — か、YAML を直接編集してください。[キー名の中のドット](../reference/cli-commands.md#dots-inside-key-names)を参照してください。
:::

:::note OpenAI Responses（`openai-api`、`openai-codex`）
`reasoning_effort: none` は、それを受け付けるモデル（GPT-5.x）では `reasoning.effort: "none"` として明示的に送信されます: フィールドを省略すると、モデルのデフォルトの労力が有効なままになるためです（GPT-5.6 のデフォルトは `medium`）。フィールドを省略するのは、労力が未設定の状態だけです。`api.openai.com` 上のチャット時代のモデル（`gpt-4o`、`gpt-4.1`、それらの `-mini` 版やファインチューン）は `reasoning` パラメータを一切拒否するため、Hermes は `400 Unsupported parameter: 'reasoning.effort'` で失敗する代わりに、設定された労力にかかわらずそれらには何も送信しません。モデルが `none` を拒否した場合、Hermes は警告し、そのセッションでは無効化を取りやめ、モデルのデフォルトで再試行します。
:::

:::note ローカルの OpenAI 互換エンドポイント
カスタムの `base_url`（`http://localhost:11434/v1`、vLLM、SGLang、またはルーターのエンドポイント）は、解決された労力 — `agent.reasoning_effort` または一致するモデルごとのオーバーライド — を、標準のトップレベルの `reasoning_effort` リクエストフィールドとして受け取ります。値は OpenAI 互換のワイヤーが受け付けるもの（`none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`）に制限されます。未設定の労力は、Nous Portal と OpenRouter の経路が適用するのと同じデフォルトである `medium` としてここでも送信されます — フィールドを省略すると選択がエンドポイントに委ねられてしまい、ホストされた推論モデル自身のデフォルトはその上限であることがあるためです（kimi-k3 のデフォルトは `max` で、`medium` の約 3 倍の推論トークンとレイテンシがかかります）。カタログや `model_overrides` が `supports_reasoning: false` とマークしたモデル、`thinking` のケーパビリティなしで pull したローカルの Ollama モデル、そしてエンドポイントがこのフィールドに `400` で応答した後のセッションの残りでは、このフィールドは送信されません。入れ子の `reasoning` オブジェクトは、任意のサーバーが未知のフィールドを HTTP 400 で拒否するため、それを受け付けることがわかっているエンドポイント（Nous Portal、OpenRouter の推論対応モデル、GitHub Models）のために予約されています。サーバーが思考の予算を別のフィールド（Ollama の `think`、vLLM の `chat_template_kwargs`、ルーター固有のキー）から読み取る場合は、カスタムプロバイダーの [`extra_body`](../integrations/providers.md#named-custom-providers) の下に設定してください。これはそこにルーティングされるすべてのリクエストにマージされます。
:::

**解決の優先順位:**

1. セッション単位の `/reasoning --session` オーバーライド（ゲートウェイのみ）
2. `agent.reasoning_overrides` からのモデルごとのオーバーライド（表記の揺れに寛容）
3. グローバルな `agent.reasoning_effort`
4. プロバイダーのデフォルト

このオーバーライドは、あらゆる場所で自動的に適用されます: CLI の起動、`hermes -p` のワンショット、メッセージングゲートウェイ、Desktop/TUI、ACP セッション、cron ジョブ、セッション途中の `/model` による切り替え（最初のメッセージの前に行われた切り替えを含む）、セッションの再開（`--resume`、`/resume`）、`/new`、フォールバックモデルの有効化です。

## Fast モード {#fast-mode}

Fast モードは、割増料金と引き換えにより高速な出力をプロバイダーに求めます: サポートされている OpenAI のモデルでの OpenAI [Priority Processing](https://openai.com/api-priority-processing/)（`service_tier: priority`）と Ultrafast（`service_tier: ultrafast`）、Grok 4.6 での xAI Priority Processing、そして Anthropic の [Fast Mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode)（`speed: fast`。Opus 4.8 / Opus 5 / Opus 5.5 のみ）です。`openai` と `openai-api` のプロバイダーは、ファーストパーティの OpenAI エンドポイントを使います。**デフォルトではオフ**です。

```yaml
agent:
  service_tier: ""          # "" / normal | fast | priority | ultrafast | auto | cold
  fast_auto_seconds: 60     # window for auto / cold
```


| モード | Fast のパラメータが送信されるタイミング | 用途 |
|------|---------------------------|------------|
| `normal`（デフォルト、`""`） | 送信しない | 最も安価。標準のレイテンシ |
| `fast` | すべてのリクエスト | 常に速度が欲しい長い対話的なセッション |
| `auto` | **毎**ターンの最初の `fast_auto_seconds` 内のリクエスト | 最初の返信を素早く。長いツールループは標準の料金に戻る |
| `cold` | 同じ時間枠だが、セッションの**最初のターン**（それ以前の履歴がない）のみ | オンボーディングの返信を高速に、その後は標準の料金 |

`/fast normal|fast|ultrafast|auto|cold` は、セッションのモードを切り替えます（`/fast ultra` は `ultrafast` の省略形です）。ChatGPT/Codex のサブスクリプションでは、Ultrafast は、アカウントの Codex のモデルカタログが現在のモデルについてそれを掲載している場合にのみ提供されます。そうでない場合、バックエンドはリクエストを受け付けて標準の速度で提供します。`config.yaml` に永続化するには `--global` を追加します。`/fast` だけで現在のモードを表示します。

**コストに関する注意:** どちらのプロバイダーも、Fast のリクエストを標準料金の倍率で課金し（Anthropic: Opus 5.5 では入力/出力 100 万トークンあたり $8 / $40、Opus 5 と Opus 4.8 では $10 / $50）、プロンプトキャッシュの料金と積み重なります。Hermes は、各 Anthropic のレスポンスを、API が `usage.speed` で報告する速度から価格付けします。`auto`/`cold` は、その割増を時間枠の間だけに限定します。Fast のパラメータは、それをサポートするファーストパーティのエンドポイント（`api.openai.com` / Codex サブスクリプション、`api.anthropic.com`、`api.x.ai`）にのみ送信されます。OpenRouter、Nous Portal、Azure、Bedrock、カスタムの `base_url` の経路が、どのモードでもそれを受け取ることはありません。

**プロンプトキャッシュ:** リクエスト間で変わるのはリクエストごとのパラメータだけで、システムプロンプト、ツール、メッセージはバイト単位で同一のままです。Anthropic は速度ごとに別々のプロンプトキャッシュを保持するため、Anthropic では `auto`/`cold` の時間枠の境界のたびに、会話のプレフィックスが新しい速度で書き直されます。長い Anthropic のセッションでは、`fast` または `normal` にすると単一の温かいキャッシュが保たれます。

Fast モードの高速化は 1 秒あたりの出力トークンにあるため、長い回答ほど大きな恩恵を受けます。

Anthropic の組織にあるモデルの Fast モードの容量がない場合（API が Fast のリクエストに Fast モードの上限 0 で応答する場合）、Hermes はそのセッションの残りの間そのモデルを標準の速度に切り替え、リクエストを再試行します。

### ゲートウェイやプロキシの背後にある Fast の段階

ファーストパーティのみというルールは意図的なものです: Fast の段階のパラメータは課金の指示であり、Hermes はその料金表を知っているエンドポイントにのみそれを送信します。独自の優先度の段階（独自の `service_tier` の値、または別の名前のフィールド）を公開する OpenAI 互換のゲートウェイ、ルーター、プロキシを運用している場合は、`agent.service_tier` ではなく、そのプロバイダーの `extra_body` を通じて要求してください。[名前付きカスタムプロバイダー](../integrations/providers.md#named-custom-providers)の `extra_body` は、そのエンドポイントにルーティングされる**すべての** chat-completions リクエストにマージされ、ゲートウェイのターンや `/fast` の変更をまたいで維持され、`/model` でそのプロバイダーから切り替えると再び取り除かれます。

```yaml
providers:
  my-gateway:
    api: https://gateway.example.com/v1
    key_env: MY_GATEWAY_KEY
    default_model: fast-lane-model
    extra_body:
      service_tier: priority     # whatever tier value your gateway documents
```

`agent.service_tier` との違い: その段階はそのプロバイダーで常に有効であり（`auto`/`cold` の時間枠はない）、`/fast` で切り替わらず、Hermes は値を検証しません — 何を受け付けて何を課金するかはゲートウェイが決めます。

## ツール使用の強制

一部のモデルは、ツール呼び出しを行う代わりに、意図したアクションをテキストとして説明することがあります（実際にターミナルを呼ぶ代わりに「テストを実行します……」）。ツール使用の強制は、モデルを実際にツールを呼ぶ方向へ導くシステムプロンプトのガイダンスを注入します。

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| 値 | 動作 |
|-------|----------|
| `"auto"`（デフォルト） | 次にマッチするモデルで有効: `gpt`、`codex`、`gemini`、`gemma`、`grok`、`glm`、`qwen`、`deepseek`、`muse`。それ以外（例: Claude）では無効。 |
| `true` | モデルにかかわらず常に有効。現在のモデルがアクションを実行する代わりに説明していると気づいた場合に有用。 |
| `false` | モデルにかかわらず常に無効。 |
| `["gpt", "codex", "qwen", "llama"]` | モデル名が列挙された部分文字列のいずれかを含む場合にのみ有効（大文字小文字を区別しない）。 |

### 注入される内容

有効にすると、2 つの層のガイダンスがシステムプロンプトに追加されることがあります。

1. **一般的なツール使用の強制**（すべてのマッチしたモデル） — 意図を説明する代わりにすぐにツール呼び出しを行い、タスクが完了するまで作業を続け、将来のアクションの約束でターンを終えないようモデルに指示します。

2. **Google の運用ガイダンス**（Gemini および Gemma モデルのみ） — 簡潔さ、絶対パス、並列ツール呼び出し、編集前検証のパターン。

これらはユーザーに対して透過的で、システムプロンプトにのみ影響します。すでにツールを確実に使うモデル（Claude など）にはこのガイダンスは不要であり、それが `"auto"` がそれらを除外している理由です。

### いつ有効にするか

デフォルトの auto リストにないモデルを使用していて、実行する代わりに *やろうとしている* ことを頻繁に説明すると気づいた場合は、`tool_use_enforcement: true` を設定するか、モデルの部分文字列をリストに追加してください。

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## 実行規律のガイダンス

ツール使用の強制とは別に、Hermes は、評価のトレースで観察された一連のエージェント的な失敗モードを共有するモデルファミリーに対して、**実行規律**のブロックを注入します: コードではなく文章で計算する、外部への書き込みの後に読み返しによる検証を省く、不正な形式の識別子を「修復」する、件数が一致しないのに完全だと主張する、すべての受け入れ基準を検証せずに「完了」を宣言する、といったものです。

```yaml
agent:
  execution_guidance: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| 値 | 動作 |
|-------|----------|
| `"auto"`（デフォルト） | 次にマッチするモデルで有効: `gpt`、`codex`、`grok`、`deepseek`、`kimi`、`qwen`、`glm`、`minimax`、`mimo`、`mistral`、`muse`。 |
| `true` | モデルにかかわらず常に有効。 |
| `false` | モデルにかかわらず常に無効。 |
| `["deepseek", "my-custom-model"]` | モデル名が列挙された部分文字列のいずれかを含む場合にのみ有効（大文字小文字を区別しない）。 |

注入されるブロックは次の内容をカバーします。

- **ツールの粘り強さ** — タスクが完了し、*かつ*検証されるまでツールを呼び続ける。空、部分的、または疑わしいほど狭い検索結果は、結論を出す前に、より広いクエリや別のクエリで再試行する。
- **ツールの必須使用** — 算術、ハッシュ、日付、システムの状態、ファイルに関する事実は、暗算ではなく常にツールから得る。
- **外部への書き込みの読み返し** — 外部システムへの状態を変える書き込みの後は、成功を主張する前に正確な対象を読み返す（ツールがすでに確認した内部のファイル編集は再検証しない）。
- **件数の照合** — 宣言された合計（`total`、`reply_count`、`has_more`）は厳格なアサーションである。一致しない場合は、再取得するかプログラムで解析する。
- **リテラルの保持** — 指定された形式に合わない識別子を正規化したり「修復」したりしない。検索が成功しても、不正な形式の元のトークンが正しいことにはならない。
- **検証に基づく完了** — 「完了」とは、名前の挙がったすべての受け入れ基準が検証されたことを意味し、もっともらしい一部分ではない。

このゲートは `tool_use_enforcement` とは独立しています — どちらか一方だけを有効にできます。ガイダンスはセッションの開始時にモデル名に基づいて一度だけ選ばれるため、システムプロンプトは会話の生涯を通じてバイト単位で安定したまま（プロンプトキャッシュに優しいまま）です。Gemini/Gemma は、より具体的な Google の運用ガイダンスを受け取るため、auto リストから除外されています。Claude は、これらの失敗モードを示さないため除外されています — 任意のモデルを `true` または部分文字列のリストでオプトインできます。

## ツールループのガードレール

Hermes は、エージェントが非生産的なツール呼び出しのループにはまっていることを検出します — 同じツール呼び出しが繰り返し失敗する、同じツールが何度も失敗する、あるいは冪等な呼び出しが進捗なしに同じ結果を返す、といった場合です。デフォルトでは、モデルが自己修正できるよう、ツール結果に**警告**を注入します。対話的な CLI、TUI、Desktop、ACP のセッションは、人が介入できるため警告のみのままです。無人のゲートウェイと cron のセッションでは、デフォルトでハードストップが有効になります。

このプラットフォームに応じたデフォルトは、無人のデプロイでは無効にでき、またすべてのプラットフォームでハードストップを明示的に有効にすることもできます。

```yaml
tool_loop_guardrails:
  warnings_enabled: true       # inject warnings into tool results (default: true)
  hard_stop_enabled: false     # also BLOCK the call past the hard-stop threshold (default: false)
  non_interactive_hard_stop_enabled: true  # default hard stops for gateway/cron
  warn_after:
    exact_failure: 2           # identical failing call repeated N times
    same_tool_failure: 3       # same tool failing N times (different args)
    idempotent_no_progress: 2  # same result, no progress, N times
  hard_stop_after:
    exact_failure: 5
    same_tool_failure: 8
    idempotent_no_progress: 5
  loop_caps:
    max_web_searches: 50       # max web_search calls per turn (0 = unlimited)
    max_subagents: 50          # max subagents spawned per turn (0 = unlimited)
```

`hard_stop_enabled` は、すべてのプラットフォームでハードストップを明示的に有効にします。これが `false` のままの場合でも、`non_interactive_hard_stop_enabled` は無人のゲートウェイ/cron 形式のプラットフォームでハードストップを有効にし、CLI、TUI、Desktop、ACP、サブエージェント、`api_server` の実行（稼働中の親やクライアントがある監督下のタスクループ）では警告のみの動作を維持します。無人のデプロイでこれをオプトアウトするには、`non_interactive_hard_stop_enabled: false` を設定してください。[Docker / 無人のデプロイ](docker.md)も参照してください。

ハードストップは、正当な反復ではなく、**リプレイ** — 間に何も起きていない、変更のない同じ呼び出し — を捕捉するように設計されています。

- **編集 → 再実行は決してループではありません。** 成功した変更を伴う呼び出し（`write_file`、`patch`、成功した `terminal`/`execute_code`、ブラウザの操作、ジョブ/メッセージ/cron の変更）があると、まだカウントされているすべての失敗した呼び出しについて進捗があったとみなされます。次の同一の再試行（修正後に失敗したテストを再実行する、クリック後にスナップショットを取り直す）は、ブロックに向けて累積するのではなく、新しい連続として開始されます。
- **異なる失敗コマンドはループではなく診断です。** ゼロ以外の終了が通常の出力であるツール（`terminal`、`execute_code`、プロセスのポーラー、`browser_exec`、`web_extract`）については、`same_tool_failure` の閾値は警告するだけで、停止させることはありません。それらを止められるのは、間に変更のない完全に同じ引数のリプレイ、または同一結果の連続だけです。
- **停止はセッションではなくターンを終了させます。** エージェントは、どのガードレールがなぜ発火したかを返信します。「continue」と返信すると、ターンごとのカウンターをリセットして再開します。

### ターンごとの暴走ループの上限

上記の失敗ベースの閾値とは別に、`loop_caps` は、1 つのエージェントループ（ターン）が行える `web_search` の呼び出し数とサブエージェントの生成数に厳格な上限を設けます。カウンターは毎ターンの開始時にリセットされるため、正当な複数ターンのセッションが不足に陥ることはありません — しかし、際限のない検索や委譲のループに陥った 1 つのターンは停止されます。これらは常に有効で、`hard_stop_enabled` にかかわらず発火します。1 つのターンで数十回の Web 検索を行ったり数十のサブエージェントを生成したりすることはすでに異常であるため、デフォルトは低めです。上限に達すると、問題のツール呼び出しは説明的なメッセージとともにブロックされ、ターンは予算の残りを使い果たす代わりにきれいに停止します。その上限を完全に無効にするには、いずれかの値を `0` に設定します。

1 つの `delegate_task` のバッチは、各タスクを `max_subagents` に対してカウントするため（3 つのバッチは 3 を消費）、上限は `delegate_task` の呼び出し回数ではなく、実際に生成されたサブエージェントを追跡します。

これは Claude Code のセッションごとの WebSearch とサブエージェントの上限（v2.1.212）を踏襲したもので、そちらもデフォルトは 200 で、`/clear` でリセットされます。

### 実行時の停滞防止ガード

上記の失敗ベースのガードレールを補完するものとして、`agent.stall_guards`（デフォルト `true`）は、無駄なターンに対する 2 つの控えめな実行時のガードを有効にします。1 つ目は**同一呼び出しのループブレーカー**です: 同じツールが同一の引数で 3 回以上連続して呼び出され、*かつ*同一の結果を返した場合、その呼び出しを繰り返さないようモデルに伝える短い 1 行の通知が、そのツール結果に追加されます — 警告のみのセッションでは呼び出しをブロックすることはなく、正当に繰り返し可能なポーラー（`process_manage`、`*_get_result`、`*_poll`）は対象外です。ハードストップが有効な場合（明示的な `hard_stop_enabled`、または無人のゲートウェイ/cron のプラットフォーム）、同じ連続が `hard_stop_after.idempotent_no_progress` 回の連続した同一呼び出しに達すると、ハードストップにもなります — これは `idempotent_no_progress` のガードレールが追跡する読み取り専用のツールだけでなく、**あらゆる**ツールが対象です — そのため、同じ成功した `terminal` や `skill_view` の呼び出しを繰り返すモデルは、イテレーション予算を使い切るのではなく停止されます（`identical_call_streak_halt`）。2 つ目は**継続意図の回復**です: モデルがツール呼び出しなしでターンを終え、その短い返信がアクションを予告しながら途切れている場合（「では、ファイルを更新します…」）、Hermes は、意図の確認の回復に使われるのと同じ制限付きの継続の仕組みを通じて（ターンごとに最大 2 回の再プロンプト）、行動するよう再度促します。どちらもキャッシュに安全で（通知は結果の構築時に追加され、遡って追加されることはありません）、まとめて無効にできます。

```yaml
agent:
  stall_guards: false
```

同じゲートは**結果参照のスタブ化**も有効にします: 再発行された同一のツール呼び出しが、バイト単位で同一の新しい結果を返した場合、重複したペイロードは、完全な出力を繰り返す代わりに、以前の結果を指す短い参照スタブ（ツール名、`tool_call_id`、引数の要約、そして最初の結果がディスクに保存された場合はその退避パス）としてコンテキストに入ります。ツールは毎回実行されるため、ポーリングのセマンティクスは保たれます: 変更された結果は常に全体が流れます。512 文字未満の結果、エラーの結果、マルチモーダルの結果はスタブ化されることはなく、ポーラーはスタブ化*されます*（変化のないポーリングは、まさに重複したペイロードが情報を持たないケースです）。

### ターンの生存ウォッチドッグ

`agent.turn_liveness` は、Hermes が強制的に回復させる前に、会話のターンが**観測可能な進捗なし**でいられる時間を制限します。ウォッチドッグはアクティビティクロック（API の待機、ストリームのトークン、ツールのハートビートを記録するのと同じシグナル — リースの更新は決してカウントされません）を基準にするため、実行の途中で無言のまま止まってしまったターン（issue #95548 として観察されたもの: ツールの実行も API 呼び出しもエラーもないのに、セッションは無期限に「busy」のまま）は、はっきりと表面化され、再試行可能な中断されたターンとして巻き戻るよう割り込まれます。そして、割り込みでその停止を巻き戻せない場合は、その永続的なターンのリースが更新されなくなるため、プロセスが強制終了されるまでハングし続けるのではなく、古いターンのクリーンアップがセッションを回収できます。

```yaml
agent:
  turn_liveness:
    timeout_s: 600.0   # idle bound; <= 0 disables the watchdog
    poll_s: 15.0        # sampling interval (seconds)
```

正当に遅い作業がペナルティを受けることはありません: ストリーミングのレスポンス、ツールのハートビート（ツールの実行中は 30 秒ごと）、承認の待機はすべてクロックを更新し続けるため、ウォッチドッグが発火するのは、制限時間いっぱいの間*まったく*進捗のないターンだけです。無効な値（タイプミス、`NaN`、`Inf`、正でない `poll_s`）は警告をログに記録してデフォルトにフォールバックします — 起動をクラッシュさせたり、ウォッチドッグを暗黙のうちに無効にしたりすることはありません。発火した中断は、回復を開始する時点で停止を報告し、割り込みが実際にコミットされて初めて、中断/リース停止という確定した結果を公開します。


## TTS 設定

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts" | "kittentts" | "piper" | "deepinfra"
  speed: 1.0                    # Global speed multiplier (fallback for all providers)
  keep_warm_seconds: 60         # Keep a local engine loaded this long after the last speech toggle turns off (0 = unload at once)
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
    speed: 1.0                  # Speed multiplier (converted to rate percentage, e.g. 1.5 → +50%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    speed: 1.0                  # Speed multiplier (clamped to 0.25–4.0 by the API)
    base_url: "https://api.openai.com/v1"  # Override for OpenAI-compatible TTS endpoints
    pcm_sample_rate: 24000      # Streaming PCM rate; overridden by the endpoint's X-Audio-Sample-Rate header
  minimax:
    speed: 1.0                  # Speech speed multiplier
    # base_url: ""              # Optional: override for OpenAI-compatible TTS endpoints
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (default)
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # or gemini-3.1-flash-tts-preview
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, etc.
    audio_tags: false           # Hidden Gemini 3.1 TTS audio-tag insertion
    persona_prompt_file: ""      # Optional Markdown/text file with Gemini voice direction
  xai:
    voice_id: "eve"             # xAI TTS voice
    language: "en"              # ISO 639-1
    sample_rate: 24000
    bit_rate: 128000            # MP3 bitrate
    # base_url: "https://api.x.ai/v1"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

これは `text_to_speech` ツールと、音声モード（CLI またはメッセージングゲートウェイでの `/voice tts`）での音声による返信の両方を制御します。

**速度のフォールバック階層:** プロバイダー固有の速度（例: `tts.edge.speed`） → グローバルな `tts.speed` → デフォルトの `1.0`。すべてのプロバイダーに一律の速度を適用するにはグローバルな `tts.speed` を設定し、きめ細かく制御するにはプロバイダーごとにオーバーライドしてください。

## 表示設定 {#display-settings}

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # Enable /verbose slash command in messaging gateway
  focus_view: false       # CLI focus view (/focus) — reduced output, display-only
  platforms: {}           # Per-platform display overrides (see below)
  interim_assistant_messages: true  # Gateway: send natural mid-turn assistant updates as separate messages
  suppress_warning_notifications: false  # Opt-in: hide automatic warning/diagnostic notices, incl. the silence-marker notice (see messaging guide)
  show_commentary: true   # Codex models: deliver commentary-channel progress narration as visible mid-turn updates
  skin: default           # Built-in or custom CLI skin (see user-guide/features/skins)
  personality: ""         # Legacy cosmetic field still surfaced in some summaries
  compact: false          # Compact output mode (less whitespace)
  cli_multiline_shortcuts: true  # CLI: Ctrl+J, \ + Enter, and supported Shift+Enter insert newlines (false = legacy c-j submit fallback)
  resume_display: full    # full (show previous messages on resume) | minimal (one-liner only)
  bell_on_complete: false # Play terminal bell when agent finishes (great for long tasks)
  bell_on_prompt: false   # Play terminal bell when a blocking prompt opens (clarify, approval, sudo password, secret capture) — works over SSH
  # Both bell flags also emit an OSC 9 desktop notification (Ghostty, iTerm2, Kitty, WezTerm raise an OS
  # notification; other terminals ignore it) and, inside Warp (TERM_PROGRAM=WarpTerminal with the CLI-agent
  # protocol advertised), a warp://cli-agent OSC 777 event (`stop` on completion, `permission_request` on
  # blocking prompts) so Warp's tab status and notification mailbox track Hermes. No extra keys needed.
  show_reasoning: true    # Show model reasoning/thinking above each response (default: true; toggle with /reasoning show|hide). On a gateway reply that was streamed, it follows the answer as its own message
  streaming: false        # Stream tokens to terminal as they arrive (real-time output)
  show_cost: false        # Show estimated $ cost in the CLI status bar
  vim_mode: false         # CLI only: vi/vim keybindings in the input composer (Esc → NORMAL, i → INSERT). The live NORMAL/INSERT/REPLACE mode shows at the right of the status bar. Config-only, read at startup.
  timestamps: false       # When true, prefixes user and assistant labels with timestamps in the CLI / TUI transcript
  timestamp_format: "%H:%M"  # strftime format for those timestamps (e.g. "%b-%d %H:%M" for month-day)
  tool_preview_length: 0  # Max chars for tool call previews (0 = no limit, show full paths/commands)
  turn_summary: true      # CLI only: print a one-line post-turn accounting footer after each interactive turn
  spinner_token_flow: true # CLI only: append live cumulative turn tokens to the spinner timer
  runtime_footer:         # Gateway: append a runtime-context footer to final replies
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  status_bar:             # CLI/TUI: choose which status-bar fields are visible
    fields: []            # empty = show the default set; see below
  file_mutation_verifier: true    # Append an advisory footer when write_file/patch calls failed this turn
  credits_notices: true   # Nous credits status-bar notices (usage bands, grant-spent, depleted). false = silence them; /usage still works
  cli_rebuild_scrollback_on_redraw: false  # Classic CLI: also wipe terminal scrollback (CSI 3J) on /redraw / Ctrl+L / width-change resize recovery. Enable when a terminal/tmux stack stamps stale prompt chrome into scrollback on maximize/restore.
  language: en            # UI language for static messages (approval prompts, some gateway replies). en | zh | zh-hant | ja | de | es | fr | tr | uk | af | ko | it | ga | pt | ru | hu
```

### ターンごとの要約とスピナーのトークンフロー

`display.turn_summary`（デフォルト `true`）は、各**対話的な CLI** のターンの後に、そのターンで実際に行ったことを要約する薄い色の集計行を 1 行出力します。

```
⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3 commands
```

この集計は、CLI がすでに受け取っているツールの進捗フィードから観測されるため、追加のコストはかかりません。詳細:

- 実時間はターンの実際の所要時間です（1 分を超えると `2m05s`）。
- ツール呼び出しは動詞（`edited`、`read`、`ran`、`searched` など）ごとに、正しい複数形でまとめられます。厳選された動詞を持たないプラグイン/MCP のツールは `called N tools` にまとめられます。
- `+X -Y` の行の差分は、ツール結果がすでに差分を報告している場合（現在は `patch`）にのみ表示されます。Hermes がそれを計算するために git を実行することはないため、`write_file` による編集は差分なしでカウントされます。
- **失敗したツール呼び出しはカウントされません** — 拒否された書き込みが成功した編集として表示されることはありません（補完的な警告については[ファイル変更の検証器](#file-mutation-verifier)を参照）。
- 長いターンは 4 つの動詞のセグメントと `+N more` の末尾までに制限されるため、行が折り返されることはありません。
- ツール呼び出しのない速いターンでは、何も出力されません。

`display.spinner_token_flow`（デフォルト `true`）は、実行中のターンの累積出力トークンを CLI のスピナーのライブタイマーに追加します。

```
  ⚡ Reading cli.py  (  2.3s · ↓ 1.2k tok)
```

この数はターンごと（セッションの合計はターンの開始時を基準とする）で、ターン内の各 API 呼び出しが使用量を報告するたびに更新されます。最初の使用量の報告が届く前は何も表示されないため、誤解を招く `↓ 0 tok` が表示されることはありません。

どちらのキーも表示専用かつ CLI 専用です: クワイエットモード、`display.tool_progress` が `off` の場合、単一クエリ/`-Q` のバッチ実行、そしてゲートウェイ/メッセージングのサーフェス（それらは代わりに `display.runtime_footer` を使います）では抑制されます。オフにするには、いずれかのキーを `false` に設定してください。

### ファイル変更の検証器 {#file-mutation-verifier}

`display.file_mutation_verifier` が `true`（デフォルト）の場合、ターン中に `write_file` または `patch` の呼び出しが失敗し、かつ対象のファイルがその後（パスのどの表記でも）正常に書き込まれず、ターンの終了前にディスク上でほかの方法でも変更されなかったときは、Hermes はアシスタントの最終レスポンスに 1 行の注意書きを追加します。これにより、編集のたびに手動で `git status` を実行しなくても、「並列のパッチの一括処理で半分が暗黙のうちに失敗したのに、モデルが成功として要約する」という種類の過大な主張を捕捉できます。

フッターの例:

```
⚠️ File-mutation verifier: 3 file edit(s) FAILED this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm what actually landed.
  • concepts/automatic-organization.md — [patch] Could not find match for old_string
  • concepts/lora.md — [patch] Could not find match for old_string
  • concepts/rag-pipeline.md — [patch] Could not find match for old_string
```

フッターを抑制するには、`file_mutation_verifier: false`（または `HERMES_FILE_MUTATION_VERIFIER=0`）を設定します。検証器が発火するのは、ターンの終了時に実際の失敗が未解決のまま残っている場合だけです — 失敗したパッチを再試行して同じターン内で成功したモデルは、そのファイルについてこれを発火させません。

**モデルの要約よりも検証器を信頼してください。** このフッターは、アシスタントの締めくくりのメッセージがタスクは完了したと言っていても、列挙された編集の呼び出しが**失敗**し、Hermes がその後それらのファイルへの変更を確認できなかったことを意味します。これが追跡するのは `write_file`/`patch` の受領記録と、ターン終了時の更新時刻のチェックだけなので、実際に何が反映されたかは `git status` または `read_file` で確認してください。よくある原因:

- **書き込みの拒否** — パスが認証情報の拒否リストに含まれているか、`HERMES_WRITE_SAFE_ROOT` の外にある（[ファイル書き込みの安全性](./security.md#file-write-safety)を参照）
- **パッチの不一致** — `old_string` がディスク上のファイルと一致しなかった
- **構文のゲート** — 書き込みの前に、候補の内容が JSON/YAML/TOML の検証に失敗した

書き込みがブロックされた場合のフッターの例:

```
⚠️ File-mutation verifier: 2 file edit(s) FAILED this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm what actually landed.
  • ~/.hermes/cron/jobs.json — [patch] Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (/path/to/project)
  • ~/.hermes/scripts/monitor.py — [write_file] Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (/path/to/project)
```

Hermes の状態（cron ジョブ、スキル、`~/.hermes/` 配下のスクリプト）への書き込みが失敗している場合は、環境で `HERMES_WRITE_SAFE_ROOT` が設定されていないか確認してください。cron の変更には、`jobs.json` を直接パッチする代わりに、`cronjob_manage` ツールまたは `hermes cron edit` を使ってください。

### 静的メッセージの UI 言語

`display.language` 設定は、ユーザー向けの静的メッセージの小さな集合 — CLI の承認プロンプト、いくつかのゲートウェイのスラッシュコマンドの返信（例: 再起動時のドレインの通知、「approval expired」、「goal cleared」）— を翻訳します。エージェントのレスポンス、ログ行、ツールの出力、エラーのトレースバック、スラッシュコマンドの説明は翻訳**しません** — それらは英語のままです。エージェント自体に別の言語で返信させたい場合は、プロンプトやシステムメッセージでそう伝えてください。

同梱されている値: `en`（デフォルト）、`zh`（簡体字中国語）、`zh-hant`（繁体字中国語）、`ja`（日本語）、`de`（ドイツ語）、`es`（スペイン語）、`fr`（フランス語）、`tr`（トルコ語）、`uk`（ウクライナ語）、`af`（アフリカーンス語）、`ko`（韓国語）、`it`（イタリア語）、`ga`（アイルランド語）、`pt`（ポルトガル語）、`ru`（ロシア語）、`hu`（ハンガリー語）、`ar`（アラビア語）。

このリストは**プラグイン可能**です: [言語パック](/docs/user-guide/features/language-packs)のプラグイン（`provides_locales`）や、部分的な `<HERMES_HOME>/locales/<lang>.yaml` のオーバーレイによって、言語を追加したり文言をオーバーライドしたりできます。`hermes config set display.language <id>` は、同梱のカタログ、あなたのオーバーレイ、またはインストール済みのパックが提供する任意の ID を受け付けます。未知の ID は、利用可能な言語の一覧とともに拒否されます。実行時に解決できない値は英語にフォールバックします。

`HERMES_LANGUAGE` 環境変数でセッションごとに設定することもでき、これは設定値をオーバーライドします。

```yaml
display:
  language: zh   # CLI approval prompts appear in Chinese
```

| モード | 表示される内容 |
|------|-------------|
| `off` | 無音 — 最終的なレスポンスのみ |
| `new` | ツールが変わったときだけツールのインジケーター |
| `all` | 短いプレビュー付きのすべてのツール呼び出し（デフォルト） |
| `verbose` | 完全な引数、結果、デバッグログ |

CLI では、`/verbose` でこれらのモードを切り替えられます。メッセージングプラットフォーム（Telegram、Discord、Slack など）で `/verbose` を使うには、上記の `display` セクションで `tool_progress_command: true` を設定してください。コマンドがモードを切り替え、設定に保存するようになります。


ツールの進捗には、進捗の更新を安全に表示できるゲートウェイのアダプターが必要です。Signal を含む、メッセージの編集をサポートしないプラットフォームでは、`/verbose` が `off` 以外のモードを保存しても、ツールの進捗の吹き出しは抑制されます。

`off` が隠すのはツール呼び出しの*装飾*だけです。Desktop アプリと TUI で独自の表示面を持つアプリケーションの状態 — タスクリスト（`todo_list`）、サブエージェントの進捗、確認の質問、MCP の同意カード — は、この設定にかかわらず流れ続けます。

### フォーカスビュー（`/focus`、CLI + TUI）

`display.focus_view: true` は**フォーカスビュー**を有効にします — 逐一の実況ではなく回答だけが欲しいときのための、出力を減らした表示モードです。これは 2 つ目の抑制経路ではなく、同じ `tool_progress` の仕組みの上にある薄い層です。

- オンにすると `tool_progress` が `off` に固定され、以前のモードは `display.focus_saved_tool_progress` に退避されます。
- `/focus off` はそのモードを正確に復元するため、`/verbose verbose` の設定は往復しても維持されます。
- 完了した各ターンの終わりには薄い色の復旧行 — `⋯ 7 tool lines hidden · /focus off to show` — が付きます。これは*フォーカス前*のモードを基準に数えられるため、すでにオフにしていた行を隠したと主張することはありません。
- 永続的な `◉ focus` のバッジがステータスバー（prompt_toolkit の CLI と Ink の TUI の両方）に表示されるため、出力を減らしたモードが見えなくなることはありません。
- フォーカスがオンの間に `/verbose` を切り替えると、モードは `/verbose` に戻され、バッジはクリアされます。

フォーカスビューは**表示専用**です。会話履歴、システムプロンプト、ツールのスキーマ、リクエストのペイロードを編集することは決してありません — 隠された詳細は画面上で抑制されるだけで破棄されることはなく、プロンプトキャッシュにはまったく影響しません。

### ステータスバーのフィールドの選択（CLI/TUI）

CLI/TUI の下部にある対話的なステータスバーには、モデル、コンテキストの使用量、圧縮回数、バックグラウンドのアクティビティのカウンター、タイマー、モードのバッジが表示されます。`display.status_bar.fields` は、それらのうちどれを表示するかを選びます — 最小限のバー（モデル + 所要時間だけ）にしたい場合や、オプトインのセッションのトークン合計を表示したい場合に便利です。

```yaml
display:
  status_bar:
    fields: ["model", "duration", "total_tokens"]   # visibility only; built-in order is preserved
```

サポートされているフィールド: `model`、`context_detail`（使用/合計トークン）、`context_pct`（パーセント + メーター）、`cache_hit`（プロンプトキャッシュのヒット率 — モデルの切り替えと圧縮でリセット）、`latency`（API レイテンシの移動平均、直近 10 回の呼び出し）、`tps`（1 秒あたりの出力トークンの移動平均、直近 10 回の呼び出し）、`compressions`、`bg_tasks`、`bg_processes`、`bg_subagents`、`goal`、`git_branch`（⎇ 作業ディレクトリの現在の git ブランチ — オプトインのみで、デフォルトでは決して表示されません。detached HEAD の場合は短縮したコミットを表示）、`duration`、`prompt_elapsed`、`idle_since`、`focus`、`yolo`、`stash`、`battery`、`title`（右寄せのセッションのバッジ）、そして `total_tokens`（セッションの Σ — オプトインのみで、デフォルトでは決して表示されません）。

注意:

- 空のリスト（デフォルト）では標準の集合 — `total_tokens` と `git_branch` 以外のすべて — が維持されます。
- この設定が制御するのは**表示/非表示であって、順序ではありません**。フィールドは組み込みの位置に表示されます。
- 幅の狭いターミナルでは、設定にかかわらず、広いモード専用のフィールド（`context_detail`、`cache_hit`、`latency`、`tps`、`prompt_elapsed`、`idle_since`）は引き続き省かれます（`cache_hit` は 52 列以上の中間の段階でも表示されます）。
- `latency`/`tps` は、API 呼び出しが記録されるまで非表示のままです（例: Codex の app-server バックエンドはレイテンシを報告しません）。
- ここでの `battery` と `title` の表示/非表示は、それぞれのトグル（`/battery`、`/title`）と組み合わさります — セグメントが表示されるには両方がオンである必要があります。
- 同じキーは **Ink TUI** のステータス行（`hermes tui`）もフィルタリングします。そこでは `cache_hit`、`latency`、`tps` は、それぞれ 96/104/110 列以上のターミナルで、幅の予算に応じた末尾のセグメント（◎ / ◷ / ↑）として表示されます。
- 表示専用です: プロンプトキャッシュやリクエストのペイロードには影響しません。変更は次のセッションの開始時に反映されます。

### ランタイムメタデータのフッター（ゲートウェイのみ）

`display.runtime_footer.enabled: true` の場合、Hermes は各ゲートウェイのターンの**最終**メッセージに、小さなランタイムコンテキストのフッターを追加します。現在のフッターには、モデル、コンテキストウィンドウの使用率、現在の作業ディレクトリを表示できます。デフォルトではオフです。チームがすべての返信にこの出所情報を含めたい場合は、ゲートウェイごとにオプトインしてください。

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # order shown; drop any to hide
```

サポートされているフィールド:

| フィールド | 表示内容 | 例 |
| --- | --- | --- |
| `model` | ベンダーのプレフィックスを除いた素のモデル ID | `gpt-5.4` |
| `context_pct` | 直近の呼び出しのコンテキスト占有率（パーセント） | `5%` |
| `latency` | ターンの実時間の所要時間 | `22s`、`1m05s` |
| `served_model` | 実際に応答したモデル（設定したものと異なる場合）: ルーティングプロキシが `x-litellm-model-id`（または `x-litellm-model-api-base`）のレスポンスヘッダーで報告したデプロイメント、または Hermes がそのターンで切り替えたフォールバックのモデル | `hermes-router → gpt-4o-2024-11-20` |
| `cwd` | ホームからの相対の作業ディレクトリ | `~` |

デフォルトのフィールドの集合は `["model", "context_pct", "cwd"]` です。`latency` と `served_model` はオプトインです — 使うには `fields` に追加してください。`served_model` は、提供されたモデルが設定したものである場合（またはプロキシがそのようなヘッダーを送らない場合）は何も表示しないため、ルーティングプロキシの背後やフォールバックが有効なときには、切り替えを可視化するフィールドになります。データが利用できないフィールドは、空の枠を表示するのではなく暗黙のうちにスキップされます。

`/footer` スラッシュコマンドで、どのセッションでも実行時にこれを切り替えられます。

Telegram/Discord/Slack の返信に追加されるフッターの例:

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

フッターが付くのはターンの**最終**メッセージだけで、途中の更新はすっきりしたままです。その最終メッセージがフッターだけになってしまう場合（テキストがすでに画面に表示されているストリーミングされた返信や、`MEDIA:` の写真だけの返信）、写真のキャプションを持つプラットフォーム（Telegram）では、キャプションの上限（Telegram では 1,024 文字）に収まる場合に限り、フッターは代わりに最後の写真のキャプションに入ります。そうでない場合は、以前と同様に独立した短いメッセージとして送信されます。

### プラットフォームごとの進捗オーバーライド

プラットフォームによって、必要な詳細さは異なります。プラットフォームごとのモードを設定するには `display.platforms` を使います。

```yaml
display:
  tool_progress: all          # global default
  platforms:
    signal:
      tool_progress: 'off'    # Signal cannot currently display tool-progress bubbles
    telegram:
      tool_progress: verbose  # detailed progress on Telegram
    slack:
      tool_progress: 'off'    # quiet in shared Slack workspace
```

CLI からは、正規のパス — `hermes config set display.platforms.telegram.streaming false` — を使ってください。省略形の `hermes config set platforms.telegram.streaming false` も受け付けます: プラットフォームごとの*表示*設定（`streaming`、`show_reasoning`、`tool_progress` など）は `display.platforms` からしか読み込まれないため、`config set`/`get`/`unset` はその省略形を正規のキーにリダイレクトし、注記を表示します。トップレベルの `platforms.<name>` ブロックの下の接続用のキー（`token`、`enabled`、`reply_to_mode`、`extra`）はリダイレクトされません。それらを入れ子のプレフィックスの下に書き込むと（`hermes config set gateway.platforms.telegram.enabled true`）、注記とともにトップレベルの `platforms.telegram.enabled` にリダイレクトされます: ゲートウェイは両方のブロックを読み込みますが、共通のキーではトップレベルのものが優先されるため、入れ子への書き込みは既存のトップレベルの値によって暗黙のうちに隠されてしまうからです。

オーバーライドのないプラットフォームは、グローバルな `tool_progress` の値にフォールバックします。有効なプラットフォームのキー: `telegram`、`discord`、`slack`、`signal`、`whatsapp`、`matrix`、`mattermost`、`email`、`sms`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot`、加えて任意のプラグインのプラットフォームの名前（例: Home Assistant プラグインの `homeassistant`）。レガシーの `display.tool_progress_overrides` キーも後方互換性のために引き続き読み込まれますが、非推奨であり、初回ロード時に `display.platforms` へ移行されます。

設定はプラットフォームごとに保存できるため Signal も有効なプラットフォームのキーとして挙げられていますが、現在の Signal のアダプターは送信済みのメッセージを編集できず、ツールの進捗の吹き出しを表示しません。Signal の `tool_progress` は `off` のままにしてください。各ツール呼び出しをライブで見る必要がある場合は、CLI か、編集が可能なメッセージングプラットフォームを使ってください。

`interim_assistant_messages` はゲートウェイ専用です。有効にすると、Hermes はターンの途中で完了したアシスタントの更新を、別々のチャットメッセージとして送信します。これは `tool_progress` とは独立しており、ゲートウェイのストリーミングを必要としません。無効の場合でも、モデルが `clarify` の質問と同じメッセージ内で書いたテキストは、質問の直前に送信されます。質問はその送信を合計で最大 15 秒待ちます。それを過ぎてもまだ実行中の送信は質問を引き留めないため、非常に遅い接続では、選択肢がその説明の終わりより先に届くことがあります。

`show_commentary`（デフォルト `true`）は、Codex Responses のモデルのコメンタリーチャンネル — これらのモデルが非公開の推論と並行して生成する、洗練された進捗の語り — を制御します。有効にすると、完了した各コメンタリーのメッセージは、見えるターン途中の更新として届けられます（ゲートウェイでは `interim_assistant_messages` も必要です）。追加の語りが煩わしい場合は `false` に設定してください: コメンタリーは推論チャンネルにフォールバックし、`show_reasoning` が有効な場合にのみ表示されます。

## プライバシー

```yaml
privacy:
  redact_pii: false  # Strip PII from LLM context (gateway only)
```

`redact_pii` が `true` の場合、サポートされているプラットフォームでは、ゲートウェイはシステムプロンプトを LLM に送信する前に、個人を特定できる情報を伏せます。

| フィールド | 扱い |
|-------|-----------|
| 電話番号（WhatsApp/Signal でのユーザー ID） | `user_<12-char-sha256>` にハッシュ化 |
| ユーザー ID | `user_<12-char-sha256>` にハッシュ化 |
| チャット ID | 数値部分をハッシュ化し、プラットフォームのプレフィックスは保持（`telegram:<hash>`） |
| ホームチャンネルの ID | 数値部分をハッシュ化 |
| ユーザー名 / ユーザーネーム | **影響なし**（ユーザーが選んだもので、公開されている） |

**プラットフォームのサポート:** 伏せ字化は WhatsApp、Signal、Telegram に適用されます。Discord と Slack は、メンションの仕組み（`<@user_id>`）が LLM のコンテキストに実際の ID を必要とするため除外されています。

ハッシュは決定論的です — 同じユーザーは常に同じハッシュに対応するため、モデルはグループチャットでも引き続きユーザーを区別できます。ルーティングと配信は、内部的には元の値を使います。

### OpenAI Codex のリクエストの身元

OpenAI は、サードパーティの Codex のハーネスに自らの身元を示すことを求めています。
公式の Codex エンドポイントへの ChatGPT で認証されたリクエストは、自動的に
`originator: hermes-agent` と `User-Agent: HermesAgent/<version>` を送信します。
既存の ChatGPT アカウントのヘッダーは保持されます。追加のプロンプトの内容や
テレメトリのリクエストが送信されることはありません。
直接の OpenAI API へのリクエストやカスタムのプロキシエンドポイントは変わりません。

## 音声認識（STT） {#speech-to-text-stt}

```yaml
stt:
  enabled: true                # Auto-transcribe inbound voice messages (default: true)
  echo_transcripts: true       # Post raw transcripts back to the chat as 🎙️ "..." (default: true)
  provider: "local"            # "local" | "groq" | "openai" | "mistral" | "xai" | "elevenlabs" | "deepinfra" | ...
  language: "en"               # Global hint; a nonempty provider language wins. Blank falls back to HERMES_LOCAL_STT_LANGUAGE, then auto-detect
  streaming: false             # live partial text while you speak (openai, xai, elevenlabs); see Voice Mode > Live transcription
  cloud_trim_silence: true     # trim long pauses with ffmpeg before uploading to a cloud provider (default: true)
  cloud_trim_threshold_db: -40 # audio quieter than this counts as silence
  cloud_trim_keep_ms: 300      # how much of each pause survives the trim (keeps natural pacing)
  # prompt: "Hermes, Teknium, Nous Research, kanban"   # Static vocabulary hint (see below)
  local:
    model: "base"              # tiny, base, small, medium, large-v3, turbo
    language: ""               # blank inherits stt.language, then HERMES_LOCAL_STT_LANGUAGE
    initial_prompt: ""         # optional whisper prompt to bias vocabulary/script (e.g. Simplified Chinese)
    vad: true                  # Silero VAD filter (default on) — silence never reaches whisper; false = raw behavior (music/ambient)
    vad_min_silence_ms: 500    # min silence (ms) that splits speech chunks when vad is on
    no_speech_prob_threshold: 0.6  # drop a segment only when no_speech_prob > this...
    logprob_threshold: -1.0        # ...AND avg_logprob < this (both must hit — quiet real speech survives)
    unload_after_idle_seconds: 0   # 0=never unload (default); e.g. 300 = release the model after 5min idle
  groq:
    language: ""               # blank inherits stt.language, then HERMES_LOCAL_STT_LANGUAGE
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe | gpt-transcribe
    language: ""               # blank inherits stt.language, then HERMES_LOCAL_STT_LANGUAGE
    timeout: 60                # seconds per transcription request; raise for self-hosted model cold starts
    max_retries: 1             # SDK transport retries (connection errors, 408/409/429/5xx); 0 = single attempt
  xai:
    model: "grok-voice-transcribe-2.0"  # or grok-voice-transcribe-1.0; sent on every request
    language: ""               # blank inherits stt.language, then HERMES_LOCAL_STT_LANGUAGE
  # model: "whisper-1"         # Legacy fallback key still respected
```

共通の言語リゾルバーは、`stt.<provider>.language`、次に `stt.language`、次に `HERMES_LOCAL_STT_LANGUAGE` を確認し、**最初の空でない文字列**を使います。

```text
stt.<provider>.language -> stt.language -> HERMES_LOCAL_STT_LANGUAGE -> no language hint
```

空の値は、「自動検出を強制する」ではなく、**次のソースに進む**という意味です。ElevenLabs は、プロバイダー固有のエイリアスとして `stt.elevenlabs.language_code` も受け付けます。**グローバルのデフォルトは `stt.language: "en"`** であるため、プロバイダーの言語だけを空にしても英語に解決されます。別の言語に固定するには、`"zh"`、`"es"`、`"uk"` などのプロバイダーまたはグローバルの言語コードを設定してください。

リクエストレベルのプラグインのフックによるオーバーライドがない場合:

| プロバイダーの言語 | グローバルの言語 | `HERMES_LOCAL_STT_LANGUAGE` | 解決されるヒント |
| --- | --- | --- | --- |
| `""` | `"en"` | 未設定 | `en` |
| `"zh"` | `""` | 未設定 | `zh` |
| `""` | `""` | `en` | `en` |
| `""` | `""` | 未設定 | なし |

**ローカルの faster-whisper** で言語の自動検出を行うには、次を使います。

```yaml
stt:
  provider: local
  language: ""
  local:
    language: ""
```

また、プロセス環境または `.env` から `HERMES_LOCAL_STT_LANGUAGE` のオーバーライドを取り除いてください。空でないプロバイダーの設定や `pre_transcription` フックによって、引き続き言語が選択される場合があります。

ヒントが解決されない場合、faster-whisper とネイティブのクラウドのハンドラーは言語の引数を省略し、プラグインのプロバイダーは `None` を受け取ります。

コマンドのバックエンドには追加の動作があります。`local_command` は、プロバイダーの設定として `stt.local.language` を使います。名前付きのコマンドプロバイダーは、まずそのコマンド設定（`stt.providers.<name>`、またはレガシーの `stt.<name>` ブロック）の `language` を使い、その後に共通のリゾルバーにフォールバックします。どちらの経路も、言語が解決されない場合は `{language}` プレースホルダーに `en` を使います。カスタムのテンプレートでは、そのプレースホルダーを省略したり、独自のエンジンの検出オプションを使ったりできます。faster-whisper が利用できないために `local` が Whisper CLI にフォールバックする場合、同じ空の設定でも、現時点では自動検出は保証されません。それらの設定については[コマンドプロバイダー](features/tts.md#stt-custom-command-providers)を参照してください。

`stt.openai.timeout` と `stt.openai.max_retries` は、`openai`、`groq`、`deepinfra` のプロバイダーが共有する OpenAI SDK の文字起こしクライアントの形を決めます（プロバイダーごとの同等のキーはまだなく、SDK はこれらについて環境変数を読み込みません）。デフォルトが以前の固定の 30 秒 / 再試行なしではなく `60` / `1` なのは、セルフホストの OpenAI 互換エンドポイントが最初のリクエストでモデルを読み込むのに 30 秒より長くかかることがあり、以前はその音声メッセージがそのまま失われていたためです。トレードオフとして、到達できないバックエンドは、Hermes が諦めるまで最大で 2 回の試行 × タイムアウトの間、音声メッセージを保持します。以前の形にするには、`timeout: 30` と `max_retries: 0` を設定してください。

ゲートウェイがエージェントのために音声メモを文字起こしするものの、生のトランスクリプトをチャットに投稿し返してはならない場合（たとえば、顧客向けの WhatsApp ボット）は、`stt.echo_transcripts: false` を設定します。

プロバイダーの動作:

- `local` は、あなたのマシンで実行される `faster-whisper` を使います。`python -c "import pm; pm.sync_venv(['stt-whisper'], explicit=True)"` で別途インストールしてください。無音時の幻覚に対する強化はデフォルトで有効です: Silero VAD フィルターによって無音/ノイズが Whisper に届くことはなく、ウィンドウをまたぐ条件付けは無効化され、モデル自身がおそらく発話ではない*かつ*信頼度が低いとフラグを立てたセグメントは破棄されます。発話以外の音声（音楽、環境音）を生の動作で文字起こしするには、`stt.local.vad: false` を設定します。低レイテンシの文字起こしのため、モデルは音声メッセージの間もメモリに読み込まれたままです。アイドル時にモデルを自動的に解放するには、`stt.local.unload_after_idle_seconds`（例: 5 分なら `300`）を設定します。これにより CUDA のホストでは GPU メモリが解放されます（ローカルの LLM が GPU を共有している場合の主な利点です）。CPU ではメモリはプロセスが再利用できるようになりますが、プロセスが他の用途にその領域を必要とするまで、OS から見えるフットプリントは縮小しないことがあります。次の音声メッセージでモデルは透過的に再読み込みされます。
- `groq` は Groq の Whisper 互換エンドポイントを使い、`GROQ_API_KEY` を読み込みます。ヒントを固定するには `stt.groq.language` または `stt.language` を設定します。`HERMES_LOCAL_STT_LANGUAGE` は、両方が空の場合にのみ使われます。
- `openai` は OpenAI の音声 API を使い、`VOICE_TOOLS_OPENAI_KEY` を読み込みます。

クラウドのプロバイダー（groq、openai、mistral、xai、elevenlabs、deepinfra）では、`ffmpeg` がインストールされている場合、デフォルトで**アップロード前の無音のトリミング**が行われます: 音声メモの長い間は、ファイルのアップロード前にクライアント側で詰められ、自然なテンポが保たれるよう各間の `cloud_trim_keep_ms` だけが残されます。音声が短くなるほど、アップロードが速くなり、音声 1 分あたりの課金が下がり、リモートのモデルによる無音時の幻覚が減ります。12 秒未満のクリップは、トリミングを完全にスキップします（そこでは節約が意味を持たず、いくつかのプロバイダーはいずれにせよリクエストごとの最低料金を課金するためです）。トリミングはベストエフォートです — ffmpeg がない、トリミングに失敗する、クリップのほとんどが無音である、あるいはトリミングによる節約が約 10% 未満である場合は、元のファイルがそのままアップロードされます。常に元のファイルをアップロードするには（例: クラウドのプロバイダーで音楽や環境音を文字起こしする場合）、`stt.cloud_trim_silence: false` を設定します。コマンド型やプラグインのプロバイダーが、トリミングされた音声を受け取ることはありません。

プロバイダーのリクエストごとのアップロード上限（OpenAI/Groq/DeepInfra では 25 MB、Mistral と xAI では 500 MB、ElevenLabs では約 5 GB。加えて OpenAI の `gpt-4o-transcribe` ファミリーと `whisper-1` にはモデルごとの長さの上限があります）を超える録音は、コンパクトな AAC に再エンコードされ、それでも大きすぎる場合は間で分割され、1 つずつ文字起こしされます。[長い録音とアップロードの上限](features/voice-mode.md#long-recordings-and-upload-limits)を参照してください。

明示的に選択された `stt.provider` は厳密に尊重されます — 利用できない場合、文字起こしはプロバイダーを切り替えるのではなく、`hermes tools` を実行するよう案内してエラーになります。プロバイダーが一度も選択されていない場合にのみ、Hermes は次の順序で自動検出します: `local` → `groq` → `openai`。

Groq と OpenAI のモデルのオーバーライドは環境変数で指定します。

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
STT_XAI_MODEL=grok-voice-transcribe-2.0
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

### 文字起こしのプロンプト（語彙のヒント） {#transcription-prompt-vocabulary-hints}

`stt.prompt` は、プロンプトに対応した STT のバックエンドに渡される、オプションの静的なヒントです。Whisper 系のモデルが聞き間違えやすい固有名詞、製品名、専門用語に使ってください。


```yaml
stt:
  provider: "local"
  prompt: "Hermes, Teknium, Nous Research, kanban, Ollama"
```

**合成。** 設定値がベースになります。[`pre_transcription`](./features/hooks.md#pre_transcription) フックを登録したプラグインは、その上に変更を加え、フィールドごとに最後に書き込んだものが優先されます。複数のプラグインのヒントは決定論的に合成されます: プラグインの探索はプラグイン ID でソートした順にプラグインを読み込み、各プラグインのコールバックはそれ自身の登録順に実行されるため、同じプラグインの集合からは常に同じ最終的なプロンプトが生成されます。`prompt` に空文字列を返すフックは、そのリクエストについて設定のプロンプトをクリアします。フックは `language` と `model` もオーバーライドできます。`file_path` は読み取り専用で、それを変更しようとする試みはログに記録されて破棄されます。フックが登録されておらず、`stt.prompt` も設定されていない場合、送信されるリクエストは以前のリリースと同一です。

**プロバイダーのサポート。**

| プロバイダー | プロンプトのパラメータ | 動作 |
|----------|-----------------|----------|
| `local`（faster-whisper） | `initial_prompt` | ローカルのモデルにそのまま転送 |
| `openai` | `prompt` | 文字起こしのリクエストでそのまま転送 |
| `groq` | `prompt` | 文字起こしのリクエストでそのまま転送 |
| `mistral` | `prompt` | 文字起こしのリクエストでそのまま転送 |
| `deepinfra` | `prompt` | OpenAI 互換の経路で、そのまま転送 |
| `xai` | サポートなし | DEBUG でログに記録され、リクエストはプロンプトなしで続行 |
| `elevenlabs` | サポートなし | DEBUG でログに記録され、リクエストはプロンプトなしで続行 |
| `local_command` | サポートなし | DEBUG でログに記録され、リクエストはプロンプトなしで続行 |
| `type: command` の `stt.providers.<name>` | サポートなし | DEBUG でログに記録され、リクエストはプロンプトなしで続行 |
| プラグインが登録したプロバイダー | `transcribe(**extra)` の kwargs の `prompt` | プロンプトが設定されている場合にのみ送信されるため、このキーより前からあるプロバイダーの呼び出しは変わらない |

**長さ。** Whisper 系のモデルが条件付けに使うのは、プロンプトの最後の約 224 トークンだけです。Whisper 系のバックエンド（`local`、`openai`、`groq`、`deepinfra`）では、Hermes はその上限をクライアント側で強制します: 長すぎる最終的なプロンプトは、警告をログに記録したうえで末尾だけに切り詰められます — プロンプトの長さが原因でリクエストがエラーになることはありません。その他のバックエンド（`mistral`、プラグインのプロバイダー）はプロンプトをそのまま受け取り、独自に検証します。いずれにせよ、ヒントは短く具体的にしてください。

:::warning プロンプトは音声とともにアップロードされる
最終的なプロンプトは、音声ファイルとともに設定された STT プロバイダーに送信されます。特にプロバイダーがローカルの `faster-whisper` ではなくホストされた API である場合は、シークレットやセッションに由来するコンテキストを `stt.prompt` や `pre_transcription` フックが返す内容に含めないでください。
:::

## 音声モード（CLI）

```yaml
voice:
  record_key: "ctrl+b"         # Push-to-talk key inside the CLI
  max_recording_seconds: 120    # Hard stop for long recordings
  auto_tts: false               # Enable spoken replies automatically when /voice on
  beep_enabled: true            # Play record start/stop beeps in CLI voice mode
  beep_volume: 0.3              # Beep amplitude (0.0-1.0); raise it on quiet systems / headphones
  silence_threshold: 200        # RMS threshold for speech detection
  silence_duration: 3.0         # Seconds of silence before auto-stop
```

CLI で `/voice on` を使うとマイクモードが有効になり、`record_key` で録音を開始/停止し、`/voice tts` で音声による返信を切り替えます。エンドツーエンドのセットアップとプラットフォーム固有の動作については、[音声モード](./features/voice-mode.md)を参照してください。

## ストリーミング

完全なレスポンスを待つ代わりに、トークンが届くたびにターミナルやメッセージングプラットフォームへストリーミングします。

### CLI ストリーミング

```yaml
display:
  streaming: true         # Stream tokens to terminal in real-time
  show_reasoning: true    # Also stream reasoning/thinking tokens (optional)
```

有効にすると、レスポンスはストリーミングボックスの中にトークンごとに表示されます。ツール呼び出しは引き続き無言で捕捉されます。プロバイダーがストリーミングをサポートしていない場合は、自動的に通常の表示にフォールバックします。

### ゲートウェイのストリーミング（Telegram、Discord、Slack）

```yaml
streaming:
  enabled: true           # Enable progressive message editing (default: false)
  transport: auto         # "auto" (default) | "edit" (progressive message editing) | "off"
  edit_interval: 0.8      # Seconds between message edits (default: 0.8)
  buffer_threshold: 24    # Characters before forcing an edit flush (default: 24)
  cursor: " ▉"            # Cursor shown during streaming
  fresh_final_after_seconds: 0    # Opt in to fresh final (Telegram) when preview is this old
```

有効にすると、ボットは最初のトークンでメッセージを送信し、その後トークンが届くにつれて段階的にそれを編集します。メッセージの編集をサポートしないプラットフォーム（Signal、Email、Home Assistant）は最初の試行で自動的に検出され — そのセッションではストリーミングが穏やかに無効化され、メッセージが大量に送られることはありません。

段階的なトークンの編集なしで、ターン途中のアシスタントの自然な更新を別々に送りたい場合は、`display.interim_assistant_messages: true` を設定してください。

**オーバーフローの処理:** ストリーミングされたテキストがプラットフォームのメッセージ長の上限（約 4096 文字）を超えると、現在のメッセージが確定され、新しいメッセージが自動的に開始されます。

**新しい最終メッセージ（Telegram）:** Telegram の `editMessageText` は元のメッセージのタイムスタンプを保持するため、長時間かかるストリーミングの返信は、完了後も最初のトークンのタイムスタンプのままになります。古いプレビューを、プレビューのベストエフォートの削除とともに真新しい最終メッセージとして届けるようオプトインするには、`fresh_final_after_seconds > 0` を設定します。デフォルトは `0` で、ストリーミングされた返信を常にその場で確定させ、両方の操作を表示するクライアントでの一瞬の重複メッセージ/削除の流れを避けます。

:::note プラットフォームごとのストリーミングのデフォルト
マスターの `streaming.enabled` スイッチはデフォルトで `false` です — 切り替えるまで何もストリーミングされません。有効にすると、ストリーミングは**プラットフォームごと**に決まります: Telegram は `display.platforms.telegram.streaming: true`（ストリーミングする）で、Discord は `display.platforms.discord.streaming: false`（しない）で出荷されています。そのため、ストリーミングを有効にした後、Telegram はそのままストリーミングし、Discord はトグルを変更するまでメッセージ全体での返信のままです。これらのプラットフォームごとのスイッチは、ダッシュボードの **Channels** のトグルから、または `~/.hermes/config.yaml` で直接調整できます。それらのプラットフォームごとのトグルはメッセージのたびに再読み込みされるため、再起動なしで適用されます。`streaming:` ブロック自体（`enabled`、`transport`、`edit_interval`、`buffer_threshold`、`cursor`、`fresh_final_after_seconds`）はゲートウェイの起動時に一度だけ取り込まれます: それらのキーを変更した後は、変更を反映させるためにゲートウェイを再起動してください。プラットフォームごとのスイッチはマスタースイッチを狭めることしかできず、単独でストリーミングを有効にすることはできない点に注意してください。`gateway.multiplex_profiles` の下では、提供される各プロファイルのターンは、ホストのプロファイルではなく、そのプロファイル自身の `config.yaml` の `streaming:` ブロック（`hermes -p <profile> config set streaming.enabled true`）を使います。独自の `streaming:` ブロックを持たない提供されるプロファイルはデフォルト（オフ）になります: ホストのプロファイルがストリーミングするからといって、それもストリーミングすることはもうありません。
:::

## グループチャットのセッション分離

CLI、TUI/ダッシュボード、メッセージングゲートウェイにわたって、同時にアクティブに開いていられる
チャットセッションの数を制限します。

```yaml
max_concurrent_sessions: null  # null/0 = unlimited; positive integer = active session cap
```

スロットが使われるのは、チャットのウィンドウが開かれたときではなく、セッションが**最初のターン**を
実行したときです。チャットを開く、再開する、再接続するだけでは、メッセージを送るまで何も消費しないため、
アイドル状態のデスクトップのタブ（や、不安定な websocket が引き起こすバックグラウンドの再開）が、
この上限を共有するメッセージングゲートウェイを枯渇させることはありません。

上限に達すると、Hermes はどのサーフェスがスロットを保持しているかを示す、直接的な上限のメッセージを返します。
既存のアクティブなセッションは通常の動作を維持します。
現在のスロットの使用状況とすべての保持者を見るには、`hermes status --full` を実行してください。

これはゲートウェイの同時ターンに対する唯一の上限です: ゲートウェイは各ターンの本体を
それぞれのスレッドで実行するため、デフォルト（未設定）では、受け付けたすべてのターンが、
他の実行中のターンの後ろにキューされるのではなく即座に開始されます。

正規のキーはトップレベルの `max_concurrent_sessions` です。Hermes はフォールバックとして
`gateway.max_concurrent_sessions` も受け付けますが、両方が設定されている場合はトップレベルのキーが優先されます。

この上限はローカルの実行時のリースファイルで強制され、ベストエフォートです: レジストリを読み取れない、
またはロックできない場合、ユーザーが立ち往生しないよう Hermes はフェイルオープンで動作します。
これは単一のホスト/プロファイルの実行環境を想定したもので、複数のマシンにまたがってマウントされた共有の
`$HERMES_HOME` 向けではありません。所有プロセスは存在するものの生存を証明できないリース（たとえば、
`hermes update` がバックエンドを再起動した後のコンテナ内の読み取れない `/proc` のエントリ）は、引き続き
上限に対してカウントされ、自身のセッション ID も引き続き保護しますが、別のセッションの取得や解放を
ブロックすることはもうありません。

共有チャットで、ルームごとに 1 つの会話を保持するか、参加者ごとに 1 つの会話を保持するかを制御します。

```yaml
group_sessions_per_user: true  # true = per-user isolation in groups/channels, false = one shared session per chat
```

- `true` がデフォルトで推奨の設定です。Discord のチャンネル、Telegram のグループ、Slack のチャンネルなどの共有コンテキストでは、プラットフォームがユーザー ID を提供する場合、各送信者が独自のセッションを持ちます。
- `false` は以前の共有ルームの動作に戻します。Hermes にチャンネルを 1 つの共同作業の会話として明示的に扱ってほしい場合には有用ですが、ユーザーがコンテキスト、トークンのコスト、割り込みの状態を共有することにもなります。
- ダイレクトメッセージには影響しません。Hermes は引き続き通常どおり、チャット/DM の ID で DM を識別します。
- スレッドは、どちらの場合も親チャンネルから分離されたままです。`true` の場合は、スレッド内でも各参加者が独自のセッションを持ちます。

動作の詳細と例については、[セッション](./sessions.md)と [Discord ガイド](./messaging/discord.md)を参照してください。

## ネイティブのホストされたルームのディスカッションの上限

サーバーがホストするネイティブのルームは、各ディスカッション内のすべてのメンバーで、これらの
ディスカッションの上限を共有します。これらはサービスが所有するすべてのネイティブのホストされたルームに適用され、
Desktop のレンダラー駆動のルームやミラーされたルームには適用されません。

```yaml
hosted_rooms:
  discussion:
    max_rounds: 3      # Default maximum scheduling rounds per discussion
    max_messages: 10   # Default total visible bot replies across all members
```

どちらのキーも省略可能です。省略したキーは、後方互換のデフォルトである
3 ラウンドと 10 件の返信を維持します。値は正の整数である必要があります: `0`、負の値、
`null`、真偽値、文字列（引用符で囲んだ数値や `"unlimited"` を含む）、浮動小数点数は
受け付けられません。無効な値は、ゲートウェイのログに警告を出して無視され、
デフォルトが適用されるため、タイプミスでグループチャットが停止することはありません。たとえば、
より大きくても上限のあるディスカッションの予算にするには `max_rounds: 6` と `max_messages: 30` を設定します
（すると 5 人のメンバーがいるルームでも 10 件の返信での停止を超えられます）。

これらのキーは、個々のルームメンバーのプロファイルの設定ではなく、ルームのストアを保持する Hermes の
ホーム（`shared-state.db` の隣にあるルートのホーム）の `config.yaml` に設定してください。これらは
ゲートウェイが提供するルームにも、Desktop/TUI のバックエンドが提供するルームにも同様に適用されます。
各サーバープロセスは起動時にこれらを一度だけ読み込みます: 変更した後は、そのルームのストアを
共有するすべてのプロセスを再起動してください。

これらは上限であって、目標やタスクの完了の保証ではありません。ディスカッションは
早く落ち着くこともあり、Stop と承認のゲートは引き続き有効です。いずれかに達すると
ディスカッションは一時停止します。ルームでの **Continue**（`groups.continue`）は、`max_rounds` と
`max_messages` の予算をもう 1 回分与えます。これはディスカッションごとに最大 5 回までです。

誰が答えるかを自動で選ぶように設定されたルーム（ルームの設定、`routing: auto`）は、
宛先のないメッセージごとに、`auxiliary.group_router` を通じて短い JSON のモデル呼び出しを 1 回行います
（他のすべての補助タスクと同様に provider/model/timeout を持ち、デフォルトのタイムアウトは 20 秒）。

## 未認可の DM の動作

未知のユーザーがダイレクトメッセージを送ってきたときに Hermes がどうするかを制御します。

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` はチャット形式の DM プラットフォームのデフォルトです。Hermes はアクセスを拒否しますが、DM で 1 回限りのペアリングコードを返信します。
- `ignore` は未認可の DM を暗黙のうちに破棄します。
- `decline` は、ペアリングコードの代わりに短く丁寧な断りのメッセージを 1 回送り、その後 24 時間はその送信者に対して何も応答しません。デフォルトの文言をオーバーライドするには:

  ```yaml
  unauthorized_dm_behavior: decline
  unauthorized_dm_decline_message: "Sorry, this assistant is private."
  ```

  `hermes gateway setup` は、許可リストを空のままにすると、これを「Politely decline unknown senders」として提示します。これは `platforms.<platform>.unauthorized_dm_behavior: decline` を書き込みます。

- Email は、受信箱に無関係な未読メールが含まれていることがあるため、`platforms.email.unauthorized_dm_behavior: pair` が設定されていない限り、デフォルトで `ignore` です。
- プラットフォームのセクションはグローバルのデフォルトをオーバーライドするため、全体ではペアリングを有効にしたまま、特定のプラットフォームだけを静かにすることができます。


## クイックコマンド {#quick-commands}

LLM を呼び出さずにシェルコマンドを実行する、またはあるスラッシュコマンドを別のものの別名にする、カスタムコマンドを定義します。exec のクイックコマンドはトークンを消費せず、メッセージングプラットフォーム（Telegram、Discord など）からサーバーを手早く確認したり、ユーティリティスクリプトを実行したりするのに便利です。

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  disk:
    type: exec
    command: df -h /
  update:
    type: exec
    command: hermes update
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

使い方: CLI または任意のメッセージングプラットフォームで `/status`、`/disk`、`/update`、`/gpu`、`/restart` と入力します。`exec` コマンドはホスト上でローカルに実行され、出力を直接返します — LLM の呼び出しはなく、トークンも消費されません。`alias` コマンドは、設定されたスラッシュコマンドのターゲットに書き換えられます。

- **30 秒のタイムアウト** — 長時間実行されるコマンドは、エラーメッセージとともに強制終了されます
- **優先順位** — クイックコマンドはスキルのコマンドより先に確認されるため、スキル名をオーバーライドできます
- **オートコンプリート** — クイックコマンドはディスパッチ時に解決され、組み込みのスラッシュコマンドのオートコンプリートの表には表示されません
- **タイプ** — サポートされるタイプは `exec` と `alias` です。それ以外のタイプはエラーを表示します
- **どこでも動作** — CLI、Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant

文字列だけのプロンプトのショートカットは、有効なクイックコマンドではありません。再利用可能なプロンプトのワークフローには、スキルを作成するか、既存のスラッシュコマンドへの別名を使ってください。

## ヒューマンディレイ

メッセージングプラットフォームで、人間らしい応答のペースをシミュレートします。

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 800                  # Minimum delay (custom mode)
  max_ms: 2500                 # Maximum delay (custom mode)
```

各プロファイル自身の `config.yaml` が読み込まれるため、多重化されたプロファイルはそれぞれ独立したペースを保ちます。プロセス環境によるオーバーライドはありません。`custom` モードでは、整数でない、負の、または大小が逆の `min_ms`/`max_ms` の組は、キーを示す警告とともに拒否され、代わりに `natural` の範囲（800〜2500 ms）が使われます。

## コード実行

`execute_code` ツールを設定します。

```yaml
code_execution:
  mode: project                # project (default) | strict
  timeout: 300                 # Max execution time in seconds
  max_tool_calls: 50           # Max tool calls within code execution
```

**`mode`** は、スクリプトの作業ディレクトリと Python インタープリターを制御します。

- **`project`**（デフォルト） — スクリプトは、アクティブな virtualenv/conda 環境の python を使って、セッションの作業ディレクトリで実行されます。プロジェクトの依存関係（`pandas`、`torch`、プロジェクトのパッケージ）と相対パス（`.env`、`./data.csv`）は自然に解決され、`terminal()` から見えるものと一致します。
- **`strict`** — スクリプトは、`sys.executable`（Hermes 自身の python）を使って、一時的なステージングディレクトリで実行されます。再現性は最大になりますが、プロジェクトの依存関係と相対パスは解決されません。

環境のスクラブ（`*_API_KEY`、`*_TOKEN`、`*_SECRET`、`*_PASSWORD`、`*_CREDENTIAL`、`*_PASSWD`、`*_AUTH` を取り除く）とツールのホワイトリストは、どちらのモードでも同じように適用されます — モードを切り替えてもセキュリティの態勢は変わりません。

## Web 検索バックエンド

`web_search` と `web_extract` ツールは、5 つのバックエンドプロバイダーをサポートしています。バックエンドは `config.yaml` で、または `hermes tools` で設定します。

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | perplexity | keenable | exa

  # Or use per-capability keys to mix providers (e.g. free search + paid extract):
  search_backend: "searxng"
  extract_backend: "firecrawl"

  # Keyless free-tier fallback (default: true). With no backend configured
  # and no API keys present, web tools rotate across the Exa/Parallel/
  # Firecrawl/Keenable free tiers. Set false to disable.
  keyless_fallback: true

  # One-shot keyless rescue (default: true). When the chosen/keyed backend
  # fails a call, that single call retries on the keyless ring; the next
  # call attempts the chosen backend again (never sticky).
  keyless_rescue: true

  # Pin Exa/Parallel to a tier (set by the hermes tools Free/Paid rows).
  # free = always the anonymous endpoint; paid = always the keyed SDK path;
  # unset = auto (key present -> paid, otherwise free).
  provider_tier:
    parallel: free
    exa: paid
```

| バックエンド | 環境変数 | 検索 | 抽出 |
|---------|---------|--------|---------|
| **Firecrawl**（デフォルト） | `FIRECRAWL_API_KEY` | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY`（任意 — キーなしの無料ティア） | ✔ | ✔ |
| **Tavily** | `TAVILY_API_KEY`（任意 — 選択時はキーなしで利用可） | ✔ | ✔ |
| **Perplexity** | `PERPLEXITY_API_KEY` | ✔ | ✔（クエリに関連するスニペット） |
| **Exa** | `EXA_API_KEY`（任意 — キーなしの無料ティア） | ✔ | ✔ |

**バックエンドの選択:** ランタイムは常に、保存された `web.backend` の選択（`hermes tools` で設定。`nous` は管理された Tool Gateway を通じてルーティングされます）を使います。Web のバックエンドが一度も選択されていない場合にのみ、利用可能な API キーから自動検出されます: `SEARXNG_URL` だけが設定されていれば SearXNG、`EXA_API_KEY` だけなら Exa、`TAVILY_API_KEY` だけなら Tavily、`PERPLEXITY_API_KEY` だけなら Perplexity、`PARALLEL_API_KEY` だけなら Parallel、`KEENABLE_API_KEY` だけなら Keenable が使われます。**選択も認証情報もまったくない**場合、リクエストはキーなしの無料ティアのリング（Exa / Parallel / Firecrawl / Keenable）をラウンドロビンで巡回し、レート制限時（検索ではベンダーの拒否時も）には自動的に次の候補へフェイルオーバーします — 詳細は [Web 検索ガイド](./features/web-search.md)を参照してください。選択が存在すると、`.env` にキーを追加しても経路は変わりません。`hermes tools` で Tavily、Firecrawl、Keenable を選択した場合も、キーなしで機能します。

**SearXNG** は、70 以上の検索エンジンに問い合わせる、無料でセルフホスト型の、プライバシーを尊重するメタ検索エンジンです。API キーは不要です — `SEARXNG_URL` を自分のインスタンス（例: `http://localhost:8080`）に設定するだけです。SearXNG は検索専用です。`web_extract` には別の抽出プロバイダーが必要です（`web.extract_backend` を設定）。Docker でのセットアップ手順は [Web 検索のセットアップガイド](./features/web-search.md)を参照してください。

**セルフホストの Firecrawl:** 自分のインスタンスを指すように `FIRECRAWL_API_URL` を設定します。カスタムの URL が設定されている場合、API キーは任意になります（認証を無効にするには、サーバーで `USE_DB_AUTHENTICATION=*** を設定します）。

**Parallel の検索モード:** 検索の動作を制御するには `PARALLEL_SEARCH_MODE` を設定します — `fast`、`one-shot`、または `agentic`（デフォルト: `agentic`）。

**Exa:** `~/.hermes/.env` に `EXA_API_KEY` を設定します。`category` によるフィルタリング（`company`、`research paper`、`news`、`people`、`personal site`、`pdf`）とドメイン/日付のフィルターをサポートしています。

## ブラウザ

ブラウザ自動化の動作を設定します。ブラウザのキーはすべて `browser:` の下にあります。
ブラウザのソース（ローカルの Chromium、クラウドプロバイダー、Lightpanda、`/browser connect`、実際のプロファイル）が
どのように組み合わさるかについては、[ブラウザ機能のページ](./features/browser.md)を参照してください。

```yaml
browser:
  inactivity_timeout: 120        # Seconds before an idle browser is closed
  command_timeout: 30            # Chromium launch timeout floor (seconds)
  headed: false                  # true = visible Chromium window (local); keeps the browser open between turns
  engine: auto                   # auto (Chromium) | lightpanda ("chrome" is accepted as auto)
  # Local Chromium launched by Hermes
  executable_path: ""            # Chromium binary override; "" = the packaged Chromium
  user_data_dir: ""              # Persistent profile shared by every session; "" = fresh profile per session
  chromium_args: []              # Extra Chromium command-line flags, e.g. ["--lang=de-DE"]
  # Cloud browser selection (written by `hermes tools` → Browser Automation)
  cloud_provider: local          # local | browserbase | browser-use | firecrawl | nous | <provider plugin>; unset = autodetect from credentials
  auto_local_for_private_urls: true  # With a cloud provider, open LAN/localhost URLs in a local Chromium
                                     # (ignores allow_private_urls; false = refuse them instead)
  allow_private_urls: false      # Legacy alias of security.allow_private_urls (guards cloud/CDP browsers, not the local Chromium)
  # Optional CDP override — when set, the browser tools attach to your own
  # Chromium-family browser (as /browser connect does) instead of launching one.
  cdp_url: ""
  use_real_profile: false        # Browse on a managed copy of your real browser profile (opt-in)
  # Dialog supervisor — how native JS dialogs (alert / confirm / prompt / beforeunload) are handled
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # Safety auto-dismiss under must_respond (seconds)
```

**実行ファイルの解決。** Hermes は最初に見つかった Chromium を起動します:
`browser.executable_path` → Hermes のパッケージマネージャーがインストールした Chromium
（`hermes pm install chromium`） → `/etc/hermes/chromium-executable-path`（公式の
Docker イメージ） → `PATH` 上のシステムの `chromium` / `chromium-browser`（Termux や、
パッケージ化されたビルドのないプラットフォーム）。どれもない場合、ブラウザツールは暗黙のうちに失敗する
代わりにヒントを報告します。

**ブラウジングをオフにする。** ブラウザの設定のスイッチはありません: `hermes tools`（または
`hermes tools disable browser [--platform NAME]`）で `browser` ツールセットを無効にしてください。

**ダイアログのポリシー:**

- `must_respond`（デフォルト） — ダイアログを捕捉し、エージェントが `browser_dialog(action=...)` を呼ぶのを待ちます。応答のないまま `dialog_timeout_s` 秒が経過すると、ページの JS スレッドが永遠に止まるのを防ぐため、ダイアログは自動的に閉じられます。
- `auto_dismiss` — 捕捉し、即座に閉じます。
- `auto_accept` — 捕捉し、即座に受け入れます。積極的な `beforeunload` プロンプトのあるページで便利です。

ダイアログのワークフロー全体については、[ブラウザ機能のページ](./features/browser.md#browser_dialog)を参照してください。

**削除されたキー（自動的に移行）。** `browser.backend` はなくなりました — `browser_exec` が
唯一のブラウザの経路です。古い `backend: off`（引用符の有無にかかわらず）は、代わりに `browser` ツールセットを
無効にするよう伝える 1 回限りの通知とともに削除されます。この通知は `hermes update` によって、または Hermes が
起動時に設定を移行した場合は最初のセッションの開始時の警告として表示されます。その他の値は、その値と
それを置き換えたもの（`camofox`: Camofox プラグイン、`local`: browser_exec が起動するパッケージ化された Chromium、
クラウドの名前: `browser.cloud_provider`）を示す注記とともに削除されます。
`snapshot_threshold`、`record_sessions`、`allow_unsafe_evaluate`、`restrict_evaluate`
は削除された組み込みのブラウザツールに属していたもので、もう効果はありません。
`browser.camofox.*` のキーは、コアではなく Camofox プラグイン（`hermes plugins install camofox`）が読み込みます。
古い `AGENT_BROWSER_*` 環境変数はもう読み込まれません。アップデート時に、対応するキーが未設定であれば、
それらの値がプロファイルの `.env` から対応するキーにコピーされます
（`AGENT_BROWSER_EXECUTABLE_PATH` → `executable_path`、`AGENT_BROWSER_PROFILE` → `user_data_dir`、
`AGENT_BROWSER_ARGS` → `chromium_args`、`AGENT_BROWSER_HEADED` → `headed`、`AGENT_BROWSER_ENGINE` → `engine`）。
以前は環境変数がそれらをオーバーライドしていたため、`engine: auto` と `headed: false` は未設定とみなされます。
相対パスの `AGENT_BROWSER_PROFILE` は作業ディレクトリに対して解決されていたため、その絶対パス
（設定を移行したプロセスの作業ディレクトリ）として書き込まれます。その際、固定したパスを示す通知が同じ方法で
表示されるため、ブラウザのログインがそこにあるかどうかを確認できます。

## タイムゾーン

IANA のタイムゾーン文字列で、サーバーのローカルのタイムゾーンをオーバーライドします。cron のスケジュールと、システムプロンプトに注入される時刻に影響します。ログファイルは変わりません: `~/.hermes/logs/` のすべての行はマシンのローカル時刻で記録され、`hermes logs --since` はそれと比較します。

```yaml
timezone: "America/New_York"   # IANA timezone (default: "" = server-local time)
```

サポートされる値: 任意の IANA タイムゾーン識別子（例: `America/New_York`、`Europe/London`、`Asia/Kolkata`、`UTC`）。サーバーのローカル時刻にするには、空のままにするか省略します。

`hermes doctor`（および起動時の設定チェック）は、ランタイムが読み込めない値を報告します — そうしないと、`Asia/Tokio` のようなタイプミスによって、エージェントの時計とすべての cron のスケジュールが暗黙のうちにサーバーのローカル時刻になってしまいます。`HERMES_TIMEZONE` は、設定されている場合にこのキーをオーバーライドします。

エージェントの時計、cron のスケジュール、時刻を扱うツールは、どの OS でもこのゾーンに従います。`execute_code` を通じて実行されるコードも、Linux と macOS ではこれを `TZ` として継承します。Windows では、それらの子プロセスは代わりに OS で設定されたゾーンを維持します（Windows の C ランタイムは POSIX 形式の `TZ` 文字列しか解析せず、そこでの IANA の名前は誤った UTC オフセットを生むためです）。そのため、子スクリプトがこのゾーンのローカル時刻を表示する必要がある場合は、Windows のゾーン自体を設定してください。

## Discord

メッセージングゲートウェイの Discord 固有の動作を設定します。

```yaml
discord:
  require_mention: true          # Require @mention to respond in server channels
  free_response_channels: ""     # Comma-separated channel IDs where bot responds without @mention
  auto_thread: true              # Auto-create threads on @mention in channels
  free_response_auto_thread: false  # Free-response channels also auto-thread (default: reply inline)
```

- `require_mention` — `true`（デフォルト）の場合、ボットはサーバーのチャンネルでは `@BotName` でメンションされたときにのみ応答します。DM はメンションなしで常に機能します。
- `free_response_channels` — メンションを必要とせず、ボットがすべてのメッセージに応答するチャンネル ID のカンマ区切りのリスト。
- `auto_thread` — `true`（デフォルト）の場合、チャンネルでのメンションは会話用のスレッドを自動的に作成し、チャンネルをすっきり保ちます（Slack のスレッドと同様）。
- `free_response_auto_thread` — `true` の場合、`free_response_channels` のチャンネルでも、トップレベルのメッセージごとにスレッドが自動作成されます。デフォルトは `false` で、自由応答のチャンネルはインラインで返信します。`auto_thread: true` が必要です。

## セキュリティ

シークレットの伏せ字化とウェブサイトのブロックリスト:

```yaml
security:
  redact_secrets: true           # Redact API key patterns in tool output and logs (on by default)
  website_blocklist:             # See Website Blocklist section below
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` — `true` の場合、ツールの出力が会話のコンテキストやログに入る前に、API キー、トークン、パスワードのように見えるパターンを自動的に検出して伏せ字にします。**デフォルトで有効**です。デバッグや伏せ字化処理の開発のために生の認証情報らしき文字列が必要な場合にのみ、明示的に `false` に設定してください。シークレットを含むファイル（`.env` 形式のファイル、シェルの rc/profile ファイル、`HERMES_HOME` 配下の Hermes の `config.yaml` とその `backups/config/` のコピー）を `read_file`、`search_files`、またはターミナルの `cat`/`grep` で読み取る場合も、値の見た目にかかわらず、認証情報の形をした代入（`SOME_API_TOKEN: …`）は再利用できない `«redacted-secret»` のマーカーで隠されます。通常のソースやプロジェクトの設定ファイルでは、`MAX_TOKENS: 100` のようなフィクスチャが損なわれないよう、ベンダーのプレフィックスのパターンだけが適用されます。

以前のリリースには、同梱のコマンドスキャナー用の `tirith_*` キーもありました。これは削除され、アップグレード時にそれらのキーは取り除かれます。コンテンツレベルのコマンドチェックについては、[コンテンツレベルのコマンドチェック](security.md#content-level-command-checks)で説明しています。


## ウェブサイトブロックリスト {#website-blocklist}

エージェントの Web およびブラウザのツールから、特定のドメインへのアクセスをブロックします。

```yaml
security:
  website_blocklist:
    enabled: false               # Enable URL blocking (default: false)
    domains:                     # List of blocked domain patterns
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # Load additional rules from external files
      - "/etc/hermes/blocked-sites.txt"
```

有効にすると、ブロックされたドメインのパターンに一致する URL は、Web またはブラウザのツールが実行される前に拒否されます。これは `web_search`、`web_extract`、ブラウザのナビゲーション、そして URL にアクセスするあらゆるツールに適用されます。

ドメインのルールがサポートするもの:
- 完全なドメイン: `admin.example.com`
- ワイルドカードのサブドメイン: `*.internal.company.com`（すべてのサブドメインをブロック）
- TLD のワイルドカード: `*.local`

共有ファイルには、1 行に 1 つのドメインのルールを記述します（空行と `#` のコメントは無視されます）。存在しない、または読み取れないファイルは警告をログに記録しますが、他の Web ツールを無効にはしません。

ポリシーは 30 秒間キャッシュされるため、設定の変更は再起動なしで速やかに反映されます。

## スマート承認

潜在的に危険なコマンドを Hermes がどう扱うかを制御します。

```yaml
approvals:
  mode: smart   # smart | manual | off
```

| モード | 動作 |
|------|----------|
| `smart`（デフォルト） | 補助 LLM を使って、フラグの立ったコマンドが実際に危険かどうかを評価します。リスクの低いコマンドは、そのコマンドに限り自動承認されます。本当に危険なコマンドは拒否され、判断が不確かなものはユーザーにエスカレーションされます。 |
| `manual` | フラグの立ったコマンドを実行する前に、ユーザーに確認します。CLI では対話的な承認ダイアログを表示します。メッセージングでは、保留中の承認リクエストをキューに入れます。 |
| `off` | すべての承認チェックをスキップします。`HERMES_YOLO_MODE=true` と同等です。**注意して使用してください。** |

スマートモードは、承認疲れを減らすのに特に役立ちます — 本当に破壊的なコマンドは引き続き捕捉しつつ、安全な操作ではエージェントがより自律的に作業できるようにします。

:::warning
`approvals.mode: off` を設定すると、ターミナルコマンドに対するすべての安全チェックが無効になります。信頼できる、サンドボックス化された環境でのみ使用してください。
:::

### 拒否のサーキットブレーカー

`approvals.denial_breaker_threshold`（デフォルト `3`）は、スマート承認のレビュアーが拒否し続けているコマンドの変形を、エージェントが再試行し続けることを防ぎます — 再試行のたびに、ガーディアン LLM の呼び出しがもう 1 回消費されるためです。セッション内でその回数だけ連続して拒否されると、拒否のメッセージは、停止してブロックされた操作を報告し、あなたに手動で実行するか `/approve` するよう求めるよう、エージェントに伝えるハードストップの指示にエスカレーションされます。承認があるとカウントはリセットされます。無効にするには `0` を設定します。

```yaml
approvals:
  denial_breaker_threshold: 3   # 0 disables the breaker
```

### 拒否ルール

`approvals.deny` は、一致するターミナルコマンドを無条件にブロックする glob パターンのリストです — `--yolo`、`/yolo`、`mode: off` の下でもブロックされます。これは、組み込みの絶対的なブロックリストに対応する、ユーザーが編集できるものです。

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
```

パターンは大文字小文字を区別しない fnmatch の glob で、YAML では引用符で囲む必要があります（先頭が裸の `*` だとパースエラーになります）。詳細は [セキュリティ — ユーザー定義の拒否ルール](./security.md#user-defined-deny-rules-approvalsdeny)を参照してください。

### カスタムのスマート承認ポリシー

`approvals.smart_policy` を使うと、スマート承認のレビュアーの指示に独自のルールを追加できます。設定すると、そのテキストはガーディアン LLM のシステムプロンプト（信頼されたチャンネル — 信頼されないコマンドのテキストと並べられることは決してありません）に追加されるため、コードを編集することなく、あなたの環境に合わせてその判断を厳しくしたり緩めたりできます。

```yaml
approvals:
  smart_policy: |
    Always ESCALATE commands that modify anything under /etc.
    APPROVE docker compose restarts in ~/deploys — they are routine here.
```


## チェックポイント

破壊的なファイル操作の前に、ファイルシステムのスナップショットを自動的に作成します。詳細は[チェックポイントとロールバック](./checkpoints-and-rollback.md)を参照してください。

```yaml
checkpoints:
  enabled: false                 # Enable automatic checkpoints (also: hermes chat --checkpoints). Default: false (opt-in).
  max_snapshots: 20              # Max checkpoints to keep per directory (default: 20)
```


## 委譲 {#delegation}

delegate ツールのサブエージェントの動作を設定します。

```yaml
delegation:
  backend: default                          # Delegation strategy routing each task: "default" = the keys below, or a plugin backend (model_map, profile_routes, ...)
  # backends:                               # Each plugin backend's own settings, keyed by its backend name
  #   model_map:
  #     options: [...]
  # model: "google/gemini-3-flash-preview"  # Override model (empty = inherit parent)
  # provider: "openrouter"                  # Override provider (empty = inherit parent)
  # base_url: "http://localhost:1234/v1"    # Direct OpenAI-compatible endpoint (takes precedence over provider)
  # api_key: "local-key"                    # API key for base_url (falls back to OPENAI_API_KEY)
  # api_mode: ""                            # Wire protocol for base_url: "chat_completions", "codex_responses", or "anthropic_messages". Empty = auto-detect from URL (e.g. /anthropic suffix → anthropic_messages). Set explicitly for non-standard endpoints the heuristic can't detect.
  compression_threshold_tokens: 0          # Optional absolute cap on a subagent's compaction trigger (>= 16000); 0 = off, children use the ratio threshold
  # request_overrides:                      # Per-child request settings sent on every subagent API call (all resolution branches).
  #   extra_body:                           # Merged into the request's extra_body — e.g. OpenRouter routing hints:
  #     provider:
  #       sort: throughput
  max_concurrent_children: 3                # Parallel children per batch (floor 1, no ceiling). Also via DELEGATION_MAX_CONCURRENT_CHILDREN env var.
  prefix_stagger_seconds: 0                 # Prompt-cache warm-up for batches: siblings start when the first child's response starts streaming; N (max 60) only caps that wait, it is not a fixed stagger. 0 = off.
  worktree_isolation: false                 # Give each child its own git worktree branched from HEAD (local backend + git repos only; inspired by Muse Code). See Subagent Delegation → Worktree Isolation.
  max_spawn_depth: 1                        # Delegation tree depth cap (1-3, clamped). 1 = flat (default): parent spawns leaves that cannot delegate. 2 = orchestrator children can spawn leaf grandchildren. 3 = three levels.
  orchestrator_enabled: true                # Global kill switch. When false, role="orchestrator" is ignored and every child is forced to leaf regardless of max_spawn_depth.
  oneshot_max_children: 2                   # Total subagents a one-shot run (hermes chat -q / --oneshot) may spawn; 0 = unlimited. Interactive and gateway sessions are never capped by this.
  resumable_ttl_hours: 48                   # A child stopped by a restart, crash or provider failure stays continuable (delegate_task action=continue) this long, and its worktree is kept from the startup sweep.
  max_resumable: 32                         # Per profile, how many such children keep their worktree pinned; the oldest beyond it loses its pin (the record stays).
  auto_resume: false                        # Opt-in: the gateway resumes children stopped by a limit (at its reset) or a restart/crash, at most twice each, silently (no completion while it waits; see delegate_task list), until you stop it, /stop or /new. Spends tokens unattended. Off = you continue them.
  auto_resume_deadline_minutes: 360         # With auto_resume: give up this long after a child's first stop and send the parent one failure.
  gateway_mode: auto                        # Gateway sessions: auto | inline | background. auto = join children inside the turn (one answer) on a messaging chat whose display.interim_assistant_messages is off. See Subagent Delegation → Messaging chats without interim messages.
```

**委譲の戦略（`backend`）:** `delegation.backend` は、各サブエージェントの経路を決める戦略プラグインを選びます。`default` は以下のキー（それらが空の場合は親の経路）に従います。`model_map`（タスクごとに名前付きのモデル）や `profile_routes`（タスクごとに Hermes のプロファイル）などのプラグインの戦略は、`delegation.backends.<backend name>` から設定を読み込みます。プラグインが存在しないか無効になっているバックエンドはフェイルクローズドで動作します: `delegate_task` は、子を別の経路で実行する代わりに、そのバックエンドと修正方法（`/delegation default`、または `hermes config set delegation.backend default`）を示すツールエラーを返します。独自の `fallback_providers` を持たないバックエンドの経路は、`delegation.fallback_providers` を使います。[サブエージェントの委譲 → 委譲の戦略](features/delegation.md#delegation-strategies-delegationbackend)を参照してください。

**サブエージェントの provider:model のオーバーライド:** デフォルトでは、サブエージェントは親エージェントのプロバイダーとモデルを継承します。サブエージェントを別の provider:model の組にルーティングするには、`delegation.provider` と `delegation.model` を設定します — たとえば、プライマリのエージェントが高価な推論モデルを実行している間、範囲の狭いサブタスクには安価/高速なモデルを使います。

**サブエージェントのフォールバックチェーン:** ワーカーに独自のチェーンを与えるには、`delegation.fallback_providers` を設定します（エントリの形はトップレベルのリストと同じです）。明示的に固定された子（プロバイダー、エンドポイント、またはモデルによって）は、そのチェーンが宣言されている場合にのみそれを使います。そうでない場合は、親エージェントの経路を借りるのではなく、はっきりと失敗します。固定されていない子については、設定がないか `null` の場合は親のチェーンの継承が維持されます。子のフォールバックを完全に無効にするには、`delegation:` の下で `fallback_providers: []` を使います。

**直接のエンドポイントのオーバーライド:** わかりやすいカスタムエンドポイントの経路が欲しい場合は、`delegation.base_url`、`delegation.api_key`、`delegation.model` を設定します。これにより、サブエージェントはその OpenAI 互換のエンドポイントに直接送られ、`delegation.provider` より優先されます。`delegation.api_key` が省略され、`delegation.provider` がプロバイダー（`custom`/`auto` 以外）を指定している場合、`delegation.base_url` はそのプロバイダーのエンドポイントのオーバーライド — メインのエージェントにおける `model.base_url` と同じもの — となり、サブエージェントは親エージェントのキーではなく、そのプロバイダー自身のキー（例: トークンプランの URL 上の `provider: xiaomi` には `XIAOMI_API_KEY`）を使います。名前付きのプロバイダーがない場合、省略された `delegation.api_key` は親エージェントのキーを継承します。`delegation.provider` が `delegation.base_url` と `delegation.api_key` と一緒に設定されている場合、明示的なエンドポイントとキーが優先されますが、そのプロバイダーのリクエスト設定（`custom_providers` のエントリにある `extra_body` のオーバーライドと最大出力トークン数）はサブエージェントに引き継がれます。

**子ごとのリクエスト設定（`request_overrides`）:** `delegation.request_overrides` は、サブエージェントのすべての API 呼び出しで送信されるリクエスト設定の dict です。トップレベルのキーは API の kwargs（例: `service_tier`）で、`extra_body` のサブ dict はリクエストの `extra_body` にマージされます。これは**3 つすべての**解決の分岐 — 直接の `base_url`、名前付きの `provider`、純粋な継承 — で尊重されるため、このキーは常に効果を持ちます。優先順位: 明示的な `request_overrides` の値は、実行時や親から導出されたオーバーライドの**上に**マージされます — トップレベルの明示的なキーが優先され、`extra_body` は 1 段階だけ深くマージされるため、実行時の `extra_body` のキー（例: プロバイダーの `thinking: {type: disabled}` の性格付け）は、あなたのキーが再定義しない限り残ります。典型的な使い方は、委譲の子に対する OpenRouter のルーティングのヒントです。

```yaml
delegation:
  model: "deepseek/deepseek-v4-flash-0731"
  base_url: "https://openrouter.ai/api/v1"
  api_key: "sk-or-..."
  request_overrides:
    extra_body:
      provider:
        sort: throughput   # route children to the fastest OpenRouter provider
```
**ワイヤープロトコル（`api_mode`）:** Hermes は `delegation.base_url` からワイヤープロトコルを自動検出します（例: `/anthropic` で終わるパス → `anthropic_messages`。Codex / ネイティブ Anthropic / Kimi-coding のホスト名は既存の検出を維持します）。ヒューリスティックで分類できないエンドポイント — たとえば Azure AI Foundry、MiniMax、Zhipu GLM、Anthropic 形式のバックエンドの前段にある LiteLLM のプロキシ — では、`delegation.api_mode` を `chat_completions`、`codex_responses`、`anthropic_messages` のいずれかに明示的に設定してください。自動検出を維持するには、空のまま（デフォルト）にします。

委譲のプロバイダーは、CLI/ゲートウェイの起動時と同じ認証情報の解決を使います。設定されているすべてのプロバイダーがサポートされます: `openrouter`、`nous`、`zai`、`kimi-coding`、`minimax`、`minimax-cn`。プロバイダーが設定されると、システムは正しいベース URL、API キー、API モードを自動的に解決します — 認証情報を手動で配線する必要はありません。

**優先順位:** 設定の `delegation.base_url` → 設定の `delegation.provider` → 親のプロバイダー（継承）。設定の `delegation.model` → 親のモデル（継承）。`provider` なしで `model` だけを設定すると、親の認証情報を維持したままモデル名だけが変わります（OpenRouter のような同じプロバイダー内でモデルを切り替えるのに便利です）。

**ワンショットの実行:** 有限の `hermes chat -q` / `--oneshot` のセッションには、委譲した結果を受け取る後のターンも、学習の対象となる後のセッションもないため、より小さなフットプリントで実行されます: スキルのプロンプトはプロセスのスキルではなくドメインのスキルだけを求め（`skill_manage` は引き続き利用できるため、実行中にスキルを作成することはできます）、`oneshot_max_children` は実行が生成できるサブエージェントの合計を制限します（デフォルト `2`、`0` = 無制限）。上限を超えると、`delegate_task` はインラインで作業を終えるようエージェントに伝えるツールエラーを返します。ワンショットには誰も答えたり舵取りしたりできないため、`clarify` ツールなし、システムプロンプトのターン途中の舵取りに関する注記なし、そしてライブ制御のパラメータ（`action` の list/steer/stop、`subagent_id`、`message`）を省いた `delegate_task` のスキーマでも実行されます: その子は呼び出しの中で合流されるため、制御すべき実行中の子が存在することはありません。

**幅と深さ:** `max_concurrent_children` は、バッチごとに並列に実行されるサブエージェントの数を制限します（デフォルト `3`、下限 1、上限なし）。`DELEGATION_MAX_CONCURRENT_CHILDREN` 環境変数でも設定できます。モデルが上限より長い `tasks` 配列を送信すると、`delegate_task` は暗黙のうちに切り詰めるのではなく、上限を説明するツールエラーを返します。`max_spawn_depth` は委譲のツリーの深さを制御します（1〜3 に制限）。デフォルトの `1` では委譲はフラットです: 子は孫を生成できず、`role="orchestrator"` を渡しても暗黙のうちに `leaf` に格下げされます。オーケストレーターの子が葉の孫を生成できるようにするには `2` に、3 階層のツリーにするには `3` に引き上げます。エージェントは呼び出しごとに `role="orchestrator"` でオーケストレーションにオプトインします。`orchestrator_enabled: false` は、それにかかわらずすべての子を葉に戻します。コストは乗算的に増えます — `max_spawn_depth: 3` で `max_concurrent_children: 3` の場合、ツリーは 3×3×3 = 27 の同時実行の葉のエージェントに達し得ます。使い方のパターンについては、[サブエージェントの委譲 → 深さの上限と入れ子のオーケストレーション](features/delegation.md#depth-limit-and-nested-orchestration)を参照してください。

**子プロセスの通知:** サブエージェントが開始したバックグラウンドプロセスは、完了/監視の通知を親の会話にルーティングしますが、そこではデフォルトで**抑制**されます — 成果物は子の集約された結果だからです。それらを（サブエージェントの帰属情報付きで）届けるには、`delegation.surface_child_process_notifications: true` を設定します。委譲の結果自体が抑制されることはありません。[サブエージェントの委譲 → 子のバックグラウンドプロセスの通知](features/delegation.md#child-background-process-notifications)を参照してください。

**バッチのためのプロンプトキャッシュのウォームアップ:** 1 つの `tasks` バッチ内の兄弟は、同じツールとシステムプロンプトの骨組みを送信し、異なるのは目標だけです（タスクが異なる `context` を持つ場合はシステムプロンプトの末尾も）。一緒に開始すると、各子の最初のリクエストはプロバイダーのプロンプトキャッシュに当たらず、プレフィックス全体の料金を支払うことになります。`prefix_stagger_seconds: N` を指定すると、最初の子は直ちに開始し、同じプロバイダー/モデルの兄弟は、その子の最初のレスポンスがストリーミングを始めるまで（その時点でそのリクエストはキャッシュを書き込んでいます）、その子が終了するまで、または N 秒のいずれか早い方まで待ちます。N はその 1 回の待機の上限であって、子どうしの間隔ではありません: N が何であれ、兄弟は通常、先頭の子から最初のトークンまでの時間 1 回分ほど後に開始します。コストはバッチに最初のトークンまでの時間がおよそ 1 回分加わることです。Hermes がプロンプトキャッシュを行わない経路（Anthropic 以外のほとんどのモデル）や、別のモデルにルーティングされた兄弟は待たずに開始します。先頭は常にバッチの最初のタスクです: タスク 0 がキャッシュされない経路または別の経路にある場合、後のタスクがキャッシュされる経路を共有していてもバッチは制御されず、1 タスクの単位（例: `independent_completions` の下）が待つことはありません。デフォルトではオフ（`0`）です。

## 確認（Clarify）

確認の質問への応答を Hermes がどれだけ待つかを設定します。1 つの値がすべてのサーフェス — クラシックな CLI のモーダル、TUI/Desktop のカード、メッセージングゲートウェイ — をカバーします。正規のキーは `agent.clarify_timeout`（デフォルト `90` 秒。`0` 以下 = 無制限）です。レガシーのトップレベルの `clarify.timeout` も、明示的に設定されていれば引き続き尊重されます。

```yaml
agent:
  clarify_timeout: 90          # Seconds to wait for user clarification response (0 or less = unlimited)
```

タイムアウトが切れると、エージェントは `"outcome": "timed_out"` でブロックを解除し（ユーザーがすでに確定させた回答は保持されます）、自力で作業を続けます。メッセージングプラットフォームでは、回答の代わりに別のメッセージを送ると質問は直ちにキャンセルされ、エージェントはそのメッセージで作業を続けます。確認のプロンプトが、ツールごとの汎用的な期限（`timeouts.tools.sequential_call`）によって打ち切られることはありません。待ち時間を制限するのは `agent.clarify_timeout` だけです。

## コンテキストファイル（SOUL.md、AGENTS.md）

Hermes は 2 種類のコンテキストのスコープを使います。

| ファイル | 目的 | スコープ |
|------|---------|-------|
| `SOUL.md` | **エージェントの主要なアイデンティティ** — エージェントが誰であるかを定義する（システムプロンプトのスロット #1） | `~/.hermes/SOUL.md` または `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | プロジェクト固有の指示（最高の優先度） | git のルートまで辿る |
| `AGENTS.md` | プロジェクト固有の指示、コーディング規約 | ディレクトリを再帰的に辿る |
| `CLAUDE.md` | Claude Code のコンテキストファイル（これも検出される） | 作業ディレクトリのみ |
| `.cursorrules` | Cursor IDE のルール（これも検出される） | 作業ディレクトリのみ |
| `.cursor/rules/*.mdc` | Cursor のルールファイル（これも検出される） | 作業ディレクトリのみ |

- **SOUL.md** はエージェントの主要なアイデンティティです。システムプロンプトのスロット #1 を占め、組み込みのデフォルトのアイデンティティを完全に置き換えます。エージェントが誰であるかを完全にカスタマイズするには、これを編集してください。
- SOUL.md が存在しない、空である、または読み込めない場合、Hermes は組み込みのデフォルトのアイデンティティにフォールバックします。
- **プロジェクトのコンテキストファイルは優先順位の仕組みを使います** — 読み込まれるのは 1 種類だけです（最初に一致したものが優先）: `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。SOUL.md は常に独立して読み込まれます。
- **AGENTS.md** は階層的です: サブディレクトリにも AGENTS.md がある場合、すべてが結合されます。
- `SOUL.md` がまだ存在しない場合、Hermes はデフォルトの `SOUL.md` を自動的に用意します。
- 読み込まれるすべてのコンテキストファイルは、スマートな切り詰めとともに `context_file_max_chars` 文字（デフォルト 20,000）に制限されます。

関連項目:
- [パーソナリティと SOUL.md](./features/personality.md)
- [コンテキストファイル](./features/context-files.md)

## 作業ディレクトリ

| コンテキスト | デフォルト |
|---------|---------|
| **CLI（`hermes`）** | コマンドを実行した現在のディレクトリ |
| **メッセージングゲートウェイ** | `~/.hermes/config.yaml` の `terminal.cwd`。未設定の場合はホームディレクトリ `~` |
| **Docker / Singularity / Modal / SSH** | コンテナまたはリモートマシン内のユーザーのホームディレクトリ |

作業ディレクトリをオーバーライドするには:
```yaml
# In ~/.hermes/config.yaml:
terminal:
  cwd: /home/myuser/projects
```

`~/.hermes/.env` 内の `MESSAGING_CWD` と直接の `TERMINAL_CWD` のエントリは、レガシーの互換性のためのフォールバックです。新しい設定では `terminal.cwd` を使ってください。

## ネットワーク

外向きの HTTP に対する接続の回避策:

```yaml
network:
  force_ipv4: false   # Force IPv4 for outbound connections (default: false)
```

`force_ipv4` — IPv6 が壊れている、または到達できないサーバーでは、Python は AAAA レコードを先に解決し、IPv4 にフォールバックする前に TCP のタイムアウトいっぱいまでハングすることがあります。Hermes はすでに、行うすべての外向きの接続で IPv6 と IPv4 を競わせているため（Happy Eyeballs、RFC 8305: IPv4 の試行は IPv6 の 250 ms 後に始まり、先に接続した方が採用されます）、広告されているのに通信が吸い込まれる IPv6 の経路でも、タイムアウトいっぱいではなく接続あたり約 0.25 秒のコストで済みます。これは HTTP だけでなく、ゲートウェイの WebSocket の接続（リレーのコネクター、プラットフォームのアダプター）もカバーします。IPv6 を完全にスキップして IPv4 で直接接続したい場合にのみ、これを `true` に設定してください。`hermes doctor` は、使えない IPv6 の経路を検出してこの設定を案内する `IPv6 route` のチェックを実行します。

## オンボーディング

最初の接触時のオンボーディングのヒントと、最初のメッセージでの提案:

```yaml
onboarding:
  profile_build: "ask"   # "ask" (default) | "off"
  seen: {}               # internal latch — leave empty
  run: null              # desktop first-run onboarding: null (fresh install) | true | false
```

- `profile_build` — TUI、デスクトップアプリ、メッセージングのダイレクトメッセージ（グループチャットでは決してありません）での、史上最初のメッセージにおける提案を制御します。`"ask"`（デフォルト）はユーザープロファイルの作成を提案します。この提案は**オプトインで同意が前提**です — エージェントは何かを調べる前に確認し、接続されたアカウントを黙って読むことは決してなく、確認された事実をユーザーメモリに保存します。`"off"` は提案だけを止めます。エージェントは引き続き簡単な自己紹介をします。提案はプロファイルごとに最大 1 回発火します。
- `seen` — 内部の状態です。Hermes は表示した各ヒントをここに記録し、二度と発火しないようにします。最初のメッセージでの提案も、一度表示されるとここに記録されます。手で編集しないでください — すべてのヒントをもう一度見たい場合は、`onboarding` セクション全体を消去してください。
- `run` — デスクトップアプリが次回の起動時に初回実行のオンボーディングのチャットを開くかどうか。未設定の場合は、新規インストール時（セッションもプロファイルもまだない）のみです。Hermes は、オンボーディングの開始時に `true` を、終了またはスキップ時に `false` を書き込みます。アプリがどのプロファイルを実行していても、これは常にデフォルトプロファイルの `config.yaml` にあります。決して表示しないようにするには `false` を設定します。管理されたインストールでは、管理された設定でこれを固定できます。

## Webapp

ゲートウェイは、[ブラウザで使う Desktop のワークスペース](/docs/user-guide/desktop#use-the-desktop-workspace-from-a-browser)を独自のリスナーで提供できます。オンにするまではオフです。

```yaml
webapp:
  enabled: false     # true: the gateway serves the Webapp
  host: 127.0.0.1    # 127.0.0.1, ::1 or localhost; or 0.0.0.0 / :: for other machines (dashboard sign-in required)
  port: 9120         # fixed, so the browser's address stays the same across gateway restarts
  public_url: ""     # the https:// address a reverse proxy or tunnel serves the Webapp at (sign-in required)
```

- `enabled` — ゲートウェイは起動時にこれを読み込みます。変更した後はゲートウェイを再起動してください（`hermes gateway restart`）。
- `host` — `127.0.0.1`、`::1`、`localhost` では、Webapp はこのマシン上にとどまります。`0.0.0.0` または `::` では他のマシンから到達できるようになり、[ダッシュボードのサインインプロバイダー](./features/web-dashboard.md#authentication-gated-mode)が必要になります。それがない場合、ゲートウェイはネイティブのクライアントのみを提供し、`hermes webapp --status` がその理由を示します。プロバイダーを追加した後はゲートウェイを再起動してください。それ以外の値は、`127.0.0.2` のような別のループバックアドレスも含め、Webapp をオフのままにします。コンテナ内では `0.0.0.0` を使ってください: 公開されたポートや別のコンテナ内のプロキシは、コンテナのループバックに到達できないためです（[Docker → Webapp の実行](./docker.md#running-the-webapp)）。
- `port` — スタンドアロンのプロファイルのゲートウェイには、デフォルトのゲートウェイとは別のポートが必要です。**Settings → Web Access → Start** は、ポートを設定していないスタンドアロンのプロファイルのために 1 つを選び（9120 より上で、他のどのプロファイルも使っていない最初の空きポート）、ここに保存します。あなたが設定したポートを置き換えることは決してありません。ポートが使用中の場合も同様に、ゲートウェイはネイティブのクライアントのみを提供し、`hermes webapp --status` がその理由を報告します。別のプログラムがそれを受け取った可能性があるため、Webapp のブラウザのサインインも置き換えられます。そのため、ポートが空いたら、開いているタブでは再度 `hermes webapp` が必要になります。
- `public_url` — あなたが運用するリバースプロキシやトンネルが、独自のアドレスで Webapp を提供する場合（`host:port` に転送する場合）に設定します。これにより、ループバックの `host` でもダッシュボードのサインインが有効になり、そこでは Webapp がリクエストを受け付ける唯一の外部のホスト名にもなり、サインインのコールバックはこれをもとに構築されます。サインインプロバイダーがない場合、`0.0.0.0` の場合と同様に、ゲートウェイはネイティブのクライアントのみを提供します。これは認証情報、クエリ、フラグメントを含まない `http://` または `https://` のアドレスである必要があります。それ以外のものは Webapp をオフのままにします。`dashboard.public_url` は `hermes dashboard` 用であり、Webapp には決して適用されません。`dashboard.trusted_proxies` は適用されます: Webapp は、プロキシの `X-Forwarded-*` ヘッダーを、ループバックとそこに列挙されたアドレスからのみ受け付けます（[Docker → トンネルまたはリバースプロキシの背後](./docker.md#webapp-behind-a-proxy)）。

サインインは、`host` が `0.0.0.0` または `::` の場合、あるいはホストにかかわらず `public_url` が設定されている場合に有効になります。その場合、`hermes webapp` は、ダッシュボードのアカウントでサインインできるよう Webapp のアドレス（設定されていれば、サインインが戻ってくる `public_url`）を開きます: その使い捨てのリンクで開けるのは、サインインを必要としない Webapp だけです。ネイティブのクライアント（CLI、TUI、Desktop アプリ）がサインインすることはありません: それら向けのゲートウェイのリスナーは、設定にかかわらずサインインを求めることはありません。

Webapp の設定は、そのプロファイルを提供するゲートウェイの設定から読み込まれます: それが提供する名前付きプロファイルについてはデフォルトプロファイルの設定、それ以外の場合はスタンドアロンのプロファイル自身の設定です。

## ダッシュボード

[Web ダッシュボード](./features/web-dashboard.md)の設定 — 見た目のテーマ、公開 URL、認証プロバイダーです。認証プロバイダー（OAuth、基本パスワード、ドレイン）は Web ダッシュボードのページで詳しく説明しています。ここでは `config.yaml` での形を示します。

```yaml
dashboard:
  theme: "default"            # "default" | "midnight" | "ember" | "mono" | "cyberpunk" | "rose"
  show_token_analytics: false # Re-enable the (local-estimate-only) token/cost analytics surfaces
  public_url: ""              # Full public authority for OAuth redirect_uri (env: HERMES_DASHBOARD_PUBLIC_URL)
  trusted_proxies: []         # Proxy IPs/CIDRs allowed to supply X-Forwarded-* headers
  oauth:                      # Portal OAuth gate (engaged with --host and not --insecure)
    client_id: ""             # agent:{instance_id} — Portal provisions this
    portal_url: ""            # blank → plugin default (production Portal)
  basic_auth:                 # Self-hosted username/password gate (dashboard_auth/basic plugin)
    username: ""              # blank → plugin no-op
    password_hash: ""         # scrypt$... (preferred — no plaintext at rest)
    password: ""              # plaintext fallback (hashed in-memory at load)
    secret: ""                # token-signing key; blank → random per-process
    session_ttl_seconds: 0    # 0 → plugin default (12h)
  drain_auth:                 # Drain-control service-credential gate (dashboard_auth/drain plugin)
    scope: "drain"            # capability label on the verified principal
    min_secret_chars: 43      # entropy bar (url-safe-b64 chars; 43 ≈ 256 bits)
  ws_ping_interval: 20.0      # Non-loopback WebSocket keepalive ping interval (seconds)
  ws_ping_timeout: 20.0       # Non-loopback WebSocket keepalive pong timeout (seconds)
  ws_orphan_reap_grace_s: 20.0 # Grace before a WS-detached session is reaped (seconds)
  ssh_isolated_idle_grace_s: 900.0 # Desktop-over-SSH backend exits after this long with no client and no running turn
  ws_orphan_activity_stale_s: 600.0 # Activity idle bound before a detached RUNNING turn is interrupted (seconds)
  startup_orphan_sweep: true  # Close session rows orphaned by a dead gateway process at boot
```

- `theme` — ダッシュボードの見た目のテーマ。
- `show_token_analytics` — デフォルトではオフです。Analytics のページとトークン/コストの数値は**ローカルの下限の推定値**（補助呼び出し、再試行、フォールバック、キャッシュの書き込みを除く）であるため、プロバイダーの請求額を大きく下回ることがあります。それが請求額ではないことを理解している場合にのみ `true` を設定してください。
- `public_url` — 設定すると、これは OAuth の `redirect_uri` を構築するもとになる完全な権限部分（スキーム + ホスト + オプションのパスのプレフィックス）になります。`X-Forwarded-*` ヘッダーを確実に転送しないリバースプロキシの背後にデプロイする場合に設定してください。プロキシのヘッダーからの再構築を使うには、空のままにします。
- `trusted_proxies` — `X-Forwarded-Proto` と `X-Forwarded-For` を提供することが許可された IP アドレス、または範囲の限られた CIDR ネットワーク。ループバックは自動的に信頼されたままです。TLS のリバースプロキシが別のコンテナやホストから接続する場合に設定してください。プロキシの正確な IP を推奨します。アドレスが動的な場合にのみ、小さな専用ネットワークを使ってください。ワイルドカードと `/0` のネットワークは拒否されます。ゲートウェイの [Webapp](#webapp) のリスナーも同じリストを読み込みます。
- `oauth` / `basic_auth` / `drain_auth` — 同梱のダッシュボード認証プラグインが読み込む、認証プロバイダーの設定。ドレインのシークレット自体はここでは設定**しません**。`HERMES_DASHBOARD_DRAIN_SECRET` 環境変数を通じて用意されます。認証の完全なセットアップについては [Web ダッシュボード](./features/web-dashboard.md)を参照してください。
- `ws_ping_interval` / `ws_ping_timeout` — ループバック以外のバインドに対する WebSocket のキープアライブの調整（ループバックの接続が ping を送ることはありません）。20 秒のデフォルトが偽の 1006 切断を生み出し得る、高レイテンシのリンク（Tailscale、遠方の SSH トンネル）では引き上げてください。
- `ssh_isolated_idle_grace_s`（デフォルト `900`） — SSH 経由で到達する、Desktop が所有する `hermes serve --isolated` のバックエンドは、接続中にスリープしたノート PC がそれを停止させられないよう、意図的に SSH セッションから切り離されています。以前は、ダークウェイクの再接続のたびに、`state.db` を保持する別のバックエンドが残されていました。現在、バックエンドは、クライアントの WebSocket がこの時間接続されておらず、エージェントのターンも実行されていない場合に、自ら退役します（ターンはそれを生かし続けます。読み取れないターンの状態もそれを生かし続けます）。ノート PC がスリープした後に、切り離されたバックエンドが長い作業を終えることに頼っている場合は、大きな値に設定してください。そのようなバックエンドは、半開きのトンネルに気づけるよう、ゆっくりとした WebSocket の ping（60 秒、タイムアウト 10 分）も送ります。この猶予とは独立して、リモートホスト上の Desktop の所有権ロックが同じ接続のより新しい起動を示すと、バックエンドは（ターンの合間に）退役するため、再接続の嵐で切り離されたバックエンドが積み重なることはありません。これは POSIX とネイティブの Windows の SSH ホストをカバーします。
- `ws_orphan_reap_grace_s` — WS から切り離されたセッションが、オーファンリーパーに回収されるまで待つ時間。クライアントの再接続が遅い場合は、キープアライブの値とあわせて引き上げてください。定期的なセッションのメンテナンスは、閉じたソケットのクリーンアップも完了させ、欠けているオーファンのタイマーを再設定するため、切り離されたチャットが、最初のクリーンアップやタイマーが失われたというだけで所有権のリースを保持し続けることはありません。再接続するとそのタイマーはキャンセルされます。アクティブな委譲の作業と健全な実行中のターンは、通常のオーファンリーパーのチェックによって引き続き保護されます。（`HERMES_TUI_WS_ORPHAN_REAP_GRACE_S` は内部のオーバーライドとして残っています。）
- `ws_orphan_activity_stale_s`（デフォルト `600`） — 切り離された**実行中**のターンのアクティビティクロック（`agent.turn_liveness` のウォッチドッグがサンプリングするのと同じクロック: API の待機、ストリームのトークン、ツールのハートビート）が、オーファンリーパーがそれを中断するまでにアイドルでなければならない時間。クライアントがいないものの、まだ活発に生成しているターンは、切り離されたまま完了まで実行され続けます — ノート PC を閉じる、モバイルアプリをバックグラウンドにする、デスクトップのアップデートといった操作で、健全な長いターンがキャンセルされることはもうありません。中断されるのは、本当に止まってしまったターンだけです。アクティビティにかかわらず猶予の時間枠で中断するには（以前の動作）、`0` を設定します。
- `startup_orphan_sweep`（デフォルト `true`） — 上記の WS のオーファン回収のタイマーはプロセス内のものであるため、それが発火する前にゲートウェイが再起動する（アップデート、クラッシュ、systemd）と、セッションの行は永遠に開いたままになります — `/resume` やダッシュボードに幽霊の「アクティブな」作業が残ります。ゲートウェイの起動のたびに — stdio の TUI（`entry.main`）と、デスクトップ/ダッシュボードの WebSocket のサイドカー（`handle_ws`）の両方で — ソースが `tui` / `desktop` / `subagent` / `unknown`（トークン集計のガードが自ら作成しなければならなかった行）で、開始時刻**と**最新のメッセージの両方がセッションの TTL（`HERMES_TUI_SESSION_TTL_S`、デフォルト 6 時間）より古い行は、`end_reason: startup_orphan_reap` で閉じられます。メッセージングプラットフォームのセッション（Telegram、Discord など）には決して触れず、メモリ上のライブのセッション（すでに再開したクライアント）は除外され、掃除されたセッションは引き続き再開できます。
