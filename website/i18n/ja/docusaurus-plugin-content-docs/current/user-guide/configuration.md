---
sidebar_position: 2
title: "設定"
description: "Hermes Agent を設定する — config.yaml、プロバイダー、モデル、API キーなど"
---

# 設定

すべての設定は、簡単にアクセスできるよう `~/.hermes/` ディレクトリに保存されます。

## ディレクトリ構成

```text
~/.hermes/
├── config.yaml     # 設定（モデル、ターミナル、TTS、圧縮など）
├── .env            # API キーとシークレット
├── auth.json       # OAuth プロバイダーの認証情報（Nous Portal など）
├── SOUL.md         # エージェントの主要なアイデンティティ（システムプロンプトのスロット #1）
├── memories/       # 永続メモリ（MEMORY.md、USER.md）
├── skills/         # エージェントが作成したスキル（skill_manage ツールで管理）
├── cron/           # スケジュールされたジョブ
├── sessions/       # ゲートウェイのセッション
└── logs/           # ログ（errors.log、gateway.log — シークレットは自動的に伏字化）
```

## 設定の管理

```bash
hermes config              # 現在の設定を表示
hermes config edit         # config.yaml をエディタで開く
hermes config set KEY VAL  # 特定の値を設定
hermes config check        # 不足しているオプションを確認（アップデート後）
hermes config migrate      # 不足しているオプションを対話的に追加

# 例:
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...  # .env に保存される
```

:::tip
`hermes config set` コマンドは、値を自動的に適切なファイルへ振り分けます。API キーは `.env` に、それ以外はすべて `config.yaml` に保存されます。
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

1 つの値の中で複数の参照を使うこともできます: `url: "${HOST}:${PORT}"`。参照先の変数が設定されていない場合、プレースホルダーはそのまま保持されます（`${UNDEFINED_VAR}` はそのまま残ります）。サポートされるのは `${VAR}` 構文のみで、裸の `$VAR` は展開されません。

AI プロバイダーのセットアップ（OpenRouter、Anthropic、Copilot、カスタムエンドポイント、セルフホスト LLM、フォールバックモデルなど）については、[AI プロバイダー](/docs/integrations/providers)を参照してください。

### プロバイダーのタイムアウト

プロバイダー全体のリクエストタイムアウトとして `providers.<id>.request_timeout_seconds` を、加えてモデル固有のオーバーライドとして `providers.<id>.models.<model>.timeout_seconds` を設定できます。これはすべてのトランスポート（OpenAI ワイヤー、ネイティブ Anthropic、Anthropic 互換）におけるプライマリのターンクライアント、フォールバックチェーン、認証情報ローテーション後の再構築、そして（OpenAI ワイヤーの場合）リクエストごとのタイムアウト kwarg に適用されます。そのため、設定した値はレガシーの `HERMES_API_TIMEOUT` 環境変数より優先されます。

非ストリーミングのスタールコール検出器として `providers.<id>.stale_timeout_seconds` を、加えてモデル固有のオーバーライドとして `providers.<id>.models.<model>.stale_timeout_seconds` を設定することもできます。これはレガシーの `HERMES_API_CALL_STALE_TIMEOUT` 環境変数より優先されます。

これらを未設定のままにすると、レガシーのデフォルト値（`HERMES_API_TIMEOUT=1800` 秒、`HERMES_API_CALL_STALE_TIMEOUT=300` 秒、ネイティブ Anthropic は 900 秒）が維持されます。現時点では AWS Bedrock には組み込まれていません（`bedrock_converse` と AnthropicBedrock SDK のいずれの経路も、独自のタイムアウト設定を持つ boto3 を使用します）。[`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example) のコメント付きの例を参照してください。

## ターミナルバックエンドの設定

Hermes は 7 種類のターミナルバックエンドをサポートしています。それぞれが、エージェントのシェルコマンドが実際にどこで実行されるかを決定します — ローカルマシン、Docker コンテナ、SSH 経由のリモートサーバー、Modal クラウドサンドボックス（直接接続、または Nous 管理のゲートウェイ経由）、Daytona ワークスペース、Vercel Sandbox、または Singularity/Apptainer コンテナです。

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | vercel_sandbox | singularity
  cwd: "."          # ゲートウェイ/cron の作業ディレクトリ（CLI は常に起動ディレクトリを使用）
  timeout: 180      # コマンドごとのタイムアウト（秒）
  env_passthrough: []  # サンドボックス実行（terminal + execute_code）へ転送する環境変数名
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Singularity バックエンド用のコンテナイメージ
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Modal バックエンド用のコンテナイメージ
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Daytona バックエンド用のコンテナイメージ
```

Modal、Daytona、Vercel Sandbox などのクラウドサンドボックスでは、`container_persistent: true` を指定すると、Hermes はサンドボックスの再作成をまたいでファイルシステムの状態を保持しようとします。ただし、同じ稼働中のサンドボックス、PID 空間、またはバックグラウンドプロセスが後で実行され続けていることを保証するものではありません。

### バックエンドの概要

| バックエンド | コマンドの実行場所 | 分離 | 適した用途 |
|---------|-------------------|-----------|----------|
| **local** | あなたのマシン上で直接 | なし | 開発、個人利用 |
| **docker** | 単一の永続的な Docker コンテナ（セッション、`/new`、サブエージェントで共有） | 完全（namespace、cap-drop） | 安全なサンドボックス化、CI/CD |
| **ssh** | SSH 経由のリモートサーバー | ネットワーク境界 | リモート開発、高性能ハードウェア |
| **modal** | Modal クラウドサンドボックス | 完全（クラウド VM） | 一時的なクラウド計算、評価 |
| **daytona** | Daytona ワークスペース | 完全（クラウドコンテナ） | マネージドなクラウド開発環境 |
| **vercel_sandbox** | Vercel Sandbox | 完全（クラウド microVM） | スナップショットによるファイルシステム永続化を備えたクラウド実行 |
| **singularity** | Singularity/Apptainer コンテナ | namespace（--containall） | HPC クラスタ、共有マシン |

### Local バックエンド

デフォルトです。コマンドは分離なしであなたのマシン上で直接実行されます。特別なセットアップは不要です。

```yaml
terminal:
  backend: local
```

:::warning
エージェントはあなたのユーザーアカウントと同じファイルシステムアクセス権を持ちます。使いたくないツールは `hermes tools` で無効化するか、サンドボックス化のために Docker へ切り替えてください。
:::

### Docker バックエンド {#docker-backend}

セキュリティハードニング（すべてのケーパビリティを削除、特権昇格なし、PID 制限）を施した Docker コンテナ内でコマンドを実行します。

**コマンドごとではなく、単一の永続的なコンテナです。** Hermes は初回利用時に長寿命のコンテナを 1 つだけ起動し、すべてのターミナル、ファイル、`execute_code` 呼び出しを `docker exec` でその同じコンテナにルーティングします — セッション、`/new`、`/reset`、`delegate_task` のサブエージェントをまたいで、Hermes プロセスが存続する間ずっとです。作業ディレクトリの変更、インストールしたパッケージ、`/workspace` 内のファイルは、ローカルシェルと同じように、あるツール呼び出しから次の呼び出しへ引き継がれます。コンテナはシャットダウン時に停止・削除されます。詳細は後述の **コンテナのライフサイクル** を参照してください。

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # 起動ディレクトリを /workspace にマウント
  docker_run_as_host_user: false   # 後述の「コンテナをホストユーザーとして実行する」を参照
  docker_forward_env:              # コンテナへ転送する環境変数
    - "GITHUB_TOKEN"
  docker_volumes:                  # ホストディレクトリのマウント
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro で読み取り専用
  # リソース制限
  container_cpu: 1                 # CPU コア数（0 = 無制限）
  container_memory: 5120           # MB（0 = 無制限）
  container_disk: 51200            # MB（XFS+pquota 上の overlay2 が必要）
  container_persistent: true       # セッションをまたいで /workspace と /root を永続化
```

**要件:** Docker Desktop または Docker Engine がインストールされ、稼働していること。Hermes は `$PATH` に加えて、一般的な macOS のインストール場所（`/usr/local/bin/docker`、`/opt/homebrew/bin/docker`、Docker Desktop のアプリバンドル）を探します。Podman は標準でサポートされています。両方がインストールされている場合に Podman を強制するには、`HERMES_DOCKER_BINARY=podman`（またはフルパス）を設定してください。

**コンテナのライフサイクル:** Hermes は、すべてのターミナルおよびファイルツール呼び出しに対して、単一の長寿命コンテナ（`docker run -d ... sleep 2h`）を再利用します — セッション、`/new`、`/reset`、`delegate_task` のサブエージェントをまたいで、Hermes プロセスが存続する間ずっとです。コマンドはログインシェルで `docker exec` を介して実行されるため、作業ディレクトリの変更、インストールしたパッケージ、`/workspace` 内のファイルはすべて、あるツール呼び出しから次の呼び出しへ永続化されます。コンテナは Hermes のシャットダウン時（またはアイドルスイープによって回収されたとき）に停止・削除されます。

`delegate_task(tasks=[...])` で生成された並列サブエージェントは、この 1 つのコンテナを共有します — 同時に行われる `cd`、環境変数の変更、同じパスへの書き込みは衝突します。サブエージェントが分離されたサンドボックスを必要とする場合は、`register_task_env_overrides()` を介してタスクごとのイメージオーバーライドを登録する必要があります。RL やベンチマーク環境（TerminalBench2、HermesSweEnv など）は、タスクごとの Docker イメージに対してこれを自動的に行います。

**セキュリティハードニング:**
- `--cap-drop ALL` に加え、`DAC_OVERRIDE`、`CHOWN`、`FOWNER` のみを戻す
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- `/tmp`（512MB）、`/var/tmp`（256MB）、`/run`（64MB）に対するサイズ制限付き tmpfs

**認証情報の転送:** `docker_forward_env` に列挙された環境変数は、まずシェル環境から、次に `~/.hermes/.env` から解決されます。スキルは `required_environment_variables` を宣言することもでき、それらは自動的にマージされます。

### SSH バックエンド

SSH 経由でリモートサーバー上でコマンドを実行します。接続の再利用のために ControlMaster を使用します（5 分間のアイドルキープアライブ）。永続シェルはデフォルトで有効です — 状態（cwd、環境変数）はコマンドをまたいで保持されます。

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # 長寿命の bash セッションを維持（デフォルト: true）
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
| `TERMINAL_SSH_PERSISTENT` | `true` | 永続シェルを有効化 |

**仕組み:** 初期化時に `BatchMode=yes` と `StrictHostKeyChecking=accept-new` で接続します。永続シェルは、リモートホスト上で単一の `bash -l` プロセスを生かし続け、一時ファイルを介して通信します。`stdin_data` や `sudo` を必要とするコマンドは、自動的にワンショットモードへフォールバックします。

### Modal バックエンド

[Modal](https://modal.com) クラウドサンドボックス内でコマンドを実行します。各タスクは、設定可能な CPU、メモリ、ディスクを備えた分離された VM を取得します。ファイルシステムはセッションをまたいでスナップショット/復元できます。

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU コア数
  container_memory: 5120           # MB（5GB）
  container_disk: 51200            # MB（50GB）
  container_persistent: true       # ファイルシステムをスナップショット/復元
```

**必須:** `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` 環境変数、または `~/.modal.toml` 設定ファイルのいずれか。

**永続化:** 有効にすると、クリーンアップ時にサンドボックスのファイルシステムがスナップショットされ、次のセッションで復元されます。スナップショットは `~/.hermes/modal_snapshots.json` で追跡されます。これはファイルシステムの状態を保持するもので、稼働中のプロセス、PID 空間、バックグラウンドジョブは保持しません。

**認証情報ファイル:** `~/.hermes/`（OAuth トークンなど）から自動的にマウントされ、各コマンドの前に同期されます。

### Daytona バックエンド

[Daytona](https://daytona.io) のマネージドワークスペース内でコマンドを実行します。永続化のための stop/resume をサポートします。

```yaml
terminal:
  backend: daytona
  container_cpu: 1                 # CPU コア数
  container_memory: 5120           # MB → GiB に変換
  container_disk: 10240            # MB → GiB に変換（最大 10 GiB）
  container_persistent: true       # 削除せず stop/resume する
```

**必須:** `DAYTONA_API_KEY` 環境変数。

**永続化:** 有効にすると、サンドボックスはクリーンアップ時に（削除されず）停止され、次のセッションで再開されます。サンドボックス名は `hermes-{task_id}` というパターンに従います。

**ディスク制限:** Daytona は最大 10 GiB を強制します。これを超えるリクエストは警告とともに上限に制限されます。

### Vercel Sandbox バックエンド

[Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) のクラウド microVM 内でコマンドを実行します。Hermes は通常のターミナルおよびファイルツールのインターフェイスを使用します。Vercel 固有のモデル向けツールはありません。

```yaml
terminal:
  backend: vercel_sandbox
  vercel_runtime: node24          # node24 | node22 | python3.13
  cwd: /vercel/sandbox            # デフォルトのワークスペースルート
  container_persistent: true      # ファイルシステムをスナップショット/復元
  container_disk: 51200           # 共有のデフォルトのみ。カスタムディスクはサポートされない
```

**必須のインストール:** オプションの SDK エクストラをインストールします。

```bash
pip install 'hermes-agent[vercel]'
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

**ランタイム:** `terminal.vercel_runtime` は `node24`、`node22`、`python3.13` をサポートします。未設定の場合、Hermes は `node24` をデフォルトとします。

**永続化:** `container_persistent: true` のとき、Hermes はクリーンアップ中にサンドボックスのファイルシステムをスナップショットし、同じタスクの後続のサンドボックスをそのスナップショットから復元します。スナップショットの内容には、サンドボックスにコピーされた Hermes 同期の認証情報、スキル、キャッシュファイルが含まれることがあります。これはファイルシステムの状態のみを保持するもので、稼働中のサンドボックスのアイデンティティ、PID 空間、シェルの状態、実行中のバックグラウンドプロセスは保持しません。

**バックグラウンドコマンド:** `terminal(background=true)` は Hermes の汎用的な非ローカルバックグラウンドプロセスフローを使用します。サンドボックスが稼働している間は、通常のプロセスツールを通じてプロセスの生成、ポーリング、待機、ログ表示、終了が行えます。Hermes は、クリーンアップや再起動の後に Vercel のネイティブなデタッチプロセス復旧を提供しません。

**ディスクサイズ設定:** 現時点で Vercel Sandbox は Hermes の `container_disk` リソースつまみをサポートしていません。`container_disk` は未設定のままにするか、共有のデフォルト `51200` のままにしてください。デフォルト以外の値は、暗黙のうちに無視されるのではなく、診断とバックエンド作成で失敗します。

### Singularity/Apptainer バックエンド

[Singularity/Apptainer](https://apptainer.org) コンテナ内でコマンドを実行します。Docker が利用できない HPC クラスタや共有マシン向けに設計されています。

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  container_cpu: 1                 # CPU コア数
  container_memory: 5120           # MB
  container_persistent: true       # 書き込み可能なオーバーレイがセッションをまたいで永続化
```

**要件:** `$PATH` に `apptainer` または `singularity` バイナリがあること。

**イメージの扱い:** Docker URL（`docker://...`）は自動的に SIF ファイルへ変換され、キャッシュされます。既存の `.sif` ファイルはそのまま使用されます。

**スクラッチディレクトリ:** 次の順序で解決されます: `TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent`（HPC の慣例） → `~/.hermes/sandboxes/singularity`。

**分離:** `--containall --no-home` を使用し、ホストのホームディレクトリをマウントせずに完全な namespace 分離を行います。

### ターミナルバックエンドのよくある問題

ターミナルコマンドが即座に失敗する、またはターミナルツールが無効と報告される場合:

- **Local** — 特別な要件はありません。始める際の最も安全なデフォルトです。
- **Docker** — `docker version` を実行して Docker が動作していることを確認してください。失敗する場合は、Docker を修正するか `hermes config set terminal.backend local` を実行してください。
- **SSH** — `TERMINAL_SSH_HOST` と `TERMINAL_SSH_USER` の両方を設定する必要があります。いずれかが欠けている場合、Hermes は明確なエラーを記録します。
- **Modal** — `MODAL_TOKEN_ID` 環境変数または `~/.modal.toml` が必要です。`hermes doctor` を実行して確認してください。
- **Daytona** — `DAYTONA_API_KEY` が必要です。Daytona SDK がサーバー URL の設定を処理します。
- **Singularity** — `$PATH` に `apptainer` または `singularity` が必要です。HPC クラスタでは一般的です。

迷ったときは、`terminal.backend` を `local` に戻し、まずそこでコマンドが実行されることを確認してください。

### ティアダウン時のリモート → ホスト間のファイル同期

**SSH**、**Modal**、**Daytona** の各バックエンド（エージェントの作業ツリーが、Hermes を実行しているホストとは別のマシン上にある場合すべて）では、Hermes はエージェントがリモートサンドボックス内で触れたファイルを追跡し、セッションのティアダウン / サンドボックスのクリーンアップ時に、**変更されたファイルをホストへ同期して戻します**。同期先は `~/.hermes/cache/remote-syncs/<session-id>/` です。

- 発火するタイミング: セッションのクローズ、`/new`、`/reset`、ゲートウェイのメッセージタイムアウト、子が リモートバックエンドを使用した場合の `delegate_task` サブエージェントの完了。
- エージェントが明示的に開いたファイルだけでなく、変更したツリー全体を対象とします。追加、編集、削除のすべてが捕捉されます。
- リモートサンドボックスは、あなたが探しに行く頃にはティアダウンされているかもしれません。ローカルの `~/.hermes/cache/remote-syncs/…` のコピーが、エージェントが変更した内容の正式な記録です。
- 大きなバイナリ出力（モデルのチェックポイント、生のデータセット）はサイズで制限されます — 同期は `file_sync_max_mb`（デフォルト `100`）を超えるファイルをスキップします。より大きな成果物が返ってくると見込む場合は、この値を引き上げてください。

```yaml
terminal:
  file_sync_max_mb: 100     # デフォルト — 1 ファイルあたり 100 MB までを同期
  file_sync_enabled: true   # デフォルト — 同期を完全にスキップするには false を設定
```

これは、セッション終了後に破棄される一時的なクラウドサンドボックスから結果を回収する方法です。すべての成果物について、エージェントに明示的に `scp` や `modal volume put` を指示する必要がありません。

### Docker のボリュームマウント

Docker バックエンドを使用する場合、`docker_volumes` でホストディレクトリをコンテナと共有できます。各エントリは標準的な Docker `-v` 構文を使用します: `host_path:container_path[:options]`。

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # 読み書き（デフォルト）
    - "/home/user/datasets:/data:ro"              # 読み取り専用
    - "/home/user/.hermes/cache/documents:/output" # ゲートウェイから見えるエクスポート
```

これは次の用途に役立ちます。
- エージェントに **ファイルを提供する**（データセット、設定、参照コード）
- エージェントから **ファイルを受け取る**（生成されたコード、レポート、エクスポート）
- あなたとエージェントの両方が同じファイルにアクセスする **共有ワークスペース**

メッセージングゲートウェイを使用し、エージェントに `MEDIA:/...` 経由で生成ファイルを送信させたい場合は、`/home/user/.hermes/cache/documents:/output` のような専用のホストから見えるエクスポートマウントを使うことをおすすめします。

- Docker 内では `/output/...` にファイルを書き込む
- `MEDIA:` には **ホストパス** を出力する。例:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`
- その正確なパスがホスト上のゲートウェイプロセスにも存在する場合を除き、`/workspace/...` や `/output/...` を出力**しない**こと

:::warning
YAML の重複キーは、先のものを暗黙のうちに上書きします。すでに `docker_volumes:` ブロックがある場合は、ファイル内の後ろに別の `docker_volumes:` キーを追加するのではなく、新しいマウントを同じリストにマージしてください。
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

デフォルトでは、Docker コンテナは `root`（UID 0）として実行されます。`/workspace` やその他のバインドマウント内で作成されたファイルはホスト上で root 所有となるため、セッション後にホストのエディタで編集するには、先に `sudo chown` する必要があります。`terminal.docker_run_as_host_user` フラグがこれを解決します。

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # デフォルト: false
```

有効にすると、Hermes は `docker run` コマンドに `--user $(id -u):$(id -g)` を追加するため、バインドマウントされたディレクトリ（`/workspace`、`/root`、`docker_volumes` 内のすべて）に書き込まれるファイルは root ではなくホストユーザーの所有となります。トレードオフとして、コンテナは `apt install` や `/root/.npm` のような root 所有のパスへの書き込みができなくなります — 両方が必要な場合は、`HOME` が非 root ユーザーの所有であるベースイメージを使う（またはイメージのビルド時に必要なツールを追加する）ようにしてください。

後方互換の動作のためには、これを `false`（デフォルト）のままにします。ワークフローがほとんど「マウントされたホストファイルを編集する」もので、`sudo chown -R` にうんざりしているときに有効にしてください。

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

### 永続シェル

デフォルトでは、各ターミナルコマンドは独自のサブプロセスで実行され、作業ディレクトリ、環境変数、シェル変数はコマンドごとにリセットされます。**永続シェル** が有効な場合、`execute()` 呼び出しをまたいで単一の長寿命 bash プロセスが生かされ、状態がコマンド間で保持されます。

これは **SSH バックエンド** で最も有用で、コマンドごとの接続オーバーヘッドも解消されます。永続シェルは **SSH ではデフォルトで有効** であり、local バックエンドでは無効です。

```yaml
terminal:
  persistent_shell: true   # デフォルト — SSH の永続シェルを有効化
```

無効にするには:

```bash
hermes config set terminal.persistent_shell false
```

**コマンドをまたいで永続化されるもの:**
- 作業ディレクトリ（`cd /tmp` が次のコマンドでも維持される）
- エクスポートした環境変数（`export FOO=bar`）
- シェル変数（`MY_VAR=hello`）

**優先順位:**

| レベル | 変数 | デフォルト |
|-------|----------|---------|
| Config | `terminal.persistent_shell` | `true` |
| SSH オーバーライド | `TERMINAL_SSH_PERSISTENT` | config に従う |
| Local オーバーライド | `TERMINAL_LOCAL_PERSISTENT` | `false` |

バックエンドごとの環境変数が最も高い優先度を持ちます。local バックエンドでも永続シェルが欲しい場合:

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
`stdin_data` や sudo を必要とするコマンドは、永続シェルの stdin が IPC プロトコルによってすでに占有されているため、自動的にワンショットモードへフォールバックします。
:::

各バックエンドの詳細は、[コード実行](features/code-execution.md)と [README のターミナルのセクション](features/tools.md)を参照してください。

## スキル設定

スキルは、SKILL.md のフロントマターを通じて独自の設定項目を宣言できます。これらはシークレットではない値（パス、設定、ドメイン固有の設定）で、`config.yaml` の `skills.config` 名前空間の下に保存されます。

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # 例 — 各スキルが独自のキーを定義する
```

**スキル設定の仕組み:**

- `hermes config migrate` は有効なすべてのスキルをスキャンし、未設定の項目を見つけて入力を促します
- `hermes config show` は、所属するスキルとともに「Skill Settings」の下にすべてのスキル設定を表示します
- スキルがロードされると、解決された設定値が自動的にスキルコンテキストへ注入されます

**値を手動で設定する:**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

自分のスキルで設定項目を宣言する方法の詳細は、[スキルの作成 — 設定項目](/docs/developer-guide/creating-skills#config-settings-configyaml)を参照してください。

### エージェントが作成したスキル書き込みに対するガード

エージェントが `skill_manage` を使ってスキルを作成、編集、パッチ、削除する際、Hermes は新規/更新されたコンテンツについて、危険なキーワードパターン（認証情報の収集、明白なプロンプトインジェクション、漏えい指示）をオプションでスキャンできます。スキャナーは **デフォルトで無効** です — `~/.ssh/` に正当に触れたり `$OPENAI_API_KEY` に言及したりする実際のエージェントのワークフローが、このヒューリスティックに頻繁に引っかかっていたためです。エージェントのスキル書き込みが反映される前に確認を促してほしい場合は、これを有効に戻してください。

```yaml
skills:
  guard_agent_created: true   # デフォルト: false
```

有効にすると、フラグの立った `skill_manage` の書き込みは、スキャナーの根拠とともに承認プロンプトとして表示されます。承認された書き込みは反映され、拒否された書き込みはエージェントに説明的なエラーを返します。

## メモリ設定

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # 約 800 トークン
  user_char_limit: 1375     # 約 500 トークン
```

## ファイル読み取りの安全性

単一の `read_file` 呼び出しが返せるコンテンツの量を制御します。上限を超える読み取りは、より小さな範囲のために `offset` と `limit` を使うよう促すエラーとともに拒否されます。これにより、ミニファイされた JS バンドルや大きなデータファイルを一度に読み取ってコンテキストウィンドウを溢れさせることを防ぎます。

```yaml
file_read_max_chars: 100000  # デフォルト — 約 25〜35K トークン
```

大きなコンテキストウィンドウを持つモデルを使用していて、大きなファイルを頻繁に読む場合は引き上げてください。小さなコンテキストのモデルでは、読み取りを効率的に保つために下げてください。

```yaml
# 大きなコンテキストのモデル（200K 以上）
file_read_max_chars: 200000

# 小さなローカルモデル（16K コンテキスト）
file_read_max_chars: 30000
```

エージェントはファイル読み取りも自動的に重複排除します — 同じファイル領域が二度読まれ、ファイルが変更されていない場合、コンテンツを再送する代わりに軽量なスタブが返されます。これはコンテキスト圧縮時にリセットされるため、コンテンツが要約されて失われた後でもエージェントはファイルを再読み取りできます。

## ツール出力の切り詰め上限

ツールが返せる生の出力量を Hermes が切り詰める前に制御する、3 つの関連する上限があります。

```yaml
tool_output:
  max_bytes: 50000        # ターミナル出力の上限（文字数）
  max_lines: 2000         # read_file のページネーション上限
  max_line_length: 2000   # read_file の行番号付きビューにおける 1 行あたりの上限
```

- **`max_bytes`** — `terminal` コマンドが標準出力/標準エラー出力の合計でこの文字数を超える出力を生成した場合、Hermes は最初の 40% と最後の 60% を保持し、その間に `[OUTPUT TRUNCATED]` の通知を挿入します。デフォルト `50000`（一般的なトークナイザで約 12〜15K トークン）。
- **`max_lines`** — 単一の `read_file` 呼び出しの `limit` パラメータの上限。これを超えるリクエストは制限され、単一の読み取りがコンテキストウィンドウを溢れさせないようにします。デフォルト `2000`。
- **`max_line_length`** — `read_file` が行番号付きビューを出力する際に適用される 1 行あたりの上限。これより長い行は、この文字数に切り詰められ、その後に `... [truncated]` が付きます。デフォルト `2000`。

呼び出しごとにより多くの生の出力を許容できる大きなコンテキストウィンドウのモデルでは、上限を引き上げてください。小さなコンテキストのモデルでは、ツール結果をコンパクトに保つために下げてください。

```yaml
# 大きなコンテキストのモデル（200K 以上）
tool_output:
  max_bytes: 150000
  max_lines: 5000

# 小さなローカルモデル（16K コンテキスト）
tool_output:
  max_bytes: 20000
  max_lines: 500
```

## ツールセットのグローバル無効化

特定のツールセットを CLI とすべてのゲートウェイプラットフォームにわたって 1 か所で抑制するには、その名前を `agent.disabled_toolsets` の下に列挙します。

```yaml
agent:
  disabled_toolsets:
    - memory       # メモリツールと MEMORY_GUIDANCE の注入を隠す
    - web          # web_search / web_extract をどこにも出さない
```

これは（`hermes tools` によって書き込まれる）プラットフォームごとのツール設定（`platform_toolsets`）の **後** に適用されるため、ここに列挙されたツールセットは常に削除されます — プラットフォームの保存された設定にまだ列挙されていても同様です。`hermes tools` の UI で 15 以上のプラットフォーム行を編集するのではなく、「どこでも X をオフにする」ための単一のスイッチが欲しい場合に使ってください。

リストを空のままにする、またはキーを省略すると、何も起こりません。

## Git ワークツリーの分離

同じリポジトリで複数のエージェントを並列実行するために、分離された git ワークツリーを有効にします。

```yaml
worktree: true    # 常にワークツリーを作成（hermes -w と同じ）
# worktree: false # デフォルト — -w フラグが渡されたときのみ
```

有効にすると、各 CLI セッションは独自のブランチを持つ新しいワークツリーを `.worktrees/` の下に作成します。エージェントは互いに干渉することなく、ファイルの編集、コミット、プッシュ、PR の作成ができます。クリーンなワークツリーは終了時に削除され、変更のあるものは手動での復旧のために保持されます。

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
  enabled: true                                     # 圧縮のオン/オフを切り替え
  threshold: 0.50                                   # コンテキスト上限のこの % で圧縮
  target_ratio: 0.20                                # 直近の末尾として保持する threshold の割合
  protect_last_n: 20                                # 未圧縮で残す直近メッセージの最小数
  hygiene_hard_message_limit: 400                   # ゲートウェイの安全弁 — 後述

# 要約モデル/プロバイダーは auxiliary の下で設定する:
auxiliary:
  compression:
    model: ""                                       # 空 = メインのチャットモデルを使用。例えば "google/gemini-3-flash-preview" でより安価/高速な圧縮にオーバーライド。
    provider: "auto"                                # プロバイダー: "auto"、"openrouter"、"nous"、"codex"、"main" など
    base_url: null                                  # カスタムの OpenAI 互換エンドポイント（プロバイダーをオーバーライド）
```

:::info レガシー設定の移行
`compression.summary_model`、`compression.summary_provider`、`compression.summary_base_url` を持つ古い設定は、初回ロード時に自動的に `auxiliary.compression.*` へ移行されます（設定バージョン 17）。手動での操作は不要です。
:::

`hygiene_hard_message_limit` はゲートウェイ専用の **圧縮前の安全弁** です。数千のメッセージを持つ暴走したセッションは、通常のコンテキストの % に対する閾値が発火する前にモデルのコンテキスト上限に達することがあります。メッセージ数がこの上限を超えると、Hermes はトークン使用量にかかわらず圧縮を強制します。デフォルト `400` — 非常に長いセッションが普通のプラットフォームでは引き上げ、より積極的な圧縮を強制するには下げてください。稼働中のゲートウェイでこの値を編集すると、次のメッセージで反映されます（後述）。

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
```
カスタムの OpenAI 互換エンドポイントを指します。認証には `OPENAI_API_KEY` を使用します。

### 3 つのつまみの相互作用

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | 結果 |
|---------------------|---------------------|--------|
| `auto`（デフォルト） | 未設定 | 利用可能な最良のプロバイダーを自動検出 |
| `nous` / `openrouter` など | 未設定 | そのプロバイダーを強制し、その認証を使用 |
| いずれか | 設定済み | カスタムエンドポイントを直接使用（プロバイダーは無視） |

:::warning 要約モデルのコンテキスト長の要件
要約モデルは、メインのエージェントモデルと **少なくとも同じ大きさ** のコンテキストウィンドウを持つ **必要があります**。圧縮器は会話の中間部分全体を要約モデルに送信します — そのモデルのコンテキストウィンドウがメインのモデルより小さいと、要約呼び出しはコンテキスト長エラーで失敗します。これが起きると、中間のターンは **要約なしで破棄され**、会話のコンテキストが暗黙のうちに失われます。モデルをオーバーライドする場合は、そのコンテキスト長がメインのモデル以上であることを確認してください。
:::

## コンテキストエンジン

コンテキストエンジンは、モデルのトークン上限に近づいたときに会話をどのように管理するかを制御します。組み込みの `compressor` エンジンは、損失のある要約を使用します（[コンテキスト圧縮](/docs/developer-guide/context-compression-and-caching)を参照）。プラグインエンジンは、これを代替戦略で置き換えられます。

```yaml
context:
  engine: "compressor"    # デフォルト — 組み込みの損失のある要約
```

プラグインエンジン（例: ロスレスなコンテキスト管理のための LCM）を使用するには:

```yaml
context:
  engine: "lcm"          # プラグインの名前と一致させる必要がある
```

プラグインエンジンは **決して自動起動されません** — `context.engine` をプラグイン名に明示的に設定する必要があります。利用可能なエンジンは `hermes plugins` → Provider Plugins → Context Engine から閲覧・選択できます。

メモリプラグインの類似の単一選択システムについては、[メモリプロバイダー](/docs/user-guide/features/memory-providers)を参照してください。

## イテレーション予算のプレッシャー

エージェントが多くのツール呼び出しを伴う複雑なタスクに取り組んでいるとき、残りが少なくなっていることに気づかないまま、イテレーション予算（デフォルト: 90 ターン）を使い切ってしまうことがあります。予算プレッシャーは、上限に近づくとモデルに自動的に警告します。

| 閾値 | レベル | モデルに見えるもの |
|-----------|-------|---------------------|
| **70%** | 注意 | `[BUDGET: 63/90. 27 iterations left. Start consolidating.]` |
| **90%** | 警告 | `[BUDGET WARNING: 81/90. Only 9 left. Respond NOW.]` |

警告は、別個のメッセージとしてではなく、最後のツール結果の JSON（`_budget_warning` フィールドとして）に注入されます — これによりプロンプトキャッシュが保たれ、会話構造を乱しません。

```yaml
agent:
  max_turns: 90                # 会話ターンあたりの最大イテレーション数（デフォルト: 90）
  api_max_retries: 3           # フォールバックが作動する前のプロバイダーごとのリトライ回数（デフォルト: 3）
```

予算プレッシャーはデフォルトで有効です。エージェントはツール結果の一部として自然に警告を見るため、イテレーションを使い切る前に作業をまとめてレスポンスを返すよう促されます。

イテレーション予算を完全に使い切ると、CLI はユーザーに通知を表示します: `⚠ Iteration budget reached (90/90) — response may be incomplete`。アクティブな作業中に予算が尽きた場合、エージェントは停止する前に、達成した内容の要約を生成します。

`agent.api_max_retries` は、フォールバックプロバイダーへの切り替えが作動する **前** に、一時的なエラー（レート制限、接続切断、5xx）でプロバイダーの API 呼び出しを Hermes が何回リトライするかを制御します。デフォルトは `3` — 合計 4 回の試行です。[フォールバックプロバイダー](/docs/user-guide/features/fallback-providers)を設定していて、より早くフェイルオーバーしたい場合は、これを `0` に下げると、プライマリでの最初の一時的なエラーが、不安定なエンドポイントに対してリトライを繰り返すのではなく、すぐにフォールバックへ引き渡されます。

### API タイムアウト

Hermes には、ストリーミング用に別々のタイムアウト層があり、加えて非ストリーミング呼び出し用のスタール検出器があります。スタール検出器は、暗黙のデフォルトのままにした場合に限り、ローカルプロバイダー向けに自動調整されます。

| タイムアウト | デフォルト | ローカルプロバイダー | Config / 環境変数 |
|---------|---------|----------------|--------------|
| ソケット読み取りタイムアウト | 120 秒 | 1800 秒に自動引き上げ | `HERMES_STREAM_READ_TIMEOUT` |
| スタールストリーム検出 | 180 秒 | 自動無効化 | `HERMES_STREAM_STALE_TIMEOUT` |
| スタール非ストリーム検出 | 300 秒 | 暗黙のままなら自動無効化 | `providers.<id>.stale_timeout_seconds` または `HERMES_API_CALL_STALE_TIMEOUT` |
| API 呼び出し（非ストリーミング） | 1800 秒 | 変更なし | `providers.<id>.request_timeout_seconds` / `timeout_seconds` または `HERMES_API_TIMEOUT` |

**ソケット読み取りタイムアウト** は、httpx がプロバイダーからの次のデータチャンクをどれだけ待つかを制御します。ローカル LLM は、最初のトークンを生成する前に、大きなコンテキストのプリフィルに数分かかることがあるため、Hermes はローカルエンドポイントを検出すると、これを 30 分に引き上げます。`HERMES_STREAM_READ_TIMEOUT` を明示的に設定すると、エンドポイント検出にかかわらずその値が常に使用されます。

**スタールストリーム検出** は、SSE のキープアライブ ping を受信するが実際のコンテンツがない接続を切断します。これは、プリフィル中にキープアライブ ping を送信しないローカルプロバイダーでは完全に無効化されます。

**スタール非ストリーム検出** は、長すぎる時間レスポンスを生成しない非ストリーミング呼び出しを切断します。デフォルトでは、Hermes は長いプリフィル中の誤検出を避けるために、ローカルエンドポイントでこれを無効化します。`providers.<id>.stale_timeout_seconds`、`providers.<id>.models.<model>.stale_timeout_seconds`、または `HERMES_API_CALL_STALE_TIMEOUT` を明示的に設定すると、その明示的な値はローカルエンドポイントでも尊重されます。

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

## 認証情報プールの戦略

同じプロバイダーに対して複数の API キーや OAuth トークンを持っている場合、ローテーション戦略を設定します。

```yaml
credential_pool_strategies:
  openrouter: round_robin    # キーを均等に巡回
  anthropic: least_used      # 常に最も使われていないキーを選択
```

オプション: `fill_first`（デフォルト）、`round_robin`、`least_used`、`random`。完全なドキュメントは[認証情報プール](/docs/user-guide/features/credential-pools)を参照してください。

## 補助モデル

Hermes は、画像分析、Web ページの要約、ブラウザのスクリーンショット分析、セッションタイトルの生成、コンテキスト圧縮などのサイドタスクに「補助」モデルを使用します。デフォルト（`auxiliary.*.provider: "auto"`）では、Hermes はすべての補助タスクを **メインのチャットモデル** — `hermes model` で選んだのと同じプロバイダー/モデル — にルーティングします。始めるにあたって何も設定する必要はありませんが、高価な推論モデル（Opus、MiniMax M2.7 など）では補助タスクが相応のコストを追加することに注意してください。メインのモデルにかかわらず安価で高速なサイドタスクが欲しい場合は、`auxiliary.<task>.provider` と `auxiliary.<task>.model` を明示的に設定してください（例: ビジョンと Web 抽出に OpenRouter 上の Gemini Flash）。

:::note なぜ "auto" がメインモデルを使うのか
以前のビルドは、アグリゲーターのユーザー（OpenRouter、Nous Portal）を、安価なプロバイダー側のデフォルトに振り分けていました。これは意外なものでした — アグリゲーターのサブスクリプションに料金を払ったユーザーが、補助トラフィックを処理する別のモデルを目にすることになっていたためです。`auto` は今ではすべての人にとってメインのモデルを使用し、`config.yaml` のタスクごとのオーバーライドは依然として優先されます（後述の[補助設定の完全なリファレンス](#full-auxiliary-config-reference)を参照）。
:::

### 補助モデルを対話的に設定する

YAML を手で編集する代わりに、`hermes model` を実行してメニューから **「Configure auxiliary models」** を選んでください。タスクごとの対話的なピッカーが表示されます。

```
$ hermes model
→ Configure auxiliary models

[ ] vision               currently: auto / main model
[ ] web_extract          currently: auto / main model
[ ] session_search       currently: openrouter / google/gemini-2.5-flash
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
[ ] triage_specifier     currently: auto / main model
```

タスクを選び、プロバイダーを選び（OAuth フローはブラウザを開きます。API キーのプロバイダーは入力を促します）、モデルを選びます。変更は `config.yaml` の `auxiliary.<task>.*` に永続化されます。メインモデルのピッカーと同じ仕組みで — 覚えるべき追加の構文はありません。

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

`base_url` が設定されている場合、Hermes はプロバイダーを無視し、そのエンドポイントを直接呼び出します（認証には `api_key` または `OPENAI_API_KEY` を使用）。`provider` のみが設定されている場合、Hermes はそのプロバイダーの組み込みの認証とベース URL を使用します。

補助タスクで利用可能なプロバイダー: `auto`、`main`、加えて[プロバイダーレジストリ](/docs/reference/environment-variables)の任意のプロバイダー — `openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`gemini`、`google-gemini-cli`、`qwen-oauth`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`ollama-cloud`、`alibaba`、`bedrock`、`huggingface`、`arcee`、`xiaomi`、`kilocode`、`opencode-zen`、`opencode-go`、`ai-gateway`、`azure-foundry` — または `custom_providers` リストにある任意の名前付きカスタムプロバイダー（例: `provider: "beans"`）。

:::tip MiniMax OAuth
`minimax-oauth` はブラウザ OAuth でログインします（API キー不要）。`hermes model` を実行し、**MiniMax (OAuth)** を選んで認証してください。補助タスクは自動的に `MiniMax-M2.7-highspeed` を使用します。[MiniMax OAuth ガイド](../guides/minimax-oauth.md)を参照してください。
:::

:::warning `"main"` は補助タスク専用
`"main"` プロバイダーオプションは「メインのエージェントが使うプロバイダーを何でも使う」という意味です — これは `auxiliary:`、`compression:`、`fallback_model:` の設定内でのみ有効です。トップレベルの `model.provider` 設定の値としては **有効ではありません**。カスタムの OpenAI 互換エンドポイントを使う場合は、`model:` セクションで `provider: custom` を設定してください。すべてのメインモデルのプロバイダーオプションについては[AI プロバイダー](/docs/integrations/providers)を参照してください。
:::

### 補助設定の完全なリファレンス {#full-auxiliary-config-reference}

```yaml
auxiliary:
  # 画像分析（vision_analyze ツール + ブラウザのスクリーンショット）
  vision:
    provider: "auto"           # "auto"、"openrouter"、"nous"、"codex"、"main" など
    model: ""                  # 例: "openai/gpt-4o"、"google/gemini-2.5-flash"
    base_url: ""               # カスタムの OpenAI 互換エンドポイント（プロバイダーをオーバーライド）
    api_key: ""                # base_url 用の API キー（OPENAI_API_KEY にフォールバック）
    timeout: 120               # 秒 — LLM API 呼び出しのタイムアウト。ビジョンのペイロードには余裕のあるタイムアウトが必要
    download_timeout: 30       # 秒 — 画像の HTTP ダウンロード。低速な接続では増やす

  # Web ページの要約 + ブラウザのページテキスト抽出
  web_extract:
    provider: "auto"
    model: ""                  # 例: "google/gemini-2.5-flash"
    base_url: ""
    api_key: ""
    timeout: 360               # 秒（6 分） — 試行ごとの LLM 要約

  # 危険なコマンドの承認分類器
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30                # 秒

  # コンテキスト圧縮のタイムアウト（compression.* 設定とは別）
  compression:
    timeout: 120               # 秒 — 圧縮は長い会話を要約するため、より多くの時間が必要

  # セッション検索 — 過去のセッションのマッチを要約
  session_search:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    max_concurrency: 3       # リクエストバーストによる 429 を減らすため、並列要約を制限
    extra_body: {}           # プロバイダー固有の OpenAI 互換リクエストフィールド

  # スキルハブ — スキルのマッチングと検索
  skills_hub:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # MCP ツールのディスパッチ
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # カンバンのトリアージ指定器 — `hermes kanban specify <id>`（または
  # ダッシュボードの Triage 列カードの ✨ Specify ボタン）がこの
  # スロットを使って一行を具体的な仕様に展開し、タスクを
  # `todo` に昇格させます。安価で高速なモデルがここではよく機能します。
  # 仕様の展開は短く、推論の深さを必要としません。
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
各補助タスクには設定可能な `timeout`（秒）があります。デフォルト: vision 120 秒、web_extract 360 秒、approval 30 秒、compression 120 秒。補助タスクに低速なローカルモデルを使う場合は、これらを増やしてください。vision には HTTP 画像ダウンロード用の別個の `download_timeout`（デフォルト 30 秒）もあります — 低速な接続やセルフホストの画像サーバーではこれを増やしてください。
:::

:::info
コンテキスト圧縮は、閾値用に独自の `compression:` ブロックを、モデル/プロバイダー設定用に `auxiliary.compression:` ブロックを持ちます — 上記の[コンテキスト圧縮](#context-compression)を参照してください。フォールバックモデルは `fallback_model:` ブロックを使用します — [フォールバックモデル](/docs/integrations/providers#fallback-model)を参照してください。これら 3 つはすべて、同じ provider/model/base_url パターンに従います。
:::

### セッション検索のチューニング

`auxiliary.session_search` に推論の重いモデルを使う場合、Hermes には 2 つの組み込みコントロールがあります。

- `auxiliary.session_search.max_concurrency`: Hermes が一度に要約するマッチしたセッション数を制限します
- `auxiliary.session_search.extra_body`: 要約呼び出しでプロバイダー固有の OpenAI 互換リクエストフィールドを転送します

例:

```yaml
auxiliary:
  session_search:
    provider: "main"
    model: "glm-4.5-air"
    timeout: 60
    max_concurrency: 2
    extra_body:
      enable_thinking: false
```

プロバイダーがリクエストバーストをレート制限していて、`session_search` に並列性を多少犠牲にして安定性を取らせたい場合に `max_concurrency` を使ってください。

`extra_body` は、そのタスクで Hermes に渡してほしい OpenAI 互換のリクエストボディフィールドをプロバイダーが文書化している場合にのみ使ってください。Hermes はオブジェクトをそのまま転送します。

:::warning
`extra_body` は、送信するフィールドをプロバイダーが実際にサポートしている場合にのみ有効です。プロバイダーがネイティブな OpenAI 互換の推論オフフラグを公開していない場合、Hermes がそれを代わりに合成することはできません。
:::

### 補助タスク向けの OpenRouter ルーティングと Pareto Code

補助タスクが OpenRouter に解決される場合（明示的に、またはメインのエージェントが OpenRouter のときに `provider: "main"` を介して）、メインのエージェントの `provider_routing` と `openrouter.min_coding_score` の設定は **伝播しません** — 設計上、各補助タスクは独立しています。特定の補助タスクで OpenRouter のプロバイダー設定を行う、または [Pareto Code ルーター](/docs/integrations/providers#openrouter-pareto-code-router)を使うには、`extra_body` を介してタスクごとに設定します。

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code         # このタスクで Pareto Code ルーターを使用
    extra_body:
      provider:                            # OpenRouter のプロバイダールーティング設定
        order: [anthropic, google]         # これらのプロバイダーを順に試す
        sort: throughput                   # または "price" | "latency"
        # only: [anthropic]                # 特定のプロバイダーに制限
        # ignore: [deepinfra]              # 特定のプロバイダーを除外
      plugins:                             # OpenRouter Pareto Code ルーターのつまみ
        - id: pareto-router
          min_coding_score: 0.5            # 0.0〜1.0。高いほどコーディングが強い
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

これらのオプションは **補助タスクの設定**（`auxiliary:`、`compression:`、`fallback_model:`）に適用され、メインの `model.provider` 設定には適用されません。

| プロバイダー | 説明 | 要件 |
|----------|-------------|-------------|
| `"auto"` | 利用可能な最良（デフォルト）。ビジョンは OpenRouter → Nous → Codex を試す。 | — |
| `"openrouter"` | OpenRouter を強制 — 任意のモデル（Gemini、GPT-4o、Claude など）にルーティング | `OPENROUTER_API_KEY` |
| `"nous"` | Nous Portal を強制 | `hermes auth` |
| `"codex"` | Codex OAuth（ChatGPT アカウント）を強制。ビジョンをサポート（gpt-5.3-codex）。 | `hermes model` → Codex |
| `"minimax-oauth"` | MiniMax OAuth（ブラウザログイン、API キー不要）を強制。補助タスクに MiniMax-M2.7-highspeed を使用。 | `hermes model` → MiniMax (OAuth) |
| `"main"` | アクティブなカスタム/メインのエンドポイントを使用。これは `OPENAI_BASE_URL` + `OPENAI_API_KEY` から、または `hermes model` / `config.yaml` で保存されたカスタムエンドポイントから来ます。OpenAI、ローカルモデル、任意の OpenAI 互換 API で動作します。**補助タスク専用 — `model.provider` には無効。** | カスタムエンドポイントの認証情報 + ベース URL |

サイドタスクにデフォルトのルーターをバイパスさせたい場合、メインのプロバイダーカタログの直接 API キープロバイダーもここで動作します。`GMI_API_KEY` を設定すれば `gmi` も有効です。

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

GMI の補助ルーティングでは、GMI の `/v1/models` エンドポイントが返す正確なモデル ID を使ってください。

### 一般的なセットアップ

**直接のカスタムエンドポイントを使う**（ローカル/セルフホストの API では `provider: "main"` より明確）:
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` は `provider` より優先されるため、これは補助タスクを特定のエンドポイントにルーティングする最も明示的な方法です。直接のエンドポイントオーバーライドでは、Hermes は設定された `api_key` を使用するか、`OPENAI_API_KEY` にフォールバックします。そのカスタムエンドポイントに `OPENROUTER_API_KEY` を再利用することはありません。

**ビジョンに OpenAI API キーを使う:**
```yaml
# ~/.hermes/.env 内:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # またはより安価な "gpt-4o-mini"
```

**ビジョンに OpenRouter を使う**（任意のモデルにルーティング）:
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # または "google/gemini-2.5-flash" など
```

**Codex OAuth を使う**（ChatGPT Pro/Plus アカウント — API キー不要）:
```yaml
auxiliary:
  vision:
    provider: "codex"     # ChatGPT OAuth トークンを使用
    # モデルはデフォルトで gpt-5.3-codex（ビジョンをサポート）
```

**MiniMax OAuth を使う**（ブラウザログイン、API キー不要）:
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
`hermes model` を実行し、**MiniMax (OAuth)** を選ぶとログインしてこれが自動設定されます。中国リージョンの場合、ベース URL は `https://api.minimaxi.com/anthropic` になります。完全な手順は [MiniMax OAuth ガイド](../guides/minimax-oauth.md)を参照してください。

**ローカル/セルフホストのモデルを使う:**
```yaml
auxiliary:
  vision:
    provider: "main"      # アクティブなカスタムエンドポイントを使用
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
| Web 抽出のプロバイダー | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| Web 抽出のモデル | `AUXILIARY_WEB_EXTRACT_MODEL` |
| Web 抽出のエンドポイント | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| Web 抽出の API キー | `AUXILIARY_WEB_EXTRACT_API_KEY` |

圧縮とフォールバックモデルの設定は config.yaml のみです。

:::tip
現在の補助モデル設定を見るには `hermes config` を実行してください。オーバーライドは、デフォルトと異なる場合にのみ表示されます。
:::

## 推論の労力

レスポンスの前にモデルがどれだけ「考える」かを制御します。

```yaml
agent:
  reasoning_effort: ""   # 空 = medium（デフォルト）。オプション: none, minimal, low, medium, high, xhigh（最大）
```

未設定（デフォルト）の場合、推論の労力はデフォルトで「medium」になります — ほとんどのタスクでうまく機能するバランスの取れたレベルです。値を設定するとそれがオーバーライドされます — 推論の労力が高いほど、より多くのトークンとレイテンシと引き換えに、複雑なタスクでより良い結果が得られます。

`/reasoning` コマンドで実行時に推論の労力を変更することもできます。

```
/reasoning           # 現在の労力レベルと表示状態を表示
/reasoning high      # 推論の労力を high に設定
/reasoning none      # 推論を無効化
/reasoning show      # 各レスポンスの上にモデルの思考を表示
/reasoning hide      # モデルの思考を非表示
```

## ツール使用の強制

一部のモデルは、ツール呼び出しを行う代わりに、意図したアクションをテキストとして説明することがあります（実際にターミナルを呼ぶ代わりに「テストを実行します……」）。ツール使用の強制は、モデルを実際にツールを呼ぶ方向へ導くシステムプロンプトのガイダンスを注入します。

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| 値 | 動作 |
|-------|----------|
| `"auto"`（デフォルト） | 次にマッチするモデルで有効: `gpt`、`codex`、`gemini`、`gemma`、`grok`。それ以外（Claude、DeepSeek、Qwen など）では無効。 |
| `true` | モデルにかかわらず常に有効。現在のモデルがアクションを実行する代わりに説明していると気づいた場合に有用。 |
| `false` | モデルにかかわらず常に無効。 |
| `["gpt", "codex", "qwen", "llama"]` | モデル名が列挙された部分文字列のいずれかを含む場合にのみ有効（大文字小文字を区別しない）。 |

### 注入される内容

有効にすると、3 つの層のガイダンスがシステムプロンプトに追加されることがあります。

1. **一般的なツール使用の強制**（すべてのマッチしたモデル） — 意図を説明する代わりにすぐにツール呼び出しを行い、タスクが完了するまで作業を続け、将来のアクションの約束でターンを終えないようモデルに指示します。

2. **OpenAI の実行規律**（GPT および Codex モデルのみ） — GPT 固有の失敗モードに対処する追加のガイダンス: 部分的な結果で作業を放棄する、前提となる調べものをスキップする、ツールを使う代わりに幻覚する、検証なしに「完了」と宣言する。

3. **Google の運用ガイダンス**（Gemini および Gemma モデルのみ） — 簡潔さ、絶対パス、並列ツール呼び出し、編集前検証のパターン。

これらはユーザーに対して透過的で、システムプロンプトにのみ影響します。すでにツールを確実に使うモデル（Claude など）にはこのガイダンスは不要であり、それが `"auto"` がそれらを除外している理由です。

### いつ有効にするか

デフォルトの auto リストにないモデルを使用していて、実行する代わりに *やろうとしている* ことを頻繁に説明すると気づいた場合は、`tool_use_enforcement: true` を設定するか、モデルの部分文字列をリストに追加してください。

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## TTS 設定

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts"
  speed: 1.0                    # グローバルな速度倍率（全プロバイダーのフォールバック）
  edge:
    voice: "en-US-AriaNeural"   # 322 種類の音声、74 言語
    speed: 1.0                  # 速度倍率（レート百分率に変換、例: 1.5 → +50%）
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    speed: 1.0                  # 速度倍率（API により 0.25〜4.0 に制限）
    base_url: "https://api.openai.com/v1"  # OpenAI 互換 TTS エンドポイント用のオーバーライド
  minimax:
    speed: 1.0                  # 発話速度の倍率
    # base_url: ""              # オプション: OpenAI 互換 TTS エンドポイント用のオーバーライド
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral（デフォルト）
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # または gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30 種類のプリビルト音声: Zephyr, Puck, Kore, Enceladus など
  xai:
    voice_id: "eve"             # xAI TTS の音声
    language: "en"              # ISO 639-1
    sample_rate: 24000
    bit_rate: 128000            # MP3 ビットレート
    # base_url: "https://api.x.ai/v1"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

これは `text_to_speech` ツールと、音声モード（CLI またはメッセージングゲートウェイの `/voice tts`）での音声返信の両方を制御します。

**速度のフォールバック階層:** プロバイダー固有の速度（例: `tts.edge.speed`） → グローバルな `tts.speed` → `1.0` デフォルト。すべてのプロバイダーに一律の速度を適用するにはグローバルな `tts.speed` を設定し、きめ細かい制御にはプロバイダーごとにオーバーライドしてください。

## 表示設定

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # メッセージングゲートウェイで /verbose スラッシュコマンドを有効化
  platforms: {}           # プラットフォームごとの表示オーバーライド（後述）
  tool_progress_overrides: {}  # 非推奨 — 代わりに display.platforms を使用
  interim_assistant_messages: true  # ゲートウェイ: 自然なターン途中のアシスタント更新を別メッセージとして送信
  skin: default           # 組み込みまたはカスタムの CLI スキン（user-guide/features/skins を参照）
  personality: "kawaii"  # 一部のサマリーで今も表示されるレガシーの装飾フィールド
  compact: false          # コンパクト出力モード（余白を減らす）
  resume_display: full    # full（再開時に過去のメッセージを表示） | minimal（一行のみ）
  bell_on_complete: false # エージェントの完了時にターミナルベルを鳴らす（長いタスクに最適）
  show_reasoning: false   # 各レスポンスの上にモデルの推論/思考を表示（/reasoning show|hide で切り替え）
  streaming: false        # トークンを到着次第ターミナルにストリーミング（リアルタイム出力）
  show_cost: false        # CLI のステータスバーに推定の $ コストを表示
  tool_preview_length: 0  # ツール呼び出しプレビューの最大文字数（0 = 制限なし、フルパス/コマンドを表示）
  runtime_footer:         # ゲートウェイ: 最終返信にランタイムコンテキストのフッターを追加
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  language: en            # 静的メッセージ（承認プロンプト、一部のゲートウェイ返信）の UI 言語。en | zh | ja | de | es | fr | tr | uk
```

### 静的メッセージの UI 言語

`display.language` 設定は、ユーザー向けの静的メッセージの小さなセット — CLI の承認プロンプト、いくつかのゲートウェイのスラッシュコマンド返信（例: 再起動のドレイン通知、「approval expired」、「goal cleared」） — を翻訳します。エージェントのレスポンス、ログ行、ツール出力、エラーのトレースバック、スラッシュコマンドの説明は翻訳 **しません** — それらは英語のままです。エージェント自身に別の言語で返信させたい場合は、プロンプトやシステムメッセージでそう伝えるだけです。

サポートされる値: `en`（デフォルト）、`zh`（簡体字中国語）、`ja`（日本語）、`de`（ドイツ語）、`es`（スペイン語）、`fr`（フランス語）、`tr`（トルコ語）、`uk`（ウクライナ語）。不明な値は英語にフォールバックします。

`HERMES_LANGUAGE` 環境変数でセッションごとに設定することもでき、これは config の値をオーバーライドします。

```yaml
display:
  language: zh   # CLI の承認プロンプトが中国語で表示される
```

| モード | 見えるもの |
|------|-------------|
| `off` | 無音 — 最終レスポンスのみ |
| `new` | ツールが変わったときのみツールインジケーターを表示 |
| `all` | 短いプレビュー付きですべてのツール呼び出しを表示（デフォルト） |
| `verbose` | すべての引数、結果、デバッグログ |

CLI では、`/verbose` でこれらのモードを切り替えます。メッセージングプラットフォーム（Telegram、Discord、Slack など）で `/verbose` を使うには、上記の `display` セクションで `tool_progress_command: true` を設定してください。すると、コマンドがモードを切り替えて config に保存します。

### ランタイムメタデータのフッター（ゲートウェイのみ）

`display.runtime_footer.enabled: true` のとき、Hermes は各ゲートウェイターンの **最終** メッセージに小さなランタイムコンテキストのフッターを追加します — CLI がステータスバーに表示するのと同じ情報（モデル、コンテキスト %、cwd、セッション時間、トークン、コスト）です。デフォルトは無効。チームがすべての返信に出所を含めたい場合は、ゲートウェイごとにオプトインしてください。

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # model, context_pct, cwd, duration, tokens, cost のいずれか
```

`/footer` スラッシュコマンドは、任意のセッションで実行時にこれを切り替えます。

Telegram/Discord/Slack の返信に追加されるフッターの例:

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

フッターが付くのはターンの **最終** メッセージのみです。途中の更新はクリーンなままです。

### プラットフォームごとの進捗オーバーライド

プラットフォームによって、必要な冗長度は異なります。例えば Signal はメッセージを編集できないため、各進捗更新が別々のメッセージになります — うるさいです。`display.platforms` を使ってプラットフォームごとのモードを設定してください。

```yaml
display:
  tool_progress: all          # グローバルのデフォルト
  platforms:
    signal:
      tool_progress: 'off'    # Signal では進捗を無音化
    telegram:
      tool_progress: verbose  # Telegram では詳細な進捗
    slack:
      tool_progress: 'off'    # 共有 Slack ワークスペースでは静かに
```

オーバーライドのないプラットフォームは、グローバルの `tool_progress` 値にフォールバックします。有効なプラットフォームキー: `telegram`、`discord`、`slack`、`signal`、`whatsapp`、`matrix`、`mattermost`、`email`、`sms`、`homeassistant`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot`。レガシーの `display.tool_progress_overrides` キーは後方互換のため今もロードされますが、非推奨であり、初回ロード時に `display.platforms` に移行されます。

`interim_assistant_messages` はゲートウェイ専用です。有効にすると、Hermes は完了したターン途中のアシスタント更新を別々のチャットメッセージとして送信します。これは `tool_progress` とは独立しており、ゲートウェイのストリーミングを必要としません。

## プライバシー

```yaml
privacy:
  redact_pii: false  # LLM のコンテキストから PII を除去（ゲートウェイのみ）
```

`redact_pii` が `true` のとき、ゲートウェイは、サポートされているプラットフォームで LLM に送信する前に、システムプロンプトから個人を特定できる情報を伏字化します。

| フィールド | 扱い |
|-------|-----------|
| 電話番号（WhatsApp/Signal のユーザー ID） | `user_<12 文字の sha256>` にハッシュ化 |
| ユーザー ID | `user_<12 文字の sha256>` にハッシュ化 |
| チャット ID | 数値部分をハッシュ化、プラットフォームのプレフィックスは保持（`telegram:<hash>`） |
| ホームチャンネル ID | 数値部分をハッシュ化 |
| ユーザー名 / ユーザーネーム | **影響なし**（ユーザーが選択、公に見える） |

**プラットフォームのサポート:** 伏字化は WhatsApp、Signal、Telegram に適用されます。Discord と Slack は、メンションシステム（`<@user_id>`）が LLM のコンテキストに実際の ID を必要とするため除外されます。

ハッシュは決定的です — 同じユーザーは常に同じハッシュにマッピングされるため、モデルはグループチャットでもユーザーを区別できます。ルーティングと配信は内部的に元の値を使用します。

## 音声認識（STT）

```yaml
stt:
  provider: "local"            # "local" | "groq" | "openai" | "mistral"
  local:
    model: "base"              # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # レガシーのフォールバックキーも今も尊重される
```

プロバイダーの動作:

- `local` は、あなたのマシン上で動作する `faster-whisper` を使用します。`pip install faster-whisper` で別途インストールしてください。
- `groq` は Groq の Whisper 互換エンドポイントを使用し、`GROQ_API_KEY` を読み取ります。
- `openai` は OpenAI の音声 API を使用し、`VOICE_TOOLS_OPENAI_KEY` を読み取ります。

要求されたプロバイダーが利用できない場合、Hermes は次の順序で自動的にフォールバックします: `local` → `groq` → `openai`。

Groq と OpenAI のモデルのオーバーライドは環境変数で行います。

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

## 音声モード（CLI）

```yaml
voice:
  record_key: "ctrl+b"         # CLI 内のプッシュトゥトークキー
  max_recording_seconds: 120    # 長い録音のハードストップ
  auto_tts: false               # /voice on のときに音声返信を自動的に有効化
  beep_enabled: true            # CLI 音声モードで録音開始/停止のビープを鳴らす
  silence_threshold: 200        # 発話検出のための RMS 閾値
  silence_duration: 3.0         # 自動停止までの無音の秒数
```

CLI でマイクモードを有効にするには `/voice on` を、録音の開始/停止には `record_key` を、音声返信の切り替えには `/voice tts` を使ってください。エンドツーエンドのセットアップとプラットフォーム固有の動作については[音声モード](/docs/user-guide/features/voice-mode)を参照してください。

## ストリーミング

完全なレスポンスを待つ代わりに、トークンを到着次第ターミナルやメッセージングプラットフォームにストリーミングします。

### CLI ストリーミング

```yaml
display:
  streaming: true         # トークンをリアルタイムでターミナルにストリーミング
  show_reasoning: true    # 推論/思考トークンもストリーミング（オプション）
```

有効にすると、レスポンスはストリーミングボックス内にトークンごとに表示されます。ツール呼び出しは引き続き静かに捕捉されます。プロバイダーがストリーミングをサポートしていない場合、自動的に通常の表示にフォールバックします。

### ゲートウェイのストリーミング（Telegram、Discord、Slack）

```yaml
streaming:
  enabled: true           # 段階的なメッセージ編集を有効化
  transport: edit         # "edit"（段階的なメッセージ編集）または "off"
  edit_interval: 0.3      # メッセージ編集の間隔（秒）
  buffer_threshold: 40    # 編集フラッシュを強制する前の文字数
  cursor: " ▉"            # ストリーミング中に表示されるカーソル
  fresh_final_after_seconds: 60   # プレビューがこの古さのとき新しい最終を送信（Telegram）。0 = 常にその場で編集
```

有効にすると、ボットは最初のトークンでメッセージを送信し、より多くのトークンが到着するにつれて段階的に編集します。メッセージ編集をサポートしないプラットフォーム（Signal、Email、Home Assistant）は最初の試行で自動検出されます — そのセッションではストリーミングが穏やかに無効化され、メッセージが氾濫することはありません。

段階的なトークン編集なしに、自然なターン途中のアシスタント更新を別々に送るには、`display.interim_assistant_messages: true` を設定してください。

**オーバーフローの処理:** ストリーミングされたテキストがプラットフォームのメッセージ長制限（約 4096 文字）を超えると、現在のメッセージが確定され、新しいものが自動的に始まります。

**新しい最終（Telegram）:** Telegram の `editMessageText` は元のメッセージのタイムスタンプを保持するため、長時間ストリーミングされた返信は、完了後も最初のトークンのタイムスタンプを保ち続けます。`fresh_final_after_seconds > 0`（デフォルト `60`）のとき、完了した返信は真新しいメッセージとして配信され（古いプレビューはベストエフォートで削除）、Telegram の表示タイムスタンプが完了時刻を反映するようにします。短いプレビューは引き続きその場で確定します。常にその場で編集するには `0` を設定してください。

:::note
ストリーミングはデフォルトで無効です。ストリーミングの UX を試すには、`~/.hermes/config.yaml` で有効にしてください。
:::

## グループチャットのセッション分離

共有チャットがルームごとに 1 つの会話を保つか、参加者ごとに 1 つの会話を保つかを制御します。

```yaml
group_sessions_per_user: true  # true = グループ/チャンネルで参加者ごとに分離、false = チャットごとに 1 つの共有セッション
```

- `true` がデフォルトであり、推奨される設定です。Discord チャンネル、Telegram グループ、Slack チャンネルなどの共有コンテキストでは、プラットフォームがユーザー ID を提供する場合、各送信者が独自のセッションを取得します。
- `false` は古い共有ルームの動作に戻します。Hermes にチャンネルを 1 つの協働的な会話として扱わせたいことが明確な場合に役立ちますが、ユーザーがコンテキスト、トークンコスト、割り込み状態を共有することにもなります。
- ダイレクトメッセージは影響を受けません。Hermes は引き続き、DM をチャット/DM ID で通常どおりキー付けします。
- スレッドはどちらの場合も親チャンネルから分離されたままです。`true` の場合、各参加者はスレッド内でも独自のセッションを取得します。

動作の詳細と例については、[セッション](/docs/user-guide/sessions)と [Discord ガイド](/docs/user-guide/messaging/discord)を参照してください。

## 未認可の DM の動作

未知のユーザーがダイレクトメッセージを送ってきたときに Hermes が何をするかを制御します。

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` がデフォルトです。Hermes はアクセスを拒否しますが、DM で一度きりのペアリングコードを返信します。
- `ignore` は未認可の DM を静かに破棄します。
- プラットフォームセクションはグローバルのデフォルトをオーバーライドするため、ペアリングを広く有効にしたまま、1 つのプラットフォームだけを静かにすることができます。

## クイックコマンド

LLM を呼び出さずにシェルコマンドを実行するか、あるスラッシュコマンドを別のものにエイリアスするカスタムコマンドを定義します。Exec のクイックコマンドはゼロトークンで、メッセージングプラットフォーム（Telegram、Discord など）からのクイックなサーバーチェックやユーティリティスクリプトに役立ちます。

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
    command: cd ~/.hermes/hermes-agent && git pull && pip install -e .
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

使い方: CLI または任意のメッセージングプラットフォームで `/status`、`/disk`、`/update`、`/gpu`、`/restart` と入力します。`exec` コマンドはホスト上でローカルに実行され、出力を直接返します — LLM 呼び出しなし、トークン消費なし。`alias` コマンドは、設定したスラッシュコマンドのターゲットに書き換えられます。

- **30 秒のタイムアウト** — 長時間実行されるコマンドはエラーメッセージとともに終了されます
- **優先度** — クイックコマンドはスキルコマンドより先にチェックされるため、スキル名をオーバーライドできます
- **オートコンプリート** — クイックコマンドはディスパッチ時に解決され、組み込みのスラッシュコマンドのオートコンプリート表には表示されません
- **タイプ** — サポートされるタイプは `exec` と `alias` です。それ以外のタイプはエラーを表示します
- **どこでも動作** — CLI、Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant

文字列のみのプロンプトショートカットは有効なクイックコマンドではありません。再利用可能なプロンプトのワークフローには、スキルを作成するか、既存のスラッシュコマンドにエイリアスしてください。

## ヒューマンディレイ

メッセージングプラットフォームで人間らしいレスポンスのペースをシミュレートします。

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 800                  # 最小ディレイ（custom モード）
  max_ms: 2500                 # 最大ディレイ（custom モード）
```

## コード実行

`execute_code` ツールを設定します。

```yaml
code_execution:
  mode: project                # project（デフォルト） | strict
  timeout: 300                 # 最大実行時間（秒）
  max_tool_calls: 50           # コード実行内での最大ツール呼び出し数
```

**`mode`** はスクリプトの作業ディレクトリと Python インタープリタを制御します。

- **`project`**（デフォルト） — スクリプトはセッションの作業ディレクトリで、アクティブな virtualenv/conda 環境の python で実行されます。プロジェクトの依存関係（`pandas`、`torch`、プロジェクトパッケージ）と相対パス（`.env`、`./data.csv`）が自然に解決され、`terminal()` が見るものと一致します。
- **`strict`** — スクリプトは一時的なステージングディレクトリで `sys.executable`（Hermes 自身の python）で実行されます。最大限の再現性が得られますが、プロジェクトの依存関係と相対パスは解決されません。

環境のスクラビング（`*_API_KEY`、`*_TOKEN`、`*_SECRET`、`*_PASSWORD`、`*_CREDENTIAL`、`*_PASSWD`、`*_AUTH` を除去）とツールのホワイトリストは、両方のモードで同一に適用されます — モードの切り替えはセキュリティの姿勢を変えません。

## Web 検索バックエンド

`web_search`、`web_extract`、`web_crawl` ツールは 5 つのバックエンドプロバイダーをサポートします。バックエンドは `config.yaml` または `hermes tools` で設定します。

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # または、機能ごとのキーでプロバイダーを混在させる（例: 無料の検索 + 有料の抽出）:
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| バックエンド | 環境変数 | 検索 | 抽出 | クロール |
|---------|---------|--------|---------|-------|
| **Firecrawl**（デフォルト） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

**バックエンドの選択:** `web.backend` が設定されていない場合、バックエンドは利用可能な API キーから自動検出されます。`SEARXNG_URL` のみが設定されている場合は SearXNG が使われます。`EXA_API_KEY` のみの場合は Exa が、`TAVILY_API_KEY` のみの場合は Tavily が、`PARALLEL_API_KEY` のみの場合は Parallel が使われます。それ以外の場合は Firecrawl がデフォルトです。

**SearXNG** は、無料でセルフホスト可能な、プライバシーを尊重するメタ検索エンジンで、70 以上の検索エンジンに問い合わせます。API キーは不要 — `SEARXNG_URL` をあなたのインスタンス（例: `http://localhost:8080`）に設定するだけです。SearXNG は検索専用です。`web_extract` と `web_crawl` には別の抽出プロバイダーが必要です（`web.extract_backend` を設定）。Docker のセットアップ手順については [Web 検索のセットアップガイド](/docs/user-guide/features/web-search)を参照してください。

**セルフホストの Firecrawl:** `FIRECRAWL_API_URL` を自分のインスタンスを指すように設定します。カスタム URL が設定されると、API キーはオプションになります（認証を無効にするには、サーバーで `USE_DB_AUTHENTICATION=***` を設定）。

**Parallel の検索モード:** `PARALLEL_SEARCH_MODE` を設定して検索動作を制御します — `fast`、`one-shot`、`agentic`（デフォルト: `agentic`）。

**Exa:** `~/.hermes/.env` に `EXA_API_KEY` を設定します。`category` フィルタ（`company`、`research paper`、`news`、`people`、`personal site`、`pdf`）とドメイン/日付フィルタをサポートします。

## ブラウザ

ブラウザ自動化の動作を設定します。

```yaml
browser:
  inactivity_timeout: 120        # アイドルセッションを自動クローズするまでの秒数
  command_timeout: 30             # ブラウザコマンド（スクリーンショット、ナビゲートなど）のタイムアウト（秒）
  record_sessions: false         # ブラウザセッションを WebM 動画として ~/.hermes/browser_recordings/ に自動録画
  # オプションの CDP オーバーライド — 設定すると、Hermes はヘッドレスブラウザを起動するのではなく、
  # （/browser connect 経由で）あなた自身の Chrome に直接アタッチします。
  cdp_url: ""
  # ダイアログスーパーバイザー — CDP バックエンドがアタッチされている場合（Browserbase、
  # /browser connect 経由のローカル Chrome）に、ネイティブの JS ダイアログ（alert / confirm / prompt）
  # をどう扱うかを制御します。Camofox とデフォルトのローカルエージェントブラウザモードでは無視されます。
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # must_respond 下での安全のための自動却下（秒）
  camofox:
    managed_persistence: false   # true のとき、Camofox セッションは再起動をまたいで Cookie/ログインを永続化
```

**ダイアログポリシー:**

- `must_respond`（デフォルト） — ダイアログを捕捉し、`browser_snapshot.pending_dialogs` に表示し、エージェントが `browser_dialog(action=...)` を呼ぶのを待ちます。`dialog_timeout_s` 秒間応答がない場合、ページの JS スレッドが永久に止まるのを防ぐため、ダイアログは自動的に却下されます。
- `auto_dismiss` — 捕捉し、即座に却下します。エージェントは後から `browser_snapshot.recent_dialogs` で `closed_by="auto_policy"` 付きのダイアログ記録を見ることができます。
- `auto_accept` — 捕捉し、即座に承諾します。攻撃的な `beforeunload` プロンプトを持つページに有用です。

完全なダイアログのワークフローについては、[ブラウザ機能ページ](./features/browser.md#browser_dialog)を参照してください。

ブラウザツールセットは複数のプロバイダーをサポートします。Browserbase、Browser Use、ローカル Chrome の CDP セットアップの詳細については、[ブラウザ機能ページ](/docs/user-guide/features/browser)を参照してください。

## タイムゾーン

サーバーローカルのタイムゾーンを IANA タイムゾーン文字列でオーバーライドします。ログのタイムスタンプ、cron のスケジューリング、システムプロンプトの時刻注入に影響します。

```yaml
timezone: "America/New_York"   # IANA タイムゾーン（デフォルト: "" = サーバーローカル時刻）
```

サポートされる値: 任意の IANA タイムゾーン識別子（例: `America/New_York`、`Europe/London`、`Asia/Kolkata`、`UTC`）。サーバーローカル時刻にするには空のままにするか省略してください。

## Discord

メッセージングゲートウェイの Discord 固有の動作を設定します。

```yaml
discord:
  require_mention: true          # サーバーチャンネルで応答するために @メンションを要求
  free_response_channels: ""     # @メンションなしでボットが応答するチャンネル ID（カンマ区切り）
  auto_thread: true              # チャンネルで @メンション時にスレッドを自動作成
```

- `require_mention` — `true`（デフォルト）のとき、ボットはサーバーチャンネルで `@BotName` とメンションされた場合にのみ応答します。DM はメンションなしで常に機能します。
- `free_response_channels` — メンションを要求せずにボットがすべてのメッセージに応答するチャンネル ID のカンマ区切りリスト。
- `auto_thread` — `true`（デフォルト）のとき、チャンネルでのメンションは会話用のスレッドを自動的に作成し、チャンネルをクリーンに保ちます（Slack のスレッド化に似ています）。

## セキュリティ

実行前のセキュリティスキャンとシークレットの伏字化:

```yaml
security:
  redact_secrets: false          # ツール出力とログ内の API キーパターンを伏字化（デフォルトで無効）
  tirith_enabled: true           # ターミナルコマンドに対する Tirith セキュリティスキャンを有効化
  tirith_path: "tirith"          # tirith バイナリへのパス（デフォルト: $PATH 内の "tirith"）
  tirith_timeout: 5              # tirith スキャンがタイムアウトするまでの待機秒数
  tirith_fail_open: true         # tirith が利用できない場合にコマンド実行を許可
  website_blocklist:             # 後述の「ウェブサイトブロックリスト」セクションを参照
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` — `true` のとき、API キー、トークン、パスワードのように見えるパターンを、会話のコンテキストとログに入る前にツール出力で自動的に検出・伏字化します。**デフォルトで無効** — ツール出力で実際の認証情報を日常的に扱い、安全網が欲しい場合に有効にしてください。オンにするには明示的に `true` を設定します。
- `tirith_enabled` — `true` のとき、ターミナルコマンドは実行前に [Tirith](https://github.com/StackGuardian/tirith) によってスキャンされ、潜在的に危険な操作を検出します。
- `tirith_path` — tirith バイナリへのパス。tirith が非標準の場所にインストールされている場合に設定します。
- `tirith_timeout` — tirith スキャンを待つ最大秒数。スキャンがタイムアウトするとコマンドは続行されます。
- `tirith_fail_open` — `true`（デフォルト）のとき、tirith が利用できないか失敗した場合にコマンドの実行が許可されます。tirith が検証できない場合にコマンドをブロックするには `false` を設定します。

## ウェブサイトブロックリスト

特定のドメインを、エージェントの Web およびブラウザツールからのアクセスからブロックします。

```yaml
security:
  website_blocklist:
    enabled: false               # URL ブロックを有効化（デフォルト: false）
    domains:                     # ブロックするドメインパターンのリスト
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # 外部ファイルから追加のルールをロード
      - "/etc/hermes/blocked-sites.txt"
```

有効にすると、ブロックされたドメインパターンにマッチする URL は、Web またはブラウザツールが実行される前に拒否されます。これは `web_search`、`web_extract`、`browser_navigate`、および URL にアクセスする任意のツールに適用されます。

ドメインルールのサポート:
- 完全一致のドメイン: `admin.example.com`
- ワイルドカードのサブドメイン: `*.internal.company.com`（すべてのサブドメインをブロック）
- TLD ワイルドカード: `*.local`

共有ファイルには 1 行に 1 つのドメインルールを記述します（空行と `#` のコメントは無視されます）。存在しないか読み取れないファイルは警告を記録しますが、他の Web ツールを無効にはしません。

ポリシーは 30 秒間キャッシュされるため、設定変更は再起動なしですぐに反映されます。

## スマート承認

Hermes が潜在的に危険なコマンドをどう扱うかを制御します。

```yaml
approvals:
  mode: manual   # manual | smart | off
```

| モード | 動作 |
|------|----------|
| `manual`（デフォルト） | フラグの立ったコマンドを実行する前にユーザーに確認します。CLI では対話的な承認ダイアログを表示します。メッセージングでは保留中の承認リクエストをキューに入れます。 |
| `smart` | 補助 LLM を使って、フラグの立ったコマンドが実際に危険かどうかを評価します。低リスクのコマンドはセッションレベルの永続化で自動承認されます。本当にリスクのあるコマンドはユーザーにエスカレーションされます。 |
| `off` | すべての承認チェックをスキップします。`HERMES_YOLO_MODE=true` と同等です。**注意して使ってください。** |

スマートモードは承認疲れの軽減に特に有用です — 安全な操作ではエージェントがより自律的に作業できるようにしつつ、本当に破壊的なコマンドは依然として捕捉します。

:::warning
`approvals.mode: off` を設定すると、ターミナルコマンドのすべての安全チェックが無効になります。信頼できるサンドボックス化された環境でのみ使用してください。
:::

## チェックポイント

破壊的なファイル操作の前の自動的なファイルシステムスナップショット。詳細は[チェックポイントとロールバック](/docs/user-guide/checkpoints-and-rollback)を参照してください。

```yaml
checkpoints:
  enabled: false                 # 自動チェックポイントを有効化（hermes chat --checkpoints でも可）。デフォルト: false（オプトイン）。
  max_snapshots: 20              # ディレクトリごとに保持する最大チェックポイント数（デフォルト: 20）
```


## 委譲

delegate ツールのサブエージェントの動作を設定します。

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # モデルをオーバーライド（空 = 親を継承）
  # provider: "openrouter"                  # プロバイダーをオーバーライド（空 = 親を継承）
  # base_url: "http://localhost:1234/v1"    # 直接の OpenAI 互換エンドポイント（provider より優先）
  # api_key: "local-key"                    # base_url 用の API キー（OPENAI_API_KEY にフォールバック）
  max_concurrent_children: 3                # バッチあたりの並列の子（最小 1、上限なし）。DELEGATION_MAX_CONCURRENT_CHILDREN 環境変数でも可。
  max_spawn_depth: 1                        # 委譲ツリーの深さの上限（1〜3、制限）。1 = フラット（デフォルト）: 親は委譲できないリーフを生成。2 = オーケストレーターの子がリーフの孫を生成可能。3 = 3 レベル。
  orchestrator_enabled: true                # グローバルのキルスイッチ。false のとき、role="orchestrator" は無視され、max_spawn_depth にかかわらずすべての子がリーフに強制される。
```

**サブエージェントの provider:model オーバーライド:** デフォルトでは、サブエージェントは親エージェントのプロバイダーとモデルを継承します。`delegation.provider` と `delegation.model` を設定すると、サブエージェントを別の provider:model のペアにルーティングできます — 例えば、プライマリのエージェントが高価な推論モデルを実行している間、範囲の狭いサブタスクには安価/高速なモデルを使うなど。

**直接のエンドポイントオーバーライド:** 分かりやすいカスタムエンドポイントの経路が欲しい場合は、`delegation.base_url`、`delegation.api_key`、`delegation.model` を設定してください。これはサブエージェントをその OpenAI 互換エンドポイントへ直接送り、`delegation.provider` より優先されます。`delegation.api_key` を省略すると、Hermes は `OPENAI_API_KEY` のみにフォールバックします。

委譲プロバイダーは、CLI/ゲートウェイの起動時と同じ認証情報解決を使用します。設定済みのすべてのプロバイダーがサポートされます: `openrouter`、`nous`、`copilot`、`zai`、`kimi-coding`、`minimax`、`minimax-cn`。プロバイダーが設定されると、システムは正しいベース URL、API キー、API モードを自動的に解決します — 手動の認証情報の配線は不要です。

**優先順位:** config の `delegation.base_url` → config の `delegation.provider` → 親プロバイダー（継承）。config の `delegation.model` → 親モデル（継承）。`provider` なしで `model` だけを設定すると、親の認証情報を保ちつつモデル名のみを変更します（OpenRouter のように同じプロバイダー内でモデルを切り替えるのに有用）。

**幅と深さ:** `max_concurrent_children` は、バッチあたりに何個のサブエージェントを並列実行するかを制限します（デフォルト `3`、最小 1、上限なし）。`DELEGATION_MAX_CONCURRENT_CHILDREN` 環境変数でも設定できます。モデルが上限より長い `tasks` 配列を送信すると、`delegate_task` は静かに切り詰めるのではなく、制限を説明するツールエラーを返します。`max_spawn_depth` は委譲ツリーの深さを制御します（1〜3 に制限）。デフォルトの `1` では、委譲はフラットです: 子は孫を生成できず、`role="orchestrator"` を渡しても静かに `leaf` に降格します。`2` に上げると、オーケストレーターの子がリーフの孫を生成でき、`3` で 3 レベルのツリーになります。エージェントは呼び出しごとに `role="orchestrator"` でオーケストレーションを選択します。`orchestrator_enabled: false` は、すべての子をリーフに強制的に戻します。コストは乗算的にスケールします — `max_spawn_depth: 3` かつ `max_concurrent_children: 3` では、ツリーは 3×3×3 = 27 個の並列リーフエージェントに達することがあります。使用パターンについては[サブエージェントの委譲 → 深さの制限とネストされたオーケストレーション](features/delegation.md#depth-limit-and-nested-orchestration)を参照してください。

## 確認（Clarify）

確認プロンプトの動作を設定します。

```yaml
clarify:
  timeout: 120                 # ユーザーの確認応答を待つ秒数
```

## コンテキストファイル（SOUL.md、AGENTS.md）

Hermes は 2 つの異なるコンテキストスコープを使用します。

| ファイル | 目的 | スコープ |
|------|---------|-------|
| `SOUL.md` | **エージェントの主要なアイデンティティ** — エージェントが誰であるかを定義（システムプロンプトのスロット #1） | `~/.hermes/SOUL.md` または `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | プロジェクト固有の指示（最高優先度） | git ルートまで遡る |
| `AGENTS.md` | プロジェクト固有の指示、コーディング規約 | ディレクトリを再帰的に走査 |
| `CLAUDE.md` | Claude Code のコンテキストファイル（これも検出） | 作業ディレクトリのみ |
| `.cursorrules` | Cursor IDE のルール（これも検出） | 作業ディレクトリのみ |
| `.cursor/rules/*.mdc` | Cursor のルールファイル（これも検出） | 作業ディレクトリのみ |

- **SOUL.md** はエージェントの主要なアイデンティティです。システムプロンプトのスロット #1 を占め、組み込みのデフォルトのアイデンティティを完全に置き換えます。エージェントが誰であるかを完全にカスタマイズするために編集してください。
- SOUL.md が欠けている、空である、またはロードできない場合、Hermes は組み込みのデフォルトのアイデンティティにフォールバックします。
- **プロジェクトのコンテキストファイルは優先システムを使用します** — ロードされるのは 1 つのタイプのみです（最初にマッチしたものが優先）: `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。SOUL.md は常に独立してロードされます。
- **AGENTS.md** は階層的です: サブディレクトリにも AGENTS.md がある場合、すべてが結合されます。
- Hermes は、デフォルトの `SOUL.md` がまだ存在しない場合、自動的にシードします。
- ロードされるすべてのコンテキストファイルは、スマートな切り詰めとともに 20,000 文字に制限されます。

関連項目:
- [パーソナリティと SOUL.md](/docs/user-guide/features/personality)
- [コンテキストファイル](/docs/user-guide/features/context-files)

## 作業ディレクトリ

| コンテキスト | デフォルト |
|---------|---------|
| **CLI（`hermes`）** | コマンドを実行する現在のディレクトリ |
| **メッセージングゲートウェイ** | ホームディレクトリ `~`（`MESSAGING_CWD` でオーバーライド） |
| **Docker / Singularity / Modal / SSH** | コンテナまたはリモートマシン内のユーザーのホームディレクトリ |

作業ディレクトリをオーバーライドする:
```bash
# ~/.hermes/.env または ~/.hermes/config.yaml 内:
MESSAGING_CWD=/home/myuser/projects    # ゲートウェイのセッション
TERMINAL_CWD=/workspace                # すべてのターミナルセッション
```
