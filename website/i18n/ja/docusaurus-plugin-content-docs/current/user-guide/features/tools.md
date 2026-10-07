---
sidebar_position: 1
title: "ツールとツールセット"
description: "Hermes Agent のツールの概要 — 何が利用可能か、ツールセットの仕組み、ターミナルバックエンド"
---

# ツールとツールセット

ツールは、エージェントの機能を拡張する関数です。それらは論理的な**ツールセット**にまとめられ、プラットフォームごとに有効化または無効化できます。

## 利用可能なツール

Hermes は、Web 検索、ブラウザ自動化、ターミナル実行、ファイル編集、メモリ、委譲、RL トレーニング、メッセージング配信、Home Assistant などをカバーする広範な組み込みツールレジストリを同梱しています。

:::note
**Honcho クロスセッションメモリ**は、組み込みツールセットではなく、メモリプロバイダープラグイン（`plugins/memory/honcho/`）として利用可能です。インストールについては [Plugins](./plugins.md) を参照してください。
:::

ハイレベルなカテゴリ:

| カテゴリ | 例 | 説明 |
|----------|-----|------|
| **Web** | `web_search`、`web_extract` | Web を検索し、ページコンテンツを抽出します。 |
| **ターミナルとファイル** | `terminal`、`process`、`read_file`、`patch` | コマンドを実行し、ファイルを操作します。 |
| **ブラウザ** | `browser_navigate`、`browser_snapshot`、`browser_vision` | テキストとビジョンをサポートするインタラクティブなブラウザ自動化。 |
| **メディア** | `vision_analyze`、`image_generate`、`text_to_speech` | マルチモーダルな分析と生成。 |
| **エージェントオーケストレーション** | `todo`、`clarify`、`execute_code`、`delegate_task` | プランニング、明確化、コード実行、サブエージェント委譲。 |
| **メモリと想起** | `memory`、`session_search` | 永続メモリとセッション検索。 |
| **自動化と配信** | `cronjob`、`send_message` | create/list/update/pause/resume/run/remove アクションを持つスケジュールタスク、加えてアウトバウンドのメッセージング配信。 |
| **統合** | `ha_*`、MCP サーバーツール、`rl_*` | Home Assistant、MCP、RL トレーニング、その他の統合。 |

権威あるコード由来のレジストリについては、[Built-in Tools Reference](/docs/reference/tools-reference) と [Toolsets Reference](/docs/reference/toolsets-reference) を参照してください。

:::tip Nous Tool Gateway
有料の [Nous Portal](https://portal.nousresearch.com) のサブスクライバーは、**[Tool Gateway](tool-gateway.md)** を通じて Web 検索、画像生成、TTS、ブラウザ自動化を利用できます — 別個の API キーは不要です。`hermes model` を実行して有効化するか、`hermes tools` で個別のツールを設定してください。
:::

## ツールセットの使用

```bash
# 特定のツールセットを使用
hermes chat --toolsets "web,terminal"

# 利用可能なすべてのツールを表示
hermes tools

# プラットフォームごとにツールを設定（インタラクティブ）
hermes tools
```

一般的なツールセットには、`web`、`search`、`terminal`、`file`、`browser`、`vision`、`image_gen`、`moa`、`skills`、`tts`、`todo`、`memory`、`session_search`、`cronjob`、`code_execution`、`delegation`、`clarify`、`homeassistant`、`messaging`、`spotify`、`discord`、`discord_admin`、`debugging`、`safe`、`rl` があります。

`hermes-cli`、`hermes-telegram` などのプラットフォームプリセットや、`mcp-<server>` のような動的 MCP ツールセットを含む全セットについては、[Toolsets Reference](/docs/reference/toolsets-reference) を参照してください。

## ターミナルバックエンド

ターミナルツールは、さまざまな環境でコマンドを実行できます:

| バックエンド | 説明 | ユースケース |
|---------|------|----------|
| `local` | 自分のマシンで実行（デフォルト） | 開発、信頼できるタスク |
| `docker` | 隔離されたコンテナ | セキュリティ、再現性 |
| `ssh` | リモートサーバー | サンドボックス化、エージェントを自身のコードから遠ざける |
| `singularity` | HPC コンテナ | クラスタコンピューティング、rootless |
| `modal` | クラウド実行 | サーバーレス、スケール |
| `daytona` | クラウドサンドボックスワークスペース | 永続的なリモート開発環境 |
| `vercel_sandbox` | Vercel Sandbox クラウド microVM | スナップショットベースのファイルシステム永続化を伴うクラウド実行 |

### 設定

```yaml
# ~/.hermes/config.yaml 内
terminal:
  backend: local    # または: docker, ssh, singularity, modal, daytona, vercel_sandbox
  cwd: "."          # 作業ディレクトリ
  timeout: 180      # コマンドタイムアウト（秒）
```

### Docker バックエンド

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**プロセス全体で共有される、1 つの永続コンテナ。** Hermes は初回使用時に単一の長寿命コンテナを起動し（`docker run -d ... sleep 2h`）、すべての terminal、file、`execute_code` の呼び出しを `docker exec` 経由でその同じコンテナにルーティングします。作業ディレクトリの変更、インストールされたパッケージ、環境の調整、`/workspace` に書き込まれたファイルはすべて、`/new`、`/reset`、`delegate_task` のサブエージェントをまたいで、1 回のツールコールから次のツールコールへと、Hermes プロセスのライフタイムにわたって引き継がれます。コンテナはシャットダウン時に停止・削除されます。

これは、Docker バックエンドがコマンドごとに新しいコンテナではなく、永続的なサンドボックス VM のように振る舞うことを意味します。一度 `pip install foo` すれば、セッションの残りの間それは存在します。`cd /workspace/project` すれば、以降の `ls` 呼び出しはそのディレクトリを見ます。完全なライフサイクルの詳細と、Hermes の再起動をまたいで `/workspace` と `/root` が生き残るかどうかを制御する `container_persistent` フラグについては、[Configuration → Docker Backend](../configuration.md#docker-backend) を参照してください。

### SSH バックエンド

セキュリティのために推奨されます — エージェントが自身のコードを変更できません:

```yaml
terminal:
  backend: ssh
```
```bash
# ~/.hermes/.env でクレデンシャルを設定
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer

```bash
# 並列ワーカー用に SIF を事前ビルド
apptainer build ~/python.sif docker://python:3.11-slim

# 設定
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

### Modal（サーバーレスクラウド）

```bash
uv pip install modal
modal setup
hermes config set terminal.backend modal
```

### Vercel Sandbox

```bash
pip install 'hermes-agent[vercel]'
hermes config set terminal.backend vercel_sandbox
hermes config set terminal.vercel_runtime node24
```

`VERCEL_TOKEN`、`VERCEL_PROJECT_ID`、`VERCEL_TEAM_ID` の 3 つすべてで認証します。このアクセストークンによるセットアップは、Render、Railway、Docker などのホストでのデプロイメントや通常の長時間実行 Hermes プロセスに対してサポートされるパスです。サポートされるランタイムは `node24`、`node22`、`python3.13` です。Hermes はリモートワークスペースのルートとして `/vercel/sandbox` をデフォルトにします。

一度限りのローカル開発には、Hermes は短命の Vercel OIDC トークンも受け入れます:

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

リンクされた Vercel プロジェクトディレクトリから:

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

`container_persistent: true` の場合、Hermes は Vercel スナップショットを使用して、同じタスクのサンドボックス再作成をまたいでファイルシステムの状態を保持します。これには、サンドボックス内の Hermes が同期したクレデンシャル、スキル、キャッシュファイルが含まれることがあります。スナップショットは、ライブプロセス、PID 空間、または同じライブサンドボックスのアイデンティティを保持しません。

バックグラウンドのターミナルコマンドは、Hermes の汎用的な非ローカルプロセスフローを使用します: spawn、poll、wait、log、kill は、サンドボックスが生きている間は通常の process ツールを通じて機能しますが、Hermes はクリーンアップや再起動の後にネイティブな Vercel のデタッチプロセス回復を提供しません。

`container_disk` は未設定のまま、または共有デフォルトの `51200` のままにしてください。カスタムディスクサイジングは Vercel Sandbox ではサポートされておらず、診断/バックエンド作成に失敗します。

### コンテナリソース

すべてのコンテナバックエンドの CPU、メモリ、ディスク、永続化を設定します:

```yaml
terminal:
  backend: docker  # または singularity, modal, daytona, vercel_sandbox
  container_cpu: 1              # CPU コア数（デフォルト: 1）
  container_memory: 5120        # メモリ（MB）（デフォルト: 5GB）
  container_disk: 51200         # ディスク（MB）（デフォルト: 50GB）
  container_persistent: true    # セッションをまたいでファイルシステムを永続化（デフォルト: true）
```

`container_persistent: true` の場合、インストールされたパッケージ、ファイル、設定はセッションをまたいで生き残ります。

### コンテナセキュリティ

すべてのコンテナバックエンドはセキュリティハードニングを施して実行されます:

- 読み取り専用のルートファイルシステム（Docker）
- すべての Linux ケーパビリティをドロップ
- 権限昇格なし
- PID 制限（256 プロセス）
- 完全な名前空間の隔離
- 書き込み可能なルートレイヤーではなく、ボリュームを介した永続ワークスペース

Docker は、`terminal.docker_forward_env` を介して明示的な環境変数の許可リストを受け取ることがオプションで可能ですが、転送された変数はコンテナ内のコマンドから見えるため、そのセッションに公開されているものとして扱うべきです。

## バックグラウンドプロセス管理

バックグラウンドプロセスを開始して管理します:

```python
terminal(command="pytest -v tests/", background=true)
# 返り値: {"session_id": "proc_abc123", "pid": 12345}

# その後 process ツールで管理:
process(action="list")       # 実行中のすべてのプロセスを表示
process(action="poll", session_id="proc_abc123")   # ステータスを確認
process(action="wait", session_id="proc_abc123")   # 完了までブロック
process(action="log", session_id="proc_abc123")    # 完全な出力
process(action="kill", session_id="proc_abc123")   # 終了
process(action="write", session_id="proc_abc123", data="y")  # 入力を送信
```

PTY モード（`pty=true`）は、Codex や Claude Code のようなインタラクティブな CLI ツールを有効化します。

## sudo サポート

コマンドに sudo が必要な場合、パスワードの入力を求められます（セッション中はキャッシュされます）。または `~/.hermes/.env` で `SUDO_PASSWORD` を設定してください。

:::warning
メッセージングプラットフォームでは、sudo が失敗すると、出力に `SUDO_PASSWORD` を `~/.hermes/.env` に追加するヒントが含まれます。
:::
