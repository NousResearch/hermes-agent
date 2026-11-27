---
sidebar_position: 1
title: "ツールとツールセット"
description: "Hermes Agent のツールの概要 — 何が利用可能か、ツールセットの仕組み、ターミナルバックエンド"
---

# ツールとツールセット

このページの Python 依存関係コマンドは、
[PM で準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を使用します。
依存関係を変更した後は、チェックアウトを再アクティベートして Hermes を再起動してください。

ツールは、エージェントの機能を拡張する関数です。それらは論理的な**ツールセット**にまとめられ、プラットフォームごとに有効化または無効化できます。

## 利用可能なツール

Hermes は、Web 検索、ブラウザ自動化、ターミナル実行、ファイル編集、メモリ、委譲、スケジュールタスクなどをカバーする広範な組み込みツールレジストリを同梱しています。プラグインはさらにツールを追加します — たとえば Home Assistant のデバイス制御は `homeassistant` カタログプラグインから提供されます。

:::note
**Honcho クロスセッションメモリ**は、組み込みツールセットではなく、プラグインカタログのメモリプロバイダープラグイン（`hermes plugins install honcho`）として利用可能です。[Memory Providers](./memory-providers.md#honcho) を参照してください。
:::

ハイレベルなカテゴリ:

| カテゴリ | 例 | 説明 |
|----------|-----|------|
| **Web** | `web_search`、`web_extract` | Web を検索し、ページコンテンツを抽出します。 |
| **X Search** | `x_search` | xAI の組み込み `x_search` Responses ツールを介して X（Twitter）の投稿やスレッドを検索します — xAI のクレデンシャル（SuperGrok OAuth または `XAI_API_KEY`）が必要です。デフォルトではオフで、`hermes tools` → 🐦 X (Twitter) Search からオプトインします。 |
| **ターミナルとファイル** | `terminal`、`process`、`read_file`、`patch` | コマンドを実行し、ファイルを操作します。 |
| **ブラウザ** | `browser_exec`、`browser_cdp`、`browser_dialog`、`browser_vault` | ブラウザ自動化: `browser_exec` は CDP 経由で Python から Chromium を操作します（スクリーンショットを含む）。Camofox ブラウザは別のカタログプラグイン（`hermes plugins install camofox`）です。 |
| **メディア** | `vision_analyze`、`image_generate`、`text_to_speech` | マルチモーダルな分析と生成。 |
| **エージェントオーケストレーション** | `todo`、`clarify`、`execute_code`、`delegate_task` | プランニング、明確化、コード実行、サブエージェント委譲。 |
| **メモリと想起** | `memory`、`session_search` | 永続メモリとセッション検索。 |
| **自動化** | `cronjob` | create/list/update/pause/resume/run/remove アクションを持つスケジュールタスク。アウトバウンド配信は、エージェントが呼び出せるツールではなく、cron 自身の配信、`hermes send` CLI、ゲートウェイの通知機能によって処理されます。 |
| **統合** | MCP サーバーツール、プラグインツール | MCP とプラグインの統合。例: Home Assistant（`ha_*`、[`homeassistant` カタログプラグイン](../messaging/homeassistant.md)から）。 |

権威あるコード由来のレジストリについては、[Built-in Tools Reference](../../reference/tools-reference.md) と [Toolsets Reference](../../reference/toolsets-reference.md) を参照してください。

:::tip Nous Tool Gateway
有料の [Nous Portal](https://portal.nousresearch.com) のサブスクライバーは、**[Tool Gateway](tool-gateway.md)** を通じて Web 検索、画像生成、TTS、ブラウザ自動化を利用できます — 別個の API キーは不要です。`hermes model` を実行して有効化するか、`hermes tools` で個別のツールを設定してください。
:::

## ツールセットの使用

```bash
# Use specific toolsets
hermes chat --toolsets "web,terminal"

# See all available tools
hermes tools

# Configure tools per platform (interactive)
hermes tools
```

一般的なツールセットには、`web`、`search`、`terminal`、`file`、`browser`、`vision`、`image_gen`、`skills`、`tts`、`todo`、`memory`、`session_search`、`cronjob`、`code_execution`、`delegation`、`clarify`、`messaging`、`discord`、`discord_admin`、`debugging`、`safe` があります。

`hermes-cli`、`hermes-telegram` などのプラットフォームプリセットや、`mcp-<server>` のような動的 MCP ツールセットを含む全セットについては、[Toolsets Reference](../../reference/toolsets-reference.md) を参照してください。

## ツール結果の注記

エージェントのトランスクリプトを読む際に知っておくと役立つツールの挙動がいくつかあります:

- **シグナルによる終了は説明されます。** ターミナルコマンドがシグナルによって強制終了された場合、結果には単なる数値コードではなく人間が読める注記が含まれます — たとえば終了コード `-9`/`137` は「terminated by signal 9: SIGKILL — often the kernel OOM killer on memory exhaustion, or an explicit kill -9」となり、セグメンテーション違反、abort、SIGTERM、broken pipe、CPU/ファイルサイズ制限も同様にラベル付けされます。負のコード（subprocess のセマンティクス）は断定的に記述されます。シェルの `128+signum` 規約については、アプリケーションが正当にそれらのコードで終了することもあるため、「usually」と含みを持たせた表現になります。
- **UTF-16 のテキストファイルは拒否されず、変換されます。** `read_file` は UTF-16（BOM またはバイトパターンのヒューリスティックにより、どちらのエンディアンでも — Windows のメモ帳のファイルや PowerShell の `>` リダイレクトでよく見られます）を検出し、ファイルをバイナリとして扱う代わりに、表示用に UTF-8 へ変換します。結果には変換を示すヒントが含まれます。`patch`/`write_file` による編集は UTF-8 として再エンコードされます。10 MB を超えるファイルと本当にバイナリのファイルは、引き続きバイナリファイルとして拒否されます。

## ターミナルバックエンド {#terminal-backends}

ターミナルツールは、さまざまな環境でコマンドを実行できます:

| バックエンド | 説明 | ユースケース |
|---------|------|----------|
| `local` | 自分のマシンで実行（デフォルト） | 開発、信頼できるタスク |
| `docker` | 隔離されたコンテナ | セキュリティ、再現性 |
| `ssh` | リモートサーバー | サンドボックス化、エージェントを自身のコードから遠ざける |
| `singularity`（プラグイン） | HPC コンテナ | クラスタコンピューティング、rootless |
| `sprites`（プラグイン） | ステートフルな Fly.io クラウドサンドボックス | チェックポイントとリストアを備えた永続的なクラウド環境 |
| `modal` | クラウド実行 | サーバーレス、スケール |
| `daytona` | クラウドサンドボックスワークスペース | 永続的なリモート開発環境 |
| `vercel_sandbox` | Vercel Sandbox クラウド microVM | スナップショットベースのファイルシステム永続化を伴うクラウド実行 |

### 設定

```yaml
# In ~/.hermes/config.yaml
terminal:
  backend: local    # or: docker, ssh, modal, daytona, vercel_sandbox, or a plugin backend (singularity, sprites)
  cwd: "."          # Working directory
  timeout: 180      # Command timeout in seconds
```

### シェルの起動ファイルと非インタラクティブなコマンド

エージェントのターミナル呼び出しは、シェルを**非インタラクティブ**に実行します — TTY はなく、プロンプトの前に人間もいません。通常のターミナルでは気づかないような重い、あるいはインタラクティブなシェル初期化が、エージェントが実行するすべてのコマンドを壊したり、大幅に遅くしたりすることがあります:

- **遅い初期化（`nvm`、バージョンマネージャー、ネットワークにアクセスするプロンプト）:** 典型的な `nvm.sh` の読み込みは、*すべての*シェル起動に目立つ遅延を加え、エージェントは多数のシェルを起動します。数秒かかる rc ファイルは、手早い `git status` をタイムアウトの危険にさらします。
- **TTY を前提としたブロック:** `.bashrc`/`.zshrc` 内で、プロンプトを出す、`tmux`/`screen` にアタッチする、`read` を呼ぶ、メニューを表示するといった処理は、非インタラクティブなシェルをハングさせます — コマンドが永遠に実行されているように見え、その後タイムアウトします。
- **無条件の出力:** バナーを `echo` する rc ファイルは、エージェントが解析しなければならないすべてのコマンド出力を汚染します。

対処法は、ほとんどのディストリビューションが `.bashrc` の先頭にすでに同梱している標準的なガードです — シェルが非インタラクティブな場合は早期に return し、重い処理やインタラクティブな処理はその下に置きます:

```bash
# ~/.bashrc — keep this guard near the top
case $- in
  *i*) ;;      # interactive: continue
  *) return;;  # non-interactive: stop here
esac

# heavy/interactive init goes BELOW the guard
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

Zsh ユーザーの場合: ログイン専用のセットアップは `.zprofile` に、インタラクティブ専用のセットアップは `.zshrc` に置きます。`.zshenv` は非インタラクティブなものを含むすべてのシェルで実行されるため、最小限に保ってください。rc ファイルでしか `PATH` に追加されないツールがエージェントに本当に必要な場合は、その `PATH` の変更をガードの*上*で export する（パスの export は軽量です）か、バイナリを `~/.local/bin` にシンボリックリンクしてください。

自分のターミナルでは動作するのに、エージェントのターミナルコマンドがハングしたりすぐにタイムアウトしたりする場合は、まずシェルの初期化を疑ってください。

### Docker バックエンド

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**プロセス全体で共有される、1 つの永続コンテナ。** Hermes は初回使用時に単一の長寿命コンテナを起動し（`docker run -d ... sleep infinity`）、すべての terminal、file、`execute_code` の呼び出しを `docker exec` 経由でその同じコンテナにルーティングします。作業ディレクトリの変更、インストールされたパッケージ、環境の調整、`/workspace` に書き込まれたファイルはすべて、`/new`、`/reset`、`delegate_task` のサブエージェントをまたいで、1 回のツールコールから次のツールコールへと、Hermes プロセスのライフタイムにわたって引き継がれます。コンテナはシャットダウン時に停止・削除されます。

これは、Docker バックエンドがコマンドごとに新しいコンテナではなく、永続的なサンドボックス VM のように振る舞うことを意味します。一度 `pip install foo` すれば、セッションの残りの間それは存在します。`cd /workspace/project` すれば、以降の `ls` 呼び出しはそのディレクトリを見ます。完全なライフサイクルの詳細と、Hermes の再起動をまたいで `/workspace` と `/root` が生き残るかどうかを制御する `container_persistent` フラグについては、[Configuration → Docker Backend](../configuration.md#docker-backend) を参照してください。

### SSH バックエンド

セキュリティのために推奨されます — エージェントが自身のコードを変更できません:

```yaml
terminal:
  backend: ssh
```
```bash
# Set credentials in ~/.hermes/.env
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer

公式プラグイン（[`singularity`](https://github.com/NousResearch/hermes-official-plugins/tree/main/singularity)）で、以前は組み込みでした。すでにこれを使用しているプロファイルには、`hermes update` がインストールします。

```bash
hermes plugins install singularity

# Pre-build SIF for parallel workers
apptainer build ~/python.sif docker://python:3.11-slim

# Configure
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

### Sprites

Fly.io 上のステートフルな [Sprite](https://fly.io/sprites) でコマンドを実行するプラグイン（[`sprites`](https://github.com/NousResearch/hermes-plugin-sprites)）です。[Sprites Backend](../configuration.md#sprites-backend) を参照してください。

```bash
hermes plugins install NousResearch/hermes-plugin-sprites --enable
hermes config set terminal.backend sprites
# Add SPRITES_TOKEN to ~/.hermes/.env
```

### Modal（サーバーレスクラウド）

```bash
python -c "import pm; pm.sync_venv(['modal'], explicit=True)"
modal setup
hermes config set terminal.backend modal
```

### Vercel Sandbox

```bash
python -c "import pm; pm.sync_venv(['vercel'], explicit=True)"
hermes config set terminal.backend vercel_sandbox
hermes config set terminal.vercel_image vercel/sandbox/universal:latest
```

`VERCEL_TOKEN`、`VERCEL_PROJECT_ID`、`VERCEL_TEAM_ID` の 3 つすべてで認証します。このアクセストークンによるセットアップは、Render、Railway、Docker などのホストでのデプロイメントや通常の長時間実行 Hermes プロセスに対してサポートされるパスです。新しいサンドボックスは `terminal.vercel_image`（デフォルトは `vercel/sandbox/universal:latest`。従来の `vercel_runtime` プリセットは Vercel により非推奨になっています）から起動します。Hermes はリモートワークスペースのルートとして `/vercel/sandbox` をデフォルトにします。

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
  backend: docker  # or modal, daytona, vercel_sandbox, singularity (plugin)
  container_cpu: 1              # CPU cores (default: 1)
  container_memory: 5120        # Memory in MB (default: 5GB)
  container_disk: 51200         # Disk in MB (default: 50GB)
  container_persistent: true    # Persist filesystem across sessions (default: true)
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
# Returns: {"session_id": "proc_abc123", "pid": 12345}

# Then manage with the process tool:
process(action="list")       # Show all running processes
process(action="poll", session_id="proc_abc123")   # Check status
process(action="wait", session_id="proc_abc123")   # Block until done
process(action="log", session_id="proc_abc123")    # Full output
process(action="kill", session_id="proc_abc123")   # Terminate
process(action="write", session_id="proc_abc123", data="y")  # Send input
```

PTY モード（`pty=true`）は、Codex や Claude Code のようなインタラクティブな CLI ツールを有効化します。

末尾の `&` だけでバックグラウンド化しているフォアグラウンドコマンド（たとえば
`python3 -m http.server 8765 &` や `cd app && npm start > app.log 2>&1 &`）は、
`&` を取り除いたうえで `background=true` として実行され、結果には `session_id` を含む
`auto_backgrounded` の注記が付きます。これは他のバックグラウンドコマンドと同じ承認チェックを通ります。
他のコマンドが混在する形（`server & curl ...`、`a; b &`、複数行のスクリプト）、
`nohup`/`disown`/`setsid` によるラッパー、コマンド置換（`$(...)`、バッククォート、`<(...)`）、
および `&` を取り除くと承認カテゴリが変わってしまうコマンドは、どの部分を実行し続けるべきかを
Hermes が安全に判断できないため、引き続きガイダンス付きで拒否されます。

### 静かなプロセスへの繰り返しの wait

まだ実行中で新しい出力を何も出していないプロセスに対して、エージェントが `process(action="wait")` または `process(action="poll")` を
2 回続けて呼び出すと、Hermes はそのプロセスの `notify_on_complete` を有効にし、結果でエージェントに確認をやめるよう伝えます:
終了（コードと最終出力）は後で独立したメッセージとして届くため、エージェントは変化のない確認のたびにモデル呼び出しを
消費する代わりに、他の作業を行ったりターンを終えたりできます。新しい出力、プロセスの終了、またはそのプロセスに対する
その他のアクションがあると、カウントはリセットされます。

これは、完了通知がエージェントに届く場面に適用されます: インタラクティブ CLI（インタラクティブな
`hermes chat -q` を含む）、TUI とデスクトップアプリ、メッセージングチャットです。非インタラクティブな `hermes chat -q`、
ワンショットの `hermes -z`、cron ジョブ、Kanban ワーカー、ステートレスな API リクエスト、委譲されたサブエージェントは、
後から完了通知を受け取れないため従来の挙動のままです。`display.background_process_notifications: off` を設定した
プロファイル（そこでは完了がエージェントのターンを開始することはありません）も同様です。完了通知が別の相手に届くプロセス
（別のセッションのプロセスや、引き継がれていないサブエージェントのプロセス）を確認する場合も、通常のポーリングのままです。
管理者が管理する設定や `${VAR}` の値も考慮されます。
どこでも通常のポーリングを維持するには、次のように設定します:

```yaml
terminal:
  process_wait_auto_notify: false
```

完了したバックグラウンドコマンドは、終了ステータスとキャプチャされた出力をアクティブなプロファイル内に
保持します。コマンドを起動した会話（またはその圧縮された続き）を再開し、元の `session_id` を使って、
出力は `process(action="log")` で、終了ステータスは `process(action="poll")` で取得してください。
無関係な会話や、所有セッションに紐付いていないリクエストは、正確なプロセスハンドルがあっても
保持されたレシートを読み取れません。`process(action="list")` には、現在のタスクまたは会話の
保持された結果も含まれます。

Hermes は最新の **64 件の完了結果**を、**完了後最大 7 日間**、プロファイルの Hermes ホームの
`logs/process-results/` 配下に保持します。各レシートには、既存のローリング方式の
**200,000 文字の出力末尾**までが含まれ、ライブ出力のリダクションが無効になっている場合でも、
ターミナルのシークレットリダクションルールが常に適用されます。レシートは、その後の結果の
読み取りまたは書き込み時に期限切れになります。復元によってコマンドが再実行されたり、完了通知が
再送されたりすることはありません。これは親が生きている間に完了した作業を保持するものであり、
タイムアウトやクラッシュの後に未完了の子プロセスを生かし続けるものではありません。

## sudo サポート {#sudo-support}

インタラクティブな親セッションでは、サポートされている sudo コマンドはマスクされたパスワードプロンプトを使用します（セッション中はキャッシュされます）。これには、リテラルの絶対パスまたはクォートされた実行ファイルパスや、通常のオプションと代入を伴う `env` プレフィックス（`env -u UNUSED /usr/bin/sudo id` など）が含まれます。パスワード不要の sudo ではプロンプトは不要です。エージェントマシン上のプロファイルの `.env` ファイルで `SUDO_PASSWORD` を設定することもできます。これはコマンドを実行するターミナルバックエンドに送信されるため、別のマシンのパスワードではなく、そのバックエンドの sudo パスワード（SSH ならリモートホスト、Docker ならコンテナユーザー）を設定してください。コンテナ、クラウドサンドボックス、リモートバックエンドで実行されるプログラムはそこでパスワードを読み取れるため、その環境でのみ有効なパスワードを使用するか、そのイメージでパスワード不要の sudo を設定してください。Hermes は、ローカル以外のバックエンドに初めて sudo パスワードを送信する際に、1 回だけ警告をログに記録します。

バックエンドの sudo が askpass ヘルパー（`sudo -A`）をサポートし、その一時ディレクトリで小さなヘルパースクリプトを実行できる場合、パスワードはそのヘルパーを通じて各 sudo 呼び出しに渡されるため、同じコマンドライン内の他のコマンドが stdin からそれを読み取ることはありません。それ以外の場合、およびすでに独自に `-S` または `-A` を渡しているコマンドでは、パスワードは stdin を通じて sudo に渡されます。

`bash -c 'sudo id'` のようなシェルペイロード、`env -S` の分割文字列、動的な実行ファイルパス、認識されない `env` オプションは、パスワード書き換え処理では解釈されません。インタラクティブなプロンプトが必要な場合は、sudo を直接呼び出してください。この処理は承認ルールや、エージェントが指定した sudo パスワードに対するガードを変更するものではありません。

委譲されたサブエージェントはパスワードプロンプトを開けません: それらの並行作業には、直列化された人間用のパスワードチャネルがないためです。代わりに親セッションでコマンドを実行するか、`SUDO_PASSWORD` をローカルで設定してください。メッセージング/ヘッドレスセッションには安全なパスワード返信チャネルがありません。チャットでパスワードを送信しないでください。

:::warning
メッセージングプラットフォームでは、sudo が失敗すると、出力に `SUDO_PASSWORD` を `~/.hermes/.env` に追加するヒントが含まれます。
:::
