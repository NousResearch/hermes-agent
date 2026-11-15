---
sidebar_position: 7
---

# プロファイルコマンドリファレンス

このページでは、[Hermesプロファイル](../user-guide/profiles.md)に関連するすべてのコマンドを説明します。一般的なCLIコマンドについては、[CLIコマンドリファレンス](./cli-commands.md)を参照してください。

## `hermes profile`

```bash
hermes profile <subcommand>
```

プロファイルを管理するためのトップレベルコマンドです。サブコマンドなしで `hermes profile` を実行すると、ヘルプが表示されます。

| サブコマンド | 説明 |
|------------|-------------|
| `list` | すべてのプロファイルを一覧表示します。 |
| `use` | アクティブな（デフォルトの）プロファイルを設定します。 |
| `create` | 新しいプロファイルを作成します。 |
| `describe` | プロファイルの説明を読み取り、または設定します（kanban オーケストレーターがルーティングに使用します）。 |
| `delete` | プロファイルを削除します。 |
| `show` | プロファイルの詳細を表示します。 |
| `alias` | プロファイルのシェルエイリアスを再生成します。 |
| `rename` | プロファイル名を変更します。 |
| `export` | プロファイルをtar.gzアーカイブにエクスポートします。 |
| `import` | tar.gzアーカイブからプロファイルをインポートします。 |
| `install` | gitのURLまたはローカルディレクトリからプロファイルディストリビューションをインストールします。[プロファイルディストリビューション](../user-guide/profile-distributions.md)を参照してください。 |
| `update` | ディストリビューション管理されたプロファイルを再取得し、そのバンドルを再適用します。 |
| `info` | プロファイルのディストリビューションメタデータ（オリジンURL、コミット、最終更新）を表示します。 |

## `hermes profile list`

```bash
hermes profile list
```

すべてのプロファイルを一覧表示します。現在アクティブなプロファイルには `*` が付きます。

**例:**

```bash
$ hermes profile list
  default
* work
  dev
  personal
```

オプションはありません。

## `hermes profile use`

```bash
hermes profile use <name>
```

`<name>` をアクティブなプロファイルとして設定します。以降のすべての `hermes` コマンド（`-p` なし）はこのプロファイルを使用します。

| 引数 | 説明 |
|----------|-------------|
| `<name>` | アクティブにするプロファイル名。ベースプロファイルに戻すには `default` を使用します。 |

**例:**

```bash
hermes profile use work
hermes profile use default
```

## `hermes profile create`

```bash
hermes profile create <name> [options]
```

新しいプロファイルを作成します。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<name>` | 新しいプロファイルの名前。有効なディレクトリ名（英数字、ハイフン、アンダースコア）である必要があります。 |
| `--clone` | 現在のプロファイルから `config.yaml`、`.env`、`SOUL.md`、スキル、厳選された `memories/MEMORY.md` / `memories/USER.md`、およびアクティブな `memory.provider` 自身の設定（`<provider>/` または `<provider>.json`、例: `hindsight/config.json`）をコピーします。セッション、`state.db`、cron ジョブはコピーされません。 |
| `--clone-all` | 現在のプロファイルからすべて（設定、メモリ、スキル、プラグイン）をコピーします。プロファイルごとの履歴（セッション、`state.db`、バックアップ、state-snapshots、チェックポイント）と cron ジョブは除外されます。cron ジョブはソースプロファイルに紐づいたままになります（クローンが引き継ぐと、すべてのジョブが 2 回実行されてしまうためです）。ソースがデフォルトプロファイルの場合、マシンスコープのローカルモデルツリー（`models/`、`runtimes/`、`node/`）もスキップされます。これは `hermes backup` が除外するのと同じツリーです。 |
| `--clone-from <profile>` | 現在のプロファイルの代わりに、特定のプロファイルから config/skills/SOUL をクローンします。`--clone-all` と組み合わせない限り、`--clone` を暗黙的に指定します。 |
| `--no-alias` | ラッパースクリプトの作成をスキップします。 |
| `--description "<text>"` | このプロファイルが得意なことを 1〜2 文で説明します。kanban オーケストレーターが、プロファイル名だけでなく役割に基づいてタスクをルーティングするために使用します。省略して、後から `hermes profile describe` で追加することもできます。`<profile_dir>/profile.yaml` に保存されます。 |
| `--no-skills` | バンドルされたスキルが 1 つも有効になっていない**空の**プロファイルを作成します。プロファイルに `.no-bundled-skills` マーカーを書き込むため、今後の `hermes update` 実行でバンドルセットが再シードされることはありません。また、`--clone`、`--clone-from`、`--clone-all`（いずれにせよスキルをコピーしてしまう）との併用は拒否されます。範囲を絞ったオーケストレータープロファイルや、完全なスキルカタログを引き継ぐべきでないサンドボックスプロファイルに便利です。作成済みのプロファイル（デフォルトの `~/.hermes` を含む）でこれを切り替えるには、`hermes skills opt-out` / `hermes skills opt-in` を使用してください。 |

プロファイルを作成しても、そのプロファイルディレクトリがターミナルコマンドのデフォルトのプロジェクト/ワークスペースディレクトリになる**わけではありません**。プロファイルを特定のプロジェクトで開始したい場合は、そのプロファイルの `config.yaml` で `terminal.cwd` を設定してください。

**例:**

```bash
# Blank profile — needs full setup
hermes profile create mybot

# Clone config only from current profile
hermes profile create work --clone

# Clone everything from current profile
hermes profile create backup --clone-all

# Clone config from a specific profile
hermes profile create work2 --clone-from work

# Clone everything from a specific profile
hermes profile create work2-backup --clone-from work --clone-all
```

## `hermes profile describe`

```bash
hermes profile describe [<name>] [options]
```

プロファイルの説明を読み取り、または設定します。この説明は kanban オーケストレーターが利用し、プロファイル名だけから推測するのではなく、各プロファイルが得意なことに基づいてタスクをルーティングします。`<profile_dir>/profile.yaml` に保存されるため、再起動後も保持され、ゲートウェイとも共有されます。

フラグなしの場合は、現在の説明を出力します（空の場合は `(no description set for '<name>')`）。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<name>` | 説明を扱うプロファイル。`--all --auto` を使用する場合を除き必須です。 |
| `--text "<text>"` | 説明をこの正確なテキスト（ユーザー作成）に設定します。既存の説明は上書きされます。 |
| `--auto` | プロファイルにインストールされたスキル、設定されたモデル、名前に基づいて、補助 LLM で 1〜2 文の説明を自動生成します。モデルは `config.yaml` の `auxiliary.profile_describer` で設定します。自動生成された説明には `description_auto: true` が付けられ、ダッシュボードでレビュー対象として示されます。 |
| `--overwrite` | `--auto` と併用すると、ユーザーが作成した説明も置き換えます（デフォルト: 明示的に設定された説明を持つプロファイルはスキップ）。 |
| `--all` | `--auto` と併用すると、説明のないすべてのプロファイルを一括処理します。 |

**例:**

```bash
# Read the current description
hermes profile describe researcher

# Set it explicitly
hermes profile describe researcher --text "Reads source code and writes findings."

# Let the LLM generate one
hermes profile describe researcher --auto

# Fill in descriptions for every profile that doesn't have one
hermes profile describe --all --auto
```

## `hermes profile delete`

```bash
hermes profile delete <name> [options]
```

プロファイルを削除し、そのシェルエイリアスを削除します。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<name>` | 削除するプロファイル。 |
| `--yes`, `-y` | 確認プロンプトをスキップします。 |

**例:**

```bash
hermes profile delete mybot
hermes profile delete mybot --yes
```

:::warning
これは、すべての設定、メモリ、セッション、スキルを含むプロファイルのディレクトリ全体を完全に削除します。`default` プロファイル（`~/.hermes`）は削除できません。すべてを削除するには `hermes uninstall` を使用してください。
:::

## `hermes profile show`

```bash
hermes profile show <name>
```

ホームディレクトリ、設定されたモデル、ゲートウェイのステータス、スキル数、設定ファイルの状態など、プロファイルの詳細を表示します。

ここ（および `hermes profile list`）でのスキル数はその場でカウントされます。Desktop とダッシュボードのプロファイル一覧は数秒ごとにポーリングされるため、最後に判明したカウントを表示し、バックグラウンドで更新します。起動したばかりのバックエンドでは、最初のバックグラウンドカウントが届くまで、プロファイルのスキル数が一時的に `0` と表示されることがあります。また、インストールしたばかりのスキルは約 1 分以内にそれらの一覧に表示されます。

これはプロファイルのHermesホームディレクトリを表示するものであり、ターミナルの作業ディレクトリではありません。ターミナルコマンドは `terminal.cwd`（またはローカルバックエンドで `cwd: "."` の場合は起動ディレクトリ）から開始されます。

| 引数 | 説明 |
|----------|-------------|
| `<name>` | 確認するプロファイル。 |

**例:**

```bash
$ hermes profile show work
Profile: work
Path:    ~/.hermes/profiles/work
Model:   anthropic/claude-sonnet-4 (anthropic)
Gateway: stopped
Skills:  12
.env:    exists
SOUL.md: exists
Alias:   ~/.local/bin/work
```

## `hermes profile alias`

```bash
hermes profile alias <name> [options]
```

`~/.local/bin/<name>`（Windows では `<name>.bat`）にあるシェルエイリアススクリプトを再生成します。エイリアスを誤って削除してしまった場合や、Hermesのインストールを移動した後に更新する必要がある場合に便利です。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<name>` | エイリアスを作成/更新するプロファイル。 |
| `--remove` | エイリアスを作成する代わりに、ラッパースクリプトを削除します。プロファイルが削除された孤立エイリアスも削除できます: エイリアス名を渡してください（`hermes doctor` で一覧表示されます）。 |
| `--name <alias>` | カスタムエイリアス名（デフォルト: プロファイル名）。 |

**例:**

```bash
hermes profile alias work
# Creates/updates ~/.local/bin/work

hermes profile alias work --name mywork
# Creates ~/.local/bin/mywork

hermes profile alias work --remove
# Removes the wrapper script
```

## `hermes profile rename`

```bash
hermes profile rename <old-name> <new-name>
```

プロファイル名を変更します。ディレクトリとシェルエイリアスを更新します。古い名前でインストールされたゲートウェイサービス（`hermes -p <old-name> gateway install`）は、ゲートウェイが実行中かどうかにかかわらず削除されます。次回ログイン時に古い名前で起動してしまうためです。`hermes -p <new-name> gateway install` で再インストールしてください。Docker イメージ内では、s6 のゲートウェイスロットが新しい名前に移動します。

| 引数 | 説明 |
|----------|-------------|
| `<old-name>` | 現在のプロファイル名。 |
| `<new-name>` | 新しいプロファイル名。 |

**例:**

```bash
hermes profile rename mybot assistant
# ~/.hermes/profiles/mybot → ~/.hermes/profiles/assistant
# ~/.local/bin/mybot → ~/.local/bin/assistant
```

名前の変更では、プロファイルに保存されたセッション/ルーティングの ID（セッションキー（`agent:<old>:*`）、`sessions.profile_name`、ハートビート、ルーティング/配信の行）も新しい名前に移行されます。稼働中の多重化ゲートウェイがこの移行を担当する（ルーティングインデックスをメモリ上に保持しているため）ので、ゲートウェイが実行中の場合、CLI はゲートウェイに処理を委ねます。プロファイルディレクトリ内にあるワークスペースのチェックポイント（`/rollback`）履歴も新しいパスにキーが付け直されるため、名前の変更後も引き続きアクセスできます。このステップが失敗と報告された場合は、`hermes profile migrate-identity` で再試行することもできます。

## `hermes profile migrate-identity`

```bash
hermes profile migrate-identity <old-name> <new-name>
```

すでに完了した名前変更の ID 移行を再試行します。`hermes profile rename` が、稼働中のゲートウェイがセッション ID を移行できなかったと警告した場合に実行してください。ゲートウェイを再起動する（データベースからルーティングインデックスを再読み込みするため、移行が反映されます）か、停止してください。ストアを保持するゲートウェイがない場合、コマンド自身が永続的な書き換えを実行します。

移行はまだ `<old>` を指している行に基づいて行われるため、`profiles/<old>` が存在する必要はありません。チェックされるのは `<new>` のみです。冪等であり、完了済みの移行を再実行しても、付け直すキーがない状態で成功します。稼働中のゲートウェイが移行を拒否した場合、またはデータベースが書き換えを拒否した場合（ルーティングの衝突、ロック、または 2 つのデータベースの一方が失敗しもう一方が成功した場合）は、データベースとエラーを示して非ゼロで終了します。

**例:**

```bash
hermes profile rename mybot assistant
# ⚠ Profile was renamed, but the live gateway could not migrate session identity (…).
#   Restart the gateway, then run:
#     hermes profile migrate-identity mybot assistant

hermes profile migrate-identity mybot assistant
# ✓ Session/routing identity migrated: mybot → assistant
```

## `hermes profile purge-identity`

```bash
hermes profile purge-identity <name>
```

すでに完了した削除の ID パージを再試行します。`hermes profile delete` が、セッション/ルーティング ID の整理がまだ保留中であると報告した場合に実行してください。ゲートウェイを再起動する（データベースからルーティングインデックスを再読み込みするため、パージが反映されます）か、停止してください。ストアを保持するゲートウェイがない場合、コマンド自身が永続的な削除を実行します。

パージは `<name>` のみをキーとするため、プロファイルディレクトリが存在する必要はありません。ただし、その名前で再び稼働しているプロファイルは拒否されます。ID は名前で整理されるため、パージすると新しいプロファイルのルーティングまで消えてしまうからです。ルーティングキー（`agent:<name>:*`）、ハートビート行、プロファイルの Telegram トピックのバインディング/モードの行は削除されます。`delivery_obligations` の行は削除されるのではなく `abandoned` としてマークされるため、保留中の配信状態が黙って失われることはありません。セッション行はパージ自体では削除されません。パージは ID を整理するものであり、履歴を整理するものではないからです。会話の記録が削除後も残るかどうかは `hermes profile delete` によって決まり、こちらはプロファイル自身の `profiles/<name>/`（その `state.db` を含む）を削除します。冪等であり、完了済みのパージを再実行しても、パージするものがない状態で成功します。その名前が再び稼働中のプロファイルである場合、稼働中のゲートウェイがパージを拒否した場合、またはデータベースが削除を拒否した場合（ロック、または部分的な失敗）は非ゼロで終了します。

**例:**

```bash
hermes profile delete mybot
# ⚠ Profile was deleted, but the live gateway could not purge its session identity (…).
#   Restart the gateway, then run:
#     hermes profile purge-identity mybot

hermes profile purge-identity mybot
# ✓ Session/routing identity purged: mybot
```

## `hermes profile export`

```bash
hermes profile export <name> [options]
```

プロファイルを圧縮されたtar.gzアーカイブとしてエクスポートします。バックアップしたり、別のマシンに移したり、他の人に渡したりできるポータブルなスナップショットです。`auth.json`、`.env`、その他の認証情報ストアは常に除外されます。[エクスポートファイルの内容](../user-guide/profile-distributions.md#what-an-export-file-contains)を参照してください。

チャットでは [`/export`](./slash-commands.md) として、デスクトップアプリでは **⌘K → Export profile…** またはプロファイルの四角形の右クリックメニューからも利用できます。デスクトップからのエクスポートでは、さらに `desktop.json`（スキン、ライト/ダークモード、カスタムテーマ、レールの色、ウィンドウレイアウト）がアーカイブに含められます。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<name>` | エクスポートするプロファイル。 |
| `-o`, `--output <path>` | 出力ファイルのパス（デフォルト: `<name>.tar.gz`）。 |

**例:**

```bash
hermes profile export work
# Creates work.tar.gz in the current directory

hermes profile export work -o ./work-2026-03-29.tar.gz
```

アーカイブに何が含まれるか、また他の人に送る前に何を確認すべきかについては、[プロファイルファイルのエクスポートとインポート](../user-guide/profile-distributions.md#export-and-import-a-profile-file)を参照してください。

## `hermes profile import`

```bash
hermes profile import <archive> [options]
```

tar.gzアーカイブから新しいプロファイルとしてインポートします。既存のプロファイルの上書きは拒否され、`default`（組み込みのルートプロファイル）としてインポートすることもできません。いずれの場合も `--name` を渡してください。名前が既存のコマンドと衝突しない場合は、シェルラッパーが作成されます。

チャットでは [`/import`](./slash-commands.md) として、デスクトップアプリでは **⌘K → Import profile…** またはプロファイルレールの **+** の横にあるインポートボタンからも利用できます。デスクトップからのインポートでは、同梱された `desktop.json` オーバーレイ（テーマ、レイアウト）も適用され、新しいプロファイルに切り替わります。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<archive>` | インポートするtar.gzアーカイブへのパス。 |
| `--name <name>` | インポートするプロファイルの名前（デフォルト: アーカイブから推測）。 |

**例:**

```bash
hermes profile import ./work-2026-03-29.tar.gz
# Infers profile name from the archive

hermes profile import ./work-2026-03-29.tar.gz --name work-restored
```

## ディストリビューションコマンド {#distribution-commands}

:::tip
**ディストリビューションは初めてですか？** まず[プロファイルディストリビューションのユーザーガイド](../user-guide/profile-distributions.md)から始めてください。そこでは、その理由、タイミング、方法を完全な例とともに説明しています。以下のセクションは、何が必要かわかっている場合のための簡素なCLIリファレンスです。
:::

ディストリビューションは、プロファイルを**gitリポジトリ**として公開される、共有可能でバージョン管理されたアーティファクトに変換します。受信者は単一のコマンドでディストリビューションをインストールでき、後でローカルのメモリ、セッション、認証情報に触れることなく、その場で更新できます。

`auth.json` と `.env` はディストリビューションの一部になることはありません。これらはインストールするユーザーのマシンに残ります。

受信者のユーザーデータ（メモリ、セッション、認証、`.env` への独自の編集）は、初回インストールおよびそれ以降の更新を通じて常に保持されます。

:::info
プロファイルを共有する方法は 2 つあり、互いに補完し合います。`hermes profile export` / `import`（チャットでは `/export` と `/import`）は**単一のファイル**を生成します。リポジトリもマニフェストも不要で、デスクトップからのエクスポートにはテーマとレイアウトも含まれます。ディストリビューション（`install` / `update` / `info`）はプロファイルを **git リポジトリ**として公開し、受信者が後からバージョン管理された更新を取得できるようにします。バックアップと復元は、エクスポートファイルのもう 1 つの役割です。[プロファイルを共有する 2 つの方法](../user-guide/profile-distributions.md#two-ways-to-share-a-profile)を参照してください。
:::

### `hermes profile install`

```bash
hermes profile install <source> [--name <name>] [--alias] [--force] [--yes]
```

gitのURLまたはローカルディレクトリからプロファイルディストリビューションをインストールします。

| オプション | 説明 |
|--------|-------------|
| `<source>` | gitのURL（`github.com/user/repo`、`https://...`、`git@...`、`ssh://`、`git://`）、またはルートに `distribution.yaml` を含むローカルディレクトリ。 |
| `--name NAME` | マニフェストのプロファイル名を上書きします。 |
| `--alias` | シェルラッパーも作成します（例: `telemetry` → `hermes -p telemetry`）。 |
| `--force` | 同名の既存プロファイルを上書きします。ユーザーデータは引き続き保持されます。 |
| `-y`, `--yes` | マニフェストプレビューの確認プロンプトをスキップします。 |

インストーラーはマニフェストを表示し、必要な環境変数を一覧表示し、確認を求める前にcronジョブについて警告します。必要な環境変数は `.env.EXAMPLE` ファイルに記載され、それを `.env` にコピーして入力します。

**例:**

```bash
# Install from a GitHub repo (shorthand)
hermes profile install github.com/kyle/telemetry-distribution --alias

# Install from a full HTTPS git URL
hermes profile install https://github.com/kyle/telemetry-distribution.git

# Install from SSH
hermes profile install git@github.com:kyle/telemetry-distribution.git

# Install from a local directory during development
hermes profile install ./telemetry/
```

### `hermes profile update`

```bash
hermes profile update <name> [--force-config] [--yes]
```

記録されたソースからディストリビューションを再クローンし、更新を適用します。ディストリビューションが所有するファイル（SOUL.md、mcp.json）は上書きされ、ディストリビューションに同梱されたスキルと cron ジョブは置き換えられます。`skills/` や `cron/` の下に自分で追加したスキルや cron ジョブはそのまま残ります。ユーザーデータ（メモリ、セッション、認証、.env）には決して触れられません。シンボリックリンクになっている `skills/`、`cron/`、またはスキルカテゴリディレクトリは、何かが書き込まれる前に拒否されます。リンクを実際のディレクトリに置き換えてから再実行してください。

`config.yaml` はローカルのオーバーライドを保持するために、デフォルトで保持されます。`--force-config` を渡すと、ディストリビューションに同梱された設定にリセットされます。

### `hermes profile info`

```bash
hermes profile info <name>
```

プロファイルのディストリビューションマニフェスト（名前、バージョン、必要なHermesバージョン、作成者、環境変数の要件、ソースのURL/パス、およびディストリビューションが最後に `install` または `update` された際に記録された `Installed:` タイムスタンプ）を出力します。共有されたプロファイルをインストールする前に何が必要かを確認したり、「このプロファイルは6か月前にインストールされ、更新されていない」といった状況を見つけたりするのに便利です。

`hermes profile list` も `Distribution` 列にディストリビューション名とバージョンを表示し、`hermes profile show <name>` / `delete <name>` はソースURLを表示するため、どのプロファイルがgitリポジトリから来たのか、ローカルで作成されたのかを一目で判別できます。

### プライベートディストリビューション

プライベートgitリポジトリは、追加の設定なしでディストリビューションのソースとして機能します。インストールは通常の `git` バイナリにシェルアウトするため、シェルがすでにセットアップしている認証（SSHキー、`git credential` ヘルパー、GitHub CLIに保存されたHTTPS認証情報）が透過的に適用されます。

```bash
# Uses your SSH key, the same as any other `git clone`
hermes profile install git@github.com:your-org/internal-assistant.git

# Uses your git credential helper
hermes profile install https://github.com/your-org/internal-assistant.git
```

インストール中にターミナルで対話的に認証情報を求めるクローンが発生した場合、そのプロンプトはそのまま流れます。最初に同じリポジトリに対して通常の `git clone` を使用するのと同じように認証をセットアップしてから、インストールしてください。

### ディストリビューションマニフェスト（`distribution.yaml`）

すべてのディストリビューションは、リポジトリのルートに `distribution.yaml` を持っています。

```yaml
name: telemetry
version: 0.1.0
description: "Compliance monitoring harness"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "Memory graph URL"
    required: false
    default: "http://127.0.0.1:8000/sse"
distribution_owned:   # optional; defaults to SOUL.md, config.yaml,
                      #   mcp.json, skills/, cron/, distribution.yaml
  - SOUL.md
  - skills/compliance/
  - cron/
```

`hermes_requires` は `>=`、`<=`、`==`、`!=`、`>`、`<`、またはバージョンのみ（`>=` として扱われる）をサポートします。現在のHermesバージョンが仕様を満たさない場合、インストールは明確なエラーで失敗します。

`distribution_owned` はオプションです。設定されている場合、それらのパスのみが更新され、プロファイル内のそれ以外のものはユーザー所有のままです。`skills/` や `skills/research/` のようなカテゴリなど、スキルのディレクトリはスキル単位でマージされます。ディストリビューションに同梱されたスキルは置き換えられ、そこに自分で追加したスキルは残ります。作成者が後でディストリビューションから外したスキルは、トップレベルの `skills/` と同様に、更新時にもそのまま残されます。省略された場合は、上記のデフォルトが適用されます。

### ディストリビューションの公開

ディストリビューションの作成は、単なるgit pushです。

1. プロファイルディレクトリで、少なくとも `name` と `version` を含む `distribution.yaml` を作成します。
2. gitリポジトリを初期化（または既存のものを使用）し、GitHub / GitLab / Hermesがクローンできる任意のホストにpushします。
3. 受信者に `hermes profile install <your-repo-url>` を実行するよう伝えます。

バージョン管理されたリリースにはgitタグを使用してください。`HEAD` をクローンする受信者は最新の状態を取得し、マニフェストの `version:` はいつでも更新できます。

## `hermes -p` / `hermes --profile`

```bash
hermes -p <name> <command> [options]
hermes --profile <name> <command> [options]
```

任意のHermesコマンドを特定のプロファイルで実行するためのグローバルフラグです。固定されたデフォルトを変更しません。これは、コマンドの実行中だけアクティブなプロファイルを上書きします。

| オプション | 説明 |
|--------|-------------|
| `-p <name>`, `--profile <name>` | このコマンドで使用するプロファイル。 |

**例:**

```bash
hermes -p work chat -q "Check the server status"
hermes --profile dev gateway start
hermes -p personal skills list
hermes -p work config edit
```

## `hermes completion`

```bash
hermes completion <shell>
```

シェル補完スクリプトを生成します。プロファイル名とプロファイルサブコマンドの補完が含まれます。

| 引数 | 説明 |
|----------|-------------|
| `<shell>` | 補完を生成するシェル: `bash`、`zsh`、または `fish`。 |

**例:**

```bash
# Install completions
hermes completion bash >> ~/.bashrc
hermes completion zsh >> ~/.zshrc
hermes completion fish > ~/.config/fish/completions/hermes.fish

# Reload shell
source ~/.bashrc
```

インストール後、以下に対してタブ補完が機能します。
- `hermes profile <TAB>` — サブコマンド（list、use、create など）
- `hermes profile use <TAB>` — プロファイル名
- `hermes -p <TAB>` — プロファイル名

## 関連項目

- [プロファイルユーザーガイド](../user-guide/profiles.md)
- [CLIコマンドリファレンス](./cli-commands.md)
- [FAQ — プロファイルのセクション](./faq.md#profiles)
