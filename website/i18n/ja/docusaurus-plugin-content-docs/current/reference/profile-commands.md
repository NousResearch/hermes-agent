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
| `--clone` | 現在のプロファイルから `config.yaml`、`.env`、`SOUL.md` をコピーします。 |
| `--clone-all` | 現在のプロファイルからすべて（設定、メモリ、スキル、セッション、状態）をコピーします。 |
| `--clone-from <profile>` | 現在のプロファイルの代わりに、特定のプロファイルからクローンします。`--clone` または `--clone-all` と併用します。 |
| `--no-alias` | ラッパースクリプトの作成をスキップします。 |

プロファイルを作成しても、そのプロファイルディレクトリがターミナルコマンドのデフォルトのプロジェクト/ワークスペースディレクトリになる**わけではありません**。プロファイルを特定のプロジェクトで開始したい場合は、そのプロファイルの `config.yaml` で `terminal.cwd` を設定してください。

**例:**

```bash
# 空のプロファイル — 完全なセットアップが必要
hermes profile create mybot

# 現在のプロファイルから設定のみをクローン
hermes profile create work --clone

# 現在のプロファイルからすべてをクローン
hermes profile create backup --clone-all

# 特定のプロファイルから設定をクローン
hermes profile create work2 --clone --clone-from work
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
これは、すべての設定、メモリ、セッション、スキルを含むプロファイルのディレクトリ全体を完全に削除します。現在アクティブなプロファイルは削除できません。
:::

## `hermes profile show`

```bash
hermes profile show <name>
```

ホームディレクトリ、設定されたモデル、ゲートウェイのステータス、スキル数、設定ファイルの状態など、プロファイルの詳細を表示します。

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

`~/.local/bin/<name>` にあるシェルエイリアススクリプトを再生成します。エイリアスを誤って削除してしまった場合や、Hermesのインストールを移動した後に更新する必要がある場合に便利です。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<name>` | エイリアスを作成/更新するプロファイル。 |
| `--remove` | エイリアスを作成する代わりに、ラッパースクリプトを削除します。 |
| `--name <alias>` | カスタムエイリアス名（デフォルト: プロファイル名）。 |

**例:**

```bash
hermes profile alias work
# ~/.local/bin/work を作成/更新

hermes profile alias work --name mywork
# ~/.local/bin/mywork を作成

hermes profile alias work --remove
# ラッパースクリプトを削除
```

## `hermes profile rename`

```bash
hermes profile rename <old-name> <new-name>
```

プロファイル名を変更します。ディレクトリとシェルエイリアスを更新します。

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

## `hermes profile export`

```bash
hermes profile export <name> [options]
```

プロファイルを圧縮されたtar.gzアーカイブとしてエクスポートします。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<name>` | エクスポートするプロファイル。 |
| `-o`, `--output <path>` | 出力ファイルのパス（デフォルト: `<name>.tar.gz`）。 |

**例:**

```bash
hermes profile export work
# カレントディレクトリに work.tar.gz を作成

hermes profile export work -o ./work-2026-03-29.tar.gz
```

## `hermes profile import`

```bash
hermes profile import <archive> [options]
```

tar.gzアーカイブからプロファイルをインポートします。

| 引数 / オプション | 説明 |
|-------------------|-------------|
| `<archive>` | インポートするtar.gzアーカイブへのパス。 |
| `--name <name>` | インポートするプロファイルの名前（デフォルト: アーカイブから推測）。 |

**例:**

```bash
hermes profile import ./work-2026-03-29.tar.gz
# アーカイブからプロファイル名を推測

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
`hermes profile export` / `import` は、自分のマシン上でプロファイルを**ローカルにバックアップおよび復元**するための適切なコマンドです。ディストリビューション（`install` / `update` / `info`）は別の概念です。gitを介してプロファイルを配布し、他の誰かがインストールできるようにするものです。
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
# GitHubリポジトリからインストール（短縮形）
hermes profile install github.com/kyle/telemetry-distribution --alias

# 完全なHTTPS gitのURLからインストール
hermes profile install https://github.com/kyle/telemetry-distribution.git

# SSHからインストール
hermes profile install git@github.com:kyle/telemetry-distribution.git

# 開発中にローカルディレクトリからインストール
hermes profile install ./telemetry/
```

### `hermes profile update`

```bash
hermes profile update <name> [--force-config] [--yes]
```

記録されたソースからディストリビューションを再クローンし、更新を適用します。ディストリビューションが所有するファイル（SOUL.md、skills/、cron/、mcp.json）は上書きされます。ユーザーデータ（メモリ、セッション、認証、.env）には決して触れられません。

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
# 他のあらゆる `git clone` と同様に、SSHキーを使用します
hermes profile install git@github.com:your-org/internal-assistant.git

# git credential ヘルパーを使用します
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
distribution_owned:   # オプション。デフォルトは SOUL.md、config.yaml、
                      #   mcp.json、skills/、cron/、distribution.yaml
  - SOUL.md
  - skills/compliance/
  - cron/
```

`hermes_requires` は `>=`、`<=`、`==`、`!=`、`>`、`<`、またはバージョンのみ（`>=` として扱われる）をサポートします。現在のHermesバージョンが仕様を満たさない場合、インストールは明確なエラーで失敗します。

`distribution_owned` はオプションです。設定されている場合、更新時にはそれらのパスのみが置き換えられ、プロファイル内のそれ以外のものはユーザー所有のままです。省略された場合は、上記のデフォルトが適用されます。

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
# 補完をインストール
hermes completion bash >> ~/.bashrc
hermes completion zsh >> ~/.zshrc
hermes completion fish > ~/.config/fish/completions/hermes.fish

# シェルをリロード
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
