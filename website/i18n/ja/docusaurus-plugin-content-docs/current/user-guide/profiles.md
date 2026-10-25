---
sidebar_position: 2
---

# プロファイル: 複数のエージェントを実行する

同じマシン上で複数の独立した Hermes エージェントを実行できます。それぞれが独自の設定、APIキー、メモリ、セッション、スキル、ゲートウェイ状態を持ちます。

## プロファイルとは何ですか？

プロファイルは独立した Hermes ホームディレクトリです。各プロファイルは、独自の `config.yaml`、`.env`、`SOUL.md`、メモリ、セッション、スキル、cronジョブ、状態データベースを含む専用のディレクトリを持ちます。プロファイルを使うと、コーディングアシスタント、個人用ボット、リサーチエージェントなど、目的の異なる複数のエージェントを Hermes の状態を混在させることなく実行できます。

プロファイルを作成すると、自動的にそれ自体が1つのコマンドになります。`coder` というプロファイルを作成すれば、すぐに `coder chat`、`coder setup`、`coder gateway start` などが使えるようになります。

## クイックスタート

```bash
hermes profile create coder       # プロファイル + "coder" コマンドのエイリアスを作成
coder setup                       # APIキーとモデルを設定
coder chat                        # チャットを開始
```

これだけです。`coder` は、独自の設定、メモリ、状態を持つ Hermes プロファイルになりました。

## プロファイルを作成する

### 空のプロファイル

```bash
hermes profile create mybot
```

バンドルされたスキルをシードした新しいプロファイルを作成します。`mybot setup` を実行して、APIキー、モデル、ゲートウェイトークンを設定してください。

### 設定のみをクローン（`--clone`）

```bash
hermes profile create work --clone
```

現在のプロファイルの `config.yaml`、`.env`、`SOUL.md` を新しいプロファイルにコピーします。APIキーとモデルは同じですが、セッションとメモリは新規になります。異なるAPIキーを使うには `~/.hermes/profiles/work/.env` を、異なるパーソナリティにするには `~/.hermes/profiles/work/SOUL.md` を編集してください。

### すべてをクローン（`--clone-all`）

```bash
hermes profile create backup --clone-all
```

**すべて** をコピーします。設定、APIキー、パーソナリティ、すべてのメモリ、完全なセッション履歴、スキル、cronジョブ、プラグイン。完全なスナップショットです。バックアップや、すでにコンテキストを持つエージェントをフォークするのに便利です。

### 特定のプロファイルからクローンする

```bash
hermes profile create work --clone --clone-from coder
```

:::tip Honcho メモリ + プロファイル
Honcho が有効な場合、`--clone` は新しいプロファイル専用のAIピアを自動的に作成しつつ、同じユーザーワークスペースを共有します。各プロファイルは独自の観察結果とアイデンティティを構築します。詳細は [Honcho -- マルチエージェント / プロファイル](./features/memory-providers.md#honcho) を参照してください。
:::

## プロファイルを使用する

### コマンドエイリアス

すべてのプロファイルには、`~/.local/bin/<name>` にコマンドエイリアスが自動的に作成されます。

```bash
coder chat                    # coder エージェントとチャット
coder setup                   # coder の設定を構成
coder gateway start           # coder のゲートウェイを起動
coder doctor                  # coder のヘルスをチェック
coder skills list             # coder のスキルを一覧表示
coder config set model.default anthropic/claude-sonnet-4
```

このエイリアスはすべての hermes サブコマンドで機能します。内部的には単に `hermes -p <name>` です。

### `-p` フラグ

どのコマンドでも、プロファイルを明示的に指定することもできます。

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # どの位置でも機能します
```

### 既定値を固定（`hermes profile use`）

```bash
hermes profile use coder
hermes chat                   # これで coder が対象になります
hermes tools                  # coder のツールを設定します
hermes profile use default    # 元に戻します
```

既定値を設定し、素の `hermes` コマンドがそのプロファイルを対象とするようにします。`kubectl config use-context` のようなものです。

### 現在地を把握する

CLIは常にどのプロファイルがアクティブかを表示します。

- **プロンプト**: `❯` の代わりに `coder ❯`
- **バナー**: 起動時に `Profile: coder` を表示
- **`hermes profile`**: 現在のプロファイル名、パス、モデル、ゲートウェイの状態を表示

## プロファイル vs ワークスペース vs サンドボックス

プロファイルはワークスペースやサンドボックスとよく混同されますが、別物です。

- **プロファイル** は Hermes に独自の状態ディレクトリを与えます。`config.yaml`、`.env`、`SOUL.md`、セッション、メモリ、ログ、cronジョブ、ゲートウェイ状態です。
- **ワークスペース** または **作業ディレクトリ** は、ターミナルコマンドが開始される場所です。これは `terminal.cwd` によって別途制御されます。
- **サンドボックス** はファイルシステムへのアクセスを制限するものです。プロファイルはエージェントを **サンドボックス化しません**。

デフォルトの `local` ターミナルバックエンドでは、エージェントはあなたのユーザーアカウントと同じファイルシステムアクセスを持ちます。プロファイルは、プロファイルディレクトリ外のフォルダへのアクセスを止めるものではありません。

プロファイルを特定のプロジェクトフォルダで開始させたい場合は、そのプロファイルの `config.yaml` に明示的な絶対パスの `terminal.cwd` を設定してください。

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

ローカルバックエンドで `cwd: "."` を使用すると、「プロファイルディレクトリ」ではなく「Hermes が起動されたディレクトリ」を意味します。

また、次の点に注意してください。

- `SOUL.md` はモデルを導くことができますが、ワークスペースの境界を強制するものではありません。
- `SOUL.md` への変更は新しいセッションでクリーンに反映されます。既存のセッションは古いプロンプト状態を使用し続けている可能性があります。
- モデルに「どのディレクトリにいますか？」と尋ねることは、信頼できる隔離テストではありません。ツールに予測可能な開始ディレクトリが必要な場合は、`terminal.cwd` を明示的に設定してください。

## ゲートウェイを実行する

各プロファイルは、独自のボットトークンを持つ別個のプロセスとして独自のゲートウェイを実行します。

```bash
coder gateway start           # coder のゲートウェイを起動
assistant gateway start       # assistant のゲートウェイを起動（別プロセス）
```

### 異なるボットトークン

各プロファイルには独自の `.env` ファイルがあります。それぞれに異なる Telegram/Discord/Slack のボットトークンを設定してください。

```bash
# coder のトークンを編集
nano ~/.hermes/profiles/coder/.env

# assistant のトークンを編集
nano ~/.hermes/profiles/assistant/.env
```

### 安全機構: トークンロック

2つのプロファイルが誤って同じボットトークンを使用した場合、2つ目のゲートウェイは、競合するプロファイル名を明示する明確なエラーでブロックされます。Telegram、Discord、Slack、WhatsApp、Signal でサポートされています。

### 永続サービス

```bash
coder gateway install         # hermes-gateway-coder systemd/launchd サービスを作成
assistant gateway install     # hermes-gateway-assistant サービスを作成
```

各プロファイルは独自のサービス名を持ちます。それらは独立して実行されます。

## プロファイルを設定する

各プロファイルは、それぞれ独自の以下を持ちます。

- **`config.yaml`** — モデル、プロバイダー、ツールセット、すべての設定
- **`.env`** — APIキー、ボットトークン
- **`SOUL.md`** — パーソナリティと指示

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

このプロファイルをデフォルトで特定のプロジェクトで動作させたい場合は、独自の `terminal.cwd` も設定してください。

```bash
coder config set terminal.cwd /absolute/path/to/project
```

## アップデート

`hermes update` はコードを一度だけ（共有で）取得し、新しいバンドルスキルを **すべての** プロファイルに自動的に同期します。

```bash
hermes update
# → Code updated (12 commits)
# → Skills synced: default (up to date), coder (+2 new), assistant (+2 new)
```

ユーザーが変更したスキルは決して上書きされません。

## プロファイルを管理する

```bash
hermes profile list           # すべてのプロファイルを状態とともに表示
hermes profile show coder     # 1つのプロファイルの詳細情報
hermes profile rename coder dev-bot   # 名前を変更（エイリアス + サービスを更新）
hermes profile export coder   # coder.tar.gz にエクスポート
hermes profile import coder.tar.gz   # アーカイブからインポート
```

## プロファイルを削除する

```bash
hermes profile delete coder
```

これはゲートウェイを停止し、systemd/launchd サービスを削除し、コマンドエイリアスを削除し、すべてのプロファイルデータを削除します。確認のためにプロファイル名の入力を求められます。

確認をスキップするには `--yes` を使用します: `hermes profile delete coder --yes`

:::note
デフォルトプロファイル（`~/.hermes`）は削除できません。すべてを削除するには `hermes uninstall` を使用してください。
:::

## タブ補完

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

補完を永続化するには、この行を `~/.bashrc` または `~/.zshrc` に追加してください。`-p` の後のプロファイル名、プロファイルのサブコマンド、トップレベルのコマンドを補完します。

## 仕組み

プロファイルは `HERMES_HOME` 環境変数を使用します。`coder chat` を実行すると、ラッパースクリプトが hermes を起動する前に `HERMES_HOME=~/.hermes/profiles/coder` を設定します。コードベース内の119以上のファイルが `get_hermes_home()` を介してパスを解決するため、Hermes の状態は自動的にプロファイルのディレクトリにスコープされます。設定、セッション、メモリ、スキル、状態データベース、ゲートウェイPID、ログ、cronジョブがそうです。

これはターミナルの作業ディレクトリとは別物です。ツールの実行は `terminal.cwd`（ローカルバックエンドで `cwd: "."` の場合は起動ディレクトリ）から始まり、`HERMES_HOME` から自動的に始まるわけではありません。

デフォルトプロファイルは単に `~/.hermes` そのものです。移行は不要で、既存のインストールは同一に動作します。

## プロファイルをディストリビューションとして共有する

あるマシンで構築したプロファイルは、**gitリポジトリ** としてパッケージ化し、別のマシン（自分のワークステーション、チームメイトのラップトップ、コミュニティユーザーの環境）に1つのコマンドでインストールできます。共有パッケージには、SOUL、設定、スキル、cronジョブ、MCP接続が含まれます。認証情報、メモリ、セッションはマシンごとに保持されます。

```bash
# gitリポジトリからエージェント全体をインストール
hermes profile install github.com/you/research-bot --alias

# 作者が新バージョンをリリースしたら後で更新（メモリ + .env は保持）
hermes profile update research-bot
```

オーサリング、公開、更新セマンティクス、セキュリティモデル、ユースケースの完全なガイドは **[プロファイルディストリビューション: エージェント全体を共有する](./profile-distributions.md)** を参照してください。
