---
sidebar_position: 3
title: "更新とアンインストール"
description: "Hermes Agent を最新バージョンに更新する方法、またはアンインストールする方法"
---

# 更新とアンインストール

## 更新

次の単一のコマンドで最新バージョンに更新できます。

```bash
hermes update
```

これにより最新のコードを取得し、依存関係を更新し、前回の更新以降に追加された新しいオプションを設定するように促されます。

:::tip
`hermes update` は新しい設定オプションを自動的に検出し、追加するように促します。そのプロンプトをスキップした場合は、手動で `hermes config check` を実行して不足しているオプションを確認し、`hermes config migrate` で対話的に追加できます。
:::

### 更新中に何が起こるか

`hermes update` を実行すると、次の手順が行われます。

1. **ペアリングデータのスナップショット** — 軽量な更新前の状態スナップショットが保存されます（`~/.hermes/pairing/`、Feishu のコメントルール、その他実行時に変更されるステートファイルを対象とします）。[スナップショットとロールバック](../user-guide/checkpoints-and-rollback.md) で説明されているスナップショット復元フローを使うか、Hermes が `~/.hermes/` ディレクトリの隣に書き込んだ最新のクイックスナップショットの zip を展開することで復元できます。
2. **Git pull** — `main` ブランチから最新のコードを取得し、サブモジュールを更新します
3. **依存関係のインストール** — `uv pip install -e ".[all]"` を実行して、新規または変更された依存関係を取り込みます
4. **設定のマイグレーション** — お使いのバージョン以降に追加された新しい設定オプションを検出し、設定するように促します
5. **ゲートウェイの自動再起動** — 更新の完了後に、実行中のゲートウェイがリフレッシュされ、新しいコードが直ちに反映されます。サービスとして管理されているゲートウェイ（Linux では systemd、macOS では launchd）は、サービスマネージャーを通じて再起動されます。手動のゲートウェイは、Hermes が実行中の PID をプロファイルに対応付けられる場合に自動的に再起動されます。

### プレビューのみ: `hermes update --check`

実際に pull する前に `origin/main` より遅れているかどうかを知りたい場合は、`hermes update --check` を実行してください。fetch を行い、ローカルのコミットと最新のリモートのコミットを並べて表示し、同期していれば `0` を、遅れていれば `1` を返して終了します。ファイルは変更されず、ゲートウェイも再起動されません。「更新があるかどうか」を判定するスクリプトや cron ジョブで役立ちます。

### 更新前の完全バックアップ: `--backup`

重要度の高いプロファイル（本番環境のゲートウェイ、チーム共有のインストール）では、pull の前に `HERMES_HOME`（設定、認証、セッション、スキル、ペアリング）の完全バックアップを取得するように選択できます。

```bash
hermes update --backup
```

または、すべての実行でこれをデフォルトにすることもできます。

```yaml
# ~/.hermes/config.yaml
updates:
  pre_update_backup: true
```

`--backup` は以前のビルドでは常に有効な動作でしたが、大きなホームでは毎回の更新に数分を追加していたため、現在はオプトインになっています。前述の軽量なペアリングデータのスナップショットは、引き続き無条件で実行されます。

想定される出力は次のようになります。

```
$ hermes update
Updating Hermes Agent...
📥 Pulling latest code...
Already up to date.  (or: Updating abc1234..def5678)
📦 Updating dependencies...
✅ Dependencies updated
🔍 Checking for new config options...
✅ Config is up to date  (or: Found 2 new options — running migration...)
🔄 Restarting gateways...
✅ Gateway restarted
✅ Hermes Agent updated successfully!
```

### 推奨される更新後の検証

`hermes update` は主要な更新パスを処理しますが、簡単な検証ですべてが問題なく反映されたことを確認できます。

1. `git status --short` — ツリーが想定外に変更されている場合は、続行する前に確認してください
2. `hermes doctor` — 設定、依存関係、サービスの状態をチェックします
3. `hermes --version` — バージョンが想定どおりに上がったことを確認します
4. ゲートウェイを使用している場合: `hermes gateway status`
5. `doctor` が npm audit の問題を報告した場合: 指摘されたディレクトリで `npm audit fix` を実行します

:::warning 更新後の作業ツリーが汚れている場合
`hermes update` の後に `git status --short` が想定外の変更を表示する場合は、続行する前に停止して確認してください。これは通常、ローカルの変更が更新されたコードの上に再適用されたか、依存関係のステップがロックファイルを更新したことを意味します。
:::

### 更新の途中でターミナルが切断された場合

`hermes update` は、ターミナルの予期しない切断から自身を保護します。

- 更新は `SIGHUP` を無視するため、SSH セッションやターミナルウィンドウを閉じても、インストールの途中で終了させてしまうことはなくなりました。`pip` と `git` の子プロセスもこの保護を継承するため、接続が切断されても Python 環境がインストール途中の状態で残ることはありません。
- 更新の実行中、すべての出力は `~/.hermes/logs/update.log` にミラーリングされます。ターミナルが消えた場合は、再接続してログを確認し、更新が完了したか、ゲートウェイの再起動が成功したかを確認できます。

```bash
tail -f ~/.hermes/logs/update.log
```

- `Ctrl-C`（SIGINT）とシステムシャットダウン（SIGTERM）は引き続き尊重されます。これらは意図的なキャンセルであり、事故ではないためです。

ターミナルの切断に耐えるために `hermes update` を `screen` や `tmux` でラップする必要はなくなりました。

### 現在のバージョンの確認

```bash
hermes version
```

[GitHub リリースページ](https://github.com/NousResearch/hermes-agent/releases) で最新リリースと比較してください。

### メッセージングプラットフォームからの更新

Telegram、Discord、Slack、WhatsApp、Teams から直接、次を送信して更新することもできます。

```
/update
```

これにより最新のコードを取得し、依存関係を更新し、実行中のゲートウェイを再起動します。再起動中、ボットは短時間オフラインになり（通常 5〜15 秒）、その後再開します。

### 手動での更新

（クイックインストーラーではなく）手動でインストールした場合は次のとおりです。

```bash
cd /path/to/hermes-agent
export VIRTUAL_ENV="$(pwd)/venv"

# Pull latest code and submodules
git pull origin main
git submodule update --init --recursive

# Reinstall (picks up new dependencies)
uv pip install -e ".[all]"
uv pip install -e "./tinker-atropos"

# Check for new config options
hermes config check
hermes config migrate   # Interactively add any missing options
```

### ロールバックの手順

更新によって問題が発生した場合は、以前のバージョンにロールバックできます。

```bash
cd /path/to/hermes-agent

# List recent versions
git log --oneline -10

# Roll back to a specific commit
git checkout <commit-hash>
git submodule update --init --recursive
uv pip install -e ".[all]"

# Restart the gateway if running
hermes gateway restart
```

特定のリリースタグにロールバックするには次のとおりです。

```bash
git checkout v0.6.0
git submodule update --init --recursive
uv pip install -e ".[all]"
```

:::warning
新しいオプションが追加されていた場合、ロールバックにより設定の非互換が発生することがあります。ロールバック後に `hermes config check` を実行し、エラーが発生した場合は `config.yaml` から認識されないオプションを削除してください。
:::

### Nix ユーザー向けの注意

Nix flake 経由でインストールした場合、更新は Nix パッケージマネージャーを通じて管理されます。

```bash
# Update the flake input
nix flake update hermes-agent

# Or rebuild with the latest
nix profile upgrade hermes-agent
```

Nix のインストールはイミュータブルであり、ロールバックは Nix の世代システムによって処理されます。

```bash
nix profile rollback
```

詳細については [Nix のセットアップ](./nix-setup.md) を参照してください。

---

## アンインストール

```bash
hermes uninstall
```

アンインストーラーでは、将来の再インストールに備えて設定ファイル（`~/.hermes/`）を残すオプションを選択できます。

### 手動でのアンインストール

```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # Optional — keep if you plan to reinstall
```

:::info
ゲートウェイをシステムサービスとしてインストールした場合は、まず停止して無効化してください。
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::
