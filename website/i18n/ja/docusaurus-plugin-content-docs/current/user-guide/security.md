---
sidebar_position: 8
title: "セキュリティ"
description: "セキュリティモデル、危険なコマンドの承認、ユーザー認可、コンテナ分離、本番デプロイのベストプラクティス"
---

# セキュリティ

Hermes Agentは、多層防御（defense-in-depth）のセキュリティモデルで設計されています。このページでは、コマンド承認からコンテナ分離、メッセージングプラットフォームでのユーザー認可まで、あらゆるセキュリティ境界を解説します。

## 概要

セキュリティモデルは7つのレイヤーで構成されています:

1. **ユーザー認可** — 誰がエージェントと会話できるか（許可リスト、DMペアリング）
2. **危険なコマンドの承認** — 破壊的な操作に対する人間の介在（human-in-the-loop）
3. **コンテナ分離** — 強化された設定によるDocker/Singularity/Modalのサンドボックス化
4. **MCP認証情報のフィルタリング** — MCPサブプロセスの環境変数の分離
5. **コンテキストファイルのスキャン** — プロジェクトファイル内のプロンプトインジェクション検出
6. **セッション間の分離** — セッションは互いのデータや状態にアクセスできません。cronジョブのストレージパスはパストラバーサル攻撃に対して強化されています
7. **入力のサニタイズ** — ターミナルツールのバックエンドにおける作業ディレクトリパラメータは、シェルインジェクションを防ぐため許可リストに対して検証されます

## 危険なコマンドの承認

コマンドを実行する前に、Hermesはそれを厳選された危険なパターンのリストと照合します。一致が見つかった場合、ユーザーは明示的にそれを承認する必要があります。

### 承認モード

承認システムは、`~/.hermes/config.yaml` の `approvals.mode` で設定する3つのモードをサポートします:

```yaml
approvals:
  mode: manual    # manual | smart | off
  timeout: 60     # ユーザーの応答を待つ秒数（デフォルト: 60）
```

| Mode | Behavior |
|------|----------|
| **manual**（デフォルト） | 危険なコマンドに対して常にユーザーに承認を求めます |
| **smart** | 補助LLMを使ってリスクを評価します。低リスクのコマンド（例: `python -c "print('hello')"`）は自動承認されます。本当に危険なコマンドは自動拒否されます。不確実なケースは手動プロンプトにエスカレーションされます。 |
| **off** | すべての承認チェックを無効化 — `--yolo` での実行と同等です。すべてのコマンドがプロンプトなしで実行されます。 |

:::warning
`approvals.mode: off` を設定すると、すべての安全プロンプトが無効になります。信頼できる環境（CI/CD、コンテナなど）でのみ使用してください。
:::

### YOLOモード

YOLOモードは、現在のセッションで**すべての**危険なコマンド承認プロンプトをバイパスします。次の3つの方法で有効化できます:

1. **CLIフラグ**: `hermes --yolo` または `hermes chat --yolo` でセッションを開始
2. **スラッシュコマンド**: セッション中に `/yolo` と入力してオン/オフを切り替え
3. **環境変数**: `HERMES_YOLO_MODE=1` を設定

`/yolo` コマンドは**トグル**です — 使用するたびにモードのオン/オフが切り替わります:

```
> /yolo
  ⚡ YOLO mode ON — all commands auto-approved. Use with caution.

> /yolo
  ⚠ YOLO mode OFF — dangerous commands will require approval.
```

YOLOモードはCLIとゲートウェイの両方のセッションで利用できます。内部的には `HERMES_YOLO_MODE` 環境変数を設定し、これがすべてのコマンド実行の前にチェックされます。

:::danger
YOLOモードは、セッションの**すべての**危険なコマンド安全チェックを無効化します — ただしハードラインのブロックリスト（後述）は**例外**です。生成されるコマンドを完全に信頼できる場合（例: 使い捨て環境での十分にテストされた自動化スクリプト）にのみ使用してください。
:::

### ハードラインのブロックリスト（常時有効の最低ライン）

一部のコマンドは、取り返しのつかないファイルシステムの消去、フォーク爆弾、ブロックデバイスへの直接書き込みなど、あまりに壊滅的であるため、Hermesは次の条件に**関係なく**実行を拒否します:

- `--yolo` / `/yolo` がオン
- `approvals.mode: off`
- cronジョブがヘッドレスの `approve` モードで実行中
- ユーザーが明示的に「常に許可」をクリック

ブロックリストは `--yolo` の下にある最低ラインです。承認レイヤーがコマンドを目にする**前に**作動し、上書きフラグはありません。現在カバーされているパターン（網羅的ではなく、`tools/approval.py::UNRECOVERABLE_BLOCKLIST` と同期されています）:

| Pattern | Why it's hardline |
|---|---|
| `rm -rf /` および明白な変種 | ファイルシステムのルートを消去 |
| `rm -rf --no-preserve-root /` | 「はい、ルートのつもりです」という明示的な変種 |
| `:(){ :\|:& };:`（bashフォーク爆弾） | 再起動するまでホストを占有 |
| マウントされたルートデバイスへの `mkfs.*` | 稼働中のシステムをフォーマット |
| `dd if=/dev/zero of=/dev/sd*` | 物理ディスクをゼロ埋め |
| rootfsトップレベルで信頼できないURLを `sh` にパイプ | リモートコード実行の攻撃ベクトルであり、承認するには範囲が広すぎる |

ブロックリストに該当した場合、ツール呼び出しはエージェントに説明的なエラーを返し、何も実行されません。正当なワークフローでこれらのコマンドのいずれかが必要な場合（例: あなたがワイプ＆再インストールのパイプラインのオペレーターである場合）は、エージェントの外で実行してください。

### 承認タイムアウト

危険なコマンドのプロンプトが表示されると、ユーザーは設定可能な時間内に応答する必要があります。タイムアウト内に応答がない場合、コマンドはデフォルトで**拒否**されます（フェイルクローズ）。

`~/.hermes/config.yaml` でタイムアウトを設定します:

```yaml
approvals:
  timeout: 60  # 秒（デフォルト: 60）
```

### 承認をトリガーするもの

次のパターンが承認プロンプトをトリガーします（`tools/approval.py` で定義）:

| Pattern | Description |
|---------|-------------|
| `rm -r` / `rm --recursive` | 再帰的な削除 |
| `rm ... /` | ルートパスでの削除 |
| `chmod 777/666` / `o+w` / `a+w` | 全員/他者が書き込み可能な権限 |
| 安全でない権限での `chmod --recursive` | 再帰的に全員/他者が書き込み可能（ロングフラグ） |
| `chown -R root` / `chown --recursive root` | rootへの再帰的なchown |
| `mkfs` | ファイルシステムのフォーマット |
| `dd if=` | ディスクコピー |
| `> /dev/sd` | ブロックデバイスへの書き込み |
| `DROP TABLE/DATABASE` | SQLのDROP |
| `DELETE FROM`（WHEREなし） | WHEREのないSQLのDELETE |
| `TRUNCATE TABLE` | SQLのTRUNCATE |
| `> /etc/` | システム設定の上書き |
| `systemctl stop/restart/disable/mask` | システムサービスの停止/再起動/無効化 |
| `kill -9 -1` | すべてのプロセスの強制終了 |
| `pkill -9` | プロセスの強制終了 |
| フォーク爆弾のパターン | フォーク爆弾 |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | `-c` フラグによるシェルコマンド実行（`-lc` のような結合フラグを含む） |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | `-e`/`-c` フラグによるスクリプト実行 |
| `curl ... \| sh` / `wget ... \| sh` | リモートコンテンツをシェルにパイプ |
| `bash <(curl ...)` / `sh <(wget ...)` | プロセス置換によるリモートスクリプトの実行 |
| `/etc/`、`~/.ssh/`、`~/.hermes/.env` への `tee` | teeによる機密ファイルの上書き |
| `/etc/`、`~/.ssh/`、`~/.hermes/.env` への `>` / `>>` | リダイレクトによる機密ファイルの上書き |
| `xargs rm` | rmを伴うxargs |
| `find -exec rm` / `find -delete` | 破壊的なアクションを伴うfind |
| `/etc/` への `cp`/`mv`/`install` | システム設定へのファイルのコピー/移動 |
| `/etc/` に対する `sed -i` / `sed --in-place` | システム設定のインプレース編集 |
| hermes/gateway に対する `pkill`/`killall` | 自己終了の防止 |
| `&`/`disown`/`nohup`/`setsid` を伴う `gateway run` | サービスマネージャーの外でのゲートウェイ起動を防止 |

:::info
**コンテナでのバイパス**: `docker`、`singularity`、`modal`、`daytona`、`vercel_sandbox` バックエンドで実行している場合、コンテナ自体がセキュリティ境界となるため、危険なコマンドのチェックは**スキップ**されます。コンテナ内の破壊的なコマンドはホストに害を与えられません。
:::

### 承認フロー（CLI）

対話型CLIでは、危険なコマンドはインラインの承認プロンプトを表示します:

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

4つの選択肢:

- **once** — この1回の実行のみを許可
- **session** — このパターンをセッションの残りの間許可
- **always** — 永続的な許可リストに追加（`config.yaml` に保存）
- **deny**（デフォルト） — コマンドをブロック

### 承認フロー（ゲートウェイ/メッセージング）

メッセージングプラットフォームでは、エージェントが危険なコマンドの詳細をチャットに送信し、ユーザーの返信を待ちます:

- 承認するには **yes**、**y**、**approve**、**ok**、または **go** と返信
- 拒否するには **no**、**n**、**deny**、または **cancel** と返信

ゲートウェイを実行すると、`HERMES_EXEC_ASK=1` 環境変数が自動的に設定されます。

### 永続的な許可リスト

「always」で承認されたコマンドは `~/.hermes/config.yaml` に保存されます:

```yaml
# 永続的に許可された危険なコマンドのパターン
command_allowlist:
  - rm
  - systemctl
```

これらのパターンは起動時に読み込まれ、以降のすべてのセッションで暗黙的に承認されます。

:::tip
`hermes config edit` を使って、永続的な許可リストからパターンを確認または削除してください。
:::

## ユーザー認可（ゲートウェイ）

メッセージングゲートウェイを実行する際、Hermesは階層化された認可システムを通じて、誰がボットと対話できるかを制御します。

### 認可チェックの順序

`_is_user_authorized()` メソッドは次の順序でチェックします:

1. **プラットフォームごとのall-allowフラグ**（例: `DISCORD_ALLOW_ALL_USERS=true`）
2. **DMペアリングの承認済みリスト**（ペアリングコードで承認されたユーザー）
3. **プラットフォーム固有の許可リスト**（例: `TELEGRAM_ALLOWED_USERS=12345,67890`）
4. **グローバル許可リスト**（`GATEWAY_ALLOWED_USERS=12345,67890`）
5. **グローバルall-allow**（`GATEWAY_ALLOW_ALL_USERS=true`）
6. **デフォルト: 拒否**

### プラットフォームの許可リスト

許可するユーザーIDを `~/.hermes/.env` にカンマ区切りの値として設定します:

```bash
# プラットフォーム固有の許可リスト
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# プラットフォーム横断の許可リスト（すべてのプラットフォームでチェック）
GATEWAY_ALLOWED_USERS=123456789

# プラットフォームごとのall-allow（慎重に使用）
DISCORD_ALLOW_ALL_USERS=true

# グローバルall-allow（細心の注意を払って使用）
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
**許可リストが何も設定されておらず**、`GATEWAY_ALLOW_ALL_USERS` も設定されていない場合、**すべてのユーザーが拒否されます**。ゲートウェイは起動時に警告をログ出力します:

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DMペアリングシステム

より柔軟な認可のため、Hermesにはコードベースのペアリングシステムが含まれています。事前にユーザーIDを要求する代わりに、未知のユーザーはワンタイムのペアリングコードを受け取り、ボットの所有者がCLIで承認します。

**仕組み:**

1. 未知のユーザーがボットにDMを送信
2. ボットが8文字のペアリングコードで返信
3. ボットの所有者がCLIで `hermes pairing approve <platform> <code>` を実行
4. そのユーザーはそのプラットフォームで永続的に承認される

`~/.hermes/config.yaml` で、認可されていないダイレクトメッセージの扱い方を制御します:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` がデフォルトです。認可されていないDMはペアリングコードの返信を受け取ります。
- `ignore` は認可されていないDMを黙って破棄します。
- プラットフォームセクションはグローバルなデフォルトを上書きするため、Telegramではペアリングを有効にしつつ、WhatsAppでは無音を保つことができます。

**セキュリティ機能**（OWASP + NIST SP 800-63-4 のガイダンスに基づく）:

| Feature | Details |
|---------|---------|
| コード形式 | 32文字の曖昧さのないアルファベット（0/O/1/Iなし）から8文字 |
| ランダム性 | 暗号学的（`secrets.choice()`） |
| コードのTTL | 1時間で失効 |
| レート制限 | ユーザーあたり10分に1リクエスト |
| 保留中の上限 | プラットフォームあたり最大3つの保留中コード |
| ロックアウト | 承認の失敗5回 → 1時間のロックアウト |
| ファイルのセキュリティ | すべてのペアリングデータファイルに `chmod 0600` |
| ロギング | コードがstdoutにログ出力されることはありません |

**ペアリングのCLIコマンド:**

```bash
# 保留中および承認済みのユーザーを一覧表示
hermes pairing list

# ペアリングコードを承認
hermes pairing approve telegram ABC12DEF

# ユーザーのアクセスを取り消し
hermes pairing revoke telegram 123456789

# すべての保留中コードをクリア
hermes pairing clear-pending
```

**ストレージ:** ペアリングデータは `~/.hermes/pairing/` にプラットフォームごとのJSONファイルで保存されます:
- `{platform}-pending.json` — 保留中のペアリングリクエスト
- `{platform}-approved.json` — 承認済みユーザー
- `_rate_limits.json` — レート制限とロックアウトの追跡

## コンテナ分離

`docker` ターミナルバックエンドを使用する場合、Hermesはすべてのコンテナに厳格なセキュリティ強化を適用します。

### Dockerのセキュリティフラグ

すべてのコンテナは次のフラグで実行されます（`tools/environments/docker.py` で定義）:

```python
_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # すべてのLinux capabilityを削除
    "--cap-add", "DAC_OVERRIDE",                  # rootがバインドマウントされたディレクトリに書き込めるように
    "--cap-add", "CHOWN",                         # パッケージマネージャーがファイル所有権を必要とする
    "--cap-add", "FOWNER",                        # パッケージマネージャーがファイル所有権を必要とする
    "--security-opt", "no-new-privileges",         # 権限昇格をブロック
    "--pids-limit", "256",                         # プロセス数を制限
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # サイズ制限された /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # exec不可の /var/tmp
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",   # exec不可の /run
]
```

### リソース制限

コンテナのリソースは `~/.hermes/config.yaml` で設定できます:

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # 明示的な許可リストのみ。空のままにすると機密情報はコンテナの外に保たれる
  container_cpu: 1        # CPUコア数
  container_memory: 5120  # MB（デフォルト 5GB）
  container_disk: 51200   # MB（デフォルト 50GB、XFS上のoverlay2が必要）
  container_persistent: true  # セッションをまたいでファイルシステムを永続化
```

### ファイルシステムの永続化

- **永続モード**（`container_persistent: true`）: `~/.hermes/sandboxes/docker/<task_id>/` から `/workspace` と `/root` をバインドマウント
- **エフェメラルモード**（`container_persistent: false`）: ワークスペースにtmpfsを使用 — クリーンアップ時にすべて失われる

:::tip
本番のゲートウェイデプロイでは、`docker`、`modal`、`daytona`、`vercel_sandbox` バックエンドを使用して、エージェントのコマンドをホストシステムから分離してください。これにより、危険なコマンドの承認が完全に不要になります。
:::

:::warning
`terminal.docker_forward_env` に名前を追加すると、それらの変数はターミナルコマンドのために意図的にコンテナに注入されます。これは `GITHUB_TOKEN` のようなタスク固有の認証情報に便利ですが、コンテナ内で実行されるコードがそれらを読み取って外部に持ち出せることも意味します。
:::

## ターミナルバックエンドのセキュリティ比較

| Backend | Isolation | Dangerous Cmd Check | Best For |
|---------|-----------|-------------------|----------|
| **local** | なし — ホスト上で実行 | ✅ あり | 開発、信頼できるユーザー |
| **ssh** | リモートマシン | ✅ あり | 別サーバーでの実行 |
| **docker** | コンテナ | ❌ スキップ（コンテナが境界） | 本番ゲートウェイ |
| **singularity** | コンテナ | ❌ スキップ | HPC環境 |
| **modal** | クラウドサンドボックス | ❌ スキップ | スケーラブルなクラウド分離 |
| **daytona** | クラウドサンドボックス | ❌ スキップ | 永続的なクラウドワークスペース |
| **vercel_sandbox** | クラウドmicroVM | ❌ スキップ | スナップショット永続化を伴うクラウド実行 |

## 環境変数のパススルー {#environment-variable-passthrough}

`execute_code` と `terminal` はどちらも、LLMが生成したコードによる認証情報の持ち出しを防ぐため、子プロセスから機密性の高い環境変数を取り除きます。ただし、`required_environment_variables` を宣言するスキルは、正当にそれらの変数へのアクセスを必要とします。

### 仕組み

2つのメカニズムにより、特定の変数がサンドボックスのフィルターを通過できます:

**1. スキルスコープのパススルー（自動）**

スキルが（`skill_view` または `/skill` コマンドで）読み込まれ、`required_environment_variables` を宣言している場合、実際に環境に設定されているそれらの変数はすべて自動的にパススルーとして登録されます。欠落している変数（まだセットアップが必要な状態のもの）は登録され**ません**。

```yaml
# スキルのSKILL.md frontmatter内
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
```

このスキルを読み込むと、`TENOR_API_KEY` は `execute_code`、`terminal`（ローカル）、**そしてリモートバックエンド（Docker、Modal）**にパススルーされます — 手動の設定は不要です。

:::info Docker と Modal
v0.5.1より前は、Dockerの `forward_env` はスキルのパススルーとは別のシステムでした。現在はこれらが統合されています — スキルが宣言した環境変数は、`docker_forward_env` に手動で追加することなく、自動的にDockerコンテナとModalサンドボックスに転送されます。
:::

**2. 設定ベースのパススルー（手動）**

どのスキルでも宣言されていない環境変数については、`config.yaml` の `terminal.env_passthrough` に追加します:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

### 認証情報ファイルのパススルー（OAuthトークンなど） {#credential-file-passthrough}

一部のスキルは、環境変数だけでなく**ファイル**をサンドボックス内に必要とします — 例えば、Google Workspaceはアクティブなプロファイルの `HERMES_HOME` の下に `google_token.json` としてOAuthトークンを保存します。スキルはこれらをfrontmatterで宣言します:

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

読み込み時、Hermesはこれらのファイルがアクティブなプロファイルの `HERMES_HOME` に存在するかをチェックし、マウント用に登録します:

- **Docker**: 読み取り専用のバインドマウント（`-v host:container:ro`）
- **Modal**: サンドボックス作成時にマウント + 各コマンドの前に同期（セッション途中のOAuthセットアップに対応）
- **Local**: 不要（ファイルは既にアクセス可能）

`config.yaml` で認証情報ファイルを手動で列挙することもできます:

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

パスは `~/.hermes/` からの相対パスです。ファイルはコンテナ内の `/root/.hermes/` にマウントされます。

### 各サンドボックスがフィルターするもの

| Sandbox | Default Filter | Passthrough Override |
|---------|---------------|---------------------|
| **execute_code** | 名前に `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL`、`PASSWD`、`AUTH` を含む変数をブロック。安全なプレフィックスの変数のみ通過 | ✅ パススルー変数は両方のチェックをバイパス |
| **terminal**（ローカル） | 明示的なHermesインフラ変数（プロバイダーキー、ゲートウェイトークン、ツールAPIキー）をブロック | ✅ パススルー変数はブロックリストをバイパス |
| **terminal**（Docker） | デフォルトではホストの環境変数なし | ✅ パススルー変数 + `docker_forward_env` が `-e` で転送される |
| **terminal**（Modal） | デフォルトではホストの環境変数/ファイルなし | ✅ 認証情報ファイルがマウントされる。環境変数のパススルーは同期経由 |
| **MCP** | 安全なシステム変数 + 明示的に設定された `env` を除くすべてをブロック | ❌ パススルーの影響を受けない（代わりにMCPの `env` 設定を使用） |

### セキュリティ上の考慮事項

- パススルーは、あなたまたはあなたのスキルが明示的に宣言した変数にのみ影響します — 任意のLLM生成コードに対するデフォルトのセキュリティ姿勢は変わりません
- 認証情報ファイルはDockerコンテナに**読み取り専用**でマウントされます
- Skills Guardは、インストール前にスキルのコンテンツを疑わしい環境変数アクセスのパターンについてスキャンします
- 欠落/未設定の変数は決して登録されません（存在しないものは漏洩しようがありません）
- Hermesのインフラの機密情報（プロバイダーAPIキー、ゲートウェイトークン）は決して `env_passthrough` に追加すべきではありません — これらには専用のメカニズムがあります

## MCP認証情報の取り扱い

MCP（Model Context Protocol）サーバーのサブプロセスは、認証情報の偶発的な漏洩を防ぐため、**フィルタリングされた環境**を受け取ります。

### 安全な環境変数

これらの変数のみがホストからMCPのstdioサブプロセスに渡されます:

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

加えて任意の `XDG_*` 変数も渡されます。それ以外のすべての環境変数（APIキー、トークン、機密情報）は**取り除かれます**。

MCPサーバーの `env` 設定で明示的に定義された変数は渡されます:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # これのみが渡される
```

### 認証情報の伏せ字

MCPツールからのエラーメッセージは、LLMに返される前にサニタイズされます。次のパターンは `[REDACTED]` に置き換えられます:

- GitHub PAT（`ghp_...`）
- OpenAI形式のキー（`sk-...`）
- Bearerトークン
- `token=`、`key=`、`API_KEY=`、`password=`、`secret=` パラメータ

### ウェブサイトアクセスポリシー

エージェントがWebおよびブラウザツールを通じてアクセスできるウェブサイトを制限できます。これは、エージェントが内部サービス、管理パネル、その他の機密URLにアクセスするのを防ぐのに役立ちます。

```yaml
# ~/.hermes/config.yaml 内
security:
  website_blocklist:
    enabled: true
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

ブロックされたURLがリクエストされると、ツールはそのドメインがポリシーによってブロックされていることを説明するエラーを返します。ブロックリストは `web_search`、`web_extract`、`browser_navigate`、およびすべてのURL対応ツールにわたって適用されます。

詳細は設定ガイドの [ウェブサイトブロックリスト](/docs/user-guide/configuration#website-blocklist) を参照してください。

### SSRF保護

すべてのURL対応ツール（Web検索、Web抽出、ビジョン、ブラウザ）は、サーバーサイドリクエストフォージェリ（SSRF）攻撃を防ぐため、フェッチする前にURLを検証します。ブロックされるアドレスには次が含まれます:

- **プライベートネットワーク**（RFC 1918）: `10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`
- **ループバック**: `127.0.0.0/8`、`::1`
- **リンクローカル**: `169.254.0.0/16`（`169.254.169.254` のクラウドメタデータを含む）
- **CGNAT / 共有アドレス空間**（RFC 6598）: `100.64.0.0/10`（Tailscale、WireGuard VPN）
- **クラウドメタデータのホスト名**: `metadata.google.internal`、`metadata.goog`
- **予約済み、マルチキャスト、未指定のアドレス**

SSRF保護はインターネットに面した利用では常に有効で、DNSの失敗はブロックとして扱われます（フェイルクローズ）。リダイレクトチェーンは、リダイレクトベースのバイパスを防ぐため、各ホップで再検証されます。

#### 意図的にプライベートURLを許可する

一部の設定は、正当にプライベート/内部URLアクセスを必要とします — `home.arpa` をRFC 1918空間に解決するホームネットワーク、LANのみのOllama/llama.cppエンドポイント、内部wiki、クラウドメタデータのデバッグなどです。そうしたケースのために、グローバルなオプトアウトがあります:

```yaml
security:
  allow_private_urls: true   # デフォルト: false
```

オンにすると、Webツール、ブラウザ、ビジョンのURLフェッチ、ゲートウェイのメディアダウンロードは、RFC 1918 / ループバック / リンクローカル / CGNAT / クラウドメタデータの宛先を拒否しなくなります。**これは意図的な信頼境界です** — ローカルネットワークに対して任意のプロンプトインジェクションされたURLをエージェントが実行することが許容できるリスクであるマシンでのみ有効にしてください。公開向けのゲートウェイではオフのままにすべきです。

ホストの部分文字列ガード（基盤となるIPが公開されている場合でも、紛らわしいUnicodeドメインのトリックをブロックする）は、この設定に関係なくオンのままです。

### Tirithによる実行前セキュリティスキャン

Hermesは、実行前のコンテンツレベルのコマンドスキャンのために [tirith](https://github.com/sheeki03/tirith) を統合しています。Tirithは、パターンマッチングだけでは見逃す脅威を検出します:

- ホモグラフURLのなりすまし（国際化ドメイン攻撃）
- インタープリターへのパイプパターン（`curl | bash`、`wget | sh`）
- ターミナルインジェクション攻撃

Tirithは初回使用時に、SHA-256チェックサム検証（およびcosignが利用可能な場合はcosignの来歴検証）を伴って、GitHubリリースから自動インストールされます。

```yaml
# ~/.hermes/config.yaml 内
security:
  tirith_enabled: true       # tirithスキャンの有効/無効（デフォルト: true）
  tirith_path: "tirith"      # tirithバイナリへのパス（デフォルト: PATHから検索）
  tirith_timeout: 5          # サブプロセスのタイムアウト（秒）
  tirith_fail_open: true     # tirithが利用できないときに実行を許可（デフォルト: true）
```

`tirith_fail_open` が `true`（デフォルト）の場合、tirithがインストールされていないかタイムアウトしても、コマンドは続行されます。高セキュリティ環境では `false` に設定して、tirithが利用できないときにコマンドをブロックしてください。

Tirithの判定は承認フローと統合されます: 安全なコマンドは通過し、疑わしいコマンドとブロックされたコマンドは両方とも、tirithの完全な検出結果（深刻度、タイトル、説明、より安全な代替案）を伴うユーザー承認をトリガーします。ユーザーは承認または拒否できます — デフォルトの選択肢は拒否で、無人のシナリオを安全に保ちます。

### コンテキストファイルのインジェクション保護

コンテキストファイル（AGENTS.md、.cursorrules、SOUL.md）は、システムプロンプトに含められる前にプロンプトインジェクションについてスキャンされます。スキャナーは次をチェックします:

- 以前の指示を無視/無効化する指示
- 疑わしいキーワードを含む隠されたHTMLコメント
- 機密情報（`.env`、`credentials`、`.netrc`）を読み取ろうとする試み
- `curl` による認証情報の持ち出し
- 不可視のUnicode文字（ゼロ幅スペース、双方向オーバーライド）

ブロックされたファイルは警告を表示します:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## 本番デプロイのベストプラクティス

### ゲートウェイデプロイのチェックリスト

1. **明示的な許可リストを設定する** — 本番環境では決して `GATEWAY_ALLOW_ALL_USERS=true` を使わない
2. **コンテナバックエンドを使う** — config.yamlで `terminal.backend: docker` を設定
3. **リソース制限を厳しくする** — 適切なCPU、メモリ、ディスクの制限を設定
4. **機密情報を安全に保存する** — APIキーを適切なファイル権限で `~/.hermes/.env` に保管
5. **DMペアリングを有効にする** — 可能な限りユーザーIDをハードコードする代わりにペアリングコードを使う
6. **コマンド許可リストを見直す** — config.yamlの `command_allowlist` を定期的に監査
7. **`MESSAGING_CWD` を設定する** — エージェントを機密ディレクトリから操作させない
8. **非rootで実行する** — 決してゲートウェイをrootで実行しない
9. **ログを監視する** — `~/.hermes/logs/` で不正アクセスの試みを確認
10. **最新に保つ** — セキュリティパッチのため `hermes update` を定期的に実行

### APIキーの保護

```bash
# .env ファイルに適切な権限を設定
chmod 600 ~/.hermes/.env

# 異なるサービスには別々のキーを保持
# .env ファイルをバージョン管理にコミットしない
```

### ネットワーク分離

最大限のセキュリティのため、ゲートウェイを別のマシンやVMで実行してください。`config.yaml` で `terminal.backend: ssh` を設定し、`~/.hermes/.env` の環境変数でホストの詳細を指定します:

```yaml
# ~/.hermes/config.yaml
terminal:
  backend: ssh
```

```bash
# ~/.hermes/.env
TERMINAL_SSH_HOST=agent-worker.local
TERMINAL_SSH_USER=hermes
TERMINAL_SSH_KEY=~/.ssh/hermes_agent_key
```

SSH接続の詳細は（`config.yaml` ではなく）`.env` に置かれるため、プロファイルのエクスポートと一緒にチェックインや共有がされません。これにより、ゲートウェイのメッセージング接続をエージェントのコマンド実行から分離して保ちます。
