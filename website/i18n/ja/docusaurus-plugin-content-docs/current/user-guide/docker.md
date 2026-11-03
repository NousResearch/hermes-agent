---
sidebar_position: 7
title: "Docker"
description: "Hermes Agent を Docker で実行し、Docker をターミナルバックエンドとして使う"
---

# Hermes Agent — Docker

Docker が Hermes Agent と交わる方法は、明確に異なる 2 つがあります:

1. **Hermes を Docker 内で実行する** — エージェント自体がコンテナ内で動作する（このページの主な焦点）
2. **Docker をターミナルバックエンドとして使う** — エージェントはホスト上で動作するが、すべてのコマンドを単一の永続的な Docker サンドボックスコンテナ内で実行する。このコンテナはツール呼び出し、`/new`、サブエージェントをまたいで、Hermes プロセスの生存期間中ずっと維持される（[設定 → Docker バックエンド](./configuration.md#docker-backend)を参照）

このページではオプション 1 を扱います。コンテナは、すべてのユーザーデータ（設定、API キー、セッション、スキル、メモリ）を、ホストから `/opt/data` にマウントされた単一のディレクトリに保存します。イメージ自体はステートレスで、設定を失うことなく新しいバージョンをプルしてアップグレードできます。

## クイックスタート

Hermes Agent を初めて実行する場合は、ホスト上にデータディレクトリを作成し、コンテナをインタラクティブに起動してセットアップウィザードを実行します:

```sh
mkdir -p ~/.hermes
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent setup
```

これでセットアップウィザードが起動し、API キーの入力を求められ、それらが `~/.hermes/.env` に書き込まれます。これは一度だけ行えば済みます。この時点で、ゲートウェイが連携するチャットシステムをセットアップしておくことを強く推奨します。

## ゲートウェイモードで実行する

設定が完了したら、コンテナを永続的なゲートウェイ（Telegram、Discord、Slack、WhatsApp など）としてバックグラウンドで実行します:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

ポート 8642 は、ゲートウェイの [OpenAI 互換 API サーバー](./features/api-server.md)とヘルスエンドポイントを公開します。チャットプラットフォーム（Telegram、Discord など）のみを使う場合はオプションですが、ダッシュボードや外部ツールからゲートウェイに到達させたい場合は必須です。

注意: API サーバーは `API_SERVER_ENABLED=true` でゲートされています。コンテナ内の `127.0.0.1` を超えて公開するには、`API_SERVER_HOST=0.0.0.0` と `API_SERVER_KEY`（最低 8 文字 — `openssl rand -hex 32` で生成）も設定します。例:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -e API_SERVER_ENABLED=true \
  -e API_SERVER_HOST=0.0.0.0 \
  -e API_SERVER_KEY=your_api_key_here \
  -e API_SERVER_CORS_ORIGINS='*' \
  nousresearch/hermes-agent gateway run
```

インターネットに面したマシンでポートを開くことはセキュリティリスクです。リスクを理解していない限り、行うべきではありません。

## ダッシュボードを実行する {#running-the-dashboard}

組み込みの Web ダッシュボードは、ゲートウェイと同じコンテナ内で、オプションのサイドプロセスとして動作します。`HERMES_DASHBOARD=1` を設定し、ゲートウェイの `8642` と並べてポート `9119` を公開します:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -p 9119:9119 \
  -e HERMES_DASHBOARD=1 \
  nousresearch/hermes-agent gateway run
```

エントリポイントは、メインコマンドを `exec` する前に、`hermes dashboard` をバックグラウンドで（非 root の `hermes` ユーザーとして）起動します。ダッシュボードの出力は `docker logs` で `[dashboard]` というプレフィックスが付くため、ゲートウェイのログと簡単に区別できます。

| 環境変数 | 説明 | デフォルト |
|---------------------|-------------|---------|
| `HERMES_DASHBOARD` | `1`（または `true` / `yes`）に設定すると、メインコマンドと並べてダッシュボードを起動 | *(未設定 — ダッシュボードは起動しない)* |
| `HERMES_DASHBOARD_HOST` | ダッシュボード HTTP サーバーのバインドアドレス | `0.0.0.0` |
| `HERMES_DASHBOARD_PORT` | ダッシュボード HTTP サーバーのポート | `9119` |
| `HERMES_DASHBOARD_TUI` | `1` に設定すると、ブラウザ内の Chat タブを公開（PTY/WebSocket 経由で `hermes --tui` を埋め込み） | *(未設定)* |

デフォルトの `HERMES_DASHBOARD_HOST=0.0.0.0` は、ホストが公開ポート経由でダッシュボードに到達するために必要です。その場合、エントリポイントは `hermes dashboard` に `--insecure` を自動的に渡します。ダッシュボードをコンテナ内アクセスのみに制限したい場合（例: サイドカー内のリバースプロキシの背後）は `127.0.0.1` に上書きしてください。

:::note
ダッシュボードのサイドプロセスは **監視されません** — クラッシュした場合、コンテナが再起動するまでダウンしたままになります。別のコンテナとして実行することはサポートされていません: ダッシュボードのゲートウェイ生存検出には、ゲートウェイプロセスとの共有 PID 名前空間が必要です。
:::

## インタラクティブに実行する（CLI チャット）

実行中のデータディレクトリに対してインタラクティブなチャットセッションを開くには:

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent
```

または、実行中のコンテナ内ですでにターミナルを開いている場合（例えば Docker Desktop 経由）は、次を実行するだけです:

```sh
/opt/hermes/.venv/bin/hermes
```

## 永続ボリューム

`/opt/data` ボリュームは、すべての Hermes 状態の単一の信頼できる情報源です。ホストの `~/.hermes/` ディレクトリにマッピングされ、以下を含みます:

| パス | 内容 |
|------|----------|
| `.env` | API キーとシークレット |
| `config.yaml` | すべての Hermes 設定 |
| `SOUL.md` | エージェントのパーソナリティ/アイデンティティ |
| `sessions/` | 会話履歴 |
| `memories/` | 永続メモリストア |
| `skills/` | インストール済みスキル |
| `cron/` | スケジュールされたジョブ定義 |
| `hooks/` | イベントフック |
| `logs/` | ランタイムログ |
| `skins/` | カスタム CLI スキン |

:::warning
同じデータディレクトリに対して 2 つの Hermes **ゲートウェイ** コンテナを同時に実行しないでください — セッションファイルとメモリストアは同時書き込みアクセス向けに設計されていません。
:::

## マルチプロファイルのサポート

Hermes は[複数のプロファイル](../reference/profile-commands.md)をサポートします — 単一のインストールから独立したエージェント（異なる SOUL、スキル、メモリ、セッション、認証情報）を実行できる、別々の `~/.hermes/` ディレクトリです。**Docker 下で実行する場合、Hermes の組み込みマルチプロファイル機能の使用は推奨されません。**

代わりに推奨されるパターンは **プロファイルごとに 1 コンテナ** で、各コンテナがそれぞれのホストディレクトリを `/opt/data` としてバインドマウントします:

```sh
# 仕事用プロファイル
docker run -d \
  --name hermes-work \
  --restart unless-stopped \
  -v ~/.hermes-work:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run

# 個人用プロファイル
docker run -d \
  --name hermes-personal \
  --restart unless-stopped \
  -v ~/.hermes-personal:/opt/data \
  -p 8643:8642 \
  nousresearch/hermes-agent gateway run
```

Docker でプロファイルよりも別々のコンテナを使う理由:

- **隔離** — 各コンテナは独自のファイルシステム、プロセステーブル、リソース制限を持ちます。あるプロファイルでのクラッシュ、依存関係の変更、暴走したセッションが、他のプロファイルに影響を与えることはありません。
- **独立したライフサイクル** — 各エージェントを個別にアップグレード、再起動、一時停止、ロールバックできます（`docker restart hermes-work` は `hermes-personal` に影響しません）。
- **クリーンなポートとネットワークの分離** — 各ゲートウェイは独自のホストポートにバインドします。チャットプラットフォームや API サーバー間のクロストークのリスクはありません。
- **よりシンプルなメンタルモデル** — コンテナ *が* プロファイルです。バックアップ、移行、権限はすべてバインドマウントされたディレクトリに従い、覚えておくべき余分な `--profile` フラグはありません。
- **同時書き込みリスクの回避** — 同じデータディレクトリに対して 2 つのゲートウェイを実行しないという上記の警告は、単一コンテナ内のプロファイルにも引き続き適用されます。

Docker Compose では、これは単に、それぞれ異なる `container_name`、`volumes`、`ports` を持つサービスをプロファイルごとに 1 つ宣言することを意味します:

```yaml
services:
  hermes-work:
    image: nousresearch/hermes-agent:latest
    container_name: hermes-work
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes-work:/opt/data

  hermes-personal:
    image: nousresearch/hermes-agent:latest
    container_name: hermes-personal
    restart: unless-stopped
    command: gateway run
    ports:
      - "8643:8642"
    volumes:
      - ~/.hermes-personal:/opt/data
```

## 環境変数の転送

API キーはコンテナ内の `/opt/data/.env` から読み取られます。環境変数を直接渡すこともできます:

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  nousresearch/hermes-agent
```

直接の `-e` フラグは `.env` の値を上書きします。これは、キーをディスク上に置きたくない CI/CD やシークレットマネージャー統合に便利です。

## Docker Compose の例

ゲートウェイとダッシュボードの両方を含む永続デプロイには、`docker-compose.yaml` が便利です:

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"   # ゲートウェイ API
      - "9119:9119"   # ダッシュボード（HERMES_DASHBOARD=1 のときのみ到達可能）
    volumes:
      - ~/.hermes:/opt/data
    environment:
      - HERMES_DASHBOARD=1
      # .env ファイルの代わりに特定の環境変数を転送する場合はコメントを外す:
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

`docker compose up -d` で起動し、`docker compose logs -f` でログを表示します。ダッシュボードの出力には `[dashboard]` というプレフィックスが付くため、ゲートウェイのログから簡単にフィルタリングできます。

## リソース制限

Hermes コンテナは中程度のリソースを必要とします。推奨される最小値:

| リソース | 最小 | 推奨 |
|----------|---------|-------------|
| メモリ | 1 GB | 2〜4 GB |
| CPU | 1 コア | 2 コア |
| ディスク（データボリューム） | 500 MB | 2 GB 以上（セッション/スキルとともに増加） |

ブラウザ自動化（Playwright/Chromium）は最もメモリを消費する機能です。ブラウザツールが不要なら 1 GB で十分です。ブラウザツールを有効にする場合は、少なくとも 2 GB を割り当ててください。

Docker で制限を設定します:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  --memory=4g --cpus=2 \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

## Dockerfile が行うこと

公式イメージは `debian:13.4` をベースにしており、以下を含みます:

- すべての Hermes 依存関係を備えた Python 3（`uv pip install -e ".[all]"`）
- Node.js + npm（ブラウザ自動化と WhatsApp ブリッジ用）
- Chromium 付きの Playwright（`npx playwright install --with-deps chromium --only-shell`）
- システムユーティリティとしての ripgrep、ffmpeg、git、tini
- **`docker-cli`** — コンテナ内で動作するエージェントが、`docker build`、`docker run`、コンテナの検査などのためにホストの Docker デーモンを操作できるようにします（`/var/run/docker.sock` をバインドマウントしてオプトイン）。
- **`openssh-client`** — コンテナ内から [SSH ターミナルバックエンド](/docs/user-guide/configuration#ssh-backend)を有効にします。SSH バックエンドはシステムの `ssh` バイナリにシェルアウトします。これがないと、コンテナ化されたインストールでは静かに失敗していました。
- WhatsApp ブリッジ（`scripts/whatsapp-bridge/`）

エントリポイントスクリプト（`docker/entrypoint.sh`）は、初回実行時にデータボリュームをブートストラップします:
- ディレクトリ構造を作成する（`sessions/`、`memories/`、`skills/` など）
- `.env` が存在しない場合は `.env.example` → `.env` をコピーする
- 不足している場合はデフォルトの `config.yaml` をコピーする
- 不足している場合はデフォルトの `SOUL.md` をコピーする
- マニフェストベースのアプローチでバンドルされたスキルを同期する（ユーザーの編集を保持）
- `HERMES_DASHBOARD=1` の場合、オプションでバックグラウンドのサイドプロセスとして `hermes dashboard` を起動する（[ダッシュボードを実行する](#running-the-dashboard)を参照）
- 続いて、渡した任意の引数とともに `hermes` を実行する

:::warning
コマンドチェーンに `/opt/hermes/docker/entrypoint.sh` を残さない限り、イメージのエントリポイントを上書きしないでください。エントリポイントは、ゲートウェイの状態ファイルが作成される前に root 権限を `hermes` ユーザーに降格します。公式イメージ内で `hermes gateway run` を root として起動することは、`/opt/data` に root 所有のファイルを残し、後のダッシュボードやゲートウェイの起動を壊す可能性があるため、デフォルトで拒否されます。そのリスクを意図的に受け入れる場合にのみ `HERMES_ALLOW_ROOT_GATEWAY=1` を設定してください。
:::

## アップグレード

最新のイメージをプルしてコンテナを再作成します。データディレクトリはそのまま維持されます。

```sh
docker pull nousresearch/hermes-agent:latest
docker rm -f hermes
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

または Docker Compose で:

```sh
docker compose pull
docker compose up -d
```

## スキルと認証情報ファイル

Docker を実行環境として使う場合（上記の方法ではなく、エージェントが Docker サンドボックス内でコマンドを実行する場合 — [設定 → Docker バックエンド](./configuration.md#docker-backend)を参照）、Hermes はすべてのツール呼び出しに対して単一の長命なコンテナを再利用し、スキルディレクトリ（`~/.hermes/skills/`）とスキルが宣言する認証情報ファイルを、読み取り専用ボリュームとしてそのコンテナに自動的にバインドマウントします。スキルのスクリプト、テンプレート、参照は、手動設定なしでサンドボックス内で利用可能になり、コンテナが Hermes プロセスの生存期間中持続するため、インストールした依存関係や書き込んだファイルは次のツール呼び出しまで残ります。

同じ同期は SSH と Modal バックエンドでも行われます — スキルと認証情報ファイルは、各コマンドの前に rsync または Modal マウント API を介してアップロードされます。

## ローカル推論サーバーへの接続（vLLM、Ollama など）

Hermes を Docker で実行し、推論サーバー（vLLM、Ollama、text-generation-inference など）もホスト上または別のコンテナで実行している場合、ネットワーキングには追加の注意が必要です。

### Docker Compose（推奨）

両方のサービスを同じ Docker ネットワークに配置します。これが最も信頼性の高いアプローチです:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --served-model-name my-model
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    networks:
      - hermes-net
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes:/opt/data
    networks:
      - hermes-net

networks:
  hermes-net:
    driver: bridge
```

次に、`~/.hermes/config.yaml` で **コンテナ名** をホスト名として使います:

```yaml
model:
  provider: custom
  model: my-model
  base_url: http://vllm:8000/v1
  api_key: "none"
```

:::tip 重要なポイント
- ホスト名には **コンテナ名**（`vllm`）を使ってください — `localhost` や `127.0.0.1` ではありません。これらは Hermes コンテナ自身を指します。
- `model` の値は、vLLM に渡した `--served-model-name` と一致する必要があります。
- `api_key` は空でない任意の文字列に設定してください（vLLM はヘッダーを要求しますが、デフォルトでは検証しません）。
- `base_url` の末尾にスラッシュを **含めない** でください。
:::

### スタンドアロンの Docker run（Compose なし）

推論サーバーがホスト上で直接（Docker 内ではなく）動作している場合、macOS/Windows では `host.docker.internal` を、Linux では `--network host` を使います:

**macOS / Windows:**

```sh
docker run -d \
  --name hermes \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

```yaml
# config.yaml
model:
  provider: custom
  model: my-model
  base_url: http://host.docker.internal:8000/v1
  api_key: "none"
```

**Linux（ホストネットワーキング）:**

```sh
docker run -d \
  --name hermes \
  --network host \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

```yaml
# config.yaml
model:
  provider: custom
  model: my-model
  base_url: http://127.0.0.1:8000/v1
  api_key: "none"
```

:::warning `--network host` を使うと `-p` フラグは無視されます — すべてのコンテナポートがホスト上に直接公開されます。
:::

### 接続性の確認

Hermes コンテナ内から、推論サーバーに到達可能であることを確認します:

```sh
docker exec hermes curl -s http://vllm:8000/v1/models
```

サーブされているモデルを列挙する JSON 応答が表示されるはずです。これが失敗する場合は、次を確認してください:

1. 両方のコンテナが同じ Docker ネットワーク上にあること（`docker network inspect hermes-net`）
2. 推論サーバーが `127.0.0.1` ではなく `0.0.0.0` でリッスンしていること
3. ポート番号が一致していること

### Ollama

Ollama も同じように動作します。Ollama がホスト上で動作している場合は、`host.docker.internal:11434`（macOS/Windows）または `127.0.0.1:11434`（`--network host` を使う Linux）を使います。Ollama が同じ Docker ネットワーク上の独自のコンテナで動作している場合:

```yaml
model:
  provider: custom
  model: llama3
  base_url: http://ollama:11434/v1
  api_key: "none"
```

## トラブルシューティング

### コンテナがすぐに終了する

ログを確認します: `docker logs hermes`。よくある原因:
- `.env` ファイルが欠落しているか無効 — まずインタラクティブに実行してセットアップを完了する
- ポートを公開して実行している場合のポート競合

### 「Permission denied」エラー

コンテナのエントリポイントは、`gosu` を介して権限を非 root の `hermes` ユーザー（UID 10000）に降格します。ホストの `~/.hermes/` が別の UID によって所有されている場合は、`HERMES_UID`/`HERMES_GID` をホストユーザーに合わせて設定するか、データディレクトリが書き込み可能であることを確認します:

```sh
chmod -R 755 ~/.hermes
```

### ブラウザツールが動作しない

Playwright は共有メモリを必要とします。Docker run コマンドに `--shm-size=1g` を追加します:

```sh
docker run -d \
  --name hermes \
  --shm-size=1g \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

### ネットワーク障害後にゲートウェイが再接続しない

`--restart unless-stopped` フラグが、ほとんどの一時的な障害を処理します。ゲートウェイがスタックしている場合は、コンテナを再起動します:

```sh
docker restart hermes
```

### コンテナの健全性を確認する

```sh
docker logs --tail 50 hermes          # 最近のログ
docker run -it --rm nousresearch/hermes-agent:latest version     # バージョンを確認
docker stats hermes                    # リソース使用状況
```
