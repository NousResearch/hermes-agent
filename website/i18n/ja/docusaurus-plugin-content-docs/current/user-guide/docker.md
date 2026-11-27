---
sidebar_position: 7
title: "Hermes Docker セットアップ"
description: "Hermes Agent を Docker で実行し、Docker をターミナルバックエンドとして使う"
---

# Hermes Docker セットアップ

Docker が Hermes Agent と交わる方法は、明確に異なる 2 つがあります:

1. **Hermes を Docker 内で実行する** — エージェント自体がコンテナ内で動作する（このページの主な焦点）
2. **Docker をターミナルバックエンドとして使う** — エージェントはホスト上で動作するが、すべてのコマンドを単一の永続的な Docker サンドボックスコンテナ内で実行する。このコンテナはツール呼び出し、`/new`、サブエージェントをまたいで、Hermes プロセスの生存期間中ずっと維持される（[設定 → Docker バックエンド](./configuration.md#docker-backend)を参照）

このページではオプション 1 を扱います。コンテナは、すべてのユーザーデータ（設定、API キー、セッション、スキル、メモリ）を、ホストから `/opt/data` にマウントされた単一のディレクトリに保存します。イメージ自体はステートレスで、設定を失うことなく新しいバージョンをプルしてアップグレードできます。

## イメージのチャネルとランタイムの所有権

| イメージタグ | 意味 |
|---|---|
| `latest` / `stable` | 安定版リリースのゲートで承認されたイメージ。 |
| `main` | main ブランチのビルドから公開される開発用イメージ。 |
| `X.Y.Z` | バージョン付きの安定版イメージ。デプロイを正確に固定するにはダイジェストを使ってください。 |

ワークフローは amd64 と arm64 のイメージをビルドしてテストします。安定版の公開では、
再ビルドするのではなく、テスト済みのイメージアーカイブを使います。`stable` と `latest` を動かすのは
完全なリリースのプロモーションだけで、main へのプッシュではこれらのタグは進みません。

イメージの Python 環境は `pyproject.toml` と `uv.lock`（現在は
Python 3.14）に従います。キュレーションされた extra は、ネイティブのデスクトップバンドルの
`--all-extras` セットとは異なります。Electron のデスクトップアプリは含まれません。

## クイックスタート

Hermes Agent を初めて実行する場合は、ホスト上にデータディレクトリを作成し、コンテナをインタラクティブに起動してセットアップウィザードを実行します:

:::caution インストールコマンドにブラウザベースの VPS コンソールを使わない
一部の VPS プロバイダー（Hetzner Cloud など）は、ホスト管理用のブラウザベースの
コンソールを提供しています。これらのコンソールは特殊文字を
正しく送信しません — `:` が `;` として届いたり、`@` が誤って表示されたりし、英語以外の
キーボードレイアウトではさらに悪化します — そのため、`-v ~/.hermes:/opt/data`、`-e KEY=value`、
貼り付けた API キー / トークンなどの `docker run` の引数が静かに壊れてしまいます。

コピー&ペーストしても安全にコマンドを入力するには、**代わりに SSH で接続してください**（`ssh root@<host>`）。
どうしてもブラウザコンソールを使う必要がある場合は、貼り付けずにコマンドを手で入力し、
Enter を押す前に結果のすべての `:`、`@`、`=`、`/` を再確認してください。
:::

```sh
mkdir -p ~/.hermes
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent setup
```

これでセットアップウィザードが起動し、API キーの入力を求められ、それらが `~/.hermes/.env` に書き込まれます。これは一度だけ行えば済みます。この時点で、ゲートウェイが連携するチャットシステムをセットアップしておくことを強く推奨します。

:::tip
コンテナ内で一度 `hermes setup --portal` を実行してください — リフレッシュトークンはマウントされた `~/.hermes` ボリュームに永続化されます。[Nous Portal](/docs/integrations/nous-portal) を参照してください。
:::

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

:::tip ゲートウェイは監視下で実行されます
公式 Docker イメージ内では、`gateway run` は **s6-overlay によって自動的に監視されます**: ゲートウェイプロセスがクラッシュしても、コンテナを失うことなく数秒以内に再起動され、ダッシュボード（`HERMES_DASHBOARD=1` が設定されている場合）も並んで監視されます。`gateway run` の CMD プロセス自体は、s6 が実際のゲートウェイプロセスを管理している間コンテナを生かし続ける `sleep infinity` のハートビートです — そのため `docker stop` は引き続きすべてをクリーンにシャットダウンしますが、`docker logs` には監視下のゲートウェイの出力が表示されます。

`docker logs` には、このアップグレードを確認する 1 行の目印が表示されます。オプトアウトして、従来の「ゲートウェイがコンテナのメインプロセスであり、コンテナの終了 = ゲートウェイの終了」というセマンティクスにするには、`--no-supervise` を渡すか `HERMES_GATEWAY_NO_SUPERVISE=1` を設定します。このオプトアウトは、コンテナをゲートウェイのステータスコードで終了させたい CI のスモークテストに便利です。本番デプロイでは、監視下のデフォルトの方が明確に優れています。

この挙動は s6 ベースのイメージにのみ適用されます。以前の（tini ベースの）イメージでは、`gateway run` は引き続きフォアグラウンドのメインプロセスとして実行されます。
:::

:::note ゲートウェイのログの出力先
完全なルーティングマップ（プロファイルごとのゲートウェイ、ダッシュボード、ブート時のリコンサイラー、コンテナ全体の `docker logs`）については、下記の[ログの出力先](#where-the-logs-go)セクションを参照してください。
:::

TUI は、メッセージングプラットフォームが設定されていない場合も含め、この同じ監視下のゲートウェイをローカルバックエンドとして使います。インタラクティブな `--tui` の起動は、登録された s6 ゲートウェイを自動的に開始し、準備ができるまで待機します。通常の名前付きプロファイルは、それらを提供するルートゲートウェイにアタッチします。`gateway.standalone: true` が明示的に設定されたプロファイルは、独自の登録済みスロットを使います。クライアントの起動時に、サービスをインストールしたり、管理されていないプロセスで s6 をバイパスしたりすることはありません。

:::note 無人のゲートウェイでのツールループの強制停止
無人のゲートウェイと cron のセッションでは、`non_interactive_hard_stop_enabled` によってツールループの強制停止がデフォルトで有効になります。インタラクティブな CLI、TUI、Desktop、ACP のセッションは警告のみのままです。無人のデプロイでオプトアウトするには、プロファイルの `config.yaml` で:

```yaml
tool_loop_guardrails:
  non_interactive_hard_stop_enabled: false
```
:::

注意: API サーバーは `API_SERVER_ENABLED=true` でゲートされています。コンテナ内の `127.0.0.1` を超えて公開するには、`API_SERVER_HOST=0.0.0.0` と `API_SERVER_KEY`（最低 8 文字 — `openssl rand -hex 32` で生成）も設定します。例:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -e API_SERVER_ENABLED=true \
  -e API_SERVER_HOST=0.0.0.0 \
  -e API_SERVER_KEY="$(openssl rand -hex 32)" \
  -e API_SERVER_CORS_ORIGINS='*' \
  nousresearch/hermes-agent gateway run
```

インターネットに面したマシンでポートを開くことはセキュリティリスクです。リスクを理解していない限り、行うべきではありません。

## ダッシュボードを実行する {#running-the-dashboard}

組み込みの Web ダッシュボードは、同じコンテナ内でゲートウェイと並んで、監視下の s6-rc サービスとして動作します。起動するには `HERMES_DASHBOARD=1` を設定します:

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

ダッシュボードは s6 によって監視されます — クラッシュした場合、`s6-supervise` が短いバックオフの後に自動的に再起動します。ダッシュボードの stdout/stderr は `docker logs <container>` に転送されます（プレフィックスなし。ゲートウェイ自身の出力は現在プロファイルごとの s6-log ファイルにあるため — 下記の[ログの出力先](#where-the-logs-go)を参照 — 2 つのストリームが衝突することはありません）。

| 環境変数 | 説明 | デフォルト |
|---------------------|-------------|---------|
| `HERMES_DASHBOARD` | `1`（または `true` / `yes`）に設定すると、監視下のダッシュボードサービスを有効化 | *(未設定 — サービスは登録されるがダウンしたまま)* |
| `HERMES_DASHBOARD_HOST` | ダッシュボード HTTP サーバーのバインドアドレス | `0.0.0.0` |
| `HERMES_DASHBOARD_PORT` | ダッシュボード HTTP サーバーのポート | `9119` |
| `HERMES_DASHBOARD_INSECURE` | **非推奨 / 何もしない。** 以前は認証ゲートをバイパスしていましたが、2026 年 6 月の強化以降、認証を無効にすることはなくなりました。非ループバックのバインドには常に認証プロバイダーが必要です | *(無視される — 代わりにプロバイダーを設定してください)* |

コンテナ内のダッシュボードは、デフォルトで `0.0.0.0` にバインドします — そうしないと、公開した `-p 9119:9119` のポートにホストから到達できません。バインドをコンテナのループバックに制限するには（サイドカー / リバースプロキシの構成向け）、`HERMES_DASHBOARD_HOST=127.0.0.1` を設定します。

ダッシュボードの認証ゲートは、次の両方が真のときに自動的に有効になります:

1. バインドホストが非ループバックである（例: コンテナ内のデフォルトの `0.0.0.0`）、**かつ**
2. `DashboardAuthProvider` プラグインが登録されている。

2 つ目の条件を満たす同梱の方法は 3 つあります:

- **ユーザー名/パスワード** — 信頼できるネットワーク上や VPN の背後にある、セルフホスト / オンプレミス / ホームラボのコンテナにとって最も簡単な方法です: `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`（再起動後もセッションを維持するには `HERMES_DASHBOARD_BASIC_AUTH_SECRET` も）を設定します。公開インターネットへの直接の公開には適していません。
- **OAuth（Nous Portal）** — ホスト型/公開デプロイ向け: `HERMES_DASHBOARD_OAUTH_CLIENT_ID` が設定されると、`dashboard_auth/nous` プロバイダーが有効になります。
- **セルフホストの OIDC** — 標準の OpenID Connect を介して自分の ID プロバイダーで認証する場合: `HERMES_DASHBOARD_OIDC_ISSUER` + `HERMES_DASHBOARD_OIDC_CLIENT_ID` が設定されると、`dashboard_auth/self_hosted` プロバイダーが有効になります。

どれを選んでも、ゲートは保護されたルートに到達する前に呼び出し元をログインページにリダイレクトします。3 つのプロバイダーすべてについては [Web ダッシュボード → 認証](features/web-dashboard.md#authentication-gated-mode) を参照してください。

Traefik や nginx などのリバースプロキシが別のコンテナで動作している場合、その
ブリッジネットワークのアドレスはデフォルトでは信頼されません。マウントした `config.yaml` で、ダッシュボードの公開
URL を設定し、そのプロキシの正確な IP、または専用のプロキシネットワーク用の範囲を限定した CIDR だけを
信頼してください:

```yaml
dashboard:
  public_url: "https://dashboard.example.com"
  trusted_proxies:
    - "172.20.0.5"
    # Or, if the proxy address is dynamic on a dedicated network:
    # - "172.20.0.0/24"
```

これにより、プロキシの `X-Forwarded-Proto: https` がセキュアな OAuth の
Cookie を制御できるようになり、他のピアからの転送ヘッダーは信頼されないままになります。
`*`、`0.0.0.0/0`、`::/0` は使わないでください。Hermes はこれらの範囲のないエントリを拒否します。

プロバイダーが登録されておらず、バインドが非ループバックの場合、ダッシュボードは、欠けている環境変数を示す具体的なエラーとともに **起動時にフェイルクローズします**。公開バインドで認証なしにダッシュボードを提供する逃げ道はもうありません: `HERMES_DASHBOARD_INSECURE=1` は現在、非推奨の何もしない設定です（警告をログに出力して無視されます）。プロバイダーを設定するか、`HERMES_DASHBOARD_HOST=127.0.0.1` にバインドして、代わりに SSH トンネル / Tailscale 経由でダッシュボードにアクセスしてください。

:::warning `--insecure` が削除された理由
認証なしの公開ダッシュボードは、2026 年 6 月の MCP 設定の永続化キャンペーンの入り口でした: インターネットのスキャナーが公開されたダッシュボード（および OpenAI API サーバー）に到達し、エージェントに SSH キーのバックドアを仕掛けさせました。認証ゲートは現在、すべての非ループバックのバインドで必須です。信頼できる LAN / ホームラボのマシンでは、同梱のユーザー名/パスワードのプロバイダー（`HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `_PASSWORD`）が、インフラ不要でこれを満たす方法です。
:::

ダッシュボードを別のコンテナとして実行することは、そのコンテナがホストの PID とネットワーク名前空間を共有している場合（例: リポジトリ自身の `docker-compose.yml` が行っているように `network_mode: host` — その `dashboard` サービスを参照）は **サポートされます**。ゲートウェイの生存検出にはゲートウェイプロセスとの共有 PID 名前空間が必要なため、この制限が当てはまるのは、共有 PID 名前空間のない分離されたブリッジネットワークのコンテナで実行されるダッシュボードだけです。

## Webapp を実行する {#running-the-webapp}

イメージには [Webapp](/docs/user-guide/desktop)（ブラウザで動く Hermes Desktop）がビルド済みで同梱されています。マウントした `config.yaml` でそれを有効にすると、ゲートウェイが独自のポートでそれを提供します（[設定 → Webapp](./configuration.md#webapp) を参照）:

```yaml
webapp:
  enabled: true
  host: 0.0.0.0
  port: 9120
  public_url: "http://localhost:9120"  # the address your browser opens
```

コンテナ内では `host: 0.0.0.0` のままにしてください。デフォルトの `127.0.0.1` はコンテナ自身のループバックであり、公開したポートからも別のコンテナのプロキシからも到達できません。コンテナ内の `hermes webapp --status` は引き続き `Serving at http://127.0.0.1:9120` と報告しますが、それはそこで提供しているからです。

ダッシュボードと同様に、ポートを公開してサインインのプロバイダーを設定します:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 9120:9120 \
  -e HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin \
  -e HERMES_DASHBOARD_BASIC_AUTH_PASSWORD="$(openssl rand -hex 16)" \
  nousresearch/hermes-agent gateway run
```

非ループバックの `host` には[ダッシュボードの認証プロバイダー](#running-the-dashboard)のいずれかが必要で、それがないとリスナーは起動を拒否します。`public_url` がない場合、コンテナ内で `hermes webapp` が表示するリンクは、公開されたアドレスではなくコンテナ自身のアドレスを使います。

### トンネルやリバースプロキシの背後で {#webapp-behind-a-proxy}

Cloudflare Tunnel、Caddy、Traefik、nginx が別のコンテナで動作し、独自の HTTPS アドレスで Webapp を提供する場合は、次の 3 つすべてを設定します:

```yaml
webapp:
  enabled: true
  host: 0.0.0.0                              # reachable from the proxy's container
  port: 9120
  public_url: "https://hermes.example.com"   # the address the browser opens
dashboard:
  trusted_proxies:
    - "172.30.0.0/24"                        # a network only the proxy and Hermes are on
```

- **`host: 0.0.0.0`**: デフォルトの `127.0.0.1` ではプロキシの接続が拒否され、Cloudflare はそれを **502 Bad Gateway** として報告します。
- **`public_url`**: プロキシが TLS を終端してプレーンな HTTP を転送するため、これがないと Webapp はサインインのコールバックを `http://hermes.example.com/auth/callback` として組み立て、サインインのプロバイダーに登録された `https://` のコールバックと一致しません。
- **`dashboard.trusted_proxies`**: Webapp のリスナーはダッシュボードと同じプロキシの一覧を読み、`X-Forwarded-*` ヘッダーをループバックとその一覧のアドレスからのみ受け取ります。プロキシが信頼されていれば、その `X-Forwarded-Proto: https` が Hermes に届き、サインインの Cookie に `Secure` が設定されます。プロキシの正確な IP、またはプロキシと Hermes だけが共有するネットワーク用の範囲を限定した CIDR を列挙してください。Hermes は `*`、`0.0.0.0/0`、`::/0` を拒否します。

Cloudflare Tunnel では `cloudflared` が外向きに接続するため、ポートの公開は不要です。Hermes とネットワークを共有するだけで済みます。`trusted_proxies` 用に固定のサブネットを使った Compose の概略:

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent
    command: gateway run
    restart: unless-stopped
    volumes:
      - ~/.hermes:/opt/data
    networks: [tunnel]
  cloudflared:
    image: cloudflare/cloudflared
    command: tunnel --no-autoupdate run
    environment:
      - TUNNEL_TOKEN=${TUNNEL_TOKEN}
    restart: unless-stopped
    networks: [tunnel]
networks:
  tunnel:
    ipam:
      config:
        - subnet: 172.30.0.0/24
```

トンネルの公開ホスト名には、サービス `http://hermes:9120` を指定してください。

### ダッシュボードと並べて {#webapp-next-to-the-dashboard}

ダッシュボード（`HERMES_DASHBOARD=1`、ポート 9119）と Webapp（ポート 9120）は、それぞれ独自のホスト名で 1 つのコンテナ内で実行できます: `dashboard.public_url` はダッシュボードのアドレス、`webapp.public_url` は Webapp のアドレスで、どちらも相手のものは読みません。両者はサインインのプロバイダーを共有するため、上記のユーザー名/パスワード、Nous、OIDC の設定で両方にサインインできます。

ユーザー名とパスワードは、そのまま両方で機能します。Nous のサインインにはもう 1 つ手順が必要です: Nous Portal のサインインクライアントは 1 つのコールバックアドレスで登録され、Webapp はダッシュボードと同じ `HERMES_DASHBOARD_OAUTH_CLIENT_ID`（または `dashboard.oauth.client_id`）でサインインします。そのため、1 つのクライアントは 2 つのホスト名のうち 1 つにしか対応できません。

- **Webapp のみで Nous サインイン:** Portal の [Local Dashboards](https://portal.nousresearch.com/local-dashboards) ページ、または `hermes dashboard register --redirect-uri` で、Webapp のコールバック `https://<webapp host>/auth/callback` を使ってクライアントを登録します（[ダッシュボードの登録](features/web-dashboard.md#registering-a-dashboard)を参照）。このインストールにすでにダッシュボードのクライアントがある場合、`hermes dashboard register` はそのクライアントをその場で更新するためダッシュボードのコールバックが移動し、また Webapp のアドレスを `HERMES_DASHBOARD_PUBLIC_URL` に書き込みます。
- **両方で Nous サインイン:** ネイティブの Desktop の **Settings → Web Access** は、Webapp 専用のクライアント（`webapp/sign-in-registration.json` に保存）を登録し、ダッシュボードのものには手を付けません。ただし同じマシン上の Desktop アプリからしか機能しないため、ヘッドレスのコンテナにはまだそのためのコマンドがありません。それができるまでは、一方のホスト名用に Nous クライアントを登録し、もう一方ではユーザー名とパスワードでサインインしてください。どちらのログインページにも両方のプロバイダーが表示されます。Nous サインインは、クライアントが登録されていないホスト名では失敗します。

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

## 永続ボリューム {#persistent-volumes}

`/opt/data` ボリュームは、すべての Hermes 状態の単一の信頼できる情報源です。ホストの `~/.hermes/` ディレクトリにマッピングされ、以下を含みます:

| パス | 内容 |
|------|----------|
| `.env` | API キーとシークレット |
| `config.yaml` | すべての Hermes 設定 |
| `SOUL.md` | エージェントのパーソナリティ/アイデンティティ |
| `sessions/` | 会話履歴 |
| `memories/` | 永続メモリストア |
| `skills/` | インストール済みスキル |
| `home/` | Hermes のツールのサブプロセス（`git`、`ssh`、`gh`、`npm`、スキルの CLI）用の、プロファイルごとの HOME |
| `cron/` | スケジュールされたジョブ定義 |
| `hooks/` | イベントフック |
| `logs/` | ランタイムログ |
| `skins/` | カスタム CLI スキン |

### コンテナ内の `state.db` に対するファイルシステムの要件

Hermes はセッションを SQLite データベース（`/opt/data/state.db`）に保存し、デフォルトでは WAL ジャーナルモードで開きます。WAL は、ファイルを開いているすべてのプロセス間で共有メモリ（`state.db-shm`）の一貫性が保たれていることに依存します。VM の境界をまたぐバインドマウントはそれを提供しません: **virtiofs**（macOS の Docker Desktop と Podman、OrbStack）と **9p / drvfs**（Windows の Docker Desktop）はどちらも、メインファイルが `PRAGMA integrity_check` に合格したまま、並行する書き手が WAL データベースを静かに破損させてしまいます。

Hermes の対処（v2026.9.14 以降）:

- ディレクトリが virtiofs/9p のマウント上にある **新規** データベースは、ロールバック（`DELETE`）ジャーナルモードで作成され、1 回限りの警告がログに出力されます。何もする必要はありません。
- そのようなマウント上の **既存の** WAL データベースが稼働中にダウングレードされることはありません — 他の Hermes プロセスがそれを開いている可能性があり、稼働中の切り替えはそれらのチェックポイントされていないコミットを破壊します。代わりに、すべてのプロセスが起動時に 1 回限りのエラーをログに出力し、`hermes doctor` がデータベースにフラグを立てます。次の 2 つの方法のいずれかで修正してください:
  1. そのデータベースを使うすべての Hermes プロセスを停止し、イメージに同梱されている Python（`sqlite3` シェルはありません）で 1 回限りのオフライン変換を実行します: `docker exec hermes python3 -c "import sqlite3; print(sqlite3.connect('/opt/data/state.db').execute('PRAGMA journal_mode=DELETE').fetchone()[0])"`。後で開いたときに WAL に戻らないよう、`config.yaml` で `database.journal_mode: delete` を設定してください。
  2. データディレクトリをネイティブなボリュームに移します — 名前付きの Docker ボリューム（`-v hermes-data:/opt/data`）は VM 自身の ext4 ファイルシステム上にあり、WAL を通常どおりサポートします。

検出はコンテナ内の `/proc/self/mountinfo` を読むため、ホストのオペレーティングシステムに関係なく機能します。NFS、CIFS/SMB、sshfs のマウントは別に検出されます: WAL の共有メモリのロックはすべてネットワークの往復になり、並行する接続がそこで停滞するため、そこでの新規データベースは、`database.journal_mode` が設定されていない限り（WAL を強制するには `wal` を設定）ロールバック（`DELETE`）ジャーナルモードで始まります。そこにある既存の WAL データベースは WAL のままで、起動時に `hermes sessions set-journal-mode delete` という対処法を示す 1 回限りの警告をログに出力し、`hermes doctor` によってフラグが立てられます。一般的な FUSE マウントは分類されません。それらの上では `database.journal_mode: delete` を明示的に設定してください。ゲートウェイ、cron、ワーカーのプロセスはデータベースを並行して開くため、Hermes は代替として SQLite の `locking_mode=EXCLUSIVE` を提供しません。

### 不変のインストールツリー

ホスト型および公開されている Docker イメージでは、`/opt/hermes` がインストールされたアプリケーションのツリーです。これは root が所有し、ランタイムの `hermes` ユーザーには読み取り専用なので、エージェントのターン、ゲートウェイのセッション、ダッシュボードの操作、通常の `docker exec hermes hermes ...` コマンドが、コアのソース、同梱の `.venv`、`node_modules`、TUI バンドルをその場で編集することはできません。

変更可能な Hermes の状態はすべて `/opt/data` 配下に属します: 設定、`.env`、プロファイル、スキル、メモリ、セッション、ログ、ダッシュボードのアップロード、プラグイン、その他ユーザーが管理するファイルです。イメージはまた、`/opt/hermes` への実行時の `.pyc` の書き込みと Hermes の遅延依存関係インストールを無効にしています。公開イメージが必要とするオプションのプラットフォーム依存関係は、イメージに焼き込むか、新しいイメージのビルドを通じてインストールする必要があります。

ホスト型/公開イメージでは、エージェントの自己改善は `/opt/data` 配下のスキル、メモリ、プラグイン、設定にスコープされます。`/opt/hermes` 配下のインストール済みのコアソースは不変です。コアの変更はリポジトリへの PR で行われ、稼働中のインストールをその場で編集するのではなく、イメージの更新によって出荷されます。

オペレーターが `/opt/data` 以外のファイルを修復または調査する必要がある場合は、意図的に root シェルを使ってください。`hermes` シムは通常、`docker exec hermes hermes ...` をランタイムユーザーに戻します。明示的に root のセマンティクスが必要な場合は、1 回限りの root での呼び出しのために `HERMES_DOCKER_EXEC_AS_ROOT=1` を設定してください。

認証情報を `~` 配下に保存するスキルの CLI は、データボリュームのルートだけでなく、サブプロセスの HOME に対して初期化する必要があります。たとえば [xurl スキル](/docs/user-guide/skills/bundled/social-media/social-media-xurl)は OAuth の状態を `~/.xurl` に保存します。公式の Docker レイアウトでは、Hermes のツール呼び出しはそれを `/opt/data/home/.xurl` として読むため、手動の xurl 認証は `HOME=/opt/data/home` で実行し、`HOME=/opt/data/home xurl auth status` で確認してください。

:::warning
同じデータディレクトリに対して 2 つの Hermes **ゲートウェイ** コンテナを同時に実行しないでください — セッションファイルとメモリストアは同時書き込みアクセス向けに設計されていません。
:::

## マルチプロファイルのサポート {#multi-profile-support}

Hermes は[複数のプロファイル](../reference/profile-commands.md)をサポートします — 単一のインストールから独立したエージェント（異なる SOUL、スキル、メモリ、セッション、認証情報）を実行できる、別々の `~/.hermes/` サブディレクトリです。**公式 Docker イメージ内では、s6 の監視ツリーが各プロファイルを第一級の監視対象サービスとして扱う** ため、推奨されるデプロイは **すべてのプロファイルをホストする 1 つのコンテナ** です。

`hermes profile create <name>` で作成された各プロファイルには、次が与えられます:

- `/run/hermes-service/gateway-<name>/` にある専用の s6 サービススロット。ランタイムによって動的に登録されます — コンテナの再ビルドは不要です。
- クラッシュ時の自動再起動。`s6-supervise` によってバックオフが管理されます。
- `${HERMES_HOME}/logs/gateways/<name>/current` にある、プロファイルごとのローテーションされるログ（1 MB × 10 アーカイブ）。
- コンテナの再起動をまたいだ状態の永続化: ブート時のリコンサイラーが各プロファイルディレクトリの `gateway_state.json` を読み、最後に記録された状態が `running` だったプロファイルのスロットだけを復帰させます。再起動をまたいでダウンしたままになるのは、明示的に停止した（`hermes gateway stop`）ゲートウェイだけです — コンテナの再起動、イメージのアップグレード、予期しない終了では記録された状態は `running` のままなので、ゲートウェイは次のブート時に自動的に起動します。

バインドマウントされた `~/.hermes` に対して **ホスト** から作成されたプロファイルは、ディレクトリは得ますがスロットは得ません（ホストのプロセスはコンテナの `/run/hermes-service` に到達できないため）。コンテナ内では、`hermes -p <name> gateway start` が欠けているスロットを必要に応じて登録して起動します — `docker restart` は不要です。これを行うのは `start` だけで、実際のプロファイルディレクトリ（`SOUL.md` を含むもの）に限られます。未登録のプロファイルに対する `stop`/`restart` や、`-p` の名前の打ち間違いは、引き続き `✗ no such gateway` で失敗します。

ホスト上で実行するライフサイクルコマンドは、コンテナ内からも同じように機能します:

```sh
# Create a profile — registers the gateway-<name> s6 slot.
docker exec hermes hermes profile create coder

# Start / stop / restart — dispatches s6-svc; the gateway lifecycle survives docker restart.
docker exec hermes hermes -p coder gateway start
docker exec hermes hermes -p coder gateway stop
docker exec hermes hermes -p coder gateway restart

# Status — reports `Manager: s6 (container supervisor)` inside the container.
docker exec hermes hermes -p coder gateway status

# Remove a profile — tears down the s6 slot too.
docker exec hermes hermes profile delete coder
```

内部的には、コンテナ内の `hermes gateway start/stop/restart` は横取りされ、正しいサービスディレクトリに対する `s6-svc` にルーティングされます。s6 のコマンドを直接覚える必要はありません。生の監視状態を見るには `/command/s6-svstat /run/hermes-service/gateway-<name>` を使ってください（`/command/` が PATH に含まれるのは監視ツリーから起動されたプロセスだけなので、`docker exec` から呼び出すときは絶対パスを渡してください）。

### コンテナの外から複数のプロファイルに到達する

コンテナの外からプロファイルのゲートウェイに到達するサーフェスは 2 つあり、挙動が異なります — 混同しないでください:

**Hermes Desktop（および Web ダッシュボード）。** Desktop アプリの **Remote Gateway** 接続は、OpenAI API サーバー *ではなく*、`hermes dashboard` バックエンド（デフォルトは **ポート 9119**、`HERMES_DASHBOARD=1` で有効化）と通信します。1 つのダッシュボードバックエンドが、同じ場所にある **すべての** プロファイルを提供します: アプリのプロファイルスイッチャーが各リクエストで対象のプロファイルを送り、バックエンドがディスク上のそのプロファイルの `HERMES_HOME` を開きます。そのため Desktop では、プロファイルごとに 2 つ目のポート — や 2 つ目の接続 — は **不要** です。1 つの `:9119` の接続で、スイッチャーを通じてすべてをカバーできます。

**OpenAI 互換の API クライアント（Open WebUI、LobeChat、`/v1/...`）。** これらは各プロファイルの **API サーバー** と通信し、それは **すべてのプロファイルでポート 8642** にバインドします（`API_SERVER_PORT` / `platforms.api_server.extra.port` から解決されます — 自動割り当てはなく、`config.yaml`/`gateway.port` のキーもありません）。クライアントを *特定の* 2 つ目のプロファイルに到達させたい場合は、そのプロファイル **自身の** `.env` で別の `API_SERVER_PORT` を与えてください。そうしないと、そのゲートウェイも 8642 にバインドしようとして、デフォルトプロファイルと競合します:

```sh
# Create the profile (registers its gateway-<name> s6 slot)
docker exec hermes hermes profile create work

# Point its API server at a free port (write to the profile's own .env)
cat >> /opt/data/profiles/work/.env <<'EOF'
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
EOF

docker exec hermes hermes -p work gateway restart
```

`API_SERVER_PORT` は各プロファイル **自身の** `.env` に置き、コンテナ全体の `environment:` ブロックには決して置かないでください — グローバルな値はすべてのプロファイルを同じポートに強制し、それらが衝突してしまいます。ブリッジネットワーキングでは、追加のポートを `docker-compose.yml` で公開します（`- "8643:8643"`）。`network_mode: host` ではすでにホスト上で到達可能です。デフォルトプロファイルの 8642 の接続には影響しません。

### 多数のコンテナではなく、多数のプロファイルを持つ 1 つのコンテナにする理由

s6 への移行以前は、複数のゲートウェイを管理するコンテナ内のスーパーバイザーがなかったため、「プロファイルごとに 1 コンテナ」が推奨パターンでした。s6 が PID 1 になった今、それはもはや必要なく、単一コンテナのレイアウトはほぼあらゆる面でよりシンプルです:

| | 1 つのコンテナ、多数のプロファイル | プロファイルごとに 1 コンテナ |
|---|---|---|
| ディスクのオーバーヘッド | 1 つのイメージ、1 つの同梱 venv、1 つの Playwright キャッシュ | N 個のイメージ / N 個のキャッシュ |
| メモリのオーバーヘッド | 共有の Python インタープリターキャッシュ、共有の node_modules | コンテナごとに重複 |
| プロファイルの作成 | `docker exec ... hermes profile create <name>`（数秒） | 新しい `docker run` の呼び出し + ポートの割り当て + バインドマウントの設定 |
| プロファイルごとのクラッシュからの回復 | `s6-supervise` による自動再起動 | Docker の `--restart unless-stopped`（より遅く、兄弟の作業も止める） |
| ログ | `s6-log` によるプロファイルごとのローテーションファイルと、コンテナのブート監査ログ | コンテナごとの `docker logs <name>` — 組み込みのローテーションなし |
| バックアップ | 1 つの `~/.hermes` ディレクトリ | 調整が必要な N 個のディレクトリ |

デフォルトプロファイル（`default`）は初回ブート時に常に登録されるため、新しいコンテナには最初から 1 つの監視下のゲートウェイが付属します。追加のプロファイルは純粋に実行時の追加です。

### 別のコンテナが必要な場合

コンテナ内のプロファイルがデフォルトです。プロファイルごとに別のコンテナを実行するのは、特定の理由がある場合だけにしてください:

- **ワークロードごとのリソース分離** — たとえば、プロファイル A の暴走したブラウザツールのセッションが、プロファイル B を OOM にできないようにする。コンテナならプロファイルごとに `--memory` / `--cpus` を与えられます。
- **独立したイメージの固定** — ワークロードごとに異なる上流のイメージタグ。
- **ネットワークの分割** — プロファイルごとに別の Docker ネットワーク（例: 顧客向けのものと内部用のもの）。
- **コンプライアンス / 影響範囲** — 別々の認証情報が OS レベルのプロセスツリーを決して共有しない。

その場合は、それぞれ異なる `container_name`、`volumes`、`ports` を持つサービスをプロファイルごとに 1 つ宣言します:

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

[永続ボリューム](#persistent-volumes)の警告は引き続き適用されます: 2 つのコンテナを同時に同じ `~/.hermes` ディレクトリに向けないでください。各コンテナ内の s6 スーパーバイザーはそれぞれのプロファイルのセットを管理します。データボリュームをコンテナ間で共有すると、セッションファイルとメモリストアが破損します。

## ログの出力先 {#where-the-logs-go}

s6 コンテナには 4 つの異なるログのサーフェスがあり、「なぜゲートウェイが `docker logs` に何も表示しないのか」はよくある驚きです。早見表:

| ソース | 出力先 | 読み方 |
|---|---|---|
| **プロファイルごとのゲートウェイ**（`hermes gateway run` と s6 配下のプロファイルごとのゲートウェイ） | 2 か所に tee されます: `docker logs <container>`（リアルタイム、追加のプレフィックスなし）**と** `${HERMES_HOME}/logs/gateways/<profile>/current`（ローテーション、ISO-8601 タイムスタンプ付き、1 MB × 10 アーカイブ） | ホスト上で `docker logs -f hermes` または `tail -F ~/.hermes/logs/gateways/default/current` |
| **ダッシュボード**（`HERMES_DASHBOARD=1` の場合） | `docker logs <container>`（プレフィックスなし） | `docker logs -f hermes` — ゲートウェイの行と混ざって表示 |
| **ブート時のリコンサイラー**（各コンテナ起動時にどのプロファイルのゲートウェイが復元されたかを記録） | `${HERMES_HOME}/logs/container-boot.log`（追記専用の監査ログ） | `tail -F ~/.hermes/logs/container-boot.log` |
| **一般的な Hermes のログ**（`agent.log`、`errors.log`） | `${HERMES_HOME}/logs/`（プロファイルを認識） | `docker exec hermes hermes logs --follow [--level WARNING] [--session <id>]` |

ターミナルの入出力を持つインタラクティブなコンテナ（`docker run -it ... --tui` や `bash` など）では、フォアグラウンドのコマンドがターミナルを所有します。エントリポイントはデフォルトで `S6_LOGGING=2` と `S6_CMD_USE_TERMINAL=1` を設定するため、監視下のゲートウェイとダッシュボードの出力がその画面を上書きすることはありません。それらのコンソール出力は、コンテナ内の `/run/uncaught-logs/current` にある s6 のローテーションされる受け皿ログに送られます。プロファイルごとのゲートウェイのファイルログと通常の Hermes のログは、上記のパスで引き続き利用できます。受け皿ディレクトリはランタイムのストレージなので、コンテナの再起動や削除を越えて残りません。

ヘッドレスのコンテナでは、引き続きサービスの出力を `docker logs` に転送します。明示的な `S6_LOGGING` と `S6_CMD_USE_TERMINAL` の設定は、これらのインタラクティブ用のデフォルトより優先されます。

知っておくべき実用上の帰結が 2 つあります:

- コンテナの再起動を越えて残るのは、`logs/gateways/<profile>/current` のファイルのコピーです。`docker logs` は現在のコンテナの生存期間中の出力しか保持せず（`docker rm` で消去されます）、ローテーションされるファイルはバインドマウントされたボリューム上に残ります。
- ブート時のリコンサイラーの監査行の形は `<iso-timestamp> profile=<name> prior_state=<state> action=<registered|started>` なので、`grep profile=coder ~/.hermes/logs/container-boot.log` を実行すれば、特定のプロファイルが最後にいつ復元され、s6 がそれを自動起動したかどうかがすぐに分かります。

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

:::note Docker を **ターミナルバックエンド** として使いたい場合は？
このページは、Hermes 自体を Docker 内で実行する方法を扱います。エージェントの `terminal` / `execute_code` の呼び出しを Docker サンドボックスコンテナ（Hermes プロセス間で共有される 1 つの長命なコンテナ — issue #20561 を参照）内で実行させたい場合、それは別の設定ブロックです — `terminal.backend: docker` に加え、`terminal.docker_image`、`terminal.docker_volumes`、`terminal.docker_forward_env`、`terminal.docker_env`、`terminal.docker_run_as_host_user`、`terminal.docker_extra_args`、`terminal.docker_persist_across_processes`、`terminal.docker_orphan_reaper` です。コンテナのライフサイクルのルールを含む全体については [設定 → Docker バックエンド](configuration.md#docker-backend) を参照してください。
:::

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
      - "8642:8642"   # gateway API
      - "9119:9119"   # dashboard (only reached when HERMES_DASHBOARD=1)
    volumes:
      - ~/.hermes:/opt/data
    environment:
      - HERMES_DASHBOARD=1
      # Uncomment to forward specific env vars instead of using .env file:
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

`docker compose up -d` で起動し、`docker compose logs -f` でログを表示します。監視下のゲートウェイの stdout は、ボリューム上の `${HERMES_HOME}/logs/gateways/<profile>/current` にも tee されます — 完全なルーティングマップについては[ログの出力先](#where-the-logs-go)を参照してください。

## オプション: Linux デスクトップのオーディオブリッジ {#optional-linux-desktop-audio-bridge}

Docker でボイスモードを動かすには、2 つの別々のことが必要です: Hermes がコンテナ内でオーディオデバイスをプローブできること、そしてコンテナがホストのオーディオサーバーに到達できることです。以下のセットアップは、多くの PipeWire の構成を含め、PulseAudio 互換のソケットを公開する Linux デスクトップ向けのホスト側のオーディオの配管を扱います。

:::caution
これは Linux デスクトップ向けの回避策であり、Docker Desktop の一般的な機能ではありません。ホストのオーディオがすでに動作していて、Hermes コンテナ内で CLI のボイスモードを使いたい場合に役立ちます。Hermes が引き続き `Running inside Docker container -- no audio devices` と報告する場合は、`PULSE_SERVER` / `PIPEWIRE_REMOTE` に対する Docker のオーディオプローブのサポートを含むビルドを使ってください。
:::

まず、Compose ファイルの隣に ALSA の設定を作成します:

```conf title="asound.conf"
pcm.!default {
    type pulse
    hint {
        show on
        description "Default ALSA Output (PulseAudio)"
    }
}

pcm.pulse {
    type pulse
}

ctl.!default {
    type pulse
}
```

次に、ALSA の PulseAudio プラグインをインストールした小さな派生イメージをビルドします:

```dockerfile title="Dockerfile.audio"
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*
```

Compose でそのイメージを使い、ホストユーザーの PulseAudio のソケットと Cookie を渡します:

```yaml
services:
  hermes:
    build:
      context: .
      dockerfile: Dockerfile.audio
    image: hermes-agent-audio
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    volumes:
      - ~/.hermes:/opt/data
      - /run/user/${HERMES_UID}/pulse:/run/user/${HERMES_UID}/pulse
      # no-tmp: ok — path inside the container
      - ~/.config/pulse/cookie:/tmp/pulse-cookie:ro
      - ./asound.conf:/etc/asound.conf:ro
    environment:
      - HERMES_UID=${HERMES_UID}
      - HERMES_GID=${HERMES_GID}
      - XDG_RUNTIME_DIR=/run/user/${HERMES_UID}
      - PULSE_SERVER=unix:/run/user/${HERMES_UID}/pulse/native
      # no-tmp: ok — path inside the container
      - PULSE_COOKIE=/tmp/pulse-cookie
```

コンテナのプロセスがユーザーごとのオーディオソケットにアクセスできるよう、ホストの UID/GID で起動します:

```sh
export HERMES_UID="$(id -u)"
export HERMES_GID="$(id -g)"
docker compose up -d --build
```

コンテナ内で PortAudio に何が見えているかを確認するには:

```sh
docker exec hermes /opt/hermes/.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
```

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

## Dockerfile が行うこと {#what-the-dockerfile-does}

イメージは Debian 13.4 を使い、以下を含みます:


- コミットされた `uv.lock` から同期された Python 3.14 環境と、それに続く
  依存関係なしの Hermes の editable インストール。
- キュレーションされた extra の `all`、`messaging`、`otlp`、`anthropic`、`bedrock`、
  `azure-identity`、`matrix`。これは `--all-extras` ではありません。
- ダイジェストで固定された Node のソースイメージからの Node.js 26 と npm。
- `/opt/hermes/tools` 内の、PM で固定された uv、フル版の Chromium、FFmpeg、ripgrep。
- システムの Git、OpenSSH、Docker CLI、Chromium の共有ライブラリ。
- ビルド済みの TUI/ダッシュボードのアセットと、焼き込まれた Photon サイドカーの依存関係。
- 監視とゾンビプロセスのクリーンアップのための s6-overlay。

Chromium は `npx playwright install` ではなく、PM を通じて配置されます。ビルドは
解決された実行ファイルを `/etc/hermes/chromium-executable-path` に記録し、Hermes は
ブラウザを起動するときにそれを読みます。
`PLAYWRIGHT_BROWSERS_PATH` は、データマウントの外にある `/opt/hermes/tools` を指します。

サフィックスのない（`-desktop` でない）タグを含むすべてのイメージは、Playwright の
軽量なヘッドレスシェルではなくフル版の Chromium ビルドを持っています: 固定され
チェックサム検証された 1 つのブラウザが、ヘッドレスのブラウジングとヘッド付きの Bot Screen の
セッションの両方を担います。その代償はイメージのサイズです — フル版のビルドは、以前のイメージが同梱していた
ヘッドレスシェルより大きくなります。

オプトインのバックエンド SDK（Edge TTS、Firecrawl、Exa、プラットフォームアダプター、プラグインの
依存関係）は、初回使用時に `/opt/data/installs` 配下の PM の依存関係の世代に
インストールされるため、コンテナの再作成やイメージの更新を越えて残ります。
イメージ自身の `/opt/hermes/.venv` が変更されることはありません。ブートのたびに、
コンテナはサービスが開始する前に、記録された選択を新しいイメージのロックに対して再解決します。
それが失敗した場合（たとえばオフライン）、イメージ自身の環境でブートし、記録された extra を
次のブートやインストールのために保持します。オンデマンドのインストールを拒否するには
`security.allow_lazy_installs: false` を設定します。旧来の `lazy-packages` オーバーレイは使われません。

イメージの来歴は、ソースとデータの両方のマウントの外にある `/etc/hermes/image-provenance.json` にあります。
ビルドのスタンプは `/opt/hermes/install-stamp.json` にあります。
スタンプを指定しないローカルビルドは、コミットをでっち上げるのではなく、不明なリビジョンを報告します。
`hermes update` はイメージが所有するコードの変更を拒否します。アプリケーションを更新するには
イメージを置き換えてください。

コンテナの `ENTRYPOINT` は小さなディスパッチャー（`docker/entrypoint-dispatch.sh`）です。コンテナが PID 1 を所有している場合（通常の Docker / Podman）、それは s6-overlay の `/init` を exec し、下記で説明する完全な監視ツリーが得られます。プラットフォームがイメージのエントリポイントを独自の PID 1 の init の下でラップする場合（Fly.io Machines、`docker run --init`、一部の Nomad/Kubernetes の構成）、`/init` は `s6-overlay-suexec: fatal: can only run as pid 1` で中断してしまうため、ディスパッチャーは代わりに stage2 のブートストラップを直接実行し、s6 なしでメインのラッパーを exec します。そのフォールバックの経路でも要求されたコマンドは実行されますが、監視下のサービス（ダッシュボード、プロファイルごとのゲートウェイ）は利用できません。

PID 1 の経路では、`/init` は次を行います:
1. `/etc/cont-init.d/01-hermes-setup`（= `docker/stage2-hook.sh`）を root として実行します: オプションの UID/GID の再マッピング、ボリュームの所有権の修正、初回ブート時の `.env` / `config.yaml` / `SOUL.md` のシード、`HERMES_SKIP_CONFIG_MIGRATION=1` でない限り非インタラクティブな設定スキーマの移行の実行、同梱スキルの同期。
2. `/etc/cont-init.d/02-reconcile-profiles`（= `hermes_cli.container_boot`）を実行します: `$HERMES_HOME/profiles/<name>/` を走査し、`/run/hermes-service/gateway-<profile>/` 配下にプロファイルごとのゲートウェイの s6 サービススロットを再作成し、最後に記録された状態が `running` だったものだけを自動起動します（[プロファイルごとのゲートウェイの監視](#per-profile-gateway-supervision)を参照）。
3. 静的な `main-hermes` と `dashboard` の s6-rc サービスを開始します。
4. コンテナの CMD をメインプログラム（`/opt/hermes/docker/main-wrapper.sh`）として exec し、ユーザーが `docker run` に渡した引数をルーティングします:
   - 引数なし → `hermes`（デフォルト）
   - 最初の引数が PATH 上の実行ファイル（例: `sleep`、`bash`）→ それを直接 exec
   - それ以外 → `hermes <args>`（サブコマンドのパススルー）
   コンテナは、このメインプログラムが終了するとその終了コードで終了します。

:::warning s6 以前のイメージからの破壊的変更
コンテナの ENTRYPOINT は現在、`/usr/bin/tini` ではなく、`entrypoint-dispatch.sh` ディスパッチャー（PID 1 では s6-overlay の `/init` に委譲）です。ドキュメント化されている 5 つの `docker run` の呼び出しパターン（引数なし、`chat -q "…"`、`sleep infinity`、`bash`、`--tui`）はすべて、tini ベースのイメージと同じように振る舞います。tini 固有のシグナルの挙動や、ハードコードされた `/usr/bin/tini --` の呼び出しに依存する下流のラッパーがある場合は、以前のイメージタグに固定してください。
:::

:::warning 権限モデル
コマンドチェーンに `/init`（または同等の、stage2 フックに転送するレガシーの `docker/entrypoint.sh` シム）を残さない限り、イメージのエントリポイントを上書きしないでください。s6-overlay の `/init` は、初回ブート時にボリュームを chown できるよう root として実行され、その後すべての監視下のサービスとメインプログラムについて `s6-setuidgid` で `hermes` ユーザーに降格します。公式イメージ内で `hermes gateway run` を root として起動することは、`/opt/data` に root 所有のファイルを残し、後のダッシュボードやゲートウェイの起動を壊す可能性があるため、デフォルトで拒否されます。そのリスクを意図的に受け入れる場合にのみ `HERMES_ALLOW_ROOT_GATEWAY=1` を設定してください。
:::

:::warning `entrypoint:` を上書きするとゾンビの刈り取りもなくなります
孤立した孫プロセス（ヘッドレスブラウザ、MCP サーバー、ツールが起動した `git`/`npm` のヘルパー）を刈り取るのは `/init` です。たとえばダッシュボードを非 root ユーザーで実行するために `entrypoint:` を上書きして `hermes` を直接呼び出す Compose のサービスでは、hermes プロセス自体が PID 1 になり、その上で `wait()` を呼ぶものが何もなくなります: すべての孤児が永遠に `<defunct>` のエントリとして残ります（あるデプロイでは 3 時間足らずで 284 個のゾンビに達しました）。その構成では、Hermes は起動時に `[hermes] WARNING: this process is PID 1 with no init above it` と表示します。

どうしてもエントリポイントを上書きする必要がある場合は、孤児が刈り取られるよう、Docker の init を PID 1 として追加してください:

```yaml
services:
  hermes-dashboard:
    image: nousresearch/hermes-agent:latest
    init: true                                      # docker-init becomes PID 1 and reaps orphans
    entrypoint: ["/opt/hermes/.venv/bin/hermes"]
    command: ["dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open", "--skip-build"]
```

（`docker run --init …` が同等のフラグです。）これで解決するのはゾンビの蓄積だけです — `/init` がチェーンから外れているため、s6 の監視ツリーは依然としてありません: ダッシュボード、`hermes gateway run`、プロファイルごとのゲートウェイは監視されず、ディスパッチャー自身の非 PID 1 の警告が述べているとおりです。可能な限りデフォルトの `ENTRYPOINT` を維持してください。
:::

### `docker exec` は自動的に `hermes` ユーザーに降格します {#docker-exec-automatically-drops-to-the-hermes-user}

`docker exec hermes <cmd>` はデフォルトでコンテナ内で root として実行されますが、イメージには `/opt/hermes/bin/hermes`（PATH の先頭）に薄いシムが同梱されており、root の呼び出し元を検出して `s6-setuidgid hermes` 経由で透過的に再 exec します。そのため、`docker exec hermes hermes login`、`docker exec hermes hermes profile create …`、`docker exec hermes hermes setup` などはすべて、UID 10000 が所有する — つまり監視下のゲートウェイが読める — ファイルを書き込み、追加の `--user` フラグは不要です。非 root の呼び出し元（監視下のプロセス自体、`docker exec --user hermes`、コンテナ内の kanban のサブエージェント）は、venv のバイナリを直接 exec する近道を通るため、ホットパスにオーバーヘッドはありません。

root のセマンティクスを保持した `docker exec` が特に必要な場合（診断セッション、root 専用の状態の調査、たまたま root が所有している `/opt/data` 以外のファイル）は、呼び出しごとにオプトアウトします:

```sh
docker exec -e HERMES_DOCKER_EXEC_AS_ROOT=1 hermes <cmd>
```

シムは `1` / `true` / `yes`（大文字小文字を区別しない）を受け付けます。それ以外 — `=0` のような打ち間違いを含む — はすべて降格にフォールスルーするため、気づかないうちにオプトアウトしてしまうことはありません。`s6-setuidgid` が利用できない場合（s6-overlay を取り除いたカスタムビルド）、シムは root として実行することを拒否し、代わりに終了コード 126 で終了します。これにより、`docker exec hermes login` が `auth.json` を `root:root` として書き込み、チャットプラットフォームのメッセージのたびに監視下のゲートウェイの認証を壊していた従来の落とし穴に戻るのではなく、壊れた権限モデルをはっきりと表面化させます。

### プロファイルごとのゲートウェイの監視 {#per-profile-gateway-supervision}

`hermes profile create <name>` で作成された各プロファイルには、`/run/hermes-service/gateway-<name>/` に登録された s6 監視下のゲートウェイサービスが自動的に与えられ、コンテナの再起動をまたいで状態を保持した自動再起動が行われます。ユーザー向けのワークフローとライフサイクルコマンドについては、上記の[マルチプロファイルのサポート](#multi-profile-support)を参照してください。

**s6 以前のイメージと比べた監視の利点:**

- ゲートウェイのクラッシュは、約 1 秒のバックオフの後に `s6-supervise` によって自動的に再起動されます。
- `HERMES_DASHBOARD=1` で有効にしたダッシュボードは、同じ監視ツリー上で監視され、同じ自動再起動の扱いを受けます。
- `docker restart`、イメージのアップグレード（`docker compose up -d --force-recreate`）、予期しない終了でも、実行中のゲートウェイは維持されます: cont-init のリコンサイラーが `$HERMES_HOME/profiles/<name>/gateway_state.json` を読み、最後に記録された状態が `running` であればスロットを復帰させます。`stopped` を記録して再起動をまたいでゲートウェイをダウンさせたままにするのは、明示的な `hermes gateway stop` だけです。再起動やアップグレードの際にコンテナ/s6 が送る SIGTERM は「まだ実行中」として扱われ、自動起動されます。
- プロファイルごとのゲートウェイのログは `$HERMES_HOME/logs/gateways/<profile>/current`（`s6-log` によってローテーション）に残り、リコンサイラーの操作はブートごとに `$HERMES_HOME/logs/container-boot.log` に追記されます。完全なルーティングマップについては[ログの出力先](#where-the-logs-go)を参照してください。

コンテナ内の `hermes status --full` は `Manager: s6 (container supervisor)` と報告します。生の監視ビューには `/command/s6-svstat /run/hermes-service/gateway-<name>` を使ってください（`/command/` が PATH に含まれるのは監視ツリーのプロセスだけなので、`docker exec` から呼び出すときは絶対パスを渡してください）。

## アップグレード

最新のイメージをプルしてコンテナを再作成します。データディレクトリは
そのまま維持され、コンテナはゲートウェイを起動する前に、マウントされた `$HERMES_HOME/config.yaml` に対して
非インタラクティブな設定スキーマの移行を実行します。
移行が必要な場合、Hermes はまず `config.yaml` と `.env` の隣に
タイムスタンプ付きのバックアップを書き込みます。

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

新しいイメージに書き換えさせる前に、永続化された設定を手動で調査または移行する必要がある場合にのみ、
`HERMES_SKIP_CONFIG_MIGRATION=1` を設定してください。

## スキルと認証情報ファイル

Docker を実行環境として使う場合（上記の方法ではなく、エージェントが Docker サンドボックス内でコマンドを実行する場合 — [設定 → Docker バックエンド](./configuration.md#docker-backend)を参照）、Hermes はすべてのツール呼び出しに対して単一の長命なコンテナを再利用し、スキルディレクトリ（`~/.hermes/skills/`）とスキルが宣言する認証情報ファイルを、読み取り専用ボリュームとしてそのコンテナに自動的にバインドマウントします。スキルのスクリプト、テンプレート、参照は、手動設定なしでサンドボックス内で利用可能になり、コンテナが Hermes プロセスの生存期間中持続するため、インストールした依存関係や書き込んだファイルは次のツール呼び出しまで残ります。

同じ同期は SSH と Modal バックエンドでも行われます — スキルと認証情報ファイルは、各コマンドの前に rsync または Modal マウント API を介してアップロードされます。

## コンテナにツールを追加でインストールする

公式イメージにはキュレーションされたユーティリティのセットが同梱されていますが（[Dockerfile が行うこと](#what-the-dockerfile-does)を参照）、エージェントが必要とし得るすべてのツールがプリインストールされているわけではありません。推奨される方法は 5 つあり、手間と永続性の小さい順に並べています。

### npm や Python のツール — `npx` または `uvx` を使う

npm や PyPI で公開されているツールについては、`npx`（npm）または `uvx`（Python）で実行し、そのコマンドを永続メモリに覚えておくよう Hermes に指示します。ツールに設定ファイルや認証情報が必要な場合は、それらを `/opt/data` 配下（例: `/opt/data/<tool>/config.yaml`）に置くよう指示します。

依存関係は必要に応じて取得され、コンテナの生存期間中キャッシュされます。`/opt/data` 配下に書き込まれた設定は、バインドマウントされたホストのディレクトリにあるため、コンテナの再起動を越えて残ります。パッケージキャッシュ自体は `docker rm` の後に再構築されますが、`npx` と `uvx` は次にツールを実行するときに透過的に再取得します。

### その他のツール（apt パッケージ、バイナリ）— インストールして覚えておく

ランタイムユーザーはシステムの APT パッケージをインストールできません。たまに使うツールについては、
オペレーターが意図的に root シェルを使えます。それらの変更は、コンテナが置き換えられるまでしか
持続しません。コンテナの再起動では書き込み可能なレイヤーが保持されますが、
再作成では保持されません。インストールコマンドを覚えておくことは、自動的なプロビジョニングではありません。
繰り返し必要なシステムの依存関係には、派生イメージを使ってください。

これは、すばやくインストールでき、たまに使うツールに向いています。常に使うツールには、次の方法をおすすめします。

### 永続的なインストール — 派生イメージをビルドする

ツールを、再インストールの遅延なしで、すべてのコンテナ起動時にすぐ利用できるようにする必要がある場合は、`nousresearch/hermes-agent` を継承し、ツールをレイヤーでインストールする新しいイメージをビルドします:

```dockerfile
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends <your-package> \
    && rm -rf /var/lib/apt/lists/*
# Keep the root entrypoint; s6 drops privileges for the runtime.
```

ビルドして、公式イメージの代わりに使います:

```sh
docker build -t my-hermes:latest .
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  my-hermes:latest gateway run
```

エントリポイントスクリプトと `/opt/data` のセマンティクスは変更なしで継承されるため、このページの残りの内容は引き続き当てはまります。新しい上流の `nousresearch/hermes-agent` をプルしたときは、イメージを再ビルドすることを忘れないでください。

### 複雑なツールやマルチサービスのスタック — サイドカーコンテナを実行する

独自のサービス（データベース、Web サーバー、キュー、ヘッドレスブラウザのファーム）を伴うツールや、Hermes コンテナ内に置くには重すぎるツールは、共有の Docker ネットワーク上の別のコンテナとして実行します。Hermes は、ローカル推論サーバーに到達するのと同じ方法で、コンテナ名でサイドカーに到達します（[ローカル推論サーバーへの接続](#connecting-to-local-inference-servers-vllm-ollama-etc)を参照）。

```yaml
services:
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

  my-tool:
    image: example/my-tool:latest
    container_name: my-tool
    restart: unless-stopped
    networks:
      - hermes-net

networks:
  hermes-net:
    driver: bridge
```

Hermes コンテナ内からは、サイドカーには `http://my-tool:<port>`（またはそれが提供するプロトコル）で到達できます。このパターンでは、各サービスのライフサイクル、リソース制限、アップグレードの頻度を独立させられ、1 つのツールだけが必要とする依存関係で Hermes のイメージが肥大化するのを避けられます。

### 広く役立つツール — issue やプルリクエストを作成する

ツールがほとんどの Hermes Agent ユーザーに役立ちそうな場合は、プライベートな派生イメージで抱え込むのではなく、上流への貢献を検討してください。ツールとそのユースケースを説明する issue またはプルリクエストを [hermes-agent リポジトリ](https://github.com/NousResearch/hermes-agent)に作成してください。公式イメージに同梱されたツールはすべてのユーザーに恩恵をもたらし、下流のフォークの保守負担も避けられます。

## ローカル推論サーバーへの接続（vLLM、Ollama など） {#connecting-to-local-inference-servers-vllm-ollama-etc}

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

コンテナの stage2 フックは、各監視下のサービス内で `s6-setuidgid` を介して権限を非 root の `hermes` ユーザー（UID 10000）に降格します。ホストの `~/.hermes/` が別の UID によって所有されている場合は、`HERMES_UID`/`HERMES_GID` — または LinuxServer.io や NAS のイメージとの互換のための別名 `PUID`/`PGID` — をホストユーザーに合わせて設定するか、データディレクトリが書き込み可能であることを確認します:

データツリー全体を誰でも読める状態にしないでください。そこには認証情報が含まれています。
代わりに、コンテナの UID/GID をバインドマウントの所有者に合わせてください。

NAS（UGOS、Synology、unRAID）では、データディレクトリは通常、コンテナが `chown` できないホストの UID が所有する **バインドマウント** です。ランタイムが UID 10000 ではなくマウントの所有者として実行されるよう、`PUID`/`PGID`（または `HERMES_UID`/`HERMES_GID`）をそのホストユーザーに設定してください:

```sh
docker run -d \
  --name hermes \
  -e PUID=1000 -e PGID=10 \
  -v /volume1/docker/hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

`docker exec hermes <cmd>` も自動的に UID 10000 に降格します — 詳細と呼び出しごとのオプトアウトについては [`docker exec` は自動的に `hermes` ユーザーに降格します](#docker-exec-automatically-drops-to-the-hermes-user) を参照してください。

### 共有データディレクトリが `0700` にリセットされ続ける

コンテナの外では、Hermes は起動のたびに `HERMES_HOME`（およびその `cron/`、`sessions/`、`logs/`、`memories/` サブディレクトリ）を所有者のみの `0700` にロックします。コンテナ内ではディレクトリのモードに手を付けないため、別の UID で動作する兄弟のコンテナ（Web UI、権限修正ツール）と共有されるバインドマウントは、ホスト上で設定したモードと ACL を維持します。それでも特定のディレクトリモードを強制するには、`HERMES_HOME_MODE`（8 進数、例: `HERMES_HOME_MODE=0755`）を設定します。これはコンテナ内でも適用されます。

### すべての `docker exec` で「Permission denied」（インストールディレクトリが 0700 にロックされている）

2026 年 8 月下旬より前にビルドされたイメージには、`/opt/hermes` の直下に認証情報ファイルを書き込むとそのディレクトリが `0700` に制限され、`hermes` ユーザー（UID 10000）がインストールツリーから締め出されるバグがありました。その後、新しい `docker exec` はすべて `Permission denied` で失敗します。

新しいイメージをプルしてコンテナを再作成すると恒久的に修正されます（インストールディレクトリは `0755` で出荷され、現在のリリースではそれを制限しなくなりました）。再作成せずに稼働中のコンテナをその場で回復する必要がある場合は:

```sh
docker exec -u root hermes chmod 0755 /opt/hermes
```

### PID 1 の下にゾンビ（`<defunct>`）プロセスがたまる

コンテナ内で `ps -eo stat,ppid,comm | awk '$1 ~ /^Z/'` を実行すると、刈り取られなかった死んだ子プロセスが一覧表示されます。これは hermes 自体が PID 1 である場合に起こります — ほとんどの場合、Compose のサービスが `entrypoint:` を上書きしたため、`docker/entrypoint-dispatch.sh` → `/init` がスキップされたことが原因です。Hermes も起動時にそれについて警告します（`this process is PID 1 with no init above it`）。デフォルトのエントリポイントを復元するか、`docker-init` が孤児を刈り取るよう `init: true`（Compose）/ `docker run --init` を追加してください。[Dockerfile が行うこと](#what-the-dockerfile-does)を参照してください。コンテナを再作成すると、既存のゾンビはクリアされます。

### ブラウザツールが動作しない

Chromium は共有メモリを必要とします。Docker run コマンドに `--shm-size=1g` を追加します:

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
docker logs --tail 50 hermes          # Recent logs
docker run -it --rm nousresearch/hermes-agent:latest --version   # Verify version
docker stats hermes                    # Resource usage
```
