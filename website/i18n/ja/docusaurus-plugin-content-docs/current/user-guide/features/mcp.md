---
sidebar_position: 4
title: "MCP（Model Context Protocol）"
description: "MCP経由でHermes Agentを外部ツールサーバーに接続し、Hermesがロードするツールを厳密に制御する"
---

# MCP（Model Context Protocol）

このページのPython依存関係コマンドは、
[PMで準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を使用します。
依存関係を変更した後は、チェックアウトを再アクティベートしてHermesを再起動してください。

MCPを使うと、Hermes Agentは外部ツールサーバーに接続でき、エージェントはHermes自体の外部に存在するツール — GitHub、データベース、ファイルシステム、ブラウザスタック、内部APIなど — を利用できます。

Hermesに、どこか別の場所に既に存在するツールを使わせたいと思ったことがあるなら、MCPは通常それを実現する最もクリーンな方法です。

:::tip Claude Codeから移行しますか?
`~/.claude.json`の`mcpServers`ブロックは、Hermesの`config.yaml`の`mcp_servers`に対応します。また、（`agent-importers`プラグインの）`hermes import-agent claude-code`を使えば、スキルや指示とあわせてそれを移行できます。[他のエージェントからのインポート](/docs/user-guide/import-from-other-agents)を参照してください。
:::

## MCPで得られるもの

- 先にネイティブのHermesツールを書かなくても、外部のツールエコシステムにアクセスできる
- ローカルのstdioサーバーとリモートのHTTP MCPサーバーを同じ設定に含められる
- 起動時の自動的なツール検出と登録
- サーバーがサポートする場合、MCPのリソースとプロンプトに対するユーティリティラッパー
- サーバーごとのフィルタリングにより、実際にHermesに見せたいMCPツールだけを公開できる

## クイックスタート

1. MCPサポートは標準インストールに含まれています — 追加の手順は不要です。

2. `~/.hermes/config.yaml`にMCPサーバーを追加します。

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
```

3. Hermesを起動します。

```bash
hermes chat
```

4. MCPに支えられた機能を使うようHermesに依頼します。

例えば次のように依頼します。

```text
List the files in /home/user/projects and summarize the repo structure.
```

Hermesは、そのMCPサーバーのツールを検出し、他のツールと同じように使用します。

## カタログ: Nous承認済みMCPのワンクリックインストール

Hermesには、Nousのスタッフがレビューしてマージした、厳選されたMCPサーバーのカタログが同梱されています。
これらはデフォルトで無効になっています — 実際に必要なものだけをインストールしてください。

チャットで「Linear MCPを追加して」と依頼することもできます。エージェントは`mcp: true`ターゲットを指定して
`manage_connections`を呼び出し、セットアップカードが表示されます。このカードは、デスクトップアプリ（ダイアログ）、
ターミナルUI（`hermes --tui`、コンポーザーの上のコールアウト）、クラシックCLI（パネル）のいずれでも同じように動作します。

1. **フィールド。** エントリがセットアップ値を宣言している場合、カードはそれらをすべて一度に表示します。
   通常の値はデフォルト値で事前入力されます。シークレットはマスクされます。
   入力中は何も保存されません。
2. **接続またはキャンセル。** キャンセルするとそのサーバーだけがスキップされ、同じリクエスト内の
   他のサーバーは続行されます。
3. **認可。** OAuthエントリの場合、カードは認可リンクを表示します。
   Hermesが自らブラウザを開くことはありません。デスクトップでは**Open in browser**をクリックし、
   ターミナルではEnterを押します。SSH経由の場合、カードはコールバックポートに到達する方法、
   またはリダイレクトされたURLを貼り付ける方法を案内します。
4. **保存。** サーバーが新しいトークンを受け入れ、最初の接続が戻ってきた時点で、Hermesはサーバー設定、
   トークン、セットアップ値をまとめて保存します。サーバーがトークンを拒否した場合、またはその時点より前に
   キャンセルした場合、その試行の内容は何も保持されず、以前の設定とトークンはそのまま残り、
   失敗したフォームは入力した内容で再度開かれます。既に認可済みのサーバーは保存済みのトークンで接続し、
   Hermesはそれらが機能しなくなったときにのみ再認可を求めます。
5. **ツール。** その後、Hermesはサーバーのツールを一覧表示して登録します。
   エージェントは同じターン内でそれらを呼び出せます。認可は成功したもののツールリストの取得に失敗した場合、
   カードには「Authorized. Tools unavailable.」と表示され、エージェントは再認可を求めることなく
   後で検出を再実行できます。

メッセージングアプリにはカードがなく、エージェントが以下のコマンドを中継します。

```bash
hermes mcp                   # interactive picker (default)
hermes mcp catalog           # plain-text list, scriptable
hermes mcp install deepwiki  # install a catalog entry by name
```

ピッカーは各エントリを現在のステータスとともに表示します。

```
deepwiki     available              Ask questions about public GitHub repositories
linear       enabled                Linear issue/project management (remote OAuth)
github       installed (disabled)   GitHub repo + PR tools
```

行の上で`Enter`を押すと、インストール（必要な認証情報の入力手順も案内されます）、
有効化、無効化、アンインストールができます。カタログエントリはhermes-agentリポジトリの
`optional-mcps/`の下に保存されており、そのディレクトリに存在することがNousの承認を意味します。
コミュニティによる投稿の枠はなく、エントリはPRのマージによって追加されます。

サードパーティのn8nブリッジは、カタログからインストールできなくなりました。
既存のインストールでは、`mcp_servers`の設定、認証情報、インストール済みファイル、
選択したツールがそのまま保持されます。それらは設定済みのMCPサーバーとして引き続きロードされ、
ピッカーではカスタムエントリとして表示されます。そこでツールの設定や有効化・無効化は引き続き可能です。
カタログからの再インストールはできなくなりました。この変更によって、既存の接続が
[n8nの公式MCPサーバー](https://docs.n8n.io/connect/connect-to-n8n-mcp-server/)に移行されることはありません。

カタログエントリは次のものを必要とする場合があります。

- **APIキー** — Hermesはインストール時に入力を求め、その値を`~/.hermes/.env`に書き込みます。
  シークレットでない値（ベースURL）も同じファイルに書き込まれます。
- **OAuth**（リモートMCP） — 設定に`auth: oauth`として書き込まれます。MCPクライアントは
  最初の接続時にブラウザを開きます。
- **OAuth**（Google/GitHubのようなサードパーティプロバイダー） — まだ認証していない場合、
  Hermesは`hermes auth <provider>`を案内します。

### n8nの公式MCPサーバー

`n8n-official`カタログエントリは、ブラウザOAuthを使ってHTTP経由でn8n Cloudまたは
セルフホストのインスタンスに直接接続します。ローカルのブリッジやn8nのAPIキーは不要です。

1. オーナーまたは管理者に、n8nで**Settings > Instance-level MCP**を有効にしてもらいます。
2. **Connect**を開き、エディターのURLではなく、`/mcp-server/http`で終わる完全な**Server URL**を
   コピーします。古いバージョンでは、MCP設定ページにエンドポイントが直接表示されます。
3. `hermes mcp install n8n-official`を実行し、プロンプトが表示されたらそのURLを入力します。
4. ブラウザOAuthを完了します。必要に応じて、`hermes mcp login n8n-official`を実行するか、
   Desktopまたはダッシュボードで設定済みサーバーの**Authorize**を使います。
5. `hermes mcp configure n8n-official`でツールを確認し、新しいセッションを開始するか
   `/reload-mcp`を使います。

Hermesのバックエンドがそのアドレスに到達できる必要があります。権限とワークフローの公開範囲は
n8nが制御します。一部のツールはワークフローを変更または実行します。
[n8nの接続ガイド](https://docs.n8n.io/connect/connect-to-n8n-mcp-server/)を参照してください。

このエントリは、既存のカタログのセットアップおよび保存の動作を使用します。
廃止された`n8n`ブリッジとは別物であるため、既存の接続、認証情報、インストール済みファイル、
ツールの選択が置き換えられることはありません。

### インストール時のツール選択

認証情報が設定された後、HermesはMCPサーバーをプローブして公開されているすべてのツールを一覧表示し、
チェックリストを提示します。

```
Select tools for 'linear' (SPACE toggle, ENTER confirm)
  [x] find_issues       Find issues matching a query
  [x] get_issue         Get a single issue
  [x] create_issue      Create a new issue
  [ ] delete_workspace  Delete a Linear workspace
  ...
```

事前にチェックされる行は次のものから決まります。

1. **以前の選択** — このエントリを以前にインストールしたことがある場合（再インストールでは
   以前の選択が保持され、マニフェストのデフォルトで上書きされることはありません）
2. **マニフェストの`tools.default_enabled`** — エントリがそれを宣言している場合（一部の
   カタログエントリは、変更を伴うツールやほとんど役に立たないツールを事前に除外しています）
3. **すべて** — どちらにも該当しない場合

非常に大規模な自動生成された面を持つ一部のエントリ（例: `cloudflare`、
約3,300個のOpenAPIエンドポイントツール）は、代わりに`tools.default_excluded` — 名前とglobパターンの
厳選されたブロックリスト — を宣言しています。これらのいずれかをインストールすると、チェックリストは
完全にスキップされ、`tools.exclude`が書き込まれます。マッチしなかったものは、サーバーが後で追加する
ツールも含めてすべて有効のままになります。あるファミリーを再度有効にするには、config.yamlの
`mcp_servers.<name>.tools.exclude`を編集してください。

ENTERでチェックリストを確定します。チェックされたツールだけが
`mcp_servers.<name>.tools.include`に入ります。すべてを選択した場合、フィルタは書き込まれません
（最もクリーンな設定の形で、動作は同一です）。

**プローブが失敗した場合**（サーバーに到達できない、OAuthがまだ完了していない、
バックエンドのサービスが稼働していない）でも、インストールは成功します。マニフェストの
`tools.default_enabled`が（宣言されていれば）直接適用され、（宣言されていなければ）フィルタは
書き込まれません。サーバーに到達できるようになったら、`hermes mcp configure <name>`を再実行して
調整してください。

### 信頼モデル

カタログエントリをインストールすると、マニフェストが指定するもの — `git clone`、
エントリの`bootstrap`コマンド（`pip install`、`npm install`など）、そして最終的にはMCPサーバー自身の
コード — が実行されます。マニフェストはhermes-agentリポジトリへのPRレビューによって管理されているため、
各エントリは出荷前にNousによってレビューされています — **それでも、インストール前にマニフェストを
読むべきです**。特に`source:`フィールドのリポジトリ、`install.bootstrap:`コマンド、
そして`transport.command:`の呼び出しを確認してください。

マニフェストはGitHub上の
[`optional-mcps/<name>/manifest.yaml`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps)
にあります。ピッカーはインストール時にマニフェストの`source:` URLも表示するため、アップストリームの
リポジトリをすばやく確認できます。WebダッシュボードのMCPページでも、カタログエントリごとに同じ詳細 —
トランスポート、認証タイプ、エンドポイントURL（HTTP）またはコマンド＋引数（stdio）、gitのインストール元/ref、
bootstrapコマンド、セットアップメモ — が表示され、`source:`はクリック可能なリンクとして表示されるため、
Installをクリックする前に、エントリが何に接続し何を実行するのかを正確に確認できます。

### マニフェストのバージョン互換性

マニフェストは`manifest_version`を固定します。カタログは前方互換です。インストール済みのHermesが
理解できるよりも新しい`manifest_version`を持つエントリがPRで追加された場合、ピッカーはそのエントリを
黙って非表示にするのではなく、警告（`⚠ '<name>' requires a newer
Hermes`）を表示します。それが表示されたら、`hermes update`を実行して最新のHermesをインストールしてください。

### ランタイムの`${ENV_VAR}`置換

エントリの`transport.command`、`transport.args`、`transport.url`、
`headers`の中では、`${VAR}`プレースホルダーがサーバー接続時に環境変数（`~/.hermes/.env`の内容も
すべて含みます）から解決されます。これは、ユーザーが別の場所で設定した値をカタログエントリが
参照したい場合に便利です — 例: `${HOME}/foo`や`${MY_PROVIDER_TOKEN}`。

Cursor形式のコンテキスト変数も置換されます（大文字と小文字を区別）:
`${userHome}`（ホームディレクトリ）、`${workspaceFolder}`（セッションのワークスペースルート）、
`${workspaceFolderBasename}`、そして`${pathSeparator}` / `${/}`
（OSのパス区切り文字）。詳細は
[MCP設定リファレンス](../../reference/mcp-config-reference.md)を参照してください。

これは、カタログのマニフェストにおける`${INSTALL_DIR}`とは別物である点に注意してください。
`${INSTALL_DIR}`は、インストール時に、カタログがエントリのリポジトリをクローンしたパスで置換されます。

### 独自のOAuthアプリが必要なエントリ（DCRなし）

一部のベンダーは、リモートMCPをOAuthの背後で運用していますが、Dynamic Client Registrationを
**提供していません** — すべてのクライアントは、ユーザーがベンダーの開発者コンソールで事前登録した
アプリでなければなりません。AsanaのV2サーバー
（`https://mcp.asana.com/v2/mcp`）が同梱されている例です。廃止されたV1の
`https://mcp.asana.com/sse`サーバーは任意のクライアントを受け入れましたが、V2はそうではありません。

このようなマニフェストは、`auth.env`の下で認証情報を宣言し、`auth.oauth`の下でクライアントを固定します。
そのため、それをインストールすると（CLIピッカー、Webダッシュボード、Desktopのいずれでも）
Client ID / Client secretの入力が求められ、それらはプロファイルの`.env`に保存され、
`config.yaml`には`${VAR}`参照だけが書き込まれます。

```yaml
mcp_servers:
  asana:
    url: https://mcp.asana.com/v2/mcp
    auth: oauth
    oauth:
      client_id: "${ASANA_CLIENT_ID}"
      client_secret: "${ASANA_CLIENT_SECRET}"
      redirect_host: localhost      # the vendor matches the redirect URL exactly
      redirect_port: 27890          # register http://localhost:27890/callback on the app
```

登録すべき正確なアプリタイプとリダイレクトURLについては、エントリの`post_install`メモを読み、
その後`hermes mcp login <name>`を実行して、ツールを公開すべきセッションまたはゲートウェイを
再起動（または`/reload-mcp`）してください。ダッシュボード / Desktopの**Authorize**ボタンも使えます。
クライアントは固定された`redirect_port`で事前登録されているため、Hermesはダッシュボード自身の
コールバックURLではなく、登録済みのループバックコールバック（`http://localhost:27890/callback`）を
維持します — そのため、承認を行うブラウザはHermesプロセスと同じマシン上で実行されている必要があります。
リモートホストの場合は、SSHポートフォワーディング経由で`hermes mcp login`を使ってください。

### 後からツール選択を更新する

```bash
hermes mcp configure linear
```

現在の選択が事前にチェックされた状態で、同じチェックリストを再度開きます。より多くのツールを
有効にしたい場合や、サーバーに新しいツールが追加されてそれを利用したい場合に使います。

### カタログのマニフェストを更新する

MCPが自動更新されることはありません。Hermesの更新後にマニフェストのバージョンが変わっていた場合は、
`hermes mcp install <name>`を再実行してリフレッシュしてください。

カタログにMCPを追加するには、
[`optional-mcps/`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps)に対してPRを作成してください。

### 提案メタデータ（`suggest:`）

マニフェストは、`keywords:`および/または`hosts:`のリストを持つ任意の`suggest:`ブロックを宣言できます。
UIの面（現在はDesktopアプリのコンポーザー）はこれを使い、下書きがキーワードのいずれかを完結した単語として
含む場合、またはホスト名がいずれかのホストサフィックスで終わる貼り付けられたリンクを含む場合に、
ワンクリックの「Add &lt;server&gt;」ピルを提示します。これは純粋に助言的なもので — インストールは
引き続き同じ検証済みのカタログ/設定パスを通ります — ほとんどのホスト型リモートエントリ
（Atlassian、Sentry、Notion、Stripe、Vercel、Supabaseなど）がこれを宣言しています。
任意の`applications:`（最大16個のアプリ名）、`requires_app: true`
（そのMCPにはそのローカルアプリが必要であることを示し、このようなエントリがワンクリックのピルとして
提示されることはありません）、`examples:`（最大6個の1行の成果）は、デスクトップアプリケーションを
操作するエントリを記述します。

GitHubは意図的にカタログに**含まれていません**。GitHubのホスト型MCPでは各クライアントが独自のOAuthアプリを
用意する必要があり（汎用的なDynamic Client Registrationは拒否されます）、`gh` CLIを操作するHermesの
同梱`github/*`スキルの方がより高機能な統合だからです。Desktopでは、GitHubへの言及があると、代わりに
`gh`がまだサインインしていない場合に`github-auth`スキルが提示されます。

## 2種類のMCPサーバー

### stdioサーバー

stdioサーバーはローカルのサブプロセスとして実行され、stdin/stdout経由で通信します。

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
```

stdioサーバーを使うのは次の場合です。
- サーバーがローカルにインストールされている
- ローカルリソースへの低レイテンシなアクセスが欲しい
- `command`、`args`、`env`を示すMCPサーバーのドキュメントに従っている

### HTTPサーバー

HTTP MCPサーバーは、Hermesが直接接続するリモートのエンドポイントです。

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ***"
```

HTTPサーバーを使うのは次の場合です。
- MCPサーバーが別の場所でホストされている
- 組織が内部のMCPエンドポイントを公開している
- その統合のためにHermesにローカルのサブプロセスを生成させたくない

HTTPおよびSSEサーバーは標準のプロキシ設定に従います。まず`HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`（`socks://`エイリアスは`socks5://`に正規化されます）、次にOSのプロキシ（Windowsのレジストリ、macOSのシステム設定）が使われ、`NO_PROXY`のホスト — CIDR範囲や`*.example.com`パターンを含みます — には直接接続します。

### OAuth認証のHTTPサーバー {#oauth-authenticated-http-servers}

ほとんどのホスト型MCPサーバー（Cloudflare、Linear、Sentry、Atlassian、Asana、Figma、Stripeなど）は、静的なBearerトークンではなくOAuth 2.1を必要とします。`auth: oauth`を設定すると、HermesはMCP Python SDKを介して、検出、クライアント識別、PKCE、トークン交換、リフレッシュ、ステップアップ認証を処理します。

Hermesは、それをサポートするサーバーでは[Client ID Metadata Document](../../reference/mcp-config-reference.md#client-identification-cimd-and-dcr)で自身を識別し、サポートしないサーバーではDynamic Client Registrationにフォールバックします。どちらも自動で行われ、設定は不要です。

:::tip FigmaのリモートMCP
Figmaのホスト型エンドポイント（`https://mcp.figma.com/mcp`）は、Dynamic Client Registrationを**`client_name`の完全一致**で許可リスト化しています — 素の`"Hermes Agent"`は403になり、`"Claude Code"`と`"Codex"`は成功します。Hermesは`mcp.figma.com`に対して`oauth.client_name: "Claude Code"`を自動設定するため、特別な工夫なしでインストール/ログインが機能します。

```yaml
mcp_servers:
  figma:
    url: "https://mcp.figma.com/mcp"
    auth: oauth
```

または、`hermes mcp install figma`を実行し、その後`hermes mcp login figma`を実行します。
:::

```yaml
mcp_servers:
  linear:
    url: "https://mcp.linear.app/mcp"
    auth: oauth
```

初回接続時、Hermesは認可URLを表示し、可能であればブラウザを開き、ローカルのループバックポートでOAuthコールバックを待ちます。トークンは0o600のパーミッションで`~/.hermes/mcp-tokens/<server>.json`にキャッシュされ、以降の実行ではリフレッシュが失敗するまで黙って再利用されます。

リフレッシュトークンは、それを発行した認可サーバーに紐付けられます。Hermesは検出したissuerをキャッシュ済みトークンと一緒に記録し、サーバーが公示する認可サーバーが変わった場合（サーバーの移行、メタデータの編集、ハイジャック）、保存済みのリフレッシュトークンを新しいissuerに送信する代わりに破棄します。現在のアクセストークンは有効期限が切れるまで機能し続け、その後新しいissuerに対して通常の再認可が実行されます。

認可サーバーからのリダイレクトは、RFC 9207に照らして検査されます。サーバーのメタデータが`authorization_response_iss_parameter_supported`を公示している場合、一致する`iss`のないリダイレクトは拒否されます。Figmaの認可サーバー（`https://api.figma.com`）はこのサポートを公示しておきながら`iss`を省略します。Hermesはその1つのissuerに限り、検出したissuerで欠落した値を補い、警告をログに記録するため、`hermes mcp login figma`は完了します。存在するが異なる`iss`は引き続き拒否され、他のサーバーにはこの例外は適用されません。

認可サーバーのメタデータドキュメントは、リソースが公示したサーバーを指名していなければなりません（RFC 8414 §3.3）。別のサーバーのドキュメントは、登録やログインの前に拒否されます。完全一致でなくても受け入れられる形が1つあります。パス付き（`https://host/path`）で公示されたサーバーで、`https://host/.well-known/oauth-authorization-server/path`から取得したドキュメントが、オリジン`https://host`をissuerとして指名している場合です — StravaのMCPコネクタはまさにその組み合わせを公開しています。そのwell-knownの場所を制御できるのはオリジンの運営者だけなので、そのドキュメントは公示されたサーバー自身のものとして扱われます。別のオリジンや別のパスを指名するドキュメント、またはリダイレクトやフォールバックの場所を通じてのみ到達したドキュメントは、引き続き`Authorization server metadata issuer mismatch`で失敗します。

**Googleがホストするサーバー（Gmail、Calendar）。** Googleは、認可リクエストに`access_type=offline`が含まれている場合にのみリフレッシュトークンを発行しますが、MCPの検出ではこれが公示されることはありません。Hermesは、検出された認可サーバーが`accounts.google.com`である場合は常にこれを（再ログイン時にも再発行されるよう`prompt=consent`とあわせて）追加するため、接続は再起動後も維持され、`hermes gateway`からも機能します。他のissuerへのリクエストには手を加えません。

**リモート / ヘッドレスホスト。** Hermesがブラウザとは別のマシンで実行されている場合、ループバックコールバックはあなたのラップトップに到達できません。フローを完了する方法は次のとおりです。

- **Hermes Desktop（自動）:** リモートのバックエンドに対して、DesktopアプリのMCPセットアップUIからOAuthサインインを実行すると、Desktopは*あなたの*マシン上でコールバックリスナーをホストし、認可結果を自動的にゲートウェイへ中継します — トンネル、貼り付け、プロキシは不要です。Desktopアプリとバックエンドの両方が最新である必要があります。
- **貼り付け（セットアップ不要）:** インタラクティブなターミナルでは、Hermesは認可URLと一緒に「Or paste the redirect URL here…」を表示します。ブラウザでURLを開いて承認し、ブラウザが最終的に表示する完全なURL（リダイレクト先は接続エラーを表示しますが、それは想定どおりです）をコピーして、プロンプトに貼り付けます。素の`?code=…&state=…`クエリ文字列でも機能します。
- **デバイスコードログイン（コールバック一切なし）:** サーバーの認可サーバーがデバイス認可エンドポイントを公示している場合、Hermesを実行しているマシンで`hermes mcp login <server> --flow device`を実行します。検証URLと短いコードが表示されるので、任意のデバイスでURLを開いてコードを入力すると、Hermesが承認をポーリングします。ホスト上でブラウザが起動されることはなく、コールバックリスナーも不要です。サーバーに`oauth.flow: device`を設定すると、`login`と`reauth`がデフォルトでこれを使うようになります。詳細: [デバイスコードログイン](../../reference/mcp-config-reference.md#device-code-login-rfc-8628)。
- **SSHポートフォワード:** 別のターミナルで`ssh -N -L <port>:127.0.0.1:<port> user@host`を実行し、その後リダイレクトを通常どおり流します。
- **プロキシ経由のコールバック（`redirect_uri`）:** 公開HTTPSエンドポイントがホストへ転送している場合（例: コールバックポートを指すTailscale Funnelやリバースプロキシ）、`oauth.redirect_uri`を設定すると、ブラウザのリダイレクトが自力でHermesに到達します — トンネルや貼り付けは不要です。

```yaml
mcp_servers:
  myserver:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      redirect_port: 8765                                # fixed port for the proxy to target
      redirect_uri: "https://oauth.example.ts.net/callback"
```

完全にヘッドレスなゲートウェイ（メッセージングボットで、インタラクティブなターミナルが一切ない）の場合は、任意の[`mcp-oauth-remote-gateway`スキル](/docs/user-guide/skills/optional/mcp/mcp-mcp-oauth-remote-gateway)が、フローを手動で完了してHermesが期待する場所にトークンを書き込む手順をエージェントに案内します。

**落とし穴 — WAFが`127.0.0.1`のリダイレクトURIを拒否する。** 一部のプロバイダーは認可サーバーの前段にWAFを置いており、クエリ文字列にリテラルの`127.0.0.1`を含む認可リクエストをすべて403にします（Reclaim.aiのAWS API Gatewayが既知の例です — すべての試行がOAuthアプリに到達する前に`{"message":"Forbidden"}`を返します）。`oauth.redirect_host: localhost`を設定すると、代わりに`http://localhost:<port>/callback`が使われます。いずれの場合も、コールバックリスナーは`127.0.0.1`にバインドされます。

DCRに対応しないサーバー（例: Slack）、事前登録した`client_id`/`client_secret`、スコープのカスタマイズ、`hermes mcp login <server>`による再認証を含む完全な手順については、[SSH / リモートホスト経由のOAuth](/docs/guides/oauth-over-ssh#mcp-servers)を参照してください。

**落とし穴 — 自動登録をサポートしないプロバイダー（Google Drive、Atlassian）。** 一部のサーバーは、素の`auth: oauth`が依存するDynamic Client Registrationの手順（RFC 7591）を拒否します — Googleの公式Driveサーバー（`https://drivemcp.googleapis.com/mcp/v1`）は`400 Bad Request`を返すため、OAuthクライアントは作成されず、トークンも取得されません。症状はわかりにくいものです。これらのサーバーは認証*なし*でも`tools/list`を提供するため、`hermes mcp login`はツールを一覧表示でき、成功したように見えますが、その後の実際のツール呼び出しはすべてタイムアウトします。`hermes mcp login`は現在これを検出し（トークンが実際にディスクに保存されたかを確認します）、独自のOAuthクライアントを用意するよう案内します。プロバイダーのコンソールでクライアントを作成し、設定に追加してください。

```yaml
mcp_servers:
  googledrive:
    url: "https://drivemcp.googleapis.com/mcp/v1"
    auth: oauth
    oauth:
      client_id: "<your-oauth-client-id>"
      client_secret: "<your-oauth-client-secret>"
```

その後、`hermes mcp login googledrive`を実行します — 事前登録されたクライアントがあれば、Hermesは登録をスキップして通常のブラウザ認可フローを実行します。

**落とし穴 — 設定の自動リロードの競合。** 実行中のHermesセッション内から`~/.hermes/config.yaml`を編集すると、CLIは30秒のタイムアウトでMCP接続を自動リロードします。これはインタラクティブなOAuthフローには十分ではありません。エントリを追加したら、新しいターミナルから`hermes mcp login <server>`を実行してください — 認証の完了を最大5分間待ちます。

**承認に5分以上必要ですか?** サーバーエントリに`oauth.timeout`（秒）を設定してください。`hermes mcp login`、ダッシュボード、Desktopの再認証はいずれも`oauth.timeout` + 15秒（またはエントリの`connect_timeout`のいずれか長い方）待機します。それでも時間切れになったログインは、空の失敗行ではなく、両方の設定項目を挙げた`Connecting to MCP server '<name>' timed out after Ns`を報告します。

## mTLS / クライアント証明書

相互TLS（クライアント証明書認証）を必要とするリモートHTTP MCPサーバーは、`client_cert` / `client_key`によってサポートされます。Hermesは、解決した証明書をTLSハンドシェイクのために下層のHTTPクライアントに渡します。

`client_cert`は3つの形を受け付けます。

- **単一の結合PEMパス** — 証明書と秘密鍵の両方を含む1つのファイル:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/.certs/mcp-client.pem"
```

- **`[cert, key]`の2要素タプル** — 証明書と鍵が別々のファイルにある場合（`client_cert` + `client_key`を設定するのと同等）:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key"]
```

- **`[cert, key, password]`の3要素タプル** — 秘密鍵が暗号化されている場合、3番目の要素は鍵のパスフレーズです:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key", "${MCP_KEY_PASSWORD}"]
```

`client_cert`（結合PEM）に加えて明示的な`client_key`を使い、証明書と鍵を完全に分けておくこともできます。パスは`~`の展開をサポートします。ファイルが存在しない場合は、不透明なTLSハンドシェイクの失敗ではなく、サーバー単位のわかりやすいエラーが発生します。

## ユーザーごとのアイデンティティヘッダー

呼び出し元のアイデンティティに応じて動作を変えるリモートHTTP/SSE MCPサーバー（ユーザーごとのレート制限、監査証跡、マルチテナントのルーティング）には、`identity_header`を使って毎回のリクエストにアイデンティティヘッダーを送信できます。

```yaml
mcp_servers:
  team_api:
    url: "https://mcp.team.example.com/mcp"
    identity_header:
      name: "X-User-Id"
      value_from: "static"   # "static" (default) or "profile"
      value: "alice"         # required for static
```

- `value_from: static`は、config.yamlのリテラルの`value`を送信します。
- `value_from: profile`は、接続時に一度だけ解決されるアクティブなHermesプロファイル名を送信します — 1台のマシン上の複数のプロファイルが同じサーバーと通信し、サーバーがそれらを区別する必要がある場合に便利です。

サーバーの`headers`マッピング内に同じ名前（大文字小文字は問わない）の明示的なエントリがある場合は、常にそちらが優先されます。アイデンティティヘッダーがあなた自身のヘッダー設定を上書きすることはありません。無効な`identity_header`ブロックは警告されたうえで無視され、サーバーの接続を妨げることはありません。stdioサーバーでは、このキーは警告付きで無視されます（stdioトランスポートにはヘッダーがありません）。

## 基本的な設定リファレンス

Hermesは`~/.hermes/config.yaml`の`mcp_servers`の下からMCP設定を読み取ります。

### 共通キー

| キー | 型 | 意味 |
|---|---|---|
| `command` | 文字列 | stdio MCPサーバーの実行ファイル |
| `args` | リスト | stdioサーバーの引数 |
| `env` | マッピング | stdioサーバーに渡される環境変数 |
| `cwd` | 文字列 | stdioサーバープロセスの作業ディレクトリ。デフォルト: セッションの作業ディレクトリが固定されている場合（ACP/ゲートウェイのセッション、`terminal.cwd`）はそれ、それ以外はHermesプロセスのディレクトリ |
| `url` | 文字列 | HTTP MCPエンドポイント |
| `headers` | マッピング | リモートサーバー向けのHTTPヘッダー |
| `client_cert` | 文字列 \| リスト | mTLS用のクライアント証明書 — 結合PEMパス、または`[cert, key]` / `[cert, key, password]` |
| `client_key` | 文字列 | クライアント秘密鍵のPEMパス（`client_cert`と別の場合） |
| `identity_header` | マッピング | HTTP/SSEサーバー向けの任意のユーザーごとのアイデンティティヘッダー — `{name, value_from: static\|profile, value}` |
| `timeout` | 数値 | ツール呼び出しのタイムアウト |
| `connect_timeout` | 数値 | 初回接続のタイムアウト（MCPの`initialize`ハンドシェイクの上限にもなる） |
| `lazy` | bool | `true`の場合、起動時にスキーマキャッシュからサーバーのツールを登録し、最初のツール呼び出し時にのみ起動/接続する（デフォルト`false`）。キャッシュを埋めるために、事前に一度ライブ接続している必要がある。 |
| `idle_timeout_seconds` | 数値 | ツール呼び出しがないままこの秒数が経過すると、stdioサーバーをリサイクルする（`0` = しない、デフォルト）。サーバーは次のツール呼び出し時に透過的に再起動する。 |
| `max_lifetime_seconds` | 数値 | 起動からの総経過時間がこの値に達すると、stdioサーバーをリサイクルする（`0` = しない、デフォルト）。次回の使用時に透過的に再起動する。 |
| `enabled` | bool | `false`の場合、Hermesはそのサーバーを完全にスキップする |
| `supports_parallel_tool_calls` | bool | `true`の場合、このサーバーのツールは並行して実行される可能性がある |
| `events` | bool | `true`の場合、このサーバーに対して実験的なMCPイベントツールを登録する（デフォルト`false`）。[MCPイベント](#mcp-events-experimental)を参照 |
| `tools` | マッピング | サーバーごとのツールフィルタリングとユーティリティポリシー |

### 最小のstdioの例

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
```

### メモリを大量に消費するstdioサーバーのリサイクル

ブラウザベースのMCPサーバー（例: `@playwright/mcp`）は、最初のツール呼び出し以降、完全なChromiumを
常駐させ続けます — 解放されることのない数百MBです。自動リサイクルをオプトインすると、
アイドル/ライフタイムの上限に達した後にサーバーが停止され、そのツールのいずれかが次に呼び出されたときに
透過的に再起動されます（その間もツールはずっと登録されたままです）。

```yaml
mcp_servers:
  playwright:
    command: "npx"
    args: ["-y", "@playwright/mcp@latest", "--headless"]
    idle_timeout_seconds: 900     # recycle after 15 min without a tool call
    max_lifetime_seconds: 86400   # and at least once a day regardless
```

### 最小のHTTPの例

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

## 組み込みプリセット

よく知られたMCPサーバーについては、`hermes mcp add`が`--preset`フラグを受け付け、トランスポートの詳細を埋めてくれるため、コマンドや引数を調べる必要がありません。プリセットはデフォルト値を提供するだけで — 同じコマンドラインで渡したその他の値（環境変数、ヘッダー、フィルタリング）が引き続き優先されます。

| プリセット | 設定される内容 |
|---|---|
| `codex` | Codex CLIのMCPサーバー（stdio経由の`codex mcp-server`）。PATH上に`codex` CLIが必要。 |

```bash
# Add Codex CLI as an MCP server in one line
hermes mcp add codex --preset codex
```

これは次と同等の内容を書き込みます。

```yaml
mcp_servers:
  codex:
    command: "codex"
    args: ["mcp-server"]
```

任意のローカル名を選べます（`hermes mcp add my-codex --preset codex`でも問題ありません）。プリセットが提供するのは`command`/`args`のデフォルト値だけです。

## HermesがMCPツールを登録する方法

Hermesは組み込みの名前と衝突しないように、MCPツールにプレフィックスを付けます。

```text
mcp_<server_name>_<tool_name>
```

例:

| サーバー | MCPツール | 登録名 |
|---|---|---|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

実際には、プレフィックス付きの名前を手動で呼び出す必要は通常ありません — Hermesはツールを認識し、通常の推論の中でそれを選択します。

### ツール結果のサニタイズと`_meta`

すべてのMCPツール結果には、モデルがそれを見る前に2つの処理が適用されます。

- **不可視のUnicode TAG文字は除去されます。** U+E0000–U+E007Fの範囲の文字は、ターミナルやチャットUIでは何も表示されませんが、モデルからは完全に見えます — 悪意のある、または侵害されたサーバーにとって典型的なプロンプトインジェクションの密輸経路です。Hermesはツール結果、リソースの内容、ツールの説明からこれらを除去します。正当な絵文字タグシーケンス（🏴󠁧󠁢󠁳󠁣󠁴󠁿のような地域の旗）は保持されます。
- **ベンダーの`_meta`は表示され、プロトコル予約キーは表示されません。** サーバーがツール結果に`_meta`マッピング（`com.example/handoff`のようなベンダー名前空間）を付けた場合、Hermesはそれを結果の内容と一緒にモデルへ渡します。プロトコル予約のプレフィックス — `modelcontextprotocol`または`mcp`のラベルの後に別のラベルが続くもの、例: `modelcontextprotocol.io/...`や`tools.mcp.com/...` — の下にあるキーは、MCP仕様のキー名ルールに従って破棄されます。モデル向けのものが何も残らない場合、`_meta`フィールドは完全に省略されます。

### 画像の結果 {#image-results}

`ImageContent`ブロック（スクリーンショット、レンダリングされたチャート、ページのキャプチャ）はHermesの画像キャッシュに保存され、そのパスが結果の中で`MEDIA:/path/to/image.png`として報告されるため、添付ファイルに対応した任意の面でそのファイルを要求できます。メインモデルがツール結果内の画像を見られる場合は、画像そのものも結果に添付されるため、モデルはピクセルを直接読み取ります。MCPの結果は`vision_analyze`と同じルールに従います。`agent.image_input_mode`、明示的な`auxiliary.vision`バックエンド、モデルのビジョン機能がすべて適用されます（[ビジョン](vision.md#vision_analyze-has-the-same-dual-behavior)を参照）。添付された画像は以降のすべてのターンで再送信されるため、`vision.embed_target_bytes`まで縮小されます（結果ごとに最大4枚）。テキストのみのモデルにはパスだけが渡されます。

## MCPユーティリティツール

サポートされている場合、HermesはMCPのリソースとプロンプトに関するユーティリティツールも登録します。

- `list_resources`
- `read_resource`
- `list_prompts`
- `get_prompt`

これらは同じプレフィックスパターンでサーバーごとに登録されます。例えば次のようになります。

- `mcp_github_list_resources`
- `mcp_github_get_prompt`

### 重要

これらのユーティリティツールは、現在は機能を認識します。
- Hermesは、MCPセッションが実際にリソース操作をサポートしている場合にのみリソースユーティリティを登録します
- Hermesは、MCPセッションが実際にプロンプト操作をサポートしている場合にのみプロンプトユーティリティを登録します

したがって、呼び出し可能なツールは公開するがリソース/プロンプトを持たないサーバーには、それらの追加ラッパーは付きません。

## サーバーごとのフィルタリング

各MCPサーバーがHermesに提供するツールを制御でき、ツール名前空間のきめ細かな管理が可能になります。

### サーバーを完全に無効化する

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

`enabled: false`の場合、Hermesはそのサーバーを完全にスキップし、接続すら試みません。

### サーバーツールをホワイトリスト化する

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues]
```

それらのMCPサーバーツールだけが登録されます。

`include`/`exclude`のエントリには、globパターン（`*`、`?`、`[...]`、
大文字と小文字を区別してマッチ）も使えます。`include: ["*_dns_*"]`は、名前に`_dns_`を含む
すべてのツールを登録します。メタ文字を含まない通常のエントリは、引き続き完全一致です。
globは、何千もの自動生成されたエンドポイントツールを公開するサーバーを、製品ファミリー単位で
フィルタリングするための実用的な方法です。

### サーバーツールをブラックリスト化する

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

除外されたもの以外のすべてのサーバーツールが登録されます。

### globパターン

どちらのリストも、完全な名前と並んでfnmatch形式のglobを受け付けます — CloudflareのAPI MCP
（`?codemode=false`、約3,300個のツール）のような巨大でフラットな面では不可欠です。そうした面では、
製品領域をエンドポイント1つずつ除外するのは現実的ではありません。

```yaml
mcp_servers:
  cloudflare:
    url: "https://mcp.cloudflare.com/mcp?codemode=false"
    auth: oauth
    tools:
      exclude: ["*_radar_*", "*_accounts_dlp_*", "*_zones_web3_*"]
```

globのメタ文字（`*`、`?`、`[`）を含まないエントリは完全一致します — `docs`は
`docs`という名前のツールだけを除外し、`docs_search`を除外することはありません。

### 優先順位ルール

両方が存在する場合:

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

`include`が優先されます。

### ユーティリティツールもフィルタリングする

Hermesが追加したユーティリティラッパーを個別に無効化することもできます。

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

これは次を意味します。
- `tools.resources: false`は`list_resources`と`read_resource`を無効化する
- `tools.prompts: false`は`list_prompts`と`get_prompt`を無効化する

### 完全な例

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues, search_code]
      prompts: false

  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer]
      resources: false

  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

## すべてフィルタリングで除外されたらどうなる?

設定がすべての呼び出し可能なツールをフィルタリングで除外し、サポートされているすべてのユーティリティを無効化または省略している場合、Hermesはそのサーバーに対して空のランタイムMCPツールセットを作成しません。

これによりツールリストがクリーンに保たれます。

## ランタイムの動作

### 検出のタイミング

Hermesは起動時にMCPサーバーを検出し、それらのツールを通常のツールレジストリに登録します。

サーバーは、検出パス（起動時、`/reload-mcp`、設定ウォッチャー）ごとに**同時に最大4つ**まで接続されます。
各stdioサーバーはそれぞれ独自の子プロセスツリーを生成するため、多数のサーバーを無制限に処理するパスでは、
以前はそれらすべてが同じ瞬間に起動していました — CPU/RAMの急増と、複数プロファイルのフリートでは
プロバイダー呼び出しの同時集中が発生していました。`config.yaml`で調整できます。

```yaml
mcp:
  discovery_concurrency: 4   # max simultaneous server connects; 0 = unlimited
```

上限より多くのサーバーを持つパスはウェーブ単位で実行されます。各ウェーブには通常の120秒の予算があり
（パス全体の上限は300秒）、遅いフリートはタイムアウトするのではなく、遅れて完了します。

### 遅延起動

`lazy: true`のサーバーは、代わりにディスク上のスキーマキャッシュから登録されます。そのツールはすぐにレジストリに表示され、プロセスは最初のツール呼び出し時に生成されます（またはHTTPエンドポイントに接続されます）。キャッシュはライブ接続のたびに書き込まれるため、新規または変更されたサーバーの初回実行は常に即時起動になります。バナーとTUIのセッションパネルは、このようなサーバーをキャッシュされたツール数とともに**lazy**として表示し（`3 tool(s) (lazy, starts on first use)`）— これは失敗したサーバーではなく、正常に機能するサーバーです — 起動時の検出サマリーではこれを`N lazy, not spawned yet`として数えます。

### 動的ツール検出 {#dynamic-tool-discovery}

MCPサーバーは、`notifications/tools/list_changed`通知を送信することで、利用可能なツールが実行時に変化したことをHermesに通知できます。Hermesがこの通知を受け取ると、サーバーのツールリストを自動的に再取得してレジストリを更新します — 手動の`/reload-mcp`は不要です。

これは、機能が動的に変化するMCPサーバー（例えば、新しいデータベーススキーマがロードされたときにツールを追加するサーバーや、サービスがオフラインになったときにツールを削除するサーバー）に便利です。

このリフレッシュはロックで保護されているため、同じサーバーからの連続した通知が重複するリフレッシュを引き起こすことはありません。プロンプトとリソースの変更通知（`prompts/list_changed`、`resources/list_changed`）は受信されますが、まだ対応されていません。

### リロード

MCP設定を変更した場合は、次を使います。

```text
/reload-mcp
```

これは設定からMCPサーバーをリロードし、利用可能なツールリストをリフレッシュします。これは、利用可能性によって制限されるツール（Docker、`HASS_TOKEN`のようなプラグインの認証情報、OAuthなど）を再プローブする明示的な方法でもあります。セッションのツールセットはそれ以外では固定されているため、セッションの途中で現れた認証情報やデーモンは、`/reload-mcp`、`/new`、またはコンテキスト圧縮のときにのみ取り込まれます。サーバー自身がプッシュするランタイムのツール変更については、上記の[動的ツール検出](#dynamic-tool-discovery)を参照してください。

実行中のメッセージングゲートウェイ（`hermes gateway run`）も、それ自体で`config.yaml`を監視しています。`mcp_servers`のエントリを削除するか`enabled: false`を設定すると、約1分以内にそのサーバーの接続が切断されます。新しく追加されたエントリは接続されます。最初の接続に失敗したサーバー（到達できないホストや、まだトークンを持っていないヘッドレスマシン上のOAuthサーバー）は、原因を修正すれば、接続クールダウンのスケジュール（30秒から始まり、最大10分まで倍増）に従って自動的に再試行されます。編集を反映させるのに再起動や`/reload-mcp`は不要です。

**バックグラウンドでの期限切れOAuthトークン。** ゲートウェイ、`/reload-mcp`、停止中のサーバーの定期的な自己プローブは、ブラウザを開くことはありません — フローを完了する人がそこにいないからです。リフレッシュトークンが無効になると、サーバーは`gateway.log`に警告を出して停止状態になります。`hermes mcp login <server>`（またはDesktop/ダッシュボードの*Authorize*ボタン）で一度再認可すると、停止中のサーバーは次のプローブで新しいトークンを取り込みます。トークンファイルが変更されるまで、新しいセッションを開始しても拒否されたトークンは再試行されません。再試行するのは時間指定の自己プローブだけなので、無効になった認可がプロバイダーのトークンエンドポイントに大量のリクエストを送ることはありません。

### ツールセット

設定された各MCPサーバーは、少なくとも1つの登録されたツールを提供する場合、ランタイムのツールセットも作成します。

```text
mcp-<server>
```

これにより、MCPサーバーをツールセットのレベルで扱いやすくなります。

## セキュリティモデル

### stdioのenvフィルタリング

stdioサーバーについて、Hermesはあなたの完全なシェル環境をそのまま渡すことはしません。

明示的に設定された`env`に加え、安全なベースラインのみが渡されます。これにより、偶発的なシークレットの漏洩が減ります。

### 設定レベルの公開制御

新しいフィルタリングサポートはセキュリティ制御でもあります。
- モデルに見せたくない危険なツールを無効化する
- 機微なサーバーには最小限のホワイトリストだけを公開する
- その面を公開したくない場合はリソース/プロンプトのラッパーを無効化する

## ユースケースの例

### 最小限のissue管理機能を持つGitHubサーバー

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue]
      prompts: false
      resources: false
```

次のように使います。

```text
Show me open issues labeled bug, then draft a new issue for the flaky MCP reconnection behavior.
```

### 危険なアクションを削除したStripeサーバー

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

次のように使います。

```text
Look up the last 10 failed payments and summarize common failure reasons.
```

### 単一のプロジェクトルート向けのファイルシステムサーバー

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

次のように使います。

```text
Inspect the project root and explain the directory layout.
```

## トラブルシューティング

### MCPサーバーが接続しない

確認してください。

```bash
# Verify MCP deps are installed (already included in standard install)
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['mcp'], explicit=True)"

node --version
npx --version
```

その後、設定を確認してHermesを再起動してください。

`agent.log`の起動時サマリーには、登録されなかったすべてのサーバーが、記録された接続エラーとともに
名前付きで表示されるため、消去法で失敗しているサーバーを突き止める必要はありません。

```
MCP: registered 116 tool(s) from 4 server(s) (2 failed: github (Connection closed); notion (HTTP 401 from POST https://mcp.notion.com/mcp))
```

以前の失敗によるリトライのクールダウン中であるためにこのパスでスキップされたサーバーは、
`not attempted (in retry cooldown)`として表示されます。

### リモート（HTTP）サーバーが接続を拒否する

`hermes mcp test <name>`は、サーバーが実際に何を返したかを報告します。MCP SDKが
`Server returned an error response`としか言えない場合（本文がJSON-RPCエラーではない4xx/5xx）、Hermesは
HTTPステータス、リクエストしたURL、レスポンス本文の先頭を付け加えます。

```
Streamable HTTP: Server returned an error response (HTTP 400 from POST http://host:27200/mcp:
{"jsonrpc":"2.0","error":{"code":-32020,"message":"Unsupported MCP-Protocol-Version"}})
```

まずステータスと本文を読んでください。`initialize`のPOSTに対する`400`/`405`は通常、そのエンドポイントが
SSEのみに対応している（`transport: sse`を設定する）か、前段のプロキシがリクエストを拒否していることを意味します。
`401`/`403`はトークンまたはOAuthの認可が誤っていることを意味します。HTML本文は、URLがMCPエンドポイントではなく
Webページを指していることを意味します。`hermes logs --level debug`では、さらに各接続試行で使われた正確なエンドポイントが表示されます。

### サーバーのレスポンスが大きすぎる

Hermesは、1つのHTTPレスポンス、またはストリーミングレスポンスの1つのSSEイベントにつき最大10 MiBまで受け付けるため、
大きな`tools/list`カタログやツール結果（例えばエディターのスクリーンショット）も収まります。レガシーのSSE
トランスポート（`transport: sse`）では、MCP SDKが各イベントを1 MiBに制限しており、それを引き上げる方法はありません。
サーバーがStreamable HTTPにも対応している場合は、そのエントリから`transport: sse`を削除してください。

### ツールが表示されない

考えられる原因:
- サーバーの接続に失敗した
- 検出に失敗した
- フィルタ設定がそのツールを除外した
- そのサーバーにユーティリティ機能が存在しない
- サーバーが`enabled: false`で無効化されている

意図的にフィルタリングしている場合は、これは想定どおりです。

### なぜリソースやプロンプトのユーティリティが表示されなかったのか?

Hermesは現在、次の両方が真の場合にのみそれらのラッパーを登録するためです。
1. あなたの設定がそれらを許可している
2. サーバーセッションが実際にその機能をサポートしている

これは意図的なもので、ツールリストを正直に保ちます。

## 並列ツール呼び出し

デフォルトでは、MCPツールは逐次的に — 1つずつ — 実行されます。MCPサーバーが、並行して実行しても安全なツール（例: 読み取り専用のクエリ、独立したAPI呼び出し）を公開している場合は、並列実行をオプトインできます。

```yaml
mcp_servers:
  docs:
    command: "docs-server"
    supports_parallel_tool_calls: true
```

`supports_parallel_tool_calls`が`true`の場合、Hermesは組み込みの読み取り専用ツール（web_search、read_fileなど）と同じように、1つのツール呼び出しバッチ内でそのサーバーの複数のツールを同時に実行することがあります。

:::caution
並列呼び出しは、ツールを同時に実行しても安全なMCPサーバーに対してのみ有効にしてください。ツールが共有状態、ファイル、データベース、外部リソースの読み書きを行う場合は、この設定を有効にする前に読み書きの競合状態を確認してください。
:::

## MCPサンプリングのサポート

MCPサーバーは、`sampling/createMessage`プロトコルを介してHermesにLLM推論をリクエストできます。これにより、MCPサーバーはHermesに代わってテキストを生成するよう依頼できます — LLMの機能を必要とするが独自のモデルアクセスを持たないサーバーに便利です。

サンプリングは、すべてのMCPサーバーに対して**デフォルトで有効**です（MCP SDKがサポートしている場合）。サーバーごとに`sampling`キーの下で設定します。

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    sampling:
      enabled: true            # Enable sampling (default: true)
      model: "openai/gpt-4o"  # Override model for sampling requests (optional)
      max_tokens_cap: 4096     # Max tokens per sampling response (default: 4096)
      timeout: 30              # Timeout in seconds per request (default: 30)
      max_rpm: 10              # Rate limit: max requests per minute (default: 10)
      max_tool_rounds: 5       # Max tool-use rounds in sampling loops (default: 5)
      allowed_models: []       # Allowlist of model names the server may request (empty = any)
      log_level: "info"        # Audit log level: debug, info, or warning (default: info)
```

サンプリングハンドラには、暴走する使用を防ぐためのスライディングウィンドウ式レートリミッター、リクエストごとのタイムアウト、ツールループの深さ制限が含まれます。メトリクス（リクエスト数、エラー、使用トークン数）はサーバーインスタンスごとに追跡されます。

特定のサーバーでサンプリングを無効化するには:

```yaml
mcp_servers:
  untrusted_server:
    url: "https://mcp.example.com"
    sampling:
      enabled: false
```

## ツールのコストと支出上限 {#tool-costs-and-spend-limits}

一部のMCPサーバーは、呼び出しのたびに実際のお金やクレジットを消費します（画像、動画、3Dの生成、有料API）。サーバーエントリでは、各ツールのコストとチャットごとの予算を宣言できます。

```yaml
mcp_servers:
  renders:
    command: "render-mcp"
    trust: untrusted
    tool_costs:
      create_image: 20                                   # about 20 credits per call
      animate: {credits: [5, 20], note: "per animation"}  # a range, with a short note
      get_status: 0                                       # free; same as leaving it out
    spend_limit: 100     # per chat; 0 or absent = no limit
    cost_unit: credits   # label in prompts (default: credits)
```

- 承認プロンプトにはコストが表示されます: `Costs about 20 credits.`または`Costs 5-20 credits (per animation).`。さらに、チャットで何らかの支出があった時点で`This chat has used about 40 credits on 'renders' (limit 100).`が表示されます。
- Hermesはチャットごと・サーバーごとに累計を保持し、承認して開始したコスト付きの呼び出しごとに宣言された値（範囲の場合は上限値）を加算します。拒否またはブロックされた呼び出しは何も加算しません。サーバーが呼び出しを受け取る前に停止した場合でも加算され、これはより早く確認を求める方向に倒すためです。累計はメモリ上に保持されます。これは1つの会話の予算であり、ベンダーの請求額ではありません。
- コンテキスト圧縮や`/branch`によって会話が新しいセッションIDで継続される場合、累計は会話に引き継がれます。新しいチャットを開始すると（CLIでの`/new`、Desktopでの新しいチャット）新しい累計が始まります。メッセージングプラットフォームでは累計はチャットに属するため、そこでは`/new`の後も維持されます。各cron実行はそれぞれ独自の累計を持ちます。Hermesを再起動するとすべての累計がリセットされます。
- `spend_limit`がある場合、チャットの上限を超えることになるコスト付きの呼び出しは、それ以外では決して確認を求めない`trust: full`サーバーであっても事前に確認を求め、そのチャットでのそれ以降のコスト付きの呼び出しもすべて再度確認を求めます。このプロンプトはAllow OnceとDenyのみを提示し、そのツールに対する以前のSessionまたはAlwaysの回答によって抑制されることはありません。誰も回答できない場所（cronジョブ、`hermes chat -q`、承認者のいないゲートウェイ）では、呼び出しは実行されずに拒否され、モデルには再試行しないよう伝えられます。
- 上限未満では、`trust: full`サーバーは引き続き確認なしでコスト付きのツールを実行します。`trust: untrusted`サーバーでは、書き込み可能な呼び出しはすべて既に確認を求めており、そのプロンプトにコストが表示されるようになります。`readOnlyHint: true`のアノテーションが付いたツールは、予算上必要な場合を除き、確認なしのままです。
- [YOLO](../security.md#yolo-mode)（`/yolo`、`--yolo`）と`approvals.mode: off`は`trust: untrusted`のプロンプトをスキップしますが（`/yolo`はそのチャット、`--yolo`はそのプロセス、`approvals.mode: off`はそのプロファイルに対して）、支出上限はスキップしません。上限未満ではコスト付きの呼び出しは確認なしで実行され、上限を超えることになる呼び出しは引き続き確認を求めます（または、`approvals.mode: off`のcronジョブのように誰も回答できない場所では拒否されます）。
- 不正な形式の`tool_costs`、`spend_limit`、`cost_unit`があると、そのキーを挙げたエラーとともにサーバーがスキップされるため、宣言した予算が黙って破棄されることはありません。サーバーが提供していない`tool_costs`のキー（タイプミスや、Hermesの`mcp_<server>_`という表示用プレフィックスをコピーしてしまった場合）は、接続時に警告で名前が挙げられます。サーバー自身のツール名を使ってください。

[ポータブルプラグインパッケージ](../../developer-guide/plugins/index.md#portable-agent-plugins-v1-packages)は、そのサーバーについて同じテーブルを宣言できます。同じ名前の`mcp_servers`エントリを追加した場合（例えば`trust: full`を設定するため）でも、あなたのエントリがそれらのキー自体を設定しない限り、パッケージの`tool_costs`、`spend_limit`、`cost_unit`は引き続き適用されます。`spend_limit: 0`でパッケージの上限を無効にでき、`tool_costs: {}`でそのコストを破棄できます。

## MCPエリシテーションのサポート {#mcp-elicitation-support}

MCPサーバーは、`elicitation/create`プロトコル（mcp Python SDK ≥ 1.11.0）を介して、ツール呼び出しの途中でユーザーに構造化された入力を求めることができます。Hermesは**フォームモード**のエリシテーションを既存の承認の仕組み — CLI/TUIのインタラクティブなプロンプト、またはTelegramやSlackのようなゲートウェイプラットフォームの承認ボタン — を通じてルーティングするため、セッションがどこにあってもリクエストがあなたに届きます。各回答はその1つのリクエストだけを対象とするため、プロンプトはAllow OnceとDenyのみを提示します。YOLOと`approvals.mode: off`がこれに代わりに回答することはありません。サーバーは許可を求めているのではなく、あなたに質問しているからです。**URLモード**のエリシテーション（サーバーが外部URLを案内するもの）は、未サポートとして拒否されます。

エリシテーションは、サーバーごとに**デフォルトで有効**です。`elicitation`キーの下で設定します。

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    elicitation:
      enabled: true    # default: true
      timeout: 300     # seconds to wait for your answer (default: 300)
```

5分のデフォルトタイムアウトはゲートウェイの承認のデフォルトに合わせたもので、非同期の面にいるユーザーが、サーバーが諦める前に応答する時間を確保できます。サーバーごとのメトリクス（リクエスト、承諾、拒否、エラー）はハンドラ上で追跡されます。

## MCPイベント（実験的） {#mcp-events-experimental}

一部のMCPサーバーは、上流で何かが起きたとき（新しいメール、インシデント、フィードの新しい項目）に、
[MCPイベントの設計スケッチ](https://github.com/modelcontextprotocol/experimental-ext-triggers-events/blob/main/docs/design-sketch-proposal.md)の
`events/*`メソッドを通じてエージェントに通知できます。
この提案はまだワーキンググループのドラフトです。Hermesはその**webhook**配信モードを実装しています。
サーバーは各イベントをゲートウェイの[webhookリスナー](../messaging/webhooks.md)にPOSTし、各イベントは
他のwebhookルートと同じようにエージェントの実行を開始します。

サーバーごとにこれを有効にし、webhookリスナーの公開HTTPSアドレスをHermesに伝えてください（サーバーは
プレーンHTTPのコールバックを拒否するため、ポート8644の前段にTLS終端のリバースプロキシまたはトンネルを置いてください）。

```yaml
mcp_servers:
  ambush:
    url: "https://example.com/mcp"
    events: true          # registers the event utility tools for this server

platforms:
  webhook:
    enabled: true
    extra:
      public_url: "https://hermes.example.com"
```

その後、サーバーにはさらに3つのユーティリティツールが追加されます。

- `list_event_types`: サーバーが提供するイベントタイプと、それらに対する現在のサブスクリプション
- `subscribe_event`: イベントの引数、イベント到着時に行うこと
  （`instructions`、最大2000文字）、各実行の結果の送り先（`deliver`、
  `deliver_chat_id`）を指定してサブスクライブします。サブスクリプションは無人のエージェント実行を開始するため、
  すべてのサブスクリプションについて承認を求められます。承認には指示と送り先が完全な形で表示されます。
- `unsubscribe_event`: サブスクリプションをキャンセルします

各サブスクリプションは、独自の`whsec_`署名シークレットを持つ動的なwebhookルートとして保存されます（`hermes webhook list`で表示されます）。
認可とリプレイカーソルはその横の`mcp_event_subscriptions.json`に保持されるため、ルート自体が変わるのは、
異なる指示や異なる送り先で再サブスクライブしたときだけです。Hermesはサーバーの署名付き検証チャレンジに応答し、
`eventId`でイベントの重複を排除し、MCP接続が維持されている間は、サーバーが付与した有効期間が切れる前に
サブスクリプションを更新します（再接続のたびにも更新します）。`lazy: true`や
`idle_timeout_seconds`を使うと接続が常に維持されるとは限らないため、次の接続で再サブスクライブされるまで
サブスクリプションが失効することがあります。サーバーが恒久的に拒否した更新
（アクセスの取り消し、イベントタイプの削除、引数が無効になった）はルートを削除します。その他の
失敗はバックオフ付きで再試行されます。webhookダッシュボードでルートをオフにするとサブスクリプションは
失効し、再びオンにすると再サブスクライブされます。サーバーがサブスクリプションを終了した場合、
Hermesはルートを削除して`deliver`ターゲットに通知します。イベントは
`hermes gateway run`が実行されている間にのみ届きます。イベントのペイロードは、最大32,000文字の
JSONまでそのまま実行に渡され、切り詰められた場合は注記が付きます。イベントのペイロードは外部データです。
イベントの実行では、他のwebhookルートと同じ制限付きのwebhookツールセットが使われます。
リスナーのルートごとのレート制限（`platforms.webhook.extra.rate_limit`、デフォルトは1分あたり30回）は
イベントの配信にも適用されます。サーバーは拒否された配信を再試行しない場合があるため、頻繁に発生するイベントタイプでは
上限を引き上げてください。

まだサポートされていないもの: `poll`および`push`配信モード、そして
`gateway.multiplex_profiles`の下のセカンダリプロファイルから行うサブスクリプション。

## HermesをMCPサーバーとして実行する {#running-hermes-as-an-mcp-server}

Hermesは自身がMCPサーバー**になる**こともできます。これにより、Claude Code、Claude Desktop、Cursor、Codex、または任意のMCPクライアントが、メッセージングの会話を一覧表示して読み取り、新しいメッセージを待ち、Hermesが接続しているすべてのプラットフォームをまたいでメッセージを送信でき、さらに任意でHermes自身のWeb、ブラウザ、ビジョン、画像生成、スキル、TTSのツールを呼び出すこともできます。これは公式の**mcp-serve**プラグインとして提供されています。

```bash
hermes plugins install mcp-serve
```

```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp-serve"]
    }
  }
}
```

以前は`hermes mcp serve`として組み込まれていました。`"args": ["mcp", "serve"]`で設定されたMCPクライアントは、それを`["mcp-serve"]`に変更します。ツール名とパラメータは変わっておらず、`hermes mcp serve`は現在、新しい表記とインストール行を表示します。ツール、`--tools`モードとその安全上の制限については、[プラグインのREADME](https://github.com/NousResearch/hermes-official-plugins/tree/main/mcp-serve)に記載されています。

## 関連ドキュメント

- [HermesでMCPを使う](../../guides/use-mcp-with-hermes.md)
- [CLIコマンド](../../reference/cli-commands.md)
- [スラッシュコマンド](../../reference/slash-commands.md)
- [FAQ](../../reference/faq.md)
