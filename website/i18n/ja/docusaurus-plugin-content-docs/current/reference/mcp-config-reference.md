---
sidebar_position: 8
title: "MCP 設定リファレンス"
description: "Hermes Agent の MCP 設定キー、フィルタリングのセマンティクス、ユーティリティツールポリシーのリファレンス"
---

# MCP 設定リファレンス

このページは、メインの MCP ドキュメントへのコンパクトなリファレンスの補助です。

概念的なガイダンスについては、次を参照してください。
- [MCP (Model Context Protocol)](../user-guide/features/mcp.md)
- [Hermes で MCP を使う](../guides/use-mcp-with-hermes.md)

## ルート設定の形

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # stdio servers
    args: []
    env: {}

    # OR
    url: "..."          # HTTP servers
    headers: {}

    # Optional HTTP/SSE TLS settings:
    ssl_verify: true                # bool or path to a CA bundle (PEM)
    client_cert: "/path/to/cert.pem"  # mTLS client certificate (see below)
    # client_key: "/path/to/key.pem"  # optional, when key lives in a separate file

    enabled: true
    timeout: 120
    connect_timeout: 60
    supports_parallel_tool_calls: false
    tools:
      include: []
      exclude: []
      resources: true
      prompts: true
```

## サーバーキー

| キー | 型 | 適用対象 | 意味 |
|---|---|---|---|
| `command` | string | stdio | 起動する実行ファイル |
| `args` | list | stdio | サブプロセスの引数 |
| `env` | mapping | stdio | サブプロセスに渡される環境変数 |
| `url` | string | HTTP | リモート MCP エンドポイント |
| `headers` | mapping | HTTP | リモートサーバーリクエストのヘッダー |
| `ssl_verify` | bool または string | HTTP | TLS 検証。`true`（デフォルト）はシステムの CA を使用、`false` は検証を無効化（安全ではない）、または文字列でカスタム CA バンドル（PEM）へのパスを指定 |
| `client_cert` | string または list | HTTP | mTLS クライアント証明書。文字列 = 証明書 + 鍵を含む PEM ファイルへのパス。リスト `[cert, key]` = 別々のファイル。リスト `[cert, key, password]` = 暗号化された鍵 |
| `client_key` | string | HTTP | `client_cert` が文字列で、鍵が別ファイルにある場合のクライアント秘密鍵へのパス |
| `enabled` | bool | 両方 | false のときサーバーを完全にスキップ |
| `timeout` | number | 両方 | ツール呼び出しのタイムアウト（秒、デフォルト: `300`） |
| `connect_timeout` | number | 両方 | 初回接続のタイムアウト（秒、デフォルト: `60`） |
| `protocol` | string | 両方 | プロトコル世代のネゴシエーション: `auto`（デフォルト — まずレガシーの `initialize` ハンドシェイクを行い、サーバーがハンドシェイクをモダン専用として拒否した場合は 2026-07-28 の `server/discover` ステートレスプローブにフォールバック）、`stateless`（まず `server/discover` をプローブし、レガシーで 1 回リトライ）、`legacy`（ハンドシェイクのみ、フォールバックなし） |
| `supports_parallel_tool_calls` | bool | 両方 | このサーバーのツールを並行実行できるようにする |
| `skip_preflight` | bool | HTTP | HEAD/GET が MCP 以外の content type を返す有効な Streamable HTTP エンドポイントに対して、早期失敗の content-type プローブをバイパスする（デフォルト: `false`） |
| `transport` | string | HTTP | Streamable HTTP の代わりに SSE トランスポートを使うには `sse` に設定 |
| `keepalive_interval` | number | 両方 | 死活確認の ping 間隔（秒、下限は 5 秒）。HTTP のデフォルトは `180`。サーバーがアイドルセッションをすぐに GC する場合は、サーバーのセッション TTL より短く設定してください。stdio では省略時に keepalive が無効になります。明示的に有効にするには値を設定します |
| `lazy` | bool | 両方 | 起動時にディスク上のスキーマキャッシュからサーバーのツールを登録し、最初のツール呼び出し時にのみ起動／接続する（デフォルト: `false`）。キャッシュを埋めるには事前に 1 回のライブ接続が必要です。エントリが存在しないか古い場合は、通常の即時接続にフォールバックします。ステータス表示では、初回使用までサーバーがキャッシュ済みのツール数とともに `lazy` として表示されます |
| `idle_timeout_seconds` | number | stdio | 任意。アイドル時間経過後に stdio サーバーを再生成（`0` で無効）。`lifecycle:` マッピングの下に置くこともできます |
| `max_lifetime_seconds` | number | stdio | 任意。経過時間後に stdio サーバーを再生成（`0` で無効）。`lifecycle:` マッピングの下に置くこともできます |
| `tools` | mapping | 両方 | フィルタリングとユーティリティツールのポリシー |
| `auth` | string | HTTP | 認証方法。PKCE による OAuth 2.1 を有効にするには `oauth` に設定 |
| `sampling` | mapping | 両方 | サーバー起点の LLM リクエストポリシー（MCP ガイドを参照） |
| `elicitation` | mapping | 両方 | サーバー起点のユーザー入力リクエスト。`enabled`（デフォルト `true`）と `timeout`（秒、デフォルト `300`）。フォームモードのリクエストは承認サーフェスを経由し、URL モードは拒否されます（MCP ガイドを参照） |
| `events` | bool | 両方 | Webhook 配信による実験的な MCP イベントクライアント（デフォルト: `false`）。このサーバーに `list_event_types`、`subscribe_event`、`unsubscribe_event` を登録します。`platforms.webhook.extra.public_url`（HTTPS）と稼働中のゲートウェイが必要です。Hermes がサブスクリプションを更新するのはサーバーに接続している間だけなので、`lazy: true` や `idle_timeout_seconds` を使うと、次の接続で再購読されるまでサブスクリプションが失効することがあります。[MCP Events](../user-guide/features/mcp.md#mcp-events-experimental) を参照 |
| `trust` | string | 両方 | 信頼レベル: `full`（デフォルト）または `untrusted`。`untrusted` のサーバーでは、書き込み可能なすべてのツール呼び出し（`readOnlyHint: true` アノテーションのないツール）は、実行前に標準の承認サーフェスでユーザーの承認が必要です。プロンプトには Allow Once、Session（そのツールについて、チャットの残りの期間）、Always（`command_allowlist` に `mcp:<server>/<tool>` として書き込まれます。エントリを削除すると他のすべてのチャットで取り消されますが、Always と回答したチャットはそのチャットが終わるまで保持します）があり、それぞれ 1 つのツールのみを対象とし、サーバーの他のツールには及びません。[YOLO](../user-guide/security.md#yolo-mode) と `approvals.mode: off` はこのプロンプトをスキップします（`/yolo` はそのチャット、`--yolo` はそのプロセス、`approvals.mode: off` はそのプロファイルが対象）。ただし宣言された `spend_limit` は引き続き確認を求めます。`readOnlyHint` はサーバーが提供する*ヒント*です — 嘘をつくサーバーにできるのは、読み取り専用と主張するツールの承認をスキップすることだけで、追加のアクセス権を得ることはできません — したがって、完全に管理していないサーバーは `untrusted` にしてください。同じヒントによって、呼び出し中にトランスポートセッションが期限切れになった後に呼び出しを透過的にリトライするかどうかも決まります: リプレイされるのは `readOnlyHint: true` のツールだけで、アノテーションのない（書き込み可能な）ツールは `outcome_uncertain` エラーを返します — アイドルセッションを期限切れにする Streamable-HTTP サーバーでは、アイドル期間後の最初のアノテーションなしの呼び出しが失敗することがあり、再実行の前に確認が必要です。認識されない値は `untrusted` として扱われます（フェイルクローズ）。ポータブルパッケージは、`plugin.json` で自身のサーバーに `untrusted` を要求できます（[Portable Agent Plugins](../developer-guide/plugins/index.md#portable-agent-plugins-v1-packages) を参照）。同名の `config.yaml` エントリが常に優先されます |
| `tool_costs` | mapping | 両方 | 各ツールの消費量。MCP ツール名をキーとし、非負の数値、`[min, max]` の範囲、または `{credits: <number or range>, note: "per animation"}` を指定します。承認プロンプトにはコストとチャットの累計が表示されます。宣言されていないツールやコストがゼロのツールには影響しません。[Tool costs and spend limits](../user-guide/features/mcp.md#tool-costs-and-spend-limits) を参照 |
| `spend_limit` | number | 両方 | このサーバーのコスト付きツールに対するチャットごとの予算（`0`／未指定 = 制限なし）。チャットがこれを超えることになるコスト付きの呼び出しは、`trust: full` のサーバーであっても先に確認を求め、誰も回答できない場面（cron、`chat -q`）では拒否されます |
| `cost_unit` | string | 両方 | コストと合計に表示されるラベル（デフォルト: `credits`） |

## 環境変数の参照

サーバーエントリ内の任意の文字列値（`env`、`headers`、`args`、`url` など）では、`${VAR}` または Cursor 形式の SecretRef 形式 `${env:VAR}` で環境変数を参照できます — どちらも同じ変数に解決されるため、Cursor / Claude の設定からコピーした MCP スニペットはそのまま動作します。

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${env:GITHUB_TOKEN}"   # same as "${GITHUB_TOKEN}"
```

値はアクティブなプロファイルのシークレットスコープから解決されます（プロセスの環境にフォールバック）。したがって、シークレットは `~/.hermes/.env` に置いてください。未設定（または空）の変数はリテラルのプレースホルダーのまま残りますが、例外が 2 つあります。stdio サーバーの `env` エントリで、値全体が未設定の変数への 1 つの参照である場合（`GHIDRA_INSTALL_DIR: "${GHIDRA_INSTALL_DIR}"`）、そのエントリは存在しないものとして扱われます: サーバーにはプレースホルダーが渡されず、変数はその行がない場合にサーバーが持つ値のままになります（通常は未設定。`PATH`、`HOME`、`TERM` などの基本的な変数は、Hermes がすべてのサーバーに渡す値を保持します）。そのため、設定やプラグインは任意の変数を安全に転送できます。設定済みの変数は、たとえ値が `${...}` のように見えても常にそのまま渡されます。参照を含むだけの値（`"${OPT_DIR}/bin"`）は、引き続きプレースホルダー付きで渡されます。そしてリモートサーバーの `url` や `headers` では: リテラルの `${VAR}` を送ると確実に 401 になり、不正な認証情報のように見えるため、代わりに接続はフェイルクローズし、サーバーのステータスに変数名が示されます（`MCP server 'name': ${VAR} in url/headers is not set in this profile's .env or secret source`）。プロファイルの `.env`（またはそのシークレットソース）を修正してください。次の接続試行で参照が新たに展開されます。

### コンテキスト変数

環境変数に加えて、Cursor 形式のコンテキスト変数も展開されます（名前は大文字小文字を区別します）。

| 変数 | 解決先 |
|---|---|
| `${userHome}` | 現在のユーザーのホームディレクトリ |
| `${workspaceFolder}` | セッションのワークスペースルート（判明していればセッションのターミナルの cwd、そうでなければプロセスの cwd） |
| `${workspaceFolderBasename}` | `${workspaceFolder}` のベース名 |
| `${pathSeparator}` / `${/}` | OS のパス区切り文字（`os.sep`） |

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
    env:
      CACHE_DIR: "${userHome}${/}.cache${/}mcp"
```

その他の `${...}` 参照は、上記の環境変数の参照にフォールスルーします。

## `tools` ポリシーキー

| キー | 型 | 意味 |
|---|---|---|
| `include` | string または list | サーバーネイティブな MCP ツールをホワイトリスト化。エントリには正確な名前、または fnmatch 形式の glob（`*_radar_*`、`get_zones_*`）を指定できます |
| `exclude` | string または list | サーバーネイティブな MCP ツールをブラックリスト化。正確な名前／glob のセマンティクスは `include` と同じです |
| `resources` | bool 様 | `list_resources` + `read_resource` を有効／無効化 |
| `prompts` | bool 様 | `list_prompts` + `get_prompt` を有効／無効化 |

## フィルタリングのセマンティクス

### `include`

`include` が設定されている場合、それらのサーバーネイティブな MCP ツールのみが登録されます。

```yaml
tools:
  include: [create_issue, list_issues]
```

### `exclude`

`exclude` が設定され、`include` が設定されていない場合、それらの名前を除くすべてのサーバーネイティブな MCP ツールが登録されます。

```yaml
tools:
  exclude: [delete_customer]
```

### 優先順位

両方が設定されている場合、`include` が優先されます。

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

結果:
- `create_issue` は引き続き許可されます
- `delete_issue` は、`include` が優先されるため無視されます

## ユーティリティツールのポリシー

Hermes は、MCP サーバーごとにこれらのユーティリティラッパーを登録することがあります。

リソース:
- `list_resources`
- `read_resource`

プロンプト:
- `list_prompts`
- `get_prompt`

イベント（`events: true` のサーバーのみ）:
- `list_event_types`
- `subscribe_event`
- `unsubscribe_event`

### リソースを無効化する

```yaml
tools:
  resources: false
```

### プロンプトを無効化する

```yaml
tools:
  prompts: false
```

### ケイパビリティを考慮した登録

`resources: true` または `prompts: true` であっても、Hermes は MCP セッションが実際に対応するケイパビリティを公開している場合にのみ、それらのユーティリティツールを登録します。

したがって、これは正常です。
- プロンプトを有効にする
- しかしプロンプトのユーティリティが表示されない
- サーバーがプロンプトをサポートしていないため

イベントのユーティリティは例外です。MCP SDK はパース時にサーバーの `events` ケイパビリティを
削除してしまうため、Hermes はそれを確認できず、`events: true` が設定されていれば常にそれらを登録します。

## `enabled: false`

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

動作:
- 接続の試行なし
- 検出なし
- ツールの登録なし
- 設定は後で再利用するためにそのまま残ります

## 空の結果の動作

フィルタリングがすべてのサーバーネイティブツールを除去し、ユーティリティツールが1つも登録されない場合、Hermes はそのサーバーに対して空の MCP ランタイムツールセットを作成しません。

## 設定の例

### 安全な GitHub 許可リスト

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      resources: false
      prompts: false
```

### Stripe のブラックリスト

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### リソースのみのドキュメントサーバー

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      include: []
      resources: true
      prompts: false
```

### TLS クライアント証明書（mTLS）

クライアント証明書を必要とする HTTP/SSE サーバーでは、`client_cert`（および任意で `client_key`）を設定します。

```yaml
mcp_servers:
  # Combined cert + key in a single PEM file
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/secrets/mcp-client.pem"

  # Separate cert and key files
  partner_api:
    url: "https://mcp.partner.example.com/mcp"
    client_cert: "~/secrets/client.crt"
    client_key: "~/secrets/client.key"

  # Encrypted key with a passphrase (3-element list form)
  bank_api:
    url: "https://mcp.bank.example.com/mcp"
    client_cert: ["~/secrets/client.crt", "~/secrets/client.key", "my-passphrase"]

  # Custom CA bundle (private CA / self-signed server)
  lab_api:
    url: "https://mcp.lab.local/mcp"
    ssl_verify: "~/secrets/lab-ca.pem"
    client_cert: "~/secrets/lab-client.pem"
```

注意:
- パスは `~` の展開をサポートします。ファイルが存在しない場合は、接続時にサーバー単位のエラーメッセージとともに即座に失敗します。
- `ssl_verify: false` はサーバー証明書の検証を完全に無効にします。実際のサービスでは使用しないでください。
- Streamable HTTP と SSE の両方のトランスポートで動作します。

## 設定の再読み込み

MCP 設定を変更した後、次でサーバーを再読み込みします。

```text
/reload-mcp
```

## ツールの命名

サーバーネイティブな MCP ツールは次のようになります。

```text
mcp__<server>__<tool>
```

例:
- `mcp__github__create_issue`
- `mcp__filesystem__read_file`
- `mcp__my_api__query_data`

ユーティリティツールも同じプレフィックス付けのパターンに従います。
- `mcp__<server>__list_resources`
- `mcp__<server>__read_resource`
- `mcp__<server>__list_prompts`
- `mcp__<server>__get_prompt`

二重アンダースコアの区切り（`mcp__…__…`）は Claude Code、Codex、OpenCode で使われている慣例に合わせたもので、どちらかの構成要素にアンダースコアが含まれていても、サーバーとツールの境界を明確にします。

### 名前のサニタイズ

サーバー名とツール名の両方に含まれる、英字・数字・アンダースコア以外の文字（ハイフン、ドット、スペースなど）は、登録前にアンダースコアに置き換えられます。これにより、ツール名が LLM の関数呼び出し API に対して有効な識別子になります。

たとえば、`list-items.v2` というツールを公開する `my-api` という名前のサーバーは、次のようになります。

```text
mcp__my_api__list_items_v2
```

`include` / `exclude` フィルタを書くときはこれを念頭に置いてください。サニタイズ後のバージョンではなく、**元の** MCP ツール名（ハイフン／ドット付き）を使用してください。

## OAuth 2.1 認証

OAuth を必要とする HTTP サーバーの場合、サーバーエントリに `auth: oauth` を設定します。

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
```

動作:
- Hermes は MCP SDK の OAuth 2.1 PKCE フロー（メタデータ検出、クライアントの識別、トークン交換、リフレッシュ）を使用します
- 初回接続時に、認可のためのブラウザウィンドウが開きます
- トークンは `~/.hermes/mcp-tokens/<server>.json`（名前付きプロファイルでは `~/.hermes/profiles/<name>/mcp-tokens/`）に永続化され、セッションをまたいで再利用されます
- トークンのリフレッシュは自動です。再認可はリフレッシュが失敗した場合にのみ発生します
- HTTP/StreamableHTTP トランスポート（`url` ベースのサーバー）にのみ適用されます
- [多重化ゲートウェイ](/docs/user-guide/multi-profile-gateways)のもとでは、OAuth 接続がプロファイル間で共有されることはありません: `mcp_servers` のエントリが同一であっても、各プロファイルは自身のトークンで認証し、自身の接続を開きます

### デバイスコードログイン（RFC 8628） {#device-code-login-rfc-8628}

`device_authorization_endpoint` を公開している認可サーバーでは、Hermes を実行しているマシンの
ターミナルからデバイス認可を明示的に選択します。

```bash
hermes mcp login protected_api --flow device
```

表示された検証 URL を任意のデバイスで開き、表示されたユーザーコードを入力します。
Hermes は承認をポーリングし、`authorization_pending` と `slow_down` を尊重し、拒否または
期限切れで停止します。ブラウザは起動されず、コールバックリスナーも不要です。
`oauth.timeout` は承認の待機時間の上限（デフォルト 300 秒）で、コードの有効期間によっても制限されます。
サーバーの保護リソースメタデータに複数の認可サーバーが記載されている場合、デバイス
ログインはそれらを順に調べ、メタデータの issuer が公開されている URL と一致し、かつ
`device_code` グラントを提供する最初のものを使います（先頭に記載されたブラウザ専用のサーバーはスキップされます）。
唯一許容される差異は、MCP 機能ガイドの「OAuth-authenticated HTTP servers」で説明されているパススコープの形
（`https://host/path` として公開され、`/.well-known/oauth-authorization-server/path` のドキュメントが
`https://host` を示すサーバー）だけです。

サーバーで `oauth.flow: device` を設定すると、`hermes mcp login` と `hermes mcp reauth`
（`reauth --all` を含む）がデバイス認可を使うようになります。`login --flow browser` は 1 回のログインに限り
その設定を上書きします。ブラウザ PKCE は引き続きデフォルトです。サポートされていないメタデータの場合は、
黙って別のフローにフォールバックするのではなく、対処可能なエラーになります。

デバイスログインは、動的登録時にデバイスグラントとリフレッシュグラントを要求するか、
設定済みの `oauth.client_id`、`oauth.client_secret`、`oauth.token_endpoint_auth_method` を使います。
登録されたクライアントはデバイス認可を許可している必要があります。ブラウザ用の CIMD ドキュメントは使われません。
`oauth.scope` はデバイス認可リクエストで送信され、`oauth.user_agent` はトークンのポーリングにも
適用されます。トークン、登録情報、issuer メタデータはアクティブなプロファイルの
MCP トークンストアに保存され、既存のランタイムのリフレッシュ経路が再起動後にそれらを再利用します。
失敗したデバイスグラントが、以前に保存された認証情報を置き換えることはありません。

初回のデバイスログインはターミナル専用です: ダッシュボード／ブラウザのコールバックやバックグラウンドの再接続は
デバイス認可を開始しません。ログインコマンドは、ゲートウェイと **同じプロファイルとホスト** で
実行してください。使用可能なリフレッシュトークンがない状態でデバイスグラントが期限切れ／拒否された場合は、
再度明示的にログインする必要があります。

### クライアントの識別: CIMD と DCR {#client-identification-cimd-and-dcr}

Hermes は、**Client ID Metadata Document**（CIMD）で認可サーバーに自身を識別させます。これは MCP の `2026-07-28` 仕様が Dynamic Client Registration の代わりに採用した仕組みです。ドキュメントは
`https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json` で公開されており、その URL *自体*が `client_id` です — 認可サーバーはそれを取得して、Hermes の名前、ロゴ、許可されたリダイレクト URI を知ります。インストールごとに登録されるものはなく、ユーザー固有のものもありません。

最終的な選択は認可サーバーに委ねられます: SDK がドキュメントの URL を `client_id` として送るのは、サーバーがメタデータで `client_id_metadata_document_supported: true` を公開している場合だけで、それ以外の場合は以前とまったく同じように DCR で登録します。DCR は MCP 仕様で非推奨ですが、現在デプロイされているほぼすべてのサーバーが使っているのは依然として DCR です。

#### コールバックポート

ドキュメントは固定のループバックリダイレクト URI のセットを宣言しており、仕様では認可リクエストのリダイレクト URI がそのいずれかと*文字列として完全一致*することが求められます — そのため CIMD フローでは、Hermes が通常選ぶランダムな高位ポートを使えません。そこで Hermes はコールバックをポート `27890`–`27894` のいずれかに固定します。

この固定は、サーバーのケイパビリティが分かる前に選ぶ必要があります。リダイレクト URI はフローの開始時に固定される一方、サーバーのメタデータはフローの途中で届くからです。そのため Hermes は、CIMD を使う*可能性がある*すべてのフローでポートを固定し、それ以外ではランダムなポートに戻します。

- Hermes が以前に接続したことがあり、キャッシュされたメタデータが CIMD を公開していないサーバーは、これまでどおりランダムなポートを使い続けます。
- Hermes が一度も到達したことのないサーバーは、その初回ログインで固定ポートを使います。CIMD が使われ得る唯一の方法は推測することだからです。
- コールバックを他の場所に移すものがあれば、それも元に戻ります: 事前登録された `oauth.client_id`、`oauth.client_secret`、カスタムの `oauth.client_name` や `oauth.token_endpoint_auth_method`、`oauth.redirect_uri` や `oauth.redirect_port` の上書き、ダッシュボードやデスクトップから起動されたログイン、ディスク上の既存のクライアント登録、または 5 つのポートすべてが他のプロセスに使われている場合です。

固定された各ポートは選ばれた時点でバインドされ、ブラウザのリダイレクトが届くまで保持されるため、2 つの並行ログイン — 2 つ目のプロファイル、または同じプロセス内の別のサーバー — が同じリスナーに着地することはありません。

#### サーバーがドキュメントを拒否した場合

サーバーがドキュメントを取得したうえで*トークン*エンドポイントで拒否した場合（`invalid_client`）、Hermes はその拒否をログに記録し、`~/.hermes/mcp-tokens/<server>.cimd-off` に記録して、以降そのサーバーでは DCR を使います。

ドキュメントをまったく取得または検証できないサーバーは、代わりに*認可*エンドポイントで、リダイレクトが起こる前に中断します。そこには Hermes が観測できるシグナルがないため、ブラウザには invalid-client エラーが表示され、ログインは 5 分後にタイムアウトします。タイムアウトのメッセージはドキュメントを示し、`cimd: false` を案内します。`hermes mcp login <server>` を実行すると記録された拒否がクリアされるため、修正されたドキュメントにはもう一度チャンスが与えられます。

#### 任意のサーバーごとのキー

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      client_metadata_url: "https://example.com/my-cimd.json"  # self-hosted document
      cimd: false                                              # force DCR
      user_agent: "My-MCP-Client/1.0"                          # token-request User-Agent
```

`client_metadata_url` は、パスを持つ HTTPS URL（オリジンのみ、フラグメント、userinfo、`.`/`..` セグメントは不可）で、**リダイレクトなし**で `200` と `Content-Type: application/json` を返す必要があります — 認可サーバーは、取得時にリダイレクトをたどることを禁止されています。Hermes は引き続きコールバックを同じ `27890`–`27894` の範囲に固定するため、セルフホストのドキュメントは 10 個すべてのループバック URI（各ポートについて `http://127.0.0.1:<port>/callback` と `http://localhost:<port>/callback`）を宣言する必要があり、その `client_id` は自身の URL でなければなりません。

`user_agent` は、**トークンエンドポイントへのリクエストのみ**（認可コードの交換、リフレッシュ、デバイスフローのトークンポーリング）でデフォルトの `User-Agent` を置き換えます。指定しない場合、Hermes は `Hermes-Agent/<version>` を送ります — ヘッダーなしのリクエストを送ることはありません。WAF の背後にある認可サーバーは、そのようなリクエストに `403` を返すためです。MCP のトラフィックには決して適用されず、その他のトークンリクエストのヘッダーは設定できません。空または null の値は無視されます。交換が失敗すると、ステータスとエラー本文の短い（秘匿化された）抜粋が報告されるため（WAF の「Request blocked」ページなのか、issuer の `invalid_grant` なのか）、パケットキャプチャなしで原因が分かります。`hermes mcp login` は再認可の前に古いグラントとクライアント登録をクリアしますが、キャッシュされた認可サーバーのメタデータは保持するため、メタデータドキュメントを再取得できない場合でも、案内される認可 URL は検出されたものになります。

OAuth の検出と動的クライアント登録のリクエスト（`/.well-known/...` のメタデータドキュメントと `registration_endpoint` への POST）は、常に `User-Agent: Hermes-Agent/<version>` を付けて送られます。MCP SDK はこれらのリクエストをクライアントのデフォルトヘッダーなしで組み立てるため、WAF の背後にある認可サーバーはヘッダーなしのリクエストに `403` を返します — するとメタデータドキュメントが読めないように見え、登録は MCP ホスト上の推測された `/register` にフォールバックし、ログインは `Registration failed: 404` で失敗します。すべてのメタデータ取得が失敗した場合、エラーは登録フォールバック自体のエラーの前に、まずそれらのステータスを示すようになりました（`Could not read authorization-server metadata (403 from https://…/.well-known/oauth-authorization-server; …)`）。

## Add to Hermes リンク {#add-to-hermes-link}

MCP ベンダーやドキュメントは、ワンクリックの **「Add to Hermes」** ボタンを提供できます。これは Cursor の `cursor://anysphere.cursor-deeplink/mcp/install` スキームと同様に、事前入力されたサーバー設定で Hermes デスクトップアプリを開きます。

```text
hermes://mcp/install?name=NAME&config=BASE64
```

- `name` — サーバー名。`^[A-Za-z0-9._-]{1,64}$` に一致する必要があります。
- `config` — **base64url エンコードされた JSON** としてのサーバー設定オブジェクト（標準の base64 も受け付けます）。デコードされた JSON は、文字列の `url` フィールド（`http://`/`https://` のみ）または文字列の `command` フィールドを持つオブジェクトでなければならず、上記で説明した任意のサーバーキーを含めることができます。32KB を超えるペイロードは拒否されます。

例（JavaScript）:

```js
const config = { url: 'https://mcp.example.com/mcp' }
const link = `hermes://mcp/install?name=example&config=${btoa(JSON.stringify(config))
  .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')}`
```

リンクを開いただけで何かがインストールされることはありません: デスクトップアプリはサーバー名と整形された設定全体を表示する確認ダイアログを示し（ローカルプロセスを実行する `command` ベースのサーバーには追加の注意が表示されます）、ユーザーが明示的に確認する必要があります。既存のサーバー名が上書きされることはありません — ユーザーには名前の変更かキャンセルが求められます。
