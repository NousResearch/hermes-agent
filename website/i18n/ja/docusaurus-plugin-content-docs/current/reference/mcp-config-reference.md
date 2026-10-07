---
sidebar_position: 8
title: "MCP 設定リファレンス"
description: "Hermes Agent の MCP 設定キー、フィルタリングのセマンティクス、ユーティリティツールポリシーのリファレンス"
---

# MCP 設定リファレンス

このページは、メインの MCP ドキュメントへのコンパクトなリファレンスの補助です。

概念的なガイダンスについては、次を参照してください。
- [MCP (Model Context Protocol)](/docs/user-guide/features/mcp)
- [Hermes で MCP を使う](/docs/guides/use-mcp-with-hermes)

## ルート設定の形

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # stdio サーバー
    args: []
    env: {}

    # または
    url: "..."          # HTTP サーバー
    headers: {}

    enabled: true
    timeout: 120
    connect_timeout: 60
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
| `enabled` | bool | 両方 | false のときサーバーを完全にスキップ |
| `timeout` | number | 両方 | ツール呼び出しのタイムアウト |
| `connect_timeout` | number | 両方 | 初回接続のタイムアウト |
| `tools` | mapping | 両方 | フィルタリングとユーティリティツールのポリシー |
| `auth` | string | HTTP | 認証方法。PKCE による OAuth 2.1 を有効にするには `oauth` に設定 |
| `sampling` | mapping | 両方 | サーバー起点の LLM リクエストポリシー（MCP ガイドを参照） |

## `tools` ポリシーキー

| キー | 型 | 意味 |
|---|---|---|
| `include` | string または list | サーバーネイティブな MCP ツールをホワイトリスト化 |
| `exclude` | string または list | サーバーネイティブな MCP ツールをブラックリスト化 |
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

## 設定の再読み込み

MCP 設定を変更した後、次でサーバーを再読み込みします。

```text
/reload-mcp
```

## ツールの命名

サーバーネイティブな MCP ツールは次のようになります。

```text
mcp_<server>_<tool>
```

例:
- `mcp_github_create_issue`
- `mcp_filesystem_read_file`
- `mcp_my_api_query_data`

ユーティリティツールも同じプレフィックス付けのパターンに従います。
- `mcp_<server>_list_resources`
- `mcp_<server>_read_resource`
- `mcp_<server>_list_prompts`
- `mcp_<server>_get_prompt`

### 名前のサニタイズ

サーバー名とツール名の両方に含まれるハイフン（`-`）とドット（`.`）は、登録前にアンダースコアに置き換えられます。これにより、ツール名が LLM の関数呼び出し API に対して有効な識別子になります。

たとえば、`list-items.v2` というツールを公開する `my-api` という名前のサーバーは、次のようになります。

```text
mcp_my_api_list_items_v2
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
- Hermes は MCP SDK の OAuth 2.1 PKCE フロー（メタデータ検出、動的クライアント登録、トークン交換、リフレッシュ）を使用します
- 初回接続時に、認可のためのブラウザウィンドウが開きます
- トークンは `~/.hermes/mcp-tokens/<server>.json` に永続化され、セッションをまたいで再利用されます
- トークンのリフレッシュは自動です。再認可はリフレッシュが失敗した場合にのみ発生します
- HTTP/StreamableHTTP トランスポート（`url` ベースのサーバー）にのみ適用されます
