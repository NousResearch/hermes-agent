---
sidebar_position: 4
title: "MCP（Model Context Protocol）"
description: "MCP経由でHermes Agentを外部ツールサーバーに接続し、Hermesがロードするツールを厳密に制御する"
---

# MCP（Model Context Protocol）

MCPを使うと、Hermes Agentは外部ツールサーバーに接続でき、エージェントはHermes自体の外部に存在するツール — GitHub、データベース、ファイルシステム、ブラウザスタック、内部APIなど — を利用できます。

Hermesに、どこか別の場所に既に存在するツールを使わせたいと思ったことがあるなら、MCPは通常それを実現する最もクリーンな方法です。

## MCPで得られるもの

- 先にネイティブのHermesツールを書かなくても、外部のツールエコシステムにアクセスできる
- ローカルのstdioサーバーとリモートのHTTP MCPサーバーを同じ設定に含められる
- 起動時の自動的なツール検出と登録
- サーバーがサポートする場合、MCPのリソースとプロンプトに対するユーティリティラッパー
- サーバーごとのフィルタリングにより、実際にHermesに見せたいMCPツールだけを公開できる

## クイックスタート

1. MCPサポートをインストールします（標準のインストールスクリプトを使った場合は既に含まれています）。

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

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

## 基本的な設定リファレンス

Hermesは`~/.hermes/config.yaml`の`mcp_servers`の下からMCP設定を読み取ります。

### 共通キー

| キー | 型 | 意味 |
|---|---|---|
| `command` | 文字列 | stdio MCPサーバーの実行ファイル |
| `args` | リスト | stdioサーバーの引数 |
| `env` | マッピング | stdioサーバーに渡される環境変数 |
| `url` | 文字列 | HTTP MCPエンドポイント |
| `headers` | マッピング | リモートサーバー向けのHTTPヘッダー |
| `timeout` | 数値 | ツール呼び出しのタイムアウト |
| `connect_timeout` | 数値 | 初回接続のタイムアウト |
| `enabled` | bool | `false`の場合、Hermesはそのサーバーを完全にスキップする |
| `tools` | マッピング | サーバーごとのツールフィルタリングとユーティリティポリシー |

### 最小のstdioの例

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### 最小のHTTPの例

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

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

### サーバーツールをブラックリスト化する

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

除外されたもの以外のすべてのサーバーツールが登録されます。

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

### 動的ツール検出 {#dynamic-tool-discovery}

MCPサーバーは、`notifications/tools/list_changed`通知を送信することで、利用可能なツールが実行時に変化したことをHermesに通知できます。Hermesがこの通知を受け取ると、サーバーのツールリストを自動的に再取得してレジストリを更新します — 手動の`/reload-mcp`は不要です。

これは、機能が動的に変化するMCPサーバー（例えば、新しいデータベーススキーマがロードされたときにツールを追加するサーバーや、サービスがオフラインになったときにツールを削除するサーバー）に便利です。

このリフレッシュはロックで保護されているため、同じサーバーからの連続した通知が重複するリフレッシュを引き起こすことはありません。プロンプトとリソースの変更通知（`prompts/list_changed`、`resources/list_changed`）は受信されますが、まだ対応されていません。

### リロード

MCP設定を変更した場合は、次を使います。

```text
/reload-mcp
```

これは設定からMCPサーバーをリロードし、利用可能なツールリストをリフレッシュします。サーバー自身がプッシュするランタイムのツール変更については、上記の[動的ツール検出](#dynamic-tool-discovery)を参照してください。

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
# MCP依存関係がインストールされているか確認（標準インストールには既に含まれる）
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

node --version
npx --version
```

その後、設定を確認してHermesを再起動してください。

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

## MCPサンプリングのサポート

MCPサーバーは、`sampling/createMessage`プロトコルを介してHermesにLLM推論をリクエストできます。これにより、MCPサーバーはHermesに代わってテキストを生成するよう依頼できます — LLMの機能を必要とするが独自のモデルアクセスを持たないサーバーに便利です。

サンプリングは、すべてのMCPサーバーに対して**デフォルトで有効**です（MCP SDKがサポートしている場合）。サーバーごとに`sampling`キーの下で設定します。

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    sampling:
      enabled: true            # サンプリングを有効化（デフォルト: true）
      model: "openai/gpt-4o"  # サンプリングリクエスト用のモデルを上書き（任意）
      max_tokens_cap: 4096     # サンプリング応答ごとの最大トークン数（デフォルト: 4096）
      timeout: 30              # リクエストごとのタイムアウト秒数（デフォルト: 30）
      max_rpm: 10              # レート制限: 1分あたりの最大リクエスト数（デフォルト: 10）
      max_tool_rounds: 5       # サンプリングループ内のツール使用の最大ラウンド数（デフォルト: 5）
      allowed_models: []       # サーバーがリクエストできるモデル名の許可リスト（空 = 任意）
      log_level: "info"        # 監査ログレベル: debug、info、warning（デフォルト: info）
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

## HermesをMCPサーバーとして実行する {#running-hermes-as-an-mcp-server}

MCPサーバー**に**接続するのに加え、Hermesは自身がMCPサーバー**になる**こともできます。これにより、他のMCP対応エージェント（Claude Code、Cursor、Codex、または任意のMCPクライアント）がHermesのメッセージング機能を利用できます — 会話の一覧表示、メッセージ履歴の読み取り、接続済みのすべてのプラットフォームをまたいだメッセージ送信が可能です。

### これを使うべきとき

- Claude Code、Cursor、または別のコーディングエージェントに、Hermes経由でTelegram/Discord/Slackのメッセージを送受信させたい
- Hermesの接続済みメッセージングプラットフォームすべてに一度にブリッジする単一のMCPサーバーが欲しい
- 接続済みプラットフォームを持つHermesゲートウェイが既に稼働している

### クイックスタート

```bash
hermes mcp serve
```

これはstdio MCPサーバーを起動します。プロセスのライフサイクルは（あなたではなく）MCPクライアントが管理します。

### MCPクライアントの設定

MCPクライアントの設定にHermesを追加します。例えば、Claude Codeの`~/.claude/claude_desktop_config.json`では次のようになります。

```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

または、特定の場所にHermesをインストールした場合は次のようになります。

```json
{
  "mcpServers": {
    "hermes": {
      "command": "/home/user/.hermes/hermes-agent/venv/bin/hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

### 利用可能なツール

MCPサーバーは10個のツールを公開します。これはOpenClawのチャンネルブリッジの面に加えて、Hermes固有のチャンネルブラウザを揃えたものです。

| ツール | 説明 |
|------|-------------|
| `conversations_list` | アクティブなメッセージング会話を一覧表示する。プラットフォームでフィルタするか名前で検索する。 |
| `conversation_get` | セッションキーで1つの会話の詳細情報を取得する。 |
| `messages_read` | 会話の最近のメッセージ履歴を読み取る。 |
| `attachments_fetch` | 特定のメッセージから非テキストの添付ファイル（画像、メディア）を抽出する。 |
| `events_poll` | カーソル位置以降の新しい会話イベントをポーリングする。 |
| `events_wait` | 次のイベントが到着するまでロングポーリング / ブロックする（ほぼリアルタイム）。 |
| `messages_send` | プラットフォーム経由でメッセージを送信する（例: `telegram:123456`、`discord:#general`）。 |
| `channels_list` | すべてのプラットフォームにわたって利用可能なメッセージングターゲットを一覧表示する。 |
| `permissions_list_open` | このブリッジセッション中に観測された保留中の承認リクエストを一覧表示する。 |
| `permissions_respond` | 保留中の承認リクエストを許可または拒否する。 |

### イベントシステム

MCPサーバーには、Hermesのセッションデータベースをポーリングして新着メッセージを検出するライブイベントブリッジが含まれます。これにより、MCPクライアントは着信する会話をほぼリアルタイムで把握できます。

```
# 新しいイベントをポーリング（非ブロッキング）
events_poll(after_cursor=0)

# 次のイベントを待つ（タイムアウトまでブロック）
events_wait(after_cursor=42, timeout_ms=30000)
```

イベントタイプ: `message`、`approval_requested`、`approval_resolved`

イベントキューはインメモリで、ブリッジが接続したときに開始されます。より古いメッセージは`messages_read`を通じて利用できます。

### オプション

```bash
hermes mcp serve              # 通常モード
hermes mcp serve --verbose    # stderrにデバッグログを出力
```

### 仕組み

MCPサーバーは、Hermesのセッションストア（`~/.hermes/sessions/sessions.json`とSQLiteデータベース）から直接会話データを読み取ります。バックグラウンドスレッドがデータベースの新着メッセージをポーリングし、インメモリのイベントキューを維持します。メッセージの送信には、Hermesエージェント自身と同じ`send_message`インフラを使用します。

読み取り操作（会話の一覧表示、履歴の読み取り、イベントのポーリング）には、ゲートウェイが稼働している必要はありません。送信操作には、プラットフォームアダプターがアクティブな接続を必要とするため、ゲートウェイの稼働が必要です。

### 現在の制限

- stdioトランスポートのみ（HTTP MCPトランスポートはまだ未対応）
- mtime最適化されたDBポーリングによる約200ms間隔のイベントポーリング（ファイルが変更されていない場合は処理をスキップ）
- `claude/channel`プッシュ通知プロトコルはまだ未対応
- テキストのみの送信（`messages_send`経由でのメディア/添付ファイルの送信は不可）

## 関連ドキュメント

- [HermesでMCPを使う](/docs/guides/use-mcp-with-hermes)
- [CLIコマンド](/docs/reference/cli-commands)
- [スラッシュコマンド](/docs/reference/slash-commands)
- [FAQ](/docs/reference/faq)
