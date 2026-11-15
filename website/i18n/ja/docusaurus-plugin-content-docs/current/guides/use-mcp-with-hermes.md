---
sidebar_position: 6
title: "Hermes で MCP を使う"
description: "MCP サーバーを Hermes Agent に接続し、ツールをフィルタリングし、実際のワークフローで安全に使うための実践ガイド"
---

# Hermes で MCP を使う

このページの Python 依存関係コマンドは、
[PM で準備されたソースチェックアウト](/docs/reference/package-management#developer-workflow) を前提としています。
依存関係を変更した後は、チェックアウトを再アクティベートして Hermes を再起動してください。

このガイドでは、日々のワークフローで Hermes Agent と MCP を実際にどう使うかを説明します。

機能ページが MCP とは何かを説明するものだとすれば、このガイドは MCP から素早く安全に価値を引き出す方法についてのものです。

## MCP はいつ使うべきか？

MCP を使うべきケース:
- ツールがすでに MCP 形式で存在し、ネイティブの Hermes ツールを作りたくない場合
- クリーンな RPC レイヤーを通じて、ローカルまたはリモートのシステムに対して Hermes を動作させたい場合
- サーバーごとの公開範囲をきめ細かく制御したい場合
- Hermes のコアを変更せずに、社内 API、データベース、社内システムに Hermes を接続したい場合

MCP を使うべきでないケース:
- 組み込みの Hermes ツールですでにその仕事を十分にこなせる場合
- サーバーが膨大で危険なツール群を公開しており、それをフィルタリングする準備ができていない場合
- ごく限定的な統合が 1 つ必要なだけで、ネイティブツールの方がシンプルかつ安全な場合

## メンタルモデル

MCP はアダプターレイヤーだと考えてください:

- Hermes はエージェントのままである
- MCP サーバーがツールを提供する
- Hermes は起動時またはリロード時にそれらのツールを検出する
- モデルは通常のツールと同じようにそれらを使える
- 各サーバーをどこまで見せるかをあなたが制御する

最後の点が重要です。優れた MCP の使い方とは、単に「すべてを接続する」ことではありません。「正しいものを、有用な最小限の範囲で接続する」ことです。

## ステップ 1: MCP サポートをインストールする

標準のインストールスクリプトで Hermes をインストールした場合、MCP サポートはすでに含まれています。PM は宣言された `all` エクストラを選択します。

エクストラなしでインストールしており、MCP を個別に追加する必要がある場合:

```bash
cd ~/.hermes/hermes-agent
python -c "import pm; pm.sync_venv(['mcp'], explicit=True)"
```

npm ベースのサーバーの場合、Node.js と `npx` が利用可能であることを確認してください。

多くの Python MCP サーバーでは、`uvx` が手軽なデフォルトとして適しています。

## ステップ 2: まず 1 つのサーバーを追加する

安全なサーバーを 1 つから始めましょう。

例: 1 つのプロジェクトディレクトリのみへのファイルシステムアクセス。

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

次に Hermes を起動します:

```bash
hermes chat
```

そして具体的なことを尋ねます:

```text
Inspect this project and summarize the repo layout.
```

## ステップ 3: MCP がロードされたことを確認する

MCP はいくつかの方法で確認できます:

- 設定済みの場合、Hermes のバナー/ステータスに MCP 統合が表示されるはず
- Hermes に利用可能なツールを尋ねる
- 設定変更後は `/reload-mcp` を使う
- サーバーの接続に失敗した場合はログを確認する
- シェルから `hermes mcp test <server>` を実行する — 接続して検出されたツールを一覧表示し、
  接続が完了すれば `0`、接続に失敗すれば `1`、サーバーが設定にない場合は `3`
  （`2` は argparse の使用法エラー）で終了するため、ヘルスプローブや cron のウォッチドッグは
  出力をパースする代わりに `$?` で分岐できます

実用的なテストプロンプト:

```text
Tell me which MCP-backed tools are available right now.
```

## ステップ 4: すぐにフィルタリングを始める

サーバーが多数のツールを公開している場合、後回しにしないでください。

### 例: 必要なものだけをホワイトリストに登録する

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
```

これは通常、機密性の高いシステムにとって最良のデフォルトです。

## WSL2: WSL 内の Hermes を Windows の Chrome にブリッジする {#wsl2-bridge-hermes-in-wsl-to-windows-chrome}

これは次のような場合の実用的なセットアップです:

- Hermes が WSL2 内で動作している
- 制御したいブラウザが、通常のサインイン済みの Windows 上の Chrome である
- WSL から `/browser connect` がうまくいかない、または信頼性に欠ける

このセットアップでは、Hermes は Chrome に直接接続 **しません**。代わりに:

- Hermes は WSL 内で動作する
- Hermes はローカルの stdio MCP サーバーを起動する
- その MCP サーバーは Windows interop（`cmd.exe` または `powershell.exe`）を通じて起動される
- MCP サーバーがあなたのライブな Windows Chrome セッションにアタッチする

メンタルモデル:

```text
Hermes (WSL) -> MCP stdio bridge -> Windows Chrome
```

### このモードが有用な理由

- 実際の Windows ブラウザのプロファイル、Cookie、ログインを維持できる
- Hermes はサポートされている Unix 環境（WSL2）にとどまる
- Hermes コアのブラウザトランスポートに頼るのではなく、ブラウザ制御が MCP ツールとして公開される

### 推奨サーバー

`chrome-devtools-mcp` を使います。

Windows の Chrome ですでに `chrome://inspect/#remote-debugging` からライブのリモートデバッグが有効になっている場合、WSL から次のように追加します:

```bash
hermes mcp add chrome-devtools-win --command cmd.exe --args /c npx -y chrome-devtools-mcp@latest --autoConnect --no-usage-statistics
```

サーバーを保存した後:

```bash
hermes mcp test chrome-devtools-win
```

次に、新しい Hermes セッションを開始するか、以下を実行します:

```text
/reload-mcp
```

### 典型的なプロンプト

ロードされると、Hermes は MCP プレフィックス付きのブラウザツールを直接使えます。例えば:

```text
调用 MCP 工具 mcp_chrome_devtools_win_list_pages，列出当前浏览器标签页。
```

### `/browser connect` が誤ったツールである場合

Hermes が WSL で動作し、Chrome が Windows で動作している場合、Chrome が開いていてデバッグ可能であっても `/browser connect` は失敗することがあります。

よくある理由:

- WSL が、Chrome が Windows 側ツールに公開しているのと同じホストローカルエンドポイントに到達できない
- 新しい Chrome のライブデバッグフローは、従来の `ws://localhost:9222` とは異なる
- `chrome-devtools-mcp` のような Windows 側ヘルパーからの方がブラウザにアタッチしやすい

そうしたケースでは、`/browser connect` は同一環境のセットアップ用に残し、WSL から Windows へのブラウザブリッジには MCP を使ってください。

### 既知の落とし穴

- MCP 経由で Windows の stdio 実行ファイルを使う場合は、`/mnt/c/Users/<you>` や `/mnt/c/workspace/...` のような Windows マウントパスから Hermes を起動してください。
- `/root` や `/home/...` から Hermes を起動すると、MCP サーバーの起動前に Windows が `UNC` カレントディレクトリの警告を出すことがあります。
- `chrome-devtools-mcp --autoConnect` がページの列挙中にタイムアウトする場合は、Chrome のバックグラウンド/フリーズ状態のタブを減らして再試行してください。

### 例: 危険なアクションをブラックリストに登録する

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### 例: ユーティリティラッパーも無効化する

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

## フィルタリングは実際に何に影響するのか？

Hermes で MCP が公開する機能には、2 つのカテゴリーがあります:

1. サーバーネイティブの MCP ツール
- 以下でフィルタリング:
  - `tools.include`
  - `tools.exclude`

2. Hermes が追加するユーティリティラッパー
- 以下でフィルタリング:
  - `tools.resources`
  - `tools.prompts`

### 目にする可能性のあるユーティリティラッパー

Resources:
- `list_resources`
- `read_resource`

Prompts:
- `list_prompts`
- `get_prompt`

これらのラッパーは、次の場合にのみ表示されます:
- 設定がそれらを許可しており、かつ
- MCP サーバーのセッションが実際にそれらの機能をサポートしている

そのため、Hermes はサーバーが resources/prompts を持っていないのに持っているかのように振る舞うことはありません。

## よくあるパターン

### パターン 1: ローカルプロジェクトアシスタント

境界の定まったワークスペース上で Hermes に推論させたい場合は、リポジトリローカルのファイルシステムや git サーバーに MCP を使います。

```yaml
mcp_servers:
  fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]

  git:
    command: "uvx"
    args: ["mcp-server-git", "--repository", "/home/user/project"]
```

良いプロンプト:

```text
Review the project structure and identify where configuration lives.
```

```text
Check the local git state and summarize what changed recently.
```

### パターン 2: Open Scaffold によるリポジトリネイティブの作業記録

リポジトリの永続的な AI 作業記録（ミッション、計画、エビデンスノート、ハンドオフパケット、レビュー/ゲートの結果）を Hermes に読ませたい場合は、[Open Scaffold](https://github.com/graphanov/open-scaffold) を使います。Hermes はエージェントのままであり、Open Scaffold はリポジトリローカルの記録のままです。

スキャフォールド済みの 1 つのリポジトリ用にサーバーを追加します:

```bash
hermes mcp add open_scaffold --command npx --args -y open-scaffold@latest mcp serve --repo /absolute/path/to/repo
hermes mcp test open_scaffold
```

次に、公開する範囲を読み取り中心に保ちます。`hermes mcp add` のプロンプトで `select` を選ぶか、後で `config.yaml` を編集します:

```yaml
mcp_servers:
  open_scaffold:
    command: "npx"
    args: ["-y", "open-scaffold@latest", "mcp", "serve", "--repo", "/absolute/path/to/repo"]
    tools:
      include:
        - list_plans
        - get_plan
        - get_mission
        - list_evidence
        - get_evidence
        - get_status
        - search_plans
        - list_amendments
        - get_handoff
        - analyze_loop
        - gate_loop
      prompts: false
```

良いプロンプト:

```text
Use the Open Scaffold MCP tools to compile the current handoff packet and tell me the next legal action.
```

```text
Inspect the active plans and evidence notes, then say whether this repo is ready for human review or needs another attempt.
```

境界に関する注意:

- Open Scaffold MCP はローカルファーストで、デフォルトでは読み取り専用です。
- 書き込みツールを使うには、サーバーを `--allow-write` 付きで起動する必要があります。Hermes に `.osc` ファイルを変更させたいと明示的に望むまでは有効にしないでください。
- Open Scaffold は作業を記録しゲートするものであり、Hermes にマージ、公開、デプロイ、ランタイムの起動を許可するものではありません。
- 再現可能なツールスキーマが必要な場合は、`@latest` の代わりに `open-scaffold@<version>` に固定してください。

### パターン 3: GitHub トリアージアシスタント

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      prompts: false
      resources: false
```

良いプロンプト:

```text
List open issues about MCP, cluster them by theme, and draft a high-quality issue for the most common bug.
```

```text
Search the repo for uses of _discover_and_register_server and explain how MCP tools are registered.
```

### パターン 4: 社内 API アシスタント

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      include: [list_customers, get_customer, list_invoices]
      resources: false
      prompts: false
```

良いプロンプト:

```text
Look up customer ACME Corp and summarize recent invoice activity.
```

ここは、exclude リストよりも厳格なホワイトリストの方がはるかに優れているような場面です。

### パターン 4: ドキュメント / ナレッジサーバー

一部の MCP サーバーは、直接的なアクションというよりも共有ナレッジアセットに近い prompts や resources を公開します。

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: true
      resources: true
```

良いプロンプト:

```text
List available MCP resources from the docs server, then read the onboarding guide and summarize it.
```

```text
List prompts exposed by the docs server and tell me which ones would help with incident response.
```

## チュートリアル: フィルタリングを含むエンドツーエンドのセットアップ

ここに実践的な進め方を示します。

### フェーズ 1: 厳格なホワイトリストで GitHub MCP を追加する

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
      prompts: false
      resources: false
```

Hermes を起動して尋ねます:

```text
Search the codebase for references to MCP and summarize the main integration points.
```

### フェーズ 2: 必要になったときだけ拡張する

後で issue の更新も必要になった場合:

```yaml
tools:
  include: [list_issues, create_issue, update_issue, search_code]
```

次にリロードします:

```text
/reload-mcp
```

### フェーズ 3: 異なるポリシーで 2 つ目のサーバーを追加する

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      prompts: false
      resources: false

  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]
```

これで Hermes はそれらを組み合わせられます:

```text
Inspect the local project files, then create a GitHub issue summarizing the bug you find.
```

ここが MCP が強力になる場面です: Hermes コアを変更せずに、複数システムにまたがるワークフローを実現できます。

## 安全な使い方の推奨事項

### 危険なシステムにはホワイトリストを優先する

金銭的、顧客対面、または破壊的なものについては:
- `tools.include` を使う
- 可能な限り最小のセットから始める

### 未使用のユーティリティを無効化する

モデルにサーバー提供の resources/prompts を閲覧させたくない場合は、オフにします:

```yaml
tools:
  resources: false
  prompts: false
```

### サーバーの範囲を狭く保つ

例:
- ホームディレクトリ全体ではなく、1 つのプロジェクトディレクトリにルート化されたファイルシステムサーバー
- 1 つのリポジトリを指す git サーバー
- デフォルトで読み取り中心のツール公開にした社内 API サーバー

### 設定変更後はリロードする

```text
/reload-mcp
```

以下を変更した後に実行してください:
- include/exclude リスト
- 有効化フラグ
- resources/prompts のトグル
- 認証ヘッダー / 環境変数

## 症状別トラブルシューティング

### 「サーバーは接続するが、期待したツールが見当たらない」

考えられる原因:
- `tools.include` によってフィルタリングされている
- `tools.exclude` によって除外されている
- `resources: false` または `prompts: false` でユーティリティラッパーが無効化されている
- サーバーが実際には resources/prompts をサポートしていない

### 「サーバーは設定したが何もロードされない」

確認事項:
- `enabled: false` が設定に残っていないか
- コマンド/ランタイムが存在するか（`npx`、`uvx` など）
- HTTP エンドポイントに到達可能か
- 認証用の環境変数やヘッダーが正しいか

### 「MCP サーバーが公開しているよりも少ないツールしか表示されないのはなぜ？」

それは、Hermes がサーバーごとのポリシーと機能を考慮した登録を尊重するようになったためです。これは想定どおりの動作であり、通常は望ましいことです。

### 「設定を削除せずに MCP サーバーを取り除くには？」

以下を使います:

```yaml
enabled: false
```

これで設定は残しつつ、接続と登録を防げます。

## 推奨される最初の MCP セットアップ

ほとんどのユーザーに適した最初のサーバー:
- ファイルシステム
- git
- GitHub
- fetch / ドキュメント系 MCP サーバー
- 1 つの限定的な社内 API

あまり適さない最初のサーバー:
- 破壊的なアクションが多く、フィルタリングのない巨大な業務システム
- 制約をかけられるほど十分に理解していないもの

## 関連ドキュメント

- [MCP（Model Context Protocol）](../user-guide/features/mcp.md)
- [FAQ](../reference/faq.md)
- [スラッシュコマンド](../reference/slash-commands.md)
