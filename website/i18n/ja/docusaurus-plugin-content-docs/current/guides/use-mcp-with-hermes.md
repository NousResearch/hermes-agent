---
sidebar_position: 6
title: "Hermes で MCP を使う"
description: "MCP サーバーを Hermes Agent に接続し、ツールをフィルタリングし、実際のワークフローで安全に使うための実践ガイド"
---

# Hermes で MCP を使う

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

標準のインストールスクリプトで Hermes をインストールした場合、MCP サポートはすでに含まれています（インストーラーは `uv pip install -e ".[all]"` を実行します）。

エクストラなしでインストールしており、MCP を個別に追加する必要がある場合:

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
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
このプロジェクトを調べて、リポジトリのレイアウトを要約してください。
```

## ステップ 3: MCP がロードされたことを確認する

MCP はいくつかの方法で確認できます:

- 設定済みの場合、Hermes のバナー/ステータスに MCP 統合が表示されるはず
- Hermes に利用可能なツールを尋ねる
- 設定変更後は `/reload-mcp` を使う
- サーバーの接続に失敗した場合はログを確認する

実用的なテストプロンプト:

```text
今この瞬間に利用可能な、MCP 由来のツールを教えてください。
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
プロジェクト構造をレビューして、設定がどこにあるかを特定してください。
```

```text
ローカルの git の状態を確認して、最近何が変わったかを要約してください。
```

### パターン 2: GitHub トリアージアシスタント

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
MCP に関するオープンな issue を一覧表示し、テーマ別にクラスタリングして、最も多いバグについて質の高い issue を起草してください。
```

```text
リポジトリで _discover_and_register_server の使用箇所を検索し、MCP ツールがどのように登録されるかを説明してください。
```

### パターン 3: 社内 API アシスタント

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
顧客 ACME Corp を調べて、最近の請求アクティビティを要約してください。
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
docs サーバーから利用可能な MCP リソースを一覧表示し、オンボーディングガイドを読んで要約してください。
```

```text
docs サーバーが公開している prompts を一覧表示し、インシデント対応に役立つものを教えてください。
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
コードベースで MCP への参照を検索し、主な統合ポイントを要約してください。
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
ローカルのプロジェクトファイルを調べ、見つけたバグを要約した GitHub issue を作成してください。
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

- [MCP（Model Context Protocol）](/docs/user-guide/features/mcp)
- [FAQ](/docs/reference/faq)
- [スラッシュコマンド](/docs/reference/slash-commands)
