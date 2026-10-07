---
title: ブラウザ自動化
description: 複数のプロバイダー、CDP経由のローカルChrome、またはクラウドブラウザでブラウザを制御し、Web操作、フォーム入力、スクレイピングなどを行います。
sidebar_label: ブラウザ
sidebar_position: 5
---

# ブラウザ自動化

Hermes Agentには、複数のバックエンドオプションを備えた本格的なブラウザ自動化ツールセットが含まれています:

- **Browserbaseクラウドモード** — [Browserbase](https://browserbase.com) を介したマネージドクラウドブラウザとアンチボットツール
- **Browser Useクラウドモード** — 代替のクラウドブラウザプロバイダーとしての [Browser Use](https://browser-use.com)
- **Firecrawlクラウドモード** — スクレイピング機能を組み込んだクラウドブラウザを提供する [Firecrawl](https://firecrawl.dev)
- **Camofoxローカルモード** — ローカルでのアンチ検出ブラウジングのための [Camofox](https://github.com/jo-inc/camofox-browser)（Firefoxベースのフィンガープリント偽装）
- **CDP経由のローカルChrome** — `/browser connect` を使って、自分のChromeインスタンスにブラウザツールを接続
- **ローカルブラウザモード** — `agent-browser` CLIとローカルのChromiumインストールを利用

いずれのモードでも、エージェントはWebサイトのナビゲーション、ページ要素の操作、フォーム入力、情報抽出を行えます。

## 概要

ページは**アクセシビリティツリー**（テキストベースのスナップショット）として表現されるため、LLMエージェントに最適です。インタラクティブな要素には ref ID（`@e1`、`@e2` など）が付与され、エージェントはこれを使ってクリックや入力を行います。

主な機能:

- **マルチプロバイダーのクラウド実行** — Browserbase、Browser Use、またはFirecrawl — ローカルブラウザは不要
- **ローカルChrome連携** — CDP経由で稼働中のChromeにアタッチし、実際の手動操作を伴うブラウジングが可能
- **組み込みのステルス機能** — ランダムなフィンガープリント、CAPTCHA解決、レジデンシャルプロキシ（Browserbase）
- **セッション分離** — 各タスクが独自のブラウザセッションを取得
- **自動クリーンアップ** — 非アクティブなセッションはタイムアウト後に閉じられる
- **ビジョン分析** — スクリーンショット + AI分析による視覚的な理解

## セットアップ

:::tip Nous購読者の方へ
有料の [Nous Portal](https://portal.nousresearch.com) 購読をお持ちの場合、別途APIキーを用意することなく、**[Tool Gateway（ツールゲートウェイ）](tool-gateway.md)** を通じてブラウザ自動化を利用できます。`hermes model` または `hermes tools` を実行して有効化してください。
:::

### Browserbaseクラウドモード

Browserbaseのマネージドクラウドブラウザを使用するには、次を追加します:

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

認証情報は [browserbase.com](https://browserbase.com) で取得できます。

### Browser Useクラウドモード

Browser Useをクラウドブラウザプロバイダーとして使用するには、次を追加します:

```bash
# Add to ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

APIキーは [browser-use.com](https://browser-use.com) で取得できます。Browser UseはREST API経由でクラウドブラウザを提供します。BrowserbaseとBrowser Useの両方の認証情報が設定されている場合は、Browserbaseが優先されます。

### Firecrawlクラウドモード

Firecrawlをクラウドブラウザプロバイダーとして使用するには、次を追加します:

```bash
# Add to ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***
```

APIキーは [firecrawl.dev](https://firecrawl.dev) で取得できます。その後、ブラウザプロバイダーとしてFirecrawlを選択します:

```bash
hermes setup tools
# → Browser Automation → Firecrawl
```

オプション設定:

```bash
# Self-hosted Firecrawl instance (default: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# Session TTL in seconds (default: 300)
FIRECRAWL_BROWSER_TTL=600
```

### ハイブリッドルーティング: 公開URLにはクラウド、LAN/localhostにはローカル

クラウドプロバイダーが設定されている場合、Hermesはプライベート/ループバック/LANアドレス（`localhost`、`127.0.0.1`、`192.168.x.x`、`10.x.x.x`、`172.16-31.x.x`、`*.local`、`*.lan`、`*.internal`、IPv6ループバック `::1`、リンクローカル `169.254.x.x`）に解決されるURLに対して、**ローカルのChromiumサイドカー**を自動的に起動します。公開URLは同じ会話内で引き続きクラウドプロバイダーを使用します。

これにより、よくある「ローカルで開発しつつBrowserbaseを使っている」ワークフローが解決します — エージェントは `http://localhost:3000` にあるダッシュボードのスクリーンショットを撮りつつ、プロバイダーを切り替えたりSSRFガードを無効化したりすることなく `https://github.com` をスクレイピングできます。クラウドプロバイダーがプライベートURLを見ることはありません。

この機能は**デフォルトで有効**です。無効化する（すべてのURLが従来どおり設定済みのクラウドプロバイダーに送られる）には:

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

自動ルーティングを無効化すると、`browser.allow_private_urls: true` も設定しない限り、プライベートURLは `"Blocked: URL targets a private or internal address"` で拒否されます（これを設定するとクラウドプロバイダーがアクセスを試みますが、BrowserbaseなどはあなたのLANに到達できないため、通常は機能しません）。

要件: ローカルサイドカーは純粋なローカルモードと同じ `agent-browser` CLIを使用するため、これをインストールしておく必要があります（`hermes setup tools → Browser Automation` で自動インストールされます）。公開URLからプライベートアドレスへのナビゲーション後のリダイレクトは引き続きブロックされます（内部へのリダイレクトのトリックを使って、公開経路を介してLANに到達することはできません）。

### Camofoxローカルモード

[Camofox](https://github.com/jo-inc/camofox-browser) は、Camoufox（C++フィンガープリント偽装を備えたFirefoxフォーク）をラップするセルフホスト型のNode.jsサーバーです。クラウドへの依存なしに、ローカルでのアンチ検出ブラウジングを提供します。

```bash
# Clone the Camofox browser server first
git clone https://github.com/jo-inc/camofox-browser
cd camofox-browser

# Build and start with Docker using the default container settings
# (auto-detects arch: aarch64 on M1/M2, x86_64 on Intel)
make up

# Stop and remove the default container
make down

# Force a clean rebuild (for example, after upgrading VERSION/RELEASE)
make reset

# Just download binaries without building
make fetch

# Override arch or version explicitly
make up ARCH=x86_64
make up VERSION=135.0.1 RELEASE=beta.24
```

`make up` はデフォルトのコンテナをただちに起動します。より大きなNodeヒープ、VNC、永続的なプロファイルディレクトリといったカスタムのランタイム設定が必要な場合は、まずイメージをビルドしてから自分で実行します:

```bash
# Build the image without starting the default container
make build

# Start with persistence, VNC live view, and a larger Node heap
mkdir -p ~/.camofox-docker
docker run -d \
  --name camofox-browser \
  --restart unless-stopped \
  -p 9377:9377 \
  -p 6080:6080 \
  -p 5901:5900 \
  -e CAMOFOX_PORT=9377 \
  -e ENABLE_VNC=1 \
  -e VNC_BIND=0.0.0.0 \
  -e VNC_RESOLUTION=1920x1080 \
  -e MAX_OLD_SPACE_SIZE=2048 \
  -v ~/.camofox-docker:/root/.camofox \
  camofox-browser:135.0.1-aarch64
```

VNCを有効にすると、ブラウザはヘッド付きモードで実行され、ブラウザの `http://localhost:6080`（noVNC）でライブ表示を見られます。ネイティブのVNCクライアントを `localhost:5901` に接続することもできます。

すでに `make up` を実行している場合は、カスタムのコンテナを起動する前に、そのデフォルトコンテナを停止して削除してください:

```bash
make down
# then run the custom docker run command above
```

その後、`~/.hermes/.env` に次を設定します:

```bash
CAMOFOX_URL=http://localhost:9377
```

または `hermes tools` → Browser Automation → Camofox から設定します。

`CAMOFOX_URL` が設定されると、すべてのブラウザツールはBrowserbaseやagent-browserの代わりに、自動的にCamofoxを経由してルーティングされます。

#### 永続的なブラウザセッション

デフォルトでは、各Camofoxセッションはランダムなアイデンティティを取得します — Cookieやログインはエージェントの再起動をまたいで保持されません。永続的なブラウザセッションを有効にするには、`~/.hermes/config.yaml` に次を追加します:

```yaml
browser:
  camofox:
    managed_persistence: true
```

その後、新しい設定が反映されるようにHermesを完全に再起動してください。

:::warning ネストされたパスが重要
Hermesは `browser.camofox.managed_persistence` を読み取ります。**トップレベルの** `managed_persistence` ではありません。よくある間違いは次のように書くことです:

```yaml
# ❌ Wrong — Hermes ignores this
managed_persistence: true
```

フラグが誤ったパスに置かれると、Hermesは静かにランダムな一時的 `userId` にフォールバックし、ログイン状態はセッションごとに失われます。
:::

##### Hermesが行うこと
- プロファイルにスコープされた決定論的な `userId` をCamofoxに送信し、サーバーがセッションをまたいで同じFirefoxプロファイルを再利用できるようにします。
- クリーンアップ時にサーバー側のコンテキスト破棄をスキップするため、Cookieとログインがエージェントのタスク間で保持されます。
- `userId` をアクティブなHermesプロファイルにスコープするため、異なるHermesプロファイルは異なるブラウザプロファイルを取得します（プロファイル分離）。

##### Hermesが行わないこと
- Camofoxサーバーに永続化を強制することはありません。Hermesは安定した `userId` を送信するだけで、サーバーがその `userId` を永続的なFirefoxプロファイルディレクトリにマッピングして対応する必要があります。
- お使いのCamofoxサーバーのビルドがすべてのリクエストを一時的なものとして扱う場合（例えば、保存されたプロファイルを読み込まずに常に `browser.newContext()` を呼び出す場合）、Hermesはそれらのセッションを永続化できません。userIdベースのプロファイル永続化を実装したCamofoxビルドを実行していることを確認してください。

##### 動作確認

1. HermesとCamofoxサーバーを起動します。
2. ブラウザタスクでGoogle（または任意のログインサイト）を開き、手動でサインインします。
3. ブラウザタスクを通常どおり終了します。
4. 新しいブラウザタスクを開始します。
5. 同じサイトをもう一度開きます — まだサインインしているはずです。

ステップ5でログアウトされる場合、Camofoxサーバーが安定した `userId` を尊重していません。設定のパスを再確認し、`config.yaml` を編集した後にHermesを完全に再起動したことを確認し、お使いのCamofoxサーバーのバージョンがユーザーごとの永続プロファイルをサポートしていることを検証してください。

##### 状態の保存場所

Hermesは、プロファイルにスコープされたディレクトリ `~/.hermes/browser_auth/camofox/`（デフォルト以外のプロファイルでは `$HERMES_HOME` 配下の同等のディレクトリ）から安定した `userId` を導出します。実際のブラウザプロファイルデータは、その `userId` をキーとしてCamofoxサーバー側に保存されます。永続プロファイルを完全にリセットするには、Camofoxサーバー上でクリアし、対応するHermesプロファイルの状態ディレクトリを削除してください。

#### VNCライブ表示

Camofoxがヘッド付きモード（表示可能なブラウザウィンドウあり）で実行されると、ヘルスチェックのレスポンスにVNCポートが公開されます。Hermesはこれを自動的に検出し、ナビゲーションのレスポンスにVNC URLを含めるため、エージェントはブラウザをライブで見るためのリンクを共有できます。

### CDP経由のローカルChrome（`/browser connect`）

クラウドプロバイダーの代わりに、Chrome DevTools Protocol（CDP）を介して、自分の稼働中のChromeインスタンスにHermesのブラウザツールをアタッチできます。これは、エージェントの動作をリアルタイムで見たい場合、自分のCookie/セッションを必要とするページを操作したい場合、またはクラウドブラウザのコストを避けたい場合に便利です。

:::note
`/browser connect` は**対話型CLIのスラッシュコマンド**です — ゲートウェイによってディスパッチされません。WebUI、Telegram、Discord、その他のゲートウェイチャットの中で実行しようとすると、メッセージはプレーンテキストとしてエージェントに送られ、コマンドは実行されません。ターミナルからHermesを起動し（`hermes` または `hermes chat`）、そこで `/browser connect` を実行してください。
:::

CLIでは、次を使用します:

```
/browser connect              # Connect to Chrome at ws://localhost:9222
/browser connect ws://host:port  # Connect to a specific CDP endpoint
/browser status               # Check current connection
/browser disconnect            # Detach and return to cloud/local mode
```

Chromeがリモートデバッグ付きでまだ起動していない場合、Hermesは `--remote-debugging-port=9222` で自動起動を試みます。

:::tip
CDPを有効にしてChromeを手動で起動するには、専用の user-data-dir を使ってください。そうすれば、通常のプロファイルでChromeがすでに起動していても、デバッグポートが実際に立ち上がります:

```bash
# Linux
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# macOS
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &
```

その後、Hermes CLIを起動して `/browser connect` を実行します。

**なぜ `--user-data-dir` が必要か？** これがないと、通常のChromeインスタンスがすでに起動している状態でChromeを起動すると、たいてい既存のプロセス上で新しいウィンドウが開きます — そしてその既存プロセスは `--remote-debugging-port` で起動されていないため、ポート9222は決して開きません。専用の user-data-dir を使うと、デバッグポートが実際にリッスンする新しいChromeプロセスが強制的に立ち上がります。`--no-first-run --no-default-browser-check` は、新しいプロファイルの初回起動ウィザードをスキップします。
:::

CDP経由で接続すると、すべてのブラウザツール（`browser_navigate`、`browser_click` など）は、クラウドセッションを立ち上げる代わりに、あなたのライブなChromeインスタンス上で動作します。

### WSL2 + Windows Chrome: `/browser connect` よりMCPを優先 {#wsl2--windows-chrome-prefer-mcp-over-browser-connect}

HermesがWSL2内で実行されていて、制御したいChromeウィンドウがWindowsホスト上で実行されている場合、`/browser connect` は最善の方法ではないことがよくあります。

理由:

- `/browser connect` は、Hermes自身が利用可能なCDPエンドポイントに到達できることを前提としています
- 最近のChromeのライブデバッグセッションは、しばしばホストローカルなエンドポイントを公開しますが、それは従来の `9222` ポートと同じようにはWSLから直接到達できません
- Windows Chromeがデバッグ可能な場合でも、最もクリーンな統合は、Windows側のブラウザMCPサーバーにChromeをアタッチさせ、HermesにそのMCPサーバーと通信させることである場合が多いです

そのセットアップには、Hermes MCPサポートを通じた `chrome-devtools-mcp` を優先してください。

実用的なセットアップについては、MCPガイドを参照してください:

- [HermesでMCPを使う](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

### ローカルブラウザモード

クラウドの認証情報を一切設定せず、`/browser connect` も使わない場合でも、Hermesは `agent-browser` で駆動されるローカルのChromiumインストールを通じてブラウザツールを利用できます。

### オプションの環境変数

```bash
# Residential proxies for better CAPTCHA solving (default: "true")
BROWSERBASE_PROXIES=true

# Advanced stealth with custom Chromium — requires Scale Plan (default: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# Session reconnection after disconnects — requires paid plan (default: "true")
BROWSERBASE_KEEP_ALIVE=true

# Custom session timeout in milliseconds (default: project default)
# Examples: 600000 (10min), 1800000 (30min)
BROWSERBASE_SESSION_TIMEOUT=600000

# Inactivity timeout before auto-cleanup in seconds (default: 120)
BROWSER_INACTIVITY_TIMEOUT=120
```

### agent-browser CLIのインストール

```bash
npm install -g agent-browser
# Or install locally in the repo:
npm install
```

:::info
`browser` ツールセットは、設定の `toolsets` リストに含めるか、`hermes config set toolsets '["hermes-cli", "browser"]'` で有効化する必要があります。
:::

## 利用可能なツール

### `browser_navigate`

URLに移動します。他のどのブラウザツールよりも先に呼び出す必要があります。Browserbaseセッションを初期化します。

```
Navigate to https://github.com/NousResearch
```

:::tip
単純な情報取得には、`web_search` または `web_extract` を優先してください — そのほうが高速で安価です。ブラウザツールは、ページを**操作**する必要があるとき（ボタンのクリック、フォーム入力、動的コンテンツの処理）に使用してください。
:::

### `browser_snapshot`

現在のページのアクセシビリティツリーのテキストベースのスナップショットを取得します。`browser_click` および `browser_type` で使うための、`@e1`、`@e2` のような ref ID を持つインタラクティブ要素を返します。

- **`full=false`**（デフォルト）: インタラクティブ要素のみを表示するコンパクトビュー
- **`full=true`**: ページコンテンツ全体

8000文字を超えるスナップショットは、自動的にLLMによって要約されます。

### `browser_click`

スナップショットの ref ID で識別される要素をクリックします。

```
Click @e5 to press the "Sign In" button
```

### `browser_type`

入力フィールドにテキストを入力します。最初にフィールドをクリアしてから、新しいテキストを入力します。

```
Type "hermes agent" into the search field @e3
```

### `browser_scroll`

ページを上下にスクロールして、さらにコンテンツを表示します。

```
Scroll down to see more results
```

### `browser_press`

キーボードのキーを押します。フォームの送信やナビゲーションに便利です。

```
Press Enter to submit the form
```

サポートされるキー: `Enter`、`Tab`、`Escape`、`ArrowDown`、`ArrowUp` など。

### `browser_back`

ブラウザ履歴の前のページに戻ります。

### `browser_get_images`

現在のページのすべての画像を、URLとalt textとともに一覧表示します。分析する画像を見つけるのに便利です。

### `browser_vision`

スクリーンショットを撮り、ビジョンAIで分析します。テキストスナップショットでは重要な視覚情報が捉えられない場合に使用します — 特にCAPTCHA、複雑なレイアウト、視覚的な検証課題に便利です。

スクリーンショットは永続的に保存され、AI分析と並んでファイルパスが返されます。メッセージングプラットフォーム（Telegram、Discord、Slack、WhatsApp）では、エージェントにスクリーンショットの共有を依頼できます — `MEDIA:` の仕組みを介して、ネイティブの写真添付として送信されます。

```
What does the chart on this page show?
```

スクリーンショットは `~/.hermes/cache/screenshots/` に保存され、24時間後に自動的にクリーンアップされます。

### `browser_console`

現在のページからブラウザコンソールの出力（log/warn/errorメッセージ）とキャッチされていないJavaScript例外を取得します。アクセシビリティツリーに現れないサイレントなJSエラーを検出するのに不可欠です。

```
Check the browser console for any JavaScript errors
```

`clear=True` を使うと読み取り後にコンソールをクリアできるため、以降の呼び出しでは新しいメッセージのみが表示されます。

`browser_console` は、`expression` 引数を指定して呼び出すとJavaScriptも評価します — DevToolsコンソールと同じ形で、結果はパースされて返されます（JSONシリアライズされたオブジェクトはdictになり、プリミティブ値はプリミティブのまま）。

```
browser_console(expression="document.querySelector('h1').textContent")
browser_console(expression="JSON.stringify(performance.timing)")
```

現在のセッションでCDPスーパーバイザーがアクティブな場合（CDP対応バックエンドに対して `browser_navigate` を実行した任意のセッションでは一般的です）、評価はスーパーバイザーの永続的なWebSocket上で実行されます — サブプロセスの起動コストがありません。そうでない場合は、標準のagent-browser CLI経路にフォールスルーします。動作はどちらでも同一で、変わるのはレイテンシだけです。

### `browser_cdp`

生のChrome DevTools Protocolパススルー — 他のツールでカバーされないブラウザ操作のための回避策です。ネイティブダイアログの処理、iframeスコープの評価、Cookie/ネットワーク制御、またはエージェントが必要とする任意のCDP動詞に使用します。

**セッション開始時にCDPエンドポイントに到達可能な場合にのみ利用できます** — つまり `/browser connect` が稼働中のChromeにアタッチしているか、`config.yaml` に `browser.cdp_url` が設定されている場合です。デフォルトのローカルagent-browserモード、Camofox、およびクラウドプロバイダー（Browserbase、Browser Use、Firecrawl）は、現在このツールにCDPを公開していません — クラウドプロバイダーにはセッションごとのCDP URLがありますが、ライブセッションのルーティングは今後の対応です。

**CDPメソッドリファレンス:** https://chromedevtools.github.io/devtools-protocol/ — エージェントは特定のメソッドのページを `web_extract` して、パラメータと戻り値の形を調べることができます。

よくあるパターン:

```
# List tabs (browser-level, no target_id)
browser_cdp(method="Target.getTargets")

# Handle a native JS dialog on a tab
browser_cdp(method="Page.handleJavaScriptDialog",
            params={"accept": true, "promptText": ""},
            target_id="<tabId>")

# Evaluate JS in a specific tab
browser_cdp(method="Runtime.evaluate",
            params={"expression": "document.title", "returnByValue": true},
            target_id="<tabId>")

# Get all cookies
browser_cdp(method="Network.getAllCookies")
```

ブラウザレベルのメソッド（`Target.*`、`Browser.*`、`Storage.*`）は `target_id` を省略します。ページレベルのメソッド（`Page.*`、`Runtime.*`、`DOM.*`、`Emulation.*`）は `Target.getTargets` から得た `target_id` を必要とします。各ステートレスな呼び出しは独立しています — 呼び出し間でセッションは保持されません。

**クロスオリジンのiframe:** `frame_id`（`browser_snapshot.frame_tree.children[]` のうち `is_oopif=true` のもの）を渡すと、CDP呼び出しがそのiframe用のスーパーバイザーのライブセッションを経由してルーティングされます。これが、Browserbaseでクロスオリジンiframe内の `Runtime.evaluate` が機能する仕組みで、そこではステートレスなCDP接続だと署名付きURLの期限切れに当たってしまいます。例:

```
browser_cdp(
  method="Runtime.evaluate",
  params={"expression": "document.title", "returnByValue": True},
  frame_id="<frame_id from browser_snapshot>",
)
```

同一オリジンのiframeに `frame_id` は不要です — 代わりに、トップレベルの `Runtime.evaluate` から `document.querySelector('iframe').contentDocument` を使ってください。

### `browser_dialog`

ネイティブのJSダイアログ（`alert` / `confirm` / `prompt` / `beforeunload`）に応答します。このツールが存在する前は、ダイアログがページのJavaScriptスレッドを静かにブロックし、後続の `browser_*` 呼び出しがハングまたは例外を投げていました。現在は、エージェントが `browser_snapshot` の出力で保留中のダイアログを確認し、明示的に応答します。

**ワークフロー:**
1. `browser_snapshot` を呼び出します。ダイアログがページをブロックしている場合、`pending_dialogs: [{"id": "d-1", "type": "alert", "message": "..."}]` として表示されます。
2. `browser_dialog(action="accept")` または `browser_dialog(action="dismiss")` を呼び出します。`prompt()` ダイアログには、応答を渡すために `prompt_text="..."` を指定します。
3. 再スナップショット — `pending_dialogs` は空になり、ページのJSスレッドが再開しています。

**検出は自動的に行われます** — 永続的なCDPスーパーバイザー（タスクごとに1つのWebSocketで、Page/Runtime/Targetイベントを購読）を介して行われます。スーパーバイザーはスナップショットに `frame_tree` フィールドも埋め込むため、エージェントはクロスオリジン（OOPIF）iframeを含む現在のページのiframe構造を確認できます。

**対応マトリクス:**

| バックエンド | `pending_dialogs` による検出 | 応答（`browser_dialog` ツール） |
|---|---|---|
| `/browser connect` または `browser.cdp_url` 経由のローカルChrome | ✓ | ✓ 完全なワークフロー |
| Browserbase | ✓ | ✓ 完全なワークフロー（注入されたXHRブリッジ経由） |
| Camofox / デフォルトのローカルagent-browser | ✗ | ✗（CDPエンドポイントなし） |

**Browserbaseでの仕組み。** BrowserbaseのCDPプロキシは、実際のネイティブダイアログをサーバー側で約10ms以内に自動的に却下するため、`Page.handleJavaScriptDialog` を使えません。スーパーバイザーは `Page.addScriptToEvaluateOnNewDocument` を介して小さなスクリプトを注入し、`window.alert`/`confirm`/`prompt` を同期XHRでオーバーライドします。それらのXHRを `Fetch.enable` を介してインターセプトします — ページのJSスレッドは、エージェントの応答とともに `Fetch.fulfillRequest` を呼び出すまでXHRでブロックされたままになります。`prompt()` の戻り値は、変更されることなくページのJSに往復して戻ります。

**ダイアログポリシー**は、`config.yaml` の `browser.dialog_policy` の下で設定します:

| ポリシー | 動作 |
|--------|------|
| `must_respond`（デフォルト） | キャプチャし、スナップショットに表示し、明示的な `browser_dialog()` 呼び出しを待ちます。`browser.dialog_timeout_s`（デフォルト300秒）後に安全のため自動却下するため、バグのあるエージェントが永久に停止することはありません。 |
| `auto_dismiss` | キャプチャし、即座に却下します。エージェントは引き続き `browser_state` 履歴でダイアログを確認できますが、対応する必要はありません。 |
| `auto_accept` | キャプチャし、即座に承諾します。アグレッシブな `beforeunload` プロンプトのあるページを移動する際に便利です。 |

`browser_snapshot.frame_tree` 内の**フレームツリー**は、広告の多いページでペイロードを抑えるため、30フレームかつOOPIF深さ2に制限されます。制限に達したときは `truncated: true` フラグが表示され、完全なツリーが必要なエージェントは `browser_cdp` を `Page.getFrameTree` とともに使用できます。

## 実用的な例

### Webフォームへの入力

```
User: Sign up for an account on example.com with my email john@example.com

Agent workflow:
1. browser_navigate("https://example.com/signup")
2. browser_snapshot()  → sees form fields with refs
3. browser_type(ref="@e3", text="john@example.com")
4. browser_type(ref="@e5", text="SecurePass123")
5. browser_click(ref="@e8")  → clicks "Create Account"
6. browser_snapshot()  → confirms success
```

### 動的コンテンツのリサーチ

```
User: What are the top trending repos on GitHub right now?

Agent workflow:
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → reads trending repo list
3. Returns formatted results
```

## セッション記録

ブラウザセッションをWebM動画ファイルとして自動的に記録します:

```yaml
browser:
  record_sessions: true  # default: false
```

有効にすると、最初の `browser_navigate` で記録が自動的に開始され、セッションが閉じるときに `~/.hermes/browser_recordings/` に保存されます。ローカルモードとクラウド（Browserbase）モードの両方で動作します。72時間より古い録画は自動的にクリーンアップされます。

## ステルス機能

Browserbaseは自動的なステルス機能を提供します:

| 機能 | デフォルト | 備考 |
|---------|---------|-------|
| 基本ステルス | 常にオン | ランダムなフィンガープリント、ビューポートのランダム化、CAPTCHA解決 |
| レジデンシャルプロキシ | オン | より良いアクセスのためレジデンシャルIPを経由 |
| 高度なステルス | オフ | カスタムChromiumビルド、Scale Planが必要 |
| Keep Alive | オン | ネットワークの不調後のセッション再接続 |

:::note
有料機能がお使いのプランで利用できない場合、Hermesは自動的にフォールバックします — まず `keepAlive` を無効にし、次にプロキシを無効にします — そのため無料プランでもブラウジングは引き続き動作します。
:::

## セッション管理

- 各タスクは、Browserbaseを介して分離されたブラウザセッションを取得します
- セッションは非アクティブ後に自動的にクリーンアップされます（デフォルト: 2分）
- バックグラウンドスレッドが30秒ごとに古いセッションをチェックします
- 孤立したセッションを防ぐため、プロセス終了時に緊急クリーンアップが実行されます
- セッションはBrowserbase API（`REQUEST_RELEASE` ステータス）を介して解放されます

## 制限事項

- **テキストベースの操作** — ピクセル座標ではなくアクセシビリティツリーに依存します
- **スナップショットのサイズ** — 大きなページは8000文字で切り詰められるか、LLMで要約される場合があります
- **セッションタイムアウト** — クラウドセッションは、プロバイダーのプラン設定に基づいて期限切れになります
- **コスト** — クラウドセッションはプロバイダーのクレジットを消費します。セッションは会話が終了したとき、または非アクティブ後に自動的にクリーンアップされます。無料のローカルブラウジングには `/browser connect` を使用してください。
- **ファイルダウンロード不可** — ブラウザからファイルをダウンロードすることはできません
