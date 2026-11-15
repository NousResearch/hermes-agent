---
title: ブラウザ自動化
description: browser_exec で実際のブラウザを制御します — Hermes が起動する Chromium、クラウドブラウザ、Lightpanda、または CDP 経由の自分の Chrome — Web 操作、フォーム入力、スクレイピングなどに使えます。
sidebar_label: ブラウザ
sidebar_position: 5
---

# ブラウザ自動化

`browser` ツールセットは、エージェントに本物の Web ブラウザを提供します。操作を担うのは 1 つのツール
— **`browser_exec`** で、モデルが書いた Python を Chrome DevTools Protocol（CDP）経由で
ブラウザに対して実行します — で、これに 3 つの補助ツールが加わります
（`browser_cdp` と `browser_dialog` は CDP エンドポイントが設定されている間のみ。
[補助ツール](#companion-tools)を参照）:

| ツール | 用途 |
|------|---------|
| `browser_exec` | 事前にインポートされたブラウザヘルパーとともに Python を実行し、ナビゲーション、読み取り、クリック、入力、スクレイピング、スクリーンショットを行います。 |
| `browser_cdp` | 生の CDP コマンドを 1 つ送信します（Cookie、ネットワーク制御、iframe スコープの評価など、ヘルパーでカバーされないあらゆる操作）。 |
| `browser_dialog` | ページをブロックしているネイティブの JavaScript ダイアログ（`alert` / `confirm` / `prompt` / `beforeunload`）を承諾または却下します。 |
| `browser_vault` | 保存済みのログイン情報、カード、住所、2FA コードを、シークレットを会話に含めることなく現在のページに入力します。 |

ブラウザの取得元は別途選択します:

- **ローカル Chromium（デフォルト）** — Hermes が同梱の Chromium をヘッドレスで、セッションごとに 1 つ起動します。API キーもセットアップも不要です。
- **クラウドブラウザ** — [Nous サブスクリプション](tool-gateway.md)、[Browserbase](https://browserbase.com)、[Browser Use](https://browser-use.com) クラウド、[Firecrawl](https://firecrawl.dev)、またはブラウザプロバイダープラグイン。
- **[Lightpanda](https://lightpanda.io)** — 小さなマシンでテキスト中心のブラウジングを行うための軽量なヘッドレスエンジン。
- **自分のブラウザ** — `/browser connect` で、稼働中の Chrome、Brave、Chromium、Edge にアタッチします。
- **実際のプロファイル** — 普段使いのブラウザプロファイルの管理されたコピーで、ログイン情報も含まれます（オプトイン）。

:::tip
単純な情報取得には、`web_search` または `web_extract` を優先してください — そのほうが高速で安価です。ブラウザは、エージェントがページを**操作**する必要があるとき（ボタンのクリック、フォーム入力、動的コンテンツの処理）に使用してください。
:::

## `browser_exec` の仕組み

各呼び出しでは Python のブロック（`code`）が送信され、ブラウザヘルパーがすでに
インポートされた状態で実行されます。コードが出力した内容が結果として返されます。
ブラウザセッション（開いているタブ、Cookie、現在のページ）は呼び出しをまたいで保持されますが、
Python の変数は保持されません — 呼び出しのたびに新しいインタープリタで開始されます。
セッションごとのワークスペースディレクトリ（`$BH_AGENT_WORKSPACE`、`workspace` としても返されます）は
呼び出し間で保持されるため、長い抽出処理ではバッチをファイルに追記し、最後に集約できます。

ヘルパー:

| ヘルパー | 動作 |
|--------|------|
| `new_tab(url)` | URL を開きます（最初のナビゲーションに使用）。 |
| `goto_url(url)` | 現在のタブでナビゲーションします（以降のナビゲーション）。 |
| `wait_for_load()` | ページの読み込み完了を待ちます。 |
| `page_info()` | 現在のページの状態を要約します。 |
| `js(expr)` | JavaScript を評価して値を返します（関数本体は `js('(() => {...})()')` のようにラップします）。 |
| `fill_input(selector, text)` | 入力欄に入力します。 |
| `click_at_xy(x, y)` | ビューポート座標をクリックします。 |
| `capture_screenshot()` | スクリーンショットを保存し、そのパスを出力します。 |
| `cdp('Domain.method', **kwargs)` | 生の CDP — 例: アクセシビリティツリーには `cdp('Accessibility.getFullAXTree')`、クリック座標には `cdp('DOM.getBoxModel', backendNodeId=n)`。 |
| `ensure_real_tab()` | 古くなったタブや内部タブから復帰します。 |

モデルは、アクションごとに呼び出しを使うのではなく、一連のサブ手順（ナビゲーション、待機、抽出、操作）を
1 回の呼び出しにまとめます:

```python
# Searching GitHub trending for Python repos
new_tab("https://github.com/trending/python")
wait_for_load()
print(js("[...document.querySelectorAll('article h2 a')].slice(0, 10).map(a => a.textContent.trim())"))
```

**スクリーンショット。** ネイティブのビジョン機能を持つモデルでは、`capture_screenshot()` で
保存された画像がツールの結果に添付され、モデルが直接確認します。テキストのみのモデルには
テキスト中心で作業するよう指示されます（`page_info()`、`js()`、`fill_input()`）。それでも、
保存されたパスを `vision_analyze` に渡すことはできます。メッセージングプラットフォームでは、
エージェントにスクリーンショットの送信を依頼できます — ネイティブの写真添付として届きます。

**パラメータ。** `code` のほかに、このツールは `timeout_s`（デフォルト 300、最大
1800）と `session` を受け取ります。`session` の名前ごとに独自のドライバープロセスと独自の
ブラウザ（ローカルでは別の Chromium、クラウドプロバイダーでは別のクラウドブラウザ）が割り当てられるため、
並列のサブエージェントや同時進行のチャットが 1 つの共有ブラウザを奪い合うことはありません。
`session` を省略すると共有のデフォルトセッションが使用され、一度に 1 つずつブラウジングするなら
それで十分です。[実際のプロファイルでのブラウジング](#real-profile-browsing-use-your-own-logins)
が有効な場合、ツールは `local` フラグも公開します（そのセクションを参照）。

:::note
`browser_exec` はモデルが書いた Python をあなたのマシン上で実行するため、`terminal`
ツールセットも持つセッションにのみ提供されます。ターミナルアクセスなしで設定された
プラットフォーム（例えば、制限されたメッセージング環境）ではブラウザは利用できません。
:::

ドライバーは Browser Use CLI のエンジンである [browser-harness](https://github.com/browser-use/browser-use) です。
これは Hermes の通常の Python 依存関係であるため、Desktop アプリを含むすべてのインストールに
別途ダウンロードなしで同梱されています。

## ローカル Chromium（デフォルト） {#local-chromium-default}

クラウドプロバイダー、`/browser connect` のエンドポイント、実際のプロファイルのトグルの
いずれも設定されていない場合、Hermes はセッション用の Chromium を自ら起動し、
`browser_exec` をそれに向けます。インストール済みの Chrome には一切触れません —
有効化すべき `chrome://inspect` のトグルも、「Allow remote debugging?」のポップアップもありません —
また、デスクトップブラウザがまったくないヘッドレスサーバーでも動作します。

- **セッションごとに 1 つのブラウザ**。デフォルトでは使い捨ての新しいプロファイルを使用します:
  エージェントは何にもログインしていない状態で開始し、セッション後には何も残りません。
- **デフォルトでヘッドレス。** 表示可能なウィンドウにするには `browser.headed: true` を設定します
  （[後述](#headed-mode-visible-browser-window)を参照）。
- **自動クリーンアップ。** ブラウザは、`browser.inactivity_timeout` 秒（デフォルト 120）
  アイドル状態が続いたとき、Hermes の終了時、そしてプロセスがクラッシュした場合は孤立プロセスの
  スイープによって閉じられます。
- **プライベートホストにも到達可能。** ローカルのターミナルバックエンドでは、ローカルの
  Chromium は `security.allow_private_urls` の設定にかかわらず `localhost` や LAN のアドレスを
  開けます: ターミナルが到達できないものには到達しません。
  プライベートアドレスのガードは、ブラウザがターミナルの到達できないネットワークに到達できる場合
  （クラウドプロバイダー、`/browser connect` のエンドポイント、または
  Docker/SSH/Modal のターミナルバックエンド）に適用されます。エージェントが書いたクラウドメタデータの URL は
  常に拒否されます。

**どの Chromium を使うか。** Hermes は、存在する最初のものを選びます:

1. 設定されている場合は `browser.executable_path`。
2. Hermes のパッケージマネージャーがインストールした Chromium（Chrome for Testing） —
   インストーラーと `hermes update` がデフォルトでインストールします。macOS のデスクトップアプリには
   同梱されていません: 最初のブラウザ使用時に
   `~/.hermes/tools` にダウンロードされます（約 195 MB）。
3. 公式 Docker イメージでは `/etc/hermes/chromium-executable-path`。
4. `PATH` 上のシステムの `chromium` / `chromium-browser`（Android/Termux、および
   パッケージマネージャーが Chromium ビルドを持たないプラットフォーム）。

どれも見つからない場合、ブラウザツールは黙って失敗するのではなく、明確なヒントを報告します。
`--skip-browser` / `-SkipBrowser` を付けてインストールした場合や、ダウンロードに失敗した場合は、
次でインストールします:

```bash
hermes pm install chromium
```

これにより以前の `--skip-browser` の選択も取り消されるため、更新時にブラウザが最新に保たれます。
Linux では、Chromium にディストリビューションのシステムライブラリが必要な場合もあります
（Debian/Ubuntu では `npx playwright install-deps chromium` でインストールできます）。
Android/Termux では、リポジトリが提供している場合は `pkg install chromium` を実行してください。
システムの Chromium がない場合、そのデバイスではブラウジングを利用できません。

**ローカルのオプション**（`~/.hermes/config.yaml`）:

```yaml
browser:
  executable_path: ""     # Chromium binary to launch instead of the packaged one
  user_data_dir: ""       # persistent profile directory; "" = fresh profile per session
  chromium_args: []       # extra Chromium command-line flags, e.g. ["--lang=de-DE"]
  headed: false           # true = visible window
  command_timeout: 30     # Chromium launch timeout floor (seconds)
  inactivity_timeout: 120 # close an idle browser after N seconds
```

`browser.user_data_dir` を設定すると、Cookie とログイン情報がセッションをまたいで保持されます:
すべてのセッションがその 1 つのプロファイルディレクトリを共有します。分離された使い捨ての
ブラウジングにしたい場合は空のままにしてください。

### ヘッド付きモード（表示可能なブラウザウィンドウ） {#headed-mode-visible-browser-window}

```yaml
browser:
  headed: true  # default: false
```

ヘッド付きモードは次の 2 つを行います:

1. **表示可能なウィンドウで Chromium を起動**するため、エージェントの作業を見守り、
   手動で介入できます（サインインのチャレンジ、CAPTCHA）。
2. **ターン間でブラウザを開いたままにします。** 通常、ブラウザはエージェントが応答するたびに
   クリーンアップされますが、ヘッド付きモードではターンごとのクリーンアップがスキップされるため、
   会話を通じてログイン状態が維持されます。

アイドル状態のブラウザは引き続き `browser.inactivity_timeout` 後に回収され、シャットダウン時には
すべて閉じられます。ディスプレイのないホスト（サーバー、CI）では、ブラウザは常にヘッドレスで
実行されます。ヘッド付きモードはローカルブラウザにのみ影響し、クラウドセッションには影響しません。

## クラウドブラウザ

:::tip Nous購読者の方へ
有料の [Nous Portal](https://portal.nousresearch.com) 購読をお持ちの場合、別途APIキーを用意することなく、**[Tool Gateway（ツールゲートウェイ）](tool-gateway.md)** を通じてクラウドブラウザを利用できます。新規インストールでは `hermes setup --portal` を実行してログインし、すべてのゲートウェイツールを一度に有効化できます。既存のインストールでは、`hermes model` または `hermes tools` からブラウザプロバイダーとして **Nous Subscription** を選択できます。
:::

`hermes tools` → Browser Automation でプロバイダーを選択します。すると `browser_exec` は、
ローカルの Chromium を操作するのとまったく同じように、そのプロバイダーのブラウザを操作します。

### Browser Use クラウド

```bash
# Add to ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

APIキーは [browser-use.com](https://browser-use.com) で取得できます。Browser Use Cloud は、[ステルス](https://docs.browser-use.com/cloud/browser/stealth)と[レジデンシャルプロキシ](https://docs.browser-use.com/cloud/browser/proxies)がデフォルトで有効なマネージド Chromium を実行し、CAPTCHA 解決を含み、Cookie、localStorage、保存済みパスワードのための[永続プロファイル](https://docs.browser-use.com/cloud/guides/authentication)をサポートしています。

### Browserbase

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

認証情報は [browserbase.com](https://browserbase.com) で取得できます。オプション設定:

```bash
# Residential proxies for better CAPTCHA solving (default: "true")
BROWSERBASE_PROXIES=true

# Advanced stealth with custom Chromium — requires Scale Plan (default: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# Session reconnection after disconnects — requires paid plan (default: "true")
BROWSERBASE_KEEP_ALIVE=true

# Custom session timeout in seconds (max 21600 = 6 hours) (default: project default)
BROWSERBASE_SESSION_TIMEOUT=1800
```

Browserbase のステルス機能:

| 機能 | デフォルト | 備考 |
|---------|---------|-------|
| 基本ステルス | 常にオン | ランダムなフィンガープリント、ビューポートのランダム化、CAPTCHA解決 |
| レジデンシャルプロキシ | オン | より良いアクセスのためレジデンシャルIPを経由 |
| 高度なステルス | オフ | カスタムChromiumビルド、Scale Planが必要 |
| Keep Alive | オン | ネットワークの不調後のセッション再接続 |

有料機能がお使いのプランで利用できない場合、Hermesは自動的にフォールバックします — まず `keepAlive` を無効にし、次にプロキシを無効にします — そのため無料プランでもブラウジングは引き続き動作します。

### Firecrawl

```bash
# Add to ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***

# Optional: self-hosted Firecrawl instance (default: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# Optional: session TTL in seconds (default: 300)
FIRECRAWL_BROWSER_TTL=600
```

APIキーは [firecrawl.dev](https://firecrawl.dev) で取得し、`hermes tools` → Browser Automation で **Firecrawl** を選択します。

:::note プロバイダーの選択
上記の `.env` のキーは**認証情報のみ**を提供します。アクティブなブラウザは、`hermes tools` → Browser Automation によって書き込まれる `browser.cloud_provider` の選択（`local`、`browserbase`、`browser-use`、`firecrawl`、または Nous Subscription の場合は `nous`。プロバイダープラグインは独自の名前を追加します）で決まります。選択が存在すると、キーを追加・削除してもプロバイダーは切り替わりません — また、キーが欠けている選択済みのプロバイダーは、黙って別経路に切り替えるのではなく、`hermes tools` を実行するよう案内するエラーになります。一度も設定されていないセットアップでは、引き続き利用可能な認証情報から自動検出されます。
:::

クラウドセッションはプロバイダーのクレジットを消費します。Hermes は、会話が終了したとき、
`browser.inactivity_timeout` の経過後、そしてプロセス終了時にそれらを解放します。

### ハイブリッドルーティング: 公開URLにはクラウド、LAN/localhostにはローカル {#hybrid-routing-cloud-for-public-urls-local-for-lanlocalhost}

クラウドプロバイダーが設定されている場合、Hermesはプライベート/ループバック/LANアドレス（`localhost`、`127.0.0.1`、`192.168.x.x`、`10.x.x.x`、`172.16-31.x.x`、`*.local`、`*.lan`、`*.internal`、IPv6ループバック `::1`、リンクローカル `169.254.x.x`）に解決されるURLに対して、**ローカルのChromiumサイドカー**を自動的に起動します。公開URLは同じ会話内で引き続きクラウドプロバイダーを使用します。

これにより、よくある「ローカルで開発しつつBrowserbaseを使っている」ワークフローが解決します — エージェントは `http://localhost:3000` にあるダッシュボードのスクリーンショットを撮りつつ、プロバイダーを切り替えたりSSRFガードを無効化したりすることなく `https://github.com` をスクレイピングできます。クラウドプロバイダーがプライベートURLを見ることはありません。

この機能は**デフォルトで有効**で、`security.allow_private_urls` の影響を受けません:
この設定はクラウドブラウザと Web ツールを保護しますが、サイドカーは Hermes ホスト上で実行され、
この設定がオフでもプライベートホストを開きます。ターミナルから `curl http://localhost` を実行するのと同じです。
したがって、ホスティングされた Hermes（Docker、VPS）では、エージェントはサイドカーを通じて
そのホスト自身のループバックサービスと LAN をブラウズできます。代わりにプライベート URL を拒否する
（すべての URL が設定済みのクラウドプロバイダーに送られる）には:

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

自動ルーティングを無効化すると、`security.allow_private_urls: true`（またはレガシーの `browser.allow_private_urls: true`）も
設定しない限り、プライベートURLは `"Blocked: URL targets a private or internal address"` で拒否されます。
これを設定するとクラウドプロバイダーに渡されますが、BrowserbaseなどはあなたのLANに到達できないため、通常は役に立ちません。
`browser_exec` のコード内に書かれた、または生の CDP 呼び出しが取得する URL に含まれるクラウドメタデータの URL
（`169.254.169.254`、`metadata.google.internal`）は、サイドカーを含むすべての経路で拒否されます。
ただし、ページが自らそこへリダイレクトする場合は拒否されません
（[セキュリティ](../security.md#what-allow_private_urls-covers-and-what-it-doesnt)を参照）。

サイドカーは[上記](#local-chromium-default)と同じローカル Chromium であるため、
その Chromium がインストールされている必要があります。ルーティングは `browser_exec` のコードに
書かれた URL から決定されます: 後からプライベートアドレスへリダイレクトする公開ページは
クラウドブラウザに留まり、クラウドブラウザはあなたのネットワークに到達できません。

## 実際のプロファイルでのブラウジング（自分のログイン情報を使う） {#real-profile-browsing-use-your-own-logins}

デフォルトでは、ローカルのブラウジングはクリーンな使い捨てのプロファイルで実行されます — エージェントは
何にもログインしていません。**実際のプロファイルでのブラウジング**を有効にすると、エージェントは
既存のログイン情報と Cookie を使って*あなた*としてブラウズできます:

```yaml
# ~/.hermes/config.yaml
browser:
  use_real_profile: true
```

有効にすると、Hermes はデフォルトブラウザの**アクティブな**プロファイル — 実際に使っているもの
（`Local State → profile.last_used`）で、Cookie、保存済みログイン情報、設定を含みます — を
`~/.hermes/browser-profile/<browser>/` 配下の管理されたスナップショットにコピーし、そのスナップショット上で
**実際のブラウザのバイナリ**を起動して、ブラウジングエンジンをそれにアタッチします。
（mock-keychain のスイッチを付けた同梱の Chromium ではなく）実際のバイナリを起動することで、
OS で暗号化された Cookie を復号できる状態が保たれます — macOS では Chrome の Cookie は
キーチェーンを通じて暗号化されており、mock-keychain で起動すると Cookie がすべて黙って破棄され、
サインアウトした状態で開いてしまいます。ライブのブラウザプロファイルが**直接開かれることはありません**:
スナップショットは別のディレクトリであるため、稼働中のブラウザとプロファイルロックを奪い合うことはなく、
Chrome 136+ によるデフォルトプロファイルディレクトリのリモートデバッグのブロックも回避できます。
認証ファイル（Cookie/ログイン情報/設定）は、新しいセッションが起動されるたびに実際のプロファイルから
再同期されるため、自分のブラウザで行ったログインがエージェントのセッションにも反映されます。
コピーされるのはアクティブなプロファイルのみで、他の Chrome プロファイルがスナップショットされることはありません。

スナップショットのブラウザは**ヘッドレス**で実行されます — 表示されるウィンドウなしでバックグラウンドで
プロファイルを操作し、フォーカスを奪うこともないため、エージェントがあなたに代わってツイートしたり、
フォームに入力したり、スクレイピングしたりしている間も作業を続けられます。
（ここでのヘッドレスは Chrome の*新しい*ヘッドレスモードを使用しており、通常の Cookie ストアを
読み込むため、ログイン情報も読み込まれます。）作業を見たい場合は、同じ
[ヘッド付きモード](#headed-mode-visible-browser-window)のトグルが適用されます —
`browser.headed: true` を設定すると、実際のプロファイルでのブラウジングでも表示可能なウィンドウが開きます。
ディスプレイのないホスト（サーバー、CI）では、設定にかかわらず常にヘッドレスで実行されます。

ブラウザに複数のプロファイル（例えば仕事用と個人用）があり、「最後に触ったプロファイル」で
エージェントのアイデンティティが決まるのを避けたい場合は、スナップショットの取得元を明示的に固定します:

```yaml
# ~/.hermes/config.yaml
browser:
  use_real_profile: true
  real_profile_pin: "Profile 2"   # directory name under the browser's user-data dir
```

存在しないプロファイルディレクトリを指定した固定設定が、黙って最後に使ったプロファイルに
フォールバックすることはありません: エージェントにはログイン情報のない新しいブラウザと、
誤った固定設定を示すメッセージが渡されます。

トグルをオフに戻すと、Hermes は次回のブラウザ使用時にスナップショットストア
（`~/.hermes/browser-profile/`）を削除するため、同意を取り消した後にコピーされた
認証情報が残ることはありません。

:::note プロファイルをコピーできない場合でもブラウザは動作します
実際のプロファイルでのブラウジングによってブラウザが使えなくなることはありません。Hermes が
プロファイルをコピーできない場合、エージェントはそのセッションで**ログイン情報のない**新しいブラウザを取得し、
最初のブラウザ結果でその理由（`real_profile_unavailable`）が示されるため、サインインしているかのように
振る舞うのではなく、あなたにそれを伝えられます。これは次の場合に起こります:

- **Windows、初回使用時のみ:** 稼働中の Chrome/Edge/Brave は Cookie データベースを
  排他的（deny-all）ロックで保持しているため、最初のコピーにはブラウザを**完全に終了**させる必要があります。
  バックグラウンド/トレイのインスタンスも含みます（Chrome の「アプリを閉じた際にバックグラウンド アプリの
  処理を続行する」設定により、ウィンドウを閉じた後も `chrome.exe` が動作し続けます）。一度コピーに成功すると、
  以降はブラウザが開いた状態で起動してもそのコピーが再利用されます: ログイン情報は最後にコピーした時点のものとなり、
  次に自分のブラウザを閉じた状態で Hermes がブラウザを起動したときに更新されます。
- **デフォルトブラウザがサポート対象の Chromium ではない**場合（Firefox、Beta/Dev/Canary
  チャンネル）、または固定したプロファイルが存在しない場合。
- **Cookie データベースが読み取れないか破損している**場合: メッセージにそのデータベースと
  SQLite のエラーが示されます。

macOS と Linux では、稼働中の Chrome は `Login Data`、`Login Data For Account`、
`Web Data`（保存済みパスワードと自動入力）を書き込みロックで保持しますが、`Cookies`
はコピーでき、サイトへのサインイン状態を保つのは Cookie です。Hermes は Cookie を
コピーし、ロックされたパスワード/自動入力ファイルはスキップするため、ブラウザを開いたままにできます。
各データベースは SQLite のオンラインバックアップを通じてコピーされます（最近のログインが失われる
可能性のある生のファイルコピーは行いません）。

`browser.real_profile_autoclose: true` を設定すると、ブラウザがプロファイルロックを保持している場合
（Windows のケース）に、Hermes が**ブラウザを閉じることを提案**できるようになります。これを有効にしても、
Hermes が自動的に閉じることはありません: エージェントがまずあなたに確認し、承認された場合にのみ
`hermes browser close-profile` を実行します（そのプロファイルに紐づくブラウザのプロセスツリーを終了するため、
保存されていないタブは失われます）。その後、新しいブラウザセッションがあなたのプロファイルを取得します。
:::

- **サポートされるブラウザ:** Chrome、Edge、Brave、Brave Origin、Chromium（OS のデフォルトになっているもの）。
  Chromium 以外のデフォルト（例: Firefox）では、どのプロファイルを使うかを推測するのではなく、
  新しいブラウザと明確なメッセージが渡されます。
- **あらゆるブラウザの取得元で動作します。** ローカルではトグルをオンにすれば自動的に適用されます。
  **クラウド**ブラウザプロバイダーの下でも、エージェントは `browser_exec` ツールの `local`
  引数を介して、必要に応じて実際のプロファイルのローカルセッションを開けます（ツールがこの引数を公開するのは
  このトグルがオンの場合のみです） — それ以外はすべて引き続きクラウドプロバイダーが処理します。
- **セキュリティ上の位置付け:** これは同意に基づく利便機能であり、分離境界ではありません。
  エージェントが訪れるページはあなたの実際のログイン情報とともに実行されるため、エージェントに
  あなたとして行動させたい場合にのみ有効にしてください。デフォルトではオフです。
- **Desktop:** **Capabilities → Tools → Browser → Use My Real
  Browser Profile**（スイッチはバックエンドのオプションの上にあります）で切り替えるか、
  Settings → Config の `browser` セクションで設定します。

### スケジュール実行と無人実行 {#scheduled-and-unattended-runs}

実際のプロファイルのセッションを使うと、すでにサインインしているサイトを `cronjob` で操作できます。
スナップショットはデフォルトでヘッドレスで実行され、認証ファイル（`Cookies`、
`Login Data` など）は新しいセッションごとに実際のプロファイルから再同期されます —
すでに開いているセッションはそのまま再利用されるため、同期は新しいセッションが起動されたときに行われます。
保存済みの Vault の認証情報がない場合、そのサイトのセッションが期限切れになるとログインページが表示され、
無人のティックは待ち続けるのではなくそれを報告します。

スケジュールを設定する前に、次の 3 点を設定してください:

- **トグルをオンにします。** `browser.use_real_profile` のデフォルトは `false` で、
  これがないとジョブは認証されていないクリーンなプロファイルを使います — 上記を参照してください。
- **ジョブにブラウザツールセットを与えます:**
  `cronjob(action="create", enabled_toolsets=["browser", ...], ...)` — ジョブごとの
  リストは cron プラットフォームの設定より優先されます
  （[詳細](./cron.md#toolsets-available-to-cron-jobs)）。
- **何も入力を求められない前提にします。** cron、webhook、API、
  `hermes chat -q` のセッションにはプロンプトに応答する人がいないため、同期された Cookie だけでは
  利用できないサイト — ログインフォームや新たな 2FA チャレンジ — では、認証システムのキーを含め、
  事前に認証情報を保存しておく必要があります。それは
  [認証情報 Vault](/docs/user-guide/features/credential-vault#headless-sessions) の役割であり、
  あなた自身のセッションが期限切れになった後の実行を支えるのもこれです。

**Windows の前提条件:** 最初のスナップショットにはブラウザを完全に終了させる必要があります。
その後は、ブラウザが開いている間に実行されたティックは最後のスナップショット（そのコピー時点のログイン情報）を
再利用します。スナップショットがまだない場合、ティックはサインアウトした新しいブラウザで実行されます。
Hermes が事前に確認せずにブラウザを閉じることはありません（上記の注記を参照）。

同梱の `product-price-monitor` スキルは、繰り返し実行の形を示す実例です —
セットアップ中に JSON の監視契約を書き込み、その後スケジュールされた 1 つのティックがそれを再読み込みして
比較し、変化があれば通知します — ただし、扱っているのはスケジュールの部分で、ログインは扱っていません。

## Lightpanda ローカルエンジン

[Lightpanda](https://lightpanda.io) は、ゼロから書かれたオープンソースのヘッドレスブラウザです。即座に起動し、Chrome より 9 倍高速に動作し、メモリ使用量は 16 分の 1 です。これは、小さな VM 上で長時間稼働するエージェントにとって重要です。

Lightpanda は**ローカルエンジン**（ローカル Chromium と同様のブラウザの取得元）であり、クラウドプロバイダーではありません。バイナリをインストールして `PATH` に置き（[Lightpanda のインストールガイド](https://lightpanda.io/docs/run-locally/installation/one-liner)を参照）、`hermes tools` → Browser Automation で **Lightpanda** を選択するか、次を設定します:

```yaml
# Add to ~/.hermes/config.yaml
browser:
  cloud_provider: local
  engine: lightpanda   # auto (default, Chromium) | lightpanda
```

Hermes は `lightpanda serve --host 127.0.0.1 --port <free>` を自ら起動し — `browser_exec` のセッション名ごと（またはタスクごと）に 1 プロセス — `browser_exec` をそれに向けます。Chromium や Node.js は不要です。プロセスは `browser.inactivity_timeout` の経過後、終了時、そして Hermes がクラッシュした場合は孤立プロセスのスイープによって回収されます。これらのプロセスはすべて `$HERMES_HOME/cache/browser-use/lightpanda/http-cache` にある 1 つのディスク上の HTTP キャッシュを共有するため、再訪問時にはアセットの再ダウンロードがスキップされます。Hermes は、インストールされている Lightpanda がキャッシュフラグをサポートしている場合（0.3.x+）にのみそれを渡します。古いバイナリは単にキャッシュなしで実行されます。キャッシュをクリアするには、まず Lightpanda のセッションを停止してから、そのディレクトリを削除してください。

Lightpanda にはグラフィカルなレンダラーがないため `capture_screenshot()` は利用できず、ツールの説明でモデルにテキスト中心で作業するよう指示されます。また、セッションごとに 1 ページしか保持しないため、モデルには `new_tab()` を一度呼び出し、以降は `goto_url()` を使うよう指示されます（アップストリームの [lightpanda-io/browser#1962](https://github.com/lightpanda-io/browser/issues/1962) で追跡されています）。

**エンジンが無視される場合。** `browser.engine` は最も優先度の低いブラウザ設定です: クラウドプロバイダー（Nous サブスクリプションのブラウザを含みます — また、一度も設定されていないセットアップでは、`~/.hermes/.env` 内の `BROWSERBASE_API_KEY` / `BROWSER_USE_API_KEY` があれば自動的に選択されます）、`browser.cdp_url` / `/browser connect` のオーバーライド、または `browser.use_real_profile` がすべて優先されます。`hermes tools` で Lightpanda を選ぶと `cloud_provider: local` が自動的に書き込まれます。`/browser status` と `hermes doctor` は、エンジンが設定されているのに別の設定に隠されている場合に、それと原因を報告します。

## CDP 経由の自分のブラウザ（`/browser connect`） {#your-own-browser-via-cdp-browser-connect}

Hermes が起動する Chromium やクラウドプロバイダーの代わりに、Chrome DevTools Protocol（CDP）を介して、自分の稼働中の Chrome、Brave、Chromium、Edge のインスタンスにブラウザツールをアタッチできます。これは、エージェントの動作をリアルタイムで見たい場合、自分のCookie/セッションを必要とするページを操作したい場合、またはクラウドブラウザのコストを避けたい場合に便利です。

:::note
`/browser connect` は**対話型CLIのスラッシュコマンド**です — ゲートウェイによってディスパッチされません。WebUI、Telegram、Discord、その他のゲートウェイチャットの中で実行しようとすると、メッセージはプレーンテキストとしてエージェントに送られ、コマンドは実行されません。ターミナルからHermesを起動し（`hermes` または `hermes chat`）、そこで `/browser connect` を実行してください。
:::

CLIでは、次を使用します:

```
/browser connect                 # Auto-launch/connect to a local Chromium-family browser at http://127.0.0.1:9222
/browser connect ws://host:port  # Connect to a specific CDP endpoint
/browser status                  # Check current connection
/browser disconnect              # Detach and return to cloud/local mode
```

リモートデバッグ付きのブラウザがまだ起動していない場合、Hermes はサポートされている Chromium 系ブラウザを `--remote-debugging-port=9222` で自動起動しようとします。検出対象には Brave、Brave Origin/Nightly、Google Chrome、Chromium、Microsoft Edge が含まれ、`brave-origin`、`brave-origin-nightly`、`/opt/brave.com/brave-origin/brave-origin`、`/opt/brave.com/brave-origin-nightly/brave-origin`、`/opt/brave-bin/brave`、`/snap/bin/brave` といった Linux の一般的なインストールパスやバイナリ名も含まれます。

:::tip
CDP を有効にして Chromium 系ブラウザを手動で起動するには、専用の user-data-dir を使ってください。そうすれば、通常のプロファイルでブラウザがすでに起動していても、デバッグポートが実際に立ち上がります:

```bash
# Linux — Brave
brave-browser \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# Linux — Google Chrome
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# macOS — Brave
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &

# macOS — Google Chrome
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &
```

その後、Hermes CLIを起動して `/browser connect` を実行します。

**なぜ `--user-data-dir` が必要か？** これがないと、通常のインスタンスがすでに起動している状態で Chromium 系ブラウザを起動すると、たいてい既存のプロセス上で新しいウィンドウが開きます — そしてその既存プロセスは `--remote-debugging-port` で起動されていないため、ポート9222は決して開きません。専用の user-data-dir を使うと、デバッグポートが実際にリッスンする新しいブラウザプロセスが強制的に立ち上がります。`--no-first-run --no-default-browser-check` は、新しいプロファイルの初回起動ウィザードをスキップします。

**Chrome 136+ では専用プロファイルが必須です。** 2 つの別々の仕組みが関係しており、デフォルト以外の `--user-data-dir` を渡せばどちらも適用されません。まず、[Chrome 136](https://developer.chrome.com/blog/remote-debugging-port) 以降、`--remote-debugging-port` と `--remote-debugging-pipe` は「デフォルトの Chrome データディレクトリをデバッグしようとする場合は尊重されなくなります」 — フラグは黙って無視され、ダイアログも表示されず、コールドスタートからでも `/browser connect`（または `curl http://127.0.0.1:9222/json/version`）は接続を拒否されます。次に、[Chrome 144+](https://developer.chrome.com/blog/chrome-devtools-mcp-debug-your-browser-session) では、実際のプロファイルをデバッグするためのオプトインの*承認*フローが追加されています: `chrome://inspect/#remote-debugging` で有効にすると、Chrome は**受信する接続ごとに**（起動ごとに 1 回ではなく）**「Allow remote debugging?」**ダイアログを表示し、**Allow** を押すまで何もリッスンしません。このダイアログが表示される場合は、フラグの経路ではなく承認の経路にいます。解決策はまさに上記のコマンドです: `--user-data-dir` をデフォルトのプロファイルディレクトリ以外の場所（例: `$HOME/.hermes/chrome-debug`）に向ければ、トグルもダイアログも不要です。これは、この変更を取り込んだ Chrome、Chromium、Edge、Brave のビルドに当てはまります。

専用プロファイルは、すべてからサインアウトした状態で始まります。既存のログイン情報を使い、*かつ*承認ダイアログなしでエージェントにブラウズさせたい場合は、代わりに [`browser.use_real_profile`](#real-profile-browsing-use-your-own-logins) を使ってください: アクティブなプロファイルをコピーにスナップショットしてそれを操作するため、デフォルト以外の user-data-dir となり、どちらの仕組みも発動しません。
:::

接続中は、`browser_exec`、`browser_cdp`、`browser_dialog`、`browser_vault` のすべてが、独自のブラウザを起動する代わりにあなたのライブのブラウザを操作します。エンドポイントは、`config.yaml` の `browser.cdp_url` または環境変数 `BROWSER_CDP_URL` で永続的に設定することもできます。

### WSL2 + Windows Chrome: `/browser connect` よりMCPを優先 {#wsl2--windows-chrome-prefer-mcp-over-browser-connect}

HermesがWSL2内で実行されていて、制御したいChromeウィンドウがWindowsホスト上で実行されている場合、`/browser connect` は最善の方法ではないことがよくあります。

理由:

- `/browser connect` は、Hermes自身が利用可能なCDPエンドポイントに到達できることを前提としています
- 最近のChromeのライブデバッグセッションは、しばしばホストローカルなエンドポイントを公開しますが、それは従来の `9222` ポートと同じようにはWSLから直接到達できません
- Windows Chromeがデバッグ可能な場合でも、最もクリーンな統合は、Windows側のブラウザMCPサーバーにChromeをアタッチさせ、HermesにそのMCPサーバーと通信させることである場合が多いです

そのセットアップには、Hermes MCPサポートを通じた `chrome-devtools-mcp` を優先してください。

実用的なセットアップについては、MCPガイドを参照してください:

- [HermesでMCPを使う](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

## Bot Desktop 画面

[Bot Desktop 画面](/docs/user-guide/features/bot-screen)が稼働しているゲートウェイでは、ボットのブラウザと
ドックの **Browser** アイコンは同じヘッド付きの Chromium で、ボットごとに 1 つの永続プロファイルを持ちます。
そのため、操作を引き継いでサイトにサインインし、セッションをエージェントに戻すことができます。
プロファイルの場所と、そこで `browser.executable_path` / `browser.user_data_dir` がどう適用されるかについては、
そのページを参照してください。

## 補助ツール {#companion-tools}

`browser_vault` は、`browser_exec` が操作するのと同じブラウザセッション —
ローカル Chromium、クラウドブラウザ、または `/browser connect` でアタッチしたブラウザ — 上で動作します。
`browser_cdp` と `browser_dialog` は、CDP エンドポイントが設定されている間
— `/browser connect`、`BROWSER_CDP_URL`、または `browser.cdp_url` — にのみ提供され、
そのブラウザに対して動作します。起動されたローカルセッションやクラウドセッションにはこれらは付属しません。

### `browser_cdp`

生のChrome DevTools Protocolパススルー — 単発の操作のための回避策です: ネイティブダイアログの処理、
iframeスコープの評価、Cookie/ネットワーク制御、またはエージェントが必要とする任意のCDP動詞に使用します。
（`browser_exec` の中では、複数ステップの作業向けに `cdp()` ヘルパーが同じ範囲をカバーします。）

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

**クロスオリジンのiframe:** アウトオブプロセス（クロスオリジン）iframe の `frame_id` を渡すと、呼び出しがそのフレーム用の CDP スーパーバイザーのライブセッションを経由してルーティングされます。これが、Browserbaseでクロスオリジンiframe内の `Runtime.evaluate` が機能する仕組みで、そこではステートレスなCDP接続だと署名付きURLの期限切れに当たってしまいます。同一オリジンのiframeに `frame_id` は不要です — 代わりに、トップレベルの `Runtime.evaluate` から `document.querySelector('iframe').contentDocument` を使ってください。

### `browser_dialog`

ネイティブのJSダイアログ（`alert` / `confirm` / `prompt` / `beforeunload`）に応答します。開いているダイアログはページのJavaScriptスレッドをブロックするため、応答されるまでそのページでのブラウザ作業は停止します。

永続的な **CDP スーパーバイザー** — タスクごとに 1 つの WebSocket で、Page/Runtime/Target イベントを購読 — が、ダイアログが開いた時点でそれを検出します。エージェントは `browser_dialog(action="accept")` または `browser_dialog(action="dismiss")` で応答します。`prompt()` ダイアログには `prompt_text="..."` を渡し、複数のダイアログがキューにある場合は `dialog_id` を渡して 1 つを選びます。その後、ページの JS スレッドが再開します。

**Browserbaseでの仕組み。** BrowserbaseのCDPプロキシは、実際のネイティブダイアログをサーバー側で約10ms以内に自動的に却下するため、`Page.handleJavaScriptDialog` を使えません。スーパーバイザーは `Page.addScriptToEvaluateOnNewDocument` を介して小さなスクリプトを注入し、`window.alert`/`confirm`/`prompt` を同期XHRでオーバーライドします。それらのXHRを `Fetch.enable` を介してインターセプトします — ページのJSスレッドは、スーパーバイザーがエージェントの応答とともに `Fetch.fulfillRequest` を呼び出すまでXHRでブロックされたままになります。`prompt()` の戻り値は、変更されることなくページのJSに往復して戻ります。

**ダイアログポリシー**は、`config.yaml` の `browser.dialog_policy` の下で設定します:

| ポリシー | 動作 |
|--------|----------|
| `must_respond`（デフォルト） | ダイアログをキャプチャし、明示的な `browser_dialog()` 呼び出しを待ちます。`browser.dialog_timeout_s`（デフォルト300秒）後に安全のため自動却下するため、バグのあるエージェントが永久に停止することはありません。 |
| `auto_dismiss` | キャプチャし、即座に却下します。エージェントは対応する必要がありません。 |
| `auto_accept` | キャプチャし、即座に承諾します。アグレッシブな `beforeunload` プロンプトのあるページを移動する際に便利です。 |

### `browser_vault`

保存済みのログイン情報、パスワード、カード、住所、ワンタイムコードを、シークレットを会話に
一切含めることなく現在のページで使用します。エージェントは `browser_vault(action="list")` を呼び出して
保存済みのアイテムをハンドルとして確認し、その後 `fill`、`save_login`、`enter_code`、
または `unlock`（1Password / Bitwarden）を使います。ページのオリジンが保存済みアイテムと一致しない限り、
入力は拒否されます。セットアップ、パスワードマネージャーのバックエンド、ヘッドレスでの使用については
[認証情報 Vault](/docs/user-guide/features/credential-vault) のページを参照してください。

## ブラウジングをオフにする

ブラウザ用の設定スイッチはありません — 代わりに `browser` ツールセットを無効にしてください。
`hermes tools` を実行し、ブラウズさせたくないプラットフォームで **Browser Automation** のチェックを外すか、
スクリプトから次を実行します:

```bash
hermes tools disable browser                      # CLI
hermes tools disable browser --platform telegram  # one gateway platform
```

`terminal` ツールセットのないセッションでも `browser_exec` は表示されなくなります。

## Camofox

Camofox はプラグインに移行しました: `hermes plugins install camofox`。既存の Camofox の
セットアップ（`CAMOFOX_URL` が設定されている、または `browser.cloud_provider: camofox`）は、更新時に自動的に
移行されます: プラグインがインストールされ、そのプロファイルではコアの `browser` ツールセットがオフになります。
すべての場所で `browser` を無効にしていたプロファイル（`agent.disabled_toolsets`）では `camofox` も
無効になるため、ブラウジングはオフのままです。
`CAMOFOX_URL` のない `browser.camofox` ブロックは Camofox のセットアップではありません。更新後に
プラグインをセットアップしたプロファイルでは、`browser_exec` もそのまま併用できます。

## 実用的な例

### Webフォームへの入力

```
User: Sign up for an account on example.com with my email john@example.com

Agent: browser_exec(code="""
# Opening the signup form
new_tab("https://example.com/signup")
wait_for_load()
fill_input("input[name=email]", "john@example.com")
js("document.querySelector('button[type=submit]').click()")
wait_for_load()
print(page_info())
""")
```

パスワードについては、エージェントは自らシークレットを入力するのではなく、`browser_vault`
（`save_login` はマスクされたフィールドであなたに入力を求めます）を使用します。

### 動的コンテンツのリサーチ

```
User: What are the top trending repos on GitHub right now?

Agent: browser_exec(code="""
# Reading GitHub trending
new_tab("https://github.com/trending")
wait_for_load()
print(js("[...document.querySelectorAll('article h2 a')].map(a => a.textContent.replace(/\\s+/g, ' ').trim())"))
""")
```

## 制限事項

- **セッションタイムアウト** — クラウドセッションは、プロバイダーのプラン設定に基づいて期限切れになります。
- **コスト** — クラウドセッションはプロバイダーのクレジットを消費します。セッションは会話が終了したとき、または非アクティブ後にクリーンアップされます。ローカル Chromium と `/browser connect` は無料です。
- **Lightpanda** — スクリーンショットなし、セッションごとに 1 ページ。
- **ターミナルがなければブラウザもなし** — `browser_exec` は Python を実行するため、`terminal` ツールセットのないセッションでは利用できません。
