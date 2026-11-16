---
title: ウェブ検索＆抽出
description: 複数のバックエンドプロバイダー（無料でセルフホストできるSearXNGを含む）で、ウェブを検索し、ページコンテンツを抽出します。
sidebar_label: ウェブ検索
sidebar_position: 6
---

# ウェブ検索＆抽出

このページのPython依存関係コマンドは、
[PMで準備されたソースチェックアウト](/docs/reference/package-management#developer-workflow) を前提としています。
依存関係を変更した後は、チェックアウトを再度有効化してHermesを再起動してください。

Hermes Agentには、複数のプロバイダーに対応した、モデルから呼び出せる2つのウェブツールが含まれます:

- **`web_search`** — ウェブを検索し、ランク付けされた結果を返す
- **`web_extract`** — 1つ以上のURLから読みやすいコンテンツを取得・抽出する

どちらも単一のバックエンド選択で設定されます。プロバイダーは `hermes tools` で選ぶか、`config.yaml` に直接設定します。

## バックエンド

| プロバイダー | 環境変数 | 検索 | 抽出 | 無料枠 |
|----------|---------|--------|---------|-----------|
| **Firecrawl**（デフォルト） | `FIRECRAWL_API_KEY`（任意 — 選択時はキー不要） | ✔ | ✔ | 500クレジット/月 · 選択時はキー不要のクラウド |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | ✔ 無料（セルフホスト） |
| **Brave Search（無料枠）** | `BRAVE_SEARCH_API_KEY` | ✔ | — | 2,000クエリ/月 |
| **DDGS (DuckDuckGo)** | —（キー不要） | ✔ | — | ✔ 無料 |
| **Exa** | `EXA_API_KEY`（任意） | ✔ | ✔ | ✔ キー不要リングのメンバー · キーありで1,000検索/月 |
| **Parallel** | `PARALLEL_API_KEY`（任意） | ✔ | ✔ | ✔ キー不要リングのメンバー · キーありは有料 |
| **Tavily** | `TAVILY_API_KEY`（任意） | ✔ | ✔ | ✔ 選択時にオプトインでキー不要 |
| **Perplexity** | `PERPLEXITY_API_KEY` | ✔ | ✔（クエリに関連するスニペット） | 有料（Search APIのリクエスト単位の料金） |
| **Keenable** | `KEENABLE_API_KEY`（任意） | ✔ | ✔ | ✔ キー不要リングのメンバー · キーありは有料 |
| **xAI (Grok)** | `XAI_API_KEY` または `hermes auth add xai-oauth` | ✔ | — | 有料（SuperGrokまたはトークン単位） |
| **OpenAI Native (Codex)** | `hermes auth add openai-codex` | ✔ | — | ChatGPT/Codexのサブスクリプションが必要 |

Brave Search、DDGS、xAI、OpenAI Nativeは **検索専用** です — `web_extract` も必要な場合は、いずれかをFirecrawl/Tavily/Perplexity/Keenable/Exa/Parallelと組み合わせてください。DDGSは内部で [`ddgs` Pythonパッケージ](https://pypi.org/project/ddgs/) を使います。まだインストールされていない場合は `python -c "import pm; pm.sync_venv(['ddgs'], explicit=True)"` を実行してください（または初回使用時にHermesに遅延インストールさせることもできます）。xAIはResponses API上でGrokのサーバーサイド `web_search` ツールを実行します — 結果はインデックスに基づくものではなくLLMが生成したものであり、タイトル、説明、URLの選択はすべてモデルの出力です（下記の [信頼モデルに関する注意](#xai-grok) を参照）。OpenAI Nativeは、Codex Responsesエンドポイント上で同種のプロバイダー実行型ツールを宣言します（[下記](#openai-native) を参照）。

**機能ごとの分割:** 検索と抽出で異なるプロバイダーを個別に使えます — 例えば検索にはSearXNG（無料）、抽出にはFirecrawlといった具合です。下記の [機能ごとの設定](#per-capability-configuration) を参照してください。

:::info Works out of the box — keyless free-tier rotation
ウェブの認証情報が **一切ない** 新規インストールでも、`web_search` と `web_extract` がすぐに動作します: リクエストはリングに属するベンダーの公開無料枠 — **Exa、Parallel、Firecrawl、Keenable** — の間でラウンドロビンに分散されて負荷が均等になり、レート制限を受けたリクエストはリング内の次のベンダーで自動的に再試行されます（いずれかが応答するか、すべてが失敗するまでマルチホップで）。ベンダーが拒否した検索（401/402/403のアクセス制限や5xx）も同様に再試行されます。サインアップもキーも不要です。この枠は厳密に最後の手段です — 設定済みのバックエンドや存在するAPIキーが常に優先されます — また、リクエストにはユーザー識別子は含まれません（プロセスごとのランダムなセッションIDのみで、再起動時にローテーションされます）。確実でスロットリングのないサービスが必要な場合は、キー付きのプロバイダーを設定してください。キー不要の枠を完全に無効にするには `web.keyless_fallback: false` を設定します。
:::

**無料と有料を明示的に選ぶ:** `hermes tools` では、Exa、Parallel、Keenableがそれぞれ **Free (keyless)** と **Paid (API key)** の2行で表示されます。Freeを選ぶとそのベンダーの匿名エンドポイントに固定され（後でキーを追加しても同様）、Paidを選ぶとキー付きの経路に固定されます（その場合、キーがないと無料枠に黙ってダウングレードするのではなくエラーになります）。この選択は `web.provider_tier.<name>: free|paid` として保存されます。未設定のままにすると自動になります（キーあり → 有料、それ以外 → キー不要リング）。

:::tip Nous Subscribers
有料の [Nous Portal](https://portal.nousresearch.com) サブスクリプションをお持ちの場合、ウェブ検索と抽出はマネージドウェブ検索として **[Tool Gateway](tool-gateway.md)** 経由で利用できます — APIキーは不要です。新規インストールでは `hermes setup --portal` を実行するとログインしてすべてのゲートウェイツールを一度に有効にできます。既存のインストールでは `hermes tools` でウェブだけを切り替えられます。
:::

---

## `web_extract` が長いページを処理する方法 {#how-web_extract-handles-long-pages}

バックエンドは生のページMarkdownを返しますが、これは膨大になることがあります（フォーラムのスレッド、ドキュメントサイト、コメントが埋め込まれたニュース記事など）。コンテキストウィンドウを使い物になる状態に保つため、`web_extract` は **決定的な文字数バジェット** を適用します — LLMによる要約は一切行われません:

| ページサイズ（文字数） | 何が起きるか |
|------------------------|--------------|
| バジェット以下（デフォルト15,000） | そのまま返される — 完全なMarkdownがエージェントに届く |
| バジェット超過 | 先頭＋末尾のウィンドウ（約75%が先頭／約25%が末尾、Markdownの行境界で切断）に、明示的な `[TRUNCATED]` フッターが付く。クリーンな全文はディスクに保存され、フッターにはそのファイルパスと、省略された中間部分をページングして読むための正確な `read_file` 呼び出しが示される |
| 2,000,000超 | 保存されるテキストは2 MBに制限される |

ページごとのバジェットは `config.yaml` の `web.extract_char_limit` で設定でき（デフォルト `15000`、2,000～500,000の範囲に制限）、エージェントはツールの `char_limit` 引数で呼び出しごとに引き上げることもできます。

各プロバイダーへのディスパッチには、実時間のタイムアウトも設定されています（`config.yaml` の `web.extract_timeout`、デフォルト `120` 秒、`0` で無効）。完了しないままレスポンスを開き続けるバックエンドは、ツール呼び出しを無期限に停止させる代わりに、URLごとのタイムアウトエラーを返します。

### 切り詰めが邪魔になるとき

抽出されたMarkdownではなくライブのDOMがどうしても必要な場合 — 例えば、抽出してもほとんどコンテンツが返らないJSを多用したページ — は、代わりに `browser_exec` を使ってください: `new_tab(url)` + `wait_for_load()` でページを開き、`page_info()` または `js()` によるDOM抽出で読み取ります。これはライブでレンダリング済みのページに対して動作します。

---

## 結果のキャッシュ

短時間内に繰り返されるウェブ呼び出しは、有料のバックエンドではなくキャッシュから提供されます — これにより、重複が起こりやすい2つのパターンでクレジットとレイテンシを節約できます: サブエージェントのファンアウト（複数の委譲先エージェントが同じトピックを調査する）と、エージェントが数分前に読んだページを再確認する場合です。

| 呼び出し | キャッシュ | スコープ |
|------|-------|-------|
| `web_search` — 同じクエリ（大文字小文字・空白を区別しない）、同じプロバイダー | インメモリのメモ | プロセスごと |
| `web_extract` — 同じURL、同じフォーマット、同じプロバイダー | 全文を `~/.hermes/cache/web/` に保存 | CLI、ゲートウェイ、cron、サブエージェントの各プロセス間で共有 |

同時に発生する同一の検索（並列のサブエージェントのファンアウトが同じクエリを一斉に送信する場合）は、**単一のバックエンドリクエストに統合** されます — 最初の呼び出し元が料金を負担し、残りはそのレスポンスを共有します。要求された検索件数の上限は10/20/50/100に切り上げてバケット化されるため、ほぼ同一のリクエスト（`limit=5` と `limit=8`）は1つのエントリを共有し、各呼び出し元はそれぞれ要求した件数を受け取ります。

キャッシュされるのは成功したレスポンスのみで、それぞれプロバイダーが報告した要求URLの下に保存されます（プロバイダーが要求URLを示さずに返したページは提供はされますがキャッシュされないため、部分的または順序が入れ替わったバッチによって、あるページが別のURLのキーで保存されることはありません）。失敗した場合は常にバックエンドで再試行され、一回限りのキー不要レスキューで提供されたレスポンスはキャッシュされず（次の呼び出しでは選択したバックエンドが再度試されます）、`security.website_blocklist` に一致するURLがキャッシュから提供されることはありません。キャッシュされた抽出結果は通常の切り詰めパイプラインを再実行するため、2回目の呼び出しで異なる `char_limit` を指定しても、同じ保存済みスクレイプを元に動作します。

**ローカル開発用のURLはキャッシュされません。** `localhost`、`127.0.0.1`、`*.local`、単一ラベルのLANホスト名、またはプライベート／リンクローカルのIP範囲（`192.168.*`、`10.*`、`172.16-31.*`）上のものはすべて抽出キャッシュを完全にバイパスします — 開発サーバー、ホットリロードのビルド、チャットGUIのアーティファクトプレビューは保存のたびに変わるため、キャッシュされたコピーでは古いビルドが表示されてしまいます。ローカルページの取得はすべてライブで行われます。（これらのURLにそもそも到達できるのは、`security.allow_private_urls` が有効な場合のみです。）

**公開インターネット経由でテストしていますか？** ステージングのデプロイやトンネルのURLは公開DNSであるため、ローカル開発のルールでは捕捉できません — それらを `web.cache_exempt_hosts` に列挙すれば、常にライブで取得されます。エントリは完全一致、`*.` ワイルドカード、またはドメインサフィックス（`mysite.dev` は `preview.mysite.dev` もカバー）として照合されます:

```yaml
# ~/.hermes/config.yaml
web:
  cache_exempt_hosts:
    - mysite.vercel.app
    - "*.ngrok-free.app"
```

```yaml
# ~/.hermes/config.yaml
web:
  cache_enabled: true      # default; set false to disable both caches
  cache_ttl_minutes: 20    # freshness window, clamped 1–1440
```

本当にライブなデータ（スコア、価格、速報ニュース）を調査していて、毎回新しい結果が必要な場合は、TTLを下げるか `web.cache_enabled: false` を設定してください。

---

## セットアップ

### `hermes tools` による簡易セットアップ

`hermes tools` を実行し、**Web Search & Extract** に移動して、プロバイダーを選びます。ウィザードは必要なURLまたはAPIキーの入力を求め、設定に書き込みます。

```bash
hermes tools
```

---

### Firecrawl（デフォルト）

フル機能の検索と抽出。ほとんどのユーザーに推奨します。

```bash
# ~/.hermes/.env
FIRECRAWL_API_KEY=fc-your-key-here
```

キーは [firecrawl.dev](https://firecrawl.dev) で取得できます。無料枠には月500クレジットが含まれます。

**セルフホストのFirecrawl:** クラウドAPIの代わりに、自前のインスタンスを指定します:

```bash
# ~/.hermes/.env
FIRECRAWL_API_URL=http://localhost:3002
```

`FIRECRAWL_API_URL` が設定されている場合、APIキーは任意です（`USE_DB_AUTHENTICATION=false` でサーバー認証を無効にできます）。

---

### SearXNG（無料、セルフホスト）

SearXNGは、70以上の検索エンジンから結果を集約する、プライバシーを尊重したオープンソースのメタ検索エンジンです。**APIキーは不要** — Hermesを稼働中のSearXNGインスタンスに向けるだけです。

SearXNGは **検索専用** です — `web_extract` には別の抽出プロバイダーが必要です。

#### 選択肢A — Dockerでセルフホスト（推奨） {#option-a--self-host-with-docker-recommended}

これにより、レート制限のないプライベートなインスタンスが手に入ります。

**1. 作業ディレクトリを作成:**

```bash
mkdir -p ~/searxng/searxng
cd ~/searxng
```

**2. `docker-compose.yml` を書く:**

```yaml
# ~/searxng/docker-compose.yml
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888/
    restart: unless-stopped
```

**3. コンテナを起動:**

```bash
docker compose up -d
```

**4. JSON APIフォーマットを有効にする:**

SearXNGはデフォルトでJSON出力を無効にして出荷されます。生成された設定をコピーして有効にします:

```bash
# Copy the auto-generated config out of the container
docker cp searxng:/etc/searxng/settings.yml ~/searxng/searxng/settings.yml
```

`~/searxng/searxng/settings.yml` を開きます。
`use_default_settings: true` が存在する場合、このファイルにはあなたによるオーバーライドだけが含まれます。その他の設定はすべて組み込みのデフォルトから継承されます。
HermesのためにJSONレスポンスを有効にするには、次のオーバーライドを追加します:

```yaml
search:
  formats:
    - html
    - json
```

`settings.yml` は次のようになるはずです:

```yaml
# Read the documentation before extending the defaults:
# https://docs.searxng.org/admin/settings/

use_default_settings: true

server:
  secret_key: "abcdef12345678"
  image_proxy: true

search:
  formats:
    - html
    - json
```

**5. 再起動して適用:**

```bash
docker cp ~/searxng/searxng/settings.yml searxng:/etc/searxng/settings.yml
docker restart searxng
```

**6. 動作を確認:**

```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"results\"])} results')"
```

`10 results` のような表示が出るはずです。`403 Forbidden` が出る場合、JSONフォーマットがまだ無効です — ステップ4を再確認してください。

**7. Hermesを設定:**

```bash
# ~/.hermes/.env
SEARXNG_URL=http://localhost:8888
```

次に、`~/.hermes/config.yaml` で検索バックエンドとしてSearXNGを選択します:

```yaml
web:
  search_backend: "searxng"
```

または `hermes tools` → Web Search & Extract → SearXNG で設定します。

---

#### 選択肢B — パブリックインスタンスを使う

パブリックなSearXNGインスタンスは [searx.space](https://searx.space/) に一覧があります。**JSONフォーマットが有効** なインスタンス（表に表示されます）で絞り込んでください。

```bash
# ~/.hermes/.env
SEARXNG_URL=https://searx.example.com
```

:::caution Public instances
パブリックインスタンスにはレート制限があり、稼働状況も一定せず、いつでもJSONフォーマットを無効にする可能性があります。本番利用では、セルフホストを強く推奨します。
:::

---

#### SearXNGを抽出プロバイダーと組み合わせる

SearXNGは検索を担当します。`web_extract` には別のプロバイダーが必要です。機能ごとのキーを使います:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # or tavily, perplexity, keenable, exa, parallel
```

この設定では、Hermesはすべての検索クエリにSearXNGを、URL抽出にFirecrawlを使います — 無料の検索と高品質な抽出を組み合わせられます。

---

### Tavily

AI向けに最適化された検索と抽出。`hermes tools` でTavilyを選択する（または `web.backend: tavily` を設定する）と、アカウントなしの **キー不要** で利用できます（レート制限あり）。より高い上限が必要な場合はAPIキーを設定してください。

```bash
# optional — skip this for keyless access after selecting Tavily
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

キーは [app.tavily.com](https://app.tavily.com/home) で取得できます。[Tavily keyless](https://docs.tavily.com/documentation/keyless) を参照してください。

---

### Perplexity

[PerplexityのSearch API](https://docs.perplexity.ai/docs/search/quickstart) は、Perplexity独自のインデックスからランク付けされ日付の付いた結果を返します（`web_search`）。`web_extract` では、公式の `pplx` CLIと同じ、クエリに関連する *スニペット* の経路を使います: ページ全体をそのままダンプするのではなく、各ページの重要な箇所が、省略部分を `…` で示した形で得られます — ページ全体が必要な場合は、`web.extract_backend` にFirecrawl / Exa / Parallelを選んでください。キーが必要なのはPerplexity自体への直接接続のみです。マネージドのNous経路（ゲストを含む任意のNousアイデンティティ）では、キーなしで `web_search` が提供されます。

```bash
# ~/.hermes/.env
PERPLEXITY_API_KEY=pplx-your-key-here
```

キーは [perplexity.ai/account/api](https://www.perplexity.ai/account/api) で取得できます。プロキシ経由でルーティングするには `PERPLEXITY_BASE_URL` を設定します。

---

### Exa

意味理解を備えたニューラル検索。リサーチや概念的に関連するコンテンツの発見に適しています。

```bash
# ~/.hermes/.env
EXA_API_KEY=your-exa-key-here
```

キーは [exa.ai](https://exa.ai) で取得できます。無料枠には月1,000検索が含まれます。

---

### Parallel

ディープリサーチ機能を備えた、AIネイティブな検索と抽出。

```bash
# ~/.hermes/.env
PARALLEL_API_KEY=your-parallel-key-here
```

アクセスは [parallel.ai](https://parallel.ai) で取得できます。

---

### xAI (Grok) {#xai-grok}

`web_search` を、Responses API上のGrokのサーバーサイド [web_searchツール](https://docs.x.ai/developers/tools/web-search) 経由でルーティングします。実際の検索はGrokが行い、上位の結果を構造化されたJSONとして返します。

どちらの認証情報経路でも動作します — 新しい環境変数も新しいセットアップウィザードも不要です:

```bash
# ~/.hermes/.env (env-var path)
XAI_API_KEY=sk-xai-your-key-here
```

またはSuperGrokのサブスクライバーの場合:

```bash
hermes auth add xai-oauth
```

次に、検索バックエンドとしてxAIを選択します:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "xai"
```

**任意の設定項目:**

```yaml
web:
  backend: "xai"
  xai:
    model: grok-build-0.1        # reasoning model required by web_search (default)
    allowed_domains:             # optional, max 5 — mutex with excluded_domains
      - arxiv.org
    excluded_domains:            # optional, max 5
      - example-spam.com
    timeout: 90                  # seconds (default)
```

**検索専用** です — `web_extract` も必要な場合は、Firecrawl / Tavily / Keenable / Exa / Parallelと組み合わせてください。401が返った場合、プロバイダーはOAuthトークンの強制リフレッシュを1回だけ行って再試行します（有効期間中の失効や、事前の有効期限チェックではデコードできない不透明なトークンに対応するため）。環境変数による認証情報では再試行は行われません。

:::caution Trust model
検索エンジンの結果をそのまま返すインデックスベースのプロバイダー（Brave、Tavily、Exa）とは異なり、xAIはLLMがどのURLを提示するかを選び、タイトルと説明も自ら書きます。クエリの *内容* が出力に影響するため、悪意を持って作られたクエリ（例えば、エージェントが拾った信頼できない上流の入力から注入されたもの）によって、原理的にはGrokに攻撃者が選んだURLを出力させることが可能です。返されたURLは、モデルが生成したあらゆるリンクと同じように扱ってください — 特にクエリが信頼できない入力から来た場合は、取得する前に検証してください。
:::

### OpenAI Native (Codex Responses) {#openai-native}

Codex Responsesエンドポイント（ChatGPT/Codexのサブスクリプション）上で、OpenAIのプロバイダー実行型 `web_search` ツールを宣言します。モデルがサーバーサイドで検索を実行し、その結果を自身の回答に組み込みます — このモードでは、Hermesがクライアントサイドで検索を実行することはありません。

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "openai-native"
```

要件と適用範囲:

- **認証情報**: openai-codexのOAuthログイン（`hermes auth add openai-codex`）。このバックエンドには独自のAPIキーはなく、ログインしていない場合は単に利用できません。
- **トランスポート**: 組み込みツールを公開しているのはCodex Responsesエンドポイントのみです。それ以外のトランスポート — カスタムのOpenAI互換 `base_url`、またはOpenAI以外のモデル — では、エンドポイントがツールをホストしていると当てにできないため、クライアントサイドの `web_search` 関数はそのまま残されます。そうした場合は `web.search_backend` に通常のプロバイダーを指定してください。
- **検索のみ**: 組み込みツールがカバーするのは検索であり、抽出ではありません。`web_extract` も必要な場合は、`web.extract_backend` を通じてFirecrawl（または抽出に対応した別のバックエンド）と組み合わせてください。

**どちらにしてもツールは1つ。** このバックエンドを選択すると、クライアントサイドの `web_search` 関数が組み込みツールに1対1で置き換えられます — 追加で付与されるわけではありません。ツールセットに `web_search` がないセッションに、サーバーサイド検索が注入されることはありません。

---

## 設定

### 単一バックエンド

すべてのウェブ機能に1つのプロバイダーを設定します:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | brave-free | ddgs | tavily | perplexity | keenable | exa | parallel | xai
```

### 機能ごとの設定 {#per-capability-configuration}

検索と抽出で異なるプロバイダーを使います。これにより、無料の検索（SearXNG）と有料の抽出プロバイダー、あるいはその逆を組み合わせられます:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # used by web_search
  extract_backend: "firecrawl"  # used by web_extract
```

機能ごとのキーが空の場合、その機能は `web.backend` にフォールバックします。**共有** のウェブ選択（`web.backend` またはマネージドの `hermes tools` の行）が一度も書き込まれていない場合に限り、存在するAPIキー／URLからバックエンドが自動検出されます — 共有の選択が存在すれば、ランタイムは常にそれを使い、`.env` にキーを追加してもウェブのトラフィックの経路は変わりません。機能ごとのキーはその機能にのみ影響します: `web.extract_backend` だけを設定した場合、`web_search` は自動検出されたバックエンドのままです。

**優先順位（機能ごと）:**
1. `web.search_backend` / `web.extract_backend`（機能ごとの明示指定。`nous` = その機能のみマネージドTool Gateway、[Tool Gateway](./tool-gateway.md#mixing-your-own-key-and-the-gateway-web-only) を参照）
2. `web.backend`（共有のフォールバック。`nous` = マネージドTool Gateway）
3. 環境変数からの自動検出（共有の選択が書き込まれていない場合）

### 自動検出

共有のバックエンドが **一度も** 選択されていない場合（あなたまたは `hermes tools` によって `web.backend` が書き込まれていない場合）、Hermesは設定されている認証情報に基づいて、利用可能な最初のものを選びます:

| 存在する認証情報 | 自動選択されるバックエンド |
|--------------------|-----------------------|
| `TAVILY_API_KEY` | tavily |
| `PERPLEXITY_API_KEY` | perplexity |
| `EXA_API_KEY` | exa |
| `PARALLEL_API_KEY` | parallel |
| `FIRECRAWL_API_KEY` または `FIRECRAWL_API_URL` | firecrawl |
| Nous Tool Gatewayが利用可能（Portalサブスクリプション） | Tool Gateway経由のマネージドウェブ検索 |
| `SEARXNG_URL` | searxng |
| `BRAVE_SEARCH_API_KEY` | brave-free |
| `ddgs` パッケージがインポート可能 | ddgs |
| Nousアイデンティティが存在（無料枠や匿名ゲストを含む任意のアカウント） | マネージドウェブ検索（`web_search` のみ）。抽出と下記のキー不要リングは変わらない |
| *（何も設定されていない）* | キー不要リング: exa / parallel / firecrawl / keenable（ラウンドロビン） |

**キー不要の無料枠リング:** 上記の認証情報が *何も* 存在しない場合、リクエストはリングに属するベンダーの公開無料枠（Exa、Parallel、Firecrawl、Keenable）の間でローテーションされるため、新規インストールでも設定ゼロでウェブツールが動作します — また、レート制限を受けたリクエストや、ベンダーが拒否した検索は、リング内の次のベンダーに自動的にフェイルオーバーします。`hermes tools` で1つのベンダーに固定するとローテーションが止まります（その場合、リングはスロットリング時のフェイルオーバーの順番としてのみ使われます）。すべての無料枠は、バースト負荷時にはベンダー側でレート制限されますが、継続的な通常の利用であれば問題なく通ります。この枠を無効にするには `web.keyless_fallback: false` を設定します — 無効にして認証情報もない場合、プロバイダーが設定されるまでウェブツールは利用できません。Nousアイデンティティ（ゲストを含む）はこのスイッチの影響を受けません: マネージドの `web_search` は引き続き適用されますが、`web_extract` には設定済みのプロバイダーが必要です。

**キー付きバックエンド向けの一回限りのキー不要レスキュー:** 選択した／キー付きのバックエンド — Nous Tool Gatewayの経路（`web.backend: nous`）を含む — で呼び出しが失敗した場合（不正なキー、障害、到達不能なゲートウェイ、上流の5xx）、その1回の呼び出しはエラーにする代わりに、キー不要の無料枠リングで自動的に再試行されます — 結果には、どのベンダーがなぜ応答したかが記録されます（`rescued_from` / `backend_error`）。このフェイルオーバーが固定されることはありません: 直後の `web_search`/`web_extract` 呼び出しでは、選択したバックエンドが再度試されます。無効にするには `web.keyless_rescue: false` を設定します（`keyless_fallback` が無効な場合も常に無効になります）。

xAI Web Searchは自動検出チェーンに **含まれません** — `XAI_API_KEY` が設定されている（またはxAI Grok OAuthでサインインしている）からといって、ウェブのトラフィックが自動的にxAI経由になることはありません。これらの認証情報は推論／TTS／画像生成にも使われるため、ユーザーがウェブには別のバックエンドを望む可能性があるからです。`web.backend: "xai"` で明示的にオプトインしてください。

---

## セットアップを確認する

`hermes setup` を実行して、どのウェブバックエンドが検出されたか確認します:

```
✅ Web Search & Extract (searxng)
```

ソースチェックアウトの場合は、
[PMの有効化](/docs/reference/package-management#developer-workflow) の後にモジュールを確認することもできます。確認したいウェブ設定を持つホームを使ってください:

```bash
# From the Hermes source checkout, in a clean shell
source ./activate
python -m tools.web_tools
```

これにより、アクティブなバックエンドとそのステータスが表示されます:

```
✅ Web backend: searxng
   Using SearXNG (search only): http://localhost:8888
```

---

## トラブルシューティング

### `web_search` が `{"success": false}` を返す

- `SEARXNG_URL` に到達できるか確認: `curl -s "http://localhost:8888/search?q=test&format=json"`
- HTTP 403が返る場合、JSONフォーマットが無効です — `settings.yml` の `formats` リストに `json` を追加して再起動してください
- 接続エラーが返る場合、コンテナが稼働していない可能性があります: `docker ps | grep searxng`

### `web_extract` が「search-only backend」と言う

SearXNGはURLコンテンツを抽出できません。`web.extract_backend` を、抽出をサポートするプロバイダーに設定してください:

```yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"  # or tavily / perplexity / keenable / exa / parallel
```

### SearXNGが0件の結果を返す

一部のパブリックインスタンスは、特定の検索エンジンやカテゴリを無効にしています。次を試してください:
- 別のクエリ
- [searx.space](https://searx.space/) の別のパブリックインスタンス
- 信頼できる結果を得るために、自前のインスタンスをセルフホストする

### パブリックインスタンスでレート制限を受ける

セルフホストのインスタンスに切り替えてください（上記の [選択肢A](#option-a--self-host-with-docker-recommended) を参照）。Dockerなら、自前のインスタンスにレート制限はありません。

### `web_extract` が `[TRUNCATED]` フッター付きで切り詰められたコンテンツを返す

文字数バジェットを超えるページでは想定どおりの動作です。フッターには、クリーンな全文を保持するディスク上のファイル名と、省略された中間部分をページングして読むための正確な `read_file` 呼び出しが示されます。インラインでより多く表示するには、`config.yaml` の `web.extract_char_limit` を引き上げるか、呼び出し時により大きな `char_limit` を渡してください。
