---
title: ウェブ検索＆抽出
description: 複数のバックエンドプロバイダー（無料でセルフホストできるSearXNGを含む）で、ウェブ検索、ページコンテンツの抽出、ウェブサイトのクロールを行います。
sidebar_label: ウェブ検索
sidebar_position: 6
---

# ウェブ検索＆抽出

Hermes Agentには、複数のプロバイダーに対応した、モデルから呼び出せる2つのウェブツールが含まれます:

- **`web_search`** — ウェブを検索し、ランク付けされた結果を返す
- **`web_extract`** — 1つ以上のURLから読みやすいコンテンツを取得・抽出する（バックエンドが提供する場合は組み込みのディープクロールにも対応）

どちらも単一のバックエンド選択で設定されます。プロバイダーは `hermes tools` で選ぶか、`config.yaml` に直接設定します。再帰的なクロール機能（Firecrawl/Tavily）は、独立した `web_crawl` ツールとしてではなく `web_extract` を通じて公開されます。

## バックエンド

| プロバイダー | 環境変数 | 検索 | 抽出 | クロール | 無料枠 |
|----------|---------|--------|---------|-------|-----------|
| **Firecrawl**（デフォルト） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ | 500クレジット/月 |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — | ✔ 無料（セルフホスト） |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ | 1,000検索/月 |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — | 1,000検索/月 |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — | 有料 |

**機能ごとの分割:** 検索と抽出で異なるプロバイダーを個別に使えます — 例えば検索にはSearXNG（無料）、抽出にはFirecrawlといった具合です。下記の [機能ごとの設定](#per-capability-configuration) を参照してください。

:::tip Nous Subscribers
有料の [Nous Portal](https://portal.nousresearch.com) サブスクリプションをお持ちの場合、ウェブ検索と抽出はマネージドFirecrawlを通じて **[Tool Gateway](tool-gateway.md)** 経由で利用できます — APIキーは不要です。`hermes tools` を実行して有効にしてください。
:::

---

## `web_extract` が長いページを処理する方法 {#how-web_extract-handles-long-pages}

バックエンドは生のページMarkdownを返しますが、これは膨大になることがあります（フォーラムのスレッド、ドキュメントサイト、コメントが埋め込まれたニュース記事など）。コンテキストウィンドウを使い物になる状態に保ち、コストを抑えるため、`web_extract` は返されたコンテンツを、エージェントに渡す前に **`web_extract` 補助モデル** に通します。挙動は純粋にサイズに基づいて決まります:

| ページサイズ（文字数） | 何が起きるか |
|------------------------|--------------|
| 5,000未満 | そのまま返される — LLM呼び出しなし、完全なMarkdownがエージェントに届く |
| 5,000～500,000 | `web_extract` 補助モデルによる単一パスの要約。出力は約5,000文字に制限 |
| 500,000～2,000,000 | チャンク化: 10万文字のチャンクに分割し、各チャンクを並列で要約してから、最終要約（約5,000文字）を合成 |
| 2,000,000超 | 拒否され、絞り込んだ抽出指示付きの `web_crawl` か、より具体的なソースを使うようヒントが返される |

要約は、引用、コードブロック、重要な事実を元の書式のまま保持します — これは言い換えではなく、コンテンツの圧縮機です。要約が失敗するかタイムアウトした場合、Hermesは役に立たないエラーを返す代わりに、生コンテンツの最初の約5,000文字にフォールバックします。

### 要約を行うのはどのモデルですか？

`web_extract` の補助タスクです。デフォルト（`auxiliary.web_extract.provider: "auto"`）では、これはあなたの **メインチャットモデル** です — `hermes model` と同じプロバイダー、同じモデルです。ほとんどの構成では問題ありませんが、高価な推論モデル（Opus、MiniMax M2.7など）では、長いページの抽出が発生するたびに無視できないコストが加算されます。

メインモデルに関係なく、抽出の要約を安価で高速なモデルにルーティングするには:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  web_extract:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 360       # 秒。要約のタイムアウトが発生する場合は引き上げる
```

または対話的に選びます: `hermes model` → **Configure auxiliary models** → `web_extract`。

完全なリファレンスとタスクごとのオーバーライドのパターンについては、[補助モデル](/docs/user-guide/configuration#auxiliary-models) を参照してください。

### 要約が邪魔になるとき

要約されていない生のページコンテンツが必要な場合 — 例えば、LLMの要約が重要なフィールドを落としてしまうような構造化されたページをスクレイピングする場合 — は、代わりに `browser_navigate` + `browser_snapshot` を使ってください。ブラウザツールは、補助モデルによる書き換えなしにライブのアクセシビリティツリーを返します（巨大なページでは独自の8,000文字のスナップショット上限が適用されます）。

---

## セットアップ

### `hermes tools` による簡易セットアップ

`hermes tools` を実行し、**Web Search & Extract** に移動して、プロバイダーを選びます。ウィザードは必要なURLまたはAPIキーの入力を求め、設定に書き込みます。

```bash
hermes tools
```

---

### Firecrawl（デフォルト）

フル機能の検索、抽出、クロール。ほとんどのユーザーに推奨します。

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

SearXNGは **検索専用** です — `web_extract`（そのクロールモードを含む）には別の抽出プロバイダーが必要です。

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
# 自動生成された設定をコンテナからコピーして取り出す
docker cp searxng:/etc/searxng/settings.yml ~/searxng/searxng/settings.yml
```

`~/searxng/searxng/settings.yml` を開き、`formats` ブロック（84行目あたり）を見つけます:

```yaml
# 変更前（デフォルト — JSON無効）:
formats:
  - html

# 変更後（Hermes向けにJSONを有効化）:
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

SearXNGは検索を担当します。`web_extract`（ディープクロールモードを含む）には別のプロバイダーが必要です。機能ごとのキーを使います:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # または tavily、exa、parallel
```

この設定では、Hermesはすべての検索クエリにSearXNGを、URL抽出にFirecrawlを使います — 無料の検索と高品質な抽出を組み合わせられます。

---

### Tavily

潤沢な無料枠を備えた、AI向けに最適化された検索、抽出、クロール。

```bash
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

キーは [app.tavily.com](https://app.tavily.com/home) で取得できます。無料枠には月1,000検索が含まれます。

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

## 設定

### 単一バックエンド

すべてのウェブ機能に1つのプロバイダーを設定します:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | tavily | exa | parallel
```

### 機能ごとの設定 {#per-capability-configuration}

検索と抽出で異なるプロバイダーを使います。これにより、無料の検索（SearXNG）と有料の抽出プロバイダー、あるいはその逆を組み合わせられます:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # web_searchが使用
  extract_backend: "firecrawl"  # web_extract（およびそのディープクロールモード）が使用
```

機能ごとのキーが空の場合、両方とも `web.backend` にフォールバックします。`web.backend` も空の場合、存在するAPIキー／URLからバックエンドが自動検出されます。

**優先順位（機能ごと）:**
1. `web.search_backend` / `web.extract_backend`（機能ごとの明示指定）
2. `web.backend`（共有のフォールバック）
3. 環境変数からの自動検出

### 自動検出

バックエンドが明示的に設定されていない場合、Hermesは設定されている認証情報に基づいて、利用可能な最初のものを選びます:

| 存在する認証情報 | 自動選択されるバックエンド |
|--------------------|-----------------------|
| `FIRECRAWL_API_KEY` または `FIRECRAWL_API_URL` | firecrawl |
| `PARALLEL_API_KEY` | parallel |
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `SEARXNG_URL` | searxng |

---

## セットアップを確認する

`hermes setup` を実行して、どのウェブバックエンドが検出されたか確認します:

```
✅ Web Search & Extract (searxng)
```

または、CLIで確認します:

```bash
# venvを有効化し、ウェブツールモジュールを直接実行
source ~/.hermes/hermes-agent/.venv/bin/activate
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
  extract_backend: "firecrawl"  # または tavily / exa / parallel
```

### SearXNGが0件の結果を返す

一部のパブリックインスタンスは、特定の検索エンジンやカテゴリを無効にしています。次を試してください:
- 別のクエリ
- [searx.space](https://searx.space/) の別のパブリックインスタンス
- 信頼できる結果を得るために、自前のインスタンスをセルフホストする

### パブリックインスタンスでレート制限を受ける

セルフホストのインスタンスに切り替えてください（上記の [選択肢A](#option-a--self-host-with-docker-recommended) を参照）。Dockerなら、自前のインスタンスにレート制限はありません。

### `web_extract` が「summarization timed out」の注記とともに切り詰められたコンテンツを返す

補助モデルが、設定されたタイムアウト内に要約を完了できませんでした。次のいずれかを行ってください:

- `config.yaml` の `auxiliary.web_extract.timeout` を引き上げる（新規インストールではデフォルト360秒、キーがない場合は30秒）
- `web_extract` の補助タスクをより高速なモデル（例: `google/gemini-3-flash-preview`）に切り替える — [`web_extract` が長いページを処理する方法](#how-web_extract-handles-long-pages) を参照
- 要約が適切でないページでは、代わりに `browser_navigate` を使う

---

## 任意のスキル: `searxng-search`

エージェントが（例えばウェブツールセットが利用できないときのフォールバックとして）`curl` で直接SearXNGを使う必要がある場合は、任意のスキル `searxng-search` をインストールします:

```bash
hermes skills install official/research/searxng-search
```

これにより、エージェントに次の方法を教えるスキルが追加されます:
- `curl` またはPythonでSearXNG JSON APIを呼び出す
- カテゴリ（`general`、`news`、`science` など）でフィルタリングする
- ページネーションとエラーケースを処理する
- SearXNGに到達できないときに優雅にフォールバックする
