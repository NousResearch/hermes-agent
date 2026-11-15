---
sidebar_position: 3
---

# モデルの設定

Hermes は 2 種類のモデルスロットを使用します。

- **メインモデル** — エージェントが思考に用いるモデルです。すべてのユーザーメッセージ、すべてのツール呼び出しループ、すべてのストリーミング応答がこのモデルを通ります。
- **補助モデル** — エージェントが切り出すサイドジョブ用のモデルです。コンテキスト圧縮、ビジョン（画像分析）、承認スコアリング、セッションタイトル生成、`/review` によるレビュー、ボイスチャットの応答などがあります。それぞれが独自のスロットを持ち、個別に上書きできます。ダッシュボードで表示されるスロットは[補助モデルの設定](#setting-auxiliary-models)に一覧があります。

このページでは、両方をダッシュボードから設定する方法を説明します。設定ファイルや CLI を使いたい場合は、ページ下部の[代替手段](#alternative-methods)に移動してください。クラウドプロバイダーではなく自分のマシンでモデルを実行する場合は、[ローカルモデル](/docs/user-guide/local-models)を参照してください。

:::tip 最速の方法: Nous Portal
[Nous Portal](./features/tool-gateway.md) は、1 つのサブスクリプションで 300 以上のモデルを提供します。新規インストール時に `hermes setup --portal` を実行すると、ログインと Nous のプロバイダー設定を 1 つのコマンドで行えます。接続状況は `hermes portal info` で確認できます。

- Portal の購読者は、**トークン課金のプロバイダーが 10% 割引**にもなります。
:::

:::note `model:` スキーマ — 空文字列とマッピング
新規インストール直後の同梱の既定設定では `model: ""`（「まだ設定されていない」ことを意味する空文字列のセンチネル）になっています。初めて `hermes setup` または `hermes model` を実行すると、このキーはその場で `provider`、`default`、`base_url`、`api_mode` のサブキーを持つマッピングにアップグレードされます。これは、このページ全体や [`profiles.md`](./profiles.md) / [`configuration.md`](./configuration.md) で示されている形式です。`config.yaml` に空文字列が見られる場合は、`hermes model` を実行する（またはダッシュボードで **Change** をクリックする）と、Hermes が辞書形式を書き込みます。
:::

## Models ページ

ダッシュボードを開き、サイドバーの **Models** をクリックします。次の 2 つのセクションが表示されます。

1. **Model Settings** — 上部のパネルで、モデルをスロットに割り当てます。
2. **Usage analytics** — 選択した期間中にセッションを実行したすべてのモデルを、トークン数、コスト、機能バッジとともにランク付けして表示するカードです。

![Models ページの概要](/img/docs/dashboard-models/overview.png)

最上部のカードが **Model Settings** パネルです。メインの行には、エージェントが新しいセッション用に起動するモデルが常に表示されます。**Change** をクリックするとピッカーが開きます。

## メインモデルの設定

Main model の行で **Change** をクリックします。

![モデルピッカーダイアログ](/img/docs/dashboard-models/picker-dialog.png)

ピッカーは 2 つの列で構成されています。

- **左** — 認証済みのプロバイダー。セットアップ済みのプロバイダー（API キーを設定済み、OAuth 済み、またはカスタムエンドポイントとして定義済み）のみがここに表示されます。目的のプロバイダーが見当たらない場合は、**Keys** に移動してクレデンシャルを追加してください。
- **右** — 選択したプロバイダー向けの厳選されたモデルリスト。これらは Hermes がそのプロバイダーに推奨するエージェント向けモデルであり、生の `/models` の一覧（OpenRouter では TTS、画像生成、リランカーを含む 400 以上のモデルが含まれます）ではありません。

フィルターボックスに入力すると、プロバイダー名、スラッグ、モデル ID で絞り込めます。

モデルを選んで **Switch** を押すと、Hermes は `~/.hermes/config.yaml` の `model` セクションにそれを書き込みます。**これは新しいセッションにのみ適用されます** — すでに開いているチャットタブは、開始時のモデルをそのまま使い続けます。現在のチャットをホットスワップするには、その中で `/model` スラッシュコマンドを使用してください。

### サブスクリプションの速度と拡張コンテキスト

速度は推論エフォートとは別の設定です。対応するサブスクリプションモデルでは、**Fast** は Priority を、**Ultrafast** は Ultrafast ティアをリクエストします。一方を選択すると、もう一方は解除されます。選択したティアはモデルごとに記憶され、新しいチャットにも引き継がれます。プロファイルの既定値は **Settings → Model → Main model → Speed**（`agent.service_tier`: `normal`、`fast`、`ultrafast`）にあります。Ultrafast は、それを提供するモデルにのみ表示され、サブスクリプションの場合はアカウントの Codex カタログがそのモデルに対して掲載している場合に限られます。従来の設定は **Use standard speed** で解除できます。リクエストが受け付けられても、OpenAI が要求したティアで処理したことは保証されません。

Hermes は、検出された対象の GPT-6.1 Sol モデルから **GPT-6.1 Sol-900k** の選択肢を作成します。`-900k` サフィックスはローカルのコンテキスト選択であり、OpenAI に送信されるモデル ID からは除去されます。Hermes はその予算をアカウントカタログの最大値に制限し、Codex は独自の予約分を差し引いた後、より小さな利用可能コンテキストを報告する場合があります。新しいモデルが表示されない場合は **Refresh models** を使用してください。検出に失敗した場合は、成功したアカウントカタログとしてキャッシュせず、短期間有効な互換性ヒントを使用します。

### セッション途中での切り替えとコンテキストの警告

**アクティブなセッション内で**モデルを切り替えると（Herm TUI のモデルピッカー、`hermes` CLI、Telegram/Discord での `/model`）、Hermes は**次のメッセージ**が新しいモデルのウィンドウに対して**事前のコンテキスト圧縮**を実行するかどうかを見積もります。セッションがすでにそのモデルの圧縮しきい値付近またはそれを超えている場合（[コンテキスト圧縮](./configuration.md#context-compression)を参照）、切り替えの応答に警告が含まれます。これは高価なモデルの通知に使われるのと同じ `warning_message` の経路です。切り替え自体はすぐに適用され、圧縮は**切り替え後の最初のユーザーメッセージ**で、モデルが応答する前に実行されます。

:::warning セッション途中の切り替えはプロンプトキャッシュをリセットします
プロンプトキャッシュはリクエストを処理するモデルに紐付けられているため、会話の途中でのモデル変更 — 明示的な `/model` による切り替え、[自動フォールバック](./features/fallback-providers.md)、または[クレデンシャルプール](./features/credential-pools.md)による別アカウントへのローテーション — はいずれも、次のメッセージで会話全体をキャッシュ済み（約 75〜90% 割引）の料金ではなく、通常の入力トークン料金で読み直すことを意味します。長いセッションでは、この 1 回限りの読み直しが 2 つのモデル間のトークン単価の差を大きく上回ることがあります。必要なときは切り替えてかまいませんが、できれば会話の早い段階か、新しいセッションを開始した直後に行ってください。

明示的な `/model` による切り替えでは、エージェントが自身の実行環境を把握できるよう、システムプロンプト内の `Model:`（および `Provider:`）の行も書き換えられます。これは意図的な 1 回限りの変更です。切り替え後の最初のリクエストでキャッシュ済みのプレフィックスを 1 回だけ外し、システムプロンプト以降はすべてバイト単位で同一のままです。自動フォールバックは同じ行をメモリ上でのみ書き換えます。保存されたセッションにはプライマリのラベルが残るため、プライマリが復元されるとバイト単位で同一のプロンプトが再生されます。
:::

この 1 回限りの読み直しコストがあるため、ライブセッションがすでに大きなコンテキスト（既定: **100,000 トークン**、プロバイダーが最後に課金したプロンプトサイズから測定）を保持している場合、Hermes はセッション途中の切り替えを適用する前に**明示的な確認**を求めます。この確認は、ライブセッションが切り替わるあらゆる場所 — CLI と TUI の `/model` コマンドおよびピッカー、アクティブなエージェントのいるチャットで入力されたゲートウェイの `/model` — で、高価なモデルやデータ学習の警告と同じ選択ガードのプロンプトとして表示されます。`config.yaml` で調整または無効化できます。

```yaml
model:
  # Ask before mid-session switches when the session exceeds this many
  # context tokens (the next reply re-reads them uncached). 0 disables.
  switch_context_confirm_tokens: 100000
```

すでに使用中のモデルを再選択した場合は確認が表示されません（キャッシュは温かいままです）。また、測定済みのコンテキストがないセッション（新規セッション、ライブでない画面）は対象外です。

### 無人実行時のデータ学習ティア

`-contributor` サフィックスの付いたモデル（例: `muse-spark-1.2-contributor`、`muse-spark-1.3-contributor`）は、ベンダーがプロンプトと補完を学習に使用する可能性があるため割引されています。インタラクティブなモデル選択では常に確認プロンプトが表示されます。Kanban ワーカーや cron エージェントなどの非インタラクティブな起動経路は、その確認を行えないため、安全側に倒して失敗します。

無人実行のワークロードのデータが学習に使われても問題ない場合は、永続的な承認を記録します。

```bash
hermes config set security.allow_data_training_tiers_noninteractive true
```

Hermes は無人実行で起動するたびに、データポリシーの警告全文と承認キーを出力し続けるため、ワーカーのログには監査証跡が残ります。この設定は、高価なモデルやプロバイダールーティングの警告を承認するものではなく、インタラクティブな確認プロンプトの代わりにもなりません。取り消すには `hermes config unset security.allow_data_training_tiers_noninteractive` を実行します。

## 補助モデルの設定 {#setting-auxiliary-models}

**Show auxiliary** をクリックすると、補助タスクごとに 1 行ずつ表示されます（プラグインがさらに登録できます）。

![補助パネルを展開した状態](/img/docs/dashboard-models/auxiliary-expanded.png)

すべての補助タスクは既定で `auto` になっています。これは、Hermes がそのジョブにもメインモデルを試すことを意味します。その経路が利用できない場合や、容量系のエラーが発生した場合、`auto` はタスク固有の `auxiliary.<task>.fallback_chain` に従い、次にメインの `fallback_providers` / `fallback_model` チェーンに従います。設定していないプロバイダーを推測することはありません。メインプロバイダーが選択されていてフォールバックが宣言されていない場合、そのサイドタスクは、たまたまログインしている別アカウントに課金されるのではなく、警告とともにスキップされます（Hermes の組み込みの検出チェーンは、メインプロバイダーがまったく選択されていない場合にのみ実行されます）。サイドジョブにより安価または高速なモデルを使いたい場合に、特定のタスクを上書きしてください。

### よくある上書きパターン

| タスク | 上書きする場面 |
|---|---|
| **Title Gen** | メインモデルと揃えることよりも、タイトル生成の遅延やコストが重要な場合。実績のある flash モデルを固定するか、`auxiliary.title_generation.prefer_fast_model: true` を設定して Hermes にプロバイダーの高速ティアを選ばせます。 |
| **Vision** | メインモデルがビジョンに対応していない場合。`google/gemini-2.5-flash` または `gpt-4o-mini` を指定します。 |
| **Compression** | コンテキストを要約するだけのために Opus/M2.7 で推論トークンを浪費している場合。高速なチャットモデルなら 1/50 のコストで同じ仕事をこなします。 |
| **Approval** | `approval_mode: smart` 向け。高速／安価なモデル（haiku、flash、gpt-5-mini）が、低リスクなコマンドを自動承認するかどうかを判断します。ここで高価なモデルを使うのは無駄です。 |
| **Review** | `/review` のレビュアーサブエージェント。`auto` はメインモデルを使用します。レビュー用に別の（多くの場合より強力な）モデルを固定できます。[レビューモデル](./features/delegation.md#review-model)を参照してください。 |
| **MCP** | MCP ツールルーティング。通常は `auto` で問題ありません。 |
| **Voice Chat** | 音声によるボイスチャットの応答。メインより高速なモデルを指定します。ボイスのターンでは既定で推論がオフになり、GPT-Live のボイスチャットはこのスロットを無視します。[ボイスチャットモデル](./features/voice-mode.md#voice-chat-model)を参照してください。 |
| **Triage Specifier** | 大まかな 1 行の内容を具体的な仕様に展開する Kanban のトリアージ仕様化（`hermes kanban specify`）をルーティングします。安価で有能なモデルが適しています。 |
| **Kanban Decomposer** | Kanban のタスク分解をルーティングします。トリアージタスクを、専門プロファイル向けの子タスクのグラフに分割します。 |
| **Profile Describer** | プロファイル説明の生成（`hermes profile describe --auto` / ダッシュボードの自動生成ボタン）をルーティングします。短く安価な呼び出しです。 |
| **Curator** | キュレーターによるスキル使用状況のレビューパスをルーティングします。推論モデルでは数分かかることがあるため、より安価な補助モデルを使う価値があることが多いです。 |

ダッシュボードに表示されないタスクを含む、`config.yaml` での補助モデルのその他のオプションについては、[補助モデル](./configuration.md#auxiliary-models)を参照してください。

### タスクごとの上書き

任意の補助行で **Change** をクリックします。同じピッカーが開き、同じ動作をします。プロバイダーとモデルを選んで Switch を押してください。行が `auto (use main model)` ではなく `provider · model` を表示するように更新されます。

### すべてを auto にリセット

チューニングしすぎて最初からやり直したい場合は、補助セクション上部の **Reset all to auto** をクリックします。すべてのスロットがメインモデルを使用する状態に戻ります。

## 「Use as」ショートカット

ページ上のすべてのモデルカードには **Use as** ドロップダウンがあります。これは最短の経路です。analytics に表示されているモデルを選び、**Use as** をクリックすれば、メインスロットまたは任意の特定の補助タスクにワンクリックで割り当てられます。

![Use as ドロップダウン](/img/docs/dashboard-models/use-as-dropdown.png)

ドロップダウンには次の項目があります。

- **Main model** — メイン行で Change をクリックするのと同じです。
- **All auxiliary tasks** — このモデルをページ上のすべての補助スロットに一度に割り当てます。すべてのサイドジョブを安価な flash モデルにまとめたいときに便利です。
- **個別のタスクオプション** — Vision、Review、Compression など。各タスクに現在割り当てられているモデルには `current` と表示されます。

カードが現在いずれかに割り当てられている場合は `main` または `aux · <task>` のバッジが付くため、過去に使ったモデルのうちどれがどこに紐付いているかを一目で確認できます。

## `config.yaml` に書き込まれる内容

ダッシュボード経由で保存すると、Hermes は `~/.hermes/config.yaml` に書き込みます。

**メインモデル:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # cleared on provider switch
  api_mode: chat_completions
```

**補助の上書き（例 — vision を gemini-flash に）:**
```yaml
auxiliary:
  vision:
    provider: openrouter
    model: google/gemini-2.5-flash
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
    download_timeout: 30
```

**補助を auto に（既定）:**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... other fields unchanged
```

`provider: auto` と `model: ''` の組み合わせは、そのタスクにメインモデルを使うよう Hermes に指示します。メインの経路で補助呼び出しを処理できない場合も、フォールバックポリシーは引き続き尊重されます。

タスク固有のフォールバックチェーンは、同じ補助タスクの下に任意で設定できます。

```yaml
auxiliary:
  title_generation:
    provider: auto
    model: ''
    fallback_chain:
      - provider: openrouter
        model: inclusionai/ring-2.6-1t:free
```

`fallback_chain` がない場合、`auto` はトップレベルの `fallback_providers` チェーンを使用します。それもなく、メインプロバイダーが呼び出しを処理できない場合、そのタスクは警告とともにスキップされます。Hermes がログイン済みの他のプロバイダーにフォールスルーすることはありません。

## プロバイダーごとのリクエストオプション

プロバイダーのエントリ（`providers:` 辞書内の `providers.<name>`、または従来の `custom_providers` リストの項目）は、Hermes がエンドポイントとやり取りする方法を調整するオプションを受け付けます。

**`extra_headers`** — そのプロバイダーのベース URL にルーティングされるすべての LLM リクエストに付加する追加の HTTP ヘッダーのマッピングです。URL／プロファイルの既定値やユーザーによるヘッダーの上書きの後、最後に適用されるため、クレデンシャルの入れ替えやクライアントの再構築後も保持されます。Cloudflare Access のサービストークン、プロキシ認証、独自の bearer スキームなどに便利です。

```yaml
providers:
  my-gateway:
    api: https://llm.internal.example.com/v1
    api_key: sk-...
    extra_headers:
      CF-Access-Client-Id: "xxxx.access"
      CF-Access-Client-Secret: "yyyy"
```

ヘッダーの値にはクレデンシャルが含まれることが多いため、Hermes がそれらをログに記録することはありません。`extra_headers` は、OpenAI 互換の経路と `anthropic_messages` の経路（メインクライアント、`/model` による切り替え、再構築、補助クライアントのいずれも）に適用されます。`bedrock_converse` では使用されません。SDK の既定の `User-Agent` を拒否する WAF の背後にあるリレー（403「Your request was blocked」やブラウザチャレンジのページ）が、これを設定する典型的な理由です。Hermes はそのような 403 を、API キーの拒否ではなくファイアウォール／CDN によるブロックとして報告します。

**`session_affinity_header`** — そのプロバイダーへのすべてのリクエスト（`chat_completions`、`anthropic_messages`、`codex_responses` でのメインのターンに加え、圧縮やタイトルなどの補助呼び出し）で Hermes の会話 ID を運ぶヘッダーの「名前」です。設定しない限りオフで、Hermes は要求していないエンドポイントにセッション識別子を送信することはありません。ステートフルなバックエンドの前段にあるセッション対応プロキシ（LiteLLM の `x-litellm-session-id`、セルフホストの Claude/OpenAI ゲートウェイ）は、これがないとエージェントループを関連付ける手がかりがなく、ほぼすべてのリクエストを新しい会話として扱い、ターンごとに履歴全体を上流へ再送信してしまいます。値は不透明で、1 つの会話のターン間（コンパクション後も含む）で安定しており、会話ごとに異なります。

```yaml
providers:
  my-proxy:
    api: http://127.0.0.1:4000/v1
    api_key: sk-...
    session_affinity_header: x-litellm-session-id
```

**`discover_models`** — `false` に設定すると（既定は `true`）、エンドポイントの `/models` 一覧の問い合わせをスキップし、エントリに設定した `models` のみを使用します。モデル一覧が遅い、不安定、またはノイズが多いゲートウェイで便利です。

```yaml
providers:
  my-gateway:
    api: https://llm.internal.example.com/v1
    discover_models: false
    models:
      - my-finetune-v2
      - my-finetune-v1
```

検出をオフにすると、モデルピッカー（`hermes model`、`/model`）はライブの問い合わせではなく、設定済みのリストを表示します。

**`openai_native_compaction`** — この機能は、会話の内容を任せられる信頼できる OpenAI 互換エンドポイントに対してのみ `true` に設定してください。ネイティブのコンパクションは、そのプロバイダーに設定された `base_url` にペイロードを送信します。

```yaml
providers:
  trusted-proxy:
    api: https://llm.internal.example.com/v1
    capabilities:
      openai_native_compaction: true
```

リクエストを受け取ってから初めて素のモデルエイリアスを解決するゲートウェイでは、モデルごとの `prompt_caching` 機能を使って、そのエイリアスでプロンプトキャッシュのマーカーを有効にします。

```yaml
providers:
  model-proxy:
    api: https://gateway.example.com/v1
    transport: openai_chat  # or anthropic_messages
    models:
      fable:
        context_length: 1000000
        prompt_caching: true
```

Hermes はこの宣言を、正確なプロバイダーの経路と実行時のモデル ID に照合します。エイリアスを書き換えたり、プロバイダー名、ホスト、モデルファミリーから対応状況を推測したりすることはありません。マーカーの配置は設定されたトランスポートに従います。`openai_chat` は OpenAI 互換のエンベロープ配置を使用し、`anthropic_messages` はネイティブの内側ブロック配置を使用します。`prompt_caching: false` を設定すると、そのモデルのキャッシュマーカーを明示的に無効化できます。省略した場合、Hermes は通常のプロバイダーとモデルの機能検出を維持します。

一部の OpenAI 互換サーバーは、`content` を空にして、回答全体を `reasoning` フィールドで返します。ほとんどのプロバイダーはそこに非公開の思考の連鎖を入れるため、Hermes は既定では推論を応答として表示せず、代わりにリトライします。サーバーのパーサーが本当に最終的な回答を `reasoning` に入れる場合は、`answer_in_reasoning` でそのモデルを有効にします。

```yaml
providers:
  local-vllm:
    api: http://127.0.0.1:8000/v1
    models:
      my-reasoner:
        answer_in_reasoning: true
```

プロバイダー上のすべてのモデルで有効にするには、このフラグをプロバイダーレベルの `capabilities:` ブロック（`capabilities: {answer_in_reasoning: true}`）に記述します。Hermes は CLI、TUI、ゲートウェイのいずれでも、毎ターンアクティブな経路上の有効化設定を読み直すため、同じプロバイダー上の別のモデルへの `/model` による切り替え後も保持されます。フォールバックやその他のプロバイダーには引き継がれず、OpenRouter の経路はフラグが設定されていても信頼されません。

:::note 従来の形式
古い設定では、トップレベルの `custom_providers:` リスト（`api` の代わりに `base_url` を使用）が使われていました。これは引き続き動作し、`hermes update`（config v12）で `providers:` 辞書に自動移行されます。
:::

### Nous Portal: Claude をどのワイヤーで運ぶか

Nous Portal は、`anthropic/*` モデルを 2 つの経路で提供しています。OpenAI 互換の `/v1/chat/completions` と、ネイティブな Anthropic Messages ワイヤーの `/v1/messages` です。`nous.anthropic_wire` でどちらかを選択します。

```yaml
nous:
  anthropic_wire: chat     # default. "native" = the Anthropic Messages wire; "auto" = decide per session
```

現時点では `chat` が既定です。ネイティブワイヤーのほうが優れたトランスポートです（署名付きの thinking ブロックがそのまま通過し、ネイティブの `cache_control` スコープが使えます）が、Portal の OpenRouter 経由の経路では、並行するツールループの連続呼び出しのうち 14〜20% で直前のターンのプロンプトキャッシュを書き直しており、これはファンアウトのキャッシュ書き込み料金の 15〜20% に相当します。同じテストで chat の経路は 0 でした。再び有効にするには `native` を設定してください（たとえば Portal 側の修正がリリースされた後など）。影響を受けるのは `anthropic/*` モデルのみで、Nous 上のその他のモデルはすでに chat/completions を使用しています。

`auto` は、Portal が同じモデルを複数の上流から提供している場合のための設定です。セッションは chat で開始し、Hermes は最初の呼び出しにどの上流が応答したかを読み取り、その上流がネイティブで問題がないと確認されているものである場合にのみ、そのセッションをネイティブに切り替えます（切り替えは呼び出しの間に行われるため、処理中の応答や温まったキャッシュが失われることはありません）。現在、確認済みの上流はないため、`auto` は `chat` とまったく同じように動作します。これは、設定変更ではなく測定結果に基づいて切り替えを行えるようにするために存在します。

## いつ反映されるか

- **CLI**（`hermes chat`）: 次回の `hermes chat` の実行時。
- **Gateway**（Telegram、Discord、Slack など）: 次の*新規*セッション。既存のセッションはモデルを維持します。すべてのセッションに変更を強制的に反映させたい場合は、ゲートウェイを再起動してください（`hermes gateway restart`）。
- **ダッシュボードのチャットタブ**（`/chat`）: 次の新しい PTY。現在開いているチャットはモデルを維持します。ホットスワップするには、その中で `/model` を使用してください。

変更が実行中のセッションのプロンプトキャッシュを無効化することはありません。これは意図的なものです。セッション内でメインモデルを入れ替えるにはキャッシュのリセットが必要であり（システムプロンプトにはモデル固有の内容が含まれます）、これはチャット内の明示的な `/model` スラッシュコマンド専用としています。

## トラブルシューティング

### ピッカーに「No authenticated providers」と表示される

Hermes は、有効なクレデンシャルを持つプロバイダーのみを一覧表示します。サイドバーの **Keys** を確認してください。API キー、成功した OAuth、カスタムエンドポイント URL のいずれかが表示されているはずです。目的のプロバイダーがそこにない場合は、`hermes setup` を実行して設定するか、**Keys** に移動して環境変数を追加してください。

### 実行中のチャットでメインモデルが変わらない

想定どおりの動作です。ダッシュボードは `config.yaml` を書き込み、これは新しいセッションが読み込みます。現在開いているチャットは稼働中のエージェントプロセスであり、起動時のモデルをそのまま使い続けます。その特定のセッションをホットスワップするには、チャット内で `/model <name>` を使用してください。

### 補助の上書きが「反映されない」

確認すべき 3 点は次のとおりです。

1. **新しいセッションを開始しましたか？** 既存のチャットは設定を再読み込みしません。
2. **`provider` が `auto` 以外に設定されていますか？** フィールドが `auto` を表示している場合、そのタスクは引き続きメインモデルを使用しています。**Change** をクリックして実際のプロバイダーを選んでください。
3. **そのプロバイダーは認証されていますか？** あるタスクに `minimax` を割り当てたのに MiniMax の API キーがない場合、そのタスクは openrouter の既定にフォールバックし、`agent.log` に警告を記録します。

### モデルを選んだのに Hermes がプロバイダーを勝手に切り替えた

OpenRouter（または任意のアグリゲーター）では、修飾なしのモデル名はまずアグリゲーター*内*で解決されます。そのため OpenRouter 上の `claude-sonnet-4` は `anthropic/claude-sonnet-4.6` になり、OpenRouter の認証に留まります。しかし、ネイティブな Anthropic 認証で `claude-sonnet-4` と入力した場合は `claude-sonnet-4-6` のままになります。予期しないプロバイダーの切り替えが見られた場合は、現在のプロバイダーが想定どおりかを確認してください。ピッカーは常にダイアログ上部に現在のメインを表示します。

## 代替手段 {#alternative-methods}

### CLI スラッシュコマンド

任意の `hermes chat` セッション内で実行します。

```
/model gpt-5.4 --provider openrouter             # session-only
/model gpt-5.4 --provider openrouter --global    # also persists to config.yaml
/model claude-opus-4.6 --once                    # next turn only, then auto-restores
```

`--global` は、ダッシュボードの **Change** ボタンと同じことを行い、加えて実行中のセッションをその場で切り替えます。

`--once` は 1 ターンだけ切り替え、その後 — 成功、エラー、中断のいずれの場合でも — 元のモデルに戻します。何も永続化されないため、ターンの途中でゲートウェイを再起動すると元のモデルで復帰します。難しい質問 1 つだけを高価なモデルにエスカレーションしたい場合（「今回だけ Opus に聞く」）や、使い捨てのクエリのために安価なモデルに落としたい場合に便利です。

:::note プロンプトキャッシュのコスト
1 ターンだけの切り替えは、プロバイダーのプロンプトキャッシュのプレフィックスを 2 回（切り替え時と戻す時）壊します。キャッシュ済みプレフィックスを持つプロバイダー（Anthropic、OpenAI）での長いセッションでは、次のターンで入力コストを全額支払い直すことになります。`--once` は短いセッションや安価→高価へのエスカレーションには有利ですが、長く高価なセッション内でのちょっとした脇道の質問では、節約分よりもコストが上回ることがあります。
:::

### カスタムエイリアス

よく使うモデルに自分用の短い名前を定義しておき、実行中のセッションでは `/model <alias>`、起動時には `hermes chat --model <alias>` を使用します。同等の形式が 2 つあるので、ワークフローに合うほうを選んでください。

**正式な形式（トップレベルの `model_aliases:`）** — provider と base_url を完全に制御できます。

```yaml
# ~/.hermes/config.yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
```

独自のエンドポイントを指すエイリアスは、そのエンドポイントのクレデンシャルを `api_key`（リテラル、または `"${VAR}"` 参照）または `key_env`（環境変数の名前）で持たせることもできます。両方が設定されている場合は `api_key` が優先されます。

```yaml
model_aliases:
  theta:
    model: theta-1
    provider: custom
    base_url: "https://theta.example.com/v1"
    key_env: THETA_API_KEY        # or: api_key: "${THETA_API_KEY}"
```

どちらも設定されていない場合、キーはエイリアスの**ホスト**から解決されます。`ollama.com` のエンドポイントなら `OLLAMA_API_KEY`、`api.deepseek.com` なら `DEEPSEEK_API_KEY` といった具合です。切り替え前にたまたまアクティブだったプロバイダーから引き継がれることはないため、エイリアスに切り替えたことで、あるプロバイダーのシークレットが別のプロバイダーのホストに送信されることはありません。

**短い文字列形式（`model.aliases.<name>: provider/model`）** — `hermes config set` はスカラーを書き込み、現在はインラインのリスト／マッピングのリテラルも解析できるため、シェルから使うのに便利です。ただし、この短いエイリアス形式では独自の `base_url` を持たせることはできません。

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

> `hermes config set` は、インラインの**リスト／マッピングのリテラル**（JSON/YAML のフロースタイル）も受け付けます。シェルがそのまま渡すよう、クォートで囲んでください。
>
> ```bash
> hermes config set platform_toolsets.line '["clarify", "file", "web"]'
> hermes config set display.tool_progress_overrides '{"terminal": "off"}'
> ```

どちらの方法も同じローダー（`hermes_cli/model_switch.py`）に渡されます。`model_aliases:` で宣言したエントリは、同じ名前の `model.aliases:` のエントリより優先されます。

その後、チャットで `/model fav` または `/model grok` と入力します。ユーザーエイリアスは組み込みの短縮名（`sonnet`、`kimi`、`opus` など）を上書きします。完全なリファレンスは[カスタムモデルエイリアス](../reference/slash-commands.md#custom-model-aliases)を参照してください。

### `hermes model` サブコマンド

```bash
hermes model            # Interactive provider + model picker (the canonical way to switch defaults)
```

`hermes model` は、プロバイダーの選択、認証（OAuth フローはブラウザを開き、API キー方式のプロバイダーはキーの入力を求めます）、そしてそのプロバイダーの厳選カタログから特定のモデルを選ぶまでを案内します。選択内容は `~/.hermes/config.yaml` の `model.provider` と `model.default` に書き込まれます。新しいモデルを保存すると、続いて推論エフォートのステップ（`minimal` … `ultra`、**Disable reasoning**、または現在の値を維持する **Skip**）が表示され、`agent.reasoning_effort` に書き込まれます。カタログで推論の制御がないとされているモデルでは、このステップはスキップされます。プロバイダー一覧には、エフォートだけを変更するための **Reasoning effort for the current model...** 行もあります。

**Configure auxiliary models...** は、タスクごとのサイドモデルピッカー（vision、compression、approval、delegation など）を開きます。各タスクのプロバイダー → モデルの選択の最後に同じエフォートのステップがあり、`auxiliary.<task>.reasoning_effort`（または `delegation.reasoning_effort`）に保存されます。さらに、レベルをプロバイダーに任せる **Provider default** 行も追加されます。ブロックに設計上 `reasoning_effort` キーを持たないタスク（MoA スロット、メモリクエリの書き換え）では、このステップはスキップされます。

ピッカーを起動せずにプロバイダー／モデルを一覧表示するには、ダッシュボードか以下の REST エンドポイントを使用してください。CLI が今まさに使用する内容を確認するには `hermes config get model --json` と `hermes status` を実行します。

### 設定の直接編集

`~/.hermes/config.yaml` を編集し、それを読み込むものを再起動します。完全なスキーマは[設定リファレンス](./configuration.md)を参照してください。

### REST API

ダッシュボードは 3 つのエンドポイントを使用します。スクリプト化に便利です。

```bash
# List authenticated providers + curated model lists
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# Read current main + auxiliary assignments
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# Set the main model
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# Override a single auxiliary task
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Assign one model to every auxiliary task
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Reset all auxiliary tasks to auto
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

セッショントークンは起動時にダッシュボードの HTML に注入され、サーバーを再起動するたびにローテーションされます。実行中のダッシュボードに対してスクリプトを実行する場合は、ブラウザの devtools（`window.__HERMES_SESSION_TOKEN__`）から取得してください。
