---
sidebar_position: 3
---

# モデルの設定

Hermes は 2 種類のモデルスロットを使用します。

- **メインモデル** — エージェントが思考に用いるモデルです。すべてのユーザーメッセージ、すべてのツール呼び出しループ、すべてのストリーミング応答がこのモデルを通ります。
- **補助モデル** — エージェントが切り出す、より小さなサイドジョブ用のモデルです。コンテキスト圧縮、ビジョン（画像分析）、Web ページの要約、セッション検索、承認スコアリング、MCP ツールルーティング、セッションタイトル生成、スキル検索などがあります。それぞれが独自のスロットを持ち、個別に上書きできます。

このページでは、両方をダッシュボードから設定する方法を説明します。設定ファイルや CLI を使いたい場合は、ページ下部の[代替手段](#alternative-methods)に移動してください。

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

## 補助モデルの設定

**Show auxiliary** をクリックすると、8 つのタスクスロットが表示されます。

![補助パネルを展開した状態](/img/docs/dashboard-models/auxiliary-expanded.png)

すべての補助タスクは既定で `auto` になっています。これは、そのジョブにもメインモデルを使用することを意味します。サイドジョブにより安価または高速なモデルを使いたい場合に、特定のタスクを上書きしてください。

### よくある上書きパターン

| タスク | 上書きする場面 |
|---|---|
| **Title Gen** | ほぼ常に。$0.10/M の flash モデルでも、Opus と同程度にセッションタイトルを書けます。既定の設定では OpenRouter の `google/gemini-3-flash-preview` に設定されています。 |
| **Vision** | メインモデルがビジョン非対応のコーディングモデル（例: Kimi、DeepSeek）の場合。`google/gemini-2.5-flash` または `gpt-4o-mini` を指定します。 |
| **Compression** | コンテキストを要約するだけのために Opus/M2.7 で推論トークンを浪費している場合。高速なチャットモデルなら 1/50 のコストで同じ仕事をこなします。 |
| **Session Search** | 想起クエリが分散する場合（既定の max_concurrency は 3）。安価なモデルなら請求額を予測可能に保てます。 |
| **Approval** | `approval_mode: smart` 向け。高速／安価なモデル（haiku、flash、gpt-5-mini）が、低リスクなコマンドを自動承認するかどうかを判断します。ここで高価なモデルを使うのは無駄です。 |
| **Web Extract** | `web_extract` を多用する場合。圧縮と同じ理屈で、要約に推論は不要です。 |
| **Skills Hub** | `hermes skills search` がこれを使用します。通常は `auto` で問題ありません。 |
| **MCP** | MCP ツールルーティング。通常は `auto` で問題ありません。 |

### タスクごとの上書き

任意の補助行で **Change** をクリックします。同じピッカーが開き、同じ動作をします。プロバイダーとモデルを選んで Switch を押してください。行が `auto (use main model)` ではなく `provider · model` を表示するように更新されます。

### すべてを auto にリセット

チューニングしすぎて最初からやり直したい場合は、補助セクション上部の **Reset all to auto** をクリックします。すべてのスロットがメインモデルを使用する状態に戻ります。

## 「Use as」ショートカット

ページ上のすべてのモデルカードには **Use as** ドロップダウンがあります。これは最短の経路です。analytics に表示されているモデルを選び、**Use as** をクリックすれば、メインスロットまたは任意の特定の補助タスクにワンクリックで割り当てられます。

![Use as ドロップダウン](/img/docs/dashboard-models/use-as-dropdown.png)

ドロップダウンには次の項目があります。

- **Main model** — メイン行で Change をクリックするのと同じです。
- **All auxiliary tasks** — このモデルを 8 つの補助スロットすべてに一度に割り当てます。すべてのサイドジョブを安価な flash モデルにまとめたいときに便利です。
- **個別のタスクオプション** — Vision、Web Extract、Compression など。各タスクに現在割り当てられているモデルには `current` と表示されます。

カードが現在いずれかに割り当てられている場合は `main` または `aux · <task>` のバッジが付くため、過去に使ったモデルのうちどれがどこに紐付いているかを一目で確認できます。

## `config.yaml` に書き込まれる内容

ダッシュボード経由で保存すると、Hermes は `~/.hermes/config.yaml` に書き込みます。

**メインモデル:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # プロバイダー切り替え時にクリアされる
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
    # ... その他のフィールドは変更なし
```

`provider: auto` と `model: ''` の組み合わせは、そのタスクにメインモデルを使うよう Hermes に指示します。

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
/model gpt-5.4 --provider openrouter             # セッション限定
/model gpt-5.4 --provider openrouter --global    # config.yaml にも永続化
```

`--global` は、ダッシュボードの **Change** ボタンと同じことを行い、加えて実行中のセッションをその場で切り替えます。

### カスタムエイリアス

よく使うモデルに自分用の短い名前を定義しておき、CLI や任意のメッセージングプラットフォームで `/model <alias>` を使用します。

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

あるいはシェルから（短い形式、`provider/model`）:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

その後、チャットで `/model fav` または `/model grok` と入力します。ユーザーエイリアスは組み込みの短縮名（`sonnet`、`kimi`、`opus` など）を上書きします。完全なリファレンスは[カスタムモデルエイリアス](/docs/reference/slash-commands#custom-model-aliases)を参照してください。

### `hermes model` サブコマンド

```bash
hermes model            # インタラクティブなプロバイダー + モデルピッカー（既定を切り替える標準的な方法）
```

`hermes model` は、プロバイダーの選択、認証（OAuth フローはブラウザを開き、API キー方式のプロバイダーはキーの入力を求めます）、そしてそのプロバイダーの厳選カタログから特定のモデルを選ぶまでを案内します。選択内容は `~/.hermes/config.yaml` の `model.provider` と `model.model` に書き込まれます。

ピッカーを起動せずにプロバイダー／モデルを一覧表示するには、ダッシュボードか以下の REST エンドポイントを使用してください。CLI が今まさに使用する内容を確認するには `hermes config get model` と `hermes status` を実行します。

### 設定の直接編集

`~/.hermes/config.yaml` を編集し、それを読み込むものを再起動します。完全なスキーマは[設定リファレンス](./configuration.md)を参照してください。

### REST API

ダッシュボードは 3 つのエンドポイントを使用します。スクリプト化に便利です。

```bash
# 認証済みプロバイダー + 厳選モデルリストを一覧表示
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# 現在のメイン + 補助の割り当てを読み取り
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# メインモデルを設定
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# 単一の補助タスクを上書き
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 1 つのモデルをすべての補助タスクに割り当て
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# すべての補助タスクを auto にリセット
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

セッショントークンは起動時にダッシュボードの HTML に注入され、サーバーを再起動するたびにローテーションされます。実行中のダッシュボードに対してスクリプトを実行する場合は、ブラウザの devtools（`window.__HERMES_SESSION_TOKEN__`）から取得してください。
