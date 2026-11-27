---
title: プロバイダールーティング
description: OpenRouter のプロバイダー設定を構成し、コスト・速度・品質を最適化します。
sidebar_label: プロバイダールーティング
sidebar_position: 7
---

# プロバイダールーティング

[OpenRouter](https://openrouter.ai) を LLM プロバイダーとして使用する場合、Hermes Agent は**プロバイダールーティング**をサポートします — どの基盤 AI プロバイダーがリクエストを処理し、それらをどのように優先付けするかをきめ細かく制御できます。

OpenRouter は多数のプロバイダー（例: Anthropic、Google、AWS Bedrock、Together AI）にリクエストをルーティングします。プロバイダールーティングを使うと、コスト・速度・品質を最適化したり、特定のプロバイダー要件を強制したりできます。2 つのゲートウェイを比較したい場合は、[Nous Portal vs OpenRouter](https://portal.nousresearch.com/compare/openrouter) で料金、無料モデル、プランを確認できます。

:::note
[Nous Portal](/docs/integrations/nous-portal) はモデルごとに中央でルーティングを決定し、呼び出し元が指定したプロバイダー設定を受け付けません。Hermes は Portal に `provider` オブジェクトを送信しないため、そこでは `provider_routing` は単に無視されます。
:::

## 設定

`~/.hermes/config.yaml` に `provider_routing` セクションを追加します:

```yaml
provider_routing:
  sort: "price"           # How to rank providers
  only: []                # Whitelist: only use these providers
  ignore: []              # Blacklist: never use these providers
  order: []               # Explicit provider priority order
  require_parameters: false  # Only use providers that support all parameters
  data_collection: null   # Control data collection ("allow" or "deny")
```

:::info
プロバイダールーティングは OpenRouter を使用する場合にのみ適用されます。Nous Portal や直接のプロバイダー接続（例: Anthropic API への直接接続）では効果がありません。
:::

## オプション

### `sort`

OpenRouter がリクエストに対して利用可能なプロバイダーをどのようにランク付けするかを制御します。

| 値 | 説明 |
|-------|-------------|
| `"price"` | 最も安価なプロバイダーを優先 |
| `"throughput"` | 1 秒あたりのトークン数が最速のものを優先 |
| `"latency"` | 最初のトークンまでの時間が最短のものを優先 |

```yaml
provider_routing:
  sort: "price"
```

### `only`

プロバイダースラッグのホワイトリスト。設定すると、**これらのプロバイダーのみ**が使用されます。それ以外はすべて除外されます。各プロバイダーについて OpenRouter が表示する小文字のスラッグを使用してください。

```yaml
provider_routing:
  only:
    - "anthropic"
    - "google"
```

### `ignore`

プロバイダー名のブラックリスト。これらのプロバイダーは、最も安価または最速の選択肢であっても**決して**使用されません。

```yaml
provider_routing:
  ignore:
    - "together"
    - "deepinfra"
```

### `order`

明示的な優先順位。先頭に並んだプロバイダーが優先されます。記載されていないプロバイダーはフォールバックとして使用されます。

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
    - "amazon-bedrock"
```

### `require_parameters`

`true` の場合、OpenRouter はリクエスト内の**すべての**パラメータ（`temperature`、`top_p`、`tools` など）をサポートするプロバイダーにのみルーティングします。これにより、パラメータが暗黙のうちに無視されるのを防ぎます。

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection`

プロバイダーがあなたのプロンプトを学習に使用できるかどうかを制御します。オプションは `"allow"` または `"deny"` です。

```yaml
provider_routing:
  data_collection: "deny"
```

### モデルごとのオーバーライド（`models`） {#per-model-overrides-models}

モデルごとに異なるプロバイダーセットをピン留めします。`models` 配下のキーはモデル ID で、各エントリは同じ
`sort` / `only` / `ignore` / `order` / `require_parameters` / `data_collection` キーを取り、そのモデルに対してのみ
フラットな値をオーバーライドします。モデルごとに設定しなかった項目は、フラットなデフォルトにフォールスルーします。

```yaml
provider_routing:
  sort: "price"                      # applies to every model
  models:
    "openai/gpt-6-astra":
      only: ["openai"]               # never let a reseller serve this one
    "anthropic/claude-fable-5.1":
      only: ["anthropic"]
    "moonshotai/kimi-k2.6":
      order: ["moonshotai", "together"]
      sort: "throughput"
```

マッチングは `agent.reasoning_overrides` と同様に表記ゆれを許容します（`claude-fable-5.1` / `claude-fable-5-1`、
`openrouter/` プレフィックスの有無を問わない）。オーバーライドはエージェントが *現在* 使用しているモデルに従うため、
`/model` による切り替え、フォールバックの有効化、cron ジョブ、別のモデルで動く委任サブエージェントは、それぞれ独自の
ピン留めを得ます。これらのキーは `config.yaml` を直接編集してください: モデル ID にはドットが含まれ、`hermes config set`
はそれをパス区切りとして読み取ります。

## 実践例

### コストの最適化

最も安価な利用可能プロバイダーにルーティングします。大量利用や開発に適しています:

```yaml
provider_routing:
  sort: "price"
```

### 速度の最適化

インタラクティブな用途のために低レイテンシのプロバイダーを優先します:

```yaml
provider_routing:
  sort: "latency"
```

### スループットの最適化

1 秒あたりのトークン数が重要となる長文生成に最適です:

```yaml
provider_routing:
  sort: "throughput"
```

### 特定プロバイダーへの固定

一貫性のためにすべてのリクエストを特定のプロバイダー経由にします:

```yaml
provider_routing:
  only:
    - "anthropic"
```

### 特定プロバイダーの回避

使用したくないプロバイダーを除外します（例: データプライバシーのため）:

```yaml
provider_routing:
  ignore:
    - "together"
    - "lepton"
  data_collection: "deny"
```

### フォールバック付きの優先順位

優先するプロバイダーを最初に試し、利用できない場合は他にフォールバックします:

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
  require_parameters: true
```

## 仕組み

プロバイダールーティングの設定は、エージェントのチャットリクエストと反復上限時のサマリーにおいて、`extra_body.provider` フィールドを介して OpenRouter に渡されます。（`extra_body` は OpenAI Python SDK の引数で、JSON リクエストではトップレベルの `provider` オブジェクトになります。）圧縮やタイトル生成などの補助タスクは、`auxiliary.<task>.extra_body` で個別に設定します。

- **CLI モード** — `~/.hermes/config.yaml` で設定され、起動時に読み込まれます
- **ゲートウェイモード** — 同じ設定ファイルです。メッセージングプラットフォームのターンは、ゲートウェイ起動時に読み込まれたコピーを使用します。ローカルの `hermes chat` / TUI セッションは、ゲートウェイがセッションを作成する時点で自身のプロファイルから読み込みます: 編集はゲートウェイを再起動せずに次の新しいセッションに適用され、開いているセッションは作成時の値を保持します

ルーティング設定は `config.yaml` から読み込まれ、`AIAgent` の作成時にパラメータとして渡されます:

```
providers_allowed  ← from provider_routing.only
providers_ignored  ← from provider_routing.ignore
providers_order    ← from provider_routing.order
provider_sort      ← from provider_routing.sort
provider_require_parameters ← from provider_routing.require_parameters
provider_data_collection    ← from provider_routing.data_collection
```

:::tip
複数のオプションを組み合わせられます。例えば、価格でソートしつつ、特定のプロバイダーを除外し、パラメータサポートを要求できます:

```yaml
provider_routing:
  sort: "price"
  ignore: ["together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## デフォルトの動作

`provider_routing` セクションが設定されていない場合（デフォルト）、アグリゲーターは独自のデフォルトルーティングロジックを使用し、一般にコストと可用性を自動的にバランスします。

:::tip プロバイダールーティング vs. フォールバックモデル
プロバイダールーティングは、**OpenRouter 内のどのサブプロバイダー**がリクエストを処理するかを制御します。プライマリモデルが失敗したときに、まったく別のプロバイダーへ自動フェイルオーバーする方法については、[フォールバックプロバイダー](./fallback-providers.md)を参照してください。
:::
