---
title: プロバイダールーティング
description: OpenRouter のプロバイダー設定を構成し、コスト・速度・品質を最適化します。
sidebar_label: プロバイダールーティング
sidebar_position: 7
---

# プロバイダールーティング

[OpenRouter](https://openrouter.ai) を LLM プロバイダーとして使用する場合、Hermes Agent は**プロバイダールーティング**をサポートします — どの基盤 AI プロバイダーがリクエストを処理し、それらをどのように優先付けするかをきめ細かく制御できます。

OpenRouter は多数のプロバイダー（例: Anthropic、Google、AWS Bedrock、Together AI）にリクエストをルーティングします。プロバイダールーティングを使うと、コスト・速度・品質を最適化したり、特定のプロバイダー要件を強制したりできます。

## 設定

`~/.hermes/config.yaml` に `provider_routing` セクションを追加します:

```yaml
provider_routing:
  sort: "price"           # プロバイダーのランク付け方法
  only: []                # ホワイトリスト: これらのプロバイダーのみ使用
  ignore: []              # ブラックリスト: これらのプロバイダーは決して使用しない
  order: []               # 明示的なプロバイダー優先順位
  require_parameters: false  # すべてのパラメータをサポートするプロバイダーのみ使用
  data_collection: null   # データ収集の制御（"allow" または "deny"）
```

:::info
プロバイダールーティングは OpenRouter を使用する場合にのみ適用されます。直接のプロバイダー接続（例: Anthropic API への直接接続）では効果がありません。
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

プロバイダー名のホワイトリスト。設定すると、**これらのプロバイダーのみ**が使用されます。それ以外はすべて除外されます。

```yaml
provider_routing:
  only:
    - "Anthropic"
    - "Google"
```

### `ignore`

プロバイダー名のブラックリスト。これらのプロバイダーは、最も安価または最速の選択肢であっても**決して**使用されません。

```yaml
provider_routing:
  ignore:
    - "Together"
    - "DeepInfra"
```

### `order`

明示的な優先順位。先頭に並んだプロバイダーが優先されます。記載されていないプロバイダーはフォールバックとして使用されます。

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
    - "AWS Bedrock"
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
    - "Anthropic"
```

### 特定プロバイダーの回避

使用したくないプロバイダーを除外します（例: データプライバシーのため）:

```yaml
provider_routing:
  ignore:
    - "Together"
    - "Lepton"
  data_collection: "deny"
```

### フォールバック付きの優先順位

優先するプロバイダーを最初に試し、利用できない場合は他にフォールバックします:

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
  require_parameters: true
```

## 仕組み

プロバイダールーティングの設定は、すべての API 呼び出しで `extra_body.provider` フィールドを介して OpenRouter API に渡されます。これは次の両方に適用されます:

- **CLI モード** — `~/.hermes/config.yaml` で設定され、起動時に読み込まれます
- **ゲートウェイモード** — 同じ設定ファイルで、ゲートウェイ起動時に読み込まれます

ルーティング設定は `config.yaml` から読み込まれ、`AIAgent` の作成時にパラメータとして渡されます:

```
providers_allowed  ← provider_routing.only から
providers_ignored  ← provider_routing.ignore から
providers_order    ← provider_routing.order から
provider_sort      ← provider_routing.sort から
provider_require_parameters ← provider_routing.require_parameters から
provider_data_collection    ← provider_routing.data_collection から
```

:::tip
複数のオプションを組み合わせられます。例えば、価格でソートしつつ、特定のプロバイダーを除外し、パラメータサポートを要求できます:

```yaml
provider_routing:
  sort: "price"
  ignore: ["Together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## デフォルトの動作

`provider_routing` セクションが設定されていない場合（デフォルト）、OpenRouter は独自のデフォルトルーティングロジックを使用し、一般にコストと可用性を自動的にバランスします。

:::tip プロバイダールーティング vs. フォールバックモデル
プロバイダールーティングは、**OpenRouter 内のどのサブプロバイダー**がリクエストを処理するかを制御します。プライマリモデルが失敗したときに、まったく別のプロバイダーへ自動フェイルオーバーする方法については、[フォールバックプロバイダー](/docs/user-guide/features/fallback-providers)を参照してください。
:::
