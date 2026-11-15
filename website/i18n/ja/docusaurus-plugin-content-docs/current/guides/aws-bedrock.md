---
sidebar_position: 14
title: "AWS Bedrock"
description: "Hermes Agent を Amazon Bedrock で利用 — ネイティブ Converse API、Anthropic SDK によるルーティング、Bedrock Mantle 経由の OpenAI モデル、IAM 認証、Guardrails、クロスリージョン推論"
---

# AWS Bedrock

このページの Python 依存関係コマンドは、[PM で準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を使用します。
依存関係を変更した後は、チェックアウトを再アクティブ化して Hermes を再起動してください。

Hermes Agent は Amazon Bedrock をネイティブプロバイダーとしてサポートします。これにより、Bedrock エコシステムへのフルアクセスが得られます: IAM 認証、Guardrails、クロスリージョン推論プロファイル、そしてすべての基盤モデルです。

Hermes は、各モデルファミリーを最適な API 経由でルーティングします:

| モデルファミリー | API ルート | 理由 |
|---|---|---|
| Anthropic Claude | Anthropic SDK（`AnthropicBedrock`） | プロンプトキャッシュ、thinking バジェット、adaptive thinking — Converse では公開されていない機能 |
| 素の `openai.*` ID の OpenAI GPT-6（Astra、Sol、Luna）、GPT-6.1（Sol、Terra）、GPT-5.6（Sol、Terra、Luna）、GPT-5.5、GPT-5.4 | Bedrock Mantle の **OpenAI Responses** エンドポイント（`bedrock-mantle.<region>.api.aws/openai/v1`） | Bedrock Mantle は素のリージョン内 ID を提供します。`bedrock-runtime` はそれを受け付けません |
| `us.` または `global.` の推論プロファイル ID を持つ同じ OpenAI モデル | ネイティブ **Converse API**（`bedrock-runtime`） | クロスリージョンプロファイルは `bedrock-runtime` にのみ存在し、モデルカードでもそこでは Converse がサポートされていると記載されています |
| その他すべて（Nova、DeepSeek、Llama、GPT-OSS など） | ネイティブ **Converse API**（`bedrock-runtime`） | Bedrock の全機能: Guardrails、推論プロファイル、ストリーミング |

3 つのルートはすべて同じ AWS 資格情報チェーンとリージョン解決を共有するため、個別の設定は不要です。Mantle エンドポイントへのリクエストは、`AWS_BEARER_TOKEN_BEDROCK` が設定されていればそれで認証され、そうでなければ標準の boto3 資格情報チェーンによって SigV4 署名されます。

## 前提条件

- **AWS 資格情報** — [boto3 の資格情報チェーン](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)がサポートする任意のソース:
  - IAM インスタンスロール（EC2、ECS、Lambda — 設定不要）
  - `AWS_ACCESS_KEY_ID` ＋ `AWS_SECRET_ACCESS_KEY` 環境変数
  - SSO または名前付きプロファイル用の `AWS_PROFILE`
  - ローカル開発用の `aws configure`
- **boto3** — `cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['bedrock'], explicit=True)"` でインストール
- **IAM 権限** — 最低限:
  - `bedrock:InvokeModel` と `bedrock:InvokeModelWithResponseStream`（推論用）
  - `bedrock:ListFoundationModels` と `bedrock:ListInferenceProfiles`（モデル探索用）
  - `bedrock:GetInferenceProfile`（`model.default` がアプリケーション推論プロファイルの ARN である場合のみ — ラップされたモデルからコンテキストウィンドウのサイズを決めるために使用）

:::tip EC2 / ECS / Lambda
AWS のコンピュート上では、`AmazonBedrockFullAccess` を持つ IAM ロールをアタッチすれば完了です。API キーも `.env` 設定も不要 — Hermes がインスタンスロールを自動的に検出します。
:::

## クイックスタート

```bash
# Install with Bedrock support
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['bedrock'], explicit=True)"

# Select Bedrock as your provider
hermes model
# → Choose "More providers..." → "AWS Bedrock"
# → Select your region and model

# Start chatting
hermes chat
```

## 設定

`hermes model` を実行すると、`~/.hermes/config.yaml` に次が含まれます:

```yaml
model:
  default: us.anthropic.claude-sonnet-4-6
  provider: bedrock
  base_url: https://bedrock-runtime.us-east-2.amazonaws.com

bedrock:
  region: us-east-2
```

### リージョン

AWS リージョンは次のいずれかの方法で設定します（優先度の高い順）:

1. `config.yaml` の `bedrock.region`
2. `AWS_REGION` 環境変数
3. `AWS_DEFAULT_REGION` 環境変数
4. デフォルト: `us-east-1`

### Mantle での Ultrafast

**`us-east-1` の Mantle 上の GPT-6.1 Sol** では、CLI とメッセージングゲートウェイのターンビルダーが既存の `agent.service_tier: ultrafast` 設定をサポートします:

```yaml
model:
  provider: bedrock
  default: openai.gpt-6.1-sol
bedrock:
  region: us-east-1
agent:
  service_tier: ultrafast
```

これにより、`https://bedrock-mantle.us-east-1.api.aws/openai/v1` にある既存の Responses ルートを通じて `service_tier: "ultrafast"` が送信されます。CLI は、この解決済みルートで `/fast ultrafast` も受け付けます。デフォルトのティアにするには `agent.service_tier: normal` を設定してください（または省略してください）。既存のベアラートークンと IAM 署名のパスは変わりません。Ultrafast は有料のオプトインであり、単なる `/fast fast` の意味ではありません。

- **正確なスコープ:** 有効になるのは、この Mantle エンドポイント上の `openai.gpt-6.1-sol` のみです。Priority、Flex、Reserved、および auto/cold の fast ウィンドウは有効になりません。
- **ランタイムプロファイル:** AWS は `us.openai.gpt-6.1-sol` と `global.openai.gpt-6.1-sol` についても Responses 経由の Ultrafast を記載しています。Hermes はこれらの ID を引き続き Converse 経由でルーティングします。この設定はそれらの Ultrafast パスを有効にせず、API を切り替えず、Ultrafast を Converse の `serviceTier` に変換することもありません。
- **その他のエントリーポイント:** これによって、速度ピッカーやゲートウェイの `/fast` コマンドに Bedrock Ultrafast が追加されるわけではありません。`request_overrides` を構築せずに `service_tier` のみを設定するコンストラクタは、既存の制限をそのまま保持します。
- **サービスの挙動:** リクエストしたティアは、実際に提供されたティアやアカウントのアクセス権の証明にはなりません。プロバイダーのレスポンスを確認してください。この統合の検証に AWS のライブ推論は使用していません。

料金と提供状況については、[AWS Sol のサービスティアのドキュメント](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-1-sol.html#model-card-openai-gpt-6-1-sol-tiers)を参照してください。Mantle Responses には Guardrails は適用されません。ティアをリクエストするためだけに、ガードレールに依存するワークロードを Converse から切り替えないでください。

### Guardrails

すべてのモデル呼び出しに [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) を適用するには:

```yaml
bedrock:
  region: us-east-2
  guardrail:
    guardrail_identifier: "abc123def456"  # From the Bedrock console
    guardrail_version: "1"                # Version number or "DRAFT"
    stream_processing_mode: "async"       # "sync" or "async"
    trace: "disabled"                     # "enabled", "disabled", or "enabled_full"
```

ガードレールは Converse ルート（`guardrailConfig`）と Claude ルート（Anthropic Bedrock SDK 経由の InvokeModel ヘッダー。そのためプロンプトキャッシュと thinking は維持されます）にアタッチされます。ブロックされたリクエストは、モデルのテキストとしてではなく、コンテンツフィルターによる拒否として表面化します。`stream_processing_mode` は Converse にのみ適用されます。

AWS は、素の `openai.gpt-*` ID が使用する Mantle Responses エンドポイントには Guardrails を適用しません（[AWS ドキュメント](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html)）。ガードレールが必要な場合は、`us.` または `global.` の OpenAI 推論プロファイルなど、Converse で提供されるモデルを使用してください。

### モデル探索

Hermes は Bedrock コントロールプレーン経由で利用可能なモデルを自動探索します。探索をカスタマイズできます:

```yaml
bedrock:
  discovery:
    enabled: true
    provider_filter: ["anthropic", "amazon"]  # Only show these providers
    refresh_interval: 3600                     # Cache for 1 hour
```

### プロンプトキャッシュ（cachePoint）

Hermes は、Bedrock **Converse API** パスで、システムプロンプト、ツール定義、最新のメッセージの後に `cachePoint` マーカーを挿入して、プロンプトキャッシュを自動的に適用します。`cachePoint` ブロックをサポートしていないモデルに送ると `ValidationException` が発生するため、マーカーは動作が確認済みの許可リスト（Anthropic Claude と Amazon Nova のモデル ID）にあるモデルにのみ追加されます。不明なモデルではデフォルトでキャッシュマーカーは付きません。Claude モデルは通常、独自のプロンプトキャッシュを持つ AnthropicBedrock SDK パスを使用します。Converse の `cachePoint` パスは Nova と、ベアラートークンによる Claude のフォールバックをカバーします。設定は不要で、キャッシュの読み取り/書き込みは使用量の集計に表示されます。

### コンテキストウィンドウのプローブ

Hermes の静的テーブルにコンテキストウィンドウがないモデルについては、固定のティア（約 1.3M と約 2.2M トークン）で過大なリクエストを送信し、Bedrock の長さ検証エラーで報告される `maximum` を解析することで、実際の上限をプローブできます。プローブされた値は静的テーブルと同じメタデータキャッシュに入ります。モデルのウィンドウを過小報告している古いキャッシュエントリ（例: モデルの 1M ウィンドウが GA になる前に作られたエントリ）は、より大きな既知の値を優先して自動的に破棄されます。

静的テーブルに Anthropic、OpenAI、xAI のモデルの行がない場合、Hermes は一般的なモデルカタログ（`agent/model_metadata.py`）にあるそのモデルファミリーのウィンドウを使用します。これにより、Bedrock のテーブルが更新される前でも、新しいティアが実際のウィンドウを得られます。その他のベンダーのカタログの行はベンダー直接のウィンドウであり、Bedrock が提供するものを超える場合があるため（Llama 3 8B は Bedrock では 8K）、それらのモデルはカタログの値を借用しません。

どこにもモデルの情報がない場合、Hermes は 128,000 トークンのフォールバックを使用し、モデル名を示す WARNING をログに記録します。バナー、ステータスバー、`/usage`、`/model`、ゲートウェイのセッション行では、そのウィンドウに `fallback` と表示され、検出された上限と誤解されないようになっています。`config.yaml` の `model.context_length`（または `model_overrides.bedrock.<model id>.context_window`）にモデルの実際のウィンドウを設定してください。

**アプリケーション推論プロファイル。** `arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abcdef123456` のような ARN はモデル名を含まないため、プローブでも静的テーブルでもサイズを決められません。Hermes は ARN のリージョンで `bedrock:GetInferenceProfile` を呼び出し、プロファイルがラップしているモデルからウィンドウのサイズを決めます（Claude Sonnet 4.6 をラップするプロファイルなら 1M）。その権限がない場合は 128,000 トークンのデフォルトが適用され、プロファイル名を示す WARNING が出ます。いずれの場合も、`model.context_length` を明示的に設定すれば上書きできます。

## 利用可能なモデル

Bedrock モデルはオンデマンド呼び出しに**推論プロファイル ID** を使います。`hermes model` のピッカーはこれらを自動的に表示し、推奨モデルを上位に出します:

| モデル | ID | 備考 |
|-------|-----|-------|
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | 推奨 — 速度と能力のバランスが最良 |
| Claude Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` | 最も高性能 |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 最速の Claude |
| OpenAI GPT-6 Astra | `openai.gpt-6-astra` | 最も高性能な OpenAI モデル、1,050,000 トークンのウィンドウ |
| OpenAI GPT-6.1 Sol | `openai.gpt-6.1-sol` | 1M トークンのウィンドウ。Mantle のリージョン内アクセスは `us-east-1` のみ |
| OpenAI GPT-6.1 Terra | `openai.gpt-6.1-terra` | バランス型の Terra ティア、1,050,000 トークンのウィンドウ |
| OpenAI GPT-6 Sol | `openai.gpt-6-sol` | コーディングとエージェント的な作業向け、1,050,000 トークンのウィンドウ |
| OpenAI GPT-6 Luna | `openai.gpt-6-luna` | 焦点を絞った大量処理タスク向け、1,050,000 トークンのウィンドウ |
| OpenAI GPT-5.6 Sol | `openai.gpt-5.6-sol` | 1,050,000 トークンのウィンドウ |
| OpenAI GPT-5.6 Terra | `openai.gpt-5.6-terra` | バランス型、1,050,000 トークンのウィンドウ |
| OpenAI GPT-5.6 Luna | `openai.gpt-5.6-luna` | 高速・低価格、1,050,000 トークンのウィンドウ |
| OpenAI GPT-5.5 | `openai.gpt-5.5` | Mantle のみ、推論プロファイルなし |
| OpenAI GPT-5.4 | `openai.gpt-5.4` | 1,050,000 トークンのウィンドウ |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` | Amazon のフラッグシップ |
| Amazon Nova Micro | `us.amazon.nova-micro-v1:0` | 最速・最安 |
| DeepSeek V3.2 | `deepseek.v3.2` | 強力なオープンモデル |
| Llama 4 Scout 17B | `us.meta.llama4-scout-17b-instruct-v1:0` | Meta の最新 |

:::info クロスリージョン推論
`us.` で始まるモデルはクロスリージョン推論プロファイルを使い、より優れたキャパシティと AWS リージョンをまたぐ自動フェイルオーバーを提供します。`global.` で始まるモデルは世界中の利用可能な全リージョンにルーティングします。

OpenAI モデルには 2 つのエンドポイント形式があります。素の `openai.*` ID はリージョン内のもので、Hermes は設定されたリージョンの Bedrock Mantle の Responses エンドポイントに送信します。`us.openai.*` と `global.openai.*` の ID はクロスリージョン推論プロファイルです。Hermes はこれらを、他の推論プロファイルと同じパスである Converse API 経由で `bedrock-runtime` に送信します。各 AWS モデルカードには、モデルが提供するプロファイルと、Mantle がそのモデルを提供するリージョンが記載されています。
:::

## セッション途中でのモデル切り替え

会話中に `/model` コマンドを使います:

```
/model us.amazon.nova-pro-v1:0
/model deepseek.v3.2
/model us.anthropic.claude-opus-4-6-v1
```

## 診断

```bash
hermes doctor
```

doctor は次を確認します:
- AWS 資格情報が利用可能か（env 変数、IAM ロール、SSO）
- `boto3` がインストールされているか
- Bedrock API に到達可能か（ListFoundationModels）
- リージョンで利用可能なモデルの数

## ゲートウェイ（メッセージングプラットフォーム）

Bedrock はすべての Hermes ゲートウェイプラットフォーム（Telegram、Discord、Slack、Feishu など）で動作します。Bedrock をプロバイダーとして設定し、通常どおりゲートウェイを起動します:

```bash
hermes gateway setup
hermes gateway start
```

ゲートウェイは `config.yaml` を読み込み、同じ Bedrock プロバイダー設定を使用します。

## トラブルシューティング

### 「No API key found」/「No AWS credentials」

Hermes は次の順序で資格情報を確認します:
1. `AWS_BEARER_TOKEN_BEDROCK`
2. `AWS_ACCESS_KEY_ID` ＋ `AWS_SECRET_ACCESS_KEY`
3. `AWS_PROFILE`
4. EC2 インスタンスメタデータ（IMDS）
5. ECS コンテナ資格情報
6. Lambda 実行ロール

どれも見つからない場合は、`aws configure` を実行するか、コンピュートインスタンスに IAM ロールをアタッチしてください。

### 「Invocation of model ID ... with on-demand throughput isn't supported」

素の基盤モデル ID の代わりに、**推論プロファイル ID**（`us.` または `global.` で始まる）を使ってください。例:
- ❌ `anthropic.claude-sonnet-4-6`
- ✅ `us.anthropic.claude-sonnet-4-6`

### 「ThrottlingException」

Bedrock のモデルごとのレート制限に達しました。Hermes はバックオフ付きで自動的にリトライします。制限を引き上げるには、[AWS Service Quotas コンソール](https://console.aws.amazon.com/servicequotas/)でクォータの引き上げをリクエストしてください。

## ワンクリック AWS デプロイ

CloudFormation を使った EC2 上の完全自動デプロイには:

**[sample-hermes-agent-on-aws-with-bedrock](https://github.com/JiaDe-Wu/sample-hermes-agent-on-aws-with-bedrock)** — VPC、IAM ロール、EC2 インスタンスを作成し、Bedrock を自動的に設定します。任意のリージョンにワンクリックでデプロイできます。
