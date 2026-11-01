---
sidebar_position: 14
title: "AWS Bedrock"
description: "Hermes Agent を Amazon Bedrock で利用 — ネイティブ Converse API、IAM 認証、Guardrails、クロスリージョン推論"
---

# AWS Bedrock

Hermes Agent は、OpenAI 互換エンドポイントではなく **Converse API** を使って Amazon Bedrock をネイティブプロバイダーとしてサポートします。これにより、Bedrock エコシステムへのフルアクセスが得られます: IAM 認証、Guardrails、クロスリージョン推論プロファイル、そしてすべての基盤モデルです。

## 前提条件

- **AWS 資格情報** — [boto3 の資格情報チェーン](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)がサポートする任意のソース:
  - IAM インスタンスロール（EC2、ECS、Lambda — 設定不要）
  - `AWS_ACCESS_KEY_ID` ＋ `AWS_SECRET_ACCESS_KEY` 環境変数
  - SSO または名前付きプロファイル用の `AWS_PROFILE`
  - ローカル開発用の `aws configure`
- **boto3** — `pip install hermes-agent[bedrock]` でインストール
- **IAM 権限** — 最低限:
  - `bedrock:InvokeModel` と `bedrock:InvokeModelWithResponseStream`（推論用）
  - `bedrock:ListFoundationModels` と `bedrock:ListInferenceProfiles`（モデル探索用）

:::tip EC2 / ECS / Lambda
AWS のコンピュート上では、`AmazonBedrockFullAccess` を持つ IAM ロールをアタッチすれば完了です。API キーも `.env` 設定も不要 — Hermes がインスタンスロールを自動的に検出します。
:::

## クイックスタート

```bash
# Bedrock サポート付きでインストール
pip install hermes-agent[bedrock]

# Bedrock をプロバイダーとして選択
hermes model
# → "More providers..." → "AWS Bedrock" を選択
# → リージョンとモデルを選択

# チャットを開始
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

### Guardrails

すべてのモデル呼び出しに [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) を適用するには:

```yaml
bedrock:
  region: us-east-2
  guardrail:
    guardrail_identifier: "abc123def456"  # Bedrock コンソールから
    guardrail_version: "1"                # バージョン番号または "DRAFT"
    stream_processing_mode: "async"       # "sync" または "async"
    trace: "disabled"                     # "enabled"、"disabled"、または "enabled_full"
```

### モデル探索

Hermes は Bedrock コントロールプレーン経由で利用可能なモデルを自動探索します。探索をカスタマイズできます:

```yaml
bedrock:
  discovery:
    enabled: true
    provider_filter: ["anthropic", "amazon"]  # これらのプロバイダーのみ表示
    refresh_interval: 3600                     # 1 時間キャッシュ
```

## 利用可能なモデル

Bedrock モデルはオンデマンド呼び出しに**推論プロファイル ID** を使います。`hermes model` のピッカーはこれらを自動的に表示し、推奨モデルを上位に出します:

| モデル | ID | 備考 |
|-------|-----|-------|
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | 推奨 — 速度と能力のバランスが最良 |
| Claude Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` | 最も高性能 |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 最速の Claude |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` | Amazon のフラッグシップ |
| Amazon Nova Micro | `us.amazon.nova-micro-v1:0` | 最速・最安 |
| DeepSeek V3.2 | `deepseek.v3.2` | 強力なオープンモデル |
| Llama 4 Scout 17B | `us.meta.llama4-scout-17b-instruct-v1:0` | Meta の最新 |

:::info クロスリージョン推論
`us.` で始まるモデルはクロスリージョン推論プロファイルを使い、より優れたキャパシティと AWS リージョンをまたぐ自動フェイルオーバーを提供します。`global.` で始まるモデルは世界中の利用可能な全リージョンにルーティングします。
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
