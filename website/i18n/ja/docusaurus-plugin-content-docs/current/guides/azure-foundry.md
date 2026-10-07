---
sidebar_position: 15
title: "Azure AI Foundry"
description: "Hermes Agent を Azure AI Foundry で利用 — OpenAI スタイルと Anthropic スタイルのエンドポイント、トランスポートとデプロイ済みモデルの自動検出"
---

# Azure AI Foundry

Hermes Agent は、Azure AI Foundry（および Azure OpenAI）をファーストクラスのプロバイダーとしてサポートします。単一の Azure リソースが、2 つの異なるワイヤーフォーマットを持つモデルをホストできます:

- **OpenAI スタイル** — `https://<resource>.openai.azure.com/openai/v1` のようなエンドポイントでの `POST /v1/chat/completions`。GPT-4.x、GPT-5.x、Llama、Mistral、およびほとんどのオープンウェイトモデルに使用。
- **Anthropic スタイル** — `https://<resource>.services.ai.azure.com/anthropic` のようなエンドポイントでの `POST /v1/messages`。Azure Foundry が Anthropic Messages API フォーマットで Claude モデルを提供するときに使用。

セットアップウィザードはエンドポイントを探り、どのトランスポートを使うか、どのデプロイが利用可能か、各モデルのコンテキスト長を自動検出します。

## 前提条件

- 少なくとも 1 つのデプロイを持つ Azure AI Foundry または Azure OpenAI リソース
- そのリソースの API キー（Azure ポータルの「キーとエンドポイント」で取得可能）
- デプロイのエンドポイント URL

## クイックスタート

```bash
hermes model
# → "Azure Foundry" を選択
# → エンドポイント URL を入力
# → API キーを入力
# Hermes がエンドポイントを探り、トランスポート + モデルを自動検出
# → リストからモデルを選択（または手動でデプロイ名を入力）
```

ウィザードは次のようにします:

1. **URL パスを嗅ぎ分ける** — `/anthropic` で終わる URL は Azure Foundry Claude ルートとして認識されます。
2. **`GET <base>/models` を探る** — エンドポイントが OpenAI 形状のモデルリストを返す場合、Hermes は `chat_completions` に切り替え、返されたデプロイ ID でピッカーを事前入力します。
3. **Anthropic Messages 形状を探る** — `/models` を公開しないが Anthropic Messages フォーマットを受け入れるエンドポイント向けのフォールバック。
4. **手動入力にフォールバック** — すべての探査を拒否するプライベート/ゲート付きエンドポイントでも動作します。API モードを選び、デプロイ名を手で入力します。

選択したモデルのコンテキスト長は、Hermes の標準メタデータチェーン（`models.dev`、プロバイダーメタデータ、ハードコードされたファミリーフォールバック）を介して解決され、モデルが自身のコンテキストウィンドウを正しくサイズ設定できるよう `config.yaml` に保存されます。

## 設定（`config.yaml` に書き込まれる）

ウィザードを実行すると、次のようなものが表示されます:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions         # または "anthropic_messages"
  default: gpt-5.4-mini              # あなたのデプロイ / モデル名
  context_length: 400000             # 自動検出
```

そして `~/.hermes/.env` には:

```
AZURE_FOUNDRY_API_KEY=<your-azure-key>
```

## OpenAI スタイルのエンドポイント（GPT、Llama など）

Azure OpenAI の v1 GA エンドポイントは、最小限の変更で標準的な `openai` Python クライアントを受け入れます:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  default: gpt-5.4
```

重要な動作:

- **GPT-5.x、codex、o シリーズは Responses API に自動ルーティングされます。** Azure Foundry は GPT-5 / codex / o1 / o3 / o4 モデルを Responses API 専用としてデプロイします — それらに対して `/chat/completions` を呼び出すと `400 "The requested operation is unsupported."` を返します。Hermes はこれらのモデルファミリーを名前で検出し、`config.yaml` がまだ `api_mode: chat_completions` であっても、`api_mode` を `codex_responses` に透過的にアップグレードします。GPT-4、GPT-4o、Llama、Mistral、その他のデプロイは `/chat/completions` のままです。
- **`max_completion_tokens` が自動的に使われます。** Azure OpenAI は（直接の OpenAI と同様に）gpt-4o、o シリーズ、gpt-5.x モデルに対して `max_completion_tokens` を必要とします。Hermes はエンドポイントに基づいて正しいパラメータを送ります。
- **`api-version` を必要とする v1 以前のエンドポイント。** `https://<resource>.openai.azure.com/openai?api-version=2025-04-01-preview` のようなレガシーのベース URL がある場合、Hermes はクエリ文字列を抽出し、すべてのリクエストで `default_query` 経由で転送します（そうしないと OpenAI SDK はパスを結合する際にそれを落としてしまいます）。

## Anthropic スタイルのエンドポイント（Azure Foundry 経由の Claude）

Claude のデプロイには Anthropic スタイルのルートを使います:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.services.ai.azure.com/anthropic
  api_mode: anthropic_messages
  default: claude-sonnet-4-6
```

重要な動作:

- **`/v1` はベース URL から取り除かれます。** Anthropic SDK はすべてのリクエスト URL に `/v1/messages` を付加します — Hermes は二重 `/v1` パスを避けるため、URL を SDK に渡す前に末尾の `/v1` を削除します。
- **`api-version` は URL に付加されず、`default_query` 経由で送られます。** Azure Anthropic は `api-version` クエリ文字列を必要とします。それをベース URL に焼き込むと `/anthropic?api-version=.../v1/messages` のような不正なパスになり 404 を返します。Hermes は代わりに Anthropic SDK の `default_query` 経由で `api-version=2025-04-15` を渡します。
- **OAuth トークンのリフレッシュは無効化されます。** Azure のデプロイは静的な API キーを使います。Anthropic Console に適用される `~/.claude/.credentials.json` の OAuth トークンリフレッシュループは、Claude Code の OAuth トークンがセッション途中で Azure キーを上書きするのを防ぐため、Azure エンドポイントでは明示的にスキップされます。

## 代替案: `provider: anthropic` ＋ Azure ベース URL

既に `provider: anthropic` を設定済みで、Claude のためにそれを Azure AI Foundry に向けたいだけなら、`azure-foundry` プロバイダーを完全にスキップできます:

```yaml
model:
  provider: anthropic
  base_url: https://my-resource.services.ai.azure.com/anthropic
  key_env: AZURE_ANTHROPIC_KEY
  default: claude-sonnet-4-6
```

`~/.hermes/.env` に `AZURE_ANTHROPIC_KEY` を設定します。Hermes はベース URL に `azure.com` を検出し、Claude Code の OAuth トークンチェーンをショートサーキットして、Azure キーが `x-api-key` 認証で直接使われるようにします。

`key_env` は正規の snake_case フィールド名です；`api_key_env`（および camelCase の `keyEnv` / `apiKeyEnv`）はエイリアスとして受け入れられます。`key_env` と `AZURE_ANTHROPIC_KEY`/`ANTHROPIC_API_KEY` の両方が設定されている場合、`key_env` で指定された env 変数が優先されます。

## モデル探索

Azure は、*デプロイ済み*のモデルデプロイを一覧表示する純粋な API キーのみのエンドポイントを**公開していません**。デプロイの列挙には、推論 API キーではなく、Azure AD プリンシパルを伴う Azure Resource Manager 認証（`az cognitiveservices account deployment list`）が必要です。

Hermes ができること:

- Azure OpenAI v1 エンドポイント（`<resource>.openai.azure.com/openai/v1`）は、リソースの**利用可能な**モデルカタログとともに `GET /models` を公開します。Hermes はこのリストを使ってモデルピッカーを事前入力します。
- Azure Foundry の `/anthropic` ルート: URL パスで検出され、モデル名は手動で入力されます。
- プライベート / ファイアウォール内のエンドポイント: フレンドリーな「探れませんでした」メッセージとともに手動入力。

デプロイ名はいつでも直接入力できます — Hermes は返されたリストに対して検証しません。

## 環境変数

| 変数 | 目的 |
|----------|---------|
| `AZURE_FOUNDRY_API_KEY` | Azure AI Foundry / Azure OpenAI のプライマリ API キー |
| `AZURE_FOUNDRY_BASE_URL` | エンドポイント URL（`hermes model` で設定；env 変数はフォールバックとして使われる） |
| `AZURE_ANTHROPIC_KEY` | `provider: anthropic` ＋ Azure ベース URL で使用（`ANTHROPIC_API_KEY` の代替） |

## トラブルシューティング

**gpt-5.x デプロイで 401 Unauthorized。**
Azure は gpt-5.x を `/responses` ではなく `/chat/completions` で提供します。URL に `openai.azure.com` が含まれる場合、Hermes はこれを自動的に処理しますが、`Invalid API key` のボディとともに 401 が表示される場合は、`config.yaml` の `api_mode` が `chat_completions` であることを確認してください。

**`/v1/messages?api-version=.../v1/messages` で 404。**
これは修正前の Azure Anthropic セットアップにおける不正な URL のバグです。Hermes をアップグレードしてください — `api-version` パラメータはベース URL に焼き込まれるのではなく `default_query` 経由で渡されるようになったため、SDK が URL の結合中にそれを破損できなくなりました。

**ウィザードが「Auto-detection incomplete.」と言う。**
エンドポイントが `/models` 探査と Anthropic Messages 探査の両方を拒否しました。これは、ファイアウォールの背後にあるプライベートエンドポイントや IP 許可リストを持つエンドポイントでは正常です。手動の API モード選択にフォールバックし、デプロイ名を入力してください — すべて動作します。Hermes はピッカーを事前入力できないだけです。

**間違ったトランスポートが選ばれた。**
`hermes model` を再度実行すると、ウィザードが再探査します。探査がまだ間違ったモードを選ぶ場合、`config.yaml` を直接編集できます:

```yaml
model:
  provider: azure-foundry
  api_mode: anthropic_messages   # または chat_completions
```

## 関連

- [環境変数](/docs/reference/environment-variables)
- [設定](/docs/user-guide/configuration)
- [AWS Bedrock](/docs/guides/aws-bedrock) — もう 1 つの主要なクラウドプロバイダー連携
