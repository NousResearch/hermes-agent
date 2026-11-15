---
sidebar_position: 15
title: "Microsoft Foundry"
description: "Hermes Agent を Microsoft Foundry で利用 — OpenAI スタイルと Anthropic スタイルのエンドポイント、トランスポートとデプロイ済みモデルの自動検出"
---

# Microsoft Foundry

このページの Python 依存関係コマンドは、
[PM で準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を前提としています。
依存関係を変更した後は、チェックアウトを再アクティベートして Hermes を再起動してください。

Hermes Agent の `azure-foundry` プロバイダーは、Microsoft Foundry（旧 Azure AI Foundry）と Azure OpenAI をサポートします。単一の Foundry リソースが、2 つの異なるワイヤーフォーマットを持つモデルをホストできます:

- **OpenAI スタイル** — `https://<resource>.openai.azure.com/openai/v1` のようなエンドポイントでの `POST /v1/chat/completions`。GPT-4.x、GPT-5.x、Llama、Mistral、およびほとんどのオープンウェイトモデルに使用。
- **Anthropic スタイル** — `https://<resource>.services.ai.azure.com/anthropic` のようなエンドポイントでの `POST /v1/messages`。Microsoft Foundry が Anthropic Messages API フォーマットで Claude モデルを提供するときに使用。

セットアップウィザードはエンドポイントを探り、どのトランスポートを使うか、どのデプロイが利用可能か、各モデルのコンテキスト長を自動検出します。

## 前提条件

- 少なくとも 1 つのデプロイを持つ Microsoft Foundry または Azure OpenAI リソース
- デプロイのエンドポイント URL
- API キー（Azure ポータルの「キーとエンドポイント」で取得可能）、**または** Microsoft Entra ID（Microsoft が推奨するキーレス方式）を使う場合は Foundry リソースに対する **Azure AI User** RBAC ロール。Microsoft による名称変更の展開中は、テナントによってはこのロールが **Foundry User** と表示される場合があります。

## クイックスタート

```bash
hermes model
# → Select "Azure Foundry"
# → Enter your endpoint URL
# → Choose Authentication:
#     1. API key
#     2. Microsoft Entra ID  (managed identity / workload identity / az login)
# → (Entra) Hermes probes DefaultAzureCredential; on success it never asks for a key
# → (API key) Enter your API key
# Hermes probes the endpoint and auto-detects transport + models
# → Pick a model from the list (or type a deployment name manually)
```

ウィザードは次のようにします:

1. **URL パスを嗅ぎ分ける** — `/anthropic` で終わる URL は Microsoft Foundry Claude ルートとして認識されます。
2. **`GET <base>/models` を探る** — エンドポイントが OpenAI 形状のモデルリストを返す場合、Hermes は `chat_completions` に切り替え、返されたデプロイ ID でピッカーを事前入力します。
3. **Anthropic Messages 形状を探る** — `/models` を公開しないが Anthropic Messages フォーマットを受け入れるエンドポイント向けのフォールバック。
4. **手動入力にフォールバック** — すべての探査を拒否するプライベート/ゲート付きエンドポイントでも動作します。API モードを選び、デプロイ名を手で入力します。

選択したモデルのコンテキスト長は、Hermes の標準メタデータチェーン（`models.dev`、プロバイダーメタデータ、ハードコードされたファミリーフォールバック）を介して解決され、モデルが自身のコンテキストウィンドウを正しくサイズ設定できるよう `config.yaml` に保存されます。

## Microsoft Entra ID（キーレス、RBAC）— 推奨

Microsoft は本番の Foundry ワークロードに対して [Microsoft Entra ID によるキーレス認証](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id)を推奨しています。Hermes は**両方**の API サーフェスで Entra ID をサポートします:

- **OpenAI スタイル**（`api_mode: chat_completions` / `codex_responses`）— GPT-4/5、Llama、Mistral、DeepSeek など。
- **Anthropic スタイル**（`api_mode: anthropic_messages`）— Microsoft Foundry 上の Claude モデル。

Foundry の RBAC はリソース単位であり（`Azure AI User` で両方のサーフェスへのアクセスが付与されます。テナントによっては `Foundry User` と表示されます）、Microsoft は両方に同じ推論スコープ（`https://ai.azure.com/.default`）を記載しています。内部的には:

- OpenAI スタイルは、OpenAI Python SDK ネイティブの callable な `api_key=` 契約を使います — SDK がリクエストごとに新しい JWT を自動的に発行します。
- Anthropic スタイルは、`agent.azure_identity_adapter.build_bearer_http_client` がリクエストイベントフックをインストールした `httpx.Client` を使います。Anthropic SDK は callable な `auth_token` をネイティブに受け付けないためです。このフックは送信リクエストごとに `Authorization: Bearer <fresh-jwt>` を書き換えます。同じ Microsoft RBAC、同じ Foundry スコープであり、違いは SDK の契約だけです。

### なぜ Entra ID を使うのか？

- ローテーションや失効が必要な長期 API キーが不要。
- RBAC によるアクセス制御 — Foundry リソースで `Azure AI User` を付与または削除するだけで、設定の書き換えは不要。
- すべての呼び出し元が 1 つの静的キーを共有するのではなく、アクセスログと監査ログが割り当て先ごとに分離される。
- マネージド ID を介して、Azure VM、AKS ポッド、App Service、Functions、Container Apps、Foundry Agent Service で単一の認証サーフェスを利用できる。
- CI/CD パイプライン向けのワークロード ID およびサービスプリンシパルのフロー。

### 初回セットアップ（Azure 側）

1. Azure ポータルで Foundry リソースを開き → **アクセス制御 (IAM)** → **追加 → ロールの割り当ての追加** を選択します。
2. **Azure AI User** ロール（テナントで名称変更済みのロールがある場合は **Foundry User**）を選びます。
3. 次に割り当てます:
   - `az login` を使ったローカル開発には **自分のユーザーアカウント**。
   - Azure ホストのコンピュートには **マネージド ID またはワークロード ID**（本番環境に推奨）。
   - Hermes がホステッドエージェント内で動作する場合は **Foundry Agent Service ホステッドエージェントのエージェント ID**。
   - ワークロード ID が利用できない CI/CD パイプラインには **サービスプリンシパル**。
4. ロールが反映されるまで約 5 分待ちます。

Azure CLI で同等の操作:

```bash
az role assignment create \
  --assignee <principal-or-agent-identity-client-id> \
  --role "Azure AI User" \
  --scope <foundry-resource-id>
```

### 初回セットアップ（Hermes 側）

```bash
hermes model
# → Select "Azure Foundry"
# → Enter your endpoint URL
# → Authentication: 2 (Microsoft Entra ID)
# → (optional) user-assigned managed identity client ID
# → (optional) Azure tenant ID
# → Hermes probes DefaultAzureCredential() and reports which inner
#    credential succeeded (e.g. AzureCliCredential, ManagedIdentityCredential)
```

ウィザードは時間制限付きのプリフライト探査（10 秒のタイムアウト）を実行します。失敗した場合は「save anyway, validate later」（とりあえず保存して後で検証）を提案します — まだ認証情報を持たないが実行時には持つマシンで設定する場合（例: マネージド ID デプロイ用の設定を準備する場合）に便利です。

`azure-identity` は、Hermes の遅延インストールパスを介して初回使用時に自動的にインストールされます。事前にインストールするには:

```bash
python -c "import pm; pm.sync_venv(['azure-identity'], explicit=True)"
```

### `config.yaml` に書き込まれる設定

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  auth_mode: entra_id
  default: gpt-4o
  context_length: 128000
  entra:
    scope: https://ai.azure.com/.default        # only when overriding the default
```

Hermes が `config.yaml` で管理する Entra 固有の設定項目は 1 つだけです:

- **`scope`** — OAuth リソーススコープ。デフォルトは Microsoft が記載している推論スコープ（`https://ai.azure.com/.default`）です。リソースが非標準のオーディエンスに対してプロビジョニングされている場合にのみ上書きしてください。

それ以外（テナント、サービスプリンシパルのシークレット、フェデレーションされたトークンファイル、ソブリンクラウドの機関、ブローカー設定）はすべて、`azure-identity` が標準の `AZURE_*` 環境変数から直接読み取ります — 下記の[認証情報の解決順序](#credential-resolution-order)を参照してください。これらは Microsoft の SDK リファレンスの説明どおりに、`~/.hermes/.env` またはデプロイ環境に設定してください。

Entra モードでは `~/.hermes/.env` にシークレットは保存されません — `azure-identity` はトークンをプロセス内（利用可能な場合は OS のキーチェーン / `~/.IdentityService`）にキャッシュします。

### 認証情報の解決順序 {#credential-resolution-order}

`azure-identity` の `DefaultAzureCredential` は、トークン要求のたびに次のチェーンをたどり、最初にトークンを返した認証情報で停止します:

1. **環境認証情報** — `AZURE_TENANT_ID` ＋ `AZURE_CLIENT_ID` ＋ `AZURE_CLIENT_SECRET`（または `AZURE_CLIENT_CERTIFICATE_PATH` / `AZURE_FEDERATED_TOKEN_FILE`）。
2. **ワークロード ID** — `AZURE_FEDERATED_TOKEN_FILE`（AKS のフェデレーションされたトークン / OIDC）。
3. **マネージド ID** — 仮想マシンでは IMDS エンドポイント（`169.254.169.254`）、App Service / Functions / Container Apps では `IDENTITY_ENDPOINT`。Foundry Agent Service のホステッドエージェントは、ホステッドエージェントのエージェント ID を使います。
4. **Visual Studio Code** — Azure アカウント拡張機能。
5. **Azure CLI** — `az login` セッション。
6. **Azure Developer CLI** — `azd auth login`。
7. **Azure PowerShell** — `Connect-AzAccount`。
8. **ブローカー**（Windows / WSL のみ）— Web アカウントマネージャー。

無人で実行される Hermes では、対話型ブラウザー認証情報はデフォルトで除外されます。代わりに Azure CLI、Azure Developer CLI、マネージド ID、ワークロード ID、またはサービスプリンシパルの認証情報を使ってください。

**多重化プロファイル（`gateway.multiplex_profiles: true`）:** このチェーンのすべてのソースは*プロセス*から解決されます — 起動プロファイルの `AZURE_*`、その `az login` セッション、ホストのマネージド ID です。そのため、独自の `AZURE_*` を設定していない提供対象プロファイルは、起動時の ID を借用するのではなく拒否されます（Vertex アダプターがアプリケーションのデフォルト認証情報に適用するのと同じルールです）。各プロファイルの `.env` に、それぞれ独自の `AZURE_TENANT_ID` ＋ `AZURE_CLIENT_ID` ＋ `AZURE_CLIENT_SECRET`（または `AZURE_FEDERATED_TOKEN_FILE`）を設定してください。`AZURE_CLIENT_ID` だけを設定すると、そのプロファイルはホストのユーザー割り当てマネージド ID を使うようになります。単一プロファイルでの実行（`hermes`、`hermes -p beta`）では完全なチェーンが維持されます。

### デプロイパターン

**ローカル開発:**
```bash
az login
hermes model   # pick Azure Foundry → Entra ID
hermes         # uses your az login token
```

**Azure VM / Functions / App Service / Container Apps（システム割り当てマネージド ID）:**
1. コンピュートリソースでシステム割り当て ID を有効にします。
2. Foundry リソースでその ID に `Azure AI User`（または `Foundry User`）を付与します。
3. config.yaml で `model.auth_mode: entra_id` を設定します — 環境変数は不要です。

**Azure VM / Functions / App Service / Container Apps（ユーザー割り当てマネージド ID）:**
- `DefaultAzureCredential` が正しい ID を選ぶように、`AZURE_CLIENT_ID` にユーザー割り当て ID のクライアント ID を設定します。

**Foundry Agent Service ホステッドエージェント:**
- ホステッドエージェントを作成し、Foundry リソースでそのエージェントの ID に `Azure AI User`（または `Foundry User`）を付与します。Hermes はホステッドエージェント内から `ManagedIdentityCredential` を使います。ロールの割り当ては、親プロジェクトや自分のユーザーだけでなく、エージェント ID に対して行う必要があります。

**AKS ワークロード ID（AAD Pod Identity の後継）:**
- ポッドのサービスアカウントにワークロード ID のクライアント ID をアノテーションします。
- ポッドのフェデレーションされたトークンファイルは `AZURE_FEDERATED_TOKEN_FILE` を介して自動検出されます。
- `model.auth_mode: entra_id` は追加の設定変更なしで動作します。

**CI でのサービスプリンシパル:**
- ランナーの環境に `AZURE_TENANT_ID`、`AZURE_CLIENT_ID`、`AZURE_CLIENT_SECRET` を設定します。

#### ソブリンクラウド（Government、China） {#sovereign-clouds-government-china}

`AZURE_AUTHORITY_HOST` をエクスポートします（例: Azure Government では `https://login.microsoftonline.us`、Azure China では `https://login.partner.microsoftonline.cn`）。`azure-identity` がこれを直接読み取ります。

### ヘルスチェック

`model.auth_mode: entra_id` の場合、`hermes doctor` は `DefaultAzureCredential` に対して 10 秒の探査を実行し、どの内部認証情報が成功したか（環境変数の有無、マネージド ID エンドポイントへの到達可否など）を報告します。

`hermes auth` は構造化されたステータスブロックを表示します:

```
azure-foundry (Microsoft Entra ID):
  Endpoint: https://my-resource.openai.azure.com/openai/v1
  Scope: https://ai.azure.com/.default
  Status: configured; live token probe is skipped here
```

メインモデルに従う補助タスク（`provider: auto` — セッションタイトル、コンテキスト圧縮、スマート承認）は、再認証せずにメインセッションの Entra トークンプロバイダーを再利用します。CLI の `--api-key` 文字列は、一時的なテスト用に引き続きそれを上書きできます。

### 制限事項

- **Anthropic スタイルのエンドポイントは httpx イベントフックを使います。** Anthropic Python SDK は callable な `auth_token` をネイティブに受け付けません（≤ 0.86.0）。Hermes はカスタム `httpx.Client` にリクエストイベントフックをインストールし、送信リクエストごとに新しい JWT を発行して `Authorization: Bearer <jwt>` を書き換えます。これは OpenAI SDK ネイティブの `Callable[[], str]` 契約と機能的に同等ですが、間接層が 1 つ増えます。将来のリリースで Anthropic SDK が callable 認証をファーストクラスでサポートした場合、Hermes は透過的にそれに切り替えます。
- **バッチジョブと `multiprocessing.Pool`。** Entra トークンプロバイダーはクロージャであり、プロセス境界を越えて pickle できません。`batch_runner.py` はワーカー設定から callable を自動的に取り除き、各ワーカープロセスが `config.yaml` から独自のプロバイダーを再構築できるようにします — ユーザーの操作は不要ですが、各ワーカーは起動時にチェーンを 1 回たどります。
- **`auth.json` に Bearer JWT は永続化されません。** Hermes は `azure-identity` 内部のトークンキャッシュを複製しません。コールドスタート時は最初の推論で認証情報チェーンをたどります。

## 設定（`config.yaml` に書き込まれる）

ウィザードを実行すると、次のようなものが表示されます:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions         # or "anthropic_messages"
  default: gpt-5.4-mini              # your deployment / model name
  context_length: 400000             # auto-detected
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

- **GPT-5.x、codex、o シリーズは Responses API に自動ルーティングされます。** Microsoft Foundry は GPT-5 / codex / o1 / o3 / o4 モデルを Responses API 専用としてデプロイします — それらに対して `/chat/completions` を呼び出すと `400 "The requested operation is unsupported."` を返します。Hermes はこれらのモデルファミリーを名前で検出し、`config.yaml` がまだ `api_mode: chat_completions` であっても、`api_mode` を `codex_responses` に透過的にアップグレードします。GPT-4、GPT-4o、Llama、Mistral、その他のデプロイは `/chat/completions` のままです。
- **`api_mode: responses` は `codex_responses` の別表記として受け付けられます。** このエイリアスは `model.api_mode`、`fallback_providers` のエントリ、タスクごとの `auxiliary.<task>.api_mode`（例: GPT-5.x デプロイへの `auxiliary.vision` ルート）で機能し、同じ Responses アダプターを選択します。
- **`max_completion_tokens` が自動的に使われます。** Azure OpenAI は（直接の OpenAI と同様に）gpt-4o、o シリーズ、gpt-5.x モデルに対して `max_completion_tokens` を必要とします。Hermes はエンドポイントに基づいて正しいパラメータを送ります。
- **`api-version` を必要とする v1 以前のエンドポイント。** `https://<resource>.openai.azure.com/openai?api-version=2025-04-01-preview` のようなレガシーのベース URL がある場合、Hermes はクエリ文字列を抽出し、すべてのリクエストで `default_query` 経由で転送します（そうしないと OpenAI SDK はパスを結合する際にそれを落としてしまいます）。

## Anthropic スタイルのエンドポイント（Microsoft Foundry 経由の Claude）

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
- **`x-api-key` の代わりに Bearer 認証が使われます。** Azure の Anthropic 互換ルートは、Anthropic ネイティブの `x-api-key` ヘッダーではなく `Authorization: Bearer <key>` を必要とします。Hermes はベース URL に `azure.com` を検出し、正しいヘッダーがアップストリームに届くよう、API キーを SDK の `auth_token` フィールド経由で渡します。
- **1M コンテキストウィンドウのベータヘッダーは維持されます。** Azure では依然として、100 万トークンの Claude コンテキスト（Opus 4.6/4.7、Sonnet 4.6）が `anthropic-beta: context-1m-2025-08-07` ヘッダーで制限されています。Hermes は Azure のパスではこのベータヘッダーを維持します（一部のサブスクリプションが拒否するためネイティブの Anthropic OAuth リクエストからは取り除かれますが、Azure では必須です）。
- **OAuth トークンのリフレッシュは無効化されます。** Azure のデプロイは静的な API キーを使います。Anthropic Console に適用される `~/.claude/.credentials.json` の OAuth トークンリフレッシュループは、Claude Code の OAuth トークンがセッション途中で Azure キーを上書きするのを防ぐため、Azure エンドポイントでは明示的にスキップされます。
- **`hermes doctor` は同じルートを探査します。** `/anthropic` ルートには `GET /models` がないため、接続チェックはランタイムと同じ Bearer 認証と `api-version` クエリを使って 1 トークンの `POST /v1/messages` を送信します。200（または Messages API からの 400）ならエンドポイントは正常、401/403 なら認証の問題として報告されます。

## 代替案: `provider: anthropic` ＋ Azure ベース URL

既に `provider: anthropic` を設定済みで、Claude のためにそれを Microsoft Foundry に向けたいだけなら、`azure-foundry` プロバイダーを完全にスキップできます:

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

- Azure OpenAI v1 エンドポイント（`<resource>.openai.azure.com/openai/v1`）は、リソースの**利用可能な**モデルカタログとともに `GET /models` を公開します。Hermes はこのリストを使って、セットアップウィザードのモデルピッカー**および**セッション内の `/model azure-foundry` ピッカー（CLI、TUI、Desktop、ゲートウェイ）を事前入力するため、`hermes setup` を再実行せずにデプロイを切り替えられます。
- Microsoft Foundry の `/anthropic` ルート: URL パスで検出され、モデル名は手動で入力されます（そこには `/models` がないため、`/model` ピッカーには現在の選択と、宣言した `providers.azure-foundry.models` だけが表示されます）。
- プライベート / ファイアウォール内のエンドポイント: フレンドリーな「探れませんでした」メッセージとともに手動入力。
- Entra ID（`model.auth_mode: entra_id`、`AZURE_FOUNDRY_API_KEY` なし）: `model.base_url`（または `AZURE_FOUNDRY_BASE_URL`）が設定されるとすぐに、`/model` ピッカーにプロバイダーが表示されます — 行を表示するためだけにトークンが発行されることはありません。

デプロイ名はいつでも直接入力できます — Hermes は返されたリストに対して検証しません。

ピッカーを実際に使うデプロイに固定したい場合（カタログは長くなることがあります）、または `/models` のないエンドポイントでデプロイを一覧表示したい場合は、`config.yaml` で宣言してください。宣言したデプロイはライブカタログより先に表示されます:

```yaml
providers:
  azure-foundry:
    models:
      - gpt-5.4
      - kimi-k2.6
```

Azure Foundry がアクティブなプロバイダーである間、ランタイムピッカーは `model.base_url` からエンドポイントを解決します。別のプロバイダーに切り替えた後も行を表示したままにしたい場合は、`AZURE_FOUNDRY_BASE_URL` も設定してください。

## 環境変数

| 変数 | 目的 |
|----------|---------|
| `AZURE_FOUNDRY_API_KEY` | Microsoft Foundry / Azure OpenAI のプライマリ API キー（api_key モード） |
| `AZURE_FOUNDRY_BASE_URL` | エンドポイント URL（`hermes model` で設定；env 変数はフォールバックとして使われる） |
| `AZURE_ANTHROPIC_KEY` | `provider: anthropic` ＋ Azure ベース URL で使用（`ANTHROPIC_API_KEY` の代替） |
| `AZURE_TENANT_ID` | サービスプリンシパルフロー用の Entra ID テナント |
| `AZURE_CLIENT_ID` | Entra ID クライアント ID（サービスプリンシパル、ワークロード ID、またはユーザー割り当てマネージド ID） |
| `AZURE_CLIENT_SECRET` | サービスプリンシパルのシークレット |
| `AZURE_CLIENT_CERTIFICATE_PATH` | サービスプリンシパルの証明書（シークレットの代替） |
| `AZURE_FEDERATED_TOKEN_FILE` | ワークロード ID のフェデレーションされたトークンのパス（AKS） |
| `AZURE_AUTHORITY_HOST` | ソブリンクラウドの機関ホストの上書き |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | App Service、Functions、Container Apps 用のマネージド ID エンドポイント。VM は通常 IMDS を使います |

Azure SDK は `AZURE_*` 環境変数を直接読み取ります。Hermes は、`hermes doctor` の出力でどのソースが存在するかを報告する以外に、これらを検査することはありません。

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
  api_mode: anthropic_messages   # or chat_completions
```

**Entra ID: `auth_mode: entra_id` に切り替えた後に「credential chain exhausted」または 401 Unauthorized。**
- `az login` を実行して開発者セッションを更新してください（キャッシュされたトークンの有効期限が切れている可能性があります）。
- `Azure AI User`（または `Foundry User`）のロール割り当てが有効になっていることを確認してください: `az role assignment list --assignee <user-or-identity-id>` で Foundry リソース上のロールが一覧表示されるはずです。ロールの反映には最大 5 分かかることがあります。
- ユーザー割り当てマネージド ID の場合は、`AZURE_CLIENT_ID` がコンピュートリソースにアタッチされた ID と一致していることを再確認してください。
- `hermes doctor` を実行してください — Azure Entra の探査がトークン取得の成否を報告し、対処のヒントを示します。

**Entra ID: ウィザードのプリフライトが停止する、またはタイムアウトする。**
10 秒のプリフライトはソフトチェックです。「Save anyway and validate later」を選び、ターゲット環境にデプロイした後で `hermes doctor` を実行してください。一般的な原因は、トークンサービスに到達できないことや、ローカルのログイン状態が古いことです — CI ではワークロード ID を優先し、サービスプリンシパルを使う場合は `AZURE_TENANT_ID`＋`AZURE_CLIENT_ID`＋`AZURE_CLIENT_SECRET` を設定し、ローカル開発では `az login` を実行してください。

**Entra ID 使用時に Anthropic スタイルのエンドポイントで 401。**
Foundry リソースに同じ `Azure AI User`（または `Foundry User`）ロールが割り当てられていることを確認してください（このロールは `/openai/v1` と `/anthropic` の両方のパスをカバーします）。ウィザード中の OpenAI スタイルの探査は成功するのに実行時に `claude-*` リクエストが失敗する場合、最も一般的な原因は以前のウィザード実行で残った古い `model.entra.scope` です — ランタイムがデフォルトの `https://ai.azure.com/.default` スコープにフォールバックするよう、`config.yaml` から `entra.scope` の行を削除してください。

## 関連

- [環境変数](../reference/environment-variables.md)
- [設定](../user-guide/configuration.md)
- [AWS Bedrock](./aws-bedrock.md) — もう 1 つの主要なクラウドプロバイダー連携
- [Microsoft: Configure Entra ID for Foundry](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id) — キーレス方式のアップストリームドキュメント
