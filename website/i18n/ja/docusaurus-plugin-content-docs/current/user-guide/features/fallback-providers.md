---
title: フォールバックプロバイダー
description: プライマリモデルが利用できないときに、バックアップのLLMプロバイダーへ自動的にフェイルオーバーするよう設定します。
sidebar_label: フォールバックプロバイダー
sidebar_position: 8
---

# フォールバックプロバイダー

Hermes Agentには、プロバイダーに問題が発生してもセッションを動かし続ける、3つの回復力（レジリエンス）のレイヤーがあります:

1. **[クレデンシャルプール](./credential-pools.md)** — *同じ* プロバイダーの複数のAPIキーをローテーションする（最初に試される）
2. **プライマリモデルのフォールバック** — メインモデルが失敗したとき、*別の* provider:modelに自動的に切り替える
3. **補助タスクのフォールバック** — ビジョン、圧縮、ウェブ抽出などのサイドタスクに対する独立したプロバイダー解決

クレデンシャルプールは同一プロバイダー内のローテーション（例: 複数のOpenRouterキー）を扱います。このページでは、プロバイダーをまたぐフォールバックを扱います。どちらも任意であり、独立して動作します。

## プライマリモデルのフォールバック

メインのLLMプロバイダーがエラー — レート制限、サーバー過負荷、認証失敗、接続切断 — に遭遇したとき、Hermesは会話を失うことなく、セッションの途中でバックアップのprovider:modelのペアに自動的に切り替えられます。

### 設定

最も簡単な方法は対話型マネージャーです:

```bash
hermes fallback
```

`hermes fallback` は `hermes model` のプロバイダーピッカーを再利用します — 同じプロバイダーリスト、同じ認証情報プロンプト、同じ検証です。チェーンを管理するには、サブコマンド `add`、`list`（エイリアス `ls`）、`remove`（エイリアス `rm`）、`clear` を使います。変更は `config.yaml` のトップレベルの `fallback_providers:` リストの下に永続化されます。

YAMLを直接編集したい場合は、`~/.hermes/config.yaml` に `fallback_model` セクションを追加します:

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

`provider` と `model` は両方とも **必須** です。どちらかが欠けていると、フォールバックは無効になります。

:::note `fallback_model` vs `fallback_providers`
`fallback_model`（単数形）はレガシーな単一フォールバックのキーです — Hermesは後方互換性のために引き続き尊重します。`fallback_providers`（複数形のリスト）は、順番に試される複数のフォールバックをサポートします。`hermes fallback` はこのキーに書き込みます。両方が設定されている場合、Hermesは `fallback_providers` を優先してマージします。
:::

### サポート対象プロバイダー

| プロバイダー | 値 | 要件 |
|----------|-------|-------------|
| AI Gateway | `ai-gateway` | `AI_GATEWAY_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes auth`（OAuth） |
| OpenAI Codex | `openai-codex` | `hermes model`（ChatGPT OAuth） |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`、`GH_TOKEN`、または `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | 外部プロセス（エディタ連携） |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` またはClaude Codeの認証情報 |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax（中国） | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY`（任意: `NVIDIA_BASE_URL`） |
| GMI Cloud | `gmi` | `GMI_API_KEY`（任意: `GMI_BASE_URL`） |
| StepFun | `stepfun` | `STEPFUN_API_KEY`（任意: `STEPFUN_BASE_URL`） |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google Gemini（OAuth） | `google-gemini-cli` | `hermes model`（Google OAuth。任意: `HERMES_GEMINI_PROJECT_ID`） |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY`（エイリアス: `GEMINI_API_KEY`） |
| xAI（Grok） | `xai`（エイリアス `grok`） | `XAI_API_KEY`（任意: `XAI_BASE_URL`） |
| AWS Bedrock | `bedrock` | 標準のboto3認証（`AWS_REGION` + `AWS_PROFILE` または `AWS_ACCESS_KEY_ID`） |
| Qwen Portal（OAuth） | `qwen-oauth` | `hermes model`（Qwen Portal OAuth。任意: `HERMES_QWEN_BASE_URL`） |
| MiniMax（OAuth） | `minimax-oauth` | `hermes model`（MiniMaxポータルOAuth） |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Xiaomi MiMo | `xiaomi` | `XIAOMI_API_KEY` |
| Arcee AI | `arcee` | `ARCEEAI_API_KEY` |
| GMI Cloud | `gmi` | `GMI_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Alibaba Coding Plan | `alibaba-coding-plan` | `ALIBABA_CODING_PLAN_API_KEY`（`DASHSCOPE_API_KEY` にフォールバック） |
| Kimi / Moonshot（中国） | `kimi-coding-cn` | `KIMI_CN_API_KEY` |
| StepFun | `stepfun` | `STEPFUN_API_KEY` |
| Tencent TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Azure AI Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio（ローカル） | `lmstudio` | `LM_API_KEY`（ローカルの場合はなしでも可） + `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| カスタムエンドポイント | `custom` | `base_url` + `key_env`（下記参照） |

### カスタムエンドポイントのフォールバック

カスタムのOpenAI互換エンドポイントの場合は、`base_url` と、任意で `key_env` を追加します:

```yaml
fallback_model:
  provider: custom
  model: my-local-model
  base_url: http://localhost:8000/v1
  key_env: MY_LOCAL_KEY              # APIキーを含む環境変数名
```

### フォールバックが発動するとき

フォールバックは、プライマリモデルが次のような失敗をしたときに自動的に発動します:

- **レート制限**（HTTP 429） — リトライの試行を使い切った後
- **サーバーエラー**（HTTP 500、502、503） — リトライの試行を使い切った後
- **認証失敗**（HTTP 401、403） — 即座に（リトライしても無意味なため）
- **見つからない**（HTTP 404） — 即座に
- **不正な応答** — APIが不正または空の応答を繰り返し返したとき

発動すると、Hermesは:

1. フォールバックプロバイダーの認証情報を解決する
2. 新しいAPIクライアントを構築する
3. モデル、プロバイダー、クライアントをその場で入れ替える
4. リトライカウンターをリセットし、会話を続行する

切り替えはシームレスです — 会話履歴、ツール呼び出し、コンテキストは保持されます。エージェントは、別のモデルを使うだけで、中断したまさにその場所から続行します。

:::info Per-Turn, Not Per-Session
フォールバックは **ターン単位** です: 新しいユーザーメッセージごとに、プライマリモデルが復元された状態で始まります。プライマリがターンの途中で失敗した場合、フォールバックはそのターンに限り発動します。次のメッセージでは、Hermesは再びプライマリを試します。単一のターン内では、フォールバックは最大1回しか発動しません — フォールバックも失敗した場合は、通常のエラー処理（リトライ、その後エラーメッセージ）が引き継ぎます。これにより、ターン内でのフェイルオーバーループの連鎖を防ぎつつ、毎ターン、プライマリモデルに新たなチャンスを与えます。
:::

### 例

**AnthropicネイティブのフォールバックとしてのOpenRouter:**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

**OpenRouterのフォールバックとしてのNous Portal:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_model:
  provider: nous
  model: nous-hermes-3
```

**クラウドのフォールバックとしてのローカルモデル:**
```yaml
fallback_model:
  provider: custom
  model: llama-3.1-70b
  base_url: http://localhost:8000/v1
  key_env: LOCAL_API_KEY
```

**フォールバックとしてのCodex OAuth:**
```yaml
fallback_model:
  provider: openai-codex
  model: gpt-5.3-codex
```

### フォールバックが機能する場所

| コンテキスト | フォールバック対応 |
|---------|-------------------|
| CLIセッション | ✔ |
| メッセージングゲートウェイ（Telegram、Discordなど） | ✔ |
| サブエージェントへの委譲 | ✘（サブエージェントはフォールバック設定を継承しません） |
| cronジョブ | ✘（固定のプロバイダーで実行されます） |
| 補助タスク（ビジョン、圧縮） | ✘（独自のプロバイダーチェーンを使います — 下記参照） |

:::tip
`fallback_model` に対する環境変数はありません — 設定は `config.yaml` を通じてのみ行います。これは意図的なものです: フォールバックの設定は意図的な選択であり、古いシェルのexportが上書きしてよいものではありません。
:::

---

## 補助タスクのフォールバック

Hermesは、サイドタスクに別の軽量モデルを使います。各タスクには独自のプロバイダー解決チェーンがあり、これが組み込みのフォールバックシステムとして機能します。

### 独立したプロバイダー解決を持つタスク

| タスク | 内容 | 設定キー |
|------|-------------|-----------|
| Vision | 画像分析、ブラウザのスクリーンショット | `auxiliary.vision` |
| Web Extract | ウェブページの要約 | `auxiliary.web_extract` |
| Compression | コンテキスト圧縮の要約 | `auxiliary.compression` |
| Session Search | 過去のセッションの要約 | `auxiliary.session_search` |
| Skills Hub | スキルの検索と発見 | `auxiliary.skills_hub` |
| MCP | MCPヘルパー操作 | `auxiliary.mcp` |
| Approval | スマートなコマンド承認の分類 | `auxiliary.approval` |
| Title Generation | セッションタイトルの要約 | `auxiliary.title_generation` |
| Triage Specifier | `hermes kanban specify` / ダッシュボードの✨ボタン — 一行のトリアージタスクを実際の仕様に肉付けする | `auxiliary.triage_specifier` |

### 自動検出チェーン

タスクのプロバイダーが `"auto"`（デフォルト）に設定されている場合、Hermesは1つが動作するまで順番にプロバイダーを試します:

**テキストタスク（圧縮、ウェブ抽出など）の場合:**

```text
OpenRouter → Nous Portal → カスタムエンドポイント → Codex OAuth →
APIキープロバイダー（z.ai、Kimi、MiniMax、Xiaomi MiMo、Hugging Face、Anthropic） → 断念
```

**ビジョンタスクの場合:**

```text
メインプロバイダー（ビジョン対応の場合） → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → カスタムエンドポイント → 断念
```

解決されたプロバイダーが呼び出し時に失敗した場合、Hermesには内部リトライもあります: プロバイダーがOpenRouterでなく、明示的な `base_url` が設定されていない場合、最後の手段のフォールバックとしてOpenRouterを試します。

### 補助プロバイダーの設定

各タスクは `config.yaml` で個別に設定できます:

```yaml
auxiliary:
  vision:
    provider: "auto"              # auto | openrouter | nous | codex | main | anthropic
    model: ""                     # 例: "openai/gpt-4o"
    base_url: ""                  # 直接エンドポイント（providerより優先される）
    api_key: ""                   # base_url用のAPIキー

  web_extract:
    provider: "auto"
    model: ""

  compression:
    provider: "auto"
    model: ""

  session_search:
    provider: "auto"
    model: ""
    timeout: 30
    max_concurrency: 3
    extra_body: {}

  skills_hub:
    provider: "auto"
    model: ""

  mcp:
    provider: "auto"
    model: ""
```

上記のすべてのタスクは、同じ **provider / model / base_url** のパターンに従います。コンテキスト圧縮は `auxiliary.compression` の下で設定します:

```yaml
auxiliary:
  compression:
    provider: main                                    # 他の補助タスクと同じプロバイダーオプション
    model: google/gemini-3-flash-preview
    base_url: null                                    # カスタムのOpenAI互換エンドポイント
```

そしてフォールバックモデルは次を使います:

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
  # base_url: http://localhost:8000/v1               # 任意のカスタムエンドポイント
```

`auxiliary.session_search` では、Hermesは次もサポートします:

- `max_concurrency` — 同時に実行するセッション要約の数を制限する
- `extra_body` — 要約呼び出しに、プロバイダー固有のOpenAI互換リクエストフィールドを渡す

例:

```yaml
auxiliary:
  session_search:
    provider: main
    model: glm-4.5-air
    max_concurrency: 2
    extra_body:
      enable_thinking: false
```

プロバイダーがネイティブのOpenAI互換な推論制御フィールドをサポートしていない場合、その部分について `extra_body` は役に立ちません。その場合でも、リクエストのバースト的な429を減らすために `max_concurrency` は引き続き有用です。

3つすべて — 補助、圧縮、フォールバック — は同じように動作します: `provider` で誰がリクエストを処理するかを選び、`model` でどのモデルかを選び、`base_url` でカスタムエンドポイントを指定します（providerを上書きします）。

### 補助タスクのプロバイダーオプション

これらのオプションは `auxiliary:`、`compression:`、`fallback_model:` の設定にのみ適用されます — `"main"` は、トップレベルの `model.provider` の有効な値では **ありません**。カスタムエンドポイントには、`model:` セクションで `provider: custom` を使ってください（[AIプロバイダー](/docs/integrations/providers) を参照）。

| プロバイダー | 説明 | 要件 |
|----------|-------------|-------------|
| `"auto"` | 1つが動作するまで順番にプロバイダーを試す（デフォルト） | 少なくとも1つのプロバイダーが設定されていること |
| `"openrouter"` | OpenRouterを強制 | `OPENROUTER_API_KEY` |
| `"nous"` | Nous Portalを強制 | `hermes auth` |
| `"codex"` | Codex OAuthを強制 | `hermes model` → Codex |
| `"main"` | メインエージェントが使うプロバイダーをそのまま使う（補助タスクのみ） | アクティブなメインプロバイダーが設定されていること |
| `"anthropic"` | Anthropicネイティブを強制 | `ANTHROPIC_API_KEY` またはClaude Codeの認証情報 |

### 直接エンドポイントのオーバーライド

任意の補助タスクで、`base_url` を設定するとプロバイダー解決が完全にバイパスされ、リクエストがそのエンドポイントへ直接送られます:

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` は `provider` より優先されます。Hermesは認証に設定された `api_key` を使い、設定されていない場合は `OPENAI_API_KEY` にフォールバックします。カスタムエンドポイントに対して `OPENROUTER_API_KEY` を **再利用しません**。

---

## コンテキスト圧縮のフォールバック

コンテキスト圧縮は、どのモデルとプロバイダーが要約を処理するかを制御するために `auxiliary.compression` の設定ブロックを使います:

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info Legacy migration
`compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url` を持つ古い設定は、初回読み込み時（設定バージョン17）に自動的に `auxiliary.compression.*` へ移行されます。
:::

圧縮に利用できるプロバイダーがない場合、Hermesはセッションを失敗させる代わりに、要約を生成せずに中間の会話ターンを破棄します。

---

## 委譲プロバイダーのオーバーライド

`delegate_task` によって生成されるサブエージェントは、プライマリのフォールバックモデルを **使いません**。ただし、コスト最適化のために別のprovider:modelのペアへルーティングできます:

```yaml
delegation:
  provider: "openrouter"                      # すべてのサブエージェントのプロバイダーを上書き
  model: "google/gemini-3-flash-preview"      # モデルを上書き
  # base_url: "http://localhost:1234/v1"      # または直接エンドポイントを使う
  # api_key: "local-key"
```

完全な設定の詳細は [サブエージェントへの委譲](/docs/user-guide/features/delegation) を参照してください。

---

## cronジョブのプロバイダー

cronジョブは、実行時に設定されているプロバイダーで実行されます。フォールバックモデルはサポートしません。cronジョブに別のプロバイダーを使うには、cronジョブ自体に `provider` と `model` のオーバーライドを設定します:

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

完全な設定の詳細は [スケジュールタスク（Cron）](/docs/user-guide/features/cron) を参照してください。

---

## まとめ

| 機能 | フォールバックの仕組み | 設定の場所 |
|---------|-------------------|----------------|
| メインエージェントモデル | config.yamlの `fallback_model` — エラー時のターン単位フェイルオーバー（毎ターン、プライマリが復元される） | `fallback_model:`（トップレベル） |
| Vision | 自動検出チェーン + 内部のOpenRouterリトライ | `auxiliary.vision` |
| Web抽出 | 自動検出チェーン + 内部のOpenRouterリトライ | `auxiliary.web_extract` |
| コンテキスト圧縮 | 自動検出チェーン。利用できなければ要約なしに降格 | `auxiliary.compression` |
| セッション検索 | 自動検出チェーン | `auxiliary.session_search` |
| Skills Hub | 自動検出チェーン | `auxiliary.skills_hub` |
| MCPヘルパー | 自動検出チェーン | `auxiliary.mcp` |
| 承認の分類 | 自動検出チェーン | `auxiliary.approval` |
| タイトル生成 | 自動検出チェーン | `auxiliary.title_generation` |
| トリアージ仕様化 | 自動検出チェーン | `auxiliary.triage_specifier` |
| 委譲 | プロバイダーのオーバーライドのみ（自動フォールバックなし） | `delegation.provider` / `delegation.model` |
| cronジョブ | ジョブごとのプロバイダーのオーバーライドのみ（自動フォールバックなし） | ジョブごとの `provider` / `model` |
