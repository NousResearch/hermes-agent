---
sidebar_position: 2
title: "環境変数"
description: "Hermes Agentが使用するすべての環境変数の完全なリファレンス"
---

# 環境変数リファレンス

すべての変数は `~/.hermes/.env` に記述します。`hermes config set VAR value` で設定することもできます。

## LLMプロバイダー

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter APIキー（柔軟性の点で推奨） |
| `OPENROUTER_BASE_URL` | OpenRouter互換のベースURLを上書き |
| `HERMES_OPENROUTER_CACHE` | OpenRouterのレスポンスキャッシュを有効化（`1`/`true`/`yes`/`on`）。config.yamlの `openrouter.response_cache` を上書きします。[Response Caching](https://openrouter.ai/docs/guides/features/response-caching) を参照してください。 |
| `HERMES_OPENROUTER_CACHE_TTL` | キャッシュのTTL（秒単位、1〜86400）。config.yamlの `openrouter.response_cache_ttl` を上書きします。 |
| `NOUS_BASE_URL` | Nous PortalのベースURLを上書き（通常は不要。開発・テスト用途のみ） |
| `NOUS_INFERENCE_BASE_URL` | Nousの推論エンドポイントを直接上書き |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway APIキー（[ai-gateway.vercel.sh](https://ai-gateway.vercel.sh)） |
| `AI_GATEWAY_BASE_URL` | AI GatewayのベースURLを上書き（デフォルト: `https://ai-gateway.vercel.sh/v1`） |
| `OPENAI_API_KEY` | カスタムOpenAI互換エンドポイント用のAPIキー（`OPENAI_BASE_URL` と併用） |
| `OPENAI_BASE_URL` | カスタムエンドポイント（VLLM、SGLangなど）のベースURL |
| `COPILOT_GITHUB_TOKEN` | Copilot API用のGitHubトークン — 最優先（OAuth `gho_*` またはきめ細かいPAT `github_pat_*`。クラシックPAT `ghp_*` は**サポートされていません**） |
| `GH_TOKEN` | GitHubトークン — Copilotの第2優先（`gh` CLIでも使用） |
| `GITHUB_TOKEN` | GitHubトークン — Copilotの第3優先 |
| `HERMES_COPILOT_ACP_COMMAND` | Copilot ACP CLIバイナリのパスを上書き（デフォルト: `copilot`） |
| `COPILOT_CLI_PATH` | `HERMES_COPILOT_ACP_COMMAND` のエイリアス |
| `HERMES_COPILOT_ACP_ARGS` | Copilot ACPの引数を上書き（デフォルト: `--acp --stdio`） |
| `COPILOT_ACP_BASE_URL` | Copilot ACPのベースURLを上書き |
| `GLM_API_KEY` | z.ai / ZhipuAI GLM APIキー（[z.ai](https://z.ai)） |
| `ZAI_API_KEY` | `GLM_API_KEY` のエイリアス |
| `Z_AI_API_KEY` | `GLM_API_KEY` のエイリアス |
| `GLM_BASE_URL` | z.aiのベースURLを上書き（デフォルト: `https://api.z.ai/api/paas/v4`） |
| `KIMI_API_KEY` | Kimi / Moonshot AI APIキー（[moonshot.ai](https://platform.moonshot.ai)） |
| `KIMI_BASE_URL` | KimiのベースURLを上書き（デフォルト: `https://api.moonshot.ai/v1`） |
| `KIMI_CN_API_KEY` | Kimi / Moonshot China APIキー（[moonshot.cn](https://platform.moonshot.cn)） |
| `ARCEEAI_API_KEY` | Arcee AI APIキー（[chat.arcee.ai](https://chat.arcee.ai/)） |
| `ARCEE_BASE_URL` | ArceeのベースURLを上書き（デフォルト: `https://api.arcee.ai/api/v1`） |
| `GMI_API_KEY` | GMI Cloud APIキー（[gmicloud.ai](https://www.gmicloud.ai/)） |
| `GMI_BASE_URL` | GMI CloudのベースURLを上書き（デフォルト: `https://api.gmi-serving.com/v1`） |
| `MINIMAX_API_KEY` | MiniMax APIキー — グローバルエンドポイント（[minimax.io](https://www.minimax.io)）。**`minimax-oauth` では使用されません**（OAuthパスはブラウザログインを使用）。 |
| `MINIMAX_BASE_URL` | MiniMaxのベースURLを上書き（デフォルト: `https://api.minimax.io/anthropic` — HermesはMiniMaxのAnthropic Messages互換エンドポイントを使用）。**`minimax-oauth` では使用されません**。 |
| `MINIMAX_CN_API_KEY` | MiniMax APIキー — 中国エンドポイント（[minimaxi.com](https://www.minimaxi.com)）。**`minimax-oauth` では使用されません**（OAuthパスはブラウザログインを使用）。 |
| `MINIMAX_CN_BASE_URL` | MiniMax ChinaのベースURLを上書き（デフォルト: `https://api.minimaxi.com/anthropic`）。**`minimax-oauth` では使用されません**。 |
| `KILOCODE_API_KEY` | Kilo Code APIキー（[kilo.ai](https://kilo.ai)） |
| `KILOCODE_BASE_URL` | Kilo CodeのベースURLを上書き（デフォルト: `https://api.kilo.ai/api/gateway`） |
| `XIAOMI_API_KEY` | Xiaomi MiMo APIキー（[platform.xiaomimimo.com](https://platform.xiaomimimo.com)） |
| `XIAOMI_BASE_URL` | Xiaomi MiMoのベースURLを上書き（デフォルト: `https://api.xiaomimimo.com/v1`） |
| `TOKENHUB_API_KEY` | Tencent TokenHub APIキー（[tokenhub.tencentmaas.com](https://tokenhub.tencentmaas.com)） |
| `TOKENHUB_BASE_URL` | Tencent TokenHubのベースURLを上書き（デフォルト: `https://tokenhub.tencentmaas.com/v1`） |
| `AZURE_FOUNDRY_API_KEY` | Azure AI Foundry / Azure OpenAI APIキー（[ai.azure.com](https://ai.azure.com/)） |
| `AZURE_FOUNDRY_BASE_URL` | Azure AI FoundryのエンドポイントURL（例: OpenAIスタイルは `https://<resource>.openai.azure.com/openai/v1`、Anthropicスタイルは `https://<resource>.services.ai.azure.com/anthropic`） |
| `AZURE_ANTHROPIC_KEY` | `provider: anthropic` + Azure FoundryのClaudeデプロイメントを指す `base_url` 用のAzure Anthropic APIキー（AnthropicとAzure Anthropicの両方を設定している場合の `ANTHROPIC_API_KEY` の代替） |
| `HF_TOKEN` | Inference Providers用のHugging Faceトークン（[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)） |
| `HF_BASE_URL` | Hugging FaceのベースURLを上書き（デフォルト: `https://router.huggingface.co/v1`） |
| `GOOGLE_API_KEY` | Google AI Studio APIキー（[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)） |
| `GEMINI_API_KEY` | `GOOGLE_API_KEY` のエイリアス |
| `GEMINI_BASE_URL` | Google AI StudioのベースURLを上書き |
| `HERMES_GEMINI_CLIENT_ID` | `google-gemini-cli` のPKCEログイン用OAuthクライアントID（任意。デフォルトはGoogleの公開gemini-cliクライアント） |
| `HERMES_GEMINI_CLIENT_SECRET` | `google-gemini-cli` 用のOAuthクライアントシークレット（任意） |
| `HERMES_GEMINI_PROJECT_ID` | 有料Geminiティア用のGCPプロジェクトID（無料ティアは自動プロビジョニング） |
| `ANTHROPIC_API_KEY` | Anthropic Console APIキー（[console.anthropic.com](https://console.anthropic.com/)） |
| `ANTHROPIC_TOKEN` | 手動またはレガシーのAnthropic OAuth/セットアップトークンの上書き |
| `DASHSCOPE_API_KEY` | Qwenモデル用のAlibaba Cloud DashScope APIキー（[modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)） |
| `DASHSCOPE_BASE_URL` | カスタムDashScopeベースURL（デフォルト: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`。中国本土リージョンでは `https://dashscope.aliyuncs.com/compatible-mode/v1` を使用） |
| `DEEPSEEK_API_KEY` | DeepSeekへ直接アクセスするためのDeepSeek APIキー（[platform.deepseek.com](https://platform.deepseek.com/api_keys)） |
| `DEEPSEEK_BASE_URL` | カスタムDeepSeek APIベースURL |
| `NVIDIA_API_KEY` | NVIDIA NIM APIキー — Nemotronおよびオープンモデル（[build.nvidia.com](https://build.nvidia.com)） |
| `NVIDIA_BASE_URL` | NVIDIAのベースURLを上書き（デフォルト: `https://integrate.api.nvidia.com/v1`。ローカルNIMエンドポイントの場合は `http://localhost:8000/v1` を設定） |
| `STEPFUN_API_KEY` | StepFun APIキー — Stepシリーズモデル（[platform.stepfun.com](https://platform.stepfun.com)） |
| `STEPFUN_BASE_URL` | StepFunのベースURLを上書き（デフォルト: `https://api.stepfun.com/v1`） |
| `OLLAMA_API_KEY` | Ollama Cloud APIキー — ローカルGPU不要のマネージドOllamaカタログ（[ollama.com/settings/keys](https://ollama.com/settings/keys)） |
| `OLLAMA_BASE_URL` | Ollama CloudのベースURLを上書き（デフォルト: `https://ollama.com/v1`） |
| `XAI_API_KEY` | チャット + TTS用のxAI（Grok）APIキー（[console.x.ai](https://console.x.ai/)） |
| `XAI_BASE_URL` | xAIのベースURLを上書き（デフォルト: `https://api.x.ai/v1`） |
| `MISTRAL_API_KEY` | Voxtral TTSおよびVoxtral STT用のMistral APIキー（[console.mistral.ai](https://console.mistral.ai)） |
| `AWS_REGION` | Bedrock推論用のAWSリージョン（例: `us-east-1`、`eu-central-1`）。boto3が読み取ります。 |
| `AWS_PROFILE` | Bedrock認証用のAWS名前付きプロファイル（`~/.aws/credentials` を読み取り）。未設定のままにするとデフォルトのboto3認証チェーンを使用します。 |
| `BEDROCK_BASE_URL` | Bedrockランタイムのベースを上書き（デフォルト: `https://bedrock-runtime.us-east-1.amazonaws.com`。通常は未設定のままにして `AWS_REGION` を使用） |
| `HERMES_QWEN_BASE_URL` | Qwen PortalのベースURLを上書き（デフォルト: `https://portal.qwen.ai/v1`） |
| `OPENCODE_ZEN_API_KEY` | OpenCode Zen APIキー — 厳選されたモデルへの従量課金アクセス（[opencode.ai](https://opencode.ai/auth)） |
| `OPENCODE_ZEN_BASE_URL` | OpenCode ZenのベースURLを上書き |
| `OPENCODE_GO_API_KEY` | OpenCode Go APIキー — オープンモデル向けの月額$10サブスクリプション（[opencode.ai](https://opencode.ai/auth)） |
| `OPENCODE_GO_BASE_URL` | OpenCode GoのベースURLを上書き |
| `CLAUDE_CODE_OAUTH_TOKEN` | 手動でエクスポートする場合の明示的なClaude Codeトークンの上書き |
| `HERMES_MODEL` | プロセスレベルでモデル名を上書き（cronスケジューラーで使用。通常の用途では `config.yaml` を推奨） |
| `VOICE_TOOLS_OPENAI_KEY` | OpenAIのspeech-to-textおよびtext-to-speechプロバイダー用の優先OpenAIキー |
| `HERMES_LOCAL_STT_COMMAND` | 任意のローカルspeech-to-textコマンドテンプレート。`{input_path}`、`{output_dir}`、`{language}`、`{model}` のプレースホルダーをサポート |
| `HERMES_LOCAL_STT_LANGUAGE` | `HERMES_LOCAL_STT_COMMAND` に渡されるデフォルト言語、または自動検出されるローカル `whisper` CLIフォールバック（デフォルト: `en`） |
| `HERMES_HOME` | Hermesの設定ディレクトリを上書き（デフォルト: `~/.hermes`）。ゲートウェイのPIDファイルとsystemdサービス名のスコープも設定するため、複数のインストールを同時実行できます |
| `HERMES_GIT_BASH_PATH` | **Windowsのみ。** ターミナルツール用の `bash.exe` 検出を上書き。任意のbashを指定可能 — Git-for-Windowsのフルインストール、シンボリックリンク経由のWSL bash、MSYS2、Cygwin。インストーラーはプロビジョニングしたPortableGitを自動的にこれに設定します。[Windows（ネイティブ）ガイド](../user-guide/windows-native.md#how-hermes-runs-shell-commands-on-windows)を参照してください |
| `HERMES_DISABLE_WINDOWS_UTF8` | **Windowsのみ。** `1` に設定するとUTF-8 stdioシム（`configure_windows_stdio()`）を無効化し、コンソールのロケールコードページにフォールバックします。エンコーディングバグの二分探索に便利ですが、通常の運用で適切な設定になることはまれです |
| `HERMES_KANBAN_HOME` | かんばんボード（DB + ワークスペース + ワーカーログ）の起点となる共有Hermesルートを上書き。`get_default_hermes_root()`（アクティブなプロファイルの親）にフォールバックします。テストや特殊なデプロイに便利です |
| `HERMES_KANBAN_BOARD` | このプロセスのアクティブなかんばんボードを固定。`~/.hermes/kanban/current` より優先されます。ディスパッチャーがこれをワーカーのサブプロセス環境に注入するため、ワーカーは他のボードのタスクを物理的に参照できません。デフォルトは `default`。スラッグ検証: 小文字の英数字 + ハイフン + アンダースコア、1〜64文字 |
| `HERMES_KANBAN_DB` | かんばんデータベースファイルのパスを直接固定（最優先。`HERMES_KANBAN_BOARD` と `HERMES_KANBAN_HOME` に優先）。ディスパッチャーがこれをワーカーのサブプロセス環境に注入するため、プロファイルワーカーはディスパッチャーのボードに収束します |
| `HERMES_KANBAN_WORKSPACES_ROOT` | かんばんワークスペースルートを直接固定（ワークスペースについては最優先。`HERMES_KANBAN_HOME` に優先）。ディスパッチャーがこれをワーカーのサブプロセス環境に注入します |

## プロバイダー認証（OAuth）

ネイティブのAnthropic認証では、Hermesは存在する場合にClaude Code自身の認証情報ファイルを優先します。これらの認証情報は自動的に更新できるためです。**AnthropicへのOAuthには、追加利用クレジットを購入したClaude Maxプランが必要です** — HermesはClaude Codeとしてルーティングし、Maxプランの基本割り当てではなく追加・超過クレジットからのみ消費し、Claude Proでは動作しません。Max + 追加クレジットがない場合は、代わりにAPIキーを使用してください。`ANTHROPIC_TOKEN` などの環境変数は手動の上書きとして引き続き有用ですが、Claude Maxログインの推奨パスではなくなりました。

| Variable | Description |
|----------|-------------|
| `HERMES_INFERENCE_PROVIDER` | プロバイダー選択を上書き: `auto`、`custom`、`openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`huggingface`、`gemini`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`（ブラウザOAuthログイン — APIキー不要。[MiniMax OAuthガイド](../guides/minimax-oauth.md)を参照）、`kilocode`、`xiaomi`、`arcee`、`gmi`、`stepfun`、`alibaba`、`alibaba-coding-plan`（エイリアス `alibaba_coding`）、`deepseek`、`nvidia`、`ollama-cloud`、`xai`（エイリアス `grok`）、`google-gemini-cli`、`qwen-oauth`、`bedrock`、`opencode-zen`、`opencode-go`、`ai-gateway`、`tencent-tokenhub`（デフォルト: `auto`） |
| `HERMES_PORTAL_BASE_URL` | Nous PortalのURLを上書き（開発・テスト用） |
| `NOUS_INFERENCE_BASE_URL` | Nous推論APIのURLを上書き |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | 再発行までのエージェントキーの最小TTL（デフォルト: 1800 = 30分） |
| `HERMES_NOUS_TIMEOUT_SECONDS` | Nous認証情報 / トークンフローのHTTPタイムアウト |
| `HERMES_DUMP_REQUESTS` | APIリクエストのペイロードをログファイルにダンプ（`true`/`false`） |
| `HERMES_PREFILL_MESSAGES_FILE` | API呼び出し時に注入されるエフェメラルなプレフィルメッセージのJSONファイルへのパス |
| `HERMES_TIMEZONE` | IANAタイムゾーンの上書き（例: `America/New_York`） |

## ツールAPI

| Variable | Description |
|----------|-------------|
| `PARALLEL_API_KEY` | AIネイティブWeb検索（[parallel.ai](https://parallel.ai/)） |
| `FIRECRAWL_API_KEY` | Webスクレイピングおよびクラウドブラウザ（[firecrawl.dev](https://firecrawl.dev/)） |
| `FIRECRAWL_API_URL` | セルフホストインスタンス用のカスタムFirecrawl APIエンドポイント（任意） |
| `TAVILY_API_KEY` | AIネイティブWeb検索・抽出・クロール用のTavily APIキー（[app.tavily.com](https://app.tavily.com/home)） |
| `SEARXNG_URL` | 無料のセルフホストWeb検索用SearXNGインスタンスのURL — APIキー不要（[searxng.github.io](https://searxng.github.io/searxng/)） |
| `TAVILY_BASE_URL` | Tavily APIエンドポイントを上書き。企業プロキシやセルフホストのTavily互換検索バックエンドに便利です。`GROQ_BASE_URL` と同じパターンです。 |
| `EXA_API_KEY` | AIネイティブWeb検索とコンテンツ用のExa APIキー（[exa.ai](https://exa.ai/)） |
| `BROWSERBASE_API_KEY` | ブラウザ自動化（[browserbase.com](https://browserbase.com/)） |
| `BROWSERBASE_PROJECT_ID` | BrowserbaseプロジェクトID |
| `BROWSER_USE_API_KEY` | Browser Useクラウドブラウザ APIキー（[browser-use.com](https://browser-use.com/)） |
| `FIRECRAWL_BROWSER_TTL` | FirecrawlブラウザセッションのTTL（秒単位、デフォルト: 300） |
| `BROWSER_CDP_URL` | ローカルブラウザ用のChrome DevTools Protocol URL（`/browser connect` で設定、例: `ws://localhost:9222`） |
| `CAMOFOX_URL` | Camofoxローカル検出回避ブラウザのURL（デフォルト: `http://localhost:9377`） |
| `BROWSER_INACTIVITY_TIMEOUT` | ブラウザセッションの非アクティブタイムアウト（秒単位） |
| `FAL_KEY` | 画像生成（[fal.ai](https://fal.ai/)） |
| `GROQ_API_KEY` | Groq Whisper STT APIキー（[groq.com](https://groq.com/)） |
| `ELEVENLABS_API_KEY` | ElevenLabsのプレミアムTTSボイス（[elevenlabs.io](https://elevenlabs.io/)） |
| `STT_GROQ_MODEL` | Groq STTモデルを上書き（デフォルト: `whisper-large-v3-turbo`） |
| `GROQ_BASE_URL` | GroqのOpenAI互換STTエンドポイントを上書き |
| `STT_OPENAI_MODEL` | OpenAI STTモデルを上書き（デフォルト: `whisper-1`） |
| `STT_OPENAI_BASE_URL` | OpenAI互換STTエンドポイントを上書き |
| `GITHUB_TOKEN` | Skills Hub用のGitHubトークン（より高いAPIレート制限、スキルの公開） |
| `HONCHO_API_KEY` | セッションをまたぐユーザーモデリング（[honcho.dev](https://honcho.dev/)） |
| `HONCHO_BASE_URL` | セルフホストHonchoインスタンス用のベースURL（デフォルト: Honchoクラウド）。ローカルインスタンスではAPIキー不要 |
| `HINDSIGHT_TIMEOUT` | HindsightメモリプロバイダーへのAPI呼び出しのタイムアウト（秒単位、デフォルト: `60`）。`/sync` や `on_session_switch` の際にHindsightインスタンスの応答が遅く、`errors.log` にタイムアウトが出る場合は増やしてください。 |
| `SUPERMEMORY_API_KEY` | プロファイルの想起とセッション取り込みを備えたセマンティック長期メモリ（[supermemory.ai](https://supermemory.ai)） |
| `TINKER_API_KEY` | RLトレーニング（[tinker-console.thinkingmachines.ai](https://tinker-console.thinkingmachines.ai/)） |
| `WANDB_API_KEY` | RLトレーニングのメトリクス（[wandb.ai](https://wandb.ai/)） |
| `DAYTONA_API_KEY` | Daytonaクラウドサンドボックス（[daytona.io](https://daytona.io/)） |
| `VERCEL_TOKEN` | Vercel Sandboxアクセストークン（[vercel.com](https://vercel.com/)） |
| `VERCEL_PROJECT_ID` | VercelプロジェクトID（`VERCEL_TOKEN` と併用必須） |
| `VERCEL_TEAM_ID` | VercelチームID（`VERCEL_TOKEN` と併用必須） |
| `VERCEL_OIDC_TOKEN` | Vercelの短命OIDCトークン（開発専用の代替） |

### Langfuse可観測性

バンドルされた [`observability/langfuse`](/docs/user-guide/features/built-in-plugins#observabilitylangfuse) プラグイン用の環境変数です。`hermes tools → Langfuse Observability` から、または `~/.hermes/.env` で手動で設定します。これらを有効にするには、事前にプラグインを有効化（`hermes plugins enable observability/langfuse`）しておく必要があります。

| Variable | Description |
|----------|-------------|
| `HERMES_LANGFUSE_PUBLIC_KEY` | Langfuseプロジェクトの公開キー（`pk-lf-...`）。必須。 |
| `HERMES_LANGFUSE_SECRET_KEY` | Langfuseプロジェクトのシークレットキー（`sk-lf-...`）。必須。 |
| `HERMES_LANGFUSE_BASE_URL` | LangfuseサーバーのURL（デフォルト: `https://cloud.langfuse.com`）。セルフホスト時に設定します。 |
| `HERMES_LANGFUSE_ENV` | トレースの環境タグ（`production`、`staging` など） |
| `HERMES_LANGFUSE_RELEASE` | トレースのリリース/バージョンタグ |
| `HERMES_LANGFUSE_SAMPLE_RATE` | SDKのサンプリングレート 0.0〜1.0（デフォルト: `1.0`） |
| `HERMES_LANGFUSE_MAX_CHARS` | シリアライズされたペイロードのフィールドごとの切り詰め（デフォルト: `12000`） |
| `HERMES_LANGFUSE_DEBUG` | `true` で `agent.log` への詳細なプラグインログを有効化 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | 標準のLangfuse SDK名。`HERMES_LANGFUSE_*` 相当が未設定の場合にフォールバックとして受け入れられます。 |

### Nous Tool Gateway

これらの変数は、有料Nousサブスクライバーまたはセルフホストのゲートウェイデプロイ向けに [Tool Gateway](/docs/user-guide/features/tool-gateway) を設定します。ほとんどのユーザーはこれらを設定する必要はありません — ゲートウェイは `hermes model` や `hermes tools` を通じて自動的に設定されます。

| Variable | Description |
|----------|-------------|
| `TOOL_GATEWAY_DOMAIN` | Tool Gatewayルーティングのベースドメイン（デフォルト: `nousresearch.com`） |
| `TOOL_GATEWAY_SCHEME` | ゲートウェイURLのHTTPまたはHTTPSスキーム（デフォルト: `https`） |
| `TOOL_GATEWAY_USER_TOKEN` | Tool Gatewayの認証トークン（通常はNous認証から自動入力） |
| `FIRECRAWL_GATEWAY_URL` | Firecrawlゲートウェイエンドポイント専用のURLを上書き |

## ターミナルバックエンド

| Variable | Description |
|----------|-------------|
| `TERMINAL_ENV` | バックエンド: `local`、`docker`、`ssh`、`singularity`、`modal`、`daytona`、`vercel_sandbox` |
| `HERMES_DOCKER_BINARY` | Hermesがシェルアウトするコンテナバイナリを上書き（例: `podman`、`/usr/local/bin/docker`）。未設定の場合、Hermesは `PATH` 上の `docker` または `podman` を自動検出します。両方インストールされていてデフォルト以外を使いたい場合や、バイナリが `PATH` の外にある場合に必要です。 |
| `TERMINAL_DOCKER_IMAGE` | Dockerイメージ（デフォルト: `nikolaik/python-nodejs:python3.11-nodejs20`） |
| `TERMINAL_DOCKER_FORWARD_ENV` | Dockerターミナルセッションに明示的に転送する環境変数名のJSON配列。注: スキルが宣言した `required_environment_variables` は自動的に転送されます — どのスキルでも宣言されていない変数の場合のみ必要です。 |
| `TERMINAL_DOCKER_VOLUMES` | 追加のDockerボリュームマウント（カンマ区切りの `host:container` ペア） |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | 高度なオプトイン: 起動時のcwdをDockerの `/workspace` にマウント（`true`/`false`、デフォルト: `false`） |
| `TERMINAL_SINGULARITY_IMAGE` | Singularityイメージまたは `.sif` パス |
| `TERMINAL_MODAL_IMAGE` | Modalコンテナイメージ |
| `TERMINAL_DAYTONA_IMAGE` | Daytonaサンドボックスイメージ |
| `TERMINAL_VERCEL_RUNTIME` | Vercel Sandboxランタイム（`node24`、`node22`、`python3.13`） |
| `TERMINAL_TIMEOUT` | コマンドのタイムアウト（秒単位） |
| `TERMINAL_LIFETIME_SECONDS` | ターミナルセッションの最大存続期間（秒単位） |
| `TERMINAL_CWD` | ターミナルセッションの作業ディレクトリ（ゲートウェイ/cronのみ。CLIは起動ディレクトリを使用） |
| `SUDO_PASSWORD` | 対話的プロンプトなしでsudoを有効化 |

クラウドサンドボックスバックエンドの場合、永続性はファイルシステム指向です。`TERMINAL_LIFETIME_SECONDS` はHermesがアイドル状態のターミナルセッションをクリーンアップするタイミングを制御し、後で再開する際は同じライブプロセスを実行し続けるのではなく、サンドボックスを再作成する場合があります。

## SSHバックエンド

| Variable | Description |
|----------|-------------|
| `TERMINAL_SSH_HOST` | リモートサーバーのホスト名 |
| `TERMINAL_SSH_USER` | SSHユーザー名 |
| `TERMINAL_SSH_PORT` | SSHポート（デフォルト: 22） |
| `TERMINAL_SSH_KEY` | 秘密鍵へのパス |
| `TERMINAL_SSH_PERSISTENT` | SSH用の永続シェルを上書き（デフォルト: `TERMINAL_PERSISTENT_SHELL` に従う） |

## コンテナリソース（Docker、Singularity、Modal、Daytona）

| Variable | Description |
|----------|-------------|
| `TERMINAL_CONTAINER_CPU` | CPUコア数（デフォルト: 1） |
| `TERMINAL_CONTAINER_MEMORY` | メモリ（MB単位、デフォルト: 5120） |
| `TERMINAL_CONTAINER_DISK` | ディスク（MB単位、デフォルト: 51200） |
| `TERMINAL_CONTAINER_PERSISTENT` | セッションをまたいでコンテナのファイルシステムを永続化（デフォルト: `true`） |
| `TERMINAL_SANDBOX_DIR` | ワークスペースとオーバーレイ用のホストディレクトリ（デフォルト: `~/.hermes/sandboxes/`） |

## 永続シェル

| Variable | Description |
|----------|-------------|
| `TERMINAL_PERSISTENT_SHELL` | ローカル以外のバックエンドで永続シェルを有効化（デフォルト: `true`）。config.yamlの `terminal.persistent_shell` でも設定可能 |
| `TERMINAL_LOCAL_PERSISTENT` | ローカルバックエンドで永続シェルを有効化（デフォルト: `false`） |
| `TERMINAL_SSH_PERSISTENT` | SSHバックエンド用の永続シェルを上書き（デフォルト: `TERMINAL_PERSISTENT_SHELL` に従う） |

## メッセージング

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegramボットトークン（@BotFatherから取得） |
| `TELEGRAM_ALLOWED_USERS` | ボットの使用を許可するユーザーIDのカンマ区切りリスト（DM、グループ、フォーラムに適用） |
| `TELEGRAM_GROUP_ALLOWED_USERS` | グループ/フォーラムでのみ認可される送信者ユーザーIDのカンマ区切りリスト（DMアクセスは付与しません）。チャットIDの形式の値（`-` で始まる）は、#17686以前の設定との後方互換性のため引き続きチャットIDとして扱われますが、非推奨の警告が表示されます。 |
| `TELEGRAM_GROUP_ALLOWED_CHATS` | グループ/フォーラムのチャットIDのカンマ区切りリスト。メンバー全員が認可されます |
| `TELEGRAM_HOME_CHANNEL` | cron配信用のデフォルトTelegramチャット/チャンネル |
| `TELEGRAM_HOME_CHANNEL_NAME` | Telegramホームチャンネルの表示名 |
| `TELEGRAM_WEBHOOK_URL` | Webhookモード用の公開HTTPS URL（ポーリングの代わりにWebhookを有効化） |
| `TELEGRAM_WEBHOOK_PORT` | Webhookサーバーのローカルリッスンポート（デフォルト: `8443`） |
| `TELEGRAM_WEBHOOK_SECRET` | 検証のためにTelegramが各更新で返すシークレットトークン。**`TELEGRAM_WEBHOOK_URL` が設定されている場合は常に必須** — これがないとゲートウェイは起動を拒否します（GHSA-3vpc-7q5r-276h）。`openssl rand -hex 32` で生成します。 |
| `TELEGRAM_REACTIONS` | 処理中のメッセージへの絵文字リアクションを有効化（デフォルト: `false`） |
| `TELEGRAM_REPLY_TO_MODE` | 返信参照の動作: `off`、`first`（デフォルト）、または `all`。Discordのパターンに合わせています。 |
| `TELEGRAM_IGNORED_THREADS` | ボットが応答しないTelegramフォーラムトピック/スレッドIDのカンマ区切りリスト |
| `TELEGRAM_PROXY` | Telegram接続用のプロキシURL — `HTTPS_PROXY` を上書き。`http://`、`https://`、`socks5://` をサポート |
| `DISCORD_BOT_TOKEN` | Discordボットトークン |
| `DISCORD_ALLOWED_USERS` | ボットの使用を許可するDiscordユーザーIDのカンマ区切りリスト |
| `DISCORD_ALLOWED_ROLES` | ボットの使用を許可するDiscordロールIDのカンマ区切りリスト（`DISCORD_ALLOWED_USERS` とのOR）。Membersインテントを自動的に有効化します。モデレーションチームの入れ替わりがある場合に便利です — ロールの付与は自動的に反映されます。 |
| `DISCORD_ALLOWED_CHANNELS` | DiscordチャンネルIDのカンマ区切りリスト。設定すると、ボットはこれらのチャンネル（許可されていればDMも）でのみ応答します。`config.yaml` の `discord.allowed_channels` を上書きします。 |
| `DISCORD_PROXY` | Discord接続用のプロキシURL — `HTTPS_PROXY` を上書き。`http://`、`https://`、`socks5://` をサポート |
| `DISCORD_HOME_CHANNEL` | cron配信用のデフォルトDiscordチャンネル |
| `DISCORD_HOME_CHANNEL_NAME` | Discordホームチャンネルの表示名 |
| `DISCORD_COMMAND_SYNC_POLICY` | Discordスラッシュコマンドの起動時同期ポリシー: `safe`（差分を取って調整）、`bulk`（レガシーの `tree.sync()`）、または `off` |
| `DISCORD_REQUIRE_MENTION` | サーバーチャンネルで応答する前に@メンションを必須にする |
| `DISCORD_FREE_RESPONSE_CHANNELS` | メンションが不要なチャンネルIDのカンマ区切りリスト |
| `DISCORD_AUTO_THREAD` | サポートされている場合に長い返信を自動的にスレッド化 |
| `DISCORD_REACTIONS` | 処理中のメッセージへの絵文字リアクションを有効化（デフォルト: `true`） |
| `DISCORD_IGNORED_CHANNELS` | ボットが応答しないチャンネルIDのカンマ区切りリスト |
| `DISCORD_NO_THREAD_CHANNELS` | ボットが自動スレッド化せずに応答するチャンネルIDのカンマ区切りリスト |
| `DISCORD_REPLY_TO_MODE` | 返信参照の動作: `off`、`first`（デフォルト）、または `all` |
| `DISCORD_ALLOW_MENTION_EVERYONE` | ボットが `@everyone`/`@here` にpingすることを許可（デフォルト: `false`）。[メンション制御](../user-guide/messaging/discord.md#mention-control)を参照してください。 |
| `DISCORD_ALLOW_MENTION_ROLES` | ボットが `@role` メンションにpingすることを許可（デフォルト: `false`）。 |
| `DISCORD_ALLOW_MENTION_USERS` | ボットが個別の `@user` メンションにpingすることを許可（デフォルト: `true`）。 |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | メッセージに返信する際に作成者にpingする（デフォルト: `true`）。 |
| `SLACK_BOT_TOKEN` | Slackボットトークン（`xoxb-...`） |
| `SLACK_APP_TOKEN` | Slackアプリレベルトークン（`xapp-...`、Socket Modeに必要） |
| `SLACK_ALLOWED_USERS` | SlackユーザーIDのカンマ区切りリスト |
| `SLACK_HOME_CHANNEL` | cron配信用のデフォルトSlackチャンネル |
| `SLACK_HOME_CHANNEL_NAME` | Slackホームチャンネルの表示名 |
| `GOOGLE_CHAT_PROJECT_ID` | Pub/Subトピックをホストするのに使われるGCPプロジェクト（`GOOGLE_CLOUD_PROJECT` にフォールバック） |
| `GOOGLE_CHAT_SUBSCRIPTION_NAME` | Pub/Subサブスクリプションの完全パス、`projects/{proj}/subscriptions/{sub}`（レガシーエイリアス: `GOOGLE_CHAT_SUBSCRIPTION`） |
| `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` | Service Account JSONへのパス、またはJSONをインラインで指定（`GOOGLE_APPLICATION_CREDENTIALS` にフォールバック） |
| `GOOGLE_CHAT_ALLOWED_USERS` | ボットとのチャットを許可するユーザーメールのカンマ区切りリスト |
| `GOOGLE_CHAT_ALLOW_ALL_USERS` | 任意のGoogle Chatユーザーがボットをトリガーすることを許可（開発専用） |
| `GOOGLE_CHAT_HOME_CHANNEL` | cron配信用のデフォルトスペース（例: `spaces/AAAA...`） |
| `GOOGLE_CHAT_HOME_CHANNEL_NAME` | Google Chatホームスペースの表示名 |
| `GOOGLE_CHAT_MAX_MESSAGES` | Pub/Sub FlowControlの最大処理中メッセージ数（デフォルト: `1`） |
| `GOOGLE_CHAT_MAX_BYTES` | Pub/Sub FlowControlの最大処理中バイト数（デフォルト: `16777216`、16 MiB） |
| `GOOGLE_CHAT_BOOTSTRAP_SPACES` | ボット自身の `users/{id}` を解決する際に起動時にプローブする追加スペースIDのカンマ区切りリスト |
| `GOOGLE_CHAT_DEBUG_RAW` | 任意の値を設定すると、機密情報を伏せたPub/SubエンベロープをDEBUGレベルでログ出力（デバッグ用のみ） |
| `WHATSAPP_ENABLED` | WhatsAppブリッジを有効化（`true`/`false`） |
| `WHATSAPP_MODE` | `bot`（別番号）または `self-chat`（自分自身にメッセージ） |
| `WHATSAPP_ALLOWED_USERS` | 電話番号のカンマ区切りリスト（国番号付き、`+` なし）、またはすべての送信者を許可する `*` |
| `WHATSAPP_ALLOW_ALL_USERS` | 許可リストなしですべてのWhatsApp送信者を許可（`true`/`false`） |
| `WHATSAPP_DEBUG` | トラブルシューティング用にブリッジで生のメッセージイベントをログ出力（`true`/`false`） |
| `SIGNAL_HTTP_URL` | signal-cliデーモンのHTTPエンドポイント（例: `http://127.0.0.1:8080`） |
| `SIGNAL_ACCOUNT` | E.164形式のボット電話番号 |
| `SIGNAL_ALLOWED_USERS` | E.164電話番号またはUUIDのカンマ区切りリスト |
| `SIGNAL_GROUP_ALLOWED_USERS` | グループIDのカンマ区切りリスト、またはすべてのグループ用の `*` |
| `SIGNAL_HOME_CHANNEL_NAME` | Signalホームチャンネルの表示名 |
| `SIGNAL_IGNORE_STORIES` | Signalのストーリー/ステータス更新を無視 |
| `SIGNAL_ALLOW_ALL_USERS` | 許可リストなしですべてのSignalユーザーを許可 |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID（テレフォニースキルと共有） |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token（テレフォニースキルと共有。Webhook署名検証にも使用） |
| `TWILIO_PHONE_NUMBER` | E.164形式のTwilio電話番号（テレフォニースキルと共有） |
| `SMS_WEBHOOK_URL` | Twilio署名検証用の公開URL — Twilio ConsoleのWebhook URLと一致する必要があります（必須） |
| `SMS_WEBHOOK_PORT` | 受信SMS用のWebhookリスナーポート（デフォルト: `8080`） |
| `SMS_WEBHOOK_HOST` | Webhookのバインドアドレス（デフォルト: `0.0.0.0`） |
| `SMS_INSECURE_NO_SIGNATURE` | `true` に設定するとTwilio署名検証を無効化（ローカル開発のみ — 本番環境では使用しないでください） |
| `SMS_ALLOWED_USERS` | チャットを許可するE.164電話番号のカンマ区切りリスト |
| `SMS_ALLOW_ALL_USERS` | 許可リストなしですべてのSMS送信者を許可 |
| `SMS_HOME_CHANNEL` | cronジョブ / 通知配信用の電話番号 |
| `SMS_HOME_CHANNEL_NAME` | SMSホームチャンネルの表示名 |
| `EMAIL_ADDRESS` | Emailゲートウェイアダプター用のメールアドレス |
| `EMAIL_PASSWORD` | メールアカウントのパスワードまたはアプリパスワード |
| `EMAIL_IMAP_HOST` | メールアダプター用のIMAPホスト名 |
| `EMAIL_IMAP_PORT` | IMAPポート |
| `EMAIL_SMTP_HOST` | メールアダプター用のSMTPホスト名 |
| `EMAIL_SMTP_PORT` | SMTPポート |
| `EMAIL_ALLOWED_USERS` | ボットへのメッセージを許可するメールアドレスのカンマ区切りリスト |
| `EMAIL_HOME_ADDRESS` | プロアクティブなメール配信のデフォルト受信者 |
| `EMAIL_HOME_ADDRESS_NAME` | メールホームターゲットの表示名 |
| `EMAIL_POLL_INTERVAL` | メールのポーリング間隔（秒単位） |
| `EMAIL_ALLOW_ALL_USERS` | すべての受信メール送信者を許可 |
| `DINGTALK_CLIENT_ID` | デベロッパーポータルから取得するDingTalkボットのAppKey（[open.dingtalk.com](https://open.dingtalk.com)） |
| `DINGTALK_CLIENT_SECRET` | デベロッパーポータルから取得するDingTalkボットのAppSecret |
| `DINGTALK_ALLOWED_USERS` | ボットへのメッセージを許可するDingTalkユーザーIDのカンマ区切りリスト |
| `FEISHU_APP_ID` | [open.feishu.cn](https://open.feishu.cn/) から取得するFeishu/LarkボットのApp ID |
| `FEISHU_APP_SECRET` | Feishu/LarkボットのApp Secret |
| `FEISHU_DOMAIN` | `feishu`（中国）または `lark`（国際版）。デフォルト: `feishu` |
| `FEISHU_CONNECTION_MODE` | `websocket`（推奨）または `webhook`。デフォルト: `websocket` |
| `FEISHU_ENCRYPT_KEY` | Webhookモード用の任意の暗号化キー |
| `FEISHU_VERIFICATION_TOKEN` | Webhookモード用の任意の検証トークン |
| `FEISHU_ALLOWED_USERS` | ボットへのメッセージを許可するFeishuユーザーIDのカンマ区切りリスト |
| `FEISHU_ALLOW_BOTS` | `none`（デフォルト）/ `mentions` / `all` — 他のボットからの受信メッセージを受け入れる。[ボット間メッセージング](../user-guide/messaging/feishu.md#bot-to-bot-messaging)を参照してください |
| `FEISHU_REQUIRE_MENTION` | `true`（デフォルト）/ `false` — グループメッセージでボットへの@メンションを必須にするか。`group_rules.<chat_id>.require_mention` でチャットごとに上書きできます。 |
| `FEISHU_HOME_CHANNEL` | cron配信と通知用のFeishuチャットID |
| `WECOM_BOT_ID` | 管理コンソールから取得するWeCom AI BotのID |
| `WECOM_SECRET` | WeCom AI Botのシークレット |
| `WECOM_WEBSOCKET_URL` | カスタムWebSocket URL（デフォルト: `wss://openws.work.weixin.qq.com`） |
| `WECOM_ALLOWED_USERS` | ボットへのメッセージを許可するWeComユーザーIDのカンマ区切りリスト |
| `WECOM_HOME_CHANNEL` | cron配信と通知用のWeComチャットID |
| `WECOM_CALLBACK_CORP_ID` | 自社開発アプリのコールバック用WeCom企業Corp ID |
| `WECOM_CALLBACK_CORP_SECRET` | 自社開発アプリのCorpシークレット |
| `WECOM_CALLBACK_AGENT_ID` | 自社開発アプリのAgent ID |
| `WECOM_CALLBACK_TOKEN` | コールバック検証トークン |
| `WECOM_CALLBACK_ENCODING_AES_KEY` | コールバック暗号化用のAESキー |
| `WECOM_CALLBACK_HOST` | コールバックサーバーのバインドアドレス（デフォルト: `0.0.0.0`） |
| `WECOM_CALLBACK_PORT` | コールバックサーバーのポート（デフォルト: `8645`） |
| `WECOM_CALLBACK_ALLOWED_USERS` | 許可リスト用のユーザーIDのカンマ区切りリスト |
| `WECOM_CALLBACK_ALLOW_ALL_USERS` | `true` に設定すると許可リストなしですべてのユーザーを許可 |
| `WEIXIN_ACCOUNT_ID` | iLink Bot API経由のQRログインで取得するWeixinアカウントID |
| `WEIXIN_TOKEN` | iLink Bot API経由のQRログインで取得するWeixin認証トークン |
| `WEIXIN_BASE_URL` | Weixin iLink Bot APIのベースURLを上書き（デフォルト: `https://ilinkai.weixin.qq.com`） |
| `WEIXIN_CDN_BASE_URL` | メディア用のWeixin CDNベースURLを上書き（デフォルト: `https://novac2c.cdn.weixin.qq.com/c2c`） |
| `WEIXIN_DM_POLICY` | ダイレクトメッセージポリシー: `open`、`allowlist`、`pairing`、`disabled`（デフォルト: `open`） |
| `WEIXIN_GROUP_POLICY` | グループメッセージポリシー: `open`、`allowlist`、`disabled`（デフォルト: `disabled`） |
| `WEIXIN_ALLOWED_USERS` | ボットへのDMを許可するWeixinユーザーIDのカンマ区切りリスト |
| `WEIXIN_GROUP_ALLOWED_USERS` | ボットとの対話を許可するWeixinの**グループチャットID**（メンバーのユーザーIDではない）のカンマ区切りリスト。変数名はレガシーで、グループIDを想定しています。iLinkが実際にグループイベントを配信する場合にのみ有効です。QRログインのiLinkボットID（`...@im.bot`）は通常、一般的なWeChatグループメッセージを受信しません。 |
| `WEIXIN_HOME_CHANNEL` | cron配信と通知用のWeixinチャットID |
| `WEIXIN_HOME_CHANNEL_NAME` | Weixinホームチャンネルの表示名 |
| `WEIXIN_ALLOW_ALL_USERS` | 許可リストなしですべてのWeixinユーザーを許可（`true`/`false`） |
| `BLUEBUBBLES_SERVER_URL` | BlueBubblesサーバーのURL（例: `http://192.168.1.10:1234`） |
| `BLUEBUBBLES_PASSWORD` | BlueBubblesサーバーのパスワード |
| `BLUEBUBBLES_WEBHOOK_HOST` | Webhookリスナーのバインドアドレス（デフォルト: `127.0.0.1`） |
| `BLUEBUBBLES_WEBHOOK_PORT` | Webhookリスナーのポート（デフォルト: `8645`） |
| `BLUEBUBBLES_HOME_CHANNEL` | cron/通知配信用の電話番号/メール |
| `BLUEBUBBLES_ALLOWED_USERS` | 認可されたユーザーのカンマ区切りリスト |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | すべてのユーザーを許可（`true`/`false`） |
| `QQ_APP_ID` | [q.qq.com](https://q.qq.com) から取得するQQ BotのApp ID |
| `QQ_CLIENT_SECRET` | [q.qq.com](https://q.qq.com) から取得するQQ BotのApp Secret |
| `QQ_STT_API_KEY` | 外部STTフォールバックプロバイダー用のAPIキー（任意。QQの組み込みASRがテキストを返さない場合に使用） |
| `QQ_STT_BASE_URL` | 外部STTプロバイダーのベースURL（任意） |
| `QQ_STT_MODEL` | 外部STTプロバイダーのモデル名（任意） |
| `QQ_ALLOWED_USERS` | ボットへのメッセージを許可するQQユーザーのopenIDのカンマ区切りリスト |
| `QQ_GROUP_ALLOWED_USERS` | グループ@メッセージアクセス用のQQグループIDのカンマ区切りリスト |
| `QQ_ALLOW_ALL_USERS` | すべてのユーザーを許可（`true`/`false`、`QQ_ALLOWED_USERS` を上書き） |
| `QQBOT_HOME_CHANNEL` | cron配信と通知用のQQユーザー/グループのopenID |
| `QQBOT_HOME_CHANNEL_NAME` | QQホームチャンネルの表示名 |
| `QQ_PORTAL_HOST` | QQポータルホストを上書き（`sandbox.q.qq.com` に設定するとサンドボックスゲートウェイ経由でルーティング。デフォルト: `q.qq.com`）。 |
| `MATTERMOST_URL` | MattermostサーバーのURL（例: `https://mm.example.com`） |
| `MATTERMOST_TOKEN` | Mattermost用のボットトークンまたは個人アクセストークン |
| `MATTERMOST_ALLOWED_USERS` | ボットへのメッセージを許可するMattermostユーザーIDのカンマ区切りリスト |
| `MATTERMOST_HOME_CHANNEL` | プロアクティブなメッセージ配信（cron、通知）用のチャンネルID |
| `MATTERMOST_REQUIRE_MENTION` | チャンネルで `@mention` を必須にする（デフォルト: `true`）。`false` に設定するとすべてのメッセージに応答します。 |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | ボットが `@mention` なしで応答するチャンネルIDのカンマ区切りリスト |
| `MATTERMOST_REPLY_MODE` | 返信スタイル: `thread`（スレッド化された返信）または `off`（フラットなメッセージ、デフォルト） |
| `MATRIX_HOMESERVER` | MatrixホームサーバーのURL（例: `https://matrix.org`） |
| `MATRIX_ACCESS_TOKEN` | ボット認証用のMatrixアクセストークン |
| `MATRIX_USER_ID` | MatrixユーザーID（例: `@hermes:matrix.org`） — パスワードログインには必須、アクセストークンを使う場合は任意 |
| `MATRIX_PASSWORD` | Matrixパスワード（アクセストークンの代替） |
| `MATRIX_ALLOWED_USERS` | ボットへのメッセージを許可するMatrixユーザーIDのカンマ区切りリスト（例: `@alice:matrix.org`） |
| `MATRIX_HOME_ROOM` | プロアクティブなメッセージ配信用のルームID（例: `!abc123:matrix.org`） |
| `MATRIX_ENCRYPTION` | エンドツーエンド暗号化を有効化（`true`/`false`、デフォルト: `false`） |
| `MATRIX_DEVICE_ID` | 再起動をまたいだE2EE永続化用の安定したMatrixデバイスID（例: `HERMES_BOT`）。これがないと、E2EEキーは起動のたびにローテーションされ、過去のルームの復号が壊れます。 |
| `MATRIX_REACTIONS` | 受信メッセージへの処理ライフサイクル絵文字リアクションを有効化（デフォルト: `true`）。`false` に設定すると無効化します。 |
| `MATRIX_REQUIRE_MENTION` | ルームで `@mention` を必須にする（デフォルト: `true`）。`false` に設定するとすべてのメッセージに応答します。 |
| `MATRIX_FREE_RESPONSE_ROOMS` | ボットが `@mention` なしで応答するルームIDのカンマ区切りリスト |
| `MATRIX_AUTO_THREAD` | ルームメッセージ用のスレッドを自動作成（デフォルト: `true`） |
| `MATRIX_DM_MENTION_THREADS` | DMでボットが `@mention` された際にスレッドを作成（デフォルト: `false`） |
| `MATRIX_RECOVERY_KEY` | デバイスキーのローテーション後のクロス署名検証用のリカバリーキー。クロス署名を有効にしたE2EE設定で推奨されます。 |
| `HASS_TOKEN` | Home Assistantの長期アクセストークン（HAプラットフォーム + ツールを有効化） |
| `HASS_URL` | Home AssistantのURL（デフォルト: `http://homeassistant.local:8123`） |
| `WEBHOOK_ENABLED` | Webhookプラットフォームアダプターを有効化（`true`/`false`） |
| `WEBHOOK_PORT` | Webhookを受信するHTTPサーバーのポート（デフォルト: `8644`） |
| `WEBHOOK_SECRET` | Webhook署名検証用のグローバルHMACシークレット（ルートが独自のものを指定しない場合のフォールバックとして使用） |
| `API_SERVER_ENABLED` | OpenAI互換APIサーバーを有効化（`true`/`false`）。他のプラットフォームと並行して動作します。 |
| `API_SERVER_KEY` | APIサーバー認証用のBearerトークン。非ループバックバインディングでは強制されます。 |
| `API_SERVER_CORS_ORIGINS` | APIサーバーを直接呼び出すことを許可するブラウザオリジンのカンマ区切りリスト（例: `http://localhost:3000,http://127.0.0.1:3000`）。デフォルト: 無効。 |
| `API_SERVER_PORT` | APIサーバーのポート（デフォルト: `8642`） |
| `API_SERVER_HOST` | APIサーバーのホスト/バインドアドレス（デフォルト: `127.0.0.1`）。ネットワークアクセスには `0.0.0.0` を使用 — `API_SERVER_KEY` と狭い `API_SERVER_CORS_ORIGINS` 許可リストが必要です。 |
| `API_SERVER_MODEL_NAME` | `/v1/models` で公開されるモデル名。デフォルトはプロファイル名（デフォルトプロファイルの場合は `hermes-agent`）。Open WebUIのようなフロントエンドが接続ごとに異なるモデル名を必要とするマルチユーザー設定に便利です。 |
| `GATEWAY_PROXY_URL` | メッセージを転送するリモートHermes APIサーバーのURL（[プロキシモード](/docs/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)）。設定すると、ゲートウェイはプラットフォームのI/Oのみを処理し、すべてのエージェント作業はリモートサーバーに委譲されます。`config.yaml` の `gateway.proxy_url` でも設定可能です。 |
| `GATEWAY_PROXY_KEY` | プロキシモードでリモートAPIサーバーと認証するためのBearerトークン。リモートホストの `API_SERVER_KEY` と一致する必要があります。 |
| `MESSAGING_CWD` | メッセージングモードでのターミナルコマンドの作業ディレクトリ（デフォルト: `~`） |
| `GATEWAY_ALLOWED_USERS` | すべてのプラットフォームで許可するユーザーIDのカンマ区切りリスト |
| `GATEWAY_ALLOW_ALL_USERS` | 許可リストなしですべてのユーザーを許可（`true`/`false`、デフォルト: `false`） |

### Microsoft Graph（Teams会議）

近日対応予定のTeams会議サマリーパイプラインで使用されるMicrosoft Graph RESTクライアント用のアプリ専用認証情報です。Azureポータルでの手順と必要な正確なAPI権限については、[Microsoft Graphアプリケーションの登録](/docs/guides/microsoft-graph-app-registration)を参照してください。

| Variable | Description |
|----------|-------------|
| `MSGRAPH_TENANT_ID` | Graphアプリ登録用のAzure ADテナントID（ディレクトリGUID）。 |
| `MSGRAPH_CLIENT_ID` | Azureアプリ登録のアプリケーション（クライアント）ID。 |
| `MSGRAPH_CLIENT_SECRET` | アプリ登録のクライアントシークレット値。`chmod 600` で `~/.hermes/.env` に保存し、Azureポータルで定期的にローテーションしてください。 |
| `MSGRAPH_SCOPE` | クライアント認証情報トークンリクエスト用のOAuth2スコープ（デフォルト: `https://graph.microsoft.com/.default`）。 |
| `MSGRAPH_AUTHORITY_URL` | Microsoft IDプラットフォームのオーソリティ（デフォルト: `https://login.microsoftonline.com`）。国家/ソブリンクラウド向けにのみ上書きします（例: GCC Highの場合は `https://login.microsoftonline.us`）。 |

### Microsoft Graph Webhookリスナー

Graphイベント（Teams会議、カレンダー、チャットなど）の受信変更通知リスナーです。セットアップとセキュリティ強化については、[Microsoft Graph Webhookリスナー](/docs/user-guide/messaging/msgraph-webhook)を参照してください。

| Variable | Description |
|----------|-------------|
| `MSGRAPH_WEBHOOK_ENABLED` | `msgraph_webhook` ゲートウェイプラットフォームを有効化（`true`/`1`/`yes`）。 |
| `MSGRAPH_WEBHOOK_PORT` | リスナーがバインドするポート（デフォルト: `8646`）。 |
| `MSGRAPH_WEBHOOK_CLIENT_STATE` | Graphがすべての通知で返す共有シークレット。`hmac.compare_digest` で比較されます。`openssl rand -hex 32` で生成します。 |
| `MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES` | Graphリソースパス/パターンのカンマ区切り許可リスト（例: `communications/onlineMeetings,chats/*/messages`）。末尾の `*` は前方一致です。空 = すべて受け入れる。 |
| `MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS` | リスナーへのPOSTを許可するCIDR範囲のカンマ区切りリスト（例: `52.96.0.0/14,52.104.0.0/14`）。空 = すべて許可（デフォルト）。本番環境ではMicrosoft Graphの公開エグレス範囲に制限してください。 |

### Teams会議サマリー配信

[`teams_pipeline` プラグイン](/docs/user-guide/messaging/msgraph-webhook)が有効な場合にのみ使用されます。設定は `config.yaml` の `platforms.teams.extra` でも構成可能です — 両方設定されている場合は環境変数が優先されます。[Microsoft Teams → 会議サマリー配信](/docs/user-guide/messaging/teams#meeting-summary-delivery-teams-meeting-pipeline)を参照してください。

| Variable | Description |
|----------|-------------|
| `TEAMS_DELIVERY_MODE` | `graph` または `incoming_webhook`。 |
| `TEAMS_INCOMING_WEBHOOK_URL` | Teamsが生成したWebhook URL。`TEAMS_DELIVERY_MODE=incoming_webhook` の場合に必須。 |
| `TEAMS_GRAPH_ACCESS_TOKEN` | Graph配信用に事前取得した委譲アクセストークン。ほとんど不要 — 未設定の場合、ライターは `MSGRAPH_*` アプリ認証情報にフォールバックします。 |
| `TEAMS_TEAM_ID` | チャンネル配信（`graph` モード）の対象チームID。 |
| `TEAMS_CHANNEL_ID` | 対象チャンネルID（`TEAMS_TEAM_ID` とペア）。 |
| `TEAMS_CHAT_ID` | 対象の1:1またはグループチャットID（`graph` モードでのチーム+チャンネルの代替）。 |

### LINE Messaging API

バンドルされたLINEプラットフォームプラグイン（`plugins/platforms/line/`）で使用されます。完全なセットアップについては、[メッセージングゲートウェイ → LINE](/docs/user-guide/messaging/line)を参照してください。

| Variable | Description |
|----------|-------------|
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Developers Console（Messaging APIタブ）から取得する長期チャンネルアクセストークン。必須。 |
| `LINE_CHANNEL_SECRET` | チャンネルシークレット（Basic settingsタブ）。HMAC-SHA256のWebhook署名検証に使用されます。必須。 |
| `LINE_HOST` | Webhookのバインドホスト（デフォルト: `0.0.0.0`）。 |
| `LINE_PORT` | Webhookのバインドポート（デフォルト: `8646`）。 |
| `LINE_PUBLIC_URL` | 公開HTTPSベースURL（例: `https://my-tunnel.example.com`）。画像 / 音声 / 動画の送信に必須 — LINEはHTTPSで到達可能なURLのみを受け入れます。 |
| `LINE_ALLOWED_USERS` | ボットへのDMを許可するユーザーID（`U` 始まり）のカンマ区切りリスト。 |
| `LINE_ALLOWED_GROUPS` | ボットが応答するグループID（`C` 始まり）のカンマ区切りリスト。 |
| `LINE_ALLOWED_ROOMS` | ボットが応答するルームID（`R` 始まり）のカンマ区切りリスト。 |
| `LINE_ALLOW_ALL_USERS` | 開発専用の抜け道 — 任意のソースを受け入れます。デフォルト: `false`。 |
| `LINE_HOME_CHANNEL` | `deliver: line` のcronジョブのデフォルト配信先。 |
| `LINE_SLOW_RESPONSE_THRESHOLD` | 低速LLMのTemplate Buttonsポストバックが発火するまでの秒数（デフォルト: `45`）。`0` を設定すると無効化され、常にPushフォールバックします。 |
| `LINE_PENDING_TEXT` | ポストバックボタンと一緒に表示されるバブルテキスト。 |
| `LINE_BUTTON_LABEL` | ポストバックボタンのラベル（デフォルト: `Get answer`）。 |
| `LINE_DELIVERED_TEXT` | 既に配信済みのポストバックが再度タップされたときの返信（デフォルト: `Already replied ✅`）。 |
| `LINE_INTERRUPTED_TEXT` | `/stop` で孤立したポストバックボタンがタップされたときの返信（デフォルト: `Run was interrupted before completion.`）。 |

### 高度なメッセージングチューニング

送信メッセージバッチャーをスロットリングするためのプラットフォームごとの高度なつまみです。ほとんどのユーザーはこれらに触れる必要はありません。デフォルトは、もたつきを感じさせずに各プラットフォームのレート制限を尊重するように設定されています。

| Variable | Description |
|----------|-------------|
| `HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS` | キューに入れたTelegramテキストチャンクをフラッシュするまでの猶予時間（デフォルト: `0.6`）。 |
| `HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS` | 単一のTelegramメッセージが長さ制限を超えた場合の分割チャンク間の遅延（デフォルト: `2.0`）。 |
| `HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS` | キューに入れたTelegramメディアをフラッシュするまでの猶予時間（デフォルト: `0.6`）。 |
| `HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS` | 最後のストリームチャンクとの競合を避けるため、エージェントの終了後にフォローアップを送信するまでの遅延。 |
| `HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT` / `_READ_TIMEOUT` / `_WRITE_TIMEOUT` / `_POOL_TIMEOUT` | 基盤となる `python-telegram-bot` のHTTPタイムアウト（秒）を上書き。 |
| `HERMES_TELEGRAM_HTTP_POOL_SIZE` | Telegram APIへの最大同時HTTP接続数。 |
| `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS` | DNS失敗時に使用されるハードコードされたCloudflareフォールバックIPを無効化（`true`/`false`）。 |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | キューに入れたDiscordテキストチャンクをフラッシュするまでの猶予時間（デフォルト: `0.6`）。 |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Discordメッセージが長さ制限を超えた場合の分割チャンク間の遅延（デフォルト: `2.0`）。 |
| `HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | TelegramバッチつまみのMatrix版。 |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` / `_MAX_CHARS` / `_MAX_MESSAGES` | Feishuバッチャーのチューニング — 遅延、分割遅延、メッセージあたりの最大文字数、バッチあたりの最大メッセージ数。 |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | Feishuメディアのフラッシュ遅延。 |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | Feishu Webhookの重複排除キャッシュのサイズ（デフォルト: `1024`）。 |
| `HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | WeComバッチャーのチューニング。 |
| `HERMES_VISION_DOWNLOAD_TIMEOUT` | 画像をビジョンモデルに渡す前にダウンロードするタイムアウト（秒単位、デフォルト: `30`）。 |
| `HERMES_RESTART_DRAIN_TIMEOUT` | ゲートウェイ: `/restart` 時に再起動を強制する前に、アクティブな実行が完了するのを待つ秒数（デフォルト: `900`）。 |
| `HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT` | ゲートウェイ起動中のプラットフォームごとの接続タイムアウト（秒）。 |
| `HERMES_GATEWAY_BUSY_INPUT_MODE` | ゲートウェイのデフォルトのビジー入力動作: `queue`、`steer`、または `interrupt`。`/busy` でチャットごとに上書きできます。 |
| `HERMES_GATEWAY_BUSY_ACK_ENABLED` | エージェントがビジー状態のときにユーザーが入力を送信した際、ゲートウェイが確認メッセージ（⚡/⏳/⏩）を送信するかどうか（デフォルト: `true`）。`false` に設定するとこれらのメッセージを完全に抑制します — 入力は通常どおりキューイング/ステアリング/割り込みされ、チャット返信のみが抑制されます。`config.yaml` の `display.busy_ack_enabled` からブリッジされます。 |
| `HERMES_CRON_TIMEOUT` | cronジョブのエージェント実行の非アクティブタイムアウト（秒単位、デフォルト: `600`）。エージェントはツールを能動的に呼び出している間やストリームトークンを受信している間は無期限に実行できます — これはアイドル時にのみトリガーされます。`0` を設定すると無制限になります。 |
| `HERMES_CRON_SCRIPT_TIMEOUT` | cronジョブに付随する事前実行スクリプトのタイムアウト（秒単位、デフォルト: `120`）。より長い実行を必要とするスクリプト（例: 反ボットタイミング用のランダム化された遅延）の場合に上書きします。`config.yaml` の `cron.script_timeout_seconds` でも設定可能です。 |
| `HERMES_CRON_MAX_PARALLEL` | ティックごとに並列実行するcronジョブの最大数（デフォルト: `4`）。 |

## エージェントの動作

| Variable | Description |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | 会話ごとの最大ツール呼び出し反復回数（デフォルト: 90） |
| `HERMES_INFERENCE_MODEL` | プロセスレベルでモデル名を上書き（セッションについては `config.yaml` より優先）。`-m`/`--model` フラグでも設定可能。 |
| `HERMES_YOLO_MODE` | `1` に設定すると危険なコマンドの承認プロンプトをバイパス。`--yolo` と同等。 |
| `HERMES_ACCEPT_HOOKS` | `config.yaml` で宣言された未確認のシェルフックを、TTYプロンプトなしで自動承認。`--accept-hooks` または `hooks_auto_accept: true` と同等。 |
| `HERMES_IGNORE_USER_CONFIG` | `~/.hermes/config.yaml` をスキップして組み込みのデフォルトを使用（`.env` の認証情報は引き続き読み込まれます）。`--ignore-user-config` と同等。 |
| `HERMES_IGNORE_RULES` | `AGENTS.md`、`SOUL.md`、`.cursorrules`、メモリ、プリロードされたスキルの自動注入をスキップ。`--ignore-rules` と同等。 |
| `HERMES_MD_NAMES` | 自動注入するルールファイル名のカンマ区切りリスト（デフォルト: `AGENTS.md,CLAUDE.md,.cursorrules,SOUL.md`）。 |
| `HERMES_TOOL_PROGRESS` | ツール進捗表示用の非推奨の互換変数。`config.yaml` の `display.tool_progress` を推奨します。 |
| `HERMES_TOOL_PROGRESS_MODE` | ツール進捗モード用の非推奨の互換変数。`config.yaml` の `display.tool_progress` を推奨します。 |
| `HERMES_HUMAN_DELAY_MODE` | 応答のペース配分: `off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | カスタム遅延範囲の最小値（ms） |
| `HERMES_HUMAN_DELAY_MAX_MS` | カスタム遅延範囲の最大値（ms） |
| `HERMES_QUIET` | 必須でない出力を抑制（`true`/`false`） |
| `HERMES_API_TIMEOUT` | LLM API呼び出しのタイムアウト（秒単位、デフォルト: `1800`） |
| `HERMES_API_CALL_STALE_TIMEOUT` | 非ストリーミングの停滞呼び出しタイムアウト（秒単位、デフォルト: `300`）。ローカルプロバイダーでは未設定のままだと自動的に無効化されます。`config.yaml` の `providers.<id>.stale_timeout_seconds` または `providers.<id>.models.<model>.stale_timeout_seconds` でも設定可能です。 |
| `HERMES_STREAM_READ_TIMEOUT` | ストリーミングソケットの読み取りタイムアウト（秒単位、デフォルト: `120`）。ローカルプロバイダーでは自動的に `HERMES_API_TIMEOUT` まで増加します。ローカルLLMが長いコード生成中にタイムアウトする場合は増やしてください。 |
| `HERMES_STREAM_STALE_TIMEOUT` | 停滞ストリーム検出タイムアウト（秒単位、デフォルト: `180`）。ローカルプロバイダーでは自動的に無効化されます。このウィンドウ内にチャンクが到着しない場合、接続を切断します。 |
| `HERMES_STREAM_RETRIES` | 一時的なネットワークエラー時のストリーム途中の再接続試行回数（デフォルト: `3`）。 |
| `HERMES_AGENT_TIMEOUT` | 実行中のエージェントに対するゲートウェイの非アクティブタイムアウト（秒単位、デフォルト: `900`）。すべてのツール呼び出しとストリームトークンでリセットされます。`0` を設定すると無効化します。 |
| `HERMES_AGENT_TIMEOUT_WARNING` | ゲートウェイ: この秒数の非アクティブの後に警告メッセージを送信（デフォルト: `HERMES_AGENT_TIMEOUT` の75%）。 |
| `HERMES_AGENT_NOTIFY_INTERVAL` | ゲートウェイ: 長時間実行されるエージェントターンでの進捗通知の間隔（秒単位）。 |
| `HERMES_CHECKPOINT_TIMEOUT` | ファイルシステムチェックポイント作成のタイムアウト（秒単位、デフォルト: `30`）。 |
| `HERMES_EXEC_ASK` | ゲートウェイモードで実行承認プロンプトを有効化（`true`/`false`） |
| `HERMES_ENABLE_PROJECT_PLUGINS` | `./.hermes/plugins/` からのリポジトリローカルプラグインの自動検出を有効化（`true`/`false`、デフォルト: `false`） |
| `HERMES_PLUGINS_DEBUG` | `1`/`true` で、詳細なプラグイン検出ログをstderrに表示 — スキャンしたディレクトリ、解析したマニフェスト、スキップ理由、解析または `register()` 失敗時の完全なトレースバック。プラグイン作成者向けです。 |
| `HERMES_BACKGROUND_NOTIFICATIONS` | ゲートウェイのバックグラウンドプロセス通知モード: `all`（デフォルト）、`result`、`error`、`off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | API呼び出し時に注入されるエフェメラルなシステムプロンプト（セッションには永続化されません） |
| `HERMES_PREFILL_MESSAGES_FILE` | API呼び出し時に注入されるエフェメラルなプレフィルメッセージのJSONファイルへのパス。 |
| `HERMES_ALLOW_PRIVATE_URLS` | `true`/`false` — ツールがlocalhost/プライベートネットワークのURLを取得することを許可。ゲートウェイモードではデフォルトでオフ。 |
| `HERMES_REDACT_SECRETS` | `true`/`false` — ツール出力、ログ、チャット応答での機密情報の伏せ字を制御（デフォルト: `true`）。 |
| `HERMES_WRITE_SAFE_ROOT` | `write_file`/`patch` の書き込みを制限する任意のディレクトリプレフィックス。それ以外のパスは承認が必要です。 |
| `HERMES_DISABLE_FILE_STATE_GUARD` | `1` に設定すると、`patch`/`write_file` の「読み取ってからファイルが変更された」ガードをオフにします。 |
| `HERMES_CORE_TOOLS` | 正規のコアツールリストのカンマ区切り上書き（高度。ほとんど不要）。 |
| `HERMES_BUNDLED_SKILLS` | 起動時に読み込まれるバンドルスキルのリストのカンマ区切り上書き。 |
| `HERMES_OPTIONAL_SKILLS` | 初回実行時に自動インストールする任意スキル名のカンマ区切りリスト。 |
| `HERMES_DEBUG_INTERRUPT` | `1` に設定すると、詳細な割り込み/キャンセルトレースを `agent.log` にログ出力。 |
| `HERMES_DUMP_REQUESTS` | APIリクエストのペイロードをログファイルにダンプ（`true`/`false`） |
| `HERMES_DUMP_REQUEST_STDOUT` | APIリクエストのペイロードをログファイルではなくstdoutにダンプ。 |
| `HERMES_OAUTH_TRACE` | `1` に設定すると、OAuthトークン交換とリフレッシュの試行をログ出力。伏せ字されたタイミング情報を含みます。 |
| `HERMES_OAUTH_FILE` | OAuth認証情報の保存に使用するパスを上書き（デフォルト: `~/.hermes/auth.json`）。 |
| `HERMES_AGENT_HELP_GUIDANCE` | カスタムデプロイ用にシステムプロンプトへ追加のガイダンステキストを追記。 |
| `HERMES_AGENT_LOGO` | CLI起動時のASCIIバナーロゴを上書き。 |
| `DELEGATION_MAX_CONCURRENT_CHILDREN` | `delegate_task` バッチあたりの最大並列サブエージェント数（デフォルト: `3`、下限1、上限なし）。`config.yaml` の `delegation.max_concurrent_children` でも設定可能 — 設定値が優先されます。 |

## インターフェース

| Variable | Description |
|----------|-------------|
| `HERMES_TUI` | `1` に設定すると、クラシックCLIの代わりに [TUI](../user-guide/tui.md) を起動。`--tui` を渡すのと同等。 |
| `HERMES_TUI_DIR` | ビルド済み `ui-tui/` ディレクトリへのパス（`dist/entry.js` と populated な `node_modules` を含む必要があります）。ディストリビューションやNixが初回起動時の `npm install` をスキップするために使用します。 |
| `HERMES_TUI_RESUME` | 起動時に特定のTUIセッションをIDで再開。設定すると、`hermes --tui` は新しいセッションを生成せず、指定されたセッションを引き継ぎます — 切断やターミナルクラッシュ後の再接続に便利です。 |
| `HERMES_TUI_THEME` | TUIのカラーテーマを強制: `light`、`dark`、または生の6文字の背景16進数（例: `ffffff` や `1a1a2e`）。未設定の場合、Hermesは `COLORFGBG` とターミナル背景クエリを使って自動検出します。この変数は、`COLORFGBG` を設定しないターミナル（Ghostty、Warp、iTerm2など）での検出を上書きします。 |
| `HERMES_INFERENCE_MODEL` | `config.yaml` を変更せずに `hermes -z` / `hermes chat` のモデルを強制。`HERMES_INFERENCE_PROVIDER` とペアで使用します。実行ごとにデフォルトモデルを上書きする必要があるスクリプト呼び出し元（sweeper、CI、バッチランナー）に便利です。 |

## セッション設定

| Variable | Description |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | N分間の非アクティブ後にセッションをリセット（デフォルト: 1440） |
| `SESSION_RESET_HOUR` | 24時間形式での毎日のリセット時刻（デフォルト: 4 = 午前4時） |

## コンテキスト圧縮（config.yamlのみ）

コンテキスト圧縮は `config.yaml` を通じてのみ設定されます — そのための環境変数はありません。しきい値の設定は `compression:` ブロックにあり、要約モデル/プロバイダーは `auxiliary.compression:` の下にあります。

```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20         # 直近のテールとして保持するしきい値の割合
  protect_last_n: 20         # 圧縮せずに保持する直近メッセージの最小数
```

:::info レガシーからの移行
`compression.summary_model`、`compression.summary_provider`、`compression.summary_base_url` を持つ古い設定は、初回読み込み時に自動的に `auxiliary.compression.*` へ移行されます。
:::

## 補助タスクの上書き

| Variable | Description |
|----------|-------------|
| `AUXILIARY_VISION_PROVIDER` | ビジョンタスク用のプロバイダーを上書き |
| `AUXILIARY_VISION_MODEL` | ビジョンタスク用のモデルを上書き |
| `AUXILIARY_VISION_BASE_URL` | ビジョンタスク用の直接OpenAI互換エンドポイント |
| `AUXILIARY_VISION_API_KEY` | `AUXILIARY_VISION_BASE_URL` とペアになるAPIキー |
| `AUXILIARY_WEB_EXTRACT_PROVIDER` | Web抽出/要約用のプロバイダーを上書き |
| `AUXILIARY_WEB_EXTRACT_MODEL` | Web抽出/要約用のモデルを上書き |
| `AUXILIARY_WEB_EXTRACT_BASE_URL` | Web抽出/要約用の直接OpenAI互換エンドポイント |
| `AUXILIARY_WEB_EXTRACT_API_KEY` | `AUXILIARY_WEB_EXTRACT_BASE_URL` とペアになるAPIキー |

タスク固有の直接エンドポイントについては、Hermesはそのタスクに設定されたAPIキーまたは `OPENAI_API_KEY` を使用します。これらのカスタムエンドポイントに `OPENROUTER_API_KEY` を再利用することはありません。

## フォールバックプロバイダー（config.yamlのみ）

プライマリモデルのフォールバックチェーンは `config.yaml` を通じてのみ設定されます — そのための環境変数はありません。`provider` と `model` のキーを持つトップレベルの `fallback_providers` リストを追加すると、メインモデルがエラーに遭遇したときに自動フェイルオーバーが有効になります。

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

古いトップレベルの `fallback_model` 単一プロバイダー形式も後方互換性のため引き続き読み込まれますが、新しい設定では `fallback_providers` を使用してください。

詳細は [フォールバックプロバイダー](/docs/user-guide/features/fallback-providers) を参照してください。

## プロバイダールーティング（config.yamlのみ）

これらは `~/.hermes/config.yaml` の `provider_routing` セクションに記述します:

| Key | Description |
|-----|-------------|
| `sort` | プロバイダーのソート: `"price"`（デフォルト）、`"throughput"`、または `"latency"` |
| `only` | 許可するプロバイダースラッグのリスト（例: `["anthropic", "google"]`） |
| `ignore` | スキップするプロバイダースラッグのリスト |
| `order` | 順番に試すプロバイダースラッグのリスト |
| `require_parameters` | すべてのリクエストパラメータをサポートするプロバイダーのみを使用（`true`/`false`） |
| `data_collection` | `"allow"`（デフォルト）または `"deny"` でデータを保存するプロバイダーを除外 |

:::tip
環境変数の設定には `hermes config set` を使用してください — 適切なファイル（シークレットは `.env`、それ以外は `config.yaml`）に自動的に保存されます。
:::
