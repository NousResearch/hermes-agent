---
sidebar_position: 2
title: "環境変数"
description: "Hermes Agentが使用するすべての環境変数の完全なリファレンス"
---

# 環境変数リファレンス

Hermesは環境変数をプロセス環境から読み込み、ユーザーが管理するシークレットについては `~/.hermes/.env` からも読み込みます。APIキー、ボットトークン、OAuthシークレット、その他の認証情報は `.env` に記述してください。シークレットではない動作設定については、対応する設定キーが存在する場合は `config.yaml` を優先してください。以下の変数の中には、プロセス専用の上書きや内部ブリッジ変数もあり、ここに記載されているからといって `.env` にコミットすべきではありません。プラットフォームYAMLブリッジ（およびTelegram/Signalのメンションフォールバック）が書き込んだ値は、その設定が変更されると更新されます。これはデーモンの置き換えをまたいでも同様です。YAMLの完全な読み込みに成功した後は、削除されたキーに対応する、変更されていないブリッジ所有の値も取り消されます。読み込みに失敗した場合はこのクリーンアップは行われず、明示的なプロセス、ユーザーの `.env`、または管理された `.env` の値が引き続き優先されます。ブリッジは自身が書き込んだ内容を `HERMES_BRIDGED_<NAME>` 変数で追跡します。この変数は値そのものではなく値のSHA-256ダイジェストを保持し、すべてのエージェントサブプロセスから削除されます。プロセスでエクスポートされた値が、たまたまブリッジ自身が以前に書き込んだ値と一致する場合は、ブリッジのものとして扱われ、`config.yaml` に従います。

## LLMプロバイダー

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter APIキー（柔軟性の点で推奨） |
| `OPENROUTER_BASE_URL` | OpenRouter互換のベースURLを上書き |
| `FIREWORKS_API_KEY` | Fireworks AI APIキー（[app.fireworks.ai](https://app.fireworks.ai/settings/users/api-keys)）。エンドポイントの上書きは `config.yaml` の `model.base_url` で設定します。 |
| `HERMES_OPENROUTER_CACHE` | OpenRouterのレスポンスキャッシュを有効化（`1`/`true`/`yes`/`on`）。config.yamlの `openrouter.response_cache` を上書きします。[Response Caching](https://openrouter.ai/docs/guides/features/response-caching) を参照してください。 |
| `HERMES_OPENROUTER_CACHE_TTL` | キャッシュのTTL（秒単位、1〜86400）。config.yamlの `openrouter.response_cache_ttl` を上書きします。 |
| `NOUS_BASE_URL` | Nous PortalのベースURLを上書き（通常は不要。開発・テスト用途のみ） |
| `NOUS_INFERENCE_BASE_URL` | Nousの推論エンドポイントを直接上書き |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway APIキー（[ai-gateway.vercel.sh](https://ai-gateway.vercel.sh)） |
| `AI_GATEWAY_BASE_URL` | AI GatewayのベースURLを上書き（デフォルト: `https://ai-gateway.vercel.sh/v1`） |
| `OPENAI_API_KEY` | OpenAI APIキー（`openai-api` プロバイダー）、または `OPENAI_BASE_URL` が設定されている場合はカスタムOpenAI互換エンドポイント用のキー。後者の場合、キーはそのURLの正確なオリジン（スキーム、ホスト、ポート）にのみ送信され、別のポートや同一ホストの `http://` には送信されません。`sk-or-` で始まる場合にのみOpenRouterキーとして扱われます。OpenRouterキーは `OPENROUTER_API_KEY` に設定してください |
| `OPENAI_BASE_URL` | カスタムエンドポイント（VLLM、SGLangなど）のベースURL |
| `HERMES_CODEX_BASE_URL` | `openai-codex`（ChatGPTサブスクリプション）プロバイダーを、デフォルトのCodexバックエンドではなくプロキシ経由でルーティングします。プールの解決、補助/rawクライアント、401/429時の認証情報ローテーションなど、認証情報が使用されるすべての箇所に適用されます。これが未設定の場合、`model.provider: openai-codex` の下の `model.base_url` が第2の上書きになります。 |
| `LM_API_KEY` | LM Studio（`lmstudio` プロバイダー）用のAPIキー。ローカルサーバーではプレースホルダーであることが多いです |
| `LM_BASE_URL` | LM StudioのベースURL（デフォルト: `http://localhost:1234/v1`） |
| `GH_TOKEN` | GitHubトークン（`gh` CLIでも使用）。Hermesが `GITHUB_TOKEN` を読み込むすべての箇所で使用されます |
| `GITHUB_TOKEN` | GitHub APIアクセス用のGitHubトークン（Skills Hub、gitの認証情報、更新チェック。下のSkills Hubの行を参照） |
| `GLM_API_KEY` | z.ai / ZhipuAI GLM APIキー（[z.ai](https://z.ai)） |
| `ZAI_API_KEY` | `GLM_API_KEY` のエイリアス |
| `Z_AI_API_KEY` | `GLM_API_KEY` のエイリアス |
| `GLM_BASE_URL` | z.aiのベースURLを上書き（デフォルト: `https://api.z.ai/api/paas/v4`） |
| `KIMI_API_KEY` | Kimi / Moonshot AI APIキー（[moonshot.ai](https://platform.moonshot.ai)） |
| `KIMI_CODING_API_KEY` | `kimi-coding` プロバイダー用のエイリアスキー（`KIMI_API_KEY` と並んで受け付けられます） |
| `KIMI_BASE_URL` | KimiのベースURLを上書き（デフォルト: `https://api.moonshot.ai/v1`） |
| `KIMI_CN_API_KEY` | Kimi / Moonshot China APIキー（[moonshot.cn](https://platform.moonshot.cn)） |
| `ARCEEAI_API_KEY` | Arcee AI APIキー（[chat.arcee.ai](https://chat.arcee.ai/)） |
| `ARCEE_BASE_URL` | ArceeのベースURLを上書き（デフォルト: `https://api.arcee.ai/api/v1`） |
| `GMI_API_KEY` | GMI Cloud APIキー（[gmicloud.ai](https://www.gmicloud.ai/)） |
| `GMI_BASE_URL` | GMI CloudのベースURLを上書き（デフォルト: `https://api.gmi-serving.com/v1`） |
| `ACTUAL_API_KEY` | Actual Computerの推論キー（`ac_...`、[actual.inc/user/keys](https://actual.inc/user/keys)）。ローカルデーモンでは不要です。 |
| `ACTUAL_BASE_URL` | ActualのベースURL用のレガシーフォールバック。代わりに `config.yaml` で `model.provider: actual` と `model.base_url` を設定してください。YAMLのURLが優先されます。デフォルトは `https://api.actual.inc/v1` です。 |
| `MINIMAX_API_KEY` | MiniMax APIキー — グローバルエンドポイント（[minimax.io](https://www.minimax.io)）。**`minimax-oauth` では使用されません**（OAuthパスはブラウザログインを使用）。 |
| `MINIMAX_BASE_URL` | MiniMaxのベースURLを上書き（デフォルト: `https://api.minimax.io/anthropic` — HermesはMiniMaxのAnthropic Messages互換エンドポイントを使用）。**`minimax-oauth` では使用されません**。 |
| `MINIMAX_CN_API_KEY` | MiniMax APIキー — 中国エンドポイント（[minimaxi.com](https://www.minimaxi.com)）。**`minimax-oauth` では使用されません**（OAuthパスはブラウザログインを使用）。 |
| `MINIMAX_CN_BASE_URL` | MiniMax ChinaのベースURLを上書き（デフォルト: `https://api.minimaxi.com/anthropic`）。**`minimax-oauth` では使用されません**。 |
| `KILOCODE_API_KEY` | Kilo Code APIキー（[kilo.ai](https://kilo.ai)） |
| `KILOCODE_BASE_URL` | Kilo CodeのベースURLを上書き（デフォルト: `https://api.kilo.ai/api/gateway`） |
| `XIAOMI_API_KEY` | Xiaomi MiMo APIキー（[platform.xiaomimimo.com](https://platform.xiaomimimo.com)） |
| `XIAOMI_BASE_URL` | Xiaomi MiMoのベースURLを上書き（デフォルト: `https://api.xiaomimimo.com/v1`） |
| `UPSTAGE_API_KEY` | Solarモデル用のUpstage APIキー（[console.upstage.ai](https://console.upstage.ai/api-keys)） |
| `UPSTAGE_BASE_URL` | UpstageのベースURLを上書き（デフォルト: `https://api.upstage.ai/v1`） |
| `TOKENHUB_API_KEY` | Tencent TokenHub APIキー（[tokenhub.tencentmaas.com](https://tokenhub.tencentmaas.com)） |
| `TOKENHUB_BASE_URL` | Tencent TokenHubのベースURLを上書き（デフォルト: `https://tokenhub.tencentmaas.com/v1`） |
| `TOKENPLAN_API_KEY` | Tencent TokenPlan APIキー（LKEAP。Anthropic Messagesエンドポイント） |
| `TOKENPLAN_BASE_URL` | Tencent TokenPlanのベースURLを上書き（デフォルト: `https://api.lkeap.cloud.tencent.com/plan/anthropic`） |
| `AZURE_FOUNDRY_API_KEY` | Microsoft Foundry / Azure OpenAI APIキー（[ai.azure.com](https://ai.azure.com/)）。`model.auth_mode: entra_id` の場合は不要 |
| `AZURE_FOUNDRY_BASE_URL` | Microsoft FoundryのエンドポイントURL（例: OpenAIスタイルは `https://<resource>.openai.azure.com/openai/v1`、Anthropicスタイルは `https://<resource>.services.ai.azure.com/anthropic`） |
| `AZURE_ANTHROPIC_KEY` | `provider: anthropic` + Microsoft FoundryのClaudeデプロイメントを指す `base_url` 用のAzure Anthropic APIキー（AnthropicとAzure Anthropicの両方を設定している場合の `ANTHROPIC_API_KEY` の代替） |
| `AZURE_TENANT_ID` | Entra IDのテナントID（サービスプリンシパルのフロー。`model.auth_mode: entra_id` の場合に `azure-identity` が使用） |
| `AZURE_CLIENT_ID` | Entra IDのクライアントID（サービスプリンシパル、ワークロードID、またはユーザー割り当てマネージドID） |
| `AZURE_CLIENT_SECRET` | `EnvironmentCredential` が使用するサービスプリンシパルのシークレット |
| `AZURE_CLIENT_CERTIFICATE_PATH` | サービスプリンシパルの証明書（`AZURE_CLIENT_SECRET` の代替） |
| `AZURE_FEDERATED_TOKEN_FILE` | AKS Workload Identity / OIDCフロー用のフェデレーショントークンファイルのパス |
| `AZURE_AUTHORITY_HOST` | ソブリンクラウド用のオーソリティの上書き（例: Azure Governmentの場合は `https://login.microsoftonline.us`）。[Azure Foundryガイド](../guides/azure-foundry.md#sovereign-clouds-government-china)を参照してください |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | App Service、Functions、Container Apps用のマネージドIDエンドポイント。VMでは通常代わりにIMDSを使用し、これらは設定しません |
| `HF_TOKEN` | Inference Providers用のHugging Faceトークン（[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)） |
| `HF_BASE_URL` | Hugging FaceのベースURLを上書き（デフォルト: `https://router.huggingface.co/v1`） |
| `GOOGLE_API_KEY` | Google AI Studio APIキー（[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)） |
| `GEMINI_API_KEY` | `GOOGLE_API_KEY` のエイリアス |
| `GEMINI_BASE_URL` | Google AI StudioのベースURLを上書き |
| `VERTEX_CREDENTIALS_PATH` | Vertex AI（Gemini）用のGoogle CloudサービスアカウントJSONへのパス。Vertexは静的なAPIキーではなくOAuth2を使用します。`GOOGLE_APPLICATION_CREDENTIALS`、次にADC（`gcloud auth application-default login`）にフォールバックします。プロジェクト/リージョンは `config.yaml` の `vertex:` の下で設定します |
| `ANTHROPIC_API_KEY` | Anthropic Console APIキー（[console.anthropic.com](https://console.anthropic.com/)） |
| `ANTHROPIC_BASE_URL` | AnthropicのAPIベースURLを上書き |
| `ANTHROPIC_TOKEN` | 手動またはレガシーのAnthropic OAuth/セットアップトークンの上書き |
| `DASHSCOPE_API_KEY` | Qwenモデル用のQwen Cloud（Alibaba DashScope）APIキー（[modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)） |
| `DASHSCOPE_BASE_URL` | カスタムDashScopeベースURL（デフォルト: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`。中国本土リージョンでは `https://dashscope.aliyuncs.com/compatible-mode/v1` を使用） |
| `DASHSCOPE_CN_BASE_URL` | `alibaba-cn` の中国本土向けDashScopeベースURLを上書き |
| `ALIBABA_CODING_PLAN_API_KEY` | Qwen Coding Plan APIキー（`alibaba-coding-plan`。`alibaba-coding-plan-cn` のフォールバックにもなります） |
| `ALIBABA_CODING_PLAN_CN_API_KEY` | 中国本土向け `alibaba-coding-plan-cn` プロバイダー用のQwen Coding Plan APIキー（共有キーより先にチェックされるため、CNの行のみが有効になります） |
| `ALIBABA_CODING_PLAN_BASE_URL` | Qwen Coding PlanのベースURLを上書き（国際版） |
| `ALIBABA_CODING_PLAN_CN_BASE_URL` | Qwen Coding PlanのベースURLを上書き（中国本土） |
| `ALIBABA_TOKEN_PLAN_API_KEY` | Alibaba Model Studio Token Plan APIキー（`alibaba-token-plan`。`alibaba-token-plan-cn` のフォールバックにもなります） |
| `ALIBABA_TOKEN_PLAN_CN_API_KEY` | 中国本土向け `alibaba-token-plan-cn` プロバイダー用のToken Plan APIキー（共有キーより先にチェックされます） |
| `ALIBABA_TOKEN_PLAN_BASE_URL` | Token PlanのベースURLを上書き（国際版） |
| `ALIBABA_TOKEN_PLAN_CN_BASE_URL` | Token PlanのベースURLを上書き（中国本土） |
| `DEEPSEEK_API_KEY` | DeepSeekへ直接アクセスするためのDeepSeek APIキー（[platform.deepseek.com](https://platform.deepseek.com/api_keys)） |
| `DEEPSEEK_BASE_URL` | カスタムDeepSeek APIベースURL |
| `DEEPINFRA_API_KEY` | DeepInfra APIキー（[deepinfra.com](https://deepinfra.com/dash/api_keys)） |
| `DEEPINFRA_BASE_URL` | DeepInfraのベースURLの上書き |
| `NOVITA_API_KEY` | NovitaAI APIキー — Model API、Agent Sandbox、GPU Cloudを提供するAIネイティブクラウド（[novita.ai/settings/key-management](https://novita.ai/settings/key-management)） |
| `NOVITA_BASE_URL` | NovitaAIのベースURLを上書き（デフォルト: `https://api.novita.ai/openai/v1`） |
| `RAMP_ROUTER_API_KEY` | Ramp Router APIキー（[app.router.com/keys](https://app.router.com/keys)）。エイリアス `ROUTER_API_KEY` も受け付けます |
| `RAMP_ROUTER_BASE_URL` | Ramp RouterのベースURLを上書き（デフォルト: `https://api.router.com/v1`） |
| `NEBIUS_API_KEY` | Nebius Token Factory APIキー（[tokenfactory.nebius.com](https://tokenfactory.nebius.com/)）。`NEBIUS_TOKEN_FACTORY_API_KEY` も受け付けます |
| `NEBIUS_BASE_URL` | Nebius Token FactoryのベースURLを上書き（デフォルト: `https://api.tokenfactory.nebius.com/v1`） |
| `NVIDIA_API_KEY` | NVIDIA NIM APIキー — Nemotronおよびオープンモデル（[build.nvidia.com](https://build.nvidia.com)） |
| `NVIDIA_BASE_URL` | NVIDIAのベースURLを上書き（デフォルト: `https://integrate.api.nvidia.com/v1`。ローカルNIMエンドポイントの場合は `http://localhost:8000/v1` を設定） |
| `STEPFUN_API_KEY` | StepFun APIキー — Stepシリーズモデル（[platform.stepfun.com](https://platform.stepfun.com)） |
| `STEPFUN_BASE_URL` | StepFunのベースURLを上書き（デフォルト: `https://api.stepfun.com/v1`） |
| `OLLAMA_API_KEY` | Ollama Cloud APIキー — ローカルGPU不要のマネージドOllamaカタログ（[ollama.com/settings/keys](https://ollama.com/settings/keys)） |
| `OLLAMA_BASE_URL` | Ollama CloudのベースURLを上書き（デフォルト: `https://ollama.com/v1`） |
| `XAI_API_KEY` | チャット + TTS + Web検索用のxAI（Grok）APIキー（[console.x.ai](https://console.x.ai/)） |
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
| `HERMES_LOCAL_STT_LANGUAGE` | フォールバック用のSTT言語ヒント。プロバイダー自身の `language` とグローバルな `stt.language` の両方が空の場合にのみ使用されます。コマンドバックエンド（`HERMES_LOCAL_STT_COMMAND`、ローカルの `whisper` CLI）は、ヒントが解決されない場合に独自のデフォルト（`en`）を適用することがあります。 |
| `HERMES_HOME` | 設定とユーザーデータのホームを選択します。値に含まれるリテラルの `~` や `$VAR` は展開されるため（fishは `VAR=~/…` 内の `~` を展開しません）、カレントディレクトリからの相対パスとして解決されることはありません。デフォルトはPOSIXでは `~/.hermes`、Windowsでは `%LOCALAPPDATA%\hermes` です。公式Dockerイメージでは `/opt/data` を使用します。プロファイル/ランタイムのコンテキストによって、より具体的なホームが選択される場合があります。 |
| `HERMES_CONFIG_BACKEND` | プロファイルの `config.yaml` ユーザーレイヤーを読み書きする場所を選択します。`file`（デフォルト）はローカルの `config.yaml`、`remote` は [Remote Config](/docs/user-guide/remote-config) です。それ以外の値、または設定を提供できないリモートバックエンドの場合、Hermesはデフォルトにフォールバックせず、起動時にエラーで停止します。`.env` で明示的に空の値を設定すると、古いシェルの値が `file` にリセットされます。起動環境、起動ホームの `.env`、および管理された `.env` から読み込まれ、`config.yaml` からは読み込まれません。[Config backend](/docs/developer-guide/config-backend) を参照してください。 |
| `HERMES_CONFIG_REMOTE_URL` | [Remote Config](/docs/user-guide/remote-config) プレーンのベースURL（デフォルト `https://config-config.nousresearch.com`。`https://` のみ、プレーンな `http://` はループバックへのみ、ユーザー情報は不可）。設定の取得元を示すもので、設定を読み込む前に必要となるため環境変数になっています。 |
| `HERMES_CONFIG_INSTANCE_ID` | [Remote Config](/docs/user-guide/remote-config) プレーンにおけるこのエージェントのID。`HERMES_CONFIG_BACKEND=remote` の場合は必須です。プレーンが最初の設定読み込みを提供するために必要なため、環境変数になっています。 |
| `HERMES_DISABLE_WINDOWS_UTF8` | **Windowsのみ。** `1` に設定するとUTF-8 stdioシム（`configure_windows_stdio()`）を無効化し、コンソールのロケールコードページにフォールバックします。エンコーディングバグの二分探索に便利ですが、通常の運用で適切な設定になることはまれです |
| `HERMES_KANBAN_HOME` | かんばんボード（DB + ワークスペース + ワーカーログ）の起点となる共有Hermesルートを上書き。`get_default_hermes_root()`（アクティブなプロファイルの親）にフォールバックします。テストや特殊なデプロイに便利です |
| `HERMES_KANBAN_BOARD` | このプロセスのアクティブなかんばんボードを固定。`~/.hermes/kanban/current` より優先されます。ディスパッチャーがこれをワーカーのサブプロセス環境に注入するため、ワーカーは他のボードのタスクを物理的に参照できません。デフォルトは `default`。スラッグ検証: 小文字の英数字 + ハイフン + アンダースコア、1〜64文字 |
| `HERMES_KANBAN_DB` | かんばんデータベースファイルのパスを直接固定（最優先。`HERMES_KANBAN_BOARD` と `HERMES_KANBAN_HOME` に優先）。ディスパッチャーがこれをワーカーのサブプロセス環境に注入するため、プロファイルワーカーはディスパッチャーのボードに収束します |
| `HERMES_KANBAN_WORKSPACES_ROOT` | かんばんワークスペースルートを直接固定（ワークスペースについては最優先。`HERMES_KANBAN_HOME` に優先）。ディスパッチャーがこれをワーカーのサブプロセス環境に注入します |
| `HERMES_KANBAN_DISPATCH_IN_GATEWAY` | `kanban.dispatch_in_gateway` のランタイム上書き。`0`、`false`、`no`、または `off` に設定すると、ゲートウェイが組み込みのKanbanディスパッチャーを起動しなくなります。それ以外の空でない値では有効になります。別のディスパッチャープロセスがボードを所有している場合に便利です。 |

## プロバイダー認証（OAuth）

ネイティブのAnthropic認証では、Hermesは存在する場合にClaude Code自身の認証情報ファイルを優先します。これらの認証情報は自動的に更新できるためです。**AnthropicへのOAuthには、追加利用クレジットを購入したClaude Maxプランが必要です** — HermesはClaude Codeとしてルーティングし、Maxプランの基本割り当てではなく追加・超過クレジットからのみ消費し、Claude Proでは動作しません。Max + 追加クレジットがない場合は、代わりにAPIキーを使用してください。`ANTHROPIC_TOKEN` などの環境変数は手動の上書きとして引き続き有用ですが、Claude Maxログインの推奨パスではなくなりました。

| Variable | Description |
|----------|-------------|
| `HERMES_PORTAL_BASE_URL` | Nous PortalのURLを上書き（開発・テスト用）。マルチプレキシング時はプロファイルごと: 提供されるプロファイルの `.env` に設定します。 |
| `NOUS_INFERENCE_BASE_URL` | Nousの推論API URLを上書き。Portalのレスポンスが指定できる唯一の非本番ホストでもあります: Portalが返した推論URLがこの上書きと一致する場合、本番環境に修復されずに受け入れられ、永続化されます。マルチプレキシング時はプロファイルごと。 |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | 再発行までのエージェントキーの最小TTL（デフォルト: 1800 = 30分） |
| `HERMES_NOUS_TIMEOUT_SECONDS` | Nous認証情報 / トークンフローのHTTPタイムアウト |
| `HERMES_DUMP_REQUESTS` | APIリクエストのペイロードをログファイルにダンプ（`true`/`false`） |
| `HERMES_PREFILL_MESSAGES_FILE` | API呼び出し時に注入されるエフェメラルなプレフィルメッセージのJSONファイルへのパス |
| `HERMES_TIMEZONE` | IANAタイムゾーンの上書き（例: `America/New_York`）。Linux/macOSでは `execute_code` の子プロセスにも `TZ` としてエクスポートされます。Windowsでは、Windows Cランタイムが理解できるのはPOSIX形式の `TZ` 文字列のみで、IANA名を誤ったオフセットとして解釈してしまうため、子プロセスは代わりにOSのタイムゾーンを維持します |

## ツールAPI

| Variable | Description |
|----------|-------------|
| `PARALLEL_API_KEY` | AIネイティブWeb検索（[parallel.ai](https://parallel.ai/)） |
| `FIRECRAWL_API_KEY` | Webスクレイピングおよびクラウドブラウザ（[firecrawl.dev](https://firecrawl.dev/)） |
| `FIRECRAWL_API_URL` | セルフホストインスタンス用のカスタムFirecrawl APIエンドポイント（任意） |
| `TAVILY_API_KEY` | 検索/抽出の上限を引き上げるためのオプションのTavily APIキー。TavilyをWebバックエンドとして選択した後は、キーなしでもアクセスできます（[app.tavily.com](https://app.tavily.com/home)、[keyless docs](https://docs.tavily.com/documentation/keyless)） |
| `TAVILY_BASE_URL` | Tavily APIエンドポイントを上書き。企業プロキシやセルフホストのTavily互換検索バックエンドに便利です。`GROQ_BASE_URL` と同じパターンです。 |
| `PERPLEXITY_API_KEY` | `perplexity` Webバックエンド用のPerplexity Search APIキー — ランク付けされた検索結果と、抽出用のクエリに関連するページスニペット（[perplexity.ai/account/api](https://www.perplexity.ai/account/api)） |
| `PERPLEXITY_BASE_URL` | プロキシ用にPerplexity APIエンドポイントを上書き（デフォルト `https://api.perplexity.ai`）（オプション） |
| `SEARXNG_URL` | 無料のセルフホストWeb検索用SearXNGインスタンスのURL — APIキー不要（[searxng.github.io](https://searxng.github.io/searxng/)） |
| `EXA_API_KEY` | AIネイティブWeb検索とコンテンツ用のExa APIキー（[exa.ai](https://exa.ai/)） |
| `BRAVE_SEARCH_API_KEY` | Web検索用のBrave Search APIサブスクリプショントークン（無料枠あり）（[brave.com/search/api](https://brave.com/search/api/)） |
| `BROWSERBASE_API_KEY` | ブラウザ自動化（[browserbase.com](https://browserbase.com/)） |
| `BROWSERBASE_PROJECT_ID` | BrowserbaseプロジェクトID |
| `BROWSER_USE_API_KEY` | Browser Useクラウドブラウザ APIキー（[browser-use.com](https://browser-use.com/)） |
| `FIRECRAWL_BROWSER_TTL` | FirecrawlブラウザセッションのTTL（秒単位、デフォルト: 300） |
| `BROWSER_CDP_URL` | Chromiumを起動する代わりにブラウザツールが接続するChrome DevTools Protocolエンドポイント（`/browser connect` で設定、例: `ws://localhost:9222`。`browser.cdp_url` と同じ） |
| `CAMOFOX_*` | Camofoxプラグイン（`hermes plugins install camofox`）のみが読み込みます。Hermes本体はこれらを無視します。 |
| `BROWSER_INACTIVITY_TIMEOUT` | ブラウザセッションの非アクティブタイムアウト（秒単位） |
| `AGENT_BROWSER_*` | 現在は読み込まれません。更新時、キーが未設定の場合はプロファイルの `.env` の値が設定にコピーされます（`browser.engine: auto` と `browser.headed: false` は未設定とみなされ、相対パスの `AGENT_BROWSER_PROFILE` は絶対パスになります）: `AGENT_BROWSER_EXECUTABLE_PATH` → `browser.executable_path`、`AGENT_BROWSER_PROFILE` → `browser.user_data_dir`、`AGENT_BROWSER_ARGS` → `browser.chromium_args`、`AGENT_BROWSER_HEADED` → `browser.headed`、`AGENT_BROWSER_ENGINE` → `browser.engine`。 |
| `FAL_KEY` | 画像生成（[fal.ai](https://fal.ai/)） |
| `KREA_API_KEY` | Krea 3およびKrea 2による画像生成用のKrea APIキー（[krea.ai](https://krea.ai/)） |
| `GROQ_API_KEY` | Groq Whisper STT APIキー（[groq.com](https://groq.com/)） |
| `ELEVENLABS_API_KEY` | ElevenLabsのプレミアムTTSボイス（[elevenlabs.io](https://elevenlabs.io/)） |
| `PORCUPINE_ACCESS_KEY` | Picovoice Porcupineウェイクワードエンジン（[console.picovoice.ai](https://console.picovoice.ai/)） — Porcupineを選択した場合は必須。openWakeWordとsherpaはキー不要です |
| `STT_GROQ_MODEL` | Groq STTモデルを上書き（デフォルト: `whisper-large-v3-turbo`） |
| `GROQ_BASE_URL` | GroqのOpenAI互換STTエンドポイントを上書き |
| `STT_OPENAI_MODEL` | OpenAI STTモデルを上書き（デフォルト: `whisper-1`） |
| `STT_OPENAI_BASE_URL` | OpenAI互換STTエンドポイントを上書き |
| `GITHUB_TOKEN` | Skills Hub（APIレート制限の緩和、スキルの公開）およびデスクトップアプリの更新チェック用のGitHubトークン（`GH_TOKEN` も使用されます。どちらもない場合、デスクトップは `gh` CLIのログイン、次に匿名アクセスにフォールバックします） |
| `HONCHO_API_KEY` | セッションをまたぐユーザーモデリング（[honcho.dev](https://honcho.dev/)） |
| `HONCHO_BASE_URL` | セルフホストHonchoインスタンス用のベースURL（デフォルト: Honchoクラウド）。ローカルインスタンスではAPIキー不要 |
| `HINDSIGHT_API_KEY` | グラフ対応の永続メモリ用のHindsight APIキー（[hindsight.vectorize.io](https://hindsight.vectorize.io)） |
| `HINDSIGHT_API_URL` | Hindsight APIのベースURL（デフォルト: `https://api.hindsight.vectorize.io`） |
| `HINDSIGHT_TIMEOUT` | HindsightメモリプロバイダーへのAPI呼び出しのタイムアウト（秒単位、デフォルト: `60`）。`/sync` や `on_session_switch` の際にHindsightインスタンスの応答が遅く、`errors.log` にタイムアウトが出る場合は増やしてください。 |
| `MEM0_API_KEY` | セマンティックな永続メモリ用のMem0 Platform APIキー（[app.mem0.ai](https://app.mem0.ai)） |
| `MEM0_MODE` | Mem0のバックエンドモード: `platform`（デフォルト）または `oss` — [Memory Providers](../user-guide/features/memory-providers.md) を参照してください |
| `MEM0_HOST` | セルフホストのMem0サーバーのベースURL（プラグインをPlatform APIから切り替えます） |
| `MEM0_USER_ID` | Mem0のメモリを保存するユーザーIDを上書き |
| `MEM0_AGENT_ID` | Mem0のメモリにタグ付けするエージェントIDを上書き |
| `RETAINDB_API_KEY` | 永続メモリ用のRetainDB APIキー（[retaindb.com](https://retaindb.com)） |
| `RETAINDB_BASE_URL` | セルフホストのRetainDBインスタンスのベースURL（デフォルト: `https://api.retaindb.com`） |
| `OPENVIKING_API_KEY` | OpenViking APIキー（ローカル開発モードでは空のままにします） |
| `OPENVIKING_ENDPOINT` | OpenVikingサーバーのURL（デフォルト: `http://127.0.0.1:1933`） |
| `BRV_API_KEY` | ByteRover APIキー（オプション、クラウド同期用 — デフォルトはローカルファースト）（[app.byterover.dev](https://app.byterover.dev)） |
| `SUPERMEMORY_API_KEY` | プロファイルの想起とセッション取り込みを備えたセマンティック長期メモリ（[supermemory.ai](https://supermemory.ai)） |
| `DAYTONA_API_KEY` | Daytonaクラウドサンドボックス（[daytona.io](https://daytona.io/)） |
| `SPRITES_TOKEN` | Spritesクラウドサンドボックス。[`sprites` プラグイン](https://github.com/NousResearch/hermes-plugin-sprites)が読み込みます（[fly.io/sprites](https://fly.io/sprites)） |
| `VERCEL_TOKEN` | Vercel Sandboxアクセストークン（[vercel.com](https://vercel.com/)） |
| `VERCEL_PROJECT_ID` | VercelプロジェクトID（`VERCEL_TOKEN` と併用必須） |
| `VERCEL_TEAM_ID` | VercelチームID（`VERCEL_TOKEN` と併用必須） |
| `VERCEL_OIDC_TOKEN` | Vercelの短命OIDCトークン（開発専用の代替） |

### スキルAPIキー

特定のバンドル済み / オプションのスキルが使用するシークレットです。いずれも、対応するスキルを使用する場合にのみ必要です。

| Variable | Used by skill | Description |
|----------|---------------|-------------|
| `NOTION_API_KEY` | Teams会議パイプライン | Notionシンク用のNotionインテグレーショントークン。 |
| `LINEAR_API_KEY` | `linear` | Linearの個人APIキー。 |
| `TENOR_API_KEY` | `gif-search` | GIF検索用のTenor APIキー。 |

### Langfuse可観測性

公式の [`langfuse`](https://github.com/NousResearch/hermes-official-plugins/tree/main/langfuse) カタログプラグイン（以前は `observability/langfuse` としてバンドル）用の環境変数です。これらは `~/.hermes/.env` に設定します。これらが有効になる前に、プラグインをインストール（`hermes plugins install langfuse`）しておく必要があります。

| Variable | Description |
|----------|-------------|
| `HERMES_LANGFUSE_PUBLIC_KEY` | Langfuseプロジェクトの公開キー（`pk-lf-...`）。必須。 |
| `HERMES_LANGFUSE_SECRET_KEY` | Langfuseプロジェクトのシークレットキー（`sk-lf-...`）。必須。 |
| `HERMES_LANGFUSE_BASE_URL` | LangfuseサーバーのURL（デフォルト: `https://cloud.langfuse.com`）。セルフホスト時に設定します。 |
| `HERMES_LANGFUSE_ENV` | トレースの環境タグ（`production`、`staging` など） |
| `HERMES_LANGFUSE_RELEASE` | トレースのリリース/バージョンタグ |
| `HERMES_LANGFUSE_SAMPLE_RATE` | SDKのサンプリングレート 0.0〜1.0（デフォルト: `1.0`） |
| `HERMES_LANGFUSE_MAX_CHARS` | シリアライズされたペイロードのフィールドごとの切り詰め（デフォルト: `12000`） |
| `HERMES_LANGFUSE_MAX_DEPTH` | キャプチャされたツールの入出力で、値が `<max-depth>` になる前に保持されるネストの深さ（デフォルト: `4`。無効な値の場合は警告を出してデフォルトを維持） |
| `HERMES_LANGFUSE_CAPTURE` | コンテンツのキャプチャモード: `metadata`（コンテンツなし）、`sanitized`（デフォルト: シークレットの墨消し + 切り詰め）、または `full`（切り詰めのみ） |
| `HERMES_LANGFUSE_DEBUG` | `true` で `agent.log` への詳細なプラグインログを有効化 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | 標準のLangfuse SDK名。`HERMES_LANGFUSE_*` 相当が未設定の場合にフォールバックとして受け入れられます。 |

### Nous Tool Gateway

これらの変数は、有料Nousサブスクライバーまたはセルフホストのゲートウェイデプロイメント向けに [Tool Gateway](../user-guide/features/tool-gateway.md) を設定します。ほとんどのユーザーはこれらを設定する必要はありません — ゲートウェイは `hermes model` または `hermes tools` によって自動的に設定されます。

| Variable | Description |
|----------|-------------|
| `TOOL_GATEWAY_DOMAIN` | Tool Gatewayルーティングのベースドメイン（デフォルト: `nousresearch.com`） |
| `TOOL_GATEWAY_SCHEME` | ゲートウェイURLのHTTPまたはHTTPSスキーム（デフォルト: `https`） |
| `TOOL_GATEWAY_USER_TOKEN` | Tool Gatewayの認証トークン（通常はNous認証から自動入力） |
| `FIRECRAWL_GATEWAY_URL` | Firecrawlゲートウェイエンドポイント専用のURLを上書き |

## ターミナルバックエンド

| Variable | Description |
|----------|-------------|
| `TERMINAL_ENV` | バックエンド: `local`、`docker`、`ssh`、`modal`、`daytona`、`vercel_sandbox`、またはプラグインバックエンド（`singularity`、`sprites`） |
| `HERMES_DOCKER_BINARY` | Hermesがシェルアウトするコンテナバイナリを上書き（例: `podman`、`/usr/local/bin/docker`）。未設定の場合、Hermesは `PATH` 上の `docker` または `podman` を自動検出します。両方インストールされていてデフォルト以外を使いたい場合や、バイナリが `PATH` の外にある場合に必要です。 |
| `TERMINAL_DOCKER_IMAGE` | Dockerイメージ（デフォルト: `nousresearch/hermes-sandbox:desktop`） |
| `TERMINAL_DOCKER_FORWARD_ENV` | Dockerターミナルセッションに明示的に転送する環境変数名のJSON配列。注: スキルが宣言した `required_environment_variables` は自動的に転送されます — どのスキルでも宣言されていない変数の場合のみ必要です。 |
| `TERMINAL_DOCKER_VOLUMES` | 追加のDockerボリュームマウント（カンマ区切りの `host:container` ペア） |
| `TERMINAL_DOCKER_ENV` | Dockerターミナルセッション内で設定する追加の環境変数のJSONオブジェクト（例: `{"FOO":"bar"}`） |
| `TERMINAL_DOCKER_EXTRA_ARGS` | 追加の `docker run` 引数のJSON配列（例: `["--memory","4g"]`） |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | 高度なオプトイン: 起動時のcwdをDockerの `/workspace` にマウント（`true`/`false`、デフォルト: `false`） |
| `TERMINAL_SINGULARITY_IMAGE` | Singularityイメージまたは `.sif` パス（[`singularity` プラグイン](https://github.com/NousResearch/hermes-official-plugins/tree/main/singularity)が読み込みます） |
| `TERMINAL_MODAL_IMAGE` | Modalコンテナイメージ |
| `TERMINAL_DAYTONA_IMAGE` | Daytonaサンドボックスイメージ |
| `TERMINAL_VERCEL_RUNTIME` | Vercel Sandboxランタイム（`node24`、`node22`、`python3.13`） |
| `TERMINAL_TIMEOUT` | コマンドのタイムアウト（秒単位） |
| `TERMINAL_LIFETIME_SECONDS` | ターミナルセッションの最大存続期間（秒単位） |
| `TERMINAL_CWD` | ゲートウェイ/cronのターミナルセッション用の非推奨の直接上書き。`config.yaml` の `terminal.cwd` を優先してください。CLIは引き続き起動ディレクトリを使用します。 |
| `SUDO_PASSWORD` | 対話的プロンプトなしでsudoを有効化。コマンドを実行するバックエンドに送信されます。[Sudo Support](../user-guide/features/tools.md#sudo-support) を参照してください |

クラウドサンドボックスバックエンドの場合、永続性はファイルシステム指向です。`TERMINAL_LIFETIME_SECONDS` はHermesがアイドル状態のターミナルセッションをクリーンアップするタイミングを制御し、後で再開する際は同じライブプロセスを実行し続けるのではなく、サンドボックスを再作成する場合があります。

## SSHバックエンド

| Variable | Description |
|----------|-------------|
| `TERMINAL_SSH_HOST` | リモートサーバーのホスト名 |
| `TERMINAL_SSH_USER` | SSHユーザー名 |
| `TERMINAL_SSH_PORT` | SSHポート（デフォルト: 22） |
| `TERMINAL_SSH_KEY` | 秘密鍵へのパス |

## コンテナリソース（Docker、Modal、Daytona、コンテナプラグインバックエンド）

| Variable | Description |
|----------|-------------|
| `TERMINAL_CONTAINER_CPU` | CPUコア数（デフォルト: 1） |
| `TERMINAL_CONTAINER_MEMORY` | メモリ（MB単位、デフォルト: 5120） |
| `TERMINAL_CONTAINER_DISK` | ディスク（MB単位、デフォルト: 51200） |
| `TERMINAL_CONTAINER_PERSISTENT` | セッションをまたいでコンテナのファイルシステムを永続化（デフォルト: `true`） |
| `TERMINAL_SANDBOX_DIR` | ワークスペースとオーバーレイ用のホストディレクトリ（デフォルト: `~/.hermes/sandboxes/`） |

## 永続シェル

有効な間は、エクスポートされた環境変数がコマンドから次のコマンドへ引き継がれます。作業ディレクトリは常に引き継がれます。[Persistent Shell](../user-guide/configuration.md#persistent-shell) を参照してください。

| Variable | Description |
|----------|-------------|
| `TERMINAL_PERSISTENT_SHELL` | SSHバックエンド: エクスポートされた環境変数をコマンド間で引き継ぐ（デフォルト: `true`）。config.yamlの `terminal.persistent_shell` でも設定可能 |
| `TERMINAL_SSH_PERSISTENT` | SSHバックエンドの上書き。`TERMINAL_PERSISTENT_SHELL` より優先されます（デフォルト: それに従う） |
| `TERMINAL_LOCAL_PERSISTENT` | ローカルバックエンド: エクスポートされた環境変数をコマンド間で引き継ぐ（デフォルト: `true`） |

## Egressプロキシ（サンドボックスに注入）

これらの環境変数はホスト上では設定されません — `proxy.enabled: true` の場合に、[Egress proxy](/docs/user-guide/egress/iron-proxy) 統合によってDockerサンドボックスに注入されます。このリリースで対応しているバックエンドはDockerのみです。

| Variable | Description |
|----------|-------------|
| `HERMES_EGRESS_PROXY` | Egressプロキシが有効な場合、サンドボックス内で `1` に設定されます。エージェントのコードはこれをチェックして、TLSを傍受するプロキシの背後で実行されていることを知ることができます。 |
| プロバイダーの環境変数（`OPENROUTER_API_KEY`、`OPENAI_API_KEY`、…） | 実際の上流シークレットではなく不透明なプロキシトークンに設定されるため、既存のSDKは標準の環境変数名を引き続き読み込めます。iron-proxyはネットワーク境界でこれらのトークンを実際の上流シークレットに置き換えます。 |
| `HERMES_PROXY_TOKEN_<ENV_NAME>` | 発行された各プロバイダーマッピングの診断用エイリアス。例: `HERMES_PROXY_TOKEN_OPENROUTER_API_KEY=hermes-proxy-openrouter-…`。標準のプロバイダー環境変数と同じトークン値です。 |
| `HTTPS_PROXY` / `HTTP_PROXY` | `HTTPS_PROXY` はCONNECT/MITM用に `http://host.docker.internal:<tunnel_port>` を指します。`HTTP_PROXY` はプレーンHTTP転送用に `<tunnel_port + 1>` を指します。 |
| `NO_PROXY` | `127.0.0.1,localhost,::1`。サンドボックス内のループバック開発サーバーがプロキシをバイパスするようにします。 |
| `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE` / `NODE_EXTRA_CA_CERTS` | サンドボックス内にマウントされたHermes egress CA証明書へのパス（`/etc/ssl/certs/hermes-egress-ca.crt`）。各言語ランタイムが、iron-proxyがMITMで発行したリーフ証明書を信頼できるようにします。 |
| `NODE_OPTIONS` | `--use-openssl-ca` が追加され（既存のフラグは保持されます）、Node.jsが他のCAバンドル変数が制御するOpenSSLストアを経由するようにします。[Node.js asymmetric CA caveat](/docs/user-guide/egress/iron-proxy#nodejs-asymmetric-ca-caveat) の影響を狭めます。 |
| `HERMES_IRON_PROXY_NONCE` | iron-proxyデーモンプロセス自体に設定されます（サンドボックス内ではありません）。PIDが再利用されても、候補のPIDが引き続き*自分たちの*管理するバイナリを指していることを `_pid_alive` が確認するために使用されます。 |

これらは、`proxy.enabled: true` かつデーモンが実行中の場合に、Dockerターミナルバックエンドによって自動的に設定されます。自分で設定する必要はありません。運用者向けの関連設定は `~/.hermes/config.yaml` の `proxy:` セクションにあります — [Egress proxy → Configuration](/docs/user-guide/egress/iron-proxy#configuration) を参照してください。

## メッセージング

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegramボットトークン（@BotFatherから取得） |
| `TELEGRAM_ALLOWED_USERS` | ボットの使用を許可するユーザーIDのカンマ区切りリスト（DM、グループ、フォーラムに適用） |
| `TELEGRAM_ALLOW_ALL_USERS` | 任意のTelegramユーザーがボットをトリガーすることを許可（開発専用）。 |
| `TELEGRAM_GROUP_ALLOWED_USERS` | グループ/フォーラムでのみ認可される送信者ユーザーIDのカンマ区切りリスト（DMアクセスは付与しません）。チャットIDの形式の値（`-` で始まる）は、#17686以前の設定との後方互換性のため引き続きチャットIDとして扱われますが、非推奨の警告が表示されます。 |
| `TELEGRAM_GROUP_ALLOWED_CHATS` | グループ/フォーラムのチャットIDのカンマ区切りリスト。メンバー全員が認可されます |
| `TELEGRAM_HOME_CHANNEL` | cron配信用のデフォルトTelegramチャット/チャンネル |
| `TELEGRAM_HOME_CHANNEL_NAME` | Telegramホームチャンネルの表示名 |
| `TELEGRAM_CRON_THREAD_ID` | cron配信を受け取るフォーラムトピックID。cronに限り `TELEGRAM_HOME_CHANNEL_THREAD_ID` を上書きします。トピックモードで使用すると、cronメッセージへの返信がシステムロビーに届く代わりに新しいセッションを開始します（#24409）。 |
| `TELEGRAM_WEBHOOK_URL` | Webhookモード用の公開HTTPS URL（ポーリングの代わりにWebhookを有効化） |
| `TELEGRAM_WEBHOOK_PORT` | Webhookサーバーのローカルリッスンポート（デフォルト: `8443`） |
| `TELEGRAM_WEBHOOK_HOST` | Webhookサーバーのローカルバインドホスト。デフォルト: 未設定 → すべてのインターフェースでデュアルスタック（IPv4+IPv6）でバインド。IPv6専用のデプロイメント（例: Fly.io 6PN）では明示的に設定してください。`platforms.telegram.extra.webhook_host` と同等です。 |
| `TELEGRAM_WEBHOOK_SECRET` | 検証のためにTelegramが各更新で返すシークレットトークン。**`TELEGRAM_WEBHOOK_URL` が設定されている場合は常に必須** — これがないとゲートウェイは起動を拒否します（GHSA-3vpc-7q5r-276h）。`openssl rand -hex 32` で生成します。 |
| `TELEGRAM_ALLOW_BOTS` | 他のボットからのメッセージを処理することを許可（`none` = すべてのボットメッセージを拒否、`mentions` = ボットが@メンションされた場合のみ許可、`all` = すべてのボットメッセージを許可。デフォルト: `none`）。`telegram.allow_bots` と同等です。 |
| `TELEGRAM_GUEST_MODE` | 許可リストにないグループが、毎ターン明示的な@メンションによってのみボットをトリガーできるようにします（デフォルト: `false`）。`telegram.guest_mode` と同等です。 |
| `TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES` | メンションのないグループメッセージを共有セッションコンテキストに受動的に取り込みます。許可リストに含まれるチャット/トピックのみが対象です（デフォルト: `false`）。`telegram.observe_unmentioned_group_messages` と同等です。 |
| `TELEGRAM_REACTIONS` | 処理中のメッセージへの絵文字リアクションを有効化（デフォルト: `false`） |
| `TELEGRAM_REQUIRE_MENTION` | Telegramグループで応答する前に明示的なトリガーを必須にします。`config.yaml` の `telegram.require_mention` と同等です。 |
| `TELEGRAM_MENTION_PATTERNS` | Telegramグループのメンションゲーティングが有効な場合に受け付けるウェイクワードの正規表現パターン。JSON配列、改行区切りリスト、またはカンマ区切りリストで指定します。`telegram.mention_patterns` と同等です。 |
| `TELEGRAM_EXCLUSIVE_BOT_MENTIONS` | 有効にすると、Telegramグループ内の明示的な `@...bot` メンションは、返信やウェイクワードのフォールバックが実行される前に、メンションされたボットのユーザー名にのみルーティングされます。デフォルト: `true`。`telegram.exclusive_bot_mentions` と同等です。 |
| `TELEGRAM_BOTS_REQUIRE_MENTION` | 有効にすると、他のボットが送信したメッセージで応答をトリガーするには明示的に `@thisbot` する必要があります — 引用返信だけでは無視されるため、2つのボットが互いに永遠に返信し合うのを防げます。人間の返信には影響しません。デフォルト: `false`。`telegram.bots_require_mention` と同等です。 |
| `TELEGRAM_REPLY_TO_MODE` | 返信参照の動作: `off`、`first`（デフォルト）、または `all`。Discordのパターンに合わせています。 |
| `TELEGRAM_IGNORED_THREADS` | ボットが応答しないTelegramフォーラムトピック/スレッドIDのカンマ区切りリスト |
| `TELEGRAM_ALLOWED_TOPICS` | ボットが処理するTelegramフォーラムトピック/スレッドIDのカンマ区切りリスト。設定すると、それ以外のグループ/フォーラムトピックからのメッセージは無視されます（チャット/トピックごとに最初の破棄がINFOでログ出力されます）。DMはフィルタリングされません。TelegramのGeneralトピックはID `1` に正規化されます。`TELEGRAM_IGNORED_THREADS` より前に適用されます。`config.yaml` の `telegram.allowed_topics` と同等です。 |
| `TELEGRAM_FREE_RESPONSE_CHATS` | `require_mention` をバイパスし、Hermesがすべてのグループメッセージに応答するチャットIDのカンマ区切りリスト。`telegram.free_response_chats` と同等です。 |
| `TELEGRAM_FREE_RESPONSE_TOPICS` | 単一のフォーラムトピックに限って `require_mention` をバイパスする `<chat_id>:<thread_id>` ペアのカンマ区切りリスト。TelegramのGeneralトピックはスレッドID `1` に正規化されます。`telegram.free_response_topics` と同等です。 |
| `TELEGRAM_PROXY` | Telegram接続用のプロキシURL — `HTTPS_PROXY` を上書き。`http://`、`https://`、`socks5://` をサポート |
| `DISCORD_BOT_TOKEN` | Discordボットトークン |
| `DISCORD_ALLOWED_USERS` | ボットの使用を許可するDiscordユーザーIDのカンマ区切りリスト |
| `DISCORD_ALLOW_ALL_USERS` | 任意のDiscordユーザーがボットをトリガーすることを許可（開発専用）。 |
| `DISCORD_ALLOWED_ROLES` | ボットの使用を許可するDiscordロールIDのカンマ区切りリスト（`DISCORD_ALLOWED_USERS` とのOR）。Membersインテントを自動的に有効化します。モデレーションチームの入れ替わりがある場合に便利です — ロールの付与は自動的に反映されます。 |
| `DISCORD_ALLOWED_CHANNELS` | DiscordチャンネルIDのカンマ区切りリスト。設定すると、ボットはこれらのチャンネル（許可されていればDMも）でのみ応答します。`config.yaml` の `discord.allowed_channels` を上書きします。 |
| `DISCORD_PROXY` | Discord接続用のプロキシURL — `HTTPS_PROXY` を上書き。`http://`、`https://`、`socks5://` をサポート |
| `DISCORD_HOME_CHANNEL` | cron配信用のデフォルトDiscordチャンネル |
| `DISCORD_HOME_CHANNEL_NAME` | Discordホームチャンネルの表示名 |
| `DISCORD_COMMAND_SYNC_POLICY` | Discordスラッシュコマンドの起動時同期ポリシー: `safe`（差分を取って調整）、`bulk`（レガシーの `tree.sync()`）、または `off` |
| `DISCORD_REQUIRE_MENTION` | サーバーチャンネルで応答する前に@メンションを必須にする |
| `DISCORD_FREE_RESPONSE_CHANNELS` | メンションが不要なチャンネルIDのカンマ区切りリスト |
| `DISCORD_AUTO_THREAD` | サポートされている場合に長い返信を自動的にスレッド化 |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | `true` の場合、任意のファイル形式の添付ファイルを受け付けます（組み込みのPDF/テキスト/zip/Office許可リストに限りません）。不明な形式はキャッシュされ、ローカルパスとしてエージェントに渡されるため、`terminal` / `read_file` / `ffprobe` で調べることができます。デフォルト `false`。 |
| `DISCORD_MAX_ATTACHMENT_BYTES` | ゲートウェイがキャッシュする添付ファイル1件あたりの最大バイト数。デフォルト `33554432`（32 MiB）。`0` に設定すると上限なしになります（添付ファイルは書き込み中メモリに保持されます）。 |
| `DISCORD_FREE_RESPONSE_AUTO_THREAD` | `true` の場合、フリーレスポンスチャンネル（`DISCORD_FREE_RESPONSE_CHANNELS` に記載）でも、トップレベルのメッセージごとにスレッドを自動作成します。デフォルト `false` — フリーレスポンスチャンネルではインラインで返信します。`DISCORD_AUTO_THREAD=true` が必要です。`DISCORD_NO_THREAD_CHANNELS` が引き続き優先されます。 |
| `DISCORD_REACTIONS` | 処理中のメッセージへの絵文字リアクションを有効化（デフォルト: `true`） |
| `DISCORD_IGNORED_CHANNELS` | ボットが応答しないチャンネルIDのカンマ区切りリスト |
| `DISCORD_NO_THREAD_CHANNELS` | ボットが自動スレッド化せずに応答するチャンネルIDのカンマ区切りリスト |
| `DISCORD_REPLY_TO_MODE` | 返信参照の動作: `off`、`first`（デフォルト）、または `all` |
| `DISCORD_ALLOW_MENTION_EVERYONE` | ボットが `@everyone`/`@here` にpingすることを許可（デフォルト: `false`）。[メンション制御](../user-guide/messaging/discord.md#mention-control)を参照してください。 |
| `DISCORD_ALLOW_MENTION_ROLES` | ボットが `@role` メンションにpingすることを許可（デフォルト: `false`）。 |
| `DISCORD_ALLOW_MENTION_USERS` | ボットが個別の `@user` メンションにpingすることを許可（デフォルト: `true`）。 |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | メッセージに返信する際に作成者にpingする（デフォルト: `true`）。 |
| `DISCORD_MISSED_MESSAGE_BACKFILL` | `discord.missed_message_backfill.enabled` の環境変数フォールバック: 切断中に見逃したメッセージを再生します（デフォルト: `false`）。[Missed message backfill](../user-guide/messaging/discord.md#discordmissed_message_backfill) を参照してください。 |
| `DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS` | スキャンするチャンネルIDのカンマ区切りリスト（`discord.missed_message_backfill.channels` のフォールバック。空 = `discord.free_response_channels`、`*` = 到達可能なすべてのテキストチャンネル）。 |
| `DISCORD_MISSED_MESSAGE_BACKFILL_WINDOW_SECONDS` | スキャンがさかのぼれる期間（`window_seconds` のフォールバック、デフォルト `21600`、最小 `60`）。 |
| `DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT` | 1回のスキャンでチャンネルごとに取得する最大メッセージ数（`limit` のフォールバック、デフォルト `100`、1〜500）。 |
| `DISCORD_MISSED_MESSAGE_BACKFILL_MAX_DISPATCHES` | 1回のスキャンで再ディスパッチする最大メッセージ数（`max_dispatches` のフォールバック、デフォルト `10`、1〜100）。 |
| `DISCORD_MISSED_MESSAGE_BACKFILL_MAX_ATTEMPTS` | 再接続をまたいだ、1つのメッセージに対する生涯の再ディスパッチ上限（`max_attempts` のフォールバック、デフォルト `3`、1〜100）。 |
| `SLACK_BOT_TOKEN` | Slackボットトークン（`xoxb-...`） |
| `SLACK_APP_TOKEN` | Slackアプリレベルトークン（`xapp-...`、Socket Modeに必要） |
| `SLACK_ALLOWED_USERS` | SlackユーザーIDのカンマ区切りリスト |
| `SLACK_ALLOW_ALL_USERS` | 任意のSlackユーザーがボットをトリガーすることを許可（開発専用）。 |
| `SLACK_ALLOW_BOTS` | 他のSlackボットからのメッセージを受け付けます: `none`（デフォルト）、`mentions`、または `all`。ボットは常に自身のメッセージを無視します。 |
| `SLACK_THREAD_REQUIRE_MENTION` | トップレベルのフリーレスポンスチャンネルを維持しつつ、Slackのスレッド返信に明示的な@メンションを必須にします |
| `SLACK_THREAD_FOLLOWUP_SILENCE` | 宛先のないSlackチャンネルスレッドの返信で、予期しない沈黙の通知を投稿する代わりに、単独の沈黙マーカーをそのまま残します（デフォルト `false`） |
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
| `GOOGLE_CHAT_HTTP_EVENTS_URL` | Chatメッセージイベント用の認証付きHTTPエンドポイント（Pub/Subの代替） |
| `GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE` | Googleが署名したHTTPイベントのBearerトークンで期待されるオーディエンス（デフォルトは `GOOGLE_CHAT_HTTP_EVENTS_URL`） |
| `GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL` | HTTPイベントのBearerトークンで期待されるGoogleサービスアカウントのメールアドレス |
| `WHATSAPP_ENABLED` | WhatsAppブリッジを有効化（`true`/`false`） |
| `WHATSAPP_MODE` | `bot`（別番号）または `self-chat`（自分自身にメッセージ） |
| `WHATSAPP_ALLOWED_USERS` | 電話番号のカンマ区切りリスト（国番号付き、`+` なし）、またはすべての送信者を許可する `*` |
| `WHATSAPP_ALLOW_ALL_USERS` | 許可リストなしですべてのWhatsApp送信者を許可（`true`/`false`） |
| `WHATSAPP_GROUP_POLICY` | グループの受け入れ: `pairing`（デフォルト、グループからは何も転送しない）、`allowlist`（下記のグループJID）、`open`（すべてのグループ。参加者には引き続き `WHATSAPP_ALLOWED_USERS`、ペアリング、または `WHATSAPP_ALLOW_ALL_USERS` が必要）、または `disabled` |
| `WHATSAPP_GROUP_ALLOWED_USERS` | `WHATSAPP_GROUP_POLICY=allowlist` の下で受け入れるグループJIDのカンマ区切りリスト（例: `120363001234567890@g.us`） |
| `WHATSAPP_HOME_CHANNEL` | cron / 通知配信用のデフォルトチャットID。 |
| `WHATSAPP_HOME_CHANNEL_NAME` | WhatsAppホームチャンネルの表示名。 |
| `WHATSAPP_DEBUG` | トラブルシューティング用にブリッジで生のメッセージイベントをログ出力（`true`/`false`） |
| `HERMES_WHATSAPP_BRIDGE_TOKEN` | ゲートウェイがブリッジのサブプロセスにのみ設定します（`.env` には決して記述しません）: `platforms/whatsapp/session/.bridge-token` にあるセッションごとのシークレットで、ブリッジのすべてのHTTPルートがBearerトークンとして要求します。これなしで起動したブリッジはリクエストの処理を拒否します。 |
| `WHATSAPP_CLOUD_PHONE_NUMBER_ID` | WhatsApp Business Cloud APIのMeta Phone Number ID（15〜17桁。電話番号そのものでは**ありません**） |
| `WHATSAPP_CLOUD_ACCESS_TOKEN` | Metaのアクセストークン（`EAA` で始まる）。一時トークンは24時間で失効し、System Userトークンは無期限です |
| `WHATSAPP_CLOUD_APP_SECRET` | 受信Webhookの署名検証に使用する32文字の16進数アプリシークレット |
| `WHATSAPP_CLOUD_VERIFY_TOKEN` | MetaのWebhook検証ハンドシェイク用の共有シークレット（セットアップウィザードが自動生成） |
| `WHATSAPP_CLOUD_ALLOWED_USERS` | ボットへのメッセージを許可する `wa_id`（国番号付き、`+` なしの電話番号）のカンマ区切りリスト |
| `WHATSAPP_CLOUD_ALLOW_ALL_USERS` | 許可リストなしですべてのWhatsApp Cloud送信者を許可（`true`/`false`） |
| `WHATSAPP_CLOUD_APP_ID` | オプションのMeta App ID（将来の分析統合用） |
| `WHATSAPP_CLOUD_WABA_ID` | オプションのWhatsApp Business Account ID。設定すると、受信Webhookはこれと一致する必要があります |
| `WHATSAPP_CLOUD_WEBHOOK_HOST` | 受信Webhookサーバーがバインドするインターフェース（デフォルト `0.0.0.0`） |
| `WHATSAPP_CLOUD_WEBHOOK_PORT` | 受信Webhookサーバーがバインドするポート（デフォルト `8090`） |
| `WHATSAPP_CLOUD_WEBHOOK_PATH` | Metaが受信メッセージをPOSTするURLパス（デフォルト `/whatsapp/webhook`） |
| `WHATSAPP_CLOUD_API_VERSION` | 呼び出すMeta Graph APIのバージョン（デフォルト `v20.0`） |
| `WHATSAPP_CLOUD_HOME_CHANNEL` | ボットのホームチャンネルとして使用する `wa_id`（cronジョブなど用） |
| `WHATSAPP_CLOUD_DM_POLICY` | Cloudアダプター用のDMゲーティング（`open`/`allowlist`/`disabled`）。未設定の場合は `WHATSAPP_DM_POLICY` にフォールバック |
| `WHATSAPP_CLOUD_ALLOW_FROM` | `dm_policy: allowlist` の場合に許可する送信者のカンマ区切りリスト（素の `wa_id`。Baileys形式のJIDは正規化されます） |
| `WHATSAPP_CLOUD_GROUP_POLICY` | Cloudアダプター用のグループゲーティング（`open`/`allowlist`/`disabled`）。未設定の場合は `WHATSAPP_GROUP_POLICY` にフォールバック |
| `WHATSAPP_CLOUD_GROUP_ALLOW_FROM` | `group_policy: allowlist` の場合に許可するグループチャットIDのカンマ区切りリスト |
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
| `SMS_WEBHOOK_HOST` | Webhookのバインドアドレス（デフォルト: `127.0.0.1`） |
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
| `EMAIL_AUTHSERV_ID` | 受信サーバーの最上位の `Authentication-Results` ヘッダーにある正確なauthserv-id。送信者認証を無効にしている場合（`EMAIL_TRUST_FROM_HEADER=true`）を除き必須 |
| `EMAIL_HOME_ADDRESS` | プロアクティブなメール配信のデフォルト受信者 |
| `EMAIL_HOME_ADDRESS_NAME` | メールホームターゲットの表示名 |
| `EMAIL_POLL_INTERVAL` | メールのポーリング間隔（秒単位） |
| `EMAIL_ALLOW_ALL_USERS` | すべての受信メール送信者を許可 |
| `DINGTALK_CLIENT_ID` | デベロッパーポータルから取得するDingTalkボットのAppKey（[open.dingtalk.com](https://open.dingtalk.com)） |
| `DINGTALK_CLIENT_SECRET` | デベロッパーポータルから取得するDingTalkボットのAppSecret |
| `DINGTALK_ALLOWED_USERS` | ボットへのメッセージを許可するDingTalkユーザーIDのカンマ区切りリスト |
| `DINGTALK_WEBHOOK_URL` | クロスプラットフォーム / cron配信用の静的なロボットWebhook URL。 |
| `DINGTALK_HOME_CHANNEL` | cron / 通知配信用のデフォルト会話ID。 |
| `DINGTALK_HOME_CHANNEL_NAME` | DingTalkホームチャンネルの表示名。 |
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
| `FEISHU_HOME_CHANNEL_NAME` | Feishuホームチャンネルの表示名。 |
| `FEISHU_ALLOW_ALL_USERS` | 任意のFeishuユーザーがボットをトリガーすることを許可（開発専用）。 |
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
| `QQ_SANDBOX` | 開発テスト用にQQサンドボックスモードを有効化（`true`/`false`） |
| `MATTERMOST_URL` | MattermostサーバーのURL（例: `https://mm.example.com`） |
| `MATTERMOST_TOKEN` | Mattermost用のボットトークンまたは個人アクセストークン |
| `MATTERMOST_ALLOWED_USERS` | ボットへのメッセージを許可するMattermostユーザーIDのカンマ区切りリスト |
| `MATTERMOST_ALLOW_ALL_USERS` | 任意のMattermostユーザーがボットをトリガーすることを許可（開発専用）。 |
| `MATTERMOST_ALLOWED_CHANNELS` | 設定すると、ボットはこれらのチャンネルでのみ応答します（ホワイトリスト）。 |
| `MATTERMOST_HOME_CHANNEL` | プロアクティブなメッセージ配信（cron、通知）用のチャンネルID |
| `MATTERMOST_REQUIRE_MENTION` | チャンネルで `@mention` を必須にする（デフォルト: `true`）。`false` に設定するとすべてのメッセージに応答します。 |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | ボットが `@mention` なしで応答するチャンネルIDのカンマ区切りリスト |
| `MATTERMOST_REPLY_MODE` | 返信スタイル: `thread`（スレッド化された返信）または `off`（フラットなメッセージ、デフォルト） |
| `MATRIX_HOMESERVER` | MatrixホームサーバーのURL（例: `https://matrix.org`） |
| `MATRIX_ACCESS_TOKEN` | ボット認証用のMatrixアクセストークン |
| `MATRIX_USER_ID` | MatrixユーザーID（例: `@hermes:matrix.org`） — パスワードログインには必須、アクセストークンを使う場合は任意 |
| `MATRIX_PASSWORD` | Matrixパスワード（アクセストークンの代替） |
| `MATRIX_ALLOWED_USERS` | ボットへのメッセージを許可するMatrixユーザーIDのカンマ区切りリスト（例: `@alice:matrix.org`） |
| `MATRIX_ALLOW_ALL_USERS` | 任意のMatrixユーザーがボットをトリガーすることを許可（開発専用）。 |
| `MATRIX_HOME_CHANNEL` | cron / 通知配信用のデフォルトルームID。 |
| `MATRIX_HOME_CHANNEL_NAME` | Matrixホームルームの表示名。 |
| `MATRIX_ALLOWED_ROOMS` | ボットの応答をトリガーできるMatrixルームIDのカンマ区切りリスト。DMとして自動分類されたルーム（名前に関係なく、参加メンバーが2人以下のルーム）には適用されません — それらでは常に応答します。 |
| `MATRIX_HOME_ROOM` | プロアクティブなメッセージ配信用のルームID（例: `!abc123:matrix.org`） |
| `MATRIX_ENCRYPTION` | エンドツーエンド暗号化を有効化（`true`/`false`、デフォルト: `false`） |
| `MATRIX_E2EE_MODE` | MatrixのE2EE動作: `off`、`optional`、または `required`。設定すると `MATRIX_ENCRYPTION` を上書きします。 |
| `MATRIX_DEVICE_ID` | 再起動をまたいだE2EE永続化用の安定したMatrixデバイスID（例: `HERMES_BOT`）。これがないと、E2EEキーは起動のたびにローテーションされ、過去のルームの復号が壊れます。 |
| `MATRIX_REACTIONS` | 受信メッセージへの処理ライフサイクル絵文字リアクションを有効化（デフォルト: `true`）。`false` に設定すると無効化します。 |
| `MATRIX_REQUIRE_MENTION` | ルームで `@mention` を必須にする（デフォルト: `true`）。`false` に設定するとすべてのメッセージに応答します。参加メンバーが2人以下のルームはDMとして自動分類され、この設定に関係なくメンションは不要です — 意図的に2人にしたルームを通常のルームのように動作させたい場合は、3人目のメンバーを追加してください。 |
| `MATRIX_FREE_RESPONSE_ROOMS` | ボットが `@mention` なしで応答するルームIDのカンマ区切りリスト。DMとして自動分類されたルーム（参加メンバーが2人以下）はもともとメンションなしで応答し、このリストを無視します。 |
| `MATRIX_IGNORE_USER_PATTERNS` | 無視するMatrixブリッジ/appserviceのゴーストユーザーIDの正規表現のカンマ区切りリスト |
| `MATRIX_PROCESS_NOTICES` | 受信したMatrixの `m.notice` イベントを処理（デフォルト: `false`） |
| `MATRIX_SESSION_SCOPE` | プロジェクトルーム用のMatrixセッションスコープ: `auto`、`room`、または `thread`（デフォルト: `auto`） |
| `MATRIX_ALLOW_ROOM_MENTIONS` | 送信する `@room` メンションでルームの全メンバーに通知することを許可（デフォルト: `false`） |
| `MATRIX_AUTO_THREAD` | ルームメッセージ用のスレッドを自動作成（デフォルト: `true`）。DMとして自動分類されたルーム（参加メンバーが2人以下）には適用されません — それらは代わりに `MATRIX_DM_AUTO_THREAD` に従います。 |
| `MATRIX_DM_AUTO_THREAD` | MatrixのDMメッセージ用のスレッドを自動作成（デフォルト: `false`） |
| `MATRIX_DM_MENTION_THREADS` | DMでボットが `@mention` された際にスレッドを作成（デフォルト: `false`） |
| `MATRIX_APPROVAL_REQUIRE_SENDER` | 承認/モデルピッカーのリアクションが、判明している場合は元のリクエスト者からのものであることを必須にする（デフォルト: `true`） |
| `MATRIX_APPROVAL_TIMEOUT_SECONDS` | Matrixのリアクションによる承認/モデルピッカープロンプトのタイムアウト（デフォルト: `300`） |
| `MATRIX_ALLOW_PUBLIC_ROOMS` | Matrixのルーム作成ツールが公開ルームを作成することを許可（デフォルト: `false`） |
| `MATRIX_MAX_MEDIA_BYTES` | Matrixメディアのアップロード/ダウンロードの最大サイズ（バイト単位、デフォルト: `104857600`） |
| `MATRIX_RECOVERY_KEY` | デバイスキーのローテーション後のクロス署名検証用のリカバリーキー。クロス署名を有効にしたE2EE設定で推奨されます。 |
| `MATRIX_RECOVERY_KEY_OUTPUT_FILE` | 生成されたMatrixリカバリーキーを書き出すオプションの1回限りのパス。モード `0600` で作成され、上書きされることはありません。 |
| `HASS_TOKEN` | Home Assistantプラグイン: 長期アクセストークン（HAプラットフォーム + ツールを有効化。`homeassistant` カタログプラグインが必要です。[Home Assistant](../user-guide/messaging/homeassistant.md) を参照） |
| `HASS_URL` | Home Assistantプラグイン: Home AssistantのURL（デフォルト: `http://homeassistant.local:8123`） |
| `HASS_HOME_CHANNEL` | Home Assistantプラグイン: 素の `deliver: homeassistant`（cron、Webhook）のデフォルト通知先 |
| `WEBHOOK_ENABLED` | Webhookプラットフォームアダプターを有効化（`true`/`false`） |
| `WEBHOOK_PORT` | Webhookを受信するHTTPサーバーのポート（デフォルト: `8644`） |
| `WEBHOOK_SECRET` | Webhook署名検証用のグローバルHMACシークレット（ルートが独自のものを指定しない場合のフォールバックとして使用） |
| `API_SERVER_ENABLED` | OpenAI互換APIサーバーを有効化（`true`/`false`）。他のプラットフォームと並行して動作します。 |
| `API_SERVER_KEY` | APIサーバー認証用のBearerトークン。APIサーバーを有効にする場合は常に必須です。 |
| `API_SERVER_CORS_ORIGINS` | APIサーバーを直接呼び出すことを許可するブラウザオリジンのカンマ区切りリスト（例: `http://localhost:3000,http://127.0.0.1:3000`）。デフォルト: 無効。 |
| `API_SERVER_PORT` | APIサーバーのポート（デフォルト: `8642`） |
| `API_SERVER_HOST` | APIサーバーのホスト/バインドアドレス（デフォルト: `127.0.0.1`）。ループバックでも `API_SERVER_KEY` は必須です。ブラウザからのアクセスには狭い `API_SERVER_CORS_ORIGINS` 許可リストを使用してください。 |
| `API_SERVER_MODEL_NAME` | `/v1/models` で公開されるモデル名。デフォルトはプロファイル名（デフォルトプロファイルの場合は `hermes-agent`）。Open WebUIのようなフロントエンドが接続ごとに異なるモデル名を必要とするマルチユーザー設定に便利です。 |
| `GATEWAY_PROXY_URL` | メッセージを転送するリモートHermes APIサーバーのURL（[プロキシモード](../user-guide/messaging/matrix.md#proxy-mode-e2ee-on-macos)）。設定すると、ゲートウェイはプラットフォームのI/Oのみを処理し、すべてのエージェント作業はリモートサーバーに委譲されます。`config.yaml` の `gateway.proxy_url` でも設定可能です。 |
| `GATEWAY_PROXY_KEY` | プロキシモードでリモートAPIサーバーと認証するためのBearerトークン。リモートホストの `API_SERVER_KEY` と一致する必要があります。 |
| `MESSAGING_CWD` | ゲートウェイの作業ディレクトリ用の非推奨の互換フォールバック。`config.yaml` の `terminal.cwd` を優先してください。 |
| `GATEWAY_ALLOWED_USERS` | すべてのプラットフォームで許可するユーザーIDのカンマ区切りリスト |
| `GATEWAY_ALLOW_ALL_USERS` | 許可リストなしですべてのユーザーを許可（`true`/`false`、デフォルト: `false`）。`config.yaml` の `gateway.allow_all_users` でも設定可能です。両方設定されている場合は環境変数が優先されます。 |

### Webダッシュボード & Hermes Desktop {#web-dashboard--hermes-desktop}

[Webダッシュボード](../user-guide/features/web-dashboard.md)の認証と、[Hermes Desktopのリモートバックエンドへの接続](../user-guide/features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend)のための認証です。シークレットのみを記述するという規約に従い、認証情報は `~/.hermes/.env` に記述します。OAuthの `client_id` は `config.yaml` の `dashboard.oauth` の下で設定するのが適切です（環境変数が設定されている場合はそちらが優先されます）。

ダッシュボード認証プロバイダーは3種類が同梱されています。リモートのHermes Desktop接続やインターネットに公開するダッシュボードには、**OAuth（Nous Portal）** プロバイダーを推奨します — `HERMES_DASHBOARD_OAUTH_CLIENT_ID` を設定してください（`hermes dashboard register` で発行します）。バンドルされた**ユーザー名/パスワード**プロバイダー（`HERMES_DASHBOARD_BASIC_AUTH_*`）は、信頼できるLAN上やVPNの背後にあるバックエンド向けの最も手軽な選択肢ですが、パブリックインターネットへの直接公開には適していません。独自のIDプロバイダーで認証するには、**セルフホストOIDC**プロバイダー（`HERMES_DASHBOARD_OIDC_*`）を使用してください。いずれの場合も、非ループバックのバインド（`hermes dashboard --host 0.0.0.0`）では認証ゲートが有効になります。全体像については [Web Dashboard → Authentication](../user-guide/features/web-dashboard.md#authentication-gated-mode) を参照してください。

| Variable | Description |
|----------|-------------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | バンドルされたユーザー名/パスワードのダッシュボード認証プロバイダー（`plugins/dashboard_auth/basic`）用のユーザー名。パスワードと一緒に設定するとプロバイダーが有効になります。`dashboard.basic_auth.username` を上書きします。 |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | basicプロバイダー用の平文パスワード（読み込み時にメモリ上でハッシュ化されます）。設定ファイルの `password_hash` より優先されるため、環境変数でローテーションできます。`dashboard.basic_auth.password` を上書きします。 |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | basicプロバイダー用のscryptパスワードハッシュ（推奨 — 平文が保存されません）。`python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"` で計算します。`dashboard.basic_auth.password_hash` を上書きします。 |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | basicプロバイダーのステートレスなセッショントークンに署名するHMACキー（32バイト以上、base64/hex/raw）。再起動後もセッションを維持したり複数のワーカーにまたがらせたりするには明示的に設定してください。空の場合 → プロセスごとにランダム（再起動のたびにログアウトされます）。`$(openssl rand -base64 32)` のような未展開のシェルテキストを含んだままの値はエラーとともに無視され、代わりにプロセスごとのランダムキーが使用されます。`dashboard.basic_auth.secret` を上書きします。 |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | basicプロバイダーのアクセストークンの有効期間（デフォルト12時間）。`dashboard.basic_auth.session_ttl_seconds` を上書きします。 |
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | ゲート付き/公開ダッシュボード用のOAuthクライアントID（`agent:{instance_id}`）。Nous（`plugins/dashboard_auth/nous`）プロバイダーを有効にします。`dashboard.oauth.client_id` を上書きします。`hermes dashboard register` で発行します。 |
| `HERMES_DASHBOARD_SESSION_TOKEN` | ダッシュボードの機密性の高い `/api` ルート用のプロセスごとのセッショントークン。`hermes dashboard` を起動するランチャー（Desktopシェル、リンク形式の統合）が発行します。親プロセスが注入した値はそのまま保持されます: `~/.hermes/.env` 内の `HERMES_DASHBOARD_SESSION_TOKEN` 行で置き換えられることはありません。未設定の場合、サーバーは起動ごとに新しいトークンを発行します。 |
| `HERMES_DASHBOARD_PUBLIC_URL` | リバースプロキシの背後でダッシュボードにアクセスする完全な公開URL。OAuthコールバックの構築を制御し、その正確なホスト名をHTTP Host/WebSocket Originガードに追加します。また、バックエンドがループバックにバインドしている場合でも、非ループバックの公開ホストに対して認証ゲートを必須にします。`dashboard.public_url` を上書きします。 |
| `HERMES_DASHBOARD_OIDC_ISSUER` | バンドルされたセルフホストOIDCプロバイダー（`plugins/dashboard_auth/self_hosted`）用のOIDC発行者URL。有効化に必須です。`dashboard.oauth.self_hosted.issuer` を上書きします。 |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | セルフホストOIDCプロバイダー用の公開OIDCクライアントID（認可コード + PKCE）。有効化に必須です。`dashboard.oauth.self_hosted.client_id` を上書きします。 |
| `HERMES_DASHBOARD_OIDC_SCOPES` | セルフホストOIDCプロバイダーで要求するOIDCスコープ（デフォルト `openid profile email`）。`dashboard.oauth.self_hosted.scopes` を上書きします。 |
| `HERMES_DESKTOP_REMOTE_URL` | （Desktop側）リモートバックエンドのベースURL、例: `http://host:9119`。設定すると、アプリ内のGateway URLを上書きします。サインインは引き続きGateway設定パネルから行います（バックエンドが提示する方式に応じて、OAuthリダイレクトまたはユーザー名/パスワード）。 |
| `HERMES_DESKTOP_HERMES` | Desktopのバックエンドコマンドの上書き。パッケージャー/Nixやトラブルシューティングで、変更可能な管理インストールを確認する前に、Electronを特定の `hermes` 実行ファイルに向けるために使用します。 |
| `HERMES_DESKTOP_HERMES_ROOT` | `hermes desktop --hermes-root` が使用するDesktopのソースチェックアウトの上書き。パッケージ化された初回起動時のインストールや `PATH` 上の既存の `hermes` より先にチェックされます。 |
| `HERMES_DESKTOP_IGNORE_EXISTING` | `1` に設定すると、バックエンドの解決時にDesktopがインストール済みのランタイム（`~/.hermes/hermes-agent`、Windowsでは `%LOCALAPPDATA%\hermes\hermes-agent`）をスキップするため、ローカルバックエンドは起動せず、Desktopは接続またはインストールの選択肢を表示します。バンドルされたランタイム、`HERMES_DESKTOP_HERMES_ROOT`、パッケージ化されていないソースチェックアウト、および `HERMES_DESKTOP_HERMES` は引き続き優先されます。この起動中にインストールされたランタイムは使用されます。`hermes desktop --ignore-existing` と同等です。 |
| `HERMES_DESKTOP_CWD` | Desktopのチャットセッションの初期プロジェクトディレクトリ。`hermes desktop --cwd` によって設定されます。 |
| `HERMES_DESKTOP_PYTHON` | バックエンド用のPythonインタープリターへの絶対パス。Electronがソースチェックアウト用に自動解決する前にチェックされます。ワークツリーの開発ヘルパー（[TUI & Desktop from Worktrees](/docs/developer-guide/worktree-ui-dev) を参照）が共有venvを再利用するために使用します。 |
| `HERMES_DESKTOP_DEV_SERVER` | Electronシェルがパッケージ化されたバンドルの代わりに読み込むVite開発サーバーのURL（例: `http://127.0.0.1:5174`）。`pnpm run dev` によって自動的に設定されます。アプリを改造する場合にのみ関係します。これを継承したバックエンドは、CORSとWebSocketアップグレードについてそのレンダラーのオリジンも信頼します。 |
| `HERMES_DESKTOP_CDP_PORT` | DOM/CSS検査ツール用にレンダラーが `127.0.0.1` で公開するChrome DevTools Protocolポートを上書きします（デフォルト `9222`）。開発サーバーでの実行（`pnpm run dev`、`hgui`）では自動的に開かれます。パッケージ化されたアプリでは決して開かれず、ここでの値によってそれが変わることはありません。開発実行で無効にするには `off` に設定します。このポートに到達できるものは何でも、レンダラー内でコードを実行できます。 |
| `HERMES_DESKTOP_NVIDIA_SWIFTSHADER` | （Linux上のDesktop、NVIDIAのみ）EGLプローブが壊れているドライバーシリーズ（`580.x`、#40077）でレンダリングをSwiftShader経由にするEGLフォールバックの上書き: `1` はフォールバックを強制的に有効にします — 将来のシリーズでクラッシュが再発したものの、まだ既知のリストに含まれていない場合の回避手段です。`0` はオプトアウトしてネイティブのGPUパスを使用します（自己責任）。別のゲートですでにGPUが無効になっている場合（リモートディスプレイ、WSLg、`HERMES_DESKTOP_DISABLE_GPU=0`）は、強制有効は適用されません。 |
| `HTTPS_PROXY` / `HTTP_PROXY` / `NO_PROXY` | （Desktop側）アプリ内の更新チェック（`Help → Check for Updates…` と受動的な更新バナー）は、これらの標準変数が指定するプロキシを経由して `api.github.com` にアクセスし、`NO_PROXY` の除外も尊重します — `curl`、`npm`、`git` と同じ規約です。未設定の場合は直接接続します。 |

### Microsoft Graph（Teams会議） {#microsoft-graph-teams-meetings}

近日対応予定のTeams会議サマリーパイプラインで使用されるMicrosoft Graph RESTクライアント用のアプリ専用認証情報です。Azureポータルでの手順と必要な正確なAPI権限については、[Microsoft Graphアプリケーションの登録](../guides/microsoft-graph-app-registration.md)を参照してください。

| Variable | Description |
|----------|-------------|
| `MSGRAPH_TENANT_ID` | Graphアプリ登録用のAzure ADテナントID（ディレクトリGUID）。 |
| `MSGRAPH_CLIENT_ID` | Azureアプリ登録のアプリケーション（クライアント）ID。 |
| `MSGRAPH_CLIENT_SECRET` | アプリ登録のクライアントシークレット値。`chmod 600` で `~/.hermes/.env` に保存し、Azureポータルで定期的にローテーションしてください。 |
| `MSGRAPH_SCOPE` | クライアント認証情報トークンリクエスト用のOAuth2スコープ（デフォルト: `https://graph.microsoft.com/.default`）。 |
| `MSGRAPH_AUTHORITY_URL` | Microsoft IDプラットフォームのオーソリティ（デフォルト: `https://login.microsoftonline.com`）。国家/ソブリンクラウド向けにのみ上書きします（例: GCC Highの場合は `https://login.microsoftonline.us`）。 |

### Microsoft Graph Webhookリスナー

Graphイベント（Teams会議、カレンダー、チャットなど）の受信変更通知リスナーです。セットアップとセキュリティ強化については、[Microsoft Graph Webhookリスナー](../user-guide/messaging/msgraph-webhook.md)を参照してください。

| Variable | Description |
|----------|-------------|
| `MSGRAPH_WEBHOOK_ENABLED` | `msgraph_webhook` ゲートウェイプラットフォームを有効化（`true`/`1`/`yes`）。 |
| `MSGRAPH_WEBHOOK_PORT` | リスナーがバインドするポート（デフォルト: `8646`）。 |
| `MSGRAPH_WEBHOOK_CLIENT_STATE` | Graphがすべての通知で返す共有シークレット。`hmac.compare_digest` で比較されます。`openssl rand -hex 32` で生成します。 |
| `MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES` | Graphリソースパス/パターンのカンマ区切り許可リスト（例: `communications/onlineMeetings,chats/*/messages`）。末尾の `*` は前方一致です。空 = すべて受け入れる。 |
| `MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS` | リスナーへのPOSTを許可するCIDR範囲のカンマ区切りリスト（例: `52.96.0.0/14,52.104.0.0/14`）。空 = すべて許可（デフォルト）。本番環境ではMicrosoft Graphの公開エグレス範囲に制限してください。 |

### Teams会議サマリー配信

[`teams_pipeline` プラグイン](../user-guide/messaging/msgraph-webhook.md)が有効な場合にのみ使用されます。設定は `config.yaml` の `platforms.teams.extra` でも構成可能です — 両方設定されている場合は環境変数が優先されます。[Microsoft Teams → 会議サマリー配信](../user-guide/messaging/teams.md#meeting-summary-delivery-teams-meeting-pipeline)を参照してください。

| Variable | Description |
|----------|-------------|
| `TEAMS_DELIVERY_MODE` | `graph` または `incoming_webhook`。 |
| `TEAMS_INCOMING_WEBHOOK_URL` | Teamsが生成したWebhook URL。`TEAMS_DELIVERY_MODE=incoming_webhook` の場合に必須。 |
| `TEAMS_GRAPH_ACCESS_TOKEN` | Graph配信用に事前取得した委譲アクセストークン。ほとんど不要 — 未設定の場合、ライターは `MSGRAPH_*` アプリ認証情報にフォールバックします。 |
| `TEAMS_TEAM_ID` | チャンネル配信（`graph` モード）の対象チームID。 |
| `TEAMS_CHANNEL_ID` | 対象チャンネルID（`TEAMS_TEAM_ID` とペア）。 |
| `TEAMS_CHAT_ID` | 対象の1:1またはグループチャットID（`graph` モードでのチーム+チャンネルの代替）。 |

### LINE Messaging API

バンドルされたLINEプラットフォームプラグイン（`plugins/platforms/line/`）で使用されます。完全なセットアップについては、[メッセージングゲートウェイ → LINE](../user-guide/messaging/line.md)を参照してください。

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
| `LINE_EXPIRED_TEXT` | キャッシュされた回答が失われた（期限切れ / プロセス状態とともに消失した）ポストバックボタンがタップされたときの返信（デフォルト: `That request has expired — send your message again.`）。 |

### ntfy（プッシュ通知）

[ntfy](https://ntfy.sh/) は軽量なHTTPベースのプッシュ通知サービスです。[ntfyモバイルアプリ](https://ntfy.sh/docs/subscribe/phone/)からトピックを購読し、そのトピックに発行することでエージェントと会話できます。

| Variable | Description |
|----------|-------------|
| `NTFY_TOPIC` | 購読するトピック（受信メッセージ）。必須。 |
| `NTFY_SERVER_URL` | サーバーURL（デフォルト: `https://ntfy.sh`）。プライバシーのためにセルフホストのntfyを指定できます。 |
| `NTFY_TOKEN` | オプションの認証トークン。Bearerトークン（例: `tk_xyz`）、またはBasic認証用の `user:pass`。 |
| `NTFY_PUBLISH_TOPIC` | 送信する返信用のトピック（デフォルトは `NTFY_TOPIC`）。 |
| `NTFY_MARKDOWN` | `true` に設定すると、`X-Markdown: true` ヘッダー付きで返信を送信します。デフォルト: `false`。 |
| `NTFY_ALLOWED_USERS` | 許可リスト（ユーザーIDとして扱われます。ntfyではこれらはトピック名です）。通常は `NTFY_TOPIC` と同じ値に設定します。 |
| `NTFY_ALLOW_ALL_USERS` | 開発専用の抜け道 — アクセス制御されたプライベートトピックでのみ安全です。デフォルト: `false`。 |
| `NTFY_HOME_CHANNEL` | `deliver: ntfy` を指定したcronジョブのデフォルト配信先。 |
| `NTFY_HOME_CHANNEL_NAME` | ホームチャンネルの人間向けラベル（デフォルトはトピック名）。 |

信頼できないトピックでデプロイする前に、[ntfyメッセージングガイド](/docs/user-guide/messaging/ntfy) — 特に**アイデンティティモデル**のセクション — を参照してください。

### IRC

HermesをIRCサーバーに接続します。外部依存関係はありません。[IRCメッセージングガイド](/docs/user-guide/messaging/irc)を参照してください。

| Variable | Description |
|----------|-------------|
| `IRC_SERVER` | IRCサーバーのホスト名（例: `irc.libera.chat`）。必須。 |
| `IRC_CHANNEL` | 参加するチャンネル（例: `#hermes`）。複数の場合はカンマ区切り。必須。 |
| `IRC_NICKNAME` | ボットのニックネーム（デフォルト: `hermes-bot`）。必須。 |
| `IRC_PORT` | サーバーポート（デフォルト: TLSありは `6697`、なしは `6667`）。 |
| `IRC_USE_TLS` | TLSを使用（`true`/`false`。ポート6697ではデフォルト `true`）。 |
| `IRC_SERVER_PASSWORD` | `PASS` コマンド用のサーバーパスワード（オプション）。 |
| `IRC_NICKSERV_PASSWORD` | 接続時に自動でIDENTIFYするためのNickServパスワード（オプション）。 |
| `IRC_ALLOWED_USERS` | ボットとの会話を許可するニックネームのカンマ区切りリスト。 |
| `IRC_ALLOW_ALL_USERS` | チャンネル内の誰でもボットと会話できるようにする（開発専用）。 |
| `IRC_HOME_CHANNEL` | cron / 通知配信用のチャンネル（デフォルトは `IRC_CHANNEL`）。 |

### SimpleX

ローカルの `simplex-chat` デーモンを介して、Hermesを [SimpleX Chat](https://simplex.chat/) ネットワークに接続します。[SimpleXメッセージングガイド](/docs/user-guide/messaging/simplex)を参照してください。

| Variable | Description |
|----------|-------------|
| `SIMPLEX_WS_URL` | simplex-chatデーモンのWebSocket URL（例: `ws://127.0.0.1:5225`）。 |
| `SIMPLEX_ALLOWED_USERS` | ボットとの会話を許可するSimpleXコンタクトIDのカンマ区切りリスト。 |
| `SIMPLEX_ALLOW_ALL_USERS` | 任意のコンタクトがボットと会話できるようにする（開発専用 — 許可リストを無効化）。 |
| `SIMPLEX_AUTO_ACCEPT` | 受信したコンタクトリクエストを自動承認（デフォルト: `true`）。 |
| `SIMPLEX_GROUP_ALLOWED` | ボットが参加するSimpleXグループIDのカンマ区切りリスト、または任意のグループを許可する `*`。省略するとグループメッセージを完全に無視します（より安全なデフォルト — そうしないと、グループ内のボットはすべてのメンバーのトラフィックを処理します）。 |
| `SIMPLEX_HOME_CHANNEL` | cron / 通知配信用のデフォルトのコンタクト/グループID。 |
| `SIMPLEX_HOME_CHANNEL_NAME` | ホームチャンネルの人間向けラベル（デフォルトはID）。 |

### Photon

Nodeサイドカーを介して、Hermesを [Photon](https://photon.codes/) / Spectrum（iMessageおよびその他のSpectrumプラットフォーム）に接続します。[Photonメッセージングガイド](/docs/user-guide/messaging/photon)を参照してください。

| Variable | Description |
|----------|-------------|
| `PHOTON_PROJECT_ID` | SpectrumのプロジェクトID（プロジェクトの `spectrumProjectId`。`hermes photon setup` によって設定されます）。 |
| `PHOTON_PROJECT_SECRET` | SpectrumのプロジェクトIDと対になるプロジェクトシークレット（`hermes photon setup` によって設定されます）。 |
| `PHOTON_ALLOWED_USERS` | ボットとの会話を許可するE.164電話番号のカンマ区切りリスト。 |
| `PHOTON_ALLOW_ALL_USERS` | 任意の送信者がボットをトリガーすることを許可（開発専用 — 許可リストを無効化）。 |
| `PHOTON_REQUIRE_MENTION` | メンションのウェイクワードに一致しない限り、グループチャットのメッセージを無視します（`true`/`false`、デフォルト `false`）。 |
| `PHOTON_MENTION_PATTERNS` | グループチャット用のメンションウェイクワードの正規表現（JSONリストまたはカンマ/改行区切り。デフォルトはHermesのウェイクワード）。 |
| `PHOTON_HOME_CHANNEL` | cron / 通知配信用のデフォルトのPhoton宛先: SpectrumのスペースID、DMのGUID、または素のE.164電話番号。 |
| `PHOTON_HOME_CHANNEL_NAME` | ホームチャンネルの人間向けラベル。 |
| `PHOTON_MARKDOWN` | エージェントの返信をMarkdownとして送信 — iMessageはネイティブにレンダリングし、その他のSpectrumプラットフォームではプレーンテキストに劣化します（`true`/`false`、デフォルト `true`）。 |
| `PHOTON_REACTIONS` | 処理状況としてメッセージに👀/👍/👎のTapbackを付け、ボットのメッセージへのTapbackをエージェントにルーティングします（`true`/`false`、デフォルト `false`）。 |
| `PHOTON_READ_RECEIPTS` | Hermesに転送した後、受信したiMessageを既読にします（`true`/`false`、デフォルト `true`）。 |
| `PHOTON_TELEMETRY` | サイドカーでSpectrum SDKのテレメトリを有効化（`true`/`false`、デフォルト `false`。`hermes photon telemetry on|off` で切り替え）。 |
| `PHOTON_SIDECAR_PORT` | Nodeサイドカーの制御 + 受信チャネル用のループバックポート（デフォルト `8789`）。 |
| `PHOTON_SIDECAR_AUTOSTART` | 接続時にNodeサイドカーを起動（`true`/`false`、デフォルト `true`）。 |
| `PHOTON_DASHBOARD_HOST` | Photon Dashboard APIのホスト（デフォルト `https://app.photon.codes`）。 |
| `PHOTON_SPECTRUM_HOST` | Photon Spectrum APIのホスト（デフォルト `https://spectrum.photon.codes`）。 |

### Buzz（Nostrコミュニティ）

| Variable | Description |
|----------|-------------|
| `BUZZ_RELAY_URL` | BuzzコミュニティリレーのベースURL（例: `https://mycommunity.communities.buzz.xyz`） |
| `BUZZ_PRIVATE_KEY` | エージェントのBuzzアイデンティティ用のNostr秘密鍵（nsecまたはhex） — Buzzの唯一のシークレット |
| `BUZZ_CREDENTIALS_FILE` | nsecを保持するJSON認証情報ファイル（`BUZZ_PRIVATE_KEY` が未設定の場合のフォールバック） |
| `BUZZ_CHANNELS` | 監視するチャンネルUUIDのカンマ区切りリスト（デフォルト: 参加しているすべてのチャンネル） |
| `BUZZ_HOME_CHANNEL` | cron / 通知配信用のチャンネルUUID（デフォルトは最初に監視するチャンネル） |
| `BUZZ_ALLOWED_USERS` | エージェントとの会話を許可するnpubまたはhex公開鍵のカンマ区切りリスト |
| `BUZZ_ALLOW_ALL_USERS` | 任意のコミュニティメンバーがエージェントと会話できるようにする（`true`/`false`） |
| `BUZZ_TRANSPORT` | 受信トランスポート: `auto`（ポーリングフォールバック付きWebSocket、デフォルト）、`websocket`、または `poll` |
| `BUZZ_POLL_INTERVAL` | 受信ポーリングの間隔（秒単位、デフォルト: `4`） |
| `BUZZ_AUTH_TAG` | NIP-42 WebSocket認証用のオプションのNIP-OAオーナー証明認証タグJSON |
| `BUZZ_CLI_PATH` | buzz CLIバイナリへのパス（デフォルト: PATH上の `buzz`、次に `~/bin/buzz`） |

### Microsoft Teams（アダプター）

Microsoft Teamsプラットフォームアダプター（Bot Framework / Azure AD）です。上記の [Microsoft Graph（Teams会議）](#microsoft-graph-teams-meetings) 統合とは別物です。[Teamsメッセージングガイド](../user-guide/messaging/teams.md)を参照してください。

| Variable | Description |
|----------|-------------|
| `TEAMS_CLIENT_ID` | Azure ADアプリケーション（Bot Framework）のクライアントID。 |
| `TEAMS_CLIENT_SECRET` | Azure ADアプリケーションのクライアントシークレット。 |
| `TEAMS_TENANT_ID` | ボットアプリケーションをホストするAzure ADのテナントID。 |
| `TEAMS_HOST` | Webhookのバインドホスト（デフォルト: 未設定 → デュアルスタック、すべてのインターフェースでIPv4+IPv6）。 |
| `TEAMS_PORT` | Webhookのリッスンポート（Bot Frameworkのデフォルト: `3978`）。 |
| `TEAMS_ALLOWED_USERS` | ボットとの会話を許可するTeamsユーザーID / UPNのカンマ区切りリスト。 |
| `TEAMS_ALLOW_ALL_USERS` | 任意のTeamsユーザーがボットをトリガーすることを許可（開発専用）。 |
| `TEAMS_HOME_CHANNEL` | cron / 通知配信用のデフォルトのチャット/チャンネルID。 |
| `TEAMS_HOME_CHANNEL_NAME` | Teamsホームチャンネルの表示名。 |

### Raft

| Variable | Description |
|----------|-------------|
| `RAFT_PROFILE` | Raftエージェントのプロファイルスラッグ — 設定するとアダプターが自動的に有効になります。 |

### 高度なメッセージングチューニング

送信メッセージバッチャーをスロットリングするためのプラットフォームごとの高度なつまみです。ほとんどのユーザーはこれらに触れる必要はありません。デフォルトは、もたつきを感じさせずに各プラットフォームのレート制限を尊重するように設定されています。

| Variable | Description |
|----------|-------------|
| `HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS` | 1,024文字を超えるTelegramテキストをフラッシュするまでの猶予時間（デフォルト: `1.5`）。それより短いテキストでは、適応的な0.18/0.24秒の高速パスが維持されます。 |
| `HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS` | 単一のTelegramメッセージが長さ制限を超えた場合の分割チャンク間の遅延（デフォルト: `2.0`）。 |
| `HERMES_SIMPLEX_TEXT_BATCH_DELAY` | 立て続けに届く受信テキストメッセージを1つのMessageEventに連結するための静止期間（秒単位、デフォルト: `0.8`） — Telegramのテキストバッチ処理と同じパターンです。 |
| `HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS` | キューに入れたTelegramメディアをフラッシュするまでの猶予時間（デフォルト: `0.6`）。 |
| `HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS` | ターン開始後、プレーンなTelegramテキストのフォローアップがそのターンを中断またはリダイレクトせずにキューに加わる秒数（デフォルト `3`。`0` で猶予を無効化）。コマンドとメディアは保留されません。 |
| `HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT` / `_READ_TIMEOUT` / `_WRITE_TIMEOUT` / `_POOL_TIMEOUT` | 基盤となる `python-telegram-bot` のHTTPタイムアウト（秒）を上書き。 |
| `HERMES_TELEGRAM_INIT_TIMEOUT` | ゲートウェイ起動中のTelegram `initialize()` 接続チェーンの試行ごとの上限（秒）。到達不能なフォールバックIPチェーンによって起動が無期限にブロックされないようにします（デフォルト: `30`）。 |
| `HERMES_TELEGRAM_HTTP_POOL_SIZE` | Telegram APIへの最大同時HTTP接続数。 |
| `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS` | DNS失敗時に使用されるハードコードされたCloudflareフォールバックIPを無効化（`true`/`false`）。 |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | キューに入れたDiscordテキストチャンクをフラッシュするまでの猶予時間（デフォルト: `0.6`）。 |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Discordメッセージが長さ制限を超えた場合の分割チャンク間の遅延（デフォルト: `2.0`）。 |
| `HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS` | `discord.websocket_liveness_interval_seconds` の互換/手動上書き。アクティブなDiscord Gateway WebSocketをサンプリングする間隔（デフォルト: `15`。`0` に設定すると無効化）。`config.yaml` のキーを優先してください。 |
| `HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD` | `discord.websocket_liveness_failure_threshold` の互換/手動上書き。再接続を強制するまでの、連続した異常なWebSocketサンプル数（デフォルト: `2`）。ソフトシグナルにのみ適用されます — トランスポートが閉じている場合（`socket_closed` / `client_closed`）は最初のサンプルで再接続を強制します（#118487）。`config.yaml` のキーを優先してください。 |
| `HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | TelegramバッチつまみのMatrix版。 |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` / `_MAX_CHARS` / `_MAX_MESSAGES` | Feishuバッチャーのチューニング — 遅延、分割遅延、メッセージあたりの最大文字数、バッチあたりの最大メッセージ数。 |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | Feishuメディアのフラッシュ遅延。 |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | Feishu Webhookの重複排除キャッシュのサイズ（デフォルト: `1024`）。 |
| `HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | WeComバッチャーのチューニング。 |
| `HERMES_VISION_DOWNLOAD_TIMEOUT` | 画像をビジョンモデルに渡す前にダウンロードするタイムアウト（秒単位、デフォルト: `30`）。 |
| `HERMES_VISION_MAX_CONCURRENCY` | プロセス全体での画像の**エンコード/リサイズ**バーストの最大同時実行数（`auxiliary.vision.max_concurrency` の上書き。デフォルト: ホストのCPUコア数、上限なし）。CPUバウンドのエンコード処理のみを制限するため、動画フレームのファンアウトがすべてのコアを飽和させてイベントループを枯渇させることはありません — LLM呼び出しは完全に並行したままです。`< 1` の値は無視されます。 |
| `HERMES_RESTART_DRAIN_TIMEOUT` | ゲートウェイ: `agent.restart_drain_timeout` を上書きします。これは、ゲートウェイが停止または再起動した後、処理中のチャットターンが中断されるまで実行を続けられる秒数です（デフォルト: `0`、即座に中断）。[Gateway Restart Drain](../user-guide/configuration.md#gateway-restart-drain) を参照してください。 |
| `HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT` | ゲートウェイの起動中および再接続中のプラットフォームごとの接続タイムアウト（秒。`0`/負の値では無期限に待機）。接続試行*と*Discordアダプターの準備待ちに適用されるため、同期するスラッシュコマンドが多いアカウントが起動途中で強制終了されることはありません。`config.yaml` の `gateway.platform_connect_timeout`（デフォルト `30`）からブリッジされます。この環境変数は手動の上書きで、明示的に設定された場合はこちらが優先されます。 |
| `HERMES_GATEWAY_BUSY_INPUT_MODE` | ゲートウェイのデフォルトのビジー入力動作: `queue`、`steer`、または `interrupt`。`/busy` でアクティブなプロファイルに対して上書きできます。 |
| `HERMES_GATEWAY_BUSY_ACK_ENABLED` | エージェントがビジー状態のときにユーザーが入力を送信した際、ゲートウェイが確認メッセージ（⚡/⏳/⏩）を送信するかどうか（デフォルト: `true`）。`false` に設定するとこれらのメッセージを完全に抑制します — 入力は通常どおりキューイング/ステアリング/割り込みされ、チャット返信のみが抑制されます。`config.yaml` の `display.busy_ack_enabled` からブリッジされます。 |
| `HERMES_GATEWAY_NO_SUPERVISE` | s6-overlay Dockerイメージ内で `hermes gateway run` を実行する際に自動監視をオプトアウトし、s6以前のフォアグラウンドのセマンティクス（自動再起動なし、ゲートウェイがコンテナのメインプロセス）を使用します。真とみなされる値: `1`、`true`、`yes`。CLIフラグ `--no-supervise` と同等です。s6イメージ以外では何もしません。 |
| `HERMES_GATEWAY_BOOTSTRAP_STATE` | s6-overlay Dockerイメージ内で、新しいボリューム上のゲートウェイの**初期**監視状態を宣言します。空のボリュームには永続化された `gateway_state.json` がないため、ブートリコンサイラーは `gateway-default` スロットを登録しますが、**停止**状態のままにします（最後に記録された状態が `running` の場合にのみ自動起動します）。これを `running` に設定すると、初回起動時のセットアップフックがリコンサイラーの実行*前に* `gateway_state.json` を作成するため、ゲートウェイはまさに最初の起動時に立ち上がります。リテラル値 `running` のみが有効です。初回起動時のみ: 既存の `gateway_state.json` が上書きされることはないため、意図的に停止したゲートウェイは再起動後も停止したままです。s6イメージ以外では何もしません。 |
| `GATEWAY_RELAY_URL` | 実験的なリレーコネクターのWebSocketベースURL。設定すると、ゲートウェイは汎用の `relay` アダプターを登録し、コネクターに外向きに接続します。`config.yaml` の `gateway.relay_url` に対応します。 |
| `GATEWAY_RELAY_ID` | `hermes gateway enroll` またはマネージドなセルフプロビジョニングによって割り当てられるリレーゲートウェイの識別子。`gateway.relay_id` に対応します。 |
| `GATEWAY_RELAY_SECRET` | WebSocketの認証に使用されるゲートウェイごとのリレーシークレット。すでに設定されている場合、マネージドなセルフプロビジョニングはスキップされます。`gateway.relay_secret` に対応します。 |
| `GATEWAY_RELAY_DELIVERY_KEY` | リレー/パススルー認証の互換性のために保持される、コネクターが発行した配信キー。現在のリレーの受信メッセージは、ゲートウェイ側のHTTPレシーバーではなく外向きのWebSocket上で届きます。 |
| `GATEWAY_RELAY_ENROLL_TOKEN` | `--token` が明示的に渡されなかった場合に `hermes gateway enroll` が使用する登録トークン。 |
| `GATEWAY_RELAY_PLATFORM` | リレーの機能ディスクリプターで通知されるオプションのプラットフォーム名。 |
| `GATEWAY_RELAY_BOT_ID` | リレーの機能ディスクリプターで通知されるオプションのボット識別子。 |
| `GATEWAY_RELAY_ENDPOINT` | コールバック/パススルーURLを必要とするコネクターモード向けに通知されるオプションのゲートウェイエンドポイント。デフォルトのWSのみの受信リレーパスでは不要です。`gateway.relay_endpoint` に対応します。 |
| `GATEWAY_RELAY_ROUTE_KEYS` | コネクターに通知されるリレールートキーのカンマ区切りリスト。`gateway.relay_route_keys` に対応します。 |
| `HERMES_FILE_MUTATION_VERIFIER` | ターンごとのファイル変更検証フッターを有効化（デフォルト: `true`）。有効にすると、Hermesはターン中に失敗し、その後の成功した書き込みで置き換えられなかった `write_file` / `patch` 呼び出しを一覧にした注意書きを追記します。抑制するには `0`、`false`、`no`、または `off` に設定します。`config.yaml` の `display.file_mutation_verifier` に対応し、設定されている場合は環境変数が優先されます。 |
| `HERMES_CRON_TIMEOUT` | cronジョブのエージェント実行の非アクティブタイムアウト（秒単位、デフォルト: `600`）。エージェントはツールを能動的に呼び出している間やストリームトークンを受信している間は無期限に実行できます — これはアイドル時にのみトリガーされます。`0` を設定すると無制限になります。 |
| `HERMES_CRON_SCRIPT_TIMEOUT` | cronジョブに付随する事前実行スクリプトのタイムアウト（秒単位、デフォルト: `3600`）。スクリプトのみを制限します — スキル/エージェントのジョブは別の `HERMES_CRON_TIMEOUT` 非アクティブ予算を使用します。`config.yaml` の `cron.script_timeout_seconds` でも設定可能です。 |
| `HERMES_CRON_MEDIA_SEND_TIMEOUT` | ライブのゲートウェイアダプター経由でのcron配信中、各メディア添付ファイルを送信する際のタイムアウト（秒単位、デフォルト: `300`）。大きな添付ファイル（長いTTS音声、大きなエクスポート）のアップロード中にタイムアウトする場合は引き上げてください。`config.yaml` の `cron.media_send_timeout_seconds` でも設定可能です。 |
| `HERMES_CRON_MAX_PARALLEL` | ティックごとに並列実行するcronジョブの最大数（デフォルト: `4`）。 |

## NeMo Relay

| Variable | Description |
|----------|-------------|
| `HERMES_NEMO_RELAY_PLUGINS_TOML` | Hermesコアがプロセス全体で読み込む、標準のNeMo Relay `plugins.toml` へのオプションの明示的なパス。未設定の場合、Relayは通常のユーザー設定と、それより優先度の高いマシン全体のシステム設定を使用します。設定すると、このファイルがユーザー設定を置き換えますが、システム設定は引き続きその上に適用されます。削除された `HERMES_NEMO_RELAY_ATOF_*` および `HERMES_NEMO_RELAY_ATIF_*` 変数は無視されます。`hermes update` / `hermes migrate relay` がそれらを `<hermes home>/relay-plugins.toml` に変換し、この変数を設定します — [移行に関する注意と完全な例](../user-guide/features/built-in-plugins.md#nemo-relay-native-integration-migration-note)を参照してください。[NeMo Relay observability configuration](https://docs.nvidia.com/nemo/relay/configure-plugins/observability/about) を参照してください。 |

## エージェントの動作

| Variable | Description |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | 会話ごとの最大ツール呼び出し反復回数（デフォルト: 500） |
| `HERMES_INFERENCE_MODEL` | プロセスレベルでモデル名を上書き（セッションについては `config.yaml` より優先）。`-m`/`--model` フラグでも設定可能。 |
| `HERMES_YOLO_MODE` | `1` に設定すると危険なコマンドの承認プロンプトをバイパス。`--yolo` と同等。 |
| `HERMES_ACCEPT_HOOKS` | `config.yaml` で宣言された未確認のシェルフックを、TTYプロンプトなしで自動承認。`--accept-hooks` または `hooks_auto_accept: true` と同等。 |
| `HERMES_IGNORE_USER_CONFIG` | `~/.hermes/config.yaml` をスキップして組み込みのデフォルトを使用（`.env` の認証情報は引き続き読み込まれます）。`--ignore-user-config` と同等。 |
| `HERMES_IGNORE_RULES` | `AGENTS.md`、`SOUL.md`、`.cursorrules`、メモリ、プリロードされたスキルの自動注入をスキップ。`--ignore-rules` と同等。 |
| `HERMES_SAFE_MODE` | レガシーなプロセス内トラブルシューティング用スイッチ: そのプロセスでプラグインの検出、MCPサーバーの読み込み、シェルフックの登録をスキップします。`hermes chat --safe-mode` はもうこれを設定しません — 代わりにゲートウェイがセーフポリシーをセッションに固定し、分離されたワーカーで実行するため、実行中のデーモンの環境が変更されることはありません。 |
| `HERMES_TOOL_PROGRESS` | config-v12のサポート下限以降はサポートされていません — この変数は無視されます。`config.yaml` の `display.tool_progress` を使用してください。 |
| `HERMES_TOOL_PROGRESS_MODE` | ツール進捗モード用の非推奨の互換変数（ゲートウェイはフォールバックとして引き続き読み込みます）。`config.yaml` の `display.tool_progress` を推奨します。 |
| `HERMES_HUMAN_DELAY_MODE` / `HERMES_HUMAN_DELAY_MIN_MS` / `HERMES_HUMAN_DELAY_MAX_MS` | 現在は読み込まれません。応答のペース配分は各プロファイルの `config.yaml` の `human_delay` セクション（`mode`、`min_ms`、`max_ms`）で設定するため、マルチプレキシングされたプロファイルはそれぞれ独立したペース配分を保ちます。 |
| `HERMES_QUIET` | 必須でない出力を抑制（`true`/`false`） |
| `CODEX_HOME` | [Codex app-serverランタイム](../user-guide/features/codex-app-server-runtime)が有効な場合に、Codex CLIが設定と認証情報を読み込むディレクトリを上書き（デフォルト: `~/.codex`）。Hermesの移行処理は管理ブロックを `<CODEX_HOME>/config.toml` に書き込みます。 |
| `HERMES_KANBAN_TASK` | ワーカーを生成する際にkanbanディスパッチャーが設定します（タスクUUID）。ワーカーと、生成された `hermes-tools` MCPサブプロセスがこれを継承するため、kanbanツールが正しくゲートされます。手動で設定しないでください。 |
| `HERMES_ACP_SKIP_CONFIGURED_MCP` | 廃止済み。現在は読み込まれず、効果もありません。`hermes acp` はゲートウェイデーモンのビューアーであり、`config.yaml` のMCPサーバーを自分で起動することはありません。ゲートウェイは起動時にそのプロファイルの `mcp_servers` を起動し（ACP向けには `platform_toolsets.acp` でフィルタリング）、エディターの `session/new` の `mcpServers` はいずれにせよゲートウェイによってアタッチされます。[ACP host integration](../user-guide/features/acp#host-integration) を参照してください。 |
| `HERMES_API_TIMEOUT` | LLM API呼び出しのタイムアウト（秒単位、デフォルト: `1800`） |
| `HERMES_API_CALL_STALE_TIMEOUT` | 非ストリーミングの停滞呼び出しタイムアウト（秒単位、デフォルト: `90`）。ローカルプロバイダーでは未設定のままだと自動的に無効化され、非常に大きなコンテキストでは上方にスケールする場合があります。`config.yaml` の `providers.<id>.stale_timeout_seconds` または `providers.<id>.models.<model>.stale_timeout_seconds` でも設定可能です。 |
| `HERMES_STREAM_READ_TIMEOUT` | ストリーミングソケットの読み取りタイムアウト（秒単位、デフォルト: `120`）。ローカルプロバイダーでは自動的に `HERMES_API_TIMEOUT` まで増加します。ローカルLLMが長いコード生成中にタイムアウトする場合は増やしてください。 |
| `HERMES_STREAM_STALE_TIMEOUT` | 停滞ストリーム検出タイムアウト（秒単位、デフォルト: `180`）。ローカルプロバイダーでは自動的に無効化されます。このウィンドウ内にチャンクが到着しない場合、接続を切断します。 |
| `HERMES_LOCAL_STREAM_STALE_TIMEOUT` | ローカルプロバイダー（Ollama、oMLX、llama-cpp）用の停滞ストリームの上限（秒単位、デフォルト: `900`）。基本の停滞タイムアウトがデフォルトのままでローカルエンドポイントが検出された場合、以前の無限の無効化に代わってこの有限の上限が使用されるため、固まったローカルサーバーは永遠にハングする代わりに最終的に検出器に引っかかります。`config.yaml` の `agent.local_stream_stale_timeout` でも設定可能です。 |
| `HERMES_STREAM_RETRIES` | 一時的なネットワークエラー時のストリーム途中の再接続試行回数（デフォルト: `3`）。 |
| `HERMES_STREAM_STALE_GIVEUP` | ターンをまたいだサーキットブレーカー: 完了した応答がないまま、この回数だけ連続して停滞による強制終了（ストリーミングまたは非ストリーミング）が発生すると、停滞タイムアウトを再び待つ代わりに、各呼び出しを対処可能なエラーとともに即座に中止します（デフォルト: `5`、`0` で無効化）。完了した応答、`/model` の切り替え、フォールバックの有効化、またはターン開始時のプライマリ復元のいずれかでリセットされます。 |
| `HERMES_AGENT_TIMEOUT` | 実行中のエージェントに対するゲートウェイの非アクティブタイムアウト（秒単位、デフォルト: `1800`、30分）。すべてのツール呼び出しとストリームトークンでリセットされます。`0` を設定すると無効化します。 |
| `HERMES_GATEWAY_MAX_STARTS` | 再起動ストームのサーキットブレーカー: ストームを断ち切るために指数バックオフでスリープするまでに、ウィンドウ内で許可されるゲートウェイの（再）起動の最大回数（デフォルト: `5`、`0` で無効化）。`config.yaml` の `gateway.respawn_storm.max_starts` でも設定可能です。 |
| `HERMES_GATEWAY_START_WINDOW_S` | 再起動ストームブレーカーのウィンドウ（秒単位、デフォルト: `120`）。`config.yaml` の `gateway.respawn_storm.window_seconds` でも設定可能です。 |
| `HERMES_STARTUP_WATCHDOG` | `hermes gateway run` 用の起動時生存監視ウォッチドッグ: タイムアウト内にプロセスがライブのイベントループに到達せず、進捗リースを保持せず、CPUの進捗も見られない場合、すべてのスレッドのスタックを `logs/gateway-startup-watchdog.log` にダンプし、コード `75` で終了するため、サービススーパーバイザー（systemd、s6、Windowsタスク）がそれを再起動します。オプトアウトするには `0` に設定します。`config.yaml` の解析自体が監視対象の期間内にあるため、環境変数のみで設定します。未設定の場合、`config.yaml` の `gateway.startup_watchdog: false` がこの変数にブリッジされます。 |
| `HERMES_STARTUP_WATCHDOG_TIMEOUT_S` | 起動ウォッチドッグのタイムアウト（秒単位、デフォルト: `300`）。遅いが生きているフェーズ（state.dbのスキーマ移行、修復、構築時のアーカイブ/プルーニング/VACUUM）はそれぞれ独自の進捗リースを保持するため、大規模なインストールの起動がそれらのフェーズの*外で*正当に5分より長くかかる場合（多数のマルチプレキシングされたプロファイル、遅いディスク上の数千のスキル）にのみ引き上げてください。未設定の場合は `config.yaml` の `gateway.startup_watchdog_timeout_seconds` からブリッジされます。 |
| `HERMES_AGENT_TIMEOUT_WARNING` | ゲートウェイ: この秒数の非アクティブの後に警告メッセージを送信（デフォルト: `HERMES_AGENT_TIMEOUT` の75%）。 |
| `HERMES_AGENT_NOTIFY_INTERVAL` | ゲートウェイ: 長時間実行されるエージェントターンでの進捗通知の間隔（秒単位）。 |
| `HERMES_CHECKPOINT_TIMEOUT` | ファイルシステムチェックポイント作成のタイムアウト（秒単位、デフォルト: `30`）。 |
| `HERMES_EXEC_ASK` | ゲートウェイモードで実行承認プロンプトを有効化（`true`/`false`） |
| `HERMES_ENABLE_PROJECT_PLUGINS` | エージェントローダーとダッシュボードWebサーバーの両方で、`./.hermes/plugins/` からのリポジトリローカルプラグインの自動検出を有効化します。標準の真とみなされる値のセットを受け付けます: `1` / `true` / `yes` / `on`（大文字小文字を区別しない）。それ以外 — `0`、`false`、`no`、`off`、空文字列を含む — はすべて**無効**（デフォルト）として扱われます。注: GHSA-5qr3-c538-wm9j（#29156）以降、この変数が有効であっても、ダッシュボードWebサーバーはプロジェクトプラグインのPython `api` ファイルの自動インポートを拒否します — プロジェクトプラグインは静的なJS/CSSでUIを拡張できますが、そのバックエンドルートは `~/.hermes/plugins/` の下に移動した場合にのみ読み込まれます。 |
| `HERMES_PLUGINS_DEBUG` | `1`/`true` で、詳細なプラグイン検出ログをstderrに表示 — スキャンしたディレクトリ、解析したマニフェスト、スキップ理由、解析または `register()` 失敗時の完全なトレースバック。プラグイン作成者向けです。 |
| `HERMES_BACKGROUND_NOTIFICATIONS` | ゲートウェイのバックグラウンドプロセス通知モード: `concise`（デフォルト）、`all`、`result`、`error`、`off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | API呼び出し時に注入されるエフェメラルなシステムプロンプト（セッションには永続化されません） |
| `HERMES_PREFILL_MESSAGES_FILE` | API呼び出し時に注入されるエフェメラルなプレフィルメッセージのJSONファイルへのパス。 |
| `HERMES_ALLOW_PRIVATE_URLS` | `true`/`false` — ツールがlocalhost/プライベートネットワークのURLを取得することを許可。ゲートウェイモードではデフォルトでオフ。 |
| `HERMES_REDACT_SECRETS` | `true`/`false` — ツール出力、ログ、チャット応答での機密情報の伏せ字を制御（デフォルト: `true`）。 |
| `HERMES_WRITE_SAFE_ROOT` | 列挙したルートの外への `write_file`/`patch` の書き込みを**ハードブロック**する任意のディレクトリプレフィックス（承認プロンプトなし）。`os.pathsep`（Unixでは `:`、Windowsでは `;`）で区切って複数のディレクトリを指定できます。下記の [HERMES_WRITE_SAFE_ROOT](#hermes_write_safe_root) を参照してください。 |
| `HERMES_DISABLE_LAZY_INSTALLS` | テストとインストールプローブで使用される内部のPMポリシー。真とみなされる値ではオンデマンドのインストールを拒否します。ユーザー向けの `security.allow_lazy_installs` 設定を上書きします。`.env` には記述しないでください。 |
| `HERMES_DISABLE_FILE_STATE_GUARD` | `1` に設定すると、`patch`/`write_file` の「読み取ってからファイルが変更された」ガードをオフにします。 |
| `HERMES_BUNDLED_SKILLS` | 起動時に読み込まれるバンドルスキルのリストのカンマ区切り上書き。 |
| `HERMES_OPTIONAL_SKILLS` | 初回実行時に自動インストールする任意スキル名のカンマ区切りリスト。 |
| `HERMES_DEBUG_INTERRUPT` | `1`/`true` に設定すると、詳細な割り込み/キャンセルトレースを `agent.log` にログ出力。`0`/`false`/`off`（または未設定）では無効のままです。 |
| `HERMES_DUMP_REQUESTS` | APIリクエストのペイロードをログファイルにダンプ（`true`/`false`） |
| `HERMES_DUMP_REQUEST_STDOUT` | APIリクエストのペイロードをログファイルではなくstdoutにダンプ。 |
| `HERMES_OAUTH_TRACE` | `1` に設定すると、OAuthトークン交換とリフレッシュの試行をログ出力。伏せ字されたタイミング情報を含みます。 |
| `HERMES_AGENT_HELP_GUIDANCE` | カスタムデプロイ用にシステムプロンプトへ追加のガイダンステキストを追記。 |
| `HERMES_AGENT_LOGO` | CLI起動時のASCIIバナーロゴを上書き。 |
| `DELEGATION_MAX_CONCURRENT_CHILDREN` | `delegate_task` バッチあたりの最大並列サブエージェント数（デフォルト: `3`、下限1、上限なし）。`config.yaml` の `delegation.max_concurrent_children` でも設定可能 — 設定値が優先されます。 |

### HERMES_WRITE_SAFE_ROOT {#hermes_write_safe_root}

この変数が設定されている場合、`write_file` と `patch` は列挙したディレクトリプレフィックス内のパスのみを対象にできます。それらのルートの外にあるパスは**即座に拒否されます** — 書き込みは危険なコマンドの承認システムを経由せず、上書きするためのプロンプトもありません。

公式Dockerイメージは `HERMES_HOME=/opt/data` と合わせて `HERMES_WRITE_SAFE_ROOT=/opt/data` を設定するため、エージェントはマウントされたデータボリュームの外に出ることができません。

**書き込みをサンドボックス化する意図がない限り、これを `~/.hermes/.env` に追加しないでください。** よくある間違いは、エージェントに `~/.hermes/cron/jobs.json`、`~/.hermes/skills/`、またはプロファイル配下のスクリプトを編集させるつもりで、この変数をプロジェクトディレクトリに向けてしまうことです — それらのパスはサンドボックスの外にあるため、それらへの `write_file`/`patch` はすべて `outside HERMES_WRITE_SAFE_ROOT` エラーで失敗します。

ワークスペースとHermesの状態の両方を許可するには、両方のプレフィックスを列挙します（順序は問いません）:

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

変数を未設定にするか `.env` から削除すると、通常の書き込みに戻ります（ただし、認証情報パスの拒否リストは引き続き適用されます — [File write safety](../user-guide/security.md#file-write-safety) を参照してください）。

### 内部ブリッジ変数

Hermesは、まだ `config.yaml` が存在しない境界や、2つのプロセスが合意する必要がある境界をまたいで状態を受け渡すために、これらを自分で設定します。プロセス環境やログでこれらを見分けられるように記載していますが、自分で設定しないでください。また、決して `.env` に記述しないでください。

| Variable | Description |
|----------|-------------|
| `HERMES_DATA_DIR_SUFFIX` | デスクトップバンドルの環境に組み込まれます（`--bundle-env`、`HERMES_BUNDLE_ENV_JSON`、またはチャンネル固有のデータディレクトリを持つチャンネルビルド。後者は `-channel-build-<channel>` を使用）。これにより、テストビルドやチャンネルビルドが独自のデータを保持できます。デフォルトのHermesホームとデフォルトのElectron `userData` ディレクトリに、区切り文字なしでそのまま付加されます: POSIXでは `-channel-build-canary` により `~/.hermes-channel-build-canary` が選択されます。明示的な `HERMES_HOME` と `HERMES_DESKTOP_USER_DATA_DIR` が優先され、サフィックスは付加されません。`.env` と `config.yaml` を保持するホームを選択するため、起動前に起動環境に設定されている必要があります。 |
| `HERMES_REPO_URL` | インストーラー（`scripts/install.sh`、`scripts/install.ps1`）がクローン元とし、再実行時に `origin` を向け直すGitリモート。インストーラーはHermesの設定が存在する前に実行されるため、環境変数になっています。CIやリハーサルスクリプトがフォークやミラーからインストールするために使用します。未設定の場合、インストーラーは公式リポジトリを使用します。 |
| `HERMES_UPDATE_STATUS_FILE` | デスクトップの更新シム（`scripts/desktop-update/posix.sh`）が、その進捗ウィンドウが描画するステータスJSONのパスとともにエクスポートします。`hermes update` の引き継ぎ子プロセスは、ウィンドウが動き続けるように、長時間実行されるステージをそのファイルに書き込みます。存在しない場合（古いシム）、更新マーカー内のシムのPIDで命名されたステータスファイルにフォールバックし、UIが監視していない場合は何も書き込みません。 |
| `HERMES_UPDATE_UI_ACTIVE` | 独自のウィンドウを持たない古いシムのために、更新の子プロセスがネイティブのmacOSステータスパネルを開いた後に `1` に設定します。子プロセスはこれを継承するため、パネルは更新チェーンごとに最大1回しか開かれません。 |

## インターフェース

| Variable | Description |
|----------|-------------|
| `HERMES_TUI` | `1` に設定すると、クラシックCLIの代わりに [TUI](../user-guide/tui.md) を起動。`--tui` を渡すのと同等。 |
| `HERMES_TUI_DIR` | ビルド済み `ui-tui/` ディレクトリへのパス（`dist/entry.js` と populated な `node_modules` を含む必要があります）。ディストリビューションやNixが初回起動時の `pnpm install` をスキップするために使用します。 |
| `HERMES_TUI_RESUME` | 起動時に特定のTUIセッションをIDで再開。設定すると、`hermes --tui` は新しいセッションを生成せず、指定されたセッションを引き継ぎます — 切断やターミナルクラッシュ後の再接続に便利です。 |
| `HERMES_TUI_THEME` | TUIのカラーテーマを強制: `light`、`dark`、または生の6文字の背景16進数（例: `ffffff` や `1a1a2e`）。未設定の場合、Hermesは `COLORFGBG` とターミナル背景クエリを使って自動検出します。この変数は、`COLORFGBG` を設定しないターミナル（Ghostty、Warp、iTerm2など）での検出を上書きします。 |
| `HERMES_INFERENCE_MODEL` | `config.yaml` を変更せずに `hermes -z` / `hermes chat` のモデルを強制します。`--provider` フラグと組み合わせて使用します。実行ごとにデフォルトモデルを上書きする必要があるスクリプト化された呼び出し元（sweeper、CI、バッチランナー）に便利です。 |

## セッション設定

| Variable | Description |
|----------|-------------|
| `HERMES_SESSION_ID` | **Hermesが生成するすべてのツールサブプロセスに自動的にエクスポートされます**（`terminal`、`execute_code`、永続シェル、Docker/Singularityバックエンド、委譲されたサブエージェントの実行）。エージェントが現在のセッションIDに設定します。ツールから呼び出されるユーザースクリプトはこれを読み取って、自身の出力、テレメトリ、副作用を元のHermesセッションと関連付けることができます。**手動で設定すべきではありません** — 親シェルから上書きしても効果があるのはエージェントの実行外のみで、エージェントがセッションを開始した瞬間に上書きされます。 |
| `AI_AGENT` | **CLIとゲートウェイのエントリーポイントによって `hermes-agent` に設定され**（外側のハーネスによってまだ設定されていない場合のみ）、リモートバックエンド（Docker、SSH、Modal、Daytona、Singularity、Vercel）を含むすべてのterminalツールのシェルにエクスポートされます。子プロセスの帰属を示すための、エージェント横断の新しい標準です — 汎用ツール（例: huggingface_hubのエージェント検出）はこれを読み取って、AIエージェントの下で実行されていることを知ります。値は公開のエージェントハーネスレジストリにおけるHermesのIDと一致します。手動で設定しないでください。 |
| `HERMES_AGENT` | **CLIとゲートウェイのエントリーポイントによって `true` に設定され**、すべてのterminalツールのシェルにエクスポートされるため、子プロセスは自分が特にHermes内で実行されていることを検出できます。手動で設定しないでください。 |

ターミナルセッションのスナップショットには、注入されたセッション/エージェントの帰属変数は永続化されません。Hermesは各コマンドに現在の値を提供するため、前のターミナルコマンド内でエクスポートしても、次のセッションのアイデンティティが再定義されることはありません。

## コンテキスト圧縮（config.yamlのみ）

コンテキスト圧縮は `config.yaml` を通じてのみ設定されます — そのための環境変数はありません。しきい値の設定は `compression:` ブロックにあり、要約モデル/プロバイダーは `auxiliary.compression:` の下にあります。

```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20         # fraction of threshold to preserve as recent tail
  protect_last_n: 20         # minimum recent messages to keep uncompressed
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

:::note
`AUXILIARY_WEB_EXTRACT_*` 変数は廃止されました: `web_extract` はもう補助LLMを使用しません。長いページは決定的に切り詰められ、全文は `read_file` によるページングのためにディスクに保存されます。
:::

タスク固有の直接エンドポイントについては、Hermesはそのタスクに設定されたAPIキーを使用します。それがない場合は、メインエンドポイントの正確なオリジンに対してメインモデルのキーを使用し、次に `OPENAI_API_KEY` をOpenAIホスト（`*.openai.com`、`*.openai.azure.com`）またはそれと対になる正確な `OPENAI_BASE_URL` に対してのみ使用します。これらのカスタムエンドポイントに `OPENROUTER_API_KEY` を再利用することはありません。

## フォールバックプロバイダー（config.yamlのみ）

プライマリモデルのフォールバックチェーンは `config.yaml` を通じてのみ設定されます — そのための環境変数はありません。`provider` と `model` のキーを持つトップレベルの `fallback_providers` リストを追加すると、メインモデルがエラーに遭遇したときに自動フェイルオーバーが有効になります。プロバイダーが `auto` の補助タスクも、Hermes組み込みの補助検出チェーンより先にこのチェーンを参照します。

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

古いトップレベルの `fallback_model` 単一プロバイダー形式も後方互換性のため引き続き読み込まれますが、新しい設定では `fallback_providers` を使用してください。タスク固有の補助ポリシーには、`config.yaml` の `auxiliary.<task>.fallback_chain` を使用してください。対応する環境変数はありません。

詳細は [フォールバックプロバイダー](../user-guide/features/fallback-providers.md) を参照してください。

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
環境変数の設定には `hermes config set` を使用してください — このページのすべての `UPPER_SNAKE` 名（およびその他の環境変数の形をした名前）は `.env` に保存されます。これはセットアップフローが書き込み、ランタイムが読み込むのと同じファイルであり、`config.yaml` に書き込まれることはありません。環境変数ライターの拒否リストにある名前（`HERMES_HOME`、`HERMES_YOLO_MODE`、`PATH`、…）は拒否されます。ドット区切りの `config.yaml` 設定は `config.yaml` に保存されます。
:::
