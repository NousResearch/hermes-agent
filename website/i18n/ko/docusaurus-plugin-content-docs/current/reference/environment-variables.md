---
sidebar_position: 2
title: "환경 변수"
description: "Hermes Agent에서 사용하는 모든 환경 변수에 대한 완전한 참조"
---

# 환경 변수 참조

Hermes는 프로세스 환경에서 환경 변수를 읽으며, 사용자가 관리하는 비밀 정보는 `~/.hermes/.env`에서도 읽습니다. API 키, 봇 토큰, OAuth 비밀 정보 및 기타 자격 증명은 `.env`에 보관하고, 구성 키가 있는 비밀이 아닌 동작 설정에는 `config.yaml`을 우선 사용하세요. 아래 변수 중 일부는 프로세스 전용 재정의 변수이거나 내부 브리지 변수이므로, 여기에 문서화되어 있다는 이유만으로 `.env`에 커밋해서는 안 됩니다.

## LLM 제공자

| 변수 | 설명 |
|----------|-------------|
| `OPENROUTER_API_KEY` | 유연성을 위해 권장되는 OpenRouter API 키 |
| `OPENROUTER_BASE_URL` | OpenRouter 호환 기본 URL 재정의 |
| `FIREWORKS_API_KEY` | Fireworks AI API 키 ([app.fireworks.ai](https://app.fireworks.ai/settings/users/api-keys)). 엔드포인트 재정의는 `config.yaml`의 `model.base_url`로 구성합니다. |
| `HERMES_OPENROUTER_CACHE` | OpenRouter 응답 캐싱 활성화 (`1`/`true`/`yes`/`on`). `config.yaml`의 `openrouter.response_cache`를 재정의합니다. [응답 캐싱](https://openrouter.ai/docs/guides/features/response-caching)을 참조하세요. |
| `HERMES_OPENROUTER_CACHE_TTL` | 캐시 TTL(초)(1-86400). `config.yaml`의 `openrouter.response_cache_ttl`을 재정의합니다. |
| `NOUS_BASE_URL` | Nous Portal 기본 URL 재정의(일반적으로 필요하지 않으며 개발/테스트 전용) |
| `NOUS_INFERENCE_BASE_URL` | Nous 추론 엔드포인트를 직접 재정의 |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API 키 ([ai-gateway.vercel.sh](https://ai-gateway.vercel.sh)) |
| `AI_GATEWAY_BASE_URL` | AI Gateway 기본 URL 재정의(기본값: `https://ai-gateway.vercel.sh/v1`) |
| `OPENAI_API_KEY` | 사용자 지정 OpenAI 호환 엔드포인트용 API 키(`OPENAI_BASE_URL`과 함께 사용) |
| `OPENAI_BASE_URL` | 사용자 지정 엔드포인트의 기본 URL(VLLM, SGLang 등) |
| `LM_API_KEY` | LM Studio(`lmstudio` 제공자)용 API 키. 로컬 서버에서는 자리 표시자인 경우가 많습니다. |
| `LM_BASE_URL` | LM Studio 기본 URL(기본값: `http://localhost:1234/v1`) |
| `COPILOT_GITHUB_TOKEN` | Copilot API용 GitHub 토큰 — 첫 번째 우선순위(OAuth `gho_*` 또는 세분화된 PAT `github_pat_*`; 클래식 PAT `ghp_*`는 **지원되지 않음**) |
| `GH_TOKEN` | GitHub 토큰 — Copilot에서 두 번째 우선순위(`gh` CLI에서도 사용) |
| `GITHUB_TOKEN` | GitHub 토큰 — Copilot에서 세 번째 우선순위 |
| `HERMES_COPILOT_ACP_COMMAND` | Copilot ACP CLI 바이너리 경로 재정의(기본값: `copilot`) |
| `COPILOT_CLI_PATH` | `HERMES_COPILOT_ACP_COMMAND`의 별칭 |
| `HERMES_COPILOT_ACP_ARGS` | Copilot ACP 인수 재정의(기본값: `--acp --stdio`) |
| `COPILOT_ACP_BASE_URL` | Copilot ACP 기본 URL 재정의 |
| `COPILOT_API_BASE_URL` | Copilot API 기본 URL 재정의(`copilot` 제공자) |
| `GLM_API_KEY` | z.ai / ZhipuAI GLM API 키 ([z.ai](https://z.ai)) |
| `ZAI_API_KEY` | `GLM_API_KEY`의 별칭 |
| `Z_AI_API_KEY` | `GLM_API_KEY`의 별칭 |
| `GLM_BASE_URL` | z.ai 기본 URL 재정의(기본값: `https://api.z.ai/api/paas/v4`) |
| `KIMI_API_KEY` | Kimi / Moonshot AI API 키 ([moonshot.ai](https://platform.moonshot.ai)) |
| `KIMI_CODING_API_KEY` | `kimi-coding` 제공자의 별칭 키(`KIMI_API_KEY`와 함께 허용) |
| `KIMI_BASE_URL` | Kimi 기본 URL 재정의(기본값: `https://api.moonshot.ai/v1`) |
| `KIMI_CN_API_KEY` | Kimi / Moonshot 중국 API 키 ([moonshot.cn](https://platform.moonshot.cn)) |
| `ARCEEAI_API_KEY` | Arcee AI API 키 ([chat.arcee.ai](https://chat.arcee.ai/)) |
| `ARCEE_BASE_URL` | Arcee 기본 URL 재정의(기본값: `https://api.arcee.ai/api/v1`) |
| `GMI_API_KEY` | GMI Cloud API 키 ([gmicloud.ai](https://www.gmicloud.ai/)) |
| `GMI_BASE_URL` | GMI Cloud 기본 URL 재정의(기본값: `https://api.gmi-serving.com/v1`) |
| `ACTUAL_API_KEY` | Actual Computer 추론 키(`ac_...`, [actual.inc/user/keys](https://actual.inc/user/keys)). 로컬 데몬에는 필요하지 않습니다. |
| `ACTUAL_BASE_URL` | Actual Computer 기본 URL 재정의(기본값: `https://api.actual.inc/v1`). 로컬 오프라인 데몬에는 `http://127.0.0.1:8080`을 설정하세요 — 루프백 호스트에는 API 키가 필요하지 않습니다. |
| `MINIMAX_API_KEY` | MiniMax API 키 — 글로벌 엔드포인트 ([minimax.io](https://www.minimax.io)). **`minimax-oauth`에서는 사용되지 않음**(OAuth 경로는 브라우저 로그인을 사용). |
| `MINIMAX_BASE_URL` | MiniMax 기본 URL 재정의(기본값: `https://api.minimax.io/anthropic` — Hermes는 MiniMax의 Anthropic Messages 호환 엔드포인트를 사용). **`minimax-oauth`에서는 사용되지 않음**. |
| `MINIMAX_CN_API_KEY` | MiniMax API 키 — 중국 엔드포인트 ([minimaxi.com](https://www.minimaxi.com)). **`minimax-oauth`에서는 사용되지 않음**(OAuth 경로는 브라우저 로그인을 사용). |
| `MINIMAX_CN_BASE_URL` | MiniMax 중국 기본 URL 재정의(기본값: `https://api.minimaxi.com/anthropic`). **`minimax-oauth`에서는 사용되지 않음**. |
| `KILOCODE_API_KEY` | Kilo Code API 키 ([kilo.ai](https://kilo.ai)) |
| `KILOCODE_BASE_URL` | Kilo Code 기본 URL 재정의(기본값: `https://api.kilo.ai/api/gateway`) |
| `XIAOMI_API_KEY` | Xiaomi MiMo API 키 ([platform.xiaomimimo.com](https://platform.xiaomimimo.com)) |
| `XIAOMI_BASE_URL` | Xiaomi MiMo 기본 URL 재정의(기본값: `https://api.xiaomimimo.com/v1`) |
| `UPSTAGE_API_KEY` | Solar 모델용 Upstage API 키 ([console.upstage.ai](https://console.upstage.ai/api-keys)) |
| `UPSTAGE_BASE_URL` | Upstage 기본 URL 재정의(기본값: `https://api.upstage.ai/v1`) |
| `TOKENHUB_API_KEY` | Tencent TokenHub API 키 ([tokenhub.tencentmaas.com](https://tokenhub.tencentmaas.com)) |
| `TOKENHUB_BASE_URL` | Tencent TokenHub 기본 URL 재정의(기본값: `https://tokenhub.tencentmaas.com/v1`) |
| `AZURE_FOUNDRY_API_KEY` | Microsoft Foundry / Azure OpenAI API 키 ([ai.azure.com](https://ai.azure.com/)). `model.auth_mode: entra_id`인 경우 필요하지 않습니다. |
| `AZURE_FOUNDRY_BASE_URL` | Microsoft Foundry 엔드포인트 URL(예: OpenAI 스타일은 `https://<resource>.openai.azure.com/openai/v1`, Anthropic 스타일은 `https://<resource>.services.ai.azure.com/anthropic`) |
| `AZURE_ANTHROPIC_KEY` | `provider: anthropic` 및 Microsoft Foundry Claude 배포를 가리키는 `base_url`에 사용하는 Azure Anthropic API 키(Anthropic과 Azure Anthropic을 모두 구성한 경우 `ANTHROPIC_API_KEY`의 대안) |
| `AZURE_TENANT_ID` | Entra ID 테넌트 ID(서비스 주체 흐름; `model.auth_mode: entra_id`일 때 `azure-identity`가 사용) |
| `AZURE_CLIENT_ID` | Entra ID 클라이언트 ID(서비스 주체, 워크로드 ID 또는 사용자가 할당한 관리 ID) |
| `AZURE_CLIENT_SECRET` | `EnvironmentCredential`에서 사용하는 서비스 주체 비밀 정보 |
| `AZURE_CLIENT_CERTIFICATE_PATH` | 서비스 주체 인증서(`AZURE_CLIENT_SECRET`의 대안) |
| `AZURE_FEDERATED_TOKEN_FILE` | AKS Workload Identity / OIDC 흐름용 페더레이션 토큰 파일 경로 |
| `AZURE_AUTHORITY_HOST` | 소버린 클라우드 권한 호스트 재정의(예: Azure Government의 경우 `https://login.microsoftonline.us`). [Azure Foundry 가이드](/guides/azure-foundry#sovereign-clouds-government-china)를 참조하세요. |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | App Service, Functions 및 Container Apps용 관리 ID 엔드포인트; VM은 일반적으로 대신 IMDS를 사용하므로 이를 설정하지 않습니다. |
| `HF_TOKEN` | Inference Providers용 Hugging Face 토큰 ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |
| `HF_BASE_URL` | Hugging Face 기본 URL 재정의(기본값: `https://router.huggingface.co/v1`) |
| `GOOGLE_API_KEY` | Google AI Studio API 키 ([aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) |
| `GEMINI_API_KEY` | `GOOGLE_API_KEY`의 별칭 |
| `GEMINI_BASE_URL` | Google AI Studio 기본 URL 재정의 |
| `VERTEX_CREDENTIALS_PATH` | Vertex AI(Gemini)용 Google Cloud 서비스 계정 JSON 경로. Vertex는 정적 API 키가 아닌 OAuth2를 사용합니다. `GOOGLE_APPLICATION_CREDENTIALS`로 폴백한 다음 ADC(`gcloud auth application-default login`)로 폴백합니다. 프로젝트/리전은 `config.yaml`의 `vertex:` 아래에 설정하세요. |
| `ANTHROPIC_API_KEY` | Anthropic Console API 키 ([console.anthropic.com](https://console.anthropic.com/)) |
| `ANTHROPIC_BASE_URL` | Anthropic API 기본 URL 재정의 |
| `ANTHROPIC_TOKEN` | 수동 또는 레거시 Anthropic OAuth/setup-token 재정의 |
| `DASHSCOPE_API_KEY` | Qwen 모델용 Qwen Cloud(Alibaba DashScope) API 키 ([modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)) |
| `DASHSCOPE_BASE_URL` | 사용자 지정 DashScope 기본 URL(기본값: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`; 중국 본토 리전에는 `https://dashscope.aliyuncs.com/compatible-mode/v1` 사용) |
| `ALIBABA_CODING_PLAN_API_KEY` | Qwen Coding Plan API 키(`alibaba-coding-plan` 제공자) |
| `ALIBABA_CODING_PLAN_BASE_URL` | Qwen Coding Plan 기본 URL 재정의 |
| `DEEPSEEK_API_KEY` | DeepSeek에 직접 액세스하기 위한 DeepSeek API 키 ([platform.deepseek.com](https://platform.deepseek.com/api_keys)) |
| `DEEPSEEK_BASE_URL` | 사용자 지정 DeepSeek API 기본 URL |
| `DEEPINFRA_API_KEY` | DeepInfra API 키 ([deepinfra.com](https://deepinfra.com/dash/api_keys)) |
| `DEEPINFRA_BASE_URL` | DeepInfra 기본 URL 재정의 |
| `NOVITA_API_KEY` | NovitaAI API 키 — Model API, Agent Sandbox 및 GPU Cloud를 위한 AI 네이티브 클라우드 ([novita.ai/settings/key-management](https://novita.ai/settings/key-management)) |
| `NOVITA_BASE_URL` | NovitaAI 기본 URL 재정의(기본값: `https://api.novita.ai/openai/v1`) |
| `NVIDIA_API_KEY` | NVIDIA NIM API 키 — Nemotron 및 오픈 모델 ([build.nvidia.com](https://build.nvidia.com)) |
| `NVIDIA_BASE_URL` | NVIDIA 기본 URL 재정의(기본값: `https://integrate.api.nvidia.com/v1`; 로컬 NIM 엔드포인트에는 `http://localhost:8000/v1`로 설정) |
| `STEPFUN_API_KEY` | StepFun API 키 — Step 시리즈 모델 ([platform.stepfun.com](https://platform.stepfun.com)) |
| `STEPFUN_BASE_URL` | StepFun 기본 URL 재정의(기본값: `https://api.stepfun.com/v1`) |
| `OLLAMA_API_KEY` | Ollama Cloud API 키 — 로컬 GPU 없이 관리형 Ollama 카탈로그 사용 ([ollama.com/settings/keys](https://ollama.com/settings/keys)) |
| `OLLAMA_BASE_URL` | Ollama Cloud 기본 URL 재정의(기본값: `https://ollama.com/v1`) |
| `XAI_API_KEY` | 채팅 + TTS + 웹 검색용 xAI(Grok) API 키 ([console.x.ai](https://console.x.ai/)) |
| `XAI_BASE_URL` | xAI 기본 URL 재정의(기본값: `https://api.x.ai/v1`) |
| `MISTRAL_API_KEY` | Voxtral TTS 및 Voxtral STT용 Mistral API 키 ([console.mistral.ai](https://console.mistral.ai)) |
| `AWS_REGION` | Bedrock 추론용 AWS 리전(예: `us-east-1`, `eu-central-1`). boto3가 읽습니다. |
| `AWS_PROFILE` | Bedrock 인증용 AWS 명명 프로필(`~/.aws/credentials` 읽기). 기본 boto3 자격 증명 체인을 사용하려면 설정하지 않은 상태로 두세요. |
| `BEDROCK_BASE_URL` | Bedrock 런타임 기본 URL 재정의(기본값: `https://bedrock-runtime.us-east-1.amazonaws.com`; 일반적으로 설정하지 않고 대신 `AWS_REGION`을 사용) |
| `HERMES_QWEN_BASE_URL` | Qwen Portal 기본 URL 재정의(기본값: `https://portal.qwen.ai/v1`) |
| `OPENCODE_ZEN_API_KEY` | OpenCode Zen API 키 — 엄선된 모델에 대한 사용량 기반 결제 액세스 ([opencode.ai](https://opencode.ai/auth)) |
| `OPENCODE_ZEN_BASE_URL` | OpenCode Zen 기본 URL 재정의 |
| `OPENCODE_GO_API_KEY` | OpenCode Go API 키 — 오픈 모델용 월 $10 구독 ([opencode.ai](https://opencode.ai/auth)) |
| `OPENCODE_GO_BASE_URL` | OpenCode Go 기본 URL 재정의 |
| `CLAUDE_CODE_OAUTH_TOKEN` | 수동으로 내보낸 경우 Claude Code 토큰을 명시적으로 재정의 |
| `HERMES_MODEL` | 프로세스 수준에서 모델 이름 재정의(크론 스케줄러에서 사용; 일반적인 사용에는 `config.yaml`을 우선) |
| `VOICE_TOOLS_OPENAI_KEY` | OpenAI 음성-텍스트 및 텍스트-음성 제공자에 사용하는 우선 OpenAI 키 |
| `HERMES_LOCAL_STT_COMMAND` | 선택적 로컬 음성-텍스트 명령 템플릿. `{input_path}`, `{output_dir}`, `{language}`, `{model}` 자리 표시자를 지원합니다. |
| `HERMES_LOCAL_STT_LANGUAGE` | STT 기본 언어 힌트. `config.yaml`에서 제공자별 `language`를 설정하지 않은 경우 `local`(faster-whisper) 제공자, `HERMES_LOCAL_STT_COMMAND`, 로컬 `whisper` CLI 폴백(기본값: `en`), Groq 및 xAI에서 사용됩니다. |
| `HERMES_HOME` | Hermes 구성 디렉터리 재정의(기본값: `~/.hermes`). 게이트웨이 PID 파일과 systemd 서비스 이름에도 적용되므로 여러 설치를 동시에 실행할 수 있습니다. |
| `HERMES_GIT_BASH_PATH` | **Windows 전용.** 터미널 도구의 `bash.exe` 검색 재정의. 모든 bash를 가리킬 수 있습니다 — 전체 Git-for-Windows 설치, 심볼릭 링크를 통한 WSL bash, MSYS2, Cygwin. 설치 프로그램은 프로비저닝한 PortableGit을 자동으로 가리키도록 설정합니다. [Windows(네이티브) 가이드](../user-guide/windows-native.md#how-hermes-runs-shell-commands-on-windows)를 참조하세요. |
| `HERMES_DISABLE_WINDOWS_UTF8` | **Windows 전용.** UTF-8 표준 입출력 shim(`configure_windows_stdio()`)을 비활성화하고 콘솔의 로캘 코드 페이지로 폴백하려면 `1`로 설정합니다. 인코딩 버그를 이분 탐색할 때 유용하지만, 일반적인 운영에서는 설정하지 않는 것이 좋습니다. |
| `HERMES_KANBAN_HOME` | 칸반 보드를 고정하는 공유 Hermes 루트(db + 워크스페이스 + 워커 로그) 재정의. `get_default_hermes_root()`(활성 프로필의 상위 디렉터리)로 폴백합니다. 테스트와 일반적이지 않은 배포에 유용합니다. |
| `HERMES_KANBAN_BOARD` | 이 프로세스의 활성 칸반 보드를 고정. `~/.hermes/kanban/current`보다 우선하며, 디스패처는 이를 워커 하위 프로세스 환경에 주입해 워커가 다른 보드의 작업을 물리적으로 볼 수 없게 합니다. 기본값은 `default`입니다. 슬러그 검증: 소문자 영숫자 + 하이픈 + 밑줄, 1-64자. |
| `HERMES_KANBAN_DB` | 칸반 데이터베이스 파일 경로를 직접 고정(최우선순위; `HERMES_KANBAN_BOARD` 및 `HERMES_KANBAN_HOME`보다 우선). 디스패처는 이를 워커 하위 프로세스 환경에 주입해 프로필 워커가 디스패처의 보드로 수렴하도록 합니다. |
| `HERMES_KANBAN_WORKSPACES_ROOT` | 칸반 워크스페이스 루트를 직접 고정(워크스페이스에서 최우선순위; `HERMES_KANBAN_HOME`보다 우선). 디스패처는 이를 워커 하위 프로세스 환경에 주입합니다. |
| `HERMES_KANBAN_DISPATCH_IN_GATEWAY` | `kanban.dispatch_in_gateway`의 런타임 재정의. 게이트웨이가 내장 칸반 디스패처를 시작하지 않게 하려면 `0`, `false`, `no` 또는 `off`로 설정합니다. 비어 있지 않은 다른 값은 이를 활성화합니다. 별도의 디스패처 프로세스가 보드를 소유할 때 유용합니다. |
## 제공자 인증(OAuth)

기본 Anthropic 인증의 경우 Hermes는 Claude Code의 자체 자격 증명 파일이 있으면 이를 우선 사용합니다. 이러한 자격 증명은 자동으로 갱신될 수 있기 때문입니다. **Anthropic에 대한 OAuth를 사용하려면 추가 사용량 크레딧을 구매한 Claude Max 요금제가 필요합니다.** Hermes는 Claude Code로 라우팅되며, Claude Code는 Max 요금제의 기본 제공량이 아니라 Max 요금제의 추가/초과 사용량 크레딧만 사용하기 때문입니다. Claude Pro에서는 작동하지 않습니다. Max 요금제와 추가 크레딧이 없다면 대신 API 키를 사용하세요. `ANTHROPIC_TOKEN` 같은 환경 변수는 수동 재정의에 여전히 유용하지만, Claude Max 로그인에 권장되는 방식은 아닙니다.

| 변수 | 설명 |
|----------|-------------|
| `HERMES_PORTAL_BASE_URL` | Nous Portal URL 재정의(개발/테스트용) |
| `NOUS_INFERENCE_BASE_URL` | Nous 추론 API URL 재정의 |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | 에이전트 키를 다시 발급하기 전 필요한 최소 TTL(기본값: 1800 = 30분) |
| `HERMES_NOUS_TIMEOUT_SECONDS` | Nous 자격 증명/토큰 흐름의 HTTP 시간 제한 |
| `HERMES_DUMP_REQUESTS` | API 요청 페이로드를 로그 파일에 덤프(`true`/`false`) |
| `HERMES_PREFILL_MESSAGES_FILE` | API 호출 시점에 주입할 임시 prefill 메시지가 담긴 JSON 파일 경로 |
| `HERMES_TIMEZONE` | IANA 시간대 재정의(예: `America/New_York`) |

## 도구 API

| 변수 | 설명 |
|----------|-------------|
| `PARALLEL_API_KEY` | AI 네이티브 웹 검색([parallel.ai](https://parallel.ai/)) |
| `FIRECRAWL_API_KEY` | 웹 스크래핑 및 클라우드 브라우저([firecrawl.dev](https://firecrawl.dev/)) |
| `FIRECRAWL_API_URL` | 자체 호스팅 인스턴스용 사용자 지정 Firecrawl API 엔드포인트(선택 사항) |
| `TAVILY_API_KEY` | AI 네이티브 웹 검색, 추출 및 크롤링을 위한 Tavily API 키([app.tavily.com](https://app.tavily.com/home)) |
| `SEARXNG_URL` | 무료 자체 호스팅 웹 검색을 위한 SearXNG 인스턴스 URL — API 키 불필요([searxng.github.io](https://searxng.github.io/searxng/)) |
| `TAVILY_BASE_URL` | Tavily API 엔드포인트 재정의. 기업 프록시 및 자체 호스팅 Tavily 호환 검색 백엔드에 유용합니다. `GROQ_BASE_URL`과 동일한 패턴입니다. |
| `EXA_API_KEY` | AI 네이티브 웹 검색 및 콘텐츠를 위한 Exa API 키([exa.ai](https://exa.ai/)) |
| `BRAVE_SEARCH_API_KEY` | 웹 검색용 Brave Search API 구독 토큰(무료 등급 제공)([brave.com/search/api](https://brave.com/search/api/)) |
| `BROWSERBASE_API_KEY` | 브라우저 자동화([browserbase.com](https://browserbase.com/)) |
| `BROWSERBASE_PROJECT_ID` | Browserbase 프로젝트 ID |
| `BROWSER_USE_API_KEY` | Browser Use 클라우드 브라우저 API 키([browser-use.com](https://browser-use.com/)) |
| `FIRECRAWL_BROWSER_TTL` | Firecrawl 브라우저 세션 TTL(초, 기본값: 300) |
| `BROWSER_CDP_URL` | 로컬 브라우저용 Chrome DevTools Protocol URL(`/browser connect`를 통해 설정, 예: `ws://localhost:9222`) |
| `CAMOFOX_URL` | Camofox 로컬 안티 디텍션 브라우저 URL(기본값: `http://localhost:9377`) |
| `CAMOFOX_API_KEY` | 원격/인증된 Camofox 서버로 보내는 Authorization 헤더의 선택적 bearer 토큰 |
| `CAMOFOX_USER_ID` | 공유되는 표시 세션을 위한 외부 관리 Camofox 사용자 ID(선택 사항) |
| `CAMOFOX_SESSION_KEY` | `CAMOFOX_USER_ID`용 탭 생성 시 사용하는 선택적 Camofox 세션 키 |
| `CAMOFOX_ADOPT_EXISTING_TAB` | 새 탭을 만들기 전에 기존 Camofox 탭을 재사용하려면 `true`로 설정 |
| `BROWSER_INACTIVITY_TIMEOUT` | 브라우저 세션 비활성 시간 제한(초) |
| `AGENT_BROWSER_ARGS` | 추가 Chromium 실행 플래그(쉼표 또는 줄바꿈으로 구분). Hermes는 root로 실행되거나 AppArmor가 제한한 권한 없는 사용자 네임스페이스(Ubuntu 23.10+, DGX Spark, 많은 컨테이너 이미지)에서 실행될 때 `--no-sandbox,--disable-dev-shm-usage`를 자동으로 주입합니다. 이를 수동으로 설정하는 경우는 재정의하거나 다른 플래그를 추가할 때뿐입니다. |
| `AGENT_BROWSER_ENGINE` | 로컬 모드의 브라우저 엔진: `auto`(기본값 — CDP를 통한 Chromium 계열) 또는 특정 엔진 재정의 |
| `FAL_KEY` | 이미지 생성([fal.ai](https://fal.ai/)) |
| `KREA_API_KEY` | Krea 2 이미지 생성을 위한 Krea API 키([krea.ai](https://krea.ai/)) |
| `GROQ_API_KEY` | Groq Whisper STT API 키([groq.com](https://groq.com/)) |
| `ELEVENLABS_API_KEY` | ElevenLabs 프리미엄 TTS 음성([elevenlabs.com](https://elevenlabs.io/)) |
| `PORCUPINE_ACCESS_KEY` | Picovoice Porcupine 웨이크 워드 엔진([console.picovoice.ai](https://console.picovoice.ai/)) — `wake_word.provider: porcupine`일 때만 필요합니다. 기본 openWakeWord 및 sherpa 엔진에는 키가 필요하지 않습니다. |
| `STT_GROQ_MODEL` | Groq STT 모델 재정의(기본값: `whisper-large-v3-turbo`) |
| `GROQ_BASE_URL` | Groq OpenAI 호환 STT 엔드포인트 재정의 |
| `STT_OPENAI_MODEL` | OpenAI STT 모델 재정의(기본값: `whisper-1`) |
| `STT_OPENAI_BASE_URL` | OpenAI 호환 STT 엔드포인트 재정의 |
| `GITHUB_TOKEN` | Skills Hub용 GitHub 토큰(더 높은 API 요청 한도, 스킬 게시) |
| `HONCHO_API_KEY` | 세션 간 사용자 모델링([honcho.dev](https://honcho.dev/)) |
| `HONCHO_BASE_URL` | 자체 호스팅 Honcho 인스턴스의 기본 URL(기본값: Honcho 클라우드). 로컬 인스턴스에는 API 키가 필요하지 않습니다. |
| `HINDSIGHT_API_KEY` | 그래프 인식 영구 메모리를 위한 Hindsight API 키([hindsight.vectorize.io](https://hindsight.vectorize.io)) |
| `HINDSIGHT_API_URL` | Hindsight API의 기본 URL(기본값: `https://api.hindsight.vectorize.io`) |
| `HINDSIGHT_TIMEOUT` | Hindsight 메모리 제공자 API 호출의 시간 제한(초, 기본값: `60`). Hindsight 인스턴스가 `/sync` 또는 `on_session_switch` 중 느리게 응답하여 `errors.log`에 시간 제한 오류가 표시된다면 이 값을 늘리세요. |
| `MEM0_API_KEY` | 의미 기반 영구 메모리를 위한 Mem0 Platform API 키([app.mem0.ai](https://app.mem0.ai)) |
| `MEM0_MODE` | Mem0 백엔드 모드: `platform`(기본값) 또는 `oss` — [메모리 제공자](/user-guide/features/memory-providers) 참고 |
| `MEM0_HOST` | 자체 호스팅 Mem0 서버의 기본 URL(Platform API 대신 플러그인을 사용하도록 전환) |
| `MEM0_USER_ID` | Mem0 메모리를 저장할 사용자 ID 재정의 |
| `MEM0_AGENT_ID` | Mem0 메모리에 태그할 에이전트 ID 재정의 |
| `RETAINDB_API_KEY` | 영구 메모리를 위한 RetainDB API 키([retaindb.com](https://retaindb.com)) |
| `RETAINDB_BASE_URL` | 자체 호스팅 RetainDB 인스턴스의 기본 URL(기본값: `https://api.retaindb.com`) |
| `OPENVIKING_API_KEY` | OpenViking API 키(로컬 개발 모드에서는 비워 둠) |
| `OPENVIKING_ENDPOINT` | OpenViking 서버 URL(기본값: `http://127.0.0.1:1933`) |
| `BRV_API_KEY` | ByteRover API 키(선택 사항, 클라우드 동기화용 — 기본값은 로컬 우선)([app.byterover.dev](https://app.byterover.dev)) |
| `SUPERMEMORY_API_KEY` | 프로필 회상 및 세션 수집을 지원하는 의미 기반 장기 메모리([supermemory.ai](https://supermemory.ai)) |
| `DAYTONA_API_KEY` | Daytona 클라우드 샌드박스([daytona.io](https://daytona.io/)) |
| `VERCEL_TOKEN` | Vercel Sandbox 액세스 토큰([vercel.com](https://vercel.com/)) |
| `VERCEL_PROJECT_ID` | Vercel 프로젝트 ID(`VERCEL_TOKEN`과 함께 필요) |
| `VERCEL_TEAM_ID` | Vercel 팀 ID(`VERCEL_TOKEN`과 함께 필요) |
| `VERCEL_OIDC_TOKEN` | 단기 Vercel OIDC 토큰(개발 전용 대안) |

### 스킬 API 키

특정 번들/선택적 스킬에서 사용하는 비밀입니다. 해당 스킬을 사용할 때만 필요합니다.

| 변수 | 스킬에서 사용 | 설명 |
|----------|---------------|-------------|
| `NOTION_API_KEY` | `notion` | Notion 통합 토큰 |
| `LINEAR_API_KEY` | `linear` | Linear 개인 API 키 |
| `AIRTABLE_API_KEY` | `airtable` | Airtable 개인 액세스 토큰 |
| `TENOR_API_KEY` | `gif-search` | GIF 검색을 위한 Tenor API 키 |

### Langfuse 관측성

번들된 [`observability/langfuse`](/user-guide/features/built-in-plugins#observabilitylangfuse) 플러그인을 위한 환경 변수입니다. 이 값을 `~/.hermes/.env`에 설정하세요. 이 값들이 적용되려면 플러그인도 활성화되어 있어야 합니다(`hermes plugins enable observability/langfuse`를 실행하거나 `hermes plugins`에서 상자를 선택).

| 변수 | 설명 |
|----------|-------------|
| `HERMES_LANGFUSE_PUBLIC_KEY` | Langfuse 프로젝트 공개 키(`pk-lf-...`). 필수입니다. |
| `HERMES_LANGFUSE_SECRET_KEY` | Langfuse 프로젝트 비밀 키(`sk-lf-...`). 필수입니다. |
| `HERMES_LANGFUSE_BASE_URL` | Langfuse 서버 URL(기본값: `https://cloud.langfuse.com`). 자체 호스팅에 맞게 설정합니다. |
| `HERMES_LANGFUSE_ENV` | 트레이스의 환경 태그(`production`, `staging`, …) |
| `HERMES_LANGFUSE_RELEASE` | 릴리스/버전 태그 |
| `HERMES_LANGFUSE_SAMPLE_RATE` | SDK 샘플링 비율 0.0–1.0(기본값: `1.0`) |
| `HERMES_LANGFUSE_MAX_CHARS` | 직렬화된 페이로드의 필드별 잘라내기 길이(기본값: `12000`) |
| `HERMES_LANGFUSE_DEBUG` | `true`로 설정하면 자세한 플러그인 로그를 `agent.log`에 기록 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | 표준 Langfuse SDK 이름. `HERMES_LANGFUSE_*`에 해당하는 값이 설정되지 않은 경우 대체값으로 허용됩니다. |

### Nous 도구 게이트웨이

이 변수들은 유료 Nous 구독자 또는 자체 호스팅 게이트웨이 배포를 위한 [도구 게이트웨이](/user-guide/features/tool-gateway)를 구성합니다. 대부분의 사용자는 이를 설정할 필요가 없습니다. 게이트웨이는 `hermes model` 또는 `hermes tools`를 통해 자동으로 구성됩니다.

| 변수 | 설명 |
|----------|-------------|
| `TOOL_GATEWAY_DOMAIN` | 도구 게이트웨이 라우팅의 기본 도메인(기본값: `nousresearch.com`) |
| `TOOL_GATEWAY_SCHEME` | 게이트웨이 URL의 HTTP 또는 HTTPS 스킴(기본값: `https`) |
| `TOOL_GATEWAY_USER_TOKEN` | 도구 게이트웨이 인증 토큰(일반적으로 Nous 인증에서 자동으로 채워짐) |
| `FIRECRAWL_GATEWAY_URL` | Firecrawl 게이트웨이 엔드포인트 전용 URL 재정의 |

## 터미널 백엔드

| 변수 | 설명 |
|----------|-------------|
| `TERMINAL_ENV` | 백엔드: `local`, `docker`, `ssh`, `singularity`, `modal`, `daytona`, `vercel_sandbox` |
| `HERMES_DOCKER_BINARY` | Hermes가 셸을 실행할 때 호출하는 컨테이너 바이너리 재정의(예: `podman`, `/usr/local/bin/docker`). 설정하지 않으면 Hermes가 `PATH`에서 `docker` 또는 `podman`을 자동으로 검색합니다. 두 바이너리가 모두 설치되어 기본값이 아닌 것을 사용하려 하거나 바이너리가 `PATH` 외부에 있을 때 필요합니다. |
| `TERMINAL_DOCKER_IMAGE` | Docker 이미지(기본값: `nikolaik/python-nodejs:python3.11-nodejs20`) |
| `TERMINAL_DOCKER_FORWARD_ENV` | Docker 터미널 세션으로 명시적으로 전달할 환경 변수 이름의 JSON 배열. 참고: 스킬에 선언된 `required_environment_variables`는 자동으로 전달되므로, 어떤 스킬에도 선언되지 않은 변수에만 이 설정이 필요합니다. |
| `TERMINAL_DOCKER_VOLUMES` | 추가 Docker 볼륨 마운트(쉼표로 구분된 `host:container` 쌍) |
| `TERMINAL_DOCKER_ENV` | Docker 터미널 세션 내부에 설정할 추가 환경 변수의 JSON 객체(예: `{"FOO":"bar"}`) |
| `TERMINAL_DOCKER_EXTRA_ARGS` | 추가 `docker run` 인자의 JSON 배열(예: `["--memory","4g"]`) |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | 고급 선택 사항: 실행 cwd를 Docker의 `/workspace`에 마운트(`true`/`false`, 기본값: `false`) |
| `TERMINAL_SINGULARITY_IMAGE` | Singularity 이미지 또는 `.sif` 경로 |
| `TERMINAL_MODAL_IMAGE` | Modal 컨테이너 이미지 |
| `TERMINAL_DAYTONA_IMAGE` | Daytona 샌드박스 이미지 |
| `TERMINAL_VERCEL_RUNTIME` | Vercel Sandbox 런타임(`node24`, `node22`, `python3.13`) |
| `TERMINAL_TIMEOUT` | 명령 시간 제한(초) |
| `TERMINAL_LIFETIME_SECONDS` | 터미널 세션의 최대 수명(초) |
| `TERMINAL_CWD` | 게이트웨이/cron 터미널 세션의 사용 중단된 직접 재정의. `config.yaml`의 `terminal.cwd`를 대신 사용하세요. CLI는 여전히 실행 디렉터리를 사용합니다. |
| `SUDO_PASSWORD` | 대화형 프롬프트 없이 sudo 활성화 |

클라우드 샌드박스 백엔드의 지속성은 파일 시스템을 기준으로 합니다. `TERMINAL_LIFETIME_SECONDS`는 Hermes가 유휴 터미널 세션을 정리하는 시점을 제어하며, 나중에 재개할 때 동일한 실행 중 프로세스를 유지하는 대신 샌드박스를 다시 만들 수 있습니다.

## SSH 백엔드

| 변수 | 설명 |
|----------|-------------|
| `TERMINAL_SSH_HOST` | 원격 서버 호스트 이름 |
| `TERMINAL_SSH_USER` | SSH 사용자 이름 |
| `TERMINAL_SSH_PORT` | SSH 포트(기본값: 22) |
| `TERMINAL_SSH_KEY` | 개인 키 경로 |
| `TERMINAL_SSH_PERSISTENT` | SSH의 지속 셸 재정의(기본값: `TERMINAL_PERSISTENT_SHELL`을 따름) |
## 컨테이너 리소스(Docker, Singularity, Modal, Daytona)

| 변수 | 설명 |
|----------|-------------|
| `TERMINAL_CONTAINER_CPU` | CPU 코어 수(기본값: 1) |
| `TERMINAL_CONTAINER_MEMORY` | 메모리(MB) |
| `TERMINAL_CONTAINER_DISK` | 디스크(MB) |
| `TERMINAL_CONTAINER_PERSISTENT` | 컨테이너 파일 시스템을 유지할지 여부(기본값: `true`) |
| `TERMINAL_SANDBOX_DIR` | 작업 공간과 오버레이를 저장할 호스트 디렉터리(기본값: `~/.hermes/sandboxes/`) |

## 영구 셸

| 변수 | 설명 |
|----------|-------------|
| `TERMINAL_PERSISTENT_SHELL` | 로컬이 아닌 백엔드에서 영구 셸을 활성화합니다(기본값: `true`). `config.yaml`의 `terminal.persistent_shell`로도 설정할 수 있습니다. |
| `TERMINAL_LOCAL_PERSISTENT` | 로컬 백엔드에서 영구 셸을 활성화합니다(기본값: `false`). |
| `TERMINAL_SSH_PERSISTENT` | SSH의 영구 셸 설정을 재정의합니다(기본값: `TERMINAL_PERSISTENT_SHELL`을 따름). |

## Egress 프록시(샌드박스에서 주입)

이 환경 변수들은 호스트에 설정되지 않으며, [Egress 프록시](../user-guide/egress/iron-proxy.md) 통합을 통해 `proxy.enabled: true`일 때 Docker 샌드박스에 주입됩니다. 이 릴리스에서 연결된 백엔드는 Docker뿐입니다.

| 변수 | 설명 |
|----------|-------------|
| `HERMES_EGRESS_PROXY` | Egress 프록시가 활성화된 샌드박스 내부에서 `1`로 설정됩니다. 에이전트 코드는 TLS를 가로채는 프록시 뒤에서 실행 중인지 확인할 수 있습니다. |
| 제공업체 환경 변수(`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, …) | 실제 upstream 시크릿이 아닌 불투명한 프록시 토큰으로 설정되므로, 기존 SDK가 표준 환경 변수 이름을 계속 읽을 수 있습니다. iron-proxy가 네트워크 경계에서 이 토큰을 실제 upstream 시크릿으로 교체합니다. |
| `HERMES_PROXY_TOKEN_<ENV_NAME>` | 발급된 각 제공업체 매핑의 진단용 별칭입니다. 예: `HERMES_PROXY_TOKEN_OPENROUTER_API_KEY=hermes-proxy-openrouter-…`. 표준 제공업체 환경 변수와 동일한 토큰 값입니다. |
| `HTTPS_PROXY` / `HTTP_PROXY` | `HTTPS_PROXY`는 CONNECT/MITM을 위해 `http://host.docker.internal:<tunnel_port>`를 가리킵니다. `HTTP_PROXY`는 일반 HTTP 전달을 위해 `<tunnel_port + 1>`을 가리킵니다. |
| `NO_PROXY` | `127.0.0.1,localhost,::1`로 설정되어 샌드박스 내부의 루프백 개발 서버가 프록시를 우회합니다. |
| `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE` / `NODE_EXTRA_CA_CERTS` | 샌드박스에 마운트된 Hermes Egress CA 인증서의 경로입니다(`/etc/ssl/certs/hermes-egress-ca.crt`). 언어 런타임이 iron-proxy가 MITM을 통해 발급한 리프 인증서를 신뢰하도록 합니다. |
| `NODE_OPTIONS` | `--use-openssl-ca`가 추가됩니다(기존 플래그는 유지됨). 따라서 Node.js는 다른 CA 번들 변수가 제어하는 OpenSSL 저장소를 사용합니다. [Node.js 비대칭 CA 주의 사항](../user-guide/egress/iron-proxy.md#nodejs-asymmetric-ca-caveat)의 범위를 좁힙니다. |
| `HERMES_IRON_PROXY_NONCE` | iron-proxy 데몬 프로세스 자체에 설정됩니다(샌드박스 내부에는 설정되지 않음). `_pid_alive`가 PID 재활용 상황에서도 후보 PID가 *당사가 관리하는* 바이너리를 가리키는지 확인하는 데 사용됩니다. |

이 변수들은 `proxy.enabled: true`이고 데몬이 실행 중일 때 Docker 터미널 백엔드가 자동으로 설정합니다. 직접 설정할 필요가 없습니다. 운영자가 조정하는 관련 설정은 `~/.hermes/config.yaml`의 `proxy:` 섹션에 있습니다. 자세한 내용은 [Egress 프록시 → 구성](../user-guide/egress/iron-proxy.md#configuration)을 참조하세요.

## 메시징

| 변수 | 설명 |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram 봇 토큰(@BotFather에서 발급) |
| `TELEGRAM_ALLOWED_USERS` | 봇을 사용할 수 있는 사용자 ID를 쉼표로 구분한 목록(DM, 그룹, 포럼에 적용) |
| `TELEGRAM_ALLOW_ALL_USERS` | 모든 Telegram 사용자가 봇을 호출할 수 있도록 허용합니다(개발 전용). |
| `TELEGRAM_GROUP_ALLOWED_USERS` | 그룹/포럼에서만 권한이 있는 발신자 사용자 ID를 쉼표로 구분한 목록(DM 접근 권한은 부여하지 않음). `-`로 시작하는 채팅 ID 형식의 값은 pre-#17686 설정과의 하위 호환성을 위해 계속 채팅 ID로 처리되며, 지원 중단 경고가 표시됩니다. |
| `TELEGRAM_GROUP_ALLOWED_CHATS` | 쉼표로 구분한 그룹/포럼 채팅 ID입니다. 모든 구성원이 권한을 가집니다. |
| `TELEGRAM_HOME_CHANNEL` | cron 전달에 사용할 기본 Telegram 채팅/채널 |
| `TELEGRAM_HOME_CHANNEL_NAME` | Telegram 홈 채널의 표시 이름 |
| `TELEGRAM_CRON_THREAD_ID` | cron 전달을 받을 포럼 주제 ID입니다. cron에 한해서 `TELEGRAM_HOME_CHANNEL_THREAD_ID`를 재정의합니다. 주제 모드에서 사용하면 cron 메시지에 대한 답장이 시스템 로비로 가지 않고 새 세션을 엽니다(#24409). |
| `TELEGRAM_WEBHOOK_URL` | webhook 모드의 공개 HTTPS URL입니다(폴링 대신 webhook을 활성화). |
| `TELEGRAM_WEBHOOK_PORT` | webhook 서버의 로컬 수신 포트(기본값: `8443`) |
| `TELEGRAM_WEBHOOK_SECRET` | Telegram이 각 업데이트에 다시 포함하는 검증용 시크릿 토큰입니다. `TELEGRAM_WEBHOOK_URL`이 설정된 경우 **항상 필요합니다**. 이 값이 없으면 게이트웨이가 시작되지 않습니다(GHSA-3vpc-7q5r-276h). `openssl rand -hex 32`로 생성하세요. |
| `TELEGRAM_REACTIONS` | 처리 중인 메시지에 이모지 반응을 활성화합니다(기본값: `false`). |
| `TELEGRAM_REQUIRE_MENTION` | Telegram 그룹에서 응답하기 전에 명시적인 트리거를 요구합니다. `config.yaml`의 `telegram.require_mention`과 동일합니다. |
| `TELEGRAM_MENTION_PATTERNS` | Telegram 그룹 멘션 게이팅이 활성화된 경우 허용할 정규식 깨우기 단어 패턴의 JSON 배열, 줄바꿈 구분 목록 또는 쉼표 구분 목록입니다. `telegram.mention_patterns`와 동일합니다. |
| `TELEGRAM_EXCLUSIVE_BOT_MENTIONS` | 활성화하면 Telegram 그룹의 명시적인 `@...bot` 멘션이 답장 또는 깨우기 단어 대체 동작보다 먼저 멘션된 봇 사용자 이름으로만 라우팅됩니다. 기본값: `true`. `telegram.exclusive_bot_mentions`와 동일합니다. |
| `TELEGRAM_REPLY_TO_MODE` | 답장 참조 동작: `off`, `first`(기본값) 또는 `all`. Discord 패턴과 일치합니다. |
| `TELEGRAM_IGNORED_THREADS` | 봇이 절대 응답하지 않는 Telegram 포럼 주제/스레드 ID를 쉼표로 구분한 목록 |
| `TELEGRAM_PROXY` | Telegram 연결용 프록시 URL입니다. `HTTPS_PROXY`를 재정의하며 `http://`, `https://`, `socks5://`를 지원합니다. |
| `DISCORD_BOT_TOKEN` | Discord 봇 토큰 |
| `DISCORD_ALLOWED_USERS` | 봇을 사용할 수 있는 Discord 사용자 ID를 쉼표로 구분한 목록 |
| `DISCORD_ALLOW_ALL_USERS` | 모든 Discord 사용자가 봇을 호출할 수 있도록 허용합니다(개발 전용). |
| `DISCORD_ALLOWED_ROLES` | 봇을 사용할 수 있는 Discord 역할 ID를 쉼표로 구분한 목록입니다(`DISCORD_ALLOWED_USERS`와 OR 조건). Members 인텐트를 자동으로 활성화합니다. 중재 팀이 자주 바뀌는 경우 유용하며 역할 권한 부여가 자동으로 적용됩니다. |
| `DISCORD_ALLOWED_CHANNELS` | Discord 채널 ID를 쉼표로 구분한 목록입니다. 설정하면 봇은 이 채널에서만 응답합니다(허용된 DM 제외). `config.yaml`의 `discord.allowed_channels`를 재정의합니다. |
| `DISCORD_PROXY` | Discord 연결용 프록시 URL입니다. `HTTPS_PROXY`를 재정의하며 `http://`, `https://`, `socks5://`를 지원합니다. |
| `DISCORD_HOME_CHANNEL` | cron 전달에 사용할 기본 Discord 채널 |
| `DISCORD_HOME_CHANNEL_NAME` | Discord 홈 채널의 표시 이름 |
| `DISCORD_COMMAND_SYNC_POLICY` | Discord 슬래시 명령 시작 시 동기화 정책: `safe`(차이를 확인하고 조정), `bulk`(기존 `tree.sync()`), 또는 `off` |
| `DISCORD_REQUIRE_MENTION` | 서버 채널에서 응답하기 전에 @멘션을 요구합니다. |
| `DISCORD_FREE_RESPONSE_CHANNELS` | 멘션이 필요하지 않은 채널 ID를 쉼표로 구분한 목록 |
| `DISCORD_AUTO_THREAD` | 지원되는 경우 긴 답변을 자동으로 스레드로 만듭니다. |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | `true`이면 첨부 파일 형식에 관계없이 허용합니다(기본 제공 PDF/텍스트/zip/office 허용 목록에 한정하지 않음). 알 수 없는 형식은 캐시되고 로컬 경로로 에이전트에 전달되어 `terminal` / `read_file` / `ffprobe`로 검사할 수 있습니다. 기본값 `false`. |
| `DISCORD_MAX_ATTACHMENT_BYTES` | 게이트웨이가 캐시할 첨부 파일 하나당 최대 바이트 수입니다. 기본값 `33554432`(32 MiB). `0`으로 설정하면 제한이 없습니다(첨부 파일을 쓰는 동안 메모리에 보관). |
| `DISCORD_REACTIONS` | 처리 중인 메시지에 이모지 반응을 활성화합니다(기본값: `true`). |
| `DISCORD_IGNORED_CHANNELS` | 봇이 절대 응답하지 않는 채널 ID를 쉼표로 구분한 목록 |
| `DISCORD_NO_THREAD_CHANNELS` | 봇이 자동 스레드 없이 응답하는 채널 ID를 쉼표로 구분한 목록 |
| `DISCORD_REPLY_TO_MODE` | 답장 참조 동작: `off`, `first`(기본값) 또는 `all` |
| `DISCORD_ALLOW_MENTION_EVERYONE` | 봇이 `@everyone`/`@here`를 핑할 수 있도록 허용합니다(기본값: `false`). [멘션 제어](../user-guide/messaging/discord.md#mention-control)를 참조하세요. |
| `DISCORD_ALLOW_MENTION_ROLES` | 봇이 `@role` 멘션을 핑할 수 있도록 허용합니다(기본값: `false`). |
| `DISCORD_ALLOW_MENTION_USERS` | 봇이 개별 `@user` 멘션을 핑할 수 있도록 허용합니다(기본값: `true`). |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | 답장할 때 작성자를 핑합니다(기본값: `true`). |
| `SLACK_BOT_TOKEN` | Slack 봇 토큰(`xoxb-...`) |
| `SLACK_APP_TOKEN` | Slack 앱 수준 토큰(`xapp-...`, Socket Mode에 필요) |
| `SLACK_ALLOWED_USERS` | Slack 사용자 ID를 쉼표로 구분한 목록 |
| `SLACK_ALLOW_ALL_USERS` | 모든 Slack 사용자가 봇을 호출할 수 있도록 허용합니다(개발 전용). |
| `SLACK_ALLOW_BOTS` | 다른 Slack 봇의 메시지를 허용합니다: `none`(기본값), `mentions` 또는 `all`. 봇은 자신의 메시지는 항상 무시합니다. |
| `SLACK_THREAD_REQUIRE_MENTION` | Slack 스레드 답장에 명시적인 @멘션을 요구하면서 최상위 자유 응답 채널은 유지합니다. |
| `SLACK_HOME_CHANNEL` | cron 전달에 사용할 기본 Slack 채널 |
| `SLACK_HOME_CHANNEL_NAME` | Slack 홈 채널의 표시 이름 |
| `GOOGLE_CHAT_PROJECT_ID` | Pub/Sub 주제를 호스팅하는 GCP 프로젝트(`GOOGLE_CLOUD_PROJECT`로 대체 가능) |
| `GOOGLE_CHAT_SUBSCRIPTION_NAME` | 전체 Pub/Sub 구독 경로인 `projects/{proj}/subscriptions/{sub}`(레거시 별칭: `GOOGLE_CHAT_SUBSCRIPTION`) |
| `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` | Service Account JSON 경로 또는 JSON 인라인 값(`GOOGLE_APPLICATION_CREDENTIALS`로 대체 가능) |
| `GOOGLE_CHAT_ALLOWED_USERS` | 봇과 채팅할 수 있는 사용자 이메일을 쉼표로 구분한 목록 |
| `GOOGLE_CHAT_ALLOW_ALL_USERS` | 모든 Google Chat 사용자가 봇을 호출할 수 있도록 허용합니다(개발 전용). |
| `GOOGLE_CHAT_HOME_CHANNEL` | cron 전달에 사용할 기본 스페이스(예: `spaces/AAAA...`) |
| `GOOGLE_CHAT_HOME_CHANNEL_NAME` | Google Chat 홈 스페이스의 표시 이름 |
| `GOOGLE_CHAT_MAX_MESSAGES` | Pub/Sub FlowControl의 처리 중 최대 메시지 수(기본값: `1`) |
| `GOOGLE_CHAT_MAX_BYTES` | Pub/Sub FlowControl의 처리 중 최대 바이트 수(기본값: `16777216`, 16 MiB) |
| `GOOGLE_CHAT_BOOTSTRAP_SPACES` | 봇 자체 `users/{id}`를 확인할 때 시작 시 추가로 조회할 스페이스 ID를 쉼표로 구분한 목록 |
| `GOOGLE_CHAT_DEBUG_RAW` | 값을 지정하면 삭제된 정보가 포함된 Pub/Sub envelope를 DEBUG 수준으로 기록합니다(디버깅 전용). |
| `GOOGLE_CHAT_HTTP_EVENTS_URL` | Chat 메시지 이벤트를 위한 인증된 HTTP 엔드포인트(Pub/Sub의 대안) |
| `GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE` | Google 서명 HTTP 이벤트 bearer 토큰에 기대하는 audience입니다(기본값은 `GOOGLE_CHAT_HTTP_EVENTS_URL`). |
| `GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL` | HTTP 이벤트 bearer 토큰에 기대하는 Google 서비스 계정 이메일 |
| `WHATSAPP_ENABLED` | WhatsApp 브리지를 활성화합니다(`true`/`false`). |
| `WHATSAPP_MODE` | `bot`(별도 번호) 또는 `self-chat`(자신에게 메시지 보내기) |
| `WHATSAPP_ALLOWED_USERS` | 국가 코드가 포함되고 `+`가 없는 전화번호를 쉼표로 구분한 목록 또는 모든 발신자를 허용하는 `*` |
| `WHATSAPP_ALLOW_ALL_USERS` | 허용 목록 없이 모든 WhatsApp 발신자를 허용합니다(`true`/`false`). |
| `WHATSAPP_HOME_CHANNEL` | cron/알림 전달에 사용할 기본 채팅 ID |
| `WHATSAPP_HOME_CHANNEL_NAME` | WhatsApp 홈 채널의 표시 이름 |
| `WHATSAPP_DEBUG` | 문제 해결을 위해 브리지에서 원시 메시지 이벤트를 기록합니다(`true`/`false`). |
| `WHATSAPP_CLOUD_PHONE_NUMBER_ID` | WhatsApp Business Cloud API의 Meta Phone Number ID(15~17자리, **전화번호 자체가 아님**) |
| `WHATSAPP_CLOUD_ACCESS_TOKEN` | Meta 액세스 토큰(`EAA`로 시작); 임시 토큰은 24시간 후 만료되며 System User 토큰은 영구적입니다. |
| `WHATSAPP_CLOUD_APP_SECRET` | 수신 webhook 서명을 검증하는 데 사용하는 32자 16진수 앱 시크릿 |
| `WHATSAPP_CLOUD_VERIFY_TOKEN` | Meta webhook 검증 핸드셰이크에 사용하는 공유 시크릿(설정 마법사가 자동 생성) |
| `WHATSAPP_CLOUD_ALLOWED_USERS` | 봇에 메시지를 보낼 수 있는 `wa_id`(국가 코드가 포함되고 `+`가 없는 전화번호)를 쉼표로 구분한 목록 |
| `WHATSAPP_CLOUD_ALLOW_ALL_USERS` | 허용 목록 없이 모든 WhatsApp Cloud 발신자를 허용합니다(`true`/`false`). |
| `WHATSAPP_CLOUD_APP_ID` | 선택적 Meta App ID(향후 분석 통합용) |
| `WHATSAPP_CLOUD_WABA_ID` | 선택적 WhatsApp Business Account ID(향후 분석 통합용) |
| `WHATSAPP_CLOUD_WEBHOOK_HOST` | 수신 webhook 서버가 바인딩되는 인터페이스(기본값 `0.0.0.0`) |
| `WHATSAPP_CLOUD_WEBHOOK_PORT` | 수신 webhook 서버가 바인딩되는 포트(기본값 `8090`) |
| `WHATSAPP_CLOUD_WEBHOOK_PATH` | Meta가 수신 메시지를 POST하는 URL 경로(기본값 `/whatsapp/webhook`) |
| `WHATSAPP_CLOUD_API_VERSION` | 호출할 Meta Graph API 버전(기본값 `v20.0`) |
| `WHATSAPP_CLOUD_HOME_CHANNEL` | 봇의 홈 채널로 사용할 `wa_id`(cron 작업 등) |
| `WHATSAPP_CLOUD_DM_POLICY` | Cloud 어댑터의 DM 게이팅: `open`/`allowlist`/`disabled`; 설정하지 않으면 `WHATSAPP_DM_POLICY`로 대체됩니다. |
| `WHATSAPP_CLOUD_ALLOW_FROM` | `dm_policy: allowlist`일 때 허용할 발신자를 쉼표로 구분한 목록(일반 `wa_id`; Baileys 형식 JID는 정규화됨) |
| `WHATSAPP_CLOUD_GROUP_POLICY` | Cloud 어댑터의 그룹 게이팅: `open`/`allowlist`/`disabled`; 설정하지 않으면 `WHATSAPP_GROUP_POLICY`로 대체됩니다. |
| `WHATSAPP_CLOUD_GROUP_ALLOW_FROM` | `group_policy: allowlist`일 때 허용할 그룹 채팅 ID를 쉼표로 구분한 목록 |
| `SIGNAL_HTTP_URL` | signal-cli 데몬 HTTP 엔드포인트(예: `http://127.0.0.1:8080`) |
| `SIGNAL_ACCOUNT` | E.164 형식의 봇 전화번호 |
| `SIGNAL_ALLOWED_USERS` | E.164 전화번호 또는 UUID를 쉼표로 구분한 목록 |
| `SIGNAL_GROUP_ALLOWED_USERS` | 그룹 ID를 쉼표로 구분한 목록 또는 모든 그룹을 허용하는 `*` |
| `SIGNAL_HOME_CHANNEL_NAME` | Signal 홈 채널의 표시 이름 |
| `SIGNAL_IGNORE_STORIES` | Signal 스토리/상태 업데이트를 무시합니다. |
| `SIGNAL_ALLOW_ALL_USERS` | 허용 목록 없이 모든 Signal 사용자를 허용합니다. |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID(telephony 스킬과 공유) |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token(telephony 스킬과 공유하며 webhook 서명 검증에도 사용) |
| `TWILIO_PHONE_NUMBER` | E.164 형식의 Twilio 전화번호(telephony 스킬과 공유) |
| `SMS_WEBHOOK_URL` | Twilio 서명 검증용 공개 URL입니다. Twilio Console의 webhook URL과 일치해야 합니다(필수). |
| `SMS_WEBHOOK_PORT` | 수신 SMS용 webhook 리스너 포트(기본값: `8080`) |
| `SMS_WEBHOOK_HOST` | webhook 바인드 주소(기본값: `127.0.0.1`) |
| `SMS_INSECURE_NO_SIGNATURE` | Twilio 서명 검증을 비활성화하려면 `true`로 설정합니다(로컬 개발 전용, 프로덕션에서는 사용하지 않음). |
| `SMS_ALLOWED_USERS` | 채팅할 수 있는 E.164 전화번호를 쉼표로 구분한 목록 |
| `SMS_ALLOW_ALL_USERS` | 허용 목록 없이 모든 SMS 발신자를 허용합니다. |
| `SMS_HOME_CHANNEL` | cron 작업/알림 전달에 사용할 전화번호 |
| `SMS_HOME_CHANNEL_NAME` | SMS 홈 채널의 표시 이름 |
| `EMAIL_ADDRESS` | Email 게이트웨이 어댑터의 이메일 주소 |
| `EMAIL_PASSWORD` | 이메일 계정의 비밀번호 또는 앱 비밀번호 |
| `EMAIL_IMAP_HOST` | 이메일 어댑터의 IMAP 호스트 이름 |
| `EMAIL_IMAP_PORT` | IMAP 포트 |
| `EMAIL_SMTP_HOST` | 이메일 어댑터의 SMTP 호스트 이름 |
| `EMAIL_SMTP_PORT` | SMTP 포트 |
| `EMAIL_ALLOWED_USERS` | 봇에게 메시지를 보낼 수 있는 이메일 주소를 쉼표로 구분한 목록 |
| `EMAIL_HOME_ADDRESS` | 사전 예방적 이메일 전달에 사용할 기본 수신자 |
| `EMAIL_HOME_ADDRESS_NAME` | 이메일 홈 대상의 표시 이름 |
| `EMAIL_POLL_INTERVAL` | 이메일 폴링 간격(초) |
| `EMAIL_ALLOW_ALL_USERS` | 수신 이메일의 모든 발신자를 허용합니다. |
| `DINGTALK_CLIENT_ID` | 개발자 포털의 DingTalk 봇 AppKey([open.dingtalk.com](https://open.dingtalk.com)) |
| `DINGTALK_CLIENT_SECRET` | 개발자 포털의 DingTalk 봇 AppSecret |
| `DINGTALK_ALLOWED_USERS` | 봇에게 메시지를 보낼 수 있는 DingTalk 사용자 ID를 쉼표로 구분한 목록 |
| `DINGTALK_WEBHOOK_URL` | 플랫폼 간 / cron 전달에 사용할 고정 로봇 webhook URL |
| `DINGTALK_HOME_CHANNEL` | cron/알림 전달에 사용할 기본 대화 ID |
| `DINGTALK_HOME_CHANNEL_NAME` | DingTalk 홈 채널의 표시 이름 |
| `FEISHU_APP_ID` | [open.feishu.cn](https://open.feishu.cn/)의 Feishu/Lark 봇 App ID |
| `FEISHU_APP_SECRET` | Feishu/Lark 봇 App Secret |
| `FEISHU_DOMAIN` | `feishu`(중국) 또는 `lark`(국제). 기본값: `feishu` |
| `FEISHU_CONNECTION_MODE` | `websocket`(권장) 또는 `webhook`. 기본값: `websocket` |
| `FEISHU_ENCRYPT_KEY` | webhook 모드용 선택적 암호화 키 |
| `FEISHU_VERIFICATION_TOKEN` | webhook 모드용 선택적 검증 토큰 |
| `FEISHU_ALLOWED_USERS` | 봇에게 메시지를 보낼 수 있는 Feishu 사용자 ID를 쉼표로 구분한 목록 |
| `FEISHU_ALLOW_BOTS` | `none`(기본값) / `mentions` / `all` — 다른 봇의 수신 메시지를 허용합니다. [봇 간 메시징](../user-guide/messaging/feishu.md#bot-to-bot-messaging)을 참조하세요. |
| `FEISHU_REQUIRE_MENTION` | `true`(기본값) / `false` — 그룹 메시지에서 봇을 @멘션해야 하는지 여부입니다. `group_rules.<chat_id>.require_mention`으로 채팅별 재정의가 가능합니다. |
| `FEISHU_HOME_CHANNEL` | cron 전달 및 알림에 사용할 Feishu 채팅 ID |
| `FEISHU_HOME_CHANNEL_NAME` | Feishu 홈 채널의 표시 이름 |
| `FEISHU_ALLOW_ALL_USERS` | 모든 Feishu 사용자가 봇을 호출할 수 있도록 허용합니다(개발 전용). |
| `WECOM_BOT_ID` | 관리자 콘솔의 WeCom AI Bot ID |
| `WECOM_SECRET` | WeCom AI Bot 시크릿 |
| `WECOM_WEBSOCKET_URL` | 사용자 지정 WebSocket URL(기본값: `wss://openws.work.weixin.qq.com`) |
| `WECOM_ALLOWED_USERS` | 봇에게 메시지를 보낼 수 있는 WeCom 사용자 ID를 쉼표로 구분한 목록 |
| `WECOM_HOME_CHANNEL` | cron 전달 및 알림에 사용할 WeCom 채팅 ID |
| `WECOM_CALLBACK_CORP_ID` | callback 자체 구축 앱용 WeCom 기업 Corp ID |
| `WECOM_CALLBACK_CORP_SECRET` | 자체 구축 앱의 Corp secret |
| `WECOM_CALLBACK_AGENT_ID` | 자체 구축 앱의 Agent ID |
| `WECOM_CALLBACK_TOKEN` | Callback 검증 토큰 |
| `WECOM_CALLBACK_ENCODING_AES_KEY` | Callback 암호화용 AES 키 |
| `WECOM_CALLBACK_HOST` | Callback 서버 바인드 주소(기본값: `0.0.0.0`) |
| `WECOM_CALLBACK_PORT` | Callback 서버 포트(기본값: `8645`) |
| `WECOM_CALLBACK_ALLOWED_USERS` | 허용 목록용 사용자 ID를 쉼표로 구분한 목록 |
| `WECOM_CALLBACK_ALLOW_ALL_USERS` | 허용 목록 없이 모든 사용자를 허용하려면 `true`로 설정합니다. |
| `WEIXIN_ACCOUNT_ID` | iLink Bot API를 통한 QR 로그인으로 얻은 Weixin 계정 ID |
| `WEIXIN_TOKEN` | iLink Bot API를 통한 QR 로그인으로 얻은 Weixin 인증 토큰 |
| `WEIXIN_BASE_URL` | Weixin iLink Bot API 기본 URL 재정의(기본값: `https://ilinkai.weixin.qq.com`) |
| `WEIXIN_CDN_BASE_URL` | 미디어용 Weixin CDN 기본 URL 재정의(기본값: `https://novac2c.cdn.weixin.qq.com/c2c`) |
| `WEIXIN_DM_POLICY` | DM 정책: `open`, `allowlist`, `pairing`, `disabled`(기본값: `open`) |
| `WEIXIN_GROUP_POLICY` | 그룹 메시지 정책: `open`, `allowlist`, `disabled`(기본값: `disabled`) |
| `WEIXIN_ALLOWED_USERS` | 봇에게 DM을 보낼 수 있는 Weixin 사용자 ID를 쉼표로 구분한 목록 |
| `WEIXIN_GROUP_ALLOWED_USERS` | 봇과 상호작용할 수 있는 Weixin **그룹 채팅 ID**(멤버 사용자 ID가 아님)를 쉼표로 구분한 목록입니다. 변수 이름은 레거시이며 그룹 ID를 기대합니다. iLink가 실제로 그룹 이벤트를 전달하는 경우에만 적용됩니다. QR 로그인한 iLink 봇 ID(`...@im.bot`)는 일반 WeChat 그룹 메시지를 대개 받지 못합니다. |
| `WEIXIN_HOME_CHANNEL` | cron 전달 및 알림에 사용할 Weixin 채팅 ID |
| `WEIXIN_HOME_CHANNEL_NAME` | Weixin 홈 채널의 표시 이름 |
| `WEIXIN_ALLOW_ALL_USERS` | 허용 목록 없이 모든 Weixin 사용자를 허용합니다(`true`/`false`). |
| `BLUEBUBBLES_SERVER_URL` | BlueBubbles 서버 URL(예: `http://192.168.1.10:1234`) |
| `BLUEBUBBLES_PASSWORD` | BlueBubbles 서버 비밀번호 |
| `BLUEBUBBLES_WEBHOOK_HOST` | webhook 리스너 바인드 주소(기본값: `127.0.0.1`) |
| `BLUEBUBBLES_WEBHOOK_PORT` | webhook 리스너 포트(기본값: `8645`) |
| `BLUEBUBBLES_HOME_CHANNEL` | cron/알림 전달에 사용할 전화번호/이메일 |
| `BLUEBUBBLES_ALLOWED_USERS` | 권한이 부여된 사용자를 쉼표로 구분한 목록 |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | 모든 사용자를 허용합니다(`true`/`false`). |
| `QQ_APP_ID` | [q.qq.com](https://q.qq.com)의 QQ Bot App ID |
| `QQ_CLIENT_SECRET` | [q.qq.com](https://q.qq.com)의 QQ Bot App Secret |
| `QQ_STT_API_KEY` | 외부 STT 대체 제공업체용 API 키(선택 사항, QQ 기본 ASR이 텍스트를 반환하지 않을 때 사용) |
| `QQ_STT_BASE_URL` | 외부 STT 제공업체의 기본 URL(선택 사항) |
| `QQ_STT_MODEL` | 외부 STT 제공업체의 모델 이름(선택 사항) |
| `QQ_ALLOWED_USERS` | 봇에게 메시지를 보낼 수 있는 QQ 사용자 openID를 쉼표로 구분한 목록 |
| `QQ_GROUP_ALLOWED_USERS` | 그룹 @멘션 메시지 접근에 사용할 QQ 그룹 ID를 쉼표로 구분한 목록 |
| `QQ_ALLOW_ALL_USERS` | 모든 사용자를 허용합니다(`true`/`false`, `QQ_ALLOWED_USERS`보다 우선). |
| `QQBOT_HOME_CHANNEL` | cron 전달 및 알림에 사용할 QQ 사용자/그룹 openID |
| `QQBOT_HOME_CHANNEL_NAME` | QQ 홈 채널의 표시 이름 |
| `QQ_PORTAL_HOST` | QQ 포털 호스트를 재정의합니다(샌드박스 게이트웨이를 통해 라우팅하려면 `sandbox.q.qq.com`으로 설정, 기본값: `q.qq.com`). |
| `QQ_SANDBOX` | 개발 테스트용 QQ 샌드박스 모드를 활성화합니다(`true`/`false`). |
| `MATTERMOST_URL` | Mattermost 서버 URL(예: `https://mm.example.com`) |
| `MATTERMOST_TOKEN` | Mattermost용 봇 토큰 또는 개인 액세스 토큰 |
| `MATTERMOST_ALLOWED_USERS` | 봇에게 메시지를 보낼 수 있는 Mattermost 사용자 ID를 쉼표로 구분한 목록 |
| `MATTERMOST_ALLOW_ALL_USERS` | 모든 Mattermost 사용자가 봇을 호출할 수 있도록 허용합니다(개발 전용). |
| `MATTERMOST_ALLOWED_CHANNELS` | 설정하면 봇이 이 채널에서만 응답합니다(허용 목록). |
| `MATTERMOST_HOME_CHANNEL` | 능동적 메시지 전달(cron, 알림)에 사용할 채널 ID |
| `MATTERMOST_REQUIRE_MENTION` | 채널에서 `@mention`을 요구합니다(기본값: `true`). 모든 메시지에 응답하려면 `false`로 설정합니다. |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | 봇이 `@mention` 없이 응답하는 채널 ID를 쉼표로 구분한 목록 |
| `MATTERMOST_REPLY_MODE` | 답장 방식: `thread`(스레드 답장) 또는 `off`(일반 메시지, 기본값) |
| `MATRIX_HOMESERVER` | Matrix homeserver URL(예: `https://matrix.org`) |
| `MATRIX_ACCESS_TOKEN` | 봇 인증용 Matrix 액세스 토큰 |
| `MATRIX_USER_ID` | Matrix 사용자 ID(예: `@hermes:matrix.org`) — 비밀번호 로그인에 필요하며 액세스 토큰을 사용하면 선택 사항 |
| `MATRIX_PASSWORD` | Matrix 비밀번호(액세스 토큰의 대안) |
| `MATRIX_ALLOWED_USERS` | 봇에게 메시지를 보낼 수 있는 Matrix 사용자 ID를 쉼표로 구분한 목록(예: `@alice:matrix.org`) |
| `MATRIX_ALLOW_ALL_USERS` | 모든 Matrix 사용자가 봇을 호출할 수 있도록 허용합니다(개발 전용). |
| `MATRIX_HOME_CHANNEL` | cron/알림 전달에 사용할 기본 룸 ID |
| `MATRIX_HOME_CHANNEL_NAME` | Matrix 홈 룸의 표시 이름 |
| `MATRIX_ALLOWED_ROOMS` | 봇 응답을 트리거할 수 있는 Matrix 룸 ID를 쉼표로 구분한 목록 |
| `MATRIX_HOME_ROOM` | 능동적 메시지 전달에 사용할 룸 ID(예: `!abc123:matrix.org`) |
| `MATRIX_ENCRYPTION` | 종단 간 암호화를 활성화합니다(`true`/`false`, 기본값: `false`). |
| `MATRIX_E2EE_MODE` | Matrix E2EE 동작: `off`, `optional` 또는 `required`. 설정하면 `MATRIX_ENCRYPTION`을 재정의합니다. |
| `MATRIX_DEVICE_ID` | 재시작 후에도 E2EE가 유지되도록 하는 안정적인 Matrix 장치 ID(예: `HERMES_BOT`). 이 값이 없으면 시작할 때마다 E2EE 키가 교체되어 이전 룸의 복호화가 중단됩니다. |
| `MATRIX_REACTIONS` | 수신 메시지의 처리 수명 주기 이모지 반응을 활성화합니다(기본값: `true`). 비활성화하려면 `false`로 설정합니다. |
| `MATRIX_REQUIRE_MENTION` | 룸에서 `@mention`을 요구합니다(기본값: `true`). 모든 메시지에 응답하려면 `false`로 설정합니다. |
| `MATRIX_FREE_RESPONSE_ROOMS` | 봇이 `@mention` 없이 응답하는 룸 ID를 쉼표로 구분한 목록 |
| `MATRIX_IGNORE_USER_PATTERNS` | 무시할 Matrix 브리지/appservice ghost 사용자 ID의 정규식을 쉼표로 구분한 목록 |
| `MATRIX_PROCESS_NOTICES` | 수신 Matrix `m.notice` 이벤트를 처리합니다(기본값: `false`). |
| `MATRIX_SESSION_SCOPE` | 프로젝트 룸의 Matrix 세션 범위: `auto`, `room` 또는 `thread`(기본값: `auto`) |
| `MATRIX_TOOLS_ALLOW_REDACTION` | Matrix 메시지 삭제 도구 실행을 허용합니다(기본값: `false`). |
| `MATRIX_TOOLS_ALLOW_INVITES` | Matrix 초대 도구 실행을 허용합니다(기본값: `false`). |
| `MATRIX_TOOLS_ALLOW_ROOM_CREATE` | Matrix 룸 생성 도구 실행을 허용합니다(기본값: `false`). |
| `MATRIX_ALLOW_ROOM_MENTIONS` | 모든 룸 구성원에게 알리기 위해 발신 `@room` 멘션을 허용합니다(기본값: `false`). |
| `MATRIX_AUTO_THREAD` | 룸 메시지에 대한 스레드를 자동으로 생성합니다(기본값: `true`). |
| `MATRIX_DM_AUTO_THREAD` | Matrix DM 메시지에 대한 스레드를 자동으로 생성합니다(기본값: `false`). |
| `MATRIX_DM_MENTION_THREADS` | DM에서 봇이 `@mentioned`되면 스레드를 생성합니다(기본값: `false`). |
| `MATRIX_APPROVAL_REQUIRE_SENDER` | 원래 요청자를 알고 있는 경우 승인/모델 선택기 반응이 원래 요청자에게서 오도록 요구합니다(기본값: `true`). |
| `MATRIX_APPROVAL_TIMEOUT_SECONDS` | Matrix 반응 승인/모델 선택기 프롬프트의 제한 시간(기본값: `300`) |
| `MATRIX_ALLOW_PUBLIC_ROOMS` | Matrix 룸 생성 도구가 공개 룸을 생성하도록 허용합니다(기본값: `false`). |
| `MATRIX_MAX_MEDIA_BYTES` | Matrix 미디어 업로드/다운로드의 최대 크기(바이트, 기본값: `104857600`) |
| `MATRIX_RECOVERY_KEY` | 장치 키 교체 후 교차 서명 검증에 사용하는 복구 키입니다. 교차 서명이 활성화된 E2EE 설정에 권장됩니다. |
| `MATRIX_RECOVERY_KEY_OUTPUT_FILE` | 생성된 Matrix 복구 키를 저장할 선택적 일회성 경로입니다. 권한 `0600`으로 생성되며 절대 덮어쓰지 않습니다. |
| `HASS_TOKEN` | Home Assistant Long-Lived Access Token(HA 플랫폼 및 도구 활성화) |
| `HASS_URL` | Home Assistant URL(기본값: `http://homeassistant.local:8123`) |
| `WEBHOOK_ENABLED` | webhook 플랫폼 어댑터를 활성화합니다(`true`/`false`). |
| `WEBHOOK_PORT` | webhook 수신용 HTTP 서버 포트(기본값: `8644`) |
| `WEBHOOK_SECRET` | webhook 서명 검증용 전역 HMAC 시크릿(라우트에 자체 시크릿이 지정되지 않은 경우 대체값으로 사용) |
| `API_SERVER_ENABLED` | OpenAI 호환 API 서버를 활성화합니다(`true`/`false`). 다른 플랫폼과 함께 실행됩니다. |
| `API_SERVER_KEY` | API 서버 인증용 Bearer 토큰입니다. API 서버가 활성화된 경우 항상 필요합니다. |
| `API_SERVER_CORS_ORIGINS` | API 서버를 직접 호출할 수 있는 브라우저 origin을 쉼표로 구분한 목록(예: `http://localhost:3000,http://127.0.0.1:3000`). 기본값: 비활성화. |
| `API_SERVER_PORT` | API 서버 포트(기본값: `8642`) |
| `API_SERVER_HOST` | API 서버의 호스트/바인드 주소(기본값: `127.0.0.1`). 루프백에서도 `API_SERVER_KEY`는 필요합니다. 브라우저 접근에는 범위를 좁힌 `API_SERVER_CORS_ORIGINS` 허용 목록을 사용하세요. |
| `API_SERVER_MODEL_NAME` | `/v1/models`에 광고할 모델 이름입니다. 기본값은 프로필 이름이며, 기본 프로필에서는 `hermes-agent`입니다. Open WebUI 같은 프런트엔드가 연결별로 서로 다른 모델 이름을 사용해야 하는 다중 사용자 설정에 유용합니다. |
| `GATEWAY_PROXY_URL` | 메시지를 전달할 원격 Hermes API 서버의 URL입니다([프록시 모드](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)). 설정하면 게이트웨이는 플랫폼 I/O만 처리하고 모든 에이전트 작업을 원격 서버에 위임합니다. `config.yaml`의 `gateway.proxy_url`로도 설정할 수 있습니다. |
| `GATEWAY_PROXY_KEY` | 프록시 모드에서 원격 API 서버 인증에 사용하는 Bearer 토큰입니다. 원격 호스트의 `API_SERVER_KEY`와 일치해야 합니다. |
| `MESSAGING_CWD` | 게이트웨이 작업 디렉터리의 호환성 유지용 레거시 대체 설정입니다. `config.yaml`의 `terminal.cwd`를 사용하세요. |
| `GATEWAY_ALLOWED_USERS` | 모든 플랫폼에서 허용되는 사용자 ID를 쉼표로 구분한 목록 |
| `GATEWAY_ALLOW_ALL_USERS` | 허용 목록 없이 모든 사용자를 허용합니다(`true`/`false`, 기본값: `false`) |
### 웹 대시보드 및 Hermes Desktop

[웹 대시보드](/user-guide/features/web-dashboard) 인증과 [Hermes Desktop을 원격 백엔드에 연결](/user-guide/features/web-dashboard#connecting-hermes-desktop-to-a-remote-backend)할 때 사용합니다. 시크릿 전용 규칙에 따라 자격 증명은 `~/.hermes/.env`에 저장하고, OAuth `client_id`는 `config.yaml`의 `dashboard.oauth` 아래에 설정하는 편이 좋습니다(환경 변수가 설정되면 환경 변수가 우선합니다).

세 가지 대시보드 인증 제공자가 기본으로 제공됩니다. 원격 Hermes Desktop 연결이나 인터넷에 공개되는 대시보드에는 **OAuth (Nous Portal)** 제공자를 권장합니다 — `HERMES_DASHBOARD_OAUTH_CLIENT_ID`를 설정하세요(`hermes dashboard register`로 발급). 번들로 제공되는 **사용자 이름/비밀번호** 제공자(`HERMES_DASHBOARD_BASIC_AUTH_*`)는 신뢰할 수 있는 LAN 또는 VPN 뒤의 백엔드에서 가장 빠른 선택이지만, 공용 인터넷에 직접 노출하기에는 적합하지 않습니다. 자체 ID 제공자에 대해 인증하려면 **셀프 호스팅 OIDC** 제공자(`HERMES_DASHBOARD_OIDC_*`)를 사용하세요. 어떤 방식을 선택하든 루프백이 아닌 바인드(`hermes dashboard --host 0.0.0.0`)는 인증 게이트를 활성화합니다. 자세한 내용은 [웹 대시보드 → 인증](/user-guide/features/web-dashboard#authentication-gated-mode)을 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | 번들 사용자 이름/비밀번호 대시보드 인증 제공자(`plugins/dashboard_auth/basic`)의 사용자 이름입니다. 비밀번호와 함께 설정하면 제공자를 활성화합니다. `dashboard.basic_auth.username`을 재정의합니다. |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | 기본 제공자의 평문 비밀번호입니다(로드 시 메모리에서 해시됨). 환경 변수로 비밀번호를 교체할 수 있도록 설정 파일의 `password_hash`보다 우선합니다. `dashboard.basic_auth.password`를 재정의합니다. |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | 기본 제공자의 scrypt 비밀번호 해시입니다(평문을 저장하지 않으므로 권장). `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`로 계산합니다. `dashboard.basic_auth.password_hash`를 재정의합니다. |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | 기본 제공자의 상태 비저장 세션 토큰에 서명하는 HMAC 키(32바이트 이상, base64/hex/raw)입니다. 세션이 재시작 후에도 유지되거나 여러 워커에 걸쳐 공유되도록 명시적으로 설정하세요. 비워 두면 프로세스마다 무작위 값이 사용되어 재시작할 때마다 로그아웃됩니다. `dashboard.basic_auth.secret`을 재정의합니다. |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | 기본 제공자의 액세스 토큰 수명입니다(기본값 12시간). `dashboard.basic_auth.session_ttl_seconds`를 재정의합니다. |
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | 게이트된/공개 대시보드용 OAuth 클라이언트 ID(`agent:{instance_id}`)입니다. Nous 제공자(`plugins/dashboard_auth/nous`)를 활성화합니다. `dashboard.oauth.client_id`를 재정의합니다. `hermes dashboard register`로 발급합니다. |
| `HERMES_DASHBOARD_PUBLIC_URL` | 대시보드에 접속하는 완전한 공개 URL입니다. 리버스 프록시 뒤에서 OAuth 콜백을 구성할 때 사용합니다. `dashboard.public_url`을 재정의합니다. |
| `HERMES_DASHBOARD_OIDC_ISSUER` | 번들 셀프 호스팅 OIDC 제공자(`plugins/dashboard_auth/self_hosted`)의 OIDC 발급자 URL입니다. 이를 설정해야 제공자가 활성화됩니다. `dashboard.oauth.self_hosted.issuer`를 재정의합니다. |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | 셀프 호스팅 OIDC 제공자용 공개 OIDC 클라이언트 ID(authorization-code + PKCE)입니다. 이를 설정해야 제공자가 활성화됩니다. `dashboard.oauth.self_hosted.client_id`를 재정의합니다. |
| `HERMES_DASHBOARD_OIDC_SCOPES` | 셀프 호스팅 OIDC 제공자에 요청할 OIDC 스코프입니다(기본값 `openid profile email`). `dashboard.oauth.self_hosted.scopes`를 재정의합니다. |
| `HERMES_DESKTOP_REMOTE_URL` | (Desktop 측) 원격 백엔드의 기본 URL입니다(예: `http://host:9119`). 설정하면 앱 내 Gateway URL을 재정의합니다. 그래도 Gateway 설정 패널에서 로그인해야 합니다(백엔드가 제공하는 방식에 따라 OAuth 리디렉션 또는 사용자 이름/비밀번호). |
| `HERMES_DESKTOP_HERMES` | Desktop 백엔드 명령 재정의입니다. 패키저/Nix 또는 문제 해결 시 Electron이 백엔드를 탐색한 후 특정 `hermes` 실행 파일을 가리키는 데 사용합니다. |
| `HERMES_DESKTOP_HERMES_ROOT` | `hermes desktop --hermes-root`에서 사용하는 Desktop 소스 체크아웃 재정의입니다. 패키징된 최초 실행 설치 또는 `PATH`에 있는 기존 `hermes`보다 먼저 확인됩니다. |
| `HERMES_DESKTOP_IGNORE_EXISTING` | Desktop이 백엔드 확인 중 `PATH`에 있는 기존 `hermes`를 무시하도록 하려면 `1`로 설정합니다. `hermes desktop --ignore-existing`와 같습니다. |
| `HERMES_DESKTOP_CWD` | Desktop 채팅 세션의 초기 프로젝트 디렉터리입니다. `hermes desktop --cwd`에서 설정합니다. |
| `HERMES_DESKTOP_PYTHON` | 백엔드에 사용할 Python 인터프리터의 절대 경로입니다. 소스 체크아웃에서 Electron이 자동으로 해석하는 경로보다 먼저 확인됩니다. 공유 venv를 재사용하는 worktree 개발 도우미에서 사용합니다([Worktree에서 TUI 및 Desktop 실행](../developer-guide/worktree-ui-dev.md) 참조). |
| `HERMES_DESKTOP_DEV_SERVER` | 패키징된 번들 대신 Electron 셸이 로드할 Vite 개발 서버 URL입니다(예: `http://127.0.0.1:5174`). `npm run dev`에서 자동으로 설정되며 앱을 개발할 때만 관련됩니다. |
| `HERMES_DESKTOP_CDP_PORT` | DOM/CSS 검사 도구를 위해 렌더러가 `127.0.0.1`에서 노출하는 Chrome DevTools Protocol 포트를 재정의합니다(기본값 `9222`). 개발 서버 실행(`npm run dev`, `hgui`)에서는 자동으로 열리지만 패키징된 앱에서는 절대 열리지 않으며, 이 값으로도 변경되지 않습니다. 개발 실행에서 비활성화하려면 `off`로 설정합니다. 이 포트에 접근할 수 있는 모든 항목은 렌더러에서 코드를 실행할 수 있습니다. |

### Microsoft Graph (Teams Meetings)

곧 제공될 Teams 회의 요약 파이프라인에서 사용하는 Microsoft Graph REST 클라이언트의 앱 전용 자격 증명입니다. Azure 포털 사용 방법과 필요한 정확한 API 권한은 [Microsoft Graph 애플리케이션 등록](/guides/microsoft-graph-app-registration)을 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `MSGRAPH_TENANT_ID` | Graph 앱 등록에 사용하는 Azure AD 테넌트 ID(디렉터리 GUID)입니다. |
| `MSGRAPH_CLIENT_ID` | Azure 앱 등록의 애플리케이션(클라이언트) ID입니다. |
| `MSGRAPH_CLIENT_SECRET` | 앱 등록의 클라이언트 시크릿 값입니다. `chmod 600` 권한으로 `~/.hermes/.env`에 저장하고 Azure 포털에서 주기적으로 교체하세요. |
| `MSGRAPH_SCOPE` | 클라이언트 자격 증명 토큰 요청에 사용하는 OAuth2 스코프입니다(기본값: `https://graph.microsoft.com/.default`). |
| `MSGRAPH_AUTHORITY_URL` | Microsoft ID 플랫폼 권한 URL입니다(기본값: `https://login.microsoftonline.com`). 국가별/소버린 클라우드에서만 재정의하세요(예: GCC High의 경우 `https://login.microsoftonline.us`). |

### Microsoft Graph Webhook Listener

Graph 이벤트(Teams 회의, 캘린더, 채팅 등)를 위한 인바운드 변경 알림 리스너입니다. 설정 및 보안 강화 방법은 [Microsoft Graph Webhook Listener](/user-guide/messaging/msgraph-webhook)를 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `MSGRAPH_WEBHOOK_ENABLED` | `msgraph_webhook` 게이트웨이 플랫폼을 활성화합니다(`true`/`1`/`yes`). |
| `MSGRAPH_WEBHOOK_PORT` | 리스너가 바인드할 포트입니다(기본값: `8646`). |
| `MSGRAPH_WEBHOOK_CLIENT_STATE` | Graph가 모든 알림에 되풀이해 포함하는 공유 시크릿입니다. `hmac.compare_digest`로 비교합니다. `openssl rand -hex 32`로 생성하세요. |
| `MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES` | 허용할 Graph 리소스 경로/패턴의 쉼표로 구분된 허용 목록입니다(예: `communications/onlineMeetings,chats/*/messages`). 뒤의 `*`는 접두사 일치입니다. 비워 두면 모두 허용합니다. |
| `MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS` | 리스너에 POST할 수 있는 CIDR 범위의 쉼표로 구분된 목록입니다(예: `52.96.0.0/14,52.104.0.0/14`). 비워 두면 모두 허용합니다(기본값). 프로덕션에서는 Microsoft Graph가 공개한 송신 범위로 제한하세요. |

### Teams Meeting Summary Delivery

[`teams_pipeline` 플러그인](/user-guide/messaging/msgraph-webhook)이 활성화된 경우에만 사용합니다. 설정은 `config.yaml`의 `platforms.teams.extra`에서도 지정할 수 있으며 두 곳 모두 설정하면 환경 변수가 우선합니다. 자세한 내용은 [Microsoft Teams → 회의 요약 전달](/user-guide/messaging/teams#meeting-summary-delivery-teams-meeting-pipeline)을 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `TEAMS_DELIVERY_MODE` | `graph` 또는 `incoming_webhook`입니다. |
| `TEAMS_INCOMING_WEBHOOK_URL` | Teams에서 생성한 웹훅 URL입니다. `TEAMS_DELIVERY_MODE=incoming_webhook`일 때 필요합니다. |
| `TEAMS_GRAPH_ACCESS_TOKEN` | Graph 전달에 사용할 사전 발급 위임 액세스 토큰입니다. 설정하지 않는 경우 작성자는 `MSGRAPH_*` 앱 자격 증명으로 대체하므로 거의 필요하지 않습니다. |
| `TEAMS_TEAM_ID` | 채널 전달 대상 Team ID입니다(`graph` 모드). |
| `TEAMS_CHANNEL_ID` | 대상 채널 ID입니다(`TEAMS_TEAM_ID`와 함께 사용). |
| `TEAMS_CHAT_ID` | 대상 1:1 또는 그룹 채팅 ID입니다(`graph` 모드에서 team+channel의 대안). |

### LINE Messaging API

번들 LINE 플랫폼 플러그인(`plugins/platforms/line/`)에서 사용합니다. 전체 설정 방법은 [Messaging Gateway → LINE](/user-guide/messaging/line)을 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Developers Console의 Messaging API 탭에서 발급하는 장기 채널 액세스 토큰입니다. 필수입니다. |
| `LINE_CHANNEL_SECRET` | 채널 시크릿(Basic settings 탭)으로, HMAC-SHA256 웹훅 서명 확인에 사용합니다. 필수입니다. |
| `LINE_HOST` | 웹훅 바인드 호스트입니다(기본값: `0.0.0.0`). |
| `LINE_PORT` | 웹훅 바인드 포트입니다(기본값: `8646`). |
| `LINE_PUBLIC_URL` | 공개 HTTPS 기본 URL입니다(예: `https://my-tunnel.example.com`). 이미지/오디오/동영상 전송에 필요합니다 — LINE은 HTTPS로 접근 가능한 URL만 허용합니다. |
| `LINE_ALLOWED_USERS` | 봇에 DM을 보낼 수 있는 사용자 ID의 쉼표로 구분된 목록입니다(`U` 접두사). |
| `LINE_ALLOWED_GROUPS` | 봇이 응답할 그룹 ID의 쉼표로 구분된 목록입니다(`C` 접두사). |
| `LINE_ALLOWED_ROOMS` | 봇이 응답할 룸 ID의 쉼표로 구분된 목록입니다(`R` 접두사). |
| `LINE_ALLOW_ALL_USERS` | 개발 전용 우회 수단으로, 모든 발신자를 허용합니다. 기본값: `false`. |
| `LINE_HOME_CHANNEL` | `deliver: line`으로 설정한 cron 작업의 기본 전달 대상입니다. |
| `LINE_SLOW_RESPONSE_THRESHOLD` | 느린 LLM Template Buttons 포스트백이 실행되기까지의 시간(초)입니다(기본값: `45`). `0`으로 설정하면 비활성화되어 항상 Push 대체 경로를 사용합니다. |
| `LINE_PENDING_TEXT` | 포스트백 버튼 옆에 표시할 버블 텍스트입니다. |
| `LINE_BUTTON_LABEL` | 포스트백 버튼 레이블입니다(기본값: `Get answer`). |
| `LINE_DELIVERED_TEXT` | 이미 전달된 포스트백을 다시 탭했을 때의 응답입니다(기본값: `Already replied ✅`). |
| `LINE_INTERRUPTED_TEXT` | `/stop`으로 고아가 된 포스트백 버튼을 탭했을 때의 응답입니다(기본값: `Run was interrupted before completion.`). |

### ntfy (push notifications)

[ntfy](https://ntfy.sh/)는 HTTP 기반의 경량 푸시 알림 서비스입니다. [ntfy 모바일 앱](https://ntfy.sh/docs/subscribe/phone/)에서 토픽을 구독하고, 해당 토픽에 메시지를 게시해 에이전트와 대화하세요.

| 변수 | 설명 |
|----------|-------------|
| `NTFY_TOPIC` | 구독할 토픽(수신 메시지)입니다. 필수입니다. |
| `NTFY_SERVER_URL` | 서버 URL입니다(기본값: `https://ntfy.sh`). 개인정보 보호를 위해 셀프 호스팅 ntfy를 지정할 수 있습니다. |
| `NTFY_TOKEN` | 선택적 인증 토큰입니다. Bearer 토큰(예: `tk_xyz`) 또는 Basic 인증용 `user:pass`입니다. |
| `NTFY_PUBLISH_TOPIC` | 발신 답변에 사용할 토픽입니다(기본값은 `NTFY_TOPIC`). |
| `NTFY_MARKDOWN` | 답변을 `X-Markdown: true` 헤더와 함께 보내려면 `true`로 설정합니다. 기본값: `false`. |
| `NTFY_ALLOWED_USERS` | 허용 목록입니다(사용자 ID로 취급하며, ntfy에서는 토픽 이름입니다). 일반적으로 `NTFY_TOPIC`과 같은 값으로 설정합니다. |
| `NTFY_ALLOW_ALL_USERS` | 개발 전용 우회 수단으로, 접근 제어된 비공개 토픽에서만 안전합니다. 기본값: `false`. |
| `NTFY_HOME_CHANNEL` | `deliver: ntfy`로 설정한 cron 작업의 기본 전달 대상입니다. |
| `NTFY_HOME_CHANNEL_NAME` | 홈 채널의 사람이 읽을 수 있는 레이블입니다(기본값은 토픽 이름). |

신뢰할 수 없는 토픽으로 배포하기 전에 [ntfy 메시징 가이드](/user-guide/messaging/ntfy), 특히 **identity model** 섹션을 참조하세요.

### IRC

Hermes를 IRC 서버에 연결합니다. 외부 종속성이 필요하지 않습니다. [IRC 메시징 가이드](/user-guide/messaging/irc)를 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `IRC_SERVER` | IRC 서버 호스트 이름입니다(예: `irc.libera.chat`). 필수입니다. |
| `IRC_CHANNEL` | 참여할 채널입니다(예: `#hermes`). 여러 채널은 쉼표로 구분합니다. 필수입니다. |
| `IRC_NICKNAME` | 봇 닉네임입니다(기본값: `hermes-bot`). 필수입니다. |
| `IRC_PORT` | 서버 포트입니다(TLS 사용 시 기본값 `6697`, 미사용 시 `6667`). |
| `IRC_USE_TLS` | TLS 사용 여부입니다(`true`/`false`; 포트 6697에서는 기본값 `true`). |
| `IRC_SERVER_PASSWORD` | `PASS` 명령에 사용할 서버 비밀번호입니다(선택 사항). |
| `IRC_NICKSERV_PASSWORD` | 연결 시 자동 IDENTIFY에 사용할 NickServ 비밀번호입니다(선택 사항). |
| `IRC_ALLOWED_USERS` | 봇과 대화할 수 있는 닉네임의 쉼표로 구분된 목록입니다. |
| `IRC_ALLOW_ALL_USERS` | 채널의 모든 사용자가 봇과 대화하도록 허용합니다(개발 전용). |
| `IRC_HOME_CHANNEL` | cron/알림을 전달할 채널입니다(기본값은 `IRC_CHANNEL`). |
### SimpleX

로컬 `simplex-chat` 데몬을 통해 Hermes를 [SimpleX Chat](https://simplex.chat/) 네트워크에 연결합니다. [SimpleX 메시징 가이드](/user-guide/messaging/simplex)를 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `SIMPLEX_WS_URL` | simplex-chat 데몬의 WebSocket URL(예: `ws://127.0.0.1:5225`). |
| `SIMPLEX_ALLOWED_USERS` | 봇과 대화할 수 있도록 허용된 쉼표로 구분된 SimpleX 연락처 ID. |
| `SIMPLEX_ALLOW_ALL_USERS` | 모든 연락처가 봇과 대화하도록 허용합니다(개발 전용 — 허용 목록을 비활성화). |
| `SIMPLEX_AUTO_ACCEPT` | 수신 연락처 요청을 자동으로 수락합니다(기본값: `true`). |
| `SIMPLEX_GROUP_ALLOWED` | 봇이 참여해야 하는 쉼표로 구분된 SimpleX 그룹 ID 또는 모든 그룹을 허용하는 `*`. 생략하면 그룹 메시지를 완전히 무시합니다(더 안전한 기본값 — 그룹의 봇은 그렇지 않으면 모든 구성원의 트래픽을 처리합니다). |
| `SIMPLEX_HOME_CHANNEL` | cron / 알림 전달을 위한 기본 연락처/그룹 ID. |
| `SIMPLEX_HOME_CHANNEL_NAME` | 홈 채널의 사람이 읽을 수 있는 레이블(ID를 기본값으로 사용). |

### Photon

Node 사이드카를 통해 Hermes를 [Photon](https://photon.codes/) / Spectrum(iMessage 및 기타 Spectrum 플랫폼)에 연결합니다. [Photon 메시징 가이드](/user-guide/messaging/photon)를 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `PHOTON_PROJECT_ID` | Spectrum 프로젝트 ID(프로젝트의 `spectrumProjectId`; `hermes photon setup`이 설정). |
| `PHOTON_PROJECT_SECRET` | Spectrum 프로젝트 ID와 쌍을 이루는 프로젝트 시크릿(`hermes photon setup`이 설정). |
| `PHOTON_ALLOWED_USERS` | 봇과 대화할 수 있도록 허용된 쉼표로 구분된 E.164 전화번호. |
| `PHOTON_ALLOW_ALL_USERS` | 모든 발신자가 봇을 트리거하도록 허용합니다(개발 전용 — 허용 목록을 비활성화). |
| `PHOTON_REQUIRE_MENTION` | 멘션 깨우기 단어와 일치하지 않는 그룹 채팅 메시지를 무시합니다(`true`/`false`, 기본값 `false`). |
| `PHOTON_MENTION_PATTERNS` | 그룹 채팅의 멘션 깨우기 단어 정규식(JSON 목록 또는 쉼표/줄바꿈으로 구분; 기본값은 Hermes 깨우기 단어). |
| `PHOTON_HOME_CHANNEL` | cron / 알림 전달을 위한 기본 Photon 대상: Spectrum 스페이스 ID, DM GUID 또는 접두사가 없는 E.164 전화번호. |
| `PHOTON_HOME_CHANNEL_NAME` | 홈 채널의 사람이 읽을 수 있는 레이블. |
| `PHOTON_MARKDOWN` | 에이전트 응답을 마크다운으로 전송합니다 — iMessage는 이를 네이티브로 렌더링하고, 다른 Spectrum 플랫폼에서는 일반 텍스트로 저하됩니다(`true`/`false`, 기본값 `true`). |
| `PHOTON_REACTIONS` | 메시지를 처리하는 동안 메시지에 Tapback 👀/👍/👎을 표시하고, 봇 메시지에 대한 Tapback을 에이전트로 라우팅합니다(`true`/`false`, 기본값 `false`). |
| `PHOTON_TELEMETRY` | 사이드카에서 Spectrum SDK 텔레메트리를 활성화합니다(`true`/`false`, 기본값 `false`; `hermes photon telemetry on|off`로 전환). |
| `PHOTON_SIDECAR_PORT` | Node 사이드카 제어 + 인바운드 채널을 위한 루프백 포트(기본값 `8789`). |
| `PHOTON_SIDECAR_AUTOSTART` | 연결 시 Node 사이드카를 생성합니다(`true`/`false`, 기본값 `true`). |
| `PHOTON_NODE_BIN` | node 바이너리 경로(기본값: `shutil.which('node')`). |
| `PHOTON_DASHBOARD_HOST` | Photon Dashboard API 호스트(기본값 `https://app.photon.codes`). |
| `PHOTON_SPECTRUM_HOST` | Photon Spectrum API 호스트(기본값 `https://spectrum.photon.codes`). |

### Buzz (Nostr 커뮤니티)

| 변수 | 설명 |
|----------|-------------|
| `BUZZ_RELAY_URL` | Buzz 커뮤니티 릴레이의 기본 URL(예: `https://mycommunity.communities.buzz.xyz`) |
| `BUZZ_PRIVATE_KEY` | 에이전트의 Buzz ID를 위한 Nostr 개인 키(nsec 또는 hex) — 유일한 Buzz 시크릿 |
| `BUZZ_CREDENTIALS_FILE` | nsec를 포함하는 JSON 자격 증명 파일(`BUZZ_PRIVATE_KEY`가 설정되지 않은 경우의 대체 수단) |
| `BUZZ_CHANNELS` | 감시할 쉼표로 구분된 채널 UUID(기본값: 가입한 모든 채널) |
| `BUZZ_HOME_CHANNEL` | cron / 알림 전달을 위한 채널 UUID(감시하는 첫 번째 채널을 기본값으로 사용) |
| `BUZZ_ALLOWED_USERS` | 에이전트와 대화할 수 있도록 허용된 쉼표로 구분된 npub 또는 hex 공개 키 |
| `BUZZ_ALLOW_ALL_USERS` | 모든 커뮤니티 구성원이 에이전트와 대화하도록 허용합니다(`true`/`false`) |
| `BUZZ_TRANSPORT` | 인바운드 전송: `auto`(WebSocket 및 폴백 poll, 기본값), `websocket` 또는 `poll` |
| `BUZZ_POLL_INTERVAL` | 인바운드 poll sweep 사이의 초(기본값: `4`) |
| `BUZZ_AUTH_TAG` | NIP-42 WebSocket 인증을 위한 선택적 NIP-OA 소유자 증명 인증 태그 JSON |
| `BUZZ_CLI_PATH` | buzz CLI 바이너리 경로(기본값: PATH의 `buzz`, 이후 `~/bin/buzz`) |

### Microsoft Teams (어댑터)

위의 [Microsoft Graph (Teams Meetings)](#microsoft-graph-teams-meetings) 통합과는 별개인 Microsoft Teams 플랫폼 어댑터(Bot Framework / Azure AD)입니다. [Teams 메시징 가이드](/user-guide/messaging/teams)를 참조하세요.

| 변수 | 설명 |
|----------|-------------|
| `TEAMS_CLIENT_ID` | Azure AD 애플리케이션(Bot Framework) 클라이언트 ID. |
| `TEAMS_CLIENT_SECRET` | Azure AD 애플리케이션 클라이언트 시크릿. |
| `TEAMS_TENANT_ID` | 봇 애플리케이션을 호스팅하는 Azure AD 테넌트 ID. |
| `TEAMS_HOST` | 웹훅 바인드 호스트(기본값: 설정되지 않음 → 듀얼 스택, 모든 인터페이스 IPv4+IPv6). |
| `TEAMS_PORT` | 웹훅 수신 포트(Bot Framework 기본값: `3978`). |
| `TEAMS_ALLOWED_USERS` | 봇과 대화할 수 있도록 허용된 쉼표로 구분된 Teams 사용자 ID / UPN. |
| `TEAMS_ALLOW_ALL_USERS` | 모든 Teams 사용자가 봇을 트리거하도록 허용합니다(개발 전용). |
| `TEAMS_HOME_CHANNEL` | cron / 알림 전달을 위한 기본 채팅/채널 ID. |
| `TEAMS_HOME_CHANNEL_NAME` | Teams 홈 채널의 표시 이름. |

### Raft

| 변수 | 설명 |
|----------|-------------|
| `RAFT_PROFILE` | Raft 에이전트 프로필 슬러그 — 설정하면 어댑터를 자동으로 활성화합니다. |

### 고급 메시징 튜닝

아웃바운드 메시지 배처의 스로틀링을 위한 고급 플랫폼별 설정입니다. 대부분의 사용자는 이를 조정할 필요가 없습니다. 기본값은 각 플랫폼의 속도 제한을 준수하면서도 답답할 정도로 느려지지 않도록 설정되어 있습니다.

| 변수 | 설명 |
|----------|-------------|
| `HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS` | 대기 중인 Telegram 텍스트 청크를 플러시하기 전의 유예 시간(기본값: `0.6`). |
| `HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS` | 하나의 Telegram 메시지가 길이 제한을 초과할 때 분할된 청크 사이의 지연(기본값: `2.0`). |
| `HERMES_SIMPLEX_TEXT_BATCH_DELAY` | 연속으로 빠르게 수신된 텍스트 메시지를 하나의 MessageEvent로 연결하는 데 사용되는 정숙 기간(초)(기본값: `0.8`) — Telegram의 텍스트 배칭과 동일한 패턴입니다. |
| `HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS` | 대기 중인 Telegram 미디어를 플러시하기 전의 유예 시간(기본값: `0.6`). |
| `HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS` | 에이전트가 완료된 후 후속 메시지를 보내기 전의 지연 시간. 마지막 스트림 청크와 경쟁하는 것을 방지합니다. |
| `HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT` / `_READ_TIMEOUT` / `_WRITE_TIMEOUT` / `_POOL_TIMEOUT` | 기본 `python-telegram-bot` HTTP 타임아웃(초)을 재정의합니다. |
| `HERMES_TELEGRAM_INIT_TIMEOUT` | 게이트웨이 시작 중 Telegram `initialize()` 연결 체인의 시도별 상한(초). 연결할 수 없는 fallback-IP 체인이 시작을 무기한 차단하지 않도록 하며, 기본값은 `30`입니다. |
| `HERMES_TELEGRAM_HTTP_POOL_SIZE` | Telegram API에 대한 최대 동시 HTTP 연결 수. |
| `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS` | DNS가 실패할 때 사용되는 하드코딩된 Cloudflare 대체 IP를 비활성화합니다(`true`/`false`). |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | 대기 중인 Discord 텍스트 청크를 플러시하기 전의 유예 시간(기본값: `0.6`). |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Discord 메시지가 길이 제한을 초과할 때 분할된 청크 사이의 지연(기본값: `2.0`). |
| `HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS` | `discord.websocket_liveness_interval_seconds`에 대한 호환성/수동 재정의. 활성 Discord Gateway WebSocket을 샘플링하는 간격(기본값: `15`; 비활성화하려면 `0`으로 설정). `config.yaml` 키를 사용하는 것이 좋습니다. |
| `HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD` | `discord.websocket_liveness_failure_threshold`에 대한 호환성/수동 재정의. 강제 재연결을 수행하기 전 연속으로 비정상인 WebSocket 샘플 수(기본값: `2`). `config.yaml` 키를 사용하는 것이 좋습니다. |
| `HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | Telegram 배치 설정에 해당하는 Matrix 설정. |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` / `_MAX_CHARS` / `_MAX_MESSAGES` | Feishu 배처 튜닝 — 지연, 분할 지연, 메시지당 최대 문자 수, 배치당 최대 메시지 수. |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | Feishu 미디어 플러시 지연. |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | Feishu 웹훅 중복 제거 캐시의 크기(기본값: `1024`). |
| `HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | WeCom 배처 튜닝. |
| `HERMES_VISION_DOWNLOAD_TIMEOUT` | 비전 모델에 전달하기 전에 이미지를 다운로드하는 타임아웃(초)(기본값: `30`). |
| `HERMES_VISION_MAX_CONCURRENCY` | 전체 프로세스에서 동시에 수행되는 이미지 **인코딩/크기 조정** 버스트의 최대 수(`auxiliary.vision.max_concurrency`에 대한 재정의; 기본값: 호스트 CPU 코어 수, 상한 없음). CPU를 사용하는 인코딩 단계만 제한하므로 비디오 프레임 fan-out이 모든 코어를 점유해 이벤트 루프를 고갈시키는 것을 방지합니다 — LLM 호출은 완전히 동시에 수행됩니다. `1` 미만의 값은 무시됩니다. |
| `HERMES_RESTART_DRAIN_TIMEOUT` | 게이트웨이: 재시작을 강제하기 전에 `/restart`에서 활성 실행이 종료되기를 기다리는 초(기본값: `900`). |
| `HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT` | 게이트웨이 시작 및 재연결 중 플랫폼별 연결 타임아웃(초; `0`/음수이면 무기한 대기). 연결 시도와 Discord 어댑터의 준비 대기 모두에 적용되므로, 동기화할 슬래시 명령이 많은 계정이 시작 중에 종료되지 않습니다. `config.yaml`의 `gateway.platform_connect_timeout`에서 전달됩니다(기본값 `30`). 이 환경 변수는 수동 재정의이며 명시적으로 설정하면 우선합니다. |
| `HERMES_GATEWAY_BUSY_INPUT_MODE` | 게이트웨이의 기본 사용 중 입력 동작: `queue`, `steer` 또는 `interrupt`. 채팅별로 `/busy`를 사용해 재정의할 수 있습니다. |
| `HERMES_GATEWAY_BUSY_ACK_ENABLED` | 사용자가 에이전트가 사용 중일 때 입력을 보낼 경우 게이트웨이가 확인 메시지(⚡/⏳/⏩)를 보내는지 여부(기본값: `true`). 이러한 메시지를 완전히 억제하려면 `false`로 설정합니다 — 입력은 여전히 대기/조정/중단되며 채팅 응답만 음소거됩니다. `config.yaml`의 `display.busy_ack_enabled`에서 전달됩니다. |
| `HERMES_GATEWAY_NO_SUPERVISE` | s6-overlay Docker 이미지 내부에서 `hermes gateway run` 실행 시 자동 감독을 사용하지 않고 pre-s6 포그라운드 의미 체계를 사용합니다(자동 재시작 없음, 게이트웨이가 컨테이너의 주 프로세스). Truthy 값: `1`, `true`, `yes`. `--no-supervise` CLI 플래그와 동일합니다. s6 이미지 외부에서는 아무 작업도 하지 않습니다. |
| `HERMES_GATEWAY_BOOTSTRAP_STATE` | s6-overlay Docker 이미지 내부에서 새 볼륨의 게이트웨이 **초기** 감독 상태를 선언합니다. 빈 볼륨에는 저장된 `gateway_state.json`이 없으므로 부트 조정자는 `gateway-default` 슬롯을 등록하지만 **down** 상태로 둡니다(마지막 기록 상태가 `running`인 경우에만 자동 시작). 이를 `running`으로 설정하면 첫 부트 설정 훅이 조정자 실행 **전에** `gateway_state.json`을 시드하므로, 최초 부트에서 게이트웨이가 시작됩니다. 리터럴 값 `running`만 적용됩니다. 최초 부트에만 적용: 기존 `gateway_state.json`은 절대 덮어쓰지 않으므로, 의도적으로 중지한 게이트웨이는 재시작 후에도 중지 상태로 유지됩니다. s6 이미지 외부에서는 아무 작업도 하지 않습니다. |
| `GATEWAY_RELAY_URL` | 실험적 릴레이 커넥터 WebSocket 기본 URL. 설정하면 게이트웨이가 일반 `relay` 어댑터를 등록하고 커넥터로 아웃바운드 다이얼합니다. `config.yaml`의 `gateway.relay_url`을 반영합니다. |
| `GATEWAY_RELAY_ID` | `hermes gateway enroll` 또는 관리형 자체 프로비저닝이 할당한 릴레이 게이트웨이 식별자. `gateway.relay_id`를 반영합니다. |
| `GATEWAY_RELAY_SECRET` | WebSocket 인증에 사용되는 게이트웨이별 릴레이 시크릿. 이미 설정되어 있으면 관리형 자체 프로비저닝을 건너뜁니다. `gateway.relay_secret`을 반영합니다. |
| `GATEWAY_RELAY_DELIVERY_KEY` | 릴레이/패스스루 인증 호환성을 위해 보존되는 커넥터 발급 전달 키. 현재 릴레이 인바운드 메시지는 게이트웨이 측 HTTP 수신기가 아니라 아웃바운드 WebSocket으로 도착합니다. |
| `GATEWAY_RELAY_ENROLL_TOKEN` | `--token`이 명시적으로 전달되지 않았을 때 `hermes gateway enroll`이 사용하는 등록 토큰. |
| `GATEWAY_RELAY_PLATFORM` | 릴레이 기능 설명자에 광고되는 선택적 플랫폼 이름. |
| `GATEWAY_RELAY_BOT_ID` | 릴레이 기능 설명자에 광고되는 선택적 봇 식별자. |
| `GATEWAY_RELAY_ENDPOINT` | 콜백/패스스루 URL이 필요한 커넥터 모드에서 기능 설명자에 광고되는 선택적 게이트웨이 엔드포인트. 기본 WS 전용 인바운드 릴레이 경로에는 필요하지 않습니다. `gateway.relay_endpoint`를 반영합니다. |
| `GATEWAY_RELAY_ROUTE_KEYS` | 커넥터에 광고되는 쉼표로 구분된 릴레이 라우트 키. `gateway.relay_route_keys`를 반영합니다. |
| `HERMES_FILE_MUTATION_VERIFIER` | 턴별 파일 변경 검증기 푸터를 활성화합니다(기본값: `true`). 활성화하면 Hermes는 해당 턴 동안 실패했으며 성공적인 쓰기로 대체되지 않은 모든 `write_file` / `patch` 호출을 나열하는 안내 목록을 추가합니다. 억제하려면 `0`, `false`, `no` 또는 `off`로 설정합니다. `config.yaml`의 `display.file_mutation_verifier`를 반영하며, 설정된 경우 환경 변수가 우선합니다. |
| `HERMES_CRON_TIMEOUT` | cron 작업 에이전트 실행의 비활성 타임아웃(초)(기본값: `600`). 도구를 활발히 호출하거나 스트림 토큰을 수신하는 동안에는 에이전트를 무기한 실행할 수 있으며, 이 값은 유휴 상태일 때만 트리거됩니다. 무제한으로 설정하려면 `0`을 사용합니다. |
| `HERMES_CRON_SCRIPT_TIMEOUT` | cron 작업에 연결된 실행 전 스크립트의 타임아웃(초)(기본값: `3600`). 스크립트에만 적용됩니다 — 스킬/에이전트 작업은 별도의 `HERMES_CRON_TIMEOUT` 비활성 예산을 사용합니다. `config.yaml`의 `cron.script_timeout_seconds`로도 설정할 수 있습니다. |
| `HERMES_CRON_MAX_PARALLEL` | 틱당 병렬로 실행되는 최대 cron 작업 수(기본값: `4`). |
## 에이전트 동작

| 변수 | 설명 |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | 대화당 도구 호출 반복 횟수의 최댓값(기본값: `500`) |
| `HERMES_INFERENCE_MODEL` | 프로세스 수준에서 모델 이름을 재정의합니다(`config.yaml`보다 우선하며 해당 세션에 적용됨). `-m`/`--model` 플래그로도 설정할 수 있습니다. |
| `HERMES_YOLO_MODE` | `1`로 설정하면 위험한 명령 승인 프롬프트를 우회합니다. `--yolo`를 전달하는 것과 같습니다. |
| `HERMES_ACCEPT_HOOKS` | TTY 프롬프트 없이 `config.yaml`에 선언된 아직 보지 않은 셸 훅을 자동 승인합니다. `--accept-hooks` 또는 `hooks_auto_accept: true`와 같습니다. |
| `HERMES_IGNORE_USER_CONFIG` | `~/.hermes/config.yaml`을 건너뛰고 기본 제공 설정을 사용합니다(자격 증명은 `.env`에서 계속 로드됨). `--ignore-user-config`와 같습니다. |
| `HERMES_IGNORE_RULES` | `AGENTS.md`, `SOUL.md`, `.cursorrules`, 메모리 및 미리 로드된 스킬의 자동 주입을 건너뜁니다. `--ignore-rules`와 같습니다. |
| `HERMES_SAFE_MODE` | 모든 커스터마이징을 비활성화하는 문제 해결 모드입니다. 플러그인 검색, MCP 서버 로드 및 셸 훅 등록을 건너뜁니다. `--safe-mode`를 사용하면 자동으로 설정되며, 이때 위의 두 플래그도 함께 설정됩니다. |
| `HERMES_TOOL_PROGRESS` | config-v12 지원 기준 이후에는 지원되지 않으며 변수가 무시됩니다. `config.yaml`에서 `display.tool_progress`를 사용하세요. |
| `HERMES_TOOL_PROGRESS_MODE` | 더 이상 사용되지 않는 호환성 변수로, 게이트웨이에서 여전히 대체 수단으로 읽습니다. `config.yaml`의 `display.tool_progress`를 우선 사용하세요. |
| `HERMES_HUMAN_DELAY_MODE` | 응답 속도 조절: `off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | 사용자 지정 지연 범위의 최솟값(ms) |
| `HERMES_HUMAN_DELAY_MAX_MS` | 사용자 지정 지연 범위의 최댓값(ms) |
| `HERMES_QUIET` | 필수적이지 않은 출력을 억제합니다(`true`/`false`). |
| `CODEX_HOME` | [Codex app-server runtime](../user-guide/features/codex-app-server-runtime)이 활성화된 경우 Codex CLI가 설정과 인증 정보를 읽는 디렉터리를 재정의합니다(기본값: `~/.codex`). Hermes의 마이그레이션은 관리 블록을 `<CODEX_HOME>/config.toml`에 기록합니다. |
| `HERMES_KANBAN_TASK` | 작업자를 생성할 때 칸반 디스패처가 설정합니다(작업 UUID). 작업자와 생성된 `hermes-tools` MCP 하위 프로세스가 이를 상속하므로 칸반 도구가 올바르게 게이트됩니다. 직접 설정하지 마세요. |
| `HERMES_ACP_SKIP_CONFIGURED_MCP` | [ACP host](../user-guide/features/acp#host-integration)가 생성하는 Hermes 하위 프로세스에 설정합니다. 세션의 MCP 서버를 `session/new`을 통해 직접 전달하는 호스트를 위해, ACP JSON-RPC 루프 전에 전역으로 구성된 `config.yaml` MCP 서버의 시작을 건너뛰도록 `1`로 설정합니다. ACP 세션이 제공한 서버는 계속 등록되며, 그 밖의 값은 기본 동작을 유지합니다. 직접 설정하지 마세요. |
| `HERMES_API_TIMEOUT` | LLM API 호출 시간 제한(초, 기본값: `1800`) |
| `HERMES_API_CALL_STALE_TIMEOUT` | 비스트리밍 호출의 응답 없음 감지 시간 제한(초, 기본값: `90`). 설정하지 않은 경우 로컬 제공자에서는 자동으로 비활성화되며, 매우 큰 컨텍스트에서는 더 늘어날 수 있습니다. `config.yaml`의 `providers.<id>.stale_timeout_seconds` 또는 `providers.<id>.models.<model>.stale_timeout_seconds`로도 설정할 수 있습니다. |
| `HERMES_STREAM_READ_TIMEOUT` | 스트리밍 소켓 읽기 시간 제한(초, 기본값: `120`). 로컬 제공자에서는 `HERMES_API_TIMEOUT`까지 자동으로 늘어납니다. 긴 코드 생성 중 로컬 LLM의 시간 제한이 초과되면 늘리세요. |
| `HERMES_STREAM_STALE_TIMEOUT` | 응답 없는 스트림 감지 시간 제한(초, 기본값: `180`). 로컬 제공자에서는 자동으로 비활성화됩니다. 이 시간 동안 청크가 도착하지 않으면 연결을 종료합니다. |
| `HERMES_LOCAL_STREAM_STALE_TIMEOUT` | 로컬 제공자(Ollama, oMLX, llama-cpp)의 응답 없는 스트림 상한(초, 기본값: `900`). 기본 응답 없음 시간 제한이 기본값이고 로컬 엔드포인트가 감지되면, 이전의 무한 비활성화 대신 이 유한한 상한을 사용합니다. 따라서 멈춘 로컬 서버도 영원히 정지하지 않고 결국 감지기를 작동시킵니다. `config.yaml`의 `agent.local_stream_stale_timeout`으로도 설정할 수 있습니다. |
| `HERMES_STREAM_RETRIES` | 일시적인 네트워크 오류 발생 시 스트림 중간에 다시 연결을 시도하는 횟수(기본값: `3`) |
| `HERMES_STREAM_STALE_GIVEUP` | 턴을 넘겨 연속으로 응답 없음 종료가 이 횟수만큼 발생하고 완료된 응답이 없으면, 응답 없음 시간 제한이 끝날 때까지 다시 기다리는 대신 각 호출을 즉시 실행 가능한 오류와 함께 중단하는 회로 차단기입니다(기본값: `5`, `0`이면 비활성화). 완료된 응답, `/model` 전환, 폴백 활성화 또는 턴 시작 시 기본 모델 복원 시 재설정됩니다. |
| `HERMES_AGENT_TIMEOUT` | 실행 중인 에이전트의 게이트웨이 비활성 시간 제한(초, 기본값: `1800`, 30분). 모든 도구 호출과 스트리밍 토큰마다 재설정됩니다. `0`으로 설정하면 비활성화합니다. |
| `HERMES_GATEWAY_MAX_STARTS` | 재생성 폭주 회로 차단기: 폭주를 끊기 위해 지수 백오프를 적용하기 전에 윈도우 내에서 허용되는 게이트웨이 (재)시작의 최댓값(기본값: `5`, `0`이면 비활성화). `config.yaml`의 `gateway.respawn_storm.max_starts`로도 설정할 수 있습니다. |
| `HERMES_GATEWAY_START_WINDOW_S` | 재생성 폭주 차단기의 윈도우(초, 기본값: `120`). `config.yaml`의 `gateway.respawn_storm.window_seconds`로도 설정할 수 있습니다. |
| `HERMES_AGENT_TIMEOUT_WARNING` | 게이트웨이: 비활성 상태가 이 시간(초)에 도달하면 경고 메시지를 보냅니다(기본값: `HERMES_AGENT_TIMEOUT`의 75%). |
| `HERMES_AGENT_NOTIFY_INTERVAL` | 게이트웨이: 장시간 실행되는 에이전트 턴에서 진행 알림을 보내는 간격(초) |
| `HERMES_CHECKPOINT_TIMEOUT` | 파일 시스템 체크포인트 생성 시간 제한(초, 기본값: `30`) |
| `HERMES_EXEC_ASK` | 게이트웨이 모드에서 실행 승인 프롬프트를 활성화합니다(`true`/`false`). |
| `HERMES_ENABLE_PROJECT_PLUGINS` | 에이전트 로더와 대시보드 웹 서버 모두에서 저장소 로컬 플러그인(`./.hermes/plugins/`)의 자동 검색을 활성화합니다. 표준 truthy 집합인 `1` / `true` / `yes` / `on`을 대소문자 구분 없이 허용합니다. 그 외 모든 값(빈 문자열과 `0`, `false`, `no`, `off` 포함)은 **비활성화**(기본값)로 처리됩니다. 참고: GHSA-5qr3-c538-wm9j (#29156) 이후 대시보드 웹 서버는 이 변수가 활성화되어 있어도 프로젝트 플러그인의 Python `api` 파일을 자동으로 가져오지 않습니다. 프로젝트 플러그인은 정적 JS/CSS로 UI를 확장할 수 있지만, 백엔드 라우트는 `~/.hermes/plugins/` 아래로 옮긴 경우에만 로드됩니다. |
| `HERMES_PLUGINS_DEBUG` | `1`/`true`로 설정하면 stderr에 자세한 플러그인 검색 로그(검색한 디렉터리, 파싱한 매니페스트, 건너뛴 이유, 파싱 또는 `register()` 실패 시 전체 트레이스백)를 표시합니다. 플러그인 작성자를 위한 옵션입니다. |
| `HERMES_BACKGROUND_NOTIFICATIONS` | 게이트웨이의 백그라운드 프로세스 알림 모드: `concise`(기본값), `all`, `result`, `error`, `off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | API 호출 시 주입되는 임시 시스템 프롬프트(세션에 저장되지 않음) |
| `HERMES_PREFILL_MESSAGES_FILE` | API 호출 시 주입할 임시 사전 입력 메시지가 담긴 JSON 파일의 경로 |
| `HERMES_ALLOW_PRIVATE_URLS` | `true`/`false` — 도구가 localhost/사설 네트워크 URL을 가져오도록 허용합니다. 게이트웨이 모드에서는 기본적으로 꺼져 있습니다. |
| `HERMES_REDACT_SECRETS` | `true`/`false` — 도구 출력, 로그 및 채팅 응답에서 비밀 정보 마스킹을 제어합니다(기본값: `true`). |
| `HERMES_WRITE_SAFE_ROOT` | 나열된 루트 외부에 대한 `write_file`/`patch` 쓰기를 **강제로 차단**하는 선택적 디렉터리 접두사입니다(승인 프롬프트 없음). `os.pathsep`(`:`(Unix), `;`(Windows))로 구분한 여러 디렉터리를 지원합니다. 아래의 [HERMES_WRITE_SAFE_ROOT](#hermes_write_safe_root)를 참조하세요. |
| `HERMES_DISABLE_LAZY_INSTALLS` | 변경할 수 없는 `/opt/hermes` 트리에 런타임 의존성을 설치하지 않도록 공식 Docker 이미지에서 자동으로 설정하는 내부 브리지 변수입니다. 사용자에게 제공되는 대응 설정은 `config.yaml`의 `security.allow_lazy_installs: false`이므로, 이 변수를 `.env`에 설정하지 마세요. |
| `HERMES_DISABLE_FILE_STATE_GUARD` | `1`로 설정하면 `patch`/`write_file`의 "읽은 이후 파일이 변경됨" 가드를 끕니다. |
| `HERMES_BUNDLED_SKILLS` | 시작 시 로드되는 번들 스킬 목록을 쉼표로 구분한 값으로 재정의합니다. |
| `HERMES_OPTIONAL_SKILLS` | 첫 실행 시 자동 설치할 선택적 스킬 이름을 쉼표로 구분한 목록입니다. |
| `HERMES_DEBUG_INTERRUPT` | `1`로 설정하면 자세한 인터럽트/취소 추적을 `agent.log`에 기록합니다. |
| `HERMES_DUMP_REQUESTS` | API 요청 페이로드를 로그 파일에 덤프합니다(`true`/`false`). |
| `HERMES_DUMP_REQUEST_STDOUT` | API 요청 페이로드를 로그 파일 대신 stdout에 덤프합니다. |
| `HERMES_OAUTH_TRACE` | `1`로 설정하면 OAuth 토큰 교환 및 갱신 시도를 기록합니다. 마스킹된 타이밍 정보가 포함됩니다. |
| `HERMES_AGENT_HELP_GUIDANCE` | 사용자 지정 배포를 위해 시스템 프롬프트에 추가 안내 텍스트를 덧붙입니다. |
| `HERMES_AGENT_LOGO` | CLI 시작 시 ASCII 배너 로고를 재정의합니다. |
| `DELEGATION_MAX_CONCURRENT_CHILDREN` | `delegate_task` 배치당 최대 병렬 하위 에이전트 수(기본값: `3`, 최솟값 1, 최댓값 제한 없음). `config.yaml`의 `delegation.max_concurrent_children`으로도 설정할 수 있으며, 구성 파일의 값이 우선합니다. |

### HERMES_WRITE_SAFE_ROOT {#hermes_write_safe_root}

이 변수가 설정되면 `write_file`과 `patch`는 나열된 디렉터리 접두사 내부의 경로만 대상으로 지정할 수 있습니다. 해당 루트 외부의 경로는 모두 **즉시 거부**되며, 위험한 명령 승인 시스템을 거치지 않고 덮어쓸 프롬프트도 표시되지 않습니다.

공식 Docker 이미지는 에이전트가 마운트된 데이터 볼륨에서 벗어날 수 없도록 `HERMES_HOME=/opt/data`와 함께 `HERMES_WRITE_SAFE_ROOT=/opt/data`를 설정합니다.

샌드박스 쓰기를 의도한 경우가 아니라면 **이를 `~/.hermes/.env`에 추가하지 마세요.** 프로젝트 디렉터리를 가리키면서도 에이전트가 `~/.hermes/cron/jobs.json`, `~/.hermes/skills/` 또는 프로필 아래의 스크립트를 편집할 수 있다고 예상하는 것이 흔한 실수입니다. 이러한 경로는 샌드박스 외부에 있으므로 모든 `write_file`/`patch`가 `outside HERMES_WRITE_SAFE_ROOT` 오류와 함께 실패합니다.

작업 공간과 Hermes 상태를 모두 허용하려면 두 접두사를 모두 나열하세요(순서는 상관없음).

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

변수를 설정 해제하거나 `.env`에서 제거하면 일반 쓰기로 돌아갑니다(여전히 자격 증명 경로 차단 목록의 적용을 받음 — [파일 쓰기 안전성](../user-guide/security.md#file-write-safety) 참조).

## 인터페이스

| 변수 | 설명 |
|----------|-------------|
| `HERMES_TUI` | `1`로 설정하면 기존 CLI 대신 [TUI](../user-guide/tui.md)를 실행합니다. `--tui`를 전달하는 것과 같습니다. |
| `HERMES_TUI_DIR` | 미리 빌드된 `ui-tui/` 디렉터리의 경로입니다(`dist/entry.js`와 채워진 `node_modules`를 포함해야 함). 배포판과 Nix에서 첫 실행 시 `npm install`을 건너뛰는 데 사용합니다. |
| `HERMES_TUI_RESUME` | 시작할 때 ID로 특정 TUI 세션을 재개합니다. 설정하면 `hermes --tui`가 새 세션을 만들지 않고 지정한 세션을 이어받습니다. 연결이 끊겼거나 터미널이 충돌한 후 다시 연결할 때 유용합니다. |
| `HERMES_TUI_THEME` | TUI 색상 테마를 강제합니다: `light`, `dark` 또는 6자리 배경 hex 원시 값(예: `ffffff`, `1a1a2e`). 설정하지 않으면 Hermes가 `COLORFGBG`와 터미널 배경 질의를 사용해 자동 감지합니다. 이 변수는 `COLORFGBG`를 설정하지 않는 터미널(Ghostty, Warp, iTerm2 등)에서 감지를 재정의합니다. |
| `HERMES_INFERENCE_MODEL` | `config.yaml`을 변경하지 않고 `hermes -z` / `hermes chat`에 사용할 모델을 강제합니다. `--provider` 플래그와 함께 사용합니다. 실행마다 기본 모델을 재정의해야 하는 스크립트 호출자(sweeper, CI, 배치 러너 등)에게 유용합니다. |

## 세션 설정

| 변수 | 설명 |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | 일정 시간 동안 활동이 없으면 세션을 초기화합니다(분, 기본값: 1440). |
| `SESSION_RESET_HOUR` | 24시간 형식의 일일 초기화 시각(기본값: 4 = 오전 4시) |
| `HERMES_SESSION_ID` | Hermes가 생성하는 모든 도구 하위 프로세스(`terminal`, `execute_code`, 영구 셸, Docker/Singularity 백엔드, 위임된 하위 에이전트 실행)에 **자동으로 내보내집니다**. 에이전트가 현재 세션 ID로 설정하며, 도구에서 호출된 사용자 스크립트는 이를 읽어 자신의 출력, 텔레메트리 또는 부수 효과를 원래 Hermes 세션과 연결할 수 있습니다. **직접 설정하지 마세요** — 상위 셸에서 재정의해도 에이전트 실행 외부에서만 적용되며, 에이전트가 세션을 시작하는 즉시 덮어써집니다. |
| `AI_AGENT` | **CLI와 게이트웨이 진입점에서 `hermes-agent`로 설정됩니다**(외부 하네스가 이미 설정한 경우에는 제외). 원격 백엔드(Docker, SSH, Modal, Daytona, Singularity, Vercel)를 포함해 모든 터미널 도구 셸에 내보냅니다. 새롭게 부상하는 하위 프로세스 어트리뷰션용 에이전트 간 표준으로, 일반 도구(예: huggingface_hub의 에이전트 감지)가 이를 읽어 자신이 AI 에이전트 아래에서 실행 중임을 파악합니다. 값은 공개 에이전트 하네스 레지스트리에 등록된 Hermes의 ID와 일치합니다. 직접 설정하지 마세요. |
| `HERMES_AGENT` | **CLI와 게이트웨이 진입점에서 `true`로 설정됩니다**. 모든 터미널 도구 셸에 내보내므로 하위 프로세스가 특히 Hermes 내부에서 실행 중인지 감지할 수 있습니다. 직접 설정하지 마세요. |
## Context Compression (config.yaml만 사용)

컨텍스트 압축은 `config.yaml`을 통해서만 구성할 수 있으며 환경 변수는 없습니다. 임계값 설정은 `compression:` 블록에, 요약 모델 및 제공자는 `auxiliary.compression:` 아래에 지정합니다.

```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20         # fraction of threshold to preserve as recent tail
  protect_last_n: 20         # minimum recent messages to keep uncompressed
```

:::info 레거시 마이그레이션
`compression.summary_model`, `compression.summary_provider`, `compression.summary_base_url`을 사용하는 이전 설정은 처음 로드할 때 자동으로 `auxiliary.compression.*`으로 마이그레이션됩니다.
:::

## 보조 작업 재정의

| 변수 | 설명 |
|----------|-------------|
| `AUXILIARY_VISION_PROVIDER` | 비전 작업에 사용할 제공자 재정의 |
| `AUXILIARY_VISION_MODEL` | 비전 작업에 사용할 모델 재정의 |
| `AUXILIARY_VISION_BASE_URL` | 비전 작업을 위한 직접 OpenAI 호환 엔드포인트 |
| `AUXILIARY_VISION_API_KEY` | `AUXILIARY_VISION_BASE_URL`에 연결된 API 키 |
| `AUXILIARY_WEB_EXTRACT_PROVIDER` | 웹 추출/요약에 사용할 제공자 재정의 |
| `AUXILIARY_WEB_EXTRACT_MODEL` | 웹 추출/요약에 사용할 모델 재정의 |
| `AUXILIARY_WEB_EXTRACT_BASE_URL` | 웹 추출/요약을 위한 직접 OpenAI 호환 엔드포인트 |
| `AUXILIARY_WEB_EXTRACT_API_KEY` | `AUXILIARY_WEB_EXTRACT_BASE_URL`에 연결된 API 키 |

작업별 직접 엔드포인트의 경우 Hermes는 해당 작업에 구성된 API 키 또는 `OPENAI_API_KEY`를 사용합니다. 이러한 사용자 지정 엔드포인트에는 `OPENROUTER_API_KEY`를 재사용하지 않습니다.

## 폴백 제공자 (config.yaml만 사용)

기본 모델 폴백 체인은 `config.yaml`을 통해서만 구성할 수 있으며 환경 변수는 없습니다. 최상위에 `provider` 및 `model` 키를 포함하는 `fallback_providers` 목록을 추가하면 기본 모델에서 오류가 발생할 때 자동 장애 조치를 활성화할 수 있습니다. 제공자가 `auto`인 보조 작업도 이 체인을 먼저 확인한 다음 Hermes에 내장된 보조 제공자 검색 체인을 사용합니다.

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

이전의 최상위 `fallback_model` 단일 제공자 형식도 이전 버전과의 호환성을 위해 계속 읽지만, 새 구성에는 `fallback_providers`를 사용해야 합니다. 작업별 보조 정책에는 `config.yaml`의 `auxiliary.<task>.fallback_chain`을 사용하며, 이에 대응하는 환경 변수는 없습니다.

[폴백 제공자](/user-guide/features/fallback-providers)에서 자세한 내용을 확인하세요.

## 제공자 라우팅 (config.yaml만 사용)

다음 항목은 `~/.hermes/config.yaml`의 `provider_routing` 섹션 아래에 지정합니다.

| 키 | 설명 |
|-----|-------------|
| `sort` | 제공자 정렬 기준: `"price"`(기본값), `"throughput"` 또는 `"latency"` |
| `only` | 허용할 제공자 슬러그 목록(예: `["anthropic", "google"]`) |
| `ignore` | 건너뛸 제공자 슬러그 목록 |
| `order` | 순서대로 시도할 제공자 슬러그 목록 |
| `require_parameters` | 모든 요청 매개변수를 지원하는 제공자만 사용(`true`/`false`) |
| `data_collection` | 데이터 저장 제공자를 제외하려면 `"allow"`(기본값) 또는 `"deny"` |

:::tip
환경 변수를 설정하려면 `hermes config set`을 사용하세요. 이 명령은 환경 변수를 올바른 파일에 자동으로 저장합니다(시크릿은 `.env`, 그 외 모든 항목은 `config.yaml`).
:::
