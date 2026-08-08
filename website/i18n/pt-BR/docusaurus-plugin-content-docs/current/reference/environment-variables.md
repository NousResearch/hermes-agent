---
sidebar_position: 2
title: "Variáveis de Ambiente"
description: "Referência completa de todas as variáveis de ambiente usadas pelo Hermes Agent"
---

# Referência de Variáveis de Ambiente

O Hermes lê variáveis de ambiente do ambiente do processo e, para segredos gerenciados pelo usuário, de `~/.hermes/.env`. Mantenha chaves de API, tokens de bot, segredos OAuth e outras credenciais em `.env`; prefira `config.yaml` para configurações de comportamento não secretas quando existir uma chave de config. Algumas variáveis abaixo são sobrescritas apenas de processo ou variáveis de bridge internas e não devem ser adicionadas ao `.env` apenas porque estão documentadas aqui.

## Provedores de LLM {#llm-providers}

| Variável | Descrição |
|----------|-------------|
| `OPENROUTER_API_KEY` | Chave de API do OpenRouter (recomendada para flexibilidade) |
| `OPENROUTER_BASE_URL` | Sobrescreve a URL base compatível com OpenRouter |
| `FIREWORKS_API_KEY` | Chave de API da Fireworks AI ([app.fireworks.ai](https://app.fireworks.ai/settings/users/api-keys)). Configure sobrescritas de endpoint com `model.base_url` no `config.yaml`. |
| `HERMES_OPENROUTER_CACHE` | Ativa o cache de resposta do OpenRouter (`1`/`true`/`yes`/`on`). Sobrescreve `openrouter.response_cache` no config.yaml. Veja [Cache de Resposta](https://openrouter.ai/docs/guides/features/response-caching). |
| `HERMES_OPENROUTER_CACHE_TTL` | TTL do cache em segundos (1-86400). Sobrescreve `openrouter.response_cache_ttl` no config.yaml. |
| `NOUS_BASE_URL` | Sobrescreve a URL base do Nous Portal (raramente necessário; apenas desenvolvimento/teste) |
| `NOUS_INFERENCE_BASE_URL` | Sobrescreve o endpoint de inferência da Nous diretamente |
| `OPENAI_API_KEY` | Chave de API para endpoints customizados compatíveis com OpenAI (usada com `OPENAI_BASE_URL`) |
| `OPENAI_BASE_URL` | URL base para endpoint customizado (VLLM, SGLang, etc.) |
| `LM_API_KEY` | Chave de API para o LM Studio (provedor `lmstudio`). Frequentemente um placeholder para servidores locais |
| `LM_BASE_URL` | URL base do LM Studio (padrão: `http://localhost:1234/v1`) |
| `COPILOT_GITHUB_TOKEN` | Token do GitHub para a API do Copilot — primeira prioridade (OAuth `gho_*` ou PAT de granularidade fina `github_pat_*`; PATs clássicos `ghp_*` **não são suportados**) |
| `GH_TOKEN` | Token do GitHub — segunda prioridade para o Copilot (também usado pelo CLI `gh`) |
| `GITHUB_TOKEN` | Token do GitHub — terceira prioridade para o Copilot |
| `HERMES_COPILOT_ACP_COMMAND` | Sobrescreve o caminho do binário CLI do Copilot ACP (padrão: `copilot`) |
| `COPILOT_CLI_PATH` | Alias para `HERMES_COPILOT_ACP_COMMAND` |
| `HERMES_COPILOT_ACP_ARGS` | Sobrescreve os argumentos do Copilot ACP (padrão: `--acp --stdio`) |
| `COPILOT_ACP_BASE_URL` | Sobrescreve a URL base do Copilot ACP |
| `COPILOT_API_BASE_URL` | Sobrescreve a URL base da API do Copilot (provedor `copilot`) |
| `GLM_API_KEY` | Chave de API GLM da z.ai / ZhipuAI ([z.ai](https://z.ai)) |
| `ZAI_API_KEY` | Alias para `GLM_API_KEY` |
| `Z_AI_API_KEY` | Alias para `GLM_API_KEY` |
| `GLM_BASE_URL` | Sobrescreve a URL base da z.ai (padrão: `https://api.z.ai/api/paas/v4`) |
| `KIMI_API_KEY` | Chave de API Kimi / Moonshot AI ([moonshot.ai](https://platform.moonshot.ai)) |
| `KIMI_CODING_API_KEY` | Chave alias para o provedor `kimi-coding` (aceita junto com `KIMI_API_KEY`) |
| `KIMI_BASE_URL` | Sobrescreve a URL base da Kimi (padrão: `https://api.moonshot.ai/v1`) |
| `KIMI_CN_API_KEY` | Chave de API Kimi / Moonshot China ([moonshot.cn](https://platform.moonshot.cn)) |
| `ARCEEAI_API_KEY` | Chave de API da Arcee AI ([chat.arcee.ai](https://chat.arcee.ai/)) |
| `ARCEE_BASE_URL` | Sobrescreve a URL base da Arcee (padrão: `https://api.arcee.ai/api/v1`) |
| `GMI_API_KEY` | Chave de API da GMI Cloud ([gmicloud.ai](https://www.gmicloud.ai/)) |
| `GMI_BASE_URL` | Sobrescreve a URL base da GMI Cloud (padrão: `https://api.gmi-serving.com/v1`) |
| `MINIMAX_API_KEY` | Chave de API MiniMax — endpoint global ([minimax.io](https://www.minimax.io)). **Não usada pelo `minimax-oauth`** (o caminho OAuth usa login por navegador em vez disso). |
| `MINIMAX_BASE_URL` | Sobrescreve a URL base do MiniMax (padrão: `https://api.minimax.io/anthropic` — o Hermes usa o endpoint compatível com Anthropic Messages da MiniMax). **Não usada pelo `minimax-oauth`**. |
| `MINIMAX_CN_API_KEY` | Chave de API MiniMax — endpoint China ([minimaxi.com](https://www.minimaxi.com)). **Não usada pelo `minimax-oauth`** (o caminho OAuth usa login por navegador em vez disso). |
| `MINIMAX_CN_BASE_URL` | Sobrescreve a URL base do MiniMax China (padrão: `https://api.minimaxi.com/anthropic`). **Não usada pelo `minimax-oauth`**. |
| `KILOCODE_API_KEY` | Chave de API do Kilo Code ([kilo.ai](https://kilo.ai)) |
| `KILOCODE_BASE_URL` | Sobrescreve a URL base do Kilo Code (padrão: `https://api.kilo.ai/api/gateway`) |
| `XIAOMI_API_KEY` | Chave de API Xiaomi MiMo ([platform.xiaomimimo.com](https://platform.xiaomimimo.com)) |
| `XIAOMI_BASE_URL` | Sobrescreve a URL base do Xiaomi MiMo (padrão: `https://api.xiaomimimo.com/v1`) |
| `UPSTAGE_API_KEY` | Chave de API da Upstage para modelos Solar ([console.upstage.ai](https://console.upstage.ai/api-keys)) |
| `UPSTAGE_BASE_URL` | Sobrescreve a URL base da Upstage (padrão: `https://api.upstage.ai/v1`) |
| `TOKENHUB_API_KEY` | Chave de API do Tencent TokenHub ([tokenhub.tencentmaas.com](https://tokenhub.tencentmaas.com)) |
| `TOKENHUB_BASE_URL` | Sobrescreve a URL base do Tencent TokenHub (padrão: `https://tokenhub.tencentmaas.com/v1`) |
| `AZURE_FOUNDRY_API_KEY` | Chave de API do Microsoft Foundry / Azure OpenAI ([ai.azure.com](https://ai.azure.com/)). Não necessária quando `model.auth_mode: entra_id` |
| `AZURE_FOUNDRY_BASE_URL` | URL de endpoint do Microsoft Foundry (ex.: `https://<resource>.openai.azure.com/openai/v1` para estilo OpenAI, ou `https://<resource>.services.ai.azure.com/anthropic` para estilo Anthropic) |
| `AZURE_ANTHROPIC_KEY` | Chave de API Azure Anthropic para `provider: anthropic` + `base_url` apontando para uma implantação Claude no Microsoft Foundry (alternativa a `ANTHROPIC_API_KEY` quando tanto Anthropic quanto Azure Anthropic estão configurados) |
| `AZURE_TENANT_ID` | ID de tenant do Entra ID (fluxos de service-principal; respeitado pelo `azure-identity` quando `model.auth_mode: entra_id`) |
| `AZURE_CLIENT_ID` | ID de cliente do Entra ID (service principal, workload identity, ou identidade gerenciada atribuída pelo usuário) |
| `AZURE_CLIENT_SECRET` | Segredo do service principal usado pelo `EnvironmentCredential` |
| `AZURE_CLIENT_CERTIFICATE_PATH` | Certificado do service principal (alternativa a `AZURE_CLIENT_SECRET`) |
| `AZURE_FEDERATED_TOKEN_FILE` | Caminho do arquivo de token federado para fluxos AKS Workload Identity / OIDC |
| `AZURE_AUTHORITY_HOST` | Sobrescrita de authority para nuvens soberanas (ex.: `https://login.microsoftonline.us` para o Azure Government). Veja o [guia do Azure Foundry](/guides/azure-foundry#sovereign-clouds-government-china) |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | Endpoint de Managed Identity para App Service, Functions e Container Apps; VMs geralmente usam IMDS em vez disso e não definem essas variáveis |
| `HF_TOKEN` | Token do Hugging Face para Inference Providers ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |
| `HF_BASE_URL` | Sobrescreve a URL base do Hugging Face (padrão: `https://router.huggingface.co/v1`) |
| `GOOGLE_API_KEY` | Chave de API do Google AI Studio ([aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) |
| `GEMINI_API_KEY` | Alias para `GOOGLE_API_KEY` |
| `GEMINI_BASE_URL` | Sobrescreve a URL base do Google AI Studio |
| `ANTHROPIC_API_KEY` | Chave de API do Anthropic Console ([console.anthropic.com](https://console.anthropic.com/)) |
| `ANTHROPIC_BASE_URL` | Sobrescreve a URL base da API Anthropic |
| `ANTHROPIC_TOKEN` | Sobrescrita manual ou legada do OAuth/setup-token da Anthropic |
| `DASHSCOPE_API_KEY` | Chave de API Qwen Cloud (Alibaba DashScope) para modelos Qwen ([modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com/)) |
| `DASHSCOPE_BASE_URL` | URL base customizada do DashScope (padrão: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`; use `https://dashscope.aliyuncs.com/compatible-mode/v1` para a região da China continental) |
| `ALIBABA_CODING_PLAN_API_KEY` | Chave de API do Qwen Coding Plan (provedor `alibaba-coding-plan`) |
| `ALIBABA_CODING_PLAN_BASE_URL` | Sobrescreve a URL base do Qwen Coding Plan |
| `DEEPSEEK_API_KEY` | Chave de API DeepSeek para acesso direto ao DeepSeek ([platform.deepseek.com](https://platform.deepseek.com/api_keys)) |
| `DEEPSEEK_BASE_URL` | URL base customizada da API DeepSeek |
| `NOVITA_API_KEY` | Chave de API NovitaAI — nuvem nativa de IA para Model API, Agent Sandbox e GPU Cloud ([novita.ai/settings/key-management](https://novita.ai/settings/key-management)) |
| `NOVITA_BASE_URL` | Sobrescreve a URL base do NovitaAI (padrão: `https://api.novita.ai/openai/v1`) |
| `NVIDIA_API_KEY` | Chave de API NVIDIA NIM — Nemotron e modelos abertos ([build.nvidia.com](https://build.nvidia.com)) |
| `NVIDIA_BASE_URL` | Sobrescreve a URL base da NVIDIA (padrão: `https://integrate.api.nvidia.com/v1`; defina como `http://localhost:8000/v1` para um endpoint NIM local) |
| `STEPFUN_API_KEY` | Chave de API StepFun — modelos da série Step ([platform.stepfun.com](https://platform.stepfun.com)) |
| `STEPFUN_BASE_URL` | Sobrescreve a URL base da StepFun (padrão: `https://api.stepfun.com/v1`) |
| `OLLAMA_API_KEY` | Chave de API Ollama Cloud — catálogo Ollama gerenciado sem GPU local ([ollama.com/settings/keys](https://ollama.com/settings/keys)) |
| `OLLAMA_BASE_URL` | Sobrescreve a URL base do Ollama Cloud (padrão: `https://ollama.com/v1`) |
| `XAI_API_KEY` | Chave de API xAI (Grok) para chat + TTS + busca web ([console.x.ai](https://console.x.ai/)) |
| `XAI_BASE_URL` | Sobrescreve a URL base da xAI (padrão: `https://api.x.ai/v1`) |
| `MISTRAL_API_KEY` | Chave de API Mistral para Voxtral TTS e Voxtral STT ([console.mistral.ai](https://console.mistral.ai)) |
| `AWS_REGION` | Região AWS para inferência Bedrock (ex.: `us-east-1`, `eu-central-1`). Lida pelo boto3. |
| `AWS_PROFILE` | Perfil AWS nomeado para autenticação Bedrock (lê `~/.aws/credentials`). Deixe indefinido para usar a cadeia de credenciais boto3 padrão. |
| `BEDROCK_BASE_URL` | Sobrescreve a URL base de runtime do Bedrock (padrão: `https://bedrock-runtime.us-east-1.amazonaws.com`; geralmente deixe indefinido e use `AWS_REGION` em vez disso) |
| `HERMES_QWEN_BASE_URL` | Sobrescrita da URL base do Qwen Portal (padrão: `https://portal.qwen.ai/v1`) |
| `OPENCODE_ZEN_API_KEY` | Chave de API OpenCode Zen — acesso pay-as-you-go a modelos selecionados ([opencode.ai](https://opencode.ai/auth)) |
| `OPENCODE_ZEN_BASE_URL` | Sobrescreve a URL base do OpenCode Zen |
| `OPENCODE_GO_API_KEY` | Chave de API OpenCode Go — assinatura de $10/mês para modelos abertos ([opencode.ai](https://opencode.ai/auth)) |
| `OPENCODE_GO_BASE_URL` | Sobrescreve a URL base do OpenCode Go |
| `CLAUDE_CODE_OAUTH_TOKEN` | Sobrescrita explícita do token do Claude Code se você exportar um manualmente |
| `HERMES_MODEL` | Sobrescreve o nome do modelo em nível de processo (usado pelo agendador de cron; prefira `config.yaml` para uso normal) |
| `VOICE_TOOLS_OPENAI_KEY` | Chave OpenAI preferida para os provedores de speech-to-text e text-to-speech da OpenAI |
| `HERMES_LOCAL_STT_COMMAND` | Template de comando opcional de speech-to-text local. Suporta os placeholders `{input_path}`, `{output_dir}`, `{language}` e `{model}` |
| `HERMES_LOCAL_STT_LANGUAGE` | Idioma padrão passado para `HERMES_LOCAL_STT_COMMAND` ou fallback do CLI local `whisper` detectado automaticamente (padrão: `en`) |
| `HERMES_HOME` | Sobrescreve o diretório de configuração do Hermes (padrão: `~/.hermes`). Também define o escopo do arquivo PID do gateway e do nome do serviço systemd, para que várias instalações possam rodar simultaneamente |
| `HERMES_GIT_BASH_PATH` | **Somente Windows.** Sobrescreve a descoberta do `bash.exe` para a ferramenta de terminal. Aponta para qualquer bash — instalação completa do Git para Windows, bash do WSL via symlink, MSYS2, Cygwin. O instalador define isso automaticamente para o PortableGit que provisionou. Veja o [Guia do Windows (Nativo)](../user-guide/windows-native.md#how-hermes-runs-shell-commands-on-windows) |
| `HERMES_DISABLE_WINDOWS_UTF8` | **Somente Windows.** Defina como `1` para desativar o shim de stdio UTF-8 (`configure_windows_stdio()`) e voltar à code page de localidade do console. Útil para isolar bugs de codificação; raramente é a configuração correta em operação normal |
| `HERMES_KANBAN_HOME` | Sobrescreve a raiz compartilhada do Hermes que ancora o quadro kanban (db + workspaces + logs de worker). Recorre a `get_default_hermes_root()` (o pai de qualquer perfil ativo). Útil para testes e implantações não usuais |
| `HERMES_KANBAN_BOARD` | Fixa o quadro kanban ativo para este processo. Tem precedência sobre `~/.hermes/kanban/current`; o dispatcher injeta isso no ambiente do subprocesso worker para que os workers fisicamente não possam ver tarefas em outros quadros. Padrão: `default`. Validação de slug: alfanuméricos minúsculos + hífens + underscores, 1-64 caracteres |
| `HERMES_KANBAN_DB` | Fixa o caminho do arquivo de banco de dados kanban diretamente (maior precedência; supera `HERMES_KANBAN_BOARD` e `HERMES_KANBAN_HOME`). O dispatcher injeta isso no ambiente do subprocesso worker para que os workers de perfil convirjam para o quadro do dispatcher |
| `HERMES_KANBAN_WORKSPACES_ROOT` | Fixa a raiz dos workspaces kanban diretamente (maior precedência para workspaces; supera `HERMES_KANBAN_HOME`). O dispatcher injeta isso no ambiente do subprocesso worker |
| `HERMES_KANBAN_DISPATCH_IN_GATEWAY` | Sobrescrita em tempo de execução para `kanban.dispatch_in_gateway`. Defina como `0`, `false`, `no`, ou `off` para impedir que o gateway inicie o dispatcher Kanban embutido; qualquer outro valor não vazio o ativa. Útil quando um processo dispatcher separado é dono do quadro. |

## Autenticação de Provedor (OAuth) {#provider-auth-oauth}

Para autenticação nativa da Anthropic, o Hermes prefere os próprios arquivos de credencial do Claude Code quando existem, porque essas credenciais podem se renovar automaticamente. **OAuth contra a Anthropic requer um plano Claude Max com créditos extras de uso comprados** — o Hermes roteia como Claude Code, que só consome dos créditos extras/de excedente do plano Max, não da cota base do Max, e não funciona no Claude Pro. Sem Max + créditos extras, use uma chave de API em vez disso. Variáveis de ambiente como `ANTHROPIC_TOKEN` permanecem úteis como sobrescritas manuais, mas não são mais o caminho preferido para login no Claude Max.

| Variável | Descrição |
|----------|-------------|
| `HERMES_PORTAL_BASE_URL` | Sobrescreve a URL do Nous Portal (para desenvolvimento/teste) |
| `NOUS_INFERENCE_BASE_URL` | Sobrescreve a URL da API de inferência Nous |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | TTL mínimo da chave de agente antes de renovar (padrão: 1800 = 30min) |
| `HERMES_NOUS_TIMEOUT_SECONDS` | Timeout HTTP para fluxos de credencial/token da Nous |
| `HERMES_DUMP_REQUESTS` | Despeja os payloads de requisição de API em arquivos de log (`true`/`false`) |
| `HERMES_PREFILL_MESSAGES_FILE` | Caminho para um arquivo JSON de mensagens de prefill efêmeras injetadas no momento da chamada de API |
| `HERMES_TIMEZONE` | Sobrescrita de timezone IANA (por exemplo, `America/New_York`) |

## APIs de Ferramentas {#tool-apis}

| Variável | Descrição |
|----------|-------------|
| `PARALLEL_API_KEY` | Busca web nativa de IA ([parallel.ai](https://parallel.ai/)) |
| `FIRECRAWL_API_KEY` | Web scraping e browser em nuvem ([firecrawl.dev](https://firecrawl.dev/)) |
| `FIRECRAWL_API_URL` | Endpoint customizado da API Firecrawl para instâncias auto-hospedadas (opcional) |
| `TAVILY_API_KEY` | Chave de API Tavily para busca web nativa de IA, extração e crawling ([app.tavily.com](https://app.tavily.com/home)) |
| `SEARXNG_URL` | URL da instância SearXNG para busca web gratuita auto-hospedada — sem necessidade de chave de API ([searxng.github.io](https://searxng.github.io/searxng/)) |
| `TAVILY_BASE_URL` | Sobrescreve o endpoint da API Tavily. Útil para proxies corporativos e backends de busca compatíveis com Tavily auto-hospedados. Mesmo padrão que `GROQ_BASE_URL`. |
| `EXA_API_KEY` | Chave de API Exa para busca web e conteúdos nativos de IA ([exa.ai](https://exa.ai/)) |
| `BRAVE_SEARCH_API_KEY` | Token de assinatura da API Brave Search para busca web (camada gratuita disponível) ([brave.com/search/api](https://brave.com/search/api/)) |
| `BROWSERBASE_API_KEY` | Automação de navegador ([browserbase.com](https://browserbase.com/)) |
| `BROWSERBASE_PROJECT_ID` | ID de projeto Browserbase |
| `BROWSER_USE_API_KEY` | Chave de API do navegador em nuvem Browser Use ([browser-use.com](https://browser-use.com/)) |
| `FIRECRAWL_BROWSER_TTL` | TTL da sessão de navegador Firecrawl em segundos (padrão: 300) |
| `BROWSER_CDP_URL` | URL do Chrome DevTools Protocol para navegador local (definida via `/browser connect`, ex.: `ws://localhost:9222`) |
| `CAMOFOX_URL` | URL do navegador anti-detecção local Camofox (padrão: `http://localhost:9377`) |
| `CAMOFOX_USER_ID` | ID de usuário Camofox opcional gerenciado externamente para sessões visíveis compartilhadas |
| `CAMOFOX_SESSION_KEY` | Chave de sessão Camofox opcional usada ao criar abas para `CAMOFOX_USER_ID` |
| `CAMOFOX_ADOPT_EXISTING_TAB` | Defina como `true` para reutilizar uma aba Camofox existente antes de criar uma nova |
| `BROWSER_INACTIVITY_TIMEOUT` | Timeout de inatividade da sessão de navegador em segundos |
| `AGENT_BROWSER_ARGS` | Flags extras de inicialização do Chromium (separadas por vírgula ou nova linha). O Hermes injeta automaticamente `--no-sandbox,--disable-dev-shm-usage` quando rodando como root ou em namespaces de usuário não privilegiado restritos por AppArmor (Ubuntu 23.10+, DGX Spark, muitas imagens de container); defina isso manualmente apenas para sobrescrever ou adicionar outras flags. |
| `AGENT_BROWSER_ENGINE` | Motor de navegador para modo local: `auto` (padrão — família Chromium via CDP), ou uma sobrescrita de motor específico. |
| `FAL_KEY` | Geração de imagem ([fal.ai](https://fal.ai/)) |
| `KREA_API_KEY` | Chave de API Krea para geração de imagem Krea 2 ([krea.ai](https://krea.ai/)) |
| `GROQ_API_KEY` | Chave de API Groq Whisper STT ([groq.com](https://groq.com/)) |
| `ELEVENLABS_API_KEY` | Vozes TTS premium da ElevenLabs ([elevenlabs.io](https://elevenlabs.io/)) |
| `STT_GROQ_MODEL` | Sobrescreve o modelo STT do Groq (padrão: `whisper-large-v3-turbo`) |
| `GROQ_BASE_URL` | Sobrescreve o endpoint STT compatível com OpenAI do Groq |
| `STT_OPENAI_MODEL` | Sobrescreve o modelo STT da OpenAI (padrão: `whisper-1`) |
| `STT_OPENAI_BASE_URL` | Sobrescreve o endpoint STT compatível com OpenAI |
| `GITHUB_TOKEN` | Token do GitHub para o Skills Hub (limites de taxa de API mais altos, publicação de skills) |
| `HONCHO_API_KEY` | Modelagem de usuário entre sessões ([honcho.dev](https://honcho.dev/)) |
| `HONCHO_BASE_URL` | URL base para instâncias Honcho auto-hospedadas (padrão: nuvem Honcho). Nenhuma chave de API necessária para instâncias locais |
| `HINDSIGHT_TIMEOUT` | Timeout em segundos para chamadas de API do provedor de memória Hindsight (padrão: `60`). Aumente isso se sua instância Hindsight for lenta para responder durante `/sync` ou `on_session_switch` e você estiver vendo timeouts em `errors.log`. |
| `SUPERMEMORY_API_KEY` | Memória semântica de longo prazo com recall de perfil e ingestão de sessão ([supermemory.ai](https://supermemory.ai)) |
| `DAYTONA_API_KEY` | Sandboxes em nuvem Daytona ([daytona.io](https://daytona.io/)) |

### Chaves de API de Skills {#skill-api-keys}

Segredos consumidos por skills específicas incluídas / opcionais. Cada uma só é necessária se você usar a skill correspondente.

| Variável | Usada pela skill | Descrição |
|----------|---------------|-------------|
| `NOTION_API_KEY` | `notion` | Token de integração do Notion. |
| `LINEAR_API_KEY` | `linear` | Chave de API pessoal do Linear. |
| `AIRTABLE_API_KEY` | `airtable` | Token de acesso pessoal do Airtable. |
| `TENOR_API_KEY` | `gif-search` | Chave de API Tenor para busca de GIFs. |

### Observabilidade Langfuse {#langfuse-observability}

Variáveis de ambiente para o plugin incluído [`observability/langfuse`](/user-guide/features/built-in-plugins#observabilitylangfuse). Defina-as em `~/.hermes/.env`. O plugin também deve estar ativado (`hermes plugins enable observability/langfuse`, ou marque a caixa em `hermes plugins`) antes que qualquer uma dessas tenha efeito.

| Variável | Descrição |
|----------|-------------|
| `HERMES_LANGFUSE_PUBLIC_KEY` | Chave pública do projeto Langfuse (`pk-lf-...`). Obrigatória. |
| `HERMES_LANGFUSE_SECRET_KEY` | Chave secreta do projeto Langfuse (`sk-lf-...`). Obrigatória. |
| `HERMES_LANGFUSE_BASE_URL` | URL do servidor Langfuse (padrão: `https://cloud.langfuse.com`). Defina para auto-hospedado. |
| `HERMES_LANGFUSE_ENV` | Tag de ambiente nos traces (`production`, `staging`, …) |
| `HERMES_LANGFUSE_RELEASE` | Tag de release/versão nos traces |
| `HERMES_LANGFUSE_SAMPLE_RATE` | Taxa de amostragem do SDK, de 0.0 a 1.0 (padrão: `1.0`) |
| `HERMES_LANGFUSE_MAX_CHARS` | Truncamento por campo para payloads serializados (padrão: `12000`) |
| `HERMES_LANGFUSE_DEBUG` | `true` ativa logging verboso do plugin em `agent.log` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | Nomes padrão do SDK Langfuse. Aceitos como fallback quando os equivalentes `HERMES_LANGFUSE_*` não estão definidos. |

### Nous Tool Gateway {#nous-tool-gateway}

Essas variáveis configuram o [Tool Gateway](/user-guide/features/tool-gateway) para assinantes pagos da Nous ou implantações de gateway auto-hospedadas. A maioria dos usuários não precisa definir isso — o gateway é configurado automaticamente via `hermes model` ou `hermes tools`.

| Variável | Descrição |
|----------|-------------|
| `TOOL_GATEWAY_DOMAIN` | Domínio base para roteamento do Tool Gateway (padrão: `nousresearch.com`) |
| `TOOL_GATEWAY_SCHEME` | Esquema HTTP ou HTTPS para URLs do gateway (padrão: `https`) |
| `TOOL_GATEWAY_USER_TOKEN` | Token de autenticação para o Tool Gateway (normalmente preenchido automaticamente pela autenticação Nous) |
| `FIRECRAWL_GATEWAY_URL` | Sobrescreve a URL especificamente para o endpoint do gateway Firecrawl |

## Backend de Terminal {#terminal-backend}

| Variável | Descrição |
|----------|-------------|
| `TERMINAL_ENV` | Backend: `local`, `docker`, `ssh`, `singularity`, `modal`, `daytona` |
| `HERMES_DOCKER_BINARY` | Sobrescreve o binário de container ao qual o Hermes delega comandos (ex.: `podman`, `/usr/local/bin/docker`). Quando indefinido, o Hermes descobre automaticamente `docker` ou `podman` no `PATH`. Necessário quando ambos estão instalados e você quer o não padrão, ou quando o binário está fora do `PATH`. |
| `TERMINAL_DOCKER_IMAGE` | Imagem Docker (padrão: `nikolaik/python-nodejs:python3.11-nodejs20`) |
| `TERMINAL_DOCKER_FORWARD_ENV` | Array JSON de nomes de variáveis de ambiente a repassar explicitamente para sessões de terminal Docker. Observação: `required_environment_variables` declaradas por skills são repassadas automaticamente — você só precisa disso para variáveis não declaradas por nenhuma skill. |
| `TERMINAL_DOCKER_VOLUMES` | Montagens de volume Docker adicionais (pares `host:container` separados por vírgula) |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | Opt-in avançado: monta o cwd de lançamento no `/workspace` do Docker (`true`/`false`, padrão: `false`) |
| `TERMINAL_SINGULARITY_IMAGE` | Imagem Singularity ou caminho `.sif` |
| `TERMINAL_MODAL_IMAGE` | Imagem de container Modal |
| `TERMINAL_DAYTONA_IMAGE` | Imagem de sandbox Daytona |
| `TERMINAL_TIMEOUT` | Timeout de comando em segundos |
| `TERMINAL_LIFETIME_SECONDS` | Tempo de vida máximo para sessões de terminal em segundos |
| `TERMINAL_CWD` | Sobrescrita direta obsoleta para sessões de terminal do gateway/cron. Prefira `terminal.cwd` no `config.yaml`; a CLI ainda usa o diretório de lançamento. |
| `SUDO_PASSWORD` | Ativa o sudo sem prompt interativo |

Para backends de sandbox em nuvem, a persistência é orientada ao sistema de arquivos. `TERMINAL_LIFETIME_SECONDS` controla quando o Hermes limpa uma sessão de terminal ociosa, e retomadas posteriores podem recriar o sandbox em vez de manter os mesmos processos ativos em execução.

## Backend SSH {#ssh-backend}

| Variável | Descrição |
|----------|-------------|
| `TERMINAL_SSH_HOST` | Hostname do servidor remoto |
| `TERMINAL_SSH_USER` | Nome de usuário SSH |
| `TERMINAL_SSH_PORT` | Porta SSH (padrão: 22) |
| `TERMINAL_SSH_KEY` | Caminho para a chave privada |
| `TERMINAL_SSH_PERSISTENT` | Sobrescreve o shell persistente para SSH (padrão: segue `TERMINAL_PERSISTENT_SHELL`) |

## Recursos de Container (Docker, Singularity, Modal, Daytona) {#container-resources-docker-singularity-modal-daytona}

| Variável | Descrição |
|----------|-------------|
| `TERMINAL_CONTAINER_CPU` | Núcleos de CPU (padrão: 1) |
| `TERMINAL_CONTAINER_MEMORY` | Memória em MB (padrão: 5120) |
| `TERMINAL_CONTAINER_DISK` | Disco em MB (padrão: 51200) |
| `TERMINAL_CONTAINER_PERSISTENT` | Persiste o sistema de arquivos do container entre sessões (padrão: `true`) |
| `TERMINAL_SANDBOX_DIR` | Diretório host para workspaces e overlays (padrão: `~/.hermes/sandboxes/`) |

## Shell Persistente {#persistent-shell}

| Variável | Descrição |
|----------|-------------|
| `TERMINAL_PERSISTENT_SHELL` | Ativa shell persistente para backends não locais (padrão: `true`). Também definível via `terminal.persistent_shell` no config.yaml |
| `TERMINAL_LOCAL_PERSISTENT` | Ativa shell persistente para o backend local (padrão: `false`) |
| `TERMINAL_SSH_PERSISTENT` | Sobrescreve o shell persistente para o backend SSH (padrão: segue `TERMINAL_PERSISTENT_SHELL`) |

## Mensagens {#messaging}

| Variável | Descrição |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Token do bot do Telegram (do @BotFather) |
| `TELEGRAM_ALLOWED_USERS` | IDs de usuário separados por vírgula autorizados a usar o bot (aplica-se a DMs, grupos e fóruns) |
| `TELEGRAM_ALLOW_ALL_USERS` | Permite que qualquer usuário do Telegram acione o bot (apenas dev). |
| `TELEGRAM_GROUP_ALLOWED_USERS` | IDs de usuário remetentes separados por vírgula autorizados apenas em grupos/fóruns (NÃO concede acesso por DM). Valores no formato de chat ID (começando com `-`) ainda são respeitados como chat IDs para compatibilidade retroativa com configs anteriores à #17686, com um aviso de descontinuação. |
| `TELEGRAM_GROUP_ALLOWED_CHATS` | IDs de chat de grupo/fórum separados por vírgula; qualquer membro é autorizado |
| `TELEGRAM_HOME_CHANNEL` | Chat/canal padrão do Telegram para entrega de cron |
| `TELEGRAM_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do Telegram |
| `TELEGRAM_CRON_THREAD_ID` | ID de tópico do fórum para receber entregas de cron; sobrescreve `TELEGRAM_HOME_CHANNEL_THREAD_ID` apenas para cron. Use no modo de tópico para que respostas a mensagens de cron abram uma nova sessão em vez de cair no lobby do sistema (#24409). |
| `TELEGRAM_WEBHOOK_URL` | URL HTTPS pública para modo webhook (ativa webhook em vez de polling) |
| `TELEGRAM_WEBHOOK_PORT` | Porta de escuta local para o servidor de webhook (padrão: `8443`) |
| `TELEGRAM_WEBHOOK_SECRET` | Token secreto que o Telegram ecoa de volta em cada update para verificação. **Obrigatório sempre que `TELEGRAM_WEBHOOK_URL` estiver definida** — o gateway se recusa a iniciar sem ela (GHSA-3vpc-7q5r-276h). Gere com `openssl rand -hex 32`. |
| `TELEGRAM_REACTIONS` | Ativa reações de emoji em mensagens durante o processamento (padrão: `false`) |
| `TELEGRAM_REQUIRE_MENTION` | Exige um gatilho explícito antes de responder em grupos do Telegram. Equivalente a `telegram.require_mention` no `config.yaml`. |
| `TELEGRAM_MENTION_PATTERNS` | Array JSON, lista separada por nova linha, ou lista separada por vírgula de padrões regex de palavra de ativação aceitos quando o filtro de menção de grupo do Telegram está ativado. Equivalente a `telegram.mention_patterns`. |
| `TELEGRAM_EXCLUSIVE_BOT_MENTIONS` | Quando ativado, menções explícitas a `@...bot` em grupos do Telegram roteiam apenas para os nomes de usuário de bot mencionados antes que fallbacks de resposta ou palavra de ativação sejam executados. Padrão: `true`. Equivalente a `telegram.exclusive_bot_mentions`. |
| `TELEGRAM_REPLY_TO_MODE` | Comportamento de referência de resposta: `off`, `first` (padrão), ou `all`. Segue o mesmo padrão do Discord. |
| `TELEGRAM_IGNORED_THREADS` | IDs de tópico/thread do fórum do Telegram separados por vírgula onde o bot nunca responde |
| `TELEGRAM_PROXY` | URL de proxy para conexões do Telegram — sobrescreve `HTTPS_PROXY`. Suporta `http://`, `https://`, `socks5://` |
| `DISCORD_BOT_TOKEN` | Token do bot do Discord |
| `DISCORD_ALLOWED_USERS` | IDs de usuário do Discord separados por vírgula autorizados a usar o bot |
| `DISCORD_ALLOW_ALL_USERS` | Permite que qualquer usuário do Discord acione o bot (apenas dev). |
| `DISCORD_ALLOWED_ROLES` | IDs de cargo do Discord separados por vírgula autorizados a usar o bot (OR com `DISCORD_ALLOWED_USERS`). Ativa automaticamente o intent de Members. Útil quando equipes de moderação têm rotatividade — concessões de cargo propagam automaticamente. |
| `DISCORD_ALLOWED_CHANNELS` | IDs de canal do Discord separados por vírgula. Quando definido, o bot só responde nesses canais (mais DMs se permitido). Sobrescreve `discord.allowed_channels` do `config.yaml`. |
| `DISCORD_PROXY` | URL de proxy para conexões do Discord — sobrescreve `HTTPS_PROXY`. Suporta `http://`, `https://`, `socks5://` |
| `DISCORD_HOME_CHANNEL` | Canal padrão do Discord para entrega de cron |
| `DISCORD_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do Discord |
| `DISCORD_COMMAND_SYNC_POLICY` | Política de sincronização de slash commands do Discord na inicialização: `safe` (diff e reconcilia), `bulk` (`tree.sync()` legado), ou `off` |
| `DISCORD_REQUIRE_MENTION` | Exige uma @menção antes de responder em canais de servidor |
| `DISCORD_FREE_RESPONSE_CHANNELS` | IDs de canal separados por vírgula onde a menção não é necessária |
| `DISCORD_AUTO_THREAD` | Cria automaticamente threads para respostas longas quando suportado |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | Quando `true`, aceita anexos de qualquer tipo de arquivo (não apenas a allowlist embutida de PDF/texto/zip/office). Tipos desconhecidos são cacheados e apresentados ao agente como um caminho local para que possa inspecioná-los via `terminal` / `read_file` / `ffprobe`. Padrão `false`. |
| `DISCORD_MAX_ATTACHMENT_BYTES` | Tamanho máximo em bytes por anexo que o gateway vai cachear. Padrão `33554432` (32 MiB). Defina como `0` para sem limite (anexos são mantidos em memória enquanto sendo escritos). |
| `DISCORD_REACTIONS` | Ativa reações de emoji em mensagens durante o processamento (padrão: `true`) |
| `DISCORD_IGNORED_CHANNELS` | IDs de canal separados por vírgula onde o bot nunca responde |
| `DISCORD_NO_THREAD_CHANNELS` | IDs de canal separados por vírgula onde o bot responde sem criar threads automaticamente |
| `DISCORD_REPLY_TO_MODE` | Comportamento de referência de resposta: `off`, `first` (padrão), ou `all` |
| `DISCORD_ALLOW_MENTION_EVERYONE` | Permite que o bot mencione `@everyone`/`@here` (padrão: `false`). Veja [Controle de Menções](../user-guide/messaging/discord.md#mention-control). |
| `DISCORD_ALLOW_MENTION_ROLES` | Permite que o bot mencione `@role` (padrão: `false`). |
| `DISCORD_ALLOW_MENTION_USERS` | Permite que o bot mencione usuários individuais `@user` (padrão: `true`). |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | Menciona o autor ao responder à mensagem dele (padrão: `true`). |
| `SLACK_BOT_TOKEN` | Token de bot do Slack (`xoxb-...`) |
| `SLACK_APP_TOKEN` | Token de nível de app do Slack (`xapp-...`, necessário para Socket Mode) |
| `SLACK_ALLOWED_USERS` | IDs de usuário do Slack separados por vírgula |
| `SLACK_ALLOW_ALL_USERS` | Permite que qualquer usuário do Slack acione o bot (apenas dev). |
| `SLACK_HOME_CHANNEL` | Canal padrão do Slack para entrega de cron |
| `SLACK_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do Slack |
| `GOOGLE_CHAT_PROJECT_ID` | Projeto GCP que hospeda o tópico Pub/Sub (recorre a `GOOGLE_CLOUD_PROJECT`) |
| `GOOGLE_CHAT_SUBSCRIPTION_NAME` | Caminho completo de assinatura Pub/Sub, `projects/{proj}/subscriptions/{sub}` (alias legado: `GOOGLE_CHAT_SUBSCRIPTION`) |
| `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` | Caminho para o JSON da Service Account, ou o JSON inline (recorre a `GOOGLE_APPLICATION_CREDENTIALS`) |
| `GOOGLE_CHAT_ALLOWED_USERS` | E-mails de usuário separados por vírgula autorizados a conversar com o bot |
| `GOOGLE_CHAT_ALLOW_ALL_USERS` | Permite que qualquer usuário do Google Chat acione o bot (apenas dev) |
| `GOOGLE_CHAT_HOME_CHANNEL` | Espaço padrão (ex.: `spaces/AAAA...`) para entrega de cron |
| `GOOGLE_CHAT_HOME_CHANNEL_NAME` | Nome de exibição para o espaço home do Google Chat |
| `GOOGLE_CHAT_MAX_MESSAGES` | Máximo de mensagens em trânsito do FlowControl do Pub/Sub (padrão: `1`) |
| `GOOGLE_CHAT_MAX_BYTES` | Máximo de bytes em trânsito do FlowControl do Pub/Sub (padrão: `16777216`, 16 MiB) |
| `GOOGLE_CHAT_BOOTSTRAP_SPACES` | IDs de espaço extras separados por vírgula para sondar na inicialização ao resolver o próprio `users/{id}` do bot |
| `GOOGLE_CHAT_DEBUG_RAW` | Defina para qualquer valor para logar envelopes Pub/Sub redigidos em nível DEBUG (apenas para depuração) |
| `WHATSAPP_ENABLED` | Ativa a bridge do WhatsApp (`true`/`false`) |
| `WHATSAPP_MODE` | `bot` (número separado) ou `self-chat` (mensagem para você mesmo) |
| `WHATSAPP_ALLOWED_USERS` | Números de telefone separados por vírgula (com código do país, sem `+`), ou `*` para permitir todos os remetentes |
| `WHATSAPP_ALLOW_ALL_USERS` | Permite todos os remetentes do WhatsApp sem allowlist (`true`/`false`) |
| `WHATSAPP_HOME_CHANNEL` | ID de chat padrão para entrega de cron / notificação. |
| `WHATSAPP_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do WhatsApp. |
| `WHATSAPP_DEBUG` | Loga eventos de mensagem brutos na bridge para solução de problemas (`true`/`false`) |
| `WHATSAPP_CLOUD_PHONE_NUMBER_ID` | ID de Número de Telefone da Meta da WhatsApp Business Cloud API (15–17 dígitos; **não** é o próprio número de telefone) |
| `WHATSAPP_CLOUD_ACCESS_TOKEN` | Token de acesso da Meta (começa com `EAA`); tokens temporários expiram em 24h, tokens de System User são permanentes |
| `WHATSAPP_CLOUD_APP_SECRET` | Segredo de app hex de 32 caracteres usado para verificar assinaturas de webhook de entrada |
| `WHATSAPP_CLOUD_VERIFY_TOKEN` | Segredo compartilhado para o handshake de verificação de webhook da Meta (gerado automaticamente pelo assistente de configuração) |
| `WHATSAPP_CLOUD_ALLOWED_USERS` | `wa_id`s separados por vírgula (números de telefone com código do país, sem `+`) autorizados a enviar mensagem ao bot |
| `WHATSAPP_CLOUD_ALLOW_ALL_USERS` | Permite todos os remetentes do WhatsApp Cloud sem allowlist (`true`/`false`) |
| `WHATSAPP_CLOUD_APP_ID` | ID de App da Meta opcional (para futura integração de analytics) |
| `WHATSAPP_CLOUD_WABA_ID` | ID de Conta WhatsApp Business opcional (para futura integração de analytics) |
| `WHATSAPP_CLOUD_WEBHOOK_HOST` | Interface à qual o servidor de webhook de entrada se vincula (padrão `0.0.0.0`) |
| `WHATSAPP_CLOUD_WEBHOOK_PORT` | Porta à qual o servidor de webhook de entrada se vincula (padrão `8090`) |
| `WHATSAPP_CLOUD_WEBHOOK_PATH` | Caminho de URL para onde a Meta envia mensagens de entrada (padrão `/whatsapp/webhook`) |
| `WHATSAPP_CLOUD_API_VERSION` | Versão da Meta Graph API a chamar (padrão `v20.0`) |
| `WHATSAPP_CLOUD_HOME_CHANNEL` | `wa_id` a usar como o canal home do bot (para jobs de cron etc.) |
| `WHATSAPP_CLOUD_DM_POLICY` | Filtro de DM para o adaptador Cloud (`open`/`allowlist`/`disabled`); recorre a `WHATSAPP_DM_POLICY` quando indefinido |
| `WHATSAPP_CLOUD_ALLOW_FROM` | Remetentes separados por vírgula permitidos quando `dm_policy: allowlist` (`wa_id`s simples; JIDs no estilo Baileys são normalizados) |
| `WHATSAPP_CLOUD_GROUP_POLICY` | Filtro de grupo para o adaptador Cloud (`open`/`allowlist`/`disabled`); recorre a `WHATSAPP_GROUP_POLICY` quando indefinido |
| `WHATSAPP_CLOUD_GROUP_ALLOW_FROM` | IDs de chat de grupo separados por vírgula permitidos quando `group_policy: allowlist` |
| `SIGNAL_HTTP_URL` | Endpoint HTTP do daemon signal-cli (por exemplo `http://127.0.0.1:8080`) |
| `SIGNAL_ACCOUNT` | Número de telefone do bot em formato E.164 |
| `SIGNAL_ALLOWED_USERS` | Números de telefone E.164 ou UUIDs separados por vírgula |
| `SIGNAL_GROUP_ALLOWED_USERS` | IDs de grupo separados por vírgula, ou `*` para todos os grupos |
| `SIGNAL_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do Signal |
| `SIGNAL_IGNORE_STORIES` | Ignora stories/atualizações de status do Signal |
| `SIGNAL_ALLOW_ALL_USERS` | Permite todos os usuários do Signal sem allowlist |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID (compartilhado com a skill telephony) |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token (compartilhado com a skill telephony; também usado para validação de assinatura de webhook) |
| `TWILIO_PHONE_NUMBER` | Número de telefone Twilio em formato E.164 (compartilhado com a skill telephony) |
| `SMS_WEBHOOK_URL` | URL pública para validação de assinatura Twilio — deve corresponder à URL de webhook no Twilio Console (obrigatória) |
| `SMS_WEBHOOK_PORT` | Porta de escuta de webhook para SMS de entrada (padrão: `8080`) |
| `SMS_WEBHOOK_HOST` | Endereço de vínculo do webhook (padrão: `127.0.0.1`) |
| `SMS_INSECURE_NO_SIGNATURE` | Defina como `true` para desativar a validação de assinatura Twilio (apenas para desenvolvimento local — não para produção) |
| `SMS_ALLOWED_USERS` | Números de telefone E.164 separados por vírgula autorizados a conversar |
| `SMS_ALLOW_ALL_USERS` | Permite todos os remetentes de SMS sem allowlist |
| `SMS_HOME_CHANNEL` | Número de telefone para entrega de job de cron / notificação |
| `SMS_HOME_CHANNEL_NAME` | Nome de exibição para o canal home de SMS |
| `EMAIL_ADDRESS` | Endereço de e-mail para o adaptador de gateway de Email |
| `EMAIL_PASSWORD` | Senha ou senha de app para a conta de e-mail |
| `EMAIL_IMAP_HOST` | Hostname IMAP para o adaptador de e-mail |
| `EMAIL_IMAP_PORT` | Porta IMAP |
| `EMAIL_SMTP_HOST` | Hostname SMTP para o adaptador de e-mail |
| `EMAIL_SMTP_PORT` | Porta SMTP |
| `EMAIL_ALLOWED_USERS` | Endereços de e-mail separados por vírgula autorizados a enviar mensagem ao bot |
| `EMAIL_HOME_ADDRESS` | Destinatário padrão para entrega proativa de e-mail |
| `EMAIL_HOME_ADDRESS_NAME` | Nome de exibição para o destino home de e-mail |
| `EMAIL_POLL_INTERVAL` | Intervalo de polling de e-mail em segundos |
| `EMAIL_ALLOW_ALL_USERS` | Permite todos os remetentes de e-mail de entrada |
| `DINGTALK_CLIENT_ID` | AppKey do bot DingTalk do portal de desenvolvedor ([open.dingtalk.com](https://open.dingtalk.com)) |
| `DINGTALK_CLIENT_SECRET` | AppSecret do bot DingTalk do portal de desenvolvedor |
| `DINGTALK_ALLOWED_USERS` | IDs de usuário DingTalk separados por vírgula autorizados a enviar mensagem ao bot |
| `DINGTALK_WEBHOOK_URL` | URL de webhook de robô estática para entrega entre plataformas / cron. |
| `DINGTALK_HOME_CHANNEL` | ID de conversa padrão para entrega de cron / notificação. |
| `DINGTALK_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do DingTalk. |
| `FEISHU_APP_ID` | App ID do bot Feishu/Lark do [open.feishu.cn](https://open.feishu.cn/) |
| `FEISHU_APP_SECRET` | App Secret do bot Feishu/Lark |
| `FEISHU_DOMAIN` | `feishu` (China) ou `lark` (internacional). Padrão: `feishu` |
| `FEISHU_CONNECTION_MODE` | `websocket` (recomendado) ou `webhook`. Padrão: `websocket` |
| `FEISHU_ENCRYPT_KEY` | Chave de criptografia opcional para o modo webhook |
| `FEISHU_VERIFICATION_TOKEN` | Token de verificação opcional para o modo webhook |
| `FEISHU_ALLOWED_USERS` | IDs de usuário Feishu separados por vírgula autorizados a enviar mensagem ao bot |
| `FEISHU_ALLOW_BOTS` | `none` (padrão) / `mentions` / `all` — aceita mensagens de entrada de outros bots. Veja [mensagens bot-a-bot](../user-guide/messaging/feishu.md#bot-to-bot-messaging) |
| `FEISHU_REQUIRE_MENTION` | `true` (padrão) / `false` — se mensagens de grupo precisam @mencionar o bot. Sobrescreva por chat via `group_rules.<chat_id>.require_mention`. |
| `FEISHU_HOME_CHANNEL` | ID de chat Feishu para entrega de cron e notificações |
| `FEISHU_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do Feishu. |
| `FEISHU_ALLOW_ALL_USERS` | Permite que qualquer usuário do Feishu acione o bot (apenas dev). |
| `WECOM_BOT_ID` | ID do WeCom AI Bot do console de admin |
| `WECOM_SECRET` | Segredo do WeCom AI Bot |
| `WECOM_WEBSOCKET_URL` | URL WebSocket customizada (padrão: `wss://openws.work.weixin.qq.com`) |
| `WECOM_ALLOWED_USERS` | IDs de usuário WeCom separados por vírgula autorizados a enviar mensagem ao bot |
| `WECOM_HOME_CHANNEL` | ID de chat WeCom para entrega de cron e notificações |
| `WECOM_CALLBACK_CORP_ID` | Corp ID da empresa WeCom para o app self-built de callback |
| `WECOM_CALLBACK_CORP_SECRET` | Segredo corporativo para o app self-built |
| `WECOM_CALLBACK_AGENT_ID` | ID de Agent do app self-built |
| `WECOM_CALLBACK_TOKEN` | Token de verificação de callback |
| `WECOM_CALLBACK_ENCODING_AES_KEY` | Chave AES para criptografia de callback |
| `WECOM_CALLBACK_HOST` | Endereço de vínculo do servidor de callback (padrão: `0.0.0.0`) |
| `WECOM_CALLBACK_PORT` | Porta do servidor de callback (padrão: `8645`) |
| `WECOM_CALLBACK_ALLOWED_USERS` | IDs de usuário separados por vírgula para allowlist |
| `WECOM_CALLBACK_ALLOW_ALL_USERS` | Defina `true` para permitir todos os usuários sem allowlist |
| `WEIXIN_ACCOUNT_ID` | ID de conta Weixin obtido via login QR pela API iLink Bot |
| `WEIXIN_TOKEN` | Token de autenticação Weixin obtido via login QR pela API iLink Bot |
| `WEIXIN_BASE_URL` | Sobrescreve a URL base da API iLink Bot do Weixin (padrão: `https://ilinkai.weixin.qq.com`) |
| `WEIXIN_CDN_BASE_URL` | Sobrescreve a URL base do CDN do Weixin para mídia (padrão: `https://novac2c.cdn.weixin.qq.com/c2c`) |
| `WEIXIN_DM_POLICY` | Política de mensagem direta: `open`, `allowlist`, `pairing`, `disabled` (padrão: `open`) |
| `WEIXIN_GROUP_POLICY` | Política de mensagem em grupo: `open`, `allowlist`, `disabled` (padrão: `disabled`) |
| `WEIXIN_ALLOWED_USERS` | IDs de usuário Weixin separados por vírgula autorizados a enviar DM ao bot |
| `WEIXIN_GROUP_ALLOWED_USERS` | **IDs de chat de grupo** Weixin separados por vírgula (não IDs de usuário membro) autorizados a interagir com o bot. O nome da variável é legado — ela espera IDs de grupo. Só tem efeito quando o iLink realmente entrega eventos de grupo; identidades de bot iLink por login QR (`...@im.bot`) normalmente não recebem mensagens de grupo comuns do WeChat. |
| `WEIXIN_HOME_CHANNEL` | ID de chat Weixin para entrega de cron e notificações |
| `WEIXIN_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do Weixin |
| `WEIXIN_ALLOW_ALL_USERS` | Permite todos os usuários do Weixin sem allowlist (`true`/`false`) |
| `BLUEBUBBLES_SERVER_URL` | URL do servidor BlueBubbles (ex.: `http://192.168.1.10:1234`) |
| `BLUEBUBBLES_PASSWORD` | Senha do servidor BlueBubbles |
| `BLUEBUBBLES_WEBHOOK_HOST` | Endereço de vínculo do ouvinte de webhook (padrão: `127.0.0.1`) |
| `BLUEBUBBLES_WEBHOOK_PORT` | Porta do ouvinte de webhook (padrão: `8645`) |
| `BLUEBUBBLES_HOME_CHANNEL` | Telefone/e-mail para entrega de cron/notificação |
| `BLUEBUBBLES_ALLOWED_USERS` | Usuários autorizados separados por vírgula |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | Permite todos os usuários (`true`/`false`) |
| `QQ_APP_ID` | App ID do QQ Bot do [q.qq.com](https://q.qq.com) |
| `QQ_CLIENT_SECRET` | App Secret do QQ Bot do [q.qq.com](https://q.qq.com) |
| `QQ_STT_API_KEY` | Chave de API para provedor de fallback STT externo (opcional, usado quando o ASR embutido do QQ não retorna texto) |
| `QQ_STT_BASE_URL` | URL base para o provedor STT externo (opcional) |
| `QQ_STT_MODEL` | Nome do modelo para o provedor STT externo (opcional) |
| `QQ_ALLOWED_USERS` | openIDs de usuário QQ separados por vírgula autorizados a enviar mensagem ao bot |
| `QQ_GROUP_ALLOWED_USERS` | IDs de grupo QQ separados por vírgula para acesso via @-mensagem em grupo |
| `QQ_ALLOW_ALL_USERS` | Permite todos os usuários (`true`/`false`, sobrescreve `QQ_ALLOWED_USERS`) |
| `QQBOT_HOME_CHANNEL` | openID de usuário/grupo QQ para entrega de cron e notificações |
| `QQBOT_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do QQ |
| `QQ_PORTAL_HOST` | Sobrescreve o host do portal QQ (defina como `sandbox.q.qq.com` para rotear pelo gateway sandbox; padrão: `q.qq.com`). |
| `QQ_SANDBOX` | Ativa o modo sandbox do QQ para testes de desenvolvimento (`true`/`false`) |
| `MATTERMOST_URL` | URL do servidor Mattermost (ex.: `https://mm.example.com`) |
| `MATTERMOST_TOKEN` | Token de bot ou token de acesso pessoal para o Mattermost |
| `MATTERMOST_ALLOWED_USERS` | IDs de usuário Mattermost separados por vírgula autorizados a enviar mensagem ao bot |
| `MATTERMOST_ALLOW_ALL_USERS` | Permite que qualquer usuário do Mattermost acione o bot (apenas dev). |
| `MATTERMOST_ALLOWED_CHANNELS` | Se definido, o bot só responde nesses canais (whitelist). |
| `MATTERMOST_HOME_CHANNEL` | ID de canal para entrega proativa de mensagem (cron, notificações) |
| `MATTERMOST_REQUIRE_MENTION` | Exige `@mention` em canais (padrão: `true`). Defina como `false` para responder a todas as mensagens. |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | IDs de canal separados por vírgula onde o bot responde sem `@mention` |
| `MATTERMOST_REPLY_MODE` | Estilo de resposta: `thread` (respostas em thread) ou `off` (mensagens simples, padrão) |
| `MATRIX_HOMESERVER` | URL do homeserver Matrix (ex.: `https://matrix.org`) |
| `MATRIX_ACCESS_TOKEN` | Token de acesso Matrix para autenticação do bot |
| `MATRIX_USER_ID` | ID de usuário Matrix (ex.: `@hermes:matrix.org`) — obrigatório para login por senha, opcional com token de acesso |
| `MATRIX_PASSWORD` | Senha Matrix (alternativa ao token de acesso) |
| `MATRIX_ALLOWED_USERS` | IDs de usuário Matrix separados por vírgula autorizados a enviar mensagem ao bot (ex.: `@alice:matrix.org`) |
| `MATRIX_ALLOW_ALL_USERS` | Permite que qualquer usuário do Matrix acione o bot (apenas dev). |
| `MATRIX_HOME_CHANNEL` | ID de sala padrão para entrega de cron / notificação. |
| `MATRIX_HOME_CHANNEL_NAME` | Nome de exibição para a sala home do Matrix. |
| `MATRIX_ALLOWED_ROOMS` | IDs de sala Matrix separados por vírgula autorizados a acionar respostas do bot |
| `MATRIX_HOME_ROOM` | ID de sala para entrega proativa de mensagem (ex.: `!abc123:matrix.org`) |
| `MATRIX_ENCRYPTION` | Ativa criptografia de ponta a ponta (`true`/`false`, padrão: `false`) |
| `MATRIX_E2EE_MODE` | Comportamento E2EE do Matrix: `off`, `optional`, ou `required`. Sobrescreve `MATRIX_ENCRYPTION` quando definido. |
| `MATRIX_DEVICE_ID` | ID de dispositivo Matrix estável para persistência E2EE entre reinicializações (ex.: `HERMES_BOT`). Sem isso, as chaves E2EE rotacionam a cada inicialização e a descriptografia de salas históricas quebra. |
| `MATRIX_REACTIONS` | Ativa reações de emoji de ciclo de vida de processamento em mensagens de entrada (padrão: `true`). Defina como `false` para desativar. |
| `MATRIX_REQUIRE_MENTION` | Exige `@mention` em salas (padrão: `true`). Defina como `false` para responder a todas as mensagens. |
| `MATRIX_FREE_RESPONSE_ROOMS` | IDs de sala separados por vírgula onde o bot responde sem `@mention` |
| `MATRIX_IGNORE_USER_PATTERNS` | Expressões regulares separadas por vírgula para IDs de usuário fantasma de bridge/appservice do Matrix a ignorar |
| `MATRIX_PROCESS_NOTICES` | Processa eventos `m.notice` de entrada do Matrix (padrão: `false`) |
| `MATRIX_SESSION_SCOPE` | Escopo de sessão Matrix para salas de projeto: `auto`, `room`, ou `thread` (padrão: `auto`) |
| `MATRIX_TOOLS_ALLOW_CROSS_ROOM` | Permite que ferramentas Matrix visem salas explícitas diferentes da sala atual (padrão: `false`) |
| `MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE` | Permite ferramentas Matrix de redação/convite entre salas; requer `MATRIX_TOOLS_ALLOW_CROSS_ROOM=true` (padrão: `false`) |
| `MATRIX_TOOLS_ALLOW_REDACTION` | Permite a execução da ferramenta de redação de mensagem Matrix (padrão: `false`) |
| `MATRIX_TOOLS_ALLOW_INVITES` | Permite a execução da ferramenta de convite Matrix (padrão: `false`) |
| `MATRIX_TOOLS_ALLOW_ROOM_CREATE` | Permite a execução da ferramenta de criação de sala Matrix (padrão: `false`) |
| `MATRIX_ALLOW_ROOM_MENTIONS` | Permite menções `@room` de saída para notificar todos os membros da sala (padrão: `false`) |
| `MATRIX_AUTO_THREAD` | Cria threads automaticamente para mensagens de sala (padrão: `true`) |
| `MATRIX_DM_AUTO_THREAD` | Cria threads automaticamente para mensagens de DM no Matrix (padrão: `false`) |
| `MATRIX_DM_MENTION_THREADS` | Cria uma thread quando o bot é `@mencionado` em uma DM (padrão: `false`) |
| `MATRIX_APPROVAL_REQUIRE_SENDER` | Exige que reações de aprovação/seletor de modelo venham do solicitante original quando conhecido (padrão: `true`) |
| `MATRIX_APPROVAL_TIMEOUT_SECONDS` | Timeout para prompts de aprovação/seletor de modelo por reação no Matrix (padrão: `300`) |
| `MATRIX_ALLOW_PUBLIC_ROOMS` | Permite que ferramentas de criação de sala Matrix criem salas públicas (padrão: `false`) |
| `MATRIX_MAX_MEDIA_BYTES` | Tamanho máximo de upload/download de mídia Matrix em bytes (padrão: `104857600`) |
| `MATRIX_RECOVERY_KEY` | Chave de recuperação para verificação cross-signing após rotação de chave de dispositivo. Recomendada para configurações E2EE com cross-signing ativado. |
| `MATRIX_RECOVERY_KEY_OUTPUT_FILE` | Caminho opcional de uso único para uma chave de recuperação Matrix gerada. Criado com modo `0600` e nunca sobrescrito. |
| `HASS_TOKEN` | Token de Acesso de Longa Duração do Home Assistant (ativa a plataforma HA + ferramentas) |
| `HASS_URL` | URL do Home Assistant (padrão: `http://homeassistant.local:8123`) |
| `WEBHOOK_ENABLED` | Ativa o adaptador de plataforma webhook (`true`/`false`) |
| `WEBHOOK_PORT` | Porta do servidor HTTP para receber webhooks (padrão: `8644`) |
| `WEBHOOK_SECRET` | Segredo HMAC global para validação de assinatura de webhook (usado como fallback quando as rotas não especificam o próprio) |
| `API_SERVER_ENABLED` | Ativa o servidor de API compatível com OpenAI (`true`/`false`). Roda junto com outras plataformas. |
| `API_SERVER_KEY` | Token Bearer para autenticação do servidor de API. Obrigatório sempre que o servidor de API estiver ativado. |
| `API_SERVER_CORS_ORIGINS` | Origens de navegador separadas por vírgula autorizadas a chamar o servidor de API diretamente (por exemplo `http://localhost:3000,http://127.0.0.1:3000`). Padrão: desativado. |
| `API_SERVER_PORT` | Porta para o servidor de API (padrão: `8642`) |
| `API_SERVER_HOST` | Endereço de host/vínculo para o servidor de API (padrão: `127.0.0.1`). `API_SERVER_KEY` ainda é necessária no loopback; use uma allowlist restrita de `API_SERVER_CORS_ORIGINS` para acesso por navegador. |
| `API_SERVER_MODEL_NAME` | Nome do modelo anunciado em `/v1/models`. Padrão é o nome do perfil (ou `hermes-agent` para o perfil padrão). Útil para configurações multiusuário onde frontends como o Open WebUI precisam de nomes de modelo distintos por conexão. |
| `GATEWAY_PROXY_URL` | URL de um servidor de API Hermes remoto para o qual encaminhar mensagens ([modo proxy](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos)). Quando definido, o gateway trata apenas o I/O de plataforma — todo o trabalho do agente é delegado ao servidor remoto. Também configurável via `gateway.proxy_url` no `config.yaml`. |
| `GATEWAY_PROXY_KEY` | Token Bearer para autenticação com o servidor de API remoto em modo proxy. Deve corresponder a `API_SERVER_KEY` no host remoto. |
| `MESSAGING_CWD` | Fallback de compatibilidade obsoleto para o diretório de trabalho do gateway. Prefira `terminal.cwd` no `config.yaml`. |
| `GATEWAY_ALLOWED_USERS` | IDs de usuário separados por vírgula autorizados em todas as plataformas |
| `GATEWAY_ALLOW_ALL_USERS` | Permite todos os usuários sem allowlists (`true`/`false`, padrão: `false`) |

### Dashboard Web e Hermes Desktop {#web-dashboard--hermes-desktop}

Autenticação para o [dashboard web](/user-guide/features/web-dashboard) e para conectar o [Hermes Desktop a um backend remoto](/user-guide/features/web-dashboard#connecting-hermes-desktop-to-a-remote-backend). Seguindo a convenção de segredos apenas, as credenciais pertencem a `~/.hermes/.env`; o `client_id` OAuth é melhor definido sob `dashboard.oauth` no `config.yaml` (o env prevalece quando definido).

Três provedores de autenticação de dashboard acompanham o pacote. Para uma conexão remota do Hermes Desktop ou qualquer dashboard exposto à internet, o provedor recomendado é **OAuth (Nous Portal)** — defina `HERMES_DASHBOARD_OAUTH_CLIENT_ID` (provisione com `hermes dashboard register`). O provedor incluído de **usuário/senha** (`HERMES_DASHBOARD_BASIC_AUTH_*`) é a opção mais rápida para um backend em uma LAN confiável ou detrás de uma VPN, mas não é adequado para exposição direta à internet pública. Para autenticar contra seu próprio provedor de identidade, use o provedor **OIDC auto-hospedado** (`HERMES_DASHBOARD_OIDC_*`). De qualquer forma, um vínculo não loopback (`hermes dashboard --host 0.0.0.0`) ativa o gate de autenticação. Veja [Dashboard Web → Autenticação](/user-guide/features/web-dashboard#authentication-gated-mode) para o quadro completo.

| Variável | Descrição |
|----------|-------------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | Nome de usuário para o provedor de autenticação de dashboard usuário/senha incluído (`plugins/dashboard_auth/basic`). Ativa o provedor quando definido junto com uma senha. Sobrescreve `dashboard.basic_auth.username`. |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | Senha em texto puro para o provedor basic (hasheada em memória no carregamento). Prevalece sobre um `password_hash` de config para que você possa rotacionar via env. Sobrescreve `dashboard.basic_auth.password`. |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | Hash de senha scrypt para o provedor basic (preferido — sem texto puro em repouso). Calcule com `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`. Sobrescreve `dashboard.basic_auth.password_hash`. |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | Chave HMAC (32+ bytes, base64/hex/raw) assinando os tokens de sessão sem estado do provedor basic. Defina explicitamente para que as sessões sobrevivam a reinicializações / abranjam vários workers; vazio → aleatório por processo (você será deslogado a cada reinicialização). Sobrescreve `dashboard.basic_auth.secret`. |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | Tempo de vida do token de acesso para o provedor basic (padrão 12h). Sobrescreve `dashboard.basic_auth.session_ttl_seconds`. |
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | ID de cliente OAuth (`agent:{instance_id}`) para o dashboard restrito/público, ativando o provedor Nous (`plugins/dashboard_auth/nous`). Sobrescreve `dashboard.oauth.client_id`. Provisione com `hermes dashboard register`. |
| `HERMES_DASHBOARD_PUBLIC_URL` | URL pública completa em que o dashboard é acessado, para construção do callback OAuth atrás de reverse proxies. Sobrescreve `dashboard.public_url`. |
| `HERMES_DASHBOARD_OIDC_ISSUER` | URL do issuer OIDC para o provedor OIDC auto-hospedado incluído (`plugins/dashboard_auth/self_hosted`). Obrigatória para ativá-lo. Sobrescreve `dashboard.oauth.self_hosted.issuer`. |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | ID de cliente OIDC público (authorization-code + PKCE) para o provedor OIDC auto-hospedado. Obrigatório para ativá-lo. Sobrescreve `dashboard.oauth.self_hosted.client_id`. |
| `HERMES_DASHBOARD_OIDC_SCOPES` | Scopes OIDC solicitados para o provedor OIDC auto-hospedado (padrão `openid profile email`). Sobrescreve `dashboard.oauth.self_hosted.scopes`. |
| `HERMES_DESKTOP_REMOTE_URL` | (Lado do Desktop) URL base do backend remoto, ex.: `http://host:9119`. Quando definida, sobrescreve a URL de Gateway no app; você ainda faz login pelo painel de configurações de Gateway (redirecionamento OAuth ou usuário/senha, o que o backend anunciar). |
| `HERMES_DESKTOP_HERMES` | Sobrescrita de comando do backend Desktop. Usada por empacotadores/Nix ou solução de problemas para apontar o Electron a um executável `hermes` específico após a sondagem de backend. |
| `HERMES_DESKTOP_HERMES_ROOT` | Sobrescrita de checkout de código-fonte do Desktop usada por `hermes desktop --hermes-root`; verificada antes da instalação empacotada de primeiro lançamento ou de um `hermes` existente no `PATH`. |
| `HERMES_DESKTOP_IGNORE_EXISTING` | Defina como `1` para fazer o Desktop ignorar um `hermes` existente no `PATH` durante a resolução de backend. Equivalente a `hermes desktop --ignore-existing`. |
| `HERMES_DESKTOP_CWD` | Diretório de projeto inicial para sessões de chat do Desktop. Definido por `hermes desktop --cwd`. |
| `HERMES_DESKTOP_PYTHON` | Caminho absoluto para um interpretador Python para o backend, verificado antes que o Electron resolva automaticamente um para o checkout de código-fonte. Usado por auxiliares de desenvolvimento de worktree (veja [TUI e Desktop a partir de Worktrees](../developer-guide/worktree-ui-dev.md)) para reutilizar um venv compartilhado. |
| `HERMES_DESKTOP_DEV_SERVER` | URL do servidor de desenvolvimento Vite que o shell Electron carrega em vez do bundle empacotado (ex.: `http://127.0.0.1:5174`). Definido automaticamente por `npm run dev`; relevante apenas ao modificar o app. |

### Microsoft Graph (Reuniões do Teams) {#microsoft-graph-teams-meetings}

Credenciais somente-app para o cliente REST Microsoft Graph usado pelo próximo pipeline de resumo de reuniões do Teams. Veja [Registrar uma aplicação Microsoft Graph](/guides/microsoft-graph-app-registration) para o passo a passo no portal Azure e as permissões de API exatas necessárias.

| Variável | Descrição |
|----------|-------------|
| `MSGRAPH_TENANT_ID` | ID de tenant do Azure AD (GUID de diretório) para o registro do app Graph. |
| `MSGRAPH_CLIENT_ID` | ID de aplicação (cliente) do registro do app Azure. |
| `MSGRAPH_CLIENT_SECRET` | Valor do segredo de cliente para o registro do app. Armazene em `~/.hermes/.env` com `chmod 600`; rotacione periodicamente via o portal Azure. |
| `MSGRAPH_SCOPE` | Scope OAuth2 para a requisição de token client-credentials (padrão: `https://graph.microsoft.com/.default`). |
| `MSGRAPH_AUTHORITY_URL` | Authority da plataforma de identidade Microsoft (padrão: `https://login.microsoftonline.com`). Sobrescreva apenas para nuvens nacionais/soberanas (ex.: `https://login.microsoftonline.us` para GCC High). |

### Ouvinte de Webhook do Microsoft Graph {#microsoft-graph-webhook-listener}

Ouvinte de notificação de mudança de entrada para eventos Graph (reuniões do Teams, calendário, chat, etc.). Veja [Ouvinte de Webhook do Microsoft Graph](/user-guide/messaging/msgraph-webhook) para configuração e hardening de segurança.

| Variável | Descrição |
|----------|-------------|
| `MSGRAPH_WEBHOOK_ENABLED` | Ativa a plataforma de gateway `msgraph_webhook` (`true`/`1`/`yes`). |
| `MSGRAPH_WEBHOOK_PORT` | Porta à qual o ouvinte se vincula (padrão: `8646`). |
| `MSGRAPH_WEBHOOK_CLIENT_STATE` | Segredo compartilhado que o Graph ecoa em cada notificação; comparado com `hmac.compare_digest`. Gere com `openssl rand -hex 32`. |
| `MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES` | Allowlist separada por vírgula de caminhos/padrões de recurso Graph (ex.: `communications/onlineMeetings,chats/*/messages`). `*` no final funciona como correspondência de prefixo. Vazio = aceita todos. |
| `MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS` | Faixas CIDR separadas por vírgula autorizadas a fazer POST no ouvinte (ex.: `52.96.0.0/14,52.104.0.0/14`). Vazio = permite todas (padrão). Restrinja às faixas de saída publicadas do Microsoft Graph em produção. |

### Entrega de Resumo de Reunião do Teams {#teams-meeting-summary-delivery}

Usado apenas quando o [plugin `teams_pipeline`](/user-guide/messaging/msgraph-webhook) está ativado. As configurações também são configuráveis sob `platforms.teams.extra` no `config.yaml` — variáveis de ambiente têm prioridade quando ambas estão definidas. Veja [Microsoft Teams → Entrega de Resumo de Reunião](/user-guide/messaging/teams#meeting-summary-delivery-teams-meeting-pipeline).

| Variável | Descrição |
|----------|-------------|
| `TEAMS_DELIVERY_MODE` | `graph` ou `incoming_webhook`. |
| `TEAMS_INCOMING_WEBHOOK_URL` | URL de webhook gerada pelo Teams; obrigatória quando `TEAMS_DELIVERY_MODE=incoming_webhook`. |
| `TEAMS_GRAPH_ACCESS_TOKEN` | Token de acesso delegado pré-adquirido para entrega via Graph. Raramente necessário — o writer recorre às credenciais de app `MSGRAPH_*` quando indefinido. |
| `TEAMS_TEAM_ID` | ID de Team alvo para entrega em canal (modo `graph`). |
| `TEAMS_CHANNEL_ID` | ID de canal alvo (pareado com `TEAMS_TEAM_ID`). |
| `TEAMS_CHAT_ID` | ID de chat 1:1 ou de grupo alvo (alternativa a team+channel para o modo `graph`). |

### LINE Messaging API {#line-messaging-api}

Usado pelo plugin de plataforma LINE incluído (`plugins/platforms/line/`). Veja [Gateway de Mensagens → LINE](/user-guide/messaging/line) para a configuração completa.

| Variável | Descrição |
|----------|-------------|
| `LINE_CHANNEL_ACCESS_TOKEN` | Token de acesso de canal de longa duração do LINE Developers Console (aba Messaging API). Obrigatório. |
| `LINE_CHANNEL_SECRET` | Segredo de canal (aba Basic settings); usado para verificação de assinatura de webhook HMAC-SHA256. Obrigatório. |
| `LINE_HOST` | Host de vínculo do webhook (padrão: `0.0.0.0`). |
| `LINE_PORT` | Porta de vínculo do webhook (padrão: `8646`). |
| `LINE_PUBLIC_URL` | URL base HTTPS pública (ex.: `https://my-tunnel.example.com`). Obrigatória para envios de imagem / áudio / vídeo — o LINE só aceita URLs acessíveis via HTTPS. |
| `LINE_ALLOWED_USERS` | IDs de usuário separados por vírgula autorizados a enviar DM ao bot (prefixo `U`). |
| `LINE_ALLOWED_GROUPS` | IDs de grupo separados por vírgula nos quais o bot responderá (prefixo `C`). |
| `LINE_ALLOWED_ROOMS` | IDs de sala separados por vírgula nos quais o bot responderá (prefixo `R`). |
| `LINE_ALLOW_ALL_USERS` | Escape hatch apenas para dev — aceita qualquer origem. Padrão: `false`. |
| `LINE_HOME_CHANNEL` | Destino de entrega padrão para jobs de cron com `deliver: line`. |
| `LINE_SLOW_RESPONSE_THRESHOLD` | Segundos antes que o postback de Template Buttons de LLM lento seja disparado (padrão: `45`). Defina `0` para desativar e sempre fazer fallback via Push. |
| `LINE_PENDING_TEXT` | Texto de bolha mostrado junto ao botão de postback. |
| `LINE_BUTTON_LABEL` | Rótulo do botão de postback (padrão: `Get answer`). |
| `LINE_DELIVERED_TEXT` | Resposta quando um postback já entregue é tocado novamente (padrão: `Already replied ✅`). |
| `LINE_INTERRUPTED_TEXT` | Resposta quando um botão de postback órfão por `/stop` é tocado (padrão: `Run was interrupted before completion.`). |

### ntfy (notificações push) {#ntfy-push-notifications}

[ntfy](https://ntfy.sh/) é um serviço leve de notificação push baseado em HTTP. Inscreva-se em um tópico pelo [app móvel do ntfy](https://ntfy.sh/docs/subscribe/phone/), publique nesse tópico para conversar com o agente.

| Variável | Descrição |
|----------|-------------|
| `NTFY_TOPIC` | Tópico ao qual se inscrever (mensagens de entrada). Obrigatório. |
| `NTFY_SERVER_URL` | URL do servidor (padrão: `https://ntfy.sh`). Aponte para um ntfy auto-hospedado para privacidade. |
| `NTFY_TOKEN` | Token de autenticação opcional. Token Bearer (ex.: `tk_xyz`) ou `user:pass` para Basic auth. |
| `NTFY_PUBLISH_TOPIC` | Tópico para respostas de saída (padrão `NTFY_TOPIC`). |
| `NTFY_MARKDOWN` | Defina `true` para enviar respostas com o header `X-Markdown: true`. Padrão: `false`. |
| `NTFY_ALLOWED_USERS` | Allowlist (tratada como IDs de usuário; no ntfy são nomes de tópico). Normalmente definida com o mesmo valor de `NTFY_TOPIC`. |
| `NTFY_ALLOW_ALL_USERS` | Escape hatch apenas para dev — seguro apenas em tópicos privados com acesso controlado. Padrão: `false`. |
| `NTFY_HOME_CHANNEL` | Destino de entrega padrão para jobs de cron com `deliver: ntfy`. |
| `NTFY_HOME_CHANNEL_NAME` | Rótulo humano para o canal home (padrão é o nome do tópico). |

Veja [o guia de mensagens ntfy](/user-guide/messaging/ntfy) — particularmente a seção do **modelo de identidade** — antes de implantar com tópicos não confiáveis.

### IRC {#irc}

Conecte o Hermes a um servidor IRC. Sem dependências externas. Veja [o guia de mensagens IRC](/user-guide/messaging/irc).

| Variável | Descrição |
|----------|-------------|
| `IRC_SERVER` | Hostname do servidor IRC (ex.: `irc.libera.chat`). Obrigatório. |
| `IRC_CHANNEL` | Canal(is) para entrar (ex.: `#hermes`); separe por vírgula para múltiplos. Obrigatório. |
| `IRC_NICKNAME` | Nickname do bot (padrão: `hermes-bot`). Obrigatório. |
| `IRC_PORT` | Porta do servidor (padrão: `6697` com TLS, `6667` sem). |
| `IRC_USE_TLS` | Usa TLS (`true`/`false`; padrão `true` na porta 6697). |
| `IRC_SERVER_PASSWORD` | Senha do servidor para o comando `PASS` (opcional). |
| `IRC_NICKSERV_PASSWORD` | Senha NickServ para IDENTIFY automático ao conectar (opcional). |
| `IRC_ALLOWED_USERS` | Nicks separados por vírgula autorizados a falar com o bot. |
| `IRC_ALLOW_ALL_USERS` | Permite que qualquer pessoa no canal fale com o bot (apenas dev). |
| `IRC_HOME_CHANNEL` | Canal para entrega de cron / notificação (padrão `IRC_CHANNEL`). |

### SimpleX {#simplex}

Conecte o Hermes a uma rede [SimpleX Chat](https://simplex.chat/) via um daemon `simplex-chat` local. Veja [o guia de mensagens SimpleX](/user-guide/messaging/simplex).

| Variável | Descrição |
|----------|-------------|
| `SIMPLEX_WS_URL` | URL WebSocket do daemon simplex-chat (ex.: `ws://127.0.0.1:5225`). |
| `SIMPLEX_ALLOWED_USERS` | IDs de contato SimpleX separados por vírgula autorizados a falar com o bot. |
| `SIMPLEX_ALLOW_ALL_USERS` | Permite que qualquer contato fale com o bot (apenas dev — desativa a allowlist). |
| `SIMPLEX_AUTO_ACCEPT` | Aceita automaticamente solicitações de contato de entrada (padrão: `true`). |
| `SIMPLEX_GROUP_ALLOWED` | IDs de grupo SimpleX separados por vírgula nos quais o bot deve participar, ou `*` para permitir qualquer grupo. Omita para ignorar mensagens de grupo completamente (padrão mais seguro — um bot em um grupo, caso contrário, processa o tráfego de todos os membros). |
| `SIMPLEX_HOME_CHANNEL` | ID de contato/grupo padrão para entrega de cron / notificação. |
| `SIMPLEX_HOME_CHANNEL_NAME` | Rótulo humano para o canal home (padrão é o ID). |

### Photon {#photon}

Conecte o Hermes ao [Photon](https://photon.codes/) / Spectrum (iMessage e outras plataformas Spectrum) via o sidecar Node. Veja [o guia de mensagens Photon](/user-guide/messaging/photon).

| Variável | Descrição |
|----------|-------------|
| `PHOTON_PROJECT_ID` | ID de projeto Spectrum (o `spectrumProjectId` do projeto; definido por `hermes photon setup`). |
| `PHOTON_PROJECT_SECRET` | Segredo de projeto pareado com o ID de projeto Spectrum (definido por `hermes photon setup`). |
| `PHOTON_ALLOWED_USERS` | Números de telefone E.164 separados por vírgula autorizados a falar com o bot. |
| `PHOTON_ALLOW_ALL_USERS` | Permite que qualquer remetente acione o bot (apenas dev — desativa a allowlist). |
| `PHOTON_REQUIRE_MENTION` | Ignora mensagens de chat de grupo a menos que correspondam a uma palavra de ativação de menção (`true`/`false`, padrão `false`). |
| `PHOTON_MENTION_PATTERNS` | Regexes de palavra de ativação de menção para chats de grupo (lista JSON ou separada por vírgula/nova linha; padrão são as palavras de ativação do Hermes). |
| `PHOTON_HOME_CHANNEL` | Destino Photon padrão para entrega de cron / notificação: ID de espaço Spectrum, GUID de DM, ou número de telefone E.164 simples. |
| `PHOTON_HOME_CHANNEL_NAME` | Rótulo humano para o canal home. |
| `PHOTON_MARKDOWN` | Envia respostas do agente como markdown — o iMessage renderiza nativamente, outras plataformas Spectrum degradam para texto simples (`true`/`false`, padrão `true`). |
| `PHOTON_REACTIONS` | Tapback 👀/👍/👎 em mensagens como status de processamento e roteia tapbacks em mensagens do bot para o agente (`true`/`false`, padrão `false`). |
| `PHOTON_TELEMETRY` | Ativa a telemetria do SDK Spectrum no sidecar (`true`/`false`, padrão `false`; alterne com `hermes photon telemetry on|off`). |
| `PHOTON_SIDECAR_PORT` | Porta de loopback para o controle do sidecar Node + canal de entrada (padrão `8789`). |
| `PHOTON_SIDECAR_AUTOSTART` | Inicia o sidecar Node ao conectar (`true`/`false`, padrão `true`). |
| `PHOTON_NODE_BIN` | Caminho para o binário node (padrão: `shutil.which('node')`). |
| `PHOTON_DASHBOARD_HOST` | Host da API do Photon Dashboard (padrão `https://app.photon.codes`). |
| `PHOTON_SPECTRUM_HOST` | Host da API do Photon Spectrum (padrão `https://spectrum.photon.codes`). |

### Microsoft Teams (adaptador) {#microsoft-teams-adapter}

O adaptador de plataforma do Microsoft Teams (Bot Framework / Azure AD), distinto da integração [Microsoft Graph (Reuniões do Teams)](#microsoft-graph-teams-meetings) acima. Veja [o guia de mensagens do Teams](/user-guide/messaging/teams).

| Variável | Descrição |
|----------|-------------|
| `TEAMS_CLIENT_ID` | ID de cliente da aplicação Azure AD (Bot Framework). |
| `TEAMS_CLIENT_SECRET` | Segredo de cliente da aplicação Azure AD. |
| `TEAMS_TENANT_ID` | ID de tenant Azure AD que hospeda a aplicação do bot. |
| `TEAMS_PORT` | Porta de escuta do webhook (padrão do Bot Framework: `3978`). |
| `TEAMS_ALLOWED_USERS` | IDs de usuário / UPNs do Teams separados por vírgula autorizados a falar com o bot. |
| `TEAMS_ALLOW_ALL_USERS` | Permite que qualquer usuário do Teams acione o bot (apenas dev). |
| `TEAMS_HOME_CHANNEL` | ID de chat/canal padrão para entrega de cron / notificação. |
| `TEAMS_HOME_CHANNEL_NAME` | Nome de exibição para o canal home do Teams. |

### Raft {#raft}

| Variável | Descrição |
|----------|-------------|
| `RAFT_PROFILE` | Slug de perfil de agente Raft — ativa automaticamente o adaptador quando definido. |

### Ajuste Avançado de Mensagens {#advanced-messaging-tuning}

Configurações avançadas por plataforma para regular o batcher de mensagens de saída. A maioria dos usuários nunca precisa tocar nisso; os padrões são definidos para respeitar os limites de taxa de cada plataforma sem parecer lento.

| Variável | Descrição |
|----------|-------------|
| `HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS` | Janela de tolerância antes de descarregar um chunk de texto do Telegram enfileirado (padrão: `0.6`). |
| `HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Atraso entre chunks divididos quando uma única mensagem do Telegram excede o limite de tamanho (padrão: `2.0`). |
| `HERMES_SIMPLEX_TEXT_BATCH_DELAY` | Segundos de período de silêncio (padrão: `0.8`) usados para concatenar mensagens de texto de entrada rápidas em um único MessageEvent — mesmo padrão do agrupamento de texto do Telegram. |
| `HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS` | Janela de tolerância antes de descarregar mídia enfileirada do Telegram (padrão: `0.6`). |
| `HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS` | Atraso antes de enviar um follow-up após o agente terminar, para evitar competir com o último chunk de stream. |
| `HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT` / `_READ_TIMEOUT` / `_WRITE_TIMEOUT` / `_POOL_TIMEOUT` | Sobrescreve os timeouts HTTP subjacentes do `python-telegram-bot` (segundos). |
| `HERMES_TELEGRAM_INIT_TIMEOUT` | Limite por tentativa (segundos) na cadeia de conexão do `initialize()` do Telegram durante a inicialização do gateway, para que uma cadeia de IP de fallback inacessível não bloqueie a inicialização indefinidamente (padrão: `30`). |
| `HERMES_TELEGRAM_HTTP_POOL_SIZE` | Máximo de conexões HTTP simultâneas com a API do Telegram. |
| `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS` | Desativa os IPs de fallback do Cloudflare fixos usados quando o DNS falha (`true`/`false`). |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | Janela de tolerância antes de descarregar um chunk de texto do Discord enfileirado (padrão: `0.6`). |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Atraso entre chunks divididos quando uma mensagem do Discord excede o limite de tamanho (padrão: `2.0`). |
| `HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS` | Sobrescrita de compatibilidade/manual para `discord.websocket_liveness_interval_seconds`. Intervalo para amostrar o WebSocket ativo do Discord Gateway (padrão: `15`; defina como `0` para desativar). Prefira a chave do `config.yaml`. |
| `HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD` | Sobrescrita de compatibilidade/manual para `discord.websocket_liveness_failure_threshold`. Amostras consecutivas de WebSocket não saudáveis antes de forçar uma reconexão (padrão: `2`). Prefira a chave do `config.yaml`. |
| `HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | Equivalentes Matrix das configurações de batch do Telegram. |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` / `_MAX_CHARS` / `_MAX_MESSAGES` | Ajuste do batcher do Feishu — atraso, atraso de divisão, máximo de caracteres por mensagem, máximo de mensagens por batch. |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | Atraso de descarga de mídia do Feishu. |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | Tamanho do cache de deduplicação de webhook do Feishu (padrão: `1024`). |
| `HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS` / `_SPLIT_DELAY_SECONDS` | Ajuste do batcher do WeCom. |
| `HERMES_VISION_DOWNLOAD_TIMEOUT` | Timeout em segundos para baixar uma imagem antes de passá-la para modelos de visão (padrão: `30`). |
| `HERMES_VISION_MAX_CONCURRENCY` | Máximo de rajadas de **encode/resize** de imagem simultâneas em todo o processo (sobrescrita para `auxiliary.vision.max_concurrency`; padrão: contagem de núcleos de CPU do host, sem limite). Limita apenas a etapa de encode ligada à CPU para que um fan-out de frames de vídeo não sature todos os núcleos e prive o event loop — as chamadas de LLM permanecem totalmente concorrentes. Valores `< 1` são ignorados. |
| `HERMES_RESTART_DRAIN_TIMEOUT` | Gateway: segundos para esperar as execuções ativas drenarem em `/restart` antes de forçar a reinicialização (padrão: `900`). |
| `HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT` | Timeout de conexão por plataforma durante a inicialização e reconexão do gateway (segundos; `0`/negativo espera indefinidamente). Aplica-se à tentativa de conexão *e* à espera de pronto do adaptador do Discord, para que contas com muitos slash commands para sincronizar não sejam mortas no meio da inicialização. Vinculado a `gateway.platform_connect_timeout` no `config.yaml` (padrão `30`); esta variável de ambiente é a sobrescrita manual e prevalece se definida explicitamente. |
| `HERMES_GATEWAY_BUSY_INPUT_MODE` | Comportamento padrão de entrada ocupada do gateway: `queue`, `steer`, ou `interrupt`. Pode ser sobrescrito por chat com `/busy`. |
| `HERMES_GATEWAY_BUSY_ACK_ENABLED` | Se o gateway envia uma mensagem de confirmação (⚡/⏳/⏩) quando um usuário envia entrada enquanto o agente está ocupado (padrão: `true`). Defina como `false` para suprimir completamente essas mensagens — a entrada ainda é enfileirada/direcionada/interrompe normalmente, apenas a resposta no chat é silenciada. Vinculado a `display.busy_ack_enabled` no `config.yaml`. |
| `HERMES_GATEWAY_NO_SUPERVISE` | Dentro da imagem Docker s6-overlay, opta por sair da auto-supervisão ao executar `hermes gateway run` e usa a semântica de primeiro plano pré-s6 (sem auto-reinicialização, o gateway é o processo principal do container). Valores verdadeiros: `1`, `true`, `yes`. Equivalente à flag CLI `--no-supervise`. Sem efeito fora da imagem s6. |
| `HERMES_GATEWAY_BOOTSTRAP_STATE` | Dentro da imagem Docker s6-overlay, declara o estado supervisionado **inicial** do gateway em um volume novo. Em um volume vazio não há `gateway_state.json` persistido, então o reconciliador de boot registra o slot `gateway-default`, mas o deixa **parado** (só inicia automaticamente quando o último estado registrado era `running`). Defina isso como `running` e o hook de configuração de primeiro boot preenche `gateway_state.json` *antes* de o reconciliador rodar, para que o gateway suba já no primeiro boot. Apenas o valor literal `running` é respeitado. Apenas primeiro-boot: um `gateway_state.json` existente nunca é sobrescrito, então um gateway deliberadamente parado permanece parado entre reinicializações. Sem efeito fora da imagem s6. |
| `GATEWAY_RELAY_URL` | URL base WebSocket do conector de relay experimental. Quando definida, o gateway registra o adaptador genérico `relay` e disca o conector de saída. Espelha `gateway.relay_url` no `config.yaml`. |
| `GATEWAY_RELAY_ID` | Identificador de gateway de relay atribuído por `hermes gateway enroll` ou auto-provisionamento gerenciado. Espelha `gateway.relay_id`. |
| `GATEWAY_RELAY_SECRET` | Segredo de relay por gateway usado para autenticar o WebSocket. Se já estiver configurado, o auto-provisionamento gerenciado é pulado. Espelha `gateway.relay_secret`. |
| `GATEWAY_RELAY_DELIVERY_KEY` | Chave de entrega emitida pelo conector, mantida para compatibilidade de autenticação de relay/passthrough. Mensagens de relay de entrada atuais chegam pelo WebSocket de saída em vez de um receptor HTTP do lado do gateway. |
| `GATEWAY_RELAY_ENROLL_TOKEN` | Token de inscrição consumido por `hermes gateway enroll` quando `--token` não é passado explicitamente. |
| `GATEWAY_RELAY_PLATFORM` | Nome de plataforma opcional anunciado no descritor de capacidade do relay. |
| `GATEWAY_RELAY_BOT_ID` | Identificador de bot opcional anunciado no descritor de capacidade do relay. |
| `GATEWAY_RELAY_ENDPOINT` | Endpoint de gateway opcional anunciado para modos de conector que precisam de uma URL de callback/passthrough; não necessário para o caminho padrão de relay de entrada apenas via WS. Espelha `gateway.relay_endpoint`. |
| `GATEWAY_RELAY_ROUTE_KEYS` | Chaves de rota de relay separadas por vírgula anunciadas ao conector. Espelha `gateway.relay_route_keys`. |
| `HERMES_FILE_MUTATION_VERIFIER` | Ativa o rodapé verificador de mutação de arquivo por turno (padrão: `true`). Quando ativado, o Hermes acrescenta um aviso listando quaisquer chamadas `write_file` / `patch` que falharam durante o turno e não foram substituídas por uma escrita bem-sucedida. Defina como `0`, `false`, `no`, ou `off` para suprimir. Espelha `display.file_mutation_verifier` no `config.yaml`; a variável de ambiente prevalece quando definida. |
| `HERMES_CRON_TIMEOUT` | Timeout de inatividade para execuções de agente de job de cron em segundos (padrão: `600`). O agente pode rodar indefinidamente enquanto chama ferramentas ativamente ou recebe tokens de stream — isso só dispara quando ocioso. Defina como `0` para ilimitado. |
| `HERMES_CRON_SCRIPT_TIMEOUT` | Timeout para scripts de pré-execução anexados a jobs de cron em segundos (padrão: `3600`). Limita apenas o script — jobs de skill/agente usam o orçamento de inatividade separado `HERMES_CRON_TIMEOUT`. Também configurável via `cron.script_timeout_seconds` no `config.yaml`. |
| `HERMES_CRON_MAX_PARALLEL` | Máximo de jobs de cron rodando em paralelo por tick (padrão: `4`). |

## Comportamento do Agente {#agent-behavior}

| Variável | Descrição |
|----------|-------------|
| `HERMES_MAX_ITERATIONS` | Máximo de iterações de chamada de ferramenta por conversa (padrão: 90) |
| `HERMES_INFERENCE_MODEL` | Sobrescreve o nome do modelo em nível de processo (tem prioridade sobre `config.yaml` para a sessão). Também definível via a flag `-m`/`--model`. |
| `HERMES_YOLO_MODE` | Defina como `1` para pular prompts de aprovação de comando perigoso. Equivalente a `--yolo`. |
| `HERMES_ACCEPT_HOOKS` | Aprova automaticamente quaisquer hooks de shell não vistos declarados no `config.yaml` sem um prompt TTY. Equivalente a `--accept-hooks` ou `hooks_auto_accept: true`. |
| `HERMES_IGNORE_USER_CONFIG` | Pula `~/.hermes/config.yaml` e usa os padrões embutidos (as credenciais em `.env` ainda são carregadas). Equivalente a `--ignore-user-config`. |
| `HERMES_IGNORE_RULES` | Pula a injeção automática de `AGENTS.md`, `SOUL.md`, `.cursorrules`, memória, e skills pré-carregadas. Equivalente a `--ignore-rules`. |
| `HERMES_SAFE_MODE` | Modo de solução de problemas: desativa TODAS as customizações — pula a descoberta de plugins, o carregamento de servidor MCP, e o registro de shell-hook. Definido automaticamente por `--safe-mode` (que também define as duas flags acima). |
| `HERMES_TOOL_PROGRESS` | Variável de compatibilidade obsoleta para exibição de progresso de ferramenta. Prefira `display.tool_progress` no `config.yaml`. |
| `HERMES_TOOL_PROGRESS_MODE` | Variável de compatibilidade obsoleta para modo de progresso de ferramenta. Prefira `display.tool_progress` no `config.yaml`. |
| `HERMES_HUMAN_DELAY_MODE` | Ritmo de resposta: `off`/`natural`/`custom` |
| `HERMES_HUMAN_DELAY_MIN_MS` | Mínimo do intervalo de atraso customizado (ms) |
| `HERMES_HUMAN_DELAY_MAX_MS` | Máximo do intervalo de atraso customizado (ms) |
| `HERMES_QUIET` | Suprime saída não essencial (`true`/`false`) |
| `CODEX_HOME` | Quando o [runtime do Codex app-server](../user-guide/features/codex-app-server-runtime) está ativado, sobrescreve o diretório de onde o Codex CLI lê sua config + auth (padrão: `~/.codex`). A migração do Hermes escreve o bloco gerenciado em `<CODEX_HOME>/config.toml`. |
| `HERMES_KANBAN_TASK` | Definida pelo dispatcher kanban ao criar um worker (UUID da tarefa). Workers e o subprocesso MCP `hermes-tools` gerado a partir deles herdam isso para que as ferramentas kanban filtrem corretamente. Não defina manualmente. |
| `HERMES_API_TIMEOUT` | Timeout de chamada de API de LLM em segundos (padrão: `1800`) |
| `HERMES_API_CALL_STALE_TIMEOUT` | Timeout de chamada obsoleta não-streaming em segundos (padrão: `90`). Desativado automaticamente para provedores locais quando indefinido, e pode escalar para cima em contextos muito grandes. Também configurável via `providers.<id>.stale_timeout_seconds` ou `providers.<id>.models.<model>.stale_timeout_seconds` no `config.yaml`. |
| `HERMES_STREAM_READ_TIMEOUT` | Timeout de leitura de socket de streaming em segundos (padrão: `120`). Aumentado automaticamente para `HERMES_API_TIMEOUT` para provedores locais. Aumente se LLMs locais derem timeout durante geração de código longa. |
| `HERMES_STREAM_STALE_TIMEOUT` | Timeout de detecção de stream obsoleto em segundos (padrão: `180`). Desativado automaticamente para provedores locais. Dispara o encerramento da conexão se nenhum chunk chegar dentro desta janela. |
| `HERMES_LOCAL_STREAM_STALE_TIMEOUT` | Teto de stream obsoleto para provedores locais (Ollama, oMLX, llama-cpp) em segundos (padrão: `900`). Quando o timeout obsoleto base está no seu padrão e um endpoint local é detectado, este teto finito substitui a antiga desativação infinita, de forma que um servidor local congelado eventualmente dispare o detector em vez de ficar pendurado para sempre. Também configurável via `agent.local_stream_stale_timeout` no `config.yaml`. |
| `HERMES_STREAM_RETRIES` | Número de tentativas de reconexão durante o stream em erros de rede transitórios (padrão: `3`). |
| `HERMES_STREAM_STALE_GIVEUP` | Circuit breaker entre turnos: após este número de encerramentos consecutivos por obsolescência (streaming ou não-streaming) sem resposta completa, aborta cada chamada imediatamente com um erro acionável em vez de esperar novamente o timeout obsoleto (padrão: `5`, `0` desativa). Reinicia em qualquer resposta completa, troca de `/model`, ativação de fallback, ou restauração do modelo primário no início do turno. |
| `HERMES_AGENT_TIMEOUT` | Timeout de inatividade do gateway para um agente em execução em segundos (padrão: `1800`, 30 minutos). Reinicia em cada chamada de ferramenta e token transmitido. Defina como `0` para desativar. |
| `HERMES_GATEWAY_MAX_STARTS` | Circuit breaker de tempestade de respawn: máximo de (re)inicializações do gateway permitidas dentro da janela antes de dormir um backoff exponencial para quebrar a tempestade (padrão: `5`, `0` desativa). Também configurável via `gateway.respawn_storm.max_starts` no `config.yaml`. |
| `HERMES_GATEWAY_START_WINDOW_S` | Janela do breaker de tempestade de respawn em segundos (padrão: `120`). Também configurável via `gateway.respawn_storm.window_seconds` no `config.yaml`. |
| `HERMES_AGENT_TIMEOUT_WARNING` | Gateway: envia uma mensagem de aviso após este número de segundos de inatividade (padrão: 75% de `HERMES_AGENT_TIMEOUT`). |
| `HERMES_AGENT_NOTIFY_INTERVAL` | Gateway: intervalo em segundos entre notificações de progresso em turnos de agente de longa duração. |
| `HERMES_CHECKPOINT_TIMEOUT` | Timeout para criação de checkpoint de sistema de arquivos em segundos (padrão: `30`). |
| `HERMES_EXEC_ASK` | Ativa prompts de aprovação de execução no modo gateway (`true`/`false`) |
| `HERMES_ENABLE_PROJECT_PLUGINS` | Ativa a descoberta automática de plugins locais do repositório em `./.hermes/plugins/` tanto para o carregador do agente quanto para o servidor web do dashboard. Aceita o conjunto padrão de valores verdadeiros: `1` / `true` / `yes` / `on` (sem diferenciar maiúsculas/minúsculas). Tudo o mais — incluindo `0`, `false`, `no`, `off`, e a string vazia — é tratado como **desativado** (padrão). Observação: a partir de GHSA-5qr3-c538-wm9j (#29156) o servidor web do dashboard se recusa a auto-importar o arquivo Python `api` de um plugin de projeto mesmo quando esta variável está ativada — plugins de projeto podem estender a UI via JS/CSS estático, mas suas rotas de backend só são carregadas quando movidas para `~/.hermes/plugins/`. |
| `HERMES_PLUGINS_DEBUG` | `1`/`true` para mostrar logs verbosos de descoberta de plugin no stderr — diretórios escaneados, manifestos analisados, motivos de pulo, e tracebacks completos em falhas de parse ou `register()`. Voltado para autores de plugins. |
| `HERMES_BACKGROUND_NOTIFICATIONS` | Modo de notificação de processo em segundo plano no gateway: `all` (padrão), `result`, `error`, `off` |
| `HERMES_EPHEMERAL_SYSTEM_PROMPT` | System prompt efêmero injetado no momento da chamada de API (nunca persistido em sessões) |
| `HERMES_PREFILL_MESSAGES_FILE` | Caminho para um arquivo JSON de mensagens de prefill efêmeras injetadas no momento da chamada de API. |
| `HERMES_ALLOW_PRIVATE_URLS` | `true`/`false` — permite que ferramentas busquem URLs localhost/de rede privada. Desativado por padrão no modo gateway. |
| `HERMES_REDACT_SECRETS` | `true`/`false` — controla a redação de segredos na saída de ferramenta, logs, e respostas de chat (padrão: `true`). |
| `HERMES_WRITE_SAFE_ROOT` | Prefixo de diretório opcional que **bloqueia rigidamente** escritas de `write_file`/`patch` fora das raízes listadas (sem prompt de aprovação). Suporta vários diretórios separados por `os.pathsep` (`:` no Unix, `;` no Windows). Veja [HERMES_WRITE_SAFE_ROOT](#hermes_write_safe_root) abaixo. |
| `HERMES_DISABLE_LAZY_INSTALLS` | Variável de bridge interna definida automaticamente na imagem Docker oficial para evitar instalações de dependência em runtime na árvore imutável `/opt/hermes`. O equivalente voltado ao usuário é `security.allow_lazy_installs: false` no `config.yaml`; não defina isso no `.env`. |
| `HERMES_DISABLE_FILE_STATE_GUARD` | Defina como `1` para desativar a guarda de "arquivo mudou desde que você o leu" em `patch`/`write_file`. |
| `HERMES_BUNDLED_SKILLS` | Sobrescrita separada por vírgula para a lista de skills incluídas carregadas na inicialização. |
| `HERMES_OPTIONAL_SKILLS` | Lista separada por vírgula de nomes de skills opcionais para auto-instalar na primeira execução. |
| `HERMES_DEBUG_INTERRUPT` | Defina como `1` para logar rastreamento detalhado de interrupção/cancelamento em `agent.log`. |
| `HERMES_DUMP_REQUESTS` | Despeja os payloads de requisição de API em arquivos de log (`true`/`false`) |
| `HERMES_DUMP_REQUEST_STDOUT` | Despeja os payloads de requisição de API no stdout em vez de arquivos de log. |
| `HERMES_OAUTH_TRACE` | Defina como `1` para logar tentativas de troca e renovação de token OAuth. Inclui informações de timing redigidas. |
| `HERMES_OAUTH_FILE` | Sobrescreve o caminho usado para armazenamento de credencial OAuth (padrão: `~/.hermes/auth.json`). |
| `HERMES_AGENT_HELP_GUIDANCE` | Acrescenta texto de orientação adicional ao system prompt para implantações customizadas. |
| `HERMES_AGENT_LOGO` | Sobrescreve o logo ASCII do banner na inicialização da CLI. |
| `DELEGATION_MAX_CONCURRENT_CHILDREN` | Máximo de subagentes paralelos por batch de `delegate_task` (padrão: `3`, piso de 1, sem teto). Também configurável via `delegation.max_concurrent_children` no `config.yaml` — o valor de config tem prioridade. |

### HERMES_WRITE_SAFE_ROOT {#hermes_write_safe_root}

Quando esta variável está definida, `write_file` e `patch` só podem visar caminhos dentro do(s) prefixo(s) de diretório listado(s). Qualquer caminho fora dessas raízes é **rejeitado imediatamente** — a escrita não passa pelo sistema de aprovação de comando perigoso e não há prompt para sobrescrever isso.

A imagem Docker oficial define `HERMES_WRITE_SAFE_ROOT=/opt/data` junto com `HERMES_HOME=/opt/data` para que o agente não possa escapar do volume de dados montado.

**Não adicione isso ao seu `~/.hermes/.env` a menos que você pretenda colocar as escritas em sandbox.** Um erro comum é apontá-la para um diretório de projeto enquanto espera que o agente edite `~/.hermes/cron/jobs.json`, `~/.hermes/skills/`, ou scripts sob um perfil — esses caminhos estão fora do sandbox e toda `write_file`/`patch` para eles falha com um erro `outside HERMES_WRITE_SAFE_ROOT`.

Para permitir tanto um workspace quanto o estado do Hermes, liste ambos os prefixos (a ordem não importa):

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

Remova a definição da variável ou remova-a do `.env` para restaurar escritas normais (ainda sujeitas à denylist de caminho de credencial — veja [Segurança de escrita de arquivo](../user-guide/security.md#file-write-safety)).

## Interface {#interface}

| Variável | Descrição |
|----------|-------------|
| `HERMES_TUI` | Inicia a [TUI](../user-guide/tui.md) em vez da CLI clássica quando definida como `1`. Equivalente a passar `--tui`. |
| `HERMES_TUI_DIR` | Caminho para um diretório `ui-tui/` pré-construído (deve conter `dist/entry.js` e `node_modules` populado). Usado por distros e Nix para pular o `npm install` de primeiro lançamento. |
| `HERMES_TUI_RESUME` | Retoma uma sessão TUI específica por ID na inicialização. Quando definida, `hermes --tui` pula a criação de uma sessão nova e retoma a sessão nomeada em vez disso — útil para reconectar após uma desconexão ou queda de terminal. |
| `HERMES_TUI_THEME` | Força o tema de cor da TUI: `light`, `dark`, ou um hex de fundo de 6 caracteres puro (ex.: `ffffff` ou `1a1a2e`). Quando indefinida, o Hermes detecta automaticamente usando `COLORFGBG` e consultas de fundo do terminal; esta variável sobrescreve a detecção em terminais (Ghostty, Warp, iTerm2, etc.) que não definem `COLORFGBG`. |
| `HERMES_INFERENCE_MODEL` | Força o modelo para `hermes -z` / `hermes chat` sem alterar o `config.yaml`. Combina com a flag `--provider`. Útil para chamadores automatizados (sweeper, CI, batch runners) que precisam sobrescrever o modelo padrão por execução. |

## Configurações de Sessão {#session-settings}

| Variável | Descrição |
|----------|-------------|
| `SESSION_IDLE_MINUTES` | Reinicia sessões após N minutos de inatividade (padrão: 1440) |
| `SESSION_RESET_HOUR` | Hora de reinício diário em formato 24h (padrão: 4 = 4h da manhã) |
| `HERMES_SESSION_ID` | **Exportada automaticamente em todo subprocesso de ferramenta** que o Hermes gera (`terminal`, `execute_code`, shell persistente, backends Docker/Singularity, execuções de subagente delegado). Definida pelo agente como o ID de sessão atual; scripts de usuário chamados a partir de ferramentas podem lê-la para correlacionar sua saída, telemetria, ou efeitos colaterais com a sessão do Hermes de origem. **Você não deve definir isso manualmente** — sobrescrevê-la a partir de um shell pai só tem efeito fora de uma execução de agente, e é sobrescrita no momento em que o agente inicia uma sessão. |

## Compressão de Contexto (apenas config.yaml) {#context-compression-configyaml-only}

A compressão de contexto é configurada exclusivamente através do `config.yaml` — não há variáveis de ambiente para isso. As configurações de limiar vivem no bloco `compression:`, enquanto o modelo/provedor de sumarização vive sob `auxiliary.compression:`.

```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20         # fraction of threshold to preserve as recent tail
  protect_last_n: 20         # minimum recent messages to keep uncompressed
```

:::info Migração legada
Configs antigas com `compression.summary_model`, `compression.summary_provider`, e `compression.summary_base_url` são migradas automaticamente para `auxiliary.compression.*` no primeiro carregamento.
:::

## Sobrescritas de Tarefa Auxiliar {#auxiliary-task-overrides}

| Variável | Descrição |
|----------|-------------|
| `AUXILIARY_VISION_PROVIDER` | Sobrescreve o provedor para tarefas de visão |
| `AUXILIARY_VISION_MODEL` | Sobrescreve o modelo para tarefas de visão |
| `AUXILIARY_VISION_BASE_URL` | Endpoint direto compatível com OpenAI para tarefas de visão |
| `AUXILIARY_VISION_API_KEY` | Chave de API pareada com `AUXILIARY_VISION_BASE_URL` |
| `AUXILIARY_WEB_EXTRACT_PROVIDER` | Sobrescreve o provedor para extração/sumarização web |
| `AUXILIARY_WEB_EXTRACT_MODEL` | Sobrescreve o modelo para extração/sumarização web |
| `AUXILIARY_WEB_EXTRACT_BASE_URL` | Endpoint direto compatível com OpenAI para extração/sumarização web |
| `AUXILIARY_WEB_EXTRACT_API_KEY` | Chave de API pareada com `AUXILIARY_WEB_EXTRACT_BASE_URL` |

Para endpoints diretos específicos de tarefa, o Hermes usa a chave de API configurada da tarefa ou `OPENAI_API_KEY`. Ele não reutiliza `OPENROUTER_API_KEY` para esses endpoints customizados.

## Provedores de Fallback (apenas config.yaml) {#fallback-providers-configyaml-only}

A cadeia de fallback do modelo primário é configurada exclusivamente através do `config.yaml` — não há variáveis de ambiente para isso. Adicione uma lista `fallback_providers` de nível superior com chaves `provider` e `model` para ativar failover automático quando seu modelo principal encontrar erros. Tarefas auxiliares cujo provedor é `auto` também consultam essa cadeia antes da cadeia de descoberta auxiliar embutida do Hermes.

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

O formato antigo de provedor único de nível superior `fallback_model` ainda é lido por compatibilidade retroativa, mas configurações novas devem usar `fallback_providers`. Para política auxiliar específica de tarefa, use `auxiliary.<task>.fallback_chain` no `config.yaml`; não há equivalente em variável de ambiente.

Veja [Provedores de Fallback](/user-guide/features/fallback-providers) para detalhes completos.

## Roteamento de Provedor (apenas config.yaml) {#provider-routing-configyaml-only}

Isso vai em `~/.hermes/config.yaml` sob a seção `provider_routing`:

| Chave | Descrição |
|-----|-------------|
| `sort` | Ordena provedores: `"price"` (padrão), `"throughput"`, ou `"latency"` |
| `only` | Lista de slugs de provedor a permitir (ex.: `["anthropic", "google"]`) |
| `ignore` | Lista de slugs de provedor a pular |
| `order` | Lista de slugs de provedor a tentar em ordem |
| `require_parameters` | Usa apenas provedores que suportam todos os parâmetros da requisição (`true`/`false`) |
| `data_collection` | `"allow"` (padrão) ou `"deny"` para excluir provedores que armazenam dados |

:::tip
Use `hermes config set` para definir variáveis de ambiente — ele automaticamente as salva no arquivo correto (`.env` para segredos, `config.yaml` para todo o resto).
:::
