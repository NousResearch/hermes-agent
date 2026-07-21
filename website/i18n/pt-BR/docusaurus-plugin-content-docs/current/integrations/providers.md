---
title: "Provedores de IA"
sidebar_label: "Provedores de IA"
sidebar_position: 1
---

# Provedores de IA {#ai-providers}

Esta página aborda a configuração de provedores de inferência para o Hermes Agent — de APIs em nuvem como OpenRouter e Anthropic, a endpoints auto-hospedados como Ollama e vLLM, até configurações avançadas de roteamento e fallback. Você precisa de pelo menos um provedor configurado para usar o Hermes.

## Provedores de Inferência {#inference-providers}

Você precisa de pelo menos uma forma de se conectar a um LLM. Use `hermes model` para alternar entre provedores e modelos interativamente, ou configure diretamente:

| Provedor | Configuração |
|----------|-------|
| **Nous Portal** | `hermes model` (OAuth, baseado em assinatura) |
| **OpenAI Codex** | `hermes model` (OAuth do ChatGPT, usa modelos Codex) |
| **GitHub Copilot** | `hermes model` (fluxo OAuth de código de dispositivo, `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, ou `gh auth token`) |
| **GitHub Copilot ACP** | `hermes model` (inicia `copilot --acp --stdio` localmente) |
| **Anthropic** | `hermes model` (Claude Max + créditos de uso extra via OAuth; também suporta chave de API da Anthropic ou setup-token manual — veja a nota abaixo) |
| **OpenRouter** | `OPENROUTER_API_KEY` em `~/.hermes/.env` |
| **Fireworks AI** | `FIREWORKS_API_KEY` em `~/.hermes/.env` (provedor: `fireworks`; aliases: `fireworks-ai`, `fw`) |
| **NovitaAI** | `NOVITA_API_KEY` em `~/.hermes/.env` (provedor: `novita`, mais de 200 modelos, Model API, Agent Sandbox, GPU Cloud) |
| **z.ai / GLM** | `GLM_API_KEY` em `~/.hermes/.env` (provedor: `zai`) |
| **Kimi / Moonshot** | `KIMI_API_KEY` em `~/.hermes/.env` (provedor: `kimi-coding`) |
| **Kimi / Moonshot (China)** | `KIMI_CN_API_KEY` em `~/.hermes/.env` (provedor: `kimi-coding-cn`; aliases: `kimi-cn`, `moonshot-cn`) |
| **Arcee AI** | `ARCEEAI_API_KEY` em `~/.hermes/.env` (provedor: `arcee`; aliases: `arcee-ai`, `arceeai`) |
| **GMI Cloud** | `GMI_API_KEY` em `~/.hermes/.env` (provedor: `gmi`; aliases: `gmi-cloud`, `gmicloud`) |
| **MiniMax** | `MINIMAX_API_KEY` em `~/.hermes/.env` (provedor: `minimax`) |
| **MiniMax China** | `MINIMAX_CN_API_KEY` em `~/.hermes/.env` (provedor: `minimax-cn`) |
| **xAI (Grok) — API Responses** | `XAI_API_KEY` em `~/.hermes/.env` (provedor: `xai`) |
| **xAI Grok OAuth (SuperGrok)** | `hermes model` → "xAI Grok OAuth (SuperGrok / Premium+)" — login pelo navegador, sem chave de API. Veja o [guia](../guides/xai-grok-oauth.md) |
| **Qwen Cloud (Alibaba DashScope)** | `DASHSCOPE_API_KEY` em `~/.hermes/.env` (provedor: `alibaba`) |
| **Alibaba Cloud (Coding Plan)** | `DASHSCOPE_API_KEY` (provedor: `alibaba-coding-plan`, alias: `alibaba_coding`) — SKU de faturamento separado, endpoint diferente |
| **Kilo Code** | `KILOCODE_API_KEY` em `~/.hermes/.env` (provedor: `kilocode`) |
| **Xiaomi MiMo** | `XIAOMI_API_KEY` em `~/.hermes/.env` (provedor: `xiaomi`, aliases: `mimo`, `xiaomi-mimo`) |
| **Tencent TokenHub** | `TOKENHUB_API_KEY` em `~/.hermes/.env` (provedor: `tencent-tokenhub`, aliases: `tencent`, `tokenhub`, `tencentmaas`) |
| **OpenCode Zen** | `OPENCODE_ZEN_API_KEY` em `~/.hermes/.env` (provedor: `opencode-zen`) |
| **OpenCode Go** | `OPENCODE_GO_API_KEY` em `~/.hermes/.env` (provedor: `opencode-go`) |
| **DeepSeek** | `DEEPSEEK_API_KEY` em `~/.hermes/.env` (provedor: `deepseek`) |
| **Hugging Face** | `HF_TOKEN` em `~/.hermes/.env` (provedor: `huggingface`, aliases: `hf`) |
| **Google / Gemini** | `GOOGLE_API_KEY` (ou `GEMINI_API_KEY`) em `~/.hermes/.env` (provedor: `gemini`) |
| **Google Vertex AI** | `hermes model` → "Google Vertex AI" (provedor: `vertex`; OAuth2 via JSON de conta de serviço ou ADC, faturamento GCP) |
| **OpenAI API (direta)** | `OPENAI_API_KEY` em `~/.hermes/.env` (provedor: `openai-api`, `OPENAI_BASE_URL` opcional) |
| **Azure AI Foundry** | `hermes model` → "Azure AI Foundry" (provedor: `azure-foundry`; usa endpoint e chave do Azure OpenAI / Foundry) |
| **AWS Bedrock** | `hermes model` → "AWS Bedrock" (provedor: `bedrock`; cadeia padrão de credenciais AWS via boto3) |
| **NVIDIA Build** | `NVIDIA_API_KEY` em `~/.hermes/.env` (provedor: `nvidia`; modelos hospedados em NIM em build.nvidia.com) |
| **Ollama Cloud** | `hermes model` → "Ollama Cloud" (provedor: `ollama-cloud`; API do Ollama hospedada na nuvem) |
| **Qwen OAuth** | `hermes model` → "Qwen OAuth" (provedor: `qwen-oauth`; login PKCE pelo navegador) |
| **MiniMax OAuth** | `hermes model` → "MiniMax (OAuth)" (provedor: `minimax-oauth`; login PKCE pelo navegador) |
| **StepFun** | `STEPFUN_API_KEY` em `~/.hermes/.env` (provedor: `stepfun`) |
| **LM Studio** | `hermes model` → "LM Studio" (provedor: `lmstudio`, `LM_API_KEY` opcional) |
| **Endpoint Personalizado** | `hermes model` → escolha "Custom endpoint" (salvo em `config.yaml`) |

Para o caminho oficial com chave de API, veja o [guia dedicado do Google Gemini](/guides/google-gemini).

:::tip Alias da chave de modelo
Na seção de configuração `model:`, você pode usar tanto `default:` quanto `model:` como nome de chave para o ID do seu modelo. Ambos `model: { default: my-model }` e `model: { model: my-model }` funcionam de forma idêntica.
:::


### Nous Portal {#nous-portal}

O [Nous Portal](https://portal.nousresearch.com) é o gateway de assinatura unificado da Nous Research e **a forma recomendada de executar o Hermes Agent**. Um único login OAuth cobre mais de 300 modelos agênticos de fronteira (Claude, GPT, Gemini, DeepSeek, Qwen, Kimi, GLM, MiniMax, Grok, ...) além do [Tool Gateway](/user-guide/features/tool-gateway) (busca na web, geração de imagens, TTS, automação de navegador) — cobrado contra sua assinatura da Nous em vez de contas separadas por provedor.

```bash
hermes setup --portal     # instalação nova — OAuth + provedor + gateway em um único comando
hermes model              # instalação existente — escolha "Nous Portal" na lista
hermes portal info        # inspecione login + roteamento a qualquer momento
```

Ainda não tem uma assinatura? Adquira uma em [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription).

**Para detalhes completos:** veja a [página dedicada de integração do Nous Portal](/integrations/nous-portal) (o que está incluído na assinatura, catálogo de modelos, solução de problemas) e o guia passo a passo [Executando o Hermes Agent com o Nous Portal](/guides/run-hermes-with-nous-portal).

**Identificação de cliente.** Toda requisição ao Portal feita pelo Hermes Agent carrega uma tag `client=hermes-client-v<versão>` (por exemplo, `client=hermes-client-v0.13.0`) alinhada automaticamente com sua versão instalada. Isso é enviado em todos os caminhos do Portal — loop principal de chat, chamadas auxiliares, resumidor de compressão, extração web — e permite que a telemetria do lado do Portal distinga o tráfego do Hermes de outros clientes. Nenhuma configuração é necessária — a tag é atualizada automaticamente quando você executa `hermes update`.

**Autenticação JWT (automática).** O Hermes prefere JWTs escopados de `inference:invoke` para requisições ao Portal, com o caminho legado de chave de sessão opaca como fallback. Nenhuma configuração é necessária — as credenciais são gerenciadas pelo fluxo OAuth e renovadas de forma transparente. Tokens de refresh revogados são colocados em quarentena para evitar loops de repetição.


:::info Nota sobre o Codex
O provedor OpenAI Codex se autentica via código de dispositivo (abra uma URL, digite um código). O Hermes armazena as credenciais resultantes em seu próprio repositório de autenticação em `~/.hermes/auth.json` e pode importar credenciais existentes da CLI do Codex a partir de `~/.codex/auth.json`, quando presentes. Nenhuma instalação da CLI do Codex é necessária.

Se uma renovação de token falhar com um erro terminal (HTTP 4xx, `invalid_grant`, concessão revogada, etc.), o Hermes marca o token de refresh como morto e para de reproduzi-lo, para que você não veja uma enxurrada de falhas de autenticação idênticas. A próxima requisição exibe uma mensagem tipada de reautenticação. Execute `hermes auth add openai-codex` (ou `hermes model` → OpenAI Codex) para iniciar um novo login por código de dispositivo; a quarentena é liberada na próxima troca bem-sucedida.
:::

:::warning
Mesmo ao usar o Nous Portal, o Codex ou um endpoint personalizado, algumas ferramentas (visão, resumo web, MoA) usam um modelo "auxiliar" separado. Por padrão (`auxiliary.*.provider: "auto"`), o Hermes roteia essas tarefas para seu **modelo principal de chat** — o mesmo modelo que você escolheu em `hermes model`. Você pode substituir cada tarefa individualmente para roteá-la para um modelo mais barato/rápido (por exemplo, Gemini Flash no OpenRouter) — veja [Modelos Auxiliares](/user-guide/configuration#auxiliary-models).
:::

:::tip Nous Tool Gateway
Assinantes pagos do Nous Portal também têm acesso ao **[Tool Gateway](/user-guide/features/tool-gateway)** — busca na web, geração de imagens, TTS e automação de navegador roteados através da sua assinatura. Nenhuma chave de API extra é necessária. Em uma instalação nova, `hermes setup --portal` faz seu login, define a Nous como seu provedor e ativa o gateway em um único comando. Usuários existentes podem ativá-lo a partir de `hermes model` ou por ferramenta a partir de `hermes tools`. Inspecione o roteamento a qualquer momento com `hermes portal info`.
:::

### Dois Comandos para Gerenciamento de Modelos {#two-commands-for-model-management}

O Hermes tem **dois** comandos de modelo que servem a propósitos diferentes:

| Comando | Onde executar | O que faz |
|---------|-------------|--------------|
| **`hermes model`** | No seu terminal (fora de qualquer sessão) | Assistente completo de configuração — adicione provedores, execute OAuth, insira chaves de API, configure endpoints |
| **`/model`** | Dentro de uma sessão de chat do Hermes | Troca rápida entre provedores e modelos **já configurados** |

Se você está tentando trocar para um provedor que ainda não configurou (por exemplo, você só tem o OpenRouter configurado e quer usar a Anthropic), você precisa de `hermes model`, não de `/model`. Saia da sua sessão primeiro (`Ctrl+C` ou `/quit`), execute `hermes model`, conclua a configuração do provedor e depois inicie uma nova sessão.


### Anthropic (Nativa) {#anthropic-native}

Use modelos Claude diretamente através da API da Anthropic — sem necessidade de proxy pelo OpenRouter. Suporta três métodos de autenticação:

:::caution Requer créditos de "uso extra" do Claude Max
Ao se autenticar via `hermes model` → Anthropic OAuth (ou via `hermes auth add anthropic --type oauth`), o Hermes roteia como o Claude Code contra sua conta Anthropic. **Isso só funciona se você estiver em um plano Claude Max e tiver comprado créditos de uso extra.** A cota base do plano Max (o uso incluído no Claude Code por padrão) não é consumida pelo Hermes — apenas os créditos extras/excedentes que você adicionou além dela são consumidos. Assinantes do Claude Pro não podem usar esse caminho.

Se você não tem o Max + créditos extras, use uma `ANTHROPIC_API_KEY` em vez disso — as requisições são cobradas por token contra a organização dessa chave (preço padrão de API, independente de qualquer assinatura do Claude).
:::

```bash
# Com uma chave de API (por token)
export ANTHROPIC_API_KEY=***
hermes chat --provider anthropic --model claude-sonnet-4-6

# Preferido: autentique-se através de `hermes model`
# O Hermes usará o repositório de credenciais do Claude Code diretamente, quando disponível
hermes model

# Sobrescrita manual com um setup-token (fallback / legado)
export ANTHROPIC_TOKEN=***  # setup-token ou token OAuth manual
hermes chat --provider anthropic

# Detecção automática de credenciais do Claude Code (se você já usa o Claude Code)
hermes chat --provider anthropic  # lê os arquivos de credenciais do Claude Code automaticamente
```

Quando você escolhe o Anthropic OAuth através de `hermes model`, o Hermes prefere o próprio repositório de credenciais do Claude Code em vez de copiar o token para `~/.hermes/.env`. Isso mantém as credenciais renováveis do Claude renováveis.

Ou defina permanentemente:
```yaml
model:
  provider: "anthropic"
  default: "claude-sonnet-4-6"
```

:::tip Aliases
`--provider claude` e `--provider claude-code` também funcionam como abreviação para `--provider anthropic`.
:::

### GitHub Copilot {#github-copilot}

O Hermes oferece suporte ao GitHub Copilot como um provedor de primeira classe com dois modos:

**`copilot` — API direta do Copilot** (recomendado). Usa sua assinatura do GitHub Copilot para acessar GPT-5.x, Claude, Gemini e outros modelos através da API do Copilot.

```bash
hermes chat --provider copilot --model gpt-5.4
```

**Opções de autenticação** (verificadas nesta ordem):

1. Variável de ambiente `COPILOT_GITHUB_TOKEN`
2. Variável de ambiente `GH_TOKEN`
3. Variável de ambiente `GITHUB_TOKEN`
4. Fallback via CLI `gh auth token`

Se nenhum token for encontrado, `hermes model` oferece um **login OAuth por código de dispositivo** — o mesmo fluxo usado pela CLI do Copilot e pelo opencode.

:::warning Tipos de token
A API do Copilot **não** suporta Personal Access Tokens clássicos (`ghp_*`). Tipos de token suportados:

| Tipo | Prefixo | Como obter |
|------|--------|------------|
| Token OAuth | `gho_` | `hermes model` → GitHub Copilot → Login with GitHub |
| PAT de granularidade fina | `github_pat_` | GitHub Settings → Developer settings → Fine-grained tokens (precisa da permissão **Copilot Requests**) |
| Token de GitHub App | `ghu_` | Via instalação do GitHub App |

Se o seu `gh auth token` retornar um token `ghp_*`, use `hermes model` para se autenticar via OAuth em vez disso.
:::

:::info Comportamento de autenticação do Copilot no Hermes
O Hermes envia um token do GitHub suportado (`gho_*`, `github_pat_*`, ou `ghu_*`) diretamente para `api.githubcopilot.com` e inclui cabeçalhos específicos do Copilot (`Editor-Version`, `Copilot-Integration-Id`, `Openai-Intent`, `x-initiator`).

Em um HTTP 401, o Hermes agora executa uma recuperação de credencial única antes do fallback:

1. Reresolve o token através da cadeia de prioridade normal (`COPILOT_GITHUB_TOKEN` → `GH_TOKEN` → `GITHUB_TOKEN` → `gh auth token`)
2. Reconstrói o cliente OpenAI compartilhado com cabeçalhos atualizados
3. Repete a requisição uma vez

Alguns proxies comunitários mais antigos usam fluxos de troca via `api.github.com/copilot_internal/v2/token`. Esse endpoint pode estar indisponível para alguns tipos de conta (retorna 404). Por isso, o Hermes mantém a autenticação por token direto como o caminho primário e depende da renovação/nova tentativa de credencial em tempo de execução para maior robustez.
:::

**Roteamento de API**: Modelos GPT-5+ (exceto `gpt-5-mini`) usam automaticamente a API Responses. Todos os outros modelos (GPT-4o, Claude, Gemini, etc.) usam Chat Completions. Os modelos são detectados automaticamente a partir do catálogo ativo do Copilot.

**`copilot-acp` — backend de agente ACP do Copilot**. Inicia a CLI local do Copilot como um subprocesso:

```bash
hermes chat --provider copilot-acp --model copilot-acp
# Requer a CLI do GitHub Copilot no PATH e uma sessão existente de `copilot login`
```

**Configuração permanente:**
```yaml
model:
  provider: "copilot"
  default: "gpt-5.4"
```

| Variável de ambiente | Descrição |
|---------------------|-------------|
| `COPILOT_GITHUB_TOKEN` | Token do GitHub para a API do Copilot (primeira prioridade) |
| `HERMES_COPILOT_ACP_COMMAND` | Sobrescreve o caminho do binário da CLI do Copilot (padrão: `copilot`) |
| `HERMES_COPILOT_ACP_ARGS` | Sobrescreve os argumentos do ACP (padrão: `--acp --stdio`) |

### Provedores de Primeira Classe com Chave de API {#first-class-api-key-providers}

Esses provedores têm suporte integrado com IDs de provedor dedicados. Defina a chave de API e use `--provider` para selecionar:

```bash
# Fireworks AI
hermes chat --provider fireworks --model accounts/fireworks/models/kimi-k2p6
# Requer: FIREWORKS_API_KEY em ~/.hermes/.env

# NovitaAI Model API
hermes chat --provider novita --model moonshotai/kimi-k2.5
# Requer: NOVITA_API_KEY em ~/.hermes/.env

# z.ai / ZhipuAI GLM
hermes chat --provider zai --model glm-5
# Requer: GLM_API_KEY em ~/.hermes/.env

# Kimi / Moonshot AI (internacional: api.moonshot.ai)
hermes chat --provider kimi-coding --model kimi-for-coding
# Requer: KIMI_API_KEY em ~/.hermes/.env

# Kimi / Moonshot AI (China: api.moonshot.cn)
hermes chat --provider kimi-coding-cn --model kimi-k2.5
# Requer: KIMI_CN_API_KEY em ~/.hermes/.env

# MiniMax (endpoint global)
hermes chat --provider minimax --model MiniMax-M2.7
# Requer: MINIMAX_API_KEY em ~/.hermes/.env

# MiniMax (endpoint China)
hermes chat --provider minimax-cn --model MiniMax-M2.7
# Requer: MINIMAX_CN_API_KEY em ~/.hermes/.env

# Qwen Cloud / DashScope (modelos Qwen)
hermes chat --provider alibaba --model qwen3.5-plus
# Requer: DASHSCOPE_API_KEY em ~/.hermes/.env

# Xiaomi MiMo
hermes chat --provider xiaomi --model mimo-v2-pro
# Requer: XIAOMI_API_KEY em ~/.hermes/.env

# Tencent TokenHub (Hy3 Preview)
hermes chat --provider tencent-tokenhub --model hy3-preview
# Requer: TOKENHUB_API_KEY em ~/.hermes/.env

# Arcee AI (modelos Trinity)
hermes chat --provider arcee --model trinity-large-thinking
# Requer: ARCEEAI_API_KEY em ~/.hermes/.env

# GMI Cloud
# Use o ID de modelo exato retornado pelo endpoint /v1/models da GMI.
hermes chat --provider gmi --model zai-org/GLM-5.1-FP8
# Requer: GMI_API_KEY em ~/.hermes/.env
```

A Fireworks usa seus IDs de catálogo nativos em formato de barra, como `accounts/fireworks/models/kimi-k2p6`. Execute `hermes model`, escolha **Fireworks AI** e selecione a partir do catálogo ativo ou digite outro ID de modelo da Fireworks. O endpoint padrão é `https://api.fireworks.ai/inference/v1`; configure um endpoint diferente através de `model.base_url` em `config.yaml`, não em `.env`.

Ou defina o provedor permanentemente em `config.yaml`:
```yaml
model:
  provider: "gmi"
  default: "zai-org/GLM-5.1-FP8"
```

As URLs base podem ser sobrescritas com as variáveis de ambiente `NOVITA_BASE_URL`, `GLM_BASE_URL`, `KIMI_BASE_URL`, `MINIMAX_BASE_URL`, `MINIMAX_CN_BASE_URL`, `DASHSCOPE_BASE_URL`, `XIAOMI_BASE_URL`, `GMI_BASE_URL`, ou `TOKENHUB_BASE_URL`.

:::note Detecção Automática de Endpoint da Z.AI
Ao usar o provedor Z.AI / GLM, o Hermes sonda automaticamente múltiplos endpoints (variantes global, China, coding) para encontrar um que aceite sua chave de API. Você não precisa definir `GLM_BASE_URL` manualmente — o endpoint funcional é detectado e armazenado em cache automaticamente.
:::

### xAI (Grok) — API Responses + Cache de Prompt {#xai-grok--responses-api--prompt-caching}

O xAI é conectado através da API Responses (transporte `codex_responses`) para suporte automático a raciocínio nos modelos Grok 4 — sem necessidade do parâmetro `reasoning_effort`, o servidor raciocina por padrão. Defina `XAI_API_KEY` em `~/.hermes/.env` e escolha o xAI em `hermes model`, ou use `grok` como abreviação em `/model grok-4-fast-reasoning`.

Assinantes do SuperGrok e do X Premium+ podem fazer login com OAuth pelo navegador em vez de usar uma chave de API — escolha **xAI Grok OAuth (SuperGrok / Premium+)** em `hermes model`, ou execute `hermes auth add xai-oauth`. O mesmo token bearer do OAuth é reutilizado automaticamente por ferramentas que se conectam diretamente ao xAI (TTS, geração de imagem, geração de vídeo, transcrição). Veja o [guia de OAuth do xAI Grok](../guides/xai-grok-oauth.md) para o fluxo completo — e, se o Hermes for executado em um host remoto, veja também [OAuth via SSH / Hosts Remotos](../guides/oauth-over-ssh.md) para o túnel `ssh -L` necessário.

Ao usar o xAI como provedor (qualquer URL base contendo `x.ai`), o Hermes ativa automaticamente o cache de prompt enviando o cabeçalho `x-grok-conv-id` em cada requisição de API. Isso roteia as requisições para o mesmo servidor dentro de uma sessão de conversa, permitindo que a infraestrutura do xAI reutilize prompts de sistema e históricos de conversa em cache.

Nenhuma configuração é necessária — o cache é ativado automaticamente quando um endpoint do xAI é detectado e um ID de sessão está disponível. Isso reduz a latência e o custo em conversas de múltiplos turnos.

O xAI também oferece um endpoint dedicado de TTS (`/v1/tts`). Selecione **xAI TTS** em `hermes tools` → Voice & TTS, ou veja a página [Voz e TTS](../user-guide/features/tts.md#text-to-speech) para a configuração.

**Migração de modelos xAI retirados (15 de maio de 2026):** o xAI está retirando `grok-4*`, `grok-3`, `grok-code-fast-1` e `grok-imagine-image-pro` em 15/05/2026. Tanto `hermes doctor` quanto a inicialização de `hermes chat` detectam qualquer configuração ainda apontando para uma referência retirada e imprimem a substituição recomendada. Use `hermes migrate xai` para uma reescrita de configuração em uma única etapa — modo dry-run por padrão, adicione `--apply` para gravar as alterações (um backup `config.yaml.bak-pre-migrate-xai-*` com timestamp é criado automaticamente).

```bash
hermes migrate xai          # visualiza as substituições
hermes migrate xai --apply  # reescreve ~/.hermes/config.yaml no local
```

**Backend de Busca na Web do xAI.** Quando o conjunto de ferramentas [Busca na Web](../user-guide/features/web-search.md) está ativado, `web.backend: xai` roteia a busca através do endpoint de busca hospedado do xAI usando as mesmas credenciais `XAI_API_KEY` / OAuth. Nenhuma configuração adicional é necessária se o xAI já estiver configurado como provedor.

### NovitaAI {#novitaai}

A [NovitaAI](https://novita.ai) é a nuvem nativa de IA para desenvolvedores e agentes. Suas três linhas de produtos são: Model API para mais de 200 modelos, Agent Sandbox para construir e executar agentes de IA, e GPU Cloud para computação escalável, todas disponíveis em uma única plataforma.

```bash
# Use qualquer modelo disponível
hermes chat --provider novita --model moonshotai/kimi-k2.5
# Requer: NOVITA_API_KEY em ~/.hermes/.env

# Alias curto
hermes chat --provider novita-ai --model deepseek/deepseek-v3-0324
```

Ou defina-o permanentemente em `config.yaml`:
```yaml
model:
  provider: "novita"
  default: "moonshotai/kimi-k2.5"
  base_url: "https://api.novita.ai/openai/v1"
```

Obtenha sua chave de API em [novita.ai/settings/key-management](https://novita.ai/settings/key-management). A URL base pode ser sobrescrita com `NOVITA_BASE_URL`.

### Ollama Cloud — Modelos Ollama Gerenciados, OAuth + Chave de API {#ollama-cloud--managed-ollama-models-oauth--api-key}

O [Ollama Cloud](https://ollama.com/cloud) hospeda o mesmo catálogo de peso aberto do Ollama local, mas sem a exigência de GPU. Escolha-o em `hermes model` como **Ollama Cloud**, cole sua chave de API de [ollama.com/settings/keys](https://ollama.com/settings/keys), e o Hermes descobre automaticamente os modelos disponíveis.

```bash
hermes model
# → escolha "Ollama Cloud"
# → cole sua OLLAMA_API_KEY
# → selecione entre os modelos descobertos (gpt-oss:120b, glm-4.6:cloud, qwen3-coder:480b-cloud, etc.)
```

Ou diretamente em `config.yaml`:
```yaml
model:
  provider: "ollama-cloud"
  default: "gpt-oss:120b"
```

O catálogo de modelos é obtido dinamicamente de `ollama.com/v1/models` e armazenado em cache por uma hora. A notação `model:tag` (por exemplo, `qwen3-coder:480b-cloud`) é preservada durante a normalização — não use hifens.

:::tip Ollama Cloud vs. Ollama local
Ambos falam a mesma API compatível com OpenAI. O Cloud é um provedor de primeira classe (`--provider ollama-cloud`, `OLLAMA_API_KEY`); o Ollama local é acessado através do fluxo de Endpoint Personalizado (URL base `http://localhost:11434/v1`, sem chave). Use o cloud para modelos grandes que você não pode executar localmente; use o local para privacidade ou trabalho offline.
:::

### AWS Bedrock {#aws-bedrock}

Anthropic Claude, Amazon Nova, DeepSeek v3.2, Meta Llama 4 e outros modelos via AWS Bedrock. Usa a cadeia de credenciais do SDK da AWS (`boto3`) — sem chave de API, apenas autenticação AWS padrão.

```bash
# Mais simples — perfil nomeado em ~/.aws/credentials
hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6

# Ou com variáveis de ambiente explícitas
AWS_PROFILE=myprofile AWS_REGION=us-east-1 hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6
```

Ou permanentemente em `config.yaml`:
```yaml
model:
  provider: "bedrock"
  default: "us.anthropic.claude-sonnet-4-6"
bedrock:
  region: "us-east-1"          # ou defina AWS_REGION
  # profile: "myprofile"       # ou defina AWS_PROFILE
  # discovery: true            # descobre a região automaticamente via IAM
  # guardrail:                 # Bedrock Guardrails opcional
  #   guardrail_identifier: "your-guardrail-id"
  #   guardrail_version: "DRAFT"
```

A autenticação usa a cadeia padrão do boto3: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` explícitos, `AWS_PROFILE` de `~/.aws/credentials`, função IAM no EC2/ECS/Lambda, IMDS ou SSO. Nenhuma variável de ambiente é necessária se você já estiver autenticado com a CLI da AWS.

O Bedrock usa a **API Converse** internamente — as requisições são traduzidas para o formato agnóstico de modelo do Bedrock, então a mesma configuração funciona para modelos Claude, Nova, DeepSeek e Llama. Defina `BEDROCK_BASE_URL` apenas se você estiver chamando um endpoint regional não padrão.

Veja o [guia do AWS Bedrock](/guides/aws-bedrock) para um passo a passo de configuração de IAM, seleção de região e inferência entre regiões.

### Google Vertex AI {#google-vertex-ai}

Modelos Gemini no Google Cloud Vertex AI via o endpoint compatível com OpenAI do Vertex. A autenticação é **OAuth2** — um token de acesso de curta duração (~1 hora) gerado a partir de um JSON de conta de serviço ou de Application Default Credentials (ADC). Não há **chave de API estática**; o Hermes gera e renova automaticamente o token para você, incluindo a regeneração em caso de `401` no meio da sessão.

```bash
# JSON de conta de serviço (recomendado para servidores / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env
# ou Application Default Credentials
gcloud auth application-default login

hermes model   # → "Google Vertex AI" → projeto → região → modelo
```

Ou em `config.yaml` (projeto/região não são secretos e ficam aqui; o caminho da credencial fica em `.env`):
```yaml
model:
  provider: "vertex"
  default: "google/gemini-3-flash-preview"   # O Vertex requer o prefixo google/
vertex:
  project_id: "my-gcp-project"   # em branco → usa o projeto embutido nas credenciais
  region: "global"               # obrigatório para as previews do Gemini 3.x
```

As variáveis de ambiente `VERTEX_PROJECT_ID` / `VERTEX_REGION` sobrescrevem os valores do `config.yaml`. Instale com `pip install 'hermes-agent[vertex]'` (ou deixe o Hermes instalar `google-auth` de forma lazy no primeiro uso). Veja o [guia do Google Vertex AI](/guides/google-vertex) para o passo a passo completo, e o [guia do Google Gemini](/guides/google-gemini) para o caminho de chave de API estática do AI Studio.

### Qwen Portal (OAuth) {#qwen-portal-oauth}

O Qwen Portal da Alibaba com login OAuth pelo navegador. Escolha **Qwen OAuth (Portal)** em `hermes model`, faça login pelo navegador, e o Hermes armazena o token de refresh.

```bash
hermes model
# → escolha "Qwen OAuth (Portal)"
# → o navegador abre; faça login com sua conta Alibaba
# → confirme — as credenciais são salvas em ~/.hermes/auth.json

hermes chat   # usa o endpoint portal.qwen.ai/v1
```

Ou configure `config.yaml`:
```yaml
model:
  provider: "qwen-oauth"
  default: "qwen3-coder-plus"
```

Defina `HERMES_QWEN_BASE_URL` apenas se o endpoint do portal for realocado (padrão: `https://portal.qwen.ai/v1`).

:::tip Qwen OAuth vs. Qwen Cloud (Alibaba DashScope)
`qwen-oauth` usa o Qwen Portal voltado ao consumidor com login OAuth — ideal para usuários individuais. O provedor `alibaba` usa o Qwen Cloud (Alibaba DashScope) com uma `DASHSCOPE_API_KEY` — ideal para cargas de trabalho programáticas/de produção. Ambos roteiam para modelos da família Qwen, mas vivem em endpoints diferentes.
:::

### Alibaba Cloud (Coding Plan) {#alibaba-cloud-coding-plan}

Se você é assinante do **Coding Plan** da Alibaba (uma SKU de preço separada do acesso padrão à API DashScope), o Hermes expõe isso como seu próprio provedor de primeira classe: `alibaba-coding-plan`. Endpoint: `https://coding-intl.dashscope.aliyuncs.com/v1`. É compatível com OpenAI como o provedor `alibaba` regular, mas com uma URL base e superfície de faturamento diferentes.

```yaml
model:
  provider: alibaba_coding     # alias para alibaba-coding-plan
  model: qwen3-coder-plus
```

Ou pela CLI:

```bash
hermes chat --provider alibaba_coding --model qwen3-coder-plus
```

`alibaba_coding` usa a mesma `DASHSCOPE_API_KEY` que sua entrada `alibaba` já usa — nenhuma chave separada é necessária, apenas um destino de roteamento diferente. Antes de este provedor ser registrado, usuários que definiam `provider: alibaba_coding` no `config.yaml` caíam silenciosamente no roteamento do OpenRouter.

### MiniMax (OAuth) {#minimax-oauth}

MiniMax-M2.7 via login OAuth pelo navegador — sem necessidade de chave de API. Escolha **MiniMax (OAuth)** em `hermes model`, faça login pelo navegador, e o Hermes armazena os tokens de acesso e refresh. Usa o endpoint compatível com Anthropic Messages (`/anthropic`) internamente.

```bash
hermes model
# → escolha "MiniMax (OAuth)"
# → o navegador abre; faça login com sua conta MiniMax (região global ou CN)
# → confirme — as credenciais são salvas em ~/.hermes/auth.json

hermes chat   # usa o endpoint api.minimax.io/anthropic
```

Ou configure `config.yaml`:
```yaml
model:
  provider: "minimax-oauth"
  default: "MiniMax-M2.7"
```

Modelos suportados: `MiniMax-M2.7` (principal) e `MiniMax-M2.7-highspeed` (configurado como modelo auxiliar padrão). O caminho OAuth ignora `MINIMAX_API_KEY` / `MINIMAX_BASE_URL`.

:::tip MiniMax OAuth vs. chave de API
`minimax-oauth` usa o portal voltado ao consumidor da MiniMax com login OAuth — sem necessidade de configuração de faturamento. Os provedores `minimax` e `minimax-cn` usam `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY` — para acesso programático. Veja o [guia de OAuth do MiniMax](/guides/minimax-oauth) para um passo a passo completo.
:::

### NVIDIA NIM {#nvidia-nim}

Nemotron e outros modelos de código aberto via [build.nvidia.com](https://build.nvidia.com) (chave de API gratuita) ou um endpoint NIM local.

```bash
# Nuvem (build.nvidia.com)
hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
# Requer: NVIDIA_API_KEY em ~/.hermes/.env

# Endpoint NIM local — sobrescreva a URL base
NVIDIA_BASE_URL=http://localhost:8000/v1 hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
```

Ou defina-o permanentemente em `config.yaml`:
```yaml
model:
  provider: "nvidia"
  default: "nvidia/nemotron-3-super-120b-a12b"
```

:::tip NIM Local
Para implantações on-prem (DGX Spark, GPU local), defina `NVIDIA_BASE_URL=http://localhost:8000/v1`. O NIM expõe a mesma API de chat completions compatível com OpenAI do build.nvidia.com, então alternar entre nuvem e local é uma mudança de variável de ambiente em uma única linha.
:::

O Hermes anexa automaticamente o cabeçalho de origem de faturamento do NIM em cada requisição para `build.nvidia.com` — nenhuma configuração é necessária. Isso roteia o consumo contra a origem correta no painel de faturamento da NVIDIA.

### GMI Cloud {#gmi-cloud}

Modelos abertos e de raciocínio via [GMI Cloud](https://www.gmicloud.ai/) — API compatível com OpenAI, autenticação por chave de API.

```bash
# GMI Cloud
hermes chat --provider gmi --model deepseek-ai/DeepSeek-V3.2
# Requer: GMI_API_KEY em ~/.hermes/.env
```

Ou defina-o permanentemente em `config.yaml`:
```yaml
model:
  provider: "gmi"
  default: "deepseek-ai/DeepSeek-V3.2"
```

A URL base pode ser sobrescrita com `GMI_BASE_URL` (padrão: `https://api.gmi-serving.com/v1`).

### StepFun {#stepfun}

Modelos da série Step via [StepFun](https://platform.stepfun.com) — API compatível com OpenAI, autenticação por chave de API.

```bash
# StepFun
hermes chat --provider stepfun --model step-3.5-flash
# Requer: STEPFUN_API_KEY em ~/.hermes/.env
```

Ou defina-o permanentemente em `config.yaml`:
```yaml
model:
  provider: "stepfun"
  default: "step-3.5-flash"
```

A URL base pode ser sobrescrita com `STEPFUN_BASE_URL` (padrão: `https://api.stepfun.com/v1`).

### Hugging Face Inference Providers {#hugging-face-inference-providers}

O [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) roteia para mais de 20 modelos abertos através de um endpoint unificado compatível com OpenAI (`router.huggingface.co/v1`). As requisições são roteadas automaticamente para o backend mais rápido disponível (Groq, Together, SambaNova, etc.) com failover automático.

```bash
# Use qualquer modelo disponível
hermes chat --provider huggingface --model Qwen/Qwen3.5-397B-A17B
# Requer: HF_TOKEN em ~/.hermes/.env

# Alias curto
hermes chat --provider hf --model deepseek-ai/DeepSeek-V3.2
```

Ou defina-o permanentemente em `config.yaml`:
```yaml
model:
  provider: "huggingface"
  default: "Qwen/Qwen3.5-397B-A17B"
```

Obtenha seu token em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — certifique-se de habilitar a permissão "Make calls to Inference Providers". Nível gratuito incluído (crédito de $0,10/mês, sem margem sobre as tarifas do provedor).

Você pode anexar sufixos de roteamento aos nomes de modelo: `:fastest` (padrão), `:cheapest`, ou `:provider_name` para forçar um backend específico.

A URL base pode ser sobrescrita com `HF_BASE_URL`.

## Provedores de LLM Personalizados e Auto-Hospedados {#custom--self-hosted-llm-providers}

O Hermes Agent funciona com **qualquer endpoint de API compatível com OpenAI**. Se um servidor implementa `/v1/chat/completions`, você pode apontar o Hermes para ele. Isso significa que você pode usar modelos locais, servidores de inferência com GPU, roteadores multi-provedor, ou qualquer API de terceiros.

### Configuração Geral {#general-setup}

Três formas de configurar um endpoint personalizado:

**Configuração interativa (recomendado):**
```bash
hermes model
# Selecione "Custom endpoint (self-hosted / VLLM / etc.)"
# Digite: URL base da API, chave de API, nome do modelo
```

**Configuração manual (`config.yaml`):**
```yaml
# Em ~/.hermes/config.yaml
model:
  default: your-model-name
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: your-key-or-leave-empty-for-local
```

:::warning Variáveis de ambiente legadas
`LLM_MODEL` em `.env` foi **removida** — `config.yaml` é a única fonte de verdade para a configuração de modelo e endpoint. `OPENAI_BASE_URL` ainda é respeitada, mas **apenas** para o provedor `openai-api` (ela sobrescreve o endpoint da OpenAI para acesso direto por chave de API). Para outros provedores e endpoints personalizados, use `hermes model` ou defina `model.base_url` diretamente em `config.yaml`. Se você tiver entradas obsoletas no seu `.env`, elas são limpas automaticamente no próximo `hermes setup` ou migração de configuração.
:::

Ambas as abordagens persistem em `config.yaml`, que é a fonte de verdade para modelo, provedor e URL base.

### Alternando Modelos com `/model` {#switching-models-with-model}

:::warning hermes model vs. /model
**`hermes model`** (executado no seu terminal, fora de qualquer sessão de chat) é o **assistente completo de configuração de provedores**. Use-o para adicionar novos provedores, executar fluxos OAuth, inserir chaves de API e configurar endpoints personalizados.

**`/model`** (digitado dentro de uma sessão de chat ativa do Hermes) só pode **alternar entre provedores e modelos que você já configurou**. Ele não pode adicionar novos provedores, executar OAuth, ou solicitar chaves de API. Se você configurou apenas um provedor (por exemplo, OpenRouter), `/model` mostrará apenas os modelos daquele provedor.

**Para adicionar um novo provedor:** Saia da sua sessão (`Ctrl+C` ou `/quit`), execute `hermes model`, configure o novo provedor e depois inicie uma nova sessão.
:::

Depois de ter pelo menos um endpoint personalizado configurado, você pode alternar modelos no meio da sessão:

```
/model custom:qwen-2.5          # Alterna para um modelo no seu endpoint personalizado
/model custom                    # Detecta automaticamente o modelo do endpoint
/model openrouter:claude-sonnet-4 # Volta para um provedor em nuvem
```

Se você tem **provedores personalizados nomeados** configurados (veja abaixo), use a sintaxe tripla:

```
/model custom:local:qwen-2.5    # Usa o provedor personalizado "local" com o modelo qwen-2.5
/model custom:work:llama3       # Usa o provedor personalizado "work" com llama3
```

Ao alternar provedores, o Hermes persiste a URL base e o provedor na configuração, para que a mudança sobreviva a reinicializações. Ao alternar de um endpoint personalizado para um provedor integrado, a URL base obsoleta é limpa automaticamente.

:::tip
`/model custom` (sem nome de modelo) consulta a API `/models` do seu endpoint e seleciona automaticamente o modelo se exatamente um estiver carregado. Útil para servidores locais executando um único modelo.
:::

Tudo abaixo segue esse mesmo padrão — apenas mude a URL, a chave e o nome do modelo.

---

### Ollama — Modelos Locais, Sem Configuração {#ollama--local-models-zero-config}

O [Ollama](https://ollama.com/) executa modelos de peso aberto localmente com um único comando. Melhor para: experimentação local rápida, trabalho sensível à privacidade, uso offline. Suporta chamadas de ferramentas via a API compatível com OpenAI.

```bash
# Instale e execute um modelo
ollama pull qwen2.5-coder:32b
ollama serve   # Inicia na porta 11434
```

Depois configure o Hermes:

```bash
hermes model
# Selecione "Custom endpoint (self-hosted / VLLM / etc.)"
# Digite a URL: http://localhost:11434/v1
# Pule a chave de API (o Ollama não precisa de uma)
# Digite o nome do modelo (por exemplo, qwen2.5-coder:32b)
```

Ou configure `config.yaml` diretamente:

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 64000   # Veja o aviso abaixo
```

:::caution O Ollama usa comprimentos de contexto muito baixos por padrão
O Ollama **não** usa a janela de contexto completa do seu modelo por padrão. Dependendo da sua VRAM, o padrão é:

| VRAM Disponível | Contexto padrão |
|----------------|----------------|
| Menos de 24 GB | **4.096 tokens** |
| 24–48 GB | 32.768 tokens |
| 48+ GB | 256.000 tokens |

O Hermes Agent requer pelo menos **64.000 tokens** de contexto para uso do agente com ferramentas. Janelas menores são rejeitadas na inicialização porque o prompt de sistema, os esquemas de ferramentas e o estado de conversa em andamento precisam de espaço suficiente para fluxos de trabalho confiáveis de múltiplas etapas.

**Como aumentá-lo** (escolha uma opção):

```bash
# Opção 1: Definir globalmente via variável de ambiente (recomendado)
OLLAMA_CONTEXT_LENGTH=64000 ollama serve

# Opção 2: Para Ollama gerenciado pelo systemd
sudo systemctl edit ollama.service
# Adicione: Environment="OLLAMA_CONTEXT_LENGTH=64000"
# Depois: sudo systemctl daemon-reload && sudo systemctl restart ollama

# Opção 3: Incorporar em um modelo personalizado (persistente por modelo)
echo -e "FROM qwen2.5-coder:32b\nPARAMETER num_ctx 64000" > Modelfile
ollama create qwen2.5-coder-64k -f Modelfile
```

**Você não pode definir o comprimento de contexto através da API compatível com OpenAI** (`/v1/chat/completions`). Ele deve ser configurado no lado do servidor ou via um Modelfile. Essa é a fonte de confusão nº 1 ao integrar o Ollama com ferramentas como o Hermes.
:::

**Verifique se o seu contexto está configurado corretamente:**

```bash
ollama ps
# Observe a coluna CONTEXT — deve mostrar o valor configurado
```

:::tip
Liste os modelos disponíveis com `ollama list`. Baixe qualquer modelo da [biblioteca do Ollama](https://ollama.com/library) com `ollama pull <model>`. O Ollama gerencia o offloading de GPU automaticamente — nenhuma configuração é necessária para a maioria das configurações.
:::

---

### vLLM — Inferência de Alto Desempenho em GPU {#vllm--high-performance-gpu-inference}

O [vLLM](https://docs.vllm.ai/) é o padrão para servir LLMs em produção. Melhor para: máximo throughput em hardware de GPU, servir modelos grandes, batching contínuo.

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8000 \
  --max-model-len 65536 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Depois configure o Hermes:

```bash
hermes model
# Selecione "Custom endpoint (self-hosted / VLLM / etc.)"
# Digite a URL: http://localhost:8000/v1
# Pule a chave de API (ou insira uma se configurou o vLLM com --api-key)
# Digite o nome do modelo: meta-llama/Llama-3.1-70B-Instruct
```

**Comprimento de contexto:** o vLLM lê o `max_position_embeddings` do modelo por padrão. Se isso exceder a memória da sua GPU, ele retorna erro e pede para você definir `--max-model-len` mais baixo. Você também pode usar `--max-model-len auto` para encontrar automaticamente o máximo que cabe. Defina `--gpu-memory-utilization 0.95` (padrão 0,9) para encaixar mais contexto na VRAM.

**Chamadas de ferramentas exigem flags explícitas:**

| Flag | Propósito |
|------|---------|
| `--enable-auto-tool-choice` | Obrigatório para `tool_choice: "auto"` (o padrão no Hermes) |
| `--tool-call-parser <name>` | Parser para o formato de chamada de ferramenta do modelo |

Parsers suportados: `hermes` (Qwen 2.5, Hermes 2/3), `llama3_json` (Llama 3.x), `mistral`, `deepseek_v3`, `deepseek_v31`, `xlam`, `pythonic`. Sem essas flags, as chamadas de ferramentas não funcionarão — o modelo emitirá as chamadas de ferramentas como texto.

**Parsers de raciocínio do Qwen:** o Hermes preserva metadados de raciocínio estruturado, como `reasoning`, `reasoning_content`, e deltas de raciocínio transmitidos, quando servidores compatíveis com OpenAI os retornam. Esses metadados são tratados como dados de rastro de raciocínio/pensamento, não como substituto da resposta visível do assistente. Para modelos de raciocínio Qwen servidos pelo vLLM, certifique-se de que a resposta final visível ao usuário ainda apareça em `content`. Se `--reasoning-parser qwen3` deixar `content` vazio na sua implantação, desative esse parser ou passe uma opção de requisição suportada pelo servidor, como `chat_template_kwargs.enable_thinking: false`, através de `extra_body`.

:::tip
O vLLM suporta tamanhos legíveis por humanos: `--max-model-len 64k` (k minúsculo = 1000, K maiúsculo = 1024).
:::

---

### SGLang — Serviço Rápido com RadixAttention {#sglang--fast-serving-with-radixattention}

O [SGLang](https://github.com/sgl-project/sglang) é uma alternativa ao vLLM com RadixAttention para reutilização de cache KV. Melhor para: conversas de múltiplos turnos (cache de prefixo), decodificação restrita, saída estruturada.

```bash
pip install "sglang[all]"
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 30000 \
  --context-length 65536 \
  --tp 2 \
  --tool-call-parser qwen
```

Depois configure o Hermes:

```bash
hermes model
# Selecione "Custom endpoint (self-hosted / VLLM / etc.)"
# Digite a URL: http://localhost:30000/v1
# Digite o nome do modelo: meta-llama/Llama-3.1-70B-Instruct
```

**Comprimento de contexto:** o SGLang lê a partir da configuração do modelo por padrão. Use `--context-length` para sobrescrever. Se você precisar exceder o máximo declarado do modelo, defina `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.

**Chamadas de ferramentas:** use `--tool-call-parser` com o parser apropriado para a família do seu modelo: `qwen` (Qwen 2.5), `llama3`, `llama4`, `deepseekv3`, `mistral`, `glm`. Sem essa flag, as chamadas de ferramentas voltam como texto simples.

:::caution O SGLang usa 128 tokens de saída máximos por padrão
Se as respostas parecerem truncadas, adicione `max_tokens` às suas requisições ou defina `--default-max-tokens` no servidor. O padrão do SGLang é de apenas 128 tokens por resposta se não especificado na requisição.
:::

---

### llama.cpp / llama-server — Inferência em CPU e Metal {#llamacpp--llama-server--cpu--metal-inference}

O [llama.cpp](https://github.com/ggml-org/llama.cpp) executa modelos quantizados em CPU, Apple Silicon (Metal) e GPUs de consumo. Melhor para: executar modelos sem uma GPU de datacenter, usuários de Mac, implantação em edge.

```bash
# Compile e inicie o llama-server
cmake -B build && cmake --build build --config Release
./build/bin/llama-server \
  --jinja -fa \
  -c 64000 \
  -ngl 99 \
  -m models/qwen2.5-coder-32b-instruct-Q4_K_M.gguf \
  --port 8080 --host 0.0.0.0
```

**Comprimento de contexto (`-c`):** builds recentes usam `0` como padrão, o que lê o contexto de treinamento do modelo a partir dos metadados do GGUF. Para modelos com contexto de treinamento de 128k+, isso pode causar OOM ao tentar alocar o cache KV completo. Defina `-c` explicitamente para pelo menos 64.000 tokens para o Hermes. Se estiver usando slots paralelos (`-np`), o contexto total é dividido entre os slots — com `-c 64000 -np 4`, cada slot recebe apenas 16k, o que fica abaixo do mínimo do Hermes por sessão ativa.

Depois configure o Hermes para apontar para ele:

```bash
hermes model
# Selecione "Custom endpoint (self-hosted / VLLM / etc.)"
# Digite a URL: http://localhost:8080/v1
# Pule a chave de API (servidores locais não precisam de uma)
# Digite o nome do modelo — ou deixe em branco para detecção automática se apenas um modelo estiver carregado
```

Isso salva o endpoint em `config.yaml` para que ele persista entre sessões.

:::caution `--jinja` é obrigatório para chamadas de ferramentas
Sem `--jinja`, o llama-server ignora o parâmetro `tools` completamente. O modelo tentará chamar ferramentas escrevendo JSON no texto da resposta, mas o Hermes não reconhecerá isso como uma chamada de ferramenta — você verá JSON bruto como `{"name": "web_search", ...}` impresso como uma mensagem, em vez de uma busca real.

Suporte nativo a chamadas de ferramentas (melhor desempenho): Llama 3.x, Qwen 2.5 (incluindo Coder), Hermes 2/3, Mistral, DeepSeek, Functionary. Todos os outros modelos usam um handler genérico que funciona, mas pode ser menos eficiente. Veja a [documentação de function calling do llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) para a lista completa.

Você pode verificar se o suporte a ferramentas está ativo checando `http://localhost:8080/props` — o campo `chat_template` deve estar presente.
:::

:::tip
Baixe modelos GGUF do [Hugging Face](https://huggingface.co/models?library=gguf). A quantização Q4_K_M oferece o melhor equilíbrio entre qualidade e uso de memória.
:::

---

### LM Studio — Aplicativo Desktop com Modelos Locais {#lm-studio--desktop-app-with-local-models}

O [LM Studio](https://lmstudio.ai/) é um aplicativo desktop para executar modelos locais com uma interface gráfica. Melhor para: usuários que preferem uma interface visual, testes rápidos de modelo, desenvolvedores em macOS/Windows/Linux.

Inicie o servidor a partir do app LM Studio (aba Developer → Start Server), ou use a CLI:

```bash
lms server start                        # Inicia na porta 1234
lms load qwen2.5-coder --context-length 64000
```

Depois configure o Hermes:

```bash
hermes model
# Selecione "LM Studio"
# Pressione Enter para usar http://localhost:1234/v1
# Escolha um dos modelos descobertos
# Se a autenticação do servidor LM Studio estiver ativada, digite LM_API_KEY quando solicitado
```

Por padrão, o Hermes solicita explicitamente ao LM Studio que carregue o modelo selecionado com 64K de contexto antes da primeira requisição.

Para alterar o comprimento de contexto no LM Studio:

1. Clique no ícone de engrenagem próximo ao seletor de modelo
2. Defina "Context Length" para pelo menos 64000 para uma experiência tranquila
3. Recarregue o modelo para que a mudança tenha efeito
4. Se sua máquina não conseguir suportar 64000, considere usar um modelo menor com comprimentos de contexto maiores.

Alternativamente, use a CLI: `lms load model-name --context-length 64000`

Você pode usar a CLI para estimar se o modelo vai caber: `lms load model-name --context-length 64000 --estimate-only`

Para definir padrões persistentes por modelo: aba My Models → ícone de engrenagem no modelo → defina o tamanho do contexto.
:::

Se você usa o recurso de carregamento Just-In-Time / Auto-Evict do LM Studio e quer que ele gerencie o carregamento e a remoção de modelos a partir de requisições de chat normais, pule a etapa explícita de pré-carregamento do Hermes:

```bash
hermes config set model.lmstudio_load_mode jit
```

Reverta para o comportamento padrão de pré-carregamento explícito com:

```bash
hermes config set model.lmstudio_load_mode explicit
```

**Chamadas de ferramentas:** suportadas desde o LM Studio 0.3.6. Modelos com treinamento nativo de chamada de ferramentas (Qwen 2.5, Llama 3.x, Mistral, Hermes) são detectados automaticamente e exibidos com um selo de ferramenta. Outros modelos usam um fallback genérico que pode ser menos confiável.

---

### Rede WSL2 (Usuários Windows) {#wsl2-networking-windows-users}

Como o Hermes Agent requer um ambiente Unix, usuários do Windows o executam dentro do WSL2. Se o seu servidor de modelo (Ollama, LM Studio, etc.) roda no **host Windows**, você precisa fazer a ponte da rede — o WSL2 usa um adaptador de rede virtual com sua própria sub-rede, então `localhost` dentro do WSL2 se refere à VM Linux, **não** ao host Windows.

:::tip Ambos no WSL2? Sem problema.
Se o seu servidor de modelo também roda dentro do WSL2 (comum para vLLM, SGLang e llama-server), `localhost` funciona como esperado — eles compartilham o mesmo namespace de rede. Pule esta seção.
:::

#### Opção 1: Modo de Rede Espelhada (Recomendado) {#option-1-mirrored-networking-mode-recommended}

Disponível no **Windows 11 22H2+**, o modo espelhado faz `localhost` funcionar bidirecionalmente entre Windows e WSL2 — a correção mais simples.

1. Crie ou edite `%USERPROFILE%\.wslconfig` (por exemplo, `C:\Users\YourName\.wslconfig`):
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. Reinicie o WSL a partir do PowerShell:
   ```powershell
   wsl --shutdown
   ```

3. Reabra seu terminal WSL2. `localhost` agora alcança os serviços do Windows:
   ```bash
   curl http://localhost:11434/v1/models   # Ollama no Windows — funciona
   ```

:::note Firewall do Hyper-V
Em algumas builds do Windows 11, o firewall do Hyper-V bloqueia conexões espelhadas por padrão. Se `localhost` ainda não funcionar depois de habilitar o modo espelhado, execute isto em um **PowerShell de Administrador**:
```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```
:::

#### Opção 2: Usar o IP do Host Windows (Windows 10 / builds mais antigas) {#option-2-use-the-windows-host-ip-windows-10--older-builds}

Se você não puder usar o modo espelhado, encontre o IP do host Windows a partir de dentro do WSL2 e use-o em vez de `localhost`:

```bash
# Obtenha o IP do host Windows (o gateway padrão da rede virtual do WSL2)
ip route show | grep -i default | awk '{ print $3 }'
# Exemplo de saída: 172.29.192.1
```

Use esse IP na sua configuração do Hermes:

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://172.29.192.1:11434/v1   # IP do host Windows, não localhost
```

:::tip Helper dinâmico
O IP do host pode mudar quando o WSL2 é reiniciado. Você pode obtê-lo dinamicamente no seu shell:
```bash
export WSL_HOST=$(ip route show | grep -i default | awk '{ print $3 }')
echo "Windows host at: $WSL_HOST"
curl http://$WSL_HOST:11434/v1/models   # Teste o Ollama
```

Ou use o nome mDNS da sua máquina (requer `libnss-mdns` no WSL2):
```bash
sudo apt install libnss-mdns
curl http://$(hostname).local:11434/v1/models
```
:::

#### Endereço de Bind do Servidor (Obrigatório para o Modo NAT) {#server-bind-address-required-for-nat-mode}

Se você estiver usando a **Opção 2** (modo NAT com o IP do host), o servidor de modelo no Windows precisa aceitar conexões de fora de `127.0.0.1`. Por padrão, a maioria dos servidores escuta apenas em localhost — conexões WSL2 no modo NAT vêm de uma sub-rede virtual diferente e serão recusadas. No modo espelhado, `localhost` é mapeado diretamente, então o bind padrão em `127.0.0.1` funciona bem.

| Servidor | Bind padrão | Como corrigir |
|--------|-------------|------------|
| **Ollama** | `127.0.0.1` | Defina a variável de ambiente `OLLAMA_HOST=0.0.0.0` antes de iniciar o Ollama (System Settings → Environment Variables no Windows, ou edite o serviço do Ollama) |
| **LM Studio** | `127.0.0.1` | Ative **"Serve on Network"** na aba Developer → configurações do servidor |
| **llama-server** | `127.0.0.1` | Adicione `--host 0.0.0.0` ao comando de inicialização |
| **vLLM** | `0.0.0.0` | Já vincula a todas as interfaces por padrão |
| **SGLang** | `127.0.0.1` | Adicione `--host 0.0.0.0` ao comando de inicialização |

**Ollama no Windows (detalhado):** o Ollama roda como um serviço do Windows. Para definir `OLLAMA_HOST`:
1. Abra **System Properties** → **Environment Variables**
2. Adicione uma nova **variável de sistema**: `OLLAMA_HOST` = `0.0.0.0`
3. Reinicie o serviço do Ollama (ou reinicie o computador)

#### Firewall do Windows {#windows-firewall}

O Firewall do Windows trata o WSL2 como uma rede separada (tanto no modo NAT quanto no espelhado). Se as conexões ainda falharem depois dos passos acima, adicione uma regra de firewall para a porta do seu servidor de modelo:

```powershell
# Execute em um PowerShell de Administrador — substitua PORT pela porta do seu servidor
New-NetFirewallRule -DisplayName "Allow WSL2 to Model Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

Portas comuns: Ollama `11434`, vLLM `8000`, SGLang `30000`, llama-server `8080`, LM Studio `1234`.

#### Verificação Rápida {#quick-verification}

A partir de dentro do WSL2, teste se você consegue alcançar seu servidor de modelo:

```bash
# Substitua a URL pelo endereço e porta do seu servidor
curl http://localhost:11434/v1/models          # Modo espelhado
curl http://172.29.192.1:11434/v1/models       # Modo NAT (use o IP real do seu host)
```

Se você receber uma resposta JSON listando seus modelos, está tudo certo. Use essa mesma URL como o `base_url` na sua configuração do Hermes.

---

### Solução de Problemas com Modelos Locais {#troubleshooting-local-models}

Esses problemas afetam **todos** os servidores de inferência locais quando usados com o Hermes.

#### "Connection refused" do WSL2 para um servidor de modelo hospedado no Windows {#connection-refused-from-wsl2-to-a-windows-hosted-model-server}

Se você estiver executando o Hermes dentro do WSL2 e seu servidor de modelo no host Windows, `http://localhost:<port>` não funcionará no modo de rede NAT padrão do WSL2. Veja [Rede WSL2](#wsl2-networking-windows-users) acima para a correção.

#### Chamadas de ferramentas aparecem como texto em vez de serem executadas {#tool-calls-appear-as-text-instead-of-executing}

O modelo emite algo como `{"name": "web_search", "arguments": {...}}` como uma mensagem em vez de realmente chamar a ferramenta.

**Causa:** seu servidor não tem chamadas de ferramentas ativadas, ou o modelo não as suporta através da implementação de chamada de ferramenta do servidor.

| Servidor | Correção |
|--------|-----|
| **llama.cpp** | Adicione `--jinja` ao comando de inicialização |
| **vLLM** | Adicione `--enable-auto-tool-choice --tool-call-parser hermes` |
| **SGLang** | Adicione `--tool-call-parser qwen` (ou o parser apropriado) |
| **Ollama** | As chamadas de ferramentas estão ativadas por padrão — certifique-se de que seu modelo as suporta (verifique com `ollama show model-name`) |
| **LM Studio** | Atualize para 0.3.6+ e use um modelo com suporte nativo a ferramentas |

#### O modelo parece esquecer o contexto ou dar respostas incoerentes {#model-seems-to-forget-context-or-give-incoherent-responses}

**Causa:** a janela de contexto é muito pequena. Quando a conversa excede o limite de contexto, a maioria dos servidores descarta silenciosamente as mensagens mais antigas. Só o prompt de sistema + esquemas de ferramentas do Hermes já podem usar de 4k a 8k tokens.

**Diagnóstico:**

```bash
# Verifique o que o Hermes acha que é o contexto
# Observe a linha de inicialização: "Context limit: X tokens"

# Verifique o contexto real do seu servidor
# Ollama: ollama ps (coluna CONTEXT)
# llama.cpp: curl http://localhost:8080/props | jq '.default_generation_settings.n_ctx'
# vLLM: verifique --max-model-len nos argumentos de inicialização
```

**Correção:** defina o contexto para pelo menos **64.000 tokens** para uso do agente. Veja a seção de cada servidor acima para a flag específica.

#### "Context limit: 2048 tokens" na inicialização {#context-limit-2048-tokens-at-startup}

O Hermes detecta automaticamente o comprimento de contexto a partir do endpoint `/v1/models` do seu servidor. Se o servidor reportar um valor baixo (ou não reportar nenhum), o Hermes usa o limite declarado do modelo, que pode estar errado.

**Correção:** defina-o explicitamente em `config.yaml`:

```yaml
model:
  default: your-model
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 64000
```

#### As respostas são cortadas no meio da frase {#responses-get-cut-off-mid-sentence}

**Possíveis causas:**
1. **Limite baixo de saída (`max_tokens`) no servidor** — o SGLang usa 128 tokens por resposta como padrão. Defina `--default-max-tokens` no servidor ou configure o Hermes com `model.max_tokens` em config.yaml. Observação: `max_tokens` controla apenas o comprimento da resposta — não tem relação com o quanto seu histórico de conversa pode durar (isso é `context_length`).
2. **Esgotamento de contexto** — o modelo preencheu sua janela de contexto. Aumente `model.context_length` ou ative a [compressão de contexto](/user-guide/configuration#context-compression) no Hermes.

---

### Proxy LiteLLM — Gateway Multi-Provedor {#litellm-proxy--multi-provider-gateway}

O [LiteLLM](https://docs.litellm.ai/) é um proxy compatível com OpenAI que unifica mais de 100 provedores de LLM sob uma única API. Melhor para: alternar entre provedores sem mudar a configuração, balanceamento de carga, cadeias de fallback, controles de orçamento.

```bash
# Instale e inicie
pip install "litellm[proxy]"
litellm --model anthropic/claude-sonnet-4 --port 4000

# Ou com um arquivo de configuração para múltiplos modelos:
litellm --config litellm_config.yaml --port 4000
```

Depois configure o Hermes com `hermes model` → Custom endpoint → `http://localhost:4000/v1`.

Exemplo de `litellm_config.yaml` com fallback:
```yaml
model_list:
  - model_name: "best"
    litellm_params:
      model: anthropic/claude-sonnet-4
      api_key: sk-ant-...
  - model_name: "best"
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-...
router_settings:
  routing_strategy: "latency-based-routing"
```

---

### ClawRouter — Roteamento Otimizado por Custo {#clawrouter--cost-optimized-routing}

O [ClawRouter](https://github.com/BlockRunAI/ClawRouter) da BlockRunAI é um proxy de roteamento local que seleciona automaticamente modelos com base na complexidade da consulta. Ele classifica as requisições em 14 dimensões e roteia para o modelo mais barato capaz de lidar com a tarefa. O pagamento é feito via criptomoeda USDC (sem chaves de API).

```bash
# Instale e inicie
npx @blockrun/clawrouter    # Inicia na porta 8402
```

Depois configure o Hermes com `hermes model` → Custom endpoint → `http://localhost:8402/v1` → nome do modelo `blockrun/auto`.

Perfis de roteamento:
| Perfil | Estratégia | Economia |
|---------|----------|---------|
| `blockrun/auto` | Equilíbrio entre qualidade/custo | 74-100% |
| `blockrun/eco` | Mais barato possível | 95-100% |
| `blockrun/premium` | Modelos de melhor qualidade | 0% |
| `blockrun/free` | Apenas modelos gratuitos | 100% |
| `blockrun/agentic` | Otimizado para uso de ferramentas | varia |

:::note
O ClawRouter requer uma carteira com fundos em USDC na Base ou Solana para pagamento. Todas as requisições são roteadas pela API de backend da BlockRun. Execute `npx @blockrun/clawrouter doctor` para verificar o status da carteira.
:::

---

### Outros Provedores Compatíveis {#other-compatible-providers}

Qualquer serviço com uma API compatível com OpenAI funciona. Algumas opções populares:

| Provedor | URL Base | Notas |
|----------|----------|-------|
| [Together AI](https://together.ai) | `https://api.together.xyz/v1` | Modelos abertos hospedados na nuvem |
| [Groq](https://groq.com) | `https://api.groq.com/openai/v1` | Inferência ultrarrápida |
| [DeepSeek](https://deepseek.com) | `https://api.deepseek.com/v1` | Modelos DeepSeek |
| [Fireworks AI](https://fireworks.ai) | `https://api.fireworks.ai/inference/v1` | Hospedagem rápida de modelos abertos |
| [GMI Cloud](https://www.gmicloud.ai/) | `https://api.gmi-serving.com/v1` | Inferência gerenciada compatível com OpenAI |
| [Cerebras](https://cerebras.ai) | `https://api.cerebras.ai/v1` | Inferência em chip wafer-scale |
| [Mistral AI](https://mistral.ai) | `https://api.mistral.ai/v1` | Modelos Mistral |
| [OpenAI](https://openai.com) | `https://api.openai.com/v1` | Acesso direto à OpenAI |
| [Azure OpenAI](https://azure.microsoft.com) | `https://YOUR.openai.azure.com/` | OpenAI empresarial |
| [LocalAI](https://localai.io) | `http://localhost:8080/v1` | Auto-hospedado, multi-modelo |
| [Jan](https://jan.ai) | `http://localhost:1337/v1` | Aplicativo desktop com modelos locais |

Configure qualquer um desses com `hermes model` → Custom endpoint, ou em `config.yaml`:

```yaml
model:
  default: meta-llama/Llama-3.1-70B-Instruct-Turbo
  provider: custom
  base_url: https://api.together.xyz/v1
  api_key: your-together-key
```

---

### Detecção de Comprimento de Contexto {#context-length-detection}

:::note Duas configurações, fáceis de confundir
**`context_length`** é a **janela de contexto total** — o orçamento combinado para tokens de entrada *e* saída (por exemplo, 200.000 para o Claude Opus 4.6). O Hermes usa isso para decidir quando comprimir o histórico e para validar requisições de API.

**`model.max_tokens`** é o **limite de saída** — o número máximo de tokens que o modelo pode gerar em uma *única resposta*. Isso não tem relação com quanto tempo seu histórico de conversa pode ser. O nome padrão da indústria `max_tokens` é uma fonte comum de confusão; a API nativa da Anthropic já o renomeou para `max_output_tokens` para maior clareza.

Defina `context_length` quando a detecção automática errar o tamanho da janela.
Defina `model.max_tokens` apenas quando precisar limitar o quanto respostas individuais podem durar.
:::

O Hermes usa uma cadeia de resolução de múltiplas fontes para detectar a janela de contexto correta para seu modelo e provedor:

1. **Sobrescrita de configuração** — `model.context_length` em config.yaml (prioridade mais alta)
2. **Provedor personalizado por modelo** — `custom_providers[].models.<id>.context_length`
3. **Cache persistente** — valores descobertos anteriormente (sobrevive a reinicializações)
4. **Endpoint `/models`** — consulta a API do seu servidor (endpoints locais/personalizados)
5. **`/v1/models` da Anthropic** — consulta a API da Anthropic para `max_input_tokens` (apenas usuários com chave de API)
6. **API do OpenRouter** — metadados de modelo em tempo real do OpenRouter
7. **Nous Portal** — corresponde por sufixo os IDs de modelo da Nous aos metadados do OpenRouter
8. **[models.dev](https://models.dev)** — registro mantido pela comunidade com comprimentos de contexto específicos por provedor para mais de 3800 modelos em mais de 100 provedores
9. **Padrões de fallback** — padrões amplos de família de modelo (padrão de 128K)

Para a maioria das configurações, isso funciona sem intervenção. O sistema é consciente do provedor — o mesmo modelo pode ter limites de contexto diferentes dependendo de quem o serve (por exemplo, `claude-opus-4.6` tem 1M diretamente na Anthropic, mas 128K no GitHub Copilot).

Para definir o comprimento de contexto explicitamente, adicione `context_length` à configuração do seu modelo:

```yaml
model:
  default: "qwen3.5:9b"
  base_url: "http://localhost:8080/v1"
  context_length: 131072  # tokens
```

Para endpoints personalizados, você também pode definir o comprimento de contexto por modelo:

```yaml
custom_providers:
  - name: "My Local LLM"
    base_url: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
      deepseek-r1:70b:
        context_length: 65536
```

`hermes model` solicitará o comprimento de contexto ao configurar um endpoint personalizado. Deixe em branco para detecção automática.

:::tip Quando definir isso manualmente
- Você está usando o Ollama com um `num_ctx` personalizado menor que o máximo do modelo
- Você quer limitar o contexto abaixo do máximo do modelo (por exemplo, 8k em um modelo de 128k para economizar VRAM)
- Você está executando por trás de um proxy que não expõe `/v1/models`
:::

---

### Provedores Personalizados Nomeados {#named-custom-providers}

Se você trabalha com múltiplos endpoints personalizados (por exemplo, um servidor de desenvolvimento local e um servidor de GPU remoto), você pode defini-los como provedores personalizados nomeados em `config.yaml`:

```yaml
custom_providers:
  - name: local
    base_url: http://localhost:8080/v1
    # api_key omitida — o Hermes usa "no-key-required" para servidores locais sem chave
  - name: work
    base_url: https://gpu-server.internal.corp/v1
    key_env: CORP_API_KEY
    api_mode: chat_completions   # definido explicitamente pelo assistente `hermes model` → Custom Endpoint; a detecção automática ainda ocorre como fallback
  - name: anthropic-proxy
    base_url: https://proxy.example.com/anthropic
    key_env: ANTHROPIC_PROXY_KEY
    api_mode: anthropic_messages  # para proxies compatíveis com Anthropic
```

Alguns endpoints compatíveis com OpenAI precisam de campos específicos do provedor no corpo da requisição. Adicione um mapa `extra_body` ao provedor personalizado correspondente e o Hermes o mesclará em cada requisição de chat-completions para esse endpoint:

```yaml
custom_providers:
  - name: gemma-local
    base_url: http://localhost:8080/v1
    model: google/gemma-4-31b-it
    extra_body:
      enable_thinking: true
      reasoning_effort: high
```

Use o formato que o seu servidor documenta. Por exemplo, implantações do Gemma no vLLM e alguns endpoints NVIDIA NIM esperam `enable_thinking` dentro de `chat_template_kwargs` em vez de um campo `extra_body` de nível superior:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: true
```

Para modelos de raciocínio Qwen servidos pelo vLLM, esse mesmo formato pode ser usado para desativar o pensamento quando um parser de raciocínio separa todo o texto gerado em campos de raciocínio e deixa o `content` do assistente vazio:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

O assistente `hermes model` → Custom Endpoint agora pergunta explicitamente pelo `api_mode` e persiste sua resposta em `config.yaml`. A detecção automática baseada em URL (por exemplo, caminhos `/anthropic` → `anthropic_messages`) ainda ocorre como fallback quando o campo é deixado em branco.

**Visão nativa para modelos de provedor personalizado.** Se seu endpoint personalizado serve um modelo com capacidade de visão que não está no models.dev, defina `model.supports_vision: true` para que o Hermes roteie as imagens anexadas nativamente (como partes `image_url`) em vez de pré-processá-las através de `vision_analyze`. Um único interruptor — não é necessário também definir `agent.image_input_mode: native`.

```yaml
model:
  provider: custom
  base_url: http://localhost:8080/v1
  default: qwen3.6-35b-a3b
  supports_vision: true   # envia imagens nativamente; caso contrário, vision_analyze as pré-descreve
```

A mesma chave é respeitada em modelos de provedor nomeado (`custom_providers[*].models[*].supports_vision`) e aceita booleanos YAML padrão (`true/false/yes/no/on/off/1/0`).

Alterne entre eles no meio da sessão com a sintaxe tripla:

```
/model custom:local:qwen-2.5       # Usa o endpoint "local" com qwen-2.5
/model custom:work:llama3-70b      # Usa o endpoint "work" com llama3-70b
/model custom:anthropic-proxy:claude-sonnet-4  # Usa o proxy
```

Você também pode selecionar provedores personalizados nomeados no menu interativo `hermes model`.

---

### Receitas: Together AI, Groq, Perplexity {#cookbook-together-ai-groq-perplexity}

Os provedores em nuvem listados em [Outros Provedores Compatíveis](#other-compatible-providers) todos falam o dialeto REST da OpenAI, então eles se conectam da mesma forma sob `custom_providers:`. Seguem três receitas testadas. Cada uma se encaixa em `~/.hermes/config.yaml`, e a chave de API correspondente vai em `~/.hermes/.env`.

#### Together AI {#together-ai}

Hospeda modelos de peso aberto (Llama, MiniMax, Gemma, DeepSeek, Qwen) a preços significativamente abaixo das APIs de primeira parte. Boa opção padrão para frotas multi-modelo.

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: together
    base_url: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
    # api_mode: chat_completions  # padrão — não é necessário definir

model:
  default: MiniMaxAI/MiniMax-M2.7   # ou qualquer modelo de together.ai/models
  provider: custom:together
```

```bash
# ~/.hermes/.env
TOGETHER_API_KEY=your-together-key
```

Alterne modelos no meio da sessão:

```
/model custom:together:meta-llama/Llama-3.3-70B-Instruct-Turbo
/model custom:together:google/gemma-4-31b-it
/model custom:together:deepseek-ai/DeepSeek-V3
```

O endpoint `/v1/models` do Together funciona, então `hermes model` pode descobrir automaticamente os modelos disponíveis.

#### Groq {#groq}

Inferência ultrarrápida (~500 tok/s no Llama-3.3-70B). Catálogo pequeno, mas forte para uso interativo sensível à latência.

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: groq
    base_url: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY

model:
  default: llama-3.3-70b-versatile
  provider: custom:groq
```

```bash
# ~/.hermes/.env
GROQ_API_KEY=your-groq-key
```

#### Perplexity {#perplexity}

Útil quando você quer um modelo que faz busca na web em tempo real e citação automaticamente. Rigoroso quanto aos modelos disponíveis — verifique [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) para a lista atual.

```yaml
# ~/.hermes/config.yaml
custom_providers:
  - name: perplexity
    base_url: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: sonar
  provider: custom:perplexity
```

```bash
# ~/.hermes/.env
PERPLEXITY_API_KEY=your-perplexity-key
```

#### Múltiplos provedores em uma única configuração {#multiple-providers-in-one-config}

As três receitas se combinam — use todas juntas e alterne por turno com `/model custom:<name>:<model>`:

```yaml
custom_providers:
  - name: together
    base_url: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
  - name: groq
    base_url: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY
  - name: perplexity
    base_url: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: MiniMaxAI/MiniMax-M2.7
  provider: custom:together      # inicia com o Together; alterne livremente depois
```

:::tip Solução de problemas
- `hermes doctor` não deve imprimir avisos de `Unknown provider` para nenhum desses nomes após as correções do validador da CLI na #15083.
- Se o endpoint `/v1/models` de um provedor estiver inacessível (a Perplexity é o caso comum), `hermes model` persistirá o modelo com um aviso em vez de rejeitar de forma definitiva — veja a #15136.
- Para pular `custom_providers:` completamente e usar `provider: custom` puro com a variável de ambiente `CUSTOM_BASE_URL`, veja a #15103.
:::

---

### Escolhendo a Configuração Certa {#choosing-the-right-setup}

| Caso de Uso | Recomendado |
|----------|-------------|
| **Só quero que funcione** | OpenRouter (padrão) ou Nous Portal |
| **Modelos locais, configuração fácil** | Ollama |
| **Serviço de GPU em produção** | vLLM ou SGLang |
| **Mac / sem GPU** | Ollama ou llama.cpp |
| **Roteamento multi-provedor** | Proxy LiteLLM ou OpenRouter |
| **Otimização de custo** | ClawRouter ou OpenRouter com `sort: "price"` |
| **Privacidade máxima** | Ollama, vLLM ou llama.cpp (totalmente local) |
| **Empresarial / Azure** | Azure OpenAI com endpoint personalizado |
| **Modelos de IA chineses** | z.ai (GLM), Kimi/Moonshot (`kimi-coding` ou `kimi-coding-cn`), MiniMax, Xiaomi MiMo, ou Tencent TokenHub (provedores de primeira classe) |

:::tip
Você pode alternar entre provedores a qualquer momento com `hermes model` — sem necessidade de reiniciar. Seu histórico de conversa, memória e habilidades são mantidos independentemente do provedor usado.
:::

## Chaves de API Opcionais {#optional-api-keys}

| Recurso | Provedor | Variável de Ambiente |
|---------|----------|--------------|
| Web scraping | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY`, `FIRECRAWL_API_URL` |
| Automação de navegador | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| Geração de imagens | [FAL](https://fal.ai/) | `FAL_KEY` |
| Vozes premium de TTS | [ElevenLabs](https://elevenlabs.io/) | `ELEVENLABS_API_KEY` |
| TTS da OpenAI + transcrição de voz | [OpenAI](https://platform.openai.com/api-keys) | `VOICE_TOOLS_OPENAI_KEY` |
| TTS do Mistral + transcrição de voz | [Mistral](https://console.mistral.ai/) | `MISTRAL_API_KEY` |
| Modelagem de usuário entre sessões | [Honcho](https://honcho.dev/) | `HONCHO_API_KEY` |
| Memória semântica de longo prazo | [Supermemory](https://supermemory.ai) | `SUPERMEMORY_API_KEY` |

### Auto-Hospedando o Firecrawl {#self-hosting-firecrawl}

Por padrão, o Hermes usa a [API em nuvem do Firecrawl](https://firecrawl.dev/) para busca e scraping na web. Se você preferir executar o Firecrawl localmente, pode apontar o Hermes para uma instância auto-hospedada. Veja o [SELF_HOST.md](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md) do Firecrawl para instruções completas de configuração.

**O que você ganha:** nenhuma chave de API necessária, sem limites de taxa, sem custos por página, soberania total dos dados.

**O que você perde:** a versão em nuvem usa o "Fire-engine" proprietário do Firecrawl para contornar proteções anti-bot avançadas (Cloudflare, CAPTCHAs, rotação de IP). A versão auto-hospedada usa fetch básico + Playwright, então alguns sites protegidos podem falhar. A busca usa o DuckDuckGo em vez do Google.

**Configuração:**

1. Clone e inicie a stack Docker do Firecrawl (5 containers: API, Playwright, Redis, RabbitMQ, PostgreSQL — requer ~4-8 GB de RAM):
   ```bash
   git clone https://github.com/firecrawl/firecrawl
   cd firecrawl
   # No .env, defina: USE_DB_AUTHENTICATION=false, HOST=0.0.0.0, PORT=3002
   docker compose up -d
   ```

2. Aponte o Hermes para sua instância (sem necessidade de chave de API):
   ```bash
   hermes config set FIRECRAWL_API_URL http://localhost:3002
   ```

Você também pode definir tanto `FIRECRAWL_API_KEY` quanto `FIRECRAWL_API_URL` se sua instância auto-hospedada tiver autenticação ativada.

## Roteamento de Provedores do OpenRouter {#openrouter-provider-routing}

Ao usar o OpenRouter, você pode controlar como as requisições são roteadas entre provedores. Adicione uma seção `provider_routing` ao `~/.hermes/config.yaml`:

```yaml
provider_routing:
  sort: "throughput"          # "price" (padrão), "throughput", ou "latency"
  # only: ["anthropic"]      # Usar apenas esses provedores
  # ignore: ["deepinfra"]    # Pular esses provedores
  # order: ["anthropic", "google"]  # Tentar provedores nessa ordem
  # require_parameters: true  # Usar apenas provedores que suportam todos os parâmetros da requisição
  # data_collection: "deny"   # Excluir provedores que podem armazenar/treinar com dados
```

**Atalhos:** anexe `:nitro` a qualquer nome de modelo para ordenação por throughput (por exemplo, `anthropic/claude-sonnet-4:nitro`), ou `:floor` para ordenação por preço.

## Roteador Pareto Code do OpenRouter {#openrouter-pareto-code-router}

O OpenRouter oferece um roteador experimental de modelos de código em `openrouter/pareto-code`, que roteia automaticamente as requisições para o modelo mais barato que atenda a um patamar de qualidade de código (classificado pela [Artificial Analysis](https://artificialanalysis.ai/)). Escolha este modelo e ajuste o parâmetro `min_coding_score` em `~/.hermes/config.yaml`:

```yaml
model:
  provider: openrouter
  model: openrouter/pareto-code

openrouter:
  min_coding_score: 0.65   # 0.0–1.0; quanto maior, mais forte (e mais caro) o codificador. Padrão 0.65.
```

Observações:

- `min_coding_score` é enviado **apenas** quando `model.model` é `openrouter/pareto-code`. Em qualquer outro modelo, o valor é um no-op.
- Defina como string vazia (ou remova a linha) para deixar o OpenRouter escolher o codificador mais forte disponível — seu comportamento documentado quando o bloco de plugins é omitido.
- A seleção é determinística por pontuação em um determinado dia, mas o modelo realmente escolhido pode mudar conforme a fronteira de Pareto se move (novos modelos, atualizações de benchmark).
- Veja a [documentação do Pareto Router](https://openrouter.ai/docs/guides/routing/routers/pareto-router) do OpenRouter para o comportamento completo do roteador.
- Para usar o roteador Pareto Code em uma **tarefa auxiliar** específica (compressão, visão, etc.) em vez do agente principal, defina `extra_body.plugins` sob essa tarefa — veja [Modelos Auxiliares → Roteamento do OpenRouter & Pareto Code para tarefas auxiliares](/user-guide/configuration#openrouter-routing--pareto-code-for-auxiliary-tasks).

## Provedores de Fallback {#fallback-providers}

Configure uma cadeia de provedores de backup que o Hermes tenta em ordem quando o modelo principal falha (limites de taxa, erros de servidor, falhas de autenticação). O formato canônico é uma lista de nível superior `fallback_providers:`:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
  - provider: anthropic
    model: claude-sonnet-4
    # base_url: http://localhost:8000/v1    # opcional, para endpoints personalizados
    # api_mode: chat_completions           # sobrescrita opcional
```

O dicionário legado de par único `fallback_model:` ainda é aceito por retrocompatibilidade:

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

Quando ativado, o fallback troca o modelo e o provedor no meio da sessão sem perder sua conversa. A cadeia é tentada entrada por entrada; a ativação é única por sessão.

Provedores suportados: `openrouter`, `nous`, `novita`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `gemini`, `qwen-oauth`, `huggingface`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `deepseek`, `nvidia`, `xai`, `xai-oauth`, `ollama-cloud`, `bedrock`, `azure-foundry`, `opencode-zen`, `opencode-go`, `kilocode`, `xiaomi`, `arcee`, `gmi`, `stepfun`, `lmstudio`, `alibaba`, `alibaba-coding-plan`, `tencent-tokenhub`, `custom`.

:::tip
O fallback é configurado exclusivamente através de `config.yaml` — ou interativamente via `hermes fallback`. Para detalhes completos sobre quando ele é acionado, como a cadeia avança e como interage com tarefas auxiliares e delegação, veja [Provedores de Fallback](/user-guide/features/fallback-providers).
:::

---

## Veja Também {#see-also}

- [Configuração](/user-guide/configuration) — Configuração geral (estrutura de diretórios, precedência de configuração, backends de terminal, memória, compressão, e mais)
- [Variáveis de Ambiente](/reference/environment-variables) — Referência completa de todas as variáveis de ambiente
